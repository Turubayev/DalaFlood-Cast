"""Фактические карты затопления по Sentinel-1 (Microsoft Planetary Computer, sentinel-1-rtc).

    python scripts/flood_extent_s1.py --aoi data/aoi/petropavl.geojson --start 2024-03-25 --end 2024-05-15
    python scripts/flood_extent_s1.py --bbox 69.05 54.80 69.25 54.92 --start 2024-04-01 --end 2024-04-30 \
        --dry-start 2023-08-01 --dry-end 2023-09-15

«Сухая» референсная сцена по умолчанию берётся из августа–сентября предыдущего года
(та же орбита), чтобы отделить паводок от постоянных водоёмов.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dalaflood import s1  # noqa: E402


def aoi_bbox(path: str) -> tuple[float, float, float, float]:
    gj = json.loads(Path(path).read_text(encoding="utf-8"))
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    xs, ys = [], []

    def walk(c):
        if isinstance(c[0], (int, float)):
            xs.append(c[0])
            ys.append(c[1])
        else:
            for cc in c:
                walk(cc)

    for f in feats:
        walk((f.get("geometry") or f)["coordinates"])
    return min(xs), min(ys), max(xs), max(ys)


def compare_with_forecast(prob_path: str, water: np.ndarray, transform, crs) -> dict:
    """Перепроецирует карту вероятности (HAND) на сетку снимка и считает IoU по порогам 0.5.

    Сравнение идёт по общей области; постоянные водоёмы входят в обе карты (в HAND у них HAND≈0).
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(prob_path) as src:
        prob = np.full(water.shape, np.nan, dtype="float32")
        reproject(rasterio.band(src, 1), prob, dst_transform=transform, dst_crs=crs, resampling=Resampling.bilinear,
                  dst_nodata=np.nan)
    valid = np.isfinite(prob)
    pred = (prob > 0.5) & valid
    return {"iou_vs_forecast": round(s1.iou(pred, water & valid), 3),
            "forecast_km2": round(float(pred.sum() * abs(transform.a * transform.e)) / 1e6, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi")
    ap.add_argument("--bbox", type=float, nargs=4)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--dry-start")
    ap.add_argument("--dry-end")
    ap.add_argument("--resolution", type=float, default=20.0)
    ap.add_argument("--max-scenes", type=int, default=6)
    ap.add_argument("--name", default="s1")
    ap.add_argument("--compare-prob", help="GeoTIFF вероятности затопления (scripts/flood_map.py) для расчёта IoU/CSI")
    a = ap.parse_args()
    bbox = tuple(a.bbox) if a.bbox else aoi_bbox(a.aoi)
    y = pd.Timestamp(a.start).year
    dry_start, dry_end = a.dry_start or f"{y - 1}-07-01", a.dry_end or f"{y - 1}-10-15"

    scenes = s1.search_scenes(bbox, a.start, a.end)
    print(f"Сцен Sentinel-1 в период: {len(scenes)}")
    if not scenes:
        return
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    rows = []
    dry_cache = {}
    for it in scenes[: a.max_scenes]:
        orbit = it.properties.get("sat:relative_orbit")
        if orbit not in dry_cache:
            dry_all = s1.search_scenes(bbox, dry_start, dry_end)
            same = [d for d in dry_all if d.properties.get("sat:relative_orbit") == orbit]
            # Та же орбита предпочтительна; иначе — любая (RTC уже скорректирован за рельеф).
            pick = (same or dry_all)
            if pick and not same:
                print(f"  нет сухой сцены орбиты {orbit}, используется орбита {pick[0].properties.get('sat:relative_orbit')}")
            dry_cache[orbit] = s1.read_vv(pick[len(pick) // 2], bbox, a.resolution) if pick else None
        vv, transform, crs = s1.read_vv(it, bbox, a.resolution)
        km2 = (a.resolution / 1000) ** 2
        if dry_cache[orbit] is not None:
            vv_dry = dry_cache[orbit][0]
            hgt, wid = min(vv.shape[0], vv_dry.shape[0]), min(vv.shape[1], vv_dry.shape[1])
            res = s1.flood_extent(vv[:hgt, :wid], vv_dry[:hgt, :wid])
            flooded_km2 = round(float(res["flooded"].sum() * km2), 2)
        else:
            # Над Казахстаном после 2021 г. снимков Sentinel-1 мало: без «сухой» сцены
            # показываем всю воду на снимке (включая постоянные водоёмы).
            print(f"  {it.datetime:%Y-%m-%d}: нет сухой референсной сцены — режим одной сцены (вся вода)")
            hgt, wid = vv.shape
            wet, thr = s1.water_mask(vv)
            res = {"flooded": wet, "water_now": wet, "permanent_water": np.zeros_like(wet), "threshold_db": thr}
            flooded_km2 = None
        row = {"date": f"{it.datetime:%Y-%m-%d}", "orbit": orbit, "threshold_db": round(res["threshold_db"], 1),
               "flooded_km2": flooded_km2, "water_km2": round(float(res["water_now"].sum() * km2), 2)}
        if a.compare_prob:
            row.update(compare_with_forecast(a.compare_prob, res["water_now"], transform, crs))
        rows.append(row)
        print(" ", row, flush=True)
        try:
            import rasterio

            prof = {"driver": "GTiff", "height": hgt, "width": wid, "count": 1, "dtype": "uint8", "crs": crs,
                    "transform": transform, "compress": "deflate"}
            with rasterio.open(out / f"{a.name}_flooded_{row['date']}.tif", "w", **prof) as dst:
                dst.write(res["flooded"].astype("uint8"), 1)
        except Exception as e:  # noqa: BLE001
            print("  GeoTIFF не записан:", e)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
            ax[0].imshow(vv[:hgt, :wid], cmap="gray", vmin=-25, vmax=0)
            ax[0].set_title(f"Sentinel-1 VV, {row['date']}")
            ax[1].imshow(vv[:hgt, :wid], cmap="gray", vmin=-25, vmax=0)
            ax[1].imshow(np.ma.masked_equal(res["flooded"], False), cmap="autumn", alpha=0.7)
            ax[1].imshow(np.ma.masked_equal(res["permanent_water"], False), cmap="winter", alpha=0.6)
            ax[1].set_title(f"Затоплено {row['flooded_km2']} км² (оранж.), постоянная вода (синий)" if flooded_km2 is not None
                            else f"Вода на снимке: {row['water_km2']} км² (без сухой сцены)")
            for x in ax:
                x.axis("off")
            fig.tight_layout()
            fig.savefig(out / f"{a.name}_flooded_{row['date']}.png", dpi=110)
            plt.close(fig)
        except ImportError:
            pass
    pd.DataFrame(rows).to_csv(out / f"{a.name}_flood_extent.csv", index=False)


if __name__ == "__main__":
    main()
