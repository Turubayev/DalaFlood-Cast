"""Карта вероятности затопления по HAND для пилотного участка.

Подъём уровня над меженью задаётся квантилями (м) — например, из прогноза уровня
(scripts/forecast_level.py) или сценарием («как в 2024 г.»).

    # реальный рельеф Copernicus DEM (≈ 13×13 км у Петропавловска), сценарий подъёма 3 / 4.5 / 6 м
    python scripts/flood_map.py --bbox 69.05 54.80 69.25 54.92 --rise 3 4.5 6 --name petropavl
    # свой GeoTIFF (уточнённая съёмка)
    python scripts/flood_map.py --dem my_dem.tif --rise 1 1.5 2 --name pilot
    # без интернета — синтетическая долина
    python scripts/flood_map.py --synthetic --rise 1 2 3
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from dalaflood.config import DISCLAIMER  # noqa: E402
from dalaflood.hand import flood_probability, hand, synthetic_valley  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"))
    ap.add_argument("--dem")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--rise", type=float, nargs=3, required=True, metavar=("Q10", "Q50", "Q90"),
                    help="подъём уровня над меженью, м (квантили 10/50/90%%)")
    ap.add_argument("--stream-km2", type=float, default=50.0, help="порог площади водосбора для водотока, км²")
    ap.add_argument("--name", default="pilot")
    ap.add_argument("--no-osm-rivers", action="store_true", help="не использовать русла OpenStreetMap")
    ap.add_argument("--settlements", action="store_true", help="оценить населённые пункты (OSM Overpass)")
    a = ap.parse_args()

    meta = {}
    if a.synthetic:
        dem, cell = synthetic_valley(), 30.0
        transform = None
    elif a.dem:
        from dalaflood.data.dem import read_geotiff

        dem, meta = read_geotiff(a.dem)
        cell, transform = meta["cellsize_m"], meta["transform"]
    elif a.bbox:
        from dalaflood.data.dem import read_dem

        dem, meta = read_dem(tuple(a.bbox))
        cell, transform = meta["cellsize_m"], meta["transform"]
    else:
        ap.error("укажите --bbox, --dem или --synthetic")

    river_mask = None
    if transform is not None and meta.get("crs") == "EPSG:4326" and not a.no_osm_rivers:
        from dalaflood.hand import main_river, rasterize_lines
        from dalaflood.impact import rivers

        try:
            lines = rivers(meta["bbox"])
            masks = [rasterize_lines([ln], transform, dem.shape) for ln in lines]
            river_mask = main_river(masks, dem)
            print(f"Русла OSM: {len(lines)} линий, главная река — {river_mask.sum()} ячеек", flush=True)
        except Exception as e:  # noqa: BLE001 — без OSM считаем только по накоплению стока
            print("OSM недоступен, русла выделяются по накоплению стока:", e)
    print(f"Рельеф {dem.shape[0]}×{dem.shape[1]} ячеек, ≈{cell:.0f} м; расчёт HAND…", flush=True)
    h, streams, _ = hand(dem, cell, a.stream_km2, river_mask=river_mask)
    prob = flood_probability(h, {0.1: a.rise[0], 0.5: a.rise[1], 0.9: a.rise[2]})
    cell_km2 = (cell / 1000) ** 2
    stats = {
        "area_km2_prob_gt_0.1": round(float((prob > 0.1).sum() * cell_km2), 2),
        "area_km2_prob_gt_0.5": round(float((prob > 0.5).sum() * cell_km2), 2),
        "area_km2_prob_gt_0.9": round(float((prob > 0.9).sum() * cell_km2), 2),
        "rise_m": a.rise,
        "note": "вероятность затопления по рельефу (HAND), не учитывает дамбы и насыпи мельче разрешения DEM",
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    if transform is not None:
        import rasterio

        prof = {"driver": "GTiff", "height": prob.shape[0], "width": prob.shape[1], "count": 1, "dtype": "float32",
                "crs": meta.get("crs", "EPSG:4326"), "transform": transform, "compress": "deflate"}
        with rasterio.open(out / f"flood_prob_{a.name}.tif", "w", **prof) as dst:
            dst.write(prob.astype("float32"), 1)
        if a.settlements and meta.get("crs") == "EPSG:4326":
            from dalaflood.impact import exposure, settlements

            try:
                ex = exposure(prob, transform, settlements(meta["bbox"]))
                stats["settlements_at_risk"] = ex[["name", "place", "flood_prob"]].round(2).to_dict(orient="records")
                print(ex.to_string(index=False))
            except Exception as e:  # noqa: BLE001
                print("Населённые пункты не оценены:", e)
    (out / f"flood_map_{a.name}.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        ext = None
        if transform is not None:
            ext = [transform.c, transform.c + transform.a * dem.shape[1], transform.f + transform.e * dem.shape[0], transform.f]
        ax[0].imshow(dem, cmap="terrain", extent=ext)
        ax[0].contour(streams, levels=[0.5], colors="b", linewidths=0.6, extent=ext, origin="upper")
        ax[0].set_title("Рельеф и водотоки")
        im = ax[1].imshow(np.ma.masked_less(prob, 0.05), cmap="Blues", vmin=0, vmax=1, extent=ext)
        ax[1].set_title(f"Вероятность затопления\nподъём {a.rise[1]} м (10–90%: {a.rise[0]}–{a.rise[2]} м)")
        fig.colorbar(im, ax=ax[1], shrink=0.8)
        fig.text(0.01, 0.01, DISCLAIMER, fontsize=7, wrap=True)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(out / f"flood_map_{a.name}.png", dpi=120)
        print("Карта:", out / f"flood_map_{a.name}.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
