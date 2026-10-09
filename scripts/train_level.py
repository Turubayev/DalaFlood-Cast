"""Обучение моделей стока для гидропоста: HBV (концептуальная) + LSTM (прогноз на 72 ч).

Цель обучения — суточный расход. Данные гидропостов Казгидромета в открытом доступе
отсутствуют, поэтому по умолчанию используется реанализ GloFAS v4 в ячейке реки у поста;
если у вас есть ряд наблюдений, передайте его: --obs file.csv (колонки date, q[, h]).

    python scripts/train_level.py --gauge esil_petropavl
    python scripts/train_level.py --gauge ural_uralsk --obs data/obs/uralsk.csv

Проверка на независимых годах (по умолчанию 2021–2024, включая паводок 2024 г.).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dalaflood import hbv  # noqa: E402
from dalaflood.config import find_gauge  # noqa: E402
from dalaflood.data import openmeteo  # noqa: E402
from dalaflood.level import LevelLSTM, build_frame, fit_rating_curve, make_windows, peak_errors  # noqa: E402

FORCING = ["temperature_2m_mean", "precipitation_sum", "snow_depth_max", "soil_temperature_7_to_28cm_mean", "temperature_2m_max"]


def forcing(points, start, end) -> pd.DataFrame:
    """Средняя по точкам водосбора метеорология (ERA5-Land + ERA5)."""
    frames = []
    for la, lo in points:
        land = openmeteo._daily(openmeteo._get(openmeteo.ARCHIVE, {
            "latitude": la, "longitude": lo, "start_date": start, "end_date": end, "models": "era5_land", "timezone": "auto",
            "daily": "temperature_2m_mean,snow_depth_max,soil_temperature_7_to_28cm_mean"}))
        era = openmeteo._daily(openmeteo._get(openmeteo.ARCHIVE, {
            "latitude": la, "longitude": lo, "start_date": start, "end_date": end, "models": "era5", "timezone": "auto",
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min"}))
        frames.append(land.join(era))
    return sum(frames) / len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gauge", default="esil_petropavl")
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--test-from", default="2021-01-01")
    ap.add_argument("--obs", help="CSV наблюдений поста: date,q[,h]")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hbv-iter", type=int, default=60)
    ap.add_argument("--calib-from", default="2018-01-01", help="годы для конформной калибровки интервала (до --test-from)")
    a = ap.parse_args()

    basin, g = find_gauge(a.gauge)
    lat, lon, qmean = openmeteo.snap_to_river(g.lat, g.lon)
    print(f"{g.name}: ячейка GloFAS {lat:.3f}, {lon:.3f} (средний расход {qmean:.1f} м³/с)", flush=True)
    rating = g.rating
    if a.obs:
        obs = pd.read_csv(a.obs, parse_dates=["date"]).set_index("date")
        q = obs["q"].astype(float)
        source = "наблюдения поста"
        if "h" in obs:
            rating = fit_rating_curve(obs["q"], obs["h"])
    else:
        q = openmeteo.glofas_discharge(lat, lon, a.start, a.end)
        source = "реанализ GloFAS v4"
    # Метео: точка поста + районы бассейна выше по течению. Талая вода к нижним постам идёт
    # неделями из всего водосбора, поэтому берём и ближайшие, и самые удалённые районы бассейна.
    by_dist = sorted(basin.districts, key=lambda d: (d.lat - g.lat) ** 2 + (d.lon - g.lon) ** 2)
    pts = list(dict.fromkeys(by_dist[:2] + by_dist[-2:]))
    met = forcing([(g.lat, g.lon)] + [(d.lat, d.lon) for d in pts], a.start, a.end)
    q = q.reindex(met.index)

    # --- HBV ---
    doy = met.index.dayofyear.to_numpy()
    pet = hbv.hargreaves_pet(met["temperature_2m_mean"], met["temperature_2m_max"], met["temperature_2m_min"], doy, g.lat)
    frost = np.clip(-met["soil_temperature_7_to_28cm_mean"].to_numpy() / 3.0, 0, 1)
    P, T = met["precipitation_sum"].fillna(0).to_numpy(), met["temperature_2m_mean"].to_numpy()
    cal = met.index < pd.Timestamp(a.test_from)
    params, kge_cal = hbv.calibrate(P[cal], T[cal], pet[cal], frost[cal], q[cal].to_numpy(), maxiter=a.hbv_iter)
    q_hbv = pd.Series(hbv.simulate(params, P, T, pet, frost), index=met.index)
    test = ~cal
    res = {"gauge": g.id, "target": source, "glofas_cell": [lat, lon], "test_from": a.test_from,
           "hbv": {"kge_calibration": round(kge_cal, 3), "nse_test": round(hbv.nse(q[test], q_hbv[test]), 3),
                   "kge_test": round(hbv.kge(q[test], q_hbv[test]), 3)}}
    print("HBV:", res["hbv"], flush=True)

    # --- LSTM на 1–3 суток ---
    df = build_frame(q, met, q_hbv)
    n_tr = int(np.searchsorted(df.index, pd.Timestamp(a.test_from)))
    n_cal = int(np.searchsorted(df.index, pd.Timestamp(a.calib_from)))
    idx_all = np.arange(30 - 1, len(df) - 3)
    tr_idx = idx_all[idx_all < n_cal - 3]
    cal_idx = idx_all[(idx_all >= n_cal) & (idx_all < n_tr - 3)]
    te_idx = idx_all[idx_all >= n_tr]
    Htr, Ftr, Ytr, _ = make_windows(df, tr_idx, seed=1)
    Hca, Fca, Yca, _ = make_windows(df, cal_idx, seed=3)
    Hte, Fte, Yte, Ite = make_windows(df, te_idx, seed=2)
    model = LevelLSTM().fit(Htr, Ftr, Ytr, epochs=a.epochs, verbose=True)
    print("Конформная поправка интервала (log), сутки 1–3:", np.round(model.conformalize(Hca, Fca, Yca), 3), flush=True)
    pred = model.predict(Hte, Fte)  # (B, 3, 3)
    obs_te = np.expm1(Yte)
    res["lstm"], res["persistence"] = {}, {}
    for k in range(3):
        res["lstm"][f"day{k + 1}"] = {"nse": round(hbv.nse(obs_te[:, k], pred[:, k, 1]), 3),
                                      "coverage_80": round(float(np.mean((obs_te[:, k] >= pred[:, k, 0]) & (obs_te[:, k] <= pred[:, k, 2]))), 3)}
        res["persistence"][f"day{k + 1}"] = {"nse": round(hbv.nse(obs_te[:, k], df["q"].to_numpy()[Ite]), 3)}
    sim3 = pd.Series(pred[:, 2, 1], index=df.index[Ite + 3])
    pe = peak_errors(df["q"].iloc[n_tr:], sim3)
    res["peaks_day3"] = pe.round(1).to_dict(orient="records")
    res["rating_curve"] = rating
    print(json.dumps({k: res[k] for k in ("lstm", "persistence")}, ensure_ascii=False, indent=1))
    print(pe.round(1).to_string(index=False))

    md = ROOT / "models"
    md.mkdir(exist_ok=True)
    model.save(md / f"level_{g.id}.pt")
    (md / f"hbv_{g.id}.json").write_text(json.dumps(hbv.params_dict(params), indent=2), encoding="utf-8")
    (md / f"gauge_{g.id}.json").write_text(json.dumps({"glofas_cell": [lat, lon], "rating": rating, "source": source,
                                                       "points": [(g.lat, g.lon)] + [(d.lat, d.lon) for d in pts]},
                                                      ensure_ascii=False, indent=2), encoding="utf-8")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"level_{g.id}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sl = slice(pd.Timestamp("2024-02-15"), pd.Timestamp("2024-06-30"))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["q"].loc[sl], color="k", lw=1.5, label=f"«факт» ({source})")
        ax.plot(q_hbv.loc[sl], color="#98a2b3", lw=1, label="HBV (без данных о расходе)")
        p3 = pd.DataFrame(pred[:, 2], index=df.index[Ite + 3], columns=["q10", "q50", "q90"]).loc[sl]
        ax.fill_between(p3.index, p3["q10"], p3["q90"], color="#0e7490", alpha=0.2, label="LSTM, 80% интервал")
        ax.plot(p3["q50"], color="#0e7490", lw=1.5, label="LSTM, прогноз за 3 суток")
        ax.set_ylabel("Расход, м³/с")
        ax.set_title(f"{g.name}: весна 2024 (независимая проверка)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out / f"level_{g.id}_2024.png", dpi=130)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
