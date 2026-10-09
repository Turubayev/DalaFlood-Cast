"""Оперативный прогноз расхода и уровня на посту на 72 ч + сравнение с ансамблем GloFAS.

    python scripts/forecast_level.py --gauge esil_petropavl

Нужны обученные модели (scripts/train_level.py). Данные: архив ERA5 (разогрев HBV за год),
прогноз погоды Open-Meteo на 16 суток, текущие расходы и ансамблевый прогноз GloFAS.
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
from dalaflood.config import DISCLAIMER, find_gauge  # noqa: E402
from dalaflood.data import openmeteo  # noqa: E402
from dalaflood.level import LEAD_DAYS, LevelLSTM, build_frame, discharge_to_level, make_windows  # noqa: E402


def recent_met(points) -> pd.DataFrame:
    """Год архива ERA5 + недавние дни и прогноз из оперативной модели, в формате climate_daily."""
    today = pd.Timestamp.today().normalize()
    frames = []
    for la, lo in points:
        arch = openmeteo._daily(openmeteo._get(openmeteo.ARCHIVE, {
            "latitude": la, "longitude": lo, "timezone": "auto", "models": "era5_seamless",
            "start_date": (today - pd.Timedelta(days=400)).strftime("%Y-%m-%d"),
            "end_date": (today - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
            "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum"}, cache=False))
        js = openmeteo._get(openmeteo.FORECAST, {
            "latitude": la, "longitude": lo, "timezone": "auto", "past_days": 14, "forecast_days": 7,
            "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum",
            "hourly": "snow_depth,soil_temperature_18cm"}, cache=False)
        fc = openmeteo._daily(js)
        hourly = pd.DataFrame(js["hourly"])
        hourly["time"] = pd.to_datetime(hourly["time"])
        hd = hourly.set_index("time").resample("D").agg({"snow_depth": "max", "soil_temperature_18cm": "mean"})
        fc = fc.join(hd.rename(columns={"snow_depth": "snow_depth_max", "soil_temperature_18cm": "soil_temperature_7_to_28cm_mean"}))
        df = pd.concat([arch, fc[~fc.index.isin(arch.index)]]).sort_index()
        frames.append(df)
    out = sum(f.reindex(frames[0].index) for f in frames) / len(frames)
    return out.interpolate(limit_direction="both")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gauge", default="esil_petropavl")
    a = ap.parse_args()
    _, g = find_gauge(a.gauge)
    md = ROOT / "models"
    meta = json.loads((md / f"gauge_{g.id}.json").read_text(encoding="utf-8"))
    params = hbv.HBVParams(**json.loads((md / f"hbv_{g.id}.json").read_text(encoding="utf-8")))
    model = LevelLSTM.load(md / f"level_{g.id}.pt")
    lat, lon = meta["glofas_cell"]

    met = recent_met(meta["points"])
    gl = openmeteo.glofas_forecast(lat, lon, days=30)
    q_obs = gl["river_discharge_median"]  # на прошедшие дни медиана = реанализ/анализ GloFAS
    today = pd.Timestamp.today().normalize()
    q = q_obs[q_obs.index < today].reindex(met.index)

    pet = hbv.hargreaves_pet(met["temperature_2m_mean"], met["temperature_2m_max"], met["temperature_2m_min"],
                             met.index.dayofyear.to_numpy(), g.lat)
    frost = np.clip(-met["soil_temperature_7_to_28cm_mean"].fillna(0).to_numpy() / 3.0, 0, 1)
    q_hbv = pd.Series(hbv.simulate(params, met["precipitation_sum"].fillna(0).to_numpy(),
                                   met["temperature_2m_mean"].to_numpy(), pet, frost), index=met.index)
    df = build_frame(q.ffill(), met, q_hbv)
    i_last = int(np.searchsorted(df.index, today)) - 1  # последний день с известным расходом
    H, F, _, _ = make_windows(df, np.array([i_last]), noise=False)
    pred = model.predict(H, F)[0]  # (3, 3)
    days = [df.index[i_last] + pd.Timedelta(days=k + 1) for k in range(LEAD_DAYS)]
    rows = []
    for k, d in enumerate(days):
        q10, q50, q90 = pred[k]
        h = discharge_to_level([q10, q50, q90], meta["rating"])
        rows.append({"date": d.strftime("%Y-%m-%d"), "q10": round(float(q10), 1), "q50": round(float(q50), 1),
                     "q90": round(float(q90), 1), "level_q50_m": round(float(h[1]), 2),
                     "glofas_median": round(float(gl["river_discharge_median"].get(d, np.nan)), 1),
                     "glofas_max": round(float(gl["river_discharge_max"].get(d, np.nan)), 1)})
    out = {"gauge": g.name, "issued": str(df.index[i_last].date()), "last_q": round(float(df["q"].iloc[i_last]), 1),
           "forecast": rows, "disclaimer": DISCLAIMER,
           "note": "уровень по кривой расходов-заглушке, пока не подобрана по данным поста (--obs в train_level.py)"}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"forecast_{g.id}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
