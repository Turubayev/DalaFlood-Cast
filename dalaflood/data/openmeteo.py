"""Открытые данные через Open-Meteo (без ключа, некоммерческое использование):

* ERA5-Land (с 1950 г.): высота снега, влажность и температура почвы по слоям;
* ERA5: осадки и водный эквивалент снегопада;
* GloFAS v4 (Copernicus EMS): реанализ расходов воды с 1984 г. и прогноз до 7 мес.;
* оперативный прогноз погоды (GFS/ICON/ECMWF) на 16 суток.

Ответы кэшируются на диск (data/cache), чтобы повторные расчёты не нагружали API.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FLOOD = "https://flood-api.open-meteo.com/v1/flood"
FORECAST = "https://api.open-meteo.com/v1/forecast"
CACHE = Path(os.environ.get("DALAFLOOD_CACHE", Path(__file__).resolve().parents[2] / "data" / "cache"))

LAND_DAILY = [
    "snow_depth_max",
    "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
    "soil_moisture_28_to_100cm_mean",
    "soil_temperature_0_to_7cm_mean",
    "soil_temperature_7_to_28cm_mean",
    "soil_temperature_28_to_100cm_mean",
    "soil_temperature_100_to_255cm_mean",
    "temperature_2m_mean",
]
ERA5_DAILY = ["precipitation_sum", "snowfall_water_equivalent_sum", "temperature_2m_max"]


def _get(url: str, params: dict, cache: bool = True) -> dict:
    key = hashlib.md5((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    path = CACHE / f"{key}.json"
    if cache and path.exists():
        return json.loads(path.read_text())
    for i in range(5):
        try:
            r = requests.get(url, params=params, timeout=120)
        except (requests.ConnectionError, requests.Timeout):
            if i == 4:
                raise
            time.sleep(5 * (i + 1))
            continue
        if r.status_code == 429 and ("Hourly" in r.text or "Daily" in r.text or "Minutely" not in r.text):
            raise RuntimeError("Open-Meteo: исчерпан лимит бесплатных запросов (час/сутки). Повторите позже — "
                               "уже скачанное сохранено в кэше data/cache.")
        if r.status_code in (429, 502, 503) and i < 4:
            time.sleep(10 * (i + 1))
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"Open-Meteo {r.status_code}: {r.text[:300]}")
        try:
            js = r.json()
        except ValueError:  # иногда шлюз отдаёт пустой/HTML-ответ — повторяем
            if i == 4:
                raise
            time.sleep(5 * (i + 1))
            continue
        if cache:
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(js))
        return js
    raise RuntimeError("Open-Meteo: превышено число попыток")


def _daily(js: dict) -> pd.DataFrame:
    df = pd.DataFrame(js["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def era5_land_daily(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    js = _get(ARCHIVE, {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                        "daily": ",".join(LAND_DAILY), "models": "era5_land", "timezone": "auto"})
    return _daily(js)


def era5_daily(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    js = _get(ARCHIVE, {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                        "daily": ",".join(ERA5_DAILY), "models": "era5", "timezone": "auto"})
    return _daily(js)


def climate_daily(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """ERA5-Land + ERA5 в одной таблице (суточно)."""
    return era5_land_daily(lat, lon, start, end).join(era5_daily(lat, lon, start, end))


def season_windows(lat: float, lon: float, spring_year: int, issue_mmdd: str = "03-10") -> pd.DataFrame:
    """Только нужные для сезонного индекса окна одной зимы (экономит лимит запросов API):
    осенняя влажность почвы, конец зимы (снег, температура почвы), снегопады с 1 ноября."""
    y0, y1 = spring_year - 1, spring_year
    issue = f"{y1}-{issue_mmdd}"
    base = {"latitude": lat, "longitude": lon, "timezone": "auto"}
    autumn = _daily(_get(ARCHIVE, {**base, "start_date": f"{y0}-10-15", "end_date": f"{y0}-11-30", "models": "era5_land",
                                   "daily": "soil_moisture_0_to_7cm_mean,soil_moisture_7_to_28cm_mean"}))
    late = _daily(_get(ARCHIVE, {**base, "start_date": f"{y1}-02-01", "end_date": issue, "models": "era5_land",
                                 "daily": "snow_depth_max,temperature_2m_mean," + ",".join(c for c in LAND_DAILY if "soil_temp" in c)}))
    snow = _daily(_get(ARCHIVE, {**base, "start_date": f"{y0}-11-01", "end_date": issue, "models": "era5",
                                 "daily": "snowfall_water_equivalent_sum"}))
    return pd.concat([autumn, late, snow]).groupby(level=0).first().sort_index()


def glofas_discharge(lat: float, lon: float, start: str, end: str) -> pd.Series:
    js = _get(FLOOD, {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end, "daily": "river_discharge"})
    return _daily(js)["river_discharge"].astype(float)


def snap_to_river(lat: float, lon: float, step: float = 0.05, radius: int = 2) -> tuple[float, float, float]:
    """Ячейка GloFAS (5 км) рядом с точкой, где средний расход максимален.

    Точка поста редко попадает точно в ячейку реки; берём окрестность ±radius·step градусов.
    Возвращает (lat, lon, средний расход за 2019 г.).
    """
    best = (lat, lon, -1.0)
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            la, lo = round(lat + i * step, 3), round(lon + j * step, 3)
            q = glofas_discharge(la, lo, "2019-01-01", "2019-12-31").mean()
            if np.isfinite(q) and q > best[2]:
                best = (la, lo, float(q))
    return best


def glofas_forecast(lat: float, lon: float, days: int = 30) -> pd.DataFrame:
    """Ансамблевый прогноз GloFAS: медиана, мин., макс. и квартили расхода."""
    js = _get(FLOOD, {"latitude": lat, "longitude": lon, "forecast_days": days, "past_days": 30,
                      "daily": "river_discharge_median,river_discharge_min,river_discharge_max,"
                               "river_discharge_p25,river_discharge_p75"}, cache=False)
    return _daily(js)


def weather_forecast(lat: float, lon: float, days: int = 16) -> pd.DataFrame:
    """Суточный прогноз погоды: температура, осадки, снег — для скорости потепления и прогноза стока."""
    js = _get(FORECAST, {"latitude": lat, "longitude": lon, "forecast_days": days, "past_days": 60, "timezone": "auto",
                         "daily": "temperature_2m_mean,temperature_2m_max,precipitation_sum,snowfall_sum"}, cache=False)
    return _daily(js)
