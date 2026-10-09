"""Сезонный индекс паводкового риска (оценка за 2–8 недель до паводка).

Факторы (по каждому району и зиме):
* **swe_mm** — запас воды в снеге: максимальная высота снега ERA5-Land за февраль–дату
  выпуска × плотность снега (≈ 250 кг/м³ в конце зимы);
* **snowfall_mm** — сумма водного эквивалента снегопадов с 1 ноября (ERA5);
* **autumn_sm** — влажность почвы 0–28 см во второй половине октября–ноябре (до промерзания):
  переувлажнённая почва не впитывает талую воду;
* **frost_cm** — глубина промерзания: глубина перехода температуры почвы через 0 °C по профилю
  ERA5-Land (слои 0–7, 7–28, 28–100, 100–255 см) во второй половине зимы;
  мёрзлая почва работает «как асфальт».

Каждый фактор переводится в z-оценку относительно климата этого же района (базовые зимы),
индекс — взвешенная сумма. Важен не один фактор, а их **сочетание**, поэтому отдельно
считается флаг «все три фактора выше нормы» — как весной 2024 г.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BASINS, Basin
from .data import openmeteo

SNOW_DENSITY = 250.0  # кг/м³
LAYER_DEPTHS_CM = np.array([3.5, 17.5, 64.0, 177.5])
LAYER_COLS = [
    "soil_temperature_0_to_7cm_mean",
    "soil_temperature_7_to_28cm_mean",
    "soil_temperature_28_to_100cm_mean",
    "soil_temperature_100_to_255cm_mean",
]
WEIGHTS = {"swe_mm": 0.35, "autumn_sm": 0.25, "frost_cm": 0.20, "snowfall_mm": 0.20}
FACTOR_NAMES = {
    "swe_mm": "Запас воды в снеге, мм",
    "snowfall_mm": "Снегопады с 1 ноября, мм в.э.",
    "autumn_sm": "Осенняя влажность почвы, м³/м³",
    "frost_cm": "Глубина промерзания, см",
}
CATEGORIES = [(-np.inf, "низкий"), (0.25, "умеренный"), (0.75, "высокий"), (1.5, "очень высокий")]


@dataclass
class Season:
    """Зима «2023-2024»: с 1 сентября 2023 по дату выпуска весной 2024."""
    label: str
    issue_mmdd: str = "03-10"

    @property
    def spring_year(self) -> int:
        return int(self.label.split("-")[1])

    @property
    def issue_date(self) -> pd.Timestamp:
        return pd.Timestamp(f"{self.spring_year}-{self.issue_mmdd}")

    @classmethod
    def from_year(cls, spring_year: int, issue_mmdd: str = "03-10") -> "Season":
        return cls(f"{spring_year - 1}-{spring_year}", issue_mmdd)


def frost_depth(temps: np.ndarray) -> float:
    """Глубина промерзания (см) по профилю средних температур слоёв: линейная интерполяция нуля."""
    t = np.asarray(temps, float)
    if not np.isfinite(t).all() or t[0] >= 0:
        return 0.0
    for k in range(1, len(t)):
        if t[k] >= 0:
            z0, z1 = LAYER_DEPTHS_CM[k - 1], LAYER_DEPTHS_CM[k]
            return float(z0 + (z1 - z0) * (-t[k - 1]) / (t[k] - t[k - 1]))
    return float(LAYER_DEPTHS_CM[-1])


def season_factors(daily: pd.DataFrame, season: Season) -> dict:
    """Факторы риска из суточной таблицы (climate_daily) для одной зимы."""
    y1, issue = season.spring_year, season.issue_date
    d = daily.loc[pd.Timestamp(f"{y1 - 1}-09-01"): issue]
    late = d.loc[pd.Timestamp(f"{y1}-02-01"): issue]
    autumn = d.loc[pd.Timestamp(f"{y1 - 1}-10-15"): pd.Timestamp(f"{y1 - 1}-11-30")]
    frost_win = d.loc[issue - pd.Timedelta(days=25): issue]
    sm = autumn[["soil_moisture_0_to_7cm_mean", "soil_moisture_7_to_28cm_mean"]].mean().mean()
    return {
        "swe_mm": float(late["snow_depth_max"].max() * SNOW_DENSITY) if len(late) else np.nan,
        "snowfall_mm": float(d.loc[pd.Timestamp(f"{y1 - 1}-11-01"):, "snowfall_water_equivalent_sum"].sum(min_count=1)),
        "autumn_sm": float(sm),
        "frost_cm": frost_depth(frost_win[LAYER_COLS].mean().to_numpy()),
        "t_last10": float(d["temperature_2m_mean"].iloc[-10:].mean()),
    }


def district_history(lat: float, lon: float, first_spring: int, last_spring: int, issue_mmdd: str = "03-10") -> pd.DataFrame:
    """Факторы для всех зим района. Каждая зима — три небольших запроса (кэшируются)."""
    latest = pd.Timestamp.today() - pd.Timedelta(days=6)  # задержка архива ERA5
    rows = []
    for y in range(first_spring, last_spring + 1):
        s = Season.from_year(y, issue_mmdd)
        if s.issue_date > latest:
            continue
        daily = openmeteo.season_windows(lat, lon, y, issue_mmdd)
        rows.append({"spring": y, **season_factors(daily, s)})
    return pd.DataFrame(rows).set_index("spring")


def risk_scores(hist: pd.DataFrame, target_spring: int, baseline: tuple[int, int] = (2001, 2023)) -> dict:
    """z-оценки факторов целевой зимы относительно базовых лет (без целевого года) и итоговый индекс."""
    base = hist.loc[(hist.index >= baseline[0]) & (hist.index <= baseline[1]) & (hist.index != target_spring)]
    row = hist.loc[target_spring]
    z = {}
    for f in WEIGHTS:
        sd = base[f].std()
        z[f] = float((row[f] - base[f].mean()) / sd) if sd > 0 else 0.0
    score = sum(WEIGHTS[f] * z[f] for f in WEIGHTS)
    combo = z["swe_mm"] > 0.5 and z["autumn_sm"] > 0.5 and z["frost_cm"] > 0.0
    return {
        **{f: float(row[f]) for f in WEIGHTS},
        **{f"z_{f}": round(v, 2) for f, v in z.items()},
        "score": round(float(score), 2),
        "category": category(score),
        "combination_2024_like": bool(combo),
        "percentile": float((base.assign(s=_scores(base, base)).s < score).mean()),
    }


def _scores(df: pd.DataFrame, base: pd.DataFrame) -> pd.Series:
    out = 0
    for f, w in WEIGHTS.items():
        out = out + w * (df[f] - base[f].mean()) / base[f].std()
    return out


def category(score: float) -> str:
    label = CATEGORIES[0][1]
    for thr, name in CATEGORIES:
        if score >= thr:
            label = name
    return label


def basin_risk(basin: Basin | str, season: Season, baseline=(2001, 2023), first_spring: int = 2001) -> pd.DataFrame:
    """Индекс риска по всем районам бассейна на дату выпуска."""
    b = BASINS[basin] if isinstance(basin, str) else basin
    rows = []
    for d in b.districts:
        hist = district_history(d.lat, d.lon, first_spring, season.spring_year, season.issue_mmdd)
        if season.spring_year not in hist.index:
            raise ValueError(f"Нет данных ERA5-Land на {season.issue_date.date()} (задержка архива ~5 суток)")
        rows.append({"district_id": d.id, "district": d.name, "region": d.region, "lat": d.lat, "lon": d.lon,
                     **risk_scores(hist, season.spring_year, baseline)})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def hindcast(basin: Basin | str, first_spring: int = 2001, last_spring: int = 2024, issue_mmdd: str = "03-10") -> pd.DataFrame:
    """Индекс риска всех районов за все годы (leave-one-out: климат считается без оцениваемого года)."""
    b = BASINS[basin] if isinstance(basin, str) else basin
    out = []
    for d in b.districts:
        hist = district_history(d.lat, d.lon, first_spring, last_spring, issue_mmdd)
        for y in hist.index:
            r = risk_scores(hist, y, baseline=(first_spring, last_spring))
            out.append({"district_id": d.id, "district": d.name, "spring": y, **r})
    return pd.DataFrame(out)
