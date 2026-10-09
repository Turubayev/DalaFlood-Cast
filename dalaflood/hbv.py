"""Концептуальная модель стока HBV (суточный шаг) с учётом мёрзлой почвы.

Классическая схема HBV-light (Seibert, 2005):
* снег: накопление при T < TT, таяние по градусо-дням CFMAX·(T − TT), водоудерживающая
  способность снега CWH, повторное замерзание CFR;
* почва: доля стока (SM/FC)^BETA, испарение до LP·FC;
* сток: верхняя ёмкость (K0, K1, порог UZL, перколяция PERC) и нижняя (K2);
* трансформация треугольной функцией MAXBAS.

Дополнение для степных рек Казахстана: при глубоком промерзании (frost_index > 0)
инфильтрация подавляется и талая вода сразу идёт в быстрый сток — так отражается эффект
«почвы-асфальта», который определил масштаб паводка 2024 г.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

PARAM_BOUNDS = {
    "TT": (-2.0, 2.0), "CFMAX": (1.0, 8.0), "SFCF": (0.6, 1.4), "CFR": (0.0, 0.1), "CWH": (0.0, 0.2),
    "FC": (50.0, 400.0), "LP": (0.3, 1.0), "BETA": (1.0, 6.0),
    "PERC": (0.0, 4.0), "UZL": (0.0, 70.0), "K0": (0.05, 0.6), "K1": (0.01, 0.3), "K2": (0.001, 0.1),
    "MAXBAS": (1.0, 30.0), "FROST": (0.0, 1.0),
}
# AREA_SCALE (мм/сут → м³/с, ≈ площадь водосбора км² / 86.4) не подбирается перебором:
# у крупных степных рек он порядка 10³, поэтому вычисляется аналитически из водного баланса.


@dataclass
class HBVParams:
    TT: float = 0.0
    CFMAX: float = 3.5
    SFCF: float = 1.0
    CFR: float = 0.05
    CWH: float = 0.1
    FC: float = 150.0
    LP: float = 0.7
    BETA: float = 2.5
    PERC: float = 1.0
    UZL: float = 20.0
    K0: float = 0.2
    K1: float = 0.08
    K2: float = 0.02
    MAXBAS: float = 3.0
    FROST: float = 0.6  # доля подавления инфильтрации на мёрзлой почве
    AREA_SCALE: float = 1.0  # мм/сут → м³/с: площадь водосбора, км² / 86.4 (из водного баланса)

    @classmethod
    def from_vector(cls, v, area_scale: float = 1.0) -> "HBVParams":
        return cls(**dict(zip(PARAM_BOUNDS, map(float, v), strict=True)), AREA_SCALE=area_scale)

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in PARAM_BOUNDS])


def _maxbas_weights(maxbas: float) -> np.ndarray:
    n = max(1, int(np.ceil(maxbas)))
    t = np.arange(n + 1)
    tri = np.interp(t, [0, maxbas / 2, maxbas], [0, 1, 0])
    w = tri[1:] + tri[:-1]
    return w / w.sum()


def simulate(p: HBVParams, precip, temp, pet, frost_index=None, state: dict | None = None, return_state: bool = False):
    """precip, temp, pet — суточные ряды (мм, °C, мм). frost_index ∈ [0, 1] — промерзание почвы.

    Возвращает расход, м³/с (через AREA_SCALE) и, по желанию, конечное состояние.
    """
    P, T, E = (np.asarray(a, float) for a in (precip, temp, pet))
    F = np.zeros_like(P) if frost_index is None else np.clip(np.asarray(frost_index, float), 0, 1)
    n = len(P)
    st = state or {"snow": 0.0, "liq": 0.0, "sm": p.FC * 0.5, "suz": 0.0, "slz": 10.0}
    snow, liq, sm, suz, slz = st["snow"], st["liq"], st["sm"], st["suz"], st["slz"]
    q = np.zeros(n)
    for t in range(n):
        # Снег.
        if T[t] < p.TT:
            snow += P[t] * p.SFCF
            rain = 0.0
            refreeze = min(liq, p.CFR * p.CFMAX * (p.TT - T[t]))
            snow += refreeze
            liq -= refreeze
        else:
            rain = P[t]
            melt = min(snow, p.CFMAX * (T[t] - p.TT))
            snow -= melt
            liq += melt
        release = max(0.0, liq - p.CWH * snow)
        liq -= release
        inflow = rain + release
        # Почва; на мёрзлой почве часть воды идёт мимо почвенного резервуара.
        bypass = inflow * p.FROST * F[t]
        infil = inflow - bypass
        recharge = infil * (min(sm / p.FC, 1.0) ** p.BETA)
        sm += infil - recharge
        et = E[t] * min(sm / (p.LP * p.FC), 1.0)
        sm = max(sm - et, 0.0)
        if sm > p.FC:
            recharge += sm - p.FC
            sm = p.FC
        # Сток.
        suz += recharge + bypass
        perc = min(suz, p.PERC)
        suz -= perc
        slz += perc
        q0 = p.K0 * max(suz - p.UZL, 0.0)
        q1 = p.K1 * suz
        q2 = p.K2 * slz
        suz -= q0 + q1
        slz -= q2
        q[t] = q0 + q1 + q2
    qr = np.convolve(q, _maxbas_weights(p.MAXBAS))[:n]
    out = qr * p.AREA_SCALE
    if return_state:
        return out, {"snow": snow, "liq": liq, "sm": sm, "suz": suz, "slz": slz}
    return out


def hargreaves_pet(temp_mean, temp_max, temp_min, doy, lat: float) -> np.ndarray:
    """Испаряемость по Харгривсу (мм/сут) — когда нет радиационных данных."""
    phi = np.radians(lat)
    doy = np.asarray(doy)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    ra = 24 * 60 / np.pi * 0.082 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    td = np.clip(np.asarray(temp_max) - np.asarray(temp_min), 0, None)
    return np.clip(0.0023 * 0.408 * ra * (np.asarray(temp_mean) + 17.8) * np.sqrt(td), 0, None)


def nse(obs, sim) -> float:
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    ok = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[ok], sim[ok]
    return float(1 - np.sum((o - s) ** 2) / np.sum((o - o.mean()) ** 2))


def kge(obs, sim) -> float:
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    ok = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[ok], sim[ok]
    r = np.corrcoef(o, s)[0, 1]
    return float(1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2))


def calibrate(precip, temp, pet, frost, q_obs, warmup: int = 365, maxiter: int = 60, seed: int = 0) -> tuple[HBVParams, float]:
    """Подбор параметров дифференциальной эволюцией по критерию KGE (с разогревом модели).

    Для каждого набора параметров масштаб AREA_SCALE берётся из водного баланса
    (средний наблюдённый расход / средний модельный слой стока), поэтому перебор ищет
    только форму гидрографа: снег, почва, промерзание, трансформация.
    """
    from scipy.optimize import differential_evolution

    q_obs = np.asarray(q_obs, float)
    obs = q_obs[warmup:]
    ok = np.isfinite(obs)
    bounds = list(PARAM_BOUNDS.values())

    def unit_and_scale(v):
        sim = simulate(HBVParams.from_vector(v), precip, temp, pet, frost)[warmup:]
        m = sim[ok].mean()
        return sim, (obs[ok].mean() / m if m > 1e-9 else 0.0)

    def loss(v):
        sim, scale = unit_and_scale(v)
        if scale <= 0:
            return 10.0
        return -kge(obs, sim * scale)

    res = differential_evolution(loss, bounds, maxiter=maxiter, popsize=10, seed=seed, tol=1e-5, polish=False)
    _, scale = unit_and_scale(res.x)
    return HBVParams.from_vector(res.x, scale), float(-res.fun)


def params_dict(p: HBVParams) -> dict:
    return asdict(p)
