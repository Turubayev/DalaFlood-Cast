"""Оперативный прогноз расхода/уровня на гидропосту на 72 ч (1–3 суток) с интервалом.

Вход модели на дату t:
* последние 30 суток: расход (наблюдения поста или реанализ GloFAS), температура, осадки,
  высота снега, температура почвы (промерзание), расход по модели HBV;
* прогноз погоды на t+1…t+3: температура, осадки (при обучении — фактические значения
  с шумом, растущим с заблаговременностью, т. е. «perfect prognosis + шум»).
Выход: квантили 0.1 / 0.5 / 0.9 log-расхода на 1, 2, 3 сутки.

Уровень воды получается из расхода через кривую расходов поста H = h0 + a·Q^b
(`fit_rating_curve` подбирает её по парам «уровень–расход», если они есть).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

QUANTILES = (0.1, 0.5, 0.9)
HIST_DAYS = 30
LEAD_DAYS = 3
HIST_COLS = ["logq", "temp", "precip", "snow", "soil_t", "logq_hbv"]
FUT_COLS = ["temp", "precip"]


def build_frame(q: pd.Series, met: pd.DataFrame, q_hbv: pd.Series | None = None) -> pd.DataFrame:
    """Суточная таблица признаков. met: temperature_2m_mean, precipitation_sum, snow_depth_max,
    soil_temperature_7_to_28cm_mean (как в openmeteo.climate_daily)."""
    df = pd.DataFrame(index=q.index)
    df["q"] = q
    df["logq"] = np.log1p(q.clip(lower=0))
    df["temp"] = met["temperature_2m_mean"].reindex(df.index) / 10
    df["precip"] = np.log1p(met["precipitation_sum"].reindex(df.index).clip(lower=0))
    df["snow"] = met["snow_depth_max"].reindex(df.index) * 3
    df["soil_t"] = met["soil_temperature_7_to_28cm_mean"].reindex(df.index) / 5
    df["logq_hbv"] = np.log1p(q_hbv.reindex(df.index).clip(lower=0)) if q_hbv is not None else df["logq"].shift(1)
    # Causal input filling only; never invent observations used as targets.
    df[HIST_COLS] = df[HIST_COLS].ffill(limit=5)
    df['target_logq'] = np.log1p(q.where(q >= 0))
    return df


def make_windows(df: pd.DataFrame, idx: np.ndarray | None = None, noise: bool = True, seed: int = 0,
                 forecast_archive: pd.DataFrame | None = None, require_targets: bool = True):
    rng = np.random.default_rng(seed)
    n = len(df)
    I = np.arange(HIST_DAYS - 1, n - LEAD_DAYS) if idx is None else idx
    I = np.asarray(I, int)
    I = I[(I >= HIST_DAYS - 1) & (I + LEAD_DAYS < n)]
    if not len(I): raise ValueError('No complete history/forecast windows')
    H = df[HIST_COLS].to_numpy(np.float32)
    F = df[FUT_COLS].to_numpy(np.float32)
    hist = np.stack([H[i - HIST_DAYS + 1 : i + 1] for i in I])
    fut = np.stack([F[i + 1 : i + 1 + LEAD_DAYS] for i in I]).copy()
    if forecast_archive is not None:
        fut = archived_weather(df.index[I], forecast_archive)
    elif noise:
        lead = np.arange(1, LEAD_DAYS + 1)[None, :]
        fut[..., 0] += rng.normal(0, 0.1 + 0.05 * lead)  # температура (×10 °C)
        fut[..., 1] *= np.exp(rng.normal(0, 0.25 + 0.1 * lead))  # осадки
    targets = df['target_logq'] if 'target_logq' in df else df['logq']
    y = np.stack([targets.to_numpy(np.float32)[i + 1 : i + 1 + LEAD_DAYS] for i in I])
    ok = np.isfinite(hist).all(axis=(1,2)) & np.isfinite(fut).all(axis=(1,2))
    if require_targets: ok &= np.isfinite(y).all(axis=1)
    if not ok.any(): raise ValueError('No finite windows; missing data must be resolved before training/forecasting')
    return hist[ok].astype(np.float32), fut[ok].astype(np.float32), y[ok].astype(np.float32), I[ok]


def archived_weather(issues, frame):
    d = frame.copy()
    cols = ['issue_time','issued_at','valid_time','temperature_2m_mean','precipitation_sum']
    if not set(cols) <= set(d): raise ValueError('Incomplete weather forecast archive')
    for c in cols[:3]: d[c] = pd.to_datetime(d[c])
    if d[cols[:3]].isna().any().any() or (d.issued_at > d.issue_time).any() or (d.valid_time <= d.issue_time).any():
        raise ValueError('Invalid forecast publication/valid time')
    if d.duplicated(['issue_time','valid_time']).any(): raise ValueError('Duplicate weather forecasts')
    vals = d[cols[3:]].to_numpy(float)
    if not np.isfinite(vals).all() or (vals[:,1]<0).any(): raise ValueError('Invalid forecast weather')
    d = d.set_index(['issue_time','valid_time'])
    keys = [(t,t+pd.Timedelta(days=h)) for t in issues for h in range(1,LEAD_DAYS+1)]
    vals = d.reindex(pd.MultiIndex.from_tuples(keys,names=d.index.names))[cols[3:]].to_numpy(float, copy=True)
    if not np.isfinite(vals).all(): raise ValueError('Forecast archive does not cover all requested issue times/leads')
    vals[:,0] /= 10; vals[:,1] = np.log1p(vals[:,1])
    return vals.reshape(len(issues),LEAD_DAYS,2).astype(np.float32)


class LevelLSTM:
    """LSTM-энкодер по истории + MLP по прогнозу погоды → квантили на 3 суток."""

    def __init__(self, hidden: int = 48, seed: int = 0):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(len(HIST_COLS), hidden, batch_first=True)
                self.head = nn.Sequential(nn.Linear(hidden + LEAD_DAYS * len(FUT_COLS) + 1, hidden), nn.ReLU(),
                                          nn.Linear(hidden, LEAD_DAYS * len(QUANTILES)))

            def forward(self, h, f):
                _, (hn, _) = self.lstm(h)
                last = h[:, -1, 0:1]  # последний log-расход: прогнозируем приращение
                z = torch.cat([hn[0], f.flatten(1), last], 1)
                out = self.head(z).view(-1, LEAD_DAYS, len(QUANTILES))
                mid = last[:, :, None] + out[..., 1:2]
                lo = mid - torch.nn.functional.softplus(out[..., 0:1])
                hi = mid + torch.nn.functional.softplus(out[..., 2:3])
                return torch.cat([lo, mid, hi], -1)

        self.net = Net()
        self.hidden = hidden
        self.margin = np.zeros(LEAD_DAYS, np.float32)  # конформная поправка интервала, log-шкала

    def fit(self, hist, fut, y, epochs: int = 40, lr: float = 3e-3, batch: int = 128, verbose: bool = False):
        import torch

        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        H, F, Y = map(torch.from_numpy, (hist, fut, y))
        q = torch.tensor(QUANTILES)
        for ep in range(epochs):
            perm = torch.randperm(len(H))
            tot = 0.0
            for i in range(0, len(H), batch):
                b = perm[i : i + batch]
                p = self.net(H[b], F[b])
                e = Y[b][..., None] - p
                loss = torch.maximum(q * e, (q - 1) * e).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            if verbose and (ep + 1) % 10 == 0:
                print(f"эпоха {ep + 1}: pinball {tot / len(H):.4f}", flush=True)
        return self

    def _raw(self, hist, fut) -> np.ndarray:
        import torch

        self.net.eval()
        with torch.no_grad():
            return self.net(torch.from_numpy(hist), torch.from_numpy(fut)).numpy()

    def conformalize(self, hist, fut, y, coverage: float = 0.8) -> np.ndarray:
        """CQR на отложенных годах: расширяет интервал 10–90 % так, чтобы он покрывал
        `coverage` наблюдений отдельно для каждой заблаговременности."""
        p = self._raw(hist, fut)
        score = np.maximum(p[..., 0] - y, y - p[..., 2])  # (B, 3)
        n = len(score)
        if not n or not np.isfinite(score).all() or not 0 < coverage < 1:
            raise ValueError('Finite nonempty calibration and coverage in (0,1) required')
        level = min(1.0, np.ceil((n + 1) * coverage) / n)
        self.margin = np.maximum(0, np.quantile(score, level, axis=0, method='higher')).astype(np.float32)
        return self.margin

    def predict(self, hist, fut) -> np.ndarray:
        """(B, 3 суток, 3 квантиля) в м³/с."""
        p = self._raw(hist, fut)
        p[..., 0] -= self.margin
        p[..., 2] += self.margin
        return np.expm1(p).clip(min=0)

    def save(self, path: str | Path):
        import torch

        torch.save({"hidden": self.hidden, "state": self.net.state_dict(), "margin": self.margin}, path)

    @classmethod
    def load(cls, path: str | Path) -> "LevelLSTM":
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(hidden=ck["hidden"])
        m.net.load_state_dict(ck["state"])
        m.margin = np.asarray(ck.get("margin", np.zeros(LEAD_DAYS)), np.float32)
        return m


def peak_errors(obs: pd.Series, sim: pd.Series) -> dict:
    """Ошибка пика весеннего половодья по величине (%) и по времени (сут) — по годам."""
    rows = []
    for y, o in obs.groupby(obs.index.year):
        o = o[(o.index.month >= 3) & (o.index.month <= 6)]
        s = sim.reindex(o.index)
        if o.isna().all() or s.isna().all():
            continue
        rows.append({"year": y, "peak_obs": o.max(), "peak_sim": s.max(),
                     "peak_err_pct": 100 * (s.max() - o.max()) / o.max(),
                     "timing_err_days": (s.idxmax() - o.idxmax()).days})
    return pd.DataFrame(rows)


def fit_rating_curve(q, h) -> tuple[float, float, float]:
    """Подбор кривой расходов H = h0 + a·Q^b по парам наблюдений."""
    from scipy.optimize import curve_fit

    q, h = np.asarray(q, float), np.asarray(h, float)
    ok = np.isfinite(q) & np.isfinite(h) & (q > 0)
    if ok.sum() < 5 or np.unique(q[ok]).size < 5: raise ValueError('Need at least 5 distinct positive discharge/level pairs')
    popt, _ = curve_fit(lambda x, h0, a, b: h0 + a * x**b, q[ok], h[ok], p0=(h[ok].min(), 0.3, 0.5),
                        bounds=([-np.inf, 1e-9, .05], [np.inf, np.inf, 2.0]), maxfev=20000)
    return tuple(float(v) for v in popt)


def discharge_to_level(q, rating: tuple[float, float, float]):
    h0, a, b = rating
    return h0 + a * np.power(np.clip(np.asarray(q, float), 0, None), b)
