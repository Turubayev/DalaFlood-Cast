"""Зоны затопления по рельефу: индекс HAND (Height Above Nearest Drainage).

HAND — превышение каждой ячейки над ближайшим водотоком, в который она дренируется.
Если прогнозный уровень воды поднимается на Δh над меженью, затопленными считаются
ячейки с HAND < Δh, гидравлически связанные с рекой. Метод быстрый и не требует
гидравлической модели; для пилотных участков рекомендуется уточнение LISFLOOD-FP / HEC-RAS.

Алгоритм (чистый NumPy + heapq, подходит для фрагментов до ~2000×2000 ячеек):
1. заполнение бессточных понижений (priority-flood с ε-уклоном, Barnes et al., 2014);
2. направления стока D8 по заполненному рельефу;
3. накопление стока и выделение водотоков по порогу площади;
4. HAND = высота ячейки − высота ячейки водотока, в которую она дренируется.
"""
from __future__ import annotations

import heapq

import numpy as np

# D8: смещения (dr, dc) и длины шага
D8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
D8_DIST = np.array([1, np.sqrt(2)] * 4)


def fill_depressions(dem: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Priority-flood с ε: каждая ячейка становится чуть выше ячейки, из которой её «залили»,
    поэтому на плоских участках сохраняется направление стока к краю."""
    nr, nc = dem.shape
    z = dem.astype(np.float64).copy()
    nodata = ~np.isfinite(z)
    z[nodata] = -np.inf
    closed = np.zeros_like(z, bool)
    pq: list = []
    for r in range(nr):
        for c in (0, nc - 1):
            heapq.heappush(pq, (z[r, c], r, c))
            closed[r, c] = True
    for c in range(1, nc - 1):
        for r in (0, nr - 1):
            heapq.heappush(pq, (z[r, c], r, c))
            closed[r, c] = True
    while pq:
        h, r, c = heapq.heappop(pq)
        for dr, dc in D8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < nr and 0 <= cc < nc and not closed[rr, cc]:
                closed[rr, cc] = True
                if z[rr, cc] <= h:
                    z[rr, cc] = h + eps
                heapq.heappush(pq, (z[rr, cc], rr, cc))
    z[nodata] = np.nan
    return z


def flow_direction(z: np.ndarray, cellsize: float = 30.0) -> np.ndarray:
    """Индекс направления D8 (0–7) по наибольшему уклону; −1 — сток за пределы/впадина."""
    nr, nc = z.shape
    pad = np.pad(z, 1, constant_values=-np.inf)  # край — сток наружу
    slopes = np.empty((8, nr, nc))
    for k, (dr, dc) in enumerate(D8):
        nb = pad[1 + dr : 1 + dr + nr, 1 + dc : 1 + dc + nc]
        slopes[k] = (z - nb) / (D8_DIST[k] * cellsize)
    slopes = np.nan_to_num(slopes, nan=-np.inf)
    fdir = np.argmax(slopes, axis=0)
    fdir[np.max(slopes, axis=0) <= 0] = -1
    return fdir


def _downstream(fdir: np.ndarray) -> np.ndarray:
    nr, nc = fdir.shape
    rr, cc = np.indices(fdir.shape)
    dr = np.array([d[0] for d in D8])[np.clip(fdir, 0, 7)]
    dc = np.array([d[1] for d in D8])[np.clip(fdir, 0, 7)]
    r2, c2 = rr + dr, cc + dc
    inside = (fdir >= 0) & (r2 >= 0) & (r2 < nr) & (c2 >= 0) & (c2 < nc)
    return np.where(inside, r2 * nc + c2, -1).ravel()


def flow_accumulation(z: np.ndarray, fdir: np.ndarray) -> np.ndarray:
    """Число ячеек, дренирующихся через каждую ячейку (включая её саму)."""
    down = _downstream(fdir)
    order = np.argsort(-np.nan_to_num(z, nan=-np.inf).ravel(), kind="stable")  # сверху вниз
    acc = np.ones(z.size)
    for i in order:
        j = down[i]
        if j >= 0:
            acc[j] += acc[i]
    return acc.reshape(z.shape)


def rasterize_lines(lines, transform, shape) -> np.ndarray:
    """Растеризация линий (список массивов [[lon, lat], …]) в маску сетки с аффинным transform."""
    mask = np.zeros(shape, bool)
    inv = ~transform
    for line in lines:
        pts = np.asarray(line, float)
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            c0, r0 = inv * (x0, y0)
            c1, r1 = inv * (x1, y1)
            n = int(max(abs(c1 - c0), abs(r1 - r0)) * 2) + 2
            rr = np.round(np.linspace(r0, r1, n)).astype(int)
            cc = np.round(np.linspace(c0, c1, n)).astype(int)
            ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
            mask[rr[ok], cc[ok]] = True
    return mask


def main_river(lines_mask: list[np.ndarray], dem: np.ndarray, tolerance_m: float = 3.0) -> np.ndarray:
    """Из нескольких растеризованных линий рек оставляет главную — самую низкую по рельефу
    (притоки и каналы на террасах лежат выше и не должны служить отсчётом уровня)."""
    if not lines_mask:
        return np.zeros(dem.shape, bool)
    levels = [np.nanmedian(dem[m]) if m.any() else np.inf for m in lines_mask]
    lowest = min(levels)
    out = np.zeros(dem.shape, bool)
    for m, lv in zip(lines_mask, levels):
        if lv <= lowest + tolerance_m:
            out |= m
    return out


def hand(dem: np.ndarray, cellsize: float = 30.0, stream_threshold_km2: float = 5.0,
         river_mask: np.ndarray | None = None, burn_m: float = 5.0):
    """Возвращает (hand, streams, acc). Порог водотока — площадь водосбора в км².

    river_mask — известное русло (например, из OpenStreetMap). На небольшом фрагменте крупная
    река приходит «снаружи», и накопления стока внутри окна не хватает, чтобы её выделить;
    поэтому русло «прожигается» в рельеф на burn_m метров (stream burning), а HAND считается
    от исходной высоты русла.
    """
    dem = dem.astype(np.float64)
    work = dem.copy()
    if river_mask is not None:
        work[river_mask] -= burn_m
    z = fill_depressions(work)
    fdir = flow_direction(z, cellsize)
    acc = flow_accumulation(z, fdir)
    cell_km2 = (cellsize / 1000) ** 2
    # Порог не может превышать площадь окна: иначе на фрагменте не окажется ни одного водотока.
    thr = min(stream_threshold_km2, 0.02 * dem.size * cell_km2)
    streams = acc * cell_km2 >= thr
    if river_mask is not None:
        # Уровень задаётся для главной реки, поэтому HAND отсчитывается только от её русла.
        streams = river_mask.copy()
    down = _downstream(fdir)
    ref = dem.ravel()  # высоты без «прожига»
    drain = np.full(z.size, np.nan)
    drain[streams.ravel()] = ref[streams.ravel()]
    order = np.argsort(np.nan_to_num(z.ravel(), nan=np.inf), kind="stable")  # снизу вверх
    for i in order:
        if np.isnan(drain[i]):
            j = down[i]
            if j >= 0:
                drain[i] = drain[j]
    h = ref - drain
    h = np.where(np.isfinite(h), np.clip(h, 0, None), np.nan)
    return h.reshape(z.shape), streams, acc


def flood_depth(hand_m: np.ndarray, stage_rise_m: float) -> np.ndarray:
    """Глубина затопления (м) при подъёме воды на stage_rise_m над меженью."""
    return np.where(hand_m < stage_rise_m, stage_rise_m - hand_m, 0.0)


def flood_probability(hand_m: np.ndarray, stage_quantiles: dict[float, float]) -> np.ndarray:
    """Вероятность затопления ячейки по квантилям прогнозного подъёма уровня.

    Интерполируем функцию распределения подъёма уровня по квантилям: P(затоплено) =
    P(подъём > HAND). Возвращает массив 0..1 (вероятность, а не «затопит / не затопит»).
    """
    qs = np.array(sorted(stage_quantiles))
    levels = np.array([stage_quantiles[q] for q in qs])
    # P(stage > h) = 1 − F(h); F интерполируется линейно между квантилями, хвосты — 0 и 1.
    xs = np.r_[levels[0] - (levels[1] - levels[0]) * qs[0] / (qs[1] - qs[0]), levels,
               levels[-1] + (levels[-1] - levels[-2]) * (1 - qs[-1]) / (qs[-1] - qs[-2])]
    fs = np.r_[0.0, qs, 1.0]
    F = np.interp(np.nan_to_num(hand_m, nan=1e9), xs, fs, left=0.0, right=1.0)
    return 1.0 - F


def synthetic_valley(n: int = 300, seed: int = 0, cellsize: float = 30.0) -> np.ndarray:
    """Тестовый рельеф: пологая степная долина с меандрирующей рекой и террасами."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] * cellsize
    river_x = n * cellsize / 2 + 1200 * np.sin(y / 2500) + 400 * np.sin(y / 700)
    dist = np.abs(x - river_x)
    floodplain = np.where(dist < 900, dist * 0.002, 1.8 + (dist - 900) * 0.012)
    z = 120 - y * 0.0004 + floodplain + rng.normal(0, 0.15, (n, n))
    return z
