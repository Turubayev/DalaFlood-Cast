"""Карты затопления по радару Sentinel-1 (видит воду сквозь облака и ночью).

Гладкая водная поверхность отражает радарный сигнал зеркально, поэтому вода тёмная
в канале VV (обычно < −15…−18 дБ). Алгоритм:
1. обратное рассеяние VV → дБ, медианный фильтр от спекл-шума;
2. порог по методу Оцу на гистограмме сцены (при бимодальности «вода / суша»);
3. маска рельефа: вода не может стоять высоко над рекой (HAND > 15 м) — убирает радарные тени;
4. затопление = вода сейчас И не вода в «сухой» референсной сцене (до паводка).

Загрузка снимков — через STAC (Microsoft Planetary Computer, коллекция sentinel-1-rtc,
уже с радиометрической коррекцией рельефа). Нужны пакеты pystac-client, planetary-computer, rasterio.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def to_db(linear: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.clip(linear, 1e-6, None))


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    v = values[np.isfinite(values)]
    hist, edges = np.histogram(v, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[np.argmax(between)])


def water_mask(vv_db: np.ndarray, hand_m: np.ndarray | None = None, max_hand: float = 15.0,
               threshold: float | None = None, min_pixels: int = 10) -> tuple[np.ndarray, float]:
    """Маска воды. Порог: заданный, либо Оцу (ограниченный разумным диапазоном −26…−14 дБ)."""
    img = ndimage.median_filter(np.nan_to_num(vv_db, nan=0.0), size=3)
    img[~np.isfinite(vv_db)] = np.nan
    thr = threshold if threshold is not None else float(np.clip(otsu_threshold(img), -26, -14))
    water = img < thr
    if hand_m is not None:
        water &= np.nan_to_num(hand_m, nan=1e9) <= max_hand
    # Убираем одиночные «шумовые» пиксели.
    lab, n = ndimage.label(water)
    if n:
        sizes = ndimage.sum(water, lab, range(1, n + 1))
        water = np.isin(lab, np.flatnonzero(sizes >= min_pixels) + 1)
    return water, thr


def flood_extent(vv_flood_db: np.ndarray, vv_dry_db: np.ndarray, hand_m=None, **kw) -> dict:
    """Затопленные территории = вода во время паводка минус постоянные водоёмы."""
    wet, thr = water_mask(vv_flood_db, hand_m, **kw)
    perm, _ = water_mask(vv_dry_db, hand_m, threshold=thr, **{k: v for k, v in kw.items() if k != "threshold"})
    # Дополнительно: заметное падение обратного рассеяния (change detection) снижает ложные срабатывания.
    drop = (vv_dry_db - vv_flood_db) > 3.0
    flooded = wet & ~perm & drop
    return {"flooded": flooded, "water_now": wet, "permanent_water": perm, "threshold_db": thr}


def iou(pred: np.ndarray, ref: np.ndarray) -> float:
    """IoU (= CSI) затопленной площади."""
    inter = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()
    return float(inter / union) if union else np.nan


def search_scenes(bbox, start: str, end: str, orbit: str | None = None, max_items: int = 50):
    """Поиск сцен sentinel-1-rtc в Planetary Computer. bbox = (запад, юг, восток, север)."""
    import planetary_computer
    import pystac_client

    cat = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    query = {"sat:orbit_state": {"eq": orbit}} if orbit else None
    items = cat.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=f"{start}/{end}", query=query,
                       max_items=max_items).item_collection()
    return sorted(items, key=lambda it: it.datetime)


def read_vv(item, bbox, resolution_m: float = 20.0):
    """Чтение канала VV в окне bbox (WGS84) с перепроецированием в UTM-сетку снимка."""
    import rasterio
    from affine import Affine
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    href = item.assets["vv"].href
    with rasterio.open(href) as src:
        b = transform_bounds("EPSG:4326", src.crs, *bbox)
        win = from_bounds(*b, transform=src.transform)
        scale = max(1, int(round(resolution_m / src.res[0])))
        out_shape = (max(1, int(win.height // scale)), max(1, int(win.width // scale)))
        arr = src.read(1, window=win, out_shape=out_shape, masked=True).filled(np.nan).astype("float32")
        transform = src.window_transform(win) * Affine.scale(win.width / out_shape[1], win.height / out_shape[0])
        return to_db(arr), transform, src.crs


def synthetic_scene(n: int = 300, seed: int = 0, flooded_frac: float = 0.15):
    """Синтетическая пара сцен (сухая/паводок) с известной «правдой» — для тестов."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n]
    river = np.abs(x - n / 2 - 20 * np.sin(y / 25)) < 4
    flood_zone = np.abs(x - n / 2 - 20 * np.sin(y / 25)) < 4 + flooded_frac * n / 2
    land = rng.normal(-9, 2.0, (n, n))
    water = rng.normal(-21, 1.5, (n, n))
    dry = np.where(river, water, land)
    wet = np.where(flood_zone, rng.normal(-21, 1.5, (n, n)), rng.normal(-9, 2.0, (n, n)))
    return dry.astype("float32"), wet.astype("float32"), flood_zone & ~river
