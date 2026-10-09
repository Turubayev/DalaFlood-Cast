"""Copernicus DEM GLO-30 из открытого бакета AWS (без регистрации).

Плитки 1°×1°, Cloud-Optimized GeoTIFF, поэтому читается только нужное окно.
Лицензия: © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018,
provided under COPERNICUS by the European Union and ESA.
"""
from __future__ import annotations

import math

import numpy as np

BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def tile_url(lat: int, lon: int) -> str:
    ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
    name = f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"
    return f"{BASE}/{name}/{name}.tif"


def read_dem(bbox: tuple[float, float, float, float]) -> tuple[np.ndarray, dict]:
    """Рельеф в окне bbox = (запад, юг, восток, север); окно должно лежать в одной плитке.

    Возвращает массив высот и метаданные (transform, cellsize_m ≈ размер ячейки в метрах).
    """
    import rasterio
    from rasterio.windows import from_bounds

    w, s, e, n = bbox
    if math.floor(w) != math.floor(e - 1e-9) or math.floor(s) != math.floor(n - 1e-9):
        raise ValueError("Окно пересекает границу плиток 1°; разбейте его или используйте свой GeoTIFF")
    with rasterio.open(tile_url(math.floor(s), math.floor(w))) as src:
        win = from_bounds(w, s, e, n, transform=src.transform)
        dem = src.read(1, window=win).astype("float32")
        transform = src.window_transform(win)
    lat_c = (s + n) / 2
    dy = abs(transform.e) * 111_320
    dx = abs(transform.a) * 111_320 * math.cos(math.radians(lat_c))
    return dem, {"transform": transform, "cellsize_m": float((dx + dy) / 2), "crs": "EPSG:4326", "bbox": bbox}


def read_geotiff(path: str) -> tuple[np.ndarray, dict]:
    """Свой рельеф (например, уточнённая съёмка населённого пункта) в метрической проекции."""
    import rasterio

    with rasterio.open(path) as src:
        dem = src.read(1, masked=True).filled(np.nan).astype("float32")
        return dem, {"transform": src.transform, "cellsize_m": float(abs(src.res[0])), "crs": str(src.crs),
                     "bbox": tuple(src.bounds)}
