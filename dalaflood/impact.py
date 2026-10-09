"""Оценка последствий: какие населённые пункты попадают в зону возможного затопления.

Населённые пункты берутся из OpenStreetMap (Overpass API). В соответствии с принципами
ответственного использования детальные данные о дамбах и уязвимой инфраструктуре в отчёт
не выводятся — только агрегированные показатели.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


def _overpass(query: str) -> dict:
    """Запрос к Overpass с перебором зеркал (основной сервер часто перегружен) и кэшем на диске."""
    import hashlib
    import json

    from .data.openmeteo import CACHE

    path = CACHE / f"osm_{hashlib.md5(query.encode()).hexdigest()}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    err = None
    for url in OVERPASS_MIRRORS:
        try:
            r = requests.post(url, data={"data": query}, timeout=90, headers={"User-Agent": "DalaFlood-Cast/0.1"})
            r.raise_for_status()
            js = r.json()
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(js), encoding="utf-8")
            return js
        except requests.RequestException as e:
            err = e
    raise RuntimeError(f"Overpass недоступен: {err}")


def settlements(bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    """Населённые пункты OSM в bbox = (запад, юг, восток, север)."""
    w, s, e, n = bbox
    q = f"""[out:json][timeout:60];
    node["place"~"city|town|village|hamlet|suburb"]({s},{w},{n},{e});
    out body;"""
    rows = [{"name": el["tags"].get("name:ru") or el["tags"].get("name") or "—", "place": el["tags"]["place"],
             "population": pd.to_numeric(el["tags"].get("population"), errors="coerce"),
             "lat": el["lat"], "lon": el["lon"]} for el in _overpass(q)["elements"]]
    return pd.DataFrame(rows)


def rivers(bbox: tuple[float, float, float, float], kinds: str = "river|canal") -> list[list[list[float]]]:
    """Линии рек OSM (waterway=river) в bbox: список линий [[lon, lat], …]."""
    w, s, e, n = bbox
    q = f"""[out:json][timeout:60];
    way["waterway"~"{kinds}"]({s},{w},{n},{e});
    out geom;"""
    return [[[p["lon"], p["lat"]] for p in el["geometry"]] for el in _overpass(q)["elements"] if "geometry" in el]


def exposure(prob: np.ndarray, transform, places: pd.DataFrame, radius_cells: int = 5, threshold: float = 0.3) -> pd.DataFrame:
    """Максимальная вероятность затопления в окрестности каждого населённого пункта.

    transform — аффинное преобразование сетки (rasterio), координаты places — в той же CRS.
    """
    if places.empty:
        return places.assign(flood_prob=[])
    inv = ~transform
    out = []
    for _, p in places.iterrows():
        col, row = inv * (p["lon"], p["lat"])
        r, c = int(row), int(col)
        if not (0 <= r < prob.shape[0] and 0 <= c < prob.shape[1]):
            continue
        win = prob[max(0, r - radius_cells): r + radius_cells + 1, max(0, c - radius_cells): c + radius_cells + 1]
        out.append({**p.to_dict(), "flood_prob": float(np.nanmax(win))})
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df[df["flood_prob"] >= threshold].sort_values("flood_prob", ascending=False).reset_index(drop=True)
