"""Бассейны, районы и гидропосты.

Координаты районов — центры (районные центры), координаты постов — точки на реке.
Они ориентировочные: для GloFAS точка автоматически «притягивается» к ячейке реки
(см. data/openmeteo.py: snap_to_river), для детальных расчётов уточните их.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class District:
    id: str
    name: str
    region: str
    lat: float
    lon: float


@dataclass(frozen=True)
class Gauge:
    id: str
    name: str
    river: str
    lat: float
    lon: float
    # Кривая расходов Q→H: H = h0 + a·Q^b (м над нулём поста). Значения-заглушки;
    # при наличии данных поста подбираются scripts/train_level.py --levels.
    rating: tuple[float, float, float] = (0.0, 0.25, 0.5)


@dataclass(frozen=True)
class Basin:
    id: str
    name: str
    bbox: tuple[float, float, float, float]  # запад, юг, восток, север
    districts: list[District] = field(default_factory=list)
    gauges: list[Gauge] = field(default_factory=list)


BASINS: dict[str, Basin] = {
    "esil": Basin(
        "esil", "Есиль (Ишим)", (66.0, 50.5, 73.5, 55.5),
        districts=[
            District("astana", "г. Астана", "Астана", 51.17, 71.43),
            District("tselinograd", "Целиноградский р-н", "Акмолинская обл.", 51.03, 71.12),
            District("arshaly", "Аршалынский р-н", "Акмолинская обл.", 50.84, 72.18),
            District("astrakhan", "Астраханский р-н", "Акмолинская обл.", 51.53, 69.80),
            District("atbasar", "Атбасарский р-н", "Акмолинская обл.", 51.81, 68.36),
            District("zhaksy", "Жаксынский р-н", "Акмолинская обл.", 51.91, 67.31),
            District("esil_akm", "Есильский р-н", "Акмолинская обл.", 51.96, 66.40),
            District("petropavl", "г. Петропавловск", "Северо-Казахстанская обл.", 54.87, 69.13),
            District("kyzylzhar", "Кызылжарский р-н", "Северо-Казахстанская обл.", 54.75, 69.00),
            District("esil_sko", "Есильский р-н", "Северо-Казахстанская обл.", 54.35, 68.45),
            District("mamlyut", "Мамлютский р-н", "Северо-Казахстанская обл.", 54.94, 68.54),
            District("zhambyl_sko", "Жамбылский р-н", "Северо-Казахстанская обл.", 54.67, 67.15),
        ],
        gauges=[
            Gauge("esil_astana", "Есиль — Астана", "Есиль", 51.15, 71.45),
            Gauge("esil_yavlenka", "Есиль — Явленка", "Есиль", 54.35, 68.47),
            Gauge("esil_petropavl", "Есиль — Петропавловск", "Есиль", 54.86, 69.15),
        ],
    ),
    "nura": Basin(
        "nura", "Нура", (69.5, 49.3, 74.0, 51.0),
        districts=[
            District("temirtau", "г. Темиртау", "Карагандинская обл.", 50.06, 72.96),
            District("nura_d", "Нуринский р-н", "Карагандинская обл.", 50.26, 71.55),
            District("abay", "Абайский р-н", "Карагандинская обл.", 49.63, 72.85),
            District("bukhar", "Бухар-Жырауский р-н", "Карагандинская обл.", 50.06, 73.72),
            District("korgalzhyn", "Коргалжынский р-н", "Акмолинская обл.", 50.58, 70.01),
        ],
        gauges=[Gauge("nura_kievka", "Нура — Киевка", "Нура", 50.27, 71.55)],
    ),
    "tobol": Basin(
        "tobol", "Тобол", (61.0, 51.5, 65.5, 55.0),
        districts=[
            District("kostanay", "г. Костанай", "Костанайская обл.", 53.22, 63.63),
            District("rudny", "г. Рудный", "Костанайская обл.", 52.96, 63.12),
            District("lisakovsk", "г. Лисаковск", "Костанайская обл.", 52.54, 62.49),
            District("karabalyk", "Карабалыкский р-н", "Костанайская обл.", 53.85, 62.08),
            District("taran", "р-н Беимбета Майлина", "Костанайская обл.", 52.63, 62.93),
        ],
        gauges=[Gauge("tobol_kostanay", "Тобол — Костанай", "Тобол", 53.22, 63.62)],
    ),
    "ural": Basin(
        "ural", "Урал (Жайык) и Илек", (50.5, 46.8, 58.0, 51.8),
        districts=[
            District("uralsk", "г. Уральск", "Западно-Казахстанская обл.", 51.23, 51.37),
            District("akzhaik", "Акжаикский р-н", "Западно-Казахстанская обл.", 50.20, 51.17),
            District("burlin", "Бурлинский р-н", "Западно-Казахстанская обл.", 51.17, 52.99),
            District("atyrau", "г. Атырау", "Атырауская обл.", 47.11, 51.92),
            District("makhambet", "Махамбетский р-н", "Атырауская обл.", 47.67, 51.58),
            District("aktobe", "г. Актобе", "Актюбинская обл.", 50.28, 57.17),
        ],
        gauges=[
            Gauge("ural_uralsk", "Урал — Уральск", "Урал", 51.20, 51.35),
            Gauge("ural_atyrau", "Урал — Атырау", "Урал", 47.12, 51.90),
            Gauge("ilek_aktobe", "Илек — Актобе", "Илек", 50.30, 57.15),
        ],
    ),
}

OFFICIAL_CHANNELS = [
    ("МЧС Республики Казахстан", "https://www.gov.kz/memleket/entities/emer"),
    ("РГП «Казгидромет»", "https://www.kazhydromet.kz"),
    ("Единый номер экстренных служб", "112"),
]

DISCLAIMER = (
    "DalaFlood-Cast — инструмент поддержки решений. Он не является официальной системой оповещения "
    "населения. Официальные предупреждения о ЧС выпускают МЧС РК и РГП «Казгидромет»."
)


def all_districts() -> list[tuple[Basin, District]]:
    return [(b, d) for b in BASINS.values() for d in b.districts]


def find_gauge(gauge_id: str) -> tuple[Basin, Gauge]:
    for b in BASINS.values():
        for g in b.gauges:
            if g.id == gauge_id:
                return b, g
    raise KeyError(f"Неизвестный пост {gauge_id}; доступны: {[g.id for b in BASINS.values() for g in b.gauges]}")
