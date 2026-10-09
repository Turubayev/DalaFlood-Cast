import numpy as np
import pandas as pd
import pytest

from dalaflood import hbv, s1
from dalaflood.hand import flood_probability, hand, synthetic_valley
from dalaflood.level import build_frame, discharge_to_level, fit_rating_curve, make_windows
from dalaflood.seasonal import Season, category, frost_depth, risk_scores, season_factors


def fake_daily(spring: int, snow: float = 0.3, sm: float = 0.3, soil_t: float = -3.0) -> pd.DataFrame:
    idx = pd.date_range(f"{spring - 1}-09-01", f"{spring}-03-31")
    n = len(idx)
    return pd.DataFrame({
        "snow_depth_max": np.where(idx.month.isin([12, 1, 2, 3]), snow, 0.0),
        "soil_moisture_0_to_7cm_mean": sm, "soil_moisture_7_to_28cm_mean": sm,
        "soil_temperature_0_to_7cm_mean": soil_t - 2, "soil_temperature_7_to_28cm_mean": soil_t,
        "soil_temperature_28_to_100cm_mean": soil_t + 2.5, "soil_temperature_100_to_255cm_mean": 2.0,
        "temperature_2m_mean": np.linspace(10, -5, n), "snowfall_water_equivalent_sum": 1.0,
    }, index=idx)


def test_frost_depth_interpolates_zero_crossing():
    assert frost_depth([1, 1, 1, 1]) == 0.0
    d = frost_depth([-4, -2, 1, 3])  # переход между 17.5 и 64 см
    assert 17.5 < d < 64
    assert frost_depth([-5, -4, -3, -1]) == pytest.approx(177.5)


def test_season_factors_and_scores():
    s = Season("2023-2024")
    f = season_factors(fake_daily(2024), s)
    assert f["swe_mm"] == pytest.approx(0.3 * 250)
    assert f["frost_cm"] > 17.5
    rng = np.random.default_rng(0)
    hist = pd.DataFrame({"swe_mm": rng.normal(60, 15, 24), "autumn_sm": rng.normal(0.25, 0.03, 24),
                         "frost_cm": rng.normal(60, 15, 24), "snowfall_mm": rng.normal(80, 20, 24)},
                        index=range(2001, 2025))
    hist.loc[2024] = [110, 0.33, 95, 140]  # 2024: всё сильно выше нормы
    r = risk_scores(hist, 2024)
    assert r["score"] > 1.5 and r["category"] == "очень высокий" and r["combination_2024_like"]
    assert category(-0.5) == "низкий" and category(0.3) == "умеренный"


def test_hbv_snowmelt_produces_spring_peak():
    days = pd.date_range("2020-10-01", "2021-09-30")
    doy = days.dayofyear.to_numpy()
    temp = -12 * np.cos(2 * np.pi * (doy - 15) / 365) + 3
    precip = np.where(np.arange(len(days)) % 4 == 0, 3.0, 0.0)
    pet = np.clip(temp, 0, None) * 0.15
    q = hbv.simulate(hbv.HBVParams(), precip, temp, pet)
    peak_month = days[np.argmax(q)].month
    assert peak_month in (3, 4, 5)
    frozen = hbv.simulate(hbv.HBVParams(FROST=1.0), precip, temp, pet, frost_index=np.where(temp < 0, 1.0, 0.5))
    assert frozen.max() > q.max()  # мёрзлая почва усиливает пик


def test_nse_kge_perfect():
    x = np.random.default_rng(1).gamma(2, 10, 200)
    assert hbv.nse(x, x) == pytest.approx(1.0)
    assert hbv.kge(x, x) == pytest.approx(1.0)


def test_hand_on_synthetic_valley():
    dem = synthetic_valley(n=120)
    h, streams, acc = hand(dem, 30.0, stream_threshold_km2=2.0)
    assert streams.any()
    assert np.nanmax(h[streams]) == pytest.approx(0.0)
    p_low = flood_probability(h, {0.1: 0.5, 0.5: 1.0, 0.9: 1.5})
    p_high = flood_probability(h, {0.1: 2.0, 0.5: 3.0, 0.9: 4.0})
    assert p_high.sum() > p_low.sum()
    assert ((p_low >= 0) & (p_low <= 1)).all()


def test_s1_flood_detection_on_synthetic_scene():
    dry, wet, truth = s1.synthetic_scene(n=200)
    res = s1.flood_extent(wet, dry)
    assert s1.iou(res["flooded"], truth) > 0.8
    assert -26 <= res["threshold_db"] <= -14


def test_level_windows_and_rating():
    idx = pd.date_range("2020-01-01", periods=200)
    q = pd.Series(50 + 30 * np.sin(np.arange(200) / 20) ** 2, index=idx)
    met = pd.DataFrame({"temperature_2m_mean": 0.0, "precipitation_sum": 1.0, "snow_depth_max": 0.1,
                        "soil_temperature_7_to_28cm_mean": -1.0}, index=idx)
    df = build_frame(q, met)
    H, F, Y, I = make_windows(df)
    assert H.shape[1:] == (30, 6) and F.shape[1:] == (3, 2) and Y.shape[1] == 3
    qq = np.linspace(10, 500, 30)
    hh = 1.2 + 0.3 * qq ** 0.45
    h0, a, b = fit_rating_curve(qq, hh)
    assert discharge_to_level([200], (h0, a, b))[0] == pytest.approx(1.2 + 0.3 * 200 ** 0.45, rel=1e-3)
