"""Ретроспективная проверка сезонного индекса: смог бы он заранее показать высокие паводки?

Для каждого года 2001–2024 индекс считается на 10 марта (без знания весны), затем
сравнивается с пиком весеннего половодья (март–июнь) на постах бассейна по реанализу GloFAS.

    python scripts/hindcast.py --basin esil
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from dalaflood.config import BASINS  # noqa: E402
from dalaflood.data import openmeteo  # noqa: E402
from dalaflood.seasonal import hindcast  # noqa: E402


def spring_peaks(basin, first: int, last: int) -> pd.DataFrame:
    rows = {}
    for g in BASINS[basin].gauges:
        lat, lon, _ = openmeteo.snap_to_river(g.lat, g.lon)
        q = openmeteo.glofas_discharge(lat, lon, f"{first}-01-01", f"{last}-07-31")
        spring = q[(q.index.month >= 3) & (q.index.month <= 6)]
        rows[g.id] = spring.groupby(spring.index.year).max()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", default="esil")
    ap.add_argument("--first", type=int, default=2001)
    ap.add_argument("--last", type=int, default=2024)
    ap.add_argument("--issue", default="03-10")
    a = ap.parse_args()
    h = hindcast(a.basin, a.first, a.last, a.issue)
    basin_score = h.groupby("spring")["score"].mean().rename("basin_score")
    peaks = spring_peaks(a.basin, a.first, a.last)
    # Пик в долях от медианы: сравнимо между постами разного размера.
    rel = (peaks / peaks.median()).mean(axis=1).rename("peak_rel")
    df = pd.concat([basin_score, rel, peaks.add_prefix("peak_")], axis=1).dropna(subset=["basin_score", "peak_rel"])
    rho, p = spearmanr(df["basin_score"], df["peak_rel"])
    top = df["peak_rel"].nlargest(3).index.tolist()
    rank_score = df["basin_score"].rank(ascending=False)
    summary = {
        "basin": a.basin, "years": [int(df.index.min()), int(df.index.max())],
        "spearman_rho": round(float(rho), 2), "p_value": round(float(p), 4),
        "top3_flood_years": [int(y) for y in top],
        "their_index_rank": {int(y): int(rank_score[y]) for y in top},
        "n_years": int(len(df)),
    }
    if 2024 in df.index:
        summary["index_rank_2024"] = int(rank_score[2024])
        summary["districts_2024"] = (h[h.spring == 2024][["district", "score", "category", "combination_2024_like"]]
                                     .sort_values("score", ascending=False).to_dict(orient="records"))
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df.round(3).to_csv(out / f"hindcast_{a.basin}.csv")
    h.to_csv(out / f"hindcast_{a.basin}_districts.csv", index=False)
    (out / f"hindcast_{a.basin}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(df[["basin_score", "peak_rel"]].round(2).to_string())
    print(json.dumps({k: v for k, v in summary.items() if k != "districts_2024"}, ensure_ascii=False, indent=1))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(df["basin_score"], df["peak_rel"], c="#0e7490")
        for y, r in df.iterrows():
            ax.annotate(str(y), (r["basin_score"], r["peak_rel"]), fontsize=7, xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel("Сезонный индекс риска на 10 марта (среднее по районам)")
        ax.set_ylabel("Пик половодья / медианный пик (GloFAS)")
        ax.set_title(f"{BASINS[a.basin].name}: ρ Спирмена = {rho:.2f}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"hindcast_{a.basin}.png", dpi=130)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
