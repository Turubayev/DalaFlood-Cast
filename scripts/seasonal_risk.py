"""Сезонный индекс паводкового риска по районам бассейна (реальные данные ERA5-Land/ERA5).

    python scripts/seasonal_risk.py --basin esil --season 2025-2026
    python scripts/seasonal_risk.py --basin all --season 2023-2024 --issue 03-01

Результат: results/seasonal_<basin>_<season>.csv/.md и web/data/seasonal.json для веб-карты.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from dalaflood.config import BASINS, DISCLAIMER  # noqa: E402
from dalaflood.seasonal import FACTOR_NAMES, Season, basin_risk  # noqa: E402


def to_markdown(df: pd.DataFrame, title: str) -> str:
    lines = [f"# {title}", "", f"> {DISCLAIMER}", "",
             "| Район | Область | Индекс | Категория | Снег, мм в.э. (z) | Влажность осенью (z) | Промерзание, см (z) | Сочетание как в 2024 |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['district']} | {r['region']} | {r['score']:+.2f} | {r['category']} | "
                     f"{r['swe_mm']:.0f} ({r['z_swe_mm']:+.1f}) | {r['autumn_sm']:.3f} ({r['z_autumn_sm']:+.1f}) | "
                     f"{r['frost_cm']:.0f} ({r['z_frost_cm']:+.1f}) | {'да' if r['combination_2024_like'] else '—'} |")
    lines += ["", "Индекс — взвешенная сумма z-оценок факторов относительно климата района (2001–2023): "
              "0 — норма, +1 — на одно стандартное отклонение выше нормы. Это оценка условий, а не прогноз ущерба."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", default="esil", help="esil | nura | tobol | ural | all")
    ap.add_argument("--season", required=True, help="зима, например 2025-2026")
    ap.add_argument("--issue", default="03-10", help="дата выпуска ММ-ДД (по умолчанию 10 марта)")
    a = ap.parse_args()
    season = Season(a.season, a.issue)
    basins = list(BASINS) if a.basin == "all" else [a.basin]
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    web = ROOT / "web" / "data"
    web.mkdir(parents=True, exist_ok=True)
    geo = json.loads((web / "seasonal.json").read_text(encoding="utf-8")) if (web / "seasonal.json").exists() else {}
    for b in basins:
        print(f"Бассейн {BASINS[b].name}, зима {a.season}, выпуск {season.issue_date.date()}…", flush=True)
        df = basin_risk(b, season)
        df.insert(0, "basin", b)
        stem = f"seasonal_{b}_{a.season}"
        df.to_csv(out_dir / f"{stem}.csv", index=False)
        (out_dir / f"{stem}.md").write_text(to_markdown(df, f"Сезонный риск паводка: {BASINS[b].name}, весна {season.spring_year}"),
                                            encoding="utf-8")
        cols = ["district", "region", "lat", "lon", "score", "category", "swe_mm", "autumn_sm", "frost_cm", "snowfall_mm",
                "z_swe_mm", "z_autumn_sm", "z_frost_cm", "z_snowfall_mm", "combination_2024_like"]
        geo.setdefault(a.season, {})[b] = json.loads(df[cols].round(3).to_json(orient="records", force_ascii=False))
        print(df[["district", "score", "category", "swe_mm", "autumn_sm", "frost_cm"]].round(2).to_string(index=False))
    geo["_factors"] = FACTOR_NAMES
    (web / "seasonal.json").write_text(json.dumps(geo, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
