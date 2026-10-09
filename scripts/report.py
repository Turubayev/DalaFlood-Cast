"""Сводный HTML-отчёт для акимата / штаба по бассейну (из готовых результатов в results/).

    python scripts/report.py --basin esil --season 2025-2026
"""
import argparse
import base64
import json
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from dalaflood.config import BASINS, DISCLAIMER, OFFICIAL_CHANNELS  # noqa: E402

CAT_COLORS = {"низкий": "#d1fadf", "умеренный": "#fef0c7", "высокий": "#fedf89", "очень высокий": "#fda29b"}


def img(path: Path) -> str:
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%">'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", default="esil")
    ap.add_argument("--season", required=True)
    a = ap.parse_args()
    b = BASINS[a.basin]
    res = ROOT / "results"
    parts = [f"<h1>Паводковая обстановка: бассейн {escape(b.name)}, весна {a.season.split('-')[1]}</h1>",
             f'<p class="warn">{escape(DISCLAIMER)}</p>',
             "<p>Официальные каналы: " + "; ".join(
                 f'<a href="{u}">{escape(n)}</a>' if u.startswith("http") else f"{escape(n)}: <b>{u}</b>"
                 for n, u in OFFICIAL_CHANNELS) + "</p>"]

    seasonal = res / f"seasonal_{a.basin}_{a.season}.csv"
    if seasonal.exists():
        df = pd.read_csv(seasonal)
        rows = "".join(
            f"<tr style='background:{CAT_COLORS.get(r.category, '#fff')}'><td>{escape(r.district)}</td><td>{escape(r.region)}</td>"
            f"<td>{r.score:+.2f}</td><td>{escape(r.category)}</td><td>{r.swe_mm:.0f} ({r.z_swe_mm:+.1f})</td>"
            f"<td>{r.autumn_sm:.3f} ({r.z_autumn_sm:+.1f})</td><td>{r.frost_cm:.0f} ({r.z_frost_cm:+.1f})</td>"
            f"<td>{'да' if r.combination_2024_like else '—'}</td></tr>" for r in df.itertuples())
        parts.append("<h2>1. Сезонный индекс риска по районам</h2>"
                     "<table><tr><th>Район</th><th>Область</th><th>Индекс</th><th>Категория</th><th>Снег, мм в.э. (z)</th>"
                     "<th>Влажность почвы осенью (z)</th><th>Промерзание, см (z)</th><th>Сочетание как в 2024</th></tr>"
                     f"{rows}</table><p class='note'>Индекс — отклонение от климата района (z-оценки, 2001–2023). "
                     "Оценка условий формирования стока, а не прогноз ущерба.</p>")
    hc = res / f"hindcast_{a.basin}.json"
    if hc.exists():
        h = json.loads(hc.read_text(encoding="utf-8"))
        parts.append(f"<h2>2. Насколько индексу можно доверять</h2><p>Проверка на {h['n_years']} годах: ранговая корреляция "
                     f"индекса на 10 марта с пиком половодья ρ = {h['spearman_rho']:.2f}. "
                     + (f"Весна 2024 г. — {h['index_rank_2024']}-е место по индексу среди всех лет." if "index_rank_2024" in h else "")
                     + f"</p>{img(res / f'hindcast_{a.basin}.png')}")
    for g in b.gauges:
        fc = res / f"forecast_{g.id}.json"
        lv = res / f"level_{g.id}.json"
        if not (fc.exists() or lv.exists()):
            continue
        parts.append(f"<h2>3. Пост {escape(g.name)}</h2>")
        if fc.exists():
            f = json.loads(fc.read_text(encoding="utf-8"))
            rows = "".join(f"<tr><td>{r['date']}</td><td>{r['q50']}</td><td>{r['q10']}–{r['q90']}</td><td>{r['glofas_median']}</td></tr>"
                           for r in f["forecast"])
            parts.append(f"<p>Выпуск {f['issued']}, последний расход {f['last_q']} м³/с.</p><table><tr><th>Дата</th>"
                         f"<th>Расход, м³/с (медиана)</th><th>Интервал 10–90%</th><th>GloFAS (медиана)</th></tr>{rows}</table>")
        if lv.exists():
            m = json.loads(lv.read_text(encoding="utf-8"))
            parts.append(f"<p class='note'>Качество модели на независимых годах (с {m['test_from'][:4]}): NSE на 3-и сутки "
                         f"{m['lstm']['day3']['nse']:.2f} (модель «как вчера»: {m['persistence']['day3']['nse']:.2f}). "
                         f"Цель обучения: {escape(m['target'])}.</p>{img(res / f'level_{g.id}_2024.png')}")
    for p in sorted(res.glob("flood_map_*.png")):
        parts.append(f"<h2>4. Зона возможного затопления: {escape(p.stem.replace('flood_map_', ''))}</h2>{img(p)}")
    for p in sorted(res.glob("*_flooded_*.png")):
        parts.append(f"<h2>5. Фактическая вода по Sentinel-1: {escape(p.stem)}</h2>{img(p)}")

    html = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>DalaFlood-Cast: {escape(b.name)}</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#1d2433}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #e4e7ec;padding:4px 6px}}
.warn{{background:#fff4e5;border-left:4px solid #f79009;padding:8px 12px}}.note{{color:#667085;font-size:13px}}</style>
</head><body>{''.join(parts)}<hr><p class="note">{escape(DISCLAIMER)}</p></body></html>"""
    out = res / f"report_{a.basin}_{a.season}.html"
    out.write_text(html, encoding="utf-8")
    print("Отчёт:", out)


if __name__ == "__main__":
    main()
