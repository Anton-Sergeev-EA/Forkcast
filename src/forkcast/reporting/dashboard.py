"""Интерактивный дашборд (Plotly, один HTML-файл, работает офлайн).

Выбор ряда (рынок, блоки, доля электро, любой из 14 сегментов) — выпадающий список; на графике:
факт, веер взвешенного прогноза (50/80/95%) и медианы всех сценариев; подсказки при наведении.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

NEUTRAL = "#3d3c39"
BLUE = "#2a78d6"


def _name(res, code):
    base = {"total_market": "Рынок, всего", "total_ice": "ДВС, всего", "total_electric": "Электро, всего",
            "electric_share": "Доля электро"}
    if code in base:
        return base[code]
    return ("ДВС " if code.startswith("ice") else "Электро ") + res.labels.get(code, code)


def build_dashboard(res, path: Path) -> Path:
    series = ["total_market", "total_ice", "total_electric", "electric_share"] + [
        c for c in res.wide.columns if c.startswith(("ice_", "el_"))]
    hist_x = [str(p) for p in res.wide.index]
    fc_x = [str(p) for p in res.weighted.periods]
    fig = go.Figure()
    groups = []
    for si, ser in enumerate(series):
        vis = si == 0
        is_share = ser == "electric_share"
        hfmt = "%{y:.1%}" if is_share else "%{y:,.0f} шт."
        idx = []
        y = res.wide[ser].to_numpy(float)
        fig.add_trace(go.Scatter(x=hist_x, y=y, mode="lines+markers", name="Факт", line=dict(color=NEUTRAL, width=2),
                                 visible=vis, hovertemplate=f"%{{x}}<br>Факт: {hfmt}<extra></extra>"))
        idx.append(len(fig.data) - 1)
        d = res.weighted.draws[ser]
        for lo, hi, a, lab in ((0.025, 0.975, 0.15, "95%"), (0.1, 0.9, 0.25, "80%"), (0.25, 0.75, 0.4, "50%")):
            qlo, qhi = np.quantile(d, lo, 0), np.quantile(d, hi, 0)
            fig.add_trace(go.Scatter(x=fc_x + fc_x[::-1], y=list(qhi) + list(qlo[::-1]), fill="toself",
                                     fillcolor=f"rgba(42,120,214,{a})", line=dict(width=0), name=f"Интервал {lab}",
                                     hoverinfo="skip", visible=vis))
            idx.append(len(fig.data) - 1)
        med = np.median(d, 0)
        fig.add_trace(go.Scatter(x=[hist_x[-1]] + fc_x, y=[y[-1]] + list(med), mode="lines", name="Взвешенный (медиана)",
                                 line=dict(color=BLUE, width=3), visible=vis,
                                 hovertemplate=f"%{{x}}<br>Медиана: {hfmt}<extra></extra>"))
        idx.append(len(fig.data) - 1)
        for sc in res.scenarios:
            m = np.median(res.forecasts[sc.key].draws[ser], 0)
            fig.add_trace(go.Scatter(x=fc_x, y=m, mode="lines", name=sc.title.split("(")[0].strip(),
                                     line=dict(color=sc.color, width=1.6, dash="dash"), visible=vis,
                                     hovertemplate=f"%{{x}}<br>{sc.title.split('(')[0].strip()}: {hfmt}<extra></extra>"))
            idx.append(len(fig.data) - 1)
        groups.append(idx)

    buttons = []
    n = len(fig.data)
    for si, ser in enumerate(series):
        vis = [False] * n
        for i in groups[si]:
            vis[i] = True
        buttons.append(dict(label=_name(res, ser), method="update",
                            args=[{"visible": vis},
                                  {"title.text": f"{_name(res, ser)}: факт и прогноз",
                                   "yaxis.tickformat": ".0%" if ser == "electric_share" else ",.0f"}]))
    fig.update_layout(
        title=dict(text=f"{_name(res, series[0])}: факт и прогноз", x=0.01),
        updatemenus=[dict(buttons=buttons, direction="down", x=0.0, y=1.16, xanchor="left", showactive=True)],
        template="plotly_white", height=620, margin=dict(l=60, r=20, t=110, b=60),
        legend=dict(orientation="h", y=-0.15), hovermode="x unified",
        yaxis=dict(tickformat=",.0f", separatethousands=True),
        font=dict(family="Segoe UI, Roboto, DejaVu Sans, Arial"),
        paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb",
    )
    fig.add_vline(x=len(hist_x) - 0.5, line=dict(color="#8a8985", dash="dot", width=1))

    k = res.tables["scenario_totals"]
    km = k[k.series == "total_market"]
    def f(x):
        return f"{x:,.0f}".replace(",", "\u202f")

    rows = "".join(
        f"<tr><td>{r['title']}</td><td>{r['weight']:.0%}</td><td>{f(r['4q_p50'])}</td>"
        f"<td>{f(r['4q_p10'])}–{f(r['4q_p90'])}</td><td>{f(r['8q_p50'])}</td>"
        f"<td>{f(r['8q_p10'])}–{f(r['8q_p90'])}</td></tr>"
        for _, r in km.iterrows())
    div = fig.to_html(full_html=False, include_plotlyjs=True, config={"displaylogo": False, "locale": "ru"})
    html = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Forkcast — дашборд</title><style>
body{{margin:0;background:#fcfcfb;color:#0b0b0b;font:14px/1.5 "Segoe UI",Roboto,"DejaVu Sans",Arial,sans-serif}}
main{{max-width:1200px;margin:0 auto;padding:20px 16px}} h1{{font-size:24px;margin:4px 0}} .sub{{color:#52514e}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:14px}} th,td{{padding:6px 8px;border-bottom:1px solid #e6e5e1;text-align:right}}
th{{background:#f3f2ee;color:#52514e}} td:first-child,th:first-child{{text-align:left}}
</style></head><body><main>
<h1>Forkcast · рынок вилочных погрузчиков РФ</h1>
<p class="sub">Данные {res.facts['first_period']}–{res.facts['last_period']}; прогноз {fc_x[0]}–{fc_x[-1]}. Выберите ряд в списке над графиком.</p>
{div}
<h3>Рынок за период по сценариям, шт.</h3>
<table><tr><th>Сценарий</th><th>Вес</th><th>4 кв. P50</th><th>4 кв. P10–P90</th><th>8 кв. P50</th><th>8 кв. P10–P90</th></tr>{rows}</table>
<p class="sub">Сергеев Антон Валентинович · avsergeev1981@gmail.com</p>
</main></body></html>"""
    path.write_text(html, encoding="utf-8")
    return path
