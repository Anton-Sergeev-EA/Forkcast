"""Самодостаточный HTML-отчёт (все графики встроены) — открывается в любом браузере без интернета."""

from __future__ import annotations

import base64
import html
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .. import __version__
from .text import LIMITATIONS, fint, fnum, fpct, key_findings, recommendations

CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8985;--line:#e6e5e1;--accent:#1c5cab;--soft:#f3f2ee;}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 "Segoe UI",Roboto,"DejaVu Sans",Arial,sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:30px;line-height:1.2;margin:0 0 6px} h2{font-size:22px;margin:44px 0 10px;padding-top:12px;border-top:1px solid var(--line)}
h3{font-size:17px;margin:24px 0 8px} p{margin:8px 0} .sub{color:var(--ink2)} .muted{color:var(--muted);font-size:13px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:22px 0}
.kpi{background:var(--soft);border-radius:10px;padding:14px 16px}.kpi b{display:block;font-size:24px;color:var(--accent)}
.kpi span{font-size:13px;color:var(--ink2)}
.callout{background:#eef4fc;border-left:4px solid var(--accent);padding:12px 16px;border-radius:6px;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 18px;display:block;overflow-x:auto}
th,td{padding:6px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{background:var(--soft);font-weight:600;color:var(--ink2)} td:first-child,th:first-child{text-align:left}
td.l,th.l{text-align:left;white-space:normal}
img{max-width:100%;height:auto;display:block;margin:12px 0;border-radius:6px}
nav{background:var(--soft);border-radius:10px;padding:12px 18px;margin:18px 0;font-size:14px;columns:2}
nav a{color:var(--accent);text-decoration:none;display:block}
code{background:var(--soft);padding:1px 5px;border-radius:4px;font-size:13px}
pre{background:var(--soft);padding:12px;border-radius:8px;overflow-x:auto;font-size:13px}
ol li,ul li{margin:4px 0} .rec b{color:var(--accent)}
footer{margin-top:50px;color:var(--muted);font-size:12px}
@media print{nav{display:none} h2{page-break-before:always}}
"""


def _img(path: Path, alt: str) -> str:
    if not path or not Path(path).exists():
        return ""
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}">'


def _table(df: pd.DataFrame, fmt: dict[str, str] | None = None, left: tuple[str, ...] = ()) -> str:
    fmt = fmt or {}
    head = "".join(f'<th class="{"l" if c in left else ""}">{html.escape(str(c))}</th>' for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            f = fmt.get(c)
            if isinstance(v, (float, np.floating)) and np.isnan(v):
                s = "—"
            elif f == "int":
                s = fint(v)
            elif f == "pct":
                s = fpct(v, 0)
            elif f == "pct1":
                s = fpct(v, 1)
            elif f == "pcts":
                s = fpct(v, 0, True)
            elif f and f.startswith("f"):
                s = fnum(v, int(f[1:]))
            else:
                s = html.escape(str(v))
            cells.append(f'<td class="{"l" if c in left else ""}">{s}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build_html_report(res, figs: dict[str, Path], path: Path) -> Path:
    k = key_findings(res)
    t = res.tables
    recs = recommendations(res)
    y0, y1 = k["y0"], k["y1"]

    # --- таблицы ---------------------------------------------------------------------------------
    ann = t["annual"].copy()
    ann["Показатель"] = [res.label(c) if not c.startswith(("ice_", "el_")) else
                         ("ДВС " if c.startswith("ice") else "Электро ") + res.labels.get(c, c) for c in ann["series"]]
    yrs = [c for c in ann.columns if c.isdigit()]
    dcols = [c for c in ann.columns if c.startswith("Δ")]
    ann_t = ann[["Показатель", *yrs, *dcols, "тренд"]]
    ann_t = ann_t.rename(columns={"тренд": "Тренд (Манн–Кендалл, 5%)"})

    sea = t["seasonality"].copy()
    sea["Ряд"] = sea["series"].map(res.label)
    sea_t = sea[["Ряд", "Q1", "Q2", "Q3", "Q4", "F", "p_value", "сезонность"]].rename(columns={"p_value": "p-value"})

    cm = t["coef_market"].copy()
    nice = {"const": "Константа", "q2": "2 квартал", "q3": "3 квартал", "q4": "4 квартал",
            "key_rate_eff": "Эффективная ставка (п.п.)", "log_investment": "log индекса инвестиций",
            "regime": "Режим с 2025Q2", "trend": "Тренд (квартал)"}
    cm["коэффициент"] = cm["коэффициент"].map(lambda x: nice.get(x, x))
    cs = t["coef_share"].copy()
    cs["коэффициент"] = cs["коэффициент"].map(lambda x: nice.get(x, x))

    st = t["scenario_totals"]
    stm = st[st.series == "total_market"][["title", "weight", "4q_p10", "4q_p50", "4q_p90", "8q_p10", "8q_p50", "8q_p90"]]
    stm = stm.rename(columns={"title": "Сценарий", "weight": "Вес", "4q_p10": "4 кв. P10", "4q_p50": "4 кв. P50",
                              "4q_p90": "4 кв. P90", "8q_p10": "8 кв. P10", "8q_p50": "8 кв. P50", "8q_p90": "8 кв. P90"})
    sts = st[st.series == "electric_share"][["title", "4q_p10", "4q_p50", "4q_p90", "8q_p10", "8q_p50", "8q_p90"]]
    sts = sts.rename(columns={"title": "Сценарий", "4q_p10": f"{k['fc_4q_end']} P10", "4q_p50": f"{k['fc_4q_end']} P50",
                              "4q_p90": f"{k['fc_4q_end']} P90", "8q_p10": f"{k['fc_8q_end']} P10",
                              "8q_p50": f"{k['fc_8q_end']} P50", "8q_p90": f"{k['fc_8q_end']} P90"})

    # квартальная таблица взвешенного прогноза
    ws = res.weighted.summary()
    rows = []
    for ser in ["total_market", "total_ice", "total_electric", "electric_share"]:
        sub = ws[ws.series == ser]
        for _, r in sub.iterrows():
            rows.append({"Показатель": res.label(ser), "Квартал": str(r["period"]), "P10": r["q100"], "P50": r["q500"],
                         "P90": r["q900"], "is_share": ser == "electric_share"})
    q = pd.DataFrame(rows)
    q_units = q[~q.is_share].pivot(index="Показатель", columns="Квартал", values="P50").reindex(
        ["Рынок, всего", "ДВС, всего", "Электро, всего"]).reset_index()
    q_band = q[q["Показатель"] == "Рынок, всего"][["Квартал", "P10", "P50", "P90"]]
    q_share = q[q.is_share][["Квартал", "P10", "P50", "P90"]]

    # сегменты: медиана 4 и 8 кварталов
    seg_rows = []
    for c in [c for c in res.wide.columns if c.startswith(("ice_", "el_"))]:
        d = res.weighted.draws[c]
        s4, s8 = d[:, :4].sum(1), d.sum(1)
        hist_y1 = res.tables["annual"].set_index("series").loc[c, y1]
        seg_rows.append({"Сегмент": ("ДВС " if c.startswith("ice") else "Электро ") + res.labels.get(c, c),
                         f"Факт {y1}": hist_y1, "4 кв. P10": np.quantile(s4, .1), "4 кв. P50": np.median(s4),
                         "4 кв. P90": np.quantile(s4, .9), "Δ к " + y1: np.median(s4) / hist_y1 - 1 if hist_y1 else np.nan,
                         "8 кв. P50": np.median(s8)})
    seg_t = pd.DataFrame(seg_rows)

    sc_rows = []
    for sc in res.scenarios:
        p = sc.path
        sc_rows.append({"Сценарий": sc.title, "Вес": sc.weight,
                        "Ставка 4кв26": p["key_rate"].iloc[2], "Ставка 4кв27": p["key_rate"].iloc[6],
                        "Инвест. 2027 ср., % г/г": p["investment_yoy"].iloc[3:7].mean(),
                        "USD/RUB 4кв27": p["usd_rub"].iloc[6], "Допущения": sc.assumptions})
    sc_t = pd.DataFrame(sc_rows)

    bs = t["backtest_scores"].copy()
    bs["series"] = bs["series"].map(res.label)
    bs = bs.rename(columns={"series": "Ряд", "model": "Модель", "cover80": "Покрытие 80%", "cover95": "Покрытие 95%",
                            "bias_%": "Смещение"})

    plan = t["plan_short"]
    plan_sum = plan[plan["period"].astype(str).str.startswith("Σ")].copy()
    plan_sum["Сегмент"] = plan_sum.apply(lambda r: ("ДВС " if r["block"] == "ice" else "Электро ") + r["label"], axis=1)
    plan_sum = plan_sum[["Сегмент", "market_p10", "market_p50", "market_p90", "silant_share", "silant_demand_p50",
                         "plan_newsvendor"]].rename(columns={
        "market_p10": "Рынок P10", "market_p50": "Рынок P50", "market_p90": "Рынок P90",
        "silant_share": "Доля СИЛАНТ", "silant_demand_p50": "Спрос СИЛАНТ P50", "plan_newsvendor": "Рекомендуемый выпуск"})

    ladder = t["key_rate_ladder"]
    lcol = [c for c in ladder.columns if c.endswith(("_median", "_p10", "_p90"))]
    lad_t = ladder[["key_rate", *lcol]].rename(columns={"key_rate": "Ключевая ставка, %"})

    dq = t["data_quality"]
    ll = t["factor_longlist"][["factor", "sign", "rationale", "data", "status"]].rename(columns={
        "factor": "Фактор", "sign": "Ожид. знак", "rationale": "Экономический механизм", "data": "Данные", "status": "Решение"})
    spec = t["spec_comparison"].rename(columns={"спецификация": "Спецификация"})
    scr = t["factor_screening"].copy()
    scr_names = {"key_rate": "Ключевая ставка", "real_key_rate": "Реальная ставка", "investment_yoy": "Инвестиции, % г/г",
                 "usd_rub": "USD/RUB", "gdp_yoy": "ВВП, % г/г"}
    scr["factor"] = scr["factor"].map(lambda x: scr_names.get(x, x))
    ub = t["uncertainty_budget"].copy()
    ub["period"] = ub["period"].astype(str)
    ub = ub.rename(columns={"period": "Квартал", "sd_log_total": "σ log(рынок)"})

    ps = t.get("prior_sensitivity")
    cap = t.get("share_cap_sensitivity")

    body = f"""
<h1>Рынок вилочных погрузчиков РФ: анализ и сценарный прогноз по сегментам</h1>
<p class="sub">Проект <b>Forkcast</b> для ЧЗСА (бренд СИЛАНТ) · данные {res.facts['first_period']}–{k['last_period']} ·
прогноз {k['fc_start']}–{k['fc_8q_end']} · сформировано {date.today():%d.%m.%Y}</p>
<p class="muted">Автор: Сергеев Антон Валентинович · avsergeev1981@gmail.com · версия {__version__}</p>

<div class="kpis">
<div class="kpi"><b>{k['market_y1']}</b><span>рынок {y1} г., шт. ({k['market_d']} к {y0})</span></div>
<div class="kpi"><b>{k['w4_p50']}</b><span>прогноз {k['fc_start']}–{k['fc_4q_end']}, шт. (P10–P90: {k['w4_p10']}–{k['w4_p90']})</span></div>
<div class="kpi"><b>{k['share_8q']}</b><span>доля электро к {k['fc_8q_end']} (сейчас {k['last_share']})</span></div>
<div class="kpi"><b>{k['beta_kr']}</b><span>изменение продаж на +1 п.п. ключевой ставки (лаг 1–3 кв.)</span></div>
</div>

<nav>
<a href="#s1">1. Главное для руководства</a><a href="#s2">2. Данные и их качество</a><a href="#s3">3. Динамика и тренды</a>
<a href="#s4">4. Сезонность</a><a href="#s5">5. Макрофакторы и их влияние</a><a href="#s6">6. Методика прогноза</a>
<a href="#s7">7. Прогноз рынка и сегментов</a><a href="#s8">8. Доля электропогрузчиков</a><a href="#s9">9. Сценарии и допущения</a>
<a href="#s10">10. Неопределённость и точность</a><a href="#s11">11. Выводы для планирования</a><a href="#s12">12. Ограничения</a>
<a href="#s13">13. Обновление прогноза</a><a href="#s14">14. Источники</a>
</nav>

<h2 id="s1">1. Главное для руководства</h2>
<ol>
<li><b>Рынок перешёл в фазу сжатия.</b> В {y1} г. продано {k['market_y1']} погрузчиков ({k['market_d']} к {y0} г.);
ДВС {k['ice_d']}, электро {k['el_d']}. В {k['last_period']} — {k['last_total']} шт. ({k['last_yoy']} г/г), минимум за всю историю наблюдений.</li>
<li><b>Причины количественно разложены:</b> из {k['dec_total']} лог-пунктов падения {y1} г. {k['dec_kr']} объясняет ключевая ставка,
{k['dec_reg']} — структурный сдвиг со 2 кв. 2025 г. (завершение цикла замещения парка после ухода западных брендов, утильсбор,
неценовые условия кредитования), {k['dec_inv']} — инвестиции в основной капитал.</li>
<li><b>Дно пройдено, но восстановление будет медленным.</b> Взвешенный по сценариям прогноз на 4 квартала
({k['fc_start']}–{k['fc_4q_end']}) — {k['w4_p50']} шт. ({k['w4_vs_y1']} к {y1} г.), 80%-интервал {k['w4_p10']}–{k['w4_p90']};
на 8 кварталов — {k['w8_p50']} шт. ({k['w8_p10']}–{k['w8_p90']}).</li>
<li><b>Электрификация необратима:</b> доля электро выросла с {k['share_2023']} (2023) до {k['share_y1']} ({y1}),
прогноз — {k['share_4q']} через 4 квартала и {k['share_8q']} к {k['fc_8q_end']}.</li>
<li><b>Сценарии расходятся во втором году:</b> на горизонте 4 кв. разброс медиан сценариев {k['sc4_min']}–{k['sc4_max']} шт.
(ближайшие кварталы предопределены уже состоявшимися решениями ЦБ), на 8 кв. — {k['sc8_min']}–{k['sc8_max']} шт.</li>
<li><b>Модель проверена на истории:</b> в бэктесте ошибка MASE {k['bt_mase']} против {k['bt_mase_sn']} у сезонного наивного
прогноза; фактические значения попали в 80%-интервал в {k['bt_cov80']} случаев.</li>
</ol>

<h2 id="s2">2. Данные и их качество</h2>
<p>Рыночные данные: 14 рядов (8 сегментов ДВС и 6 электро) × {res.facts['n_quarters']} кварталов, шт. Внешние факторы —
Банк России, Росстат, Минэкономразвития (дата выгрузки и статус проверки — в <code>data/external/sources.yaml</code>).
Итоговый датасет: <code>data/processed/model_dataset.csv</code>, лист «Датасет_модели» в Excel.</p>
{_table(dq, left=("проверка", "детали"))}

<h2 id="s3">3. Динамика рынка и тренды</h2>
{_img(figs.get('market_history'), 'История рынка')}
{_table(ann_t, {**{c: 'int' for c in yrs}, **{c: 'pcts' for c in dcols}}, left=("Показатель", "Тренд (Манн–Кендалл, 5%)"))}
<p class="muted">Тест Манна–Кендалла применён к скользящей годовой сумме (устраняет сезонность). Для малых сегментов
(средний объём &lt; 100 шт./кв.) темпы роста неинформативны из-за округления данных до 10 шт.</p>
{_img(figs.get('segment_heatmap'), 'Изменение по сегментам')}
<div class="callout">Электро 3–3,5 т и 3,8–5 т — единственные крупные сегменты с устойчивым ростом: спрос смещается к более
тяжёлым электропогрузчикам, вытесняющим ДВС в классе 3–5 т. Все сегменты ДВС 1–5 т сократились в {y1} г. примерно вдвое.</div>
<h3>Структура сегментов внутри блоков (доля в блоке по годам)</h3>
{_table(t['structure'].drop(columns=['segment']), {c: 'pct1' for c in t['structure'].columns if c.isdigit()}, left=("block", "label"))}

<h2 id="s4">4. Сезонность</h2>
{_img(figs.get('seasonality'), 'Сезонность')}
{_table(sea_t, {"Q1": "pcts", "Q2": "pcts", "Q3": "pcts", "Q4": "pcts", "F": "f2", "p-value": "f3"})}
<p>Сезонность рынка значима (F-тест, p={k['sea_p']}): 4 квартал {k['q4_eff']} к среднегодовому уровню
(закрытие бюджетов, закупки до индексации утильсбора с 1 января), 1 квартал {k['q1_eff']} (зима, новые бюджеты).
Сезонность доли электро не выявлена — структурный сдвиг идёт равномерно. Сезонные эффекты оцениваются моделью
совместно с факторами и включаются в прогноз.</p>

<h2 id="s5">5. Макроэкономические факторы и их влияние</h2>
<h3>5.1. Длинный список факторов и решения по отбору</h3>
{_table(ll, left=tuple(ll.columns))}
<h3>5.2. Факторы в динамике</h3>
{_img(figs.get('factors'), 'Факторы')}
<h3>5.3. Скрининг: корреляция сезонно очищенного рынка с факторами при разных лагах</h3>
{_table(scr, {c: 'f2' for c in scr.columns if c != 'factor'})}
<p>Связь со ставкой максимальна при лаге 2–3 квартала (r ≈ −0,76…−0,78) и слабая без лага — это механизм трансмиссии
ДКП: решение о закупке → кредит/лизинг → поставка. Поэтому в модели используется <b>эффективная ставка</b> — среднее
за кварталы t−1…t−3.</p>
<h3>5.4. Отбор спецификации по кросс-валидации</h3>
{_img(figs.get('spec'), 'Отбор спецификации')}
{_table(spec, {"R2": "f3", "AIC": "f1", "LOO_RMSE": "f3"}, left=("Спецификация",))}
<p>Итоговая спецификация (ставка + инвестиции + структурный сдвиг) даёт минимальную ошибку «исключи-одну-точку»
(LOO). Курс и ВВП не улучшают качество вне выборки — отклонены, хотя и остаются в сценариях как справочные параметры.</p>
<h3>5.5. Оценённое влияние факторов (апостериорные распределения)</h3>
{_table(cm, {c: 'f3' for c in cm.columns if c != 'коэффициент'}, left=("коэффициент",))}
<div class="callout"><b>Интерпретация:</b> +1 п.п. эффективной ключевой ставки → {k['beta_kr']} продаж
(90%-интервал {k['beta_kr_hi']}…{k['beta_kr_lo']}); +1% уровня инвестиций в ОК → +{k['beta_inv']}% продаж
(спрос на погрузчики — «усиленная» производная от инвестиций); структурный сдвиг 2025Q2 снизил уровень рынка на {k['beta_regime']}.
Столбец «информативность_данных» показывает, насколько данные сузили априорную неопределённость.</div>
<h3>5.6. Декомпозиция: что обрушило рынок в {y1} г.</h3>
{_img(figs.get('decomposition'), 'Декомпозиция')}
{_table(t['decomposition'].drop(columns=['Δlog_факт']), {c: 'f3' for c in t['decomposition'].columns if c not in ('from', 'to', 'изменение_%')} | {"изменение_%": "pcts"})}

<h2 id="s6">6. Методика прогноза</h2>
<p><b>Иерархическая байесовская факторная модель</b> — три уровня, согласованных по построению:</p>
<ol>
<li><b>Рынок в целом:</b> <code>log M<sub>t</sub> = α + Σγ<sub>q</sub>D<sub>q</sub> + β<sub>r</sub>·R<sup>eff</sup><sub>t</sub> + β<sub>i</sub>·log I<sub>t</sub> + β<sub>s</sub>·S<sub>t</sub> + ε<sub>t</sub></code>,
где R<sup>eff</sup> — средняя ставка за t−1…t−3, I — индекс уровня инвестиций в ОК (цепной из темпов г/г), S — режим с 2025Q2.</li>
<li><b>Доля электро:</b> логистическая S-кривая с потолком насыщения {int(res.model.cap * 100)}%:
<code>logit(s<sub>t</sub>/cap) = a + Σc<sub>q</sub>D<sub>q</sub> + b·t + δ·R<sup>eff</sup><sub>t</sub> + κ·S<sub>t</sub> + u<sub>t</sub></code>.</li>
<li><b>Структура сегментов</b> внутри ДВС и электро — композиционная модель в аддитивных лог-отношениях
(экспоненциальное сглаживание уровня, затухающий тренд). Сумма сегментов = блок, блоки = рынок.</li>
</ol>
<p><b>Почему байесовский подход.</b> 13 наблюдений недостаточно для устойчивого МНК с сезонностью и тремя факторами.
Слабоинформативные экономически обоснованные априорные распределения стабилизируют оценки, а апостериорные
выборки (сэмплер Гиббса) переносят неопределённость параметров в прогноз. Устойчивость к выбору априори проверена ниже.</p>
<p><b>Сценарии → прогноз.</b> Для каждого сценария симулируется {res.manifest['n_draws']} траекторий: путь факторов
(с неопределённостью внутри сценария, растущей с горизонтом) → параметры модели из апостериорного распределения → шум.
Уже известные факты (ставка за {res.macro['key_rate'].dropna().index[-1]}, инвестиции за {res.macro['investment_yoy'].dropna().index[-1]})
используются вместо сценарных значений — режим <i>наукаста</i>. Взвешенный прогноз — смесь сценариев с весами из конфигурации.</p>
<p>Горизонты: краткосрочный — 4 квартала ({k['fc_start']}–{k['fc_4q_end']}), долгосрочный — 8 кварталов (до {k['fc_8q_end']}).
Качество модели доли: R² = {fnum(res.facts['r2_share'], 2)}; модели рынка: R² = {k['r2']}.</p>
{_table(cs, {c: 'f3' for c in cs.columns if c != 'коэффициент'}, left=("коэффициент",))}

<h2 id="s7">7. Прогноз рынка и сегментов</h2>
{_img(figs.get('fan_market'), 'Прогноз рынка')}
<h3>Взвешенный прогноз по кварталам, медиана (шт.)</h3>
{_table(q_units, {c: 'int' for c in q_units.columns if c != 'Показатель'})}
<h3>Рынок: интервалы по кварталам</h3>
{_table(q_band, {"P10": "int", "P50": "int", "P90": "int"})}
{_img(figs.get('blocks'), 'ДВС и электро')}
{_img(figs.get('scenarios'), 'Сценарии')}
{_table(stm, {"Вес": "pct", **{c: "int" for c in stm.columns if "кв." in c}}, left=("Сценарий",))}
<h3>Прогноз по сегментам (взвешенный сценарий)</h3>
{_img(figs.get('silant'), 'Сегменты СИЛАНТ')}
{_table(seg_t, {c: 'int' for c in seg_t.columns if c not in ('Сегмент',) and not c.startswith('Δ')} | {c: 'pcts' for c in seg_t.columns if c.startswith('Δ')}, left=("Сегмент",))}

<h2 id="s8">8. Доля электропогрузчиков</h2>
{_img(figs.get('fan_share'), 'Доля электро')}
{_table(q_share, {"P10": "pct1", "P50": "pct1", "P90": "pct1"})}
{_table(sts, {c: 'pct1' for c in sts.columns if c != 'Сценарий'}, left=("Сценарий",))}
<p>Доля электро растёт в логит-пространстве на ~{fnum(res.facts['share_trend'], 3)} за квартал (S-кривая замещения)
и слабо зависит от сценария: электрификация — структурный, а не циклический процесс. Чувствительность к потолку насыщения:</p>
{_table(cap, {c: 'pct1' for c in cap.columns}) if cap is not None else ''}

<h2 id="s9">9. Сценарии и допущения</h2>
{_table(sc_t, {"Вес": "pct", "Ставка 4кв26": "f2", "Ставка 4кв27": "f2", "Инвест. 2027 ср., % г/г": "f1", "USD/RUB 4кв27": "f1"}, left=("Сценарий", "Допущения"))}
<p>Полные квартальные пути факторов — <code>config/scenarios.yaml</code> и лист «Сценарии_допущения» в Excel.
Веса сценариев отражают экспертную оценку вероятности и меняются в конфигурации без правки кода.</p>

<h2 id="s10">10. Неопределённость и точность</h2>
{_img(figs.get('uncertainty'), 'Бюджет неопределённости')}
{_table(ub, {"параметры модели": "pct", "случайный шум": "pct", "путь внешних факторов": "pct", "σ log(рынок)": "f3"})}
<p>Вблизи ближайших кварталов доминирует собственный шум рынка; к концу горизонта растёт вклад неопределённости
внешних факторов — именно поэтому долгосрочный прогноз подаётся через сценарии.</p>
<h3>Ретроспективная проверка (rolling-origin бэктест)</h3>
{_img(figs.get('backtest'), 'Бэктест')}
{_table(bs, {"MAPE": "pct", "MASE": "f2", "Смещение": "pcts", "Покрытие 80%": "pct", "Покрытие 95%": "pct"}, left=("Ряд", "Модель"))}
<p class="muted">{k['bt_n']} прогнозов с горизонтом 1–4 кв. от точек отсечения 2024Q4–{res.wide.index[-2]}; факторы —
фактические (условный прогноз). MASE &lt; 1 — лучше сезонного наивного прогноза.</p>
{('<h3>Устойчивость к априорным распределениям</h3>' + _table(ps, {c: 'f3' for c in ps.columns if c.startswith('β')} | {c: 'int' for c in ps.columns if 'рынок' in c} | {'prior_sd_scale': 'f1'})) if ps is not None else ''}

<h2 id="s11">11. Выводы для производственного и продуктового планирования</h2>
<ol class="rec">{''.join(f'<li><b>{html.escape(h)}.</b> {html.escape(b)}</li>' for h, b in recs)}</ol>
<h3>Коридор плана по сегментам СИЛАНТ на 4 квартала</h3>
{_table(plan_sum, {c: 'int' for c in plan_sum.columns if c not in ('Сегмент', 'Доля СИЛАНТ')} | {'Доля СИЛАНТ': 'pct'}, left=("Сегмент",))}
<p class="muted">Целевая доля СИЛАНТ (по умолчанию 5%) и экономика недо/перепроизводства задаются в
<code>config/segments.yaml</code>. Рекомендуемый выпуск — квантиль распределения спроса, минимизирующий ожидаемые потери.</p>
<h3>«Лестница ставок»</h3>
{_img(figs.get('ladder'), 'Лестница ставок')}
{_table(lad_t, {c: 'int' for c in lad_t.columns if c != 'Ключевая ставка, %'})}

<h2 id="s12">12. Ограничения</h2>
<ul>{''.join(f'<li>{html.escape(x)}</li>' for x in LIMITATIONS)}</ul>

<h2 id="s13">13. Обновление прогноза при поступлении новых данных</h2>
<pre>forkcast update --market путь/к/новому_файлу.xlsx   # проверка, пересчёт, винтаж, оценка точности
forkcast fetch-macro                                  # ключевая ставка и курс с cbr.ru
forkcast run                                          # полный пересчёт по текущей конфигурации</pre>
<p>Парсер не зависит от адресов ячеек: новые кварталы (столбцы) и новые сегменты распознаются автоматически.
Каждый прогноз архивируется в <code>forecasts/vintages/</code>; при поступлении факта рассчитываются ошибка, попадание
в интервалы и PIT — это мониторинг калибровки модели.</p>

<h2 id="s14">14. Источники внешних данных</h2>
<ul>
<li>Банк России: ключевая ставка, пресс-релизы 24.07.2026 и 11.09.2026, среднесрочный прогноз, макроэкономический опрос (сент. 2026) — cbr.ru</li>
<li>Росстат: ВВП, инвестиции в основной капитал, ИПЦ — rosstat.gov.ru (через публикации Интерфакс/Ведомости с указанием в sources.yaml)</li>
<li>Минэкономразвития: сценарные условия на 2027–2029 гг. (май 2026) — economy.gov.ru, interfax.ru</li>
<li>Курс USD/RUB: официальные курсы ЦБ (среднемесячные) — cbr.ru, mainfin.ru</li>
<li>Утильсбор: ПП РФ № 81 от 06.02.2016 (в ред.), отраслевые публикации</li>
</ul>
<p class="muted">Полный перечень URL, дата выгрузки (2026-09-23) и статус проверки каждого показателя — в <code>data/external/sources.yaml</code>
и на листе «Источники» Excel-файла.</p>
<footer>Forkcast v{__version__} · seed {res.manifest['seed']} · SHA-256 файла рынка {res.manifest['market_sha256'][:16]}… ·
© {date.today().year} Сергеев Антон Валентинович</footer>
"""
    doc = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Forkcast — рынок вилочных погрузчиков РФ</title><style>{CSS}</style></head>
<body><main>{body}</main></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path
