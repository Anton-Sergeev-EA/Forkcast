"""Формулировки выводов: числа берутся из результатов расчёта, поэтому текст отчёта
автоматически обновляется при поступлении новых данных."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fint(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.0f}".replace(",", " ")


def fth(x: float, d: int = 1) -> str:
    return f"{x / 1000:.{d}f}".replace(".", ",") + " тыс."


def fpct(x: float, d: int = 0, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{x * 100:+.{d}f}" if sign else f"{x * 100:.{d}f}"
    return s.replace(".", ",").replace("-", "−") + "%"


def fnum(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


def key_findings(res) -> dict[str, str]:
    """Набор ключевых чисел и формулировок, используемых в отчёте, презентации и саммари."""
    f = res.facts
    t = res.tables
    w = res.wide
    ann = t["annual"].set_index("series")
    y1 = f["last_full_year"]
    y0 = y1 - 1
    st = t["scenario_totals"]
    wt = st[(st.scenario == "weighted")].set_index("series")
    share_row = wt.loc["electric_share"]
    sea = t["seasonality"].set_index("series").loc["total_market"]
    dec = t.get("decomposition_annual")
    dec = dec.iloc[0] if dec is not None and len(dec) else None
    bs = t["backtest_scores"]
    fk = bs[(bs.series == "total_market") & (bs.model == "Forkcast")].iloc[0]
    sn = bs[(bs.series == "total_market") & (bs.model == "SNaive")].iloc[0]
    ladder = t["key_rate_ladder"]
    ycol = [c for c in ladder.columns if c.endswith("_median")]
    ladder_year = ycol[0].split("_")[0] if ycol else None
    breakeven = None
    if ycol:
        ok = ladder[ladder[ycol[0]] >= f["last_full_year_total"]]
        breakeven = float(ok["key_rate"].max()) if len(ok) else None
    sc_rows = st[(st.series == "total_market") & (st.scenario != "weighted")]
    fp = f["fc_periods"]

    out = {
        "y0": str(y0), "y1": str(y1),
        "market_y0": fint(ann.loc["total_market", str(y0)]), "market_y1": fint(ann.loc["total_market", str(y1)]),
        "market_d": fpct(ann.loc["total_market", f"Δ{y1}/{y0}"], 0, True),
        "ice_d": fpct(ann.loc["total_ice", f"Δ{y1}/{y0}"], 0, True),
        "el_d": fpct(ann.loc["total_electric", f"Δ{y1}/{y0}"], 0, True),
        "last_period": f["last_period"], "last_total": fint(f["last_total"]), "last_yoy": fpct(f["last_yoy"], 0, True),
        "last_share": fpct(f["last_share"]),
        "share_2023": fpct(w.loc[[p for p in w.index if p.year == w.index[0].year], "total_electric"].sum()
                           / w.loc[[p for p in w.index if p.year == w.index[0].year], "total_market"].sum()),
        "share_y1": fpct(ann.loc["total_electric", str(y1)] / ann.loc["total_market", str(y1)]),
        "q1_eff": fpct(sea["Q1"], 0, True), "q4_eff": fpct(sea["Q4"], 0, True), "sea_p": fnum(sea["p_value"], 3),
        "beta_kr": fpct(f["beta_kr"], 1, True), "beta_kr_lo": fpct(f["beta_kr_q05"], 1, True),
        "beta_kr_hi": fpct(f["beta_kr_q95"], 1, True), "beta_inv": fnum(f["beta_inv"], 2),
        "beta_regime": fpct(np.exp(f["beta_regime"]) - 1, 0, True) if f.get("beta_regime") is not None else "—",
        "dec_kr": fnum(dec["key_rate_eff"] * 100, 1) if dec is not None else "—",
        "dec_inv": fnum(dec["log_investment"] * 100, 1) if dec is not None else "—",
        "dec_reg": fnum(dec["regime"] * 100, 1) if dec is not None and "regime" in dec else "—",
        "dec_total": fnum(dec["итого_log"] * 100, 1) if dec is not None else "—",
        "fc_start": fp[0], "fc_4q_end": fp[3], "fc_8q_end": fp[-1],
        "w4_p50": fint(wt.loc["total_market", "4q_p50"]), "w4_p10": fint(wt.loc["total_market", "4q_p10"]),
        "w4_p90": fint(wt.loc["total_market", "4q_p90"]),
        "w8_p50": fint(wt.loc["total_market", "8q_p50"]), "w8_p10": fint(wt.loc["total_market", "8q_p10"]),
        "w8_p90": fint(wt.loc["total_market", "8q_p90"]),
        "w4_ice": fint(wt.loc["total_ice", "4q_p50"]), "w4_el": fint(wt.loc["total_electric", "4q_p50"]),
        "w8_ice": fint(wt.loc["total_ice", "8q_p50"]), "w8_el": fint(wt.loc["total_electric", "8q_p50"]),
        "w4_vs_y1": fpct(wt.loc["total_market", "4q_p50"] / f["last_full_year_total"] - 1, 0, True),
        "share_4q": fpct(share_row["4q_p50"]), "share_8q": fpct(share_row["8q_p50"]),
        "share_8q_lo": fpct(share_row["8q_p10"]), "share_8q_hi": fpct(share_row["8q_p90"]),
        "sc4_min": fint(sc_rows["4q_p50"].min()), "sc4_max": fint(sc_rows["4q_p50"].max()),
        "sc8_min": fint(sc_rows["8q_p50"].min()), "sc8_max": fint(sc_rows["8q_p50"].max()),
        "bt_mase": fnum(fk["MASE"], 2), "bt_mase_sn": fnum(sn["MASE"], 2), "bt_mape": fpct(fk["MAPE"], 0),
        "bt_cov80": fpct(fk.get("cover80", np.nan), 0), "bt_n": str(int(fk["n"])),
        "ladder_year": ladder_year or "—", "breakeven": fnum(breakeven, 0) if breakeven else "не достигается в диапазоне 8–16%",
        "r2": fnum(f["r2_market"], 2),
    }
    return out


def recommendations(res) -> list[tuple[str, str]]:
    k = key_findings(res)
    ann = res.tables["annual"].set_index("series")
    y1 = k["y1"]
    growing = ann[(ann.index.str.startswith("el_")) & (ann["тренд"] == "↑ рост") & (~ann["малый_объём"])]
    grow_txt = ", ".join(f"электро {r['label']}" for _, r in growing.iterrows()) or "—"
    plan = res.tables["plan_short"]
    tot = plan[plan["period"].astype(str).str.startswith("Σ")]
    q_star = float(tot["critical_ratio"].iloc[0]) if len(tot) else 0.7
    recs = [
        ("План производства на 4 квартала",
         f"Базировать план на медиане взвешенного прогноза рынка: {k['w4_p50']} шт. за {k['fc_start']}–{k['fc_4q_end']} "
         f"({k['w4_vs_y1']} к {y1} г.), держать «коридор» мощностей P10–P90: {k['w4_p10']}–{k['w4_p90']} шт. "
         f"Объём выпуска по сегментам — по правилу newsvendor (квантиль {q_star:.0%}), см. лист «План_СИЛАНТ_4кв»."),
        ("Продуктовый микс: приоритет электро",
         f"Доля электропогрузчиков выросла с {k['share_2023']} (2023) до {k['share_y1']} ({y1}) и, по прогнозу, достигнет "
         f"{k['share_8q']} к {k['fc_8q_end']} (80%-интервал {k['share_8q_lo']}–{k['share_8q_hi']}). "
         f"Устойчиво растущие сегменты: {grow_txt}. Их стоит ставить в приоритет модельного ряда и локализации."),
        ("ДВС: гибкость вместо расширения",
         f"Сегмент ДВС сократился на {k['ice_d'].lstrip('−+')} в {y1} г. и восстанавливается медленнее рынка "
         f"(прогноз {k['w4_ice']} шт. за 4 кв.). Рекомендуется сборка под заказ, унификация платформ 1,5–3,5 т и "
         f"минимальный страховой запас; расширение мощностей ДВС — только при переходе рынка к оптимистичному сценарию."),
        ("Сезонное выравнивание загрузки",
         f"Сезонность значима: 4 квартал {k['q4_eff']} к среднегодовому уровню, 1 квартал {k['q1_eff']}. "
         "Выпуск в 3 квартале стоит наращивать под пик 4 квартала, а в 1 квартале смещать ресурсы на сервис, "
         "ремонт и подготовку новых моделей."),
        ("Триггеры пересмотра плана",
         f"Главный рычаг — ключевая ставка (эластичность {k['beta_kr']} продаж на 1 п.п. с лагом 1–3 кв.). "
         f"Рынок {k['ladder_year']} г. возвращается к уровню {y1} г. при ставке не выше {k['breakeven']}%. "
         "Решения ЦБ (следующее — 23.10.2026) и квартальные данные Росстата по инвестициям — сигналы для пересчёта."),
        ("Регламент обновления",
         "Ежеквартально: загрузить новый файл рынка командой `forkcast update`, при необходимости обновить "
         "config/scenarios.yaml по свежему макроопросу ЦБ; система пересчитает прогноз, сохранит винтаж и покажет "
         "точность прошлых прогнозов (PIT, попадание в интервалы)."),
    ]
    return recs


LIMITATIONS = [
    "Короткая история: 13 квартальных точек. Оценки эластичностей стабилизированы априорными распределениями; "
    "интервалы прогноза намеренно широкие — это честная мера неопределённости, а не недостаток модели.",
    "В данных нет цен, брендов, регионов и каналов продаж — нельзя оценить ценовую эластичность, долю импорта, "
    "конкуренцию и региональную структуру. Для СИЛАНТ-специфичного плана целевая доля задаётся экспертно (config/segments.yaml).",
    "В описании кейса указано 9 сегментов ДВС, в файле их 8 (до 20–25 т). Прогноз построен по фактической структуре.",
    "Данные округлены до 10 шт.: для малых сегментов (6–25 т ДВС, 5,5–16 т электро) шум округления сопоставим с сигналом; "
    "их прогноз — через структуру блока, интервалы относительно шире.",
    "Структурный сдвиг 2 кв. 2025 г. (завершение цикла замещения парка, утильсбор, неценовые условия кредитования) "
    "идентифицирован как событийная переменная: разделить эти причины на имеющихся данных нельзя.",
    "Бэктест — условный (ex-post): используются фактические значения факторов. Архивных прогнозов факторов «на дату» "
    "в открытом доступе нет; неопределённость будущих факторов покрывается сценариями.",
    "Часть квартальных макроданных взята из вторичных источников (СМИ со ссылкой на Росстат) и подлежит сверке "
    "при ревизиях Росстата — статусы указаны в data/external/sources.yaml.",
    "Не учтены факторы без открытых квартальных рядов: ввод складских площадей, объёмы лизинга спецтехники, "
    "импорт погрузчиков по данным таможни КНР. Их подключение — первый шаг развития модели.",
]
