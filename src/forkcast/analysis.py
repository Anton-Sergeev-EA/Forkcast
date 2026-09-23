"""Аналитика: тренды, сезонность, отбор и влияние макрофакторов, декомпозиция, чувствительность."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .macro import build_features
from .model import ForkcastModel, seasonal_dummies

# ----------------------------------------------------------------------------------------------
# Тренды
# ----------------------------------------------------------------------------------------------


def mann_kendall(x: np.ndarray) -> tuple[float, float]:
    """Тест Манна–Кендалла на монотонный тренд: (статистика Z, p-value)."""
    x = np.asarray(x, float)
    n = len(x)
    s = sum(np.sign(x[j] - x[i]) for i in range(n - 1) for j in range(i + 1, n))
    var = n * (n - 1) * (2 * n + 5) / 18
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def sen_slope(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    sl = [(x[j] - x[i]) / (j - i) for i in range(len(x) - 1) for j in range(i + 1, len(x))]
    return float(np.median(sl))


def annual_table(wide: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    """Годовые объёмы, темпы роста, доля в блоке, тест на тренд по скользящей годовой сумме."""
    cols = [c for c in wide.columns if c != "electric_share"]
    years = sorted({p.year for p in wide.index if sum(q.year == p.year for q in wide.index) == 4})
    rows = []
    last = wide.index[-1]
    same_q_prev = last - 4
    for c in cols:
        rec: dict[str, Any] = {"series": c, "label": labels.get(c, c)}
        for y in years:
            rec[str(y)] = float(wide.loc[[p for p in wide.index if p.year == y], c].sum())
        for y0, y1 in zip(years[:-1], years[1:]):
            rec[f"Δ{y1}/{y0}"] = rec[str(y1)] / rec[str(y0)] - 1 if rec[str(y0)] else np.nan
        prev = float(wide.loc[same_q_prev, c]) if same_q_prev in wide.index else np.nan
        rec[f"Δ{last}/{same_q_prev}"] = float(wide.loc[last, c]) / prev - 1 if prev else np.nan
        roll = wide[c].rolling(4).sum().dropna().to_numpy()
        z, p = mann_kendall(roll)
        rec["MK_Z"] = z
        rec["MK_p"] = p
        rec["тренд"] = ("↓ снижение" if z < 0 else "↑ рост") if p < 0.05 else "не значим"
        rec["малый_объём"] = bool(wide[c].mean() < 100)
        rows.append(rec)
    return pd.DataFrame(rows)


def structure_table(wide: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    """Доли сегментов в блоке по годам (п.п.) — сдвиги продуктовой структуры."""
    out = []
    years = sorted({p.year for p in wide.index})
    for block, prefix in (("ice", "ice_"), ("electric", "el_")):
        segs = [c for c in wide.columns if c.startswith(prefix)]
        for y in years:
            sub = wide.loc[[p for p in wide.index if p.year == y], segs].sum()
            tot = sub.sum()
            for s in segs:
                out.append({"block": block, "segment": s, "label": labels.get(s, s), "year": y,
                            "share_in_block": sub[s] / tot if tot else np.nan})
    res = pd.DataFrame(out).pivot_table(index=["block", "segment", "label"], columns="year",
                                        values="share_in_block").reset_index()
    order = {c: i for i, c in enumerate(wide.columns)}
    res = res.sort_values("segment", key=lambda c: c.map(order)).reset_index(drop=True)
    res.columns = [str(c) for c in res.columns]
    return res


# ----------------------------------------------------------------------------------------------
# Сезонность
# ----------------------------------------------------------------------------------------------


def seasonality_table(wide: pd.DataFrame, series: list[str]) -> pd.DataFrame:
    """Сезонные эффекты: регрессия log(y) на кусочно-линейный тренд + сезонные дамми.

    Возвращает эффекты кварталов в % к среднегодовому уровню, их 90% ДИ и F-тест значимости
    сезонности. Тренд задан двумя участками (до/после 2 кв. 2025), чтобы спад не «съедал»
    сезонность.
    """
    rows = []
    idx = pd.PeriodIndex(wide.index, freq="Q")
    t = np.arange(len(idx), dtype=float)
    brk = np.asarray([float(p >= pd.Period("2025Q2", "Q")) for p in idx])
    for s in series:
        v = wide[s].to_numpy(float)
        y = np.log(v / (1 - v)) if s == "electric_share" else np.log(np.maximum(v, 1.0))
        D = seasonal_dummies(idx)
        X_full = np.column_stack([np.ones_like(t), t, brk, D])
        X_red = X_full[:, :3]
        b, res_full, *_ = np.linalg.lstsq(X_full, y, rcond=None)
        rss_f = float(((y - X_full @ b) ** 2).sum())
        br, *_ = np.linalg.lstsq(X_red, y, rcond=None)
        rss_r = float(((y - X_red @ br) ** 2).sum())
        df1, df2 = 3, len(y) - X_full.shape[1]
        F = ((rss_r - rss_f) / df1) / (rss_f / df2) if df2 > 0 else np.nan
        pF = 1 - stats.f.cdf(F, df1, df2) if df2 > 0 else np.nan
        sigma2 = rss_f / df2
        cov = sigma2 * np.linalg.inv(X_full.T @ X_full)
        g = np.r_[0.0, b[3:]]
        cg = np.zeros((4, 4))
        cg[1:, 1:] = cov[3:, 3:]
        # центрирование: эффект кварталa относительно среднего по году
        C = np.eye(4) - 1 / 4
        gc = C @ g
        vc = np.diag(C @ cg @ C.T)
        tcrit = stats.t.ppf(0.95, df2)
        rec = {"series": s, "F": F, "p_value": pF, "сезонность": "значима" if pF < 0.05 else ("слабая" if pF < 0.15 else "не выявлена")}
        for k in range(4):
            rec[f"Q{k + 1}"] = np.exp(gc[k]) - 1
            rec[f"Q{k + 1}_lo"] = np.exp(gc[k] - tcrit * np.sqrt(vc[k])) - 1
            rec[f"Q{k + 1}_hi"] = np.exp(gc[k] + tcrit * np.sqrt(vc[k])) - 1
        rows.append(rec)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Макрофакторы: длинный список, скрининг, сравнение спецификаций
# ----------------------------------------------------------------------------------------------

FACTOR_LONGLIST = [
    {"factor": "Ключевая ставка ЦБ (эфф., лаг 1–3 кв.)", "code": "key_rate_eff", "sign": "−",
     "rationale": "Стоимость кредита/лизинга — основной канал финансирования покупки техники; трансмиссия 2–4 кв.",
     "data": "ЦБ РФ, ежедневно", "status": "в модели"},
    {"factor": "Инвестиции в основной капитал (индекс уровня)", "code": "log_investment", "sign": "+",
     "rationale": "Погрузчики — инвестиционный товар; спрос следует за капвложениями складов, производства, логистики.",
     "data": "Росстат, ежеквартально", "status": "в модели"},
    {"factor": "Структурный сдвиг с 2 кв. 2025 (режим)", "code": "regime", "sign": "−",
     "rationale": "Завершение цикла замещения парка после ухода западных брендов, рост утильсбора, неценовые условия кредитования.",
     "data": "событийная переменная", "status": "в модели"},
    {"factor": "Время (структурный тренд электрификации)", "code": "trend", "sign": "+ (доля электро)",
     "rationale": "Замещение ДВС электрическими: складская логистика, e-commerce, экология, стоимость владения.",
     "data": "—", "status": "в модели доли"},
    {"factor": "Курс USD/RUB", "code": "usd_rub", "sign": "− (цена импорта)",
     "rationale": "Рынок преимущественно импортный (КНР): ослабление рубля удорожает технику.",
     "data": "ЦБ РФ, ежедневно", "status": "проверен, не включён: знак неустойчив, эффект поглощается ставкой"},
    {"factor": "ВВП, % г/г", "code": "gdp_yoy", "sign": "+",
     "rationale": "Общая деловая активность.",
     "data": "Росстат, ежеквартально", "status": "проверен, не включён: коллинеарен инвестициям, хуже по LOO"},
    {"factor": "Реальная ключевая ставка", "code": "real_key_rate", "sign": "−",
     "rationale": "Реальная стоимость денег для заёмщика.",
     "data": "ЦБ РФ, Росстат", "status": "проверен, альтернатива номинальной ставке (хуже по LOO)"},
    {"factor": "Утилизационный сбор на самоходную технику", "code": "policy", "sign": "− / опережающие закупки",
     "rationale": "Повышения 10.2024, 01.2025, 01.2026 → перенос закупок и рост цены импорта.",
     "data": "ПП РФ № 81", "status": "учтён через режим; отдельная переменная не идентифицируема на 13 точках"},
    {"factor": "Ввод складской недвижимости, лизинг спецтехники, PMI", "code": "-", "sign": "+",
     "rationale": "Прямые драйверы спроса на складскую технику.",
     "data": "IBC/NF Group, Эксперт РА, S&P Global — нет открытых квартальных рядов за весь период",
     "status": "ограничение: нет открытых рядов нужной частоты; рекомендуются к подключению"},
]


def factor_screening(wide: pd.DataFrame, macro: pd.DataFrame, lags=(0, 1, 2, 3)) -> pd.DataFrame:
    """Корреляции логарифма сезонно скорректированного рынка с факторами при лагах 0..3."""
    idx = pd.PeriodIndex(wide.index, freq="Q")
    y = np.log(wide["total_market"].to_numpy(float))
    D = np.column_stack([np.ones(len(idx)), seasonal_dummies(idx)])
    y_sa = y - D @ np.linalg.lstsq(D, y, rcond=None)[0]
    feats = build_features(macro, idx, [1, 2, 3])
    base = {
        "key_rate": macro["key_rate"],
        "real_key_rate": macro["real_key_rate"],
        "investment_yoy": macro["investment_yoy"],
        "usd_rub": macro["usd_rub"],
        "gdp_yoy": macro["gdp_yoy"],
    }
    rows = []
    for name, ser in base.items():
        rec = {"factor": name}
        for L in lags:
            x = np.array([ser.get(p - L, np.nan) for p in idx], float)
            ok = ~np.isnan(x)
            if ok.sum() >= 6:
                r, p = stats.pearsonr(x[ok], y_sa[ok])
                rec[f"r_lag{L}"] = r
                rec[f"p_lag{L}"] = p
        rows.append(rec)
    x = feats["log_investment"].to_numpy(float)
    r, p = stats.pearsonr(x, y_sa)
    rows.append({"factor": "log_investment (индекс)", "r_lag0": r, "p_lag0": p})
    return pd.DataFrame(rows)


def _ols_loo(X: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Точная LOO-ошибка МНК через диагональ hat-матрицы; также AIC и R²."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    H = X @ np.linalg.pinv(X.T @ X) @ X.T
    loo = r / (1 - np.clip(np.diag(H), None, 0.999))
    n, k = X.shape
    rss = float(r @ r)
    aic = n * np.log(rss / n) + 2 * k
    r2 = 1 - rss / float(((y - y.mean()) ** 2).sum())
    return float(np.sqrt(np.mean(loo**2))), aic, r2


def spec_comparison(wide: pd.DataFrame, macro: pd.DataFrame, regime_start: str = "2025Q2") -> pd.DataFrame:
    """Сравнение альтернативных спецификаций модели рынка по LOO-RMSE (прозрачный отбор факторов)."""
    idx = pd.PeriodIndex(wide.index, freq="Q")
    y = np.log(wide["total_market"].to_numpy(float))
    one = np.ones(len(idx))
    S = seasonal_dummies(idx)
    t = np.arange(len(idx), dtype=float)
    reg = np.asarray([float(p >= pd.Period(regime_start, "Q")) for p in idx])
    f123 = build_features(macro, idx, [1, 2, 3])
    f2 = build_features(macro, idx, [2])
    f12 = build_features(macro, idx, [1, 2])
    f0 = build_features(macro, idx, [0])
    rr = np.array([macro["real_key_rate"].get(p - 2, np.nan) for p in idx])
    specs = {
        "Сезонность + линейный тренд (без факторов)": [one, S, t],
        "Ставка без лага": [one, S, f0["key_rate_eff"]],
        "Ставка (лаг 2)": [one, S, f2["key_rate_eff"]],
        "Ставка (лаги 1–3)": [one, S, f123["key_rate_eff"]],
        "Инвестиции": [one, S, f123["log_investment"]],
        "Ставка (лаги 1–2) + инвестиции": [one, S, f12["key_rate_eff"], f12["log_investment"]],
        "Ставка (лаги 1–3) + инвестиции": [one, S, f123["key_rate_eff"], f123["log_investment"]],
        "Реальная ставка (лаг 2) + инвестиции": [one, S, rr, f123["log_investment"]],
        "Ставка + инвестиции + курс USD/RUB": [one, S, f123["key_rate_eff"], f123["log_investment"], np.log(f123["usd_rub"])],
        "Ставка + инвестиции + ВВП": [one, S, f123["key_rate_eff"], f123["log_investment"], f123["gdp_yoy"]],
        "Ставка + инвестиции + режим 2025Q2 (ИТОГОВАЯ)": [one, S, f123["key_rate_eff"], f123["log_investment"], reg],
    }
    rows = []
    for name, parts in specs.items():
        X = np.column_stack([np.asarray(p, float) for p in parts])
        ok = ~np.isnan(X).any(axis=1)
        loo, aic, r2 = _ols_loo(X[ok], y[ok])
        rows.append({"спецификация": name, "k": X.shape[1], "n": int(ok.sum()), "R2": r2, "AIC": aic, "LOO_RMSE": loo})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Декомпозиция изменения рынка по факторам
# ----------------------------------------------------------------------------------------------


def factor_decomposition(model: ForkcastModel, pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Вклад факторов в изменение log(рынка) между кварталами (сравниваются одинаковые кварталы,
    поэтому сезонность вычитается). Вклад = β · Δx (апостериорное среднее), остаток — необъяснённое."""
    idx = model.periods
    if pairs is None:
        last = idx[-1]
        pairs = [(str(p - 4), str(p)) for p in idx if p - 4 in idx]
        pairs = pairs[-5:]
    X = model.market.X_
    y = model.market.y_
    beta = model.market.beta_.mean(axis=0)
    names = model.market.names
    rows = []
    for a, b in pairs:
        ia, ib = idx.get_loc(pd.Period(a, "Q")), idx.get_loc(pd.Period(b, "Q"))
        dx = X[ib] - X[ia]
        contrib = beta * dx
        rec = {"from": a, "to": b, "Δlog_факт": y[ib] - y[ia], "изменение_%": np.exp(y[ib] - y[ia]) - 1}
        for nm, c in zip(names, contrib):
            if nm in ("const",):
                continue
            rec[nm] = c
        rec["сезонность"] = rec.pop("q2", 0) + rec.pop("q3", 0) + rec.pop("q4", 0)
        rec["остаток"] = rec["Δlog_факт"] - sum(v for k, v in rec.items() if k in names or k == "сезонность")
        rows.append(rec)
    return pd.DataFrame(rows)


def annual_decomposition(model: ForkcastModel, y0: int, y1: int) -> dict[str, float]:
    """Вклады факторов в изменение среднегодового log-уровня рынка между годами y0 и y1."""
    idx = model.periods
    X, y = model.market.X_, model.market.y_
    beta = model.market.beta_.mean(axis=0)
    m0 = np.asarray([p.year == y0 for p in idx])
    m1 = np.asarray([p.year == y1 for p in idx])
    dx = X[m1].mean(axis=0) - X[m0].mean(axis=0)
    out = {nm: float(b * d) for nm, b, d in zip(model.market.names, beta, dx) if nm not in ("const", "q2", "q3", "q4")}
    total = float(y[m1].mean() - y[m0].mean())
    out["остаток"] = total - sum(out.values())
    out["итого_log"] = total
    return out


# ----------------------------------------------------------------------------------------------
# Чувствительность и «лестница ставок»
# ----------------------------------------------------------------------------------------------


def key_rate_ladder(
    model: ForkcastModel,
    base_path: dict[str, np.ndarray],
    horizon: int,
    levels=(8, 9, 10, 11, 12, 13, 14, 15, 16),
    n: int = 1500,
) -> pd.DataFrame:
    """Рынок в зависимости от уровня ключевой ставки (постоянной с первого неизвестного квартала)
    при базовом пути инвестиций. Отвечает на вопрос «при какой ставке рынок вернётся к росту»."""
    rows = []
    for L in levels:
        path = {k: np.array(v, float).copy() for k, v in base_path.items()}
        path["key_rate"] = np.full(horizon, float(L))
        fc = model.forecast(path, horizon, n, scenario=f"kr{L}")
        m = fc.draws["total_market"]
        years = sorted({p.year for p in fc.periods})
        rec = {"key_rate": L}
        for y in years:
            mask = np.asarray([p.year == y for p in fc.periods])
            if mask.sum() == 4:
                tot = m[:, mask].sum(axis=1)
                rec[f"{y}_median"] = float(np.median(tot))
                rec[f"{y}_p10"] = float(np.quantile(tot, 0.1))
                rec[f"{y}_p90"] = float(np.quantile(tot, 0.9))
        rec["electric_share_end"] = float(np.median(fc.draws["electric_share"][:, -1]))
        rows.append(rec)
    return pd.DataFrame(rows)


def prior_sensitivity(
    wide: pd.DataFrame, macro: pd.DataFrame, cfg: dict[str, Any], path: dict[str, np.ndarray], horizon: int,
    scales=(0.5, 1.0, 2.0, 5.0), seed: int = 0,
) -> pd.DataFrame:
    """Устойчивость оценок к ширине априорных распределений (×scale к sd)."""
    rows = []
    for sc in scales:
        c = copy.deepcopy(cfg)
        for blk in ("market", "share"):
            for k, v in c["priors"][blk].items():
                c["priors"][blk][k] = [v[0], v[1] * sc]
        m = ForkcastModel(c, seed=seed).fit(wide, macro)
        s = m.market.summary().set_index("коэффициент")
        fc = m.forecast(path, horizon, 1500)
        tot = fc.draws["total_market"]
        rows.append({
            "prior_sd_scale": sc,
            "β_ставка": s.loc["key_rate_eff", "апостериори_среднее"],
            "β_инвестиции": s.loc["log_investment", "апостериори_среднее"],
            "β_режим": s.loc["regime", "апостериори_среднее"] if "regime" in s.index else np.nan,
            "рынок_4кв_медиана": float(np.median(tot[:, :4].sum(axis=1))),
            "рынок_8кв_медиана": float(np.median(tot.sum(axis=1))),
        })
    return pd.DataFrame(rows)


def share_cap_sensitivity(
    wide: pd.DataFrame, macro: pd.DataFrame, cfg: dict[str, Any], path: dict[str, np.ndarray], horizon: int,
    caps=(0.80, 0.85, 0.90, 0.95, 0.999), seed: int = 0,
) -> pd.DataFrame:
    rows = []
    for cap in caps:
        c = copy.deepcopy(cfg)
        c["electric_share_cap"] = cap
        try:
            m = ForkcastModel(c, seed=seed).fit(wide, macro)
        except ValueError:
            continue
        fc = m.forecast(path, horizon, 1500)
        sh = fc.draws["electric_share"]
        rows.append({"cap": cap, "доля_через_4кв": float(np.median(sh[:, min(3, horizon - 1)])),
                     "доля_конец_горизонта": float(np.median(sh[:, -1])),
                     "p10_конец": float(np.quantile(sh[:, -1], 0.1)), "p90_конец": float(np.quantile(sh[:, -1], 0.9))})
    return pd.DataFrame(rows)
