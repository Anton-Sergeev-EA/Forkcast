"""Внешние (макроэкономические) факторы: загрузка, агрегация до кварталов, признаки для модели.

Ключевые преобразования
-----------------------
* ``key_rate`` — среднедневная ключевая ставка за квартал (из журнала решений ЦБ);
* ``key_rate_eff`` — «эффективная» ставка: среднее ставки за кварталы t-1..t-3. Отражает лаг
  трансмиссии ДКП в инвестиционный спрос (решение о покупке техники → лизинг/кредит → поставка);
* ``log_investment`` — логарифм индекса инвестиций в основной капитал, построенного цепочкой
  из темпов г/г (база — соответствующий квартал 2022 г. = 1). Индекс не содержит сезонности
  и измеряет *уровень* инвестактивности, что согласуется с моделью уровня продаж;
* ``real_key_rate`` — ставка минус инфляция г/г (для анализа).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def key_rate_daily(path: Path, end: str | pd.Timestamp | None = None) -> pd.Series:
    """Дневной ряд ключевой ставки из журнала решений (ставка действует с ``effective_date``)."""
    dec = pd.read_csv(path, parse_dates=["effective_date"]).sort_values("effective_date")
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    end = max(end, dec["effective_date"].max())
    days = pd.date_range(dec["effective_date"].min(), end, freq="D")
    s = pd.Series(np.nan, index=days)
    for d, r in zip(dec["effective_date"], dec["key_rate"]):
        s.loc[d:] = r
    return s.rename("key_rate")


def quarterly_key_rate(path: Path, through: pd.Period | None = None) -> pd.Series:
    """Средняя ставка за квартал. Кварталы, закрытые не полностью, считаются по известной части
    (последнее решение действует до следующего заседания)."""
    if through is None:
        # по умолчанию факт известен до квартала последней записи журнала решений
        last = pd.read_csv(path, parse_dates=["effective_date"])["effective_date"].max()
        through = pd.Period(last, freq="Q")
    s = key_rate_daily(path, through.end_time.normalize())
    q = s.groupby(s.index.to_period("Q")).mean()
    q.index = pd.PeriodIndex(q.index, freq="Q")
    return q.loc[:through]


def quarterly_fx(path: Path) -> pd.Series:
    m = pd.read_csv(path)
    m["period"] = pd.PeriodIndex(pd.to_datetime(m["month"]), freq="Q")
    q = m.groupby("period")["usd_rub_avg"].mean()
    q.index = pd.PeriodIndex(q.index, freq="Q")
    return q.rename("usd_rub")


def chain_index(yoy_pct: pd.Series, base: float = 1.0) -> pd.Series:
    """Индекс уровня из темпов г/г: I_t = I_{t-4} · (1 + g_t/100), I_{2022Qk} = base."""
    yoy_pct = yoy_pct.sort_index()
    out = pd.Series(np.nan, index=yoy_pct.index, dtype=float)
    for p, g in yoy_pct.items():
        prev = out.get(p - 4, np.nan)
        prev = base if np.isnan(prev) else prev
        out[p] = prev * (1.0 + g / 100.0)
    return out


def effective_rate(kr: pd.Series, periods: Iterable[pd.Period], lags: Iterable[int]) -> pd.Series:
    lags = list(lags)
    vals = []
    for p in periods:
        v = [kr.get(p - L, np.nan) for L in lags]
        vals.append(float(np.nanmean(v)) if not all(np.isnan(v)) else np.nan)
    return pd.Series(vals, index=pd.PeriodIndex(list(periods), freq="Q"), name="key_rate_eff")


def load_macro(external_dir: Path, through: pd.Period | None = None) -> pd.DataFrame:
    """Собирает квартальную таблицу фактических значений внешних факторов."""
    kr = quarterly_key_rate(external_dir / "key_rate_decisions.csv", through)
    fx = quarterly_fx(external_dir / "usd_rub_monthly.csv")
    mq = pd.read_csv(external_dir / "macro_quarterly.csv")
    mq.index = pd.PeriodIndex(mq.pop("period"), freq="Q")
    df = pd.concat([kr.rename("key_rate"), fx, mq], axis=1).sort_index()
    df.index = pd.PeriodIndex(df.index, freq="Q")
    df["real_key_rate"] = df["key_rate"] - df["cpi_yoy_eop"]
    return df


def build_features(macro: pd.DataFrame, periods: Iterable[pd.Period], lags: Iterable[int]) -> pd.DataFrame:
    """Признаки модели для заданных кварталов (история или история + сценарный путь)."""
    periods = pd.PeriodIndex(list(periods), freq="Q")
    out = pd.DataFrame(index=periods)
    out["key_rate"] = macro["key_rate"].reindex(periods)
    out["key_rate_eff"] = effective_rate(macro["key_rate"], periods, lags)
    inv_idx = chain_index(macro["investment_yoy"].dropna())
    out["investment_yoy"] = macro["investment_yoy"].reindex(periods)
    out["log_investment"] = np.log(inv_idx.reindex(periods))
    out["usd_rub"] = macro["usd_rub"].reindex(periods)
    out["gdp_yoy"] = macro["gdp_yoy"].reindex(periods)
    out["real_key_rate"] = macro["real_key_rate"].reindex(periods)
    return out


def extend_with_path(
    macro: pd.DataFrame,
    path_periods: list[pd.Period],
    key_rate: np.ndarray,
    investment_yoy: np.ndarray,
    usd_rub: np.ndarray | None = None,
    gdp_yoy: np.ndarray | None = None,
    respect_actuals: bool = True,
) -> pd.DataFrame:
    """Добавляет сценарный путь факторов к фактическим данным.

    При ``respect_actuals=True`` уже известные фактические значения (например, ставка за
    текущий квартал) не перезаписываются сценарием — режим *наукаста*.
    """
    ext = macro.copy()
    idx = ext.index.union(pd.PeriodIndex(path_periods, freq="Q"))
    ext = ext.reindex(idx)
    cols = {"key_rate": key_rate, "investment_yoy": investment_yoy}
    if usd_rub is not None:
        cols["usd_rub"] = usd_rub
    if gdp_yoy is not None:
        cols["gdp_yoy"] = gdp_yoy
    for col, values in cols.items():
        for p, v in zip(path_periods, values):
            if respect_actuals and pd.notna(macro[col].get(p, np.nan)):
                continue
            ext.loc[p, col] = v
    return ext
