"""Ретроспективная проверка (rolling-origin backtest) и сравнение с эталонными моделями.

Схема: для каждой точки отсечения T (обучение на кварталах ≤ T) строится прогноз на 1..4
квартала вперёд и сравнивается с фактом. Внешние факторы берутся фактические (условный,
ex-post прогноз): так измеряется качество *модели связи рынка с факторами*, а неопределённость
самих факторов покрывается сценариями. Это стандартная практика для сценарных моделей
(ЦБ, МВФ), так как архивных прогнозов факторов «на дату» в открытом доступе нет.

Эталоны:
* **SNaive** — сезонный наивный: значение того же квартала год назад;
* **ETS** — экспоненциальное сглаживание (затухающий тренд) по сезонно скорректированному log-ряду;
* **Seasonal mean** — средний уровень последних 4 кварталов × сезонный профиль.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from .model import ForkcastModel

SERIES = ["total_market", "total_ice", "total_electric", "electric_share"]


def _seasonal_profile(y: np.ndarray, periods: pd.PeriodIndex) -> np.ndarray:
    ly = np.log(y)
    q = np.asarray(periods.quarter)
    prof = np.array([ly[q == k].mean() - ly.mean() if np.any(q == k) else 0.0 for k in (1, 2, 3, 4)])
    return prof - prof.mean()


def bench_snaive(y: np.ndarray, periods: pd.PeriodIndex, h: int) -> np.ndarray:
    return np.array([y[len(y) - 4 + ((k - 1) % 4)] for k in range(1, h + 1)])


def bench_seasonal_mean(y: np.ndarray, periods: pd.PeriodIndex, h: int) -> np.ndarray:
    prof = _seasonal_profile(y, periods)
    level = np.log(y[-4:]).mean()
    fut = pd.period_range(periods[-1] + 1, periods[-1] + h, freq="Q")
    return np.exp(level + prof[np.asarray(fut.quarter) - 1])


def bench_ets(y: np.ndarray, periods: pd.PeriodIndex, h: int) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    prof = _seasonal_profile(y, periods)
    q = np.asarray(periods.quarter) - 1
    sa = np.log(y) - prof[q]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(sa, trend="add", damped_trend=True, initialization_method="estimated").fit()
    fut = pd.period_range(periods[-1] + 1, periods[-1] + h, freq="Q")
    return np.exp(fit.forecast(h) + prof[np.asarray(fut.quarter) - 1])


BENCHMARKS = {"SNaive": bench_snaive, "ETS": bench_ets, "SeasonalMean": bench_seasonal_mean}


def rolling_backtest(
    wide: pd.DataFrame,
    macro: pd.DataFrame,
    model_cfg: dict[str, Any],
    min_train: int = 8,
    max_h: int = 4,
    n: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Возвращает длинную таблицу прогнозов и фактов для всех точек отсечения и горизонтов."""
    rows = []
    periods = pd.PeriodIndex(wide.index, freq="Q")
    for T in range(min_train, len(wide)):
        train = wide.iloc[:T]
        h = min(max_h, len(wide) - T)
        test_periods = periods[T : T + h]
        model = ForkcastModel(model_cfg, seed=seed + T).fit(train, macro)
        # фактические пути факторов на тестовый период (ex-post)
        path = {
            "key_rate": macro["key_rate"].reindex(test_periods).to_numpy(float),
            "investment_yoy": macro["investment_yoy"].reindex(test_periods).to_numpy(float),
        }
        fc = model.forecast(path, h, n, scenario="backtest")
        for s in SERIES:
            d = fc.draws[s]
            for j, p in enumerate(test_periods):
                rows.append(
                    {
                        "origin": periods[T - 1], "period": p, "h": j + 1, "series": s, "model": "Forkcast",
                        "forecast": float(np.median(d[:, j])), "actual": float(wide[s].iloc[T + j]),
                        "lo80": float(np.quantile(d[:, j], 0.1)), "hi80": float(np.quantile(d[:, j], 0.9)),
                        "lo95": float(np.quantile(d[:, j], 0.025)), "hi95": float(np.quantile(d[:, j], 0.975)),
                    }
                )
        for s in ["total_market", "total_ice", "total_electric"]:
            y = train[s].to_numpy(float)
            for name, fn in BENCHMARKS.items():
                f = fn(y, periods[:T], h)
                for j, p in enumerate(test_periods):
                    rows.append(
                        {"origin": periods[T - 1], "period": p, "h": j + 1, "series": s, "model": name,
                         "forecast": float(f[j]), "actual": float(wide[s].iloc[T + j])}
                    )
        # эталон для доли: последняя наблюдённая доля того же квартала / среднее последних 4
        sh = train["electric_share"].to_numpy(float)
        for j, p in enumerate(test_periods):
            rows.append({"origin": periods[T - 1], "period": p, "h": j + 1, "series": "electric_share",
                         "model": "SNaive", "forecast": float(sh[len(sh) - 4 + (j % 4)]),
                         "actual": float(wide["electric_share"].iloc[T + j])})
            rows.append({"origin": periods[T - 1], "period": p, "h": j + 1, "series": "electric_share",
                         "model": "SeasonalMean", "forecast": float(sh[-4:].mean()),
                         "actual": float(wide["electric_share"].iloc[T + j])})
    return pd.DataFrame(rows)


def score(bt: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """Метрики: MAPE, MASE (масштаб — MAE сезонного наивного прогноза в полной выборке),
    смещение (bias), покрытие 80%/95% интервалов (только Forkcast)."""
    out = []
    for (s, m), g in bt.groupby(["series", "model"]):
        err = g["forecast"] - g["actual"]
        y = wide[s].to_numpy(float)
        scale = np.mean(np.abs(y[4:] - y[:-4])) if len(y) > 4 else np.nan
        rec = {
            "series": s, "model": m, "n": len(g),
            "MAPE": float(np.mean(np.abs(err) / np.abs(g["actual"]))),
            "MASE": float(np.mean(np.abs(err)) / scale) if scale else np.nan,
            "bias_%": float(np.mean(err / g["actual"])),
        }
        if "lo80" in g and g["lo80"].notna().any():
            rec["cover80"] = float(((g["actual"] >= g["lo80"]) & (g["actual"] <= g["hi80"])).mean())
            rec["cover95"] = float(((g["actual"] >= g["lo95"]) & (g["actual"] <= g["hi95"])).mean())
        out.append(rec)
    res = pd.DataFrame(out)
    order = {"Forkcast": 0, "SNaive": 1, "ETS": 2, "SeasonalMean": 3}
    return res.sort_values(["series", "model"], key=lambda c: c.map(order) if c.name == "model" else c)
