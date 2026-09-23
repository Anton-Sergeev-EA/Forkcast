"""Сценарии внешних факторов и их взвешивание."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .model import ForecastDraws, ForkcastModel

FACTORS = ("key_rate", "investment_yoy", "usd_rub", "gdp_yoy")


@dataclass
class Scenario:
    key: str
    title: str
    weight: float
    color: str
    assumptions: str
    path: pd.DataFrame  # index = period, columns = FACTORS


def load_scenarios(cfg: dict[str, Any]) -> list[Scenario]:
    periods = pd.PeriodIndex([pd.Period(p, freq="Q") for p in cfg["periods"]], freq="Q")
    out = []
    for key, sc in cfg["scenarios"].items():
        data = {f: sc.get(f, [np.nan] * len(periods)) for f in FACTORS}
        for f, v in data.items():
            if len(v) != len(periods):
                raise ValueError(f"Сценарий {key}: длина '{f}' ({len(v)}) ≠ числу периодов ({len(periods)})")
        out.append(
            Scenario(
                key=key,
                title=sc.get("title", key),
                weight=float(sc.get("weight", 0.0)),
                color=sc.get("color", "#777777"),
                assumptions=" ".join(str(sc.get("assumptions", "")).split()),
                path=pd.DataFrame(data, index=periods),
            )
        )
    total = sum(s.weight for s in out)
    if total <= 0:
        raise ValueError("Сумма весов сценариев должна быть > 0")
    for s in out:
        s.weight /= total
    return out


def align_path(sc: Scenario, fc_periods: pd.PeriodIndex) -> dict[str, np.ndarray]:
    """Выравнивает путь сценария на прогнозные кварталы. За пределами заданных периодов
    значения продлеваются последним известным (с предупреждением) — это позволяет обновлять
    данные, не переписывая сценарии немедленно."""
    path = sc.path.reindex(sc.path.index.union(fc_periods)).sort_index()
    beyond = [p for p in fc_periods if p > sc.path.index.max()]
    if beyond:
        warnings.warn(
            f"Сценарий {sc.key}: для {', '.join(map(str, beyond))} значения не заданы — продлены последним значением. "
            "Обновите config/scenarios.yaml.",
            stacklevel=2,
        )
    path = path.ffill().bfill()
    return {f: path.loc[fc_periods, f].to_numpy(float) for f in FACTORS}


def run_scenarios(
    model: ForkcastModel,
    scenarios: list[Scenario],
    horizon: int,
    n: int,
    factor_uncertainty: dict[str, Any] | None,
) -> dict[str, ForecastDraws]:
    last = model.periods[-1]
    fc_periods = pd.period_range(last + 1, last + horizon, freq="Q")
    res = {}
    for i, sc in enumerate(scenarios):
        res[sc.key] = model.forecast(
            align_path(sc, fc_periods), horizon, n, scenario=sc.key,
            factor_uncertainty=factor_uncertainty, seed_offset=17 * (i + 1),
        )
    return res


def mixture(results: dict[str, ForecastDraws], scenarios: list[Scenario], seed: int = 0) -> ForecastDraws:
    """Взвешенная смесь сценариев: итоговое распределение учитывает и неопределённость внутри
    сценариев, и неопределённость выбора сценария (межсценарный разброс)."""
    rng = np.random.default_rng(seed)
    first = next(iter(results.values()))
    n = next(iter(first.draws.values())).shape[0]
    counts = rng.multinomial(n, [s.weight for s in scenarios])
    draws: dict[str, list[np.ndarray]] = {k: [] for k in first.draws}
    factors: dict[str, list[np.ndarray]] = {k: [] for k in first.factors}
    for sc, c in zip(scenarios, counts):
        if c == 0:
            continue
        r = results[sc.key]
        idx = rng.choice(n, size=c, replace=False)
        for k in draws:
            draws[k].append(r.draws[k][idx])
        for k in factors:
            factors[k].append(r.factors[k][idx])
    return ForecastDraws(
        "weighted",
        first.periods,
        {k: np.concatenate(v) for k, v in draws.items()},
        {k: np.concatenate(v) for k, v in factors.items()},
    )


def uncertainty_budget(
    model: ForkcastModel, sc: Scenario, horizon: int, n: int, factor_uncertainty: dict[str, Any]
) -> pd.DataFrame:
    """Разложение дисперсии log(рынка) по источникам неопределённости (для сценария ``sc``).

    Последовательно добавляем источники: параметры → +шум модели → +путь факторов.
    Доля источника = прирост дисперсии / полная дисперсия.
    """
    last = model.periods[-1]
    fc = pd.period_range(last + 1, last + horizon, freq="Q")
    path = align_path(sc, fc)
    v_par = np.log(model.forecast(path, horizon, n, noise=False, param_uncertainty=True).draws["total_market"]).var(0)
    v_noise = np.log(model.forecast(path, horizon, n, noise=True, param_uncertainty=True).draws["total_market"]).var(0)
    v_all = np.log(
        model.forecast(path, horizon, n, factor_uncertainty=factor_uncertainty).draws["total_market"]
    ).var(0)
    v_noise = np.maximum(v_noise, v_par)
    v_all = np.maximum(v_all, v_noise)
    return pd.DataFrame(
        {
            "period": fc,
            "параметры модели": v_par / v_all,
            "случайный шум": (v_noise - v_par) / v_all,
            "путь внешних факторов": (v_all - v_noise) / v_all,
            "sd_log_total": np.sqrt(v_all),
        }
    )
