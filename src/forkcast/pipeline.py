"""Сквозной конвейер: данные → анализ → модель → сценарии → планирование → отчёты."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from . import analysis as A
from .backtest import rolling_backtest, score
from .config import Settings, load_settings
from .data_io import DataQualityReport, read_market_excel, to_wide
from .macro import build_features, load_macro
from .model import ForecastDraws, ForkcastModel
from .planning import production_corridor, silant_addressable_market
from .scenarios import Scenario, align_path, load_scenarios, mixture, run_scenarios, uncertainty_budget
from .vintage import evaluate_vintages, save_vintage

log = logging.getLogger("forkcast")

SERIES_LABELS = {
    "total_market": "Рынок, всего",
    "total_ice": "ДВС, всего",
    "total_electric": "Электро, всего",
    "electric_share": "Доля электро",
}


@dataclass
class RunResult:
    settings: Settings
    long: pd.DataFrame
    wide: pd.DataFrame
    dq: DataQualityReport
    macro: pd.DataFrame
    features: pd.DataFrame
    labels: dict[str, str]
    model: ForkcastModel
    scenarios: list[Scenario]
    forecasts: dict[str, ForecastDraws]
    weighted: ForecastDraws
    summary: pd.DataFrame
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def last_period(self) -> pd.Period:
        return self.wide.index[-1]

    @property
    def short_h(self) -> int:
        return int(self.settings.forecast["short_horizon"])

    @property
    def long_h(self) -> int:
        return int(self.settings.forecast["long_horizon"])

    def scenario(self, key: str) -> Scenario:
        return next(s for s in self.scenarios if s.key == key)

    def label(self, code: str) -> str:
        return SERIES_LABELS.get(code) or self.labels.get(code, code)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def _annual_sums(fc: ForecastDraws, series: str, first: int) -> dict[str, float]:
    d = fc.draws[series]
    out = {}
    for name, sl in (("4q", slice(0, 4)), ("8q", slice(0, 8))):
        if d.shape[1] >= sl.stop:
            tot = d[:, sl].sum(axis=1)
            out[f"{name}_p10"] = float(np.quantile(tot, 0.1))
            out[f"{name}_p50"] = float(np.median(tot))
            out[f"{name}_p90"] = float(np.quantile(tot, 0.9))
    return out


def run(settings: Settings | None = None, market_file: str | None = None, quick: bool = False) -> RunResult:
    """Выполняет полный расчёт. ``quick=True`` уменьшает число симуляций (для тестов)."""
    s = settings or load_settings(market_file=market_file)
    fc_cfg = s.forecast
    seed = int(fc_cfg.get("seed", 0))
    n = 600 if quick else int(fc_cfg.get("n_draws", 4000))
    H = int(fc_cfg["long_horizon"])

    # 1. Данные ------------------------------------------------------------------------------
    long, dq = read_market_excel(s.market_file, s.segments)
    if not dq.ok:
        raise ValueError("Файл рыночных данных не прошёл проверку:\n" + dq.to_frame().to_string())
    n_ice = long.loc[long.block == "ice", "segment"].nunique()
    if n_ice != 9:
        dq.add("соответствие ТЗ", "INFO",
               f"в описании кейса указано 9 сегментов ДВС, в файле — {n_ice} (до 20–25 т); прогноз строится по фактическим")
    wide = to_wide(long)
    labels = dict(zip(long["segment"], long["label"]))
    macro = load_macro(s.external_dir)
    lags = s.model.get("key_rate_lags", [1, 2, 3])
    features = build_features(macro, wide.index, lags)

    # 2. Модель ------------------------------------------------------------------------------
    model_cfg = dict(s.model)
    if quick:
        model_cfg["gibbs"] = {"iterations": 1500, "burn_in": 300, "thin": 1}
    model = ForkcastModel(model_cfg, seed=seed).fit(wide, macro)

    # 3. Сценарии ----------------------------------------------------------------------------
    scenarios = load_scenarios(s.scenarios)
    unc = s.scenarios.get("factor_uncertainty")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forecasts = run_scenarios(model, scenarios, H, n, unc)
    for w in caught:
        log.warning(str(w.message))
    weighted = mixture(forecasts, scenarios, seed=seed)
    summary = pd.concat([f.summary() for f in [*forecasts.values(), weighted]], ignore_index=True)

    # 4. Аналитика ---------------------------------------------------------------------------
    t: dict[str, pd.DataFrame] = {}
    t["data_quality"] = dq.to_frame()
    t["annual"] = A.annual_table(wide, labels)
    t["structure"] = A.structure_table(wide, labels)
    t["seasonality"] = A.seasonality_table(wide, ["total_market", "total_ice", "total_electric", "electric_share"])
    t["seasonal_model"] = model.seasonal_factors()
    t["factor_longlist"] = pd.DataFrame(A.FACTOR_LONGLIST)
    t["factor_screening"] = A.factor_screening(wide, macro)
    t["spec_comparison"] = A.spec_comparison(wide, macro, str(model.regime_start or "2025Q2"))
    t["coef_market"] = model.market.summary()
    t["coef_share"] = model.share.summary()
    t["decomposition"] = A.factor_decomposition(model)
    years = sorted({p.year for p in wide.index})
    full_years = [y for y in years if sum(p.year == y for p in wide.index) == 4]
    if len(full_years) >= 2:
        t["decomposition_annual"] = pd.DataFrame([A.annual_decomposition(model, full_years[-2], full_years[-1])])
    t["fitted"] = model.fitted_frame().reset_index(names="period")

    fc_periods = pd.period_range(wide.index[-1] + 1, wide.index[-1] + H, freq="Q")
    base_sc = next((sc for sc in scenarios if sc.key == "consensus"), scenarios[0])
    base_path = align_path(base_sc, fc_periods)
    n_small = 400 if quick else 1500
    t["uncertainty_budget"] = uncertainty_budget(model, base_sc, H, n_small, unc)
    t["key_rate_ladder"] = A.key_rate_ladder(model, base_path, H, n=n_small)
    if not quick:
        t["prior_sensitivity"] = A.prior_sensitivity(wide, macro, model_cfg, base_path, H, seed=seed)
        t["share_cap_sensitivity"] = A.share_cap_sensitivity(wide, macro, model_cfg, base_path, H, seed=seed)

    # 5. Бэктест -----------------------------------------------------------------------------
    bt_cfg = s.backtest
    bt = rolling_backtest(wide, macro, model_cfg, int(bt_cfg["min_train"]), int(bt_cfg["max_horizon"]),
                          n=400 if quick else 2000, seed=seed)
    t["backtest"] = bt
    t["backtest_scores"] = score(bt, wide)

    # 6. Сводные таблицы прогноза -------------------------------------------------------------
    rows = []
    for sc in [*scenarios, None]:
        key = sc.key if sc else "weighted"
        fc = weighted if sc is None else forecasts[key]
        for ser in ["total_market", "total_ice", "total_electric"]:
            rec = {"scenario": key, "title": sc.title if sc else "Взвешенный (смесь сценариев)",
                   "weight": sc.weight if sc else 1.0, "series": ser}
            rec.update(_annual_sums(fc, ser, 0))
            rows.append(rec)
        sh = fc.draws["electric_share"]
        rows.append({"scenario": key, "title": sc.title if sc else "Взвешенный (смесь сценариев)",
                     "weight": sc.weight if sc else 1.0, "series": "electric_share",
                     "4q_p50": float(np.median(sh[:, 3])), "8q_p50": float(np.median(sh[:, -1])),
                     "4q_p10": float(np.quantile(sh[:, 3], 0.1)), "4q_p90": float(np.quantile(sh[:, 3], 0.9)),
                     "8q_p10": float(np.quantile(sh[:, -1], 0.1)), "8q_p90": float(np.quantile(sh[:, -1], 0.9))})
    t["scenario_totals"] = pd.DataFrame(rows)

    # 7. Планирование ------------------------------------------------------------------------
    t["plan_short"] = production_corridor(weighted, s.segments, labels, horizon=int(fc_cfg["short_horizon"]))
    t["plan_long"] = production_corridor(weighted, s.segments, labels, horizon=H)
    t["sam"] = silant_addressable_market(weighted, s.segments)

    # 8. Ключевые факты для текста отчёта ---------------------------------------------------------
    last = wide.index[-1]
    prev_year_total = wide.loc[[p for p in wide.index if p.year == full_years[-1]], "total_market"].sum() if full_years else np.nan
    facts = {
        "last_period": str(last),
        "first_period": str(wide.index[0]),
        "n_quarters": len(wide),
        "last_total": float(wide.loc[last, "total_market"]),
        "last_yoy": float(wide.loc[last, "total_market"] / wide.loc[last - 4, "total_market"] - 1) if last - 4 in wide.index else np.nan,
        "last_share": float(wide.loc[last, "electric_share"]),
        "last_full_year": full_years[-1] if full_years else None,
        "last_full_year_total": float(prev_year_total),
        "weighted_4q": _annual_sums(weighted, "total_market", 0),
        "fc_periods": [str(p) for p in fc_periods],
        "beta_kr": float(model.market.beta_[:, 4].mean()),
        "beta_kr_q05": float(np.quantile(model.market.beta_[:, 4], 0.05)),
        "beta_kr_q95": float(np.quantile(model.market.beta_[:, 4], 0.95)),
        "beta_inv": float(model.market.beta_[:, 5].mean()),
        "beta_regime": float(model.market.beta_[:, 6].mean()) if model.market.beta_.shape[1] > 6 else None,
        "share_trend": float(model.share.beta_[:, 4].mean()),
        "r2_market": model.market.r2(),
        "r2_share": model.share.r2(),
        "sigma_market": float(np.sqrt(model.market.sigma2_.mean())),
    }

    # 9. Манифест воспроизводимости ----------------------------------------------------------------
    manifest = {
        "forkcast_version": __version__,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "seed": seed,
        "n_draws": n,
        "quick": quick,
        "market_file": str(s.market_file.relative_to(s.root)) if s.market_file.is_relative_to(s.root) else str(s.market_file),
        "market_sha256": _sha256(s.market_file),
        "external_sha256": {p.name: _sha256(p) for p in sorted(s.external_dir.glob("*.csv"))},
        "data_last_period": str(last),
        "scenarios": {sc.key: round(sc.weight, 4) for sc in scenarios},
    }
    return RunResult(s, long, wide, dq, macro, features, labels, model, scenarios, forecasts, weighted,
                     summary, t, facts, manifest)


def save_outputs(res: RunResult, make_reports: bool = True) -> dict[str, Path]:
    """Сохраняет датасет, прогнозы, отчёты и винтаж прогноза."""
    s = res.settings
    out: dict[str, Path] = {}
    proc = s.processed_dir
    proc.mkdir(parents=True, exist_ok=True)

    # датасет
    long = res.long.copy()
    long["period"] = long["period"].astype(str)
    long.to_csv(proc / "market_long.csv", index=False)
    wide = res.wide.copy()
    wide.index = wide.index.astype(str)
    wide.to_csv(proc / "market_wide.csv", index_label="period")
    macro = res.macro.copy()
    macro.index = macro.index.astype(str)
    macro.to_csv(proc / "macro_quarterly_processed.csv", index_label="period")
    feats = res.features.copy()
    feats.index = feats.index.astype(str)
    dataset = wide.join(feats, how="left")
    dataset.to_csv(proc / "model_dataset.csv", index_label="period")
    summ = res.summary.copy()
    summ["period"] = summ["period"].astype(str)
    summ.to_csv(proc / "forecast_quantiles.csv", index=False)
    out["dataset"] = proc / "model_dataset.csv"

    vint = save_vintage(res.summary, res.last_period, s.vintages_dir)
    out["vintage"] = vint
    res.tables["vintage_eval"] = evaluate_vintages(res.wide, s.vintages_dir)

    s.reports_dir.mkdir(parents=True, exist_ok=True)
    with open(s.reports_dir / "run_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(res.manifest, fh, ensure_ascii=False, indent=2)
    out["manifest"] = s.reports_dir / "run_manifest.json"

    if make_reports:
        from .reporting import build_all

        out.update(build_all(res))
    return out
