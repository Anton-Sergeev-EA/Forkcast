"""Иерархическая факторная модель рынка вилочных погрузчиков.

Архитектура (сверху вниз, прогнозы согласованы по построению):

1. **Рынок в целом** — байесовская динамическая регрессия уровня продаж::

       log M_t = α + Σ γ_q·D_q + β_r·R_t + β_i·log I_t + ε_t

   где R_t — эффективная ключевая ставка (среднее за t-1..t-3), I_t — индекс инвестиций в ОК.

2. **Доля электропогрузчиков** — логистическая S-кривая с насыщением ``cap``::

       logit(s_t / cap) = a + Σ c_q·D_q + b·t + δ·R_t + u_t

   Тренд b — структурное замещение ДВС электрическими; δ — циклическая компонента
   (ДВС-техника сильнее зависит от стоимости кредита).

3. **Структура сегментов** внутри блоков ДВС и электро — композиционная модель в
   аддитивных лог-отношениях (ALR) с экспоненциальным сглаживанием уровня и затухающим трендом.

   Итог: M·(1−s)·w_ice + M·s·w_el — сумма сегментов всегда равна блоку, блоки — рынку.

Неопределённость прогноза складывается из трёх источников, каждый моделируется явно:
параметры (апостериорные выборки), шум модели (σ) и неопределённость пути внешних факторов
внутри сценария; плюс межсценарный разброс при взвешивании сценариев.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .bayes import BayesLinReg
from .macro import build_features, chain_index


def seasonal_dummies(periods: pd.PeriodIndex) -> np.ndarray:
    q = np.asarray(periods.quarter)
    return np.column_stack([(q == k).astype(float) for k in (2, 3, 4)])


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------------------------------------------------
# Структура сегментов
# ----------------------------------------------------------------------------------------------
@dataclass
class SegmentMix:
    """Прогноз долей сегментов внутри блока (композиционные данные)."""

    halflife: float = 3.0
    damping: float = 0.8
    pseudo: float = 5.0            # половина шага округления исходных данных (10 шт.)
    columns: list[str] = field(default_factory=list)
    ref: int = 0
    level_: np.ndarray | None = None
    slope_: np.ndarray | None = None
    resid_sd_: np.ndarray | None = None
    n_obs_: int = 0

    def fit(self, units: pd.DataFrame) -> "SegmentMix":
        self.columns = list(units.columns)
        x = units.to_numpy(float) + self.pseudo
        p = x / x.sum(axis=1, keepdims=True)
        self.ref = int(np.argmax(p.mean(axis=0)))
        alr = np.log(p) - np.log(p[:, [self.ref]])
        T = len(alr)
        w = 0.5 ** (np.arange(T)[::-1] / self.halflife)
        w /= w.sum()
        self.level_ = (w[:, None] * alr).sum(axis=0)
        t = np.arange(T) - (T - 1) / 2
        slope = (t[:, None] * (alr - alr.mean(axis=0))).sum(axis=0) / (t**2).sum()
        self.slope_ = slope
        trend_fit = alr.mean(axis=0) + np.outer(t, slope)
        self.resid_sd_ = (alr - trend_fit).std(axis=0, ddof=2) if T > 2 else np.full(alr.shape[1], 0.2)
        self.resid_sd_[self.ref] = 0.0
        self.n_obs_ = T
        return self

    def simulate(self, horizon: int, n: int, rng: np.random.Generator, noise: bool = True) -> np.ndarray:
        """Возвращает массив долей (n, horizon, J)."""
        J = len(self.columns)
        h = np.arange(1, horizon + 1)
        damp_cum = np.array([np.sum(self.damping ** np.arange(1, k + 1)) for k in h])
        mean = self.level_[None, :] + damp_cum[:, None] * self.slope_[None, :]  # (H, J)
        alr = np.broadcast_to(mean, (n, horizon, J)).copy()
        if noise:
            eff_n = max(1.0, self.halflife * 2.0)
            lvl_err = rng.standard_normal((n, 1, J)) * (self.resid_sd_ / np.sqrt(eff_n))[None, None, :]
            slope_err = rng.standard_normal((n, 1, J)) * (self.resid_sd_ / self.n_obs_)[None, None, :]
            alr += lvl_err + slope_err * damp_cum[None, :, None]
            alr += rng.standard_normal((n, horizon, J)) * self.resid_sd_[None, None, :]
        alr[:, :, self.ref] = 0.0
        e = np.exp(alr - alr.max(axis=2, keepdims=True))
        return e / e.sum(axis=2, keepdims=True)

    def expected(self, horizon: int) -> np.ndarray:
        return self.simulate(horizon, 1, np.random.default_rng(0), noise=False)[0]


# ----------------------------------------------------------------------------------------------
# Основная модель
# ----------------------------------------------------------------------------------------------
@dataclass
class ForecastDraws:
    scenario: str
    periods: pd.PeriodIndex
    draws: dict[str, np.ndarray]          # серия -> (n, H)
    factors: dict[str, np.ndarray]        # фактор -> (n, H)

    def summary(self, qs: tuple[float, ...] = (0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975)) -> pd.DataFrame:
        rows = []
        for name, arr in self.draws.items():
            qv = np.quantile(arr, qs, axis=0)
            for j, p in enumerate(self.periods):
                row = {"scenario": self.scenario, "series": name, "period": p, "mean": arr[:, j].mean()}
                row.update({f"q{int(round(q * 1000)):03d}": qv[i, j] for i, q in enumerate(qs)})
                rows.append(row)
        return pd.DataFrame(rows)


class ForkcastModel:
    """Факторная иерархическая модель. Использование::

        model = ForkcastModel(settings.model, seed=...).fit(wide, macro)
        fc = model.forecast(macro, scenario_paths, horizon=8, n=4000)
    """

    MARKET_NAMES = ["const", "q2", "q3", "q4", "key_rate_eff", "log_investment"]
    SHARE_NAMES = ["const", "q2", "q3", "q4", "trend", "key_rate_eff"]

    def __init__(self, cfg: dict[str, Any], seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self.lags = list(cfg.get("key_rate_lags", [1, 2, 3]))
        self.cap = float(cfg.get("electric_share_cap", 0.85))
        g = cfg.get("gibbs", {})
        self.gibbs = dict(iterations=g.get("iterations", 6000), burn_in=g.get("burn_in", 1000), thin=g.get("thin", 1))
        reg = cfg.get("regime") or {}
        self.regime_start = pd.Period(reg["start"], freq="Q") if reg.get("start") else None
        self.regime_prior = reg.get("prior", {"market": [0.0, 0.25], "share": [0.0, 0.3]})
        mix = cfg.get("segment_mix", {})
        self.mix_halflife = mix.get("halflife_quarters", 3)
        self.mix_damping = mix.get("trend_damping", 0.8)

    # --- матрицы признаков -----------------------------------------------------------------------
    def _regime(self, periods: pd.PeriodIndex) -> np.ndarray:
        return np.asarray([float(p >= self.regime_start) for p in periods])

    def _market_X(self, periods: pd.PeriodIndex, kre: np.ndarray, loginv: np.ndarray) -> np.ndarray:
        n = len(periods)
        cols = [np.ones(n), seasonal_dummies(periods), kre, loginv]
        if self.regime_start is not None:
            cols.append(self._regime(periods))
        return np.column_stack(cols)

    def _share_X(self, periods: pd.PeriodIndex, kre: np.ndarray) -> np.ndarray:
        n = len(periods)
        t = np.asarray([(p - self.t0).n for p in periods], dtype=float)
        cols = [np.ones(n), seasonal_dummies(periods), t, kre]
        if self.regime_start is not None:
            cols.append(self._regime(periods))
        return np.column_stack(cols)

    # --- обучение --------------------------------------------------------------------------------
    def fit(self, wide: pd.DataFrame, macro: pd.DataFrame) -> "ForkcastModel":
        self.hist = wide.copy()
        self.periods = pd.PeriodIndex(wide.index, freq="Q")
        self.t0 = self.periods[0]
        self.macro = macro
        feats = build_features(macro, self.periods, self.lags)
        if feats[["key_rate_eff", "log_investment"]].isna().any().any():
            missing = feats.index[feats[["key_rate_eff", "log_investment"]].isna().any(axis=1)]
            raise ValueError(f"Нет внешних факторов для кварталов: {list(map(str, missing))}")
        self.features = feats
        pri = self.cfg["priors"]

        # 1) рынок
        y = np.log(wide["total_market"].to_numpy(float))
        X = self._market_X(self.periods, feats["key_rate_eff"].to_numpy(), feats["log_investment"].to_numpy())
        pm = pri["market"]
        m_kr, m_inv = pm["key_rate"][0], pm["log_investment"][0]
        icpt = float(y.mean() - m_kr * X[:, 4].mean() - m_inv * X[:, 5].mean())
        mean_m = [icpt, pm["seasonal"][0], pm["seasonal"][0], pm["seasonal"][0], m_kr, m_inv]
        sd_m = [3.0, pm["seasonal"][1], pm["seasonal"][1], pm["seasonal"][1], pm["key_rate"][1], pm["log_investment"][1]]
        names_m = list(self.MARKET_NAMES)
        if self.regime_start is not None:
            mean_m.append(self.regime_prior["market"][0])
            sd_m.append(self.regime_prior["market"][1])
            names_m.append("regime")
        self.market = BayesLinReg(
            prior_mean=np.array(mean_m), prior_sd=np.array(sd_m),
            seed=self.seed, names=names_m, **self.gibbs,
        ).fit(X, y)

        # 2) доля электро
        s = wide["electric_share"].to_numpy(float)
        if np.any(s >= self.cap):
            raise ValueError(
                f"Доля электро в истории ({s.max():.1%}) ≥ потолка насыщения {self.cap:.0%}. "
                "Увеличьте model.electric_share_cap в config/settings.yaml"
            )
        ys = logit(s / self.cap)
        Xs = self._share_X(self.periods, feats["key_rate_eff"].to_numpy())
        ps = pri["share"]
        mean_s = [ys.mean(), ps["seasonal"][0], ps["seasonal"][0], ps["seasonal"][0], ps["trend"][0], ps["key_rate"][0]]
        sd_s = [3.0, ps["seasonal"][1], ps["seasonal"][1], ps["seasonal"][1], ps["trend"][1], ps["key_rate"][1]]
        names_s = list(self.SHARE_NAMES)
        if self.regime_start is not None:
            mean_s.append(self.regime_prior["share"][0])
            sd_s.append(self.regime_prior["share"][1])
            names_s.append("regime")
        self.share = BayesLinReg(
            prior_mean=np.array(mean_s), prior_sd=np.array(sd_s),
            a0=2.0, b0=0.05, seed=self.seed + 1, names=names_s, **self.gibbs,
        ).fit(Xs, ys)

        # 3) структура сегментов
        self.segment_cols = {
            "ice": [c for c in wide.columns if c.startswith("ice_")],
            "electric": [c for c in wide.columns if c.startswith("el_")],
        }
        self.mix = {
            b: SegmentMix(self.mix_halflife, self.mix_damping).fit(wide[cols])
            for b, cols in self.segment_cols.items()
        }
        return self

    # --- симуляция факторов ----------------------------------------------------------------------
    def _simulate_factors(
        self,
        fc_periods: pd.PeriodIndex,
        path: dict[str, np.ndarray],
        n: int,
        rng: np.random.Generator,
        unc: dict[str, dict[str, float]] | None,
    ) -> dict[str, np.ndarray]:
        H = len(fc_periods)
        out = {}
        for col in ("key_rate", "investment_yoy"):
            base = np.asarray(path[col], dtype=float)
            known = np.array([pd.notna(self.macro[col].get(p, np.nan)) for p in fc_periods])
            vals = np.where(known, [self.macro[col].get(p, np.nan) for p in fc_periods], base)
            sim = np.broadcast_to(vals, (n, H)).copy()
            if unc and col in unc:
                u = unc[col]
                steps = np.cumsum(~known)                       # номер «неизвестного» квартала
                z0 = rng.standard_normal((n, 1)) * u["base"]
                rw = np.cumsum(rng.standard_normal((n, H)) * u["step"] * (~known)[None, :], axis=1)
                noise = np.clip(z0 * (steps > 0)[None, :] + rw, -u["max"], u["max"])
                sim = sim + noise
            out[col] = sim
        out["key_rate"] = np.clip(out["key_rate"], 4.0, 30.0)
        return out

    def _features_from_paths(self, fc_periods: pd.PeriodIndex, fac: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        n, H = fac["key_rate"].shape
        # эффективная ставка: история + симулированный путь
        hist_kr = self.macro["key_rate"].dropna()
        kre = np.zeros((n, H))
        for j, p in enumerate(fc_periods):
            vals = []
            for L in self.lags:
                q = p - L
                k = fc_periods.get_loc(q) if q in fc_periods else None
                if k is not None and k < j:
                    vals.append(fac["key_rate"][:, k])
                else:
                    vals.append(np.full(n, float(hist_kr.get(q, hist_kr.iloc[-1]))))
            kre[:, j] = np.mean(vals, axis=0)
        # индекс инвестиций: цепочка от известных уровней
        hist_idx = chain_index(self.macro["investment_yoy"].dropna())
        log_idx = {p: np.full(n, np.log(v)) for p, v in hist_idx.items()}
        loginv = np.zeros((n, H))
        for j, p in enumerate(fc_periods):
            if p in log_idx and pd.notna(self.macro["investment_yoy"].get(p, np.nan)):
                loginv[:, j] = log_idx[p]
            else:
                prev = log_idx.get(p - 4, np.zeros(n))
                log_idx[p] = prev + np.log1p(fac["investment_yoy"][:, j] / 100.0)
                loginv[:, j] = log_idx[p]
        return kre, loginv

    # --- прогноз ---------------------------------------------------------------------------------
    def forecast(
        self,
        path: dict[str, Any],
        horizon: int,
        n: int,
        scenario: str = "scenario",
        factor_uncertainty: dict[str, dict[str, float]] | None = None,
        noise: bool = True,
        param_uncertainty: bool = True,
        seed_offset: int = 0,
    ) -> ForecastDraws:
        rng = np.random.default_rng(self.seed + 1000 + seed_offset)
        last = self.periods[-1]
        fc_periods = pd.period_range(last + 1, last + horizon, freq="Q")
        fac = self._simulate_factors(fc_periods, path, n, rng, factor_uncertainty)
        kre, loginv = self._features_from_paths(fc_periods, fac)

        Xm = np.stack([self._market_X(fc_periods, kre[i], loginv[i]) for i in range(n)])
        log_m = self.market.predict_draws(Xm, rng, n, noise=noise, param_uncertainty=param_uncertainty)
        Xs = np.stack([self._share_X(fc_periods, kre[i]) for i in range(n)])
        lg = self.share.predict_draws(Xs, rng, n, noise=noise, param_uncertainty=param_uncertainty)

        market = np.exp(log_m)
        share = self.cap * expit(lg)
        draws = {
            "total_market": market,
            "total_ice": market * (1 - share),
            "total_electric": market * share,
            "electric_share": share,
        }
        for b, total_key in (("ice", "total_ice"), ("electric", "total_electric")):
            w = self.mix[b].simulate(horizon, n, rng, noise=noise)
            for k, col in enumerate(self.segment_cols[b]):
                draws[col] = draws[total_key] * w[:, :, k]
        factors = {"key_rate": fac["key_rate"], "investment_yoy": fac["investment_yoy"], "key_rate_eff": kre}
        return ForecastDraws(scenario, fc_periods, draws, factors)

    # --- диагностика -----------------------------------------------------------------------------
    def fitted_frame(self) -> pd.DataFrame:
        fm = np.exp(self.market.fitted())
        fs = self.cap * expit(self.share.fitted())
        return pd.DataFrame(
            {
                "total_market": self.hist["total_market"].to_numpy(),
                "total_market_fit": fm,
                "electric_share": self.hist["electric_share"].to_numpy(),
                "electric_share_fit": fs,
            },
            index=self.periods,
        )

    def seasonal_factors(self) -> pd.DataFrame:
        """Сезонные коэффициенты рынка (Q1 = база) в виде % отклонения от среднегодового уровня."""
        b = self.market.beta_[:, 1:4]
        full = np.column_stack([np.zeros(len(b)), b])
        full = full - full.mean(axis=1, keepdims=True)
        eff = np.exp(full) - 1
        return pd.DataFrame(
            {
                "квартал": ["Q1", "Q2", "Q3", "Q4"],
                "эффект_среднее": eff.mean(axis=0),
                "q05": np.quantile(eff, 0.05, axis=0),
                "q95": np.quantile(eff, 0.95, axis=0),
            }
        )
