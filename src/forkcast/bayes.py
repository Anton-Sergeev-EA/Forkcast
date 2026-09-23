"""Байесовская линейная регрессия с независимыми нормальными априорными распределениями.

Почему байесовский подход
-------------------------
Ряд содержит 13 квартальных наблюдений, а модели нужно оценить сезонность и влияние 2–3 факторов.
Классический МНК на такой выборке неустойчив (коэффициенты «прыгают» при добавлении одной точки).
Слабоинформативные экономически обоснованные априорные распределения стабилизируют оценки,
а апостериорное распределение параметров напрямую даёт *неопределённость параметров*, которая
затем честно переносится в прогнозные интервалы.

Модель: y = Xβ + ε, ε ~ N(0, σ²);  β_j ~ N(m_j, s_j²) независимо;  σ² ~ InvGamma(a0, b0).
Оценивание — сэмплер Гиббса (полусопряжённая схема, точные условные распределения).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BayesLinReg:
    prior_mean: np.ndarray
    prior_sd: np.ndarray
    a0: float = 2.0
    b0: float = 0.02
    iterations: int = 6000
    burn_in: int = 1000
    thin: int = 1
    seed: int = 0
    names: list[str] = field(default_factory=list)

    beta_: np.ndarray | None = None      # (S, k) апостериорные выборки коэффициентов
    sigma2_: np.ndarray | None = None    # (S,)   апостериорные выборки дисперсии шума
    X_: np.ndarray | None = None
    y_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesLinReg":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, k = X.shape
        rng = np.random.default_rng(self.seed)
        m0 = np.asarray(self.prior_mean, dtype=float)
        V0_inv = np.diag(1.0 / np.asarray(self.prior_sd, dtype=float) ** 2)
        XtX, Xty = X.T @ X, X.T @ y

        # старт — из МНК с гребневой регуляризацией
        beta = np.linalg.solve(XtX + V0_inv * 1e-3, Xty + V0_inv @ m0 * 1e-3)
        keep_b, keep_s = [], []
        a_n = self.a0 + n / 2.0
        for it in range(self.iterations):
            resid = y - X @ beta
            b_n = self.b0 + 0.5 * float(resid @ resid)
            sigma2 = 1.0 / rng.gamma(a_n, 1.0 / b_n)
            prec = XtX / sigma2 + V0_inv
            cov = np.linalg.inv(prec)
            cov = 0.5 * (cov + cov.T)
            mean = cov @ (Xty / sigma2 + V0_inv @ m0)
            beta = rng.multivariate_normal(mean, cov, method="cholesky")
            if it >= self.burn_in and (it - self.burn_in) % self.thin == 0:
                keep_b.append(beta)
                keep_s.append(sigma2)
        self.beta_ = np.asarray(keep_b)
        self.sigma2_ = np.asarray(keep_s)
        self.X_, self.y_ = X, y
        return self

    # ------------------------------------------------------------------------------------------
    @property
    def n_draws(self) -> int:
        return 0 if self.beta_ is None else self.beta_.shape[0]

    def predict_draws(
        self,
        X_new: np.ndarray,
        rng: np.random.Generator,
        n: int,
        noise: bool = True,
        param_uncertainty: bool = True,
    ) -> np.ndarray:
        """Выборка из апостериорного прогнозного распределения.

        X_new: (n, m, k) — матрица признаков для каждой симуляции (факторы могут различаться
        между симуляциями из-за неопределённости сценарного пути) либо (m, k).
        Возвращает массив (n, m).
        """
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 2:
            X_new = np.broadcast_to(X_new, (n, *X_new.shape))
        if param_uncertainty:
            idx = rng.integers(0, self.n_draws, size=n)
            beta = self.beta_[idx]
            sigma = np.sqrt(self.sigma2_[idx])
        else:
            beta = np.broadcast_to(self.beta_.mean(axis=0), (n, self.beta_.shape[1]))
            sigma = np.full(n, np.sqrt(self.sigma2_.mean()))
        mu = np.einsum("nmk,nk->nm", X_new, beta)
        if noise:
            mu = mu + sigma[:, None] * rng.standard_normal(mu.shape)
        return mu

    def fitted(self) -> np.ndarray:
        return self.X_ @ self.beta_.mean(axis=0)

    def residuals(self) -> np.ndarray:
        return self.y_ - self.fitted()

    def r2(self) -> float:
        r = self.residuals()
        return 1.0 - float(r @ r) / float(((self.y_ - self.y_.mean()) ** 2).sum())

    def summary(self) -> pd.DataFrame:
        b = self.beta_
        names = self.names or [f"b{i}" for i in range(b.shape[1])]
        df = pd.DataFrame(
            {
                "коэффициент": names,
                "априори_среднее": self.prior_mean,
                "априори_sd": self.prior_sd,
                "апостериори_среднее": b.mean(axis=0),
                "апостериори_sd": b.std(axis=0),
                "q05": np.quantile(b, 0.05, axis=0),
                "q95": np.quantile(b, 0.95, axis=0),
                "P(>0)": (b > 0).mean(axis=0),
            }
        )
        # степень «обучения на данных»: 1 − sd_post/sd_prior (0 — данные ничего не добавили)
        df["информативность_данных"] = 1.0 - df["апостериори_sd"] / df["априори_sd"]
        return df

    def loo_rmse(self) -> float:
        """Приближённая LOO-ошибка через переобучение (дёшево: n ≤ 20)."""
        X, y = self.X_, self.y_
        errs = []
        for i in range(len(y)):
            mask = np.arange(len(y)) != i
            m = BayesLinReg(
                self.prior_mean, self.prior_sd, self.a0, self.b0, iterations=1500, burn_in=300,
                seed=self.seed + i,
            ).fit(X[mask], y[mask])
            errs.append(y[i] - X[i] @ m.beta_.mean(axis=0))
        return float(np.sqrt(np.mean(np.square(errs))))
