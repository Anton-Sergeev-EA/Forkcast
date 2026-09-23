import numpy as np

from forkcast.bayes import BayesLinReg


def test_recovers_coefficients():
    rng = np.random.default_rng(0)
    n = 200
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = X @ np.array([2.0, -0.5]) + rng.normal(scale=0.1, size=n)
    m = BayesLinReg(np.zeros(2), np.full(2, 10.0), iterations=2000, burn_in=500, seed=1).fit(X, y)
    assert np.allclose(m.beta_.mean(0), [2.0, -0.5], atol=0.03)
    assert 0.08 < np.sqrt(m.sigma2_.mean()) < 0.12


def test_prior_dominates_without_data():
    X = np.ones((3, 1))
    y = np.array([0.0, 0.1, -0.1])
    m = BayesLinReg(np.array([5.0]), np.array([0.01]), iterations=1500, burn_in=300, seed=2).fit(X, y)
    assert abs(m.beta_.mean() - 5.0) < 0.05
