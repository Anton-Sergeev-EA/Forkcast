import numpy as np
import pandas as pd

from forkcast.macro import build_features, chain_index, effective_rate


def test_key_rate_quarterly(macro):
    kr = macro["key_rate"]
    assert kr[pd.Period("2023Q1", "Q")] == 7.5
    assert kr[pd.Period("2025Q1", "Q")] == 21.0
    assert 20.3 < kr[pd.Period("2024Q4", "Q")] < 20.5


def test_chain_index():
    s = pd.Series([10.0, 0.0, 0.0, 0.0, -10.0], index=pd.period_range("2023Q1", periods=5, freq="Q"))
    idx = chain_index(s)
    assert np.isclose(idx.iloc[0], 1.1)
    assert np.isclose(idx.iloc[4], 1.1 * 0.9)


def test_effective_rate():
    kr = pd.Series([10.0, 12.0, 14.0, 16.0], index=pd.period_range("2024Q1", periods=4, freq="Q"))
    eff = effective_rate(kr, [pd.Period("2024Q4", "Q")], [1, 2, 3])
    assert np.isclose(eff.iloc[0], 12.0)


def test_features_complete(macro, market):
    _, _, wide = market
    f = build_features(macro, wide.index, [1, 2, 3])
    assert not f[["key_rate_eff", "log_investment"]].isna().any().any()
