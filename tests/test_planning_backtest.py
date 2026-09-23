import numpy as np
import pandas as pd

from forkcast.backtest import rolling_backtest, score
from forkcast.model import ForecastDraws
from forkcast.planning import critical_ratio, production_corridor


def test_critical_ratio():
    assert abs(critical_ratio({"underage_cost": 3, "overage_cost": 1}) - 0.75) < 1e-12


def test_corridor_order(settings):
    rng = np.random.default_rng(0)
    per = pd.period_range("2026Q2", periods=4, freq="Q")
    draws = {s["code"]: rng.lognormal(7, 0.3, (500, 4)) for b in settings.segments["segments"].values() for s in b}
    fc = ForecastDraws("t", per, draws, {})
    pl = production_corridor(fc, settings.segments, {}, 4)
    assert (pl["market_p10"] <= pl["market_p50"]).all() and (pl["market_p50"] <= pl["market_p90"]).all()
    assert (pl["plan_newsvendor"] >= pl["silant_demand_p50"]).all()   # q* > 0.5 => план выше медианы


def test_backtest_beats_naive(market, macro, fast_cfg):
    _, _, wide = market
    bt = rolling_backtest(wide, macro, fast_cfg, 8, 4, n=300, seed=1)
    sc = score(bt, wide).set_index(["series", "model"])
    assert sc.loc[("total_market", "Forkcast"), "MASE"] < sc.loc[("total_market", "SNaive"), "MASE"]
