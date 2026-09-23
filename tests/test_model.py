import numpy as np
import pandas as pd

from forkcast.model import ForkcastModel
from forkcast.scenarios import align_path, load_scenarios, mixture, run_scenarios


def _fit(market, macro, cfg, seed=7):
    return ForkcastModel(cfg, seed=seed).fit(market[2], macro)


def test_coherence(market, macro, fast_cfg, settings):
    model = _fit(market, macro, fast_cfg)
    sc = load_scenarios(settings.scenarios)[0]
    fc_p = pd.period_range(market[2].index[-1] + 1, periods=8, freq="Q")
    fc = model.forecast(align_path(sc, fc_p), 8, 300, factor_uncertainty=settings.scenarios["factor_uncertainty"])
    d = fc.draws
    assert np.allclose(d["total_ice"] + d["total_electric"], d["total_market"])
    ice = sum(d[c] for c in model.segment_cols["ice"])
    el = sum(d[c] for c in model.segment_cols["electric"])
    assert np.allclose(ice, d["total_ice"]) and np.allclose(el, d["total_electric"])
    assert (d["electric_share"] > 0).all() and (d["electric_share"] < model.cap).all()
    assert all((v >= 0).all() for v in d.values())


def test_reproducible(market, macro, fast_cfg, settings):
    sc = load_scenarios(settings.scenarios)[0]
    fc_p = pd.period_range(market[2].index[-1] + 1, periods=4, freq="Q")
    a = _fit(market, macro, fast_cfg, 3).forecast(align_path(sc, fc_p), 4, 200).draws["total_market"]
    b = _fit(market, macro, fast_cfg, 3).forecast(align_path(sc, fc_p), 4, 200).draws["total_market"]
    assert np.array_equal(a, b)


def test_economic_signs(market, macro, fast_cfg):
    m = _fit(market, macro, fast_cfg)
    s = m.market.summary().set_index("коэффициент")
    assert s.loc["key_rate_eff", "апостериори_среднее"] < 0       # дорогие деньги снижают продажи
    assert s.loc["log_investment", "апостериори_среднее"] > 0     # инвестиции поддерживают спрос
    assert m.share.summary().set_index("коэффициент").loc["trend", "P(>0)"] > 0.95  # электрификация


def test_lower_rate_means_bigger_market(market, macro, fast_cfg, settings):
    m = _fit(market, macro, fast_cfg)
    fc_p = pd.period_range(market[2].index[-1] + 1, periods=8, freq="Q")
    path = align_path(load_scenarios(settings.scenarios)[0], fc_p)
    lo = dict(path, key_rate=np.full(8, 8.0))
    hi = dict(path, key_rate=np.full(8, 16.0))
    a = np.median(m.forecast(lo, 8, 400, noise=False).draws["total_market"][:, 4:].sum(1))
    b = np.median(m.forecast(hi, 8, 400, noise=False).draws["total_market"][:, 4:].sum(1))
    assert a > b


def test_mixture_weights(market, macro, fast_cfg, settings):
    m = _fit(market, macro, fast_cfg)
    scs = load_scenarios(settings.scenarios)
    assert abs(sum(s.weight for s in scs) - 1) < 1e-9
    res = run_scenarios(m, scs, 4, 200, None)
    mx = mixture(res, scs)
    assert mx.draws["total_market"].shape == (200, 4)
