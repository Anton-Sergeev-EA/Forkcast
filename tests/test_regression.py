"""Регрессионный тест: ключевые результаты не уходят от эталона (reports/, seed 20260923).

Допуски учитывают быстрый режим (меньше симуляций) и разные версии библиотек: тест ловит
ошибки в коде и данных, а не шум Монте-Карло.
"""

import numpy as np

from forkcast.pipeline import run

REFERENCE = {
    "market_2025": 82_480,          # факт, точное значение
    "weighted_4q_p50": 82_000,      # эталон ~81 950–82 020
    "weighted_8q_p50": 169_900,     # эталон ~169 700–169 960
    "beta_key_rate": -0.021,
    "share_8q_p50": 0.79,
}


def test_key_results_match_reference(settings):
    res = run(settings, quick=True)
    st = res.tables["scenario_totals"]
    w = st[st.scenario == "weighted"].set_index("series")
    assert res.facts["last_full_year_total"] == REFERENCE["market_2025"]
    assert abs(w.loc["total_market", "4q_p50"] / REFERENCE["weighted_4q_p50"] - 1) < 0.03
    assert abs(w.loc["total_market", "8q_p50"] / REFERENCE["weighted_8q_p50"] - 1) < 0.04
    assert abs(res.facts["beta_kr"] - REFERENCE["beta_key_rate"]) < 0.006
    assert abs(w.loc["electric_share", "8q_p50"] - REFERENCE["share_8q_p50"]) < 0.02
    # иерархическая согласованность взвешенного прогноза
    d = res.weighted.draws
    assert np.allclose(d["total_ice"] + d["total_electric"], d["total_market"])
