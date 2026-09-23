"""Перевод прогноза рынка на язык производственного плана СИЛАНТ.

Для сегментов линейки СИЛАНТ (1,5–5 т, ДВС и электро) строится «коридор плана»:
рыночный объём P10/P50/P90 × целевая доля СИЛАНТ, и рекомендуемый объём выпуска по правилу
газетчика (newsvendor): Q* = F⁻¹(Cu / (Cu + Co)), где F — прогнозное распределение спроса на
продукцию СИЛАНТ, Cu — потери от недопроизводства (упущенная маржа), Co — потери от
перепроизводства (склад, омертвление оборотного капитала). Такое правило оптимально при
асимметричных потерях и напрямую использует оценку неопределённости прогноза.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .model import ForecastDraws


def critical_ratio(cfg: dict[str, Any]) -> float:
    cu, co = float(cfg.get("underage_cost", 0.25)), float(cfg.get("overage_cost", 0.10))
    return cu / (cu + co)


def production_corridor(
    fc: ForecastDraws, segments_cfg: dict[str, Any], labels: dict[str, str], horizon: int | None = None
) -> pd.DataFrame:
    plan_cfg = segments_cfg.get("planning", {})
    q_star = critical_ratio(plan_cfg)
    shares = plan_cfg.get("target_share", {})
    default_share = float(shares.get("default", 0.05))
    rows = []
    H = horizon or len(fc.periods)
    blocks = segments_cfg["segments"]
    for block, segs in blocks.items():
        for seg in segs:
            if not seg.get("silant"):
                continue
            code = seg["code"]
            if code not in fc.draws:
                continue
            share = float(shares.get(code, default_share))
            d = fc.draws[code][:, :H]
            for j, p in enumerate(fc.periods[:H]):
                market = d[:, j]
                demand = market * share
                plan = float(np.quantile(demand, q_star))
                rows.append({
                    "block": block, "segment": code, "label": labels.get(code, seg["label"]),
                    "period": p, "market_p10": float(np.quantile(market, 0.1)),
                    "market_p50": float(np.median(market)), "market_p90": float(np.quantile(market, 0.9)),
                    "silant_share": share, "silant_demand_p50": float(np.median(demand)),
                    "plan_newsvendor": plan, "critical_ratio": q_star,
                    "P(излишек)": float((demand < plan).mean()),
                })
            tot = (d.sum(axis=1)) * share
            rows.append({
                "block": block, "segment": code, "label": labels.get(code, seg["label"]),
                "period": f"Σ {fc.periods[0]}–{fc.periods[H - 1]}",
                "market_p10": float(np.quantile(d.sum(axis=1), 0.1)),
                "market_p50": float(np.median(d.sum(axis=1))),
                "market_p90": float(np.quantile(d.sum(axis=1), 0.9)),
                "silant_share": share, "silant_demand_p50": float(np.median(tot)),
                "plan_newsvendor": float(np.quantile(tot, q_star)), "critical_ratio": q_star,
                "P(излишек)": float((tot < np.quantile(tot, q_star)).mean()),
            })
    return pd.DataFrame(rows)


def silant_addressable_market(fc: ForecastDraws, segments_cfg: dict[str, Any]) -> pd.DataFrame:
    """Адресуемый рынок СИЛАНТ (сумма сегментов 1,5–5 т) по кварталам и его доля в рынке."""
    codes = [s["code"] for segs in segments_cfg["segments"].values() for s in segs if s.get("silant")]
    codes = [c for c in codes if c in fc.draws]
    sam = sum(fc.draws[c] for c in codes)
    total = fc.draws["total_market"]
    return pd.DataFrame({
        "period": fc.periods,
        "SAM_p10": np.quantile(sam, 0.1, axis=0), "SAM_p50": np.median(sam, axis=0),
        "SAM_p90": np.quantile(sam, 0.9, axis=0),
        "SAM_share_of_market": np.median(sam / total, axis=0),
    })
