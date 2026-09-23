"""Архив прогнозов («винтажи») и мониторинг точности по мере поступления фактических данных.

Каждый запуск сохраняет прогноз с меткой последнего квартала данных. Когда приходит факт за
новый квартал, ``evaluate_vintages`` сопоставляет его со всеми ранее сделанными прогнозами:
ошибку, попадание в интервалы и PIT (перцентиль факта в прогнозном распределении). Устойчиво
смещённый PIT — сигнал к перекалибровке модели или пересмотру сценариев.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

KEEP_SERIES = ["total_market", "total_ice", "total_electric", "electric_share"]


def save_vintage(summary: pd.DataFrame, last_period: pd.Period, vintages_dir: Path) -> Path:
    vintages_dir.mkdir(parents=True, exist_ok=True)
    sub = summary[summary["series"].isin(KEEP_SERIES) | summary["series"].str.match(r"^(ice|el)_")].copy()
    sub["period"] = sub["period"].astype(str)
    sub.insert(0, "data_last_period", str(last_period))
    sub.insert(1, "created_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
    path = vintages_dir / f"vintage_{last_period}.csv"
    sub.to_csv(path, index=False)
    return path


def _pit(row: pd.Series, actual: float) -> float:
    qcols = sorted([c for c in row.index if c.startswith("q") and c[1:].isdigit()], key=lambda c: int(c[1:]))
    qs = np.array([int(c[1:]) / 1000 for c in qcols])
    vals = row[qcols].to_numpy(float)
    return float(np.interp(actual, vals, qs, left=0.0, right=1.0))


def evaluate_vintages(wide: pd.DataFrame, vintages_dir: Path) -> pd.DataFrame:
    rows = []
    if not vintages_dir.exists():
        return pd.DataFrame()
    for f in sorted(vintages_dir.glob("vintage_*.csv")):
        v = pd.read_csv(f)
        v["period"] = pd.PeriodIndex(v["period"], freq="Q")
        v = v[v["period"].isin(wide.index)]
        for _, r in v.iterrows():
            if r["series"] not in wide.columns:
                continue
            actual = float(wide.loc[r["period"], r["series"]])
            rows.append({
                "vintage": r["data_last_period"], "scenario": r["scenario"], "series": r["series"],
                "period": str(r["period"]), "h": (r["period"] - pd.Period(r["data_last_period"], "Q")).n,
                "forecast_p50": r["q500"], "actual": actual, "error_%": r["q500"] / actual - 1 if actual else np.nan,
                "in80": bool(r["q100"] <= actual <= r["q900"]), "in95": bool(r["q025"] <= actual <= r["q975"]),
                "PIT": _pit(r, actual),
            })
    return pd.DataFrame(rows)
