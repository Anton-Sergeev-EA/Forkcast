"""Формирование отчётных материалов: графики, Excel, HTML-отчёт, дашборд, презентация."""

from __future__ import annotations

from pathlib import Path


def build_all(res) -> dict[str, Path]:
    from .charts import build_charts
    from .dashboard import build_dashboard
    from .excel import build_excel
    from .html_report import build_html_report
    from .presentation import build_presentation

    rep = res.settings.reports_dir
    figs = build_charts(res, rep / "figures")
    out: dict[str, Path] = {f"fig_{k}": v for k, v in figs.items()}
    out["excel"] = build_excel(res, rep / "forkcast_forecast.xlsx")
    out["html"] = build_html_report(res, figs, rep / "forkcast_report.html")
    out["dashboard"] = build_dashboard(res, rep / "forkcast_dashboard.html")
    out["pptx"] = build_presentation(res, figs, rep / "forkcast_presentation.pptx")
    return out
