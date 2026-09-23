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
    pdf = _pptx_to_pdf(out["pptx"])
    if pdf:
        out["pdf"] = pdf
    return out


def _pptx_to_pdf(pptx: Path) -> Path | None:
    """PDF-версия презентации, если установлен LibreOffice (необязательно)."""
    import shutil
    import subprocess

    exe = shutil.which("soffice") or shutil.which("libreoffice")
    if not exe:
        return None
    try:
        subprocess.run([exe, "--headless", "--convert-to", "pdf", "--outdir", str(pptx.parent), str(pptx)],
                       check=True, capture_output=True, timeout=180)
    except Exception:  # noqa: BLE001 — PDF опционален
        return None
    pdf = pptx.with_suffix(".pdf")
    return pdf if pdf.exists() else None
