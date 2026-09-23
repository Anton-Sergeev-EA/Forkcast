"""Статические графики (matplotlib) для отчёта и презентации.

Правила оформления: одна ось Y на график (без двойных осей), фиксированный порядок цветов
категорий, тонкие линии 2 px, сдержанная сетка, подписи — цветом текста, а не серии.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e6e5e1"
BLUE = "#2a78d6"      # электро / базовая серия
ORANGE = "#eb6834"    # ДВС
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
NEUTRAL = "#3d3c39"   # рынок в целом
DIV_NEG, DIV_MID, DIV_POS = "#e34948", "#f0efec", "#2a78d6"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlecolor": TEXT, "axes.titlelocation": "left",
    "font.size": 10, "font.family": "DejaVu Sans", "legend.frameon": False, "lines.linewidth": 2,
})


def _thousands(ax, axis="y"):
    fmt = mtick.FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " "))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _pct(ax, axis="y", decimals=0):
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(mtick.PercentFormatter(1.0, decimals=decimals))


def _qlabels(periods) -> list[str]:
    return [f"{p.quarter}кв\n{p.year}" if p.quarter == 1 else f"{p.quarter}кв" for p in periods]


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _note(fig, text: str):
    fig.text(0.01, -0.02, text, fontsize=8, color=MUTED, ha="left", va="top")


# ----------------------------------------------------------------------------------------------
def market_history(res, path: Path) -> Path:
    w = res.wide
    x = np.arange(len(w))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), height_ratios=[2.2, 1], sharex=True)
    ax1.bar(x, w["total_ice"], color=ORANGE, width=0.72, label="ДВС", edgecolor=SURFACE, linewidth=1)
    ax1.bar(x, w["total_electric"], bottom=w["total_ice"], color=BLUE, width=0.72, label="Электро",
            edgecolor=SURFACE, linewidth=1)
    for xi, v in zip(x, w["total_market"]):
        ax1.text(xi, v + 400, f"{v / 1000:.1f}", ha="center", va="bottom", fontsize=8, color=TEXT2)
    ax1.set_title("Рынок вилочных погрузчиков РФ, шт. в квартал (подписи — тыс. шт.)")
    ax1.legend(loc="upper right", ncols=2)
    _thousands(ax1)
    ax2.plot(x, w["electric_share"], color=BLUE, marker="o", ms=5)
    for xi, v in zip(x, w["electric_share"]):
        ax2.text(xi, v + 0.02, f"{v:.0%}", ha="center", fontsize=8, color=TEXT2)
    ax2.set_ylim(0.3, 0.85)
    _pct(ax2)
    ax2.set_title("Доля электропогрузчиков, %", fontsize=11)
    ax2.set_xticks(x, _qlabels(w.index))
    _note(fig, "Источник: данные ЧЗСА (сводные по рынку), расчёт Forkcast")
    return _save(fig, path)


def segment_yoy_heatmap(res, path: Path) -> Path:
    w = res.wide
    segs = [c for c in w.columns if c.startswith(("ice_", "el_"))]
    big = [c for c in segs if w[c].mean() >= 100]
    yoy = (w[big] / w[big].shift(4) - 1).iloc[4:]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
    data = yoy.T.to_numpy(float)
    im = ax.imshow(data, cmap=cmap, norm=TwoSlopeNorm(vmin=-0.8, vcenter=0, vmax=0.8), aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(yoy)), [str(p) for p in yoy.index], rotation=0, fontsize=8)
    ylab = [("ДВС " if c.startswith("ice") else "Эл. ") + res.labels.get(c, c) for c in big]
    ax.set_yticks(range(len(big)), ylab)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:+.0%}", ha="center", va="center", fontsize=7.5,
                    color=TEXT if abs(v) < 0.5 else "white")
    ax.set_title("Изменение продаж к тому же кварталу прошлого года, по сегментам")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, format=mtick.PercentFormatter(1.0))
    cb.outline.set_visible(False)
    _note(fig, "Сегменты со средним объёмом < 100 шт./кв. не показаны (высокий шум округления).")
    return _save(fig, path)


def seasonality(res, path: Path) -> Path:
    t = res.tables["seasonality"].set_index("series")
    series = [("total_market", "Рынок", NEUTRAL), ("total_electric", "Электро", BLUE), ("total_ice", "ДВС", ORANGE)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.26
    for i, (code, name, col) in enumerate(series):
        r = t.loc[code]
        vals = [r[f"Q{k}"] for k in range(1, 5)]
        lo = [r[f"Q{k}"] - r[f"Q{k}_lo"] for k in range(1, 5)]
        hi = [r[f"Q{k}_hi"] - r[f"Q{k}"] for k in range(1, 5)]
        xs = np.arange(4) + (i - 1) * width
        ax.bar(xs, vals, width=width - 0.03, color=col, label=f"{name} (F-тест p={r['p_value']:.3f})")
        ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor=TEXT2, elinewidth=1, capsize=3)
    ax.axhline(0, color=TEXT2, lw=1)
    ax.set_xticks(range(4), ["1 квартал", "2 квартал", "3 квартал", "4 квартал"])
    _pct(ax)
    ax.set_title("Сезонность: отклонение квартала от среднегодового уровня (90% ДИ)")
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, path)


def factors_history(res, path: Path) -> Path:
    f = res.features
    w = res.wide
    x = np.arange(len(w))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    ax.plot(x, f["key_rate"], color=NEUTRAL, label="Ключевая ставка (ср. за кв.)")
    ax.plot(x, f["key_rate_eff"], color=BLUE, ls="--", label="Эффективная (лаг 1–3 кв.)")
    ax.set_title("Ключевая ставка, %")
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    ax.bar(x, f["investment_yoy"], color=[BLUE if v >= 0 else RED for v in f["investment_yoy"]], width=0.7)
    ax.axhline(0, color=TEXT2, lw=1)
    ax.set_title("Инвестиции в ОК, % г/г")
    ax = axes[2]
    y_sa = np.log(w["total_market"].to_numpy(float))
    ax.scatter(f["key_rate_eff"], w["total_market"], s=60, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    for xi, yi, p in zip(f["key_rate_eff"], w["total_market"], w.index):
        if p.quarter == 1 or p == w.index[-1]:
            ax.annotate(str(p), (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=7.5, color=TEXT2)
    b = np.polyfit(f["key_rate_eff"], y_sa, 1)
    xx = np.linspace(f["key_rate_eff"].min(), f["key_rate_eff"].max(), 20)
    ax.plot(xx, np.exp(np.polyval(b, xx)), color=MUTED, lw=1.5, ls=":")
    ax.set_title("Рынок, шт./кв. vs эффективная ставка")
    ax.set_xlabel("Эффективная ставка, %")
    _thousands(ax)
    for a in axes[:2]:
        a.set_xticks(x[::2], [str(p) for p in w.index[::2]], rotation=45, fontsize=8)
    fig.tight_layout()
    _note(fig, "Источники: Банк России, Росстат; данные ЧЗСА.")
    return _save(fig, path)


def decomposition_waterfall(res, path: Path) -> Path:
    t = res.tables.get("decomposition_annual")
    if t is None or t.empty:
        return path
    r = t.iloc[0]
    names = {"key_rate_eff": "Ключевая ставка", "log_investment": "Инвестиции в ОК", "regime": "Структурный сдвиг",
             "остаток": "Необъяснённое"}
    items = [(names[k], float(r[k])) for k in names if k in r and pd.notna(r[k])]
    total = float(r["итого_log"])
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    cum = 0.0
    for i, (nm, v) in enumerate(items):
        ax.bar(i, v, bottom=cum, color=RED if v < 0 else BLUE, width=0.55)
        ax.text(i + 0.31, cum + v / 2, f"{v * 100:+.1f}", ha="left", va="center", fontsize=9, color=TEXT)
        cum += v
    ax.bar(len(items), total, color=NEUTRAL, width=0.55)
    ax.text(len(items) + 0.31, total / 2, f"{total * 100:+.1f}\n({np.exp(total) - 1:+.0%})", ha="left",
            va="center", fontsize=9, color=TEXT)
    ax.set_xlim(-0.5, len(items) + 0.9)
    ax.axhline(0, color=TEXT2, lw=1)
    ax.set_xticks(range(len(items) + 1), [n.replace(" ", "\n", 1) for n, _ in items] + ["Итого"])
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
    y0, y1 = res.facts["last_full_year"] - 1, res.facts["last_full_year"]
    ax.set_title(f"Что обрушило рынок: вклад факторов в изменение {y1} к {y0}, лог-пункты")
    _note(fig, "Лог-пункты ≈ % изменения при малых величинах; вклады аддитивны. Оценка по апостериорным средним модели.")
    return _save(fig, path)


def fan_chart(res, series: str, path: Path, title: str, pct: bool = False, horizon: int | None = None) -> Path:
    H = horizon or res.long_h
    w = res.wide
    fc = res.weighted
    hist_x = np.arange(len(w))
    fx = np.arange(len(w), len(w) + H)
    d = fc.draws[series][:, :H]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(hist_x, w[series], color=NEUTRAL, marker="o", ms=4, label="Факт")
    bands = [(0.025, 0.975, 0.15, "95%"), (0.1, 0.9, 0.25, "80%"), (0.25, 0.75, 0.4, "50%")]
    for lo, hi, a, lab in bands:
        ax.fill_between(fx, np.quantile(d, lo, axis=0), np.quantile(d, hi, axis=0), color=BLUE, alpha=a, lw=0,
                        label=f"Интервал {lab} (взвешенный)")
    med = np.median(d, axis=0)
    ax.plot(np.r_[hist_x[-1], fx], np.r_[w[series].iloc[-1], med], color=BLUE, lw=2, label="Медиана (взвешенный)")
    for sc in res.scenarios:
        m = np.median(res.forecasts[sc.key].draws[series][:, :H], axis=0)
        ax.plot(fx, m, color=sc.color, lw=1.4, ls="--", label=sc.title.split("(")[0].strip())
    ax.axvline(len(w) - 0.5, color=MUTED, lw=1, ls=":")
    ax.text(len(w) - 0.4, ax.get_ylim()[1], " прогноз →", color=MUTED, fontsize=8, va="top")
    allp = list(w.index) + list(fc.periods[:H])
    ax.set_xticks(range(len(allp)), _qlabels(pd.PeriodIndex(allp)), fontsize=8)
    if pct:
        _pct(ax)
    else:
        _thousands(ax)
    ax.set_title(title)
    ax.legend(fontsize=8, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    return _save(fig, path)


def scenario_bars(res, path: Path) -> Path:
    t = res.tables["scenario_totals"]
    t = t[t.series == "total_market"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    ref = res.facts["last_full_year_total"]
    for ax, key, ttl in ((axes[0], "4q", "Краткосрочный горизонт (4 кв.)"), (axes[1], "8q", "Долгосрочный горизонт (8 кв.)")):
        vals = t[f"{key}_p50"].to_numpy()
        lo = vals - t[f"{key}_p10"].to_numpy()
        hi = t[f"{key}_p90"].to_numpy() - vals
        cols = [res.scenario(k).color if k != "weighted" else NEUTRAL for k in t["scenario"]]
        y = np.arange(len(t))
        ax.barh(y, vals, color=cols, height=0.6)
        ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor=TEXT2, capsize=3, elinewidth=1)
        for yi, v in zip(y, vals):
            ax.text(v * 0.03, yi, f"{v / 1000:.1f} тыс.", fontsize=8.5, color="white", ha="left", va="center",
                    fontweight="bold")
        names = [("Взвешенный" if k == "weighted" else res.scenario(k).title.split("(")[0].strip()) for k in t["scenario"]]
        ax.set_yticks(y, names if ax is axes[0] else [""] * len(y))
        ax.invert_yaxis()
        if key == "4q":
            ax.axvline(ref, color=MUTED, ls=":", lw=1.2)
            ax.text(ref, len(t) - 0.4, f" {res.facts['last_full_year']} г. факт", fontsize=8, color=MUTED)
        _thousands(ax, "x")
        ax.set_title(ttl, fontsize=11)
    fig.suptitle("Объём рынка по сценариям, шт. за период (медиана, интервал P10–P90)", x=0.01, ha="left",
                 fontweight="bold", color=TEXT)
    fig.tight_layout()
    return _save(fig, path)


def blocks_fan(res, path: Path) -> Path:
    H = res.long_h
    w = res.wide
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, ser, col, ttl in ((axes[0], "total_ice", ORANGE, "ДВС"), (axes[1], "total_electric", BLUE, "Электро")):
        d = res.weighted.draws[ser][:, :H]
        hx = np.arange(len(w))
        fx = np.arange(len(w), len(w) + H)
        ax.plot(hx, w[ser], color=NEUTRAL, marker="o", ms=3.5)
        ax.fill_between(fx, np.quantile(d, 0.1, 0), np.quantile(d, 0.9, 0), color=col, alpha=0.25, lw=0, label="80%")
        ax.fill_between(fx, np.quantile(d, 0.25, 0), np.quantile(d, 0.75, 0), color=col, alpha=0.4, lw=0, label="50%")
        ax.plot(np.r_[hx[-1], fx], np.r_[w[ser].iloc[-1], np.median(d, 0)], color=col)
        allp = list(w.index) + list(res.weighted.periods[:H])
        ax.set_xticks(range(0, len(allp), 2), [str(p) for p in allp[::2]], rotation=45, fontsize=8)
        ax.set_title(f"{ttl}: факт и прогноз, шт./кв.")
        _thousands(ax)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def silant_segments(res, path: Path) -> Path:
    H = res.long_h
    w = res.wide
    codes = [s["code"] for b in res.settings.segments["segments"].values() for s in b if s.get("silant")]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    for ax, code in zip(axes.flat, codes):
        col = ORANGE if code.startswith("ice") else BLUE
        d = res.weighted.draws[code][:, :H]
        hx = np.arange(len(w))
        fx = np.arange(len(w), len(w) + H)
        ax.plot(hx, w[code], color=NEUTRAL, lw=1.5)
        ax.fill_between(fx, np.quantile(d, 0.1, 0), np.quantile(d, 0.9, 0), color=col, alpha=0.25, lw=0)
        ax.plot(np.r_[hx[-1], fx], np.r_[w[code].iloc[-1], np.median(d, 0)], color=col)
        ax.set_title(("ДВС " if code.startswith("ice") else "Электро ") + res.labels.get(code, code), fontsize=10)
        _thousands(ax)
        ax.set_ylim(bottom=0)
    allp = list(w.index) + list(res.weighted.periods[:H])
    for ax in axes[1]:
        ax.set_xticks(range(0, len(allp), 4), [str(p) for p in allp[::4]], fontsize=8)
    fig.suptitle("Сегменты линейки СИЛАНТ (1–5 т): факт и прогноз, медиана и 80% интервал", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def uncertainty_budget(res, path: Path) -> Path:
    t = res.tables["uncertainty_budget"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(t))
    bottom = np.zeros(len(t))
    for col, c in (("параметры модели", VIOLET), ("случайный шум", MUTED), ("путь внешних факторов", AQUA)):
        ax.bar(x, t[col], bottom=bottom, color=c, label=col, width=0.7, edgecolor=SURFACE, linewidth=2)
        bottom += t[col].to_numpy()
    ax.set_xticks(x, [str(p) for p in t["period"]], fontsize=8)
    _pct(ax)
    ax.set_title("Бюджет неопределённости прогноза рынка (сценарий «Консенсус»): доля дисперсии")
    ax.legend(ncols=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return _save(fig, path)


def backtest(res, path: Path) -> Path:
    sc = res.tables["backtest_scores"]
    sub = sc[sc.series.isin(["total_market", "total_ice", "total_electric"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    models = ["Forkcast", "SNaive", "ETS", "SeasonalMean"]
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    series = [("total_market", "Рынок"), ("total_ice", "ДВС"), ("total_electric", "Электро")]
    width = 0.2
    for i, (m, c) in enumerate(zip(models, colors)):
        vals = [float(sub[(sub.series == s) & (sub.model == m)]["MASE"].iloc[0]) if len(sub[(sub.series == s) & (sub.model == m)]) else np.nan for s, _ in series]
        ax.bar(np.arange(3) + (i - 1.5) * width, vals, width=width - 0.02, color=c, label=m)
    ax.axhline(1, color=TEXT2, lw=1, ls=":")
    ax.set_xticks(range(3), [n for _, n in series])
    ax.set_title("MASE в бэктесте (меньше — лучше)")
    ax.legend(fontsize=8)
    ax = axes[1]
    bt = res.tables["backtest"]
    b = bt[(bt.series == "total_market") & (bt.model == "Forkcast") & (bt.h == 1)]
    x = np.arange(len(b))
    ax.fill_between(x, b["lo80"], b["hi80"], color=BLUE, alpha=0.25, lw=0, label="80% интервал")
    ax.plot(x, b["forecast"], color=BLUE, marker="o", ms=5, label="Прогноз на 1 кв. вперёд")
    ax.plot(x, b["actual"], color=NEUTRAL, marker="s", ms=5, label="Факт")
    ax.set_xticks(x, [str(p) for p in b["period"]], fontsize=8)
    _thousands(ax)
    ax.set_title("Рынок: прогноз на 1 кв. вперёд и факт")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def key_rate_ladder(res, path: Path) -> Path:
    t = res.tables["key_rate_ladder"]
    years = sorted({c.split("_")[0] for c in t.columns if c.endswith("_median")})
    if not years:
        return path
    y = years[0]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.fill_between(t["key_rate"], t[f"{y}_p10"], t[f"{y}_p90"], color=BLUE, alpha=0.2, lw=0, label="P10–P90")
    ax.plot(t["key_rate"], t[f"{y}_median"], color=BLUE, marker="o", ms=6, label="Медиана")
    ref = res.facts["last_full_year_total"]
    ax.axhline(ref, color=MUTED, ls=":", lw=1.2)
    ax.text(t["key_rate"].min(), ref, f" уровень {res.facts['last_full_year']} г.: {ref / 1000:.1f} тыс.", fontsize=8,
            color=TEXT2, va="bottom")
    ax.invert_xaxis()
    ax.set_xlabel("Ключевая ставка, % (постоянная с ближайшего квартала без факта)")
    _thousands(ax)
    ax.set_title(f"«Лестница ставок»: рынок {y} г., шт., при разных уровнях ключевой ставки")
    ax.legend(fontsize=8)
    return _save(fig, path)


def spec_comparison(res, path: Path) -> Path:
    t = res.tables["spec_comparison"].sort_values("LOO_RMSE", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    cols = [BLUE if "ИТОГОВАЯ" in s else "#b9b8b3" for s in t["спецификация"]]
    ax.barh(range(len(t)), t["LOO_RMSE"], color=cols, height=0.65)
    ax.set_yticks(range(len(t)), t["спецификация"], fontsize=8.5)
    for i, v in enumerate(t["LOO_RMSE"]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8, color=TEXT2)
    ax.set_title("Отбор факторов: ошибка кросс-валидации LOO (log), меньше — лучше")
    return _save(fig, path)


def build_charts(res, fig_dir: Path) -> dict[str, Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "market_history": market_history(res, fig_dir / "01_market_history.png"),
        "segment_heatmap": segment_yoy_heatmap(res, fig_dir / "02_segment_yoy_heatmap.png"),
        "seasonality": seasonality(res, fig_dir / "03_seasonality.png"),
        "factors": factors_history(res, fig_dir / "04_factors.png"),
        "spec": spec_comparison(res, fig_dir / "05_factor_selection.png"),
        "decomposition": decomposition_waterfall(res, fig_dir / "06_decomposition.png"),
        "fan_market": fan_chart(res, "total_market", fig_dir / "07_fan_market.png",
                                "Прогноз рынка вилочных погрузчиков, шт./кв.: сценарии и интервалы неопределённости"),
        "fan_share": fan_chart(res, "electric_share", fig_dir / "08_fan_electric_share.png",
                               "Прогноз доли электропогрузчиков", pct=True),
        "blocks": blocks_fan(res, fig_dir / "09_blocks.png"),
        "scenarios": scenario_bars(res, fig_dir / "10_scenarios.png"),
        "silant": silant_segments(res, fig_dir / "11_silant_segments.png"),
        "uncertainty": uncertainty_budget(res, fig_dir / "12_uncertainty_budget.png"),
        "backtest": backtest(res, fig_dir / "13_backtest.png"),
        "ladder": key_rate_ladder(res, fig_dir / "14_key_rate_ladder.png"),
    }
    return out
