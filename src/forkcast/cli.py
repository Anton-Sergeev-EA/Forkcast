"""Командная строка Forkcast.

Примеры::

    forkcast run                              # полный расчёт и все отчёты
    forkcast run --quick                      # быстрый прогон (меньше симуляций)
    forkcast validate --market new.xlsx       # только проверка качества файла
    forkcast update --market new.xlsx         # принять новые данные, пересчитать, оценить точность прошлых прогнозов
    forkcast fetch-macro                      # обновить ключевую ставку и курс с cbr.ru
    forkcast backtest                         # только ретроспективная проверка
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from . import __version__
from .config import load_settings

log = logging.getLogger("forkcast")


def _print_summary(res) -> None:
    from .reporting.text import key_findings

    k = key_findings(res)
    print()
    print("=" * 78)
    print(f" FORKCAST · данные {res.facts['first_period']}–{k['last_period']} · прогноз {k['fc_start']}–{k['fc_8q_end']}")
    print("=" * 78)
    print(f" Рынок {k['y1']} г.: {k['market_y1']} шт. ({k['market_d']} к {k['y0']}); ДВС {k['ice_d']}, электро {k['el_d']}")
    print(f" Прогноз 4 кв. ({k['fc_start']}–{k['fc_4q_end']}): {k['w4_p50']} шт.  [P10–P90: {k['w4_p10']}–{k['w4_p90']}]")
    print(f" Прогноз 8 кв. (до {k['fc_8q_end']}):          {k['w8_p50']} шт.  [P10–P90: {k['w8_p10']}–{k['w8_p90']}]")
    print(f" Доля электро: {k['last_share']} → {k['share_4q']} (4 кв.) → {k['share_8q']} (8 кв.)")
    print(f" Эластичность по ставке: {k['beta_kr']} на +1 п.п.; бэктест MASE {k['bt_mase']} (SNaive {k['bt_mase_sn']})")
    print("=" * 78)


def cmd_run(args) -> int:
    from .pipeline import run, save_outputs

    s = load_settings(market_file=args.market)
    t0 = time.time()
    res = run(s, quick=args.quick)
    out = save_outputs(res, make_reports=not args.no_reports)
    _print_summary(res)
    print("\nФайлы:")
    for k, v in out.items():
        if not k.startswith("fig_"):
            print(f"  {k:10s} {Path(v).relative_to(s.root) if Path(v).is_relative_to(s.root) else v}")
    env = res.manifest.get("environment", {})
    if env.get("exact_match"):
        print("\nОкружение совпадает с requirements-lock.txt: результат воспроизводим бит-в-бит.")
    else:
        diff = ", ".join(f"{k} {v['installed']}≠{v['lock']}" for k, v in list(env.get("mismatches", {}).items())[:5])
        print("\nВНИМАНИЕ: окружение отличается от requirements-lock.txt "
              f"(Python {env.get('python')}{'; ' + diff if diff else ''}). Расчёт корректен, но возможны "
              "расхождения с эталоном в пределах погрешности Монте-Карло (~0,3%). "
              "Для точного совпадения: pip install -r requirements-lock.txt")
    print(f"\nГотово за {time.time() - t0:.1f} с.")
    return 0


def cmd_validate(args) -> int:
    from .data_io import read_market_excel, to_wide

    s = load_settings(market_file=args.market)
    long, dq = read_market_excel(s.market_file, s.segments)
    with pd.option_context("display.max_colwidth", 120, "display.width", 200):
        print(dq.to_frame().to_string(index=False))
    if not long.empty:
        w = to_wide(long)
        print(f"\nПериоды: {w.index[0]}–{w.index[-1]} ({len(w)} кв.), сегментов: {long['segment'].nunique()}")
    return 0 if dq.ok else 1


def _set_market_file(root: Path, rel: str) -> None:
    cfg = root / "config" / "settings.yaml"
    txt = cfg.read_text(encoding="utf-8")
    txt = re.sub(r"(market_file:\s*).*", rf"\g<1>{rel}", txt, count=1)
    cfg.write_text(txt, encoding="utf-8")


def cmd_update(args) -> int:
    """Приём нового файла рынка: проверка → копия в data/raw → пересчёт → оценка прошлых прогнозов."""
    from .data_io import read_market_excel, to_wide
    from .pipeline import run, save_outputs

    s = load_settings()
    src = Path(args.market).resolve()
    long, dq = read_market_excel(src, s.segments)
    if not dq.ok:
        print(dq.to_frame().to_string(index=False))
        print("\nФайл не принят: исправьте ошибки и повторите.")
        return 1
    w = to_wide(long)
    dst = s.root / "data" / "raw" / f"market_forklifts_{w.index[0]}_{w.index[-1]}.xlsx"
    if src != dst.resolve():
        shutil.copy2(src, dst)
    rel = dst.relative_to(s.root).as_posix()
    _set_market_file(s.root, rel)
    print(f"Файл принят: {rel} ({w.index[0]}–{w.index[-1]}). Настройка paths.market_file обновлена.")
    if args.fetch:
        cmd_fetch(args)
    s = load_settings()
    res = run(s, quick=args.quick)
    save_outputs(res, make_reports=not args.no_reports)
    _print_summary(res)
    ev = res.tables.get("vintage_eval")
    if ev is not None and len(ev):
        sub = ev[(ev.scenario == "weighted") & (ev.series.isin(["total_market", "total_ice", "total_electric"]))]
        if len(sub):
            print("\nТочность прошлых прогнозов (взвешенный сценарий):")
            print(sub[["vintage", "series", "period", "h", "forecast_p50", "actual", "error_%", "in80", "PIT"]]
                  .round(3).to_string(index=False))
    return 0


def cmd_fetch(args) -> int:
    from .fetch import update_all

    s = load_settings()
    rep = update_all(s.external_dir)
    for k, v in rep.items():
        print(f"{k:10s} {v}")
    return 0


def cmd_backtest(args) -> int:
    from .backtest import rolling_backtest, score
    from .data_io import read_market_excel, to_wide
    from .macro import load_macro

    s = load_settings(market_file=args.market)
    long, _ = read_market_excel(s.market_file, s.segments)
    w = to_wide(long)
    m = load_macro(s.external_dir)
    bt = rolling_backtest(w, m, s.model, int(s.backtest["min_train"]), int(s.backtest["max_horizon"]),
                          n=1500, seed=int(s.forecast["seed"]))
    print(score(bt, w).round(3).to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="forkcast", description="Прогноз рынка вилочных погрузчиков РФ по сегментам")
    p.add_argument("--version", action="version", version=f"forkcast {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="полный расчёт и отчёты")
    r.add_argument("--market", help="путь к Excel с данными рынка (по умолчанию — из settings.yaml)")
    r.add_argument("--quick", action="store_true", help="быстрый режим (меньше симуляций)")
    r.add_argument("--no-reports", action="store_true", help="без HTML/PPTX/Excel")
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("validate", help="проверить файл рыночных данных")
    v.add_argument("--market")
    v.set_defaults(func=cmd_validate)

    u = sub.add_parser("update", help="принять новые данные и пересчитать прогноз")
    u.add_argument("--market", required=True)
    u.add_argument("--fetch", action="store_true", help="также обновить ставку и курс с cbr.ru")
    u.add_argument("--quick", action="store_true")
    u.add_argument("--no-reports", action="store_true")
    u.set_defaults(func=cmd_update)

    f = sub.add_parser("fetch-macro", help="обновить ключевую ставку и курс с cbr.ru")
    f.set_defaults(func=cmd_fetch)

    b = sub.add_parser("backtest", help="ретроспективная проверка модели")
    b.add_argument("--market")
    b.set_defaults(func=cmd_backtest)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
