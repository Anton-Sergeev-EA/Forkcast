"""Автообновление внешних данных из официальных источников Банка России.

* Ключевая ставка — таблица https://www.cbr.ru/hd_base/KeyRate/ (дневные значения →
  точки изменения → ``key_rate_decisions.csv``);
* Курс USD/RUB — XML-сервис https://www.cbr.ru/scripts/XML_dynamic.asp (R01235) →
  среднемесячные значения → ``usd_rub_monthly.csv``.

Квартальные показатели Росстата (ВВП, инвестиции, ИПЦ) публикуются в PDF/XLSX без стабильного
API, поэтому добавляются в ``macro_quarterly.csv`` вручную (1 строка в квартал) — шаблон и
источники описаны в README. Если сеть недоступна, используется зафиксированный снимок данных.
"""

from __future__ import annotations

import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (Forkcast market model; +https://www.cbr.ru)"}


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (официальный источник)
        return r.read()


def fetch_key_rate_daily(start: str = "01.01.2021", end: str | None = None) -> pd.Series:
    end = end or date.today().strftime("%d.%m.%Y")
    url = (
        "https://www.cbr.ru/hd_base/KeyRate/?UniDbQuery.Posted=True"
        f"&UniDbQuery.From={start}&UniDbQuery.To={end}"
    )
    html = _get(url).decode("utf-8", errors="ignore")
    pairs = re.findall(r"<td>(\d{2}\.\d{2}\.\d{4})</td>\s*<td>([\d,]+)</td>", html)
    if not pairs:
        raise RuntimeError("Не удалось разобрать таблицу ключевой ставки (изменился формат страницы?)")
    s = pd.Series({pd.to_datetime(d, dayfirst=True): float(v.replace(",", ".")) for d, v in pairs}).sort_index()
    return s.rename("key_rate")


def update_key_rate_file(path: Path) -> int:
    """Добавляет новые изменения ставки в журнал решений. Возвращает число добавленных строк."""
    daily = fetch_key_rate_daily()
    changes = daily[daily.diff().fillna(1) != 0]
    cur = pd.read_csv(path, parse_dates=["effective_date"])
    last = cur["effective_date"].max()
    new = changes[changes.index > last]
    if new.empty:
        return 0
    add = pd.DataFrame({
        "effective_date": new.index, "key_rate": new.values, "decision_date": "",
        "comment": "автообновление с cbr.ru",
    })
    out = pd.concat([cur, add], ignore_index=True)
    out["effective_date"] = pd.to_datetime(out["effective_date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
    return len(add)


def fetch_usd_monthly(start: str = "01/01/2023", end: str | None = None) -> pd.Series:
    end = end or date.today().strftime("%d/%m/%Y")
    url = f"https://www.cbr.ru/scripts/XML_dynamic.asp?date_req1={start}&date_req2={end}&VAL_NM_RQ=R01235"
    root = ET.fromstring(_get(url))
    rec = []
    for r in root.findall("Record"):
        d = pd.to_datetime(r.attrib["Date"], dayfirst=True)
        nominal = float(r.find("Nominal").text.replace(",", "."))
        val = float(r.find("Value").text.replace(",", ".")) / nominal
        rec.append((d, val))
    s = pd.Series(dict(rec)).sort_index()
    return s.groupby(s.index.to_period("M")).mean()


def update_usd_file(path: Path) -> int:
    monthly = fetch_usd_monthly()
    cur = pd.read_csv(path)
    cur_map = dict(zip(cur["month"], cur["usd_rub_avg"]))
    today_m = pd.Period(date.today(), "M")
    added = 0
    for m, v in monthly.items():
        if m >= today_m:  # незакрытый месяц не фиксируем
            continue
        key = str(m)
        if key not in cur_map:
            added += 1
        cur_map[key] = round(float(v), 4)
    out = pd.DataFrame(sorted(cur_map.items()), columns=["month", "usd_rub_avg"])
    out.to_csv(path, index=False)
    return added


def update_all(external_dir: Path) -> dict[str, str]:
    report = {}
    for name, fn, fname in (
        ("key_rate", update_key_rate_file, "key_rate_decisions.csv"),
        ("usd_rub", update_usd_file, "usd_rub_monthly.csv"),
    ):
        try:
            n = fn(external_dir / fname)
            report[name] = f"OK: добавлено записей {n}"
        except Exception as exc:  # noqa: BLE001 — сеть/формат: работаем на снимке
            report[name] = f"пропущено ({type(exc).__name__}: {exc}); используется сохранённый снимок"
    return report
