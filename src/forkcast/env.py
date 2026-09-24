"""Проверка окружения: совпадают ли установленные версии библиотек с requirements-lock.txt.

Эталонные результаты в reports/ получены на зафиксированных версиях. Если версии отличаются,
Монте-Карло-оценки могут расходиться с эталоном на доли процента — расчёт корректен, но не
бит-в-бит. Проверка выводит предупреждение и пишет расхождения в манифест.
"""

from __future__ import annotations

import platform
import re
from importlib import metadata
from pathlib import Path

LOCK_PYTHON = "3.12"


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_lock(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, ver = line.split("==", 1)
            pins[_norm(name)] = ver.strip()
    return pins


def installed_versions(names) -> dict[str, str | None]:
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def check_environment(root: Path) -> dict:
    pins = read_lock(root / "requirements-lock.txt")
    inst = installed_versions(pins)
    mismatches = {n: {"lock": v, "installed": inst.get(n)} for n, v in pins.items() if inst.get(n) != v}
    py = ".".join(platform.python_version_tuple()[:2])
    return {
        "python": platform.python_version(),
        "python_matches_lock": py == LOCK_PYTHON,
        "lock_file_found": bool(pins),
        "exact_match": bool(pins) and not mismatches and py == LOCK_PYTHON,
        "mismatches": mismatches,
        "packages": {n: inst.get(n) for n in sorted(pins)},
    }
