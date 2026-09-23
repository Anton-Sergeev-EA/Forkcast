"""Загрузка конфигурации проекта (YAML) и разрешение путей."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def find_project_root(start: Path | None = None) -> Path:
    """Ищет корень проекта (каталог с ``config/settings.yaml``), поднимаясь вверх от ``start``.

    Переменная окружения ``FORKCAST_ROOT`` имеет приоритет.
    """
    env = os.environ.get("FORKCAST_ROOT")
    if env:
        return Path(env).resolve()
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "config" / "settings.yaml").exists():
            return candidate
    # запасной вариант — установка из исходников (src/forkcast/config.py -> корень)
    pkg_root = Path(__file__).resolve().parents[2]
    if (pkg_root / "config" / "settings.yaml").exists():
        return pkg_root
    raise FileNotFoundError(
        "Не найден config/settings.yaml. Запустите команду из корня проекта или задайте FORKCAST_ROOT."
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class Settings:
    """Все настройки проекта в одном объекте."""

    root: Path
    raw: dict[str, Any]
    scenarios: dict[str, Any]
    segments: dict[str, Any]
    overrides: dict[str, Any] = field(default_factory=dict)

    # --- пути ---------------------------------------------------------------------------------
    def path(self, key: str) -> Path:
        value = self.overrides.get(key) or self.raw["paths"][key]
        p = Path(value)
        return p if p.is_absolute() else self.root / p

    @property
    def market_file(self) -> Path:
        return self.path("market_file")

    @property
    def external_dir(self) -> Path:
        return self.path("external_dir")

    @property
    def processed_dir(self) -> Path:
        return self.path("processed_dir")

    @property
    def reports_dir(self) -> Path:
        return self.path("reports_dir")

    @property
    def vintages_dir(self) -> Path:
        return self.path("vintages_dir")

    # --- параметры ----------------------------------------------------------------------------
    @property
    def forecast(self) -> dict[str, Any]:
        return self.raw["forecast"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def backtest(self) -> dict[str, Any]:
        return self.raw["backtest"]


def load_settings(root: Path | None = None, **overrides: Any) -> Settings:
    """Читает ``settings.yaml``, ``scenarios.yaml`` и ``segments.yaml``."""
    root = find_project_root(root)
    cfg = root / "config"
    return Settings(
        root=root,
        raw=_load_yaml(cfg / "settings.yaml"),
        scenarios=_load_yaml(cfg / "scenarios.yaml"),
        segments=_load_yaml(cfg / "segments.yaml"),
        overrides={k: v for k, v in overrides.items() if v is not None},
    )
