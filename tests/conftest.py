import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forkcast.config import load_settings  # noqa: E402
from forkcast.data_io import read_market_excel, to_wide  # noqa: E402
from forkcast.macro import load_macro  # noqa: E402


@pytest.fixture(scope="session")
def settings():
    return load_settings(ROOT)


@pytest.fixture(scope="session")
def market(settings):
    long, dq = read_market_excel(settings.market_file, settings.segments)
    return long, dq, to_wide(long)


@pytest.fixture(scope="session")
def macro(settings):
    return load_macro(settings.external_dir)


@pytest.fixture(scope="session")
def fast_cfg(settings):
    cfg = dict(settings.model)
    cfg["gibbs"] = {"iterations": 1500, "burn_in": 300, "thin": 1}
    return cfg
