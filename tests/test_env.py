from forkcast.env import check_environment, read_lock


def test_lock_file_parsed(settings):
    pins = read_lock(settings.root / "requirements-lock.txt")
    for pkg in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib"):
        assert pkg in pins and pins[pkg].count(".") >= 1


def test_environment_report(settings):
    env = check_environment(settings.root)
    assert env["lock_file_found"]
    assert set(env["packages"]) >= {"numpy", "pandas", "scipy"}
