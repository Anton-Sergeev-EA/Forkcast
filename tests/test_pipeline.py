from forkcast.config import load_settings
from forkcast.pipeline import run, save_outputs


def test_end_to_end_quick(settings, tmp_path):
    s = load_settings(settings.root, processed_dir=str(tmp_path / "p"), reports_dir=str(tmp_path / "r"),
                      vintages_dir=str(tmp_path / "v"))
    res = run(s, quick=True)
    out = save_outputs(res, make_reports=True)
    for key in ("dataset", "vintage", "manifest", "excel", "html", "dashboard", "pptx"):
        assert out[key].exists() and out[key].stat().st_size > 0, key
    st = res.tables["scenario_totals"]
    w = st[(st.scenario == "weighted") & (st.series == "total_market")].iloc[0]
    assert w["4q_p10"] < w["4q_p50"] < w["4q_p90"]
    assert res.manifest["market_sha256"]
