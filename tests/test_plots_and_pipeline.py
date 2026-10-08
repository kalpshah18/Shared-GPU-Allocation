"""
tests/test_plots_and_pipeline.py — Owner: Kalp Shah

Figure generation from saved summaries and the single-command pipeline.
"""

import subprocess
import sys
from pathlib import Path

import pytest

import analysis.plots as plots
from experiments import (
    e1_scarcity, e2_frontier, e3_strategic, e3b_rollout, e3c_rollout_horizon, e4_heterogeneous,
    e5_persistence, e6_scalability, e7_sensitivity, e8_timing,
)
from experiments.common import load_seeds

ROOT = Path(__file__).resolve().parent.parent
SEEDS = load_seeds(n=2)
KW = dict(n=20, T=80, n_bootstrap=100, verbose=False)

EXPECTED_STEMS = (
    ["e1_scarcity", "e2_frontier", "e3b_rollout", "e3c_rollout_horizon", "e4_heterogeneous",
     "e5_persistence", "e6_scalability", "e7_sensitivity", "e8_timing"]
    + [f"e3_all_mechs_{m}" for m in ("M_mean", "M_uni", "M_uni_max", "frac_pos_uni", "PoS", "WR", "J_A")]
)


@pytest.fixture(scope="module")
def smoke_results(tmp_path_factory):
    d = tmp_path_factory.mktemp("results")
    e1_scarcity.run_e1(SEEDS, str(d), **KW)
    e2_frontier.run_e2(SEEDS, str(d), **KW)
    e3_strategic.run_e3(SEEDS, str(d), **KW)
    e3b_rollout.run_e3b(SEEDS, str(d), n=10, T=30, n_bootstrap=100, verbose=False, n_rollouts=20)
    e3c_rollout_horizon.run_e3c(SEEDS, str(d), n=10, T=30, n_bootstrap=100, verbose=False,
                                n_rollouts=15, horizons=[2, 4])
    e4_heterogeneous.run_e4(SEEDS, str(d), **KW)
    e5_persistence.run_e5(SEEDS, str(d), **KW)
    e6_scalability.run_e6(SEEDS, str(d), T=30, n_bootstrap=100, verbose=False, n_values=[10, 20])
    e7_sensitivity.run_e7(SEEDS, str(d), **KW)
    e8_timing.run_e8(SEEDS, str(d), **KW)
    return d


def test_every_figure_is_written_as_pdf_and_png(smoke_results, tmp_path, monkeypatch):
    monkeypatch.setattr(plots, "FIGURES_DIR", plots.FIGURES_DIR)       # restored after the test
    plots.run_all_plots(str(smoke_results), str(tmp_path))
    for stem in EXPECTED_STEMS:
        for ext in ("pdf", "png"):
            f = tmp_path / f"{stem}.{ext}"
            assert f.exists() and f.stat().st_size > 2000, f
    assert not (tmp_path / "e3_GreedyMechanism_M_uni.png").exists()    # no per-mechanism clutter


def test_plot_functions_respect_save_false(smoke_results, tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(plots, "FIGURES_DIR", tmp_path)
    plots.plot_e1(json.loads((smoke_results / "e1" / "summary.json").read_text()), save=False)
    assert not list(tmp_path.iterdir())


def test_e3_plot_skips_unknown_and_empty_metrics(smoke_results, tmp_path, monkeypatch, capsys):
    import json
    monkeypatch.setattr(plots, "FIGURES_DIR", tmp_path)
    rows = json.loads((smoke_results / "e3" / "summary.json").read_text())
    plots.plot_e3(rows, metric="not_a_metric")
    assert "SKIP" in capsys.readouterr().out and not list(tmp_path.iterdir())


def test_missing_results_are_skipped_not_fatal(tmp_path, capsys):
    plots.run_all_plots(str(tmp_path / "nothing"), str(tmp_path / "figs"))
    assert capsys.readouterr().out.count("[SKIP]") == 10


def test_plots_cli(smoke_results, tmp_path):
    out = subprocess.run([sys.executable, "analysis/plots.py", "--all", "--results-dir",
                          str(smoke_results), "--figures-dir", str(tmp_path)],
                         cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "e2_frontier.png").exists()


@pytest.mark.slow
def test_run_all_quick_pipeline_end_to_end(tmp_path):
    res, fig = tmp_path / "results", tmp_path / "figures"
    out = subprocess.run([sys.executable, "run_all.py", "--quick", "--results-dir", str(res),
                          "--figures-dir", str(fig)],
                         cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert "All 10 E0" in out.stdout or "all 10 E0" in out.stdout
    for exp in ("e1", "e2", "e3", "e3b", "e3c", "e4", "e5", "e6", "e7", "e8"):
        assert (res / exp / "summary.json").exists(), exp
        assert (res / exp / "config.json").exists() and (res / exp / "raw").is_dir()
    for stem in EXPECTED_STEMS:
        assert (fig / f"{stem}.png").exists(), stem


# ── report generator & seeds ──────────────────────────────────────────────────

def test_report_renders_every_section(smoke_results):
    from analysis.report import build_report
    text = build_report(str(smoke_results))
    for heading in ("## E1", "## E2", "## E3 ", "## E3b", "## E3c", "## E4", "## E5", "## E6",
                    "## E7", "## E8"):
        assert heading in text, heading
    assert "no summary found" not in text and "★" in text and "n/a" in text   # rho=0 coalition gain is n/a


def test_report_handles_missing_results(tmp_path):
    from analysis.report import build_report
    assert build_report(str(tmp_path)).count("no summary found") == 10


def test_report_cli_writes_file(smoke_results, tmp_path):
    out = tmp_path / "R.md"
    res = subprocess.run([sys.executable, "analysis/report.py", "--results-dir", str(smoke_results),
                          "--out", str(out)], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    assert out.read_text(encoding="utf-8").startswith("# Experiment results")


def test_master_seeds_file_is_locked_and_reproducible():
    import json
    import numpy as np
    data = json.loads((ROOT / "seeds" / "master_seeds.json").read_text())
    assert data["n_seeds"] == len(data["seeds"]) == 30
    regenerated = np.random.default_rng(0xFA1_5EED).integers(0, 2**31, size=30).tolist()
    assert data["seeds"] == regenerated                          # generate_seeds.py reproduces the lock
