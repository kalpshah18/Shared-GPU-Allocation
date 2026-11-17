"""Audit saved E19/E20 seeds, configuration, raw data and paired inference."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.bootstrap import paired_differences, summarise_seeds
from experiments.common import STRATEGIC_KEYS
from experiments.e19_value_threshold import CONTROLLED_KEYS, PAIRED_KEYS
from sim.environment import save_json


def _load(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON in {path}: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def _equal(actual, expected):
    if isinstance(expected, dict):
        for key, value in expected.items():
            _equal(actual[key], value)
    elif expected is None:
        assert actual is None
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def verify_feedback(results_dir="results", experiments=("e19", "e20")):
    rd = Path(results_dir)
    audit = {"passed": True, "experiments": list(experiments), "profile_cells": 0,
             "controlled_cells": 0, "paired_comparisons": 0}
    for exp in experiments:
        directory = rd / exp
        cfg = _load(directory / "config.json")
        assert cfg["n_seeds"] == len(cfg["seeds"]) == len(set(cfg["seeds"]))
        profile_rows = {}
        for summary in _load(directory / "summary.json"):
            policy = summary["variant" if exp == "e19" else "policy"]
            path = directory / "raw" / f"{summary['label'].replace(' ', '_')}_{policy}.json"
            raw = _load(path)
            rows, cell_cfg = raw["rows"], raw["config"]
            assert [r["seed"] for r in rows] == cfg["seeds"]
            assert summary["n_seeds"] == cfg["n_seeds"] == cell_cfg["n_seeds"]
            assert cell_cfg["n_bootstrap"] == cfg["n_bootstrap"]
            for key in ("n", "k", "T", "rho"):
                assert cell_cfg[key] == cfg[key]
            _equal(summary, summarise_seeds(rows, STRATEGIC_KEYS, cfg["n_bootstrap"]))
            profile_rows[(summary["label"], policy)] = rows
            audit["profile_cells"] += 1
        if exp != "e19":
            continue
        controls = {}
        for summary in _load(directory / "controlled.json"):
            label, tau = summary["label"], summary["tau"]
            raw = _load(directory / "raw" / f"controlled_{label.replace(' ', '_')}_tau{tau:g}.json")
            rows = raw["rows"]
            assert [r["seed"] for r in rows] == cfg["seeds"]
            for row in rows:
                records = row["focal_results"]
                assert records and len({r["focal"] for r in records}) == len(records)
                for focal in records:
                    _equal(focal["sandbag_gain"], focal["sandbag_utility"] - focal["truth_utility"])
                    _equal(focal["boost_gain"], focal["boost_utility"] - focal["truth_utility"])
                    _equal(focal["timing_premium"], focal["sandbag_utility"] - focal["boost_utility"])
                for key in CONTROLLED_KEYS:
                    _equal(row[key], np.mean([r[key] for r in records]))
            _equal(summary, summarise_seeds(rows, CONTROLLED_KEYS, cfg["n_bootstrap"]))
            # The controlled boost arm reproduces the population boost cell:
            # same strategic set, same truthful counterfactual for each focal.
            for controlled_row, profile_row in zip(rows, profile_rows[(label, f"boost_{tau:g}")]):
                _equal(controlled_row["boost_gain"], profile_row["M_uni"])
            controls[(summary["lambda_"], tau)] = rows
            audit["controlled_cells"] += 1
        for comparison in _load(directory / "paired.json"):
            tau = comparison["tau"]
            if comparison["comparison"] == "lambda_effect":
                groups = {"reference": controls[(comparison["reference_lambda"], tau)],
                          "comparison": controls[(comparison["lambda_"], tau)]}
                keys = ["timing_premium"]
            else:
                label = comparison["label"]
                variant = "sandbag" if comparison["comparison"] == "sandbag - boost" else "max_claim"
                name = f"sandbag_{tau:g}" if variant == "sandbag" else variant
                groups = {"reference": profile_rows[(label, f"boost_{tau:g}")],
                          "comparison": profile_rows[(label, name)]}
                keys = PAIRED_KEYS
            expected = paired_differences(groups, "reference", keys, cfg["n_bootstrap"])["comparison"]
            _equal(comparison["metrics"], expected)
            audit["paired_comparisons"] += 1
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    audit = verify_feedback(args.results_dir)
    save_json(audit, Path(args.results_dir) / "FEEDBACK_AUDIT.json")
    print(json.dumps(audit))
