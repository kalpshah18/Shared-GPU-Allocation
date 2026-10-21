"""
analysis/bootstrap.py
=====================
Paired 95% bootstrap confidence intervals — Owner: Kalp Shah

From proposal §4.3: "Reported results will include seed means and paired
95% bootstrap confidence intervals based on 10,000 resamples."

All functions operate on arrays of per-seed scalar metric values.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 95.0,
    seed: int = 42,
) -> dict:
    """
    Compute the mean and bootstrap confidence interval for an array of
    per-seed scalar metric values.

    Parameters
    ----------
    values     : 1-D array of per-seed metric values (length = n_seeds)
    n_resamples: number of bootstrap resamples (default 10,000)
    ci         : confidence level as a percentage (default 95)
    seed       : RNG seed for reproducibility of the bootstrap itself

    Returns
    -------
    dict with keys: 'mean', 'ci_lower', 'ci_upper', 'std'
    """
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = values[idx].mean(axis=1)

    alpha = (100.0 - ci) / 2.0
    return {
        "mean"    : float(np.mean(values)),
        "ci_lower": float(np.percentile(boot_means, alpha)),
        "ci_upper": float(np.percentile(boot_means, 100.0 - alpha)),
        "std"     : float(np.std(values, ddof=1)) if n > 1 else 0.0,
    }


def paired_bootstrap_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 95.0,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap CI for the difference (A − B).

    Because each seed generates paired observations (same ω, different
    mechanism or policy), we resample pairs to preserve the pairing.

    Returns
    -------
    dict: 'mean_diff', 'ci_lower', 'ci_upper', 'std_diff'
    """
    rng = np.random.default_rng(seed)
    diffs = np.asarray(values_a, dtype=np.float64) - np.asarray(values_b, dtype=np.float64)
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot = diffs[idx].mean(axis=1)

    alpha = (100.0 - ci) / 2.0
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_lower" : float(np.percentile(boot, alpha)),
        "ci_upper" : float(np.percentile(boot, 100.0 - alpha)),
        "std_diff" : float(np.std(diffs, ddof=1)) if n > 1 else 0.0,
    }


def summarise_seeds(
    per_seed_results: list[dict],
    metric_keys: list[str],
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Given a list of per-seed result dicts, compute bootstrap CIs for each
    metric key and return a summary dict.

    Parameters
    ----------
    per_seed_results : list of dicts, one per seed
    metric_keys      : list of keys to summarise
    n_resamples      : bootstrap resamples
    seed             : bootstrap RNG seed

    Returns
    -------
    dict mapping metric_key -> bootstrap_ci output dict.  Undefined (NaN)
    per-seed values are dropped; if none remain, every statistic is None so
    the summary serialises as valid JSON (null rather than NaN).
    """
    summary = {}
    for key in metric_keys:
        vals = np.array([r[key] for r in per_seed_results], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            summary[key] = {"mean": None, "ci_lower": None, "ci_upper": None, "std": None}
            continue
        summary[key] = bootstrap_ci(vals, n_resamples=n_resamples, seed=seed)
    return summary


def paired_differences(
    per_group: dict,
    reference: str,
    metric_keys: list,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap CIs of (group - reference) for every group and metric.

    `per_group` maps a group name to its list of per-seed result dicts.  The
    lists must be aligned by seed (same length, same seed order); resampling
    keeps each seed's pair together.  Seeds where either value is NaN are
    dropped for that metric.  A metric with no usable pairs maps to None
    statistics.

    Returns {group: {metric: {'mean_diff', 'ci_lower', 'ci_upper', 'std_diff'}}}.
    """
    ref_rows = per_group[reference]
    out = {}
    for name, rows in per_group.items():
        if name == reference:
            continue
        if len(rows) != len(ref_rows):
            raise ValueError(f"group '{name}' has {len(rows)} seeds, reference has {len(ref_rows)}.")
        out[name] = {}
        for key in metric_keys:
            a = np.array([r[key] for r in rows], dtype=np.float64)
            b = np.array([r[key] for r in ref_rows], dtype=np.float64)
            ok = ~(np.isnan(a) | np.isnan(b))
            if not ok.any():
                out[name][key] = {"mean_diff": None, "ci_lower": None,
                                  "ci_upper": None, "std_diff": None}
            else:
                out[name][key] = paired_bootstrap_ci(a[ok], b[ok], n_resamples=n_resamples, seed=seed)
    return out
