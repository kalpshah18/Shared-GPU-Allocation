"""Generate final-report figures, tables and an audit from saved seed-level data.

Run from the repo root: python analysis/build_final_report.py
This does not rerun or overwrite experiments. Compile the LaTeX separately.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from analysis.bootstrap import paired_bootstrap_ci, summarise_seeds
from analysis.verify_feedback import verify_feedback

OUT = ROOT / "Project" / "final_assets"
OUT.mkdir(parents=True, exist_ok=True)
CELLS = {}
SOURCES = {}
MACROS = {}


def read(path):
    path = ROOT / path
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8"))


def same(actual, expected):
    if isinstance(expected, dict):
        for key, value in expected.items():
            same(actual[key], value)
    elif expected is None:
        assert actual is None
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def cell(exp, filename, **selector):
    """Verify all summary statistics against raw seed rows before using a cell."""
    key = f"{exp}/{filename}"
    if key in CELLS:
        return CELLS[key]["summary"]
    rows = read(f"results/{exp}/summary.json")
    found = [r for r in rows if all(r.get(k) == v for k, v in selector.items())]
    assert len(found) == 1, (exp, selector, len(found))
    summary = found[0]
    raw_path = f"results/{exp}/raw/{filename}.json"
    raw = read(raw_path)
    seed_rows = raw["rows"]
    assert len(seed_rows) == summary["n_seeds"] == 30
    ids = [r["seed"] for r in seed_rows]
    assert len(set(ids)) == 30
    seed_file = "seeds/extension_seeds.json" if exp in ("e12", "e14", "e15") else "seeds/master_seeds.json"
    assert ids == read(seed_file)["seeds"]
    keys = [k for k, value in summary.items() if isinstance(value, dict)]
    same(summary, summarise_seeds(seed_rows, keys, n_resamples=10_000))
    CELLS[key] = {"summary": summary, "raw_source": raw_path,
                  "seed_file": seed_file, "verified_statistics": keys,
                  "config": raw["config"]}
    return summary


def control(label, tau):
    key = f"e19/controlled_{label.replace(' ', '_')}_tau{tau:g}"
    if key in CELLS:
        return CELLS[key]["summary"]
    rows = read("results/e19/controlled.json")
    summary = next(r for r in rows if r["label"] == label and r["tau"] == tau)
    raw_path = f"results/{key}.json".replace("e19/controlled_", "e19/raw/controlled_")
    raw = read(raw_path)
    assert [r["seed"] for r in raw["rows"]] == read("seeds/master_seeds.json")["seeds"]
    same(summary, summarise_seeds(raw["rows"], ["sandbag_gain", "boost_gain", "timing_premium"], 10_000))
    CELLS[key] = {"summary": summary, "raw_source": raw_path,
                  "seed_file": "seeds/master_seeds.json", "config": raw["config"],
                  "verified_statistics": ["sandbag_gain", "boost_gain", "timing_premium"]}
    return summary


def number(x, digits=3, signed=False):
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def value(row, key, digits=3, ci=False, signed=False, multiplier=1):
    x = row[key]
    mean = number(x["mean"] * multiplier, digits, signed)
    if ci:
        mean += rf"\;[{number(x['ci_lower'] * multiplier, digits)},\,{number(x['ci_upper'] * multiplier, digits)}]"
    return rf"\({mean}\)"


def macro(name, row, key, digits=3, ci=False, signed=False, multiplier=1):
    MACROS[name] = value(row, key, digits, ci, signed, multiplier)


def table(name, spec, header, rows):
    text = (rf"\begin{{tabular}}{{{spec}}}" + "\n\\toprule\n" + header + r" \\" +
            "\n\\midrule\n" + "\n".join(" & ".join(row) + r" \\" for row in rows) +
            "\n\\bottomrule\n\\end{tabular}\n")
    (OUT / f"{name}.tex").write_text(text, encoding="utf-8")


plt.rcParams.update({"font.family": "serif", "font.size": 8.5,
                     "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "legend.fontsize": 7.5, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#666666"]


def errorbar(ax, xs, rows, key, label, color, marker="o", **kwargs):
    y = np.array([r[key]["mean"] for r in rows])
    low = np.array([r[key]["ci_lower"] for r in rows])
    high = np.array([r[key]["ci_upper"] for r in rows])
    ax.errorbar(xs, y, yerr=[np.maximum(0, y-low), np.maximum(0, high-y)],
                label=label, color=color, marker=marker, ms=3.8, capsize=2,
                lw=1.2, **kwargs)


def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build():
    # The feedback audit includes all cells, focal utility identities and paired comparisons.
    feedback = verify_feedback(ROOT / "results")
    mechs = [("RandomMechanism", "M1 Random"), ("RoundRobinMechanism", "M2 Round-robin"),
             ("GreedyMechanism", "M3 Greedy"), ("ScoreMechanism", r"M4 Score ($\lambda=1$)"),
             ("VickreyMechanism", "M5 Vickrey")]
    truth = []
    for mechanism, label in mechs:
        base = cell("e1", f"{mechanism}_kn_0.2", mechanism=mechanism, kn_ratio=.2)
        mixed = cell("e4", f"{mechanism}_mixed", mechanism=mechanism, dist="mixed")
        truth.append([label, value(base, "WR"), value(base, "J_A"),
                      value(base, "SR_delta", 2, multiplier=100), value(base, "Q_max", 1),
                      value(mixed, "WR"), value(mixed, "J_B")])
        if mechanism == "ScoreMechanism":
            macro("TruthScoreWR", base, "WR")
            macro("TruthScoreSR", base, "SR_delta", 2, multiplier=100)
            macro("MixedScoreWR", mixed, "WR")
            macro("MixedScoreJB", mixed, "J_B")
        if mechanism == "GreedyMechanism":
            macro("TruthGreedySR", base, "SR_delta", 2, multiplier=100)
            macro("MixedGreedyJB", mixed, "J_B")
            macro("TruthGreedyQ", base, "Q_max", 1)
    table("truth", "lrrrrrr", r"Rule & WR & $J_A$ & SR (\%) & $Q_{\max}$ & Mixed WR & Mixed $J_B$", truth)

    fig, ax = plt.subplots(figsize=(6.5, 2.25), layout="constrained")
    for c, color, marker in zip([1.25, 1.5, 2.], COLORS, ["o", "s", "^"]):
        rows = [cell("e7", f"M4_lambda={lam:g}_c_{c:g}", mechanism="ScoreMechanism",
                     lambda_=lam, c=c) for lam in [.5, 1., 2., 5.]]
        errorbar(ax, [.5, 1, 2, 5], rows, "M_uni", f"c = {c:g}", color, marker)
    ax.axhline(0, color="black", lw=.7)
    ax.set(xlabel=r"History penalty $\lambda$", ylabel="Unilateral gain (1,000 rounds)",
           xscale="log", xticks=[.5, 1, 2, 5], xticklabels=["0.5", "1", "2", "5"])
    ax.grid(alpha=.25)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(ncol=3, loc="upper right")
    savefig(fig, "m4_inflation")
    for name, lam, c in [("MildFive", 5., 1.25), ("DoubleTwo", 2., 2.),
                          ("MediumTwo", 2., 1.5), ("MediumFive", 5., 1.5)]:
        row = cell("e7", f"M4_lambda={lam:g}_c_{c:g}", mechanism="ScoreMechanism", lambda_=lam, c=c)
        macro(name, row, "M_uni", 2, ci=True, signed=True)

    # Controlled timing and original population cells are distinct estimands.
    timing = []
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5), layout="constrained")
    for tau, color, marker in zip([.5, .75, .9], COLORS, ["o", "s", "^"]):
        rows = [control(f"M4 lambda={lam:g}", tau) for lam in [.5, 1, 2, 5]]
        ax = axes[1] if tau == .9 else axes[0]
        errorbar(ax, [.5, 1, 2, 5], rows, "timing_premium", rf"$\tau={tau:g}$", color, marker)
    for ax in axes:
        ax.set(xscale="log", xticks=[.5, 1, 2, 5], xticklabels=["0.5", "1", "2", "5"],
               xlabel=r"History penalty $\lambda$")
        ax.axhline(0, color="black", lw=.7)
        ax.grid(alpha=.25)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend(loc="best")
    axes[0].set_ylabel("Controlled timing premium")
    axes[0].set_title("Small effects: low / middle thresholds")
    axes[1].set_title("High threshold: separate y scale")
    savefig(fig, "m4_timing")
    for lam in [.5, 1, 2, 5]:
        controlled = control(f"M4 lambda={lam:g}", .75)
        profile = cell("e19", f"M4_lambda={lam:g}_sandbag_0.75", label=f"M4 lambda={lam:g}", variant="sandbag_0.75")
        timing.append([f"{lam:g}", value(profile, "M_uni", 2, ci=True, signed=True),
                       value(controlled, "sandbag_gain", 2, ci=True, signed=True),
                       value(controlled, "timing_premium", 3, ci=True, signed=True)])
    table("timing", "rrrr", r"$\lambda$ & Profile gain & Controlled sandbag gain & Timing premium", timing)
    macro("TimingLow", control("M4 lambda=0.5", .75), "timing_premium", 3, ci=True, signed=True)
    macro("TimingHigh", control("M4 lambda=5", .75), "timing_premium", 3, ci=True, signed=True)
    comparisons = read("results/e19/paired.json")
    trend = next(r for r in comparisons if r["comparison"] == "lambda_effect" and r["tau"] == .75 and r["lambda_"] == 5)
    low = read("results/e19/raw/controlled_M4_lambda=0.5_tau0.75.json")["rows"]
    high = read("results/e19/raw/controlled_M4_lambda=5_tau0.75.json")["rows"]
    expected = paired_bootstrap_ci([r["timing_premium"] for r in high], [r["timing_premium"] for r in low])
    same(trend["metrics"]["timing_premium"], expected)
    effect = {"effect": {**expected, "mean": expected["mean_diff"]}}
    macro("TimingTrend", effect, "effect", 3, ci=True, signed=True)
    macro("CreditTiming", control("M5c rate=0.5", .75), "timing_premium", 2, ci=True, signed=True)

    rates = [.25, .5, 1., 2.]
    credit_table, credit_truth, credit_policies = [], [], {}
    for rate in rates:
        label = f"M5c rate={rate:g}"
        tr = cell("e20", f"{label.replace(' ', '_')}_truthful", label=label, policy="truthful")
        sb = cell("e20", f"{label.replace(' ', '_')}_sandbag_0.75", label=label, policy="sandbag_0.75")
        cap = cell("e20", f"{label.replace(' ', '_')}_cap_1.25", label=label, policy="cap_1.25")
        credit_truth.append(tr)
        credit_table.append([f"{rate:g}", value(tr, "WR"), value(tr, "J_A", 4), value(tr, "Q_max", 1),
                             value(cap, "M_uni", 2, ci=True, signed=True), value(sb, "M_uni", 2, ci=True, signed=True)])
        for policy in ["cap_1.25", "cap_2", "max_claim", "sandbag_0.75"]:
            credit_policies.setdefault(policy, []).append(cell("e20", f"{label.replace(' ', '_')}_{policy}", label=label, policy=policy))
    table("credits", "rrrrrr", r"Rate & Truth WR & Truth $J_A$ & Truth $Q_{\max}$ & Gain: $c=1.25$ & Gain: sandbag$_{.75}$", credit_table)
    macro("CreditTightWR", credit_truth[0], "WR")
    macro("CreditTightQ", credit_truth[0], "Q_max", 1)
    macro("CreditHalfWR", credit_truth[1], "WR")
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5), layout="constrained")
    for key, label, color in [("WR", "WR", COLORS[0]), ("J_A", r"$J_A$", COLORS[2])]:
        errorbar(axes[0], rates, credit_truth, key, label, color)
    for i, (policy, rows) in enumerate(credit_policies.items()):
        errorbar(axes[1], rates, rows, "M_uni", policy.replace("_", " "), COLORS[i], ["o", "s", "^", "D"][i])
    axes[0].set(ylabel=r"Truthful WR / $J_A$", title="Credit budget trade-off")
    axes[1].set(ylabel="Unilateral gain (1,000 rounds)", title="Policy-specific opponent populations")
    axes[1].axhline(0, color="black", lw=.7)
    for ax in axes:
        ax.set(xscale="log", xticks=rates, xticklabels=["0.25", "0.5", "1", "2"], xlabel="Credit income rate")
        ax.grid(alpha=.25)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend(loc="best", fontsize=7)
    savefig(fig, "credit_budget")

    # Matched-cap evaluation: all outcomes below use the fresh extension seeds.
    extension_table = []
    for family, label in [("Score", "Score-Cap"), ("Karma", "Karma-Cap"), ("Rank", "Rank-Cap")]:
        truthful = {}
        for condition in ["base", "mixed", "persistent"]:
            truthful[condition] = cell("e12", f"{condition}_truthful_rho0_{family}_W10",
                condition=condition, family=family, wait_cap=10, policy="truthful", rho=0.)
        attack = cell("e12", f"base_cap_1.25_rho0.25_{family}_W10", condition="base", family=family,
                      wait_cap=10, policy="cap_1.25", rho=.25)
        extension_table.append([label] + [value(truthful[c], "WR") for c in ["base", "mixed", "persistent"]]
                               + [value(attack, "M_uni", 2, ci=True, signed=True)])
    table("extensions", "lrrrr", r"Rule ($W=10$) & Base WR & Mixed WR & Persistent WR & Mild-inflation gain", extension_table)
    fig, ax = plt.subplots(figsize=(6.5, 1.95), layout="constrained")
    planner_labels = ["Score", "Score lambda=2", "Score-Cap", "Karma-Cap", "Rank-Cap"]
    for i, label in enumerate(planner_labels):
        mode = "updating" if label == "Rank-Cap" else "exact"
        row = cell("e15", f"{label}_{mode}_H40_R80", label=label, mode=mode, H=40, n_rollouts=80)
        errorbar(ax, [i], [row], "M_rollout", None, COLORS[i], "o")
        if label == "Score lambda=2":
            macro("ScorePlanner", row, "M_rollout", 2, ci=True, signed=True)
        if label == "Rank-Cap":
            macro("RankPlanner", row, "M_rollout", 2, ci=True, signed=True)
    ax.set(xticks=range(5), xticklabels=[r"Score $\lambda=1$", r"Score $\lambda=2$", "Score-Cap", "Karma-Cap", "Rank-Cap"],
           ylabel="Focal gain (120 rounds)")
    ax.axhline(0, color="black", lw=.7)
    ax.grid(axis="y", alpha=.25)
    savefig(fig, "planning_attack")
    oracle = []
    for label in ["Score-Cap", "Karma-Cap", "Rank-Cap"]:
        row = cell("e14", f"mixed_{label}_W6", label=label, condition="mixed", wait_cap=6)
        oracle.append([label, value(row, "constrained_efficiency", 3, ci=True),
                       value(row, "unavoidable_loss"), value(row, "algorithm_loss")])
    table("oracle", "lrrr", r"Rule & Constrained efficiency & Service loss & Remaining loss", oracle)
    # Count unique solves, not one duplicate per online mechanism.
    solves = {}
    for path in sorted((ROOT / "results/e14/raw").glob("*.json")):
        raw = read(path.relative_to(ROOT))
        condition, _, W = path.stem.partition("_")
        for row in raw["rows"]:
            key = (condition, raw["config"]["wait_cap"], row["seed"])
            assert row["oracle_optimal"] == 1
            solves[key] = True
    assert len(solves) == 270
    MACROS["CertifiedSolves"] = str(len(solves))

    (OUT / "numbers.tex").write_text("% Generated from audited raw data; do not edit.\n" +
        "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())) + "\n", encoding="utf-8")
    commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--",
                                      "results", "sim", "experiments", "seeds", "config"],
                                     cwd=ROOT, text=True).strip()
    for relative in ("Project/FAI_Project_Proposal.tex", "Project/FAI_Project_Final_Report.tex",
                     "Project/PRIMEarxiv.sty", "Project/final_references.bib",
                     "analysis/build_final_report.py", "analysis/compile_final_report.py",
                     "analysis/verify_final_replay.py"):
        path = ROOT / relative
        SOURCES[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    evidence = {"passed": True, "data_commit": commit, "bootstrap_resamples": 10_000,
                "feedback_audit": feedback, "verified_report_cells": len(CELLS),
                "certified_unique_oracle_solves": len(solves), "cells": CELLS,
                "source_sha256": SOURCES, "generated_macros": MACROS,
                "audit_render_runtime": {"python": platform.python_version(), "numpy": np.__version__,
                                         "matplotlib": matplotlib.__version__},
                "interpretation": "Pointwise exploratory intervals; master and fresh extension seeds remain distinct."}
    replay_path = OUT / "replay_checks.json"
    if replay_path.exists():
        evidence["representative_replay"] = {
            "file": replay_path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest(),
            **json.loads(replay_path.read_text())}
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Verified {len(CELLS)} report cells and {len(solves)} unique optimal oracle solves; generated assets in {OUT}")


if __name__ == "__main__":
    build()
