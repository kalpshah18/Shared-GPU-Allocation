"""Scientific figures for E12-E15, rendered from saved summary statistics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"Score": "#2171b5", "Karma": "#238b45", "Rank": "#d95f0e"}


def errorbar(ax, x, rows, key, **kwargs):
    y = np.array([r[key]["mean"] for r in rows], dtype=float)
    lo = np.array([r[key]["ci_lower"] for r in rows], dtype=float)
    hi = np.array([r[key]["ci_upper"] for r in rows], dtype=float)
    ax.errorbar(x, y, yerr=[np.maximum(0, y - lo), np.maximum(0, hi - y)], capsize=3, **kwargs)


def e12(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    caps = sorted({r["wait_cap"] for r in rows if r["cap_enabled"]})
    positions = list(range(len(caps) + 1))
    for ax, policy, metric, ylabel in zip(axes, ["truthful", "cap_1.25"], ["WR", "M_uni"],
                                         ["True welfare / unconstrained optimum", "Unilateral gain (episode utility)"]):
        for family, color in COLORS.items():
            cells = [r for r in rows if r["family"] == family and r["condition"] == "base"
                     and r["policy"] == policy and r["rho"] == (0 if policy == "truthful" else 0.25)]
            cells.sort(key=lambda r: caps.index(r["wait_cap"]) if r["cap_enabled"] else len(caps))
            errorbar(ax, positions, cells, metric, marker="o", color=color, label=family)
        ax.set_xticks(positions, [str(w) for w in caps] + ["Off"])
        ax.set_xlabel("Service override threshold W")
        ax.set_ylabel(ylabel)
        ax.set_title("Truthful homogeneous demand" if policy == "truthful" else "Mild exaggeration: c=1.25, rho=0.25")
        if metric == "M_uni":
            ax.axhline(0, color="#666", linewidth=0.8, linestyle="--")
        ax.legend(frameon=False, loc="center right" if metric == "M_uni" else "upper left")
    fig.suptitle("E12: matched cap thresholds isolate the incentive component")
    fig.tight_layout()
    return fig


def e13(rows):
    starts = list(dict.fromkeys(r["initial"] for r in rows))
    fig, axes = plt.subplots(2, len(starts), figsize=(12, 7), squeeze=False)
    for column, initial in enumerate(starts):
        selected = [r for r in rows if r["initial"] == initial]
        labels = [r["label"] for r in selected]
        x = np.arange(len(selected))
        for row, key, ylabel in [(0, "WR", "Welfare ratio"),
                                 (1, "restricted_exploitability", "Maximum expected unilateral gain")]:
            ax = axes[row, column]
            ax.bar(x, [r[key]["mean"] for r in selected], color=plt.cm.tab10(np.arange(len(x))), alpha=0.75)
            errorbar(ax, x, selected, key, fmt="none", ecolor="#333")
            ax.set_xticks(x, labels, rotation=30, ha="right")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Initial population: {initial}")
            if row == 0:
                ax.set_ylim(0, 1.06)
            else:
                ax.set_ylim(bottom=0)
    fig.suptitle("E13: independently trained profiles, heldout welfare and restricted regret")
    fig.tight_layout()
    return fig


def e14(rows):
    conditions = list(dict.fromkeys(r["condition"] for r in rows))
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 4.8), squeeze=False)
    for ax, condition in zip(axes[0], conditions):
        for label, color in zip(["Score-Cap", "Karma-Cap", "Rank-Cap"], COLORS.values()):
            selected = sorted([r for r in rows if r["condition"] == condition and r["label"] == label],
                              key=lambda r: r["certified_bound"])
            errorbar(ax, [r["certified_bound"] for r in selected], selected, "constrained_efficiency",
                     marker="o", color=color, label=label)
        ax.axhline(1, color="#666", linewidth=0.8, linestyle="--")
        ax.set_title(condition)
        ax.set_xlabel("Shared certified maximum wait B")
        ax.set_ylabel("Online welfare / constrained offline optimum")
        ax.legend(frameon=False)
    fig.suptitle("E14: performance after accounting for the unavoidable service cost")
    fig.tight_layout()
    return fig


def e15(rows):
    counts = sorted({r["n_rollouts"] for r in rows})
    fig, axes = plt.subplots(2, len(counts), figsize=(6 * len(counts), 9), squeeze=False)
    labels = list(dict.fromkeys(r["label"] for r in rows))
    for column, count in enumerate(counts):
        for row in range(2):
            ax = axes[row, column]
            for index, label in enumerate(labels):
                if row == 1 and label == "Greedy":
                    continue
                selected = sorted([r for r in rows if r["label"] == label and r["mode"] != "frozen"
                                   and r["n_rollouts"] == count], key=lambda r: r["H"])
                errorbar(ax, [r["H"] for r in selected], selected, "M_rollout", marker="o",
                         color=plt.cm.tab10(index), label=label)
            frozen = sorted([r for r in rows if r["mode"] == "frozen" and r["n_rollouts"] == count], key=lambda r: r["H"])
            if frozen:
                errorbar(ax, [r["H"] for r in frozen], frozen, "M_rollout", marker="x", linestyle="none",
                         color=plt.cm.tab10(labels.index("Rank-Cap")), label="Rank-Cap frozen control")
            ax.axhline(0, color="#666", linewidth=0.8, linestyle="--")
            ax.set_title(f"MC futures: {count}" + (" — all mechanisms" if row == 0 else " — detail excluding Greedy"))
            ax.set_xlabel("Planning horizon H (truncated at episode end)")
            ax.set_ylabel("Focal utility gain vs truthful reporting")
            if row == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("E15: stronger planning and computation budget (95% intervals)")
    fig.tight_layout()
    return fig


PLOTS = [("e12", "e12_ablations", e12), ("e13", "e13_adaptive", e13),
         ("e14", "e14_constrained", e14), ("e15", "e15_attack_budget", e15)]


def run_course_plots(results_dir="results", figures_dir="figures"):
    destination = Path(figures_dir)
    for experiment, stem, plot in PLOTS:
        source = Path(results_dir) / experiment / "summary.json"
        if not source.exists():
            continue
        rows = json.loads(source.read_text())
        if not rows:
            continue
        fig = plot(rows)
        destination.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf"):
            fig.savefig(destination / f"{stem}.{extension}", bbox_inches="tight", dpi=160)
        plt.close(fig)
        print(f"Saved {destination / stem}.{{png,pdf}}")
