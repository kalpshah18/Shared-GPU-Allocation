"""
report/make_figures.py
======================
Draw the figures of the project report from the saved results
(results/*/summary.json).  The project's own figures in figures/ are not reused:
these are sized for the single-column template, avoid label collisions, and use
the same colour and marker for each mechanism throughout.

Usage (from the repository root):  python report/make_figures.py
Output: report/media/*.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "media"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.5, "figure.dpi": 150, "pdf.fonttype": 42,
})

STYLE = {  # colour-blind-safe (Okabe-Ito) colours plus a marker per mechanism
    "RandomMechanism": ("M1 Random", "#999999", "o"),
    "RoundRobinMechanism": ("M2 Round-Robin", "#E69F00", "s"),
    "GreedyMechanism": ("M3 Greedy", "#009E73", "^"),
    "ScoreMechanism": ("M4 Score", "#D55E00", "D"),
    "VickreyMechanism": ("M5 Vickrey", "#0072B2", "v"),
}


def load(exp: str) -> list[dict]:
    with open(ROOT / "results" / exp / "summary.json") as fh:
        return json.load(fh)


def err(rows, key):
    m = np.array([r[key]["mean"] for r in rows])
    lo = np.array([r[key]["ci_lower"] for r in rows])
    hi = np.array([r[key]["ci_upper"] for r in rows])
    return m, [np.maximum(0, m - lo), np.maximum(0, hi - m)]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_e1():
    d = load("e1")
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5), constrained_layout=True)
    for ax, (key, yl) in zip(axes, [("WR", "Welfare ratio"), ("SR_delta", "Starvation rate"),
                                    ("Q_max", "Maximum wait $Q_{\\max}$")]):
        for m, (lab, col, mk) in STYLE.items():
            rows = sorted([r for r in d if r["mechanism"] == m], key=lambda r: r["kn_ratio"])
            xs = [r["kn_ratio"] for r in rows]
            mean, e = err(rows, key)
            ax.errorbar(xs, mean, yerr=e, color=col, marker=mk, ms=3.5, lw=1.1, capsize=1.5,
                        label=lab)
        ax.set_xlabel("$k/n$")
        ax.set_ylabel(yl)
        ax.set_xticks([0.1, 0.2, 0.4, 0.6, 0.8])
    axes[2].set_yscale("log")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=5, frameon=False)
    save(fig, "fig_e1")


def fig_lambda():
    d7, d2 = load("e7"), load("e2")
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4), constrained_layout=True)
    ax = axes[0]
    cols = {1.25: "#0072B2", 1.5: "#E69F00", 2.0: "#D55E00"}
    mk = {1.25: "o", 1.5: "s", 2.0: "^"}
    lams = [0.5, 1.0, 2.0, 5.0]
    for c in (1.25, 1.5, 2.0):
        rows = [next(r for r in d7 if r["label"] == f"M4 lambda={l:g}" and r["c"] == c) for l in lams]
        mean, e = err(rows, "M_uni")
        ax.errorbar(range(len(lams)), mean, yerr=e, color=cols[c], marker=mk[c], ms=3.5, lw=1.1,
                    capsize=1.5, label=f"$c={c:g}$")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(lams)))
    ax.set_xticklabels([f"{l:g}" for l in lams])
    ax.set_xlabel("History penalty $\\lambda$")
    ax.set_ylabel("Unilateral gain $M_{\\mathrm{uni}}$")
    ax.set_title("(a) Does inflating pay under M4?")
    ax.legend(frameon=False)
    ax = axes[1]
    ls = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    rows = [next(r for r in d2 if r["mechanism"] == "ScoreMechanism" and r["lambda_"] == l) for l in ls]
    x = range(len(ls))
    for key, lab, col, mkr in (("WR", "WR", "#009E73", "o"), ("J_A", "$J_A$", "#0072B2", "s")):
        mean, e = err(rows, key)
        ax.errorbar(x, mean, yerr=e, color=col, marker=mkr, ms=3.5, lw=1.1, capsize=1.5, label=lab)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{l:g}" for l in ls])
    ax.set_xlabel("History penalty $\\lambda$")
    ax.set_ylabel("WR and $J_A$ ($\\rho=0.25$, $c=2$)")
    ax.set_title("(b) Welfare and fairness under manipulation")
    ax2 = ax.twinx()
    mean, e = err(rows, "SR_delta")
    ax2.errorbar(x, mean, yerr=e, color="#D55E00", marker="D", ms=3.5, lw=1.1, capsize=1.5,
                 ls="--", label="$\\mathrm{SR}_\\Delta$ (right)")
    ax2.set_ylabel("Starvation rate")
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", frameon=False)
    save(fig, "fig_lambda")


def fig_rollout():
    d = load("e3c")
    fig, ax = plt.subplots(figsize=(3.4, 2.4), constrained_layout=True)
    Hs = [5, 10, 20, 40, 80]
    for lab, col, mk, name in (("M4 lambda=1", "#D55E00", "D", "$\\lambda=1$"),
                               ("M4 lambda=2", "#0072B2", "o", "$\\lambda=2$")):
        rows = sorted([r for r in d if r["label"] == lab], key=lambda r: r["H"])
        mean, e = err(rows, "M_rollout")
        ax.errorbar(range(len(Hs)), mean, yerr=e, color=col, marker=mk, ms=3.5, lw=1.1, capsize=1.5,
                    label=f"rollout, {name}")
        ref = rows[0]["M_cap1.25"]["mean"]
        ax.axhline(ref, color=col, ls=":", lw=0.9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(Hs)))
    ax.set_xticklabels([str(h) for h in Hs])
    ax.set_xlabel("Planning horizon $H$")
    ax.set_ylabel("Gain of the attacker")
    ax.text(0.02, 0.97, "dotted: capped, $c=1.25$", transform=ax.transAxes, va="top", fontsize=6.5)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig_rollout")


def fig_e6():
    d = load("e6")
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5), constrained_layout=True)
    for m, (lab, col, mk) in STYLE.items():
        rows = sorted([r for r in d if r["mechanism"] == m], key=lambda r: r["n"])
        ns = [r["n"] for r in rows]
        axes[0].plot(ns, [r["time_per_round_us"]["mean"] for r in rows], color=col, marker=mk,
                     ms=3.5, lw=1.1, label=lab)
        axes[1].plot(ns, [r["peak_memory_kib"]["mean"] / 1024 for r in rows], color=col, marker=mk,
                     ms=3.5, lw=1.1, label=lab)
    for ax, yl in zip(axes, ["Time per round ($\\mu$s)", "Peak traced memory (MiB)"]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Users $n$ ($k=0.2n$)")
        ax.set_ylabel(yl)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=5, frameon=False)
    save(fig, "fig_e6")


def fig_e9():
    d = load("e9")
    fig, ax = plt.subplots(figsize=(4.6, 2.9), constrained_layout=True)
    base = {r["label"]: r for r in d if r["condition"] == "base"}
    pts = [("M1 Random", "#999999", "o", (-44, -3), "M1 Random"),
           ("M2 Round-Robin", "#E69F00", "s", (5, 4), "M2 Round-Robin"),
           ("M3 Greedy", "#009E73", "^", (-62, 6), "M3 Greedy = M5 Vickrey"),
           ("M4 Score λ=1", "#D55E00", "D", (-58, -12), "M4 Score ($\\lambda=1$)"),
           ("M5 Vickrey", "#0072B2", "v", None, None)]
    for lab, col, mk, off, name in pts:
        r = base[lab]
        ax.scatter(r["Q_max"]["mean"], r["WR"]["mean"], color=col, marker=mk, s=22, zorder=3)
        if lab.startswith("M4"):
            ax.annotate(name, (r["Q_max"]["mean"], r["WR"]["mean"]), xytext=(24, 0.885),
                        textcoords="data", fontsize=6.5, ha="center",
                        arrowprops=dict(arrowstyle="-", lw=0.6, color="#666666"))
        elif off is not None:
            ax.annotate(name, (r["Q_max"]["mean"], r["WR"]["mean"]), xytext=off,
                        textcoords="offset points", fontsize=6.5)
    for lab, col, mk in (("M6 Karma-Cap", "#CC79A7", "o"), ("M7 Rank-Cap", "#56B4E9", "s")):
        rows = sorted([r for r in d if r["condition"] == "wait_frontier" and r["label"] == lab],
                      key=lambda r: r["wait_cap"])
        ax.plot([r["Q_max"]["mean"] for r in rows], [r["WR"]["mean"] for r in rows], color=col,
                marker=mk, ms=3.5, lw=1.1, label=lab)
        for r in rows:
            dy = -9 if lab.startswith("M6") else 4
            ax.annotate(f"{r['wait_cap']}", (r["Q_max"]["mean"], r["WR"]["mean"]), xytext=(2, dy),
                        textcoords="offset points", fontsize=6, color=col)
    ax.set_xscale("log")
    ax.set_xlabel("Observed maximum wait $Q_{\\max}$ (log scale)")
    ax.set_ylabel("Welfare ratio (truthful)")
    ax.set_xticks([4, 6, 10, 20, 40])
    ax.set_xticklabels(["4", "6", "10", "20", "40"])
    ax.minorticks_off()
    ax.text(0.02, 0.97, "numbers on lines: wait cap $W$", transform=ax.transAxes, fontsize=6.5, va="top")
    ax.legend(frameon=False, loc="center right")
    save(fig, "fig_e9")


def fig_e14():
    d = load("e14")
    fig, ax = plt.subplots(figsize=(4.6, 2.5), constrained_layout=True)
    conds = ["base", "mixed", "persistent"]
    rules = [("Score-Cap", "#D55E00"), ("Karma-Cap", "#CC79A7"), ("Rank-Cap", "#56B4E9")]
    w = 0.26
    for j, (rule, col) in enumerate(rules):
        for i, c in enumerate(conds):
            r = next(x for x in d if x["label"] == rule and x["condition"] == c and x["wait_cap"] == 6)
            x0 = i + (j - 1) * w
            u, a = r["unavoidable_loss"]["mean"], r["algorithm_loss"]["mean"]
            ax.bar(x0, u, w * 0.92, color="#444444", label="service-constraint cost" if (i, j) == (0, 0) else None)
            ax.bar(x0, a, w * 0.92, bottom=u, color=col, label=rule if i == 0 else None)
    ax.set_xticks(range(3))
    ax.set_xticklabels(conds)
    ax.set_ylabel("Welfare loss versus unconstrained optimum")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig_e14")


if __name__ == "__main__":
    fig_e1()
    fig_lambda()
    fig_rollout()
    fig_e6()
    fig_e9()
    fig_e14()
