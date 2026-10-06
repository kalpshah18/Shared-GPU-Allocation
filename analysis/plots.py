"""
analysis/plots.py
=================
All figure generation — Owner: Kalp Shah

Each `plot_*` function accepts the aggregated results (list of summary dicts)
and saves figures (PDF and PNG) to the `figures/` directory.

Style: clean academic style with error bars (95% bootstrap CI).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import numpy as np

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

MECHANISM_LABELS = {
    "RandomMechanism"    : "M1 Random",
    "RoundRobinMechanism": "M2 Round-Robin",
    "GreedyMechanism"    : "M3 Greedy",
    "ScoreMechanism"     : "M4 Score",
    "VickreyMechanism"   : "M5 Vickrey",
}

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
MARKERS = ["o", "s", "^", "D", "v"]

plt.rcParams.update({
    "font.family"       : "sans-serif",
    "font.size"         : 11,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "figure.dpi"        : 150,
})


# ── E1: Resource Scarcity ─────────────────────────────────────────────────────

def plot_e1(results: list[dict], save: bool = True) -> None:
    """
    Line plots of WR, J_A, SR_Δ, and pct95_wait vs k/n per mechanism.
    """
    metrics = [
        ("WR", "Welfare Ratio (WR)"),
        ("J_A", "Jain Allocation Fairness (J_A)"),
        ("SR_delta", "Starvation Rate (SR_Δ)"),
        ("pct95_wait", "95th-pct Consecutive Wait"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), tight_layout=True)
    axes = axes.flatten()

    mechanisms = [m for m in MECHANISM_LABELS.keys() if any(r["mechanism"] == m for r in results)]
    if not mechanisms:
        mechanisms = sorted({r["mechanism"] for r in results})

    for ax, (metric, ylabel) in zip(axes, metrics):
        for idx, mech in enumerate(mechanisms):
            subset = [r for r in results if r["mechanism"] == mech]
            subset.sort(key=lambda r: r["kn_ratio"])
            xs    = [r["kn_ratio"] for r in subset]
            means = [r[metric]["mean"] for r in subset]
            lows  = [r[metric]["ci_lower"] for r in subset]
            highs = [r[metric]["ci_upper"] for r in subset]
            yerr  = [
                np.maximum(0, np.array(means) - np.array(lows)),
                np.maximum(0, np.array(highs) - np.array(means)),
            ]
            label = MECHANISM_LABELS.get(mech, mech)
            ax.errorbar(
                xs, means, yerr=yerr, label=label,
                color=COLORS[idx % len(COLORS)], marker=MARKERS[idx % len(MARKERS)],
                capsize=3, linewidth=1.8, markersize=6,
            )
        ax.set_xlabel("k / n (Scarcity Ratio)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight="bold")
        ax.legend(fontsize=9, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("E1 — Resource Scarcity Sweep (n=50, T=1000, truthful; M4 λ=1)", fontsize=14, fontweight="bold")
    if save:
        fig.savefig(FIGURES_DIR / "e1_scarcity.pdf", bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "e1_scarcity.png", bbox_inches="tight")
        print("  Saved figures/e1_scarcity.{pdf,png}")
    plt.close(fig)


# ── E2: Fairness–Frontier Scatter ─────────────────────────────────────────────

# Diverging map for signed gains: blue (< 0, manipulation does not pay) ->
# neutral gray (0) -> red (> 0, manipulation pays).
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "gain_diverging", ["#1c5cab", "#f0efec", "#b8322f"]
)


def _signed_norm(values) -> TwoSlopeNorm:
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)])
    lim_lo = min(-1e-9, float(vals.min())) if len(vals) else -1.0
    lim_hi = max(1e-9, float(vals.max())) if len(vals) else 1.0
    return TwoSlopeNorm(vmin=lim_lo, vcenter=0.0, vmax=lim_hi)


def plot_e2(results: list[dict], save: bool = True) -> None:
    """
    Scatter: WR (x) vs J_A (y), coloured by the unilateral manipulation gain
    M_uni (the individual incentive to inflate).  M3, M5 and M4 at λ = 0 share
    one allocation rule and therefore one (WR, J_A) point; M5 is drawn as an
    outer ring because its payments give it a different gain.  The crowded
    high-λ cluster is repeated in a zoomed inset.
    """
    color_key = "M_uni" if "M_uni" in results[0] else "M_mean"

    def short(r: dict) -> str:
        if r["mechanism"] == "ScoreMechanism":
            return f"M4 λ={r['lambda_']:g}"
        return MECHANISM_LABELS.get(r["mechanism"], r["mechanism"])

    # Merge only points that share position AND gain
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        key = (round(r["WR"]["mean"], 4), round(r["J_A"]["mean"], 4),
               round(r[color_key]["mean"], 2))
        groups.setdefault(key, []).append(r)
    pts = []
    for (wr, ja, g), rs in groups.items():
        rs.sort(key=lambda r: (r["mechanism"] != "ScoreMechanism", r.get("lambda_") or 0))
        pts.append(dict(wr=wr, ja=ja, g=g, label=" = ".join(short(r) for r in rs),
                        ring=rs[0]["mechanism"] == "VickreyMechanism"))

    norm = _signed_norm([p["g"] for p in pts])
    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)

    def draw(axis, subset, fs, offsets):
        for p in sorted(subset, key=lambda p: not p["ring"]):
            axis.scatter(p["wr"], p["ja"], c=[p["g"]], cmap=DIVERGING_CMAP, norm=norm,
                         s=260 if p["ring"] else 90, edgecolors="#333333",
                         linewidths=0.8, zorder=2 if p["ring"] else 3)
            dx, dy, ha = offsets(p)
            axis.annotate(p["label"], (p["wr"], p["ja"]), xytext=(dx, dy),
                          textcoords="offset points", fontsize=fs, ha=ha, va="center")

    in_cluster = lambda p: p["wr"] > 0.94 and p["ja"] > 0.95

    def main_offsets(p):
        if p["ring"]:
            return (12, -12, "left")
        if p["wr"] < 0.7:
            return (9, -9 if "Random" in p["label"] else 7, "left")
        return (-10, 0, "right")

    draw(ax, [p for p in pts if not in_cluster(p)], 8.5, main_offsets)
    cluster = [p for p in pts if in_cluster(p)]
    sc = ax.scatter([p["wr"] for p in cluster], [p["ja"] for p in cluster],
               c=[p["g"] for p in cluster], cmap=DIVERGING_CMAP, norm=norm,
               s=90, edgecolors="#333333", linewidths=0.8, zorder=3)

    if cluster:
        x0, x1 = min(p["wr"] for p in cluster) - 0.004, max(p["wr"] for p in cluster) + 0.004
        y0, y1 = min(p["ja"] for p in cluster) - 0.006, 1.004
        axins = ax.inset_axes([0.33, 0.12, 0.40, 0.48])
        draw(axins, cluster, 8.5, lambda p: (8, 0, "left"))
        axins.set_xlim(x0, x1 + 0.006)
        axins.set_ylim(y0, y1)
        axins.tick_params(labelsize=8)
        axins.grid(True, linestyle="--", alpha=0.4)
        axins.set_title("zoom: M4 with λ ≥ 0.5", fontsize=9)
        ax.indicate_inset_zoom(axins, edgecolor="#888888")

    cbar = fig.colorbar(sc, ax=ax)
    lo, hi = norm.vmin, norm.vmax
    # Truncate toward zero so no tick falls outside [vmin, vmax]
    ticks = {0.0} | {float(np.trunc(t)) for t in np.r_[np.linspace(lo, 0, 3), np.linspace(0, hi, 4)]}
    cbar.set_ticks(sorted(ticks))
    cbar.set_label("Unilateral manipulation gain $M_{uni}$ (> 0: inflating pays)"
                   if color_key == "M_uni" else "Mean manipulation gain M",
                   rotation=270, labelpad=18)

    ax.set_xlim(0.52, 1.0)
    ax.set_xlabel("Welfare Ratio (WR)")
    ax.set_ylabel("Jain Allocation Fairness (J_A)")
    ax.set_title("E2 — Fairness / Efficiency / Manipulation Frontier "
                 "(ρ = 0.25, capped exaggeration c = 2)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

    if save:
        fig.savefig(FIGURES_DIR / "e2_frontier.pdf", bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "e2_frontier.png", bbox_inches="tight")
        print("  Saved figures/e2_frontier.{pdf,png}")
    plt.close(fig)


# ── E3: Strategic Population Heatmap ─────────────────────────────────────────

E3_POLICY_ORDER = ["truthful", "cap_2", "max_claim"]
E3_TITLES = {
    "M_mean": "Coalition manipulation gain (strategic set vs. all truthful)",
    "M_uni" : "Unilateral manipulation gain (one user deviates, others fixed)",
    "PoS"   : "Price of Strategy (welfare loss vs. all truthful)",
}


def _e3_grid(results, mech, metric, rhos, policies):
    data = np.full((len(rhos), len(policies)), np.nan)
    for r in results:
        if r["mechanism"] != mech or metric not in r:
            continue
        v = r[metric]["mean"]
        if v is not None:
            data[rhos.index(r["rho"]), policies.index(r["policy"])] = v
    return data


def _draw_e3_panel(ax, data, cmap, norm, rhos, policies, fontsize):
    masked = np.ma.masked_invalid(data)
    cm = cmap.copy()
    cm.set_bad("#e6e6e3")
    im = ax.imshow(masked, aspect="auto", cmap=cm, norm=norm)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=30, ha="right", fontsize=9)
    for i in range(len(rhos)):
        for j in range(len(policies)):
            val = data[i, j]
            txt = "n/a" if np.isnan(val) else f"{val:.2f}" if abs(val) < 10 else f"{val:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=fontsize, color="#1a1a19")
    return im


def plot_e3(results: list[dict], metric: str = "M_mean", save: bool = True) -> None:
    """
    Heatmaps: ρ (rows) × policy (columns) for one metric.  All panels share one
    colour scale (diverging around 0 for signed gains, sequential for PoS) so
    colours are comparable across mechanisms.  Undefined cells show "n/a".
    """
    if not any(metric in r for r in results):
        print(f"  [SKIP] E3 metric {metric} not in results")
        return
    rhos     = sorted({r["rho"] for r in results})
    present  = {r["policy"] for r in results}
    policies = [p for p in E3_POLICY_ORDER if p in present] + sorted(present - set(E3_POLICY_ORDER))
    mechs    = [m for m in MECHANISM_LABELS if any(r["mechanism"] == m for r in results)]

    grids = {m: _e3_grid(results, m, metric, rhos, policies) for m in mechs}
    allv  = np.concatenate([g[~np.isnan(g)] for g in grids.values()])
    if metric == "PoS":
        cmap = plt.get_cmap("Blues")
        norm = Normalize(vmin=0.0, vmax=max(float(allv.max()), 1e-9))
    else:
        cmap = DIVERGING_CMAP
        norm = _signed_norm(allv)
    title = E3_TITLES.get(metric, metric)

    # Individual mechanism heatmaps
    for mech in mechs:
        fig, ax = plt.subplots(figsize=(6.5, 4.8), tight_layout=True)
        im = _draw_e3_panel(ax, grids[mech], cmap, norm, rhos, policies, 9)
        ax.set_yticks(range(len(rhos)))
        ax.set_yticklabels([f"ρ = {r}" for r in rhos])
        plt.colorbar(im, ax=ax, label=metric)
        ax.set_title(f"E3 — {MECHANISM_LABELS.get(mech, mech)}: {metric}", fontweight="bold")
        if save:
            fig.savefig(FIGURES_DIR / f"e3_{mech}_{metric}.pdf", bbox_inches="tight")
            fig.savefig(FIGURES_DIR / f"e3_{mech}_{metric}.png", bbox_inches="tight")
        plt.close(fig)

    # Unified panel figure with one shared colour bar
    fig, axes = plt.subplots(1, len(mechs), figsize=(3.6 * len(mechs) + 1.2, 4.6),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, mech in zip(axes, mechs):
        im = _draw_e3_panel(ax, grids[mech], cmap, norm, rhos, policies, 8)
        ax.set_title(MECHANISM_LABELS.get(mech, mech), fontsize=11, fontweight="bold")
    axes[0].set_yticks(range(len(rhos)))
    axes[0].set_yticklabels([f"ρ = {r}" for r in rhos])
    axes[0].set_ylabel("Strategic fraction ρ")
    fig.colorbar(im, ax=list(axes), shrink=0.9, label=metric)
    fig.suptitle(f"E3 — {title}  (n=50, k=10; M4 λ=1)", fontsize=13, fontweight="bold")

    if save:
        fig.savefig(FIGURES_DIR / f"e3_all_mechs_{metric}.pdf", bbox_inches="tight")
        fig.savefig(FIGURES_DIR / f"e3_all_mechs_{metric}.png", bbox_inches="tight")
        print(f"  Saved figures/e3_all_mechs_{metric}.{{pdf,png}} and individual heatmaps")
    plt.close(fig)


# ── E4: Heterogeneous Users ───────────────────────────────────────────────────

def plot_e4(results: list[dict], save: bool = True) -> None:
    """
    Two-panel comparison of J_A vs J_B:
      Left: Homogeneous (Uniform)
      Right: Heterogeneous (Mixed Beta(2,5) / Beta(5,2))
    """
    dists = ["uniform", "mixed"]
    dist_titles = {
        "uniform": "Homogeneous Population (Uniform)",
        "mixed"  : "Heterogeneous Groups (Beta(2,5) & Beta(5,2))",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True, tight_layout=True)

    mechanisms = [m for m in MECHANISM_LABELS.keys() if any(r["mechanism"] == m for r in results)]
    if not mechanisms:
        mechanisms = sorted({r["mechanism"] for r in results})

    x = np.arange(len(mechanisms))
    width = 0.35

    for ax, dist in zip(axes, dists):
        subset = [r for r in results if r.get("dist") == dist]
        if not subset:
            continue

        for i, (metric, label, offset, color) in enumerate([
            ("J_A", "J_A (Allocation Fairness)", -width/2, "#2b5c8f"),
            ("J_B", "J_B (Benefit Fairness)",     width/2, "#d95f02"),
        ]):
            means = []
            errs  = []
            for m in mechanisms:
                matched = [r for r in subset if r["mechanism"] == m]
                if matched:
                    r = matched[0]
                    means.append(r[metric]["mean"])
                    errs.append(max(0, (r[metric]["ci_upper"] - r[metric]["ci_lower"]) / 2))
                else:
                    means.append(0.0)
                    errs.append(0.0)

            ax.bar(
                x + offset, means, width, label=label, yerr=errs,
                capsize=4, color=color, alpha=0.85, edgecolor="black", linewidth=0.6,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([MECHANISM_LABELS.get(m, m) for m in mechanisms], rotation=25, ha="right", fontsize=9.5)
        ax.set_title(dist_titles.get(dist, dist), fontweight="bold")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        ax.legend(fontsize=9, frameon=True)

    axes[0].set_ylabel("Jain Fairness Index", fontweight="bold")
    fig.suptitle("E4 — Equal-Service vs Equal-Benefit Fairness Divergence (M4 λ=1)", fontsize=14, fontweight="bold")

    if save:
        fig.savefig(FIGURES_DIR / "e4_heterogeneous.pdf", bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "e4_heterogeneous.png", bbox_inches="tight")
        print("  Saved figures/e4_heterogeneous.{pdf,png}")
    plt.close(fig)


# ── E5: Temporal Persistence ──────────────────────────────────────────────────

def plot_e5(results: list[dict], save: bool = True) -> None:
    """
    E5: Line plots with 95% CI bands vs AR(1) persistence coefficient α.
    """
    metrics = [
        ("WR", "Welfare Ratio (WR)"),
        ("J_A", "Jain Allocation Fairness (J_A)"),
        ("SR_delta", "Starvation Rate (SR_Δ)"),
        ("pct95_wait", "95th-pct Consecutive Wait"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), tight_layout=True)
    axes = axes.flatten()

    mechanisms = [m for m in MECHANISM_LABELS.keys() if any(r["mechanism"] == m for r in results)]
    if not mechanisms:
        mechanisms = sorted({r["mechanism"] for r in results})

    for ax, (metric, ylabel) in zip(axes, metrics):
        for idx, mech in enumerate(mechanisms):
            subset = [r for r in results if r["mechanism"] == mech]
            subset.sort(key=lambda r: r["alpha"])
            xs    = [r["alpha"] for r in subset]
            means = [r[metric]["mean"] for r in subset]
            lows  = [r[metric]["ci_lower"] for r in subset]
            highs = [r[metric]["ci_upper"] for r in subset]
            yerr  = [
                np.maximum(0, np.array(means) - np.array(lows)),
                np.maximum(0, np.array(highs) - np.array(means)),
            ]
            label = MECHANISM_LABELS.get(mech, mech)
            ax.errorbar(
                xs, means, yerr=yerr, label=label,
                color=COLORS[idx % len(COLORS)], marker=MARKERS[idx % len(MARKERS)],
                capsize=3, linewidth=1.8, markersize=6,
            )
        ax.set_xlabel("AR(1) Persistence α (0 = i.i.d., 0.9 = high autocorrelation)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight="bold")
        ax.legend(fontsize=9, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("E5 — Temporal Persistence in Valuations (M4 λ=1)", fontsize=14, fontweight="bold")
    if save:
        fig.savefig(FIGURES_DIR / "e5_persistence.pdf", bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "e5_persistence.png", bbox_inches="tight")
        print("  Saved figures/e5_persistence.{pdf,png}")
    plt.close(fig)


# ── E6: Scalability ───────────────────────────────────────────────────────────

def plot_e6(results: list[dict], save: bool = True) -> None:
    """
    E6: Log-log wall-clock time and peak memory vs n.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), tight_layout=True)

    mechanisms = [m for m in MECHANISM_LABELS.keys() if any(r["mechanism"] == m for r in results)]
    if not mechanisms:
        mechanisms = sorted({r["mechanism"] for r in results})

    for idx, mech in enumerate(mechanisms):
        subset = [r for r in results if r["mechanism"] == mech]
        subset.sort(key=lambda r: r["n"])
        ns = [r["n"] for r in subset]
        times = [r["time_per_round_us"]["mean"] for r in subset]
        mems  = [r["peak_memory_kib"]["mean"] for r in subset]
        label = MECHANISM_LABELS.get(mech, mech)

        ax1.loglog(ns, times, marker=MARKERS[idx % len(MARKERS)], color=COLORS[idx % len(COLORS)],
                   linewidth=1.8, markersize=6, label=label)
        ax2.loglog(ns, mems, marker=MARKERS[idx % len(MARKERS)], color=COLORS[idx % len(COLORS)],
                   linewidth=1.8, markersize=6, label=label)

    ax1.set_xlabel("Population Size n (k = 0.2·n)", fontweight="bold")
    ax1.set_ylabel("Runtime per Round (µs)", fontweight="bold")
    ax1.set_title("Runtime Scaling", fontweight="bold")
    ax1.legend(fontsize=9, frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.4, which="both")

    ax2.set_xlabel("Population Size n (k = 0.2·n)", fontweight="bold")
    ax2.set_ylabel("State array size (KiB, analytic)", fontweight="bold")
    ax2.set_title("Simulation State Size", fontweight="bold")
    ax2.legend(fontsize=9, frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.4, which="both")

    fig.suptitle("E6 — Computational Scalability Benchmark", fontsize=14, fontweight="bold")
    if save:
        fig.savefig(FIGURES_DIR / "e6_scalability.pdf", bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "e6_scalability.png", bbox_inches="tight")
        print("  Saved figures/e6_scalability.{pdf,png}")
    plt.close(fig)


# ── Master loader and CLI entry point ─────────────────────────────────────────

def run_all_plots() -> None:
    results_dir = Path("results")

    # E1
    e1_path = results_dir / "e1" / "summary.json"
    if e1_path.exists():
        with open(e1_path) as fh:
            data = json.load(fh)
        plot_e1(data)
    else:
        print(f"[SKIP] {e1_path} not found.")

    # E2
    e2_path = results_dir / "e2" / "summary.json"
    if e2_path.exists():
        with open(e2_path) as fh:
            data = json.load(fh)
        plot_e2(data)
    else:
        print(f"[SKIP] {e2_path} not found.")

    # E3
    e3_path = results_dir / "e3" / "summary.json"
    if e3_path.exists():
        with open(e3_path) as fh:
            data = json.load(fh)
        plot_e3(data, metric="M_mean")
        plot_e3(data, metric="M_uni")
        plot_e3(data, metric="PoS")
    else:
        print(f"[SKIP] {e3_path} not found.")

    # E4
    e4_path = results_dir / "e4" / "summary.json"
    if e4_path.exists():
        with open(e4_path) as fh:
            data = json.load(fh)
        plot_e4(data)
    else:
        print(f"[SKIP] {e4_path} not found.")

    # E5
    e5_path = results_dir / "e5" / "summary.json"
    if e5_path.exists():
        with open(e5_path) as fh:
            data = json.load(fh)
        plot_e5(data)
    else:
        print(f"[SKIP] {e5_path} not found.")

    # E6
    e6_path = results_dir / "e6" / "summary.json"
    if e6_path.exists():
        with open(e6_path) as fh:
            data = json.load(fh)
        plot_e6(data)
    else:
        print(f"[SKIP] {e6_path} not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all figures from saved results.")
    parser.add_argument("--all", action="store_true", help="Regenerate all figures from results/ summaries.")
    args = parser.parse_args()

    if args.all:
        run_all_plots()
    else:
        run_all_plots()
