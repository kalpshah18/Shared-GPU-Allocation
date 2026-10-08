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

import matplotlib

matplotlib.use("Agg")                       # headless: figures are written to files
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sim.mechanisms import MECHANISM_LABELS      # noqa: E402

FIGURES_DIR = Path("figures")              # overridden by run_all_plots(figures_dir=...)


def _save(fig, stem: str) -> None:
    """Write `stem`.pdf and `stem`.png into FIGURES_DIR."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.png", bbox_inches="tight")
    print(f"  Saved {FIGURES_DIR / stem}.{{pdf,png}}")


def _mechanisms_in(results: list) -> list:
    present = [m for m in MECHANISM_LABELS if any(r.get("mechanism") == m for r in results)]
    return present or sorted({r["mechanism"] for r in results})


def _errbars(rows: list, metric: str):
    means = np.array([r[metric]["mean"] for r in rows], dtype=float)
    lows  = np.array([r[metric]["ci_lower"] for r in rows], dtype=float)
    highs = np.array([r[metric]["ci_upper"] for r in rows], dtype=float)
    return means, [np.maximum(0, means - lows), np.maximum(0, highs - means)]


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

def plot_e1(results: list, save: bool = True) -> None:
    """Line plots of WR, J_A, J_B, Q_max, 95th-pct wait and SR_Delta vs k/n."""
    metrics = [
        ("WR", "Welfare Ratio (WR)"),
        ("J_A", "Jain Allocation Fairness (J_A)"),
        ("J_B", "Jain Benefit Fairness (J_B)"),
        ("Q_max", "Maximum Consecutive Wait (Q_max)"),
        ("pct95_wait", "95th-pct Consecutive Wait"),
        ("SR_delta", "Starvation Rate (SR_Δ)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), tight_layout=True)
    mechanisms = _mechanisms_in(results)
    for ax, (metric, ylabel) in zip(axes.flatten(), metrics):
        for idx, mech in enumerate(mechanisms):
            subset = sorted((r for r in results if r["mechanism"] == mech), key=lambda r: r["kn_ratio"])
            means, yerr = _errbars(subset, metric)
            ax.errorbar([r["kn_ratio"] for r in subset], means, yerr=yerr,
                        label=MECHANISM_LABELS.get(mech, mech), color=COLORS[idx % len(COLORS)],
                        marker=MARKERS[idx % len(MARKERS)], capsize=3, linewidth=1.8, markersize=6)
        ax.set_xlabel("k / n (Scarcity Ratio)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight="bold")
        ax.legend(fontsize=8, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("E1 — Resource Scarcity Sweep (truthful reports; M4 λ=1; 95% bootstrap CIs)",
                 fontsize=14, fontweight="bold")
    if save:
        _save(fig, "e1_scarcity")
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
               round(r[color_key]["mean"], 2), bool(r.get("pareto", False)))
        groups.setdefault(key, []).append(r)
    pts = []
    for (wr, ja, g, par), rs in groups.items():
        rs.sort(key=lambda r: (r["mechanism"] != "ScoreMechanism", r.get("lambda_") or 0))
        pts.append(dict(wr=wr, ja=ja, g=g, label=" = ".join(short(r) for r in rs),
                        ring=rs[0]["mechanism"] == "VickreyMechanism", pareto=par))

    norm = _signed_norm([p["g"] for p in pts])
    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)

    def draw(axis, subset, fs, offsets):
        for p in sorted(subset, key=lambda p: not p["ring"]):
            axis.scatter(p["wr"], p["ja"], c=[p["g"]], cmap=DIVERGING_CMAP, norm=norm,
                         s=260 if p["ring"] else 90,
                         edgecolors="#000000" if p["pareto"] else "#999999",
                         linewidths=2.2 if p["pareto"] else 0.8, zorder=2 if p["ring"] else 3)
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
               s=90, edgecolors=["#000000" if p["pareto"] else "#999999" for p in cluster],
               linewidths=[2.2 if p["pareto"] else 0.8 for p in cluster], zorder=3)

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

    ax.set_xlim(max(0.0, min(p["wr"] for p in pts) - 0.05), 1.0)
    ax.set_xlabel("Welfare Ratio (WR)")
    ax.set_ylabel("Jain Allocation Fairness (J_A)")
    ax.set_title("E2 — Fairness / Efficiency / Manipulation Frontier "
                 "(ρ = 0.25, capped exaggeration c = 2)\nbold outline = Pareto non-dominated "
                 "(WR↑, J_A↑, SR_Δ↓, M_uni↓)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

    if save:
        _save(fig, "e2_frontier")
    plt.close(fig)


# ── E3: Strategic Population Heatmap ─────────────────────────────────────────

E3_POLICY_ORDER = ["truthful", "cap_2", "max_claim"]
E3_TITLES = {
    "M_mean"      : "Coalition manipulation gain (strategic set vs. all truthful)",
    "M_uni"       : "Unilateral manipulation gain, mean over focal users",
    "M_uni_max"   : "Unilateral manipulation gain, maximum over focal users",
    "frac_pos_uni": "Fraction of focal users who gain from deviating",
    "PoS"         : "Price of Strategy (welfare loss vs. all truthful)",
    "WR"          : "Welfare ratio under the strategic profile",
    "J_A"         : "Jain allocation fairness under the strategic profile",
}
E3_SEQUENTIAL = {"PoS", "WR", "J_A", "frac_pos_uni"}


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
    cm = cmap.with_extremes(bad="#e6e6e3")
    im = ax.imshow(masked, aspect="auto", cmap=cm, norm=norm)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=30, ha="right", fontsize=9)
    for i in range(len(rhos)):
        for j in range(len(policies)):
            val = data[i, j]
            txt = "n/a" if np.isnan(val) else f"{val:.2f}" if abs(val) < 10 else f"{val:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=fontsize, color="#1a1a19")
    return im


def plot_e3(results: list, metric: str = "M_mean", save: bool = True) -> None:
    """
    Heatmaps: rho (rows) x policy (columns) for one metric, one panel per
    mechanism, all panels on one shared colour scale (diverging around 0 for
    signed gains, sequential otherwise).  Undefined cells show "n/a".
    """
    if not any(metric in r for r in results):
        print(f"  [SKIP] E3 metric {metric} not in results")
        return
    rhos     = sorted({r["rho"] for r in results})
    present  = {r["policy"] for r in results}
    policies = [p for p in E3_POLICY_ORDER if p in present] + sorted(present - set(E3_POLICY_ORDER))
    mechs    = _mechanisms_in(results)

    grids = {m: _e3_grid(results, m, metric, rhos, policies) for m in mechs}
    allv  = np.concatenate([g[~np.isnan(g)] for g in grids.values()])
    if allv.size == 0:
        print(f"  [SKIP] E3 metric {metric} has no defined cells")
        return
    if metric in E3_SEQUENTIAL:
        cmap = plt.get_cmap("Blues")
        norm = Normalize(vmin=min(0.0, float(allv.min())), vmax=max(float(allv.max()), 1e-9))
    else:
        cmap = DIVERGING_CMAP
        norm = _signed_norm(allv)
    title = E3_TITLES.get(metric, metric)

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
    fig.suptitle(f"E3 — {title}  (M4 λ=1)", fontsize=13, fontweight="bold")
    if save:
        _save(fig, f"e3_all_mechs_{metric}")
    plt.close(fig)


# ── E3b: Rollout attack ───────────────────────────────────────────────────────

def plot_e3b(results: list, save: bool = True) -> None:
    """Bars: unilateral gain of the rollout attack vs capped exaggeration."""
    opps = [o for o in ("truthful", "cap_2") if any(r["opponents"] == o for r in results)]
    labels = list(dict.fromkeys(r["label"] for r in results))
    fig, axes = plt.subplots(1, len(opps), figsize=(6.5 * len(opps), 5), sharey=True, tight_layout=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(labels))
    w = 0.38
    for ax, opp in zip(axes, opps):
        for off, key, name, color in [(-w / 2, "M_rollout", "rollout attack (H=5)", "#b8322f"),
                                      (w / 2, "M_cap2", "capped exaggeration c=2", "#1c5cab")]:
            rows = [next(r for r in results if r["label"] == lab and r["opponents"] == opp) for lab in labels]
            means, yerr = _errbars(rows, key)
            ax.bar(x + off, means, w, yerr=yerr, capsize=4, color=color, alpha=0.85,
                   edgecolor="black", linewidth=0.6, label=name)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(f"Opponents: {opp}", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Unilateral gain of the focal user")
    fig.suptitle("E3b — Rollout attack vs scalable policy (n=10; gain > 0: manipulation pays)",
                 fontsize=13, fontweight="bold")
    if save:
        _save(fig, "e3b_rollout")
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
        _save(fig, "e4_heterogeneous")
    plt.close(fig)


# ── E5: Temporal Persistence ──────────────────────────────────────────────────

def plot_e5(results: list, save: bool = True) -> None:
    """
    E5: metrics vs the AR(1) persistence alpha.  Solid lines use the proposal's
    recursion; dashed lines use the marginal-preserving copula (alpha = 0 is
    shared).  Error bars are 95% bootstrap CIs.
    """
    metrics = [
        ("WR", "Welfare Ratio (WR)"),
        ("J_A", "Jain Allocation Fairness (J_A)"),
        ("SR_delta", "Starvation Rate (SR_Δ)"),
        ("pct95_wait", "95th-pct Consecutive Wait"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), tight_layout=True)
    mechanisms = _mechanisms_in(results)
    for ax, (metric, ylabel) in zip(axes.flatten(), metrics):
        for idx, mech in enumerate(mechanisms):
            for mode, ls in (("proposal", "-"), ("copula", "--")):
                subset = sorted((r for r in results if r["mechanism"] == mech
                                 and (r.get("ar1_mode", "proposal") == mode
                                      or (mode == "copula" and r["alpha"] == 0.0))),
                                key=lambda r: r["alpha"])
                if not subset:
                    continue
                means, yerr = _errbars(subset, metric)
                name = MECHANISM_LABELS.get(mech, mech)
                ax.errorbar([r["alpha"] for r in subset], means, yerr=yerr, linestyle=ls,
                            label=name if mode == "proposal" else f"{name} (copula)",
                            color=COLORS[idx % len(COLORS)], marker=MARKERS[idx % len(MARKERS)],
                            capsize=3, linewidth=1.6, markersize=5, alpha=0.9 if ls == "-" else 0.6)
        ax.set_xlabel("AR(1) persistence α")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0, 0].legend(fontsize=7, frameon=True, ncol=2)
    fig.suptitle("E5 — Temporal persistence (M4 λ=1). Solid: proposal AR(1); dashed: marginal-preserving copula",
                 fontsize=12, fontweight="bold")
    if save:
        _save(fig, "e5_persistence")
    plt.close(fig)


# ── E6: Scalability ───────────────────────────────────────────────────────────

def plot_e6(results: list, save: bool = True) -> None:
    """E6: log-log wall-clock time per round and measured peak memory vs n."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), tight_layout=True)
    for idx, mech in enumerate(_mechanisms_in(results)):
        subset = sorted((r for r in results if r["mechanism"] == mech), key=lambda r: r["n"])
        ns = [r["n"] for r in subset]
        style = dict(marker=MARKERS[idx % len(MARKERS)], color=COLORS[idx % len(COLORS)],
                     linewidth=1.8, markersize=6, label=MECHANISM_LABELS.get(mech, mech))
        ax1.loglog(ns, [r["time_per_round_us"]["mean"] for r in subset], **style)
        ax2.loglog(ns, [r["peak_memory_kib"]["mean"] for r in subset], **style)
    ax1.set_xlabel("Population Size n (k = 0.2·n)", fontweight="bold")
    ax1.set_ylabel("Runtime per Round (µs)", fontweight="bold")
    ax1.set_title("Runtime Scaling", fontweight="bold")
    ax2.set_xlabel("Population Size n (k = 0.2·n)", fontweight="bold")
    ax2.set_ylabel("Peak heap memory (KiB, tracemalloc)", fontweight="bold")
    ax2.set_title("Measured Peak Memory (T=1000)", fontweight="bold")
    for ax in (ax1, ax2):
        ax.legend(fontsize=9, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.4, which="both")
    fig.suptitle("E6 — Computational Scalability Benchmark", fontsize=14, fontweight="bold")
    if save:
        _save(fig, "e6_scalability")
    plt.close(fig)


# ── E3c: Rollout horizon ──────────────────────────────────────────────────────

def plot_e3c(results: list, save: bool = True) -> None:
    """Rollout-attack gain on M4 versus planning horizon H, against capped exaggeration."""
    lams = sorted({r["lambda_"] for r in results})
    fig, axes = plt.subplots(1, len(lams), figsize=(6.5 * len(lams), 5), sharey=True, tight_layout=True)
    axes = np.atleast_1d(axes)
    for ax, lam in zip(axes, lams):
        rows = sorted((r for r in results if r["lambda_"] == lam), key=lambda r: r["H"])
        means, yerr = _errbars(rows, "M_rollout")
        ax.errorbar([r["H"] for r in rows], means, yerr=yerr, color="#b8322f", marker="o",
                    capsize=3, linewidth=2, label="rollout attack")
        for key, color, name in (("M_cap2", "#1c5cab", "capped c=2"), ("M_cap1.25", "#2ca02c", "capped c=1.25")):
            ax.axhline(rows[0][key]["mean"], color=color, linestyle="--", linewidth=1.5, label=name)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([r["H"] for r in rows])
        ax.set_xticklabels([str(r["H"]) for r in rows])
        ax.set_xlabel("Rollout horizon H (rounds)")
        ax.set_title(f"M4 λ={lam:g}, opponents truthful", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Unilateral gain of the focal user")
    fig.suptitle("E3c — Does a longer horizon find a profitable attack on Score? (n=10)",
                 fontsize=13, fontweight="bold")
    if save:
        _save(fig, "e3c_rollout_horizon")
    plt.close(fig)


# ── E8: Timing attack ─────────────────────────────────────────────────────────

E8_STYLES = {"always": ("#333333", "-", "o"), "timed": ("#b8322f", "-", "s"),
             "timed_q25": ("#ff7f0e", "--", "^"), "timed_q75": ("#1c5cab", "--", "v")}


def plot_e8(results: list, save: bool = True) -> None:
    """
    Top row: unilateral gain of M4 vs lambda for always-inflating and three timed
    variants.  Bottom row: the timing premium (timed minus always, same omega).
    One column per exaggeration factor c; a positive premium means timing helps.
    """
    cs = sorted({r["c"] for r in results})
    fig, axes = plt.subplots(2, len(cs), figsize=(6.5 * len(cs), 9), sharex=True, tight_layout=True,
                             squeeze=False)
    m4 = [r for r in results if r["mechanism"] == "ScoreMechanism"]
    for j, c in enumerate(cs):
        top, bottom = axes[0, j], axes[1, j]
        base = {r["lambda_"]: r["M_uni"]["mean"] for r in m4 if r["c"] == c and r["variant"] == "always"}
        for vname, (color, ls, mk) in E8_STYLES.items():
            rows = sorted((r for r in m4 if r["c"] == c and r["variant"] == vname), key=lambda r: r["lambda_"])
            if not rows:
                continue
            xs = [r["lambda_"] for r in rows]
            means, yerr = _errbars(rows, "M_uni")
            top.errorbar(xs, means, yerr=yerr, color=color, linestyle=ls, marker=mk, capsize=3,
                         linewidth=1.8, label=vname)
            if vname != "always":
                bottom.plot(xs, [m - base[x] for m, x in zip(means, xs)], color=color, linestyle=ls,
                            marker=mk, linewidth=1.8, label=vname)
        for ax in (top, bottom):
            ax.axhline(0, color="#333333", linewidth=0.8)
            ax.set_xscale("log")
            ax.set_xticks([0.5, 1, 2, 5])
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, linestyle="--", alpha=0.4)
        top.set_title(f"c = {c:g}", fontweight="bold")
        top.set_ylabel("Unilateral gain $M_{uni}$")
        top.legend(fontsize=9)
        bottom.set_ylabel("Timing premium  M(timed) - M(always)")
        bottom.set_xlabel("History penalty λ")
        bottom.legend(fontsize=9)
    fig.suptitle("E8 — Does timing reports pay? (M4, ρ=0.25).  Premium > 0 would support H2",
                 fontsize=13, fontweight="bold")
    if save:
        _save(fig, "e8_timing")
    plt.close(fig)


def plot_e7(results: list, save: bool = True) -> None:
    """
    Left: unilateral gain of M4 vs lambda, one line per exaggeration factor c.
    Right: PoS vs c for every mechanism.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), tight_layout=True)
    m4 = [r for r in results if r["mechanism"] == "ScoreMechanism"]
    cs = sorted({r["c"] for r in results})
    palette = {c: col for c, col in zip(cs, ["#2ca02c", "#ff7f0e", "#b8322f", "#1c5cab"])}
    for c in cs:
        rows = sorted((r for r in m4 if r["c"] == c), key=lambda r: r["lambda_"])
        if not rows:
            continue
        means, yerr = _errbars(rows, "M_uni")
        ax1.errorbar([r["lambda_"] for r in rows], means, yerr=yerr, color=palette[c], marker="o",
                     capsize=3, linewidth=1.8, label=f"c = {c:g}")
    ax1.axhline(0, color="#333333", linewidth=0.8)
    ax1.set_xscale("log")
    ax1.set_xticks(sorted({r["lambda_"] for r in m4}))
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.xaxis.set_minor_formatter(NullFormatter())
    ax1.set_xlabel("History penalty λ (M4)")
    ax1.set_ylabel("Unilateral gain $M_{uni}$ (> 0: inflating pays)")
    ax1.set_title("How much penalty deters each lie?", fontweight="bold")
    ax1.legend(title="exaggeration", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    labels = list(dict.fromkeys(r["label"] for r in results))
    for idx, lab in enumerate(labels):
        subset = sorted((r for r in results if r["label"] == lab), key=lambda r: r["c"])
        means, yerr = _errbars(subset, "PoS")
        ax2.errorbar([r["c"] for r in subset], means, yerr=yerr, label=lab,
                     color=COLORS[idx % len(COLORS)], marker=MARKERS[idx % len(MARKERS)],
                     capsize=3, linewidth=1.6, markersize=5)
    ax2.set_xlabel("Exaggeration factor c")
    ax2.set_ylabel("Price of Strategy (PoS)")
    ax2.set_xticks(cs)
    ax2.set_title("Welfare cost of the lie", fontweight="bold")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("E7 — Sensitivity to the capped-exaggeration factor (ρ = 0.25)",
                 fontsize=14, fontweight="bold")
    if save:
        _save(fig, "e7_sensitivity")
    plt.close(fig)


# ── E9-E11: proposed mechanisms ───────────────────────────────────────────────

PROPOSED_COLORS = {
    "M1 Random": "#b0b0b0", "M2 Round-Robin": "#8c8c8c", "M3 Greedy": "#2ca02c",
    "M4 Score λ=1": "#6baed6", "M4 Score λ=2": "#2171b5", "M5 Vickrey": "#9467bd",
    "M6 Karma-Cap": "#d62728", "M7 Rank-Cap": "#ff7f0e", "M7 Rank-Cap β=0.25": "#fdae6b",
}


def _label_order(results: list) -> list:
    return [lab for lab in PROPOSED_COLORS if any(r.get("label") == lab for r in results)]


def plot_e9(results: list, save: bool = True) -> None:
    """Truthful comparison of baselines and proposed mechanisms across conditions."""
    conds = [c for c in ("base", "scarce", "loose", "mixed", "persist_proposal", "persist_copula")
             if any(r["condition"] == c for r in results)]
    labels = _label_order([r for r in results if r["condition"] in conds])
    metrics = [("WR", "Welfare Ratio (WR)"), ("J_B", "Jain benefit fairness (J_B)"),
               ("Q_max", "Maximum consecutive wait Q_max"), ("SR_delta", "Starvation rate SR_Δ")]
    fig, axes = plt.subplots(2, 2, figsize=(17, 9.5), tight_layout=True)
    width = 0.85 / max(len(labels), 1)
    for ax, (metric, title) in zip(axes.flatten(), metrics):
        for j, lab in enumerate(labels):
            rows = [next((r for r in results if r["condition"] == c and r["label"] == lab), None) for c in conds]
            vals = [r[metric]["mean"] if r else np.nan for r in rows]
            errs = [max(0.0, (r[metric]["ci_upper"] - r[metric]["ci_lower"]) / 2) if r else 0.0 for r in rows]
            ax.bar(np.arange(len(conds)) + (j - (len(labels) - 1) / 2) * width, vals, width, yerr=errs,
                   color=PROPOSED_COLORS[lab], label=lab, edgecolor="black", linewidth=0.4, capsize=1.5)
        ax.set_xticks(np.arange(len(conds)))
        ax.set_xticklabels(conds, rotation=15)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    axes[0, 0].legend(fontsize=7, ncol=3, loc="lower left")
    fig.suptitle("E9 — Proposed mechanisms vs baselines under truthful reports (95% bootstrap CIs)",
                 fontsize=14, fontweight="bold")
    if save:
        _save(fig, "e9_proposed_truthful")
    plt.close(fig)


def plot_e9_frontier(results: list, save: bool = True) -> None:
    """Welfare vs worst-case wait: baselines as points, M6/M7 as curves over the cap W."""
    base = [r for r in results if r["condition"] == "base"]
    fig, ax = plt.subplots(figsize=(10, 6.5), tight_layout=True)
    for r in base:
        lab = r["label"]
        if lab in ("M6 Karma-Cap", "M7 Rank-Cap", "M7 Rank-Cap β=0.25"):
            continue
        ax.scatter(r["Q_max"]["mean"], r["WR"]["mean"], s=110, color=PROPOSED_COLORS[lab],
                   edgecolor="black", zorder=3)
        ax.annotate(lab, (r["Q_max"]["mean"], r["WR"]["mean"]), xytext=(7, 5), textcoords="offset points",
                    fontsize=9)
    for lab, mk in (("M6 Karma-Cap", "o"), ("M7 Rank-Cap", "s")):
        pts = sorted((r for r in results if r["condition"] == "wait_frontier" and r["label"] == lab),
                     key=lambda r: r["wait_cap"])
        if not pts:
            continue
        ax.plot([r["Q_max"]["mean"] for r in pts], [r["WR"]["mean"] for r in pts], marker=mk,
                color=PROPOSED_COLORS[lab], linewidth=2, markersize=7, label=f"{lab} (cap W swept)")
        for r in pts:
            ax.annotate(f"W={r['wait_cap']}", (r["Q_max"]["mean"], r["WR"]["mean"]), xytext=(4, -12),
                        textcoords="offset points", fontsize=7, color=PROPOSED_COLORS[lab])
    ax.set_xscale("log")
    ax.set_xlabel("Worst-case wait Q_max (log scale; lower is better)")
    ax.set_ylabel("Welfare ratio WR (higher is better)")
    ax.set_title("E9 — Welfare vs guaranteed waiting (truthful, n=50, k=10)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, which="both")
    ax.legend(fontsize=9, loc="center right")
    if save:
        _save(fig, "e9_wait_frontier")
    plt.close(fig)


E10_POLICIES = ["cap_1.25", "cap_1.5", "cap_2", "max_claim", "timed_cap_1.25", "timed_cap_2"]


def plot_e10(results: list, save: bool = True) -> None:
    """Heatmap of the unilateral gain: mechanism x lie (rho = 0.25), plus rho = 1 stress columns."""
    cols = [(p, 0.25) for p in E10_POLICIES] + [("cap_2", 1.0), ("max_claim", 1.0)]
    cols = [c for c in cols if any(r["policy"] == c[0] and r["rho"] == c[1] for r in results)]
    labels = _label_order([r for r in results if r["policy"] in E10_POLICIES])
    labels += [lab for lab in dict.fromkeys(r["label"] for r in results) if lab not in labels]
    data = np.full((len(labels), len(cols)), np.nan)
    for i, lab in enumerate(labels):
        for j, (pol, rho) in enumerate(cols):
            r = next((r for r in results if r["label"] == lab and r["policy"] == pol and r["rho"] == rho), None)
            if r and r["M_uni"]["mean"] is not None:
                data[i, j] = r["M_uni"]["mean"]
    # signed log scale so +306 (Greedy) does not wash out the +/-few differences that matter
    shown = np.sign(data) * np.log1p(np.abs(data))
    lim = np.nanmax(np.abs(shown)) if np.isfinite(shown).any() else 1.0
    fig, ax = plt.subplots(figsize=(12.5, 0.55 * len(labels) + 3), tight_layout=True)
    im = ax.imshow(np.ma.masked_invalid(shown), aspect="auto", cmap=DIVERGING_CMAP,
                   norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{p}\nρ={r:g}" for p, r in cols], fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(cols)):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:+.1f}", ha="center", va="center", fontsize=8.5,
                        fontweight="bold" if data[i, j] > 0 else "normal")
    fig.colorbar(im, ax=ax, label="signed log colour scale (red: lying pays, blue: lying loses)")
    ax.set_title("E10 — Unilateral gain from lying, by mechanism and lie (numbers = utility units; bold = lying pays)",
                 fontsize=11, fontweight="bold")
    if save:
        _save(fig, "e10_proposed_strategic")
    plt.close(fig)


def plot_e11(results: list, score_ref: list | None = None, save: bool = True) -> None:
    """Rollout-attack gain versus horizon for the proposed mechanisms (and Score from E3c)."""
    fig, ax = plt.subplots(figsize=(9, 5.5), tight_layout=True)
    series = []
    for lab in dict.fromkeys(r["label"] for r in results):
        series.append((lab, sorted((r for r in results if r["label"] == lab), key=lambda r: r["H"]),
                       PROPOSED_COLORS.get(lab, "#333333"), "o"))
    if score_ref:
        for lam, color in ((1.0, "#6baed6"), (2.0, "#2171b5")):
            rows = sorted((r for r in score_ref if r["lambda_"] == lam), key=lambda r: r["H"])
            if rows:
                series.append((f"M4 Score λ={lam:g} (E3c)", rows, color, "s"))
    for lab, rows, color, mk in series:
        means, yerr = _errbars(rows, "M_rollout")
        ax.errorbar([r["H"] for r in rows], means, yerr=yerr, color=color, marker=mk, capsize=3,
                    linewidth=2, label=lab)
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_xscale("log")
    hs = sorted({r["H"] for r in results})
    ax.set_xticks(hs)
    ax.set_xticklabels([str(h) for h in hs])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Rollout horizon H (rounds)")
    ax.set_ylabel("Unilateral gain of the far-sighted attacker")
    ax.set_title("E11 — Far-sighted attacker vs proposed mechanisms (n=10, T=300)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    if save:
        _save(fig, "e11_proposed_rollout")
    plt.close(fig)


# ── Master loader and CLI entry point ─────────────────────────────────────────

def _load(path: Path):
    if not path.exists():
        print(f"[SKIP] {path} not found.")
        return None
    with open(path) as fh:
        return json.load(fh)


def run_all_plots(results_dir: str = "results", figures_dir: str = "figures") -> None:
    """Regenerate every figure from results/*/summary.json."""
    global FIGURES_DIR
    FIGURES_DIR = Path(figures_dir)
    rd = Path(results_dir)

    if (d := _load(rd / "e1" / "summary.json")) is not None:
        plot_e1(d)
    if (d := _load(rd / "e2" / "summary.json")) is not None:
        plot_e2(d)
    if (d := _load(rd / "e3" / "summary.json")) is not None:
        for metric in ("M_mean", "M_uni", "M_uni_max", "frac_pos_uni", "PoS", "WR", "J_A"):
            plot_e3(d, metric=metric)
    if (d := _load(rd / "e3b" / "summary.json")) is not None:
        plot_e3b(d)
    if (d := _load(rd / "e3c" / "summary.json")) is not None:
        plot_e3c(d)
    if (d := _load(rd / "e4" / "summary.json")) is not None:
        plot_e4(d)
    if (d := _load(rd / "e5" / "summary.json")) is not None:
        plot_e5(d)
    if (d := _load(rd / "e6" / "summary.json")) is not None:
        plot_e6(d)
    if (d := _load(rd / "e7" / "summary.json")) is not None:
        plot_e7(d)
    if (d := _load(rd / "e8" / "summary.json")) is not None:
        plot_e8(d)
    if (d := _load(rd / "e9" / "summary.json")) is not None:
        plot_e9(d)
        plot_e9_frontier(d)
    if (d := _load(rd / "e10" / "summary.json")) is not None:
        plot_e10(d)
    if (d := _load(rd / "e11" / "summary.json")) is not None:
        e3c = rd / "e3c" / "summary.json"
        plot_e11(d, json.loads(e3c.read_text()) if e3c.exists() else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all figures from saved results.")
    parser.add_argument("--all", action="store_true", help="(default) regenerate every figure")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()
    run_all_plots(args.results_dir, args.figures_dir)
