"""
analysis/report.py
==================
Render the saved experiment summaries as Markdown tables (results/RESULTS.md),
so every number quoted in the README can be regenerated rather than retyped.

Usage
-----
    python analysis/report.py [--results-dir results] [--out results/RESULTS.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sim.mechanisms import MECHANISM_LABELS, MECHANISM_NAMES   # noqa: E402


def _f(row: dict, key: str, digits: int = 3, ci: bool = False, signed: bool = False) -> str:
    cell = row.get(key)
    if not cell or cell.get("mean") is None:
        return "n/a"
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}"
    text = fmt.format(cell["mean"])
    if ci and cell.get("ci_lower") is not None:
        text += f" [{cell['ci_lower']:.{digits}f}, {cell['ci_upper']:.{digits}f}]"
    return text


def _table(header: list, rows: list) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _load(rd: Path, exp: str):
    p = rd / exp / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def _pick(rows, **conds):
    for r in rows:
        if all(r.get(k) == v for k, v in conds.items()):
            return r
    return None


def section_e1(rows) -> str:
    out = ["## E1 — Resource scarcity (truthful reports)\n"]
    for kn in sorted({r["kn_ratio"] for r in rows}):
        out.append(f"\n**k/n = {kn}**\n")
        body = []
        for m in MECHANISM_NAMES:
            r = _pick(rows, mechanism=m, kn_ratio=kn)
            if r:
                body.append([MECHANISM_LABELS[m], _f(r, "WR"), _f(r, "J_A", 4), _f(r, "J_B", 4),
                             _f(r, "Q_max", 1), _f(r, "pct95_wait", 1), _f(r, "SR_delta", 4), _f(r, "PoF")])
        out.append(_table(["Mechanism", "WR", "J_A", "J_B", "Q_max", "p95 wait", "SR_Δ", "PoF"], body))
    return "\n".join(out)


def section_e2(rows) -> str:
    body = []
    for r in rows:
        name = (MECHANISM_LABELS[r["mechanism"]] if r["lambda_"] is None
                else f"M4 λ={r['lambda_']:g}")
        body.append([name + (" ★" if r.get("pareto") else ""), _f(r, "WR"), _f(r, "J_A", 4),
                     _f(r, "SR_delta", 3), _f(r, "M_mean", 1, signed=True),
                     _f(r, "M_uni", 1, ci=True, signed=True), _f(r, "frac_pos_uni", 2), _f(r, "PoS", 3)])
    return ("## E2 — Fairness / manipulation frontier (ρ=0.25, capped exaggeration c=2)\n\n"
            "★ = Pareto non-dominated on (WR↑, J_A↑, SR_Δ↓, M_uni↓).\n\n"
            + _table(["Setting", "WR", "J_A", "SR_Δ", "Coalition M", "Unilateral M_uni [95% CI]",
                      "frac M_i>0", "PoS"], body))


def section_e3(rows) -> str:
    out = ["## E3 — Strategic population × attack type\n"]
    for m in MECHANISM_NAMES:
        out.append(f"\n**{MECHANISM_LABELS[m]}**\n")
        body = []
        for pol in ("cap_2", "max_claim"):
            for rho in sorted({r["rho"] for r in rows}):
                r = _pick(rows, mechanism=m, policy=pol, rho=rho)
                if r:
                    body.append([pol, f"{rho:g}", _f(r, "WR"), _f(r, "J_A", 3), _f(r, "PoS"),
                                 _f(r, "M_mean", 1, signed=True), _f(r, "M_uni", 2, ci=True, signed=True),
                                 _f(r, "M_uni_max", 1, signed=True), _f(r, "frac_pos_uni", 2)])
        out.append(_table(["Policy", "ρ", "WR", "J_A", "PoS", "Coalition M", "M_uni [95% CI]",
                           "max M_i", "frac M_i>0"], body))
    return "\n".join(out)


def section_e3b(rows) -> str:
    body = []
    for r in rows:
        body.append([r["label"], r["opponents"], _f(r, "M_rollout", 1, ci=True, signed=True),
                     _f(r, "M_cap2", 1, ci=True, signed=True), _f(r, "frac_inflated", 2),
                     _f(r, "frac_max_report", 2), _f(r, "mean_inflation", 3, signed=True)])
    return ("## E3b — Rollout attack (n=10, H=5, 100 rollouts, grid {0,…,1})\n\n"
            + _table(["Mechanism", "Opponents", "M_rollout [95% CI]", "M_cap2 [95% CI]",
                      "frac rounds inflated", "frac rounds at v_max", "mean (report − v)"], body))


def section_e4(rows) -> str:
    out = ["## E4 — Heterogeneous users\n"]
    for dist in ("uniform", "mixed"):
        out.append(f"\n**{dist}**\n")
        body = [[MECHANISM_LABELS[m], _f(r, "WR"), _f(r, "J_A"), _f(r, "J_B"), _f(r, "SR_delta", 3)]
                for m in MECHANISM_NAMES if (r := _pick(rows, mechanism=m, dist=dist))]
        out.append(_table(["Mechanism", "WR", "J_A", "J_B", "SR_Δ"], body))
    return "\n".join(out)


def section_e5(rows) -> str:
    out = ["## E5 — Temporal persistence\n"]
    for mode in ("proposal", "copula"):
        out.append(f"\n**AR(1) mode = {mode}**\n")
        body = []
        for m in MECHANISM_NAMES:
            for a in sorted({r["alpha"] for r in rows}):
                r = _pick(rows, mechanism=m, alpha=a, ar1_mode=mode) or \
                    (_pick(rows, mechanism=m, alpha=a, ar1_mode="proposal") if a == 0.0 else None)
                if r:
                    body.append([MECHANISM_LABELS[m], f"{a:g}", _f(r, "WR"), _f(r, "J_A", 4),
                                 _f(r, "Q_max", 1), _f(r, "pct95_wait", 1), _f(r, "SR_delta", 3)])
        out.append(_table(["Mechanism", "α", "WR", "J_A", "Q_max", "p95 wait", "SR_Δ"], body))
    return "\n".join(out)


def section_e6(rows) -> str:
    body = []
    for n in sorted({r["n"] for r in rows}):
        line = [str(n)]
        for m in MECHANISM_NAMES:
            r = _pick(rows, mechanism=m, n=n)
            line.append(f"{r['time_per_round_us']['mean']:.1f}" if r else "n/a")
        r0 = _pick(rows, mechanism=MECHANISM_NAMES[0], n=n)
        line.append(f"{r0['peak_memory_kib']['mean']:.0f}" if r0 else "n/a")
        body.append(line)
    return ("## E6 — Scalability (µs per round; measured peak heap in KiB)\n\n"
            + _table(["n"] + [MECHANISM_LABELS[m] for m in MECHANISM_NAMES] + ["peak heap (KiB)"], body))


def section_e7(rows) -> str:
    body = [[r["label"], f"{r['c']:g}", _f(r, "WR"), _f(r, "PoS"), _f(r, "M_uni", 2, ci=True, signed=True),
             _f(r, "frac_pos_uni", 2)] for r in rows]
    return ("## E7 — Sensitivity to the exaggeration factor c (ρ=0.25)\n\n"
            + _table(["Mechanism", "c", "WR", "PoS", "M_uni [95% CI]", "frac M_i>0"], body))


def section_e3c(rows) -> str:
    body = [[r["label"], str(r["H"]), _f(r, "M_rollout", 2, ci=True, signed=True),
             _f(r, "M_cap2", 2, signed=True), _f(r, "M_cap1.25", 2, signed=True),
             _f(r, "frac_max_report", 2), _f(r, "mean_inflation", 3, signed=True)] for r in rows]
    return ("## E3c — Rollout attack vs horizon (n=10, opponents truthful)\n\n"
            + _table(["Mechanism", "H", "M_rollout [95% CI]", "M_cap2", "M_cap1.25",
                      "frac rounds at v_max", "mean (report - v)"], body))


def section_e8(rows) -> str:
    out = ["## E8 — Timing attack (ρ=0.25); premium = M_uni(timed) - M_uni(always)\n"]
    for c in sorted({r["c"] for r in rows}):
        out.append(f"\n**c = {c:g}**\n")
        body = []
        for lab in dict.fromkeys(r["label"] for r in rows):
            base = _pick(rows, label=lab, c=c, variant="always")
            for v in ("always", "timed", "timed_q25", "timed_q75"):
                r = _pick(rows, label=lab, c=c, variant=v)
                if r and base:
                    prem = "—" if v == "always" else f"{r['M_uni']['mean'] - base['M_uni']['mean']:+.2f}"
                    body.append([lab, v, _f(r, "M_uni", 2, ci=True, signed=True), prem,
                                 _f(r, "frac_pos_uni", 2), _f(r, "PoS", 3)])
        out.append(_table(["Mechanism", "Policy", "M_uni [95% CI]", "Premium", "frac M_i>0", "PoS"], body))
    return "\n".join(out)


def section_e9(rows) -> str:
    out = ["## E9 — Proposed mechanisms under truthful reports\n"]
    for cond in dict.fromkeys(r["condition"] for r in rows if r["condition"] != "wait_frontier"):
        out.append(f"\n**{cond}**\n")
        body = [[r["label"], _f(r, "WR"), _f(r, "J_A", 4), _f(r, "J_B", 4), _f(r, "Q_max", 1),
                 _f(r, "pct95_wait", 1), _f(r, "SR_delta", 4)]
                for r in rows if r["condition"] == cond]
        out.append(_table(["Mechanism", "WR", "J_A", "J_B", "Q_max", "p95 wait", "SR_Δ"], body))
    fr = [r for r in rows if r["condition"] == "wait_frontier"]
    if fr:
        out.append("\n**Waiting-cap frontier (base condition)**\n")
        body = [[r["label"], str(r["wait_cap"]), _f(r, "WR"), _f(r, "Q_max", 1), _f(r, "SR_delta", 4),
                 _f(r, "J_A", 4)] for r in fr]
        out.append(_table(["Mechanism", "W", "WR", "Q_max", "SR_Δ", "J_A"], body))
    return "\n".join(out)


def section_e10(rows) -> str:
    out = ["## E10 — Proposed mechanisms under strategic reporting (unilateral gain M_uni)\n"]
    for rho in sorted({r["rho"] for r in rows}):
        out.append(f"\n**ρ = {rho:g}**\n")
        body = [[r["label"], r["policy"], _f(r, "M_uni", 2, ci=True, signed=True), _f(r, "M_uni_max", 1, signed=True),
                 _f(r, "frac_pos_uni", 2), _f(r, "WR"), _f(r, "PoS")]
                for r in rows if r["rho"] == rho]
        out.append(_table(["Mechanism", "Lie", "M_uni [95% CI]", "max M_i", "frac M_i>0", "WR", "PoS"], body))
    return "\n".join(out)


def section_e11(rows) -> str:
    body = [[r["label"], str(r["H"]), _f(r, "M_rollout", 2, ci=True, signed=True),
             _f(r, "M_cap2", 2, signed=True), _f(r, "M_cap1.25", 2, signed=True),
             _f(r, "frac_max_report", 2), _f(r, "mean_inflation", 3, signed=True)] for r in rows]
    return ("## E11 — Far-sighted attacker vs proposed mechanisms (n=10, T=300, opponents truthful)\n\n"
            + _table(["Mechanism", "H", "M_rollout [95% CI]", "M_cap2", "M_cap1.25",
                      "frac rounds at v_max", "mean (report - v)"], body))


SECTIONS = [("e1", section_e1), ("e2", section_e2), ("e3", section_e3), ("e3b", section_e3b), ("e3c", section_e3c),
            ("e4", section_e4), ("e5", section_e5), ("e6", section_e6), ("e7", section_e7), ("e8", section_e8), ("e9", section_e9), ("e10", section_e10), ("e11", section_e11)]


def build_report(results_dir: str = "results") -> str:
    rd = Path(results_dir)
    parts = ["# Experiment results\n\nGenerated by `python analysis/report.py` from "
             f"`{results_dir}/*/summary.json`. Values are seed means; brackets are 95% bootstrap CIs.\n"]
    for exp, fn in SECTIONS:
        rows = _load(rd, exp)
        parts.append(fn(rows) if rows else f"## {exp.upper()}\n\n_no summary found_")
    return "\n\n".join(parts) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default=None, help="output file (default: <results-dir>/RESULTS.md)")
    args = ap.parse_args()
    text = build_report(args.results_dir)
    out = Path(args.out) if args.out else Path(args.results_dir) / "RESULTS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out} ({len(text.splitlines())} lines)")
