"""
report/make_tables.py
=====================
Generate the LaTeX tables of the project report directly from the saved
results (results/*/summary.json), so every number in a table is copied from
the data and never typed by hand.

Usage (from the repository root):  python report/make_tables.py
Output: report/generated/*.tex
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "generated"
OUT.mkdir(exist_ok=True)

MECH = {
    "RandomMechanism": "M1 Random",
    "RoundRobinMechanism": "M2 Round-Robin",
    "GreedyMechanism": "M3 Greedy",
    "ScoreMechanism": "M4 Score",
    "VickreyMechanism": "M5 Vickrey",
}
MECH_ORDER = list(MECH)


def load(exp: str) -> list[dict]:
    with open(ROOT / "results" / exp / "summary.json") as fh:
        return json.load(fh)


def pick(rows: list[dict], **kw) -> dict:
    hits = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    if len(hits) != 1:
        raise KeyError(f"expected one row for {kw}, found {len(hits)}")
    return hits[0]


def num(v: float, d: int = 2, sign: bool = False) -> str:
    if v is None:
        return "n/a"
    s = f"{abs(v):.{d}f}"
    if float(s) == 0.0:
        return s if not sign else f"{s}"
    if v < 0:
        return f"$-${s}"
    return f"+{s}" if sign else s


def ci(m: dict, d: int = 2, sign: bool = False) -> str:
    return f"{num(m['mean'], d, sign)} [{num(m['ci_lower'], d)}, {num(m['ci_upper'], d)}]"


def lam(label: str) -> str:
    """Typeset a mechanism label that contains 'lambda=...' or the unicode lambda."""
    out = (label.replace(" λ=", " ($\\lambda=").replace(" lambda=", " ($\\lambda=")
           .replace(" β=", " ($\\beta=").replace("b0=", "($b_0="))
    if "$" in out and not out.endswith(")"):
        out += "$)"
    return out


def write(name: str, body: str, caption: str, label: str, size: str = "\\footnotesize",
          spec: str | None = None, star: bool = False) -> None:
    text = (f"\\begin{{table}}[!htb]\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\centering\n"
            f"{size}\n\\setlength{{\\tabcolsep}}{{4pt}}\n\\adjustbox{{max width=\\textwidth}}{{%\n{body}}}\n\\end{{table}}\n")
    (OUT / f"{name}.tex").write_text(text, encoding="utf-8")


def tabular(spec: str, header: list[str], rows: list[list[str]], midrules: set[int] | None = None) -> str:
    midrules = midrules or set()
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for i, r in enumerate(rows):
        if i in midrules:
            lines.append("\\midrule")
        lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- E1
def t_e1():
    d = load("e1")
    rows = []
    for m in MECH_ORDER:
        r = pick(d, mechanism=m, kn_ratio=0.2)
        rows.append([MECH[m], ci(r["WR"], 3), num(r["J_A"]["mean"], 4), num(r["NSW"]["mean"], 1),
                     num(r["Q_max"]["mean"], 1), num(r["pct95_wait"]["mean"], 1),
                     num(r["SR_delta"]["mean"], 4)])
    body = tabular("lrrrrrr", ["Mechanism", "WR [95\\% CI]", "$J_A$", "NSW", "$Q_{\\max}$",
                               "p95 wait", "$\\mathrm{SR}_\\Delta$"], rows)
    write("e1_table", body, "E1 at $k/n=0.2$ ($n=50$, $k=10$, truthful reports, 30 seeds). "
          "WR is the welfare ratio, NSW the Nash social welfare, $Q_{\\max}$ the mean maximum "
          "consecutive wait.", "tab:e1")


# ---------------------------------------------------------------- E2
def t_e2():
    d = load("e2")
    order = [("RandomMechanism", None), ("RoundRobinMechanism", None), ("GreedyMechanism", None),
             ("VickreyMechanism", None)] + [("ScoreMechanism", l) for l in
                                            (0.0, 0.25, 0.5, 1.0, 2.0, 5.0)]
    rows = []
    for m, l in order:
        r = pick(d, mechanism=m, lambda_=l)
        name = MECH[m] if l is None else f"M4 Score ($\\lambda={l:g}$)"
        mark = "$\\checkmark$" if r.get("pareto") else ""
        rows.append([name, num(r["WR"]["mean"], 3), num(r["J_A"]["mean"], 3),
                     num(r["SR_delta"]["mean"], 3), num(r["M_mean"]["mean"], 1, True),
                     ci(r["M_uni"], 1, True), num(r["frac_pos_uni"]["mean"], 2), mark])
    body = tabular("lrrrrlrc", ["Mechanism", "WR", "$J_A$", "$\\mathrm{SR}_\\Delta$",
                                "Coalition $M$", "Unilateral $M_{\\mathrm{uni}}$ [95\\% CI]",
                                "Frac. gaining", "Pareto"], rows, {4})
    write("e2_table", body, "E2: $\\lambda$ sweep with $\\rho=0.25$ of users using capped "
          "exaggeration with $c=2$. The last column marks settings that are non-dominated on "
          "(WR, $J_A$, $\\mathrm{SR}_\\Delta$, $M_{\\mathrm{uni}}$).", "tab:e2")


# ---------------------------------------------------------------- E7
def t_e7():
    d = load("e7")
    labels = ["M3", "M4 lambda=0.5", "M4 lambda=1", "M4 lambda=2", "M4 lambda=5", "M5"]
    names = {"M3": "M3 Greedy", "M5": "M5 Vickrey"}
    rows = []
    for lab in labels:
        cells = [names[lab] if lab in names else "M4 Score ($\\lambda=" + lab.split("=")[1] + "$)"]
        for c in (1.25, 1.5, 2.0):
            r = pick(d, label=lab, c=c)
            cells.append(ci(r["M_uni"], 1, True) + f" ({r['frac_pos_uni']['mean']:.2f})")
        rows.append(cells)
    body = tabular("lccc", ["Mechanism", "$c=1.25$", "$c=1.5$", "$c=2$"], rows, {1, 5})
    write("e7_table", body, "E7: unilateral gain $M_{\\mathrm{uni}}$ [95\\% CI] from capped exaggeration "
          "with factor $c$ ($\\rho=0.25$); the fraction of focal users who gain is in parentheses.",
          "tab:e7")


# ---------------------------------------------------------------- E3
def t_e3():
    d = load("e3")
    rows = []
    for m in ("GreedyMechanism", "ScoreMechanism", "VickreyMechanism"):
        for pol, pname in (("cap_2", "cap 2"), ("max_claim", "max claim")):
            cells = [MECH[m] if m != "ScoreMechanism" else "M4 Score ($\\lambda=1$)", pname]
            for rho in (0.0, 0.25, 1.0):
                r = pick(d, mechanism=m, policy=pol, rho=rho)
                cells.append(num(r["M_uni"]["mean"], 1, True))
            r = pick(d, mechanism=m, policy=pol, rho=0.25)
            cells += [num(r["WR"]["mean"], 3), num(r["PoS"]["mean"], 3)]
            rows.append(cells)
    body = tabular("llrrrrr", ["Mechanism", "Attack", "$M_{\\mathrm{uni}}$ $\\rho=0$",
                               "$\\rho=0.25$", "$\\rho=1$", "WR ($\\rho=0.25$)",
                               "PoS ($\\rho=0.25$)"], rows, {2, 4})
    write("e3_table", body, "E3: unilateral gain by strategic fraction $\\rho$ and the welfare "
          "ratio and price of strategy at $\\rho=0.25$. M1 and M2 have zero gain and zero PoS in "
          "every cell.", "tab:e3")


# ---------------------------------------------------------------- E3b / E3c
def t_e3c():
    d = load("e3c")
    rows = []
    for H in (5, 10, 20, 40, 80):
        cells = [str(H)]
        for lab in ("M4 lambda=1", "M4 lambda=2"):
            r = pick(d, label=lab, H=H)
            cells += [ci(r["M_rollout"], 2, True), num(r["frac_max_report"]["mean"], 2)]
        rows.append(cells)
    r1, r2 = pick(d, label="M4 lambda=1", H=5), pick(d, label="M4 lambda=2", H=5)
    rows.append(["capped $c=2$", num(r1["M_cap2"]["mean"], 2, True), "", num(r2["M_cap2"]["mean"], 2, True), ""])
    rows.append(["capped $c=1.25$", num(r1["M_cap1.25"]["mean"], 2, True), "", num(r2["M_cap1.25"]["mean"], 2, True), ""])
    body = tabular("lrrrr", ["Horizon $H$", "$\\lambda=1$ gain [95\\% CI]", "Share at $v_{\\max}$",
                             "$\\lambda=2$ gain [95\\% CI]", "Share at $v_{\\max}$"], rows, {5})
    write("e3c_table", body, "E3c (historical checkpoint): gain of one rollout attacker against M4 "
          "with truthful opponents ($n=10$, $T=300$) as the planning horizon grows. The last two "
          "rows give the capped-exaggeration gains on the same workloads.", "tab:e3c")


def t_e3b():
    d = load("e3b")
    rows = []
    names = {"M3": "M3 Greedy", "M4 lambda=1": "M4 Score ($\\lambda=1$)",
             "M4 lambda=2": "M4 Score ($\\lambda=2$)", "M5": "M5 Vickrey"}
    for lab in ("M3", "M4 lambda=1", "M4 lambda=2", "M5"):
        cells = [names[lab]]
        for opp in ("truthful", "cap_2"):
            r = pick(d, label=lab, opponents=opp)
            cells += [num(r["M_rollout"]["mean"], 1, True), num(r["M_cap2"]["mean"], 1, True)]
        cells.append(num(pick(d, label=lab, opponents="truthful")["frac_max_report"]["mean"], 2))
        rows.append(cells)
    body = tabular("lrrrrr", ["Mechanism", "Rollout (truthful opp.)", "Cap 2",
                              "Rollout (cap-2 opp.)", "Cap 2", "Share at $v_{\\max}$"], rows)
    write("e3b_table", body, "E3b (historical checkpoint): proposal-style rollout attack with $H=5$ "
          "($n=10$, $T=500$). Entries are the mean gain of the focal user.", "tab:e3b")


# ---------------------------------------------------------------- E4
def t_e4():
    d = load("e4")
    rows = []
    for m in MECH_ORDER:
        r = pick(d, mechanism=m, dist="mixed")
        rows.append([MECH[m] if m != "ScoreMechanism" else "M4 Score ($\\lambda=1$)",
                     num(r["WR"]["mean"], 3), num(r["J_A"]["mean"], 3), num(r["J_B"]["mean"], 3),
                     num(r["SR_delta"]["mean"], 3), num(r["NSW"]["mean"], 1)])
    body = tabular("lrrrrr", ["Mechanism", "WR", "$J_A$", "$J_B$", "$\\mathrm{SR}_\\Delta$", "NSW"], rows)
    write("e4_table", body, "E4: heterogeneous population (half $\\mathrm{Beta}(2,5)$, half "
          "$\\mathrm{Beta}(5,2)$), truthful reports.", "tab:e4")


# ---------------------------------------------------------------- E5
def t_e5():
    d = load("e5")
    rows = []
    for m in MECH_ORDER:
        cells = [MECH[m] if m != "ScoreMechanism" else "M4 Score ($\\lambda=1$)"]
        for mode in ("proposal", "copula"):
            r = pick(d, mechanism=m, alpha=0.9, ar1_mode=mode)
            cells += [num(r["WR"]["mean"], 3), num(r["J_A"]["mean"], 3),
                      num(r["Q_max"]["mean"], 1), num(r["SR_delta"]["mean"], 3)]
        rows.append(cells)
    body = tabular("lrrrrrrrr", ["Mechanism", "WR", "$J_A$", "$Q_{\\max}$", "$\\mathrm{SR}_\\Delta$",
                                 "WR", "$J_A$", "$Q_{\\max}$", "$\\mathrm{SR}_\\Delta$"], rows)
    body = body.replace("\\toprule\n", "\\toprule\n & \\multicolumn{4}{c}{Proposal recursion} & "
                        "\\multicolumn{4}{c}{Copula control} \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n", 1)
    write("e5_table", body, "E5 at $\\alpha=0.9$, truthful reports. The proposal recursion shrinks the "
          "value spread; the copula control preserves the marginal distribution.", "tab:e5")


# ---------------------------------------------------------------- E8
def t_e8():
    d = load("e8")
    rows = []
    for lab, ln in (("M4 lambda=1", "1"), ("M4 lambda=2", "2"), ("M4 lambda=5", "5")):
        for c in (1.25, 2.0):
            cells = [f"M4 Score ($\\lambda={ln}$)", f"{c:g}"]
            for v in ("always", "timed", "timed_q25", "timed_q75"):
                cells.append(num(pick(d, label=lab, c=c, variant=v)["M_uni"]["mean"], 1, True))
            rows.append(cells)
    body = tabular("llrrrr", ["Mechanism", "$c$", "Always", "Timed (mean)", "Timed (q25)",
                              "Timed (q75)"], rows, {2, 4})
    write("e8_table", body, "E8: unilateral gain of inflating always versus inflating only while "
          "the user's own allocation count is at most the population mean, 25th or 75th "
          "percentile ($\\rho=0.25$).", "tab:e8")


# ---------------------------------------------------------------- E9
def t_e9():
    d = load("e9")
    conds = [("base", "base"), ("scarce", "scarce ($k/n=0.1$)"), ("mixed", "mixed groups"),
             ("persist_copula", "persistent (copula, $\\alpha=0.9$)")]
    labs = [("M3 Greedy", "M3 Greedy"), ("M4 Score λ=1", "M4 Score ($\\lambda=1$)"),
            ("M6 Karma-Cap", "M6 Karma-Cap"), ("M7 Rank-Cap", "M7 Rank-Cap")]
    rows = []
    for ck, cn in conds:
        for lk, ln in labs:
            r = pick(d, condition=ck, label=lk)
            rows.append([cn if lk == "M3 Greedy" else "", ln, num(r["WR"]["mean"], 3),
                         num(r["J_B"]["mean"], 3), num(r["Q_max"]["mean"], 1),
                         num(r["SR_delta"]["mean"], 3)])
    body = tabular("llrrrr", ["Condition", "Mechanism", "WR", "$J_B$", "$Q_{\\max}$",
                              "$\\mathrm{SR}_\\Delta$"], rows, {4, 8, 12})
    write("e9_table", body, "E9: truthful reports on the locked seeds. M6 and M7 use the wait cap "
          "$W=10$ (the default $W=\\Delta$).", "tab:e9")


def t_e9w():
    d = load("e9")
    rows = []
    for lab, name in (("M6 Karma-Cap", "M6 Karma-Cap"), ("M7 Rank-Cap", "M7 Rank-Cap")):
        cells = [name]
        for W in (6, 10, 15, 20, 30):
            r = pick(d, condition="wait_frontier", label=lab, wait_cap=W)
            cells.append(num(r["WR"]["mean"], 3))
        rows.append(cells)
    body = tabular("lccccc", ["Mechanism", "$W=6$", "$W=10$", "$W=15$", "$W=20$", "$W=30$"], rows)
    write("e9w_table", body, "E9: truthful welfare ratio as the wait cap $W$ varies "
          "($n=50$, $k=10$).", "tab:e9w")


# ---------------------------------------------------------------- E10
def t_e10():
    d = load("e10")
    labs = [("M3 Greedy", "M3 Greedy"), ("M4 Score λ=1", "M4 ($\\lambda=1$)"),
            ("M4 Score λ=2", "M4 ($\\lambda=2$)"), ("M5 Vickrey", "M5 Vickrey"),
            ("M6 Karma-Cap", "M6 Karma-Cap"), ("M7 Rank-Cap", "M7 Rank-Cap")]
    cols = [("cap_1.25", 0.25, "$c=1.25$"), ("cap_1.5", 0.25, "$c=1.5$"), ("cap_2", 0.25, "$c=2$"),
            ("max_claim", 0.25, "max"), ("timed_cap_1.25", 0.25, "timed 1.25"),
            ("cap_2", 1.0, "$\\rho{=}1$, $c{=}2$"), ("max_claim", 1.0, "$\\rho{=}1$, max")]
    rows = []
    for lk, ln in labs:
        cells = [ln]
        for pol, rho, _ in cols:
            r = pick(d, label=lk, policy=pol, rho=rho)
            cells.append(num(r["M_uni"]["mean"], 1, True))
        rows.append(cells)
    body = tabular("l" + "r" * len(cols), ["Mechanism"] + [c[2] for c in cols], rows, {1, 4})
    write("e10_table", body, "E10: unilateral gain $M_{\\mathrm{uni}}$ from each scripted lie at "
          "$\\rho=0.25$ (last two columns: $\\rho=1$).", "tab:e10")


def t_e11():
    d = load("e11")
    d3 = load("e3c")
    rows = []
    for H in (5, 40, 80):
        cells = [str(H)]
        for lab in ("M6 Karma-Cap", "M7 Rank-Cap"):
            cells.append(num(pick(d, label=lab, H=H)["M_rollout"]["mean"], 2, True))
        for lab in ("M4 lambda=1", "M4 lambda=2"):
            cells.append(num(pick(d3, label=lab, H=H)["M_rollout"]["mean"], 2, True))
        rows.append(cells)
    body = tabular("lrrrr", ["Horizon $H$", "M6 Karma-Cap", "M7 Rank-Cap",
                             "M4 ($\\lambda=1$)", "M4 ($\\lambda=2$)"], rows)
    write("e11_table", body, "E11 (historical checkpoint): mean gain of the rollout attacker "
          "($n=10$, $T=300$, truthful opponents). Score columns repeat E3c.", "tab:e11")


# ---------------------------------------------------------------- E12
def t_e12():
    d = load("e12")
    fam = [("Score", "Score"), ("Score-Cap W=10", "Score-Cap"), ("Karma", "Karma"),
           ("Karma-Cap W=10", "Karma-Cap"), ("Rank", "Rank"), ("Rank-Cap W=10", "Rank-Cap")]
    rows = []
    for lab, name in fam:
        t = pick(d, condition="base", label=lab, policy="truthful", rho=0.0)
        a = pick(d, condition="base", label=lab, policy="cap_1.25", rho=0.25)
        b = pick(d, condition="base", label=lab, policy="cap_2", rho=1.0)
        rows.append([name, num(t["WR"]["mean"], 3), num(t["Q_max"]["mean"], 1),
                     ci(a["M_uni"], 1, True), num(a["frac_pos_uni"]["mean"], 2),
                     ci(b["M_uni"], 1, True), num(b["frac_pos_uni"]["mean"], 2)])
    body = tabular("lrrlrlr", ["Rule", "WR (truthful)", "$Q_{\\max}$", "Gain, $c=1.25$, $\\rho=0.25$",
                               "Frac.", "Gain, $c=2$, $\\rho=1$", "Frac."], rows, {2, 4})
    write("e12_table", body, "E12 (fresh held-out seeds): cap and incentive components on the "
          "homogeneous workload. Capped rules use $W=10$. ``Frac.'' is the fraction of focal "
          "users with a positive gain.", "tab:e12")


def t_e12c():
    d = load("e12")
    pairs = [("Score-Cap W=10", "Score"), ("Karma-Cap W=10", "Karma"), ("Rank-Cap W=10", "Rank")]
    rows = []
    for cap, base in pairs:
        a = pick(d, condition="base", label=cap, policy="cap_1.25", rho=0.25)
        b = pick(d, condition="base", label=base, policy="cap_1.25", rho=0.25)
        rows.append([f"{cap.split(' ')[0]} minus {base}", num(a["WR"]["mean"] - b["WR"]["mean"], 3, True),
                     num(a["M_uni"]["mean"] - b["M_uni"]["mean"], 2, True)])
    for fam in ("Karma-Cap W=10", "Rank-Cap W=10"):
        a = pick(d, condition="base", label=fam, policy="cap_1.25", rho=0.25)
        b = pick(d, condition="base", label="Score-Cap W=10", policy="cap_1.25", rho=0.25)
        rows.append([f"{fam.split(' ')[0]} minus Score-Cap", num(a["WR"]["mean"] - b["WR"]["mean"], 3, True),
                     num(a["M_uni"]["mean"] - b["M_uni"]["mean"], 2, True)])
    body = tabular("lrr", ["Comparison ($c=1.25$, $\\rho=0.25$)", "WR difference", "Gain difference"], rows, {3})
    write("e12c_table", body, "E12: differences of means. The first three rows isolate adding the cap; "
          "the last two compare incentive components at the same wait cap. Paired intervals "
          "are in the saved \\texttt{paired.json}.", "tab:e12c")


# ---------------------------------------------------------------- E13 / E18
def _regret(m: dict) -> str:
    return f"{m['mean']:.2f} [{m['ci_lower']:.2f}, {m['ci_upper']:.2f}]"


def t_e13():
    d = load("e13")
    rows = []
    for init in ("truthful", "cap_2"):
        for lab in ("Greedy", "Score", "Score-Cap", "Vickrey", "Karma-Cap", "Rank-Cap"):
            r = pick(d, label=lab, initial=init)
            rows.append([init.replace("_", " ") if lab == "Greedy" else "", lab,
                         r["training_status"].replace("_", " "), str(r["training_sweeps"]),
                         num(r["WR"]["mean"], 3), num(r["mean_utility"]["mean"], 2),
                         _regret(r["restricted_exploitability"]), num(r["truthful_fraction"]["mean"], 2)])
    body = tabular("lllcrrrr", ["Start", "Rule", "Training", "Sweeps", "WR", "Mean utility",
                                "Regret [band]", "Truthful"], rows, {6})
    write("e13_table", body, "E13: adaptive finite-policy populations ($n=6$, $k=2$, $T=120$). "
          "Regret is the largest seed-averaged gain from any single policy switch, with a "
          "simultaneous bootstrap band.", "tab:e13")


def t_e18():
    d = load("e18")
    rows = []
    for lab in ("Score-Cap", "Karma-Cap", "Rank-Cap"):
        for init in ("truthful", "cap_2"):
            for b in (4, 20):
                r = pick(d, label=lab, initial=init, budget=b)
                rows.append([lab if (init == "truthful" and b == 4) else "",
                             init.replace("_", " ") if b == 4 else "", str(b),
                             r["training_status"].replace("_", " "), str(r["training_sweeps"]),
                             f"{r['training_exploitability']:.3f}", _regret(r["restricted_exploitability"])])
    body = tabular("lllllrl", ["Rule", "Start", "Budget", "Status", "Sweeps used", "Training regret",
                               "Independent regret [band]"], rows, {4, 8})
    write("e18_table", body, "E18: four versus twenty training sweeps, evaluated on 30 independent "
          "seeds. Greedy, Score and Vickrey reach a fixed point within four sweeps in every start "
          "and are omitted.", "tab:e18")


# ---------------------------------------------------------------- E14
def t_e14():
    d = load("e14")
    rows = []
    for cond, cn in (("base", "base"), ("mixed", "mixed"), ("persistent", "persistent")):
        for lab in ("Score-Cap", "Karma-Cap", "Rank-Cap"):
            r = pick(d, label=lab, condition=cond, wait_cap=6)
            rows.append([cn if lab == "Score-Cap" else "", lab, str(r["certified_bound"]),
                         ci(r["constrained_efficiency"], 3), num(r["unavoidable_loss"]["mean"], 3),
                         num(r["algorithm_loss"]["mean"], 3), num(r["oracle_optimal"]["mean"], 2)])
    body = tabular("llrlrrr", ["Demand", "Rule", "$B$", "Efficiency [95\\% CI]", "Service loss",
                               "Remaining loss", "Optimal"], rows, {3, 6})
    write("e14_table", body, "E14 at $W=6$ ($n=6$, $k=2$, $T=40$): welfare of each online rule as a "
          "fraction of the constrained offline optimum. ``Optimal'' is the fraction of oracle "
          "solves certified optimal.", "tab:e14")


# ---------------------------------------------------------------- E15 / E16 / E17
def t_e15():
    d = load("e15")
    rows = []
    for lab, mode in (("Greedy", "exact"), ("Score", "exact"), ("Score lambda=2", "exact"),
                      ("Score-Cap", "exact"), ("Vickrey", "exact"), ("Karma-Cap", "exact"),
                      ("Rank-Cap", "updating"), ("Rank-Cap", "frozen")):
        r = pick(d, label=lab, mode=mode, H=40, n_rollouts=80)
        name = lab.replace("Score lambda=2", "Score ($\\lambda=2$)")
        rows.append([name, mode, ci(r["M_rollout"], 2, True), num(r["frac_truthful_report"]["mean"], 2)])
    body = tabular("llrr", ["Rule", "History model", "Focal gain [95\\% CI]", "Truthful share"], rows, {6})
    write("e15_table", body, "E15 at the largest budget ($H=40$, 80 Monte Carlo futures per candidate; "
          "$n=10$, $k=2$, $T=120$): gain of the planning attacker.", "tab:e15")


def t_e16():
    d = load("e16")
    rows = []
    for mode in ("updating", "frozen"):
        r = pick(d, mode=mode)
        rows.append([mode, ci(r["M_rollout"], 2, True), num(r["forced_gain"]["mean"], 2, True),
                     num(r["auction_gain"]["mean"], 2, True), ci(r["clean_reference_gain"], 2, True),
                     ci(r["ledger_effect"], 2, True)])
    body = tabular("lrrrll", ["Attacker", "Actual gain [CI]", "Forced", "Auction",
                              "Truthful-ref. replay [CI]", "Actual minus replay [CI]"], rows)
    write("e16_table", body, "E16: Rank-Cap attacker decomposition ($n=10$, $k=2$, $T=120$, $H=40$, "
          "80 futures).", "tab:e16")


def t_e17():
    d = load("e17")
    rows = []
    for lab in ("Karma", "Karma-Cap"):
        for rho in (0.25, 1.0):
            r = pick(d, label=lab, rho=rho)
            rows.append([lab if rho == 0.25 else "", f"{rho:g}", ci(r["M_uni"], 2, True),
                         num(r["forced_gain"]["mean"], 2, True), num(r["auction_gain"]["mean"], 2, True),
                         num(r["wins_change"]["mean"], 2, True), num(r["credit_spent_change"]["mean"], 2, True)])
    body = tabular("llrrrrr", ["Rule", "$\\rho$", "Total gain [CI]", "Forced", "Auction",
                               "$\\Delta$ services", "$\\Delta$ credits spent"], rows, {2})
    write("e17_table", body, "E17: decomposition of the cap-2 deviation gain of a focal user "
          "($n=50$, $k=10$, $T=1000$).", "tab:e17")


def main():
    for fn in (t_e1, t_e2, t_e7, t_e3, t_e3c, t_e3b, t_e4, t_e5, t_e8, t_e9, t_e9w, t_e10, t_e11,
               t_e12, t_e12c, t_e13, t_e18, t_e14, t_e15, t_e16, t_e17):
        fn()
    print("wrote", len(list(OUT.glob("*.tex"))), "tables to", OUT)


if __name__ == "__main__":
    main()
