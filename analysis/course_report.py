"""Tables for the fresh-seed course extension; also included by report.py."""
from analysis.report import _f, _table


def section_e12(rows):
    out = ["## E12 — Cap and incentive-component ablations (fresh heldout seeds)\n"]
    for condition, policy, rho in dict.fromkeys((r["condition"], r["policy"], r["rho"]) for r in rows):
        out.append(f"\n**{condition}; {policy}; rho={rho:g}**\n")
        selected = [r for r in rows if (r["condition"], r["policy"], r["rho"]) == (condition, policy, rho)]
        body = [[r["label"], _f(r, "WR", 3, ci=True), _f(r, "J_B", 3), _f(r, "Q_max", 1),
                 _f(r, "SR_delta", 3), _f(r, "M_uni", 2, ci=True, signed=True), _f(r, "frac_pos_uni", 2)] for r in selected]
        out.append(_table(["Mechanism", "WR [95% CI]", "J_B", "Q_max", "SR", "Individual gain [95% CI]", "frac gain>0"], body))
    return "\n".join(out)


def section_e13(rows):
    out = ["## E13 — Adaptive finite-policy populations (fresh heldout seeds)\n",
           "Policies are trained on independent seeds. Regret maximises expected paired gain after averaging; "
           "its interval comes from a simultaneous bootstrap band over all user/policy switches.\n"]
    for initial in dict.fromkeys(r["initial"] for r in rows):
        out.append(f"\n**Initial policy: {initial}**\n")
        selected = [r for r in rows if r["initial"] == initial]
        body = [[r["label"], r["training_status"], str(r["training_sweeps"]),
                 _f(r, "WR", 3, ci=True), _f(r, "mean_utility", 2),
                 _f(r, "restricted_exploitability", 2, ci=True), _f(r, "truthful_fraction", 2)] for r in selected]
        out.append(_table(["Mechanism", "Training status", "Sweeps", "WR [95% CI]", "Mean utility", "Restricted regret [band]", "Truthful fraction"], body))
        out.append("")
        for r in selected:
            out.append(f"- {r['label']}: `{', '.join(r['profile'])}`")
    return "\n".join(out)


def section_e14(rows):
    out = ["## E14 — Welfare relative to a constrained offline oracle\n",
           "B=W+ceil(n/k)-1 is the shared certified wait bound, including terminal wait. "
           "Efficiency is online welfare divided by the solver's welfare upper bound. "
           "The loss decomposition is exact when the solver certifies optimality.\n"]
    for condition in dict.fromkeys(r["condition"] for r in rows):
        out.append(f"\n**{condition}**\n")
        body = [[r["label"], str(r["wait_cap"]), str(r["certified_bound"]), _f(r, "WR", 3),
                 _f(r, "constrained_efficiency", 3, ci=True), _f(r, "unavoidable_loss", 3),
                 _f(r, "algorithm_loss", 3), _f(r, "oracle_optimal", 2), _f(r, "oracle_mip_gap", 4)]
                for r in rows if r["condition"] == condition]
        out.append(_table(["Mechanism", "W", "B", "WR", "Constrained efficiency [CI]", "Service loss", "Algorithm loss", "Optimal fraction", "MIP gap"], body))
    return "\n".join(out)


def section_e15(rows):
    body = [[r["label"], r["mode"], str(r["H"]), str(r["n_rollouts"]),
             _f(r, "M_rollout", 2, ci=True, signed=True), _f(r, "frac_truthful_report", 2),
             _f(r, "frac_max_report", 2)] for r in rows]
    return ("## E15 — Corrected planning attack and computation budgets\n\n"
            "Exact truth is a candidate; horizons stop at episode end; rank histories update except in explicit frozen controls.\n\n"
            + _table(["Mechanism", "State model", "H", "MC count", "Gain [95% CI]", "Truthful fraction", "Max-report fraction"], body))


def section_e16(rows):
    return ("## E16 — Rank history diagnostics (existing workloads)\n\n"
            "Truthful-reference replay uses privileged information on fixed reports; it is a diagnostic intervention.\n\n"
            + _table(["Attacker", "Actual gain [CI]", "Replay gain [CI]", "Actual minus replay [CI]", "Underreport fraction", "Overreport fraction"],
                     [[r["mode"], _f(r,"M_rollout",2,ci=True,signed=True), _f(r,"clean_reference_gain",2,ci=True,signed=True),
                       _f(r,"ledger_effect",2,ci=True,signed=True), _f(r,"underreport_fraction",3), _f(r,"overreport_fraction",3)] for r in rows]))


def section_e17(rows):
    return ("## E17 — Karma population reversal and service accounting\n\n"
            "Gain decomposes exactly into changes in forced and auction benefits. Forced users pay internal credits.\n\n"
            + _table(["Rule", "rho", "Total gain [CI]", "Forced gain [CI]", "Auction gain [CI]", "Change in wins", "Change in credit spending"],
                     [[r["label"],str(r["rho"]),_f(r,"M_uni",2,ci=True,signed=True),_f(r,"forced_gain",2,ci=True,signed=True),
                       _f(r,"auction_gain",2,ci=True,signed=True),_f(r,"wins_change",2,signed=True),_f(r,"credit_spent_change",2,signed=True)] for r in rows]))


def section_e18(rows):
    return ("## E18 — Extended adaptive training (independent new evaluation seeds)\n\n"
            "Recurrences are recorded without stopping longer training. Regret bands cover all inspected switches.\n\n"
            + _table(["Rule", "Initial policy", "Budget", "Status", "Sweeps", "Training regret", "Independent regret [band]", "WR [CI]"],
                     [[r["label"],r["initial"],str(r["budget"]),r["training_status"],str(r["training_sweeps"]),
                       f"{r['training_exploitability']:.3f}",_f(r,"restricted_exploitability",3,ci=True),_f(r,"WR",3,ci=True)] for r in rows]))


SECTIONS = [("e12", section_e12), ("e13", section_e13), ("e14", section_e14), ("e15", section_e15),
            ("e16",section_e16),("e17",section_e17),("e18",section_e18)]
