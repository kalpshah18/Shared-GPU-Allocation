"""Build the course report from the fixed protocol and saved E12-E15 results."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.report import _f, _table
from analysis.course_report import section_e13


def _difference(metrics, key, digits=2):
    value = metrics[key]
    return _f({key: {**value, "mean": value["mean_diff"]}}, key, digits, ci=True, signed=True)


def build_manuscript(results_dir="results", followup_report="FAILURE_CASE_STUDIES.md"):
    rd = Path(results_dir)
    data, configs = {}, {}
    for exp in ("e12", "e13", "e14", "e15"):
        data[exp] = json.loads((rd / exp / "summary.json").read_text())
        configs[exp] = json.loads((rd / exp / "config.json").read_text())
    counts = {cfg["n_seeds"] for cfg in configs.values()}
    seed_note = ("All four studies use 30 fresh heldout evaluation seeds."
                 if counts == {30} else f"Development output: evaluation seed counts are {sorted(counts)}; these are not the full study.")
    e12 = data["e12"]
    e15 = data["e15"]
    uncapped_score = next(r for r in e12 if r["family"] == "Score" and not r["cap_enabled"]
                          and r["condition"] == "base" and r["policy"] == "cap_1.25" and r["rho"] == 0.25)
    main_W = min(r["wait_cap"] for r in e12 if r["cap_enabled"])
    cap_score = next(r for r in e12 if r["family"] == "Score" and r["wait_cap"] == main_W
                     and r["condition"] == "base" and r["policy"] == "cap_1.25" and r["rho"] == 0.25)
    strongest = max((r["H"], r["n_rollouts"]) for r in e15)
    rank = next(r for r in e15 if r["label"] == "Rank-Cap" and r["mode"] == "updating"
                and (r["H"], r["n_rollouts"]) == strongest)
    rank_gain = rank["M_rollout"]
    if rank_gain["ci_lower"] > 0:
        rank_statement = "The corrected planning attacker earns a positive mean gain against Rank-Cap, with a 95% interval above zero."
    elif rank_gain["mean"] > 0:
        rank_statement = "The corrected planning attacker has positive mean gain against Rank-Cap, but its 95% interval includes zero."
    else:
        rank_statement = "The corrected planning attacker has non-positive mean gain against Rank-Cap at the largest tested budget."
    parts = ["# Fairness, service guarantees, and strategic behaviour in shared GPU allocation\n",
             "Aayush Kuloor, Harsh Dhru, Raj Modi, Kalp Shah — CS 329 course project\n",
             "## Abstract\n",
             "We study repeated allocation of identical GPUs to unit-demand users with private per-round values. "
             "The course extension separates the service override from two non-monetary incentive components, "
             "evaluates policy choice by adaptive users, strengthens the planning attacker, and compares online welfare "
             "with an offline optimum subject to the same service requirement. " + seed_note + " "
             f"A wait cap alone leaves mild exaggeration profitable for Score: its mean individual gain changes from "
             f"{_f(uncapped_score, 'M_uni', 2, signed=True)} to {_f(cap_score, 'M_uni', 2, signed=True)} "
             f"at W={main_W}. " + rank_statement + " "
             "The results distinguish service guarantees, ordinal invariance, and empirical incentive robustness; "
             "none implies unrestricted dynamic strategyproofness.\n",
             "## 1. Research question and model\n",
             "**How much welfare must online allocation sacrifice to guarantee service while limiting profitable strategic reporting?** "
             "We ask which gains are caused by the cap itself, whether agents select the scripted attacks when utility matters, "
             "and whether stronger planning changes the robustness conclusion.\n\n"
             "In each round t, n users privately observe values v_i,t in [0,1], submit bounded reports, and compete for k<n GPUs. "
             "Exactly k users receive one round of service. Utility is the sum of true allocation benefits minus payments. "
             "Social welfare excludes payments, and WR divides true welfare by the sum of the top-k true values in each round. "
             "Allocation history and the valuation distribution are available to the simulated policy/attacker. The planning "
             "attacker samples opponents' current values rather than observing them and samples iid future values.\n\n"
             "Waiting q_i(t) counts unserved rounds before round t. A threshold W gives eligible users oldest-first priority. "
             "The certified bound is B=W+ceil(n/k)-1, including the terminal state. Starvation uses the separate threshold "
             "Delta=2*ceil(n/k). The count and normalised-benefit Jain indices are reported alongside waits and welfare.\n",
             "## 2. Mechanisms and formal properties\n",
             "Greedy selects top reports; Score selects report/(1+prior allocations)^lambda; Vickrey uses top reports "
             "and charges each winner the next-highest bid. Score-Cap adds the shared waiting override. Karma-Cap "
             "caps bids by internal balances, charges uniform-price karma and redistributes it; karma is not a monetary "
             "utility deduction. Rank-Cap substitutes each user's empirical report quantile for raw report value before "
             "applying the history penalty and service override.\n\n"
             "[THEORY.md](THEORY.md) proves two properties. First, the oldest-first override guarantees "
             "q_i(t)<=W+ceil(n/k)-1 for every report sequence, starting with zero waits. Second, with pure rank and "
             "fixed tie-breaking, a fixed strictly increasing transform of each user's entire report sequence preserves "
             "all allocations. This invariance covers neither clipping nor selective/time-varying distortions.\n\n"
             "The oracle uses binary x_i,t, capacity equality and at least one service in every B+1-round window per user. "
             "With zero initial waits these constraints are equivalent to q_i(t)<=B, including q_i(T). "
             "For a feasible online schedule, W_online<=W*_B<=W*, so total welfare loss decomposes into "
             "the unavoidable service cost (W*-W*_B)/W* and the remaining loss (W*_B-W_online)/W*. "
             "The latter includes the disadvantage of not knowing future values.\n",
             "## 3. Experimental design\n",
             seed_note + " The holdout is disjoint from the earlier 30 seeds and both development seed lists. "
             "Mechanism parameters and comparisons were fixed in [EXTENSION_PROTOCOL.md](EXTENSION_PROTOCOL.md) "
             "before examining the new outcomes. This is a local design record, not a claim of external preregistration. "
             "The previous results serve as development evidence. No new mechanism tuning uses the fresh holdout.\n"]
    table = []
    for exp, question in [("e12", "Cap/component ablations"), ("e13", "Adaptive policy populations"),
                          ("e14", "Constrained offline welfare"), ("e15", "Planning and computation budgets")]:
        cfg = configs[exp]
        table.append([exp.upper(), question, str(cfg["n"]), str(cfg.get("k", round(cfg["n"] * .2))), str(cfg["T"]), str(cfg["n_seeds"])])
    parts.append(_table(["Study", "Question", "Users", "GPUs", "Rounds", "Seeds"], table))
    parts.append("\nEach comparison uses paired valuation and tie-breaking streams. Ordinary intervals are 95% "
                 "percentile bootstrap intervals with 10,000 resamples in the full study. Raw data, configs, seed roles, "
                 "paired differences and source digests are saved.\n\n"
                 "E13 trains episode-long policy choices on five separate workloads using asynchronous best responses, "
                 "starting from all-truthful and all-cap_2 populations. Its library contains truthful reporting, three "
                 "exaggeration levels, maximum claims, two timed policies, and a high-value-only policy. "
                 "Users optimise expected episode utility under fixed opponent policies; their final profiles are frozen "
                 "before evaluation. The audit exhausts every unilateral policy switch and maximises the expected "
                 "gain AFTER averaging across heldout seeds. Simultaneous bootstrap bands cover all inspected switches. "
                 "Convergence refers to the training policy game. A cycle status records a repeated sweep-boundary profile, "
                 "not proof of a periodic orbit under the shuffled update orders.\n\n"
                 "E15 adds exact truth to the report grid, truncates horizons to the remaining episode and advances "
                 "mechanism/opponent state in hypothetical futures. Rank histories update in the main evaluation; "
                 "matched frozen controls isolate the former approximation. The continuation policy is still truthful "
                 "after the first hypothetical decision, so this remains a limited planning heuristic.\n")
    parts.append("## 4. Results\n\n### 4.1 What the cap and incentive components contribute\n")
    for policy, rho in [("truthful", 0.0), ("cap_1.25", 0.25), ("cap_2", 1.0)]:
        selected = [r for r in e12 if r["condition"] == "base" and r["policy"] == policy
                    and r["rho"] == rho and (not r["cap_enabled"] or r["wait_cap"] == main_W)]
        parts.append(f"**{policy}, rho={rho:g}; capped rows use W={main_W}.**\n")
        parts.append(_table(["Mechanism", "WR [CI]", "Mean maximum wait", "Individual gain [CI]", "Fraction gaining"],
                            [[r["label"], _f(r, "WR", 3, ci=True), _f(r, "Q_max", 1),
                              _f(r, "M_uni", 2, ci=True, signed=True), _f(r, "frac_pos_uni", 2)] for r in selected]))
    parts.append("A reduction in average gain is distinct from eliminating gains for every user. "
                 "The within-family paired comparisons isolate adding the cap; matched-W comparisons against Score-Cap "
                 "isolate the incentive component. Truthful-row gains are zero by construction because both sides "
                 "use truthful reporting; they are not a deviation audit. Full sensitivity tables are in "
                 "[RESULTS.md](../results/RESULTS.md).\n")
    cap_pairs = rd / "e12" / "paired.json"
    if cap_pairs.exists():
        selected = [r for r in json.loads(cap_pairs.read_text()) if r["condition"] == "base"
                    and r["policy"] == "cap_1.25" and r["rho"] == .25
                    and r["label"].endswith(f"W={main_W}")]
        parts.append(f"**Paired contrasts for mild exaggeration at W={main_W}; candidate minus reference.**\n")
        parts.append(_table(["Candidate", "Reference", "WR difference [CI]", "Gain difference [CI]"],
                            [[r["label"], r["reference"], _difference(r["metrics"], "WR", 3),
                              _difference(r["metrics"], "M_uni")] for r in selected]))
    parts.append("### 4.2 What utility-seeking agents choose\n")
    parts.append(section_e13(data["e13"]).replace("## E13 — Adaptive finite-policy populations (fresh heldout seeds)\n", ""))
    parts.append("Training convergence need not generalise exactly to the holdout. A sweep-limit profile may retain "
                 "profitable switches even on the training set. These are restricted-policy results, not claims of "
                 "unrestricted equilibrium, and the two starts do not exhaust basins of attraction.\n")
    if all(r["restricted_exploitability"]["ci_lower"] == 0 for r in data["e13"]):
        parts.append("Every heldout regret band includes zero. Positive point estimates therefore do not establish "
                     "statistically positive expected regret after accounting for all inspected switches. "
                     "A training sweep limit and residual training incentives still prevent an equilibrium claim.\n")
    parts.append("### 4.3 Welfare after accounting for the service requirement\n")
    oracle_W = sorted({r["wait_cap"] for r in data["e14"]})[len({r["wait_cap"] for r in data["e14"]}) // 2]
    selected = [r for r in data["e14"] if r["wait_cap"] == oracle_W]
    parts.append(f"The middle threshold W={oracle_W} is shown below. Optimal fraction reports solver certification, not an assumption.\n")
    parts.append(_table(["Demand", "Mechanism", "B", "Efficiency [CI]", "Service loss", "Remaining loss", "Optimal fraction"],
                        [[r["condition"], r["label"], str(r["certified_bound"]), _f(r, "constrained_efficiency", 3, ci=True),
                          _f(r, "unavoidable_loss", 3), _f(r, "algorithm_loss", 3), _f(r, "oracle_optimal", 2)] for r in selected]))
    parts.append("When all solves are optimal, the decomposition is exact. Otherwise efficiency is bounded using "
                 "the solver's primal and dual objectives and the loss columns using the dual bound are not exact "
                 "unavoidable/algorithm costs. The oracle experiments are deliberately small and have future information.\n")
    parts.append("### 4.4 Stronger planning and rank-history manipulation\n")
    selected = [r for r in e15 if (r["H"], r["n_rollouts"]) == strongest]
    parts.append(f"Largest computation budget: H={strongest[0]}, {strongest[1]} Monte Carlo futures per candidate.\n")
    parts.append(_table(["Mechanism", "History model", "Focal gain [CI]", "Truthful fraction"],
                        [[r["label"], r["mode"], _f(r, "M_rollout", 2, ci=True, signed=True),
                          _f(r, "frac_truthful_report", 2)] for r in selected]))
    parts.append(rank_statement + " Paired updating-minus-frozen differences measure the approximation's effect. "
                 "A mean gain with an interval above zero supports a profitable tested deviation in this workload model; "
                 "a negative gain only means that this particular attacker failed. Budget sensitivity makes that limitation visible.\n")
    paired_path = rd / "e15" / "paired.json"
    if paired_path.exists():
        comparisons = json.loads(paired_path.read_text())
        effect = next((r for r in comparisons if r["comparison"] == "rank_history_effect"
                       and (r["H"], r["n_rollouts"]) == strongest), None)
        if effect:
            parts.append("At the largest budget, updating-minus-frozen Rank-Cap gain is "
                         + _difference(effect["metrics"], "M_rollout")
                         + ". This is a paired comparison on the same workloads and attacker random streams.\n")
    if all((rd / exp / "summary.json").exists() for exp in ("e16", "e17", "e18")):
        parts.append("### 4.5 Failure-mode follow-ups\n\n"
                     f"The [failure case studies]({followup_report}) trace reported and truthful rank references, "
                     "replay identical attack sequences with a truthful reference, and decompose Karma gains into "
                     "forced/auction service with verified credit accounting. The Rank and Karma diagnostics reuse "
                     "existing workloads for explanation. A four-vs-twenty-sweep adaptive comparison uses a separate "
                     "new evaluation set; it records recurrence without treating a repeated profile as convergence. "
                     "These analyses qualify causal explanations of the failures and the original four-sweep findings.\n")
    parts.append("## 5. Related work and limitations\n\n"
                 "Published [Karma (OSDI 2023)](https://www.usenix.org/conference/osdi23/presentation/vuppalapati) "
                 "credits donated unused capacity that others borrow in a dynamic-demand model. Our uniform-price bid "
                 "auction with redistribution is a different mechanism; its guarantees do not transfer. "
                 "[Fikioris, Agarwal and Tardos](https://arxiv.org/abs/2109.12401) study incentives for dynamic MMF/DRF "
                 "and show that history-aware allocation can lose incentive compatibility. Their demand/resource model "
                 "also differs from this private-value unit-demand model. Our contribution is a controlled component "
                 "study and finite-policy strategic evaluation, together with explicit service and ordinal-invariance proofs.\n\n"
                 "The model omits arrivals, job duration, placement, multi-GPU jobs, preemption and production workloads. "
                 "Adaptive strategies are trained on five traces and checked within eight policies; finite training and "
                 "even twenty sweeps can leave residual incentives in the follow-up study. The planning attacker has limited information, finite "
                 "Monte Carlo computation and truthful continuation. The constrained oracle has future information "
                 "and is evaluated only on small instances. All values are synthetic. Negative sampled gains cannot "
                 "establish dynamic strategyproofness.\n\n"
                 "Historical E3b/E3c/E11 results are retained from checkpoint `1f54943` and explicitly labelled in "
                 "the tables. E15 replaces their attacker assumptions for the fresh evaluation; re-running the old "
                 "scripts with corrected code changes those numbers.\n")
    parts.append("## 6. Reproduction\n\n"
                 "From the repository root, activate `.venv`, install `requirements-lock.txt`, and run:\n\n"
                 "```text\npython -m pytest\npython run_all.py --extensions-only\n```\n\n"
                 "The extension command validates E0, runs all four studies on their fresh holdout, and regenerates "
                 "figures, result tables and this report. For development use `--quick` with separate results/figure "
                 "directories. The full pipeline also reruns earlier experiments with the corrected attacker.\n")
    if counts == {30}:
        parts.append("PDF figures: [component ablations](../figures/e12_ablations.pdf), "
                     "[adaptive policies](../figures/e13_adaptive.pdf), "
                     "[constrained welfare](../figures/e14_constrained.pdf), "
                     "[attack budgets](../figures/e15_attack_budget.pdf). "
                     "The final [data audit](../results/EXTENSION_AUDIT.json) verifies seed separation, "
                     "saved means, paired differences and oracle certification.\n")
    return "\n\n".join(parts) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="docs/COURSE_REPORT.md")
    ap.add_argument("--followup-report", default="FAILURE_CASE_STUDIES.md")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_manuscript(args.results_dir, args.followup_report), encoding="utf-8")
    print(f"Wrote {out}")
