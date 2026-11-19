# Final-report coverage of completed work

The report covers the original five baselines, **five history/service extensions**
(Score-Cap, Karma, Karma-Cap, Rank, Rank-Cap), and the separate Credit-Vickrey
budget extension. Uncapped Karma and Rank use `enable_wait_cap=False` in the
corresponding capped implementation. There are nine registered classes and
eleven named rule/configuration variants; parameter settings are not counted as
additional methods.

The eight-page body selects representative numerical results rather than printing
every grid cell. The table below maps every implemented study to its treatment in
the paper. Full grids remain in `results/e*/summary.json` and their `raw/` folders.

| Study | Completed work | Treatment in the final report |
|---|---|---|
| E0 | Exhaustive finite-instance validation and negative controls | Section 3.4; validation scope, without claiming a general dynamic incentive proof |
| E1 | Scarcity sweep over the five baselines | Methodology overview and Section 4.1; Table 1 selects k/n=.2 |
| E2 | Welfare/fairness/manipulation frontier over history penalties | Sections 3.4 and 4.1; sensitivity interpretation, not a new optimality theorem |
| E3 | Mechanism, policy and strategic-fraction factorial | Sections 3.4 and 4.1; own-population unilateral comparisons distinguished from a common-opponent ranking |
| E3b | Historical short-horizon planning attack | Historical status disclosed in the methodology overview and Section 4.4; numerical claims superseded by E15 |
| E3c | Historical planning-horizon sweep | Historical status disclosed; superseded by corrected E15, rather than represented as current-code output |
| E4 | Heterogeneous Beta groups and normalised benefits | Section 4.1, Table 1; mixed-demand loss also in Table 4 |
| E5 | Proposal recursion versus marginal-preserving persistence | Sections 3.4 and 4.1; generated Score starvation statistic and variance confound |
| E6 | Runtime and Python-heap scalability, n=10–500 | Section 4.1; saved runtime statistic, timing protocol and three-seed memory scope |
| E7 | Penalty × exaggeration-factor sensitivity | Section 4.1, Figure 1 and generated numerical comparisons |
| E8 | History-triggered reporting | Sections 3.2 and 3.4; explains why changing opponent profiles does not isolate timing |
| E9 | Truthful extension settings, balances, rank blending and wait thresholds | Section 3.3 and study overview; E12 supplies matched component results |
| E10 | Scripted attacks and parameter sensitivity for extension rules | Section 3.3 and overview; Table 4 and E17 supply matched/population failure examples |
| E11 | Historical extension planning attack | Historical status disclosed; Figure 4 uses corrected E15 instead |
| E12 | Score/Karma/Rank with and without a wait override, varied workloads and policies | Section 4.4, Table 4 now includes all six capped/uncapped rows and the all-inflating Karma-Cap caveat |
| E13 | Eight-policy asynchronous best responses, two starts, independent training/evaluation | Sections 3.4 and 4.5; basis for E18, not an unrestricted equilibrium claim |
| E14 | Offline optimum subject to the same service requirement | Section 4.4, Table 5; all 270 unique small-instance solves certified optimal |
| E15 | Corrected rollout: exact truth candidate, finite horizon, evolving histories and compute budgets | Section 4.4, Figure 4 selects H=40/R=80; the full budget grid remains in saved outputs |
| E16 | Rank-history fixed-report truthful-reference diagnostic | Section 4.5; generated history-effect interval, privileged reference and reused-seed qualifications |
| E17 | Karma forced/auction utility decomposition and credit-accounting diagnostics | Section 4.5; forced/auction gain components and uncapped reversal, without a unique-cause claim |
| E18 | Longer adaptive training and independent held-out evaluation | Section 4.5, Table 6; four/twenty sweeps, both starts, training status and simultaneous library-regret bands |
| E19 | Value-threshold policy and fixed-opponent timing controls | Sections 3.2 and 4.2, Table 2/Figure 2; every tested threshold shown |
| E20 | Renewable Credit-Vickrey budgets and attacks | Sections 3.3 and 4.3, Table 3/Figure 3; explicit budget model and limited incentive conclusions |

## Evidence and interpretation

- `analysis/build_final_report.py` regenerates each displayed statistic from saved
  seed-level rows, including the simultaneous adaptive regret bands. It checks
  training metadata and separates master, extension and follow-up seed roles.
- `analysis/verify_final_replay.py` reruns the first saved seed in 13 selected
  cells, including uncapped Karma and Rank. This is a representative code-to-data
  check, not a full-grid rerun.
- `analysis/verify_followups.py` checks diagnostic identities, paired summaries,
  credit transitions and reproduction against the original seed-level outcomes.
- The paper's Reproducibility section contains only dependencies, seeds and the
  code/results reproduction commands. Report construction and evidence details
  belong in `Project/FINAL_REPORT_README.md`.
