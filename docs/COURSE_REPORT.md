# Fairness, service guarantees, and strategic behaviour in shared GPU allocation


Aayush Kuloor, Harsh Dhru, Raj Modi, Kalp Shah — CS 329 course project


## Abstract


We study repeated allocation of identical GPUs to unit-demand users with private per-round values. The primary incentive question is the history-penalised Score rule M4: whether suppressing low reports preserves enough future priority to pay, and how that effect changes with the penalty. Greedy's manipulability and monetary Vickrey's per-round truthfulness serve as theory controls. The course extension separates the service override from two non-monetary incentive components, evaluates policy choice by adaptive users, strengthens the planning attacker, and compares online welfare with an offline optimum subject to the same service requirement. All four extension studies use 30 fresh heldout evaluation seeds. A wait cap alone leaves mild exaggeration profitable for Score: its mean individual gain changes from +20.96 to +20.77 at W=10. The corrected planning attacker earns a positive mean gain against Rank-Cap, with a 95% interval above zero. The results distinguish service guarantees, ordinal invariance, and empirical incentive robustness; none implies unrestricted dynamic strategyproofness.


## Core M4 analysis and reviewer feedback

The primary open question is how the history penalty changes profitable reporting and timing. Greedy's vulnerability and monetary Vickrey's per-round truthfulness are theoretical controls. E2/E7 vary the lie and penalty; E8 studies history-triggered inflation; E19 adds a scalable value-threshold policy at n=50, k=10, T=1000, with 30 seeds. E20 tests a non-monetary credit-budget version of Vickrey. These feedback experiments reuse existing master seeds and are exploratory; they are separate from the fresh E12-E15 holdout. The E13/E18 eight-policy libraries are unchanged and do not include these new policies, so their restricted convergence claims do not extend to them.

Strategic welfare loss is **SWL=(W_truthful-W_strategic)/W_truthful**, paired on the same workload. SWL may be negative if strategic reports improve welfare. The name avoids confusion with price of stability. The old Python function name remains a compatibility alias.


## E19 — Value-threshold timing attack (ρ=0.25)

sandbag_τ reports 0 below τ·v_max and v_max above; boost_τ is truthful below and v_max above. Profile-level M_uni holds opponents fixed within each gain, but changes their policy between cells. Its sandbag-minus-boost difference is descriptive and does not isolate timing.

**Controlled timing comparison.** For each focal user, hold opponents at boost_τ on S minus the focal user and truthful elsewhere; compare focal truth, boost and sandbag from fresh identical mechanism states on the same workload. Timing premium = U_f(sandbag) − U_f(boost). Average focal users within each seed, then bootstrap seeds. All intervals are pointwise, exploratory 95% intervals.

| Mechanism | τ | Sandbag gain [95% CI] | Boost gain [95% CI] | Timing premium [95% CI] |
|---|---|---|---|---|
| M3 | 0.5 | +268.50 [266.40, 270.46] | +268.50 [266.40, 270.46] | +0.000 [0.000, 0.000] |
| M3 | 0.75 | +52.88 [51.64, 54.11] | +57.77 [56.55, 58.99] | -4.889 [-5.122, -4.655] |
| M3 | 0.9 | -84.68 [-85.83, -83.52] | +0.31 [0.20, 0.43] | -84.989 [-86.140, -83.826] |
| M4 lambda=0.5 | 0.5 | +34.51 [33.64, 35.37] | +34.51 [33.64, 35.37] | +0.000 [0.000, 0.000] |
| M4 lambda=0.5 | 0.75 | +35.44 [34.86, 36.01] | +35.66 [35.14, 36.19] | -0.223 [-0.350, -0.107] |
| M4 lambda=0.5 | 0.9 | -83.21 [-84.29, -82.12] | +0.34 [0.26, 0.42] | -83.547 [-84.622, -82.475] |
| M4 lambda=1 | 0.5 | +4.48 [3.72, 5.27] | +4.48 [3.72, 5.27] | +0.001 [0.000, 0.004] |
| M4 lambda=1 | 0.75 | +20.76 [20.36, 21.14] | +20.65 [20.29, 21.00] | +0.112 [0.004, 0.211] |
| M4 lambda=1 | 0.9 | -82.74 [-83.86, -81.58] | +0.29 [0.23, 0.37] | -83.033 [-84.123, -81.902] |
| M4 lambda=2 | 0.5 | -10.98 [-11.57, -10.37] | -11.00 [-11.59, -10.38] | +0.017 [0.007, 0.029] |
| M4 lambda=2 | 0.75 | +10.70 [10.34, 11.04] | +10.14 [9.80, 10.46] | +0.565 [0.466, 0.660] |
| M4 lambda=2 | 0.9 | -81.88 [-82.97, -80.75] | +0.33 [0.27, 0.39] | -82.209 [-83.283, -81.095] |
| M4 lambda=5 | 0.5 | -19.09 [-19.50, -18.69] | -19.15 [-19.56, -18.74] | +0.057 [0.036, 0.081] |
| M4 lambda=5 | 0.75 | +4.44 [4.17, 4.71] | +3.07 [2.82, 3.32] | +1.372 [1.260, 1.494] |
| M4 lambda=5 | 0.9 | -80.31 [-81.39, -79.19] | +0.48 [0.42, 0.55] | -80.787 [-81.863, -79.681] |
| M5 | 0.5 | -75.88 [-76.51, -75.23] | -75.88 [-76.51, -75.23] | +0.000 [0.000, 0.000] |
| M5 | 0.75 | -4.39 [-4.51, -4.27] | -4.18 [-4.31, -4.06] | -0.205 [-0.217, -0.192] |
| M5 | 0.9 | -6.70 [-6.83, -6.57] | -0.00 [-0.01, -0.00] | -6.699 [-6.831, -6.570] |
| M5c rate=0.5 | 0.5 | -1.20 [-1.64, -0.76] | -2.45 [-2.83, -2.08] | +1.249 [0.923, 1.565] |
| M5c rate=0.5 | 0.75 | +15.66 [15.14, 16.17] | -0.18 [-0.39, 0.03] | +15.838 [15.323, 16.366] |
| M5c rate=0.5 | 0.9 | -55.86 [-57.01, -54.65] | +0.00 [0.00, 0.00] | -55.857 [-57.007, -54.648] |

**H2: paired change in timing premium relative to the smallest λ.** A positive interval supports an increase for this threshold and comparison, rather than establishing monotonicity for all λ or policies.

| τ | λ | Reference λ | Premium change [95% CI] |
|---|---|---|---|
| 0.5 | 1 | 0.5 | +0.00 [0.00, 0.00] |
| 0.5 | 2 | 0.5 | +0.02 [0.01, 0.03] |
| 0.5 | 5 | 0.5 | +0.06 [0.04, 0.08] |
| 0.75 | 1 | 0.5 | +0.34 [0.23, 0.45] |
| 0.75 | 2 | 0.5 | +0.79 [0.66, 0.93] |
| 0.75 | 5 | 0.5 | +1.60 [1.45, 1.75] |
| 0.9 | 1 | 0.5 | +0.51 [0.17, 0.85] |
| 0.9 | 2 | 0.5 | +1.34 [0.82, 1.85] |
| 0.9 | 5 | 0.5 | +2.76 [2.07, 3.46] |

**Population-profile results (opponents differ between reporting rules).**

| Mechanism | τ | M_uni sandbag [95% CI] | M_uni boost | Profile gain difference | M_uni max_claim | frac M_i>0 (sandbag) | SWL (sandbag) |
|---|---|---|---|---|---|---|---|
| M3 | 0.5 | +268.50 [266.40, 270.46] | +268.50 | +0.00 | +417.59 | 1.00 | 0.080 |
| M3 | 0.75 | +51.48 [50.27, 52.66] | +57.77 | -6.29 | +417.59 | 1.00 | 0.005 |
| M3 | 0.9 | -106.13 [-107.49, -104.78] | +0.31 | -106.44 | +417.59 | 0.00 | 0.011 |
| M4 lambda=0.5 | 0.5 | +34.51 [33.64, 35.37] | +34.51 | +0.00 | -29.85 | 1.00 | 0.038 |
| M4 lambda=0.5 | 0.75 | +35.28 [34.70, 35.85] | +35.66 | -0.38 | -29.85 | 1.00 | 0.004 |
| M4 lambda=0.5 | 0.9 | -104.46 [-105.70, -103.23] | +0.34 | -104.80 | -29.85 | 0.00 | 0.011 |
| M4 lambda=1 | 0.5 | +4.48 [3.72, 5.27] | +4.48 | +0.00 | -53.57 | 0.85 | 0.035 |
| M4 lambda=1 | 0.75 | +20.72 [20.30, 21.11] | +20.65 | +0.07 | -53.57 | 1.00 | 0.003 |
| M4 lambda=1 | 0.9 | -103.87 [-105.06, -102.63] | +0.29 | -104.17 | -53.57 | 0.00 | 0.010 |
| M4 lambda=2 | 0.5 | -11.00 [-11.58, -10.39] | -11.00 | +0.00 | -65.44 | 0.00 | 0.033 |
| M4 lambda=2 | 0.75 | +10.76 [10.41, 11.08] | +10.14 | +0.62 | -65.44 | 1.00 | 0.003 |
| M4 lambda=2 | 0.9 | -103.01 [-104.19, -101.78] | +0.33 | -103.34 | -65.44 | 0.00 | 0.009 |
| M4 lambda=5 | 0.5 | -19.09 [-19.50, -18.69] | -19.15 | +0.06 | -71.04 | 0.00 | 0.032 |
| M4 lambda=5 | 0.75 | +4.56 [4.28, 4.83] | +3.07 | +1.49 | -71.04 | 0.99 | 0.002 |
| M4 lambda=5 | 0.9 | -101.50 [-102.66, -100.29] | +0.48 | -101.98 | -71.04 | 0.00 | 0.007 |
| M5 | 0.5 | -75.88 [-76.51, -75.23] | -75.88 | +0.00 | -416.08 | 0.00 | 0.080 |
| M5 | 0.75 | -4.53 [-4.65, -4.40] | -4.18 | -0.34 | -416.08 | 0.00 | 0.005 |
| M5 | 0.9 | -10.95 [-11.15, -10.77] | -0.00 | -10.95 | -416.08 | 0.00 | 0.011 |
| M5c rate=0.5 | 0.5 | -1.39 [-1.72, -1.03] | -2.45 | +1.06 | -50.21 | 0.24 | 0.001 |
| M5c rate=0.5 | 0.75 | +15.05 [14.46, 15.59] | -0.18 | +15.23 | -50.21 | 1.00 | -0.034 |
| M5c rate=0.5 | 0.9 | -69.58 [-70.63, -68.47] | +0.00 | -69.58 | -50.21 | 0.00 | 0.008 |

At tau=.75, the paired premium change between lambda=0.5 and 5 is +1.60 [1.45, 1.75]. This supports the direction of H2 for this threshold and comparison. Threshold .9 is also reported; a large penalty does not make every suppression policy profitable. These pointwise exploratory intervals establish neither universal monotonicity nor dynamic strategyproofness.


M5c credits carry no utility and cannot be converted to money. Income per round is credit_rate*(k/n)*v_max; balances start at the ceiling of credit_horizon rounds of income. Income above the ceiling is discarded, existing credits do not age, bids are balance-capped, and winners' uniform-price payments are destroyed. Utility is allocation value only. This is a dynamic budget model and does not inherit monetary Vickrey's DSIC guarantee.


## E20 — Vickrey paid in credits instead of money (ρ=0.25)


**Truthful reports**

| Mechanism | WR | J_A | Q_max | p95 wait | SR_Δ |
|---|---|---|---|---|---|
| M5c rate=0.25 | 0.703 | 1.0000 | 10.9 | 5.0 | 0.0000 |
| M5c rate=0.5 | 0.844 | 1.0000 | 18.2 | 6.0 | 0.0030 |
| M5c rate=1 | 0.998 | 0.9969 | 41.5 | 12.7 | 0.0810 |
| M5c rate=2 | 1.000 | 0.9960 | 41.7 | 12.9 | 0.0844 |
| M5 (money) | 1.000 | 0.9960 | 41.7 | 12.9 | 0.0844 |
| M3 | 1.000 | 0.9960 | 41.7 | 12.9 | 0.0844 |

**Unilateral gain M_uni [95% CI] (fraction of focal users who gain)**

| Mechanism | cap_1.25 | cap_2 | max_claim | sandbag_0.75 |
|---|---|---|---|---|
| M5c rate=0.25 | -5.20 [-5.88, -4.52] (0.14) | -12.98 [-13.65, -12.25] (0.00) | -25.46 [-26.15, -24.79] (0.00) | +40.42 [39.75, 41.08] (1.00) |
| M5c rate=0.5 | -9.99 [-10.36, -9.60] (0.00) | -24.60 [-25.22, -23.96] (0.00) | -50.21 [-50.91, -49.50] (0.00) | +15.05 [14.46, 15.59] (1.00) |
| M5c rate=1 | +38.61 [37.08, 40.11] (1.00) | +9.87 [8.31, 11.42] (0.85) | -43.02 [-44.60, -41.46] (0.00) | +29.23 [28.27, 30.21] (1.00) |
| M5c rate=2 | +126.66 [124.85, 128.48] (1.00) | +218.88 [217.70, 220.07] (1.00) | +117.70 [116.50, 118.86] (1.00) | +51.48 [50.27, 52.66] (1.00) |
| M5 (money) | -14.04 [-14.27, -13.80] (0.00) | -101.83 [-102.57, -101.10] (0.00) | -416.08 [-417.73, -414.45] (0.00) | -4.53 [-4.65, -4.40] (0.00) |
| M3 | +126.66 [124.85, 128.48] (1.00) | +305.66 [303.75, 307.41] (1.00) | +417.59 [416.17, 418.99] (1.00) | +51.48 [50.27, 52.66] (1.00) |

Figures: [controlled M4 timing](../figures/e19_value_threshold.pdf), [credit-budget tradeoffs](../figures/e20_credit_budget.pdf).


Reproduce with `python experiments/e19_value_threshold.py`, `python experiments/e20_credit_budget.py`, `python analysis/verify_feedback.py`, `python analysis/plots.py`, `python analysis/report.py`, and `python analysis/course_manuscript.py`. See [FEEDBACK_PROTOCOL.md](FEEDBACK_PROTOCOL.md).


## 1. Research question and model


**How much welfare must online allocation sacrifice to guarantee service while limiting profitable strategic reporting?** We ask which gains are caused by the cap itself, whether agents select the scripted attacks when utility matters, and whether stronger planning changes the robustness conclusion.

In each round t, n users privately observe values v_i,t in [0,1], submit bounded reports, and compete for k<n GPUs. Exactly k users receive one round of service. Utility is the sum of true allocation benefits minus payments. Social welfare excludes payments, and WR divides true welfare by the sum of the top-k true values in each round. Allocation history and the valuation distribution are available to the simulated policy/attacker. The planning attacker samples opponents' current values rather than observing them and samples iid future values.

Waiting q_i(t) counts unserved rounds before round t. A threshold W gives eligible users oldest-first priority. The certified bound is B=W+ceil(n/k)-1, including the terminal state. Starvation uses the separate threshold Delta=2*ceil(n/k). The count and normalised-benefit Jain indices are reported alongside waits and welfare.


## 2. Mechanisms and formal properties


Greedy selects top reports; Score selects report/(1+prior allocations)^lambda; Vickrey uses top reports and charges each winner the next-highest bid. Score-Cap adds the shared waiting override. Karma-Cap caps bids by internal balances, charges uniform-price karma and redistributes it; karma is not a monetary utility deduction. Rank-Cap substitutes each user's empirical report quantile for raw report value before applying the history penalty and service override.

[THEORY.md](THEORY.md) proves two properties. First, the oldest-first override guarantees q_i(t)<=W+ceil(n/k)-1 for every report sequence, starting with zero waits. Second, with pure rank and fixed tie-breaking, a fixed strictly increasing transform of each user's entire report sequence preserves all allocations. This invariance covers neither clipping nor selective/time-varying distortions.

The oracle uses binary x_i,t, capacity equality and at least one service in every B+1-round window per user. With zero initial waits these constraints are equivalent to q_i(t)<=B, including q_i(T). For a feasible online schedule, W_online<=W*_B<=W*, so total welfare loss decomposes into the unavoidable service cost (W*-W*_B)/W* and the remaining loss (W*_B-W_online)/W*. The latter includes the disadvantage of not knowing future values.


## 3. Experimental design


All four extension studies use 30 fresh heldout evaluation seeds. The holdout is disjoint from the earlier 30 seeds and both development seed lists. Mechanism parameters and comparisons were fixed in [EXTENSION_PROTOCOL.md](EXTENSION_PROTOCOL.md) before examining the new outcomes. This is a local design record, not a claim of external preregistration. The previous results serve as development evidence. No new mechanism tuning uses the fresh holdout.


| Study | Question | Users | GPUs | Rounds | Seeds |
|---|---|---|---|---|---|
| E12 | Cap/component ablations | 50 | 10 | 1000 | 30 |
| E13 | Adaptive policy populations | 6 | 2 | 120 | 30 |
| E14 | Constrained offline welfare | 6 | 2 | 40 | 30 |
| E15 | Planning and computation budgets | 10 | 2 | 120 | 30 |


Each comparison uses paired valuation and tie-breaking streams. Ordinary intervals are 95% percentile bootstrap intervals with 10,000 resamples in the full study. Raw data, configs, seed roles, paired differences and source digests are saved.

E13 trains episode-long policy choices on five separate workloads using asynchronous best responses, starting from all-truthful and all-cap_2 populations. Its library contains truthful reporting, three exaggeration levels, maximum claims, two timed policies, and a high-value-only policy. Users optimise expected episode utility under fixed opponent policies; their final profiles are frozen before evaluation. The audit exhausts every unilateral policy switch and maximises the expected gain AFTER averaging across heldout seeds. Simultaneous bootstrap bands cover all inspected switches. Convergence refers to the training policy game. A cycle status records a repeated sweep-boundary profile, not proof of a periodic orbit under the shuffled update orders.

E15 adds exact truth to the report grid, truncates horizons to the remaining episode and advances mechanism/opponent state in hypothetical futures. Rank histories update in the main evaluation; matched frozen controls isolate the former approximation. The continuation policy is still truthful after the first hypothetical decision, so this remains a limited planning heuristic.


## 4. Results

### 4.1 What the cap and incentive components contribute


**truthful, rho=0; capped rows use W=10.**


| Mechanism | WR [CI] | Mean maximum wait | Individual gain [CI] | Fraction gaining |
|---|---|---|---|---|
| Score | 0.996 [0.996, 0.996] | 39.2 | +0.00 [0.00, 0.00] | 0.00 |
| Score-Cap W=10 | 0.948 [0.947, 0.948] | 10.0 | +0.00 [0.00, 0.00] | 0.00 |
| Karma | 0.974 [0.973, 0.974] | 36.0 | +0.00 [0.00, 0.00] | 0.00 |
| Karma-Cap W=10 | 0.941 [0.941, 0.942] | 10.0 | +0.00 [0.00, 0.00] | 0.00 |
| Rank | 0.993 [0.993, 0.993] | 39.5 | +0.00 [0.00, 0.00] | 0.00 |
| Rank-Cap W=10 | 0.945 [0.944, 0.945] | 10.0 | +0.00 [0.00, 0.00] | 0.00 |

**cap_1.25, rho=0.25; capped rows use W=10.**


| Mechanism | WR [CI] | Mean maximum wait | Individual gain [CI] | Fraction gaining |
|---|---|---|---|---|
| Score | 0.993 [0.993, 0.993] | 40.2 | +20.96 [20.51, 21.37] | 1.00 |
| Score-Cap W=10 | 0.944 [0.943, 0.944] | 10.0 | +20.77 [20.27, 21.25] | 1.00 |
| Karma | 0.957 [0.956, 0.957] | 35.1 | -13.68 [-14.07, -13.29] | 0.00 |
| Karma-Cap W=10 | 0.931 [0.930, 0.931] | 10.0 | -8.16 [-8.49, -7.85] | 0.00 |
| Rank | 0.992 [0.992, 0.992] | 39.7 | +2.48 [2.25, 2.72] | 0.91 |
| Rank-Cap W=10 | 0.943 [0.942, 0.944] | 10.0 | +2.16 [1.79, 2.53] | 0.80 |

**cap_2, rho=1; capped rows use W=10.**


| Mechanism | WR [CI] | Mean maximum wait | Individual gain [CI] | Fraction gaining |
|---|---|---|---|---|
| Score | 0.839 [0.838, 0.839] | 17.9 | -3.05 [-3.69, -2.45] | 0.23 |
| Score-Cap W=10 | 0.832 [0.831, 0.832] | 10.0 | +3.92 [3.15, 4.66] | 0.84 |
| Karma | 0.840 [0.840, 0.841] | 31.0 | +29.43 [28.85, 30.01] | 1.00 |
| Karma-Cap W=10 | 0.814 [0.813, 0.815] | 10.0 | +46.06 [45.40, 46.74] | 1.00 |
| Rank | 0.840 [0.839, 0.840] | 18.1 | -37.44 [-38.27, -36.64] | 0.00 |
| Rank-Cap W=10 | 0.835 [0.835, 0.836] | 10.0 | -34.32 [-35.26, -33.45] | 0.00 |

A reduction in average gain is distinct from eliminating gains for every user. The within-family paired comparisons isolate adding the cap; matched-W comparisons against Score-Cap isolate the incentive component. Truthful-row gains are zero by construction because both sides use truthful reporting; they are not a deviation audit. Full sensitivity tables are in [RESULTS.md](../results/RESULTS.md).


**Paired contrasts for mild exaggeration at W=10; candidate minus reference.**


| Candidate | Reference | WR difference [CI] | Gain difference [CI] |
|---|---|---|---|
| Score-Cap W=10 | Score | -0.049 [-0.050, -0.048] | -0.19 [-0.56, 0.17] |
| Karma-Cap W=10 | Karma | -0.026 [-0.027, -0.025] | +5.51 [5.10, 5.94] |
| Rank-Cap W=10 | Rank | -0.049 [-0.049, -0.048] | -0.33 [-0.64, -0.03] |
| Karma-Cap W=10 | Score-Cap W=10 | -0.013 [-0.014, -0.012] | -28.93 [-29.54, -28.32] |
| Rank-Cap W=10 | Score-Cap W=10 | -0.001 [-0.001, -0.000] | -18.61 [-18.99, -18.23] |

### 4.2 What utility-seeking agents choose



Policies are trained on independent seeds. Regret maximises expected paired gain after averaging; its interval comes from a simultaneous bootstrap band over all user/policy switches.


**Initial policy: truthful**

| Mechanism | Training status | Sweeps | WR [95% CI] | Mean utility | Restricted regret [band] | Truthful fraction |
|---|---|---|---|---|---|---|
| Greedy | converged | 2 | 0.643 [0.638, 0.648] | 20.20 | 0.00 [0.00, 1.55] | 0.00 |
| Score | converged | 4 | 0.976 [0.974, 0.977] | 30.67 | 0.17 [0.00, 1.12] | 0.00 |
| Score-Cap | sweep_limit | 4 | 0.948 [0.945, 0.951] | 29.80 | 0.40 [0.00, 1.29] | 0.00 |
| Vickrey | converged | 1 | 1.000 [1.000, 1.000] | 8.38 | 0.00 [0.00, 1.36] | 1.00 |
| Karma-Cap | sweep_limit | 4 | 0.954 [0.950, 0.958] | 29.99 | 0.58 [0.00, 1.50] | 0.00 |
| Rank-Cap | sweep_limit | 4 | 0.948 [0.945, 0.951] | 29.78 | 0.63 [0.00, 1.49] | 0.33 |

- Greedy: `max_claim, max_claim, max_claim, max_claim, max_claim, max_claim`
- Score: `cap_1.5, cap_1.25, cap_1.25, cap_1.25, cap_1.5, cap_1.25`
- Score-Cap: `cap_1.5, timed_cap_1.25, cap_1.5, timed_cap_1.25, cap_1.5, cap_1.5`
- Vickrey: `truthful, truthful, truthful, truthful, truthful, truthful`
- Karma-Cap: `cap_1.25, timed_cap_1.25, timed_cap_1.25, cap_1.25, cap_1.25, cap_1.25`
- Rank-Cap: `timed_cap_1.25, truthful, cap_1.25, truthful, cap_1.25, high_only`

**Initial policy: cap_2**

| Mechanism | Training status | Sweeps | WR [95% CI] | Mean utility | Restricted regret [band] | Truthful fraction |
|---|---|---|---|---|---|---|
| Greedy | converged | 2 | 0.643 [0.638, 0.648] | 20.20 | 0.00 [0.00, 1.55] | 0.00 |
| Score | converged | 4 | 0.976 [0.974, 0.977] | 30.67 | 0.17 [0.00, 1.12] | 0.00 |
| Score-Cap | sweep_limit | 4 | 0.952 [0.949, 0.955] | 29.92 | 0.62 [0.00, 1.51] | 0.00 |
| Vickrey | converged | 2 | 1.000 [1.000, 1.000] | 8.38 | 0.00 [0.00, 1.36] | 1.00 |
| Karma-Cap | sweep_limit | 4 | 0.936 [0.933, 0.940] | 29.43 | 0.80 [0.00, 1.83] | 0.00 |
| Rank-Cap | sweep_limit | 4 | 0.948 [0.945, 0.951] | 29.78 | 0.63 [0.00, 1.49] | 0.33 |

- Greedy: `max_claim, max_claim, max_claim, max_claim, max_claim, max_claim`
- Score: `cap_1.5, cap_1.25, cap_1.25, cap_1.25, cap_1.5, cap_1.25`
- Score-Cap: `cap_1.5, cap_1.25, timed_cap_1.25, cap_1.25, cap_1.25, cap_1.5`
- Vickrey: `truthful, truthful, truthful, truthful, truthful, truthful`
- Karma-Cap: `cap_1.5, cap_1.25, cap_1.5, cap_1.5, timed_cap_1.25, cap_1.5`
- Rank-Cap: `timed_cap_1.25, truthful, cap_1.25, truthful, cap_1.25, high_only`

Training convergence need not generalise exactly to the holdout. A sweep-limit profile may retain profitable switches even on the training set. These are restricted-policy results, not claims of unrestricted equilibrium, and the two starts do not exhaust basins of attraction.


Every heldout regret band includes zero. Positive point estimates therefore do not establish statistically positive expected regret after accounting for all inspected switches. A training sweep limit and residual training incentives still prevent an equilibrium claim.


### 4.3 Welfare after accounting for the service requirement


The middle threshold W=6 is shown below. Optimal fraction reports solver certification, not an assumption.


| Demand | Mechanism | B | Efficiency [CI] | Service loss | Remaining loss | Optimal fraction |
|---|---|---|---|---|---|---|
| base | Score-Cap | 8 | 0.969 [0.964, 0.973] | 0.002 | 0.031 | 1.00 |
| base | Karma-Cap | 8 | 0.971 [0.967, 0.975] | 0.002 | 0.029 | 1.00 |
| base | Rank-Cap | 8 | 0.928 [0.921, 0.935] | 0.002 | 0.071 | 1.00 |
| mixed | Score-Cap | 8 | 0.917 [0.912, 0.921] | 0.049 | 0.079 | 1.00 |
| mixed | Karma-Cap | 8 | 0.832 [0.826, 0.838] | 0.049 | 0.160 | 1.00 |
| mixed | Rank-Cap | 8 | 0.824 [0.815, 0.832] | 0.049 | 0.168 | 1.00 |
| persistent | Score-Cap | 8 | 0.886 [0.880, 0.892] | 0.054 | 0.108 | 1.00 |
| persistent | Karma-Cap | 8 | 0.865 [0.855, 0.875] | 0.054 | 0.128 | 1.00 |
| persistent | Rank-Cap | 8 | 0.787 [0.770, 0.804] | 0.054 | 0.201 | 1.00 |

When all solves are optimal, the decomposition is exact. Otherwise efficiency is bounded using the solver's primal and dual objectives and the loss columns using the dual bound are not exact unavoidable/algorithm costs. The oracle experiments are deliberately small and have future information.


### 4.4 Stronger planning and rank-history manipulation


Largest computation budget: H=40, 80 Monte Carlo futures per candidate.


| Mechanism | History model | Focal gain [CI] | Truthful fraction |
|---|---|---|---|
| Greedy | exact | +39.64 [38.78, 40.53] | 0.02 |
| Score | exact | +3.47 [3.07, 3.88] | 0.38 |
| Score-Cap | exact | +2.60 [2.29, 2.94] | 0.36 |
| Vickrey | exact | +0.00 [0.00, 0.00] | 1.00 |
| Karma-Cap | exact | -0.32 [-0.80, 0.16] | 0.41 |
| Rank-Cap | updating | +2.58 [2.19, 2.98] | 0.36 |
| Rank-Cap | frozen | -0.92 [-1.80, -0.09] | 0.31 |
| Score lambda=2 | exact | +2.24 [1.95, 2.52] | 0.45 |

The corrected planning attacker earns a positive mean gain against Rank-Cap, with a 95% interval above zero. Paired updating-minus-frozen differences measure the approximation's effect. A mean gain with an interval above zero supports a profitable tested deviation in this workload model; a negative gain only means that this particular attacker failed. Budget sensitivity makes that limitation visible.


At the largest budget, updating-minus-frozen Rank-Cap gain is +3.50 [2.75, 4.33]. This is a paired comparison on the same workloads and attacker random streams.


### 4.5 Failure-mode follow-ups

The [failure case studies](FAILURE_CASE_STUDIES.md) trace reported and truthful rank references, replay identical attack sequences with a truthful reference, and decompose Karma gains into forced/auction service with verified credit accounting. The Rank and Karma diagnostics reuse existing workloads for explanation. A four-vs-twenty-sweep adaptive comparison uses a separate new evaluation set; it records recurrence without treating a repeated profile as convergence. These analyses qualify causal explanations of the failures and the original four-sweep findings.


## 5. Related work and limitations

Published [Karma (OSDI 2023)](https://www.usenix.org/conference/osdi23/presentation/vuppalapati) credits donated unused capacity that others borrow in a dynamic-demand model. Our uniform-price bid auction with redistribution is a different mechanism; its guarantees do not transfer. [Fikioris, Agarwal and Tardos](https://arxiv.org/abs/2109.12401) study incentives for dynamic MMF/DRF and show that history-aware allocation can lose incentive compatibility. Their demand/resource model also differs from this private-value unit-demand model. Our contribution is a controlled component study and finite-policy strategic evaluation, together with explicit service and ordinal-invariance proofs.

The model omits arrivals, job duration, placement, multi-GPU jobs, preemption and production workloads. Adaptive strategies are trained on five traces and checked within eight policies; finite training and even twenty sweeps can leave residual incentives in the follow-up study. The planning attacker has limited information, finite Monte Carlo computation and truthful continuation. The constrained oracle has future information and is evaluated only on small instances. All values are synthetic. Negative sampled gains cannot establish dynamic strategyproofness.

Historical E3b/E3c/E11 results are retained from checkpoint `1f54943` and explicitly labelled in the tables. E15 replaces their attacker assumptions for the fresh evaluation; re-running the old scripts with corrected code changes those numbers.


## 6. Reproduction

From the repository root, activate `.venv`, install `requirements-lock.txt`, and run:

```text
python -m pytest
python run_all.py --extensions-only
```

The extension command validates E0, runs all four studies on their fresh holdout, and regenerates figures, result tables and this report. For development use `--quick` with separate results/figure directories. The full pipeline also reruns earlier experiments with the corrected attacker.


PDF figures: [component ablations](../figures/e12_ablations.pdf), [adaptive policies](../figures/e13_adaptive.pdf), [constrained welfare](../figures/e14_constrained.pdf), [attack budgets](../figures/e15_attack_budget.pdf). The final [data audit](../results/EXTENSION_AUDIT.json) verifies seed separation, saved means, paired differences and oracle certification.
