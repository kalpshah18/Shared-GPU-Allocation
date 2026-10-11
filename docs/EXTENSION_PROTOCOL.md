# Course extension protocol (v1)

This protocol is fixed before E12-E15 evaluation. The previous 30 master seeds
and results are development evidence for this extension. They are not renamed
or reused as a new holdout. `seeds/extension_seeds.json` locks 30 fresh evaluation
seeds, disjoint from the previous seeds, M6 development seeds 101-105, and policy
training seeds 201-205. No mechanism hyperparameter is selected using new results.

## Question and primary comparisons

How much welfare is lost when online unit-demand allocation must guarantee
service and withstand individually profitable reporting policies? Component
ablations isolate the waiting override from the incentive component. Adaptive
policy choice and a stronger planning attacker challenge scripted-attack results.

## E12: component ablations

- n=50, k=10, T=1000; Score lambda=1, karma supply=2, pure rank lambda=1.
- Score, Karma and Rank without a cap, and their capped versions at W=10,15,20.
- Truthful homogeneous, mixed Beta, and persistent copula alpha=0.9 workloads.
- Homogeneous attacks: cap_1.25, cap_2 and timed_cap_1.25 at rho=0.25;
  cap_2 at rho=1. Five focal users per seed measure unilateral gain.
- Paired component comparisons against Score-Cap at identical W, and within-family
  cap effects. Report WR, benefit/count fairness, waiting and unilateral gains.

## E13: adaptive policy populations

- Small repeated game: n=6, k=2, T=120. Each user chooses an episode-long policy.
- Six mechanisms: Greedy, Score lambda=1, Score-Cap, Vickrey, Karma-Cap, Rank-Cap;
  capped versions use W=Delta=6.
- Library: truthful, cap_1.25, cap_1.5, cap_2, max claim, timed_cap_1.25,
  timed_cap_2 and high_only (truthful above v_max/2, zero below).
- All-truthful and all-cap_2 starts. Asynchronous best responses optimise expected
  utility on five training seeds; fixed shuffled order seed 329, at most four
  sweeps, current policy retained on ties. Record convergence, recurrence or limit.
- Freeze each trained profile, evaluate it on all 30 fresh seeds, and exhaust all
  unilateral policy switches. Exploitability is the maximum AFTER seed-averaging,
  not the average maximum on each known workload. Bootstrap simultaneous bands
  across all user/policy deviations; report welfare and residual exploitability.
- This is a finite-policy empirical game, not unrestricted equilibrium computation.
  Cycle detection is recurrence of a sweep-boundary profile under the update schedule.

## E14: constrained welfare oracle

- n=6, k=2, T=40; homogeneous, mixed, and copula alpha=0.9; W=3,6,9.
- Online Score-Cap, Karma-Cap, Rank-Cap all share the certified bound
  B=W+ceil(n/k)-1, rather than an observed wait from the sampled workload.
- Offline binary MILP maximises true welfare with exactly k winners per round and
  at least one service per user in every window of length B+1. Initial waits are
  zero; include the terminal q(T) state. The oracle observes future values.
- Independently validate solver allocations, capacity and wait. A 30-second solve
  limit is recorded. Report solver optimality and gap; use dual upper bounds when
  a solve times out, explicitly bounding rather than claiming exact efficiency.
- Decompose loss against unconstrained welfare into the service-constraint cost
  and remaining online algorithm loss when optimality is certified.

## E15: attack computation and report-history dynamics

- n=10, k=2, T=120; truthful opponents; focal user 0.
- Six E13 mechanisms plus Score lambda=2.
- (horizon, Monte Carlo count) = (5,20), (5,80), (40,20), (40,80).
- Reports {0,0.1,...,1} plus exact truth, common random numbers across candidates,
  horizon truncated to remaining episode length. Simulated opponent policies read
  the simulated allocation state. Future values are iid; focal continuation is truthful.
- Rank histories update in hypothetical futures; matched frozen controls at H=40
  isolate the former approximation. Stateful karma and wait counters evolve exactly.
- Report focal utility gain, truthful/max-report frequency and paired budget effects.
  Negative gain is a failure of this attacker, not a truthfulness guarantee.

## Reporting and provenance

All primary results use 30 seeds and 10,000 bootstrap resamples. Smoke runs write
to separate directories. Save per-seed data, full configs, comparisons, seed roles
and a SHA-256 digest of the Python source. Fixed protocols are not proof of
preregistration; this file is a local, reviewable record of the design.

Source digests describe the on-disk Python files when outputs are saved. Report
generation and metadata improvements can change that digest without changing
the simulated algorithms. `results/EXTENSION_AUDIT.json` records the final
seed/mean/comparison audit and descriptor-only metadata corrections; numerical
outcomes and the original saved digests are retained.

Historical E3b/E3c/E11 results at checkpoint `1f54943` used a grid-only candidate
set, horizons beyond episode end and frozen rank quantiles. They remain historical
evidence; E15 is the corrected fresh-seed evaluation. Re-running those old scripts
now uses the corrected attacker and will not reproduce their historical numbers.
