# Fairness, Efficiency, and Strategic Behaviour in Shared GPU Allocation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A simulation study comparing repeated GPU-allocation mechanisms on allocative efficiency (welfare ratio WR),
> long-run fairness (Jain indices J_A, J_B; waiting and starvation), and resistance to bounded strategic
> misreporting (paired manipulation gain M, price of strategy). The full design is in
> [`FAI_Project_Proposal.pdf`](FAI_Project_Proposal.pdf); every item it asks for is implemented, tested and
> traceable in the [coverage table](#proposal-coverage) below.

---

## Course extension: component effects and strategic policy choice

The current research question is: **how much welfare must online GPU allocation
sacrifice to guarantee service while limiting profitable manipulation?**

The extension adds a **Score-Cap baseline**, explicit uncapped Karma and Rank
ablations, adaptive episode-long policy choice, an offline welfare oracle with
the same certified service requirement, and a stronger planning attacker.
E12-E15 use **30 fresh heldout seeds**. Policies are trained on five separate
development seeds; the earlier results are development evidence for this extension.
The fixed design and seed roles are documented in
[EXTENSION_PROTOCOL.md](docs/EXTENSION_PROTOCOL.md).

- **E12:** isolate caps from incentive components at identical W; test truthful,
  heterogeneous, persistent and strategic workloads.
- **E13:** asynchronous best responses within eight reporting policies; freeze
  trained profiles, then audit every unilateral switch on heldout workloads.
- **E14:** a binary offline optimisation benchmark with capacity and service
  windows, including terminal waiting; separate unavoidable service cost from
  the remaining online algorithm loss.
- **E15:** exact truth among report candidates, horizons truncated at episode
  end, evolving Rank-Cap report histories, multiple computation budgets, and
  matched frozen-history controls.

[THEORY.md](docs/THEORY.md) gives proofs of the service bound and pure-rank
invariance under fixed strictly increasing report transformations. These are
distinct from dynamic strategyproofness. The full course report is generated
from the saved data in [COURSE_REPORT.md](docs/COURSE_REPORT.md).

The fresh evaluation gives three useful findings:

- **A cap alone does not deter mild exaggeration.** At W=10, c=1.25 and rho=0.25,
  mean unilateral gain is +20.77 for Score-Cap, -8.16 for Karma-Cap, and +2.16
  for Rank-Cap over 1,000 rounds. Identical caps isolate the incentive component.
- **Rank-Cap is vulnerable to history-aware planning.** At H=40 with 80 sampled
  futures per candidate, the corrected attacker gains +2.58 utility over 120
  rounds, with a 95% interval [2.19, 2.98]. Fixed-monotone rank invariance does
  not protect against selective report-history manipulation.
- **Welfare loss extends beyond the service requirement.** All 270 small oracle
  instances are certified optimal. In heterogeneous demand at W=6, Score-Cap
  attains about 92% of constrained optimal welfare, versus about 82-83% for
  Karma-Cap and Rank-Cap. Adaptive policy training also leaves residual training
  incentives after four sweeps for the capped rules; heldout regret bands include
  zero, so their positive estimates need qualification.

```bash
python run_all.py --extensions-only
# Development outputs stay separate from the evaluated results:
python run_all.py --extensions-only --quick --results-dir tmp/course-results --figures-dir tmp/course-figures
```

**Historical evidence:** E3b, E3c and E11 tables below retain checkpoint
`1f54943`, which used grid-only candidates, horizons beyond episode end and
frozen rank quantiles. E15 is the corrected fresh-seed evaluation. Re-running
the older attack scripts now uses corrected code and changes their numbers.

### Explaining the failures

E16-E18 investigate the mechanisms behind the results: Rank-Cap trajectories and
fixed-report truthful-reference replay, Karma's forced/auction benefit and credit
accounting, and four-vs-twenty-sweep adaptive training. Rank and Karma diagnostics
reuse the previous evaluation workloads; extended adaptation uses 30 separate
evaluation seeds. The [follow-up protocol](docs/FOLLOWUP_PROTOCOL.md) defines the
comparisons and the [case-study report](docs/FAILURE_CASE_STUDIES.md) presents the
results, figures, and a course presentation outline. The runner audits seed
separation, replay/service identities, and credit transitions; full diagnostic
runs also reproduce all 180 corresponding E12/E15 seed-level outcomes.

- **Rank-Cap:** changing report history contributes +1.31 utility [0.88, 1.73]
  to the updating attack in a fixed-report replay, but the attack also profits
  with a truthful reference. Selective reporting matters; it is not explained
  solely by sacrificing low-value rounds.
- **Karma-Cap:** at rho=1, +66.96 auction benefit outweighs -20.90 forced-service
  benefit. The reversal also occurs without the cap. Forced service is charged,
  and inflated opponents make the deviator's credit constraint less binding.
- **Longer adaptation:** Karma-Cap converges on the training game at sweep 18
  from both initial profiles. Score-Cap and Rank-Cap still have residual training
  incentives at sweep 20. One four-sweep Karma-Cap profile has positive regret
  on the independent evaluation set; a training fixed point is a restricted claim.

```bash
python run_all.py --followups-only
```

---

## Earlier study: what the results do and do not show

1. **"λ≈2 deters manipulation" is true only for one lie.** Against capped exaggeration with c=2, individual inflation stops
   paying at λ=2 (E2). The deterrence threshold moves *up* as the lie gets milder: c=1.5 needs λ between 2 and 5, and
   at c=1.25 inflation is still (slightly) profitable at λ=5 (E7). Mild lies are the dangerous ones for Score.
2. **A sufficiently far-sighted attacker beats Score even at λ=2.** The proposal's 5-round rollout attack finds nothing
   on Score (E3b), but that is a horizon artefact: with a 40–80-round horizon the same search earns a significant gain
   at λ=1 and λ=2 (E3c), including where capped exaggeration loses. The null result in E3b is *not* evidence of robustness.
3. **Simple "timing" does not pay** (E8): inflating only when one's own history is favourable earns less than inflating
   always wherever inflation is profitable. The profitable attack in (2) is selective in a way these rules are not.
4. **For c=2, the effects at λ≥1 are small in absolute terms** (a few utility units per user over 1000 rounds, against +306 for
   Greedy). They are statistically clear, but whether they matter in practice is a modelling question, not a result.
5. **E5:** the proposal's AR(1) recursion also narrows the value spread; its welfare trend for report-blind rules is that
   artefact, not persistence (a marginal-preserving control is included).
6. **We also designed two new mechanisms (M6 Karma-Cap, M7 Rank-Cap; see "Proposed mechanisms").** They trade welfare for
   a *hard* waiting guarantee and much stronger deterrence of lying than Score, but neither wins everywhere: both lose
   welfare under heterogeneous or persistent demand, M6 loses its deterrence if *everyone else* already lies, and M7 is
   still (weakly) gameable by very mild lies. They are a different point on the trade-off, not a free improvement.

---

## Repository structure

```
sim/                         Pure-Python simulator
  config.py                  Validated Config dataclass (+ YAML loading)
  environment.py             Seeded valuations (i.i.d. / Beta / two AR(1) variants), History, strict-JSON store
  runner.py                  Round loop: mixed populations, paired runs, unilateral gains, rollout runner
  metrics.py                 WR, J_A, J_B, NSW, Q_max, p95 wait, SR_Δ, PoF, PoS, coalition & unilateral M
  mechanisms/                M1 Random · M2 Round-Robin · M3 Greedy · M4 Score · M5 Vickrey (+ factory)
                             M6 Karma-Cap · M7 Rank-Cap (proposed) · shared wait cap (_waitcap.py)
  policies/strategic.py      Truthful · capped exaggeration · maximum claim · rollout attack (batched)
experiments/
  e0_validation.py           Exhaustive checks on small discrete instances (67k instances)
  e1_scarcity.py             k/n sweep                       e5_persistence.py   AR(1) persistence (+ sensitivity)
  e2_frontier.py             λ sweep, Pareto frontier        e6_scalability.py   time per round, measured heap
  e3_strategic.py            ρ × policy factorial            e7_sensitivity.py   λ × exaggeration-factor grid
  e3b_rollout.py             rollout attack (H=5)            e4_heterogeneous.py equal service vs equal benefit
  e3c_rollout_horizon.py     rollout attack vs horizon       e8_timing.py        timing attack (hypothesis H2)
  e9_proposed_truthful.py    M6/M7 truthful + wait frontier  e10_proposed_strategic.py  M6/M7 vs every lie
  e11_proposed_rollout.py    M6/M7 vs far-sighted attacker   proposed.py         specs + development seeds
  common.py                  Seeds, CLI, raw/summary persistence, strategic-cell measurement
analysis/
  bootstrap.py               95% bootstrap CIs (10,000 resamples) and paired-difference CIs
  pareto.py                  Multi-objective dominance filter
  plots.py                   All figures (PDF + PNG)             report.py   results/RESULTS.md tables
tests/                       501 tests (see "Tests")
docs/                        Fixed extension protocol, proofs, generated course report
config/base.yaml             Base configuration (n=50, k=10, T=1000, ρ=0.25, …)
seeds/                       generate_seeds.py + the 30 locked master seeds
results/  figures/           Generated outputs  (results/RESULTS.md = every table)
run_all.py / run_all.sh      One command: validate → run everything → regenerate every figure
```

## Project report (LaTeX)

`report/main.tex` is the full project report in the PRIME AI style (`report/PRIMEarxiv.sty`). Its tables and
figures are generated from the saved results, so no number is typed by hand:

```bash
python report/make_tables.py     # report/generated/*.tex
python report/make_figures.py    # report/media/*.pdf
cd report && latexmk -pdf main.tex
```

`report/CITATIONS.md` records how each reference was verified.

## Setup

The project is developed and tested in a dedicated virtual environment (Python 3.13; 3.10+ works).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell:  .venv\Scripts\Activate.ps1   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # exact versions used for the reported results: requirements-lock.txt
```

## Quick start

```bash
python -m pytest                  # 501 tests; add -m "not slow" to skip the end-to-end pipeline test
python experiments/e0_validation.py
python run_all.py                 # everything, from the locked seeds → results/ and figures/  (tens of minutes)
python run_all.py --quick         # 2-seed smoke run on tiny instances, for development only
python analysis/report.py         # regenerate results/RESULTS.md from results/*/summary.json
```

Every experiment script accepts `--seeds N`, `--results-dir DIR`, `--n`, `--T`, `--bootstrap`.

## Tests

`python -m pytest` runs 501 tests (`-m "not slow"` skips the one end-to-end pipeline test). They are organised by component:

| File | What it pins down |
|---|---|
| `test_config.py` | validation of every field, derived Δ, YAML loading, `base.yaml == defaults` |
| `test_environment.py` | determinism, paired randomness, nested strategic sets, bounds, marginal means, AR(1) autocorrelation / variance (both variants), History bookkeeping, strict JSON |
| `test_mechanisms.py` | capacity/payment invariants for every mechanism and size, report-invariance, uniform tie-breaking, scale-free ties (regression for the jitter bug), M2 wait bound, M3 oracle, M4 formula and monotonicity, M5 payments and **exhaustive DSIC**, negative controls, batch ≡ single allocation, determinism |
| `test_metrics.py`, `test_metrics_extra.py` | hand-computed 2-user cases, pre-round waiting semantics, Jain-index properties, J_B, NSW, payments excluded from welfare, coalition / unilateral / PoS / PoF |
| `test_policies.py` | elementwise bounded policies, timed policy (thresholds, per-user history, never above always-inflating), registry, rollout attack (inflates under Greedy, exactly truthful under Vickrey, deterministic, side-effect free, scales with `v_max`) |
| `test_runner.py` | mixed / paired / unilateral runners vs manual counterfactuals, stateful-mechanism reset, report validation, rollout runner |
| `test_proposed_mechanisms.py` | the shared wait cap, hand-computed Karma-Cap rounds (price, dividend, forced payment), karma conservation / ceiling / bid-cap, Rank-Cap quantiles, **invariance to increasing distortions**, `β=1 ≡ Score`, the hard waiting bound under adversarial reports, rollout interface ≡ real allocation, rollout attacker side-effect freedom |
| `test_proposed_experiments.py` | E9–E11 schema / determinism / qualitative design goals; development seeds disjoint from the locked seeds |
| `test_analysis.py` | bootstrap CI coverage, pairing benefit, NaN handling, paired differences, Pareto filter vs brute force |
| `test_experiments.py` | every experiment end to end on tiny instances: schema, strict JSON, raw files, determinism, and the qualitative findings (Greedy rewarded / Vickrey punished, E5 variance confound, E7 monotonicity, …); E0 **fault injection** (a broken Vickrey / capacity bug must be caught) |
| `test_plots_and_pipeline.py` | every figure is written, graceful skips, report tables, locked seed file, `run_all.py --quick` end to end |
| `test_course_extensions.py` | multi-round real-vs-rollout trajectories; active-cap rank invariance; scale-independent service priority; exhaustive oracle schedules; training/evaluation separation; expected-gain audit; E12-E15 data, figures and tables |
| `test_failure_diagnostics.py` | actual-vs-observed Rank/Karma trajectories; truthful-reference replay; exact credit accounting; recurrence continuation; follow-up seed roles, strict JSON and figures |

## Model in one paragraph

`n` unit-demand users compete for `k < n` identical GPUs in each of `T` rounds. User `i` has a private value
`v_{i,t} ∈ [0, v_max]` and reports `v̂_{i,t}`. A mechanism returns an allocation `x_t` (exactly `k` winners) and
payments `p_t` (zero for M1–M4). Utility is quasi-linear, `u = v·x − p`. All comparisons are **paired**: for one master
seed every mechanism and every counterfactual run reads the same valuation tensor, tie-breaking seeds and strategic set.

| ID | Mechanism | Reports | History | Payments | Formal property |
|---|---|---|---|---|---|
| M1 | Random: `k` users uniformly | no | no | no | report-invariant |
| M2 | Round-robin over a seeded queue | no | yes | no | report-invariant; `Q_max ≤ ⌈n/k⌉−1` |
| M3 | Greedy: top-`k` reports | yes | no | no | first-best under truthful reports only |
| M4 | Score `s = v̂ / (1+a_i)^λ`, top-`k` (default λ = 1) | yes | yes | no | `λ = 0 ≡ M3`; empirically stress-tested |
| M5 | `k`-unit Vickrey, winners pay the `(k+1)`-st bid | yes | no | yes | per-round DSIC (VCG) |
| **M6** | **Karma-Cap** (proposed): karma auction + hard wait cap, below | yes | yes | no (internal karma) | `Q_max ≤ W + ⌈n/k⌉ − 1`; truthfulness empirical |
| **M7** | **Rank-Cap** (proposed): rank-normalised score + wait cap, below | yes | yes | no | `Q_max ≤ W + ⌈n/k⌉ − 1`; invariant to fixed strictly increasing distortions in pure-rank mode; truthfulness empirical |

## Metrics

| Metric | Definition |
|---|---|
| WR | `W / W*`, `W = Σ v·x` (payments are transfers), `W*` = sum of the top-`k` true values per round |
| PoF | `(W* − W_F) / W*` — welfare lost by mechanism `F` under truthful reports |
| J_A | Jain's index of cumulative allocations `A_i` |
| J_B | Jain's index of normalised gross benefits `B_i = G_i / (T·μ_i)`, `μ_i = E_{D_i}[v]` |
| NSW | `Σ log(G_i + 10⁻⁸)` |
| `q_i(t)`, Q_max | consecutive rounds *before* round `t` without service (`q_i(0)=0`); `Q_max = max q_i(t)` |
| p95 wait | 95th percentile of all `q_i(t)` |
| SR_Δ | `mean 1{q_i(t) > Δ}`, `Δ = 2⌈n/k⌉` |
| Coalition gain | `M̄, M_max, frac_pos` over the strategic set: `U_i(S deviates) − U_i(all truthful)`. Strategic users crowd each other out, so a negative value does **not** mean an individual is better off truthful |
| **Unilateral gain** `M_i` | `U_i(σ_i, σ_{−i}; ω) − U_i(truthful, σ_{−i}; ω)` for `n_focal = 5` strategic users per seed (members of `S` drop out of the deviating set; at ρ=0 each focal user is a lone deviator). Reported as `M_uni` (mean), `M_uni_max`, `frac_pos_uni`. This is the proposal's individual incentive |
| PoS | `(W_truthful − W_strategic) / W_truthful` |

Every number is a mean over the 30 master seeds with a 95% percentile-bootstrap CI (10,000 resamples); differences
between cells that share seeds use the **paired** bootstrap (`results/*/paired.json`).

## Experiments

| | Question | Design |
|---|---|---|
| E0 | Is the implementation correct? | Exhaustive: capacity & payment invariants, report-invariance, M3≡M4(λ=0), RR wait bound for all `1≤k<n≤9`, welfare oracle, M5 DSIC / payments / individual rationality, **negative controls** (the DSIC enumerator must find profitable lies under M3, M4), determinism, batch-vs-single allocation, vectorised-vs-naive metrics |
| E1 | Scarcity | `k/n ∈ {0.1,0.2,0.4,0.6,0.8}`, truthful |
| E2 | Fairness / strategy frontier | `λ ∈ {0,0.05,0.1,0.25,0.5,1,2,5}` + M1,M2,M3,M5; ρ=0.25, cap_2; Pareto flags on (WR↑, J_A↑, SR_Δ↓, M_uni↓) |
| E3 | Population × attack | `ρ ∈ {0,0.1,0.25,0.5,1}` × {truthful, cap_2, max_claim} × 5 mechanisms |
| E3b | Rollout attack (diagnostic) | n=10, T=500, H=5, 100 rollouts, grid {0,0.1,…,1}; M3, M4(λ=1,2), M5; opponents truthful or cap_2 |
| E3c | Rollout horizon | M4 (λ=1,2), opponents truthful, `H ∈ {5,10,20,40,80}`, n=10, T=300, vs capped c=2 and c=1.25 |
| E4 | Heterogeneous users | uniform vs `Beta(2,5)` / `Beta(5,2)` halves; J_A vs J_B |
| E5 | Persistence (stretch) | `α ∈ {0,0.5,0.9}`; proposal AR(1) **and** a marginal-preserving copula variant |
| E6 | Scalability (stretch) | `n ∈ {10,…,500}`, `k/n=0.2`; µs per round, **measured** (`tracemalloc`) peak heap |
| E7 | λ × c sensitivity | `c ∈ {1.25,1.5,2}` × M4 `λ ∈ {0.5,1,2,5}`, plus M3, M5 |
| E9 | Proposed mechanisms, truthful | M6, M7 (+ M7 β=0.25) vs M1–M5 on base, scarce, loose, mixed, persistent (proposal and copula) conditions; wait-cap frontier `W ∈ {6,10,15,20,30}` |
| E10 | Proposed mechanisms, strategic | M6, M7 vs M3, M4, M5 on every lie of E2/E3/E7/E8 at ρ=0.25 and the ρ=1 stress test; karma-supply sensitivity |
| E11 | Proposed mechanisms vs far-sighted attacker | rollout attack, `H ∈ {5,40,80}`, n=10, T=300, against M6 and M7 (Score reference: E3c) |
| E8 | Timing attack (H2) | M4 `λ ∈ {0.5,1,2,5}`, `c ∈ {1.25,2}`; inflate only when own history is favourable (`a_i ≤` mean / 25th / 75th percentile) vs always; paired premium |

Outputs per experiment `eX`: `summary.json` (means + CIs, one row per cell), `paired.json`, `config.json`
(configuration + master seeds), and `raw/<cell>.json` (every per-seed row).

## Results

All results: n=50, k=10, T=1000, 30 master seeds, M4 at λ=1 unless stated (E3b: n=10, T=500). Full tables with CIs:
[`results/RESULTS.md`](results/RESULTS.md). Figures: `figures/`.

### E1 — scarcity (truthful)
- **Report-blind rules pay in welfare, value-aware rules pay in starvation.** At k/n=0.2 Random/Round-Robin reach
  J_A≈0.996/1.000 but only WR≈0.56 (PoF 0.44; 0.53→0.84 across k/n=0.1→0.8). Greedy and Vickrey are first-best
  (WR=1) at J_A=0.996. Score gives up ≤0.8% of welfare (WR 0.992 at k/n=0.1, 0.996 at 0.2) and has the highest J_A
  among value-aware rules (0.9996). Under i.i.d. values *fairness in counts* is not the problem: even Greedy has
  J_A≥0.99.
- **Waiting.** Only Round-Robin bounds waiting: `Q_max=⌈n/k⌉−1` (9 at k/n=0.1, 4 at 0.2), SR_Δ=0. Every other rule
  starves users at a similar rate (k/n=0.1: Greedy Q_max 81.7 / SR_Δ 10.8%, Random 85.5 / 10.7%, Score 67.1 / 7.7%;
  k/n=0.2: 8.4% vs 7.5% for Greedy vs Score). Starvation comes from the absence of a service guarantee, not from greed.

### E2 — λ sweep (ρ=0.25, capped exaggeration c=2)

| λ | WR | J_A | SR_Δ | Coalition M | Unilateral M_uni [95% CI] | frac. of focal users who gain |
|---|---|---|---|---|---|---|
| 0 (≡ Greedy) | 0.893 | 0.506 | 0.270 | +220.1 | +305.7 [303.7, 307.4] | 1.00 |
| 0.25 | 0.953 | 0.898 | 0.104 | +59.4 | +86.8 [86.0, 87.5] | 1.00 |
| 0.5 | 0.959 | 0.967 | 0.082 | +19.7 | +33.7 [32.8, 34.6] | 1.00 |
| **1** | 0.960 | 0.991 | 0.069 | −3.2 | **+3.5 [2.7, 4.3]** | 0.82 |
| **2** | 0.957 | 0.998 | 0.060 | −15.3 | **−12.5 [−13.1, −11.8]** | 0.00 |
| 5 | 0.949 | 0.9995 | 0.045 | −22.2 | −21.3 [−21.7, −20.9] | 0.00 |
| M5 Vickrey | 0.893 | 0.506 | 0.270 | −117.4 | −101.8 [−102.6, −101.1] | 0.00 |

- Greedy is extremely manipulable (+306 per inflating user). The history penalty *raises* welfare under manipulation
  (0.893→0.96) while pushing J_A to ≈1, and the individual gain shrinks monotonically.
- **λ=1 does not make inflation unprofitable for an individual:** the coalition gain is slightly negative (−3.2) only
  because the 12 inflaters crowd each other out; one user inflating alone still gains +3.5 (82% of focal users
  gain). The unilateral gain turns negative between λ=1 and λ=2.
- **Caveat:** this threshold is specific to c=2 — see E7 for milder lies and E3c for a far-sighted attacker.
- Pareto-non-dominated settings: M2, M5, and M4 at λ∈{1, 2, 5} (M4 λ=1 is not dominated because it has the best WR
  among settings with J_A>0.99).

### E3 — population × attack
- **Round-Robin and Random are unaffected** by every attack (all gains and PoS exactly 0).
- **Greedy: inflating always pays individually** (`M_uni` +99 … +418, 100% of focal users gain), even when everyone
  already inflates (+149 cap_2, +99 max_claim at ρ=1). Max-claim at ρ≥0.25 drops WR to the random level (0.561;
  PoS 0.44) and J_A to 0.24.
- **Vickrey: inflating never pays** (`M_uni` from −50 to −416; 0% gain) — consistent with per-round DSIC. Its welfare
  collapse under max-claim therefore requires users to act against their own interest: a stress test, not an equilibrium.
- **Score (λ=1):** max-claim is unprofitable at every ρ (−47 … −54). Capped exaggeration pays a little when few
  others inflate (+5.5 at ρ=0, +4.5 at 0.1, +3.5 at 0.25), is indistinguishable from 0 at ρ=0.5 (+0.41 [−0.22, 1.03]) and
  negative at ρ=1 (−2.7). Welfare degrades gracefully (cap_2: 0.996→0.838; max_claim: 0.996→0.560) and J_A stays ≥0.99.

### E3b — historical rollout attack (n=10; gain of one focal user)

| Mechanism | opponents truthful: rollout / cap_2 | opponents cap_2: rollout / cap_2 | focal reports v_max in |
|---|---|---|---|
| M3 Greedy | **+164.8** / +123.8 | **+98.0** / +73.1 | 100% of rounds |
| M4 λ=1 | −18.3 / +0.9 | −22.2 / −1.8 | 91–95% |
| M4 λ=2 | −23.3 / −6.2 | −24.2 / −6.8 | 86–92% |
| M5 Vickrey | −0.2 / −41.6 | −0.0 / −24.0 | 0–6% |

With the proposal's 5-round horizon the search finds a *stronger* attack than capped exaggeration on Greedy,
rediscovers truth-telling on Vickrey (gain ≈0, report ≈ value), and finds **no profitable attack on Score**.
That last result does not hold up: it is a horizon artefact (E3c below). Five rounds see the immediate win from
reporting v_max but not the history penalty that persists afterwards, so the attacker over-inflates and loses.
It is a finite search heuristic, not a best response, and never proves anything is unmanipulable.

### E3c — historical rollout attack versus planning horizon (n=10, M4, opponents truthful)

| H | λ=1: rollout | λ=2: rollout | frac. rounds at v_max (λ=1 / 2) |
|---|---|---|---|
| 5 | −9.6 [−10.3, −8.9] | −12.5 [−13.2, −11.8] | 0.93 / 0.88 |
| 10 | −7.0 | −8.6 | 0.85 / 0.74 |
| 20 | −2.9 | −2.7 | 0.72 / 0.53 |
| 40 | **+3.0 [+2.4, +3.6]** | **+3.2 [+2.8, +3.6]** | 0.52 / 0.29 |
| 80 | **+7.2 [+6.8, +7.6]** | **+3.2 [+2.8, +3.7]** | 0.31 / 0.19 |
| *reference: capped c=2 / c=1.25* | *+0.9 / +3.9* | *−3.5 / +1.3* | |

The gain rises monotonically with the horizon and turns significantly positive once the attacker can see the
penalty's cost (H≥40). At λ=2 it plateaus near +3.2 and beats both capped rules, including c=1.25; at λ=1 it is still
rising at H=80 (+7.2, above both capped rules). The mean report converges to the true value (−0.03 at λ=2, H=80):
the attack is *selective* — it inflates in a few well-chosen rounds rather than every round. So λ=2 stops the simple lie
but not a planner. This is a lower bound on what a sophisticated attacker can do (it is a heuristic search at n=10).

### E4 — heterogeneous users (Beta(2,5) / Beta(5,2) halves)

| | WR | J_A | J_B | SR_Δ |
|---|---|---|---|---|
| Random | 0.581 | 0.996 | 0.995 | 0.085 |
| Round-Robin | 0.581 | 1.000 | 0.999 | 0.000 |
| Greedy = Vickrey | 1.000 | 0.507 | 0.517 | 0.480 |
| Score (λ=1) | 0.897 | 0.925 | 0.995 | 0.098 |

Greedy/Vickrey give almost every GPU to the high-value group, so both fairness indices collapse to ≈0.51. Score keeps
benefit fairness at the Random/Round-Robin level (J_B=0.995) while recovering most welfare (0.897 vs 0.581); it serves
the high-value group somewhat more (J_A=0.925), i.e. it trades equal *service* for near-equal *normalised benefit*.

### E5 — persistent demand
- With the **proposal's recursion** `v = αv′ + (1−α)ε`, Random/Round-Robin welfare *rises* with α (0.56→0.85 at α=0.9).
  This is **not** persistence: the recursion also shrinks the value spread by `√((1−α)/(1+α))`, making all users look alike.
  With the **marginal-preserving copula** the same two rules stay flat at 0.56. (An earlier explanation of this rise
  as "last round's random winners are still high-value" was wrong; round-robin and random ignore values.)
- Greedy/Vickrey keep WR=1 but monopolise: J_A 0.996→0.957, `Q_max` 41.7→204.7 (210 under the copula), SR_Δ 8.4%→49.2% (49.8%).
- Score keeps J_A≥0.998 (0.9955 under the copula) and WR≈0.97, and starves less than Greedy (SR_Δ 37.7% proposal, 44.5%
  copula, vs 49%) but far more than Random (8.5%) or Round-Robin (0%): cumulative-allocation penalties equalise
  *totals*, not *waiting times*. The proposal-AR(1) series understates the persistence effect on Score.

### E6 — scalability
Time per round grows gently from ≈20–55 µs (n=10) to ≈40–110 µs (n=500); the slowest rule at n=500 is Score/Greedy
(≈110 µs). Measured peak heap (T=1000, including the seed package and per-round records) grows linearly from 0.5 MiB
(n=10) to 11.7 MiB (n=500) and is identical across mechanisms to within 0.1%. Timings are machine-dependent
(best of 3 after a warm-up, one core).

### E7 — λ × exaggeration-factor grid (ρ=0.25; unilateral gain, M4 unless noted)

| | c=1.25 | c=1.5 | c=2 |
|---|---|---|---|
| M3 Greedy | +126.7 | +208.7 | +305.7 |
| M4 λ=0.5 | +41.1 | +45.7 | +33.7 |
| M4 λ=1 | +21.0 | +17.5 | +3.5 |
| M4 λ=2 | +9.0 | +2.0 | **−12.5** |
| M4 λ=5 | +1.1 (77% of users gain) | **−7.2** | −21.3 |
| M5 Vickrey | −14.0 | −41.8 | −101.8 |

The penalty needed to deter a lie *grows as the lie gets milder*: λ=2 suffices for c=2, λ∈(2,5] for c=1.5, and at
c=1.25 even λ=5 still leaves a (small) positive gain for 77% of focal users. Vickrey deters every lie at every size.
This is the main qualification of the "Score with λ≈2 is a non-monetary alternative to Vickrey" conclusion.

### E8 — does timing reports pay? (hypothesis H2)

The timed policy inflates (capped, factor c) only while the user's own cumulative allocation is at or below the
population mean (`timed`), 25th percentile (`timed_q25`) or 75th percentile (`timed_q75`); otherwise it is truthful.

| M4 | c | always | timed | timed_q25 | timed_q75 |
|---|---|---|---|---|---|
| λ=1 | 1.25 | +21.0 | +0.7 | +0.2 | +2.6 |
| λ=2 | 1.25 | +9.0 | −0.5 | −0.7 | +0.4 |
| λ=1 | 2 | +3.5 | −2.8 | −2.4 | −5.0 |
| λ=2 | 2 | −12.5 | −5.4 | −4.5 | −10.3 |
| λ=5 | 2 | −21.3 | −9.1 | −8.0 | −16.5 |

Timing never beats indiscriminate inflation where inflation is profitable (premium −3 to −39), and its own gain is
≤ +5.6 for M4 across the grid above (about 0 or negative for λ≥2). The premium becomes positive only at c=2, λ≥2, i.e. only
because always-inflating has become expensive and timing means *lying less*. **H2 in its simple form is not
supported**: the value of this kind of timing does not rise with λ. The far-sighted rollout attack (E3c) shows the
vulnerability is real but needs planning, not a fixed rule.

## Proposed mechanisms: M6 Karma-Cap and M7 Rank-Cap

### Why, and how they work
The analysis above shows three gaps in the baselines: a cumulative-allocation penalty (Score) does not bound waiting, only
partly deters lying (milder lies pay, a far-sighted attacker gains), and Vickrey needs money. The target is: **a hard waiting bound, plus
lying that is costly in future priority, plus value-aware selection.** Both mechanisms share a *wait cap*: any user whose
consecutive wait reaches `W` (default `W = Δ = 2⌈n/k⌉`) is served first, oldest first, at most `k` per round.

- **M6 Karma-Cap.** Each user holds a karma balance (mean `karma_init = 2`, ceiling 3×). A report becomes the bid
  `min(report, balance)`; the top bids win a uniform-price auction (winners pay the `(k+1)`-st highest bid in karma), the
  wait cap overrides, and all karma paid is redistributed equally to everyone. Karma is internal and non-tradable, and the
  mechanism returns **zero payments**: welfare and utility are in value only. Exaggerating wins rounds you value less than
  the price, which drains the karma you need for the rounds you value most.
- **M7 Rank-Cap.** The score uses the *quantile* of the report within the user's own past reports instead of the raw value,
  `s = q_i / (1 + a_i)^λ`, plus the wait cap. A fixed strictly increasing transform of one's entire report sequence
  leaves quantiles unchanged. Clipping can raise some quantiles and lower others. Users with
  different value scales are treated alike. (Idea: linking decisions, Jackson & Sonnenschein.)

**Parameter selection** used five development seeds (101–105, disjoint from the 30 locked seeds, enforced by a test): the
karma supply was the largest with non-positive individual gain on every lie of the development grid; `W` was left at Δ.
All numbers below are on the locked seeds. E11 (the far-sighted attacker) was *not* used in tuning. E10 includes the
sensitivity to the supply (`b0 = 1, 5`).

### E9 — truthful reports (30 seeds; n=50, k=10 unless stated)

| condition | | WR | J_B | Q_max | SR_Δ |
|---|---|---|---|---|---|
| base | M3 Greedy | 1.000 | 0.996 | 41.7 | 0.084 |
| | M4 Score λ=1 | 0.996 | 0.999 | 37.7 | 0.075 |
| | **M6 Karma-Cap** | 0.942 | 1.000 | **10.0** | **0.000** |
| | **M7 Rank-Cap** | 0.945 | 0.999 | **10.0** | **0.000** |
| scarce (k/n=0.1) | M4 Score λ=1 | 0.992 | 0.999 | 67.1 | 0.077 |
| | M6 / M7 | 0.933 / 0.930 | 0.999 | 20.4 / 20.0 | 0 |
| mixed users (E4) | M4 Score λ=1 | 0.897 | 0.995 | 49.8 | 0.098 |
| | M6 / M7 | 0.722 / 0.804 | 0.940 / 0.971 | 11.9 / 10.0 | 0 |
| persistent α=0.9, copula | M4 Score λ=1 | 0.969 | 0.994 | 165.7 | 0.445 |
| | M6 / M7 | 0.752 / 0.813 | 0.998 / 0.995 | 11.1 / 10.0 | 0 |

**The wait cap has a welfare cost.** Truthful welfare as `W` is swept (observed `Q_max` equals `W` in these cells):

| W | 6 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|
| M6 Karma-Cap WR | 0.817 | 0.942 | 0.969 | 0.973 | 0.974 |
| M7 Rank-Cap WR | 0.813 | 0.945 | 0.981 | 0.990 | 0.993 |

For comparison Score has `Q_max≈38` at WR=0.996, Greedy `Q_max≈42`, Round-Robin `Q_max=4` at WR=0.56
(figure `e9_wait_frontier`). M7 with W=15–20 gives WR 0.98–0.99 with observed maximum waits of 15–20 rounds;
its general guarantee is `W + ceil(n/k) - 1`, against ≈38 rounds observed for Score. Under heterogeneous or strongly persistent demand the cap forces service to low-value
users, so the welfare cost is much larger (0.72–0.81 vs 0.90–0.97 for Score): the guarantee is expensive exactly when values differ.

### E10 — individual gain from lying (ρ=0.25; fraction of focal users who gain in brackets)

| | cap 1.25 | cap 1.5 | cap 2 | max claim | timed 1.25 | timed 2 | ρ=1 cap 2 | ρ=1 max |
|---|---|---|---|---|---|---|---|---|
| M3 Greedy | +126.7 (1.00) | +208.7 | +305.7 | +417.6 | +7.5 | +6.7 | +148.5 | +99.0 |
| M4 Score λ=1 | +21.0 (1.00) | +17.5 (1.00) | +3.5 (0.82) | −53.6 | +0.7 (0.57) | −2.8 | −2.7 | −52.6 |
| M4 Score λ=2 | +9.0 (1.00) | +2.0 (0.78) | −12.5 | −65.4 | −0.5 | −5.4 | −13.7 | −63.3 |
| M5 Vickrey | −14.0 | −41.8 | −101.8 | −416.1 | −0.9 | −2.3 | −49.6 | −99.2 |
| **M6 Karma-Cap** | **−8.3 (0.01)** | −17.1 | −29.1 | −69.0 | −2.3 (0.25) | −14.0 | **+46.1 (1.00)** | **+6.1 (0.89)** |
| **M7 Rank-Cap** | **+1.8 (0.77)** | −10.6 | −35.9 | −113.9 | −1.3 (0.35) | −22.9 | −34.1 | −141.8 |

- At ρ=0.25 **Karma-Cap has negative mean individual gain for every tested scripted lie, including mild c=1.25 and timed lies.**
  Some focal users still gain. Rank-Cap has negative mean gain for all but the mildest lie (c=1.25, +1.8; versus +21.0 for Score λ=1).
- **Karma-Cap's deterrence reverses when everyone else already lies** (ρ=1: +46.1, every focal user gains; max claim +6.1).
  Truthful reporting is therefore not a robust equilibrium of Karma-Cap, a standard worry for karma economies. Score,
  Vickrey and Rank-Cap stay negative at ρ=1. E17 traces the reversal to auction benefit and reduced budget binding;
  it also occurs without the cap.
- **Supply matters:** with `b0 = 5` timed lies pay again (+2.3, +0.9); `b0 = 1` is more deterring and less efficient.
- Welfare under attack (cap 2, ρ=0.25): Karma 0.903, Rank 0.921, Score 0.960, Greedy/Vickrey 0.893.

### E11 — historical far-sighted attacker (n=10, T=300; gain of one focal user)

| H | M6 Karma-Cap | M7 Rank-Cap | Score λ=1 (E3c) | Score λ=2 (E3c) |
|---|---|---|---|---|
| 5 | −7.05 | −20.94 | −9.61 | −12.47 |
| 40 | −2.32 | −11.85 | **+2.97** | **+3.19** |
| 80 | −2.17 | −4.41 | **+7.20** | **+3.24** |

In the historical E11 results, neither proposed mechanism has positive mean attack gain, whereas Score is attacked from H=40. Karma-Cap's
loss is flat (≈ −2.2); **Rank-Cap's loss is shrinking with the horizon (−20.9 → −11.9 → −4.4), so a longer horizon could
turn it positive, as it did for Score.** In the rollout Rank-Cap's quantile functions are frozen at the current round (the
attacker cannot reshape its own history within a rollout), which favours Rank-Cap; Karma-Cap's state is simulated exactly.

### Historical scoreboard (criteria fixed before the original locked-seed runs)

| criterion | Score λ=1 | M6 Karma-Cap | M7 Rank-Cap |
|---|---|---|---|
| 1. welfare near Greedy (truthful, homogeneous) | **0.996** | 0.942 (−5.8%) | 0.945 (−5.5%); 0.98–0.99 at W=15–20 |
| 2. hard waiting bound / observed zero starvation | ✗ (Q_max≈38) | **✓** (≤ W+⌈n/k⌉−1) | **✓** (≤ W+⌈n/k⌉−1) |
| 3. non-positive mean gain from every scripted lie at ρ=0.25 | ✗ (+21.0 at c=1.25) | **✓** (all means ≤ 0) | ✗ (+1.8 at c=1.25) |
| 4. no profit for a far-sighted attacker | ✗ (+3 … +7) | **✓** (≈ −2.2, flat) | ✓ so far (−4.4, rising) |
| 5. welfare under heterogeneous / persistent demand | **0.90 / 0.97** | ✗ 0.72 / 0.75 | ✗ 0.80 / 0.81 |
| 6. (extra) deterrence when everyone else lies (ρ=1) | ✓ | **✗ (+46)** | ✓ |

The fresh E12-E15 evaluation qualifies this scoreboard. Score-Cap provides the same certified service bound, so the
cap alone cannot explain Karma-Cap's lower scripted gains. Rank-Cap is profitably attacked by the corrected planner;
Karma-Cap's mean scripted gain reverses when everyone lies. The constrained oracle also reveals additional welfare
loss beyond the service requirement. Mechanism choice therefore depends on the workload and strategic policies;
none of these capped rules establishes dynamic strategyproofness.

### Research questions and hypotheses
- **Q1 (history penalty vs starvation / welfare).** Under truthful i.i.d. values, a moderate penalty costs <1% welfare
  but reduces starvation only slightly (7.5% vs 8.4%; 37.7% vs 49.2% under strong persistence). Its large effect appears when users are strategic
  (SR_Δ 27%→6.9% at λ=0→1) or heterogeneous (J_B 0.52→0.995). **H1 supported for fairness, weak for starvation.**
- **Q2 (does the penalty change the value of manipulating?).** Yes — it removes most of it (+306→+3.5 at λ=1 against c=2),
  but how much penalty is needed depends on the lie (E7: λ=2 for c=2, more for milder lies) and on the attacker: a
  far-sighted rollout attacker still gains at λ=2 (E3c), while simple timing does not (E8). **H2 (timing becomes more
  valuable with λ) is not supported for the fixed timing rules tested**; it is not refuted for planning attackers.
- **Q3 (do equal-count conclusions survive heterogeneity / persistence?).** No for the value-aware rules (E4, E5).
  **H3 is only partly supported:** Round-Robin scores best on J_A and, perhaps surprisingly, *also* on J_B (0.999):
  value-blind rules equalise expected normalised benefit as well, so J_A and J_B diverge only for rules that read values (Greedy/Vickrey/Score).
- **H4.** M5 has a negative unilateral gain in every tested setting (supported); M3 is vulnerable (supported); M4 is
  vulnerable at λ=1, to mild lies at λ≥2, and to a far-sighted attacker even at λ=2 (supported, more strongly than the
  proposal expected).

### Conclusions
1. The extremes each fail on one axis: Round-Robin kills starvation and manipulation but loses 44% of welfare at k/n=0.2;
   Greedy is first-best and the most manipulable (+306), and concentrates service under heterogeneous or persistent values.
2. A history penalty buys fairness almost for free (Score λ=1: WR≥0.99 truthful, 0.96 against 25% inflaters, 0.90 heterogeneous).
3. Strategy-resistance needs a stronger penalty than fairness does, and the required strength depends on the attack:
   λ≈2 against c=2, more against milder lies, and no fixed λ tested here stops a far-sighted attacker.
4. A cumulative-allocation penalty does not bound waiting. Round-Robin and the oldest-first cap supply explicit guarantees.
5. Vickrey has non-positive gain for the tested deviations, but needs real transfers. Score with λ≈2
   deters c=2 exaggeration while retaining ≈96% welfare; milder lies and planning still pay in the tested settings.
   These are empirical results for bounded policies and a heuristic search, **not** truthfulness guarantees
   (simulation cannot establish dynamic strategyproofness).
6. A hard wait cap plus an incentive component (karma, or rank normalisation) reduces observed starvation and much of the mean profit from
   lying, at a welfare price that is small for homogeneous demand (≈5%) and large for heterogeneous or persistent demand.
   The two mechanisms fail differently (Karma-Cap when others lie, Rank-Cap for mild lies and corrected history-aware
   planning), so the choice is a design trade-off. The general wait bound includes overflow beyond W.

## Proposal coverage

| Proposal item | Where | Verified by |
|---|---|---|
| Model: unit demand, `k<n`, quasi-linear utility, bounded reports, seeded ties | `sim/config.py`, `sim/environment.py`, `sim/mechanisms/_utils.py` | `test_config`, `test_environment`, `test_mechanisms` |
| M1–M5 incl. payments `p=x·v̂_{(k+1)}` | `sim/mechanisms/` | E0 (exhaustive), `test_mechanisms` |
| Policies: truthful, capped (c∈{1.25,1.5,2}), max claim, **rollout (n=10, H=5, 100 MC, grid)** | `sim/policies/strategic.py`, `sim/runner.py::run_rollout`, `experiments/e3b_rollout.py` | `test_policies`, `test_runner`, `test_experiments` |
| WR, J_A, J_B, NSW, Q_max, p95 wait, SR_Δ, PoF, PoS | `sim/metrics.py` | hand-computed cases (`test_metrics*`), vectorised-vs-naive check (E0) |
| Paired manipulation gain `M̄, M_max, frac_pos` (coalition **and** unilateral per-user `M_i`) | `sim/metrics.py`, `sim/runner.py::unilateral_gains` | `test_runner`, `test_metrics_extra` |
| Paired counterfactual design on one ω | `SeedPackage`, `run_paired` | `test_environment`, `test_runner` |
| Pareto-dominance over (WR, J_A, SR_Δ, M) | `analysis/pareto.py`, E2 `pareto` flag | `test_analysis` |
| 30 locked seeds, paired 95% bootstrap CIs (10,000 resamples), raw per-seed results, config files, seed list | `seeds/`, `analysis/bootstrap.py`, `experiments/common.py` | `test_analysis`, `test_experiments` |
| E0 exhaustive validation | `experiments/e0_validation.py` | `test_experiments` (incl. fault-injection tests) |
| E1–E4 (required), E5, E6 (stretch), rollout (stretch) | `experiments/` (+ E3c horizon study, E7 λ×c grid, E8 timing) | `test_experiments`, `test_plots_and_pipeline` |
| One command regenerates every main figure | `run_all.py`, `run_all.sh` | `test_run_all_quick_pipeline_end_to_end` |
| *Extension beyond the proposal:* two new mechanisms and their evaluation | `sim/mechanisms/m6_karma.py`, `m7_rank.py`, E9–E11 | `test_proposed_mechanisms`, `test_proposed_experiments` |
| Hypotheses H1–H4 evaluated, including contradicted ones | "Research questions and hypotheses" above | — |
| Anticipated limitations | "Limitations" below | — |

## What changed in the overhaul

Defects fixed and additions beyond the first version (all results regenerated):
- **Tie-break bug (M4).** Winners were chosen after adding a 1e-12 jitter to scores, which distorts the ordering when
  scores are tiny (λ=5 with ~200 prior allocations gives scores ≈3e-12). Ties are now broken by a secondary sort key.
- **Waiting times** now follow the proposal (state *before* round `t`); metrics are vectorised and cross-checked against a naive loop.
- **Unilateral gain** is measured for several strategic users per seed (`M_uni`, `M_uni_max`, `frac_pos_uni`), not a single focal user.
- **Rollout attack** was implemented but never run, and simulated the focal user's future values as constant. It now
  samples future values, is batched with common random numbers, and is run as E3b.
- **E0** is exhaustive (it previously sampled random profiles) and includes negative controls and fault-injection tests.
- **E6 memory** is measured with `tracemalloc` (it was an analytic array-size estimate); timing is best-of-3.
- **E5** exposes the variance-shrinkage confound of the proposal's AR(1) recursion and adds a marginal-preserving control.
- **New mechanisms M6 Karma-Cap and M7 Rank-Cap** with a shared wait cap, a rollout interface that handles stateful
  mechanisms, and experiments E9–E11 (development seeds for tuning, locked seeds for evaluation).
- **E7** (λ × c sensitivity grid), **E3c** (rollout horizon) and **E8** (timing attack for H2) are new; E2–E4/E6 now save raw per-seed results and configs; paired CIs are produced.
- Deterministic mechanism order in E3 (it iterated over a `set`), strict JSON output (no `NaN`), validated configs,
  a mechanism factory replacing five duplicated copies, and 463 tests (previously 63).

## Related work

A literature search was run before designing M6/M7 (three parallel searches; claims below are from abstracts or search
summaries unless stated, and have not been checked against the full papers).

The course report makes the model distinction explicit: published Karma credits
donated/borrowed capacity; our internal uniform-price auction is a different
mechanism and does not inherit its online strategy-proofness guarantees.

- **Fair GPU scheduling.** Tiresias (NSDI'19; attained-service priority, the shape of our Score rule), Themis (NSDI'20;
  finish-time fairness, a partial-allocation auction among the most-behind users with "hidden payments" paid in withheld
  resources; truthfulness proved per round for homogeneous valuations), Gavel (OSDI'20; priority = target/received share,
  an effective bounded-waiting rule; strategy-proofness left to future work), Shockwave (NSDI'23), Gandiva_fair, AlloX,
  HiveD, OEF (Middleware'24; strategy-proofness by equalising outcomes, with an efficiency-vs-strategyproofness
  impossibility). Strategic misreporting is rarely treated, and none addresses cross-round incentives.
- **Dynamic fairness and incentives.** Fikioris, Agarwal & Tardos (arXiv 2109.12401; checked against the abstract):
  history-aware dynamic max-min / DRF is *not* incentive compatible, but is (1+ρ)-IC and, for one resource, 3/2-IC (√2
  lower bound), which matches our finding that a history penalty only partially deters lying. Karma (Vuppalapati et al.,
  OSDI'23; checked against the abstract): credit-based allocation under dynamic demands with claimed strategy-proofness.
- **Non-monetary repeated mechanism design.** Artificial currencies (Gorokh, Banerjee & Iyer, Math. OR 2021; Elokda et al.
  karma games), linking decisions (Jackson & Sonnenschein, Econometrica 2007, with the 2022 Ball–Jackson–Kattwinkel
  comment), storable votes (Casella 2005), promised utility (Balseiro, Gurkan & Sun, Oper. Res. 2019; Blanchard & Jaillet
  2024). The initial search did not identify an exact truthful, efficient, fair mechanism with a service guarantee
  for this private-value unit-demand model. This is a search limitation rather than an impossibility theorem.
- **Bounded-lag schedulers.** Deficit Round Robin (Shreedhar & Varghese 1995), lottery and stride scheduling (Waldspurger
  & Weihl 1994/95): the origin of our wait cap; no incentive analysis.

M6 adapts the artificial-currency idea (a bid costs future priority) and M7 the linking idea (reports are meaningful only
relative to one's own distribution); the hard wait cap is the Deficit-Round-Robin/Gavel idea. We did not find a paper combining
all three for unit-demand GPU allocation, but the search was not exhaustive and we make no novelty claim beyond that.

## Limitations
The model omits job duration, multi-GPU jobs, placement and preemption; findings concern repeated unit-demand
allocation, not a production cluster. Synthetic values and utility-equivalent payments limit external validity.
Jain's index is one notion of equality, so waiting and normalised-benefit metrics are reported alongside it. The attack
policies include fixed rules, a finite-horizon search heuristic, and an independently trained finite-policy game; replacing M5's payments by
reusable credits would need a dynamic budget model and would void its DSIC guarantee. In the rollout attack the focal
user reports truthfully after the first simulated round and futures are drawn i.i.d., even in the AR(1) setting.
E6 timings depend on the machine.
For M6/M7: the hyperparameters were chosen on development seeds against the same *families* of lies that E10 then
evaluates (E11 was not used), and the rollout attack is run only at n=10. Historical Rank-Cap attacks froze quantile
functions; E15 updates them and finds positive gains. Adaptive training uses five traces and eight policies;
E18 extends the budget to twenty sweeps but Score-Cap and Rank-Cap still have training incentives.
Karma-Cap's ρ=1 reversal is diagnosed for cap_2 in E17, rather than characterised for every strategic population;
and `W = Δ` is a convention, not an optimum (the frontier shows how much welfare it costs).

## Reproducibility & seed policy
- The 30 master seeds are fixed in `seeds/master_seeds.json` (regenerable bit-for-bit by `seeds/generate_seeds.py`; a test checks it).
- Each seed derives independent valuation, tie-breaking and strategic-set streams via `SeedSequence.spawn`; strategic
  sets are nested in ρ. Mechanisms never mutate the seed package, and stateful mechanisms (M2) are deep-copied per run.
- Re-running `python run_all.py --extensions-only` reproduces the fresh extension summaries, comparisons, figures and report
  (solver certification is recorded). The full pipeline also reruns legacy attacks with corrected code; their checkpoint
  numbers are historical, rather than bit-for-bit targets of the new attacker. E6 timings are machine-dependent.
- Exact dependency versions: `requirements-lock.txt`.

## License
[MIT](LICENSE).
