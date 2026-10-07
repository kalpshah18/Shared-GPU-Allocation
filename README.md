# Fairness, Efficiency, and Strategic Behaviour in Shared GPU Allocation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A simulation study comparing five repeated GPU-allocation mechanisms on allocative efficiency (welfare ratio WR),
> long-run fairness (Jain indices J_A, J_B; waiting and starvation), and resistance to bounded strategic
> misreporting (paired manipulation gain M, price of strategy). The full design is in
> [`FAI_Project_Proposal.pdf`](FAI_Project_Proposal.pdf); every item it asks for is implemented, tested and
> traceable in the [coverage table](#proposal-coverage) below.

---

## Repository structure

```
sim/                         Pure-Python simulator
  config.py                  Validated Config dataclass (+ YAML loading)
  environment.py             Seeded valuations (i.i.d. / Beta / two AR(1) variants), History, strict-JSON store
  runner.py                  Round loop: mixed populations, paired runs, unilateral gains, rollout runner
  metrics.py                 WR, J_A, J_B, NSW, Q_max, p95 wait, SR_Δ, PoF, PoS, coalition & unilateral M
  mechanisms/                M1 Random · M2 Round-Robin · M3 Greedy · M4 Score · M5 Vickrey (+ factory)
  policies/strategic.py      Truthful · capped exaggeration · maximum claim · rollout attack (batched)
experiments/
  e0_validation.py           Exhaustive checks on small discrete instances (67k instances)
  e1_scarcity.py             k/n sweep                       e5_persistence.py   AR(1) persistence (+ sensitivity)
  e2_frontier.py             λ sweep, Pareto frontier        e6_scalability.py   time per round, measured heap
  e3_strategic.py            ρ × policy factorial            e7_sensitivity.py   exaggeration factor c
  e3b_rollout.py             finite-horizon rollout attack   e4_heterogeneous.py equal service vs equal benefit
  common.py                  Seeds, CLI, raw/summary persistence, strategic-cell measurement
analysis/
  bootstrap.py               95% bootstrap CIs (10,000 resamples) and paired-difference CIs
  pareto.py                  Multi-objective dominance filter
  plots.py                   All figures (PDF + PNG)             report.py   results/RESULTS.md tables
tests/                       329 tests (see "Tests")
config/base.yaml             Base configuration (n=50, k=10, T=1000, ρ=0.25, …)
seeds/                       generate_seeds.py + the 30 locked master seeds
results/  figures/           Generated outputs  (results/RESULTS.md = every table)
run_all.py / run_all.sh      One command: validate → run everything → regenerate every figure
```

## Setup

The project is developed and tested in a dedicated virtual environment (Python 3.13; 3.10+ works).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell:  .venv\Scripts\Activate.ps1   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # exact versions used for the reported results: requirements-lock.txt
```

## Quick start

```bash
python -m pytest                  # 329 tests (~2.5 min; add -m "not slow" to skip the end-to-end pipeline test)
python experiments/e0_validation.py
python run_all.py                 # everything, from the locked seeds → results/ and figures/  (tens of minutes)
python run_all.py --quick         # 2-seed smoke run on tiny instances, for development only
python analysis/report.py         # regenerate results/RESULTS.md from results/*/summary.json
```

Every experiment script accepts `--seeds N`, `--results-dir DIR`, `--n`, `--T`, `--bootstrap`.

## Tests

`python -m pytest` runs 329 tests (`-m "not slow"` skips the one end-to-end pipeline test). They are organised by component:

| File | What it pins down |
|---|---|
| `test_config.py` | validation of every field, derived Δ, YAML loading, `base.yaml == defaults` |
| `test_environment.py` | determinism, paired randomness, nested strategic sets, bounds, marginal means, AR(1) autocorrelation / variance (both variants), History bookkeeping, strict JSON |
| `test_mechanisms.py` | capacity/payment invariants for every mechanism and size, report-invariance, uniform tie-breaking, scale-free ties (regression for the jitter bug), M2 wait bound, M3 oracle, M4 formula and monotonicity, M5 payments and **exhaustive DSIC**, negative controls, batch ≡ single allocation, determinism |
| `test_metrics.py`, `test_metrics_extra.py` | hand-computed 2-user cases, pre-round waiting semantics, Jain-index properties, J_B, NSW, payments excluded from welfare, coalition / unilateral / PoS / PoF |
| `test_policies.py` | elementwise bounded policies, registry, rollout attack (inflates under Greedy, exactly truthful under Vickrey, deterministic, side-effect free, scales with `v_max`) |
| `test_runner.py` | mixed / paired / unilateral runners vs manual counterfactuals, stateful-mechanism reset, report validation, rollout runner |
| `test_analysis.py` | bootstrap CI coverage, pairing benefit, NaN handling, paired differences, Pareto filter vs brute force |
| `test_experiments.py` | every experiment end to end on tiny instances: schema, strict JSON, raw files, determinism, and the qualitative findings (Greedy rewarded / Vickrey punished, E5 variance confound, E7 monotonicity, …); E0 **fault injection** (a broken Vickrey / capacity bug must be caught) |
| `test_plots_and_pipeline.py` | every figure is written, graceful skips, report tables, locked seed file, `run_all.py --quick` end to end |

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
| E4 | Heterogeneous users | uniform vs `Beta(2,5)` / `Beta(5,2)` halves; J_A vs J_B |
| E5 | Persistence (stretch) | `α ∈ {0,0.5,0.9}`; proposal AR(1) **and** a marginal-preserving copula variant |
| E6 | Scalability (stretch) | `n ∈ {10,…,500}`, `k/n=0.2`; µs per round, **measured** (`tracemalloc`) peak heap |
| E7 | Sensitivity | `c ∈ {1.25,1.5,2}` for M3, M4(λ=1,2), M5 |

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

### E3b — rollout attack (n=10; gain of one focal user)

| Mechanism | opponents truthful: rollout / cap_2 | opponents cap_2: rollout / cap_2 | focal reports v_max in |
|---|---|---|---|
| M3 Greedy | **+164.8** / +123.8 | **+98.0** / +73.1 | 100% of rounds |
| M4 λ=1 | −18.3 / +0.9 | −22.2 / −1.8 | 91–95% |
| M4 λ=2 | −23.3 / −6.2 | −24.2 / −6.8 | 86–92% |
| M5 Vickrey | −0.2 / −41.6 | −0.0 / −24.0 | 0–6% |

The finite-horizon search finds a *stronger* attack than capped exaggeration on Greedy, rediscovers truth-telling
on Vickrey (gain ≈0, report ≈ value), and — notably — **does not find a profitable attack on Score**: a five-round
horizon sees the immediate win from reporting v_max but not the penalty that persists afterwards, so it over-inflates and loses.
This is a finite search heuristic, not a best response; the proposal's caveat applies (it does not show Score is unmanipulable).

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

### E7 — sensitivity to the exaggeration factor (ρ=0.25; unilateral gain)

| c | M3 | M4 λ=1 | M4 λ=2 | M5 |
|---|---|---|---|---|
| 1.25 | +126.7 | +21.0 | +9.0 | −14.0 |
| 1.5 | +208.8 | +17.5 | +2.0 | −41.8 |
| 2 | +305.7 | +3.5 | **−12.5** | −101.8 |

The λ=2 "deterrence" found in E2 depends on the attack: against milder exaggeration (c=1.25, 1.5) an individual
still profits at λ=2 (+9.0, +2.0), and at λ=1 every focal user gains for c≤1.5 (frac=1.00). Mild lies are the more
dangerous ones for Score; Vickrey punishes all of them.

### Research questions and hypotheses
- **Q1 (history penalty vs starvation / welfare).** Under truthful i.i.d. values, a moderate penalty costs <1% welfare
  but reduces starvation only slightly (7.5% vs 8.4%; 37.7% vs 49.2% under strong persistence). Its large effect appears when users are strategic
  (SR_Δ 27%→6.9% at λ=0→1) or heterogeneous (J_B 0.52→0.995). **H1 supported for fairness, weak for starvation.**
- **Q2 (does the penalty change the value of manipulating?).** Yes — it removes most of it (+306→+3.5 at λ=1), but
  only a stronger penalty makes inflation unprofitable, and only against c=2 (E7). **H2 as stated (penalty may *increase*
  the value of timing reports) is not supported against the tested policies**; the rollout search finds no profitable attack on Score either.
- **Q3 (do equal-count conclusions survive heterogeneity / persistence?).** No for the value-aware rules (E4, E5).
  **H3 is only partly supported:** Round-Robin scores best on J_A and, perhaps surprisingly, *also* on J_B (0.999):
  value-blind rules equalise expected normalised benefit as well, so J_A and J_B diverge only for rules that read values (Greedy/Vickrey/Score).
- **H4.** M5 has a negative unilateral gain in every tested setting (supported); M3 is vulnerable (supported); M4 is
  vulnerable at λ=1 and for mild lies at λ=2, but not to c=2 at λ≥2 (partly supported).

### Conclusions
1. The extremes each fail on one axis: Round-Robin kills starvation and manipulation but loses 44% of welfare at k/n=0.2;
   Greedy is first-best and the most manipulable (+306), and concentrates service under heterogeneous or persistent values.
2. A history penalty buys fairness almost for free (Score λ=1: WR≥0.99 truthful, 0.96 against 25% inflaters, 0.90 heterogeneous).
3. Strategy-resistance needs a stronger penalty than fairness does, and depends on the attack (λ≈2 against c=2 only).
4. A cumulative-allocation penalty does not bound waiting; only Round-Robin's explicit service guarantee does.
5. Vickrey is the only rule with a negative individual gain in every setting, but needs real transfers; Score with λ≈2 is a
   non-monetary alternative that deters the tested policies while retaining ≈96% welfare — an empirical result for these
   bounded policies, **not** a truthfulness guarantee (simulation cannot establish dynamic strategyproofness).

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
| E1–E4 (required), E5, E6 (stretch), rollout (stretch) | `experiments/` | `test_experiments`, `test_plots_and_pipeline` |
| One command regenerates every main figure | `run_all.py`, `run_all.sh` | `test_run_all_quick_pipeline_end_to_end` |
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
- **E7** (sensitivity to `c`) is new; E2–E4/E6 now save raw per-seed results and configs; paired CIs are produced.
- Deterministic mechanism order in E3 (it iterated over a `set`), strict JSON output (no `NaN`), validated configs,
  a mechanism factory replacing five duplicated copies, and 329 tests (previously 63).

## Limitations
The model omits job duration, multi-GPU jobs, placement and preemption; findings concern repeated unit-demand
allocation, not a production cluster. Synthetic values and utility-equivalent payments limit external validity.
Jain's index is one notion of equality, so waiting and normalised-benefit metrics are reported alongside it. The attack
policies are fixed rules plus a finite-horizon search heuristic, not an equilibrium analysis; replacing M5's payments by
reusable credits would need a dynamic budget model and would void its DSIC guarantee. In the rollout attack the focal
user reports truthfully after the first simulated round and futures are drawn i.i.d., even in the AR(1) setting.
E6 timings depend on the machine.

## Reproducibility & seed policy
- The 30 master seeds are fixed in `seeds/master_seeds.json` (regenerable bit-for-bit by `seeds/generate_seeds.py`; a test checks it).
- Each seed derives independent valuation, tie-breaking and strategic-set streams via `SeedSequence.spawn`; strategic
  sets are nested in ρ. Mechanisms never mutate the seed package, and stateful mechanisms (M2) are deep-copied per run.
- Re-running `python run_all.py` reproduces all `summary.json`, `paired.json`, and figures exactly (E6 timings excepted).
- Exact dependency versions: `requirements-lock.txt`.

## License
[MIT](LICENSE).
