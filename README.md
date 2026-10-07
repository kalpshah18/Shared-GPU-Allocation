# Fairness, Efficiency, and Strategic Behaviour in Shared GPU Allocation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

> A simulation study comparing five dynamic GPU allocation mechanisms across allocative efficiency (Welfare Ratio), long-run fairness (Jain's indices $J_A, J_B$), and resistance to strategic manipulation (manipulation gain $M$).


---

## 📁 Repository Structure

```
FAI_Project/
├── sim/                           # Pure-Python simulation engine
│   ├── __init__.py
│   ├── config.py                  # Dataclass for all hyperparameters & YAML loading
│   ├── environment.py             # Valuations (Uniform/Beta/AR(1)), History, SeedPackage
│   ├── runner.py                  # Single, paired, and mixed simulation runners
│   ├── metrics.py                 # All metrics (WR, J_A, J_B, Q_max, SR_Δ, M, PoF, PoS)
│   ├── mechanisms/                # Mechanism implementations
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract Mechanism interface
│   │   ├── _utils.py              # Seeded tie-breaking and rank utilities
│   │   ├── m1_random.py           # M1: Uniform Random allocation
│   │   ├── m2_roundrobin.py       # M2: Round-Robin queue
│   │   ├── m3_greedy.py           # M3: Greedy reported value (first-best truthful oracle)
│   │   ├── m4_score.py            # M4: History-penalised score s = v̂ / (1 + a)^λ
│   │   └── m5_vickrey.py          # M5: k-unit Vickrey auction with (k+1)-th price
│   └── policies/                  # Strategic reporting policies
│       ├── __init__.py
│       └── strategic.py           # Truthful, capped exaggeration, max claim, rollout attack (rollout implemented, not run)
├── experiments/                   # Experiment scripts
│   ├── __init__.py
│   ├── e0_validation.py           # E0: Invariant and DSIC sanity checks
│   ├── e1_scarcity.py             # E1: Resource scarcity sweep k/n ∈ {0.1..0.8}
│   ├── e2_frontier.py             # E2: Fairness-efficiency-strategy frontier (λ sweep)
│   ├── e3_strategic.py            # E3: Strategic population factorial (ρ × policy)
│   ├── e4_heterogeneous.py        # E4: Heterogeneous users (equal service vs benefit)
│   ├── e2b_cap_sensitivity.py     # E2b: λ sweep for exaggeration c ∈ {1.25, 1.5, 2}
│   ├── e5_persistence.py          # E5 (Stretch): AR(1) temporal persistence sweep
│   ├── e6_scalability.py          # E6 (Stretch): Wall-clock & memory scaling benchmark
│   └── e7_paired.py               # E7: Paired mechanism comparisons with bootstrap CIs
├── analysis/                      # Statistical analysis and plotting
│   ├── __init__.py
│   ├── bootstrap.py               # Vectorized 95% paired bootstrap CI (10,000 resamples)
│   ├── pareto.py                  # Multi-objective Pareto-dominance filter
│   └── plots.py                   # Academic-grade figures (PDF + PNG)
├── tests/                         # Full automated test suite (69 unit tests)
│   ├── __init__.py
│   ├── test_environment.py       # Determinism, bounds, AR(1), History tracking
│   ├── test_mechanisms.py        # Capacity, payments, invariance, wait bounds, DSIC
│   ├── test_metrics.py           # Hand-computed 2-user, 2-round validation cases
│   ├── test_runner.py            # Paired/unilateral runners, NaN-safe summaries
│   ├── test_analysis.py          # Pareto filter and paired bootstrap
│   └── test_policies.py          # Reporting bounds, identities, capped scaling
├── config/
│   └── base.yaml                  # Default benchmark parameters (n=50, k=10, T=1000)
├── seeds/
│   ├── generate_seeds.py          # Script to generate locked master seeds
│   └── master_seeds.json          # 30 locked master seeds
├── figures/                       # Generated publication plots
├── results/                       # Generated raw and aggregated JSON results
├── pytest.ini                     # Pytest configuration
├── requirements.txt               # Pinned Python package dependencies
├── run_all.sh                     # Bash master pipeline script
├── run_all.py                     # Cross-platform Python master pipeline script
└── README.md
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kalpshah18/Shared-GPU-Allocation
   cd Shared-GPU-Allocation
   ```

2. **Create a virtual environment (Python 3.10+ recommended):**
   ```bash
   python -m venv venv
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧪 Running the Test Suite

The test suite contains 69 automated tests covering environment determinism, mechanism invariants, metric calculations on hand-computed examples, paired/unilateral runners, and strategic policy bounds:

```bash
pytest
```

All 69 tests pass in under 1 second.

---

## 🔬 Experimental Pipeline

### Phase 2: Correctness Invariants (E0 Validation)
Before any main experiments run, the E0 validation suite verifies foundational theoretical invariants:
- **Capacity invariant**: $\sum_i x_{i,t} = k$ for all $t$ across all mechanisms.
- **Payment invariant**: $p_{i,t} = 0$ whenever $x_{i,t} = 0$.
- **Report-invariance**: M1 and M2 produce identical allocations regardless of reports.
- **M3 $\equiv$ M4 at $\lambda = 0$**: Score mechanism matches Greedy allocation exactly.
- **Round-robin wait bound**: No user goes more than $\lceil n/k \rceil - 1$ consecutive rounds without service (checked every round).
- **Welfare oracle**: M3 with truthful reports achieves first-best social welfare $W^*$.
- **M5 one-round DSIC**: Dominant-strategy incentive compatibility verified over a fine discrete grid.

Run E0 checks directly:
```bash
python experiments/e0_validation.py
```

### Running Experiments E1–E7 Individually

| Experiment | Description | Command | Output |
|---|---|---|---|
| **E1** | Resource Scarcity Sweep ($k/n \in \{0.1, 0.2, 0.4, 0.6, 0.8\}$) | `python experiments/e1_scarcity.py` | `results/e1/summary.json` |
| **E2** | Fairness-Vulnerability Frontier ($\lambda \in \{0 \dots 5.0\}$ for M4), truthful sweep, Pareto set | `python experiments/e2_frontier.py` | `results/e2/{summary,truthful_sweep,pareto}.json` |
| **E2b** | $\lambda$ sweep × exaggeration $c \in \{1.25, 1.5, 2\}$ | `python experiments/e2b_cap_sensitivity.py` | `results/e2/cap_sensitivity.json` |
| **E3** | Strategic Factorial Design ($\rho \in \{0 \dots 1.0\} \times 5$ policies) | `python experiments/e3_strategic.py` | `results/e3/summary.json` |
| **E4** | Heterogeneous Users ($\text{Beta}(2,5)$ vs $\text{Beta}(5,2)$) | `python experiments/e4_heterogeneous.py` | `results/e4/summary.json` |
| **E5** | Temporal Persistence ($\alpha \in \{0.0, 0.5, 0.9\}$ AR(1)) *(Stretch)* | `python experiments/e5_persistence.py` | `results/e5/summary.json` |
| **E6** | Scalability Benchmark ($n \in \{10 \dots 500\}$) *(Stretch)* | `python experiments/e6_scalability.py` | `results/e6/summary.json` |
| **E7** | Paired mechanism comparisons (30 seeds, paired bootstrap) | `python experiments/e7_paired.py` | `results/e7/paired.{json,md}` |

### Generating Figures
After experiments have run, generate all figures in both `.pdf` and `.png`:
```bash
python analysis/plots.py --all
```

### Full End-to-End Pipeline (Single Command)
To run all validation checks, execute experiments, and generate all figures from the locked master seeds:

- **Cross-platform (Windows / macOS / Linux):**
  ```bash
  python run_all.py
  ```
- **Bash (Linux / macOS):**
  ```bash
  bash run_all.sh
  ```

---

## 📊 Summary of Evaluated Mechanisms

| ID | Name | Description | Key Theoretical Property |
|---|---|---|---|
| **M1** | Random | $k$ users sampled uniformly without replacement | Report-invariant, fair in expectation, welfare-oblivious |
| **M2** | Round-Robin | Cyclic service queue of capacity $k$ | Report-invariant, served at least once every $\lceil n/k \rceil$ rounds ($Q_{\max} \le \lceil n/k \rceil - 1$) |
| **M3** | Greedy Reported Value | Allocate to top-$k$ reported valuations ($p=0$) | First-best welfare under truthfulness; vulnerable to inflation |
| **M4** | History-Penalised Score | Score $s_{i,t} = \hat{v}_{i,t} / (1 + a_i(t))^\lambda$ | Continuous trade-off between efficiency and long-run fairness; $\lambda = 0$ is exactly M3. Default $\lambda = 1$ (all experiments except the E2 sweep) |
| **M5** | $k$-unit Vickrey Auction | Top-$k$ bids win, pay $(k+1)$-th highest bid | Per-round DSIC quasi-linear benchmark |

---

## 📈 Evaluation Metrics

All metrics accept raw per-seed output tensors and support paired bootstrap analysis:
- **Welfare Ratio ($WR$)**: $W / W^*$ where $W^*$ is the offline truthful maximum welfare.
- **Jain's Allocation Fairness ($J_A$)**: $J_A = \frac{(\sum_i A_i)^2}{n \sum_i A_i^2}$, measuring dispersion in received GPU slots.
- **Jain's Benefit Fairness ($J_B$)**: Jain's index on normalized utility $B_i = G_i / (T \cdot \mu_i)$, distinguishing equal service from equal benefit.
- **Maximum Consecutive Wait ($Q_{\max}$)**: $\max_{i,t} q_i(t)$.
- **Starvation Rate ($\text{SR}_\Delta$)**: Proportion of user-rounds where consecutive wait exceeds $\Delta = 2\lceil n/k \rceil$.
- **Coalition Manipulation Gain ($M$, $M_{\max}$, frac_pos)**: For each strategic user, utility when the whole strategic set deviates minus utility when everyone is truthful (same $\omega$). Strategic users compete with each other, so $M < 0$ does **not** mean an individual is better off truthful.
- **Unilateral Manipulation Gain ($M_{\text{uni}}$)**: $U_f(\sigma_f, \sigma_{-f};\omega) - U_f(\text{truthful}, \sigma_{-f};\omega)$ for one focal user $f$, all other users' behaviour held fixed. This is the individual incentive to manipulate. At $\rho = 0$ the focal user is a lone deviator in a truthful population.
- **Price of Strategy (PoS)**: Social welfare loss $(W_{\text{truth}} - W_{\text{strat}}) / W_{\text{truth}}$ caused by strategic reporting.
- **Price of Fairness (PoF)**: Welfare sacrifice $(W^* - W_{\text{fair}}) / W^*$ necessary to enforce equitable distribution.

---

## 🔌 How to Add a New Mechanism

To implement and evaluate a new mechanism $M_{\text{new}}$:

1. Create `sim/mechanisms/m_new.py` subclassing `Mechanism`:
   ```python
   from sim.mechanisms.base import Mechanism
   from sim.config import Config
   from sim.environment import History
   import numpy as np

   class NewMechanism(Mechanism):
       def __init__(self, cfg: Config):
           super().__init__(cfg)

       def allocate(
           self,
           reports: np.ndarray,
           history: History,
           tie_seed: int,
       ) -> tuple[np.ndarray, np.ndarray]:
           # 1. Compute allocation x of shape (n,) with exactly k ones
           # 2. Compute non-negative payment p of shape (n,)
           ...
           return x, p
   ```

2. Register the class in `sim/mechanisms/__init__.py`.
3. Add the mechanism to `experiments/e0_validation.py` and run `pytest` to automatically verify capacity invariants, payment bounds, and tie-breaking determinism.

---

## 🏁 Final Project Results

This section synthesizes the findings from the simulation pipeline. All results use $n = 50$, $T = 1000$ and 30 locked master seeds (means; 95% bootstrap CIs are in `results/*/summary.json`). Unless stated otherwise $k = 10$ ($k/n = 0.2$) and **M4 Score uses $\lambda = 1$** (E2 sweeps $\lambda$). Manipulation is reported two ways: the coalition gain $M$ (all strategic users deviate together) and the unilateral gain $M_{\text{uni}}$ (one user deviates, everyone else fixed). Only $M_{\text{uni}}$ measures an individual's incentive to manipulate.

### E1: Resource Scarcity
$k/n$ varied from $0.1$ to $0.8$ under truthful reporting.
- **Welfare vs. fairness:** Random and Round-Robin reach $J_A \approx 0.99$–$1.0$ but only $WR \approx 0.53$–$0.84$, rising with $k/n$ (PoF $0.47 \to 0.16$). Greedy and Vickrey reach first-best $WR = 1.0$. Score ($\lambda = 1$) gives up at most 0.8% ($WR = 0.992$ at $k/n = 0.1$, $0.996$ at $0.2$) and has the highest $J_A$ of the value-aware mechanisms ($0.9996$ vs. $0.996$ at $k/n = 0.2$).
- **Nash social welfare** ($\sum_i \log(G_i + 10^{-8})$, higher is better) follows the welfare ranking: at $k/n = 0.2$ Greedy/Vickrey $259.1$, Score $259.0$, Round-Robin $230.2$, Random $230.1$.
- **Waiting:** Round-Robin bounds the consecutive wait at $Q_{\max} = \lceil n/k \rceil - 1$ ($9$ at $k/n = 0.1$, $4$ at $0.2$) with $\text{SR}_\Delta = 0$. With i.i.d. values every other mechanism starves users at a similar rate: at $k/n = 0.1$, Greedy has $Q_{\max} \approx 81.7$, $\text{SR}_\Delta \approx 10.8\%$, Random $85.5$ / $10.7\%$, and Score $67.1$ / $7.7\%$. Starvation comes from the absence of a service guarantee, not from greed specifically.

### E2: Fairness / Strategic-Vulnerability Frontier ($\lambda$ sweep)
$\lambda \in \{0, 0.05, 0.1, 0.25, 0.5, 1, 2, 5\}$ for M4. The strategic sweep uses $\rho = 0.25$ with capped exaggeration $c = 2$; a truthful sweep and an exaggeration-strength sensitivity sweep (E2b) accompany it. Figures: `e2_frontier`, `e2_truthful_sweep`, `e2_cap_sensitivity`.

**Strategic sweep ($c = 2$)**

| $\lambda$ | WR | $J_A$ | $\text{SR}_\Delta$ | Coalition $M$ | Unilateral $M_{\text{uni}}$ [95% CI] |
|---|---|---|---|---|---|
| 0 (≡ Greedy) | 0.893 | 0.506 | 0.270 | +220.0 | +303.3 [301.0, 305.6] |
| 0.25 | 0.953 | 0.898 | 0.104 | +59.2 | +85.2 |
| 0.5 | 0.959 | 0.967 | 0.082 | +19.6 | +32.7 |
| 1 | 0.960 | 0.991 | 0.070 | −3.4 | +2.5 [1.3, 3.6] |
| 2 | 0.957 | 0.998 | 0.060 | −15.5 | −13.0 [−14.0, −12.0] |
| 5 | 0.946 | 0.999 | 0.047 | −21.7 | −20.9 |

**Exaggeration-strength sensitivity (E2b): unilateral gain $M_{\text{uni}}$ of M4, $\rho = 0.25$**

| $\lambda$ | $c = 1.25$ | $c = 1.5$ | $c = 2$ |
|---|---|---|---|
| 0 | +127.2 | +207.5 | +303.3 |
| 0.5 | +41.0 | +44.9 | +32.7 |
| 1 | +20.5 | +16.9 | +2.5 |
| 2 | +8.5 | +1.4 | −13.0 |
| 3 | +4.2 | −3.6 | −18.4 |
| 5 | +0.5 [0.0, 1.0] | −7.6 | −20.9 |

(All 95% CIs in `results/e2/cap_sensitivity.json`.)

**Truthful sweep (tests H1)**

| $\lambda$ | WR | $\text{SR}_\Delta$ | 95th-pct wait | $Q_{\max}$ |
|---|---|---|---|---|
| 0 | 1.000 | 0.0845 | 12.9 | 41.7 |
| 1 | 0.996 | 0.0748 | 12.1 | 37.7 |
| 2 | 0.992 | 0.0688 | 12.0 | 37.0 |
| 5 | 0.981 | 0.0556 | 11.0 | 32.9 |

**Non-dominated settings** over (WR ↑, $J_A$ ↑, $\text{SR}_\Delta$ ↓, $M_{\text{uni}}$ ↓), saved in `results/e2/pareto.json`: M4 at $\lambda \in \{1, 2, 5\}$, M5 Vickrey and M2 Round-Robin. Random, Greedy and M4 at $\lambda \le 0.5$ are dominated (Random, for example, by M4 at $\lambda = 5$).

- **Greedy is highly manipulable:** at $\lambda = 0$ one inflating user gains $\approx 127$–$303$ depending on $c$.
- **The history penalty shrinks the gain monotonically in $\lambda$ for every $c$.** Under manipulation, welfare *rises* from 0.893 to $\approx 0.96$ as $\lambda$ grows from 0 to 1, because the penalty limits how much inflators can grab, and $J_A$ approaches 1.
- **At $\lambda = 1$ inflating still pays an individual user** (+2.5 to +20.5, CIs exclude 0). The coalition gain is negative ($-3.4$ at $c = 2$) only because the 12 inflating users crowd each other out.
- **Deterrence needs a large $\lambda$ and depends on attack strength.** The gain turns negative at $\lambda \approx 1.5$ for $c = 2$, $\lambda \approx 3$ for $c = 1.5$, and is still marginally positive (+0.5, CI [0.0, 1.0]) at $\lambda = 5$ for $c = 1.25$. Mild exaggeration is the hardest attack to deter because it rarely exhausts the user's allocation history. A recommendation of "$\lambda \approx 2$" would only have held against $c = 2$.
- **H1 (moderate $\lambda$ reduces tail waiting before it costs welfare): partially supported, effect modest.** Under truthful reports, $\lambda = 2$ cuts $\text{SR}_\Delta$ from 8.5% to 6.9% (−19% relative) and $Q_{\max}$ from 41.7 to 37.0 (−11%) for a 0.8 pp welfare loss. Even at $\lambda = 5$ starvation is still 5.6%, far from Round-Robin's 0.
- **H2 (a larger $\lambda$ increases the value of strategic *timing*):** not tested. Only report-inflation policies were run; none withholds reports to manage the allocation history.

### E3: Strategic Population and Attack Type
$\rho \in \{0, 0.1, 0.25, 0.5, 1\}$ × {truthful, capped exaggeration $c \in \{1.25, 1.5, 2\}$, maximum claim}. The diagnostic rollout attack is implemented in `sim/policies/strategic.py` but was **not run**, so no rollout results are reported.
- **Report-invariant stability:** Random ($WR = 0.561$) and Round-Robin ($0.560$) are unaffected by every attack ($M = M_{\text{uni}} = 0$, PoS $= 0$).
- **Greedy:** inflating always pays individually. $M_{\text{uni}}$ ranges from $+99$ to $+418$ across $c = 2$ and max-claim, and from $+115$ to $+152$ at $c = 1.25$. It is still positive when everyone already inflates (e.g. $+149$ at $c = 2$, $\rho = 1$). Max-claim at $\rho \ge 0.25$ drops WR to the random level ($0.561$, PoS $0.44$) with $J_A$ as low as $0.24$.
- **Vickrey:** inflating never pays ($M_{\text{uni}}$: $-13$ to $-17$ at $c = 1.25$, $-32$ to $-48$ at $c = 1.5$, $-50$ to $-101$ at $c = 2$, down to $-414$ for max-claim), consistent with per-round DSIC. Its welfare collapse under max-claim ($WR = 0.561$ at $\rho \ge 0.25$) therefore needs users to act against their own interest. It is a stress test, not a realistic equilibrium.
- **Score ($\lambda = 1$):** **mild exaggeration is the profitable attack**, not strong exaggeration.

  | $\rho$ | $c = 1.25$ | $c = 1.5$ | $c = 2$ | max-claim |
  |---|---|---|---|---|
  | 0 (lone deviator) | +21.1 | +18.0 | +5.0 | −48.4 |
  | 0.25 | +20.5 | +16.9 | +2.5 | −52.7 |
  | 1.0 | +19.6 | +13.4 | −2.8 | −54.1 |

  $M_{\text{uni}}$ for M4 at $\lambda = 1$ (all CIs in `results/e3/summary.json`). Max-claim is unprofitable at every $\rho$ because it burns through allocation history. The welfare cost of these attacks is small: with 25% of users using $c = 1.25$, $WR = 0.993$ and $J_A = 0.995$; at $c = 2$, $WR = 0.960$. Greedy under the same $c = 1.25$ attack drops $J_A$ to $0.878$.

### E4: Heterogeneous Users
Half the users draw values from $\text{Beta}(2,5)$ (mean $2/7$) and half from $\text{Beta}(5,2)$ (mean $5/7$); truthful reporting.

| Mechanism | WR | $J_A$ | $J_B$ | $\text{SR}_\Delta$ |
|---|---|---|---|---|
| Random | 0.581 | 0.996 | 0.995 | 0.085 |
| Round-Robin | 0.581 | 1.000 | 0.999 | 0.000 |
| Greedy / Vickrey | 1.000 | 0.507 | 0.517 | 0.481 |
| Score ($\lambda = 1$) | 0.897 | 0.925 | 0.995 | 0.098 |

- **Nash social welfare** (higher is better) separates the mechanisms most clearly here: Score $245.4$ [245.4, 245.5], Round-Robin $225.2$, Random $225.1$, Greedy/Vickrey $142.1$ [135.3, 148.5] (the low-value group is nearly starved, so their $\log G_i$ terms are very negative). Score is the only mechanism that beats the report-invariant baselines on this fairness-weighted welfare measure.
- Greedy and Vickrey give almost all GPUs to the high-value group, so both fairness indices collapse ($\approx 0.51$). This addresses **Hypothesis 3**: with heterogeneous values, welfare maximisation does not produce equal service.
- Score keeps benefit fairness at the Random level ($J_B = 0.995$) while recovering most of the welfare (0.897 vs. 0.581). It still serves the high-value group somewhat more ($J_A = 0.925$), so it trades equal service for near-equal *normalised benefit*.

### E5: Temporally Persistent Demand (AR(1))
$\alpha \in \{0.0, 0.5, 0.9\}$; truthful reporting.
- Random and Round-Robin welfare rises with persistence ($WR = 0.561 \to 0.843$ at $\alpha = 0.9$), because last round's random winners are likely still high-value.
- Greedy and Vickrey stay at $WR = 1.0$, but persistent high-value users monopolise GPUs. $J_A$ falls from $0.996$ to $0.957$, $Q_{\max}$ rises from $41.7$ to $206.4$, and $\text{SR}_\Delta$ from $8.4\%$ to $49.4\%$.
- Score keeps $J_A \ge 0.998$ and $WR = 0.969$ at $\alpha = 0.9$ and starves less than Greedy ($Q_{\max} = 130$, $\text{SR}_\Delta = 37.7\%$). It still starves far more than Random ($8.5\%$) or Round-Robin ($0\%$). A cumulative-allocation penalty equalises *totals*, not *waiting times*.

### E6: Scalability Benchmark
- Per-round runtime grows slowly from $n = 10$ to $n = 500$ ($\approx 5$–$62\,\mu\text{s}$). The slowest case is Vickrey at $n = 500$ ($\approx 62\,\mu\text{s}$ per round). Times are machine-dependent.
- Peak memory is measured with `tracemalloc` in a separate run from the timing (seed-package generation plus the full $T = 1000$ simulation): $\approx 0.47$ MiB at $n = 10$ and $\approx 11.7$ MiB at $n = 500$, essentially identical across mechanisms because state is dominated by the $n \times T$ valuation and history arrays.

### E7: Paired Mechanism Comparisons
Mean per-seed differences A − B with 95% paired bootstrap CIs over the 30 locked seeds (`results/e7/paired.md`, `paired.json`). Selected results:
- **Base case, truthful:** Score($\lambda = 1$) − Greedy: WR $-0.0040$ [$-0.0040$, $-0.0039$], $J_A$ $+0.0035$, $\text{SR}_\Delta$ $-0.0097$, $Q_{\max}$ $-3.9$ [$-5.7$, $-2.2$]. All CIs exclude 0: the welfare cost of the penalty is tiny but statistically unambiguous, as are its fairness gains. Score($\lambda = 1$) − Round-Robin: WR $+0.436$, $\text{SR}_\Delta$ $+0.075$, $Q_{\max}$ $+33.7$.
- **Heterogeneous groups, truthful:** Score($\lambda = 1$) − Greedy: WR $-0.104$, $J_A$ $+0.419$, $J_B$ $+0.479$, $\text{SR}_\Delta$ $-0.383$, NSW $+103$ [97, 110]. Score($\lambda = 1$) − Random: $J_B$ $+0.00002$ [$-0.0004$, $+0.0004$], i.e. no detectable difference in benefit fairness, with WR $+0.316$.
- **Strategic, $c = 2$, $\rho = 0.25$:** Score($\lambda = 1$) − Greedy: $M_{\text{uni}}$ $-300.9$ [$-303$, $-298$], WR $+0.067$. Score($\lambda = 2$) − Score($\lambda = 1$): $M_{\text{uni}}$ $-15.5$ [$-16.0$, $-14.9$] for WR $-0.0025$. Score is **more** manipulable than Vickrey ($M_{\text{uni}}$ $+103.3$ [101, 105] at $\lambda = 1$).

### 🎯 Final Conclusions

1. **The extremes each fail on one axis.** Round-Robin eliminates starvation and manipulation but sacrifices 16–47% of attainable welfare (44% at the base $k/n = 0.2$). Greedy achieves first-best welfare under truth-telling but is the most manipulable mechanism (individual gains of $+115$ to $+418$), and under persistent or heterogeneous values it starves and concentrates service.
2. **A history penalty buys fairness cheaply.** Score at $\lambda = 1$ keeps $WR \ge 0.99$ under truthful reports in every i.i.d. setting (paired welfare cost vs. Greedy: $-0.004$ at the base case), $\ge 0.98$ against 25% mild exaggerators, and $0.90$ with heterogeneous groups, with $J_A \ge 0.99$ in homogeneous populations and $J_B \approx 0.995$ in heterogeneous ones. In the heterogeneous population it has the highest Nash social welfare of all five mechanisms (245 vs. 225 for Random and 142 for Greedy).
3. **Manipulation resistance needs a much larger $\lambda$ than fairness does, and depends on the attack.** At $\lambda = 1$ a lone user still gains from inflating (+2.5 at $c = 2$, +17 to +21 at $c = 1.25$–$1.5$). Against $c = 2$ the gain turns negative at $\lambda \approx 1.5$; against $c = 1.5$ at $\lambda \approx 3$; against $c = 1.25$ it is still $\approx 0$ at $\lambda = 5$, where truthful welfare has fallen to 0.981. Maximum-claim is unprofitable already at $\lambda = 1$. Mild exaggeration, not blatant inflation, is the attack a history penalty handles worst.
4. **Starvation is not solved by a penalty on cumulative allocations.** Under truthful reports $\text{SR}_\Delta$ falls only from 8.5% ($\lambda = 0$) to 6.9% ($\lambda = 2$) and 5.6% ($\lambda = 5$); under strong persistence Score still starves 37.7% of user-rounds ($49.4\%$ for Greedy). Only Round-Robin's explicit service guarantee gives $\text{SR}_\Delta = 0$. A wait-time-aware penalty would be needed to bound starvation.
5. **Payments vs. no payments.** Vickrey is the only mechanism whose unilateral manipulation gain is negative in every tested setting, but it needs real transfers. Score is *more* manipulable than Vickrey at every tested $\lambda$ against capped exaggeration (e.g. $+103$ at $\lambda = 1$, $c = 2$). Raising $\lambda$ trades welfare for deterrence; it does not reach Vickrey's guarantee. These are empirical results for bounded policies, not truthfulness proofs.
6. **Hypotheses.** H1 partially supported (modest starvation reduction for small welfare loss). H3 partially supported: Round-Robin wins on $J_A$ and loses on WR, but the conjecture that it need not dominate on $J_B$ is not borne out here: it still has the highest $J_B$ ($0.999$ vs. $0.995$ for Random and Score) in the Beta(2,5)/Beta(5,2) population. H4 is mixed: Vickrey has no positive gain, as predicted, and Greedy is vulnerable, but Score's vulnerability varies strongly with $c$ and $\lambda$. H2 (timing attacks under larger $\lambda$) is untested.

---

## 📜 Reproducibility & Seed Policy

- Master seeds are pre-generated by `seeds/generate_seeds.py` and locked in `seeds/master_seeds.json`.
- The simulation enforces a **paired-randomness contract**: every mechanism within a seed trial experiences the exact same valuation realizations $v[n, T]$ and seeded tie-breaking streams.
- All statistical comparisons report paired bootstrap 95% confidence intervals based on 10,000 resamples (`analysis/bootstrap.py`).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
