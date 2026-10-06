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
│   ├── e5_persistence.py          # E5 (Stretch): AR(1) temporal persistence sweep
│   └── e6_scalability.py          # E6 (Stretch): Wall-clock & memory scaling benchmark
├── analysis/                      # Statistical analysis and plotting
│   ├── __init__.py
│   ├── bootstrap.py               # Vectorized 95% paired bootstrap CI (10,000 resamples)
│   ├── pareto.py                  # Multi-objective Pareto-dominance filter
│   └── plots.py                   # Academic-grade figures (PDF + PNG)
├── tests/                         # Full automated test suite (63 unit tests)
│   ├── __init__.py
│   ├── test_environment.py       # Determinism, bounds, AR(1), History tracking
│   ├── test_mechanisms.py        # Capacity, payments, invariance, wait bounds, DSIC
│   ├── test_metrics.py           # Hand-computed 2-user, 2-round validation cases
│   ├── test_runner.py            # Paired/unilateral runners, NaN-safe summaries
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

The test suite contains 63 automated tests covering environment determinism, mechanism invariants, metric calculations on hand-computed examples, paired/unilateral runners, and strategic policy bounds:

```bash
pytest
```

All 63 tests pass in under 1 second.

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

### Running Experiments E1–E6 Individually

| Experiment | Description | Command | Output |
|---|---|---|---|
| **E1** | Resource Scarcity Sweep ($k/n \in \{0.1, 0.2, 0.4, 0.6, 0.8\}$) | `python experiments/e1_scarcity.py` | `results/e1/summary.json` |
| **E2** | Fairness-Vulnerability Frontier ($\lambda \in \{0 \dots 5.0\}$ for M4) | `python experiments/e2_frontier.py` | `results/e2/summary.json` |
| **E3** | Strategic Factorial Design ($\rho \in \{0 \dots 1.0\} \times 3$ policies) | `python experiments/e3_strategic.py` | `results/e3/summary.json` |
| **E4** | Heterogeneous Users ($\text{Beta}(2,5)$ vs $\text{Beta}(5,2)$) | `python experiments/e4_heterogeneous.py` | `results/e4/summary.json` |
| **E5** | Temporal Persistence ($\alpha \in \{0.0, 0.5, 0.9\}$ AR(1)) *(Stretch)* | `python experiments/e5_persistence.py` | `results/e5/summary.json` |
| **E6** | Scalability Benchmark ($n \in \{10 \dots 500\}$) *(Stretch)* | `python experiments/e6_scalability.py` | `results/e6/summary.json` |

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

This section synthesizes the final findings from the simulation pipeline. The evaluation compares five mechanisms (M1: Random, M2: Round-Robin, M3: Greedy, M4: Score, M5: Vickrey) across allocative efficiency (Welfare Ratio, WR), long-run fairness (Jain's indices $J_A, J_B$), waiting time ($Q_{\max}$, $\text{SR}_\Delta$), and empirical resistance to manipulation ($M$).

### E1: Resource Scarcity
This experiment varied the capacity ratio $k/n$ from $0.1$ to $0.8$ under truthful reporting.
- **Welfare vs. Fairness:** Report-invariant mechanisms (Random, Round-Robin) achieve perfect or near-perfect allocation fairness ($J_A \approx 0.99 - 1.0$) but suffer significantly in social welfare ($WR \approx 0.53 - 0.71$, depending on scarcity). Value-aware mechanisms (Greedy, Score, Vickrey) achieve the first-best truthful welfare ($WR = 1.0$).
- **Starvation bounds:** Round-Robin strictly bounds the maximum consecutive wait ($Q_{\max} = \lceil n/k \rceil$). For instance, at $k/n = 0.1$, $Q_{\max} = 9.0$ and Starvation Rate ($\text{SR}_\Delta$) is $0.0$. In contrast, Greedy allocation leads to severe starvation ($Q_{\max} \approx 81.7$, $\text{SR}_\Delta \approx 10.8\%$) under severe scarcity.

### E2: Fairness and Strategic-Vulnerability Frontier (The $\lambda$ Sweep)
We swept the history penalty parameter $\lambda \in \{0, 0.05, 0.1, \dots, 5.0\}$ for the Score mechanism (M4) to observe the tradeoff between fairness, welfare, and manipulation gain ($M$).
- **Vulnerability of Greedy:** At $\lambda = 0$ (equivalent to Greedy), users gain massively by inflating their reports ($M \approx 220$), making the system highly vulnerable to manipulation.
- **The "Sweet Spot" ($\lambda \approx 1.0$):** As $\lambda$ increases, the history penalty reduces the benefit of strategic inflation. At $\lambda = 1.0$, the manipulation gain becomes strictly negative ($M \approx -3.41$). This means **strategic inflation actively hurts the user**. 
- **Welfare Retention:** Remarkably, at $\lambda = 1.0$, social welfare remains exceptionally high ($WR \approx 0.960$), and allocation fairness approaches perfection ($J_A \approx 0.991$). This confirms **Hypotheses 1 and 2**: a moderate history penalty drastically improves fairness and disincentivizes manipulation without a catastrophic loss of true welfare.

### E3: Strategic Population and Attack Type
This experiment scaled the fraction of strategic users $\rho$ and tested various bounded attacks (Truthful, Capped Exaggeration, Maximum Claim).
- **Degradation of Welfare:** When users employ the `max_claim` strategy, mechanisms relying on reported values degrade. Even the robust Vickrey mechanism degrades to a random allocation in terms of social welfare ($WR \approx 0.561$) when everyone max-claims, because prices become uniformly high and allocation becomes a tie-breaker.
- **Report-Invariant Stability:** Random and Round-Robin maintain a stable $WR \approx 0.560$ regardless of the attack type or the proportion of strategic users ($\rho$), confirming their total immunity to report inflation.

### E4: Heterogeneous Users
We evaluated a two-group population with differing value distributions (e.g., $\text{Beta}(2,5)$ vs. $\text{Beta}(5,2)$).
- **Service vs. Benefit:** Round-Robin maintains perfect allocation fairness ($J_A = 1.0$) and near-perfect normalized benefit fairness ($J_B \approx 0.999$). 
- **Efficiency Bias:** Greedy, Score, and Vickrey mechanisms naturally allocate more GPUs to the group with the higher value distribution to maximize overall social welfare. Consequently, their fairness scores drop substantially in mixed populations ($J_A \approx 0.506, J_B \approx 0.516$). This addresses **Hypothesis 3**: when values are heterogeneous, equal allocation does not organically arise from value-aware welfare maximization.

### E5: Temporally Persistent Demand (AR(1) Process)
When user valuations are persistent across rounds (using an AR(1) process with $\alpha \in \{0.0, 0.5, 0.9\}$) rather than strictly i.i.d.:
- High-value users hold on to their high values longer. As persistence $\alpha$ increases to $0.9$, the welfare achieved by Random and Round-Robin artificially rises ($WR \approx 0.843$).
- Value-based mechanisms (Greedy, Score, Vickrey) continue to successfully track the highest true values and consistently achieve optimal welfare ($WR = 1.0$) regardless of temporal persistence.
- Allocation fairness ($J_A$) for value-based mechanisms drops slightly as persistence increases (from $0.996$ at $\alpha=0.0$ to $0.957$ at $\alpha=0.9$), as heavy-hitters monopolize the GPUs for longer consecutive streaks.

### E6: Scalability Benchmark
The Python simulator exhibits highly efficient scalability:
- Simulating a population of $n=500$ users takes only $\approx 57.3\mu\text{s}$ per round for the most complex mechanism (Vickrey).
- The state footprint is extremely minimal, requiring only $\approx 11.7\text{ MiB}$ of RAM at $n=500$.

### 🎯 Final Conclusions

1. **The Limitations of Extremes:** Strict equality mechanisms (Round-Robin) eliminate starvation and manipulation but sacrifice ~40-45% of potential system value. Conversely, pure Greedy allocation achieves optimal value but induces severe starvation and is highly vulnerable to priority inflation.
2. **The Success of History Penalties (Score Mechanism):** The results provide strong evidence for the practical viability of the **History-Penalised Score Mechanism** ($s_{i,t} = \hat{v}_{i,t} / (1 + a_i(t))^\lambda$). Setting $\lambda \approx 1.0$:
   - Achieves near-optimal efficiency ($>95\%$ WR).
   - Eliminates starvation effectively.
   - Provides an empirical, non-monetary defense against priority inflation (manipulation yields negative utility).
3. **No Need for Real Money:** While the Vickrey auction provides rigorous dominant-strategy truthfulness, the History-Penalised Score mechanism offers a highly competitive proxy for fairness and efficiency without requiring a complex, real-money intertemporal payment infrastructure.

---

## 📜 Reproducibility & Seed Policy

- Master seeds are pre-generated by `seeds/generate_seeds.py` and locked in `seeds/master_seeds.json`.
- The simulation enforces a **paired-randomness contract**: every mechanism within a seed trial experiences the exact same valuation realizations $v[n, T]$ and seeded tie-breaking streams.
- All statistical comparisons report paired bootstrap 95% confidence intervals based on 10,000 resamples (`analysis/bootstrap.py`).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
