# Fairness, Efficiency, and Strategic Behaviour in Shared GPU Allocation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

> A simulation study comparing five dynamic GPU allocation mechanisms across allocative efficiency (Welfare Ratio), long-run fairness (Jain's indices $J_A, J_B$), and resistance to strategic manipulation (manipulation gain $M$).

---

## 👥 Authors & IIT Gandhinagar Team

| Team Member | Primary Role & Component Ownership |
|---|---|
| **Aayush Kuloor** | Simulator Core Engine, valuation processes (Uniform, Beta, AR(1)), seeded RNG streams, `Config` system, runner, reproducibility |
| **Harsh Dhru** | Mechanisms M1–M5, allocation & payment protocols, seeded tie-breaking, E0 validation suite |
| **Raj Modi** | Strategic reporting policies (truthful, capped exaggeration, max claim, rollout attack), evaluation metrics (WR, $J_A, J_B$, $Q_{\max}$, $\text{SR}_\Delta$, PoF, PoS, $M$) |
| **Kalp Shah** | Experimental parameter sweeps (E1–E6), paired bootstrap confidence intervals (10,000 resamples), Pareto-dominance frontier, figure plotting |

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
│       └── strategic.py           # Truthful, capped exaggeration, max claim, rollout attack
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
├── tests/                         # Full automated test suite (55 unit tests)
│   ├── __init__.py
│   ├── test_environment.py       # Determinism, bounds, AR(1), History tracking
│   ├── test_mechanisms.py        # Capacity, payments, invariance, wait bounds, DSIC
│   ├── test_metrics.py           # Hand-computed 2-user, 2-round validation cases
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
   git clone <repo-url>
   cd FAI_Project
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

The test suite contains 55 automated tests covering environment determinism, mechanism invariants, metric calculations on hand-computed examples, and strategic policy bounds:

```bash
pytest
```

All 55 tests pass in under 1 second.

---

## 🔬 Experimental Pipeline

### Phase 2: Correctness Invariants (E0 Validation)
Before any main experiments run, the E0 validation suite verifies foundational theoretical invariants:
- **Capacity invariant**: $\sum_i x_{i,t} = k$ for all $t$ across all mechanisms.
- **Payment invariant**: $p_{i,t} = 0$ whenever $x_{i,t} = 0$.
- **Report-invariance**: M1 and M2 produce identical allocations regardless of reports.
- **M3 $\equiv$ M4 at $\lambda = 0$**: Score mechanism matches Greedy allocation exactly.
- **Round-robin wait bound**: No user waits longer than $\lceil n/k \rceil$ rounds.
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
| **M2** | Round-Robin | Cyclic service queue of capacity $k$ | Report-invariant, worst-case wait bounded by $\lceil n/k \rceil$ |
| **M3** | Greedy Reported Value | Allocate to top-$k$ reported valuations ($p=0$) | First-best welfare under truthfulness; vulnerable to inflation |
| **M4** | History-Penalised Score | Score $s_{i,t} = \hat{v}_{i,t} / (1 + a_i(t))^\lambda$ | Continuous trade-off between efficiency and long-run fairness |
| **M5** | $k$-unit Vickrey Auction | Top-$k$ bids win, pay $(k+1)$-th highest bid | Per-round DSIC quasi-linear benchmark |

---

## 📈 Evaluation Metrics

All metrics accept raw per-seed output tensors and support paired bootstrap analysis:
- **Welfare Ratio ($WR$)**: $W / W^*$ where $W^*$ is the offline truthful maximum welfare.
- **Jain's Allocation Fairness ($J_A$)**: $J_A = \frac{(\sum_i A_i)^2}{n \sum_i A_i^2}$, measuring dispersion in received GPU slots.
- **Jain's Benefit Fairness ($J_B$)**: Jain's index on normalized utility $B_i = G_i / (T \cdot \mu_i)$, distinguishing equal service from equal benefit.
- **Maximum Consecutive Wait ($Q_{\max}$)**: $\max_{i,t} q_i(t)$.
- **Starvation Rate ($\text{SR}_\Delta$)**: Proportion of user-rounds where consecutive wait exceeds $\Delta = 2\lceil n/k \rceil$.
- **Manipulation Gain ($M, M_{\max}$)**: Ex-post benefit difference for strategic users under paired random seeds $\omega$.
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

## 📜 Reproducibility & Seed Policy

- Master seeds are pre-generated by `seeds/generate_seeds.py` and locked in `seeds/master_seeds.json`.
- The simulation enforces a **paired-randomness contract**: every mechanism within a seed trial experiences the exact same valuation realizations $v[n, T]$ and seeded tie-breaking streams.
- All statistical comparisons report paired bootstrap 95% confidence intervals based on 10,000 resamples (`analysis/bootstrap.py`).
