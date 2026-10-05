# Final Project Results: Fairness, Efficiency, and Strategic Behaviour in Shared GPU Allocation

This document synthesizes the final findings from the simulation pipeline, addressing the research questions posed in the project proposal. The evaluation compares five mechanisms (M1: Random, M2: Round-Robin, M3: Greedy, M4: Score, M5: Vickrey) across allocative efficiency (Welfare Ratio, WR), long-run fairness (Jain's indices $J_A, J_B$), waiting time ($Q_{\max}$, $\text{SR}_\Delta$), and empirical resistance to manipulation ($M$).

## E1: Resource Scarcity
This experiment varied the capacity ratio $k/n$ from $0.1$ to $0.8$ under truthful reporting.
- **Welfare vs. Fairness:** Report-invariant mechanisms (Random, Round-Robin) achieve perfect or near-perfect allocation fairness ($J_A \approx 0.99 - 1.0$) but suffer significantly in social welfare ($WR \approx 0.53 - 0.71$, depending on scarcity). Value-aware mechanisms (Greedy, Score, Vickrey) achieve the first-best truthful welfare ($WR = 1.0$).
- **Starvation bounds:** Round-Robin strictly bounds the maximum consecutive wait ($Q_{\max} = \lceil n/k \rceil$). For instance, at $k/n = 0.1$, $Q_{\max} = 9.0$ and Starvation Rate ($\text{SR}_\Delta$) is $0.0$. In contrast, Greedy allocation leads to severe starvation ($Q_{\max} \approx 81.7$, $\text{SR}_\Delta \approx 10.8\%$) under severe scarcity.

## E2: Fairness and Strategic-Vulnerability Frontier (The $\lambda$ Sweep)
We swept the history penalty parameter $\lambda \in \{0, 0.05, 0.1, \dots, 5.0\}$ for the Score mechanism (M4) to observe the tradeoff between fairness, welfare, and manipulation gain ($M$).
- **Vulnerability of Greedy:** At $\lambda = 0$ (equivalent to Greedy), users gain massively by inflating their reports ($M \approx 220$), making the system highly vulnerable to manipulation.
- **The "Sweet Spot" ($\lambda \approx 1.0$):** As $\lambda$ increases, the history penalty reduces the benefit of strategic inflation. At $\lambda = 1.0$, the manipulation gain becomes strictly negative ($M \approx -3.41$). This means **strategic inflation actively hurts the user**. 
- **Welfare Retention:** Remarkably, at $\lambda = 1.0$, social welfare remains exceptionally high ($WR \approx 0.960$), and allocation fairness approaches perfection ($J_A \approx 0.991$). This confirms **Hypotheses 1 and 2**: a moderate history penalty drastically improves fairness and disincentivizes manipulation without a catastrophic loss of true welfare.

## E3: Strategic Population and Attack Type
This experiment scaled the fraction of strategic users $\rho$ and tested various bounded attacks (Truthful, Capped Exaggeration, Maximum Claim).
- **Degradation of Welfare:** When users employ the `max_claim` strategy, mechanisms relying on reported values degrade. Even the robust Vickrey mechanism degrades to a random allocation in terms of social welfare ($WR \approx 0.561$) when everyone max-claims, because prices become uniformly high and allocation becomes a tie-breaker.
- **Report-Invariant Stability:** Random and Round-Robin maintain a stable $WR \approx 0.560$ regardless of the attack type or the proportion of strategic users ($\rho$), confirming their total immunity to report inflation.

## E4: Heterogeneous Users
We evaluated a two-group population with differing value distributions (e.g., $\text{Beta}(2,5)$ vs. $\text{Beta}(5,2)$).
- **Service vs. Benefit:** Round-Robin maintains perfect allocation fairness ($J_A = 1.0$) and near-perfect normalized benefit fairness ($J_B \approx 0.999$). 
- **Efficiency Bias:** Greedy, Score, and Vickrey mechanisms naturally allocate more GPUs to the group with the higher value distribution to maximize overall social welfare. Consequently, their fairness scores drop substantially in mixed populations ($J_A \approx 0.506, J_B \approx 0.516$). This addresses **Hypothesis 3**: when values are heterogeneous, equal allocation does not organically arise from value-aware welfare maximization.

## E5: Temporally Persistent Demand (AR(1) Process)
When user valuations are persistent across rounds (using an AR(1) process with $\alpha \in \{0.0, 0.5, 0.9\}$) rather than strictly i.i.d.:
- High-value users hold on to their high values longer. As persistence $\alpha$ increases to $0.9$, the welfare achieved by Random and Round-Robin artificially rises ($WR \approx 0.843$).
- Value-based mechanisms (Greedy, Score, Vickrey) continue to successfully track the highest true values and consistently achieve optimal welfare ($WR = 1.0$) regardless of temporal persistence.
- Allocation fairness ($J_A$) for value-based mechanisms drops slightly as persistence increases (from $0.996$ at $\alpha=0.0$ to $0.957$ at $\alpha=0.9$), as heavy-hitters monopolize the GPUs for longer consecutive streaks.

## E6: Scalability Benchmark
The Python simulator exhibits highly efficient scalability:
- Simulating a population of $n=500$ users takes only $\approx 57.3\mu\text{s}$ per round for the most complex mechanism (Vickrey).
- The state footprint is extremely minimal, requiring only $\approx 11.7\text{ MiB}$ of RAM at $n=500$.

---

## 🎯 Final Conclusions

1. **The Limitations of Extremes:** Strict equality mechanisms (Round-Robin) eliminate starvation and manipulation but sacrifice ~40-45% of potential system value. Conversely, pure Greedy allocation achieves optimal value but induces severe starvation and is highly vulnerable to priority inflation.
2. **The Success of History Penalties (Score Mechanism):** The results provide strong evidence for the practical viability of the **History-Penalised Score Mechanism** ($s_{i,t} = \hat{v}_{i,t} / (1 + a_i(t))^\lambda$). Setting $\lambda \approx 1.0$:
   - Achieves near-optimal efficiency ($>95\%$ WR).
   - Eliminates starvation effectively.
   - Provides an empirical, non-monetary defense against priority inflation (manipulation yields negative utility).
3. **No Need for Real Money:** While the Vickrey auction provides rigorous dominant-strategy truthfulness, the History-Penalised Score mechanism offers a highly competitive proxy for fairness and efficiency without requiring a complex, real-money intertemporal payment infrastructure.
