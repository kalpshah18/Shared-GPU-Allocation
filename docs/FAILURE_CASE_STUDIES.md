# Why the allocation mechanisms fail


Follow-up to the component study: report-history dynamics, internal credit accounting, and longer adaptive training.


## Design and interpretation


E16 and E17 reuse 30 existing evaluation workloads to explain previous results. E18 uses 30 newly locked independent evaluation seeds and trains only on seeds [201, 202, 203, 204, 205]. The [fixed follow-up protocol](FOLLOWUP_PROTOCOL.md) specifies budgets and comparisons before these runs. No mechanism tuning uses these results. Per-seed bootstrap intervals are 95%; full runs use 10,000 resamples. Mechanistic identities, replay interventions, and empirical gains answer different questions.


## 1. Rank-Cap: modelling the cost of changing report history


The two attackers use n=10, k=2, T=120, H=40, and 80 simulated futures per candidate. Both actual mechanisms update real histories. The frozen control freezes only hypothetical histories inside the planner. Workloads, ties, and attacker random streams are paired.


| Attacker | Actual gain [CI] | Forced gain | Auction gain | Truthful-reference replay gain [CI] | Actual minus replay [CI] |
|---|---|---|---|---|---|
| updating | +2.58 [2.19, 2.98] | -0.90 | +3.49 | +1.27 [0.89, 1.65] | +1.31 [0.88, 1.73] |
| frozen | -0.92 [-1.80, -0.09] | -0.38 | -0.54 | +2.10 [1.54, 2.64] | -3.02 [-3.47, -2.58] |

Updating-minus-frozen utility gain is +3.505 [2.753, 4.332]. This isolates the planner's history approximation, not the causal utility contribution of the report ledger.


| Attacker | Underreport fraction | Overreport fraction | Mean true value when understating | Rank shift: all queries [CI] | Rank shift: high-value queries [CI] |
|---|---|---|---|---|---|
| updating | 0.320 | 0.323 | 0.747 | +0.001 [-0.008, 0.012] | +0.068 [0.056, 0.082] |
| frozen | 0.150 | 0.540 | 0.486 | -0.035 [-0.061, -0.011] | -0.114 [-0.147, -0.083] |

**The direct history effect.** For a fixed current query z, replace one past truthful report v by r in a ledger of size m. The empirical rank changes by

```text
([r<z] + 0.5[r=z] - [v<z] - 0.5[v=z]) / (m+1).
```

Understatement raises the future rank for r<z<v; overstatement lowers it for v<z<r. This compares the same query and ledger size. The logged shift compares the current TRUE value against reported and truthful past values, so it does not confuse a changed current report with a changed reference distribution. Allocation counts and waits introduce additional feedback.


The updating planner exaggerates less often than the frozen planner. Understatements also occur on relatively high values, so the trace does not support a simple rule of sacrificing only low-value rounds to build a favourable reference. A planner can lower an already sufficient winning report and thereby limit the later rank penalty of earlier exaggeration. The per-seed paired rank-shift and reporting contrasts are saved alongside the utility comparison.


**Report history contributes to the gain, but is not the whole explanation.** Replacing the attacked ledger with truthful reference values reduces the updating attacker's fixed-sequence utility gain, with a paired interval above zero for actual-minus-replay. This intervention lets allocation counters and waits evolve, so its effect includes feedback from changed service. The measured rank shift for high-value queries shows that selective reporting can improve the reference at valuable future queries even when its overall average rank shift is close to zero. The frozen planner instead exaggerates more often and suffers a negative mean reference shift.


The updating attack is still profitable under truthful-reference replay. Its benefit therefore includes current reporting/allocation effects beyond changes to the ledger. The paired ledger effect is not an additive decomposition of the optimal adaptive attack's incentives: reports are held fixed, and the attacker is not reoptimised. Truthful-reference replay uses private truth and is a diagnostic, not a feasible strategy-proof replacement mechanism.


The descriptive trajectory uses the first locked seed, 757622112, rather than the most profitable attack. All seed trajectories are saved. [Trajectory figure](../figures/e16_rank_trajectory.pdf); [aggregate channels](../figures/e16_rank_channels.pdf).


The first-seed ledger effect is -0.95, whereas the aggregate estimate is +1.31 [0.88, 1.73]. The example illustrates variation and should not be treated as representative of the mean effect.


## 2. Karma: service channels and the price of a deviation


E17 uses n=50, k=10, T=1000 and the original focal users. For each pair, only the focal changes between truthful and cap_2 reporting; all opponents keep their policies. The benefit identity is exact:

```text
Gain = change in forced-service benefit + change in auction-service benefit.
```


| Rule | rho | Total gain [CI] | Forced benefit gain [CI] | Auction benefit gain [CI] | Change in services | Change in credits spent |
|---|---|---|---|---|---|---|
| Karma | 0.25 | -35.82 [-36.22, -35.43] | +0.00 [0.00, 0.00] | -35.82 [-36.22, -35.43] | +1.22 | +1.59 |
| Karma-Cap | 0.25 | -29.23 [-29.75, -28.71] | -11.21 [-11.59, -10.85] | -18.01 [-18.69, -17.33] | +0.35 | +1.64 |
| Karma | 1.0 | +29.43 [28.85, 30.01] | +0.00 [0.00, 0.00] | +29.43 [28.85, 30.01] | +65.16 | +93.64 |
| Karma-Cap | 1.0 | +46.06 [45.40, 46.74] | -20.90 [-21.41, -20.40] | +66.96 [66.28, 67.64] | +63.44 | +80.67 |

| Rule | rho | Deviator bids limited by balance | Truthful bids limited by balance | Deviator value/service | Truthful value/service |
|---|---|---|---|---|---|
| Karma | 0.25 | 0.649 | 0.098 | 0.688 | 0.871 |
| Karma-Cap | 0.25 | 0.622 | 0.070 | 0.694 | 0.842 |
| Karma | 1.0 | 0.175 | 0.009 | 0.750 | 0.894 |
| Karma-Cap | 1.0 | 0.136 | 0.001 | 0.727 | 0.727 |

| Rule | rho | Deviator mean balance | Truthful mean balance | Forced credit change | Auction credit change |
|---|---|---|---|---|---|
| Karma | 0.25 | 0.633 | 1.879 | +0.00 | +1.59 |
| Karma-Cap | 0.25 | 0.666 | 2.029 | -16.98 | +18.63 |
| Karma | 1.0 | 1.706 | 4.460 | +0.00 | +93.64 |
| Karma-Cap | 1.0 | 1.815 | 5.169 | -38.41 | +119.08 |

**Forced service is charged.** An auction winner pays the remaining auction's uniform price; a forced winner pays min(overall slot price, its balance). Credits are redistributed equally, then balances are clipped at the ceiling. Every observed transition is checked against the real rule. Karma is not deducted from utility: a large credit payment affects later bids, not current net benefit.


**The cap is not necessary for the population reversal.** When everyone else exaggerates, positive mean deviation gains occur even with the cap disabled. Comparing the exact benefit channels distinguishes a cap contribution from the auction's strategic competition. Higher reports also encounter balance caps; price, credit recycling, service count and value per win must be considered together. The accounting describes the reversal but does not prove a unique equilibrium or identify one universally sufficient cause.


At rho=0.25 the deviator obtains nearly the same number of services but a lower true value per service than its truthful counterpart. At rho=1, exaggeration instead obtains substantially more services; with the cap, the two value-per-service averages are similar. The capped gain comes from additional auction benefit while forced-service benefit falls. This supports a competition/quantity explanation rather than a benefit from uncharged forced service. The deviator spends more credits, and those credits are a future bidding constraint rather than an immediate utility cost. These are population-dependent diagnostics on the tested cap_2 policy, not a general characterisation of karma games.


For Karma-Cap the deviator's balance limits substantially fewer bids at rho=1 than at rho=0.25, and its mean balance is higher. Thus the opportunity-cost deterrent seen against mostly truthful opponents is weaker in the inflated-bid population. This observed budget slack accompanies the service-count reversal; it is not a claim that changing the initial credit supply alone would reproduce all of the population effect.


[Accounting figure](../figures/e17_karma_accounting.pdf).


## 3. Does a longer adaptive training budget resolve residual incentives?


The four- and twenty-sweep profiles are trained independently of evaluation. A repeated sweep-boundary profile is recorded but does not stop the longer search; shuffled future orders may behave differently. Converged means that an entire sweep has no profitable policy changes on the training workload game.


**Initial policy: truthful.**


| Rule | Budget | Status | Sweeps used | Recurrences | Training regret | Independent regret [band] | WR [CI] |
|---|---|---|---|---|---|---|---|
| Greedy | 4 | converged | 2 | 0 | 0.000 | 0.000 [0.000, 1.623] | 0.635 [0.626, 0.643] |
| Greedy | 20 | converged | 2 | 0 | 0.000 | 0.000 [0.000, 1.623] | 0.635 [0.626, 0.643] |
| Score | 4 | converged | 4 | 0 | 0.000 | 0.281 [0.000, 1.239] | 0.977 [0.975, 0.978] |
| Score | 20 | converged | 4 | 0 | 0.000 | 0.281 [0.000, 1.239] | 0.977 [0.975, 0.978] |
| Score-Cap | 4 | sweep_limit | 4 | 0 | 1.163 | 0.208 [0.000, 1.108] | 0.946 [0.943, 0.949] |
| Score-Cap | 20 | sweep_limit | 20 | 4 | 0.682 | 0.663 [0.000, 1.599] | 0.950 [0.947, 0.952] |
| Vickrey | 4 | converged | 1 | 0 | 0.000 | 0.000 [0.000, 1.323] | 1.000 [1.000, 1.000] |
| Vickrey | 20 | converged | 1 | 0 | 0.000 | 0.000 [0.000, 1.323] | 1.000 [1.000, 1.000] |
| Karma-Cap | 4 | sweep_limit | 4 | 0 | 0.457 | 0.588 [0.000, 1.501] | 0.954 [0.951, 0.957] |
| Karma-Cap | 20 | converged | 18 | 0 | 0.000 | 0.074 [0.000, 1.031] | 0.948 [0.945, 0.951] |
| Rank-Cap | 4 | sweep_limit | 4 | 0 | 0.620 | 0.585 [0.000, 1.470] | 0.948 [0.945, 0.950] |
| Rank-Cap | 20 | sweep_limit | 20 | 3 | 0.235 | 0.274 [0.000, 1.265] | 0.948 [0.945, 0.950] |

**Initial policy: cap_2.**


| Rule | Budget | Status | Sweeps used | Recurrences | Training regret | Independent regret [band] | WR [CI] |
|---|---|---|---|---|---|---|---|
| Greedy | 4 | converged | 2 | 0 | 0.000 | 0.000 [0.000, 1.623] | 0.635 [0.626, 0.643] |
| Greedy | 20 | converged | 2 | 0 | 0.000 | 0.000 [0.000, 1.623] | 0.635 [0.626, 0.643] |
| Score | 4 | converged | 4 | 0 | 0.000 | 0.281 [0.000, 1.239] | 0.977 [0.975, 0.978] |
| Score | 20 | converged | 4 | 0 | 0.000 | 0.281 [0.000, 1.239] | 0.977 [0.975, 0.978] |
| Score-Cap | 4 | sweep_limit | 4 | 0 | 0.694 | 0.627 [0.000, 1.605] | 0.952 [0.950, 0.955] |
| Score-Cap | 20 | sweep_limit | 20 | 4 | 0.682 | 0.663 [0.000, 1.599] | 0.950 [0.947, 0.952] |
| Vickrey | 4 | converged | 2 | 0 | 0.000 | 0.000 [0.000, 1.323] | 1.000 [1.000, 1.000] |
| Vickrey | 20 | converged | 2 | 0 | 0.000 | 0.000 [0.000, 1.323] | 1.000 [1.000, 1.000] |
| Karma-Cap | 4 | sweep_limit | 4 | 0 | 0.477 | 1.542 [0.587, 2.496] | 0.939 [0.935, 0.942] |
| Karma-Cap | 20 | converged | 18 | 4 | 0.000 | 0.074 [0.000, 1.031] | 0.948 [0.945, 0.951] |
| Rank-Cap | 4 | sweep_limit | 4 | 0 | 0.620 | 0.585 [0.000, 1.470] | 0.948 [0.945, 0.950] |
| Rank-Cap | 20 | sweep_limit | 20 | 3 | 0.235 | 0.274 [0.000, 1.265] | 0.948 [0.945, 0.950] |

At budget 20, Rank-Cap, Score-Cap still have residual training incentives. Longer training therefore does not establish a fixed point for every mechanism.


Karma-Cap reaches a training fixed point from both starting profiles after 18 sweeps. Its longer-trained welfare and independent regret must still be read from the table; convergence refers only to the training game.


Independent regret bands exclude zero for Karma-Cap from cap_2 at budget 4. Thus the earlier zero-containing bands do not establish robust equilibrium behaviour on a new evaluation set. The fixed budgets were not selected using these results.


A training fixed point is a restricted pure-policy equilibrium on five training traces, not unrestricted dynamic strategyproofness. An evaluation interval containing zero does not establish positive expected regret; a sweep limit or recurrence with residual training regret prevents a training equilibrium claim. [Training-budget figure](../figures/e18_extended_adaptation.pdf).


## Course presentation outline

1. State the repeated private-value allocation model and the welfare/service/incentive question.
2. Show Score-Cap as the matched service baseline and the formal waiting bound.
3. Show the corrected Rank-Cap attack, the first-seed trace, and truthful-reference replay.
4. Explain Karma's reversal using forced/auction benefit and credit accounting.
5. Compare four-vs-twenty training sweeps on independent evaluation seeds.
6. Close with the constrained welfare benchmark and the precise limits of each claim.


## Reproduction

```text
python run_all.py --followups-only
```

This runs E0 and E16-E18, then generates the figures and this report. Use `--quick` and separate results/figure directories for development. Older experiment outputs are retained.


The [follow-up audit](../results/FOLLOWUP_AUDIT.json) checks independent evaluation seeds, saved means, paired differences, credit transitions and original seed-level reproduction.
