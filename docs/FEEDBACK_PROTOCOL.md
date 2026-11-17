# Reviewer-feedback experiments: M4 timing and credit budgets

The core question is how M4's cumulative-allocation penalty changes profitable
reporting, particularly when a user can withhold low reports to preserve future
priority. Greedy (M3) and monetary Vickrey (M5) are theory controls. E2/E7 study
penalty strength and fixed inflation; E8 studies history-triggered inflation;
E19 adds the value-threshold rule to the core pipeline at n=50. M6/M7 and
E12-E18 remain extensions of that question.

E19/E20 reuse the 30 locked master seeds, with n=50, k=10, T=1000, rho=.25
and up to five focal strategic users per seed. These are exploratory feedback
experiments, not a new holdout or a preregistered confirmation. Thresholds and
rates form sensitivity grids. Average focal users within each seed, then
bootstrap the 30 independent seeds using 10,000 resamples. Intervals are
pointwise 95% intervals without adjustment for all grid comparisons.

## E19: value-threshold timing

For tau in {.5, .75, .9}, sandbag reports zero below tau*v_max and v_max
otherwise; boost reports truth below tau*v_max and v_max otherwise. A zero
report can still win a zero-bid tie; it does not guarantee abstention. The
policy uses only the current private value, without observing future values.

Run M4 at lambda in {.5, 1, 2, 5}, plus M3, M5 and Credit-Vickrey at rate .5.
Save population-profile cells for sandbag, boost and always-max, with standard
welfare, fairness, waiting, SWL and unilateral gain metrics. Each unilateral
gain compares reporting to truth with opponents fixed within that cell.
However, opponents use sandbag in one cell and boost in another. Subtracting
these cell-level gains does **not** isolate timing. E8's profile differences
have the same limitation.

The primary timing comparison therefore fixes opponents at boost_tau on the
strategic set excluding the focal user, and truthful elsewhere. Change only
the focal user's episode-long policy between truth, boost and sandbag. All
runs use fresh identical mechanism states, valuations and tie-breaking streams.
Save every focal utility and gain. The controlled timing premium is
U_f(sandbag)-U_f(boost). It measures the net effect of suppressing low reports,
including forgone current benefits and downstream allocation/budget effects.
It is not a decomposition of the entire strategic gain into additive causes.

H2 concerns an increase with lambda. Report paired changes in the controlled
premium for each larger lambda relative to .5, with seed-bootstrap intervals.
Positive premium means suppression helps at that lambda; a positive premium
*change* supports the direction of H2 for that threshold and comparison.
Neither establishes a universal monotonic relationship or strategyproofness.
All thresholds, including negative results at tau=.9, appear in tables and
the controlled-comparison figure.

E13/E18 retain the original frozen eight-policy game. Their convergence and
regret claims are restricted to that library; adding these policies would
define a different empirical game requiring new training and evaluation.

## E20: renewable credit Vickrey (M5c)

Income I=credit_rate*(k/n)*v_max is added before each round. Balances start at
C=credit_horizon*I, default horizon 20, and are clipped to C after income.
Income above C is discarded; existing credits have no age-based expiry. Bid
min(report, available balance); the top k users win with seeded ties and pay
the (k+1)-st bid. Payments are destroyed, with no redistribution, borrowing,
transfers or cash redemption. Credits are an internal bidding resource and
have zero direct utility: utility is true GPU allocation benefit only.

The exact transition is balance_next=min(balance+I,C)-payment. Winners never
pay more than their bids or available balances; losers pay zero. Test each
transition, the total spending bound, real-vs-rollout state consistency, and
the allocation reduction to Greedy under a sufficiently loose finite budget.

Sweep credit_rate in {.25, .5, 1, 2}, with M3 and monetary M5 references.
Policies: truth, cap_1.25, cap_2, max claim and sandbag_.75. Report truthful
WR, Jain fairness and waiting plus strategic gains and SWL. These policy-cell
gains establish tested profitable deviations, not a causal decomposition
of credit saving. E19 supplies the fixed-opponent timing control at rate .5.
No monetary Vickrey DSIC claim transfers to this dynamic budget model. Balance
ceilings also provide no hard service guarantee.

## Reproduction and terminology

SWL=(W_truthful-W_strategic)/W_truthful replaces the ambiguous historical
abbreviation. Its values can be negative. `price_of_strategy` remains a Python
compatibility alias; legacy numerical results were migrated by key rename only.
Existing historical planning-attack results retain their original provenance.

```
python experiments/e19_value_threshold.py
python experiments/e20_credit_budget.py
python analysis/verify_feedback.py
python analysis/plots.py
python analysis/report.py
python analysis/course_manuscript.py
```

The full `python run_all.py` includes both studies and the feedback data audit.
For smoke runs use `--quick --results-dir tmp/results --figures-dir tmp/figures`.
Configs and paired/controlled outputs are tracked alongside raw seed data and
figures. The audit reconstructs summaries and paired intervals from raw data,
checks seed alignment and utility identities, and rejects non-finite JSON.
