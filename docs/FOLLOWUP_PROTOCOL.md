# Failure-mode follow-up protocol

Fixed before running the follow-up analyses. Mechanism parameters and the eight
reporting policies stay as in E12-E15. These studies explain observed failures;
they do not tune a replacement mechanism.

## E16: Rank-Cap history channel

Reproduce the updating and frozen H=40, R=80 attacks on all 30 E15 evaluation
seeds, n=10, k=2, T=120, focal user 0. Use the same attacker random streams.
Record values, reports, empirical ranks, allocation counters, waits, forced
service, and cumulative utility. The example trajectory is the first locked
seed, chosen by list order, not by attack success.

At each round evaluate the SAME true value against the attacked report history
and a truthful history. Their rank difference measures the direct reference-
distribution effect, holding the query fixed. It is not a utility decomposition.

Replay each saved attack report sequence using truthful focal values to update
the focal rank reference. This privileged-information intervention disables
report-history contamination while allowing allocation counters and waits to
evolve. It is a fixed-sequence counterfactual, not a deployable rule or the
attacker's best response to that counterfactual. Compare utility with ordinary
replay on the identical sequence. Separate forced and ordinary service gains.

The 30 existing seeds are reused for explanation. No new holdout claim is made.

## E17: Karma accounting and population reversal

Reproduce E12 cap_2 with rho=0.25 and rho=1, both without the cap and with W=10,
n=50, k=10, T=1000, on its existing 30 seeds and original five focal users per
seed. Hold each focal user's opponents fixed in the truthful/deviating pair.

Record internal balances, effective bids, uniform auction prices, forced-service
charges, dividends and ceiling losses. Verify every balance transition against
the actual mechanism. Decompose unilateral utility gain exactly into benefit
from forced and auction service. Also compare win counts, value per win, credit
spending and the fraction of bids constrained by balances. Use the first locked
seed and first focal for the descriptive trajectory.

Forced winners DO pay karma in the implemented mechanism. Do not assume that
forced service is free. Accounting decompositions and cap ablations do not
alone identify an equilibrium or a unique causal explanation of the reversal.

## E18: extending the adaptive training budget

Keep n=6, k=2, T=120, training seeds 201-205, order seed 329, six mechanisms,
eight policies and both initial populations. Compare four and twenty sweeps.
Record sweep-boundary recurrence but continue after it at the twenty-sweep
budget, because a repeated profile under shuffled orders is not a periodic
orbit. Retain policies on ties and stop when an entire sweep makes no changes.

Audit frozen final profiles on 30 newly locked follow-up seeds, disjoint from
the original and E12-E15 evaluation seeds and both development lists. Maximise
unilateral gain only AFTER seed averaging and use the same simultaneous bands
as E13. Compare four-vs-twenty welfare on paired workloads. The sweep budgets
are fixed here; no evaluation-based budget selection is permitted.

Save strict JSON, full configs, per-seed outcomes, training traces, comparisons
and source digests. Full runs use 10,000 bootstrap resamples. Smoke outputs go
to separate directories. Figures and a case-study report are generated from
saved data, with descriptive diagnostics separated from statistical evidence.
