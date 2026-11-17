# PR #3 review and completion

The review started from clean `main` at `5680208` and inspected PR head
`56e0e53`. The project simulates repeated unit-demand GPU allocation, pairing
valuation, tie-breaking and strategic-set randomness across counterfactuals.
M4 scores reports using cumulative allocation history. M3 and monetary M5
provide theory controls; the capped mechanisms, empirical policy game and
offline service oracle are separate extensions.

## Findings addressed

| Finding | Correction |
|---|---|
| E19 subtracted unilateral gains from different opponent populations and called the difference the value of timing | Add a focal-only truth/boost/sandbag comparison with opponent policies fixed; retain the original profile cells with descriptive labels |
| A positive premium at one penalty was treated as evidence that the premium increases with the penalty | Save seed-paired premium changes relative to the smallest lambda and show their intervals; qualify H2 by threshold and comparison |
| New paired results and configs were missing because `.gitignore` excluded them | Track E19/E20 configs, E19 paired and controlled summaries, focal-level raw utilities, and the feedback audit |
| The figure omitted the negative high-threshold timing result and showed unpaired profile differences as a timing effect | Plot every threshold using the controlled premium and seed-bootstrap error bars |
| The generated course report did not incorporate the feedback experiments | Generate a leading core M4 section with E19/E20 tables, model details, SWL terminology and seed/library limitations |
| The credit implementation capped balances but documentation claimed age-based expiration | Document capped accumulation and discarded excess income consistently; add default parameters to `base.yaml` |
| Non-finite credit parameters could produce invalid budgets | Reject non-finite rates/horizons; test exact credit transitions and a concrete profitable deviation |
| Paired comparisons trusted list position without checking seed IDs | Reject reordered or missing seed IDs when seed metadata is present |

The SWL rename was checked against `main`: all **311** modified legacy result
JSON files equal their original contents after recursively renaming only
`PoS` to `SWL`. Historical planning-attack numbers retain their checkpoint
labels. The Python compatibility alias remains available.

## Validation and interpretation

`tests/test_feedback.py` checks a hand-computed M4 sacrifice, Greedy/Vickrey
controls, batch replay against independent episode simulations for all four
E19 mechanisms, seed alignment, metadata, raw-data audits and report tables.
Credit tests check each balance transition, feasible payments, cumulative
spending, single/batch consistency and the loss of monetary DSIC.

The full E19/E20 design uses 30 master seeds and 10,000 bootstrap resamples.
The data audit reconstructs all profile/controlled summaries and paired
intervals, verifies focal utility identities, and checks that the controlled
boost arm reproduces the original boost-cell unilateral gain. The complete
quick pipeline includes both studies, audit, figures and report generation.

Both experiments were rerun in full. `results/FEEDBACK_AUDIT.json` records a
passing audit of **79 population-profile cells, 21 controlled timing cells and
51 paired comparisons**. At tau=.75, M4's controlled timing premium rises from
-0.223 at lambda=.5 to +1.372 at lambda=5; the paired increase is +1.595 with
95% interval [1.454, 1.750]. This supports the direction of H2 for that
comparison, with the threshold and exploratory-inference qualifications below.

Validation passed: the full suite at that checkpoint had **521 passing tests**,
including the complete quick pipeline; the updated source passed **516 tests**
outside the plotting/pipeline module and **42 focused tests**. Together these
cover all **526 tests** collected after the new regressions were added. The
updated E19/E20 figures were also rendered and visually inspected. The full
30-seed E20 summary reproduces the original statistics to within 1e-15.

Validation used Python 3.13.4, NumPy 2.2.6, SciPy 1.16.0, Matplotlib 3.10.5,
PyYAML 6.0.3 and pytest 9.0.1 available in the review environment. These satisfy
`requirements.txt`; this is not a claim of validation under the exact versions
in `requirements-lock.txt`. The pytest base directory was placed inside
`.pytest_cache` to accommodate the execution sandbox.

E19/E20 reuse existing seeds and are exploratory; intervals are pointwise and
do not adjust for all grid comparisons. E13/E18 retain their originally fixed
eight-policy library, so their restricted convergence claims do not cover the
new policies. Credit-Vickrey is one explicit budget model, with no service
bound or inherited truthfulness guarantee. These limitations are recorded in
[FEEDBACK_PROTOCOL.md](FEEDBACK_PROTOCOL.md) and the generated report.

```
python -m pytest -q --basetemp .pytest_cache/pr3-review
python experiments/e19_value_threshold.py
python experiments/e20_credit_budget.py
python analysis/verify_feedback.py
python analysis/plots.py
python analysis/report.py
python analysis/course_manuscript.py
```
