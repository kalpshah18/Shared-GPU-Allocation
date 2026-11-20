# PR #5 review follow-up

Validated on 2026-11-19 after clarifying the report's adaptive-regret discussion.

## Report clarification

Section 4.5 now states that the twenty-sweep profiles' held-out regret bands
include zero. It separately identifies the four-sweep Karma-Cap profile from
the doubling start as having a positive band, as shown in Table 6. The numerical
results and the finite-policy scope are unchanged.

## Smoke-test timeout investigation

The full suite passed: **526 tests in 414.82 seconds**. The end-to-end quick
pipeline test passed in **208.12 seconds**, using its existing **900-second**
subprocess timeout. No timeout, test, simulation, or plotting code was changed.

The completed smoke run generated all 22 study summaries and raw/configuration
outputs, all 30 figure pairs (60 PDF/PNG files), the results table document,
course report and failure-case report. Its feedback and follow-up data audits
also passed. This verifies completion beyond the plotting stage where the
earlier run timed out.

The previous timeout did not reproduce in this environment; this successful
run does not establish the cause of that earlier timeout. The PR's earlier
timeout-qualified result is superseded by the complete passing run recorded here.

Runtime: Python 3.13.11, NumPy 2.5.3 and Matplotlib 3.11.2, using the project's
virtual environment and pinned package versions.

Reproduce from the repository root:

```text
python -m pytest -q --durations=15
```

To run only the original timeout-qualified check:

```text
python -m pytest tests/test_plots_and_pipeline.py::test_run_all_quick_pipeline_end_to_end -q --durations=1
```

## Rebuilt submission

The report audit reconstructed all 115 displayed cells and verified the 270
unique certified oracle records. The refreshed PDF retains eight body pages
and one reference page; only page 8's text changed. The portable source archive
compiled independently to matching text on all nine pages, with no overfull
boxes or unresolved references. The quick-run command retains literal ASCII
hyphens when copied from the PDF. Source checksums and archive contents were
checked against the final files.
