# Final project report

The report completes `FAI_Project_Proposal.tex`, the review source supplied by
the group, and retains its PRIME style. The original proposal is preserved.
The final submission has **eight body pages plus one page of references**.
All four primary contribution assignments were confirmed fulfilled by the group.

- PDF: `../output/pdf/Shared_GPU_Allocation_Final_Report.pdf`
- Editable source: `FAI_Project_Final_Report.tex`
- Portable source ZIP: `../output/latex/Shared_GPU_Allocation_Final_Report_Source.zip`
- Result/evidence assets: `final_assets/`
- Repository: https://github.com/kalpshah18/Shared-GPU-Allocation

## Build from the repository root

```
python analysis/build_final_report.py
python analysis/compile_final_report.py
```

The first command uses the project's Python dependencies. It reconstructs
reported summary statistics from raw seed rows using 10,000 bootstrap resamples,
checks seed roles, and regenerates four plots, six result tables and numeric text macros.
Adaptive regret bands are recomputed with their simultaneous, library-specific
bootstrap rather than treated as ordinary scalar means. Training metadata and
training/evaluation seed separation are checked too.
It audits all feedback cells and verifies the count of 270 unique optimal oracle
solves. It does not rerun experiments or overwrite their outputs.

For a code-to-data check, run `python analysis/verify_final_replay.py` before
building the assets. It replays the first saved seed in 13 representative
cells, including controlled timing, the credit mechanism, the constrained
oracle, uncapped Karma/Rank and the corrected planning attack. Its record is `final_assets/replay_checks.json`.
This is a representative replay check, not a full-grid simulation rerun.

The second command requires PDFLaTeX and BibTeX in PATH. It compiles with shell
escape disabled, requires the body to end on page eight, and rejects overfull
boxes or unresolved references. It writes the final PDF and source archive.
Intermediate compilation files are in `final_build/` and are ignored by Git.

For Overleaf, upload the source ZIP and select `FAI_Project_Final_Report.tex`
as the main document. The archive includes the style, bibliography, generated
tables and PDF figures, so it can compile without the simulation repository.

## Results and provenance

The evidence file records the data commit, hashes of source files and input
JSONs, raw cell paths, checked statistics, seed roles and audit/render versions.
Every displayed measurement is generated from these inputs; narrative numbers
use LaTeX macros rather than separately typed statistics.

| Paper evidence | Source experiment | Conditions |
|---|---|---|
| Table 1 | E1/E4 | Truthful homogeneous/mixed demand, n=50, k=10, T=1000 |
| Figure 1 | E7 | M4 inflation sensitivity, rho=.25, 30 master seeds |
| Table 2, Figure 2 | E19 | Fixed-opponent timing and distinct population-profile gains |
| Table 3, Figure 3 | E20 | Renewable credit budgets and policy-specific populations |
| Table 4 | E12 | All six Score/Karma/Rank capped/uncapped rows; 30 extension seeds |
| Figure 4 | E15 | Corrected planner, n=10, k=2, T=120, H=40, 80 sampled futures |
| Table 5 | E14 | Mixed demand, n=6, k=2, T=40, W=6, B=8 |
| Table 6 | E18 | Two starts, 4/20 training sweeps; 30 independent follow-up seeds |
| Workload/benchmark text | E5/E6 | Marginal-preserving persistence; saved machine-dependent timing |
| Diagnostic text | E16/E17 | Rank reference replay and forced/auction accounting; reused extension seeds |

The [coverage map](../docs/REPORT_COVERAGE.md) accounts for every implemented
study, including historical checkpoints, parameter sweeps and selected results.
Five history/service extensions are shown explicitly: Score-Cap, Karma,
Karma-Cap, Rank and Rank-Cap. Credit-Vickrey is a separate budget extension.

E19/E20 reuse existing master seeds and have pointwise exploratory intervals.
The paper distinguishes those from the E12--E15 extension evaluation set,
its descriptive reuse in E16/E17, the independent E18 follow-up set, and
the historical E3b/E3c/E11 attacker checkpoints. Waiting maxima are averages
of pre-round per-episode maxima; the formal cap bound also includes terminal
waiting. The report makes no production-system or dynamic-strategyproofness claim.

## Layout verification

The PDF is rendered page by page for visual checks. The final layout has four
readable plots, six result tables, a study overview, the contribution table and six references.
The source archive is also compiled independently to check portability.
