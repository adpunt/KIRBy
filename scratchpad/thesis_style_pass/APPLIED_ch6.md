# Applied: PLAN_ch6.md (2026-10-10)

Every edited paragraph already had a "% TODO: manual review" marker above it (checked in thesis.tex); markers kept, no new TODO lines needed.

- E1 applied.
- E2 applied.
- E3 applied.
- E4 applied.
- E5 applied.
- E6 applied.
- E7 applied.
- E8 applied.
- E9 applied.
- E10 applied (caption of tab:knn_settings, in thesis.tex, not a figure file).
- E11 applied.
- E12 applied.
- E13 applied.
- E14 applied.
- E15 applied.
- E16 adapted: new definition sentence reads "A training molecule is a cliff molecule when ..." instead of "A cliff molecule is one that ...", because the next sentence defines held-out cliff molecules by a different rule.
- E17 applied.
- E18 applied.
- E19 applied.
- E20 applied.
- E21 adapted: the numbers had already been cut from the paragraph (2026-10-10); only "tracks activity" changed to "tracks label differences". The plan's re-added hERG K_i pair was not put back.
- E22 adapted: paragraph already rewritten with numbers cut; only "track activity" -> "track label differences" and "model hybrids" -> "predictive hybrids".
- E23 adapted: sentence already shortened with numbers cut; only "best follows activity" -> "best follows the label".
- E24 adapted: first sentence reworded to "a hybrid's prediction score peaks at more features than the correlation between its distances and label differences does"; the "Part of this gap is built into the experiment" sentence removed; RMSE numbers not re-added (current text says "predicts better at 2,000 features than at 500"). Checked s08_optimal_dimension.py: neighbour numbers are computed on the general hybrids at each general budget (neighbour_by_budget.general_hybrids, line 775), so the LogD 0.2315/0.1535 pair is the same hybrid at the same budgets. Its docstring still says the distance side sweeps distance budgets separately (decision 2026-10-06), so the removed sentence describes a different comparison than the example.
- E25 skipped: the paragraph's numbers were already cut to the table on 2026-10-10; no "activity" wording left to fix.
