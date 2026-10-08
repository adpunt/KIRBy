# Result caveats: what to state when results from run_2026_10_06 are written up

One entry per change or known property of the results. Each says what was done, which
results it touches, and the commit. Written by Claude on 2026-10-08 from the code and
the result files; the decisions are Adelaide's.

## dnn predictions are capped at the training-label range (2026-10-08)

- **What:** `DNNRegressor.predict` (`src/kirby/models/dnn.py`) clips every prediction to
  the lowest and highest label among that fit's training rows.
- **Why:** adme_hlm row 824 has pdv `Ipc` = 1.95e17 (largest training value on its fold
  1.15e11) and mordred feature 13 = 2.57e12 (next largest in the dataset 8.3e4). The
  dnn passed it through unbounded and predicted ~5e5 away from the label, so one fold's
  RMSE was 1e3 to 1e5 in c06, c11, c17, c21 and c22 on adme_hlm and adme_rlm. Without
  that molecule the fold scores 0.476 RMSE.
- **Which results:** every dnn result written after this commit. The dnn units of c06,
  c11, c17, c21 and c22 on adme_hlm and adme_rlm are re-run.
- **Not re-run:** the per-fold searches on adme_hlm and adme_rlm. Their stored dnn scores
  on inner folds were computed before the cap.
- **Two rules exist side by side:** c01, c07 and c08 also limit each held-out feature to
  its training range; the other experiments do not.

## gp starts its kernel width at the square root of the number of features (2026-10-08)

- **What:** the downstream gp (`WidthScaledGPR`, `src/kirby/find_best_hybrid.py`, commit
  ffcd254) starts the RBF length scale at sqrt(features) instead of 1.
- **Why:** on inputs of 1,000+ features a start of 1 left the fit stuck; whether it
  escaped depended on the random seed, and a stuck fit predicted one value for every
  held-out row (spearman NaN). Reproduced on abagym_Ang2: seed 42 fit, seeds 43 and 44
  stuck on all 5 folds.
- **Which results:** abagym_Ang2, abagym_VEGF, ginkgo_acsins, ginkgo_hic, ginkgo_pr_cho
  (gp is one of their two models) are re-run from the start, searches included. No other
  dataset uses gp as a model.
- **Unchanged:** the gp the search uses internally to score candidate hybrids
  (`src/kirby/hybrid.py`) still starts at width 1. It is used only on these five datasets,
  when scoring for the gp model.

## s11 skips caco2 and logd (2026-10-08, commit de3c1b0)

The loader clips caco2 and logd labels at their 1st and 99th percentiles, so the top 1% of
molecules share one label (22 molecules on caco2). Every s11 starting set already held one,
so all 20 conditions scored 167.8871 on caco2. The existing caco2 s11 file carries no
information.

## Labels clipped at the 1st and 99th percentile on caco2 and logd

`tests/find_best_hybrid_molecular.py:244-247`, on by default, in every experiment. The
caco2 label is log10 of the Caco-2 efflux ratio (column `Caco-2 Permeability Efflux`),
not permeability.

## tcga_brca_pam50 outer fold 0 holds 3 patients

It trains on 818 and holds out 3, so every fold-0 score is 1 or a multiple of one third.
Report means with and without fold 0. tz_her2_binary outer fold 4 holds 13 rows (fold 3
holds 3,035).

## c18 and c22 re-run with their old files moved aside (2026-10-08)

The files written before commits e917e9d (c18: equal split over every representation) and
5d0caf1/8e515d0 (c22: SHAP and permutation importance for every model) lacked those
conditions, and the reruns of 7 October kept the old files: the resume check compares the
data and the folds but not the result format. The old files are in
`tests/results/run_2026_10_06/superseded/`.
