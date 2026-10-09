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

## Fixes of 2026-10-08 evening (from the fault list of the 2026-10-08 analysis)

Each entry: what changed, why, which results it touches. The re-runs are in the run
block of the same day. Written by Claude; decisions are Adelaide's.

### The search's internal gp and the other gp starts now begin at sqrt(features)
- **What:** the gp the search uses to score candidate hybrids (`src/kirby/hybrid.py`,
  internal_evaluator='gp'), n09's `heteroscedastic_gp` (`src/kirby/utils/lnl_methods.py`)
  and s11's surrogate gp (default `KIRBY_S11_LENGTHSCALE=sqrt`; 'fixed' and 'median'
  still selectable) start the RBF length scale at sqrt(number of features).
- **Why:** on abagym_Ang2 antiberty (1,024 features) the internal gp never left length
  scale 1 on any of 5 folds and predicted one value for every held-out row, so the scores
  it gave the search were rounding noise. With the new start it fits 62.6 to 93.2 and
  scores 0.09 to 0.60 spearman per fold.
- **Which results:** the per-fold searches on abagym_Ang2, abagym_VEGF, ginkgo_acsins,
  ginkgo_hic, ginkgo_pr_cho and everything reading them; c23 svm `B_slight_off` (its
  near-family evaluator is this gp) on every svm dataset; n09; s11.
- **Not changed:** the gp classifier (`find_best_hybrid._get_gp_classifier`) still starts
  at 1; gp is a decided model on no classification dataset.

### n09's own networks clip predictions to the training-label range
- **What:** the survey methods in `lnl_methods.py` clip regression predictions to the
  lowest and highest training label, the rule `DNNRegressor.predict` has had since
  c18ee30. Methods that drop rows clip to the kept rows' labels.
- **Which results:** every n09 unit.

### Noise cache: the first stored value is kept
- **What:** `tests/experiments/noise/noise_cache.py` never replaces a stored entry; a
  process that computed second reads the stored value (counter `kept_first`).
- **Why:** concurrent jobs all missed the cache, computed, and overwrote each other. The
  hybrid build itself differs slightly across thread counts (deeplift is a torch fit), so
  the same recipe on the same fold and clean labels held different scores in different
  files: caco2 n07 dnn `hybrid` 0.233311 rmse in three files, 0.247412 in one; herg_ki
  dnn `hybrid` 0.598948 against 0.597232 inside one file.
- **Which results:** dnn and lightgbm scores of caco2, herg_ki, herg and logd in the
  noise experiments. Values after a cache wipe or on another cluster are still decided by
  the first job to compute them.

### Ledger records ok only when the result file exists and says ok
- **What:** `ledger.py finish_unit` reads the expected result file: missing or unreadable
  -> failed; a skip file -> skipped; a status other than ok -> failed.
- **Why:** c33 ginkgo_hic and s08 ddg_combined were ok with no file; seven grouped_shifted
  units on antibody datasets were ok after 3 to 48 s with no file.
- **Also:** grouped_shifted and grouped_wider write a skip file with the reason on datasets
  without SMILES (`noise_helpers.run_all`); n20 records failed when every damaged draw
  failed; c10's in-progress snapshot is `partial`, not `empty`; c07 writes a skip when no
  partition can be scored; `experiment_result_is_empty` treats partial and failed files as
  unfinished, so a resume recomputes them.

### Noise refits are compared with no cleaning on the same draws
- **What:** n09, n12, n13, n21 and n35 store the no-cleaning score per (fold, draw) and
  the refit and no-cleaning means over the same passes. n13's neighbour-setting scores were
  keyed without the draw and kept only the last draw; they are now the mean over the draws
  that kept the setting, with the per-pass values stored.
- n18 and n20 store the specific reason a cleanup declined (e.g. gp_smooth over
  GP_SMOOTH_MAX_ROWS=1000 training molecules; constant_lift with too few molecules on the
  largest label).
- **Which results:** n13 must be re-run before `track_a_by_similarity` is read. The others
  only gain keys.

### Re-picks choose only spaces scored on the held-out fold
- **What:** `_shared.fold_reselection` (s05, s13 and every other user) and s10's kNN re-pick
  consider only candidates with a score on the held-out fold; folds with none are listed.
- **Why:** on ddg_combined the re-pick chose a space scored on fold 0 only and left folds
  1 to 4 empty.

### Similarity objective undefined on ddg_combined; suzuki cut
- **ddg_combined:** one protein complex holds 69 of 429 rows. On outer folds 1 to 4 it is
  alone in an inner held-out block, and predicting each row from its five nearest
  neighbours needs two complexes there, so every configuration was refused. Added to
  `SIMILARITY_UNDEFINED_OTHER`. When a search drops every configuration, the stored reason
  now names the refusals (`SearchResult.config_failures`).
- **suzuki:** 12 ligand groups of 480. On outer folds 0 and 1 the inner split holds out one
  ligand alone and every search configuration lost that fold. Added to `CUT_DATASETS`.

### Packages
- `cleanlab==2.9.0` added to requirements and setup.sh (n32 and n41 recorded "No module
  named 'cleanlab'" 140 times). `gauche==0.1.6` installed with `--no-deps` by setup.sh (s11
  recorded its gauche surrogate unavailable). In skip mode setup.sh now puts hmmscan on
  PATH; ANARCI needs it, and its absence on some nodes is the likely (unverified) reason
  s01's CDR counts and two CDR representations were skipped.

## c17's PCA no longer rescales its components (2026-10-09)

- **What:** in the comparison of the hybrid against PCA, mRMR, recursive elimination and
  random features at the same width, the model now fits the principal components as
  projected (`_model_score(..., rescale=False)` for the PCA arm only). Before, every
  component was rescaled to unit variance first. c02 and c11 never rescaled, so the three
  experiments now fit PCA the same way. mRMR, recursive elimination, random and the hybrid
  arms are unchanged.
- **Why:** Adelaide's decision of 2026-10-09 (Chapter 5 writing). The rescaled PCA got worse
  as components were added; hERG Ki svm at 2,000 features: 0.8961 RMSE against c11's 0.5806.
- **Which results:** every c17 file written before this commit. Re-run with the old files moved
  aside (the resume check does not compare the result format): the 15 regression datasets with
  a c17 file, abagym_EGFR abcov_kd adme_mdr1 buchwald_hartwig caco2 ddg_combined
  flab_expression_koenig flab_thermo_tresanco ginkgo_acsins ginkgo_colloidal ginkgo_pr_cho
  ginkgo_titer herg_ki logd qm9. The four classification skip files need nothing.
