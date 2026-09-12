# Antibody geometry survey — result

Measured 2026-09-11 from `tests/results/geometry_survey/`, read by
`tests/analysis/analyse_geometry_survey.py`.

## What was run

`tests/experiments/structural_geometry_survey.py` on 8 antibody datasets:
abcov_kd, ddg_combined, flab_thermo_tresanco, ginkgo_acsins, ginkgo_colloidal,
ginkgo_hic, ginkgo_pr_cho, ginkgo_titer.

Per dataset: the 10 representations of that dataset's rep-pruning pool
(`DATASET_REPS[ds]['general']`), 7 geometry feature groups plus their
concatenation, rf and mlp, 5 folds, group-aware CV. Structures from
ABodyBuilder2, or NanoBodyBuilder2 where the dataset is single-domain.

Each representation was scored on its own, so every number below is a
difference against that same representation without geometry.

## Concatenating geometry onto a tuned representation

Pre-specified cell: `all_geometry` added to whichever representation scores
best alone on that dataset, same model. Chosen without reference to any
augmentation result.

| dataset | best representation | model | alone | with all_geometry | delta |
|---|---|---|---|---|---|
| abcov_kd | abroberta | rf | 0.4125 | 0.3374 | −0.0751 |
| ddg_combined | onehot | rf | 0.2611 | 0.1727 | −0.0884 |
| flab_thermo_tresanco | saprot | rf | 0.6266 | 0.6032 | −0.0234 |
| ginkgo_acsins | ablang2 | mlp | 0.5584 | 0.5484 | −0.0100 |
| ginkgo_colloidal | esm3 | mlp | 0.2102 | 0.0328 | −0.1774 |
| ginkgo_hic | esm2_concat | rf | 0.3267 | 0.3409 | +0.0142 |
| ginkgo_pr_cho | prot_t5 | rf | 0.4610 | 0.4041 | −0.0569 |
| ginkgo_titer | esm2_entropy_top20 | mlp | 0.3919 | 0.3433 | −0.0486 |

1 up, 7 down. Median −0.0528. Two-sided sign test p = 0.0703. With 8 datasets
the smallest attainable p is 0.0078.

Across the whole grid (representation × geometry group × model) the counts are
near even and no group reaches significance on a per-dataset sign test. The two
leaning positive are the VH/VL groups, interface 77/59 and angles 75/61, and
both are undefined for single-domain datasets so they count on 7 datasets.

## Selection

The best augmentation per dataset ranges +0.0709 to +0.2286. Each is the
maximum of 120 to 160 comparisons on that dataset. ginkgo_pr_cho is
−0.0569 in the pre-specified cell and +0.2155 as the best of 160 — the same
dataset and the same geometry group, a different representation and model.

## Geometry scored on its own

ρ 0.13 to 0.37 standalone. Below the best representation on 7 of 8 datasets,
gaps −0.048 to −0.311. On ginkgo_hic it is above: 0.3388 against 0.3267.

## Scope and what is not measured

The other 12 datasets in the survey's registry have no entry in `DATASET_REPS`
and therefore no precomputed representations, so they did not run.

The `kirby_hybrid` arm, in which the search selects features rather than
geometry being concatenated onto a fixed representation, had not run at the
time of this measurement: `find_best_hybrid` takes `return_fitted`, it defaults
to False, and the survey did not pass it, so `fitted_results` came back empty
on every job. Fixed in `84fd85a`. Nothing here measures whether selecting among
geometry features helps.

## Open defects, recorded 2026-09-12

**The hybrid arm's score is not held out.** `find_best_hybrid` selects features
using every row, and the resulting matrix is then scored by 5-fold CV over
those same rows, so each held-out fold was seen during selection. The arm is
marked `hybrid_selection_fitted_on_all_rows` in every result file it writes and
the analysis prints a warning beside it. Its number is optimistic by an
unmeasured amount and is not comparable to the pre-specified cell above, where
both sides are scored the same way. A held-out version needs the selection
redone inside each fold: `create_hybrid` costs about 336s per call, so 5 folds
across 8 datasets is roughly 40 calls.

**Resume no longer counts the March files as done.** The 15 files that run
wrote carry no `status` key and have `geometry_baselines` and `augmentations`
both empty, so `--skip-existing` treated them as finished work and skipped them
permanently. It now redoes any file with no scores in it, whatever that file
says about itself.

**The internal evaluator axis has lost `mlp`.** `DEFAULT_INTERNAL_EVALUATORS`
is `['match', 'rf', 'lightgbm', 'dnn']`. `'match'` maps to whichever model is
being evaluated downstream, including `mlp` and `svm`
(`src/kirby/hybrid.py:5509-5530`), and the searched model list is
`rf, xgboost, lightgbm, svm, dnn` — so `svm` still reaches the axis through
`match`, and `mlp` does not, because no searched model maps to it. Changing
`DEFAULT_INTERNAL_EVALUATORS` invalidates every cached search, so it is left as
found and recorded here.

## Measured resources

Peak RSS 0.38 GB on abcov_kd with ten representations loaded; `/usr/bin/time -v`
reports 391 MB. The arrays themselves are 0.03 GB — the footprint is the
interpreter and libraries, not the data. Every result file now records
`peak_rss_gb`, `base_reps_gb` and `geometry_gb`.
