# HANDOFF — 49 verified faults in the similarity and noise experiment code

> **STATUS, 2026-08-27.** The two tables below were written BEFORE the repairs. Of the
> 21 rows in sections 2 and 3, **12 are now fully addressed, 7 partly, 2 unchanged** —
> the per-row status is in section 0. Every repair adds a key beside the old one; no
> stored quantity was renamed or deleted. Sections 1, 4 and 5 are current. Section 6 is
> replaced by section 0.

## 0. Per-row status after the 2026-08-26 repairs

Every fix ADDS a labelled quantity beside the original. Old keys keep their old values,
so no stored result file stopped loading.

**Fully addressed (12).** The undefined read-across correlation is stored as NaN rather
than 0.0, and 20 ranking sites across 15 scripts skip non-finite entries. The
random split in the Bayesian-optimisation comparator is replaced by a group-aware
splitter, and the splitter used is recorded per seed. The cliff delta keeps its raw sign
and gains a direction-corrected twin, with a conventions block naming the formula behind
each of the three cliff deltas. The paired per-fold difference in the importance
experiment is stored beside the unpaired one, with the paired fold count and the folds
whose build raised. Base and augmentation pools are screened before the search: an
all-NaN or all-zero matrix is dropped with its reason recorded, and fewer than two
survivors raises. A run where every task failed writes an error key and a per-task reason.
Both cliff reductions are stored, one against unweighted single representations and one
against importance-weighted ones. A failed hybrid search in the cliff and kNN experiments
writes a status record naming the exception, so a file with per-representation numbers and
no hybrid number says which search failed. The blueprint-fetch loop records which models
got a blueprint and which did not. The contamination rate is scored under both fold
schemes and each is labelled. The cut objective is refused at the command line and at
five further points, instead of running a prediction search under the similarity name.

**Partly addressed (7).** The within-fold correlation is stored beside the pooled one,
but the cross-dataset summary still aggregates the pooled value. The tautological
error-source pair is labelled and the label reaches the summary, but the aggregated
difference is still that pair. NaN-skipped twins of every optimum are stored, with the
budgets that had no score, but the summary still reads the original keys. A second
held-out arm scores both sides with unweighted distances, separating the weighting effect
from feature selection, and the original weighted arm stays the one aggregated. The
post-hoc maximum over budgets and neighbour counts is still aggregated, now labelled
selection-optimistic, with the pre-registered and held-out counterparts added as further
series. The saved file now records whether it holds a measurement, but neither resume
check reads that stamp, so an empty file still blocks a resubmission. Precompute failures
are recorded in a structure and read at four call sites.

**Unchanged (2).** The fold-count mismatch in the cross-domain summary comparator, and
the unbacked S12 pointer: the one file that exists for those six datasets holds
`"error": "no_cliffs_found"` and no cliff delta.

## 1. What this document is

**Written:** 2026-08-26 · **Updated:** 2026-08-27 · **Branch:**
`similarity-metrics-study` · **Indexed from:** `SIMILARITY_METRICS_ROADMAP.md` §D


A list of faults found by a code audit and then INDIVIDUALLY VERIFIED against the file
and line named. Every row below was opened and read. Nothing here is a measured
experimental result, a prediction about what a rerun would show, or an opinion about
what should be done. Each row says only what the code computes today.

Counts: **49 confirmed, 3 refuted, 2 already fixed.** The 3 refuted rows are recorded at
the bottom so they are not raised again.

## How each row was checked

An audit read the similarity suite and the shared experiment code and proposed the
faults. A second pass then opened each named file at the named line and either quoted
the line that proves it, quoted a line that disproves it, or marked it unproven. Only
confirmed rows are in the two tables. The quoted evidence for every row is in the
workflow journal for run `wf_2d94726f-8a0`.

---

## 2. SAC — where it stands

SAC is the Spearman correlation between how far apart two molecules sit in a
representation and how different their measured property is.

**Cut as a search objective, 2026-08-26, by the owner.** `_VALID_OBJECTIVES` in
`src/kirby/find_best_hybrid.py:1248` is now `(predictive, similarity)`. The evidence on
both sides is recorded in the comment block at `find_best_hybrid.py:1215-1244`; it says
not to re-derive it.

**The budget split no longer ranks by SAC.** Verified at `src/kirby/hybrid.py:1899-1913`:
under a geometry objective the quantity that decides how the budget is split across
representations is scaffold-grouped kNN activity read-across. A geometry search without
scaffold groups now raises rather than scoring every representation identically
(`hybrid.py:1904-1911`).

**Nothing in `src/kirby/similarity/` selects by SAC.** Verified: no match for SAC in that
package. Feature dropping there is judged by held-out MCS structural fidelity
(`selection.py:110-112`).

### ★ The one live fault SAC left behind

`tests/experiments/similarity_optimized_hybrid.py:863` still defaults
`--sim-objective` to `similarity_sac`. `_is_geometry_objective` at
`find_best_hybrid.py:1251-1256` now returns True only for `similarity`. So running that
experiment with no arguments does **not** raise. The similarity arm falls through to the
`else` branch at line 209-215 and becomes an **RMSE-optimised predictive search**. The
experiment then compares a predictive search against a predictive search and labels one
of them the similarity hybrid. The per-fold rebuild has the matching fault at lines
251-257: under `similarity_sac` it passes a random forest with no scoring function, so
every candidate allocation scores minus infinity. Errors are written into the per-dataset
JSON as an error entry rather than stopping the run (lines 895-898).

**FIXED 2026-08-26.** The default is now activity read-across, and asking for the cut
name raises at the command line and at five further points inside the script. The
experiment had never been run — `tests/results/sim_optimized_hybrid/` was and remains
empty — so no stored result was affected. The owner confirmed on 2026-08-26 that SAC
continuing to be REPORTED is fine; only its use as a search objective was cut.

### SAC is still REPORTED here, which the cut did not touch

Whether SAC should also stop being reported is an open decision for the owner, not a
fault. Confirmed sites:

- `s01_similarity_baseline.py:114, 124, 152`
- `s02_hybrid_similarity.py:140, 234, 249, 282, 296, 348, 394`
- `s08_optimal_dimension.py:165, 238, 298, 313, 338`
- `similarity_helpers.py:68, 193`
- `f18_distance_metric_sweep.py:283, 439, 463`
- `f19_objective_scoring.py:95, 143, 206`
- `f21_budget_split_signal.py:72, 87`
- `find_best_hybrid.py:1259`, `noise_mitigation_methods.py:229`
- `similarity_experiments.py:110, 419, 549, 1761` (the superseded combined runner)
- `read_sim_optimized_results.py:71-75`

`s13_clustering_retrieval.py` computes no SAC number — checked, no match.

Remaining `similarity_sac` references are the raising guard at
`find_best_hybrid.py:5420-5427`, the retained constant at `hybrid.py:57` and
`find_best_hybrid.py:1246`, cache tier names at
`similarity_optimized_hybrid.py:750-751`, and comments.

---

## 3. Faults that make a reported number wrong (as first written, see §0 for status)

| file | line | what comes out wrong |
|---|---|---|
| `tests/experiments/similarity/s09_cross_dataset_summary.py` | 139-153 | The kNN delta s09 prints and Wilcoxon-tests is the best of every budget and k, inflated by that selection, and s05's pre-registered honest delta never reaches the summary. |
| `tests/experiments/similarity/s09_cross_dataset_summary.py` | 103-108 | The cliff RMSE reduction percentage credits importance weighting plus hybrid mixing together, because the importance-weighted single-representation cliff RMSE that s04 measured is never used as the baseline. |
| `tests/experiments/similarity/s09_cross_dataset_summary.py` | 84-239 | Every cross-dataset number and every Wilcoxon p-value is computed from the selection-optimistic arm, while the held-out arm each experiment already computed is silently discarded. |
| `tests/experiments/similarity/s06_applicability_domain.py` | 264-286 | The applicability-domain delta s09 reports gives the hybrid a built-in advantage, since the errors it is scored against come from a model trained in the hybrid space itself. |
| `tests/experiments/similarity/s06_applicability_domain.py` | 247-267 | Between-fold differences in distance scale enter the correlation as real signal, so the reported Spearman is not the within-fold correlation the docstring implies. |
| `tests/experiments/similarity/s08_optimal_dimension.py` | 428-434 | Yes: the score is present-but-NaN (the `_default` never fires), and since `max`/`min` seed on the first item and every NaN comparison is False, a NaN at the smallest sorted budget is reported as the optimal budget for that model or for SAC. |
| `tests/experiments/similarity/s08_optimal_dimension.py` | 236 vs 337 | The held-out SAC delta in `results['sac_honest']` credits the hybrid with importance weighting that no single-representation baseline receives, so the reported hybrid-minus-best-single SAC gap mixes weighting with feature selection. |
| `tests/experiments/similarity/s07_noise_bridge.py` | 103 | It runs leave-one-scaffold-group-out, so it produces one fold per distinct scaffold group (typically hundreds) instead of the configured 3-5, and each fold's rate is a tiny-denominator fraction, usually exactly 0.0, 1.0, or NaN. |
| `tests/experiments/similarity/s11_bayesian_optimization.py` | 550 vs 826 | The best_single comparator representation is picked on a shuffled random split, so scaffold leakage inflates its selection R2 and the hybrid-versus-best_single BO curves compare two different split regimes. |
| `tests/experiments/similarity/s12_cross_domain_cliffs.py` | 458 | On an RMSE dataset a positive `hybrid_cliff_delta` means the hybrid's cliff RMSE is higher, i.e. the hybrid lost, the opposite of the sign convention used by `hybrid_cliff_delta_honest` printed a few lines below. |
| `tests/experiments/similarity/s10_importance_downstream.py` | 382-387 | The two arms append NaN from independent try/except blocks at lines 368 and 380, so the reported delta can subtract means taken over different fold subsets; the same pattern repeats at 241-246, 440-445, 506, 564, and 662-670. |
| `tests/experiments/similarity/similarity_helpers.py` | 832-836 | An augmentation featuriser that failed on every sample becomes an all-zero matrix, is passed to precompute_hybrid_searches at line 865, receives budget, and scores as a valid representation. |
| `tests/experiments/similarity/similarity_helpers.py` | 817-822 | A base featuriser at 100 percent NaN prints one warning line, becomes all-zero, and still enters the rep pool at line 861 where it draws budget and is scored. |
| `tests/experiments/similarity_metrics/simeval.py` | 160 | An undefined read-across Spearman is stored as exactly 0.0, indistinguishable from a real zero correlation; a scan of all 29 JSON files in that directory found 0 stored read-across values equal to 0.0. |
| `tests/experiments/complexity/c14_cross_domain_summary.py` | 448-449 (empty return), 140-146 and 249 (fold counts) | A total failure saves a file with no error key, and a representation scored on two surviving folds is ranked against one scored on five while the file states the configured fold count. |

---

## 4. Faults that let a failure look like a finished run (as first written, see §0 for status)

| file | line | what comes out wrong |
|---|---|---|
| `tests/experiments/_shared.py` | 4828-4859 (writer); resume test at tests/experiments/similarity/similarity_helpers.py:914 and tests/experiments/noise/noise_helpers.py:296 | An experiment that returns empty sections still writes its dataset JSON, and every later resubmission skips that dataset forever unless force_rerun is passed. |
| `tests/experiments/_shared.py` | 3915-3918 and 3997-4002 | A failed augmentation condition contributes no hybrid score, and a failed distance-tier precompute leaves zero distance searches, with only a printed line as the record. |
| `tests/experiments/similarity/s04_activity_cliffs.py` | 111, 146-148, 295 | Both hybrid cliff scores, unweighted and weighted, are absent from the saved file while the per-representation scores look complete. |
| `tests/experiments/similarity/s05_knn_prediction.py` | 95-97, 127 | Every budget arm's hybrid kNN score is missing and the honest re-score block is empty, with per-representation baselines still filling the file. |
| `tests/experiments/noise/n07_remediation.py` | 275-301 (loop), 301-303 (guard) | With one model out of the configured set holding a blueprint the run proceeds, and the remediation strategy scores for every other model are silently absent. |
| `tests/experiments/similarity/s04_activity_cliffs.py` | 66-71 | The skip names S12 as the replacement, but zero of the six datasets — flab_thermo_jain, ginkgo_acsins, ginkgo_colloidal, ginkgo_hic, ginkgo_pr_cho, ginkgo_titer — have any S12 cliff result. |

---

## 5. Fixed before this document was written, do not re-raise

- **`s02_hybrid_similarity.py`** — `best_sac` was seeded with the best single
  representation's SAC and only ever raised, so a hybrid that lost reported a delta of
  exactly 0. It is now seeded with `None`, a losing hybrid reports its real negative
  delta, and a search that failed writes `best_sac: null` plus a `hybrid_status` string
  instead of the single representation's number.
- **`s09_cross_dataset_summary.py`** — six quantities were reported as a mean and standard
  deviation across datasets. The mean and standard deviation are gone; the per-dataset
  values are named and printed individually. The paired Wilcoxon test over those values
  is kept, because it is a sign test rather than a summary statistic.

Also fixed the same day, outside this audit: the MCS pair caches now store a hash of the
molecule list their row numbers were built against and refuse any other list; the
antibody geometry features raise instead of returning zeros when ImmuneBuilder is absent;
and the cross-dataset aggregator refuses and exits non-zero when no dataset produced a
metric.

---

## 6. Refuted — checked and NOT true

- **`s13_clustering_retrieval.py` reports SAC.** It does not. No match in the file.
- **`src/kirby/similarity/` selects by SAC.** It does not. No match in the package.
- **Every quantity `s09` aggregates is a whole-dataset statistic.** Three of the six are,
  but the `s05` and `s08` inputs are cross-validation means picked in-sample
  (`s05:205`, `s08:257`), which is a different fault. `s06` has no held-out alternative
  at all, so there is nothing there for `s09` to have read instead.

---

## 7. What was NOT done — SUPERSEDED, see §0

- ~~No fault in sections 2 or 3 has been fixed.~~ **Out of date the day it was written.**
  See §0: 12 fully addressed, 7 partly, 2 unchanged.
- ~~No experiment was rerun.~~ The noise-cleaning experiment WAS rerun on herg_ki, logd
  and adme_solubility under a scaffold split of the molecules. Detection beat the
  count-fingerprint baseline on 3 of 3 (+0.0238, +0.0050, +0.0215); recovery on 1 of 3
  (-0.0114, -0.0008, +0.0233), against in-sample +0.0281, +0.0365, +0.0083.
  Nothing else was rerun.
- `tests/experiments/noise/_similarity_hybrid_space.py:88` still loads
  `mcs_pairs_<dataset>.npz` directly and feeds the pair row numbers straight into the
  descriptor drop at line 95, with no check that those row numbers belong to the molecule
  list in hand. Those three files store no molecule-list hash.
- The 15 files in `tests/results_server/geometry_survey/` still hold their empty result
  blocks. No server cache was inspected or cleared.
