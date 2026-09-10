# Does a complementarity-aware feature selector help PREDICTION? — NO

**Date:** 2026-07-18 · **Branch:** `similarity-metrics-study` · **Owner:** Adelaide
**Code:** `tests/experiments/feature_selectors/selector_prediction.py`
**Raw results:** `tests/experiments/feature_selectors/selector_prediction.json`, `full_run.log`
**No changes to `hybrid.py` / `find_best_hybrid.py`** (rebuild-roadmap lock respected).

---

## The question

The similarity-metrics study found that two *bespoke* feature rankers beat everything
else on its yardsticks: **R9 greedy-forward** (rewards complementarity) and
**R5 RReliefF** (local / nearest-neighbour). That was on a *similarity* objective.

KIRBy's prediction hybrid selects features with `compute_feature_importance`
(`hybrid.py:2382`) and then takes top-k per rep (`hybrid.py:1107`). Both steps are
**marginal**: a feature's score is computed with its rep in isolation and does not
depend on what has already been chosen, so the selector can never prefer a feature
*because it supplies the signal the already-chosen features lack*. That is the
complementarity gap in `project_hybrid_marginal_allocation`.

Prior knob work (`project_knob_sensitivity_2026_07_16`) measured the importance knob
as weak (0.0001–0.012), **but only ever compared marginal methods to each other**
(lightgbm_gain vs treeshap vs deeplift). So the open question was narrow and real:

> Does a **non-marginal** selector beat marginal importance on the actual prediction
> task, at a fixed feature budget?

## Answer

**No.** Across both datasets and both budget arms, **every** non-marginal selector
loses to the plain marginal `lightgbm_gain` baseline. Nothing came close to winning.

Pooled over both datasets (10 scaffold-grouped folds), paired per-fold Δρ vs
`lightgbm_gain`:

| selector | arm | Δρ | win/loss | p (Wilcoxon) |
|---|---|---|---|---|
| r9_greedy_omp (COMPLEMENTARITY) | top4×100 | **−0.0058** | 3/7 | 0.232 |
| f_regression (marginal, sklearn) | top2×100 | −0.0078 | 3/7 | 0.160 |
| f_regression (marginal, sklearn) | top4×100 | −0.0079 | 4/6 | 0.232 |
| mrmr (REDUNDANCY-AWARE) | top4×100 | −0.0086 | 3/7 | 0.193 |
| r5_rrelieff (LOCAL/NN) | top4×100 | −0.0089 | 2/8 | 0.064 |
| mrmr (REDUNDANCY-AWARE) | top2×100 | −0.0115 | 1/9 | 0.084 |
| r9_greedy_omp (COMPLEMENTARITY) | top2×100 | −0.0161 | 2/8 | 0.105 |
| r5_rrelieff (LOCAL/NN) | top2×100 | −0.0193 | 3/7 | 0.084 |

All eight comparisons are negative. The best non-marginal result (greedy at top4×100,
−0.0058) is a loss, not a win. On herg_ki the losses are consistent and one-sided
(0/5 folds at top2×100 for greedy, RReliefF, and mRMR); on logd everything sits inside
fold noise.

**This closes the gap the knob-sensitivity work left open.** Widening the importance
menu from "marginal methods only" to "marginal + complementarity + local + redundancy-
aware" does not turn importance into a real lever — it only adds ways to lose. The
existing lock (keep one cheap marginal scorer, `lightgbm_gain`) is now supported by
direct evidence rather than by absence of evidence.

## Two positive side-findings (leak-free, both datasets, 5/0 and 5/0)

Reference arms measured on the same folds, with the best single rep chosen by
**train-only inner CV** (no test peeking):

| comparison | herg_ki | logd |
|---|---|---|
| full 4-rep concat (2423 feats) − best single rep | **+0.0184** (5/0 folds) | **+0.0184** (5/0 folds) |
| top4×100 marginal selection (400 feats) − best single rep | +0.0098 (4/1) | +0.0259 (5/0) |

1. **The hybrid genuinely beats the best single rep** on both datasets, every fold, on
   a decorrelated 4-rep pool. That is the rebuild roadmap's target relationship, and it
   holds without any clever selector.
2. **The concentrate knob (top2 vs top4) moves more than the selector knob.** Dropping
   from 4 reps to 2 costs the baseline ~0.005 (logd 0.826→0.810) and hurts every
   non-marginal selector far more (up to −0.041). Consistent with the knob-sensitivity
   ordering: allocation/top-N > importance.

## Why the similarity result did not transfer (the interesting part)

The similarity rankers score features on a **pair-difference** problem — design matrix
`|f_i − f_j|` over sampled pairs, target `|Δy|` or MCS distance. Prediction is the
direct problem `f → y`. So the methods had to be re-derived, not copied:

* **R9 greedy-forward → matching pursuit on `(X, y)`**: repeatedly take the feature
  most correlated with the *residual* of y, then project that direction out.
  Complementarity-aware by construction. (Implemented without the marginal pre-filter
  the similarity version used, so the result is not contaminated by a marginal screen.)
* **R5 RReliefF** needed no translation — `rankers.r5_rrelieff` already operates on
  `(X, y)`. Reused verbatim.
* **mRMR** added because the rebuild roadmap names it as *the* sanctioned
  redundancy-aware candidate; it is the natural third non-marginal method.

The plausible reason for the non-transfer: **LightGBM already handles redundancy
downstream.** A greedy/mRMR selector spends its budget de-duplicating features, but a
gradient-boosted tree ensemble is largely insensitive to redundant inputs — it just
ignores the duplicates. So the selector pays a real cost (it drops individually strong
features to buy decorrelation) and collects no benefit. In the similarity study there
*was* no downstream model to absorb redundancy: the selected features went straight
into a distance computation, where every redundant feature directly re-weights the
metric. That is exactly where complementarity pays, and it is why the same selector
wins there and loses here.

## Method (why these numbers are trustworthy)

* **Pool** = the same 4 complementary reps as the similarity study
  (`ecfp4_count`, `atompair_count`, `desc`, `maccs`; 2423 features) via
  `_data.load_prepared`, so the two studies are directly comparable and no server data
  is needed. herg_ki n=1415; logd n=1427 (the study's scaffold-grouped subsample).
* **Selection happens inside each training fold.** Fitting a selector on all the data
  and then cross-validating leaks labels; every selector here is fit on train only.
* **Scaffold-grouped `GroupKFold`, 5 folds. No random splits.**
* **The rep set is chosen identically for every selector** (train-only inner 3-fold CV),
  so the only thing varying between arms is *which features* are picked inside each rep.
  Rank-based selectors emit per-rep orderings that are not comparable across reps, so
  letting the selector also pick the rep set would have confounded the comparison.
* **Budget held fixed** — per-rep 100 features × top-N reps (the allocation knob, as in
  the similarity study and the original working hybrid at c4afb0e). A selector can only
  win by choosing *better* features, not more.
* Downstream model = lightgbm (roadmap-locked, ridge removed). Reported as **paired
  per-fold deltas**, not just means.

**One correction made mid-study, disclosed:** the first run chose the top-2 rep set
using test-fold scores. That leaked, and it changed which reps were picked (fold 0 of
herg_ki: `atompair_count` under the leak vs `ecfp4_count` under inner CV). Fixed by
selecting the rep order on train-only inner CV and re-run from scratch; all numbers
above are from the clean re-run. The leaky run is kept as
`full_run_LEAKY_top2.log` / `selector_prediction_LEAKY_top2.json` for comparison — its
conclusion was the same, so the leak did not manufacture the negative result.

**Compute:** ~475s (herg_ki) + ~415s (logd) local, no server, no GPU.

## Limits of this result

* Two molecular regression datasets, one downstream model (lightgbm). The
  redundancy-absorption argument above predicts the result would be **weaker for a
  linear or distance-based downstream model**, where redundancy is not free — untested.
* One budget scale (100/rep). The roadmap's budget work suggests ≥500 matters; a
  complementarity selector might behave differently when the budget is large enough
  that marginal top-k starts stacking near-duplicates.
* No antibody dataset (the similarity study's antibody arm is blocked on dataset
  choice; `flab_thermo_tresanco` is staged locally and would be the natural third).

## Recommendation

**Do not wire a complementarity-aware selector into `compute_feature_importance`.**
The similarity-study finding is real but is specific to a similarity objective; it does
not transfer to prediction. Keep the single cheap marginal scorer (`lightgbm_gain`) and
spend the search budget on the levers that actually move — top-N / concentration and
budget.

Open decision 4 in `SIMILARITY_METRICS_ROADMAP.md` ("if a similarity-aware ranker wins
standalone → propose wiring it into `compute_feature_importance`") should be marked
**resolved for the prediction path, NO** — while remaining live for the similarity path,
where R9/R5 did win.
