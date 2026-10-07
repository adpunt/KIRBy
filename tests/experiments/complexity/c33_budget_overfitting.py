#!/usr/bin/env python3
"""C33: Does the search overfit at feature budgets above the molecule count?

Paper: Complexity

THE QUESTION
    Until 2026-09-10 the engine filtered the budget list to
    `[b for b in budgets if b <= n_samples]`. That cap was found to be a defect
    on 2026-07-14: the base representation was scored at its full native width
    while the hybrid was held to the molecule count, so p>n was enforced on one
    side only. The cap was removed and `allow_budget_exceeds_n` now defaults
    True.

    Nothing has measured what happens above the line. The 2026-07-14 evidence
    was herg_ki, 1,412 molecules, where the hybrid crossed zero at 250 features
    and plateaued between 500 and 1,000 — every one of those budgets is BELOW
    the molecule count. And no cached search can answer it, because every
    search built under the cap stops at exactly its own molecule count.

    So on a 672-molecule dataset, is a 2,000-feature budget a real gain or a
    selection the held-out folds do not support?

WHAT IS MEASURED, AND WHY IT NEEDS NO NEW MACHINERY
    nested_cv_evaluate already rebuilds EVERY budget's winning blueprint on
    outer-train and scores it on the untouched outer test fold, then reports
    per model per budget in by_model_budget (find_best_hybrid.py, in
    nested_cv_evaluate; line numbers move). This experiment makes one nested
    run and reads that dict. It runs no second search.

    reselected_mean   the held-out score at that budget
    search_mean   what the inner search said the same budget was worth
    bias          signed so POSITIVE MEANS THE INNER SEARCH OVER-PROMISED, in
                  both metric directions

    Overfitting at high budgets is exactly: reselected_mean peaking below the
    molecule count and falling above it, and/or bias growing with budget.

DOUBLE DESCENT — WHY THE BUDGET GRID IS NOT JUST FULL_BUDGETS
    Test error does not have to rise monotonically once features outnumber
    rows. The classical shape peaks where the number of parameters equals the
    number of training rows — the interpolation threshold — and then falls
    again as parameters grow past it. If that is what happens here, a cap at
    the molecule count would put every dataset at exactly the worst point on
    its own curve, which is the opposite of the intended fix.

    FULL_BUDGETS cannot see it. Its steps are 10, 25, 50, 100, 250, 500, 1000,
    2000 features, fixed for every dataset, so on 451 training rows the
    threshold falls in the gap between 250 and 500 and no budget sits near it.
    So this experiment ADDS budgets placed at ratios of the dataset's own
    smallest outer-fold training row count — 0.5x, 0.75x, 1x, 1.25x, 1.5x, 2x
    — which is where the peak would be if there is one.

    ⛔ Those extra budgets are part of the search cache key, so c33 does NOT
    reuse a precompute warmed on FULL_BUDGETS alone. Pass
    config['interpolation_ratios'] = [] to drop them and reuse the cache; the
    double-descent question then cannot be answered, only the cap question.

    ⛔ FOR FOUR OF THE FIVE DEFAULT MODELS THE THRESHOLD IS NOT AT budget =
    training rows, and an earlier version of this docstring said it was. The
    parameter counts, measured 2026-09-10:

      dnn   hidden_sizes=(256,128,64), so budget x 256 first-layer weights on
            top of a FIXED 41,024 in the hidden layers. At budget 10 against
            281 training rows that is already ~43,600 parameters — 155x
            overparameterised at the smallest budget on the grid. Its own
            threshold sits near budget 1.
      svm   SVR/SVC are fitted in the dual: at most one coefficient per
            training molecule. It can NEVER be overparameterised, at any
            budget.
      rf    100 trees, max_depth=None, min_samples_leaf=1, so each tree can
            grow one leaf per training row. Also always past its threshold.
      xgboost / lightgbm  features are sampled per split; capacity is trees,
            depth and leaf count.

    So a flat shape from those five is NOT evidence that double descent is
    absent. It is what you get when every model sits on the far side of its own
    threshold across the whole grid.

    REMOVED 2026-10-05: 'linear'. It was added as the one model whose parameter
    count equals the budget (Ridge(alpha=1e-6) / LogisticRegression(C=1e6)).
    It is ridge under another name and is outside the dataset's two decided
    models, and ridge is removed everywhere (handoff 2.1 R1, 2.3). c33 now
    scores only the two decided models, so its shape column answers the cap
    question; whether any model can answer the double-descent question here is
    the owner's to decide.

THE TWO NUMBERS THE ANSWER TURNS ON, per model:
    the budget each outer fold actually PICKED, and the budget with the best
    reselected_mean. If the search picks 2,000 features on a 672-molecule dataset
    while 250 is honestly best, the process overfits and a cap has a measured
    reason — per dataset by its own molecule count, never a budget dropped from
    FULL_BUDGETS everywhere. If the honest curve keeps climbing past the
    molecule count, removing the cap was right.

READ IT PER MODEL. rf with max_features='sqrt' draws about 45 of 2,000
features at each split, so more features than molecules barely touches it. The
curves may disagree, and that disagreement is the result.

ALIGNED 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4).
    Models: the dataset's two decided models (decided_grid.decided_models).
    Search: one search per outer fold on that fold's training rows
    (nested_cv_evaluate, cache namespace ocv<n>f<i>s<seed>), on the decided
    importance methods, allocation methods, internal evaluators and
    representation counts (own_search_kwargs with the real dataset name),
    pruning off. The budget axis is c33's own list, because the budget is what
    c33 measures: the decided budgets plus FULL_BUDGETS, 4,000 and 8,000 and
    the ratio budgets. It does not go through run_nested_cv, which narrows the
    budget list to the decided range.
    DECIDED 2026-10-06 (Adelaide delegated): c33 keeps its own budget list,
    including the budgets outside the decided range, because the budget is
    the question c33 asks.
    Selection: per decided model per outer fold, the 2 best base
    representations and the 2 best hybrids, both ranked on that fold's own
    search (training rows only); a second hybrid whose selected features equal
    the first's is skipped for the next. Each is scored on the held-out fold.
    Outer folds: config['n_splits'] (was 3 here, owner 2026-09-29); 1 is one
    held-out split, and every spread then reads "one fold, no spread".
    Augmentation: not searched. Step 8 keeps it only on the distance search of
    ginkgo_colloidal and ginkgo_hic, and c33 runs the general search.
    Removed: 'linear' (ridge), the distance nested run and its distance1 slot
    (handoff 2.3 item 1: distance hybrids only where the experiment measures
    neighbours or distances), and general2 read through c14.

BUDGETS. C33_BUDGETS is FULL_BUDGETS plus 4,000 and 8,000. Measured on
humanness_therapeutic 2026-09-10: rf and svm both put their best honest score
at the LARGEST budget tested, and a best score at the edge of the range means
the range is too short. A budget at or above a dataset's pool width asks for
every feature; create_hybrid takes min(budget, n_features). That can happen on
any dataset whose pool is narrower than the budget, not only tcga_brca_pam50
(corrected 2026-10-05). Asking is not getting: create_hybrid also stops at the
last feature scoring above zero, so the width built can stay far below the
pool. The SATURATED line and 'saturated_by_width_built' therefore come from
the width each outer fold actually built, not from the budget.

NOT c16. c16 asks whether the hybrid's budget subsumes model-level
regularization, sweeping regularization strength through build_fold_hybrid +
fold_reselection. It does not report the honest per-budget curve.

USAGE
    python tests/experiments/run_dataset.py complexity herg_ki --experiments c33
    python c33_budget_overfitting.py --dataset flab_thermo_tresanco --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import spearmanr

from experiments._shared import (
    first_outer_folds,
    own_search_kwargs,
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric,
    top_base_reps, top_hybrids,
    build_fold_hybrid, fold_hybrid_scoring_kwargs,
    STANDARD_N_BASE, STANDARD_N_HYBRIDS,
    FULL_BUDGETS,
)
# 2026-10-05: the feature signature fold_standard_selection's identical-features
# check uses. c33 ranks its hybrids on its own per-fold search (the budget list
# is c33's, not the decided one), so it runs the check itself, on the same build
# it scores, instead of calling fold_standard_selection, which would run a
# second search per fold.
from experiments._shared import hybrid_feature_signature
# 2026-10-05: no longer imports rule_general2_outer / rule_slot_vs_single from
# c14. The general2 and distance1 slots they served are replaced by the
# standard selection below (2 base + 2 hybrids per decided model per fold).

# Bumped whenever what one result file holds changes, so a reader can tell a
# file written by this version from an older one. 2026-10-05: per-fold standard
# selection, decided models, no 'linear', no distance run, folds from
# config['n_splits'].
RESULT_IDENTITY = 'c33_per_fold_decided_grid_2026_10_05'
ONE_FOLD_NOTE = 'one fold, no spread'


# --------------------------------------------------------------- arithmetic
# Copied from f26_redundancy_penalty.py rather than imported: a complexity
# experiment importing a framework one couples two suites that are run
# separately. Same arithmetic nested_cv_evaluate uses for
# delta_mean_of_folds, applied at ONE budget, which it does not report.
def _paired_delta(hybrid_folds, baseline_folds, lower_is_better):
    """Mean of the per-fold (hybrid - best single rep) differences."""
    per_fold, paired = [], []
    n = max(len(hybrid_folds or []), len(baseline_folds or []))
    for f in range(n):
        hv = hybrid_folds[f] if hybrid_folds and f < len(hybrid_folds) else None
        bv = baseline_folds[f] if baseline_folds and f < len(baseline_folds) else None
        if hv is None or bv is None or not np.isfinite(hv) or not np.isfinite(bv):
            per_fold.append(None)
            continue
        d = (bv - hv) if lower_is_better else (hv - bv)
        per_fold.append(float(d))
        paired.append(float(d))
    return (float(np.mean(paired)) if paired else None), per_fold, len(paired)


def _fold_spread(per_fold):
    """Standard deviation across the outer folds' deltas, ddof=1."""
    vals = [float(v) for v in (per_fold or [])
            if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return None
    return float(np.std(vals, ddof=1))


def _f(x):
    """float, or None when the number is missing or not finite."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _better(a, b, lower_is_better):
    """True when a is the better score of the two."""
    return (a < b) if lower_is_better else (a > b)


# 2026-10-05: _outer_train_rows removed. run() builds the outer folds once
# with nested_cv_outer_folds and reads the training row counts off them.


# ------------------------------------------------------- interpolation grid
# Where the classical peak sits: features equal to training rows. The ratios
# straddle it so the curve has points on both sides of the threshold and one
# on it. Under-parameterised at 0.5x and 0.75x, at the threshold at 1x,
# over-parameterised at 1.25x, 1.5x and 2x.
# FULL_BUDGETS stops at 2,000. On the 2026-09-10 humanness_therapeutic run rf
# and svm both scored best AT 2x training rows, the largest budget tested, and a
# best score at the edge of a range means the range is too short. 4,000 is under
# every local pool width; 8,000 is above tcga_brca_pam50's 4,732, where it
# asks for every feature and is kept, not dropped. Owner decision 2026-09-10.
# Since 2026-10-05 a budget is reported SATURATED only when the width built
# reached the pool width on every outer fold (saturated_by_width_built).
C33_SHIPPED_EXTRA_BUDGETS = (4000, 8000)


def _local_debug_run():
    """True on a laptop run, i.e. when KIRBY_LOCAL_GRID is set.

    The cluster never sets it, so every default that differs below is taken
    only locally and the server run is byte-for-byte the shipped one.
    """
    return bool(os.environ.get('KIRBY_LOCAL_GRID'))


def _env_list(var, default, cast):
    """A comma list from the environment, or `default`. 'none' means empty."""
    raw = os.environ.get(var)
    if raw is None:
        return [cast(v) for v in default]
    raw = raw.strip()
    if raw == '' or raw.lower() == 'none':
        return []
    return [cast(p.strip()) for p in raw.split(',') if p.strip()]


# ⛔ LOCAL COMPUTE CUT, NEVER A FINDING. KIRBY_C33_EXTRA_BUDGETS names the
# budgets c33 adds on top of FULL_BUDGETS. It defaults to the shipped 4,000 and
# 8,000 on the cluster and to none under KIRBY_LOCAL_GRID.
#
# WHY, MEASURED 2026-09-25 on flab_thermo_tresanco (672 antibodies — every one
# of them; no row is cut here or anywhere else in this file). The local grid in
# env.sh asks for 2 budgets x 2 models x 3 allocation methods = 12
# configurations per inner search. c33 printed this instead:
#     budgets [100, 224, 250, 336, 448, 560, 672, 896, 4000, 8000]
#     models  ['rf', 'svm', 'linear']
#     Evaluating 90 configs (all 3 folds)
# 90 configurations, 7.5x what the grid asked for, and the run died there
# without reaching configuration 1 of 90.
#
# WHAT DROPPING 4,000 AND 8,000 COSTS. The pool on this dataset is 5 000
# features wide after the local per-representation width cap, so 8,000 was
# already flagged SATURATED — at or above the pool width create_hybrid takes
# min(budget, n_features) and the arm stops selecting. Both of those budgets
# build a hybrid 4,000-5,000 features wide against 448 training rows. Losing
# them loses the top of the curve: the 2026-09-10 humanness_therapeutic
# question of whether the best honest score still sits at the largest budget
# tested cannot be asked locally. The largest budget left is 2x the training
# rows, which is the top of the over-parameterised region the ratios below
# already cover. Restore them with
#     KIRBY_C33_EXTRA_BUDGETS=4000,8000
C33_EXTRA_BUDGETS = _env_list(
    'KIRBY_C33_EXTRA_BUDGETS',
    () if _local_debug_run() else C33_SHIPPED_EXTRA_BUDGETS,
    int)
C33_BUDGETS = list(FULL_BUDGETS) + list(C33_EXTRA_BUDGETS)

# 2026-10-05: C33_MODELS and C33_EXTRA_MODELS ('linear', i.e. Ridge with
# alpha=1e-6) are removed. Models are the dataset's two decided models only
# (decided_grid.decided_models), and ridge is removed everywhere.

C33_SHIPPED_INTERPOLATION_RATIOS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)

# ⛔ LOCAL COMPUTE CUT, NEVER A FINDING. Six ratios on the cluster, two under
# KIRBY_LOCAL_GRID: 1.0x and 2.0x the training row count. Both are ratios the
# shipped six already contains.
#
# WHY THESE TWO AND NOT ANY OTHER TWO. UNDER_MAX and INTERP_MAX below cut
# under at < 0.75x rows, interpolation at 0.75-1.5x and over at > 1.5x. 1.5x is
# INSIDE the interpolation band, not above it, so 1.5x cannot serve as the over
# point: a set of 0.5/1.0/1.5 leaves the over region empty, _pick('over')
# returns nothing, _double_descent's fall is None and the shape column reads
# inconclusive whatever the curve did. 1.0x is the only ratio in the
# interpolation band and 2.0x the only one above it, so these two are forced.
# The under region needs no ratio of its own: the local grid's own budgets, 100
# and 250 features against 448 training rows, are 0.22x and 0.56x, both under.
# On flab_thermo_tresanco that gives budgets 100, 250, 448, 896 in place of
# 100, 224, 250, 336, 448, 560, 672, 896, 4000, 8000.
#
# WHAT IT COSTS. The curve keeps one point in each of the three regions and
# loses the second point in each: 0.5x, 0.75x, 1.25x and 1.5x of the training
# rows are not scored. Rise and fall keep their sign, because each is read off
# one budget per region. What goes is the resolution of WHERE the peak sits —
# it is now located to within half the training row count instead of a quarter
# — and any check that two budgets inside one region agree.
#
# WHY IT WAS CUT, MEASURED. See the extra-budget note above for the 90-config
# starting point. At three ratios the search was still 45 configurations per
# outer fold and ran at 78 s per configuration on this laptop (20 done in 26
# minutes, 31% of one core, load average 437, 26 other python processes), so
# outer fold 1 alone would have taken 58 minutes against a 40-minute cap. Two
# ratios take it to 24 configurations per outer fold. Restore the shipped six
# with
#     KIRBY_C33_INTERPOLATION_RATIOS=0.5,0.75,1.0,1.25,1.5,2.0
DEFAULT_INTERPOLATION_RATIOS = tuple(_env_list(
    'KIRBY_C33_INTERPOLATION_RATIOS',
    (1.0, 2.0) if _local_debug_run() else C33_SHIPPED_INTERPOLATION_RATIOS,
    float))

# How the three regions are cut, as multiples of the training row count. A
# budget at 0.9x rows is not meaningfully under-parameterised, which is why the
# interpolation region is a band and not the single point 1.0.
UNDER_MAX = 0.75      # budget < 0.75 x rows      -> under-parameterised
INTERP_MAX = 1.5      # 0.75 x rows .. 1.5 x rows -> interpolation region
                      # budget > 1.5 x rows       -> over-parameterised


def interpolation_budgets(min_train_rows, total_width, base_budgets,
                          ratios=DEFAULT_INTERPOLATION_RATIOS):
    """Budgets placed around this dataset's own interpolation threshold.

    Dropped: anything already in base_budgets, anything at or above the pool
    width (create_hybrid takes min(budget, n_features), so such a budget stops
    selecting and is the 'use everything' arm), and anything below 10.
    """
    out = []
    for r in ratios:
        b = int(round(float(r) * float(min_train_rows)))
        if b < 10 or b >= total_width or b in base_budgets or b in out:
            continue
        out.append(b)
    return sorted(out)


def _region(budget, min_train_rows):
    """under | interpolation | over, by multiples of the training row count."""
    ratio = float(budget) / float(min_train_rows)
    if ratio < UNDER_MAX:
        return 'under'
    if ratio <= INTERP_MAX:
        return 'interpolation'
    return 'over'


def _row_region(row, budget, min_train_rows):
    """The region this budget's hybrid ACTUALLY sat in.

    From the width it built, not the width it asked for. `n_features_mean` is
    the mean across folds of what create_hybrid returned, and it already counts
    any augmentation features appended outside the budget -- which is the other
    way the two differ. Falls back to the requested budget only when no width
    was recorded, and `region_source` on the row says which was used.
    """
    w = row.get('n_features_mean')
    if w:
        return _region(float(w), min_train_rows)
    return _region(int(budget), min_train_rows)


def _double_descent(rows, lower_is_better, min_train_rows):
    """Best under the threshold, worst on it, best above it — and the shape.

    Nothing is called double descent unless BOTH the rise into the
    interpolation region and the fall out of it are larger than the folds' own
    spread at the budgets involved. A shape smaller than the fold spread is a
    shape the folds could have produced, and is reported as inconclusive with
    its numbers, not as a finding.
    """
    scored = {b: r for b, r in rows.items() if r['reselected_mean'] is not None}
    if not scored:
        return None

    def _pick(region, want_best):
        cand = [(b, r['reselected_mean']) for b, r in scored.items()
                if _row_region(r, b, min_train_rows) == region]
        if not cand:
            return None, None
        best_b, best_m = cand[0]
        for b, m in cand[1:]:
            take = _better(m, best_m, lower_is_better) if want_best else \
                _better(best_m, m, lower_is_better)
            if take:
                best_b, best_m = b, m
        return best_b, best_m

    under_b, under_m = _pick('under', True)          # best below the threshold
    interp_b, interp_m = _pick('interpolation', False)   # worst on it
    over_b, over_m = _pick('over', True)             # best above it

    def _worse_by(a, b):
        """How much worse a is than b, positive when a really is worse."""
        if a is None or b is None:
            return None
        return float(a - b) if lower_is_better else float(b - a)

    rise = _worse_by(interp_m, under_m)   # got worse approaching the threshold
    fall = _worse_by(interp_m, over_m)    # recovered past it

    # The yardstick: the largest fold spread among the three budgets named.
    spreads = [rows[b]['reselected_std'] for b in (under_b, interp_b, over_b)
               if b is not None and rows[b]['reselected_std'] is not None]
    spread = max(spreads) if spreads else None

    # All four sign combinations are named, so 'inconclusive' means a number
    # was missing — never that the curve had a shape with no label for it.
    shape = 'inconclusive'
    if spread is not None and rise is not None and fall is not None:
        if rise > spread and fall > spread:
            # Worse at the threshold, better again above it.
            shape = 'double_descent'
        elif rise > spread and fall <= spread:
            # Worse at the threshold and it never comes back.
            shape = 'overfits_and_stays_worse'
        elif rise <= spread and fall > spread:
            # No peak on the threshold; bigger budgets are simply better.
            shape = 'improves_past_the_threshold'
        else:
            shape = 'flat_across_the_threshold'
    if (spread is not None and rise is not None and fall is None
            and rise > spread):
        shape = 'overfits_no_budget_above_threshold'

    return {
        'interpolation_threshold_rows': int(min_train_rows),
        'best_under_budget': under_b, 'best_under_reselection': under_m,
        'worst_interpolation_budget': interp_b,
        'worst_interpolation_reselected': interp_m,
        'best_over_budget': over_b, 'best_over_reselection': over_m,
        # Positive = worse. rise into the threshold, fall back out of it.
        'rise_to_threshold': rise,
        'fall_after_threshold': fall,
        'fold_spread_yardstick': spread,
        'shape': shape,
        'n_budgets_under': sum(1 for b, r in scored.items()
                               if _row_region(r, b, min_train_rows) == 'under'),
        'n_budgets_interpolation': sum(
            1 for b, r in scored.items()
            if _row_region(r, b, min_train_rows) == 'interpolation'),
        'n_budgets_over': sum(1 for b, r in scored.items()
                              if _row_region(r, b, min_train_rows) == 'over'),
        # Which width the regions above were decided on.
        'regions_from': ('realised width'
                         if all(r.get('n_features_mean') for r in scored.values())
                         else 'mixed: realised width where recorded, '
                              'requested budget otherwise'),
    }


# ------------------------------------------------------------------ summary
def _summarise_model(per_budget, base_folds, lower_is_better, min_train_rows):
    """One model's budget curve, and whether its pick is the honest best."""
    rows = {}
    for budget, b in sorted(per_budget.items(), key=lambda kv: int(kv[0])):
        d, per_fold, n_paired = _paired_delta(b.get('fold_scores'), base_folds,
                                              lower_is_better)
        _w = b.get('n_features_mean')
        rows[int(budget)] = {
            # THE REGION IS THE ONE THE HYBRID ACTUALLY SAT IN: from the width
            # it built, falling back to the budget it asked for only when no
            # width was recorded. The requested-budget label is kept beside it
            # as region_from_requested_budget.
            'region': (_region(float(_w), min_train_rows) if _w
                       else _region(int(budget), min_train_rows)),
            'region_source': 'realised width' if _w else 'requested budget',
            'region_from_requested_budget': _region(int(budget), min_train_rows),
            'budget_over_train_rows': round(float(budget) / float(min_train_rows), 3),
            'realised_width_over_train_rows': (
                None if not _w
                else round(float(_w) / float(min_train_rows), 3)),
            'reselected_mean': _f(b.get('reselected_mean')),
            'reselected_std': _f(b.get('reselected_std')),
            'fold_scores': b.get('fold_scores'),
            'search_mean': _f(b.get('search_mean')),
            # Positive = the inner search over-promised at this budget.
            'bias': _f(b.get('bias')),
            'delta_vs_best_single': d,
            'per_fold_delta': per_fold,
            'per_fold_delta_std': _fold_spread(per_fold),
            'n_paired_folds': n_paired,
            'budget_exceeds_train_rows': bool(int(budget) > min_train_rows),
            'width_exceeds_train_rows': (
                None if not _w else bool(float(_w) > min_train_rows)),
            # ⛔ THE REGION ABOVE IS LABELLED BY THE BUDGET ASKED FOR, AND THE
            # HYBRID DOES NOT ALWAYS GET IT. create_hybrid cuts a quota to the
            # features the importance method scored above zero, and
            # lightgbm_gain writes exactly 0.0 for any feature its trees never
            # split on. Measured on herg_ki: budget 1,000 against 1,129 training
            # rows is labelled 'interpolation' and realises 826 features, which
            # is 'under'. A budget of 2,000 labelled 'over' that realises 1,050
            # was never over the line. c33's whole question is what happens ON
            # and ABOVE that line, so the label has to come from the width the
            # hybrid actually had.
            'n_features_per_fold': b.get('n_features_per_fold'),
            'n_features_mean': _f(b.get('n_features_mean')),
            'width_below_budget': (
                None if b.get('n_features_mean') is None
                else bool(float(b['n_features_mean']) < int(budget))),
            'width_shortfall': (
                None if b.get('n_features_mean') is None
                else int(budget) - float(b['n_features_mean'])),
            # Kept: True where the budget asked for and the width built fall
            # in different regions, which is the case this experiment is most
            # likely to be misread on.
            'region_disagrees_with_width': (
                None if not _w
                else _region(int(budget), min_train_rows)
                     != _region(float(_w), min_train_rows)),
        }

    scored = [(b, r['reselected_mean']) for b, r in rows.items()
              if r['reselected_mean'] is not None]
    reselected_best_budget = None
    reselected_best_mean = None
    if scored:
        reselected_best_budget, reselected_best_mean = scored[0]
        for b, m in scored[1:]:
            if _better(m, reselected_best_mean, lower_is_better):
                reselected_best_budget, reselected_best_mean = b, m

    # Does the search's over-promise grow with budget? Spearman over the
    # budgets that produced both numbers. Positive = the bigger the budget,
    # the more the inner search over-promised.
    bias_pairs = [(b, r['bias']) for b, r in rows.items() if r['bias'] is not None]
    bias_trend = None
    if len(bias_pairs) >= 3:
        rho = spearmanr([b for b, _ in bias_pairs],
                        [v for _, v in bias_pairs]).correlation
        bias_trend = _f(rho)

    return rows, {
        'double_descent': _double_descent(rows, lower_is_better, min_train_rows),
        'reselected_best_budget': reselected_best_budget,
        'reselected_best_mean': reselected_best_mean,
        'reselected_best_exceeds_train_rows': (
            None if reselected_best_budget is None
            else bool(reselected_best_budget > min_train_rows)),
        'bias_vs_budget_spearman': bias_trend,
    }


def _picked_budgets(model_entry):
    """The budget each outer fold's winning recipe actually used."""
    out = []
    for bp in (model_entry.get('blueprints_per_fold') or []):
        if bp is None:
            out.append(None)
            continue
        b = bp.get('budget') if isinstance(bp, dict) else getattr(bp, 'budget', None)
        out.append(None if b is None else int(b))
    return out


def _modal(values):
    vals = [v for v in (values or []) if v is not None]
    if not vals:
        return None
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    # Ties break to the SMALLER budget: the conservative pick, so a tie never
    # reads as evidence that the search wanted a big budget.
    top = max(counts.values())
    return min(b for b, c in counts.items() if c == top)


# ------------------------------------------------- budgets that built one width
def _same_width_budgets(rows):
    """Budgets whose hybrids BUILT the same width on every outer fold.

    Dirty-run finding (2026-09-28 local run, buchwald_hartwig): budgets 2,519
    and 5,038 both built about 927 features, because create_hybrid stops at the
    last feature scoring above zero. Such budgets are one width condition
    under two budget labels. Reported here (2026-10-05); nothing is merged or
    dropped. Same width is not proof of the same features.

    Returns [[budget, budget, ...], ...], each group of 2 or more budgets with
    identical per-fold widths (every fold recorded).
    """
    by_widths = {}
    for b, r in rows.items():
        w = r.get('n_features_per_fold') or []
        if not w or any(x is None for x in w):
            continue
        by_widths.setdefault(tuple(int(x) for x in w), []).append(int(b))
    return [sorted(bs) for bs in by_widths.values() if len(bs) > 1]


# ------------------------------------------------ standard selection, one fold
def _standard_selection_fold(isearch, fold_i, tr, te, reps, labels, groups, *,
                             models, dataset_name, task_type, metric_name,
                             metric_fn, lower_is_better, seed,
                             baselines_by_model):
    """2 base representations + 2 hybrids per decided model for ONE outer fold.

    Both rankings come from `isearch`, this fold's own search on its training
    rows (the inner search nested_cv_evaluate ran). Base representations are
    ranked by that search's baselines; their held-out score is the one
    nested_cv_evaluate already measured on this fold (fitted on the training
    rows, FeatureScaler). Hybrids are the top configurations in that search's
    all_results for the model; a lower-ranked one whose selected features equal
    one already chosen is skipped for the next (the check is run on the same
    build that is scored, 2026-10-05). Each chosen hybrid is rebuilt with build_fold_hybrid on the training rows,
    with the per-representation scores that same fold search measured
    (blueprint.rep_scores of a per-fold blueprint; dirty-run bug 1), and scored
    on the held-out rows with the scorer nested_cv_evaluate uses.

    Added 2026-10-05 (handoff 2.4 step 5).
    """
    from kirby.find_best_hybrid import get_model, _outer_fit_score, _resolve_tune

    rec = {'fold': int(fold_i), 'n_train': int(len(tr)), 'n_test': int(len(te)),
           'selection_scope': 'fold_train_rows',
           'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'aug_condition': 'none', 'models': {}, 'search_failed': None}
    if isearch is None:
        rec['search_failed'] = ('the search on this fold\'s training rows '
                                'raised; nested_cv_evaluate printed the error')
        return rec

    tr = np.asarray(tr)
    te = np.asarray(te)
    y = np.asarray(labels)
    y_tr, y_te = y[tr], y[te]
    g_tr = None if groups is None else np.asarray(groups)[tr]
    tr_reps = {k: np.asarray(v)[tr] for k, v in reps.items()}
    te_reps = {k: np.asarray(v)[te] for k, v in reps.items()}
    tuned = getattr(isearch, 'tuned_models', None) or {}
    tune_outer = _resolve_tune(None)

    for m in models:
        ctx = f'c33 {dataset_name} fold {fold_i} model {m}'
        mrec = {'base': [], 'hybrids': [],
                'hybrids_skipped_identical_features': [], 'shortfall': []}

        # ---- 2 base representations ----
        try:
            base = top_base_reps(isearch, m, n=STANDARD_N_BASE,
                                 require_full=False, context=ctx)
        except Exception as e:
            base = {'selected': [],
                    'shortfall_reason': f'{type(e).__name__}: {e}'}
        _bbm = (baselines_by_model or {}).get(m) or {}
        for i, e in enumerate(base.get('selected') or [], 1):
            rep = e['rep']
            fs = (_bbm.get(rep) or {}).get('fold_scores') or []
            mrec['base'].append({
                'slot': f'base{i}', 'rep': rep,
                # Mean over the search's inner folds, inside this outer fold's
                # training rows.
                'training_rows_score': _f(e.get('score')),
                'heldout': _f(fs[fold_i]) if fold_i < len(fs) else None,
                'n_features': e.get('dimension')})
        if len(mrec['base']) < STANDARD_N_BASE:
            mrec['shortfall'].append(
                base.get('shortfall_reason')
                or f'{len(mrec["base"])} base representation(s) ranked')

        # ---- 2 hybrids, identical-features check ----
        try:
            th = top_hybrids(isearch, m, n=STANDARD_N_HYBRIDS + 4,
                             require_full=False, context=ctx)
            pooled = [dict(e, aug_condition='none') for e in th['selected']]
        except Exception as e:
            pooled = []
            mrec['shortfall'].append(f'ranking hybrids raised '
                                     f'{type(e).__name__}: {e}')
        # 2026-10-05: one build per candidate, used for both the
        # identical-features check and the held-out score. The shared check
        # (_distinct_feature_hybrids) builds without the fold search's
        # representation scores, so create_hybrid re-scores them and can split
        # the budget differently from the build scored here. A candidate whose
        # build raises keeps its slot with the error recorded, as the shared
        # check does.
        seen = {}
        n_chosen = 0
        for e in pooled:
            if n_chosen >= STANDARD_N_HYBRIDS:
                break
            bp = e.get('blueprint')
            row = {'slot': f'hybrid{n_chosen + 1}', 'label': e.get('label'),
                   'allocation_method': e.get('allocation_method'),
                   'importance_method': e.get('importance_method'),
                   'budget': e.get('budget'),
                   'internal_evaluator': e.get('internal_evaluator'),
                   'rep_count': e.get('rep_count'),
                   'step_size': e.get('step_size'),
                   'aug_condition': e.get('aug_condition'),
                   'training_rows_score': _f(e.get('score_mean')),
                   'heldout': None}
            try:
                if bp is None:
                    raise ValueError('the ranked configuration carries no '
                                     'blueprint')
                model = get_model(m, task_type, n_samples=len(tr))
                if model is None:
                    raise ValueError(f'{m} is unavailable at {len(tr)} '
                                     f'training rows')
                X_tr_h, X_te_h, fi = build_fold_hybrid(
                    tr_reps, te_reps, y_tr, bp, groups=g_tr,
                    precomputed_alloc_scores=getattr(bp, 'rep_scores', None),
                    **fold_hybrid_scoring_kwargs(
                        bp, model=model, metric_fn=metric_fn,
                        lower_is_better=lower_is_better))
                sig = hybrid_feature_signature(fi)
                if sig is not None and sig in seen:
                    mrec['hybrids_skipped_identical_features'].append({
                        'label': e.get('label'),
                        'aug_condition': e.get('aug_condition'),
                        'score_mean': e.get('score_mean'),
                        'same_features_as': seen[sig]})
                    print(f"  {dataset_name} {m}: hybrid {e.get('label')} "
                          f"selects the same features as one already chosen; "
                          f"taking the next")
                    continue
                if sig is not None:
                    seen[sig] = e.get('label')
                row['n_features_built'] = int(np.asarray(X_tr_h).shape[1])
                row['features_by_rep'] = {
                    r: int(len(info['selected_indices']))
                    for r, info in (fi or {}).items()
                    if isinstance(info, dict) and 'selected_indices' in info}
                score, _ = _outer_fit_score(
                    X_tr_h, y_tr, X_te_h, y_te, m, task_type, metric_fn,
                    seed=seed, train_groups=g_tr,
                    tuned_model=tuned.get(m), tune_per_fold=tune_outer)
                row['heldout'] = _f(score)
            except Exception as ex:
                row['error'] = f'{type(ex).__name__}: {ex}'
                print(f"  !! c33 fold {fold_i} {m} {row['slot']} failed: "
                      f"{row['error']}")
            n_chosen += 1
            mrec['hybrids'].append(row)
        if n_chosen < STANDARD_N_HYBRIDS:
            mrec['shortfall'].append(
                f'{n_chosen} distinct hybrid(s) in this fold\'s search for '
                f'{m}, {STANDARD_N_HYBRIDS} asked')
        rec['models'][m] = mrec
    return rec


def _slot_summary(per_fold, models, lower_is_better, n_outer):
    """Per model, per slot (base1, base2, hybrid1, hybrid2): held-out score per
    fold, mean, spread, and the paired per-fold delta against base1 (positive =
    better than base1). One fold gives no spread and says so."""
    out = {}
    for m in models:
        slots = {}
        for rec in per_fold:
            mr = (rec.get('models') or {}).get(m)
            if not mr:
                continue
            for row in (mr.get('base') or []) + (mr.get('hybrids') or []):
                s = slots.setdefault(row['slot'], [None] * n_outer)
                s[rec['fold']] = row.get('heldout')
        base1 = slots.get('base1')
        summ = {}
        for slot, fs in sorted(slots.items()):
            vals = [v for v in fs if v is not None]
            d, per_fold_d, n_paired = (_paired_delta(fs, base1, lower_is_better)
                                       if base1 is not None and slot != 'base1'
                                       else (None, None, 0))
            summ[slot] = {
                'heldout_per_fold': fs,
                'heldout_mean': float(np.mean(vals)) if vals else None,
                'heldout_std': (float(np.std(vals, ddof=1))
                                if len(vals) > 1 else None),
                'spread_note': ONE_FOLD_NOTE if len(vals) == 1 else None,
                'delta_vs_base1_mean': d,
                'delta_vs_base1_per_fold': per_fold_d,
                'n_paired_folds': n_paired,
            }
        out[m] = summ
    return out


# ---------------------------------------------------------------------- run
def run(data, config, output_dir, dataset_name):
    """C33: honest score per feature budget, against the dataset's size."""
    from kirby.find_best_hybrid import (nested_cv_evaluate,
                                        nested_cv_outer_folds,
                                        nested_cv_cache_namespace,
                                        _apply_budget_policy)
    from experiments.decided_grid import decided_models, decided_aug_conditions
    from experiments.complexity.complexity_helpers import DATASET_REPS

    print("\n" + "=" * 70)
    print("EXPERIMENT 33: BUDGET vs MOLECULE COUNT — DOES THE SEARCH OVERFIT?")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    task_type = data['task_type']

    # Representations: DATASET_REPS[ds]['general'] only (handoff 2.4 step 9,
    # 2026-10-05). run_all loads exactly that list (or a local subset of it);
    # anything else reaching here is refused rather than searched.
    _approved = list((DATASET_REPS.get(dataset_name) or {}).get('general')
                     or [])
    _outside = sorted(set(reps) - set(_approved))
    if _outside:
        raise ValueError(
            f"c33 {dataset_name}: representations {_outside} are not in "
            f"DATASET_REPS[{dataset_name!r}]['general'] {_approved}. Load "
            f"the general list (run_all does).")

    # Augmentation: c33 runs the general search, and Step 8 keeps augmentation
    # only on the distance search of ginkgo_colloidal and ginkgo_hic (handoff
    # 2.3 item 4, 2026-10-05). It used to run 'uncapped' wherever the dataset
    # had augmentation representations.
    _aug_kept = [c for c in decided_aug_conditions(dataset_name, 'general')
                 if c != 'none']
    if _aug_kept:
        raise ValueError(
            f"c33 {dataset_name}: DECISIONS.md keeps augmentation {_aug_kept} "
            f"on the general search. c33's nested run takes one condition per "
            f"call; which to run is undecided.")
    augmentation_note = None
    if data.get('augmentations'):
        augmentation_note = (
            f"augmentation not searched: DECISIONS.md Step 8 keeps it only on "
            f"the distance search of ginkgo_colloidal and ginkgo_hic; c33 runs "
            f"the general search ({len(data['augmentations'])} augmentation "
            f"representations loaded, none used)")
        print(f"  {augmentation_note}")

    # Outer folds: one setting, config['n_splits'] (handoff 2.4 step 14,
    # 2026-10-05). Was the literal 3 (owner, 2026-09-29). 1 = one held-out
    # split; every spread over folds then reads "one fold, no spread".
    n_outer_folds = int(config['n_splits'])
    seed = config.get('seed', 42)

    # Models: the dataset's two decided models (R1). knn runs only on a
    # distance search, which c33 does not run.
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    n_molecules = int(len(labels))
    pool_widths = {k: int(v.shape[1]) for k, v in reps.items()}
    total_width = int(sum(pool_widths.values()))

    outer_folds = nested_cv_outer_folds(reps, labels, groups, n_outer_folds,
                                        seed, task_type)
    # n_outer_folds stays the layout's count (namespaces, the slot tables);
    # the folds run may be the first few of it (OUTER_FOLDS_RUN, 2026-10-07).
    outer_folds = first_outer_folds(outer_folds, 'c33')
    n_outer_folds_run = len(outer_folds)
    train_rows = [int(len(tr)) for tr, _te in outer_folds]
    # ONE NUMBER LABELS EVERY FOLD. Group-aware outer folds have unequal
    # training row counts, so a budget that is 'over' for the smallest fold can
    # be 'under' for the largest. The spread is recorded below so a reader can
    # see how far the single threshold is from each fold's own.
    min_train_rows = int(min(train_rows)) if train_rows else n_molecules
    train_rows_spread = (sorted(int(t) for t in train_rows)
                         if train_rows else [])

    # The decided grid for every axis but the budget (R3), pruning off. The
    # budget list is c33's own: the decided budgets, FULL_BUDGETS, 4,000 and
    # 8,000, and the ratio budgets. The budget is the axis c33 measures.
    # DECIDED 2026-10-06 (Adelaide delegated): c33 keeps its own budget list,
    # an exception to R3 (decided budgets only), because the budget is its
    # question.
    search_kw = own_search_kwargs(n_molecules, dataset_name=dataset_name,
                                  tier='general')
    decided_budgets = sorted(int(b) for b in (search_kw.get('budgets') or []))
    search_kw['models'] = list(models)
    budgets = sorted(set(decided_budgets) | set(C33_BUDGETS))

    # Budgets placed around this dataset's own interpolation threshold, on top
    # of the fixed list. Without them the threshold falls in a gap and the
    # double-descent question has no points near where the peak would be.
    ratios = config.get('interpolation_ratios', DEFAULT_INTERPOLATION_RATIOS)
    extra = interpolation_budgets(min_train_rows, total_width, set(budgets),
                                  ratios) if ratios else []
    budgets = sorted(set(budgets) | set(extra))
    search_kw['budgets'] = list(budgets)
    budgets_outside_decided = [b for b in budgets if b not in decided_budgets]

    print(f"  dataset:        {dataset_name}")
    print(f"  molecules:      {n_molecules}")
    print(f"  pool:           {len(reps)} representations, {total_width} features")
    print(f"  outer folds:    {n_outer_folds}, training rows {train_rows}")
    print(f"  models:         {models} (decided)")
    print(f"  decided budgets {decided_budgets}; c33 adds "
          f"{budgets_outside_decided}")
    print(f"  budgets:        {budgets}")
    print(f"  importance {search_kw.get('importance_methods')}, allocation "
          f"{search_kw.get('allocation_methods')}, evaluators "
          f"{search_kw.get('internal_evaluators')}, representation counts "
          f"{search_kw.get('rep_counts')} (decided); pruning off")
    print(f"  added for the interpolation threshold "
          f"({min_train_rows} training rows): {extra if extra else 'none'}")
    above = [b for b in budgets if b > min_train_rows]
    print(f"  above the line: {above if above else 'none'} "
          f"(> {min_train_rows} training rows)")
    # A budget at or above the pool width stops selecting: create_hybrid takes
    # min(budget, n_features). Such a budget is the same arm as "use
    # everything" and its curve must not be read as a selection result.
    # 2026-10-05: this list is only what was ASKED. Whether a budget really
    # took every feature is decided from the width BUILT, after the run
    # ('saturated_by_width_built' in the results): create_hybrid stops at the
    # last feature scoring above zero, so a budget at the pool width can
    # still build far fewer features than the pool holds.
    saturating = [b for b in budgets if b >= total_width]
    if saturating:
        print(f"  asked at or above pool width {total_width}: {saturating} "
              f"(saturation is checked against the width built, below)")

    # A budget policy registered by the experiment layer rewrites the budget
    # list inside find_best_hybrid, AFTER this function has printed what it
    # asked for. None is registered since 2026-09-10, when the owner removed
    # dataset_policy's cap at 1000 features -- it had been silently dropping
    # every above-the-line budget on herg_ki, logd, caco2, nt_promoter_tata and
    # flab_expression_koenig, which is the whole measurement this experiment
    # exists to make. This check stays so that re-registering one cannot bring
    # the defect back unseen: the curve would otherwise just stop early with no
    # sign of why.
    surviving = list(_apply_budget_policy(dataset_name, list(budgets)))
    cut = [b for b in budgets if b not in surviving]
    cut_above = [b for b in cut if b > min_train_rows]
    budget_policy_cut = {
        'requested': list(budgets),
        'surviving': surviving,
        'dropped': cut,
        'dropped_above_train_rows': cut_above,
    }
    if cut:
        print(f"\n  {'!' * 68}")
        print(f"  A REGISTERED BUDGET POLICY DROPS {len(cut)} OF THE "
              f"{len(budgets)} BUDGETS c33 ASKED FOR.")
        print(f"  dropped:                   {cut}")
        print(f"  of those, above the line:  "
              f"{cut_above if cut_above else 'none'}")
        if cut_above:
            print(f"  c33 CANNOT ANSWER ITS OWN QUESTION ON {dataset_name!r}: "
                  f"every budget above")
            print(f"  the {min_train_rows} training rows that the policy "
                  f"removed is a budget this")
            print(f"  experiment exists to score. The curve below stops at "
                  f"{max(surviving)} features.")
            print(f"  Registered via find_best_hybrid."
                  f"register_budget_policy. There is no cap in the repo as of")
            print(f"  2026-09-10, so something has re-registered one.")
        print(f"  {'!' * 68}\n")

    # One search per outer fold, on that fold's training rows (R2), filed
    # under ocv<n>f<i>s<seed>. Called directly, not through run_nested_cv:
    # run_nested_cv narrows a named budget list to the decided budgets, which
    # would drop every budget c33 adds. Every other axis is already the
    # decided one (own_search_kwargs above). own_search=True, so
    # KIRBY_NO_COLD_SEARCH never refuses it (R6).
    # 2026-10-05: no condition_suffix, on purpose. The stored search file
    # (<ds>__ocv<n>f<i>s<seed>__general__none.pkl) is the same one the other
    # complexity experiments' per-fold searches write with the decided budgets
    # only, so the two lists replace each other's file. A condition_suffix would
    # stop that, but it also rides on the result store's tier
    # (find_best_hybrid.py, "condition_suffix reached the search-pickle
    # FILENAME"), so c33 would then re-fit every configuration at the decided
    # budgets instead of reading the stored rows. With the shared name, the
    # stored rows are read back by configuration whatever the budget list
    # (_run_search seeds from the result database).
    result = nested_cv_evaluate(
        base_reps=reps,
        labels=labels,
        groups=groups,
        n_outer_folds=n_outer_folds,
        first_outer_folds=n_outer_folds_run,
        seed=seed,
        verbose=True,
        dataset_name=dataset_name,
        task_type=task_type,
        metric=metric_name,
        cache=True,
        augmentations=None,
        augmentation_strategy='none',
        **search_kw,
    )

    lower = bool(lower_is_better)
    if bool(result.lower_is_better) != lower:
        raise ValueError(
            f"c33 {dataset_name}: the nested run scored {result.metric_name} "
            f"with lower_is_better={result.lower_is_better}, but the dataset's "
            f"metric {metric_name} has lower_is_better={lower}.")

    missing = []
    by_model_budget = {}
    summary = {}

    # ---- the standard selection, per outer fold (handoff 2.4 step 5) ----
    _inner = list(getattr(result, 'inner_searches', None) or [])
    per_fold = []
    for i, (tr, te) in enumerate(outer_folds):
        isearch = _inner[i] if i < len(_inner) else None
        rec = _standard_selection_fold(
            isearch, i, tr, te, reps, labels, groups, models=models,
            dataset_name=dataset_name, task_type=task_type,
            metric_name=metric_name, metric_fn=metric_fn,
            lower_is_better=lower, seed=seed,
            baselines_by_model=getattr(result, 'baselines_by_model', None))
        rec['cache_namespace'] = nested_cv_cache_namespace(i, n_outer_folds,
                                                           seed)
        if rec['search_failed']:
            missing.append(f'outer fold {i}: {rec["search_failed"]}')
        for m, mr in (rec.get('models') or {}).items():
            for s in mr.get('shortfall') or []:
                missing.append(f'outer fold {i} {m}: {s}')
        per_fold.append(rec)
    standard_by_model = _slot_summary(per_fold, models, lower, n_outer_folds)

    # Rank-2 base representation per fold, from the standard selection.
    rank2_by_model = {}
    for m in models:
        reps2, folds2 = [], []
        for rec in per_fold:
            b2 = [r for r in ((rec.get('models') or {}).get(m) or {})
                  .get('base', []) if r['slot'] == 'base2']
            reps2.append(b2[0]['rep'] if b2 else None)
            folds2.append(b2[0]['heldout'] if b2 else None)
        rank2_by_model[m] = (reps2, folds2)

    for model, per_budget in (result.by_model_budget or {}).items():
        entry = (result.by_model or {}).get(model, {}) or {}
        base_folds = entry.get('best_single_fold_scores')
        rows, curve = _summarise_model(per_budget, base_folds, lower,
                                       min_train_rows)
        by_model_budget[model] = {str(b): r for b, r in rows.items()}

        # Budgets that built the same width on every fold (dirty-run finding,
        # 2026-10-05): one width condition under two budget labels.
        same_width = _same_width_budgets(rows)
        for grp in same_width:
            for b in grp:
                by_model_budget[model][str(b)]['same_width_as'] = [
                    x for x in grp if x != b]

        # THE SAME CURVE AGAINST THE RANK-2 BASE REPRESENTATION (2026-09-18).
        # `delta_vs_best_single` is measured against rank 1 only, so a budget
        # that beats rank 1 and loses to rank 2 reads as a win. Since
        # 2026-10-05 rank 2 is base2 of the standard selection: ranked on the
        # fold's own search, held-out score from nested_cv_evaluate.
        rank2_reps, rank2_folds = rank2_by_model.get(model, ([], []))
        if any(v is not None for v in rank2_folds):
            for b, r in rows.items():
                _d, _pf, _n = _paired_delta(r.get('fold_scores'), rank2_folds,
                                            lower)
                by_model_budget[model][str(b)]['delta_vs_rank2_single'] = _d
                by_model_budget[model][str(b)]['delta_vs_rank2_n_paired'] = _n

        picked = _picked_budgets(entry)
        picked_modal = _modal(picked)

        reselected_at_picked = None
        if picked_modal is not None and picked_modal in rows:
            reselected_at_picked = rows[picked_modal]['reselected_mean']

        # Positive = the budget the search picked scored WORSE, held out, than
        # the budget that was honestly best. That difference is what the
        # process costs by picking the way it does.
        overfit_gap = None
        if reselected_at_picked is not None and curve['reselected_best_mean'] is not None:
            overfit_gap = (
                float(reselected_at_picked - curve['reselected_best_mean']) if lower
                else float(curve['reselected_best_mean'] - reselected_at_picked))

        # What the process actually delivers. nested CV picks a budget PER
        # OUTER FOLD, so hybrid_mean is the honest score of those per-fold
        # picks — not the honest score at any one budget. Positive gap = the
        # process scored worse, held out, than the best single fixed budget
        # would have. That is the cost of letting the search choose.
        process_gap = None
        hyb = _f(entry.get('hybrid_mean'))
        if hyb is not None and curve['reselected_best_mean'] is not None:
            process_gap = (float(hyb - curve['reselected_best_mean']) if lower
                           else float(curve['reselected_best_mean'] - hyb))

        summary[model] = {
            'process_reselected_mean': hyb,
            'process_gap': process_gap,
            'picked_budget_per_fold': picked,
            'picked_budget_modal': picked_modal,
            'picked_exceeds_train_rows': (
                None if picked_modal is None
                else bool(picked_modal > min_train_rows)),
            'reselected_at_picked_budget': reselected_at_picked,
            'overfit_gap': overfit_gap,
            # Which representation was second best on each outer fold, for the
            # delta_vs_rank2_single curve added beside the rank-1 one.
            'rank2_rep_per_fold': rank2_reps,
            'budgets_building_same_width': same_width,
            # The gap is only worth reading if it is bigger than the amount the
            # folds themselves move by at the picked budget.
            'picked_budget_per_fold_delta_std': (
                rows.get(picked_modal, {}).get('per_fold_delta_std')
                if picked_modal is not None else None),
            'spread_note': ONE_FOLD_NOTE if n_outer_folds == 1 else None,
            **curve,
        }

    # 2026-10-05: saturation from the width BUILT, per (model, budget). A
    # budget saturates only where every outer fold built at least the pool's
    # width (total_width base features). The width can include augmentation
    # extras, so at or above the pool width is the test, not equality.
    saturated_by_width = {}
    for _m, _rows in by_model_budget.items():
        for _b_str, _r in _rows.items():
            _w = [w for w in (_r.get('n_features_per_fold') or [])
                  if w is not None]
            if _w and all(int(w) >= total_width for w in _w):
                saturated_by_width.setdefault(_m, []).append(int(_b_str))
    for _m in saturated_by_width:
        saturated_by_width[_m] = sorted(saturated_by_width[_m])
    if saturated_by_width:
        print(f"\n  ⛔ SATURATED (width built >= pool width {total_width} on "
              f"every fold): {saturated_by_width}")
    else:
        print(f"\n  No budget built the whole pool ({total_width} features) "
              f"on every fold.")
    _same_any = {m: s['budgets_building_same_width'] for m, s in summary.items()
                 if s['budgets_building_same_width']}
    if _same_any:
        print(f"  Budgets that built the same width on every fold (one width "
              f"condition under two budget labels): {_same_any}")

    results = {
        'experiment': 'c33_budget_overfitting',
        'result_identity': RESULT_IDENTITY,
        'dataset': dataset_name,
        'metric_name': metric_name,
        'lower_is_better': lower,
        'selection_scope': 'fold_train_rows',
        'n_outer_folds': int(n_outer_folds),
        'n_outer_folds_run': int(n_outer_folds_run),
        'spread_note': ONE_FOLD_NOTE if n_outer_folds == 1 else None,
        'models': list(models),
        'n_molecules': n_molecules,
        'outer_train_rows': train_rows,
        'min_train_rows': min_train_rows,
        'train_rows_per_outer_fold': train_rows_spread,
        'pool': {
            'reps': sorted(reps.keys()),
            'width_per_rep': pool_widths,
            'total_width': total_width,
        },
        'search_grid': {
            'tier': search_kw.get('tier'),
            'importance_methods': search_kw.get('importance_methods'),
            'allocation_methods': search_kw.get('allocation_methods'),
            'internal_evaluators': search_kw.get('internal_evaluators'),
            'rep_counts': search_kw.get('rep_counts'),
            'aug_condition': 'none',
            'pruning': None,
        },
        'augmentation_note': augmentation_note,
        'decided_budgets': decided_budgets,
        'budgets_outside_decided': budgets_outside_decided,
        'budgets': budgets,
        'budget_policy_cut': budget_policy_cut,
        'budgets_above_train_rows': above,
        # What was asked; NOT whether it saturated (2026-10-05).
        'budgets_at_or_above_pool_width': saturating,
        # {model: [budgets]} where every outer fold BUILT at least the pool
        # width. The widths per fold are in by_model_budget[...]
        # ['n_features_per_fold'].
        'saturated_by_width_built': saturated_by_width,
        'interpolation_budgets_added': extra,
        'interpolation_ratios': list(ratios) if ratios else [],
        'by_model': {
            m: {
                'delta_mean_of_folds': _f(e.get('delta_mean_of_folds')),
                'n_paired_folds': int(e.get('n_paired_folds') or 0),
                'hybrid_mean': _f(e.get('hybrid_mean')),
                'best_single_mean': _f(e.get('best_single_mean')),
                'best_single_reps_per_fold': e.get('best_single_reps_per_fold'),
                'per_fold_delta': e.get('per_fold_delta'),
                'per_fold_delta_std': _fold_spread(e.get('per_fold_delta')),
            }
            for m, e in (result.by_model or {}).items()
        },
        'by_model_budget': by_model_budget,
        'summary': summary,
        # The standard selection (2026-10-05): per outer fold, per decided
        # model, 2 base representations and 2 hybrids ranked on that fold's
        # own search, each with its training-row score and held-out score.
        'per_fold': per_fold,
        'standard_selection_by_model': standard_by_model,
        'missing': missing,
    }

    # ---- printed table: one row is one (model, budget) --------------------
    print(f"\n  HONEST SCORE PER BUDGET ({metric_name}, "
          f"{'lower' if lower else 'higher'} is better)")
    print(f"  * = the hybrid BUILT more than the {min_train_rows} training "
          f"rows in the smallest outer fold. ~ = it ASKED for more than that "
          f"and built fewer. A budget is a maximum, not a width.")
    print(f"  bias > 0 means the inner search over-promised at that budget.\n")
    hdr = (f"  {'model':<10} {'budget':>7} {'honest':>9} {'+/-':>7} "
           f"{'search':>9} {'bias':>8} {'vs best rep':>12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for model in sorted(by_model_budget):
        for b_str in sorted(by_model_budget[model], key=int):
            r = by_model_budget[model][b_str]
            # The star follows the width built, not the width asked for.
            if r.get('width_exceeds_train_rows'):
                star = '*'
            elif r['budget_exceeds_train_rows']:
                star = '~'
            else:
                star = ' '
            def _s(v, w=9, p=4):
                return f"{v:>{w}.{p}f}" if v is not None else f"{'-':>{w}}"
            print(f"  {model:<10} {b_str + star:>7} {_s(r['reselected_mean'])} "
                  f"{_s(r['reselected_std'], 7, 3)} {_s(r['search_mean'])} "
                  f"{_s(r['bias'], 8)} {_s(r['delta_vs_best_single'], 12)}")
    if n_outer_folds == 1:
        print(f"  +/- is blank: {ONE_FOLD_NOTE}.")

    print(f"\n  WHAT THE SEARCH PICKED vs WHAT WAS HONESTLY BEST")
    print(f"  {'model':<10} {'picked':>8} {'honest best':>12} "
          f"{'process gap':>12} {'modal gap':>10} {'fold sd':>9} {'bias trend':>11}")
    print("  " + "-" * 78)
    for model in sorted(summary):
        s = summary[model]
        def _s(v, w=9, p=4):
            return f"{v:>{w}.{p}f}" if v is not None else f"{'-':>{w}}"
        pk = s['picked_budget_modal']
        hb = s['reselected_best_budget']
        print(f"  {model:<10} "
              f"{(str(pk) + ('*' if s['picked_exceeds_train_rows'] else '')) if pk else '-':>8} "
              f"{(str(hb) + ('*' if s['reselected_best_exceeds_train_rows'] else '')) if hb else '-':>12} "
              f"{_s(s['process_gap'], 12)} "
              f"{_s(s['overfit_gap'], 10)} "
              f"{_s(s['picked_budget_per_fold_delta_std'])} "
              f"{_s(s['bias_vs_budget_spearman'], 11, 3)}")
    print("\n  process gap > 0 = the per-fold budget the search chose scored "
          "worse, held out,")
    print("                    than the best single fixed budget. THE number: "
          "what choosing costs.")
    print("  modal gap    > 0 = same, at the one budget the folds picked most "
          "often.")
    print("  Read either against the fold sd beside it: a gap smaller than "
          "that is a gap the folds could have produced.")

    # ---- double descent: one row is one model ---------------------------
    print(f"\n  ACROSS THE INTERPOLATION THRESHOLD "
          f"({min_train_rows} features = {min_train_rows} training rows)")
    print(f"  Budgets are grouped: under < {UNDER_MAX}x rows, interpolation "
          f"{UNDER_MAX}-{INTERP_MAX}x, over > {INTERP_MAX}x.")
    print(f"  {'model':<10} {'best under':>11} {'worst at':>10} "
          f"{'best over':>10} {'rise':>8} {'fall':>8} {'fold sd':>8}  shape")
    print("  " + "-" * 84)
    for model in sorted(summary):
        dd = summary[model].get('double_descent')
        if not dd:
            print(f"  {model:<10} {'no scored budget':>11}")
            continue
        def _b(v):
            return f"{v:>10}" if v is not None else f"{'-':>10}"
        def _s(v, w=8, p=4):
            return f"{v:>{w}.{p}f}" if v is not None else f"{'-':>{w}}"
        print(f"  {model:<10} {_b(dd['best_under_budget']):>11} "
              f"{_b(dd['worst_interpolation_budget'])} "
              f"{_b(dd['best_over_budget'])} "
              f"{_s(dd['rise_to_threshold'])} {_s(dd['fall_after_threshold'])} "
              f"{_s(dd['fold_spread_yardstick'])}  {dd['shape']}")
    print("\n  rise > 0 = the score got worse approaching the threshold.")
    print("  fall > 0 = it recovered again above the threshold.")
    print("  'double_descent' is printed only when BOTH exceed the fold sd. "
          "Otherwise the")
    print("  numbers stand and the shape reads inconclusive — a shape smaller "
          "than the fold")
    print("  spread is a shape the folds could have produced.")
    print("\n  ⛔ The dnn carries 41,024 hidden weights at any budget, the svm "
          "at most one")
    print("  coefficient per training molecule, and an rf one leaf per "
          "training row: all")
    print("  sit past their own threshold at every budget here, so a flat "
          "shape from them")
    print("  says nothing about double descent.")

    # ---- the standard selection: one row is one (model, slot) -----------
    print(f"\n  STANDARD SELECTION PER FOLD ({metric_name}, held out; "
          f"ranked on each fold's training rows)")
    print(f"  {'model':<10} {'slot':<8} {'mean':>9} {'sd':>8} "
          f"{'vs base1':>9}  per fold")
    for m in sorted(standard_by_model):
        for slot, srow in standard_by_model[m].items():
            def _s(v, w=9, p=4):
                return f"{v:>{w}.{p}f}" if v is not None else f"{'-':>{w}}"
            print(f"  {m:<10} {slot:<8} {_s(srow['heldout_mean'])} "
                  f"{_s(srow['heldout_std'], 8)} "
                  f"{_s(srow['delta_vs_base1_mean'])}  "
                  f"{srow['heldout_per_fold']}")
    if missing:
        print(f"\n  missing ({len(missing)}): {missing}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser(
        'C33: Budget vs molecule count — does the search overfit?',
        has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)
    # 2026-10-05: the dataset's general representations, as run_all loads
    # them (handoff 2.4 step 9), instead of the --reps preset.
    from experiments.complexity.complexity_helpers import DATASET_REPS
    data = load_data(args.dataset, rep_preset='all',
                     rep_names=DATASET_REPS[args.dataset]['general'],
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(convert_numpy(results), args.output,
                            'c33_budget_overfitting', args.dataset,
                            metric_name=results.get('metric_name'),
                            lower_is_better=results.get('lower_is_better'))
