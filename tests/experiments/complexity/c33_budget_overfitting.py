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
    outer-train and scores it on the untouched outer test fold
    (find_best_hybrid.py:9263-9325), then reports per model per budget in
    by_model_budget (:9553-9581). This experiment makes one nested run and
    reads that dict. It runs no second search.

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

    'linear' is in C33_MODELS for exactly this reason: Ridge(alpha=1e-6) /
    LogisticRegression(C=1e6) has one weight per selected feature and nothing
    else, so budget = training rows IS its interpolation threshold. It is the
    only row of the threshold table whose shape column can mean what it says.
    It is tier 'extra' in DEFAULT_MODELS, so naming it here does not change any
    other experiment.

THE TWO NUMBERS THE ANSWER TURNS ON, per model:
    the budget each outer fold actually PICKED, and the budget with the best
    reselected_mean. If the search picks 2,000 features on a 672-molecule dataset
    while 250 is honestly best, the process overfits and a cap has a measured
    reason — per dataset by its own molecule count, never a budget dropped from
    FULL_BUDGETS everywhere. If the honest curve keeps climbing past the
    molecule count, removing the cap was right.

READ IT PER MODEL. rf with max_features='sqrt' draws about 45 of 2,000
features at each split, so more features than molecules barely touches it. The
curves may disagree, and that disagreement is the result. Only 'linear' answers
the double-descent question; the other five answer the cap question.

BUDGETS. C33_BUDGETS is FULL_BUDGETS plus 4,000 and 8,000. Measured on
humanness_therapeutic 2026-09-10: rf and svm both put their best honest score
at the LARGEST budget tested, and a best score at the edge of the range means
the range is too short. 8,000 exceeds the pool width on tcga_brca_pam50 (4,732
features), where create_hybrid takes min(budget, n_features) and the arm
becomes "use every feature" — that is the genuine no-selection endpoint, and
the SATURATED line below names it rather than dropping it.

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
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, run_nested_cv,
    top_base_reps,
    FULL_BUDGETS,
)


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


# ------------------------------------------------------------- outer splits
def _outer_train_rows(reps, labels, groups, n_outer_folds, seed, task_type):
    """Training rows in each outer fold, from the SAME splitter nested CV uses.

    nested_cv_outer_folds is the single source of truth for those splits, so
    the row counts here are the row counts the budgets were actually selected
    against — not a re-derivation that could drift.
    """
    from kirby.find_best_hybrid import nested_cv_outer_folds
    folds = nested_cv_outer_folds(reps, labels, groups, n_outer_folds, seed,
                                  task_type)
    return [int(len(tr)) for tr, _te in folds]


# ------------------------------------------------------- interpolation grid
# Where the classical peak sits: features equal to training rows. The ratios
# straddle it so the curve has points on both sides of the threshold and one
# on it. Under-parameterised at 0.5x and 0.75x, at the threshold at 1x,
# over-parameterised at 1.25x, 1.5x and 2x.
# FULL_BUDGETS stops at 2,000. On the 2026-09-10 humanness_therapeutic run rf
# and svm both scored best AT 2x training rows, the largest budget tested, and a
# best score at the edge of a range means the range is too short. 4,000 is under
# every local pool width; 8,000 is above tcga_brca_pam50's 4,732, where it
# becomes the "use every feature" arm and is reported as SATURATED rather than
# dropped. Owner decision 2026-09-10.
C33_BUDGETS = list(FULL_BUDGETS) + [4000, 8000]

# The five default models plus 'linear'. 'linear' is the only one whose fitted
# parameter count equals the budget, so it is the only one for which
# budget = training rows is a real interpolation threshold. Naming models here
# means c33 does NOT share a cache key with a models=None precompute -- it
# already does not, because the interpolation budgets are in that key too.
C33_MODELS = ['rf', 'xgboost', 'lightgbm', 'svm', 'dnn', 'linear']

DEFAULT_INTERPOLATION_RATIOS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)

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
                if _region(b, min_train_rows) == region]
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
        'n_budgets_under': sum(1 for b in scored
                               if _region(b, min_train_rows) == 'under'),
        'n_budgets_interpolation': sum(1 for b in scored
                                       if _region(b, min_train_rows) == 'interpolation'),
        'n_budgets_over': sum(1 for b in scored
                              if _region(b, min_train_rows) == 'over'),
    }


# ------------------------------------------------------------------ summary
def _summarise_model(per_budget, base_folds, lower_is_better, min_train_rows):
    """One model's budget curve, and whether its pick is the honest best."""
    rows = {}
    for budget, b in sorted(per_budget.items(), key=lambda kv: int(kv[0])):
        d, per_fold, n_paired = _paired_delta(b.get('fold_scores'), base_folds,
                                              lower_is_better)
        rows[int(budget)] = {
            'region': _region(int(budget), min_train_rows),
            'budget_over_train_rows': round(float(budget) / float(min_train_rows), 3),
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


# ---------------------------------------------------------------------- run
def run(data, config, output_dir, dataset_name):
    """C33: honest score per feature budget, against the dataset's size."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 33: BUDGET vs MOLECULE COUNT — DOES THE SEARCH OVERFIT?")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    task_type = data['task_type']
    augs = data.get('augmentations') or None
    n_outer_folds = config['n_splits']
    seed = config.get('seed', 42)
    budgets = list(config.get('budgets') or C33_BUDGETS)
    models = list(config.get('models') or C33_MODELS)

    metric_name, _metric_fn, lower_is_better = get_domain_metric(data)

    n_molecules = int(len(labels))
    pool_widths = {k: int(v.shape[1]) for k, v in reps.items()}
    total_width = int(sum(pool_widths.values()))

    train_rows = _outer_train_rows(reps, labels, groups, n_outer_folds, seed,
                                   task_type)
    min_train_rows = int(min(train_rows)) if train_rows else n_molecules

    # Budgets placed around this dataset's own interpolation threshold, on top
    # of the fixed list. Without them the threshold falls in a gap and the
    # double-descent question has no points near where the peak would be.
    ratios = config.get('interpolation_ratios', DEFAULT_INTERPOLATION_RATIOS)
    extra = interpolation_budgets(min_train_rows, total_width, set(budgets),
                                  ratios) if ratios else []
    budgets = sorted(set(budgets) | set(extra))

    print(f"  dataset:        {dataset_name}")
    print(f"  molecules:      {n_molecules}")
    print(f"  pool:           {len(reps)} representations, {total_width} features")
    print(f"  outer folds:    {n_outer_folds}, training rows {train_rows}")
    print(f"  budgets:        {budgets}")
    print(f"  models:         {models}")
    print(f"  added for the interpolation threshold "
          f"({min_train_rows} training rows): {extra if extra else 'none'}")
    above = [b for b in budgets if b > min_train_rows]
    print(f"  above the line: {above if above else 'none'} "
          f"(> {min_train_rows} training rows)")
    # A budget at or above the pool width stops selecting: create_hybrid takes
    # min(budget, n_features). Such a budget is the same arm as "use
    # everything" and its curve must not be read as a selection result.
    saturating = [b for b in budgets if b >= total_width]
    if saturating:
        print(f"  ⛔ SATURATED:   {saturating} >= pool width {total_width} — "
              f"these take every feature, not a selection")

    # A budget policy registered by the experiment layer rewrites the budget
    # list inside find_best_hybrid, AFTER this function has printed what it
    # asked for. None is registered since 2026-09-10, when the owner removed
    # dataset_policy's cap at 1000 features -- it had been silently dropping
    # every above-the-line budget on herg_ki, logd, caco2, nt_promoter_tata and
    # flab_expression_koenig, which is the whole measurement this experiment
    # exists to make. This check stays so that re-registering one cannot bring
    # the defect back unseen: the curve would otherwise just stop early with no
    # sign of why.
    from kirby.find_best_hybrid import _apply_budget_policy
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

    result = run_nested_cv(
        reps, labels, groups, config,
        dataset_name=dataset_name,
        augmentations=augs,
        task_type=task_type,
        metric=metric_name,
        domain=data.get('domain'),
        tier='general',
        budgets=budgets,
        models=models,
        n_outer_folds=n_outer_folds,
        seed=seed,
        verbose=True,
    )

    lower = bool(result.lower_is_better)
    by_model_budget = {}
    summary = {}
    # {model: [rank-2 representation name per outer fold]}. NOT stored inside
    # by_model_budget[model]: that dict is iterated with sorted(..., key=int)
    # at the printing loop below, so a key that is not a budget raises there.
    rank2_by_model = {}

    for model, per_budget in (result.by_model_budget or {}).items():
        entry = (result.by_model or {}).get(model, {}) or {}
        base_folds = entry.get('best_single_fold_scores')
        rows, curve = _summarise_model(per_budget, base_folds, lower,
                                       min_train_rows)
        by_model_budget[model] = {str(b): r for b, r in rows.items()}

        # THE SAME CURVE AGAINST THE RANK-2 BASE REPRESENTATION. Added
        # 2026-09-18. `delta_vs_best_single` is measured against rank 1 only,
        # so a budget that beats rank 1 and loses to rank 2 reads as a win.
        # This adds the second curve; every existing key is untouched and still
        # measured against rank 1.
        #
        # 0 new model fits. top_base_reps reads each outer fold's stored inner
        # search, and the rank-2 representation's held-out score for that fold
        # is already in result.baselines_by_model.
        #
        # ⛔ ONE DIRECTION CAVEAT, UNRESOLVED. Nested CV picks rank 1 using the
        # `_baselines_lower_is_better` parameter; top_base_reps resolves the
        # direction from the stored search. The two agree on a builtin metric
        # under a non-blend objective, which is every c33 run today, and
        # disagree under a blend. c33 runs no blend, so this is recorded rather
        # than guarded.
        rank2_folds, rank2_rep_per_fold = [], []
        _inner = getattr(result, 'inner_searches', None) or []
        _bbm = (getattr(result, 'baselines_by_model', None) or {}).get(model, {})
        for i, _isearch in enumerate(_inner):
            _rep = None
            try:
                _sel = top_base_reps(_isearch, model, n=2, require_full=False,
                                     context=f'c33 {dataset_name} fold {i}')
                if len(_sel['selected']) > 1:
                    _rep = _sel['selected'][1]['rep']
            except Exception:
                _rep = None
            rank2_rep_per_fold.append(_rep)
            _fs = (_bbm.get(_rep) or {}).get('fold_scores') if _rep else None
            rank2_folds.append(_fs[i] if (_fs and i < len(_fs)) else None)

        if any(v is not None for v in rank2_folds):
            for b, r in rows.items():
                _d, _pf, _n = _paired_delta(r.get('fold_scores'), rank2_folds,
                                            lower)
                by_model_budget[model][str(b)]['delta_vs_rank2_single'] = _d
                by_model_budget[model][str(b)]['delta_vs_rank2_n_paired'] = _n
            rank2_by_model[model] = rank2_rep_per_fold

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
            'rank2_rep_per_fold': rank2_by_model.get(model),
            # The gap is only worth reading if it is bigger than the amount the
            # folds themselves move by at the picked budget.
            'picked_budget_per_fold_delta_std': (
                rows.get(picked_modal, {}).get('per_fold_delta_std')
                if picked_modal is not None else None),
            **curve,
        }

    results = {
        'experiment': 'c33_budget_overfitting',
        'dataset': dataset_name,
        'metric_name': result.metric_name,
        'lower_is_better': lower,
        'n_outer_folds': int(result.n_outer_folds),
        'n_molecules': n_molecules,
        'outer_train_rows': train_rows,
        'min_train_rows': min_train_rows,
        'pool': {
            'reps': sorted(reps.keys()),
            'width_per_rep': pool_widths,
            'total_width': total_width,
        },
        'budgets': budgets,
        'budget_policy_cut': budget_policy_cut,
        'budgets_above_train_rows': above,
        'budgets_at_or_above_pool_width': saturating,
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
    }

    # ---- printed table: one row is one (model, budget) --------------------
    print(f"\n  HONEST SCORE PER BUDGET ({metric_name}, "
          f"{'lower' if lower else 'higher'} is better)")
    print(f"  A budget marked * is above the {min_train_rows} training rows "
          f"in the smallest outer fold.")
    print(f"  bias > 0 means the inner search over-promised at that budget.\n")
    hdr = (f"  {'model':<10} {'budget':>7} {'honest':>9} {'+/-':>7} "
           f"{'search':>9} {'bias':>8} {'vs best rep':>12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for model in sorted(by_model_budget):
        for b_str in sorted(by_model_budget[model], key=int):
            r = by_model_budget[model][b_str]
            star = '*' if r['budget_exceeds_train_rows'] else ' '
            def _s(v, w=9, p=4):
                return f"{v:>{w}.{p}f}" if v is not None else f"{'-':>{w}}"
            print(f"  {model:<10} {b_str + star:>7} {_s(r['reselected_mean'])} "
                  f"{_s(r['reselected_std'], 7, 3)} {_s(r['search_mean'])} "
                  f"{_s(r['bias'], 8)} {_s(r['delta_vs_best_single'], 12)}")

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
    print("\n  ⛔ Only the 'linear' row can show double descent. Its fitted "
          "parameter count")
    print("  IS the budget, so budget = training rows is its real threshold. "
          "The dnn carries")
    print("  41,024 hidden weights at any budget, the svm at most one "
          "coefficient per training")
    print("  molecule, and an rf one leaf per training row — all four sit past "
          "their own")
    print("  threshold at every budget here, so a flat shape from them says "
          "nothing about it.")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser(
        'C33: Budget vs molecule count — does the search overfit?',
        has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(convert_numpy(results), args.output,
                            'c33_budget_overfitting', args.dataset,
                            metric_name=results.get('metric_name'),
                            lower_is_better=results.get('lower_is_better'))
