#!/usr/bin/env python3
"""f32 — how a representation's score is normalised before it sizes its share
of the budget.

RAN 2026-09-10, nine datasets, 45 folds, no build failures. DECIDED: one hardcoded
rule per metric, NOT an axis in the search, and the f1_weighted null is the f1 of
always predicting the majority training class. Full write-up with every number in
FRAMEWORK_FINALIZATION.md section 5; the answers are on the ">>" lines of section
3.9. The headline: the split moved in 978 of 1,800 arm-against-today comparisons,
and of the 206 cases where it changed, 8 were better on all five folds, 29 were
worse on all five, and 169 changed sign fold to fold. Re-running this file is only
worth it on the cluster, where the second auc and second mcc dataset have pools.

`create_hybrid` gives each representation a share of the budget proportional to
its score. That is only meaningful when zero on the score means "this
representation contributes nothing". Spearman and MCC have that property. A
negated RMSE does not — its zero is a perfect prediction. AUROC does not —
chance is 0.5, and 0.5 is a positive number.

Measured on 2026-09-09 by calling the real allocation functions with illustrative
scores (BUDGET_SPLIT_NORMALIZATION_HANDOFF_2026_09_09.md): five representations
whose Spearman scores span 0.700 to 0.693 split a budget of 100 features as
20/20/20/20/20; the same five as negated RMSE spanning -0.300 to -0.307 split it
50/29/14/7/0. A representation sitting exactly at chance AUROC took 16 of 100
features.

WHAT THIS EXPERIMENT ANSWERS. Not "which normalisation is best" — for 68 of the
70 registry entries the rule is settled by what the metric means. It answers
whether the best normalisation follows the METRIC or the DATASET. Two datasets
reporting the same metric that want different normalisations cannot be served by
one rule per metric, and the choice would have to become an axis inside the
search instead.

HOW IT INJECTS THE ARMS, WITHOUT TOUCHING src/. `_sanitize_baseline_scores`
(src/kirby/hybrid.py:821) subtracts the minimum ONLY when the minimum is
negative, so a score vector already at or above zero passes through it unchanged.
`create_hybrid` takes `precomputed_alloc_scores` and hands them to both
`allocate_performance_weighted` and `allocate_weighted_global`. So each arm
transforms the score vector itself and hands the result in. The engine is
imported, never edited.

TWO TRAPS, both found by reading the code on 2026-09-09.

1. `create_hybrid` scores each representation with `metric_fn` only when one is
   passed; without it `_score_rep_for_budget` keeps the evaluator's own r2 (or
   accuracy). r2 is ALREADY one minus the error over the error of predicting the
   training mean — the very skill score this file tests. f31 does not pass
   metric_fn. This one does, from the registry, or the `raw` arm would not be
   what a real search does.
2. `BUILTIN_METRICS['auc']` is applied to `m.predict(...)`, which for a
   classifier is hard class labels. On hard labels that number is
   (sensitivity + specificity) / 2. Chance is still 0.5, so the 2*(auc - 0.5)
   rule is unaffected. Recorded in every result file as `auc_on_hard_labels`.
   NOT fixed here.

Usage:
    python tests/experiments/framework/f32_budget_split_normalisation.py --smoke
    python tests/experiments/framework/f32_budget_split_normalisation.py --check --datasets caco2
    python tests/experiments/framework/f32_budget_split_normalisation.py --datasets caco2 --folds 5
"""

import os
os.environ.setdefault('KIRBY_CACHE_DIR', '/Volumes/seagate/kirby_cache')
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import faulthandler
faulthandler.enable()

import sys
import json
import time
import argparse
import warnings
import traceback
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'tests'))

import numpy as np

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# Two datasets per metric, so "the metric decides" can be told apart from "the
# dataset decides". The metric itself is NEVER written here — it is read from
# kirby.registry at runtime, and _check_pairs asserts the pairing below still
# holds, so a registry edit fails this file loudly instead of silently
# regrouping it.
#
# tcga_brca_pam50 is the ninth and is not a pair. It is the only f1_weighted
# dataset with a representation pool anywhere, and f1_weighted is the one metric
# with no agreed zero. On one dataset it can say WHICH null is better; it cannot
# say whether that answer follows the metric or the dataset. Reported with that
# limit stated.
DATASETS = [
    'herg_ki', 'caco2',                                # rmse
    'flab_thermo_tresanco', 'flab_expression_koenig',  # spearman
    'humanness_therapeutic', 'tcga_luad_vs_lusc',      # auc
    'nt_promoter_tata', 'gb_human_enhancers_cohn',     # mcc
    'tcga_brca_pam50',                                 # f1_weighted, unpaired
]

# Five arms. Every one returns values at or above zero, so
# _sanitize_baseline_scores is a no-op on them and the arm reaches the weights
# intact. `raw` is the exception and the point: it hands the scores back
# untouched, shift and all, which is what the engine does today.
NORMALISATIONS = ['raw', 'subtract_min', 'minmax', 'skill', 'rank']

# The sixth arm, defined only where f1_weighted is the metric: the same skill
# score against a permutation null instead of against the majority class. Not a
# cut elsewhere — it is undefined elsewhere, because every other metric has an
# agreed null.
F1_ONLY_NORMALISATIONS = ['skill_perm']

# The five shipped defaults (src/kirby/find_best_hybrid.py:2987). rank_exp_d0.3
# reads only the ORDER of the scores, so its split must not move under any arm —
# that is the harness check in _check_invariants. wglobal_power2 is the one
# default that pools features globally instead of giving each representation a
# quota, and it runs the same weighting code, so it is exposed to the same
# problem.
WEIGHTINGS = ['performance_weighted', 'power_2', 'softmax_T1',
              'rank_exp_d0.3', 'wglobal_power2']

# The ends of the search's own default list (src/kirby/find_best_hybrid.py:6819
# is [50, 150, 300]). The shape of the split matters most at the small budget.
BUDGETS = [50, 300]

DOWNSTREAM_MODEL = 'lightgbm'
INTERNAL_EVALUATOR = 'lightgbm'
IMPORTANCE_METHOD = 'lightgbm_gain'
N_OUTER_FOLDS = 5
SEED = 42

# How many random label draws make the f1_weighted permutation null.
N_PERMUTATIONS = 200

# Metrics whose zero already means "no signal", so the skill arm is the identity
# on them. These are the control families: the rule says they do not move.
ZERO_IS_NO_SIGNAL = ('spearman', 'mcc', 'pearson', 'r2')
# Metrics whose chance value is 0.5.
CHANCE_IS_HALF = ('auc', 'ci')
# Metrics that are an ERROR, scored against a reference model's error.
ERROR_METRICS = ('rmse', 'mae')


# =============================================================================
# THE ARMS
# =============================================================================

def _finite_items(scores):
    return {rn: float(s) for rn, s in scores.items() if np.isfinite(s)}


def _raw_metric_value(score, lower_is_better):
    """Undo the sign flip _score_rep_for_budget applies.

    `_score_rep_for_budget` ends `return -raw if lower_is_better else raw`
    (src/kirby/hybrid.py:2567), so on an rmse dataset the stored score is minus
    the RMSE. This turns it back into the number the metric reports.
    """
    return -score if lower_is_better else score


def reference_levels(y_inner_train, y_inner_val, metric, task_type, seed=SEED):
    """What the skill arm measures each representation against.

    Computed on the SAME held-out rows the per-representation scores were
    measured on — the carve `internal_holdout_split` makes inside the fold's
    training rows, which is the carve create_hybrid itself makes.

    Returns a dict; which keys are filled depends on the metric.
      rmse : the RMSE of predicting the training mean
      mae  : the MAE of predicting the training median
      f1_weighted : the f1 of always predicting the majority training class,
                    and the mean f1 of labels drawn at random with the training
                    class proportions (N_PERMUTATIONS draws, fixed seed)
    """
    y_tr = np.asarray(y_inner_train)
    y_va = np.asarray(y_inner_val)
    out = {}
    if metric == 'rmse':
        out['rmse_ref'] = float(np.sqrt(np.mean((y_va - float(np.mean(y_tr))) ** 2)))
    elif metric == 'mae':
        out['mae_ref'] = float(np.mean(np.abs(y_va - float(np.median(y_tr)))))
    elif metric == 'f1_weighted':
        from sklearn.metrics import f1_score
        classes, counts = np.unique(y_tr, return_counts=True)
        majority = classes[int(np.argmax(counts))]
        out['f1_majority_ref'] = float(
            f1_score(y_va, np.full(y_va.shape, majority), average='weighted'))
        proportions = counts / counts.sum()
        rng = np.random.RandomState(seed)
        draws = [float(f1_score(y_va,
                                rng.choice(classes, size=len(y_va), p=proportions),
                                average='weighted'))
                 for _ in range(N_PERMUTATIONS)]
        out['f1_permutation_ref'] = float(np.mean(draws))
        out['f1_permutation_sd'] = float(np.std(draws))
    return out


def normalise(arm, raw_scores, metric, lower_is_better, refs):
    """One arm's transform of the per-representation score vector.

    Returns (scores, notes). Every returned score is finite and at or above
    zero, so `_sanitize_baseline_scores` leaves it alone. A representation that
    could not be scored (non-finite, from a numerical failure in
    `_evaluate_allocation`) becomes 0.0, which is what the engine's own shift
    does with it.

    `raw` is the exception: it hands the vector back exactly as it came, so the
    engine applies its own shift and this arm IS today's behaviour.
    """
    notes = {}
    if arm == 'raw':
        return dict(raw_scores), notes

    finite = _finite_items(raw_scores)
    if not finite:
        notes['all_non_finite'] = True
        return {rn: 0.0 for rn in raw_scores}, notes

    lo, hi = min(finite.values()), max(finite.values())

    if arm == 'subtract_min':
        out = {rn: (finite[rn] - lo if rn in finite else 0.0) for rn in raw_scores}
    elif arm == 'minmax':
        span = hi - lo
        if span <= 0:
            notes['degenerate_range'] = True
            out = {rn: (1.0 if rn in finite else 0.0) for rn in raw_scores}
        else:
            out = {rn: ((finite[rn] - lo) / span if rn in finite else 0.0)
                   for rn in raw_scores}
    elif arm == 'rank':
        # Worst finite representation gets 1, best gets n. Ties keep the
        # roster order, which is the same for every arm.
        order = sorted(finite.items(), key=lambda kv: kv[1])
        ranks = {rn: float(i) for i, (rn, _s) in enumerate(order, start=1)}
        out = {rn: ranks.get(rn, 0.0) for rn in raw_scores}
    elif arm in ('skill', 'skill_perm'):
        out = {}
        for rn in raw_scores:
            if rn not in finite:
                out[rn] = 0.0
                continue
            value = _raw_metric_value(finite[rn], lower_is_better)
            out[rn] = _skill(arm, value, metric, refs)
    else:
        raise ValueError(f'unknown normalisation arm {arm!r}')

    out = {rn: (float(v) if np.isfinite(v) else 0.0) for rn, v in out.items()}
    n_zero = sum(1 for v in out.values() if v <= 0.0)
    if n_zero:
        # A representation at zero gets no budget at all. Counted, because it is
        # the difference between "weighted less" and "dropped".
        notes['reps_at_zero'] = int(n_zero)
    return {rn: max(0.0, v) for rn, v in out.items()}, notes


def _skill(arm, value, metric, refs):
    """One representation's skill score, in the metric's own terms.

    The standard skill-score construction: 1 - (this error / a reference
    model's error), so a perfect prediction is 1 and no-better-than-the-
    reference is 0.
    """
    if metric in ZERO_IS_NO_SIGNAL:
        return value                      # already zero at no signal
    if metric in CHANCE_IS_HALF:
        return 2.0 * (value - 0.5)
    if metric == 'rmse':
        ref = refs.get('rmse_ref')
        return 0.0 if not ref else 1.0 - (value / ref)
    if metric == 'mae':
        ref = refs.get('mae_ref')
        return 0.0 if not ref else 1.0 - (value / ref)
    if metric == 'f1_weighted':
        ref = refs.get('f1_permutation_ref' if arm == 'skill_perm'
                       else 'f1_majority_ref')
        if ref is None or ref >= 1.0:
            return 0.0
        return (value - ref) / (1.0 - ref)
    raise ValueError(
        f"no skill-score rule for metric {metric!r}. Every metric in the "
        f"registry needs one before it can run here — add it to _skill, or "
        f"say in the result file why the dataset is undefined for this arm.")


def arms_for(metric):
    return list(NORMALISATIONS) + (list(F1_ONLY_NORMALISATIONS)
                                   if metric == 'f1_weighted' else [])


# =============================================================================
# ONE FOLD
# =============================================================================

def _selection_key(info):
    """What was actually selected, as a hashable key.

    Two arms that pick the same features are the same fit. Deduplicating on this
    — rather than on the claim that min-max and subtract-the-minimum coincide —
    means every arm still gets a reported number and no arm is dropped.
    """
    sel = {rn: tuple(int(i) for i in np.asarray(v['selected_indices']).ravel())
           for rn, v in info.items()
           if isinstance(v, dict) and 'selected_indices' in v}
    return tuple(sorted(sel.items()))


def _score_sanity(raw_scores, metric, lower_is_better):
    """Flag scores that cannot be the metric they claim to be.

    `_score_rep_for_budget` falls back to the evaluator's own r2 or accuracy
    when `metric_fn` raises on the predictions, and says nothing when it does.
    A positive score on an rmse dataset, or an AUROC outside [0, 1], is that
    fallback showing through — and the skill arm would then be converting the
    wrong quantity.
    """
    bad = []
    for rn, s in raw_scores.items():
        if not np.isfinite(s):
            continue
        value = _raw_metric_value(s, lower_is_better)
        if metric in ('rmse', 'mae') and value < 0:
            bad.append(rn)
        elif metric in ('auc', 'ci', 'f1_weighted') and not (0.0 <= value <= 1.0):
            bad.append(rn)
        elif metric in ('spearman', 'pearson', 'mcc') and not (-1.0 <= value <= 1.0):
            bad.append(rn)
    return bad


def run_fold(ds, fold_idx, budgets, weightings, out_dir, folds=N_OUTER_FOLDS,
             verbose=True):
    from experiments._shared import load_data, cv_split
    from kirby.hybrid import (create_hybrid, apply_feature_selection,
                              compute_feature_importance,
                              internal_holdout_split)
    from kirby.find_best_hybrid import (get_model, BUILTIN_METRICS,
                                        _predict_for_metric)
    from kirby.registry import require

    info_reg = require(ds)
    metric = info_reg.metric
    lower_is_better = info_reg.lower_is_better
    task_type = info_reg.task_type
    is_clf = (task_type == 'classification')
    metric_fn, _lib = BUILTIN_METRICS[metric]

    data = load_data(ds)
    reps, y, groups = data['reps'], data['labels'], np.asarray(data['groups'])
    if data['task_type'] != task_type:
        raise ValueError(f'{ds}: loader says {data["task_type"]!r}, registry '
                         f'says {task_type!r}')
    X0 = next(iter(reps.values()))

    splits = list(cv_split(groups, folds, X0, y, SEED, task_type=task_type))
    tr, te = splits[fold_idx]
    train_reps = {k: v[tr] for k, v in reps.items()}
    val_reps = {k: v[te] for k, v in reps.items()}
    y_tr, y_te, g_tr = y[tr], y[te], groups[tr]

    # The carve create_hybrid makes inside the training rows. Reproduced here
    # with the same function so the importance is computed on the rows
    # create_hybrid would use, and so the skill arm's reference error is
    # measured on the rows the per-representation scores came from.
    inner_tr, inner_va = internal_holdout_split(len(y_tr), y_tr, g_tr)
    t0 = time.time()
    imp = compute_feature_importance(
        {k: v[inner_tr] for k, v in train_reps.items()},
        y_tr[inner_tr], method=IMPORTANCE_METHOD)
    t_imp = time.time() - t0
    if verbose:
        print(f'[{ds} fold {fold_idx}] importance {t_imp:.1f}s  '
              f'({len(tr)} training rows, {len(te)} held out)', flush=True)

    hybrid_kwargs = dict(
        importance_method=IMPORTANCE_METHOD,
        internal_evaluator=INTERNAL_EVALUATOR,
        precomputed_importance=imp,
        metric_fn=metric_fn,
        lower_is_better=lower_is_better,
        is_clf=is_clf,
        groups=g_tr,
    )

    # One scoring pass. Fitting the internal evaluator once per representation
    # is the expensive part of a build, it does not depend on the allocation
    # method or the budget, and create_hybrid hands the scores back RAW —
    # before its own shift — for exactly this reason (src/kirby/hybrid.py:5583).
    t0 = time.time()
    _X, harvest_info = create_hybrid(
        train_reps, y_tr, allocation_method='performance_weighted',
        budget=budgets[0], **hybrid_kwargs)
    t_scores = time.time() - t0
    raw_scores = harvest_info.get('allocation_baseline_scores')
    if not raw_scores:
        raise RuntimeError(
            f'{ds} fold {fold_idx}: create_hybrid returned no '
            f'allocation_baseline_scores. Only the performance_weighted branch '
            f'fills them; nothing downstream here can run without them.')
    raw_scores = {rn: float(s) for rn, s in raw_scores.items()}
    suspect = _score_sanity(raw_scores, metric, lower_is_better)
    if verbose:
        print(f'[{ds} fold {fold_idx}] scored {len(raw_scores)} representations '
              f'in {t_scores:.1f}s, metric {metric}', flush=True)
        if suspect:
            print(f'  WARNING: {len(suspect)} scores are outside what {metric} '
                  f'can return: {suspect}', flush=True)

    refs = reference_levels(y_tr[inner_tr], y_tr[inner_va], metric, task_type)

    rows = []
    fits = {}          # selection key -> the scores that fit already produced
    n_fits = 0
    for arm in arms_for(metric):
        scores, notes = normalise(arm, raw_scores, metric, lower_is_better, refs)
        for budget in budgets:
            for weighting in weightings:
                t0 = time.time()
                try:
                    X_tr_h, binfo = create_hybrid(
                        train_reps, y_tr, allocation_method=weighting,
                        budget=budget,
                        precomputed_alloc_scores=scores, **hybrid_kwargs)
                except Exception as e:
                    rows.append({'dataset': ds, 'fold': fold_idx, 'arm': arm,
                                 'weighting': weighting, 'budget': budget,
                                 'error': f'{type(e).__name__}: {e}'})
                    if verbose:
                        print(f'  {arm}/{weighting}/b{budget}: BUILD FAILED '
                              f'{type(e).__name__}: {e}', flush=True)
                    continue
                t_build = time.time() - t0
                allocation = {k: int(v) for k, v in
                              (binfo.get('allocation') or {}).items()}
                key = _selection_key(binfo)

                # An empty key would collide across every cell and silently
                # reuse one fit for all of them, so it never dedupes.
                if key and key in fits:
                    primary, t_fit, reused = fits[key], 0.0, True
                else:
                    reused = False
                    X_te_h = apply_feature_selection(val_reps, binfo)
                    t1 = time.time()
                    primary = _fit_and_score(
                        X_tr_h, X_te_h, y_tr, y_te, task_type, metric_fn,
                        get_model, _predict_for_metric)
                    t_fit = time.time() - t1
                    if key:
                        fits[key] = primary
                    n_fits += 1

                rows.append({
                    'dataset': ds, 'fold': fold_idx, 'arm': arm,
                    'weighting': weighting, 'budget': budget,
                    'metric': metric, 'primary': primary,
                    'n_features': int(X_tr_h.shape[1]),
                    'allocation': allocation,
                    'reps_with_no_features': int(
                        sum(1 for v in allocation.values() if v == 0)),
                    'normalised_scores': {
                        k: (round(float(v), 6) if np.isfinite(v) else None)
                        for k, v in scores.items()},
                    'arm_notes': notes,
                    'selection_reused': reused,
                    'build_seconds': round(t_build, 2),
                    'fit_seconds': round(t_fit, 2),
                    'error': None,
                })
                if verbose:
                    shown = 'reused' if reused else f'{t_fit:.1f}s'
                    print(f'  {arm}/{weighting}/b{budget}: '
                          f'{metric}={primary if primary is None else round(primary, 4)}'
                          f'  ({shown})', flush=True)

    checks = _check_invariants(ds, fold_idx, rows, raw_scores, metric)

    payload = {
        'dataset': ds, 'fold': fold_idx, 'metric': metric,
        'lower_is_better': bool(lower_is_better), 'task_type': task_type,
        'budgets': list(budgets), 'weightings': list(weightings),
        'arms': arms_for(metric),
        'downstream_model': DOWNSTREAM_MODEL,
        'internal_evaluator': INTERNAL_EVALUATOR,
        'importance_method': IMPORTANCE_METHOD,
        'n_outer_folds': folds, 'seed': SEED,
        'reps': sorted(reps), 'n_samples': int(len(y)),
        'n_train_rows': int(len(tr)), 'n_heldout_rows': int(len(te)),
        'raw_scores': {k: (float(v) if np.isfinite(v) else None)
                       for k, v in raw_scores.items()},
        'raw_scores_outside_metric_range': suspect,
        'skill_reference_levels': refs,
        'auc_on_hard_labels': (metric == 'auc'),
        'checks': checks,
        'scoring_pass_seconds': round(t_scores, 1),
        'importance_seconds': round(t_imp, 1),
        'downstream_fits': n_fits,
        'cells': len(rows),
        'written': datetime.now().isoformat(timespec='seconds'),
        'rows': rows,
    }
    out = Path(out_dir) / f'f32_{ds}__fold{fold_idx}.json'
    out.write_text(json.dumps(payload, indent=1))
    if verbose:
        print(f'[{ds} fold {fold_idx}] {len(rows)} cells, {n_fits} downstream '
              f'fits, -> {out}', flush=True)
    return payload


def _fit_and_score(X_tr, X_te, y_tr, y_te, task_type, metric_fn,
                   get_model, predict_for_metric):
    from sklearn.preprocessing import StandardScaler
    a = np.nan_to_num(np.asarray(X_tr, dtype=float), nan=0.0,
                      posinf=1e10, neginf=-1e10)
    b = np.nan_to_num(np.asarray(X_te, dtype=float), nan=0.0,
                      posinf=1e10, neginf=-1e10)
    sc = StandardScaler()
    a, b = sc.fit_transform(a), sc.transform(b)
    model = get_model(DOWNSTREAM_MODEL, task_type=task_type, n_samples=len(y_tr))
    if model is None:
        return None
    try:
        model.fit(a, y_tr)
        pred = predict_for_metric(model, b, metric_fn)
        value = float(metric_fn(np.asarray(y_te), np.asarray(pred)))
    except Exception:
        return None
    return value if np.isfinite(value) else None


# =============================================================================
# THE TWO CONTROLS
# =============================================================================

def _check_invariants(ds, fold_idx, rows, raw_scores, metric):
    """Two checks the arms must satisfy, and the conditions under which they do.

    1. `rank_exp_d0.3` reads only the ORDER of the scores, so its allocation
       must be identical under every arm. It can legitimately differ when an arm
       CLIPS A SCORE to zero, because two representations then tie at zero and
       the order between them falls back to the roster. MEASURED on
       tcga_brca_pam50 fold 2: mirna_top200's f1_weighted is 0.25899 and the f1
       of always predicting the majority class is 0.25899, so its skill is
       exactly zero; wsi_ctranspath sits below the same reference and is also
       zero. The two swapped 7 features for 5 at a budget of 300. So the check
       is exact only when no arm clipped a score, and is reported rather than
       asserted otherwise.

    2. On a metric whose zero already means no signal, the `skill` arm is the
       identity, so it must give the same allocation as `raw` — for every
       weighting. It can legitimately differ when a raw score is NEGATIVE: the
       engine's own shift then moves the whole vector up while `skill` clips at
       zero. That divergence is a real difference in today's code, not a harness
       fault, so it is recorded with the count of negative scores that caused it.
    """
    ok = {}
    finite = [s for s in raw_scores.values() if np.isfinite(s)]
    n_negative = int(sum(1 for s in finite if s < 0))

    def allocations(arm=None, weighting=None):
        return {(r['arm'], r['weighting'], r['budget']): r['allocation']
                for r in rows if not r.get('error')
                and (arm is None or r['arm'] == arm)
                and (weighting is None or r['weighting'] == weighting)}

    # -- 1. the order-only allocation must not move --
    rank_rows = [r for r in rows
                 if r.get('weighting') == 'rank_exp_d0.3' and not r.get('error')]
    by_budget = {}
    for r in rank_rows:
        by_budget.setdefault(r['budget'], []).append(r)
    moved = []
    for budget, group in by_budget.items():
        first = group[0]['allocation']
        for r in group[1:]:
            if r['allocation'] != first:
                moved.append({'budget': budget, 'arm': r['arm']})
    # Strictness is about scores clipped to zero, NOT about representations
    # that ended up with no features. Those are different things: at a budget of
    # 300 every representation gets features even when two of them scored zero.
    clipped = sum((r.get('arm_notes') or {}).get('reps_at_zero', 0)
                  for r in rank_rows)
    ok['rank_exp_split_unchanged'] = {
        'arms_that_moved': moved,
        'holds': not moved,
        'strict': clipped == 0,
        'n_scores_clipped_to_zero': int(clipped),
        'note': (f'{clipped} scores were clipped to zero across the arms, and '
                 f'two representations tied at zero take the roster order, so a '
                 f'move here is expected rather than wrong' if clipped else
                 'no score was clipped to zero, so this is exact'),
    }

    # -- 2. the control families must not move under the skill arm --
    ok['skill_matches_raw'] = None
    if metric in ZERO_IS_NO_SIGNAL:
        diffs = []
        for (arm, weighting, budget), alloc in allocations(arm='skill').items():
            raw_alloc = allocations(arm='raw').get(('raw', weighting, budget))
            if raw_alloc is not None and raw_alloc != alloc:
                diffs.append({'weighting': weighting, 'budget': budget})
        ok['skill_matches_raw'] = {
            'holds': not diffs,
            'differs_on': diffs,
            'strict': n_negative == 0,
            'n_negative_raw_scores': n_negative,
            'note': (f'{n_negative} representations scored below zero, so the '
                     f"engine's shift moved the whole vector and skill clipped "
                     f'instead — a real difference, not a harness fault'
                     if n_negative else
                     'every score was at or above zero, so this is exact'),
        }
    return ok


def check_pools(datasets):
    """Load each dataset's representations and say how many came back.

    This is the gate the SLURM `check` stage runs before the array. Two of the
    nine were never staged on this machine — tcga_luad_vs_lusc and
    gb_human_enhancers_cohn — so a missing pool shows up here, by name, rather
    than as a task that dies twenty minutes into the grid.
    """
    from experiments._shared import load_data
    missing = []
    for ds in datasets:
        try:
            data = load_data(ds)
            reps = data['reps']
            widths = ', '.join(f'{k}={v.shape[1]}' for k, v in
                               sorted(reps.items()))
            print(f'  {ds}: {len(reps)} representations, '
                  f'{len(data["labels"])} rows ({widths})')
        except Exception as e:
            print(f'  {ds}: POOL MISSING — {type(e).__name__}: {e}')
            missing.append(ds)
    return missing


def check_pairs(datasets):
    """Which metric each dataset reports, and whether the pairs still hold.

    The dataset list in this file carries no metric of its own. This reads the
    registry and prints the grouping, so a registry edit that regroups these
    nine shows up here rather than in the analysis.
    """
    from experiments import dataset_policy  # noqa: F401  (populates DATASET_REGISTRY)
    from kirby.registry import require
    by_metric = {}
    for ds in datasets:
        info = require(ds)
        by_metric.setdefault(info.metric, []).append(ds)
    unpaired = []
    for metric, names in sorted(by_metric.items()):
        mark = '' if len(names) >= 2 else '   <- ONE DATASET, cannot separate metric from dataset'
        print(f'  {metric:12s} {len(names)} dataset(s): {", ".join(names)}{mark}')
        if len(names) < 2:
            unpaired.append(metric)
    return by_metric, unpaired


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--folds', type=int, default=N_OUTER_FOLDS)
    ap.add_argument('--fold', type=int, default=None,
                    help='run one fold only (the SLURM array does this)')
    ap.add_argument('--budgets', nargs='+', type=int, default=BUDGETS)
    ap.add_argument('--weightings', nargs='+', default=WEIGHTINGS)
    ap.add_argument('--smoke', action='store_true',
                    help='caco2, fold 0, one budget — minutes, not a grid')
    ap.add_argument('--check', action='store_true',
                    help='print the metric pairing and run one fold with the '
                         'controls, without writing a grid')
    ap.add_argument('--output',
                    default=os.environ.get(
                        'F32_OUT',
                        str(_ROOT / 'tests' / 'results' / 'framework')))
    args = ap.parse_args()

    from experiments import dataset_policy  # noqa: F401  (populates DATASET_REGISTRY)

    print('metric pairing, read from the registry:')
    _by_metric, unpaired = check_pairs(args.datasets)
    if unpaired:
        print(f'  metrics with one dataset: {", ".join(unpaired)}')

    if args.check:
        print('\nrepresentation pools:')
        missing = check_pools(args.datasets)
        if missing:
            print(f'\nREFUSING to run: no representation pool for '
                  f'{", ".join(missing)}. Precompute them — dropping a metric '
                  f'family leaves the question it was there to answer open.')
            return 2

    if args.smoke:
        args.datasets = ['caco2']
        args.fold = 0
        args.budgets = [args.budgets[0]]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = [args.fold] if args.fold is not None else list(range(args.folds))
    failed = []
    for ds in args.datasets:
        for f in folds:
            print(f'\n=== {ds} fold {f} ===', flush=True)
            t0 = time.time()
            try:
                payload = run_fold(ds, f, args.budgets, args.weightings,
                                   out_dir, folds=args.folds)
            except Exception:
                traceback.print_exc()
                failed.append((ds, f))
                continue
            for name, result in (payload.get('checks') or {}).items():
                if result is None:
                    continue
                verdict = 'holds' if result.get('holds') else 'FAILS'
                strict = 'exact' if result.get('strict') else 'conditional'
                print(f'  check {name}: {verdict} ({strict}) — '
                      f'{result.get("note", "")}', flush=True)
            print(f'=== {ds} fold {f} done in {(time.time()-t0)/60:.1f} min ===',
                  flush=True)
            if args.check:
                break
        if args.check:
            break
    if failed:
        print(f'\nFAILED: {failed}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
