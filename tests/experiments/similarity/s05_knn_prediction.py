#!/usr/bin/env python3
"""S05: kNN Prediction with Hybrid Distance.

Paper: Similarity
Question: does predicting each molecule's activity from its nearest neighbours
in a hybrid's features beat doing the same in a single representation's
features?

What is computed (2026-10-05, HANDOFF_EXPERIMENT_ALIGNMENT.md Part 2, chat 4;
group (C) of section 2.3, "switch to 2 + 2, with distance hybrids where
neighbours are used").

Per outer fold (sim_outer_folds, count = config['n_splits']; 1 is one held-out
split), everything is chosen on that fold's TRAINING rows by
fold_similarity_selection(..., neighbour_spaces=True), which runs the per-fold
searches on the dataset's decided grid (DECISIONS.md), pruning off:
  single_knn1, single_knn2   knn's 2 best single representations, ranked by
                             the per-fold distance search (knn_base)
  general_<model>_1, _2      each decided model's 2 best general hybrids (a
                             second hybrid with the same selected features as
                             the first is skipped by the selection)
  distance_hybrid            knn's top hybrid of the per-fold distance search
  similarity_hybrid          the top hybrid of the per-fold similarity-objective
                             search (features chosen so distance follows
                             activity difference)
Each arm is a feature matrix built on the fold's training rows
(build_fold_entry, with the fold search's own representation scores) and
scaled with FeatureScaler fitted on the training rows (0/1 features
unscaled, the rest standardised, held-out values cut to the training range).
Every arm is then scored on the held-out rows by predicting each molecule from
its nearest training molecules at every decided kNN setting (knn_axes:
the dataset group's distances, 5/10/20 neighbours, rank and inverse_square
counting, the scaffold group not excluded), in two conditions:
  unweighted            the scaled features
  importance_weighted   the scaled features multiplied by the importance
                        weights (importance_feature_weights). For a hybrid the
                        importance is the recipe's own (flatten_fold_weights);
                        for a single representation it is computed on the
                        fold's training rows with SINGLE_IMPORTANCE_PREFERENCE
                        below, restricted to the dataset's decided methods.
Importance weighting in a similarity experiment's own distance (its power and
its rescaling) is still open (handoff 2.10); the code and power are as they were.

DECIDED 2026-10-06 (Adelaide delegated): no budget axis. Each hybrid is built
at the budget its per-fold search chose; the 3 decided distance budgets are not
swept here (sweeping them would multiply hybrid builds and kNN sweeps by 3).

Numbers written per fold: every kNN score carries its distance, neighbour count
and counting scheme. Across folds: per-fold scores, mean and spread, never a
test. The held-out comparison (fold_reselection_result, s09's
knn_delta_heldout) re-picks the hybrid cell and the single-representation cell
for each held-out fold from the other folds' held-out scores; no fold is
dropped.

Removed on 2026-10-05: the stored whole-dataset pool reads
(get_fold_pool_recipes, get_rule_hybrids), the budget loop over 10-500
features (each hybrid is now built at the budget its per-fold search chose),
the kNN score of every loaded representation, the reference table over six
models read from stored scores (DISTANCE_MODELS), the named reference
representations for herg_ki and logd, the k_values argument and
KIRBY_S05_K_VALUES, KIRBY_S05_KNN_VOTE and sklearn's equal-vote default, the
process-wide distance (KIRBY_DISTANCE_METRIC), StandardScaler, and the rule
that dropped fold 0 from the held-out comparison.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from experiments.similarity.similarity_helpers import _CODE_FAULTS

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, get_domain_metric, get_model_factories,
    flatten_fold_weights, importance_feature_weights, fold_reselection,
)
from experiments.decided_grid import decided_models, decided_search, is_decided
from experiments.similarity.similarity_helpers import (
    sim_outer_folds, fold_spread_note, knn_axes, scale_fold, knn_sweep,
    best_knn_setting, fold_similarity_selection, build_fold_entry,
    entry_record,
)
from kirby.hybrid import compute_feature_importance
from experiments.similarity._summary_keys import (
    SELECTION_RULE as _SELECTION_RULE)


CONDITIONS = ('unweighted', 'importance_weighted')

# Bumped whenever what a fold unit holds changes, so a sidecar written under
# the old layout (budget units and baseline_rep units over the stored pool) is
# moved aside instead of read back.
CHECKPOINT_VERSION = 's05_per_fold_selection_2026_10_05b'

# Importance method for the importance_weighted single-representation arms.
# Until 2026-10-05 this was lightgbm_gain on every dataset; DECISIONS.md Step 3
# cuts lightgbm_gain on several (buchwald_hartwig, cdr_flexibility,
# ginkgo_pr_cho, ginkgo_titer among them). The first name here that the
# dataset's distance search decided is used (the distance search is the one
# knn_base is ranked by); if neither is decided, the first decided method.
SINGLE_IMPORTANCE_PREFERENCE = ('lightgbm_gain', 'treeshap')

# The pre-registered comparison s09 reads as knn_delta_fixed_ref: the distance
# hybrid against knn's best single representation, both unweighted, at the
# dataset group's first distance (manhattan in both groups), 5 neighbours, rank
# counting. Fixed before the run; nothing is picked on the folds it is scored on.
FIXED_HYBRID_ARM = 'distance_hybrid'
FIXED_SINGLE_ARM = 'single_knn1'
FIXED_CONDITION = 'unweighted'
FIXED_K = 5
FIXED_SCHEME = 'rank'


def single_importance_method(dataset_name):
    """(method, tier it was read from, the decided list)."""
    tier = ('general_distance' if is_decided(dataset_name, 'general_distance')
            else 'general')
    methods = list(decided_search(dataset_name, tier)['importance_methods'])
    for pref in SINGLE_IMPORTANCE_PREFERENCE:
        if pref in methods:
            return pref, tier, methods
    return methods[0], tier, methods


def _cell_name(arm, condition, row):
    return (f"{arm}|{condition}|{row['distance']}|k{row['k']}"
            f"|{row['scheme']}")


def _mean_std(values):
    vals = [float(v) for v in values
            if v is not None and np.isfinite(v)]
    if not vals:
        return None, None, 'no fold scored'
    note = fold_spread_note(vals)
    return (float(np.mean(vals)),
            (None if note else float(np.std(vals))), note)


# =============================================================================
# EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """kNN prediction in each per-fold arm at every decided kNN setting."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S05: kNN PREDICTION WITH HYBRID DISTANCE")
    print("=" * 70)

    reps = data['reps']
    labels = np.asarray(data['labels'])
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    axes = knn_axes(dataset_name, len(labels))
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    imp_method, imp_tier, imp_decided = single_importance_method(dataset_name)

    folds = sim_outer_folds(data, config)
    n_folds = len(folds)

    results = {
        'selection_rule': _SELECTION_RULE,
        'selection_scope': 'fold_train_rows',
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'n_outer_folds': n_folds,
        'models': models,
        'knn_settings': dict(axes),
        'conditions': list(CONDITIONS),
        'single_importance': {
            'method': imp_method,
            'read_from_tier': imp_tier,
            'decided_methods': imp_decided,
            'preference': list(SINGLE_IMPORTANCE_PREFERENCE),
            'note': ('importance method for the importance_weighted single '
                     'representation arms; a hybrid arm is weighted by its own '
                     "recipe's importance"),
        },
        'importance_exponent_note': (
            'importance_feature_weights: the clipped importance raised to '
            'KIRBY_IMPORTANCE_EXPONENT (0.5 when unset) multiplies the scaled '
            'features. Open, handoff 2.10.'),
        'per_fold': {},
    }
    print(f"  metric={metric_name} (lower_is_better={lower_is_better}), "
          f"{n_folds} outer folds, models={models}")
    print(f"  kNN settings ({axes['group']}): distances={axes['metrics']}, "
          f"neighbours={axes['ks']}, counting={axes['schemes']}, scaffold "
          f"group not excluded")
    print(f"  single-representation importance: {imp_method} "
          f"(decided on the {imp_tier} search: {imp_decided})")

    factories = {}
    for m in list(models) + ['knn']:
        try:
            f = get_model_factories([m], task_type=task_type,
                                    n_samples=len(labels)).get(m)
        except Exception:
            f = None
        factories[m] = f

    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)
    _ck = Checkpoint(
        experiment_sidecar(output_dir, 's05_knn_prediction', dataset_name),
        run_identity('s05_knn_prediction', dataset_name, data, config,
                     kwargs=kwargs, version=CHECKPOINT_VERSION,
                     knn_axes={k: v for k, v in axes.items() if k != 'source'},
                     conditions=list(CONDITIONS),
                     single_importance=imp_method))

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        if _ck.done(fold=fold_i):
            results['per_fold'][fold_i] = _ck.get(fold=fold_i)
            print(f"\n  Fold {fold_i}: read back from the partial file")
            continue
        print(f"\n  Fold {fold_i} ({len(train_idx)} training rows, "
              f"{len(val_idx)} held-out rows)")
        rec, threw = _run_fold(
            data, dataset_name, fold_i, n_folds, train_idx, val_idx,
            models=models, axes=axes, factories=factories,
            imp_method=imp_method, metric_name=metric_name,
            metric_fn=metric_fn, lower_is_better=lower_is_better,
            is_clf=is_clf)
        results['per_fold'][fold_i] = rec
        if not threw:
            _ck.put(rec, fold=fold_i)

    _summarise(results, n_folds, axes, lower_is_better, metric_name)
    return results


def _run_fold(data, dataset_name, fold_i, n_folds, train_idx, val_idx, *,
              models, axes, factories, imp_method, metric_name, metric_fn,
              lower_is_better, is_clf):
    """One outer fold: select on the training rows, score every arm on the
    held-out rows. Returns (record, threw)."""
    reps = data['reps']
    labels = np.asarray(data['labels'])
    y_tr, y_va = labels[train_idx], labels[val_idx]
    threw = False

    rec = {
        'fold': int(fold_i),
        'n_train': int(len(train_idx)),
        'n_val': int(len(val_idx)),
        'selection_scope': 'fold_train_rows',
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'models': {},
        'knn_base': [],
        'knn_base_train_scores': {},
        'distance_hybrid': None,
        'similarity_hybrid': None,
        'unavailable': {},
        'arms': {},
        'arms_missing': {},
    }

    try:
        sel = fold_similarity_selection(
            data, dataset_name, train_idx=train_idx, fold_i=fold_i,
            n_outer_folds=n_folds, neighbour_spaces=True)
    except _CODE_FAULTS:
        raise  # a fault in the code, not a failed search
    except Exception as exc:
        rec['selection_error'] = f'{type(exc).__name__}: {exc}'
        print(f"    [warn] selection raised {rec['selection_error']}")
        return rec, True

    rec['unavailable'] = dict(sel.get('unavailable') or {})
    for m in models:
        s = (sel.get('models') or {}).get(m)
        if s is None:
            rec['models'][m] = None
            continue
        bd = s.get('base_detail') or {}
        rec['models'][m] = {
            'base': list(s.get('base') or []),
            'base_train_scores': {e['rep']: e.get('score')
                                  for e in (bd.get('selected') or [])},
            'hybrids': [entry_record(h) for h in (s.get('hybrids') or [])],
            'n_hybrids_skipped_identical_features': len(
                s.get('hybrids_skipped_identical_features') or []),
        }
    rec['knn_base'] = list(sel.get('knn_base') or [])
    rec['knn_base_train_scores'] = {
        e['rep']: e.get('score')
        for e in ((sel.get('knn_base_detail') or {}).get('selected') or [])}
    rec['distance_hybrid'] = entry_record(sel.get('distance_hybrid'))
    rec['similarity_hybrid'] = entry_record(sel.get('similarity_hybrid'))

    # The arms this fold should have, and what fills each.
    arms = []
    for i in range(2):
        name = f'single_knn{i + 1}'
        if i < len(rec['knn_base']):
            arms.append((name, 'single', rec['knn_base'][i], None))
        else:
            rec['arms_missing'][name] = (
                rec['unavailable'].get('knn_base')
                or rec['unavailable'].get('distance_search')
                or 'the per-fold distance search gave knn fewer than '
                   f'{i + 1} single representations')
    for m in models:
        s = (sel.get('models') or {}).get(m)
        hyb = (s or {}).get('hybrids') or []
        for j in range(2):
            name = f'general_{m}_{j + 1}'
            if j < len(hyb):
                arms.append((name, 'hybrid', hyb[j], m))
            else:
                rec['arms_missing'][name] = (
                    rec['unavailable'].get(f'general/{m}')
                    or f'the per-fold general search gave {m} fewer than '
                       f'{j + 1} hybrids with distinct selected features')
    for name, key, model in (('distance_hybrid', 'distance_hybrid', 'knn'),
                             ('similarity_hybrid', 'similarity_hybrid', None)):
        if sel.get(key) is not None:
            arms.append((name, 'hybrid', sel[key], model))
        else:
            rec['arms_missing'][name] = (
                rec['unavailable'].get(key)
                or rec['unavailable'].get('distance_search')
                or 'not selected')

    for name, kind, payload, model in arms:
        arm = {'kind': kind, 'conditions': {}, 'conditions_unavailable': {}}
        # Build and scale the arm's features, then sweep the unweighted
        # condition. Nothing here depends on importance, so an importance
        # failure below cannot remove these scores (summary.fixed_ref and
        # s09's knn_delta_fixed_ref are built on the unweighted cells).
        try:
            if kind == 'single':
                arm['representation'] = payload
                X = np.asarray(reps[payload])
                A, B = scale_fold(X[train_idx], X[val_idx])
                feat_info = None
            else:
                arm['hybrid'] = entry_record(payload)
                arm['selected_for_model'] = model
                X_tr_h, X_va_h, feat_info = build_fold_entry(
                    data, payload, train_idx=train_idx, val_idx=val_idx,
                    model_factory=factories.get(model) if model else None,
                    metric_fn=metric_fn, lower_is_better=lower_is_better)
                A, B = scale_fold(X_tr_h, X_va_h)
            arm['width_built'] = int(A.shape[1])
            rows = knn_sweep(A, y_tr, B, y_va, axes=axes,
                             metric_name=metric_name, metric_fn=metric_fn,
                             is_clf=is_clf, scale=False)
            arm['conditions']['unweighted'] = {
                'rows': rows,
                'best_on_this_fold': best_knn_setting(rows, lower_is_better),
            }
        except Exception as exc:
            threw = True
            arm['error'] = f'{type(exc).__name__}: {exc}'
            print(f"    [warn] {name} failed: {arm['error'][:120]}")
            rec['arms'][name] = arm
            continue
        b = arm['conditions']['unweighted']['best_on_this_fold']
        print(f"    {name:<22} width={arm['width_built']:>5}  best "
              f"unweighted {metric_name}="
              + (f"{b['score']:.4f} ({b['distance']}, k={b['k']}, "
                 f"{b['scheme']})" if b else 'none'))

        # Importance-weighted condition. If the importance call fails or its
        # width does not match the arm's features, only this condition is
        # recorded as unavailable, with the reason; the unweighted scores
        # above stay, and the fold is still checkpointed.
        try:
            if kind == 'single':
                imp = compute_feature_importance(
                    {payload: X[train_idx]}, y_tr, method=imp_method)
                w = importance_feature_weights(imp[payload])
                arm['importance_method'] = imp_method
            else:
                w = importance_feature_weights(flatten_fold_weights(feat_info))
                arm['importance_method'] = arm['hybrid'].get('importance')
            w = np.asarray(w, dtype=np.float64)
            if w.shape[0] != A.shape[1]:
                raise ValueError(f'weight/feature mismatch: w={w.shape[0]} '
                                 f'vs X={A.shape[1]}')
            rows_w = knn_sweep(A * w, y_tr, B * w, y_va, axes=axes,
                               metric_name=metric_name, metric_fn=metric_fn,
                               is_clf=is_clf, scale=False)
            arm['conditions']['importance_weighted'] = {
                'rows': rows_w,
                'best_on_this_fold': best_knn_setting(rows_w,
                                                      lower_is_better),
            }
        except Exception as exc:
            why = f'{type(exc).__name__}: {exc}'
            arm['conditions_unavailable']['importance_weighted'] = why
            print(f"    [warn] {name} importance_weighted not available: "
                  f"{why[:120]}")
        rec['arms'][name] = arm

    for name, why in rec['arms_missing'].items():
        print(f"    {name}: not available ({why})")
    return rec, threw


def _summarise(results, n_folds, axes, lower_is_better, metric_name):
    """Across folds: one cell per (arm, condition, distance, neighbour count,
    counting scheme), its per-fold held-out scores, mean and spread; the
    held-out re-pick; and the two cells s09 reads."""
    cells = {}
    for fold_i, rec in results['per_fold'].items():
        fi = int(fold_i)
        for arm, a in (rec.get('arms') or {}).items():
            for cond, c in (a.get('conditions') or {}).items():
                for row in c.get('rows') or []:
                    name = _cell_name(arm, cond, row)
                    cell = cells.setdefault(name, {
                        'arm': arm, 'kind': a.get('kind'), 'condition': cond,
                        'distance': row['distance'], 'k': row['k'],
                        'scheme': row['scheme'], 'metric': metric_name,
                        'fold_scores': [None] * n_folds})
                    cell['fold_scores'][fi] = row.get('score')
    for cell in cells.values():
        m, s, note = _mean_std(cell['fold_scores'])
        cell['score_mean'], cell['score_std'] = m, s
        cell['n_folds_scored'] = sum(
            1 for v in cell['fold_scores']
            if v is not None and np.isfinite(v))
        if note:
            cell['spread_note'] = note
    results['cells'] = cells

    # Arms with no fold at all, and why.
    reasons = {}
    for rec in results['per_fold'].values():
        for arm, why in (rec.get('arms_missing') or {}).items():
            reasons.setdefault(arm, [])
            if why not in reasons[arm]:
                reasons[arm].append(why)
        for arm, a in (rec.get('arms') or {}).items():
            if a.get('error'):
                reasons.setdefault(arm, [])
                if a['error'] not in reasons[arm]:
                    reasons[arm].append(a['error'])
            for cond, why in (a.get('conditions_unavailable') or {}).items():
                key = f'{arm}/{cond}'
                reasons.setdefault(key, [])
                if why not in reasons[key]:
                    reasons[key].append(why)
    results['arm_reasons_missing'] = reasons

    # Held-out re-pick: hybrid cells (both conditions) against the single
    # representation cells (unweighted, as the baseline side has been since
    # the per-fold reselection was added). Every fold counts.
    hybrid_cands = [{'name': n, 'fold_scores': list(c['fold_scores'])}
                    for n, c in cells.items() if c['kind'] == 'hybrid']
    base_cands = {n: list(c['fold_scores']) for n, c in cells.items()
                  if c['kind'] == 'single' and c['condition'] == 'unweighted'}
    results['fold_reselection'] = {
        'lower_is_better': bool(lower_is_better),
        'recipe_selection': 'fold_train_rows',
        'exclude_folds': [],
        'candidates_by_model': {'knn': hybrid_cands} if hybrid_cands else {},
        'baselines_by_model': {'knn': base_cands} if base_cands else {},
        'note': ('every hybrid and representation was chosen on its own '
                 "fold's training rows; the re-pick is over kNN cells (arm, "
                 'condition, distance, neighbour count, counting scheme) by '
                 "the other folds' held-out scores. Baseline side: knn's two "
                 'single representations, unweighted.'),
    }
    if not hybrid_cands or not base_cands:
        results['fold_reselection_status'] = {
            'computed': False,
            'n_hybrid_cells': len(hybrid_cands),
            'n_baseline_cells': len(base_cands),
            'reason': ('no hybrid cell was scored' if not hybrid_cands
                       else 'no single-representation cell was scored'),
        }
        print(f"\n  Held-out re-pick: not computed "
              f"({results['fold_reselection_status']['reason']})")
    else:
        results['fold_reselection_status'] = {
            'computed': True, 'n_hybrid_cells': len(hybrid_cands),
            'n_baseline_cells': len(base_cands), 'exclude_folds': [],
            'reason': None}
        try:
            hr = fold_reselection(
                results['fold_reselection']['candidates_by_model'],
                results['fold_reselection']['baselines_by_model'],
                bool(lower_is_better), exclude_folds=[],
                recipe_selection='fold_train_rows')
            results['fold_reselection_result'] = hr
            k = hr.get('knn')
            if k is not None:
                flag = (' [one fold: plain best, no held-out re-pick]'
                        if k.get('single_fold') else '')
                print(f"\n  Held-out re-pick (knn){flag}: "
                      f"hybrid={k['hybrid_reselected']:.4f}, "
                      f"single={k['best_single_reselected']:.4f}, "
                      f"difference={k['delta']:+.4f} (positive = hybrid "
                      f"better, {metric_name})")
        except Exception as e:
            print(f"  [warn] fold_reselection failed: {type(e).__name__}: {e}")

    sign = -1 if lower_is_better else 1
    summary = {}
    # Pre-registered cell (s09: knn_delta_fixed_ref).
    fixed_dist = axes['metrics'][0]
    hk = _cell_name(FIXED_HYBRID_ARM, FIXED_CONDITION,
                    {'distance': fixed_dist, 'k': FIXED_K,
                     'scheme': FIXED_SCHEME})
    bk = _cell_name(FIXED_SINGLE_ARM, FIXED_CONDITION,
                    {'distance': fixed_dist, 'k': FIXED_K,
                     'scheme': FIXED_SCHEME})
    hc, bc = cells.get(hk), cells.get(bk)
    if hc and bc and hc['score_mean'] is not None and bc['score_mean'] is not None:
        per_fold = [(sign * (h - b)) if (h is not None and b is not None
                                         and np.isfinite(h) and np.isfinite(b))
                    else None
                    for h, b in zip(hc['fold_scores'], bc['fold_scores'])]
        paired = [d for d in per_fold if d is not None]
        d_mean = float(np.mean(paired)) if paired else None
        summary['fixed_ref'] = {
            'hybrid_cell': hk, 'single_cell': bk,
            'condition': FIXED_CONDITION, 'k': FIXED_K,
            'distance': fixed_dist, 'scheme': FIXED_SCHEME,
            'budget': None,
            'hybrid_score_mean': hc['score_mean'],
            'single_score_mean': bc['score_mean'],
            'per_fold_delta': per_fold,
            'delta_vs_best_rep': d_mean,
            'n_paired_folds': len(paired),
            'note': ("distance hybrid against knn's best single "
                     'representation, both unweighted, same distance, '
                     'neighbour count and counting scheme, fixed before the '
                     'run; mean of per-fold differences, positive = hybrid '
                     'better. budget is per fold, in the hybrid record.'),
        }
        print(f"  Fixed cell {hk} vs {bk}: difference "
              + (f"{d_mean:+.4f}" if d_mean is not None else 'n/a')
              + f" over {len(paired)} folds")
    else:
        _why = (results['arm_reasons_missing'].get(FIXED_HYBRID_ARM, [])
                + results['arm_reasons_missing'].get(FIXED_SINGLE_ARM, []))
        summary['fixed_ref_status'] = {
            'computed': False,
            'reason': (f"cell {hk} or {bk} was not scored "
                       f"({'; '.join(_why) or 'no fold'})"),
        }

    # Optimistic maximum (s09: knn_delta_posthoc_max): best hybrid cell mean
    # minus best single-representation cell mean, both picked on the folds
    # they are scored on.
    def _best(kind, cond=None):
        ok = [c for c in cells.values() if c['kind'] == kind
              and c['score_mean'] is not None
              and (cond is None or c['condition'] == cond)]
        if not ok:
            return None
        return (min if lower_is_better else max)(
            ok, key=lambda c: c['score_mean'])
    bh, bs = _best('hybrid'), _best('single', 'unweighted')
    if bh and bs:
        summary['posthoc_max'] = {
            'arm': bh['arm'], 'condition': bh['condition'],
            'distance': bh['distance'], 'k': bh['k'], 'scheme': bh['scheme'],
            'budget': None,
            'score_mean': bh['score_mean'],
            'single_arm': bs['arm'], 'single_distance': bs['distance'],
            'single_k': bs['k'], 'single_scheme': bs['scheme'],
            'single_score_mean': bs['score_mean'],
            'delta_vs_best_rep': sign * (bh['score_mean'] - bs['score_mean']),
            'note': ('best hybrid cell minus best unweighted single '
                     'representation cell, each the maximum over its cells on '
                     'the folds it is scored on; optimistic, not the headline'),
        }
    else:
        summary['posthoc_max_status'] = {
            'computed': False,
            'reason': 'no hybrid cell or no single-representation cell'}
    results['summary'] = summary


if __name__ == '__main__':
    parser = create_experiment_parser('S05: kNN Prediction with Hybrid Distance',
                                      has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 's05_knn_prediction', args.dataset)
