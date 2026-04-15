#!/usr/bin/env python3
"""F00: Rep screening — where should A/B filtering of representations go?

Paper: Framework
Question: When you pass a large rep set (some may be useless), where should
    the search filter them? Tests 4 options and compares hybrid quality,
    wall time, and whether screening causes "regret" (missing a rep that
    would have helped).

Options tested:
    A) Pre-search hard gate: drop reps below absolute baseline threshold
    B) Implicit (current): all reps enter search, allocation ignores bad ones
    C) Relative threshold: drop reps below X% of best rep's baseline score
    D) Augmentation pathway: core reps as base, uncertain reps as augmentations

Run this FIRST — it determines how reps enter all subsequent experiments.

Usage:
    python f00_rep_screening.py --dataset logd --reps all --mode full
    python f00_rep_screening.py --dataset flab_thermo_tresanco --reps all --mode full
    python f00_rep_screening.py --dataset tcga_luad_vs_lusc --reps all --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, cv_split,
    create_hybrid, apply_feature_selection,
    parse_rep_args, FRAMEWORK_TASKS,
    INTERNAL_EVALUATORS, DOWNSTREAM_MODELS, extract_top3_base_reps,
)
from kirby.find_best_hybrid import (
    _evaluate_single, _resolve_metric, _score_to_comparable,
    BUILTIN_METRICS, _detect_task_type,
    _select_top_features_per_fold, HIGH_DIM_FILTER_THRESHOLD,
)


# =============================================================================
# DOMAIN-AWARE METRIC SELECTION
# =============================================================================

def _get_primary_metric(data):
    """Return (metric_name, metric_fn, lower_is_better) for the domain."""
    domain = data.get('domain', 'molecular')
    task_type = data.get('task_type', 'regression')

    if domain == 'tcga' and task_type == 'classification':
        return 'accuracy', accuracy_score, False
    elif domain == 'dta':
        # CI is primary for DTA, but expensive — use spearman as proxy
        from scipy.stats import spearmanr
        return 'spearman', lambda y, p: float(spearmanr(y, p)[0]), False
    elif domain == 'antibody':
        from scipy.stats import spearmanr
        return 'spearman', lambda y, p: float(spearmanr(y, p)[0]), False
    else:
        # molecular, reaction
        return 'rmse', lambda y, p: float(np.sqrt(np.mean((y - p) ** 2))), True


# =============================================================================
# BASELINE EVALUATION
# =============================================================================

def evaluate_baselines(reps, labels, groups, model_factory, metric_fn,
                       n_folds, mode='full'):
    """Evaluate each rep individually. Returns {rep_name: mean_score}.

    Applies the same per-fold high-dim feature-importance filter as
    find_best_hybrid._evaluate_single when a rep exceeds
    HIGH_DIM_FILTER_THRESHOLD. Filter is per-fold (fit LightGBM on train
    only, pick top N, apply to train+val) so no val-label leakage. Target
    N is 500 if dataset has >= 1000 samples else 200.
    """
    sample_X = next(iter(reps.values()))
    baselines = {}

    for rep_name, X in reps.items():
        X_arr = np.asarray(X)

        # Decide filter params once per rep
        _apply_hd_filter = X_arr.shape[1] > HIGH_DIM_FILTER_THRESHOLD
        if _apply_hd_filter:
            _hd_target_n = 500 if len(labels) >= 1000 else 200
            if _hd_target_n >= X_arr.shape[1]:
                _apply_hd_filter = False
            else:
                _hd_task_type = _detect_task_type(labels)

        scores = []
        for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                           labels, mode=mode):
            X_train_raw = X_arr[train_idx]
            X_val_raw = X_arr[val_idx]
            y_train = labels[train_idx]

            if _apply_hd_filter:
                top_idx = _select_top_features_per_fold(
                    X_train_raw, y_train, _hd_target_n, _hd_task_type)
                X_train_raw = X_train_raw[:, top_idx]
                X_val_raw = X_val_raw[:, top_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_val = scaler.transform(X_val_raw)

            model = model_factory()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            scores.append(metric_fn(labels[val_idx], y_pred))

        baselines[rep_name] = float(np.mean(scores)) if scores else float('nan')

    return baselines


# =============================================================================
# HYBRID EVALUATION HELPER
# =============================================================================

def evaluate_hybrid(reps, labels, groups, model_factory, metric_fn,
                    n_folds, mode='full', budget=100,
                    allocation_method='greedy', importance_method='treeshap',
                    augmentations=None, augmentation_strategy='none',
                    augmentation_budget=20):
    """Run hybrid with given reps and return (mean_score, std, n_features, allocation_dict, wall_time)."""
    sample_X = next(iter(reps.values()))
    scores = []
    n_feats = []
    last_alloc = {}
    t0 = time.time()

    for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                       labels, mode=mode):
        train_reps = {k: v[train_idx] for k, v in reps.items()}
        val_reps = {k: v[val_idx] for k, v in reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Split augmentations per fold
        train_augs = None
        val_augs = None
        if augmentations is not None:
            train_augs = {k: v[train_idx] for k, v in augmentations.items()}
            val_augs = {k: v[val_idx] for k, v in augmentations.items()}

        try:
            X_train_h, feature_info = create_hybrid(
                train_reps, y_train,
                allocation_method=allocation_method,
                budget=budget,
                importance_method=importance_method,
                augmentations=train_augs,
                augmentation_strategy=augmentation_strategy,
                augmentation_budget=augmentation_budget,
            )
            X_val_h = apply_feature_selection(val_reps, feature_info)

            # Apply augmentation selection to val
            aug_info = feature_info.get('augmentation_info')
            if aug_info and val_augs is not None:
                from kirby.hybrid import apply_augmentation_selection
                aug_val = apply_augmentation_selection(val_augs, aug_info)
                if aug_val is not None:
                    X_val_h = np.hstack([X_val_h, aug_val])

        except Exception as e:
            print(f"    Hybrid failed: {e}")
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_h)
        X_val_s = scaler.transform(X_val_h)

        model = model_factory()
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_val_s)
        scores.append(metric_fn(y_val, y_pred))
        n_feats.append(X_train_h.shape[1])
        last_alloc = feature_info.get('allocation', {})

    wall_time = time.time() - t0

    if not scores:
        return float('nan'), float('nan'), 0, {}, wall_time

    return (
        float(np.mean(scores)),
        float(np.std(scores)),
        int(np.mean(n_feats)),
        dict(last_alloc),
        wall_time,
    )


# =============================================================================
# OPTION A: PRE-SEARCH HARD GATE
# =============================================================================

def option_a_hard_gate(reps, labels, groups, model_factory, metric_fn,
                       baselines, threshold, lower_is_better,
                       n_folds, mode, budget=100):
    """Screen reps with absolute threshold, then run hybrid on survivors."""
    surviving = {}
    screened = {}

    for rep_name, score in baselines.items():
        if np.isnan(score):
            screened[rep_name] = score
            continue
        comparable = _score_to_comparable(score, lower_is_better)
        threshold_comparable = _score_to_comparable(threshold, lower_is_better)
        if comparable >= threshold_comparable:
            surviving[rep_name] = reps[rep_name]
        else:
            screened[rep_name] = score

    if len(surviving) < 2:
        # Not enough reps survived — can't build hybrid
        return {
            'score_mean': float('nan'),
            'score_std': float('nan'),
            'n_features': 0,
            'allocation': {},
            'wall_time': 0,
            'n_surviving': len(surviving),
            'n_screened': len(screened),
            'screened_reps': screened,
            'surviving_reps': list(surviving.keys()),
        }

    score_mean, score_std, n_features, alloc, wall_time = evaluate_hybrid(
        surviving, labels, groups, model_factory, metric_fn, n_folds, mode, budget
    )

    return {
        'score_mean': score_mean,
        'score_std': score_std,
        'n_features': n_features,
        'allocation': alloc,
        'wall_time': wall_time,
        'n_surviving': len(surviving),
        'n_screened': len(screened),
        'screened_reps': screened,
        'surviving_reps': list(surviving.keys()),
    }


# =============================================================================
# OPTION B: IMPLICIT (CURRENT BEHAVIOR)
# =============================================================================

def option_b_implicit(reps, labels, groups, model_factory, metric_fn,
                      n_folds, mode, budget=100):
    """Pass all reps to hybrid search — let allocation handle filtering."""
    score_mean, score_std, n_features, alloc, wall_time = evaluate_hybrid(
        reps, labels, groups, model_factory, metric_fn, n_folds, mode, budget
    )

    # Identify which reps got 0 allocation (implicitly screened)
    zero_alloc = {k: 0 for k, v in alloc.items() if v == 0}
    active = {k: v for k, v in alloc.items() if v > 0}

    return {
        'score_mean': score_mean,
        'score_std': score_std,
        'n_features': n_features,
        'allocation': alloc,
        'wall_time': wall_time,
        'n_surviving': len(active),
        'n_screened': len(zero_alloc),
        'screened_reps': zero_alloc,
        'surviving_reps': list(active.keys()),
    }


# =============================================================================
# OPTION C: RELATIVE THRESHOLD
# =============================================================================

def option_c_relative(reps, labels, groups, model_factory, metric_fn,
                      baselines, fraction, lower_is_better,
                      n_folds, mode, budget=100):
    """Screen reps more than `fraction` away from the best rep's score, then run hybrid."""
    # Find best baseline score (in comparable space)
    valid_scores = {k: v for k, v in baselines.items() if not np.isnan(v)}
    if not valid_scores:
        return {
            'score_mean': float('nan'), 'score_std': float('nan'),
            'n_features': 0, 'allocation': {}, 'wall_time': 0,
            'n_surviving': 0, 'n_screened': len(reps),
            'screened_reps': baselines, 'surviving_reps': [],
        }

    comparable_scores = {k: _score_to_comparable(v, lower_is_better)
                         for k, v in valid_scores.items()}
    best_comparable = max(comparable_scores.values())
    # Gap-based threshold: keep reps within `fraction` of the best score.
    # e.g. fraction=0.1 means "drop reps more than 10% away from best".
    # Using abs() handles negative comparable scores (lower-is-better metrics).
    cutoff_comparable = best_comparable - fraction * abs(best_comparable)

    surviving = {}
    screened = {}

    for rep_name in reps:
        comp = comparable_scores.get(rep_name, -np.inf)
        if comp >= cutoff_comparable:
            surviving[rep_name] = reps[rep_name]
        else:
            screened[rep_name] = baselines.get(rep_name, float('nan'))

    if len(surviving) < 2:
        return {
            'score_mean': float('nan'), 'score_std': float('nan'),
            'n_features': 0, 'allocation': {}, 'wall_time': 0,
            'n_surviving': len(surviving), 'n_screened': len(screened),
            'screened_reps': screened, 'surviving_reps': list(surviving.keys()),
        }

    score_mean, score_std, n_features, alloc, wall_time = evaluate_hybrid(
        surviving, labels, groups, model_factory, metric_fn, n_folds, mode, budget
    )

    return {
        'score_mean': score_mean,
        'score_std': score_std,
        'n_features': n_features,
        'allocation': alloc,
        'wall_time': wall_time,
        'n_surviving': len(surviving),
        'n_screened': len(screened),
        'screened_reps': screened,
        'surviving_reps': list(surviving.keys()),
    }


# =============================================================================
# OPTION D: AUGMENTATION PATHWAY
# =============================================================================

def option_d_augmentation(reps, labels, groups, model_factory, metric_fn,
                          baselines, lower_is_better,
                          n_folds, mode, budget=100, n_core=4,
                          augmentation_budget=20):
    """Top n_core reps as base, rest as augmentations with greedy_ablation."""
    # Rank reps by baseline score
    comparable = {k: _score_to_comparable(v, lower_is_better)
                  for k, v in baselines.items() if not np.isnan(v)}
    ranked = sorted(comparable.keys(), key=lambda k: comparable[k], reverse=True)

    core_names = ranked[:n_core]
    candidate_names = ranked[n_core:]

    # Need at least 2 core reps for hybrid
    if len(core_names) < 2:
        return {
            'score_mean': float('nan'), 'score_std': float('nan'),
            'n_features': 0, 'allocation': {}, 'wall_time': 0,
            'n_surviving': 0, 'n_screened': len(reps),
            'screened_reps': {}, 'surviving_reps': [],
            'core_reps': core_names, 'candidate_reps': candidate_names,
            'kept_augmentations': [],
        }

    core_reps = {k: reps[k] for k in core_names}
    aug_reps = {k: reps[k] for k in candidate_names if k in reps}

    if not aug_reps:
        # No candidates — just run hybrid on core
        score_mean, score_std, n_features, alloc, wall_time = evaluate_hybrid(
            core_reps, labels, groups, model_factory, metric_fn,
            n_folds, mode, budget
        )
        return {
            'score_mean': score_mean, 'score_std': score_std,
            'n_features': n_features, 'allocation': alloc,
            'wall_time': wall_time,
            'n_surviving': len(core_names), 'n_screened': 0,
            'screened_reps': {}, 'surviving_reps': core_names,
            'core_reps': core_names, 'candidate_reps': [],
            'kept_augmentations': [],
        }

    score_mean, score_std, n_features, alloc, wall_time = evaluate_hybrid(
        core_reps, labels, groups, model_factory, metric_fn,
        n_folds, mode, budget,
        augmentations=aug_reps,
        augmentation_strategy='greedy_ablation',
        augmentation_budget=augmentation_budget,
    )

    return {
        'score_mean': score_mean,
        'score_std': score_std,
        'n_features': n_features,
        'allocation': alloc,
        'wall_time': wall_time,
        'n_surviving': len(core_names),
        'n_screened': len(candidate_names),
        'screened_reps': {k: baselines.get(k, float('nan')) for k in candidate_names},
        'surviving_reps': core_names,
        'core_reps': core_names,
        'candidate_reps': candidate_names,
        'kept_augmentations': [],  # TODO: extract from feature_info
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, only_option=None, **kwargs):
    """Run rep screening comparison across all 4 options (or a single option)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT F00: REP SCREENING — WHERE SHOULD FILTERING GO?")
    if only_option:
        print(f"  (Running ONLY option {only_option})")
    print("=" * 70)

    metric_name, metric_fn, lower_is_better = _get_primary_metric(data)
    task_type = data.get('task_type', 'regression')
    model_factory = get_model_factories(['rf'], task_type=task_type,
                                        n_samples=len(data['labels']))['rf']
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    budget = 100

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    domain = data.get('domain', 'molecular')

    n_reps = len(reps)
    dims = {k: v.shape[1] for k, v in reps.items()}
    print(f"  Dataset: {dataset_name} ({domain})")
    print(f"  Metric: {metric_name} ({'lower' if lower_is_better else 'higher'} is better)")
    print(f"  Reps: {n_reps} ({', '.join(f'{k}={v}D' for k, v in dims.items())})")
    print(f"  Folds: {n_folds}, Budget: {budget}")

    # --- Step 1: Evaluate baselines ---
    print(f"\n  Evaluating baselines ({n_reps} reps)...")
    baselines = evaluate_baselines(reps, labels, groups, model_factory,
                                   metric_fn, n_folds, mode)

    print(f"\n  Baseline scores ({metric_name}):")
    for rep_name in sorted(baselines, key=lambda k: baselines[k],
                           reverse=not lower_is_better):
        print(f"    {rep_name:<30} {baselines[rep_name]:.4f}")

    results = {
        'baselines': baselines,
        'options': {},
    }

    # --- Option B: Implicit (current behavior, run first as reference) ---
    if not only_option or only_option == 'B':
        print(f"\n  Option B: Implicit (all {n_reps} reps, no screening)...")
        result_b = option_b_implicit(reps, labels, groups, model_factory,
                                     metric_fn, n_folds, mode, budget)
        results['options']['B_implicit'] = result_b
        print(f"    Score: {result_b['score_mean']:.4f} +/- {result_b['score_std']:.3f}, "
              f"dim={result_b['n_features']}, "
              f"active={result_b['n_surviving']}/{n_reps}, "
              f"time={result_b['wall_time']:.1f}s")

    # --- Option A: Pre-search hard gate ---
    if not only_option or only_option == 'A':
        # Determine thresholds based on domain
        if lower_is_better:
            # For RMSE: lower is better, so threshold = max acceptable RMSE
            # Use percentiles of baseline scores
            valid_scores = [v for v in baselines.values() if not np.isnan(v)]
            if valid_scores:
                thresholds_a = [
                    ('p25', float(np.percentile(valid_scores, 25))),
                    ('p50', float(np.percentile(valid_scores, 50))),
                    ('p75', float(np.percentile(valid_scores, 75))),
                ]
            else:
                thresholds_a = []
        else:
            # For R², accuracy, spearman: higher is better
            thresholds_a = [
                ('>=0', 0.0),
                ('>=0.1', 0.1),
                ('>=0.3', 0.3),
                ('>=0.5', 0.5),
            ]

        for label, thresh in thresholds_a:
            print(f"\n  Option A: Hard gate (threshold {label} = {thresh:.3f})...")
            result_a = option_a_hard_gate(
                reps, labels, groups, model_factory, metric_fn,
                baselines, thresh, lower_is_better, n_folds, mode, budget
            )
            key = f'A_hard_gate_{label}'
            results['options'][key] = result_a
            print(f"    Score: {result_a['score_mean']:.4f} +/- {result_a['score_std']:.3f}, "
                  f"surviving={result_a['n_surviving']}/{n_reps}, "
                  f"time={result_a['wall_time']:.1f}s")
            if result_a['screened_reps']:
                print(f"    Screened: {list(result_a['screened_reps'].keys())}")

    # --- Option C: Relative threshold ---
    if not only_option or only_option == 'C':
        relative_fractions = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
        for frac in relative_fractions:
            print(f"\n  Option C: Relative threshold ({frac:.0%} of best)...")
            result_c = option_c_relative(
                reps, labels, groups, model_factory, metric_fn,
                baselines, frac, lower_is_better, n_folds, mode, budget
            )
            key = f'C_relative_{frac}'
            results['options'][key] = result_c
            print(f"    Score: {result_c['score_mean']:.4f} +/- {result_c['score_std']:.3f}, "
                  f"surviving={result_c['n_surviving']}/{n_reps}, "
                  f"time={result_c['wall_time']:.1f}s")
            if result_c['screened_reps']:
                print(f"    Screened: {list(result_c['screened_reps'].keys())}")

    # --- Option D: Augmentation pathway ---
    if not only_option or only_option == 'D':
        for n_core in [3, 4, 5]:
            if n_core >= n_reps:
                continue
            print(f"\n  Option D: Augmentation pathway (top {n_core} core, rest as augmentations)...")
            result_d = option_d_augmentation(
                reps, labels, groups, model_factory, metric_fn,
                baselines, lower_is_better, n_folds, mode, budget,
                n_core=n_core,
            )
            key = f'D_augmentation_core{n_core}'
            results['options'][key] = result_d
            print(f"    Score: {result_d['score_mean']:.4f} +/- {result_d['score_std']:.3f}, "
                  f"core={result_d['core_reps']}, "
                  f"time={result_d['wall_time']:.1f}s")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    direction = -1 if lower_is_better else 1

    # Use option B as reference if available, otherwise skip comparative summary
    b_result = results['options'].get('B_implicit')
    if b_result:
        ref_score = b_result['score_mean']
        ref_time = b_result['wall_time']

        print(f"\n  Reference (Option B implicit): {metric_name}={ref_score:.4f}, "
              f"time={ref_time:.1f}s")
        print(f"\n  {'Option':<35} {'Score':>8} {'Delta':>8} {'Reps':>6} {'Time':>7} {'Speedup':>8}")
        print("  " + "-" * 75)

        for opt_name in sorted(results['options'].keys()):
            opt = results['options'][opt_name]
            score = opt['score_mean']
            delta = (score - ref_score) * direction  # positive = better
            n_surv = opt['n_surviving']
            wt = opt['wall_time']
            speedup = ref_time / wt if wt > 0 else float('inf')

            print(f"  {opt_name:<35} {score:>8.4f} {delta:>+8.4f} "
                  f"{n_surv:>4}/{n_reps} {wt:>6.1f}s {speedup:>7.1f}x")

        # --- Regret analysis ---
        print(f"\n  REGRET ANALYSIS (did screening miss a useful rep?)")
        for opt_name, opt in results['options'].items():
            if opt_name == 'B_implicit':
                continue
            score = opt['score_mean']
            regret = (ref_score - score) * direction
            if regret > 0.01:
                print(f"    {opt_name}: REGRET = {regret:.4f} "
                      f"(screened: {list(opt.get('screened_reps', {}).keys())})")
            elif regret > 0:
                print(f"    {opt_name}: minor regret = {regret:.4f}")
            else:
                print(f"    {opt_name}: no regret (matched or beat implicit)")
    else:
        # Single-option mode: just print scores
        print(f"\n  {'Option':<35} {'Score':>8} {'Reps':>6} {'Time':>7}")
        print("  " + "-" * 60)
        for opt_name in sorted(results['options'].keys()):
            opt = results['options'][opt_name]
            score = opt['score_mean']
            n_surv = opt['n_surviving']
            wt = opt['wall_time']
            print(f"  {opt_name:<35} {score:>8.4f} {n_surv:>4}/{n_reps} {wt:>6.1f}s")

    results['meta'] = {
        'dataset': dataset_name,
        'domain': domain,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'n_reps': n_reps,
        'n_samples': len(labels),
        'n_folds': n_folds,
        'budget': budget,
        'rep_dims': dims,
    }

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('F00: Rep Screening Comparison',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--only-option', type=str, default=None,
                        choices=['A', 'B', 'C', 'D'],
                        help='Run only this screening option (default: all)')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    if args.only_option:
        print(f"Only option: {args.only_option}")

    rep_preset, rep_names = parse_rep_args(args)
    data = load_data(args.dataset, rep_preset, rep_names=rep_names,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  only_option=getattr(args, 'only_option', None))
    # Use separate output subdir when running a single option
    exp_id = 'f00_rep_screening'
    if args.only_option:
        exp_id = f'f00_rep_screening_option{args.only_option}'
    save_experiment_results(results, args.output, exp_id, args.dataset)
