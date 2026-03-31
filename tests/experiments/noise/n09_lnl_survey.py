#!/usr/bin/env python3
"""N09: LNL Method Survey with Hybrids — 12 Learning with Noisy Labels methods.

Paper: Noise
Tests whether hybrid representations change which LNL methods work best.
Replicates confirmation report LNL survey, then repeats with hybrids.

6 conditions × 5 noise levels × 12 methods:
  A: base_no_lnl          — ECFP4 + standard DNN (baseline)
  B: hybrid_no_lnl         — Hybrid + standard DNN (baseline)
  C: base_lnl              — ECFP4 + each LNL method
  D: hybrid_lnl            — Hybrid + each LNL method
  E: weighted_hybrid_no_lnl — Importance-weighted hybrid + standard DNN
  F: weighted_hybrid_lnl    — Importance-weighted hybrid + each LNL method

Usage:
    python n09_lnl_survey.py --dataset logd --mode intermediary
    python n09_lnl_survey.py --dataset qm9 --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric,
    get_precomputed_search, build_fold_hybrid,
)
from kirby.utils.lnl_methods import (
    get_lnl_method, LNL_REGISTRY, _standard_dnn_train_predict,
)

SIGMAS = [0.0, 0.1, 0.3, 0.6, 1.0]
ALL_METHODS = list(LNL_REGISTRY.keys())


def _inject_noise(y, sigma_frac, seed=42):
    """Inject Gaussian noise at sigma_frac × label range into training labels."""
    rng = np.random.RandomState(seed)
    y_range = np.max(y) - np.min(y)
    noise = rng.normal(0, sigma_frac * y_range, len(y))
    return y + noise


def _run_baseline(X_train, y_train, X_val, y_val, random_state=42):
    """Standard DNN baseline (no LNL)."""
    y_pred = _standard_dnn_train_predict(
        X_train, y_train, X_val,
        epochs=200, hidden_sizes=(256, 128, 64),
        dropout=0.2, lr=0.001, batch_size=256,
        patience=20, random_state=random_state,
    )
    return y_pred


def run(data, config, output_dir, dataset_name, **kwargs):
    """N09: LNL method survey with hybrids."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N09: LNL METHOD SURVEY WITH HYBRIDS")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')
    augmentation_strategy = 'uncapped'
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    sigmas = kwargs.get('sigmas', SIGMAS)
    budget = kwargs.get('budget', 100)
    methods_to_test = kwargs.get('methods', ALL_METHODS)

    sample_X = next(iter(reps.values()))

    # Get hybrid blueprint ONCE (shared across all sigmas/folds)
    search = get_precomputed_search(
        reps, labels, groups, budget=budget, aug_condition='uncapped',
        augmentations=augmentations, speed=config.get('search_speed', 'fast'),
        precomputed=data.get('precomputed_searches'))
    bp = search.best_overall['blueprint'] if search.best_overall else None
    if bp is None:
        print("  WARNING: No blueprint found, skipping")
        return {'error': 'no_blueprint', 'metric': metric_name}

    # Per-fold weights computed from feat_info (not global) — see conditions E/F below

    # Check if ECFP4 is available as base rep
    base_rep_name = None
    for candidate in ['ecfp4', 'pdv', 'blosum']:
        if candidate in reps:
            base_rep_name = candidate
            break
    if base_rep_name is None:
        base_rep_name = sorted(reps.keys())[0]

    print(f"  Dataset: {dataset_name} ({len(labels)} samples)")
    print(f"  Metric: {metric_name}")
    print(f"  Base rep: {base_rep_name} (dim={reps[base_rep_name].shape[1]})")
    print(f"  Sigmas: {sigmas}")
    print(f"  Methods: {len(methods_to_test)}")
    print(f"  Folds: {n_folds}")

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'base_rep': base_rep_name,
        'sigmas': sigmas,
        'methods': methods_to_test,
        'budget': budget,
        'per_sigma': {},
    }

    for sigma in sigmas:
        print(f"\n  === Sigma={sigma} ===")
        t_sigma = time.time()

        sigma_results = {
            'base_no_lnl': [],
            'hybrid_no_lnl': [],
            'weighted_hybrid_no_lnl': [],
        }
        if sigma > 0:
            sigma_results['base_lnl'] = {m: [] for m in methods_to_test}
            sigma_results['hybrid_lnl'] = {m: [] for m in methods_to_test}
            sigma_results['weighted_hybrid_lnl'] = {m: [] for m in methods_to_test}

        fold_idx = 0
        for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                            labels, mode=mode):
            fold_idx += 1
            y_train_clean = labels[train_idx]
            y_val = labels[val_idx]

            # Inject noise (training only)
            if sigma > 0:
                y_train = _inject_noise(y_train_clean, sigma,
                                        seed=42 + fold_idx)
            else:
                y_train = y_train_clean.copy()

            # Build base rep features
            X_base_train = reps[base_rep_name][train_idx]
            X_base_val = reps[base_rep_name][val_idx]
            scaler_base = StandardScaler()
            X_base_train_s = scaler_base.fit_transform(X_base_train)
            X_base_val_s = scaler_base.transform(X_base_val)

            # Build hybrid features from blueprint
            train_reps = {k: v[train_idx] for k, v in reps.items()}
            val_reps = {k: v[val_idx] for k, v in reps.items()}
            try:
                train_augs = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
                val_augs = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None
                X_hybrid_train, X_hybrid_val, fi = build_fold_hybrid(
                    train_reps, val_reps, y_train, bp,
                    augmentations_train=train_augs, augmentations_val=val_augs)
                scaler_hyb = StandardScaler()
                X_hybrid_train_s = scaler_hyb.fit_transform(X_hybrid_train)
                X_hybrid_val_s = scaler_hyb.transform(X_hybrid_val)
                hybrid_ok = True
            except Exception as e:
                print(f"    Hybrid failed (fold {fold_idx}): {e}")
                hybrid_ok = False

            # --- Condition A: base, no LNL ---
            try:
                y_pred_a = _run_baseline(X_base_train_s, y_train,
                                         X_base_val_s, y_val,
                                         random_state=42 + fold_idx)
                score_a = float(metric_fn(y_val, y_pred_a))
                sigma_results['base_no_lnl'].append(score_a)
            except Exception as e:
                print(f"    Condition A failed: {e}")

            # --- Condition B: hybrid, no LNL ---
            if hybrid_ok:
                try:
                    y_pred_b = _run_baseline(X_hybrid_train_s, y_train,
                                             X_hybrid_val_s, y_val,
                                             random_state=42 + fold_idx)
                    score_b = float(metric_fn(y_val, y_pred_b))
                    sigma_results['hybrid_no_lnl'].append(score_b)
                except Exception as e:
                    print(f"    Condition B failed: {e}")

            # --- Build weighted hybrid features for conditions E/F ---
            # Use per-fold feat_info weights (not global) to avoid dimension mismatch
            weighted_ok = False
            if hybrid_ok:
                try:
                    from experiments._shared import flatten_fold_weights
                    fold_weights = flatten_fold_weights(fi)
                    n_cols = X_hybrid_train_s.shape[1]
                    if len(fold_weights) == n_cols:
                        w = np.sqrt(np.maximum(fold_weights, 0))
                        w = w / (w.sum() + 1e-10)
                        X_whybrid_train_s = X_hybrid_train_s * w
                        X_whybrid_val_s = X_hybrid_val_s * w
                        weighted_ok = True
                    else:
                        print(f"    Weight dim mismatch: {len(fold_weights)} vs {n_cols}")
                except Exception as e:
                    print(f"    Weighted hybrid build failed: {e}")

            # --- Condition E: weighted hybrid, no LNL ---
            if weighted_ok:
                try:
                    y_pred_e = _run_baseline(X_whybrid_train_s, y_train,
                                             X_whybrid_val_s, y_val,
                                             random_state=42 + fold_idx)
                    score_e = float(metric_fn(y_val, y_pred_e))
                    sigma_results['weighted_hybrid_no_lnl'].append(score_e)
                except Exception as e:
                    print(f"    Condition E failed: {e}")

            # --- Conditions C, D & F: LNL methods (skip at sigma=0) ---
            if sigma > 0:
                for method_name in methods_to_test:
                    # Condition C: base + LNL
                    try:
                        method_c = get_lnl_method(method_name,
                                                  random_state=42 + fold_idx)
                        result_c = method_c.train_and_predict(
                            X_base_train_s, y_train, X_base_val_s, y_val)
                        score_c = float(metric_fn(y_val, result_c['y_pred']))
                        sigma_results['base_lnl'][method_name].append(score_c)
                    except Exception as e:
                        print(f"    C/{method_name} failed: {e}")

                    # Condition D: hybrid + LNL
                    if hybrid_ok:
                        try:
                            method_d = get_lnl_method(method_name,
                                                      random_state=42 + fold_idx)
                            result_d = method_d.train_and_predict(
                                X_hybrid_train_s, y_train,
                                X_hybrid_val_s, y_val)
                            score_d = float(metric_fn(y_val, result_d['y_pred']))
                            sigma_results['hybrid_lnl'][method_name].append(score_d)
                        except Exception as e:
                            print(f"    D/{method_name} failed: {e}")

                    # Condition F: weighted hybrid + LNL
                    if weighted_ok:
                        try:
                            method_f = get_lnl_method(method_name,
                                                      random_state=42 + fold_idx)
                            result_f = method_f.train_and_predict(
                                X_whybrid_train_s, y_train,
                                X_whybrid_val_s, y_val)
                            score_f = float(metric_fn(y_val, result_f['y_pred']))
                            sigma_results['weighted_hybrid_lnl'][method_name].append(score_f)
                        except Exception as e:
                            print(f"    F/{method_name} failed: {e}")

        # Aggregate fold results
        agg = {}
        for cond_key in ['base_no_lnl', 'hybrid_no_lnl', 'weighted_hybrid_no_lnl']:
            scores = sigma_results[cond_key]
            if scores:
                agg[cond_key] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'n_folds': len(scores),
                }
            else:
                agg[cond_key] = {'mean': float('nan'), 'n_folds': 0}

        if sigma > 0:
            for cond_key in ['base_lnl', 'hybrid_lnl', 'weighted_hybrid_lnl']:
                agg[cond_key] = {}
                for method_name in methods_to_test:
                    scores = sigma_results[cond_key][method_name]
                    if scores:
                        agg[cond_key][method_name] = {
                            'mean': float(np.mean(scores)),
                            'std': float(np.std(scores)),
                            'n_folds': len(scores),
                        }
                    else:
                        agg[cond_key][method_name] = {
                            'mean': float('nan'), 'n_folds': 0,
                        }

        results['per_sigma'][str(sigma)] = convert_numpy(agg)

        # Print summary for this sigma
        a_score = agg['base_no_lnl']['mean']
        b_score = agg['hybrid_no_lnl']['mean']
        e_score = agg['weighted_hybrid_no_lnl']['mean']
        print(f"    Base DNN: {a_score:.4f}, Hybrid DNN: {b_score:.4f}, "
              f"Weighted Hybrid DNN: {e_score:.4f}")
        if sigma > 0:
            # Best LNL method per condition
            for cond in ['base_lnl', 'hybrid_lnl', 'weighted_hybrid_lnl']:
                best_name = None
                best_score = float('-inf') if not lower_is_better else float('inf')
                for m in methods_to_test:
                    s = agg[cond].get(m, {}).get('mean', float('nan'))
                    if np.isnan(s):
                        continue
                    if (not lower_is_better and s > best_score) or \
                       (lower_is_better and s < best_score):
                        best_score = s
                        best_name = m
                if best_name:
                    print(f"    Best {cond}: {best_name} ({best_score:.4f})")

        print(f"    [{time.time() - t_sigma:.0f}s]")

    # =========================================================================
    # RETENTION COMPUTATION
    # =========================================================================
    clean_sigma = results['per_sigma'].get('0.0', {})
    clean_base = clean_sigma.get('base_no_lnl', {}).get('mean', float('nan'))
    clean_hybrid = clean_sigma.get('hybrid_no_lnl', {}).get('mean', float('nan'))
    clean_whybrid = clean_sigma.get('weighted_hybrid_no_lnl', {}).get('mean', float('nan'))

    retention = {}
    for sigma_key, sigma_data in results['per_sigma'].items():
        if sigma_key == '0.0':
            continue
        sigma_ret = {}

        # Baselines
        noisy_base = sigma_data.get('base_no_lnl', {}).get('mean', float('nan'))
        noisy_hybrid = sigma_data.get('hybrid_no_lnl', {}).get('mean', float('nan'))
        noisy_whybrid = sigma_data.get('weighted_hybrid_no_lnl', {}).get('mean', float('nan'))

        if not np.isnan(clean_base) and abs(clean_base) > 1e-10:
            sigma_ret['base_no_lnl'] = float(noisy_base / clean_base * 100)
        if not np.isnan(clean_hybrid) and abs(clean_hybrid) > 1e-10:
            sigma_ret['hybrid_no_lnl'] = float(noisy_hybrid / clean_hybrid * 100)
        if not np.isnan(clean_whybrid) and abs(clean_whybrid) > 1e-10:
            sigma_ret['weighted_hybrid_no_lnl'] = float(noisy_whybrid / clean_whybrid * 100)

        # LNL methods
        for cond in ['base_lnl', 'hybrid_lnl', 'weighted_hybrid_lnl']:
            sigma_ret[cond] = {}
            if 'weighted_hybrid' in cond:
                clean_ref = clean_whybrid
            elif 'hybrid' in cond:
                clean_ref = clean_hybrid
            else:
                clean_ref = clean_base
            if np.isnan(clean_ref) or abs(clean_ref) < 1e-10:
                continue
            for method_name in methods_to_test:
                m_score = sigma_data.get(cond, {}).get(method_name, {}).get(
                    'mean', float('nan'))
                if not np.isnan(m_score):
                    sigma_ret[cond][method_name] = float(m_score / clean_ref * 100)

        retention[sigma_key] = sigma_ret

    results['retention'] = convert_numpy(retention)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    summary = {
        'clean_base': clean_base,
        'clean_hybrid': clean_hybrid,
        'clean_weighted_hybrid': clean_whybrid,
    }

    # Best method per condition at highest noise
    highest_sigma = str(max(s for s in sigmas if s > 0))
    hs_data = results['per_sigma'].get(highest_sigma, {})

    for cond in ['base_lnl', 'hybrid_lnl', 'weighted_hybrid_lnl']:
        best_name = None
        best_score = float('-inf') if not lower_is_better else float('inf')
        for m in methods_to_test:
            s = hs_data.get(cond, {}).get(m, {}).get('mean', float('nan'))
            if np.isnan(s):
                continue
            if (not lower_is_better and s > best_score) or \
               (lower_is_better and s < best_score):
                best_score = s
                best_name = m
        summary[f'best_{cond}'] = best_name
        summary[f'best_{cond}_score'] = float(best_score) if best_name else None

    # Methods where hybrid > base
    hybrid_improves = []
    hybrid_harms = []
    for m in methods_to_test:
        base_s = hs_data.get('base_lnl', {}).get(m, {}).get('mean', float('nan'))
        hyb_s = hs_data.get('hybrid_lnl', {}).get(m, {}).get('mean', float('nan'))
        if np.isnan(base_s) or np.isnan(hyb_s):
            continue
        if (not lower_is_better and hyb_s > base_s) or \
           (lower_is_better and hyb_s < base_s):
            hybrid_improves.append(m)
        elif (not lower_is_better and hyb_s < base_s) or \
             (lower_is_better and hyb_s > base_s):
            hybrid_harms.append(m)

    summary['hybrid_improves'] = hybrid_improves
    summary['hybrid_harms'] = hybrid_harms
    summary['n_hybrid_improves'] = len(hybrid_improves)
    summary['n_hybrid_harms'] = len(hybrid_harms)

    # Methods where weighted hybrid > hybrid (does weighting help?)
    whybrid_improves = []
    whybrid_harms = []
    for m in methods_to_test:
        hyb_s = hs_data.get('hybrid_lnl', {}).get(m, {}).get('mean', float('nan'))
        whyb_s = hs_data.get('weighted_hybrid_lnl', {}).get(m, {}).get('mean', float('nan'))
        if np.isnan(hyb_s) or np.isnan(whyb_s):
            continue
        if (not lower_is_better and whyb_s > hyb_s) or \
           (lower_is_better and whyb_s < hyb_s):
            whybrid_improves.append(m)
        elif (not lower_is_better and whyb_s < hyb_s) or \
             (lower_is_better and whyb_s > hyb_s):
            whybrid_harms.append(m)

    summary['weighted_improves_over_hybrid'] = whybrid_improves
    summary['weighted_harms_vs_hybrid'] = whybrid_harms
    summary['n_weighted_improves'] = len(whybrid_improves)
    summary['n_weighted_harms'] = len(whybrid_harms)

    # Did the top-3 ranking change?
    def _top3(cond_data):
        scores = {}
        for m in methods_to_test:
            s = cond_data.get(m, {}).get('mean', float('nan'))
            if not np.isnan(s):
                scores[m] = s
        sorted_m = sorted(scores, key=scores.get,
                          reverse=not lower_is_better)
        return sorted_m[:3]

    top3_base = _top3(hs_data.get('base_lnl', {}))
    top3_hybrid = _top3(hs_data.get('hybrid_lnl', {}))
    top3_whybrid = _top3(hs_data.get('weighted_hybrid_lnl', {}))
    summary['top3_base'] = top3_base
    summary['top3_hybrid'] = top3_hybrid
    summary['top3_weighted_hybrid'] = top3_whybrid
    summary['ranking_changed'] = top3_base != top3_hybrid
    summary['ranking_changed_weighted'] = top3_hybrid != top3_whybrid

    results['summary'] = convert_numpy(summary)

    print(f"\n  === SUMMARY ===")
    print(f"  Clean: base={clean_base:.4f}, hybrid={clean_hybrid:.4f}, "
          f"weighted_hybrid={clean_whybrid:.4f}")
    print(f"  Best at sigma={highest_sigma}: "
          f"base_lnl={summary.get('best_base_lnl')}, "
          f"hybrid_lnl={summary.get('best_hybrid_lnl')}, "
          f"weighted_hybrid_lnl={summary.get('best_weighted_hybrid_lnl')}")
    print(f"  Hybrid improves {len(hybrid_improves)}/{len(methods_to_test)} methods")
    print(f"  Weighted improves over hybrid: "
          f"{len(whybrid_improves)}/{len(methods_to_test)} methods")
    print(f"  Ranking changed: {summary['ranking_changed']}")
    print(f"  Ranking changed (weighted): {summary['ranking_changed_weighted']}")
    print(f"  Top-3 base: {top3_base}")
    print(f"  Top-3 hybrid: {top3_hybrid}")
    print(f"  Top-3 weighted hybrid: {top3_whybrid}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('N09: LNL Method Survey')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n09_lnl_survey', args.dataset)
