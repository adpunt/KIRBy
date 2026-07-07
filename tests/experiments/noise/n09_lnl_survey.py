#!/usr/bin/env python3
"""N09: LNL Method Survey with Hybrids — Learning with Noisy Labels methods.

Paper: Noise

Scope / claim (honest framing): this experiment asks a RELATIVE question —
does adding the hybrid representation change the RANKING of learning-with-
noisy-labels (LNL) methods — under SYNTHETIC noise injected on the TRAINING
labels only (Gaussian-additive for regression, label-flipping for
classification; see _inject_noise). Validation labels are left clean. It does
NOT establish an absolute noise-robustness result for any method or rep: the
signal of interest is purely whether the base→hybrid (and hybrid→weighted-
hybrid) swap re-orders which LNL methods come out on top.

Methods tested:
  - 12 total in LNL_REGISTRY
  - 10 support both regression and classification (MSE↔CE swap inside the
    method, see lnl_methods.py).
  - 2 are regression-only (heteroscedastic_gp, evidential_kernel); they are
    auto-filtered when task_type='classification' via REGRESSION_ONLY_LNL.

6 conditions × 5 noise levels × {12 or 10} methods:
  A: base_no_lnl            — Base rep + standard DNN (baseline)
  B: hybrid_no_lnl          — Hybrid + standard DNN
  C: base_lnl               — Base rep + each LNL method
  D: hybrid_lnl             — Hybrid + each LNL method
  E: weighted_hybrid_no_lnl — Importance-weighted hybrid + standard DNN
  F: weighted_hybrid_lnl    — Importance-weighted hybrid + each LNL method

Outputs per-sigma:
  - delta_from_clean: signed (noisy_score - clean_score); always defined.
  - score_ratio_pct: ratio*100 when both sides are positive, else NaN.
    (Renamed from `retention_pct`: it is a plain ratio, NOT bounded to 100 — it
    legitimately exceeds 100% when denoising beats the clean baseline. R² and
    MCC can be negative, so a naive ratio was historically misleading.)

Feature selection (AUDIT P1#5 — this is the FAIR version, unlike n10):
the hybrid feature mix is re-searched per (noise level, data split) on the
NOISY training labels using `find_best_hybrid` with a reduced budget grid
(NOISY_SEARCH_BUDGETS). The previous version pulled the mix from a precomputed
pool that ran on CLEAN labels, which gave the methods oracle access to ground
truth they wouldn't have in a real noisy-label setting; the per-fold re-search
removes that clean-label oracle so BOTH the blueprint selection (which reps,
which allocation/importance strategy) AND the feature construction see only
noisy labels. The clean-label pool blueprint `bp` survives only as a fallback
if the inner search fails. The inner search uses random forest as a fast proxy
downstream model — DNN-based search would be more matched but ~10x slower per
(noise level, fold). Note: the inner search runs at speed='full' (the 'fast'
speed tier was removed in the F12/F13 tier reduction); runtime is kept
tractable by the trimmed NOISY_SEARCH_BUDGETS grid and small inner n_folds,
not by a speed tier.

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

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories,
    get_best_search_for_downstream, build_fold_hybrid,
    extract_top3_base_reps,
)
from kirby.utils.lnl_methods import (
    get_lnl_method, LNL_REGISTRY, REGRESSION_ONLY_LNL,
    _standard_dnn_train_predict, _standard_dnn_train_predict_clf,
)
from kirby.find_best_hybrid import find_best_hybrid

SIGMAS = [0.0, 0.1, 0.3, 0.6, 1.0]
ALL_METHODS = list(LNL_REGISTRY.keys())

# Reduced budget grid for the per-(noise level, fold) feature search.
# The full pool covers [10, 25, 50, 100, 250, 500, 1000, 2000]; we trim to
# three representative budgets here because the search runs inside the
# noise injection loop and full-grid runtime would be prohibitive. Pattern
# mirrors c06's per-variant searches with reduced budgets (the 'fast' speed
# tier no longer exists — only 'full' — so the inner search runs at 'full'
# speed but with this trimmed budget grid to stay tractable).
NOISY_SEARCH_BUDGETS = [50, 100, 250]


def _inject_noise(y, sigma_frac, seed=42, is_clf=False):
    """Inject noise. Regression: Gaussian additive. Classification: label flipping."""
    rng = np.random.RandomState(seed)
    if is_clf:
        n = len(y)
        n_flip = max(1, int(n * sigma_frac))
        y_noisy = y.copy()
        flip_idx = rng.choice(n, n_flip, replace=False)
        classes = np.unique(y)
        for i in flip_idx:
            other = classes[classes != y[i]]
            if len(other) > 0:
                y_noisy[i] = rng.choice(other)
        return y_noisy
    else:
        y_range = np.max(y) - np.min(y)
        noise = rng.normal(0, sigma_frac * y_range, len(y))
        return y + noise


def _run_baseline(X_train, y_train, X_val, y_val, random_state=42,
                  is_clf=False, classes_=None):
    """Standard DNN baseline (no LNL). Routes to clf trainer when is_clf=True."""
    if is_clf:
        return _standard_dnn_train_predict_clf(
            X_train, y_train, X_val,
            epochs=200, hidden_sizes=(256, 128, 64),
            dropout=0.2, lr=0.001, batch_size=256,
            patience=20, random_state=random_state,
            classes_=classes_,
        )
    return _standard_dnn_train_predict(
        X_train, y_train, X_val,
        epochs=200, hidden_sizes=(256, 128, 64),
        dropout=0.2, lr=0.001, batch_size=256,
        patience=20, random_state=random_state,
    )


def run(data, config, output_dir, dataset_name, **kwargs):
    """N09: LNL method survey with hybrids."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N09: LNL METHOD SURVEY WITH HYBRIDS")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    sigmas = kwargs.get('sigmas', SIGMAS)
    methods_to_test = kwargs.get('methods', ALL_METHODS)

    # heteroscedastic_gp and evidential_kernel have no classification analog
    # (continuous-noise-variance / NIG distribution math doesn't transform via
    # a loss-function swap). All 10 other methods support clf via MSE→CE in
    # their inner training. See REGRESSION_ONLY_LNL in lnl_methods.py.
    if is_clf:
        methods_to_test = [m for m in methods_to_test
                           if m not in REGRESSION_ONLY_LNL]

    # Resolve the full class set up front so per-fold y_train subsets all
    # train classifiers with the same n_classes outputs.
    classes_ = np.unique(labels) if is_clf else None

    sample_X = next(iter(reps.values()))

    # Get hybrid blueprint ONCE via pool helper. Prefer dnn-best blueprint;
    # fall back to rf if dnn isn't in the pool.
    search, meta = get_best_search_for_downstream(
        reps, labels, groups,
        downstream_model='dnn', budget=None,
        augmentations=augmentations,
        speed=config.get('search_speed', 'full'),
        dataset_name=dataset_name, task_type=task_type,
        data=data,
    )
    if meta is None:
        search, meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model='rf', budget=None,
            augmentations=augmentations,
            speed=config.get('search_speed', 'full'),
            dataset_name=dataset_name, task_type=task_type,
            data=data,
        )
    if search is not None:
        top3 = extract_top3_base_reps(search)
        if top3:
            # extract_top3_base_reps ranks by RF baselines regardless of which
            # downstream model the blueprint pool selected — make that explicit
            # in the log so readers don't conflate the two.
            print(f"  Top-3 base reps (ranked by RF baseline): "
                  f"{', '.join(f'{n}={s:.4f}' for n, s in top3)}")
    bp = meta['best_config']['blueprint'] if meta else None
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
        # Record the actual budget the pool selected, not the kwargs default
        # (previously hard-coded to 100 regardless of what the pool returned).
        'budget': int(bp.budget) if bp is not None else None,
        'task_type': task_type,
        'is_clf': is_clf,
        'n_classes': int(len(classes_)) if is_clf else None,
        'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
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
                                        seed=42 + fold_idx, is_clf=is_clf)
            else:
                y_train = y_train_clean.copy()

            # Build base rep features
            X_base_train = reps[base_rep_name][train_idx]
            X_base_val = reps[base_rep_name][val_idx]
            scaler_base = StandardScaler()
            X_base_train_s = scaler_base.fit_transform(X_base_train)
            X_base_val_s = scaler_base.transform(X_base_val)

            # Build hybrid features by running a fresh feature-mix search on
            # the NOISY training labels (matching the c06 per-variant pattern).
            # The pool blueprint `bp` above was fit on clean labels, which
            # would give the LNL methods a feature mix chosen with oracle
            # access to ground truth. Refitting per (noise level, fold) on
            # noisy labels removes that leak. Uses fast speed + reduced budget
            # grid to keep runtime tractable; downstream model is 'rf' as a
            # fast proxy for DNN (RF-best and DNN-best feature mixes are
            # usually well-correlated on tabular molecular features).
            train_reps = {k: v[train_idx] for k, v in reps.items()}
            val_reps = {k: v[val_idx] for k, v in reps.items()}
            try:
                train_augs = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
                val_augs = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None

                # Run search WITHIN the train fold using noisy labels.
                # Subset groups (if present) so inner CV stays inside the train fold.
                train_groups_subset = (groups[train_idx]
                                       if groups is not None else None)
                inner_n_folds = max(2, min(3, n_folds))
                _rf_factory = get_model_factories(
                    ['rf'], task_type=task_type,
                    n_samples=len(train_idx))['rf']
                inner_search = find_best_hybrid(
                    base_reps={k: v[train_idx] for k, v in reps.items()},
                    labels=y_train,
                    groups=train_groups_subset,
                    models={'rf': _rf_factory()},
                    budgets=NOISY_SEARCH_BUDGETS,
                    task_type=task_type,
                    metric=metric_name,
                    # SPEED_TIERS only defines 'full' now (the 'fast' tier was
                    # removed in the F12/F13 tier reduction). Passing
                    # speed='fast' raised KeyError('fast') inside
                    # find_best_hybrid, which the try/except below swallowed —
                    # killing the ENTIRE hybrid arm (conditions B/D/E/F) on
                    # every fold and leaving only base_lnl populated. Runtime
                    # is kept tractable by the reduced NOISY_SEARCH_BUDGETS grid
                    # and the small inner n_folds, not by a speed tier.
                    speed='full',
                    seed=42 + fold_idx,
                    n_folds=inner_n_folds,
                    cache=False,
                    augmentations=({k: v[train_idx] for k, v in augmentations.items()}
                                   if augmentations else None),
                    augmentation_strategy=meta.get('aug_condition', 'uncapped') if meta else 'uncapped',
                    verbose=False,
                )
                # Pick the best blueprint across the reduced budget grid
                per_budget = inner_search.best_by_model_budget.get('rf', {})
                _best_bp = None
                _best_score = float('inf') if lower_is_better else float('-inf')
                for _budget, _entry in per_budget.items():
                    if _entry is None:
                        continue
                    _score = _entry.get('score_mean', float('nan'))
                    if np.isnan(_score):
                        continue
                    if (lower_is_better and _score < _best_score) or \
                       (not lower_is_better and _score > _best_score):
                        _best_score = _score
                        _best_bp = _entry['blueprint']
                inner_bp = _best_bp if _best_bp is not None else bp

                X_hybrid_train, X_hybrid_val, fi = build_fold_hybrid(
                    train_reps, val_reps, y_train, inner_bp,
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
                                         random_state=42 + fold_idx,
                                         is_clf=is_clf, classes_=classes_)
                score_a = float(metric_fn(y_val, y_pred_a))
                sigma_results['base_no_lnl'].append(score_a)
            except Exception as e:
                print(f"    Condition A failed: {e}")

            # --- Condition B: hybrid, no LNL ---
            if hybrid_ok:
                try:
                    y_pred_b = _run_baseline(X_hybrid_train_s, y_train,
                                             X_hybrid_val_s, y_val,
                                             random_state=42 + fold_idx,
                                             is_clf=is_clf, classes_=classes_)
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
                        # Importance-weight each hybrid column with MEAN-1
                        # normalisation. w is normalised to mean 1 (not sum 1),
                        # so columns keep ~unit scale on average and the input
                        # magnitude is preserved — this avoids the original
                        # scale-collapse (sum-1 weights shrank every column by
                        # ~1/n_cols ≈0.01, crushing DNN/LNL inputs to near zero).
                        # We deliberately do NOT re-fit a StandardScaler here:
                        # X_hybrid_*_s is already unit-variance, so multiplying
                        # column j by w_j gives std w_j and a fresh per-column
                        # StandardScaler would divide it straight back out —
                        # exactly cancelling the weighting and collapsing
                        # conditions E→B and F→D. Mean-1 scaling is the fix that
                        # keeps the weighting hypothesis testable.
                        w = np.sqrt(np.maximum(fold_weights, 0))
                        w = w / (w.mean() + 1e-10)
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
                                             random_state=42 + fold_idx,
                                             is_clf=is_clf, classes_=classes_)
                    score_e = float(metric_fn(y_val, y_pred_e))
                    sigma_results['weighted_hybrid_no_lnl'].append(score_e)
                except Exception as e:
                    print(f"    Condition E failed: {e}")

            # --- Conditions C, D & F: LNL methods (skip at sigma=0) ---
            if sigma > 0:
                lnl_kwargs = dict(random_state=42 + fold_idx,
                                  is_clf=is_clf, classes_=classes_)
                for method_name in methods_to_test:
                    # Condition C: base + LNL
                    try:
                        method_c = get_lnl_method(method_name, **lnl_kwargs)
                        result_c = method_c.train_and_predict(
                            X_base_train_s, y_train, X_base_val_s, y_val)
                        score_c = float(metric_fn(y_val, result_c['y_pred']))
                        sigma_results['base_lnl'][method_name].append(score_c)
                    except Exception as e:
                        print(f"    C/{method_name} failed: {e}")

                    # Condition D: hybrid + LNL
                    if hybrid_ok:
                        try:
                            method_d = get_lnl_method(method_name, **lnl_kwargs)
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
                            method_f = get_lnl_method(method_name, **lnl_kwargs)
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
    # For each (sigma, condition) record TWO quantities:
    #   delta_from_clean: signed (noisy_score - clean_score); always defined.
    #   retention_pct:    ratio expressed as a percentage; only defined when
    #                     both scores have the same sign AND the denominator
    #                     is bounded away from zero. The original ratio-only
    #                     formulation produced garbage for negative R²
    #                     (e.g. clean=-0.1, noisy=-0.5 → 500%). The signed
    #                     delta is the safer headline quantity.
    clean_sigma = results['per_sigma'].get('0.0', {})
    clean_base = clean_sigma.get('base_no_lnl', {}).get('mean', float('nan'))
    clean_hybrid = clean_sigma.get('hybrid_no_lnl', {}).get('mean', float('nan'))
    clean_whybrid = clean_sigma.get('weighted_hybrid_no_lnl', {}).get('mean', float('nan'))

    def _score_ratio_pct(noisy_val, clean_val):
        """Score ratio in percent (renamed from `retention_pct`).

        This is a PLAIN ratio, not bounded to 100 — when denoising beats the
        clean baseline it legitimately exceeds 100%. Returns NaN when
        ill-defined (opposite signs / near-zero denominator).
        """
        if np.isnan(noisy_val) or np.isnan(clean_val):
            return float('nan')
        if lower_is_better:
            # clean/noisy; both should be positive (e.g. RMSE)
            if clean_val <= 1e-10 or noisy_val <= 1e-10:
                return float('nan')
            return float(clean_val / noisy_val * 100)
        # higher-is-better; both should be positive (e.g. R^2, AUROC)
        if clean_val <= 1e-10 or noisy_val <= 1e-10:
            return float('nan')
        return float(noisy_val / clean_val * 100)

    def _delta(noisy_val, clean_val):
        if np.isnan(noisy_val) or np.isnan(clean_val):
            return float('nan')
        return float(noisy_val - clean_val)

    def _pair(noisy_val, clean_val):
        return {
            'delta_from_clean': _delta(noisy_val, clean_val),
            'score_ratio_pct': _score_ratio_pct(noisy_val, clean_val),
        }

    retention = {}
    for sigma_key, sigma_data in results['per_sigma'].items():
        if sigma_key == '0.0':
            continue
        sigma_ret = {}

        noisy_base = sigma_data.get('base_no_lnl', {}).get('mean', float('nan'))
        noisy_hybrid = sigma_data.get('hybrid_no_lnl', {}).get('mean', float('nan'))
        noisy_whybrid = sigma_data.get('weighted_hybrid_no_lnl', {}).get('mean', float('nan'))

        sigma_ret['base_no_lnl'] = _pair(noisy_base, clean_base)
        sigma_ret['hybrid_no_lnl'] = _pair(noisy_hybrid, clean_hybrid)
        sigma_ret['weighted_hybrid_no_lnl'] = _pair(noisy_whybrid, clean_whybrid)

        for cond in ['base_lnl', 'hybrid_lnl', 'weighted_hybrid_lnl']:
            sigma_ret[cond] = {}
            if 'weighted_hybrid' in cond:
                clean_ref = clean_whybrid
            elif 'hybrid' in cond:
                clean_ref = clean_hybrid
            else:
                clean_ref = clean_base
            for method_name in methods_to_test:
                m_score = sigma_data.get(cond, {}).get(method_name, {}).get(
                    'mean', float('nan'))
                sigma_ret[cond][method_name] = _pair(m_score, clean_ref)

        retention[sigma_key] = sigma_ret

    results['retention'] = convert_numpy(retention)
    results['retention_schema'] = (
        'Each entry has {delta_from_clean: noisy-clean, score_ratio_pct: '
        'ratio*100 when both sides are positive else NaN — a plain ratio, may '
        'exceed 100 when denoising beats the clean baseline}. Prefer delta '
        'for direction-robust comparison.')

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
