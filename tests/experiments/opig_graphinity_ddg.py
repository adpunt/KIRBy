#!/usr/bin/env python3
"""
Graphinity DDG Prediction Experiment (OPIG B2)

Replicates Hummer et al. 2025 "Investigating the Volume and Diversity of
Data Needed for Generalizable Antibody-Antigen DDG Prediction" —
tests whether sequence-based reps + simple ML can approach the
structure-based Graphinity EGNN for DDG prediction.

Scientific questions:
  1. How does transfer from synthetic → experimental compare?
  2. Does hybrid augmentation help for DDG prediction?
  3. How far are simple rep→model approaches from the structure-based EGNN?

Conditions:
  - transfer_foldx:  Train on FoldX synthetic (--max-foldx rows, default 5000),
                     evaluate on ALL experimental SKEMPI.
  - experimental:    Train/eval on experimental SKEMPI only (baseline).

Removed 2026-09-04: within_foldx and within_rosetta, which trained and scored
inside graphinity_foldx and graphinity_rosetta. Neither dataset is registered,
so both conditions were skipped on every run since they were written; both
datasets were cut on 2026-07-09 because their labels are FoldX and Rosetta
output rather than measurements. The question they answered — can sequence
representations predict simulated binding free energy change — is dropped with
them. FoldX rows are still loaded for transfer_foldx, which trains on them and
scores on measured SKEMPI.

Paper baselines (Hummer et al. 2025, Nature Computational Science):
  Graphinity EGNN (structure-based):  Pearson r ~0.87 on experimental test
  Key finding: orders of magnitude more experimental data needed for
  generalizable DDG prediction.

Usage:
    # Smoke test (~3 min)
    python tests/experiments/opig_graphinity_ddg.py \\
        --conditions experimental --n-folds 2 --reps blosum,onehot,aac

    # Full evaluation
    python tests/experiments/opig_graphinity_ddg.py

Reference:
    Hummer et al. (2025). Nature Computational Science.
    Zenodo: https://doi.org/10.5281/zenodo.15384945
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from antibody_baseline_survey import load_dataset, DATASET_REGISTRY
from experiments._shared import (
    convert_numpy, get_model_factories, DOWNSTREAM_MODELS, INTERNAL_EVALUATORS,
    extract_top3_base_reps, FULL_BUDGETS,
    honest_rescore, build_honest_rescore_input,
    hr_exclude_folds,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

ALL_CONDITIONS = ['experimental', 'transfer_foldx']

DEFAULT_REPS = [
    'blosum', 'onehot', 'aac',           # classical
    'esm2', 'esm2_concat', 'antiberty',  # LMs
    'ablang2', 'prot_t5',                 # additional LMs
]

MODELS_LIST = DOWNSTREAM_MODELS


def _smoke_search_kwargs():
    """Smoke-only hybrid-search shrink, gated on KIRBY_SMOKE_N.

    Caps the find_best_hybrid pool to one small budget + fast tier for local
    smoke runs (default is minutes of compute).
    No-op when KIRBY_SMOKE_N is unset.
    """
    if not os.environ.get('KIRBY_SMOKE_N', '').strip():
        return {}
    # speed is now accepted but ignored (the former tier dict was removed; defaults
    # live in the DEFAULT_* module constants). Smoke shrink is via tiny budgets
    # only. internal_evaluators=['rf'] cuts the search ~5x (DEFAULT_INTERNAL_EVALUATORS
    # is match/rf/svm/lightgbm/mlp); rf alone still exercises the hybrid pipeline.
    # Also drop importance to lightgbm_gain (cheap): DEFAULT_IMPORTANCE_METHODS runs
    # deeplift+treeshap, which compute SHAP over the (here 5600-dim) features
    # per fold and dominate the smoke wall time. lightgbm_gain still produces a
    # real per-rep importance ranking so the hybrid pipeline is exercised.
    return {'speed': 'full', 'budgets': [25], 'internal_evaluators': ['rf'],
            'importance_methods': ['lightgbm_gain']}


# Condition → dataset name mapping
CONDITION_DATASETS = {
    'experimental':   'ddg_combined',
    'transfer_foldx': None,  # handled specially: train on synthetic, eval on experimental
}

# Paper baselines (Hummer et al. 2025, Nature Computational Science)
# IMPORTANT: The paper's core finding is that random splits inflate performance.
# Stringent CDR identity cutoffs reveal much lower generalization.
# Numbers below are from the repo's results notebooks (Figure-2b, Figure-2c).
PAPER_BASELINES = {
    'Graphinity EGNN synthetic FoldX (random)':     {'pearson': 0.896},
    'Graphinity EGNN synthetic FoldX (cutoff_100)':  {'pearson': 0.894},
    'Graphinity EGNN synthetic FoldX (cutoff_90)':   {'pearson': 0.893},
    'Graphinity EGNN synthetic FoldX (cutoff_70)':   {'pearson': 0.894},
    # Experimental: -RevMuts -NonBinders (closest to ddg_combined)
    'Graphinity EGNN experimental (random)':         {'pearson': 0.525},
    'Graphinity EGNN experimental (cutoff_100)':     {'pearson': 0.271},
    'Graphinity EGNN experimental (cutoff_90)':      {'pearson': 0.291},
    'Graphinity EGNN experimental (cutoff_70)':      {'pearson': 0.216},
    # Experimental: +NonBinders variant (main Figure-2c plot)
    'Graphinity EGNN exp+nonbinders (random)':       {'pearson': 0.382},
    'Graphinity EGNN exp+nonbinders (cutoff_70)':    {'pearson': 0.040},
    # Reference: FoldX computed vs experimental correlation
    'FoldX vs experimental':                         {'pearson': 0.46},
}
# Note: KIRBy groups by PDB complex (≈cutoff_100). Synthetic FoldX is stable
# across cutoffs (~0.89), but experimental collapses from 0.53 → 0.22 at cutoff_70.
# The "0.87" number previously cited is from the +RevMuts variant with random splits.


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_rep_ddg(X, labels, groups, model, n_folds):
    """Evaluate one (rep, model) for DDG regression via GroupKFold.

    Single-pass CV: runs folds once, collecting per-fold metrics AND the
    held-out predictions, then computes global-concatenation Pearson r
    (paper's method: concatenate all held-out predictions, compute a single
    Pearson r on the full array). This guarantees per-fold-mean Pearson and
    pearson_global use the SAME feature space (incl. _filter_high_dim_per_fold).

    Returns dict with pearson, spearman, r2, rmse as {metric: {mean, std}},
    plus pearson_global (single value, comparable to paper's reported numbers).
    """
    from scipy.stats import pearsonr
    from sklearn.base import clone
    from run_baseline_survey import (
        _make_cv, _maybe_scale, _filter_high_dim_per_fold, _score,
    )

    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    cv, cv_kwargs = _make_cv(groups, n_folds, labels)

    fold_metrics = []
    all_preds = np.full_like(labels, np.nan)
    for train_idx, val_idx in cv.split(X, labels, **cv_kwargs):
        X_tr, X_va = X[train_idx], X[val_idx]
        X_tr, X_va, _ = _filter_high_dim_per_fold(
            X_tr, X_va, labels[train_idx], 'regression')
        X_tr, X_va = _maybe_scale(X_tr, X_va)
        try:
            m = clone(model)
            m.fit(X_tr, labels[train_idx])
            preds = m.predict(X_va)
            all_preds[val_idx] = preds
            fold_metrics.append(_score(labels[val_idx], preds, 'regression'))
        except Exception as e:
            # Log rather than silently drop the fold (OPIG-18) — a missing fold
            # biases both the per-fold mean and pearson_global.
            print(f"      [fold failed: {type(model).__name__} "
                  f"{type(e).__name__}: {e}]", flush=True)
            continue

    if not fold_metrics:
        return None
    all_keys = fold_metrics[0].keys()
    result = {}
    for k in all_keys:
        vals = [f[k] for f in fold_metrics if k in f]
        result[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    valid = ~np.isnan(all_preds)
    if valid.sum() > 2:
        r_global, _ = pearsonr(labels[valid], all_preds[valid])
        result['pearson_global'] = float(r_global)
    else:
        result['pearson_global'] = float('nan')

    return result


def evaluate_transfer(rep_names, n_folds, max_foldx=5000):
    """Transfer learning: train on FoldX synthetic, evaluate on experimental.

    Train on (optionally subsampled) FoldX synthetic, test on ALL experimental.
    Groups experimental by PDB complex for reporting variance.
    """
    from scipy.stats import pearsonr, spearmanr
    from sklearn.base import clone
    from run_baseline_survey import _maybe_scale, _filter_high_dim_per_fold

    print(f"\n{'=' * 60}")
    print(f"Condition: transfer_foldx (train=synthetic, test=experimental)")
    print(f"{'=' * 60}")

    from kirby.datasets.graphinity import load_graphinity_dataset
    train_data = load_graphinity_dataset(preset='foldx', max_samples=max_foldx)
    test_data = load_dataset('ddg_combined')

    from experiments._shared import _smoke_subsample
    train_data = _smoke_subsample(train_data)
    test_data = _smoke_subsample(test_data)

    # Guard: unpaired sequences → empty strings
    for d in [train_data, test_data]:
        if d['light_seqs'] is None:
            d['light_seqs'] = [''] * len(d['heavy_seqs'])

    # PDB-level leakage filter (P0-5): FoldX synthetic is built from the same
    # SAbDab PDBs SKEMPI uses. Drop synthetic rows whose PDB also appears
    # in the experimental test set; otherwise sequences leak verbatim into
    # `pearson_global` and overstate transfer.
    test_pdbs = set()
    for key in ('pdb_ids', 'pdb'):
        if key in test_data and test_data[key] is not None:
            test_pdbs.update(str(p).upper() for p in test_data[key])
    if not test_pdbs:
        # Silence here means the filter did nothing and every number below is a
        # transfer score with the same complexes on both sides.
        raise ValueError(
            "transfer_foldx: the experimental test set carries no PDB "
            "identifiers, so the overlap filter cannot run. Without it the "
            "synthetic training rows share complexes with the test rows and "
            "the reported correlation is not a transfer measurement.")
    if test_pdbs:
        train_pdb_col = train_data.get('pdb_ids', train_data.get('pdb'))
        if train_pdb_col is None:
            raise ValueError(
                "transfer_foldx: the synthetic training set carries no PDB "
                "identifiers, so overlapping complexes cannot be dropped.")
        if train_pdb_col is not None:
            train_pdb_arr = np.asarray([str(p).upper() for p in train_pdb_col])
            keep_mask = ~np.isin(train_pdb_arr, list(test_pdbs))
            n_removed = int((~keep_mask).sum())
            if n_removed:
                for fld in ('heavy_seqs', 'light_seqs', 'labels',
                            'mutations', 'pdb_ids', 'pdb', 'groups'):
                    if fld in train_data and train_data[fld] is not None:
                        col = train_data[fld]
                        if hasattr(col, '__len__') and len(col) == len(keep_mask):
                            if isinstance(col, np.ndarray):
                                train_data[fld] = col[keep_mask]
                            else:
                                train_data[fld] = [v for v, k in zip(col, keep_mask) if k]
                print(f"  PDB leakage filter: removed {n_removed} synthetic rows "
                      f"whose PDB is in the experimental test set")

    n_train = len(train_data['labels'])
    n_test = len(test_data['labels'])
    print(f"  Train: {n_train} synthetic FoldX samples")
    print(f"  Test: {n_test} experimental SKEMPI samples")

    # Generate representations for both sets (with disk caching)
    from experiments.rep_cache import compute_or_load_reps

    reps_train = compute_or_load_reps(
        rep_names=rep_names,
        heavy_seqs=train_data['heavy_seqs'],
        light_seqs=train_data['light_seqs'],
        mutations=train_data.get('mutations'),
        dataset_tag=f"graphinity_foldx_{n_train}",
    )
    reps_test = compute_or_load_reps(
        rep_names=rep_names,
        heavy_seqs=test_data['heavy_seqs'],
        light_seqs=test_data['light_seqs'],
        mutations=test_data.get('mutations'),
        dataset_tag=f"ddg_combined_{n_test}",
    )

    # Only keep reps present in both train and test
    common_reps = sorted(set(reps_train) & set(reps_test))
    reps_train = {k: reps_train[k] for k in common_reps}
    reps_test = {k: reps_test[k] for k in common_reps}

    if not reps_train:
        print("  No reps available, skipping condition")
        return None

    y_train = train_data['labels']
    y_test = test_data['labels']
    models = get_model_factories(MODELS_LIST, 'regression', n_samples=n_train)

    print(f"\n  Evaluating {len(reps_train)} reps × {len(models)} models "
          f"(train-on-synthetic, test-on-experimental)...")
    results = {}
    for rep_name in sorted(reps_train.keys()):
        X_tr = reps_train[rep_name]
        X_te = reps_test[rep_name]
        # Match within-CV: filter high-dim reps before scaling so transfer
        # and within_* conditions use the same feature space.
        X_tr_f, X_te_f, _ = _filter_high_dim_per_fold(
            X_tr, X_te, y_train, 'regression'
        )
        # Hoist scaling out of inner model loop — same scaler for every model.
        X_tr_s, X_te_s = _maybe_scale(X_tr_f, X_te_f)
        rep_results = {}
        for model_name, factory in models.items():
            try:
                m = factory()
                m.fit(X_tr_s, y_train)
                preds = m.predict(X_te_s)
                r_p, _ = pearsonr(y_test, preds)
                r_s, _ = spearmanr(y_test, preds)
                rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
                # Single train/test pair — no fold variance to report.
                # Use flat values; consumers below handle both shapes via
                # `_metric_value()` (P1-11 fix).
                rep_results[model_name] = {
                    'pearson': float(r_p),
                    'spearman': float(r_s),
                    'rmse': rmse,
                    'pearson_global': float(r_p),
                }
                print(f"    {rep_name:25s} {model_name:8s}  "
                      f"r={r_p:.4f}  rho={r_s:.4f}  rmse={rmse:.4f}")
            except Exception as e:
                print(f"    {rep_name:25s} {model_name:8s}  FAILED ({e})")
        if rep_results:
            results[rep_name] = {'dim': X_tr.shape[1], 'models': rep_results}

    return {
        'n_samples': n_test,
        'n_train': n_train,
        'condition': 'transfer_foldx',
        'dataset': 'graphinity_foldx -> ddg_combined',
        'results': results,
    }


def evaluate_condition(condition, rep_names, n_folds, max_foldx=5000,
                       pruning_mode='default_pruning'):
    """Evaluate one condition: load data, generate reps, evaluate all.

    Returns dict with per-rep results.
    """
    if condition == 'transfer_foldx':
        return evaluate_transfer(rep_names, n_folds, max_foldx=max_foldx)

    ds_name = CONDITION_DATASETS[condition]

    print(f"\n{'=' * 60}")
    print(f"Condition: {condition} (dataset: {ds_name})")
    print(f"{'=' * 60}")

    data = load_dataset(ds_name)

    # A loader that returns pdb_ids but no groups gets a per-row 'groups' array
    # here, BEFORE smoke subsampling, so _smoke_subsample (which knows 'groups'
    # but not 'pdb_ids') keeps groups aligned with the subsampled labels and
    # sequences. Without this, groups would stay full-length and GroupKFold
    # would see inconsistent sample counts under KIRBY_SMOKE_N.
    if data.get('groups') is None and data.get('pdb_ids') is not None:
        pdb_ids = data['pdb_ids']
        unique_ids = {pid: i for i, pid in enumerate(sorted(set(pdb_ids)))}
        data['groups'] = np.array([unique_ids[pid] for pid in pdb_ids])

    from experiments._shared import _smoke_subsample
    data = _smoke_subsample(data)
    heavy_seqs = data['heavy_seqs']
    light_seqs = data['light_seqs']
    if light_seqs is None:
        light_seqs = [''] * len(heavy_seqs)
    labels = data['labels']
    groups = data.get('groups')
    n = len(labels)

    print(f"  {n} samples, DDG range [{labels.min():.2f}, {labels.max():.2f}]")
    if groups is not None:
        print(f"  {len(np.unique(groups))} groups (PDB complexes)")

    # Generate representations (with disk caching for expensive LM reps)
    from experiments.rep_cache import compute_or_load_reps
    reps = compute_or_load_reps(
        rep_names=rep_names,
        heavy_seqs=heavy_seqs,
        light_seqs=light_seqs,
        mutations=data.get('mutations'),
        dataset_tag=f"{ds_name}_{n}",
    )

    if not reps:
        print("  No reps available, skipping condition")
        return None

    # Honest per-model, held-out re-score block (contract json_convention).
    # Populated from the SEARCH's own per-fold candidate/baseline scores — the
    # search selects hybrid features on the TRAIN fold ONLY (per fold, inside
    # find_best_hybrid) and records held-out fold scores per DOWNSTREAM model,
    # so it is the honest counterpart to the full-data hybrid built at :454 and
    # the max-over-models baseline in print_analysis (both selection-optimistic).
    # None when no search was produced; reader must guard.
    honest_block = None

    # Build hybrid with aac as augmentation rep (matches DDG domain config)
    try:
        from kirby.hybrid import apply_feature_selection, apply_augmentation_selection
        if groups is None:
            # GroupKFold without real groups silently degenerates to KFold,
            # which violates the lab-group prohibition on random splits.
            raise ValueError(
                f"evaluate_condition({condition}): groups is None for "
                f"{ds_name}; refusing to fall back to np.arange(n) "
                "(would degenerate GroupKFold to KFold = random split)."
            )
        grp = groups
        AUG_REPS = {'aac'}
        base = {k: v for k, v in reps.items() if k not in AUG_REPS}
        aug_dict = {k: v for k, v in reps.items() if k in AUG_REPS}
        if not aug_dict:
            print(f"  WARNING: augmentation reps {AUG_REPS} not found in loaded reps, "
                  f"proceeding without augmentation")
        from experiments._shared import (
            resolve_pruning_kwargs, get_best_search_for_downstream)
        _pkw = resolve_pruning_kwargs(pruning_mode, len(labels),
                                       dataset_label=f"{ds_name}/n{n}")
        _smoke = _smoke_search_kwargs()
        search, _meta = get_best_search_for_downstream(
            base if base else reps, labels, grp,
            downstream_model='rf', budget=None,
            augmentations=aug_dict if aug_dict else None,
            task_type='regression',
            dataset_name=ds_name,
            return_fitted=True,
            # _smoke may override the evaluator list (rf-only) for local runs;
            # falls back to the full INTERNAL_EVALUATORS on server.
            internal_evaluators=_smoke.get('internal_evaluators',
                                           INTERNAL_EVALUATORS),
            condition_suffix=f'n{n}',
            adaptive_pruning=0.25,
            **{k: v for k, v in _smoke.items()
               if k not in ('speed', 'internal_evaluators')},
            **_pkw,
        )
        # Persist per-fold scores for BOTH arms straight from the SearchResult.
        # build_honest_rescore_input groups hybrid candidates by DOWNSTREAM model
        # (each with its per-fold 'fold_scores') and inverts search.baseline_fold_scores
        # into {model: {rep: [folds]}} — respecting whatever model gate the search
        # ran under (never a fixed set). skip_secondary defaults True.
        if search is not None:
            honest_block = build_honest_rescore_input(search)
        top3 = extract_top3_base_reps(search) if search is not None else None
        if top3:
            print(f"  Top-3 base reps (by RF): {', '.join(f'{nm}={s:.4f}' for nm, s in top3)}")
        fitted = (search.fitted_results.get('rf')
                  if search is not None else None)
        if fitted is not None:
            fi = {r: {'selected_indices': idx}
                  for r, idx in fitted.selected_indices.items()}
            X_hybrid = apply_feature_selection(reps, fi)
            if fitted.augmentation_info and aug_dict:
                X_aug = apply_augmentation_selection(aug_dict, fitted.augmentation_info)
                if X_aug is not None:
                    X_hybrid = np.hstack([X_hybrid, X_aug])
            reps['kirby_hybrid'] = X_hybrid
            print(f"  kirby_hybrid: {reps['kirby_hybrid'].shape[1]}D (hybrid)")
    except Exception as e:
        print(f"  Hybrid build failed: {e}")

    # Evaluate
    models = get_model_factories(MODELS_LIST, 'regression', n_samples=n)
    print(f"\n  Evaluating {len(reps)} reps × {len(models)} models × {n_folds}-fold CV...")
    results = {}
    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
        rep_results = {}
        for model_name, factory in models.items():
            t0 = time.time()
            res = evaluate_rep_ddg(X, labels, groups, factory(), n_folds)
            elapsed = time.time() - t0
            if res is not None:
                rep_results[model_name] = res
                r_p = res['pearson']['mean']
                r_g = res.get('pearson_global', float('nan'))
                rho = res['spearman']['mean']
                tag = ' ***' if rep_name == 'kirby_hybrid' else ''
                print(f"    {rep_name:25s} {model_name:8s}  "
                      f"r={r_p:.4f}  r_g={r_g:.4f}  rho={rho:.4f}  "
                      f"({elapsed:.1f}s){tag}")
        if rep_results:
            results[rep_name] = {'dim': X.shape[1], 'models': rep_results}

    return {
        'n_samples': n,
        'condition': condition,
        'dataset': ds_name,
        'results': results,
        # Honest arithmetic input (contract key names). One block per condition
        # since each condition is its own dataset/search; a reader loads it and
        # calls honest_rescore(block['candidates_by_model'],
        # block['baselines_by_model'], block['lower_is_better']).
        'honest_rescore': honest_block,
    }


# =============================================================================
# ANALYSIS
# =============================================================================

def _metric_value(metrics, key):
    """Return the mean of a metric whether it's stored as a flat float or
    as {'mean': float, 'std': float}. Lets transfer_foldx (single train/test,
    no std) coexist with k-fold conditions (mean+std) without lying about
    a `std=0.0` for the former (P1-11).
    """
    v = metrics.get(key)
    if isinstance(v, dict):
        return v.get('mean', float('nan'))
    return float('nan') if v is None else float(v)


def print_analysis(all_results):
    """Print cross-condition comparison table."""
    print(f"\n{'=' * 100}")
    print(f"GRAPHINITY DDG COMPARISON (OPIG B2)")
    print(f"{'=' * 100}")

    # Paper baselines
    print(f"\n  Paper baselines:")
    for method, metrics in PAPER_BASELINES.items():
        r_p = metrics.get('pearson', float('nan'))
        print(f"    {method}: Pearson r = {r_p:.2f}")

    # Per-condition best results
    print(f"\n  {'Condition':20s} {'N':>7s} {'Best rep':>25s} {'Model':>8s} "
          f"{'Pearson r':>10s} {'Spearman':>10s} {'Hybrid r':>10s} {'Delta':>8s}")
    print(f"  {'-'*20} {'-'*7} {'-'*25} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for cond in ALL_CONDITIONS:
        if cond not in all_results:
            continue

        cr = all_results[cond]
        n = cr['n_samples']
        results = cr['results']

        best_indiv_name, best_model_name = None, None
        best_pearson = float('-inf')
        hybrid_pearson = float('nan')

        for rep_name, rep_data in results.items():
            for model_name, metrics in rep_data['models'].items():
                r_p = _metric_value(metrics, 'pearson')
                if rep_name == 'kirby_hybrid':
                    if np.isnan(hybrid_pearson) or r_p > hybrid_pearson:
                        hybrid_pearson = r_p
                else:
                    if r_p > best_pearson:
                        best_pearson = r_p
                        best_indiv_name = rep_name
                        best_model_name = model_name

        # Also get spearman for best individual
        best_spearman = float('nan')
        if best_indiv_name and best_model_name:
            best_spearman = _metric_value(
                results[best_indiv_name]['models'][best_model_name], 'spearman')

        delta = hybrid_pearson - best_pearson if not np.isnan(hybrid_pearson) else float('nan')
        h_s = f"{hybrid_pearson:.4f}" if not np.isnan(hybrid_pearson) else "—"
        d_s = f"{delta:+.4f}" if not np.isnan(delta) else "—"
        rep_label = best_indiv_name or "—"

        print(f"  {cond:20s} {n:7d} {rep_label:>25s} {best_model_name or '—':>8s} "
              f"{best_pearson:10.4f} {best_spearman:10.4f} {h_s:>10s} {d_s:>8s}")

    # Gap analysis — compare to EGNN at cutoff_100 (closest to KIRBy's GroupKFold by PDB)
    # The paper reports global-concatenation Pearson (single r over concatenated
    # held-out predictions), so compare apples-to-apples via 'pearson_global'
    # rather than per-fold-mean Pearson. Also: exclude kirby_hybrid from the
    # best_r aggregation — we want the best INDIVIDUAL rep vs. the paper's
    # structure-based model; the hybrid is reported separately above.
    egnn_r = PAPER_BASELINES['Graphinity EGNN experimental (cutoff_100)']['pearson']
    print(f"\n  Structure-based EGNN reference (cutoff_100): Pearson r = {egnn_r:.3f}")
    for cond in ALL_CONDITIONS:
        if cond not in all_results:
            continue
        results = all_results[cond]['results']
        best_r = max(
            (v for rep_name, rd in results.items() if rep_name != 'kirby_hybrid'
             for m in rd['models'].values()
             for v in (_metric_value(m, 'pearson_global'),) if not np.isnan(v)),
            default=float('nan'),
        )
        gap = egnn_r - best_r
        print(f"    {cond}: best individual Pearson r_g = {best_r:.4f}, "
              f"gap to EGNN = {gap:.4f}")


def print_honest_analysis(all_results):
    """Per-model, held-out honest comparison (selection-optimism corrected).

    Replaces the same-fold max-over-models baseline in print_analysis: for each
    condition and each DOWNSTREAM model actually evaluated, the shared helper
    re-picks the winning hybrid recipe and the best single rep using the OTHER
    folds and scores on the held-out fold (identical tax to both arms). Reuses
    experiments._shared.honest_rescore — no re-fitting, pure arithmetic.
    """
    print(f"\n{'=' * 100}")
    print(f"HONEST PER-MODEL HELD-OUT RE-SCORE (Pearson r; higher better)")
    print(f"{'=' * 100}")
    for cond in ALL_CONDITIONS:
        cr = all_results.get(cond)
        if not cr:
            continue
        block = cr.get('honest_rescore')
        if not block:
            # transfer_foldx (single train/test, no search) and any condition
            # whose hybrid build failed carry no honest block.
            print(f"\n  {cond}: no honest_rescore block (no CV search)")
            continue
        res = honest_rescore(
            block.get('candidates_by_model'),
            block.get('baselines_by_model'),
            block.get('lower_is_better', False),
            exclude_folds=hr_exclude_folds(block),
        )
        if not res:
            print(f"\n  {cond}: honest_rescore produced no models")
            continue
        print(f"\n  {cond}:")
        print(f"    {'Model':>8s} {'Hybrid':>10s} {'BestSingle':>11s} "
              f"{'Delta':>9s} {'Folds':>6s}")
        for model in sorted(res):
            r = res[model]
            h = r['hybrid_honest']
            b = r['best_single_honest']
            d = r['delta']
            hs = f"{h:.4f}" if not np.isnan(h) else "—"
            bs = f"{b:.4f}" if not np.isnan(b) else "—"
            ds = f"{d:+.4f}" if not np.isnan(d) else "—"
            flag = ' (single-fold: optimistic)' if r.get('single_fold') else ''
            print(f"    {model:>8s} {hs:>10s} {bs:>11s} {ds:>9s} "
                  f"{r.get('n_folds_used', 0):>6d}{flag}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Graphinity DDG experiment (OPIG B2). '
                    'Tests sequence-based reps for DDG prediction.')
    parser.add_argument(
        '--conditions', type=str, default=None,
        help='Comma-separated conditions '
             f'(default: {",".join(ALL_CONDITIONS)})')
    parser.add_argument(
        '--reps', type=str, default=None,
        help='Comma-separated rep names (default: hand-picked set of 8)')
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help='Number of GroupKFold folds (default: 5)')
    parser.add_argument(
        '--max-foldx', type=int, default=5000,
        help='Max samples for FoldX synthetic preset (default: 5000). '
             'FoldX raw size is ~943K rows; subsampling is essential.')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON path '
             '(default: tests/results/comparisons/opig_graphinity_ddg.json)')
    parser.add_argument(
        '--pruning-mode', type=str, default='default_pruning',
        help="Adaptive pruning mode (default 'default_pruning'). "
             "'aggressive_pruning' = F16-validated multi-stage pruning "
             "(checkpoints=[25,50,100], td=0.5) applied globally to all models.")

    args = parser.parse_args()

    conditions = (args.conditions.split(',') if args.conditions
                  else ALL_CONDITIONS)
    rep_names = (args.reps.split(',') if args.reps else DEFAULT_REPS)

    max_foldx = args.max_foldx

    print("=" * 70)
    print("GRAPHINITY DDG EXPERIMENT (OPIG B2)")
    print(f"  Paper: Hummer et al. 2025, Nature Computational Science")
    print(f"  Conditions: {conditions}")
    print(f"  Reps: {rep_names}")
    print(f"  Models: {MODELS_LIST}")
    print(f"  Folds: {args.n_folds}")
    print(f"  Max samples (FoldX): {max_foldx}")
    print("=" * 70)

    all_results = {}
    for cond in conditions:
        if cond not in CONDITION_DATASETS:
            print(f"WARNING: Unknown condition '{cond}', skipping")
            continue
        ds_name = CONDITION_DATASETS[cond]
        # transfer_foldx has ds_name=None (handled specially in evaluate_condition)
        if ds_name is not None and ds_name not in DATASET_REGISTRY:
            print(f"WARNING: Dataset '{ds_name}' not in registry, skipping")
            continue
        try:
            cond_results = evaluate_condition(
                cond, rep_names, args.n_folds,
                max_foldx=max_foldx,
                pruning_mode=args.pruning_mode)
            if cond_results is not None:
                all_results[cond] = cond_results
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Cross-condition analysis
    if all_results:
        print_analysis(all_results)
        # Honest per-model held-out comparison (the leak-free counterpart to the
        # same-fold table above).
        print_honest_analysis(all_results)

    # Save results
    output_path = (args.output or
                   'tests/results/comparisons/opig_graphinity_ddg.json')
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\nResults saved: {p}")


if __name__ == '__main__':
    main()
