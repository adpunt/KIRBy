#!/usr/bin/env python3
"""
Imrie Transfer Learning Experiment (OPIG C4)

Tests whether pre-training DNNs on broad ChEMBL bioactivity data improves
molecular property prediction, and whether this changes the rep ranking.

Scientific questions:
  1. Does TL help? (per-rep: dnn_tl_full RMSE vs dnn_scratch RMSE)
  2. Does TL change rep ranking? (Spearman rank correlation under DNN vs TL-DNN)
  3. Does model improvement beat rep improvement?
     (TL-DNN on ECFP4 vs Ridge on mordred)

Conditions per (dataset, rep) — 4 total (GP dropped, see build_models):
  - dnn_scratch:  DNNRegressor (no pre-training, control)
  - dnn_tl_full:  PretrainedDNNRegressor(freeze=0, lr=1e-4) — full fine-tuning
  - dnn_tl_frozen: PretrainedDNNRegressor(freeze=-1, lr=1e-3) — linear probe
  - ridge:        RidgeCV (simple model control)

Usage:
    # Smoke test (local, ~5 min)
    python tests/experiments/imrie_transfer_learning.py \
        --datasets logd --reps ecfp4 --n-folds 2

    # Full evaluation
    python tests/experiments/imrie_transfer_learning.py \
        --datasets logd,caco2,herg_ki \
        --reps ecfp4,maccs,pdv,mol2vec,mordred,avalon \
        --n-folds 5

Reference:
    Imrie et al. (2018). JCIM 58(11):2319-2330.
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from find_best_hybrid_molecular import (
    load_dataset,
    get_representations,
    assign_scaffold_groups,
)
from run_baseline_survey import _make_cv, _score, _maybe_scale
from experiments._shared import convert_numpy

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_DATASETS = ['logd', 'caco2', 'herg_ki']
DEFAULT_REPS = ['ecfp4', 'maccs', 'pdv', 'mol2vec', 'mordred', 'avalon']
DEFAULT_PRETRAIN_DIR = '/Volumes/seagate/kirby/chembl_pretrain'
HIDDEN_SIZES = (256, 128, 64)


# =============================================================================
# MODEL BUILDERS
# =============================================================================

def list_conditions(rep_name, pretrain_dir):
    """Return (condition_names, has_pretrained, pt_path) for a rep.

    Used by evaluate_dataset to know which conditions to iterate; per-fold
    instantiation happens in build_model_for_fold so each fold gets its
    own random_state (no clone-induced seed sharing).
    """
    hs_str = '_'.join(str(h) for h in HIDDEN_SIZES)
    pt_path = os.path.join(pretrain_dir, f'{rep_name}_{hs_str}.pt')
    has_pretrained = os.path.exists(pt_path)

    # GP previously enabled only for rep_dim <= 500 — that produced an
    # asymmetric model set across reps (ECFP4/mordred skipped GP) and
    # muddied the cross-rep model comparison. Dropped entirely (P1-2).
    conds = ['dnn_scratch', 'ridge']
    if has_pretrained:
        conds += ['dnn_tl_full', 'dnn_tl_frozen']
    else:
        print(f"    WARNING: No pre-trained weights at {pt_path}")
        print(f"    Skipping TL conditions for {rep_name}")
    return conds, has_pretrained, pt_path


def build_model_for_fold(cond_name, pt_path, fold_idx):
    """Instantiate a fresh model for one fold with fold-varying seed.

    Building inside the fold loop avoids the sklearn.base.clone() trap where
    all folds share random_state=42 and produce identical init noise.
    """
    from kirby.models.dnn import DNNRegressor
    from kirby.models.dnn_pretrained import PretrainedDNNRegressor
    from sklearn.linear_model import RidgeCV

    seed = 42 + fold_idx

    if cond_name == 'dnn_scratch':
        return DNNRegressor(
            hidden_sizes=HIDDEN_SIZES, dropout=0.2, lr=0.001,
            epochs=200, batch_size=256, patience=20, random_state=seed)
    if cond_name == 'ridge':
        return RidgeCV(alphas=np.logspace(-4, 4, 50))
    if cond_name == 'dnn_tl_full':
        return PretrainedDNNRegressor(
            hidden_sizes=HIDDEN_SIZES, dropout=0.2,
            pretrained_path=pt_path, finetune_lr=0.0001,
            freeze_layers=0, epochs=200, batch_size=256,
            patience=20, random_state=seed)
    if cond_name == 'dnn_tl_frozen':
        return PretrainedDNNRegressor(
            hidden_sizes=HIDDEN_SIZES, dropout=0.2,
            pretrained_path=pt_path, finetune_lr=0.001,
            freeze_layers=-1, epochs=200, batch_size=256,
            patience=20, random_state=seed)
    raise ValueError(f"Unknown condition: {cond_name}")


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_condition(X, labels, groups, cond_name, pt_path, n_folds):
    """Evaluate one condition via scaffold GroupKFold.

    Builds a fresh model per fold with seed=42+fold_idx so init variance is
    visible across folds (sklearn.base.clone would otherwise pin one seed).
    Passes scaffold groups through to DNN ES splits.

    Returns dict of {metric: value} with mean and std.
    """
    cv, cv_kwargs = _make_cv(groups, n_folds, labels)
    splits = list(cv.split(X, labels, **cv_kwargs))

    from kirby.models.dnn import DNNRegressor
    from kirby.models.dnn_pretrained import PretrainedDNNRegressor

    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = _maybe_scale(X[train_idx], X[val_idx])
        try:
            m = build_model_for_fold(cond_name, pt_path, fold_idx)
            # Pass groups to DNN models so internal ES split is scaffold-aware.
            # Only the two custom DNN models accept `groups=`; Ridge etc.
            # raise ValueError (not TypeError) on sklearn 1.8+ metadata routing,
            # so gate on model type rather than catching exceptions.
            if isinstance(m, (DNNRegressor, PretrainedDNNRegressor)):
                m.fit(X_train, labels[train_idx],
                      groups=np.asarray(groups)[train_idx])
            else:
                m.fit(X_train, labels[train_idx])
            preds = m.predict(X_val)
            if not np.all(np.isfinite(preds)):
                print(" [NaN]", end='')
                continue
            fold_metrics.append(_score(labels[val_idx], preds, 'regression'))
        except Exception as e:
            print(f" [err: {e}]", end='')
            continue

    if not fold_metrics:
        return {'mean': {}, 'std': {}}

    all_keys = fold_metrics[0].keys()
    mean_m = {k: float(np.mean([f[k] for f in fold_metrics if k in f]))
              for k in all_keys}
    std_m = {k: float(np.std([f[k] for f in fold_metrics if k in f]))
             for k in all_keys}
    return {'mean': mean_m, 'std': std_m}


def evaluate_dataset(dataset_name, rep_names, n_folds, pretrain_dir):
    """Evaluate all reps x conditions for one dataset.

    Returns dict of results.
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    smiles, labels, task_type = load_dataset(dataset_name)
    # Registry cross-check — catches silent drift between molecular
    # loaders and DATASET_REGISTRY.
    from kirby.registry import crosscheck_task_type
    crosscheck_task_type(dataset_name, task_type)
    groups, n_scaffolds = assign_scaffold_groups(smiles)
    print(f"  {len(labels)} samples, {n_scaffolds} scaffolds, task={task_type}")

    # Generate reps (only the requested fast ones)
    rep_str = ','.join(rep_names)
    reps = get_representations(smiles, rep_str)

    # Pull the kirby_hybrid CV score from the cached search rather than
    # re-evaluating with a separately-fit hybrid: re-fitting feature
    # selection on the full dataset and then evaluating on (any subset of)
    # that same dataset leaks the val rows into the selection step. The
    # search's `score_mean` is the cleanly cross-validated number.
    hybrid_search_score = None
    try:
        from experiments._shared import get_best_search_for_downstream
        # Use the pool helper so we benefit from any precomputed (tier, aug)
        # cache that already exists for logd/caco2. augmentations=None ->
        # pool only iterates aug_condition='none' (helper sets aug_strat
        # per iteration), so no cache collision is possible.
        _search, _meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model='rf', budget=None,
            augmentations=None,
            speed='full', metric='rmse', task_type=task_type,
            dataset_name=dataset_name,
            return_fitted=True, adaptive_pruning=0.25,
        )
        if isinstance(_meta, dict) and 'score_mean' in _meta:
            hybrid_search_score = float(_meta['score_mean'])
            print(f"  kirby_hybrid search CV {_meta.get('metric', 'rmse')}"
                  f"={hybrid_search_score:.4f} (from cached search)")
    except Exception as e:
        print(f"  Hybrid search lookup failed: {e}")

    ds_results = {
        'n_samples': len(labels),
        'n_scaffolds': n_scaffolds,
        'task_type': task_type,
        'reps': {},
        'kirby_hybrid_search_score': hybrid_search_score,
    }

    for rep_name in rep_names:
        if rep_name not in reps:
            print(f"  {rep_name}: SKIPPED (not generated)")
            continue

        X = reps[rep_name]
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] != len(labels):
            print(f"  {rep_name}: SKIPPED (shape mismatch)")
            continue

        print(f"\n  --- {rep_name} (dim={X.shape[1]}) ---")
        conds, _has_pt, pt_path = list_conditions(rep_name, pretrain_dir)

        rep_results = {'dim': X.shape[1], 'conditions': {}}
        for cond_name in conds:
            t0 = time.time()
            print(f"    {cond_name}...", end='', flush=True)
            result = evaluate_condition(
                X, labels, groups, cond_name, pt_path, n_folds)
            elapsed = time.time() - t0

            rmse = result['mean'].get('rmse', float('nan'))
            r2 = result['mean'].get('r2', float('nan'))
            rho = result['mean'].get('spearman', float('nan'))
            print(f" RMSE={rmse:.4f}, R²={r2:.4f}, ρ={rho:.4f} ({elapsed:.1f}s)")

            result['time'] = elapsed
            rep_results['conditions'][cond_name] = result

        ds_results['reps'][rep_name] = rep_results

    return ds_results


# =============================================================================
# ANALYSIS
# =============================================================================

def print_analysis(all_results):
    """Print key analyses across all datasets."""
    print(f"\n{'='*80}")
    print("TRANSFER LEARNING ANALYSIS")
    print(f"{'='*80}")

    for ds_name, ds_res in all_results.items():
        print(f"\n--- {ds_name} (n={ds_res['n_samples']}) ---")

        # Collect per-rep RMSE for each condition
        conditions = set()
        rep_data = {}
        for rep_name, rep_res in ds_res['reps'].items():
            rep_data[rep_name] = {}
            for cond_name, cond_res in rep_res['conditions'].items():
                conditions.add(cond_name)
                rep_data[rep_name][cond_name] = cond_res['mean'].get('rmse', float('nan'))

        conditions = sorted(conditions)

        # Table header
        header = f"  {'Rep':15s}"
        for c in conditions:
            header += f"  {c:>14s}"
        if 'dnn_scratch' in conditions and 'dnn_tl_full' in conditions:
            header += f"  {'delta_TL':>10s}"
        print(header)
        print(f"  {'-'*15}" + f"  {'-'*14}" * len(conditions) +
              (f"  {'-'*10}" if 'dnn_scratch' in conditions and 'dnn_tl_full' in conditions else ""))

        for rep_name in sorted(rep_data.keys()):
            line = f"  {rep_name:15s}"
            for c in conditions:
                val = rep_data[rep_name].get(c, float('nan'))
                line += f"  {val:>14.4f}" if not np.isnan(val) else f"  {'N/A':>14s}"

            # Delta: negative = TL helped (lower RMSE is better)
            if 'dnn_scratch' in conditions and 'dnn_tl_full' in conditions:
                scratch = rep_data[rep_name].get('dnn_scratch', float('nan'))
                tl = rep_data[rep_name].get('dnn_tl_full', float('nan'))
                if not np.isnan(scratch) and not np.isnan(tl):
                    delta = tl - scratch
                    line += f"  {delta:>+10.4f}"
                else:
                    line += f"  {'N/A':>10s}"
            print(line)

        # Rank correlation analysis
        if 'dnn_scratch' in conditions and 'dnn_tl_full' in conditions:
            scratch_vals = []
            tl_vals = []
            rep_names_ordered = sorted(rep_data.keys())
            for rn in rep_names_ordered:
                s = rep_data[rn].get('dnn_scratch', float('nan'))
                t = rep_data[rn].get('dnn_tl_full', float('nan'))
                if not np.isnan(s) and not np.isnan(t):
                    scratch_vals.append(s)
                    tl_vals.append(t)

            if len(scratch_vals) >= 3:
                rho_rank, p_rank = spearmanr(scratch_vals, tl_vals)
                print(f"\n  Rep ranking preservation (DNN vs TL-DNN):")
                print(f"    Spearman ρ = {rho_rank:.4f} (p={p_rank:.4f})")
                if abs(rho_rank) > 0.8:
                    print(f"    → Rankings PRESERVED: TL doesn't change which reps are best")
                elif abs(rho_rank) > 0.5:
                    print(f"    → Rankings PARTIALLY preserved")
                else:
                    print(f"    → Rankings CHANGED: TL reshuffles rep ordering")

        # Model vs rep trade-off
        if 'ridge' in conditions and 'dnn_tl_full' in conditions:
            print(f"\n  Model vs Rep trade-off:")
            for rep_name in sorted(rep_data.keys()):
                tl_rmse = rep_data[rep_name].get('dnn_tl_full', float('nan'))
                ridge_rmse = rep_data[rep_name].get('ridge', float('nan'))
                if not np.isnan(tl_rmse) and not np.isnan(ridge_rmse):
                    if tl_rmse < ridge_rmse:
                        print(f"    {rep_name}: TL-DNN beats Ridge "
                              f"({tl_rmse:.4f} vs {ridge_rmse:.4f})")
                    else:
                        print(f"    {rep_name}: Ridge beats TL-DNN "
                              f"({ridge_rmse:.4f} vs {tl_rmse:.4f})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Imrie transfer learning experiment (OPIG C4). '
                    'Tests pre-trained DNN vs scratch DNN across molecular reps.')
    parser.add_argument(
        '--datasets', type=str, default=None,
        help='Comma-separated dataset names (default: logd,caco2,herg)')
    parser.add_argument(
        '--reps', type=str, default=None,
        help='Comma-separated rep names (default: ecfp4,maccs,pdv,mol2vec,mordred,avalon)')
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help='Number of scaffold GroupKFold folds (default: 5)')
    parser.add_argument(
        '--pretrain-dir', type=str, default=DEFAULT_PRETRAIN_DIR,
        help=f'Directory with pre-trained weights (default: {DEFAULT_PRETRAIN_DIR})')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON path (default: tests/results/comparisons/imrie_transfer_learning.json)')

    args = parser.parse_args()

    datasets = (args.datasets.split(',') if args.datasets
                else DEFAULT_DATASETS)
    rep_names = (args.reps.split(',') if args.reps
                 else DEFAULT_REPS)

    print("=" * 70)
    print("IMRIE TRANSFER LEARNING EXPERIMENT (OPIG C4)")
    print(f"  Datasets: {datasets}")
    print(f"  Reps: {rep_names}")
    print(f"  Folds: {args.n_folds}")
    print(f"  Pre-trained weights: {args.pretrain_dir}")
    print("=" * 70)

    # Check pre-trained weights exist
    hs_str = '_'.join(str(h) for h in HIDDEN_SIZES)
    missing = []
    for rn in rep_names:
        pt_path = os.path.join(args.pretrain_dir, f'{rn}_{hs_str}.pt')
        if not os.path.exists(pt_path):
            missing.append(rn)
    if missing:
        print(f"\nWARNING: Missing pre-trained weights for: {missing}")
        print(f"  Run pre-training first:")
        print(f"  python -m kirby.models.pretrain_chembl_dnn "
              f"--rep {','.join(missing)} --output-dir {args.pretrain_dir}")
        print(f"  Continuing with available weights + scratch baselines...\n")

    all_results = {}
    for ds_name in datasets:
        try:
            ds_results = evaluate_dataset(ds_name, rep_names, args.n_folds,
                                          args.pretrain_dir)
            all_results[ds_name] = ds_results
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Print analysis
    if all_results:
        print_analysis(all_results)

    # Coverage summary: which (rep, condition) pairs actually ran, and which
    # were skipped because pretrain weights were missing.
    print(f"\n{'='*70}")
    print("COVERAGE SUMMARY (rep, condition pairs)")
    print(f"{'='*70}")
    hs_str = '_'.join(str(h) for h in HIDDEN_SIZES)
    for ds_name, ds_res in all_results.items():
        reps_block = ds_res.get('reps', {}) if isinstance(ds_res, dict) else {}
        print(f"  {ds_name}:")
        for rn in rep_names:
            if rn not in reps_block:
                print(f"    {rn}: rep not generated / skipped")
                continue
            ran = sorted(reps_block[rn].get('conditions', {}).keys())
            pt_path = os.path.join(args.pretrain_dir, f'{rn}_{hs_str}.pt')
            missing_pt = not os.path.exists(pt_path)
            note = " [no pretrain weights]" if missing_pt else ""
            print(f"    {rn}: {', '.join(ran) if ran else '(none)'}{note}")

    # Save results
    output_path = (args.output or
                   'tests/results/comparisons/imrie_transfer_learning.json')
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\nResults saved: {p}")


if __name__ == '__main__':
    main()
