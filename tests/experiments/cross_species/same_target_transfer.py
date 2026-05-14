#!/usr/bin/env python3
"""same_target_transfer (T-PT.3) — does human chemistry transfer at a shared target?

Cross-species pretraining variant `same_target_transfer` from
`docs/cross_species_pretraining_design.md` §2. Pretrain a DNN on bulk human
ChEMBL (the C4 Imrie corpus, ~500K activities), then fine-tune on the four
small vet ChEMBL eval targets at the *same* target. Question: does a model
that learned human chemistry at target T transfer to dog/cow at target T?

This is the heavier counterpart to foundation_hybrid_vet (T-PT.2): T-PT.2 asks
"can find_best_hybrid alone handle small vet data"; T-PT.3 asks "does explicit
human-corpus pretraining beat from-scratch."

Conditions per (vet target × rep):
  - dnn_scratch    DNNRegressor, no pretraining (control)
  - dnn_tl_full    PretrainedDNNRegressor(freeze_layers=0,  lr=1e-4) full fine-tune
  - dnn_tl_frozen  PretrainedDNNRegressor(freeze_layers=-1, lr=1e-3) linear probe
  - zero_shot      PretrainedDNNRegressor(zero_shot=True) — pretrained weights +
                   pretrain feature scaler, NO fine-tune
  - rf             RandomForest via kirby.find_best_hybrid.get_model — simple
                   model control (Ridge intentionally excluded, see
                   feedback_no_ridge)
Plus a per-target `kirby_hybrid` rep (find_best_hybrid on the standard rep set),
evaluated with the non-TL conditions only.

Datasets (locked in docs/cross_species_vet_eval_targets.json):
  - cow_CA4   (CHEMBL281,  n≈1,541)
  - cow_DRD2  (CHEMBL3998, n≈406)
  - dog_PTGS2 (CHEMBL4033, n≈110)
  - cow_PTGS2 (CHEMBL3331, n≈28)

Reps: ecfp4, maccs, pdv, mol2vec, mordred, avalon — the C4 set (the reps with
pretrained ChEMBL DNN checkpoints). TL + zero_shot conditions are skipped for
any rep whose checkpoint is missing; dnn_scratch + rf still run.

Splits: 5-fold scaffold GroupKFold, 3-fold fallback at n < 150.

Usage:
    # Smoke test (local, fast)
    python tests/experiments/cross_species/same_target_transfer.py \\
        --keys cow_DRD2 --reps ecfp4 --n-folds 2

    # Full run (ARC)
    python tests/experiments/cross_species/same_target_transfer.py

Reference:
    Imrie et al. (2018). JCIM 58(11):2319-2330.
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

warnings.filterwarnings('ignore')

from find_best_hybrid_molecular import (
    get_representations, assign_scaffold_groups,
)
from run_baseline_survey import _make_cv, _score, _maybe_scale
from experiments._shared import convert_numpy, FULL_BUDGETS
from experiments.cross_species.vet_eval_data import (
    load_vet_dataset, list_vet_keys, DEFAULT_CACHE_DIR as VET_DEFAULT_CACHE_DIR,
)

# C4 rep set — the reps that have pretrained ChEMBL DNN checkpoints
# (pretrain_chembl_dnn.py supports exactly these six).
DEFAULT_REPS = ['ecfp4', 'maccs', 'pdv', 'mol2vec', 'mordred', 'avalon']
HIDDEN_SIZES = (256, 128, 64)
DEFAULT_PRETRAIN_DIR = '/Volumes/seagate/kirby/chembl_pretrain'

SMALL_N_THRESHOLD = 150
SMALL_N_FOLDS = 3
DEFAULT_N_FOLDS = 5


def _resolve_folds(n: int, requested: int) -> int:
    if n < SMALL_N_THRESHOLD:
        return min(requested, SMALL_N_FOLDS)
    return requested


def build_models(rep_name, pretrain_dir, n_samples, task_type='regression'):
    """Build the conditions for one rep.

    dnn_scratch + rf always; the three pretrained conditions only if the
    rep's ChEMBL checkpoint exists on disk.
    """
    from kirby.models.dnn import DNNRegressor
    from kirby.models.dnn_pretrained import PretrainedDNNRegressor
    from kirby.find_best_hybrid import get_model

    hs_str = '_'.join(str(h) for h in HIDDEN_SIZES)
    pt_path = os.path.join(pretrain_dir, f'{rep_name}_{hs_str}.pt')
    has_pretrained = os.path.exists(pt_path)

    models = {
        'dnn_scratch': DNNRegressor(
            hidden_sizes=HIDDEN_SIZES, dropout=0.2, lr=0.001,
            epochs=200, batch_size=256, patience=20, random_state=42),
    }
    rf = get_model('rf', task_type=task_type, n_samples=n_samples)
    if rf is not None:
        models['rf'] = rf

    if has_pretrained:
        models['dnn_tl_full'] = PretrainedDNNRegressor(
            hidden_sizes=HIDDEN_SIZES, dropout=0.2, pretrained_path=pt_path,
            finetune_lr=0.0001, freeze_layers=0, epochs=200, batch_size=256,
            patience=20, random_state=42)
        models['dnn_tl_frozen'] = PretrainedDNNRegressor(
            hidden_sizes=HIDDEN_SIZES, dropout=0.2, pretrained_path=pt_path,
            finetune_lr=0.001, freeze_layers=-1, epochs=200, batch_size=256,
            patience=20, random_state=42)
        models['zero_shot'] = PretrainedDNNRegressor(
            hidden_sizes=HIDDEN_SIZES, dropout=0.2, pretrained_path=pt_path,
            zero_shot=True, random_state=42)
    else:
        print(f"    WARNING: no pretrained weights at {pt_path}")
        print(f"    Skipping TL + zero_shot for {rep_name} "
              f"(dnn_scratch + rf still run)")

    return models


def evaluate_condition(X, y, groups, model, n_folds):
    """Scaffold GroupKFold for one (rep, condition); mean/std metric dict."""
    from sklearn.base import clone

    cv, cv_kwargs = _make_cv(groups, n_folds, y)
    splits = list(cv.split(X, y, **cv_kwargs))

    fold_metrics = []
    for tr, va in splits:
        Xtr, Xva = _maybe_scale(X[tr], X[va])
        try:
            m = clone(model)
            m.fit(Xtr, y[tr])
            p = m.predict(Xva)
            if not np.all(np.isfinite(p)):
                continue
            fold_metrics.append(_score(y[va], p, 'regression'))
        except Exception as e:
            print(f" [err: {e}]", end='')
            continue

    if not fold_metrics:
        return {'mean': {}, 'std': {}, 'n_folds_completed': 0}

    keys = fold_metrics[0].keys()
    return {
        'mean': {k: float(np.mean([f[k] for f in fold_metrics if k in f]))
                 for k in keys},
        'std':  {k: float(np.std([f[k] for f in fold_metrics if k in f]))
                 for k in keys},
        'n_folds_completed': len(fold_metrics),
    }


def _build_kirby_hybrid(reps, y, groups, key, task_type):
    """find_best_hybrid on the standard rep set; returns the selected-feature
    matrix as an extra 'rep', mirroring imrie_transfer_learning.py."""
    try:
        from kirby.find_best_hybrid import find_best_hybrid
        from kirby.hybrid import apply_feature_selection
        search = find_best_hybrid(
            base_reps=reps, labels=y, groups=groups,
            budgets=FULL_BUDGETS, speed='full', return_fitted=True,
            verbose=False, cache=True, dataset_name=f"vet_{key}",
            tier='general', metric='rmse', task_type=task_type,
            augmentation_strategy='none', adaptive_pruning=0.25)
        fitted = (search.fitted_results or {}).get('rf')
        if fitted is None:
            return None
        fi = {r: {'selected_indices': idx}
              for r, idx in fitted.selected_indices.items()}
        return apply_feature_selection(reps, fi)
    except Exception as e:
        print(f"  kirby_hybrid build failed: {e}")
        return None


def evaluate_target(key, rep_names, n_folds_requested, pretrain_dir,
                    vet_cache_dir=VET_DEFAULT_CACHE_DIR):
    print(f"\n{'='*70}")
    print(f"Vet target: {key}")
    print('='*70)

    smiles, y, task_type = load_vet_dataset(key, cache_dir=vet_cache_dir)
    n = len(smiles)
    print(f"  {n} compounds, pActivity {y.min():.2f}..{y.max():.2f}")
    if n < 10:
        print(f"  SKIPPED: only {n} compounds")
        return {'n_samples': n, 'skipped': True}

    groups, n_scaffolds = assign_scaffold_groups(smiles)
    n_folds = _resolve_folds(n, n_folds_requested)
    print(f"  {n_scaffolds} scaffolds, {n_folds}-fold CV")

    reps = get_representations(smiles, ','.join(rep_names))

    # kirby_hybrid baseline (non-TL conditions only — no checkpoint exists
    # for a hybrid rep, so build_models naturally skips its TL conditions).
    eval_reps = list(rep_names)
    hybrid_X = _build_kirby_hybrid(reps, y, groups, key, task_type)
    if hybrid_X is not None:
        reps['kirby_hybrid'] = hybrid_X
        eval_reps.append('kirby_hybrid')
        print(f"  kirby_hybrid built: {hybrid_X.shape[1]}D")

    ds_results = {
        'n_samples': n,
        'n_scaffolds': int(n_scaffolds),
        'n_folds': n_folds,
        'task_type': task_type,
        'reps': {},
    }

    for rep_name in eval_reps:
        if rep_name not in reps:
            print(f"  {rep_name}: SKIPPED (rep failed to build)")
            continue
        X = np.asarray(reps[rep_name], dtype=np.float64)
        if X.ndim != 2 or X.shape[0] != n:
            print(f"  {rep_name}: SKIPPED (shape mismatch)")
            continue

        print(f"\n  --- {rep_name} (dim={X.shape[1]}) ---")
        models = build_models(rep_name, pretrain_dir, n_samples=n,
                              task_type=task_type)
        rep_block = {'dim': int(X.shape[1]), 'conditions': {}}
        for cond_name, model in models.items():
            t0 = time.time()
            print(f"    {cond_name:14s}", end='', flush=True)
            res = evaluate_condition(X, y, groups, model, n_folds)
            res['time_s'] = time.time() - t0
            rmse = res['mean'].get('rmse', float('nan'))
            rho = res['mean'].get('spearman', float('nan'))
            print(f" RMSE={rmse:.4f} ρ={rho:.4f} "
                  f"({res['time_s']:.1f}s, folds={res['n_folds_completed']})")
            rep_block['conditions'][cond_name] = res
        ds_results['reps'][rep_name] = rep_block

    return ds_results


def print_analysis(all_results, title="SAME-TARGET TRANSFER ANALYSIS"):
    """Per-target condition table + transfer verdict (§4 decision rule:
    transfer 'works' ⇔ dnn_tl_full beats dnn_scratch by ≥0.05 RMSE).

    Shared by same_target_transfer (T-PT.3) and rodent_to_vet (T-PT.4) — the
    table shape is identical, only the pretraining corpus differs.
    """
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")

    for key, ds in all_results.items():
        if ds.get('skipped') or 'reps' not in ds:
            print(f"\n--- {key}: skipped/failed ---")
            continue
        print(f"\n--- {key} (n={ds['n_samples']}, {ds['n_folds']}-fold) ---")

        conds = []
        for rep_res in ds['reps'].values():
            for c in rep_res['conditions']:
                if c not in conds:
                    conds.append(c)

        header = f"  {'rep':14s}" + ''.join(f"  {c:>13s}" for c in conds)
        if 'dnn_scratch' in conds and 'dnn_tl_full' in conds:
            header += f"  {'Δ_TL':>9s}"
        print(header)
        print(f"  {'-'*14}" + ''.join(f"  {'-'*13}" for _ in conds))

        scratch_vals, tl_vals = [], []
        for rep_name in sorted(ds['reps']):
            block = ds['reps'][rep_name]['conditions']
            line = f"  {rep_name:14s}"
            for c in conds:
                v = block.get(c, {}).get('mean', {}).get('rmse', float('nan'))
                line += f"  {v:>13.4f}" if not np.isnan(v) else f"  {'N/A':>13s}"
            s = block.get('dnn_scratch', {}).get('mean', {}).get('rmse', float('nan'))
            t = block.get('dnn_tl_full', {}).get('mean', {}).get('rmse', float('nan'))
            if not (np.isnan(s) or np.isnan(t)):
                line += f"  {t - s:>+9.4f}"  # negative = TL helped
                scratch_vals.append(s)
                tl_vals.append(t)
            elif 'dnn_scratch' in conds and 'dnn_tl_full' in conds:
                line += f"  {'N/A':>9s}"
            print(line)

        if len(scratch_vals) >= 2:
            deltas = np.array(tl_vals) - np.array(scratch_vals)
            n_helped = int((deltas <= -0.05).sum())
            mean_delta = float(deltas.mean())
            print(f"  transfer: mean Δ_TL={mean_delta:+.4f} RMSE; "
                  f"TL beats scratch by ≥0.05 on {n_helped}/{len(deltas)} reps")
            if len(scratch_vals) >= 3:
                rho_rank, _ = spearmanr(scratch_vals, tl_vals)
                print(f"  rep-ranking preservation (scratch vs TL): ρ={rho_rank:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--keys', type=str, default=None,
                    help='Comma-separated vet target keys '
                         '(default: all in cross_species_vet_eval_targets.json)')
    ap.add_argument('--reps', type=str, default=None,
                    help=f'Comma-separated rep names (default: {",".join(DEFAULT_REPS)})')
    ap.add_argument('--n-folds', type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument('--pretrain-dir', type=str, default=DEFAULT_PRETRAIN_DIR,
                    help=f'Directory with pretrained ChEMBL DNN checkpoints '
                         f'(default: {DEFAULT_PRETRAIN_DIR})')
    ap.add_argument('--vet-cache-dir', type=str, default=VET_DEFAULT_CACHE_DIR,
                    help=f'ChEMBL activity CSV cache dir for vet targets '
                         f'(default: {VET_DEFAULT_CACHE_DIR}). Server-local on ARC.')
    ap.add_argument('--output', type=str, default=None,
                    help='Output JSON path (default: '
                         'tests/results/cross_species/same_target_transfer.json)')
    args = ap.parse_args()

    keys = args.keys.split(',') if args.keys else list_vet_keys()
    reps = args.reps.split(',') if args.reps else DEFAULT_REPS

    print("=" * 70)
    print("same_target_transfer  (T-PT.3)")
    print(f"  Vet targets:  {keys}")
    print(f"  Reps:         {reps}")
    print(f"  Folds:        {args.n_folds} (3-fold fallback at n<{SMALL_N_THRESHOLD})")
    print(f"  Pretrain dir: {args.pretrain_dir}")
    print(f"  Vet cache:    {args.vet_cache_dir}")
    print("=" * 70)

    hs_str = '_'.join(str(h) for h in HIDDEN_SIZES)
    missing = [r for r in reps
               if not os.path.exists(os.path.join(args.pretrain_dir,
                                                   f'{r}_{hs_str}.pt'))]
    if missing:
        print(f"\nWARNING: missing pretrained checkpoints for: {missing}")
        print(f"  Run pretraining first:")
        print(f"  python -m kirby.models.pretrain_chembl_dnn "
              f"--rep {','.join(missing)} --output-dir {args.pretrain_dir}")
        print(f"  Continuing with available checkpoints + scratch/rf baselines.\n")

    results = {}
    for key in keys:
        try:
            results[key] = evaluate_target(
                key, reps, args.n_folds, args.pretrain_dir,
                vet_cache_dir=args.vet_cache_dir)
        except Exception as e:
            print(f"  FAILED on {key}: {e}")
            import traceback
            traceback.print_exc()
            results[key] = {'error': str(e)[:200]}

    print_analysis(results)

    out = Path(args.output or
               'tests/results/cross_species/same_target_transfer.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(convert_numpy(results), indent=2))
    print(f"\nResults written: {out}")


if __name__ == '__main__':
    main()
