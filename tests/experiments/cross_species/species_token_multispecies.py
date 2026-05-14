#!/usr/bin/env python3
"""species_token_multispecies (T-PT.5) — does a species token earn its keep?

Cross-species pretraining variant `species_token_multispecies` from
`docs/cross_species_pretraining_design.md` §2. The species-token DNN
(`SpeciesTokenDNNRegressor`) is jointly pretrained on all six mammalian
species at all targets (corpus `multispecies_chembl`), conditioned on a
learned 16-D species embedding + an N-D target one-hot, then fine-tuned on the
four vet ChEMBL eval targets.

The headline test is the species-token toggle: each held-out vet compound is
predicted twice from the SAME fitted model — once with its real species index
(cow/dog), once forcing the human index. The RMSE delta quantifies what the
species token actually learned. Per the §4 decision rule, the token "earns its
keep" if |delta| >= 0.03 RMSE.

The target-token vocabulary size N is swept N in {10, 50, 200} (design §6 Q4):
each N has its own pretrained checkpoint set and its own target_vocab.json,
produced by `kirby.models.species_token_dnn` pretraining.

Conditions per (vet target x rep):
  baselines (N-independent):
    - dnn_scratch  DNNRegressor, no pretraining
    - rf           RandomForest via kirby.find_best_hybrid.get_model
                   (Ridge intentionally excluded, see feedback_no_ridge)
  token conditions per N:
    - pt_finetune  SpeciesTokenDNNRegressor(freeze_layers=0) fine-tuned;
                   scored with the real species token AND the human token
    - pt_zero_shot SpeciesTokenDNNRegressor(zero_shot=True), no fine-tune;
                   same real-vs-human toggle
Plus a per-target `kirby_hybrid` rep (find_best_hybrid on the standard rep
set), evaluated with the baselines only.

Datasets (locked in docs/cross_species_vet_eval_targets.json):
  cow_CA4 (n~1541), cow_DRD2 (n~406), dog_PTGS2 (n~110), cow_PTGS2 (n~28)

Reps: ecfp4, maccs, pdv, mol2vec, mordred, avalon — the C4 set. Token
conditions are skipped for any (rep, N) whose pretrained checkpoint is
missing; baselines still run.

Splits: 5-fold scaffold GroupKFold, 3-fold fallback at n < 150.

Usage:
    # Smoke test (local, fast) — needs a pretrained checkpoint set for N
    python tests/experiments/cross_species/species_token_multispecies.py \\
        --keys cow_DRD2 --reps ecfp4 --n-folds 2

    # Full run (ARC)
    python tests/experiments/cross_species/species_token_multispecies.py

Reference:
    Imrie et al. (2018). JCIM 58(11):2319-2330.
    TACTIC — species-conditional multi-task QSAR.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

warnings.filterwarnings('ignore')

from find_best_hybrid_molecular import get_representations, assign_scaffold_groups
from run_baseline_survey import _make_cv, _score
from experiments._shared import convert_numpy
from experiments.cross_species.vet_eval_data import (
    load_vet_dataset, list_vet_keys, load_vet_targets,
    DEFAULT_CACHE_DIR as VET_DEFAULT_CACHE_DIR,
)
# Reuse the same_target_transfer machinery verbatim where the shape is
# identical — only the model/conditions differ for the token variant.
from experiments.cross_species.same_target_transfer import (
    evaluate_condition, _build_kirby_hybrid, _resolve_folds, SMALL_N_THRESHOLD,
)
from kirby.models.species_token_dnn import (
    SpeciesTokenDNNRegressor, pack_token_columns, HUMAN_IDX,
)

DEFAULT_REPS = ['ecfp4', 'maccs', 'pdv', 'mol2vec', 'mordred', 'avalon']
HIDDEN_SIZES = (256, 128, 64)
DEFAULT_PRETRAIN_DIR = '/Volumes/seagate/kirby/chembl_pretrain/species_token'
N_SWEEP = [10, 50, 200]
DEFAULT_N_FOLDS = 5
TOGGLE_DELTA_THRESHOLD = 0.03  # §4: species token "earns its keep" at >=0.03


def _load_target_vocab(pretrain_dir, n_targets):
    """Read the shared target_vocab.json for one N (written by pretraining)."""
    path = Path(pretrain_dir) / f'N{n_targets}' / 'target_vocab.json'
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())['target_vocab']
    except Exception as e:
        print(f"    WARNING: could not read {path}: {e}")
        return None


def build_baseline_models(n_samples, task_type='regression'):
    """dnn_scratch + rf — N-independent, evaluated once per (target, rep)."""
    from kirby.models.dnn import DNNRegressor
    from kirby.find_best_hybrid import get_model

    models = {
        'dnn_scratch': DNNRegressor(
            hidden_sizes=HIDDEN_SIZES, dropout=0.2, lr=0.001,
            epochs=200, batch_size=256, patience=20, random_state=42),
    }
    rf = get_model('rf', task_type=task_type, n_samples=n_samples)
    if rf is not None:
        models['rf'] = rf
    return models


def _agg_folds(fold_metrics):
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


def evaluate_token_condition(X_packed, y, groups, model, n_folds):
    """Scaffold GroupKFold for one token condition.

    Each fold fits ONE SpeciesTokenDNNRegressor and predicts the held-out
    fold twice: with the row's real species index, then forcing the human
    index. No feature scaling is applied here — the model scales its
    molecular columns internally and the trailing two token columns must stay
    integer indices.

    Returns {'species_token': agg, 'human_token': agg, 'toggle_delta_rmse'}.
    delta = RMSE(human) - RMSE(real); positive => the species token lowered RMSE.
    """
    from sklearn.base import clone

    cv, cv_kwargs = _make_cv(groups, n_folds, y)
    splits = list(cv.split(X_packed, y, **cv_kwargs))

    sp_metrics, hu_metrics = [], []
    for tr, va in splits:
        try:
            m = clone(model)
            m.fit(X_packed[tr], y[tr])
            m.force_species_idx = None
            p_sp = m.predict(X_packed[va])
            m.force_species_idx = HUMAN_IDX
            p_hu = m.predict(X_packed[va])
            m.force_species_idx = None
            if not (np.all(np.isfinite(p_sp)) and np.all(np.isfinite(p_hu))):
                continue
            sp_metrics.append(_score(y[va], p_sp, 'regression'))
            hu_metrics.append(_score(y[va], p_hu, 'regression'))
        except Exception as e:
            print(f" [err: {e}]", end='')
            continue

    sp_agg, hu_agg = _agg_folds(sp_metrics), _agg_folds(hu_metrics)
    sr = sp_agg['mean'].get('rmse')
    hr = hu_agg['mean'].get('rmse')
    delta = float(hr - sr) if (sr is not None and hr is not None) else None
    return {'species_token': sp_agg, 'human_token': hu_agg,
            'toggle_delta_rmse': delta}


def _fmt(v, w=8):
    return f"{v:>{w}.4f}" if (v is not None and not np.isnan(v)) else f"{'N/A':>{w}}"


def evaluate_target(key, rep_names, n_folds_requested, pretrain_dir,
                    vet_cache_dir=VET_DEFAULT_CACHE_DIR):
    print(f"\n{'='*70}")
    print(f"Vet target: {key}")
    print('=' * 70)

    smiles, y, task_type = load_vet_dataset(key, cache_dir=vet_cache_dir)
    n = len(smiles)
    print(f"  {n} compounds, pActivity {y.min():.2f}..{y.max():.2f}")
    if n < 10:
        print(f"  SKIPPED: only {n} compounds")
        return {'n_samples': n, 'skipped': True}

    entry = load_vet_targets()[key]
    species = entry['species']
    target_id = entry['target_chembl_id']

    groups, n_scaffolds = assign_scaffold_groups(smiles)
    n_folds = _resolve_folds(n, n_folds_requested)
    print(f"  species={species}, target={target_id}, "
          f"{n_scaffolds} scaffolds, {n_folds}-fold CV")

    reps = get_representations(smiles, ','.join(rep_names))

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
        'species': species,
        'target_chembl_id': target_id,
        'n_sweep': N_SWEEP,
        'reps': {},
    }

    baseline_models = build_baseline_models(n_samples=n, task_type=task_type)
    hs_str = '_'.join(str(h) for h in HIDDEN_SIZES)

    for rep_name in eval_reps:
        if rep_name not in reps:
            print(f"  {rep_name}: SKIPPED (rep failed to build)")
            continue
        X_mol = np.asarray(reps[rep_name], dtype=np.float64)
        if X_mol.ndim != 2 or X_mol.shape[0] != n:
            print(f"  {rep_name}: SKIPPED (shape mismatch)")
            continue

        print(f"\n  --- {rep_name} (dim={X_mol.shape[1]}) ---")
        rep_block = {'dim': int(X_mol.shape[1]), 'baselines': {}, 'by_N': {}}

        for cond_name, model in baseline_models.items():
            t0 = time.time()
            print(f"    {cond_name:16s}", end='', flush=True)
            res = evaluate_condition(X_mol, y, groups, model, n_folds)
            res['time_s'] = time.time() - t0
            print(f" RMSE={_fmt(res['mean'].get('rmse'))} "
                  f"({res['time_s']:.1f}s, folds={res['n_folds_completed']})")
            rep_block['baselines'][cond_name] = res

        # Token conditions per N. kirby_hybrid has no pretrained checkpoint,
        # so it gets baselines only.
        if rep_name != 'kirby_hybrid':
            for N in N_SWEEP:
                vocab = _load_target_vocab(pretrain_dir, N)
                pt_path = os.path.join(pretrain_dir, f'N{N}',
                                       f'{rep_name}_{hs_str}.pt')
                if vocab is None or not os.path.exists(pt_path):
                    print(f"    N={N:<3d} SKIPPED "
                          f"(missing checkpoint/vocab: {pt_path})")
                    continue

                X_packed = pack_token_columns(
                    X_mol, [species] * n, [target_id] * n, vocab)
                n_block = {'n_targets': N, 'pt_finetune': None,
                           'pt_zero_shot': None}

                ft_model = SpeciesTokenDNNRegressor(
                    n_targets=N, hidden_sizes=HIDDEN_SIZES, dropout=0.2,
                    pretrained_path=pt_path, finetune_lr=0.0001,
                    freeze_layers=0, epochs=200, batch_size=256,
                    patience=20, random_state=42)
                t0 = time.time()
                print(f"    N={N:<3d} pt_finetune  ", end='', flush=True)
                ft = evaluate_token_condition(X_packed, y, groups,
                                              ft_model, n_folds)
                ft['time_s'] = time.time() - t0
                d = ft['toggle_delta_rmse']
                print(f"sp_RMSE={_fmt(ft['species_token']['mean'].get('rmse'))} "
                      f"hu_RMSE={_fmt(ft['human_token']['mean'].get('rmse'))} "
                      f"Δ={(f'{d:+.4f}' if d is not None else 'N/A'):>9s} "
                      f"({ft['time_s']:.1f}s)")
                n_block['pt_finetune'] = ft

                zs_model = SpeciesTokenDNNRegressor(
                    n_targets=N, hidden_sizes=HIDDEN_SIZES, dropout=0.2,
                    pretrained_path=pt_path, zero_shot=True, random_state=42)
                t0 = time.time()
                print(f"    N={N:<3d} pt_zero_shot ", end='', flush=True)
                zs = evaluate_token_condition(X_packed, y, groups,
                                              zs_model, n_folds)
                zs['time_s'] = time.time() - t0
                d = zs['toggle_delta_rmse']
                print(f"sp_RMSE={_fmt(zs['species_token']['mean'].get('rmse'))} "
                      f"hu_RMSE={_fmt(zs['human_token']['mean'].get('rmse'))} "
                      f"Δ={(f'{d:+.4f}' if d is not None else 'N/A'):>9s} "
                      f"({zs['time_s']:.1f}s)")
                n_block['pt_zero_shot'] = zs

                rep_block['by_N'][str(N)] = n_block

        ds_results['reps'][rep_name] = rep_block

    return ds_results


def print_analysis(all_results, title="SPECIES-TOKEN MULTISPECIES ANALYSIS"):
    """Per-target table: baseline RMSE + per-N token RMSE and toggle delta.

    §4 decision rule — the species token "earns its keep" if the fine-tuned
    real-vs-human toggle moves RMSE by >= 0.03 on held-out vet rows.
    """
    print(f"\n{'='*88}")
    print(title)
    print(f"{'='*88}")

    for key, ds in all_results.items():
        if ds.get('skipped') or 'reps' not in ds:
            print(f"\n--- {key}: skipped/failed ---")
            continue
        print(f"\n--- {key} (n={ds['n_samples']}, species={ds.get('species')}, "
              f"{ds['n_folds']}-fold) ---")

        toggle_deltas = []  # fine-tune toggle deltas across (rep, N)
        for rep_name in sorted(ds['reps']):
            block = ds['reps'][rep_name]
            base = block['baselines']
            scratch = base.get('dnn_scratch', {}).get('mean', {}).get('rmse')
            rf = base.get('rf', {}).get('mean', {}).get('rmse')
            print(f"  {rep_name:14s}  dnn_scratch={_fmt(scratch)}  rf={_fmt(rf)}")
            for N in N_SWEEP:
                nb = block['by_N'].get(str(N))
                if nb is None:
                    continue
                ft = nb.get('pt_finetune') or {}
                zs = nb.get('pt_zero_shot') or {}
                ft_sp = ft.get('species_token', {}).get('mean', {}).get('rmse')
                ft_hu = ft.get('human_token', {}).get('mean', {}).get('rmse')
                ft_d = ft.get('toggle_delta_rmse')
                zs_sp = zs.get('species_token', {}).get('mean', {}).get('rmse')
                zs_d = zs.get('toggle_delta_rmse')
                if ft_d is not None:
                    toggle_deltas.append(ft_d)
                flag = ''
                if ft_d is not None and abs(ft_d) >= TOGGLE_DELTA_THRESHOLD:
                    flag = '  <- token effect'
                print(f"      N={N:<3d}  ft_sp={_fmt(ft_sp)} ft_hu={_fmt(ft_hu)} "
                      f"ft_Δ={(f'{ft_d:+.4f}' if ft_d is not None else 'N/A'):>9s}  "
                      f"zs_sp={_fmt(zs_sp)} "
                      f"zs_Δ={(f'{zs_d:+.4f}' if zs_d is not None else 'N/A'):>9s}"
                      f"{flag}")

        if toggle_deltas:
            arr = np.array(toggle_deltas)
            n_meas = int((np.abs(arr) >= TOGGLE_DELTA_THRESHOLD).sum())
            print(f"  species token: mean fine-tune toggle Δ={arr.mean():+.4f} "
                  f"RMSE; |Δ|>={TOGGLE_DELTA_THRESHOLD} on {n_meas}/{len(arr)} "
                  f"(rep x N) cells")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--keys', type=str, default=None,
                    help='Comma-separated vet target keys '
                         '(default: all in cross_species_vet_eval_targets.json)')
    ap.add_argument('--reps', type=str, default=None,
                    help=f'Comma-separated rep names '
                         f'(default: {",".join(DEFAULT_REPS)})')
    ap.add_argument('--n-folds', type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument('--pretrain-dir', type=str, default=DEFAULT_PRETRAIN_DIR,
                    help=f'Root of species-token pretrained checkpoints; each '
                         f'N lives in {{pretrain_dir}}/N{{N}}/ '
                         f'(default: {DEFAULT_PRETRAIN_DIR})')
    ap.add_argument('--vet-cache-dir', type=str, default=VET_DEFAULT_CACHE_DIR,
                    help=f'ChEMBL activity CSV cache dir for vet targets '
                         f'(default: {VET_DEFAULT_CACHE_DIR}). Server-local on ARC.')
    ap.add_argument('--output', type=str, default=None,
                    help='Output JSON path (default: tests/results/'
                         'cross_species/species_token_multispecies.json)')
    args = ap.parse_args()

    keys = args.keys.split(',') if args.keys else list_vet_keys()
    reps = args.reps.split(',') if args.reps else DEFAULT_REPS

    print("=" * 70)
    print("species_token_multispecies  (T-PT.5)")
    print(f"  Vet targets:  {keys}")
    print(f"  Reps:         {reps}")
    print(f"  N sweep:      {N_SWEEP}")
    print(f"  Folds:        {args.n_folds} "
          f"(3-fold fallback at n<{SMALL_N_THRESHOLD})")
    print(f"  Pretrain dir: {args.pretrain_dir}")
    print(f"  Vet cache:    {args.vet_cache_dir}")
    print("=" * 70)

    hs_str = '_'.join(str(h) for h in HIDDEN_SIZES)
    missing = []
    for N in N_SWEEP:
        for r in reps:
            p = os.path.join(args.pretrain_dir, f'N{N}', f'{r}_{hs_str}.pt')
            if not os.path.exists(p):
                missing.append(f'N{N}/{r}')
    if missing:
        print(f"\nWARNING: missing species-token checkpoints: {missing}")
        print(f"  Pretrain first, e.g.:")
        print(f"  python -m kirby.models.species_token_dnn "
              f"--rep {','.join(reps)} --n-targets 50 "
              f"--output-dir {args.pretrain_dir}")
        print(f"  Continuing with available checkpoints + baselines.\n")

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
               'tests/results/cross_species/species_token_multispecies.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(convert_numpy(results), indent=2))
    print(f"\nResults written: {out}")


if __name__ == '__main__':
    main()
