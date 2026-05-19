#!/usr/bin/env python3
"""x_ab_ness — cross-species antibody-ness classifier (T-Ab.4).

Trains a one-vs-rest VH classifier using the T-Ab.3 cross-species loader:

  --target-species dog  → canineness (DoggifAI vs the 5-species background)
  --target-species cow  → bovineness (PRJNA690660 vs the 5-species background)

Wires the loader output through `find_best_hybrid` exactly as
`opig_humanness.py` does for the existing humanness task: classification
metric (AUROC), no rep augmentations, StratifiedGroupKFold using the
ANARCI-derived V-gene groups, the same curated antibody rep subset that
the OPIG humanness experiment uses, the standard DOWNSTREAM_MODELS
*minus Ridge* per the no-Ridge feedback.

Outputs one JSON to ``tests/results/cross_species/x_ab_<target>ness.json``
keyed by per-rep + hybrid metrics + the find_best_hybrid SearchResult
summary. The cached SearchResult under
``~/.cache/kirby/searches/{target}ness_v1__general__none__full.pkl`` is
what T-Ab.5 consumes for the therapeutic eval.

Usage
-----

    # Dry-run on tiny budgets — local sanity check only.
    python tests/experiments/cross_species/x_ab_ness.py \\
        --target-species dog --dry-run

    # Full canineness run (ARC).
    sbatch tests/slurm_scripts/submit_x_ab_ness.sh dog

    # Full bovineness run (ARC).
    sbatch tests/slurm_scripts/submit_x_ab_ness.sh cow
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

warnings.filterwarnings('ignore')

from experiments._shared import (  # noqa: E402
    convert_numpy, get_model_factories,
    DOWNSTREAM_MODELS, INTERNAL_EVALUATORS, FULL_BUDGETS,
)
from kirby.datasets.cross_species_ness import (  # noqa: E402
    V1_SUPPORTED_SPECIES, load_cross_species_ness,
)


# Curated VH rep subset — mirrors opig_humanness.DEFAULT_REPS so the
# canineness / bovineness classifier picks from the same vocabulary as
# the published humanness study.
DEFAULT_REPS = [
    'blosum', 'onehot', 'aac',          # classical
    'ablang2', 'esm2', 'antiberty',     # LMs
    'igt5', 'prot_t5',                  # additional LMs
    'abroberta',                         # Ab-RoBERTa
    'paratope',                          # structural (sequence-based)
]

# DOWNSTREAM_MODELS minus 'ridge' per [[feedback_no_ridge]]. Ridge is
# never wired into cross-species classifiers regardless of what the
# shared list contains.
DOWNSTREAM_MODELS_NO_RIDGE = [m for m in DOWNSTREAM_MODELS if m != 'ridge']

DEFAULT_N_FOLDS = 5
SMALL_N_THRESHOLD = 150
SMALL_N_FOLDS = 3


def _resolve_folds(n: int, requested: int) -> int:
    if n < SMALL_N_THRESHOLD:
        return min(requested, SMALL_N_FOLDS)
    return requested


def _score_clf(y_true: np.ndarray, y_pred: np.ndarray,
               y_score: Optional[np.ndarray]) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score, matthews_corrcoef,
    )
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
    }
    try:
        if y_score is not None:
            out['auroc'] = float(roc_auc_score(y_true, y_score))
        else:
            out['auroc'] = float('nan')
    except ValueError:
        out['auroc'] = float('nan')
    return out


def _evaluate_rep_clf(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                     model, n_folds: int) -> Dict[str, Dict[str, float]]:
    """StratifiedGroupKFold; per-fold AUROC+ACC+F1w+MCC; mean+std at the end."""
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []
    for tr, va in cv.split(X, y, groups=groups):
        try:
            m = clone(model)
            m.fit(X[tr], y[tr])
            pred = m.predict(X[va])
            if hasattr(m, 'predict_proba'):
                proba = m.predict_proba(X[va])
                score = proba[:, 1] if proba.shape[1] == 2 else None
            elif hasattr(m, 'decision_function'):
                score = m.decision_function(X[va])
            else:
                score = None
            fold_metrics.append(_score_clf(y[va], pred, score))
        except Exception as exc:
            print(f"      [fold error: {exc}]")
            continue
    if not fold_metrics:
        return {'mean': {}, 'std': {}, 'n_folds_completed': 0}
    keys = fold_metrics[0].keys()
    return {
        'mean': {k: float(np.mean([f[k] for f in fold_metrics
                                   if k in f and np.isfinite(f[k])]))
                 for k in keys},
        'std':  {k: float(np.std([f[k] for f in fold_metrics
                                  if k in f and np.isfinite(f[k])]))
                 for k in keys},
        'n_folds_completed': len(fold_metrics),
    }


def evaluate_target(
    target_species: str, rep_names: List[str],
    n_folds_requested: int, dry_run: bool = False,
) -> Dict:
    print(f"\n{'=' * 70}")
    print(f"Target: {target_species}ness  v1")
    print('=' * 70)

    data = load_cross_species_ness(
        target_species=target_species,
        mode='one_vs_rest',
        # explicit source pool keeps the cache key + log readable
        source_species=V1_SUPPORTED_SPECIES,
    )
    heavy_seqs: List[str] = data['heavy_seqs']
    light_seqs: List[str] = data['light_seqs']
    labels: np.ndarray = data['labels']
    groups: np.ndarray = data['groups']
    n = len(heavy_seqs)
    n_pos = int((labels == 1).sum())
    print(f"  n={n}  positives={n_pos}  ({100 * n_pos / max(n, 1):.1f}%)")
    print(f"  per-species counts: {data['per_species_counts']}")
    n_folds = _resolve_folds(n, n_folds_requested)
    if dry_run:
        n_folds = 2
        print(f"  DRY-RUN: forcing n_folds={n_folds}")

    # --- Build all reps once. compute_or_load_reps wraps the rep cache so
    # repeated runs reuse the per-(rep, dataset_tag) embedding files.
    from experiments.rep_cache import compute_or_load_reps
    dataset_tag = f"{target_species}ness_v1"
    reps = compute_or_load_reps(
        rep_names=rep_names,
        heavy_seqs=heavy_seqs,
        light_seqs=light_seqs,
        dataset_tag=dataset_tag,
    )
    if not reps:
        return {'error': 'no reps built; check rep cache'}

    out: Dict = {
        'target_species': target_species,
        'n_samples': n,
        'n_positives': n_pos,
        'n_folds': n_folds,
        'per_species_counts': data['per_species_counts'],
        'baselines_per_rep': {},
        'hybrid': {},
    }

    # --- Per-rep baseline survey
    models = get_model_factories(
        DOWNSTREAM_MODELS_NO_RIDGE, 'classification', n_samples=n)
    print(f"\n  Per-rep baselines ({len(reps)} reps × {len(models)} models × "
          f"{n_folds}-fold)...")
    for rep_name in sorted(reps.keys()):
        X = np.asarray(reps[rep_name], dtype=np.float64)
        if X.ndim != 2 or X.shape[0] != n:
            print(f"    {rep_name:25s}  SKIPPED shape={X.shape}")
            continue
        rep_block = {'dim': int(X.shape[1]), 'models': {}}
        for model_name, factory in models.items():
            t0 = time.time()
            res = _evaluate_rep_clf(X, labels, groups, factory(), n_folds)
            res['time_s'] = time.time() - t0
            auc = res['mean'].get('auroc', float('nan'))
            f1w = res['mean'].get('f1_weighted', float('nan'))
            print(f"    {rep_name:25s} {model_name:8s}  "
                  f"AUROC={auc:.4f}  F1w={f1w:.4f}  "
                  f"({res['time_s']:.1f}s)")
            rep_block['models'][model_name] = res
        out['baselines_per_rep'][rep_name] = rep_block

    # --- find_best_hybrid (classification, no aug, AUC metric)
    print(f"\n  find_best_hybrid ({len(reps)} reps × budgets="
          f"{FULL_BUDGETS if not dry_run else [FULL_BUDGETS[0]]})...")
    try:
        from kirby.find_best_hybrid import find_best_hybrid
        t0 = time.time()
        search = find_best_hybrid(
            base_reps=reps,
            labels=labels, groups=groups,
            metric='auc',
            task_type='classification',
            budgets=FULL_BUDGETS if not dry_run else [FULL_BUDGETS[0]],
            augmentation_strategy='none',
            internal_evaluators=INTERNAL_EVALUATORS,
            speed='full' if not dry_run else 'fast',
            return_fitted=True, verbose=False, cache=True,
            dataset_name=dataset_tag, tier='general',
            adaptive_pruning=0.25 if not dry_run else False,
        )
        elapsed = time.time() - t0
        per_model = {}
        for m, rec in (search.best_by_model or {}).items():
            per_model[m] = {
                'budget': rec.get('budget'),
                'score_mean': rec.get('score_mean'),
                'score_std': rec.get('score_std'),
                'n_features': rec.get('n_features'),
            }
        out['hybrid'] = {
            'time_s': elapsed,
            'metric': 'auc',
            'per_model': per_model,
        }
        for m, r in per_model.items():
            print(f"    {m:10s} budget={r['budget']:>5}  AUC="
                  f"{r['score_mean']:.4f}+/-{r['score_std']:.3f}  "
                  f"dim={r['n_features']}")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        out['hybrid'] = {'error': str(exc)[:300]}

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target-species', type=str, required=True,
                    choices=['dog', 'cow'])
    ap.add_argument('--reps', type=str, default=None,
                    help='Comma-separated rep names (default = DEFAULT_REPS).')
    ap.add_argument('--n-folds', type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--output', type=str, default=None)
    args = ap.parse_args()

    rep_names = args.reps.split(',') if args.reps else DEFAULT_REPS

    print('=' * 70)
    print(f"x_ab_ness  (T-Ab.4)  target={args.target_species}  "
          f"reps={len(rep_names)}  dry_run={args.dry_run}")
    print('=' * 70)

    res = evaluate_target(
        target_species=args.target_species,
        rep_names=rep_names,
        n_folds_requested=args.n_folds,
        dry_run=args.dry_run,
    )

    suffix = '_dry' if args.dry_run else ''
    out_default = (f'tests/results/cross_species/'
                   f'x_ab_{args.target_species}ness{suffix}.json')
    out = Path(args.output or out_default)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(convert_numpy(res), indent=2))
    print(f"\nResults written: {out}")


if __name__ == '__main__':
    main()
