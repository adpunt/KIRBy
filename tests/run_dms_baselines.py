#!/usr/bin/env python3
"""
DMS Per-Rep Baseline Survey — R² matrix for Bill of Materials pruning.

Evaluates each rep individually on each DMS dataset using GroupKFold CV.
Outputs a JSON matrix: {dataset: {rep: {model: {r2_mean, r2_std}}}}.

This is simpler than the full hybrid search — it tests each rep in isolation
to identify which reps are useful and which should be pruned from presets.

9 DMS datasets × 20 reps × 3 models = 540 cells.

Usage:
    # Single dataset, subset of reps (local test)
    python tests/run_dms_baselines.py flab_expression_koenig --reps aac,blosum62,biophysical_delta

    # Single dataset, all reps
    python tests/run_dms_baselines.py flab_expression_koenig

    # All 9 datasets, all reps (server)
    python tests/run_dms_baselines.py all

    # All datasets, specific reps
    python tests/run_dms_baselines.py all --reps aac,blosum62,varpos_cdr_physchem
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
warnings.filterwarnings('ignore')


# ── Datasets ────────────────────────────────────────────────────────────────

DMS_DATASETS = [
    'flab_expression_koenig',
    'flab_binding_koenig',
    'flab_thermo_hie',
    'flab_binding_rosace',
    'abagym_Ang2_2017_G6',
    'abagym_EGFR_2013_Cetuximab',
    'abagym_lysozyme_2019_D441',
    'abagym_VEGF_2017b_G6',
    'abagym_HER2_2021_trastuzumab',
]

# All reps in dms_full preset (20 total)
DMS_FULL_REPS = [
    'lm_score_ablang2', 'lm_score_esm2', 'lm_score_prot_t5', 'lm_score_antiberty',
    'positional', 'biophysical_delta',
    'esm2_site',
    'aac', 'blosum62', 'developability',
    'varpos_onehot', 'varpos_blosum', 'varpos_physchem',
    'varpos_cdr_onehot', 'varpos_cdr_physchem',
    'varpos_cdrh3_physchem', 'varpos_fw_physchem',
    'antifold_score', 'sapiens_score', 'esm1v_score',
]


# ── Data loading ────────────────────────────────────────────────────────────

def load_dataset(name: str):
    """Load a DMS dataset via find_best_hybrid_antibody's loader."""
    # Import inline to keep startup fast
    sys.path.insert(0, str(Path(__file__).parent))
    from find_best_hybrid_antibody import load_dataset as _load
    return _load(name)


# ── Rep generation ──────────────────────────────────────────────────────────

def generate_single_rep(rep_name: str, heavy_seqs, light_seqs, wt_heavy, wt_light,
                        mutations, cache: bool = True) -> Optional[np.ndarray]:
    """Generate a single rep using the dispatch table from find_best_hybrid_antibody."""
    sys.path.insert(0, str(Path(__file__).parent))
    from find_best_hybrid_antibody import get_representations

    try:
        reps = get_representations(
            heavy_seqs, light_seqs,
            preset_or_list=rep_name,  # single rep name
            cache=cache,
            wt_heavy=wt_heavy,
            wt_light=wt_light,
            mutations=mutations,
        )
        if rep_name in reps:
            return reps[rep_name]
        return None
    except Exception as e:
        print(f"    Rep generation failed: {e}")
        return None


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_rep(X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray],
                 n_folds: int = 5) -> Dict[str, Dict[str, float]]:
    """Train rf, ridge, xgboost on a single rep and return R² scores.

    Returns:
        {model_name: {r2_mean, r2_std, r2_folds}}
    """
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    try:
        from xgboost import XGBRegressor
        has_xgb = True
    except ImportError:
        has_xgb = False

    # Handle NaN/inf in features
    if np.any(~np.isfinite(X)):
        finite_cols = np.all(np.isfinite(X), axis=0)
        if finite_cols.sum() == 0:
            return {}
        X = X[:, finite_cols]

    # Choose CV strategy
    if groups is not None:
        n_unique = len(np.unique(groups))
        actual_folds = min(n_folds, n_unique)
        if actual_folds < 2:
            # Not enough groups for CV
            return {}
        cv = GroupKFold(n_splits=actual_folds)
        split_args = (X, y, groups)
    else:
        actual_folds = n_folds
        cv = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
        split_args = (X, y)

    models = {
        'rf': lambda: RandomForestRegressor(n_estimators=100, max_depth=10,
                                            n_jobs=-1, random_state=42),
        'ridge': lambda: Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0)),
        ]),
    }
    if has_xgb:
        models['xgboost'] = lambda: XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            n_jobs=-1, random_state=42, verbosity=0,
        )

    results = {}
    for model_name, model_fn in models.items():
        fold_scores = []
        for train_idx, test_idx in cv.split(*split_args):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = model_fn()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # R² score
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - y_test.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            fold_scores.append(r2)

        results[model_name] = {
            'r2_mean': float(np.mean(fold_scores)),
            'r2_std': float(np.std(fold_scores)),
            'r2_folds': [float(s) for s in fold_scores],
        }

    return results


# ── Identity grouping ───────────────────────────────────────────────────────

def assign_identity_groups(heavy_sequences: List[str], n_folds: int = 5):
    """Group by VH sequence identity for GroupKFold (reuse from hybrid script)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from find_best_hybrid_antibody import assign_identity_groups as _aig
    groups, n_groups = _aig(heavy_sequences, threshold=0.8, min_groups=n_folds)
    return groups


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='DMS Per-Rep Baseline Survey')
    parser.add_argument('dataset', type=str,
                        help='Dataset name (e.g. flab_expression_koenig) or "all"')
    parser.add_argument('--reps', type=str, default=None,
                        help='Comma-separated rep names (default: all dms_full reps)')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: auto-generated)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable rep caching')
    args = parser.parse_args()

    # Resolve datasets
    if args.dataset.lower() == 'all':
        datasets = DMS_DATASETS
    else:
        datasets = [args.dataset]

    # Resolve reps
    if args.reps:
        rep_names = [r.strip() for r in args.reps.split(',')]
    else:
        rep_names = DMS_FULL_REPS

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(__file__).parent / 'results' / 'dms_baselines'
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = args.dataset.lower().replace(' ', '_')
        output_path = output_dir / f'dms_baselines_{tag}_{timestamp}.json'

    print("=" * 72)
    print(f"DMS PER-REP BASELINE SURVEY")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Reps:     {len(rep_names)}")
    print(f"  Folds:    {args.folds}")
    print(f"  Output:   {output_path}")
    print(f"  Started:  {datetime.now().isoformat()}")
    print("=" * 72)

    all_results = {}
    total_cells = 0
    total_pass = 0
    total_fail = 0
    t_start = time.time()

    for ds_name in datasets:
        print(f"\n{'━' * 60}")
        print(f"Dataset: {ds_name}")
        print(f"{'━' * 60}")

        # Load dataset
        try:
            t0 = time.time()
            heavy_seqs, light_seqs, labels, task_type, metadata = load_dataset(ds_name)
            n_samples = len(heavy_seqs)
            load_time = time.time() - t0
            print(f"  Loaded: {n_samples} samples ({load_time:.1f}s)")
        except Exception as e:
            print(f"  LOAD FAILED: {e}")
            all_results[ds_name] = {'_error': str(e)}
            continue

        wt_heavy = metadata.get('wildtype_heavy')
        wt_light = metadata.get('wildtype_light')
        mutations = metadata.get('mutations')
        y = labels

        # Identity grouping for CV
        groups = assign_identity_groups(heavy_seqs, n_folds=args.folds)
        if groups is not None:
            n_groups = len(np.unique(groups))
            print(f"  Groups: {n_groups} identity clusters")
        else:
            print(f"  Groups: None (using ungrouped KFold)")

        ds_results = {}

        for rep_name in rep_names:
            total_cells += 1
            print(f"\n  Rep: {rep_name}")

            # Generate rep
            t0 = time.time()
            X = generate_single_rep(
                rep_name, heavy_seqs, light_seqs,
                wt_heavy, wt_light, mutations,
                cache=not args.no_cache,
            )
            gen_time = time.time() - t0

            if X is None:
                print(f"    SKIP (generation returned None)")
                ds_results[rep_name] = {'_error': 'generation_failed'}
                total_fail += 1
                continue

            print(f"    Shape: {X.shape}, generated in {gen_time:.1f}s")

            # Evaluate
            t0 = time.time()
            scores = evaluate_rep(X, y, groups, n_folds=args.folds)
            eval_time = time.time() - t0

            if not scores:
                print(f"    SKIP (evaluation failed)")
                ds_results[rep_name] = {'_error': 'evaluation_failed'}
                total_fail += 1
                continue

            ds_results[rep_name] = {
                'dim': int(X.shape[1]),
                'n_samples': int(X.shape[0]),
                'gen_time_s': round(gen_time, 1),
                'eval_time_s': round(eval_time, 1),
                'scores': scores,
            }
            total_pass += 1

            # Print summary for this rep
            best_model = max(scores, key=lambda m: scores[m]['r2_mean'])
            best_r2 = scores[best_model]['r2_mean']
            best_std = scores[best_model]['r2_std']
            print(f"    Best: {best_model}={best_r2:.3f}±{best_std:.3f}  "
                  f"({X.shape[1]}D, {eval_time:.1f}s)")

        all_results[ds_name] = ds_results

    total_time = time.time() - t_start

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        '_meta': {
            'timestamp': datetime.now().isoformat(),
            'datasets': datasets,
            'reps': rep_names,
            'folds': args.folds,
            'total_time_s': round(total_time, 1),
            'cells_pass': total_pass,
            'cells_fail': total_fail,
        }
    }
    meta.update(all_results)
    with open(output_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # ── Summary table ───────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"SUMMARY: {total_pass}/{total_cells} cells evaluated in {total_time:.0f}s")
    print(f"{'=' * 72}")

    # Print best R² per (dataset, rep) sorted by best
    print(f"\n{'Dataset':<35s} {'Rep':<28s} {'Best Model':>10s} {'R²':>8s} {'±':>6s} {'Dim':>5s}")
    print('─' * 95)

    summary_rows = []
    for ds_name in datasets:
        if ds_name not in all_results or isinstance(all_results[ds_name], dict) and '_error' in all_results[ds_name]:
            continue
        for rep_name in rep_names:
            if rep_name not in all_results[ds_name]:
                continue
            entry = all_results[ds_name][rep_name]
            if '_error' in entry:
                continue
            scores = entry['scores']
            best_model = max(scores, key=lambda m: scores[m]['r2_mean'])
            best_r2 = scores[best_model]['r2_mean']
            best_std = scores[best_model]['r2_std']
            summary_rows.append((ds_name, rep_name, best_model, best_r2, best_std, entry['dim']))

    # Sort by R² descending
    summary_rows.sort(key=lambda r: r[3], reverse=True)
    for ds, rep, model, r2, std, dim in summary_rows:
        ds_short = ds.replace('flab_', 'f_').replace('abagym_', 'a_')
        print(f"{ds_short:<35s} {rep:<28s} {model:>10s} {r2:>8.3f} {std:>6.3f} {dim:>5d}")

    # ── Pruning candidates ──────────────────────────────────────────────────
    if len(datasets) >= 3:
        print(f"\n{'=' * 72}")
        print("PRUNING ANALYSIS")
        print(f"{'=' * 72}")

        for rep_name in rep_names:
            scores_by_ds = []
            for ds_name in datasets:
                if ds_name not in all_results:
                    continue
                entry = all_results[ds_name].get(rep_name, {})
                if '_error' in entry:
                    continue
                scores = entry.get('scores', {})
                if scores:
                    best = max(scores[m]['r2_mean'] for m in scores)
                    scores_by_ds.append((ds_name, best))

            if not scores_by_ds:
                print(f"  {rep_name:<28s} — no data")
                continue

            best_any = max(s for _, s in scores_by_ds)
            mean_r2 = np.mean([s for _, s in scores_by_ds])
            n_positive = sum(1 for _, s in scores_by_ds if s > 0.01)
            dim = all_results[scores_by_ds[0][0]].get(rep_name, {}).get('dim', '?')

            flag = ''
            if best_any < 0.01:
                flag = ' ← DROP (never positive)'
            elif isinstance(dim, int) and dim > 1000 and best_any < 0.1:
                flag = ' ← DROP candidate (high dim, low signal)'

            print(f"  {rep_name:<28s} dim={str(dim):>5s}  "
                  f"best={best_any:>6.3f}  mean={mean_r2:>6.3f}  "
                  f"positive={n_positive}/{len(scores_by_ds)}{flag}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
