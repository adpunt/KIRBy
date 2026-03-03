#!/usr/bin/env python3
"""
Universal Baseline Survey

For every (rep x dataset x model) cell, compute baseline R²/accuracy with
both default and tuned hyperparameters. No hybrids. 3-fold GroupKFold.

Usage:
    python tests/run_baseline_survey.py logd
    python tests/run_baseline_survey.py logd --reps fast
    python tests/run_baseline_survey.py molecular              # all molecular datasets
    python tests/run_baseline_survey.py everything             # all domains
    python tests/run_baseline_survey.py tcga --finetune        # include FT reps (GPU)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Block stale system bottleneck (Anaconda 2022.05)
sys.modules.setdefault('bottleneck', None)

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

# Path setup
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from experiments._shared import convert_numpy

warnings.filterwarnings('ignore')


# =============================================================================
# DATASET REGISTRY
# =============================================================================

# {dataset_name: (domain, default_rep_preset)}
DATASET_REGISTRY = {
    # Molecular
    'logd':  ('molecular', 'all'),
    'caco2': ('molecular', 'all'),
    'herg':  ('molecular', 'all'),
    'qm9':   ('molecular', 'all'),
    # Antibody DMS (intra-family)
    'flab_expression_koenig': ('ab_dms', 'dms'),
    'flab_binding_koenig':    ('ab_dms', 'dms'),
    'abagym_Ang2':            ('ab_dms', 'dms'),
    'abagym_EGFR':            ('ab_dms', 'dms'),
    'abagym_lysozyme':        ('ab_dms', 'dms'),
    'abagym_VEGF':            ('ab_dms', 'dms'),
    'abagym_HER2':            ('ab_dms', 'dms'),
    # Antibody inter-family
    'ddg_combined':          ('ab_inter', 'inter_family'),
    'abagym_all':            ('ab_inter', 'inter_family'),
    'abcov_kd':              ('ab_inter', 'inter_family'),
    'flab_thermo_jain':      ('ab_inter', 'inter_family'),
    'flab_aggregation_jain': ('ab_inter', 'inter_family'),
    # TCGA
    'tcga_luad_vs_lusc':  ('tcga', 'genomic_fast'),
    'tcga_brca_pam50':    ('tcga', 'genomic_fast'),
    'tcga_brca_survival': ('tcga', 'genomic_fast'),
    # DTA
    'davis': ('dta', 'fast'),
    'kiba':  ('dta', 'fast'),
}

# Build domain -> datasets mapping
DOMAIN_DATASETS = {}
for _ds, (_dom, _) in DATASET_REGISTRY.items():
    DOMAIN_DATASETS.setdefault(_dom, []).append(_ds)

SURVEY_MODELS = ['rf', 'xgboost', 'ridge', 'knn', 'lightgbm']


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset_and_reps(dataset_name: str, rep_preset: Optional[str] = None) -> dict:
    """Load dataset + generate representations. Returns standard dict."""
    domain, default_preset = DATASET_REGISTRY[dataset_name]
    preset = rep_preset or default_preset

    if domain == 'dta':
        return _load_dta(dataset_name, preset)

    # Molecular, antibody, TCGA — handled by _shared.load_data
    from experiments._shared import load_data
    return load_data(dataset_name, rep_preset=preset)


def _load_dta(dataset_name: str, rep_preset: str) -> dict:
    """Load DTA dataset (not handled by _shared.load_data)."""
    from find_best_hybrid_dta import (
        load_dataset as _load_dta_ds,
        get_representations as _get_dta_reps,
    )
    from kirby.datasets.dta import assign_cold_target_groups

    data = _load_dta_ds(dataset_name)
    drug_smiles = data['drug_smiles']
    protein_seqs = data['protein_seqs']
    labels = data['labels']
    task_type = data['task_type']
    metadata = data['metadata']
    protein_ids = metadata['protein_ids']

    # Subsample KIBA (117K pairs too large for baseline survey)
    if len(labels) > 30000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(labels), 30000, replace=False)
        idx.sort()
        drug_smiles = [drug_smiles[i] for i in idx]
        protein_seqs = [protein_seqs[i] for i in idx]
        labels = labels[idx]
        protein_ids = [protein_ids[i] for i in idx]
        print(f"  Subsampled to 30000 pairs")

    reps = _get_dta_reps(drug_smiles, protein_seqs, rep_preset)
    groups, n_groups = assign_cold_target_groups(protein_ids)

    print(f"  {dataset_name}: {len(labels)} pairs, {n_groups} cold-target groups, "
          f"{len(reps)} reps")

    return {
        'labels': labels,
        'reps': reps,
        'groups': groups,
        'n_groups': n_groups,
        'task_type': task_type,
        'domain': 'dta',
        'augmentations': None,
    }


# =============================================================================
# MODEL FACTORIES
# =============================================================================

def get_default_models(task_type: str) -> Dict[str, Any]:
    """Return default model instances for the 5 survey models."""
    from kirby.find_best_hybrid import _resolve_models
    # speed='long' includes all tiers (rf/xgb=fast, ridge/knn=medium, lgbm=long)
    models = _resolve_models(SURVEY_MODELS, task_type, 'long')
    # Set n_jobs=1 for fork safety (avoids SIGSEGV on macOS with LightGBM).
    # Parallelism happens at the SLURM job level, not within models.
    for m in models.values():
        if hasattr(m, 'n_jobs'):
            m.n_jobs = 1
    return models


# =============================================================================
# EVALUATION
# =============================================================================

def _make_cv(groups, n_folds, labels):
    """Create CV splitter and split kwargs."""
    if groups is not None:
        n_unique = len(np.unique(groups))
        actual_folds = min(n_folds, n_unique)
        # Classification: try StratifiedGroupKFold for class balance
        if labels is not None and np.issubdtype(np.asarray(labels).dtype, np.integer):
            try:
                from sklearn.model_selection import StratifiedGroupKFold
                return (
                    StratifiedGroupKFold(
                        n_splits=actual_folds, shuffle=True, random_state=42),
                    {'groups': groups},
                )
            except (ImportError, Exception):
                pass
        return GroupKFold(n_splits=actual_folds), {'groups': groups}
    return KFold(n_splits=n_folds, shuffle=True, random_state=42), {}


def _score(y_true, y_pred, task_type):
    """Compute appropriate metric."""
    if task_type == 'classification':
        return accuracy_score(y_true, y_pred)
    return r2_score(y_true, y_pred)


def evaluate_rep(X, labels, groups, model, n_folds, task_type):
    """Evaluate one (rep, model) pair via CV. Returns (mean, std)."""
    cv, cv_kwargs = _make_cv(groups, n_folds, labels)
    scores = []

    for train_idx, val_idx in cv.split(X, labels, **cv_kwargs):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        try:
            m = clone(model)
            m.fit(X_train, labels[train_idx])
            scores.append(_score(labels[val_idx], m.predict(X_val), task_type))
        except Exception:
            continue

    if not scores:
        return float('nan'), float('nan')
    return float(np.mean(scores)), float(np.std(scores))


def evaluate_ft_rep(factory, labels, groups, model, n_folds, task_type):
    """Evaluate a fine-tuning factory via CV. Returns (mean, std)."""
    cv, cv_kwargs = _make_cv(groups, n_folds, labels)
    scores = []
    dummy_X = np.zeros((len(labels), 1))

    for train_idx, val_idx in cv.split(dummy_X, labels, **cv_kwargs):
        try:
            X_train, X_val = factory(train_idx, val_idx, labels)
        except Exception as e:
            print(f" [ft error: {e}]", end='')
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        try:
            m = clone(model)
            m.fit(X_train, labels[train_idx])
            scores.append(_score(labels[val_idx], m.predict(X_val), task_type))
        except Exception:
            continue

    if not scores:
        return float('nan'), float('nan')
    return float(np.mean(scores)), float(np.std(scores))


# =============================================================================
# FINE-TUNING FACTORY CREATION (TCGA only)
# =============================================================================

def make_ft_factories(data, dataset_name, task_type, ft_epochs=3, ft_cache=True):
    """Create fine-tuning factories for TCGA datasets. Returns {} for non-TCGA."""
    domain = data.get('domain', '')
    if domain != 'tcga':
        return {}

    expression = data.get('expression')
    metadata = data.get('metadata', {})
    gene_names = metadata.get('gene_names')
    sample_ids = metadata.get('sample_ids')

    if expression is None or gene_names is None:
        return {}

    factories = {}

    try:
        from kirby.finetuning.tcga_geneformer import make_geneformer_factory
        factories['geneformer'] = make_geneformer_factory(
            expression_data=expression, gene_names=gene_names,
            sample_ids=sample_ids, dataset_name=dataset_name,
            task_type=task_type, epochs=ft_epochs, cache=ft_cache,
        )
    except Exception as e:
        print(f"  geneformer factory failed: {e}")

    try:
        from kirby.finetuning.tcga_scfoundation import make_scfoundation_factory
        factories['scfoundation'] = make_scfoundation_factory(
            expression_data=expression, gene_names=gene_names,
            sample_ids=sample_ids, dataset_name=dataset_name,
            task_type=task_type, epochs=ft_epochs, cache=ft_cache,
        )
    except Exception as e:
        print(f"  scfoundation factory failed: {e}")

    return factories


# =============================================================================
# BEST-SCORE HELPERS
# =============================================================================

def _add_bests(entry):
    """Add best_default, best_tuned, best_overall to a rep entry."""
    if entry['default']:
        best_def = max(entry['default'].items(), key=lambda x: x[1])
        entry['best_default'] = {'model': best_def[0], 'score': best_def[1]}
    else:
        entry['best_default'] = None

    if entry['tuned']:
        best_tun = max(entry['tuned'].items(), key=lambda x: x[1])
        entry['best_tuned'] = {'model': best_tun[0], 'score': best_tun[1]}
    else:
        entry['best_tuned'] = None

    # Best overall across default + tuned
    all_scores = {}
    for m, s in entry['default'].items():
        all_scores[(m, 'default')] = s
    for m, s in entry['tuned'].items():
        all_scores[(m, 'tuned')] = s

    if all_scores:
        best_key = max(all_scores, key=all_scores.get)
        entry['best_overall'] = {
            'model': best_key[0],
            'score': all_scores[best_key],
            'source': best_key[1],
        }
    else:
        entry['best_overall'] = {'model': None, 'score': float('nan'), 'source': None}


# =============================================================================
# MAIN SURVEY LOOP
# =============================================================================

def run_survey(dataset_name, rep_preset, n_folds, do_tune, tune_iter,
               do_finetune, seed):
    """Run the full baseline survey for one dataset. Returns result dict."""
    print(f"\n{'='*70}")
    print(f"SURVEY: {dataset_name}")
    print(f"{'='*70}")

    t0 = time.time()

    # 1. Load data + reps
    data = load_dataset_and_reps(dataset_name, rep_preset)
    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    task_type = data.get('task_type', 'regression')
    n_groups = len(np.unique(groups))
    metric_name = 'accuracy' if task_type == 'classification' else 'r2'

    # 2. Default models
    default_models = get_default_models(task_type)
    model_names = list(default_models.keys())

    # 3. Tune models (on concatenated reps)
    tuned_models = None
    tuned_params = {}
    if do_tune:
        from kirby.find_best_hybrid import tune_models
        print(f"\n--- Tuning models ({tune_iter} iters, {n_folds}-fold) ---")
        tuned_models = tune_models(
            base_reps=reps, labels=labels, groups=groups,
            task_type=task_type, speed='long',
            n_iter=tune_iter, n_folds=n_folds, seed=seed,
        )
        # Keep only the 5 survey models (tune_models returns all in the tier)
        tuned_models = {k: v for k, v in tuned_models.items()
                        if k in SURVEY_MODELS}
        # Set n_jobs=1 for fork safety
        for m in tuned_models.values():
            if hasattr(m, 'n_jobs'):
                m.n_jobs = 1
        for name, model in tuned_models.items():
            params = model.get_params()
            tuned_params[name] = {
                k: v for k, v in params.items()
                if k not in ('verbose', 'verbosity', 'n_jobs', 'random_state',
                             'callbacks', 'class_weight')
            }

    # 4. Evaluate each frozen rep
    baselines = {}
    print(f"\n--- Evaluating {len(reps)} reps x {len(model_names)} models ---")

    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
        dim = X.shape[1]
        print(f"  {rep_name} ({dim}d):", end='', flush=True)

        entry = {'dimension': dim, 'default': {}, 'tuned': {}}

        # Default models
        for mname in model_names:
            mean_s, _ = evaluate_rep(
                X, labels, groups, default_models[mname], n_folds, task_type)
            entry['default'][mname] = round(mean_s, 4)
            print(f" {mname}={mean_s:.3f}", end='', flush=True)

        # Tuned models
        if tuned_models:
            print(" | tuned:", end='', flush=True)
            for mname in model_names:
                if mname in tuned_models:
                    mean_s, _ = evaluate_rep(
                        X, labels, groups, tuned_models[mname], n_folds, task_type)
                    entry['tuned'][mname] = round(mean_s, 4)
                    print(f" {mname}={mean_s:.3f}", end='', flush=True)

        _add_bests(entry)
        baselines[rep_name] = entry
        print()

    # 5. Fine-tuned reps (TCGA only)
    ft_baselines = {}
    if do_finetune:
        ft_factories = make_ft_factories(data, dataset_name, task_type)
        if ft_factories:
            print(f"\n--- Fine-tuned reps: {list(ft_factories.keys())} ---")
            for ft_name, factory in ft_factories.items():
                print(f"  {ft_name}:", end='', flush=True)
                entry = {'dimension': '?', 'default': {}, 'tuned': {}}

                for mname in model_names:
                    mean_s, _ = evaluate_ft_rep(
                        factory, labels, groups, default_models[mname],
                        n_folds, task_type)
                    entry['default'][mname] = round(mean_s, 4)
                    print(f" {mname}={mean_s:.3f}", end='', flush=True)

                if tuned_models:
                    print(" | tuned:", end='', flush=True)
                    for mname in model_names:
                        if mname in tuned_models:
                            mean_s, _ = evaluate_ft_rep(
                                factory, labels, groups, tuned_models[mname],
                                n_folds, task_type)
                            entry['tuned'][mname] = round(mean_s, 4)
                            print(f" {mname}={mean_s:.3f}", end='', flush=True)

                _add_bests(entry)
                ft_baselines[ft_name] = entry
                print()

    elapsed = time.time() - t0

    # 6. Summary
    all_entries = {**baselines, **ft_baselines}
    all_bests = [(name, e['best_overall']) for name, e in all_entries.items()
                 if e['best_overall'] and e['best_overall']['score'] is not None]
    if all_bests:
        overall = max(all_bests, key=lambda x: x[1]['score'])
        summary = {
            'best_rep': overall[0],
            'best_score': overall[1]['score'],
            'best_model': overall[1]['model'],
            'source': overall[1]['source'],
        }
    else:
        summary = {'best_rep': None, 'best_score': None, 'best_model': None,
                   'source': None}

    result = {
        'metadata': {
            'dataset': dataset_name,
            'domain': data.get('domain', ''),
            'n_samples': len(labels),
            'n_groups': n_groups,
            'n_folds': n_folds,
            'n_reps': len(reps),
            'rep_names': sorted(reps.keys()),
            'models': model_names,
            'metric': metric_name,
            'task_type': task_type,
            'tuned': do_tune,
            'tune_iter': tune_iter if do_tune else None,
            'seed': seed,
            'elapsed_seconds': round(elapsed, 1),
            'timestamp': datetime.now().isoformat(),
        },
        'baselines': baselines,
        'tuned_params': tuned_params or None,
        'finetuned_baselines': ft_baselines or None,
        'summary': summary,
    }

    return result


# =============================================================================
# OUTPUT
# =============================================================================

def print_summary_table(result):
    """Print compact summary table."""
    meta = result['metadata']
    baselines = result['baselines']
    summary = result['summary']
    models = meta['models']
    metric = meta['metric']

    print(f"\n{'='*70}")
    print(f"SUMMARY: {meta['dataset']} ({meta['n_samples']} samples, "
          f"{meta['n_groups']} groups, {metric})")
    print(f"{'='*70}")

    # Header
    header = f"{'Rep':<25} {'Dim':>5}"
    for m in models:
        header += f" {m:>8}"
    header += f"  {'Best':>8}"
    print(header)
    print("-" * len(header))

    # Default scores
    for rep_name in sorted(baselines.keys()):
        r = baselines[rep_name]
        row = f"{rep_name:<25} {r['dimension']:>5}"
        for m in models:
            s = r['default'].get(m, float('nan'))
            row += f" {s:>8.4f}"
        best = r.get('best_default')
        if best:
            row += f"  {best['score']:>8.4f}"
        print(row)

    # Tuned scores
    if result.get('tuned_params'):
        print()
        print("TUNED:")
        print("-" * len(header))
        for rep_name in sorted(baselines.keys()):
            r = baselines[rep_name]
            if not r.get('tuned'):
                continue
            row = f"{rep_name:<25} {r['dimension']:>5}"
            for m in models:
                s = r['tuned'].get(m, float('nan'))
                row += f" {s:>8.4f}"
            best = r.get('best_tuned')
            if best:
                row += f"  {best['score']:>8.4f}"
            print(row)

    # Fine-tuned scores
    ft = result.get('finetuned_baselines')
    if ft:
        print()
        print("FINE-TUNED:")
        print("-" * len(header))
        for ft_name in sorted(ft.keys()):
            r = ft[ft_name]
            row = f"{ft_name:<25} {'?':>5}"
            for m in models:
                s = r['default'].get(m, float('nan'))
                row += f" {s:>8.4f}"
            best = r.get('best_default')
            if best:
                row += f"  {best['score']:>8.4f}"
            print(row)

    # Winner
    print()
    if summary.get('best_rep'):
        print(f"WINNER: {summary['best_rep']} + {summary['best_model']} = "
              f"{summary['best_score']:.4f} ({metric}, {summary['source']})")
    print(f"  Time: {meta['elapsed_seconds']:.0f}s")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Universal baseline survey: every (rep x dataset x model) cell.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Datasets:
  logd, caco2, herg, qm9                          (molecular)
  flab_expression_koenig, flab_binding_koenig,     (antibody DMS)
    abagym_Ang2/EGFR/lysozyme/VEGF/HER2
  ddg_combined, abagym_all, abcov_kd,             (antibody inter-family)
    flab_thermo_jain, flab_aggregation_jain
  tcga_luad_vs_lusc, tcga_brca_pam50,             (TCGA)
    tcga_brca_survival
  davis, kiba                                      (DTA)

Shortcuts:
  molecular, ab_dms, ab_inter, tcga, dta           (run all in domain)
  everything                                       (run all domains)

Examples:
  %(prog)s logd --reps fast
  %(prog)s molecular --tune --output tests/results/universal_survey/
  %(prog)s everything --tune
  %(prog)s tcga --finetune --tune
        """
    )

    parser.add_argument('dataset', type=str,
                       help='Dataset name, domain shortcut, or "everything"')
    parser.add_argument('--reps', type=str, default=None,
                       help='Override rep preset (default: per-dataset from registry)')
    parser.add_argument('--n-folds', type=int, default=3,
                       help='Number of CV folds (default: 3)')
    parser.add_argument('--tune', action='store_true',
                       help='Also evaluate with tuned HPs (RandomizedSearchCV)')
    parser.add_argument('--tune-iter', type=int, default=20,
                       help='RandomizedSearchCV iterations per model (default: 20)')
    parser.add_argument('--finetune', action='store_true',
                       help='Include fine-tuned reps (TCGA: geneformer, scfoundation)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (JSON file or directory for multi-dataset)')

    args = parser.parse_args()

    # Resolve dataset list
    ds_input = args.dataset
    if ds_input == 'everything':
        datasets = list(DATASET_REGISTRY.keys())
    elif ds_input in DOMAIN_DATASETS:
        datasets = DOMAIN_DATASETS[ds_input]
    elif ds_input in DATASET_REGISTRY:
        datasets = [ds_input]
    else:
        print(f"ERROR: Unknown dataset '{ds_input}'")
        print(f"Available: {sorted(DATASET_REGISTRY.keys())}")
        print(f"Domains: {sorted(DOMAIN_DATASETS.keys())}, everything")
        return 1

    print(f"Universal Baseline Survey")
    print(f"  Datasets: {len(datasets)} ({', '.join(datasets)})")
    print(f"  Folds: {args.n_folds}, Tune: {args.tune}, Finetune: {args.finetune}")

    # Output handling
    output_dir = None
    output_file = None
    if args.output:
        p = Path(args.output)
        if p.suffix == '.json':
            output_file = p
            output_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = p
            output_dir.mkdir(parents=True, exist_ok=True)

    # Run surveys
    all_results = {}
    pass_count = 0
    fail_count = 0

    for ds in datasets:
        try:
            result = run_survey(
                ds, args.reps, args.n_folds, args.tune, args.tune_iter,
                args.finetune, args.seed,
            )
            result = convert_numpy(result)
            print_summary_table(result)
            all_results[ds] = result
            pass_count += 1

            # Save per-dataset JSON
            if output_dir:
                out_path = output_dir / f"{ds}.json"
                with open(out_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"  Saved: {out_path}")

        except Exception as e:
            print(f"\n  FAILED: {ds}: {e}")
            traceback.print_exc()
            fail_count += 1

    # Save single-file output
    if output_file and all_results:
        payload = (list(all_results.values())[0]
                   if len(all_results) == 1 else all_results)
        with open(output_file, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved: {output_file}")

    # Cross-dataset summary
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("CROSS-DATASET SUMMARY")
        print(f"{'='*70}")
        for ds, r in all_results.items():
            s = r['summary']
            m = r['metadata']
            if s.get('best_rep'):
                print(f"  {ds:<30} {s['best_rep']:<20} + {s['best_model']:<10} = "
                      f"{s['best_score']:.4f} ({m['metric']})")
            else:
                print(f"  {ds:<30} FAILED")

    print(f"\nDone: {pass_count} passed, {fail_count} failed")
    return 1 if fail_count > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
