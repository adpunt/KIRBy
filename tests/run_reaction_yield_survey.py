"""Reaction Yield Baseline Survey.

Compares reaction representations on Buchwald-Hartwig, Suzuki-Miyaura, and
Doyle lab BH plates yield prediction benchmarks.

Representations (8 total):
  - DRFP (2048D differential reaction fingerprints)
  - Concat ECFP4 (per-component Morgan fingerprints concatenated)
  - Concat MACCS (per-component 167-bit MACCS keys)
  - Concat PDV (per-component RDKit descriptors)
  - Product ECFP4 (product-only Morgan fingerprint)
  - Reactant ECFP4 (OR-combined reactant Morgan fingerprints)
  - RXNFP (256D learned transformer fingerprints, IBM RXN)
  - One-Hot Components (factorial structure encoding)

Models: All DEFAULT_MODELS from find_best_hybrid (filterable via --models)

Evaluation:
  - BH: 5-fold GroupKFold on additive (23 groups) — leave-additive-out
  - Suzuki: 5-fold GroupKFold on ligand (12 groups) — leave-ligand-out
  - BH Doyle plates: 5-fold GroupKFold on additive (23 groups) — leave-additive-out

Datasets:
  buchwald_hartwig  — Ahneman et al. Science 2018 (~3955 reactions)
  suzuki            — Perera et al. Science 2018 (~5760 reactions)
  bh_doyle_plates   — Doyle lab HTE plates, doylelab/rxnpredict (~4599 reactions)
                      NOTE: different dataset from buchwald_hartwig despite same reaction type

Usage:
    python tests/run_reaction_yield_survey.py
    python tests/run_reaction_yield_survey.py --dataset buchwald_hartwig
    python tests/run_reaction_yield_survey.py --dataset bh_doyle_plates
    python tests/run_reaction_yield_survey.py --dataset suzuki --n-folds 5
    python tests/run_reaction_yield_survey.py --random-splits  # use old random 70/30
"""

import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

DATASETS = ['buchwald_hartwig', 'suzuki', 'bh_doyle_plates']


def load_data(dataset_name):
    from kirby.datasets.reaction_yields import (
        load_buchwald_hartwig, load_suzuki, load_bh_doyle_plates,
    )
    if dataset_name == 'buchwald_hartwig':
        return load_buchwald_hartwig()
    elif dataset_name == 'suzuki':
        return load_suzuki()
    elif dataset_name == 'bh_doyle_plates':
        return load_bh_doyle_plates()
    raise ValueError(f"Unknown dataset: {dataset_name}. "
                     f"Available: {DATASETS}")


def generate_features(rep_name, data):
    from kirby.representations.reaction import (
        create_drfp, create_concat_ecfp4, create_concat_maccs,
        create_concat_pdv, create_product_ecfp4, create_reactant_ecfp4,
        create_rxnfp, create_onehot_components,
    )

    if rep_name == 'drfp_2048':
        return create_drfp(data['rxn_smiles'])
    elif rep_name == 'concat_ecfp4':
        return create_concat_ecfp4(data['components'])
    elif rep_name == 'concat_maccs':
        return create_concat_maccs(data['components'])
    elif rep_name == 'concat_pdv':
        return create_concat_pdv(data['components'])
    elif rep_name == 'product_ecfp4':
        return create_product_ecfp4(data['rxn_smiles'])
    elif rep_name == 'reactant_ecfp4':
        return create_reactant_ecfp4(data['rxn_smiles'])
    elif rep_name == 'rxnfp':
        return create_rxnfp(data['rxn_smiles'])
    elif rep_name == 'onehot_components':
        return create_onehot_components(data['components'])
    return None


def build_models():
    from copy import deepcopy
    from kirby.find_best_hybrid import get_model

    _model_names = ['ridge', 'rf', 'xgboost', 'lightgbm',
                     'gp', 'mlp', 'dnn', 'cnn', 'catboost']
    models = {}
    for name in _model_names:
        try:
            m = get_model(name, task_type='regression')
        except Exception:
            continue
        if m is None:
            continue
        # Reaction yield surveys use n_estimators=500 for tree models
        if hasattr(m, 'n_estimators'):
            m.set_params(n_estimators=500)
        # Use canonical key names (xgboost -> xgb for backwards compat)
        key = 'xgb' if name == 'xgboost' else name
        models[key] = lambda _m=m: deepcopy(_m)

    return models


def evaluate_split(X, labels, train_idx, test_idx, model_factory):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = model_factory()
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"    WARNING: model fit/predict failed: {e}")
        return {
            'r2': float('nan'), 'rmse': float('nan'),
            'mae': float('nan'), 'spearman': float('nan'),
        }

    return {
        'r2': float(r2_score(y_test, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'spearman': float(spearmanr(y_test, y_pred)[0]),
    }


def _make_splits(data, n_folds, use_random):
    """Create train/test splits: GroupKFold on groups if available, else random."""
    groups = data.get('groups')
    labels = data['labels']

    if groups is not None and not use_random:
        n_unique = len(np.unique(groups))
        actual_folds = min(n_folds, n_unique)
        cv = GroupKFold(n_splits=actual_folds)
        splits = list(cv.split(np.zeros(len(labels)), labels, groups=groups))
        group_key = data.get('metadata', {}).get('group_key', 'unknown')
        print(f"  Splitting: {actual_folds}-fold GroupKFold on {group_key} "
              f"({n_unique} groups)")
        return splits

    # Fallback: pre-computed random splits
    splits = data.get('splits', [])
    if splits:
        print(f"  Splitting: random 70/30 x {len(splits)}")
        return splits

    # Last resort: generate random splits
    n = len(labels)
    n_train = int(0.7 * n)
    splits = []
    for i in range(n_folds):
        rng = np.random.RandomState(42 + i)
        perm = rng.permutation(n)
        splits.append((perm[:n_train], perm[n_train:]))
    print(f"  Splitting: random 70/30 x {n_folds} (generated)")
    return splits


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dataset', choices=DATASETS, default=None,
                        help='Dataset to evaluate (default: all). '
                             'bh_doyle_plates is a DIFFERENT dataset from '
                             'buchwald_hartwig despite both being BH C-N coupling.')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--random-splits', action='store_true',
                        help='Use old random 70/30 splits instead of GroupKFold')
    parser.add_argument('--output', default='tests/results/reaction_yields',
                        help='Output directory')
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else DATASETS
    models = build_models()

    rep_configs = {
        'drfp_2048': 'DRFP (2048D)',
        'concat_ecfp4': 'Concat ECFP4',
        'concat_maccs': 'Concat MACCS',
        'concat_pdv': 'Concat PDV',
        'product_ecfp4': 'Product ECFP4',
        'reactant_ecfp4': 'Reactant ECFP4',
        'rxnfp': 'RXNFP (256D)',
        'onehot_components': 'One-Hot Components',
    }

    all_results = {}

    for ds_name in datasets:
        print(f"\n{'=' * 70}")
        print(f"Dataset: {ds_name}")
        print('=' * 70)

        data = load_data(ds_name)
        splits = _make_splits(data, args.n_folds, args.random_splits)

        for rep_name, rep_label in rep_configs.items():
            print(f"\n  Representation: {rep_label}")
            t0 = time.time()

            try:
                X = generate_features(rep_name, data)
            except ImportError as e:
                print(f"    Skipped ({e})")
                continue

            if X is None:
                print(f"    Skipped (not available for {ds_name})")
                continue

            print(f"    Shape: {X.shape}, generation time: {time.time() - t0:.1f}s")

            for model_name, model_factory in models.items():
                split_results = []
                for train_idx, test_idx in splits:
                    res = evaluate_split(
                        X, data['labels'], train_idx, test_idx, model_factory,
                    )
                    split_results.append(res)

                avg = {
                    'r2': float(np.mean([r['r2'] for r in split_results])),
                    'r2_std': float(np.std([r['r2'] for r in split_results])),
                    'rmse': float(np.mean([r['rmse'] for r in split_results])),
                    'rmse_std': float(np.std([r['rmse'] for r in split_results])),
                    'mae': float(np.mean([r['mae'] for r in split_results])),
                    'mae_std': float(np.std([r['mae'] for r in split_results])),
                    'spearman': float(np.mean([r['spearman'] for r in split_results])),
                    'spearman_std': float(np.std([r['spearman'] for r in split_results])),
                }

                key = f"{ds_name}/{rep_name}/{model_name}"
                all_results[key] = avg

                print(f"    {model_name:>8}: R²={avg['r2']:.4f}±{avg['r2_std']:.4f}  "
                      f"RMSE={avg['rmse']:.2f}±{avg['rmse_std']:.2f}  "
                      f"MAE={avg['mae']:.2f}±{avg['mae_std']:.2f}  "
                      f"ρ={avg['spearman']:.4f}±{avg['spearman_std']:.4f}")

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary")
    print('=' * 70)
    print(f"{'Dataset/Rep/Model':<45} {'R²':>14} {'RMSE':>14} {'MAE':>14} {'ρ':>14}")
    print('-' * 105)
    for key, r in sorted(all_results.items()):
        print(f"{key:<45} {r['r2']:.4f}±{r['r2_std']:.4f}  "
              f"{r['rmse']:>5.2f}±{r['rmse_std']:.2f}  "
              f"{r['mae']:>5.2f}±{r['mae_std']:.2f}  "
              f"{r['spearman']:.4f}±{r['spearman_std']:.4f}")

    # Save
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'reaction_yield_survey.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[saved {out_file}]")


if __name__ == '__main__':
    main()
