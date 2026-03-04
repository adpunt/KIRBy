"""Reaction Yield Baseline Survey.

Compares reaction representations on Buchwald-Hartwig and Suzuki-Miyaura
yield prediction benchmarks.

Representations:
  - DRFP (2048D differential reaction fingerprints)
  - Concat ECFP4 (per-component Morgan fingerprints concatenated)

Models: RidgeCV, Random Forest, XGBoost

Evaluation: 10 random 70/30 splits, R²/RMSE/MAE averaged.

Usage:
    python tests/run_reaction_yield_survey.py
    python tests/run_reaction_yield_survey.py --dataset buchwald_hartwig
    python tests/run_reaction_yield_survey.py --dataset suzuki --n-splits 5
"""

import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

DATASETS = ['buchwald_hartwig', 'suzuki']


def load_data(dataset_name):
    from kirby.datasets.reaction_yields import load_buchwald_hartwig, load_suzuki
    if dataset_name == 'buchwald_hartwig':
        return load_buchwald_hartwig()
    elif dataset_name == 'suzuki':
        return load_suzuki()
    raise ValueError(f"Unknown dataset: {dataset_name}")


def generate_features(rep_name, data):
    from kirby.representations.reaction import create_drfp, create_concat_ecfp4

    if rep_name == 'drfp_2048':
        return create_drfp(data['rxn_smiles'])
    elif rep_name == 'concat_ecfp4':
        return create_concat_ecfp4(data['components'])
    return None


def build_models():
    models = {}
    models['ridge'] = lambda: RidgeCV(alphas=np.logspace(-4, 4, 50))

    from sklearn.ensemble import RandomForestRegressor
    models['rf'] = lambda: RandomForestRegressor(
        n_estimators=500, max_features='sqrt', random_state=42, n_jobs=-1,
    )

    try:
        from xgboost import XGBRegressor
        models['xgb'] = lambda: XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            colsample_bytree=0.5, random_state=42, n_jobs=-1, verbosity=0,
        )
    except ImportError:
        pass

    return models


def evaluate_split(X, labels, train_idx, test_idx, model_factory):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = model_factory()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        'r2': float(r2_score(y_test, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mae': float(mean_absolute_error(y_test, y_pred)),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dataset', choices=DATASETS, default=None,
                        help='Dataset to evaluate (default: all)')
    parser.add_argument('--n-splits', type=int, default=10,
                        help='Number of random splits (default: 10)')
    parser.add_argument('--output', default='tests/results/reaction_yields',
                        help='Output directory')
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else DATASETS
    models = build_models()

    rep_configs = {
        'drfp_2048': 'DRFP (2048D)',
        'concat_ecfp4': 'Concat ECFP4',
    }

    all_results = {}

    for ds_name in datasets:
        print(f"\n{'=' * 70}")
        print(f"Dataset: {ds_name}")
        print('=' * 70)

        data = load_data(ds_name)
        splits = data['splits'][:args.n_splits]

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
                }

                key = f"{ds_name}/{rep_name}/{model_name}"
                all_results[key] = avg

                print(f"    {model_name:>8}: R²={avg['r2']:.4f}±{avg['r2_std']:.4f}  "
                      f"RMSE={avg['rmse']:.2f}±{avg['rmse_std']:.2f}  "
                      f"MAE={avg['mae']:.2f}±{avg['mae_std']:.2f}")

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary")
    print('=' * 70)
    print(f"{'Dataset/Rep/Model':<45} {'R²':>14} {'RMSE':>14} {'MAE':>14}")
    print('-' * 90)
    for key, r in sorted(all_results.items()):
        print(f"{key:<45} {r['r2']:.4f}±{r['r2_std']:.4f}  "
              f"{r['rmse']:>5.2f}±{r['rmse_std']:.2f}  "
              f"{r['mae']:>5.2f}±{r['mae_std']:.2f}")

    # Save
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'reaction_yield_survey.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[saved {out_file}]")


if __name__ == '__main__':
    main()
