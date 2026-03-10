"""QM9 Three-Way Comparison: Frozen ECFP vs Frozen SchNet vs End-to-End SchNet.

Demonstrates WHEN and WHY frozen embeddings break down: tasks requiring 3D
geometry + sufficient data for end-to-end training.

Three methods:
  1. Frozen ECFP4 + XGBoost (KIRBy's standard approach)
  2. Frozen SchNet embeddings + Ridge (current frozen approach for 3D)
  3. End-to-end SchNet (trained on 3D coordinates, expected winner)

Usage:
    python tests/experiments/qm9_3d_comparison.py --property gap
    python tests/experiments/qm9_3d_comparison.py --property mu --n-samples 5000
    python tests/experiments/qm9_3d_comparison.py --property gap --n-samples 0  # full dataset (~130K)
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

PROPERTY_MAP = {
    'mu': 0, 'alpha': 1, 'homo': 2, 'lumo': 3, 'gap': 4, 'r2': 5,
    'zpve': 6, 'U0': 7, 'U': 8, 'H': 9, 'G': 10, 'Cv': 11,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--property', default='gap', choices=list(PROPERTY_MAP.keys()),
                        help='QM9 property to predict')
    parser.add_argument('--n-samples', type=int, default=10000,
                        help='Number of samples (0 = full dataset, default: 10000)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs for end-to-end SchNet')
    parser.add_argument('--output', default='tests/results',
                        help='Output directory')
    args = parser.parse_args()

    prop_idx = PROPERTY_MAP[args.property]
    n_samples = args.n_samples if args.n_samples > 0 else None

    # Load QM9 with 3D coordinates
    from kirby.datasets.qm9 import load_qm9_with_3d
    data = load_qm9_with_3d(n_samples=n_samples, property_idx=prop_idx)

    smiles = data['smiles']
    labels = data['labels']
    data_list = data['data_list']

    # Filter to RDKit-valid SMILES (some QM9 SMILES have exotic valences)
    from rdkit import Chem
    valid = [i for i, s in enumerate(smiles) if Chem.MolFromSmiles(s) is not None]
    if len(valid) < len(smiles):
        print(f"  Filtered {len(smiles) - len(valid)} invalid SMILES "
              f"({len(valid)} remaining)")
        smiles = [smiles[i] for i in valid]
        labels = labels[valid]
        data_list = [data_list[i] for i in valid]

    n = len(labels)
    print(f"\n{n} molecules, property: {args.property}")

    # Random split (80/10/10)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    y_train = labels[train_idx]
    y_test = labels[test_idx]

    results = {}

    # --- Method 1: Frozen ECFP4 + XGBoost ---
    print("\n" + "=" * 60)
    print("Method 1: Frozen ECFP4 + XGBoost")
    print("=" * 60)
    t0 = time.time()
    results['ecfp4_xgb'] = run_ecfp4_xgb(
        smiles, labels, train_idx, val_idx, test_idx
    )
    results['ecfp4_xgb']['time'] = time.time() - t0

    # --- Method 2: Frozen SchNet embeddings + Ridge ---
    print("\n" + "=" * 60)
    print("Method 2: Frozen SchNet Embeddings + Ridge")
    print("=" * 60)
    t0 = time.time()
    results['frozen_schnet'] = run_frozen_schnet(
        smiles, labels, train_idx, val_idx, test_idx
    )
    results['frozen_schnet']['time'] = time.time() - t0

    # --- Method 3: End-to-end SchNet ---
    print("\n" + "=" * 60)
    print("Method 3: End-to-End SchNet (3D)")
    print("=" * 60)
    t0 = time.time()
    results['e2e_schnet'] = run_e2e_schnet(
        data_list, labels, train_idx, val_idx, test_idx,
        prop_idx=prop_idx, epochs=args.epochs,
    )
    results['e2e_schnet']['time'] = time.time() - t0

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"QM9 {args.property} — Three-Way Comparison ({n} molecules)")
    print("=" * 60)
    print(f"{'Method':<30} {'R²':>8} {'MAE':>10} {'Time':>8}")
    print("-" * 60)
    for method, r in results.items():
        print(f"{method:<30} {r['test_r2']:>8.4f} {r['test_mae']:>10.4f} "
              f"{r['time']:>7.1f}s")

    # Save
    out_dir = Path(args.output) / 'qm9_3d_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'{args.property}_{n}.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved {out_file}]")


def run_ecfp4_xgb(smiles, labels, train_idx, val_idx, test_idx):
    """Frozen ECFP4 fingerprints + XGBoost."""
    from kirby.representations.molecular import create_ecfp4

    print("  Generating ECFP4...")
    X = create_ecfp4(smiles)
    print(f"  ECFP4 shape: {X.shape}")

    X_train, X_test = X[train_idx].astype(np.float32), X[test_idx].astype(np.float32)
    y_train, y_test = labels[train_idx], labels[test_idx]

    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1,
                             colsample_bytree=0.5, random_state=42, n_jobs=-1,
                             verbosity=0)
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=500, max_features='sqrt',
                                       random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"  Test R²: {r2:.4f}, MAE: {mae:.4f}")

    return {'test_r2': float(r2), 'test_mae': float(mae)}


def run_frozen_schnet(smiles, labels, train_idx, val_idx, test_idx):
    """Frozen SchNet embeddings + Ridge regression."""
    from kirby.representations.molecular import create_schnet
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    print("  Generating frozen SchNet embeddings...")
    try:
        X = create_schnet(smiles)
        print(f"  SchNet shape: {X.shape}")
    except Exception as e:
        print(f"  SchNet failed: {e}")
        return {'test_r2': float('nan'), 'test_mae': float('nan'),
                'error': str(e)}

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RidgeCV(alphas=np.logspace(-4, 4, 50))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"  Test R²: {r2:.4f}, MAE: {mae:.4f}")

    return {'test_r2': float(r2), 'test_mae': float(mae)}


def run_e2e_schnet(data_list, labels, train_idx, val_idx, test_idx,
                   prop_idx=4, epochs=100):
    """End-to-end SchNet trained on 3D coordinates.

    Requires: torch-cluster (for radius_graph in SchNet interaction layer).
    Install: pip install torch-cluster -f https://data.pyg-team.org/whl/torch-{TORCH_VERSION}+{CUDA}.html
    """
    import torch
    try:
        import torch_cluster  # noqa: F401
    except ImportError:
        raise ImportError(
            "torch-cluster is required for SchNet end-to-end training. "
            "Install: pip install torch-cluster -f "
            "https://data.pyg-team.org/whl/torch-$(python -c 'import torch; print(torch.__version__)')+"
            "$(python -c 'import torch; print(\"cu\" + torch.version.cuda.replace(\".\",\"\") if torch.cuda.is_available() else \"cpu\")')"
            ".html"
        )
    from torch_geometric.loader import DataLoader

    try:
        from torch_geometric.nn import SchNet
    except ImportError:
        print("  SchNet requires torch_geometric.nn.SchNet")
        return {'test_r2': float('nan'), 'test_mae': float('nan'),
                'error': 'SchNet not available'}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Prepare data: set data.y to the target property
    for i, d in enumerate(data_list):
        d.y = d.y[0, prop_idx].unsqueeze(0)  # scalar target

    train_data = [data_list[i] for i in train_idx]
    val_data = [data_list[i] for i in val_idx]
    test_data = [data_list[i] for i in test_idx]

    # Compute mean/std for normalization
    y_train_vals = np.array([d.y.item() for d in train_data])
    y_mean = float(y_train_vals.mean())
    y_std = float(y_train_vals.std())

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)

    # Build SchNet
    model = SchNet(
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_mae = float('inf')
    best_state = None

    print(f"  Training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.z, batch.pos, batch.batch)
            target = (batch.y - y_mean) / y_std
            loss = torch.nn.functional.mse_loss(pred.squeeze(), target.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs

        train_loss /= len(train_data)

        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.z, batch.pos, batch.batch)
                val_preds.append(pred.squeeze().cpu() * y_std + y_mean)
                val_true.append(batch.y.squeeze().cpu())

        val_preds = torch.cat(val_preds).numpy()
        val_true = torch.cat(val_true).numpy()
        val_mae = mean_absolute_error(val_true, val_preds)
        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            val_r2 = r2_score(val_true, val_preds)
            lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch:3d}: train_loss={train_loss:.4f} "
                  f"val_MAE={val_mae:.4f} val_R²={val_r2:.4f} lr={lr:.1e}")

    # Load best model and evaluate on test
    model.load_state_dict(best_state)
    model.eval()

    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.z, batch.pos, batch.batch)
            test_preds.append(pred.squeeze().cpu() * y_std + y_mean)
            test_true.append(batch.y.squeeze().cpu())

    test_preds = torch.cat(test_preds).numpy()
    test_true = torch.cat(test_true).numpy()

    r2 = r2_score(test_true, test_preds)
    mae = mean_absolute_error(test_true, test_preds)
    print(f"  Test R²: {r2:.4f}, MAE: {mae:.4f}")
    print(f"  Best val MAE: {best_val_mae:.4f}")

    return {
        'test_r2': float(r2),
        'test_mae': float(mae),
        'best_val_mae': float(best_val_mae),
        'epochs_trained': epochs,
    }


if __name__ == '__main__':
    main()
