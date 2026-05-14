"""Pre-train DNN on bulk ChEMBL bioactivity data.

Standalone script (run once via SLURM, not per-fold). Trains one DNN per
molecular representation on broad ChEMBL IC50 data, saving weights for
downstream fine-tuning.

Usage:
    python -m kirby.models.pretrain_chembl_dnn --rep ecfp4
    python -m kirby.models.pretrain_chembl_dnn --rep ecfp4,maccs,pdv,mol2vec,mordred,avalon

    # Smoke test (fast, small data)
    python -m kirby.models.pretrain_chembl_dnn --rep ecfp4 --n-compounds 10000

Reference:
    Imrie et al. (2018). JCIM 58(11):2319-2330.
"""

import os
import sys
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))          # src/
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'tests'))  # tests/

warnings.filterwarnings('ignore')


def build_net(input_dim, hidden_sizes=(256, 128, 64), dropout=0.2):
    """Build DNN architecture matching DNNRegressor."""
    layers = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def pretrain_one_rep(
    smiles, pactivities, rep_name, hidden_sizes, output_dir,
    epochs=200, batch_size=1024, lr=0.001, patience=30, seed=42,
):
    """Pre-train DNN on ChEMBL data for one representation.

    Saves checkpoint to {output_dir}/{rep_name}_{h1}_{h2}_{h3}.pt
    """
    from kirby.representations.molecular import (
        create_ecfp4, create_maccs, create_avalon, create_pdv,
        create_mordred, create_mol2vec,
    )

    rep_funcs = {
        'ecfp4': create_ecfp4,
        'maccs': create_maccs,
        'avalon': create_avalon,
        'pdv': create_pdv,
        'mordred': create_mordred,
        'mol2vec': create_mol2vec,
    }

    if rep_name not in rep_funcs:
        print(f"  ERROR: Unknown rep '{rep_name}'. "
              f"Available: {', '.join(rep_funcs.keys())}")
        return None

    # Generate representation
    print(f"\n{'='*60}")
    print(f"Pre-training: {rep_name}")
    print(f"{'='*60}")

    t0 = time.time()
    print(f"  Generating {rep_name} for {len(smiles)} compounds...", flush=True)
    try:
        X = rep_funcs[rep_name](smiles)
    except Exception as e:
        print(f"  FAILED to generate {rep_name}: {e}")
        return None

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(pactivities, dtype=np.float32)

    # Remove any rows with NaN/inf in features
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X = X[valid]
    y = y[valid]

    elapsed_rep = time.time() - t0
    print(f"  Rep shape: {X.shape} ({elapsed_rep:.1f}s)")

    # Z-normalize features (prevents NaN from large values in linear layers)
    from sklearn.preprocessing import StandardScaler
    feat_scaler = StandardScaler()
    X = feat_scaler.fit_transform(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Z-normalize targets
    y_mean = float(y.mean())
    y_std = float(y.std()) if y.std() > 1e-8 else 1.0
    y_norm = (y - y_mean) / y_std

    # 90/10 holdout
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    split = int(0.9 * len(y))
    train_idx, ho_idx = idx[:split], idx[split:]

    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y_norm[train_idx])
    X_ho = torch.from_numpy(X[ho_idx])
    y_ho = torch.from_numpy(y_norm[ho_idx])

    print(f"  Train: {len(train_idx)}, Holdout: {len(ho_idx)}")

    device = torch.device('cpu')
    net = build_net(X.shape[1], hidden_sizes).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    best_ho_loss = float('inf')
    best_state = None
    patience_counter = 0
    t0_train = time.time()

    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(X_train), batch_size):
            batch_idx = perm[start:start + batch_size]
            xb = X_train[batch_idx].to(device)
            yb = y_train[batch_idx].to(device)

            optimizer.zero_grad()
            pred = net(xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Holdout eval
        net.eval()
        with torch.no_grad():
            ho_pred = net(X_ho.to(device)).squeeze(-1)
            ho_loss = criterion(ho_pred, y_ho.to(device)).item()

        scheduler.step(ho_loss)

        if ho_loss < best_ho_loss:
            best_ho_loss = ho_loss
            best_state = {k: v.cpu().clone()
                          for k, v in net.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            # Convert holdout MSE back to pIC50 units
            ho_rmse_pIC50 = np.sqrt(ho_loss) * y_std
            print(f"  Epoch {epoch+1:3d}: train_loss={epoch_loss/n_batches:.4f}, "
                  f"ho_loss={ho_loss:.4f} (RMSE={ho_rmse_pIC50:.3f} pIC50), "
                  f"patience={patience_counter}/{patience}", flush=True)

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    elapsed_train = time.time() - t0_train

    if best_state is not None:
        net.load_state_dict(best_state)

    # Compute final holdout metrics in original scale
    net.eval()
    with torch.no_grad():
        ho_pred_final = net(X_ho.to(device)).squeeze(-1).numpy()
    ho_pred_orig = ho_pred_final * y_std + y_mean
    ho_true_orig = y[ho_idx]
    ho_mse = float(np.mean((ho_pred_orig - ho_true_orig) ** 2))
    ho_rmse = float(np.sqrt(ho_mse))

    print(f"\n  Final holdout: MSE={ho_mse:.4f}, RMSE={ho_rmse:.3f} pIC50")
    print(f"  Training time: {elapsed_train:.1f}s")

    # Save checkpoint
    os.makedirs(output_dir, exist_ok=True)
    hs_str = '_'.join(str(h) for h in hidden_sizes)
    save_path = os.path.join(output_dir, f'{rep_name}_{hs_str}.pt')

    checkpoint = {
        'state_dict': best_state if best_state is not None else net.state_dict(),
        'input_dim': X.shape[1],
        'hidden_sizes': hidden_sizes,
        'n_compounds': len(y),
        'holdout_mse': ho_mse,
        'holdout_rmse': ho_rmse,
        'y_mean': y_mean,
        'y_std': y_std,
        'rep_name': rep_name,
        # Pretrain feature scaler (mean_/scale_ as plain lists so the
        # checkpoint stays loadable under torch.load(weights_only=True)).
        # Required for faithful zero-shot transfer: the net's weights only
        # make sense on input scaled by THIS scaler, not one re-fit on the
        # downstream set. Consumed by PretrainedDNNRegressor(zero_shot=True).
        'feat_mean': feat_scaler.mean_.tolist(),
        'feat_scale': feat_scaler.scale_.tolist(),
    }
    torch.save(checkpoint, save_path)
    print(f"  Saved: {save_path}")

    return {
        'rep_name': rep_name,
        'input_dim': X.shape[1],
        'n_compounds': len(y),
        'holdout_mse': ho_mse,
        'holdout_rmse': ho_rmse,
        'save_path': save_path,
        'train_time': elapsed_train,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Pre-train DNN on ChEMBL bioactivity data for transfer learning.')
    parser.add_argument(
        '--rep', type=str, required=True,
        help='Comma-separated rep names (ecfp4,maccs,pdv,mol2vec,mordred,avalon)')
    parser.add_argument(
        '--n-compounds', type=int, default=500_000,
        help='Max compounds from ChEMBL (default: 500000)')
    parser.add_argument(
        '--max-targets', type=int, default=200,
        help='Number of top targets by activity count (default: 200)')
    parser.add_argument(
        '--hidden-sizes', type=str, default='256,128,64',
        help='DNN hidden layer sizes (default: 256,128,64)')
    parser.add_argument(
        '--output-dir', type=str,
        default='/Volumes/seagate/kirby/chembl_pretrain',
        help='Directory to save pre-trained weights')
    parser.add_argument(
        '--data-dir', type=str,
        default='/Volumes/seagate/kirby/chembl',
        help='Directory for cached ChEMBL data')
    parser.add_argument(
        '--epochs', type=int, default=200,
        help='Max training epochs (default: 200)')
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Pre-training learning rate (default: 0.001)')
    parser.add_argument(
        '--patience', type=int, default=30,
        help='Early stopping patience (default: 30)')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)')

    args = parser.parse_args()

    rep_names = [r.strip() for r in args.rep.split(',')]
    hidden_sizes = tuple(int(h) for h in args.hidden_sizes.split(','))

    print("=" * 70)
    print("ChEMBL DNN Pre-Training (Imrie Transfer Learning)")
    print(f"  Reps: {rep_names}")
    print(f"  Hidden sizes: {hidden_sizes}")
    print(f"  Max compounds: {args.n_compounds}")
    print(f"  Max targets: {args.max_targets}")
    print(f"  Output: {args.output_dir}")
    print("=" * 70)

    # Collect downstream InChIKeys for leakage prevention
    from kirby.datasets.chembl_bioactivity import (
        download_chembl_bioactivity, collect_downstream_inchikeys,
    )

    print("\nCollecting downstream InChIKeys for leakage prevention...")
    exclude_keys = collect_downstream_inchikeys()

    # Download/load ChEMBL data
    print("\nLoading ChEMBL bioactivity data...")
    df = download_chembl_bioactivity(
        data_dir=args.data_dir,
        max_targets=args.max_targets,
        max_compounds=args.n_compounds,
        exclude_inchikeys=exclude_keys,
        random_seed=args.seed,
    )

    smiles = df['smiles'].tolist()
    pactivities = df['pActivity'].values

    # Pre-train each rep
    results = []
    for rep_name in rep_names:
        result = pretrain_one_rep(
            smiles, pactivities, rep_name, hidden_sizes, args.output_dir,
            epochs=args.epochs, batch_size=1024, lr=args.lr,
            patience=args.patience, seed=args.seed,
        )
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("PRE-TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Rep':12s}  {'Dim':>6s}  {'N':>8s}  {'RMSE':>8s}  {'Time':>8s}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in results:
        print(f"  {r['rep_name']:12s}  {r['input_dim']:6d}  "
              f"{r['n_compounds']:8d}  {r['holdout_rmse']:8.3f}  "
              f"{r['train_time']:7.1f}s")
    print(f"\nWeights saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
