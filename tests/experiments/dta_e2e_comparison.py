"""DTA End-to-End Comparison: KIRBy Frozen Reps vs DeepDTA vs GraphDTA.

Demonstrates WHEN frozen embeddings are competitive with end-to-end deep learning
on drug-target affinity prediction.

Three methods:
  1. Frozen ECFP4+ESM2 + XGBoost (KIRBy's approach)
  2. DeepDTA (CNN on SMILES + CNN on protein sequence, Ozturk 2018)
  3. GraphDTA-GIN (GNN on molecular graph + CNN on protein, Nguyen 2021)

All evaluated with cold-target GroupKFold to match KIRBy's standard protocol.
Note: original DeepDTA/GraphDTA papers use random splits — cold-target is harder.

Usage:
    python tests/experiments/dta_e2e_comparison.py davis
    python tests/experiments/dta_e2e_comparison.py kiba --max-pairs 30000
    python tests/experiments/dta_e2e_comparison.py davis --epochs 100

Requires: torch, torch_geometric (for GraphDTA GIN layers)
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


def concordance_index(y_true, y_pred):
    """Concordance Index (CI) — efficient O(n log n) via sorting."""
    from scipy.stats import kendalltau
    tau, _ = kendalltau(y_true, y_pred)
    return (tau + 1) / 2  # Convert Kendall's tau to CI

# ── SMILES / protein integer encoding ────────────────────────────────────────

SMILES_CHARS = sorted(set(
    "CNOSFPClBrI=#@+\\-[]()/%.0123456789cnos"
))
SMILES_DICT = {c: i + 1 for i, c in enumerate(SMILES_CHARS)}

AA_CHARS = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
AA_DICT = {c: i + 1 for i, c in enumerate(AA_CHARS)}


def encode_smiles(smi, max_len=100):
    x = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(smi[:max_len]):
        x[i] = SMILES_DICT.get(ch, len(SMILES_DICT) + 1)
    return x


def encode_protein(seq, max_len=1000):
    x = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(seq[:max_len]):
        x[i] = AA_DICT.get(ch, 0)
    return x


# ── DeepDTA model ────────────────────────────────────────────────────────────

def _build_deepdta(smi_vocab_size, prot_vocab_size, smi_max_len, prot_max_len):
    """Build DeepDTA: CNN on SMILES + CNN on protein → FC → affinity."""
    import torch
    import torch.nn as nn

    class CNNBlock(nn.Module):
        def __init__(self, vocab_size, embed_dim=128, channels=32, kernel_size=4):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size + 2, embed_dim, padding_idx=0)
            self.conv1 = nn.Conv1d(embed_dim, channels, kernel_size)
            self.conv2 = nn.Conv1d(channels, channels * 2, kernel_size)
            self.conv3 = nn.Conv1d(channels * 2, channels * 3, kernel_size)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveMaxPool1d(1)

        def forward(self, x):
            x = self.embedding(x).permute(0, 2, 1)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            return self.pool(x).squeeze(-1)

    class DeepDTA(nn.Module):
        def __init__(self):
            super().__init__()
            ch = 32
            self.drug_cnn = CNNBlock(smi_vocab_size, kernel_size=4)
            self.prot_cnn = CNNBlock(prot_vocab_size, kernel_size=8)
            self.fc = nn.Sequential(
                nn.Linear(ch * 6, 1024), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(512, 1),
            )

        def forward(self, drug, protein):
            d = self.drug_cnn(drug)
            p = self.prot_cnn(protein)
            return self.fc(torch.cat([d, p], dim=1)).squeeze(-1)

    return DeepDTA()


# ── GraphDTA GIN model ───────────────────────────────────────────────────────

def _smiles_to_graph(smi):
    """Convert SMILES to PyG Data with 78-dim atom features."""
    from rdkit import Chem
    import torch
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    # 78-dim atom features (matching GraphDTA)
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
                  'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
                  'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                  'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
                  'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']

    def atom_features(atom):
        sym = atom.GetSymbol()
        sym_enc = [int(sym == t) for t in atom_types]
        if sum(sym_enc) == 0:
            sym_enc[-1] = 1
        deg = [int(atom.GetDegree() == i) for i in range(11)]
        hs = [int(atom.GetTotalNumHs() == i) for i in range(11)]
        val = [int(atom.GetImplicitValence() == i) for i in range(11)]
        aro = [int(atom.GetIsAromatic())]
        feat = sym_enc + deg + hs + val + aro  # 44+11+11+11+1 = 78
        s = sum(feat)
        return [f / s if s > 0 else 0 for f in feat]

    feats = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.FloatTensor(feats)

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([[i, j], [j, i]])

    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.LongTensor(edges).t().contiguous()

    return Data(x=x, edge_index=edge_index)


def _build_graphdta_gin(prot_vocab_size):
    """Build GraphDTA with GIN (best variant on Davis)."""
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GINConv, global_add_pool

    class GraphDTAGIN(nn.Module):
        def __init__(self, n_features=78, dim=32, n_layers=5):
            super().__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            for i in range(n_layers):
                in_dim = n_features if i == 0 else dim
                mlp = nn.Sequential(
                    nn.Linear(in_dim, dim), nn.ReLU(), nn.Linear(dim, dim),
                )
                self.convs.append(GINConv(mlp))
                self.bns.append(nn.BatchNorm1d(dim))

            self.graph_fc = nn.Linear(dim, 128)

            # Protein CNN (same as DeepDTA)
            self.prot_embed = nn.Embedding(prot_vocab_size + 2, 128, padding_idx=0)
            self.prot_conv = nn.Conv1d(128, 128, kernel_size=8)
            self.prot_pool = nn.AdaptiveMaxPool1d(1)

            self.fc = nn.Sequential(
                nn.Linear(256, 1024), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 1),
            )
            self.relu = nn.ReLU()

        def forward(self, graph_batch, protein):
            # GNN on molecular graph
            x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
            for conv, bn in zip(self.convs, self.bns):
                x = self.relu(bn(conv(x, edge_index)))
            x = global_add_pool(x, batch)
            x = self.relu(self.graph_fc(x))  # (batch, 128)

            # Protein CNN
            p = self.prot_embed(protein).permute(0, 2, 1)
            p = self.relu(self.prot_conv(p))
            p = self.prot_pool(p).squeeze(-1)  # (batch, 128)

            return self.fc(torch.cat([x, p], dim=1)).squeeze(-1)

    return GraphDTAGIN()


# ── Training helpers ─────────────────────────────────────────────────────────

def _train_deepdta(model, train_drugs, train_prots, train_y,
                   val_drugs, val_prots, val_y,
                   epochs=100, lr=1e-3, batch_size=256, device='cpu'):
    """Train DeepDTA and return test predictions."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    train_ds = TensorDataset(
        torch.LongTensor(train_drugs), torch.LongTensor(train_prots),
        torch.FloatTensor(train_y),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_mse = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for d, p, y in train_loader:
            d, p, y = d.to(device), p.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(d, p), y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            vd = torch.LongTensor(val_drugs).to(device)
            vp = torch.LongTensor(val_prots).to(device)
            val_pred = model(vd, vp).cpu().numpy()
            val_mse = float(mean_squared_error(val_y, val_pred))

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: val_MSE={val_mse:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        vd = torch.LongTensor(val_drugs).to(device)
        vp = torch.LongTensor(val_prots).to(device)
        return model(vd, vp).cpu().numpy()


def _train_graphdta(model, train_graphs, train_prots, train_y,
                    val_graphs, val_prots, val_y,
                    epochs=100, lr=1e-3, batch_size=64, device='cpu'):
    """Train GraphDTA-GIN and return test predictions."""
    import torch
    from torch_geometric.loader import DataLoader as PyGLoader
    from torch_geometric.data import Batch

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Attach protein tensors and labels to graph data objects
    for i, g in enumerate(train_graphs):
        g.prot = torch.LongTensor(train_prots[i:i+1])
        g.label = torch.FloatTensor([train_y[i]])

    train_loader = PyGLoader(train_graphs, batch_size=batch_size, shuffle=True)

    best_val_mse = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            prot = batch.prot.to(device)
            y = batch.label.to(device)
            optimizer.zero_grad()
            pred = model(batch, prot)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for i in range(0, len(val_graphs), batch_size):
                j = min(i + batch_size, len(val_graphs))
                batch_graphs = Batch.from_data_list(val_graphs[i:j]).to(device)
                prot = torch.LongTensor(val_prots[i:j]).to(device)
                pred = model(batch_graphs, prot)
                val_preds.extend(pred.cpu().numpy())

        val_pred = np.array(val_preds)
        val_mse = float(mean_squared_error(val_y, val_pred))

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: val_MSE={val_mse:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    val_preds = []
    with torch.no_grad():
        from torch_geometric.data import Batch
        for i in range(0, len(val_graphs), batch_size):
            j = min(i + batch_size, len(val_graphs))
            batch_graphs = Batch.from_data_list(val_graphs[i:j]).to(device)
            prot = torch.LongTensor(val_prots[i:j]).to(device)
            pred = model(batch_graphs, prot)
            val_preds.extend(pred.cpu().numpy())

    return np.array(val_preds)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', choices=['davis', 'kiba'])
    parser.add_argument('--max-pairs', type=int, default=None,
                        help='Subsample (default: 30000 for KIBA)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--output', default='tests/results')
    args = parser.parse_args()

    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load dataset
    from kirby.datasets.dta import load_dataset, assign_cold_target_groups
    data = load_dataset(args.dataset)
    drug_smiles = data['drug_smiles']
    protein_seqs = data['protein_seqs']
    labels = data['labels']
    metadata = data['metadata']

    # Subsample
    max_pairs = args.max_pairs
    if max_pairs is None and args.dataset == 'kiba':
        max_pairs = 30000
    if max_pairs and len(labels) > max_pairs:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(labels), max_pairs, replace=False)
        idx.sort()
        drug_smiles = [drug_smiles[i] for i in idx]
        protein_seqs = [protein_seqs[i] for i in idx]
        labels = labels[idx]
        metadata['protein_ids'] = [metadata['protein_ids'][i] for i in idx]

    groups, n_groups = assign_cold_target_groups(metadata['protein_ids'])
    n = len(labels)
    print(f"\n{n} pairs, {n_groups} protein groups, dataset: {args.dataset}")

    # Pre-encode all inputs
    print("Encoding inputs...")
    smi_max_len = 100 if args.dataset == 'kiba' else 85
    prot_max_len = 1000

    all_drugs_enc = np.array([encode_smiles(s, smi_max_len) for s in drug_smiles])
    all_prots_enc = np.array([encode_protein(s, prot_max_len) for s in protein_seqs])

    smi_vocab = len(SMILES_DICT) + 2
    prot_vocab = len(AA_DICT) + 2

    # Pre-convert molecular graphs for GraphDTA
    print("Converting SMILES to graphs...")
    all_graphs = []
    for smi in drug_smiles:
        g = _smiles_to_graph(smi)
        if g is None:
            # Fallback: single-atom dummy graph
            g = _smiles_to_graph('C')
        all_graphs.append(g)

    # Cold-target GroupKFold CV
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=args.n_folds)

    results = {'kirby_frozen': [], 'deepdta': [], 'graphdta_gin': []}

    for fold, (train_idx, test_idx) in enumerate(gkf.split(labels, labels, groups)):
        print(f"\n--- Fold {fold+1}/{args.n_folds} (train={len(train_idx)}, test={len(test_idx)}) ---")

        y_train, y_test = labels[train_idx], labels[test_idx]

        # ── Method 1: Frozen ECFP4+ESM2 + XGBoost ──
        print("  [1] Frozen ECFP4+ESM2 + XGBoost...")
        t0 = time.time()
        try:
            from kirby.representations.molecular import create_ecfp4
            from kirby.representations.protein import create_esm2_protein_embeddings
            # Deduplicate drugs and proteins for efficiency
            unique_drugs = sorted(set(drug_smiles))
            drug_to_idx = {s: i for i, s in enumerate(unique_drugs)}
            drug_feats = create_ecfp4(unique_drugs)
            X_drug = drug_feats[np.array([drug_to_idx[drug_smiles[i]] for i in range(n)])]

            unique_prots = sorted(set(protein_seqs))
            prot_to_idx = {s: i for i, s in enumerate(unique_prots)}
            prot_feats = create_esm2_protein_embeddings(unique_prots)
            X_prot = prot_feats[np.array([prot_to_idx[protein_seqs[i]] for i in range(n)])]

            X = np.hstack([X_drug, X_prot])
            X_train, X_test = X[train_idx], X[test_idx]

            try:
                from xgboost import XGBRegressor
                model = XGBRegressor(n_estimators=100, colsample_bytree=0.5,
                                     n_jobs=-1, random_state=42, verbosity=0)
            except ImportError:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, max_features='sqrt',
                                               n_jobs=-1, random_state=42)

            model.fit(X_train, y_train)
            kirby_pred = model.predict(X_test)
            kirby_r2 = r2_score(y_test, kirby_pred)
            kirby_mse = mean_squared_error(y_test, kirby_pred)
            kirby_ci = concordance_index(y_test, kirby_pred)
            kirby_spearman = float(spearmanr(y_test, kirby_pred).correlation)
            print(f"    CI={kirby_ci:.4f}, R²={kirby_r2:.4f}, MSE={kirby_mse:.4f}, ρ={kirby_spearman:.4f} ({time.time()-t0:.1f}s)")
            results['kirby_frozen'].append({'r2': kirby_r2, 'mse': kirby_mse, 'ci': kirby_ci, 'spearman': kirby_spearman})
        except Exception as e:
            print(f"    FAILED: {e}")
            results['kirby_frozen'].append({'r2': float('nan'), 'mse': float('nan'), 'ci': float('nan'), 'spearman': float('nan')})

        # ── Method 2: DeepDTA ──
        print("  [2] DeepDTA...")
        t0 = time.time()
        try:
            deepdta = _build_deepdta(smi_vocab, prot_vocab, smi_max_len, prot_max_len)
            deepdta_pred = _train_deepdta(
                deepdta, all_drugs_enc[train_idx], all_prots_enc[train_idx], y_train,
                all_drugs_enc[test_idx], all_prots_enc[test_idx], y_test,
                epochs=args.epochs, device=device,
            )
            dt_r2 = r2_score(y_test, deepdta_pred)
            dt_mse = mean_squared_error(y_test, deepdta_pred)
            dt_ci = concordance_index(y_test, deepdta_pred)
            dt_spearman = float(spearmanr(y_test, deepdta_pred).correlation)
            print(f"    CI={dt_ci:.4f}, R²={dt_r2:.4f}, MSE={dt_mse:.4f}, ρ={dt_spearman:.4f} ({time.time()-t0:.1f}s)")
            results['deepdta'].append({'r2': dt_r2, 'mse': dt_mse, 'ci': dt_ci, 'spearman': dt_spearman})
            del deepdta
        except Exception as e:
            print(f"    FAILED: {e}")
            results['deepdta'].append({'r2': float('nan'), 'mse': float('nan'), 'ci': float('nan'), 'spearman': float('nan')})

        # ── Method 3: GraphDTA-GIN ──
        print("  [3] GraphDTA-GIN...")
        t0 = time.time()
        try:
            gin = _build_graphdta_gin(prot_vocab)
            train_graphs = [all_graphs[i] for i in train_idx]
            test_graphs = [all_graphs[i] for i in test_idx]
            gin_pred = _train_graphdta(
                gin, train_graphs, all_prots_enc[train_idx], y_train,
                test_graphs, all_prots_enc[test_idx], y_test,
                epochs=args.epochs, device=device,
            )
            gin_r2 = r2_score(y_test, gin_pred)
            gin_mse = mean_squared_error(y_test, gin_pred)
            gin_ci = concordance_index(y_test, gin_pred)
            gin_spearman = float(spearmanr(y_test, gin_pred).correlation)
            print(f"    CI={gin_ci:.4f}, R²={gin_r2:.4f}, MSE={gin_mse:.4f}, ρ={gin_spearman:.4f} ({time.time()-t0:.1f}s)")
            results['graphdta_gin'].append({'r2': gin_r2, 'mse': gin_mse, 'ci': gin_ci, 'spearman': gin_spearman})
            del gin
        except Exception as e:
            print(f"    FAILED: {e}")
            results['graphdta_gin'].append({'r2': float('nan'), 'mse': float('nan'), 'ci': float('nan'), 'spearman': float('nan')})

        if device == 'cuda':
            torch.cuda.empty_cache()

    # ── Summary ──
    print("\n" + "=" * 70)
    print(f"DTA End-to-End Comparison: {args.dataset.upper()} ({n} pairs, cold-target)")
    print("=" * 70)
    print(f"{'Method':<25} {'CI (mean±std)':>18} {'R² (mean±std)':>18} {'Spearman':>12}")
    print("-" * 78)
    for method, scores in results.items():
        r2s = [s['r2'] for s in scores if not np.isnan(s['r2'])]
        mses = [s['mse'] for s in scores if not np.isnan(s['mse'])]
        cis = [s['ci'] for s in scores if not np.isnan(s['ci'])]
        rhos = [s['spearman'] for s in scores if not np.isnan(s['spearman'])]
        if r2s:
            print(f"{method:<25} {np.mean(cis):>7.4f}±{np.std(cis):.4f}   "
                  f"{np.mean(r2s):>7.4f}±{np.std(r2s):.4f}   "
                  f"{np.mean(rhos):>7.4f}")
        else:
            print(f"{method:<25} {'FAILED':>18} {'FAILED':>18} {'FAILED':>12}")

    # Save
    out_dir = Path(args.output) / 'dta_e2e_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'{args.dataset}.json'
    with open(out_file, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'n_pairs': n,
            'n_folds': args.n_folds,
            'epochs': args.epochs,
            'results': results,
        }, f, indent=2)
    print(f"\n[saved {out_file}]")


if __name__ == '__main__':
    main()
