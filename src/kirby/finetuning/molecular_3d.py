"""
Molecular 3D GNN fine-tuning factory.

Fine-tunes SchNet or DimeNet++ end-to-end on 3D conformers, then extracts
penultimate-layer embeddings (128D for SchNet, hidden_channels for DimeNet++).

Architecture during training:
    SMILES → 3D conformer (RDKit) → 3D GNN → regression head → property
Extraction: penultimate-layer embeddings (before final linear layers).

Requires: torch, torch_geometric, rdkit
Optional: torch-cluster (for SchNet interaction_graph; fallback provided)
"""

import os
from typing import Callable, List, Tuple

import numpy as np

from kirby.finetuning._base import (
    fold_cache_key,
    get_device,
    load_or_compute,
)

# (embed_dim, model_class_name)
_MODEL_CONFIGS = {
    'schnet': (128, 'SchNet'),
    'dimenetpp': (128, 'DimeNetPlusPlus'),
}


def _smiles_to_3d_data(smiles):
    """Convert SMILES to PyG Data with 3D coordinates via RDKit conformer."""
    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, randomSeed=42)
    if result == -1:
        result = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
    if result == -1:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass

    conf = mol.GetConformer()
    z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()],
                     dtype=torch.long)
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
    return Data(z=z, pos=pos)


def _convert_smiles_3d(smiles_list):
    """Convert SMILES list to 3D Data objects. Returns (graphs, failed_indices)."""
    import torch
    from torch_geometric.data import Data

    graphs = []
    failed = []
    for i, smi in enumerate(smiles_list):
        g = _smiles_to_3d_data(smi)
        if g is None:
            failed.append(i)
            # Placeholder: single carbon at origin
            g = Data(
                z=torch.tensor([6], dtype=torch.long),
                pos=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float),
            )
        graphs.append(g)

    if failed:
        print(f"    Warning: {len(failed)}/{len(smiles_list)} molecules "
              f"failed 3D conformer generation")
    return graphs, failed


def _extract_schnet_embeddings(model, graphs, batch_size, device):
    """Extract 128D penultimate embeddings from SchNet (after interaction blocks)."""
    import torch
    from torch_geometric.data import Batch
    from torch_geometric.nn import global_mean_pool

    model.eval()
    all_embs = []

    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch_graphs = graphs[start:start + batch_size]
            batch = Batch.from_data_list(batch_graphs).to(device)

            # Manual forward: embedding → interactions → mean pool
            h = model.embedding(batch.z)

            try:
                edge_index, edge_weight = model.interaction_graph(
                    batch.pos, batch.batch)
            except (ImportError, RuntimeError):
                # Fallback if torch-cluster not available
                edge_index, edge_weight = _compute_radius_graph(
                    batch.pos, batch.batch, cutoff=10.0, device=device)

            edge_attr = model.distance_expansion(edge_weight)

            for interaction in model.interactions:
                h = h + interaction(h, edge_index, edge_weight, edge_attr)

            mol_embs = global_mean_pool(h, batch.batch)
            all_embs.append(mol_embs.cpu().numpy())

    return np.vstack(all_embs).astype(np.float32)


def _extract_dimenetpp_embeddings(model, graphs, batch_size, device,
                                  hidden_channels=128):
    """Extract hidden_channels-D embeddings from DimeNet++.

    DimeNet++ operates on edge-level embeddings internally. Its output_blocks
    scatter edges→nodes and project to out_channels. We built the model with
    out_channels=hidden_channels so the forward pass directly returns
    (batch_size, hidden_channels) graph-level embeddings.
    """
    import torch
    from torch_geometric.data import Batch

    model.eval()
    all_embs = []

    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch_graphs = graphs[start:start + batch_size]
            batch = Batch.from_data_list(batch_graphs).to(device)

            try:
                # Forward returns (batch_size, out_channels=hidden_channels)
                h = model(batch.z, batch.pos, batch.batch)
                all_embs.append(h.cpu().numpy())
            except Exception as e:
                print(f"    Warning: DimeNet++ batch failed: {e}")
                n_batch = len(batch_graphs)
                all_embs.append(np.zeros(
                    (n_batch, hidden_channels), dtype=np.float32))

    return np.vstack(all_embs).astype(np.float32)


def _compute_radius_graph(pos, batch, cutoff=10.0, max_neighbors=32, device='cpu'):
    """Fallback radius graph computation without torch-cluster."""
    import torch

    edge_index_list = []
    edge_weight_list = []

    unique_batches = batch.unique()
    for b in unique_batches:
        mol_mask = (batch == b)
        mol_indices = torch.where(mol_mask)[0]
        mol_pos = pos[mol_mask]

        dists = torch.cdist(mol_pos, mol_pos)
        for i in range(len(mol_pos)):
            within_cutoff = (dists[i] < cutoff) & (dists[i] > 0)
            neighbors = torch.where(within_cutoff)[0]

            if len(neighbors) > max_neighbors:
                _, top_idx = dists[i][neighbors].topk(
                    max_neighbors, largest=False)
                neighbors = neighbors[top_idx]

            for j in neighbors:
                edge_index_list.append(
                    [mol_indices[i].item(), mol_indices[j].item()])
                edge_weight_list.append(dists[i][j].item())

    if not edge_index_list:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)
        edge_weight = torch.tensor([0.1], dtype=torch.float, device=device)
    else:
        edge_index = torch.tensor(
            edge_index_list, dtype=torch.long, device=device).t()
        edge_weight = torch.tensor(
            edge_weight_list, dtype=torch.float, device=device)

    return edge_index, edge_weight


def _finetune_and_extract(
    train_smiles: List[str],
    val_smiles: List[str],
    train_labels: np.ndarray,
    model_name: str,
    hidden_channels: int,
    epochs: int,
    lr: float,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fine-tune 3D GNN and extract penultimate-layer embeddings."""
    import torch
    import torch.nn as nn
    from torch_geometric.data import Batch
    from torch_geometric.loader import DataLoader as PyGDataLoader

    device = get_device()

    # Convert SMILES to 3D graphs
    print(f"    Generating 3D conformers (train={len(train_smiles)}, "
          f"val={len(val_smiles)})...")
    train_graphs, train_failed = _convert_smiles_3d(train_smiles)
    val_graphs, val_failed = _convert_smiles_3d(val_smiles)

    # Build model
    if model_name == 'schnet':
        from torch_geometric.nn import SchNet
        model = SchNet(
            hidden_channels=hidden_channels,
            num_filters=hidden_channels,
            num_interactions=6,
            num_gaussians=50,
            cutoff=10.0,
            readout='mean',  # match mean-pool in _extract_schnet_embeddings
        )
    elif model_name == 'dimenetpp':
        from torch_geometric.nn.models import DimeNetPlusPlus
        # out_channels=hidden_channels so forward() returns (batch, hidden_channels)
        # embeddings directly — used for both training (via head) and extraction
        model = DimeNetPlusPlus(
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_blocks=4,
            int_emb_size=64,
            basis_emb_size=8,
            out_emb_channels=256,
            num_spherical=7,
            num_radial=6,
            cutoff=5.0,
        )
    else:
        raise ValueError(f"Unknown 3D model: {model_name}")

    # Regression head for DimeNet++ (out_channels=hidden_channels → scalar)
    # SchNet has built-in regression output (scalar), so head is unused for SchNet
    if model_name == 'dimenetpp':
        head = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    else:
        head = None

    model = model.to(device)
    if head is not None:
        head = head.to(device)

    # Attach labels to train graphs
    for i, g in enumerate(train_graphs):
        g.y = torch.FloatTensor([train_labels[i]])

    # Early stopping holdout (10%)
    n_train = len(train_graphs)
    n_holdout = max(1, int(0.1 * n_train))
    perm = np.random.RandomState(42).permutation(n_train)
    ho_idx = perm[:n_holdout]
    fit_idx = perm[n_holdout:]

    fit_graphs = [train_graphs[i] for i in fit_idx]
    ho_graphs = [train_graphs[i] for i in ho_idx]

    fit_loader = PyGDataLoader(fit_graphs, batch_size=batch_size, shuffle=True)

    # Normalize targets
    y_train = np.array([g.y.item() for g in fit_graphs])
    y_mean, y_std = float(y_train.mean()), float(max(y_train.std(), 1e-8))

    params = list(model.parameters())
    if head is not None:
        params += list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6,
    )
    criterion = nn.MSELoss()
    best_ho_loss = float('inf')
    best_state = None
    best_head_state = None
    patience = 10
    patience_counter = 0

    def _predict(model, head, batch):
        """Forward pass: SchNet returns scalar directly, DimeNet++ needs head."""
        out = model(batch.z, batch.pos, batch.batch)
        if head is not None:
            out = head(out)
        return out

    print(f"    Fine-tuning {model_name} for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        if head is not None:
            head.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in fit_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred = _predict(model, head, batch)
            target = (batch.y.to(device) - y_mean) / y_std
            loss = criterion(pred.squeeze(), target.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Holdout evaluation (batched to avoid OOM)
        model.eval()
        if head is not None:
            head.eval()
        ho_loss_total = 0.0
        ho_n = 0
        with torch.no_grad():
            for ho_start in range(0, len(ho_graphs), batch_size):
                ho_sub = ho_graphs[ho_start:ho_start + batch_size]
                ho_batch = Batch.from_data_list(ho_sub).to(device)
                ho_pred = _predict(model, head, ho_batch)
                ho_target = (ho_batch.y.to(device) - y_mean) / y_std
                ho_loss_total += criterion(
                    ho_pred.squeeze(), ho_target.squeeze()
                ).item() * len(ho_sub)
                ho_n += len(ho_sub)
        ho_loss = ho_loss_total / max(ho_n, 1)

        scheduler.step(ho_loss)

        if ho_loss < best_ho_loss:
            best_ho_loss = ho_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            if head is not None:
                best_head_state = {k: v.cpu().clone()
                                   for k, v in head.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"      Early stop at epoch {epoch}")
                break

        if epoch % 20 == 0 or epoch == 1:
            print(f"      Epoch {epoch:3d}: train_loss={avg_loss:.4f} "
                  f"ho_loss={ho_loss:.4f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    if head is not None and best_head_state is not None:
        head.load_state_dict(best_head_state)
    model = model.to(device)

    # Extract penultimate-layer embeddings (discard prediction head)
    if model_name == 'schnet':
        train_embs = _extract_schnet_embeddings(
            model, train_graphs, batch_size, device)
        val_embs = _extract_schnet_embeddings(
            model, val_graphs, batch_size, device)
    else:
        train_embs = _extract_dimenetpp_embeddings(
            model, train_graphs, batch_size, device, hidden_channels)
        val_embs = _extract_dimenetpp_embeddings(
            model, val_graphs, batch_size, device, hidden_channels)

    # Zero out failed molecules
    for idx in train_failed:
        train_embs[idx] = 0.0
    for idx in val_failed:
        val_embs[idx] = 0.0

    print(f"    Extracted {model_name} embeddings: "
          f"train={train_embs.shape}, val={val_embs.shape}")

    # Cleanup
    del model, optimizer
    if head is not None:
        del head
    if device == 'cuda':
        torch.cuda.empty_cache()

    return train_embs, val_embs


def make_molecular_3d_factory(
    smiles_list: List[str],
    dataset_name: str,
    base_model: str = 'schnet',
    hidden_channels: int = 128,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    cache: bool = True,
) -> Callable:
    """Create a 3D GNN fine-tuning factory for molecular property prediction.

    Fine-tunes SchNet (128D) or DimeNet++ on per-fold 3D conformers,
    then extracts penultimate-layer embeddings.

    Args:
        smiles_list: SMILES strings.
        dataset_name: Used for cache key.
        base_model: 'schnet' or 'dimenetpp'.
        hidden_channels: Hidden embedding dimension (default: 128).
        epochs: Training epochs (default: 100).
        lr: Learning rate (default: 1e-3).
        batch_size: Batch size (default: 64).
        cache: Cache fold embeddings to disk.

    Returns:
        Factory callable: (train_idx, val_idx, train_labels) -> (train_embs, val_embs)
        Output: (N, hidden_channels) embeddings from penultimate layer.
    """
    if base_model not in _MODEL_CONFIGS:
        raise ValueError(
            f"Unknown base_model '{base_model}'. "
            f"Available: {list(_MODEL_CONFIGS.keys())}"
        )

    hyperparams = {
        'base_model': base_model,
        'hidden_channels': hidden_channels,
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
    }

    rep_name = f'ft_{base_model}'

    def factory(
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        train_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        key = fold_cache_key(dataset_name, rep_name, train_idx, hyperparams)

        def _compute():
            train_smi = [smiles_list[i] for i in train_idx]
            val_smi = [smiles_list[i] for i in val_idx]
            return _finetune_and_extract(
                train_smiles=train_smi,
                val_smiles=val_smi,
                train_labels=train_labels,
                model_name=base_model,
                hidden_channels=hidden_channels,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
            )

        return load_or_compute(key, _compute, cache=cache)

    return factory
