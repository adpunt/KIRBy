"""
Molecular GNN fine-tuning factory.

Fine-tunes pretrained MolCLR (GIN) or GraphMVP (GIN) GNNs on per-fold
training data, then extracts graph-level embeddings.

Requires: torch, torch_geometric, rdkit
"""

import os
import sys
from typing import Callable, List, Tuple

import numpy as np

from kirby.finetuning._base import (
    fold_cache_key,
    get_device,
    load_or_compute,
)

# (embed_dim, model_type)
_MODEL_CONFIGS = {
    'molclr': (512, 'gin'),
    'graphmvp': (300, 'gin'),
}


def _get_models_dir():
    """Get models directory (same logic as molecular.py)."""
    if 'KIRBY_MODELS_DIR' in os.environ:
        return os.environ['KIRBY_MODELS_DIR']
    defaults = [
        '/Volumes/seagate/kirby_models',
        os.path.expanduser('~/kirby_models'),
        os.path.expanduser('~/repos'),
    ]
    for path in defaults:
        if os.path.exists(path):
            return path
    return os.path.expanduser('~')


def _safe_torch_load(path, map_location=None):
    """Load checkpoint with PyTorch 2.6+ compatibility."""
    import torch
    torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
    if torch_version >= (2, 6):
        return torch.load(path, map_location=map_location, weights_only=False)
    return torch.load(path, map_location=map_location)


def _smiles_to_molclr_graph(smiles):
    """Convert SMILES to torch_geometric Data with MolCLR-style features."""
    import torch
    from rdkit import Chem
    from torch_geometric.data import Data

    ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    CHIRALITY_LIST = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ]
    BOND_LIST = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    BONDDIR_LIST = [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        atom_idx = ATOM_LIST.index(sym) if sym in ATOM_LIST else len(ATOM_LIST) - 1
        chiral = atom.GetChiralTag()
        chiral_idx = CHIRALITY_LIST.index(chiral) if chiral in CHIRALITY_LIST else 0
        atom_features.append([atom_idx, chiral_idx])

    x = torch.tensor(atom_features, dtype=torch.long)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        bond_idx = BOND_LIST.index(bt) if bt in BOND_LIST else 0
        bd = bond.GetBondDir()
        dir_idx = BONDDIR_LIST.index(bd) if bd in BONDDIR_LIST else 0
        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.extend([[bond_idx, dir_idx], [bond_idx, dir_idx]])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _smiles_to_graphmvp_graph(smiles):
    """Convert SMILES to torch_geometric Data with OGB-style features."""
    import torch
    from rdkit import Chem
    from torch_geometric.data import Data
    from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.long)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ef = bond_to_feature_vector(bond)
        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.extend([ef, ef])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _convert_smiles(smiles_list, model_name):
    """Convert SMILES to graph Data objects."""
    import torch
    from torch_geometric.data import Data

    convert_fn = _smiles_to_molclr_graph if model_name == 'molclr' else _smiles_to_graphmvp_graph
    n_edge_feat = 2 if model_name == 'molclr' else 3
    n_node_feat = 2 if model_name == 'molclr' else 9

    graphs = []
    failed = []
    for i, smi in enumerate(smiles_list):
        g = convert_fn(smi)
        if g is None:
            failed.append(i)
            g = Data(
                x=torch.zeros((1, n_node_feat), dtype=torch.long),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, n_edge_feat), dtype=torch.long),
            )
        graphs.append(g)

    if failed:
        print(f"    Warning: {len(failed)} molecules failed graph conversion")
    return graphs, failed


def _load_molclr_model(device):
    """Load pretrained MolCLR GIN model."""
    import torch

    models_dir = _get_models_dir()
    search_paths = [
        os.path.join(models_dir, 'MolCLR'),
        os.path.expanduser('~/repos/MolCLR'),
        os.path.expanduser('~/kirby_models/MolCLR'),
    ]
    molclr_dir = None
    for path in search_paths:
        if os.path.exists(os.path.join(path, 'ckpt')):
            molclr_dir = os.path.abspath(path)
            break

    if molclr_dir is None:
        raise FileNotFoundError("MolCLR directory not found")

    if molclr_dir not in sys.path:
        sys.path.insert(0, molclr_dir)

    ckpt_path = os.path.join(molclr_dir, 'ckpt', 'pretrained_gin', 'checkpoints', 'model.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"MolCLR checkpoint not found at {ckpt_path}")

    from models.ginet_finetune import GINet
    model = GINet(task='classification', num_layer=5, emb_dim=300,
                  feat_dim=512, drop_ratio=0, pool='mean')
    state_dict = _safe_torch_load(ckpt_path, map_location=device)
    model.load_my_state_dict(state_dict)
    return model


def _load_graphmvp_model(device):
    """Load pretrained GraphMVP GIN model (inline architecture)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import MessagePassing, global_mean_pool

    class CustomAtomEncoder(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.atom_embedding_list = nn.ModuleList([
                nn.Embedding(119, emb_dim), nn.Embedding(4, emb_dim),
                nn.Embedding(12, emb_dim), nn.Embedding(12, emb_dim),
                nn.Embedding(10, emb_dim), nn.Embedding(6, emb_dim),
                nn.Embedding(6, emb_dim), nn.Embedding(2, emb_dim),
                nn.Embedding(2, emb_dim),
            ])

        def forward(self, x):
            out = 0
            for i, emb in enumerate(self.atom_embedding_list):
                out = out + emb(x[:, i])
            return out

    class CustomBondEncoder(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.bond_embedding_list = nn.ModuleList([
                nn.Embedding(5, emb_dim), nn.Embedding(6, emb_dim),
                nn.Embedding(2, emb_dim),
            ])

        def forward(self, edge_attr):
            out = 0
            for i, emb in enumerate(self.bond_embedding_list):
                out = out + emb(edge_attr[:, i])
            return out

    class GINConv(MessagePassing):
        def __init__(self, emb_dim):
            super().__init__(aggr="add")
            self.mlp = nn.Sequential(
                nn.Linear(emb_dim, 2 * emb_dim),
                nn.BatchNorm1d(2 * emb_dim),
                nn.ReLU(),
                nn.Linear(2 * emb_dim, emb_dim),
            )
            self.eps = nn.Parameter(torch.zeros(1))
            self.bond_encoder = CustomBondEncoder(emb_dim)

        def forward(self, x, edge_index, edge_attr):
            edge_embedding = self.bond_encoder(edge_attr)
            return self.mlp(
                (1 + self.eps) * x
                + self.propagate(edge_index, x=x, edge_attr=edge_embedding)
            )

        def message(self, x_j, edge_attr):
            return F.relu(x_j + edge_attr)

    class GraphMVP_GNN(nn.Module):
        def __init__(self, num_layer=5, emb_dim=300, drop_ratio=0.0):
            super().__init__()
            self.num_layer = num_layer
            self.drop_ratio = drop_ratio
            self.atom_encoder = CustomAtomEncoder(emb_dim)
            self.gnns = nn.ModuleList([GINConv(emb_dim) for _ in range(num_layer)])
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer)])

        def forward(self, x, edge_index, edge_attr, batch):
            h = self.atom_encoder(x)
            for layer in range(self.num_layer):
                h = self.gnns[layer](h, edge_index, edge_attr)
                h = self.batch_norms[layer](h)
                if layer < self.num_layer - 1:
                    h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
                else:
                    h = F.dropout(h, self.drop_ratio, training=self.training)
            return global_mean_pool(h, batch)

    models_dir = _get_models_dir()
    search_paths = [
        os.path.join(models_dir, 'GraphMVP'),
        os.path.expanduser('~/repos/GraphMVP'),
        os.path.expanduser('~/kirby_models/GraphMVP'),
    ]
    ckpt_path = None
    for base in search_paths:
        candidates = [
            os.path.join(base, 'pretrained_models', 'GraphMVP.pth'),
            os.path.join(base, 'MoleculeSTM_weights', 'pretrained_GraphMVP', 'GraphMVP_C', 'model.pth'),
            os.path.join(base, 'MoleculeSTM_weights', 'pretrained_GraphMVP', 'GraphMVP_G', 'model.pth'),
        ]
        for p in candidates:
            if os.path.exists(p):
                ckpt_path = p
                break
        if ckpt_path:
            break

    if ckpt_path is None:
        raise FileNotFoundError("GraphMVP checkpoint not found")

    emb_dim = 300
    model = GraphMVP_GNN(num_layer=5, emb_dim=emb_dim, drop_ratio=0.0)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def _finetune_and_extract(
    train_smiles: List[str],
    val_smiles: List[str],
    train_labels: np.ndarray,
    model_name: str,
    epochs: int,
    lr: float,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fine-tune pretrained GNN and extract graph-level embeddings."""
    import torch
    import torch.nn as nn
    from torch_geometric.data import Batch
    from torch_geometric.loader import DataLoader as PyGDataLoader

    embed_dim = _MODEL_CONFIGS[model_name][0]
    device = get_device()

    # Load pretrained model
    if model_name == 'molclr':
        model = _load_molclr_model(device)
    else:
        model = _load_graphmvp_model(device)

    # Make all params trainable + add regression head
    for param in model.parameters():
        param.requires_grad = True

    head = nn.Sequential(
        nn.Linear(embed_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 1),
    )

    model = model.to(device)
    head = head.to(device)

    # Convert SMILES to graphs
    train_graphs, train_failed = _convert_smiles(train_smiles, model_name)
    val_graphs, val_failed = _convert_smiles(val_smiles, model_name)

    # Attach labels for DataLoader
    for i, g in enumerate(train_graphs):
        g.y = torch.FloatTensor([train_labels[i]])

    # Early stopping holdout
    n_train = len(train_graphs)
    n_holdout = max(1, int(0.1 * n_train))
    perm = np.random.RandomState(42).permutation(n_train)
    ho_idx = perm[:n_holdout]
    fit_idx = perm[n_holdout:]

    fit_graphs = [train_graphs[i] for i in fit_idx]
    ho_graphs = [train_graphs[i] for i in ho_idx]

    fit_loader = PyGDataLoader(fit_graphs, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=0.01,
    )
    criterion = nn.MSELoss()
    best_ho_loss = float('inf')
    patience = 2
    patience_counter = 0

    def _forward(model, batch):
        """Forward pass for both MolCLR and GraphMVP."""
        if model_name == 'molclr':
            h, _ = model(batch)
        else:
            h = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        return h

    for epoch in range(epochs):
        model.train()
        head.train()
        for batch in fit_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            h = _forward(model, batch)
            pred = head(h).squeeze(-1)
            loss = criterion(pred, batch.y.to(device).squeeze(-1))
            loss.backward()
            optimizer.step()

        # Holdout check
        model.eval()
        head.eval()
        with torch.no_grad():
            ho_batch = Batch.from_data_list(ho_graphs).to(device)
            ho_h = _forward(model, ho_batch)
            ho_pred = head(ho_h).squeeze(-1)
            ho_y = ho_batch.y.to(device).squeeze(-1)
            ho_loss = criterion(ho_pred, ho_y).item()

        if ho_loss < best_ho_loss:
            best_ho_loss = ho_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Extract embeddings (head discarded)
    model.eval()

    def _extract(graphs, failed_indices):
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(graphs), batch_size):
                batch_graphs = graphs[i:i + batch_size]
                batch = Batch.from_data_list(batch_graphs).to(device)
                h = _forward(model, batch)
                all_embs.append(h.cpu().numpy())
        result = np.vstack(all_embs).astype(np.float32)
        for idx in failed_indices:
            result[idx] = 0.0
        return result

    train_embs = _extract(train_graphs, train_failed)
    val_embs = _extract(val_graphs, val_failed)

    # Cleanup
    del model, head, optimizer
    if device == 'cuda':
        import torch
        torch.cuda.empty_cache()

    return train_embs, val_embs


def make_molecular_gnn_factory(
    smiles_list: List[str],
    dataset_name: str,
    base_model: str = 'molclr',
    epochs: int = 5,
    lr: float = 1e-4,
    batch_size: int = 32,
    cache: bool = True,
) -> Callable:
    """Create a GNN fine-tuning factory for molecular graphs.

    Fine-tunes MolCLR (512D) or GraphMVP (300D) on per-fold data,
    then extracts graph-level embeddings.

    Args:
        smiles_list: SMILES strings.
        dataset_name: Used for cache key.
        base_model: 'molclr' or 'graphmvp'.
        epochs: Fine-tuning epochs (default: 5).
        lr: Learning rate (default: 1e-4).
        batch_size: Batch size (default: 32).
        cache: Cache fold embeddings to disk.

    Returns:
        Factory callable: (train_idx, val_idx, train_labels) -> (train_embs, val_embs)
    """
    if base_model not in _MODEL_CONFIGS:
        raise ValueError(
            f"Unknown base_model '{base_model}'. "
            f"Available: {list(_MODEL_CONFIGS.keys())}"
        )

    hyperparams = {
        'base_model': base_model,
        'epochs': epochs,
        'lr': lr,
    }

    rep_name = f'gnn_{base_model}'

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
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
            )

        return load_or_compute(key, _compute, cache=cache)

    return factory
