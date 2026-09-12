"""
Molecular GROVER fine-tuning factory.

Fine-tunes GROVER (self-supervised graph transformer) on per-fold
training data, then extracts atom+bond fingerprints (~5000D).

Uses GROVER's own data pipeline (MolCollator, MoleculeDataset).

Requires: torch, GROVER repo cloned
"""

import copy
import os
import sys
import tempfile
from typing import Callable, List, Tuple

import numpy as np

from kirby.finetuning._base import (
    fold_cache_key,
    get_device,
    load_or_compute,
)

# Factory version — bump when the training recipe, tokenizer, or base-model
# weights change in a way that invalidates prior cached fold embeddings.
_FACTORY_VERSION = 'v1'


def _get_models_dir():
    """Get models directory."""
    if 'KIRBY_MODELS_DIR' in os.environ:
        return os.environ['KIRBY_MODELS_DIR']
    defaults = [
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


def _find_grover_paths():
    """Find GROVER repo directory and checkpoint path."""
    models_dir = _get_models_dir()

    search_dirs = [
        os.path.join(models_dir, 'grover'),
        os.path.expanduser('~/repos/grover'),
        os.path.expanduser('~/kirby_models/grover'),
        os.path.expanduser('~/grover'),
    ]
    grover_dir = None
    for path in search_dirs:
        if os.path.exists(os.path.join(path, 'grover', 'data')):
            grover_dir = os.path.abspath(path)
            break

    if grover_dir is None:
        raise FileNotFoundError(
            f"GROVER repo not found. Searched: {search_dirs}\n"
            "Clone to: git clone https://github.com/tencent-ailab/grover.git"
        )

    search_ckpts = [
        os.path.join(grover_dir, 'grover_large.pt'),
        os.path.join(grover_dir, 'grover_base.pt'),
        os.path.join(models_dir, 'grover_large.pt'),
        os.path.join(models_dir, 'grover_base.pt'),
    ]
    checkpoint_path = None
    for p in search_ckpts:
        if os.path.exists(p):
            checkpoint_path = p
            break

    if checkpoint_path is None:
        raise FileNotFoundError(
            f"GROVER checkpoint not found. Searched: {search_ckpts}"
        )

    return grover_dir, checkpoint_path


def _smiles_to_csv(smiles_list, labels=None):
    """Write SMILES (and optional labels) to a temp CSV for GROVER's pipeline."""
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    if labels is not None:
        f.write('smiles,target\n')
        for smi, y in zip(smiles_list, labels):
            f.write(f'{smi},{y}\n')
    else:
        f.write('smiles\n')
        for smi in smiles_list:
            f.write(f'{smi}\n')
    f.close()
    return f.name


def _finetune_and_extract(
    train_smiles: List[str],
    val_smiles: List[str],
    train_labels: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fine-tune GROVER and extract fingerprints."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from argparse import Namespace

    device = get_device()
    grover_dir, checkpoint_path = _find_grover_paths()

    # Add GROVER to path
    if grover_dir not in sys.path:
        sys.path.insert(0, grover_dir)

    from grover.util.utils import load_checkpoint, get_data
    from grover.data import MolCollator, MoleculeDataset

    # Load pretrained model
    grover_args = Namespace(
        parser_name='fingerprint',
        fingerprint_source='both',
        cuda=(device == 'cuda'),
        features_path=None,
        no_features=True,
        checkpoint_paths=[checkpoint_path],
        max_data_size=None,
        use_compound_names=False,
        features_generator=None,
        features_scaling=False,
        no_cache=True,
        atom_messages=False,
        bond_drop_rate=0,
        dropout=0.0,
        ffn_num_layers=2,
        ffn_hidden_size=None,
    )

    import logging
    logger = logging.getLogger('grover_ft')
    logger.setLevel(logging.WARNING)

    # Patch torch.load for compatibility
    original_load = torch.load
    torch.load = _safe_torch_load
    try:
        model = load_checkpoint(checkpoint_path, cuda=(device == 'cuda'),
                                current_args=grover_args, logger=logger)
    finally:
        torch.load = original_load

    # Save initial state for reset between conceptual uses
    initial_state = copy.deepcopy(model.state_dict())

    # Determine fingerprint dim by running a small forward pass
    test_csv = _smiles_to_csv(train_smiles[:2])
    try:
        test_data = get_data(path=test_csv, args=grover_args,
                             use_compound_names=False, skip_invalid_smiles=False)
        collator = MolCollator(args=grover_args, shared_dict={})
        test_dataset = MoleculeDataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False,
                                 num_workers=0, collate_fn=collator)
        with torch.no_grad():
            for item in test_loader:
                _, batch_data, features_batch, _, _ = item
                test_fp = model(batch_data, features_batch)
                embed_dim = test_fp.shape[1]
                break
    finally:
        os.unlink(test_csv)

    # All params trainable + FFN readout head
    for param in model.parameters():
        param.requires_grad = True

    head = nn.Sequential(
        nn.Linear(embed_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 1),
    )

    if device == 'cuda':
        model = model.cuda()
        head = head.cuda()

    # Prepare training data
    train_csv = _smiles_to_csv(train_smiles, train_labels)
    try:
        train_data = get_data(path=train_csv, args=grover_args,
                              use_compound_names=False, skip_invalid_smiles=False)
    finally:
        os.unlink(train_csv)

    # Early stopping holdout
    n_train = len(train_smiles)
    n_holdout = max(1, int(0.1 * n_train))
    perm = np.random.RandomState(42).permutation(n_train)
    ho_idx = perm[:n_holdout]
    fit_idx = perm[n_holdout:]

    # Split data into fit and holdout
    fit_data = [train_data[i] for i in fit_idx]
    ho_data = [train_data[i] for i in ho_idx]
    ho_labels = train_labels[ho_idx]

    collator = MolCollator(args=grover_args, shared_dict={})
    fit_dataset = MoleculeDataset(fit_data)
    fit_loader = DataLoader(fit_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, collate_fn=collator)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=0.01,
    )
    criterion = nn.MSELoss()
    best_ho_loss = float('inf')
    patience = 2
    patience_counter = 0

    grover_args.bond_drop_rate = 0

    for epoch in range(epochs):
        model.train()
        head.train()
        for item in fit_loader:
            _, batch_data, features_batch, targets_batch, _ = item
            if device == 'cuda':
                targets = torch.FloatTensor(
                    [t[0] if isinstance(t, list) else t for t in targets_batch]
                ).cuda()
            else:
                targets = torch.FloatTensor(
                    [t[0] if isinstance(t, list) else t for t in targets_batch]
                )

            optimizer.zero_grad()
            fp = model(batch_data, features_batch)
            pred = head(fp).squeeze(-1)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()

        # Holdout check
        model.eval()
        head.eval()
        with torch.no_grad():
            ho_dataset = MoleculeDataset(ho_data)
            ho_loader = DataLoader(ho_dataset, batch_size=batch_size, shuffle=False,
                                   num_workers=0, collate_fn=collator)
            ho_preds = []
            for item in ho_loader:
                _, batch_data, features_batch, _, _ = item
                fp = model(batch_data, features_batch)
                pred = head(fp).squeeze(-1)
                ho_preds.extend(pred.cpu().numpy())
            ho_preds = np.array(ho_preds)
            ho_loss = float(np.mean((ho_preds - ho_labels) ** 2))

        if ho_loss < best_ho_loss:
            best_ho_loss = ho_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Extract fingerprints (head discarded)
    model.eval()

    def _extract(smiles_list):
        csv_path = _smiles_to_csv(smiles_list)
        try:
            data = get_data(path=csv_path, args=grover_args,
                            use_compound_names=False, skip_invalid_smiles=False)
        finally:
            os.unlink(csv_path)

        dataset = MoleculeDataset(data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collator)

        all_fps = []
        with torch.no_grad():
            for item in loader:
                _, batch_data, features_batch, _, _ = item
                fp = model(batch_data, features_batch)
                all_fps.extend(fp.data.cpu().numpy())
        return np.array(all_fps, dtype=np.float32)

    train_embs = _extract(train_smiles)
    val_embs = _extract(val_smiles)

    # Cleanup
    del model, head, optimizer
    if device == 'cuda':
        torch.cuda.empty_cache()

    return train_embs, val_embs


def make_molecular_grover_factory(
    smiles_list: List[str],
    dataset_name: str,
    epochs: int = 5,
    lr: float = 1e-4,
    batch_size: int = 32,
    cache: bool = True,
) -> Callable:
    """Create a GROVER fine-tuning factory.

    Fine-tunes GROVER on per-fold data, extracts atom+bond fingerprints (~5000D).

    Args:
        smiles_list: SMILES strings.
        dataset_name: Used for cache key.
        epochs: Fine-tuning epochs (default: 5).
        lr: Learning rate (default: 1e-4).
        batch_size: Batch size (default: 32).
        cache: Cache fold embeddings to disk.

    Returns:
        Factory callable: (train_idx, val_idx, train_labels) -> (train_embs, val_embs)
    """
    hyperparams = {
        'model': 'grover',
        'epochs': epochs,
        'lr': lr,
    }

    rep_name = 'grover_ft'

    def factory(
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        train_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        key = fold_cache_key(dataset_name, rep_name, train_idx, hyperparams, factory_version=_FACTORY_VERSION)

        def _compute():
            train_smi = [smiles_list[i] for i in train_idx]
            val_smi = [smiles_list[i] for i in val_idx]
            return _finetune_and_extract(
                train_smiles=train_smi,
                val_smiles=val_smi,
                train_labels=train_labels,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
            )

        return load_or_compute(key, _compute, cache=cache)

    return factory
