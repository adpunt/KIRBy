"""
Chemprop D-MPNN fine-tuning factory.

Fine-tunes a Directed Message Passing Neural Network (D-MPNN) end-to-end,
then extracts 300D molecular embeddings from the learned message-passing
representation.

Architecture during training:
    SMILES → molecular graph → D-MPNN → FFN head → property
Extraction: 300D MPN fingerprint (message-passing output, before FFN).

Requires: chemprop (pip install chemprop==1.6.1)
"""

import os
import tempfile
from typing import Callable, List, Tuple

import numpy as np

from kirby.finetuning._base import (
    fold_cache_key,
    get_device,
    load_or_compute,
)


def _finetune_and_extract(
    train_smiles: List[str],
    val_smiles: List[str],
    train_labels: np.ndarray,
    hidden_size: int,
    depth: int,
    epochs: int,
    task_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fine-tune Chemprop D-MPNN and extract MPN fingerprints."""
    import torch
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, MoleculeDataLoader
    from chemprop.models import MoleculeModel
    from chemprop.args import TrainArgs

    device = get_device()

    # Write temporary CSV files (chemprop expects file-based input)
    tmpdir = tempfile.mkdtemp(prefix='kirby_chemprop_')

    train_csv = os.path.join(tmpdir, 'train.csv')
    with open(train_csv, 'w') as f:
        f.write('smiles,target\n')
        for smi, lbl in zip(train_smiles, train_labels):
            f.write(f'{smi},{lbl}\n')

    val_csv = os.path.join(tmpdir, 'val.csv')
    with open(val_csv, 'w') as f:
        f.write('smiles,target\n')
        for smi in val_smiles:
            f.write(f'{smi},0.0\n')  # dummy labels for val

    save_dir = os.path.join(tmpdir, 'model')

    # Build args
    dataset_type = 'classification' if task_type == 'classification' else 'regression'
    arg_list = [
        '--data_path', train_csv,
        '--dataset_type', dataset_type,
        '--save_dir', save_dir,
        '--hidden_size', str(hidden_size),
        '--depth', str(depth),
        '--epochs', str(epochs),
        '--batch_size', '64',
        '--quiet',
    ]
    if device == 'cpu':
        arg_list.append('--no_cuda')
    else:
        arg_list.extend(['--gpu', '0'])
    args = TrainArgs().parse_args(arg_list)

    # Train
    print(f"    Fine-tuning Chemprop D-MPNN ({hidden_size}D, depth={depth}, "
          f"epochs={epochs})...")

    # Use chemprop's training pipeline
    from chemprop.data.utils import get_data
    from chemprop.utils import build_optimizer, build_lr_scheduler
    from chemprop.nn_utils import param_count

    train_data = get_data(path=train_csv, args=args)
    args.train_data_size = len(train_data)

    # Build model
    model = MoleculeModel(args)
    model = model.to(device)
    print(f"      Model params: {param_count(model):,}")

    # Build optimizer
    optimizer = build_optimizer(model, args)
    scheduler = build_lr_scheduler(optimizer, args)

    # Train loop
    train_loader = MoleculeDataLoader(
        dataset=MoleculeDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
    )

    loss_fn = torch.nn.BCEWithLogitsLoss() if dataset_type == 'classification' else torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            mol_batch, features_batch, target_batch, *_ = batch
            targets = torch.tensor(
                [[t[0] if t[0] is not None else 0.0 for t in target_batch]],
                dtype=torch.float,
            ).T.to(device)

            optimizer.zero_grad()
            preds = model(mol_batch, features_batch)
            loss = loss_fn(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"      Epoch {epoch:3d}: loss={avg_loss:.4f}")

    # Extract fingerprints (MPN embeddings)
    model.eval()

    def _extract_fps(smiles_list):
        data = [MoleculeDatapoint(smiles=[smi]) for smi in smiles_list]
        dataset = MoleculeDataset(data)
        loader = MoleculeDataLoader(dataset=dataset, batch_size=64, shuffle=False)

        all_fps = []
        with torch.no_grad():
            for batch in loader:
                mol_batch, features_batch, *_ = batch
                fps = model.fingerprint(mol_batch, features_batch)
                all_fps.append(fps.cpu().numpy())

        return np.vstack(all_fps).astype(np.float32)

    train_embs = _extract_fps(train_smiles)
    val_embs = _extract_fps(val_smiles)

    print(f"    Extracted Chemprop embeddings: "
          f"train={train_embs.shape}, val={val_embs.shape}")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    del model, optimizer
    if device == 'cuda':
        torch.cuda.empty_cache()

    return train_embs, val_embs


def make_molecular_chemprop_factory(
    smiles_list: List[str],
    dataset_name: str,
    hidden_size: int = 300,
    depth: int = 3,
    epochs: int = 50,
    task_type: str = 'regression',
    cache: bool = True,
) -> Callable:
    """Create a Chemprop D-MPNN fine-tuning factory.

    Fine-tunes a D-MPNN per fold, then extracts hidden_size-D MPN fingerprints.

    Args:
        smiles_list: SMILES strings.
        dataset_name: Used for cache key.
        hidden_size: MPN hidden dimension (default: 300).
        depth: Message passing depth (default: 3).
        epochs: Training epochs (default: 50).
        task_type: 'regression' or 'classification'.
        cache: Cache fold embeddings to disk.

    Returns:
        Factory callable: (train_idx, val_idx, train_labels) -> (train_embs, val_embs)
        Output: (N, hidden_size) embeddings from D-MPNN.
    """
    hyperparams = {
        'hidden_size': hidden_size,
        'depth': depth,
        'epochs': epochs,
        'task_type': task_type,
    }

    rep_name = 'ft_chemprop'

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
                hidden_size=hidden_size,
                depth=depth,
                epochs=epochs,
                task_type=task_type,
            )

        return load_or_compute(key, _compute, cache=cache)

    return factory
