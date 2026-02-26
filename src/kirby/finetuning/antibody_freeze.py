"""
Antibody Freeze+Head fine-tuning factory.

Freezes all backbone parameters, trains only a small regression head,
then extracts hidden-state embeddings (at mutation sites or mean-pooled).

Requires: torch, transformers
"""

from typing import Callable, List, Optional, Tuple

import numpy as np

from kirby.finetuning._base import (
    fold_cache_key,
    get_device,
    load_or_compute,
)

# Same model configs as antibody_lora (without LoRA target_modules)
_MODEL_CONFIGS = {
    'esm2': (
        'facebook/esm2_t33_650M_UR50D',
        'AutoTokenizer',
        'EsmModel',
        1280,
    ),
    'esm2_small': (
        'facebook/esm2_t6_8M_UR50D',
        'AutoTokenizer',
        'EsmModel',
        320,
    ),
    'ablang2': (
        'qilowoq/AbLang2',
        'AutoTokenizer',
        'AutoModel',
        480,
    ),
}


def _parse_mutation_positions(mutations: List[List], max_len: int) -> List[List[int]]:
    """Extract 0-indexed mutation positions from mutation lists.

    Each entry is a list of mutation dicts or strings like 'H:W98A'.
    Returns list of position indices per sample.
    """
    all_positions = []
    for muts in mutations:
        positions = []
        for m in muts:
            if isinstance(m, dict):
                pos = m.get('position', m.get('site', 0))
            elif isinstance(m, str) and ':' in m:
                try:
                    residue_part = m.split(':')[1]
                    pos = int(''.join(c for c in residue_part[1:-1] if c.isdigit()))
                except (IndexError, ValueError):
                    pos = 0
            else:
                pos = 0
            positions.append(min(pos + 1, max_len - 1))  # +1 for [CLS]
        if not positions:
            positions = [0]  # fallback: use [CLS]
        all_positions.append(positions)
    return all_positions


def _finetune_and_extract(
    train_seqs: List[str],
    val_seqs: List[str],
    train_labels: np.ndarray,
    train_mutations: List[List],
    val_mutations: List[List],
    model_name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    extract_at_mutations: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Freeze backbone, train head, extract embeddings."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModel

    config = _MODEL_CONFIGS[model_name]
    model_id, tok_cls, model_cls, embed_dim = config

    device = get_device()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load base model
    if model_cls == 'EsmModel':
        from transformers import EsmModel
        backbone = EsmModel.from_pretrained(model_id)
    else:
        backbone = AutoModel.from_pretrained(model_id)

    # Freeze ALL backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False

    # Trainable regression head
    head = nn.Sequential(
        nn.Linear(embed_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 1),
    )

    backbone = backbone.to(device)
    head = head.to(device)

    # Tokenize
    def _tokenize(seqs, max_length=256):
        return tokenizer(
            seqs, padding=True, truncation=True,
            max_length=max_length, return_tensors='pt',
        )

    train_enc = _tokenize(train_seqs)

    # Early stopping holdout
    n_train = len(train_seqs)
    n_holdout = max(1, int(0.1 * n_train))
    perm = np.random.RandomState(42).permutation(n_train)
    ho_idx = perm[:n_holdout]
    fit_idx = perm[n_holdout:]

    fit_dataset = TensorDataset(
        train_enc['input_ids'][fit_idx],
        train_enc['attention_mask'][fit_idx],
        torch.FloatTensor(train_labels[fit_idx]),
    )
    fit_loader = DataLoader(fit_dataset, batch_size=batch_size, shuffle=True)

    ho_ids = train_enc['input_ids'][ho_idx].to(device)
    ho_mask = train_enc['attention_mask'][ho_idx].to(device)
    ho_labels_t = torch.FloatTensor(train_labels[ho_idx]).to(device)

    # Only train head (backbone frozen)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()
    best_ho_loss = float('inf')
    patience = 2
    patience_counter = 0

    for epoch in range(epochs):
        backbone.eval()  # backbone stays in eval
        head.train()
        for batch_ids, batch_mask, batch_y in fit_loader:
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                out = backbone(input_ids=batch_ids, attention_mask=batch_mask)
            pooled = out.last_hidden_state.mean(dim=1)
            pred = head(pooled).squeeze(-1)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

        # Holdout check
        head.eval()
        with torch.no_grad():
            ho_out = backbone(input_ids=ho_ids, attention_mask=ho_mask)
            ho_pooled = ho_out.last_hidden_state.mean(dim=1)
            ho_pred = head(ho_pooled).squeeze(-1)
            ho_loss = criterion(ho_pred, ho_labels_t).item()

        if ho_loss < best_ho_loss:
            best_ho_loss = ho_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Extract embeddings (head is discarded)
    backbone.eval()

    def _extract(seqs, mutation_list):
        enc = _tokenize(seqs)
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                j = min(i + batch_size, len(seqs))
                ids = enc['input_ids'][i:j].to(device)
                mask = enc['attention_mask'][i:j].to(device)
                out = backbone(input_ids=ids, attention_mask=mask)
                hidden = out.last_hidden_state

                if extract_at_mutations and mutation_list is not None:
                    batch_muts = mutation_list[i:j]
                    positions = _parse_mutation_positions(
                        batch_muts, hidden.shape[1]
                    )
                    for k, pos_list in enumerate(positions):
                        site_embs = hidden[k, pos_list, :]
                        all_embs.append(site_embs.mean(dim=0).cpu().numpy())
                else:
                    pooled = hidden.mean(dim=1)
                    all_embs.extend(pooled.cpu().numpy())

        return np.array(all_embs, dtype=np.float32)

    train_embs = _extract(train_seqs, train_mutations)
    val_embs = _extract(val_seqs, val_mutations)

    # Cleanup
    del backbone, head, optimizer
    if device == 'cuda':
        torch.cuda.empty_cache()

    return train_embs, val_embs


def make_antibody_freeze_factory(
    heavy_seqs: List[str],
    light_seqs: Optional[List[str]],
    mutations: Optional[List[List]],
    dataset_name: str,
    base_model: str = 'esm2',
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 8,
    extract_at_mutations: bool = True,
    cache: bool = True,
) -> Callable:
    """Create a Freeze+Head fine-tuning factory for antibody LMs.

    Freezes backbone, trains a regression head, then extracts
    hidden-state embeddings (at mutation sites or mean-pooled).

    Args:
        heavy_seqs: Heavy chain sequences.
        light_seqs: Light chain sequences (optional, concatenated with heavy).
        mutations: Per-sample mutation lists.
        dataset_name: Used for cache key.
        base_model: Model key: 'esm2', 'esm2_small', 'ablang2'.
        epochs: Training epochs per fold (default: 5).
        lr: Learning rate for head (default: 1e-3).
        batch_size: Batch size (default: 8).
        extract_at_mutations: If True, extract at mutation positions.
        cache: Cache fold embeddings to disk.

    Returns:
        Factory callable: (train_idx, val_idx, train_labels) -> (train_embs, val_embs)
    """
    if base_model not in _MODEL_CONFIGS:
        raise ValueError(
            f"Unknown base_model '{base_model}'. "
            f"Available: {list(_MODEL_CONFIGS.keys())}"
        )

    # Prepare sequences
    if light_seqs is not None:
        seqs = [h + l for h, l in zip(heavy_seqs, light_seqs)]
    else:
        seqs = list(heavy_seqs)

    has_mutations = mutations is not None and any(len(m) > 0 for m in mutations)
    actual_extract_at_mutations = extract_at_mutations and has_mutations

    hyperparams = {
        'base_model': base_model,
        'epochs': epochs,
        'lr': lr,
        'extract_at_mutations': actual_extract_at_mutations,
    }

    rep_name = f'freeze_{base_model}'

    def factory(
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        train_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        key = fold_cache_key(dataset_name, rep_name, train_idx, hyperparams)

        def _compute():
            train_seqs = [seqs[i] for i in train_idx]
            val_seqs = [seqs[i] for i in val_idx]
            train_muts = [mutations[i] for i in train_idx] if has_mutations else None
            val_muts = [mutations[i] for i in val_idx] if has_mutations else None

            return _finetune_and_extract(
                train_seqs=train_seqs,
                val_seqs=val_seqs,
                train_labels=train_labels,
                train_mutations=train_muts,
                val_mutations=val_muts,
                model_name=base_model,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                extract_at_mutations=actual_extract_at_mutations,
            )

        return load_or_compute(key, _compute, cache=cache)

    return factory
