"""
Antibody LoRA fine-tuning factory.

Fine-tunes ESM-2 / AntiBERTy / AbLang2 with LoRA adapters on per-fold
training data, then extracts hidden-state embeddings at mutation positions.

Requires: torch, transformers, peft
"""

import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from kirby.finetuning._base import (
    fold_cache_key,
    get_device,
    load_or_compute,
)

# Model configs: (HF model ID, tokenizer class, model class, embed_dim, target_modules)
_MODEL_CONFIGS = {
    'esm2': (
        'facebook/esm2_t33_650M_UR50D',
        'AutoTokenizer',
        'EsmModel',
        1280,
        ['query', 'value'],
    ),
    'esm2_small': (
        'facebook/esm2_t6_8M_UR50D',
        'AutoTokenizer',
        'EsmModel',
        320,
        ['query', 'value'],
    ),
    'antiberty': (
        'alchemab/antiberta2',
        'RoFormerTokenizer',
        'RoFormerModel',
        512,
        ['query', 'value'],
    ),
    'ablang2': (
        'qilowoq/AbLang2',
        'AutoTokenizer',
        'AutoModel',
        480,
        ['query', 'value'],
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
                # Format: 'H:W98A' -> position 98
                try:
                    residue_part = m.split(':')[1]
                    pos = int(''.join(c for c in residue_part[1:-1] if c.isdigit()))
                except (IndexError, ValueError):
                    pos = 0
            else:
                pos = 0
            # Clamp to valid token range (account for special tokens)
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
    lora_rank: int,
    lora_alpha: int,
    epochs: int,
    lr: float,
    batch_size: int,
    extract_at_mutations: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Core fine-tuning + extraction logic."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModel
    from peft import LoraConfig, get_peft_model, TaskType

    config = _MODEL_CONFIGS[model_name]
    model_id, tok_cls, model_cls, embed_dim, target_modules = config

    device = get_device()

    # Load tokenizer
    if tok_cls == 'RoFormerTokenizer':
        from transformers import RoFormerTokenizer
        tokenizer = RoFormerTokenizer.from_pretrained(model_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load base model
    if model_cls == 'EsmModel':
        from transformers import EsmModel
        base_model = EsmModel.from_pretrained(model_id)
    elif model_cls == 'RoFormerModel':
        from transformers import RoFormerModel
        base_model = RoFormerModel.from_pretrained(model_id)
    else:
        base_model = AutoModel.from_pretrained(model_id)

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias='none',
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    peft_model = get_peft_model(base_model, lora_config)

    # Regression head
    head = nn.Sequential(
        nn.Linear(embed_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 1),
    )

    peft_model = peft_model.to(device)
    head = head.to(device)

    # Tokenize sequences
    def _tokenize(seqs, max_length=256):
        # Space-separate for protein tokenizers that expect it
        if tok_cls == 'RoFormerTokenizer':
            seqs = [' '.join(list(s)) for s in seqs]
        return tokenizer(
            seqs, padding=True, truncation=True,
            max_length=max_length, return_tensors='pt',
        )

    train_enc = _tokenize(train_seqs)
    max_seq_len = train_enc['input_ids'].shape[1]

    # Early stopping: hold out 10% of train
    n_train = len(train_seqs)
    n_holdout = max(1, int(0.1 * n_train))
    perm = np.random.RandomState(42).permutation(n_train)
    ho_idx = perm[:n_holdout]
    fit_idx = perm[n_holdout:]

    fit_ids = train_enc['input_ids'][fit_idx]
    fit_mask = train_enc['attention_mask'][fit_idx]
    fit_labels_t = torch.FloatTensor(train_labels[fit_idx])

    ho_ids = train_enc['input_ids'][ho_idx]
    ho_mask = train_enc['attention_mask'][ho_idx]
    ho_labels_t = torch.FloatTensor(train_labels[ho_idx])

    fit_dataset = TensorDataset(fit_ids, fit_mask, fit_labels_t)
    fit_loader = DataLoader(fit_dataset, batch_size=batch_size, shuffle=True)

    # Train
    optimizer = torch.optim.AdamW(
        list(peft_model.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=0.01,
    )
    criterion = nn.MSELoss()
    best_ho_loss = float('inf')
    patience = 2
    patience_counter = 0

    for epoch in range(epochs):
        peft_model.train()
        head.train()
        for batch_ids, batch_mask, batch_y in fit_loader:
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            out = peft_model(input_ids=batch_ids, attention_mask=batch_mask)
            pooled = out.last_hidden_state.mean(dim=1)  # mean pool
            pred = head(pooled).squeeze(-1)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

        # Holdout check
        peft_model.eval()
        head.eval()
        with torch.no_grad():
            ho_out = peft_model(
                input_ids=ho_ids.to(device),
                attention_mask=ho_mask.to(device),
            )
            ho_pooled = ho_out.last_hidden_state.mean(dim=1)
            ho_pred = head(ho_pooled).squeeze(-1)
            ho_loss = criterion(ho_pred, ho_labels_t.to(device)).item()

        if ho_loss < best_ho_loss:
            best_ho_loss = ho_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Extract embeddings
    peft_model.eval()

    def _extract(seqs, mutation_list):
        enc = _tokenize(seqs)
        all_embs = []
        with torch.no_grad():
            # Process in batches
            for i in range(0, len(seqs), batch_size):
                j = min(i + batch_size, len(seqs))
                ids = enc['input_ids'][i:j].to(device)
                mask = enc['attention_mask'][i:j].to(device)
                out = peft_model(input_ids=ids, attention_mask=mask)
                hidden = out.last_hidden_state  # (batch, seq_len, embed_dim)

                if extract_at_mutations and mutation_list is not None:
                    # Extract at mutation positions
                    batch_muts = mutation_list[i:j]
                    positions = _parse_mutation_positions(
                        batch_muts, hidden.shape[1]
                    )
                    for k, pos_list in enumerate(positions):
                        site_embs = hidden[k, pos_list, :]  # (n_sites, embed_dim)
                        all_embs.append(site_embs.mean(dim=0).cpu().numpy())
                else:
                    # Mean pool
                    pooled = hidden.mean(dim=1)  # (batch, embed_dim)
                    all_embs.extend(pooled.cpu().numpy())

        return np.array(all_embs, dtype=np.float32)

    train_embs = _extract(train_seqs, train_mutations)
    val_embs = _extract(val_seqs, val_mutations)

    # Cleanup GPU memory
    del peft_model, head, optimizer
    if device == 'cuda':
        torch.cuda.empty_cache()

    return train_embs, val_embs


def make_antibody_lora_factory(
    heavy_seqs: List[str],
    light_seqs: Optional[List[str]],
    mutations: Optional[List[List]],
    dataset_name: str,
    base_model: str = 'esm2',
    lora_rank: int = 8,
    lora_alpha: int = 16,
    epochs: int = 5,
    lr: float = 2e-4,
    batch_size: int = 8,
    extract_at_mutations: bool = True,
    cache: bool = True,
) -> Callable:
    """Create a LoRA fine-tuning factory for antibody language models.

    The returned factory is a closure over the raw sequences and mutations.
    When called with (train_idx, val_idx, train_labels), it:
    1. Fine-tunes the LM with LoRA on the training fold
    2. Extracts hidden-state embeddings (at mutation sites or mean-pooled)
    3. Returns (train_embs, val_embs) as numpy arrays

    Args:
        heavy_seqs: Heavy chain sequences.
        light_seqs: Light chain sequences (optional, concatenated with heavy).
        mutations: Per-sample mutation lists (for site-specific extraction).
        dataset_name: Used for cache key.
        base_model: Model key: 'esm2', 'esm2_small', 'antiberty', 'ablang2'.
        lora_rank: LoRA rank (default: 8).
        lora_alpha: LoRA alpha (default: 16).
        epochs: Fine-tuning epochs per fold (default: 5).
        lr: Learning rate (default: 2e-4).
        batch_size: Batch size (default: 8).
        extract_at_mutations: If True, extract at mutation positions; else mean-pool.
        cache: Cache fold embeddings to disk.

    Returns:
        Factory callable: (train_idx, val_idx, train_labels) -> (train_embs, val_embs)
    """
    if base_model not in _MODEL_CONFIGS:
        raise ValueError(
            f"Unknown base_model '{base_model}'. "
            f"Available: {list(_MODEL_CONFIGS.keys())}"
        )

    # Prepare sequences (concat heavy+light if available)
    if light_seqs is not None:
        seqs = [h + l for h, l in zip(heavy_seqs, light_seqs)]
    else:
        seqs = list(heavy_seqs)

    has_mutations = mutations is not None and any(len(m) > 0 for m in mutations)
    actual_extract_at_mutations = extract_at_mutations and has_mutations

    hyperparams = {
        'base_model': base_model,
        'lora_rank': lora_rank,
        'lora_alpha': lora_alpha,
        'epochs': epochs,
        'lr': lr,
        'extract_at_mutations': actual_extract_at_mutations,
    }

    rep_name = f'lora_{base_model}'

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
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                extract_at_mutations=actual_extract_at_mutations,
            )

        return load_or_compute(key, _compute, cache=cache)

    return factory
