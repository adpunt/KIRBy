"""
Antibody ESM-2 LoRA fine-tuning factory.

Fine-tunes ESM-2 650M with LoRA on antibody VH sequences for fitness/property
prediction. Light chain sequences are embedded frozen and concatenated during
training to provide context, but only heavy chain embeddings are extracted.

Architecture during training:
    heavy_seq → LoRA-ESM-2 → heavy_emb (1280D)
    light_seq → frozen-ESM-2 → light_emb (1280D)
    concat(heavy_emb, light_emb) → regression_head → fitness

Extraction: heavy_emb (1280D) from LoRA-ESM-2 only.
Concatenation with light chain (or other reps) is handled downstream.

Requires: torch, transformers, peft
"""

from typing import Callable, List, Tuple

import numpy as np

from kirby.finetuning._base import (
    fold_cache_key,
    get_device,
    load_or_compute,
)

_MODEL_CONFIGS = {
    'esm2': (
        'facebook/esm2_t33_650M_UR50D',
        1280,
        ['query', 'value'],
    ),
}


def _finetune_and_extract(
    train_heavy: List[str],
    train_light: List[str],
    val_heavy: List[str],
    val_light: List[str],
    train_labels: np.ndarray,
    model_name: str,
    lora_rank: int,
    lora_alpha: int,
    epochs: int,
    lr: float,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fine-tune ESM-2 with LoRA on antibody VH, extract embeddings."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModel
    from peft import LoraConfig, get_peft_model, TaskType

    model_id, embed_dim, target_modules = _MODEL_CONFIGS[model_name]
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # LoRA model for heavy chain
    base_model = AutoModel.from_pretrained(model_id)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias='none',
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    peft_model = get_peft_model(base_model, lora_config)

    # Frozen model for light chain context
    frozen_model = AutoModel.from_pretrained(model_id)
    for p in frozen_model.parameters():
        p.requires_grad = False

    # Regression head: heavy_emb + light_emb → fitness
    head = nn.Sequential(
        nn.Linear(embed_dim * 2, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 1),
    )

    peft_model = peft_model.to(device)
    frozen_model = frozen_model.eval().to(device)
    head = head.to(device)

    def _tokenize(seqs, max_length=512):
        spaced = [' '.join(list(s)) for s in seqs]
        return tokenizer(
            spaced, padding=True, truncation=True,
            max_length=max_length, return_tensors='pt',
        )

    # Tokenize training sequences
    train_h_enc = _tokenize(train_heavy)
    train_l_enc = _tokenize(train_light)

    # Pre-compute frozen light chain embeddings for training
    frozen_model.eval()
    train_l_embs = []
    with torch.no_grad():
        for i in range(0, len(train_light), batch_size):
            j = min(i + batch_size, len(train_light))
            ids = train_l_enc['input_ids'][i:j].to(device)
            mask = train_l_enc['attention_mask'][i:j].to(device)
            out = frozen_model(input_ids=ids, attention_mask=mask)
            pooled = out.last_hidden_state.mean(dim=1)
            train_l_embs.append(pooled.cpu())
    train_l_tensor = torch.cat(train_l_embs, dim=0)  # (N_train, 1280)

    # Early stopping holdout
    n_train = len(train_heavy)
    n_holdout = max(1, int(0.1 * n_train))
    perm = np.random.RandomState(42).permutation(n_train)
    ho_idx = perm[:n_holdout]
    fit_idx = perm[n_holdout:]

    fit_dataset = TensorDataset(
        train_h_enc['input_ids'][fit_idx],
        train_h_enc['attention_mask'][fit_idx],
        train_l_tensor[fit_idx],
        torch.FloatTensor(train_labels[fit_idx]),
    )
    fit_loader = DataLoader(fit_dataset, batch_size=batch_size, shuffle=True)

    ho_h_ids = train_h_enc['input_ids'][ho_idx].to(device)
    ho_h_mask = train_h_enc['attention_mask'][ho_idx].to(device)
    ho_l_embs = train_l_tensor[ho_idx].to(device)
    ho_labels_t = torch.FloatTensor(train_labels[ho_idx]).to(device)

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
        for batch_h_ids, batch_h_mask, batch_l_emb, batch_y in fit_loader:
            batch_h_ids = batch_h_ids.to(device)
            batch_h_mask = batch_h_mask.to(device)
            batch_l_emb = batch_l_emb.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            out = peft_model(input_ids=batch_h_ids, attention_mask=batch_h_mask)
            h_emb = out.last_hidden_state.mean(dim=1)
            combined = torch.cat([h_emb, batch_l_emb], dim=1)
            pred = head(combined).squeeze(-1)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

        # Holdout check
        peft_model.eval()
        head.eval()
        with torch.no_grad():
            ho_out = peft_model(input_ids=ho_h_ids, attention_mask=ho_h_mask)
            ho_h_emb = ho_out.last_hidden_state.mean(dim=1)
            ho_combined = torch.cat([ho_h_emb, ho_l_embs], dim=1)
            ho_pred = head(ho_combined).squeeze(-1)
            ho_loss = criterion(ho_pred, ho_labels_t).item()

        if ho_loss < best_ho_loss:
            best_ho_loss = ho_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Extract heavy-chain embeddings only (LoRA-adapted)
    peft_model.eval()

    def _extract_heavy(seqs):
        enc = _tokenize(seqs)
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                j = min(i + batch_size, len(seqs))
                ids = enc['input_ids'][i:j].to(device)
                mask = enc['attention_mask'][i:j].to(device)
                out = peft_model(input_ids=ids, attention_mask=mask)
                pooled = out.last_hidden_state.mean(dim=1)
                all_embs.append(pooled.cpu().numpy())
        return np.concatenate(all_embs, axis=0).astype(np.float32)

    train_embs = _extract_heavy(train_heavy)
    val_embs = _extract_heavy(val_heavy)

    # Cleanup
    del peft_model, frozen_model, head, optimizer
    if device == 'cuda':
        torch.cuda.empty_cache()

    return train_embs, val_embs


def make_antibody_lora_factory(
    heavy_seqs: List[str],
    light_seqs: List[str],
    dataset_name: str,
    base_model: str = 'esm2',
    lora_rank: int = 8,
    lora_alpha: int = 16,
    epochs: int = 5,
    lr: float = 2e-5,
    batch_size: int = 8,
    cache: bool = True,
) -> Callable:
    """Create a LoRA fine-tuning factory for antibody ESM-2.

    Trains LoRA-adapted ESM-2 on heavy chain with frozen light chain context,
    then extracts LoRA-adapted heavy chain embeddings (1280D).

    Args:
        heavy_seqs: Heavy chain sequences.
        light_seqs: Light chain sequences (used as frozen context during training).
        dataset_name: Used for cache key.
        base_model: 'esm2' (ESM-2 650M).
        lora_rank: LoRA rank (default: 8).
        lora_alpha: LoRA alpha (default: 16).
        epochs: Fine-tuning epochs per fold (default: 5).
        lr: Learning rate (default: 2e-5).
        batch_size: Batch size (default: 8).
        cache: Cache fold embeddings to disk.

    Returns:
        Factory callable: (train_idx, val_idx, train_labels) -> (train_embs, val_embs)
        Output: 1280D heavy chain embeddings from LoRA-ESM-2.
    """
    if base_model not in _MODEL_CONFIGS:
        raise ValueError(f"Unknown base_model '{base_model}'. "
                         f"Available: {list(_MODEL_CONFIGS.keys())}")

    hyperparams = {
        'base_model': base_model,
        'lora_rank': lora_rank,
        'lora_alpha': lora_alpha,
        'epochs': epochs,
        'lr': lr,
    }

    rep_name = f'antibody_lora_{base_model}'

    def factory(
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        train_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        key = fold_cache_key(dataset_name, rep_name, train_idx, hyperparams)

        def _compute():
            train_h = [heavy_seqs[i] for i in train_idx]
            train_l = [light_seqs[i] for i in train_idx]
            val_h = [heavy_seqs[i] for i in val_idx]
            val_l = [light_seqs[i] for i in val_idx]

            return _finetune_and_extract(
                train_heavy=train_h,
                train_light=train_l,
                val_heavy=val_h,
                val_light=val_l,
                train_labels=train_labels,
                model_name=base_model,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
            )

        return load_or_compute(key, _compute, cache=cache)

    return factory
