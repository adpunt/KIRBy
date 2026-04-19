"""CGR-SMILES LoRA fine-tuning factory.

Fine-tunes the CGR-SMILES pretrained ChemBERTa model with LoRA adapters
on per-fold reaction yield data, then extracts mean-pooled embeddings.

Uses the MLM-pretrained checkpoint from cgr_smiles_pretrain as base.
Falls back to base ChemBERTa + extended tokenizer if no checkpoint found.

Follows molecular_lora.py pattern exactly.

Requires: torch, transformers, peft, chython, rxnmapper
"""

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


def _finetune_and_extract(
    train_cgr_smiles: List[str],
    val_cgr_smiles: List[str],
    train_labels: np.ndarray,
    model_path: str,
    embed_dim: int,
    lora_rank: int,
    lora_alpha: int,
    epochs: int,
    lr: float,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Core LoRA fine-tuning + extraction for CGR-SMILES model."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModel
    from peft import LoraConfig, get_peft_model, TaskType

    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModel.from_pretrained(model_path)

    # If using base ChemBERTa, extend tokenizer + resize
    if 'ChemBERTa' in model_path and 'cgr_smiles' not in model_path:
        from kirby.finetuning.cgr_smiles_pretrain import get_dynamic_bond_tokens
        new_tokens = get_dynamic_bond_tokens()
        existing = set(tokenizer.get_vocab().keys())
        to_add = [t for t in new_tokens if t not in existing]
        if to_add:
            tokenizer.add_special_tokens({'additional_special_tokens': to_add})
            base_model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=['query', 'value'],
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

    # Tokenize
    def _tokenize(texts, max_length=256):
        return tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors='pt',
        )

    # Filter empty CGR-SMILES (use a placeholder to maintain indices)
    train_texts = [s if s else '[PAD]' for s in train_cgr_smiles]
    train_enc = _tokenize(train_texts)

    # Early stopping holdout
    n_train = len(train_texts)
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

    # Train
    optimizer = torch.optim.AdamW(
        list(peft_model.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=0.01,
    )
    criterion = nn.MSELoss()
    best_ho_loss = float('inf')
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
            pooled = out.last_hidden_state.mean(dim=1)
            pred = head(pooled).squeeze(-1)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

        # Holdout check
        peft_model.eval()
        head.eval()
        with torch.no_grad():
            ho_out = peft_model(input_ids=ho_ids, attention_mask=ho_mask)
            ho_pooled = ho_out.last_hidden_state.mean(dim=1)
            ho_pred = head(ho_pooled).squeeze(-1)
            ho_loss = criterion(ho_pred, ho_labels_t).item()

        if ho_loss < best_ho_loss:
            best_ho_loss = ho_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 2:
                break

    # Extract embeddings
    peft_model.eval()

    def _extract(texts):
        texts = [s if s else '[PAD]' for s in texts]
        enc = _tokenize(texts)
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                j = min(i + batch_size, len(texts))
                ids = enc['input_ids'][i:j].to(device)
                mask = enc['attention_mask'][i:j].to(device)
                out = peft_model(input_ids=ids, attention_mask=mask)
                pooled = out.last_hidden_state.mean(dim=1)
                all_embs.extend(pooled.cpu().numpy())
        return np.array(all_embs, dtype=np.float32)

    train_embs = _extract(train_cgr_smiles)
    val_embs = _extract(val_cgr_smiles)

    # Cleanup
    del peft_model, head, optimizer
    if str(device) == 'cuda':
        torch.cuda.empty_cache()

    return train_embs, val_embs


def make_cgr_smiles_lora_factory(
    rxn_smiles: List[str],
    dataset_name: str,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    epochs: int = 5,
    lr: float = 2e-4,
    batch_size: int = 16,
    cache: bool = True,
) -> Callable:
    """Create a LoRA fine-tuning factory for CGR-SMILES embeddings.

    Pre-converts all rxn_smiles to CGR-SMILES ONCE (amortized), then
    returns a factory that fine-tunes per fold.

    Args:
        rxn_smiles: Reaction SMILES (will be converted to CGR-SMILES).
        dataset_name: Used for cache key.
        lora_rank: LoRA rank (default 8).
        lora_alpha: LoRA alpha (default 16).
        epochs: Max training epochs (default 5).
        lr: Learning rate (default 2e-4).
        batch_size: Training batch size (default 16).
        cache: Whether to cache per-fold results (default True).

    Returns:
        Factory callable: (train_idx, val_idx, train_labels) -> (train_embs, val_embs)
    """
    from kirby.representations.reaction import (
        _rxn_to_cgr_smiles_batch, _find_cgr_smiles_checkpoint,
    )

    # Pre-convert all reactions to CGR-SMILES (amortized)
    print(f"  CGR-SMILES LoRA: pre-converting {len(rxn_smiles)} reactions...")
    cgr_smiles = _rxn_to_cgr_smiles_batch(rxn_smiles)

    # Find model checkpoint
    model_path = _find_cgr_smiles_checkpoint()
    embed_dim = 384

    hyperparams = {
        'lora_rank': lora_rank,
        'lora_alpha': lora_alpha,
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'model_path': model_path,
    }

    def factory(train_idx, val_idx, train_labels):
        train_cgr = [cgr_smiles[i] for i in train_idx]
        val_cgr = [cgr_smiles[i] for i in val_idx]

        if cache:
            cache_key = fold_cache_key(
                dataset_name, 'cgr_smiles_lora', train_idx, hyperparams, factory_version=_FACTORY_VERSION)
            return load_or_compute(
                cache_key,
                lambda: _finetune_and_extract(
                    train_cgr, val_cgr, train_labels,
                    model_path, embed_dim,
                    lora_rank, lora_alpha, epochs, lr, batch_size),
            )
        return _finetune_and_extract(
            train_cgr, val_cgr, train_labels,
            model_path, embed_dim,
            lora_rank, lora_alpha, epochs, lr, batch_size)

    return factory
