"""
Molecular LoRA fine-tuning factory.

Fine-tunes ChemBERTa / MolFormer with LoRA adapters on per-fold
training data, then extracts [CLS] or mean-pooled embeddings.

Requires: torch, transformers, peft
"""

from typing import Callable, List, Optional, Tuple

import numpy as np

from kirby.finetuning._base import (
    fold_cache_key,
    get_device,
    load_or_compute,
)

# Model configs: (HF model ID, embed_dim, target_modules)
_MODEL_CONFIGS = {
    'chemberta': (
        'DeepChem/ChemBERTa-77M-MLM',
        384,
        ['query', 'value'],
    ),
    'molformer': (
        'ibm/MoLFormer-XL-both-10pct',
        768,
        ['query', 'value'],
    ),
    'selformer': (
        'HUBioDataLab/SELFormer',
        768,
        ['query', 'value'],
    ),
    'gp_molformer': (
        'ibm-research/GP-MoLFormer-Uniq',
        768,
        ['query', 'value'],
    ),
    'chembert': (
        'jonghyunlee/ChemBERT_ChEMBL_pretrained',
        256,
        ['query', 'value'],
    ),
}


def _finetune_and_extract(
    train_smiles: List[str],
    val_smiles: List[str],
    train_labels: np.ndarray,
    model_name: str,
    lora_rank: int,
    lora_alpha: int,
    epochs: int,
    lr: float,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Core fine-tuning + extraction logic for molecular models."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModel
    from peft import LoraConfig, get_peft_model, TaskType

    model_id, embed_dim, target_modules = _MODEL_CONFIGS[model_name]
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

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

    # Tokenize
    def _tokenize(smiles_list, max_length=128):
        return tokenizer(
            smiles_list, padding=True, truncation=True,
            max_length=max_length, return_tensors='pt',
        )

    train_enc = _tokenize(train_smiles)

    # Early stopping holdout
    n_train = len(train_smiles)
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
            # Use [CLS] token (index 0) for BERT-style, or mean pool
            if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                pooled = out.pooler_output
            else:
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
            if hasattr(ho_out, 'pooler_output') and ho_out.pooler_output is not None:
                ho_pooled = ho_out.pooler_output
            else:
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

    # Extract embeddings
    peft_model.eval()

    def _extract(smiles_list):
        enc = _tokenize(smiles_list)
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(smiles_list), batch_size):
                j = min(i + batch_size, len(smiles_list))
                ids = enc['input_ids'][i:j].to(device)
                mask = enc['attention_mask'][i:j].to(device)
                out = peft_model(input_ids=ids, attention_mask=mask)
                if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                    pooled = out.pooler_output
                else:
                    pooled = out.last_hidden_state.mean(dim=1)
                all_embs.extend(pooled.cpu().numpy())
        return np.array(all_embs, dtype=np.float32)

    train_embs = _extract(train_smiles)
    val_embs = _extract(val_smiles)

    # Cleanup
    del peft_model, head, optimizer
    if device == 'cuda':
        import torch
        torch.cuda.empty_cache()

    return train_embs, val_embs


def make_molecular_lora_factory(
    smiles_list: List[str],
    dataset_name: str,
    base_model: str = 'chemberta',
    lora_rank: int = 8,
    lora_alpha: int = 16,
    epochs: int = 5,
    lr: float = 2e-4,
    batch_size: int = 16,
    cache: bool = True,
) -> Callable:
    """Create a LoRA fine-tuning factory for molecular language models.

    Args:
        smiles_list: SMILES strings.
        dataset_name: Used for cache key.
        base_model: Model key: 'chemberta', 'molformer'.
        lora_rank: LoRA rank (default: 8).
        lora_alpha: LoRA alpha (default: 16).
        epochs: Fine-tuning epochs per fold (default: 5).
        lr: Learning rate (default: 2e-4).
        batch_size: Batch size (default: 16).
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
        'lora_rank': lora_rank,
        'lora_alpha': lora_alpha,
        'epochs': epochs,
        'lr': lr,
    }

    rep_name = f'lora_{base_model}'

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
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
            )

        return load_or_compute(key, _compute, cache=cache)

    return factory
