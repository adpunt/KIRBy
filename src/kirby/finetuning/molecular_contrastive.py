"""
Molecular contrastive fine-tuning factory.

Fine-tunes ChemBERTa / MolFormer with LoRA adapters using supervised
contrastive loss (no regression head). Pairs with similar labels are
treated as positives.

After training, extracts mean-pooled backbone embeddings.

Requires: torch, transformers, peft
"""

from typing import Callable, List, Tuple

import numpy as np

from kirby.finetuning._base import (
    fold_cache_key,
    get_device,
    load_or_compute,
)

# Same model configs as molecular_lora
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
}


def _supervised_contrastive_loss(embeddings, labels, temperature=0.1, threshold=0.3):
    """Supervised contrastive loss for regression labels.

    Pairs whose normalized label difference is < threshold are positives.

    Args:
        embeddings: (batch, embed_dim) L2-normalized embeddings
        labels: (batch,) regression labels
        temperature: softmax temperature
        threshold: label similarity threshold for positive pairs

    Returns:
        scalar loss
    """
    import torch
    import torch.nn.functional as F

    batch_size = embeddings.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=embeddings.device)

    # L2-normalize
    embeddings = F.normalize(embeddings, dim=1)

    # Similarity matrix
    sim = torch.mm(embeddings, embeddings.t()) / temperature

    # Label distance matrix -> positive mask
    label_diff = torch.abs(labels.unsqueeze(0) - labels.unsqueeze(1))
    # Normalize by label range
    label_range = labels.max() - labels.min()
    if label_range > 0:
        label_diff = label_diff / label_range
    positive_mask = (label_diff < threshold).float()

    # Remove diagonal
    eye = torch.eye(batch_size, device=embeddings.device)
    positive_mask = positive_mask * (1 - eye)

    # If no positive pairs exist, skip
    if positive_mask.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)

    # Log-sum-exp for denominator (all pairs except self)
    neg_mask = 1 - eye
    exp_sim = torch.exp(sim) * neg_mask
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # Average log-prob of positive pairs
    log_prob = sim - log_denom
    mean_log_prob = (positive_mask * log_prob).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-8)

    loss = -mean_log_prob.mean()
    return loss


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
    temperature: float,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Contrastive fine-tuning + extraction."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModel
    from peft import LoraConfig, get_peft_model, TaskType

    model_id, embed_dim, target_modules = _MODEL_CONFIGS[model_name]
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

    # Apply LoRA (same as molecular_lora)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias='none',
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    peft_model = get_peft_model(base_model, lora_config)

    # Projection head for contrastive loss (discarded after training)
    proj_head = nn.Sequential(
        nn.Linear(embed_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )

    peft_model = peft_model.to(device)
    proj_head = proj_head.to(device)

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
    # Larger batch sizes help contrastive learning (more pairs)
    effective_bs = min(batch_size * 2, len(fit_idx))
    fit_loader = DataLoader(fit_dataset, batch_size=effective_bs, shuffle=True)

    ho_ids = train_enc['input_ids'][ho_idx].to(device)
    ho_mask = train_enc['attention_mask'][ho_idx].to(device)
    ho_labels_t = torch.FloatTensor(train_labels[ho_idx]).to(device)

    optimizer = torch.optim.AdamW(
        list(peft_model.parameters()) + list(proj_head.parameters()),
        lr=lr, weight_decay=0.01,
    )
    best_ho_loss = float('inf')
    patience = 2
    patience_counter = 0

    def _pool(out):
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state.mean(dim=1)

    for epoch in range(epochs):
        peft_model.train()
        proj_head.train()
        for batch_ids, batch_mask, batch_y in fit_loader:
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            out = peft_model(input_ids=batch_ids, attention_mask=batch_mask)
            pooled = _pool(out)
            projected = proj_head(pooled)
            loss = _supervised_contrastive_loss(
                projected, batch_y, temperature=temperature, threshold=threshold
            )
            loss.backward()
            optimizer.step()

        # Holdout check
        peft_model.eval()
        proj_head.eval()
        with torch.no_grad():
            ho_out = peft_model(input_ids=ho_ids, attention_mask=ho_mask)
            ho_pooled = _pool(ho_out)
            ho_projected = proj_head(ho_pooled)
            ho_loss = _supervised_contrastive_loss(
                ho_projected, ho_labels_t, temperature=temperature, threshold=threshold
            ).item()

        if ho_loss < best_ho_loss:
            best_ho_loss = ho_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Extract backbone embeddings (projection head discarded)
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
                pooled = _pool(out)
                all_embs.extend(pooled.cpu().numpy())
        return np.array(all_embs, dtype=np.float32)

    train_embs = _extract(train_smiles)
    val_embs = _extract(val_smiles)

    # Cleanup
    del peft_model, proj_head, optimizer
    if device == 'cuda':
        import torch
        torch.cuda.empty_cache()

    return train_embs, val_embs


def make_molecular_contrastive_factory(
    smiles_list: List[str],
    dataset_name: str,
    base_model: str = 'chemberta',
    lora_rank: int = 8,
    lora_alpha: int = 16,
    epochs: int = 5,
    lr: float = 2e-4,
    batch_size: int = 16,
    temperature: float = 0.1,
    threshold: float = 0.3,
    cache: bool = True,
) -> Callable:
    """Create a contrastive fine-tuning factory for molecular LMs.

    Uses LoRA adapters with supervised contrastive loss (no regression head).
    Pairs with similar labels (normalized diff < threshold) are positives.
    After training, extracts mean-pooled backbone embeddings.

    Args:
        smiles_list: SMILES strings.
        dataset_name: Used for cache key.
        base_model: 'chemberta' or 'molformer'.
        lora_rank: LoRA rank (default: 8).
        lora_alpha: LoRA alpha (default: 16).
        epochs: Training epochs (default: 5).
        lr: Learning rate (default: 2e-4).
        batch_size: Batch size (default: 16).
        temperature: Contrastive temperature (default: 0.1).
        threshold: Label similarity threshold for positives (default: 0.3).
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
        'temperature': temperature,
        'threshold': threshold,
    }

    rep_name = f'contrastive_{base_model}'

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
                temperature=temperature,
                threshold=threshold,
            )

        return load_or_compute(key, _compute, cache=cache)

    return factory
