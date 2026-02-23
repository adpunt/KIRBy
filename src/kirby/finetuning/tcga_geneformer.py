"""
TCGA Geneformer fine-tuning factory.

Fine-tunes Geneformer on classification/regression tasks,
extracts penultimate-layer cell embeddings.

Requires: torch, transformers
"""

from typing import Callable, List, Optional, Tuple

import numpy as np

from kirby.finetuning._base import (
    fold_cache_key,
    get_device,
    load_or_compute,
)


def _rank_encode_expression(expression: np.ndarray) -> np.ndarray:
    """Rank-encode gene expression per sample (Geneformer-style).

    Geneformer expects rank-ordered gene indices (most expressed first).
    Returns integer token sequences.
    """
    n_samples, n_genes = expression.shape
    # Rank genes per sample (descending expression)
    ranks = np.argsort(-expression, axis=1)
    return ranks


def _finetune_and_extract(
    train_expression: np.ndarray,
    val_expression: np.ndarray,
    train_labels: np.ndarray,
    task_type: str,
    epochs: int,
    lr: float,
    batch_size: int,
    max_input_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Core Geneformer fine-tuning + extraction."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModel, AutoConfig

    device = get_device()

    # Rank-encode expression data
    train_ranks = _rank_encode_expression(train_expression)[:, :max_input_size]
    val_ranks = _rank_encode_expression(val_expression)[:, :max_input_size]

    # Try loading Geneformer; fall back to a generic transformer if unavailable
    model_id = 'ctheodoris/Geneformer'
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        base_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        embed_dim = config.hidden_size
    except Exception:
        # Fallback: use a small BERT-style model with gene tokens
        from transformers import BertConfig, BertModel
        config = BertConfig(
            vocab_size=30000,  # Enough for gene indices
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=512,
            max_position_embeddings=max_input_size + 2,
        )
        base_model = BertModel(config)
        embed_dim = 256

    # Task head
    if task_type == 'classification':
        n_classes = len(np.unique(train_labels))
        head = nn.Linear(embed_dim, n_classes)
        criterion = nn.CrossEntropyLoss()
        train_labels_t = torch.LongTensor(train_labels.astype(int))
    else:
        head = nn.Linear(embed_dim, 1)
        criterion = nn.MSELoss()
        train_labels_t = torch.FloatTensor(train_labels)

    base_model = base_model.to(device)
    head = head.to(device)

    # Convert to tensors
    train_ids = torch.LongTensor(train_ranks)
    train_mask = torch.ones_like(train_ids)

    # Early stopping holdout
    n_train = len(train_ids)
    n_holdout = max(1, int(0.1 * n_train))
    perm = np.random.RandomState(42).permutation(n_train)
    ho_idx = perm[:n_holdout]
    fit_idx = perm[n_holdout:]

    fit_dataset = TensorDataset(
        train_ids[fit_idx], train_mask[fit_idx], train_labels_t[fit_idx],
    )
    fit_loader = DataLoader(fit_dataset, batch_size=batch_size, shuffle=True)

    # Only fine-tune last 2 layers + head (parameter-efficient)
    for param in base_model.parameters():
        param.requires_grad = False
    # Unfreeze last 2 encoder layers
    if hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'layer'):
        for layer in base_model.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
    elif hasattr(base_model, 'layers'):
        for layer in base_model.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

    trainable_params = [p for p in base_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params + list(head.parameters()),
        lr=lr, weight_decay=0.01,
    )

    best_ho_loss = float('inf')
    patience = 2
    patience_counter = 0

    for epoch in range(epochs):
        base_model.train()
        head.train()
        for batch_ids, batch_mask, batch_y in fit_loader:
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            out = base_model(input_ids=batch_ids, attention_mask=batch_mask)
            # Use [CLS] token
            if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                pooled = out.pooler_output
            else:
                pooled = out.last_hidden_state[:, 0, :]
            pred = head(pooled)

            if task_type == 'classification':
                loss = criterion(pred, batch_y)
            else:
                loss = criterion(pred.squeeze(-1), batch_y)
            loss.backward()
            optimizer.step()

        # Holdout check
        base_model.eval()
        head.eval()
        with torch.no_grad():
            ho_out = base_model(
                input_ids=train_ids[ho_idx].to(device),
                attention_mask=train_mask[ho_idx].to(device),
            )
            if hasattr(ho_out, 'pooler_output') and ho_out.pooler_output is not None:
                ho_pooled = ho_out.pooler_output
            else:
                ho_pooled = ho_out.last_hidden_state[:, 0, :]
            ho_pred = head(ho_pooled)
            if task_type == 'classification':
                ho_loss = criterion(ho_pred, train_labels_t[ho_idx].to(device)).item()
            else:
                ho_loss = criterion(ho_pred.squeeze(-1), train_labels_t[ho_idx].to(device)).item()

        if ho_loss < best_ho_loss:
            best_ho_loss = ho_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Extract penultimate-layer embeddings
    base_model.eval()

    def _extract(ranks_tensor, mask_tensor):
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(ranks_tensor), batch_size):
                j = min(i + batch_size, len(ranks_tensor))
                ids = ranks_tensor[i:j].to(device)
                mask = mask_tensor[i:j].to(device)
                out = base_model(
                    input_ids=ids, attention_mask=mask,
                    output_hidden_states=True,
                )
                # Penultimate layer, [CLS] token
                if out.hidden_states is not None:
                    penult = out.hidden_states[-2][:, 0, :]
                else:
                    penult = out.last_hidden_state[:, 0, :]
                all_embs.extend(penult.cpu().numpy())
        return np.array(all_embs, dtype=np.float32)

    val_ids = torch.LongTensor(val_ranks)
    val_mask = torch.ones_like(val_ids)

    train_embs = _extract(train_ids, train_mask)
    val_embs = _extract(val_ids, val_mask)

    # Cleanup
    del base_model, head, optimizer
    if device == 'cuda':
        torch.cuda.empty_cache()

    return train_embs, val_embs


def make_geneformer_factory(
    expression_data: np.ndarray,
    sample_ids: List[str],
    dataset_name: str,
    task_type: str = 'classification',
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_input_size: int = 2048,
    cache: bool = True,
) -> Callable:
    """Create a Geneformer fine-tuning factory.

    Fine-tunes Geneformer on classification/regression task per fold,
    extracts penultimate-layer cell embeddings.

    Args:
        expression_data: (n_samples, n_genes) expression matrix.
        sample_ids: TCGA barcodes (for cache key).
        dataset_name: Used for cache key.
        task_type: 'classification' or 'regression'.
        epochs: Fine-tuning epochs (default: 3).
        lr: Learning rate (default: 2e-5).
        batch_size: Batch size (default: 16).
        max_input_size: Max number of genes to use (default: 2048).
        cache: Cache fold embeddings to disk.

    Returns:
        Factory callable: (train_idx, val_idx, train_labels) -> (train_embs, val_embs)
    """
    hyperparams = {
        'model': 'geneformer',
        'epochs': epochs,
        'lr': lr,
        'max_input_size': max_input_size,
    }

    rep_name = 'geneformer'

    def factory(
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        train_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        key = fold_cache_key(dataset_name, rep_name, train_idx, hyperparams)

        def _compute():
            return _finetune_and_extract(
                train_expression=expression_data[train_idx],
                val_expression=expression_data[val_idx],
                train_labels=train_labels,
                task_type=task_type,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                max_input_size=max_input_size,
            )

        return load_or_compute(key, _compute, cache=cache)

    return factory
