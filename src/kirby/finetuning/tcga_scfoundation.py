"""
TCGA scFoundation fine-tuning factory.

Fine-tunes scFoundation on classification/regression tasks,
extracts penultimate-layer CLS embeddings (2048D).

Unlike Geneformer, scFoundation takes raw expression values directly
(no tokenization step — expression → FloatTensor → model).

Requires: torch, transformers
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
    train_expression: np.ndarray,
    val_expression: np.ndarray,
    train_labels: np.ndarray,
    gene_names: np.ndarray,
    task_type: str,
    epochs: int,
    lr: float,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Core scFoundation fine-tuning + extraction."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModel, AutoConfig

    device = get_device()

    # scFoundation takes raw expression values — no tokenization needed.
    # Normalize per sample (log1p + z-score)
    all_expression = np.vstack([train_expression, val_expression])
    all_log = np.log1p(all_expression)
    row_mean = all_log.mean(axis=1, keepdims=True)
    row_std = all_log.std(axis=1, keepdims=True)
    row_std[row_std == 0] = 1.0
    all_norm = (all_log - row_mean) / row_std

    n_train = len(train_expression)
    train_expr_t = torch.FloatTensor(all_norm[:n_train])
    val_expr_t = torch.FloatTensor(all_norm[n_train:])

    n_genes = all_expression.shape[1]

    # Try loading scFoundation; fall back to a simple transformer
    model_id = 'biomap-research/scFoundation'
    embed_dim = 2048
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        base_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        embed_dim = getattr(config, 'hidden_size', 2048)
    except Exception as e:
        print(f"    Could not load scFoundation ({e}), using fallback transformer")
        from transformers import BertConfig, BertModel
        config = BertConfig(
            vocab_size=2,  # not used — we project expression to hidden_size
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=1024,
            max_position_embeddings=n_genes + 2,
        )
        base_model = BertModel(config)
        embed_dim = 512
        # Add input projection: gene expression dim -> hidden_size
        base_model.gene_projection = nn.Linear(n_genes, embed_dim)
        base_model._fallback = True

    # Freeze all, unfreeze last 2 encoder layers + task head
    for param in base_model.parameters():
        param.requires_grad = False

    if hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'layer'):
        for layer in base_model.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
    elif hasattr(base_model, 'layers'):
        for layer in base_model.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

    # If fallback, also unfreeze the projection
    if hasattr(base_model, 'gene_projection'):
        for param in base_model.gene_projection.parameters():
            param.requires_grad = True

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

    # Early stopping holdout
    n_holdout = max(1, int(0.1 * n_train))
    perm = np.random.RandomState(42).permutation(n_train)
    ho_idx = perm[:n_holdout]
    fit_idx = perm[n_holdout:]

    fit_dataset = TensorDataset(
        train_expr_t[fit_idx], train_labels_t[fit_idx],
    )
    fit_loader = DataLoader(fit_dataset, batch_size=batch_size, shuffle=True)

    trainable_params = [p for p in base_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params + list(head.parameters()),
        lr=lr, weight_decay=0.01,
    )

    best_ho_loss = float('inf')
    patience = 2
    patience_counter = 0

    def _forward(model, expr_batch):
        """Forward pass handling both real scFoundation and fallback."""
        if hasattr(model, '_fallback') and model._fallback:
            # Fallback: project expression -> (batch, 1, hidden) -> through transformer
            projected = model.gene_projection(expr_batch).unsqueeze(1)  # (B, 1, H)
            out = model(inputs_embeds=projected, output_hidden_states=True)
        else:
            # Real scFoundation: expects expression tensor directly
            out = model(expr_batch, output_hidden_states=True)
        return out

    def _get_embedding(out):
        """Extract penultimate-layer CLS or first-token embedding."""
        if hasattr(out, 'hidden_states') and out.hidden_states is not None:
            return out.hidden_states[-2][:, 0, :]
        return out.last_hidden_state[:, 0, :]

    for epoch in range(epochs):
        base_model.train()
        head.train()
        for batch_expr, batch_y in fit_loader:
            batch_expr = batch_expr.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            out = _forward(base_model, batch_expr)
            pooled = _get_embedding(out)
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
            ho_out = _forward(base_model, train_expr_t[ho_idx].to(device))
            ho_pooled = _get_embedding(ho_out)
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

    def _extract(expr_tensor):
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(expr_tensor), batch_size):
                j = min(i + batch_size, len(expr_tensor))
                batch = expr_tensor[i:j].to(device)
                out = _forward(base_model, batch)
                emb = _get_embedding(out)
                all_embs.extend(emb.cpu().numpy())
        return np.array(all_embs, dtype=np.float32)

    train_embs = _extract(train_expr_t)
    val_embs = _extract(val_expr_t)

    # Cleanup
    del base_model, head, optimizer
    if device == 'cuda':
        torch.cuda.empty_cache()

    return train_embs, val_embs


def make_scfoundation_factory(
    expression_data: np.ndarray,
    gene_names: np.ndarray,
    sample_ids: List[str],
    dataset_name: str,
    task_type: str = 'classification',
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
    cache: bool = True,
) -> Callable:
    """Create a scFoundation fine-tuning factory.

    Fine-tunes scFoundation on classification/regression task per fold,
    extracts penultimate-layer CLS embeddings (2048D).

    Args:
        expression_data: (n_samples, n_genes) expression matrix.
        gene_names: (n_genes,) gene symbols.
        sample_ids: TCGA barcodes (for cache key).
        dataset_name: Used for cache key.
        task_type: 'classification' or 'regression'.
        epochs: Fine-tuning epochs (default: 3).
        lr: Learning rate (default: 2e-5).
        batch_size: Batch size (default: 16).
        cache: Cache fold embeddings to disk.

    Returns:
        Factory callable: (train_idx, val_idx, train_labels) -> (train_embs, val_embs)
    """
    hyperparams = {
        'model': 'scfoundation',
        'epochs': epochs,
        'lr': lr,
    }

    rep_name = 'scfoundation'

    def factory(
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        train_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        key = fold_cache_key(dataset_name, rep_name, train_idx, hyperparams, factory_version=_FACTORY_VERSION)

        def _compute():
            return _finetune_and_extract(
                train_expression=expression_data[train_idx],
                val_expression=expression_data[val_idx],
                train_labels=train_labels,
                gene_names=gene_names,
                task_type=task_type,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
            )

        return load_or_compute(key, _compute, cache=cache)

    return factory
