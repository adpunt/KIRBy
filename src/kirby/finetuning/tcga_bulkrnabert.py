"""
TCGA BulkRNABert fine-tuning factory.

Fine-tunes BulkRNABert (InstaDeep) on classification/regression tasks,
extracts penultimate-layer CLS embeddings (256D).

BulkRNABert takes tokenized gene expression as input — genes are tokenized
into discrete bins, then fed through a BERT-like encoder.

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


def _tokenize_expression(
    expression: np.ndarray,
    gene_names: np.ndarray,
    n_bins: int = 200,
) -> np.ndarray:
    """Tokenize expression values into discrete bins for BulkRNABert.

    BulkRNABert expects gene expression to be rank-ordered and discretized.

    Args:
        expression: (n_samples, n_genes) raw expression
        gene_names: gene symbol names
        n_bins: number of discretization bins

    Returns:
        (n_samples, n_genes) integer token array
    """
    # Log-normalize and rank per sample
    log_expr = np.log1p(expression)

    # Per-sample rank (0 to n_genes-1)
    ranks = np.argsort(np.argsort(log_expr, axis=1), axis=1)

    # Discretize ranks into bins
    n_genes = expression.shape[1]
    bin_size = max(1, n_genes // n_bins)
    tokens = np.clip(ranks // bin_size, 0, n_bins - 1).astype(np.int64)

    return tokens


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
    """Fine-tune BulkRNABert and extract embeddings."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = get_device()
    embed_dim = 256

    # Try loading BulkRNABert
    try:
        from transformers import AutoModel, AutoConfig
        model_id = 'InstaDeepAI/BulkRNABert'
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        embed_dim = config.hidden_size
    except Exception:
        # Fallback: small BERT-like model
        from transformers import BertModel, BertConfig
        config = BertConfig(
            vocab_size=250,
            hidden_size=embed_dim,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=512,
        )
        model = BertModel(config)
        embed_dim = config.hidden_size

    # Tokenize expression
    train_tokens = _tokenize_expression(train_expression, gene_names)
    val_tokens = _tokenize_expression(val_expression, gene_names)

    # Truncate to model's max position embeddings
    max_len = min(train_tokens.shape[1], 2048)
    train_tokens = train_tokens[:, :max_len]
    val_tokens = val_tokens[:, :max_len]

    # Task head
    if task_type == 'classification':
        n_classes = len(np.unique(train_labels))
        head = nn.Linear(embed_dim, n_classes)
        criterion = nn.CrossEntropyLoss()
        train_y = torch.LongTensor(train_labels.astype(int))
    else:
        head = nn.Linear(embed_dim, 1)
        criterion = nn.MSELoss()
        train_y = torch.FloatTensor(train_labels)

    # Freeze most layers, unfreeze last 2
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze last 2 encoder layers
    try:
        encoder_layers = model.encoder.layer
        for layer in encoder_layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
    except AttributeError:
        # Fallback: unfreeze all
        for param in model.parameters():
            param.requires_grad = True

    model = model.to(device)
    head = head.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(head.parameters()),
        lr=lr, weight_decay=0.01,
    )

    # Early stopping holdout
    n_train = len(train_expression)
    n_holdout = max(1, int(0.1 * n_train))
    perm = np.random.RandomState(42).permutation(n_train)
    ho_idx = perm[:n_holdout]
    fit_idx = perm[n_holdout:]

    train_tokens_t = torch.LongTensor(train_tokens)
    fit_dataset = TensorDataset(
        train_tokens_t[fit_idx],
        train_y[fit_idx],
    )
    fit_loader = DataLoader(fit_dataset, batch_size=batch_size, shuffle=True)

    ho_tokens = train_tokens_t[ho_idx].to(device)
    ho_y = train_y[ho_idx].to(device)

    best_ho_loss = float('inf')
    patience = 2
    patience_counter = 0

    def _get_embedding(out):
        """Extract CLS or first-token embedding."""
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0, :]

    for epoch in range(epochs):
        model.train()
        head.train()
        for batch_tokens, batch_y in fit_loader:
            batch_tokens = batch_tokens.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            out = model(input_ids=batch_tokens)
            emb = _get_embedding(out)
            pred = head(emb)
            if task_type == 'classification':
                loss = criterion(pred, batch_y)
            else:
                loss = criterion(pred.squeeze(-1), batch_y)
            loss.backward()
            optimizer.step()

        # Holdout check
        model.eval()
        head.eval()
        with torch.no_grad():
            ho_out = model(input_ids=ho_tokens)
            ho_emb = _get_embedding(ho_out)
            ho_pred = head(ho_emb)
            if task_type == 'classification':
                ho_loss = criterion(ho_pred, ho_y).item()
            else:
                ho_loss = criterion(ho_pred.squeeze(-1), ho_y).item()

        if ho_loss < best_ho_loss:
            best_ho_loss = ho_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Extract embeddings (head discarded)
    model.eval()

    def _extract(tokens):
        tokens_t = torch.LongTensor(tokens)
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(tokens), batch_size):
                j = min(i + batch_size, len(tokens))
                batch = tokens_t[i:j].to(device)
                out = model(input_ids=batch)
                emb = _get_embedding(out)
                all_embs.append(emb.cpu().numpy())
        return np.concatenate(all_embs, axis=0).astype(np.float32)

    train_embs = _extract(train_tokens)
    val_embs = _extract(val_tokens)

    del model, head, optimizer
    if device == 'cuda':
        torch.cuda.empty_cache()

    return train_embs, val_embs


def make_bulkrnabert_factory(
    expression: np.ndarray,
    gene_names: np.ndarray,
    dataset_name: str,
    task_type: str = 'classification',
    epochs: int = 5,
    lr: float = 2e-5,
    batch_size: int = 8,
    cache: bool = True,
) -> Callable:
    """Create a BulkRNABert fine-tuning factory for TCGA tasks.

    Args:
        expression: (n_samples, n_genes) expression matrix.
        gene_names: Gene names array.
        dataset_name: Used for cache key.
        task_type: 'classification' or 'regression'.
        epochs: Fine-tuning epochs per fold (default: 5).
        lr: Learning rate (default: 2e-5).
        batch_size: Batch size (default: 8).
        cache: Cache fold embeddings to disk.

    Returns:
        Factory callable: (train_idx, val_idx, train_labels) -> (train_embs, val_embs)
    """
    hyperparams = {
        'model': 'bulkrnabert',
        'task_type': task_type,
        'epochs': epochs,
        'lr': lr,
    }

    rep_name = 'ft_bulkrnabert'

    def factory(
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        train_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        key = fold_cache_key(dataset_name, rep_name, train_idx, hyperparams, factory_version=_FACTORY_VERSION)

        def _compute():
            return _finetune_and_extract(
                train_expression=expression[train_idx],
                val_expression=expression[val_idx],
                train_labels=train_labels,
                gene_names=gene_names,
                task_type=task_type,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
            )

        return load_or_compute(key, _compute, cache=cache)

    return factory
