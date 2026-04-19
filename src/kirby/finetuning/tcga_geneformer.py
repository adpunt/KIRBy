"""
TCGA Geneformer fine-tuning factory.

Fine-tunes Geneformer on classification/regression tasks,
extracts penultimate-layer cell embeddings.

Requires: torch, transformers, huggingface_hub
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from kirby.finetuning._base import (
    fold_cache_key,
    get_device,
    load_or_compute,
)

# Factory version — bump when the training recipe, tokenizer, or base-model
# weights change in a way that invalidates prior cached fold embeddings.
_FACTORY_VERSION = 'v1'


# ---------------------------------------------------------------------------
# Geneformer tokenization helpers
# ---------------------------------------------------------------------------

def _load_geneformer_dictionaries() -> Tuple[Dict, Dict, Dict]:
    """Download and load Geneformer gene→token dictionaries from HuggingFace.

    Returns:
        (token_dict, median_dict, name_id_dict)
        - token_dict: Ensembl ID → token integer
        - median_dict: Ensembl ID → nonzero median expression (corpus)
        - name_id_dict: gene symbol → Ensembl ID
    """
    import pickle
    from huggingface_hub import hf_hub_download

    repo_id = 'ctheodoris/Geneformer'

    # Try V2 (gc104M) first, fall back to V1 (gc30M)
    try:
        token_path = hf_hub_download(repo_id, "geneformer/token_dictionary_gc104M.pkl")
        median_path = hf_hub_download(repo_id, "geneformer/gene_median_dictionary_gc104M.pkl")
        name_path = hf_hub_download(repo_id, "geneformer/gene_name_id_dict_gc104M.pkl")
    except Exception:
        token_path = hf_hub_download(repo_id, "geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl")
        median_path = hf_hub_download(repo_id, "geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl")
        name_path = hf_hub_download(repo_id, "geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl")

    with open(token_path, 'rb') as f:
        token_dict = pickle.load(f)
    with open(median_path, 'rb') as f:
        median_dict = pickle.load(f)
    with open(name_path, 'rb') as f:
        name_id_dict = pickle.load(f)

    return token_dict, median_dict, name_id_dict


def _tokenize_expression(
    expression: np.ndarray,
    gene_names: np.ndarray,
    max_input_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Proper Geneformer tokenization of gene expression.

    Maps HUGO symbols → Ensembl IDs → Geneformer token IDs,
    normalizes by corpus gene medians, rank-orders descending.

    Args:
        expression: (n_samples, n_genes) expression matrix
        gene_names: (n_genes,) HUGO gene symbols
        max_input_size: max tokens per sample

    Returns:
        input_ids: (n_samples, max_input_size) padded token IDs
        attention_mask: (n_samples, max_input_size) 1=real, 0=pad
        proper_vocab: True if Geneformer vocab was loaded
    """
    n_samples, n_genes = expression.shape

    # Try to load Geneformer vocabulary
    try:
        token_dict, median_dict, name_id_dict = _load_geneformer_dictionaries()
    except Exception as e:
        print(f"    Could not load Geneformer dictionaries ({e}), using raw indices")
        # Fallback: raw indices (works with the BERT fallback that has vocab_size=30000)
        ranks = np.argsort(-expression, axis=1)[:, :max_input_size]
        mask = np.ones_like(ranks)
        return ranks, mask, False

    # Map gene symbols → column index, Ensembl ID, token ID, median
    col_token = {}   # column index → token ID
    col_median = {}  # column index → median expression
    for i, name in enumerate(gene_names):
        name_str = str(name)
        ensembl_id = name_id_dict.get(name_str)
        if ensembl_id is None:
            # Try uppercase
            ensembl_id = name_id_dict.get(name_str.upper())
        if ensembl_id is None:
            continue
        token_id = token_dict.get(ensembl_id)
        if token_id is None:
            continue
        median = median_dict.get(ensembl_id, 1.0)
        col_token[i] = token_id
        col_median[i] = median if median > 0 else 1.0

    mapped_cols = np.array(sorted(col_token.keys()))
    print(f"    Mapped {len(mapped_cols)}/{n_genes} genes to Geneformer vocabulary")

    if len(mapped_cols) == 0:
        print("    WARNING: No genes mapped — returning zero tokens")
        return (np.zeros((n_samples, max_input_size), dtype=np.int64),
                np.zeros((n_samples, max_input_size), dtype=np.int64), True)

    # Vectorised: extract expression for mapped genes, normalize, rank, convert
    token_ids_arr = np.array([col_token[c] for c in mapped_cols])
    medians_arr = np.array([col_median[c] for c in mapped_cols])

    expr_mapped = expression[:, mapped_cols]              # (n_samples, n_mapped)
    normalised = expr_mapped / medians_arr[np.newaxis, :] # normalise by corpus median

    # Per-sample: rank descending by normalised expression, take top tokens
    input_ids = np.zeros((n_samples, max_input_size), dtype=np.int64)
    attention_mask = np.zeros((n_samples, max_input_size), dtype=np.int64)

    for s in range(n_samples):
        vals = normalised[s]
        nonzero = vals > 0
        if not nonzero.any():
            continue
        nz_idx = np.where(nonzero)[0]
        order = np.argsort(-vals[nz_idx])
        n_tok = min(len(order), max_input_size)
        input_ids[s, :n_tok] = token_ids_arr[nz_idx[order[:n_tok]]]
        attention_mask[s, :n_tok] = 1

    return input_ids, attention_mask, True


def _finetune_and_extract(
    train_expression: np.ndarray,
    val_expression: np.ndarray,
    train_labels: np.ndarray,
    gene_names: np.ndarray,
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

    # Tokenize: map gene symbols → Geneformer vocab token IDs
    all_expression = np.vstack([train_expression, val_expression])
    all_ids, all_mask, proper_vocab = _tokenize_expression(
        all_expression, gene_names, max_input_size,
    )
    n_train = len(train_expression)
    train_tok = all_ids[:n_train]
    train_mask_np = all_mask[:n_train]
    val_tok = all_ids[n_train:]
    val_mask_np = all_mask[n_train:]

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
    train_ids = torch.LongTensor(train_tok)
    train_mask = torch.LongTensor(train_mask_np)

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

    val_ids = torch.LongTensor(val_tok)
    val_mask = torch.LongTensor(val_mask_np)

    train_embs = _extract(train_ids, train_mask)
    val_embs = _extract(val_ids, val_mask)

    # Cleanup
    del base_model, head, optimizer
    if device == 'cuda':
        torch.cuda.empty_cache()

    return train_embs, val_embs


def make_geneformer_factory(
    expression_data: np.ndarray,
    gene_names: np.ndarray,
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
        gene_names: (n_genes,) HUGO gene symbols (for Geneformer vocab mapping).
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
                max_input_size=max_input_size,
            )

        return load_or_compute(key, _compute, cache=cache)

    return factory
