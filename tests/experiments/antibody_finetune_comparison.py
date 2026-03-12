#!/usr/bin/env python3
"""
Antibody Fine-Tuning Comparison Experiment

Compares frozen vs fine-tuned antibody LM embeddings:
  1. Frozen ESM-2 → Ridge (baseline)
  2. LoRA ESM-2 → Ridge (existing factory)
  3. Frozen AbLang2 → Ridge (baseline)
  4. Fine-tuned AbLang2 → Ridge (new)
  5. Frozen AntiBERTy → Ridge (baseline)
  6. Fine-tuned AntiBERTy → Ridge (new)

Tests on large DMS datasets (≥1000 samples) where fine-tuning has
enough signal, plus inter-family datasets where there's room to improve.

Usage:
    python tests/experiments/antibody_finetune_comparison.py
    python tests/experiments/antibody_finetune_comparison.py --datasets flab_binding_phillips_h3
    python tests/experiments/antibody_finetune_comparison.py --methods esm2_lora,ablang2_ft
    python tests/experiments/antibody_finetune_comparison.py --epochs 10
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

# Datasets to test: large DMS + key inter-family
DEFAULT_DATASETS = [
    'flab_binding_phillips_h3',   # 5000, DMS, ESM-2 R²=0.96
    'flab_expression_koenig',     # 4275, DMS, ESM-2 R²=0.48
    'flab_binding_koenig',        # 4275, DMS, ESM-2 R²=0.30
    'flab_binding_adams',         # 5000, DMS, ESM-2 R²=0.41
    'flab_thermo_tresanco',       # 672, inter-family, ProtT5 R²=0.54
    'abcov_kd',                   # ~1780, inter-family, AbLang2 R²=0.15
]


def load_dataset(name):
    """Load antibody dataset — delegates to antibody_baseline_survey."""
    from antibody_baseline_survey import load_dataset as _load
    return _load(name)


def evaluate_frozen(X, labels, groups, n_folds=5):
    """Evaluate a frozen rep with RidgeCV."""
    n_unique = len(np.unique(groups)) if groups is not None else 0

    if groups is None or n_unique < 2:
        print("  WARNING: <2 unique groups, skipping dataset")
        return {'r2': (float('nan'), float('nan')),
                'spearman': (float('nan'), float('nan'))}

    effective_folds = min(n_folds, n_unique)
    cv = GroupKFold(n_splits=effective_folds)
    splits = list(cv.split(X, labels, groups=groups))

    scores = []
    for train_idx, val_idx in splits:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])

        model = RidgeCV(alphas=np.logspace(-6, 6, 13))
        model.fit(X_tr, labels[train_idx])
        preds = model.predict(X_va)
        r2 = r2_score(labels[val_idx], preds)
        rho = float(spearmanr(labels[val_idx], preds).correlation)
        scores.append({'r2': r2, 'spearman': rho})

    r2_vals = [s['r2'] for s in scores]
    rho_vals = [s['spearman'] for s in scores]
    return {'r2': (float(np.mean(r2_vals)), float(np.std(r2_vals))),
            'spearman': (float(np.mean(rho_vals)), float(np.std(rho_vals)))}


def evaluate_factory(factory, labels, groups, n_folds=5):
    """Evaluate a fine-tuning factory with RidgeCV."""
    n_unique = len(np.unique(groups)) if groups is not None else 0
    dummy = np.zeros((len(labels), 1))

    if groups is None or n_unique < 2:
        print("  WARNING: <2 unique groups, skipping dataset")
        return {'r2': (float('nan'), float('nan')),
                'spearman': (float('nan'), float('nan'))}

    effective_folds = min(n_folds, n_unique)
    cv = GroupKFold(n_splits=effective_folds)
    splits = list(cv.split(dummy, labels, groups=groups))

    scores = []
    for train_idx, val_idx in splits:
        try:
            X_tr, X_va = factory(train_idx, val_idx, labels[train_idx])
        except Exception as e:
            print(f" [FT error: {e}]", end='')
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)

        model = RidgeCV(alphas=np.logspace(-6, 6, 13))
        model.fit(X_tr, labels[train_idx])
        preds = model.predict(X_va)
        r2 = r2_score(labels[val_idx], preds)
        rho = float(spearmanr(labels[val_idx], preds).correlation)
        scores.append({'r2': r2, 'spearman': rho})

    if not scores:
        return {'r2': (float('nan'), float('nan')),
                'spearman': (float('nan'), float('nan'))}
    r2_vals = [s['r2'] for s in scores]
    rho_vals = [s['spearman'] for s in scores]
    return {'r2': (float(np.mean(r2_vals)), float(np.std(r2_vals))),
            'spearman': (float(np.mean(rho_vals)), float(np.std(rho_vals)))}


# =========================================================================
# FINE-TUNING METHODS
# =========================================================================

def make_esm2_lora(heavy_seqs, light_seqs, dataset_name, epochs=5, cache=True):
    """ESM-2 LoRA factory (existing)."""
    from kirby.finetuning.antibody_lora import make_antibody_lora_factory
    return make_antibody_lora_factory(
        heavy_seqs=heavy_seqs, light_seqs=light_seqs,
        dataset_name=dataset_name, base_model='esm2',
        epochs=epochs, cache=cache,
    )


def _finetune_lm_and_extract(
    train_heavy, train_light, val_heavy, val_light, train_labels,
    model_id, epochs, lr, batch_size,
):
    """Dispatch to model-specific fine-tuning."""
    from kirby.finetuning._base import get_device

    device = get_device()

    if 'ablang2' in model_id.lower() or 'ablang' in model_id.lower():
        return _finetune_ablang2(
            train_heavy, train_light, val_heavy, val_light, train_labels,
            epochs, lr, batch_size, device,
        )
    elif 'antiberty' in model_id.lower():
        return _finetune_antiberty(
            train_heavy, train_light, val_heavy, val_light, train_labels,
            epochs, lr, batch_size, device,
        )
    else:
        raise ValueError(f"Unsupported model: {model_id}")


def _finetune_ablang2(
    train_heavy, train_light, val_heavy, val_light, train_labels,
    epochs, lr, batch_size, device,
):
    """Fine-tune AbLang2 on antibody sequences.

    Bypasses model.__call__/seqcoding (which internally uses torch.no_grad()
    and returns numpy) — instead calls tokenizer + AbRep directly for
    gradient-compatible embedding extraction.
    """
    import torch
    import torch.nn as nn

    try:
        import ablang2
        from ablang2.pretrained import format_seq_input
    except ImportError:
        raise ImportError("AbLang2 fine-tuning requires: pip install ablang2")

    # Load AbLang2 model
    model = ablang2.pretrained(model_to_use='ablang2-paired',
                               random_init=False, ncpu=1, device=device)

    # Make encoder trainable
    for param in model.AbRep.parameters():
        param.requires_grad = True

    embed_dim = 480  # AbLang2 paired dim
    head = nn.Sequential(
        nn.Linear(embed_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 1),
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.AbRep.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=0.01,
    )
    criterion = nn.MSELoss()

    # Early stopping holdout
    n_train = len(train_heavy)
    n_holdout = max(1, int(0.1 * n_train))
    perm = np.random.RandomState(42).permutation(n_train)
    ho_idx = perm[:n_holdout]
    fit_idx = perm[n_holdout:]

    best_ho_loss = float('inf')
    patience_counter = 0

    def _embed_batch(heavy_list, light_list):
        """Get AbLang2 embeddings — calls tokenizer + AbRep directly."""
        # AbLang2 expects list of (heavy, light) tuples
        seqs = [(h, l) for h, l in zip(heavy_list, light_list)]
        formatted, _chain = format_seq_input(seqs)
        tokens = model.tokenizer(
            formatted, pad=True, w_extra_tkns=False, device=device)
        hidden = model.AbRep(tokens).last_hidden_states  # (B, L, 480)
        return hidden.mean(dim=1)  # (B, 480)

    print(f"    Fine-tuning AbLang2 for {epochs} epochs...")
    for epoch in range(epochs):
        model.AbRep.train()
        head.train()

        # Mini-batch training
        fit_perm = np.random.RandomState(epoch).permutation(len(fit_idx))
        for start in range(0, len(fit_idx), batch_size):
            batch_ii = fit_perm[start:start + batch_size]
            batch_idx = fit_idx[batch_ii]
            h_batch = [train_heavy[i] for i in batch_idx]
            l_batch = [train_light[i] for i in batch_idx]
            y_batch = torch.FloatTensor(
                train_labels[batch_idx]).to(device)

            optimizer.zero_grad()
            embs = _embed_batch(h_batch, l_batch)
            pred = head(embs).squeeze(-1)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.AbRep.parameters()) + list(head.parameters()),
                1.0)
            optimizer.step()

        # Holdout check
        model.AbRep.eval()
        head.eval()
        with torch.no_grad():
            ho_h = [train_heavy[i] for i in ho_idx]
            ho_l = [train_light[i] for i in ho_idx]
            ho_embs = _embed_batch(ho_h, ho_l)
            ho_pred = head(ho_embs).squeeze(-1)
            ho_y = torch.FloatTensor(train_labels[ho_idx]).to(device)
            ho_loss = criterion(ho_pred, ho_y).item()

        if ho_loss < best_ho_loss:
            best_ho_loss = ho_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 3:
                print(f"      Early stop at epoch {epoch + 1}")
                break

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"      Epoch {epoch + 1}: ho_loss={ho_loss:.4f}")

    # Extract embeddings (discard head)
    model.AbRep.eval()

    def _extract(heavy_list, light_list):
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(heavy_list), batch_size):
                j = min(i + batch_size, len(heavy_list))
                embs = _embed_batch(heavy_list[i:j], light_list[i:j])
                all_embs.append(embs.cpu().numpy())
        return np.vstack(all_embs).astype(np.float32)

    train_embs = _extract(train_heavy, train_light)
    val_embs = _extract(val_heavy, val_light)

    del model, head, optimizer
    if device == 'cuda':
        torch.cuda.empty_cache()

    return train_embs, val_embs


def _finetune_antiberty(
    train_heavy, train_light, val_heavy, val_light, train_labels,
    epochs, lr, batch_size, device,
):
    """Fine-tune AntiBERTy on antibody sequences.

    Embeds H and L chains separately through model.bert (the underlying
    BertModel, which returns standard last_hidden_state), then concatenates
    to 1024D to match the frozen embed_antiberty baseline.
    """
    import torch
    import torch.nn as nn

    try:
        from antiberty import AntiBERTyRunner
    except ImportError:
        raise ImportError("AntiBERTy fine-tuning requires: pip install antiberty")

    runner = AntiBERTyRunner()
    # Use .bert (BertModel) directly — AntiBERTy's forward() returns
    # AntiBERTyOutput which lacks last_hidden_state
    bert = runner.model.bert
    tokenizer = runner.tokenizer

    for param in bert.parameters():
        param.requires_grad = True

    embed_dim = 1024  # 512 (H) + 512 (L) concatenated
    head = nn.Sequential(
        nn.Linear(embed_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 1),
    ).to(device)

    bert = bert.to(device)

    optimizer = torch.optim.AdamW(
        list(bert.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=0.01,
    )
    criterion = nn.MSELoss()

    n_train = len(train_heavy)
    n_holdout = max(1, int(0.1 * n_train))
    perm = np.random.RandomState(42).permutation(n_train)
    ho_idx = perm[:n_holdout]
    fit_idx = perm[n_holdout:]

    best_ho_loss = float('inf')
    patience_counter = 0

    def _embed_single_chain(seq_list):
        """Embed a list of single-chain sequences → (B, 512)."""
        enc = tokenizer(
            [' '.join(list(s)) for s in seq_list],
            padding=True, truncation=True, max_length=512,
            return_tensors='pt',
        )
        ids = enc['input_ids'].to(device)
        mask = enc['attention_mask'].to(device)
        out = bert(input_ids=ids, attention_mask=mask)
        return out.last_hidden_state.mean(dim=1)  # (B, 512)

    def _embed_batch(heavy_list, light_list):
        """Embed H+L → (B, 1024) by concatenating per-chain embeddings."""
        h_emb = _embed_single_chain(heavy_list)
        l_emb = _embed_single_chain(light_list)
        return torch.cat([h_emb, l_emb], dim=-1)  # (B, 1024)

    print(f"    Fine-tuning AntiBERTy for {epochs} epochs...")
    for epoch in range(epochs):
        bert.train()
        head.train()

        fit_perm = np.random.RandomState(epoch).permutation(len(fit_idx))
        for start in range(0, len(fit_idx), batch_size):
            batch_ii = fit_perm[start:start + batch_size]
            batch_idx = fit_idx[batch_ii]
            h_batch = [train_heavy[i] for i in batch_idx]
            l_batch = [train_light[i] for i in batch_idx]
            y_batch = torch.FloatTensor(
                train_labels[batch_idx]).to(device)

            optimizer.zero_grad()
            embs = _embed_batch(h_batch, l_batch)
            pred = head(embs).squeeze(-1)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(bert.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()

        # Holdout
        bert.eval()
        head.eval()
        with torch.no_grad():
            ho_h = [train_heavy[i] for i in ho_idx]
            ho_l = [train_light[i] for i in ho_idx]
            ho_embs = _embed_batch(ho_h, ho_l)
            ho_pred = head(ho_embs).squeeze(-1)
            ho_y = torch.FloatTensor(train_labels[ho_idx]).to(device)
            ho_loss = criterion(ho_pred, ho_y).item()

        if ho_loss < best_ho_loss:
            best_ho_loss = ho_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 3:
                print(f"      Early stop at epoch {epoch + 1}")
                break

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"      Epoch {epoch + 1}: ho_loss={ho_loss:.4f}")

    # Extract
    bert.eval()

    def _extract(heavy_list, light_list):
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(heavy_list), batch_size):
                j = min(i + batch_size, len(heavy_list))
                embs = _embed_batch(heavy_list[i:j], light_list[i:j])
                all_embs.append(embs.cpu().numpy())
        return np.vstack(all_embs).astype(np.float32)

    train_embs = _extract(train_heavy, train_light)
    val_embs = _extract(val_heavy, val_light)

    del bert, head, optimizer
    if device == 'cuda':
        torch.cuda.empty_cache()

    return train_embs, val_embs


def make_ablang2_ft(heavy_seqs, light_seqs, dataset_name, epochs=5,
                    lr=2e-5, batch_size=16, cache=True):
    """AbLang2 fine-tuning factory."""
    from kirby.finetuning._base import fold_cache_key, load_or_compute

    hyperparams = {'model': 'ablang2', 'epochs': epochs, 'lr': lr}
    rep_name = 'ft_ablang2'

    def factory(train_idx, val_idx, train_labels):
        key = fold_cache_key(dataset_name, rep_name, train_idx, hyperparams)

        def _compute():
            from kirby.finetuning._base import get_device
            device = get_device()
            return _finetune_ablang2(
                [heavy_seqs[i] for i in train_idx],
                [light_seqs[i] for i in train_idx],
                [heavy_seqs[i] for i in val_idx],
                [light_seqs[i] for i in val_idx],
                train_labels, epochs, lr, batch_size, device,
            )

        return load_or_compute(key, _compute, cache=cache)

    return factory


def make_antiberty_ft(heavy_seqs, light_seqs, dataset_name, epochs=5,
                      lr=2e-5, batch_size=8, cache=True):
    """AntiBERTy fine-tuning factory."""
    from kirby.finetuning._base import fold_cache_key, load_or_compute

    hyperparams = {'model': 'antiberty', 'epochs': epochs, 'lr': lr}
    rep_name = 'ft_antiberty'

    def factory(train_idx, val_idx, train_labels):
        key = fold_cache_key(dataset_name, rep_name, train_idx, hyperparams)

        def _compute():
            from kirby.finetuning._base import get_device
            device = get_device()
            return _finetune_antiberty(
                [heavy_seqs[i] for i in train_idx],
                [light_seqs[i] for i in train_idx],
                [heavy_seqs[i] for i in val_idx],
                [light_seqs[i] for i in val_idx],
                train_labels, epochs, lr, batch_size, device,
            )

        return load_or_compute(key, _compute, cache=cache)

    return factory


def make_lora_factory(heavy_seqs, light_seqs, dataset_name, base_model,
                      epochs=5, cache=True):
    """Generic LoRA factory for any model in antibody_lora._MODEL_CONFIGS."""
    from kirby.finetuning.antibody_lora import make_antibody_lora_factory
    return make_antibody_lora_factory(
        heavy_seqs=heavy_seqs, light_seqs=light_seqs,
        dataset_name=dataset_name, base_model=base_model,
        epochs=epochs, cache=cache,
    )


def _finetune_igbert(
    train_heavy, train_light, val_heavy, val_light, train_labels,
    epochs, lr, batch_size, device,
):
    """Fine-tune IgBERT (paired model) on antibody sequences."""
    import torch
    import torch.nn as nn

    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError("IgBERT requires transformers")

    model_id = 'Exscientia/IgBERT'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)

    for param in model.parameters():
        param.requires_grad = True

    embed_dim = 1024
    head = nn.Sequential(
        nn.Linear(embed_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 1),
    ).to(device)

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=0.01,
    )
    criterion = nn.MSELoss()

    n_train = len(train_heavy)
    n_holdout = max(1, int(0.1 * n_train))
    perm = np.random.RandomState(42).permutation(n_train)
    ho_idx = perm[:n_holdout]
    fit_idx = perm[n_holdout:]

    best_ho_loss = float('inf')
    patience_counter = 0

    def _embed_batch(heavy_list, light_list):
        # IgBERT: concatenate H and L with [SEP]
        paired = [f"{h} [SEP] {l}" for h, l in zip(heavy_list, light_list)]
        enc = tokenizer(paired, padding=True, truncation=True,
                        max_length=512, return_tensors='pt')
        ids = enc['input_ids'].to(device)
        mask = enc['attention_mask'].to(device)
        out = model(input_ids=ids, attention_mask=mask)
        return out.last_hidden_state.mean(dim=1)

    for epoch in range(epochs):
        model.train()
        head.train()

        fit_perm = np.random.RandomState(epoch).permutation(len(fit_idx))
        for start in range(0, len(fit_idx), batch_size):
            batch_ii = fit_perm[start:start + batch_size]
            batch_idx = fit_idx[batch_ii]
            h_batch = [train_heavy[i] for i in batch_idx]
            l_batch = [train_light[i] for i in batch_idx]
            y_batch = torch.FloatTensor(train_labels[batch_idx]).to(device)

            optimizer.zero_grad()
            embs = _embed_batch(h_batch, l_batch)
            pred = head(embs).squeeze(-1)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()

        model.eval()
        head.eval()
        with torch.no_grad():
            ho_h = [train_heavy[i] for i in ho_idx]
            ho_l = [train_light[i] for i in ho_idx]
            ho_embs = _embed_batch(ho_h, ho_l)
            ho_pred = head(ho_embs).squeeze(-1)
            ho_y = torch.FloatTensor(train_labels[ho_idx]).to(device)
            ho_loss = criterion(ho_pred, ho_y).item()

        if ho_loss < best_ho_loss:
            best_ho_loss = ho_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 3:
                break

    model.eval()

    def _extract(heavy_list, light_list):
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(heavy_list), batch_size):
                j = min(i + batch_size, len(heavy_list))
                embs = _embed_batch(heavy_list[i:j], light_list[i:j])
                all_embs.append(embs.cpu().numpy())
        return np.vstack(all_embs).astype(np.float32)

    train_embs = _extract(train_heavy, train_light)
    val_embs = _extract(val_heavy, val_light)

    del model, head, optimizer
    if device == 'cuda':
        torch.cuda.empty_cache()

    return train_embs, val_embs


def _finetune_igt5(
    train_heavy, train_light, val_heavy, val_light, train_labels,
    epochs, lr, batch_size, device,
):
    """Fine-tune IgT5 (paired model) on antibody sequences."""
    import torch
    import torch.nn as nn

    try:
        from transformers import T5EncoderModel, T5Tokenizer
    except ImportError:
        raise ImportError("IgT5 requires transformers")

    model_id = 'Exscientia/IgT5'
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5EncoderModel.from_pretrained(model_id)

    for param in model.parameters():
        param.requires_grad = True

    embed_dim = 1024
    head = nn.Sequential(
        nn.Linear(embed_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 1),
    ).to(device)

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=0.01,
    )
    criterion = nn.MSELoss()

    n_train = len(train_heavy)
    n_holdout = max(1, int(0.1 * n_train))
    perm = np.random.RandomState(42).permutation(n_train)
    ho_idx = perm[:n_holdout]
    fit_idx = perm[n_holdout:]

    best_ho_loss = float('inf')
    patience_counter = 0

    def _embed_batch(heavy_list, light_list):
        # IgT5: space-separated residues, H and L concatenated
        seqs = [' '.join(list(h)) + ' </s> ' + ' '.join(list(l))
                for h, l in zip(heavy_list, light_list)]
        enc = tokenizer(seqs, padding=True, truncation=True,
                        max_length=512, return_tensors='pt')
        ids = enc['input_ids'].to(device)
        mask = enc['attention_mask'].to(device)
        out = model(input_ids=ids, attention_mask=mask)
        return out.last_hidden_state.mean(dim=1)

    for epoch in range(epochs):
        model.train()
        head.train()

        fit_perm = np.random.RandomState(epoch).permutation(len(fit_idx))
        for start in range(0, len(fit_idx), batch_size):
            batch_ii = fit_perm[start:start + batch_size]
            batch_idx = fit_idx[batch_ii]
            h_batch = [train_heavy[i] for i in batch_idx]
            l_batch = [train_light[i] for i in batch_idx]
            y_batch = torch.FloatTensor(train_labels[batch_idx]).to(device)

            optimizer.zero_grad()
            embs = _embed_batch(h_batch, l_batch)
            pred = head(embs).squeeze(-1)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()

        model.eval()
        head.eval()
        with torch.no_grad():
            ho_h = [train_heavy[i] for i in ho_idx]
            ho_l = [train_light[i] for i in ho_idx]
            ho_embs = _embed_batch(ho_h, ho_l)
            ho_pred = head(ho_embs).squeeze(-1)
            ho_y = torch.FloatTensor(train_labels[ho_idx]).to(device)
            ho_loss = criterion(ho_pred, ho_y).item()

        if ho_loss < best_ho_loss:
            best_ho_loss = ho_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 3:
                break

    model.eval()

    def _extract(heavy_list, light_list):
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(heavy_list), batch_size):
                j = min(i + batch_size, len(heavy_list))
                embs = _embed_batch(heavy_list[i:j], light_list[i:j])
                all_embs.append(embs.cpu().numpy())
        return np.vstack(all_embs).astype(np.float32)

    train_embs = _extract(train_heavy, train_light)
    val_embs = _extract(val_heavy, val_light)

    del model, head, optimizer
    if device == 'cuda':
        torch.cuda.empty_cache()

    return train_embs, val_embs


def make_igbert_ft(heavy_seqs, light_seqs, dataset_name, epochs=5,
                   lr=2e-5, batch_size=8, cache=True):
    """IgBERT full fine-tuning factory."""
    from kirby.finetuning._base import fold_cache_key, load_or_compute

    hyperparams = {'model': 'igbert', 'epochs': epochs, 'lr': lr}
    rep_name = 'ft_igbert'

    def factory(train_idx, val_idx, train_labels):
        key = fold_cache_key(dataset_name, rep_name, train_idx, hyperparams)

        def _compute():
            from kirby.finetuning._base import get_device
            device = get_device()
            return _finetune_igbert(
                [heavy_seqs[i] for i in train_idx],
                [light_seqs[i] for i in train_idx],
                [heavy_seqs[i] for i in val_idx],
                [light_seqs[i] for i in val_idx],
                train_labels, epochs, lr, batch_size, device,
            )

        return load_or_compute(key, _compute, cache=cache)

    return factory


def make_igt5_ft(heavy_seqs, light_seqs, dataset_name, epochs=5,
                 lr=2e-5, batch_size=8, cache=True):
    """IgT5 full fine-tuning factory."""
    from kirby.finetuning._base import fold_cache_key, load_or_compute

    hyperparams = {'model': 'igt5', 'epochs': epochs, 'lr': lr}
    rep_name = 'ft_igt5'

    def factory(train_idx, val_idx, train_labels):
        key = fold_cache_key(dataset_name, rep_name, train_idx, hyperparams)

        def _compute():
            from kirby.finetuning._base import get_device
            device = get_device()
            return _finetune_igt5(
                [heavy_seqs[i] for i in train_idx],
                [light_seqs[i] for i in train_idx],
                [heavy_seqs[i] for i in val_idx],
                [light_seqs[i] for i in val_idx],
                train_labels, epochs, lr, batch_size, device,
            )

        return load_or_compute(key, _compute, cache=cache)

    return factory


# =========================================================================
# FROZEN BASELINES
# =========================================================================

def get_frozen_baselines(heavy_seqs, light_seqs):
    """Generate frozen embeddings for all antibody LMs."""
    from kirby.representations.antibody import (
        embed_esm2, embed_ablang2, embed_antiberty,
        embed_esmc, embed_esm3, embed_saprot,
        embed_prot_t5, embed_amplify, embed_abroberta,
        embed_mint, embed_igbert, embed_igt5,
    )

    baselines = {}

    all_models = [
        ('esm2', embed_esm2),
        ('ablang2', embed_ablang2),
        ('antiberty', embed_antiberty),
        ('esmc', embed_esmc),
        ('esm3', embed_esm3),
        ('saprot', embed_saprot),
        ('prott5', embed_prot_t5),
        ('amplify', embed_amplify),
        ('abroberta', embed_abroberta),
        ('mint', embed_mint),
        ('igbert', embed_igbert),
        ('igt5', embed_igt5),
    ]

    for name, fn in all_models:
        try:
            print(f"    Generating frozen {name}...", end='', flush=True)
            X = fn(heavy_seqs, light_seqs)
            baselines[name] = X
            print(f" ({X.shape[1]}D)")
        except Exception as e:
            print(f" FAILED: {e}")

    return baselines


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Antibody fine-tuning comparison experiment')
    parser.add_argument('--datasets', type=str, default=None,
                       help='Comma-separated dataset names')
    parser.add_argument('--methods', type=str, default=None,
                       help='Comma-separated methods: esm2_lora,esmc_lora,'
                            'esm3_lora,saprot_lora,prott5_lora,amplify_lora,'
                            'abroberta_lora,mint_lora,ablang2_ft,antiberty_ft,'
                            'igbert_ft,igt5_ft (default: all)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Fine-tuning epochs (default: 5)')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of CV folds (default: 5)')
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    datasets = (args.datasets.split(',') if args.datasets
                else DEFAULT_DATASETS)
    methods = (args.methods.split(',') if args.methods
               else ['esm2_lora', 'esmc_lora', 'esm3_lora', 'saprot_lora',
                      'prott5_lora', 'amplify_lora', 'abroberta_lora',
                      'mint_lora', 'ablang2_ft', 'antiberty_ft',
                      'igbert_ft', 'igt5_ft'])
    cache = not args.no_cache

    print("=" * 70)
    print("ANTIBODY FINE-TUNING COMPARISON")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Methods: {methods}")
    print(f"  Epochs: {args.epochs}, Folds: {args.n_folds}")
    print("=" * 70)

    all_results = {}

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        try:
            data = load_dataset(ds_name)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        heavy_seqs = data['heavy_seqs']
        light_seqs = data['light_seqs']
        labels = data['labels']
        groups = data['groups']
        n = len(labels)

        print(f"  {n} samples, {len(np.unique(groups))} groups")

        ds_results = {'n_samples': n, 'frozen': {}, 'finetuned': {}}

        # Frozen baselines
        print("\n  --- Frozen baselines ---")
        frozen = get_frozen_baselines(heavy_seqs, light_seqs)
        for rep_name, X in frozen.items():
            result = evaluate_frozen(
                X, labels, groups, args.n_folds)
            r2_mean, r2_std = result['r2']
            rho_mean, rho_std = result['spearman']
            ds_results['frozen'][rep_name] = {
                'r2_mean': round(r2_mean, 4),
                'r2_std': round(r2_std, 4),
                'spearman_mean': round(rho_mean, 4),
                'spearman_std': round(rho_std, 4),
                'dim': X.shape[1],
            }
            print(f"    {rep_name}: R²={r2_mean:.4f} ± {r2_std:.4f}"
                  f"  ρ={rho_mean:.4f} ± {rho_std:.4f}")

        # Fine-tuned methods
        print("\n  --- Fine-tuned ---")
        factories = {}

        # LoRA-based methods (all use antibody_lora.py with different base_model)
        lora_methods = {
            'esm2_lora': 'esm2',
            'esmc_lora': 'esmc',
            'esm3_lora': 'esm3',
            'saprot_lora': 'saprot',
            'prott5_lora': 'prott5',
            'amplify_lora': 'amplify',
            'abroberta_lora': 'abroberta',
            'mint_lora': 'mint',
        }

        for method_name, base_model in lora_methods.items():
            if method_name in methods:
                try:
                    factories[method_name] = make_lora_factory(
                        heavy_seqs, light_seqs, ds_name,
                        base_model=base_model,
                        epochs=args.epochs, cache=cache)
                except Exception as e:
                    print(f"    {method_name}: FAILED to create factory: {e}")

        # Full fine-tuning methods
        if 'ablang2_ft' in methods:
            try:
                factories['ablang2_ft'] = make_ablang2_ft(
                    heavy_seqs, light_seqs, ds_name,
                    epochs=args.epochs, cache=cache)
            except Exception as e:
                print(f"    ablang2_ft: FAILED to create factory: {e}")

        if 'antiberty_ft' in methods:
            try:
                factories['antiberty_ft'] = make_antiberty_ft(
                    heavy_seqs, light_seqs, ds_name,
                    epochs=args.epochs, cache=cache)
            except Exception as e:
                print(f"    antiberty_ft: FAILED to create factory: {e}")

        if 'igbert_ft' in methods:
            try:
                factories['igbert_ft'] = make_igbert_ft(
                    heavy_seqs, light_seqs, ds_name,
                    epochs=args.epochs, cache=cache)
            except Exception as e:
                print(f"    igbert_ft: FAILED to create factory: {e}")

        if 'igt5_ft' in methods:
            try:
                factories['igt5_ft'] = make_igt5_ft(
                    heavy_seqs, light_seqs, ds_name,
                    epochs=args.epochs, cache=cache)
            except Exception as e:
                print(f"    igt5_ft: FAILED to create factory: {e}")

        for ft_name, factory in factories.items():
            print(f"    {ft_name}:", end='', flush=True)
            result = evaluate_factory(
                factory, labels, groups, args.n_folds)
            r2_mean, r2_std = result['r2']
            rho_mean, rho_std = result['spearman']
            ds_results['finetuned'][ft_name] = {
                'r2_mean': round(r2_mean, 4),
                'r2_std': round(r2_std, 4),
                'spearman_mean': round(rho_mean, 4),
                'spearman_std': round(rho_std, 4),
            }
            print(f" R²={r2_mean:.4f} ± {r2_std:.4f}"
                  f"  ρ={rho_mean:.4f} ± {rho_std:.4f}")

            # Compare to frozen baseline
            # Map method name to frozen baseline name
            ft_to_frozen = {
                'esm2_lora': 'esm2', 'esmc_lora': 'esmc', 'esm3_lora': 'esm3',
                'saprot_lora': 'saprot', 'prott5_lora': 'prott5',
                'amplify_lora': 'amplify', 'abroberta_lora': 'abroberta',
                'mint_lora': 'mint', 'ablang2_ft': 'ablang2',
                'antiberty_ft': 'antiberty', 'igbert_ft': 'igbert',
                'igt5_ft': 'igt5',
            }
            base_name = ft_to_frozen.get(ft_name, ft_name.split('_')[0])
            if base_name in ds_results['frozen']:
                base_r2 = ds_results['frozen'][base_name]['r2_mean']
                delta = r2_mean - base_r2
                sign = '+' if delta >= 0 else ''
                print(f"      vs frozen {base_name}: {sign}{delta:.4f}")

        all_results[ds_name] = ds_results

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for ds, res in all_results.items():
        print(f"\n  {ds} ({res['n_samples']} samples):")
        for name, r in res['frozen'].items():
            print(f"    frozen {name}: R²={r['r2_mean']:.4f}  "
                  f"ρ={r['spearman_mean']:.4f}")
        for name, r in res['finetuned'].items():
            print(f"    {name}: R²={r['r2_mean']:.4f}  "
                  f"ρ={r['spearman_mean']:.4f}")

    # Save
    output_path = args.output or 'tests/results/comparisons'
    p = Path(output_path)
    if p.suffix == '.json':
        out_file = p
    else:
        p.mkdir(parents=True, exist_ok=True)
        out_file = p / 'antibody_finetune_comparison.json'
    out_file.parent.mkdir(parents=True, exist_ok=True)

    from experiments._shared import convert_numpy
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert_numpy)
    print(f"\nResults saved: {out_file}")


if __name__ == '__main__':
    main()
