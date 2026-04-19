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
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, roc_auc_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
    _HAS_STRATIFIED_GROUP = True
except ImportError:
    _HAS_STRATIFIED_GROUP = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

# Datasets to test. Default is broad — all registered antibody datasets
# (regression + classification). Override with --datasets on CLI.
#
# Regression (spearman): inter-family (10) + DMS (16) + ginkgo (6) = 32.
# Classification (auc / f1_weighted): tz_her2_binary, tz_her2_multiclass,
# humanness_therapeutic. Classification datasets route through a
# separate evaluation path with StratifiedGroupKFold and AUC/F1 metrics.
DEFAULT_DATASETS_REGRESSION = [
    # Inter-family
    'flab_thermo_tresanco', 'flab_pk_jain_heprt', 'flab_pk_jain_fcrnrt',
    'flab_polyreact_jain_cic', 'flab_aggregation_jain_hicrt',
    'flab_thermo_jain', 'abcov_kd', 'flab_aggregation_jain',
    'flab_immunogenicity_marks', 'flab_polyreact_jain_bvp',
    # DMS + mutation scan
    'ddg_combined', 'flab_binding_koenig', 'flab_expression_koenig',
    'flab_binding_adams', 'flab_expression_adams',
    'flab_binding_phillips_h3',
    'abagym_Ang2', 'abagym_EGFR', 'abagym_HER2', 'abagym_lysozyme',
    'abagym_VEGF',
    # Ginkgo developability
    'ginkgo_acsins', 'ginkgo_colloidal', 'ginkgo_heparin',
    'ginkgo_hic', 'ginkgo_pr_cho', 'ginkgo_titer',
]

DEFAULT_DATASETS_CLASSIFICATION = [
    'tz_her2_binary',       # AUC
    'tz_her2_multiclass',   # F1 weighted (multi-class)
    'humanness_therapeutic',  # AUC
]

DEFAULT_DATASETS = DEFAULT_DATASETS_REGRESSION + DEFAULT_DATASETS_CLASSIFICATION


def load_dataset(name):
    """Load antibody dataset — delegates to antibody_baseline_survey.

    Cross-checks the loader's task_type against DATASET_REGISTRY so any
    silent drift between loader and registry fails loudly here rather
    than producing wrong numbers downstream.
    """
    from antibody_baseline_survey import load_dataset as _load
    from kirby.registry import crosscheck_task_type, lookup
    data = _load(name)
    # antibody_baseline_survey.load_dataset may not set 'task_type' on
    # the returned dict — infer from registry when missing.
    loader_task = data.get('task_type')
    if loader_task is None:
        info = lookup(name)
        loader_task = info.task_type if info else 'regression'
        data['task_type'] = loader_task
    crosscheck_task_type(name, loader_task)
    return data


def _clf_metric_names(n_classes: int) -> List[str]:
    """Keys reported per fold for classification."""
    if n_classes > 2:
        return ['f1_weighted']
    return ['auc', 'f1']


def _score_clf_fold(y_true, y_pred_labels, y_pred_proba, n_classes):
    """Return dict of {metric_name: value} for one classification fold."""
    out = {}
    try:
        if n_classes > 2:
            out['f1_weighted'] = f1_score(y_true, y_pred_labels,
                                           average='weighted')
        else:
            if y_pred_proba is not None:
                out['auc'] = roc_auc_score(y_true, y_pred_proba)
            else:
                out['auc'] = float('nan')
            out['f1'] = f1_score(y_true, y_pred_labels)
    except Exception:
        for k in _clf_metric_names(n_classes):
            out[k] = float('nan')
    return out


def _make_clf_cv(labels, groups, n_folds):
    """Stratified group k-fold when possible, else GroupKFold. Returns splits list."""
    n_unique = len(np.unique(groups)) if groups is not None else 0
    effective_folds = min(n_folds, n_unique)
    if _HAS_STRATIFIED_GROUP and n_unique >= effective_folds:
        cv = StratifiedGroupKFold(n_splits=effective_folds, shuffle=True,
                                   random_state=42)
        return list(cv.split(np.zeros((len(labels), 1)), labels, groups=groups))
    cv = GroupKFold(n_splits=effective_folds)
    return list(cv.split(np.zeros((len(labels), 1)), labels, groups=groups))


def evaluate_frozen(X, labels, groups, n_folds=5, task_type='regression'):
    """Evaluate a frozen rep. Regression → RidgeCV, classification → LogisticRegressionCV.

    Returns dict keyed by metric name, where each value is (mean, std)
    across folds. Regression keys: 'r2', 'spearman'. Classification
    keys: 'auc' + 'f1' (binary) or 'f1_weighted' (multi-class).
    """
    n_unique = len(np.unique(groups)) if groups is not None else 0
    if groups is None or n_unique < 2:
        print("  WARNING: <2 unique groups, skipping dataset")
        if task_type == 'classification':
            n_classes = len(np.unique(labels))
            return {k: (float('nan'), float('nan'))
                    for k in _clf_metric_names(n_classes)}
        return {'r2': (float('nan'), float('nan')),
                'spearman': (float('nan'), float('nan'))}

    if task_type == 'classification':
        n_classes = len(np.unique(labels))
        splits = _make_clf_cv(labels, groups, n_folds)

        per_fold = []
        for train_idx, val_idx in splits:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_va = scaler.transform(X[val_idx])

            model = LogisticRegressionCV(
                Cs=10, max_iter=2000, n_jobs=-1,
                scoring='roc_auc_ovr_weighted' if n_classes > 2 else 'roc_auc',
            )
            model.fit(X_tr, labels[train_idx])
            y_pred = model.predict(X_va)
            if n_classes == 2 and hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_va)[:, 1]
            else:
                y_proba = None
            per_fold.append(_score_clf_fold(labels[val_idx], y_pred, y_proba, n_classes))

        # Aggregate per metric
        out = {}
        for k in _clf_metric_names(n_classes):
            vals = [d[k] for d in per_fold if not np.isnan(d.get(k, np.nan))]
            if vals:
                out[k] = (float(np.mean(vals)), float(np.std(vals)))
            else:
                out[k] = (float('nan'), float('nan'))
        return out

    # Regression path (unchanged)
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


def evaluate_factory(factory, labels, groups, n_folds=5, task_type='regression'):
    """Evaluate a fine-tuning factory with RidgeCV (reg) or LogisticRegressionCV (clf).

    Same metric contract as evaluate_frozen.
    """
    n_unique = len(np.unique(groups)) if groups is not None else 0
    dummy = np.zeros((len(labels), 1))

    if groups is None or n_unique < 2:
        print("  WARNING: <2 unique groups, skipping dataset")
        if task_type == 'classification':
            n_classes = len(np.unique(labels))
            return {k: (float('nan'), float('nan'))
                    for k in _clf_metric_names(n_classes)}
        return {'r2': (float('nan'), float('nan')),
                'spearman': (float('nan'), float('nan'))}

    if task_type == 'classification':
        n_classes = len(np.unique(labels))
        splits = _make_clf_cv(labels, groups, n_folds)

        per_fold = []
        for train_idx, val_idx in splits:
            try:
                X_tr, X_va = factory(train_idx, val_idx, labels[train_idx])
            except Exception as e:
                print(f" [FT error: {e}]", end='')
                continue
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)

            model = LogisticRegressionCV(
                Cs=10, max_iter=2000, n_jobs=-1,
                scoring='roc_auc_ovr_weighted' if n_classes > 2 else 'roc_auc',
            )
            model.fit(X_tr, labels[train_idx])
            y_pred = model.predict(X_va)
            if n_classes == 2 and hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_va)[:, 1]
            else:
                y_proba = None
            per_fold.append(_score_clf_fold(
                labels[val_idx], y_pred, y_proba, n_classes))

        out = {}
        for k in _clf_metric_names(n_classes):
            vals = [d[k] for d in per_fold if not np.isnan(d.get(k, np.nan))]
            if vals:
                out[k] = (float(np.mean(vals)), float(np.std(vals)))
            else:
                out[k] = (float('nan'), float('nan'))
        return out

    # Regression path (unchanged)
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
    """Generate frozen embeddings for the broad antibody baseline set.

    Covers what's in DATASET_REPS for antibody: the 12 full-sequence LMs
    + their _cdr variants + esm2 sub-variants (layer/entropy/concat/cdr3)
    + mutation signal reps (llr / wt_delta / mutsite) + classical
    (onehot/blosum/aac). Mapping follows the names used as keys in
    complexity_helpers.DATASET_REPS so FT vs frozen comparisons can be
    wired later.

    Reps that can't generate (module missing, OOM, seq-pair mismatch)
    are skipped with a WARNING; the run continues.
    """
    from kirby.representations.antibody import (
        embed_esm2, embed_ablang2, embed_antiberty,
        embed_esmc, embed_esm3, embed_saprot,
        embed_prot_t5, embed_abroberta,
        embed_mint, embed_igbert, embed_igt5,
    )

    baselines = {}

    # Try-import CDR and sub-variant embedders — these may not all be
    # available depending on the kirby.representations.antibody version.
    _optional = {}
    _tried = [
        # CDR-only variants
        ('antiberty_cdr', 'embed_antiberty_cdr'),
        ('esm2_cdr', 'embed_esm2_cdr'),
        ('esm2_cdr3', 'embed_esm2_cdr3'),
        ('esmc_cdr', 'embed_esmc_cdr'),
        ('saprot_cdr', 'embed_saprot_cdr'),
        ('esm3_cdr', 'embed_esm3_cdr'),
        ('prot_t5_cdr', 'embed_prot_t5_cdr'),
        ('abroberta_cdr', 'embed_abroberta_cdr'),
        # esm2 sub-variants
        ('esm2_midlayers', 'embed_esm2_midlayers'),
        ('esm2_layer24', 'embed_esm2_layer24'),
        ('esm2_early', 'embed_esm2_early'),
        ('esm2_concat', 'embed_esm2_concat'),
        ('esm2_per_cdr', 'embed_esm2_per_cdr'),
        ('esm2_per_region', 'embed_esm2_per_region'),
        ('esm2_entropy_top20', 'embed_esm2_entropy_top20'),
        # Mutation signal
        ('esm2_wt_delta', 'embed_esm2_wt_delta'),
        ('esm2_mutsite', 'embed_esm2_mutsite'),
        ('esm2_llr', 'embed_esm2_llr'),
        ('antiberty_llr', 'embed_antiberty_llr'),
        ('esmc_llr', 'embed_esmc_llr'),
    ]
    import kirby.representations.antibody as _ab_mod
    for rep_name, attr in _tried:
        fn = getattr(_ab_mod, attr, None)
        if fn is not None:
            _optional[rep_name] = fn

    # Classical reps (simpler — separate module)
    try:
        from kirby.representations.antibody import (
            embed_onehot, embed_blosum, embed_aac,
        )
        _optional['onehot'] = embed_onehot
        _optional['blosum'] = embed_blosum
        _optional['aac'] = embed_aac
    except ImportError:
        pass

    # Paratope + structural + mutation_features (if available)
    for rep_name, attr in [
        ('paratope', 'embed_paratope'),
        ('structural', 'embed_structural'),
        ('mutation_features', 'embed_mutation_features'),
    ]:
        fn = getattr(_ab_mod, attr, None)
        if fn is not None:
            _optional[rep_name] = fn

    all_models = [
        ('esm2', embed_esm2),
        ('ablang2', embed_ablang2),
        ('antiberty', embed_antiberty),
        ('esmc', embed_esmc),
        ('esm3', embed_esm3),
        ('saprot', embed_saprot),
        ('prot_t5', embed_prot_t5),
        ('abroberta', embed_abroberta),
        ('mint', embed_mint),
        ('igbert', embed_igbert),
        ('igt5', embed_igt5),
    ] + list(_optional.items())

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
                            'igbert_ft (default: all; igt5_ft removed — OOM on 44GB GPU)')
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
                      'igbert_ft'])
    # igt5_ft removed: OOM on 44GB GPU (IgT5 paired T5 too large for LoRA FT)
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
        task_type = data.get('task_type', 'regression')
        n = len(labels)

        print(f"  {n} samples, {len(np.unique(groups))} groups, "
              f"task={task_type}")

        ds_results = {'n_samples': n, 'task_type': task_type,
                      'frozen': {}, 'finetuned': {}}

        # Frozen baselines
        print("\n  --- Frozen baselines ---")
        frozen = get_frozen_baselines(heavy_seqs, light_seqs)
        for rep_name, X in frozen.items():
            result = evaluate_frozen(
                X, labels, groups, args.n_folds, task_type=task_type)
            entry = {'dim': X.shape[1]}
            for metric_name, (m_mean, m_std) in result.items():
                entry[f'{metric_name}_mean'] = round(m_mean, 4)
                entry[f'{metric_name}_std'] = round(m_std, 4)
            ds_results['frozen'][rep_name] = entry
            # Terse one-liner: show primary metric for the task
            if task_type == 'classification':
                primary = 'f1_weighted' if len(np.unique(labels)) > 2 else 'auc'
                m, s = result[primary]
                print(f"    {rep_name}: {primary}={m:.4f} ± {s:.4f}")
            else:
                r2_m, r2_s = result['r2']
                rho_m, rho_s = result['spearman']
                print(f"    {rep_name}: R²={r2_m:.4f} ± {r2_s:.4f}"
                      f"  ρ={rho_m:.4f} ± {rho_s:.4f}")

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
                factory, labels, groups, args.n_folds, task_type=task_type)
            entry = {}
            for metric_name, (m_mean, m_std) in result.items():
                entry[f'{metric_name}_mean'] = round(m_mean, 4)
                entry[f'{metric_name}_std'] = round(m_std, 4)
            ds_results['finetuned'][ft_name] = entry
            if task_type == 'classification':
                primary = 'f1_weighted' if len(np.unique(labels)) > 2 else 'auc'
                m, s = result[primary]
                print(f" {primary}={m:.4f} ± {s:.4f}")
            else:
                r2_m, r2_s = result['r2']
                rho_m, rho_s = result['spearman']
                print(f" R²={r2_m:.4f} ± {r2_s:.4f}"
                      f"  ρ={rho_m:.4f} ± {rho_s:.4f}")

            # Compare to frozen baseline on the primary metric
            ft_to_frozen = {
                'esm2_lora': 'esm2', 'esmc_lora': 'esmc', 'esm3_lora': 'esm3',
                'saprot_lora': 'saprot', 'prott5_lora': 'prot_t5',
                'amplify_lora': 'amplify', 'abroberta_lora': 'abroberta',
                'mint_lora': 'mint', 'ablang2_ft': 'ablang2',
                'antiberty_ft': 'antiberty', 'igbert_ft': 'igbert',
                'igt5_ft': 'igt5',
            }
            base_name = ft_to_frozen.get(ft_name, ft_name.split('_')[0])
            if base_name in ds_results['frozen']:
                if task_type == 'classification':
                    primary = 'f1_weighted' if len(np.unique(labels)) > 2 else 'auc'
                    base = ds_results['frozen'][base_name].get(f'{primary}_mean')
                    cur = entry.get(f'{primary}_mean')
                    if base is not None and cur is not None:
                        delta = cur - base
                        sign = '+' if delta >= 0 else ''
                        print(f"      vs frozen {base_name} ({primary}): {sign}{delta:.4f}")
                else:
                    base_r2 = ds_results['frozen'][base_name].get('r2_mean')
                    cur_r2 = entry.get('r2_mean')
                    if base_r2 is not None and cur_r2 is not None:
                        delta = cur_r2 - base_r2
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
