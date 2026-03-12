"""
DTA protein-side LoRA fine-tuning factory.

Fine-tunes protein language models (ESM-2, ESM-C, ESM3, SaProt, AMPLIFY, MINT)
with LoRA on the protein side of drug-target affinity prediction. The training
signal comes from affinity labels, with frozen drug ECFP4 features concatenated
to provide drug context.

Architecture during training:
    protein_seq → LoRA-PLM → protein_emb (D)
    drug_smiles → frozen ECFP4 (2048D)
    concat(protein_emb, drug_ecfp4) → regression_head → affinity

Extraction: only protein embeddings are returned (no drug features).
The hybrid search handles drug+protein combination downstream.

Key DTA-specific design:
    - Many pairs share the same protein — deduplicates for efficient extraction
    - Training uses pair-level labels but proteins repeat across drugs
    - Frozen drug ECFP4 provides drug context during training without fine-tuning

Requires: torch, transformers, peft
"""

from typing import Callable, List, Tuple

import numpy as np

from kirby.finetuning._base import (
    fold_cache_key,
    get_device,
    load_or_compute,
)

# Model configs: (HF model ID, embed_dim, target_modules)
_MODEL_CONFIGS = {
    'esm2': (
        'facebook/esm2_t33_650M_UR50D',
        1280,
        ['query', 'value'],
    ),
    'esmc': (
        'EvolutionaryScale/esmc-600m-2024-12',
        1152,
        ['query', 'value'],
    ),
    'esm3': (
        'EvolutionaryScale/esm3-sm-open-v1',
        1536,
        ['query', 'value'],
    ),
    'saprot': (
        'westlake-repl/SaProt_650M_AF2',
        1280,
        ['query', 'value'],
    ),
    'amplify': (
        'chandar-lab/AMPLIFY_350M',
        960,
        ['query', 'value'],
    ),
    'mint': (
        'AmelieSchreiber/MINT',
        1280,
        ['query', 'value'],
    ),
}


def _finetune_and_extract(
    train_protein_seqs: List[str],
    train_drug_smiles: List[str],
    val_protein_seqs: List[str],
    val_drug_smiles: List[str],
    train_labels: np.ndarray,
    model_name: str,
    lora_rank: int,
    lora_alpha: int,
    epochs: int,
    lr: float,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Core fine-tuning + extraction for DTA protein models.

    Training: LoRA-PLM(protein) concat ECFP4(drug) → head → affinity
    Extraction: LoRA-PLM(protein) → mean-pooled embedding only
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModel
    from peft import LoraConfig, get_peft_model, TaskType

    from kirby.representations.molecular import create_ecfp4

    model_id, embed_dim, target_modules = _MODEL_CONFIGS[model_name]
    device = get_device()
    drug_dim = 2048  # ECFP4

    # --- Frozen drug features ---
    all_drug_smiles = list(set(train_drug_smiles + val_drug_smiles))
    drug_to_idx = {s: i for i, s in enumerate(all_drug_smiles)}
    all_drug_feats = create_ecfp4(all_drug_smiles)  # (n_unique_drugs, 2048)

    train_drug_feats = np.array([all_drug_feats[drug_to_idx[s]] for s in train_drug_smiles])
    train_drug_tensor = torch.FloatTensor(train_drug_feats)

    # --- Protein LM + LoRA ---
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias='none',
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    peft_model = get_peft_model(base_model, lora_config)

    # Regression head: protein_emb + drug_ecfp4 → affinity
    head = nn.Sequential(
        nn.Linear(embed_dim + drug_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 1),
    )

    peft_model = peft_model.to(device)
    head = head.to(device)

    # --- Tokenize all training proteins ---
    def _tokenize(seqs, max_length=1024):
        spaced = [' '.join(list(s)) for s in seqs]
        return tokenizer(
            spaced, padding=True, truncation=True,
            max_length=max_length, return_tensors='pt',
        )

    train_enc = _tokenize(train_protein_seqs)

    # Early stopping holdout (10% of training pairs)
    n_train = len(train_protein_seqs)
    n_holdout = max(1, int(0.1 * n_train))
    perm = np.random.RandomState(42).permutation(n_train)
    ho_idx = perm[:n_holdout]
    fit_idx = perm[n_holdout:]

    fit_dataset = TensorDataset(
        train_enc['input_ids'][fit_idx],
        train_enc['attention_mask'][fit_idx],
        train_drug_tensor[fit_idx],
        torch.FloatTensor(train_labels[fit_idx]),
    )
    fit_loader = DataLoader(fit_dataset, batch_size=batch_size, shuffle=True)

    ho_ids = train_enc['input_ids'][ho_idx].to(device)
    ho_mask = train_enc['attention_mask'][ho_idx].to(device)
    ho_drug = train_drug_tensor[ho_idx].to(device)
    ho_labels_t = torch.FloatTensor(train_labels[ho_idx]).to(device)

    # --- Train ---
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
        for batch_ids, batch_mask, batch_drug, batch_y in fit_loader:
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)
            batch_drug = batch_drug.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            out = peft_model(input_ids=batch_ids, attention_mask=batch_mask)
            prot_emb = out.last_hidden_state.mean(dim=1)
            combined = torch.cat([prot_emb, batch_drug], dim=1)
            pred = head(combined).squeeze(-1)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

        # Holdout check
        peft_model.eval()
        head.eval()
        with torch.no_grad():
            ho_out = peft_model(input_ids=ho_ids, attention_mask=ho_mask)
            ho_prot = ho_out.last_hidden_state.mean(dim=1)
            ho_combined = torch.cat([ho_prot, ho_drug], dim=1)
            ho_pred = head(ho_combined).squeeze(-1)
            ho_loss = criterion(ho_pred, ho_labels_t).item()

        if ho_loss < best_ho_loss:
            best_ho_loss = ho_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # --- Extract protein-only embeddings ---
    # Deduplicate: compute unique protein embeddings, expand to pairs
    peft_model.eval()

    def _extract_unique(seqs):
        unique_seqs = sorted(set(seqs))
        seq_to_idx = {s: i for i, s in enumerate(unique_seqs)}

        enc = _tokenize(unique_seqs)
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(unique_seqs), batch_size):
                j = min(i + batch_size, len(unique_seqs))
                ids = enc['input_ids'][i:j].to(device)
                mask = enc['attention_mask'][i:j].to(device)
                out = peft_model(input_ids=ids, attention_mask=mask)
                pooled = out.last_hidden_state.mean(dim=1)
                all_embs.append(pooled.cpu().numpy())

        unique_embs = np.concatenate(all_embs, axis=0).astype(np.float32)
        # Expand back to per-pair
        pair_indices = np.array([seq_to_idx[s] for s in seqs])
        return unique_embs[pair_indices]

    train_embs = _extract_unique(train_protein_seqs)
    val_embs = _extract_unique(val_protein_seqs)

    # Cleanup
    del peft_model, head, optimizer
    if device == 'cuda':
        torch.cuda.empty_cache()

    return train_embs, val_embs


def make_dta_protein_lora_factory(
    protein_seqs: List[str],
    drug_smiles: List[str],
    dataset_name: str,
    base_model: str = 'esm2',
    lora_rank: int = 8,
    lora_alpha: int = 16,
    epochs: int = 5,
    lr: float = 2e-5,
    batch_size: int = 8,
    cache: bool = True,
) -> Callable:
    """Create a LoRA fine-tuning factory for protein LMs on DTA tasks.

    Trains LoRA-adapted PLM with drug ECFP4 context to predict affinity,
    then extracts protein-only embeddings. Drug context is frozen and used
    only during training — not in the extracted embeddings.

    Supported base_model values:
        esm2 (1280D), esmc (1152D), esm3 (1536D), saprot (1280D),
        amplify (960D), mint (1280D)

    Args:
        protein_seqs: Protein sequences per pair (with repeats).
        drug_smiles: Drug SMILES per pair (with repeats).
        dataset_name: Used for cache key ('davis', 'kiba').
        base_model: Model key from _MODEL_CONFIGS.
        lora_rank: LoRA rank (default: 8, recommended 4-8 for >5K samples).
        lora_alpha: LoRA alpha (default: 16).
        epochs: Fine-tuning epochs per fold (default: 5).
        lr: Learning rate (default: 2e-5, lower than molecular due to larger model).
        batch_size: Batch size (default: 8, smaller due to PLM memory).
        cache: Cache fold embeddings to disk.

    Returns:
        Factory callable: (train_idx, val_idx, train_labels) -> (train_embs, val_embs)
        Output embeddings are protein-only.
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

    rep_name = f'dta_lora_{base_model}'

    def factory(
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        train_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        key = fold_cache_key(dataset_name, rep_name, train_idx, hyperparams)

        def _compute():
            train_prots = [protein_seqs[i] for i in train_idx]
            train_drugs = [drug_smiles[i] for i in train_idx]
            val_prots = [protein_seqs[i] for i in val_idx]
            val_drugs = [drug_smiles[i] for i in val_idx]

            return _finetune_and_extract(
                train_protein_seqs=train_prots,
                train_drug_smiles=train_drugs,
                val_protein_seqs=val_prots,
                val_drug_smiles=val_drugs,
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
