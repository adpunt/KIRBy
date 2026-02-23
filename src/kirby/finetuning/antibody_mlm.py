"""
Antibody MLM fine-tuning factory.

Self-supervised: fine-tunes masked language model objective on training fold
sequences, then computes log P(mutant_AA | context) at mutation positions
as representation. No labels used in training.

Requires: torch, transformers
"""

from typing import Callable, List, Optional, Tuple

import numpy as np

from kirby.finetuning._base import (
    fold_cache_key,
    get_device,
    load_or_compute,
)


# Model configs: (HF model ID, tokenizer approach, model class, embed_dim)
_MODEL_CONFIGS = {
    'esm2': (
        'facebook/esm2_t33_650M_UR50D',
        'auto',
        'EsmForMaskedLM',
        1280,
    ),
    'esm2_small': (
        'facebook/esm2_t6_8M_UR50D',
        'auto',
        'EsmForMaskedLM',
        320,
    ),
    'antiberty': (
        'alchemab/antiberta2',
        'roformer',
        'RoFormerForMaskedLM',
        512,
    ),
}


def _finetune_mlm_and_extract(
    train_seqs: List[str],
    val_seqs: List[str],
    train_mutations: List[List],
    val_mutations: List[List],
    model_name: str,
    mlm_epochs: int,
    lr: float,
    batch_size: int,
    mask_prob: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fine-tune MLM on training sequences, then extract mutation-site log-probs."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    config = _MODEL_CONFIGS[model_name]
    model_id, tok_approach, model_cls_name, embed_dim = config
    device = get_device()

    # Load tokenizer
    if tok_approach == 'roformer':
        from transformers import RoFormerTokenizer
        tokenizer = RoFormerTokenizer.from_pretrained(model_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load MLM model
    if model_cls_name == 'EsmForMaskedLM':
        from transformers import EsmForMaskedLM
        model = EsmForMaskedLM.from_pretrained(model_id)
    elif model_cls_name == 'RoFormerForMaskedLM':
        from transformers import RoFormerForMaskedLM
        model = RoFormerForMaskedLM.from_pretrained(model_id)
    else:
        from transformers import AutoModelForMaskedLM
        model = AutoModelForMaskedLM.from_pretrained(model_id)

    model = model.to(device)

    # Tokenize training sequences
    def _tokenize(seqs, max_length=256):
        if tok_approach == 'roformer':
            seqs = [' '.join(list(s)) for s in seqs]
        return tokenizer(
            seqs, padding=True, truncation=True,
            max_length=max_length, return_tensors='pt',
        )

    # === Phase 1: MLM fine-tuning on training fold (self-supervised) ===
    train_enc = _tokenize(train_seqs)
    mask_token_id = tokenizer.mask_token_id

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(mlm_epochs):
        model.train()
        # Process in batches
        n = train_enc['input_ids'].shape[0]
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]

            input_ids = train_enc['input_ids'][batch_idx].clone().to(device)
            attention_mask = train_enc['attention_mask'][batch_idx].to(device)
            labels = input_ids.clone()

            # Random masking
            prob_matrix = torch.full(input_ids.shape, mask_prob)
            # Don't mask special tokens
            special_mask = (input_ids == tokenizer.pad_token_id)
            if tokenizer.cls_token_id is not None:
                special_mask |= (input_ids == tokenizer.cls_token_id)
            if tokenizer.sep_token_id is not None:
                special_mask |= (input_ids == tokenizer.sep_token_id)
            if tokenizer.eos_token_id is not None:
                special_mask |= (input_ids == tokenizer.eos_token_id)
            prob_matrix.masked_fill_(special_mask, 0.0)

            masked_indices = torch.bernoulli(prob_matrix).bool()
            labels[~masked_indices] = -100  # Only compute loss on masked tokens
            input_ids[masked_indices] = mask_token_id

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            outputs.loss.backward()
            optimizer.step()

    # === Phase 2: Extract mutation-site log-probabilities ===
    model.eval()

    def _extract_logprobs(seqs, mutation_list):
        """For each sample, compute log P(mutant_AA | context) at each mutation site."""
        enc = _tokenize(seqs)
        all_features = []

        with torch.no_grad():
            for i in range(len(seqs)):
                input_ids = enc['input_ids'][i:i+1].to(device)
                attention_mask = enc['attention_mask'][i:i+1].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[0]  # (seq_len, vocab_size)
                log_probs = torch.log_softmax(logits, dim=-1)

                muts = mutation_list[i] if mutation_list else []
                if not muts:
                    # No mutations: use mean log-prob across sequence as single feature
                    mean_lp = log_probs.mean().item()
                    all_features.append([mean_lp, 0.0, 0.0, 0.0, 0.0])
                    continue

                site_scores = []
                for m in muts:
                    # Parse mutation position and target AA
                    if isinstance(m, dict):
                        pos = m.get('position', m.get('site', 0))
                        target_aa = m.get('mutant', m.get('to', ''))
                    elif isinstance(m, str) and ':' in m:
                        try:
                            residue_part = m.split(':')[1]
                            pos = int(''.join(c for c in residue_part[1:-1] if c.isdigit()))
                            target_aa = residue_part[-1]
                        except (IndexError, ValueError):
                            pos = 0
                            target_aa = ''
                    else:
                        pos = 0
                        target_aa = ''

                    # Token position (account for [CLS])
                    tok_pos = min(pos + 1, log_probs.shape[0] - 1)

                    if target_aa and target_aa in tokenizer.get_vocab():
                        target_id = tokenizer.convert_tokens_to_ids(target_aa)
                        score = log_probs[tok_pos, target_id].item()
                    else:
                        score = log_probs[tok_pos].max().item()

                    site_scores.append(score)

                scores_arr = np.array(site_scores)
                all_features.append([
                    float(scores_arr.mean()),
                    float(scores_arr.min()),
                    float(scores_arr.max()),
                    float(scores_arr.sum()),
                    float(np.sum(scores_arr < np.log(0.01))),  # n_unfavorable
                ])

        return np.array(all_features, dtype=np.float32)

    train_embs = _extract_logprobs(train_seqs, train_mutations)
    val_embs = _extract_logprobs(val_seqs, val_mutations)

    # Cleanup
    del model, optimizer
    if device == 'cuda':
        torch.cuda.empty_cache()

    return train_embs, val_embs


def make_antibody_mlm_factory(
    heavy_seqs: List[str],
    mutations: Optional[List[List]],
    dataset_name: str,
    base_model: str = 'esm2',
    mlm_epochs: int = 3,
    lr: float = 5e-5,
    batch_size: int = 8,
    mask_prob: float = 0.15,
    cache: bool = True,
) -> Callable:
    """Create an MLM fine-tuning factory for antibody sequences.

    Self-supervised: fine-tunes MLM objective on training fold sequences,
    then computes log P(mutant_AA | context) at mutation positions.
    No labels used in training (train_labels accepted for signature compat).

    Args:
        heavy_seqs: Heavy chain sequences.
        mutations: Per-sample mutation lists.
        dataset_name: Used for cache key.
        base_model: Model key: 'esm2', 'esm2_small', 'antiberty'.
        mlm_epochs: MLM fine-tuning epochs (default: 3).
        lr: Learning rate (default: 5e-5).
        batch_size: Batch size (default: 8).
        mask_prob: Masking probability (default: 0.15).
        cache: Cache fold embeddings to disk.

    Returns:
        Factory callable: (train_idx, val_idx, train_labels) -> (train_embs, val_embs)
    """
    if base_model not in _MODEL_CONFIGS:
        raise ValueError(
            f"Unknown base_model '{base_model}'. "
            f"Available: {list(_MODEL_CONFIGS.keys())}"
        )

    seqs = list(heavy_seqs)
    has_mutations = mutations is not None and any(len(m) > 0 for m in mutations)

    hyperparams = {
        'base_model': base_model,
        'mlm_epochs': mlm_epochs,
        'lr': lr,
        'mask_prob': mask_prob,
    }

    rep_name = f'mlm_{base_model}'

    def factory(
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        train_labels: np.ndarray,  # Unused (self-supervised), kept for interface compat
    ) -> Tuple[np.ndarray, np.ndarray]:
        key = fold_cache_key(dataset_name, rep_name, train_idx, hyperparams)

        def _compute():
            train_seqs = [seqs[i] for i in train_idx]
            val_seqs = [seqs[i] for i in val_idx]
            train_muts = [mutations[i] for i in train_idx] if has_mutations else None
            val_muts = [mutations[i] for i in val_idx] if has_mutations else None

            return _finetune_mlm_and_extract(
                train_seqs=train_seqs,
                val_seqs=val_seqs,
                train_mutations=train_muts,
                val_mutations=val_muts,
                model_name=base_model,
                mlm_epochs=mlm_epochs,
                lr=lr,
                batch_size=batch_size,
                mask_prob=mask_prob,
            )

        return load_or_compute(key, _compute, cache=cache)

    return factory
