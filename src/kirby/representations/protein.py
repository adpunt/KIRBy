"""
Protein Representations — General-Purpose

Representations for arbitrary protein sequences (not antibody-specific).
Used primarily for drug-target affinity (DTA) prediction where target
proteins are kinases, GPCRs, etc.

Functions:
- create_esm2_protein_embeddings: ESM-2 mean-pooled embeddings (1280D)
- create_aido_protein_embeddings: AIDO.Protein-16B MoE embeddings (2304D)
- create_protein_aac: amino acid composition (20D)
- create_protein_descriptors: biophysical descriptors (~30D)
- create_protein_kmer: k-mer frequency (default k=3, 8000D)
"""

import os
import numpy as np
from typing import List, Optional

# Standard amino acid order (matches antibody.py)
STANDARD_AAS = 'ACDEFGHIKLMNPQRSTVWY'

# Physicochemical properties per amino acid
AA_PROPERTIES = {
    'A': {'hydrophobicity':  1.8, 'charge':  0.0, 'mw':  89.09, 'volume':  88.6, 'polarity': 0.0, 'flexibility': 0.360},
    'R': {'hydrophobicity': -4.5, 'charge':  1.0, 'mw': 174.20, 'volume': 173.4, 'polarity': 1.0, 'flexibility': 0.530},
    'N': {'hydrophobicity': -3.5, 'charge':  0.0, 'mw': 132.12, 'volume': 114.1, 'polarity': 1.0, 'flexibility': 0.460},
    'D': {'hydrophobicity': -3.5, 'charge': -1.0, 'mw': 133.10, 'volume': 111.1, 'polarity': 1.0, 'flexibility': 0.510},
    'C': {'hydrophobicity':  2.5, 'charge':  0.0, 'mw': 121.15, 'volume': 108.5, 'polarity': 0.0, 'flexibility': 0.350},
    'E': {'hydrophobicity': -3.5, 'charge': -1.0, 'mw': 147.13, 'volume': 138.4, 'polarity': 1.0, 'flexibility': 0.500},
    'Q': {'hydrophobicity': -3.5, 'charge':  0.0, 'mw': 146.15, 'volume': 143.8, 'polarity': 1.0, 'flexibility': 0.490},
    'G': {'hydrophobicity': -0.4, 'charge':  0.0, 'mw':  75.03, 'volume':  60.1, 'polarity': 0.0, 'flexibility': 0.540},
    'H': {'hydrophobicity': -3.2, 'charge':  0.1, 'mw': 155.16, 'volume': 153.2, 'polarity': 1.0, 'flexibility': 0.320},
    'I': {'hydrophobicity':  4.5, 'charge':  0.0, 'mw': 131.17, 'volume': 166.7, 'polarity': 0.0, 'flexibility': 0.460},
    'L': {'hydrophobicity':  3.8, 'charge':  0.0, 'mw': 131.17, 'volume': 166.7, 'polarity': 0.0, 'flexibility': 0.400},
    'K': {'hydrophobicity': -3.9, 'charge':  1.0, 'mw': 146.19, 'volume': 168.6, 'polarity': 1.0, 'flexibility': 0.535},
    'M': {'hydrophobicity':  1.9, 'charge':  0.0, 'mw': 149.21, 'volume': 162.9, 'polarity': 0.0, 'flexibility': 0.410},
    'F': {'hydrophobicity':  2.8, 'charge':  0.0, 'mw': 165.19, 'volume': 189.9, 'polarity': 0.0, 'flexibility': 0.310},
    'P': {'hydrophobicity': -1.6, 'charge':  0.0, 'mw': 115.13, 'volume': 112.7, 'polarity': 0.0, 'flexibility': 0.510},
    'S': {'hydrophobicity': -0.8, 'charge':  0.0, 'mw': 105.09, 'volume':  89.0, 'polarity': 1.0, 'flexibility': 0.510},
    'T': {'hydrophobicity': -0.7, 'charge':  0.0, 'mw': 119.12, 'volume': 116.1, 'polarity': 1.0, 'flexibility': 0.440},
    'W': {'hydrophobicity': -0.9, 'charge':  0.0, 'mw': 204.23, 'volume': 227.8, 'polarity': 0.0, 'flexibility': 0.310},
    'Y': {'hydrophobicity': -1.3, 'charge':  0.0, 'mw': 181.19, 'volume': 193.6, 'polarity': 1.0, 'flexibility': 0.420},
    'V': {'hydrophobicity':  4.2, 'charge':  0.0, 'mw': 117.15, 'volume': 140.0, 'polarity': 0.0, 'flexibility': 0.390},
}


def _esm2_per_residue_cache_dir() -> 'Path':
    """Return cache directory for per-residue ESM-2 embeddings."""
    from pathlib import Path
    cache_dir = Path(os.environ.get('KIRBY_CACHE_DIR',
                     str(Path.home() / '.cache' / 'kirby'))) / 'esm2_per_residue'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _seq_hash(seq: str) -> str:
    import hashlib
    return hashlib.sha256(seq.upper().strip().encode()).hexdigest()[:16]


def create_esm2_protein_embeddings(
    sequences: List[str],
    model_name: str = 'facebook/esm2_t33_650M_UR50D',
    device: str = 'cpu',
    batch_size: int = 4,
    cache_per_residue: bool = True,
) -> np.ndarray:
    """Extract ESM-2 mean-pooled embeddings for protein sequences.

    Args:
        sequences: List of protein amino acid sequences.
        model_name: HuggingFace ESM-2 model name.
        device: 'cpu' or 'cuda'.
        batch_size: Sequences per batch (proteins can be long, use small batch).
        cache_per_residue: If True, cache per-residue embeddings to disk for
            later use by pocket-local features, contact graphs, etc.

    Returns:
        np.ndarray: (n_sequences, 1280) embeddings.
    """
    import torch
    try:
        from transformers import AutoTokenizer, EsmModel
    except ImportError:
        raise ImportError("ESM-2 requires transformers: pip install transformers")

    per_res_dir = _esm2_per_residue_cache_dir() if cache_per_residue else None
    n = len(sequences)

    # Check which sequences already have cached per-residue embeddings
    all_embeddings = [None] * n
    need_compute = []
    if per_res_dir is not None:
        for i, seq in enumerate(sequences):
            cache_file = per_res_dir / f'{_seq_hash(seq)}_mean.npy'
            if cache_file.exists():
                all_embeddings[i] = np.load(cache_file)
            else:
                need_compute.append(i)
    else:
        need_compute = list(range(n))

    n_cached = n - len(need_compute)
    if n_cached > 0:
        print(f"  ESM-2: {n_cached}/{n} sequences cached, computing {len(need_compute)} remaining...")

    if not need_compute:
        result = np.stack(all_embeddings, axis=0)
        print(f"  Generated ESM-2 protein embeddings: {result.shape} (all cached)")
        return result

    print(f"  Loading ESM-2 ({model_name.split('/')[-1]}) for protein embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device).eval()

    with torch.no_grad():
        for batch_start in range(0, len(need_compute), batch_size):
            batch_indices = need_compute[batch_start:batch_start + batch_size]
            batch = [sequences[i].upper().strip()[:1022] for i in batch_indices]
            inputs = tokenizer(
                batch, return_tensors='pt', padding=True,
                truncation=True, max_length=1024
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state  # (B, L, 1280)

            # Mean pool excluding BOS/EOS
            attention_mask = inputs['attention_mask'].clone()
            attention_mask[:, 0] = 0
            for j in range(len(batch)):
                last_tok = attention_mask[j].sum().item()
                if last_tok > 0:
                    attention_mask[j, int(last_tok)] = 0
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_emb = (hidden * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            mean_emb = (sum_emb / count).cpu().numpy()

            # Save per-residue and mean embeddings per sequence
            for j, idx in enumerate(batch_indices):
                all_embeddings[idx] = mean_emb[j]
                if per_res_dir is not None:
                    sh = _seq_hash(sequences[idx])
                    np.save(per_res_dir / f'{sh}_mean.npy', mean_emb[j])
                    # Save per-residue (excluding BOS/EOS padding)
                    seq_len = int(attention_mask[j].sum().item())
                    if seq_len > 0:
                        # Residue embeddings: positions 1..seq_len (skip BOS at 0)
                        per_res = hidden[j, 1:1 + seq_len].cpu().numpy()
                        np.save(per_res_dir / f'{sh}_per_residue.npy', per_res)

            done = min(batch_start + batch_size, len(need_compute))
            if done < len(need_compute):
                print(f"    Progress: {done}/{len(need_compute)}")

    result = np.stack(all_embeddings, axis=0)
    print(f"  Generated ESM-2 protein embeddings: {result.shape}")
    return result


def load_esm2_per_residue(sequence: str) -> Optional[np.ndarray]:
    """Load cached per-residue ESM-2 embeddings for a single protein.

    Returns:
        np.ndarray of shape (seq_len, 1280) or None if not cached.
        Call create_esm2_protein_embeddings() first to populate the cache.
    """
    cache_file = _esm2_per_residue_cache_dir() / f'{_seq_hash(sequence)}_per_residue.npy'
    if cache_file.exists():
        return np.load(cache_file)
    return None


def create_esm2_midlayers_protein_embeddings(
    sequences: List[str],
    model_name: str = 'facebook/esm2_t33_650M_UR50D',
    device: str = 'cpu',
    batch_size: int = 4,
    layer_start: int = 20,
    layer_end: int = 33,
) -> np.ndarray:
    """ESM-2 intermediate layer averaged embeddings for protein sequences.

    Averages hidden states from layers [layer_start, layer_end] instead of using
    the final layer only. Published evidence shows this improves unsupervised
    clustering ARI from 0.268 → 0.354 for ESM-2 650M.

    Args:
        sequences: List of protein amino acid sequences.
        model_name: HuggingFace ESM-2 model name.
        device: 'cpu' or 'cuda'.
        batch_size: Sequences per batch.
        layer_start: First layer to average (0-indexed).
        layer_end: Last layer to average (inclusive).

    Returns:
        np.ndarray: (n_sequences, 1280) embeddings.
    """
    import torch
    try:
        from transformers import AutoTokenizer, EsmModel
    except ImportError:
        raise ImportError("ESM-2 requires transformers: pip install transformers")

    print(f"  Loading ESM-2 midlayers ({model_name.split('/')[-1]}, layers {layer_start}-{layer_end})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name, output_hidden_states=True).to(device).eval()

    all_embeddings = []
    n = len(sequences)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = [s.upper().strip()[:1022] for s in sequences[start:start + batch_size]]
            inputs = tokenizer(
                batch, return_tensors='pt', padding=True,
                truncation=True, max_length=1024
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            # Average selected hidden layers
            selected = torch.stack(
                outputs.hidden_states[layer_start:layer_end + 1], dim=0
            ).mean(dim=0)  # (B, L, D)
            # Mean pool excluding BOS/EOS
            attention_mask = inputs['attention_mask'].clone()
            attention_mask[:, 0] = 0
            for j in range(len(batch)):
                last_tok = attention_mask[j].sum().item()
                if last_tok > 0:
                    attention_mask[j, int(last_tok)] = 0
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_emb = (selected * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            mean_emb = (sum_emb / count).cpu().numpy()
            all_embeddings.append(mean_emb)

            end = min(start + batch_size, n)
            if end < n and end % 50 == 0:
                print(f"    Progress: {end}/{n}")

    result = np.concatenate(all_embeddings, axis=0)
    print(f"  Generated ESM-2 midlayer protein embeddings: {result.shape}")
    return result


def create_esmc_protein_embeddings(
    sequences: List[str],
    device: str = 'cpu',
) -> np.ndarray:
    """Extract ESM-C 600M mean-pooled embeddings for protein sequences.

    Drop-in ESM-2 replacement. Matches ESM-2 3B quality at 5x fewer params.
    Uses EvolutionaryScale's `esm` package.

    Note: Non-commercial license (EvolutionaryScale Cambrian).

    Args:
        sequences: List of protein amino acid sequences.
        device: 'cpu' or 'cuda'.

    Returns:
        np.ndarray: (n_sequences, 1152) embeddings.

    Requires: pip install esm
    """
    import torch
    try:
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
    except ImportError:
        raise ImportError("ESM-C requires the esm package: pip install esm")

    print("  Loading ESM-C 600M for protein embeddings...")
    model = ESMC.from_pretrained("esmc_600m").to(device)
    model.eval()

    config = LogitsConfig(sequence=True, return_embeddings=True)
    all_embeddings = []
    n = len(sequences)

    with torch.no_grad():
        for i, seq in enumerate(sequences):
            seq = seq.upper().strip()[:2048]  # ESM-C max length
            protein = ESMProtein(sequence=seq)
            protein_tensor = model.encode(protein)
            output = model.logits(protein_tensor, config)
            # output.embeddings: (1, L, 1152) — mean-pool excluding BOS/EOS
            emb = output.embeddings[:, 1:-1, :].mean(dim=1).squeeze(0).cpu().numpy()
            all_embeddings.append(emb)

            if (i + 1) % 50 == 0:
                print(f"    Progress: {i + 1}/{n}")

    result = np.vstack(all_embeddings)
    print(f"  Generated ESM-C protein embeddings: {result.shape}")
    return result


def create_esm3_protein_embeddings(
    sequences: List[str],
    device: str = 'cpu',
) -> np.ndarray:
    """Extract ESM3 mean-pooled embeddings for protein sequences.

    ESM3 is a multimodal protein language model from EvolutionaryScale that
    reasons over sequence, structure, and function simultaneously.
    Uses the same `esm` package as ESM-C.

    Args:
        sequences: List of protein amino acid sequences.
        device: 'cpu' or 'cuda'.

    Returns:
        np.ndarray: (n_sequences, 1536) embeddings.

    Requires: pip install esm (Python >= 3.10)
    """
    import torch
    try:
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein, LogitsConfig
    except ImportError:
        raise ImportError("ESM3 requires the esm package: pip install esm")

    print("  Loading ESM3 (esm3_sm_open_v1) for protein embeddings...")
    model = ESM3.from_pretrained("esm3_sm_open_v1")
    if device == 'cuda' and not torch.cuda.is_bf16_supported():
        model = model.half()
    model = model.to(device).eval()

    config = LogitsConfig(sequence=True, return_embeddings=True)
    all_embeddings = []
    n = len(sequences)

    with torch.no_grad():
        for i, seq in enumerate(sequences):
            seq = seq.upper().strip()[:2048]  # ESM3 max length
            try:
                protein = ESMProtein(sequence=seq)
                protein_tensor = model.encode(protein)
                output = model.logits(protein_tensor, config)
                # output.embeddings: (1, L, 1536) — mean-pool excluding BOS/EOS
                emb = output.embeddings[:, 1:-1, :].mean(dim=1).squeeze(0).cpu().numpy()
                all_embeddings.append(emb)
            except Exception as e:
                print(f"    WARNING: ESM3 failed on seq {i} (len={len(seq)}): {e}")
                all_embeddings.append(np.zeros(1536, dtype=np.float32))

            if (i + 1) % 50 == 0:
                print(f"    Progress: {i + 1}/{n}")

    result = np.vstack(all_embeddings)
    print(f"  Generated ESM3 protein embeddings: {result.shape}")
    return result


# ── AIDO.Protein-16B (MoE protein foundation model) ──────────────────────────

# Global singletons to avoid reloading the large model
_AIDO_MODEL = None
_AIDO_TOKENIZER = None


def create_aido_protein_embeddings(
    sequences: List[str],
    model_name: str = 'genbio-ai/AIDO.Protein-16B',
    device: str = 'cpu',
    batch_size: int = 1,
) -> np.ndarray:
    """Extract AIDO.Protein-16B mean-pooled embeddings for protein sequences.

    AIDO.Protein-16B is a Mixture-of-Experts (MoE) protein foundation model
    from GenBio AI with 16B total parameters but only ~4B active per token
    (top-2 routing across 8 experts). Uses RoPE positional embeddings and
    SwiGLU activation.

    WARNING: This model requires ~32GB VRAM in float16. Use batch_size=1
    to avoid OOM. On CPU this will be extremely slow. CUDA strongly
    recommended.

    Args:
        sequences: List of protein amino acid sequences.
        model_name: HuggingFace model name.
        device: 'cpu' or 'cuda'. CUDA strongly recommended.
        batch_size: Sequences per batch (default 1 due to model size).

    Returns:
        np.ndarray: (n_sequences, 2304) embeddings.

    Requires: pip install transformers torch
    Model auto-downloaded: genbio-ai/AIDO.Protein-16B (~64GB)
    """
    import torch
    global _AIDO_MODEL, _AIDO_TOKENIZER

    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError(
            "AIDO.Protein requires transformers: pip install transformers"
        )

    if _AIDO_MODEL is None:
        print(f"  Loading AIDO.Protein-16B ({model_name.split('/')[-1]}, float16)...")
        print(f"  WARNING: Model requires ~32GB VRAM in float16.")
        _AIDO_TOKENIZER = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        _AIDO_MODEL = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float16
        ).to(device).eval()

    model = _AIDO_MODEL
    tokenizer = _AIDO_TOKENIZER

    all_embeddings = []
    n = len(sequences)
    embed_dim = 2304  # model.config.hidden_size

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = [
                s.upper().strip()[:2046]  # max_position_embeddings=2048, leave room for special tokens
                for s in sequences[start:start + batch_size]
            ]
            for j, seq in enumerate(batch):
                idx = start + j
                try:
                    inputs = tokenizer(
                        seq, return_tensors='pt', padding=False,
                        truncation=True, max_length=2048,
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    # last_hidden_state: (1, L, 2304)
                    # Mean pool excluding special tokens (SEP at end, PAD=0)
                    hidden = outputs.last_hidden_state  # (1, L, 2304)
                    if 'attention_mask' in inputs:
                        mask = inputs['attention_mask'].clone()
                    else:
                        mask = torch.ones(hidden.shape[:2], device=device)
                    # Exclude last token (SEP) if present
                    n_tokens = mask[0].sum().item()
                    if n_tokens > 1:
                        mask[0, int(n_tokens) - 1] = 0
                    mask = mask.unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    emb = pooled.squeeze(0).cpu().float().numpy()
                    all_embeddings.append(emb)
                except Exception as e:
                    print(f"    WARNING: AIDO.Protein failed on seq {idx} "
                          f"(len={len(seq)}): {e}")
                    all_embeddings.append(np.zeros(embed_dim, dtype=np.float32))

            end = min(start + batch_size, n)
            if end < n and end % 20 == 0:
                print(f"    Progress: {end}/{n}")

    result = np.vstack(all_embeddings)
    print(f"  Generated AIDO.Protein embeddings: {result.shape}")
    return result


# ── SaProt (structure-aware protein LM) ──────────────────────────────────────

# Global singletons for ProstT5 and SaProt models (avoid re-loading)
_PROSTT5_MODEL = None
_PROSTT5_TOKENIZER = None
_SAPROT_MODEL = None
_SAPROT_TOKENIZER = None


def _predict_3di_with_prostt5(
    sequences: List[str],
    device: str = 'cpu',
) -> List[str]:
    """Predict Foldseek 3Di structural tokens from amino acid sequences using ProstT5.

    ProstT5 (Rostlab/ProstT5) translates AA sequences to 3Di tokens directly
    without requiring structure prediction. Pure Python, no binary dependencies.

    Args:
        sequences: List of protein amino acid sequences.
        device: 'cpu' or 'cuda'.

    Returns:
        List of 3Di strings (one per input sequence).
    """
    import torch
    global _PROSTT5_MODEL, _PROSTT5_TOKENIZER

    try:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
    except ImportError:
        raise ImportError("ProstT5 requires transformers: pip install transformers")

    if _PROSTT5_MODEL is None:
        print("  Loading ProstT5 (Rostlab/ProstT5) for 3Di prediction...")
        _PROSTT5_TOKENIZER = T5Tokenizer.from_pretrained(
            'Rostlab/ProstT5', do_lower_case=False
        )
        _PROSTT5_MODEL = T5ForConditionalGeneration.from_pretrained(
            'Rostlab/ProstT5'
        ).to(device).eval()

    model = _PROSTT5_MODEL
    tokenizer = _PROSTT5_TOKENIZER
    threedi_seqs = []

    with torch.no_grad():
        for i, seq in enumerate(sequences):
            try:
                seq_clean = seq.upper().strip()[:1000]  # ProstT5 practical limit
                # ProstT5 expects: '<AA2fold> ' prefix + space-separated residues
                input_str = '<AA2fold> ' + ' '.join(list(seq_clean))
                inputs = tokenizer(
                    input_str, return_tensors='pt',
                    add_special_tokens=True, max_length=2048,
                    truncation=True,
                ).to(device)
                output = model.generate(
                    **inputs,
                    max_new_tokens=len(seq_clean) + 10,
                    do_sample=False,
                )
                # Decode 3Di tokens, strip special tokens
                decoded = tokenizer.decode(output[0], skip_special_tokens=True)
                # ProstT5 outputs space-separated 3Di tokens
                threedi = decoded.replace(' ', '')
                # Ensure length matches (truncate or pad with '#')
                if len(threedi) < len(seq_clean):
                    threedi = threedi + '#' * (len(seq_clean) - len(threedi))
                elif len(threedi) > len(seq_clean):
                    threedi = threedi[:len(seq_clean)]
                threedi_seqs.append(threedi)
            except Exception as e:
                print(f"    WARNING: ProstT5 failed on seq {i} (len={len(seq)}): {e}")
                threedi_seqs.append('#' * len(seq.strip()))

            if (i + 1) % 20 == 0:
                print(f"    ProstT5 progress: {i + 1}/{len(sequences)}")

    return threedi_seqs


def _interleave_aa_3di(aa_seq: str, threedi_seq: str) -> str:
    """Interleave amino acid and 3Di sequences for SaProt input.

    SaProt expects alternating AA and 3Di tokens: "MdTpKv..." where
    uppercase = AA, lowercase = 3Di.

    Args:
        aa_seq: Amino acid sequence (e.g., "MTK...").
        threedi_seq: 3Di token sequence (e.g., "dpv...").

    Returns:
        Interleaved string (e.g., "MdTpKv...").
    """
    min_len = min(len(aa_seq), len(threedi_seq))
    interleaved = []
    for i in range(min_len):
        interleaved.append(aa_seq[i].upper())
        interleaved.append(threedi_seq[i].lower())
    return ''.join(interleaved)


def create_saprot_protein_embeddings(
    sequences: List[str],
    device: str = 'cpu',
    batch_size: int = 4,
) -> np.ndarray:
    """Extract SaProt 650M mean-pooled embeddings for protein sequences.

    SaProt is a structure-aware protein language model that uses Foldseek 3Di
    structural tokens interleaved with amino acid sequences. 3Di tokens are
    predicted using ProstT5 (pure Python, no binary dependencies).

    Args:
        sequences: List of protein amino acid sequences.
        device: 'cpu' or 'cuda'.
        batch_size: Sequences per batch.

    Returns:
        np.ndarray: (n_sequences, 1280) embeddings.

    Requires: pip install transformers (HuggingFace)
    Models auto-downloaded: westlake-repl/SaProt_650M_AF2 (~2.5GB),
                            Rostlab/ProstT5 (~5.5GB)
    """
    import torch
    global _SAPROT_MODEL, _SAPROT_TOKENIZER

    try:
        from transformers import EsmTokenizer, EsmModel
    except ImportError:
        raise ImportError("SaProt requires transformers: pip install transformers")

    # Step 1: Predict 3Di tokens for all sequences via ProstT5
    print("  Step 1/2: Predicting 3Di structural tokens with ProstT5...")
    threedi_seqs = _predict_3di_with_prostt5(sequences, device)

    # Step 2: Interleave AA + 3Di
    sa_seqs = [
        _interleave_aa_3di(s.upper().strip(), di)
        for s, di in zip(sequences, threedi_seqs)
    ]

    # Step 3: Load SaProt model
    if _SAPROT_MODEL is None:
        print("  Step 2/2: Loading SaProt (westlake-repl/SaProt_650M_AF2)...")
        _SAPROT_TOKENIZER = EsmTokenizer.from_pretrained(
            'westlake-repl/SaProt_650M_AF2'
        )
        _SAPROT_MODEL = EsmModel.from_pretrained(
            'westlake-repl/SaProt_650M_AF2'
        ).to(device).eval()

    model = _SAPROT_MODEL
    tokenizer = _SAPROT_TOKENIZER

    all_embeddings = []
    n = len(sequences)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = sa_seqs[start:start + batch_size]
            batch_embs = []
            for j, sa_seq in enumerate(batch):
                idx = start + j
                try:
                    inputs = tokenizer(
                        sa_seq, return_tensors='pt', padding=False,
                        truncation=True, max_length=2048,
                    ).to(device)
                    outputs = model(**inputs)
                    # Mean pool excluding BOS/EOS
                    emb = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1)
                    batch_embs.append(emb.squeeze(0).cpu().numpy())
                except Exception as e:
                    print(f"    WARNING: SaProt failed on seq {idx}: {e}")
                    batch_embs.append(np.zeros(1280, dtype=np.float32))
            all_embeddings.extend(batch_embs)

            end = min(start + batch_size, n)
            if end < n and end % 20 == 0:
                print(f"    SaProt progress: {end}/{n}")

    result = np.vstack(all_embeddings)
    print(f"  Generated SaProt protein embeddings: {result.shape}")
    return result


# ── Boltz-2 (structure prediction embeddings) ────────────────────────────────


def create_boltz2_protein_embeddings(
    sequences: List[str],
    device: str = 'cpu',
    cache_dir: str = None,
) -> np.ndarray:
    """Extract Boltz-2 single-representation protein embeddings (384D).

    Boltz-2 is a structure prediction model. We use its CLI `boltz predict`
    with `--write_embeddings` to extract per-residue single representations,
    then mean-pool to a fixed 384D vector per sequence.

    Embeddings are cached to disk keyed on sequence hash to avoid re-running.

    Note: Boltz-2 requires CUDA for practical use.

    Args:
        sequences: List of protein amino acid sequences.
        device: 'cpu' or 'cuda' (CUDA strongly recommended).
        cache_dir: Directory to cache embeddings. Defaults to ~/.cache/kirby/boltz2/.

    Returns:
        np.ndarray: (n_sequences, 384) embeddings.

    Requires: pip install boltz (CUDA recommended)
    """
    import subprocess
    import tempfile
    import hashlib as _hashlib
    from pathlib import Path

    try:
        import boltz  # noqa: F401
    except ImportError:
        raise ImportError("Boltz-2 requires the boltz package: pip install boltz")

    if cache_dir is None:
        cache_dir = str(Path(os.environ.get(
            'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')
        )) / 'boltz2')
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    n = len(sequences)
    all_embeddings = []
    print(f"  Generating Boltz-2 embeddings for {n} proteins...")

    for i, seq in enumerate(sequences):
        seq_clean = seq.upper().strip()
        seq_hash = _hashlib.sha256(seq_clean.encode()).hexdigest()[:16]
        cached_file = cache_path / f'{seq_hash}.npy'

        # Check cache
        if cached_file.exists():
            emb = np.load(cached_file)
            all_embeddings.append(emb)
            continue

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Write FASTA file
                fasta_path = os.path.join(tmpdir, 'input.fasta')
                with open(fasta_path, 'w') as f:
                    f.write(f'>seq_{i}\n{seq_clean}\n')

                out_dir = os.path.join(tmpdir, 'output')

                # Run boltz predict with embeddings
                cmd = [
                    'boltz', 'predict', fasta_path,
                    '--write_embeddings',
                    '--out_dir', out_dir,
                ]
                if device != 'cpu':
                    cmd.extend(['--accelerator', 'gpu'])

                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300,
                )
                if proc.returncode != 0:
                    raise RuntimeError(f"boltz predict failed: {proc.stderr[:200]}")

                # Find and load embeddings npz
                out_path = Path(out_dir)
                npz_files = list(out_path.rglob('*embeddings*.npz'))
                if not npz_files:
                    # Try loading from the predictions directory
                    npz_files = list(out_path.rglob('*.npz'))

                if npz_files:
                    data = np.load(npz_files[0])
                    # Boltz-2 single representations: (N_tokens, 384)
                    # Try common key names
                    for key in ['single', 's_s', 'prot_embeddings', 'embeddings']:
                        if key in data:
                            token_embs = data[key]
                            break
                    else:
                        # Use first array in npz
                        token_embs = data[list(data.keys())[0]]

                    # Mean pool over token dimension
                    if token_embs.ndim == 3:
                        token_embs = token_embs[0]  # remove batch dim
                    emb = token_embs.mean(axis=0).astype(np.float32)

                    # Handle variable embedding dims — pad/truncate to 384
                    if emb.shape[0] != 384:
                        if emb.shape[0] > 384:
                            emb = emb[:384]
                        else:
                            emb = np.pad(emb, (0, 384 - emb.shape[0]))
                else:
                    raise FileNotFoundError("No embedding files found in boltz output")

            # Cache the result
            np.save(cached_file, emb)
            all_embeddings.append(emb)

        except Exception as e:
            print(f"    WARNING: Boltz-2 failed on seq {i} (len={len(seq_clean)}): {e}")
            all_embeddings.append(np.zeros(384, dtype=np.float32))

        if (i + 1) % 10 == 0:
            print(f"    Boltz-2 progress: {i + 1}/{n}")

    result = np.vstack(all_embeddings)
    print(f"  Generated Boltz-2 protein embeddings: {result.shape}")
    return result


def create_amplify_protein_embeddings(
    sequences: List[str],
    model_name: str = 'chandar-lab/AMPLIFY_350M',
    device: str = 'cpu',
    batch_size: int = 4,
) -> np.ndarray:
    """Extract AMPLIFY 350M mean-pooled embeddings for protein sequences.

    Drop-in ESM-2 alternative. 350M parameter protein LM from Mila/Chandar Lab.

    Args:
        sequences: List of protein amino acid sequences.
        model_name: HuggingFace model name.
        device: 'cpu' or 'cuda'.
        batch_size: Sequences per batch.

    Returns:
        np.ndarray: (n_sequences, 960) embeddings.

    Requires: pip install transformers
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    print(f"  Loading AMPLIFY ({model_name.split('/')[-1]}) for protein embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()

    all_embeddings = []
    n = len(sequences)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = [s.upper().strip()[:1022] for s in sequences[start:start + batch_size]]
            inputs = tokenizer(
                batch, return_tensors='pt', padding=True,
                truncation=True, max_length=1024
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            # Mean pool excluding BOS/EOS tokens
            mask = inputs['attention_mask'].clone()
            mask[:, 0] = 0  # exclude BOS
            for j in range(mask.shape[0]):
                last_tok = mask[j].sum().item()
                if last_tok > 0:
                    mask[j, int(last_tok)] = 0  # exclude EOS
            mask = mask.unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_embeddings.append(pooled.cpu().numpy())

            end = min(start + batch_size, n)
            if end < n and end % 50 == 0:
                print(f"    Progress: {end}/{n}")

    result = np.concatenate(all_embeddings, axis=0)
    print(f"  Generated AMPLIFY protein embeddings: {result.shape}")
    return result


def create_protein_aac(sequences: List[str]) -> np.ndarray:
    """Compute amino acid composition for protein sequences.

    Args:
        sequences: List of protein amino acid sequences.

    Returns:
        np.ndarray: (n_sequences, 20) frequency matrix.
    """
    n = len(sequences)
    features = np.zeros((n, 20), dtype=np.float64)
    aa_to_idx = {aa: i for i, aa in enumerate(STANDARD_AAS)}

    for i, seq in enumerate(sequences):
        seq = seq.upper().strip()
        if not seq:
            continue
        for aa in seq:
            if aa in aa_to_idx:
                features[i, aa_to_idx[aa]] += 1
        total = sum(1 for aa in seq if aa in aa_to_idx)
        if total > 0:
            features[i] /= total

    print(f"  Generated protein AAC features: ({n}, 20)")
    return features


def create_protein_descriptors(sequences: List[str]) -> np.ndarray:
    """Compute biophysical descriptors for protein sequences.

    Features (30D), interleaved per property (mean/std/min/max):
        0-3:   hydrophobicity (mean, std, min, max)
        4-7:   charge (mean, std, min, max)
        8-11:  molecular weight (mean, std, min, max)
        12-15: volume (mean, std, min, max)
        16-19: polarity (mean, std, min, max)
        20-23: flexibility (mean, std, min, max)
        24: sequence length (normalized by 1000)
        25: molecular weight estimate (sum of residue MWs / 1e5)
        26: net charge at pH 7.4
        27: fraction hydrophobic (I, V, L, F, C, M, A, W)
        28: fraction charged (D, E, K, R, H)
        29: net charge ratio ((n_positive - n_negative) / seq_len)

    Args:
        sequences: List of protein amino acid sequences.

    Returns:
        np.ndarray: (n_sequences, 30) descriptor matrix.
    """
    props = ['hydrophobicity', 'charge', 'mw', 'volume', 'polarity', 'flexibility']
    hydrophobic = set('IVLFCMAW')
    charged = set('DEKRH')

    n = len(sequences)
    features = np.zeros((n, 30), dtype=np.float64)

    for i, seq in enumerate(sequences):
        seq = seq.upper().strip()
        if not seq:
            continue

        # Per-residue property vectors
        prop_vals = {p: [] for p in props}
        for aa in seq:
            if aa in AA_PROPERTIES:
                for p in props:
                    prop_vals[p].append(AA_PROPERTIES[aa][p])

        if not prop_vals['hydrophobicity']:
            continue

        # Statistics for each property (6 × 4 = 24D)
        for j, p in enumerate(props):
            vals = np.array(prop_vals[p])
            features[i, j * 4]     = vals.mean()
            features[i, j * 4 + 1] = vals.std()
            features[i, j * 4 + 2] = vals.min()
            features[i, j * 4 + 3] = vals.max()

        # Global descriptors
        valid_aas = [aa for aa in seq if aa in AA_PROPERTIES]
        seq_len = len(valid_aas)
        features[i, 24] = seq_len / 1000.0
        features[i, 25] = sum(AA_PROPERTIES[aa]['mw'] for aa in valid_aas) / 1e5
        features[i, 26] = sum(AA_PROPERTIES[aa]['charge'] for aa in valid_aas)
        features[i, 27] = sum(1 for aa in valid_aas if aa in hydrophobic) / max(seq_len, 1)
        features[i, 28] = sum(1 for aa in valid_aas if aa in charged) / max(seq_len, 1)

        # Simple pI estimate (count of charged residues)
        n_pos = sum(1 for aa in valid_aas if AA_PROPERTIES[aa]['charge'] > 0)
        n_neg = sum(1 for aa in valid_aas if AA_PROPERTIES[aa]['charge'] < 0)
        features[i, 29] = (n_pos - n_neg) / max(seq_len, 1)

    print(f"  Generated protein descriptors: ({n}, 30)")
    return features


def create_esmfold_protein_features(
    sequences: List[str],
    device: str = 'cpu',
) -> np.ndarray:
    """Extract structural features from ESMFold predicted structures.

    ESMFold predicts 3D structure from a single sequence (no MSA needed).
    Much faster than AlphaFold2. From the predicted structure we extract:
      - pLDDT per-residue confidence → stats (mean, min, max, std, fraction>70)
      - Contact map summary → contact density, avg contacts per residue
      - Radius of gyration estimate
      - Secondary structure fractions (helix, sheet, coil from dssp-like)

    Features (12D):
      0: mean pLDDT
      1: min pLDDT
      2: max pLDDT
      3: std pLDDT
      4: fraction pLDDT > 70 (confident residues)
      5: fraction pLDDT > 90 (very confident)
      6: mean contacts per residue (8Å cutoff)
      7: contact density (contacts / n_residues²)
      8: radius of gyration (from Cα positions)
      9: sequence length / 1000 (normalised)
      10: fraction residues in contact with >5 others
      11: mean pairwise Cα distance

    Requires: pip install transformers torch
    """
    import torch

    print("  Loading ESMFold for structure prediction...")
    try:
        from transformers import EsmForProteinFolding, AutoTokenizer
    except ImportError:
        raise ImportError("ESMFold requires transformers>=4.30: pip install transformers")

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    # Force fp32 to avoid "LayerNormKernelImpl not implemented for Half"
    # which can happen when bitsandbytes or autocast interfere
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", torch_dtype=torch.float32,
    ).to(device).float()
    model.eval()

    n = len(sequences)
    features = np.zeros((n, 12), dtype=np.float32)

    print(f"  Predicting structures for {n} proteins with ESMFold...")
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=False):
        for i, seq in enumerate(sequences):
            seq = seq.upper().strip()[:512]  # ESMFold max practical length
            try:
                inputs = tokenizer([seq], return_tensors="pt",
                                   add_special_tokens=False).to(device)
                output = model(**inputs)

                # pLDDT: per-residue confidence (0-100 scale in output.plddt)
                plddt = output.plddt[0, :len(seq)].cpu().numpy().flatten()
                if plddt.max() <= 1.0:
                    plddt *= 100.0  # normalize to 0-100 if needed

                features[i, 0] = plddt.mean()
                features[i, 1] = plddt.min()
                features[i, 2] = plddt.max()
                features[i, 3] = plddt.std()
                features[i, 4] = (plddt > 70).mean()
                features[i, 5] = (plddt > 90).mean()

                # Extract Cα positions from predicted structure
                # output.positions: (1, n_res, 37, 3) — atom37 representation
                # Cα is atom index 1 in atom37
                positions = output.positions[0, :len(seq), 1, :].cpu().numpy()  # (L, 3)

                # Contact map (8Å cutoff on Cα-Cα distances)
                from scipy.spatial.distance import pdist, squareform
                dist_matrix = squareform(pdist(positions))
                n_res = len(seq)
                contacts = (dist_matrix < 8.0) & (np.abs(np.arange(n_res)[:, None] - np.arange(n_res)[None, :]) > 5)
                contacts_per_res = contacts.sum(axis=1).astype(np.float32)

                features[i, 6] = contacts_per_res.mean()
                features[i, 7] = contacts.sum() / max(n_res * n_res, 1)
                features[i, 8] = np.sqrt(((positions - positions.mean(axis=0)) ** 2).sum(axis=1).mean())
                features[i, 9] = n_res / 1000.0
                features[i, 10] = (contacts_per_res > 5).mean()
                features[i, 11] = dist_matrix.mean()

            except Exception as e:
                print(f"    WARNING: ESMFold failed on seq {i} (len={len(seq)}): {e}")
                features[i, 9] = len(seq) / 1000.0  # at least set length

            if (i + 1) % 20 == 0 or i == n - 1:
                print(f"    Progress: {i + 1}/{n}")

    print(f"  Generated ESMFold features: ({n}, 12)")
    return features


def _resolve_uniprot_ids(
    protein_ids: List[str],
) -> List[Optional[str]]:
    """Resolve protein identifiers to UniProt accession IDs.

    Handles two cases:
      - Already UniProt accessions (e.g. 'O00141', 'P04629'): returned as-is
      - Gene names (e.g. 'AAK1', 'ABL1(E255K)'): looked up via UniProt REST API

    Results are cached to disk so the API is only queried once per gene name.
    """
    import re
    import json
    from pathlib import Path

    cache_dir = Path(os.environ.get('KIRBY_CACHE_DIR',
                     str(Path.home() / '.cache' / 'kirby')))
    cache_file = cache_dir / 'uniprot_gene_map.json'

    # Load existing cache
    gene_map = {}
    if cache_file.exists():
        try:
            gene_map = json.loads(cache_file.read_text())
        except Exception:
            pass

    # UniProt accession pattern: [OPQ][0-9][A-Z0-9]{3}[0-9] or [A-NR-Z][0-9]...
    _uniprot_re = re.compile(r'^[A-Z][0-9][A-Z0-9]{3}[0-9]$')

    result = []
    genes_to_lookup = set()

    for pid in protein_ids:
        if pid is None:
            result.append(None)
            continue
        pid_clean = pid.strip()
        if _uniprot_re.match(pid_clean):
            result.append(pid_clean)
        else:
            # Extract base gene name: 'ABL1(E255K)-phosphorylated' → 'ABL1'
            base_gene = re.split(r'[(\-/]', pid_clean)[0].strip()
            if base_gene in gene_map:
                result.append(gene_map[base_gene])
            else:
                result.append(base_gene)  # placeholder, will be resolved
                genes_to_lookup.add(base_gene)

    # Batch lookup missing gene names via UniProt REST API
    if genes_to_lookup:
        import requests
        print(f"    Resolving {len(genes_to_lookup)} gene names to UniProt IDs...")
        for gene in sorted(genes_to_lookup):
            if gene in gene_map:
                continue
            try:
                url = (
                    f"https://rest.uniprot.org/uniprotkb/search?"
                    f"query=gene_exact:{gene}+AND+organism_id:9606&"
                    f"format=json&size=1&fields=accession"
                )
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('results'):
                        uid = data['results'][0]['primaryAccession']
                        gene_map[gene] = uid
                    else:
                        gene_map[gene] = None
                else:
                    gene_map[gene] = None
            except Exception:
                gene_map[gene] = None

        # Save updated cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(gene_map, indent=2))

        # Replace placeholders with resolved IDs
        _uniprot_re2 = re.compile(r'^[A-Z][0-9][A-Z0-9]{3}[0-9]$')
        for i, pid in enumerate(protein_ids):
            if pid is None:
                continue
            pid_clean = pid.strip()
            if not _uniprot_re2.match(pid_clean):
                base_gene = re.split(r'[(\-/]', pid_clean)[0].strip()
                result[i] = gene_map.get(base_gene)

        resolved = sum(1 for r in result if r is not None)
        print(f"    Resolved {resolved}/{len(result)} protein IDs to UniProt accessions")

    return result


def create_alphafold_db_features(
    sequences: List[str],
    uniprot_ids: Optional[List[str]] = None,
) -> np.ndarray:
    """Extract structural features from AlphaFold Database pre-computed structures.

    Downloads predicted structures from the AlphaFold Protein Structure Database
    (https://alphafold.ebi.ac.uk/) for proteins with known UniProt IDs. Extracts
    contact map + distance matrix summary features.

    If UniProt IDs are not provided, attempts to map sequences via UniProt ID
    mapping API. Falls back to sequence-based descriptors for unmapped proteins.

    Features (15D):
      0: mean pLDDT (from B-factor column)
      1: min pLDDT
      2: fraction pLDDT > 70
      3: fraction pLDDT > 90
      4: mean contacts per residue (8Å Cα)
      5: contact density
      6: radius of gyration
      7: mean pairwise Cα distance
      8: max pairwise Cα distance
      9: sequence length / 1000
      10: long-range contact fraction (|i-j| > 12)
      11: fraction residues with >5 contacts
      12: mean Cα distance for contacts (short range, |i-j| < 6)
      13: mean Cα distance for contacts (long range, |i-j| > 12)
      14: pLDDT std

    Requires: biopython, requests
    """
    import requests
    import tempfile
    from pathlib import Path

    print(f"  Fetching AlphaFold DB structures for {len(sequences)} proteins...")

    n = len(sequences)
    features = np.zeros((n, 15), dtype=np.float32)

    # Cache directory
    cache_dir = Path(os.environ.get('KIRBY_CACHE_DIR',
                     str(Path.home() / '.cache' / 'kirby'))) / 'alphafold_db'
    cache_dir.mkdir(parents=True, exist_ok=True)

    for i, seq in enumerate(sequences):
        seq = seq.upper().strip()
        uid = uniprot_ids[i] if uniprot_ids and i < len(uniprot_ids) else None

        # Try to get UniProt ID if not provided
        if uid is None:
            # Use sequence hash as a lookup key — skip API mapping for speed
            features[i, 9] = len(seq) / 1000.0
            continue

        pdb_path = cache_dir / f'AF-{uid}-F1-model_v4.pdb'

        # Download if not cached
        if not pdb_path.exists():
            url = f'https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v4.pdb'
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    pdb_path.write_bytes(resp.content)
                else:
                    features[i, 9] = len(seq) / 1000.0
                    continue
            except Exception:
                features[i, 9] = len(seq) / 1000.0
                continue

        # Parse PDB and extract features
        try:
            from Bio.PDB import PDBParser
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('af', str(pdb_path))
            model = structure[0]

            # Extract Cα atoms
            ca_atoms = []
            plddt_values = []
            for chain in model.get_chains():
                for residue in chain.get_residues():
                    if residue.get_id()[0] != ' ':
                        continue
                    if 'CA' in residue:
                        ca = residue['CA']
                        ca_atoms.append(ca.get_vector().get_array())
                        plddt_values.append(ca.get_bfactor())

            if not ca_atoms:
                features[i, 9] = len(seq) / 1000.0
                continue

            ca_positions = np.array(ca_atoms)
            plddt = np.array(plddt_values)
            n_res = len(ca_positions)

            # pLDDT features
            features[i, 0] = plddt.mean()
            features[i, 1] = plddt.min()
            features[i, 2] = (plddt > 70).mean()
            features[i, 3] = (plddt > 90).mean()

            # Contact map features
            from scipy.spatial.distance import pdist, squareform
            dist_matrix = squareform(pdist(ca_positions))
            seq_sep = np.abs(np.arange(n_res)[:, None] - np.arange(n_res)[None, :])
            contacts = (dist_matrix < 8.0) & (seq_sep > 5)
            contacts_per_res = contacts.sum(axis=1).astype(np.float32)

            features[i, 4] = contacts_per_res.mean()
            features[i, 5] = contacts.sum() / max(n_res * n_res, 1)
            features[i, 6] = np.sqrt(((ca_positions - ca_positions.mean(axis=0)) ** 2).sum(axis=1).mean())
            features[i, 7] = dist_matrix.mean()
            features[i, 8] = dist_matrix.max()
            features[i, 9] = n_res / 1000.0

            # Long-range contacts
            long_range = contacts & (seq_sep > 12)
            features[i, 10] = long_range.sum() / max(contacts.sum(), 1)
            features[i, 11] = (contacts_per_res > 5).mean()

            # Short vs long range average distances
            short_contacts = (dist_matrix < 8.0) & (seq_sep > 0) & (seq_sep <= 6)
            long_contacts = (dist_matrix < 8.0) & (seq_sep > 12)
            features[i, 12] = dist_matrix[short_contacts].mean() if short_contacts.any() else 0.0
            features[i, 13] = dist_matrix[long_contacts].mean() if long_contacts.any() else 0.0
            features[i, 14] = plddt.std()

        except Exception as e:
            print(f"    WARNING: AlphaFold DB feature extraction failed for {uid}: {e}")
            features[i, 9] = len(seq) / 1000.0

        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"    Progress: {i + 1}/{n}")

    mapped = (features[:, 0] != 0).sum()
    print(f"  Generated AlphaFold DB features: ({n}, 15), {mapped}/{n} mapped")
    return features


def create_mint_protein_embeddings(
    sequences: List[str],
    checkpoint_path: str = None,
    device: str = 'cpu',
) -> np.ndarray:
    """Extract MINT mean-pooled embeddings for protein sequences.

    MINT (Masked INTeraction modeling) is a protein language model built
    on top of ESM-2, pretrained for protein-protein interaction prediction.
    Produces 1280D embeddings per sequence.

    Args:
        sequences: List of protein amino acid sequences.
        checkpoint_path: Path to mint.ckpt. If None, searches
            $KIRBY_MODELS_DIR/mint/mint.ckpt, ~/kirby_models/mint/mint.ckpt.
        device: 'cpu' or 'cuda'.

    Returns:
        np.ndarray: (n_sequences, 1280) embeddings.

    Requires: git clone https://github.com/VarunUllanat/mint.git && pip install -e mint
              Download checkpoint from MINT GitHub releases.
    """
    import torch

    try:
        from mint.helpers.extract import load_config, MINTWrapper
    except ImportError:
        raise ImportError(
            "MINT requires the mint package: "
            "git clone https://github.com/VarunUllanat/mint.git && pip install -e mint\n"
            "Download checkpoint from https://github.com/VarunUllanat/mint "
            "and place at $KIRBY_MODELS_DIR/mint/mint.ckpt"
        )

    if checkpoint_path is None:
        search_paths = [
            os.path.join(os.environ.get('KIRBY_MODELS_DIR', ''), 'mint', 'mint.ckpt'),
            os.path.expanduser('~/kirby_models/mint/mint.ckpt'),
            'mint.ckpt',
            '../mint.ckpt',
        ]
        found = None
        for p in search_paths:
            if p and os.path.isfile(p):
                found = p
                break
        if found is None:
            raise FileNotFoundError(
                f"Could not find MINT checkpoint in: {search_paths}\n"
                "Download from https://github.com/VarunUllanat/mint "
                "and set KIRBY_MODELS_DIR or place at ~/kirby_models/mint/mint.ckpt"
            )
        checkpoint_path = found

    print(f"  Loading MINT from {checkpoint_path}...")
    # load_config requires path to MINT's config JSON
    mint_pkg_dir = os.path.dirname(os.path.dirname(
        __import__('mint').__file__))
    config_path = os.path.join(mint_pkg_dir, 'data', 'esm2_t33_650M_UR50D.json')
    if not os.path.isfile(config_path):
        # Fallback: search relative to KIRBy repo
        for candidate in [
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'MINT', 'data', 'esm2_t33_650M_UR50D.json'),
            os.path.join(os.environ.get('KIRBY_MODELS_DIR', ''), 'mint', 'esm2_t33_650M_UR50D.json'),
        ]:
            if candidate and os.path.isfile(candidate):
                config_path = candidate
                break
    config = load_config(config_path)
    model = MINTWrapper(config, checkpoint_path)
    model = model.to(device)
    model.eval()

    all_embeddings = []
    n = len(sequences)

    with torch.no_grad():
        for i, seq in enumerate(sequences):
            seq = seq.upper().strip()[:1022]  # ESM-2 backbone limit
            try:
                emb = model.get_embedding(seq)
                if isinstance(emb, torch.Tensor):
                    emb = emb.cpu().numpy()
                emb = np.atleast_1d(emb.flatten())
                # Expect 1280D from ESM-2 backbone
                if emb.shape[0] != 1280:
                    emb = emb[:1280] if emb.shape[0] > 1280 else np.pad(
                        emb, (0, 1280 - emb.shape[0]))
                all_embeddings.append(emb)
            except Exception as e:
                print(f"    WARNING: MINT failed on seq {i} (len={len(seq)}): {e}")
                all_embeddings.append(np.zeros(1280, dtype=np.float32))

            if (i + 1) % 50 == 0:
                print(f"    Progress: {i + 1}/{n}")

    result = np.vstack(all_embeddings).astype(np.float32)
    print(f"  Generated MINT protein embeddings: {result.shape}")
    return result


def create_xtrimopglm_protein_embeddings(
    sequences: List[str],
    device: str = 'cpu',
    batch_size: int = 2,
) -> np.ndarray:
    """Extract xTrimoPGLM-1B mean-pooled embeddings for protein sequences.

    xTrimoPGLM is a unified 100B-parameter protein language model from BioMap
    that integrates protein understanding and generation. We use the 1B MLM
    variant (proteinglm-1b-mlm) for efficient embedding extraction.

    Args:
        sequences: List of protein amino acid sequences.
        device: 'cpu' or 'cuda'.
        batch_size: Sequences per batch (default 2 — model is large).

    Returns:
        np.ndarray: (n_sequences, hidden_dim) embeddings.

    Requires: pip install transformers torch
    Note: Auto-downloads HuggingFace model on first use. Requires
          trust_remote_code=True for custom model architecture.
    """
    import torch
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError(
            "xTrimoPGLM requires transformers and torch: "
            "pip install transformers torch"
        )

    model_name = 'biomap-research/proteinglm-1b-mlm'
    print(f"  Loading xTrimoPGLM-1B ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    # Try bfloat16 first, fall back to float16, then float32
    dtype = torch.float32
    if device == 'cuda':
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=dtype,
    ).to(device).eval()

    # Determine hidden dim from model config
    hidden_dim = getattr(model.config, 'hidden_size', 2048)

    all_embeddings = []
    n = len(sequences)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_seqs = sequences[start:start + batch_size]
            batch_embs = []
            for j, seq in enumerate(batch_seqs):
                idx = start + j
                seq = seq.upper().strip()[:2048]
                try:
                    inputs = tokenizer(
                        seq, return_tensors='pt', padding=False,
                        truncation=True, max_length=2050,
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    # Mean pool excluding special tokens (BOS/EOS)
                    hidden = outputs.last_hidden_state
                    if hidden.shape[1] > 2:
                        emb = hidden[:, 1:-1, :].mean(dim=1)
                    else:
                        emb = hidden.mean(dim=1)
                    batch_embs.append(
                        emb.squeeze(0).float().cpu().numpy()
                    )
                except Exception as e:
                    print(f"    WARNING: xTrimoPGLM failed on seq {idx} "
                          f"(len={len(seq)}): {e}")
                    batch_embs.append(
                        np.zeros(hidden_dim, dtype=np.float32)
                    )
            all_embeddings.extend(batch_embs)

            end = min(start + batch_size, n)
            if end < n and end % 20 == 0:
                print(f"    xTrimoPGLM progress: {end}/{n}")

    result = np.vstack(all_embeddings).astype(np.float32)
    print(f"  Generated xTrimoPGLM protein embeddings: {result.shape}")
    return result


def create_protein_kmer(sequences: List[str], k: int = 3) -> np.ndarray:
    """Compute k-mer frequency features for protein sequences.

    Args:
        sequences: List of protein amino acid sequences.
        k: k-mer size. Default 3 → 8000D.

    Returns:
        np.ndarray: (n_sequences, 20^k) k-mer frequency matrix.
    """
    n_aas = len(STANDARD_AAS)
    dim = n_aas ** k
    aa_to_idx = {aa: i for i, aa in enumerate(STANDARD_AAS)}

    n = len(sequences)
    features = np.zeros((n, dim), dtype=np.float64)

    print(f"  Computing {k}-mer frequency for {n} proteins (dim={dim})...")

    for i, seq in enumerate(sequences):
        seq = seq.upper().strip()
        if len(seq) < k:
            continue

        count = 0
        for j in range(len(seq) - k + 1):
            kmer = seq[j:j + k]
            idx = 0
            valid = True
            for c, char in enumerate(kmer):
                if char not in aa_to_idx:
                    valid = False
                    break
                idx += aa_to_idx[char] * (n_aas ** (k - 1 - c))
            if valid:
                features[i, idx] += 1
                count += 1

        if count > 0:
            features[i] /= count

    print(f"  Generated protein {k}-mer features: ({n}, {dim})")
    return features
