"""
Antibody representations — simple LM embeddings + sequence encodings.

Each function takes heavy_seqs + light_seqs (lists of str) and returns
an (N, D) numpy array. LM functions mean-pool hidden states over residues.
For single-chain models, H and L are embedded separately and concatenated.
For paired models, a single forward pass handles both chains.

Usage:
    from kirby.representations.antibody import ANTIBODY_REPS, embed_esm2

    X = embed_esm2(heavy_seqs, light_seqs)  # (N, 2560)

    # Iterate all reps:
    for name, func in ANTIBODY_REPS.items():
        X = func(heavy_seqs, light_seqs)
"""

import numpy as np
from typing import List, Optional

# Amino acid alphabet (standard 20)
AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}

# BLOSUM62 matrix (standard 20 AA, symmetric)
_BLOSUM62 = {
    'A': {'A': 4,'C': 0,'D':-2,'E':-1,'F':-2,'G': 0,'H':-2,'I':-1,'K':-1,'L':-1,'M':-1,'N':-2,'P':-1,'Q':-1,'R':-1,'S': 1,'T': 0,'V': 0,'W':-3,'Y':-2},
    'C': {'A': 0,'C': 9,'D':-3,'E':-4,'F':-2,'G':-3,'H':-3,'I':-1,'K':-3,'L':-1,'M':-1,'N':-3,'P':-3,'Q':-3,'R':-3,'S':-1,'T':-1,'V':-1,'W':-2,'Y':-2},
    'D': {'A':-2,'C':-3,'D': 6,'E': 2,'F':-3,'G':-1,'H':-1,'I':-3,'K':-1,'L':-4,'M':-3,'N': 1,'P':-1,'Q': 0,'R':-2,'S': 0,'T':-1,'V':-3,'W':-4,'Y':-3},
    'E': {'A':-1,'C':-4,'D': 2,'E': 5,'F':-3,'G':-2,'H': 0,'I':-3,'K': 1,'L':-3,'M':-2,'N': 0,'P':-1,'Q': 2,'R': 0,'S': 0,'T':-1,'V':-2,'W':-3,'Y':-2},
    'F': {'A':-2,'C':-2,'D':-3,'E':-3,'F': 6,'G':-3,'H':-1,'I': 0,'K':-3,'L': 0,'M': 0,'N':-3,'P':-4,'Q':-3,'R':-3,'S':-2,'T':-2,'V':-1,'W': 1,'Y': 3},
    'G': {'A': 0,'C':-3,'D':-1,'E':-2,'F':-3,'G': 6,'H':-2,'I':-4,'K':-2,'L':-4,'M':-3,'N': 0,'P':-2,'Q':-2,'R':-2,'S': 0,'T':-2,'V':-3,'W':-2,'Y':-3},
    'H': {'A':-2,'C':-3,'D':-1,'E': 0,'F':-1,'G':-2,'H': 8,'I':-3,'K':-1,'L':-3,'M':-2,'N': 1,'P':-2,'Q': 0,'R': 0,'S':-1,'T':-2,'V':-3,'W':-2,'Y': 2},
    'I': {'A':-1,'C':-1,'D':-3,'E':-3,'F': 0,'G':-4,'H':-3,'I': 4,'K':-3,'L': 2,'M': 1,'N':-3,'P':-3,'Q':-3,'R':-3,'S':-2,'T':-1,'V': 3,'W':-3,'Y':-1},
    'K': {'A':-1,'C':-3,'D':-1,'E': 1,'F':-3,'G':-2,'H':-1,'I':-3,'K': 5,'L':-2,'M':-1,'N': 0,'P':-1,'Q': 1,'R': 2,'S': 0,'T':-1,'V':-2,'W':-3,'Y':-2},
    'L': {'A':-1,'C':-1,'D':-4,'E':-3,'F': 0,'G':-4,'H':-3,'I': 2,'K':-2,'L': 4,'M': 2,'N':-3,'P':-3,'Q':-2,'R':-2,'S':-2,'T':-1,'V': 1,'W':-2,'Y':-1},
    'M': {'A':-1,'C':-1,'D':-3,'E':-2,'F': 0,'G':-3,'H':-2,'I': 1,'K':-1,'L': 2,'M': 5,'N':-2,'P':-2,'Q': 0,'R':-1,'S':-1,'T':-1,'V': 1,'W':-1,'Y':-1},
    'N': {'A':-2,'C':-3,'D': 1,'E': 0,'F':-3,'G': 0,'H': 1,'I':-3,'K': 0,'L':-3,'M':-2,'N': 6,'P':-2,'Q': 0,'R': 0,'S': 1,'T': 0,'V':-3,'W':-4,'Y':-2},
    'P': {'A':-1,'C':-3,'D':-1,'E':-1,'F':-4,'G':-2,'H':-2,'I':-3,'K':-1,'L':-3,'M':-2,'N':-2,'P': 7,'Q':-1,'R':-2,'S':-1,'T':-1,'V':-2,'W':-4,'Y':-3},
    'Q': {'A':-1,'C':-3,'D': 0,'E': 2,'F':-3,'G':-2,'H': 0,'I':-3,'K': 1,'L':-2,'M': 0,'N': 0,'P':-1,'Q': 5,'R': 1,'S': 0,'T':-1,'V':-2,'W':-2,'Y':-1},
    'R': {'A':-1,'C':-3,'D':-2,'E': 0,'F':-3,'G':-2,'H': 0,'I':-3,'K': 2,'L':-2,'M':-1,'N': 0,'P':-2,'Q': 1,'R': 5,'S':-1,'T':-1,'V':-3,'W':-3,'Y':-2},
    'S': {'A': 1,'C':-1,'D': 0,'E': 0,'F':-2,'G': 0,'H':-1,'I':-2,'K': 0,'L':-2,'M':-1,'N': 1,'P':-1,'Q': 0,'R':-1,'S': 4,'T': 1,'V':-2,'W':-3,'Y':-2},
    'T': {'A': 0,'C':-1,'D':-1,'E':-1,'F':-2,'G':-2,'H':-2,'I':-1,'K':-1,'L':-1,'M':-1,'N': 0,'P':-1,'Q':-1,'R':-1,'S': 1,'T': 5,'V': 0,'W':-2,'Y':-2},
    'V': {'A': 0,'C':-1,'D':-3,'E':-2,'F':-1,'G':-3,'H':-3,'I': 3,'K':-2,'L': 1,'M': 1,'N':-3,'P':-2,'Q':-2,'R':-3,'S':-2,'T': 0,'V': 4,'W':-3,'Y':-1},
    'W': {'A':-3,'C':-2,'D':-4,'E':-3,'F': 1,'G':-2,'H':-2,'I':-3,'K':-3,'L':-2,'M':-1,'N':-4,'P':-4,'Q':-2,'R':-3,'S':-3,'T':-2,'V':-3,'W':11,'Y': 2},
    'Y': {'A':-2,'C':-2,'D':-3,'E':-2,'F': 3,'G':-3,'H': 2,'I':-1,'K':-2,'L':-1,'M':-1,'N':-2,'P':-3,'Q':-1,'R':-2,'S':-2,'T':-2,'V':-1,'W': 2,'Y': 7},
}


# ── Sequence encodings (no dependencies) ─────────────────────────────────────

def aac_encode(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """Amino acid composition: 20 frequencies per chain → (N, 40)."""
    n = len(heavy_seqs)
    X = np.zeros((n, 40), dtype=np.float32)
    for i in range(n):
        for j, seq in enumerate([heavy_seqs[i], light_seqs[i]]):
            L = len(seq)
            if L == 0:
                continue
            offset = j * 20
            for aa in seq:
                idx = AA_TO_IDX.get(aa)
                if idx is not None:
                    X[i, offset + idx] += 1.0
            X[i, offset:offset + 20] /= L
    return X


def onehot_encode(
    heavy_seqs: List[str],
    light_seqs: List[str],
    max_heavy: int = 150,
    max_light: int = 130,
) -> np.ndarray:
    """One-hot encoding: pad/truncate to fixed length → (N, (max_heavy+max_light)*20)."""
    n = len(heavy_seqs)
    n_aa = len(AA_ALPHABET)
    total_dim = (max_heavy + max_light) * n_aa
    X = np.zeros((n, total_dim), dtype=np.float32)

    for i in range(n):
        for j, (seq, maxlen) in enumerate([
            (heavy_seqs[i], max_heavy),
            (light_seqs[i], max_light),
        ]):
            offset = j * max_heavy * n_aa if j == 0 else max_heavy * n_aa
            for pos, aa in enumerate(seq[:maxlen]):
                idx = AA_TO_IDX.get(aa)
                if idx is not None:
                    X[i, offset + pos * n_aa + idx] = 1.0
    return X


def blosum_encode(
    heavy_seqs: List[str],
    light_seqs: List[str],
    max_heavy: int = 150,
    max_light: int = 130,
) -> np.ndarray:
    """BLOSUM62 encoding: each residue → 20D BLOSUM row → (N, (max_heavy+max_light)*20)."""
    n = len(heavy_seqs)
    n_aa = len(AA_ALPHABET)
    total_dim = (max_heavy + max_light) * n_aa
    X = np.zeros((n, total_dim), dtype=np.float32)

    for i in range(n):
        for j, (seq, maxlen) in enumerate([
            (heavy_seqs[i], max_heavy),
            (light_seqs[i], max_light),
        ]):
            offset = j * max_heavy * n_aa if j == 0 else max_heavy * n_aa
            for pos, aa in enumerate(seq[:maxlen]):
                if aa in _BLOSUM62:
                    row = _BLOSUM62[aa]
                    for k, target_aa in enumerate(AA_ALPHABET):
                        X[i, offset + pos * n_aa + k] = row.get(target_aa, 0)
    return X


# ── CDR extraction ────────────────────────────────────────────────────────────

# IMGT CDR boundaries (inclusive)
_IMGT_CDR_RANGES = {
    'CDR1': (27, 38),
    'CDR2': (56, 65),
    'CDR3': (105, 117),
}


def get_cdr_residue_indices(sequence: str) -> List[int]:
    """Get 0-based residue indices of CDR positions in a sequence using ANARCI/IMGT.

    Returns indices into the raw sequence (not tokenized). If ANARCI fails to
    number the sequence, returns all residue indices (falls back to full pooling).
    """
    try:
        from anarci import anarci as run_anarci
    except ImportError:
        return list(range(len(sequence)))

    results = run_anarci([('q', sequence)], scheme='imgt')
    numbering = results[0]

    if not numbering or not numbering[0]:
        return list(range(len(sequence)))

    positions = numbering[0][0][0]  # first domain of first query

    # Walk through numbered positions, tracking raw sequence index
    cdr_indices = []
    raw_idx = 0
    for (pos_num, _insertion), aa in positions:
        if aa == '-':
            continue
        if any(lo <= pos_num <= hi for lo, hi in _IMGT_CDR_RANGES.values()):
            cdr_indices.append(raw_idx)
        raw_idx += 1

    if not cdr_indices:
        return list(range(len(sequence)))

    return cdr_indices


# ── LM embeddings ────────────────────────────────────────────────────────────

def _mean_pool(outputs, attention_mask):
    """Mean-pool last hidden state over non-padding tokens."""
    import torch
    hidden = outputs.last_hidden_state  # (B, L, D)
    mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled.cpu().numpy()


def _cdr_pool(hidden_states, cdr_indices_list, bos_offset: int = 1):
    """Mean-pool hidden states over CDR residue positions only.

    Args:
        hidden_states: (B, L, D) tensor of hidden states
        cdr_indices_list: list of lists of 0-based residue indices per sequence
        bos_offset: number of special tokens before residues (1 for BOS/CLS)

    Returns:
        np.ndarray: (B, D) CDR-pooled embeddings
    """
    import torch
    B, L, D = hidden_states.shape
    pooled = []
    for i in range(B):
        # Offset indices to account for BOS/CLS token
        token_indices = [idx + bos_offset for idx in cdr_indices_list[i]
                        if idx + bos_offset < L]
        if not token_indices:
            # Fallback: mean over all non-special tokens
            pooled.append(hidden_states[i, bos_offset:, :].mean(dim=0))
        else:
            pooled.append(hidden_states[i, token_indices, :].mean(dim=0))
    return torch.stack(pooled).cpu().numpy()


def _embed_single_chain_model(
    heavy_seqs: List[str],
    light_seqs: List[str],
    model_name: str,
    batch_size: int = 8,
) -> np.ndarray:
    """Embed H and L separately with a single-chain model, concatenate."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)

    def _embed_seqs(seqs):
        all_embs = []
        for start in range(0, len(seqs), batch_size):
            batch = seqs[start:start + batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True,
                               truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            all_embs.append(_mean_pool(outputs, inputs['attention_mask']))
        return np.vstack(all_embs)

    h_embs = _embed_seqs(heavy_seqs)
    l_embs = _embed_seqs(light_seqs)
    return np.hstack([h_embs, l_embs]).astype(np.float32)


def embed_esm2(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """ESM-2 650M embeddings: H+L concatenated → (N, 2560)."""
    return _embed_single_chain_model(
        heavy_seqs, light_seqs,
        'facebook/esm2_t33_650M_UR50D',
    )


def embed_antiberty(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """AntiBERTy embeddings: H+L concatenated → (N, 1024)."""
    return _embed_single_chain_model(
        heavy_seqs, light_seqs,
        'Exscientia/AntiBERTy',
    )


def embed_prot_t5(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """ProtT5-XL-U50 embeddings: H+L concatenated → (N, 2048).

    ProtT5 expects space-separated amino acid sequences.
    Uses the encoder only (no decoder).
    """
    import torch
    from transformers import T5Tokenizer, T5EncoderModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'Rostlab/prot_t5_xl_uniref50'
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name).eval().to(device)
    batch_size = 4  # ProtT5 is large

    def _embed_seqs(seqs):
        # ProtT5 expects space-separated AA sequences
        spaced = [' '.join(list(s)) for s in seqs]
        all_embs = []
        for start in range(0, len(spaced), batch_size):
            batch = spaced[start:start + batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True,
                               truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            all_embs.append(_mean_pool(outputs, inputs['attention_mask']))
        return np.vstack(all_embs)

    h_embs = _embed_seqs(heavy_seqs)
    l_embs = _embed_seqs(light_seqs)
    return np.hstack([h_embs, l_embs]).astype(np.float32)


def embed_ablang2(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """AbLang2 paired embeddings → (N, 768).

    AbLang2 is a paired heavy+light antibody language model.
    Single forward pass for both chains.
    """
    import torch
    import ablang2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ablang2.pretrained(model_to_use='ablang2-paired', random_init=False,
                               device=device)
    model.freeze()

    # AbLang2 takes list of (heavy, light) tuples
    seqs = list(zip(heavy_seqs, light_seqs))
    batch_size = 8
    all_embs = []

    for start in range(0, len(seqs), batch_size):
        batch = seqs[start:start + batch_size]
        # ablang2.encode returns (B, D) with rescoding="mean"
        embs = model(batch, mode='seqcoding')
        if isinstance(embs, torch.Tensor):
            embs = embs.cpu().numpy()
        all_embs.append(embs)

    return np.vstack(all_embs).astype(np.float32)


def embed_igbert(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """IgBert paired embeddings → (N, 768).

    IgBert is a paired antibody language model based on RoFormer.
    Uses tokenizer text-pair handling for proper [SEP] insertion.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'Exscientia/IgBert'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    batch_size = 8
    all_embs = []

    for start in range(0, len(heavy_seqs), batch_size):
        h_batch = heavy_seqs[start:start + batch_size]
        l_batch = light_seqs[start:start + batch_size]
        # Use tokenizer's text-pair handling for proper SEP token insertion
        inputs = tokenizer(h_batch, l_batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        all_embs.append(_mean_pool(outputs, inputs['attention_mask']))

    return np.vstack(all_embs).astype(np.float32)


def embed_esmc(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """ESM-C 600M embeddings: H+L concatenated → (N, 2304).

    Drop-in ESM-2 replacement. Matches ESM-2 3B quality at 5x fewer params.
    Uses EvolutionaryScale's `esm` package (not HuggingFace transformers).

    Note: Non-commercial license (EvolutionaryScale Cambrian).

    Requires: pip install esm
    """
    import torch
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ESMC.from_pretrained("esmc_600m").to(device)
    model.eval()

    config = LogitsConfig(sequence=True, return_embeddings=True)
    batch_size = 8

    def _embed_seqs(seqs):
        all_embs = []
        for start in range(0, len(seqs), batch_size):
            batch = seqs[start:start + batch_size]
            batch_embs = []
            for seq in batch:
                protein = ESMProtein(sequence=seq)
                protein_tensor = model.encode(protein)
                with torch.no_grad():
                    output = model.logits(protein_tensor, config)
                # output.embeddings: (1, L, 1152) — mean-pool over residues
                emb = output.embeddings.mean(dim=1).squeeze(0).cpu().numpy()
                batch_embs.append(emb)
            all_embs.append(np.vstack(batch_embs))
        return np.vstack(all_embs)

    h_embs = _embed_seqs(heavy_seqs)
    l_embs = _embed_seqs(light_seqs)
    return np.hstack([h_embs, l_embs]).astype(np.float32)


# ── CDR-specific LM embeddings ───────────────────────────────────────────────


def _embed_single_chain_cdr(
    heavy_seqs: List[str],
    light_seqs: List[str],
    model_name: str,
    batch_size: int = 8,
) -> np.ndarray:
    """Embed H and L with CDR-only pooling, concatenate.

    Same as _embed_single_chain_model but mean-pools only over CDR residues
    (IMGT positions 27-38, 56-65, 105-117) instead of all residues.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)

    def _embed_seqs(seqs):
        # Pre-compute CDR indices for all sequences
        all_cdr_indices = [get_cdr_residue_indices(s) for s in seqs]

        all_embs = []
        for start in range(0, len(seqs), batch_size):
            batch = seqs[start:start + batch_size]
            batch_cdr = all_cdr_indices[start:start + batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True,
                               truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            # CDR-pool instead of mean-pool
            all_embs.append(_cdr_pool(outputs.last_hidden_state, batch_cdr))
        return np.vstack(all_embs)

    h_embs = _embed_seqs(heavy_seqs)
    l_embs = _embed_seqs(light_seqs)
    return np.hstack([h_embs, l_embs]).astype(np.float32)


def embed_esm2_cdr(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """ESM-2 CDR-only pooled: H+L concatenated → (N, 2560)."""
    return _embed_single_chain_cdr(
        heavy_seqs, light_seqs,
        'facebook/esm2_t33_650M_UR50D',
    )


def embed_antiberty_cdr(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """AntiBERTy CDR-only pooled: H+L concatenated → (N, 1024)."""
    return _embed_single_chain_cdr(
        heavy_seqs, light_seqs,
        'Exscientia/AntiBERTy',
    )


def embed_prot_t5_cdr(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """ProtT5 CDR-only pooled: H+L concatenated → (N, 2048).

    ProtT5 uses space-separated sequences and T5EncoderModel (no decoder).
    BOS offset = 0 for T5 (no CLS token, uses pad_token at start).
    """
    import torch
    from transformers import T5Tokenizer, T5EncoderModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'Rostlab/prot_t5_xl_uniref50'
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name).eval().to(device)
    batch_size = 4

    def _embed_seqs(seqs):
        all_cdr_indices = [get_cdr_residue_indices(s) for s in seqs]
        spaced = [' '.join(list(s)) for s in seqs]
        all_embs = []
        for start in range(0, len(spaced), batch_size):
            batch = spaced[start:start + batch_size]
            batch_cdr = all_cdr_indices[start:start + batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True,
                               truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            # T5 has no BOS token; residues start at position 0
            all_embs.append(_cdr_pool(outputs.last_hidden_state, batch_cdr,
                                      bos_offset=0))
        return np.vstack(all_embs)

    h_embs = _embed_seqs(heavy_seqs)
    l_embs = _embed_seqs(light_seqs)
    return np.hstack([h_embs, l_embs]).astype(np.float32)


def embed_esmc_cdr(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """ESM-C 600M CDR-only pooled: H+L concatenated → (N, 2304)."""
    import torch
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ESMC.from_pretrained("esmc_600m").to(device)
    model.eval()

    config = LogitsConfig(sequence=True, return_embeddings=True)

    def _embed_seqs(seqs):
        all_cdr_indices = [get_cdr_residue_indices(s) for s in seqs]
        all_embs = []
        for i, seq in enumerate(seqs):
            protein = ESMProtein(sequence=seq)
            protein_tensor = model.encode(protein)
            with torch.no_grad():
                output = model.logits(protein_tensor, config)
            # CDR-pool over embeddings (1, L, 1152)
            emb = _cdr_pool(output.embeddings, [all_cdr_indices[i]])[0]
            all_embs.append(emb)
        return np.vstack(all_embs)

    h_embs = _embed_seqs(heavy_seqs)
    l_embs = _embed_seqs(light_seqs)
    return np.hstack([h_embs, l_embs]).astype(np.float32)


# ── Intermediate layer extraction ────────────────────────────────────────────


def _embed_single_chain_midlayers(
    heavy_seqs: List[str],
    light_seqs: List[str],
    model_name: str,
    layer_start: int,
    layer_end: int,
    batch_size: int = 8,
) -> np.ndarray:
    """Embed H and L by averaging intermediate hidden layers, concatenate.

    Instead of using the last layer (which may be too task-specific for the
    pretraining objective), average layers [layer_start, layer_end] inclusive.
    For ESM-2 650M (33 layers): averaging layers 20-33 improved unsupervised
    clustering ARI from 0.268 → 0.354.

    Args:
        layer_start: First layer to include (0-indexed, layer 0 = embedding layer)
        layer_end: Last layer to include (inclusive)
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).eval().to(device)

    def _embed_seqs(seqs):
        all_embs = []
        for start in range(0, len(seqs), batch_size):
            batch = seqs[start:start + batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True,
                               truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            # outputs.hidden_states: tuple of (B, L, D), one per layer + embedding
            # Average selected layers
            selected = torch.stack(
                outputs.hidden_states[layer_start:layer_end + 1], dim=0
            ).mean(dim=0)  # (B, L, D)
            # Mean-pool over non-padding tokens
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            pooled = (selected * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_embs.append(pooled.cpu().numpy())
        return np.vstack(all_embs)

    h_embs = _embed_seqs(heavy_seqs)
    l_embs = _embed_seqs(light_seqs)
    return np.hstack([h_embs, l_embs]).astype(np.float32)


def embed_esm2_midlayers(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """ESM-2 650M mid-layer averaged (layers 20-33): H+L → (N, 2560)."""
    return _embed_single_chain_midlayers(
        heavy_seqs, light_seqs,
        'facebook/esm2_t33_650M_UR50D',
        layer_start=20, layer_end=33,
    )


# ── AMPLIFY 350M ──────────────────────────────────────────────────────────────


def embed_amplify(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """AMPLIFY 350M embeddings: H+L concatenated → (N, 1920).

    Drop-in ESM-2 alternative. 350M parameter protein language model
    from Mila/Chandar Lab.

    Requires: pip install transformers
    HF model: chandar-lab/AMPLIFY_350M
    """
    return _embed_single_chain_model(
        heavy_seqs, light_seqs,
        'chandar-lab/AMPLIFY_350M',
    )


def embed_amplify_cdr(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """AMPLIFY 350M CDR-only pooled: H+L concatenated → (N, 1920)."""
    return _embed_single_chain_cdr(
        heavy_seqs, light_seqs,
        'chandar-lab/AMPLIFY_350M',
    )


# ── Registry ──────────────────────────────────────────────────────────────────

ANTIBODY_REPS = {
    'esm2': embed_esm2,                # 2560D
    'esmc': embed_esmc,                # 2304D
    'ablang2': embed_ablang2,           # 768D
    'igbert': embed_igbert,             # 768D
    'antiberty': embed_antiberty,       # 1024D
    'prot_t5': embed_prot_t5,           # 2048D
    'onehot': onehot_encode,            # 5600D
    'blosum': blosum_encode,            # 5600D
    'aac': aac_encode,                  # 40D
    # CDR-specific (same dims, pooled over CDR residues only)
    'esm2_cdr': embed_esm2_cdr,        # 2560D
    'esmc_cdr': embed_esmc_cdr,        # 2304D
    'antiberty_cdr': embed_antiberty_cdr,  # 1024D
    'prot_t5_cdr': embed_prot_t5_cdr,  # 2048D
    # Intermediate layer averaged
    'esm2_midlayers': embed_esm2_midlayers,  # 2560D (layers 20-33 avg)
    # AMPLIFY 350M (Mila/Chandar Lab)
    'amplify': embed_amplify,                # 1920D (960 × 2)
    'amplify_cdr': embed_amplify_cdr,        # 1920D (CDR-pooled)
}

# LM reps (require GPU / torch / transformers)
ANTIBODY_LM_REPS = {
    'esm2', 'esmc', 'ablang2', 'igbert', 'antiberty', 'prot_t5',
    'esm2_cdr', 'esmc_cdr', 'antiberty_cdr', 'prot_t5_cdr',
    'esm2_midlayers', 'amplify', 'amplify_cdr',
}

# Fast reps (no dependencies beyond numpy)
ANTIBODY_FAST_REPS = {'onehot', 'blosum', 'aac'}
