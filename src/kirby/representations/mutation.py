"""Mutation-aware representations for ΔΔG and DMS prediction.

These reps use mutation metadata (position, wt/mut AA identity) to capture
the effect of specific amino acid substitutions — critical for ΔΔG prediction
where full-sequence embeddings fail because a single-point mutation barely
changes the mean-pooled representation.

Four representation types:
  mutation_features  (~55D)  Hand-crafted mutation descriptors
  esm2_mutsite       (1280D) ESM-2 per-residue embeddings at mutation site(s)
  esm2_wt_delta      (2560D) Difference between WT and mutant ESM-2 embeddings
  esm2_llr           (1-3D)  ESM-2 log-likelihood ratio (masked marginal)
"""

import numpy as np
from typing import Dict, List, Optional


# ── Amino acid properties ────────────────────────────────────────────────────

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

# Kyte-Doolittle hydrophobicity
HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
}

# Molecular weight (Da)
MW = {
    'A': 89, 'C': 121, 'D': 133, 'E': 147, 'F': 165,
    'G': 75, 'H': 155, 'I': 131, 'K': 146, 'L': 131,
    'M': 149, 'N': 132, 'P': 115, 'Q': 146, 'R': 174,
    'S': 105, 'T': 119, 'V': 117, 'W': 204, 'Y': 181,
}

# Charge at pH 7
CHARGE = {
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
    'G': 0, 'H': 0.1, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
    'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0,
}

# BLOSUM62 substitution matrix (upper triangle, symmetric)
_BLOSUM62_RAW = {
    ('A','A'): 4,('A','C'):0,('A','D'):-2,('A','E'):-1,('A','F'):-2,('A','G'):0,('A','H'):-2,('A','I'):-1,('A','K'):-1,('A','L'):-1,('A','M'):-1,('A','N'):-2,('A','P'):-1,('A','Q'):-1,('A','R'):-1,('A','S'):1,('A','T'):0,('A','V'):0,('A','W'):-3,('A','Y'):-2,
    ('C','C'):9,('C','D'):-3,('C','E'):-4,('C','F'):-2,('C','G'):-3,('C','H'):-3,('C','I'):-1,('C','K'):-3,('C','L'):-1,('C','M'):-1,('C','N'):-3,('C','P'):-3,('C','Q'):-3,('C','R'):-3,('C','S'):-1,('C','T'):-1,('C','V'):-1,('C','W'):-2,('C','Y'):-2,
    ('D','D'):6,('D','E'):2,('D','F'):-3,('D','G'):-1,('D','H'):-1,('D','I'):-3,('D','K'):-1,('D','L'):-4,('D','M'):-3,('D','N'):1,('D','P'):-1,('D','Q'):0,('D','R'):-2,('D','S'):0,('D','T'):-1,('D','V'):-3,('D','W'):-4,('D','Y'):-3,
    ('E','E'):5,('E','F'):-3,('E','G'):-2,('E','H'):0,('E','I'):-3,('E','K'):1,('E','L'):-3,('E','M'):-2,('E','N'):0,('E','P'):-1,('E','Q'):2,('E','R'):0,('E','S'):0,('E','T'):-1,('E','V'):-2,('E','W'):-3,('E','Y'):-2,
    ('F','F'):6,('F','G'):-3,('F','H'):-1,('F','I'):0,('F','K'):-3,('F','L'):0,('F','M'):0,('F','N'):-3,('F','P'):-4,('F','Q'):-3,('F','R'):-3,('F','S'):-2,('F','T'):-2,('F','V'):-1,('F','W'):1,('F','Y'):3,
    ('G','G'):6,('G','H'):-2,('G','I'):-4,('G','K'):-2,('G','L'):-4,('G','M'):-3,('G','N'):0,('G','P'):-2,('G','Q'):-2,('G','R'):-2,('G','S'):0,('G','T'):-2,('G','V'):-3,('G','W'):-2,('G','Y'):-3,
    ('H','H'):8,('H','I'):-3,('H','K'):-1,('H','L'):-3,('H','M'):-2,('H','N'):1,('H','P'):-2,('H','Q'):0,('H','R'):0,('H','S'):-1,('H','T'):-2,('H','V'):-3,('H','W'):-2,('H','Y'):2,
    ('I','I'):4,('I','K'):-3,('I','L'):2,('I','M'):1,('I','N'):-3,('I','P'):-3,('I','Q'):-3,('I','R'):-3,('I','S'):-2,('I','T'):-1,('I','V'):3,('I','W'):-3,('I','Y'):-1,
    ('K','K'):5,('K','L'):-2,('K','M'):-1,('K','N'):0,('K','P'):-1,('K','Q'):1,('K','R'):2,('K','S'):0,('K','T'):-1,('K','V'):-2,('K','W'):-3,('K','Y'):-2,
    ('L','L'):4,('L','M'):2,('L','N'):-3,('L','P'):-3,('L','Q'):-2,('L','R'):-2,('L','S'):-2,('L','T'):-1,('L','V'):1,('L','W'):-2,('L','Y'):-1,
    ('M','M'):5,('M','N'):-2,('M','P'):-2,('M','Q'):0,('M','R'):-1,('M','S'):-1,('M','T'):-1,('M','V'):1,('M','W'):-1,('M','Y'):-1,
    ('N','N'):6,('N','P'):-2,('N','Q'):0,('N','R'):0,('N','S'):1,('N','T'):0,('N','V'):-3,('N','W'):-4,('N','Y'):-2,
    ('P','P'):7,('P','Q'):-1,('P','R'):-2,('P','S'):-1,('P','T'):-1,('P','V'):-2,('P','W'):-4,('P','Y'):-3,
    ('Q','Q'):5,('Q','R'):1,('Q','S'):0,('Q','T'):-1,('Q','V'):-2,('Q','W'):-2,('Q','Y'):-1,
    ('R','R'):5,('R','S'):-1,('R','T'):-1,('R','V'):-3,('R','W'):-3,('R','Y'):-2,
    ('S','S'):4,('S','T'):1,('S','V'):-2,('S','W'):-3,('S','Y'):-2,
    ('T','T'):5,('T','V'):0,('T','W'):-2,('T','Y'):-2,
    ('V','V'):4,('V','W'):-3,('V','Y'):-1,
    ('W','W'):11,('W','Y'):2,
    ('Y','Y'):7,
}
BLOSUM62 = {}
for (a, b), v in _BLOSUM62_RAW.items():
    BLOSUM62[(a, b)] = v
    BLOSUM62[(b, a)] = v


def _get_blosum(wt, mut):
    return BLOSUM62.get((wt, mut), -4)


def _aa_onehot(aa):
    v = np.zeros(20, dtype=np.float32)
    idx = AA_TO_IDX.get(aa, -1)
    if idx >= 0:
        v[idx] = 1.0
    return v


def _prop_delta(wt, mut, prop_dict, default=0.0):
    return prop_dict.get(mut, default) - prop_dict.get(wt, default)


# ── Mutation features (hand-crafted) ─────────────────────────────────────────

def create_mutation_features(
    heavy_seqs: List[str],
    light_seqs: Optional[List[str]],
    mutations: List[List[Dict]],
) -> np.ndarray:
    """Hand-crafted mutation descriptors (~55D).

    For each sample, averages over all mutations (handles multi-point):
      - Wildtype AA one-hot (20D)
      - Mutant AA one-hot (20D)
      - BLOSUM62 substitution score (1D)
      - Hydrophobicity delta (1D)
      - Molecular weight delta (1D)
      - Charge delta (1D)
      - Normalized mutation position (1D, relative to sequence length)
      - Is in heavy chain (1D)
      - Number of mutations (1D)
      - Self-substitution flag (1D) — WT==MUT (silent mutation)
      Total: ~48D (some features aggregated over multi-point mutations)

    Args:
        heavy_seqs: Mutant heavy chain sequences
        light_seqs: Mutant light chain sequences (or None)
        mutations: Per-sample list of mutation dicts with keys:
            pos (int), wt_aa (str), mut_aa (str), chain ('heavy'/'light')

    Returns:
        (N, ~48D) float32 array
    """
    features = []

    for i, mut_list in enumerate(mutations):
        if not mut_list:
            features.append(np.zeros(48, dtype=np.float32))
            continue

        # Aggregate over mutations
        wt_onehots = []
        mut_onehots = []
        blosum_scores = []
        hydro_deltas = []
        mw_deltas = []
        charge_deltas = []
        norm_positions = []
        is_heavy = []
        is_self = []

        for m in mut_list:
            wt, mt = m['wt_aa'], m['mut_aa']
            chain = m.get('chain', 'heavy')
            pos = m.get('pos', 0)

            seq = heavy_seqs[i] if chain == 'heavy' else (
                light_seqs[i] if light_seqs else '')
            seq_len = max(len(seq), 1)

            wt_onehots.append(_aa_onehot(wt))
            mut_onehots.append(_aa_onehot(mt))
            blosum_scores.append(_get_blosum(wt, mt))
            hydro_deltas.append(_prop_delta(wt, mt, HYDROPHOBICITY))
            mw_deltas.append(_prop_delta(wt, mt, MW))
            charge_deltas.append(_prop_delta(wt, mt, CHARGE))
            norm_positions.append(pos / seq_len)
            is_heavy.append(1.0 if chain == 'heavy' else 0.0)
            is_self.append(1.0 if wt == mt else 0.0)

        feat = np.concatenate([
            np.mean(wt_onehots, axis=0),           # 20D
            np.mean(mut_onehots, axis=0),           # 20D
            [np.mean(blosum_scores)],               # 1D
            [np.mean(hydro_deltas)],                # 1D
            [np.mean(mw_deltas)],                   # 1D
            [np.mean(charge_deltas)],               # 1D
            [np.mean(norm_positions)],              # 1D
            [np.mean(is_heavy)],                    # 1D
            [len(mut_list)],                        # 1D
            [np.mean(is_self)],                     # 1D
        ]).astype(np.float32)
        features.append(feat)

    return np.vstack(features)


# ── ESM-2 mutation-site embeddings ───────────────────────────────────────────

def _extract_per_residue_esm2(
    sequences: List[str],
    batch_size: int = 4,
) -> List[np.ndarray]:
    """Extract per-residue ESM-2 hidden states. Returns list of (L_i, 1280)."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'facebook/esm2_t33_650M_UR50D'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)

    all_embs = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        for j in range(len(batch)):
            seq_len = len(batch[j])
            # Skip BOS token (index 0), take residue tokens 1..seq_len
            emb = hidden[j, 1:seq_len + 1, :].cpu().numpy().astype(np.float32)
            all_embs.append(emb)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_embs


def create_esm2_mutsite(
    heavy_seqs: List[str],
    light_seqs: Optional[List[str]],
    mutations: List[List[Dict]],
    window: int = 3,
) -> np.ndarray:
    """ESM-2 embeddings at mutation site(s) with local window.

    For each sample, extracts per-residue ESM-2 on the mutant sequence,
    then pools residues within ±window of each mutation position.

    Args:
        window: Number of residues on each side of mutation to include.

    Returns:
        (N, 1280) float32 array (or 2560D if paired H+L mutations)
    """
    # Collect unique sequences to avoid re-embedding
    all_seqs = []
    seq_to_idx = {}
    sample_chain_map = []  # (heavy_seq_idx, light_seq_idx_or_None)

    for i in range(len(heavy_seqs)):
        h = heavy_seqs[i]
        if h not in seq_to_idx:
            seq_to_idx[h] = len(all_seqs)
            all_seqs.append(h)
        h_idx = seq_to_idx[h]

        l_idx = None
        if light_seqs and light_seqs[i]:
            l = light_seqs[i]
            if l not in seq_to_idx:
                seq_to_idx[l] = len(all_seqs)
                all_seqs.append(l)
            l_idx = seq_to_idx[l]

        sample_chain_map.append((h_idx, l_idx))

    print(f"    Extracting per-residue ESM-2 for {len(all_seqs)} unique seqs...")
    per_residue = _extract_per_residue_esm2(all_seqs)

    features = []
    dim = 1280

    for i, mut_list in enumerate(mutations):
        h_idx, l_idx = sample_chain_map[i]

        if not mut_list:
            features.append(np.zeros(dim, dtype=np.float32))
            continue

        site_embs = []
        for m in mut_list:
            chain = m.get('chain', 'heavy')
            pos = m.get('pos', 0)

            if chain == 'heavy':
                emb_matrix = per_residue[h_idx]
            elif l_idx is not None:
                emb_matrix = per_residue[l_idx]
            else:
                emb_matrix = per_residue[h_idx]

            seq_len = emb_matrix.shape[0]
            start = max(0, pos - window)
            end = min(seq_len, pos + window + 1)
            site_embs.append(emb_matrix[start:end].mean(axis=0))

        features.append(np.mean(site_embs, axis=0))

    return np.vstack(features)


# ── ESM-2 wildtype-mutant delta ──────────────────────────────────────────────

def _reconstruct_wildtype(seq: str, mutations_for_chain: List[Dict]) -> str:
    """Reverse-apply mutations to get the wildtype sequence."""
    seq_list = list(seq)
    for m in mutations_for_chain:
        pos = m.get('pos', 0)
        if pos < len(seq_list):
            seq_list[pos] = m['wt_aa']
    return ''.join(seq_list)


def create_esm2_wt_delta(
    heavy_seqs: List[str],
    light_seqs: Optional[List[str]],
    mutations: List[List[Dict]],
) -> np.ndarray:
    """Delta between wildtype and mutant ESM-2 mean-pool embeddings.

    Reconstructs WT from mutant + mutation info, embeds both, returns difference.
    Captures the embedding-space shift caused by the mutation.

    Returns:
        (N, 2560) float32 array (H delta concat L delta, or H delta doubled)
    """
    from kirby.representations.antibody import embed_esm2

    # Reconstruct wildtype sequences
    wt_heavy = []
    wt_light = [] if light_seqs else None

    for i in range(len(heavy_seqs)):
        h_muts = [m for m in mutations[i] if m.get('chain') == 'heavy']
        wt_h = _reconstruct_wildtype(heavy_seqs[i], h_muts)
        wt_heavy.append(wt_h)

        if light_seqs and light_seqs[i]:
            l_muts = [m for m in mutations[i] if m.get('chain') == 'light']
            wt_l = _reconstruct_wildtype(light_seqs[i], l_muts)
            wt_light.append(wt_l)
        elif wt_light is not None:
            wt_light.append('')

    light_for_embed = light_seqs if light_seqs else [''] * len(heavy_seqs)
    wt_light_for_embed = wt_light if wt_light else [''] * len(heavy_seqs)

    print("    Embedding mutant sequences...")
    X_mut = embed_esm2(heavy_seqs, light_for_embed)
    print("    Embedding wildtype sequences...")
    X_wt = embed_esm2(wt_heavy, wt_light_for_embed)

    return (X_mut - X_wt).astype(np.float32)


# ── ESM-2 log-likelihood ratio ───────────────────────────────────────────────

def create_esm2_llr(
    heavy_seqs: List[str],
    light_seqs: Optional[List[str]],
    mutations: List[List[Dict]],
) -> np.ndarray:
    """ESM-2 pseudo-log-likelihood ratio for each mutation.

    For each mutation position, masks the residue in the WILDTYPE sequence,
    runs ESM-2 forward pass, and computes:
        LLR = log P(mutant_aa | context) - log P(wildtype_aa | context)

    This is the masked marginal scoring approach from Meier et al. (2021).
    Negative LLR → mutation is disfavored by the protein language model.

    For multi-point mutations, returns sum, mean, and min LLR.

    Returns:
        (N, 3) float32 array: [sum_llr, mean_llr, min_llr]
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'facebook/esm2_t33_650M_UR50D'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).eval().to(device)

    # ESM-2 uses per-residue tokenization: each AA is one token
    # Token IDs: vocab maps single AAs to token IDs
    aa_to_token = {}
    for aa in AA_LIST:
        toks = tokenizer.encode(aa, add_special_tokens=False)
        if toks:
            aa_to_token[aa] = toks[0]

    mask_token_id = tokenizer.mask_token_id

    features = []

    for i in range(len(heavy_seqs)):
        mut_list = mutations[i] if i < len(mutations) else []
        if not mut_list:
            features.append(np.zeros(3, dtype=np.float32))
            continue

        llrs = []
        for m in mut_list:
            chain = m.get('chain', 'heavy')
            pos = m.get('pos', 0)
            wt_aa = m['wt_aa']
            mut_aa = m['mut_aa']

            # Get the mutant sequence and reconstruct WT at this position
            if chain == 'heavy':
                seq = heavy_seqs[i]
            elif light_seqs and light_seqs[i]:
                seq = light_seqs[i]
            else:
                seq = heavy_seqs[i]

            # Reconstruct wildtype at this position
            wt_seq = list(seq)
            if pos < len(wt_seq):
                wt_seq[pos] = wt_aa
            wt_seq = ''.join(wt_seq)

            # Tokenize WT sequence, mask the mutation position
            tokens = tokenizer.encode(wt_seq, add_special_tokens=True,
                                      return_tensors='pt').to(device)
            # Position in token sequence: +1 for BOS token
            mask_pos = pos + 1
            if mask_pos >= tokens.shape[1] - 1:
                llrs.append(0.0)
                continue

            tokens[0, mask_pos] = mask_token_id

            with torch.no_grad():
                logits = model(tokens).logits[0, mask_pos]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            wt_tok = aa_to_token.get(wt_aa)
            mut_tok = aa_to_token.get(mut_aa)
            if wt_tok is not None and mut_tok is not None:
                llr = (log_probs[mut_tok] - log_probs[wt_tok]).item()
            else:
                llr = 0.0
            llrs.append(llr)

        if llrs:
            features.append(np.array([
                sum(llrs),
                np.mean(llrs),
                min(llrs),
            ], dtype=np.float32))
        else:
            features.append(np.zeros(3, dtype=np.float32))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.vstack(features)


# ── ESM-C log-likelihood ratio ──────────────────────────────────────────────

def create_esmc_llr(
    heavy_seqs: List[str],
    light_seqs: Optional[List[str]],
    mutations: List[List[Dict]],
) -> np.ndarray:
    """ESM-C 600M masked marginal log-likelihood ratio for each mutation.

    Same approach as ESM-2 LLR but uses EvolutionaryScale's ESM-C model.
    ESM-C matches ESM-2 3B quality at 5x fewer params — a different model
    gives a different LLR signal that the hybrid search can leverage.

    Returns:
        (N, 3) float32 array: [sum_llr, mean_llr, min_llr]
    """
    import torch
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ESMC.from_pretrained("esmc_600m")
    if device == 'cpu' or (device == 'cuda' and not torch.cuda.is_bf16_supported()):
        model = model.float()
    model = model.to(device)
    model.eval()

    # Build vocab mapping: single AA → token ID
    # ESM-C encode() returns ESMProteinTensor with .sequence shape (L+2,) or (1, L+2)
    aa_to_token = {}
    for aa in AA_LIST:
        protein = ESMProtein(sequence=aa)
        tensor = model.encode(protein)
        seq_t = tensor.sequence
        # Handle both 1D (L,) and 2D (1, L) sequence tensors
        aa_to_token[aa] = int(seq_t[1] if seq_t.ndim == 1 else seq_t[0, 1])

    from esm.sdk.api import LogitsConfig
    config = LogitsConfig(sequence=True, return_embeddings=False)

    features = []

    for i in range(len(heavy_seqs)):
        mut_list = mutations[i] if i < len(mutations) else []
        if not mut_list:
            features.append(np.zeros(3, dtype=np.float32))
            continue

        llrs = []
        for m in mut_list:
            chain = m.get('chain', 'heavy')
            pos = m.get('pos', 0)
            wt_aa = m['wt_aa']
            mut_aa = m['mut_aa']

            if chain == 'heavy':
                seq = heavy_seqs[i]
            elif light_seqs and light_seqs[i]:
                seq = light_seqs[i]
            else:
                seq = heavy_seqs[i]

            # Reconstruct wildtype at this position
            wt_seq = list(seq)
            if pos < len(wt_seq):
                wt_seq[pos] = wt_aa
            wt_seq = ''.join(wt_seq)

            # Encode and mask
            protein = ESMProtein(sequence=wt_seq)
            tensor = model.encode(protein)
            seq_t = tensor.sequence
            mask_pos = pos + 1  # +1 for BOS
            seq_len = seq_t.shape[-1]  # works for both 1D and 2D
            if mask_pos >= seq_len - 1:
                llrs.append(0.0)
                continue

            # Mask the target position (ESM family uses token ID 32 for mask)
            if seq_t.ndim == 1:
                seq_t[mask_pos] = 32
            else:
                seq_t[0, mask_pos] = 32

            with torch.no_grad():
                output = model.logits(tensor, config)
                logits_t = output.logits
                # Handle both 2D (L, V) and 3D (1, L, V)
                if logits_t.ndim == 3:
                    logits = logits_t[0, mask_pos]
                else:
                    logits = logits_t[mask_pos]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            wt_tok = aa_to_token.get(wt_aa)
            mut_tok = aa_to_token.get(mut_aa)
            if wt_tok is not None and mut_tok is not None:
                llr = (log_probs[mut_tok] - log_probs[wt_tok]).item()
            else:
                llr = 0.0
            llrs.append(llr)

        if llrs:
            features.append(np.array([
                sum(llrs), np.mean(llrs), min(llrs),
            ], dtype=np.float32))
        else:
            features.append(np.zeros(3, dtype=np.float32))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.vstack(features)


# ── AntiBERTy log-likelihood ratio ─────────────────────────────────────────

def create_antiberty_llr(
    heavy_seqs: List[str],
    light_seqs: Optional[List[str]],
    mutations: List[List[Dict]],
) -> np.ndarray:
    """AntiBERTy masked marginal log-likelihood ratio for each mutation.

    AntiBERTy is a BERT-style masked LM trained specifically on antibody
    sequences. Its LLR captures antibody-specific evolutionary plausibility,
    complementary to the general-purpose ESM-2 LLR.

    Uses the `antiberty` pip package (jeffreyruffolo/AntiBERTy).
    Accesses the underlying BertForMaskedLM model and tokenizer from the runner.

    Returns:
        (N, 3) float32 array: [sum_llr, mean_llr, min_llr]
    """
    import torch
    from antiberty import AntiBERTyRunner

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    runner = AntiBERTyRunner()
    # AntiBERTyRunner exposes .model (BertForMaskedLM) and .tokenizer
    model = runner.model.to(device).eval()
    tokenizer = runner.tokenizer

    # Build AA → token ID mapping
    aa_to_token = {}
    for aa in AA_LIST:
        toks = tokenizer.encode(aa, add_special_tokens=False)
        if toks:
            aa_to_token[aa] = toks[0]

    mask_token_id = tokenizer.mask_token_id

    features = []

    for i in range(len(heavy_seqs)):
        mut_list = mutations[i] if i < len(mutations) else []
        if not mut_list:
            features.append(np.zeros(3, dtype=np.float32))
            continue

        llrs = []
        for m in mut_list:
            chain = m.get('chain', 'heavy')
            pos = m.get('pos', 0)
            wt_aa = m['wt_aa']
            mut_aa = m['mut_aa']

            if chain == 'heavy':
                seq = heavy_seqs[i]
            elif light_seqs and light_seqs[i]:
                seq = light_seqs[i]
            else:
                seq = heavy_seqs[i]

            # Reconstruct wildtype at this position
            wt_seq = list(seq)
            if pos < len(wt_seq):
                wt_seq[pos] = wt_aa
            wt_seq = ''.join(wt_seq)

            # Tokenize WT sequence, mask the mutation position
            tokens = tokenizer.encode(wt_seq, add_special_tokens=True,
                                      return_tensors='pt').to(device)
            # +1 for CLS/BOS token
            mask_pos = pos + 1
            if mask_pos >= tokens.shape[1] - 1:
                llrs.append(0.0)
                continue

            tokens[0, mask_pos] = mask_token_id

            with torch.no_grad():
                logits = model(tokens).prediction_logits[0, mask_pos]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            wt_tok = aa_to_token.get(wt_aa)
            mut_tok = aa_to_token.get(mut_aa)
            if wt_tok is not None and mut_tok is not None:
                llr = (log_probs[mut_tok] - log_probs[wt_tok]).item()
            else:
                llr = 0.0
            llrs.append(llr)

        if llrs:
            features.append(np.array([
                sum(llrs), np.mean(llrs), min(llrs),
            ], dtype=np.float32))
        else:
            features.append(np.zeros(3, dtype=np.float32))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.vstack(features)


# ── Registry ─────────────────────────────────────────────────────────────────

MUTATION_REPS = {
    'mutation_features': create_mutation_features,    # ~48D
    'esm2_mutsite': create_esm2_mutsite,              # 1280D
    'esm2_wt_delta': create_esm2_wt_delta,            # 2560D
    'esm2_llr': create_esm2_llr,                      # 3D
    'esmc_llr': create_esmc_llr,                      # 3D
    'antiberty_llr': create_antiberty_llr,            # 3D
}
