"""
Protein Representations — General-Purpose

Representations for arbitrary protein sequences (not antibody-specific).
Used primarily for drug-target affinity (DTA) prediction where target
proteins are kinases, GPCRs, etc.

Functions:
- create_esm2_protein_embeddings: ESM-2 mean-pooled embeddings (1280D)
- create_protein_aac: amino acid composition (20D)
- create_protein_descriptors: biophysical descriptors (~30D)
- create_protein_kmer: k-mer frequency (default k=3, 8000D)
"""

import numpy as np
from typing import List

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


def create_esm2_protein_embeddings(
    sequences: List[str],
    model_name: str = 'facebook/esm2_t33_650M_UR50D',
    device: str = 'cpu',
    batch_size: int = 4,
) -> np.ndarray:
    """Extract ESM-2 mean-pooled embeddings for protein sequences.

    Args:
        sequences: List of protein amino acid sequences.
        model_name: HuggingFace ESM-2 model name.
        device: 'cpu' or 'cuda'.
        batch_size: Sequences per batch (proteins can be long, use small batch).

    Returns:
        np.ndarray: (n_sequences, 1280) embeddings.
    """
    import torch
    try:
        from transformers import AutoTokenizer, EsmModel
    except ImportError:
        raise ImportError("ESM-2 requires transformers: pip install transformers")

    print(f"  Loading ESM-2 ({model_name.split('/')[-1]}) for protein embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device).eval()

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
            # Mean pool excluding BOS/EOS
            attention_mask = inputs['attention_mask'].clone()
            attention_mask[:, 0] = 0
            for j in range(len(batch)):
                last_tok = attention_mask[j].sum().item()
                if last_tok > 0:
                    attention_mask[j, int(last_tok)] = 0
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_emb = (outputs.last_hidden_state * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            mean_emb = (sum_emb / count).cpu().numpy()
            all_embeddings.append(mean_emb)

            end = min(start + batch_size, n)
            if end < n:
                print(f"    Progress: {end}/{n}")

    result = np.concatenate(all_embeddings, axis=0)
    print(f"  Generated ESM-2 protein embeddings: {result.shape}")
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
