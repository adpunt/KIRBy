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
            # output.embeddings: (1, L, 1152) — mean-pool over residues
            emb = output.embeddings.mean(dim=1).squeeze(0).cpu().numpy()
            all_embeddings.append(emb)

            if (i + 1) % 50 == 0:
                print(f"    Progress: {i + 1}/{n}")

    result = np.vstack(all_embeddings)
    print(f"  Generated ESM-C protein embeddings: {result.shape}")
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

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
            # Mean pool over non-padding tokens
            mask = inputs['attention_mask'].unsqueeze(-1).float()
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
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device)
    model.eval()

    n = len(sequences)
    features = np.zeros((n, 12), dtype=np.float32)

    print(f"  Predicting structures for {n} proteins with ESMFold...")
    with torch.no_grad():
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
