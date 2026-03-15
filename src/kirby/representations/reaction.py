"""Reaction fingerprints for yield prediction.

DRFP (Differential Reaction Fingerprints):
  Symmetric difference of circular substructure n-grams between
  reactants and products. Binary, hashed, no pretraining.
  Probst, Schwaller & Reymond, Digital Discovery 2022.

Concat ECFP4:
  Concatenation of Morgan (ECFP4) fingerprints of individual
  reaction components. Standard non-reaction-aware baseline.

Concat MACCS:
  Concatenation of 167-bit MACCS keys per component.

Concat PDV:
  Concatenation of RDKit physicochemical descriptors per component.

Product ECFP4:
  Morgan fingerprint of the product molecule only.

Reactant ECFP4:
  OR-combined Morgan fingerprint of all reactants.

RXNFP:
  Learned reaction fingerprints from IBM's RXNBERT (256D).
  Schwaller et al., Nature Machine Intelligence 2021.

One-Hot Components:
  One-hot encoding of factorial reaction components (categorical baseline).
"""

import numpy as np
from typing import List, Dict


def create_drfp(rxn_smiles: List[str], n_bits: int = 2048) -> np.ndarray:
    """Generate DRFP reaction fingerprints.

    Args:
        rxn_smiles: Reaction SMILES in 'reactants>>products' format.
        n_bits: Fingerprint length (default 2048).

    Returns:
        np.ndarray of shape (n_reactions, n_bits), binary.

    Requires: pip install drfp
    """
    from drfp import DrfpEncoder

    fps = DrfpEncoder.encode(
        rxn_smiles,
        n_folded_length=n_bits,
        show_progress_bar=len(rxn_smiles) > 500,
    )
    return np.array(fps, dtype=np.float32)


def create_concat_ecfp4(components: List[Dict[str, str]],
                        n_bits: int = 1024) -> np.ndarray:
    """Concatenate ECFP4 fingerprints of individual reaction components.

    Each component gets its own Morgan fingerprint (radius=2), then all
    are concatenated. Standard non-reaction-aware baseline.

    Args:
        components: List of dicts with component SMILES as values.
            Keys must be consistent across all entries.
        n_bits: Bits per component fingerprint (default 1024).

    Returns:
        np.ndarray of shape (n_reactions, n_components * n_bits).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    if not components:
        raise ValueError("Empty components list")

    keys = sorted(components[0].keys())
    total_dim = len(keys) * n_bits
    X = np.zeros((len(components), total_dim), dtype=np.float32)

    for i, comp in enumerate(components):
        for j, key in enumerate(keys):
            smi = comp.get(key, '')
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
                arr = np.zeros(n_bits, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                X[i, j * n_bits:(j + 1) * n_bits] = arr

    return X


def create_concat_maccs(components: List[Dict[str, str]]) -> np.ndarray:
    """Concatenate MACCS keys (167-bit) of individual reaction components."""
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, DataStructs

    if not components:
        raise ValueError("Empty components list")

    keys = sorted(components[0].keys())
    total_dim = len(keys) * 167
    X = np.zeros((len(components), total_dim), dtype=np.float32)

    for i, comp in enumerate(components):
        for j, key in enumerate(keys):
            smi = comp.get(key, '')
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is not None:
                fp = MACCSkeys.GenMACCSKeys(mol)
                arr = np.zeros(167, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                X[i, j * 167:(j + 1) * 167] = arr

    return X


def create_concat_pdv(components: List[Dict[str, str]]) -> np.ndarray:
    """Concatenate RDKit physicochemical descriptors per reaction component."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

    if not components:
        raise ValueError("Empty components list")

    keys = sorted(components[0].keys())
    desc_names = [d[0] for d in Descriptors._descList]
    calc = MolecularDescriptorCalculator(desc_names)
    n_desc = len(desc_names)
    total_dim = len(keys) * n_desc
    X = np.zeros((len(components), total_dim), dtype=np.float32)

    for i, comp in enumerate(components):
        for j, key in enumerate(keys):
            smi = comp.get(key, '')
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is not None:
                vals = calc.CalcDescriptors(mol)
                arr = np.array(vals, dtype=np.float32)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                X[i, j * n_desc:(j + 1) * n_desc] = arr

    return X


def create_product_ecfp4(rxn_smiles: List[str], n_bits: int = 2048) -> np.ndarray:
    """Morgan fingerprint of the product molecule only."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    X = np.zeros((len(rxn_smiles), n_bits), dtype=np.float32)
    for i, rxn in enumerate(rxn_smiles):
        parts = rxn.split('>>')
        product_smi = parts[-1] if len(parts) >= 2 else parts[0]
        mol = Chem.MolFromSmiles(product_smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            arr = np.zeros(n_bits, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            X[i] = arr
    return X


def create_reactant_ecfp4(rxn_smiles: List[str], n_bits: int = 2048) -> np.ndarray:
    """OR-combined Morgan fingerprint of all reactants."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    X = np.zeros((len(rxn_smiles), n_bits), dtype=np.float32)
    for i, rxn in enumerate(rxn_smiles):
        parts = rxn.split('>>')
        reactant_smi = parts[0] if len(parts) >= 2 else ''
        combined = np.zeros(n_bits, dtype=np.float32)
        for smi in reactant_smi.split('.'):
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
                arr = np.zeros(n_bits, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                combined = np.maximum(combined, arr)  # OR combination
        X[i] = combined
    return X


def create_product_maccs(rxn_smiles: List[str]) -> np.ndarray:
    """MACCS keys (167-bit) of the product molecule only."""
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, DataStructs

    X = np.zeros((len(rxn_smiles), 167), dtype=np.float32)
    for i, rxn in enumerate(rxn_smiles):
        parts = rxn.split('>>')
        product_smi = parts[-1] if len(parts) >= 2 else parts[0]
        mol = Chem.MolFromSmiles(product_smi)
        if mol is not None:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros(167, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            X[i] = arr
    return X


def create_reactant_maccs(rxn_smiles: List[str]) -> np.ndarray:
    """OR-combined MACCS keys (167-bit) of all reactants."""
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, DataStructs

    X = np.zeros((len(rxn_smiles), 167), dtype=np.float32)
    for i, rxn in enumerate(rxn_smiles):
        parts = rxn.split('>>')
        reactant_smi = parts[0] if len(parts) >= 2 else ''
        combined = np.zeros(167, dtype=np.float32)
        for smi in reactant_smi.split('.'):
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is not None:
                fp = MACCSkeys.GenMACCSKeys(mol)
                arr = np.zeros(167, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                combined = np.maximum(combined, arr)
        X[i] = combined
    return X


def create_product_pdv(rxn_smiles: List[str]) -> np.ndarray:
    """RDKit physicochemical descriptors of the product molecule only."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

    desc_names = [d[0] for d in Descriptors._descList]
    calc = MolecularDescriptorCalculator(desc_names)
    n_desc = len(desc_names)
    X = np.zeros((len(rxn_smiles), n_desc), dtype=np.float32)
    for i, rxn in enumerate(rxn_smiles):
        parts = rxn.split('>>')
        product_smi = parts[-1] if len(parts) >= 2 else parts[0]
        mol = Chem.MolFromSmiles(product_smi)
        if mol is not None:
            vals = calc.CalcDescriptors(mol)
            arr = np.array(vals, dtype=np.float32)
            X[i] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def create_reactant_pdv(rxn_smiles: List[str]) -> np.ndarray:
    """Mean-pooled RDKit physicochemical descriptors of all reactants."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

    desc_names = [d[0] for d in Descriptors._descList]
    calc = MolecularDescriptorCalculator(desc_names)
    n_desc = len(desc_names)
    X = np.zeros((len(rxn_smiles), n_desc), dtype=np.float32)
    for i, rxn in enumerate(rxn_smiles):
        parts = rxn.split('>>')
        reactant_smi = parts[0] if len(parts) >= 2 else ''
        reactant_descs = []
        for smi in reactant_smi.split('.'):
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is not None:
                vals = calc.CalcDescriptors(mol)
                arr = np.array(vals, dtype=np.float32)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                reactant_descs.append(arr)
        if reactant_descs:
            X[i] = np.mean(reactant_descs, axis=0)
    return X


def create_rxnfp(rxn_smiles: List[str]) -> np.ndarray:
    """Generate learned reaction fingerprints via IBM's RXNBERT (256D).

    Transformer-based fingerprints trained on ~1M reactions.
    Schwaller et al., Nature Machine Intelligence 2021.

    Args:
        rxn_smiles: Reaction SMILES in 'reactants>>products' format.

    Returns:
        np.ndarray of shape (n_reactions, 256), float32.

    Requires: pip install rxnfp
    Note: Auto-downloads HuggingFace model on first use (~500MB).
    """
    from rxnfp.transformer_fingerprints import (
        RXNBERTFingerprintGenerator, get_default_model_and_tokenizer,
    )
    model, tokenizer = get_default_model_and_tokenizer()
    gen = RXNBERTFingerprintGenerator(model, tokenizer)
    from tqdm import tqdm
    fps = [gen.convert(rxn) for rxn in tqdm(rxn_smiles, desc='rxnfp')]
    return np.array(fps, dtype=np.float32)


def create_reactiont5(rxn_smiles: List[str], batch_size: int = 16) -> np.ndarray:
    """Generate ReactionT5 encoder embeddings for reaction SMILES (~768D).

    ReactionT5v2-forward is a T5-based model trained on reaction SMILES.
    We extract mean-pooled encoder hidden states as reaction representations.

    Sagawa et al., 2024.

    Args:
        rxn_smiles: Reaction SMILES in 'reactants>>products' format.
        batch_size: Sequences per batch (default 16).

    Returns:
        np.ndarray of shape (n_reactions, 768), float32.

    Requires: pip install transformers torch
    Note: Auto-downloads HuggingFace model on first use (~900MB).
    """
    try:
        import torch
        from transformers import AutoTokenizer, T5EncoderModel
    except ImportError:
        raise ImportError(
            "ReactionT5 requires transformers and torch: "
            "pip install transformers torch"
        )

    model_name = 'sagawa/ReactionT5v2-forward'
    print(f"  Loading ReactionT5 ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name).eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    all_embeddings = []
    n = len(rxn_smiles)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = rxn_smiles[start:start + batch_size]
            inputs = tokenizer(
                batch, return_tensors='pt', padding=True,
                truncation=True, max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # T5EncoderModel doesn't accept special_tokens_mask
            inputs.pop('special_tokens_mask', None)
            outputs = model(**inputs)
            # Mean pool over non-padding tokens
            attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
            sum_emb = (outputs.last_hidden_state * attention_mask).sum(dim=1)
            count = attention_mask.sum(dim=1).clamp(min=1)
            mean_emb = (sum_emb / count).cpu().numpy()
            all_embeddings.append(mean_emb)

            end = min(start + batch_size, n)
            if end < n and end % 100 == 0:
                print(f"    ReactionT5 progress: {end}/{n}")

    result = np.vstack(all_embeddings).astype(np.float32)
    print(f"  Generated ReactionT5 embeddings: {result.shape}")
    return result


def create_onehot_components(components: List[Dict[str, str]]) -> np.ndarray:
    """One-hot encode reaction components (categorical baseline).

    Encodes the factorial structure of HTE datasets (e.g. 15 aryl_halides ×
    4 ligands × 3 bases × 23 additives for Buchwald-Hartwig). No external deps.

    Args:
        components: List of dicts with component SMILES/names as values.
            Keys must be consistent across all entries.

    Returns:
        np.ndarray of shape (n_reactions, sum_of_vocab_sizes), float32.
    """
    if not components:
        raise ValueError("Empty components list")

    keys = sorted(components[0].keys())
    vocabs = {}
    for key in keys:
        unique_vals = sorted(set(comp.get(key, '') for comp in components))
        vocabs[key] = {val: i for i, val in enumerate(unique_vals)}

    total_dim = sum(len(v) for v in vocabs.values())
    X = np.zeros((len(components), total_dim), dtype=np.float32)
    offset = 0
    for key in keys:
        vocab = vocabs[key]
        for i, comp in enumerate(components):
            val = comp.get(key, '')
            if val in vocab:
                X[i, offset + vocab[val]] = 1.0
        offset += len(vocab)
    return X
