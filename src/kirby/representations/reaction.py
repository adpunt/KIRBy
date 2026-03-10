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
