"""Reaction fingerprints for yield prediction.

DRFP (Differential Reaction Fingerprints):
  Symmetric difference of circular substructure n-grams between
  reactants and products. Binary, hashed, no pretraining.
  Probst, Schwaller & Reymond, Digital Discovery 2022.

Concat ECFP4:
  Concatenation of Morgan (ECFP4) fingerprints of individual
  reaction components. Standard non-reaction-aware baseline.
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
