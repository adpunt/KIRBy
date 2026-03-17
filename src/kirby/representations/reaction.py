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


# =========================================================================
# DIFFERENCE FINGERPRINTS — what changed in the reaction
# =========================================================================

def create_diff_ecfp4(rxn_smiles: List[str], n_bits: int = 2048) -> np.ndarray:
    """Product ECFP4 minus reactant ECFP4 — captures structural change.

    Positive bits = substructures gained in product.
    Negative bits = substructures lost from reactants.
    Unlike DRFP (symmetric difference), this preserves directionality.

    Returns:
        np.ndarray of shape (n_reactions, n_bits), float32 in {-1, 0, 1}.
    """
    return create_product_ecfp4(rxn_smiles, n_bits) - create_reactant_ecfp4(rxn_smiles, n_bits)


def create_diff_maccs(rxn_smiles: List[str]) -> np.ndarray:
    """Product MACCS minus reactant MACCS — captures structural change.

    Returns:
        np.ndarray of shape (n_reactions, 167), float32 in {-1, 0, 1}.
    """
    return create_product_maccs(rxn_smiles) - create_reactant_maccs(rxn_smiles)


# =========================================================================
# COMPONENT PHYSICOCHEMICAL DESCRIPTORS
# =========================================================================

_COMPONENT_PROPS = [
    'MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
    'NumRotatableBonds', 'NumAromaticRings', 'NumHeteroatoms',
    'FractionCSP3', 'LabuteASA', 'BalabanJ', 'HallKierAlpha',
    'NumValenceElectrons', 'NumRadicalElectrons',
]


def _compute_mol_props(smi: str) -> np.ndarray:
    """Compute focused physicochemical properties for one molecule."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    n = len(_COMPONENT_PROPS)
    props = np.zeros(n, dtype=np.float32)

    mol = Chem.MolFromSmiles(smi) if smi and smi.strip() else None
    if mol is None:
        return props

    _fn_map = {
        'MolWt': Descriptors.MolWt,
        'MolLogP': Descriptors.MolLogP,
        'TPSA': Descriptors.TPSA,
        'NumHDonors': Descriptors.NumHDonors,
        'NumHAcceptors': Descriptors.NumHAcceptors,
        'NumRotatableBonds': Descriptors.NumRotatableBonds,
        'NumAromaticRings': Descriptors.NumAromaticRings,
        'NumHeteroatoms': Descriptors.NumHeteroatoms,
        'FractionCSP3': Descriptors.FractionCSP3,
        'LabuteASA': Descriptors.LabuteASA,
        'BalabanJ': Descriptors.BalabanJ,
        'HallKierAlpha': Descriptors.HallKierAlpha,
        'NumValenceElectrons': Descriptors.NumValenceElectrons,
        'NumRadicalElectrons': Descriptors.NumRadicalElectrons,
    }

    for i, name in enumerate(_COMPONENT_PROPS):
        try:
            props[i] = float(_fn_map[name](mol))
        except Exception:
            pass

    return np.nan_to_num(props, nan=0.0, posinf=0.0, neginf=0.0)


def create_component_descriptors(components: List[Dict[str, str]]) -> np.ndarray:
    """Per-component physicochemical descriptors concatenated.

    For each component role (aryl_halide, ligand, base, additive, etc.),
    computes 14 RDKit properties (MW, LogP, TPSA, HBD, HBA, rotatable bonds,
    aromatic rings, heteroatoms, Fsp3, ASA, Balaban J, Hall-Kier alpha,
    valence electrons, radical electrons). Concatenated across roles.

    Encodes *what kind of* component, not just *which* component.

    Args:
        components: List of dicts with component SMILES as values.

    Returns:
        np.ndarray of shape (n_reactions, n_components * 14), float32.
    """
    if not components:
        raise ValueError("Empty components list")

    # Use SMILES-valued keys only (skip categorical name keys)
    # Heuristic: if a key has '_smiles' suffix, prefer it; otherwise use
    # the key directly if its values look like SMILES
    sample = components[0]
    keys = sorted(sample.keys())

    # For BH Doyle plates, prefer *_smiles keys over categorical name keys
    smiles_keys = [k for k in keys if k.endswith('_smiles')]
    if smiles_keys:
        keys = smiles_keys
    else:
        # Filter to keys whose values contain SMILES-like characters
        from rdkit import Chem
        keys = [k for k in keys if Chem.MolFromSmiles(str(sample.get(k, ''))) is not None]

    n_props = len(_COMPONENT_PROPS)
    total_dim = len(keys) * n_props
    X = np.zeros((len(components), total_dim), dtype=np.float32)

    for i, comp in enumerate(components):
        for j, key in enumerate(keys):
            smi = str(comp.get(key, ''))
            X[i, j * n_props:(j + 1) * n_props] = _compute_mol_props(smi)

    print(f"  Component descriptors: {X.shape} "
          f"({len(keys)} components × {n_props} props)")
    return X


# =========================================================================
# LIGAND STERIC / ELECTRONIC DESCRIPTORS
# =========================================================================

def create_ligand_descriptors(components: List[Dict[str, str]]) -> np.ndarray:
    """Steric and electronic descriptors for phosphine/amine ligands.

    For Pd-catalyzed couplings (Buchwald-Hartwig, Suzuki), ligand choice
    dominates yield. This encodes ligand properties that correlate with
    catalytic activity:

    Steric (8D):
      - MW, num heavy atoms, num rotatable bonds
      - num atoms alpha/beta to P (proxy for cone angle)
      - Fraction sp3 carbons, num rings, num aromatic rings
      - Approximate Tolman cone angle from substituent MW

    Electronic (8D):
      - Gasteiger partial charge on P (or N for amine ligands)
      - Electronegativity-weighted sum near coordinating atom
      - LogP (lipophilicity), TPSA (polarity)
      - Num lone pairs on coordinating atom
      - Num electron-donating groups (alkyl on P)
      - Num electron-withdrawing groups (aryl on P)
      - HOMO energy proxy (HallKierAlpha)
      - Total valence electrons

    Falls back to zeros if no ligand key found or ligand is not parseable.

    Args:
        components: List of dicts with component SMILES.
            Must contain 'ligand' or 'ligand_smiles' key.

    Returns:
        np.ndarray of shape (n_reactions, 16), float32.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem

    if not components:
        raise ValueError("Empty components list")

    # Find the ligand key
    sample = components[0]
    if 'ligand_smiles' in sample:
        lig_key = 'ligand_smiles'
    elif 'ligand' in sample:
        lig_key = 'ligand'
    else:
        raise ValueError("No 'ligand' or 'ligand_smiles' key in components")

    X = np.zeros((len(components), 16), dtype=np.float32)

    for i, comp in enumerate(components):
        smi = str(comp.get(lig_key, ''))
        mol = Chem.MolFromSmiles(smi) if smi and smi.strip() else None
        if mol is None:
            continue

        try:
            # --- Steric (8D) ---
            X[i, 0] = Descriptors.MolWt(mol)
            X[i, 1] = mol.GetNumHeavyAtoms()
            X[i, 2] = Descriptors.NumRotatableBonds(mol)

            # Atoms near coordinating atom (P or N)
            coord_atom = None
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'P':
                    coord_atom = atom
                    break
            if coord_atom is None:
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == 'N' and atom.GetDegree() <= 3:
                        coord_atom = atom
                        break

            if coord_atom is not None:
                # Alpha atoms (directly bonded to P/N)
                alpha = [n for n in coord_atom.GetNeighbors()]
                X[i, 3] = len(alpha)
                # Beta atoms (2 bonds from P/N)
                beta = set()
                for a in alpha:
                    for b in a.GetNeighbors():
                        if b.GetIdx() != coord_atom.GetIdx():
                            beta.add(b.GetIdx())
                X[i, 4] = len(beta)

                # Substituent MW proxy for cone angle
                # Sum MW of all fragments attached to P/N
                sub_mw = sum(n.GetMass() for n in alpha)
                X[i, 5] = sub_mw
            else:
                X[i, 3:6] = 0.0

            X[i, 6] = Descriptors.FractionCSP3(mol)
            X[i, 7] = Descriptors.NumAromaticRings(mol)

            # --- Electronic (8D) ---
            AllChem.ComputeGasteigerCharges(mol)

            if coord_atom is not None:
                charge = coord_atom.GetDoubleProp('_GasteigerCharge')
                X[i, 8] = charge if np.isfinite(charge) else 0.0
                # Electronegativity-weighted neighbor sum
                neighbor_charge = 0.0
                for n in coord_atom.GetNeighbors():
                    nc = n.GetDoubleProp('_GasteigerCharge')
                    if np.isfinite(nc):
                        neighbor_charge += nc
                X[i, 9] = neighbor_charge

                # Count alkyl vs aryl substituents on coordinating atom
                n_alkyl = sum(1 for n in coord_atom.GetNeighbors()
                              if not n.GetIsAromatic() and n.GetSymbol() == 'C')
                n_aryl = sum(1 for n in coord_atom.GetNeighbors()
                             if n.GetIsAromatic())
                X[i, 13] = n_alkyl  # electron-donating
                X[i, 14] = n_aryl   # electron-withdrawing (relative)

            X[i, 10] = Descriptors.MolLogP(mol)
            X[i, 11] = Descriptors.TPSA(mol)
            X[i, 12] = Descriptors.HallKierAlpha(mol)
            X[i, 15] = Descriptors.NumValenceElectrons(mol)

        except Exception:
            pass

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Ligand descriptors: {X.shape} (8 steric + 8 electronic)")
    return X
