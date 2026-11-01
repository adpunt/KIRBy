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

import os
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


# =========================================================================
# CHYTORCH GRAPHORMER (PRETRAINED GRAPH-BASED REACTION EMBEDDING)
# =========================================================================


def _chytorch_py311_patch():
    """Patch chytorch NamedTuple+DataTypeMixin for Python 3.11+ compatibility.

    chytorch uses `class Batch(NamedTuple, DataTypeMixin)` which is rejected
    by Python 3.11's NamedTupleMeta. We strip `DataTypeMixin` from the
    inheritance in the 3 affected source files. Idempotent — only patches once.
    """
    import sys
    if sys.version_info < (3, 11):
        return

    import site

    # Search both system and user site-packages
    search_paths = list(site.getsitepackages() or [])
    try:
        search_paths.append(site.getusersitepackages())
    except AttributeError:
        pass

    base = None
    for p in search_paths:
        candidate = os.path.join(p, 'chytorch', 'utils', 'data')
        if os.path.isdir(candidate):
            base = candidate
            break
    if base is None:
        return

    targets = [
        os.path.join(base, 'molecule', 'conformer.py'),
        os.path.join(base, 'molecule', 'encoder.py'),
        os.path.join(base, 'reaction', 'encoder.py'),
    ]
    for fpath in targets:
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath) as f:
                src = f.read()
            if 'NamedTuple, DataTypeMixin' in src:
                with open(fpath, 'w') as f:
                    f.write(src.replace('NamedTuple, DataTypeMixin',
                                        'NamedTuple'))
        except (PermissionError, OSError):
            pass  # readonly site-packages — user must patch manually


def create_chytorch_graphormer(rxn_smiles: List[str],
                               batch_size: int = 20) -> np.ndarray:
    """Chytorch Graphormer CLS embedding — pretrained graph-based reaction rep.

    Extracts CLS token from a Bidirectional Graphormer pretrained on ~2M
    reactions (USPTO + Pistachio) for atom-mapping. Operates on reaction
    graph structure directly — no SMILES tokenization, no separate AAM step.

    The model (chytorch-rxnmap) natively handles atom-atom mapping, so
    unmapped reaction SMILES work directly (unlike CGR Morgan which needs
    rxnmapper). CLS embedding is 1024D (Graphormer hidden dimension).

    Nugmanov et al. / chytorch-rxnmap.

    Args:
        rxn_smiles: Reaction SMILES in 'reactants>>products' format.
        batch_size: Reactions per batch (default 20).

    Returns:
        np.ndarray of shape (n_reactions, 1024), float32.

    Requires: pip install chytorch chytorch-rxnmap torch
    Note: Auto-downloads pretrained weights on first use (~78MB).

    A reaction that chython cannot parse, or a batch whose forward pass
    raises, now raises RuntimeError naming how many failed and the first
    error (2026-11-01). Until then each failure was swallowed and left a zero
    row, and a failure of every batch returned an all-zero matrix with no
    error: the cached buchwald_hartwig matrix of 2026-05-01 (3,955 x 1,024)
    is all zeros, which every model then scored as the mean (27.8035 RMSE
    under ridge, xgboost and lightgbm alike).
    """
    D = 1024  # Graphormer hidden dimension (verified from model architecture)

    if not rxn_smiles:
        return np.zeros((0, D), dtype=np.float32)

    try:
        import torch
    except ImportError:
        raise ImportError(
            "Chytorch Graphormer requires PyTorch: pip install torch"
        )
    try:
        _chytorch_py311_patch()
        from chytorch.zoo.rxnmap import Model
        from chytorch.utils.data import (
            ReactionEncoderDataset, collate_encoded_reactions,
        )
    except ImportError:
        raise ImportError(
            "Chytorch Graphormer requires chytorch-rxnmap: "
            "pip install chytorch chytorch-rxnmap"
        )
    try:
        from chython import smiles as chython_smiles
    except ImportError:
        raise ImportError(
            "Chytorch Graphormer requires chython: pip install chython"
        )
    from math import inf

    print(f"  Loading Chytorch Graphormer (pretrained rxnmap)...")
    model = Model()  # __init__ loads weights + calls eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Parse reactions → ReactionContainer objects
    parsed = []       # (index, ReactionContainer) for successful parses
    failed_idx = []   # indices of failed parses
    first_error = None
    for i, smi in enumerate(rxn_smiles):
        try:
            rxn = chython_smiles(smi)
            rxn.canonicalize()
            parsed.append((i, rxn))
        except Exception as e:
            failed_idx.append(i)
            if first_error is None:
                first_error = f'reaction {i}: {type(e).__name__}: {e}'
    if failed_idx:
        raise RuntimeError(
            f"Chytorch Graphormer: chython could not parse {len(failed_idx)} of "
            f"{len(rxn_smiles)} reactions; first: {first_error}")

    # Extract embeddings in batches
    X = np.zeros((len(rxn_smiles), D), dtype=np.float32)
    indices = [idx for idx, _ in parsed]
    rxn_objs = [rxn for _, rxn in parsed]
    ds = ReactionEncoderDataset(rxn_objs)

    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            end = min(start + batch_size, len(ds))
            try:
                items = [ds[j] for j in range(start, end)]
                batch = collate_encoded_reactions(items)
                atoms, neighbors, distances, roles = batch
                atoms = atoms.to(device)
                neighbors = neighbors.to(device)
                distances = distances.to(device)
                roles = roles.to(device)

                # Parent-class forward: get embeddings (not attention weights)
                n = atoms.size(1)
                d_mask = (torch.zeros_like(roles, dtype=torch.float)
                          .masked_fill_(roles == 0, -inf)
                          .view(-1, 1, 1, n))
                d_mask = (d_mask.expand(-1, model.nhead, n, -1)
                          .flatten(end_dim=1))

                x = (model.molecule_encoder((atoms, neighbors, distances))
                     * (roles > 1).unsqueeze_(-1))
                x = x + model.role_encoder(roles)
                for lr in model.layers:
                    x, _ = lr(x, d_mask)

                cls_tokens = x[:, 0, :].cpu().numpy().astype(np.float32)
                for k, global_idx in enumerate(indices[start:end]):
                    X[global_idx] = cls_tokens[k]
            except Exception as e:
                raise RuntimeError(
                    f"Chytorch Graphormer: the forward pass failed on reactions "
                    f"{start}-{end - 1}: {type(e).__name__}: {e}") from e

            if end < len(ds) and end % 100 == 0:
                print(f"    Graphormer progress: {end}/{len(ds)}")

    n_ok = int((X.sum(axis=1) != 0).sum())
    print(f"  Chytorch Graphormer embeddings: {X.shape} "
          f"({n_ok}/{len(rxn_smiles)} non-zero rows)")
    return X


# =========================================================================
# CGR-SMILES LANGUAGE MODEL (PRETRAINED SEQUENTIAL EMBEDDING)
# =========================================================================

# Global cache for CGR-SMILES LM model/tokenizer
_CGR_SMILES_MODEL = None
_CGR_SMILES_TOKENIZER = None


def _rxn_to_cgr_smiles_batch(rxn_smiles: List[str]) -> List[str]:
    """Batch convert reaction SMILES to CGR-SMILES strings.

    Atom-maps first via rxnmapper, then converts to CGR-SMILES via
    chython compose + str. Shared by create_cgr_smiles_lm and
    cgr_smiles_lora.

    Args:
        rxn_smiles: Reaction SMILES (unmapped or mapped).

    Returns:
        List of CGR-SMILES strings (empty string for failures).
    """
    from chython import smiles as chython_smiles
    _patch_cgr_len()

    if not rxn_smiles:
        return []

    # Atom-map first (required for CGR compose)
    mapped = _map_reactions(rxn_smiles)

    cgr_smiles = []
    for smi in mapped:
        try:
            rxn = chython_smiles(smi)
            if rxn is None:
                cgr_smiles.append('')
                continue
            rxn.canonicalize()
            cgr = rxn.compose()
            cgr_smiles.append(str(cgr))
        except Exception:
            cgr_smiles.append('')

    n_ok = sum(1 for s in cgr_smiles if s)
    print(f"  CGR-SMILES conversion: {n_ok}/{len(rxn_smiles)} success")
    return cgr_smiles


def _find_cgr_smiles_checkpoint() -> str:
    """Find the MLM-pretrained CGR-SMILES checkpoint.

    Searches:
    1. KIRBY_CGR_CHECKPOINT env var
    2. {KIRBY_CACHE_DIR}/cgr_smiles_pretrain/checkpoint/
    3. /data/stat-ecr/scat9264/kirby_data/cgr_smiles_pretrain/checkpoint/  (post-2026-05-07)
    4. /data/stat-cadd/scat9264/kirby_data/cgr_smiles_pretrain/checkpoint/ (legacy)

    Returns:
        Path string to checkpoint, or base model ID as fallback.
    """
    from pathlib import Path

    # Env var override
    env_path = os.environ.get('KIRBY_CGR_CHECKPOINT')
    if env_path and Path(env_path).is_dir():
        return env_path

    # Default cache locations
    cache_root = os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby'))
    candidates = [
        Path(cache_root) / 'cgr_smiles_pretrain' / 'checkpoint',
        Path('/data/stat-ecr/scat9264/kirby_data/cgr_smiles_pretrain/checkpoint'),
        Path('/data/stat-cadd/scat9264/kirby_data/cgr_smiles_pretrain/checkpoint'),
    ]
    for p in candidates:
        if p.is_dir() and (p / 'config.json').exists():
            return str(p)

    # Fallback: base ChemBERTa (no CGR pretraining)
    print("  WARNING: No CGR-SMILES pretrained checkpoint found. "
          "Using base ChemBERTa (dynamic bond token embeddings will be "
          "near-zero). Run cgr_smiles_pretrain first for best results.")
    return 'DeepChem/ChemBERTa-77M-MLM'


def create_cgr_smiles_lm(rxn_smiles: List[str],
                         batch_size: int = 32,
                         checkpoint_dir: str = None) -> np.ndarray:
    """CGR-SMILES language model embeddings (frozen, 384D).

    Extracts mean-pooled hidden states from a ChemBERTa model that was
    continued-pretrained on CGR-SMILES.

    Pipeline: rxn_smiles -> atom-map -> CGR-SMILES -> tokenize -> encode -> mean-pool

    Falls back to base ChemBERTa with extended tokenizer if no checkpoint
    is found (dynamic bond token embeddings will be near-zero).

    Args:
        rxn_smiles: Reaction SMILES in 'reactants>>products' format.
        batch_size: Sequences per batch (default 32).
        checkpoint_dir: Path to MLM-pretrained checkpoint. If None, auto-detects.

    Returns:
        np.ndarray of shape (n_reactions, 384), float32.

    Requires: pip install chython rxnmapper transformers torch
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    D = 384  # ChemBERTa-77M-MLM hidden_size

    if not rxn_smiles:
        return np.zeros((0, D), dtype=np.float32)

    # Load model (global cache)
    global _CGR_SMILES_MODEL, _CGR_SMILES_TOKENIZER

    if _CGR_SMILES_MODEL is None:
        model_path = checkpoint_dir or _find_cgr_smiles_checkpoint()
        print(f"  Loading CGR-SMILES model from {model_path}...")

        _CGR_SMILES_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
        _CGR_SMILES_MODEL = AutoModel.from_pretrained(model_path)

        # If using base model (no checkpoint), extend tokenizer + resize
        if model_path == 'DeepChem/ChemBERTa-77M-MLM':
            from kirby.finetuning.cgr_smiles_pretrain import (
                get_dynamic_bond_tokens,
            )
            new_tokens = get_dynamic_bond_tokens()
            existing = set(_CGR_SMILES_TOKENIZER.get_vocab().keys())
            to_add = [t for t in new_tokens if t not in existing]
            if to_add:
                _CGR_SMILES_TOKENIZER.add_special_tokens(
                    {'additional_special_tokens': to_add})
                _CGR_SMILES_MODEL.resize_token_embeddings(
                    len(_CGR_SMILES_TOKENIZER))

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _CGR_SMILES_MODEL = _CGR_SMILES_MODEL.to(device).eval()
        print(f"  CGR-SMILES model loaded on {device} "
              f"(vocab={len(_CGR_SMILES_TOKENIZER)})")

    tokenizer = _CGR_SMILES_TOKENIZER
    model = _CGR_SMILES_MODEL
    device = next(model.parameters()).device

    # Convert reactions to CGR-SMILES
    cgr_smiles = _rxn_to_cgr_smiles_batch(rxn_smiles)

    # Track which indices have valid CGR-SMILES
    valid_idx = [i for i, s in enumerate(cgr_smiles) if s]
    valid_strings = [cgr_smiles[i] for i in valid_idx]

    if not valid_strings:
        print("  WARNING: All CGR-SMILES conversions failed")
        return np.zeros((len(rxn_smiles), D), dtype=np.float32)

    # Extract embeddings in batches
    all_embeddings = []
    with torch.no_grad():
        for start in range(0, len(valid_strings), batch_size):
            batch = valid_strings[start:start + batch_size]
            try:
                inputs = tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=256, return_tensors='pt',
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)

                # Mean pooling over non-padding tokens
                attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
                sum_emb = (outputs.last_hidden_state * attention_mask).sum(dim=1)
                count = attention_mask.sum(dim=1).clamp(min=1)
                mean_emb = (sum_emb / count).cpu().numpy()
                all_embeddings.append(mean_emb)
            except Exception as e:
                # Zero vectors for failed batches
                all_embeddings.append(
                    np.zeros((len(batch), D), dtype=np.float32))
                print(f"  WARNING: CGR-SMILES batch failed: {e}")

    valid_embs = np.vstack(all_embeddings).astype(np.float32)

    # Build full array with zero vectors for failures
    X = np.zeros((len(rxn_smiles), D), dtype=np.float32)
    for local_i, global_i in enumerate(valid_idx):
        X[global_i] = valid_embs[local_i]

    n_zero = np.sum(np.all(X == 0, axis=1))
    print(f"  CGR-SMILES LM embeddings: {X.shape} "
          f"({len(rxn_smiles) - n_zero}/{len(rxn_smiles)} non-zero rows)")
    return X


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
# CGR (CONDENSED GRAPH OF REACTION) FINGERPRINTS
# =========================================================================
# CGR superimposes reactant and product graphs into a single graph where
# bonds that change during the reaction are "dynamic bonds" (order != p_order).
# This encodes the reaction center as a first-class structural feature.
#
# CGR Morgan FP: extends standard Morgan/ECFP circular hashing to CGR graphs.
# Atom invariants include both reactant and product properties (charge, p_charge,
# etc.). Bond invariants include (order, p_order) — dynamic bonds hash differently
# from static bonds, giving substructure patterns that natively encode the
# transformation context. This is NOT a published method; it is a synthesis of
# standard Morgan iteration with CGR graph properties.
#
# Requires: chython (CGR parsing), rxnmapper (atom-atom mapping).
# =========================================================================


def _patch_cgr_len():
    """Patch CGRContainer.__len__ if missing.

    chython's CGRContainer doesn't define __len__, but str() on complex
    CGRs can trigger len(self) internally (via atoms_order/Morgan). Adding
    __len__ = len(self._atoms) fixes it across all chython versions.
    """
    from chython.containers.cgr import CGRContainer
    if not hasattr(CGRContainer, '__len__'):
        CGRContainer.__len__ = lambda self: len(self._atoms)


def _stable_hash(tup) -> int:
    """Deterministic hash of a tuple for Morgan fingerprinting.

    Python's built-in hash() is randomized (PYTHONHASHSEED) since 3.3.
    We need deterministic hashing for reproducible fingerprints and caching.
    """
    import hashlib
    import struct
    data = str(tup).encode('utf-8')
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return struct.unpack('<Q', digest)[0]


def _map_reactions(rxn_smiles: List[str], batch_size: int = 10) -> List[str]:
    """Atom-map reaction SMILES using rxnmapper.

    Atom-atom mapping (AAM) is required for CGR construction — chython's
    compose() needs to know which atoms correspond between reactant and
    product sides. Uses rxnmapper (Schwaller et al.) for attention-based AAM.

    Args:
        rxn_smiles: Unmapped reaction SMILES.
        batch_size: Reactions per rxnmapper batch.

    Returns:
        List of atom-mapped reaction SMILES (same length as input).
    """
    try:
        from rxnmapper import RXNMapper
    except ImportError:
        raise ImportError(
            "CGR Morgan requires rxnmapper for atom-atom mapping: "
            "pip install rxnmapper"
        )

    if not rxn_smiles:
        return []

    rxn_mapper = RXNMapper()
    mapped = []

    for start in range(0, len(rxn_smiles), batch_size):
        batch = rxn_smiles[start:start + batch_size]
        try:
            results = rxn_mapper.get_attention_guided_atom_maps(batch)
            mapped.extend(r['mapped_rxn'] for r in results)
        except Exception:
            # Fall back to unmapped — will fail at compose() → zero vector
            mapped.extend(batch)

        done = min(start + batch_size, len(rxn_smiles))
        if done < len(rxn_smiles) and done % 500 == 0:
            print(f"    rxnmapper progress: {done}/{len(rxn_smiles)}")

    return mapped


def _cgr_morgan_single(mapped_rxn_smi: str, radius: int = 2,
                        n_bits: int = 2048) -> np.ndarray:
    """Compute CGR Morgan fingerprint for a single atom-mapped reaction.

    Parses mapped SMILES → CGR → runs Morgan iteration with dynamic
    atom/bond invariants → folds into bit vector.

    Args:
        mapped_rxn_smi: Atom-mapped reaction SMILES (with :N tokens).
        radius: Morgan iteration radius (default 2, like ECFP4).
        n_bits: Fingerprint length (default 2048).

    Returns:
        1D uint8 array of length n_bits. Zero vector on any failure.
    """
    try:
        from chython import smiles as chython_smiles
    except ImportError:
        raise ImportError("CGR Morgan requires chython: pip install chython")
    _patch_cgr_len()

    try:
        rxn = chython_smiles(mapped_rxn_smi)
        cgr = rxn.compose()
    except Exception:
        return np.zeros(n_bits, dtype=np.uint8)

    atom_ids = list(cgr._atoms.keys())
    if not atom_ids:
        return np.zeros(n_bits, dtype=np.uint8)

    # Build adjacency: atom_id -> [(neighbor_id, bond_invariant), ...]
    adjacency = {aid: [] for aid in atom_ids}
    for n1, n2, bond in cgr.bonds():
        bond_inv = _stable_hash((bond.order, bond.p_order))
        adjacency[n1].append((n2, bond_inv))
        adjacency[n2].append((n1, bond_inv))

    # Atom invariants (radius 0)
    current_hash = {}
    for aid in atom_ids:
        atom = cgr._atoms[aid]
        current_hash[aid] = _stable_hash((
            atom.atomic_number,
            atom.isotope,
            atom.charge,
            atom.p_charge,
            atom.is_radical,
            atom.p_is_radical,
            len(adjacency[aid]),
        ))

    # Collect environment hashes across all radii
    all_hashes = set(current_hash.values())

    # Morgan iteration
    for _ in range(radius):
        new_hash = {}
        for aid in atom_ids:
            neighbor_info = sorted(
                (bond_inv, current_hash[nbr_id])
                for nbr_id, bond_inv in adjacency[aid]
            )
            new_hash[aid] = _stable_hash((current_hash[aid], tuple(neighbor_info)))
            all_hashes.add(new_hash[aid])
        current_hash = new_hash

    # Fold into bit vector
    fp = np.zeros(n_bits, dtype=np.uint8)
    for h in all_hashes:
        fp[h % n_bits] = 1

    return fp


def create_cgr_morgan(rxn_smiles: List[str], radius: int = 2,
                      n_bits: int = 2048) -> np.ndarray:
    """CGR Morgan fingerprint — extended Morgan iteration on CGR graph.

    Parses reaction SMILES → atom-maps via rxnmapper → builds CGR via
    chython → runs Morgan iteration with dynamic bond invariants → folds
    into binary fingerprint. Dynamic bonds (where order != p_order) hash
    differently from static bonds, encoding the reaction center natively.

    This is a novel method (synthesis of Morgan iteration + CGR properties),
    not a published fingerprint. Closest published: DOPtools CircuS
    (Sidorov et al., Digital Discovery 2025).

    Args:
        rxn_smiles: Reaction SMILES in 'reactants>>products' format.
        radius: Morgan iteration radius (default 2, like ECFP4).
        n_bits: Fingerprint length (default 2048).

    Returns:
        np.ndarray of shape (n_reactions, n_bits), float32 with binary values.

    Requires: pip install chython rxnmapper
    """
    if not rxn_smiles:
        return np.zeros((0, n_bits), dtype=np.float32)

    print(f"  CGR Morgan: mapping {len(rxn_smiles)} reactions with rxnmapper...")
    mapped = _map_reactions(rxn_smiles)

    X = np.zeros((len(rxn_smiles), n_bits), dtype=np.float32)
    n_failed = 0
    for i, msmi in enumerate(mapped):
        try:
            X[i] = _cgr_morgan_single(msmi, radius, n_bits).astype(np.float32)
        except Exception:
            n_failed += 1

    n_nonzero = int((X.sum(axis=1) > 0).sum())
    if n_failed > 0:
        print(f"  CGR Morgan: {n_failed}/{len(rxn_smiles)} reactions failed "
              f"CGR conversion")
    print(f"  CGR Morgan fingerprint: {X.shape} "
          f"({n_nonzero}/{len(rxn_smiles)} non-zero rows)")
    return X


# =========================================================================
# CGR ATOM-PAIR, CGR TORSION, CGR BOND CHANGES (hand-written, 2026-11-01)
# =========================================================================
#
# Three views on the same condensed graph of reaction (CGR) that
# _cgr_morgan_single reads, with the same atom and bond invariants: an atom is
# (atomic number, isotope, charge before, charge after, radical before,
# radical after, number of CGR neighbours); a bond is (order before, order
# after), where None means the bond does not exist on that side. None of the
# three is a published fingerprint.


def _map_reactions_cached(rxn_smiles: List[str]) -> List[str]:
    """_map_reactions, with the result kept on disk.

    rxnmapper is the slow step of every CGR view, and each CGR view would
    otherwise map the same reactions again. The mapped SMILES are stored at
    {KIRBY_CACHE_DIR}/rxn_mapped/<sha1 of the input SMILES>.json, one file per
    list of reactions, so the list and its order are the key. The mapping
    itself is _map_reactions, unchanged.
    """
    import hashlib
    import json
    from pathlib import Path
    if not rxn_smiles:
        return []
    key = hashlib.sha1('\n'.join(rxn_smiles).encode('utf-8')).hexdigest()
    root = Path(os.environ.get('KIRBY_CACHE_DIR',
                               str(Path.home() / '.cache' / 'kirby'))) / 'rxn_mapped'
    f = root / f'{key}.json'
    if f.exists():
        mapped = json.loads(f.read_text())
        if len(mapped) == len(rxn_smiles):
            return mapped
    mapped = _map_reactions(rxn_smiles)
    root.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f'{key}.tmp.{os.getpid()}.json')
    tmp.write_text(json.dumps(mapped))
    os.replace(tmp, f)
    return mapped


def _cgr_graph(mapped_rxn_smi: str):
    """(atom ids, atom invariant hash per atom, adjacency) of one reaction's
    CGR, with _cgr_morgan_single's invariants. adjacency[a] is a list of
    (neighbour, (order before, order after)). Raises when chython cannot
    parse or compose the reaction."""
    from chython import smiles as chython_smiles
    _patch_cgr_len()
    rxn = chython_smiles(mapped_rxn_smi)
    cgr = rxn.compose()
    atom_ids = list(cgr._atoms.keys())
    adjacency = {aid: [] for aid in atom_ids}
    for n1, n2, bond in cgr.bonds():
        b = (bond.order, bond.p_order)
        adjacency[n1].append((n2, b))
        adjacency[n2].append((n1, b))
    inv = {}
    for aid in atom_ids:
        atom = cgr._atoms[aid]
        inv[aid] = _stable_hash((
            atom.atomic_number, atom.isotope, atom.charge, atom.p_charge,
            atom.is_radical, atom.p_is_radical, len(adjacency[aid]),
        ))
    return atom_ids, inv, adjacency, cgr


def _bfs_distances(adjacency, src, keep):
    """Shortest-path bond counts from src over the bonds keep(b) accepts."""
    dist = {src: 0}
    frontier = [src]
    while frontier:
        nxt = []
        for a in frontier:
            for nb, b in adjacency[a]:
                if nb not in dist and keep(b):
                    dist[nb] = dist[a] + 1
                    nxt.append(nb)
        frontier = nxt
    return dist


def _cgr_atompair_single(mapped_rxn_smi: str, n_bits: int = 2048,
                         max_distance: int = 30) -> np.ndarray:
    """CGR atom-pair bits for one atom-mapped reaction (see
    create_cgr_atompair). Zero vector when the CGR cannot be built."""
    try:
        atom_ids, inv, adjacency, _ = _cgr_graph(mapped_rxn_smi)
    except Exception:
        return np.zeros(n_bits, dtype=np.uint8)
    any_bond = lambda b: True                       # noqa: E731
    before = lambda b: b[0] is not None             # noqa: E731
    after = lambda b: b[1] is not None              # noqa: E731
    fp = np.zeros(n_bits, dtype=np.uint8)
    for i, a in enumerate(atom_ids):
        d_cgr = _bfs_distances(adjacency, a, any_bond)
        d_bef = _bfs_distances(adjacency, a, before)
        d_aft = _bfs_distances(adjacency, a, after)
        for b in atom_ids[i + 1:]:
            d = d_cgr.get(b)
            if d is None or d > max_distance:
                continue
            lo, hi = sorted((inv[a], inv[b]))
            h = _stable_hash((lo, d_bef.get(b, -1), d_aft.get(b, -1), hi))
            fp[h % n_bits] = 1
    return fp


def _cgr_torsion_single(mapped_rxn_smi: str, n_bits: int = 2048) -> np.ndarray:
    """CGR topological-torsion bits for one atom-mapped reaction (see
    create_cgr_torsion). Zero vector when the CGR cannot be built."""
    try:
        atom_ids, inv, adjacency, _ = _cgr_graph(mapped_rxn_smi)
    except Exception:
        return np.zeros(n_bits, dtype=np.uint8)
    fp = np.zeros(n_bits, dtype=np.uint8)
    bond_h = {}
    for a in atom_ids:
        for b, bb in adjacency[a]:
            for c, bc in adjacency[b]:
                if c == a:
                    continue
                for d, bd in adjacency[c]:
                    if d in (a, b):
                        continue
                    fwd = (inv[a], bond_h.setdefault(bb, _stable_hash(bb)),
                           inv[b], bond_h.setdefault(bc, _stable_hash(bc)),
                           inv[c], bond_h.setdefault(bd, _stable_hash(bd)), inv[d])
                    rev = fwd[::-1]
                    fp[_stable_hash(min(fwd, rev)) % n_bits] = 1
    return fp


def _fold_cgr_bits(rxn_smiles: List[str], single_fn, label: str,
                   n_bits: int = 2048) -> np.ndarray:
    if not rxn_smiles:
        return np.zeros((0, n_bits), dtype=np.float32)
    print(f"  {label}: mapping {len(rxn_smiles)} reactions with rxnmapper (cached)...")
    mapped = _map_reactions_cached(rxn_smiles)
    X = np.zeros((len(rxn_smiles), n_bits), dtype=np.float32)
    for i, msmi in enumerate(mapped):
        X[i] = single_fn(msmi, n_bits=n_bits)
    n_zero = int((X.sum(axis=1) == 0).sum())
    print(f"  {label}: {X.shape}, {n_zero}/{len(rxn_smiles)} rows all zero "
          f"(the CGR could not be built)")
    return X


def create_cgr_atompair(rxn_smiles: List[str], n_bits: int = 2048) -> np.ndarray:
    """CGR atom-pair fingerprint: a hand-written port of RDKit's atom-pair
    definition (Carhart et al. 1985) to the CGR graph. Not a published
    fingerprint.

    RDKit's atom pair is (atom code, topological distance, atom code) for
    every pair of atoms of a molecule. Here, for every pair of atoms in the
    reaction's CGR that are joined by a path of at most 30 bonds: (atom
    invariant, distance over the bonds that exist before the reaction,
    distance over the bonds that exist after it, atom invariant), with -1 when
    no path exists on that side. The atom invariant is _cgr_morgan_single's
    (element, isotope, charge and radical before and after, CGR degree). The
    two distances are how the bond changes enter: RDKit's atom pair has no
    bond term. Hashed with blake2b and folded to n_bits presence bits.

    Atom-mapped with rxnmapper, as create_cgr_morgan. A reaction whose CGR
    cannot be built is a zero row (counted in the printout).

    Returns: (n_reactions, n_bits) float32, 0/1.
    Requires: chython, rxnmapper.
    """
    return _fold_cgr_bits(rxn_smiles, _cgr_atompair_single, 'CGR atom-pair', n_bits)


def create_cgr_torsion(rxn_smiles: List[str], n_bits: int = 2048) -> np.ndarray:
    """CGR topological-torsion fingerprint: a hand-written port of RDKit's
    topological-torsion definition (Nilakantan et al. 1987) to the CGR graph.
    Not a published fingerprint.

    RDKit's torsion is every linear path of four atoms, coded by its four atom
    codes. Here, every path of four distinct atoms along CGR bonds, coded by
    the four atom invariants and the three bond invariants between them, the
    same invariants as _cgr_morgan_single (atom: element, isotope, charge and
    radical before and after, CGR degree; bond: order before and after, None
    where the bond does not exist on that side). A path and its reverse are
    the same torsion (the smaller of the two tuples is hashed). Hashed with
    blake2b and folded to n_bits presence bits.

    Atom-mapped with rxnmapper, as create_cgr_morgan. A reaction whose CGR
    cannot be built is a zero row (counted in the printout).

    Returns: (n_reactions, n_bits) float32, 0/1.
    Requires: chython, rxnmapper.
    """
    return _fold_cgr_bits(rxn_smiles, _cgr_torsion_single, 'CGR torsion', n_bits)


# Element pairs the bond-change counts are broken down by. Fixed, not learned
# from the data, so the view is the same on every fold.
_BOND_CHANGE_PAIRS = ('C-C', 'C-N', 'C-O', 'B-C', 'C-Cl', 'Br-C', 'C-I')
_ORDER_NAMES = {1: '1', 2: '2', 3: '3', 4: 'arom'}

CGR_BOND_CHANGE_FEATURES = (
    ['bonds_formed', 'bonds_broken', 'bonds_order_changed']
    + [f'formed_order_{v}' for v in _ORDER_NAMES.values()]
    + [f'broken_order_{v}' for v in _ORDER_NAMES.values()]
    + ['changed_order_up', 'changed_order_down']
    + [f'formed_{p}' for p in _BOND_CHANGE_PAIRS] + ['formed_other_pair']
    + [f'broken_{p}' for p in _BOND_CHANGE_PAIRS] + ['broken_other_pair']
    + ['atoms_charge_changed']
)


def _cgr_bond_changes_single(mapped_rxn_smi: str):
    """The CGR_BOND_CHANGE_FEATURES counts for one reaction, or None when the
    CGR cannot be built."""
    try:
        _, _, _, cgr = _cgr_graph(mapped_rxn_smi)
    except Exception:
        return None
    idx = {n: i for i, n in enumerate(CGR_BOND_CHANGE_FEATURES)}
    v = np.zeros(len(CGR_BOND_CHANGE_FEATURES), dtype=np.float32)
    as_num = {1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5}
    for n1, n2, bond in cgr.bonds():
        o, p = bond.order, bond.p_order
        if o == p:
            continue
        pair = '-'.join(sorted((cgr._atoms[n1].atomic_symbol,
                                cgr._atoms[n2].atomic_symbol)))
        pair = pair if pair in _BOND_CHANGE_PAIRS else 'other_pair'
        if o is None:
            v[idx['bonds_formed']] += 1
            if p in _ORDER_NAMES:
                v[idx[f'formed_order_{_ORDER_NAMES[p]}']] += 1
            v[idx[f'formed_{pair}']] += 1
        elif p is None:
            v[idx['bonds_broken']] += 1
            if o in _ORDER_NAMES:
                v[idx[f'broken_order_{_ORDER_NAMES[o]}']] += 1
            v[idx[f'broken_{pair}']] += 1
        else:
            v[idx['bonds_order_changed']] += 1
            if o in as_num and p in as_num:
                v[idx['changed_order_up' if as_num[p] > as_num[o]
                      else 'changed_order_down']] += 1
    for aid in cgr._atoms:
        atom = cgr._atoms[aid]
        if atom.charge != atom.p_charge:
            v[idx['atoms_charge_changed']] += 1
    return v


def create_cgr_bond_changes(rxn_smiles: List[str]) -> np.ndarray:
    """Counts of what changes in a reaction, read off its CGR. Hand-written,
    not a published descriptor set; 30 features, named in
    CGR_BOND_CHANGE_FEATURES:

      bonds formed, broken, and changed in order (3);
      bonds formed by order after (single, double, triple, aromatic) (4);
      bonds broken by order before (4);
      bonds whose order went up, and down, counting aromatic as 1.5 (2);
      bonds formed by element pair: C-C, C-N, C-O, B-C, C-Cl, Br-C, C-I,
      any other pair (8); bonds broken by the same pairs (8);
      atoms whose formal charge differs before and after (1).

    Bond orders are chython's (4 = aromatic); a bond of another order (8,
    'any') counts in the three totals only. Atom-mapped with rxnmapper, as
    create_cgr_morgan. A reaction whose CGR cannot be built is a zero row.

    Returns: (n_reactions, 30) float32 counts.
    Requires: chython, rxnmapper.
    """
    k = len(CGR_BOND_CHANGE_FEATURES)
    if not rxn_smiles:
        return np.zeros((0, k), dtype=np.float32)
    print(f"  CGR bond changes: mapping {len(rxn_smiles)} reactions with rxnmapper (cached)...")
    mapped = _map_reactions_cached(rxn_smiles)
    X = np.zeros((len(rxn_smiles), k), dtype=np.float32)
    n_fail = 0
    for i, msmi in enumerate(mapped):
        v = _cgr_bond_changes_single(msmi)
        if v is None:
            n_fail += 1
        else:
            X[i] = v
    print(f"  CGR bond changes: {X.shape}, {n_fail}/{len(rxn_smiles)} reactions "
          f"whose CGR could not be built (zero rows)")
    return X


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



# =========================================================================
# GENERIC REACTION VARIANTS OF MOLECULAR REPS
# =========================================================================
# Apply any molecular rep function as reactant_/product_/diff_ variants.
# Product: rep of the product molecule.
# Reactant: aggregated rep of all reactant molecules (OR for binary FPs,
#           mean-pool for continuous embeddings).
# Diff: product minus reactant — captures what changed.
# =========================================================================

# Registry: mol_rep_name → (module, function_name, binary)
_MOL_REP_VARIANTS = {
    'mordred':         ('kirby.representations.molecular', 'create_mordred', False),
    'mhg_gnn':         ('kirby.representations.molecular', 'create_mhg_gnn', False),
    'grover':          ('kirby.representations.molecular', 'create_grover', False),
    'mol2vec':         ('kirby.representations.molecular', 'create_mol2vec', False),
    'chemberta':       ('kirby.representations.molecular', 'create_chemberta', False),
    'molformer':       ('kirby.representations.molecular', 'create_molformer', False),
    'avalon':          ('kirby.representations.molecular', 'create_avalon', True),
    'pharmacophore3d': ('kirby.representations.molecular', 'create_pharmacophore3d', False),
}


def _get_mol_rep_fn(mol_rep_name: str):
    """Lazily import and return a molecular rep function."""
    import importlib
    module_path, fn_name, _ = _MOL_REP_VARIANTS[mol_rep_name]
    mod = importlib.import_module(module_path)
    return getattr(mod, fn_name)


def _make_product_rep(rxn_smiles: List[str], mol_rep_fn) -> np.ndarray:
    """Apply a molecular rep function to reaction products.

    Extracts the product SMILES (after '>>') from each reaction and
    computes the molecular representation in a single batch.
    """
    product_smiles = []
    for rxn in rxn_smiles:
        parts = rxn.split('>>')
        product_smiles.append(parts[-1] if len(parts) >= 2 else parts[0])
    return mol_rep_fn(product_smiles)


def _make_reactant_rep(rxn_smiles: List[str], mol_rep_fn,
                       binary: bool = False) -> np.ndarray:
    """Apply a molecular rep function to reaction reactants.

    For multi-reactant reactions (dot-separated), aggregates per-reaction:
      - binary=True: OR-combine (element-wise max)
      - binary=False: mean-pool

    Flattens all individual reactant molecules into one batch for efficiency
    (important for transformer-based reps), then aggregates back.
    """
    flat_smiles = []
    rxn_indices = []
    for i, rxn in enumerate(rxn_smiles):
        parts = rxn.split('>>')
        reactant_smi = parts[0] if len(parts) >= 2 else ''
        for smi in reactant_smi.split('.'):
            smi = smi.strip()
            if smi:
                flat_smiles.append(smi)
                rxn_indices.append(i)

    if not flat_smiles:
        # Fallback: compute on products just to discover dimension, return zeros
        product_smiles = []
        for rxn in rxn_smiles:
            parts = rxn.split('>>')
            product_smiles.append(parts[-1] if len(parts) >= 2 else parts[0])
        dim = mol_rep_fn(product_smiles[:1]).shape[1]
        return np.zeros((len(rxn_smiles), dim), dtype=np.float32)

    flat_reps = mol_rep_fn(flat_smiles)
    dim = flat_reps.shape[1]

    n = len(rxn_smiles)
    result = np.zeros((n, dim), dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)

    for j, rxn_i in enumerate(rxn_indices):
        if binary:
            result[rxn_i] = np.maximum(result[rxn_i], flat_reps[j])
        else:
            result[rxn_i] += flat_reps[j]
        counts[rxn_i] += 1

    if not binary:
        mask = counts > 0
        result[mask] /= counts[mask, np.newaxis]

    return result


def create_reaction_rep_variant(rxn_smiles: List[str], mol_rep_name: str,
                                role: str) -> np.ndarray:
    """Create a reaction variant of a molecular representation.

    Args:
        rxn_smiles: Reaction SMILES in 'reactants>>products' format.
        mol_rep_name: Name of molecular rep (e.g. 'mordred', 'avalon').
        role: One of 'product', 'reactant', 'diff'.

    Returns:
        np.ndarray of shape (n_reactions, rep_dim).
    """
    if mol_rep_name not in _MOL_REP_VARIANTS:
        raise ValueError(
            f"Unknown mol rep '{mol_rep_name}'. "
            f"Available: {sorted(_MOL_REP_VARIANTS)}"
        )

    fn = _get_mol_rep_fn(mol_rep_name)
    _, _, binary = _MOL_REP_VARIANTS[mol_rep_name]

    if role == 'product':
        return _make_product_rep(rxn_smiles, fn)
    elif role == 'reactant':
        return _make_reactant_rep(rxn_smiles, fn, binary=binary)
    elif role == 'diff':
        product = _make_product_rep(rxn_smiles, fn)
        reactant = _make_reactant_rep(rxn_smiles, fn, binary=binary)
        return product - reactant
    else:
        raise ValueError(f"Unknown role '{role}'. Use 'product', 'reactant', or 'diff'.")
