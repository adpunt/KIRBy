"""
Molecule-specific feature importance and attribution methods.

These methods provide chemical interpretability by mapping feature importance
back to molecular substructures, atoms, or tokens.

Methods:
- explain_fingerprint_bits: Map ECFP/Morgan bit importance to substructures
- explain_smiles_tokens: Token-level attribution for transformer models
- substructure_shap: SHAP values for predefined functional groups
"""

import numpy as np
from collections import defaultdict


# =============================================================================
# FINGERPRINT BIT ATTRIBUTION
# =============================================================================

def get_fingerprint_bit_info(smiles_list, radius=2, n_bits=2048):
    """
    Get bit information for Morgan fingerprints - maps bits to atom environments.

    Args:
        smiles_list: List of SMILES strings
        radius: Fingerprint radius (default: 2 for ECFP4)
        n_bits: Number of bits (default: 2048)

    Returns:
        tuple: (fingerprints, bit_info_list)
            - fingerprints: np.ndarray (n_molecules, n_bits)
            - bit_info_list: List of dicts mapping bit_idx -> [(atom_idx, radius), ...]
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise ImportError("Fingerprint attribution requires RDKit. Install with: conda install rdkit")

    fingerprints = []
    bit_info_list = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        bit_info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=n_bits, bitInfo=bit_info
        )

        fingerprints.append(np.array(fp))
        bit_info_list.append(bit_info)

    return np.array(fingerprints, dtype=np.float32), bit_info_list


def explain_fingerprint_bits(smiles_list, importance_scores, radius=2, n_bits=2048,
                            top_k=20, aggregate='mean'):
    """
    Map fingerprint bit importance scores back to molecular substructures.

    Given importance scores for each fingerprint bit, identifies the substructures
    (atom environments) responsible for each bit and aggregates importance by
    substructure pattern.

    Args:
        smiles_list: List of SMILES strings used to generate fingerprints
        importance_scores: Array of shape (n_bits,) with importance per bit
        radius: Fingerprint radius used (default: 2 for ECFP4)
        n_bits: Number of bits used (default: 2048)
        top_k: Number of top substructures to return (default: 20)
        aggregate: How to aggregate across molecules ('mean', 'max', 'sum')

    Returns:
        list: Top substructures as dicts with keys:
            - 'smarts': SMARTS pattern for the substructure
            - 'importance': Aggregated importance score
            - 'bit_idx': Original fingerprint bit index
            - 'radius': Atom environment radius
            - 'n_molecules': Number of molecules containing this substructure
            - 'example_smiles': Example molecule containing this substructure
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise ImportError("Fingerprint attribution requires RDKit. Install with: conda install rdkit")

    importance_scores = np.asarray(importance_scores)
    if importance_scores.shape[0] != n_bits:
        raise ValueError(f"importance_scores has {importance_scores.shape[0]} values, expected {n_bits}")

    # Collect substructure info across all molecules
    substructure_info = defaultdict(lambda: {
        'importance_values': [],
        'bit_idx': None,
        'radius': None,
        'molecules': [],
        'smarts': None
    })

    for mol_idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        bit_info = {}
        AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=n_bits, bitInfo=bit_info
        )

        for bit_idx, atom_envs in bit_info.items():
            if importance_scores[bit_idx] == 0:
                continue

            for atom_idx, env_radius in atom_envs:
                # Get SMARTS for this atom environment
                try:
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, env_radius, atom_idx)
                    atoms = set()
                    for bond_idx in env:
                        bond = mol.GetBondWithIdx(bond_idx)
                        atoms.add(bond.GetBeginAtomIdx())
                        atoms.add(bond.GetEndAtomIdx())
                    if not atoms:
                        atoms.add(atom_idx)

                    # Create submol and get SMARTS
                    atoms = list(atoms)
                    smarts = Chem.MolFragmentToSmiles(mol, atoms, canonical=True)
                except:
                    smarts = f"bit_{bit_idx}"

                key = (bit_idx, smarts)
                substructure_info[key]['importance_values'].append(importance_scores[bit_idx])
                substructure_info[key]['bit_idx'] = bit_idx
                substructure_info[key]['radius'] = env_radius
                substructure_info[key]['molecules'].append(smiles)
                substructure_info[key]['smarts'] = smarts

    # Aggregate importance scores
    results = []
    for key, info in substructure_info.items():
        if not info['importance_values']:
            continue

        values = np.array(info['importance_values'])
        if aggregate == 'mean':
            agg_importance = values.mean()
        elif aggregate == 'max':
            agg_importance = values.max()
        elif aggregate == 'sum':
            agg_importance = values.sum()
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")

        results.append({
            'smarts': info['smarts'],
            'importance': float(agg_importance),
            'bit_idx': info['bit_idx'],
            'radius': info['radius'],
            'n_molecules': len(set(info['molecules'])),
            'example_smiles': info['molecules'][0] if info['molecules'] else None
        })

    # Sort by importance and return top_k
    results.sort(key=lambda x: x['importance'], reverse=True)
    return results[:top_k]


# =============================================================================
# SMILES TOKEN ATTRIBUTION
# =============================================================================

def explain_smiles_tokens(smiles_list, model_name='chemberta', importance_method='attention',
                         top_k=10, device=None):
    """
    Get token-level importance for transformer models (ChemBERTa, MolFormer).

    Args:
        smiles_list: List of SMILES strings
        model_name: 'chemberta' or 'molformer'
        importance_method: 'attention' (fast) or 'gradient' (more accurate)
        top_k: Number of top tokens to return per molecule
        device: 'cpu' or 'cuda'

    Returns:
        list: Per-molecule token importance as list of dicts with keys:
            - 'smiles': Original SMILES
            - 'tokens': List of tokens
            - 'importance': Array of importance per token
            - 'top_tokens': List of (token, importance) tuples for top_k tokens
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError("Token attribution requires torch and transformers")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model and tokenizer
    if model_name == 'chemberta':
        model_id = "DeepChem/ChemBERTa-77M-MTR"
    elif model_name == 'molformer':
        model_id = "ibm/MoLFormer-XL-both-10pct"
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'chemberta' or 'molformer'")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, output_attentions=True)
    model = model.to(device).eval()

    results = []

    for smiles in smiles_list:
        # Tokenize
        encoded = tokenizer(smiles, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in encoded.items()}
        tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

        if importance_method == 'attention':
            # Use attention weights
            with torch.no_grad():
                outputs = model(**inputs)

            # Average attention across all heads and layers
            attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)
            # Stack and average: (layers, batch, heads, seq, seq) -> (seq,)
            attn_weights = torch.stack(attentions).mean(dim=(0, 1, 2))  # (seq, seq)
            # Sum attention received by each token
            token_importance = attn_weights.sum(dim=0).cpu().numpy()

        elif importance_method == 'gradient':
            # Use gradient-based attribution (input x gradient)
            inputs['input_ids'].requires_grad = False
            embeddings = model.embeddings(inputs['input_ids'])
            embeddings.retain_grad()

            outputs = model(inputs_embeds=embeddings, attention_mask=inputs['attention_mask'])

            # Use mean of last hidden state as scalar output
            output = outputs.last_hidden_state.mean()
            output.backward()

            # Input x gradient
            token_importance = (embeddings * embeddings.grad).sum(dim=-1).abs()
            token_importance = token_importance[0].detach().cpu().numpy()
        else:
            raise ValueError(f"Unknown importance_method: {importance_method}")

        # Normalize
        token_importance = token_importance / (token_importance.sum() + 1e-10)

        # Get top tokens (excluding special tokens)
        token_scores = list(zip(tokens, token_importance))
        # Filter out special tokens
        filtered = [(t, s) for t, s in token_scores if t not in ['<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '[PAD]']]
        filtered.sort(key=lambda x: x[1], reverse=True)

        results.append({
            'smiles': smiles,
            'tokens': tokens,
            'importance': token_importance,
            'top_tokens': filtered[:top_k]
        })

    return results


# =============================================================================
# SUBSTRUCTURE SHAP
# =============================================================================

# Common functional groups for chemistry
DEFAULT_FUNCTIONAL_GROUPS = {
    'hydroxyl': '[OX2H]',
    'carbonyl': '[CX3]=[OX1]',
    'carboxyl': '[CX3](=O)[OX2H1]',
    'amine_primary': '[NX3;H2;!$(NC=O)]',
    'amine_secondary': '[NX3;H1;!$(NC=O)]',
    'amine_tertiary': '[NX3;H0;!$(NC=O)]',
    'amide': '[NX3][CX3](=[OX1])',
    'nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])]',
    'sulfonyl': '[SX4](=[OX1])(=[OX1])',
    'sulfhydryl': '[SX2H]',
    'ether': '[OX2]([#6])[#6]',
    'ester': '[CX3](=O)[OX2][#6]',
    'halogen_f': '[F]',
    'halogen_cl': '[Cl]',
    'halogen_br': '[Br]',
    'halogen_i': '[I]',
    'aromatic_ring': 'c1ccccc1',
    'heteroaromatic_n': '[nR1]',
    'heteroaromatic_o': '[oR1]',
    'heteroaromatic_s': '[sR1]',
    'alkene': '[CX3]=[CX3]',
    'alkyne': '[CX2]#[CX2]',
    'nitrile': '[CX2]#[NX1]',
    'phosphate': '[PX4](=O)([OX2])([OX2])[OX2]',
}


def compute_substructure_features(smiles_list, functional_groups=None):
    """
    Compute binary features indicating presence of functional groups.

    Args:
        smiles_list: List of SMILES strings
        functional_groups: Dict of {name: SMARTS} or None for defaults

    Returns:
        tuple: (features, group_names)
            - features: np.ndarray (n_molecules, n_groups) binary matrix
            - group_names: List of functional group names
    """
    try:
        from rdkit import Chem
    except ImportError:
        raise ImportError("Substructure features require RDKit")

    if functional_groups is None:
        functional_groups = DEFAULT_FUNCTIONAL_GROUPS

    group_names = list(functional_groups.keys())
    patterns = {name: Chem.MolFromSmarts(smarts) for name, smarts in functional_groups.items()}

    # Check for invalid SMARTS
    for name, pattern in patterns.items():
        if pattern is None:
            raise ValueError(f"Invalid SMARTS for '{name}': {functional_groups[name]}")

    features = np.zeros((len(smiles_list), len(group_names)), dtype=np.float32)

    for mol_idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        for group_idx, name in enumerate(group_names):
            if mol.HasSubstructMatch(patterns[name]):
                features[mol_idx, group_idx] = 1.0

    return features, group_names


def substructure_shap(smiles_list, labels, functional_groups=None,
                     background_samples=100, nsamples='auto', random_state=42):
    """
    Compute SHAP values for predefined functional groups.

    Instead of computing importance for individual features, this method
    groups molecular features by functional group and computes SHAP values
    at the group level, providing chemically interpretable importance.

    Args:
        smiles_list: List of SMILES strings
        labels: Target values (for training the model)
        functional_groups: Dict of {name: SMARTS} or None for defaults
        background_samples: Number of background samples for KernelSHAP
        nsamples: Number of SHAP samples ('auto' or int)
        random_state: Random seed

    Returns:
        dict: Results with keys:
            - 'group_names': List of functional group names
            - 'shap_values': np.ndarray (n_molecules, n_groups) SHAP values
            - 'mean_importance': np.ndarray (n_groups,) mean |SHAP| per group
            - 'top_groups': List of (name, importance) sorted by importance
    """
    try:
        import shap
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("Substructure SHAP requires shap and sklearn")

    # Compute substructure features
    features, group_names = compute_substructure_features(smiles_list, functional_groups)

    # Check if we have any non-zero features
    if features.sum() == 0:
        raise ValueError("No functional groups found in any molecule")

    labels = np.asarray(labels)

    # Determine if classification or regression
    if np.issubdtype(labels.dtype, np.integer) or (
        np.issubdtype(labels.dtype, np.floating) and
        np.allclose(labels, labels.astype(int)) and
        len(np.unique(labels)) < 20
    ):
        is_classification = True
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    else:
        is_classification = False
        model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)

    # Train model on substructure features
    model.fit(features, labels)

    # Set up SHAP explainer
    n_bg = min(background_samples, features.shape[0])
    np.random.seed(random_state)
    bg_idx = np.random.choice(features.shape[0], n_bg, replace=False)
    background = features[bg_idx]

    if is_classification:
        predict_fn = lambda x: model.predict_proba(x)[:, 1] if len(model.classes_) == 2 else model.predict_proba(x)
    else:
        predict_fn = model.predict

    explainer = shap.KernelExplainer(predict_fn, background)

    # Compute SHAP values
    shap_values = explainer.shap_values(features, nsamples=nsamples, silent=True)

    # Handle multiclass
    if isinstance(shap_values, list):
        shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)

    # Compute mean importance per group
    mean_importance = np.mean(np.abs(shap_values), axis=0)

    # Sort groups by importance
    sorted_idx = np.argsort(mean_importance)[::-1]
    top_groups = [(group_names[i], float(mean_importance[i])) for i in sorted_idx]

    return {
        'group_names': group_names,
        'shap_values': shap_values,
        'mean_importance': mean_importance,
        'top_groups': top_groups,
        'features': features  # Include for reference
    }


def count_substructure_occurrences(smiles_list, functional_groups=None):
    """
    Count occurrences of each functional group in each molecule.

    Unlike compute_substructure_features which returns binary presence,
    this returns the count of each group.

    Args:
        smiles_list: List of SMILES strings
        functional_groups: Dict of {name: SMARTS} or None for defaults

    Returns:
        tuple: (counts, group_names)
            - counts: np.ndarray (n_molecules, n_groups) count matrix
            - group_names: List of functional group names
    """
    try:
        from rdkit import Chem
    except ImportError:
        raise ImportError("Substructure counting requires RDKit")

    if functional_groups is None:
        functional_groups = DEFAULT_FUNCTIONAL_GROUPS

    group_names = list(functional_groups.keys())
    patterns = {name: Chem.MolFromSmarts(smarts) for name, smarts in functional_groups.items()}

    counts = np.zeros((len(smiles_list), len(group_names)), dtype=np.float32)

    for mol_idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        for group_idx, name in enumerate(group_names):
            matches = mol.GetSubstructMatches(patterns[name])
            counts[mol_idx, group_idx] = len(matches)

    return counts, group_names
