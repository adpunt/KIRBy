"""
Molecular Representation Generators

This module provides both static and learned molecular representations for
the KIRBy (Knowledge Integration by Representation Borrowing) framework.

WORKFLOW:
1. Generate/fine-tune representations on training data
2. Extract embeddings for train/test sets
3. Combine into hybrid representations
4. Train final model

Available Representations:
=========================

STATIC (No Training):
- ECFP4: Extended-connectivity fingerprints
- PDV: Physical descriptor vectors

PRETRAINED (Can be used frozen or fine-tuned):
- mol2vec: Word2vec on molecular substructures
- MHG-GNN: Pretrained GNN encoder

STANDALONE LEARNED:
- Graph Kernel: Deterministic graph similarity features

FINE-TUNABLE:
- MoLFormer: Transformer on SMILES/SELFIES (fine-tune on YOUR data)
- ChemBERTa: BERT-style on SMILES/SELFIES (fine-tune on YOUR data)
- GNN: Graph neural networks (train on YOUR data)

GAUSSIAN PROCESSES:
- GAUCHE GP: Train GP with graph kernels

HYBRID:
- create_hybrid: Combine multiple representations
"""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_DESCRIPTOR_LIST = [
    'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v',
    'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'EState_VSA1', 'EState_VSA10',
    'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6',
    'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2',
    'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 'HeavyAtomMolWt',
    'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'MaxAbsEStateIndex', 'MaxAbsPartialCharge',
    'MaxEStateIndex', 'MaxPartialCharge', 'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex',
    'MinPartialCharge', 'MolLogP', 'MolMR', 'MolWt', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles',
    'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
    'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRadicalElectrons',
    'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings',
    'NumValenceElectrons', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13',
    'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7',
    'PEOE_VSA8', 'PEOE_VSA9', 'RingCount', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4',
    'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11',
    'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7',
    'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3',
    'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'fr_Al_COO',
    'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO',
    'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2',
    'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde',
    'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline',
    'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic',
    'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido',
    'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan',
    'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile',
    'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime',
    'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester',
    'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide',
    'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan',
    'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'qed'
]


# =============================================================================
# STATIC REPRESENTATIONS
# =============================================================================

def create_ecfp4(smiles_list, radius=2, n_bits=2048):
    """
    Create ECFP4 fingerprints (static, no training).
    
    Args:
        smiles_list: List of SMILES strings
        radius: Fingerprint radius (default: 2 for ECFP4)
        n_bits: Number of bits (default: 2048)
        
    Returns:
        np.ndarray: Binary fingerprint matrix (n_molecules, n_bits)
    """
    fingerprints = []
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = generator.GetFingerprint(mol)
            fingerprints.append(np.array(fp))
        else:
            fingerprints.append(np.zeros(n_bits))
    
    return np.array(fingerprints, dtype=np.float32)


def create_pdv(smiles_list, descriptor_list=None):
    """
    Create Physical Descriptor Vectors (static, no training).
    
    Args:
        smiles_list: List of SMILES strings
        descriptor_list: List of RDKit descriptor names (default: 200 descriptors)
        
    Returns:
        np.ndarray: Descriptor matrix (n_molecules, n_descriptors)
    """
    if descriptor_list is None:
        descriptor_list = DEFAULT_DESCRIPTOR_LIST
    
    calc = MolecularDescriptorCalculator(descriptor_list)
    descriptors = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            desc = np.array(calc.CalcDescriptors(mol), dtype=np.float32)
            desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
            descriptors.append(desc)
        else:
            descriptors.append(np.zeros(len(descriptor_list), dtype=np.float32))
    
    return np.array(descriptors)

def create_sns(smiles_list, reference_featurizer=None, return_featurizer=False,
               max_radius=2, pharm_atom_invs=False, bond_invs=True,
               chirality=False, sub_counts=True, vec_dimension=1024):
    """
    Create Sort & Slice ECFP features (SNS).
    
    Sort & Slice is a learned pooling method that selects the most prevalent
    substructures from the training set. Must be fitted on training data first.
    
    Args:
        smiles_list: List of SMILES strings
        reference_featurizer: Featurizer fitted on training (for test set)
        return_featurizer: If True, return (features, featurizer) tuple
        max_radius: Morgan fingerprint radius (default: 2)
        pharm_atom_invs: Use pharmacophore atom invariants (default: False)
        bond_invs: Use bond types (default: True)
        chirality: Include chirality (default: False)
        sub_counts: Use substructure counts vs binary (default: True)
        vec_dimension: Output dimension (default: 1024)
        
    Returns:
        np.ndarray: SNS features (n_molecules, vec_dimension)
        or tuple: (features, featurizer) if return_featurizer=True
        
    Usage:
        # Fit on training data
        sns_train, featurizer = create_sns(train_smiles, return_featurizer=True)
        
        # Transform test data
        sns_test = create_sns(test_smiles, reference_featurizer=featurizer)
    """
    from rdkit.Chem import rdFingerprintGenerator
    
    if reference_featurizer is not None:
        # Transform mode - use provided featurizer
        features = _transform_sns(smiles_list, reference_featurizer)
        return features
    
    # Fit mode - create new featurizer
    print(f"Fitting SNS featurizer on {len(smiles_list)} molecules...")
    
    # Convert SMILES to mol objects
    mols = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mols.append(mol)
    
    # Setup Morgan generator
    if pharm_atom_invs:
        atom_inv_gen = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    else:
        atom_inv_gen = rdFingerprintGenerator.GetMorganAtomInvGen(includeRingMembership=True)
    
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=max_radius,
        atomInvariantsGenerator=atom_inv_gen,
        useBondTypes=bond_invs,
        includeChirality=chirality
    )
    
    # Function to enumerate substructure IDs and counts
    def sub_id_enumerator(mol):
        if mol is None:
            return {}
        fp = morgan_generator.GetSparseCountFingerprint(mol)
        return fp.GetNonzeroElements()
    
    # Count prevalence of each substructure across training set
    sub_ids_to_prevs = {}
    for mol in mols:
        for sub_id in sub_id_enumerator(mol).keys():
            sub_ids_to_prevs[sub_id] = sub_ids_to_prevs.get(sub_id, 0) + 1
    
    # Sort substructures by prevalence (most common first)
    sub_ids_sorted = sorted(
        sub_ids_to_prevs.keys(),
        key=lambda sub_id: (sub_ids_to_prevs[sub_id], sub_id),
        reverse=True
    )
    
    # Keep only top vec_dimension substructures
    top_sub_ids = set(sub_ids_sorted[:vec_dimension])
    sub_id_to_index = {sub_id: i for i, sub_id in enumerate(sub_ids_sorted[:vec_dimension])}
    
    print(f"  Found {len(sub_ids_to_prevs)} unique substructures, keeping top {vec_dimension}")
    
    # Store featurizer parameters
    featurizer = {
        'morgan_generator': morgan_generator,
        'sub_id_to_index': sub_id_to_index,
        'top_sub_ids': top_sub_ids,
        'vec_dimension': vec_dimension,
        'sub_counts': sub_counts
    }
    
    # Transform training data
    features = _transform_sns(smiles_list, featurizer)
    
    if return_featurizer:
        return features, featurizer
    return features


def _transform_sns(smiles_list, featurizer):
    """Transform SMILES using fitted SNS featurizer"""
    morgan_generator = featurizer['morgan_generator']
    sub_id_to_index = featurizer['sub_id_to_index']
    top_sub_ids = featurizer['top_sub_ids']
    vec_dimension = featurizer['vec_dimension']
    sub_counts = featurizer['sub_counts']
    
    features = np.zeros((len(smiles_list), vec_dimension), dtype=np.float32)
    
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        # Get substructure counts
        fp = morgan_generator.GetSparseCountFingerprint(mol)
        sub_id_counts = fp.GetNonzeroElements()
        
        # Build feature vector
        for sub_id, count in sub_id_counts.items():
            if sub_id in top_sub_ids:
                idx = sub_id_to_index[sub_id]
                if sub_counts:
                    features[i, idx] = count
                else:
                    features[i, idx] = 1
    
    return features

# =============================================================================
# PRETRAINED REPRESENTATIONS
# =============================================================================

# Global cache for mol2vec
_MOL2VEC_MODEL = None

def create_mol2vec(smiles_list, model_path=None, radius=1, unseen='UNK'):
    """
    Create mol2vec embeddings (pretrained word2vec on molecular substructures).
    
    Mol2vec learns vector representations of molecular substructures using
    the word2vec algorithm. This is a pretrained model that can be used frozen.
    
    Args:
        smiles_list: List of SMILES strings
        model_path: Path to mol2vec model (default: searches common locations)
        radius: Radius for Morgan substructures (default: 1)
        unseen: How to handle unseen substructures ('UNK', 'zero', or 'mean')
        
    Returns:
        np.ndarray: Mol2vec embeddings (n_molecules, 300)
    """
    try:
        from mol2vec.features import mol2alt_sentence, MolSentence
        from gensim.models import word2vec
    except ImportError:
        raise ImportError("mol2vec requires: pip install mol2vec gensim")
    
    import os
    
    global _MOL2VEC_MODEL
    
    # Load model once
    if _MOL2VEC_MODEL is None:
        if model_path is None:
            # Search common locations
            search_paths = [
                'model_300dim.pkl',  # Current directory
                '../model_300dim.pkl',  # Parent directory
                'tests/model_300dim.pkl',  # tests subdirectory
                '../tests/model_300dim.pkl',  # tests in parent
                os.path.expanduser('~/model_300dim.pkl'),  # Home directory
            ]
            
            found_path = None
            for path in search_paths:
                if os.path.exists(path):
                    found_path = path
                    break
            
            if found_path is None:
                raise RuntimeError(
                    f"Could not find mol2vec model in any of: {search_paths}\n"
                    f"Download from: https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl"
                )
            
            print(f"Loading mol2vec model from {found_path}...")
            _MOL2VEC_MODEL = word2vec.Word2Vec.load(found_path)
        else:
            _MOL2VEC_MODEL = word2vec.Word2Vec.load(model_path)
    
    # Convert SMILES to molecules and sentences
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    sentences = [MolSentence(mol2alt_sentence(m, radius)) for m in mols if m is not None]
    
    # Manual embedding generation (Gensim 4.x compatible)
    embeddings = []
    for sentence in sentences:
        vec = np.zeros(_MOL2VEC_MODEL.wv.vector_size)
        count = 0
        
        for word in sentence:
            if word in _MOL2VEC_MODEL.wv:
                vec += _MOL2VEC_MODEL.wv[word]
                count += 1
        
        if count > 0:
            vec = vec / count
        
        embeddings.append(vec)
    
    return np.array(embeddings, dtype=np.float32)


# Global cache for MHG-GNN
_MHG_GNN_MODEL = None

def create_mhg_gnn(smiles_list, n_features=None, batch_size=32, 
                   materials_repo_path=None, model_pickle_path=None):
    """
    Create MHG-GNN embeddings (pretrained GNN encoder from IBM).

    This implementation relies on a locally available copy of the MHG-GNN
    model code and a separately downloaded pretrained weight file.

    CODE REQUIREMENT:
    A directory containing:
        models/mhg_model/
    This directory is auto-detected from common locations, e.g.:
        ~/repos/materials
        ~/materials
        ../materials

    WEIGHTS REQUIREMENT:
    The pretrained weight file
        mhggnn_pretrained_model_0724_2023.pickle
    must be downloaded separately (e.g. from Hugging Face
    ibm-research/materials.mhg-ged) and placed at one of the searched paths
    or provided explicitly via `model_pickle_path`.

    This function does NOT automatically download code or weights.

    Args:
        smiles_list: List of SMILES strings
        n_features: Optional PCA output dimension
        batch_size: Batch size for encoding
        materials_repo_path: Path to directory containing models/mhg_model
        model_pickle_path: Path to pretrained pickle file

    Returns:
        np.ndarray: MHG-GNN embeddings
    """
    import sys
    import os
    import torch
    
    global _MHG_GNN_MODEL
    
    if _MHG_GNN_MODEL is None:
        # Find materials repo
        if materials_repo_path is None:
            search_paths = [
                os.path.expanduser('~/repos/materials'),
                os.path.expanduser('~/materials'),
                '../materials',
                '../../materials',
            ]
            for path in search_paths:
                if os.path.exists(os.path.join(path, 'models', 'mhg_model')):
                    materials_repo_path = path
                    break
            
            if materials_repo_path is None:
                raise RuntimeError(
                    f"Could not find a local MHG-GNN code directory. Searched: {search_paths}\n"
                    f"Expected to find: <path>/models/mhg_model\n"
                    f"Fix: place the MHG code at one of the searched locations (e.g. ~/repos/materials)\n"
                    f"or pass materials_repo_path='/path/to/dir/that/contains/models/mhg_model'."
                )
        
        # Add to path
        models_path = os.path.join(materials_repo_path, 'models')
        if models_path not in sys.path:
            sys.path.insert(0, models_path)
        
        print(f"Loading MHG-GNN from {materials_repo_path}...")
        
        # Import after adding to path
        from mhg_model.load import PretrainedModelWrapper
        
        # Find model pickle
        if model_pickle_path is None:
            search_paths = [
                os.path.expanduser('~/repos/materials.mhg-ged/mhggnn_pretrained_model_0724_2023.pickle'),
                os.path.expanduser('~/materials.mhg-ged/mhggnn_pretrained_model_0724_2023.pickle'),
                '../materials.mhg-ged/mhggnn_pretrained_model_0724_2023.pickle',
            ]
            for path in search_paths:
                if os.path.exists(path):
                    model_pickle_path = path
                    break
            
            if model_pickle_path is None:
                raise RuntimeError(
                    f"Could not find model pickle. Searched: {search_paths}\n"
                    f"Download from: https://huggingface.co/ibm-research/materials.mhg-ged"
                )
        
        # Load pickle
        import pickle
        with open(model_pickle_path, 'rb') as f:
            model_dict = pickle.load(f)
        
        # Wrap with PretrainedModelWrapper
        _MHG_GNN_MODEL = PretrainedModelWrapper(model_dict)
        _MHG_GNN_MODEL.model.eval()
        print(f"MHG-GNN model loaded successfully")
    
    # Encode SMILES
    print(f"Encoding {len(smiles_list)} molecules with MHG-GNN...")
    
    all_embeddings = []
    
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        
        # encode() returns list of tensors
        batch_embeddings = _MHG_GNN_MODEL.encode(batch)
        
        # Convert to numpy and stack
        batch_np = torch.stack(batch_embeddings).cpu().detach().numpy()
        all_embeddings.append(batch_np)
    
    embeddings = np.vstack(all_embeddings)
    
    # Optionally reduce dimensionality
    if n_features is not None and n_features < embeddings.shape[1]:
        # Check if we have enough samples for PCA
        if embeddings.shape[0] < n_features:
            print(f"WARNING: Cannot reduce to {n_features} dims with only {embeddings.shape[0]} samples.")
            print(f"         Returning full {embeddings.shape[1]}-dim embeddings instead.")
            print(f"         Fit PCA on training set first, then transform test set using same PCA.")
        else:
            from sklearn.decomposition import PCA
            print(f"Reducing from {embeddings.shape[1]} to {n_features} dims via PCA...")
            pca = PCA(n_components=n_features)
            embeddings = pca.fit_transform(embeddings)
    
    return embeddings


def create_graph_kernel(smiles_list, kernel='weisfeiler_lehman', n_iter=5, 
                        n_features=None, reference_vocabulary=None, return_vocabulary=False):
    """
    Create graph kernel features using WL histogram approach.
    
    Extracts WL label histograms as features - the proper way to use graph kernels
    with non-kernel methods like Random Forest.
    
    Args:
        smiles_list: List of SMILES strings
        kernel: 'weisfeiler_lehman' (others not properly implemented for histograms)
        n_iter: Number of WL iterations (default: 5)
        n_features: Ignored (kept for API compatibility)
        reference_vocabulary: Set of WL labels from training (for test set)
        return_vocabulary: If True, return (features, vocabulary) tuple
        
    Returns:
        np.ndarray: WL histogram features (n_molecules, n_labels)
        or tuple: (features, vocabulary) if return_vocabulary=True
    """
    from collections import Counter
    
    print(f"Converting {len(smiles_list)} molecules to graphs...")
    
    # Convert SMILES to molecular graphs
    graphs_data = []
    failed = 0
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumBonds() == 0:
            graphs_data.append(None)
            failed += 1
            continue
        
        # Extract graph structure
        node_labels = {atom.GetIdx(): atom.GetSymbol() for atom in mol.GetAtoms()}
        edges = {atom.GetIdx(): [] for atom in mol.GetAtoms()}
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges[i].append(j)
            edges[j].append(i)
        
        graphs_data.append((node_labels, edges))
    
    if failed > 0:
        print(f"  Warning: {failed}/{len(smiles_list)} molecules failed")
    
    print(f"Computing WL histograms with {n_iter} iterations...")
    
    # Run WL algorithm and collect histograms
    all_histograms = []
    discovered_labels = set()
    
    for graph_data in graphs_data:
        if graph_data is None:
            all_histograms.append(None)
            continue
        
        node_labels, edges = graph_data
        current_labels = node_labels.copy()
        histogram = Counter()
        
        # Collect labels from all iterations
        for iteration in range(n_iter + 1):
            # Add current labels to histogram
            for label in current_labels.values():
                label_str = f"{iteration}_{label}"
                histogram[label_str] += 1
                # Only add to discovered labels if we're fitting (not transforming)
                if reference_vocabulary is None:
                    discovered_labels.add(label_str)
            
            # WL relabeling
            if iteration < n_iter:
                new_labels = {}
                for node, label in current_labels.items():
                    # Get sorted neighbor labels
                    neighbor_labels = sorted([current_labels[n] for n in edges[node]])
                    # New label is combination of current + neighbors
                    new_label = f"{label}_{'_'.join(neighbor_labels)}"
                    new_labels[node] = new_label
                current_labels = new_labels
        
        all_histograms.append(histogram)
    
    # Determine which vocabulary to use
    if reference_vocabulary is not None:
        # Transforming test data - use reference vocabulary
        vocabulary = reference_vocabulary
        print(f"  Using {len(vocabulary)} reference WL labels (from training)")
    else:
        # Fitting on training data - use discovered labels
        vocabulary = discovered_labels
        print(f"  Extracted {len(vocabulary)} unique WL labels as features")
    
    # Convert to feature matrix using vocabulary
    sorted_labels = sorted(vocabulary)
    features = np.zeros((len(smiles_list), len(sorted_labels)), dtype=np.float32)
    
    for i, histogram in enumerate(all_histograms):
        if histogram is not None:
            for j, label in enumerate(sorted_labels):
                features[i, j] = histogram.get(label, 0)
    
    if return_vocabulary:
        return features, vocabulary
    return features

# =============================================================================
# FINE-TUNING: MoLFormer
# =============================================================================

def finetune_molformer(train_smiles, train_labels, val_smiles, val_labels,
                       epochs=10, batch_size=32, learning_rate=1e-4,
                       embedding_dim=768, freeze_encoder=False, use_selfies=False):
    """
    Fine-tune MoLFormer on YOUR task-specific data.
    
    Supports both SMILES and SELFIES input. The model learns task-specific
    patterns and the embeddings become optimized for YOUR property.
    
    Args:
        train_smiles: Training SMILES
        train_labels: Training labels
        val_smiles: Validation SMILES
        val_labels: Validation labels
        epochs: Number of training epochs (default: 10)
        batch_size: Batch size (default: 32)
        learning_rate: Learning rate (default: 1e-4)
        embedding_dim: Embedding dimension (default: 768)
        freeze_encoder: If True, only train head (default: False)
        use_selfies: If True, use SELFIES instead of SMILES (default: False)
        
    Returns:
        dict: {
            'model': Fine-tuned model,
            'tokenizer': Tokenizer,
            'embedding_dim': Dimension,
            'use_selfies': Whether SELFIES is used,
            'device': Device
        }
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        from transformers import AutoModel, AutoTokenizer
        if use_selfies:
            import selfies as sf
    except ImportError:
        deps = "torch transformers" + (" selfies" if use_selfies else "")
        raise ImportError(f"MoLFormer fine-tuning requires: pip install {deps}")
    
    # Convert to SELFIES if requested
    if use_selfies:
        print("Converting SMILES to SELFIES...")
        train_input = [sf.encoder(s) for s in train_smiles]
        val_input = [sf.encoder(s) for s in val_smiles]
    else:
        train_input = train_smiles
        val_input = val_smiles
    
    print("Loading pretrained MoLFormer...")
    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    base_model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    
    # Regression model
    class MoLFormerRegressor(nn.Module):
        def __init__(self, base_model, embedding_dim=768, freeze_encoder=False):
            super().__init__()
            self.encoder = base_model
            self.regression_head = nn.Linear(embedding_dim, 1)
            
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        
        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return self.regression_head(embeddings), embeddings
    
    model = MoLFormerRegressor(base_model, embedding_dim, freeze_encoder)
    
    # Dataset
    class MoleculeDataset(Dataset):
        def __init__(self, inputs, labels, tokenizer):
            self.inputs = inputs
            self.labels = torch.tensor(labels, dtype=torch.float32)
            self.tokenizer = tokenizer
        
        def __len__(self):
            return len(self.inputs)
        
        def __getitem__(self, idx):
            encoded = self.tokenizer(
                self.inputs[idx],
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            return {
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'labels': self.labels[idx]
            }
    
    train_dataset = MoleculeDataset(train_input, train_labels, tokenizer)
    val_dataset = MoleculeDataset(val_input, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    input_type = "SELFIES" if use_selfies else "SMILES"
    print(f"Fine-tuning MoLFormer on {input_type} for {epochs} epochs on {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            predictions, _ = model(input_ids, attention_mask)
            loss = criterion(predictions.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                predictions, _ = model(input_ids, attention_mask)
                loss = criterion(predictions.squeeze(), labels)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}")
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'embedding_dim': embedding_dim,
        'use_selfies': use_selfies,
        'device': device
    }


# =============================================================================
# FINE-TUNING: ChemBERTa
# =============================================================================

def finetune_chemberta(train_smiles, train_labels, val_smiles, val_labels,
                       epochs=10, batch_size=32, learning_rate=1e-4,
                       embedding_dim=768, freeze_encoder=False, use_selfies=False):
    """
    Fine-tune ChemBERTa on YOUR task-specific data.
    
    Supports both SMILES and SELFIES input. Similar to MoLFormer but uses
    BERT-style architecture.
    
    Args:
        train_smiles: Training SMILES
        train_labels: Training labels
        val_smiles: Validation SMILES
        val_labels: Validation labels
        epochs: Number of training epochs (default: 10)
        batch_size: Batch size (default: 32)
        learning_rate: Learning rate (default: 1e-4)
        embedding_dim: Embedding dimension (default: 768)
        freeze_encoder: If True, only train head (default: False)
        use_selfies: If True, use SELFIES instead of SMILES (default: False)
        
    Returns:
        dict: {'model', 'tokenizer', 'embedding_dim', 'use_selfies', 'device'}
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        from transformers import AutoModel, AutoTokenizer
        if use_selfies:
            import selfies as sf
    except ImportError:
        deps = "torch transformers" + (" selfies" if use_selfies else "")
        raise ImportError(f"ChemBERTa fine-tuning requires: pip install {deps}")
    
    # Convert to SELFIES if requested
    if use_selfies:
        print("Converting SMILES to SELFIES...")
        train_input = [sf.encoder(s) for s in train_smiles]
        val_input = [sf.encoder(s) for s in val_smiles]
    else:
        train_input = train_smiles
        val_input = val_smiles
    
    print("Loading pretrained ChemBERTa...")
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    base_model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    
    # Regression model
    class ChemBERTaRegressor(nn.Module):
        def __init__(self, base_model, embedding_dim=768, freeze_encoder=False):
            super().__init__()
            self.encoder = base_model
            self.regression_head = nn.Linear(embedding_dim, 1)
            
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        
        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return self.regression_head(embeddings), embeddings
    
    model = ChemBERTaRegressor(base_model, embedding_dim, freeze_encoder)
    
    # Dataset
    class MoleculeDataset(Dataset):
        def __init__(self, inputs, labels, tokenizer):
            self.inputs = inputs
            self.labels = torch.tensor(labels, dtype=torch.float32)
            self.tokenizer = tokenizer
        
        def __len__(self):
            return len(self.inputs)
        
        def __getitem__(self, idx):
            encoded = self.tokenizer(
                self.inputs[idx],
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            return {
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'labels': self.labels[idx]
            }
    
    train_dataset = MoleculeDataset(train_input, train_labels, tokenizer)
    val_dataset = MoleculeDataset(val_input, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    input_type = "SELFIES" if use_selfies else "SMILES"
    print(f"Fine-tuning ChemBERTa on {input_type} for {epochs} epochs on {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            predictions, _ = model(input_ids, attention_mask)
            loss = criterion(predictions.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                predictions, _ = model(input_ids, attention_mask)
                loss = criterion(predictions.squeeze(), labels)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}")
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'embedding_dim': embedding_dim,
        'use_selfies': use_selfies,
        'device': device
    }


# =============================================================================
# FINE-TUNING: Graph Neural Networks
# =============================================================================

def finetune_gnn(train_smiles, train_labels, val_smiles, val_labels,
                 gnn_type='gcn', hidden_dim=128, num_layers=3, epochs=100,
                 batch_size=32, learning_rate=1e-3):
    """
    Fine-tune a GNN (GCN/GAT/MPNN) on YOUR task-specific data.
    
    Trains GNN from scratch (or pretrained if available) on YOUR data.
    
    Args:
        train_smiles: Training SMILES
        train_labels: Training labels
        val_smiles: Validation SMILES
        val_labels: Validation labels
        gnn_type: 'gcn', 'gat', or 'mpnn' (default: 'gcn')
        hidden_dim: Hidden dimension (default: 128)
        num_layers: Number of GNN layers (default: 3)
        epochs: Number of training epochs (default: 100)
        batch_size: Batch size (default: 32)
        learning_rate: Learning rate (default: 1e-3)
        
    Returns:
        dict: {'model', 'gnn_type', 'hidden_dim', 'device'}
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.data import Data, Batch
        from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        raise ImportError("GNN fine-tuning requires: pip install torch torch-geometric")
    
    # Convert SMILES to PyG graphs
    def smiles_to_graph(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append([
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetIsAromatic())
            ])
        
        x = torch.tensor(atom_features, dtype=torch.float)
        
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros((2, 0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)
    
    print(f"Converting SMILES to graphs...")
    train_graphs = [smiles_to_graph(s) for s in train_smiles]
    val_graphs = [smiles_to_graph(s) for s in val_smiles]
    
    train_data = [(g, l) for g, l in zip(train_graphs, train_labels) if g is not None]
    val_data = [(g, l) for g, l in zip(val_graphs, val_labels) if g is not None]
    
    def collate_fn(batch):
        graphs, labels = zip(*batch)
        batched_graph = Batch.from_data_list(graphs)
        labels = torch.tensor(labels, dtype=torch.float32)
        return batched_graph, labels
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn)
    
    # GNN model
    num_node_features = 4
    
    class GNNRegressor(nn.Module):
        def __init__(self, gnn_type, num_features, hidden_dim, num_layers):
            super().__init__()
            self.gnn_type = gnn_type
            self.convs = nn.ModuleList()
            
            if gnn_type == 'gcn':
                self.convs.append(GCNConv(num_features, hidden_dim))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(num_features, hidden_dim, heads=4, concat=True))
                hidden_dim = hidden_dim * 4
            
            for _ in range(num_layers - 1):
                if gnn_type == 'gcn':
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                elif gnn_type == 'gat':
                    self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1))
            
            self.regression_head = nn.Linear(hidden_dim, 1)
        
        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=0.1, training=self.training)
            
            x = global_mean_pool(x, batch)
            return self.regression_head(x), x
    
    model = GNNRegressor(gnn_type, num_node_features, hidden_dim, num_layers)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Fine-tuning {gnn_type.upper()} for {epochs} epochs on {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            predictions, _ = model(data)
            loss = criterion(predictions.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, labels in val_loader:
                    data = data.to(device)
                    labels = labels.to(device)
                    
                    predictions, _ = model(data)
                    loss = criterion(predictions.squeeze(), labels)
                    val_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss/len(val_loader):.4f}")
    
    return {
        'model': model,
        'gnn_type': gnn_type,
        'hidden_dim': hidden_dim,
        'device': device
    }


# =============================================================================
# ENCODING: Extract embeddings from fine-tuned models
# =============================================================================

def encode_from_model(finetuned_model_dict, smiles_list, batch_size=32):
    """
    Extract task-specific embeddings from a fine-tuned model.
    
    Works with models returned from finetune_molformer(), finetune_chemberta(),
    or finetune_gnn().
    
    Args:
        finetuned_model_dict: Dictionary from fine-tuning function
        smiles_list: List of SMILES to encode
        batch_size: Batch size (default: 32)
        
    Returns:
        np.ndarray: Task-specific embeddings (n_molecules, embedding_dim)
    """
    import torch
    
    model = finetuned_model_dict['model']
    device = finetuned_model_dict['device']
    model.eval()
    
    all_embeddings = []
    
    # Transformer model
    if 'tokenizer' in finetuned_model_dict:
        tokenizer = finetuned_model_dict['tokenizer']
        use_selfies = finetuned_model_dict.get('use_selfies', False)
        
        # Convert to SELFIES if needed
        if use_selfies:
            import selfies as sf
            input_list = [sf.encoder(s) for s in smiles_list]
        else:
            input_list = smiles_list
        
        for i in range(0, len(input_list), batch_size):
            batch = input_list[i:i + batch_size]
            
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            with torch.no_grad():
                _, embeddings = model(input_ids, attention_mask)
                all_embeddings.append(embeddings.cpu().numpy())
    
    # GNN model
    elif 'gnn_type' in finetuned_model_dict:
        from torch_geometric.data import Data, Batch
        
        def smiles_to_graph(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            atom_features = []
            for atom in mol.GetAtoms():
                atom_features.append([
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    int(atom.GetIsAromatic())
                ])
            
            x = torch.tensor(atom_features, dtype=torch.float)
            
            edge_index = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_index.append([i, j])
                edge_index.append([j, i])
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros((2, 0), dtype=torch.long)
            
            return Data(x=x, edge_index=edge_index)
        
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i + batch_size]
            graphs = [smiles_to_graph(s) for s in batch_smiles]
            
            valid_graphs = [g for g in graphs if g is not None]
            
            if valid_graphs:
                batched = Batch.from_data_list(valid_graphs)
                batched = batched.to(device)
                
                with torch.no_grad():
                    _, embeddings = model(batched)
                    all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


# =============================================================================
# GAUCHE: Train Gaussian Process with graph kernels
# =============================================================================
def train_gauche_gp(train_smiles, train_labels, kernel='weisfeiler_lehman',
                    n_iter=5, num_epochs=100, learning_rate=0.1):
    """
    Train a Gaussian Process with graph kernel on YOUR data.
    
    This trains a GP directly on molecular graphs (not feature extraction).
    
    Args:
        train_smiles: Training SMILES
        train_labels: Training labels
        kernel: 'weisfeiler_lehman', 'shortest_path', or 'subtree'
        n_iter: WL iterations (default: 5)
        num_epochs: GP training epochs (default: 100)
        learning_rate: Learning rate (default: 0.1)
        
    Returns:
        dict: {'gp_model', 'likelihood', 'kernel_obj', 'train_graphs', 'train_labels'}
    """
    try:
        import torch
        import gpytorch
        from grakel.kernels import WeisfeilerLehman
        from grakel import Graph
    except ImportError:
        raise ImportError("GAUCHE GP requires: pip install gauche[graphs] gpytorch")
    
    print(f"Converting {len(train_smiles)} molecules to graphs...")
    graphs = []
    for smiles in train_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            graphs.append(None)
            continue
        
        # Build adjacency dictionary for grakel
        node_labels = {}
        edges = {}
        
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            node_labels[idx] = atom.GetSymbol()
            edges[idx] = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges[i].append(j)
            edges[j].append(i)
        
        g = Graph(edges, node_labels=node_labels)
        graphs.append(g)
    
    valid_indices = [i for i, g in enumerate(graphs) if g is not None]
    valid_graphs = [graphs[i] for i in valid_indices]
    valid_labels = torch.tensor([train_labels[i] for i in valid_indices], dtype=torch.float32)
    
    print(f"Computing kernel matrix for {len(valid_graphs)} valid molecules...")
    
    # Initialize graph kernel
    if kernel == 'weisfeiler_lehman':
        kernel_obj = WeisfeilerLehman(n_iter=n_iter, normalize=True)
    else:
        raise ValueError(f"Kernel '{kernel}' not implemented. Use 'weisfeiler_lehman'")
    
    # Compute kernel matrix (this is what grakel does)
    K_train = kernel_obj.fit_transform(valid_graphs)
    K_train_tensor = torch.tensor(K_train, dtype=torch.float32)
    
    # Add jitter for numerical stability
    jitter = 1e-4
    K_train_tensor = K_train_tensor + jitter * torch.eye(K_train_tensor.shape[0])
    
    print(f"Training GP on kernel matrix of shape {K_train_tensor.shape}...")
    
    # Create index tensor for GPyTorch
    train_indices = torch.arange(len(valid_graphs), dtype=torch.float32).unsqueeze(-1)
    
    class PrecomputedKernelGP(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, K_train):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            # Store precomputed kernel matrix
            self.register_buffer('K_train', K_train)
            # Scale kernel for the precomputed matrix
            self.outputscale = torch.nn.Parameter(torch.ones(1))
        
        def forward(self, x):
            mean_x = self.mean_module(x)
            # Use precomputed kernel matrix scaled
            covar_x = self.outputscale * self.K_train
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = PrecomputedKernelGP(train_indices, valid_labels, likelihood, K_train_tensor)
    
    gp_model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(gp_model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
    
    print(f"Training GP for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = gp_model(train_indices)
        loss = -mll(output, valid_labels)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")
    
    return {
        'gp_model': gp_model,
        'likelihood': likelihood,
        'kernel_obj': kernel_obj,  # Save this for test predictions
        'train_graphs': valid_graphs,
        'train_labels': valid_labels
    }


def predict_gauche_gp(gp_dict, test_smiles):
    """
    Make predictions with trained GAUCHE GP.
    
    Args:
        gp_dict: Dictionary from train_gauche_gp()
        test_smiles: Test SMILES
        
    Returns:
        dict: {'predictions': mean, 'uncertainties': std}
    """
    import torch
    from grakel import Graph
    
    gp_model = gp_dict['gp_model']
    likelihood = gp_dict['likelihood']
    kernel_obj = gp_dict['kernel_obj']
    train_graphs = gp_dict['train_graphs']
    
    print(f"Converting {len(test_smiles)} test molecules to graphs...")
    test_graphs = []
    for smiles in test_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            test_graphs.append(None)
            continue
        
        # Build adjacency dictionary for grakel
        node_labels = {}
        edges = {}
        
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            node_labels[idx] = atom.GetSymbol()
            edges[idx] = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges[i].append(j)
            edges[j].append(i)
        
        g = Graph(edges, node_labels=node_labels)
        test_graphs.append(g)
    
    valid_test_graphs = [g for g in test_graphs if g is not None]
    
    print(f"Computing kernel matrix between {len(valid_test_graphs)} test and {len(train_graphs)} train graphs...")
    # Compute K(test, train)
    K_test_train = kernel_obj.transform(valid_test_graphs)
    K_test_train_tensor = torch.tensor(K_test_train, dtype=torch.float32)
    
    gp_model.eval()
    likelihood.eval()
    
    # Manual GP prediction with precomputed kernel
    with torch.no_grad():
        # Get training data
        train_labels = gp_dict['train_labels']
        K_train = gp_model.K_train
        noise_var = likelihood.noise.item()
        
        # K_train + noise*I
        K_train_noise = K_train + noise_var * torch.eye(K_train.shape[0])
        
        # Solve (K + noise*I)^{-1} @ y
        alpha = torch.linalg.solve(K_train_noise, train_labels)
        
        # Mean: K_test_train @ alpha
        mean = (K_test_train_tensor @ alpha).numpy()
        
        # For uncertainty, we'd need K_test_test which requires all test graphs upfront
        # For now, return constant uncertainty (simplification)
        std = torch.ones(len(valid_test_graphs)).numpy() * noise_var**0.5
    
    return {
        'predictions': mean,
        'uncertainties': std
    }

# =============================================================================
# HYBRID: Combine representations
# =============================================================================

def create_hybrid(embeddings_dict, selection_method='concat', n_features=None):
    """
    Combine task-specific embeddings from fine-tuned models.
    
    This is the CORE of KIRBy - combining embeddings optimized for YOUR task.
    
    Args:
        embeddings_dict: Dict of embeddings
            Example: {
                'ecfp4': ecfp4_train,
                'molformer': molformer_embeddings,
                'gnn': gnn_embeddings
            }
        selection_method: 'concat' or 'pca' (default: 'concat')
        n_features: For PCA, target dimensionality (default: None)
        
    Returns:
        np.ndarray: Hybrid representation tailored to your task
    """
    arrays = list(embeddings_dict.values())
    
    n_samples = arrays[0].shape[0]
    for arr in arrays:
        if arr.shape[0] != n_samples:
            raise ValueError(f"All embeddings must have same number of samples")
    
    if selection_method == 'concat':
        hybrid = np.hstack(arrays)
        
        if n_features is not None and n_features < hybrid.shape[1]:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_features)
            hybrid = pca.fit_transform(hybrid)
        
        return hybrid
    
    elif selection_method == 'pca':
        if n_features is None:
            raise ValueError("Must specify n_features for PCA")
        
        hybrid = np.hstack(arrays)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_features)
        return pca.fit_transform(hybrid)
    
    else:
        raise ValueError(f"Unknown selection_method: {selection_method}")