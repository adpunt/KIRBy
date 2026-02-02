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
- MACCS: MACCS keys fingerprints (167 bits)
- PDV: Physical descriptor vectors

PRETRAINED (Frozen):
- mol2vec: Word2vec on molecular substructures
- MHG-GNN: Pretrained GNN encoder
- ChemBERTa: BERT-style on SMILES (frozen extraction)
- MolFormer: Transformer on SMILES (frozen extraction)

STANDALONE LEARNED:
- Graph Kernel: Deterministic graph similarity features

FINE-TUNABLE:
- GNN: Graph neural networks (train on YOUR data)

GAUSSIAN PROCESSES:
- GAUCHE GP: Train GP with graph kernels

AUGMENTATIONS:
- Graph topology: degree stats, clustering, diameter
- Spectral features: Laplacian eigenvalues
- Subgraph counts: rings, BRICS, functional groups
- Graph distances: path lengths, Wiener index
"""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

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
    
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {i}: {smiles}")
        fp = generator.GetFingerprint(mol)
        fingerprints.append(np.array(fp))
    
    return np.array(fingerprints, dtype=np.float32)


def create_maccs(smiles_list):
    """
    Create MACCS keys fingerprints (167 bits).
    
    MACCS (Molecular ACCess System) keys are a set of 166 structural keys
    plus 1 unused key, designed for substructure searching.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        np.ndarray: Binary fingerprint matrix (n_molecules, 167)
    """
    fingerprints = []
    
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {i}: {smiles}")
        fp = MACCSkeys.GenMACCSKeys(mol)
        fingerprints.append(np.array(fp))
    
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
    
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {i}: {smiles}")
        desc = np.array(calc.CalcDescriptors(mol), dtype=np.float32)
        desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
        descriptors.append(desc)
    
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
    if reference_featurizer is not None:
        features = _transform_sns(smiles_list, reference_featurizer)
        return features
    
    print(f"Fitting SNS featurizer on {len(smiles_list)} molecules...")
    
    mols = []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {i}: {smiles}")
        mols.append(mol)
    
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
    
    def sub_id_enumerator(mol):
        fp = morgan_generator.GetSparseCountFingerprint(mol)
        return fp.GetNonzeroElements()
    
    sub_ids_to_prevs = {}
    for mol in mols:
        for sub_id in sub_id_enumerator(mol).keys():
            sub_ids_to_prevs[sub_id] = sub_ids_to_prevs.get(sub_id, 0) + 1
    
    sub_ids_sorted = sorted(
        sub_ids_to_prevs.keys(),
        key=lambda sub_id: (sub_ids_to_prevs[sub_id], sub_id),
        reverse=True
    )
    
    top_sub_ids = set(sub_ids_sorted[:vec_dimension])
    sub_id_to_index = {sub_id: i for i, sub_id in enumerate(sub_ids_sorted[:vec_dimension])}
    
    print(f"  Found {len(sub_ids_to_prevs)} unique substructures, keeping top {vec_dimension}")
    
    featurizer = {
        'morgan_generator': morgan_generator,
        'sub_id_to_index': sub_id_to_index,
        'top_sub_ids': top_sub_ids,
        'vec_dimension': vec_dimension,
        'sub_counts': sub_counts
    }
    
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
            raise ValueError(f"Invalid SMILES at index {i}: {smiles}")
        
        fp = morgan_generator.GetSparseCountFingerprint(mol)
        sub_id_counts = fp.GetNonzeroElements()
        
        for sub_id, count in sub_id_counts.items():
            if sub_id in top_sub_ids:
                idx = sub_id_to_index[sub_id]
                if sub_counts:
                    features[i, idx] = count
                else:
                    features[i, idx] = 1
    
    return features


# =============================================================================
# PRETRAINED REPRESENTATIONS (FROZEN)
# =============================================================================

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
    from mol2vec.features import mol2alt_sentence, MolSentence
    from gensim.models import word2vec
    import os
    
    global _MOL2VEC_MODEL
    
    if _MOL2VEC_MODEL is None:
        if model_path is None:
            search_paths = [
                'model_300dim.pkl',
                '../model_300dim.pkl',
                'tests/model_300dim.pkl',
                '../tests/model_300dim.pkl',
                os.path.expanduser('~/model_300dim.pkl'),
            ]
            
            found_path = None
            for path in search_paths:
                if os.path.exists(path):
                    found_path = path
                    break
            
            if found_path is None:
                raise FileNotFoundError(
                    f"Could not find mol2vec model in: {search_paths}\n"
                    f"Download from: https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl"
                )
            
            print(f"Loading mol2vec model from {found_path}...")
            _MOL2VEC_MODEL = word2vec.Word2Vec.load(found_path)
        else:
            _MOL2VEC_MODEL = word2vec.Word2Vec.load(model_path)
    
    embeddings = []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {i}: {smiles}")
        
        sentence = MolSentence(mol2alt_sentence(mol, radius))
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


_MHG_GNN_MODEL = None

def create_mhg_gnn(smiles_list, n_features=None, batch_size=32, 
                   materials_repo_path=None, model_pickle_path=None):
    """
    Create MHG-GNN embeddings (pretrained GNN encoder from IBM).

    CODE REQUIREMENT:
    A directory containing models/mhg_model/

    WEIGHTS REQUIREMENT:
    The pretrained weight file mhggnn_pretrained_model_0724_2023.pickle

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
    import pickle
    
    global _MHG_GNN_MODEL
    
    if _MHG_GNN_MODEL is None:
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
                raise FileNotFoundError(
                    f"Could not find MHG-GNN code directory. Searched: {search_paths}\n"
                    f"Expected: <path>/models/mhg_model"
                )
        
        models_path = os.path.join(materials_repo_path, 'models')
        if models_path not in sys.path:
            sys.path.insert(0, models_path)
        
        print(f"Loading MHG-GNN from {materials_repo_path}...")
        
        from mhg_model.load import PretrainedModelWrapper
        
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
                raise FileNotFoundError(
                    f"Could not find model pickle. Searched: {search_paths}\n"
                    f"Download from: https://huggingface.co/ibm-research/materials.mhg-ged"
                )
        
        with open(model_pickle_path, 'rb') as f:
            model_dict = pickle.load(f)
        
        _MHG_GNN_MODEL = PretrainedModelWrapper(model_dict)
        _MHG_GNN_MODEL.model.eval()
        print(f"MHG-GNN model loaded successfully")
    
    print(f"Encoding {len(smiles_list)} molecules with MHG-GNN...")
    
    all_embeddings = []
    
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        batch_embeddings = _MHG_GNN_MODEL.encode(batch)
        batch_np = torch.stack(batch_embeddings).cpu().detach().numpy()
        all_embeddings.append(batch_np)
    
    embeddings = np.vstack(all_embeddings)
    
    if n_features is not None and n_features < embeddings.shape[1]:
        if embeddings.shape[0] < n_features:
            raise ValueError(
                f"Cannot reduce to {n_features} dims with only {embeddings.shape[0]} samples. "
                f"Fit PCA on training set first."
            )
        from sklearn.decomposition import PCA
        print(f"Reducing from {embeddings.shape[1]} to {n_features} dims via PCA...")
        pca = PCA(n_components=n_features)
        embeddings = pca.fit_transform(embeddings)
    
    return embeddings


_CHEMBERTA_MODEL = None
_CHEMBERTA_TOKENIZER = None

def create_chemberta(smiles_list, batch_size=32, device=None):
    """
    Create ChemBERTa embeddings (frozen, no fine-tuning).

    Uses DeepChem's ChemBERTa-77M-MTR, trained on 77M molecules with
    multi-task regression pretraining for better molecular representations.

    Args:
        smiles_list: List of SMILES strings
        batch_size: Batch size for encoding (default: 32)
        device: 'cpu' or 'cuda' (default: auto-detect)

    Returns:
        np.ndarray: ChemBERTa embeddings (n_molecules, 768)
    """
    import os
    # Prevent tokenizer parallelism issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch
    from transformers import RobertaModel, RobertaTokenizer

    global _CHEMBERTA_MODEL, _CHEMBERTA_TOKENIZER

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if _CHEMBERTA_MODEL is None:
        print("Loading ChemBERTa-77M-MTR...")
        _CHEMBERTA_TOKENIZER = RobertaTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
        _CHEMBERTA_MODEL = RobertaModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
        _CHEMBERTA_MODEL = _CHEMBERTA_MODEL.to(device).eval()
        print(f"  Loaded on {device}")
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            
            encoded = _CHEMBERTA_TOKENIZER(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in encoded.items()}
            
            outputs = _CHEMBERTA_MODEL(**inputs)
            
            # Mean pooling over sequence length (excluding padding)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
            sum_mask = attention_mask.sum(dim=1)
            embeddings = sum_embeddings / sum_mask
            
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings).astype(np.float32)


_MOLFORMER_MODEL = None
_MOLFORMER_TOKENIZER = None

def create_molformer(smiles_list, batch_size=32, device=None):
    """
    Create MolFormer embeddings (frozen, no fine-tuning).

    Uses IBM's MoLFormer-XL trained on 1.1B molecules from PubChem and ZINC.
    Linear attention mechanism allows processing of long SMILES sequences.

    Args:
        smiles_list: List of SMILES strings
        batch_size: Batch size for encoding (default: 32)
        device: 'cpu' or 'cuda' (default: auto-detect)

    Returns:
        np.ndarray: MolFormer embeddings (n_molecules, 768)
    """
    import os
    # Prevent tokenizer parallelism and torch extension issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Import torchvision BEFORE torch CUDA initialization to avoid circular import
    # Catch all exceptions since torchvision may be incompatible with torch version
    try:
        import torchvision  # noqa: F401 - imported for side effects
    except Exception:
        pass  # torchvision not required, just prevents circular import if present

    import torch
    from transformers import AutoModel, AutoTokenizer

    global _MOLFORMER_MODEL, _MOLFORMER_TOKENIZER

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if _MOLFORMER_MODEL is None:
        print("Loading MoLFormer-XL...")
        _MOLFORMER_TOKENIZER = AutoTokenizer.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            trust_remote_code=True
        )
        _MOLFORMER_MODEL = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            trust_remote_code=True
        )
        _MOLFORMER_MODEL = _MOLFORMER_MODEL.to(device).eval()
        print(f"  Loaded on {device}")
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            
            encoded = _MOLFORMER_TOKENIZER(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in encoded.items()}
            
            outputs = _MOLFORMER_MODEL(**inputs)
            
            # Mean pooling over sequence length (excluding padding)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
            sum_mask = attention_mask.sum(dim=1)
            embeddings = sum_embeddings / sum_mask
            
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings).astype(np.float32)


# =============================================================================
# GRAPH KERNEL FEATURES
# =============================================================================

def create_graph_kernel(smiles_list, kernel='weisfeiler_lehman', n_iter=5, 
                        n_features=None, reference_vocabulary=None, return_vocabulary=False):
    """
    Create graph kernel features using WL histogram approach.
    
    Extracts WL label histograms as features - the proper way to use graph kernels
    with non-kernel methods like Random Forest.
    
    Args:
        smiles_list: List of SMILES strings
        kernel: 'weisfeiler_lehman' (others not implemented)
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
    
    graphs_data = []
    
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {i}: {smiles}")
        if mol.GetNumBonds() == 0:
            raise ValueError(f"Molecule at index {i} has no bonds: {smiles}")
        
        node_labels = {atom.GetIdx(): atom.GetSymbol() for atom in mol.GetAtoms()}
        edges = {atom.GetIdx(): [] for atom in mol.GetAtoms()}
        for bond in mol.GetBonds():
            i_atom, j_atom = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges[i_atom].append(j_atom)
            edges[j_atom].append(i_atom)
        
        graphs_data.append((node_labels, edges))
    
    print(f"Computing WL histograms with {n_iter} iterations...")
    
    all_histograms = []
    discovered_labels = set()
    
    for graph_data in graphs_data:
        node_labels, edges = graph_data
        current_labels = node_labels.copy()
        histogram = Counter()
        
        for iteration in range(n_iter + 1):
            for label in current_labels.values():
                label_str = f"{iteration}_{label}"
                histogram[label_str] += 1
                if reference_vocabulary is None:
                    discovered_labels.add(label_str)
            
            if iteration < n_iter:
                new_labels = {}
                for node, label in current_labels.items():
                    neighbor_labels = sorted([current_labels[n] for n in edges[node]])
                    new_label = f"{label}_{'_'.join(neighbor_labels)}"
                    new_labels[node] = new_label
                current_labels = new_labels
        
        all_histograms.append(histogram)
    
    if reference_vocabulary is not None:
        vocabulary = reference_vocabulary
        print(f"  Using {len(vocabulary)} reference WL labels")
    else:
        vocabulary = discovered_labels
        print(f"  Extracted {len(vocabulary)} unique WL labels")
    
    sorted_labels = sorted(vocabulary)
    features = np.zeros((len(smiles_list), len(sorted_labels)), dtype=np.float32)
    
    for i, histogram in enumerate(all_histograms):
        for j, label in enumerate(sorted_labels):
            features[i, j] = histogram.get(label, 0)
    
    if return_vocabulary:
        return features, vocabulary
    return features


# =============================================================================
# GRAPH AUGMENTATIONS
# =============================================================================

def compute_graph_topology(smiles_list):
    """
    Compute graph topology features from molecular graphs.
    
    Features (10 total):
    - num_atoms: Number of atoms
    - num_bonds: Number of bonds  
    - degree_mean: Mean node degree
    - degree_max: Maximum node degree
    - degree_std: Standard deviation of node degrees
    - density: Graph density (2*edges / (nodes*(nodes-1)))
    - num_components: Number of connected components
    - diameter: Graph diameter (-1 if disconnected)
    - radius: Graph radius (-1 if disconnected)
    - clustering: Average clustering coefficient
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        np.ndarray: Topology features (n_molecules, 10)
    """
    import networkx as nx
    
    features = []
    
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {i}: {smiles}")
        
        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx())
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        
        num_atoms = G.number_of_nodes()
        num_bonds = G.number_of_edges()
        
        degrees = [d for n, d in G.degree()]
        degree_mean = np.mean(degrees)
        degree_max = np.max(degrees)
        degree_std = np.std(degrees)
        
        density = nx.density(G)
        num_components = nx.number_connected_components(G)
        
        if num_components == 1 and num_atoms > 1:
            diameter = nx.diameter(G)
            radius = nx.radius(G)
        else:
            diameter = -1
            radius = -1
        
        clustering = nx.average_clustering(G)
        
        features.append(np.array([
            num_atoms, num_bonds, degree_mean, degree_max, degree_std,
            density, num_components, diameter, radius, clustering
        ], dtype=np.float32))
    
    return np.array(features)


def compute_spectral_features(smiles_list, k=10):
    """
    Compute spectral features from molecular graph Laplacian.
    
    Extracts the k smallest non-trivial eigenvalues of the normalized
    graph Laplacian. These capture the global structure of the molecule.
    
    Args:
        smiles_list: List of SMILES strings
        k: Number of eigenvalues to return (default: 10)
        
    Returns:
        np.ndarray: Spectral features (n_molecules, k)
    """
    features = []
    
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {i}: {smiles}")
        
        num_atoms = mol.GetNumAtoms()
        
        if num_atoms < 2:
            raise ValueError(f"Molecule at index {i} has fewer than 2 atoms: {smiles}")
        
        # Build adjacency matrix
        adj = np.zeros((num_atoms, num_atoms), dtype=np.float32)
        for bond in mol.GetBonds():
            i_atom, j_atom = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adj[i_atom, j_atom] = 1
            adj[j_atom, i_atom] = 1
        
        # Compute normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        degrees = adj.sum(axis=1)
        degrees[degrees == 0] = 1  # Avoid division by zero for isolated atoms
        d_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        laplacian = np.eye(num_atoms) - d_inv_sqrt @ adj @ d_inv_sqrt
        
        # Compute eigenvalues
        eigenvalues = np.sort(np.linalg.eigvalsh(laplacian))
        
        # Pad or truncate to k values
        if len(eigenvalues) < k:
            eigenvalues = np.pad(eigenvalues, (0, k - len(eigenvalues)), mode='constant')
        else:
            eigenvalues = eigenvalues[:k]
        
        features.append(eigenvalues.astype(np.float32))
    
    return np.array(features)


def compute_subgraph_counts(smiles_list):
    """
    Compute subgraph and motif counts from molecules.
    
    Features (20 total):
    - ring_3 through ring_8: Count of rings of each size
    - aromatic_rings: Number of aromatic rings
    - aliphatic_rings: Number of aliphatic rings
    - brics_fragments: Number of BRICS decomposition fragments
    - heterocycles: Number of heterocyclic rings
    - spiro_atoms: Number of spiro atoms
    - bridgehead_atoms: Number of bridgehead atoms
    - rotatable_bonds: Number of rotatable bonds
    - hbd: Number of hydrogen bond donors
    - hba: Number of hydrogen bond acceptors
    - stereocenters: Number of stereocenters
    - amide_bonds: Number of amide bonds
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        np.ndarray: Subgraph count features (n_molecules, 20)
    """
    from rdkit.Chem import BRICS, rdMolDescriptors
    
    features = []
    
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {i}: {smiles}")
        
        feat = []
        
        # Ring counts by size
        ring_info = mol.GetRingInfo()
        ring_sizes = [len(r) for r in ring_info.AtomRings()]
        for size in range(3, 9):
            feat.append(ring_sizes.count(size))
        
        # Ring type counts
        feat.append(rdMolDescriptors.CalcNumAromaticRings(mol))
        feat.append(rdMolDescriptors.CalcNumAliphaticRings(mol))
        
        # BRICS fragments
        brics_bonds = list(BRICS.FindBRICSBonds(mol))
        feat.append(len(brics_bonds) + 1 if brics_bonds else 1)
        
        # Other structural features
        feat.append(rdMolDescriptors.CalcNumHeterocycles(mol))
        feat.append(rdMolDescriptors.CalcNumSpiroAtoms(mol))
        feat.append(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
        feat.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        feat.append(rdMolDescriptors.CalcNumHBD(mol))
        feat.append(rdMolDescriptors.CalcNumHBA(mol))
        feat.append(rdMolDescriptors.CalcNumAtomStereoCenters(mol))
        feat.append(rdMolDescriptors.CalcNumAmideBonds(mol))
        
        features.append(np.array(feat, dtype=np.float32))
    
    return np.array(features)


def compute_graph_distances(smiles_list):
    """
    Compute graph distance statistics from molecular graphs.
    
    Features (8 total):
    - mean_path_length: Mean shortest path length between all pairs
    - max_path_length: Maximum shortest path length (diameter)
    - wiener_index: Sum of all shortest path lengths
    - path_length_std: Standard deviation of path lengths
    - eccentricity_mean: Mean eccentricity of nodes
    - eccentricity_std: Standard deviation of eccentricity
    - periphery_size: Number of nodes in periphery
    - center_size: Number of nodes in center
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        np.ndarray: Graph distance features (n_molecules, 8)
    """
    import networkx as nx
    
    features = []
    
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {i}: {smiles}")
        
        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx())
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        
        if G.number_of_nodes() < 2:
            raise ValueError(f"Molecule at index {i} has fewer than 2 atoms: {smiles}")
        
        # Handle disconnected graphs by taking largest component
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        
        # Compute all shortest paths
        path_lengths = dict(nx.all_pairs_shortest_path_length(G))
        all_paths = [
            length 
            for source in path_lengths 
            for target, length in path_lengths[source].items() 
            if source < target
        ]
        
        mean_path = np.mean(all_paths)
        max_path = np.max(all_paths)
        wiener = np.sum(all_paths)
        std_path = np.std(all_paths)
        
        # Eccentricity
        eccentricities = list(nx.eccentricity(G).values())
        ecc_mean = np.mean(eccentricities)
        ecc_std = np.std(eccentricities)
        
        # Periphery and center
        periphery_size = len(nx.periphery(G))
        center_size = len(nx.center(G))
        
        features.append(np.array([
            mean_path, max_path, wiener, std_path,
            ecc_mean, ecc_std, periphery_size, center_size
        ], dtype=np.float32))
    
    return np.array(features)


# =============================================================================
# FINE-TUNING: Graph Neural Networks
# =============================================================================

def finetune_gnn(train_smiles, train_labels, val_smiles, val_labels,
                 gnn_type='gcn', hidden_dim=128, num_layers=3, epochs=100,
                 batch_size=32, learning_rate=1e-3):
    """
    Fine-tune a GNN (GCN/GAT/MPNN) on YOUR task-specific data.
    
    Trains GNN from scratch on YOUR data.
    
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
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch.utils.data import DataLoader
    
    def smiles_to_graph(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
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
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)
    
    print(f"Converting SMILES to graphs...")
    train_graphs = [smiles_to_graph(s) for s in train_smiles]
    val_graphs = [smiles_to_graph(s) for s in val_smiles]
    
    train_data = list(zip(train_graphs, train_labels))
    val_data = list(zip(val_graphs, val_labels))
    
    def collate_fn(batch):
        graphs, labels = zip(*batch)
        batched_graph = Batch.from_data_list(graphs)
        labels = torch.tensor(labels, dtype=torch.float32)
        return batched_graph, labels
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn)
    
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


def encode_from_model(finetuned_model_dict, smiles_list, batch_size=32):
    """
    Extract embeddings from a fine-tuned GNN model.
    
    Args:
        finetuned_model_dict: Dictionary from finetune_gnn()
        smiles_list: List of SMILES to encode
        batch_size: Batch size (default: 32)
        
    Returns:
        np.ndarray: Embeddings (n_molecules, hidden_dim)
    """
    import torch
    from torch_geometric.data import Data, Batch
    
    model = finetuned_model_dict['model']
    device = finetuned_model_dict['device']
    model.eval()
    
    def smiles_to_graph(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
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
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)
    
    all_embeddings = []
    
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i + batch_size]
        graphs = [smiles_to_graph(s) for s in batch_smiles]
        batched = Batch.from_data_list(graphs).to(device)
        
        with torch.no_grad():
            _, embeddings = model(batched)
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


# =============================================================================
# GAUCHE: Gaussian Process with Graph Kernels
# =============================================================================

def train_gauche_gp(train_smiles, train_labels, kernel='weisfeiler_lehman',
                    n_iter=5, num_epochs=100, learning_rate=0.1):
    """
    Train a Gaussian Process with graph kernel on YOUR data.
    
    Args:
        train_smiles: Training SMILES
        train_labels: Training labels
        kernel: 'weisfeiler_lehman'
        n_iter: WL iterations (default: 5)
        num_epochs: GP training epochs (default: 100)
        learning_rate: Learning rate (default: 0.1)
        
    Returns:
        dict: {'gp_model', 'likelihood', 'kernel_obj', 'train_graphs', 'train_labels'}
    """
    import torch
    import gpytorch
    from grakel.kernels import WeisfeilerLehman
    from grakel import Graph
    
    print(f"Converting {len(train_smiles)} molecules to graphs...")
    graphs = []
    for i, smiles in enumerate(train_smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {i}: {smiles}")
        
        node_labels = {}
        edges = {}
        
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            node_labels[idx] = atom.GetSymbol()
            edges[idx] = []
        
        for bond in mol.GetBonds():
            i_atom = bond.GetBeginAtomIdx()
            j_atom = bond.GetEndAtomIdx()
            edges[i_atom].append(j_atom)
            edges[j_atom].append(i_atom)
        
        g = Graph(edges, node_labels=node_labels)
        graphs.append(g)
    
    valid_labels = torch.tensor(train_labels, dtype=torch.float32)
    
    print(f"Computing kernel matrix for {len(graphs)} molecules...")
    
    if kernel == 'weisfeiler_lehman':
        kernel_obj = WeisfeilerLehman(n_iter=n_iter, normalize=True)
    else:
        raise ValueError(f"Kernel '{kernel}' not implemented. Use 'weisfeiler_lehman'")
    
    K_train = kernel_obj.fit_transform(graphs)
    K_train_tensor = torch.tensor(K_train, dtype=torch.float32)
    
    jitter = 1e-4
    K_train_tensor = K_train_tensor + jitter * torch.eye(K_train_tensor.shape[0])
    
    print(f"Training GP on kernel matrix of shape {K_train_tensor.shape}...")
    
    train_indices = torch.arange(len(graphs), dtype=torch.float32).unsqueeze(-1)
    
    class PrecomputedKernelGP(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, K_train):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.register_buffer('K_train', K_train)
            self.outputscale = torch.nn.Parameter(torch.ones(1))
        
        def forward(self, x):
            mean_x = self.mean_module(x)
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
        'kernel_obj': kernel_obj,
        'train_graphs': graphs,
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
    for i, smiles in enumerate(test_smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {i}: {smiles}")
        
        node_labels = {}
        edges = {}
        
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            node_labels[idx] = atom.GetSymbol()
            edges[idx] = []
        
        for bond in mol.GetBonds():
            i_atom = bond.GetBeginAtomIdx()
            j_atom = bond.GetEndAtomIdx()
            edges[i_atom].append(j_atom)
            edges[j_atom].append(i_atom)
        
        g = Graph(edges, node_labels=node_labels)
        test_graphs.append(g)
    
    print(f"Computing kernel matrix between {len(test_graphs)} test and {len(train_graphs)} train graphs...")
    K_test_train = kernel_obj.transform(test_graphs)
    K_test_train_tensor = torch.tensor(K_test_train, dtype=torch.float32)
    
    gp_model.eval()
    likelihood.eval()
    
    with torch.no_grad():
        train_labels = gp_dict['train_labels']
        K_train = gp_model.K_train
        noise_var = likelihood.noise.item()
        
        K_train_noise = K_train + noise_var * torch.eye(K_train.shape[0])
        alpha = torch.linalg.solve(K_train_noise, train_labels)
        mean = (K_test_train_tensor @ alpha).numpy()
        std = torch.ones(len(test_graphs)).numpy() * noise_var**0.5
    
    return {
        'predictions': mean,
        'uncertainties': std
    }


# =============================================================================
# HYBRID: Simple Concatenation (use kirby.hybrid for advanced allocation)
# =============================================================================

def create_hybrid(embeddings_dict, selection_method='concat', n_features=None):
    """
    Combine embeddings via simple concatenation or PCA.
    
    For advanced allocation (greedy, performance-weighted), use kirby.hybrid.create_hybrid.
    
    Args:
        embeddings_dict: Dict of embeddings
        selection_method: 'concat' or 'pca' (default: 'concat')
        n_features: For PCA, target dimensionality (default: None)
        
    Returns:
        np.ndarray: Hybrid representation
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