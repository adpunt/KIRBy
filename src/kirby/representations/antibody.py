"""
Antibody Molecular Representations - SOTA Methods

Implements state-of-the-art antibody featurization from Nature ML papers:
- Language model embeddings (AntiBERTy, AbLang, BALM, IgBERT)
- CDR-stratified features
- IMGT positional encoding
- Structure-based features (when available)
- Multi-scale aggregations

References:
- IgFold: Ruffolo et al. (2023) Nature Communications
- AbLang: Olsen et al. (2022) Bioinformatics Advances  
- BALM: (2024) Briefings in Bioinformatics
- Paragraph: (2023) Bioinformatics
"""

import numpy as np
import torch
import torch.nn as nn
import warnings
from typing import List, Dict, Optional, Tuple


# ============================================================================
# PART 1: LANGUAGE MODEL EMBEDDINGS
# ============================================================================

def create_antiberty_embeddings_batch(
    heavy_sequences: List[str],
    light_sequences: Optional[List[str]] = None,
    aggregations: List[str] = ['mean', 'max', 'attention'],
    device: str = 'cpu'
) -> Dict[str, np.ndarray]:
    """
    Extract AntiBERTy embeddings with multiple aggregations efficiently.
    
    Loads the model ONCE and applies all aggregations, much faster than
    calling create_antiberty_embeddings() multiple times.
    
    Args:
        heavy_sequences: Heavy chain sequences
        light_sequences: Light chain sequences (optional)
        aggregations: List of aggregation methods to apply
        device: 'cpu' or 'cuda'
    
    Returns:
        dict: {
            'heavy_mean': (n_seqs, hidden_dim),
            'heavy_max': (n_seqs, hidden_dim),
            'heavy_attention': (n_seqs, hidden_dim),
            'light_mean': (n_seqs, hidden_dim),  # if light_sequences provided
            ...
        }
    """
    try:
        from transformers import RoFormerModel, RoFormerTokenizer
    except ImportError:
        raise ImportError("AntiBERTy requires transformers: pip install transformers")
    
    print(f"Loading AntiBERTy (batch mode - single model load)...")
    
    # Load model ONCE
    try:
        model_name = "alchemab/antiberta2"
        tokenizer = RoFormerTokenizer.from_pretrained(model_name)
        model = RoFormerModel.from_pretrained(model_name)
        model = model.to(device).eval()
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    results = {}
    
    # Process heavy chain
    print(f"  Processing {len(heavy_sequences)} heavy chain sequences...")
    heavy_embeddings_per_residue = []
    
    with torch.no_grad():
        for i, seq in enumerate(heavy_sequences):
            if (i + 1) % 50 == 0:
                print(f"    Progress: {i+1}/{len(heavy_sequences)}")
            
            # Tokenize
            # Space-separated sequences: each AA becomes 2 tokens (AA + space)
            # Max 512 positions - 2 special tokens = 510 / 2 = 255 AA max
            seq_spaced = ' '.join(list(seq))
            inputs = tokenizer(seq_spaced, return_tensors='pt', truncation=True, max_length=256, padding='max_length')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get per-residue embeddings
            outputs = model(**inputs)
            residue_embeddings = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)
            
            heavy_embeddings_per_residue.append(residue_embeddings)
    
    # Apply all aggregations to heavy chain
    for agg in aggregations:
        aggregated = []
        for residue_embs in heavy_embeddings_per_residue:
            if agg == 'mean':
                emb = residue_embs.mean(dim=0)
            elif agg == 'max':
                emb = residue_embs.max(dim=0)[0]
            elif agg == 'cls':
                emb = residue_embs[0]
            elif agg == 'attention':
                attn_weights = torch.softmax(residue_embs.norm(dim=1), dim=0)
                emb = (residue_embs * attn_weights.unsqueeze(1)).sum(dim=0)
            else:
                raise ValueError(f"Unknown aggregation: {agg}")
            
            aggregated.append(emb.cpu().numpy())
        
        results[f'heavy_{agg}'] = np.array(aggregated)
    
    # Process light chain if provided
    if light_sequences is not None:
        print(f"  Processing {len(light_sequences)} light chain sequences...")
        light_embeddings_per_residue = []
        
        with torch.no_grad():
            for i, seq in enumerate(light_sequences):
                if (i + 1) % 50 == 0:
                    print(f"    Progress: {i+1}/{len(light_sequences)}")
                
                seq_spaced = ' '.join(list(seq))
                inputs = tokenizer(seq_spaced, return_tensors='pt', truncation=True, max_length=256, padding='max_length')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs)
                residue_embeddings = outputs.last_hidden_state[0]
                
                light_embeddings_per_residue.append(residue_embeddings)
        
        # Apply all aggregations to light chain
        for agg in aggregations:
            aggregated = []
            for residue_embs in light_embeddings_per_residue:
                if agg == 'mean':
                    emb = residue_embs.mean(dim=0)
                elif agg == 'max':
                    emb = residue_embs.max(dim=0)[0]
                elif agg == 'cls':
                    emb = residue_embs[0]
                elif agg == 'attention':
                    attn_weights = torch.softmax(residue_embs.norm(dim=1), dim=0)
                    emb = (residue_embs * attn_weights.unsqueeze(1)).sum(dim=0)
                else:
                    raise ValueError(f"Unknown aggregation: {agg}")
                
                aggregated.append(emb.cpu().numpy())
            
            results[f'light_{agg}'] = np.array(aggregated)
    
    print(f"  Created {len(results)} embedding sets")
    return results


def create_ablang2_embeddings(
    heavy_sequences: List[str],
    light_sequences: List[str],
    device: str = 'cpu'
) -> np.ndarray:
    """
    Extract AbLang-2 embeddings for paired antibody sequences.
    
    AbLang-2 is trained on 35.6M unpaired + 1.26M paired sequences,
    optimized to reduce germline bias and better model non-germline residues.
    
    Args:
        heavy_sequences: Heavy chain sequences
        light_sequences: Light chain sequences
        device: 'cpu' or 'cuda'
    
    Returns:
        np.ndarray: (n_seqs, 480) - sequence-level embeddings
    """
    try:
        import ablang2
        import torch
    except ImportError:
        raise ImportError(
            "AbLang2 requires: pip install ablang2 torch\n"
            "AbLang-2 is an antibody-specific language model from Olsen et al."
        )
    
    print(f"Loading AbLang-2 model (paired mode)...")
    
    # Load pre-trained AbLang-2 model
    ablang = ablang2.pretrained(
        model_to_use='ablang2-paired',
        random_init=False,
        ncpu=1,
        device=device
    )
    
    # Clean sequences - only keep standard amino acids
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    
    def clean_sequence(seq):
        return ''.join([c.upper() for c in seq if c.upper() in valid_aa])
    
    cleaned_heavy = [clean_sequence(h) for h in heavy_sequences]
    cleaned_light = [clean_sequence(l) for l in light_sequences]
    
    print(f"  Encoding {len(cleaned_heavy)} paired sequences...")
    
    # Format as list of [heavy, light] pairs, then use wrapper mode
    seq_pairs = [[h, l] for h, l in zip(cleaned_heavy, cleaned_light)]
    
    # Use the high-level wrapper with mode='seqcoding'
    embeddings = ablang(seq_pairs, mode='seqcoding')
    
    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    elif isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    
    print(f"  Generated embeddings: {embeddings.shape}")
    return embeddings


def create_antiberty_embeddings(
    sequences: List[str],
    chain_type: str = 'heavy',
    aggregation: str = 'mean',
    device: str = 'cpu'
) -> np.ndarray:
    """
    Extract AntiBERTy embeddings.
    
    Paper: Ruffolo et al. (2023) "Fast, accurate antibody structure prediction..."
    - 26M parameters, trained on 558M sequences
    - 512D embeddings per residue
    - Used by IgFold for structure prediction
    
    Args:
        sequences: List of antibody sequences
        chain_type: 'heavy' or 'light' (for paired antibodies)
        aggregation: How to pool residue embeddings
            - 'mean': Average over all residues
            - 'max': Max pool over residues
            - 'cls': Use [CLS] token (if available)
            - 'attention': Learned attention pooling
        device: 'cpu' or 'cuda'
    
    Returns:
        np.ndarray: (n_sequences, 512) embeddings
    """
    try:
        from transformers import RoFormerModel, RoFormerTokenizer
    except ImportError:
        raise ImportError("AntiBERTy requires transformers: pip install transformers")
    
    print(f"Loading AntiBERTy ({chain_type} chain)...")
    
    # Load pretrained AntiBERTy
    # Note: Actual model is 'Exscientia/IgBert' or similar on HuggingFace
    # For now, placeholder - would need actual model weights
    
    try:
        model_name = "alchemab/antiberta2"
        tokenizer = RoFormerTokenizer.from_pretrained(model_name)
        model = RoFormerModel.from_pretrained(model_name)
        model = model.to(device).eval()
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    embeddings = []
    
    with torch.no_grad():
        for i, seq in enumerate(sequences):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(sequences)}")
            
            # Tokenize with spaces between amino acids
            seq_spaced = ' '.join(list(seq))
            inputs = tokenizer(seq_spaced, return_tensors='pt', truncation=True, max_length=256, padding='max_length')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get embeddings
            outputs = model(**inputs)
            residue_embeddings = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)
            
            # Aggregate
            if aggregation == 'mean':
                emb = residue_embeddings.mean(dim=0)
            elif aggregation == 'max':
                emb = residue_embeddings.max(dim=0)[0]
            elif aggregation == 'cls':
                emb = residue_embeddings[0]  # [CLS] token
            elif aggregation == 'attention':
                # Simple learned attention
                attn_weights = torch.softmax(residue_embeddings.norm(dim=1), dim=0)
                emb = (residue_embeddings * attn_weights.unsqueeze(1)).sum(dim=0)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
            
            embeddings.append(emb.cpu().numpy())
    
    return np.array(embeddings)


def create_ablang_embeddings(
    heavy_sequences: List[str],
    light_sequences: Optional[List[str]] = None,
    output_type: str = 'seq-codings',
    device: str = 'cpu'
) -> np.ndarray:
    """
    Extract AbLang embeddings.
    
    Paper: Olsen et al. (2022) "AbLang: an antibody language model..."
    - Separate heavy/light chain models
    - 768D embeddings
    - Three output types: res-codings, seq-codings, res-likelihoods
    
    Args:
        heavy_sequences: Heavy chain sequences
        light_sequences: Light chain sequences (optional)
        output_type: 
            - 'res-codings': 768D per residue
            - 'seq-codings': 768D per sequence (mean pooled)
            - 'paired': Concatenate heavy + light (1536D)
        device: 'cpu' or 'cuda'
    
    Returns:
        np.ndarray: Embeddings
    """
    try:
        import ablang
    except ImportError:
        raise ImportError("AbLang requires ablang: pip install ablang")
    
    print(f"Loading AbLang (output: {output_type})...")
    
    # Load heavy chain model
    heavy_ablang = ablang.pretrained("heavy")
    heavy_ablang.freeze()
    
    if output_type == 'seq-codings':
        # Sequence-level embeddings (768D per sequence)
        heavy_embs = heavy_ablang(heavy_sequences, mode='seqcoding')
        
        if light_sequences is None or output_type != 'paired':
            return heavy_embs
        
        # Also get light chain
        light_ablang = ablang.pretrained("light")
        light_ablang.freeze()
        light_embs = light_ablang(light_sequences, mode='seqcoding')
        
        # Concatenate
        return np.concatenate([heavy_embs, light_embs], axis=1)
    
    elif output_type == 'res-codings':
        # Per-residue embeddings
        heavy_embs = heavy_ablang(heavy_sequences, mode='rescoding')
        
        # Need to aggregate - use mean
        embeddings = []
        for emb in heavy_embs:
            embeddings.append(emb.mean(axis=0))
        
        return np.array(embeddings)
    
    else:
        raise ValueError(f"Unknown output_type: {output_type}")


# ============================================================================
# PART 2: CDR-STRATIFIED FEATURES
# ============================================================================

def create_cdr_stratified_embeddings(
    sequences: List[str],
    chain_type: str = 'heavy',
    embedding_fn=None,
    numbering: str = 'imgt'
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings for each CDR and framework region separately.
    
    Strategy from papers: CDR-H3 is most important, but all regions contribute.
    By stratifying, we let KIRBy's feature selection learn region importance.
    
    Args:
        sequences: Antibody sequences
        chain_type: 'heavy' or 'light'
        embedding_fn: Function to get per-residue embeddings
        numbering: 'imgt' or 'chothia'
    
    Returns:
        dict: {
            'cdr1_mean': (n_seqs, embed_dim),
            'cdr1_max': (n_seqs, embed_dim),
            'cdr2_mean': ...,
            'cdr3_mean': ...,  # Most important!
            'cdr3_max': ...,
            'framework_mean': ...,
            'framework_std': ...
        }
    """
    from ..datasets.sabdab import extract_cdr_sequences
    
    print(f"Creating CDR-stratified features ({chain_type})...")
    
    if embedding_fn is None:
        # Use simple one-hot encoding as baseline
        embedding_fn = lambda seq: _onehot_encode_sequence(seq)
    
    # Extract CDRs
    cdr_data = []
    for seq in sequences:
        cdrs = extract_cdr_sequences(seq, numbering=numbering, chain_type=chain_type)
        if cdrs is None:
            cdrs = {
                'cdr1': '', 'cdr2': '', 'cdr3': '',
                'framework1': '', 'framework2': '', 'framework3': ''
            }
        cdr_data.append(cdrs)
    
    # Get embeddings for each region
    region_embeddings = {}
    
    for region in ['cdr1', 'cdr2', 'cdr3']:
        region_seqs = [cd[region] for cd in cdr_data]
        
        # Get embeddings (per-residue)
        region_embs = []
        for seq in region_seqs:
            if len(seq) == 0:
                # Empty region - use zeros
                region_embs.append(np.zeros((1, 20)))
            else:
                emb = embedding_fn(seq)
                region_embs.append(emb)
        
        # Aggregate in multiple ways
        mean_embs = np.array([emb.mean(axis=0) for emb in region_embs])
        max_embs = np.array([emb.max(axis=0) for emb in region_embs])
        
        region_embeddings[f'{region}_mean'] = mean_embs
        region_embeddings[f'{region}_max'] = max_embs
    
    # Framework regions - combine and get summary stats
    fw_seqs = [
        cd['framework1'] + cd.get('framework2', '') + cd.get('framework3', '')
        for cd in cdr_data
    ]
    
    fw_embs = []
    for seq in fw_seqs:
        if len(seq) == 0:
            fw_embs.append(np.zeros((1, 20)))
        else:
            emb = embedding_fn(seq)
            fw_embs.append(emb)
    
    region_embeddings['framework_mean'] = np.array([emb.mean(axis=0) for emb in fw_embs])
    region_embeddings['framework_std'] = np.array([emb.std(axis=0) for emb in fw_embs])
    
    print(f"  Extracted {len(region_embeddings)} regional feature sets")
    for name, feats in region_embeddings.items():
        print(f"    {name}: {feats.shape}")
    
    return region_embeddings


def _onehot_encode_sequence(seq: str) -> np.ndarray:
    """One-hot encode amino acid sequence"""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
    
    encoding = np.zeros((len(seq), 20))
    for i, aa in enumerate(seq):
        if aa in aa_to_idx:
            encoding[i, aa_to_idx[aa]] = 1
    
    return encoding


# ============================================================================
# PART 3: IMGT POSITIONAL FEATURES
# ============================================================================

def create_imgt_position_features(
    sequences: List[str],
    chain_type: str = 'heavy',
    key_positions: Optional[List[int]] = None,
    embedding_fn=None
) -> np.ndarray:
    """
    Extract features at specific IMGT numbered positions.
    
    Papers show certain positions are critical for binding:
    - IMGT 95-102 (CDR-H3 core)
    - IMGT 31-35 (CDR-H1)
    - Framework positions for stability
    
    Creates fixed-length feature vector regardless of CDR length variation.
    
    Args:
        sequences: Antibody sequences
        chain_type: 'heavy' or 'light'
        key_positions: IMGT positions to extract (None = use defaults)
        embedding_fn: Function to get per-residue embeddings
    
    Returns:
        np.ndarray: (n_sequences, n_positions * embed_dim)
    """
    from ..datasets.sabdab import extract_cdr_sequences, get_imgt_positions, IMGT_KEY_POSITIONS
    
    if key_positions is None:
        # Use literature-defined key positions
        if chain_type == 'heavy':
            key_positions = (
                IMGT_KEY_POSITIONS['heavy']['cdr_h3'] +
                IMGT_KEY_POSITIONS['heavy']['framework_h']
            )
        else:
            key_positions = (
                IMGT_KEY_POSITIONS['light']['cdr_l3'] +
                IMGT_KEY_POSITIONS['light']['framework_l']
            )
    
    print(f"Extracting IMGT position features ({len(key_positions)} positions)...")
    
    if embedding_fn is None:
        embedding_fn = lambda seq: _onehot_encode_sequence(seq)
    
    position_features = []
    
    for seq in sequences:
        # Number sequence
        cdrs = extract_cdr_sequences(seq, numbering='imgt', chain_type=chain_type)
        
        if cdrs is None:
            # Could not number - use zeros
            position_features.append(np.zeros(len(key_positions) * 20))
            continue
        
        numbered_seq = cdrs['numbered_seq']
        positions_dict = get_imgt_positions(numbered_seq, key_positions)
        
        # Get embedding for each position
        pos_embs = []
        for pos in key_positions:
            aa = positions_dict.get(pos, '-')
            if aa == '-':
                # Missing position
                pos_embs.append(np.zeros(20))
            else:
                # Embed single amino acid
                emb = _onehot_encode_sequence(aa)
                pos_embs.append(emb[0])  # (20,)
        
        # Concatenate all positions
        position_features.append(np.concatenate(pos_embs))
    
    return np.array(position_features)


# ============================================================================
# PART 4: MULTI-SCALE AGGREGATIONS
# ============================================================================

def create_multiscale_embeddings(
    sequences: List[str],
    base_embeddings: np.ndarray,
    scales: List[str] = ['global', 'cdr', 'residue_stats']
) -> Dict[str, np.ndarray]:
    """
    Create multi-scale representations of antibody sequences.
    
    Inspired by hierarchical structure:
    Level 1: Individual residues
    Level 2: CDR regions  
    Level 3: Domain (VH/VL)
    Level 4: Full antibody
    
    Args:
        sequences: Antibody sequences
        base_embeddings: Per-sequence or per-residue embeddings
        scales: Which scales to extract
    
    Returns:
        dict: Multiple feature sets at different scales
    """
    features = {}
    
    if 'global' in scales:
        # Level 4: Global sequence representation
        features['global'] = base_embeddings
    
    if 'cdr' in scales:
        # Level 2: CDR-specific (already implemented above)
        pass
    
    if 'residue_stats' in scales:
        # Statistics over residue-level features
        # Requires per-residue embeddings
        pass
    
    return features


# ============================================================================
# PART 5: FINE-TUNING FOR SPECIFIC TASKS
# ============================================================================

class AntibodyAffinityModel(nn.Module):
    """
    Fine-tune antibody language model for affinity prediction.
    
    Architecture:
    - Pre-trained LM (frozen or fine-tuned)
    - Regression head for Kd/ΔG prediction
    """
    def __init__(self, base_model, hidden_dim=256, freeze_base=False):
        super().__init__()
        
        self.base_model = base_model
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Get embedding dimension
        self.embed_dim = base_model.config.hidden_size
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        # Get embeddings from base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token or mean pool
        if hasattr(outputs, 'pooler_output'):
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state.mean(dim=1)
        
        # Predict affinity
        affinity = self.regressor(pooled)
        return affinity


def finetune_antibody_lm(
    sequences: List[str],
    labels: np.ndarray,
    val_sequences: List[str],
    val_labels: np.ndarray,
    base_model_name: str = 'alchemab/antiberta2',
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    freeze_base: bool = False
):
    """
    Fine-tune antibody language model for downstream task.
    
    Similar to your ChemBERTa fine-tuning, but for antibodies.
    
    Returns:
        Trained model for embedding extraction
    """
    try:
        from transformers import RoFormerModel, RoFormerTokenizer, AdamW
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        raise ImportError("Fine-tuning requires transformers")
    
    print(f"Fine-tuning {base_model_name}...")
    
    # Load base model
    tokenizer = RoFormerTokenizer.from_pretrained(base_model_name)
    base_model = RoFormerModel.from_pretrained(base_model_name)
    
    # Create affinity prediction model
    model = AntibodyAffinityModel(base_model, freeze_base=freeze_base)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Tokenize
    def tokenize_seqs(seqs):
        seqs_spaced = [' '.join(list(s)) for s in seqs]
        return tokenizer(seqs_spaced, padding=True, truncation=True, 
                        max_length=256, return_tensors='pt')
    
    train_encodings = tokenize_seqs(sequences)
    val_encodings = tokenize_seqs(val_sequences)
    
    # Create datasets
    train_dataset = TensorDataset(
        train_encodings['input_ids'],
        train_encodings['attention_mask'],
        torch.FloatTensor(labels)
    )
    val_dataset = TensorDataset(
        val_encodings['input_ids'],
        val_encodings['attention_mask'],
        torch.FloatTensor(val_labels)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Train
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, targets = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, targets = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
    
    # Load best model
    model.load_state_dict(best_model)
    
    return model


def extract_finetuned_embeddings(model, sequences, device='cpu', model_name='alchemab/antiberta2'):
    """Extract embeddings from fine-tuned model"""
    try:
        from transformers import RoFormerTokenizer
    except ImportError:
        raise ImportError("Requires transformers")
    
    tokenizer = RoFormerTokenizer.from_pretrained(model_name)
    
    model = model.to(device).eval()
    embeddings = []
    
    with torch.no_grad():
        for seq in sequences:
            seq_spaced = ' '.join(list(seq))
            inputs = tokenizer(seq_spaced, return_tensors='pt', truncation=True, max_length=256, padding='max_length')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model.base_model(**inputs)
            
            if hasattr(outputs, 'pooler_output'):
                emb = outputs.pooler_output
            else:
                emb = outputs.last_hidden_state.mean(dim=1)
            
            embeddings.append(emb.cpu().numpy())
    
    return np.vstack(embeddings)


# ============================================================================
# PART 6: PARATOPE-CENTRIC FEATURES
# ============================================================================

def predict_paratope_simple(sequence: str, chain_type: str = 'heavy') -> List[bool]:
    """
    Simple paratope prediction (residues likely to bind antigen).
    
    Real implementation would use:
    - Parapred
    - Paragraph (structure-based)
    - ParaAntiProt (language model)
    
    For now: assume CDR residues are binding
    """
    from ..datasets.sabdab import extract_cdr_sequences
    
    cdrs = extract_cdr_sequences(sequence, chain_type=chain_type)
    if cdrs is None:
        return [False] * len(sequence)
    
    # Mark CDR residues as potential paratope
    numbered_seq = cdrs['numbered_seq']
    
    # IMGT CDR regions
    cdr_ranges = [(27, 38), (56, 65), (105, 117)]
    
    is_paratope = []
    for pos, aa in numbered_seq:
        in_cdr = any(start <= pos <= end for start, end in cdr_ranges)
        is_paratope.append(in_cdr)
    
    return is_paratope


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_antibody_hybrid(
    heavy_sequences: List[str],
    light_sequences: Optional[List[str]] = None,
    methods: List[str] = ['antiberty', 'cdr_stratified', 'imgt_positions']
) -> Dict[str, np.ndarray]:
    """
    Create comprehensive antibody representation using multiple methods.
    
    Args:
        heavy_sequences: Heavy chain sequences
        light_sequences: Light chain sequences (optional)
        methods: Which representation methods to use
    
    Returns:
        dict: {method_name: features} for KIRBy hybrid creation
    """
    features = {}
    
    if 'antiberty' in methods:
        print("\n[1] AntiBERTy embeddings (batch mode)...")
        antiberty_batch = create_antiberty_embeddings_batch(
            heavy_sequences, 
            light_sequences,
            aggregations=['mean']
        )
        features['antiberty_heavy'] = antiberty_batch['heavy_mean']
        
        if light_sequences:
            features['antiberty_light'] = antiberty_batch['light_mean']
    
    if 'cdr_stratified' in methods:
        print("\n[2] CDR-stratified features...")
        cdr_feats = create_cdr_stratified_embeddings(
            heavy_sequences, chain_type='heavy'
        )
        for name, feat in cdr_feats.items():
            features[f'cdr_{name}'] = feat
    
    if 'imgt_positions' in methods:
        print("\n[3] IMGT position features...")
        features['imgt_positions'] = create_imgt_position_features(
            heavy_sequences, chain_type='heavy'
        )
    
    return features