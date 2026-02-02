"""
Sequence Representation Generators (DNA/RNA)

This module provides representations for nucleotide sequences for the
KIRBy (Knowledge Integration by Representation Borrowing) framework.

Supports DNA and RNA sequences for genomics, transcriptomics, and related tasks.

Available Representations:
=========================

PRETRAINED (Frozen):
- Nucleotide Transformer: InstaDeepAI transformer trained on multi-species DNA
- DNABERT-2: BERT-style model for DNA sequences
- HyenaDNA: Long-range DNA model using Hyena operator
- Caduceus: Mamba-based bidirectional DNA model

AUGMENTATIONS (supplementary features):
- k-mer frequencies: Count distribution of k-length subsequences
- GC content: Guanine-Cytosine ratio and positional features
- Sequence complexity: Entropy, repeat content, linguistic complexity
"""

import numpy as np
from typing import List, Optional, Dict, Union
from collections import Counter


# =============================================================================
# CONSTANTS
# =============================================================================

DNA_ALPHABET = set('ACGT')
RNA_ALPHABET = set('ACGU')
AMBIGUOUS_DNA = set('ACGTRYSWKMBDHVN')
AMBIGUOUS_RNA = set('ACGURYSWKMBDHVN')


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _clean_sequence(seq: str, alphabet: set = DNA_ALPHABET, replace_char: str = 'N') -> str:
    """Clean sequence by replacing invalid characters."""
    seq = seq.upper()
    cleaned = []
    for char in seq:
        if char in alphabet or char == replace_char:
            cleaned.append(char)
        elif char in 'ACGTU':
            # Handle T/U conversion
            if 'U' in alphabet and char == 'T':
                cleaned.append('U')
            elif 'T' in alphabet and char == 'U':
                cleaned.append('T')
            else:
                cleaned.append(char)
        else:
            cleaned.append(replace_char)
    return ''.join(cleaned)


def _detect_sequence_type(sequences: List[str]) -> str:
    """Detect if sequences are DNA or RNA based on T/U content."""
    t_count = sum(seq.upper().count('T') for seq in sequences[:100])
    u_count = sum(seq.upper().count('U') for seq in sequences[:100])
    return 'RNA' if u_count > t_count else 'DNA'


# =============================================================================
# PRETRAINED REPRESENTATIONS (FROZEN)
# =============================================================================

# Global caches for models
_NT_MODEL = None
_NT_TOKENIZER = None
_DNABERT2_MODEL = None
_DNABERT2_TOKENIZER = None
_HYENADNA_MODEL = None
_HYENADNA_TOKENIZER = None
_CADUCEUS_MODEL = None
_CADUCEUS_TOKENIZER = None


def create_nucleotide_transformer(
    sequences: List[str],
    batch_size: int = 8,
    max_length: int = 1000,
    device: str = None
) -> np.ndarray:
    """
    Create embeddings using Nucleotide Transformer v2.
    
    Model: InstaDeepAI/nucleotide-transformer-v2-500m-multi-species
    Trained on DNA from 850 species. Handles sequences up to 12kb.
    
    Args:
        sequences: List of DNA sequences
        batch_size: Batch size for encoding (default: 8, lower for long seqs)
        max_length: Maximum sequence length in tokens (default: 1000)
        device: 'cpu' or 'cuda' (default: auto-detect)
        
    Returns:
        np.ndarray: Embeddings (n_sequences, 1280)
    """
    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError:
        raise ImportError("Nucleotide Transformer requires: pip install torch transformers")
    
    global _NT_MODEL, _NT_TOKENIZER
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model once
    if _NT_MODEL is None:
        print("Loading Nucleotide Transformer v2 (500M)...")
        model_name = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
        _NT_TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        _NT_MODEL = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        _NT_MODEL = _NT_MODEL.to(device).eval()
    
    model = _NT_MODEL
    tokenizer = _NT_TOKENIZER
    
    print(f"Encoding {len(sequences)} sequences with Nucleotide Transformer...")
    
    # Clean sequences
    cleaned = [_clean_sequence(seq, DNA_ALPHABET) for seq in sequences]
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i:i + batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Get hidden states
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Use last hidden state, mean pool over sequence
            hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
            
            # Mask padding tokens
            mask = attention_mask.unsqueeze(-1).float()
            masked_hidden = hidden_states * mask
            summed = masked_hidden.sum(dim=1)
            counts = mask.sum(dim=1)
            embeddings = summed / counts.clamp(min=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
            
            if (i + batch_size) % 100 == 0:
                print(f"  Progress: {min(i + batch_size, len(sequences))}/{len(sequences)}")
    
    return np.vstack(all_embeddings).astype(np.float32)


def create_dnabert2(
    sequences: List[str],
    batch_size: int = 16,
    max_length: int = 512,
    device: str = None
) -> np.ndarray:
    """
    Create embeddings using DNABERT-2.

    Model: zhihan1996/DNABERT-2-117M
    Uses Byte Pair Encoding (BPE) tokenization, no k-mer preprocessing needed.
    Trained on multi-species genome data.

    Args:
        sequences: List of DNA sequences
        batch_size: Batch size for encoding (default: 16)
        max_length: Maximum sequence length in tokens (default: 512)
        device: 'cpu' or 'cuda' (default: auto-detect)

    Returns:
        np.ndarray: Embeddings (n_sequences, 768)
    """
    try:
        import torch
        from transformers import AutoTokenizer, BertModel, BertConfig
    except ImportError:
        raise ImportError("DNABERT-2 requires: pip install torch transformers")

    global _DNABERT2_MODEL, _DNABERT2_TOKENIZER

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model once
    if _DNABERT2_MODEL is None:
        print("Loading DNABERT-2 (117M)...")
        model_name = "zhihan1996/DNABERT-2-117M"

        # Load tokenizer with trust_remote_code
        _DNABERT2_TOKENIZER = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load config and model separately to avoid config class mismatch
        # DNABERT-2 uses a custom BertConfig but we can load weights into standard BertModel
        try:
            from transformers import AutoModel
            _DNABERT2_MODEL = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                ignore_mismatched_sizes=True
            )
        except ValueError:
            # Fallback: load config first, then model with standard BertModel
            print("  Using fallback loading method...")
            import json
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(repo_id=model_name, filename="config.json")
            with open(config_path) as f:
                config_dict = json.load(f)

            # Create standard BertConfig from the downloaded config
            config = BertConfig(**{k: v for k, v in config_dict.items()
                                   if k in BertConfig().to_dict()})

            _DNABERT2_MODEL = BertModel.from_pretrained(
                model_name,
                config=config,
                ignore_mismatched_sizes=True
            )

        _DNABERT2_MODEL = _DNABERT2_MODEL.to(device).eval()
        print(f"  Loaded on {device}")

    model = _DNABERT2_MODEL
    tokenizer = _DNABERT2_TOKENIZER

    print(f"Encoding {len(sequences)} sequences with DNABERT-2...")

    # Clean sequences
    cleaned = [_clean_sequence(seq, DNA_ALPHABET) for seq in sequences]

    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i:i + batch_size]

            # Tokenize
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            # Get embeddings
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Mean pooling over sequence (excluding padding)
            hidden_states = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            masked_hidden = hidden_states * mask
            summed = masked_hidden.sum(dim=1)
            counts = mask.sum(dim=1)
            embeddings = summed / counts.clamp(min=1)

            all_embeddings.append(embeddings.cpu().numpy())

            if (i + batch_size) % 200 == 0:
                print(f"  Progress: {min(i + batch_size, len(sequences))}/{len(sequences)}")

    return np.vstack(all_embeddings).astype(np.float32)


def create_hyenadna(
    sequences: List[str],
    batch_size: int = 8,
    max_length: int = 1024,
    device: str = None
) -> np.ndarray:
    """
    Create embeddings using HyenaDNA.

    Model: LongSafari/hyenadna-small-32k-seqlen-hf
    Uses Hyena operator for efficient long-range sequence modeling.

    REQUIREMENTS:
        pip install torch transformers>=4.35.0

    Args:
        sequences: List of DNA sequences
        batch_size: Batch size for encoding (default: 8)
        max_length: Maximum sequence length (default: 1024, model supports up to 32k)
        device: 'cpu' or 'cuda' (default: auto-detect)

    Returns:
        np.ndarray: Embeddings (n_sequences, 256)
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError("HyenaDNA requires: pip install torch transformers>=4.35.0")

    global _HYENADNA_MODEL, _HYENADNA_TOKENIZER

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model once
    if _HYENADNA_MODEL is None:
        print("Loading HyenaDNA (small-32k)...")

        # Use the HuggingFace-compatible version of HyenaDNA
        model_name = "LongSafari/hyenadna-small-32k-seqlen-hf"

        try:
            _HYENADNA_TOKENIZER = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side='right'
            )

            _HYENADNA_MODEL = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )

        except Exception as e:
            # Provide helpful error message
            raise ImportError(
                f"HyenaDNA loading failed: {e}\n\n"
                f"HyenaDNA uses a custom architecture. Options:\n"
                f"  1. Ensure transformers>=4.35.0: pip install -U transformers\n"
                f"  2. Use Nucleotide Transformer instead (reliable, similar quality):\n"
                f"     embeddings = create_nucleotide_transformer(sequences)\n"
                f"  3. Use DNABERT-2 (also works well):\n"
                f"     embeddings = create_dnabert2(sequences)"
            )

        _HYENADNA_MODEL = _HYENADNA_MODEL.to(device).eval()
        print(f"  Loaded on {device}")

        # Add pad token if missing
        if _HYENADNA_TOKENIZER.pad_token is None:
            _HYENADNA_TOKENIZER.pad_token = _HYENADNA_TOKENIZER.eos_token
            if _HYENADNA_TOKENIZER.pad_token is None:
                _HYENADNA_TOKENIZER.add_special_tokens({'pad_token': '[PAD]'})

    model = _HYENADNA_MODEL
    tokenizer = _HYENADNA_TOKENIZER

    print(f"Encoding {len(sequences)} sequences with HyenaDNA...")

    # Clean sequences
    cleaned = [_clean_sequence(seq, DNA_ALPHABET) for seq in sequences]

    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i:i + batch_size]

            # Tokenize
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(device)

            # Get embeddings - HyenaDNA outputs hidden states
            outputs = model(input_ids=input_ids, output_hidden_states=True)

            # Get last hidden state
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]
            elif hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs[0]

            # Mean pooling
            embeddings = hidden_states.mean(dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

            if (i + batch_size) % 50 == 0:
                print(f"  Progress: {min(i + batch_size, len(sequences))}/{len(sequences)}")

    return np.vstack(all_embeddings).astype(np.float32)


def create_caduceus(
    sequences: List[str],
    batch_size: int = 8,
    max_length: int = 131072,
    device: str = None
) -> np.ndarray:
    """
    Create embeddings using Caduceus (Mamba-based DNA model).

    Model: kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16
    Bidirectional Mamba architecture for DNA. Can handle very long sequences (131k).

    REQUIREMENTS:
        pip install torch transformers mamba-ssm causal-conv1d

    Note: mamba-ssm requires CUDA. For CPU-only, use Nucleotide Transformer instead.

    Args:
        sequences: List of DNA sequences
        batch_size: Batch size for encoding (default: 8)
        max_length: Maximum sequence length (default: 131072)
        device: 'cpu' or 'cuda' (default: auto-detect)

    Returns:
        np.ndarray: Embeddings (n_sequences, 256)
    """
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("Caduceus requires: pip install torch transformers")

    # Check for mamba-ssm before attempting to load model
    try:
        import mamba_ssm
    except ImportError:
        raise ImportError(
            "Caduceus requires the mamba-ssm package (Mamba state-space model).\n"
            "This is a pip package, NOT the mamba package manager.\n\n"
            "Install with:\n"
            "  pip install mamba-ssm causal-conv1d\n\n"
            "Note: mamba-ssm requires CUDA. For CPU-only environments, use:\n"
            "  - Nucleotide Transformer (create_nucleotide_transformer)\n"
            "  - DNABERT-2 (create_dnabert2)\n"
            "These work on CPU and provide similar quality embeddings."
        )

    global _CADUCEUS_MODEL, _CADUCEUS_TOKENIZER

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Caduceus/Mamba works best on CUDA
    if device == 'cpu':
        import warnings
        warnings.warn(
            "Caduceus uses Mamba architecture which is optimized for CUDA. "
            "CPU inference may be slow. Consider using Nucleotide Transformer instead."
        )

    # Load model once
    if _CADUCEUS_MODEL is None:
        print("Loading Caduceus (Mamba-based, 131k context)...")
        model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"

        try:
            from transformers import AutoModel

            _CADUCEUS_TOKENIZER = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            _CADUCEUS_MODEL = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            _CADUCEUS_MODEL = _CADUCEUS_MODEL.to(device).eval()
            print(f"  Loaded on {device}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load Caduceus model: {e}\n\n"
                f"Ensure you have installed:\n"
                f"  pip install mamba-ssm causal-conv1d\n\n"
                f"If on CPU-only, use Nucleotide Transformer instead."
            )

    model = _CADUCEUS_MODEL
    tokenizer = _CADUCEUS_TOKENIZER

    print(f"Encoding {len(sequences)} sequences with Caduceus...")

    # Clean sequences
    cleaned = [_clean_sequence(seq, DNA_ALPHABET) for seq in sequences]

    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i:i + batch_size]

            # Tokenize
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(device)

            # Get embeddings
            outputs = model(input_ids=input_ids, output_hidden_states=True)

            # Get last hidden state
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]
            elif hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs[0]

            # Mean pooling
            embeddings = hidden_states.mean(dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

            if (i + batch_size) % 50 == 0:
                print(f"  Progress: {min(i + batch_size, len(sequences))}/{len(sequences)}")

    return np.vstack(all_embeddings).astype(np.float32)


# =============================================================================
# STATIC REPRESENTATIONS
# =============================================================================

def create_kmer_features(
    sequences: List[str],
    k: int = 6,
    normalize: bool = True,
    include_lower_k: bool = True
) -> np.ndarray:
    """
    Create k-mer frequency features from sequences.
    
    Counts occurrences of all possible k-mers. Can also include lower-order
    k-mers (1 to k-1) for a multi-scale representation.
    
    Args:
        sequences: List of DNA/RNA sequences
        k: Maximum k-mer length (default: 6)
        normalize: Normalize counts to frequencies (default: True)
        include_lower_k: Include k-mers from 1 to k-1 (default: True)
        
    Returns:
        np.ndarray: k-mer features (n_sequences, n_kmers)
                    n_kmers = 4^k if not include_lower_k
                    n_kmers = sum(4^i for i in 1..k) if include_lower_k
    """
    # Detect sequence type
    seq_type = _detect_sequence_type(sequences)
    alphabet = 'ACGT' if seq_type == 'DNA' else 'ACGU'
    
    # Determine which k values to use
    if include_lower_k:
        k_values = list(range(1, k + 1))
    else:
        k_values = [k]
    
    # Build vocabulary of all possible k-mers
    from itertools import product
    
    vocab = {}
    idx = 0
    for kv in k_values:
        for kmer in product(alphabet, repeat=kv):
            kmer_str = ''.join(kmer)
            vocab[kmer_str] = idx
            idx += 1
    
    n_features = len(vocab)
    print(f"Creating {k}-mer features ({n_features} features, include_lower={include_lower_k})...")
    
    features = np.zeros((len(sequences), n_features), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        seq = seq.upper()
        # Convert U to T or T to U based on detected type
        if seq_type == 'DNA':
            seq = seq.replace('U', 'T')
        else:
            seq = seq.replace('T', 'U')
        
        # Count k-mers for each k value
        for kv in k_values:
            for j in range(len(seq) - kv + 1):
                kmer = seq[j:j + kv]
                if kmer in vocab:
                    features[i, vocab[kmer]] += 1
        
        # Normalize
        if normalize:
            total = features[i].sum()
            if total > 0:
                features[i] /= total
    
    return features


def create_onehot_encoding(
    sequences: List[str],
    max_length: int = None,
    flatten: bool = True
) -> np.ndarray:
    """
    Create one-hot encoding of sequences.
    
    Args:
        sequences: List of DNA/RNA sequences
        max_length: Pad/truncate to this length (default: max sequence length)
        flatten: Flatten to 2D array (default: True)
        
    Returns:
        np.ndarray: One-hot encoding
                    If flatten: (n_sequences, max_length * 4)
                    If not flatten: (n_sequences, max_length, 4)
    """
    seq_type = _detect_sequence_type(sequences)
    alphabet = 'ACGT' if seq_type == 'DNA' else 'ACGU'
    char_to_idx = {c: i for i, c in enumerate(alphabet)}
    
    if max_length is None:
        max_length = max(len(s) for s in sequences)
    
    n_sequences = len(sequences)
    encoding = np.zeros((n_sequences, max_length, 4), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        seq = seq.upper()
        if seq_type == 'DNA':
            seq = seq.replace('U', 'T')
        else:
            seq = seq.replace('T', 'U')
        
        for j, char in enumerate(seq[:max_length]):
            if char in char_to_idx:
                encoding[i, j, char_to_idx[char]] = 1.0
    
    if flatten:
        return encoding.reshape(n_sequences, -1)
    return encoding


# =============================================================================
# AUGMENTATIONS (supplementary features)
# =============================================================================

def compute_gc_content(sequences: List[str]) -> np.ndarray:
    """
    Compute GC content and related features.
    
    Features computed (per sequence):
    - gc_content: Overall GC ratio
    - gc_skew: (G - C) / (G + C), indicator of strand asymmetry
    - at_content: AT ratio (complement of GC)
    - purine_content: (A + G) ratio
    - gc_variance: Variance of GC in sliding windows
    - gc_max_window: Max GC in any 100bp window
    - gc_min_window: Min GC in any 100bp window
    
    Args:
        sequences: List of DNA/RNA sequences
        
    Returns:
        np.ndarray: GC features (n_sequences, 7)
    """
    features = []
    
    for seq in sequences:
        seq = seq.upper().replace('U', 'T')
        length = len(seq)
        
        if length == 0:
            features.append(np.zeros(7, dtype=np.float32))
            continue
        
        # Count bases
        g_count = seq.count('G')
        c_count = seq.count('C')
        a_count = seq.count('A')
        t_count = seq.count('T')
        
        gc_count = g_count + c_count
        at_count = a_count + t_count
        
        # Basic ratios
        gc_content = gc_count / length if length > 0 else 0
        at_content = at_count / length if length > 0 else 0
        
        # GC skew
        gc_skew = (g_count - c_count) / gc_count if gc_count > 0 else 0
        
        # Purine content
        purine_content = (a_count + g_count) / length if length > 0 else 0
        
        # Sliding window GC (100bp windows)
        window_size = min(100, length)
        if length >= window_size:
            gc_windows = []
            for i in range(length - window_size + 1):
                window = seq[i:i + window_size]
                window_gc = (window.count('G') + window.count('C')) / window_size
                gc_windows.append(window_gc)
            
            gc_variance = np.var(gc_windows) if gc_windows else 0
            gc_max = max(gc_windows) if gc_windows else gc_content
            gc_min = min(gc_windows) if gc_windows else gc_content
        else:
            gc_variance = 0
            gc_max = gc_content
            gc_min = gc_content
        
        features.append(np.array([
            gc_content,
            gc_skew,
            at_content,
            purine_content,
            gc_variance,
            gc_max,
            gc_min
        ], dtype=np.float32))
    
    return np.array(features)


def compute_sequence_complexity(sequences: List[str]) -> np.ndarray:
    """
    Compute sequence complexity metrics.
    
    Features computed (per sequence):
    - shannon_entropy: Information entropy of base distribution
    - dinucleotide_entropy: Entropy of dinucleotide distribution
    - linguistic_complexity: Ratio of observed to possible k-mers
    - repeat_fraction: Fraction of sequence in simple repeats
    - longest_homopolymer: Length of longest single-base run
    - compression_ratio: Approximate compressibility (lower = more repetitive)
    - cpg_ratio: Observed/expected CpG dinucleotide ratio
    - unique_kmer_ratio: Ratio of unique 6-mers to total 6-mers
    
    Args:
        sequences: List of DNA/RNA sequences
        
    Returns:
        np.ndarray: Complexity features (n_sequences, 8)
    """
    features = []
    
    for seq in sequences:
        seq = seq.upper().replace('U', 'T')
        length = len(seq)
        
        if length < 2:
            features.append(np.zeros(8, dtype=np.float32))
            continue
        
        # Shannon entropy of bases
        base_counts = Counter(seq)
        total = sum(base_counts.values())
        probs = [count / total for count in base_counts.values() if count > 0]
        shannon_entropy = -sum(p * np.log2(p) for p in probs) if probs else 0
        
        # Dinucleotide entropy
        dinucs = [seq[i:i+2] for i in range(length - 1)]
        dinuc_counts = Counter(dinucs)
        total_dinucs = sum(dinuc_counts.values())
        dinuc_probs = [count / total_dinucs for count in dinuc_counts.values() if count > 0]
        dinuc_entropy = -sum(p * np.log2(p) for p in dinuc_probs) if dinuc_probs else 0
        
        # Linguistic complexity (using 3-mers)
        k = min(3, length)
        if length >= k:
            kmers = set(seq[i:i+k] for i in range(length - k + 1))
            max_possible = min(4**k, length - k + 1)
            linguistic_complexity = len(kmers) / max_possible if max_possible > 0 else 0
        else:
            linguistic_complexity = 0
        
        # Simple repeat detection
        repeat_patterns = ['AT', 'TA', 'GC', 'CG', 'AA', 'TT', 'GG', 'CC',
                         'AAA', 'TTT', 'GGG', 'CCC', 'ATAT', 'TATA', 'GCGC', 'CGCG']
        repeat_count = 0
        for pattern in repeat_patterns:
            repeat_count += seq.count(pattern * 3) * len(pattern) * 3
        repeat_fraction = min(repeat_count / length, 1.0) if length > 0 else 0
        
        # Longest homopolymer
        max_run = 1
        current_run = 1
        for i in range(1, length):
            if seq[i] == seq[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        longest_homopolymer = max_run
        
        # Compression ratio approximation (unique k-mers / total k-mers)
        k = 4
        if length >= k:
            kmers = [seq[i:i+k] for i in range(length - k + 1)]
            unique_kmers = len(set(kmers))
            compression_ratio = unique_kmers / len(kmers) if kmers else 1
        else:
            compression_ratio = 1
        
        # CpG ratio (observed/expected)
        c_count = seq.count('C')
        g_count = seq.count('G')
        cpg_count = seq.count('CG')
        expected_cpg = (c_count * g_count) / length if length > 0 else 0
        cpg_ratio = cpg_count / expected_cpg if expected_cpg > 0 else 0
        
        # Unique 6-mer ratio
        k = 6
        if length >= k:
            kmers = [seq[i:i+k] for i in range(length - k + 1)]
            unique_kmer_ratio = len(set(kmers)) / len(kmers) if kmers else 0
        else:
            unique_kmer_ratio = 0
        
        features.append(np.array([
            shannon_entropy,
            dinuc_entropy,
            linguistic_complexity,
            repeat_fraction,
            longest_homopolymer,
            compression_ratio,
            cpg_ratio,
            unique_kmer_ratio
        ], dtype=np.float32))
    
    return np.array(features)


def compute_positional_features(
    sequences: List[str],
    n_bins: int = 10
) -> np.ndarray:
    """
    Compute positional base composition features.
    
    Divides each sequence into n_bins and computes base frequencies
    in each bin. Useful for capturing position-dependent patterns.
    
    Features: 4 bases × n_bins = 4 * n_bins features
    
    Args:
        sequences: List of DNA/RNA sequences
        n_bins: Number of positional bins (default: 10)
        
    Returns:
        np.ndarray: Positional features (n_sequences, 4 * n_bins)
    """
    seq_type = _detect_sequence_type(sequences)
    alphabet = 'ACGT' if seq_type == 'DNA' else 'ACGU'
    
    features = []
    
    for seq in sequences:
        seq = seq.upper()
        if seq_type == 'DNA':
            seq = seq.replace('U', 'T')
        else:
            seq = seq.replace('T', 'U')
        
        length = len(seq)
        
        if length == 0:
            features.append(np.zeros(4 * n_bins, dtype=np.float32))
            continue
        
        feat = []
        bin_size = length / n_bins
        
        for bin_idx in range(n_bins):
            start = int(bin_idx * bin_size)
            end = int((bin_idx + 1) * bin_size)
            if bin_idx == n_bins - 1:
                end = length
            
            window = seq[start:end]
            window_len = len(window)
            
            for base in alphabet:
                freq = window.count(base) / window_len if window_len > 0 else 0
                feat.append(freq)
        
        features.append(np.array(feat, dtype=np.float32))
    
    return np.array(features)


def compute_motif_features(
    sequences: List[str],
    motifs: List[str] = None
) -> np.ndarray:
    """
    Count occurrences of specific sequence motifs.
    
    Default motifs include common regulatory elements, splice sites,
    and biologically relevant patterns.
    
    Args:
        sequences: List of DNA/RNA sequences
        motifs: List of motif strings (default: common regulatory motifs)
        
    Returns:
        np.ndarray: Motif counts (n_sequences, n_motifs)
    """
    if motifs is None:
        # Common regulatory and structural motifs
        motifs = [
            # TATA box and variants
            'TATAAA', 'TATAA', 'TATA',
            # GC box
            'GGGCGG', 'CCGCCC',
            # CAAT box
            'CCAAT', 'ATTGG',
            # Splice sites
            'GT', 'AG',  # Donor/acceptor
            'GTAAGT', 'GTRAGT',  # Extended donor
            # Kozak sequence
            'GCCACC', 'ACCATG',
            # CpG
            'CG', 'CGG', 'CCG', 'CGCG',
            # Poly-A
            'AATAAA', 'ATTAAA',
            # Common repeats
            'AAAA', 'TTTT', 'GGGG', 'CCCC',
            'ATAT', 'GCGC', 'TATA',
            # G-quadruplex motif
            'GGG',
        ]
    
    n_motifs = len(motifs)
    features = np.zeros((len(sequences), n_motifs), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        seq = seq.upper().replace('U', 'T')
        
        for j, motif in enumerate(motifs):
            # Handle IUPAC ambiguity codes if present
            motif = motif.replace('R', '[AG]').replace('Y', '[CT]')
            motif = motif.replace('W', '[AT]').replace('S', '[GC]')
            motif = motif.replace('M', '[AC]').replace('K', '[GT]')
            
            if '[' in motif:
                # Regex matching for ambiguous motifs
                import re
                features[i, j] = len(re.findall(motif, seq))
            else:
                features[i, j] = seq.count(motif)
    
    return features


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_sequence_hybrid(
    sequences: List[str],
    methods: List[str] = ['dnabert2', 'kmer', 'gc', 'complexity'],
    kmer_k: int = 6,
    device: str = None
) -> Dict[str, np.ndarray]:
    """
    Create comprehensive sequence representation using multiple methods.
    
    Args:
        sequences: List of DNA/RNA sequences
        methods: Which representation methods to use
            - 'nucleotide_transformer': NT v2 embeddings (1280D)
            - 'dnabert2': DNABERT-2 embeddings (768D)
            - 'hyenadna': HyenaDNA embeddings (256D)
            - 'caduceus': Caduceus/Mamba embeddings (256D)
            - 'kmer': k-mer frequencies
            - 'onehot': One-hot encoding
            - 'gc': GC content features
            - 'complexity': Sequence complexity features
            - 'positional': Positional base composition
            - 'motif': Motif occurrence counts
        kmer_k: k-mer length (default: 6)
        device: 'cpu' or 'cuda' (default: auto-detect)
        
    Returns:
        dict: {method_name: features} for KIRBy hybrid creation
    """
    features = {}
    
    if 'nucleotide_transformer' in methods:
        print("\n[1] Nucleotide Transformer embeddings...")
        features['nucleotide_transformer'] = create_nucleotide_transformer(sequences, device=device)
    
    if 'dnabert2' in methods:
        print("\n[2] DNABERT-2 embeddings...")
        features['dnabert2'] = create_dnabert2(sequences, device=device)
    
    if 'hyenadna' in methods:
        print("\n[3] HyenaDNA embeddings...")
        features['hyenadna'] = create_hyenadna(sequences, device=device)
    
    if 'caduceus' in methods:
        print("\n[4] Caduceus embeddings...")
        features['caduceus'] = create_caduceus(sequences, device=device)
    
    if 'kmer' in methods:
        print(f"\n[5] {kmer_k}-mer features...")
        features['kmer'] = create_kmer_features(sequences, k=kmer_k)
    
    if 'onehot' in methods:
        print("\n[6] One-hot encoding...")
        features['onehot'] = create_onehot_encoding(sequences)
    
    if 'gc' in methods:
        print("\n[7] GC content features...")
        features['gc'] = compute_gc_content(sequences)
    
    if 'complexity' in methods:
        print("\n[8] Sequence complexity features...")
        features['complexity'] = compute_sequence_complexity(sequences)
    
    if 'positional' in methods:
        print("\n[9] Positional features...")
        features['positional'] = compute_positional_features(sequences)
    
    if 'motif' in methods:
        print("\n[10] Motif features...")
        features['motif'] = compute_motif_features(sequences)
    
    return features