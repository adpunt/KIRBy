"""
Nucleotide Representation Registry (DNA/RNA Genomics Domain)

All rep functions follow: f(sequences: List[str]) -> np.ndarray(n, d).

Classical Features (Phase G0):
  kmer3 (64D), kmer4 (256D), kmer5 (1024D), kmer6 (4096D),
  gc_content (5D), dinuc_freq (16D), tetranuc_freq (256D),
  tnf_canonical (136D), dna_shape (8D), complexity (5D),
  rna_structure (6D), codon_usage (61D)

Foundation Models (Phase G1/G2 — stubs):
  dnabert2 (768D), nt_v2_250m (768D), nt_v2_500m (1024D),
  hyenadna (256D), caduceus (256D), gena_lm (768D), gpn (512D),
  dnabert_s (768D), splicebert (512D), rnafm (640D), rinalmo (1280D)
"""

import numpy as np
from itertools import product
from collections import Counter
from typing import List, Callable, Dict


# =============================================================================
# CLASSICAL FEATURES
# =============================================================================

_BASE_MAP = np.full(256, 255, dtype=np.uint8)
for _ci, _ch in enumerate('ACGT'):
    _BASE_MAP[ord(_ch)] = _ci
    _BASE_MAP[ord(_ch.lower())] = _ci
_BASE_MAP[ord('U')] = _BASE_MAP[ord('T')]
_BASE_MAP[ord('u')] = _BASE_MAP[ord('T')]


def kmer_frequencies(sequences: List[str], k: int = 4) -> np.ndarray:
    """Normalized k-mer frequency vectors.

    Uses vectorized base-4 encoding — ~50x faster than dict lookups.

    Args:
        sequences: DNA/RNA sequences
        k: k-mer length

    Returns:
        np.ndarray: (n, 4^k) normalized frequency vectors
    """
    n_features = 4 ** k
    powers = np.array([4 ** (k - 1 - j) for j in range(k)], dtype=np.int64)
    features = np.zeros((len(sequences), n_features), dtype=np.float32)

    for i, seq in enumerate(sequences):
        arr = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
        coded = _BASE_MAP[arr]
        n = len(arr)
        if n < k:
            continue
        # Build (n-k+1, k) matrix of k-mer base codes
        strides = np.lib.stride_tricks.as_strided(
            coded,
            shape=(n - k + 1, k),
            strides=(coded.strides[0], coded.strides[0]),
        )
        # Mask out k-mers containing invalid bases
        valid_mask = (strides < 4).all(axis=1)
        valid_strides = strides[valid_mask]
        if len(valid_strides) == 0:
            continue
        # Convert k-mer to index via dot product with base-4 powers
        indices = valid_strides.astype(np.int64).dot(powers)
        np.add.at(features[i], indices, 1)
        total = features[i].sum()
        if total > 0:
            features[i] /= total

    return features


def compute_gc_features(sequences: List[str]) -> np.ndarray:
    """GC content features: global GC, GC skew, AT skew, windowed GC mean/std.

    Returns:
        np.ndarray: (n, 5)
    """
    features = np.zeros((len(sequences), 5), dtype=np.float32)

    for i, seq in enumerate(sequences):
        seq = seq.upper().replace('U', 'T')
        n = len(seq)
        if n == 0:
            continue

        g = seq.count('G')
        c = seq.count('C')
        a = seq.count('A')
        t = seq.count('T')

        gc = (g + c) / n
        gc_skew = (g - c) / (g + c) if (g + c) > 0 else 0.0
        at_skew = (a - t) / (a + t) if (a + t) > 0 else 0.0

        # Windowed GC (100bp windows, half-overlapping)
        w = min(100, n)
        if n >= w:
            gc_wins = []
            step = max(1, w // 2)
            for j in range(0, n - w + 1, step):
                win = seq[j:j + w]
                gc_wins.append((win.count('G') + win.count('C')) / w)
            gc_win_mean = float(np.mean(gc_wins))
            gc_win_std = float(np.std(gc_wins))
        else:
            gc_win_mean = gc
            gc_win_std = 0.0

        features[i] = [gc, gc_skew, at_skew, gc_win_mean, gc_win_std]

    return features


def compute_dinuc_freq(sequences: List[str]) -> np.ndarray:
    """All 16 dinucleotide frequencies.

    Returns:
        np.ndarray: (n, 16)
    """
    return kmer_frequencies(sequences, k=2)


def compute_tetranuc_freq(sequences: List[str]) -> np.ndarray:
    """Standard tetranucleotide frequencies (z-scored per sequence).

    Returns:
        np.ndarray: (n, 256)
    """
    raw = kmer_frequencies(sequences, k=4)
    means = raw.mean(axis=1, keepdims=True)
    stds = raw.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return ((raw - means) / stds).astype(np.float32)


def compute_tnf_canonical(sequences: List[str]) -> np.ndarray:
    """Reverse-complement collapsed tetranucleotide frequencies (136D).

    Merges each 4-mer with its reverse complement. Standard in metagenomics.

    Returns:
        np.ndarray: (n, 136)
    """
    # Build mapping from 4-mer index (0-255) to canonical index (0-135)
    comp_base = np.array([3, 2, 1, 0], dtype=np.uint8)  # A->T, C->G, G->C, T->A
    powers4 = np.array([64, 16, 4, 1], dtype=np.int64)

    # For each of 256 4-mers, compute its canonical index
    kmer_to_canonical = np.zeros(256, dtype=np.int32)
    canonical_map = {}
    n_canonical = 0
    for idx in range(256):
        bases = np.array([(idx >> (2 * (3 - j))) & 3 for j in range(4)], dtype=np.uint8)
        rc_bases = comp_base[bases[::-1]]
        rc_idx = int(rc_bases.dot(powers4))
        canon_idx = min(idx, rc_idx)
        if canon_idx not in canonical_map:
            canonical_map[canon_idx] = n_canonical
            n_canonical += 1
        kmer_to_canonical[idx] = canonical_map[canon_idx]

    assert n_canonical == 136

    features = np.zeros((len(sequences), 136), dtype=np.float32)

    for i, seq in enumerate(sequences):
        arr = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
        coded = _BASE_MAP[arr]
        n = len(arr)
        if n < 4:
            continue
        strides = np.lib.stride_tricks.as_strided(
            coded, shape=(n - 3, 4), strides=(coded.strides[0], coded.strides[0]))
        valid_mask = (strides < 4).all(axis=1)
        valid_strides = strides[valid_mask]
        if len(valid_strides) == 0:
            continue
        indices = valid_strides.astype(np.int64).dot(powers4)
        canonical_indices = kmer_to_canonical[indices]
        np.add.at(features[i], canonical_indices, 1)
        total = features[i].sum()
        if total > 0:
            features[i] /= total

    return features


def compute_dna_shape(sequences: List[str]) -> np.ndarray:
    """DNA shape features: [mean, std] x 4 structural parameters.

    Parameters: Minor Groove Width, Propeller Twist, Helix Twist, Roll.
    Uses dinucleotide lookup tables (Rohs et al., 2009).

    Returns:
        np.ndarray: (n, 8)
    """
    # Dinucleotide-level shape parameter lookup tables
    MGW = {
        'AA': 3.38, 'AC': 4.39, 'AG': 3.92, 'AT': 3.38,
        'CA': 4.71, 'CC': 4.41, 'CG': 4.71, 'CT': 3.92,
        'GA': 4.41, 'GC': 4.65, 'GG': 4.41, 'GT': 4.39,
        'TA': 5.31, 'TC': 4.41, 'TG': 4.71, 'TT': 3.38,
    }
    PROT = {
        'AA': -18.66, 'AC': -13.10, 'AG': -14.00, 'AT': -15.01,
        'CA': -9.45, 'CC': -8.27, 'CG': -10.03, 'CT': -14.00,
        'GA': -13.48, 'GC': -11.08, 'GG': -8.27, 'GT': -13.10,
        'TA': -11.85, 'TC': -13.48, 'TG': -9.45, 'TT': -18.66,
    }
    HELT = {
        'AA': 35.62, 'AC': 34.40, 'AG': 27.70, 'AT': 31.50,
        'CA': 34.50, 'CC': 33.67, 'CG': 29.80, 'CT': 27.70,
        'GA': 36.90, 'GC': 40.00, 'GG': 33.67, 'GT': 34.40,
        'TA': 36.00, 'TC': 36.90, 'TG': 34.50, 'TT': 35.62,
    }
    ROLL = {
        'AA': -0.57, 'AC': 0.09, 'AG': 0.78, 'AT': -1.84,
        'CA': 1.43, 'CC': 0.33, 'CG': 4.15, 'CT': 0.78,
        'GA': 0.07, 'GC': 0.26, 'GG': 0.33, 'GT': 0.09,
        'TA': 3.39, 'TC': 0.07, 'TG': 1.43, 'TT': -0.57,
    }

    params = [MGW, PROT, HELT, ROLL]
    features = np.zeros((len(sequences), 8), dtype=np.float32)

    for i, seq in enumerate(sequences):
        seq = seq.upper().replace('U', 'T')
        if len(seq) < 2:
            continue
        for p, param in enumerate(params):
            values = []
            for j in range(len(seq) - 1):
                dinuc = seq[j:j + 2]
                if dinuc in param:
                    values.append(param[dinuc])
            if values:
                features[i, 2 * p] = np.mean(values)
                features[i, 2 * p + 1] = np.std(values)

    return features


def compute_complexity(sequences: List[str]) -> np.ndarray:
    """Sequence complexity metrics.

    Features: Shannon entropy, dinucleotide entropy, trinucleotide entropy,
    linguistic complexity, alphabet usage fraction.

    Returns:
        np.ndarray: (n, 5)
    """
    features = np.zeros((len(sequences), 5), dtype=np.float32)
    powers2 = np.array([4, 1], dtype=np.int64)
    powers3 = np.array([16, 4, 1], dtype=np.int64)

    for i, seq in enumerate(sequences):
        arr = np.frombuffer(seq.upper().encode('ascii'), dtype=np.uint8)
        coded = _BASE_MAP[arr]
        n = len(arr)
        if n < 3:
            continue

        valid = coded < 4

        # Shannon entropy of base frequencies
        base_counts = np.bincount(coded[valid], minlength=4)[:4]
        total = base_counts.sum()
        if total == 0:
            continue
        probs = base_counts[base_counts > 0] / total
        shannon = -float(np.sum(probs * np.log2(probs)))

        # Dinucleotide entropy (using base-4 indices)
        di_valid = valid[:-1] & valid[1:]
        if di_valid.any():
            strides2 = np.lib.stride_tricks.as_strided(
                coded, shape=(n - 1, 2), strides=(coded.strides[0],) * 2)
            di_idx = strides2[di_valid].astype(np.int64).dot(powers2)
            di_counts = np.bincount(di_idx, minlength=16)
            di_total = di_counts.sum()
            di_probs = di_counts[di_counts > 0] / di_total
            dinuc_entropy = -float(np.sum(di_probs * np.log2(di_probs)))
        else:
            dinuc_entropy = 0.0

        # Trinucleotide entropy + linguistic complexity
        tri_valid = valid[:-2] & valid[1:-1] & valid[2:]
        if tri_valid.any():
            strides3 = np.lib.stride_tricks.as_strided(
                coded, shape=(n - 2, 3), strides=(coded.strides[0], coded.strides[0]))
            tri_idx = strides3[tri_valid].astype(np.int64).dot(powers3)
            tri_counts = np.bincount(tri_idx, minlength=64)
            tri_total = tri_counts.sum()
            tri_probs = tri_counts[tri_counts > 0] / tri_total
            trinuc_entropy = -float(np.sum(tri_probs * np.log2(tri_probs)))
            n_unique_tri = int((tri_counts > 0).sum())
            max_possible = min(64, int(tri_valid.sum()))
            lc = n_unique_tri / max_possible if max_possible > 0 else 0
        else:
            trinuc_entropy = 0.0
            lc = 0.0

        alpha_usage = float((base_counts > 0).sum()) / 4.0

        features[i] = [shannon, dinuc_entropy, trinuc_entropy, lc, alpha_usage]

    return features


def compute_rna_structure(sequences: List[str]) -> np.ndarray:
    """RNA secondary structure features via ViennaRNA.

    Features: MFE, ensemble energy, diversity, mean BPP, frac paired, MFE/nt.

    Returns:
        np.ndarray: (n, 6)
    """
    try:
        import RNA
    except ImportError:
        raise ImportError(
            "RNA structure features require ViennaRNA.\n"
            "Install: conda install -c bioconda viennarna\n"
            "Or: pip install ViennaRNA"
        )

    features = np.zeros((len(sequences), 6), dtype=np.float32)

    for i, seq in enumerate(sequences):
        seq = seq.upper().replace('T', 'U')
        n = len(seq)
        if n < 5:
            continue
        if n > 2000:
            seq = seq[:2000]
            n = 2000

        try:
            structure, mfe = RNA.fold(seq)

            fc = RNA.fold_compound(seq)
            _, ensemble_energy = fc.pf()

            bpp = fc.bpp()
            bp_probs = []
            for j in range(1, n + 1):
                for k in range(j + 1, n + 1):
                    if bpp[j][k] > 0.01:
                        bp_probs.append(bpp[j][k])

            mean_bpp = np.mean(bp_probs) if bp_probs else 0.0
            n_paired = structure.count('(') + structure.count(')')
            frac_paired = n_paired / n
            diversity = ensemble_energy - mfe

            features[i] = [mfe, ensemble_energy, diversity,
                           mean_bpp, frac_paired, mfe / n]
        except Exception:
            pass

    return features


def compute_codon_usage(sequences: List[str]) -> np.ndarray:
    """Relative Synonymous Codon Usage (RSCU) vector.

    For coding sequences. Non-coding or short sequences get zeros.

    Returns:
        np.ndarray: (n, 61)
    """
    CODON_TABLE = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'TAT': 'Y', 'TAC': 'Y', 'TGT': 'C', 'TGC': 'C',
        'TGG': 'W', 'CGT': 'R', 'CGC': 'R', 'CGA': 'R',
        'CGG': 'R', 'AGT': 'S', 'AGC': 'S', 'AGA': 'R',
        'AGG': 'R', 'GGT': 'G', 'GGC': 'G', 'GGA': 'G',
        'GGG': 'G', 'GAT': 'D', 'GAC': 'D', 'GAA': 'E',
        'GAG': 'E', 'AAT': 'N', 'AAC': 'N', 'AAA': 'K',
        'AAG': 'K', 'CAT': 'H', 'CAC': 'H', 'CAA': 'Q',
        'CAG': 'Q',
    }

    aa_codons = {}
    for codon, aa in CODON_TABLE.items():
        aa_codons.setdefault(aa, []).append(codon)

    sense_codons = sorted(CODON_TABLE.keys())
    codon_idx = {c: i for i, c in enumerate(sense_codons)}

    features = np.zeros((len(sequences), 61), dtype=np.float32)

    for i, seq in enumerate(sequences):
        seq = seq.upper().replace('U', 'T')
        codons = [seq[j:j+3] for j in range(0, len(seq) - 2, 3)]
        codons = [c for c in codons if c in CODON_TABLE]

        if len(codons) < 10:
            continue

        counts = Counter(codons)
        for codon, count in counts.items():
            aa = CODON_TABLE[codon]
            n_synonyms = len(aa_codons[aa])
            aa_total = sum(counts.get(c, 0) for c in aa_codons[aa])
            expected = aa_total / n_synonyms if n_synonyms > 0 else 1
            features[i, codon_idx[codon]] = count / expected if expected > 0 else 0

    return features


# =============================================================================
# FACTORY REPS (per-fold, leakage-free)
# =============================================================================

def make_onehot_pca50_factory(sequences: List[str]) -> Callable:
    """Per-fold PCA of one-hot encoding (no data leakage).

    Pre-computes one-hot once, fits PCA on training data only per fold.

    Returns:
        Callable: factory(train_idx, val_idx, train_labels) -> (X_train, X_val)
    """
    from .sequence import create_onehot_encoding

    max_len = min(max(len(s) for s in sequences), 512)
    full_X = create_onehot_encoding(sequences, max_length=max_len, flatten=True)

    def factory(train_idx, val_idx, train_labels):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        X_train = full_X[train_idx]
        X_val = full_X[val_idx]
        nc = min(50, X_train.shape[1], X_train.shape[0])

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        pca = PCA(n_components=nc, random_state=42)
        X_train_pca = pca.fit_transform(X_train_s)
        X_val_pca = pca.transform(X_val_s)

        return X_train_pca.astype(np.float32), X_val_pca.astype(np.float32)

    return factory


# =============================================================================
# FOUNDATION MODEL STUBS (Phase G1/G2)
# =============================================================================

def _lm_stub(name: str, dim: int, install: str = '') -> Callable:
    """Create a stub that raises ImportError for unimplemented LM reps."""
    def stub(sequences: List[str]) -> np.ndarray:
        raise ImportError(
            f"'{name}' embeddings not yet implemented (Phase G1/G2).\n"
            f"Expected output: (n, {dim}) float32.\n"
            + (f"Will require: {install}\n" if install else "")
            + "Use classical reps (--reps fast) for now."
        )
    stub.__name__ = f'embed_{name}'
    return stub


# =============================================================================
# REGISTRY
# =============================================================================

NUCLEOTIDE_REPS: Dict[str, Callable] = {
    # Foundation models (stubs)
    'dnabert2':    _lm_stub('dnabert2', 768, 'transformers'),
    'nt_v2_250m':  _lm_stub('nt_v2_250m', 768, 'transformers'),
    'nt_v2_500m':  _lm_stub('nt_v2_500m', 1024, 'transformers'),
    'hyenadna':    _lm_stub('hyenadna', 256, 'transformers'),
    'caduceus':    _lm_stub('caduceus', 256, 'mamba_ssm (CUDA only)'),
    'gena_lm':     _lm_stub('gena_lm', 768, 'transformers'),
    'gpn':         _lm_stub('gpn', 512, 'gpn'),
    'splicebert':  _lm_stub('splicebert', 512, 'multimolecule'),
    'rnafm':       _lm_stub('rnafm', 640, 'rna-fm'),
    'rinalmo':     _lm_stub('rinalmo', 1280, 'multimolecule, flash-attn (CUDA)'),
    'dnabert_s':   _lm_stub('dnabert_s', 768, 'transformers'),
    # Classical features
    'kmer3':          lambda seqs: kmer_frequencies(seqs, 3),     # 64D
    'kmer4':          lambda seqs: kmer_frequencies(seqs, 4),     # 256D
    'kmer5':          lambda seqs: kmer_frequencies(seqs, 5),     # 1024D
    'kmer6':          lambda seqs: kmer_frequencies(seqs, 6),     # 4096D
    'gc_content':     compute_gc_features,                        # 5D
    'dinuc_freq':     compute_dinuc_freq,                         # 16D
    'tetranuc_freq':  compute_tetranuc_freq,                      # 256D
    'tnf_canonical':  compute_tnf_canonical,                      # 136D
    'dna_shape':      compute_dna_shape,                          # 8D
    'complexity':     compute_complexity,                         # 5D
    'rna_structure':  compute_rna_structure,                      # 6D
    'codon_usage':    compute_codon_usage,                        # 61D
}

NUCLEOTIDE_LM_REPS = {
    'dnabert2', 'nt_v2_250m', 'nt_v2_500m', 'hyenadna', 'caduceus',
    'gena_lm', 'gpn', 'splicebert', 'rnafm', 'rinalmo', 'dnabert_s',
}

NUCLEOTIDE_FAST_REPS = {
    'kmer4', 'kmer6', 'gc_content', 'dinuc_freq',
    'tetranuc_freq', 'complexity',
}

NUCLEOTIDE_CLASSICAL_REPS = {
    'kmer3', 'kmer4', 'kmer5', 'kmer6', 'gc_content',
    'dinuc_freq', 'tetranuc_freq', 'tnf_canonical',
    'dna_shape', 'complexity', 'codon_usage',
}

NUCLEOTIDE_FACTORY_REPS = {
    'onehot_pca50',
}
