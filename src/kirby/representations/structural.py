"""
Structural Context Features for Antibodies

Derives features from predicted 3D structures:
- SASA features (~15D): solvent-accessible surface area analysis
- Confidence features (~5D): prediction quality from B-factor/ensemble RMSD
- Paratope features (~8D): antigen-binding residue analysis
- Combined structural rep (~28D): all concatenated

Structure prediction uses ABodyBuilder2 (via ImmuneBuilder) when available,
otherwise gracefully degrades to sequence-only paratope features.

Dependencies:
- BioPython (required): SASA computation
- ImmuneBuilder (optional, server-recommended): structure prediction
- ANARCI (optional): CDR extraction for paratope
"""

import os
import hashlib
import traceback
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from kirby.representations.protein import AA_PROPERTIES


# ============================================================================
# CACHE
# ============================================================================

# Every structure-prediction failure, in order, as "ExceptionType: message".
# predict_structure returns None on failure and the geometry features come back
# as zeros, so this list is the only record of WHY. Read it with
# prediction_failures() after a batch.
_PREDICTION_FAILURES: List[str] = []


def prediction_failures() -> List[str]:
    """Every structure-prediction failure recorded so far, oldest first."""
    return list(_PREDICTION_FAILURES)


def _structure_cache_dir() -> Path:
    """Get structure cache directory, preferring /Volumes/seagate."""
    seagate = Path('/Volumes/seagate/kirby/structures')
    if seagate.parent.parent.exists():
        seagate.mkdir(parents=True, exist_ok=True)
        return seagate
    cache_root = Path(os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')))
    fallback = cache_root / 'structures'
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _seq_hash(heavy_seq: str, light_seq: Optional[str] = None) -> str:
    """Hash sequence pair for caching."""
    key = heavy_seq + ('|' + light_seq if light_seq else '')
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ============================================================================
# STRUCTURE PREDICTION (ABodyBuilder2 via ImmuneBuilder)
# ============================================================================

_PREDICTOR_CACHE = {}


def predict_structure(
    heavy_seq: str,
    light_seq: Optional[str] = None,
) -> Optional[Path]:
    """Predict antibody 3D structure using ABodyBuilder2.

    Uses ImmuneBuilder's ABodyBuilder2 (~5s on GPU, ~60s on CPU).
    Results are cached to disk. The predictor model is instantiated once
    and reused across calls.

    Args:
        heavy_seq: Heavy chain amino acid sequence.
        light_seq: Light chain sequence (optional, for paired prediction).

    Returns:
        Path to predicted PDB file, or None if prediction failed.
    """
    cache_dir = _structure_cache_dir()
    cache_key = _seq_hash(heavy_seq, light_seq)
    cached_pdb = cache_dir / f'{cache_key}.pdb'

    if cached_pdb.exists():
        return cached_pdb

    try:
        from ImmuneBuilder import ABodyBuilder2
    except ImportError:
        return None

    try:
        if 'ab2' not in _PREDICTOR_CACHE:
            _PREDICTOR_CACHE['ab2'] = ABodyBuilder2()
        predictor = _PREDICTOR_CACHE['ab2']

        sequences = {'H': heavy_seq}
        if light_seq:
            sequences['L'] = light_seq

        structure = predictor.predict(sequences)
        structure.save(str(cached_pdb))
        return cached_pdb

    except Exception as e:
        # PRINTED, not warned. structural_geometry_survey.py sets
        # warnings.filterwarnings('ignore'), so a warning here is discarded and
        # the caller sees an all-zero feature with no reason attached. That is
        # what produced two rounds of empty geometry results: 3706 predictions
        # failed in 8 minutes and reported "0 raised".
        _PREDICTION_FAILURES.append(f"{type(e).__name__}: {e}")
        if len(_PREDICTION_FAILURES) == 1:
            print(f"  STRUCTURE PREDICTION FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            print("  (further failures counted, not printed)")
        return None


# ============================================================================
# CDR DEFINITIONS (fallback when ANARCI not available)
# ============================================================================

# Approximate Chothia CDR boundaries (0-indexed)
_CDR_HEAVY = {
    'H1': (26, 32), 'H2': (52, 56), 'H3': (95, 102),
}
_CDR_LIGHT = {
    'L1': (24, 34), 'L2': (50, 56), 'L3': (89, 97),
}

# Hydrophobic residues (Kyte-Doolittle > 1.5)
_HYDROPHOBIC = set('AILMFVW')

# Three-letter to one-letter amino acid map
_THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}


def _get_cdr_residue_indices(seq: str, chain_type: str = 'heavy') -> Dict[str, List[int]]:
    """Get CDR residue indices using approximate boundaries.

    Args:
        seq: Amino acid sequence.
        chain_type: 'heavy' or 'light'.

    Returns:
        Dict mapping CDR name → list of 0-indexed positions.
    """
    cdrs = _CDR_HEAVY if chain_type == 'heavy' else _CDR_LIGHT
    result = {}
    for cdr_name, (start, end) in cdrs.items():
        indices = list(range(start, min(end + 1, len(seq))))
        result[cdr_name] = indices
    return result


# ============================================================================
# SASA FEATURES (~15D)
# ============================================================================

def compute_sasa_features(
    pdb_path: Path,
    heavy_seq: str,
    light_seq: Optional[str] = None,
) -> np.ndarray:
    """Compute SASA-based structural features from a PDB file.

    Features (15D):
      0: Total SASA (Å²)
      1: Heavy chain SASA
      2: Light chain SASA (0 if no light chain)
      3-5: CDR-H1, CDR-H2, CDR-H3 SASA
      6-8: CDR-L1, CDR-L2, CDR-L3 SASA (0 if no light chain)
      9: Fraction exposed in CDRs vs framework
      10: Fraction exposed in CDR-H3 vs total
      11: Hydrophobic SASA (nonpolar residue exposed area)
      12: SASA variance across residues
      13: Buried surface area proxy (total - sum of per-chain)
      14: CDR-H3 SASA / total SASA ratio

    Args:
        pdb_path: Path to PDB file.
        heavy_seq: Heavy chain sequence (for CDR identification).
        light_seq: Light chain sequence (optional).

    Returns:
        np.ndarray: (15,) feature vector.
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley

    features = np.zeros(15, dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('struct', str(pdb_path))

    model = structure[0]

    # Compute SASA
    sr = ShrakeRupley()
    sr.compute(model, level='R')  # per-residue SASA

    # Collect per-residue SASA by chain
    chain_sasa = {}
    residue_sasa_list = []

    for chain in model.get_chains():
        chain_id = chain.id
        residue_sasas = []
        for residue in chain.get_residues():
            if residue.get_id()[0] != ' ':
                continue
            sasa = residue.sasa if hasattr(residue, 'sasa') else 0.0
            residue_sasas.append(sasa)
            residue_sasa_list.append(sasa)
        chain_sasa[chain_id] = residue_sasas

    total_sasa = sum(residue_sasa_list)
    features[0] = total_sasa

    # Heavy chain SASA
    heavy_sasa_list = chain_sasa.get('H', [])
    features[1] = sum(heavy_sasa_list)

    # Light chain SASA
    light_sasa_list = chain_sasa.get('L', [])
    features[2] = sum(light_sasa_list)

    # CDR SASA (heavy)
    heavy_cdrs = _get_cdr_residue_indices(heavy_seq, 'heavy')
    cdr_sasa_total = 0.0
    framework_sasa_total = 0.0

    for i, (cdr_name, feat_idx) in enumerate(zip(['H1', 'H2', 'H3'], [3, 4, 5])):
        indices = heavy_cdrs.get(cdr_name, [])
        cdr_sasa = sum(heavy_sasa_list[j] for j in indices if j < len(heavy_sasa_list))
        features[feat_idx] = cdr_sasa
        cdr_sasa_total += cdr_sasa

    # CDR SASA (light)
    if light_seq:
        light_cdrs = _get_cdr_residue_indices(light_seq, 'light')
        for i, (cdr_name, feat_idx) in enumerate(zip(['L1', 'L2', 'L3'], [6, 7, 8])):
            indices = light_cdrs.get(cdr_name, [])
            cdr_sasa = sum(light_sasa_list[j] for j in indices if j < len(light_sasa_list))
            features[feat_idx] = cdr_sasa
            cdr_sasa_total += cdr_sasa

    # Framework SASA = total heavy+light SASA - CDR SASA
    framework_sasa_total = features[1] + features[2] - cdr_sasa_total

    # Fraction exposed in CDRs vs framework
    features[9] = cdr_sasa_total / max(cdr_sasa_total + framework_sasa_total, 1.0)

    # Fraction CDR-H3 vs total
    features[10] = features[5] / max(total_sasa, 1.0)

    # Hydrophobic SASA: sum SASA of hydrophobic residues
    hydrophobic_sasa = 0.0
    for chain in model.get_chains():
        for residue in chain.get_residues():
            if residue.get_id()[0] != ' ':
                continue
            resname = residue.get_resname()
            one_letter = _THREE_TO_ONE.get(resname, '')
            if one_letter in _HYDROPHOBIC:
                hydrophobic_sasa += residue.sasa if hasattr(residue, 'sasa') else 0.0
    features[11] = hydrophobic_sasa

    # SASA variance across residues
    if residue_sasa_list:
        features[12] = np.var(residue_sasa_list)

    # Buried surface area proxy: total - (heavy + light)
    # (difference indicates interchain contacts)
    features[13] = max(0.0, total_sasa - features[1] - features[2])

    # CDR-H3 SASA / heavy chain SASA ratio (different from [10] which uses total)
    features[14] = features[5] / max(features[1], 1.0)

    return features


# ============================================================================
# CONFIDENCE FEATURES (~5D)
# ============================================================================

def compute_confidence_features(
    pdb_path: Path,
    heavy_seq: Optional[str] = None,
) -> np.ndarray:
    """Compute prediction confidence features from B-factor column.

    ABodyBuilder2 stores ensemble RMSD in B-factor (lower = better).
    For experimental PDBs, B-factor is crystallographic displacement.
    Either way, these features capture structural certainty.

    Features (5D):
      0: Mean B-factor (mean confidence)
      1: Min B-factor (best-predicted region)
      2: CDR-H3 mean B-factor (binding loop confidence)
      3: Fraction of residues with B-factor < 2.0 Å
      4: B-factor variance

    Args:
        pdb_path: Path to PDB file.
        heavy_seq: Heavy chain sequence (for CDR-H3 identification).

    Returns:
        np.ndarray: (5,) feature vector.
    """
    from Bio.PDB import PDBParser

    features = np.zeros(5, dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('struct', str(pdb_path))

    model = structure[0]

    # Collect all CA B-factors
    all_bfactors = []
    heavy_bfactors = []
    residue_idx = 0

    for chain in model.get_chains():
        for residue in chain.get_residues():
            if residue.get_id()[0] != ' ':
                continue
            # Use CA atom B-factor
            if 'CA' in residue:
                bf = residue['CA'].get_bfactor()
                all_bfactors.append(bf)
                if chain.id == 'H':
                    heavy_bfactors.append((residue_idx, bf))
            residue_idx += 1

    if not all_bfactors:
        return features

    bf_array = np.array(all_bfactors)
    features[0] = bf_array.mean()
    features[1] = bf_array.min()

    # CDR-H3 B-factors
    if heavy_seq and heavy_bfactors:
        h3_start, h3_end = _CDR_HEAVY['H3']
        h3_bfactors = [
            bf for idx, bf in enumerate(
                [b for _, b in heavy_bfactors]
            ) if h3_start <= idx <= min(h3_end, len(heavy_bfactors) - 1)
        ]
        if h3_bfactors:
            features[2] = np.mean(h3_bfactors)

    # Fraction with B-factor < 2.0
    features[3] = np.mean(bf_array < 2.0)

    # B-factor variance
    features[4] = bf_array.var()

    return features


# ============================================================================
# PARATOPE FEATURES (~8D)
# ============================================================================

def compute_paratope_features(
    heavy_seq: str,
    light_seq: Optional[str] = None,
) -> np.ndarray:
    """Compute paratope (antigen-binding) features.

    Uses CDR-based paratope prediction (CDR residues ≈ paratope).

    Features (8D):
      0: N paratope residues
      1: Fraction paratope (n_paratope / total_length)
      2: Paratope net charge
      3: Paratope mean hydrophobicity
      4: Paratope aromatic fraction (F/W/Y)
      5: Per-CDR paratope fraction: H1
      6: Per-CDR paratope fraction: H2
      7: Per-CDR paratope fraction: H3

    Args:
        heavy_seq: Heavy chain amino acid sequence.
        light_seq: Light chain sequence (optional).

    Returns:
        np.ndarray: (8,) feature vector.
    """
    features = np.zeros(8, dtype=np.float64)

    # Identify paratope residues (CDR-based) — count heavy and light separately
    heavy_cdrs = _get_cdr_residue_indices(heavy_seq, 'heavy')
    heavy_paratope = set()
    for indices in heavy_cdrs.values():
        heavy_paratope.update(i for i in indices if i < len(heavy_seq))

    light_paratope = set()
    if light_seq:
        light_cdrs = _get_cdr_residue_indices(light_seq, 'light')
        for indices in light_cdrs.values():
            light_paratope.update(i for i in indices if i < len(light_seq))

    n_paratope = len(heavy_paratope) + len(light_paratope)
    total_len = len(heavy_seq) + (len(light_seq) if light_seq else 0)

    features[0] = n_paratope
    features[1] = n_paratope / max(total_len, 1)

    # Paratope physicochemical properties (heavy chain CDRs only)
    heavy_paratope_aas = [
        heavy_seq[i] for i in sorted(heavy_paratope)
        if heavy_seq[i] in AA_PROPERTIES
    ]

    if heavy_paratope_aas:
        # Net charge
        features[2] = sum(AA_PROPERTIES[aa]['charge'] for aa in heavy_paratope_aas)
        # Mean hydrophobicity
        features[3] = np.mean([AA_PROPERTIES[aa]['hydrophobicity'] for aa in heavy_paratope_aas])
        # Aromatic fraction
        aromatic = set('FWY')
        features[4] = sum(1 for aa in heavy_paratope_aas if aa in aromatic) / len(heavy_paratope_aas)

    # Per-CDR paratope fraction
    for i, cdr_name in enumerate(['H1', 'H2', 'H3']):
        cdr_indices = heavy_cdrs.get(cdr_name, [])
        valid_indices = [j for j in cdr_indices if j < len(heavy_seq)]
        features[5 + i] = len(valid_indices) / max(total_len, 1)

    return features


# ============================================================================
# COMBINED STRUCTURAL FEATURES
# ============================================================================

def compute_structural_features(
    heavy_seqs: List[str],
    light_seqs: Optional[List[str]] = None,
) -> np.ndarray:
    """Compute combined structural features for a batch of antibodies.

    Orchestrates: structure prediction → SASA → confidence → paratope.
    Falls back gracefully: without ImmuneBuilder, only paratope features
    are computed (sequence-only, 8D). With structures, full 28D.

    Args:
        heavy_seqs: List of heavy chain sequences.
        light_seqs: Optional list of light chain sequences.

    Returns:
        np.ndarray: (n_samples, 28) structural feature matrix.
            Columns 0-14: SASA features (0 if no structure)
            Columns 15-19: Confidence features (0 if no structure)
            Columns 20-27: Paratope features (always computed)
    """
    n = len(heavy_seqs)
    features = np.zeros((n, 28), dtype=np.float64)

    # Check if ImmuneBuilder is available
    has_immunebuilder = False
    try:
        from ImmuneBuilder import ABodyBuilder2
        has_immunebuilder = True
    except ImportError:
        pass

    if has_immunebuilder:
        print(f"  Computing structural features with ABodyBuilder2 ({n} structures)...")
    else:
        print(f"  ImmuneBuilder not available — computing paratope features only (8D)")

    for i in range(n):
        heavy = heavy_seqs[i]
        light = light_seqs[i] if light_seqs else None

        # Structure prediction + SASA + confidence
        if has_immunebuilder:
            pdb_path = predict_structure(heavy, light)
            if pdb_path is not None:
                try:
                    features[i, :15] = compute_sasa_features(pdb_path, heavy, light)
                except Exception as e:
                    warnings.warn(f"SASA failed for sample {i}: {e}")

                try:
                    features[i, 15:20] = compute_confidence_features(pdb_path, heavy)
                except Exception as e:
                    warnings.warn(f"Confidence failed for sample {i}: {e}")

        # Paratope features (always, from sequence)
        try:
            features[i, 20:28] = compute_paratope_features(heavy, light)
        except Exception as e:
            warnings.warn(f"Paratope failed for sample {i}: {e}")

        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"    Progress: {i + 1}/{n}")

    # Count non-zero columns to report actual dimensionality
    active_cols = np.any(features != 0, axis=0).sum()
    print(f"  Generated structural features: ({n}, 28), {active_cols} active columns")
    return features


# ============================================================================
# ANTIFOLD INVERSE FOLDING SCORES (5D)
# ============================================================================

_ANTIFOLD_MODEL_CACHE = {}


def _run_antifold(model, pdb_path: Path, has_light: bool = True,
                  cache_dir: Optional[Path] = None):
    """Run AntiFold on a PDB, return heavy-chain log-probs array.

    Args:
        model: Loaded AntiFold model.
        pdb_path: Path to PDB file (from ABodyBuilder2).
        has_light: Whether the PDB contains a light chain.
        cache_dir: Directory for caching logprobs. Defaults to structure cache.

    Returns:
        (logprobs_H, aa_list): logprobs_H is (n_H_residues, 20) numpy array,
        aa_list is the 20 amino acid characters in column order.
        Returns (None, None) on failure.
    """
    import pandas as pd
    from antifold.antiscripts import get_pdbs_logits, df_logits_to_logprobs

    pdb_path = Path(pdb_path)
    if cache_dir is None:
        cache_dir = _structure_cache_dir()

    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    af_cache = cache_dir / f'{pdb_path.stem}_antifold_logprobs.npy'

    if af_cache.exists():
        logprobs_H = np.load(af_cache)
        return logprobs_H, aa_list

    pdb_dir = str(pdb_path.parent)
    pdb_name = pdb_path.stem

    row = {"pdb": [pdb_name], "Hchain": ["H"]}
    if has_light:
        row["Lchain"] = ["L"]
    pdbs_csv = pd.DataFrame(row)

    try:
        df_logits_list = get_pdbs_logits(
            model=model,
            pdbs_csv_or_dataframe=pdbs_csv,
            pdb_dir=pdb_dir,
        )
        df_logprobs = df_logits_to_logprobs(df_logits_list[0])

        # Filter to heavy chain only
        df_H = df_logprobs[df_logprobs['pdb_chain'] == 'H'].reset_index(drop=True)
        logprobs_H = df_H[aa_list].values.astype(np.float64)

        np.save(af_cache, logprobs_H)
        return logprobs_H, aa_list

    except Exception as e:
        warnings.warn(f"AntiFold failed on {pdb_path}: {e}")
        return None, None


def compute_antifold_scores(
    wt_seq: str,
    mut_seqs: List[str],
    mutations: List[List[Dict]],
    model_name: str = 'antifold',
    wt_light: Optional[str] = None,
) -> np.ndarray:
    """Compute AntiFold inverse folding scores for mutations (5D).

    Scores mutations using structure-conditioned log-probabilities from both
    WT and mutant predicted structures (structural relaxation score):

        score = log P(mut_aa | mut_structure) - log P(wt_aa | wt_structure)

    This captures whether a mutation is self-consistent with the new predicted
    fold — a positive score means AntiFold "agrees" with the mutation given
    the relaxed structure.

    Features per sample (5D):
      0: mean score across mutations
      1: min score
      2: max score
      3: sum of scores
      4: number of unfavorable mutations (score < 0)

    Args:
        wt_seq: Wild-type heavy chain sequence.
        mut_seqs: List of mutant heavy chain sequences.
        mutations: Per-sample mutation lists. Each is a list of dicts with
            keys: chain ('heavy'/'light'), position (0-indexed), wt_aa, mut_aa.
        model_name: Kept for dispatch compatibility.
        wt_light: Wild-type light chain sequence (optional).

    Returns:
        np.ndarray: (n_samples, 5) feature matrix. Zeros if AntiFold or
        ImmuneBuilder is unavailable.
    """
    n = len(mut_seqs)
    features = np.zeros((n, 5), dtype=np.float64)

    try:
        from antifold.antiscripts import load_model
    except ImportError:
        print("  WARNING: antifold not installed — returning zeros")
        return features

    print(f"  Computing AntiFold scores ({n} samples)...")

    if 'antifold' not in _ANTIFOLD_MODEL_CACHE:
        _ANTIFOLD_MODEL_CACHE['antifold'] = load_model()
    model = _ANTIFOLD_MODEL_CACHE['antifold']

    cache_dir = _structure_cache_dir()
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    aa_to_col = {aa: i for i, aa in enumerate(aa_list)}
    has_light = wt_light is not None and len(wt_light) > 0

    # --- WT scoring ---
    wt_pdb = predict_structure(wt_seq, wt_light)
    if wt_pdb is None:
        print("  WARNING: WT structure prediction failed — returning zeros")
        return features

    wt_logprobs, _ = _run_antifold(model, wt_pdb, has_light=has_light,
                                   cache_dir=cache_dir)
    if wt_logprobs is None:
        print("  WARNING: AntiFold failed on WT — returning zeros")
        return features

    # --- Deduplicate mutant sequences ---
    unique_seqs = {}
    for i, seq in enumerate(mut_seqs):
        if seq not in unique_seqs:
            unique_seqs[seq] = []
        unique_seqs[seq].append(i)
    print(f"    {len(unique_seqs)} unique sequences (of {n} total)")

    # --- Process each unique mutant ---
    processed = 0
    for seq, indices in unique_seqs.items():
        mut_pdb = predict_structure(seq, wt_light)
        if mut_pdb is None:
            processed += len(indices)
            continue

        mut_logprobs, _ = _run_antifold(model, mut_pdb, has_light=has_light,
                                        cache_dir=cache_dir)
        if mut_logprobs is None:
            processed += len(indices)
            continue

        for idx in indices:
            sample_mutations = mutations[idx]
            if not sample_mutations:
                continue

            scores = []
            for mut in sample_mutations:
                if mut.get('chain') != 'heavy':
                    continue
                pos = mut['position']
                wt_aa = mut['wt_aa']
                mut_aa = mut['mut_aa']

                if pos >= len(wt_logprobs) or pos >= len(mut_logprobs):
                    continue
                if wt_aa not in aa_to_col or mut_aa not in aa_to_col:
                    continue

                score = float(
                    mut_logprobs[pos, aa_to_col[mut_aa]]
                    - wt_logprobs[pos, aa_to_col[wt_aa]]
                )
                scores.append(score)

            if scores:
                features[idx, 0] = np.mean(scores)
                features[idx, 1] = np.min(scores)
                features[idx, 2] = np.max(scores)
                features[idx, 3] = np.sum(scores)
                features[idx, 4] = sum(1 for s in scores if s < 0)

        processed += len(indices)
        if processed % 50 < len(indices) or processed == n:
            print(f"    Progress: {processed}/{n}")

    active = np.any(features != 0, axis=0).sum()
    print(f"  Generated AntiFold scores: ({n}, 5), {active} active columns")
    return features
