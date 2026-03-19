"""
Geometry-based structural features for antibodies.

Computed from predicted 3D structures (ABodyBuilder2 via ImmuneBuilder).
7 feature groups capturing VH-VL orientation, CDR loop geometry,
paratope physicochemistry, and interface properties.

Feature groups:
  1. vhvl_angles (6D)          — ABangle-style VH-VL orientation
  2. cdr_shape (60D)           — USR-like shape moments per CDR loop
  3. cdr_canonical (12D)       — CDR canonical class + geometry
  4. paratope_charge (12D)     — Electrostatics of paratope residues
  5. paratope_hydrophobicity (12D) — Hydrophobicity of paratope
  6. cdr_contacts (30D)        — Inter-CDR pairwise distances
  7. vhvl_interface (30D)      — VH-VL interface composition

All require ImmuneBuilder for structure prediction; return zeros if unavailable.

Usage:
    from kirby.representations.antibody_geometry import (
        GEOMETRY_REPS, compute_all_geometry_features)

    # All 7 as a dict of (N, D) arrays:
    geo = compute_all_geometry_features(heavy_seqs, light_seqs)

    # Single rep:
    X = GEOMETRY_REPS['vhvl_angles'](heavy_seqs, light_seqs)
"""

import warnings
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from kirby.representations.structural import (
    predict_structure, _structure_cache_dir,
    _CDR_HEAVY, _CDR_LIGHT, _THREE_TO_ONE,
)


# ============================================================================
# PHYSICOCHEMICAL SCALES
# ============================================================================

# Formal charge at pH ~7.4
_CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'Q': 0, 'E': -1, 'G': 0, 'H': 0.5, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
}

# Eisenberg consensus hydrophobicity
_EISENBERG = {
    'A': 0.620, 'R': -2.530, 'N': -0.780, 'D': -0.900, 'C': 0.290,
    'Q': -0.850, 'E': -0.740, 'G': 0.480, 'H': -0.400, 'I': 1.380,
    'L': 1.060, 'K': -1.500, 'M': 0.640, 'F': 1.190, 'P': 0.120,
    'S': -0.180, 'T': -0.050, 'W': 0.810, 'Y': 0.260, 'V': 1.080,
}

_HYDROPHOBIC = set('AILMFVW')
_AROMATIC = set('FWY')
_POLAR = set('STNQDE')


# ============================================================================
# CDR CANONICAL CLASS DEFINITIONS (by loop length, IMGT/Chothia-approximate)
# ============================================================================

# Canonical class assignment based on CDR loop length.
# Key = CDR length, value = class ID (0-indexed).
# Unlisted lengths → class = max_id + 1 ("other").
_CANONICAL_BY_LENGTH = {
    'H1': {10: 0, 11: 1, 12: 2, 13: 3},
    'H2': {7: 0, 9: 1, 10: 2},
    'H3': None,   # H3 has no canonical classes
    'L1': {10: 0, 11: 1, 12: 2, 14: 3},
    'L2': {3: 0, 7: 1},
    'L3': {7: 0, 9: 1, 11: 2},
}

# Number of canonical classes per CDR (including "other")
_N_CLASSES = {
    'H1': 5, 'H2': 4, 'H3': 3, 'L1': 5, 'L2': 3, 'L3': 4,
}


# ============================================================================
# PDB PARSING
# ============================================================================

def _parse_pdb_chains(pdb_path: Path) -> Dict[str, dict]:
    """Parse PDB into per-chain Cα coordinates and sequences.

    Returns dict: chain_id → {'ca_coords': (N, 3), 'sequence': str,
                               'residues': list of {aa, coord, resid}}.
    """
    from Bio.PDB import PDBParser

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('ab', str(pdb_path))

    model = structure[0]
    chains = {}

    for chain in model.get_chains():
        residues = []
        for res in chain.get_residues():
            if res.get_id()[0] != ' ':
                continue
            if 'CA' not in res:
                continue
            aa = _THREE_TO_ONE.get(res.get_resname(), 'X')
            coord = res['CA'].get_vector().get_array().copy()
            residues.append({'aa': aa, 'coord': coord, 'resid': res.get_id()[1]})

        if residues:
            chains[chain.id] = {
                'ca_coords': np.array([r['coord'] for r in residues]),
                'sequence': ''.join(r['aa'] for r in residues),
                'residues': residues,
            }

    return chains


def _get_cdr_coords(chain_data: dict, chain_type: str) -> Dict[str, np.ndarray]:
    """Extract per-CDR Cα coordinates from a parsed chain.

    Args:
        chain_data: Output of _parse_pdb_chains for one chain.
        chain_type: 'heavy' or 'light'.

    Returns:
        Dict mapping CDR name (H1/H2/H3 or L1/L2/L3) → (n_cdr, 3) array.
    """
    cdrs = _CDR_HEAVY if chain_type == 'heavy' else _CDR_LIGHT
    ca = chain_data['ca_coords']
    n = len(ca)

    result = {}
    for cdr_name, (start, end) in cdrs.items():
        indices = list(range(start, min(end + 1, n)))
        if indices:
            result[cdr_name] = ca[indices]
        else:
            result[cdr_name] = np.zeros((0, 3))

    return result


def _get_framework_coords(chain_data: dict, chain_type: str) -> np.ndarray:
    """Extract framework (non-CDR) Cα coordinates."""
    cdrs = _CDR_HEAVY if chain_type == 'heavy' else _CDR_LIGHT
    ca = chain_data['ca_coords']
    n = len(ca)

    cdr_positions = set()
    for start, end in cdrs.values():
        cdr_positions.update(range(start, min(end + 1, n)))

    fw_indices = [i for i in range(n) if i not in cdr_positions]
    if fw_indices:
        return ca[fw_indices]
    return ca  # fallback to full chain


def _get_cdr_aas(chain_data: dict, chain_type: str) -> Dict[str, str]:
    """Extract per-CDR amino acid sequences from a parsed chain."""
    cdrs = _CDR_HEAVY if chain_type == 'heavy' else _CDR_LIGHT
    seq = chain_data['sequence']
    n = len(seq)

    result = {}
    for cdr_name, (start, end) in cdrs.items():
        result[cdr_name] = seq[start:min(end + 1, n)]

    return result


# ============================================================================
# 1. VH-VL ORIENTATION ANGLES (6D)
# ============================================================================

def _compute_domain_frame(ca_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute coordinate frame from Cα positions via SVD.

    Returns (center, axes) where axes is (3, 3): rows are principal axes
    ordered by decreasing variance.
    """
    center = ca_coords.mean(axis=0)
    centered = ca_coords - center
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return center, Vt


def compute_vhvl_angles(
    h_chain: dict,
    l_chain: Optional[dict],
) -> np.ndarray:
    """ABangle-style VH-VL orientation descriptors (6D).

    Features:
      0: HL  — angle between VH and VL principal axes (degrees)
      1: HC1 — VH 2nd axis angle to inter-domain vector
      2: HC2 — VH 3rd axis angle to inter-domain vector
      3: LC1 — VL 2nd axis angle to inter-domain vector
      4: LC2 — VL 3rd axis angle to inter-domain vector
      5: dc  — distance between domain centers (Å)
    """
    features = np.zeros(6, dtype=np.float64)

    if l_chain is None:
        return features

    vh_fw = _get_framework_coords(h_chain, 'heavy')
    vl_fw = _get_framework_coords(l_chain, 'light')

    if len(vh_fw) < 3 or len(vl_fw) < 3:
        return features

    vh_center, vh_axes = _compute_domain_frame(vh_fw)
    vl_center, vl_axes = _compute_domain_frame(vl_fw)

    # dc: distance between domain centers
    dc = np.linalg.norm(vh_center - vl_center)
    features[5] = dc

    if dc < 1e-6:
        return features

    interdomain = (vl_center - vh_center) / dc

    # HL: angle between first principal axes
    cos_hl = np.dot(vh_axes[0], vl_axes[0])
    features[0] = np.degrees(np.arccos(np.clip(abs(cos_hl), 0, 1)))

    # HC1, HC2: VH 2nd/3rd axis relative to inter-domain vector
    features[1] = np.degrees(np.arccos(np.clip(
        abs(np.dot(vh_axes[1], interdomain)), 0, 1)))
    features[2] = np.degrees(np.arccos(np.clip(
        abs(np.dot(vh_axes[2], interdomain)), 0, 1)))

    # LC1, LC2: VL 2nd/3rd axis relative to inter-domain vector
    features[3] = np.degrees(np.arccos(np.clip(
        abs(np.dot(vl_axes[1], -interdomain)), 0, 1)))
    features[4] = np.degrees(np.arccos(np.clip(
        abs(np.dot(vl_axes[2], -interdomain)), 0, 1)))

    return features


# ============================================================================
# 2. CDR LOOP SHAPE (60D = 6 CDRs × 10 moments)
# ============================================================================

def _usr_moments(coords: np.ndarray) -> np.ndarray:
    """USR-like shape descriptors for a set of 3D coordinates.

    Returns 10D: 4 reference points × (mean, variance) distances + end2end + Rg.
    """
    if len(coords) < 3:
        return np.zeros(10, dtype=np.float64)

    centroid = coords.mean(axis=0)
    dists_centroid = np.linalg.norm(coords - centroid, axis=1)

    # 4 USR reference points
    closest = coords[np.argmin(dists_centroid)]
    farthest = coords[np.argmax(dists_centroid)]
    dists_farthest = np.linalg.norm(coords - farthest, axis=1)
    farthest2 = coords[np.argmax(dists_farthest)]

    refs = [centroid, closest, farthest, farthest2]
    moments = []
    for ref in refs:
        d = np.linalg.norm(coords - ref, axis=1)
        moments.append(d.mean())
        moments.append(d.var())

    # End-to-end distance
    end2end = np.linalg.norm(coords[0] - coords[-1])
    moments.append(end2end)

    # Radius of gyration
    rg = np.sqrt(np.mean(dists_centroid ** 2))
    moments.append(rg)

    return np.array(moments, dtype=np.float64)


def compute_cdr_shape(
    h_chain: dict,
    l_chain: Optional[dict],
) -> np.ndarray:
    """USR-like shape descriptors for each CDR loop (60D).

    6 CDRs × 10 moments. Order: H1, H2, H3, L1, L2, L3.
    L CDRs are zero if no light chain.
    """
    features = np.zeros(60, dtype=np.float64)

    # Heavy CDRs
    h_cdrs = _get_cdr_coords(h_chain, 'heavy')
    for i, name in enumerate(['H1', 'H2', 'H3']):
        coords = h_cdrs.get(name, np.zeros((0, 3)))
        if len(coords) >= 3:
            features[i * 10:(i + 1) * 10] = _usr_moments(coords)

    # Light CDRs
    if l_chain is not None:
        l_cdrs = _get_cdr_coords(l_chain, 'light')
        for i, name in enumerate(['L1', 'L2', 'L3']):
            coords = l_cdrs.get(name, np.zeros((0, 3)))
            if len(coords) >= 3:
                features[(i + 3) * 10:(i + 4) * 10] = _usr_moments(coords)

    return features


# ============================================================================
# 3. CDR CANONICAL CLASSES (12D = 6 CDRs × 2)
# ============================================================================

def compute_cdr_canonical(
    h_chain: dict,
    l_chain: Optional[dict],
) -> np.ndarray:
    """CDR canonical class + geometric descriptor per loop (12D).

    Per CDR (2 features):
      - Canonical class ID (normalized 0-1; based on loop length)
      - End-to-end Cα distance (Å)

    Order: H1, H2, H3, L1, L2, L3. L CDRs zero if no light chain.
    """
    features = np.zeros(12, dtype=np.float64)
    cdr_order = ['H1', 'H2', 'H3', 'L1', 'L2', 'L3']

    # Heavy CDRs
    h_cdrs = _get_cdr_coords(h_chain, 'heavy')
    h_aas = _get_cdr_aas(h_chain, 'heavy')

    # Light CDRs
    l_cdrs = {} if l_chain is None else _get_cdr_coords(l_chain, 'light')
    l_aas = {} if l_chain is None else _get_cdr_aas(l_chain, 'light')

    all_cdrs = {**h_cdrs, **l_cdrs}
    all_aas = {**h_aas, **l_aas}

    for i, name in enumerate(cdr_order):
        coords = all_cdrs.get(name, np.zeros((0, 3)))
        seq = all_aas.get(name, '')
        length = len(seq)

        if length == 0:
            continue

        # Canonical class
        class_map = _CANONICAL_BY_LENGTH.get(name)
        n_classes = _N_CLASSES.get(name, 3)

        if class_map is None:
            # H3: use length bin (short ≤8, medium 9-14, long ≥15)
            if length <= 8:
                cls = 0
            elif length <= 14:
                cls = 1
            else:
                cls = 2
        else:
            cls = class_map.get(length, max(class_map.values()) + 1)

        features[i * 2] = cls / max(n_classes - 1, 1)

        # End-to-end distance
        if len(coords) >= 2:
            features[i * 2 + 1] = np.linalg.norm(coords[0] - coords[-1])

    return features


# ============================================================================
# 4. PARATOPE SURFACE ELECTROSTATICS (12D)
# ============================================================================

def compute_paratope_charge(
    h_chain: dict,
    l_chain: Optional[dict],
) -> np.ndarray:
    """Electrostatic features of paratope (CDR) residues (12D).

    Features:
      0: Net charge (sum)
      1: Mean charge per residue
      2: Charge variance
      3: Positive charge fraction (fraction of residues with charge > 0)
      4: Negative charge fraction
      5: Charge asymmetry: (pos - neg) / (pos + neg + 1e-8)
      6-11: Per-CDR net charge (H1, H2, H3, L1, L2, L3)
    """
    features = np.zeros(12, dtype=np.float64)
    cdr_names = ['H1', 'H2', 'H3', 'L1', 'L2', 'L3']

    h_aas = _get_cdr_aas(h_chain, 'heavy')
    l_aas = {} if l_chain is None else _get_cdr_aas(l_chain, 'light')
    all_aas = {**h_aas, **l_aas}

    all_charges = []
    for i, name in enumerate(cdr_names):
        seq = all_aas.get(name, '')
        charges = [_CHARGE.get(aa, 0.0) for aa in seq]
        if charges:
            features[6 + i] = sum(charges)
        all_charges.extend(charges)

    if not all_charges:
        return features

    charges_arr = np.array(all_charges)
    features[0] = charges_arr.sum()
    features[1] = charges_arr.mean()
    features[2] = charges_arr.var()
    features[3] = np.mean(charges_arr > 0)
    features[4] = np.mean(charges_arr < 0)

    pos_sum = charges_arr[charges_arr > 0].sum()
    neg_sum = abs(charges_arr[charges_arr < 0].sum())
    features[5] = (pos_sum - neg_sum) / (pos_sum + neg_sum + 1e-8)

    return features


# ============================================================================
# 5. PARATOPE HYDROPHOBICITY (12D)
# ============================================================================

def compute_paratope_hydrophobicity(
    h_chain: dict,
    l_chain: Optional[dict],
) -> np.ndarray:
    """Hydrophobicity profile of paratope (CDR) residues (12D).

    Features:
      0: Mean Eisenberg hydrophobicity
      1: Max hydrophobicity
      2: Hydrophobicity variance
      3: Fraction hydrophobic residues
      4: Fraction aromatic (F/W/Y)
      5: Fraction polar (S/T/N/Q/D/E)
      6-11: Per-CDR mean hydrophobicity (H1, H2, H3, L1, L2, L3)
    """
    features = np.zeros(12, dtype=np.float64)
    cdr_names = ['H1', 'H2', 'H3', 'L1', 'L2', 'L3']

    h_aas = _get_cdr_aas(h_chain, 'heavy')
    l_aas = {} if l_chain is None else _get_cdr_aas(l_chain, 'light')
    all_aas = {**h_aas, **l_aas}

    all_hydro = []
    all_residues = []
    for i, name in enumerate(cdr_names):
        seq = all_aas.get(name, '')
        hydro = [_EISENBERG.get(aa, 0.0) for aa in seq]
        if hydro:
            features[6 + i] = np.mean(hydro)
        all_hydro.extend(hydro)
        all_residues.extend(seq)

    if not all_hydro:
        return features

    h_arr = np.array(all_hydro)
    features[0] = h_arr.mean()
    features[1] = h_arr.max()
    features[2] = h_arr.var()
    features[3] = sum(1 for aa in all_residues if aa in _HYDROPHOBIC) / len(all_residues)
    features[4] = sum(1 for aa in all_residues if aa in _AROMATIC) / len(all_residues)
    features[5] = sum(1 for aa in all_residues if aa in _POLAR) / len(all_residues)

    return features


# ============================================================================
# 6. CDR CONTACT MAP (30D = 15 pairs × 2)
# ============================================================================

def compute_cdr_contacts(
    h_chain: dict,
    l_chain: Optional[dict],
) -> np.ndarray:
    """Inter-CDR pairwise distance features (30D).

    For each of 15 CDR pairs (6 choose 2):
      - Minimum Cα distance (Å)
      - Mean Cα distance (Å)

    Pair order: H1-H2, H1-H3, H1-L1, H1-L2, H1-L3,
                H2-H3, H2-L1, H2-L2, H2-L3,
                H3-L1, H3-L2, H3-L3,
                L1-L2, L1-L3, L2-L3
    """
    features = np.zeros(30, dtype=np.float64)
    cdr_names = ['H1', 'H2', 'H3', 'L1', 'L2', 'L3']

    h_cdrs = _get_cdr_coords(h_chain, 'heavy')
    l_cdrs = {} if l_chain is None else _get_cdr_coords(l_chain, 'light')
    all_cdrs = {**h_cdrs, **l_cdrs}

    pair_idx = 0
    for i in range(6):
        for j in range(i + 1, 6):
            name_i = cdr_names[i]
            name_j = cdr_names[j]
            coords_i = all_cdrs.get(name_i, np.zeros((0, 3)))
            coords_j = all_cdrs.get(name_j, np.zeros((0, 3)))

            if len(coords_i) > 0 and len(coords_j) > 0:
                # Compute pairwise distance matrix
                diff = coords_i[:, None, :] - coords_j[None, :, :]
                dists = np.linalg.norm(diff, axis=2)
                features[pair_idx * 2] = dists.min()
                features[pair_idx * 2 + 1] = dists.mean()

            pair_idx += 1

    return features


# ============================================================================
# 7. VH-VL INTERFACE COMPOSITION (30D)
# ============================================================================

def compute_vhvl_interface(
    h_chain: dict,
    l_chain: Optional[dict],
    contact_threshold: float = 8.0,
) -> np.ndarray:
    """VH-VL interface composition and geometry (30D).

    Interface = residues with Cα within contact_threshold Å across chains.

    Features:
      0: N interface residues (VH)
      1: N interface residues (VL)
      2: Interface fraction (VH interface / VH total)
      3: Interface fraction (VL interface / VL total)
      4: Mean contact distance across interface
      5: N contacts (VH-VL Cα pairs within threshold)
      6: Contact density (N contacts / (N_H_interface × N_L_interface))
      7: Interface charged fraction
      8: Interface hydrophobic fraction
      9: Interface aromatic fraction
      10-29: Interface AA composition (20D, over both chains combined)
    """
    features = np.zeros(30, dtype=np.float64)

    if l_chain is None:
        return features

    h_ca = h_chain['ca_coords']
    l_ca = l_chain['ca_coords']
    h_seq = h_chain['sequence']
    l_seq = l_chain['sequence']

    if len(h_ca) == 0 or len(l_ca) == 0:
        return features

    # Pairwise Cα distances between H and L
    diff = h_ca[:, None, :] - l_ca[None, :, :]
    dists = np.linalg.norm(diff, axis=2)

    # Interface residues: those with at least one cross-chain contact
    h_min_dist = dists.min(axis=1)
    l_min_dist = dists.min(axis=0)

    h_interface = np.where(h_min_dist < contact_threshold)[0]
    l_interface = np.where(l_min_dist < contact_threshold)[0]

    features[0] = len(h_interface)
    features[1] = len(l_interface)
    features[2] = len(h_interface) / max(len(h_ca), 1)
    features[3] = len(l_interface) / max(len(l_ca), 1)

    # Contacts within threshold
    contacts = dists < contact_threshold
    n_contacts = contacts.sum()
    features[5] = n_contacts

    if n_contacts > 0:
        features[4] = dists[contacts].mean()
    if len(h_interface) > 0 and len(l_interface) > 0:
        features[6] = n_contacts / (len(h_interface) * len(l_interface))

    # Interface AA properties
    interface_aas = []
    for idx in h_interface:
        if idx < len(h_seq):
            interface_aas.append(h_seq[idx])
    for idx in l_interface:
        if idx < len(l_seq):
            interface_aas.append(l_seq[idx])

    if interface_aas:
        charged = set('RDEHK')
        features[7] = sum(1 for aa in interface_aas if aa in charged) / len(interface_aas)
        features[8] = sum(1 for aa in interface_aas if aa in _HYDROPHOBIC) / len(interface_aas)
        features[9] = sum(1 for aa in interface_aas if aa in _AROMATIC) / len(interface_aas)

        # AA composition (20D)
        aa_alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        for aa_char in interface_aas:
            idx = aa_alphabet.find(aa_char)
            if idx >= 0:
                features[10 + idx] += 1
        total = sum(features[10:30])
        if total > 0:
            features[10:30] /= total

    return features


# ============================================================================
# BATCH COMPUTATION
# ============================================================================

def _compute_single(
    heavy_seq: str,
    light_seq: Optional[str],
) -> Dict[str, np.ndarray]:
    """Compute all 7 geometry feature groups for one antibody.

    Returns dict mapping feature group name → 1D array.
    """
    result = {
        'vhvl_angles': np.zeros(6, dtype=np.float64),
        'cdr_shape': np.zeros(60, dtype=np.float64),
        'cdr_canonical': np.zeros(12, dtype=np.float64),
        'paratope_charge': np.zeros(12, dtype=np.float64),
        'paratope_hydrophobicity': np.zeros(12, dtype=np.float64),
        'cdr_contacts': np.zeros(30, dtype=np.float64),
        'vhvl_interface': np.zeros(30, dtype=np.float64),
    }

    light = light_seq if light_seq and len(light_seq) > 0 else None

    # Structure prediction
    pdb_path = predict_structure(heavy_seq, light)
    if pdb_path is None:
        return result

    # Parse structure
    try:
        chains = _parse_pdb_chains(pdb_path)
    except Exception as e:
        warnings.warn(f"PDB parsing failed: {e}")
        return result

    h_chain = chains.get('H')
    l_chain = chains.get('L')

    if h_chain is None:
        return result

    try:
        result['vhvl_angles'] = compute_vhvl_angles(h_chain, l_chain)
    except Exception:
        pass

    try:
        result['cdr_shape'] = compute_cdr_shape(h_chain, l_chain)
    except Exception:
        pass

    try:
        result['cdr_canonical'] = compute_cdr_canonical(h_chain, l_chain)
    except Exception:
        pass

    try:
        result['paratope_charge'] = compute_paratope_charge(h_chain, l_chain)
    except Exception:
        pass

    try:
        result['paratope_hydrophobicity'] = compute_paratope_hydrophobicity(h_chain, l_chain)
    except Exception:
        pass

    try:
        result['cdr_contacts'] = compute_cdr_contacts(h_chain, l_chain)
    except Exception:
        pass

    try:
        result['vhvl_interface'] = compute_vhvl_interface(h_chain, l_chain)
    except Exception:
        pass

    return result


# Feature dimensions
GEOMETRY_DIMS = {
    'vhvl_angles': 6,
    'cdr_shape': 60,
    'cdr_canonical': 12,
    'paratope_charge': 12,
    'paratope_hydrophobicity': 12,
    'cdr_contacts': 30,
    'vhvl_interface': 30,
}


def compute_all_geometry_features(
    heavy_seqs: List[str],
    light_seqs: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Compute all 7 geometry feature groups for a batch of antibodies.

    Returns dict mapping feature group name → (N, D) array.
    """
    n = len(heavy_seqs)

    # Initialize output arrays
    arrays = {name: np.zeros((n, dim), dtype=np.float64)
              for name, dim in GEOMETRY_DIMS.items()}

    # Check ImmuneBuilder availability
    has_ib = False
    try:
        from ImmuneBuilder import ABodyBuilder2
        has_ib = True
    except ImportError:
        pass

    if not has_ib:
        print("  WARNING: ImmuneBuilder not available — all geometry features will be zeros")
        return arrays

    print(f"  Computing geometry features ({n} structures)...")

    for i in range(n):
        heavy = heavy_seqs[i]
        light = light_seqs[i] if light_seqs else None

        try:
            single = _compute_single(heavy, light)
            for name in GEOMETRY_DIMS:
                arrays[name][i] = single[name]
        except Exception as e:
            warnings.warn(f"Geometry features failed for sample {i}: {e}")

        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"    Progress: {i + 1}/{n}")

    # Report active columns per feature group
    for name, arr in arrays.items():
        active = np.any(arr != 0, axis=0).sum()
        print(f"    {name}: ({n}, {GEOMETRY_DIMS[name]}), "
              f"{active} active columns")

    return arrays


# ============================================================================
# INDIVIDUAL EMBED FUNCTIONS (match ANTIBODY_REPS interface)
# ============================================================================

def embed_vhvl_angles(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """VH-VL orientation angles → (N, 6)."""
    return compute_all_geometry_features(heavy_seqs, list(light_seqs))['vhvl_angles'].astype(np.float32)


def embed_cdr_shape(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """CDR loop shape (USR moments) → (N, 60)."""
    return compute_all_geometry_features(heavy_seqs, list(light_seqs))['cdr_shape'].astype(np.float32)


def embed_cdr_canonical(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """CDR canonical class features → (N, 12)."""
    return compute_all_geometry_features(heavy_seqs, list(light_seqs))['cdr_canonical'].astype(np.float32)


def embed_paratope_charge(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """Paratope electrostatics → (N, 12)."""
    return compute_all_geometry_features(heavy_seqs, list(light_seqs))['paratope_charge'].astype(np.float32)


def embed_paratope_hydrophobicity(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """Paratope hydrophobicity → (N, 12)."""
    return compute_all_geometry_features(heavy_seqs, list(light_seqs))['paratope_hydrophobicity'].astype(np.float32)


def embed_cdr_contacts(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """CDR contact map → (N, 30)."""
    return compute_all_geometry_features(heavy_seqs, list(light_seqs))['cdr_contacts'].astype(np.float32)


def embed_vhvl_interface(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """VH-VL interface composition → (N, 30)."""
    return compute_all_geometry_features(heavy_seqs, list(light_seqs))['vhvl_interface'].astype(np.float32)


def embed_all_geometry(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """All geometry features concatenated → (N, 162)."""
    geo = compute_all_geometry_features(heavy_seqs, list(light_seqs))
    order = ['vhvl_angles', 'cdr_shape', 'cdr_canonical', 'paratope_charge',
             'paratope_hydrophobicity', 'cdr_contacts', 'vhvl_interface']
    return np.hstack([geo[name] for name in order]).astype(np.float32)


# ============================================================================
# REGISTRY
# ============================================================================

GEOMETRY_REPS = {
    'vhvl_angles': embed_vhvl_angles,                       # 6D
    'cdr_shape': embed_cdr_shape,                            # 60D
    'cdr_canonical': embed_cdr_canonical,                    # 12D
    'paratope_charge': embed_paratope_charge,                # 12D
    'paratope_hydrophobicity': embed_paratope_hydrophobicity,# 12D
    'cdr_contacts': embed_cdr_contacts,                      # 30D
    'vhvl_interface': embed_vhvl_interface,                  # 30D
    'all_geometry': embed_all_geometry,                      # 162D
}
