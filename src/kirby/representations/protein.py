"""
Protein Representations — General-Purpose

Representations for arbitrary protein sequences (not antibody-specific).
Used primarily for drug-target affinity (DTA) prediction where target
proteins are kinases, GPCRs, etc.

Functions:
- create_esm2_protein_embeddings: ESM-2 mean-pooled embeddings (1280D)
- create_aido_protein_embeddings: AIDO.Protein-16B MoE embeddings (2304D)
- create_protein_aac: amino acid composition (20D)
- create_protein_descriptors: biophysical descriptors (~30D)
- create_protein_kmer: k-mer frequency (default k=3, 8000D)
- create_contact_graph_features: residue contact graph topology (~80D)
- create_usr_shape_features: 3D shape descriptors, rotation-invariant (~60D)
- create_pocket_features: binding pocket geometry & composition (~55D)
- create_esm2_pocket_embeddings: pocket-local ESM-2 embeddings (1280D)
- create_pocket_composition_features: pocket AA composition (20D)
"""

import os
import numpy as np
from typing import List, Optional

# Standard amino acid order (matches antibody.py)
STANDARD_AAS = 'ACDEFGHIKLMNPQRSTVWY'

# Physicochemical properties per amino acid
AA_PROPERTIES = {
    'A': {'hydrophobicity':  1.8, 'charge':  0.0, 'mw':  89.09, 'volume':  88.6, 'polarity': 0.0, 'flexibility': 0.360},
    'R': {'hydrophobicity': -4.5, 'charge':  1.0, 'mw': 174.20, 'volume': 173.4, 'polarity': 1.0, 'flexibility': 0.530},
    'N': {'hydrophobicity': -3.5, 'charge':  0.0, 'mw': 132.12, 'volume': 114.1, 'polarity': 1.0, 'flexibility': 0.460},
    'D': {'hydrophobicity': -3.5, 'charge': -1.0, 'mw': 133.10, 'volume': 111.1, 'polarity': 1.0, 'flexibility': 0.510},
    'C': {'hydrophobicity':  2.5, 'charge':  0.0, 'mw': 121.15, 'volume': 108.5, 'polarity': 0.0, 'flexibility': 0.350},
    'E': {'hydrophobicity': -3.5, 'charge': -1.0, 'mw': 147.13, 'volume': 138.4, 'polarity': 1.0, 'flexibility': 0.500},
    'Q': {'hydrophobicity': -3.5, 'charge':  0.0, 'mw': 146.15, 'volume': 143.8, 'polarity': 1.0, 'flexibility': 0.490},
    'G': {'hydrophobicity': -0.4, 'charge':  0.0, 'mw':  75.03, 'volume':  60.1, 'polarity': 0.0, 'flexibility': 0.540},
    'H': {'hydrophobicity': -3.2, 'charge':  0.1, 'mw': 155.16, 'volume': 153.2, 'polarity': 1.0, 'flexibility': 0.320},
    'I': {'hydrophobicity':  4.5, 'charge':  0.0, 'mw': 131.17, 'volume': 166.7, 'polarity': 0.0, 'flexibility': 0.460},
    'L': {'hydrophobicity':  3.8, 'charge':  0.0, 'mw': 131.17, 'volume': 166.7, 'polarity': 0.0, 'flexibility': 0.400},
    'K': {'hydrophobicity': -3.9, 'charge':  1.0, 'mw': 146.19, 'volume': 168.6, 'polarity': 1.0, 'flexibility': 0.535},
    'M': {'hydrophobicity':  1.9, 'charge':  0.0, 'mw': 149.21, 'volume': 162.9, 'polarity': 0.0, 'flexibility': 0.410},
    'F': {'hydrophobicity':  2.8, 'charge':  0.0, 'mw': 165.19, 'volume': 189.9, 'polarity': 0.0, 'flexibility': 0.310},
    'P': {'hydrophobicity': -1.6, 'charge':  0.0, 'mw': 115.13, 'volume': 112.7, 'polarity': 0.0, 'flexibility': 0.510},
    'S': {'hydrophobicity': -0.8, 'charge':  0.0, 'mw': 105.09, 'volume':  89.0, 'polarity': 1.0, 'flexibility': 0.510},
    'T': {'hydrophobicity': -0.7, 'charge':  0.0, 'mw': 119.12, 'volume': 116.1, 'polarity': 1.0, 'flexibility': 0.440},
    'W': {'hydrophobicity': -0.9, 'charge':  0.0, 'mw': 204.23, 'volume': 227.8, 'polarity': 0.0, 'flexibility': 0.310},
    'Y': {'hydrophobicity': -1.3, 'charge':  0.0, 'mw': 181.19, 'volume': 193.6, 'polarity': 1.0, 'flexibility': 0.420},
    'V': {'hydrophobicity':  4.2, 'charge':  0.0, 'mw': 117.15, 'volume': 140.0, 'polarity': 0.0, 'flexibility': 0.390},
}


def _esm2_per_residue_cache_dir() -> 'Path':
    """Return cache directory for per-residue ESM-2 embeddings."""
    from pathlib import Path
    cache_dir = Path(os.environ.get('KIRBY_CACHE_DIR',
                     str(Path.home() / '.cache' / 'kirby'))) / 'esm2_per_residue'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _seq_hash(seq: str) -> str:
    import hashlib
    return hashlib.sha256(seq.upper().strip().encode()).hexdigest()[:16]


# =============================================================================
# SHARED STRUCTURAL COORDINATE CACHE
# =============================================================================

def _struct_cache_dir():
    """Return cache directory for protein structure coordinates."""
    from pathlib import Path
    seagate = Path('/Volumes/seagate/kirby/protein_structures')
    if seagate.parent.parent.exists():
        seagate.mkdir(parents=True, exist_ok=True)
        return seagate
    cache_dir = Path(os.environ.get('KIRBY_CACHE_DIR',
                     str(Path.home() / '.cache' / 'kirby'))) / 'protein_structures'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_ca_positions(
    sequence: str,
    uniprot_id: Optional[str] = None,
    device: str = 'cpu',
    source: str = 'auto',
    _esmfold_model=None,
    _esmfold_tokenizer=None,
) -> tuple:
    """Get Cα positions and pLDDT for a protein from cache or prediction.

    Args:
        sequence: Amino acid sequence.
        uniprot_id: Optional UniProt ID for AlphaFold DB lookup.
        device: Device for ESMFold prediction.
        source: 'auto' (try cache→AF→ESMFold), 'esmfold', 'alphafold'.
        _esmfold_model: Pre-loaded ESMFold model (to avoid reloading).
        _esmfold_tokenizer: Pre-loaded ESMFold tokenizer.

    Returns:
        (ca_positions: np.ndarray (N, 3), plddt: np.ndarray (N,)) or (None, None).
    """
    seq = sequence.upper().strip()
    sh = _seq_hash(seq)
    cache_dir = _struct_cache_dir()

    # Check cache first
    ca_file = cache_dir / f'{sh}_ca.npy'
    plddt_file = cache_dir / f'{sh}_plddt.npy'
    if ca_file.exists() and plddt_file.exists():
        return np.load(ca_file), np.load(plddt_file)

    ca_positions = None
    plddt = None

    # Try AlphaFold DB if UniProt ID available
    if source in ('auto', 'alphafold') and uniprot_id is not None:
        try:
            import requests
            from pathlib import Path
            af_cache = Path(os.environ.get('KIRBY_CACHE_DIR',
                           str(Path.home() / '.cache' / 'kirby'))) / 'alphafold_db'
            af_cache.mkdir(parents=True, exist_ok=True)
            pdb_path = af_cache / f'AF-{uniprot_id}-F1-model_v4.pdb'
            if not pdb_path.exists():
                url = f'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb'
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    pdb_path.write_bytes(resp.content)
            if pdb_path.exists():
                from Bio.PDB import PDBParser
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure('af', str(pdb_path))
                model = structure[0]
                ca_atoms, plddt_vals = [], []
                for chain in model.get_chains():
                    for residue in chain.get_residues():
                        if residue.get_id()[0] != ' ':
                            continue
                        if 'CA' in residue:
                            ca_atoms.append(residue['CA'].get_vector().get_array())
                            plddt_vals.append(residue['CA'].get_bfactor())
                if ca_atoms:
                    ca_positions = np.array(ca_atoms, dtype=np.float32)
                    plddt = np.array(plddt_vals, dtype=np.float32)
        except Exception:
            pass

    # Try ESMFold prediction
    if ca_positions is None and source in ('auto', 'esmfold'):
        try:
            import torch
            if _esmfold_model is not None and _esmfold_tokenizer is not None:
                model_fold = _esmfold_model
                tokenizer_fold = _esmfold_tokenizer
            else:
                from transformers import EsmForProteinFolding, AutoTokenizer
                tokenizer_fold = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
                model_fold = EsmForProteinFolding.from_pretrained(
                    "facebook/esmfold_v1", torch_dtype=torch.float32,
                ).to(device).float().eval()

            seq_trunc = seq[:512]
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=False):
                inputs = tokenizer_fold([seq_trunc], return_tensors="pt",
                                        add_special_tokens=False).to(device)
                output = model_fold(**inputs)
                plddt_raw = output.plddt[0, :len(seq_trunc)].cpu().numpy().flatten()
                if plddt_raw.max() <= 1.0:
                    plddt_raw *= 100.0
                positions = output.positions[0, :len(seq_trunc), 1, :].cpu().numpy()
                ca_positions = positions.astype(np.float32)
                plddt = plddt_raw.astype(np.float32)
        except Exception:
            pass

    # Cache results
    if ca_positions is not None:
        np.save(ca_file, ca_positions)
        np.save(plddt_file, plddt)

    return ca_positions, plddt


def _get_ca_positions_batch(
    sequences: List[str],
    uniprot_ids: Optional[List[str]] = None,
    device: str = 'cpu',
    source: str = 'auto',
) -> List[tuple]:
    """Batch wrapper for _get_ca_positions. Loads ESMFold once for all sequences."""
    n = len(sequences)
    results = [None] * n
    need_esmfold = []

    # First pass: try cache and AlphaFold DB
    for i, seq in enumerate(sequences):
        uid = uniprot_ids[i] if uniprot_ids and i < len(uniprot_ids) else None
        ca, pl = _get_ca_positions(seq, uniprot_id=uid, device=device,
                                   source='alphafold' if source == 'auto' else source)
        if ca is not None:
            results[i] = (ca, pl)
        else:
            need_esmfold.append(i)

    cached = n - len(need_esmfold)
    if cached > 0:
        print(f"    Structure cache: {cached}/{n} proteins cached")

    # Second pass: ESMFold for remaining
    if need_esmfold and source in ('auto', 'esmfold'):
        try:
            import torch
            from transformers import EsmForProteinFolding, AutoTokenizer
            print(f"    Loading ESMFold for {len(need_esmfold)} proteins...")
            tokenizer_fold = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
            model_fold = EsmForProteinFolding.from_pretrained(
                "facebook/esmfold_v1", torch_dtype=torch.float32,
            ).to(device).float().eval()

            for j, idx in enumerate(need_esmfold):
                ca, pl = _get_ca_positions(
                    sequences[idx], device=device, source='esmfold',
                    _esmfold_model=model_fold, _esmfold_tokenizer=tokenizer_fold,
                )
                results[idx] = (ca, pl) if ca is not None else (None, None)
                if (j + 1) % 20 == 0:
                    print(f"    ESMFold progress: {j + 1}/{len(need_esmfold)}")

            del model_fold
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"    WARNING: ESMFold failed: {e}")
            for idx in need_esmfold:
                if results[idx] is None:
                    results[idx] = (None, None)

    # Fill remaining None entries
    for i in range(n):
        if results[i] is None:
            results[i] = (None, None)

    return results


def _build_contact_graph(ca_positions, cutoff=8.0, min_seq_sep=5):
    """Build residue contact graph from Cα positions.

    Args:
        ca_positions: (N, 3) array of Cα coordinates.
        cutoff: Distance cutoff in Angstroms.
        min_seq_sep: Minimum sequence separation for contacts.

    Returns:
        networkx.Graph with residue nodes and contact edges.
    """
    import networkx as nx
    from scipy.spatial.distance import pdist, squareform

    n_res = len(ca_positions)
    G = nx.Graph()
    G.add_nodes_from(range(n_res))

    if n_res < 2:
        return G

    dist_matrix = squareform(pdist(ca_positions))
    seq_sep = np.abs(np.arange(n_res)[:, None] - np.arange(n_res)[None, :])
    contacts = (dist_matrix < cutoff) & (seq_sep > min_seq_sep)

    rows, cols = np.where(contacts)
    for r, c in zip(rows, cols):
        if r < c:
            G.add_edge(r, c, weight=dist_matrix[r, c], seq_sep=abs(r - c))

    return G


# =============================================================================
# PROTEIN CONTACT GRAPH FEATURES (~80D)
# =============================================================================

def _extract_graph_features(G, n_res, ca_positions):
    """Extract graph-theoretic features from a residue contact graph.

    Returns:
        np.ndarray of shape (80,).
    """
    import networkx as nx
    from scipy.stats import skew, kurtosis as scipy_kurtosis

    features = np.zeros(80, dtype=np.float64)

    if n_res < 10 or G.number_of_edges() == 0:
        return features

    degrees = np.array([d for _, d in G.degree()], dtype=np.float64)

    # --- Degree distribution (10D) ---
    features[0] = degrees.mean()
    features[1] = degrees.std()
    features[2] = degrees.max()
    features[3] = degrees.min()
    features[4] = np.median(degrees)
    features[5] = float(skew(degrees)) if degrees.std() > 0 else 0.0
    features[6] = float(scipy_kurtosis(degrees)) if degrees.std() > 0 else 0.0
    threshold = degrees.mean() + 2 * degrees.std()
    features[7] = (degrees > threshold).mean() if threshold > 0 else 0.0
    # Degree distribution entropy
    deg_counts = np.bincount(degrees.astype(int))
    deg_probs = deg_counts[deg_counts > 0] / deg_counts.sum()
    features[8] = -np.sum(deg_probs * np.log(deg_probs + 1e-12))
    # Gini coefficient of degrees
    sorted_deg = np.sort(degrees)
    n_d = len(sorted_deg)
    index = np.arange(1, n_d + 1)
    features[9] = (2 * np.sum(index * sorted_deg) / (n_d * sorted_deg.sum() + 1e-12)) - (n_d + 1) / n_d

    # --- Centrality (12D) ---
    try:
        betweenness = np.array(list(nx.betweenness_centrality(G).values()))
        features[10] = betweenness.mean()
        features[11] = betweenness.std()
        features[12] = betweenness.max()
    except Exception:
        pass

    try:
        closeness = np.array(list(nx.closeness_centrality(G).values()))
        features[13] = closeness.mean()
        features[14] = closeness.std()
        features[15] = closeness.max()
    except Exception:
        pass

    try:
        eigenvector = np.array(list(nx.eigenvector_centrality_numpy(G).values()))
        features[16] = eigenvector.mean()
        features[17] = eigenvector.std()
        features[18] = eigenvector.max()
    except Exception:
        pass

    try:
        features[19] = nx.degree_assortativity_coefficient(G)
    except Exception:
        pass
    try:
        features[20] = nx.transitivity(G)
    except Exception:
        pass

    # --- Clustering (6D) ---
    try:
        features[21] = nx.average_clustering(G)
        local_clust = np.array(list(nx.clustering(G).values()))
        features[22] = local_clust.mean()
        features[23] = local_clust.std()
    except Exception:
        pass

    try:
        triangles = np.array(list(nx.triangles(G).values()))
        features[24] = triangles.sum() / max(n_res ** 2, 1)  # normalized
    except Exception:
        pass
    try:
        features[25] = nx.graph_clique_number(G)
    except Exception:
        pass
    try:
        features[26] = sum(1 for _ in nx.find_cliques(G))
    except Exception:
        pass

    # --- Community structure (6D) ---
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
        features[27] = len(communities)
        features[28] = nx.community.modularity(G, communities)
        sizes = [len(c) / n_res for c in communities]
        features[29] = max(sizes) if sizes else 0.0
        features[30] = min(sizes) if sizes else 0.0
        features[31] = np.std(sizes) if sizes else 0.0
        # Inter-community edge fraction
        comm_map = {}
        for ci, comm in enumerate(communities):
            for node in comm:
                comm_map[node] = ci
        inter = sum(1 for u, v in G.edges() if comm_map.get(u) != comm_map.get(v))
        features[32] = inter / max(G.number_of_edges(), 1)
    except Exception:
        pass

    # --- Path/distance (10D) ---
    # Use largest connected component for path-based features
    if nx.is_connected(G):
        Gc = G
    else:
        Gc = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    try:
        features[33] = nx.average_shortest_path_length(Gc)
    except Exception:
        pass
    try:
        ecc = nx.eccentricity(Gc)
        ecc_vals = np.array(list(ecc.values()))
        features[34] = max(ecc_vals)  # diameter
        features[35] = min(ecc_vals)  # radius
        features[36] = (ecc_vals == max(ecc_vals)).sum()  # periphery size
        features[37] = (ecc_vals == min(ecc_vals)).sum()  # center size
        features[38] = ecc_vals.mean()
        features[39] = ecc_vals.std()
    except Exception:
        pass
    try:
        features[40] = nx.global_efficiency(G)
    except Exception:
        pass
    # Wiener index (normalized)
    try:
        if Gc.number_of_nodes() > 1:
            features[41] = nx.wiener_index(Gc) / max(Gc.number_of_nodes() ** 2, 1)
    except Exception:
        pass
    features[42] = nx.number_connected_components(G)

    # --- Spectral (12D) ---
    try:
        L = nx.laplacian_matrix(G).toarray().astype(np.float64)
        eigenvalues = np.sort(np.linalg.eigvalsh(L))
        # Top-10 eigenvalues (largest)
        top10 = eigenvalues[-10:] if len(eigenvalues) >= 10 else np.pad(
            eigenvalues, (10 - len(eigenvalues), 0))
        features[43:53] = top10
        # Algebraic connectivity (2nd smallest eigenvalue)
        features[53] = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
        # Spectral gap (difference between 2 largest)
        if len(eigenvalues) >= 2:
            features[54] = eigenvalues[-1] - eigenvalues[-2]
    except Exception:
        pass

    # --- Contact-specific (10D) ---
    edges = list(G.edges(data=True))
    if edges:
        seq_seps = np.array([abs(u - v) for u, v, _ in edges])
        long_mask = seq_seps > 12
        short_mask = seq_seps <= 6

        features[55] = long_mask.mean()  # long-range fraction
        features[56] = short_mask.mean()  # short-range fraction

        dists = np.array([d.get('weight', 0.0) for _, _, d in edges])
        features[57] = dists[long_mask].mean() if long_mask.any() else 0.0
        features[58] = dists[short_mask].mean() if short_mask.any() else 0.0

        # Contact order
        features[59] = seq_seps.mean() / max(n_res, 1)

    # Fraction high-degree residues
    features[60] = (degrees > threshold).mean()
    # Surface residues (degree <= 2)
    features[61] = (degrees <= 2).mean()
    # Core residues (degree > mean + sigma)
    core_threshold = degrees.mean() + degrees.std()
    features[62] = (degrees > core_threshold).mean()

    # Contact density by protein thirds
    third = n_res // 3
    if third > 0:
        n_term_contacts = sum(1 for u, v in G.edges() if u < third or v < third)
        mid_contacts = sum(1 for u, v in G.edges()
                          if (third <= u < 2 * third) or (third <= v < 2 * third))
        c_term_contacts = sum(1 for u, v in G.edges() if u >= 2 * third or v >= 2 * third)
        total = max(n_term_contacts + mid_contacts + c_term_contacts, 1)
        features[63] = n_term_contacts / total
        features[64] = mid_contacts / total
        features[65] = c_term_contacts / total

    # Pad remaining to 80
    # features[66:80] reserved for future use

    return features


def create_contact_graph_features(
    sequences: List[str],
    uniprot_ids: Optional[List[str]] = None,
    device: str = 'cpu',
    source: str = 'auto',
) -> np.ndarray:
    """Extract residue contact graph features for protein sequences (~80D).

    Builds a residue contact graph (8Å Cα cutoff, |i-j|>5 sequence separation)
    and computes graph-theoretic features: degree distribution, centrality,
    clustering, community structure, path/distance, spectral, contact-specific.

    Args:
        sequences: List of protein amino acid sequences.
        uniprot_ids: Optional UniProt IDs for AlphaFold DB structure lookup.
        device: Device for ESMFold fallback.
        source: 'auto', 'esmfold', or 'alphafold'.

    Returns:
        np.ndarray: (n_sequences, 80) feature matrix.
    """
    n = len(sequences)
    print(f"  Computing contact graph features for {n} proteins...")

    # Get structures (batch — loads ESMFold once if needed)
    struct_results = _get_ca_positions_batch(sequences, uniprot_ids, device, source)

    features = np.zeros((n, 80), dtype=np.float64)
    success = 0

    for i, (ca, plddt) in enumerate(struct_results):
        if ca is None or len(ca) < 10:
            continue
        try:
            G = _build_contact_graph(ca)
            features[i] = _extract_graph_features(G, len(ca), ca)
            success += 1
        except Exception as e:
            print(f"    WARNING: Contact graph failed for seq {i}: {e}")

        if (i + 1) % 50 == 0:
            print(f"    Progress: {i + 1}/{n}")

    print(f"  Generated contact graph features: ({n}, 80), {success}/{n} successful")
    return features.astype(np.float32)


# =============================================================================
# PROTEIN USR SHAPE FEATURES (~60D)
# =============================================================================

def _compute_usr_features(ca_positions, sequence):
    """Compute Ultrafast Shape Recognition + extended shape descriptors.

    Args:
        ca_positions: (N, 3) Cα coordinates.
        sequence: Amino acid sequence (for property-based reference points).

    Returns:
        np.ndarray of shape (60,).
    """
    from scipy.stats import skew as scipy_skew
    from scipy.stats import kurtosis as scipy_kurt

    features = np.zeros(60, dtype=np.float64)
    n = len(ca_positions)

    if n < 4:
        return features

    # --- Basic USR (12D): 4 reference points ---
    centroid = ca_positions.mean(axis=0)
    dists_to_centroid = np.linalg.norm(ca_positions - centroid, axis=1)

    closest_idx = np.argmin(dists_to_centroid)
    farthest_idx = np.argmax(dists_to_centroid)
    dists_to_farthest = np.linalg.norm(ca_positions - ca_positions[farthest_idx], axis=1)
    farthest2_idx = np.argmax(dists_to_farthest)

    ref_points = [
        centroid,
        ca_positions[closest_idx],
        ca_positions[farthest_idx],
        ca_positions[farthest2_idx],
    ]

    for j, ref in enumerate(ref_points):
        dists = np.linalg.norm(ca_positions - ref, axis=1)
        features[j * 3] = dists.mean()
        features[j * 3 + 1] = dists.var()
        features[j * 3 + 2] = float(scipy_skew(dists)) if dists.std() > 0 else 0.0

    # --- Extended reference points (12D) ---
    hydrophobic = set('IVLFCMAW')
    charged = set('DEKRH')
    seq_upper = sequence.upper().strip()

    hydro_idx = [i for i, aa in enumerate(seq_upper[:n]) if aa in hydrophobic]
    charged_idx = [i for i, aa in enumerate(seq_upper[:n]) if aa in charged]
    quarter = max(n // 4, 1)
    nterm_idx = list(range(quarter))
    cterm_idx = list(range(n - quarter, n))

    ext_groups = [hydro_idx, charged_idx, nterm_idx, cterm_idx]
    for j, idx_group in enumerate(ext_groups):
        if idx_group:
            ref = ca_positions[idx_group].mean(axis=0)
        else:
            ref = centroid
        dists = np.linalg.norm(ca_positions - ref, axis=1)
        base = 12 + j * 3
        features[base] = dists.mean()
        features[base + 1] = dists.var()
        features[base + 2] = float(scipy_skew(dists)) if dists.std() > 0 else 0.0

    # --- Higher moments (16D): kurtosis + median for all 8 refs ---
    all_refs = ref_points + [
        ca_positions[hydro_idx].mean(axis=0) if hydro_idx else centroid,
        ca_positions[charged_idx].mean(axis=0) if charged_idx else centroid,
        ca_positions[nterm_idx].mean(axis=0),
        ca_positions[cterm_idx].mean(axis=0),
    ]
    for j, ref in enumerate(all_refs):
        dists = np.linalg.norm(ca_positions - ref, axis=1)
        features[24 + j] = float(scipy_kurt(dists)) if dists.std() > 0 else 0.0
        features[32 + j] = np.median(dists)

    # --- Inertia/shape (8D) ---
    centered = ca_positions - centroid
    gyration_tensor = (centered.T @ centered) / n
    eigvals = np.sort(np.linalg.eigvalsh(gyration_tensor))[::-1]
    l1, l2, l3 = eigvals[0], eigvals[1], eigvals[2]

    features[40] = l1 - 0.5 * (l2 + l3)  # asphericity
    features[41] = l2 - l3  # acylindricity
    rg2 = l1 + l2 + l3
    features[42] = (features[40] ** 2 + 0.75 * features[41] ** 2) / max(rg2 ** 2, 1e-12)  # rel shape anisotropy
    features[43] = (l1 - l2) / max(l1 + l2, 1e-12)  # prolateness approx
    features[44] = l2 / max(l1, 1e-12)  # axis ratio 2/1
    features[45] = l3 / max(l1, 1e-12)  # axis ratio 3/1
    features[46] = l3 / max(l2, 1e-12)  # axis ratio 3/2
    features[47] = 1.0 - 3.0 * ((l1 * l2 + l2 * l3 + l1 * l3) / max(rg2 ** 2, 1e-12))  # anisotropy

    # --- Size (12D) ---
    features[48] = np.sqrt(rg2)  # radius of gyration
    all_dists = np.linalg.norm(ca_positions[:, None] - ca_positions[None, :], axis=2)
    features[49] = all_dists.max()  # max Cα distance

    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(ca_positions)
        features[50] = hull.area  # convex hull SA
        features[51] = hull.volume  # convex hull volume
        features[52] = hull.volume / max(hull.area, 1e-12)  # compactness
        features[53] = (np.pi ** (1 / 3) * (6 * hull.volume) ** (2 / 3)) / max(hull.area, 1e-12)  # sphericity
        features[54] = len(hull.vertices) / max(n, 1)  # hull vertex fraction
    except Exception:
        pass

    features[55] = all_dists.max()  # max diameter (same as [49])

    # PCA axis extents (Feret diameters)
    from numpy.linalg import svd
    _, _, Vt = svd(centered, full_matrices=False)
    projected = centered @ Vt.T
    for j in range(min(3, projected.shape[1])):
        features[56 + j] = projected[:, j].max() - projected[:, j].min()

    # End-to-end distance
    features[59] = np.linalg.norm(ca_positions[0] - ca_positions[-1])

    return features


def create_usr_shape_features(
    sequences: List[str],
    uniprot_ids: Optional[List[str]] = None,
    device: str = 'cpu',
    source: str = 'auto',
) -> np.ndarray:
    """Extract Ultrafast Shape Recognition (USR) + shape features (~60D).

    Rotation/translation invariant 3D shape descriptors from Cα coordinates.

    Args:
        sequences: List of protein amino acid sequences.
        uniprot_ids: Optional UniProt IDs for AlphaFold DB structure lookup.
        device: Device for ESMFold fallback.
        source: 'auto', 'esmfold', or 'alphafold'.

    Returns:
        np.ndarray: (n_sequences, 60) feature matrix.
    """
    n = len(sequences)
    print(f"  Computing USR shape features for {n} proteins...")

    struct_results = _get_ca_positions_batch(sequences, uniprot_ids, device, source)

    features = np.zeros((n, 60), dtype=np.float64)
    success = 0

    for i, (ca, plddt) in enumerate(struct_results):
        if ca is None or len(ca) < 4:
            continue
        try:
            features[i] = _compute_usr_features(ca, sequences[i])
            success += 1
        except Exception as e:
            print(f"    WARNING: USR failed for seq {i}: {e}")

        if (i + 1) % 50 == 0:
            print(f"    Progress: {i + 1}/{n}")

    print(f"  Generated USR shape features: ({n}, 60), {success}/{n} successful")
    return features.astype(np.float32)


# =============================================================================
# PROTEIN POCKET FEATURES (~55D) + POCKET ESM-2 EMBEDDING (1280D)
# =============================================================================

def _get_klifs_pocket_residues(uniprot_ids: List[Optional[str]]) -> dict:
    """Query KLIFS database for kinase binding pocket residue positions.

    Returns:
        Dict mapping UniProt ID -> List[int] of pocket residue indices.
    """
    import json
    from pathlib import Path

    # Cache
    seagate = Path('/Volumes/seagate/kirby/klifs')
    if seagate.parent.exists():
        cache_dir = seagate
    else:
        cache_dir = Path(os.environ.get('KIRBY_CACHE_DIR',
                         str(Path.home() / '.cache' / 'kirby'))) / 'klifs'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / 'pocket_map.json'

    pocket_map = {}
    if cache_file.exists():
        try:
            pocket_map = json.loads(cache_file.read_text())
        except Exception:
            pass

    # Find which UniProt IDs we need to look up
    to_lookup = set()
    for uid in uniprot_ids:
        if uid is not None and uid not in pocket_map:
            to_lookup.add(uid)

    if to_lookup:
        import requests
        print(f"    Querying KLIFS for {len(to_lookup)} kinases...")
        for uid in sorted(to_lookup):
            try:
                # KLIFS API: get structures for a UniProt ID
                resp = requests.get(
                    f'https://klifs.net/api/structures_list?uniprot={uid}',
                    timeout=15,
                )
                if resp.status_code == 200:
                    structures = resp.json()
                    if structures and len(structures) > 0:
                        # Use first structure's pocket residues
                        struct_id = structures[0].get('structure_ID')
                        if struct_id:
                            resp2 = requests.get(
                                f'https://klifs.net/api/structure_get_pocket?structure_ID={struct_id}',
                                timeout=15,
                            )
                            if resp2.status_code == 200:
                                pocket_data = resp2.json()
                                if pocket_data:
                                    # Extract residue numbers
                                    residues = [
                                        r.get('residue_number') for r in pocket_data
                                        if r.get('residue_number') is not None
                                    ]
                                    if residues:
                                        pocket_map[uid] = residues
                                        continue
                pocket_map[uid] = None  # Not a kinase or not in KLIFS
            except Exception:
                pocket_map[uid] = None

        # Save cache
        cache_file.write_text(json.dumps(pocket_map, indent=2))

    return pocket_map


def _run_fpocket(pdb_path: str) -> tuple:
    """Run fpocket on a PDB file to detect binding pockets.

    Returns:
        (pocket_residue_indices: List[int], pocket_info: dict) or (None, {}).
    """
    import subprocess
    import json
    from pathlib import Path

    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        return None, {}

    # Check fpocket cache
    seagate = Path('/Volumes/seagate/kirby/fpocket_cache')
    if seagate.parent.exists():
        fp_cache_dir = seagate
    else:
        fp_cache_dir = Path(os.environ.get('KIRBY_CACHE_DIR',
                           str(Path.home() / '.cache' / 'kirby'))) / 'fpocket_cache'
    fp_cache_dir.mkdir(parents=True, exist_ok=True)

    cache_key = pdb_path.stem
    cache_file = fp_cache_dir / f'{cache_key}.json'
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            return data.get('residues'), data.get('info', {})
        except Exception:
            pass

    try:
        result = subprocess.run(
            ['fpocket', '-f', str(pdb_path)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return None, {}

        # Parse fpocket output — look for pocket 1 (highest druggability)
        out_dir = pdb_path.parent / f'{pdb_path.stem}_out'
        pocket_pdb = out_dir / 'pockets' / 'pocket1_atm.pdb'
        if not pocket_pdb.exists():
            return None, {}

        residues = set()
        with open(pocket_pdb) as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    try:
                        res_num = int(line[22:26].strip())
                        residues.add(res_num)
                    except ValueError:
                        pass

        pocket_residues = sorted(residues)
        info = {'n_pockets': len(list((out_dir / 'pockets').glob('pocket*_atm.pdb')))}

        # Cache
        cache_file.write_text(json.dumps({'residues': pocket_residues, 'info': info}))

        return pocket_residues, info

    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, {}


def _get_pocket_residues(
    sequence: str,
    uniprot_id: Optional[str] = None,
    ca_positions=None,
    pdb_path: Optional[str] = None,
    dataset_hint: Optional[str] = None,
    klifs_cache: Optional[dict] = None,
) -> tuple:
    """Identify binding pocket residues using 3-tier fallback.

    Returns:
        (pocket_indices: List[int], detection_method: int)
        detection_method: 0=KLIFS, 1=fpocket, 2=contact-count heuristic.
    """
    seq = sequence.upper().strip()
    n_res = len(seq)

    # Tier 1: KLIFS (kinase datasets)
    if klifs_cache and uniprot_id and uniprot_id in klifs_cache:
        residues = klifs_cache[uniprot_id]
        if residues is not None:
            # KLIFS residue numbers are 1-indexed; convert to 0-indexed
            pocket = [r - 1 for r in residues if 0 < r <= n_res]
            if pocket:
                return pocket, 0

    # Tier 2: fpocket
    if pdb_path is not None:
        residues, info = _run_fpocket(pdb_path)
        if residues is not None and residues:
            # fpocket residues are 1-indexed PDB numbering
            pocket = [r - 1 for r in residues if 0 < r <= n_res]
            if pocket:
                return pocket, 1

    # Tier 3: Contact-count heuristic (fallback)
    if ca_positions is not None and len(ca_positions) >= 10:
        G = _build_contact_graph(ca_positions)
        degrees = np.array([d for _, d in G.degree()])
        n_pocket = min(85, max(10, int(0.2 * n_res)))
        top_indices = np.argsort(degrees)[-n_pocket:].tolist()
        top_indices.sort()
        return top_indices, 2

    # Absolute fallback: use middle 20% of sequence
    start = n_res // 4
    end = start + max(10, n_res // 5)
    return list(range(start, min(end, n_res))), 2


def _extract_pocket_features(ca_positions, plddt, sequence, pocket_indices, detection_method):
    """Extract hand-crafted pocket features.

    Returns:
        np.ndarray of shape (55,).
    """
    features = np.zeros(55, dtype=np.float64)
    seq = sequence.upper().strip()
    n_res = len(ca_positions)

    if not pocket_indices or n_res < 10:
        features[52] = detection_method
        return features

    # Ensure valid indices
    pocket_indices = [i for i in pocket_indices if 0 <= i < n_res]
    if not pocket_indices:
        features[52] = detection_method
        return features

    pocket_ca = ca_positions[pocket_indices]
    pocket_seq = ''.join(seq[i] for i in pocket_indices if i < len(seq))
    n_pocket = len(pocket_indices)

    # --- Composition (14D) ---
    hydrophobic = set('IVLFCMAW')
    charged_pos = set('KRH')
    charged_neg = set('DE')
    polar = set('STNQ')
    aromatic = set('FYW')
    small = set('GAST')
    large = set('FYWKRH')

    features[0] = sum(1 for aa in pocket_seq if aa in hydrophobic) / max(n_pocket, 1)
    features[1] = sum(1 for aa in pocket_seq if aa in charged_pos) / max(n_pocket, 1)
    features[2] = sum(1 for aa in pocket_seq if aa in charged_neg) / max(n_pocket, 1)
    features[3] = sum(1 for aa in pocket_seq if aa in polar) / max(n_pocket, 1)
    features[4] = sum(1 for aa in pocket_seq if aa in aromatic) / max(n_pocket, 1)
    features[5] = sum(1 for aa in pocket_seq if aa in small) / max(n_pocket, 1)
    features[6] = sum(1 for aa in pocket_seq if aa in large) / max(n_pocket, 1)
    features[7] = sum(1 for aa in pocket_seq if aa == 'C') / max(n_pocket, 1)
    features[8] = sum(1 for aa in pocket_seq if aa == 'P') / max(n_pocket, 1)
    features[9] = sum(1 for aa in pocket_seq if aa == 'G') / max(n_pocket, 1)

    # Mean/std hydrophobicity and volume of pocket residues
    hydro_vals = [AA_PROPERTIES.get(aa, {}).get('hydrophobicity', 0.0) for aa in pocket_seq]
    vol_vals = [AA_PROPERTIES.get(aa, {}).get('volume', 100.0) for aa in pocket_seq]
    if hydro_vals:
        features[10] = np.mean(hydro_vals)
        features[11] = np.std(hydro_vals)
    if vol_vals:
        features[12] = np.mean(vol_vals)
        features[13] = np.std(vol_vals)

    # --- Geometry (16D) ---
    features[14] = n_pocket / max(n_res, 1)  # pocket fraction

    try:
        from scipy.spatial import ConvexHull
        if n_pocket >= 4:
            hull = ConvexHull(pocket_ca)
            features[15] = hull.volume  # pocket volume
            features[16] = hull.area  # pocket SA
            features[17] = hull.volume / max(hull.area, 1e-12)  # compactness
    except Exception:
        pass

    pocket_centroid = pocket_ca.mean(axis=0)
    protein_centroid = ca_positions.mean(axis=0)

    # Depth: mean dist from pocket residues to protein surface estimate
    # (simplified: mean dist from pocket centroid to all non-pocket residues)
    non_pocket = np.array([i for i in range(n_res) if i not in set(pocket_indices)])
    if len(non_pocket) > 0:
        surface_dists = np.linalg.norm(ca_positions[non_pocket] - pocket_centroid, axis=1)
        features[18] = surface_dists.mean()

    # Openness: fraction of pocket residues with low contact count
    G_full = _build_contact_graph(ca_positions)
    pocket_degrees = np.array([G_full.degree(i) for i in pocket_indices])
    features[19] = (pocket_degrees <= 2).mean()

    # Elongation via PCA of pocket
    if n_pocket >= 3:
        centered = pocket_ca - pocket_centroid
        cov = centered.T @ centered / n_pocket
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        features[20] = eigvals[1] / max(eigvals[0], 1e-12)
        features[21] = eigvals[2] / max(eigvals[0], 1e-12)
        features[22] = eigvals[2] / max(eigvals[1], 1e-12)

    # Distance from pocket center to protein centroid
    features[23] = np.linalg.norm(pocket_centroid - protein_centroid)

    # Pocket Rg
    features[24] = np.sqrt(((pocket_ca - pocket_centroid) ** 2).sum(axis=1).mean())

    # Pocket span (max Ca distance within pocket)
    if n_pocket >= 2:
        pocket_dists = np.linalg.norm(pocket_ca[:, None] - pocket_ca[None, :], axis=2)
        features[25] = pocket_dists.max()

    # Pocket fraction of total protein
    features[26] = n_pocket / max(n_res, 1)

    # --- Pocket graph (12D) ---
    pocket_set = set(pocket_indices)
    pocket_subgraph = G_full.subgraph(pocket_indices)

    if pocket_subgraph.number_of_edges() > 0:
        sub_degrees = np.array([d for _, d in pocket_subgraph.degree()], dtype=float)
        features[27] = sub_degrees.mean()
        features[28] = sub_degrees.std()

        import networkx as nx
        features[29] = nx.average_clustering(pocket_subgraph)

        # Boundary edges / pocket edges
        boundary = sum(1 for u, v in G_full.edges()
                       if (u in pocket_set) != (v in pocket_set))
        features[30] = boundary / max(pocket_subgraph.number_of_edges(), 1)

        # Hub fraction
        hub_thresh = sub_degrees.mean() + sub_degrees.std()
        features[31] = (sub_degrees > hub_thresh).mean() if hub_thresh > 0 else 0.0

        features[32] = nx.density(pocket_subgraph)

        try:
            L = nx.laplacian_matrix(pocket_subgraph).toarray().astype(float)
            eigvals_p = np.sort(np.linalg.eigvalsh(L))
            features[33] = eigvals_p[1] if len(eigvals_p) > 1 else 0.0
        except Exception:
            pass

        features[34] = nx.number_connected_components(pocket_subgraph)

        if nx.is_connected(pocket_subgraph):
            try:
                features[35] = nx.diameter(pocket_subgraph)
            except Exception:
                pass

        try:
            betw = np.array(list(nx.betweenness_centrality(pocket_subgraph).values()))
            features[36] = betw.mean()
        except Exception:
            pass

        try:
            close = np.array(list(nx.closeness_centrality(pocket_subgraph).values()))
            features[37] = close.mean()
        except Exception:
            pass

        # Interface edge count normalized
        features[38] = boundary / max(n_res, 1)

    # --- Location (8D) ---
    third = n_res // 3
    if third > 0:
        features[39] = sum(1 for i in pocket_indices if i < third) / max(n_pocket, 1)
        features[40] = sum(1 for i in pocket_indices if third <= i < 2 * third) / max(n_pocket, 1)
        features[41] = sum(1 for i in pocket_indices if i >= 2 * third) / max(n_pocket, 1)

    # Max gap between consecutive pocket residues
    sorted_pocket = sorted(pocket_indices)
    if len(sorted_pocket) > 1:
        gaps = np.diff(sorted_pocket)
        features[42] = gaps.max() / max(n_res, 1)

    pos_normalized = np.array(sorted_pocket) / max(n_res, 1)
    features[43] = pos_normalized.mean()
    features[44] = pos_normalized.std()
    features[45] = pos_normalized.min()
    features[46] = pos_normalized.max()

    # --- Metadata (3D) ---
    features[47] = detection_method
    if plddt is not None and len(plddt) > 0:
        pocket_plddt = plddt[[i for i in pocket_indices if i < len(plddt)]]
        if len(pocket_plddt) > 0:
            features[48] = pocket_plddt.mean()
    features[49] = 1  # number of pockets (simplified)

    # features[50:55] reserved

    return features


def create_pocket_features(
    sequences: List[str],
    uniprot_ids: Optional[List[str]] = None,
    protein_ids: Optional[List[str]] = None,
    device: str = 'cpu',
    source: str = 'auto',
    dataset_hint: Optional[str] = None,
) -> np.ndarray:
    """Extract binding pocket features for protein sequences (~55D).

    Uses 3-tier pocket detection: KLIFS (kinases) → fpocket → contact heuristic.

    Args:
        sequences: List of protein amino acid sequences.
        uniprot_ids: Optional resolved UniProt IDs.
        protein_ids: Optional raw protein identifiers (gene names).
        device: Device for ESMFold fallback.
        source: 'auto', 'esmfold', or 'alphafold'.
        dataset_hint: Dataset name hint for pocket strategy (e.g. 'davis', 'kiba').

    Returns:
        np.ndarray: (n_sequences, 55) feature matrix.
    """
    n = len(sequences)
    print(f"  Computing pocket features for {n} proteins...")

    # Resolve UniProt IDs if not provided
    if uniprot_ids is None and protein_ids is not None:
        uniprot_ids = _resolve_uniprot_ids(protein_ids)

    # Pre-fetch KLIFS pocket data for kinase datasets
    klifs_cache = None
    if dataset_hint in ('davis', 'kiba') or (uniprot_ids and any(u is not None for u in uniprot_ids)):
        unique_uids = list(set(u for u in (uniprot_ids or []) if u is not None))
        if unique_uids:
            klifs_cache = _get_klifs_pocket_residues(unique_uids)

    # Get structures
    struct_results = _get_ca_positions_batch(sequences, uniprot_ids, device, source)

    features = np.zeros((n, 55), dtype=np.float64)
    success = 0

    for i, (ca, plddt) in enumerate(struct_results):
        if ca is None:
            continue
        try:
            uid = uniprot_ids[i] if uniprot_ids and i < len(uniprot_ids) else None
            pocket_indices, method = _get_pocket_residues(
                sequences[i], uniprot_id=uid, ca_positions=ca,
                dataset_hint=dataset_hint, klifs_cache=klifs_cache,
            )
            features[i] = _extract_pocket_features(ca, plddt, sequences[i], pocket_indices, method)
            success += 1
        except Exception as e:
            print(f"    WARNING: Pocket features failed for seq {i}: {e}")

        if (i + 1) % 50 == 0:
            print(f"    Progress: {i + 1}/{n}")

    print(f"  Generated pocket features: ({n}, 55), {success}/{n} successful")
    return features.astype(np.float32)


def create_esm2_pocket_embeddings(
    sequences: List[str],
    uniprot_ids: Optional[List[str]] = None,
    protein_ids: Optional[List[str]] = None,
    device: str = 'cpu',
    dataset_hint: Optional[str] = None,
) -> np.ndarray:
    """Extract pocket-local ESM-2 embeddings (1280D).

    Mean-pools ESM-2 per-residue embeddings only at pocket residue positions.
    Requires create_esm2_protein_embeddings() to have been run first.

    Falls back to full-sequence pooling if pocket detection fails or if
    per-residue ESM-2 embeddings are not cached.

    Args:
        sequences: List of protein amino acid sequences.
        uniprot_ids: Optional resolved UniProt IDs.
        protein_ids: Optional raw protein identifiers.
        device: Device for ESMFold fallback (structure for pocket detection).
        dataset_hint: Dataset name hint for pocket strategy.

    Returns:
        np.ndarray: (n_sequences, 1280) embeddings.
    """
    n = len(sequences)
    print(f"  Computing pocket-local ESM-2 embeddings for {n} proteins...")

    # Resolve UniProt IDs if needed
    if uniprot_ids is None and protein_ids is not None:
        uniprot_ids = _resolve_uniprot_ids(protein_ids)

    # Pre-fetch KLIFS
    klifs_cache = None
    if uniprot_ids:
        unique_uids = list(set(u for u in uniprot_ids if u is not None))
        if unique_uids:
            klifs_cache = _get_klifs_pocket_residues(unique_uids)

    # Get structures for pocket detection
    struct_results = _get_ca_positions_batch(sequences, uniprot_ids, device, 'auto')

    embeddings = np.zeros((n, 1280), dtype=np.float32)
    pocket_pooled = 0
    full_pooled = 0

    for i, seq in enumerate(sequences):
        # Load per-residue ESM-2
        per_res = load_esm2_per_residue(seq)

        if per_res is None:
            # Fall back to mean embedding
            cache_file = _esm2_per_residue_cache_dir() / f'{_seq_hash(seq)}_mean.npy'
            if cache_file.exists():
                embeddings[i] = np.load(cache_file)
                full_pooled += 1
            continue

        # Get pocket residues
        ca, plddt = struct_results[i]
        uid = uniprot_ids[i] if uniprot_ids and i < len(uniprot_ids) else None

        if ca is not None:
            pocket_indices, _ = _get_pocket_residues(
                seq, uniprot_id=uid, ca_positions=ca,
                dataset_hint=dataset_hint, klifs_cache=klifs_cache,
            )
            # Filter to valid indices within per-residue array
            valid_pocket = [j for j in pocket_indices if j < len(per_res)]
            if valid_pocket:
                embeddings[i] = per_res[valid_pocket].mean(axis=0)
                pocket_pooled += 1
                continue

        # Fallback: full sequence pooling
        embeddings[i] = per_res.mean(axis=0)
        full_pooled += 1

    print(f"  Generated pocket ESM-2 embeddings: ({n}, 1280), "
          f"{pocket_pooled} pocket-pooled, {full_pooled} full-sequence")
    return embeddings


def create_pocket_composition_features(
    sequences: List[str],
    uniprot_ids: Optional[List[str]] = None,
    protein_ids: Optional[List[str]] = None,
    device: str = 'cpu',
    dataset_hint: Optional[str] = None,
) -> np.ndarray:
    """Extract amino acid composition of binding pocket only (~20D).

    Lightweight version of pocket features — no graph or geometry, just AA
    composition at pocket residue positions.

    Returns:
        np.ndarray: (n_sequences, 20) feature matrix (one per standard AA).
    """
    n = len(sequences)
    print(f"  Computing pocket composition features for {n} proteins...")

    if uniprot_ids is None and protein_ids is not None:
        uniprot_ids = _resolve_uniprot_ids(protein_ids)

    klifs_cache = None
    if uniprot_ids:
        unique_uids = list(set(u for u in uniprot_ids if u is not None))
        if unique_uids:
            klifs_cache = _get_klifs_pocket_residues(unique_uids)

    struct_results = _get_ca_positions_batch(sequences, uniprot_ids, device, 'auto')

    features = np.zeros((n, 20), dtype=np.float32)
    aa_to_idx = {aa: j for j, aa in enumerate(STANDARD_AAS)}

    for i, seq in enumerate(sequences):
        ca, _ = struct_results[i]
        uid = uniprot_ids[i] if uniprot_ids and i < len(uniprot_ids) else None
        seq_upper = seq.upper().strip()

        if ca is not None:
            pocket_indices, _ = _get_pocket_residues(
                seq, uniprot_id=uid, ca_positions=ca,
                dataset_hint=dataset_hint, klifs_cache=klifs_cache,
            )
        else:
            # Fallback: middle 20%
            n_res = len(seq_upper)
            start = n_res // 4
            pocket_indices = list(range(start, min(start + max(10, n_res // 5), n_res)))

        pocket_seq = ''.join(seq_upper[j] for j in pocket_indices if j < len(seq_upper))
        if pocket_seq:
            for aa in pocket_seq:
                if aa in aa_to_idx:
                    features[i, aa_to_idx[aa]] += 1
            features[i] /= len(pocket_seq)

    print(f"  Generated pocket composition features: ({n}, 20)")
    return features


def create_esm2_protein_embeddings(
    sequences: List[str],
    model_name: str = 'facebook/esm2_t33_650M_UR50D',
    device: str = 'cpu',
    batch_size: int = 4,
    cache_per_residue: bool = True,
) -> np.ndarray:
    """Extract ESM-2 mean-pooled embeddings for protein sequences.

    Args:
        sequences: List of protein amino acid sequences.
        model_name: HuggingFace ESM-2 model name.
        device: 'cpu' or 'cuda'.
        batch_size: Sequences per batch (proteins can be long, use small batch).
        cache_per_residue: If True, cache per-residue embeddings to disk for
            later use by pocket-local features, contact graphs, etc.

    Returns:
        np.ndarray: (n_sequences, 1280) embeddings.
    """
    import torch
    try:
        from transformers import AutoTokenizer, EsmModel
    except ImportError:
        raise ImportError("ESM-2 requires transformers: pip install transformers")

    per_res_dir = _esm2_per_residue_cache_dir() if cache_per_residue else None
    n = len(sequences)

    # Check which sequences already have cached per-residue embeddings
    all_embeddings = [None] * n
    need_compute = []
    if per_res_dir is not None:
        for i, seq in enumerate(sequences):
            cache_file = per_res_dir / f'{_seq_hash(seq)}_mean.npy'
            if cache_file.exists():
                all_embeddings[i] = np.load(cache_file)
            else:
                need_compute.append(i)
    else:
        need_compute = list(range(n))

    n_cached = n - len(need_compute)
    if n_cached > 0:
        print(f"  ESM-2: {n_cached}/{n} sequences cached, computing {len(need_compute)} remaining...")

    if not need_compute:
        result = np.stack(all_embeddings, axis=0)
        print(f"  Generated ESM-2 protein embeddings: {result.shape} (all cached)")
        return result

    print(f"  Loading ESM-2 ({model_name.split('/')[-1]}) for protein embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device).eval()

    with torch.no_grad():
        for batch_start in range(0, len(need_compute), batch_size):
            batch_indices = need_compute[batch_start:batch_start + batch_size]
            batch = [sequences[i].upper().strip()[:1022] for i in batch_indices]
            inputs = tokenizer(
                batch, return_tensors='pt', padding=True,
                truncation=True, max_length=1024
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state  # (B, L, 1280)

            # Mean pool excluding BOS/EOS
            attention_mask = inputs['attention_mask'].clone()
            attention_mask[:, 0] = 0
            for j in range(len(batch)):
                last_tok = attention_mask[j].sum().item()
                if last_tok > 0:
                    attention_mask[j, int(last_tok)] = 0
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_emb = (hidden * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            mean_emb = (sum_emb / count).cpu().numpy()

            # Save per-residue and mean embeddings per sequence
            for j, idx in enumerate(batch_indices):
                all_embeddings[idx] = mean_emb[j]
                if per_res_dir is not None:
                    sh = _seq_hash(sequences[idx])
                    np.save(per_res_dir / f'{sh}_mean.npy', mean_emb[j])
                    # Save per-residue (excluding BOS/EOS padding)
                    seq_len = int(attention_mask[j].sum().item())
                    if seq_len > 0:
                        # Residue embeddings: positions 1..seq_len (skip BOS at 0)
                        per_res = hidden[j, 1:1 + seq_len].cpu().numpy()
                        np.save(per_res_dir / f'{sh}_per_residue.npy', per_res)

            done = min(batch_start + batch_size, len(need_compute))
            if done < len(need_compute):
                print(f"    Progress: {done}/{len(need_compute)}")

    result = np.stack(all_embeddings, axis=0)
    print(f"  Generated ESM-2 protein embeddings: {result.shape}")
    return result


def load_esm2_per_residue(sequence: str) -> Optional[np.ndarray]:
    """Load cached per-residue ESM-2 embeddings for a single protein.

    Returns:
        np.ndarray of shape (seq_len, 1280) or None if not cached.
        Call create_esm2_protein_embeddings() first to populate the cache.
    """
    cache_file = _esm2_per_residue_cache_dir() / f'{_seq_hash(sequence)}_per_residue.npy'
    if cache_file.exists():
        return np.load(cache_file)
    return None


def create_esm2_midlayers_protein_embeddings(
    sequences: List[str],
    model_name: str = 'facebook/esm2_t33_650M_UR50D',
    device: str = 'cpu',
    batch_size: int = 4,
    layer_start: int = 20,
    layer_end: int = 33,
) -> np.ndarray:
    """ESM-2 intermediate layer averaged embeddings for protein sequences.

    Averages hidden states from layers [layer_start, layer_end] instead of using
    the final layer only. Published evidence shows this improves unsupervised
    clustering ARI from 0.268 → 0.354 for ESM-2 650M.

    Args:
        sequences: List of protein amino acid sequences.
        model_name: HuggingFace ESM-2 model name.
        device: 'cpu' or 'cuda'.
        batch_size: Sequences per batch.
        layer_start: First layer to average (0-indexed).
        layer_end: Last layer to average (inclusive).

    Returns:
        np.ndarray: (n_sequences, 1280) embeddings.
    """
    import torch
    try:
        from transformers import AutoTokenizer, EsmModel
    except ImportError:
        raise ImportError("ESM-2 requires transformers: pip install transformers")

    print(f"  Loading ESM-2 midlayers ({model_name.split('/')[-1]}, layers {layer_start}-{layer_end})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name, output_hidden_states=True).to(device).eval()

    all_embeddings = []
    n = len(sequences)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = [s.upper().strip()[:1022] for s in sequences[start:start + batch_size]]
            inputs = tokenizer(
                batch, return_tensors='pt', padding=True,
                truncation=True, max_length=1024
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            # Average selected hidden layers
            selected = torch.stack(
                outputs.hidden_states[layer_start:layer_end + 1], dim=0
            ).mean(dim=0)  # (B, L, D)
            # Mean pool excluding BOS/EOS
            attention_mask = inputs['attention_mask'].clone()
            attention_mask[:, 0] = 0
            for j in range(len(batch)):
                last_tok = attention_mask[j].sum().item()
                if last_tok > 0:
                    attention_mask[j, int(last_tok)] = 0
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_emb = (selected * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            mean_emb = (sum_emb / count).cpu().numpy()
            all_embeddings.append(mean_emb)

            end = min(start + batch_size, n)
            if end < n and end % 50 == 0:
                print(f"    Progress: {end}/{n}")

    result = np.concatenate(all_embeddings, axis=0)
    print(f"  Generated ESM-2 midlayer protein embeddings: {result.shape}")
    return result


def create_esmc_protein_embeddings(
    sequences: List[str],
    device: str = 'cpu',
) -> np.ndarray:
    """Extract ESM-C 600M mean-pooled embeddings for protein sequences.

    Drop-in ESM-2 replacement. Matches ESM-2 3B quality at 5x fewer params.
    Uses EvolutionaryScale's `esm` package.

    Note: Non-commercial license (EvolutionaryScale Cambrian).

    Args:
        sequences: List of protein amino acid sequences.
        device: 'cpu' or 'cuda'.

    Returns:
        np.ndarray: (n_sequences, 1152) embeddings.

    Requires: pip install esm
    """
    import torch
    try:
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
    except ImportError:
        raise ImportError("ESM-C requires the esm package: pip install esm")

    print("  Loading ESM-C 600M for protein embeddings...")
    model = ESMC.from_pretrained("esmc_600m").to(device)
    model.eval()

    config = LogitsConfig(sequence=True, return_embeddings=True)
    all_embeddings = []
    n = len(sequences)

    with torch.no_grad():
        for i, seq in enumerate(sequences):
            seq = seq.upper().strip()[:2048]  # ESM-C max length
            protein = ESMProtein(sequence=seq)
            protein_tensor = model.encode(protein)
            output = model.logits(protein_tensor, config)
            # output.embeddings: (1, L, 1152) — mean-pool excluding BOS/EOS
            emb = output.embeddings[:, 1:-1, :].mean(dim=1).squeeze(0).cpu().numpy()
            all_embeddings.append(emb)

            if (i + 1) % 50 == 0:
                print(f"    Progress: {i + 1}/{n}")

    result = np.vstack(all_embeddings)
    print(f"  Generated ESM-C protein embeddings: {result.shape}")
    return result


def create_esm3_protein_embeddings(
    sequences: List[str],
    device: str = 'cpu',
) -> np.ndarray:
    """Extract ESM3 mean-pooled embeddings for protein sequences.

    ESM3 is a multimodal protein language model from EvolutionaryScale that
    reasons over sequence, structure, and function simultaneously.
    Uses the same `esm` package as ESM-C.

    Args:
        sequences: List of protein amino acid sequences.
        device: 'cpu' or 'cuda'.

    Returns:
        np.ndarray: (n_sequences, 1536) embeddings.

    Requires: pip install esm (Python >= 3.10)
    """
    import torch
    try:
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein, LogitsConfig
    except ImportError:
        raise ImportError("ESM3 requires the esm package: pip install esm")

    print("  Loading ESM3 (esm3_sm_open_v1) for protein embeddings...")
    model = ESM3.from_pretrained("esm3_sm_open_v1")
    if device == 'cuda' and not torch.cuda.is_bf16_supported():
        model = model.half()
    model = model.to(device).eval()

    config = LogitsConfig(sequence=True, return_embeddings=True)
    all_embeddings = []
    n = len(sequences)

    with torch.no_grad():
        for i, seq in enumerate(sequences):
            seq = seq.upper().strip()[:2048]  # ESM3 max length
            try:
                protein = ESMProtein(sequence=seq)
                protein_tensor = model.encode(protein)
                output = model.logits(protein_tensor, config)
                # output.embeddings: (1, L, 1536) — mean-pool excluding BOS/EOS
                emb = output.embeddings[:, 1:-1, :].mean(dim=1).squeeze(0).cpu().numpy()
                all_embeddings.append(emb)
            except Exception as e:
                print(f"    WARNING: ESM3 failed on seq {i} (len={len(seq)}): {e}")
                all_embeddings.append(np.zeros(1536, dtype=np.float32))

            if (i + 1) % 50 == 0:
                print(f"    Progress: {i + 1}/{n}")

    result = np.vstack(all_embeddings)
    print(f"  Generated ESM3 protein embeddings: {result.shape}")
    return result


# ── AIDO.Protein-16B (MoE protein foundation model) ──────────────────────────

# Global singletons to avoid reloading the large model
_AIDO_MODEL = None
_AIDO_TOKENIZER = None


def create_aido_protein_embeddings(
    sequences: List[str],
    model_name: str = 'genbio-ai/AIDO.Protein-16B',
    device: str = 'cpu',
    batch_size: int = 1,
) -> np.ndarray:
    """Extract AIDO.Protein-16B mean-pooled embeddings for protein sequences.

    AIDO.Protein-16B is a Mixture-of-Experts (MoE) protein foundation model
    from GenBio AI with 16B total parameters but only ~4B active per token
    (top-2 routing across 8 experts). Uses RoPE positional embeddings and
    SwiGLU activation.

    WARNING: This model requires ~32GB VRAM in float16. Use batch_size=1
    to avoid OOM. On CPU this will be extremely slow. CUDA strongly
    recommended.

    Args:
        sequences: List of protein amino acid sequences.
        model_name: HuggingFace model name.
        device: 'cpu' or 'cuda'. CUDA strongly recommended.
        batch_size: Sequences per batch (default 1 due to model size).

    Returns:
        np.ndarray: (n_sequences, 2304) embeddings.

    Requires: pip install transformers torch
    Model auto-downloaded: genbio-ai/AIDO.Protein-16B (~64GB)
    """
    import torch
    global _AIDO_MODEL, _AIDO_TOKENIZER

    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError(
            "AIDO.Protein requires transformers: pip install transformers"
        )

    if _AIDO_MODEL is None:
        print(f"  Loading AIDO.Protein-16B ({model_name.split('/')[-1]}, float16)...")
        print(f"  WARNING: Model requires ~32GB VRAM in float16.")
        _AIDO_TOKENIZER = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        _AIDO_MODEL = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float16
        ).to(device).eval()

    model = _AIDO_MODEL
    tokenizer = _AIDO_TOKENIZER

    all_embeddings = []
    n = len(sequences)
    embed_dim = 2304  # model.config.hidden_size

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = [
                s.upper().strip()[:2046]  # max_position_embeddings=2048, leave room for special tokens
                for s in sequences[start:start + batch_size]
            ]
            for j, seq in enumerate(batch):
                idx = start + j
                try:
                    inputs = tokenizer(
                        seq, return_tensors='pt', padding=False,
                        truncation=True, max_length=2048,
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    # last_hidden_state: (1, L, 2304)
                    # Mean pool excluding special tokens (SEP at end, PAD=0)
                    hidden = outputs.last_hidden_state  # (1, L, 2304)
                    if 'attention_mask' in inputs:
                        mask = inputs['attention_mask'].clone()
                    else:
                        mask = torch.ones(hidden.shape[:2], device=device)
                    # Exclude last token (SEP) if present
                    n_tokens = mask[0].sum().item()
                    if n_tokens > 1:
                        mask[0, int(n_tokens) - 1] = 0
                    mask = mask.unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    emb = pooled.squeeze(0).cpu().float().numpy()
                    all_embeddings.append(emb)
                except Exception as e:
                    print(f"    WARNING: AIDO.Protein failed on seq {idx} "
                          f"(len={len(seq)}): {e}")
                    all_embeddings.append(np.zeros(embed_dim, dtype=np.float32))

            end = min(start + batch_size, n)
            if end < n and end % 20 == 0:
                print(f"    Progress: {end}/{n}")

    result = np.vstack(all_embeddings)
    print(f"  Generated AIDO.Protein embeddings: {result.shape}")
    return result


# ── SaProt (structure-aware protein LM) ──────────────────────────────────────

# Global singletons for ProstT5 and SaProt models (avoid re-loading)
_PROSTT5_MODEL = None
_PROSTT5_TOKENIZER = None
_SAPROT_MODEL = None
_SAPROT_TOKENIZER = None


def _predict_3di_with_prostt5(
    sequences: List[str],
    device: str = 'cpu',
) -> List[str]:
    """Predict Foldseek 3Di structural tokens from amino acid sequences using ProstT5.

    ProstT5 (Rostlab/ProstT5) translates AA sequences to 3Di tokens directly
    without requiring structure prediction. Pure Python, no binary dependencies.

    Args:
        sequences: List of protein amino acid sequences.
        device: 'cpu' or 'cuda'.

    Returns:
        List of 3Di strings (one per input sequence).
    """
    import torch
    global _PROSTT5_MODEL, _PROSTT5_TOKENIZER

    try:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
    except ImportError:
        raise ImportError("ProstT5 requires transformers: pip install transformers")

    if _PROSTT5_MODEL is None:
        print("  Loading ProstT5 (Rostlab/ProstT5) for 3Di prediction...")
        _PROSTT5_TOKENIZER = T5Tokenizer.from_pretrained(
            'Rostlab/ProstT5', do_lower_case=False
        )
        _PROSTT5_MODEL = T5ForConditionalGeneration.from_pretrained(
            'Rostlab/ProstT5'
        ).to(device).eval()

    model = _PROSTT5_MODEL
    tokenizer = _PROSTT5_TOKENIZER
    threedi_seqs = []

    with torch.no_grad():
        for i, seq in enumerate(sequences):
            try:
                seq_clean = seq.upper().strip()[:1000]  # ProstT5 practical limit
                # ProstT5 expects: '<AA2fold> ' prefix + space-separated residues
                input_str = '<AA2fold> ' + ' '.join(list(seq_clean))
                inputs = tokenizer(
                    input_str, return_tensors='pt',
                    add_special_tokens=True, max_length=2048,
                    truncation=True,
                ).to(device)
                output = model.generate(
                    **inputs,
                    max_new_tokens=len(seq_clean) + 10,
                    do_sample=False,
                )
                # Decode 3Di tokens, strip special tokens
                decoded = tokenizer.decode(output[0], skip_special_tokens=True)
                # ProstT5 outputs space-separated 3Di tokens
                threedi = decoded.replace(' ', '')
                # Ensure length matches (truncate or pad with '#')
                if len(threedi) < len(seq_clean):
                    threedi = threedi + '#' * (len(seq_clean) - len(threedi))
                elif len(threedi) > len(seq_clean):
                    threedi = threedi[:len(seq_clean)]
                threedi_seqs.append(threedi)
            except Exception as e:
                print(f"    WARNING: ProstT5 failed on seq {i} (len={len(seq)}): {e}")
                threedi_seqs.append('#' * len(seq.strip()))

            if (i + 1) % 20 == 0:
                print(f"    ProstT5 progress: {i + 1}/{len(sequences)}")

    return threedi_seqs


def _interleave_aa_3di(aa_seq: str, threedi_seq: str) -> str:
    """Interleave amino acid and 3Di sequences for SaProt input.

    SaProt expects alternating AA and 3Di tokens: "MdTpKv..." where
    uppercase = AA, lowercase = 3Di.

    Args:
        aa_seq: Amino acid sequence (e.g., "MTK...").
        threedi_seq: 3Di token sequence (e.g., "dpv...").

    Returns:
        Interleaved string (e.g., "MdTpKv...").
    """
    min_len = min(len(aa_seq), len(threedi_seq))
    interleaved = []
    for i in range(min_len):
        interleaved.append(aa_seq[i].upper())
        interleaved.append(threedi_seq[i].lower())
    return ''.join(interleaved)


def create_saprot_protein_embeddings(
    sequences: List[str],
    device: str = 'cpu',
    batch_size: int = 4,
) -> np.ndarray:
    """Extract SaProt 650M mean-pooled embeddings for protein sequences.

    SaProt is a structure-aware protein language model that uses Foldseek 3Di
    structural tokens interleaved with amino acid sequences. 3Di tokens are
    predicted using ProstT5 (pure Python, no binary dependencies).

    Args:
        sequences: List of protein amino acid sequences.
        device: 'cpu' or 'cuda'.
        batch_size: Sequences per batch.

    Returns:
        np.ndarray: (n_sequences, 1280) embeddings.

    Requires: pip install transformers (HuggingFace)
    Models auto-downloaded: westlake-repl/SaProt_650M_AF2 (~2.5GB),
                            Rostlab/ProstT5 (~5.5GB)
    """
    import torch
    global _SAPROT_MODEL, _SAPROT_TOKENIZER

    try:
        from transformers import EsmTokenizer, EsmModel
    except ImportError:
        raise ImportError("SaProt requires transformers: pip install transformers")

    # Step 1: Predict 3Di tokens for all sequences via ProstT5
    print("  Step 1/2: Predicting 3Di structural tokens with ProstT5...")
    threedi_seqs = _predict_3di_with_prostt5(sequences, device)

    # Step 2: Interleave AA + 3Di
    sa_seqs = [
        _interleave_aa_3di(s.upper().strip(), di)
        for s, di in zip(sequences, threedi_seqs)
    ]

    # Step 3: Load SaProt model
    if _SAPROT_MODEL is None:
        print("  Step 2/2: Loading SaProt (westlake-repl/SaProt_650M_AF2)...")
        _SAPROT_TOKENIZER = EsmTokenizer.from_pretrained(
            'westlake-repl/SaProt_650M_AF2'
        )
        _SAPROT_MODEL = EsmModel.from_pretrained(
            'westlake-repl/SaProt_650M_AF2'
        ).to(device).eval()

    model = _SAPROT_MODEL
    tokenizer = _SAPROT_TOKENIZER

    all_embeddings = []
    n = len(sequences)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = sa_seqs[start:start + batch_size]
            batch_embs = []
            for j, sa_seq in enumerate(batch):
                idx = start + j
                try:
                    inputs = tokenizer(
                        sa_seq, return_tensors='pt', padding=False,
                        truncation=True, max_length=2048,
                    ).to(device)
                    outputs = model(**inputs)
                    # Mean pool excluding BOS/EOS
                    emb = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1)
                    batch_embs.append(emb.squeeze(0).cpu().numpy())
                except Exception as e:
                    print(f"    WARNING: SaProt failed on seq {idx}: {e}")
                    batch_embs.append(np.zeros(1280, dtype=np.float32))
            all_embeddings.extend(batch_embs)

            end = min(start + batch_size, n)
            if end < n and end % 20 == 0:
                print(f"    SaProt progress: {end}/{n}")

    result = np.vstack(all_embeddings)
    print(f"  Generated SaProt protein embeddings: {result.shape}")
    return result


# ── Boltz-2 (structure prediction embeddings) ────────────────────────────────


def create_boltz2_protein_embeddings(
    sequences: List[str],
    device: str = 'cpu',
    cache_dir: str = None,
) -> np.ndarray:
    """Extract Boltz-2 single-representation protein embeddings (384D).

    Boltz-2 is a structure prediction model. We use its CLI `boltz predict`
    with `--write_embeddings` to extract per-residue single representations,
    then mean-pool to a fixed 384D vector per sequence.

    Embeddings are cached to disk keyed on sequence hash to avoid re-running.

    Note: Boltz-2 requires CUDA for practical use.

    Args:
        sequences: List of protein amino acid sequences.
        device: 'cpu' or 'cuda' (CUDA strongly recommended).
        cache_dir: Directory to cache embeddings. Defaults to ~/.cache/kirby/boltz2/.

    Returns:
        np.ndarray: (n_sequences, 384) embeddings.

    Requires: pip install boltz (CUDA recommended)
    """
    import subprocess
    import tempfile
    import hashlib as _hashlib
    from pathlib import Path

    try:
        import boltz  # noqa: F401
    except ImportError:
        raise ImportError("Boltz-2 requires the boltz package: pip install boltz")

    if cache_dir is None:
        cache_dir = str(Path(os.environ.get(
            'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')
        )) / 'boltz2')
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    n = len(sequences)
    all_embeddings = []
    print(f"  Generating Boltz-2 embeddings for {n} proteins...")

    for i, seq in enumerate(sequences):
        seq_clean = seq.upper().strip()
        seq_hash = _hashlib.sha256(seq_clean.encode()).hexdigest()[:16]
        cached_file = cache_path / f'{seq_hash}.npy'

        # Check cache
        if cached_file.exists():
            emb = np.load(cached_file)
            all_embeddings.append(emb)
            continue

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Write FASTA file
                fasta_path = os.path.join(tmpdir, 'input.fasta')
                with open(fasta_path, 'w') as f:
                    f.write(f'>seq_{i}\n{seq_clean}\n')

                out_dir = os.path.join(tmpdir, 'output')

                # Run boltz predict with embeddings
                cmd = [
                    'boltz', 'predict', fasta_path,
                    '--write_embeddings',
                    '--out_dir', out_dir,
                ]
                if device != 'cpu':
                    cmd.extend(['--accelerator', 'gpu'])

                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300,
                )
                if proc.returncode != 0:
                    raise RuntimeError(f"boltz predict failed: {proc.stderr[:200]}")

                # Find and load embeddings npz
                out_path = Path(out_dir)
                npz_files = list(out_path.rglob('*embeddings*.npz'))
                if not npz_files:
                    # Try loading from the predictions directory
                    npz_files = list(out_path.rglob('*.npz'))

                if npz_files:
                    data = np.load(npz_files[0])
                    # Boltz-2 single representations: (N_tokens, 384)
                    # Try common key names
                    for key in ['single', 's_s', 'prot_embeddings', 'embeddings']:
                        if key in data:
                            token_embs = data[key]
                            break
                    else:
                        # Use first array in npz
                        token_embs = data[list(data.keys())[0]]

                    # Mean pool over token dimension
                    if token_embs.ndim == 3:
                        token_embs = token_embs[0]  # remove batch dim
                    emb = token_embs.mean(axis=0).astype(np.float32)

                    # Handle variable embedding dims — pad/truncate to 384
                    if emb.shape[0] != 384:
                        if emb.shape[0] > 384:
                            emb = emb[:384]
                        else:
                            emb = np.pad(emb, (0, 384 - emb.shape[0]))
                else:
                    raise FileNotFoundError("No embedding files found in boltz output")

            # Cache the result
            np.save(cached_file, emb)
            all_embeddings.append(emb)

        except Exception as e:
            print(f"    WARNING: Boltz-2 failed on seq {i} (len={len(seq_clean)}): {e}")
            all_embeddings.append(np.zeros(384, dtype=np.float32))

        if (i + 1) % 10 == 0:
            print(f"    Boltz-2 progress: {i + 1}/{n}")

    result = np.vstack(all_embeddings)
    print(f"  Generated Boltz-2 protein embeddings: {result.shape}")
    return result


def create_amplify_protein_embeddings(
    sequences: List[str],
    model_name: str = 'chandar-lab/AMPLIFY_350M',
    device: str = 'cpu',
    batch_size: int = 4,
) -> np.ndarray:
    """Extract AMPLIFY 350M mean-pooled embeddings for protein sequences.

    Drop-in ESM-2 alternative. 350M parameter protein LM from Mila/Chandar Lab.

    Args:
        sequences: List of protein amino acid sequences.
        model_name: HuggingFace model name.
        device: 'cpu' or 'cuda'.
        batch_size: Sequences per batch.

    Returns:
        np.ndarray: (n_sequences, 960) embeddings.

    Requires: pip install transformers
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    print(f"  Loading AMPLIFY ({model_name.split('/')[-1]}) for protein embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()

    all_embeddings = []
    n = len(sequences)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = [s.upper().strip()[:1022] for s in sequences[start:start + batch_size]]
            inputs = tokenizer(
                batch, return_tensors='pt', padding=True,
                truncation=True, max_length=1024
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            # Mean pool excluding BOS/EOS tokens
            mask = inputs['attention_mask'].clone()
            mask[:, 0] = 0  # exclude BOS
            for j in range(mask.shape[0]):
                last_tok = mask[j].sum().item()
                if last_tok > 0:
                    mask[j, int(last_tok)] = 0  # exclude EOS
            mask = mask.unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_embeddings.append(pooled.cpu().numpy())

            end = min(start + batch_size, n)
            if end < n and end % 50 == 0:
                print(f"    Progress: {end}/{n}")

    result = np.concatenate(all_embeddings, axis=0)
    print(f"  Generated AMPLIFY protein embeddings: {result.shape}")
    return result


def create_protein_aac(sequences: List[str]) -> np.ndarray:
    """Compute amino acid composition for protein sequences.

    Args:
        sequences: List of protein amino acid sequences.

    Returns:
        np.ndarray: (n_sequences, 20) frequency matrix.
    """
    n = len(sequences)
    features = np.zeros((n, 20), dtype=np.float64)
    aa_to_idx = {aa: i for i, aa in enumerate(STANDARD_AAS)}

    for i, seq in enumerate(sequences):
        seq = seq.upper().strip()
        if not seq:
            continue
        for aa in seq:
            if aa in aa_to_idx:
                features[i, aa_to_idx[aa]] += 1
        total = sum(1 for aa in seq if aa in aa_to_idx)
        if total > 0:
            features[i] /= total

    print(f"  Generated protein AAC features: ({n}, 20)")
    return features


def create_protein_descriptors(sequences: List[str]) -> np.ndarray:
    """Compute biophysical descriptors for protein sequences.

    Features (30D), interleaved per property (mean/std/min/max):
        0-3:   hydrophobicity (mean, std, min, max)
        4-7:   charge (mean, std, min, max)
        8-11:  molecular weight (mean, std, min, max)
        12-15: volume (mean, std, min, max)
        16-19: polarity (mean, std, min, max)
        20-23: flexibility (mean, std, min, max)
        24: sequence length (normalized by 1000)
        25: molecular weight estimate (sum of residue MWs / 1e5)
        26: net charge at pH 7.4
        27: fraction hydrophobic (I, V, L, F, C, M, A, W)
        28: fraction charged (D, E, K, R, H)
        29: net charge ratio ((n_positive - n_negative) / seq_len)

    Args:
        sequences: List of protein amino acid sequences.

    Returns:
        np.ndarray: (n_sequences, 30) descriptor matrix.
    """
    props = ['hydrophobicity', 'charge', 'mw', 'volume', 'polarity', 'flexibility']
    hydrophobic = set('IVLFCMAW')
    charged = set('DEKRH')

    n = len(sequences)
    features = np.zeros((n, 30), dtype=np.float64)

    for i, seq in enumerate(sequences):
        seq = seq.upper().strip()
        if not seq:
            continue

        # Per-residue property vectors
        prop_vals = {p: [] for p in props}
        for aa in seq:
            if aa in AA_PROPERTIES:
                for p in props:
                    prop_vals[p].append(AA_PROPERTIES[aa][p])

        if not prop_vals['hydrophobicity']:
            continue

        # Statistics for each property (6 × 4 = 24D)
        for j, p in enumerate(props):
            vals = np.array(prop_vals[p])
            features[i, j * 4]     = vals.mean()
            features[i, j * 4 + 1] = vals.std()
            features[i, j * 4 + 2] = vals.min()
            features[i, j * 4 + 3] = vals.max()

        # Global descriptors
        valid_aas = [aa for aa in seq if aa in AA_PROPERTIES]
        seq_len = len(valid_aas)
        features[i, 24] = seq_len / 1000.0
        features[i, 25] = sum(AA_PROPERTIES[aa]['mw'] for aa in valid_aas) / 1e5
        features[i, 26] = sum(AA_PROPERTIES[aa]['charge'] for aa in valid_aas)
        features[i, 27] = sum(1 for aa in valid_aas if aa in hydrophobic) / max(seq_len, 1)
        features[i, 28] = sum(1 for aa in valid_aas if aa in charged) / max(seq_len, 1)

        # Simple pI estimate (count of charged residues)
        n_pos = sum(1 for aa in valid_aas if AA_PROPERTIES[aa]['charge'] > 0)
        n_neg = sum(1 for aa in valid_aas if AA_PROPERTIES[aa]['charge'] < 0)
        features[i, 29] = (n_pos - n_neg) / max(seq_len, 1)

    print(f"  Generated protein descriptors: ({n}, 30)")
    return features


def create_esmfold_protein_features(
    sequences: List[str],
    device: str = 'cpu',
) -> np.ndarray:
    """Extract structural features from ESMFold predicted structures.

    ESMFold predicts 3D structure from a single sequence (no MSA needed).
    Much faster than AlphaFold2. From the predicted structure we extract:
      - pLDDT per-residue confidence → stats (mean, min, max, std, fraction>70)
      - Contact map summary → contact density, avg contacts per residue
      - Radius of gyration estimate
      - Secondary structure fractions (helix, sheet, coil from dssp-like)

    Features (12D):
      0: mean pLDDT
      1: min pLDDT
      2: max pLDDT
      3: std pLDDT
      4: fraction pLDDT > 70 (confident residues)
      5: fraction pLDDT > 90 (very confident)
      6: mean contacts per residue (8Å cutoff)
      7: contact density (contacts / n_residues²)
      8: radius of gyration (from Cα positions)
      9: sequence length / 1000 (normalised)
      10: fraction residues in contact with >5 others
      11: mean pairwise Cα distance

    Requires: pip install transformers torch
    """
    import torch

    print("  Loading ESMFold for structure prediction...")
    try:
        from transformers import EsmForProteinFolding, AutoTokenizer
    except ImportError:
        raise ImportError("ESMFold requires transformers>=4.30: pip install transformers")

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    # Force fp32 to avoid "LayerNormKernelImpl not implemented for Half"
    # which can happen when bitsandbytes or autocast interfere
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", torch_dtype=torch.float32,
    ).to(device).float()
    model.eval()

    n = len(sequences)
    features = np.zeros((n, 12), dtype=np.float32)

    print(f"  Predicting structures for {n} proteins with ESMFold...")
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=False):
        for i, seq in enumerate(sequences):
            seq = seq.upper().strip()[:512]  # ESMFold max practical length
            try:
                inputs = tokenizer([seq], return_tensors="pt",
                                   add_special_tokens=False).to(device)
                output = model(**inputs)

                # pLDDT: per-residue confidence (0-100 scale in output.plddt)
                plddt = output.plddt[0, :len(seq)].cpu().numpy().flatten()
                if plddt.max() <= 1.0:
                    plddt *= 100.0  # normalize to 0-100 if needed

                features[i, 0] = plddt.mean()
                features[i, 1] = plddt.min()
                features[i, 2] = plddt.max()
                features[i, 3] = plddt.std()
                features[i, 4] = (plddt > 70).mean()
                features[i, 5] = (plddt > 90).mean()

                # Extract Cα positions from predicted structure
                # output.positions: (1, n_res, 37, 3) — atom37 representation
                # Cα is atom index 1 in atom37
                positions = output.positions[0, :len(seq), 1, :].cpu().numpy()  # (L, 3)

                # Contact map (8Å cutoff on Cα-Cα distances)
                from scipy.spatial.distance import pdist, squareform
                dist_matrix = squareform(pdist(positions))
                n_res = len(seq)
                contacts = (dist_matrix < 8.0) & (np.abs(np.arange(n_res)[:, None] - np.arange(n_res)[None, :]) > 5)
                contacts_per_res = contacts.sum(axis=1).astype(np.float32)

                features[i, 6] = contacts_per_res.mean()
                features[i, 7] = contacts.sum() / max(n_res * n_res, 1)
                features[i, 8] = np.sqrt(((positions - positions.mean(axis=0)) ** 2).sum(axis=1).mean())
                features[i, 9] = n_res / 1000.0
                features[i, 10] = (contacts_per_res > 5).mean()
                features[i, 11] = dist_matrix.mean()

            except Exception as e:
                print(f"    WARNING: ESMFold failed on seq {i} (len={len(seq)}): {e}")
                features[i, 9] = len(seq) / 1000.0  # at least set length

            if (i + 1) % 20 == 0 or i == n - 1:
                print(f"    Progress: {i + 1}/{n}")

    print(f"  Generated ESMFold features: ({n}, 12)")
    return features


def _resolve_uniprot_ids(
    protein_ids: List[str],
) -> List[Optional[str]]:
    """Resolve protein identifiers to UniProt accession IDs.

    Handles two cases:
      - Already UniProt accessions (e.g. 'O00141', 'P04629'): returned as-is
      - Gene names (e.g. 'AAK1', 'ABL1(E255K)'): looked up via UniProt REST API

    Results are cached to disk so the API is only queried once per gene name.
    """
    import re
    import json
    from pathlib import Path

    cache_dir = Path(os.environ.get('KIRBY_CACHE_DIR',
                     str(Path.home() / '.cache' / 'kirby')))
    cache_file = cache_dir / 'uniprot_gene_map.json'

    # Load existing cache
    gene_map = {}
    if cache_file.exists():
        try:
            gene_map = json.loads(cache_file.read_text())
        except Exception:
            pass

    # UniProt accession pattern: [OPQ][0-9][A-Z0-9]{3}[0-9] or [A-NR-Z][0-9]...
    _uniprot_re = re.compile(r'^[A-Z][0-9][A-Z0-9]{3}[0-9]$')

    result = []
    genes_to_lookup = set()

    for pid in protein_ids:
        if pid is None:
            result.append(None)
            continue
        pid_clean = pid.strip()
        if _uniprot_re.match(pid_clean):
            result.append(pid_clean)
        else:
            # Extract base gene name: 'ABL1(E255K)-phosphorylated' → 'ABL1'
            base_gene = re.split(r'[(\-/]', pid_clean)[0].strip()
            if base_gene in gene_map:
                result.append(gene_map[base_gene])
            else:
                result.append(base_gene)  # placeholder, will be resolved
                genes_to_lookup.add(base_gene)

    # Batch lookup missing gene names via UniProt REST API
    if genes_to_lookup:
        import requests
        print(f"    Resolving {len(genes_to_lookup)} gene names to UniProt IDs...")
        for gene in sorted(genes_to_lookup):
            if gene in gene_map:
                continue
            try:
                url = (
                    f"https://rest.uniprot.org/uniprotkb/search?"
                    f"query=gene_exact:{gene}+AND+organism_id:9606&"
                    f"format=json&size=1&fields=accession"
                )
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('results'):
                        uid = data['results'][0]['primaryAccession']
                        gene_map[gene] = uid
                    else:
                        gene_map[gene] = None
                else:
                    gene_map[gene] = None
            except Exception:
                gene_map[gene] = None

        # Save updated cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(gene_map, indent=2))

        # Replace placeholders with resolved IDs
        _uniprot_re2 = re.compile(r'^[A-Z][0-9][A-Z0-9]{3}[0-9]$')
        for i, pid in enumerate(protein_ids):
            if pid is None:
                continue
            pid_clean = pid.strip()
            if not _uniprot_re2.match(pid_clean):
                base_gene = re.split(r'[(\-/]', pid_clean)[0].strip()
                result[i] = gene_map.get(base_gene)

        resolved = sum(1 for r in result if r is not None)
        print(f"    Resolved {resolved}/{len(result)} protein IDs to UniProt accessions")

    return result


def create_alphafold_db_features(
    sequences: List[str],
    uniprot_ids: Optional[List[str]] = None,
) -> np.ndarray:
    """Extract structural features from AlphaFold Database pre-computed structures.

    Downloads predicted structures from the AlphaFold Protein Structure Database
    (https://alphafold.ebi.ac.uk/) for proteins with known UniProt IDs. Extracts
    contact map + distance matrix summary features.

    If UniProt IDs are not provided, attempts to map sequences via UniProt ID
    mapping API. Falls back to sequence-based descriptors for unmapped proteins.

    Features (15D):
      0: mean pLDDT (from B-factor column)
      1: min pLDDT
      2: fraction pLDDT > 70
      3: fraction pLDDT > 90
      4: mean contacts per residue (8Å Cα)
      5: contact density
      6: radius of gyration
      7: mean pairwise Cα distance
      8: max pairwise Cα distance
      9: sequence length / 1000
      10: long-range contact fraction (|i-j| > 12)
      11: fraction residues with >5 contacts
      12: mean Cα distance for contacts (short range, |i-j| < 6)
      13: mean Cα distance for contacts (long range, |i-j| > 12)
      14: pLDDT std

    Requires: biopython, requests
    """
    import requests
    import tempfile
    from pathlib import Path

    print(f"  Fetching AlphaFold DB structures for {len(sequences)} proteins...")

    n = len(sequences)
    features = np.zeros((n, 15), dtype=np.float32)

    # Cache directory
    cache_dir = Path(os.environ.get('KIRBY_CACHE_DIR',
                     str(Path.home() / '.cache' / 'kirby'))) / 'alphafold_db'
    cache_dir.mkdir(parents=True, exist_ok=True)

    for i, seq in enumerate(sequences):
        seq = seq.upper().strip()
        uid = uniprot_ids[i] if uniprot_ids and i < len(uniprot_ids) else None

        # Try to get UniProt ID if not provided
        if uid is None:
            # Use sequence hash as a lookup key — skip API mapping for speed
            features[i, 9] = len(seq) / 1000.0
            continue

        pdb_path = cache_dir / f'AF-{uid}-F1-model_v4.pdb'

        # Download if not cached
        if not pdb_path.exists():
            url = f'https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v4.pdb'
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    pdb_path.write_bytes(resp.content)
                else:
                    features[i, 9] = len(seq) / 1000.0
                    continue
            except Exception:
                features[i, 9] = len(seq) / 1000.0
                continue

        # Parse PDB and extract features
        try:
            from Bio.PDB import PDBParser
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('af', str(pdb_path))
            model = structure[0]

            # Extract Cα atoms
            ca_atoms = []
            plddt_values = []
            for chain in model.get_chains():
                for residue in chain.get_residues():
                    if residue.get_id()[0] != ' ':
                        continue
                    if 'CA' in residue:
                        ca = residue['CA']
                        ca_atoms.append(ca.get_vector().get_array())
                        plddt_values.append(ca.get_bfactor())

            if not ca_atoms:
                features[i, 9] = len(seq) / 1000.0
                continue

            ca_positions = np.array(ca_atoms)
            plddt = np.array(plddt_values)
            n_res = len(ca_positions)

            # pLDDT features
            features[i, 0] = plddt.mean()
            features[i, 1] = plddt.min()
            features[i, 2] = (plddt > 70).mean()
            features[i, 3] = (plddt > 90).mean()

            # Contact map features
            from scipy.spatial.distance import pdist, squareform
            dist_matrix = squareform(pdist(ca_positions))
            seq_sep = np.abs(np.arange(n_res)[:, None] - np.arange(n_res)[None, :])
            contacts = (dist_matrix < 8.0) & (seq_sep > 5)
            contacts_per_res = contacts.sum(axis=1).astype(np.float32)

            features[i, 4] = contacts_per_res.mean()
            features[i, 5] = contacts.sum() / max(n_res * n_res, 1)
            features[i, 6] = np.sqrt(((ca_positions - ca_positions.mean(axis=0)) ** 2).sum(axis=1).mean())
            features[i, 7] = dist_matrix.mean()
            features[i, 8] = dist_matrix.max()
            features[i, 9] = n_res / 1000.0

            # Long-range contacts
            long_range = contacts & (seq_sep > 12)
            features[i, 10] = long_range.sum() / max(contacts.sum(), 1)
            features[i, 11] = (contacts_per_res > 5).mean()

            # Short vs long range average distances
            short_contacts = (dist_matrix < 8.0) & (seq_sep > 0) & (seq_sep <= 6)
            long_contacts = (dist_matrix < 8.0) & (seq_sep > 12)
            features[i, 12] = dist_matrix[short_contacts].mean() if short_contacts.any() else 0.0
            features[i, 13] = dist_matrix[long_contacts].mean() if long_contacts.any() else 0.0
            features[i, 14] = plddt.std()

        except Exception as e:
            print(f"    WARNING: AlphaFold DB feature extraction failed for {uid}: {e}")
            features[i, 9] = len(seq) / 1000.0

        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"    Progress: {i + 1}/{n}")

    mapped = (features[:, 0] != 0).sum()
    print(f"  Generated AlphaFold DB features: ({n}, 15), {mapped}/{n} mapped")
    return features


def create_mint_protein_embeddings(
    sequences: List[str],
    checkpoint_path: str = None,
    device: str = 'cpu',
) -> np.ndarray:
    """Extract MINT mean-pooled embeddings for protein sequences.

    MINT (Masked INTeraction modeling) is a protein language model built
    on top of ESM-2, pretrained for protein-protein interaction prediction.
    Produces 1280D embeddings per sequence.

    Args:
        sequences: List of protein amino acid sequences.
        checkpoint_path: Path to mint.ckpt. If None, searches
            $KIRBY_MODELS_DIR/mint/mint.ckpt, ~/kirby_models/mint/mint.ckpt.
        device: 'cpu' or 'cuda'.

    Returns:
        np.ndarray: (n_sequences, 1280) embeddings.

    Requires: git clone https://github.com/VarunUllanat/mint.git && pip install -e mint
              Download checkpoint from MINT GitHub releases.
    """
    import torch

    try:
        from mint.helpers.extract import load_config, MINTWrapper
    except ImportError:
        raise ImportError(
            "MINT requires the mint package: "
            "git clone https://github.com/VarunUllanat/mint.git && pip install -e mint\n"
            "Download checkpoint from https://github.com/VarunUllanat/mint "
            "and place at $KIRBY_MODELS_DIR/mint/mint.ckpt"
        )

    if checkpoint_path is None:
        search_paths = [
            os.path.join(os.environ.get('KIRBY_MODELS_DIR', ''), 'mint', 'mint.ckpt'),
            os.path.expanduser('~/kirby_models/mint/mint.ckpt'),
            'mint.ckpt',
            '../mint.ckpt',
        ]
        found = None
        for p in search_paths:
            if p and os.path.isfile(p):
                found = p
                break
        if found is None:
            raise FileNotFoundError(
                f"Could not find MINT checkpoint in: {search_paths}\n"
                "Download from https://github.com/VarunUllanat/mint "
                "and set KIRBY_MODELS_DIR or place at ~/kirby_models/mint/mint.ckpt"
            )
        checkpoint_path = found

    print(f"  Loading MINT from {checkpoint_path}...")
    # load_config requires path to MINT's config JSON
    mint_pkg_dir = os.path.dirname(os.path.dirname(
        __import__('mint').__file__))
    config_path = os.path.join(mint_pkg_dir, 'data', 'esm2_t33_650M_UR50D.json')
    if not os.path.isfile(config_path):
        # Fallback: search relative to KIRBy repo
        for candidate in [
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'MINT', 'data', 'esm2_t33_650M_UR50D.json'),
            os.path.join(os.environ.get('KIRBY_MODELS_DIR', ''), 'mint', 'esm2_t33_650M_UR50D.json'),
        ]:
            if candidate and os.path.isfile(candidate):
                config_path = candidate
                break
    config = load_config(config_path)
    model = MINTWrapper(config, checkpoint_path)
    model = model.to(device)
    model.eval()

    all_embeddings = []
    n = len(sequences)

    with torch.no_grad():
        for i, seq in enumerate(sequences):
            seq = seq.upper().strip()[:1022]  # ESM-2 backbone limit
            try:
                emb = model.get_embedding(seq)
                if isinstance(emb, torch.Tensor):
                    emb = emb.cpu().numpy()
                emb = np.atleast_1d(emb.flatten())
                # Expect 1280D from ESM-2 backbone
                if emb.shape[0] != 1280:
                    emb = emb[:1280] if emb.shape[0] > 1280 else np.pad(
                        emb, (0, 1280 - emb.shape[0]))
                all_embeddings.append(emb)
            except Exception as e:
                print(f"    WARNING: MINT failed on seq {i} (len={len(seq)}): {e}")
                all_embeddings.append(np.zeros(1280, dtype=np.float32))

            if (i + 1) % 50 == 0:
                print(f"    Progress: {i + 1}/{n}")

    result = np.vstack(all_embeddings).astype(np.float32)
    print(f"  Generated MINT protein embeddings: {result.shape}")
    return result


def create_xtrimopglm_protein_embeddings(
    sequences: List[str],
    device: str = 'cpu',
    batch_size: int = 2,
) -> np.ndarray:
    """Extract xTrimoPGLM-1B mean-pooled embeddings for protein sequences.

    xTrimoPGLM is a unified 100B-parameter protein language model from BioMap
    that integrates protein understanding and generation. We use the 1B MLM
    variant (proteinglm-1b-mlm) for efficient embedding extraction.

    Args:
        sequences: List of protein amino acid sequences.
        device: 'cpu' or 'cuda'.
        batch_size: Sequences per batch (default 2 — model is large).

    Returns:
        np.ndarray: (n_sequences, hidden_dim) embeddings.

    Requires: pip install transformers torch
    Note: Auto-downloads HuggingFace model on first use. Requires
          trust_remote_code=True for custom model architecture.
    """
    import torch
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError(
            "xTrimoPGLM requires transformers and torch: "
            "pip install transformers torch"
        )

    model_name = 'biomap-research/proteinglm-1b-mlm'
    print(f"  Loading xTrimoPGLM-1B ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    # Try bfloat16 first, fall back to float16, then float32
    dtype = torch.float32
    if device == 'cuda':
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=dtype,
    ).to(device).eval()

    # Determine hidden dim from model config
    hidden_dim = getattr(model.config, 'hidden_size', 2048)

    all_embeddings = []
    n = len(sequences)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_seqs = sequences[start:start + batch_size]
            batch_embs = []
            for j, seq in enumerate(batch_seqs):
                idx = start + j
                seq = seq.upper().strip()[:2048]
                try:
                    inputs = tokenizer(
                        seq, return_tensors='pt', padding=False,
                        truncation=True, max_length=2050,
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    # Mean pool excluding special tokens (BOS/EOS)
                    hidden = outputs.last_hidden_state
                    if hidden.shape[1] > 2:
                        emb = hidden[:, 1:-1, :].mean(dim=1)
                    else:
                        emb = hidden.mean(dim=1)
                    batch_embs.append(
                        emb.squeeze(0).float().cpu().numpy()
                    )
                except Exception as e:
                    print(f"    WARNING: xTrimoPGLM failed on seq {idx} "
                          f"(len={len(seq)}): {e}")
                    batch_embs.append(
                        np.zeros(hidden_dim, dtype=np.float32)
                    )
            all_embeddings.extend(batch_embs)

            end = min(start + batch_size, n)
            if end < n and end % 20 == 0:
                print(f"    xTrimoPGLM progress: {end}/{n}")

    result = np.vstack(all_embeddings).astype(np.float32)
    print(f"  Generated xTrimoPGLM protein embeddings: {result.shape}")
    return result


def create_protein_kmer(sequences: List[str], k: int = 3) -> np.ndarray:
    """Compute k-mer frequency features for protein sequences.

    Args:
        sequences: List of protein amino acid sequences.
        k: k-mer size. Default 3 → 8000D.

    Returns:
        np.ndarray: (n_sequences, 20^k) k-mer frequency matrix.
    """
    n_aas = len(STANDARD_AAS)
    dim = n_aas ** k
    aa_to_idx = {aa: i for i, aa in enumerate(STANDARD_AAS)}

    n = len(sequences)
    features = np.zeros((n, dim), dtype=np.float64)

    print(f"  Computing {k}-mer frequency for {n} proteins (dim={dim})...")

    for i, seq in enumerate(sequences):
        seq = seq.upper().strip()
        if len(seq) < k:
            continue

        count = 0
        for j in range(len(seq) - k + 1):
            kmer = seq[j:j + k]
            idx = 0
            valid = True
            for c, char in enumerate(kmer):
                if char not in aa_to_idx:
                    valid = False
                    break
                idx += aa_to_idx[char] * (n_aas ** (k - 1 - c))
            if valid:
                features[i, idx] += 1
                count += 1

        if count > 0:
            features[i] /= count

    print(f"  Generated protein {k}-mer features: ({n}, {dim})")
    return features
