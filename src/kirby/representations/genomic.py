"""
Genomic and Histopathology Representations for TCGA Data

Gene expression representations:
- HVG2000: Top 2000 highly variable genes, z-scored
- PCA100: PCA of full expression to 100 components
- Hallmark: ssGSEA scores over 50 MSigDB Hallmark pathways
- PROGENy: 14 signaling pathway activity scores
- KEGG: ssGSEA scores over ~186 KEGG pathways
- DoRothEA: ~300 transcription factor activity scores
- Combined pathway: concatenation of PROGENy + Hallmark + DoRothEA
- Mutation matrix: binary top-200 mutated gene matrix

WSI representations (all stored as .npy dicts, auto-downloaded where possible):
- GigaSSL slide-level (Mendeley): CTransPath, Phikon, H-optimus-0, GigaPath, MoCo
- TITAN slide-level (HuggingFace, gated): 768D, ~11K TCGA slides
- UNI2-h patch-level (HuggingFace, gated): 1536D, mean-pooled to slide-level
- Local CNN 320D (Seagate): pre-computed from small_cnn_320 .pt files
- Graceful fallback (zeros) for missing slides

All functions follow KIRBy convention: take input, return np.ndarray(n_samples, d).
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# GENOMIC REPRESENTATIONS (expression matrix -> feature vectors)
# =============================================================================

def create_hvg2000(expression: np.ndarray, gene_names: np.ndarray,
                   n_top: int = 2000) -> np.ndarray:
    """Select top highly variable genes, z-score normalized.

    Uses variance-based selection (no scanpy dependency required).
    Falls back to highest-variance genes if scanpy unavailable.

    Args:
        expression: (n_samples, n_genes) log2(TPM+1) matrix
        gene_names: (n_genes,) gene symbols
        n_top: number of top variable genes to select

    Returns:
        (n_samples, n_top) z-scored HVG matrix
    """
    from sklearn.preprocessing import StandardScaler

    n_top = min(n_top, expression.shape[1])

    try:
        import scanpy as sc
        import anndata as ad

        adata = ad.AnnData(X=expression.copy(), var=pd.DataFrame(index=gene_names))
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor='seurat_v3',
                                     span=0.3, check_values=False)
        hvg_mask = adata.var['highly_variable'].values
        X_hvg = expression[:, hvg_mask]
    except ImportError:
        # Fallback: select by variance
        variances = np.var(expression, axis=0)
        top_idx = np.argsort(variances)[-n_top:]
        X_hvg = expression[:, top_idx]

    # Z-score normalize
    scaler = StandardScaler()
    X_hvg = scaler.fit_transform(X_hvg)

    return X_hvg.astype(np.float32)


def create_gene_pca100(expression: np.ndarray,
                       n_components: int = 100) -> np.ndarray:
    """PCA of full expression matrix.

    Args:
        expression: (n_samples, n_genes) expression matrix
        n_components: number of PCA components

    Returns:
        (n_samples, n_components) PCA-transformed features
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n_components = min(n_components, expression.shape[1], expression.shape[0])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(expression)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_.sum()
    print(f"    PCA: {n_components} components explain {explained:.1%} variance")

    return X_pca.astype(np.float32)


def create_hallmark_scores(expression: np.ndarray,
                           gene_names: np.ndarray) -> np.ndarray:
    """ssGSEA scores for MSigDB Hallmark pathways (50 pathways).

    Requires gseapy >= 1.0.

    Returns:
        (n_samples, 50) pathway enrichment scores
    """
    try:
        import gseapy as gp
    except ImportError:
        raise ImportError("gseapy required for Hallmark scores: pip install gseapy")

    # Create DataFrame: genes as rows, samples as columns (gseapy convention)
    df = pd.DataFrame(expression.T, index=gene_names,
                       columns=[f'S{i}' for i in range(expression.shape[0])])

    # Run ssGSEA with Hallmark gene sets
    result = gp.ssgsea(
        data=df,
        gene_sets='h.all.v2023.2.Hs.symbols.gmt',
        outdir=None,
        no_plot=True,
        min_size=5,
        threads=1,
    )

    # Extract scores matrix
    scores = result.res2d.pivot(index='Name', columns='Term', values='NES')
    if scores.empty:
        # Try alternative result format
        scores = result.resultsOnSamples

    X = scores.values.T if scores.shape[0] < scores.shape[1] else scores.values
    X = np.nan_to_num(X, nan=0.0).astype(np.float32)
    return X


def create_progeny_scores(expression: np.ndarray,
                          gene_names: np.ndarray) -> np.ndarray:
    """PROGENy pathway activity scores (14 signaling pathways).

    Requires decoupler >= 1.4.

    Returns:
        (n_samples, 14) pathway activity scores
    """
    try:
        import decoupler as dc
    except ImportError:
        raise ImportError("decoupler required for PROGENy scores: pip install decoupler")

    try:
        import anndata as ad
    except ImportError:
        raise ImportError("anndata required for PROGENy scores: pip install anndata")

    adata = ad.AnnData(X=expression.copy(), var=pd.DataFrame(index=gene_names))
    adata.obs_names = [f'S{i}' for i in range(expression.shape[0])]

    # Get PROGENy pathway model
    progeny = dc.get_progeny(organism='human', top=300)

    # Run multivariate linear model
    dc.run_mlm(mat=adata, net=progeny, source='source', target='target',
               weight='weight', use_raw=False)

    # Extract activity scores from obsm
    if 'mlm_estimate' in adata.obsm:
        X = adata.obsm['mlm_estimate'].values
    else:
        # Try alternative key
        key = next((k for k in adata.obsm if 'estimate' in k), None)
        if key:
            X = adata.obsm[key].values
        else:
            raise RuntimeError("PROGENy scores not found in adata.obsm")

    return np.nan_to_num(X, nan=0.0).astype(np.float32)


def create_kegg_scores(expression: np.ndarray,
                       gene_names: np.ndarray) -> np.ndarray:
    """ssGSEA scores for KEGG pathways (~186 pathways).

    Requires gseapy >= 1.0.

    Returns:
        (n_samples, ~186) pathway enrichment scores
    """
    try:
        import gseapy as gp
    except ImportError:
        raise ImportError("gseapy required for KEGG scores: pip install gseapy")

    df = pd.DataFrame(expression.T, index=gene_names,
                       columns=[f'S{i}' for i in range(expression.shape[0])])

    result = gp.ssgsea(
        data=df,
        gene_sets='c2.cp.kegg_legacy.v2023.2.Hs.symbols.gmt',
        outdir=None,
        no_plot=True,
        min_size=5,
        threads=1,
    )

    scores = result.res2d.pivot(index='Name', columns='Term', values='NES')
    if scores.empty:
        scores = result.resultsOnSamples

    X = scores.values.T if scores.shape[0] < scores.shape[1] else scores.values
    X = np.nan_to_num(X, nan=0.0).astype(np.float32)
    return X


def create_dorothea_scores(expression: np.ndarray,
                           gene_names: np.ndarray) -> np.ndarray:
    """DoRothEA transcription factor activity scores (~300 TFs).

    Requires decoupler >= 1.4.

    Returns:
        (n_samples, ~300) TF activity scores
    """
    try:
        import decoupler as dc
    except ImportError:
        raise ImportError("decoupler required for DoRothEA scores: pip install decoupler")

    try:
        import anndata as ad
    except ImportError:
        raise ImportError("anndata required for DoRothEA scores: pip install anndata")

    adata = ad.AnnData(X=expression.copy(), var=pd.DataFrame(index=gene_names))
    adata.obs_names = [f'S{i}' for i in range(expression.shape[0])]

    # Get DoRothEA regulon network (confidence levels A, B, C)
    dorothea = dc.get_dorothea(organism='human')

    # Run multivariate linear model
    dc.run_mlm(mat=adata, net=dorothea, source='source', target='target',
               weight='weight', use_raw=False)

    if 'mlm_estimate' in adata.obsm:
        X = adata.obsm['mlm_estimate'].values
    else:
        key = next((k for k in adata.obsm if 'estimate' in k), None)
        if key:
            X = adata.obsm[key].values
        else:
            raise RuntimeError("DoRothEA scores not found in adata.obsm")

    return np.nan_to_num(X, nan=0.0).astype(np.float32)


def create_combined_pathway(expression: np.ndarray,
                            gene_names: np.ndarray) -> np.ndarray:
    """Combined pathway features: PROGENy + Hallmark + DoRothEA concatenated.

    Returns:
        (n_samples, ~364) concatenated pathway scores
    """
    parts = []

    try:
        progeny = create_progeny_scores(expression, gene_names)
        parts.append(progeny)
    except Exception as e:
        warnings.warn(f"PROGENy failed: {e}")

    try:
        hallmark = create_hallmark_scores(expression, gene_names)
        parts.append(hallmark)
    except Exception as e:
        warnings.warn(f"Hallmark failed: {e}")

    try:
        dorothea = create_dorothea_scores(expression, gene_names)
        parts.append(dorothea)
    except Exception as e:
        warnings.warn(f"DoRothEA failed: {e}")

    if not parts:
        raise RuntimeError("All pathway methods failed")

    return np.hstack(parts)


def create_mutation_matrix(sample_ids: List[str], cohort: str,
                           n_top: int = 200) -> np.ndarray:
    """Binary mutation matrix for top-N most frequently mutated genes.

    Args:
        sample_ids: TCGA sample barcodes
        cohort: TCGA cohort name (e.g., 'TCGA-BRCA')
        n_top: number of top mutated genes to include

    Returns:
        (n_samples, n_top) binary mutation matrix
    """
    from kirby.datasets.tcga import _download_mutations

    mutations = _download_mutations(cohort)

    # Count mutations per gene
    gene_counts = mutations['gene'].value_counts()
    top_genes = gene_counts.head(n_top).index.tolist()

    # Build binary matrix
    # Map sample_ids to rows
    sid_to_idx = {sid[:15]: i for i, sid in enumerate(sample_ids)}

    X = np.zeros((len(sample_ids), len(top_genes)), dtype=np.float32)
    gene_to_col = {g: j for j, g in enumerate(top_genes)}

    for _, row in mutations.iterrows():
        sid = str(row['sample'])[:15]
        gene = row['gene']
        if sid in sid_to_idx and gene in gene_to_col:
            X[sid_to_idx[sid], gene_to_col[gene]] = 1.0

    n_mutated = (X.sum(axis=1) > 0).sum()
    print(f"    Mutation matrix: {len(top_genes)} genes, "
          f"{n_mutated}/{len(sample_ids)} samples with >=1 mutation")

    return X


# =============================================================================
# WSI REPRESENTATIONS (pre-computed features)
# =============================================================================

def load_wsi_features(sample_ids: List[str], wsi_dir: Optional[Path],
                      model_name: str) -> np.ndarray:
    """Load pre-computed WSI features for given samples.

    All features stored as .npy dicts: {slide_id: embedding_vector}.
    Matches TCGA barcodes flexibly (patient/sample/slide level).

    Args:
        sample_ids: TCGA sample barcodes
        wsi_dir: path to directory containing model .npy files
        model_name: ctranspath, phikon, hoptimus0, gigapath, moco (GigaSSL),
                    titan (TITAN 768D), uni2h (UNI2-h 1536D), cnn320 (local 320D)

    Returns:
        (n_samples, d) feature matrix. Missing slides get zero vectors.
    """
    from kirby.datasets.tcga import _WSI_DIMS

    d = _WSI_DIMS.get(model_name, 768)

    if wsi_dir is None:
        warnings.warn(f"WSI directory not set — returning zeros for {model_name}")
        return np.zeros((len(sample_ids), d), dtype=np.float32)

    wsi_dir = Path(wsi_dir)
    model_file = wsi_dir / f'{model_name}.npy'

    if not model_file.exists():
        warnings.warn(f"WSI features not found: {model_file} — returning zeros")
        return np.zeros((len(sample_ids), d), dtype=np.float32)

    # Load GigaSSL dict: slide_id -> embedding
    try:
        data = np.load(model_file, allow_pickle=True).item()
    except Exception as e:
        warnings.warn(f"Failed to load {model_file}: {e} — returning zeros")
        return np.zeros((len(sample_ids), d), dtype=np.float32)

    if not isinstance(data, dict):
        warnings.warn(f"Unexpected format in {model_file} — returning zeros")
        return np.zeros((len(sample_ids), d), dtype=np.float32)

    # Build index from slide IDs to embeddings
    # GigaSSL uses various ID formats — try to match flexibly
    slide_lookup = {}
    for slide_id, embedding in data.items():
        slide_id_str = str(slide_id)
        # Store by patient ID (12 chars) and sample ID (15 chars)
        if slide_id_str.startswith('TCGA'):
            patient_id = slide_id_str[:12]
            sample_id = slide_id_str[:15] if len(slide_id_str) >= 15 else slide_id_str
            slide_lookup[patient_id] = embedding
            slide_lookup[sample_id] = embedding
            slide_lookup[slide_id_str] = embedding
        else:
            slide_lookup[slide_id_str] = embedding

    # Extract actual embedding dimension from first entry
    first_emb = next(iter(data.values()))
    if hasattr(first_emb, 'shape'):
        d = first_emb.shape[-1] if first_emb.ndim > 0 else d

    # Match sample_ids to embeddings
    X = np.zeros((len(sample_ids), d), dtype=np.float32)
    n_found = 0

    for i, sid in enumerate(sample_ids):
        # Try multiple ID truncations
        for trunc in [sid[:15], sid[:12], sid]:
            if trunc in slide_lookup:
                emb = slide_lookup[trunc]
                if hasattr(emb, 'shape') and emb.ndim == 1:
                    X[i] = emb[:d]
                elif hasattr(emb, 'shape') and emb.ndim == 2:
                    # Mean-pool if patch-level
                    X[i] = emb.mean(axis=0)[:d]
                n_found += 1
                break

    n_missing = len(sample_ids) - n_found
    if n_missing > 0:
        pct = n_missing / len(sample_ids) * 100
        warnings.warn(f"WSI {model_name}: {n_missing}/{len(sample_ids)} "
                      f"({pct:.1f}%) samples missing — filled with zeros")
    print(f"    WSI {model_name}: {n_found}/{len(sample_ids)} matched, dim={d}")

    return X
