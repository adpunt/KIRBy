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
# IMMUNE GENE SIGNATURES (hardcoded for reproducibility)
# =============================================================================

# ESTIMATE gene signatures (Yoshihara et al. 2013, Nature Communications)
# 141 stromal + 141 immune genes from SI_geneset.gmt (ESTIMATE R package)
ESTIMATE_STROMAL_GENES = [
    "DCN", "PAPPA", "SFRP4", "THBS2", "LY86", "CXCL14", "FOXF1",
    "COL10A1", "ACTG2", "APBB1IP", "SH2D1A", "SULF1", "MSR1",
    "C3AR1", "FAP", "PTGIS", "ITGBL1", "BGN", "CXCL12", "ECM2",
    "FCGR2A", "MS4A4A", "WISP1", "COL1A2", "MS4A6A", "EDNRA",
    "VCAM1", "GPR124", "SCUBE2", "AIF1", "HEPH", "LUM", "PTGER3",
    "RUNX1T1", "CDH5", "PIK3R5", "RAMP3", "LDB2", "COX7A1",
    "EDIL3", "DDR2", "FCGR2B", "LPPR4", "COL15A1", "AOC3",
    "ITIH3", "FMO1", "PRKG1", "PLXDC1", "VSIG4", "COL6A3",
    "SGCD", "COL3A1", "F13A1", "OLFML1", "IGSF6", "COMP", "HGF",
    "GIMAP5", "ABCA6", "ITGAM", "MAF", "ITM2A", "CLEC7A", "ASPN",
    "LRRC15", "ERG", "CD86", "TRAT1", "COL8A2", "TCF21", "CD93",
    "CD163", "GREM1", "LMOD1", "TLR2", "ZEB2", "C1QB", "KCNJ8",
    "KDR", "CD33", "RASGRP3", "TNFSF4", "CCR1", "CSF1R", "BTK",
    "MFAP5", "MXRA5", "ISLR", "ARHGAP28", "ZFPM2", "TLR7",
    "ADAM12", "OLFML2B", "ENPP2", "CILP", "SIGLEC1", "SPON2",
    "PLXNC1", "ADAMTS5", "SAMSN1", "CH25H", "COL14A1", "EMCN",
    "RGS4", "PCDH12", "RARRES2", "CD248", "PDGFRB", "C1QA",
    "COL5A3", "IGF1", "SP140", "TFEC", "TNN", "ATP8B4", "ZNF423",
    "FRZB", "SERPING1", "ENPEP", "CD14", "DIO2", "FPR1", "IL18R1",
    "HDC", "TXNDC3", "PDE2A", "RSAD2", "ITIH5", "FASLG", "MMP3",
    "NOX4", "WNT2", "LRRC32", "CXCL9", "ODZ4", "FBLN2", "EGFL6",
    "IL1B", "SPON1", "CD200",
]

ESTIMATE_IMMUNE_GENES = [
    "LCP2", "LSP1", "FYB", "PLEK", "HCK", "IL10RA", "LILRB1",
    "NCKAP1L", "LAIR1", "NCF2", "CYBB", "PTPRC", "IL7R", "LAPTM5",
    "CD53", "EVI2B", "SLA", "ITGB2", "GIMAP4", "MYO1F", "HCLS1",
    "MNDA", "IL2RG", "CD48", "AOAH", "CCL5", "LTB", "GMFG",
    "GIMAP6", "GZMK", "LST1", "GPR65", "LILRB2", "WIPF1", "CD37",
    "BIN2", "FCER1G", "IKZF1", "TYROBP", "FGL2", "FLI1", "IRF8",
    "ARHGAP15", "SH2B3", "TNFRSF1B", "DOCK2", "CD2", "ARHGEF6",
    "CORO1A", "LY96", "LYZ", "ITGAL", "TNFAIP3", "RNASE6",
    "TGFB1", "PSTPIP1", "CST7", "RGS1", "FGR", "SELL", "MICAL1",
    "TRAF3IP3", "ITGA4", "MAFB", "ARHGDIB", "IL4R", "RHOH",
    "HLA-DPA1", "NKG7", "NCF4", "LPXN", "ITK", "SELPLG",
    "HLA-DPB1", "CD3D", "CD300A", "IL2RB", "ADCY7", "PTGER4",
    "SRGN", "CD247", "CCR7", "MSN", "ALOX5AP", "PTGER2", "RAC2",
    "GBP2", "VAV1", "CLEC2B", "P2RY14", "NFKBIA", "S100A9",
    "IFI30", "MFSD1", "RASSF2", "TPP1", "RHOG", "CLEC4A", "GZMB",
    "PVRIG", "S100A8", "CASP1", "BCL2A1", "HLA-E", "KLRB1",
    "GNLY", "RAB27A", "IL18RAP", "TPST2", "EMP3", "GMIP", "LCK",
    "IL32", "PTPRCAP", "LGALS9", "CCDC69", "SAMHD1", "TAP1",
    "GBP1", "CTSS", "GZMH", "ADAM8", "GLRX", "PRF1", "CD69",
    "HLA-B", "HLA-DMA", "CD74", "KLRK1", "PTPRE", "HLA-DRA",
    "VNN2", "TCIRG1", "RABGAP1L", "CSTA", "ZAP70", "HLA-F",
    "HLA-G", "CD52", "CD302", "CD27",
]

# Bindea et al. 2013 (Immunity 39:782-795)
# 24 immune cell type marker gene sets for ssGSEA
BINDEA_IMMUNE_CELL_MARKERS = {
    "B_cells": [
        "MS4A1", "TCL1A", "HLA-DOB", "PNOC", "CD19", "CR2",
        "IGHG1", "FCRL2", "BLK", "COCH", "OSBPL10", "IGHA1", "TNFRSF17",
        "ABCB4", "BLNK", "GLDC", "MEF2C", "IGHM", "FAM30A", "SPIB",
        "BCL11A", "GNG7", "IGKC", "CD72", "MICAL3", "BACH2",
        "CCR9", "QRSL1", "DTNB", "HLA-DQA1", "SCN3A", "SLC15A2",
    ],
    "T_cells": [
        "PRKCQ", "CD3D", "CD3G", "CD28", "LCK", "TRAT1", "BCL11B",
        "CD2", "TRBC1", "ITM2A", "SH2D1A", "CD6", "CD96",
        "NCALD", "GIMAP5", "CD3E", "SKAP1",
    ],
    "T_helper_cells": [
        "ICOS", "LRBA", "FAM111A", "PHF10", "NUP107", "SEC24C", "NAP1L4",
        "BATF", "ASF1A", "FRYL", "RPA1", "UBE2L3", "ANP32B",
        "DDX50", "PPP2R5C", "SLC25A12", "ATF2", "CD28",
    ],
    "Th1": [
        "IFNG", "LTA", "APBB2", "DOK5", "IL12RB2", "APOD", "ZBTB32",
        "CD38", "CSF2", "CTLA4", "CD70", "DPP4", "BST2",
        "DUSP5", "LRP8", "DGKI", "CCL4", "GGT1", "LRRN3",
        "SYNGR3", "ATP9A", "BTG3", "HBEGF", "SGCB",
    ],
    "Th2": [
        "PMCH", "AHI1", "PTGIS", "CXCR6", "EVI5", "IL26",
        "GSTA4", "SMAD2", "CENPF", "ANK1", "ADCY1",
        "LAIR2", "SNRPD1", "MICAL2", "DHFR", "WDHD1", "BIRC5",
        "SLC39A14", "HELLS", "LIMA1", "CDC25C", "CDC7", "GATA3",
    ],
    "TFH": [
        "CXCL13", "PDCD1", "TOX", "MAF", "SIRPG",
        "B3GAT1", "STK39", "KCNK5",
    ],
    "Th17": ["IL17A", "IL17RA", "RORC"],
    "TReg": ["FOXP3"],
    "CD8_T_cells": [
        "CD8B", "CD8A", "PRF1", "GZMM", "FLT3LG",
        "LIME1", "ZEB1", "GADD45A", "ZFP36L2", "KLF9",
    ],
    "Tgd": ["TRGV9", "CD160", "FEZ1"],
    "Cytotoxic_cells": [
        "KLRD1", "KLRF1", "GNLY", "CTSW", "KLRB1", "KLRK1", "NKG7",
        "GZMH", "ZBTB16", "RUNX3", "APOL3", "RORA",
        "DUSP2", "GZMA",
    ],
    "NK_cells": [
        "XCL1", "XCL2", "NCR1", "BCL2", "SPN",
        "IGFBP5", "ALDH1B1", "PDLIM4", "TINAGL1",
    ],
    "NK_CD56dim_cells": [
        "KIR3DL2", "SPON2", "KIR2DL3", "GZMB", "KIR3DS1", "KIR3DL1",
        "IL21R", "KIR3DL3", "KIR2DS5", "KIR2DS2", "KIR2DS1",
    ],
    "NK_CD56bright_cells": [
        "DUSP4", "RRAD", "XCL1", "PLA2G6", "FOXJ1", "MADD", "MPPED1",
    ],
    "DC": ["CD209", "CCL17", "HSD11B1", "CCL13", "CCL22", "NPR1"],
    "iDC": [
        "CD1B", "VASH1", "F13A1", "CD1E", "MMP12", "FABP4", "CLEC10A",
        "MS4A6A", "CARD9", "ABCG2", "CD1A", "PPARG",
        "SLC7A8", "CSF1R", "CH25H", "BLVRB", "CD1C",
    ],
    "aDC": ["CCL1", "EBI3", "IDO1", "LAMP3", "OAS3"],
    "pDC": ["IL3RA"],
    "Eosinophils": [
        "IL5RA", "KCNH2", "CCR3", "THBS1", "CLC", "CYSLTR2", "HRH4",
        "RNASE2", "CAT", "GPR44", "HES1", "THBS4",
    ],
    "Macrophages": [
        "MARCO", "CXCL5", "MSR1", "CTSK", "PTGDS", "COLEC12",
        "CHIT1", "CLEC5A", "CCL7", "FN1", "CD163",
        "APOE", "CHI3L1", "CD84", "MS4A4A", "CYBB", "CD68",
    ],
    "Mast_cells": [
        "CTSG", "TPSAB1", "SLC18A2", "MS4A2", "CPA3", "TPSB2",
        "GATA2", "HDC", "SIGLEC6", "CMA1", "MLPH",
        "KIT", "TAL1", "HPGD", "PTGS1",
    ],
    "Neutrophils": [
        "CSF3R", "VNN3", "MME", "CXCR1", "CXCR2",
        "FCGR3B", "DYSF", "FCAR", "CEACAM3", "S100A12",
        "BST1", "CD93", "FPR1", "LILRB2",
    ],
}


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
        gene_sets='MSigDB_Hallmark_2020',
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

    # Normalize column names across decoupler versions
    # Old versions: genesymbol/label/value; New versions: source/target/weight
    col_map = {}
    if 'genesymbol' in progeny.columns and 'target' not in progeny.columns:
        col_map['genesymbol'] = 'target'
    if 'label' in progeny.columns and 'source' not in progeny.columns:
        col_map['label'] = 'source'
    if 'value' in progeny.columns and 'weight' not in progeny.columns:
        col_map['value'] = 'weight'
    if col_map:
        progeny = progeny.rename(columns=col_map)

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
        gene_sets='KEGG_2021_Human',
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

    # Normalize column names across decoupler versions
    col_map = {}
    if 'genesymbol' in dorothea.columns and 'target' not in dorothea.columns:
        col_map['genesymbol'] = 'target'
    if 'label' in dorothea.columns and 'source' not in dorothea.columns:
        col_map['label'] = 'source'
    if 'value' in dorothea.columns and 'weight' not in dorothea.columns:
        col_map['value'] = 'weight'
    if col_map:
        dorothea = dorothea.rename(columns=col_map)

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


def create_mutation_matrix(sample_ids: List[str], cohort: str = '',
                           n_top: int = 200) -> np.ndarray:
    """Binary mutation matrix for top-N most frequently mutated genes.

    Args:
        sample_ids: TCGA sample barcodes
        cohort: unused (kept for backward compat), filtering is by sample_ids
        n_top: number of top mutated genes to include

    Returns:
        (n_samples, n_top) binary mutation matrix
    """
    from kirby.datasets.tcga import _download_mutations

    mutations = _download_mutations(sample_ids)

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
# EXPRESSION-DERIVED REPRESENTATIONS (Part 1: no downloads needed)
# =============================================================================

def create_nmf50(expression: np.ndarray,
                 n_components: int = 50) -> np.ndarray:
    """Non-negative matrix factorization — parts-based decomposition.

    Complementary to PCA: learns additive parts rather than orthogonal axes.
    Expression values are clamped to >= 0 before NMF.

    Returns:
        (n_samples, n_components) NMF-transformed features
    """
    from sklearn.decomposition import NMF

    n_components = min(n_components, expression.shape[1], expression.shape[0])

    # NMF requires non-negative input
    X = np.clip(expression, 0, None)

    nmf = NMF(n_components=n_components, init='nndsvd', max_iter=300,
              random_state=42)
    X_nmf = nmf.fit_transform(X)

    reconstruction_err = nmf.reconstruction_err_
    print(f"    NMF: {n_components} components, reconstruction error={reconstruction_err:.1f}")

    return X_nmf.astype(np.float32)


def create_ica50(expression: np.ndarray,
                 n_components: int = 50) -> np.ndarray:
    """Independent component analysis — statistically independent sources.

    Finds maximally independent components, useful for separating
    biological signal sources (e.g., cell type programs).

    Returns:
        (n_samples, n_components) ICA-transformed features
    """
    from sklearn.decomposition import FastICA
    from sklearn.preprocessing import StandardScaler

    n_components = min(n_components, expression.shape[1], expression.shape[0])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(expression)

    ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
    X_ica = ica.fit_transform(X_scaled)

    print(f"    ICA: {n_components} independent components")

    return X_ica.astype(np.float32)


def create_gene_stats(expression: np.ndarray) -> np.ndarray:
    """Per-sample expression distribution statistics (15 features).

    Captures the global transcriptomic profile shape without
    gene-specific information. Features:
    mean, std, skewness, kurtosis, 5 quantiles (10/25/50/75/90),
    fraction zeros, entropy, range, MAD, fraction expressed, mean of expressed.

    Returns:
        (n_samples, 15) statistical feature matrix
    """
    from scipy.stats import skew, kurtosis

    n = expression.shape[0]
    stats = np.zeros((n, 15), dtype=np.float32)

    for i in range(n):
        row = expression[i]
        expressed = row[row > 0]

        stats[i, 0] = np.mean(row)
        stats[i, 1] = np.std(row)
        stats[i, 2] = skew(row) if np.std(row) > 0 else 0.0
        stats[i, 3] = kurtosis(row) if np.std(row) > 0 else 0.0
        stats[i, 4] = np.percentile(row, 10)
        stats[i, 5] = np.percentile(row, 25)
        stats[i, 6] = np.percentile(row, 50)
        stats[i, 7] = np.percentile(row, 75)
        stats[i, 8] = np.percentile(row, 90)
        stats[i, 9] = np.sum(row == 0) / len(row)  # fraction zeros
        # Entropy of expression distribution (binned, probability mass)
        hist, _ = np.histogram(row, bins=50)
        hist = hist.astype(np.float64)
        hist_sum = hist.sum()
        if hist_sum > 0:
            pmf = hist / hist_sum
            pmf = pmf[pmf > 0]
            stats[i, 10] = -np.sum(pmf * np.log(pmf))
        else:
            stats[i, 10] = 0.0
        stats[i, 11] = np.max(row) - np.min(row)  # range
        stats[i, 12] = np.median(np.abs(row - np.median(row)))  # MAD
        stats[i, 13] = len(expressed) / len(row)  # fraction expressed
        stats[i, 14] = np.mean(expressed) if len(expressed) > 0 else 0.0

    print(f"    Gene stats: 15 features per sample")
    return stats


def create_reactome_scores(expression: np.ndarray,
                           gene_names: np.ndarray) -> np.ndarray:
    """ssGSEA scores for Reactome pathways (~600 pathways).

    Requires gseapy >= 1.0.

    Returns:
        (n_samples, ~600) Reactome pathway enrichment scores
    """
    try:
        import gseapy as gp
    except ImportError:
        raise ImportError("gseapy required for Reactome scores: pip install gseapy")

    df = pd.DataFrame(expression.T, index=gene_names,
                       columns=[f'S{i}' for i in range(expression.shape[0])])

    result = gp.ssgsea(
        data=df,
        gene_sets='Reactome_2022',
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
    print(f"    Reactome: {X.shape[1]} pathways")
    return X


def create_oncogenic_scores(expression: np.ndarray,
                            gene_names: np.ndarray) -> np.ndarray:
    """ssGSEA scores for MSigDB C6 oncogenic signatures (~189 gene sets).

    Requires gseapy >= 1.0.

    Returns:
        (n_samples, ~189) oncogenic signature enrichment scores
    """
    try:
        import gseapy as gp
    except ImportError:
        raise ImportError("gseapy required for oncogenic scores: pip install gseapy")

    df = pd.DataFrame(expression.T, index=gene_names,
                       columns=[f'S{i}' for i in range(expression.shape[0])])

    result = gp.ssgsea(
        data=df,
        gene_sets='MSigDB_Oncogenic_Signatures',
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
    print(f"    Oncogenic: {X.shape[1]} signatures")
    return X


def create_immune_scores(expression: np.ndarray,
                         gene_names: np.ndarray) -> np.ndarray:
    """Immune cell deconvolution + ESTIMATE tumor microenvironment scores.

    Combines:
    - ssGSEA with Bindea et al. 2013 immune cell markers (~24 cell types)
    - ESTIMATE algorithm (Yoshihara et al. 2013): stromal score, immune score,
      tumor purity = cos(0.6049872018 + 0.0001467884 × (stromal + immune))

    Gene lists are hardcoded as module-level constants for reproducibility.

    Returns:
        (n_samples, ~27) immune/microenvironment scores
    """
    try:
        import gseapy as gp
    except ImportError:
        raise ImportError("gseapy required for immune scores: pip install gseapy")

    gene_set = set(gene_names)
    n_samples = expression.shape[0]

    # --- Bindea immune cell ssGSEA ---
    # Filter marker gene sets to genes present in expression matrix
    filtered_markers = {}
    for cell_type, markers in BINDEA_IMMUNE_CELL_MARKERS.items():
        present = [g for g in markers if g in gene_set]
        if len(present) >= 3:  # need minimum genes for ssGSEA
            filtered_markers[cell_type] = present

    df = pd.DataFrame(expression.T, index=gene_names,
                       columns=[f'S{i}' for i in range(n_samples)])

    bindea_scores = np.zeros((n_samples, len(filtered_markers)), dtype=np.float32)
    cell_type_names = sorted(filtered_markers.keys())

    if filtered_markers:
        result = gp.ssgsea(
            data=df,
            gene_sets=filtered_markers,
            outdir=None,
            no_plot=True,
            min_size=1,
            threads=1,
        )
        scores = result.res2d.pivot(index='Name', columns='Term', values='NES')
        if scores.empty:
            scores = result.resultsOnSamples
        # Align columns to cell_type_names
        for j, ct in enumerate(cell_type_names):
            if ct in scores.columns:
                col = scores[ct].values
                if len(col) == n_samples:
                    bindea_scores[:, j] = col
                elif scores.shape[0] == n_samples:
                    bindea_scores[:, j] = scores[ct].values
                else:
                    # Transposed
                    bindea_scores[:, j] = scores.loc[ct].values[:n_samples]

    # --- ESTIMATE scores ---
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    def _score_geneset(geneset):
        """Mean expression of genes in set (simple ESTIMATE scoring)."""
        idx = [gene_to_idx[g] for g in geneset if g in gene_to_idx]
        if not idx:
            return np.zeros(n_samples, dtype=np.float32)
        return expression[:, idx].mean(axis=1).astype(np.float32)

    stromal_score = _score_geneset(ESTIMATE_STROMAL_GENES)
    immune_score = _score_geneset(ESTIMATE_IMMUNE_GENES)
    # ESTIMATE tumor purity formula
    tumor_purity = np.cos(0.6049872018 + 0.0001467884 * (stromal_score + immune_score))

    estimate_scores = np.column_stack([stromal_score, immune_score, tumor_purity])

    # Combine: Bindea cell types + ESTIMATE (stromal, immune, purity)
    X = np.hstack([bindea_scores, estimate_scores]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    print(f"    Immune: {len(cell_type_names)} Bindea cell types + 3 ESTIMATE = {X.shape[1]} features")
    return X


# =============================================================================
# MULTI-OMIC REPRESENTATIONS (Part 2: require Xena downloads)
# =============================================================================

def create_cnv_gistic(sample_ids: List[str], cohort: str = '',
                      n_top: int = 200) -> np.ndarray:
    """Top-N most variable genes by GISTIC2 copy number score variance.

    Args:
        sample_ids: TCGA sample barcodes
        cohort: TCGA cohort string (e.g. 'TCGA-LUAD+TCGA-LUSC')
        n_top: number of top variable CNV genes

    Returns:
        (n_samples, n_top) GISTIC2 CNV scores for most variable genes
    """
    cnv_matrix, gene_names, cnv_ids = _load_cnv_for_samples(sample_ids, cohort)

    # Select top variable genes
    variances = np.var(cnv_matrix, axis=0)
    n_top = min(n_top, cnv_matrix.shape[1])
    top_idx = np.argsort(variances)[-n_top:]
    X = cnv_matrix[:, top_idx]

    print(f"    CNV GISTIC: top {n_top} variable genes")
    return X.astype(np.float32)


def create_cnv_pca50(sample_ids: List[str], cohort: str = '',
                     n_components: int = 50) -> np.ndarray:
    """PCA of full GISTIC2 copy number matrix.

    Returns:
        (n_samples, n_components) PCA-transformed CNV features
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    cnv_matrix, gene_names, cnv_ids = _load_cnv_for_samples(sample_ids, cohort)

    n_components = min(n_components, cnv_matrix.shape[1], cnv_matrix.shape[0])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cnv_matrix)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_.sum()
    print(f"    CNV PCA: {n_components} components, {explained:.1%} variance explained")
    return X_pca.astype(np.float32)


def _load_cnv_for_samples(sample_ids: List[str], cohort: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and match CNV data to sample_ids, handling multi-cohort datasets."""
    from kirby.datasets.tcga import _download_xena_cnv

    cohorts = _parse_cohorts(cohort)
    all_cnv, all_genes_list, all_ids = [], [], []

    for c in cohorts:
        cnv_matrix, gene_names, cnv_ids = _download_xena_cnv(c)
        all_cnv.append((cnv_matrix, gene_names))
        all_ids.extend(cnv_ids)
        all_genes_list.append(gene_names)

    # Intersect genes across cohorts if needed
    if len(all_cnv) > 1:
        common_genes = set(all_genes_list[0])
        for gn in all_genes_list[1:]:
            common_genes &= set(gn)
        common_genes = sorted(common_genes)
        aligned = []
        for cnv_matrix, gene_names in all_cnv:
            gene_to_idx = {g: i for i, g in enumerate(gene_names)}
            idx = [gene_to_idx[g] for g in common_genes]
            aligned.append(cnv_matrix[:, idx])
        combined_cnv = np.vstack(aligned)
        all_genes = np.array(common_genes)
    else:
        combined_cnv = all_cnv[0][0]
        all_genes = all_cnv[0][1]

    # Match to sample_ids order
    id_to_idx = {sid: i for i, sid in enumerate(all_ids)}
    matched = np.zeros((len(sample_ids), combined_cnv.shape[1]), dtype=np.float32)
    n_found = 0
    for i, sid in enumerate(sample_ids):
        for trunc in [sid[:15], sid[:12]]:
            if trunc in id_to_idx:
                matched[i] = combined_cnv[id_to_idx[trunc]]
                n_found += 1
                break

    print(f"    CNV: {n_found}/{len(sample_ids)} samples matched")
    return matched, all_genes, all_ids


def create_rppa(sample_ids: List[str]) -> np.ndarray:
    """RPPA protein expression features, StandardScaler-normalized.

    Uses pan-cancer RPPA data from TCGA (~260 proteins).

    Returns:
        (n_samples, ~260) normalized protein expression
    """
    from sklearn.preprocessing import StandardScaler
    from kirby.datasets.tcga import _download_xena_rppa

    rppa_matrix, protein_names = _download_xena_rppa(sample_ids)

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(rppa_matrix)

    print(f"    RPPA: {X.shape[1]} proteins")
    return X.astype(np.float32)


def create_mirna_top200(sample_ids: List[str], cohort: str = '',
                        n_top: int = 200) -> np.ndarray:
    """Top-N most variable miRNAs, z-scored.

    Returns:
        (n_samples, n_top) z-scored miRNA expression
    """
    from sklearn.preprocessing import StandardScaler
    from kirby.datasets.tcga import _download_xena_mirna

    cohorts = _parse_cohorts(cohort)
    all_mirna, all_names_list, all_ids = [], [], []

    for c in cohorts:
        mirna_matrix, mirna_names, mirna_ids = _download_xena_mirna(c)
        all_mirna.append((mirna_matrix, mirna_names))
        all_ids.extend(mirna_ids)
        all_names_list.append(mirna_names)

    # Intersect miRNAs across cohorts if needed
    if len(all_mirna) > 1:
        common_mirnas = set(all_names_list[0])
        for mn in all_names_list[1:]:
            common_mirnas &= set(mn)
        common_mirnas = sorted(common_mirnas)
        aligned = []
        for mirna_matrix, mirna_names in all_mirna:
            name_to_idx = {m: i for i, m in enumerate(mirna_names)}
            idx = [name_to_idx[m] for m in common_mirnas]
            aligned.append(mirna_matrix[:, idx])
        combined = np.vstack(aligned)
    else:
        combined = all_mirna[0][0]

    # Match to sample_ids
    id_to_idx = {sid: i for i, sid in enumerate(all_ids)}
    matched = np.zeros((len(sample_ids), combined.shape[1]), dtype=np.float32)
    n_found = 0
    for i, sid in enumerate(sample_ids):
        for trunc in [sid[:15], sid[:12]]:
            if trunc in id_to_idx:
                matched[i] = combined[id_to_idx[trunc]]
                n_found += 1
                break

    print(f"    miRNA: {n_found}/{len(sample_ids)} matched")

    # Select top variable, z-score
    variances = np.var(matched, axis=0)
    n_top = min(n_top, matched.shape[1])
    top_idx = np.argsort(variances)[-n_top:]
    X = matched[:, top_idx]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"    miRNA top {n_top}: z-scored")
    return X.astype(np.float32)


def create_clinical(sample_ids: List[str], cohort: str = '') -> np.ndarray:
    """Clinical features: age, sex, stage, grade (ordinal-encoded).

    Returns:
        (n_samples, ~10) clinical feature matrix
    """
    from kirby.datasets.tcga import _download_xena_clinical

    # Download clinical for all cohorts and merge
    cohorts = _parse_cohorts(cohort)
    clinical_frames = []
    for c in cohorts:
        clinical_frames.append(_download_xena_clinical(c))
    clinical = pd.concat(clinical_frames, ignore_index=True).drop_duplicates()

    # Find key columns
    sample_col = next(
        (c for c in clinical.columns if c.lower() in ('sample', 'sampleid', '_sample_id')),
        clinical.columns[0]
    )

    clinical['_match_id'] = clinical[sample_col].astype(str).str[:15]
    clin_lookup = {row['_match_id']: row for _, row in clinical.iterrows()}

    features = []
    for sid in sample_ids:
        row = clin_lookup.get(sid[:15], None)
        feats = _extract_clinical_features(row, clinical.columns)
        features.append(feats)

    X = np.array(features, dtype=np.float32)

    # Impute missing (NaN -> column median)
    for j in range(X.shape[1]):
        col = X[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            col[np.isnan(col)] = np.median(valid)
        else:
            col[np.isnan(col)] = 0.0

    print(f"    Clinical: {X.shape[1]} features")
    return X


def _extract_clinical_features(row, all_columns) -> List[float]:
    """Extract clinical features from a single row."""
    feats = []

    if row is None:
        return [np.nan] * 10

    # Age at diagnosis (normalize by dividing by 100)
    age = _get_clinical_val(row, all_columns,
                            ['age_at_initial_pathologic_diagnosis', 'age_at_diagnosis',
                             'years_to_birth'])
    if age is not None and age < 0:
        age = abs(age)  # years_to_birth is negative
    feats.append(age / 100.0 if age is not None else np.nan)

    # Sex (0=female, 1=male)
    sex_val = _get_clinical_str(row, all_columns, ['gender', 'sex'])
    if sex_val is not None:
        feats.append(1.0 if sex_val.upper().startswith('M') else 0.0)
    else:
        feats.append(np.nan)

    # Pathologic stage (ordinal 0-4)
    stage_val = _get_clinical_str(row, all_columns,
                                   ['pathologic_stage', 'clinical_stage',
                                    'ajcc_pathologic_tumor_stage'])
    feats.append(_encode_stage(stage_val))

    # Pathologic T (tumor size, ordinal)
    t_val = _get_clinical_str(row, all_columns, ['pathologic_T', 'ajcc_tumor_pathologic_pt'])
    feats.append(_encode_tnm(t_val))

    # Pathologic N (nodes, ordinal)
    n_val = _get_clinical_str(row, all_columns, ['pathologic_N', 'ajcc_nodes_pathologic_pn'])
    feats.append(_encode_tnm(n_val))

    # Pathologic M (metastasis, ordinal)
    m_val = _get_clinical_str(row, all_columns, ['pathologic_M', 'ajcc_metastasis_pathologic_pm'])
    feats.append(_encode_tnm(m_val))

    # Histologic grade (ordinal)
    grade_val = _get_clinical_str(row, all_columns,
                                   ['neoplasm_histologic_grade', 'histologic_grade',
                                    'histological_grade'])
    feats.append(_encode_grade(grade_val))

    # Tobacco smoking history (ordinal 1-5)
    smoke_val = _get_clinical_val(row, all_columns,
                                   ['tobacco_smoking_history'])
    feats.append(smoke_val / 5.0 if smoke_val is not None else np.nan)

    # Residual tumor (R0=0, R1=1, R2=2)
    resid_val = _get_clinical_str(row, all_columns, ['residual_tumor'])
    if resid_val is not None:
        for i, code in enumerate(['R0', 'R1', 'R2', 'RX']):
            if code in resid_val.upper():
                feats.append(i / 3.0)
                break
        else:
            feats.append(np.nan)
    else:
        feats.append(np.nan)

    # Radiation therapy (binary)
    rad_val = _get_clinical_str(row, all_columns, ['radiation_therapy'])
    if rad_val is not None:
        feats.append(1.0 if rad_val.upper() in ('YES', 'TRUE', '1') else 0.0)
    else:
        feats.append(np.nan)

    return feats


def _get_clinical_val(row, all_columns, candidates):
    """Get numeric clinical value from row."""
    for col in candidates:
        matching = [c for c in all_columns if c.lower() == col.lower()]
        if matching:
            try:
                val = float(row[matching[0]])
                if not np.isnan(val):
                    return val
            except (ValueError, TypeError):
                continue
    return None


def _get_clinical_str(row, all_columns, candidates):
    """Get string clinical value from row."""
    for col in candidates:
        matching = [c for c in all_columns if c.lower() == col.lower()]
        if matching:
            val = str(row[matching[0]])
            if val and val.lower() not in ('nan', 'none', '', '[not available]',
                                            '[not applicable]', '[unknown]'):
                return val
    return None


def _encode_stage(stage_str) -> float:
    """Encode pathologic stage as ordinal 0-4."""
    if stage_str is None:
        return np.nan
    s = stage_str.upper().replace(' ', '')
    if 'IV' in s:
        return 4.0 / 4.0
    if 'III' in s:
        return 3.0 / 4.0
    if 'II' in s:
        return 2.0 / 4.0
    if 'I' in s and 'II' not in s and 'IV' not in s:
        return 1.0 / 4.0
    if '0' in s:
        return 0.0
    return np.nan


def _encode_tnm(tnm_str) -> float:
    """Encode TNM value as ordinal."""
    if tnm_str is None:
        return np.nan
    s = tnm_str.upper().strip()
    for i in range(5):
        if str(i) in s:
            return float(i) / 4.0
    if 'X' in s:
        return np.nan
    return np.nan


def _encode_grade(grade_str) -> float:
    """Encode histologic grade as ordinal."""
    if grade_str is None:
        return np.nan
    s = grade_str.upper()
    for prefix in ['G', 'GRADE ']:
        if prefix in s:
            for i in range(1, 5):
                if str(i) in s:
                    return float(i) / 4.0
    if 'HIGH' in s:
        return 1.0
    if 'LOW' in s:
        return 0.25
    return np.nan


def _parse_cohorts(cohort: str) -> List[str]:
    """Parse cohort string into list of individual cohorts.

    Handles 'TCGA-LUAD+TCGA-LUSC' format.
    """
    if not cohort:
        return ['TCGA-LUAD']  # default
    return [c.strip() for c in cohort.split('+')]


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
