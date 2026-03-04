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

    # Pre-filter: remove genes with >90% zero expression
    nonzero_frac = (expression > 0).mean(axis=0)
    keep = nonzero_frac >= 0.1
    if keep.sum() < expression.shape[1]:
        n_removed = expression.shape[1] - keep.sum()
        expression = expression[:, keep]
        if gene_names is not None:
            gene_names = gene_names[keep]
        print(f"    Zero-expression filter: removed {n_removed} genes, {keep.sum()} remain")

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

    # Pre-filter: remove genes with >90% zero expression
    nonzero_frac = (expression > 0).mean(axis=0)
    keep = nonzero_frac >= 0.1
    if keep.sum() < expression.shape[1]:
        n_removed = expression.shape[1] - keep.sum()
        expression = expression[:, keep]
        print(f"    Zero-expression filter: removed {n_removed} genes, {keep.sum()} remain")

    n_components = min(n_components, expression.shape[1], expression.shape[0])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(expression)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_.sum()
    print(f"    PCA: {n_components} components explain {explained:.1%} variance")

    return X_pca.astype(np.float32)


def _extract_ssgsea_matrix(result, n_samples: int) -> np.ndarray:
    """Extract (n_samples, n_gene_sets) matrix from gseapy ssgsea result.

    Handles different gseapy versions where 'Name'/'Term' columns may swap
    between sample names and gene set names. Uses sample name matching
    (S0, S1, ...) to determine correct orientation.
    """
    sample_names = [f'S{i}' for i in range(n_samples)]
    sample_set = set(sample_names)

    scores = result.res2d.pivot(index='Name', columns='Term', values='NES')
    if scores.empty:
        try:
            scores = result.resultsOnSamples
        except AttributeError:
            raise RuntimeError("ssGSEA returned empty results")

    # Determine orientation by checking where sample names appear
    row_matches = sum(1 for s in sample_names[:5] if s in scores.index)
    col_matches = sum(1 for s in sample_names[:5] if s in scores.columns)

    if col_matches > row_matches:
        scores = scores.T

    # Reindex to ensure correct sample order and count
    if sample_set.issubset(set(scores.index)):
        scores = scores.reindex(sample_names)

    X = scores.values
    if X.shape[0] != n_samples:
        if X.shape[1] == n_samples:
            X = X.T
        else:
            raise RuntimeError(
                f"ssGSEA shape mismatch: got {X.shape}, expected "
                f"({n_samples}, n_gene_sets)"
            )

    return np.nan_to_num(X, nan=0.0).astype(np.float32)


def _score_gene_sets(expression: np.ndarray, gene_names: np.ndarray,
                     gene_sets: dict) -> np.ndarray:
    """Mean z-scored expression scoring for unweighted gene sets.

    Z-scores the full expression matrix once, then computes the mean z-score
    across member genes for each gene set per sample.

    Args:
        expression: (n_samples, n_genes) expression matrix
        gene_names: (n_genes,) gene symbols
        gene_sets: {set_name: [gene1, gene2, ...]} — unweighted gene lists

    Returns:
        (n_samples, n_scored_sets) pathway scores
    """
    from sklearn.preprocessing import StandardScaler

    # Z-score full matrix once
    scaler = StandardScaler()
    Z = scaler.fit_transform(expression)

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    set_names = sorted(gene_sets.keys())
    scores = np.zeros((expression.shape[0], len(set_names)), dtype=np.float32)
    kept = []

    for j, name in enumerate(set_names):
        members = gene_sets[name]
        idx = [gene_to_idx[g] for g in members if g in gene_to_idx]
        if len(idx) >= 3:  # minimum gene overlap
            scores[:, j] = Z[:, idx].mean(axis=1)
            kept.append(j)

    # Keep only sets with sufficient gene overlap
    if kept:
        scores = scores[:, kept]
    else:
        raise RuntimeError("No gene sets had >=3 overlapping genes with expression data")

    return scores


def _score_weighted_pathways(expression: np.ndarray, gene_names: np.ndarray,
                             pathway_weights: dict) -> np.ndarray:
    """Weighted linear scoring for PROGENy/DoRothEA-style networks.

    Score = sum(weight * z_expr) / n_genes per pathway per sample.

    Args:
        expression: (n_samples, n_genes) expression matrix
        gene_names: (n_genes,) gene symbols
        pathway_weights: {pathway: {gene: weight, ...}}

    Returns:
        (n_samples, n_scored_pathways) activity scores
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Z = scaler.fit_transform(expression)

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    pathway_names = sorted(pathway_weights.keys())
    scores = np.zeros((expression.shape[0], len(pathway_names)), dtype=np.float32)
    kept = []

    for j, name in enumerate(pathway_names):
        members = pathway_weights[name]
        idx = []
        weights = []
        for gene, w in members.items():
            if gene in gene_to_idx:
                idx.append(gene_to_idx[gene])
                weights.append(w)
        if len(idx) >= 3:
            w_arr = np.array(weights, dtype=np.float64)
            scores[:, j] = (Z[:, idx] * w_arr[np.newaxis, :]).sum(axis=1) / len(idx)
            kept.append(j)

    if kept:
        scores = scores[:, kept]
    else:
        raise RuntimeError("No pathways had >=3 overlapping genes with expression data")

    return scores


def create_hallmark_scores(expression: np.ndarray,
                           gene_names: np.ndarray) -> np.ndarray:
    """Mean z-score enrichment for MSigDB Hallmark pathways (50 pathways).

    Uses local gene sets (no runtime downloads).

    Returns:
        (n_samples, ~50) pathway enrichment scores
    """
    from kirby.representations.data import hallmark
    gene_sets = hallmark()
    X = _score_gene_sets(expression, gene_names, gene_sets)
    print(f"    Hallmark: {X.shape[1]} pathways scored")
    return X


def _normalize_decoupler_net(net, context: str = 'network'):
    """Normalize decoupler network DataFrame column names across versions.

    Handles v1.x (genesymbol/label/value) and v2.x (source/target/weight)
    column naming conventions. Also handles pandas NAType values.
    """
    # Normalize column names across decoupler versions
    _rename = {
        'genesymbol': 'target', 'label': 'source', 'value': 'weight',
        'gene': 'target', 'tf': 'source', 'pathway': 'source',
        'mor': 'weight',
    }
    for old, new in _rename.items():
        if old in net.columns and new not in net.columns:
            net = net.rename(columns={old: new})

    required = {'source', 'target', 'weight'}
    missing = required - set(net.columns)
    if missing:
        raise RuntimeError(
            f"{context} missing columns {missing}. "
            f"Got: {list(net.columns)}. Try: pip install --upgrade decoupler"
        )

    # Convert weight column: handle NAType, mixed types, and NaN
    net = net.copy()
    net['weight'] = pd.to_numeric(net['weight'], errors='coerce')
    # Also ensure source/target are clean strings (no NAType)
    net['source'] = net['source'].astype(str)
    net['target'] = net['target'].astype(str)
    # Drop any row with NA in any required column
    net = net.dropna(subset=['source', 'target', 'weight'])
    # Filter out rows where str conversion produced 'nan' or '<NA>'
    net = net[~net['source'].isin(['nan', '<NA>', 'None', ''])]
    net = net[~net['target'].isin(['nan', '<NA>', 'None', ''])]

    return net


def create_progeny_scores(expression: np.ndarray,
                          gene_names: np.ndarray) -> np.ndarray:
    """PROGENy pathway activity scores (14 signaling pathways).

    Uses local weighted gene sets (no runtime downloads).

    Returns:
        (n_samples, 14) pathway activity scores
    """
    from kirby.representations.data import progeny_weights
    weights = progeny_weights()
    X = _score_weighted_pathways(expression, gene_names, weights)
    print(f"    PROGENy: {X.shape[1]} pathways scored")
    return X


def create_kegg_scores(expression: np.ndarray,
                       gene_names: np.ndarray) -> np.ndarray:
    """Mean z-score enrichment for KEGG pathways (~320 pathways).

    Uses local gene sets (no runtime downloads).

    Returns:
        (n_samples, ~320) pathway enrichment scores
    """
    from kirby.representations.data import kegg
    gene_sets = kegg()
    X = _score_gene_sets(expression, gene_names, gene_sets)
    print(f"    KEGG: {X.shape[1]} pathways scored")
    return X


def create_dorothea_scores(expression: np.ndarray,
                           gene_names: np.ndarray) -> np.ndarray:
    """DoRothEA transcription factor activity scores (~367 TFs).

    Uses local weighted regulons (no runtime downloads).

    Returns:
        (n_samples, ~367) TF activity scores
    """
    from kirby.representations.data import dorothea_regulons
    regulons = dorothea_regulons()
    X = _score_weighted_pathways(expression, gene_names, regulons)
    print(f"    DoRothEA: {X.shape[1]} TFs scored")
    return X


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

    # Filter to non-silent variants only (standard practice)
    if 'effect' in mutations.columns:
        non_silent_types = {
            'Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del',
            'Frame_Shift_Ins', 'Splice_Site', 'In_Frame_Del', 'In_Frame_Ins',
            'Nonstop_Mutation', 'Translation_Start_Site',
        }
        before = len(mutations)
        mutations = mutations[mutations['effect'].isin(non_silent_types)]
        print(f"    Filtered to non-silent: {len(mutations)}/{before} mutations")

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
    """Mean z-score enrichment for Reactome pathways (~1818 pathways).

    Uses local gene sets (no runtime downloads).

    Returns:
        (n_samples, ~1818) Reactome pathway enrichment scores
    """
    from kirby.representations.data import reactome
    gene_sets = reactome()
    X = _score_gene_sets(expression, gene_names, gene_sets)
    print(f"    Reactome: {X.shape[1]} pathways scored")
    return X


def create_oncogenic_scores(expression: np.ndarray,
                            gene_names: np.ndarray) -> np.ndarray:
    """Mean z-score enrichment for MSigDB C6 oncogenic signatures (189 gene sets).

    Uses local gene sets (no runtime downloads).

    Returns:
        (n_samples, ~189) oncogenic signature enrichment scores
    """
    from kirby.representations.data import oncogenic
    gene_sets = oncogenic()
    X = _score_gene_sets(expression, gene_names, gene_sets)
    print(f"    Oncogenic: {X.shape[1]} signatures scored")
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
        # Use shared helper for correct orientation
        X_immune = _extract_ssgsea_matrix(result, n_samples)
        # Map columns from extracted matrix back to cell_type_names
        scores = result.res2d.pivot(index='Name', columns='Term', values='NES')
        if scores.empty:
            try:
                scores = result.resultsOnSamples
            except AttributeError:
                pass
        # Determine gene set names from pivot columns or rows
        sample_names = [f'S{i}' for i in range(n_samples)]
        if any(s in scores.columns for s in sample_names[:3]):
            gene_set_names = list(scores.index)
        else:
            gene_set_names = list(scores.columns)
        # Map to cell_type_names order
        for j, ct in enumerate(cell_type_names):
            if ct in gene_set_names:
                ct_idx = gene_set_names.index(ct)
                if ct_idx < X_immune.shape[1]:
                    bindea_scores[:, j] = X_immune[:, ct_idx]

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
# EXPRESSION-DERIVED REPRESENTATIONS (Part 2: new alternative reps)
# =============================================================================

def create_quantile_rank_pca(expression: np.ndarray,
                             n_components: int = 50) -> np.ndarray:
    """Rank-transform expression then PCA — robust to outliers and batch effects.

    Rank-normalizing each gene across samples removes outlier sensitivity and
    produces features orthogonal to standard PCA (which operates on raw values).

    Returns:
        (n_samples, n_components) PCA of rank-transformed expression
    """
    from sklearn.decomposition import PCA
    from scipy.stats import rankdata

    n_samples, n_genes = expression.shape
    n_components = min(n_components, n_genes, n_samples)

    # Rank-transform each gene (column) across samples
    ranked = np.zeros_like(expression, dtype=np.float32)
    for j in range(n_genes):
        ranked[:, j] = rankdata(expression[:, j], method='average')
    # Normalize ranks to [0, 1]
    ranked /= (n_samples + 1)

    pca = PCA(n_components=n_components, random_state=42)
    X = pca.fit_transform(ranked)

    explained = pca.explained_variance_ratio_.sum()
    print(f"    Quantile-rank PCA: {n_components} components, {explained:.1%} variance")
    return X.astype(np.float32)


def create_corr_modules(expression: np.ndarray,
                        n_modules: int = 30,
                        n_top_genes: int = 3000) -> np.ndarray:
    """Gene co-expression module eigengenes via spectral clustering.

    Clusters genes by correlation, then takes the first principal component
    of each cluster as the module eigengene. Captures co-regulation networks.

    Returns:
        (n_samples, n_modules) module eigengene matrix
    """
    from sklearn.cluster import SpectralClustering
    from sklearn.preprocessing import StandardScaler

    n_samples, n_genes = expression.shape
    n_top_genes = min(n_top_genes, n_genes)
    n_modules = min(n_modules, n_top_genes)

    # Select top variable genes for tractable correlation matrix
    variances = np.var(expression, axis=0)
    top_idx = np.argsort(variances)[-n_top_genes:]
    X_top = expression[:, top_idx]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_top)

    # Compute gene-gene correlation (absolute, for unsigned modules)
    corr = np.abs(np.corrcoef(X_scaled.T))
    corr = np.nan_to_num(corr, nan=0.0)

    # Spectral clustering on correlation affinity
    clustering = SpectralClustering(
        n_clusters=n_modules,
        affinity='precomputed',
        random_state=42,
        assign_labels='kmeans',
    )
    labels = clustering.fit_predict(corr)

    # Module eigengenes: first PC of each cluster
    eigengenes = np.zeros((n_samples, n_modules), dtype=np.float32)
    from sklearn.decomposition import PCA

    for k in range(n_modules):
        member_idx = np.where(labels == k)[0]
        if len(member_idx) < 2:
            eigengenes[:, k] = X_scaled[:, member_idx[0]] if len(member_idx) == 1 else 0.0
            continue
        pca = PCA(n_components=1, random_state=42)
        eigengenes[:, k] = pca.fit_transform(X_scaled[:, member_idx]).ravel()

    print(f"    Corr modules: {n_modules} modules from {n_top_genes} genes")
    return eigengenes


def create_chromosome_arm(expression: np.ndarray,
                          gene_names: np.ndarray) -> np.ndarray:
    """Per-chromosome-arm mean expression — captures chromosomal instability.

    Large-scale copy number alterations (e.g., arm-level gains/losses) shift
    mean expression of all genes on that arm. This is a simple proxy for CIN
    that requires only expression data.

    Returns:
        (n_samples, n_arms) arm-level mean expression (~39-44 arms)
    """
    from kirby.representations.data import gene_chr_arm_map

    chr_map = gene_chr_arm_map()
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Group genes by arm
    arm_genes = {}
    for gene, arm in chr_map.items():
        if gene in gene_to_idx:
            if arm not in arm_genes:
                arm_genes[arm] = []
            arm_genes[arm].append(gene_to_idx[gene])

    arm_names = sorted(arm_genes.keys())
    if not arm_names:
        raise RuntimeError(
            "No genes matched chromosome arm map. "
            "Ensure gene_names contains HUGO symbols (e.g., TP53, EGFR)."
        )

    X = np.zeros((expression.shape[0], len(arm_names)), dtype=np.float32)

    for j, arm in enumerate(arm_names):
        idx = arm_genes[arm]
        if idx:
            X[:, j] = expression[:, idx].mean(axis=1)

    print(f"    Chromosome arm: {len(arm_names)} arms")
    return X


def create_autoencoder50(expression: np.ndarray,
                         n_bottleneck: int = 50,
                         n_top_genes: int = 5000,
                         epochs: int = 50,
                         lr: float = 1e-3) -> np.ndarray:
    """2-layer autoencoder bottleneck — nonlinear manifold representation.

    Learns a compressed nonlinear representation via reconstruction loss.
    Captures structure that PCA/NMF/ICA (all linear) cannot.

    Returns:
        (n_samples, n_bottleneck) bottleneck activations
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        raise ImportError("torch required for autoencoder: pip install torch")

    from sklearn.preprocessing import StandardScaler

    n_samples, n_genes = expression.shape
    n_top_genes = min(n_top_genes, n_genes)

    # Select top variable genes
    variances = np.var(expression, axis=0)
    top_idx = np.argsort(variances)[-n_top_genes:]
    X_sub = expression[:, top_idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub).astype(np.float32)

    input_dim = X_scaled.shape[1]
    hidden_dim = max(n_bottleneck * 4, 200)

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_bottleneck),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(n_bottleneck, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.FloatTensor(X_scaled))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.shape[0]

    # Extract bottleneck activations
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        _, Z = model(X_tensor)
        bottleneck = Z.cpu().numpy()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_loss = total_loss / n_samples
    print(f"    Autoencoder: {n_bottleneck}D bottleneck, {epochs} epochs, "
          f"final loss={final_loss:.4f}")
    return bottleneck.astype(np.float32)


def create_sparse_pca50(expression: np.ndarray,
                        n_components: int = 50,
                        n_top_genes: int = 5000) -> np.ndarray:
    """Sparse PCA with L1 penalty — interpretable components with zero loadings.

    SparsePCA encourages most gene loadings to be zero, producing components
    that are easier to interpret biologically and orthogonal to standard PCA.

    Returns:
        (n_samples, n_components) sparse PCA features
    """
    from sklearn.decomposition import SparsePCA
    from sklearn.preprocessing import StandardScaler

    n_samples, n_genes = expression.shape
    n_top_genes = min(n_top_genes, n_genes)
    n_components = min(n_components, n_top_genes, n_samples)

    # Select top variable genes for tractability
    variances = np.var(expression, axis=0)
    top_idx = np.argsort(variances)[-n_top_genes:]
    X_sub = expression[:, top_idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    spca = SparsePCA(n_components=n_components, random_state=42,
                     max_iter=100, n_jobs=1)
    X_sparse = spca.fit_transform(X_scaled)

    # Report sparsity
    n_nonzero = np.count_nonzero(spca.components_)
    n_total = spca.components_.size
    sparsity = 1.0 - n_nonzero / n_total
    print(f"    Sparse PCA: {n_components} components, {sparsity:.1%} sparsity")
    return X_sparse.astype(np.float32)


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
# PRETRAINED FOUNDATION MODEL REPRESENTATIONS (frozen embeddings)
# =============================================================================

def create_geneformer_embeddings(expression: np.ndarray,
                                  gene_names: np.ndarray,
                                  max_input_size: int = 2048) -> np.ndarray:
    """Frozen Geneformer embeddings (no fine-tuning).

    Maps gene symbols to Geneformer vocabulary token IDs via Ensembl IDs,
    normalizes by corpus gene medians, rank-orders descending,
    passes through pretrained Geneformer, extracts [CLS] token embedding.
    Falls back to a small BERT model if Geneformer unavailable.

    Requires: torch, transformers, huggingface_hub

    Args:
        expression: (n_samples, n_genes) log2(TPM+1) matrix
        gene_names: (n_genes,) HUGO gene symbols
        max_input_size: max genes to include per sample

    Returns:
        (n_samples, hidden_size) frozen embeddings (768D for Geneformer, 256D fallback)
    """
    import torch

    n_samples = expression.shape[0]

    # Proper Geneformer tokenization: gene symbols → Ensembl → token IDs
    from kirby.finetuning.tcga_geneformer import _tokenize_expression
    tok_ids, tok_mask, proper_vocab = _tokenize_expression(
        expression, gene_names, max_input_size,
    )
    input_ids = torch.LongTensor(tok_ids)
    attention_mask = torch.LongTensor(tok_mask)

    # Load model
    try:
        from transformers import AutoModel, AutoConfig
        model_id = 'ctheodoris/Geneformer'
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        embed_dim = config.hidden_size
        print(f"    Loaded Geneformer ({embed_dim}D)")
    except Exception as e:
        from transformers import BertConfig, BertModel
        config = BertConfig(
            vocab_size=30000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=512,
            max_position_embeddings=max_input_size + 2,
        )
        model = BertModel(config)
        embed_dim = 256
        print(f"    Geneformer unavailable ({e}), using BERT fallback ({embed_dim}D)")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    # Extract embeddings in batches
    embeddings = np.zeros((n_samples, embed_dim), dtype=np.float32)
    batch_size = 16

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_ids = input_ids[start:end].to(device)
            batch_mask = attention_mask[start:end].to(device)
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
            # Use [CLS] token (first position)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings[start:end] = cls_emb

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings


def create_bulkrnabert_embeddings(expression: np.ndarray,
                                   gene_names: np.ndarray) -> np.ndarray:
    """Frozen BulkRNABert embeddings for bulk RNA-seq.

    BulkRNABert (InstaDeep, ML4H 2024) is a transformer pretrained ON TCGA bulk
    RNA-seq via masked gene expression modeling. Unlike Geneformer (single-cell),
    it was designed for bulk data and should be the primary foundation model for
    TCGA expression.

    Maps HUGO gene symbols → Ensembl IDs to match BulkRNABert's expected 19062-gene
    input order. Unmapped genes are zero-filled.

    Requires: transformers (from source for custom code), torch

    Args:
        expression: (n_samples, n_genes) log2(TPM+1) matrix
        gene_names: (n_genes,) HUGO gene symbols

    Returns:
        (n_samples, 256) frozen embeddings (layer 4 mean-pooled)
    """
    import torch
    from huggingface_hub import hf_hub_download

    n_samples, n_genes = expression.shape

    # 1. Get BulkRNABert's expected gene order (Ensembl IDs)
    csv_path = hf_hub_download(
        repo_id='InstaDeepAI/BulkRNABert',
        filename='data/tcga_sample.csv',
        repo_type='model',
    )
    import pandas as pd
    sample_df = pd.read_csv(csv_path, nrows=0)  # just headers
    expected_ensembl_ids = [c for c in sample_df.columns if c.startswith('ENSG')]
    n_expected = len(expected_ensembl_ids)
    print(f"    BulkRNABert expects {n_expected} genes (Ensembl IDs)")

    # 2. Map HUGO → Ensembl using Geneformer dictionaries (already available)
    try:
        from kirby.finetuning.tcga_geneformer import _load_geneformer_dictionaries
        _, _, name_id_dict = _load_geneformer_dictionaries()
    except Exception:
        # Fallback: try mygene for mapping
        name_id_dict = None

    if name_id_dict is not None:
        hugo_to_ensembl = {}
        for hugo, ensembl in name_id_dict.items():
            hugo_to_ensembl[hugo] = ensembl
            hugo_to_ensembl[hugo.upper()] = ensembl
    else:
        # Without mapping, we can't match genes. Return zeros with warning.
        print("    WARNING: No HUGO→Ensembl mapping available. Returning zero embeddings.")
        return np.zeros((n_samples, 256), dtype=np.float32)

    # 3. Build mapping: our gene columns → BulkRNABert's expected positions
    # Create lookup: ensembl_id → position in BulkRNABert's expected order
    ensembl_to_pos = {eid: i for i, eid in enumerate(expected_ensembl_ids)}

    # Map our gene names to BulkRNABert positions
    mapped = 0
    our_col_to_brb_pos = {}
    for i, gname in enumerate(gene_names):
        gname_str = str(gname)
        ensembl = hugo_to_ensembl.get(gname_str) or hugo_to_ensembl.get(gname_str.upper())
        if ensembl and ensembl in ensembl_to_pos:
            our_col_to_brb_pos[i] = ensembl_to_pos[ensembl]
            mapped += 1

    print(f"    Mapped {mapped}/{n_genes} HUGO genes to BulkRNABert's {n_expected} Ensembl IDs")

    if mapped == 0:
        print("    WARNING: No genes mapped. Returning zero embeddings.")
        return np.zeros((n_samples, 256), dtype=np.float32)

    # 4. Reorder expression to match BulkRNABert's expected gene order
    # Convert from log2(TPM+1) → TPM, then apply log10(1+x)
    tpm = np.power(2, expression) - 1
    tpm = np.maximum(tpm, 0)

    reordered = np.zeros((n_samples, n_expected), dtype=np.float32)
    for our_col, brb_pos in our_col_to_brb_pos.items():
        reordered[:, brb_pos] = tpm[:, our_col]

    # Apply BulkRNABert's expected preprocessing
    reordered = np.log10(1 + reordered)

    # 5. Load model and extract embeddings
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    config = AutoConfig.from_pretrained('InstaDeepAI/BulkRNABert', trust_remote_code=True)
    config.embeddings_layers_to_save = (4,)
    tokenizer = AutoTokenizer.from_pretrained('InstaDeepAI/BulkRNABert', trust_remote_code=True)
    model = AutoModel.from_pretrained(
        'InstaDeepAI/BulkRNABert', config=config, trust_remote_code=True
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    print(f"    Loaded BulkRNABert ({config.embed_dim}D, device={device})")

    # 6. Tokenize and extract embeddings in batches
    embed_dim = config.embed_dim  # 256
    embeddings = np.zeros((n_samples, embed_dim), dtype=np.float32)
    batch_size = 8  # 19062-length sequences are memory-intensive

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = reordered[start:end]
            input_ids = tokenizer.batch_encode_plus(
                batch, return_tensors='pt'
            )['input_ids'].to(device)
            out = model(input_ids)
            emb = out['embeddings_4'].mean(axis=1).cpu().numpy()
            embeddings[start:end] = emb

            if end < n_samples and end % 50 == 0:
                print(f"      Progress: {end}/{n_samples}")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings


def create_scfoundation_embeddings(expression: np.ndarray,
                                    gene_names: np.ndarray) -> np.ndarray:
    """Frozen scFoundation embeddings for bulk RNA-seq.

    scFoundation (Hao et al., Nature Methods 2024) uses continuous value
    encoding covering 19,264 genes. Returns CLS token embedding.

    Requires: torch, transformers, scFoundation (pip install or clone)

    Args:
        expression: (n_samples, n_genes) log2(TPM+1) matrix
        gene_names: (n_genes,) gene symbols

    Returns:
        (n_samples, 2048) frozen embeddings
    """
    import torch

    n_samples = expression.shape[0]

    try:
        from transformers import AutoModel, AutoTokenizer
        model_id = 'biomap-research/scFoundation'
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        embed_dim = 2048
        print(f"    Loaded scFoundation ({embed_dim}D)")
    except Exception as e:
        warnings.warn(f"scFoundation unavailable: {e} — returning zeros")
        return np.zeros((n_samples, 2048), dtype=np.float32)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    embeddings = np.zeros((n_samples, embed_dim), dtype=np.float32)
    batch_size = 8

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = torch.FloatTensor(expression[start:end]).to(device)
            try:
                outputs = model(batch)
                if hasattr(outputs, 'last_hidden_state'):
                    cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                elif hasattr(outputs, 'pooler_output'):
                    cls_emb = outputs.pooler_output.cpu().numpy()
                else:
                    cls_emb = outputs[0][:, 0, :].cpu().numpy()
                embeddings[start:end] = cls_emb[:, :embed_dim]
            except Exception as e:
                warnings.warn(f"scFoundation batch {start}-{end} failed: {e}")
            if start % 64 == 0 and start > 0:
                print(f"    Progress: {start}/{n_samples}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings


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


# =============================================================================
# WSI: ABMIL Aggregation (Attention-Based Multiple Instance Learning)
# =============================================================================


def _load_wsi_patch_embeddings(sample_ids, wsi_dir, model_name):
    """Load raw patch-level embeddings (not aggregated).

    Returns:
        list of np.ndarray: Each entry is (n_patches, d) or None if missing.
        int: embedding dimension
    """
    from kirby.datasets.tcga import _WSI_DIMS

    d = _WSI_DIMS.get(model_name, 768)
    wsi_dir = Path(wsi_dir) if wsi_dir else None

    if wsi_dir is None or not (wsi_dir / f'{model_name}.npy').exists():
        return [None] * len(sample_ids), d

    data = np.load(wsi_dir / f'{model_name}.npy', allow_pickle=True).item()

    # Build lookup
    slide_lookup = {}
    for slide_id, embedding in data.items():
        sid = str(slide_id)
        if sid.startswith('TCGA'):
            slide_lookup[sid[:12]] = embedding
            if len(sid) >= 15:
                slide_lookup[sid[:15]] = embedding
            slide_lookup[sid] = embedding
        else:
            slide_lookup[sid] = embedding

    # Auto-detect dim
    first_emb = next(iter(data.values()))
    if hasattr(first_emb, 'shape'):
        d = first_emb.shape[-1] if first_emb.ndim > 0 else d

    patches = []
    for sid in sample_ids:
        found = None
        for trunc in [sid[:15], sid[:12], sid]:
            if trunc in slide_lookup:
                emb = slide_lookup[trunc]
                if hasattr(emb, 'shape') and emb.ndim == 2:
                    found = emb
                elif hasattr(emb, 'shape') and emb.ndim == 1:
                    found = emb.reshape(1, -1)
                break
        patches.append(found)

    return patches, d


def gated_attention_pool(patches, d):
    """Frozen gated attention pooling (unsupervised, no training required).

    Computes attention weights based on patch norms and variance from the
    mean patch embedding. Patches that deviate more from the mean get higher
    weight, capturing informative (e.g., tumor) regions.

    Args:
        patches: list of (n_patches_i, d) arrays or None
        d: embedding dimension

    Returns:
        (n_samples, d) aggregated features
    """
    n = len(patches)
    X = np.zeros((n, d), dtype=np.float32)

    for i, patch_emb in enumerate(patches):
        if patch_emb is None or len(patch_emb) == 0:
            continue
        if patch_emb.shape[0] == 1:
            X[i] = patch_emb[0, :d]
            continue

        # Compute attention: patches far from mean get higher weight
        mean_patch = patch_emb.mean(axis=0)
        diffs = np.linalg.norm(patch_emb - mean_patch, axis=1)  # (n_patches,)
        # Gating: sigmoid of norm deviation (scaled)
        scale = np.std(diffs) + 1e-8
        gates = 1.0 / (1.0 + np.exp(-(diffs - np.mean(diffs)) / scale))
        # Normalize to sum to 1
        weights = gates / (gates.sum() + 1e-10)
        X[i] = (patch_emb[:, :d] * weights[:, None]).sum(axis=0)

    return X


def load_wsi_gated_attention(sample_ids, wsi_dir, model_name):
    """Load WSI features with frozen gated attention pooling."""
    patches, d = _load_wsi_patch_embeddings(sample_ids, wsi_dir, model_name)
    n_found = sum(1 for p in patches if p is not None)
    print(f"    WSI {model_name} (gated attention): {n_found}/{len(sample_ids)} matched, dim={d}")
    return gated_attention_pool(patches, d)


class ABMIL:
    """Attention-Based MIL aggregator (Ilse et al., ICML 2018).

    Learns a small attention network to weight-pool patch embeddings into a
    fixed-size slide-level representation. Trained with slide-level labels.

    Architecture:
        h_k = tanh(V * z_k)           (projection)
        a_k = softmax(w^T * h_k)      (attention weights)
        z_slide = sum_k a_k * z_k     (weighted pool)

    Usage:
        abmil = ABMIL(input_dim=512, hidden_dim=128)
        abmil.fit(train_patches, train_labels)
        X_test = abmil.transform(test_patches)
    """

    def __init__(self, input_dim, hidden_dim=128, lr=1e-3, epochs=20,
                 batch_size=32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def _build_model(self):
        import torch
        import torch.nn as nn

        class ABMILNet(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.attention_V = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.Tanh(),
                )
                self.attention_w = nn.Linear(hidden_dim, 1)

            def forward(self, bag):
                # bag: (n_patches, input_dim)
                h = self.attention_V(bag)        # (n_patches, hidden_dim)
                a = self.attention_w(h)           # (n_patches, 1)
                a = torch.softmax(a, dim=0)       # (n_patches, 1)
                z = (a * bag).sum(dim=0)          # (input_dim,)
                return z, a.squeeze()

        return ABMILNet(self.input_dim, self.hidden_dim)

    def fit(self, patches_list, labels, task_type='regression'):
        """Train ABMIL attention weights on slide-level labels.

        Args:
            patches_list: list of (n_patches_i, d) arrays (None = skip)
            labels: (n_samples,) target values
            task_type: 'regression' or 'classification'
        """
        import torch
        import torch.nn as nn

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._build_model().to(device)

        # Filter out missing slides
        valid = [(p, y) for p, y in zip(patches_list, labels) if p is not None]
        if len(valid) < 10:
            print("    ABMIL: too few slides, falling back to mean pool")
            self.model = None
            return self

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if task_type == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
            # Add a linear head for training
            head = nn.Linear(self.input_dim, 1).to(device)
            optimizer = torch.optim.Adam(
                list(self.model.parameters()) + list(head.parameters()),
                lr=self.lr
            )

        self.model.train()
        for epoch in range(self.epochs):
            np.random.shuffle(valid)
            epoch_loss = 0
            for patches, y in valid:
                bag = torch.FloatTensor(patches).to(device)
                z, _ = self.model(bag)

                if task_type == 'regression':
                    pred = head(z.unsqueeze(0)).squeeze()
                    target = torch.FloatTensor([y]).to(device)
                else:
                    pred = z.unsqueeze(0)
                    target = torch.LongTensor([int(y)]).to(device)

                loss = criterion(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        self.model.eval()
        self._head = head if task_type == 'regression' else None
        return self

    def transform(self, patches_list):
        """Apply learned attention to get slide-level embeddings.

        Args:
            patches_list: list of (n_patches_i, d) arrays (None → zeros)

        Returns:
            (n_samples, input_dim) aggregated features
        """
        import torch

        n = len(patches_list)
        X = np.zeros((n, self.input_dim), dtype=np.float32)

        if self.model is None:
            # Fallback to mean pool
            for i, p in enumerate(patches_list):
                if p is not None:
                    X[i] = p.mean(axis=0)[:self.input_dim]
            return X

        device = next(self.model.parameters()).device
        with torch.no_grad():
            for i, patches in enumerate(patches_list):
                if patches is None:
                    continue
                bag = torch.FloatTensor(patches).to(device)
                z, _ = self.model(bag)
                X[i] = z.cpu().numpy()[:self.input_dim]

        return X
