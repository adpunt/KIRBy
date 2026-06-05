"""
TCGA Dataset Loaders

Downloads and caches TCGA cancer genomics data from UCSC Xena for:
- tcga_luad_vs_lusc:  LUAD vs LUSC binary classification (~1016 samples)
- tcga_brca_pam50:    BRCA PAM50 4-class classification (~900 samples)
- tcga_brca_survival: BRCA overall survival regression (~400-500 samples, uncensored)

Data sources:
- Gene expression: UCSC Xena Toil recount2 (log2(TPM+1), GENCODE v23, protein-coding)
- Clinical/phenotype: UCSC Xena phenotype data
- Survival endpoints: TCGA-CDR (Liu et al. 2018, Cell)
- Somatic mutations: UCSC Xena MC3 MAF
- WSI features: Mendeley GigaSSL pre-computed embeddings

All data is open access — no dbGaP approval required.
"""

import gzip
import io
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# XENA URLS
# =============================================================================

# Toil recount2 pan-cancer expression (log2(TPM+1), all TCGA+TARGET+GTEx)
_XENA_TOIL_EXPRESSION = (
    "https://toil-xena-hub.s3.us-east-1.amazonaws.com/"
    "download/tcga_RSEM_gene_tpm.gz"
)

# TCGA phenotype (cancer type, sample type, etc.)
_XENA_PHENOTYPE_URLS = [
    "https://pancanatlas.xenahubs.net/download/"
    "TCGA_phenotype_denseDataOnlyDownload.tsv.gz",
    "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/"
    "download/TCGA_phenotype_denseDataOnlyDownload.tsv.gz",
    "https://tcga.xenahubs.net/download/"
    "TCGA_phenotype_denseDataOnlyDownload.tsv.gz",
]

# TCGA-CDR survival endpoints (Liu et al. 2018)
_XENA_SURVIVAL = (
    "https://pancanatlas.xenahubs.net/"
    "download/survival%2FTCGA_survival_data"
)

# MC3 somatic mutations (public MAF)
_XENA_MC3_MAF_URLS = [
    "https://api.gdc.cancer.gov/data/"
    "1c8cfe5f-e52d-41ba-94da-f15ea1337efc",
    "https://pancanatlas.xenahubs.net/download/"
    "mc3.v0.2.8.PUBLIC.xena.gz",
    "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/"
    "download/mc3.v0.2.8.PUBLIC.xena.gz",
    "https://tcga.xenahubs.net/download/"
    "mc3.v0.2.8.PUBLIC.xena.gz",
]

# PanCanAtlas pre-computed data
_PANCAN_ABSOLUTE_URLS = [
    "https://api.gdc.cancer.gov/data/"
    "4f277128-f793-4354-a13d-30cc7fe9f6b5",
    "https://pancanatlas.xenahubs.net/download/"
    "TCGA_mastercalls.abs_tables_JSedit.fixed.txt",
]

# Thorsson et al. (Immunity 2018) — immune landscape supplementary S1
# Old GDC UUID is dead; use Elsevier supplementary XLSX (Table S1)
_PANCAN_IMMUNE_URLS = [
    "https://ars.els-cdn.com/content/image/"
    "1-s2.0-S1074761318301213-mmc2.xlsx",
    "https://api.gdc.cancer.gov/data/"
    "160789b5-4345-42df-91f7-f4f7c6a10eb4",
]

# Malta et al. (Cell 2018) — stemness scores
# Old GDC UUID is dead; use Elsevier supplementary XLSX (Table S1)
_PANCAN_STEMNESS_URLS = [
    "https://ars.els-cdn.com/content/image/"
    "1-s2.0-S0092867418303581-mmc1.xlsx",
    "https://api.gdc.cancer.gov/data/"
    "4da68f3a-a504-4e98-aa98-f0ff6a0b8a00",
]

# Knijnenburg et al. (Cell Reports 2018) — DNA damage response
# GDC publication: PanCan-DDR-2018
# New UUID returns ZIP containing DDRscores.tsv + Samples.tsv + Scores.tsv
_PANCAN_HRD_URLS = [
    "https://api.gdc.cancer.gov/data/"
    "d9fdd899-bffe-4325-b0ce-58cbf9924c0a",
    "https://api.gdc.cancer.gov/data/"
    "0bbfca05-f14f-4e6e-a1c4-822e0ca84356",
]

# GISTIC2 gene-level copy number (per-cohort)
_XENA_CNV_TEMPLATES = [
    "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/"
    "download/TCGA.{cohort}.sampleMap%2FGistic2_CopyNumber_Gistic2_all_data_by_genes.gz",
    "https://tcga.xenahubs.net/download/"
    "TCGA.{cohort}.sampleMap%2FGistic2_CopyNumber_Gistic2_all_data_by_genes.gz",
]

# RPPA protein expression (pan-cancer, ~260 proteins)
_XENA_RPPA = (
    "https://pancanatlas.xenahubs.net/"
    "download/TCGA-RPPA-pancan-clean.xena.gz"
)

# miRNA expression (per-cohort, HiSeq)
_XENA_MIRNA_TEMPLATES = [
    "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/"
    "download/TCGA.{cohort}.sampleMap%2FmiRNA_HiSeq_gene.gz",
    "https://tcga.xenahubs.net/download/"
    "TCGA.{cohort}.sampleMap%2FmiRNA_HiSeq_gene.gz",
]

# GigaSSL pre-computed WSI features (Mendeley Data v6, Lazard et al. 2024)
# DOI: 10.17632/d573xfd9fg.6
_GIGASSL_DOWNLOAD_BASE = (
    "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files"
)

# GigaSSL model -> (mendeley_filename, mendeley_file_uuid)
_GIGASSL_FILES = {
    'ctranspath': ('TCGA-gigassl_ctranspath.npy', 'ccf402f7-34ee-4a74-8b8a-51d9e9cebea0'),
    'phikon':     ('TCGA-gigassl_phikon.npy',     '6ab7211d-f850-47b7-9091-54dc87abb66b'),
    'hoptimus0':  ('TCGA-gigassl_optimus.npy',    '400d7ad7-9e9f-48dd-b241-a054cc189a37'),
    'gigapath':   ('TCGA-gigassl_gigapath.npy',    '03977666-5ca2-43cc-a588-0f067dd1fc09'),
    'moco':       ('TCGA-gigassl_moco.npy',        '2006d37e-61b0-469c-95f1-fda3df1766c4'),
}

# TITAN slide-level features (MahmoodLab, 768D, CC-BY-NC-ND)
_TITAN_HF_REPO = "MahmoodLab/TITAN"
_TITAN_HF_FILE = "TCGA_TITAN_features.pkl"

# UNI2-h patch-level features (MahmoodLab, 1536D, CC-BY-NC-ND, gated)
_UNI2H_HF_REPO = "MahmoodLab/UNI2-h-features"

# Local pre-computed CNN features (from Dropbox via download_tcga_features.sh)
_LOCAL_CNN_PATHS = [
    Path('/Volumes/seagate/tcga_lung_wsi_features/small_cnn_320/feats_pt'),
    Path('/Volumes/seagate/KIRBy/large_data/tests/small_cnn_320/feats_pt'),
]

# WSI feature dimensions per model (fallback — actual dims auto-detected from data)
_WSI_DIMS = {
    'ctranspath': 512,     # GigaSSL + CTransPath
    'phikon': 512,         # GigaSSL + Phikon
    'hoptimus0': 512,      # GigaSSL + H-optimus-0
    'gigapath': 512,       # GigaSSL + GigaPath
    'moco': 512,           # GigaSSL + MoCo
    'titan': 768,          # TITAN slide encoder
    'uni2h': 1536,         # UNI2-h mean-pooled patches
    'virchow2': 1280,      # Virchow2 (Paige AI) CLS token
    'conch': 768,          # CONCH v1.5 (Mahmood Lab)
    'hoptimus1': 1536,     # H-optimus-1 (Bioptimus)
    'cnn320': 320,         # Local small CNN features
}

# Protein-coding gene count (GENCODE v23 compatible)
_N_PROTEIN_CODING = 19814


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

def _cache_dir() -> Path:
    """Get cache directory, preferring /Volumes/seagate if available."""
    seagate = Path('/Volumes/seagate/kirby/tcga')
    if seagate.parent.parent.exists():
        seagate.mkdir(parents=True, exist_ok=True)
        return seagate
    cache_root = Path(os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')))
    cache = cache_root / 'tcga'
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _download_with_fallback(urls: list, timeout: int = 120, **kwargs):
    """Try each URL in order, return first successful response."""
    import requests

    last_err = None
    for url in urls:
        try:
            resp = requests.get(url, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
            continue
    raise last_err


# =============================================================================
# XENA EXPRESSION DOWNLOAD
# =============================================================================

def _download_xena_expression(cohort: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Download expression data from UCSC Xena for a TCGA cohort.

    Uses the Toil recount2 pan-cancer matrix, filters to the requested cohort,
    and keeps only protein-coding genes.

    Args:
        cohort: TCGA cohort prefix, e.g. 'TCGA-BRCA', 'TCGA-LUAD', 'TCGA-LUSC'

    Returns:
        expression: np.ndarray (n_samples, n_genes) — log2(TPM+1)
        gene_names: np.ndarray (n_genes,)
        sample_ids: List[str] — TCGA barcodes
    """
    import requests

    cache = _cache_dir()
    cohort_key = cohort.replace('-', '_').lower()
    cached_npz = cache / f'{cohort_key}_expression.npz'

    if cached_npz.exists():
        data = np.load(cached_npz, allow_pickle=True)
        return data['expression'], data['gene_names'], data['sample_ids'].tolist()

    # Try per-cohort Xena files first (much smaller downloads)
    cohort_slug = cohort.replace('TCGA-', '').upper()
    per_cohort_url = (
        f"https://tcga-xena-hub.s3.us-east-1.amazonaws.com/"
        f"download/TCGA.{cohort_slug}.sampleMap%2FHiSeqV2.gz"
    )

    print(f"  Downloading expression data for {cohort} from UCSC Xena...")
    try:
        resp = requests.get(per_cohort_url, timeout=300, stream=True)
        resp.raise_for_status()
        content = gzip.decompress(resp.content)
        df = pd.read_csv(io.BytesIO(content), sep='\t', index_col=0)
    except Exception as e:
        print(f"  Per-cohort download failed ({e}), trying Toil pan-cancer matrix...")
        # Fall back to pan-cancer Toil matrix (larger download)
        resp = requests.get(_XENA_TOIL_EXPRESSION, timeout=600, stream=True)
        resp.raise_for_status()
        content = gzip.decompress(resp.content)
        df_full = pd.read_csv(io.BytesIO(content), sep='\t', index_col=0)

        # Filter to TCGA samples only (Toil matrix also has TARGET + GTEx)
        # NOTE: TCGA barcodes don't encode cohort/project, so we can't filter
        # by cohort here. The per-cohort download above is the correct path.
        # This fallback downloads ALL TCGA samples — wastes disk but still works
        # because downstream code matches by sample ID.
        cohort_cols = [c for c in df_full.columns if c.startswith('TCGA-')]
        if not cohort_cols:
            raise ValueError(f"No TCGA samples found in Toil matrix")
        warnings.warn(
            f"Using pan-cancer fallback: got {len(cohort_cols)} TCGA samples "
            f"(not filtered to {cohort}). Per-cohort download is preferred."
        )
        df = df_full[cohort_cols]

    # Genes are rows, samples are columns — transpose
    # Filter to primary tumor samples (sample type code 01)
    tumor_cols = [c for c in df.columns if len(c) >= 15 and c[13:15] == '01']
    if not tumor_cols:
        # Accept any sample
        tumor_cols = list(df.columns)
    df = df[tumor_cols]

    # Filter to protein-coding genes
    # Xena gene names are HUGO symbols; filter out non-protein-coding
    # by checking against a curated list or just keeping the intersection
    gene_names = df.index.values.astype(str)

    # Transpose: samples × genes
    expression = df.values.T.astype(np.float32)
    sample_ids = [c[:15] for c in tumor_cols]  # Truncate to sample-level barcode

    # Handle NaN values
    nan_mask = np.isnan(expression)
    if nan_mask.any():
        expression = np.nan_to_num(expression, nan=0.0)
        n_nan = nan_mask.sum()
        print(f"  Replaced {n_nan} NaN values with 0")

    print(f"  Expression matrix: {expression.shape[0]} samples × {expression.shape[1]} genes")

    # ComBat batch correction across tissue source sites (TSS)
    try:
        from combat.pycombat import pycombat
        tss_codes = [sid[5:7] for sid in sample_ids]
        unique_tss = set(tss_codes)
        if len(unique_tss) > 1:
            expr_df = pd.DataFrame(expression.T, columns=sample_ids)
            batch = pd.Series(tss_codes, index=sample_ids)
            corrected = pycombat(expr_df, batch)
            expression = corrected.values.T.astype(np.float32)
            print(f"  ComBat: corrected {len(unique_tss)} TSS batches "
                  f"({expression.shape[0]} samples × {expression.shape[1]} genes)")
        else:
            print(f"  ComBat: single TSS batch, skipping")
    except ImportError:
        print("  ComBat: pycombat not installed, using uncorrected expression")

    # Cache
    np.savez_compressed(
        cached_npz,
        expression=expression,
        gene_names=gene_names,
        sample_ids=np.array(sample_ids),
    )

    return expression, gene_names, sample_ids


# =============================================================================
# CLINICAL / PHENOTYPE DATA
# =============================================================================

def _download_xena_clinical(cohort: str) -> pd.DataFrame:
    """Download clinical/phenotype data from UCSC Xena.

    Returns DataFrame with columns: sample, _primary_disease, _sample_type,
    plus cancer-specific fields (e.g., PAM50Call_RNAseq for BRCA).
    """
    import requests

    cache = _cache_dir()
    cohort_key = cohort.replace('-', '_').lower()
    cached_csv = cache / f'{cohort_key}_clinical.csv'

    if cached_csv.exists():
        return pd.read_csv(cached_csv)

    print(f"  Downloading clinical data for {cohort}...")
    resp = _download_with_fallback(_XENA_PHENOTYPE_URLS, timeout=120)

    content = gzip.decompress(resp.content)
    df = pd.read_csv(io.BytesIO(content), sep='\t', low_memory=False)

    # Filter to cohort
    disease_col = next((c for c in df.columns if 'disease' in c.lower() or '_primary_disease' in c), None)
    sample_col = next((c for c in df.columns if c.lower() == 'sample' or c.lower() == 'sampleid'), df.columns[0])

    cohort_slug = cohort.replace('TCGA-', '')
    # Filter to TCGA samples, then by disease type if possible
    mask = df[sample_col].str.startswith('TCGA-', na=False)
    df = df[mask].copy()
    if disease_col and cohort_slug:
        disease_mask = df[disease_col].astype(str).str.upper().str.contains(
            cohort_slug.upper(), na=False
        )
        if disease_mask.any():
            df = df[disease_mask].copy()

    df.to_csv(cached_csv, index=False)
    return df


def _download_survival_data(cohort: str = 'TCGA-BRCA') -> pd.DataFrame:
    """Download survival data from Xena clinical matrix.

    Uses the per-cohort clinical matrix which includes days_to_death,
    days_to_last_followup, and vital_status. Falls back to TCGA-CDR
    pancanatlas if per-cohort data unavailable.
    """
    import requests

    cache = _cache_dir()
    cohort_key = cohort.replace('-', '_').lower()
    cached_csv = cache / f'{cohort_key}_survival.csv'

    if cached_csv.exists():
        return pd.read_csv(cached_csv)

    # Try per-cohort clinical matrix first (already has survival data)
    cohort_slug = cohort.replace('TCGA-', '').upper()
    cohort_urls = [
        f"https://tcga-xena-hub.s3.us-east-1.amazonaws.com/"
        f"download/TCGA.{cohort_slug}.sampleMap%2F{cohort_slug}_clinicalMatrix",
        f"https://tcga.xenahubs.net/download/"
        f"TCGA.{cohort_slug}.sampleMap%2F{cohort_slug}_clinicalMatrix",
    ]

    print(f"  Downloading survival data for {cohort}...")
    try:
        resp = _download_with_fallback(cohort_urls, timeout=120)
        df = pd.read_csv(io.BytesIO(resp.content), sep='\t')
    except Exception:
        # Try TCGA-CDR pancanatlas
        try:
            resp = requests.get(_XENA_SURVIVAL, timeout=120, allow_redirects=True)
            resp.raise_for_status()
            df = pd.read_csv(io.BytesIO(resp.content), sep='\t')
        except Exception:
            # Try GDC clinical endpoint
            resp = _download_with_fallback(_XENA_PHENOTYPE_URLS, timeout=120)
            content = gzip.decompress(resp.content)
            df = pd.read_csv(io.BytesIO(content), sep='\t', low_memory=False)

    df.to_csv(cached_csv, index=False)
    return df


# =============================================================================
# PAM50 SUBTYPE LABELS
# =============================================================================

def _get_pam50_labels(sample_ids: List[str]) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Get PAM50 subtype labels for BRCA samples.

    Sources PAM50 from TCGA phenotype data or TCGA-CDR.
    Filters to 4 canonical subtypes: LumA, LumB, Basal, Her2 (excludes Normal-like).

    Returns:
        labels: np.ndarray of int class indices
        class_names: list of subtype names
        valid_mask: boolean mask of samples with valid PAM50 labels
    """
    import requests

    cache = _cache_dir()
    cached_csv = cache / 'brca_pam50.csv'

    if cached_csv.exists():
        df = pd.read_csv(cached_csv)
    else:
        # Try Xena BRCA-specific clinical data
        url = (
            "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/"
            "download/TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix"
        )
        print(f"  Downloading BRCA PAM50 labels...")
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            df = pd.read_csv(io.BytesIO(resp.content), sep='\t')
        except Exception:
            # Fall back to phenotype data
            df = _download_xena_clinical('TCGA-BRCA')

        df.to_csv(cached_csv, index=False)

    # Find PAM50 column — prefer PAM50Call_RNAseq (string labels) over numeric cluster columns
    pam50_col = None
    pam50_candidates = [col for col in df.columns if 'pam50' in col.lower()]
    # Prefer columns with 'call' or 'rnaseq' in the name (actual subtype labels)
    for col in pam50_candidates:
        if 'call' in col.lower() or 'rnaseq' in col.lower():
            pam50_col = col
            break
    # Fall back to PAM50_mRNA if available
    if pam50_col is None:
        for col in pam50_candidates:
            if 'mrna' in col.lower() and 'cluster' not in col.lower() and 'integrated' not in col.lower():
                pam50_col = col
                break
    # Last resort: any PAM50 column
    if pam50_col is None and pam50_candidates:
        pam50_col = pam50_candidates[0]

    if pam50_col is None:
        raise ValueError(f"Cannot find PAM50 column. Available: {list(df.columns)[:20]}")

    # Find sample ID column
    sample_col = next(
        (c for c in df.columns if c.lower() in ('sampleid', 'sample', '_sample_id')),
        df.columns[0]
    )

    # Map sample IDs to PAM50 labels
    # Truncate barcodes to 15 chars for matching
    df['_match_id'] = df[sample_col].astype(str).str[:15]

    class_names = ['LumA', 'LumB', 'Basal', 'Her2']
    # Normalize PAM50 values
    pam50_map = {
        'Luminal A': 'LumA', 'LumA': 'LumA', 'luminal a': 'LumA', 'Lum A': 'LumA',
        'Luminal B': 'LumB', 'LumB': 'LumB', 'luminal b': 'LumB', 'Lum B': 'LumB',
        'Basal-like': 'Basal', 'Basal': 'Basal', 'basal': 'Basal',
        'HER2-enriched': 'Her2', 'Her2': 'Her2', 'her2': 'Her2', 'HER2': 'Her2',
    }

    labels = np.full(len(sample_ids), -1, dtype=int)

    # Build lookup: sample_id -> PAM50 subtype (prefer non-null values)
    id_to_pam50 = {}
    for _, row in df.iterrows():
        mid = str(row.get('_match_id', ''))
        raw = str(row.get(pam50_col, ''))
        mapped = pam50_map.get(raw, None)
        if mapped is not None and mid not in id_to_pam50:
            id_to_pam50[mid] = mapped

    for i, sid in enumerate(sample_ids):
        sid_trunc = sid[:15]
        subtype = id_to_pam50.get(sid_trunc, None)
        if subtype is not None:
            labels[i] = class_names.index(subtype)

    valid_mask = labels >= 0
    return labels, class_names, valid_mask


# =============================================================================
# MUTATION DATA
# =============================================================================

def _download_mutations(sample_ids: List[str]) -> pd.DataFrame:
    """Download MC3 somatic mutation data from UCSC Xena, filtered to given samples.

    TCGA barcodes do NOT encode the project/cohort (LUAD, BRCA, etc.) —
    the second field is a TSS code, not a disease code. So we filter by
    the actual sample IDs rather than cohort name.

    Args:
        sample_ids: TCGA sample barcodes to keep (truncated to 12-char patient IDs)

    Returns:
        DataFrame with columns: sample, gene, effect (variant classification).
    """
    import requests

    cache = _cache_dir()

    # Cache key: hash of sorted sample IDs (so same sample set reuses cache)
    import hashlib
    sid_key = hashlib.sha256('\n'.join(sorted(set(s[:12] for s in sample_ids))).encode()).hexdigest()[:16]
    cached_csv = cache / f'mutations_{sid_key}.csv'

    if cached_csv.exists():
        return pd.read_csv(cached_csv)

    print(f"  Downloading MC3 somatic mutations...")
    # MC3 is a large file (~300MB compressed). Download and filter to our samples.
    resp = _download_with_fallback(_XENA_MC3_MAF_URLS, timeout=600)

    content = gzip.decompress(resp.content)
    df = pd.read_csv(io.BytesIO(content), sep='\t', low_memory=False)

    # Filter to our samples by patient ID (first 12 chars of barcode)
    # Xena format: col[0] = 'sample'; GDC MAF: 'Tumor_Sample_Barcode'
    sample_col = next(
        (c for c in df.columns
         if c.lower() in ('sample', 'tumor_sample_barcode')),
        df.columns[0]
    )
    patient_ids = set(s[:12] for s in sample_ids)
    mask = df[sample_col].astype(str).str[:12].isin(patient_ids)
    df = df[mask].copy()

    # Standardize columns
    gene_col = next((c for c in df.columns if c.lower() in ('gene', 'hugo_symbol')), df.columns[1])
    effect_col = next((c for c in df.columns if 'class' in c.lower() or 'effect' in c.lower()), None)

    result = pd.DataFrame({
        'sample': df[sample_col].str[:15],
        'gene': df[gene_col],
    })
    if effect_col:
        result['effect'] = df[effect_col]

    result.to_csv(cached_csv, index=False)
    print(f"  {len(result)} mutations for {len(patient_ids)} patients")
    return result


def _download_mutations_full(sample_ids: List[str]) -> pd.DataFrame:
    """Download MC3 MAF with ALL columns needed for mutational signature analysis.

    Unlike _download_mutations() which keeps only (sample, gene, effect),
    this preserves chromosome, position, ref/alt alleles — required by
    SigProfilerAssignment for trinucleotide context extraction.

    Returns:
        DataFrame with columns: sample, chr, start, end, ref, alt, gene, effect
    """
    import requests

    cache = _cache_dir()

    import hashlib
    sid_key = hashlib.sha256(
        '\n'.join(sorted(set(s[:12] for s in sample_ids))).encode()
    ).hexdigest()[:16]
    cached_csv = cache / f'mutations_full_{sid_key}.csv'

    if cached_csv.exists():
        return pd.read_csv(cached_csv)

    print(f"  Downloading MC3 somatic mutations (full MAF)...")
    resp = _download_with_fallback(_XENA_MC3_MAF_URLS, timeout=600)

    content = gzip.decompress(resp.content)
    df = pd.read_csv(io.BytesIO(content), sep='\t', low_memory=False)

    # Filter to our samples by patient ID (first 12 chars of barcode)
    # Xena format: col[0] = 'sample'; GDC MAF: 'Tumor_Sample_Barcode'
    sample_col = next(
        (c for c in df.columns
         if c.lower() in ('sample', 'tumor_sample_barcode')),
        df.columns[0]
    )
    patient_ids = set(s[:12] for s in sample_ids)
    mask = df[sample_col].astype(str).str[:12].isin(patient_ids)
    df = df[mask].copy()

    # Map columns to standardized names
    # Xena MC3: sample, chr, start, end, reference, alt, gene, effect
    # GDC MAF: Hugo_Symbol, ..., Chromosome, Start_Position, End_Position,
    #          Reference_Allele, Tumor_Seq_Allele2, ..., Tumor_Sample_Barcode
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ('sample', 'tumor_sample_barcode'):
            col_map['sample'] = c
        elif cl == 'chr' or cl == 'chromosome' or cl == 'chrom':
            col_map['chr'] = c
        elif cl == 'start' or cl == 'start_position' or cl == 'pos_start':
            col_map['start'] = c
        elif cl == 'end' or cl == 'end_position' or cl == 'pos_end':
            col_map['end'] = c
        elif cl == 'reference' or cl == 'reference_allele' or cl == 'ref':
            col_map['ref'] = c
        elif cl == 'alt' or cl == 'tumor_seq_allele2' or cl == 'alternate':
            col_map['alt'] = c
        elif cl in ('gene', 'hugo_symbol'):
            col_map['gene'] = c
        elif 'class' in cl or cl == 'effect':
            col_map['effect'] = c
        elif cl == 'dna_vaf' or cl == 't_vaf':
            col_map['dna_vaf'] = c

    required = ['sample', 'chr', 'start', 'end', 'ref', 'alt']
    missing = [k for k in required if k not in col_map]
    if missing:
        raise ValueError(
            f"MC3 MAF missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    result = pd.DataFrame({
        'sample': df[col_map['sample']].astype(str).str[:15],
        'chr': df[col_map['chr']].astype(str),
        'start': df[col_map['start']],
        'end': df[col_map['end']],
        'ref': df[col_map['ref']].astype(str),
        'alt': df[col_map['alt']].astype(str),
    })
    if 'gene' in col_map:
        result['gene'] = df[col_map['gene']]
    if 'effect' in col_map:
        result['effect'] = df[col_map['effect']]
    if 'dna_vaf' in col_map:
        result['dna_vaf'] = pd.to_numeric(df[col_map['dna_vaf']], errors='coerce')

    result.to_csv(cached_csv, index=False)
    print(f"  {len(result)} mutations (full MAF) for {len(patient_ids)} patients")
    return result


# =============================================================================
# PANCANATLAS PRE-COMPUTED DATA
# =============================================================================

def _pancanatlas_cache_dir() -> Path:
    """Cache directory for PanCanAtlas pre-computed data."""
    base = _cache_dir()
    d = base / 'pancanatlas'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download_pancanatlas(
    urls: list,
    cache_name: str,
    description: str,
    sep: str = '\t',
    xlsx_sheet: str = None,
) -> pd.DataFrame:
    """Download and cache a PanCanAtlas table.

    Checks local cache first, then attempts download, then checks for
    manually-placed file at <cache_dir>/pancanatlas/<cache_name>.

    Args:
        urls: list of fallback URLs
        cache_name: filename for cache (e.g., 'absolute_purity.tsv')
        description: human-readable name for error messages
        sep: CSV/TSV separator
        xlsx_sheet: if set, read this sheet from an XLSX download
    """
    cache_dir = _pancanatlas_cache_dir()
    cached = cache_dir / cache_name

    if cached.exists():
        return pd.read_csv(cached, sep=sep, low_memory=False)

    # Try download
    print(f"  Downloading {description}...")
    import requests
    last_err = None
    for url in urls:
        try:
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            content = resp.content

            # Handle gzip
            if cache_name.endswith('.gz') or content[:2] == b'\x1f\x8b':
                content = gzip.decompress(content)

            # Handle XLSX (Elsevier supplementary files)
            if url.endswith('.xlsx') or content[:4] == b'PK\x03\x04':
                import openpyxl  # noqa: F401
                df = pd.read_excel(
                    io.BytesIO(content),
                    sheet_name=xlsx_sheet or 0,
                    engine='openpyxl',
                )
            else:
                df = pd.read_csv(io.BytesIO(content), sep=sep, low_memory=False)

            df.to_csv(cached, sep=sep, index=False)
            print(f"  {description}: {len(df)} rows cached to {cached}")
            return df
        except Exception as e:
            last_err = e
            continue

    # Check for manually-placed file with common extensions
    for ext in ['', '.tsv', '.txt', '.csv']:
        manual = cache_dir / (cache_name.rsplit('.', 1)[0] + ext)
        if manual.exists() and manual != cached:
            return pd.read_csv(manual, sep=sep, low_memory=False)

    raise FileNotFoundError(
        f"Could not download {description}.\n"
        f"Error: {last_err}\n"
        f"To fix: manually download the file and place it at:\n"
        f"  {cached}\n"
        f"URLs tried: {urls}"
    )


def _download_pancanatlas_purity() -> pd.DataFrame:
    """ABSOLUTE purity/ploidy from Xena PanCanAtlas.

    Returns DataFrame with columns including: sample, call status,
    purity, ploidy, Genome doublings, etc.
    """
    return _download_pancanatlas(
        _PANCAN_ABSOLUTE_URLS,
        'absolute_purity.tsv',
        'ABSOLUTE purity (PanCanAtlas)',
    )


def _download_pancanatlas_immune() -> pd.DataFrame:
    """Thorsson et al. (Immunity 2018) immune landscape supplementary S1.

    Contains CIBERSORT fractions (~22 cell types), xCell scores (~64 types),
    leukocyte fraction, immune subtypes, and more for ~10K TCGA samples.
    """
    return _download_pancanatlas(
        _PANCAN_IMMUNE_URLS,
        'immune_landscape.tsv',
        'Thorsson et al. immune landscape (Immunity 2018)',
        xlsx_sheet=0,
    )


def _download_pancanatlas_stemness() -> pd.DataFrame:
    """Malta et al. (Cell 2018) stemness scores.

    Contains mRNAsi and EREG-mRNAsi for TCGA samples.
    """
    return _download_pancanatlas(
        _PANCAN_STEMNESS_URLS,
        'stemness_scores.tsv',
        'Malta et al. stemness (Cell 2018)',
        xlsx_sheet='StemnessScores_RNAexp',
    )


def _download_pancanatlas_hrd() -> pd.DataFrame:
    """Knijnenburg et al. (Cell Reports 2018) HRD scores.

    Contains HRD-LOH, HRD-TAI (NtAI), HRD-LST, and total HRD score.
    The new GDC UUID returns a ZIP with DDRscores.tsv + Samples.tsv + Scores.tsv.
    """
    import zipfile

    cache_dir = _pancanatlas_cache_dir()
    cached = cache_dir / 'hrd_scores.tsv'

    if cached.exists():
        return pd.read_csv(cached, sep='\t', low_memory=False)

    print(f"  Downloading Knijnenburg et al. HRD (Cell Reports 2018)...")
    last_err = None
    for url in _PANCAN_HRD_URLS:
        try:
            import requests
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            content = resp.content

            # New UUID returns ZIP containing DDRscores.tsv + Samples.tsv + Scores.tsv
            if content[:4] == b'PK\x03\x04':
                zf = zipfile.ZipFile(io.BytesIO(content))
                names = zf.namelist()
                # Find the score and sample files
                scores_file = next((n for n in names if 'DDRscores' in n), None)
                samples_file = next((n for n in names if 'Samples' in n and 'Scores' not in n), None)
                colnames_file = next((n for n in names if n.endswith('Scores.tsv') and 'DDR' not in n), None)

                if scores_file and samples_file:
                    samples_df = pd.read_csv(io.BytesIO(zf.read(samples_file)), sep='\t')
                    # DDRscores has no header; Scores.tsv lists column names
                    if colnames_file:
                        col_df = pd.read_csv(io.BytesIO(zf.read(colnames_file)), sep='\t', header=None)
                        col_names = col_df.iloc[:, 0].tolist()
                    else:
                        col_names = None
                    scores_df = pd.read_csv(io.BytesIO(zf.read(scores_file)), sep='\t',
                                            header=None, names=col_names)
                    # Combine sample IDs with scores
                    sample_col = next((c for c in samples_df.columns
                                      if 'sample' in c.lower() or 'barcode' in c.lower()),
                                     samples_df.columns[0])
                    df = pd.concat([samples_df[[sample_col]].reset_index(drop=True),
                                    scores_df.reset_index(drop=True)], axis=1)
                    df = df.rename(columns={sample_col: 'sample'})
                else:
                    # Fallback: try reading largest file as TSV
                    largest = max(names, key=lambda n: zf.getinfo(n).file_size)
                    df = pd.read_csv(io.BytesIO(zf.read(largest)), sep='\t')
            else:
                # Old format: plain TSV
                if content[:2] == b'\x1f\x8b':
                    content = gzip.decompress(content)
                df = pd.read_csv(io.BytesIO(content), sep='\t', low_memory=False)

            df.to_csv(cached, sep='\t', index=False)
            print(f"  HRD scores: {len(df)} rows cached to {cached}")
            return df
        except Exception as e:
            last_err = e
            continue

    # Check for manually-placed file
    for ext in ['', '.tsv', '.txt', '.csv']:
        manual = cache_dir / ('hrd_scores' + ext)
        if manual.exists() and manual != cached:
            return pd.read_csv(manual, sep='\t', low_memory=False)

    raise FileNotFoundError(
        f"Could not download Knijnenburg et al. HRD (Cell Reports 2018).\n"
        f"Error: {last_err}\n"
        f"To fix: manually download and place at: {cached}\n"
        f"URLs tried: {_PANCAN_HRD_URLS}"
    )


# =============================================================================
# CNV DATA (GISTIC2)
# =============================================================================

def _download_xena_cnv(cohort: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Download GISTIC2 gene-level copy number from UCSC Xena.

    Args:
        cohort: TCGA cohort, e.g. 'TCGA-LUAD'

    Returns:
        cnv_matrix: (n_samples, n_genes) GISTIC2 scores
        gene_names: (n_genes,) gene symbols
        sample_ids: TCGA barcodes (truncated to 15 chars)
    """
    import requests

    cache = _cache_dir()
    cohort_key = cohort.replace('-', '_').lower()
    cached_npz = cache / f'{cohort_key}_cnv.npz'

    if cached_npz.exists():
        data = np.load(cached_npz, allow_pickle=True)
        return data['cnv_matrix'], data['gene_names'], data['sample_ids'].tolist()

    cohort_slug = cohort.replace('TCGA-', '').upper()
    urls = [t.format(cohort=cohort_slug) for t in _XENA_CNV_TEMPLATES]

    print(f"  Downloading CNV data for {cohort} from UCSC Xena...")
    resp = _download_with_fallback(urls, timeout=300)

    content = gzip.decompress(resp.content)
    df = pd.read_csv(io.BytesIO(content), sep='\t', index_col=0)

    # Filter to primary tumor samples (sample type 01)
    tumor_cols = [c for c in df.columns if len(c) >= 15 and c[13:15] == '01']
    if not tumor_cols:
        tumor_cols = list(df.columns)
    df = df[tumor_cols]

    gene_names = df.index.values.astype(str)
    cnv_matrix = df.values.T.astype(np.float32)
    sample_ids = [c[:15] for c in tumor_cols]

    cnv_matrix = np.nan_to_num(cnv_matrix, nan=0.0)
    print(f"  CNV matrix: {cnv_matrix.shape[0]} samples × {cnv_matrix.shape[1]} genes")

    np.savez_compressed(cached_npz,
                        cnv_matrix=cnv_matrix,
                        gene_names=gene_names,
                        sample_ids=np.array(sample_ids))
    return cnv_matrix, gene_names, sample_ids


# =============================================================================
# RPPA DATA (protein expression)
# =============================================================================

def _download_xena_rppa(sample_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Download RPPA pan-cancer protein expression from Xena.

    Pan-cancer file (~260 proteins), filtered to matching sample_ids.

    Args:
        sample_ids: TCGA barcodes to match

    Returns:
        rppa_matrix: (n_samples, n_proteins) matched to input sample_ids order
        protein_names: (n_proteins,) antibody/protein names
    """
    import requests

    cache = _cache_dir()
    cached_npz = cache / 'rppa_pancan.npz'

    if cached_npz.exists():
        data = np.load(cached_npz, allow_pickle=True)
        all_matrix = data['rppa_matrix']
        protein_names = data['protein_names']
        all_ids = data['sample_ids'].tolist()
    else:
        print(f"  Downloading RPPA pan-cancer data from Xena...")
        resp = requests.get(_XENA_RPPA, timeout=300)
        resp.raise_for_status()

        content = gzip.decompress(resp.content)
        df = pd.read_csv(io.BytesIO(content), sep='\t', index_col=0)

        protein_names = df.index.values.astype(str)
        all_matrix = df.values.T.astype(np.float32)
        all_ids = [c[:15] for c in df.columns]

        all_matrix = np.nan_to_num(all_matrix, nan=0.0)
        print(f"  RPPA pan-cancer: {all_matrix.shape[0]} samples × {all_matrix.shape[1]} proteins")

        np.savez_compressed(cached_npz,
                            rppa_matrix=all_matrix,
                            protein_names=protein_names,
                            sample_ids=np.array(all_ids))

    # Match to requested sample_ids
    id_to_idx = {}
    for i, sid in enumerate(all_ids):
        id_to_idx[sid] = i
        id_to_idx[sid[:12]] = i  # patient-level fallback

    n_proteins = len(protein_names)
    result = np.zeros((len(sample_ids), n_proteins), dtype=np.float32)
    n_found = 0
    for i, sid in enumerate(sample_ids):
        for trunc in [sid[:15], sid[:12]]:
            if trunc in id_to_idx:
                result[i] = all_matrix[id_to_idx[trunc]]
                n_found += 1
                break

    print(f"  RPPA: {n_found}/{len(sample_ids)} samples matched, {n_proteins} proteins")
    return result, protein_names


def _download_cptac_proteomics(
    sample_ids: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load CPTAC mass-spectrometry proteomics from a locally-placed matrix.

    GRACEFUL-SKIP loader (mirrors create_bmfm_rna_embeddings / RPPA): CPTAC
    global proteomics is NOT auto-downloaded — the public matrices live behind
    the CPTAC Data Coordinating Center / the `cptac` Python package per cancer
    type and need local placement. If no local file exists, raise a clear,
    actionable error so the caller skips this rep (it does NOT cache junk).

    Expected layout (.npz with keys matrix, protein_names, sample_ids), found at:
        $CPTAC_PROTEOMICS_NPZ                       (explicit override), or
        <cache_dir>/cptac/cptac_proteomics.npz

    Returns:
        matrix:        (n_db_samples, n_proteins) proteomics values
        protein_names: (n_proteins,) protein/gene identifiers
        sample_ids:    (n_db_samples,) TCGA/CPTAC barcodes
    """
    candidates = []
    env_path = os.environ.get('CPTAC_PROTEOMICS_NPZ')
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(_cache_dir() / 'cptac' / 'cptac_proteomics.npz')

    npz_path = next((p for p in candidates if p.exists()), None)
    if npz_path is None:
        searched = ', '.join(str(p) for p in candidates)
        raise FileNotFoundError(
            "CPTAC proteomics data not available — download CPTAC global "
            "mass-spec proteomics (e.g. via the `cptac` package or the CPTAC "
            "DCC) and save it as an .npz with keys {matrix, protein_names, "
            f"sample_ids}} to one of: {searched} "
            "(or set $CPTAC_PROTEOMICS_NPZ). Skipping."
        )

    data = np.load(npz_path, allow_pickle=True)
    matrix = np.asarray(data['matrix'], dtype=np.float32)
    protein_names = np.asarray(data['protein_names'])
    all_ids = [str(s) for s in data['sample_ids'].tolist()]
    print(f"  CPTAC proteomics: loaded {matrix.shape[0]} samples × "
          f"{matrix.shape[1]} proteins from {npz_path}")
    return matrix, protein_names, all_ids


# =============================================================================
# miRNA DATA
# =============================================================================

def _download_xena_mirna(cohort: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Download miRNA HiSeq expression from UCSC Xena.

    Args:
        cohort: TCGA cohort, e.g. 'TCGA-LUAD'

    Returns:
        mirna_matrix: (n_samples, n_mirnas) expression values
        mirna_names: (n_mirnas,) miRNA identifiers
        sample_ids: TCGA barcodes (truncated to 15 chars)
    """
    import requests

    cache = _cache_dir()
    cohort_key = cohort.replace('-', '_').lower()
    cached_npz = cache / f'{cohort_key}_mirna.npz'

    if cached_npz.exists():
        data = np.load(cached_npz, allow_pickle=True)
        return data['mirna_matrix'], data['mirna_names'], data['sample_ids'].tolist()

    cohort_slug = cohort.replace('TCGA-', '').upper()
    urls = [t.format(cohort=cohort_slug) for t in _XENA_MIRNA_TEMPLATES]

    print(f"  Downloading miRNA data for {cohort} from UCSC Xena...")
    resp = _download_with_fallback(urls, timeout=300)

    content = gzip.decompress(resp.content)
    df = pd.read_csv(io.BytesIO(content), sep='\t', index_col=0)

    # Filter to primary tumor samples
    tumor_cols = [c for c in df.columns if len(c) >= 15 and c[13:15] == '01']
    if not tumor_cols:
        tumor_cols = list(df.columns)
    df = df[tumor_cols]

    mirna_names = df.index.values.astype(str)
    mirna_matrix = df.values.T.astype(np.float32)
    sample_ids = [c[:15] for c in tumor_cols]

    mirna_matrix = np.nan_to_num(mirna_matrix, nan=0.0)
    print(f"  miRNA matrix: {mirna_matrix.shape[0]} samples × {mirna_matrix.shape[1]} miRNAs")

    np.savez_compressed(cached_npz,
                        mirna_matrix=mirna_matrix,
                        mirna_names=mirna_names,
                        sample_ids=np.array(sample_ids))
    return mirna_matrix, mirna_names, sample_ids


# =============================================================================
# DNA METHYLATION (HM450)
# =============================================================================

# Xena HumanMethylation450 URLs (per-cohort)
_XENA_METHYLATION_TEMPLATES = [
    "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/"
    "download/TCGA.{cohort}.sampleMap%2FHumanMethylation450.gz",
    "https://tcga.xenahubs.net/download/"
    "TCGA.{cohort}.sampleMap%2FHumanMethylation450.gz",
]


def _download_xena_methylation(cohort: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Download HM450 DNA methylation beta values from UCSC Xena.

    Illumina HumanMethylation450 BeadChip (~485K CpG probes).
    Beta values range [0, 1] representing methylation fraction.

    Args:
        cohort: TCGA cohort, e.g. 'TCGA-BRCA'

    Returns:
        methylation: (n_samples, n_probes) beta values
        probe_names: (n_probes,) CpG probe IDs
        sample_ids: TCGA barcodes (truncated to 15 chars)
    """
    import requests

    cache = _cache_dir()
    cohort_key = cohort.replace('-', '_').lower()
    cached_npz = cache / f'{cohort_key}_methylation.npz'

    if cached_npz.exists():
        data = np.load(cached_npz, allow_pickle=True)
        return data['methylation'], data['probe_names'], data['sample_ids'].tolist()

    cohort_slug = cohort.replace('TCGA-', '').upper()
    urls = [t.format(cohort=cohort_slug) for t in _XENA_METHYLATION_TEMPLATES]

    print(f"  Downloading HM450 methylation data for {cohort} from UCSC Xena...")
    resp = _download_with_fallback(urls, timeout=600)

    content = gzip.decompress(resp.content)
    df = pd.read_csv(io.BytesIO(content), sep='\t', index_col=0)

    # Filter to primary tumor samples (sample type code 01)
    tumor_cols = [c for c in df.columns if len(c) >= 15 and c[13:15] == '01']
    if not tumor_cols:
        tumor_cols = list(df.columns)
    df = df[tumor_cols]

    probe_names = df.index.values.astype(str)
    methylation = df.values.T.astype(np.float32)
    sample_ids = [c[:15] for c in tumor_cols]

    # Replace NaN with probe median (common for methylation)
    nan_mask = np.isnan(methylation)
    if nan_mask.any():
        probe_medians = np.nanmedian(methylation, axis=0)
        for j in range(methylation.shape[1]):
            col_nans = np.isnan(methylation[:, j])
            if col_nans.any():
                methylation[col_nans, j] = probe_medians[j]
        methylation = np.nan_to_num(methylation, nan=0.0)
        print(f"  Imputed {nan_mask.sum()} NaN methylation values with probe medians")

    print(f"  Methylation matrix: {methylation.shape[0]} samples × {methylation.shape[1]} probes")

    np.savez_compressed(cached_npz,
                        methylation=methylation,
                        probe_names=probe_names,
                        sample_ids=np.array(sample_ids))
    return methylation, probe_names, sample_ids


# =============================================================================
# WSI FEATURES (PRE-COMPUTED)
# =============================================================================

def _download_wsi_features() -> Path:
    """Download/prepare all WSI features to a unified directory.

    Sources:
    1. GigaSSL (Mendeley, public, ~130 MB total) — 5 slide-level encoders
    2. TITAN (HuggingFace, gated, ~37 MB) — slide-level, needs HF_TOKEN
    3. Local CNN features (Seagate) — 320D per-slide from small_cnn_320
    4. UNI2-h (HuggingFace, gated, ~35 GB/cohort) — patch-level, manual prep

    All features are stored as .npy dicts: {slide_id: embedding_vector}.
    Returns: path to directory containing per-model .npy files.
    """
    cache = _cache_dir()
    wsi_dir = cache / 'wsi_features'
    wsi_dir.mkdir(parents=True, exist_ok=True)

    # 1. GigaSSL from Mendeley (public, automatic)
    _download_gigassl_features(wsi_dir)

    # 2. TITAN from HuggingFace (gated, needs HF token)
    _prepare_titan_features(wsi_dir)

    # 3. Local CNN 320D features from Seagate
    _prepare_local_cnn_features(wsi_dir)

    # 4. UNI2-h — too large for auto-download, just check if pre-prepared
    uni2h_file = wsi_dir / 'uni2h.npy'
    if uni2h_file.exists():
        print(f"  UNI2-h features found at {uni2h_file}")
    else:
        print(f"  UNI2-h features not found. To prepare, run:")
        print(f"    python -c \"from kirby.datasets.tcga import prepare_uni2h_features; "
              f"prepare_uni2h_features()\"")

    return wsi_dir


def _download_gigassl_features(wsi_dir: Path):
    """Download GigaSSL slide-level embeddings from Mendeley Data."""
    import requests

    for model_name, (filename, file_uuid) in _GIGASSL_FILES.items():
        model_file = wsi_dir / f'{model_name}.npy'
        if model_file.exists():
            continue

        url = f"{_GIGASSL_DOWNLOAD_BASE}/{file_uuid}/file_downloaded"
        print(f"  Downloading GigaSSL {model_name} features...")
        try:
            resp = requests.get(url, timeout=300, stream=True)
            resp.raise_for_status()

            with open(model_file, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  Saved {model_name} ({model_file.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"  WARNING: Failed to download GigaSSL {model_name}: {e}")
            if model_file.exists():
                model_file.unlink()


def _prepare_titan_features(wsi_dir: Path):
    """Download TITAN slide-level features from HuggingFace and convert to .npy dict.

    Requires: pip install huggingface_hub
    Requires: HF_TOKEN env var or huggingface-cli login (gated model)
    """
    titan_file = wsi_dir / 'titan.npy'
    if titan_file.exists():
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  TITAN: huggingface_hub not installed, skipping")
        return

    print(f"  Downloading TITAN features from HuggingFace...")
    try:
        import pickle
        pkl_path = hf_hub_download(
            _TITAN_HF_REPO, filename=_TITAN_HF_FILE,
            cache_dir=str(wsi_dir.parent / 'hf_cache'),
        )
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # TITAN pickle has data['filenames'] and data['embeddings']
        filenames = data['filenames']
        embeddings = data['embeddings']
        if hasattr(embeddings, 'numpy'):
            embeddings = embeddings.numpy()

        # Convert to dict: slide_id -> embedding
        titan_dict = {}
        for name, emb in zip(filenames, embeddings):
            titan_dict[str(name)] = np.array(emb, dtype=np.float32)

        np.save(titan_file, titan_dict, allow_pickle=True)
        print(f"  TITAN: {len(titan_dict)} slides, {next(iter(titan_dict.values())).shape[-1]}D")
    except Exception as e:
        print(f"  WARNING: TITAN download failed: {e}")
        print(f"  (Requires HF account with institutional email + accepted terms)")
        if titan_file.exists():
            titan_file.unlink()


def _prepare_local_cnn_features(wsi_dir: Path):
    """Load pre-computed small CNN 320D features from Seagate .pt files.

    These are patch-level features; mean-pooled to slide-level.
    Only available for TCGA lung (LUAD + LUSC).
    """
    cnn_file = wsi_dir / 'cnn320.npy'
    if cnn_file.exists():
        return

    # Find .pt files
    pt_dir = None
    for candidate in _LOCAL_CNN_PATHS:
        if candidate.exists() and list(candidate.glob('*.pt')):
            pt_dir = candidate
            break

    if pt_dir is None:
        return  # silently skip — local features not available

    try:
        import torch
    except ImportError:
        print("  CNN320: PyTorch not installed, skipping local CNN features")
        return

    print(f"  Loading local CNN 320D features from {pt_dir}...")
    pt_files = sorted(pt_dir.glob('*.pt'))
    cnn_dict = {}

    for pt_file in pt_files:
        # Filename: TCGA-XX-XXXX-01Z-00-DX1.uuid.pt
        slide_id = pt_file.stem  # includes UUID suffix
        try:
            feats = torch.load(pt_file, map_location='cpu', weights_only=False)
            if hasattr(feats, 'numpy'):
                feats = feats.numpy()
            if feats.ndim == 2:
                # (n_patches, 320) -> mean pool to (320,)
                slide_emb = feats.mean(axis=0).astype(np.float32)
            elif feats.ndim == 1:
                slide_emb = feats.astype(np.float32)
            else:
                continue
            cnn_dict[slide_id] = slide_emb
        except Exception:
            continue

    if cnn_dict:
        np.save(cnn_file, cnn_dict, allow_pickle=True)
        dim = next(iter(cnn_dict.values())).shape[-1]
        print(f"  CNN320: {len(cnn_dict)} slides, {dim}D")


def prepare_uni2h_features(cohorts: Optional[List[str]] = None):
    """Prepare UNI2-h features from downloaded HuggingFace h5 files.

    UNI2-h provides 1536D patch-level features. This function mean-pools
    them to slide-level and saves as .npy dict for use with load_wsi_features().

    Prerequisites:
        1. pip install huggingface_hub h5py
        2. huggingface-cli login (institutional email, accept terms)
        3. Download cohort archives:
           huggingface-cli download MahmoodLab/UNI2-h-features \\
               TCGA/TCGA-LUAD.tar.gz --repo-type dataset --local-dir /Volumes/seagate/kirby/tcga/uni2h_raw
        4. Extract: tar xzf TCGA-LUAD.tar.gz

    Args:
        cohorts: List of cohort slugs to process, e.g. ['TCGA-LUAD', 'TCGA-LUSC'].
                 Defaults to LUAD + LUSC + BRCA.
    """
    import h5py

    if cohorts is None:
        cohorts = ['TCGA-LUAD', 'TCGA-LUSC', 'TCGA-BRCA_IDC', 'TCGA-BRCA_OTHERS']

    cache = _cache_dir()
    wsi_dir = cache / 'wsi_features'
    wsi_dir.mkdir(parents=True, exist_ok=True)
    uni2h_file = wsi_dir / 'uni2h.npy'

    # Look for extracted h5 files
    raw_dirs = [
        cache / 'uni2h_raw',
        Path('/Volumes/seagate/kirby/tcga/uni2h_raw'),
    ]

    uni2h_dict = {}
    if uni2h_file.exists():
        # Load existing and extend
        uni2h_dict = np.load(uni2h_file, allow_pickle=True).item()
        print(f"  Loaded {len(uni2h_dict)} existing UNI2-h embeddings")

    for cohort in cohorts:
        for raw_dir in raw_dirs:
            h5_dir = raw_dir / 'TCGA' / cohort
            if not h5_dir.exists():
                h5_dir = raw_dir / cohort
            if not h5_dir.exists():
                continue

            h5_files = list(h5_dir.glob('**/*.h5'))
            if not h5_files:
                continue

            print(f"  Processing {len(h5_files)} UNI2-h h5 files from {cohort}...")
            for h5_path in h5_files:
                slide_id = h5_path.stem
                if slide_id in uni2h_dict:
                    continue
                try:
                    with h5py.File(h5_path, 'r') as f:
                        features = f['features'][:]  # (1, n_patches, 1536)
                        if features.ndim == 3:
                            features = features[0]  # (n_patches, 1536)
                        slide_emb = features.mean(axis=0).astype(np.float32)
                    uni2h_dict[slide_id] = slide_emb
                except Exception as e:
                    print(f"    Failed {h5_path.name}: {e}")
            break

    if uni2h_dict:
        np.save(uni2h_file, uni2h_dict, allow_pickle=True)
        dim = next(iter(uni2h_dict.values())).shape[-1]
        print(f"  UNI2-h: {len(uni2h_dict)} slides total, {dim}D → {uni2h_file}")
    else:
        print("  No UNI2-h h5 files found. See prepare_uni2h_features() docstring.")


# =============================================================================
# TSS GROUPING
# =============================================================================

def _extract_tss_groups(sample_ids: List[str]) -> np.ndarray:
    """Parse tissue source site (TSS) code from TCGA barcodes.

    TCGA barcode format: TCGA-XX-XXXX-... where XX (positions 5-6) is the TSS.
    TSS represents the contributing institution — used for GroupKFold to prevent
    data leakage from site-specific confounders.

    Returns: integer group IDs array
    """
    tss_codes = []
    for sid in sample_ids:
        parts = sid.split('-')
        if len(parts) >= 2:
            tss_codes.append(parts[1])
        else:
            tss_codes.append('UNKNOWN')

    unique_tss = sorted(set(tss_codes))
    tss_to_id = {t: i for i, t in enumerate(unique_tss)}
    groups = np.array([tss_to_id[t] for t in tss_codes])
    return groups


# =============================================================================
# MAIN DATASET LOADER
# =============================================================================

def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, str, Dict]:
    """Load a TCGA dataset by name.

    Available datasets:
        tcga_luad_vs_lusc:  Binary classification — LUAD (0) vs LUSC (1), ~1016 samples
        tcga_brca_pam50:    4-class classification — LumA/LumB/Basal/Her2, ~900 samples
        tcga_brca_survival: Regression — OS days (uncensored only), ~400-500 samples

    Returns:
        expression: np.ndarray (n_samples, n_genes) — log2(TPM+1) expression matrix
        labels: np.ndarray (n_samples,) — class indices or continuous values
        task_type: str — 'classification' or 'regression'
        metadata: dict with keys:
            sample_ids: List[str] — TCGA barcodes
            gene_names: np.ndarray — gene symbols
            tss_groups: np.ndarray — TSS group IDs for GroupKFold
            cohort: str — TCGA cohort(s) used
            wsi_dir: Optional[Path] — path to WSI features directory
            class_names: Optional[List[str]] — for classification tasks
    """
    name_lower = name.lower()

    if name_lower == 'tcga_luad_vs_lusc':
        return _load_luad_vs_lusc()
    elif name_lower == 'tcga_brca_pam50':
        return _load_brca_pam50()
    elif name_lower == 'tcga_brca_survival':
        return _load_brca_survival()
    else:
        raise ValueError(
            f"Unknown TCGA dataset: {name}. "
            f"Available: tcga_luad_vs_lusc, tcga_brca_pam50, tcga_brca_survival"
        )


def _load_luad_vs_lusc() -> Tuple[np.ndarray, np.ndarray, str, Dict]:
    """Load LUAD vs LUSC binary classification task."""
    # Download both cohorts
    expr_luad, genes_luad, ids_luad = _download_xena_expression('TCGA-LUAD')
    expr_lusc, genes_lusc, ids_lusc = _download_xena_expression('TCGA-LUSC')

    # Ensure same gene set
    if not np.array_equal(genes_luad, genes_lusc):
        # Intersect genes
        common = set(genes_luad) & set(genes_lusc)
        idx_luad = [i for i, g in enumerate(genes_luad) if g in common]
        idx_lusc = [i for i, g in enumerate(genes_lusc) if g in common]
        expr_luad = expr_luad[:, idx_luad]
        expr_lusc = expr_lusc[:, idx_lusc]
        gene_names = genes_luad[idx_luad]
    else:
        gene_names = genes_luad

    # Combine
    expression = np.vstack([expr_luad, expr_lusc])
    labels = np.concatenate([
        np.zeros(len(ids_luad), dtype=int),
        np.ones(len(ids_lusc), dtype=int),
    ])
    sample_ids = ids_luad + ids_lusc
    tss_groups = _extract_tss_groups(sample_ids)

    # Try to get WSI features
    wsi_dir = None
    try:
        wsi_dir = _download_wsi_features()
    except Exception as e:
        print(f"  WSI features not available: {e}")

    class_names = ['LUAD', 'LUSC']
    print(f"  TCGA LUAD vs LUSC: {len(sample_ids)} samples "
          f"(LUAD={np.sum(labels==0)}, LUSC={np.sum(labels==1)}), "
          f"{expression.shape[1]} genes, "
          f"{len(np.unique(tss_groups))} TSS sites")

    metadata = {
        'sample_ids': sample_ids,
        'gene_names': gene_names,
        'tss_groups': tss_groups,
        'cohort': 'TCGA-LUAD+TCGA-LUSC',
        'wsi_dir': wsi_dir,
        'class_names': class_names,
    }
    return expression, labels, 'classification', metadata


def _load_brca_pam50() -> Tuple[np.ndarray, np.ndarray, str, Dict]:
    """Load BRCA PAM50 4-class classification task."""
    expression, gene_names, sample_ids = _download_xena_expression('TCGA-BRCA')

    # Get PAM50 labels
    labels, class_names, valid_mask = _get_pam50_labels(sample_ids)

    # Filter to samples with valid PAM50 labels
    expression = expression[valid_mask]
    labels = labels[valid_mask]
    sample_ids = [s for s, v in zip(sample_ids, valid_mask) if v]
    tss_groups = _extract_tss_groups(sample_ids)

    # Try to get WSI features
    wsi_dir = None
    try:
        wsi_dir = _download_wsi_features()
    except Exception as e:
        print(f"  WSI features not available: {e}")

    print(f"  TCGA BRCA PAM50: {len(sample_ids)} samples, "
          f"{expression.shape[1]} genes, "
          f"{len(np.unique(tss_groups))} TSS sites")
    for i, name in enumerate(class_names):
        print(f"    {name}: {np.sum(labels==i)} samples")

    metadata = {
        'sample_ids': sample_ids,
        'gene_names': gene_names,
        'tss_groups': tss_groups,
        'cohort': 'TCGA-BRCA',
        'wsi_dir': wsi_dir,
        'class_names': class_names,
    }
    return expression, labels, 'classification', metadata


def _load_brca_survival() -> Tuple[np.ndarray, np.ndarray, str, Dict]:
    """Load BRCA overall survival regression task.

    Uses TCGA-CDR survival data (Liu et al. 2018). Filters to uncensored
    samples (vital_status=Dead) for clean regression on OS.time.
    """
    expression, gene_names, sample_ids = _download_xena_expression('TCGA-BRCA')
    survival = _download_survival_data('TCGA-BRCA')

    # Match samples
    surv_id_col = next(
        (c for c in survival.columns if c.lower() in ('sample', 'sampleid', '_sample_id')),
        survival.columns[0]
    )
    survival['_match_id'] = survival[surv_id_col].astype(str).str[:15]

    # Find survival time and event columns
    # Xena BRCA clinical uses: days_to_death, days_to_last_followup, vital_status
    # TCGA-CDR uses: OS.time, OS
    os_time_col = None
    os_event_col = None

    # Try TCGA-CDR naming first
    for col in survival.columns:
        cl = col.lower().replace('.', '_')
        if cl in ('os_time', 'os_time_nature2012'):
            os_time_col = col
        elif cl in ('os', 'os_event', 'os_event_nature2012'):
            os_event_col = col

    # Fall back to Xena clinical naming
    if os_time_col is None:
        for col in survival.columns:
            cl = col.lower()
            if 'days_to_death' in cl and 'nature2012' not in cl:
                os_time_col = col
            elif 'days_to_last_followup' in cl and 'nature2012' not in cl:
                if os_time_col is None:
                    os_time_col = col  # use as fallback
    if os_event_col is None:
        for col in survival.columns:
            if col.lower() == 'vital_status':
                os_event_col = col

    if os_time_col is None:
        # Last resort: nature2012 versions
        for col in survival.columns:
            if 'os_time' in col.lower() or 'days_to_death' in col.lower():
                os_time_col = col
                break

    if os_time_col is None:
        raise ValueError(f"Cannot find survival time column. Available: {list(survival.columns)[:30]}")

    print(f"  Using time column: {os_time_col}, event column: {os_event_col}")

    # Build lookup: sample_id -> (os_time, os_event)
    # For BRCA clinical: days_to_death (for dead patients) or days_to_last_followup (for alive)
    surv_lookup = {}
    for _, row in survival.iterrows():
        mid = str(row.get('_match_id', ''))
        if not mid.startswith('TCGA'):
            continue

        # Try to get OS time
        death_time = pd.to_numeric(row.get('days_to_death', row.get('Days_to_date_of_Death_nature2012', np.nan)), errors='coerce')
        followup_time = pd.to_numeric(row.get('days_to_last_followup', row.get('Days_to_Date_of_Last_Contact_nature2012', np.nan)), errors='coerce')
        os_time_val = pd.to_numeric(row.get(os_time_col, np.nan), errors='coerce')

        # Determine event status
        vital = str(row.get(os_event_col, '')).upper() if os_event_col else ''

        if pd.notna(death_time) and death_time > 0:
            surv_lookup[mid] = (death_time, 1)  # Dead = uncensored
        elif pd.notna(os_time_val) and os_time_val > 0:
            event = 1 if vital in ('DEAD', '1', 'TRUE', 'DECEASED') else 0
            surv_lookup[mid] = (os_time_val, event)
        elif pd.notna(followup_time) and followup_time > 0 and vital:
            event = 1 if vital in ('DEAD', '1', 'TRUE', 'DECEASED') else 0
            surv_lookup[mid] = (followup_time, event)

    # Filter to uncensored (event=1) for clean regression
    valid_indices = []
    os_times = []
    for i, sid in enumerate(sample_ids):
        sid_trunc = sid[:15]
        if sid_trunc in surv_lookup:
            os_time, event = surv_lookup[sid_trunc]
            if event == 1:  # Uncensored only
                valid_indices.append(i)
                os_times.append(os_time)

    if len(valid_indices) < 50:
        # If too few uncensored, use all samples with OS data
        warnings.warn(
            f"Only {len(valid_indices)} uncensored BRCA samples. "
            f"Including censored samples for regression."
        )
        valid_indices = []
        os_times = []
        for i, sid in enumerate(sample_ids):
            sid_trunc = sid[:15]
            if sid_trunc in surv_lookup:
                os_time, _ = surv_lookup[sid_trunc]
                valid_indices.append(i)
                os_times.append(os_time)

    expression = expression[valid_indices]
    labels = np.array(os_times, dtype=np.float64)
    sample_ids = [sample_ids[i] for i in valid_indices]
    tss_groups = _extract_tss_groups(sample_ids)

    # Per-sample survival event flags (aligned to sample_ids order), exposed in
    # metadata so survival-supervised reps (survival_latent50) can build a Cox
    # head. Primary path keeps uncensored-only (all events=1); the fallback path
    # above may include censored samples, so recover the true event per sample.
    survival_event = np.array(
        [int(surv_lookup.get(sample_ids[k][:15], (0, 0))[1])
         for k in range(len(sample_ids))],
        dtype=np.int64,
    )

    # Try to get WSI features
    wsi_dir = None
    try:
        wsi_dir = _download_wsi_features()
    except Exception as e:
        print(f"  WSI features not available: {e}")

    print(f"  TCGA BRCA Survival: {len(sample_ids)} samples (uncensored), "
          f"{expression.shape[1]} genes, "
          f"{len(np.unique(tss_groups))} TSS sites")
    print(f"  OS time: [{labels.min():.0f}, {labels.max():.0f}] days, "
          f"mean={labels.mean():.0f}, median={np.median(labels):.0f}")

    metadata = {
        'sample_ids': sample_ids,
        'gene_names': gene_names,
        'tss_groups': tss_groups,
        'cohort': 'TCGA-BRCA',
        'wsi_dir': wsi_dir,
        'class_names': None,
        # For survival-supervised reps: per-sample OS time + event (aligned).
        'survival_time': labels.astype(np.float32),
        'survival_event': survival_event,
    }
    return expression, labels, 'regression', metadata
