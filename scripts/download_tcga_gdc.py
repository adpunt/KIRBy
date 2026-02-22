#!/usr/bin/env python3
"""
Download TCGA expression data from GDC API (alternative to Xena).

Standalone script — not imported by any KIRBy module.
Downloads STAR-Counts expression files from GDC, extracts TPM values,
and saves as .npz format compatible with kirby.datasets.tcga cache.

Usage:
    python scripts/download_tcga_gdc.py TCGA-BRCA
    python scripts/download_tcga_gdc.py TCGA-LUAD
    python scripts/download_tcga_gdc.py TCGA-LUSC

Saves to the same cache directory as the Xena downloader, so the loader
picks it up transparently. Requires: requests, pandas, numpy.
"""

import argparse
import glob
import gzip
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

GDC_FILES = "https://api.gdc.cancer.gov/files"
GDC_CASES = "https://api.gdc.cancer.gov/cases"


def _cache_dir() -> Path:
    """Get cache directory, matching kirby.datasets.tcga._cache_dir()."""
    seagate = Path('/Volumes/seagate/kirby/tcga')
    if seagate.parent.parent.exists():
        seagate.mkdir(parents=True, exist_ok=True)
        return seagate
    cache = Path.home() / '.cache' / 'kirby' / 'tcga'
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def get_rnaseq_manifest(project: str) -> pd.DataFrame:
    """Query GDC API for RNA-seq STAR-Counts file manifest."""
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": project}},
            {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
            {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}},
            {"op": "=", "content": {"field": "data_format", "value": "TSV"}},
        ]
    }
    fields = [
        "file_id", "file_name", "cases.case_id",
        "cases.samples.sample_id", "cases.samples.sample_type",
        "cases.submitter_id",
    ]

    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "size": "5000",
        "format": "JSON",
    }

    print(f"Querying GDC for {project} RNA-seq files...")
    resp = requests.get(GDC_FILES, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hits = data.get("data", {}).get("hits", [])
    if not hits:
        raise RuntimeError(f"No RNA-seq files found for {project}")

    records = []
    for hit in hits:
        file_id = hit["file_id"]
        file_name = hit["file_name"]
        cases = hit.get("cases", [{}])
        case_id = cases[0].get("submitter_id", "") if cases else ""

        # Get sample type (Primary Tumor = 01)
        samples = cases[0].get("samples", [{}]) if cases else [{}]
        sample_type = samples[0].get("sample_type", "") if samples else ""
        sample_id = samples[0].get("sample_id", "") if samples else ""

        records.append({
            "file_id": file_id,
            "file_name": file_name,
            "case_id": case_id,
            "sample_id": sample_id,
            "sample_type": sample_type,
        })

    df = pd.DataFrame(records)
    # Filter to primary tumor
    tumor_mask = df['sample_type'].str.contains('Primary Tumor', case=False, na=False)
    if tumor_mask.sum() > 0:
        df = df[tumor_mask]

    print(f"  Found {len(df)} primary tumor RNA-seq files")
    return df


def download_and_build_matrix(manifest: pd.DataFrame, project: str):
    """Download expression files from GDC and build matrix."""
    cache = _cache_dir()
    raw_dir = cache / 'gdc_raw' / project.replace('-', '_').lower()
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(manifest)} expression files...")

    all_data = {}
    gene_names = None

    for idx, row in manifest.iterrows():
        file_id = row['file_id']
        case_id = row['case_id']

        # Check if already downloaded
        local_files = list(raw_dir.glob(f"{file_id}/*.tsv"))
        if not local_files:
            # Download via GDC API
            url = f"https://api.gdc.cancer.gov/data/{file_id}"
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()

                # Save and extract
                file_dir = raw_dir / file_id
                file_dir.mkdir(exist_ok=True)

                # Response may be gzipped tar
                if resp.headers.get('Content-Type', '').startswith('application/x-tar'):
                    import tarfile
                    with tarfile.open(fileobj=io.BytesIO(resp.content)) as tf:
                        tf.extractall(file_dir)
                    local_files = list(file_dir.glob('**/*.tsv'))
                else:
                    out_file = file_dir / row['file_name']
                    out_file.write_bytes(resp.content)
                    local_files = [out_file]

                if idx % 50 == 0:
                    print(f"  Downloaded {idx+1}/{len(manifest)}...")
            except Exception as e:
                print(f"  Failed to download {file_id}: {e}")
                continue

        if not local_files:
            continue

        # Parse expression file
        tsv_path = local_files[0]
        try:
            df = pd.read_csv(tsv_path, sep='\t', comment='#', skiprows=1)
            # Filter to protein-coding genes
            if 'gene_type' in df.columns:
                df = df[df['gene_type'] == 'protein_coding']

            # Use TPM values
            tpm_col = next((c for c in df.columns if 'tpm' in c.lower()), None)
            if tpm_col is None:
                continue

            gene_col = next((c for c in df.columns if c.lower() == 'gene_name'), df.columns[0])
            values = df.set_index(gene_col)[tpm_col].values.astype(np.float64)

            if gene_names is None:
                gene_names = df[gene_col].values

            all_data[case_id] = values
        except Exception as e:
            print(f"  Failed to parse {tsv_path}: {e}")

    if not all_data:
        raise RuntimeError("No expression data could be parsed!")

    # Build matrix
    sample_ids = list(all_data.keys())
    expression = np.array([all_data[s] for s in sample_ids])

    # log2(TPM + 1) transformation
    expression = np.log2(expression + 1).astype(np.float32)

    print(f"Expression matrix: {expression.shape[0]} samples x {expression.shape[1]} genes")

    # Save in same format as Xena downloader
    cohort_key = project.replace('-', '_').lower()
    npz_path = cache / f'{cohort_key}_expression.npz'
    np.savez_compressed(
        npz_path,
        expression=expression,
        gene_names=gene_names,
        sample_ids=np.array(sample_ids),
    )
    print(f"Saved to {npz_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Download TCGA expression data from GDC API'
    )
    parser.add_argument('project', type=str,
                       help='TCGA project ID (e.g., TCGA-BRCA, TCGA-LUAD, TCGA-LUSC)')
    args = parser.parse_args()

    manifest = get_rnaseq_manifest(args.project)
    download_and_build_matrix(manifest, args.project)

    print("Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
