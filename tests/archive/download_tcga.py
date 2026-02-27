#!/usr/bin/env python3
"""
TCGA Data Download and Preprocessing Helper

Usage:
    python download_tcga.py lung    # Downloads and processes TCGA-LUNG
    python download_tcga.py breast  # Downloads and processes TCGA-BRCA
    python download_tcga.py all     # Downloads and processes both

Requires manifest files in data/:
    data/gdc_manifest_lung.txt
    data/gdc_manifest_breast.txt
"""

import requests
import os
import sys
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Base directory for all downloads (external drive)
BASE_DIR = "/Volumes/seagate"


def download_from_manifest(manifest_file, output_dir):
    """Download files listed in GDC manifest with retry logic."""
    print(f"Reading manifest: {manifest_file}")
    
    with open(manifest_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
    
    print(f"Found {len(lines)} files to download")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup session with retries
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    for i, line in enumerate(lines, 1):
        uuid = line.split('\t')[0]
        filename = line.split('\t')[1]  # Use filename from manifest
        filepath = os.path.join(output_dir, filename)
        
        # Skip if exists
        if os.path.exists(filepath):
            print(f"  [{i}/{len(lines)}] ✓ {filename}")
            continue
        
        # Download with retry
        url = f'https://api.gdc.cancer.gov/data/{uuid}'
        try:
            response = session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
            
            print(f"  [{i}/{len(lines)}] Downloaded {filename}")
            time.sleep(0.5)  # Small delay to avoid rate limiting
            
        except Exception as e:
            print(f"  [{i}/{len(lines)}] ERROR {filename}: {e}")
            continue
    
    print(f"✓ Download complete")


def preprocess_lung(raw_dir, output_dir):
    """Process TCGA-LUNG files into matrices."""
    print(f"\nPreprocessing LUNG data...")
    
    # Find all RNA-seq files
    all_files = glob.glob(f"{raw_dir}/*.tsv")
    print(f"Found {len(all_files)} files to process")
    
    if not all_files:
        print("ERROR: No TSV files found")
        return
    
    # Read first file to get gene names
    df_first = pd.read_csv(all_files[0], sep='\t', comment='#')
    gene_names = df_first['gene_name'].values
    
    # Build matrix
    expressions = []
    sample_ids = []
    
    for filepath in all_files:
        df = pd.read_csv(filepath, sep='\t', comment='#')
        expressions.append(df['unstranded'].values)
        sample_ids.append(os.path.basename(filepath))
    
    X = np.array(expressions, dtype=np.float32)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    np.save(f'{output_dir}/rna_seq.npy', X)
    np.save(f'{output_dir}/gene_names.npy', gene_names)
    np.save(f'{output_dir}/sample_ids.npy', np.array(sample_ids))
    
    print(f"✓ Saved to {output_dir}")
    print(f"  Shape: {X.shape}")
    print(f"  Samples: {len(sample_ids)}")
    print(f"  Genes: {len(gene_names)}")
    
    print("\nWARNING: Labels not created - you need clinical data to separate LUAD/LUSC")


def preprocess_breast(raw_dir, output_dir):
    """Process TCGA-BRCA files into matrices."""
    print(f"\nPreprocessing BREAST data...")
    
    all_files = glob.glob(f"{raw_dir}/*.tsv")
    print(f"Found {len(all_files)} files to process")
    
    if not all_files:
        print("ERROR: No TSV files found")
        return
    
    # Read first file to get gene names
    df_first = pd.read_csv(all_files[0], sep='\t', comment='#')
    gene_names = df_first['gene_name'].values
    
    # Build matrix
    expressions = []
    sample_ids = []
    
    for filepath in all_files:
        df = pd.read_csv(filepath, sep='\t', comment='#')
        expressions.append(df['unstranded'].values)
        sample_ids.append(os.path.basename(filepath))
    
    X = np.array(expressions, dtype=np.float32)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    np.save(f'{output_dir}/rna_seq.npy', X)
    np.save(f'{output_dir}/gene_names.npy', gene_names)
    np.save(f'{output_dir}/sample_ids.npy', np.array(sample_ids))
    
    print(f"✓ Saved to {output_dir}")
    print(f"  Shape: {X.shape}")
    print(f"  Samples: {len(sample_ids)}")
    print(f"  Genes: {len(gene_names)}")
    
    print("\nWARNING: PAM50 labels not created - you need clinical data")

# Add this function to prepare_tcga_labels.py
def get_uuid_to_project_mapping(uuids):
    """Query GDC API to get project for each UUID"""
    import requests
    
    mapping = {}
    batch_size = 100
    
    for i in range(0, len(uuids), batch_size):
        batch = uuids[i:i+batch_size]
        
        filters = {
            "op": "in",
            "content": {
                "field": "file_id",
                "value": batch
            }
        }
        
        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,cases.project.project_id",
            "size": batch_size
        }
        
        response = requests.post(
            "https://api.gdc.cancer.gov/files",
            headers={"Content-Type": "application/json"},
            json=params,
            timeout=30
        )
        
        for hit in response.json()['data']['hits']:
            file_id = hit['file_id']
            project_id = hit['cases'][0]['project']['project_id']
            mapping[file_id] = 0 if 'LUAD' in project_id else 1
        
        print(f"  Mapped {len(mapping)}/{len(uuids)} files...")
    
    return mapping

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    dataset = sys.argv[1].lower()
    
    # Check if external drive is mounted
    if not os.path.exists(BASE_DIR):
        print(f"ERROR: External drive not found at {BASE_DIR}")
        print("Please mount your external drive and try again")
        sys.exit(1)
    
    if dataset in ['lung', 'all']:
        print("="*70)
        print("DOWNLOADING TCGA-LUNG")
        print("="*70)
        download_from_manifest('data/gdc_manifest_lung.txt', f'{BASE_DIR}/tcga_lung_raw')
        preprocess_lung(f'{BASE_DIR}/tcga_lung_raw', f'{BASE_DIR}/tcga_lung')
    
    if dataset in ['breast', 'all']:
        print("\n" + "="*70)
        print("DOWNLOADING TCGA-BREAST")
        print("="*70)
        download_from_manifest('data/gdc_manifest_breast.txt', f'{BASE_DIR}/tcga_breast_raw')
        preprocess_breast(f'{BASE_DIR}/tcga_breast_raw', f'{BASE_DIR}/tcga_breast')
    
    print("\n" + "="*70)
    print("DONE")
    print(f"All data saved to: {BASE_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()