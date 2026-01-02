"""
TCGA Dataset Loaders

Load TCGA cancer genomics and histopathology data for:
- TCGA-LUNG: LUAD vs LUSC binary classification (~1000 samples)
- TCGA-BRCA: PAM50 4-way classification (~1100 samples)

Data should be pre-downloaded and preprocessed.
See DATA_DOWNLOAD.md for instructions.
"""

import numpy as np
import os


def load_tcga_lung(data_dir='/path/to/tcga/lung'):
    """
    Load TCGA LUAD vs LUSC data
    
    Expected data structure:
    data_dir/
        rna_seq.npy         # (n_samples, ~20000) gene expression
        gene_names.npy      # (20000,) gene names
        labels.npy          # (n_samples,) binary labels (0=LUAD, 1=LUSC)
        sample_ids.npy      # (n_samples,) TCGA sample IDs
        wsi_features/       # Pre-computed imaging features
            resnet50.npy
            uni.npy
            conch.npy
    
    Args:
        data_dir: Path to TCGA lung data directory
    
    Returns:
        data: Dictionary with 'rna', 'genes', 'labels', 'sample_ids', 'wsi_dir'
    """
    # Check if directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Please download TCGA data first. See DATA_DOWNLOAD.md"
        )
    
    # Load RNA-seq data
    rna_path = os.path.join(data_dir, 'rna_seq.npy')
    if not os.path.exists(rna_path):
        raise FileNotFoundError(f"RNA-seq data not found: {rna_path}")
    
    X_rna = np.load(rna_path)
    gene_names = np.load(os.path.join(data_dir, 'gene_names.npy'))
    y = np.load(os.path.join(data_dir, 'labels.npy'))
    sample_ids = np.load(os.path.join(data_dir, 'sample_ids.npy'))
    
    data = {
        'rna': X_rna,
        'genes': gene_names,
        'labels': y,
        'sample_ids': sample_ids,
        'wsi_dir': os.path.join(data_dir, 'wsi_features')
    }
    
    print(f"Loaded TCGA-LUNG: {X_rna.shape[0]} samples, {X_rna.shape[1]} genes")
    print(f"Class distribution: LUAD={np.sum(y==0)}, LUSC={np.sum(y==1)}")
    
    return data


def load_tcga_breast(data_dir='/path/to/tcga/breast'):
    """
    Load TCGA BRCA PAM50 data
    
    Expected data structure:
    data_dir/
        rna_seq.npy         # (n_samples, ~20000) gene expression
        gene_names.npy      # (20000,) gene names
        pam50_labels.npy    # (n_samples,) 4-class labels (0=LumA, 1=LumB, 2=Basal, 3=Her2)
        sample_ids.npy      # (n_samples,) TCGA sample IDs
        wsi_features/       # Pre-computed imaging features
            resnet50.npy
            uni.npy
            conch.npy
    
    Args:
        data_dir: Path to TCGA breast data directory
    
    Returns:
        data: Dictionary with 'rna', 'genes', 'labels', 'sample_ids', 'wsi_dir'
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Please download TCGA data first. See DATA_DOWNLOAD.md"
        )
    
    X_rna = np.load(os.path.join(data_dir, 'rna_seq.npy'))
    gene_names = np.load(os.path.join(data_dir, 'gene_names.npy'))
    y = np.load(os.path.join(data_dir, 'pam50_labels.npy'))
    sample_ids = np.load(os.path.join(data_dir, 'sample_ids.npy'))
    
    data = {
        'rna': X_rna,
        'genes': gene_names,
        'labels': y,
        'sample_ids': sample_ids,
        'wsi_dir': os.path.join(data_dir, 'wsi_features')
    }
    
    class_names = ['LumA', 'LumB', 'Basal', 'Her2']
    print(f"Loaded TCGA-BRCA: {X_rna.shape[0]} samples, {X_rna.shape[1]} genes")
    for i, name in enumerate(class_names):
        print(f"  {name}: {np.sum(y==i)} samples")
    
    return data