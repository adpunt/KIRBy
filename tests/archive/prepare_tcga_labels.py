#!/usr/bin/env python3
"""
Create TCGA Lung Labels from separate LUAD/LUSC manifests
"""
import numpy as np
import pandas as pd

BASE_DIR = "/Volumes/seagate/tcga_lung"
DATA_DIR = "data"

def main():
    print("="*70)
    print("TCGA LUNG LABEL CREATION")
    print("="*70)
    
    # Load sample IDs (filenames)
    sample_ids = np.load(f"{BASE_DIR}/sample_ids.npy")
    print(f"\nLoaded {len(sample_ids)} samples")
    print(f"Example: {sample_ids[0]}")
    
    # Load manifests
    print("\nLoading manifests...")
    luad_manifest = pd.read_csv(f"{DATA_DIR}/gdc_manifest_luad_star.txt", sep='\t')
    lusc_manifest = pd.read_csv(f"{DATA_DIR}/gdc_manifest_lusc_star.txt", sep='\t')
    
    print(f"  LUAD manifest: {len(luad_manifest)} files")
    print(f"  LUSC manifest: {len(lusc_manifest)} files")
    
    # Create filename sets
    luad_files = set(luad_manifest['filename'].values)
    lusc_files = set(lusc_manifest['filename'].values)
    
    # Match sample IDs to labels
    labels = []
    luad_count = 0
    lusc_count = 0
    unknown_count = 0
    
    for sample_id in sample_ids:
        if sample_id in luad_files:
            labels.append(0)
            luad_count += 1
        elif sample_id in lusc_files:
            labels.append(1)
            lusc_count += 1
        else:
            unknown_count += 1
    
    if unknown_count > 0:
        print(f"\nWARNING: {unknown_count}/{len(sample_ids)} files not in STAR manifests")
        print("(These might be from mutation files, will be excluded)")
    
    if len(labels) == 0:
        print("\nERROR: No samples matched!")
        return
    
    labels = np.array(labels)
    
    print(f"\nLabel distribution:")
    print(f"  LUAD (0): {luad_count}")
    print(f"  LUSC (1): {lusc_count}")
    print(f"  Total: {len(labels)}")
    
    # Save
    np.save(f"{BASE_DIR}/labels.npy", labels)
    print(f"\n✓ Saved to {BASE_DIR}/labels.npy")


if __name__ == '__main__':
    main()