from pathlib import Path
import numpy as np

# Sample IDs
sample_ids = np.load("/Volumes/seagate/tcga_lung/sample_ids.npy", allow_pickle=True)
print("Sample IDs (first 3):")
for sid in sample_ids[:3]:
    print(f"  '{sid}'")

# MAF files
maf_files = list(Path("/Volumes/seagate/tcga_lung_raw").glob("*.maf*"))
print(f"\nMAF files (first 3):")
for f in maf_files[:3]:
    print(f"  '{f.name}'")
