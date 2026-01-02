#!/bin/bash
set -e  # Exit on any error

# Usage: bash download_tcga_features.sh [base_directory]
# Default: /Volumes/seagate/tcga_lung_wsi_features

BASE="${1:-/Volumes/seagate/tcga_lung_wsi_features}"

echo "TCGA Feature Download"
echo "====================="
echo "Target directory: $BASE"
echo ""

# Check parent directory exists
PARENT=$(dirname "$BASE")
if [ ! -d "$PARENT" ]; then
    echo "ERROR: Parent directory does not exist: $PARENT"
    exit 1
fi

# Install gdown if needed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown --break-system-packages
fi

# Setup directory
if [ -d "$BASE" ]; then
    echo "Removing old directory..."
    rm -rf "$BASE"
fi
mkdir -p "$BASE"
cd "$BASE"

# Download 320-dim
echo ""
echo "Downloading 320-dim features..."
mkdir small_cnn_320
cd small_cnn_320
wget -q --show-progress https://www.dropbox.com/s/5wuvu791vwntg9o/tcga_lung_splits.csv
wget -q --show-progress https://www.dropbox.com/s/euepd2owxvuwr7v/feats_pt.zip
unzip -q feats_pt.zip
rm feats_pt.zip
cd ..

# Download 512-dim
echo ""
echo "Downloading 512-dim features..."
mkdir resnet18_simclr_512
cd resnet18_simclr_512
gdown --folder https://drive.google.com/drive/folders/1Rn_VpgM82VEfnjiVjDbObbBFHvs0V1OE --remaining-ok

# Move files out of nested "mil tcga lung" folder if it exists
if [ -d "mil tcga lung" ]; then
    echo "Moving files from nested folder..."
    mv "mil tcga lung"/* .
    rmdir "mil tcga lung"
fi
cd ..

echo ""
echo "Done. Files in: $BASE"
echo "320-dim: $(ls small_cnn_320/feats_pt/*.pt 2>/dev/null | wc -l) .pt files"
echo "512-dim: $(ls resnet18_simclr_512/*.csv 2>/dev/null | wc -l) .csv files"