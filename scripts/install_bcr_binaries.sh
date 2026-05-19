#!/bin/bash
# One-time install of the non-pip binaries needed by the BCR-seq assembly
# pipelines (T-Ab.2): sra-tools, igblast, vsearch, seqkit, entrez-direct.
#
# Pattern matches the MINT / MolE precedent in setup.sh: run once per
# machine, not on every SLURM job. setup.sh detects $REPO/bin/ and prepends
# it to PATH if present, so once this script has run nothing else needs to
# change. No conda. No module load beyond what setup.sh already does.
#
# Pinned versions — bump only after verifying the static Linux x86-64
# tarball still ships for each.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$REPO/bin"
mkdir -p "$BIN_DIR"
cd "$BIN_DIR"

SRA_VER="3.1.1"
IGBLAST_VER="1.22.0"
VSEARCH_VER="2.30.0"
SEQKIT_VER="2.8.2"

echo "Installing BCR assembly binaries into $BIN_DIR"
echo "  sra-tools $SRA_VER, igblast $IGBLAST_VER, vsearch $VSEARCH_VER,"
echo "  seqkit $SEQKIT_VER, entrez-direct (latest)"
echo ""

# ---------------------------------------------------------------------------
# 1. sra-tools — NCBI static tarball, all binaries we need (prefetch,
#    fasterq-dump) live under sratoolkit.<ver>-ubuntu64/bin/
# ---------------------------------------------------------------------------
if [ ! -x "$BIN_DIR/prefetch" ]; then
    echo "[1/5] sra-tools $SRA_VER"
    TAR="sratoolkit.$SRA_VER-ubuntu64.tar.gz"
    wget -q "https://ftp.ncbi.nlm.nih.gov/sra/sdk/$SRA_VER/$TAR"
    tar xzf "$TAR"
    rm "$TAR"
    SRC="$BIN_DIR/sratoolkit.$SRA_VER-ubuntu64/bin"
    for b in "$SRC"/*; do
        [ -f "$b" ] && [ -x "$b" ] && ln -sf "$b" "$BIN_DIR/$(basename "$b")"
    done
else
    echo "[1/5] sra-tools already installed (skipped)"
fi

# ---------------------------------------------------------------------------
# 2. igblast — NCBI static tarball. Bundles makeblastdb too.
#    IGDATA must point at the IgBLAST install root so internal_data/ and
#    optional_file/ resolve correctly at runtime. We drop a .igdata file
#    next to the binaries that setup.sh reads.
# ---------------------------------------------------------------------------
if [ ! -x "$BIN_DIR/igblastn" ]; then
    echo "[2/5] ncbi-igblast $IGBLAST_VER"
    TAR="ncbi-igblast-$IGBLAST_VER-x64-linux.tar.gz"
    wget -q "https://ftp.ncbi.nih.gov/blast/executables/igblast/release/$IGBLAST_VER/$TAR"
    tar xzf "$TAR"
    rm "$TAR"
    IGB_ROOT="$BIN_DIR/ncbi-igblast-$IGBLAST_VER"
    for b in "$IGB_ROOT/bin"/*; do
        [ -f "$b" ] && [ -x "$b" ] && ln -sf "$b" "$BIN_DIR/$(basename "$b")"
    done
    # setup.sh reads this to set $IGDATA per job
    echo "$IGB_ROOT" > "$BIN_DIR/.igdata"
else
    echo "[2/5] igblast already installed (skipped)"
fi

# ---------------------------------------------------------------------------
# 3. vsearch — GitHub static binary release
# ---------------------------------------------------------------------------
if [ ! -x "$BIN_DIR/vsearch" ]; then
    echo "[3/5] vsearch $VSEARCH_VER"
    TAR="vsearch-$VSEARCH_VER-linux-x86_64.tar.gz"
    wget -q "https://github.com/torognes/vsearch/releases/download/v$VSEARCH_VER/$TAR"
    tar xzf "$TAR"
    rm "$TAR"
    ln -sf "$BIN_DIR/vsearch-$VSEARCH_VER-linux-x86_64/bin/vsearch" "$BIN_DIR/vsearch"
else
    echo "[3/5] vsearch already installed (skipped)"
fi

# ---------------------------------------------------------------------------
# 4. seqkit — GitHub static binary release (single binary, no subdir)
# ---------------------------------------------------------------------------
if [ ! -x "$BIN_DIR/seqkit" ]; then
    echo "[4/5] seqkit $SEQKIT_VER"
    TAR="seqkit_linux_amd64.tar.gz"
    wget -q "https://github.com/shenwei356/seqkit/releases/download/v$SEQKIT_VER/$TAR"
    tar xzf "$TAR"
    rm "$TAR"
    chmod +x "$BIN_DIR/seqkit"
else
    echo "[4/5] seqkit already installed (skipped)"
fi

# ---------------------------------------------------------------------------
# 5. entrez-direct — NCBI's own installer drops esearch/efetch/etc into
#    a subdir we point at. NCBI ships a small bootstrap that fetches the
#    Perl-based tools without touching ~/.bashrc when invoked manually.
# ---------------------------------------------------------------------------
if [ ! -x "$BIN_DIR/esearch" ]; then
    echo "[5/5] entrez-direct"
    TAR="edirect.tar.gz"
    wget -q "https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/$TAR"
    tar xzf "$TAR"
    rm "$TAR"
    # The tarball unpacks to ./edirect/ with esearch, efetch, xtract, etc.
    if [ -d "$BIN_DIR/edirect" ]; then
        for b in "$BIN_DIR/edirect"/*; do
            [ -f "$b" ] && [ -x "$b" ] && ln -sf "$b" "$BIN_DIR/$(basename "$b")"
        done
    fi
else
    echo "[5/5] entrez-direct already installed (skipped)"
fi

echo ""
echo "Done. Verify with:"
echo "  $BIN_DIR/prefetch --version"
echo "  $BIN_DIR/igblastn -version | head -1"
echo "  $BIN_DIR/vsearch --version 2>&1 | head -1"
echo "  $BIN_DIR/seqkit version"
echo "  $BIN_DIR/esearch -version"
echo ""
echo "setup.sh prepends $BIN_DIR to PATH and exports IGDATA automatically."
