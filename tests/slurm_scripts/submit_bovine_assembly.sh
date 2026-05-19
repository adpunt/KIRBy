#!/bin/bash
#SBATCH --job-name=bovine_vh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# T-Ab.2 — Bovine VH assembly from NCBI SRA project PRJNA690660.
#
# PIPELINE (each stage is idempotent — skipped if its output already exists):
#   1. Resolve SRR run IDs for PRJNA690660 (via efetch).
#   2. prefetch + fasterq-dump → paired-end FASTQs.
#   3. vsearch --fastq_mergepairs → merged single-end FASTQs.
#   4. Translate merged reads to FASTA + reverse-complement scan via seqkit.
#   5. igblastn against bovine IGH V/D/J germlines → AIRR-format TSV.
#   6. scripts/parse_igblast_airr.py → filter productive + IGH, extract VH
#      amino-acid sequences, dedup, write bovine_vh.fasta.
#
# DEPENDENCIES on ARC:
#   One-time install via `bash scripts/install_bcr_binaries.sh` — downloads
#   static Linux x86-64 binaries for sra-tools, igblast, vsearch, seqkit,
#   entrez-direct into $REPO/bin/. setup.sh prepends $REPO/bin to PATH and
#   exports IGDATA automatically. No conda, no `module load` beyond what
#   setup.sh already does. (biopython is already in the venv.)
#
# OUTPUT:
#   $DATA_ROOT/bovine_antibodies/bovine_vh.fasta
#   $DATA_ROOT/bovine_antibodies/assembly_report.json
#
# See docs/cross_species_data_inventory.md §2.2 for the layout.

set -euo pipefail

cd /data/stat-ecr/scat9264/KIRBy
git pull 2>/dev/null || true
. ./setup.sh
export PYTHONWARNINGS=ignore

DATA_ROOT="${KIRBY_CROSS_SPECIES_DIR:-/data/stat-ecr/scat9264/kirby_data}"
OUT_DIR="$DATA_ROOT/bovine_antibodies"
SRA_DIR="$OUT_DIR/sra"
FASTQ_DIR="$OUT_DIR/fastq"
MERGED_DIR="$OUT_DIR/merged"
AIRR_DIR="$OUT_DIR/airr"
GERM_DIR="$OUT_DIR/germlines/bos_taurus"
FINAL_FASTA="$OUT_DIR/bovine_vh.fasta"
REPORT="$OUT_DIR/assembly_report.json"

mkdir -p "$SRA_DIR" "$FASTQ_DIR" "$MERGED_DIR" "$AIRR_DIR" "$GERM_DIR"

echo "T-Ab.2 bovine VH assembly"
echo "Start: $(date)"
echo "DATA_ROOT: $DATA_ROOT"

# -----------------------------------------------------------------------------
# 1. Resolve SRR list for PRJNA690660
# -----------------------------------------------------------------------------
SRR_LIST_FILE="$OUT_DIR/srr_list.txt"
if [ ! -s "$SRR_LIST_FILE" ]; then
    echo "[1/6] Resolving SRR accessions for PRJNA690660"
    # efetch returns a CSV runinfo; column 1 is the SRR
    esearch -db sra -query 'PRJNA690660[BioProject]' \
        | efetch -format runinfo \
        | tail -n +2 \
        | cut -d',' -f1 \
        | grep -E '^[SED]RR' \
        > "$SRR_LIST_FILE"
    echo "    Found $(wc -l < "$SRR_LIST_FILE") SRR runs"
else
    echo "[1/6] SRR list already present at $SRR_LIST_FILE"
fi

# -----------------------------------------------------------------------------
# 2. Download + convert each SRR to paired-end FASTQ
# -----------------------------------------------------------------------------
echo "[2/6] prefetch + fasterq-dump per SRR"
while read -r SRR; do
    F1="$FASTQ_DIR/${SRR}_1.fastq.gz"
    F2="$FASTQ_DIR/${SRR}_2.fastq.gz"
    if [ -s "$F1" ] && [ -s "$F2" ]; then
        echo "    $SRR already converted"
        continue
    fi
    if [ ! -f "$SRA_DIR/${SRR}/${SRR}.sra" ]; then
        prefetch -O "$SRA_DIR" "$SRR" || { echo "prefetch failed for $SRR"; continue; }
    fi
    fasterq-dump --split-files -e 8 -O "$FASTQ_DIR" \
        "$SRA_DIR/${SRR}/${SRR}.sra" || { echo "fasterq-dump failed for $SRR"; continue; }
    gzip -f "$FASTQ_DIR/${SRR}_1.fastq" "$FASTQ_DIR/${SRR}_2.fastq" 2>/dev/null || true
done < "$SRR_LIST_FILE"

# -----------------------------------------------------------------------------
# 3. Pair-end merge with vsearch
# -----------------------------------------------------------------------------
echo "[3/6] vsearch --fastq_mergepairs"
while read -r SRR; do
    MERGED="$MERGED_DIR/${SRR}.merged.fasta"
    if [ -s "$MERGED" ]; then
        echo "    $SRR already merged"
        continue
    fi
    F1="$FASTQ_DIR/${SRR}_1.fastq.gz"
    F2="$FASTQ_DIR/${SRR}_2.fastq.gz"
    if [ ! -s "$F1" ] || [ ! -s "$F2" ]; then
        echo "    $SRR FASTQs missing, skipping"
        continue
    fi
    vsearch --fastq_mergepairs "$F1" --reverse "$F2" \
        --fastaout "$MERGED" \
        --fastq_maxdiffs 30 \
        --threads 8 \
        --quiet || echo "    $SRR vsearch failed (continuing)"
done < "$SRR_LIST_FILE"

# -----------------------------------------------------------------------------
# 4. Fetch bovine IgBLAST germline DBs (one-time)
# -----------------------------------------------------------------------------
echo "[4/6] Bovine IGH germline databases"
GERMLINE_URL="https://ftp.ncbi.nih.gov/blast/executables/igblast/release/database"
for SEG in V D J; do
    DBNAME="bos_taurus_imgt_${SEG}"
    if [ ! -s "$GERM_DIR/${DBNAME}.fasta" ]; then
        wget -q "$GERMLINE_URL/${DBNAME}.fasta" -O "$GERM_DIR/${DBNAME}.fasta" \
            || { echo "    FAILED to fetch ${DBNAME}.fasta"; exit 1; }
        # Build BLAST DB
        makeblastdb -parse_seqids -dbtype nucl \
            -in "$GERM_DIR/${DBNAME}.fasta" \
            -out "$GERM_DIR/${DBNAME}"
    fi
done

# -----------------------------------------------------------------------------
# 5. igblastn per sample → AIRR TSV
# -----------------------------------------------------------------------------
echo "[5/6] igblastn"
# setup.sh exports IGDATA from $REPO/bin/.igdata when the install script
# was used. If something else is providing igblastn (e.g. a system package),
# best-effort detect from the binary location.
if [ -z "${IGDATA:-}" ]; then
    IGDATA="$(dirname "$(readlink -f "$(which igblastn)")")/.."
    export IGDATA
fi
echo "    IGDATA=$IGDATA"

while read -r SRR; do
    AIRR="$AIRR_DIR/${SRR}.airr.tsv"
    MERGED="$MERGED_DIR/${SRR}.merged.fasta"
    if [ -s "$AIRR" ]; then
        echo "    $SRR already IgBLASTed"
        continue
    fi
    if [ ! -s "$MERGED" ]; then
        echo "    $SRR merged FASTA missing, skipping"
        continue
    fi
    igblastn \
        -germline_db_V "$GERM_DIR/bos_taurus_imgt_V" \
        -germline_db_D "$GERM_DIR/bos_taurus_imgt_D" \
        -germline_db_J "$GERM_DIR/bos_taurus_imgt_J" \
        -organism bos_taurus \
        -ig_seqtype Ig \
        -domain_system imgt \
        -outfmt 19 \
        -query "$MERGED" \
        -out "$AIRR" \
        -num_threads 8 || { echo "    $SRR igblastn failed"; continue; }
done < "$SRR_LIST_FILE"

# -----------------------------------------------------------------------------
# 6. Parse + dedup → final FASTA + report
# -----------------------------------------------------------------------------
echo "[6/6] parse_igblast_airr.py"
python scripts/parse_igblast_airr.py \
    --airr-dir "$AIRR_DIR" \
    --out-fasta "$FINAL_FASTA" \
    --out-report "$REPORT" \
    --min-vh-length 90 \
    --max-vh-length 250

echo "Done: $(date)"
echo "Final FASTA: $FINAL_FASTA"
echo "Report: $REPORT"
