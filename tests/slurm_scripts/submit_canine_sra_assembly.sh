#!/bin/bash
#SBATCH --job-name=canine_vh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# T-Ab.2 — Canine VH assembly from NCBI SRA project PRJNA790470.
#
# This is the ZERO-AUTH alternative to DoggifAI (Zenodo 15125375), used to
# get preliminary canineness results while the DoggifAI access request is
# pending. PRJNA790470 (Friedrichs 2022, PLOS One) is a fully open canine
# BCR-seq study: 25 healthy + 18 lymphoma dogs, Illumina MiSeq 2×300 bp,
# heavy / light chain libraries separately, 5' RACE with constant-region
# primers. Same exact pipeline as the bovine assembly — only the
# BioProject, IgBLAST organism, and output paths differ.
#
# When DoggifAI lands, drop heavy.fasta into the same canine_antibodies/
# directory; the loader prefers it over canine_vh.fasta automatically.
#
# PIPELINE (each stage is idempotent — skipped if its output already exists):
#   1. Resolve SRR run IDs for PRJNA790470 (via efetch).
#   2. prefetch + fasterq-dump → paired-end FASTQs.
#   3. vsearch --fastq_mergepairs → merged single-end FASTAs.
#   4. Fetch canine IgBLAST V/D/J germline DBs (canis_lupus_familiaris).
#   5. igblastn against canine IGH germlines → AIRR-format TSV.
#   6. scripts/parse_igblast_airr.py → filter productive + IGH, extract VH
#      amino-acid sequences, dedup, write canine_vh.fasta.
#
# DEPENDENCIES on ARC (same as bovine):
#   One-time install via `bash scripts/install_bcr_binaries.sh` — downloads
#   static Linux x86-64 binaries for sra-tools, igblast, vsearch, seqkit,
#   entrez-direct into $REPO/bin/. setup.sh prepends $REPO/bin to PATH and
#   exports IGDATA automatically. No conda, no `module load` beyond what
#   setup.sh already does.
#
# OUTPUT:
#   $DATA_ROOT/canine_antibodies/canine_vh.fasta
#   $DATA_ROOT/canine_antibodies/assembly_report.json

set -euo pipefail

cd /data/stat-ecr/scat9264/KIRBy
git pull 2>/dev/null || true
. ./setup.sh
export PYTHONWARNINGS=ignore

BIOPROJECT="PRJNA790470"
DATA_ROOT="${KIRBY_CROSS_SPECIES_DIR:-/data/stat-ecr/scat9264/kirby_data}"
OUT_DIR="$DATA_ROOT/canine_antibodies"
SRA_DIR="$OUT_DIR/sra"
FASTQ_DIR="$OUT_DIR/fastq"
MERGED_DIR="$OUT_DIR/merged"
AIRR_DIR="$OUT_DIR/airr"
GERM_DIR="$OUT_DIR/germlines/canis_lupus_familiaris"
FINAL_FASTA="$OUT_DIR/canine_vh.fasta"
REPORT="$OUT_DIR/assembly_report.json"

mkdir -p "$SRA_DIR" "$FASTQ_DIR" "$MERGED_DIR" "$AIRR_DIR" "$GERM_DIR"

echo "T-Ab.2 canine VH assembly  (BioProject=$BIOPROJECT)"
echo "Start: $(date)"
echo "DATA_ROOT: $DATA_ROOT"

# -----------------------------------------------------------------------------
# 1. Resolve SRR list
# -----------------------------------------------------------------------------
SRR_LIST_FILE="$OUT_DIR/srr_list.txt"
if [ ! -s "$SRR_LIST_FILE" ]; then
    echo "[1/6] Resolving SRR accessions for $BIOPROJECT"
    esearch -db sra -query "${BIOPROJECT}[BioProject]" \
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
# 4. Fetch canine IgBLAST germline DBs from IMGT GENE-DB.
# NCBI ships pre-built germlines for human/mouse/rat/rabbit/rhesus only —
# bovine and canine need a custom pull from IMGT's combined reference set.
# -----------------------------------------------------------------------------
echo "[4/6] Canine IGH germline databases (IMGT GENE-DB)"
IMGT_COMBINED="$GERM_DIR/IMGTGENEDB-ReferenceSequences.fasta"
if [ ! -s "$IMGT_COMBINED" ]; then
    wget -q -O "$IMGT_COMBINED" \
        "https://www.imgt.org/download/GENE-DB/IMGTGENEDB-ReferenceSequences.fasta-nt-WithGaps-F+ORF+inframeP" \
        || { echo "    FAILED to fetch IMGT combined reference"; exit 1; }
fi

# Per-segment FASTAs (V/D/J)
if [ ! -s "$GERM_DIR/canis_lupus_familiaris_imgt_V.fasta" ]; then
    python scripts/filter_imgt_germlines.py \
        --input "$IMGT_COMBINED" \
        --species "Canis lupus familiaris" \
        --locus IGH \
        --out-dir "$GERM_DIR" \
        --out-prefix canis_lupus_familiaris_imgt \
        || { echo "    FAILED to filter canine germlines"; exit 1; }
fi

# BLAST DBs
for SEG in V D J; do
    DBNAME="canis_lupus_familiaris_imgt_${SEG}"
    if [ ! -s "$GERM_DIR/${DBNAME}.nhr" ]; then
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
    # IgBLAST has no built-in 'canis_lupus_familiaris' organism — its
    # internal_data only covers human/mouse/rat/rabbit/rhesus_monkey. We use
    # the custom canine V/D/J databases built in step 4 and pass -organism
    # mouse purely so IgBLAST has SOME annotation table available; V-D-J
    # calling + productive flag come from our custom DBs. CDR boundary
    # annotation drifts (mouse FW positions on canine sequences) but our
    # parser only consumes `productive` + `sequence_alignment_aa`.
    igblastn \
        -germline_db_V "$GERM_DIR/canis_lupus_familiaris_imgt_V" \
        -germline_db_D "$GERM_DIR/canis_lupus_familiaris_imgt_D" \
        -germline_db_J "$GERM_DIR/canis_lupus_familiaris_imgt_J" \
        -organism mouse \
        -ig_seqtype Ig \
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
