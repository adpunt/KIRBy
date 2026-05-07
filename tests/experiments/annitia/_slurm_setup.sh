#!/usr/bin/env bash
# ANNITIA SLURM stage 0 — environment setup, val_test discovery, optional
# dep install. Sourced by submit_annitia_{short,medium,long}.sh.
#
# Sets:
#   REPO                      — repo root (cd's into it)
#   ANNITIA_VAL_TEST          — absolute path to val_test.csv
#   ANNITIA_TRAIN_CSV         — absolute path to train.csv (info only)
# Activates the py311-kirby venv via setup.sh.
# Optionally pip-installs the foundation-model deps if ANNITIA_AUTOINSTALL=1.

set -e

REPO=/data/stat-ecr/scat9264/KIRBy
cd "$REPO"

echo "============================================================"
echo "  ANNITIA SLURM setup ($(date))"
echo "  REPO: $REPO"
echo "============================================================"

# 1. Activate environment
. ./setup.sh

# 2. Locate val_test.csv. First $ANNITIA_VAL_TEST if set, then search
#    a candidate list. The migration moved data from stat-cadd to stat-ecr
#    but old jobs may have left it elsewhere.
if [ -n "$ANNITIA_VAL_TEST" ] && [ -f "$ANNITIA_VAL_TEST" ]; then
    echo "  val_test (from env): $ANNITIA_VAL_TEST"
else
    CANDIDATES=(
        "$REPO/data/annitia/val_test.csv"
        "/data/stat-ecr/scat9264/KIRBy/data/annitia/val_test.csv"
        "/data/stat-ecr/scat9264/data/annitia/val_test.csv"
        "/data/stat-cadd/scat9264/KIRBy/data/annitia/val_test.csv"
        "/data/stat-cadd/scat9264/data/annitia/val_test.csv"
        "$HOME/data/annitia/val_test.csv"
    )
    ANNITIA_VAL_TEST=""
    for p in "${CANDIDATES[@]}"; do
        if [ -f "$p" ]; then
            ANNITIA_VAL_TEST="$p"
            echo "  val_test (auto-found): $ANNITIA_VAL_TEST"
            break
        fi
    done
    if [ -z "$ANNITIA_VAL_TEST" ]; then
        echo "ERROR: val_test.csv not found in any of:"
        printf '  %s\n' "${CANDIDATES[@]}"
        echo
        echo "Fix one of these:"
        echo "  (a) export ANNITIA_VAL_TEST=/full/path/to/val_test.csv before sbatch"
        echo "  (b) place the file at \$REPO/data/annitia/val_test.csv"
        echo "  (c) scp it from your laptop:"
        echo "      scp /Volumes/seagate/kirby/annitia/val_test.csv \\"
        echo "          gateway:$REPO/data/annitia/"
        exit 1
    fi
fi
export ANNITIA_VAL_TEST

# 3. Locate train.csv (informational — annitia_maslg.load_annitia uses
#    its own resolver; we just print where it found it for debugging).
TRAIN_CANDIDATES=(
    "$REPO/data/annitia/train.csv"
    "$(dirname "$ANNITIA_VAL_TEST")/train.csv"
)
ANNITIA_TRAIN_CSV=""
for p in "${TRAIN_CANDIDATES[@]}"; do
    if [ -f "$p" ]; then ANNITIA_TRAIN_CSV="$p"; break; fi
done
if [ -n "$ANNITIA_TRAIN_CSV" ]; then
    echo "  train.csv: $ANNITIA_TRAIN_CSV"
else
    echo "  WARN: train.csv not found alongside val_test — load_annitia may fail"
fi

# 4. Optional foundation-model dep install. Off by default; set
#    ANNITIA_AUTOINSTALL=1 to opt in. These conditions degrade gracefully
#    if the import fails, so missing deps just skip the corresponding
#    branches — they don't block the pipeline.
if [ "${ANNITIA_AUTOINSTALL:-0}" = "1" ]; then
    echo
    echo "  Auto-installing optional foundation-model deps..."
    pip install --quiet --upgrade tabpfn 2>&1 | tail -3 || echo "    tabpfn install failed (T1.1 will skip)"
    pip install --quiet --upgrade synthcity 2>&1 | tail -3 || echo "    synthcity install failed (T1.4 will use GMM only)"
    pip install --quiet --upgrade tab-ebm 2>&1 | tail -3 || echo "    tab-ebm install failed (T1.4 TabEBM will skip)"
    pip install --quiet --upgrade survtrace 2>&1 | tail -3 || echo "    survtrace install failed (T1.5 will skip)"
    pip install --quiet --upgrade "autogluon.tabular[mitra]" 2>&1 | tail -3 || echo "    MITRA install failed (T1.1 MITRA will skip)"
    pip install --quiet --upgrade "git+https://github.com/Lexsi-Labs/TabTune.git" 2>&1 | tail -3 || echo "    TabTune install failed (T1.2 will skip)"
fi

# 5. Defaults for pipeline knobs (overridden by user's exports)
export ANNITIA_TSFM_BACKEND=${ANNITIA_TSFM_BACKEND:-fft}
export ANNITIA_K_DISC=${ANNITIA_K_DISC:-10}
export ANNITIA_N_SYNTH=${ANNITIA_N_SYNTH:-50,100,200,500}
export ANNITIA_T23_SUBSETS=${ANNITIA_T23_SUBSETS:-fib4_high,fib4_high_2007,fib4_high_2017,t2dm_only,no_viral}

echo
echo "  Setup complete."
echo "    ANNITIA_VAL_TEST=$ANNITIA_VAL_TEST"
echo "    ANNITIA_K_DISC=$ANNITIA_K_DISC"
echo "    ANNITIA_N_SYNTH=$ANNITIA_N_SYNTH"
echo "    ANNITIA_T23_SUBSETS=$ANNITIA_T23_SUBSETS"
echo "============================================================"
