#!/bin/bash
#SBATCH --job-name=bp_trace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=11:59:00
#SBATCH --partition=short
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# Trace the best hybrid per dataset -> base rep -> chemical/biological meaning.
# Runs tests/analysis/blueprint_trace.py, which loads each dataset's precomputed
# search, re-fits the best blueprint on full data (the slow step), and writes
# tests/analysis/blueprints/<ds>_features.csv + <ds>_blueprint.json.
#
# Usage:
#   # one domain (default molecular):
#   sbatch tests/slurm_scripts/run_blueprint_trace.sh
#   sbatch --export=DOMAIN=antibody   tests/slurm_scripts/run_blueprint_trace.sh
#   sbatch --export=DOMAIN=reaction   tests/slurm_scripts/run_blueprint_trace.sh
#
#   # specific datasets:
#   sbatch --export=DATASETS="caco2 herg_ki adme_hlm" tests/slurm_scripts/run_blueprint_trace.sh
#
#   # everything (all DATASET_REPS) — use the long partition, this can be many hours:
#   sbatch --partition=long --time=23:59:00 --export=DOMAIN= tests/slurm_scripts/run_blueprint_trace.sh
#
# Optional overrides: AUG (uncapped|capped|none, default uncapped), SPEED (full|fast),
# TOP (top-K features printed, default 25).
#
# Progress streams to slurm-<jobid>.out (python -u). Results persist under
# tests/analysis/blueprints/. To send them back: from this repo on the server,
#   git add -f tests/analysis/blueprints && git commit -m 'blueprint traces' && git push

set -u

cd /data/stat-ecr/scat9264/KIRBy
git pull 2>/dev/null || true
. ./setup.sh

AUG="${AUG:-uncapped}"
SPEED="${SPEED:-full}"
TOP="${TOP:-25}"

echo "=========================================="
echo "blueprint_trace: $(date)"
echo "  DOMAIN=${DOMAIN:-<unset>}  DATASETS=${DATASETS:-<unset>}  AUG=${AUG} SPEED=${SPEED} TOP=${TOP}"
echo "=========================================="

if [[ -n "${DATASETS:-}" ]]; then
    # word-split intentional: pass each dataset as a positional arg
    python -u tests/analysis/blueprint_trace.py ${DATASETS} --aug "${AUG}" --speed "${SPEED}" --top "${TOP}"
elif [[ -n "${DOMAIN+x}" ]]; then
    # DOMAIN was provided (even if empty). Empty => trace ALL DATASET_REPS.
    if [[ -n "${DOMAIN}" ]]; then
        python -u tests/analysis/blueprint_trace.py --domain "${DOMAIN}" --aug "${AUG}" --speed "${SPEED}" --top "${TOP}"
    else
        python -u tests/analysis/blueprint_trace.py --aug "${AUG}" --speed "${SPEED}" --top "${TOP}"
    fi
else
    # nothing passed -> sensible default
    python -u tests/analysis/blueprint_trace.py --domain molecular --aug "${AUG}" --speed "${SPEED}" --top "${TOP}"
fi
RC=$?

echo "=========================================="
echo "done (rc=${RC}) at $(date)"
echo "per-dataset CSV+JSON under tests/analysis/blueprints/"
echo "=========================================="
exit ${RC}
