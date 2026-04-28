#!/bin/bash
# Reaction baseline re-survey with full rep set, including CGR reps.
#
# Pre-CGR reaction surveys (Mar 19-26) used 14 reps (drfp_2048, concat_*,
# reactant_*, diff_*, rxnfp, etc). This script runs the full 41-rep survey
# which now includes:
#   - cgr_morgan          (Condensed Graph of Reaction Morgan FP)
#   - chytorch_graphormer (ChyTorch/rxnmap Graphormer encoder)
#   - cgr_smiles_lm       (ChemBERTa continued-pretrained on USPTO CGR-SMILES,
#                          checkpoint at kirby_data/cgr_smiles_pretrain/checkpoint/)
#
# Prerequisites:
#   - CGR-SMILES pretraining checkpoint exists (job 7483446 COMPLETED 2026-04-15)
#   - chython installed (for cgr_smiles_lm rep) — confirmed on HTC
#   - Existing .npy cached reps from prior surveys remain valid; only the
#     3 new CGR reps need to be computed fresh
#
# Usage (from /data/stat-cadd/scat9264/KIRBy on HTC login):
#   bash tests/slurm_scripts/submit_reaction_cgr_resurvey.sh          # both datasets
#   bash tests/slurm_scripts/submit_reaction_cgr_resurvey.sh bh       # buchwald_hartwig only
#   bash tests/slurm_scripts/submit_reaction_cgr_resurvey.sh doyle    # bh_doyle_plates only

set -euo pipefail

MODE=${1:-all}
OUT="tests/results/phase5_baselines/reaction"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"

mkdir -p "${OUT}"

submit_ds() {
    local ds=$1
    local job=$(sbatch <<EOF | awk '{print $4}'
#!/bin/bash
#SBATCH --job-name=rxn_${ds}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=${EMAIL}
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd /data/stat-cadd/scat9264/KIRBy
git pull --ff-only
. ./setup.sh
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error

python tests/run_baseline_survey.py ${ds} --reps all --n-folds 5 --skip-existing --output ${OUT}/
EOF
)
    echo "  ${job}  rxn_${ds}  (long, 256G, 24h, GPU)"
}

echo "=== Reaction re-survey with CGR reps ==="
echo ""

case "$MODE" in
    all)
        submit_ds buchwald_hartwig
        submit_ds bh_doyle_plates
        ;;
    bh)
        submit_ds buchwald_hartwig
        ;;
    doyle)
        submit_ds bh_doyle_plates
        ;;
    *)
        echo "Usage: bash tests/slurm_scripts/submit_reaction_cgr_resurvey.sh [all|bh|doyle]"
        exit 1
        ;;
esac

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Output:   ${OUT}/<dataset>.json"
echo ""
echo "With --skip-existing, only new reps (cgr_morgan, chytorch_graphormer,"
echo "cgr_smiles_lm, plus any freshly wired molecular variants) will compute."
