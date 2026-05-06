#!/bin/bash
# Davis cold-protein baseline investigation.
#
# Tests whether the 9 drug reps tied at CI=0.7567 in the davis baseline
# survey are actually learning, or just collapsing to per-drug-mean fitting
# (because every drug appears in train under cold-protein splits).
#
# Output: tests/results/phase5_baselines/dta/davis_investigation.json
#   - trivial drug-mean predictor's CI/R²/RMSE on the same 5 folds
#   - per-fold ridge predictions for {drug_avalon, drug_chemberta,
#     drug_ecfp4, drug_maccs}
#   - pairwise correlation across reps + correlation with drug-mean
#
# CPU-only (small ridge fits + sklearn). ~1h expected.
#
# Usage (from /data/stat-ecr/scat9264/KIRBy on HTC login):
#   bash tests/slurm_scripts/submit_davis_investigation.sh

set -euo pipefail

EMAIL="adelaide.punt@stcatz.ox.ac.uk"

JOB=$(sbatch <<EOF | awk '{print $4}'
#!/bin/bash
#SBATCH --job-name=davis_invest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short
#SBATCH --mem=32G
#SBATCH --mail-user=${EMAIL}
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd /data/stat-ecr/scat9264/KIRBy
git pull --ff-only
. ./setup.sh
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error

python tests/scripts/davis_baseline_investigation.py
EOF
)

echo "  ${JOB}  davis_invest  (short, 32G, 2h, no GPU)"
echo ""
echo "Output: tests/results/phase5_baselines/dta/davis_investigation.json"
echo ""
echo "When it lands, scp it down with:"
echo "  scp scat9264@htc-login.arc.ox.ac.uk:/data/stat-ecr/scat9264/KIRBy/tests/results/phase5_baselines/dta/davis_investigation.json tests/results/phase5_baselines/dta/"
