#!/bin/bash
# Resubmit bindingdb_kd DTA baseline survey after the contact_graph speedup
# (commit d829df1: scipy.sparse.csgraph refactor, 50-500x speedup).
#
# All other reps are already cached as .npy on disk from prior runs;
# only contact_graph + 4 cheap structure reps need to be computed.
#
# Usage (from /data/stat-cadd/scat9264/KIRBy):
#   bash tests/slurm_scripts/submit_dta_bindingdb.sh

set -euo pipefail

OUT="tests/results/phase5_baselines/dta"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"

mkdir -p "${OUT}"

JOB=$(sbatch <<EOF | awk '{print $4}'
#!/bin/bash
#SBATCH --job-name=dta_bindingdb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=${EMAIL}
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd /data/stat-cadd/scat9264/KIRBy
git pull --ff-only
. ./setup.sh
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error

python tests/dta_baseline_survey.py bindingdb_kd --n-folds 5 --max-pairs 30000 --output ${OUT}/bindingdb_kd.json
EOF
)

echo "  ${JOB}  dta_bindingdb  (long, 128G, 24h, GPU)"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Output:   ${OUT}/bindingdb_kd.json"
echo "Log:      slurm-${JOB}.out"
