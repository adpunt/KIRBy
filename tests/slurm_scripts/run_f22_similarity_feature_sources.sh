#!/bin/bash
# F22 — do the four parts of the sim_* recipes work better as four separate
# representations than as the nine hand-written combinations?
#
# Compares, on scaffold-grouped held-out folds, scored on read-across Spearman
# and on agreement with maximum common substructure:
#   - each of ecfp4_count / atompair_count / desc208 / pharmacophore3d alone
#   - each of the dataset's own registered representations alone
#   - combinations across representation set x feature ranking x budget split
#     x budget x how the kept features are glued together
#
# WHY THE SERVER. pharmacophore3d generates a 3D conformer per molecule. On the
# project laptop that took 2133s for 1412 molecules, so a 5000-molecule dataset
# spends roughly two hours there before scoring anything. mordred took 1687s on
# the same 1412. Six of every molecular dataset's ten registered representations
# also need torch or a graph library, which do not import on the laptop, so a
# local run can only compare against four or five of the ten.
#
# STANDALONE AND CACHE-FREE. Reads no hybrid search, no baseline pickle, no
# hybrid-eval pickle and no result database, and writes none. Nothing here
# needs deleting before or after.
#
# ONE ARRAY TASK PER DATASET. Each task writes its own file under
# tests/results/, so there are no write races.
#
# f22 and f23 are separate processes, so each BUILDS THE REPRESENTATIONS AGAIN.
# On logd that is two passes of 3D conformer generation, not one. The wall time
# below allows for it. Sharing them would mean a cache, and both scripts are
# deliberately cache-free.
#
# QM9 RISK. src/kirby/datasets/qm9.py prefers /Volumes/seagate/QM9, which is the
# laptop's external drive and does not exist here, so on the cluster it falls
# back to $KIRBY_CACHE_DIR/QM9. If QM9 is not staged there, array task 2 fails
# at the loader. The other six tasks are unaffected.
#
# CLASSIFICATION. Both scripts refuse a classification dataset, because
# read-across here is a nearest-neighbour regression scored by Spearman. Every
# dataset below is regression. Do not add `herg` to this list.
#
# WALL TIME. Building the representations dominates and scales with molecule
# count: on 1412 molecules the build was about 70 minutes and scoring one fold
# was minutes. --time is set to 24h, which is well clear for 5000 molecules at
# 5 folds. That is a scaled estimate from the laptop, not a measured server time.
#
# --account and --partition are LIVE STATE. Run where_to_submit.sh first, then:
# Checked 2026-08-26: stat-cadd has the higher fair-share (0.0436 vs 0.0343),
# and medium has 249 jobs pending against short's 948 while offering a 2-day
# wall limit. short's 12h ceiling is too tight for logd at 20000 pairs.
#   sbatch --account=stat-cadd --partition=medium --array=0-6 \
#          tests/slurm_scripts/run_f22_similarity_feature_sources.sh
#SBATCH --job-name=f22_f23_sim
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

# Molecular datasets only: the four candidates need SMILES, 3D conformers and
# maximum-common-substructure pairs.
# Molecule counts measured on 2026-08-26 with load_dataset, after InChIKey
# deduplication. All seven load on the laptop, so none of these is a server-only
# dataset -- the server is for the 3D conformers and the five representations
# that need torch.
DATASETS=(
  herg_ki            # 1412 molecules, 769 scaffold groups
  logd               # 5028 molecules
  qm9                # 4995 sampled, computed HOMO-LUMO gap, no assay noise
  caco2              # 2158 molecules
  adme_hlm           # 3086 molecules
  adme_rlm           # 3053 molecules
  adme_mdr1          # 2641 molecules
)

ds="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "f22 array task ${SLURM_ARRAY_TASK_ID} -> ds=${ds}"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}"

# 20000 pairs, not the 4000 default. With 5 folds only the pairs whose BOTH
# molecules land in the same held-out fold can be scored, which is about
# (1/5)^2 of them. At 4000 that leaves roughly 160 pairs per fold and the
# substructure column gets noisy. Maximum common substructure cost 0.12s per
# pair on the laptop, so 20000 is about 40 minutes once per dataset.
# RUN_F22 / RUN_F23 default to 1, so plain sbatch behaves exactly as before.
# Set RUN_F22=0 to redo only the second half: on 2026-08-29 array 12906522 wrote
# all seven f22 files and then lost every f23 to one stale package, and
# rebuilding the representations for f22 a second time would buy nothing.
#   sbatch --export=ALL,RUN_F22=0 --partition=<part> --array=1-6 ...
if [ "${RUN_F22:-1}" = "1" ]; then
    python -u tests/experiments/framework/f22_similarity_feature_sources.py \
        --dataset "${ds}" --mode full --n-pairs 20000
fi

# F23 reuses f22's representation list, rankers and gluing by importing the
# module, but it is a separate process and builds the representations again. It
# asks the separate question of which space finds corrupted activity values best.
if [ "${RUN_F23:-1}" = "1" ]; then
    python -u tests/experiments/framework/f23_noise_cleaning_metric.py \
        --dataset "${ds}" --budgets 100,250,500,1000 --n-pairs 20000
fi
