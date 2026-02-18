# SLURM Submission Guide

Each experiment has its own script: `sbatch <script>.sh [mode] [dataset] [reps] [extra_args]`

Defaults: `intermediary logd fast`

## Dependencies

```
Phase 1 (independent):  c01  c02  c04  c05  c06  c10  s01  s02  s08
Phase 2 (after c02):    c07  c08
Phase 3 (after c02+c06): c09
Phase 4 (after s02):    s03  s04  s05  s06  s07
Phase 5 (cross-dataset): c03 (after all c02), s09 (after all s02-s08)
```

## Quick Start — Intermediary on logd

```bash
# Phase 1
for s in c01 c02 c04 c05 c06 c10 s01 s02 s08; do
    sbatch tests/experiments/slurm/${s}.sh intermediary logd fast
done
# Note the job IDs for c02, c06, s02 from the output above

# Phase 2 — replace JOB IDs
for s in c07 c08; do
    sbatch --dependency=afterok:<C02_JOB> tests/experiments/slurm/${s}.sh intermediary logd fast
done
sbatch --dependency=afterok:<C02_JOB>:<C06_JOB> tests/experiments/slurm/c09.sh intermediary logd fast

for s in s03 s04 s05 s06 s07; do
    sbatch --dependency=afterok:<S02_JOB> tests/experiments/slurm/${s}.sh intermediary logd fast
done
```

## Full Mode — All Datasets

```bash
for ds in logd caco2 herg qm9; do
    for s in c01 c02 c04 c05 c06 c10 s01 s02 s08; do
        sbatch tests/experiments/slurm/${s}.sh full $ds fast+hf
    done
done
# Then submit dependent phases per-dataset using job IDs as above

# Cross-dataset (after ALL datasets finish):
sbatch tests/experiments/slurm/c03.sh full fast+hf
sbatch tests/experiments/slurm/s09.sh fast+hf
```
