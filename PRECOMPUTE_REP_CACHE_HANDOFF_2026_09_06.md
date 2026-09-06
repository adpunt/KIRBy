# Handoff: precompute the representation caches

Written 2026-09-06 on branch `similarity-metrics-study`. Everything below was
read from the code or from a command run on 2026-09-06; nothing is from memory.

## The job to do

Run a GPU precompute that fills the representation cache for every dataset in
the rep-pruning roster, so that downstream experiments stop silently skipping
representations they cannot find.

## What already exists

`tests/precompute_reps.py` builds the pruned pool for one dataset and stops.
Fits no model, runs no cross-validation. `--list` prints the roster, read from
`DATASET_REPS` at run time, so the array tracks the roster with no edit. Today
that is **39 datasets**.

`tests/slurm_scripts/run_precompute_reps.sh` submits it, one dataset per array
task. Already carries `--clusters=htc`, `--gres=gpu:1`, `--partition=long`,
`--time=47:59:00`, `--mem=64G`, `--cpus-per-task=8`. Its own header says the
wall and memory are **not measured** — read MaxRSS and Elapsed off the first
finished task and reset them.

    N=$(python tests/precompute_reps.py --list | wc -l)
    sbatch --clusters=htc --array=0-$((N-1))%8 \
           tests/slurm_scripts/run_precompute_reps.sh

`tests/analysis/check_rep_cache.py` inventories what is cached. Stdlib only,
runs on a login node. It needs `KIRBY_CACHE_DIR` set or it reads a different
store than the jobs use.

## Three things that will stop this from working, all measured today

### 1. There are TWO cache layouts and filling one does not fill the other

`tests/precompute_reps.py` writes

    $KIRBY_CACHE_DIR/rep_cache/<dataset>/<rep>.npy

The antibody path — `_rep_cache_path` in `tests/antibody_baseline_survey.py`,
which `tests/experiments/structural_geometry_survey.py` loads from — reads

    $KIRBY_CACHE_DIR/reps/ab_<sha256 of the sequence list>/<rep>.npy

Keyed on a hash of the heavy and light sequences, not on the dataset name.
Changing a loader changes the hash and every cached array for that dataset
misses. This is why 20 completed geometry runs each loaded 2 of 9
representations while the cache held esm2, esm3, prot_t5, saprot, ablang2,
mint and igbert for four other antibody sequence lists.

**Open: which layout wins, and who converts the other.** Nothing in the repo
reconciles them today.

### 2. The pruned pool is per dataset, and it is not the list the geometry survey asks for

`structural_geometry_survey.py` line 67 hardcodes nine names for every dataset:
`esm2, ablang2, prot_t5, esm3, mint, saprot, antiberty, blosum, aac`.

`DATASET_REPS[ds]['general']` is ten names chosen per dataset during rep
pruning. Overlap with the survey's fixed nine, measured today:

| dataset | representations in both |
|---|---|
| abcov_kd | ablang2, blosum, esm3 |
| ddg_combined | antiberty |
| flab_thermo_tresanco | esm3, saprot |
| ginkgo_acsins | aac, ablang2, prot_t5, saprot |
| ginkgo_colloidal | ablang2, antiberty, esm3, prot_t5, saprot |
| ginkgo_hic | aac, ablang2, esm3, prot_t5 |
| ginkgo_pr_cho | aac, ablang2, prot_t5, saprot |
| ginkgo_titer | ablang2, esm2, esm3, prot_t5, saprot |

So a precompute of the pruned pools does not by itself satisfy the geometry
survey, and vice versa.

**Open: does the geometry survey drop its fixed nine and take
`DATASET_REPS[ds]['general']` instead?**

### 3. Twelve of the geometry survey's twenty datasets are not in the roster at all

Not in `DATASET_REPS`, so `precompute_reps.py --list` never emits them:

    abagym_all, flab_aggregation_jain, flab_aggregation_jain_hicrt,
    flab_immunogenicity_marks, flab_pk_jain_fcrnrt, flab_pk_jain_heprt,
    flab_polyreact_jain_bvp, flab_polyreact_jain_cic, flab_thermo_jain,
    ginkgo_aggregation, ginkgo_purity, ginkgo_tm2

**Open: do these get pruned pools, or does the precompute take a dataset list
that is not the pruning roster?**

## Evidence this is the live blocker

`tests/results/geometry_survey/` holds 20 result files, all status OK, all
reporting `base reps loaded: 2/9`. The two present are `aac` and `blosum` —
the ones `_load_base_reps` computes itself when they are absent. Everything
else is skipped with a warning and the run does not fail. Jobs 8708910 and
8708911, 12h57m and 2h33m, htc.

Read them with `python tests/analysis/read_geometry_survey.py <dir>`.

## Constraints

- `KIRBY_CACHE_DIR=/data/stat-ecr/scat9264/.cache/kirby`, set by `setup.sh`.
  Without it every path above resolves somewhere else.
- The cluster is set by the login node, not by `--partition`. A `--gres=gpu`
  job submitted from arc-login comes back `Invalid feature specification`.
  Keep `--clusters=htc`.
- `bash tests/slurm_scripts/where_to_submit.sh` before submitting; it is
  mode 644, so `bash`, not `./`.
- Do not cut datasets or representations from the roster for compute.
- Never `git checkout/reset/stash/clean/restore`.
