# The precompute and experiment run — every deliverable, 2026-09-14

One document for the whole run. Stages 0 to 4. Nothing in here is taken from a memory
file or from another document: every claim is marked with the file and line it was read
in on 2026-09-14, or marked UNVERIFIED.

---

## 0. The deliverables, as a checklist

| # | Deliverable | State |
|---|---|---|
| D1 | The demand matrix — what the experiments read, so the precompute knows what to write | **§2, read from the code 2026-09-14** |
| D2 | Representation preflight: does this dataset have its 10 representations on disk, before a job starts | **BUILT** — `tests/check_rep_pools.py`, §6.1 |
| D3 | `precompute_hybrids.py` gains a representation-count argument | **BUILT** — `--rep-counts`, §5.1 |
| D4 | Stage 0 — one dataset, three lineages, fast, inside `srun` | **WRITTEN** — §7 |
| D5 | Stage 1 submit script — the plain predictive pass, all conditions | **NOT BUILT** — §8 |
| D6 | Stage 2 submit script — the same over all 42 datasets | **NOT BUILT** — §9 |
| D7 | Stage 3 submit script — trial experiments, and the same with tuning on | **NOT BUILT** — §10 |
| D8 | Stage 4 submit script — all experiments | **NOT BUILT** — §11 |
| D9 | Progress check that reports rows written against rows expected | **BUILT** — `tests/precompute_coverage.py`, §6.2 |
| D10 | A refusal switch so an experiment cannot build a search live | **NOT BUILT** — §6.3 |
| D11 | Decision: the 6 datasets with no complete pool on ARC | **OPEN — yours**, §4.1 |
| D12 | Decision: memory and wall per job | **OPEN — needs stage 0's measurement**, §12 |

---

## 1. Where this runs

**ARC, CPU partitions. Not HTC, no GPU.** Read 2026-09-14: every `torch.cuda.is_available()`
in `src/kirby/` sits in a representation builder (`representations/sequence.py`,
`representations/forward_compat_*.py`). No downstream model asks for a device. The five
models a precompute fits are rf, lightgbm, xgboost, svm and dnn
(`find_best_hybrid.py:2550-2569`), and dnn is a CPU torch model here.

The GPU work is building the representations. That is a different job and it is already done
for the trial datasets — see §6.1 for how to check the server.

Before every submission:

```bash
cd /data/stat-ecr/scat9264/KIRBy && bash tests/slurm_scripts/where_to_submit.sh
bash tests/slurm_scripts/where_to_submit.sh --fits <cores> <memory>
```

`--emit` prints `ACCOUNT PARTITION` for a script to consume. Every submit script below reads
it that way rather than hardcoding an account.

⛔ `submit_dataset.sh:51` runs `git pull --ff-only origin main`. This work is on
`similarity-metrics-study`. Any new script must pull that branch by name or the server runs
code from main.

---

## 2. D1 — what the experiments read

Read on 2026-09-14 by walking every call in `tests/experiments/` that reaches a search:
111 call sites in 72 files (`scan_pool_calls.py`, in the session scratchpad).

### 2.1 What decides whether a stored result is reused

Two separate caches, and they behave differently.

**The result database** (`result_cache.py:46-127`) stores one row per configuration. Its
identity is `tier`, `labels_hash`, `groups_hash`, `n_folds`, `seed`, `cv_splitter_id`,
`metric`, `task_type`, `objective`, `rep_pool`, `tune`, `rep_factories_id`. Within one
identity, a row is addressed by `rep_subset`, `allocation_method`, `budget`,
`importance_method`, `model`, `internal_evaluator`, `step_size`, `rep_count`, `aug_strategy`,
`aug_names`, `aug_threshold`, `aug_budget`, `candidate_ranking`, `greedy_tolerance`,
`augmentation_margin`, `augmentation_fraction`.

**The search pickle** (`find_best_hybrid.py:525-700`) stores the whole search. Its key
includes everything above and also the model list, the budget list, the allocation /
importance / evaluator lists, the representation-count list, and `adaptive_pruning`.

So: an experiment that asks for a different model list, budget list or pruning setting misses
the pickle and rebuilds the search — but every configuration it needs is still a row in the
database, so it re-assembles rather than re-fits. An experiment that changes anything in the
database identity re-fits everything.

**This is the whole answer to "make sure crazy compute doesn't happen".** The precompute's
job is to fill the database rows. The pickle is a convenience.

### 2.2 The things that DO force a re-fit, and who asks for them

| what changes | who asks for it | cost |
|---|---|---|
| `tier='general_distance'` | c03, s04, s05, s06, s08, s10, s13 | a separate precompute pass (`--distance`) |
| `objective='similarity'` | n02, n06, n07, n11, n12, n13, s02, s04, s05, s06, s08, s10, s12, s13 | a separate pass (`--objectives similarity`) |
| `tune='tuned20_k'` | n15 only (`n15_clean_vs_noisy_hybrid.py:129-160`, owner's decision 2026-09-12) | a whole second precompute; n15 builds it live today |
| `labels_hash` changes | every noise experiment that injects label noise | by design, never precomputable |
| a private `tier` | c06 (`c06_variant`), c15 (`c15c_fast`), c16 (`c16_fast`), `general_ft`, `cache_evidence` | by design, experiment-local |

### 2.3 The things that only miss the pickle, and cost assembly not fitting

`adaptive_pruning=0.25` is passed at 13 call sites: the six opig files, the three
cross_species files, `covariate_shift.py`, `reaction_rep_comparison.py`,
`structural_geometry_survey.py`, `imrie_transfer_learning.py`, `fed05`. The precompute writes
`adaptive_pruning=False`. These miss the pickle and read the rows.

`models=RIDGE_MODELS` (c16, `c16_feature_selection_as_regularization.py:344`) adds ridge rows
only; the other five models' rows are already there.

### 2.4 The demand matrix

For each dataset, the precompute must write:

| lineage | tier | augmentation conditions | objectives | model list | budgets | evaluators |
|---|---|---|---|---|---|---|
| main | `general` | none, uncapped, capped | predictive | default five | `FULL_BUDGETS` (8) | `DEFAULT_INTERNAL_EVALUATORS` (4) |
| main | `general` | none, uncapped, capped | similarity | default five | `FULL_BUDGETS` (8) | same |
| main | `general` | none, uncapped, capped | similarity | default five **plus knn** | `FULL_BUDGETS` (8) | same |
| main | `general` | none, uncapped, capped | predictive | default five **plus ridge** | `FULL_BUDGETS` (8) | same |
| distance | `general_distance` | none, uncapped | predictive | `DISTANCE_MODELS` (6) | `DISTANCE_BUDGETS` (6) | `DISTANCE_EVALUATORS` (5) |

`uncapped` and `capped` exist only for a dataset with augmentation representations
(`_shared.py:5274` `_pool_conditions`, matched by `precompute_hybrids.py:490-510`).

The knn pass is similarity-only by the owner's rule of 2026-09-12
(`submit_precompute_extra_passes.sh:14-25`): k-nearest-neighbours is never scored as a
standard predictive model, only where nearest-neighbour prediction is the measurement.

### 2.5 How big one pass is

Counted from the code 2026-09-14, for a pool of 10 representations.

The evaluator list is `['match', 'rf', 'lightgbm', 'dnn']`. `match` is not a fourth evaluator
— it adds the model's own evaluator when that is not already in the list
(`find_best_hybrid.py:5739-5753`). So the number of (model, evaluator) pairs is:

| model | evaluators it runs | how many |
|---|---|---|
| rf | rf, lightgbm, dnn | 3 |
| lightgbm | rf, lightgbm, dnn | 3 |
| dnn | rf, lightgbm, dnn | 3 |
| xgboost | rf, lightgbm, dnn, xgboost | 4 |
| svm | rf, lightgbm, dnn, svm | 4 |
| | | **17** |

One row of the table below is one (dataset, augmentation condition, objective, model list).

| axis | how many |
|---|---|
| allocation methods | 5 |
| importance methods | 3 |
| budgets | 8 |
| representation counts | 4 (2, 3, 5, whole pool) |
| (model, evaluator) pairs | 17 |
| **configurations** | **8,160** |
| folds each | 5 |
| **model fits** | **40,800** |

Checked against the f33 record in `FRAMEWORK_FINALIZATION.md:1800`: it reports 3,060
configurations at full over three budgets, which is 1,020 per budget. The count above gives
5 × 3 × 4 × 17 = 1,020 per budget. The two agree.

Per dataset, main lineage, predictive objective, three augmentation conditions:
**24,480 configurations**, 122,400 model fits.

The distance lineage: 6 models × 5 evaluators = 30 pairs, 6 budgets, so
5 × 3 × 6 × 4 × 30 = **10,800 configurations** per condition, two conditions.

**A fast pass is a down payment, not a throwaway.** `speed` is in the search pickle's key but
not in the row database's. `FAST_REP_COUNTS` is `[3, None]`, a subset of
`[2, 3, 5, None]`; the fast allocation, importance, evaluator and model lists are subsets of
the full ones. So every row a fast pass writes is read again by the full pass
(`FRAMEWORK_FINALIZATION.md:1790`, and `result_cache.py:101` has no speed column). Stage 0
is therefore not wasted.

A fast pass is 3 allocation × 1 importance × 8 budgets × 2 counts × 6 pairs = **288
configurations**, 3.5% of the full 8,160.

---

## 3. The datasets

42 in `DATASET_REPS`. The five trial datasets, checked against the registry 2026-09-14:

| dataset | metric | task | pool | augmentation representations |
|---|---|---|---|---|
| buchwald_hartwig | rmse | regression | 10 | 4 |
| caco2 | rmse | regression | 10 | 5 |
| flab_expression_koenig | spearman | regression | 10 | 3 |
| tcga_brca_pam50 | f1_weighted | classification | 10 | 6 |
| logd | rmse | regression | 10 | 3 |

All five metrics match what you named. All five have all three augmentation conditions.

Six of the 42 have no augmentation representations and get `none` only: abagym_Ang2,
abagym_EGFR, abagym_VEGF, cdr_flexibility, humanness_therapeutic, tz_her2_binary,
tz_her2_multiclass.

Three have fewer than 10 representations in the pool and will be **refused** by
`precompute_hybrids.py:487` unless the pool is completed or the entry is corrected:
ddg_combined (8), nt_H3K4me3 (8), gb_human_enhancers_cohn (9), nt_promoter_tata (9).
*(Refused means: it prints the missing names and writes nothing. It does not build them.)*

---

## 4. What is on ARC — measured 2026-09-14

`python tests/check_rep_pools.py --all`, run on arc-login02 under the venv, cache root
`/data/stat-ecr/scat9264/.cache/kirby/rep_cache`.

**36 of 42 datasets have every representation they need.** All five trial datasets are among
them, with all three augmentation conditions:

| dataset | pool | augmentation | conditions |
|---|---|---|---|
| buchwald_hartwig | 10 of 10 | 4 of 4 | none, uncapped, capped |
| caco2 | 10 of 10 | 5 of 5 | none, uncapped, capped |
| flab_expression_koenig | 10 of 10 | 3 of 3 | none, uncapped, capped |
| tcga_brca_pam50 | 10 of 10 | 6 of 6 | none, uncapped, capped |
| logd | 10 of 10 | 3 of 3 | none, uncapped, capped |

No dataset on ARC has a *partly* built augmentation set — each one is complete or entirely
absent. That matters: a partial set would change `aug_names`, which is a row column, so the
precompute would drop the augmented conditions while the experiments kept asking for them
under a different name.

### 4.1 D11 — the six that are not ready. OPEN, yours.

| dataset | pool on disk | what it needs | note |
|---|---|---|---|
| suzuki | 9 of 10 | `cgr_morgan` | one representation, CPU |
| qm9 | 10 of 10 | 4 augmentation representations | runs `none` today |
| kiba | 0 of 10 | 5 drug + 5 protein representations | protein ones are GPU |
| bindingdb_kd | 0 of 10 | 5 drug + 5 protein representations | protein ones are GPU |
| davis | 0 of 16 | 9 drug + 7 protein representations | protein ones are GPU |
| chembl_qsar | 0 of 10 | — | not reachable, see below |

**suzuki is one representation.** `cgr_morgan` is the only name absent, and
`buchwald_hartwig` already has it, so the builder works. `python tests/precompute_reps.py
suzuki` builds it and nothing else. suzuki was added to the roster on 2026-09-12 because
`opig_cgr_comparison` loads all three reaction datasets and only two were being precomputed.
Cheap — do it.

**qm9 runs `none` today and nothing asks it for more**, because `load_data` returns no
augmentation representations, so `_pool_conditions` never builds the augmented members. The
four absent names are `top_torsion`, `pharmacophore3d`, `usrcat`, `coulomb_matrix`. Three of
the four need 3D conformers, which is the expensive part on a large molecule set.

**chembl_qsar cannot be precomputed at all.** Read 2026-09-14: the name appears nowhere in
`tests/experiments/_shared.py`, so `load_data` has no path to it. `opig_activity_cliff_qsar`
builds its own representations through `kirby.datasets.chembl_qsar`. It is in `DATASET_REPS`
and is not in the submit roster. Nothing to do — but `check_rep_pools.py` will keep reporting
it as refused, which is accurate.

**kiba, bindingdb_kd and davis have nothing on disk.** This replaces the question I asked
earlier about their `uncapped` condition — that question does not arise until the
representations exist. `davis` is already excluded from the submit roster with a parked note
(per-drug-mean degeneracy).

So the decision is: **build the protein and drug representations for kiba and bindingdb_kd
(a GPU job, on 30k-row datasets), or drop both from this run.** Ten representations each,
five of them protein language models. I have not measured what that costs and will not guess
— if you want the number, the way to get it is `python tests/precompute_reps.py kiba
--dry-run` and then one representation timed.

Whichever way that goes, the `uncapped` mismatch in §5.2 still has to be closed, because the
read side asks for a condition the write side never makes.

## 5. Faults found, 2026-09-14

### 5.1 The similarity pass loses three quarters of its grid — MEASURED

Two things that disagree.

`precompute_hybrids.py` passes `speed` and no representation-count list, so
`find_best_hybrid` takes `DEFAULT_REP_COUNTS`, which is `[2, 3, 5, None]`
(`find_best_hybrid.py:7764-7766`). Every configuration asking for a count below the pool size
needs that fold's per-representation allocation scores to decide which representations to
keep.

Under `objective='similarity'` those scores are never computed. `_skip_alloc_precompute` is
true for a geometry objective (`find_best_hybrid.py:5944-5955`), so `alloc_cache` stays
empty, and `_evaluate_hybrid_config` raises (`find_best_hybrid.py:4412-4419`).

So on the similarity objective, three of the four representation counts fail and only the
whole-pool count is stored.

Measured, not read. A 3-representation synthetic set, one budget, one model, two folds:

| objective | counts asked for | configurations stored | counts present |
|---|---|---|---|
| predictive | 2, 3, whole pool | 2 | 2, whole pool |
| similarity | 2, 3, whole pool | 1 | whole pool |
| similarity | whole pool | 1 | whole pool |

The failures are caught and printed, one line per configuration, so the job exits 0 and the
log says the search succeeded. On a 10-representation pool at full speed that is 6,120
printed failures per condition out of 8,160 configurations.

This hits two of the four passes in `submit_precompute_extra_passes.sh`: `sim`
(`--objectives predictive,similarity`) and `knn` (`--with-knn --objectives similarity`).

**FIXED 2026-09-14.** `precompute_hybrids.py` gains `--rep-counts`, taking a list where
`all` means the whole pool, and **forces the whole pool alone whenever the objective is not
predictive**. A predictive run is unchanged: with no `--rep-counts` it still takes the
speed's list. The axis is not cut for compute — it has no meaning without a per-fold ranking,
and *undefined* is one of the three stated reasons.

Naming counts below the pool size together with a non-predictive objective now prints, before
any work starts, that the predictive runs will use them and the other runs will not.

### 5.2 kiba and bindingdb_kd — the read side asks for a condition the write side never makes

`precompute_hybrids.py:513-527` drops the `uncapped` condition for these two on a 2026-06-21
cost decision. `_shared.py:5274` `_pool_conditions` builds `none`, `uncapped` and `capped`
for any dataset with augmentation representations, with no exceptions. Both names are in
`DATASET_REPS` with 10 pool representations and 3 augmentation representations.

So every experiment that reads either dataset asks for an uncapped search that was never
written. It misses, and because `aug_strategy` is a row column, the rows are not there either
— it re-fits 8,160 configurations × 5 folds inside its own wall.

Both are large: the note in the code says `general/none` alone was about 4 days on
bindingdb_kd.

Three ways out, pick one:

1. **Run the uncapped pass for both.** Costs roughly what `none` cost, times the augmentation
   overhead. No experiment changes.
2. **Add both names to `_pool_conditions` as exceptions**, so the read side stops asking. One
   line. Every experiment on those two datasets then runs on `none` and `capped` only, and
   the uncapped arm is missing from their results — stated, not silent.
3. **Drop both datasets from this run.** They are 2 of 42.

My recommendation is 2. It is the only one that costs nothing and cannot produce a wrong
number; the missing arm is visible in the output. But it is a cut to the search space on two
datasets, so it is yours.

### 5.3 A no-augmentation job was building augmentation representations — MEASURED, FIXED

Two things that disagree.

`precompute_dataset` asks `load_data` for the pool representations and the augmentation
representations together, and it does that **before** `--condition` filters the conditions
down (`precompute_hybrids.py:436-441`). `load_data` does not fail on a representation it
cannot find — `_shared._cache_rep` computes it and writes it to the cache.

So a job filtered to the no-augmentation condition asks for representations that condition
never uses, and pays to build any of them that are absent, inside its own wall.

Measured 2026-09-14, not read. `precompute_hybrids.py caco2 --condition general_none` against
a cache holding 1 of caco2's 5 augmentation representations printed
`Computing 4 missing rep(s): ['coulomb_matrix', 'pharmacophore3d', 'usr', 'usrcat']` and
began generating 3D conformers for 2,158 molecules. Three of those four need conformers.

**On ARC today this fires on qm9**, which has all 10 pool representations and none of its 4
augmentation representations — and three of those four need conformers too, on a much larger
molecule set.

**FIXED.** A `--condition general_none` or `--condition general_distance_none` job no longer
names the augmentation representations at all, and says so in its first line. This is a scope
fix, not a cut: the augmented conditions still load them and still run.

---

## 6. The checks

### 6.1 D2 — representation preflight, BUILT

`tests/check_rep_pools.py`. File existence only: no dataset is loaded, no
representation is built, no model is fitted. It takes about three seconds and is safe on a
login node.

```bash
cd /data/stat-ecr/scat9264/KIRBy
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
python tests/check_rep_pools.py --trial
python tests/check_rep_pools.py --all --missing-only
```

It exits 1 when any dataset would be refused, so a submit script can gate on it.

**Why it prints the environment first, and why that is not decoration.** The
representations live under `$KIRBY_CACHE_DIR`, which `setup.sh:60` sets to
`/data/stat-ecr/scat9264/.cache/kirby`. Run any check without sourcing `setup.sh` and the
variable is unset, the path falls back to `~/.cache/kirby` (`_shared.py:249`), every file is
absent, and the answer is "10 of 10 missing" on every dataset. **That is a wrong answer, not
an error.** So the script prints the cache root and whether it was inherited or defaulted
before it prints a single result, warns when the variable is unset, and refuses outright when
the root does not exist.

Demonstrated on this laptop 2026-09-14, same five datasets, two cache roots, neither run
raising:

| dataset | under `~/.cache/kirby` | under `/Volumes/seagate/kirby_cache` |
|---|---|---|
| buchwald_hartwig | none only | none, uncapped, capped |
| caco2 | none, uncapped, capped | none only |
| flab_expression_koenig | none, uncapped, capped | none only |
| tcga_brca_pam50 | none only | none only |
| logd | **REFUSED**, 8 of 10 pool representations absent | none, uncapped, capped |

Four of the five disagree. This is the failure mode the script exists to make visible.

### 6.1b What the environment actually is

Read from `setup.sh` 2026-09-14.

- **`setup.sh` must be sourced, not executed.** It activates a virtual environment and
  exports variables; it ends with `return 0 2>/dev/null || exit 0`. Use `. ./setup.sh`.
- **The interpreter is a virtual environment**, `/data/stat-cadd/scat9264/py311-kirby`
  (Python 3.11, `setup.sh:37`). Without it `python` is the system Anaconda, and `numpy`,
  `rdkit` and `kirby` are either absent or the wrong versions. The preflight needs `numpy`
  and `rdkit`; it does not pull `torch`.
- **`KIRBY_SKIP_INSTALLS=1` before sourcing** returns early after loading CUDA
  (`setup.sh:105-118`). Without it, `setup.sh` runs pip checks over about thirty packages,
  fast-forwards the `noise_inject` checkout, and can build HMMER 3.4 from source — roughly
  three minutes, network access, and not something to do on a login node or repeatedly inside
  an `srun`. Every precompute job should set it. `submit_precompute_extra_passes.sh:64`
  already does.
- **Reads and writes are split across two project areas**: HuggingFace, torch and pip caches
  are read-only on `/data/stat-cadd`; `KIRBY_CACHE_DIR` writes to `/data/stat-ecr`
  (`setup.sh:50-62`, after the stat-cadd quota hit 99.9% on 2026-05-07).
- **Threads are split on purpose** (`setup.sh:17-23`): `OMP_NUM_THREADS` takes the whole
  allocation because xgboost and lightgbm parallelise through OpenMP, while BLAS, MKL and
  numba are pinned to 1 so they do not multiply underneath joblib. A job that sets its own
  thread count after sourcing `setup.sh` undoes this.

**So the first line of every job and every interactive session is the same three commands:**

```bash
cd /data/stat-ecr/scat9264/KIRBy
export KIRBY_SKIP_INSTALLS=1
. ./setup.sh
```

⛔ And check the branch. `submit_dataset.sh:51` pulls `origin main`; this work is on
`similarity-metrics-study`. `git rev-parse --abbrev-ref HEAD` before anything else.

### 6.2 D9 — progress, BUILT

Two scripts, and they answer different questions.

`tests/check_precompute_status.py` (already existed) asks whether a dataset has **at least one
row** for each (tier, augmentation strategy). That was the right unit when a condition was the
unit. It is not any more: one pass is 8,160 configurations, so a dataset with 1 of them reads
as done. It also knows nothing about the objective column, the model list or the
representation count.

`tests/precompute_coverage.py` (new) reads `result_db/<dataset>.db` read-only and reports, per
pass, **configurations stored against configurations the full grid asks for**, the median
seconds a configuration actually took (off the `time_seconds` column the engine writes on
every row), and therefore the core-hours left.

```bash
python tests/precompute_coverage.py --trial
python tests/precompute_coverage.py --all --incomplete-only
```

It builds the expected count from the engine's own lists rather than from a number written
down anywhere, using the same arithmetic as §2.5, so the two cannot drift apart.

Run against this laptop's cache 2026-09-14, on two datasets an earlier session had partly
precomputed — this is the report working, not a result:

| dataset | pass | stored | asked | median seconds per configuration |
|---|---|---|---|---|
| tcga_brca_pam50 | general / predictive / none | 5,190 | 8,160 | 7.3 |
| logd | general / predictive / none | 4,350 | 8,160 | 16.3 |

The expected count it computes, 8,160, is the one §2.5 derives independently.

It also found a third thing: `buchwald_hartwig.db` on this laptop has no `results` table at
all. That is a database written under an older column list or an interrupted create. There is
no migration by design — it has to be deleted before that dataset is precomputed.

### 6.3 D10 — a refusal switch, NOT BUILT

There is no environment variable that makes a search refuse to build when nothing is
precomputed. The variables that exist are `KIRBY_CACHE_DIR`, `KIRBY_CACHE_VERBOSE`,
`KIRBY_DISTANCE_METRIC`, `KIRBY_EXPENSIVE_FIRST`, `KIRBY_POOL_VERBOSE`,
`KIRBY_PRECOMPUTE_EXPENSIVE_FIRST`, `KIRBY_RESULT_DB`, `KIRBY_SMOKE_N`.

Proposed: `KIRBY_NO_COLD_SEARCH=1` makes `find_best_hybrid` raise when `cache=True`, the
pickle misses, and the row database holds fewer than a stated fraction of the configurations
asked for. An experiment job then fails in minutes instead of burning seven days.

---

## 7. Stage 0 — the trial precompute, in `srun`

**What it is for.** To catch the faults before a queue wait, and to measure seconds per
configuration so stages 1 and 2 can be sized from a number rather than a guess.

**It is not throwaway.** `speed` is in the search pickle's key and not in the row database's,
and every fast list is a subset of its full one, so every row this writes is read again by
stage 1. Rows are flushed every 25 configurations (`find_best_hybrid.py:5418`), so an
interrupted run keeps everything it finished.

```bash
# 1. a shell. Take the exact line where_to_submit.sh section 6 prints — it only
#    offers a partition whose nodes can actually start a job.
bash tests/slurm_scripts/where_to_submit.sh
srun -p interactive --cpus-per-task=8 --mem=32G --time=4:00:00 --pty /bin/bash

# 2. environment, then prove it
cd /data/stat-ecr/scat9264/KIRBy
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
git rev-parse --abbrev-ref HEAD          # similarity-metrics-study
python tests/check_rep_pools.py --trial  # 5 of 5, all three conditions

# 3. the three lineages, smallest grid that exercises each. -u so the log
#    shows progress: without it Python buffers and the job looks hung.
python -u tests/precompute_hybrids.py caco2 --condition general_none --speed fast
python -u tests/precompute_hybrids.py caco2 --condition general_none --speed fast \
        --objectives similarity
python -u tests/precompute_hybrids.py caco2 --condition general_distance_none \
        --speed fast --distance

# 4. what it cost, and what the full grid will therefore cost
python tests/precompute_coverage.py caco2
```

**What to look for in the output, in order:**

1. `--condition general_none: not loading the N augmentation representations` — §5.3's fix.
2. `objective=similarity: representation counts forced to the whole pool` — §5.1's fix, on
   the second command only.
3. **No line matching `Computing N missing rep(s)`.** One means a representation is being
   built inside the job. Stop it and build the representation separately.
4. **No `FAILED` line.** Before §5.1 the similarity command produced one per configuration.
5. The exit code. `precompute_hybrids.py` exits 1 when anything asked for was not written; it
   used to swallow every error and exit 0.

**Why caco2 first.** 2,158 molecules, all 10 pool representations and all 5 augmentation
representations on ARC, and the cheapest of the five to measure on. `tcga_brca_pam50` is
smaller in rows (900) but its representations are far wider.

A fast pass is 3 allocation × 1 importance × 8 budgets × 2 counts × 6 (model, evaluator)
pairs = **288 configurations**, which is 3.5% of the full 8,160.

## 8. Stage 1 — precompute the trial hybrids

Five datasets, full speed. Commands go here.

---

## 9. Stage 2 — precompute all 42 datasets

Commands go here.

---

## 10. Stage 3 — trial experiments, and the tuned arm

Commands go here.

⛔ `tune` is an identity column of the row database (`result_cache.py:80-86`). A tuned run
shares no row with an untuned one. The tuned arm is a second precompute of the same size, not
a variation on the first.

---

## 11. Stage 4 — all experiments

Commands go here.

---

## 12. D12 — memory and wall

**Not guesses.** The method:

**Memory.** `where_to_submit.sh` section 3 prints GB per core for each partition. Ask for
`cores × that ratio` and no more. A request above the ratio reserves cores the job never uses
and waits for a whole node — that is what left f28 job 12923011 pending for a week at 16
cores and 256 GB. Check a shape before submitting:
`bash tests/slurm_scripts/where_to_submit.sh --fits 8 64G`. After the first job finishes,
read the real number off `sacct -j <id> --format=JobID,MaxRSS,Elapsed` and size the rest from
it.

**Wall.** Stage 0 measures seconds per configuration on each trial dataset. Wall = that
number × the configuration count from §2.5, plus 30%. A 7-day request cannot backfill into a
short idle gap; a shorter one starts sooner.

Measured numbers go here after stage 0.
