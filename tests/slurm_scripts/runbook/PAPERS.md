# Running Papers 2 and 3 on the server: one job script, one ledger

Since 2026-11-03. Built on the same ledger code as the screening ledger (`SCREEN.md`):
`tests/experiments/orchestrate/ledger.py`. A job takes the next dataset that has work and is
not held, runs one unit at a time, records each one, and moves on. Queued jobs and sruns
record into the same file.

The day-to-day commands for this ledger and the screening ledger are in `RUN_BOTH_LEDGERS.md`.

The old experiments ledger (run_2026_10_06) is no longer run. c07 and c15 were in its plan.
Paper 2 is written from their results that already exist; neither is rerun (`PAPER_2_OUTLINE.md`).

## Files in the run folder

| file | what |
|---|---|
| `RUN_SETTINGS.json` | every setting the run depends on, and the gate. Written by `init` and rewritten by `replan`; otherwise only you edit it |
| `ledger.json` | the state; every change goes through a lock on `ledger.lock` |
| `PAPERS.md` | the document: one row per dataset. Each step shows units `ok` out of units planned, then any running and failed counts, and the time summed over its units. "(held)" follows when the whole step is held and not finished. Why a step is held is in the last column. A unit held on its own, such as a meta-learner unit or an activity-cliff unit waiting for a setting in `CLIFF_SETTINGS.json`, does not add "(held)" to its step. A step with no units, or one missing from a run started before it existed, shows "–". A skipped unit is not counted as `ok`. `ledger.py progress` adds skipped units to the `ok` units in each step's count, so only its per-step counts can differ from the `ok` counts in the run's `PAPERS.md` |
| `events.jsonl` | one line per change |
| `logs/<dataset>__<step>__<unit>.log` | each unit's output |
| `cache/` | this run's own `KIRBY_CACHE_DIR` (below) |
| `complexity/ similarity/ noise/ cliffs/ meta_learner/` | the experiments' result files |

`tests/results/CURRENT_PAPERS` names the live run. The old `CURRENT_RUN` and the screen's
`CURRENT_SCREEN` are never read or written.

## The caches

Each of the three search steps has its own cache and starts with an empty score store, so no search step reuses another step's scores. The three caches are
`<run>/cache_pruned`, `<run>/cache_fast` and `<run>/cache_full`. The hours recorded for one fold cover every search the step runs on that fold, together. `<run>/cache` is what the
representation step uses.
- **These start empty in each:** search results (`result_db`, `searches`), baselines, fold
  scores and checkpoints. The list is `FRESH_CACHE_ENTRIES` in `tests/experiments/papers/ledger.py`.
- **These are links into the thesis cache** (`/data/stat-ecr/scat9264/.cache/kirby`):
  representation files and downloaded data.
- **These are not linked at all:** patient data, and leftovers no search reads (old checkpoints
  at the top of the cache, stale or trashed copies).
- **The thesis results are never touched.** The one write that reaches the thesis cache is a
  representation build adding a file that was missing.

## Steps per dataset, in order

| step | a unit is | starts when |
|---|---|---|
| reps | `tests/precompute_reps.py DATASET --include-augmentation`: builds the missing representation files and checks the rest | at once |
| search_pruned | one outer fold, full search space, calibrated pruning | reps finished without failure, the step is open, and the dataset's two models are recorded |
| search_fast | one outer fold, the engine's fast tier | same |
| search_full | one outer fold, full search space, no pruning | same |
| paper2 | one experiment of the Paper 2 run list. The meta-learner has one unit per outer fold, `meta_f0` to `meta_f4`, each also held until the dataset is named (below) | the step is open, `experiments_search` is set, and that search's folds are ok on the dataset |
| paper3 | one experiment (or experiment x noise condition) of the Paper 3 run list, or one unit of the activity-cliff study (the `cliffs` category, `PAPER3_EXPERIMENTS` in `tests/experiments/papers/plan.py`). Cliff units run on molecular and antibody datasets only | same |

**Not in the ledger.** c03, s09 and every OPIG script on the two run lists have no unit. They
read several datasets, and `tests/experiments/papers/plan.py` marks them "not planned yet". So
no job runs them. The OPIG scripts are:
- Paper 2: opig_blueprint_transfer, opig_cgr_comparison, opig_generator_scoring_swap,
  opig_modnet_feature_scores, opig_scaffold_hybrid, opig_shared_vectors_census,
  opig_shared_vectors_remedy, opig_trastuzumab_enrichment.
- Paper 3: opig_activity_cliff_qsar, opig_moleculeace_cliff_check.

**The noise grid.** It follows `PAPER_3_NOISE_EXPERIMENTS.md` (Adelaide, 2026-11-04). In the experiment that pairs every
detector with every cleanup (n44), the standard cleanup run has no Cliff put-back, Random put-back, shuffled cleanup results or shuffled detector scores. The variant cleanups run adds those checks only for the results you choose. It starts only when `KIRBY_N44_SECOND_RUN` names a JSON file listing those results. The same file must give `cliff_definitions`: "all" or a list of definition keys. Nothing in this ledger sets that variable. A worker passes its shell environment on to each unit it starts. So if the variable is exported in a job, every n44 unit that job runs is a variant cleanups run and writes `__second_run` files.

A search unit runs `orchestrate/search_unit.py DATASET --fold k`, which runs the three kinds
of search the experiments read back: the general search, the kNN search and
the similarity search.

**What the general search in each search step holds** (`decided_grid.full_space_lists`, `fast_space_lists`). The similarity search in the same step is called with these lists too (`tier='general'`). The dataset's
two models come from DECISIONS.md through `decided_grid.json`; nothing else is read from that file.

| setting | full and pruned | fast |
|---|---|---|
| budgets, in features | 10, 25, 50, 100, 250, 500, 1000, 2000 | same |
| importance methods | deeplift, lightgbm_gain, treeshap | lightgbm_gain |
| allocation methods | performance_weighted, power_2, softmax_T1, rank_exp_d0.3, wglobal_power2 | performance_weighted, power_2, rank_exp_d0.3 |
| internal evaluators | match, rf, lightgbm, dnn | same |
| representation counts | 2, 3, 5, all | 3, all |

The kNN search runs budgets of 10 to 500 features in the `search_pruned`, `search_fast` and `search_full` steps, with knn as its only downstream model and its only internal evaluator.

What pruning dropped, the folds it voted on, and the configurations before and after are in
`cache_pruned/searches/<search name>.pruning.json`.

**Why the full space, not the search grid.** "Every dataset searches the full space, with
adaptive pruning on" (Adelaide, 2026-11-02). The search grid in `decided_grid.json` was cut by
hand per dataset. It cuts five settings: budgets, importance methods, allocation methods, internal
evaluators and representation counts. Two datasets cut differently differ in several ways at
once, so they cannot be compared. Pruning does not fix that. It drops settings only after folds 0
and 1 are scored. What a search starts from must be the same on every dataset.

"Adaptive pruning is not being re-tuned now (Adelaide, 2026-11-01). She will check at the Step 3
gate that it works as intended." In this ledger that gate is the choice of `experiments_search`.

## The gate and the holds

`RUN_SETTINGS.json` has an `open` block with one entry per step past reps. A step that is not
open never starts, on any dataset, even when named by hand with `--unit`. Set one to `true` and
the next claim starts it; nothing has to be restarted. `experiments_search` is `pruned`, `fast`
or `full`: the search the experiments read, chosen once you have compared them.

`future_work` lists datasets taken out at the gate. Their units not yet ok are held, not deleted.

A dataset with no row in `decided_grid.json` is held at every step after reps, because its two
models are not recorded. Rebuild the file (`python tests/experiments/build_decided_grid.py`) and
the next claim releases it. A dataset whose representation build failed for good is also held.

Some activity-cliff units also wait for settings in `tests/experiments/cliffs/CLIFF_SETTINGS.json`.
Such a unit is held until every setting it needs has a value there. Writing the file releases it
at the next claim.

## The meta-learner in the paper2 step

Paper 2's meta-learner over models with their own hybrids, `tests/experiments/papers/meta_learner_own_hybrids.py`.
Opening paper2 does not start it: its units also wait for the `meta_learner` block in
`RUN_SETTINGS.json`, and run only on datasets you name there.

| key | what | starts as |
|---|---|---|
| `datasets` | the datasets it runs on, chosen from results and runtime | empty |
| `candidate_models` | every model searched, per outer fold; the top 5 are taken on the fold's training samples | rf, dnn, lightgbm, svm, gp, xgboost, ngboost |
| `net_sizes` | nets added to the candidates: `dnn_small` is (64, 32), `dnn_wide` is (512, 256, 128) | both |
| `inner_folds` | folds inside the training folds, for the predictions the meta-learner trains on; every fit uses the model's one fixed feature set | 5 |
| `min_rows` | a named dataset with fewer samples writes a skipped result and fits nothing | 500 |

Add a name to `datasets` and the next claim starts it; nothing has to be restarted. To time one
fold before committing to a dataset, name it, then run
`python tests/experiments/papers/worker.py --dataset herg_ki --unit meta_f0`. The unit's hours are
printed by `python tests/experiments/papers/ledger.py show herg_ki`, and each candidate's search seconds are in its result file.

Because a dataset you do not name keeps these units held, the paper2 step in its row of `PAPERS.md` never shows all units
`ok`. It stays at least 5 units short of the units planned, and that is not a failure.

Read the comparison with `python tests/experiments/papers/meta_learner_summary.py`: one line per dataset
and outer fold. Choose datasets from c02's results and the search hours.

Results: `<run>/meta_learner/<dataset>__meta_f<k>.json`: each candidate's search score and seconds,
the top models, each model's held-out score, and for every meta-learner its held-out score and its
cross-validated score on the training samples. The one that counts has the best cross-validated score on the training folds. Predictions
are in `<dataset>__meta_f<k>.predictions/`, so a killed job resumes, and adding a meta-learner fits
no downstream model again (reset the unit to rerun it).

Candidates outside the dataset's two recorded models have their own search files, named
`meta_<model>`, so they do not overwrite the two-model search the experiments read. The two recorded
models read the ledger's search. Under `pruned`, each extra candidate's pruning votes on that model
alone, while the two recorded models' pruning voted on both.

A run started before 2026-11-05 has none of this. `ledger.py extend --all` adds the units to its
paper2 step and changes no setting and no finished unit. Then add the `meta_learner` block to that
run's `RUN_SETTINGS.json` by hand. Do not use `replan` for this: it closes every step past reps.

## Never on htc-g011 (2026-11-02)

Every htc job and srun carries `--exclude=htc-g011`. On that node jobs are stopped and restarted
about every four hours. On run_2026_10_06, the searches of flab_binding_adams and
flab_expression_adams ran there on folds 0 and 3, and neither finished in 36 hours. The same two
datasets' folds 1 and 2, on ordinary nodes, reached 802 of 840 and 1,003 of 1,960 configurations
in 22.1 hours. `queue_ledgers.py` and `submit_papers_workers.sh` add the flag on htc. Nothing
else checks the node, so an srun line must carry it by hand.

## The commands

Every command, and what to do when a unit fails, is in `RUN_BOTH_LEDGERS.md`.
