# Handoff: a new server ledger for Papers 2 and 3

Written 2026-11-01. Adelaide will run this once the OPIG, similarity and noise
experiments have been sorted into the two paper outlines. By then the outlines' run
lists are final. Take the experiment list and the dataset lists from the outlines, not
from this file.

Paste everything below the line into a fresh Claude Code chat in `~/repos/KIRBy`.

---

Read `~/repos/KIRBy/CLAUDE.md` first and follow it in every reply. Then read `MEMORY.md`
in the memory folder. Adelaide is the engineer and decides. These results go into two
papers for publication, so every setting must be recorded and reproducible from a file.

Your first deliverable is a design for the new ledger, plus the open questions at the
end, for her to sign off. It is not code. Bring at most three decisions at once.

## The two papers

- **Paper 2** (`PAPER_2_OUTLINE.md`) is a general machine-learning methods paper. It
  uses every dataset: small molecules, antibodies, nucleotides, reactions, and the
  benchmarks in `NEW_DATASETS.md`.
- **Paper 3** (`PAPER_3_OUTLINE.md`) is for a chemistry journal. It covers detecting and
  handling noisy labels. It uses every small-molecule dataset and a subset of the
  antibody datasets. Its outline lists them under "Datasets (Adelaide, 2026-11-01)".
- Check both lists against each other. Where a dataset appears in only one paper,
  record which experiments it needs.

## Why everything runs again

- The thesis runs used settings chosen by hand to meet the thesis deadline.
- Each dataset's search grid was cut by hand. The cut is recorded in
  `tests/experiments/decided_grid.json`, so it cannot be reproduced from a rule.
- Many experiments have changed since those runs.

So every search and every experiment runs again, from scratch, for both papers.

## Build on the existing ledger

The old ledger worked well. Every server job pointed at one entry file. That file gave
each job the next dataset in the queue. Keep this design. Change the plan it runs: the
stages, their order, what each stage runs, and one gate where Adelaide decides.

Read all of these in full before proposing anything:

- `tests/slurm_scripts/runbook/WORKERS.md`. It covers one job script, one ledger,
  stages per dataset, memory classes and the commands.
- `tests/experiments/orchestrate/`: `plan.py`, `ledger.py`, `runner.py`,
  `search_unit.py`, `sanity.py`.
- `tests/experiments/worker.py`, `tests/slurm_scripts/worker.sh`,
  `tests/slurm_scripts/submit_workers.sh`.
- `setup.sh`, function `_kirby_ledger_redirect`. It sends older job names to
  `worker.py --from-job-name NAME --then-any`.
- `tests/experiments/run_dataset.py`, function `_ledger_hook`.
- `tests/test_orchestrate_ledger.py`.
- The screening ledger, built 2026-11-01 on the same code:
  `tests/experiments/screen/{ledger,worker,datasets}.py`,
  `tests/slurm_scripts/runbook/SCREEN.md`, `tests/test_screen_ledger.py`.
- `HANDOFF_LEDGER_CUTS.md` (2026-10-09). It records how the last ledger was cut down to
  the thesis.
- `REPRODUCIBILITY_LEDGER.md`, `RUN_COMMANDS.md`, `TIME_ESTIMATION_METHOD.md`.

Also read what is going to run:

- Both outlines, including their run lists and inventories.
- `PAPER_EXPERIMENT_TABLES.md`: one row per experiment, dated about 2026-10-10.
- `NEW_DATASETS.md`.
- `HANDOFF_NOISE_RESTRUCTURE.md`. The noise experiments are now n43 to n47, plus n15,
  n20 and n36.
- `HANDOFF_PAPER3_CLIFFS.md`. opig_activity_cliff_qsar moved to Paper 3.

## Where each dataset's settings are recorded

- **Two downstream models per dataset:** `scratchpad/precompute_cuts/DECISIONS.md`.
- **The JSON the code reads:** `tests/experiments/build_decided_grid.py` builds
  `tests/experiments/decided_grid.json` from DECISIONS.md and `stored_sets.json`. The
  JSON is generated; never edit it by hand.
- **Representations per dataset (at most 10):** `DATASET_REPS[<dataset>]['general']` in
  `tests/experiments/complexity/complexity_helpers.py`.
- **New datasets:** results and decisions go in `NEW_DATASETS.md` and also in
  DECISIONS.md, section "New datasets (results from NEW_DATASETS.md)".

Check these four against each other and against the outlines' dataset lists. Report every
disagreement as a table, one row per dataset. Adelaide wants one document that names
each dataset's two models. If DECISIONS.md cannot serve as that document, propose one.

## The five steps, in Adelaide's order

### Step 1. Precompute

- Each dataset uses its two designated models and at most 10 representations.
- **One exception (Adelaide, 2026-11-02): the reaction datasets may carry more than 10.**
  Buchwald--Hartwig and the Doyle plate run are screened over the seven condensed-graph
  representations beside their existing ten, for the Paper 2 subsection sourced from
  opig_cgr_comparison. Which ones go in waits for the reaction screen (b10).
- Datasets that already have a valid precompute keep it. Work out which those are:
  list each dataset with a stored precompute, and say whether its models and
  representations still match the record.
- New datasets, and datasets whose models or representations changed, run it.
- **Gate for datasets in `NEW_DATASETS.md`.** None of them can start until Adelaide has
  read its screening results and recorded its two models and its representations. Its
  screening results are in the screening ledger's `summary.md` for that dataset. The
  ledger must hold such a dataset until the record exists, and start it automatically
  once it does. Propose how the ledger reads that record.

### Step 2. Hybrid search with adaptive pruning

- **Every dataset searches the full space, with adaptive pruning on (Adelaide,
  2026-11-02).** Not the per-dataset grid in `decided_grid.json`. That file was cut by
  hand for the thesis deadline, and it cuts five things, not one: the budgets, which
  method ranks the features, how the budget is split across representations, the internal
  evaluator and the number of representations. Two datasets cut differently cannot be
  compared. For example hERG Ki may use 250 features, treeshap and softmax_T1 while LogD
  may use none of the three, so any result comparing those two datasets differs in
  several ways at once.
- Adaptive pruning does not fix this. It drops settings partway through a run, after
  folds 0 and 1 are scored. What a run is allowed to start from is a separate thing, and
  that is what must be the same for every dataset.
- So: find where the full search space is defined for each axis, write down what it holds,
  and show it to Adelaide before anything runs. `FULL_BUDGETS` in
  `tests/experiments/_shared.py` is the budget list; find the equivalent for the other
  four axes.
- `decided_grid.json` is then only a record of what the thesis runs used. Nothing in this
  run reads it. Say so in the ledger, because `_apply_dataset_search_speed` in
  `tests/experiments/_shared.py` (around line 1042) calls `apply_decided_grid` and will
  apply it unless that path is changed.
- **Adaptive pruning is not being re-tuned now (Adelaide, 2026-11-01).** She will check
  at the Step 3 gate that it works as intended.
- Facts about the pruning, read in the code on 2026-11-01:
  - The schedule is `DEFAULT_PRUNING_SCHEDULE` in `src/kirby/find_best_hybrid.py`
    (around line 3827). The named modes are `PRUNING_MODES` in
    `tests/experiments/_shared.py` (around line 649). Use the mode named `calibrated`.
  - Pruning waits until folds 0 and 1 are scored. Then it votes separately on three
    settings: number of representations, importance method and internal evaluator.
    Allocation methods are dropped by a separate rule.
  - It was tuned on 2026-09-16 and 09-17 on five datasets. That predates the
    per-dataset search settings of October.
  - One of those five datasets was tcga_brca_pam50. That is patient data. It must not be
    opened, run or cited, so the existing tuning cannot be repeated as it was.
- Record in the ledger, per dataset: what the pruning dropped, at which fold, and how
  many configurations were scored.

### Step 3. The check at one setting per dataset

- From the Step 2 results, take the best budget and model for each dataset. That is
  one search setting per dataset.
- At that setting, run the search twice more: once at the fast tier and once in full,
  with no pruning.
- Find where the fast tier and the full tier are defined (`speed` in
  `find_best_hybrid`, and `tests/experiments/decided_grid.py`). Write down what each
  includes before running.
- Report per dataset, in the dataset's own metric: the held-out score under pruning, the
  fast tier and the full search. Also give the number of configurations scored and the
  hours each took. This shows how much adaptive pruning loses.

### Gate: wait for most datasets

- Wait until most datasets have finished Step 3. A few stragglers can still be running.
- Then give Adelaide the Step 3 table. She decides two things:
  - **How experiments that run their own searches will search.** Choices: adaptive
    pruning, the fast tier, the full search, or settings she picks by hand per dataset.
    List which experiments run their own searches. WORKERS.md names c06, c07, c13 and
    c23. Check the Paper 2 and Paper 3 run lists for others, for example
    opig_blueprint_transfer.
  - **Which datasets move to future work.**
- Nothing after the gate starts until she has recorded both decisions in a file the
  ledger reads.

### Step 4. Paper 2 experiments

Run every experiment in `PAPER_2_OUTLINE.md`'s run list, on its datasets.

### Step 5. Paper 3 experiments

Run every experiment in `PAPER_3_OUTLINE.md`'s run list, on its datasets.

The noise experiments' grid follows `PAPER_3_NOISE_EXPERIMENTS.md` (Adelaide, 2026-11-04). Its first run has no
put-back and no twins; a second run adds those checks only for the results she chooses. So this
ledger needs two separate steps for those experiments. Its code changes are in
`HANDOFF_PAPER3_NOISE_IMPLEMENTATION.md`, and her results review and the cliff grid are still open.

## Rules that apply throughout

- Never ssh to the server or copy files from it. Adelaide runs every server command.
  Give her commands in the format in memory: one `srun ... --pty /bin/bash` line, then
  plain commands. Check memory for how she launches the workers.
- Never put `rm` in a block of commands she will paste.
- Never propose cancelling queued jobs; she keeps them for their place in the queue.
- Large files, logs and temp files go on `/Volumes/seagate`, never `/tmp`.
- No patient data: TCGA, ANNITIA and NHANES are never opened, run or counted.
- Do not cut a dataset, budget, model or setting to save compute. A cut needs a stated
  reason: the setting is degenerate, broken or undefined. Adelaide makes cuts at the gate.
- No significance tests. Report scores per fold and their spread.
- Every setting a result depends on is written to a file the ledger reads. Nothing is
  set by hand at the command line without being recorded.
- Expect many reruns. A unit must be resettable and rerunnable on its own, without
  touching the units around it. The old ledger's `reset` command did this; keep it.
- Post a short line every few minutes during long work. Her terminal disconnects.

## What to bring back first

A design document, not code. It holds:

1. One table per paper: one row per dataset, one column per stage, each cell saying
   whether that dataset needs that stage and what blocks it.
2. The stage list and their order, as a change from the old plan in
   `WORKERS.md`. Say which old stages are gone, which are new, and which are renamed.
3. The disagreement table from "Where each dataset's settings are recorded".
4. The list of experiments that run their own searches.
5. How the ledger holds a `NEW_DATASETS.md` dataset until its two models and
   representations are recorded, and how it starts once they are.
6. How the gate is enforced in the code, so Steps 4 and 5 cannot start early.
7. An estimate of hours per dataset for Steps 1 to 3, by the method in
   `TIME_ESTIMATION_METHOD.md`, and which datasets need the 128 GB memory class.

## Open questions for Adelaide

Ask these before building. Three at a time, each with the numbers attached.

1. Her two papers overlap on most datasets but not all. Should the ledger hold one run
   covering both papers, with each experiment tagged by paper, or one run per paper?
2. The standard budget list for Step 2: show her what the code's list is and ask whether
   that is the list she means.
3. Which datasets already have a valid precompute, and does she agree they can be kept
   rather than rerun?
4. At the gate, what counts as "most datasets" finished: a number, or her own look at
   the table?
5. How many folds does each stage run? The old ledger ran a one-fold screen for the
   noise stage and five folds elsewhere, and `OUTER_FOLDS_RUN` in
   `tests/experiments/_shared.py` held a per-experiment fold count that was on hold from
   2026-10-07.
6. Does the old run root (`tests/results/<run>/`) and pointer file
   (`tests/results/CURRENT_RUN`) stay, or does this run live somewhere else? Her setup
   changed since the old ledger.
