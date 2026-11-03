# Handoff: check that every dataset decision is implemented in the code

Written 2026-11-03. The step this covers is the one that has been missing: between a
decision in `scratchpad/precompute_cuts/DECISIONS.md` and the Paper 2 and Paper 3
experiment ledger, nothing checks that the decision reached the code.

Paste everything below the line into a fresh Claude Code chat in `~/repos/KIRBy`.

---

Read `~/repos/KIRBy/CLAUDE.md` first and follow it in every reply. Then read `MEMORY.md`
in the memory folder. Adelaide is the engineer and decides. Bring at most three decisions
at once.

Your deliverable is one table and a list of code changes, for her to sign off. It is not
code until she says so.

## The order this project works in

1. A candidate dataset is written up in `NEW_DATASETS.md`.
2. The screening ledger scores every candidate feature set on it
   (`tests/slurm_scripts/runbook/SCREEN.md`, `tests/experiments/screen/`).
3. Adelaide reads the screen and decides whether the dataset is in, and which feature sets
   and which two downstream models it gets.
4. That decision is written in `scratchpad/precompute_cuts/DECISIONS.md` **and implemented
   in the code**.
5. The Paper 2 and Paper 3 experiment ledger plans the dataset
   (`HANDOFF_PAPERS_2_3_LEDGER.md`).
6. The paper reports it.

Step 4's second half is what is missing. A decision written in DECISIONS.md changes
nothing on its own: no code reads that file except
`tests/experiments/build_decided_grid.py`, which builds
`tests/experiments/decided_grid.json` from it.

## What "implemented" means, per dataset

Six things. Each is a file you can open and check.

1. **Registry entry.** `DATASET_REGISTRY_DATA` in `tests/experiments/dataset_policy.py`,
   giving the task and the metric.
2. **Loader.** `_shared.load_data` loads it under that exact name.
3. **Feature sets.** `DATASET_REPS[<dataset>]['general']` in
   `tests/experiments/complexity/complexity_helpers.py`, at most ten, except the reaction
   datasets (Adelaide, 2026-11-02).
4. **Two downstream models.** Recorded in DECISIONS.md, section "Step 1". Find where, if
   anywhere, the code reads that pair, and say so. If nothing reads it, that is the
   finding.
5. **Fold grouping.** For antibodies, `DATASET_REGISTRY` in
   `tests/antibody_baseline_survey.py` maps the dataset name to a grouping strategy, and
   the strategies are built further down the same file. For molecules, `_shared.load_data`
   assigns scaffold groups.
6. **Planned by a ledger.** `orchestrate/plan.py::dataset_roster()` returns it, and
   `plan.CUT_DATASETS` does not hold it.

`tests/experiments/decided_grid.json` is a seventh thing, but read the Step 2 section of
`HANDOFF_PAPERS_2_3_LEDGER.md` first: every dataset now searches the full space, so that
file is only a record of what the thesis runs used. One experiment, c11, still refuses a
dataset that has no entry in it. Check whether c11 is in either paper's run list.

## What to bring back

**One table. One row per dataset, one column per item above.** The datasets are the union
of the Paper 2 and Paper 3 dataset lists in `PAPER_2_OUTLINE.md` and `PAPER_3_OUTLINE.md`.
Add two columns saying whether the dataset is in Paper 2 and whether it is in Paper 3.

**Then a list of code changes**, one line each, in the order she should approve them. Each
line says the file, what changes, and which decision in DECISIONS.md it comes from.

**Then the disagreements**, one line each: a dataset where DECISIONS.md, the two outlines
and the code do not say the same thing. Say which of the four is the odd one out.

## What was already measured on 2026-11-03

Read in the code that day. Re-check before repeating any of it.

Every dataset in the Paper 2 list has a registry entry, a loader and a feature-set list.
The gaps are these:

- **Six nucleotide datasets are in `plan.CUT_DATASETS`**, so no ledger plans them:
  nt_H3K27me3, nt_H3K4me3, nt_H3K9ac, nt_H3K9me3, nt_promoter_tata and
  gb_human_enhancers_cohn. They are Paper 2's nucleotide datasets, cut from 21 to 6 by
  Adelaide on 2026-07-18.
- **bh_doyle_plates is in `plan.CUT_DATASETS`.** It is in Paper 2.
- **flab_binding_adams and flab_expression_adams are in `plan.CUT_DATASETS`.** Both are in
  Paper 2 and Paper 3 as of 2026-11-03.
- **The two Adams scans are to use random folds** (Adelaide, 2026-11-03), citing FLAb
  (`Chungyoun2024`). Two places disagree with that today:
  `tests/antibody_baseline_survey.py` maps both names to the `position_set` strategy, and
  `_shared.cv_split` (line 2606) raises an error for any dataset with no groups, for every
  dataset, with a message citing the no-random-splits rule. A per-dataset exception is
  needed in both.
- **cdr_flexibility_h3 and cdr_flexibility_l3 have no entry in `decided_grid.json`.** Every
  other dataset in the Paper 2 list has one.
- **DroneRF is in `NEW_DATASETS.md` and DECISIONS.md but in neither outline.** Adelaide will
  add it to Paper 2 after its view screen (2026-11-02), so this is expected, not a fault.

The run-list column belongs to the old ledger. The new experiment ledger replaces it, so
those four rows may resolve themselves; say which ones do and which do not.

## The three benchmarks outside chemistry

Matbench, ProteinGym and the Kepler threshold crossing events are in Paper 2's dataset list.
None of them has a registry entry, because `register_datasets` accepts only the domains
`molecular, antibody, dta, tcga, reaction, nucleotide` (`src/kirby/registry.py`, read
2026-10-26 per NEW_DATASETS.md; re-check). DroneRF is the same. Give these four their own
rows in the table and say, per item, what is missing. The list in `NEW_DATASETS.md`,
section "Not yet in KIRBy", is a dated claim from 2026-10-26: check it against the code
rather than repeating it.

## Rules

- Open the code in this session for every claim. Not memory, not a document.
- `DATASET_STATUS.md` was last edited 2026-08-24 and is stale: it lists
  flab_thermo_tresanco, ginkgo_pr_cho and abagym_VEGF as active, and all three are out of
  both papers. Use it for the cut reasons it records, not for what is in.
- No patient data: TCGA, ANNITIA and NHANES are never opened, run or counted.
- Do not cut a dataset or a setting to save compute.
- Large files, logs and temp files go on `/Volumes/seagate`, never `/tmp`.
- Never ssh to the server. Adelaide runs every server command.

## Open questions for Adelaide

1. Nothing in the code reads the two-downstream-models pair from DECISIONS.md. Should the
   new ledger read it from a generated file, the way `decided_grid.json` is generated from
   DECISIONS.md, or should the pair go in `dataset_policy.py` beside the registry entry?
2. Which file should be the one place a dataset's paper membership is recorded? Today it is
   the two outlines, and DECISIONS.md repeats it in prose.
3. c11 refuses a dataset with no entry in `decided_grid.json`, and two datasets in Paper 2
   have no entry. Is c11 in either paper's run list, and if so do those two datasets need
   an entry?
