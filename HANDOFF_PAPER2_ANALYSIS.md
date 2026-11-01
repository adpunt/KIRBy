# Handoff: one analysis script for Paper 2

Written 2026-11-01. Re-read the code before repeating anything here; this file is a
dated claim, not a fact.

## The task

Write `tests/analysis/paper2_results.py`: one script that reads the result files and
prints, for every Results bullet in `PAPER_2_OUTLINE.md`, the table or figure that
bullet describes. The outline is the specification. Each Results bullet states its
shape (table / figure / one sentence) and ends "Source cXX"; that id names the result
folder. Nothing new is run. If a bullet needs a number the files do not hold, the
script prints MISSING for it and moves on.

Start with starter code: one function per Results bullet, each reading its files and
returning a table (pandas DataFrame) or saving a figure. Get every function to run
on the files that exist before polishing any one of them.

## Inputs

- Results: `/Volumes/seagate/kirby_results_2026-10-10b/complexity/<folder>/<dataset>.json`
  (the 10-06 server run). These are debug-grade for experiments whose code changed
  since (c16, c24, c33 changed 2026-10-31); the script must not care which run it
  reads: take `--results-root` as an argument.
- Each file's layout differs. Open one per experiment and print its keys first.
- `_precompute_errors/<dataset>.json` is the run's failure ledger, not a result.
- The per-fold searches (namespace `ocv<n>f<i>s42`, seed 42) are what every per-fold
  experiment reads. `tests/experiments/_shared.py:1491` `fold_searches` loads them.
  Each search keeps every configuration's score in `SearchResult.all_results`
  (`src/kirby/find_best_hybrid.py:1634`); not yet loaded and checked by hand.

## Two analyses that replace experiments (author, 2026-11-01)

1. **Allocation methods.** c31 and c32 are dropped. In their place, from the per-fold
   searches' `all_results`: per dataset and downstream model, which allocation method
   each fold picked, and the best score each allocation method reached on that fold.
   Dataset's own metric, per fold, with the fold spread. This justifies searching the
   five methods rather than fixing one.
2. **CGR representations.** Answered by `tests/experiments/opig_cgr_comparison.py`
   (c43 was deleted 2026-11-01; its one relevant measurement moved there). Per fold
   and model on Buchwald-Hartwig, bh_doyle_plates and Suzuki: the search with and
   without the CGR representations, how many hybrid features came from each CGR
   representation (`features_by_rep`), and those features scored alone beside the
   representation at full width (`cgr_selected_alone`, `cgr_full_width`). The 10-06
   run of it did not finish (ledger pending, partial file only). cgr_morgan is a
   hashed circular fingerprint on the published condensed graph of reaction (Varnek
   2005, Hoonakker 2011); not a new representation.

## Code to reuse

- `tests/analysis/thesis_figures.py`: `PATIENT_DATA_PREFIXES` (line 53) and the file
  filter after it. Use it; never open a tcga_*, annitia* or nhanes* file.
- `tests/analysis/thesis_figure_style.py`: figure style.
- `tests/analysis/ch7_numbers.py`: the pattern of one function per cited result,
  printing OK / MISSING. A good model for the structure.
- `decided_models(ds, 'general')` gives each dataset's models; never hardcode models,
  not every dataset has every model.

## Rules

- Repo-root `CLAUDE.md` applies to every message to the author.
- No patient data in any form.
- Per-fold scores and their spread. No tallies ("k of N"), no averaging across
  datasets. The author is open to a properly done bootstrap (1,000 resamples of
  molecule groups) but has not decided; ask before adding one.
- Tables: one number per cell, at most four bold per table.
- Nothing committed; the working tree is hers. Files on /Volumes/seagate, never /tmp.
- Do not write findings into `PAPER_2_OUTLINE.md`; the script prints, she writes.
