# OPEN WORK — similarity suite, 2026-08-27

**Branch:** `similarity-metrics-study` · **Indexed from:** `SIMILARITY_METRICS_ROADMAP.md` §D

Six items. Each was checked against the code on 2026-08-27; the facts below were read,
not recalled. Each carries a handoff prompt that can be pasted into a fresh session on
its own — nothing in a prompt depends on having read the others.

Three of the six (2, 3, 5) were recorded nowhere before this file.

| # | item | blocking? | size |
|---|---|---|---|
| 1 | No similarity number has run through nested cross-validation | yes, everything waits on it | a job, already queued |
| 2 | The redundancy experiment reports no fold spread | yes, for reading its results | small |
| 3 | 55 stored result files hold no computed number | no | small |
| 4 | The geometry survey and cross-dataset run were never redone | no | two runs |
| 5 | Three groups of undefined correlations still stored as 0.0 | no | medium |
| 6 | The noise suite opens a pair file without checking the molecule list | no | small |

---

## 1. Nothing has run through nested cross-validation

Every similarity number on disk is a whole-dataset statistic or a single-level split.
`f26_redundancy_penalty.py` is the only experiment wired to `nested_cv_evaluate` under a
similarity objective. SLURM job **12907696** is queued: 5 datasets, one per domain and two
molecular, 12 arms x 2 objectives x 8 budgets x 5 outer folds.

**No handoff needed — it is a job, not a task.** Results land in
`tests/experiments/results/f26_redundancy_penalty_<dataset>.json`, rewritten after every
one of the 24 cells, so partial results are readable while it runs. Read `progress` for
`cells_attempted`, `cells_planned`, `last_cell` and `complete`; a failed cell lands in
`skipped` with its error.

★ Array tasks read the code when each task STARTS. Three commits landed after the job was
submitted (the twelfth arm, the size-controlled number, the NaN convention), so a task
that had not started when the server was last pulled runs different code from one that
had. Check `progress` and the arm list per dataset before comparing datasets.

---

## 2. The redundancy experiment reports no fold spread

**Verified 2026-08-27.** `summarise_arm` already stores `per_fold_delta` for every
(objective, arm, model). Nothing reads it. `print_report` shows `delta_mean_of_folds`,
`n_paired_folds` and the difference against the reference arm, so two arms that differ by
less than fold-to-fold noise look like a winner and a loser.

This is the test that turned the earlier budget-split experiment from "this rule won by
0.003" into "no rule beats an equal split". Without it, job 12907696's results cannot be
read the way its own three-option decision needs.

### Handoff prompt

```
In /Users/apunt/repos/KIRBy on branch similarity-metrics-study.

tests/experiments/framework/f26_redundancy_penalty.py compares 12 allocation arms per
dataset. summarise_arm() already stores 'per_fold_delta' — the hybrid-minus-best-single
difference on each outer fold — for every (objective, arm, model). Nothing reads it.

Add, without changing any stored quantity's value:
1. The standard deviation across per_fold_delta, stored beside delta_mean_of_folds and
   printed as a column in print_report.
2. For every arm, whether its lead over the 'equal' arm and over its entry in
   REFERENCE_ARM exceeds that spread. Print it as a plain yes/no column, not a p-value.
3. A closing line per (objective, model) that says how many of the 12 arms cleared the
   spread against 'equal'. Zero clearing is the expected and publishable answer, not a
   failure — do not phrase it as one.

Do not add a significance test, a confidence interval or a correction for multiple
comparisons unless the owner asks. The spread is what is wanted.

Constraints: never run git checkout/reset/stash/clean/restore; do not commit unless
asked. No heavy local compute — verify with a small synthetic result dict, not a run.
Do not edit any file named f00 through f15 under tests/experiments/framework/.
```

---

## 3. 55 stored result files hold no computed number

**Measured 2026-08-27.** `tests/results` holds 1543 JSON files. Classified with
`classify_result_payload` in `tests/experiments/_shared.py`: 1403 hold a measurement, 85
are deliberate skips, **55 hold neither**. The 55 have not been listed.

An experiment that returned empty sections still writes its file, and the resume check
treats the file existing as proof the experiment finished — so each of those 55 is a
dataset that every resubmission skips. The emptiness stamp was added on 2026-08-26; the
resume checks in `similarity_helpers.py` and `noise_helpers.py` do not read it yet.

### Handoff prompt

```
In /Users/apunt/repos/KIRBy on branch similarity-metrics-study.

tests/experiments/_shared.py has classify_result_payload(), which decides whether a saved
result holds a measurement, a deliberate skip, or nothing. Run over tests/results
recursively on 2026-08-27 it gave 1403 with a measurement, 85 skips and 55 with neither,
of 1543 files.

Task one: list those 55. For each, the experiment, the dataset, the file size, its
modification date, and the reason it holds nothing if the file records one. Write the
list to a markdown file in the repository root. Facts only — no verdict on whether any of
them matters.

Task two: the resume checks at tests/experiments/similarity/similarity_helpers.py:914 and
tests/experiments/noise/noise_helpers.py:296 treat a file existing as proof the
experiment finished, so all 55 are skipped on every resubmission. Make both read the
emptiness stamp save_experiment_results writes, so a file holding no measurement is
recomputed rather than skipped. A file written before the stamp existed carries none —
decide what that should mean, say which you chose, and make it explicit in the code.

Do not delete any result file.

Constraints: never run git checkout/reset/stash/clean/restore; do not commit unless
asked. No heavy local compute. Do not edit any file named f00 through f15 under
tests/experiments/framework/.
```

---

## 4. The geometry survey and the cross-dataset run were never redone

Both failures are loud now, neither run was repeated.

The 15 files in `tests/results_server/geometry_survey/`, dated 2026-03-19, still hold
`"geometry_baselines": {}` and `"augmentations": {}`. Cause: the antibody structure
predictor was absent, so all 7 feature groups came back as zero arrays and every one was
skipped. Since 2026-08-26 that raises instead, and an all-zero group is never cached nor
trusted on read. **Rerunning needs the structure predictor installed on the server, and
the server-side geometry cache checked for zero arrays first** —
`structural_geometry_survey.py --scan-zero-cache` lists them without deleting.

No cross-dataset summary result exists at all. Its inputs exist for one dataset only
(logd), and its own significance test needs three.

### Handoff prompt

```
In /Users/apunt/repos/KIRBy on branch similarity-metrics-study. This is SERVER work.

Two runs failed silently in March and were repaired on 2026-08-26 but never repeated.

(a) The antibody geometry survey. tests/experiments/structural_geometry_survey.py over 15
antibody datasets produced 15 result files with both result blocks empty, because
ImmuneBuilder was not importable in that job's environment and the geometry features came
back as zero arrays. Before rerunning: confirm ImmuneBuilder imports on the server, then
run `python tests/experiments/structural_geometry_survey.py --scan-zero-cache` and report
which cached geometry arrays under $KIRBY_CACHE_DIR are entirely zero. A zero array
written to that cache in March would be reloaded and reported as real. Do not delete
anything without asking. Then submit the survey.

(b) The cross-dataset summary. tests/experiments/similarity/s09_cross_dataset_summary.py
aggregates s02 and s04 through s08 across datasets. On disk only logd has those inputs,
and the paired test needs at least 3 datasets. Establish which datasets have the six
inputs, and say what would have to run first. Do not run the aggregator on one dataset.

Constraints: never run git checkout/reset/stash/clean/restore. --partition is live state;
run `bash tests/slurm_scripts/where_to_submit.sh` first — it is mode 644, so `bash`, not
`./`. The server tracks whichever branch is checked out; confirm it is
similarity-metrics-study before pulling.
```

---

## 5. Three groups of undefined correlations still stored as 0.0

An undefined Spearman stored as 0.0 is indistinguishable from a measured correlation of
zero. Fixed on 2026-08-26 for the read-across correlation across the standalone scripts,
and on 2026-08-27 in the engine. **Three groups remain, all verified present on
2026-08-27:**

- **Agreement with shared substructure**, folded to 0.0 in 8 standalone scripts:
  `aggregate.py:142`, `ranker_experiment.py:213`, `augmented_descriptor_reps.py:104`,
  `structural_embeddings.py:123`, `extra_fingerprints.py:134`, `feature_type_study.py:132`,
  `scaffold_experiment.py:187,192,196`, `scaffold_ddg_experiment.py:171`.
- **The classification analogue of read-across**,
  `scaffold_experiment.readacross_classification`, returns 0.0 when no area-under-curve
  can be formed. It writes the same field as the regression path, so the
  `readacross_defined` flag added on 2026-08-26 does not distinguish it.
- **Two files outside the 2026-08-26 sweep:** `base_rep_comparison.py:90,97` (untracked at
  the time) and `triplet_similarity.py:65`, each with their own read-across.

### Handoff prompt

```
In /Users/apunt/repos/KIRBy on branch similarity-metrics-study.

An undefined Spearman or area-under-curve stored as 0.0 reads later as a measured
correlation of zero. On 2026-08-26 this was fixed for the read-across correlation across
tests/experiments/similarity_metrics/, and on 2026-08-27 in
src/kirby/find_best_hybrid.py. Three groups remain, all confirmed present on 2026-08-27:

1. Agreement with shared substructure, folded to 0.0 at aggregate.py:142,
   ranker_experiment.py:213, augmented_descriptor_reps.py:104,
   structural_embeddings.py:123, extra_fingerprints.py:134, feature_type_study.py:132,
   scaffold_experiment.py:187/192/196 and scaffold_ddg_experiment.py:171.
2. scaffold_experiment.readacross_classification returns 0.0 when no area-under-curve can
   be formed, and writes the same field as the regression path.
3. base_rep_comparison.py:90 and :97, and triplet_similarity.py:65.

Return NaN instead, and add a boolean beside each stored value saying whether it was
defined — the same convention as the readacross_defined flag already in those files.

THE PART THAT MATTERS: NaN never loses a Python max(). Before changing any return, find
every caller that sorts, maxes, or averages that value and make it skip non-finite
entries. tests/experiments/similarity_metrics/_nanrank.py already holds rank_desc,
nanmean and n_defined for exactly this. When the same change was made for read-across it
needed 20 call sites across 15 files — assume this one is comparable and find them all.
Report any caller you could not make safe rather than leaving it.

Constraints: never run git checkout/reset/stash/clean/restore; do not commit unless
asked. No heavy local compute — the noise-cleaning script takes about 8 minutes per
dataset, so run at most one. Do not edit any file named f00 through f15 under
tests/experiments/framework/.
```

---

## 6. The noise suite opens a pair file without checking the molecule list

`tests/experiments/noise/_similarity_hybrid_space.py:88` loads
`mcs_pairs_<dataset>.npz` with a bare `np.load` and feeds the pair row numbers straight
into the descriptor drop at line 95. Those three files predate the molecule-list hash and
carry none, so nothing there can prove which molecule list the row numbers belong to.

The lists genuinely differ: herg_ki is 1415 molecules in the study arrays and 1412 through
the framework loader after InChIKey deduplication; logd exists in three orderings — 5039
uncapped, 1427 after the study's scaffold subsample, 5039 through the framework. Row 900
is a different molecule in each.

Every other reader now checks. `src/kirby/similarity/mcs.py` provides
`assert_pairs_match_smiles`, and `python -m kirby.similarity.mcs_audit <dataset>` prints
the competing lists side by side.

### Handoff prompt

```
In /Users/apunt/repos/KIRBy on branch similarity-metrics-study.

tests/experiments/noise/_similarity_hybrid_space.py line 88 loads
mcs_pairs_<dataset>.npz with a bare np.load and passes the pair row numbers into the
descriptor drop at line 95, with no check that those row numbers belong to the molecule
list in hand. Pair endpoints are positions into a specific ordered molecule list, and for
one dataset name several orderings exist: herg_ki is 1415 molecules in the study arrays
and 1412 through the framework loader after InChIKey deduplication; logd is 5039
uncapped, 1427 after the study's scaffold subsample, and 5039 through the framework.
Row 900 names a different molecule in each.

Make this file refuse a mismatch instead of scoring silently. src/kirby/similarity/mcs.py
already provides assert_pairs_match_smiles, which raises on a missing or differing
ordering hash or an index past the end of the list. The three July .npz files carry no
hash, so they will be refused — decide and state whether this file should then rebuild
its pairs through kirby.similarity.mcs.get_mcs_pairs (which hashes the list and caches by
it) or stop with a message telling the user to.

Run `python -m kirby.similarity.mcs_audit herg_ki` first to see the competing lists.

Constraints: never run git checkout/reset/stash/clean/restore; do not commit unless
asked. No heavy local compute — building 8000 pairs takes about 12 minutes per dataset,
so do not build them as a test. Do not edit any file named f00 through f15 under
tests/experiments/framework/.
```
