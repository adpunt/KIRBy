# Handoff: align every experiment. Part 1, the kNN sweeps

Written 2026-10-05. Thesis due 2026-10-10.

**Read in this order before touching anything:**
1. `CLAUDE.md` at the repo root. How to write to Adelaide. Follow it in every reply.
2. `tests/experiments/framework/F38_DECISIONS.md`. Signed off, not to be argued again.
3. `tests/experiments/framework/KNN_SEARCH_SPACE_RECORD.md`. **The decisions you are implementing.**
   Every axis has a DECISION block with Adelaide's choice, the numbers it rests on, and the stated
   reason for each cut. Do not re-derive them. Do not fill a DECISION line yourself.
4. `scratchpad/precompute_cuts/DECISIONS.md`. Per-dataset models, budgets, importance methods,
   allocation methods, evaluators, representation counts, augmentation. Authoritative sections are
   the ones with a bold dated DECISION cell; the file also holds superseded proposals.
5. `HANDOFF_PER_FOLD_ALIGNMENT.md` at the repo root. Why the precompute is being redone per fold,
   and the rules Adelaide set for it.

**Where every kNN decision lives:** `tests/experiments/framework/KNN_SEARCH_SPACE_RECORD.md`.
Nowhere else. It holds all six axes, both dataset groups, the overriding rules, every asterisk, and
the open questions. The sections are headed `## DECISION, axis N (Adelaide, 2026-10-05)`.

---

## 1. The decided kNN search space

Two dataset groups, split on molecule count.

| group | which datasets |
|---|---|
| `full_knn` | 5000 molecules or fewer |
| `fast_knn` | more than 5000 molecules: logd (5028), herg (7772), qm9 (9978), kiba (30000), bindingdb_kd (30000), nt_H3K27me3, nt_H3K4me3, nt_H3K9ac, nt_H3K9me3, nt_promoter_tata, gb_human_enhancers_cohn (18000 to 33000) |

Counts come from `REPRODUCIBILITY_LEDGER.md:180` and `TIME_ESTIMATION_METHOD.md:166`, both dated
claims. The split is at 5000 because the pair work in a neighbour search grows with the square of
the molecule count, and it jumps from 13 times herg_ki's at logd to 30 times at herg.

### Axis 1 — which representation the distance is measured in

The ten candidates are in `similarity_space.py:120-131`. Numbered as the record numbers them:
1 best single representation, 2 second-best single, 3 number 1 with augmentation features,
4 number 2 with augmentation features, 5 best predictive hybrid, 6 second-best predictive hybrid,
7 best hybrid under the similarity objective, 8 raw ECFP4 bits with Tanimoto, 9 forest leaves,
10 the model's own matrix.

| group | kept | cut |
|---|---|---|
| full_knn | 1, 2, 3, 4, 5, 6, 7, 8, 9 | 10 |
| fast_knn | 1, 3, 5, 6, 7, 9 | 2, 4, 8, 10 |

**Overriding rules, which beat the group:**
- A dataset with no SMILES never uses 8, in either group. `similarity_space.py:602-604` already
  returns the space unavailable without SMILES. This covers the five `nt_` sets and
  gb_human_enhancers_cohn.
- 7 does not exist on tcga_brca_pam50 or tz_her2_binary (`similarity_space.py:217`).

### Axis 2 — which distance formula

| group | kept | cut |
|---|---|---|
| full_knn | manhattan, cosine, minkowski p=0.5 | euclidean, correlation |
| fast_knn | manhattan, cosine | euclidean, correlation, minkowski p=0.5 |

`similarity_space.py:146` maps only manhattan and euclidean, and `SimilaritySetting.__post_init__`
raises on any other name (`:184-185`). **Both of those have to change.** Adding a name also needs a
rule in `preprocessing.weight_exponent` (`preprocessing.py:342-361`) if weighting ever returns; see
axis 3.

Numbers 8 and 9 carry their own fixed distance and are unaffected.

### Axis 3 — importance weighting

**Cut entirely, both groups. No weighting.** `WEIGHTINGS` becomes `('none',)`
(`similarity_space.py:133`).

Because there is no weighting, there is no power to set. The three-way disagreement about the
power — 1.0 under manhattan in the noise suite, 0.5 everywhere through
`_shared.importance_feature_weights`, 0.5 hardcoded in s04 — is closed for the noise suite by this
decision. **It stays open for the similarity experiments**, which weight features for their own
tasks (s05, s06, s10, s13). That is Part 2.

### Axis 4 — how many neighbours

**Keep 5, 10, 20. Both groups. Cut 9.** `SWEPT_KS` becomes `(5, 10, 20)`
(`similarity_space.py:141`).

⛔ `DEFAULT_K_MAX` is 20 and `neighbours_at` raises when asked for a k the stored list does not hold
(`similarity_space.py:738-771`). It does not trim a longer list. Check `model_input_ks`
(`:688-735`) still lines up after the change, and that nothing asks for 9.

### Axis 5 — how much each neighbour counts

**Keep two: `rank` and `inverse_square`. Both groups.** Schemes are in
`src/kirby/similarity/neighbour_weighting.py` (`WEIGHTING_SCHEMES`, `weight_array`,
`sklearn_kwargs`).

This axis is **not swept today at all**. Every consumer has its own fixed answer, and they disagree:
- unweighted mean: `label_smooth` (`n07_remediation.py:995`), `knn_smooth` (`:1015`),
  `neighbor_residual` (`noise_estimation.py:366`), `local_variance_ratio` (`:443`),
  `activity_cliff_aware` (`:571`)
- one over distance: `distance_weighted` only (`noise_estimation.py:487-488`)
- read from an environment variable, defaulting to equal: s05 (`KIRBY_S05_KNN_VOTE`), s10
  (`KIRBY_S10_KNN_VOTE`)
- sklearn's own default, so equal: s04 (`s04_activity_cliffs.py:426-430`)
- equal written out: n40:418, 1041, 1048

So making this a swept axis is new work, not a cut. It doubles the refits for the methods that
refit; it is free for the methods that only score (n01, n14, n22, n32, n37, n40, n41).

### Axis 6 — whether a molecule's own scaffold group is excluded

**One list, the full one. Do not exclude.** `exclude_groups` at `similarity_space.py:433-437`
stops being used for these lists.

### The resulting settings count

| group | representations | distances | weightings | neighbour counts | schemes | settings |
|---|---|---|---|---|---|---|
| full_knn | 7 weighted + 8 and 9 | 3 | 1 | 3 | 2 | **138** |
| fast_knn | 5 weighted + 9 | 2 | 1 | 3 | 2 | **66** |

A weighted representation counts as distances x weightings; a fixed space (8, 9) counts as 1.
Against today's 59 settings x 4 neighbour counts = 236.

---

## 2. Things that will bite you

### Metric direction
Every comparison must use the dataset's own metric and its own direction. rmse is lower-is-better;
spearman, auc, ci, mcc and f1_weighted are higher-is-better. `get_domain_metric(data)` returns
`(metric_name, metric_fn, lower_is_better)` — use the third value, never assume.
`top_base_reps` returns a `direction` field (`_shared.py:7926`); carry it into any result file that
reports a ranking. The metric-direction sign bug has bitten this repo before
(memory: `project_metric_direction_signbug`).

⛔ **Classification detection scores are not rmse.** In f38, recovery on a regression dataset is
rmse to the clean label (lower better) and on herg it is the share of corrupted molecules put back
in the clean class (higher better). Any analyser that pools them is wrong.

### Censoring inverts the detection score
Under the `censoring` damage kind, clipped labels are ranked the LEAST suspicious by the neighbour
gap: detection AUROC 0.2768 on herg_ki, 0.3151 caco2, 0.3936 adme_mdr1. Replacing a clipped label
with its neighbours' mean was worse than leaving it in every row measured. Written up in
`thesis.tex` section 4.4.2, paragraph "Censoring inverts the detection score", with four
`TODO(delve)` items above it. **Do not read a censoring AUROC below 0.5 as a bad setting.** Any
analyser must exclude censoring or report it separately.

### The guard that is already gone
`noise_cache.cached_detector_scores` used to raise when handed a neighbour list in which a molecule
had a scaffold-mate. **That check was removed in this session** — see section 5. Every detector
score in n07, n18, n24, n25 and n42 moves from the restricted list to the full one, so those numbers
go up. Median change in f38 was 0.0021 to 0.0407 detection AUROC.

### The hybrid pool is not the neighbour list
A hybrid draws from all ten pruned representations (`DATASET_REPS`). Axis 1 cuts which single
representations a neighbour arm measures distance in. **Never filter `data['reps']` before passing
it to a hybrid builder.** Filter at the per-representation loop only.

### Set-overlap distances cannot reach the noise suite's feature spaces
The largest distance gaps anywhere go to dice and tanimoto on fingerprints — 0.0923 rmse on
adme_mdr1 atom_pair against euclidean, with a fold spread of 0.0200. The noise suite cannot use
them, because it z-scores every matrix first (`n07_remediation.py:1629-1631`, and
`similarity_space.py:662-667` requires it). A z-scored bit is a signed number, so there is no set to
intersect. OPEN, Adelaide's to decide: whether a base representation should ever be passed raw.

### The precompute is being rebuilt per fold
The hybrid spaces (axis 1 numbers 5, 6, 7) come from that precompute, and f38's were test spaces
rebuilt from the old whole-dataset searches. So no cut to 5, 6 or 7 rests on f38's recipes. The
per-fold rules are in `HANDOFF_PER_FOLD_ALIGNMENT.md`.

### Cost traps already found and fixed in f38, reuse them
- Cap each process at one thread: OMP, MKL, OpenBLAS, LOKY, `KIRBY_LGBM_THREADS`, `KIRBY_RF_JOBS`,
  torch. Thread pile-up was the single biggest waste.
- scipy's weighted correlation runs one pair at a time in Python.
- The noise suite recomputes importance for every neighbour list. With axis 3 cut, that work
  disappears.
- `forest_proximity` refits its forest for every list kind. One forest per model input is enough.
- Any search that is not fully stored must never start inside an experiment: set
  `KIRBY_COLD_SEARCH_MAX_CONFIGS=0`.

---

## 3. What to change, in order

Nothing below has been done except where section 5 says so.

**Step 1. `tests/experiments/noise/similarity_space.py`.**
- `WEIGHTINGS` to `('none',)`.
- `SWEPT_KS` to `(5, 10, 20)`. Check `DEFAULT_K_MAX`, `model_input_ks` and `neighbours_at`.
- `_CDIST` to hold manhattan, cosine and minkowski p=0.5, and `FEATURE_METRICS` to match.
  `SimilaritySetting.__post_init__` must accept the new names.
- `FEATURE_SPACES` and `ALL_SPACES` to drop `model_input` from the swept list. It is axis 1's
  number 10 and is cut in both groups. `enumerate_settings` already leaves it out
  (`:212-213`); `n07.sweep_settings` puts it back (`n07:1582-1598`) and must stop.
- Add the group split. A dataset's group decides its representation list and its distance list.
  `fast_knn` also drops numbers 2 and 4.
- Add how much each neighbour counts as a swept axis, two values.
- Stop building the scaffold-excluded list.
- `settings_from_env` must still parse `KIRBY_SIMILARITY_SETTINGS` and must raise on a name that no
  longer exists rather than silently dropping it.

**Step 2. The 12 unit tests in `tests/test_f38_neighbour_screen.py`** check that f38's lists match
the noise suite's, list for list. They will fail after step 1. Update them to the new lists; do not
delete them.

**Step 3. `n07_remediation.py`.** `sweep_settings` (`:1582`), `sweep_ks` (`:1609`),
`fold_similarity_spaces` (`:1634`), `fold_neighbours` (`:1788`). Also `n32.sweep_settings`
(`n32:595-609`) and `_sim_axis_n15_n22.sweep_settings` (`:86-107`), which have their own copies.

**Step 4. The consumers of how-much-each-neighbour-counts.** Every site listed under axis 5 above
has to read the swept value instead of its own hardcoded one.

**Step 5. The similarity experiments.** Cut their per-representation neighbour arms to the two best
for knn. A helper is already written — see section 5. The sites that loop over every
representation: s01:62, s02:635, s04:572 and :588, s05:774, s06:305, s08:241, s10 at four places
from :661, s13:361. ⛔ Each of these files also passes `reps` to a hybrid builder. Filter the loop,
not the dict.

**Step 6. Test that each unchanged setting still gives its old list.** The point of the change is
the cut, not a drift in what survives. Build the old and new lists for a setting that both hold and
assert they are identical.

**Step 7. The "does it run" check.** Normal settings, time-boxed per experiment, nothing shrunk.
Record which were slow and why. Never wait for one to finish. The rule is in
`~/.claude/projects/-Users-apunt-repos-KIRBy/memory/feedback_crash_check_means_timebox_not_finish.md`.

**Step 8. srun blocks for Adelaide.** One `srun ... --pty /bin/bash` line, then plain commands
typed inside. Never sbatch. Never `rm` or any delete in a block she pastes — she re-pastes whole
blocks, and an `rm` once cost 120 finished f38 tasks. Interactive sessions are limited to 4 hours
and she runs several at once, so give work that splits (`--part K/N`).

---

## 4. The asterisks. Every one rests on herg

herg has **0 of 60 f38 tasks finished**, each part-way at 192 to 402 rows of about 500, and no
overlap rows at all. Adelaide accepted this and chose to proceed. Revisit all four together if herg
lands.

- Axis 1: cutting 2 and 4 on fast_knn was a compute decision, not an evidence one. herg has no f38
  rows for 7, 9 or 10.
- Axis 2: cutting minkowski p=0.5 on fast_knn was a compute decision. herg has no overlap rows, so
  no duplicate check ran there.
- Axis 4: 9 neighbours is the best count on 81 of herg's 157 cells, more than any other, and it was
  cut.
- Axis 5: gaussian is the best scheme on herg, winning 288 of 471 cells, and it was cut.

**The srun blocks to finish herg**, if Adelaide wants to try again. Two terminals. Safe to re-paste:
a finished task is skipped by `run_task` itself (`f38_neighbour_screen.py:648-650`) and a part-way
task keeps the rows it wrote. No deletes.

Terminal 1:
```
srun --clusters=htc --account=stat-cadd -p interactive --cpus-per-task=16 --mem=64G --time=04:00:00 --pty /bin/bash
```
then inside:
```
cd /data/stat-ecr/scat9264/KIRBy
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
python tests/experiments/framework/f38_neighbour_screen.py --run-tasks f38_tasks.txt --jobs 16 --only herg --part 1/2
```
Terminal 2 is the same with `--part 2/2`. Progress:
`python tests/experiments/framework/f38_neighbour_screen.py --status f38_tasks.txt`

---

## 5. What this session already changed

Three commits plus two uncommitted edits. Check `git log` and `git diff` before you start.

**Committed:**
- `tests/experiments/framework/KNN_SEARCH_SPACE_RECORD.md` — the decision record. Commits
  `6263a63`, `d8af29d`, `18f7e94`, `157d154`, `d98f838`, `f8e6999`, `5114e9a`, `75a6e21`.
- `tests/tools/dataset_sizes.py` — read-only molecule count per dataset. Not needed any more; the
  counts are in `REPRODUCIBILITY_LEDGER.md:180`.
- `thesis.tex` and `THESIS_OUTLINE.md` — the censoring paragraph in section 4.4.2.

**Uncommitted, and both need review:**
- `tests/experiments/noise/noise_cache.py` — the scaffold-mate guard is removed from
  `cached_detector_scores`, with the reason in its docstring. Adelaide approved this.
- `tests/experiments/similarity/similarity_helpers.py` — a new `neighbour_arm_reps()` helper plus
  `SIMILARITY_BASE_REP_COUNT = 2` and `SIMILARITY_BASE_REP_MODEL = 'knn'`. **NOT WIRED INTO ANY
  EXPERIMENT YET, AND NOT RUN ONCE.** It calls `top_base_reps` with `require_full=False` so a thin
  stored search returns a note instead of raising. Read it, test it, then wire it in at step 5.

**f35 was rerun locally** on all three datasets, 2026-10-05. Its result files in
`tests/results/framework/f35_scorer_path_check/` are current. One finding: its headline gap on
`pdv` is a scaling difference, not a scoring-path difference — one path hands `pdv`'s raw
descriptors to sklearn and the other z-scores them first. `create_pdv` returns raw descriptors on
purpose (`src/kirby/representations/molecular.py:1007-1029`); scaling is the caller's job. On every
fingerprint representation the two paths agree to 0.0018 rmse or less.

---

## 6. Rules that cost Adelaide real work when broken

- Plain words. Say what is broken before the fix, in its own paragraph. One decision per line.
  Never a bare fraction, ratio or count — every number carries its unit, and a table needs a line
  above it saying what one row is.
- Never introduce something with "the" that she has not seen.
- Banned: "read-across", "SAC", "round", "stage" for runs, "columns" for features, and any acronym
  she has not used first. Say what is computed.
- She decides every cut. Bring options with the numbers; never design alone and present it as
  settled.
- A cut needs a stated reason: degenerate, broken or undefined. A theory about which value will
  lose is not a reason.
- No significance tests. Per-fold scores and the spread only.
- Never write an AI verdict into her documents. Mark Claude prose as Claude prose.
- Verify in the code this session before claiming anything. Memory files and repo documents are
  dated claims, several have been wrong.
- Never ssh or scp. She runs every command. Large result files are pulled to `/Volumes/seagate`.
- Local or srun only, never sbatch, for anything she asks to run.
- Never use the multiple-choice question tool.

---


# Part 2: align every experiment with DECISIONS.md, per fold

Written 2026-10-05 by the session that did the OPIG suite. Part 1 above (the kNN sweeps) and
Part 2 are one piece of work: every experiment has to come out of it following both the kNN
record and DECISIONS.md. **Nothing in Part 2 is to be built until Adelaide has settled the items
marked OPEN in section 2.3.** She said so on 2026-10-05: "do not proceed, we need a discussion
about the open items".

Extra reading for Part 2, after the list at the top of this file:
- `scratchpad/precompute_cuts/EXPERIMENT_AUDIT.md` (loose, never commit). One section per
  experiment, 88 of them, each rule checked with file:line. Line ranges are in section 2.6.
  Written 2026-10-04 by agents and checked by a second agent each; dated claims, re-read the code.
- `/Volumes/seagate/dirty_local_2026_09_28/DIRTY_RUN_RESULTS.md`: bugs found by running every
  experiment once locally, 28 Sep to 2 Oct. Section 2.7 below maps each bug to what to do.
- `/Volumes/seagate/dirty_local_2026_09_28/COST_BREAKDOWN.md`: why 64 experiments hit the 3-hour
  cap. Analysed LAST, section 2.9.
- `HANDOFF_PER_FOLD_ALIGNMENT.md` sections 3 and 7.

---

## 2.0 Split into separate chats (Adelaide asked 2026-10-05; Claude's proposal)

One chat cannot hold all of this. Each chat below starts by reading this file and CLAUDE.md, does
only its part, keeps the status tables in this file current, and ends by showing Adelaide what
changed. The order matters: chat 1 changes shared code every suite depends on.

| chat | does | ends with |
|---|---|---|
| 1 | Part 1 (the kNN record) if the other process has not done it; then the shared code in 2.6 "shared code every suite needs"; the 0/1 scaling rule (2.3 item 17); the duplicate-hybrid check (2.1); distance hybrids in the noise neighbour spaces (2.3 item 1) | shared tests pass; one experiment from each suite smoke-run |
| 2 | OPIG re-aligned to the 2026-10-05 answers (2.6 table), then federated | her review of both |
| 3 | complexity | her review |
| 4 | similarity | her review |
| 5 | noise, after the fold rule (2.12) is decided | her review |
| 6 | the smoke screen of every experiment (2.8), then COST_BREAKDOWN.md (2.9) | one list: pass / slow (why) / failed (error); the cost table |

## 2.1 What Adelaide asked for (her words, tidied; 2026-10-03 to 2026-10-05)

The six rules every experiment is checked against. The audit file numbers them R1 to R6.
- **R1 models.** Each dataset uses only its two models from DECISIONS.md Step 1. knn only on the
  distance search, where Step 9 runs it.
- **R2 per fold.** Every hybrid for an outer fold comes from a search on that fold's training rows.
  The whole-dataset precompute leaks the held-out fold into every stored hybrid score.
- **R3 same cuts everywhere.** The dataset's decided budgets, importance methods, allocation
  methods, internal evaluators, representation counts and augmentation apply to the general
  search, the distance search, and every other `find_best_hybrid` call or augmentation step.
  **This includes every similarity-objective search and every distance hybrid, whatever its tier
  is called** (her words: "make sure these rules are applied to anything relating to similarity
  and distance hybrids, don't want those getting out of hand bc they don't match the right
  keywords"). DECISIONS.md has no table for similarity-objective searches; the per-fold handoff
  section 3 records that they "follow the same cuts and the same two models".
- **R4 standard selection.** Per model, per outer fold: the 2 best base representations and the 2
  best hybrids. Some experiments differ by design; she says yes or no to each (section 2.5).
- **R5 noise.** Noise experiments use exactly 8 configurations: 2 models x (2 base + 2 hybrids).
- **R6 never refused.** An experiment builds any per-fold search it is missing. No separate
  precompute scripts.

And from her message of 2026-10-05:
- **Distance hybrids where the experiment's question needs them.** "Experiments that would
  benefit distance hybrids should test distance hybrids." Which ones: section 2.3, item 1.
- **Pruning is OFF in every per-fold search.** She had already decided this; DECISIONS.md is the
  cut. Do not pass a pruning schedule. Verify `adaptive_pruning_schedule` and
  `model_pruning_aggressiveness` are None in every per-fold call, and that `--pruning-mode` is
  either removed or ignored with a printed line.
- **Representations: only those from representation pruning (`DATASET_REPS[ds]['general']`) and
  from the augmentation list (`DATASET_REPS[ds]['augmentation']`).** Nothing else. Section 2.3,
  item 4 lists the scripts that add others and where each came from.
- **trastuzumab gets a per-fold AUROC table** (hybrid against best single representation) to
  replace the whole-dataset one that was removed.
- **Skip a second hybrid whose selected features are identical to the first.** She expects this
  to be rare. Compare the selected feature indices per representation, not the recipe name.
- **Smoke-screen every experiment after the changes.** These are substantial changes. "Does it
  run" = normal settings, time-boxed, locally or by srun, never sbatch.
- **Fix the dirty-run bugs along the way** (section 2.7), since many come from the same
  misalignments.
- **Analyse COST_BREAKDOWN.md at the end** (section 2.9).
- **Language.** Every reply follows `CLAUDE.md`. On 2026-10-05 she could not read four of the nine
  questions put to her (about chembl_qsar, the undecided distance searches, the remedy arms and a
  "stored" allocation method). Say what the thing is, where it came from, and what she has to
  choose. No code names as subjects.

## 2.2 What is already in the code

### Committed and pushed (commit `fb1df54`, 2026-10-05, branch similarity-metrics-study)
`fb1df54` is titled as the svm label fix but also carries three engine edits made by the OPIG
session on Adelaide's instruction of 2026-10-03 ("C", then "keep the concurrency change"):
- `find_best_hybrid.py`, the full-evaluation loop: with `KIRBY_SEARCH_REVERSE=1` a job works
  through the configurations from the end of the list.
- The same loop: every 25 configurations, after writing, the job re-reads the result store and
  skips configurations another job has stored since. No in-progress flag, so a killed job leaves
  nothing that blocks the next.
- `_own` in `find_best_hybrid`: a search that carries `cache_namespace` (every per-fold search,
  e.g. `ocv5f2s42`) is never refused by `KIRBY_NO_COLD_SEARCH`. The refusal still applies to
  whole-dataset searches, which nothing should build any more.

The half that switches the reverse order on is NOT committed (below), so on the branch the
reverse order never turns on yet. The re-read is active and changes no result, only what is
skipped. Checked locally 2026-10-03: two jobs on one search took 73 fits instead of 96.

### Uncommitted (working tree), made by the OPIG session 2026-10-04/05
- `tests/experiments/build_decided_grid.py` (new): reads DECISIONS.md and
  `scratchpad/precompute_cuts/stored_sets.json` and writes `tests/experiments/decided_grid.json`.
  Re-run it after any DECISION cell changes. DECISIONS.md is never committed, so the server reads
  the JSON.
- `scratchpad/precompute_cuts/stored_sets.json` (new, loose): which allocation methods,
  importance methods, representation counts and evaluators each stored search actually holds
  (augmentation none), read from `/Volumes/seagate/slim_precompute`. "Keep all stored methods"
  is resolved against it.
- `tests/experiments/decided_grid.json` (new, generated): 41 datasets, 38 with a distance search.
- `tests/experiments/decided_grid.py` (new): `decided_search(ds, tier)`, `decided_models`,
  `decided_aug_conditions`, `apply_decided_grid(kwargs, ds)`. Any tier name containing
  "distance" gets the distance grid. An undecided dataset or distance search raises
  `UndecidedSearch`; there is no fallback to the old five-model, eight-budget defaults. A named
  axis is narrowed to the decided members (dropped members printed); a named axis with nothing
  decided left raises.
- `tests/experiments/_shared.py`:
  - `_apply_dataset_search_speed` now just calls `apply_decided_grid`. It used to set
    speed='fast' on 14 datasets and add gp under 1,000 molecules.
  - `search_on_fold`, `precompute_nested_cv_searches`, `run_nested_cv` all apply the decided grid
    and Step 8's augmentation conditions ('none'; plus 'uncapped' on the distance search of
    ginkgo_colloidal and ginkgo_hic; 'capped' nowhere).
  - `run_nested_cv` no longer reads `config['budgets']`, `config['importance_methods']` or
    `models=None`. ⛔ It replaces a cut augmentation condition with 'none' and drops the donor
    representations; an experiment that sweeps augmentation on purpose must not go through it
    (opig_shared_vectors_remedy calls `nested_cv_evaluate` itself for this reason).
  - New `fold_searches` (memoised per-fold search), and `get_fold_best_search` now refuses a model
    outside the decided ones (it used to hand back another model's recipe under the asked name).
  - New `fold_standard_selection(...)`: the 2 best base representations and 2 best hybrids per
    decided model for one outer fold, both ranked on that fold's training rows. Takes
    `condition_suffix` and `adaptive_pruning`. A model short of 2 reports its shortfall instead of
    raising. Checked on 150 synthetic rows under abagym_VEGF's grid: 2 + 2 per model, 112 s.
  - `STANDARD_N_BASE = 2`, `STANDARD_N_HYBRIDS = 2`.
- `tests/experiments/dataset_policy.py`: `FAST_SPEED_DATASETS` is now the record's 10 fast-tier
  datasets (was 14; flab_expression_koenig, logd, bindingdb_kd and kiba removed).
- `tests/slurm_scripts/submit_dataset.sh` and `runbook/opig_one.sh`: start delay under SLURM,
  similarity 300 s, every other non-complexity category 600 s, complexity 0; a delayed job sets
  `KIRBY_SEARCH_REVERSE=1`. `KIRBY_START_DELAY` overrides.
- `tests/slurm_scripts/runbook/submit_all_stages.sh`: stage 3 submits complexity first, then
  similarity, federated, opig, cross.
- The 10 OPIG scripts (2.6). Each was edited by one agent and compiles; the reports are summarised
  in 2.6. ⛔ They were written before Adelaide's 2026-10-05 answers and do not agree with each
  other on the distance hybrid, pruning, or extra representations. Re-align them first.
- NOT this session's: `noise_cache.py`, `similarity_helpers.py`, `thesis.tex`,
  `THESIS_OUTLINE.md`, `thesis_appendix/...` (Part 1's session and others).

Checked in this session: herg_ki fold 0 general search on the decided grid ran 224
configurations (svm+dnn, budgets 250-2000, lightgbm_gain and treeshap, 4 allocation methods,
4 evaluators, whole pool); importance alone took 307 s. The distance search ran 360
configurations (svm, dnn, knn; budgets 100-500). Both stopped at the time box, no error.

## 2.3 Adelaide's answers of 2026-10-05, and what is still open

### Answered (do these)
1. **Distance hybrids only where the experiment measures something with neighbours or
   distances.** Both kinds: the top hybrid of the distance search (tier `general_distance`; knn is
   one of its models, budgets stop at 500 features) and the top hybrid of the similarity search
   (objective `similarity`; features chosen so distance follows activity difference). Candidates,
   to confirm file by file: the 23 noise experiments that read the shared neighbour sweep (n01,
   n02, n07, n09, n11, n12, n13, n14, n17, n18, n20, n21, n22, n23, n24, n25, n32, n35, n36, n37,
   n40, n41, n42), s02, s04, s05, s06, s08, s10, s12, s13, c04, opig_shared_vectors_census. Every
   other experiment uses the 2 general hybrids only. So the OPIG prediction scripts that kept a
   distance hybrid (humanness, trastuzumab, ginkgo, qsar, scaffold_hybrid, shared_vectors_remedy)
   drop it.
2. **opig_trastuzumab_enrichment keeps esmc_llr, antiberty_llr and esm2.** The one exception to
   the representation rule; it keeps the zero-shot ranking arm.
3. **bindingdb_kd, kiba, lhasa_herg get a distance search** with their general search's settings
   and budgets of 100 to 500 features. DONE in `build_decided_grid.py`; all 41 datasets now have
   a distance search in `decided_grid.json`.
4. **Augmentation comparisons are dropped on every dataset where Step 8 cut augmentation.** An
   experiment that compares with and without added features (opig_shared_vectors_remedy's arms
   B and D, c02's augmentation arms, c21, c06, noise `base_k_aug`) runs that comparison only on
   ginkgo_colloidal and ginkgo_hic (distance search), and skips it elsewhere with a printed
   reason.
5. **chembl_qsar (opig_activity_cliff_qsar's three ChEMBL targets).** Decided by Claude on her
   instruction "use similar datasets to decide ... run really simple local fits ... decide on 2
   models": see 2.3.1 below for the fits and the pick. Its other settings are taken from herg_ki
   (the most similar decided dataset: small-molecule binding affinity measured as Ki).
   ⛔ opig_activity_cliff_qsar uses its own 7 representations (`DEFAULT_REPS`: ecfp4, maccs, pdv,
   sns, mordred, mol2vec, avalon); `DATASET_REPS['chembl_qsar']` holds a pruned 10. Use the 10.
6. **Pruning off** in every per-fold search (already hers; see 2.1).
7. **trastuzumab gets a per-fold AUROC table** (hybrid against best single representation).
8. **A second hybrid whose selected features equal the first's is skipped.**
9. **flab_expression_koenig keeps wglobal_power2.** Its stored search ran it; a cut needs a
   reason.

### 2.3.1 chembl_qsar model pick (done 2026-10-05)
**lightgbm + dnn**, settings copied from herg_ki; in `decided_grid.json`. Simple local fits:
default settings, chembl_qsar's pruned representations, 5 scaffold folds, mae (lower is better).
Best mae per model, with the representation it came from:

| target | molecules | representations fitted | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|---|---|
| mpro | 1,924 | 9 | lightgbm 0.4610 | svm 0.4736 | dnn 0.4867 | xgboost 0.4918 | rf 0.5001 |
| fxa | 3,605 | 4 | lightgbm 0.6852 | dnn 0.6979 | xgboost 0.7048 | svm 0.7138 | rf 0.7157 |
| d2 | 6,333 | 2 | lightgbm 0.5180 | dnn 0.5188 | svm 0.5281 | xgboost 0.5286 | rf 0.5316 |

grover is not installed locally, so 9 of 10 at most. fxa and d2 were cut short to finish the
handoff. Chat 2 can rerun `fit.py` (copy kept at `scratchpad/precompute_cuts/qsar_fit.py`) on the
cluster if she wants all 10 on all three.

### Answered 2026-10-05, second round
10. **Noise level 0 = the labels as delivered.** DONE: `src/kirby/noise_spec.py`,
    `make_noisy_train_labels` and `TrainingNoise.held_out` return the clean labels at level 0 on
    class labels (checked: 0 of 300 flipped at level 0, 31 of 300 at 0.1). Dirty-run bug 6 closed.
11. **cgr_comparison's two groups:** the dataset's pruned 10 without cgr_morgan (9 representations)
    against the pruned 10 plus chytorch_graphormer and cgr_smiles_lm (12). ⛔ The cached
    chytorch_graphormer matrix was all zeros in the dirty run (bug 9): check the server cache first.
12. **ginkgo's paratope and structural stay out of hybrids** (Step 8). Still scored alone.
13. **ginkgo_colloidal and ginkgo_hic keep augmentation on the distance search only.** The
    general-search gains (0.410 to 0.414, 0.582 to 0.596 spearman) are inside the 3% band.
14. **herg_ic50 takes herg_ki's models (svm + dnn) and settings.** DONE in
    `build_decided_grid.py` (`decided_grid.json` now has 42 datasets). Still to do per experiment
    (n17, n19, n22, n26, n29, n36 and every other noise experiment run on it): each per-fold search
    runs on the experiment's own rows and folds, with all measurements of a molecule on one side
    (dirty-run bug 11). On 9,802 molecules svm fits; on 12,008 measurement rows it is skipped.
15. **flab_expression_koenig and flab_binding_koenig: one group per mutated position.** DONE:
    `tests/antibody_baseline_survey.py`, strategy `position` (225 groups of 19 variants; checked
    through `load_data`). Was ProteinGym's modulo-5, which gave exactly 5 groups = the 5 folds.
    ⛔ abagym_Ang2, abagym_EGFR and abagym_VEGF still use modulo-5; check whether they hit the
    same failure and ask her.
16. **f29 is framework and outside the suites** (the suites are complexity, noise, similarity,
    federated, opig; never framework). **c23 keeps each dataset's two models**, no exemption.
17. **0/1 features are never scaled; continuous features are standardised; everywhere**, including
    the 0/1 features inside a hybrid and in every experiment script that scales on its own.
    NOT DONE: it touches about 8 sites in `find_best_hybrid.py` (`StandardScaler()` at the hybrid
    scorer, the nested-CV held-out score, the tuning pipeline, the geometry paths) and about 20 in
    `hybrid.py`, and the per-experiment scripts in dirty-run bug 3. Do it in chat 1 (2.0) as one
    column-wise scaler used everywhere: a feature is 0/1 if every training-row value is 0 or 1.
    Then the per-experiment scripts in their suites. Numbers it changes (herg_ki, rmse): svm on
    ecfp4 0.6841 scaled to 0.6122 unscaled; knn on atom_pair 0.6452 to 0.6193.
18. **float32 for every representation, on load and on save.** DONE in shared code:
    `_shared.REP_DTYPE`, `as_rep_dtype`; `_finalize_load_data` casts every representation and
    augmentation matrix; `_cache_rep`, `_cache_rep_dict` reads and `_save_rep_npy` writes cast;
    `rep_cache.compute_or_load_reps` saves float32 (it saved float64) and casts on read. Still to
    do per experiment: scripts that convert to float64 themselves: n32 (`cached_scaled`), n34:616,
    n35:719, n36:340, n37:563, n38:629, and any other found while aligning.
19. **n37's noise estimate: one estimate per representation, side by side**, with each model's
    share against each estimate.
20. **c20, c31, c32, c43 and simopt stay out.** Not aligned, not run.
21. **suzuki: power_2 dropped.** DONE in `build_decided_grid.py`. Checked in the stored rows:
    power_2 and performance_weighted score identically on 132 of 144 general configurations per
    model and 147 of 180 distance ones.

### Still open
- **The 22 experiments whose selection differs by design (2.5): DECIDED 2026-10-05**, Adelaide
  "yes that sounds good" to the grouping below, including both moves marked ★.
  **n09 decided:** on a dataset whose two models do not include dnn, n09 runs its own per-fold
  dnn search on the fast tier inside that dataset's cuts. Built: `decided_grid.
  fast_model_search_kwargs(ds, 'dnn')` (each axis = fast list intersected with the decided list;
  the decided list where that is empty, named in its second return value) and
  `search_on_fold(..., fixed_grid=...)`, which uses those axes as given. Not yet wired into n09.
  - (A) No model or no selection; keep as is: scaffold_similarity, c01, c04, s01, n14, s09.
  - (B) The question is another comparison; keep the comparison, but on the dataset's two models
    and per-fold searches: c05, c06, c12, c15, c17, c21, c23, c24, fed02, ★s11, ★n09.
    c15 also drops its ridge arms (ridge is removed everywhere). c17 and c15 use rf only today.
    ★s11: its model is the Gaussian-process surrogate of the Bayesian optimisation, not a
    downstream model; the comparison is between feature spaces under that one surrogate. Its
    hybrids become 2 per decided model from the per-fold search.
    ★n09: its methods (label-noise training methods) are neural-network training schemes, so
    dnn is fixed by the question. On datasets whose two models do not include dnn, the per-fold
    search does not score dnn; n09 then needs its own per-fold search for dnn (decided grid,
    `models=['dnn']` passed explicitly, which `apply_decided_grid` would refuse today) or skips
    those datasets. She chose the own search (above).
  - (C) Built on the old 2 general + 1 distance rule or on one model; switch to 2 + 2, with
    distance hybrids only where neighbours are used: trastuzumab, ginkgo, s05, s06, n11.
- **The noise fold rule.** Not decided anywhere. Her question 2026-10-05: run the noise
  experiments on one fold first, then the promising ones on all 5? See 2.12.
- **abagym's modulo-5 grouping** (item 15).

## 2.4 How to align ONE experiment (do exactly this, every experiment)

1. Read its audit section (2.6) and the experiment file IN FULL, and every helper it imports.
2. Models: only `decided_models(ds, 'general')`; knn only via the distance search where Step 9
   runs it. Delete `DOWNSTREAM_MODELS`, `MODELS_LIST`, `get_model_factories(DOWNSTREAM_MODELS..)`,
   any hardcoded 'rf' default (`extract_top3_base_reps`, `top_base_reps(search, 'rf')`).
3. Searches: every `find_best_hybrid` call goes through `_apply_dataset_search_speed` with the
   REAL dataset name, with axes left None. A decorated name (`<ds>_n<rows>_s<seed>`,
   `<ds>__c15c_b250`) must be passed as `condition_suffix` or `cache_namespace` instead, so the
   lookup sees the real name. A custom tier name for a distance search must contain "distance".
   An objective='similarity' search follows the same rule.
4. No whole-dataset search, anywhere. Remove `get_best_search_for_downstream`,
   `get_rule_hybrids`, `get_fold_pool_recipes`, `get_precomputed_search`, `get_fitted_hybrid`,
   `get_fitted_pool_hybrid`, `noise_base_reps_by_model`, `extract_top3_base_reps`,
   `data['precomputed_searches']`, `fold_reselection`/`build_fold_reselection_input` on a stored
   whole-dataset search, and every `run_all` start-up search. (`fold_reselection` on per-fold
   held-out scores of hybrids built inside each fold is fine.)
5. Selection: `fold_standard_selection` per outer fold (folds from
   `kirby.find_best_hybrid.nested_cv_outer_folds`), unless the experiment is an approved
   exception. Rebuild each chosen hybrid with `build_fold_hybrid(..., **fold_hybrid_scoring_kwargs(
   blueprint, model=..., metric_fn=..., lower_is_better=...))` on the fold's training rows and
   score on the held-out rows. Pass the fold search's own representation scores, never the
   whole-dataset ones (dirty bug 1).
6. Skip the second hybrid if its selected features equal the first's.
7. Pruning off: no `adaptive_pruning_schedule`, no `model_pruning_aggressiveness`.
8. Augmentation: only `decided_aug_conditions(ds, tier)`.
9. Representations: only `DATASET_REPS[ds]['general']` (+ `['augmentation']` where Step 8 keeps
   it). Never filter `data['reps']` before a hybrid builder; filter per-representation loops.
10. Remove any `KIRBY_NO_COLD_SEARCH` the script sets for itself.
11. Result file: per fold, models, base representations, hybrids (allocation, importance, budget
    in features, evaluator, representation count, augmentation condition), the training-row score,
    the held-out score, `selection_scope: 'fold_train_rows'`, the metric name and direction.
    Bump the checkpoint identity so old partial files are not read back.
12. The kNN record (Part 1) for any neighbour setting.
13. Fix that experiment's dirty-run bugs (2.7).
14. Folds: the number of outer folds must come from one setting, never a literal. Adelaide is
    deciding the fold rule in a separate chat (2026-10-05); the code must work for any count,
    including 1. Read it from the experiment config (`config['n_splits']`, set by `get_config`) and
    pass it to `nested_cv_outer_folds`, `fold_standard_selection` and every per-fold search (the
    per-fold store is already keyed on it, `ocv{n}f{i}s{seed}`). Never `range(5)`,
    `n_splits=5`, `GroupKFold(5)`. A count of 1 is a single held-out split, not 1-fold CV: check
    `nested_cv_outer_folds` handles it (memory `project_nfolds1_single_split`) and that every
    summary that takes a spread over folds says "one fold, no spread" instead of dividing by 0.
    Files with a literal fold count found 2026-10-05 (each to check; some are argument defaults):
    fed03 (6 places), federated_helpers (3), s11 (2), opig_shared_vectors_census (2), c09 (2),
    s06, n22, n06, c33, c23. Inner folds of the search itself (`n_folds` in find_best_hybrid) are a
    separate setting; leave them unless she says otherwise.
15. `python -m py_compile`, then the time-boxed run (2.8).

⛔ Part 1 says "set `KIRBY_COLD_SEARCH_MAX_CONFIGS=0`" so no search starts inside an experiment.
That holds for f38 and the whole-dataset searches only. Per-fold searches are built by the
experiment (R6) and the engine never refuses them since `fb1df54`.

## 2.5 The 22 experiments whose selection differs by design (ask yes or no for each)

From the audit's R4 "by design" rows; reasons are the audit's, re-check in code. "Yes" = keep its
own selection (R1, R2, R3, R6 still apply). "No" = change it to 2 base + 2 hybrids.
- opig_scaffold_similarity: fits no model; compares five hand-built mutation-difference
  representations against substitution matrices.
- opig_trastuzumab_enrichment: scores every representation for every model (the question is which
  embedding separates binders) and keeps 3 hybrids (2 general + 1 distance).
- opig_ginkgo_developability: 3 hybrids per model per fold and 1 best single representation re-picked
  on the other folds; every representation scored.
- fed02: base representations are the five silo representations, 3 hybrids each built three ways
  (all silos, top three, clustered); the silo design is the question.
- c01: no selection; every representation scored by itself.
- c04: no models; ten representations and weighted concatenations scored by 10-neighbour prediction.
- c05: one hybrid and one representation per model at each sample size.
- c06: one hybrid per budget per capacity setting; no representations.
- c12: one representation per modality, side by side; no hybrids.
- c15: hybrids against PCA, random features and the full concatenation, rf and ridge (ridge is
  removed everywhere; check).
- c17: one model (rf), 3 hybrids, compared with PCA, mRMR, recursive elimination, random.
- c21: every representation plus up to five extra-feature conditions; fixed-recipe hybrids.
- c23: one hybrid per evaluator condition; no representations.
- c24: one recipe per fold, chosen on fewer folds.
- s01: no selection; every representation's distance described.
- s05: knn only; 2 general + 1 distance per family (prediction, similarity); 2 representations.
- s06: knn only; up to 6 hybrids; every representation an arm.
- s09: aggregator; one hybrid minus one representation per dataset.
- s11: Bayesian optimisation with a Gaussian process; 3 hybrids chosen by rf; one representation.
- n09: dnn only; 1 hybrid plus its importance-weighted copy; 2 representations.
- n11: hybrids used only as spaces for noise detection, up to 6 per model.
- n14: no models; rf's top 2 representations as similarity spaces; no hybrid spaces.

## 2.6 Every experiment: where its audit section is, and its state

Audit = `scratchpad/precompute_cuts/EXPERIMENT_AUDIT.md`, line range. The suite run lists:
complexity from `submit_dataset.sh` `resolve_experiments` (c06 c19 c22 c07 c18 c01 c02 c08 c15 c04
c13 c11 c12 c16 c09 c24 c05 c23 c33 c17 c21, plus cross jobs c03 and c14); similarity s04 s03
s01 s12 s06 s13 s02 s10 s05 s08 s11, plus cross s09; federated fed01-fed05; noise
`noise_helpers.SINGLE_DATASET_EXPERIMENTS` (37, includes c10 and n06); OPIG the 11 rows of
`OPIG_JOBS` in `submit_everything.sh`.

⛔ In no run list, so not audited: c20, c31, c32, c43 (`tests/experiments/complexity/`) and simopt
(`tests/experiments/similarity_optimized_hybrid.py`). Adelaide expects them to run; why they are
missing is in 2.3 "still open". (fed03b is part B of fed03, and the dirty run's `_fullsweep` jobs
were the same noise experiments run with the full neighbour sweep to measure cost; both are
covered.)

OPIG (edited 2026-10-05, uncommitted, compile; re-align to the 2.3 answers):

| experiment | audit lines | state |
|---|---|---|
| shared_vectors_census | 744-885 | per fold; 2 general hybrids per model, knn 2 from distance search; adds `--n-outer-folds`; fixed a crash in `print_worst` |
| scaffold_similarity | 886-969 | no change (no model, no search) |
| humanness | 970-1110 | per fold; 2+2 plus distance1; pruning off; onehot out of the search; drops `KIRBY_NO_COLD_SEARCH` itself (not needed now) |
| graphinity | 1111-1223 | per fold 2+2, no distance; transfer_foldx restricted to svm+dnn, runs no search; aac wrongly added as a representation |
| trastuzumab | 1224-1364 | per fold; whole-dataset AUROC table removed, per-fold one still to add; 13 representations |
| cgr_comparison | 1365-1520 | per fold 2+2, no distance; pruning off; each pool filed under its own namespace (`ocv5f0s42__cgr_<pool>`) because representations are not in the stored-result key |
| ginkgo | 1521-1637 | per fold; 3 hybrids; uses the competition's cluster folds, not `nested_cv_outer_folds` |
| qsar | 1638-1766 | per fold 2+2 + distance1; `condition_suffix=<target>`; stops at once (chembl_qsar undecided) |
| blueprint_transfer | 1767-1922 | per fold 2+2, no distance; transferred recipes still from one search over all logd rows (the experiment's design); only models decided on both datasets (svm, dnn) |
| scaffold_hybrid | 1923-2061 | per fold; distance tier renamed `general_distance__scaffold_cmp` |
| shared_vectors_remedy | 2062-2210 | decided grid via `nested_cv_evaluate` directly; distance tier renamed `general_distance_shared_vec`; 2 base representations per fold added |

Federated, complexity, similarity, noise: NOT STARTED. Order she set: opig, federated, complexity,
similarity, noise; show her each suite's changes before the next.

| suite | experiment | audit lines |
|---|---|---|
| federated | fed01 | 2211-2337 |
| federated | fed02 | 2338-2480 |
| federated | fed03 | 2481-2603 |
| federated | fed04 | 2604-2751 |
| federated | fed05 | 2752-2873 |
| complexity | c01 | 2874-2965 |
| complexity | c02 | 2966-3111 |
| complexity | c03 | 3112-3239 |
| complexity | c04 | 3240-3348 |
| complexity | c05 | 3349-3477 |
| complexity | c06 | 3478-3592 |
| complexity | c07 | 3593-3741 |
| complexity | c08 | 3742-3909 |
| complexity | c09 | 3910-4067 |
| complexity | c11 | 4068-4191 |
| complexity | c12 | 4192-4295 |
| complexity | c13 | 4296-4437 |
| complexity | c14 | 4438-4594 |
| complexity | c15 | 4595-4759 |
| complexity | c16 | 4760-4901 |
| complexity | c17 | 4902-5035 |
| complexity | c18 | 5036-5175 |
| complexity | c19 | 5176-5312 |
| complexity | c21 | 5313-5439 |
| complexity | c22 | 5440-5562 |
| complexity | c23 | 5563-5681 |
| complexity | c24 | 5682-5797 |
| complexity | c33 | 5798-5991 |
| similarity | s01 | 5992-6092 |
| similarity | s02 | 6093-6235 |
| similarity | s03 | 6236-6378 |
| similarity | s04 | 6379-6524 |
| similarity | s05 | 6525-6652 |
| similarity | s06 | 6653-6773 |
| similarity | s08 | 6774-6939 |
| similarity | s09 | 6940-7049 |
| similarity | s10 | 7050-7194 |
| similarity | s11 | 7195-7342 |
| similarity | s12 | 7343-7500 |
| similarity | s13 | 7501-7617 |
| noise | c10 | 7618-7743 |
| noise | n01 | 7744-7907 |
| noise | n02 | 7908-8044 |
| noise | n06 | 8045-8181 |
| noise | n07 | 8182-8359 |
| noise | n09 | 8360-8493 |
| noise | n11 | 8494-8627 |
| noise | n12 | 8628-8789 |
| noise | n13 | 8790-8936 |
| noise | n14 | 8937-9072 |
| noise | n15 | 9073-9229 |
| noise | n16 | 9230-9351 |
| noise | n17 | 9352-9497 |
| noise | n18 | 9498-9633 |
| noise | n19 | 9634-9773 |
| noise | n20 | 9774-9908 |
| noise | n21 | 9909-10056 |
| noise | n22 | 10057-10200 |
| noise | n23 | 10201-10344 |
| noise | n24 | 10345-10487 |
| noise | n25 | 10488-10633 |
| noise | n26 | 10634-10805 |
| noise | n27 | 10806-10957 |
| noise | n29 | 10958-11082 |
| noise | n30 | 11083-11251 |
| noise | n31 | 11252-11387 |
| noise | n32 | 11388-11529 |
| noise | n33 | 11530-11699 |
| noise | n34 | 11700-11855 |
| noise | n35 | 11856-12005 |
| noise | n36 | 12006-12153 |
| noise | n37 | 12154-12312 |
| noise | n38 | 12313-12457 |
| noise | n39 | 12458-12583 |
| noise | n40 | 12584-12723 |
| noise | n41 | 12724-12874 |
| noise | n42 | 12875-13043 |

Shared code that every suite also needs (from the audit's shared section, lines 106-743):
- Each `run_all` (complexity, similarity, noise, federated helpers) builds whole-dataset searches
  at start-up. Remove them; per-fold searches are built where used.
- `own_search_kwargs` (c05, c07, c33, c15c) still names the fast grid and five models. Make it
  take the real dataset name and use the decided grid.
- `c02` imports `_apply_small_dataset_models`, now unused by `_shared`; delete both once c02 is fixed.
- `noise_helpers.pool_recipe_per_downstream_model` reads `get_rule_hybrids` (2 general + 1 distance
  + 2 base = 10 configurations for two models, not 8).
- `get_config` hands every experiment five models and eight budgets.
- `should_skip_experiment` runs after the start-up searches; complexity and federated never call it.
- Four launchers set four pruning defaults (`run_dataset.py`, `submit_dataset.sh`,
  `submit_everything.sh`, `opig_one.sh`); with pruning off they must stop passing one.
- `submit_everything.sh:136` still passes `PM` to opig_cgr_comparison, which now ignores it.

## 2.7 The dirty-run bugs, and what to do with each while aligning

From `DIRTY_RUN_RESULTS.md` (local run 28 Sep-2 Oct on older code). Status as read 2026-10-05.
"Fix while aligning" means: in each experiment the bug names, fix it when that experiment is
aligned, and record it.

| bug | what it is | status | do |
|---|---|---|---|
| 1 | hybrid rebuild ranks representations by whole-dataset scores (`blueprint.rep_scores`) | fixed only in the 10 OPIG scripts and `fold_standard_selection` paths | every rebuild passes the fold search's own scores; listed in n06, n13, n17, n19, n22, n23, n26, n29, n31-n34, n36, n38-n41, c07, c08, c19, s02, s05, s06, s10(?), `fold_similarity_spaces`, `sx.build_spaces` |
| 2 | nested-CV rebuild ignored the recipe's representation count | fixed 2026-10-02 (`7e17900`, after the dirty run started) | confirm in the smoke run on buchwald_hartwig |
| 3 | 0/1 features scaled differently in four places | OPEN, her decision: unscaled everywhere or standardised everywhere | list in the bug; decide before aligning similarity |
| 4 | svm on unscaled labels | fixed `fb1df54` (search and distance scorer); NOT fixed: plain `SVR` at `_shared.py` near c06's model list | fix the rest when aligning c06 |
| 5 | float32 vs float64 give different tree scores | OPEN, her decision: one precision | n32-n41 convert to float64 |
| 6 | "noise level 0" flips labels on classification | OPEN, her decision: skip the flip or exactly 0 | n09, n13, n20, n21, n23, n24, n25, n42 |
| 7 | dropping "self" can drop a twin | partly fixed (n02, `736ec93`) | `similarity_helpers.py:334`, `noise_mitigation_methods.py:487`, n07 |
| 8 | noise jobs ignored the local 5-representation cut | local only | ignore for the cluster |
| 9 | missing representations in the local setup | local only, except: chytorch_graphormer cache all zeros; ddg_combined cache holds 429 of 534 molecules | check both caches on the server |
| 10 | flab_expression_koenig has 5 groups | OPEN, her decision: how to group it | similarity search fails; c05, c24, n01, n25 |
| 11 | herg_ic50: search on 12,008 measurements, experiments on 9,802 molecules; svm skipped above 10,000 rows | OPEN; herg_ic50 also has no DECISIONS.md row | n17, n19, n22, n26, n29, n36 |
| 12 | scripts save only at the end | fix while aligning (write per fold) | c31, c32, n06, n42, fed05, opig_humanness, scaffold_hybrid, cgr_comparison, ginkgo, trastuzumab |
| 13 | smaller faults: wrong metric stamp (fed01, fed03a, fed04); two row formats (n30-n35); controls that equal the method (quantile_clip, huber_weight); knn_smooth = label_smooth on class labels; budgets filled with zero-gain features (c31, c32, fed04); significance tests (c14, c31, c32, s09, which she excluded); stale result files (c10, n07) | fix while aligning | per experiment |

Per-experiment bugs from Part 2 of DIRTY_RUN_RESULTS.md (fix while aligning that experiment):
c05 (silent failed draw; 3-parameter curve on 3 points), c08 (ridge "low-capacity" model worse
than the mean), c11 (onehot_components in hybrid not in singles; re-picked ensemble uses held-out
scores), c13 (random-feature control never enters the hybrid), c15 (unequal feature counts;
"full dimension" is PCA; rf differs from the search's), c16 (full side is a 1,150-feature hybrid;
rf depth 10), c17 (mRMR never runs above 200; unequal widths; rf max_features), c18 (three
differences at once on buchwald), c19 (asks 100 builds 250), c20 (rf order), c21 (decoy features
admitted; `top_n` meaning), c23 (evaluator changes width), c24 (unexplained 2 rmse gain on
buchwald fold 0), c32 (cannot run on auc; fails its own engine check), c33 (two budgets build the
same width), fed01 (failed silos written as score 0.0 with 1 feature; warnings hidden), fed03a
(test folds 89% Lhasa), fed04 (zero-importance selections; chemical clusters dropped), fed05
(ignores the dataset name; 0-representation silo not a failure), s01 (SALI divides by zero
distance), s03 ("mutual information" sums one-component estimates; up to 2.13 nats on a two-class
label), s06 (best single chosen on the numbers it is compared with), s09 (its agreement check
reads a key no input has), s10 (sign flipped twice on rmse; Gaussian process with no mean term),
s11 (5% start set is one group on buchwald; stored width is the last seed's), s13 (PCA controls
standardised twice), simopt (three distances), n01 (compared against the 3 weakest
representations), n09 (needs a stored dnn search; drops singles when missing), n11 (detection
spaces on 10 representations), n13 (classification grid not cut), n15 (one configuration gives 0
by construction), n16 (column name; no level-0 floor), n18 (ties counted as losses; mean
baseline), n20 (marked done with no file), n21 (hybrid distance written twice), n23 (720
"Unknown label type: continuous" on class labels), n24 (outlier_p10 run twice under two names),
n25 (random comparison size), n37 (noise estimate from rf's top representation only; negative
intercepts averaged; SIMEX one draw), n40 (retuning never tries the default), n42 (level 0
first), opig_qsar (sns never built), opig_shared_vectors_remedy (chemberta cache has no ties;
two runs never saved), opig_trastuzumab (zero-shot sign never checked).

Decisions from DIRTY_RUN_RESULTS.md Part 3 that are still hers: bugs 3, 5, 6, 10, 11 above; which
representation n37's noise estimate comes from; dropping significance tests (she already excluded
them everywhere, so remove them; tell her in one line).

## 2.8 Smoke screen (after each suite, and once at the end)

- Per experiment: normal settings, time-boxed, locally (`KIRBY_CACHE_DIR=/Volumes/seagate/kirby_cache`,
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KIRBY_LGBM_THREADS=1`) or in an srun block for her. Record
  slow ones and why. Never wait for one to finish; never shrink a setting.
- Run with `KIRBY_NO_COLD_SEARCH=1` set, as the cluster does, to prove per-fold searches build.
- OPIG run 2026-10-05 (5 and 15 minute boxes): no crash in any. Did not reach the new code within
  15 minutes because protein embeddings were still being built locally: humanness, trastuzumab,
  graphinity, cgr_comparison, ginkgo. These need an srun on the cluster. qsar stops at once
  (chembl_qsar undecided, intended). shared_vectors_remedy stopped on its own existing check: the
  local cache's chemberta and pdv no longer match their builders (local cache only).
- A full smoke screen of every experiment after all suites, as one list with pass, slow (why),
  or failed (error).

## 2.9 Last: analyse COST_BREAKDOWN.md

Do this after alignment, because the alignment changes most of it. For each experiment that hit
the 3-hour cap, recompute the cost under the new rules and the kNN record, and say what is left:
- B (full search per outer fold, 8,160 configurations x 5 inner folds): now every experiment
  searches per fold, but on the decided grid (herg_ki general 224 configurations, distance 360) and
  each fold search is shared by every experiment on that dataset through the result store.
- A (neighbour sweep, 268 settings): Part 1 cuts it to 138 (full_knn) or 66 (fast_knn).
- C (noise grid): depends on f37 (which damage kinds and levels she keeps).
- D (duplicate controls) and section 3 of the file (duplicated work that changes no result):
  list each, say what fixing it saves, and fix only with her yes.
- E (dnn never cached).
Give per experiment: hours before, hours after, what drives what is left. Hours are estimates; say
so. 1,000 hours is 42 days of one job.

## 2.10 Carried from the old Part 2 stub (still open)

- The importance power and the weight rescaling for the similarity experiments (three ways each);
  Part 1 axis 3 closed it for the noise suite only.
- Whether a base representation is ever passed raw, so set-overlap distances can reach it.
- f36 (which label smoothings survive) and f37 (which damage kinds and levels): she analyses.
- The conditions f38 computes no rows for (gaussian, laplace, student_t kinds, grouped_shifted).
- Activity-cliff prediction, blocked on three decisions in `F38_DECISIONS.md`.
- Every svm number changed with `fb1df54`; the two-model picks were kept.

## 2.11 How to run it

This is a large workflow. Suggested shape, hers to approve: one agent per experiment edits only
that experiment's file, following 2.4 exactly; a second agent per experiment re-reads the diff
against the six rules and the dirty-run bugs and tries to refute "compliant"; then the time-boxed
run. Shared-code changes are made once, by the lead, before the suite starts; no per-experiment
agent edits `_shared.py`, `decided_grid.py`, `dataset_policy.py`, `src/kirby`, or a submit script.
Every agent's report says, per rule: before (file:line), after (file:line), or why unchanged.
The lead reviews every diff before showing her a suite.

## 2.12 Noise experiments on one fold first (her question, 2026-10-05; not decided)

Her idea: run every noise experiment on one outer fold, then run what looks promising on all 5.
Cost: the noise grid is about 1,100 to 1,450 fold-level units per dataset at 5 folds
(COST_BREAKDOWN.md, setting C), so one fold is about 220 to 290.
What one fold cannot do: give a spread. On herg_ki the fold-to-fold spread of the best
configuration is 0.044 rmse (DECISIONS.md Step 2), larger than most differences between cleaning
methods in the dirty run (0.002 to 0.02). A one-fold screen can therefore only drop what is clearly
worse; it cannot rank close methods. Her rule (F38_DECISIONS.md item 9): no cutoff fixed in
advance; she sees the one-fold results, then says what goes to 5 folds.
Questions for her: the same fixed fold (fold 0) for every dataset and experiment; whether the
screen covers every dataset or a few; and which labels the searches see (2.3 "still open").

