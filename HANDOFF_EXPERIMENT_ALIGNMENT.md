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

# Part 2: the rest of the experiment alignment

**Not started. This section is for the next piece of work.**

Known items that belong here, carried out of Part 1:

- **The importance power for the similarity experiments.** Still splits three ways: 1.0 under
  manhattan through `preprocessing.weight_exponent` in s02 and s08; 0.5 whatever the distance
  through `_shared.importance_feature_weights` in s05, s06, s10, s13; 0.5 hardcoded in s04 which
  ignores the environment variable. Axis 3 closed this for the noise suite only.
- **The weight rescaling, which also splits three ways.** Mean 1 in s02, s08 and s10 by default.
  Sum 1 in s13. None in s04, s05, s06.
- **Whether a base representation should ever be passed raw**, so set-overlap distances can reach
  it. The largest distance gaps anywhere depend on this.
- **Per-fold alignment of every experiment** against `scratchpad/precompute_cuts/DECISIONS.md`:
  which models it uses, whether it reads per-fold or whole-dataset searches, every extra
  `find_best_hybrid` call, every augmentation step. The scope is in
  `HANDOFF_PER_FOLD_ALIGNMENT.md` section 8 item 3.
- **The concurrency question**, Adelaide's to decide: how to stop two experiments building the same
  per-fold search at once, and what happens when a job times out. Options she named were
  overlapping start times, or dependencies so the complexity experiments search first. Nothing
  decided.
- **The noise-experiment fold rule**, Adelaide's to decide: how many outer folds, and how each
  noise experiment formalises its own search. She postponed it.
- **f36 and f37**, which Adelaide analyses. From f36: does any smoothing that reads the neighbours'
  labels survive — `label_smooth`, `knn_smooth`, n02's weights. From f37: which damage kinds and
  which levels she keeps. The level count multiplies every refit count in Part 1.
- **The conditions f38 computes no rows for**: gaussian, laplace, the student_t kinds and
  grouped_shifted, because every label moves and the corrupted set is not defined. Recovery would
  still be defined there and f38 does not compute it.
- **Activity-cliff prediction**, blocked on three decisions in `F38_DECISIONS.md`: finding the
  cliff partner among the neighbours or predicting the error on cliff molecules; which pairs count
  as a cliff; and the activity gap.
- **The svm label-scaling fix** (`fb1df54`, `SVRScaledTarget`) changed numbers wherever svm is used
  and the label spread is large. The two-model picks were kept, but every svm number is affected.
  `HANDOFF_PER_FOLD_ALIGNMENT.md` section 6 has what was and was not verified.
- **suzuki's allocation methods are HELD**: power_2 and performance_weighted both score exactly
  21.006 rmse, which looks like an untrained fallback. Cause not checked.
