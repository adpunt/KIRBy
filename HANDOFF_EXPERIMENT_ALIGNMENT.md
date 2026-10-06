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

## 5b. The results-audit session, and how it relates to this handoff

A second session is fixing faults found in the server results pulled 2026-10-03. Its record is
`scratchpad/results_audit_2026_10_03/TRACKER.md`. Read it before editing any experiment, and never
delete a row from it; update a row's status and add your commit hash.

**Precedence, Adelaide 2026-10-05: the decisions in `KNN_SEARCH_SPACE_RECORD.md` overrule.** Where
an audit fix and a kNN decision touch the same thing, the decision wins and the fix is re-applied on
top of it. The audit's own status value for this is SUPERSEDED, which its tracker header already
defines as "the experiment-alignment pass replaces it".

**Already committed by that session. Do not undo any of these:**
`e58106f`, `ccab773`, `e86df5c`, `e7d9ae2`, `473855a`, `6056c8d`. Run `git show --stat` on each.
Between them they fix: AUC computed from 0/1 predictions instead of probabilities (s04, s05, s08,
s10, s12, c07); fed05 running three times; c06 settings not reaching the model; mislabelled metrics;
folds dropped or misnumbered; p-values and confidence intervals removed; and the width actually
built per fold now stored beside every budget.

**In progress when this was written. Check `git log` before touching:**
`tests/experiments/complexity/` c02, c04, c05, c07, c08, c09, c15, c16, c17, c18, c19, c22, c24.

**Two fixes that session asked for and could not make**, because this session's edits were in the
files. Both files are now committed (`ec03a62`), so they are unblocked. They belong to that session,
not to this handoff:
- `tests/experiments/_shared.py` `evaluate_single_rep_cv` (about line 2930): after
  `X_val = scaler.transform(...)`, clip the validation features to the training range. One adme_hlm
  validation molecule sat 83 million training standard deviations out and the dnn's rmse ran to the
  thousands.
- `similarity_helpers.py` `compute_sali` and `compute_sali_scalable`: SALI divides by
  `(1 - sim + 1e-6)`, so a pair with identical features scores its activity difference times a
  million and dominates. Drop identical pairs and count them separately. Which similarity SALI uses
  is Adelaide's to choose, not theirs.

**Where that session's work changes a kNN decision.**

- **Duplicate hybrid recipes (its item 5, delegated by Adelaide 2026-10-05): when two selected
  hybrids are the same recipe, take the next distinct recipe.** This matters here. Axis 1 records
  that the best hybrid and the second-best share 0.805 to 0.972 of a molecule's ten nearest
  neighbours, and that no cut could rest on that because f38 built the second one as "the next-best
  stored recipe that differs", not by the rank-2 rule. Once the distinct-recipe rule is in and the
  per-fold precompute has been rebuilt, the second hybrid becomes a genuinely different recipe.
  ⛔ Revisit axis 1's keep-or-cut for it at that point. Do not act on f38's overlap numbers.
- **Datasets with fewer than ten groups (the three abagym sets and four ginkgo sets) are being made
  to skip c05 and c07**, because splitting inside a group would break the no-random-split rule.
  Those seven are all under 1000 molecules, so all seven are `full_knn`. Nothing in the kNN
  decisions changes for them, but an analyser must expect them to have no c05 or c07 rows.
- **AUC from probabilities rather than 0/1 calls.** Section 2's warning about metric direction
  stands, and this fix is why: a detection or prediction score read off hard calls is not the same
  number as one read off probabilities. Anything comparing a stored score with a fresh one has to
  know which side of `473855a` it came from.

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
| 1 | Part 1 (the kNN record) if the other process has not done it; the launcher checks in 2.8.1 for anything it changes; then the shared code in 2.6 "shared code every suite needs"; the 0/1 scaling rule (2.3 item 17); the duplicate-hybrid check (2.1); distance hybrids in the noise neighbour spaces (2.3 item 1) | shared tests pass; one experiment from each suite smoke-run |
| 2 | OPIG re-aligned to the 2026-10-05 answers (2.6 table), then federated | her review of both |
| 3 | complexity | her review |
| 4 | similarity | her review |
| 5 | noise, after the fold rule (2.12) is decided | her review |
| 6 | the smoke screen of every experiment (2.8) and every launcher (2.8.1), then COST_BREAKDOWN.md (2.9), then the results-audit items (2.13) | one list: pass / slow (why) / failed (error); the cost table |

### Dynamic workflows: Adelaide's explicit go, 2026-10-05

`F38_DECISIONS.md` item 11 says no dynamic workflows without her explicit go. **This paragraph is
that go, for the uses below and no others.**

- **Chat 1: no workflow for the edits.** Every chat-1 change is to the same few shared files
  (`find_best_hybrid.py`, `hybrid.py`, `_shared.py`, `similarity_space.py`); parallel editors would
  overwrite each other. Edit them in order, in the chat. **Before showing her the changes, run a
  read-only review workflow:** one agent per shared change (each kNN axis, the 0/1 scaling rule,
  run_all start-up searches removed, `own_search_kwargs`, `get_config`, the duplicate-hybrid
  check, distance hybrids in the neighbour spaces, the launchers' pruning, the abagym grouping
  check). Each reads this file's rule for that change and the diff, and tries to show it is wrong
  or incomplete, with file:line. No agent edits. Then fix what they confirm and report what was
  found.
- **Chats 2 to 5: a workflow for the edits.** One agent per experiment edits only that
  experiment's file, following 2.4 exactly. A second agent per experiment re-reads the diff against
  R1-R6, the 2.3 answers and that experiment's dirty-run bugs (2.7), and tries to refute
  "compliant". No agent edits shared code, a submit script or `src/kirby`; anything shared comes
  back to the chat. The chat reviews every diff itself before showing her a suite.
- **Chat 6: no workflow for the runs.** Smoke runs compete for this machine's processors; run
  them from a plain script, a few at a time. A workflow may read the logs and do the cost
  analysis.
- Keep each workflow under about 10 agents running at once (the session default).

### Chat 1 status (2026-10-05). Shared code only; no experiment file was edited.

Part 1, the kNN record:
- DONE `tests/experiments/noise/similarity_space.py`: no importance weighting; distances manhattan,
  cosine, minkowski_p0.5; neighbour counts 5, 10, 20; counting schemes rank and inverse_square
  (`SWEPT_NEIGHBOUR_SCHEMES`, `neighbour_weights`, `weighted_neighbour_label`); the two groups
  (`knn_group`, `GROUP_SPACES`, `GROUP_METRICS`, `enumerate_settings(group=...)`); model_input in
  no swept list; `settings_from_env` raises on a cut weighting or distance, and on a space or
  distance the named group cut.
- NOT DONE, axis 6: a blanket drop of `exclude_groups` in `noise_cache.cached_neighbours` was
  tried and reverted. n22 passes each record's own molecule as its "group" (its other
  measurements of the same molecule, `n22:1046`), not a scaffold group, so the drop let a record
  find its own repeats at distance 0. Axis 6 has to be done where each caller passes scaffold
  groups: n07 detector grading (and `n07:616` hands groups to every neighbour estimator), n18,
  n24, n25, n42 (chat 5).
- DONE: a molecule is never its own neighbour, even when its whole row is undefined (an all-zero
  fingerprint under cosine is +inf from every molecule, its own column included).
- DONE step 2: `tests/test_f38_neighbour_screen.py` updated, not deleted. New
  `tests/test_knn_search_space.py`.
- DONE step 6: old (HEAD) and new module give identical lists on every setting both hold (manhattan
  on three spaces, continuous and tied 0/1 data; Tanimoto; forest). Checked once, not a stored test.
- ADDED, from 2.3 item 1: `hybrid__distance1` is a neighbour space (`HYBRID_SPACES`). Settings per
  group are 26 and 13, so 156 and 78 with counts and schemes; the record's 138 and 66 are these
  without it.
- NOT DONE, experiment files (chat 5): steps 3 and 4. `n07.sweep_settings` still puts model_input
  first and calls `settings_from_env()` with no group, so until it passes
  `group=ss.knn_group(ds, n_molecules)` every dataset gets full_knn's 26. `n32.sweep_settings`,
  `_sim_axis_n15_n22.sweep_settings` and every consumer listed under axis 5 still count
  neighbours their own way. Step 5 (similarity arms) is chat 4.
- ⛔ f38 reads `ss.SWEPT_KS`, `ss.WEIGHTINGS` and `ss.FEATURE_METRICS` at import. Finishing herg's
  f38 tasks with this code would sweep different settings from the three finished datasets. Pin
  f38's own copies of the old values before re-running herg.

Part 2, shared code:
- DONE 2.3 item 17, 0/1 features: `src/kirby/scaling.py` `FeatureScaler` (StandardScaler with 0/1
  features passed through; a feature is 0/1 when all its training values are 0 or 1). Replaces
  every `StandardScaler()` in `hybrid.py` (17), `find_best_hybrid.py` (7), `_shared.py` (9),
  `_alloc_harness.py`, `noise_cache.cached_scaled` (its cache key now carries the scaler),
  `similarity_helpers._standardize`, `federated_helpers` (3). `similarity/preprocessing.standardise`
  follows the same rule when it fits its own centre and spread. Experiment scripts that scale
  for themselves are not changed (their suites' chats).
- DONE 2.1 identical-features check: `fold_standard_selection(distinct_features=True)` builds each
  ranked hybrid on the fold's training rows and skips one whose selected feature numbers equal a
  chosen one's; listed under `hybrids_skipped_identical_features`.
- DONE pruning off: `search_on_fold` drops any schedule, per-model aggressiveness or
  `adaptive_pruning` passed to it, with a printed line. `run_dataset.py --pruning-mode` has no
  effect; `submit_dataset.sh`, `submit_everything.sh` and `opig_one.sh` no longer pass one; the
  `PM` markers are gone from `OPIG_JOBS`.
- DONE start-up searches: the four `run_all` (complexity, similarity, noise, federated) no longer
  search the whole dataset (`_shared.no_startup_searches`). An experiment not yet aligned that
  still reads a stored search falls through to `get_precomputed_search`, which
  `KIRBY_NO_COLD_SEARCH=1` refuses.
- DONE configs: `_shared.decided_config` sets models, budgets, importance methods, allocation
  methods, evaluators, representation counts, augmentation conditions, and the distance search's
  models and budgets from `decided_grid`. Called by all four `run_all` and by `get_config` when
  the run names its dataset. An undecided dataset raises.
- DONE `own_search_kwargs(..., dataset_name=)`: decided grid, pruning off. It raises without
  `dataset_name`, so c05, c07, c15 and c33 stop there until chat 3 passes it.
- DONE `should_skip_experiment` is now called by the complexity and federated runners.
- DONE distance hybrids in the noise neighbour spaces: `_shared.fold_neighbour_space_recipes`, per
  outer fold, gives each decided model 2 base + 2 general hybrids (8 fitted configurations for two
  models) and the recipes for `hybrid`, `hybrid__general2`, `hybrid__distance1` and
  `hybrid_similarity`, in the form `n07.fold_similarity_spaces` reads. Not yet called by any
  noise experiment (chat 5). `noise_helpers.pool_recipe_per_downstream_model` is unchanged and
  marked superseded.
- NOT DONE: `_apply_small_dataset_models` stays until c02 stops importing it (chat 3).
- 2.3 item 15, abagym: CHECKED, same fault as flab had. All three are single-mutant scans. The
  position-modulo-5 rule gives exactly 5 groups on each, so the 5 folds are the 5 groups:
  Ang2 397 variants, groups of 55 to 114; EGFR 659, 109 to 162; VEGF 418, 57 to 114. One group per
  mutated position would give 20, 36 and 21 groups. OPEN, hers to decide. If she picks position,
  the three are no longer under ten groups, which changes 5b's c05/c07 skip for them.

Found by a read-only review of this diff (three reviewers) and fixed:
- `similarity_space._CDIST` keeps its old form (a bare scipy name) for manhattan and cosine,
  because n32 passes it straight to `cdist`; minkowski_p0.5 is left out of it so n32 raises
  instead of computing p=2. Distances go through the new `feature_distance`.
- 0/1 rule also applied in `kirby/similarity/selection.py` and `recipes.py` (`_train_stats`),
  `kirby/utils/noise_mitigation_methods.py` (3 sites) and `kirby/utils/lnl_methods.py` (3 sites);
  `find_best_hybrid._evaluate_single` lost its whole-matrix 0/1 test and uses the column rule.
  `FeatureScaler.partial_fit` on a fitted scaler raises.
- `resolve_pruning_kwargs` returns no pruning, so scripts calling `find_best_hybrid` directly
  (trastuzumab, ginkgo, c10, n20) are unpruned too. n15 hardcodes aggressive pruning itself
  (chat 5).
- The identical-features check draws 4 spare hybrids per condition, so a skipped duplicate is
  replaced; the distance hybrid is the best over every condition the distance search ran.
- `get_config`: a `--models` or `--budgets` with no decided member raises; the local
  `KIRBY_SMOKE_N` shrink is re-applied after the decided grid.

Found and left, with the reason:
- 0/1 features left unscaled have variance below 1, so wherever a block is divided by the square
  root of its width (`similarity/preprocessing.stack_blocks`, federated `stack_reps_for_distance`,
  `hybrid._pairdist_vectors`) a fingerprint block now weighs less in the distance than a
  descriptor block of the same width. Hers to decide; not changed.
- Stored searches, baselines, importance and allocation scores, and noise hybrid builds are keyed
  without the scaler, so stores written before this change hold scores under the old rule.
  Only `noise_cache.cached_scaled` carries a scaler key.
- An unaligned experiment that finds a whole-dataset search on disk still reads it; only a
  missing one is refused. Removing those reads is chats 2 to 5.
- The two scikit-learn wrappers in experiment files, n41's `LocalOutlierFactor(metric=...)` with
  'minkowski_p0.5', s08 reading the general budgets, and the smoke scripts
  (`tests/smoke/smoke_noise_similarity_foundation.py`, `noise/_smoke_n22/n32/n37.py`) that name
  cut settings ('euclidean', 'lightgbm_gain') and now raise: experiment-side, chats 4 to 6.
- The similarity-objective per-fold search runs on the general grid (budgets to 2000 features)
  and is not gated on the three datasets where it is cut; errors are caught. Unverified cost.
- 2.3 item 4 says augmentation comparisons run only on ginkgo_colloidal and ginkgo_hic; the kNN
  record keeps `base_1_aug_uncapped` as a neighbour space in both groups on every dataset. The
  two disagree. Hers to settle.

Tests: `test_knn_search_space.py`, `test_f38_neighbour_screen.py`, `test_shared_alignment.py`
pass (28). The wider shared set: 112 passed; 3 failed only because ESM3 and Boltz-2 are not
installed locally.

Smoke runs (local, normal settings, `KIRBY_NO_COLD_SEARCH=1`, one thread, 20-minute box; started
before the review fixes above):

| suite | run | outcome |
|---|---|---|
| similarity | s01 on herg_ki | pass, 120 s |
| complexity | c01 on herg_ki | pass, 609 s |
| federated | fed01 on caco2 | no error at 20 min; inside its first search (dnn and svm, budgets 500-2000 features) |
| noise | n07 on herg_ki | no error at 20 min; running a 224-configuration WHOLE-DATASET search inside the experiment, because n07 is not aligned yet |
| opig | cgr_comparison on buchwald_hartwig | no error at 20 min; still in its own rf and lightgbm scoring of every representation, before any per-fold search |
| shared | `fold_neighbour_space_recipes` on herg_ki fold 0 (10 representations, 1412 molecules) | no error at 25 min; still inside the general per-fold search (224 configurations), so the identical-features check, the distance search and the similarity search were not reached. Needs a longer run or an srun |

Launchers (2.8.1, for what chat 1 changed): `opig_one.sh --list` resolves all 11 rows; the three
shell scripts parse; `submit_everything.sh --dry-run` cannot list datasets off the cluster. Still
in `submit_everything.sh`: the whole-dataset precompute chains and `KIRBY_PRUNING` for them
(chat 6 retires them).

### Chat 2 status (2026-10-05). OPIG re-aligned, federated aligned. Uncommitted.

Done by one editing agent per experiment, a second agent per experiment trying to refute
"compliant", and a fix pass (workflow, her go in 2.0). Every OPIG and federated file compiles and
imports. The per-experiment tables are in 2.6.

Shared code chat 2 changed (the agents were barred from it):
- `federated_helpers.py`: `fed_models_for(ds)` = the dataset's two decided models (lhasa_herg
  lightgbm + xgboost, caco2 and logd dnn + svm); `run_all` fits those, not FLuID's rf, xgb, dnn.
  fed05 fits svm + dnn (all three of its antibody silo datasets are decided svm + dnn).
  rf and xgb still come from FLuID; svm, lightgbm, gp, dnn are KIRBy's own estimators, with
  FeatureScaler in front of svm, dnn, gp (FLuID's `create_classifier('dnn')` does not exist, so
  dnn on a two-class silo raised before this).
- `federated_budgets(ds)` = the decided budgets (lhasa_herg 100-2000 features, caco2 and logd
  500-2000). Was 50-1000 on every dataset.
- `fed_outer_folds` (seed 42, the per-fold store's folds; were seeded by the model seed),
  `fed_fold_selection` (one per-fold search, 2 base + 2 hybrids per model at each budget),
  `recipe_reps` (rebuild on the recipe's own representations and count).
- A silo whose build fails, or that has under 20 rows, now gets 0 features and is recorded. It
  was given score 0.0 and 1 feature (fed01's dirty run: 55 of 180 silo builds).
- `fluid_within_kirby_across` no longer draws a random 20% split when a silo has no scaffold
  groups (flat score, recorded, as `federated_feature_selection` already did).
- lhasa_herg loads its 10 pruned representations. It loaded the 'all' preset, 33 of them.
- Augmentation is not loaded on lhasa_herg, caco2, logd (Step 8 cut it on all three).
- Each result file is stamped with the metric the experiment reports (mcc / auc on two-class
  labels). fed01, fed03 part A and fed04 were stamped rmse, lower is better.
- `compute_reg_metrics` returns no p-values. `smoke_n_folds` keeps 1 fold as 1.
- `_shared.py`: `fold_standard_selection(..., budget=)`; the per-fold search memo now keys on the
  labels too (two label vectors on the same rows shared one search before);
  `top_hybrids_across_models` raised TypeError when handed `require_full`, which
  `fold_neighbour_space_recipes` does, so every per-fold similarity hybrid was lost. Fixed.
- `opig_humanness.py` unset `KIRBY_NO_COLD_SEARCH` for itself; removed (per-fold searches are
  never refused since `fb1df54`).

Found and NOT fixed (shared, or hers):
- FLuID's quality-ensemble weighting (`predict_fluid_quality_ensemble*`, three places in
  `federated_helpers.py`) cross-validates with `KFold(shuffle=True)`, a random split. It is FLuID's
  published protocol; the no-random-split rule says otherwise. Hers.
- `federate_labels_and_train_student` (fed01 C4) always fits FLuID's random forest student,
  whatever model the run is for. Hers: keep FLuID's student, or fit the run's model.
- `federated_noise_detection`'s 'federated_hybrid' setting builds with a fixed recipe
  (performance_weighted, treeshap, rf), not the fold's.
- `create_hybrid` pads a representation's quota with zero-importance features by column order;
  fed04 cuts them itself, c31, c32 and any other direct caller still get the padding.
- `fold_standard_selection` returns None for a failed model without the error; every file reads
  it back from the memoised `fold_searches`.
- Launchers: `submit_everything.sh` scaffold_similarity row names ddg_combined, which the script
  never reads; `submit_opig_shared_vectors.sh` passes `--speed`, which the remedy script no longer
  takes; `run_smoke_gate.sh:199` passes `--reps` to ginkgo. Chat 6.

Open questions for Adelaide, one line each (detail in each file at the named constant):
- fed01 `SEARCH_LABELS='clean'`: one search per fold on the clean two-class labels, shared by
  every flip level. Alternative: a search per flip level on its noisy labels, about 18 times the
  searches. Claude's choice.
- fed01, fed04 on caco2 and logd: the search runs on the two-class labels; the decided grid was
  measured on the continuous labels.
- fed02 `APPLY_REP_COUNT_TO_SILOS`: a recipe that won on 2 of 10 representations is rebuilt on 2
  of the 5 silo representations, so "all silos" can mean 2.
- fed03 part A: the ChEMBL silo is now herg_ki's ChEMBL Ki records (it was Lhasa's own file loaded
  twice). Part A's FLuID arms stay on 2048-bit Morgan fingerprints.
- fed05: C4 cannot run (the three silos share no representation name); C3 rebuilds at each
  recipe's own budget (was 30 features or fewer); abagym_EGFR still groups by position modulo 5.
- census `COUNT_KNN_DISTANCE_HYBRID=False`: each decided model's own top distance hybrid, none for
  knn. 7 hybrids per fold; 8 with knn's.
- census and remedy on ginkgo_colloidal, ginkgo_hic: the distance search's uncapped condition uses
  paratope and structural as added features, which answer 12 keeps out of hybrids.
- trastuzumab: esmc_llr, antiberty_llr, esm2 stay in the hybrid search pool (as before), not only
  in the zero-shot arm and the single-representation table.
- remedy: chemberta is not on herg_ki's pruned list, so the tied representation is chosen as the
  pruned representation that ties the most molecules (graph_kernel, 105 molecules); its second
  partner pair is ecfp4 + mhg_gnn (usrcat is not on the list).
- humanness: onehot is gone from the single-representation table too.
- ginkgo, graphinity: whether to keep cutting a matrix over 500 features to its 200 most
  important before the held-out fit (`APPLY_BASELINE_SURVEY_FEATURE_CUT`, `CUT_WIDE_REPRESENTATIONS`;
  both now off, so every arm is scored on the features the search ranked).
- qsar, cgr_comparison, ginkgo: each still scores every representation alone on every fold
  beside the 2 + 2 (a descriptive table; costs fits, no search).
- blueprint_transfer: herg_ki's own hybrids are searched on the 7 representations logd and
  herg_ki share, not herg_ki's 10.

Time-boxed runs, local (2026-10-05, 15-minute box, one thread, `KIRBY_NO_COLD_SEARCH=1`, normal
settings). Other sessions held this machine at load 25 to 240 on 8 cores, so every box covered
less than usual. One row is one run:

| run | outcome |
|---|---|
| fed01 on caco2 | no error at 15 min; inside fold 1's per-fold search |
| fed02 on caco2 | no error at 15 min; inside fold 1's per-fold search |
| fed03 on caco2 (part B) | no error at 15 min; inside fold 1's per-fold search |
| fed03 on lhasa_herg (part A) | no error at 15 min; still building representations (mhg_gnn 477 s, mordred on course for ~1.5 h). Representations are now cached per run (fixed after this run) |
| fed04 on caco2 | FAILED twice, fixed: (1) `_memo_rows_id` missing, another session's edit, restored; (2) every fold's search refused, two-class labels on a dataset the registry holds as regression; `fed_fold_selection` now passes metric auc. Third run (20-min box): no error, inside fold 1's search |
| fed05 (lhasa_herg job) | no error at 15 min; still embedding the antibody silos. srun |
| qsar | no error at 15 min; still building representations (molformer). srun |
| blueprint_transfer | no error at 15 min; inside the search over all logd rows on the 7 shared representations |
| shared_vectors_census | no error at 15 min; still embedding the antibody datasets. srun |
| shared_vectors_remedy | stopped after 254 s on its own check: the LOCAL pdv cache for herg_ki does not match the builder (150 of 150 sampled rows). Local cache only; the server's pdv was rebuilt 2026-09-18. srun |
| cgr_comparison | no error at 15 min; still scoring the 12 representations alone (rf RMSE 14.44 on cgr_morgan, buchwald_hartwig), before the per-fold searches |
| scaffold_hybrid | no error at 15 min; flab_expression_koenig fold 0, pool = the pruned 10 plus the two scaffold-diff representations |
| scaffold_similarity | finished, 153 s |
| graphinity | no error at 15 min; embedding ddg_combined (this machine loads 429 rows and its cache holds 429). srun |
| humanness | no error at 15 min; esmc cannot be built here (no `esm` package). srun |
| trastuzumab | no error at 15 min; extracting esm2 embeddings. srun |
| ginkgo | no error at 15 min; building ginkgo_hic's structural features (ABodyBuilder2). srun |

srun blocks for the runs that cannot finish here (needs the chat 2 changes committed and pulled first; no deletes, safe to re-paste):

Terminal 1 (protein OPIG experiments):
```
srun --clusters=htc --account=stat-cadd -p interactive --cpus-per-task=16 --mem=128G --time=04:00:00 --pty /bin/bash
```
then inside:
```
cd /data/stat-ecr/scat9264/KIRBy
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
export KIRBY_NO_COLD_SEARCH=1 KIRBY_START_DELAY=0 KIRBY_FORCE_RERUN=1 KIRBY_OUTPUT_DIR=tests/results/smoke_chat2 KIRBY_EXP_LOG_DIR=logs/smoke_chat2
timeout 40m bash tests/slurm_scripts/runbook/opig_one.sh humanness
timeout 40m bash tests/slurm_scripts/runbook/opig_one.sh trastuzumab
timeout 40m bash tests/slurm_scripts/runbook/opig_one.sh graphinity
timeout 40m bash tests/slurm_scripts/runbook/opig_one.sh scaffold_hybrid
timeout 40m bash tests/slurm_scripts/runbook/opig_one.sh ginkgo
```

Terminal 2 (molecule and reaction OPIG experiments, fed03 part A, fed05):
```
srun --clusters=htc --account=stat-cadd -p interactive --cpus-per-task=16 --mem=64G --time=04:00:00 --pty /bin/bash
```
then inside:
```
cd /data/stat-ecr/scat9264/KIRBy
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
export KIRBY_NO_COLD_SEARCH=1 KIRBY_START_DELAY=0 KIRBY_FORCE_RERUN=1 KIRBY_OUTPUT_DIR=tests/results/smoke_chat2 KIRBY_EXP_LOG_DIR=logs/smoke_chat2
timeout 30m bash tests/slurm_scripts/runbook/opig_one.sh qsar
timeout 30m bash tests/slurm_scripts/runbook/opig_one.sh cgr_comparison
timeout 30m bash tests/slurm_scripts/runbook/opig_one.sh shared_vectors_census
timeout 30m bash tests/slurm_scripts/runbook/opig_one.sh shared_vectors_remedy
timeout 30m python -u tests/experiments/run_dataset.py federated lhasa_herg --experiments fed03 --output tests/results/smoke_chat2/federated --force-rerun 2>&1 | tee logs/smoke_chat2/fed03_lhasa_herg.log
timeout 30m python -u tests/experiments/run_dataset.py federated lhasa_herg --experiments fed05 --output tests/results/smoke_chat2/federated --force-rerun 2>&1 | tee logs/smoke_chat2/fed05.log
```

Cache checks (either terminal, after setup.sh; read-only):
```
python -c "import numpy as np,glob,os; [print(f, np.load(f).shape, 'all zero' if not np.abs(np.load(f)).max() else 'ok') for f in glob.glob(os.environ['KIRBY_CACHE_DIR']+'/rep_cache/*/chytorch_graphormer.npy')]"
python tests/experiments/opig_shared_vectors_census.py --datasets herg_ki --reps chembert --with-hybrids 0 --check-freshness 200 --output-dir tests/results/smoke_chat2/census_chembert
```

Server checks (srun only): chytorch_graphormer cache not all zeros (cgr); ddg_combined caches hold
534 rows, not 429 (graphinity, fed05); chembert on herg_ki against its builder (census, remedy);
saprot cache for the ginkgo properties; grover builds for chembl_qsar.

### Chat 3 status (2026-10-05). Complexity aligned. Uncommitted.

Same workflow as chat 2: one editing agent per experiment (23), one refuting agent per
experiment, a fix pass. Every complexity file compiles and imports. Per-experiment states are in
the complexity table in 2.6; time-boxed runs are under "Chat 3 runs" below.

Shared code chat 3 changed:
- `complexity_helpers.run_all`: every result file it saves carries
  `complexity_alignment = 'per_fold_2026-10-05'`, and the resume check recomputes a file without
  it. Without this, a file written before alignment on the same data has the same
  `input_fingerprint` and was kept as "already done" (16 of the 23 agents reported this).
- `_shared.get_model_family_variants`: c06's regression SVM family is `SVRScaledTarget` (labels
  standardised inside each fit). Dirty-run bug 4 closed.
- `_shared._apply_small_dataset_models` deleted (c02 no longer imports it). ⛔ That deletion also
  cut `run_nested_cv`, `precompute_nested_cv_searches`, `_FOLD_SEARCH_MEMO` and `_memo_rows_id`
  out of `_shared.py` from 18:41 to about 19:10. The similarity chat restored the last two; chat 3
  restored the first two from a copy read before the deletion. Any run started in that window
  that hit a NameError must be run again.
- `run_dataset.py --c11-recipe-search` accepts only `per_fold`; `whole_data` is gone (c11 raises
  on it).

Requested by the agents and NOT done (shared or outside the suite):
- `tests/analysis/firstpass_notready.py` readers for c05, c06, c07, c17, c18, c19, c22, c24 read
  the old result layouts. They print nothing or None on aligned files until updated.
- `cross_one.sh` header still says c03 reads stored searches and fits SVR and ElasticNet.
- c09 and c33 build each chosen hybrid twice (once in the identical-features check, once to
  score). `fold_standard_selection` could hand back the features the check built.
- c15 and c33 call the private `_shared._distinct_feature_hybrids` (n09 and fed05 already do).
- c14 records a failed per-fold search and still exits 0 with an empty table (seen in the first
  run, during the NameError window). The OPIG pattern raises instead. Chat 6 to decide.

Chat 3 runs (local, normal settings, `KIRBY_NO_COLD_SEARCH=1`, one thread each, 3 at a time,
20-minute box; other chats were running their own jobs on the same machine). The logs were in
/private/tmp and were lost when the machine crashed afterwards; outcomes as read before the crash.
- c01 on herg_ki: finished, 1,151 s.
- No other run finished inside 20 minutes, and none raised. Each reached its new per-fold code,
  except where noted: c02, c03 (herg_ki + caco2), c05, c07, c09, c11, c14, c16, c18, c19, c22
  (buchwald_hartwig), c23, c33 were inside fold 0's search (c33: 896 configurations per fold, its
  own wider budget list); c06 had printed held-out scores for fold 0; c04 was computing importance
  for 5 folds x 2 methods; c08, c15 had printed their fold and model lines; c24 (buchwald_hartwig)
  was finding recipes on its 5 folds.
- Did not reach the new code: c12 (tcga_luad_vs_lusc) was still building its representations
  locally (Geneformer); c17 (suzuki) and c21 (ginkgo_hic) were still loading. These need a run on
  the cluster.
- c13: three candidate embedding models failed to load locally (CheMeleon and GLACIER need a newer
  chemprop, Mol-JEPA needs molfeat); tracker C13d already records unloadable candidates.
- A first attempt, inside the 18:41 deletion window, failed c02 and c05 on the missing
  `_memo_rows_id`, and c14 saved an empty table with exit code 0 (see above).
- ⛔ The machine crashed shortly after these runs ended, during a disk-full error in /private/tmp.
  Cause not established. Suspect: memory pressure from many Python jobs at once (these 3 plus
  other chats' runs) pushing swap onto a full disk, and possibly child processes of the timed-out
  runs that were not killed (the first driver's process-group kill raised PermissionError).
  Next time: fewer runs at once, and check that no child process survives a time box.

Open questions for Adelaide, one line each. The full wording, with the numbers, is in each
file's docstring or next to the named setting.
- c01, c21: no training-row score per fold, because neither picks anything. Adding one is an
  inner cross-validation per fold, about 5 times the fits.
- c02: the extra search with fine-tuned representations (13 datasets) is no longer run; they are
  in neither representation list.
- c02: the search at a single representation's own width (e.g. 1,280 features on flab) is no
  longer run; that width is never a decided budget.
- c02: the budget-by-model heatmap is built from training-row scores; only the 2 chosen hybrids
  per model are scored held out.
- c03: one budget per dataset for the correlation is now the median over folds of each fold's
  winning budget (in features). Alternative: the most common one.
- c03: the SVR and ElasticNet fits are removed (neither is a decided model).
- c03: the winning budget now comes from the general search (decided budgets reach 2000
  features), not the distance search (10 to 500).
- c03: the budget variance table needs a model decided on every dataset in the run; across all
  datasets that may be none, and the table is then not computed.
- c04: has no hybrid arm. Adding the distance-search hybrid and the similarity hybrid as two more
  neighbour spaces is possible (2.3 item 1 named c04 as a candidate).
- c04: its 10-neighbour arms weight neighbours by 1/distance; the kNN record keeps rank and
  inverse_square only. Its subject is importance weighting, which the record cut for kNN.
- c04: the 25-feature subsets are below the decided budgets on most datasets (250 or 500 on
  herg_ki, logd, caco2).
- c04: importance methods now follow the decided list, so the select-by-one, weight-by-another
  table is empty on logd and qm9 (one method each).
- c04: reports R-squared and MAE of the neighbour prediction, not the dataset's own metric.
- c05: the score of a recipe chosen with every label, re-built at each sample size, is gone. A
  per-fold version costs one build and fit per smaller draw, fold and model.
- c05: the 3-number learning curve is fitted only from 4 sample sizes up (5 below 5,000 molecules,
  4 above).
- c05: one search per draw per fold now: 65 at 5 folds below 5,000 molecules (was 13).
- c06: only families of the dataset's two models run. 5 of 26 regression datasets get one family
  (abcov_kd, buchwald_hartwig, ginkgo_acsins, qm9, suzuki); kNN, Lasso never run; there is no
  lightgbm family.
- c06: the budgets added below the decided range are gone; 14 datasets have 3 decided budgets.
- c06: each setting's best budget is picked on training-row scores; held-out scores are stored.
- c07: the all-rows arm now searches the capped rows (about 2,000) minus the half's held-out
  rows. It differs from the old arm in two ways at once: no held-out rows, no rows above the cap.
- c07: 80 searches per dataset at 5 folds (4 partitions x 2 halves x 5 folds x 2). A shared fold
  layout needs 45.
- c07: easy and hard halves have separate folds, compared fold number by fold number.
- c07: a second hybrid identical to the first leaves its slot empty; no next distinct one is drawn.
- c08: the 'mismatched' condition (ridge on the overfit-width hybrid) is not run. Alternative: fit
  the dataset's other decided model there.
- c08: the one cross-model best recipe is gone; each model's own hybrid1 is reported.
- c09: single representations now enter the curves as reference points (2 per model per fold).
- c09: 9b's 'variance retained' curve uses a ridge fit as a measuring tool, not as a model. Keep it?
- c09: the top-level 9b/9c curve takes, per fold and budget, the hybrid of whichever model scores
  best on training rows.
- c09: 9d still computes its own training scores and ignores c06's.
- c11: keeps 2 hybrids at every decided budget beside the standard 2 (up to 100 builds and fits on
  herg_ki against 20).
- c11: the ensemble's own inner cross-validation is 3 splits whatever the outer fold count.
- c12: on fold f, each modality's representation is picked on the other folds' held-out scores,
  whose models were trained on rows that include fold f. Kept (decided 2026-09-12).
- c12: f1_weighted is taken from the highest predicted probability, not the model's own call.
- c13: the random-feature control still gets 0 features from the budget split, so
  D - D_random equals D - C. The features it got are now stored per fold.
- c13: condition A (embedding alone under one model) is removed; with both models scored it
  duplicated B.
- c13: the candidate embedding models are outside `DATASET_REPS`; kept, since they are c13's
  subject.
- c13: above 5,000 rows it still draws 2,000 rows at random, ignoring groups.
- c14: no cross-dataset averages and no Wilcoxon test now; counts only. Two rows per dataset (one
  per model).
- c14: its distance hybrid is dropped (not in 2.3 item 1).
- c15: random features and PCA take the full budget; the hybrid stops at its last non-zero
  feature, so the arms differ in width as well as method. Widths are stored.
- c15: 15d's permuted-label test now uses the decided models; the sqrt(d/n) line beside it is a
  result for linear models.
- c15: 15a/15b statistics are measured on each fold's held-out rows (about a fifth of the rows).
- c15: 15b's length-scale sweep reads the smallest decided budget (250 on herg_ki), not 50.
- c15: no distance hybrid in 15a's neighbour statistics (not in 2.3 item 1).
- c15: 15c budgets are multiples of all rows, so features = training rows falls at about 1.25.
- c16: ridge and logistic-regression families removed (ridge is removed everywhere). This also
  drops tracker items C16b and C16c, which were delegated "keep".
- c16: 17 datasets get no family (c16 sweeps only rf depth and svm C).
- c16: heatmap rows are the decided budgets only (herg_ki loses 10, 25, 50, 100).
- c16: the 'full' column is still a hybrid at min(all features, 2 x rows, 2048), not all features.
- c17: mRMR has no cell on 20 of 27 regression datasets (cap 200 features, decided budgets start
  at 250 or 500).
- c17: the hybrid can draw from fewer representations and build fewer features than the other
  four methods.
- c18: the uniform arm and the natural arm differ in three ways at once on buchwald_hartwig: how
  the budget is split, how many representations take part (5 of 6), how many features are built
  (233 of 250).
- c18: when a fold's search gives no hybrid at a budget, the hand-built arms use lightgbm_gain if
  decided, else the first decided method.
- c18, c22: held-out features are now capped at the training range (both now fit svm and dnn).
- c18: the all-rows reference ranking (`KIRBY_C18_FULL_DATA_RANKING`) is still computed; no arm
  reads it.
- c19: 100 features is a decided budget on 16 of 43 datasets; elsewhere c19 takes the best
  hybrid over the decided budgets.
- c19: single representations are weighted with the importance method of the same model's
  rank-1 hybrid.
- c21: the with-and-without-extra-features comparison runs only on ginkgo_colloidal and
  ginkgo_hic; condition C (extras from other general representations) is skipped elsewhere too.
- c21: the fixed H0/H1 recipe was moved onto the decided grid by the agent: wglobal_power2 where
  decided, else performance_weighted; deeplift where lightgbm_gain is cut. Confirm or change.
- c21: H budgets 500, 1000, 2000 where decided; 12 datasets then have no 1000-feature headline row.
- c22: SHAP on svm, dnn and gp goes through KernelExplainer: measured 210 s per explained
  molecule for svm at 250 features, about 175 hours per model per dataset at 5 folds.
  Needs her decision before it runs on any of the 32 datasets with svm, dnn or gp.
- c23: gp has no evaluator list, so on 5 datasets c23 runs one model.
- c23: condition E is still the old {rf, lightgbm, xgboost} evaluator set, not the decided one.
- c24: runs once per decided model, so two transfer tables per dataset.
- c24: an off-diagonal cell fits its model on rows outside both folds (3/5 of the molecules
  against 4/5 on the diagonal), so it differs from the diagonal in two ways at once.
- c24: no code fault found for buchwald_hartwig fold 0's 2-rmse gain.
- c33: the distance arm is removed (not in 2.3 item 1).
- c33: its search runs budgets outside the decided range (10-100, 4,000, 8,000, and multiples of
  the training rows). R3 says decided budgets only; c33's question is the budget.
- c33: the minimum-norm linear model is gone, so the double-descent column has no model with one
  weight per feature.
- c33: fold count now from `config['n_splits']` (she set 3 on 2026-09-29).
- c33: the standard-selection hybrids and the per-budget hybrids are built two ways (with and
  without the fold search's representation scores).

### Chat 4 status (2026-10-05). Similarity aligned. Uncommitted.

Same workflow as chats 2 and 3: one editing agent per experiment (12), one refuting agent per
experiment, a fix pass; s09 was edited last, against the other eleven's new result files. Every
similarity file compiles and imports. Per-experiment states are in the similarity table in 2.6;
time-boxed runs are under "Chat 4 runs" below.

Shared code chat 4 wrote, all in `tests/experiments/similarity/similarity_helpers.py` (section
"PER-FOLD SELECTION AND kNN SETTINGS"), so the twelve files cannot drift apart:
- `sim_outer_folds(data, config)`: outer folds from `nested_cv_outer_folds`, count from
  `config['n_splits']`, seed 42 (the seed every per-fold search is filed under, so the noise suite
  and this suite share one search per fold through the result store).
- `fold_similarity_selection(..., neighbour_spaces=)`: per outer fold, on its training rows: each
  decided model's 2 base representations and 2 general hybrids (`fold_standard_selection`). With
  `neighbour_spaces=True` also: knn's 2 best single representations from the per-fold distance
  search, knn's top hybrid of the distance search, and the top hybrid of the similarity search.
  A fault in the code (NameError, TypeError, AttributeError, ImportError) is raised; a search
  that fails for another reason is recorded under `unavailable` with its reason.
- `build_fold_entry`: rebuilds a chosen hybrid on the fold's training rows with that fold
  search's representation scores (dirty-run bug 1).
- `knn_axes(dataset, n_molecules)`: the kNN record's settings, read from `similarity_space.py`.
  full_knn: manhattan, cosine, minkowski p=0.5. fast_knn: manhattan, cosine. 5, 10 and 20
  neighbours. rank and inverse_square. Scaffold group never excluded. No similarity experiment
  reads the process-wide `KIRBY_DISTANCE_METRIC` any more.
- `knn_lists`, `knn_from_lists`, `knn_sweep`, `score_prediction`: one neighbour list per distance
  serves every count and scheme; a molecule's own row is excluded by index (dirty-run bug 7); AUC
  is read off the class shares, not 0/1 calls. A molecule whose distance is undefined to every
  training molecule (an all-zero row under cosine) counts its k listed neighbours equally.
- `scale_fold`: FeatureScaler fitted on training rows (0/1 features unscaled); held-out values cut
  to the training range.
- `neighbour_arm_reps` (Part 1's helper, never wired in) now raises: it ranked on a whole-dataset
  search.
- `run_all` no longer passes s02 and s08 the old budget lists, s05 its neighbour counts, or s03
  rf, lightgbm and dnn. `SIMILARITY_EXTRA` lost `k_values` and `knn_conditions`.
- `_summary_keys.SELECTION_RULE` is now `fold_similarity_selection_2026_10_05`; s09 refuses an
  input file not written per fold.

Outside the suite: chat 4 re-added `_FOLD_SEARCH_MEMO` and `_memo_rows_id` to `_shared.py` after
the 18:41 deletion recorded under chat 3.

Requested by the agents and NOT done (outside the suite; chat 6 or hers):
- Readers of the old similarity result layouts: `tests/analysis/firstpass_notready.py` (s01, s02,
  s03, s13), `inspect_results.py`, `quick_hybrid_summary.py`, `make_supervisor_plots.py`,
  `opig_slide_figures.py`, and `tests/similarity_experiments.py` (its own cross-dataset summary).
  They find nothing in aligned files until updated.
- Smoke checks that read removed names: `tests/smoke/smoke_similarity_hybrid_spaces.py` (s02, s10,
  s13 source text) and `tests/experiments/smoke_tests_broken_fixes.py:231` (s10's
  `_orient_deltas`, which was the double sign flip).
- `tests/slurm_scripts/run_similarity_experiments.sh`, full_s09 branch: calls `run_dataset.py`,
  which refuses s09. `runbook/cross_one.sh` is the path that works.
- `_shared.estimate_intrinsic_dim_mle` (s08's intrinsic dimension) still drops the first
  neighbour column and reads the process-wide distance.
- `similarity_helpers.cliff_structural_similarity`, fallback branch (no SMILES): still uses the
  process-wide distance. s12 no longer reaches it; s04 does when ECFP4 cannot be built.
- No result file stores which molecules each fold held, so s09 checks that its inputs used the
  same folds by row counts and label and group hashes only.

Open questions for Adelaide, one line each. The full wording, with the numbers, is in each
file's docstring or next to the named setting.
- s01: still scores all 10 representations (group A in 2.3). Part 1 step 5 lists s01 among the
  loops to cut to knn's 2. The two instructions disagree.
- s01, s02, s08: the share of a molecule's neighbours with a very different activity is now
  weighted by rank or inverse_square (uniform was cut). There is no unweighted share any more.
- s01: similarity = 1/(1+distance) under cosine runs on a different scale from manhattan, so SALI
  is stored per distance and never compared across distances.
- s02: the concatenation-of-all-representations arm is removed ("baselines are individual
  representations, not concat").
- s03: its "mutual information" is renamed `summed_component_mi` with the label entropy stored
  beside it. It is still a sum over components, not a joint estimate.
- s03: keeps the 2 best hybrids at each decided budget per model per fold (up to 10 per model on
  herg_ki, against 2).
- s04, s12: which molecules are cliffs is still decided once on all rows, held-out labels
  included (unchanged).
- s04: the single kNN number s09 reads per fold uses the setting with the best mean on the other
  folds. With one outer fold there is no such number.
- s04: rf is gone; each decided model is fitted on its 2 base representations and 2 hybrids.
- s05: no budget axis; each hybrid is built at the budget its per-fold search chose. Sweeping the
  3 distance budgets would multiply hybrid builds and kNN sweeps by 3.
- s05: the fixed comparison s09 reads is the distance hybrid against knn's best single
  representation, both unweighted, manhattan, 5 neighbours, rank.
- s06: errors come from each decided model on its own best general hybrid and on all general
  representations together, not from rf.
- s06: the conformal interval is the plain absolute residual, so its width is the same for every
  molecule.
- s08: prediction sweeps the decided general budgets (herg_ki 250 to 2,000 features), distance the
  decided distance budgets (100 to 500), so the two optima can only meet at 250 or 500.
- s10: the Gaussian process and the RBF SVR stay; neither is a decided model; they are s10's
  question.
- s10: q-RASAR is fitted with the two decided models (was rf) at every kNN setting. Estimated
  about 1,850 model fits per fold on herg_ki, against about 400 before. Not measured.
- s11: each hybrid is built at the budget its per-seed search chose (herg_ki 250 to 2,000
  features), while pca_50d and random_50d stay at 50 features.
- s11: the starting set is whole groups until it holds 5% of rows and at least 2 groups (9% on
  buchwald_hartwig).
- s11: reaction-yield datasets now hunt the highest yield; adme_hlm, adme_rlm, logd and qm9 still
  hunt the low end of their label.
- s12: scores 2 base representations per model, not every representation. s12 is not in 2.5.
- s13: a neighbour's label gap is weighted by rank or inverse_square; there is no equal-weight
  cost any more.
- s09: hybrid minus the rank-1 representation of the same selection, per fold, in the input's own
  quantity. The cliff reduction in per cent is replaced by the raw difference.
- s09: reads s02, s04, s05, s06 and s08 only, as before.

Chat 4 runs: local, normal settings, one thread per process, `KIRBY_NO_COLD_SEARCH=1`, four
at a time, output to the session scratchpad. No experiment raised. None got past fold 0's
per-fold searches except s01 (which runs no search) and s11 (which searches per seed, not per
fold). One row is one run.

| experiment | dataset | box | where it was when the box ended |
|---|---|---|---|
| s01 | caco2 | 20 min | fold 2 of 5. Slow: 10 representations x 3 distances per fold; minkowski p=0.5 is the costly distance |
| s02, s03, s04, s05, s06, s08, s10, s12, s13 | caco2 | 20 min each | inside fold 0's per-fold searches |
| s11 | caco2 | 20 min | third starting set (seed) of its first condition, ecfp4_raw; no hybrid condition reached |
| s02, s04, s05, s13 | herg_ki | 40 min each | inside fold 0's per-fold searches; 20 single-representation scores read from the store, no hybrid configuration looked up yet. Why no hybrid lookup in 40 minutes is not established |
| s09 | none | | not run: it needs the other files' outputs |

Why nothing got past fold 0 locally: caco2's fold-0 general search alone (252 configurations,
dnn and svm, 1,726 training molecules) scored 145 configurations in about 3 hours on this
machine. So the code after the searches (hybrid rebuilds, kNN sweeps, result writing) has only
been run on synthetic data by the editing agents, not on a real dataset. It needs an srun.

Two faults the runs found, both fixed: `fold_searches` lost its memo (18:41, see chat 3); and
s02, s04, s05 wrote a result file and exited 0 when every per-fold search had raised that
NameError, because the per-fold handlers caught every exception. s02, s05, s08, s10, s13 and
`fold_similarity_selection` now raise on a code fault.

abcov_kd and humanness_therapeutic could not be used: both build protein embeddings locally
(ProstT5, more than 30 minutes) before any experiment starts.

Cluster run, 2026-10-05 to 06, Adelaide's srun (16 processors, 64 GB), herg_ki, commit `5566961`,
`KIRBY_NO_COLD_SEARCH=1`, one thread per process, all 11 at once, 60-minute box each. No log holds
a traceback, "FAILED" or "Error:". One row is one experiment.

| experiment | where it was at the end |
|---|---|
| s01 | finished, all 5 folds, 1,519 s |
| s05 | past fold 0's searches: rebuilt dnn's first hybrid (1,799 features) and scored kNN on it (0.6176 rmse, minkowski p=0.5, 5 neighbours, rank) |
| s03 | past fold 0's searches: rebuilding svm's fold-0 hybrids |
| s11 | building each starting set's hybrids (250 and 163 features) |
| s02, s04, s06, s08, s10, s12, s13 | inside fold 0's per-fold searches; no hybrid configuration looked up yet |

Not yet run to the end on real data: everything after the searches in s02, s04, s06, s08, s10,
s12, s13, and s09.

### Chat 5 status (2026-10-05). Noise aligned. Uncommitted.

The fold rule from Adelaide's separate chat was not pasted into this chat. So the fold count is
one setting, and nothing assumes a value. Read from `config['n_splits']` (5 today).
`KIRBY_NOISE_N_FOLDS=1` gives one group-aware 80/20 split. `KIRBY_NOISE_ONLY_FOLDS=0` runs fold 0
of the 5-fold split only. Its per-fold searches are then reused when the other folds run.

Same workflow as chats 2 to 4: one editing agent per experiment (37), one refuting agent per
experiment, one fix pass where the refuter confirmed faults (25 of 37). Every noise file compiles
and imports. Per-experiment states are in the noise table in 2.6.

Shared code chat 5 wrote. Agents were not allowed to edit it.
- `noise_helpers.py`, block "CHAT 5 ALIGNMENT":
  - `noise_outer_folds`, `noise_n_folds`, `fold_spread` (one fold gives "one fold, no spread").
  - `fold_configurations` (wraps `_shared.fold_neighbour_space_recipes`) and `fold_model_inputs`:
    per fold, per decided model, 2 base representations and 2 hybrids, so 8 fitted configurations
    for two models. The distance hybrid and the similarity hybrid are neighbour spaces only.
    `configuration_record` writes the selection into the result file, base representations' scores
    in the per-fold search included.
  - The noise grid: `DECIDED_REGRESSION_CONDITIONS` (the six kinds), `decided_levels`,
    `inject_condition` (one entry point for the three injectors), `dataset_knn_group`.
    `conditions_for_task` and `dataset_policy.noise_levels` now return the decided ones.
    Classification keeps its six flip kinds and six flip levels; f37 measured regression only.
  - `detection_is_defined` counts the flip kinds; n21, n24, n25 are limited to the decided kinds
    where some molecules are left as delivered, plus the flip kinds.
- `n07_remediation.py` (shared parts):
  - `sweep_settings(dataset_name, n_molecules, has_smiles)` gives the group's settings, without
    the model's own features as a neighbour space. `_sim_axis_n15_n22.sweep_settings` is the same.
  - `screen_settings`: every (setting, neighbour count, counting scheme) is scored with no model
    fit; only the best 3 are refitted (KNN record, "Screening"). `similarity_sweep` and n07's own
    pass call it. `KIRBY_NOISE_SCREEN_TOP` overrides 3; 0 refits all.
  - `cleaned_training_set(similarity_scheme=)`: label_smooth, knn_smooth and prox_smooth weight
    their neighbours by rank or one over distance squared when the lists carry a scheme.
  - Scaffold-mates stay in every neighbour list (axis 6), detectors included.
  - `row_models` no longer adds rf and svm to every cleaning row (R1).
  - `_scale_pair` and `_sim_axis_n15_n22._scale` use `FeatureScaler`, so 0/1 features inside a
    hybrid stay unscaled.
- `_shared.fold_neighbour_space_recipes`: takes `condition_suffix` (a row set other than the
  dataset as loaded gets its own stored searches) and passes `base_detail` on.
- `run_dataset.py --noise-condition` accepts shift_only_p10 and sparse_dprime_up_p10.
- New test `tests/test_noise_alignment.py` (4 tests, synthetic data). With
  `test_knn_search_space.py` and `test_f38_neighbour_screen.py`: 25 pass.

**Decisions only Adelaide can make, from this chat.** Each is marked `# OPEN (chat 5):` in the code.
The current code does the first option unless said otherwise.

1. Which molecules the screen predicts. Now: each training molecule from its neighbours among the
   other training molecules, scored against the damaged training labels. The other choice is how
   the record measured it: held-out molecules scored against their clean labels. But those are the
   molecules every noise experiment reports its score on, so the settings would be chosen on the
   numbers they are judged by. One fault with the current choice: the forest-neighbour setting
   always wins on training rows, because the forest was fitted on the same labels it then
   predicts. Seen on synthetic data: the top 3 kept were all forest-neighbour settings.
2. Which 2 levels the fast_knn datasets get. The record says 2 and does not say which. Now: 0 and
   1.0; censoring 0 and 0.25 of labels clipped.
3. Which labels the per-fold searches see. Now: the clean training labels, as the old precompute
   did. n15 and n20 search on damaged or cleaned labels on purpose. n09 searches its dnn on the
   damaged labels (its old behaviour). n17, n19, n26, n29, n36 search on each molecule's mean over
   its measurements.
4. The cleanups' default neighbours. n07, n17, n18, n20, n24 and n25 report a main table from
   label_smooth, knn_smooth and neighbor_residual run at f36's setting: 9 or 10 neighbours, found by
   Euclidean distance in the configuration's own features, each counting equally. The kNN record
   cut all three choices. The screened settings reach these methods only in the rows file. Option
   A: leave it (now). Option B: build the main tables from each fold's top screened setting. This
   changes every number in those tables.
5. The counting scheme does not reach the detectors. neighbor_residual, local_variance_ratio,
   distance_weighted and activity_cliff_aware (`src/kirby/utils/noise_estimation.py`) and lnl's
   distance selection average their neighbours their own way. So rank and inverse_square give the
   same detector scores. Option B adds a scheme to those four estimators. That changes every
   detector number in n06, n07, n11, n17, n22 and n24.
6. Importance-weighted distance where it is the experiment's own question: n02 (conditions 2 and
   4), n12 (the whole experiment), n21 (8 weighted distances per base representation, and
   euclidean). The kNN record cut weighting from the shared neighbour settings. Kept in all three.
   Also: whether DECISIONS.md Step 3's cut of importance methods applies to these weightings (n12,
   n16, n21 keep both lightgbm_gain and treeshap everywhere).
7. Cut damage kinds that an experiment's question needs. n23: grouped_wider was its only kind tied
   to structure that damages a subset; recommend adding grouped_shifted. n25: outlier_only_p05 (its
   old default) cannot run now. n31: its formula needs outlier_only_p05 or p10; still run.
   n33: laplace and student_t (heavy tails). n34: grouped_wider as its no-offset control; still
   run. n38: five of its nine conditions were cut; all nine still run, at two level grids.
   n40: dense_oofmse, which f37 never measured.
8. Censoring draws are identical. Censoring clips at a quantile and reads no seed, so its 3 draws
   give bit-identical labels (checked at 0.25 of labels clipped, 400 and 500 labels). n30 runs 1
   draw; n32 and n41 still run 3.
9. herg_ic50. Its 9,802 molecules make it fast_knn by the count rule; the record's list of eleven
   does not name it. At 12,008 measurement rows svm is skipped (row limit 10,000), so n18 and n41
   fit dnn only there. A molecule's own repeats count as activity-cliff pairs in n01, n14, n25.
   It is not in the submit roster, so n26 and n29 run on it only by hand. n19 and n26 build the
   same per-fold searches under two names.
10. Fits inside a method that use a model outside the two decided ones, mostly an out-of-fold
    LightGBM: n14, n27, n29, n34, n36, n38, n39, and n40's GP, kNN and kriging arms. Kept. Does R1
    cover them?
11. gp takes no per-row weight. On the five datasets where gp is decided (abagym_Ang2,
    abagym_VEGF, ginkgo_acsins, ginkgo_hic, ginkgo_pr_cho), the weighting arms of n02, n12, n21,
    n31 and n33 record "not run".
12. knn_smooth and label_smooth give the same labels on classification datasets (dirty bug 13).
    Both still run in n07 and n24.
13. The base-plus-added-features neighbour spaces run on every dataset (kNN record). 2.3 item 4
    says augmentation runs on ginkgo_colloidal and ginkgo_hic only. Chat 1 left this open too.
14. Per experiment. c10: keep its "does a search on damaged labels move the recipe" part (27 more
    per-fold searches per fold on full_knn, 9 on fast_knn)? The one-hybrid-per-budget part is gone under the
    8-configuration rule. n06 scores the distance and similarity hybrids as detector spaces; n22
    does not. n09: keep the importance-weighted copy of hybrid 1 as a fifth space? n13: its
    network training (Track A) does not run on the 16 datasets without dnn; use n09's own dnn
    search? n37 is refused on kiba and bindingdb_kd (12,000 training rows per fold).
    The full list, with costs, is in each file at its `# OPEN (chat 5):` comment.

Not done:
- f37 measured regression only, so dirty-run bug "n13 classification grid not cut" stands.
- About 20 old `_smoke_nXX.py`, `_loop_*` scripts and three analysers
  (`tests/analysis/firstpass_notready.py`, `inspect_results.py`, `noise_ceiling_analyse.py`) read
  the old result layout or patch removed names. They fail until rewritten (chat 6).
- `similarity_space.neighbour_lists` computes distances in float64 internally. Inputs and stored
  matrices are float32.
- Nothing committed.

Chat 5 runs (local, 2026-10-05; normal settings, `KIRBY_NOISE_ONLY_FOLDS=0`, one thread,
`KIRBY_NO_COLD_SEARCH=1`, 10-minute box each; the machine was at load 170 from other sessions'
runs):
- herg_ki fold 0's three per-fold searches (general, distance, similarity), built once beforehand:
  11,587 s (3.2 hours) on one thread. Both models got 2 base representations and 2 hybrids with
  different features; svm's base representations are mhg_gnn and ecfp4, dnn's mhg_gnn and grover.
  The distance hybrid was found. The similarity hybrid was not, because of the `require_full`
  fault chat 2 has since fixed. Not re-checked after the fix.
- 31 experiments on herg_ki (every noise experiment except the six below): none failed. All 31
  loaded, split fold 0 (1,129 training, 283 held-out molecules), printed their plan, and were
  inside fold 0's per-fold search at 10 minutes. So none reached its own fitting code. A run that
  reads the stored searches still spent its 10 minutes there; why is not known.
- n17, n19, n22, n26, n29, n36 on herg_ic50: `run_dataset.py` refuses herg_ic50 (not in its
  dataset list; item 9 above), exit 1 in 10 s. Run directly through `noise_helpers.run_all`: none
  failed; all six were in fold 0's search at 10 minutes. n17 and n22 used 12,008 measurement rows,
  n19, n26 and n36 9,802 molecule rows, n29 7,841 training molecules with their 9,685 measurements.
- So every noise experiment starts, and none has been seen to run past its first per-fold search.
  The next check has to be an srun where the searches finish (block below), or a local run of
  one experiment after the per-fold searches are stored.
- ⛔ The computer crashed at about 22:00 with the local disk full ("no space left on device").
  At that point this chat was running 6 herg_ic50 experiments and 1 re-read of the herg_ki searches
  at once, beside other sessions' runs. Running out of memory, so that swap filled the disk, is the
  likely cause. Not verified. Run at most 2 noise experiments at a time on this Mac.

To run the noise suite on the cluster (not yet tried there). One terminal per dataset; safe to
re-paste, because a finished condition file is skipped and a part-way one resumes. No deletes.
```
srun --clusters=htc --account=stat-cadd -p interactive --cpus-per-task=16 --mem=64G --time=04:00:00 --pty /bin/bash
```
then inside:
```
cd /data/stat-ecr/scat9264/KIRBy
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
export KIRBY_NO_COLD_SEARCH=1 KIRBY_NOISE_ONLY_FOLDS=0
python tests/experiments/run_dataset.py noise herg_ki --experiments n07
```
`KIRBY_NOISE_ONLY_FOLDS=0` runs fold 0 only (her one-fold idea, 2.12); leave it out for all
folds. `--experiments` takes several ids; leave it out to run every noise experiment in order.

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

OPIG: RE-ALIGNED by chat 2 (2026-10-05, uncommitted, all compile). Time-boxed runs: see "Chat 2
status" below.

| experiment | audit lines | state after chat 2 |
|---|---|---|
| shared_vectors_census | 744-885 | per fold: 2 general hybrids per decided model, each model's top distance hybrid, 1 similarity hybrid (7 per fold); folds from config; failures saved with dataset; "broken cache" only when a representation gives every input one row |
| scaffold_similarity | 886-969 | no model, no search (group A); small fixes only |
| humanness | 970-1110 | 2 base + 2 general per model, distance hybrid dropped; onehot dropped; per fold saved; failures saved; FeatureScaler, float32 |
| graphinity | 1111-1223 | 2+2 per model, no distance; aac removed; refuses a matrix whose row count differs from the dataset |
| trastuzumab | 1224-1364 | 2+2 per model, no distance; keeps esmc_llr, antiberty_llr, esm2 (in the search pool, as before); per-fold AUROC table hybrid vs best single (picked on training rows); zero-shot sign checked on training rows |
| cgr_comparison | 1365-1520 | 9 representations (pruned 10 minus cgr_morgan) against 12 (pruned 10 plus chytorch_graphormer, cgr_smiles_lm); 2+2 per model per group; refuses an all-zero or constant matrix |
| ginkgo | 1521-1637 | rewritten: 2+2 per model per competition fold (namespace `ginkgocv{n}f{i}`); no distance, no augmentation; paratope and structural scored alone only |
| qsar | 1638-1766 | chembl_qsar decided (lightgbm, dnn); the pruned 10; 2+2, no distance; per (target, fold) saved |
| blueprint_transfer | 1767-1922 | 2+2 per fold on the 7 representations logd and herg_ki share; transferred recipes from one logd search on the decided grid, unpruned |
| scaffold_hybrid | 1923-2061 | distance tier dropped; 2+2 plus the two scaffold-diff representations scored alone |
| shared_vectors_remedy | 2062-2210 | arms B and D only on ginkgo_colloidal and ginkgo_hic (skipped on herg_ki with a reason); no distance hybrid on C; target representation from the pruned list (chemberta is not on it) |

Federated: ALIGNED by chat 2 (2026-10-05, uncommitted, all compile).

Complexity: ALIGNED by chat 3 (2026-10-05, uncommitted, all compile and import). One row is one
experiment; runs are under "Chat 3 runs" in 2.0.

| experiment | audit lines | state after chat 3 |
|---|---|---|
| c01 | 2874-2965 | no selection (group A); each pruned representation x the 2 decided models per fold; training-range cap now in c01 itself (eda240c had put it in a function c01 no longer calls) |
| c02 | 2966-3111 | per fold 2+2; heatmap from each fold search; augmentation arms only where decided; fine-tuned and same-width searches dropped (questions); per-fold partial file |
| c03 | 3112-3239 | per fold 2+2 per dataset, models decided on each dataset; general search, no distance; SVR and ElasticNet fits removed; Spearman p-values removed; per-fold partial file |
| c04 | 3240-3348 | no selection (group A); importance methods from the decided list; no float64; no hybrid arm (question) |
| c05 | 3349-3477 | (B) per draw, per fold 2+2 on the draw's training rows; curve fitted from 4 sizes up; whole-dataset recipe score removed |
| c06 | 3478-3592 | (B) one search per capacity setting per fold; only families of the 2 decided models; decided budgets only; svm with standardised labels; per-fold checkpoint |
| c07 | 3593-3741 | per half per fold 2+2; all-rows arm on the capped rows minus held-out rows; no `own_search_kwargs` |
| c08 | 3742-3909 | per fold 2+2; ridge "mismatched" condition not run (question); no longer reads c02 |
| c09 | 3910-4067 | per fold 2+2 at each decided budget; reads c02's training-row grid for 9a; ANOVA p-values gone, shares kept |
| c11 | 4068-4191 | per fold 2+2 plus 2 hybrids per decided budget; `whole_data` removed; onehot_components out; ensemble re-pick on held-out scores removed |
| c12 | 4192-4295 | (B) no search; decided models; modality pick on other folds kept (question) |
| c13 | 4296-4437 | per fold 2+2 per condition; condition A removed; random control unchanged (question); per-fold checkpoint |
| c14 | 4438-4594 | per fold 2+2 per dataset and model; no cross-dataset averages or Wilcoxon; distance hybrid dropped; per-fold partial file |
| c15 | 4595-4759 | (B) decided models, ridge and rf removed; 15c one search per budget per fold; "full dimension" is now all features, not PCA |
| c16 | 4760-4901 | per fold 2+2 per budget; rf and svm families at the search's settings; ridge and logistic families removed (question) |
| c17 | 4902-5035 | (B) decided models (was rf only), per fold; mRMR cap 200 kept |
| c18 | 5036-5175 | per fold, per decided model (was rf only); arms kept; base representations scored; per-fold partial file |
| c19 | 5176-5312 | per fold 2+2; budget 100 where decided; self-match drop by row (bug 7) |
| c21 | 5313-5439 | (B) decided models; extra-feature comparison only on ginkgo_colloidal and ginkgo_hic; H recipe on the decided grid (question) |
| c22 | 5440-5562 | per fold 2+2; both models explained every fold; KernelExplainer cost on svm/dnn/gp (question) |
| c23 | 5563-5681 | (B) one hybrid per evaluator condition per fold, decided models; gp not run (question) |
| c24 | 5682-5797 | (B) per decided model; recipe on one fold fewer; phi without standard error |
| c33 | 5798-5991 | per fold on decided models; distance arm removed; budget list wider than decided (question); folds from config |

Similarity: ALIGNED by chat 4 (2026-10-05, uncommitted, all compile and import). One row is one
experiment; runs are under "Chat 4 runs" in 2.0. "Neighbour arms" means knn's 2 best single
representations, knn's top distance hybrid and the top similarity hybrid, all per fold.

| experiment | audit lines | state after chat 4 |
|---|---|---|
| s01 | 5992-6092 | no selection (group A); now per fold (was all rows); all 10 representations (question); every score under each decided distance; violation rate at 5/10/20 neighbours, rank and inverse_square; SALI drops identical pairs; no p-values |
| s02 | 6093-6235 | per fold 2+2 per model plus neighbour arms; budget sweep on decided budgets from the same per-fold searches; concatenation arm removed (question) |
| s03 | 6236-6378 | per fold 2+2 per decided model (was rf, lightgbm, dnn), plus 2 hybrids per decided budget; no neighbours; information number renamed `summed_component_mi` (question); measured on training and held-out rows |
| s04 | 6379-6524 | per fold neighbour arms plus 2+2 per decided model scored on cliff and non-cliff molecules; every kNN setting; rf baseline removed; 100-feature-only read gone |
| s05 | 6525-6652 | (C) per fold neighbour arms plus each model's 2 general hybrids, every kNN setting; stored reference table, every-representation re-pick and budget loop removed; fold-0 drop gone |
| s06 | 6653-6773 | (C) per fold neighbour arms plus each model's 2 general hybrids; errors from the decided models (was rf); best single picked on training rows (was on the numbers it is compared with) |
| s08 | 6774-6939 | per fold; prediction over decided general budgets, distance over decided distance budgets; failing similarity search recorded once per fold (was retried per budget) |
| s09 | 6940-7049 | (A) reads the new per-fold files of s02, s04, s05, s06, s08; refuses files not written per fold; agreement check now tests something; Wilcoxon tests removed |
| s10 | 7050-7194 | per fold neighbour arms plus 2 general hybrids per model; sign flipped twice on rmse fixed; GP mean term kept and collapsed fits flagged; q-RASAR on the decided models at every kNN setting (cost question) |
| s11 | 7195-7342 | (B) per seed 2 hybrids per decided model from a search on that seed's starting rows (was 3 chosen by rf); starting set spans at least 2 groups; width stored per seed |
| s12 | 7343-7500 | per fold 2+2 per decided model (was 5 models on every representation) plus neighbour arms scored with kNN; one cliff partition per decided distance where there are no SMILES |
| s13 | 7501-7617 | per fold neighbour arms plus each model's 2 general hybrids; retrieval at every kNN setting; PCA controls no longer standardised twice |

Noise: aligned by chat 5, 2026-10-05, uncommitted. Status in "Chat 5 status" (2.0). One row per
experiment; "bugs" are the numbers of 2.7; OPEN items are hers, each marked `# OPEN (chat 5):` in
the file.

| experiment | audit lines | state |
|---|---|---|
| c10 | 7618-7743 | per fold 2+2 per decided model (8 fits) under clean labels plus each kind x level x 3 draws; 8-budget sweep and per-representation curves removed; bugs 1, 12, 13 (stale file) fixed. OPEN: keep damaged-label recipe searches (27 per fold on full_knn), budget question |
| n01 | 7744-7907 | per fold 2+2 per decided model (8 fits), clean labels; the hybrid is now compared with the model's 2 best single representations instead of the 3 weakest; bugs 1, 12 fixed; bug 10 is in shared code. OPEN: herg_ic50 repeats counted as cliffs, cliff rule (any neighbour) and counting scheme, augmented neighbour spaces |
| n02 | 7908-8044 | per fold 2+2 per decided model (8 fits), each with 1 unweighted and 4 weighted fits; 5 models on every dataset removed; bugs 1, 7, 12 fixed. OPEN: importance weighting kept, one neighbour count (10) or 5/10/20, clean labels only, gp takes no weights; not fixed: a failed fit is checkpointed and never retried, smoke check broken |
| n06 | 8045-8181 | fits no model; per fold scores 8 spaces (2+2 per decided model) plus 2 neighbour hybrids; neighbour estimators now at 5/10/20 neighbours under each of the group's distances; bugs 1, 12, 13 (p-values) fixed. OPEN: score the 2 neighbour hybrids, one number per estimator (median over settings), pooling folds by slot rank |
| n07 | 8182-8359 | per fold 2+2 per decided model (8 fits), was 5 models plus rf on every representation; activity_cliff_aware graded only at swept settings; bugs 1, 6, 12, 13 (stale file) fixed. OPEN: knn_smooth = label_smooth on class labels, 3 cleanups never run, as_they_come repeats level 0, searches on clean labels, augmented neighbour space |
| n09 | 8360-8493 | approved exception: dnn only, 2+2 plus a weighted copy of hybrid 1 (5 spaces); one dnn search per fold and damaged label set; bugs 1, 6 and its missing-search bug fixed. OPEN: search on damaged labels (55 searches per fold), fifth space, 16 datasets without dnn, second decided model, distance_selection's own neighbours |
| n11 | 8494-8627 | per fold 2+2 per decided model (8 fits); distance and similarity hybrids used for estimation only; hybrids keep the recipe's representations (were all 10); bugs 1, 5, 6, 12, 13 (AUC) fixed, 3 on single representations only. OPEN: kept neighbour settings differ between draws, recovery denominator, cost |
| n12 | 8628-8789 | per fold 2+2 per decided model (8 fits); distance and similarity hybrids are neighbour spaces only; per-fold held-out scores added; bugs 1, 6, 7, 12, 13 (AUC on probabilities) fixed. OPEN: importance weighting kept, Step 3 importance cuts, refits at every setting (about 1,250 per model per level and draw), dataset scope, gp takes no weights |
| n13 | 8790-8936 | per fold 2+2 per decided model (8 plain fits) per level and draw; Track A only where dnn is decided; neighbour detectors only at swept settings; bugs 1, 6, 12, 13 fixed. OPEN: Track A on 16 datasets without dnn, mentornet's own 10 neighbours, six flip kinds uncut, Track B on hybrids, non-neighbour detectors in neighbour hybrids |
| n14 | 8937-9072 | fits no model (exception group A); per fold the 8 configurations and 2 augmented single representations serve as similarity spaces, built on training rows; bugs 1, 3, 5, 12, 13 (t-test) fixed. OPEN: which molecules the cliff graph covers (now all), one forest per space, herg_ic50 repeats as cliffs, LightGBM on 16,405 features |
| n15 | 9073-9229 | per fold 2+2 per decided model (8 fits) on each side, clean-searched and noisy-searched; 1 clean plus 9 noisy searches per fold on full_knn; level 0 no longer 0 by construction; bugs 1, 6 fixed. OPEN: one damage kind or six (54 searches per fold), whether base representations count |
| n16 | 9230-9351 | fits no model; per fold importance is scored on 8 configurations, with hybrids rebuilt at each level; single representations cut from about 10 to 2 per model; level-0 floor restored, feature-count name fixed, bug 1 fixed. OPEN: Step 3 importance cuts, 15 classification datasets skipped, whole-pool question, third floor |
| n17 | 9352-9497 | per fold 2+2 per decided model (8 fits) per draw, under 11 conditions, on one row per entity (mean label); forest proximity now reaches the neighbour settings; bugs 1, 5, 11, 12 fixed. OPEN: search labels (entity mean), default path's 9-10 Euclidean neighbours; smoke script broken |
| n18 | 9498-9633 | per fold 2+2 per decided model (8 fits), no noise; ties counted apart from losses; mean baseline deleted; constant_lift scored once instead of 6 times; bugs 1, 5 fixed, 3 on single representations only. OPEN: default path's own-feature neighbours, augmented neighbour spaces, herg_ic50 svm skipped above 10,000 rows, constant_lift tied to first detector |
| n19 | 9634-9773 | per fold 2+2 per decided model (8 fits) on one row per molecule (mean label), plus 6 extra learners in the first model's 4; hybrid evaluator and representation count now recorded; bugs 1, 11, 12 fixed. OPEN: keep the 6 extra learners, search labels, share searches with n17/n26/n36; similarity-hybrid smoke test fails |
| n20 | 9774-9908 | per fold 2+2 per decided model (8 fits) per search (labels as they are, each cleaned search); xgboost on another model's recipe removed; bugs 1, 5, 6, 12, 13 (knn_smooth dropped on class labels) fixed. OPEN: default cleaning ignores kNN record (not fixed), cleaned search now scored on cleaned labels (not fixed), damage kind, flag rates, cost |
| n21 | 9909-10056 | per fold 2+2 per decided model (8 fits); about 1,296 weighted refits per level and draw on its own distances; duplicated hybrid distance removed; per-fold held-out scores added; bugs 1, 6, 7, 12, 13 fixed. OPEN: refit count, which configurations a distance repairs, Euclidean and importance weighting kept, Step 3 cuts, gp weights |
| n22 | 10057-10200 | fits no model; per fold detectors run in 8 spaces (svm and dnn, 2+2 each) on herg_ic50; searches now on the graded records; bugs 1, 7, 11 fixed. OPEN: herg_ic50 neighbour group (fast_knn at 9,802 molecules), counting scheme cannot be swept, neighbour hybrids as detector spaces (n06 does this), smoke test outdated |
| n23 | 10201-10344 | per fold 2+2 per decided model (8 fits); fallback neighbour move target deleted; activity_cliff_aware only at swept settings; move conditions skipped on class labels (were 720 errors); bugs 1, 5, 6, 7, 13 fixed. OPEN: which damage kinds the uncertainty sources need, rounding moved class labels, activity_cliff_aware out of main table |
| n24 | 10345-10487 | per fold 2+2 per decided model (8 fits); main table runs non-neighbour detectors only, neighbour cleanups at the top 3 settings; double outlier_p10 run and humanness label fixed; bugs 1, 5, 6 fixed. OPEN: one damage kind or every subset kind, knn_smooth = label_smooth on class labels, top-3 rank not a fixed setting |
| n25 | 10488-10633 | per fold 2+2 per decided model (8 fits) x detectors x cleanups x 3 cleaning sets; sweeps damage kinds and refuses cut kinds; bugs 1, 5, 6, 12, 13 fixed; bug 10 in shared code. OPEN: outlier_only_p05 unreachable, outlier_p05 moves every cliff label, random set short when cliffs dominate, herg_ic50 repeats as cliffs, default-path neighbours |
| n26 | 10634-10805 | per fold 2+2 per decided model (8 fits) on one row per molecule (mean label), plus 3 refits of the first model's base_1; augmentation passed to rebuilt hybrids; bugs 1, 11 fixed. OPEN: one model in repeat check, one draw per variance, searches duplicated with n19, herg_ic50 not on submit roster |
| n27 | 10806-10957 | per fold 2+2 per decided model (8 fits); its own StandardScaler and float64 casts removed; bugs 1, 3, 5, 13 (spread on one fold) fixed. OPEN: LightGBM out-of-fold prediction for relabelling, gaussian only, 800 fits per fold on kiba and bindingdb_kd |
| n29 | 10958-11082 | per fold 2+2 per decided model (svm, dnn: 8 fits) on one row per molecule, 6 methods in each; bugs 1, 3, 5, 11 fixed. OPEN: within_document_rank needs xgboost so no longer runs, LightGBM offsets, search on molecule means |
| n30 | 11083-11251 | per fold 2+2 per decided model (8 fits), loss swaps fitted inside their parent model's 4; bugs 1, 5, 13 (row format, one-fold spread) fixed. OPEN: swaps run on none of the 15 of 26 regression datasets lacking their parent model, censoring's 3 draws are identical |
| n31 | 11252-11387 | per fold 2+2 per decided model (8 fits), hybrids rebuilt per damaged level and draw; bugs 1, 5, 13 (row format) fixed, 3 on single representations only. OPEN: formula assumes undamaged molecules keep labels, gp takes no weights, sparse_dprime_up_p10 not run |
| n32 | 11388-11529 | per fold 2+2 per decided model (8 fits), was 7 models; level 0 runs last; detectors skip passes where no label moved; cut kinds refused; old result files recomputed; bugs 1, 5, 13 fixed. OPEN: knn_shapley needs equal neighbour shares, censoring's 3 draws are identical |
| n33 | 11530-11699 | per fold 2+2 per decided model (8 fits) x clean, noisy and 5 label conditions; NGBoost pair on every regression dataset in a host model's features; summary crash fixed; bugs 1, 5, 6, 13 fixed. OPEN: heavy-tailed kinds cut, NGBoost host, loss swaps only where parent decided, gp weights, NGBoost spread too small |
| n34 | 11700-11855 | per fold 2+2 per decided model (8 fits) x 5 conditions; bugs 1, 3, 5, 13 (row format) fixed. OPEN: grouped_wider kept as zero-offset control though cut, LightGBM offsets on 6 of 7 datasets where LightGBM is not decided |
| n35 | 11856-12005 | per fold 2+2 per decided model (8 fits), plus its own close-pair refit (restored) and one refit per top-3 setting; setting-name parse and augmentation rebuild fixed; bugs 1, 5, 13 fixed, 3 on single representations only. OPEN: keep own close-pair condition, what "top 3" ranks |
| n36 | 12006-12153 | per fold 2+2 per decided model (8 fits) per seed, on one row per molecule; augmentation passed to rebuilt hybrids; bugs 1, 5, 11, 13 (row format) fixed. OPEN: search labels include the held-back measurement, leverage and sparsity ignore counting scheme, LightGBM disagreement, settings differ by fold, augmented space where Step 8 cut it |
| n37 | 12154-12312 | per fold 2+2 per decided model (8 fits), each refitted 9 times for SIMEX; one noise estimate per representation (was rf's top one); negative estimates excluded; svm now fitted on herg_ic50; bugs 1, 3, 5, 11 fixed. OPEN: SIMEX noise size, estimate's distance, which representations, negative estimates, kiba/bindingdb_kd refused |
| n38 | 12313-12457 | per fold 2+2 per decided model (8 fits); bugs 1, 5, 13 (row format) fixed. OPEN: which of its 9 conditions to sweep (5 cut), which representation sets the noise unit; smoke script broken |
| n39 | 12458-12583 | per fold 2+2 per decided model (8 fits) from one clean-label search per fold on one row per molecule; stale oracle error fixed; bugs 1, 5, 11 fixed. OPEN: LightGBM error estimate, injected error size (108 passes per fold), search on clean labels; smoke test broken |
| n40 | 12584-12723 | per fold 2+2 per decided model (8 fits); GP, kNN and kriging in the host model's 4; retuning now tries the default; auc_norm per fold; summary crash and kNN/Cp distance mismatch fixed; bugs 1, 5 fixed. OPEN: dense_oofmse levels, host model, k grid, rank-weighted Cp, neighbours in own features (not fixed) |
| n41 | 12724-12874 | per fold 2+2 per decided model (8 fits); LOF minkowski p=0.5 crash fixed; cut kinds refused; bugs 1, 5, 12, 13 (row format) fixed. OPEN: censoring's 3 draws are identical, herg_ic50 at one row per measurement drops svm, LOF under Tanimoto and forest proximity, counting scheme has nothing to act on |
| n42 | 12875-13043 | per fold 2+2 per decided model (8 fits) per level and draw, each row fitted with its own model only (extra rf and svm gone); level 0 now runs last; bugs 1, 6, 7, 12 fixed. OPEN: caching dnn fits |

Audit line ranges for the other suites:

| suite | experiment | audit lines |
|---|---|---|
| federated | fed01 | 2211-2337 |
| federated | fed02 | 2338-2480 |
| federated | fed03 | 2481-2603 |
| federated | fed04 | 2604-2751 |
| federated | fed05 | 2752-2873 |
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

Shared code that every suite also needs (from the audit's shared section, lines 106-743):
- Each `run_all` (complexity, similarity, noise, federated helpers) builds whole-dataset searches
  at start-up. Remove them; per-fold searches are built where used.
- `own_search_kwargs` (c05, c07, c33, c15c) still names the fast grid and five models. Make it
  take the real dataset name and use the decided grid.
- `c02` imports `_apply_small_dataset_models`, now unused by `_shared`; delete both once c02 is fixed.
  DONE by chat 3: c02 no longer imports it and it is deleted.
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
| 1 | hybrid rebuild ranks representations by whole-dataset scores (`blueprint.rep_scores`) | fixed only in the 10 OPIG scripts and `fold_standard_selection` paths; every complexity file fixed by chat 3; every similarity file fixed by chat 4 (`build_fold_entry`) | every rebuild passes the fold search's own scores; listed in n06, n13, n17, n19, n22, n23, n26, n29, n31-n34, n36, n38-n41, c07, c08, c19, s02, s05, s06, s10(?), `fold_similarity_spaces`, `sx.build_spaces` |
| 2 | nested-CV rebuild ignored the recipe's representation count | fixed 2026-10-02 (`7e17900`, after the dirty run started) | confirm in the smoke run on buchwald_hartwig |
| 3 | 0/1 features scaled differently in four places | DECIDED 2.3 item 17 (0/1 unscaled); shared code chat 1; every similarity file uses FeatureScaler (chat 4) | list in the bug; decide before aligning similarity |
| 4 | svm on unscaled labels | fixed `fb1df54` (search and distance scorer); c06's family list in `_shared.get_model_family_variants` fixed by chat 3 (`SVRScaledTarget`) | closed |
| 5 | float32 vs float64 give different tree scores | OPEN, her decision: one precision | n32-n41 convert to float64 |
| 6 | "noise level 0" flips labels on classification | OPEN, her decision: skip the flip or exactly 0 | n09, n13, n20, n21, n23, n24, n25, n42 |
| 7 | dropping "self" can drop a twin | partly fixed (n02, `736ec93`); no similarity experiment calls `similarity_helpers.violation_rate` any more (chat 4; the function itself is unchanged, simopt still uses it) | `similarity_helpers.py:334`, `noise_mitigation_methods.py:487`, n07 |
| 8 | noise jobs ignored the local 5-representation cut | local only | ignore for the cluster |
| 9 | missing representations in the local setup | local only, except: chytorch_graphormer cache all zeros; ddg_combined cache holds 429 of 534 molecules | check both caches on the server |
| 10 | flab_expression_koenig has 5 groups | OPEN, her decision: how to group it | similarity search fails; c05, c24, n01, n25 |
| 11 | herg_ic50: search on 12,008 measurements, experiments on 9,802 molecules; svm skipped above 10,000 rows | OPEN; herg_ic50 also has no DECISIONS.md row | n17, n19, n22, n26, n29, n36 |
| 12 | scripts save only at the end | fix while aligning (write per fold) | c31, c32, n06, n42, fed05, opig_humanness, scaffold_hybrid, cgr_comparison, ginkgo, trastuzumab |
| 13 | smaller faults: wrong metric stamp (fed01, fed03a, fed04); two row formats (n30-n35); controls that equal the method (quantile_clip, huber_weight); knn_smooth = label_smooth on class labels; budgets filled with zero-gain features (c31, c32, fed04); significance tests (c14 removed by chat 3; c31, c32, s09, which she excluded); stale result files (c10, n07) | fix while aligning | per experiment |

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

### 2.8.1 Launcher smoke screen (Adelaide, 2026-10-05: "make sure these launchers work")

The experiments run on the cluster only through these launchers, so each is tested, not just the
python. Do this in chat 1 for anything chat 1 changes, and again in full in chat 6. Nothing is
submitted: `--dry-run`, `--list` and local runs only; anything that needs the cluster goes to her
as an srun block. Record each line as pass or fail with the output.

| launcher | what to check | how |
|---|---|---|
| `tests/slurm_scripts/runbook/submit_all_stages.sh` | stage 3 order is complexity, similarity, federated, opig, cross; every job's command line names a script that exists; exports carry `KIRBY_NO_COLD_SEARCH=1` (whole-dataset searches only, per-fold never refused) and no pruning mode that turns pruning back on | `--dry-run`, `--list`; read every printed sbatch line |
| `tests/slurm_scripts/submit_everything.sh` | the dataset roster (41 + chembl_qsar now decided; herg_ic50 reached only through noise), the OPIG_JOBS rows (`PM` removed from rows whose script ignores pruning), federated datasets | `--dry-run --checks-only` |
| `tests/slurm_scripts/submit_dataset.sh` | `resolve_experiments` lists exactly the run lists in 2.6 per category, in dependency order (c08 after c02; c09 after c02 and c06; n06 after n11 and n13); the start delay is 0 for complexity, 300 s similarity, 600 s others, and only under SLURM; a delayed job sets `KIRBY_SEARCH_REVERSE=1`; the job summary row is written; the "finished in an earlier job" skip works | run it locally on one small dataset per category with a 5-minute cut-off (no SLURM_JOB_ID, so no delay); then once with `SLURM_JOB_ID=test KIRBY_START_DELAY=5` to see the delay and the reverse flag; check `_job_summaries/*.tsv` |
| `tests/slurm_scripts/runbook/opig_one.sh` | every OPIG_JOBS name resolves; the output argument is right per script; the delay and reverse flag as above | `--list`; each name with a 5-minute cut-off locally |
| `tests/slurm_scripts/runbook/cross_one.sh` | s09, c03, c14 start and find their inputs | each with a 5-minute cut-off locally |
| `tests/experiments/run_dataset.py` | accepts every experiment name in every run list, and refuses c20, c31, c32, c43, simopt, f29 | `--experiments <id>` per id, cut off as soon as the first search line prints |
| `tests/slurm_scripts/submit_all_datasets.sh`, `queue_noise_conditions.py`, `runbook/stage1_prelim_noise.py`, `runbook/server_checks.sh`, `runbook/stage2_trials.sh` | still consistent with the above, or marked retired in their header if no longer used | `--dry-run` where it exists; otherwise read |

Two runs together on one dataset (the concurrency change, 2.2): start the same per-fold search in
two local processes, the second with `KIRBY_SEARCH_REVERSE=1`
(`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KIRBY_LGBM_THREADS=1` on this Mac, or they crash), and
check that the second prints "recipes stored by another job" and that the two together fit fewer
configurations than twice one. Measured 2026-10-03: 73 fits instead of 96.

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

See 2.0: six chats in order, and where a dynamic workflow is used (her explicit go, 2026-10-05).
Each chat is started with the prompt Adelaide pastes; it does only its own part.

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


## 2.13 Chat 6: results-audit items (from the results-audit session, Claude, 2026-10-05)

Another session fixed the faults found in the 2026-10-03 server results. Its record of every
item is `scratchpad/results_audit_2026_10_03/TRACKER.md`. Read it first. Never delete a row;
when you fix or change an item, update its status with your commit hash.

**Committed and pushed, do not undo:** e58106f, ccab773, e86df5c, e7d9ae2, 473855a, 6056c8d,
82ef2a1, eda240c, 4415a50. They fix:
- AUC from 0/1 predictions (s04 s05 s08 s10 s12 c07)
- fed05 running 3x
- c06 settings not reaching the model
- mislabelled metrics
- folds dropped or misnumbered
- removed p-values and confidence intervals
- rf forests that did not match the search's (c17 c18 c24)
- held-out features capped at their training range
- SALI leaving out identical pairs

They also store the width actually built per fold beside every budget. eda240c touched
`_shared.py` and `similarity_helpers.py`; only its own lines were committed, and your
uncommitted edits there were left alone.

**Owner's decisions, 2026-10-05:**
- When two selected hybrids are the same recipe, take the next distinct recipe instead of
  reporting one hybrid twice. Build this into the hybrid selection.
- Datasets with fewer than 10 groups (abagym x3, ginkgo x4) skip c05 and c07, with the reason
  recorded. Splitting inside groups would break the no-random-split rule.

**Yours to do:**
- `opig_blueprint_transfer.py`: the rewrite is right (own searches, `own_search=True`, no
  precompute needed) but uncommitted. Commit it TOGETHER with what it imports that is not
  committed: `fold_standard_selection`, `STANDARD_N_BASE` and `STANDARD_N_HYBRIDS` in
  `_shared.py`, and `tests/experiments/decided_grid.py` + `decided_grid.json`. On its own it
  fails to import on the server.
- `own_search_kwargs` in `_shared.py` (uncommitted edit) raises unless `dataset_name=` is
  passed. c05, c07, c15 and c33 do not pass it, so all four will crash until one side is fixed.
- `opig_shared_vectors_census.py` and `opig_humanness.py`: a failed hybrid search is only
  printed, not saved to the result JSON. The census error entry (~line 560) has no dataset name.
- Two recipes can share one name (`_shared.py` ~6893: the name leaves out the number of
  representations).
- c09's training-against-validation part cannot store per-fold scores:
  `evaluate_cv_with_train` in `_shared.py` returns only means and spreads.
