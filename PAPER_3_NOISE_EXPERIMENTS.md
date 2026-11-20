# Paper 3 noise experiments: the grid

The record of what Adelaide decided about the noise grid, and what is still open.
Written 2026-11-04. Adelaide
decides everything here. The code for some of it was written on 2026-11-04 and is being tested
(section 15).

How to read it:
- **Decided** means Adelaide said so, and her words are quoted.
- **Counted** means computed from the code. A count is a grid size, not a result.
- **Claude's reading** means not reviewed by Adelaide and not a finding.
- No number in this file is a scientific result, unless it names the thesis table it
  came from.

## 1. Decisions (Adelaide, 2026-11-04)

Sections 1 and 2 were decided while looking at n44. The noise condition cuts apply to n43, n44, n45,
n46 and n47 (Adelaide, 2026-11-04: "Yes cut the noise from all"). The second base representation, `base_2`, is cut from all five (Adelaide, 2026-11-04: "okay then the cut to all", answering
Claude's recommendation to cut the second base representation from all five). n43's and n47's
other hybrids are not covered by that cut and are open (section 3).

| What | Decision | Her words |
|---|---|---|
| Put-back and controls | Results first, then the checks only for what she chooses (wording below) | "First the results, then the checks for what I need" |
| Configurations | `base_1` and `hybrid`; `base_2` goes | "Lets do this: base_1 and hybrid only" |
| Repetitions | Stay at 3 per non-zero level (1 under censoring) | "Don't modify the draws thats legit" |
| Repeated runs | The same thing must not run more than once | "just make sure the same thing doesn't get run multiple times. This should have never been the case" |
| Time flags | Add them | "Yes add time flags" |
| Time limit | None set; record every time | "yea just record the times of everything" |
| Noise conditions | Cut `dense_oofmse`, `hetero_hidden` and `sparse_dprime_up_p10`; keep `hetero_seen` | "Keep hetero_seen" |
| `hetero_seen` levels | Keeps all its levels, 0, 0.1, 0.3, 1.0 | "okay I like that, do that", answering Claude's "Keep all `hetero_seen` levels." |
| Noise levels by dataset size | Large datasets run fewer levels (0, 1.0; censoring 0, 0.25); QM9 is excepted and keeps four; datasets of 5,000 molecules or fewer keep four | "LARGE DATASETS EXCLUDING QM9 RUN LESS"; "except qm9 that's a really important baseline to have" |
| Censoring datasets | Injected censoring runs on every dataset | "nah keep censoring in for everything" |
| More timing sessions | None for now. Time recording stays on, and future runs and the handoff carry it | "No I'm good for now. But keep track of timing. They'll be tests in the future and this needs to be included in the handoff" |
| Noise grid cuts | Still needed, in the handoff | "Beyond just implementation I do still need to cut down on the noise grid" |
| Documents first | This file, then a separate chat implements it | "everything we're talking about first needs to get documented, then a separate chat is going to handle the actual implementation" |

**Put-back and controls, as she approved it (names updated 2026-11-18):** in n44's standard
cleanup run, every cleanup runs with no put-back and no controls. Those plain results go
through the normal comparison. Then the variant cleanups run adds Cliff put-back with Random
put-back beside it, plus the controls, only for the results she chooses. The controls then
were `random_twin` and shuffled cleanup results. n44 no longer writes `random_twin`, and its
variant cleanups run also writes shuffled detector scores (commit 7d07b139).

**Cutting the second base representation (`base_2`) changes an earlier rule.** On 2026-09-25 she decided that noise experiments
compare against each model's top 2 single representations. Keeping only the first base representation (`base_1`) and `hybrid` is
the change to that rule.

**`sparse_dprime_up_p10` changes an earlier decision.** It was one of the six noise conditions decided
on 2026-10-05, and the thesis has results on it. Those results stay in the thesis.

**Unchanged:** 5 folds (Adelaide, 2026-10-30) and the dataset's two decided models.

## 2. The noise conditions

What remains: `gaussian`, `outlier_p05`, `grouped_shifted`, `censoring`, `shift_only_p10`,
`hetero_seen`.

The "Assay source" column is **Claude's reading** of thesis Table `tab:noise_sources`, for
her to check.

| Noise condition | Status | What it does | Assay source | What the thesis has |
|---|---|---|---|---|
| `gaussian` | Keep | Small independent error on every label | Variation between repeats: seeding density, passage number | Noise shape changed little in Chapter 3; Huber blend helped hERG Ki under it |
| `dense_oofmse` | **Cut** | Gaussian again, sized by model error | Same | None. Thesis says only how the injector works |
| `outlier_p05` | Keep | 5% of molecules get error 3 times wider; the rest still move | Gross errors | Delete flagged helped on LogD |
| `hetero_hidden` | **Cut** | Random error width per molecule | Same idea | None |
| `shift_only_p10` | Keep | 10% of labels moved up, in label SDs; the rest untouched | One-direction bias, such as autofluorescence | Cliff-aware gap detector beat the Forest residual; two-learner deletion made the model worse on hERG Ki and Caco-2 |
| `sparse_dprime_up_p10` | **Cut** | The same move, in SDs of model error | Same | Thesis Table `tab:noise:kinds`: "As above, in a different unit" |
| `grouped_shifted` | Keep | Each scaffold family shares an offset | Differences between laboratories or sources | Hardest condition in Chapter 3; scaffold-family offset helped under injected damage, found nothing on the labels as delivered |
| `censoring` | Keep | Labels clipped at a limit | Assay detection limits | Censored EM won back much of the loss with no detector |
| `hetero_seen` | Keep | Error grows with heavy-atom count | Weak fit. No source for heavy atoms found in the thesis | None. The only noise condition where error size follows the molecule |

Why each was cut, as discussed: `dense_oofmse` is the Gaussian pattern with less noise.
`hetero_hidden` is close to `outlier_p05`, since both give molecules random error widths.
`sparse_dprime_up_p10` is the same move as `shift_only_p10` in a different unit.
`dense_oofmse`, `hetero_seen` and `hetero_hidden` were added on 2026-11-02 so that noise
sized by label spread could be compared with noise sized by model error. With two of the
three gone, that comparison now rests on `hetero_seen`.

**Assay noise that no noise condition covers (Claude's reading):**
- Plate-position error (edge, row, column effects).
- A label attached to the wrong molecule. `shift_only_p10` only moves labels up, so it
  does not match this.

**Censored EM and the scaffold-family offset are not in n44.** They are the fixes the
thesis results above rest on, and they live in n46. n44 only runs its general
detectors and cleanups under those two noise conditions.

## 3. Still open

### Noise results review: the largest remaining cut
Adelaide: "I am not sure that I want to use all the detectors, this needs to be looked over
based on preliminary results. Everything in here needs to be reviewed more thoroughly...
Lets do that review later, its important so it needs a lot of work."
Also: "those should be massive cuts. Right now I'm throwing everything at the kitchen sink
for noise."

Adelaide: it should look "at more than just detectors, but everything in that noise mitigation
grid". So it covers everything inside one configuration, at one fold, noise condition, noise level and repetition (section 5).
That means detectors, cleanups, all their settings,
detector feature spaces, neighbor settings, which detectors pair with which cleanups, and
`gp_smooth`. It has not started. **Every size in section 4 is before this review.**

**On hold (Adelaide, 2026-11-05):** "hold off on the grid review, I'm waiting for ACs." It goes in a
handoff that combines these decisions with the cliff noise grid.

### The cliff grid
Adelaide: "it gets worse when I factor in ACs... Lets just tackle this first, then we'll get
to the cliff grid." Comes next, after this document. Related, in `AC_GRID_TEST_PLAN.md`:
- The variant cleanups run's Cliff put-back needs a definition of an AC.
- n44's variant cleanups run reads the definitions of an AC named in its JSON file (`cliff_definitions`, a list of definition keys or "all", no default). The old single definition of an AC (0.55 Tanimoto, multiple 2.0) runs only when named as `n25_existing`.
- Its `cliff_aware_cleanup` waits on the best detector and best cleanup from these noise
  results.

### Smaller questions, one per line
- Whether n43 keeps `hybrid__general2` and `weighted_hybrid`, and n47 keeps `hybrid__general2`.
  The noise condition cuts and the cut of the second base representation (`base_2`) are decided for all five (section 1).
  Goes to the handoff that combines this with the cliff noise grid (Adelaide, 2026-11-05: "this needs to be pushed off to the handoff").
- Which results get the variant cleanups run. She chooses after seeing the standard cleanup run's scores.
- What to do with `gp_smooth`, which goes to the review.
- Every regression dataset's count is in `AC_GRID_TEST_PLAN.md` section N5 (recounted 2026-11-07).
- The trusted set for MentorNet DD and Meta-Weight-Net. Meta-Weight-Net uses the held-back
  selection molecules of the training half, which carry the training labels' noise. Either DD uses
  the same molecules with that caveat written beside it, or a real trusted set is named, for example
  molecules measured three or more times with a small spread on the repeat datasets. The code
  still carves the selection molecules: `src/kirby/utils/lnl_methods.py` lines 1868 to 1871 (the carve itself at line 1923 for MentorNet and line 1662 for Meta-Weight-Net), read
  2026-11-17.
- For confident learning on a regression dataset, which model cleanlab refits to find the
  suspect labels: this suite's DNN, which is the default and about 80 DNN fits per
  call, or LightGBM, which is the detector's own call (`lnl_methods.ConfidentLearning`,
  `issue_model`).
- Whether `grouped_shifted` belongs in `noise_helpers.DETECTION_IS_DEFINED`. Read 2026-11-17,
  that tuple holds `grouped_wider` and not `grouped_shifted`. It is a question about the
  injector: `grouped_shifted` moves every molecule in an affected scaffold family and leaves
  the rest alone, so a "which molecules moved" set exists for it in a way it does not for
  gaussian. Whether the injector records that set has not been checked.
- What n36's re-measurement priority ranks molecules by. `measurement_spread` is the standard
  deviation of all of a molecule's repeat measurements and reaches every molecule measured
  twice or more. `training_label_error` is how far the one measurement that became the
  training label sits from the mean of its others, and reaches only molecules measured three or
  more times. Both are in `SPREAD_SCORES` and the default is `measurement_spread`, read
  2026-11-17.
- The slow step in `n01._compute_reaction_tanimoto_sim`: it builds a fingerprint per reaction
  and then calls `scipy.spatial.distance.cdist` with the Jaccard metric over all 3,057 by
  3,057 pairs of Buchwald--Hartwig. On 2026-11-02 that step held n44 and n47 past an 8-minute
  box on that dataset. It is one computation per dataset. Buchwald--Hartwig is in Paper 2 only
  since 2026-11-02; n44 still reaches the same function through
  `n25.cliff_mask_for_dataset` when its old single definition of an AC is named.

## 4. Grid sizes

A **configuration** is one model on one input. One **result** is one cleaned training set
scored for one configuration, at one fold, noise condition, level and repetition.

**LogD in n44, before the decisions:**

| Grid number | Count | Running total |
|---|---|---|
| Combinations of noise condition, level and repetition | 46 | 46 |
| Folds | x 5 | 230 |
| Models (dnn, svm) | x 2 | 460 |
| Inputs (`base_1`, `base_2`, `hybrid`) | x 3 | 1,380 combinations of configuration, fold, noise condition, level and repetition |
| Cleaned training sets per combination of configuration, fold, noise condition, level and repetition | x about 4,850 on average | 6.7 million |
| Put-back (none, Cliff put-back, Random put-back) | x 3 | 20.1 million results |
| Controls (none, `random_twin`, shuffled cleanup results), if on | x 3 | 60.2 million results |

**Those 46 combinations of noise condition, level and repetition on LogD.** Level 0 has one repetition per noise condition.

| Noise conditions | Levels | Repetitions | Combinations of noise condition, level and repetition |
|---|---|---|---|
| `gaussian`, `outlier_p05`, `grouped_shifted`, `shift_only_p10`, `sparse_dprime_up_p10`, `dense_oofmse` | 0, 1 | 3 | 4 each, 24 total |
| `hetero_seen`, `hetero_hidden` | 0, 0.1, 0.3, 1 | 3 | 10 each, 20 total |
| `censoring` | 0, 0.25 | 1 | 2 |

Caco-2 has 84 combinations of noise condition, level and repetition, because its levels are 0, 0.5, 1, 1.5 (censoring 0, 0.1, 0.25, 0.4).
Table in `HANDOFF_PAPER3_GRID_SIZE.md` (in git history) section 2.

**After the section 1 decisions, standard cleanup run only.** Recounted 2026-11-18 from the code, with
fitting stubbed out (`scratchpad/paper3_grid_counts/count_paper3.py`). Level 0 once, three noise conditions
cut, the first base representation and the hybrid, no put-back and no controls. These are counts, not times. The hours
that stood here were computed from the older counts and are not recomputed.

One row is one dataset and one choice of `hetero_seen` levels.

| Dataset | `hetero_seen` levels | Combinations of noise condition, level and repetition | Combinations of configuration, fold, noise condition, level and repetition | Results |
|---|---|---|---|---|
| LogD | Current (0, 0.1, 0.3, 1) | 23 | 460 | 1,932,270 |
| LogD | Level 1 only | 17 | 340 | 1,445,730 |
| Caco-2 | Current (0, 0.1, 0.3, 1) | 49 | 980 | 4,175,010 |
| Caco-2 | Level 1 only | 43 | 860 | 3,688,470 |

For comparison, before the decisions: LogD 20.1 million results, Caco-2 37.1 million.

What these counts leave out:
- The variant cleanups run, which adds the checks.
- The shuffled detector scores rerun added to n44 on 2026-11-18 (commit 7d07b139).
- The results review.
- The per-fold searches. Fold 0's search on LogD did not finish in 2 hours on the laptop.

Every Paper 3 dataset, and n43, n45, n46, n47, n15, n20 and n36, are counted in
`AC_GRID_TEST_PLAN.md` section N5 and in `scratchpad/paper3_grid_counts/paper3_counts.csv`.

**The arithmetic.** A configuration on a base representation, at one fold, noise condition, level and repetition,
has 5,468 results, or 4,988 without flags of known content. A configuration on the hybrid
has 3,601, or 3,121 without. Flags of known content exist only at non-zero levels of
`outlier_p05`, `censoring` and `shift_only_p10`. So one combination of noise condition, level and repetition, on the first base
representation and the hybrid, is 5 folds x 2 models x (4,988 + 3,121) = 81,090 results.
With flags of known content it is 90,690. LogD
current: 16 x 81,090 + 7 x 90,690 = 1,932,270. The drop from the earlier 5,844 and 3,977 is
`reliability_weight`, which now runs once per neighbor setting rather than once per detector
(section 5).

## 5. Inside one configuration on a base representation, at one fold, noise condition, level and repetition

This is what the results review has to go through. Counted by running `n44.run_unit` with
fitting stubbed out. The counter gave exactly 4,032 when set up like `_smoke_n44.py`.
Recounted 2026-11-18 (`breakdown.py`); only the counts changed, not the settings.

**5,468 cleaned training sets, by detector.** One row is one group of detectors.

| Detectors | Count | Cleaned training sets |
|---|---|---|
| Detectors that give a score | 32: 11 at 118 each, 21 at 116 each | 3,734 |
| Neighbor detectors | 4, 165 each | 660 |
| `oracle`, only where the noisy molecules are known | 15 settings | 480 |
| `conformal_bh` | 6 settings | 384 |
| `two_learner_small_loss` | 3 settings | 192 |
| No detector | n/a | 18 |

The 32: `cross_val`, `ensemble`, `leave_one_out`, `hybrid`, `gaussian_process`, `hetero_gp`,
`heteroscedastic_nn`, `conformal`, `deep_ensemble`, `bayesian_rf`, `quantile_regression`,
`quantile_forest`, `ngboost`, `neighbor_label_spread`, `label_zscore`, `feature_outlier`,
`net_small_loss`, `net_loss_mixture`, `net_josrc_test`, `net_meta_weight`,
`net_evidential_spread`, `net_trusted_weight`, `oof_residual`, `ngboost_rescaled`,
`quantile_forest_rescaled`, `conformal_rescaled`, `knn_shapley`, `cleanlab`,
`isolation_forest`, `lof`, `residual_low_disagreement`, `random`.

The 4 neighbor detectors: `neighbor_residual`, `local_variance_ratio`, `distance_weighted`,
`activity_cliff_aware`.

A configuration on the hybrid has 3,601, because it scores detectors in one feature space rather than
two.

**The same 5,468, by cleanup, with the number of settings each has.** One row holds two cleanups.

| Cleanup | Settings | Results | Cleanup | Settings | Results |
|---|---|---|---|---|---|
| `exp_downweight` | 4 | 436 | `network_rewrite` | 3 | 261 |
| `reliability_weight` | 4 | 12 | `ema_rewrite` | 3 | 261 |
| `downweight` | 3 | 327 | `rewrite` | 3 | 261 |
| `huber_weight` | 3 | 327 | `top_share_downweight` | 3 | 261 |
| `js_shrink` | 3 | 327 | `bayes_shrink` | 3 | 261 |
| `oof_smooth` | 3 | 327 | `shrink_flagged_one_weight` | 3 | 261 |
| `gp_smooth` | 3 | 327 | `shrink_flagged_per_row` | 3 | 261 |
| `move_confidence` | 3 | 327 | `winsorise_flagged` | 3 | 261 |
| `label_smooth` | 3 | 324 | `score_weight` | 1 | 45 |
| `knn_smooth` | 3 | 324 | `spread_invvar_weight` | 1 | 10 |
| `remove` | 3 | 261 | `winsorise` | 3 | 3 |
| `invvar_weight` | 1 | 1 | `constant_lift` | 1 | 1 |
| `huber_teacher_blend` | 1 | 1 | `prox_smooth` | 1 | 0 here, runs only on a forest neighbor setting |

n44 has 26 cleanups, not the 23 in the earlier handoff; the smoke test only reached 23.
`label_smooth`, `knn_smooth` and `reliability_weight` also repeat under the neighbor settings
kept for that configuration, fold, noise condition, level and repetition (3 in this count).

## 6. Repeats and results that do no work

**Rule from Adelaide:** the same thing must not run more than once. Known cases, and the
implementation chat must look for others:

**Level 0.** Level 0 returns the labels as delivered for every noise condition, so all 9 noise conditions at level 0
had identical labels. Each noise condition still ran them and wrote them to its own file. On LogD that is
8 repeated noise conditions at level 0, or 3.4 million of the 20.1 million results. n44's level-0 results are the
comparison for the noise condition, so each noise condition must be able to read the one set.

**`reliability_weight`.** Results are labelled by detector, but the cleanup never reads that
detector's scores. 32 detectors x 4 strengths x 3 neighbor settings gives 384 results, with
12 different training sets. The other 4 of its 388 are given no neighbors and produce nothing.
Changed by 2026-11-07: it now runs once per neighbor setting, 12 results per configuration at one fold, noise
condition, level and repetition.

**Same training set, different names, checked 2026-11-16.** One n44 configuration on a base representation, at one
fold, noise condition, level and repetition, was run with every training set hashed. It used
made-up data: 128 training molecules, rf, `outlier_p05` level 1, without neighbor settings. Of 3,940 results with a training set, 2,911 training sets were
different, so 1,029 results repeat one already scored in that check. A configuration on the hybrid
repeated 554 of 2,213. From the code, three cleanups cause most of it:
- `shrink_flagged_one_weight` and `shrink_flagged_per_row` are labelled by detector, but they
  re-flag molecules by out-of-fold error and read only how many the detector flagged. Every
  detector at one share of training molecules flagged gives one training set, up to 38 names per set.
- `huber_weight` gives every molecule weight 1 when the detector's median score is 0, as with any
  flag set under half the molecules. One training set had 231 names.
- `bayes_shrink` changes no label when the flagged molecules fit no worse than the rest. 162
  results were the damaged labels unchanged, which is D.
None of these was changed. Each repeat is fitted once: the fit is keyed on the training set's
content, not the method's name, and read from the fit cache after that (checked 2026-11-18,
`tests/test_repeated_training_set_fitted_once.py`, 3 passed). Since 2026-11-20 the two
shrinking cleanups read no detector and run once per share of training molecules flagged (5%,
10% and 20%), in the model's own space. Her words: "Shouldn't I just used the fix number and not
run the detector?" and "okay continue doing that". Groups are in
`/Volumes/seagate/kirby_smoke_logs/2026-11-16/n44_groups_no_neighbours.json`.

With neighbor settings, 2026-11-18, same data, fold, noise condition, level and repetition: the configuration on the base representation repeated
1,228 of 5,188 results (3,960 different training sets). The configuration on the hybrid repeated
746 of 3,461 (2,715 different training sets). The
three causes above still account for most. Two more were seen on this data, cause not read.
`js_shrink` and `move_confidence` gave the same set on flags of known content at the same strength.
`neighbor_residual` and `distance_weighted` flagged the same molecules at k = 10. Groups are in
`/Volumes/seagate/kirby_smoke_logs/2026-11-16/n44_groups.json`.

**`random_twin` and shuffled cleanup results.** `controls.py` says `random_twin` and the shuffled cleanup results are the same when a method changes
every molecule, as `downweight` and proportional `label_smooth` do. They differ only when some
molecules are left alone. n44 no longer writes `random_twin` (commit 7d07b139); other experiments still do.

**Repeats reuse a saved fit, but still cost a result.** `n07.dprime_rows` says every fit goes
through `noise_cache.cached_fit_predict`, and its key is the content of the features, labels
and weights, not the noise condition. So an identical training set should not be refitted, though
this was read in the code and not measured. Removing repeats may therefore save less time than
the result counts suggest.

**`gp_smooth`, a different problem.** The grid schedules 327 results per configuration on a base representation, with its fold, noise condition, noise level and repetition, but
the code refuses above `GP_SMOOTH_MAX_ROWS`, 1,000 training molecules. LogD's fold 0 has 4,022.
Each refusal writes a result with no score and prints a line, which is what filled the server log.
So it fits nothing on LogD, and costs almost no time. It goes to the results review.

## 7. Measurements

**Speed: about 750 results per hour.** On the server, 8 cores and 64 GB, LogD, `outlier_p05`, with
Cliff put-back and Random put-back on, and `random_twin` and shuffled cleanup results off, using `KIRBY_NOISE_PROMOTE=n44`:

| Time, 2026-11-04 | Results on disk |
|---|---|
| 18:55:47 | 10,935 |
| 19:43:59 | 11,535 |

That is 600 results in 48 minutes, from one part of the first configuration at one fold, noise condition, level and repetition. Results arrive in bursts,
and some are cheap (refused `gp_smooth`, reused fits), so the real speed could differ a lot.

**Timed sessions, 2026-11-05.** Server, 8 cores, 64 GB, `outlier_p05`, code for the standard cleanup run, read part way through
from `noise_timing.py` under `/data/stat-ecr/scat9264/paper3_timing`. Claude's reading, not reviewed by Adelaide.

- Caco-2, dnn: 490 results, of which 338 fitted, 101 read from saved fits, 51 refused. 340 fits took 494 seconds,
  median 1.445 seconds. Only 10 of the 32 score detectors had been reached: `hetero_gp` 915 s, `gaussian_process`
  212 s, `deep_ensemble` 122 s. Also the neighbor search under each neighbor setting, plus the screen, 1,401 s, out-of-fold predictions 729 s in 3 runs,
  correction targets 547 s, search 458 s, `huber_teacher_blend` 182 s.
- LogD: 4,013 results, of which 3,761 read from saved fits and 252 refused. No model fit ran, so this session gives
  no fit time for LogD. Detectors that ran: `quantile_regression` 4,017 s, `ngboost` 704 s, `ngboost_rescaled` 700 s,
  `quantile_forest_rescaled` 515 s, `quantile_forest` 513 s, `conformal_rescaled` 197 s. Search 1,273 s in 2 runs,
  out-of-fold predictions 3,862 s in 5 runs. Most other detectors read saved scores, so their LogD time is unknown.

**Timed sessions, second read, 2026-11-06.** Same folder. Claude's reading, not reviewed by Adelaide. dnn only; no svm fit yet.

- dnn fit time: Caco-2 median 1.438 s (2,799 fits, 3,987 s); LogD median 2.570 s (3,100 fits, 8,106 s).
- Caco-2 results: 3,977, of which 2,797 fitted, 928 read from saved fits, 252 refused. LogD's split is not representative,
  because its fits were reused from the earlier server run.
- LogD finished its first configuration at one fold, noise condition, level and repetition (fold 0, level 1, repetition 0, the DNN on the first base representation). The log line reports 1,551 results written and 5,237 s. That
  covers only the last session, so it is not that configuration's whole time at that fold, noise condition, level and repetition.
- LogD slowest detectors, seconds: `quantile_regression` 4,017, `residual_low_disagreement` 3,864, `flag_detectors` 2,083.
  10 of its detectors read saved scores, so their LogD time is unknown.
- Caco-2 slowest detectors, seconds: `hetero_gp` 915, `residual_low_disagreement` at least 723 (stopped by an interruption),
  `quantile_regression` 475, `flag_detectors` 278, `gaussian_process` 212.
- Claude's estimate, not a measurement: about 3 hours on Caco-2 and at least 7.5 hours on LogD.
  That is for the DNN on the first base representation, at one fold, noise condition, level and repetition.

**Not yet measured:**
- The time for one whole configuration at one fold, noise condition, level and repetition. Run `grep "rows, " /data/stat-ecr/scat9264/n44_logd.log`; no line had
  appeared at the last check, with 8,853 results saved earlier.
- How many results are real fits.
- The svm, which has not been reached as far as the log shows.
- Any time per detector or cleanup.

**Time on other steps, LogD:** on the laptop `pharmacophore3d` took 25 minutes to build, and the
fold 0 search was unfinished after 2 hours with its threads capped. The server got through its
search and into n44 within its first session, though the exact time was not recorded.

**That server run is a measurement only.** It uses the old grid, and writes to
`/data/stat-ecr/scat9264/paper3_grid_size/n44_logd`. Its results are not for the paper.

## 8. Time flags

Decided: add them. No limit is set: Adelaide, 2026-11-04, "yea just record the times of everything".
Nothing is flagged or dropped; times are only recorded.

Checked in the code 2026-11-16. `tests/experiments/noise/noise_timing.py` writes one log per
experiment, dataset and noise condition, `<output>/<experiment>/<dataset>__<kind>.timing.jsonl.gz`,
one line per step with its seconds. Read with `python tests/experiments/noise/noise_timing.py <folder>`.

| Experiment | What is timed |
|---|---|
| n43 to n47 | each detector, cleanup, result, model fit read or computed, injection, search. n44, n45 and n46 also time one call per configuration at one fold, noise condition, level and repetition, logged as `piece`: `run_unit` in n44 and n46, `grade_unit` in n45. In n44, `correction_targets` and `neighbour_lists_and_screen` are timed on their own lines outside it, so `piece` is not the configuration's whole time. All five log the repetition as `draw` |
| n15 | the whole run, each search, each fit and score, each fold and each level and repetition, with the repetition logged as `draw` |
| n20 | the whole run, each search, each (regime, level, repetition, fold), with the repetition logged as `draw` |
| n36 | the whole run, each search, each (fold, seed) |
| `cliff_aware_cleanup` | the whole run, each fold, M's scoring, each cleanup, each result, the neighbor search under each neighbor setting |
| `error_by_cliff_status` | the whole run, each fold, each model fit |

n15, n20 and n36 have no noise condition file of their own, so their log is
`<dataset>__<kind or all>.timing.jsonl.gz`. The two cliff experiments write `<dataset>__all`.

## 9. Checks: what they answer

**Cliff put-back:** did cleaning hurt because it touched activity cliff molecules? **Random
put-back:** the comparison for Cliff put-back, so it only means something beside it.
**Controls:** a control is a fake cleanup, the same amount of change on the wrong molecules. If the real
cleanup cannot beat it, the cleanup was not finding bad labels.

The controls default to off. n44 writes them only in its variant cleanups run, which `KIRBY_N44_SECOND_RUN` turns on, and does not read `KIRBY_NOISE_CONTROLS`. In the other experiments `controls_enabled()` reads `KIRBY_NOISE_CONTROLS`, and neither `setup.sh` nor
the server commands used here set it.

From the thesis, `sec:noise:res:qm9test`: with the DNN, passing the shuffled-score control means
little alone. On QM9's first single representation, Neighbor-mean smoothing beat all three
shuffles with no noise to find. That test shuffled the detector's scores, which is similar to
the shuffled cleanup results control but not the same: that control shuffles the cleanup's output. Since 2026-11-18 n44's variant cleanups run also writes `shuffled_scores_twin`, which shuffles the detector's scores and runs the cleanup again.

## 10. What 3 repetitions means

The same noise condition and level, injected three times with new random noise. The random seed depends only on the condition
and the repetition (`noise_draw_seed`). So a molecule gets the same random numbers in every fold, scaled by that fold's training labels. The same
molecules are chosen at every level of one repetition. Repetitions show how much a result depends on which
molecules were hit. From thesis Table `tab:noise:res:family_offsets` (DNN on the first hybrid,
grouped-shifted, level 1.0, damaged labels, held-out RMSE):

| Dataset | Repetition 0 | Repetition 1 | Repetition 2 |
|---|---|---|---|
| LogD | 0.684 | 0.690 | 0.672 |
| QM9 | 0.909 | 0.844 | 1.295 |

Censoring has one repetition because its clipping reads no seed.

## 11. The other experiments, on Caco-2

Recounted 2026-11-18 from the code, as the Papers ledger now runs them (`count_paper3.py`).
The second column is recounted. The inside column was counted before the section 1 decisions
and is not recounted, except n47's removal curve. One row is one experiment on Caco-2.

| Experiment | Count of the experiment's own unit of work (its factors in brackets) | What multiplies inside |
|---|---|---|
| `n43` | 245 (49 combinations of noise condition, level and repetition x 5 folds) | Not recounted. Before: 1 dnn search; 5 feature spaces x 9 network methods = 45 network trainings; 268 results per feature space, or 516 under 1,000 training molecules, when the GP methods run |
| `n45` | 800 (40 combinations of noise condition, level and repetition x 2 models x 2 inputs x 5 folds) | Not recounted. Before: 120 fits per configuration, fold, noise condition, level and repetition where noise can be graded; 32 detectors and 420 neighbor grades with no fits |
| `n46` | 880 (44 combinations of noise condition, level and repetition x 2 models x 2 inputs x 5 folds) | Results per configuration, fold, noise condition, level and repetition not counted. The sources, repeat and real-censoring passes are not in this count |
| `n47` | 1,320 (44 combinations of noise condition, level and repetition x 2 models x 3 inputs x 5 folds) | Removal curve: 5 shares x 20 seeds x 2 models x 3 inputs x 5 folds = 3,000 fits in all |

n45, n46 and n47 run no `hetero_seen`; you have not said whether they should. n46 and n47 run
level 0 once per noise condition, so they have 4 more combinations of noise condition, level
and repetition than n45.

## 12. Corrections to earlier documents

- `HANDOFF_PAPER3_GRID_SIZE.md` (in git history) says 4,032 fits per configuration, fold, noise condition, level and repetition, 10.2 million on Caco-2, 5.6 million on LogD.
  The smoke test behind it has no neighbor settings, no co-teaching and no put-back. Before the 2026-11-18 recount, the counts were
  5,844 results on a base representation and 3,977 on the hybrid, per configuration, fold, noise condition, level and repetition, before put-back. The totals with put-back were 37.1 million results on Caco-2 and 20.1 million on LogD. Section 4 now gives 5,468 and 3,601, without put-back, and does not recount the totals with put-back.
- That handoff's section 9 says `KIRBY_NOISE_PROMOTE=1`. That gives the full levels on fold 0 only.
  Five folds need `n44` or `all`.
- An earlier write-up blamed the Seagate drive for the 2026-11-02 crashes. On
  2026-11-03, two laptop runs of n44 crashed at the start of fold 0, with two OpenMP libraries
  loaded (`libomp`, `libiomp5`) and a `libiomp5` thread dying inside `libomp`. With threads capped,
  a run lasted 2 hours with no crash, though parts still used about 4 cores. The drive was present
  just after the second crash. So the cause points at the libraries, unproven, and the older
  crashes have not been re-read. The server runs did not crash this way.

## 13. Status of `HANDOFF_PAPER3_GRID_SIZE.md` (in git history) section 7

| Its question | Status |
|---|---|
| Which datasets censoring runs on | Decided: every dataset (section 1) |
| Whether 3 repetitions stay 3 | Decided: they stay |
| Whether full-grid datasets keep 4 noise levels | Decided: they keep 4; larger datasets run fewer, except QM9 (section 1) |
| Whether cleanups and detectors stay whole | Open: the results review |
| Whether the two heteroscedastic noise conditions and `dense_oofmse` stay | Decided for all five: `dense_oofmse` and `hetero_hidden` cut, `hetero_seen` kept |

## 14. How the counts were made

Scripts are in `scratchpad/paper3_grid_counts/`. Run from the repo root with
`PYTHONPATH=src:tests:tests/experiments KIRBY_NOISE_PROMOTE=1`:
- `count_n44.py`: results per configuration, fold, noise condition, level and repetition, and totals.
- `breakdown.py`, given `count_n44.py`'s path: settings, detectors and cleanups.
- `lv.py`: levels per noise condition. `count_kinds.py`, `count_n43_n45.py`: the section 11 counts before
  2026-11-18.
- Since 2026-11-18: `dataset_sizes.py` loads each Paper 3 dataset and writes its molecule count
  to `dataset_sizes.json`. `count_paper3.py` reads that and writes `paper3_counts.csv`, one line per dataset and experiment.
  Its `versions` column counts combinations of noise condition, level and repetition. Its `pieces`
  column is the experiment's own unit of work, and `results` is n44's results in the standard cleanup run. All are counted as the Papers ledger runs them. Run it
  with `PYTHONPATH=src:tests:tests/experiments` and no promote flag; it sets the flag itself.
- `repeated_training_sets.py`: the section 6 check of repeated training sets.

## 15. Build status

Updated 2026-11-07. `AC_GRID_TEST_PLAN.md` Part 2 holds the code state, the test state, the recounted
grid, the open list and the prompts for the next chats. Sections 4, 5 and 11 here predated the
`reliability_weight` change and the recount there. They were recounted on 2026-11-18, except section 11's inside column (last bullet of this section).

- Noise condition cuts in n43 to n47: `noise_helpers.PAPER3_CUT_CONDITIONS`.
- The second base representation (`base_2`) cut from the configurations in all five. The spaces where neighbors are found on the larger noise grid still read it
  (`similarity_space.GROUP_SPACES`). Adelaide has not said whether the cut covers those.
- Time recording: `noise_timing.py`, one log per experiment, dataset and noise condition.
- n44 standard cleanup run: no put-back, no shuffled cleanup results and no shuffled detector scores. Variant cleanups run: `KIRBY_N44_SECOND_RUN`.
- Level 0 once in n43, n44 and n45, and `reliability_weight` once per neighbor setting. Adelaide,
  2026-11-07, on this code: "yes that code was supposed to be implemented that is correct".

Updated 2026-11-16, checked in the code that day:
- The Papers ledger runs n43 to n47 on the Paper 3 noise grid. The worker sets `KIRBY_NOISE_PROMOTE` to
  the experiment's own name for those five units only, so they run five folds and four levels on
  datasets of 5,000 molecules or fewer (QM9 loads 4,995) and two on larger ones. n15, n20 and n36
  keep the screen. `hetero_seen` is a ledger unit for n43 and n44, the two that run it.
  (`tests/experiments/papers/plan.py` `PAPER3_FULL_GRID_NOISE`, `worker.py`.)
- n47 no longer computes cliff numbers. Which molecules are cliffs and held-out error on cliff
  molecules are in `cliffs/cliff_sets.py` and `cliffs/error_by_cliff_status.py`. Whether cliffs
  come in groups or as lone pairs is now in `cliff_sets`, under every definition.
- Time recording covers n15, n20, n36 and the two cliff experiments (section 8).
- Counts recomputed 2026-11-18: sections 4, 5 and 11, and `AC_GRID_TEST_PLAN.md` section N5.
  Section 11's inside column is still the older count, except n47's removal curve.

## 16. Earlier decisions

Hers, from before the 2026-11-04 grid chat. Each line is a decision, with her words where
they were recorded.

**Three ways to pair the feature space the model is fitted on with the feature space the detector reads, per model per fold, not four (Adelaide, 2026-10-30).** The feature space the
model is fitted on and the feature space the detector reads are separate:

1. model on the base representation, detector in the base: the reference;
2. model on the base, detector in the hybrid: against 1, does the hybrid make a better
   detector;
3. model on the hybrid, detector in the hybrid: the best achievable; against 1, does the
   hybrid improve prediction under noise, cleaning included.

The fourth, model on the hybrid and detector in the base, is dropped. The main table holds the first two (model on the base representation, detector in the base or in the hybrid), one row per cleanup; the third (model and detector both in the hybrid) is one extra column. A cleanup hands back labels, weights
or a keep mask, so a cleaned set made in the hybrid space trains the base-representation model
as it is. The noise-aware training experiment keeps its own feature sets, searched on each
condition's damaged labels, because a hybrid searched on clean labels and used under noise
would be leakage.

**The names (Adelaide, 2026-10-31).** The five experiments are `n43_noise_aware_training`,
`n44_detect_and_clean`, `n45_detector_grading`, `n46_structure_specific_fixes`,
`n47_noise_measurements`. New numbers start at n43 so that n01 to n42 keep meaning the old
files the thesis and the outline cite. n15, n20 and n36 keep their numbers.

**The old files stay (Adelaide, 2026-11-02).** They stay where they are, out of
`SINGLE_DATASET_EXPERIMENTS`. The new files import code from them, so moving them to an
archive folder breaks those imports.

**Jo-SRC's clean test for regression: struck (Adelaide, 2026-11-02).** Nothing needs that
answer. Jo-SRC as published was cut on 2026-10-30 after both regression forms of its clean
test lost to the plain DNN under noise, so no method waits on a choice here. `josrc_weighting`,
`net_josrc_test` and `ema_rewrite` stay and do not use it.

**Three filters join the detector list (Adelaide, 2026-10-30).** Registry names
`neighbor_label_spread`, `label_zscore` and `feature_outlier` in `noise_estimation.py`. The
originals stay in `noise_mitigation_methods.py` because fed01 and
`representations/molecular.py` use them.

**Two decisions of 2026-10-31 on the size of the detector-by-cleanup sweep.** The flag sets
built from the injected noise at a set accuracy answer one question, how accurate a detector
must be for deletion to help, and that result is reported first in the paper, before any
detector. They read no features, so they run in one detector space rather than two. And the
`rounds_delete` cleanup is cut from that sweep. It deletes the flagged molecules, re-scores the
kept molecules with the detector, and repeats. From the stored n32 results, n32's form of it
raised the top score in 10 to 18 of 37 combinations of dataset, noise condition and level.
Deleting the same number at random raised it in 14 to 22. Under outliers and shifts, n32's form
did so in 4 of 6 and 2 of 6, against 6 of 6 for deleting once. Multistage cleaning stays whole in the noise-aware
training experiment. The branch remains and runs only when named.

**QM9 plays two parts (Adelaide, 2026-11-01).** Her words: "In some sense, QM9 will function
as any other dataset. In others, it will be used as a comparison point."

**Metrics (Adelaide, 2026-11-01).** RMSE stays the default for regression and AUC for hERG
blocking. chembl_qsar moves from MAE to RMSE. Every fit also saves MAE, R$^2$, Spearman and
Pearson, from the same held-out predictions, so no extra fit; on a classification dataset AUC,
accuracy, Matthews correlation and weighted F1. Built in `d_vs_dprime.py` (`ALL_METRICS`,
result fields `all_D`, `all_Dprime`, `all_clean`). Every experiment that writes its results through
`n07.dprime_rows` or `score_d_dprime` gets them. The old experiments that score with their own
`_fit_score` still save one metric, and are being replaced.
