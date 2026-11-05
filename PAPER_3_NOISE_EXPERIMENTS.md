# Paper 3 noise experiments: the grid

The record of what Adelaide decided about the noise grid, and what is still open.
Written 2026-11-04 from the chat that followed `HANDOFF_PAPER3_GRID_SIZE.md`. Adelaide
decides everything here. The code for some of it was written on 2026-11-04 and is being tested
(section 15). `HANDOFF_PAPER3_NOISE_IMPLEMENTATION.md` is the chat that builds this.

How to read it:
- **Decided** means Adelaide said so, and her words are quoted.
- **Counted** means computed from the code. A count is a grid size, not a result.
- **Claude's reading** means not reviewed by Adelaide and not a finding.
- No number in this file is a scientific result, unless it names the thesis table it
  came from.

## 1. Decisions (Adelaide, 2026-11-04)

Sections 1 and 2 were decided while looking at n44. The noise kind cuts apply to n43, n44, n45,
n46 and n47 (Adelaide, 2026-11-04: "Yes cut the noise from all"). `base_2` is cut from all five (Adelaide, 2026-11-04: "okay then the cut to all", answering
Claude's recommendation to cut the second base representation from all five). n43's and n47's
other hybrids are not covered by that cut and are open (section 3).

| What | Decision | Her words |
|---|---|---|
| Put-back and twins | Results first, then the checks only for what she chooses (wording below) | "First the results, then the checks for what I need" |
| Model inputs | `base_1` and `hybrid`; `base_2` goes | "Lets do this: base_1 and hybrid only" |
| Draws | Stay at 3 per non-zero level (1 under censoring) | "Don't modify the draws thats legit" |
| Repeated runs | The same thing must not run more than once | "just make sure the same thing doesn't get run multiple times. This should have never been the case" |
| Time flags | Add them | "Yes add time flags" |
| Time limit | None set; record every time | "yea just record the times of everything" |
| Noise kinds | Cut `dense_oofmse`, `hetero_hidden` and `sparse_dprime_up_p10`; keep `hetero_seen` | "Keep hetero_seen" |
| `hetero_seen` levels | Keeps all its levels, 0, 0.1, 0.3, 1.0 | "okay I like that, do that", answering Claude's "Keep all `hetero_seen` levels." |
| Documents first | This file, then a separate chat implements it | "everything we're talking about first needs to get documented, then a separate chat is going to handle the actual implementation" |

**Put-back and twins, the wording she approved:** n44 first runs every cleanup with no
put-back and no twins, and those plain results go through the normal comparison. Then a
second run adds cliff put-back with random put-back beside it, plus the random and shuffled
twins, only for the results she chooses.

**`base_2` changes an earlier rule.** On 2026-09-25 she decided that noise experiments
compare against each model's top 2 single representations. `base_1` and `hybrid` only is
the change to that rule.

**`sparse_dprime_up_p10` changes an earlier decision.** It was one of the six kinds decided
on 2026-10-05, and the thesis has results on it. Those results stay in the thesis.

**Unchanged:** 5 folds (Adelaide, 2026-10-30) and the dataset's two decided models.

## 2. The noise kinds

What remains: `gaussian`, `outlier_p05`, `grouped_shifted`, `censoring`, `shift_only_p10`,
`hetero_seen`.

The "Assay source" column is **Claude's reading** of thesis Table `tab:noise_sources`, for
her to check.

| Kind | Status | What it does | Assay source | What the thesis has |
|---|---|---|---|---|
| `gaussian` | Keep | Small independent error on every label | Variation between repeats: seeding density, passage number | Noise shape changed little in Chapter 3; Huber blend helped hERG Ki under it |
| `dense_oofmse` | **Cut** | Gaussian again, sized by model error | Same | None. Thesis says only how the injector works |
| `outlier_p05` | Keep | 5% of molecules get error 3 times wider; the rest still move | Gross errors, liquid handling | Delete flagged helped on LogD |
| `hetero_hidden` | **Cut** | Random error width per molecule | Same idea | None |
| `shift_only_p10` | Keep | 10% of labels moved up, in label SDs; the rest untouched | One-direction bias, such as autofluorescence | Cliff-aware gap detector did best; two-learner deletion made the model worse on hERG Ki and Caco-2 |
| `sparse_dprime_up_p10` | **Cut** | The same move, in SDs of model error | Same | Thesis Table `tab:noise:kinds`: "As above, in a different unit" |
| `grouped_shifted` | Keep | Each scaffold family shares an offset | Differences between laboratories or sources | Hardest condition in Chapter 3; scaffold-family offset helped under injected damage, found nothing on the labels as delivered |
| `censoring` | Keep | Labels clipped at a limit | Assay detection limits | Censored EM won back much of the loss with no detector |
| `hetero_seen` | Keep | Error grows with heavy-atom count | Weak fit. No source for heavy atoms found in the thesis | None. The only kind where error size follows the molecule |

Why each was cut, as discussed: `dense_oofmse` is the Gaussian pattern at a smaller dose.
`hetero_hidden` is close to `outlier_p05`, since both give molecules random error widths.
`sparse_dprime_up_p10` is the same move as `shift_only_p10` in a different unit.
`dense_oofmse`, `hetero_seen` and `hetero_hidden` were added on 2026-11-02 so that noise
sized by label spread could be compared with noise sized by model error. With two of the
three gone, that comparison now rests on `hetero_seen`.

**Assay noise no kind covers (Claude's reading):**
- Plate-position error (edge, row, column effects).
- A label attached to the wrong molecule. `shift_only_p10` only moves labels up, so it
  does not match this.

**Censored EM and the scaffold-family offset are not in n44.** They are the fixes the
thesis results above rest on, and they live in n46. n44 only runs its general
detectors and cleanups under those two kinds.

## 3. Still open

### Noise results review: the largest remaining cut
Adelaide: "I am not sure that I want to use all the detectors, this needs to be looked over
based on preliminary results. Everything in here needs to be reviewed more thoroughly...
Lets do that review later, its important so it needs a lot of work."
Also: "those should be massive cuts. Right now I'm throwing everything at the kitchen sink
for noise."

Adelaide: it should look "at more than just detectors, but everything in that noise mitigation
grid". So it covers everything inside a piece (section 5): detectors, cleanups, all their settings,
detector feature spaces, neighbour settings, which detectors pair with which cleanups, and
`gp_smooth`. It has not started. **Every size in section 4 is before this review.**

**On hold (Adelaide, 2026-11-05):** "hold off on the grid review, I'm waiting for ACs." It goes in a
handoff that combines these decisions with the cliff noise grid.

### The cliff grid
Adelaide: "it gets worse when I factor in ACs... Lets just tackle this first, then we'll get
to the cliff grid." Comes next, after this document. Related, in `HANDOFF_AC_STUDY.md`:
- The second run's cliff put-back needs a cliff definition.
- n44 today reaches one fixed definition (0.55 Tanimoto, multiple 2.0), per that handoff.
- Its `cliff_aware_cleanup` waits on the best detector and best cleanup from these noise
  results.

### Smaller questions, one per line
- Whether n43 keeps `hybrid__general2` and `weighted_hybrid`, and n47 keeps `hybrid__general2`.
  The noise kind cuts and the `base_2` cut are decided for all five (section 1).
- Whether Caco-2 and other full-grid datasets keep 4 noise levels.
- Which datasets censoring runs on (`HANDOFF_PAPER3_GRID_SIZE.md` section 3).
- Which results get the second run. She chooses after seeing the first run's scores.
- What to do with `gp_smooth`, which goes to the review.
- Every regression dataset's size. Only LogD and Caco-2 are counted here.

## 4. Grid sizes

One **piece** is one noise kind, level and draw, on one fold, with one model and one model
input. One **result** is one cleaned training set scored for that piece.

**LogD in n44, before the decisions:**

| Grid number | Count | Running total |
|---|---|---|
| Noisy versions of the labels | 46 | 46 |
| Folds | x 5 | 230 |
| Models (dnn, svm) | x 2 | 460 |
| Inputs (`base_1`, `base_2`, `hybrid`) | x 3 | 1,380 pieces |
| Cleaned training sets per piece | x about 4,850 on average | 6.7 million |
| Put-back (none, cliff, random) | x 3 | 20.1 million results |
| Twins, if on | x 3 | 60.2 million results |

**Those 46 versions on LogD.** Level 0 has one draw per kind.

| Kinds | Levels | Draws | Versions |
|---|---|---|---|
| `gaussian`, `outlier_p05`, `grouped_shifted`, `shift_only_p10`, `sparse_dprime_up_p10`, `dense_oofmse` | 0, 1 | 3 | 4 each, 24 total |
| `hetero_seen`, `hetero_hidden` | 0, 0.1, 0.3, 1 | 3 | 10 each, 20 total |
| `censoring` | 0, 0.25 | 1 | 2 |

Caco-2 has 84 versions, because its levels are 0, 0.5, 1, 1.5 (censoring 0, 0.1, 0.25, 0.4).
Table in `HANDOFF_PAPER3_GRID_SIZE.md` section 2.

**After the section 1 decisions, first run only.** Level 0 once, three kinds cut, `base_1` and
`hybrid`, no put-back and no twins. The hours divide results by the speed in section 7, on one
8-core session.

| Dataset | `hetero_seen` levels | Versions | Pieces | Results | Hours | Days |
|---|---|---|---|---|---|---|
| LogD | Current (0, 0.1, 0.3, 1) | 23 | 460 | 2.1 million | about 2,800 | about 117 |
| LogD | Level 1 only | 17 | 340 | 1.6 million | about 2,100 | about 87 |
| Caco-2 | Current (0, 0.1, 0.3, 1) | 49 | 980 | 4.5 million | about 6,100 | about 252 |
| Caco-2 | Level 1 only | 43 | 860 | 4.0 million | about 5,300 | about 223 |

For comparison, before the decisions: LogD 20.1 million results, Caco-2 37.1 million.

What these hours leave out:
- The second run of checks.
- The results review, which should shrink them a lot.
- The per-fold searches. Fold 0's search on LogD did not finish in 2 hours on the laptop.
- Other regression datasets, and n43, n45, n46 and n47.

The Caco-2 hours use the LogD speed. Caco-2 has fewer molecules and is likely faster, not
measured. Sessions run at once divide the hours, if the server gives that many.

**The arithmetic.** A base-input piece has 5,844 results, or 5,364 without oracle flags. A
hybrid-input piece has 3,977, or 3,497 without. Oracle flags exist only at non-zero levels of
`outlier_p05`, `censoring` and `shift_only_p10` (and `sparse_dprime_up_p10` before its cut).
So one version with `base_1` and `hybrid` is 5 folds x 2 models x (5,364 + 3,497) = 88,610
results, or 98,210 with oracle flags. LogD current: 16 x 88,610 + 7 x 98,210 = 2.1 million.

## 5. Inside one base-input piece

This is what the results review has to go through. Counted by running `n44.run_unit` with
fitting stubbed out. The counter gives exactly 4,032 when set up like `_smoke_n44.py`.

**5,844 cleaned training sets, by detector:**

| Detectors | Count | Cleaned training sets |
|---|---|---|
| Detectors that give a score | 32, about 129 each | 4,118 |
| Neighbour detectors | 4, 165 each | 660 |
| `oracle`, only where the noisy molecules are known | 15 settings | 480 |
| `conformal_bh` | 6 settings | 384 |
| `two_learner_small_loss` | 3 settings | 192 |
| No detector | n/a | 10 |

The 32: `cross_val`, `ensemble`, `leave_one_out`, `hybrid`, `gaussian_process`, `hetero_gp`,
`heteroscedastic_nn`, `conformal`, `deep_ensemble`, `bayesian_rf`, `quantile_regression`,
`quantile_forest`, `ngboost`, `neighbor_label_spread`, `label_zscore`, `feature_outlier`,
`net_small_loss`, `net_loss_mixture`, `net_josrc_test`, `net_meta_weight`,
`net_evidential_spread`, `net_trusted_weight`, `oof_residual`, `ngboost_rescaled`,
`quantile_forest_rescaled`, `conformal_rescaled`, `knn_shapley`, `cleanlab`,
`isolation_forest`, `lof`, `residual_low_disagreement`, `random`.

The 4 neighbour detectors: `neighbor_residual`, `local_variance_ratio`, `distance_weighted`,
`activity_cliff_aware`.

A hybrid-input piece has 3,977, because it scores detectors in one feature space rather than
two.

**The same 5,844, by cleanup, with the number of settings each has:**

| Cleanup | Settings | Results | Cleanup | Settings | Results |
|---|---|---|---|---|---|
| `exp_downweight` | 4 | 436 | `network_rewrite` | 3 | 261 |
| `reliability_weight` | 4 | 388 | `ema_rewrite` | 3 | 261 |
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
| `huber_teacher_blend` | 1 | 1 | `prox_smooth` | 1 | 0 here, runs only on a forest neighbour setting |

n44 has 26 cleanups, not the 23 in the earlier handoff; the smoke test only reached 23.
`label_smooth`, `knn_smooth` and `reliability_weight` also repeat under the neighbour settings
kept for that piece (3 in this count).

## 6. Repeats and results that do no work

**Rule from Adelaide:** the same thing must not run more than once. Known cases, and the
implementation chat must look for others:

**Level 0.** Level 0 returns the labels as delivered for every kind, so all 9 level-0 versions
had identical labels. Each kind still ran them and wrote them to its own file. On LogD that is
8 repeated versions, or 3.4 million of the 20.1 million results. n44's level-0 rows are the
comparison for the kind, so each kind must be able to read the one set.

**`reliability_weight`.** Results are labelled by detector, but the cleanup never reads that
detector's scores. 32 detectors x 4 strengths x 3 neighbour settings gives 384 results, with
12 different training sets. The other 4 of its 388 have no neighbour list and produce nothing.

**Twins.** `controls.py` says the random and shuffled twins are the same when a method changes
every molecule, as `downweight` and proportional `label_smooth` do. They differ only when some
molecules are left alone.

**Repeats reuse a saved fit, but still cost a result.** `n07.dprime_rows` says every fit goes
through `noise_cache.cached_fit_predict`, and its key is the content of the features, labels
and weights, not the noise kind. So an identical training set should not be refitted, though
this was read in the code and not measured. Removing repeats may therefore save less time than
the result counts suggest.

**`gp_smooth`, a different problem.** The grid schedules 327 results per base-input piece, but
the code refuses above `GP_SMOOTH_MAX_ROWS`, 1,000 training molecules. LogD's fold 0 has 4,022.
Each refusal writes a row with no score and prints a line, which is what filled the server log.
So it fits nothing on LogD, and costs almost no time. It goes to the results review.

## 7. Measurements

**Speed: about 750 results per hour.** On the server, 8 cores and 64 GB, LogD, `outlier_p05`, with
cliff and random put-back on and twins off, using `KIRBY_NOISE_PROMOTE=n44`:

| Time, 2026-11-04 | Results on disk |
|---|---|
| 18:55:47 | 10,935 |
| 19:43:59 | 11,535 |

That is 600 results in 48 minutes, from one part of the first piece. Results arrive in bursts,
and some are cheap (refused `gp_smooth`, reused fits), so the real speed could differ a lot.

**Not yet measured:**
- The time for a whole piece. Run `grep "rows, " /data/stat-ecr/scat9264/n44_logd.log`; no line had
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

Decided: add them. Today, the only time recorded is one total per piece: the `rows, ` line in the
log and `seconds` in the `.units` checkpoint. Nothing times a detector, a cleanup or a model fit.

What was discussed: time each detector's scoring, each cleanup and each model fit; write a line
when one goes over a limit; report only, and drop nothing automatically. The review of section 3
needs this timing data, so the flags should come first. No limit is set: Adelaide, 2026-11-04, "yea just record the times of everything".

## 9. Checks: what they answer

**Cliff put-back:** did cleaning hurt because it touched activity cliff molecules? **Random
put-back:** the comparison for cliff put-back, so it only means something beside it.
**Twins:** a twin is a fake cleanup, the same amount of change on the wrong molecules. If the real
cleanup cannot beat it, the cleanup was not finding bad labels.

Twins default to off: `controls_enabled()` reads `KIRBY_NOISE_CONTROLS`, and neither `setup.sh` nor
the server commands used here set it.

From the thesis, `sec:noise:res:qm9test`: with the DNN, passing the shuffled-score control means
little alone. On QM9's first single representation, Neighbor-mean smoothing beat all three
shuffles with no noise to find. That test shuffled the detector's scores, which is similar to a
twin but not the same: a twin shuffles the cleanup's output.

## 10. What 3 draws means

The same noise kind and level with three different random seeds. The seed depends only on the kind
and the draw (`noise_draw_seed`), so a molecule gets the same noise in every fold, and the same
molecules are chosen at every level of one draw. Draws show how much a result depends on which
molecules were hit. From thesis Table `tab:noise:res:family_offsets` (DNN on the first hybrid,
grouped-shifted, level 1.0, damaged labels, held-out RMSE):

| Dataset | Draw 0 | Draw 1 | Draw 2 |
|---|---|---|---|
| LogD | 0.684 | 0.690 | 0.672 |
| QM9 | 0.909 | 0.844 | 1.295 |

Censoring has one draw because its clipping reads no seed.

## 11. The other experiments, on Caco-2

Counted before the section 1 decisions, and not updated for them.

| Experiment | Pieces | What multiplies inside |
|---|---|---|
| `n43` | 420 (84 versions x 5 folds) | 1 dnn search; 5 feature spaces x 9 network methods = 45 network trainings; 268 results per feature space, or 516 under 1,000 training molecules, when the GP methods run |
| `n45` | 1,920 (64 versions x 2 models x 3 inputs x 5 folds) | 120 fits per piece where noise can be graded; 32 detectors and 420 neighbour grades with no fits |
| `n46` | 1,620 (54 versions), plus 3,240 for the sources pass (108 versions) | Results per piece not counted |
| `n47` | 2,160 (54 versions x 2 models x 4 configurations x 5 folds) | Removal curve: 5 shares x 20 draws x 2 models x 4 configurations x 5 folds = 4,000 fits, not per piece |

## 12. Corrections to earlier documents

- `HANDOFF_PAPER3_GRID_SIZE.md` says 4,032 fits per piece, 10.2 million on Caco-2, 5.6 million on LogD.
  The smoke test behind it has no neighbour settings, no co-teaching and no put-back. The counts are
  5,844 and 3,977 results per piece before put-back, and 37.1 and 20.1 million with it.
- That handoff's section 9 says `KIRBY_NOISE_PROMOTE=1`. That gives the full levels on fold 0 only.
  Five folds need `n44` or `all`.
- `HANDOFF_NOISE_RESTRUCTURE.md` section 9.9 blames the Seagate drive for the 2026-11-02 crashes. On
  2026-11-03, two laptop runs of n44 crashed at the start of fold 0, with two OpenMP libraries
  loaded (`libomp`, `libiomp5`) and a `libiomp5` thread dying inside `libomp`. With threads capped,
  a run lasted 2 hours with no crash, though parts still used about 4 cores. The drive was present
  just after the second crash. So the cause points at the libraries, unproven, and the older
  crashes have not been re-read. The server runs did not crash this way.

## 13. Status of `HANDOFF_PAPER3_GRID_SIZE.md` section 7

| Its question | Status |
|---|---|
| Which datasets censoring runs on | Open |
| Whether 3 draws stays 3 | Decided: stays |
| Whether full-grid datasets keep 4 noise levels | Open |
| Whether cleanups and detectors stay whole | Open: the results review |
| Whether the two heteroscedastic kinds and `dense_oofmse` stay | Decided for n44: `dense_oofmse` and `hetero_hidden` cut, `hetero_seen` kept |

## 14. How the counts were made

Scripts are in `scratchpad/paper3_grid_counts/`. Run from the repo root with
`PYTHONPATH=src:tests:tests/experiments KIRBY_NOISE_PROMOTE=1`:
- `count_n44.py`: results per piece and totals.
- `breakdown.py`, given `count_n44.py`'s path: settings, detectors and cleanups.
- `lv.py`: levels per kind. `count_kinds.py`, `count_n43_n45.py`: the section 11 counts.

## 15. Build status

Written 2026-11-04, not yet committed. Smoke tests running; this section records no result.

- Noise kind cuts in n43 to n47: `noise_helpers.PAPER3_CUT_CONDITIONS`. Other experiments still read the old lists.
- `base_2` cut as a model input in all five. Neighbour lists on datasets with the larger neighbour grid
  still read it (`similarity_space.GROUP_SPACES`). Adelaide has not said whether the cut covers those.
- Time recording: `noise_timing.py`, one log per experiment, dataset and noise kind.
- n44 first run: no put-back, no twins (`n44.SECOND_RUN_CHECKS`).

Not built: shared level-0 results, removal of repeats, the second run.
