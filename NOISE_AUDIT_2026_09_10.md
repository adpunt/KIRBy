# What noise the noise experiments add, and to which molecules

Written 2026-09-10. Sections 6.13, 6.14, 6.15 and 11, and decisions 11 to 15 in §10,
were added 2026-09-11 from a separate session and are marked as such where they start.
Everything in them was read out of the code that day. Some line numbers in the 09-10 body
have since moved as files were edited; the 09-11 additions carry the numbers as they stood
on 09-11.

Every statement below was read out of the code in this session. Nothing here comes from
a memory file or from an earlier document unless the sentence says so. Some numbers
below were measured during this audit rather than read. Those say
"measured here", and they name what they were measured on — either synthetic labels, or
KIRBy's own label files where the sentence gives the path.

Fifteen decisions need a ruling from you. They are in §10. The four that change what an
existing number means are in §1. The largest is number 15: the loaders delete the real
measurement error before any experiment sees it, and §11 is where that is set out.

**Read in full.** `~/repos/NoiseInject/noiseInject/core.py`. `src/kirby/noise_spec.py`.
`tests/experiments/noise/noise_helpers.py`. All ten files in
`tests/experiments/noise/`. The label-noise path in `src/kirby/find_best_hybrid.py`.
`src/kirby/utils/lnl_methods.py` and `src/kirby/utils/noise_mitigation_methods.py`. The
noise section of `tests/experiments/dataset_policy.py`. The eleven files outside
`tests/experiments/noise/` that inject label noise. For §11, added 2026-09-11: the ChEMBL
fetch in `tests/alternative_data_noise_robustness.py`, `src/kirby/datasets/herg.py`,
`src/kirby/datasets/chembl_bioactivity.py` and `src/kirby/datasets/canine_cyp.py`.

**Checked against:** `RERUN_PLAN.md`, `NOISE_DESIGN.md`, `noise_conditions.json` and the
two run books in `~/repos/qsar_qm_models`. That repository ran the same kind of
experiment and recorded every noise defect it hit. Section 1 is where KIRBy departed from
what it settled; all four of those are now fixed, and §12 says what was changed.

---

## 1. Where KIRBy departed from what the sister study settled — ALL FOUR NOW FIXED

`~/repos/qsar_qm_models` ran the same kind of experiment and wrote down what went wrong
each time. I checked KIRBy against every noise ruling in `RERUN_PLAN.md`,
`NOISE_DESIGN.md` and `noise_conditions.json`. Four departures mattered.

**All four were fixed on 2026-09-12.** What each one was, and the measurement that found
it, is kept below because the reasoning is what stops it coming back. What was changed
and what it now measures is in §12. This section describes the state BEFORE that edit.

### 1.1 A molecule's corruption depended on which fold it landed in — FIXED, §12.1

`RERUN_PLAN.md` §3.3a diagnosed this and §3.3b settled it on 2026-09-04. Draw the
corruption **once over the full label column**, keyed by a molecule's position in the
dataset. Each fold then takes its slice. Their reason: a molecule's corruption has to be
a property of the molecule, or rows cannot be joined across folds.

That fix was applied to KIRBy at commit `dd58f4f`. It changed
`tests/alternative_data_noise_robustness.py` and one new test file, and nothing else.
`noise_helpers.inject_train_labels` still draws on each fold's training labels
(`noise_helpers.py:146-177`), and solves the amount against that fold's training spread.

Measured here, 600 molecules, 5 scaffold-disjoint folds, level 0.5:

| condition | molecules carrying more than one corrupted value | amount delivered across the 5 folds |
|---|---|---|
| gaussian | 600 of 600 | 0.4229 to 0.4441 label units, 4.9% apart |
| outlier_p10 | 600 of 600 | 0.3950 to 0.4695, 17.1% apart |
| grouped_wider | 600 of 600 | 0.3752 to 0.4670, 21.1% apart |
| grouped_shifted | 600 of 600 | 0.3780 to 0.4822, 23.9% apart |

Drawing once over the whole column gives 0 of 600 in every condition.

The gate that catches this exists: `tests/test_noise_is_a_property_of_the_molecule.py`,
written for `dd58f4f`. It tests `noise_column.py`. It does not import anything from
`tests/experiments/noise/`.

**What it costs.** Two things. A recovery number averaged over folds is fine, because the
delivered amount is close on average. What is not fine is joining a molecule across
folds — under the per-fold draw that is averaging five different corruptions. The second
is sharper. The hybrid search draws once over the full column
(`find_best_hybrid.py:7756`) while the experiments draw per fold. So a recipe fitted
under one corruption is used by an experiment running a different one.

### 1.2 The check on the amount delivered could not reach you — FIXED, §12.2

`noiseInject` warns when a draw lands outside the band its own tolerance allows
(`core.py:648-660`). Its comment says what the check is for. The defect it would have
caught was a fixed scale error present on every single draw, 29 to 51% depending on the
shape.

`src/kirby/find_best_hybrid.py:74` calls `warnings.filterwarnings('ignore')` at module
scope. Every noise experiment imports that module. So by the time an injection runs,
every warning in the process is discarded. `src/kirby/utils/noise_estimation.py:39` and
`src/kirby/utils/noise_mitigation_methods.py:25` do the same.

Measured here: after importing the noise suite the first warning filter is
`('ignore', None, Warning, None, 0)`. An injection asked for 0.5 label units and made to
deliver 1.5016 — 200% over — printed nothing at all.

Nothing in `tests/experiments/noise/` reads `realised_dose_label_units` back and compares
it to what was asked for. The amount is written onto the row and never checked. That is
the same shape as the sister study's own finding: one injector recorded the delivered
amount and never read it back, so a scale error survived on every draw.

### 1.3 The labels already carry error, and nothing recorded it — FIXED, §12.3

`NOISE_DESIGN.md` §4d, raised by you on 2026-08-27. The labels start with measurement
error of their own. Noise adds in quadrature, so injecting level L on labels already
carrying `b` of their own spread gives a total of `sqrt(b² + L²)`.

Measured on KIRBy's own label files. hERG Ki is `~/.cache/kirby/chembl_herg_ki.csv`,
n = 1415. LogD is the `LogD` column of `~/.cache/kirby/openadmet_train.csv`, n = 5039.
The published errors are the ones already in `dataset_policy.py`.

| | clean label spread | published error | already there, as a fraction of the spread |
|---|---|---|---|
| logd | 1.1905 log units | 0.48 log units | 0.403 |
| herg_ki | 0.9143 pKi units | 0.54 pKi units | 0.591 |

Total noise at each level of the shared grid, in fractions of the clean label spread:

| level asked for | 0.0 | 0.15 | 0.3 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|
| logd, total | 0.403 | 0.430 | 0.503 | 0.642 | 0.852 | 1.078 |
| herg_ki, total | 0.591 | 0.609 | 0.662 | 0.774 | 0.955 | 1.161 |

Two things follow. The level-0 row is not a clean control — on herg_ki it is a run at
0.591. And the bottom of the grid barely moves the total: on herg_ki, going from level
0.0 to level 0.15 moves it from 0.591 to 0.609, a change of 3%.

`REFERENCE_ERROR_LABEL_UNITS` and `reference_error_as_fraction_of_spread` are both in
`dataset_policy.py:462-481`. Grepping the whole repository, neither is called by
anything.

The same arithmetic caps how well any model can do. A model cannot explain the part of
the variance that is measurement error, so R² is capped at `1 − b²`: 0.837 on logd and
0.651 on herg_ki.

### 1.4 A second noise condition had nowhere to write its result — FIXED, §12.4

`n09`, `n11`, `n12` and `n13` each name one condition and say in a comment that the
others are reachable by passing `noise_condition=`. No caller passes one. This is why.

`save_experiment_results` writes to `<output_dir>/<experiment>/<dataset>.json`
(`_shared.py:6016-6019`). The condition is not in the path. The resume check compares an
input fingerprint, and that fingerprint only records a noise grid when the config carries
`noise_levels`, `strategies` or `noise_seeds` (`_shared.py:5936-5960`). The config
`noise_helpers.run_all` builds carries none of them (`noise_helpers.py:246-254`), because
the condition is read from keyword arguments rather than from the config.

So running `n09` under a second condition either overwrites the first run's file or is
skipped as already done, and nothing in either file says which condition produced it.

---

## 2. What is right, and the whole directory on one page

**No held-out label is corrupted anywhere.** I checked every line where a metric is
called. That covers all ten files in `tests/experiments/noise/`. It also covers the four
noise-injecting files outside that directory which something still calls. Every
experiment corrupts training labels only. Every one grades against labels that were
never touched.

**There is one injector, and the noise suite is on it.** `src/kirby/noise_spec.py`
wraps `noiseInject`. The five private noise functions the suite used to carry are gone.
The copy of `noiseInject` vendored at `src/noise_inject/` is byte-identical to
`~/repos/NoiseInject/`, and the installed package resolves to the second one, so the
two cannot have drifted.

What is wrong is narrower. It is about **which** noise, at **which** level, and whether
the question each experiment asks can be answered under the noise it actually adds.

Four of the ten files in the noise directory add no noise at all. Six do.

One row of the table below is one experiment file. "Molecules corrupted" counts training
molecules only, out of the training molecules in a fold. "Levels" are fractions of the
clean label standard deviation.

| file | adds noise? | condition | levels | molecules corrupted | graded against |
|---|---|---|---|---|---|
| n01 activity cliffs | no | — | — | none | untouched held-out labels |
| n02 sample weighting | no | — | — | none | untouched held-out labels |
| n06 BindingDB | no — uses real disagreement between laboratories | — | — | none | no prediction score; a rank correlation |
| n07 remediation | yes | outlier contamination, 10 molecules in every 100 | one, 0.5 of the label spread | 10 in every 100 | untouched held-out labels |
| n09 method survey | yes | Gaussian only | 0, 0.15, 0.3, 0.5, 0.75, 1.0 of the label spread | every training molecule | untouched held-out labels |
| n11 detect-and-repair grid | yes | Gaussian only | the same six | every training molecule | untouched held-out labels |
| n12 weighted distance | yes | Gaussian only | the same six | every training molecule | untouched held-out labels |
| n13 error shapes | yes | ten shapes | the same six | varies by shape | untouched held-out labels |
| n14 cliff landscape | no | — | — | none | out-of-fold residuals |
| n15 clean vs noisy recipe | yes | Gaussian only | 0.0, 0.15, 0.3 of the label spread | every training molecule | untouched held-out labels |

Three of the six injecting experiments run Gaussian and nothing else. A fourth runs one
outlier setting and nothing else. Only `n13` sweeps the shape of the error.

Added 2026-09-11: none of the ten files reports a detection score. The list of molecules
whose label moved is returned on every draw and discarded on all seven lines that receive
it — §6.13. And three of the six fit one model rather than the roster — §6.14.

---

## 3. The machinery every noise experiment shares

### 3.1 One place turns a setting into corrupted labels

`src/kirby/noise_spec.py`. Two experiments call it directly — `n15` and the hybrid
search. The other four reach it through `noise_helpers.inject_train_labels`.

A fresh injector is built on every single call. `noiseInject`'s random generator
advances on every draw, so re-using one injector across a sweep would make each level's
corruption depend on how many levels ran before it
(`src/kirby/noise_spec.py:16-20`). I checked every path; none reuses one.

### 3.2 What a level means

For ten of the eleven regression conditions, the level is the **root-mean-square error
actually delivered**, written as a fraction of the clean labels' standard deviation.

A level of 0.5 on logD means the added error has a root-mean-square of half a standard
deviation of logD. Each condition solves for its own internal width so that the amount
delivered matches the level. That is what makes a difference between two conditions at
one level a difference of *shape* rather than a difference of *amount*
(`~/repos/NoiseInject/noiseInject/core.py:5-14`).

**Censoring is the exception.** Its level is the **fraction of labels clipped** at an
assay ceiling. It is not centred on the true value and its amount cannot be matched to
the others, so it has its own axis (`src/kirby/noise_spec.py:46-47`).

**On a classification dataset the level is a flip rate** — the fraction of training
molecules whose class label is changed.

### 3.3 The eleven conditions

From `src/kirby/noise_spec.py:96-113`.

**Same width for every molecule.** `gaussian`, `student_t_nu10`, `student_t_nu5`,
`student_t_nu3`, `laplace`. Every training molecule is perturbed. They differ only in
how often a badly wrong label turns up.

**A few molecules hit much harder.** `outlier_p01`, `outlier_p05`, `outlier_p10`. A
random 1, 5 or 10 molecules in every 100 get an error three times wider than the rest.
Which molecules are hit is drawn at random, not chosen by label value.

**Whole scaffold families hit.** `grouped_wider` — about 20 molecules in every 100 get
an error three times wider, and they are chosen as whole scaffold families.
`grouped_shifted` — every family's labels are pushed in one direction by a constant
offset, plus a per-molecule error on top.

**One-sided.** `censoring`. Every label past a ceiling is recorded as the ceiling.

**On a classification dataset** there are six flip models instead: `uniform`,
`class_imbalance`, `binary_asymmetric`, `instance_noise`, `class_dependent`,
`confusion_directed` (`src/kirby/noise_spec.py:115-116`).

### 3.4 The two level grids

`tests/experiments/dataset_policy.py:376` — one grid, every dataset:

    0.0, 0.15, 0.3, 0.5, 0.75, 1.0      fractions of the clean label standard deviation

`tests/experiments/dataset_policy.py:382` — censoring only:

    0.0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50      fraction of labels clipped

`tests/experiments/dataset_policy.py:472-480` — where an experiment needs one number
rather than a sweep, the answer is 0.5 of the label spread, for every dataset.

The per-dataset maps `_NOISE_DOSE_UNITS` and `_NOISE_LEVELS` are both empty
(`dataset_policy.py:369-372`). So every dataset takes the shared grid, and every dataset
reads its level as a fraction of its own label spread.

The published measurement errors — 0.54 pKi units on herg_ki, 0.35 log10 on caco2, 0.48
log units on logd — sit in `REFERENCE_ERROR_LABEL_UNITS` as an anchor for the Methods.
They are not grid entries.

### 3.5 Scaffold groups for the two grouped conditions

`grouped_wider` and `grouped_shifted` each need one integer group number per molecule.
It comes from `assign_noise_groups` (`src/kirby/noise_spec.py:276-313`), reached through
`dataset_policy.noise_groups_for`.

That grouping is deliberately separate from the one that cuts the folds. A Murcko
scaffold is the empty string for a molecule with no ring. So every acyclic molecule
would otherwise land in one group. A single offset draw would then move all of them at
once. Each acyclic molecule is therefore its own group.

Where a dataset has no SMILES — antibody, nucleotide, expression — the two grouped
conditions are dropped rather than run over invented groups. `n13` does this at
`n13_noise_distributions.py:263-270`.

### 3.6 The amount was solved against two different sets of labels — FIXED, §12.1

`src/kirby/noise_spec.py:320-330` says the amount is solved **once over the full label
column**. Its stated reason: that is what lets a cached search be reproduced fold by
fold.

`tests/experiments/noise/noise_helpers.py:155-157` says the amount is solved against
**this fold's clean training labels**, and names the Rust half of the study as the thing
it matches.

Both descriptions are accurate about their own code. `n15` and the hybrid search take
the first route. `n07`, `n09`, `n11`, `n12` and `n13` take the second. So level 0.10 is
0.10 of one standard deviation in five experiments and 0.10 of a slightly different
standard deviation in the sixth.

Which of the two is right was settled. `RERUN_PLAN.md` §3.3b, 2026-09-04: the amount has
to be solved against the full column's clean label spread, because otherwise one
realisation is scaled five different ways.

**FIXED 2026-09-12 (§12.1).** Every experiment in the directory now resolves the amount
against the full column, so there is one rule, not two. The paragraphs above describe the
state before that edit.

---

## 4. The six experiments that add noise

### n07 — which cleanup recovers the most

**What it asks.** Four cleanups are compared against doing nothing, and against clean
labels. Drop the flagged molecules. Weight them down. Pull their labels part-way toward
their neighbours' mean. Replace their labels with that mean outright.

**Which noise.** One condition, written into the file: outlier contamination with the
contaminated share set to 0.1 (`n07_remediation.py:110-112`). So 10 training molecules
in every 100 get an error three times wider than the other 90. On a classification
dataset it is symmetric class flips instead, at a flip rate of 0.1. There is no shape
sweep.

**Which level.** One level, 0.5 of the clean label spread
(`n07_remediation.py:297`). There is no level sweep.

**What happens to the molecules.** They are split into five folds by scaffold, so no
scaffold sits on both sides of a split. The held-out fifth is set aside and is only ever
graded. Inside the training molecules, a tenth are chosen at random and given the wider
error. The width is solved so that the root-mean-square error across all training labels
equals half the clean label standard deviation. That is done three times per fold, with
three different draws.

For every draw, the recipe is re-applied to that draw's corrupted training labels, to
pick a fresh set of features. Every training molecule is then given a suspicion score.
That score is how far its label sits from its ten nearest neighbours' labels. Six models
are then fitted: on clean labels, on corrupted labels untouched, and after each of the
four cleanups.

**What comes out.** The share of the clean-to-corrupted gap each cleanup put back, in
percent, per model.

**What disagrees.**

1. **The results file names two different conditions.** `noise_condition` is written as
   `'outlier'` at `n07_remediation.py:431`. Twelve lines below, `noise_injection` is
   written as `'gaussian_label_noise_train_only'` at `n07_remediation.py:435-437`. The
   injector delivers outlier contamination. One run, two names, and one of them is
   wrong.

2. **The flag rate and the corrupted share are different numbers.** `noise_fraction`
   defaults to 0.1, so 10 training molecules in every 100 are corrupted
   (`n07_remediation.py:291`). The `remove` cleanup drops everything above the 80th
   percentile of the suspicion score, so 20 training molecules in every 100 are dropped
   (`n07_remediation.py:197-198`). The flag rate is fixed and does not follow the
   corrupted share. At least 10 uncorrupted molecules in every 100 are discarded by
   construction, whatever the detector found.

3. **The comment above the level and the level itself are in different units.** The
   comment names published measurement errors in log units — 0.54 on herg_ki, 0.35 on
   caco2 (`n07_remediation.py:292-296`). The code takes 0.5 of the label spread, for
   every dataset (`dataset_policy.py:472-480`). Those two figures cannot be the same
   number. The comment describes a per-dataset grid `dataset_policy` no longer has.

4. **The clean ceiling and the corrupted runs are measured in different feature sets.**
   The clean run's features were chosen from clean labels. Every corrupted run's
   features were chosen from that draw's corrupted labels (`n07_remediation.py:563`). So
   the recovered fraction divides a difference measured in one feature set by a
   difference measured partly in another.

5. **On a classification dataset the file flips a tenth of the labels and records the
   level as 0.5 of the label spread.** Those are different quantities. Two of the four cleanups
   also collapse into one computation there. Pulling a class label part-way toward its
   neighbours' mean, and replacing it with that mean outright, give the same answer once
   the value is rounded back to a class.

---

### n09 — do the noisy-label learning methods re-rank when the features change

**What it asks.** Twelve methods for training on labels known to be wrong are ranked in
three feature sets. The first is one fixed representation. The second is a hybrid
searched inside the fold on the corrupted labels. The third is that same hybrid with
each feature scaled by its importance.

**Which noise.** Gaussian only (`n09_lnl_survey.py:95`). The comment beside it says the
other ten are reachable by passing a condition name. No caller passes one — `run_all`
calls `mod.run(data, config, output_dir, dataset_name)` with no keyword arguments
(`noise_helpers.py:487`). So every run this experiment has ever had was Gaussian.

**Which levels.** The shared six (`n09_lnl_survey.py:151-153`).

**What happens to the molecules.** They are split by scaffold. Inside each fold only the
training labels are corrupted. Three feature sets are built over those same training
molecules. The second and third come from a search that runs entirely inside the
training molecules and sees only the corrupted labels. Each feature set is fitted once
with a plain network and once with each of the twelve methods. Every fitted model then
predicts the held-out molecules.

**What comes out.** A score per level per feature set. The signed change from the clean
run. Which method leads each feature set at the strongest noise. Whether the top three
methods re-order when the feature set changes.

**What disagrees.**

1. **The signed change from clean is not that method's own change.** It is computed as
   the method's score at this level minus the **plain network's** score at level 0, in
   the same feature set. The method-carrying runs do not run at level 0 at all
   (`n09_lnl_survey.py:422`). So one number mixes "this method against no method" with
   "corrupted against clean", and nothing written to the file separates them.

2. **The two sides of the comparison stop training under different rules.** The plain
   network holds back a fifth of the training molecules chosen at random. The twelve
   methods hold back a fifth chosen so that no scaffold sits on both sides.

3. **On a classification dataset the file records the condition as `'gaussian'`** while
   the injector runs symmetric class flips, and the flip rate runs up to 1.0.

---

### n11 — which detector and which repair recovers the most

**What it asks.** Roughly fourteen detectors, in about a dozen feature sets, crossed
with seven repair rules at four strengths each.

**Which noise.** Gaussian only (`n11_estimation_remediation_grid.py:74`). No caller
passes anything else.

**Which levels.** The shared six (`n11_estimation_remediation_grid.py:377`). Each level
is drawn three separate times.

**What happens to the molecules.** They are split by scaffold. Inside each fold the
training labels are corrupted and the held-out labels are left as measured. A random
forest is trained on the fingerprint features with the clean training labels and graded
on the held-out molecules. A second is trained with the corrupted labels and graded the
same way. The distance between those two numbers is how much damage the noise did.

Then, in each feature set in turn, every detector scores every training molecule for how
out of line its label looks against its neighbours. Four strengths and seven repair
rules turn those scores into either reduced training weights or shifted training labels.
A random forest is then refitted **on the fingerprint features every time**, whichever
feature set the suspicion scores came from. That is deliberate. It means the
only thing changing across feature sets is *where the noise was detected*.

**What comes out.** Per level, the share of the clean-to-corrupted damage each
detector-and-repair pairing put back, in percent, averaged over folds and draws. Beside
it, the same share where the winning pairing is chosen on the other four folds and read
on the fold that did not choose it.

**What disagrees.**

1. **The summary reader stops on the markers the same file writes.** On a classification
   dataset the grid writes `_smoothing_skipped_clf = True` into its own results
   (`n11_estimation_remediation_grid.py:345-346`), and a failure writes `_errors` as a
   list (`:619-620`). Both are forwarded on as they are into the structure the summary
   walks (`:665-673`). The print loop skips any key starting with an underscore and any
   value that is not a dictionary (`:705-708`). The summary loop does neither — it calls
   `.get('recovery', nan)` on every value it finds (`:847-848`). Two ways they differ:
   one loop filters those entries and the other does not, and the result is a raised
   error rather than a skipped combination. On a classification dataset the boolean is
   written for every estimator. So no classification result file is written at all: the
   save sits after the run call inside the same `try` (`noise_helpers.py:491-499`).

2. **On a classification dataset the six levels are read as flip rates.** Three of the
   six sit at or past the point where nothing is learnable.

3. **The hybrid entered the comparison with a recipe chosen on clean labels.** The
   single representation it is compared against had no search of any kind. The file
   carries an oracle warning on its own output saying so
   (`n11_estimation_remediation_grid.py:933`).

---

### n12 — does scaling features by importance improve neighbour-based detection

**What it asks.** Whether multiplying each feature by the square root of its importance,
before measuring nearest-neighbour distance, improves detection of corrupted training
labels. And whether that scaling is redundant once the features come from a hybrid.

**Which noise.** Gaussian only (`n12_importance_weighted_distance.py:61`).

**Which levels.** The shared six (`n12_importance_weighted_distance.py:311-313`).

**What happens to the molecules.** They are split by scaffold. Inside each fold the
training labels are corrupted. Five feature sets are built from the corrupted training
molecules. The control representation as it is. The control scaled by importance. A
hybrid rebuilt under the corrupted labels. That hybrid scaled two ways. In each set,
every training molecule is scored for how far its label sits from its k nearest
neighbours. That is done at k of 5, 10 and 20, by three estimators. Those scores become training
weights. A random forest is fitted on the control representation with those weights and
the corrupted labels, then graded on the untouched held-out labels.

**What disagrees.**

1. **Features are scaled by the square root of importance and then compared under a
   Manhattan distance.** The square-root convention is right for the Euclidean family
   only. Under Manhattan a feature's weight in the distance is its multiplier, not the
   square of it. So the scaling applied is not the scaling the question names.

2. **Level 0.0 is in the swept grid, and it produces nothing readable.** At level 0.0
   the clean fit and the corrupted fit are the same number, so the recovered share
   divides by zero (`n12_importance_weighted_distance.py:473`). One of the six levels
   returns not-a-number in every recovery figure.

3. **The recipe for every hybrid run comes from a search over the clean labels of every
   molecule**, including each fold's held-out molecules
   (`n12_importance_weighted_distance.py:341-347`).

---

### n13 — does the shape of the error change which method wins

**What it asks.** Whether a ranking of methods measured under Gaussian error transfers
to the other shapes, and whether a hybrid keeps its advantage across every shape.

**Which noise.** All ten conditions whose level is an amount, on a regression dataset;
all six flip models on a classification dataset
(`n13_noise_distributions.py:259-262`). **Censoring is not in the sweep**, because the
list used excludes it — censoring's level is a fraction of labels clipped and cannot
share the amount axis.

**Which levels.** The shared six (`n13_noise_distributions.py:275-277`), each drawn
three separate times. On a regression dataset that is 10 shapes × 6 levels × 5 folds ×
3 draws = 900 combinations.

**What happens to the molecules.** They are split into five folds by scaffold. A second
scaffold grouping is built from the SMILES for the noise itself, with acyclic molecules
kept as singletons. Inside each fold, only the training labels are corrupted. For every
corrupted copy, five models are fitted in each feature set, and twelve noisy-label
methods are trained. Separately, two or three estimators score each training molecule
for reliability. Those scores become weights for a second round of fits.

**What comes out.** Per shape and per level, four things. A mean score per feature set.
The share of the clean-label score retained. The share of the noise-induced loss
recovered. A count of how many shapes the hybrid still beat the plain representation
in.

**What disagrees.**

1. **Censoring is absent from the shape sweep.** Where this matters: in the upstream
   study, one-sided error was the only thing that separated. `noise_conditions.json`
   records `grouped_shifted` beating `grouped_wider` by 0.0963 R² on the random forest
   and by 0.1419 R² on LightGBM. That is 2.52 and 2.71 times the run-to-run spread.
   Every pure shape effect there sat under 0.0049 R². Both conditions carrying that finding are the
   asymmetric ones — censoring for a whole dataset, `grouped_shifted` for one chemical
   family. `n13` runs `grouped_shifted` and not censoring. Excluding it is defensible on
   the stated ground that its level is a different quantity. The file does not say so
   anywhere.

2. **The hybrid's recipe was chosen with the clean labels visible; the plain
   representation had no recipe chosen for it at all**
   (`n13_noise_distributions.py:288-294`). The file records both readings of itself: it
   says the recipe is clean-searched at line 894 and says the opposite at line 587.

3. **The retention figure divides a hybrid score by a plain-representation clean score.**
   The clean anchor is fitted once per fold on the plain representation alone
   (`n13_noise_distributions.py:456-460`), and every feature set's corrupted score is
   divided by that same number (`:791-804`). The top and the bottom of that fraction
   differ in two ways: whether the training labels were corrupted, and which feature set
   was used. A retention above 100% can come entirely from a feature set being better
   with no corruption present.

4. **The structure-chosen hybrid is counted on the plain-representation side of the
   tally**, because the tally sorts by the literal name `'hybrid'`
   (`n13_noise_distributions.py:1026-1039`). On any dataset where that run is switched
   on, the headline flag compares one hybrid's mean against the mean of a plain
   representation *and a second hybrid*.

5. **On a classification dataset the shared grid is read as a flip rate.** Measured in
   this audit on 400 synthetic binary labels. At level 0.5 the realised flip share was
   0.492, and the correlation between corrupted and clean training labels was 0.019. At
   level 0.75 that correlation was −0.411. At level 1.0 it was −0.995, which is every
   label inverted.
   Three of the six classification levels carry no learnable signal.

6. **On classification the molecules that get flipped change at every level. On
   regression they do not.** The classification injector is built with the noise seed
   only. The seed that pins which molecules are hit is never passed
   (`src/kirby/noise_spec.py:419-422`). `noiseInject`'s classification constructor has
   nowhere to put it. Measured in this audit on 300 synthetic binary labels: 39
   molecules flipped at level 0.15 and 83 at level 0.3. The two sets overlapped in 16
   molecules, where chance overlap for two sets that size is about 11. So a classification level curve changes
   the flip rate and the identity of the flipped molecules at the same time.

7. **The zero-level control is not clean on classification.** Asking for a flip rate of
   0.0 returns a floor rather than zero — 0.001953 on 200 labels, 0.0078125 on 400 and
   on 1000, measured in this audit. So the level-0 combination flips a few labels, and a
   different number of them in every fold.

8. **The combination copied to the top of the output file is described three different
   ways in one file.** The code picks the level nearest 0.5 of the label spread
   (`n13_noise_distributions.py:519-520`). The comment two lines above calls it Gaussian
   at one unit of real measurement error — 0.54 log units on herg_ki, 0.35 on caco2. Two
   other comments call it Gaussian at 0.3. Three descriptions, one line of code.

---

### n15 — how much does searching the recipe on clean labels buy

**What it asks.** How much of a hybrid's apparent advantage under label noise comes from
its recipe having been chosen while the labels were still clean.

**Which noise.** Gaussian only (`src/kirby/noise_spec.py:132`).

**Which levels.** 0.0, 0.15 and 0.3 of the clean label spread, from
`noise_helpers.NOISY_SEARCH_LEVELS`. Changed on 2026-09-12 (§12.5); it was 0.05, 0.10 and
0.20, and no value in it appeared on the grid the other noise experiments sweep.

**What happens to the molecules.** One corrupted copy of the whole label column is drawn
per level, from a fixed seed. For each of five downstream models, two recipes are
fetched. One was chosen by a search that saw the true labels. The other was chosen by a
search that only ever saw the corrupted ones, both for fitting and for judging which
recipe won. How far the two recipes agree is recorded — same representations, same
allocation method, same importance method, same budget, same internal evaluator.

The molecules are then split by scaffold. Within a fold both recipes are handed the same
corrupted training labels. Each builds its own features and a model is fitted on them.
Predictions for the held-out molecules are graded against those molecules' untouched
labels. The two fold-mean scores are subtracted. A positive number means the recipe
chosen with true labels did better.

**What disagrees.**

1. **FIXED 2026-09-12 — the levels now sit on the shared grid.** `n15` used to run 0.05,
   0.10 and 0.20 of the label spread while `n11` and `n12` ran 0.0, 0.15, 0.3, 0.5, 0.75
   and 1.0. No value appeared in both lists, so `n15`'s answer could not be read against
   the experiments it exists to settle. It now runs 0.0, 0.15 and 0.3, at the same cost
   as before. 0.0 is a control that measures defect 3 below on its own. At level 0.0 the
   labels are untouched, so the only thing left to separate the two recipes is how their
   searches were run. 0.5, 0.75 and 1.0 are still not covered, and whether the
   advantage grows or shrinks as noise rises is measured nowhere.

2. **CLOSED 2026-09-11 — the precompute that nothing read has been deleted.** The
   runner used to build one full precompute per level, three per dataset, of searches
   that train on corrupted labels and pick their winner against the clean ones. `n15`
   asks for a search that picks its winner against the corrupted ones too
   (`n15_clean_vs_noisy_hybrid.py:128`). That request skips the in-memory store on
   purpose (`_shared.py:5190-5193`) and appends a marker to the cache name
   (`find_best_hybrid.py:7760-7766`). So the two could never meet, and `n15` paid for its
   own searches on top. The block is gone and the reasoning stands in its place
   (`noise_helpers.py:378-395`). The dictionary is left empty for a caller that asks for a
   noisy search without that flag. No such caller exists.

   What this did not change: `n15` still runs one full search per level per model inside
   the experiment. Those land in the on-disk store, so a second run on the same dataset
   reuses them.

3. **The two searches being compared chose from lists of different lengths.** The clean
   recipe comes from the searches held in memory, run with the rule that drops losing
   allocation methods after the first fold (`noise_helpers.py:339`, threshold 0.15,
   `find_best_hybrid.py:6181`). The noisy recipe goes through
   `get_precomputed_search`, which switches that rule off (`_shared.py:5125`). So one search tried more allocation
   methods than the other. Part of the difference between them is that, not the
   clean-label head start.

4. **The check the file promised does not exist; the claim is now removed.** The scope
   note used to say a short fingerprint of the corrupted label column is asserted against
   the cached search, so a mismatched recipe stops the run. There is no such assertion
   anywhere in the repository, and since 2026-09-11 there is no cached search either. The
   fingerprint is still computed and stored (`n15_clean_vs_noisy_hybrid.py:189`) and
   still compared to nothing. No stored search record carries a noise field at all — not
   a setting name, not a seed, not a label fingerprint
   (`find_best_hybrid.py:1344-1401`). The docstring was corrected on 2026-09-12 to say
   so; **the missing check is still missing**.

5. **The amount actually delivered is never recorded.** `n15` calls the plain
   corruption function, which returns only the corrupted column. The requested level
   reaches the output file; the realised root-mean-square does not. Every other
   injecting experiment records both.

6. **The two means can be taken over different folds.** A fold that raises in one of the
   two runs appends nothing to that run's list. The two means are then subtracted with
   no check that the fold sets match (`n15_clean_vs_noisy_hybrid.py:250-261`).

---

## 5. The four files in that directory that add no noise

**n01** marks each molecule a cliff or a non-cliff, using a similarity built outside the
ten representations, then compares prediction error on cliff molecules against non-cliff
molecules. No label is altered. It sits in the noise directory and nothing in it is
about injected noise.

**n02** builds a per-molecule training weight from how far each label sits from its
neighbours' mean, refits, and reports the change. The five things it sweeps are
weighting settings, not noise conditions. No label is altered.

**n06** is the only experiment in KIRBy that uses real label noise. BindingDB holds the
same drug measured against the same protein by more than one laboratory. The standard
deviation of each such set of pKd values is how far the laboratories disagreed. Ten
scoring methods are asked to rank molecules by noisiness, and each ranking is compared
against that disagreement by a Spearman correlation. No prediction model is fitted and
no held-out score is reported.

**n14** measures the activity-cliff landscape and runs an out-of-fold residual
diagnostic. No label is altered.

---

## 6. Where the noise does not match the question being asked

This is the part that answers "think about it logically". Each item is a case where the
experiment adds a kind of noise its own question cannot be answered under.

### 6.1 "Which molecules are unreliable" has no answer under five of the eleven conditions

Under `gaussian`, the three `student_t` settings, `laplace` and `grouped_shifted`, every
molecule receives the **same** noise width. `NoiseSpec.scale_is_degenerate` says so
(`src/kirby/noise_spec.py:120-127`) and `noiseInject` says the same at `core.py:84`.

So a rank correlation between a detector's score and "how badly this molecule was
corrupted" is a correlation against a constant. It is undefined, not zero.

**No live experiment reports such a correlation.** I checked. `n09`, `n11`, `n12` and
`n13` all measure recovered performance instead, which is defined under a constant
width. `tests/alternative_data_noise_robustness.py` prints a note naming the degenerate
conditions and carries on.

What *is* affected is precision and recall against the corrupted-molecule list.
`inject_train_labels` returns a list of every molecule whose label actually moved
(`noise_helpers.py:169-176`). Under Gaussian that is every training molecule. So a
precision figure computed against it has a chance level of 100 flagged molecules in
every 100.

**Two live experiments do exactly that.** `tests/experiments/similarity_optimized_hybrid.py:707-712`
computes precision, recall and F1 against that list, under Gaussian, at a contaminated
share of 0.25 written into the file. Under Gaussian, precision is 1.0 by construction
and recall is just the share flagged. `fed01` grades detectors the same way on the
federated side.

### 6.2 A scaffold split and scaffold-keyed noise cannot both be in one experiment

`RERUN_PLAN.md` §3.1f states this for the upstream study. The reasoning carries over.
`grouped_wider` and `grouped_shifted` key the noise to the scaffold family. Every KIRBy
noise experiment splits by scaffold family. So a held-out molecule is in a family the
selection never marked.

**This does not affect any KIRBy noise experiment as written**, because detection here
happens on training molecules, and the training block does contain the marked families.
It would affect any experiment that asked whether a *held-out* molecule was corrupted.
None does today.

### 6.3 Four of the sixteen detection filters never read a label at all

`lof`, `lof_k10`, `lof_k20` and `mahalanobis`. The base class says the method returns
the molecules whose labels are wrong, and the cleanup deletes exactly those
(`noise_mitigation_methods.py:39-60`). The local-outlier filter touches the label array
once, to cap a neighbour count. The Mahalanobis filter does not touch it at all — it
thresholds a distance in feature space at the top 10 molecules in every 100
(`:487-518`, `:777-807`).

So a recall figure for any of those four is the overlap between structural outliers and
whichever molecules the injector happened to hit. `fed01`'s detector list includes `lof`
(`federated_helpers.py:1430-1434`), so that filter is given a precision and a recall on
the federated side.

### 6.4 The nearest-neighbour filters score the neighbours, not the molecule

`KNNConsistencyFilter` takes the k nearest neighbours of a molecule, with that molecule
excluded. It computes the spread of **their** labels. It flags the molecule when that
spread is above the 90th percentile. The molecule's own label never enters its own score
(`noise_mitigation_methods.py:443-453`).

`NeighborhoodConsensusFilter`, two classes further down the same file, takes the
distance from the molecule's own label to its neighbours' mean
(`:646-655`).

Two ways they differ: whose label is measured, and whether a molecule whose own label
was corrupted can raise its own score at all. Both sit in the same detection table.
Under any condition that corrupts molecules independently, a corrupted molecule raises
its *neighbours'* scores rather than its own.

### 6.5 Five of the twelve learning methods need a clean subpopulation that Gaussian does not provide

Read out of `src/kirby/utils/lnl_methods.py`. The twelve are the entries of
`LNL_REGISTRY` at `:2225-2242`. Five of them keep or drop a share fixed in advance:

- `small_loss` keeps the 70 molecules in every 100 with the smallest squared error
  (`:549`, `:611`).
- `confident_learning` flags every molecule whose error is above the 85th percentile of
  error and whose ensemble spread is below the 25th percentile of spread (`:635`,
  `:689-690`).
- `distance_selection` flags every molecule above the 85th percentile of disagreement
  with its neighbours (`:718`, `:754`).
- `multistage_cleaning` drops 10 molecules in every 100 per stage, over three stages
  (`:785`).
- `dividemix` fits a two-part mixture to the per-molecule losses and calls the
  lower-loss part clean (`:1288-1307`).

`co_teaching`, in the other file, keeps 70 in every 100. Three of the sixteen detection
filters flag a fixed 10 molecules in every 100.

Every one of those assumes some molecules are wrong and the rest are right. That premise
holds under `outlier_p01`, `outlier_p05`, `outlier_p10` and `grouped_wider`. It is false
under `gaussian`, the three `student_t` settings, `laplace` and `grouped_shifted`, where
every training molecule is corrupted.

`n09` and `n11` run Gaussian only. So those five methods are ranked under the one family
of conditions their premise does not hold for. They still return a number, because a
fixed keep-fraction always returns a number. That number says which 70 molecules in
every 100 the network happened to fit. It does not say which labels are wrong.

`n13` runs all ten shapes, so it does cover the three outlier conditions and
`grouped_wider`. It is the one experiment where those five methods are asked something
they can answer.

### 6.6 Seven methods assume the neighbours are a clean reference, and the grouped conditions break that

Seven, counted by name: `distance_selection` from the learning file, and `knn_k5`,
`knn_k10`, `knn_k20`, `distance_weighted`, `neighborhood_consensus` and
`activity_cliffs` from the detection file
(`noise_mitigation_methods.py:1149-1174`). All seven judge a molecule against its
nearest structural neighbours.

Under `grouped_wider` and `grouped_shifted`, the noise groups are Murcko scaffolds, and
the nearest neighbours in a fingerprint space are the same scaffold family. So the
reference has been moved by the same offset as the molecule being judged. Under
censoring, the clipped molecules sit at the top of the label distribution and their near
neighbours are clipped with them.

`n13` runs those seven methods under the two grouped conditions. What comes out is a
number. It is not a measure of detection.

### 6.7 The one method that asks where the noise is has a target that is flat in five conditions

`HeteroscedasticGP` fits a Gaussian process for the mean. Each molecule's absolute
residual, over the label spread, becomes that molecule's noise estimate. A second
network is then trained to predict that residual **from the features**. Training
molecules are weighted by one over one plus that prediction
(`lnl_methods.py:2008-2050`).

The per-molecule noise width is the same number for every molecule under `gaussian`, the
three `student_t` settings, `laplace` and `grouped_shifted`. Under `outlier_p01`,
`outlier_p05` and `outlier_p10` the molecules that get the wider error are drawn at
random with no reference to the features.

So the premise holds in one of the ten conditions whose level is an amount:
`grouped_wider`.

### 6.8 One method assumes symmetric error; two conditions are deliberately asymmetric

`evidential_kernel` fits a likelihood that is a Gaussian with unknown variance
(`lnl_methods.py:2143-2152`). That is right under `gaussian`. It is misspecified under
`laplace`, the three `student_t` settings and the three outlier conditions. It is
meaningless under censoring, which is one-sided by construction, and under
`grouped_shifted`, whose family offsets are deliberately not centred.

### 6.9 The detection cut-offs are divided by the corrupted spread, while the level is a fraction of the clean spread

`DistanceWeightedPredictionFilter` flags a molecule when its error exceeds 2.0 times the
standard deviation of the label array handed in. `NeighborhoodConsensusFilter` divides
by the same quantity. `ZScoreFilter` too
(`noise_mitigation_methods.py:579`, `:626`, `:1077-1080`).

The array handed in is the corrupted one. A level on the shared grid is a fraction of
the **clean** label standard deviation (`src/kirby/noise_spec.py:337-348`).

Measured here, 2000 labels, Gaussian, the shared grid — the corrupted spread divided by
the clean spread:

| level | 0.0 | 0.15 | 0.3 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|
| corrupted spread / clean spread | 1.0000 | 1.0092 | 1.0349 | 1.1264 | 1.2474 | 1.4002 |

So a cut-off nominally at two spreads is at 2.80 clean spreads by the top of the grid.
The ruler grows with the amount of noise. The number of molecules flagged across the
level grid is therefore not a detection curve.

### 6.10 The shared grid reaches the classification injector unchanged

`dataset_policy.noise_levels()` takes a dataset name and a condition name. It takes no
task type (`dataset_policy.py:390-397`). The same six numbers become a flip rate on a
classification dataset. Three of the six sit at or past the point where nothing is
learnable, and the top of the grid inverts every label.

This affects `n13` on every classification dataset, and `n09`, `n11` and `n12` wherever
they run on one.

### 6.11 Twelve method signatures take a clean validation label array that no method body reads

The contract for the twelve learning methods names its fourth argument as validation
labels, "clean labels for early stopping and meta-learning methods"
(`lnl_methods.py:434-445`). Across the whole file that argument appears only in the
twelve signatures and in the contract text. No body reads it.

Seven methods carve a fifth of the **training** molecules off and stop on the loss over
those. The other five stop on a fifth carved inside the shared trainer. Both carves
carry the injected corruption.

Two ways the description and the code differ: corrupted rather than clean, and inside
the training fold rather than held out. The carve itself is deliberate and says so at
`:458-467`. It is the contract text that is stale.

### 6.12 Censoring is never run in the noise directory

Nothing in `tests/experiments/noise/` runs censoring. `n13` uses the list of conditions
whose level is an amount, which excludes it. `c10` does run it, across all eleven
conditions. `tests/alternative_data_noise_robustness.py` runs it as the settled roster
requires.

`NOISE_DESIGN.md` §5.8 gives the upstream evidence for why this matters. There,
one-directional error did roughly thirty times more damage than any shape effect. It is
also the one condition where "does the model know which labels are unreliable" has an
answer. That is because which molecules get clipped depends on their value.
### 6.13 The list of corrupted molecules is returned on every draw and never read

Added 2026-09-11, from a separate session. Read out of the code that day.

`inject_train_labels` returns three things: the corrupted labels, the list of molecules
whose label actually moved, and a provenance row. Its own docstring calls the second one
"the ground truth for 'did the estimator find the corrupted samples'" and says the
detection and remediation experiments need it (`noise_helpers.py:148-150`).

Five experiments call it. Every one throws that list away:

| file | line | what the list is bound to |
|---|---|---|
| n07 | 190 | `_noise_mask` |
| n07 | 620 | `_` |
| n07 | 766 | `_` |
| n09 | 282 | `_cmask` |
| n11 | 564 | `_cmask` |
| n12 | 396 | `_cmask` |
| n13 | 567 | `_cmask` |

Each of those names appears exactly once in the whole directory — on the line that binds
it. Grepping `tests/experiments/noise/` for `precision`, `recall`, `f1_score` and
`average_precision` returns nothing.

So no experiment in the noise directory reports a detection number at all. Every detector
and every cleanup is graded only by how the model scores afterwards. That one number mixes
three things: whether the detector flagged the right molecules, whether the cleanup helped,
and whether the model would have tolerated the corruption untouched.

This is separate from §6.1. §6.1 says a rank correlation against a per-molecule width is
undefined under five conditions. This says something else. Precision and recall against
the list of molecules that moved are not computed anywhere, even though that list is
already in hand. They are defined under `outlier_p01`, `outlier_p05`, `outlier_p10` and
`grouped_wider`.

### 6.14 Three of the six injecting experiments fit one model

Added 2026-09-11, same session as §6.13.

`n11` fits random forest and nothing else (`n11:370-372`). `n12` the same
(`n12:306-308`). `c10`, outside the directory, builds
`RandomForestRegressor(n_estimators=100)` straight from scikit-learn rather than from the
model roster, at `c10:302` and `c10:414`. Its file header states the conclusion "hybrids
are more noise-robust than individual reps".

`n02` fits three: `WEIGHTED_MODELS = ['rf', 'xgboost', 'lightgbm']` at `n02:54`, under the
comment "Only use models that support sample_weight" at `n02:155`. The roster is
`['rf', 'xgboost', 'svm', 'dnn', 'lightgbm']` (`_shared.py:539`). The engine's `svm` is
scikit-learn's `SVR` and `SVC` (`find_best_hybrid.py:2477-2478`), and both accept
`sample_weight` — checked against the installed scikit-learn. So `svm` is excluded from
the weighting comparison and the stated reason does not hold for it. Whether the engine's
`dnn` accepts a per-molecule weight was not checked.

`n07`, `n13` and `n15` each run the full roster.

### 6.15 Three of the six injecting experiments use one noise draw per fold

Added 2026-09-11, same session as §6.13.

`n07:97` and `n13:114` set `NOISE_SEEDS = [0, 1, 2]`. `n11:486` reads `noise_seeds` from
its arguments and defaults to the same three. So those three average three draws per fold.

`n09`, `n12` and `n15` contain no seed loop. Each runs one draw per fold. `n12`'s own
header already says the differences between its conditions may sit inside run-to-run
variation.


---

## 7. The noise experiments outside `tests/experiments/noise/`

Eleven more files inject label noise. Four are called by something. Eight are not.

### Live

**`tests/alternative_data_noise_robustness.py`** — the KIRBy-side replication of the
qsar_qm noise study, on logD, Caco-2 efflux and hERG Ki. It is the only file in KIRBy
that reads the settled condition roster from
`~/repos/qsar_qm_models/noise_conditions.json`, and the only one that runs censoring.
Submitted by three SLURM scripts. Its level grid is 0.0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5 —
seven levels, two of which are not on the shared six.

Three things in it disagree with themselves:

- Three comments say the level is a fraction of the fold's clean training label spread
  (lines 50-51, 2508, 2738). The code computes that spread over **every molecule in the
  dataset**, held-out molecules included (lines 2534, 2765). A second spread, fitted on
  the training molecules alone, is written onto every uncertainty row (line 3767). So one
  output row carries two different spreads: the level divided by the whole dataset's, the
  uncertainty divided by the training block's.

- The column describing where the noise fell and the ceiling that bounds it are drawn
  over different molecules, with different group arrays (lines 2550-2553 against
  2649-2652). Under `grouped_wider` the ceiling column on held-out molecules is one
  number repeated, because a scaffold split guarantees the training families the
  selection chose appear in none of the held-out molecules. Verified in this audit on a
  200-molecule fixture with 40 scaffolds: two distinct values in one column, one value in
  the other.

- Censoring's descriptive column is computed at a clipped fraction of 1.0 (line 2532),
  while the fractions actually run are 0.10 to 0.50. The injection path refuses a
  fraction of 1.0; the descriptive path has no such check.

**`tests/experiments/complexity/c10_noise_robustness.py`** — on the main roster
submitter. It runs **all eleven** regression conditions at the full grid
(`c10_noise_robustness.py:192-193`), and its live caller passes no condition list. Two
things to know before reading its table:

- The skill floor is switched off on both sides (`baseline_threshold=None`,
  lines 320-322 and 432-434). That was done deliberately on 2026-09-10. The comment
  at `noise_robustness.py:105-108` gives the reason. Applying a floor on one metric
  direction and not the other meant the RMSE datasets admitted every representation,
  while the Spearman and AUC ones dropped any representation under 0.3. The consequence
  now: on a higher-is-better dataset a clean score of 0.02 becomes the divisor for every
  level. A negative clean score flips the sign of every ratio.

- Ten conditions sweep an amount and censoring sweeps a fraction of labels clipped. The
  summary table prints all eleven side by side under one heading, in one number format
  (lines 622-641). The two are mean retention over different quantities.

**`tests/experiments/similarity_optimized_hybrid.py`** — submitted per dataset. Its
noise section runs three conditions: `gaussian`, `grouped_wider` and one outlier
setting. `grouped_shifted` and censoring are absent, with no reason given anywhere in
the file. It uses one level, 0.5 of the label spread, and one contaminated share, 0.25,
both written into the file at lines 141-148. The condition name recorded is
`outlier_p05` while the share delivered is 25 molecules in every 100. The name is pinned
when the condition is built, and it does not follow the overridden share.

**`tests/experiments/federated/fed01_noise_heterogeneity.py`** — on the federated
roster. One classification condition out of six. A private level grid,
`[0, 0.05, 0.1, 0.15, 0.2, 0.3]` (line 65). It imports neither `dataset_policy` nor
`noise_helpers`.

Three different numbers are called the noise level in it. Each silo is corrupted at its
own rate, evenly spaced from 0 up to the setting, so only the last silo sees the full
rate (lines 305-307). The recipe search is calibrated at the mean of those five rates,
which is half the setting (line 219). The results column is headed with the setting
itself (line 971). So a row reading 0.3 cannot be read as "under a 30 percent flip
rate".

### Not called by anything

`tests/qm9_graphs_noise_robustness.py`, `tests/noise_mitigation_extended.py`,
`tests/noise_mitigation_advanced.py`, `tests/cox2_noise_robustness.py`,
`tests/drd2_noise_robustness.py`, `tests/esol_noise_robustness.py`,
`tests/herg_noise_robustness.py`, `tests/hybrid_noise_robustness.py`.

Each was searched for across the whole repository — runners, submit scripts, SLURM
scripts, imports. Nothing calls any of them.

**All eight would stop at the first injection.** They ask for the six condition names
`noiseInject` 1.0.0 deleted — `legacy`, `quantile`, `threshold`, `outlier`, `hetero`,
`valprop`. Those names are not in the registry, so building the injector raises. This was
run in this audit and it does raise. They also read a level as an absolute error in the
label's own units, not as a fraction of the label spread. So any result file on disk from
them sits on a different axis from everything else.

---

## 8. Things in the engine that only bite if something changes

None of these affects a run today. Each is a place where two parts of the code disagree
and nothing currently exercises the disagreement.

**The engine cannot run the two grouped conditions.** `find_best_hybrid` calls the
corruption function with the labels and the setting and nothing else
(`find_best_hybrid.py:7756`), and it has no parameter through which a caller could pass
scaffold groups. `noise_spec.py:388-393` raises without them. The precompute catches
that, stores nothing for that condition, and records a failure
(`_shared.py:4756-4764`). So neither grouped condition has a noise-searched recipe
anywhere, and a table ranking conditions by how a noise-searched hybrid holds up cannot
carry those two rows.

**Under nested cross-validation the recipe would be chosen on corrupted labels and the
reported model fitted on clean ones.** The inner search takes the noise setting
(`_shared.py:870-871`). The outer refit takes `labels[tr]` from the clean array
(`find_best_hybrid.py:9518`). No experiment passes a noise setting into nested
cross-validation today, so nothing runs it.

**Classification settings drop their own parameters, and the cache name still carries
them.** For regression the condition's parameters reach the injector
(`noise_spec.py:355-363`). For classification the injector is built with a strategy and
a seed only (`:417-422`). The cache name is built from the declared parameters either
way.

**The seed that pins which molecules are hit is honoured for regression and dropped for
classification**, while the cache name claims it in both
(`noise_spec.py:243-244`, `:417-422`).

**The three levels the noise-searched recipe is cached at cannot produce a retention
curve.** `robustness_metrics` needs at least four points and one of them at zero
(`noise_robustness.py:96-98`). The cached grid is three points and none at zero
(`noise_helpers.py:81`).

---

## 9. What I checked and found correct, so it is not re-audited

- No held-out label is corrupted anywhere in `tests/experiments/noise/`. Checked at
  every metric call in all ten files.
- No held-out label is corrupted in `tests/alternative_data_noise_robustness.py`,
  `similarity_optimized_hybrid.py` or `fed01` either.
- The hybrid search trains on a corrupted copy and grades its inner validation folds on
  the clean labels (`find_best_hybrid.py:4286-4288`, `:4613`). It refuses to run
  alongside fine-tuned representations, because those would be tuned on the clean labels
  and hand the hybrid back the advantage the parameter exists to remove
  (`find_best_hybrid.py:7741-7751`).
- The noise setting rides on the cache name, and it has to. With training-only noise the
  global label array is byte-identical to a clean run, so the label fingerprint cannot
  tell the two apart (`find_best_hybrid.py:7730-7736`).
- A fresh injector is built on every call, in every path. None is reused across draws.
- `assign_noise_groups` is used for the grouped conditions everywhere they run in the
  noise directory. The fold grouping is never substituted for it.
- The two copies of `noiseInject` are byte-identical, and the installed package is the
  one in `~/repos/NoiseInject`.
- The corrupted-molecule list that `inject_train_labels` returns handles the
  constant-width conditions correctly — it returns every molecule that moved, rather
  than an empty list.
- No live experiment reports a rank correlation against a constant noise width.
- Held-out labels staying clean is the settled design, not an oversight.
  `RERUN_PLAN.md` §2.5 states it: training noisy, validation noisy with its own draw,
  test clean. §2.1 gives the reason — corrupting held-out labels adds a smooth downward
  bias that looks exactly like a robustness curve, and they fitted it at 1.288 against a
  label standard deviation of 1.293. KIRBy's early-stopping rows are carved out of the
  corrupted training block, so they carry noise too, which is what that design asks for.
- On herg_ki the fold grouping and the noise grouping are the same. 0 of 1415 molecules
  have no ring, so no molecule is pooled into a shared acyclic group. Both give 769
  groups with the largest holding 33 molecules. That has to be re-checked on any dataset
  that does contain acyclic molecules; it is not a property of the code.

---

## 10. What needs a ruling from you

Eight are now closed — seven decided by you on 2026-09-12 and one closed by
another session on 2026-09-11. What was decided and what it cost is in §12. Items 7 to
15 are open. Items 11 to 15 were added 2026-09-11 from a separate session, and come from
§6.13 to §6.15 and §11.

**7. Whether the Gaussian-only experiments stay Gaussian-only.** `n09`, `n11`, `n12` and
`n15` each name one condition, with a comment saying the other ten are reachable by
keyword. No caller passes one. Adding a condition axis to `n11` multiplies its grid by
however many conditions you add. `n13` already sweeps ten shapes, so the question is
whether the detect-and-repair grid needs them too.

**8. Censoring in the noise directory.** It runs in `c10` and in
`tests/alternative_data_noise_robustness.py`, and nowhere in
`tests/experiments/noise/`. Adding it to `n13` means a second axis. Its level is a fraction of labels clipped and
cannot share the amount axis. That is 7 censoring levels beside the 6 amount levels, on
one dataset per run.

**9. The classification level grid.** Six numbers written as fractions of the label
spread are read as flip rates on every classification dataset. Three of the six carry no
learnable signal and the top of the grid inverts every label. Either
`dataset_policy.noise_levels()` takes a task type, or the classification runs take a
separate grid. This changes what every classification noise number in `n13` means.

**10. The eight files nothing calls.** None is reachable and all eight raise at the first
injection, because they name condition strings `noiseInject` 1.0.0 deleted. Delete them,
or mark them retired in one line each. I have not touched them.

**11. Whether a detection score is written beside every repair score.** Added 2026-09-11.
The list of molecules whose label moved is already returned on every draw and thrown away
on all seven lines that receive it (§6.13). Reading it costs no extra model fits — it is
precision, recall and area under the precision-recall curve against a list already in
memory. It is only defined where a corrupted subset exists: `outlier_p01`, `outlier_p05`,
`outlier_p10` and `grouped_wider`. Under `gaussian`, the three `student_t` settings,
`laplace` and `grouped_shifted` every training molecule moves, so precision is 1.0 by
construction and the figure must be suppressed rather than printed. This decision is tied
to 7: a detection number is only worth having if the Gaussian-only experiments also gain a
condition it is defined under.

**12. Whether `n11` and `n12` keep fitting one model.** Added 2026-09-11. Both fit random
forest alone (§6.14). The roster is five. Widening either multiplies its grid by five.
`n11`'s grid is about 1,120 configurations per level before that. `c10` is the same
question outside the directory, and it builds its forest straight from scikit-learn rather
than from the roster, so it also bypasses the size guards and the classifier dispatch.

**13. Whether `svm` rejoins `n02`.** Added 2026-09-11. `n02` fits three models under the
comment "only models that support sample_weight". Scikit-learn's `SVR` and `SVC` both
accept `sample_weight`, so the stated reason does not cover `svm`. Adding it costs one
more model across `n02`'s five weighting settings. Whether the engine's `dnn` accepts a
per-molecule weight is unchecked and is the other half of this call.

**14. Whether `n15` averages more than one noise draw per fold.** Added 2026-09-11,
narrowed 2026-09-12. `n09` and `n12` were the other two and now average three (§12.8).
`n15` still runs one. Three draws is three times the noise-dependent fits, which for
`n15` means three times its inside-the-experiment searches.

**15. Whether the loaders stop deleting the real measurement error, and whether a
removal experiment is built on it.** Added 2026-09-11, from §11. This is the largest of the
fifteen and the only one that is not a change to the noise directory. Keeping the
measurement count and the spread means re-fetching hERG Ki under a recorded ChEMBL release
and changing three loaders. Nothing about showing removal on real data can start until it
is done. The count that decides whether it is worth doing — how many molecules in each
dataset carry three or more measurements — is unmeasured, and `n14` is where it would be
printed.

---

## 11. Real measurement error, and what the loaders delete before an experiment sees it

Added 2026-09-11, from a separate session. Every line number below was read out of the
code that day. The design at the end is a proposal awaiting a ruling, not a finding and
not a result.

### 11.1 hERG Ki has had its real error removed three separate ways

`tests/alternative_data_noise_robustness.py` builds `~/.cache/kirby/chembl_herg_ki.csv`,
and that file is what the hERG Ki experiments read. Three things happen on the way:

1. Line 997 queries ChEMBL with `"data_validity_comment__isnull": "true"`. Every row
   ChEMBL's own curators marked as suspect is excluded before anything else runs.
2. Line 1043 groups by SMILES and line 1045 computes each molecule's standard deviation
   across its repeat measurements. Line 1049 then deletes every molecule whose standard
   deviation is above 1.0 log unit.
3. Line 1052 writes two columns, `SMILES` and `pKi`. The spread, the measurement count and
   the assay identifiers are all discarded. The cached file on disk has those two columns
   and nothing else — checked.

So the molecules the laboratories disagreed most about are gone, the rows a human marked
wrong are gone, and the per-molecule disagreement that survived is not written down. Every
noise experiment then injects synthetic error into what is left.

The same collapse is in two more loaders, without the deletion step:

| loader | line | what it does | what it keeps |
|---|---|---|---|
| `src/kirby/datasets/herg.py` | 105 | median IC50 per SMILES | neither the count nor the spread |
| `src/kirby/datasets/chembl_bioactivity.py` | 266 | mean pActivity per (InChIKey, target) | neither the count nor the spread |

`src/kirby/datasets/canine_cyp.py:178` is the exception in the repository: it aggregates
with `n_measurements=('pActivity', 'size')`, so the count survives there.

### 11.2 The machinery for a replicate ground truth exists and points at one dataset

`n06` builds replicate groups from (drug, target) pairs and correlates ten scoring methods
against the disagreement between laboratories. It fits no model and reports no held-out
score (§5). Its dataset is written into the file: `bindingdb_kd` at `n06:942-944`.

`n14` imports the same two functions, `build_replicate_groups` and
`compute_replicate_stds` (`n14:74`). It runs them on whatever dataset it was given
(`n14:359-364`). It writes a `has_replicates` flag and a replicate-molecule count onto
its output (`n14:374-379`). So the builder is already general and is already run on every
dataset `n14` touches. What it has never been used for is a removal experiment.

### 11.3 A proposal for showing removal on experimentally noisy labels

Not a finding. Nothing below has been run, and no number below has been measured.

Three things have to come from three different places. A per-molecule error figure that
is neither the features nor the label being graded. A training set at the noisy values.
A held-out set at the consensus values.

Replicate disagreement supplies all three. For each molecule with three or more
measurements, hold one measurement out at random and use it as that molecule's training
label — that is one laboratory's number. The mean of the remaining measurements is its
held-out consensus label. Molecules measured once go into training at face value and are
never graded. Train on the single measurements, grade on the consensus.

Five ways of choosing what to train on, all graded on the same consensus-labelled
held-out molecules. One row is one of the five.

| what is trained on | what it does |
|---|---|
| no cleaning | train on every training molecule, untouched |
| drop | remove the molecules a detector flags |
| repair | replace flagged labels with the neighbour mean |
| ceiling | remove the molecules whose true replicate spread is in the top tenth |
| floor | remove the same count of molecules, chosen at random |

The ceiling says how much is available to win before any detector is involved. The floor
says how much of any gain is just from training on fewer molecules. A removal claim is
drop or repair beating no cleaning, reported against the distance between floor and
ceiling.

Two things it needs that do not exist. The measurement count and the spread have to
survive the loader (§11.1). And nobody has counted how many molecules in each dataset
carry three or more
measurements. That count is the go/no-go and `n14` already prints the neighbouring figure.

### 11.4 A second real-noise source that needs no replicates

The hERG fetch currently excludes every row ChEMBL's curators flagged (§11.1, item 1).
Pulling those rows in instead, with the flag kept as a column, gives a set of molecules a
human has already marked as wrong. Detection precision and recall against that flag is a
different claim from precision against a distribution drawn in the same script. Whether
the flagged rows are numerous enough on CHEMBL240 is unmeasured; it needs a fetch under a
recorded release.

---

## 12. Decided and built, 2026-09-12

Seven rulings from you, one closure from another session. Each is in the code as of this
edit. The measurements below were run after the change, not before.

### 12.1 A molecule's corruption is now a property of the molecule

*Your ruling: make the change.* `RERUN_PLAN.md` §3.3b, settled 2026-09-04.

Two halves, and the first alone would not have worked. `inject_train_labels` now takes
the FULL clean label column plus this fold's positions, draws once over the whole
column, and indexes the training rows out. And `noise_spec_for` no longer takes a fold
at all. With the fold still in the seed, the whole-column draw was simply repeated with a
different seed, and every molecule still came out differently per fold.

Measured, 600 molecules, 5 scaffold-disjoint folds, level 0.5:

| condition | before, molecules with more than one corrupted value | after |
|---|---|---|
| gaussian | 600 of 600 | 0 of 600 |
| outlier_p10 | 600 of 600 | 0 of 600 |
| grouped_wider | 600 of 600 | 0 of 600 |
| grouped_shifted | 600 of 600 | 0 of 600 |
| censoring | 0 of 600 | 0 of 600 |

The amount delivered was 4.9% to 23.9% apart across the five folds; it is now one number
per condition. Two things follow from the whole-column draw, and neither was separately chosen, because
there is no fold-free alternative to either. The amount is solved against the full
column's clean spread. The cut-point for censoring is the full column's quantile.

Held-out labels are untouched, as before — only the training positions are returned. A
molecule held out in one fold and training in another is corrupted in the second only.

`train_idx` is keyword-only and required, so a call written for the old signature raises
a TypeError rather than quietly meaning something different. Seven places that call it
and five smoke scripts were updated.

**What it invalidates: nothing.** There are no noise results and no cache on disk.

### 12.2 A missed draw raises

*Your ruling: stop silencing things; errors are fine, silent errors are not.*

`noiseInject` checks what it delivered against the band its own tolerance allows and
warns. That warning could not survive the process: `find_best_hybrid.py:74` and
`_shared.py` each call `warnings.filterwarnings('ignore')` at import, and every noise
experiment imports them. Measured before the change: an injection asked for 0.5 label
units and made to deliver 1.5016 printed nothing at all.

`inject_train_labels` now catches that warning and re-raises it as `noiseInject`'s own
`DoseError`, which is what that class is exported for. Measured after: the same
sabotaged draw raises. The band is `noiseInject`'s own, so the tolerance is not
implemented a second time. Two implementations of one specification is how the two halves
of the sister study drifted apart before.

The blanket `filterwarnings('ignore')` calls are untouched. They are a wider problem
than the noise suite and every engine warning routed through `kirby._warn.KirbyWarning`
already survives them.

### 12.3 What the labels already carried is recorded on every row

*Your ruling: yes, it should be recorded.* `NOISE_DESIGN.md` §4d.

Error adds in quadrature, so injecting level L on labels already carrying `b` of their
own spread leaves `sqrt(b² + L²)`. Every noisy provenance row now carries
`baseline_error_fraction_of_spread`, `baseline_error_label_units`,
`baseline_error_source`, `total_noise_fraction_of_spread` and
`r2_ceiling_from_baseline_error`.

Three fields or none. `reference_error_as_fraction_of_spread` returns nothing where no
assay error has been published — QM9, the reaction sets, the antibody sets. There the
level is the whole of the noise. Writing a zero would assert something nobody
measured. Censoring gets the first three only; its level is a fraction of labels clipped,
so it does not add in quadrature with anything.

Measured on KIRBy's own label files:

| | clean label spread | published error | already there |
|---|---|---|---|
| logd | 1.1905 log units, n = 5039 | 0.48 (Niu et al. 2024) | 0.403 of the spread |
| herg_ki | 0.9143 pKi units, n = 1415 | 0.54 (Kramer et al. 2012) | 0.591 of the spread |

So on herg_ki the level-0.0 row is a run at 0.591 of the label spread, and level 0.15 a
run at 0.609 — a change of 3%. The R² ceiling is 0.837 on logd and 0.651 on herg_ki.

**Nothing about what is injected changed.** This records what the number means.

### 12.4 The condition is in the filename

*Your ruling: yes, and there are no results files, so it breaks nothing.*

`save_experiment_results` takes `filename_suffix`, and `run_all` takes a
`noise_condition` that reaches the output path, the resume check and the experiment
itself. Two conditions on one dataset now land on two files.

An experiment that cannot honour a condition raises rather than ignoring the keyword.
`n09`, `n11`, `n12` and `n16` take a name, and `n13` takes it as a one-entry list.
`n07` is the outlier condition by construction and `n15` is Gaussian by construction.
`n01`, `n02`, `n06` and `n14` inject nothing. Asking any of those for a condition now stops the
run instead of silently producing a Gaussian result under another name.

This is what made the escape hatch real. Before it, a second condition overwrote the
first or was skipped as already done — which is why nothing ever passed one.

### 12.5 n15 runs on the shared grid

*Your ruling: run n15 at 0.0, 0.15, 0.3.*

`NOISY_SEARCH_LEVELS` is now `(0.0, 0.15, 0.3)`, the bottom three of the grid the other
noise experiments sweep. It was `(0.05, 0.10, 0.20)`, and no value in it appeared on that
grid, so n15's answer could not be read against the experiments it exists to settle. Same
count as before — 3 levels × 5 models — so the same compute.

0.0 earns its place. At level 0.0 the labels are untouched, so both recipes see the same
values and the gap between them should be zero. It is not guaranteed to be. The two searches still
choose from lists of different lengths (§4, n15 defect 3). The 0.0 row measures that on
its own, so every other level can have it subtracted.

**Still not covered: 0.5, 0.75 and 1.0.** 0.5 is the level
`dataset_policy.reference_noise_level()` names for any experiment needing one number, so
a reading of n15 cannot be carried to it. Whether the clean-label advantage grows or
shrinks as noise rises is measured nowhere.

### 12.6 The noise-searched precompute is gone

*Closed 2026-09-11 by another session, not by me.* It built three full precomputes per
dataset and nothing read them, for the reason in §4 under n15. The block is deleted and
the reasoning stands in its place at `noise_helpers.py:378-395`. n15 still runs its own
searches cold, one per level per model; they land in the on-disk store for the next run.

### 12.7 A level sweep is now one draw, scaled

*Your ruling: roll once and multiply.*

The level used to go into the seed, so going from level 0.15 to level 0.30 threw the
first set of label errors away and rolled a fresh one. A molecule pushed up at 0.15 was
pushed down at 0.30 about half the time. Measured on 1000 labels, gaussian. The error at 0.15
and the error at 0.30 had Spearman +0.037. They pointed the same way on 500 of 1000
molecules, which is a coin flip.

What that cost, measured on ridge with 400 training molecules and 10 draws per level:

| level | mean held-out RMSE | spread over 10 draws | step from the level below | spread ÷ step |
|---|---|---|---|---|
| 0.15 | 1.1126 | 0.0551 | +0.0563 | 1.0x |
| 0.3 | 1.2561 | 0.2727 | +0.1436 | 1.9x |
| 0.5 | 1.5222 | 0.3600 | +0.2661 | 1.4x |
| 0.75 | 1.9333 | 0.5908 | +0.4110 | 1.4x |
| 1.0 | 2.3492 | 0.8390 | +0.4159 | 2.0x |

The wobble between draws was as large as the gap between levels, and sometimes twice it.

`seed` is now `selection_seed`, with no level term. Measured after the change: the error at level 0.30 is
exactly 2.0000 times the error at level 0.15 on every molecule. Spearman +1.000, same
direction on 1000 of 1000, in gaussian, student_t_nu5, laplace and outlier_p10 alike. Level 0.0 still delivers exactly zero. Two replicates at one level are
still independent of each other, Spearman +0.037, so the replicate axis still works.
Compute is unchanged and the level is still in `spec_id`, so two levels never share a
cache entry.

**What this left, and what was then done about it.** With one draw serving the whole
sweep, an unlucky draw tilts every level the same way. Repeating the corruption several
ways and averaging is the only thing that damps it. n07, n11 and n13 already did three.
n09 and n12 did one — see §12.8.

### 12.8 n09 and n12 now average three corruptions per fold

*Your ruling: do it three times.*

Both ran one corruption per fold. So every number came from a single set of random label
errors, with no way to tell how much of it was that set being kind or harsh. Both now do
three, matching n07, n11 and n13. Each fold is run three times with a different
corruption, and the scores average over folds and draws together.

Written as a list of (fold, draw) passes rather than a loop inside a loop, so nothing in
the body of either experiment moved.

Two repeats are skipped because they would be identical. Level 0.0 on a regression set
delivers exactly zero, so it runs once rather than three times. Not on a classification
set: a target flip rate of 0.0 returns a floor rather than zero there, so the three draws
really do differ. And n12's clean-label fit has no noise in it, so it is
scored once per fold rather than three times.

**Cost: about three times the noise-dependent work in each.** In n12 that is 30 feature
builds becoming 90. In n09 the hybrid search sits inside the loop too, so 30 searches
become 90, plus three times the neural training.

### 12.9 What this did to n16

`n16_importance_under_noise.py`, written 2026-09-11, rolls its own injection and gave
three reasons for not using the shared route. **All three are now gone** — the fold is
out of the seed, the amount is solved on the full column, and the level is out of the
seed.

So its private injection is now a second definition of something the shared route already
does. That is the duplication the suite removed once before. Its note is corrected and
its behaviour is untouched: moving it across is a real change to another session's file
and I have not made it. Until it moves, the two have to be kept in step by hand.

### 12.10 One failure found while checking, and it is not mine

`tests/experiments/noise/_smoke_audit_n04_n05_n13.py` stops with
`KeyError: 'lower_is_better'` inside `n13._compute_summary`. I confirmed it fails the
same way with n13 reverted to its committed state, so it predates both today's edits and
the uncommitted n13 work. Untouched.

