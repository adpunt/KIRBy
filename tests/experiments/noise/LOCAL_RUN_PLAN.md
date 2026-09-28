# Local run plan: noise mitigation experiments

Written 2026-09-27. Nothing here has been run. Every statement about an
experiment was read from its code on 2026-09-27; line numbers are given so each
can be re-checked. None of it is a result.

## What every experiment has to measure

There are two questions, and both matter equally.

1. **Recorded labels.** D is the training fold as recorded, with its
   experimental noise. A mitigation method turns D into D'. The same model is
   fitted on D and on D' and scored on the same held-out fold. If the method
   worked, D' scores better.
2. **Injected noise.** Noise is added to the training labels, a method is
   applied, and the same D-versus-D' comparison is made. The fit on the clean
   labels is scored beside them.

In code, question 1 is either level 0.0 of the noise grid or a separate pass
named `as_they_come`. `d_vs_dprime.py` writes one row per comparison with
`score_D`, `score_Dprime`, `score_clean` and `gain`.

"Repeat measurements" below means datasets that keep every lab's measurement
of a molecule (herg_ic50, bindingdb_kd). Every dataset has experimental noise.
Only these two let the code estimate how much noise each molecule carries.

## Datasets

| Dataset | Molecules | Metric | Stored hybrid searches on this machine |
|---|---|---|---|
| caco2 (molecular) | 2158 | rmse | seagate `result_db`: 5 folds, 5 models, 8 budgets, predictive and similarity |
| flab_thermo_tresanco (antibody) | 672 | spearman | seagate `result_db`: 5 folds, 5 models, 8 budgets, predictive and similarity |
| herg_ic50 (repeat measurements) | 12,008 records | rmse | none; ecfp4 only in `rep_cache` |
| bindingdb_kd (repeat measurements) | 30,000 pairs drawn | ci | 25 rows at budget 25, 2 representations; not a real search |

herg_ki and logd have 5-fold predictive searches in `~/.cache/kirby` at budgets
100 and 500 only, with no similarity searches. They are a fallback, not a
first choice.

Not verified: that the `labels_hash` in the stored caco2 and tresanco searches
matches what today's loader produces. A mismatch means the search is not found.
Step 0 below checks this.

## Which experiments answer which question

"Yes" means the code fits D and D' on the same held-out fold.

| Exp | Q1: recorded labels | Q2: injected | Line |
|---|---|---|---|
| n02 | yes (no injection at all) | no level axis | n02:348, :379 |
| n07 | yes since 2026-09-27: condition `as_they_come` | yes | n07 `AS_THEY_COME` |
| n09 | yes since 2026-09-27 | methods are compared to the clean DNN, not to D | n09:470, :1071-1080 |
| n11 | yes | yes | n11:1510 |
| n12 | yes | yes | n12:1099 |
| n13 | yes on single representations; hybrid rows are read against a single-representation DNN | Track A has no D | n13:1531 |
| n18 | yes (`as_they_come` only) | not its job | n18:132 |
| n20 | yes (level 0 and `as_they_come`) | yes | n20:467 |
| n21 | yes since 2026-09-27 | yes | n21 level grid |
| n23 | yes (level 0 and `as_they_come`) | yes | n23:1173 |
| n24 | yes | yes | n24:413 |
| n25 | yes since 2026-09-27; default is now 0.0 and 0.5, one draw | yes | n25:978 |
| n30 | only the winsorising arms; lifting and dropping need a clip limit | yes, injected clipping only | n30:663 |
| n31 | **no** — at 0 it flags as many molecules as moved, which is none | yes | n31:389, :745 |
| n32 | yes | yes | n32:1228 |
| n33 | yes, label arms only; loss swaps write no row at 0 | yes | n33:487 |
| n34 | yes (condition `none`) | yes | n34:355 |
| n35 | yes | yes, one draw per level | n35:160 |
| n39 | **no** — at 0 every simulated source equals the recorded label, so D' = D | yes | n39:470 |
| n40 | yes, but writes no D-versus-D' rows | yes | n40:536, :156 |
| n41 | yes | yes | n41:512 |
| n42 | yes, except n30 and n31 columns | yes | n42:1049 |
| n17 | yes — D is one measurement per molecule | n/a | n17:411 |
| n19 | yes — D is one measurement or the mean | n/a | n19:409 |
| n29 | yes — D is the mean of all measurements | n/a | n29:583 |
| n06 | detection only, nothing fitted on D' | n/a | |
| n22 | detection only, nothing fitted on D' | n/a | |

Level 0.0 was added to n07, n09, n21 and n25 on 2026-09-27. n13 already ran
it. n31 and n39 still do not answer question 1: at level 0.0 their design
produces no D' that differs from D.

## Faults found while reading (not fixed)

Each is two things that disagree, and how many ways.

- **n07, classification.** Each condition's file is named after that condition.
  The injection ignores the condition and always flips labels uniformly
  (n07:352). So every classification file holds the same injection.
- **n09.** Each method is fitted on the noisy labels. It is compared with a DNN
  fitted on the clean labels, not with a DNN on the same noisy labels. So the
  reported number mixes the noise damage with the method's effect.
- **n09 hybrid against its single representations.** The hybrid is
  re-searched on noisy labels, with rf, budgets 50/100/250. The single
  representations come from the clean stored search. They differ in four ways
  at once.
- **n02.** FIXED 2026-09-27: each model is now fitted on its own top 2 only.
  It used to fit every loaded representation and pick a "top 3" by the
  held-out score it reported.
- **n07.** Its single representations use its own ranking, not each model's
  top 2 (n07:2681).
- **n21.** Every model uses the first model's top 2 (n21:884-897).
- **n11, n12.** Every model's top 2 is ranked from rf's search (n11:978, n12:868).
- **n32.** FIXED 2026-09-27: a run with no condition named wrote 10 files,
  then crashed on the six `sparse_dprime_*` conditions, which it had no way to
  inject. The injection is now the one n41 uses.
- **n18, n23, n24.** The hybrid is rebuilt without the augmentation features
  the stored search may have chosen. Not checked whether any winner has them.
- **n29.** `per_record_bias` is compared with `mean_label`. It differs in two
  ways: one row per measurement, and the removed offset.
- **n30.** `xgboost_aft` is compared with plain xgboost on lifted labels. It
  differs in the model and in the labels.
- **n25.** `cliff_exempt` puts back as many molecules as the cleanup flagged,
  not as many as it actually changed.
- **Every experiment using a stored search.** The recipe was chosen on the
  clean labels of every molecule, held-out ones included. Only n21 says so.

Controls: the random-flag control (the same number of molecules, chosen at
random) is off unless `KIRBY_NOISE_CONTROLS=1` (controls.py:48). Always-on
random controls exist only in n17, n18, n20, n24 (remove only), n25, n32
and n41.

## How each run is set

- `KIRBY_CACHE_DIR=/Volumes/seagate/kirby_cache`
- `KIRBY_NOISE_CONTROLS=1`, so every cleanup has its random-flag control.
- `KIRBY_NO_COLD_SEARCH=1`, so a missing search is reported, not silently run.
- `--mode full` on every direct call. The default, `intermediary`, is one
  split (`_shared.py:8876`).
- None of: `KIRBY_LOCAL_*`, `KIRBY_SMOKE_N`, `KIRBY_SIMILARITY_SETTINGS`,
  `KIRBY_NOISE_LIGHT_ESTIMATORS`, `--models`, `--levels`, `--draws`, `--reps`,
  `--skip-*`. Each cuts the sweep.
- Preferred entry: `python tests/experiments/run_dataset.py noise <dataset>
  --experiments nXX`. It loads the dataset's full representation list and
  5 folds. A direct call loads `--reps fast`, which can pick a different top 2.
- n25 needs `--levels` with the full grid and `--noise-seeds 0,1,2`; its
  defaults are one level and one draw.
- n21 needs `--levels` including 0.0, and one run per condition.

## Order

0. **Check the stores.** Run one cheap experiment (n02) on caco2 and on
   tresanco with `KIRBY_NO_COLD_SEARCH=1`. If any model reports no hybrid, the
   store does not match today's labels and nothing else starts.
1. **Question 1 on caco2 and tresanco:** n02, n18, n11, n12, n13, n20, n23,
   n24, n32, n33, n34, n35, n40, n41, n42. n11, n12 and n34 do not run on
   tresanco (molecular or SMILES only).
2. **Question 2 on caco2 and tresanco:** the same files at every injected
   level, plus n07, n09, n21, n25, n30, n31, n39.
3. **Repeat measurements:** n17, n19, n29, then n22 and n06. These need a
   search built for herg_ic50 first (Decision 3).

## How results are reported

- Each comparison gives score_D and score_Dprime in the dataset's own metric,
  then the gain, then the random-flag control's gain beside it.
- Folds won out of 5, never a pooled mean across datasets.
- The single representations compared are named, and are each model's top 2.
- Any two arms that differ in more than one way are listed as such, not ranked.

## Decisions for you

1. **n31 and n39 at level 0.0.** Added level 0.0 would give D' = D in both, so
   it was not added. Open with the owner, 2026-09-27.
2. **The faults above.** Fix before the run, or run as-is and read around them.
   I recommend fixing n09's comparison, n32's crash and the top-2 choice in
   n02, n07, n11, n12 and n21 first. Otherwise their numbers repeat the last
   run's baseline problem.
3. **Repeat-measurement datasets.** herg_ic50 has no stored search. Either
   build one locally (a full search on 12,008 records), or run n17, n19 and
   n29 in single representations only.
4. **n30 on real censored labels.** adme_hlm has 958 of 3,087 labels at the
   assay minimum, and adme_rlm has 346 of 3,054 (n30:124). No path runs n30 on
   them yet. Add one, or leave n30 as injected-only.
