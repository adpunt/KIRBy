# AUDIT: Local Method Tests (D.25, E.30, E.29)

Date: 2026-05-08. Run script: `tests/experiments/annitia/method_local_tests.py`.

Anchor: `sub_master_meta_avg_aware_weighted_v2.csv` (LB = 0.9302).
Scoring: weighted_LB = 0.7 * C_hep + 0.3 * C_death.

## Summary

| method      | hep_C  | death_C | weighted_LB | n_param | runtime_sec |
|-------------|--------|---------|-------------|---------|-------------|
| conformal   | 0.8713 | 0.9652  | 0.8995      | 24      | 12.7        |
| pseudolabel | 0.8284 | 0.7518  | 0.8054      | 16      | 17.6        |
| mixup       | 0.7921 | 0.7131  | 0.7684      | 200     | 30.0        |

`hep_C` / `death_C` for conformal and mixup are honest training OOFs.
For pseudolabel, the reported number is **in-sample** (see caveat).

Spearman vs anchor on val_test:

| method      | rho_hep | rho_death |
|-------------|---------|-----------|
| conformal   | 0.862   | 0.937     |
| pseudolabel | 0.538   | 0.197     |
| mixup       | 0.533   | 0.178     |

## Method 1 — Conformal wrapper (D.25)

**Motivation.** Calibrated risk intervals from any base model. Per-model bound
width gives a model-quality signal that doesn't require comparing to ground
truth on the val set, so it should be a less leakage-prone way to weight base
models than tuning weights on the leaderboard.

**Implementation notes.**
- Picks top-12 base OOFs per endpoint by training C-index.
- Holds out 20% of valid training patients as a calibration set.
- Per-base-model residual: `|rank01(oof) - target|`, target = 1 for events,
  0 for censored. (True calibrated split-conformal would want a continuous
  target — this is a pragmatic survival adaptation.)
- Per-model width = (1-α)-quantile of residuals on calibration; α=0.1.
- Final risk = inverse-width-weighted mean of rank-transformed val predictions.
- Both OOF and val_test use the same weights, so the OOF C is honest.

**Training OOF C-index.** hep=0.8713, death=0.9652, weighted=0.8995.

**Caveats.**
- Single-shot 80/20 calibration split — could be variance-bounded by
  averaging across multiple splits, but at this scale it doesn't matter much.
- Survival "target=event" is crude; `pseudo_residual` doesn't account for
  censoring time properly.
- The width-weighted ensemble is essentially a robust top-K average — much
  of the gain is just averaging good models, not the conformal bit.

**Honest verdict — would I submit?** **Marginal yes.** OOF C ≈ 0.90 against
the anchor's known 0.93 LB. Spearman 0.86/0.94 vs anchor — close enough that
gains over the anchor are small. Worth saving as a *diversity slot* in a
final blend (e.g., 0.8 anchor + 0.2 conformal) but not as a standalone
submission. Cheap and produces a clean ensemble — fine to keep in the
candidate pool.

## Method 2 — Pseudo-labelling (E.30)

**Motivation.** Use the anchor's val_test predictions as soft targets,
self-distill into a Cox PH on training ∪ pseudo-labelled val_test (val_test
weighted 0.1×). Goal: regularise the model toward known-good ranking on the
val distribution.

**Implementation notes.**
- 8-PC features (snapshot view → standardise → SVD → first 8 PCs).
- Pseudo-events sampled from Bernoulli(p) where p = anchor risk rescaled to
  match training event rate. Pseudo-time = median train follow-up.
- Single Cox PH (lifelines, penalizer=0.05) on the weighted union.
- Weight=0.1 on val_test rows.

**In-sample C-index.** hep=0.8284, death=0.7518, weighted=0.8054.

**Caveats — IMPORTANT.**
1. **Reported C is in-sample, not OOF.** Cox PH was fitted on the same
   training rows we score. A proper OOF would require k-fold of the union
   procedure. The number is overstated.
2. **Self-distillation on the leaderboard set is trivially leaky.** The
   pseudo-labels come from the anchor (which already overfit val_test —
   anchor LB 0.9302 vs OOF in low-0.8s, ~10pt gap). We'd be teaching a
   student to mimic an overfit teacher on the exact set being scored.
3. **A Cox PH on 8 PCs cannot recover the anchor's quality.** It's a much
   weaker model. So even if the leakage helped, the architecture caps gain.
4. The death-side correlation with the anchor is 0.20 — the model does
   *not* reproduce the anchor's death ranking. So this isn't even doing
   what the method intends.

**Honest verdict — would I submit?** **No.** The method as implemented
is a weak Cox PH that mostly ignores the anchor on the death endpoint.
Even if it worked, self-distillation on the LB target with an overfit
teacher inflates val performance without generalising. If we wanted to
seriously test E.30, we would (a) use a strong base learner, (b) pseudo-
label using a rank target preserved as continuous (not Bernoulli draws),
and (c) cross-validate the union to get an honest C. As-is, do not submit.

## Method 3 — Mixup augmentation (E.29)

**Motivation.** Add 200 synthetic patients via Beta(0.4, 0.4) feature
mixing on (X_i, X_j) pairs to regularise a single Cox PH. Tests whether
density-augmentation helps a small dataset (n≈1250).

**Implementation notes.**
- Same 8-PC features as pseudolabel.
- 5-fold OOF on training (each fold: refit Cox PH on train_fold + 200
  mixup patients; predict held-out fold). This is the honest C reported.
- Final val_test prediction: refit on full train + 200 mixup, predict
  val_test.
- Mixup outcome rule: t_mix = λ*t_i + (1-λ)*t_j; e_mix = e_i OR e_j.

**Training OOF C-index (5-fold honest).** hep=0.7921, death=0.7131, weighted=0.7684.

**Caveats.**
- 8 PCs + Cox PH is a *much* weaker base than the ensemble used by the
  anchor. The 0.77 weighted C is a baseline-Cox number, not a method
  comparison.
- Survival mixup's outcome rule (OR-events) is theoretically dubious —
  it inflates the event rate in synthetic patients and biases the
  baseline hazard.
- The cleaner test of E.29 is to mixup *inside an existing strong learner*
  (e.g., the meta-blender's input space), not on a vanilla Cox PH.

**Honest verdict — would I submit?** **No.** This is testing mixup on a
weak baseline, not on the production stack. The 0.77 OOF is well below
what the anchor delivers (0.93 LB), and the val_test predictions don't
correlate with the anchor (rho 0.53/0.18) — this would damage a blend.
Mixup is worth re-testing inside the meta-stacker if compute allows; on
its own with a Cox PH it's a no-go.

## Files

- `tests/experiments/annitia/method_local_tests.py` — script
- `tests/results/annitia/full/sub_master_method_conformal.csv`
- `tests/results/annitia/full/sub_master_method_pseudolabel.csv`
- `tests/results/annitia/full/sub_master_method_mixup.csv`
- `tests/results/annitia/full/method_test_log.csv`

## Recommendation

Only `sub_master_method_conformal.csv` is plausibly worth blending into
the final submission (small weight, e.g., 0.1–0.2). Pseudo-label and
mixup as currently implemented should not be submitted; both would need
substantial re-engineering (proper OOF + stronger base) to be competitive.
