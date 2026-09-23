# Predictions, written before the runs

⛔ WHY THIS FILE EXISTS. A result read after the fact can always be made to fit
a story. These are written down first, with the reasoning and the source, so
that when the numbers land they can only confirm or refute — not be
reinterpreted. Nothing is edited here once a run has started; an outcome is
recorded beneath its prediction, including "wrong".

Written 2026-09-23, before the method menu was extended.

Source for P1-P3: `/Users/apunt/Downloads/compass_artifact_wf-e8a9cd67-eb94-5ca5-bf28-2ecec50f9074_text_markdown.md`,
a literature review of label-noise mitigation for regression.

---

## P1. Cleaning will beat a random deletion under patterned noise, and not under symmetric noise

**The claim.** Under `grouped_shifted` (a constant offset per scaffold group)
and the heavy-tailed conditions (`student_t_nu3`, `laplace`), at least one
cleaning method will beat every same-size random deletion on most folds. Under
`gaussian` and `outlier_p10` none will.

**Why.** The document, verbatim: *"Classical additive noise ≠ bias: mean-zero
label noise mainly inflates variance; RF conditional-mean estimates stay
~unbiased. Relabeling helps most for HT, IDN, and systematic per-source bias,
and via variance reduction (weighting/shrinkage)."* Every condition this suite
has optimised against is zero-mean, which is the case it says cleaning cannot
help.

**What would refute it.** A cleaning method beating every random seed under
`gaussian`; or none beating them under `grouped_shifted` at any level.

**Outcome:** _not yet run_

---

## P2. The gain will be larger on lightgbm and dnn than on rf

**The claim.** For any cleaning method that works at all, the margin over no
cleaning will be larger on `lightgbm`, `xgboost` and `dnn` than on `rf`, on the
same folds and the same condition.

**Why.** The document: *"Random Forests are already relatively label-noise
robust by design (bagging + averaging over leaves) … it means gains from
mitigation may be modest and must be measured carefully. Boosting
(XGBoost/LightGBM/GBRT) is more outlier-sensitive and benefits more from robust
losses."* Five of the seven cleaning experiments ran on `rf` alone.

**What would refute it.** `rf` showing the largest margin, or the margins being
indistinguishable across models.

**Outcome:** _not yet run_

---

## P3. An L1 forest will take most of the gain that cleaning claims

**The claim.** A random forest with `criterion='absolute_error'`, with NO
cleaning at all, will score at least as well as the best cleaning method
applied to a squared-error forest, under the heavy-tailed conditions.

**Why.** The document lists robust losses as a complement and a baseline:
*"sklearn RF uses MSE/MAE split criteria — `criterion="absolute_error"` gives
an L1-robust forest."* Nothing in this repository reaches that criterion, so
every cleaning result so far has been measured against squared error alone.

**What would refute it.** Cleaning on a squared-error forest beating an L1
forest with no cleaning.

⛔ This is the prediction most likely to be uncomfortable. If it holds, the
honest headline is that a one-word change to the loss does what the detectors
do, and the cleaning methods have to beat THAT bar rather than the plain
baseline.

**Outcome:** _not yet run_

---

## P4. The oracle's ceiling is set by the condition, not by the dose

**The claim.** Under `outlier_p10`, the share of injected error the oracle can
remove will be about one half at EVERY level of the ladder, not rising with the
dose. Under an outlier variant whose untouched multiplier is 0 rather than 1,
it will be close to all of it.

**Why.** Measured in the injector this session, not taken from the document.
`core.py:469-473` sets the multiplier to `lam=3.0` on the selected `p` fraction
and `1.0` on the rest, so every label moves. The mean squared multiplier is
`0.10*9 + 0.90*1 = 1.8`, and the widened tenth therefore carries `0.9/1.8`
of the injected variance — exactly one half, fixed by `p` and `lam`.

By the same arithmetic the prediction for the other shares is:
`outlier_p01` gives `0.01*9/(0.01*9 + 0.99) = 8.3%`, and `outlier_p05` gives
`0.05*9/(0.05*9 + 0.95) = 32.1%`.

**What would refute it.** The oracle's share of the gap recovered rising with
the level under a fixed `p`; or `outlier_p01`'s oracle matching `outlier_p10`'s.

**Outcome:** _not yet run_

---

## P5. Exempting activity cliffs will help the label-rewriting methods and not the weighting one

**The claim.** Exempting cliff molecules will improve `knn_smooth`,
`label_smooth`, `oof_smooth` and `gp_smooth` — the methods that overwrite a
number — and will change `downweight` little or not at all.

**Why.** A molecule on a cliff has a correct label that looks wrong to any
detector comparing it with its neighbours. Rewriting that label destroys real
signal. Downweighting it only costs the information in one row, which is the
same cost as being wrong about any other molecule.

**What would refute it.** The exemption helping `downweight` as much as the
rewriting methods, which would mean the effect is about which molecules are
touched rather than about what is done to them.

**Outcome:** _not yet run_
