# ANNITIA roadmap — single source of truth

**Deadline: today, 2026-05-08.** Currently at LB **0.9302** (#22). Top is **0.97906**. Hep is the bottleneck (47 events / 1106; current single-model best ~0.88, anchor fusion ~0.90; need ~0.96 for LB 0.97).

## Done (real artifacts on disk)

### Bug fixes that were silently capping us
1. ✅ 0.7/0.3 scoring fix → `leaderboard_history.json`, `oof_to_leaderboard.py`, `build_journal.py`
2. ✅ Round 5 hparam Optuna debugged + relaunched (PID 92102) — RSF result OOF 0.8815 (+0.013)
3. ✅ NHANES T2.x had 2 silent bugs (zero-shot trunk no-op; subset env vars not forwarded — all 5 encoders trained on same cohort). Both patched.
4. ✅ MITRA torch.amp.GradScaler shim — was a 1-line fix, not a "torch <2.4 blocker"
5. ✅ pool_dirs patched in 8 scripts to include `'hparam'`
6. ✅ Variant proxy collapse identified — 14/15 top variants share the same OOF proxy. 84 true per-variant OOFs reconstructed (file `proxy_train_oof_*.npy`).

### Local experiments completed
- `experiment_be.py` re-train with `summary` + `nit` view-sets (PID 17799) — RSF on largest combo got 0.869, not the +0.05 the audit predicted
- Manual RSF hparam tuning: best 0.8815
- Multi-seed K-fold: hep no gain, death +0.0005 free
- 84 per-variant true training OOFs reconstructed
- 5+5 load-bearing patient indices identified (`sensitivity_pair_influence.csv`)
- Mahalanobis-distance discriminator for contested val patients (AUROC 0.71)
- FN/FP discriminator: hep AUROC 0.715 (FN) / 0.899 (FP)
- 0.7181 broken submission forensic — `coefs[:,-1]` smallest-alpha overfit confirmed; broken_pattern_detector flags 44/90 variants similar
- Bayesian LB predictor — LOO MAE 0.015 — but **anchored to 0.89-0.93 prior, useless for picking 0.95+ candidates**

### 88 candidate CSVs grouped by mechanism
- Failure-pattern: failure_{fn_recovery,robust,fp_suppress,hard_specialist,xref_aware} — 5
- Deep failure: deepfn_{targeted,specialist,isolation,targeted_death} — 4
- Uncertainty: unc_{disagree_soft,specialist,bootstrap_var,mahal_blend,conformal} — 5
- Noise: noise_{censoraware,huber,classbalance,multiseed,ipcw} — 5
- High-EV new: new_{per_endpoint,bma,binwise_corrected,cohort_split,inference_dropout} — 5
- Aggressive: aggressive_{specialist_only,routed_blend,xstack,topk_only,extreme} — 5
- Repdisc: repdisc_{aware,specialized,blend} — 3
- Methods: method_{conformal,pseudolabel,mixup} — 3
- Hepspec: hepspec_v2_{tuned,pure} — 2
- Death-pure: death_{supremacy,conformal,per_horizon} — 3
- Blend-search top: blend_top1..5 — 5
- v3 rebuilds: meta_avg_{aware_weighted,grand_all,aware_weighted_no_nnls}_v3 — 3
- **NEW today**: hepmax_{specialist,pure} — 2 (sparse 7-feature Cox on max-ever NIT scores; OOF 0.855 hep / 0.800 death)

### Server SLURM jobs
- **7631111** MITRA hep+death disc — RUNNING on htc-g033
- **7631175** foundations sweep (MITRA + TabPFN-2.5 K10 hep + TabSurv + synaug + SurvTRACE) — submitted
- **7631176** NHANES T2.x 60-cell ablation — submitted on medium

---

## Outstanding — MUST DO BEFORE SUBMITTING (priority order)

### P1 — actually USE the findings (not just file them)

- [x] **Per-individual-sample drill-down** — done as `sub_master_per_patient_specialist.csv` + `AUDIT_PER_PATIENT.md`. Honest result: 0/34 contested patients had train-neighbours with hep events to compute a per-patient best-model — overrode none. Audit doc shows the patient profiles for review.
- [x] **Anchor-residual stacker** — α=0 was optimal (anchor proxy already captures linear clinical signal); negative result. CSV exists.
- [x] **Fine-Gray competing-risk Cox on hep** — pooled OOF C = 0.6247. Much WORSE than standard Cox. Spline fit unstable on 47-event data. Won't help; exclude from pool.
- [x] **Mahalanobis as ensemble weight** — `sub_master_mahal_meta.csv` written. Anchor weight ranges 0–0.7 across val patients. Plugged into meta layer not just side blend.
- [x] **Class-balanced rare-event hep** — `classbal_rsf300` OOF C = **0.8661** hep, +0.002 over best. New base added to pool.
- [x] **Conformal calibration on anchor hep** — `sub_master_anchor_conformal_hep.csv` written using A37's reconstructed proxy.
- [x] **Trend / ratio features for hep**: 16-feature sparse Cox (0.8445 hep / 0.8122 death) + RSF-300 (0.8339 hep / **0.8584 death**) — competitive with full pipeline despite using only 16 features. Added to pool.

### P2 — server jobs to monitor and integrate
- [ ] **7631242** MITRA-only resubmit (was 7631111). DEAD — val_test path bug.
- [ ] **7631243** foundations resubmit (was 7631175). DEAD — val_test path bug.
- [ ] **7631244** NHANES T2.x resubmit (was 7631176). DEAD — val_test path bug.
- [ ] **7631459** foundations re-resubmit, post env-var fix (commit 153ce24).
- [ ] **7631460** NHANES T2.x re-resubmit, post env-var fix.
- [ ] **7631461** MITRA-only re-resubmit, post env-var fix.

### P1.5 — RESPONSE TO USER FAILURE-FOCUS REQUEST (2026-05-08, post-pivot)
- [x] **Failure-focused fusion** — `sub_master_failure_focused.csv` built. Per-val-patient model weighting by K-NN (K=30) local OOF C. Avg only 10/214 hep models active per patient. Most aggressive per-patient selection yet. Mechanism distinct from anomaly_router_meta (discrete buckets), hard_specialist (single override), per_patient (failed on no-event neighbours).

### P3 — final integration when P1+P2 land
- [ ] Re-run `refuse_variants.py` + `refuse_meta_aware.py` (pool_dirs include `hparam`, will auto-pick up new OOFs)
- [ ] Build `meta_avg_aware_weighted_v4` from the expanded pool
- [ ] Re-rank all candidates by Bayesian predicted_LB (caveat: predictor anchored at 0.92 prior; use SET of top-K as shortlist, not point ranking)
- [ ] Generate final 10-submission decision tree (each submit branches on result)
- [ ] pre_submit_check sweep on top-10
- [ ] ZIP regen with real (non-boilerplate) journals

### P4 — submission day
- [ ] Submit Tier 1: most-likely-best by combined OOF + Bayesian + holdout signal
- [ ] After result lands, update Bayesian predictor with the new (submission, LB) anchor
- [ ] Re-rank, submit Tier 2
- [ ] Repeat

---

### P5 — Claude Research findings (2026-05-08)

**Key claim from research**: hep gap is dominated by post-event lab leakage
in trajectory features, not by base-model quality. Audit verified.

- [x] **Bug 2 audit** — 19/39 hep events have strictly post-event labs
  (mean 14.5 leaked obs/pt, max 119; corr(max_visit_age, event_age)=0.998).
  Death clean (0/76). Confirms research's #2 hypothesis.
- [x] Bug 1 (left truncation) **NOT** present — annitia_maslg.py uses
  `time = event_age - baseline_age`, everyone enters at t=0.
- [x] Bug 3 (CR censoring) is mild — non-hep deaths censored at
  last_followup_age (≤ death_age in practice) which is conservative.

- [x] **No-leak view experiment V1** (`experiment_research.py`):
  hep RSF-300 with summary view jumps **0.8683 → 0.9574** at delta=0.25,
  with biomarker view 0.8331 → 0.8983, both views combined 0.8656 → 0.9559.
  ⚠️ **CAVEAT**: truncation removes ~19 obs/event-patient but 0 obs/censored,
  introducing visit-count leak. The +0.09 lift is partially artifact.

- [ ] **Sanitization V2** (`experiment_research_v2.py`, RUNNING):
  - ASYM: fit on truncated, predict OOF on UNTRUNCATED features. If lift
    survives, signal is real.
  - LANDMARK: uniform truncation at baseline + tau, drop pre-tau events,
    predict events after tau. The rigorous causal-survival setup.

- [ ] **KM/NA target XGB** (`km_target_xgb.py`, RUNNING): CIBMTR-Equity
  1st-place trick — train regression on KM-survival-prob + NA-cumhaz +
  AFT + Cox heads, rank-average.

- [ ] **MASLD biomarker view** (built, in `experiment_research.py`):
  AGILE 3+, modified aMAP, FIB-4 trajectory (max/slope/AUC/time-above-2.67/
  low-to-high transition), LSM trajectory (Δ%/progressor/Baveno-VII/AUC),
  TyG, TyG-BMI, APRI, BARD-equivalent. 26 continuous features.

### Server-side fixes pushed (commit 153ce24)
- `experiment_branches.py` + `experiment_external.py` now read
  `ANNITIA_VAL_TEST` env var. Server foundations job 7631243 silently
  exited because `/Volumes/seagate/...` doesn't exist on ARC. After
  user re-pulls and resubmits, foundations + NHANES T2.x runs.

### NOT doing (per-research recommendations)
- **Sub-event hep modeling** (per-HCC/per-decompensation/per-transplant max-rank):
  not feasible — train.csv has only the binary `evenements_hepatiques_majeurs`
  composite, no sub-event metadata.
- **MOTOR-T-Base fine-tune**: gating (CITI cert) + OMOP overhead too tight
  for today.
- **TrajSurv / MENSA / Dynamic-DeepHit**: 4-8h each. Deferred unless
  V2/landmark ASYM confirms a real lift first — base-model improvements
  are dominated by the leakage fix if leakage is real.

## Won't do (out of time)
- Real Sanyal 2023 AGILE coefficients (need paper access; current proxies are good enough)
- MITRA fine-tune (T1.2) via TabTune — torch issues, low ROI given MITRA in-context already running
- Foundation-model GPU scaling — server jobs handle this
- Pseudolabel iterative training — already tested, ρ_death=0.20 said no
- Mixup augmentation — already tested, weak baseline

---

## P6 — VALIDATION METHODOLOGY OVERHAUL (added 2026-05-08, post cherry-pick failure)

**Trigger.** `cherry_pick_hep_anomBlend_death_wapp` was projected at 0.9323 with bootstrap 95% CI [0.9025, 0.9569] and 99% paired-bootstrap probability of beating anchor. Actual LB **0.89788** — outside the 95% CI lower bound, below current LB best (0.9302) by 0.032. Bootstrap CI on training OOF demonstrably fails to predict LB transfer. The OOF protocol itself is suspect.

### Diagnosis: what's wrong with the current OOF
Current setup (verified in `_oof.py` + `experiment_branches.py` + ~all variants):
- **5-fold StratifiedKFold(seed=42), stratified on event flag only.** Single split, single seed. n=1106, hep events=47 → ~9 events per fold → **fold-level C-index variance is enormous** (per-fold C ranges 0.83–0.96 in the bootstrap report).
- **OOF risks concatenated across 5 different fits**, then a single Harrell C is computed on the concatenation. Cox PH baseline hazards / tree calibrations differ per fold, so cross-fold rank comparisons are not survival-statistically valid. We are summing apples and oranges.
- **Harrell's C is biased upward under heavy censoring** (Uno 2011). We have ~96% censoring on hep — exactly the regime where Harrell's C is most optimistic and Uno's IPCW C is preferred (sksurv: "for moderate-to-high censoring, use `concordance_index_ipcw` instead of `concordance_index_censored`").
- **No optimism correction.** Harrell's bootstrap optimism correction (200 resamples) is the standard for small-sample survival prognostic models (Steyerberg / Harrell RMS). We don't do it.
- **Fusion variants fit on the same OOF used for evaluation** (anomaly_blend_meta picks specialists by per-quartile max-OOF-C; weighted_anomaly_per_patient fits per-bucket weights on OOF). This is selection-bias amplification — every parameter fit on the OOF bleeds into the C-index measurement. Caruana 2004's exact warning.
- **No nested CV.** Any meta-learner / fusion-weight fit needs an inner-fold to fit and an outer-fold to evaluate. We only have one level of OOF.
- **No held-out test set actually gating submissions.** A 10% fusion-test exists (task #61) but cherry_pick was not gated on it.
- **Discrimination only.** Brier / IBS would catch calibration failures the C-index misses.
- **No adversarial validation** between train-OOF and val_test. We don't even know if the OOF distribution is representative of the LB scoring distribution.

### Validation mechanisms to add (literature-backed)

**V1 — Repeated Stratified K-Fold (5×5 or 10×5)**
Multiple seeds (e.g. 42–46, ×5 folds) → 25 OOF predictions per patient. Average rank, report SE across repeats. Reduces single-split variance from rare events. The NHS / Royston standard for small-event clinical models. Cheap.

**V2 — Harrell bootstrap optimism correction (200 resamples)**
Standard for small-sample survival (Harrell RMS, ch. 5). For each bootstrap sample: refit, compute apparent C on bootstrap, compute C on original. Optimism = mean(apparent − orig). Corrected = apparent − optimism. Replaces single-OOF-C as primary reporting statistic.

**V3 — .632+ bootstrap estimator (Efron-Tibshirani 1997)**
Better small-sample properties than Harrell's correction *unless* heavy regularization is used. Worth running both V2 and V3 and comparing — if they disagree by >0.01, our regularization is dominating and we should trust V2.

**V4 — Uno's IPCW concordance (`concordance_index_ipcw` in sksurv)**
Censoring-corrected C. With 96% censoring on hep, Harrell's C is materially upward-biased. Uno's C is the published-paper standard for our regime. Switch primary metric.

**V5 — Integrated Brier Score (IPCW-weighted, sksurv `integrated_brier_score`)**
Calibration + discrimination. Catches models that rank well but predict probabilities at the wrong scale. A variant that improves C but worsens IBS is a red flag — it's distorting calibration to chase rank.

**V6 — Time-dependent / cumulative-dynamic AUC (Heagerty)**
Discrimination at fixed horizons (e.g. 5y, 10y, 15y). The competition target is event-by-end-of-followup, but t-AUC at multiple horizons reveals when each base model is strong. Useful for fusion: weight by horizon-specific performance.

**V7 — Adversarial validation: train-OOF vs val_test**
Train a binary classifier to discriminate OOF predictions from val_test predictions per variant. AUC ≈ 0.5 ⇒ OOF distribution is representative; AUC ≫ 0.5 ⇒ distribution shift, OOF is unsafe. The Kaggle-canonical sanity check we never ran.

**V8 — Held-out fusion-test gate (already built, not enforced)**
10% carve-out (task #61). All fusion candidates *must* beat anchor on this set, not just on OOF, to be submission-eligible.

**V9 — Nested cross-validation for fusion weights**
Outer fold = candidate evaluation. Inner fold = fit fusion weights / pick variants. Eliminates "fit weights on OOF then eval on same OOF" leak. Standard stacking protocol (mlxtend `StackingCVClassifier`, Brad Boehmke HOML ch. 15).

**V10 — Permutation null distribution for ΔC**
Shuffle predictions within candidate, recompute C 1000 times. Gives a null for ΔC between candidates. Tells us how big a ΔC has to be before it's not noise. We've been treating ΔC=0.005 as signal; permutation will probably show that's well within null.

**V11 — Cross-validated Brier residual analysis**
Per-patient IBS-residual; cluster patients by who-each-variant-misses; pick variants that miss *different* patients. Diversity-by-residual instead of correlation-by-prediction.

**V12 — Royston-Sauerbrei D / Schemper-Henderson V**
Effect-size measures for survival; complement C. Useful as a second-opinion ranking when C-index is ambiguous between candidates.

### Fusion / ensembling mechanisms to add

**F1 — Caruana bagged ensemble selection (KDD 2004)**
Library = all candidates. 20 bootstrap-bag runs of forward greedy selection *with replacement*. Sort-init = top-5 by hold-out (V8) C, not by OOF. Final weights = mean across bags. Bagging the *selection process* directly addresses the cherry-pick failure mode. Reference impl: `dclambert/pyensemble`.

**F2 — Geometric mean of rank-transformed predictions**
Rank-percentile each variant, geomean across the 10 most-decorrelated bases (chosen by mutual rank-correlation, not OOF C). Zero parameters fit; inherently robust to outlier variants.

**F3 — Hodges-Lehmann pseudomedian**
Median of pairwise means. Robust location estimator that dominates median when bases are noisy. No fitting.

**F4 — Bates-Granger variance-weighted combination (1969)**
Weights = 1/Var(error) per variant, computed on V8 hold-out (not OOF). Theoretically optimal under independence, robust under correlated errors. Only one parameter (the variance) per base; minimal selection bias.

**F5 — Stacking with regularized meta-learner + nested CV (V9)**
Logistic regression (L2 strong) or isotonic regression as meta-learner. Trained on inner-fold OOF, evaluated on outer fold. Replaces all current meta-meta variants.

**F6 — Diversity-penalized forward selection ("Getting the Most Out of Ensemble Selection", Caruana 2006)**
Forward step penalizes correlation with already-selected variants. Forces breadth, prevents collapse onto a single base family.

**F7 — Bayesian Model Averaging (reference prior)**
Weights ∝ posterior (BIC-style approximation). Less prone to selection winner's curse than max-OOF-C.

**F8 — Frequentist Model Averaging (Hjort & Claeskens 2003)**
Weights based on AIC/BIC-type criteria. Theoretical guarantees absent from greedy selection.

**F9 — Trimmed mean / winsorized mean of predictions**
Drop top/bottom q% before averaging. Robust to a single rogue variant.

**F10 — Snapshot averaging within base models**
Each base model trained 5× with different seeds, predictions averaged before entering the fusion pool. Reduces base-level variance before fusion sees it. Already partially done for RSF (multi-seed, task #88); extend to all bases.

### Action sequence for remaining 9 submissions

Premise: **no submission goes out without (a) V8 hold-out gate AND (b) at least one of V2/V3/V4 corroborating ΔC vs anchor.** Bootstrap-CI-on-OOF is no longer accepted as evidence.

1. **Switch primary metric to Uno's IPCW C** (V4). Re-rank all 108 candidates. Expect re-ordering — Harrell's C has been over-rewarding heavily-censored-favorable variants.
2. **Run V2 + V3** on top-20 Uno-C candidates. Discard any whose optimism-corrected C drops below anchor.
3. **Build F1 (Caruana bagged)** on the survivors using V8 hold-out as the selection objective. Sub A.
4. **Build F2 (geomean ranks of decorrelated 10)** independently. Sub B.
5. **Build F4 (Bates-Granger)** on V8 hold-out. Sub C.
6. **Adversarial validation (V7)** between A/B/C OOFs and val_test. Drop any with AUC > 0.6.
7. Submit A, B, C in that order with an LB check between each. If A loses to anchor, abort B/C — the OOF→LB transfer is broken at a deeper level than fusion can fix.
8. If A wins by ≥0.01, proceed with F5 (proper stacking) + F6 (diversity-penalized) over the same survivor pool. Subs D/E.
9. Hold ≥2 submissions in reserve in case server foundation jobs land and contribute new diverse bases.

### What NOT to do (post-mortem rules)
- ❌ Submit any variant whose construction fits parameters on the OOF used to score it (kills all current meta-meta variants: anomaly_blend_meta, weighted_anomaly_per_patient, anomaly_router_meta, disagreement_router_meta, cluster_router_meta).
- ❌ Cherry-pick the per-endpoint OOF winners. The cherry_pick failure proved this is dominated by selection winner's curse.
- ❌ Use Harrell's C as the only metric.
- ❌ Use single-seed StratifiedKFold OOF C as a transfer estimator without optimism correction.
- ❌ Trust bootstrap CI on the same OOF used to construct candidates.
- ❌ Add new candidates to the pool without a V8 hold-out gate.

### Sources
- [Caruana et al., "Ensemble Selection from Libraries of Models", ICML 2004](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf)
- [Caruana et al., "Getting the Most Out of Ensemble Selection", ICDM 2006](http://www.niculescu-mizil.org/papers/enssel_most_long.pdf)
- [Niculescu-Mizil et al., "Winning the KDD Cup Orange Challenge with Ensemble Selection", 2009](http://proceedings.mlr.press/v7/niculescu09/niculescu09.pdf)
- [Uno et al., "On the C-statistics for Evaluating Overall Adequacy of Risk Prediction Procedures with Censored Survival Data", Stat Med 2011](https://www.researchgate.net/publication/51043571_On_the_C-statistics_for_Evaluating_Overall_Adequacy_of_Risk_Prediction_Procedures_with_Censored_Survival_Data)
- [Iba et al., "Re-evaluation of comparative effectiveness of bootstrap-based optimism correction methods", BMC Med Res Methodol 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7789544/)
- [scikit-survival evaluation guide (Uno IPCW vs Harrell)](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html)
- [Pölsterl, "Evaluating Survival Models" (k-d-w blog)](https://k-d-w.org/blog/2019/05/evaluating-survival-models/)
- [Pyensemble — Caruana ensemble selection in scikit-learn (dclambert)](https://github.com/dclambert/pyensemble)
- [Adversarial Validation overview (KDnuggets / Zak Jost)](https://www.kdnuggets.com/2020/02/adversarial-validation-overview.html)
- [Hands-On Machine Learning with R, ch. 15: Stacking](https://bradleyboehmke.github.io/HOML/stacking.html)
- [Subasi et al., "Stop Chasing the C-index", arXiv:2506.02075](https://arxiv.org/html/2506.02075v1)
