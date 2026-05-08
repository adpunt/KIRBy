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
- [ ] **7631242** MITRA-only resubmit (was 7631111, silent fail — autogluon missing in venv). **Status: submitted after install-at-start fix.**
- [ ] **7631243** foundations resubmit (was 7631175). Should produce MITRA + TabSurv + TabPFN-2.5 K10 hep + synaug + SurvTRACE.
- [ ] **7631244** NHANES T2.x resubmit (was 7631176). 60-cell ablation now wired correctly.

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

## Won't do (out of time)
- Real Sanyal 2023 AGILE coefficients (need paper access; current proxies are good enough)
- XGBoost-survival wrapper + RepeatedStratifiedKFold (~3h, marginal expected lift)
- MITRA fine-tune (T1.2) via TabTune — torch issues, low ROI given MITRA in-context already running
- Foundation-model GPU scaling — server jobs handle this
- Pseudolabel iterative training — already tested, ρ_death=0.20 said no
- Mixup augmentation — already tested, weak baseline
