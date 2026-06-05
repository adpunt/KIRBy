# AUDIT — five new high-EV candidates (2026-05-08)

Trustii ANNITIA MASLD challenge. Five **distinct** new fusion mechanisms,
all built on existing OOFs/vals (no model retraining), each emitted as
`tests/results/annitia/full/sub_master_new_<name>.csv`.

LB scoring: `LB = 0.7 * C_hep + 0.3 * C_death`.

Implementation: `tests/experiments/annitia/new_high_ev_candidates.py`
(single script, `--variant` flag).

Pool sizes (after `_collect`): hep 119 base models, death 87.

---

## Ranked by training OOF weighted score

| Rank | Variant            | Hep OOF C | Death OOF C | Weighted | pre_check |
|-----:|--------------------|----------:|------------:|---------:|----------:|
| 1    | per_endpoint       | 0.8768    | 0.9730      | **0.9057** | PASS    |
| 2    | bma                | 0.8696    | 0.9679      | 0.8991   | PASS      |
| 3    | binwise_corrected  | 0.8665    | 0.9719      | 0.8981   | PASS      |
| 4    | cohort_split       | 0.8576    | 0.9659      | 0.8901   | PASS      |
| 5    | inference_dropout  | 0.8146    | 0.9415      | 0.8527   | PASS      |

For comparison: the current best honest fusion (`rank_mean`) scored
hep OOF 0.8980 / specialist_override death 0.9736 from `oof_diagnostics.md`.
Training OOF scores below are not directly comparable to LB — the val
distribution shifts — but they rank-order new ideas and are the only
honest signal available before submission.

---

## Per-candidate

### 1. per_endpoint  (best EV)

- **Mechanism (E.31, per-target tuning).** Build HEP risk and DEATH risk
  from **different recipes** rather than forcing a single fusion to
  serve both. HEP recipe is selected from a high-trust subset (top
  pooled-C HEP base models from `oof_diagnostics.md` — bagged RSF,
  TSFM RSF/GBSA, CatBoost-D2H, TabPFN-landmark, TabPFN25-disc,
  hep_boost RSF/GBSA family, weighted-rsf permutation variants); the
  death recipe is selected from the full death pool (87 models).
  For each endpoint we try three recipes (greedy rank-mean, top-5
  simple mean, top-7 median) and pick the one with highest OOF C.
- **Why it's a different mechanism.** Existing master variants fuse
  *both* endpoints with a single ensemble logic. Per-endpoint best
  decouples them, exploits the fact that HEP is noisier (47 events,
  lower-trust pool) so deserves a *narrower* high-trust pool, while
  DEATH (76 events, mature pool) tolerates a wider greedy search.
- **Selected recipes (both greedy_mean):**
  - HEP pool size: 34 high-trust models. Members include
    `bagged-rsf-B200`, `tsfm-gbsa`, `rsf-1000` (3-model ensemble after
    greedy stops).
  - DEATH: 5-model greedy mean over `lgbm_aft` (×2),
    `ngboost`, `cox`, `catboost_cox` on the
    `meld_na+visit_density` and `fpca+visit_density+clinical_slope`
    feature views.
- **Training OOF C:** hep 0.8768, death 0.9730 → weighted **0.9057**.
- **pre_submit_check:** PASS.
- **Expected-LB rationale.** Highest OOF EV of the five. Hep recipe
  uses ONLY high-trust hep models, so reduces overfit-to-OOF risk vs.
  recipes that include LOW-trust models. Death recipe is a small
  greedy mean (5) with a strong CV signal. The 0.7/0.3 weighting
  amplifies the hep gain. Risk: hep recipe is small (n=3), so any
  one-fold artefact in those 3 models propagates linearly.
- **File:** `tests/results/annitia/full/sub_master_new_per_endpoint.csv`

---

### 2. bma  (Bayesian model averaging)

- **Mechanism (A.7).** For each base model compute a posterior weight
  `w_i ∝ exp(beta * logit(C_i))` where `C_i` is the model's OOF
  C-index, `logit(p) = log(p/(1-p))`, and `beta=8` (sharpness — top
  model gets ~50% mass, others share remainder). Aggregate base
  models by **posterior-weighted rank average**. Endpoint-separate
  posteriors so each endpoint's best models dominate its own
  prediction.
- **Why it's distinct.** Existing meta-variants either pick a single
  best, average ranks uniformly, or stack via meta-learner (Coxnet,
  XGBRanker). BMA gives **continuous soft selection** that reflects
  posterior uncertainty over which base model is "right," rather
  than a hard pick or uniform weighting.
- **Top-5 weights:**
  - HEP: bagged-rsf-B200 (0.092), weighted-rsf-ens-linear (0.077),
    weighted-rsf-perm-rank (0.072), snapshot-rsf-N5 (0.071),
    weighted-rsf-ens-rank (0.070). HEP is diffuse — no single model
    dominates.
  - DEATH: lgbm_aft on meld_na (0.629), lgbm_aft on FPCA+slopes
    (0.154), ngboost (0.052), …. DEATH is sharp — one model carries.
- **Training OOF C:** hep 0.8696, death 0.9679 → weighted **0.8991**.
- **pre_submit_check:** PASS.
- **Expected-LB rationale.** Should generalise well — soft averaging
  over many models is robust to single-fold artefacts. Hep weight
  diffusion is a feature, not a bug, given that no hep model is
  HIGH-trust per `oof_diagnostics.md`. Risk: beta=8 may overweight
  one model on death (lgbm_aft = 63%) which could blow up if that
  model overfits the val distribution.
- **File:** `tests/results/annitia/full/sub_master_new_bma.csv`

---

### 3. binwise_corrected  (D.27 bin-wise discordance correction)

- **Mechanism.** Build a 15-model rank-mean ensemble. Bin training
  patients by their ensemble rank into 10 quantile bins. Within each
  bin, find the BASE model with the lowest mean discordance on that
  bin's members. At inference: assign val patients to bins by the
  ensemble's prediction on them, then **re-rank within each bin**
  using the bin-specialist's rank, then re-embed bin-relative ranks
  into [b/10, (b+1)/10] to preserve the ensemble's coarse ordering
  but fix within-bin order via the specialist.
- **Why it's distinct.** Existing variants route by anomaly,
  disagreement, cluster, or risk-quartile (specialist_override) — but
  none route by *prediction bin* and re-rank *within* the bin while
  preserving cross-bin order. This is closest to a *quantile residual
  correction*: trust the rank-mean ensemble for bin assignment, trust
  bin specialists for within-bin order.
- **Bin specialists (HEP, 10 bins):** mostly RSF variants
  (`bagged-rsf-B200` ×2, `snapshot-rsf-N5`, `rsf` from be ×2,
  `coxnet` ×3, `gbsa-500`).
- **Bin specialists (DEATH):** `lgbm_aft` dominates 6/10 bins,
  with `ngboost` and `rsf` filling specific bins.
- **Training OOF C:** hep 0.8665, death 0.9719 → weighted **0.8981**.
- **pre_submit_check:** PASS.
- **Expected-LB rationale.** Coarse rank from a stable ensemble +
  fine rank from a bin-specialist is a good bias/variance tradeoff.
  If val distribution differs from train, the bin specialists may
  not transfer well — but cross-bin order (the ensemble's prediction)
  should. Risk: 10 bins × ≈110 patients per bin → ≈11 patients per
  bin train support — small for picking specialists. Could be too
  noisy a routing signal.
- **File:** `tests/results/annitia/full/sub_master_new_binwise_corrected.csv`

---

### 4. cohort_split  (C.22 diabetes-cohort specialists)

- **Mechanism.** Two cohorts — diabetic (`T2DM>=1`) vs non-diabetic
  (`T2DM==0`, NaN→non-diabetic). For each cohort × endpoint, find
  the base model with the lowest mean discordance on that cohort's
  valid training subset. At inference, route val patients by their
  own T2DM cohort to the cohort-specialist.
- **Why it's distinct.** Existing routing variants (cluster_route,
  anomaly_blend) use unsupervised features. Cohort split uses a
  *clinically meaningful* binary covariate that (a) is observed on
  val, (b) is known to interact strongly with hepatic outcome and
  death in MASLD. None of the existing variants use the diabetes
  covariate explicitly.
- **Cohort sizes:** train diab/non = 465/788; val diab/non = 179/244.
- **Selected specialists:**
  - HEP-diabetic: `rsf` on `snapshot+clinical_pro+fpca+visit_density+clinical_slope`.
  - HEP-non: `rsf` on `snapshot+clinical_pro+fpca`.
  - DEATH (both cohorts): `lgbm_aft` on `meld_na+visit_density`.
- **Training OOF C:** hep 0.8576, death 0.9659 → weighted **0.8901**.
- **pre_submit_check:** PASS.
- **Expected-LB rationale.** Lowest of the four single-recipe
  variants on training OOF (because each cohort gets a smaller
  effective training set). Two reasons it could over-perform on LB:
  (a) the training/val split may have different diabetes prevalence,
  in which case single-pool models suffer from prevalence shift but
  this one doesn't; (b) MASLD literature consistently shows
  diabetes-driven HRs that are non-trivial to capture without
  splitting. Risk: 465-patient hep cohort still has only ~17 events,
  so the cohort-specialist pick is high-variance.
- **File:** `tests/results/annitia/full/sub_master_new_cohort_split.csv`

---

### 5. inference_dropout  (A.6 random-subset dropout at inference)

- **Mechanism.** At inference, for each prediction step `m=1..200`:
  randomly drop (1-k_frac) fraction of N base models, average ranks
  on the remaining `k = 0.6*N` models. Average over M=200 such
  random subsets to obtain final per-patient rank. Same procedure on
  OOFs to score honestly. Endpoint-separate (different N per
  endpoint).
- **Why it's distinct from existing `bootstrap_dropout`.** The
  existing `sub_master_bootstrap_dropout.csv` is a *training-time*
  perturbation — the base models themselves are bootstrapped. This
  variant is a *post-hoc inference-time* dropout: it perturbs the
  ENSEMBLE COMPOSITION rather than the base-model training data.
- **Configuration:** M=200, k_frac=0.6, N_hep=119, N_death=87.
- **Training OOF C:** hep 0.8146, death 0.9415 → weighted **0.8527**.
- **pre_submit_check:** PASS.
- **Expected-LB rationale.** Lowest training OOF — including all 119
  hep models drags in many LOW-trust ones (per oof_diagnostics).
  Inference dropout averages over subsets but cannot escape the fact
  that the base pool has many weak members. Reason to keep it: among
  the 5 variants this is the most theoretically robust to a *shifted*
  val distribution because it is the most ensemble-of-ensembles.
  If LB shows variance reduction matters more than peak signal, this
  beats per_endpoint. Otherwise it is the safe-but-conservative pick.
- **File:** `tests/results/annitia/full/sub_master_new_inference_dropout.csv`

---

## Submission priority recommendation

Under 0.7/0.3 weighting and ~10 submissions left:

1. **per_endpoint** (sub_master_new_per_endpoint.csv) — highest OOF EV,
   distinct decoupled-recipe mechanism, low correlation with the
   existing rank_mean / specialist_override family because the hep
   recipe explicitly restricts to high-trust models only.
2. **bma** (sub_master_new_bma.csv) — different mechanism (continuous
   posterior weighting), should be partly decorrelated from rank_mean
   because of beta=8 sharpness.
3. **binwise_corrected** (sub_master_new_binwise_corrected.csv) —
   distinct routing signal (own ensemble bin) not used by any
   existing variant.
4. **cohort_split** (sub_master_new_cohort_split.csv) — clinical
   covariate routing; complements unsupervised routing variants.
5. **inference_dropout** (sub_master_new_inference_dropout.csv) —
   submit only if at least one of the above shows LB regression and
   we suspect overfitting; this is the variance-reduction floor.

---

## Files written

- `tests/experiments/annitia/new_high_ev_candidates.py` (this script).
- `tests/results/annitia/full/sub_master_new_inference_dropout.csv`
- `tests/results/annitia/full/sub_master_new_per_endpoint.csv`
- `tests/results/annitia/full/sub_master_new_bma.csv`
- `tests/results/annitia/full/sub_master_new_cohort_split.csv`
- `tests/results/annitia/full/sub_master_new_binwise_corrected.csv`
- `tests/results/annitia/full/new_high_ev_candidates_summary.json`
  (full per-variant config + selected member labels + posterior weights).

All CSVs: `trustii_id` integer 1..423, `risk_hepatic_event`, `risk_death`,
all 423 rows, no NaN, all pre_submit_check PASS.
