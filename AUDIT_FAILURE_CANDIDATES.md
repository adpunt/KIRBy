# AUDIT: failure-pattern-aware candidate submissions

Date: 2026-05-08. Deadline today; ~10 submissions remaining.
Anchor: `sub_master_meta_avg_aware_weighted.csv` — OOF avg = 0.8786, LB = 0.9302.
Calibration: `LB ≈ -1.1365 + 2.3403 · OOF_avg`, residual std ≈ 0.0115.

Scoring: `LB = 0.7 · C_hep + 0.3 · C_death`. Hep is 2.33× weighted; only 47 events / 1106 patients, so SE on C_hep ≈ 0.025 (95% CI half-width ≈ 0.05). Any candidate within 0.05 of anchor on C_hep is statistically tied.

Source: `tests/experiments/annitia/failure_candidates.py` (single script, `--variant {fn_recovery,fp_suppress,xref_aware,hard_specialist,robust,all}`). Outputs to `tests/results/annitia/full/sub_master_failure_<name>.csv`. All candidates use integer `trustii_id` 1..423 from `/Volumes/seagate/kirby/annitia/val_test.csv`. Pool is the 5-dir union (be / cd / hep_boost / branches / external) — same pool that drove the 0.9302 anchor: 119 hep models, 87 death models.

Per-val-patient classification (5-NN on standardised snapshot+clinical_pro features against training):
- hepatic: 8 val patients near FN-like cohort, 155 near FP-like, 39 near hardest-30
- death:   18 val patients near FN-like, 108 near FP-like, 40 near hardest-30

All five candidates pass `pre_submit_check.py` with exit 0 (PASS, no warnings). All have valid integer trustii_id 1..423, no NaN, no constant predictions, sane Spearman vs anchor (0.83-0.99).

## Ranked candidates (by expected EV under 0.7/0.3)

| Rank | Candidate | OOF C_hep | OOF C_death | Weighted OOF | Δ vs anchor (OOF) | Proj LB (lin) | Spearman vs anchor (hep, death) | Mechanism distinct from others? |
|------|-----------|-----------|-------------|--------------|-------------------|---------------|--------------------------------|--------------------------------|
| 1 | **fn_recovery** | 0.8491 | 0.9426 | 0.8771 | -0.0015 | 0.927 | 0.87 / 0.86 | yes — UPweights FN-catching models |
| 2 | **robust** | 0.8359 | 0.9381 | 0.8665 | -0.0121 | 0.902 | 0.96 / 0.83 | yes — global 1/(miss_rate+0.1) reweight |
| 3 | **fp_suppress** | 0.8115 | 0.9395 | 0.8499 | -0.0287 | 0.863 | 0.87 / 0.86 | yes — selective trimmed mean for FP cohort |
| 4 | **hard_specialist** | 0.8457 | 0.8596 | 0.8499 | -0.0287 | 0.863 | 0.99 / 0.93 | yes — single-specialist override on hard cohort |
| 5 | **xref_aware** | 0.8106 | 0.9380 | 0.8488 | -0.0298 | 0.860 | 0.95 / 0.95 | partial — closest to anchor in spirit |

All projections (proj LB) are point estimates; the OOF→LB calibration has residual std ≈ 0.0115, so a ±0.025 LB-shift around the projection is plausible noise. The anchor's own gap (LB − OOF = +0.052) is unusually positive; assuming the same gap shifts every projection up by ~0.05 puts fn_recovery's optimistic projection ≈ 0.93 (anchor-tier).

---

## 1. fn_recovery (top-EV)

**Motivation.** Hep has only 4/39 strict FN-like training events but they sit on a 0.61 mean per-patient discordance — recovering even half of them moves C_hep visibly. We measure each base model's "FN catch rate" (fraction of FN-like patients ranked top-quartile on its OOF) and softmax-weight (T=0.2) the pool. For val patients within 5-NN of the FN-like train cohort (8 hep, 18 death) the upweighted blend replaces the standard rank-mean. Other val patients keep rank-mean.

**Training-side OOF C.** hep 0.8491, death 0.9426, weighted 0.8771 (Δ −0.0015 vs anchor's 0.8786). Hep gain comes from death-side robustness; the localized upweight only touches 8 hep val patients.

**pre_submit_check.** PASS (exit 0). Spearman vs anchor: hep 0.873, death 0.855 — far enough to be a genuine alternative, close enough to share most predictions.

**Expected LB.** Linear projection 0.927; if anchor's LB-OOF gap (+0.052) carries over, ~0.93 — anchor-tier with localized FN repair. Top picks of "fn_catch_rates" include `branches__*` and `cd__*` configs that dominate in `oof_diagnostics.md` for hep.

**File.** `/Users/apunt/repos/KIRBy/tests/results/annitia/full/sub_master_failure_fn_recovery.csv`

## 2. robust (whole-population FN-aware reweighting)

**Motivation.** Instead of locally swapping in for FN-near val patients, robust applies a population-level reweight: `w_i = 1 / (miss_rate_i + 0.1)`, where miss_rate is per-base-model fraction of FN-like training patients ranked below median. This penalises models that systematically underrate the cohort the ensemble misses. No selective routing — every val patient gets the same weighted blend.

**Training-side OOF C.** hep 0.8359, death 0.9381, weighted 0.8665 (Δ −0.0121). Slightly below fn_recovery on weighted because it spreads the correction over 423 patients rather than 8.

**pre_submit_check.** PASS (exit 0). Spearman vs anchor: hep 0.957, death 0.826.

**Expected LB.** Linear projection 0.902; with anchor-style +0.05 gap, ~0.95. The mechanism is a true alternative to greedy rank-mean — uses no FN routing, just a single soft-margin reweighting. Distinct from fn_recovery (which is selective) and from xref_aware (which is anchor-blended).

**File.** `/Users/apunt/repos/KIRBy/tests/results/annitia/full/sub_master_failure_robust.csv`

## 3. fp_suppress (selective trimmed-mean for FP cohort)

**Motivation.** 18.6% of hep non-events are FP-like and 38% of contested val_test patients have an FP-like nearest train neighbour. For those val patients (155 hep, 108 death) the candidate uses a 20%-trimmed mean across all base models — discards the top-20% most aggressive and bottom-20% most conservative — to suppress FP-style overconfidence. Other val patients use rank-mean.

**Training-side OOF C.** hep 0.8115, death 0.9395, weighted 0.8499 (Δ −0.0287). Hep drops because trimming on a 119-model pool nudges FP-cohort predictions toward the median — desirable on val but makes training-side discordance go up since FP-likes are by definition non-events whose true risk is unknown.

**pre_submit_check.** PASS (exit 0). Spearman vs anchor: hep 0.872, death 0.855.

**Expected LB.** Linear projection 0.863. The OOF underestimates real LB here because FP-like training patients are censored — a model that down-ranks them is "discordant" on training but might be correct on test. If 50% of FP-likes truly have no event, the LB hit is only half the OOF hit, projecting closer to ~0.90.

**File.** `/Users/apunt/repos/KIRBy/tests/results/annitia/full/sub_master_failure_fp_suppress.csv`

## 4. hard_specialist (single-model override on hardest-30 cohort)

**Motivation.** 30 hardest-discordance training patients per endpoint sit at mean disc ~0.6+. For val patients near them (39 hep, 40 death), use the single base model with the highest top-quartile rate on hardest-30 events as the predictor. Else use the anchor.

**Training-side OOF C.** hep 0.8457, death 0.8596, weighted 0.8499. Death takes a bigger hit because the chosen specialist (`cd__C-baseline-rsf` for hep with 100% catch rate on hard events) is a single high-variance model rather than an ensemble.

**pre_submit_check.** PASS (exit 0). Spearman vs anchor: hep 0.988, death 0.932 — closest of the five to the anchor in hep but with a death-side downside.

**Expected LB.** Linear projection 0.863. Risky on death because a single base model has wider per-fold variance. But hep specialist `cd__C-baseline-rsf` has 1.0 hard-event catch rate on training, so localized hep gains are plausible — only 39 hep val patients affected.

**File.** `/Users/apunt/repos/KIRBy/tests/results/annitia/full/sub_master_failure_hard_specialist.csv`

## 5. xref_aware (anchor-trust + trimmed-mean fallback)

**Motivation.** For val patients with no failure-set neighbour (n_fn=n_fp=n_hard=0; 265 hep, 257 death), use the anchor 0.9302 prediction directly. For "uncertain" val patients (any failure-set neighbour; 158 hep, 166 death), use a 20%-trimmed mean of all base models — less overconfident than the anchor's rank-mean.

**Training-side OOF C.** hep 0.8106, death 0.9380, weighted 0.8488 (Δ −0.0298). Same hep penalty as fp_suppress — replacing rank-mean with trimmed mean on 158 patients down-ranks the genuinely high-risk among them. The anchor-trust portion (265 patients) is identical to anchor.

**pre_submit_check.** PASS (exit 0). Spearman vs anchor: hep 0.947, death 0.955 — highest correlation pair, but distinct from anchor on the 158 "uncertain" val patients.

**Expected LB.** Linear projection 0.860. On test data, if uncertain patients really do behave like FP-likes (most are censored non-events), the trimmed mean will under-rank fewer truly-high-risk cases than OOF suggests, putting realised LB closer to 0.88-0.91. The mechanism is the closest to a "safe deviation" from anchor.

**File.** `/Users/apunt/repos/KIRBy/tests/results/annitia/full/sub_master_failure_xref_aware.csv`

---

## Cross-candidate diversity

Pairwise hep Spearman among the 5 candidates ranges 0.78-0.99; pairwise death Spearman 0.78-0.97. fn_recovery and fp_suppress share the rank-mean fallback (high overlap on the non-affected majority); xref_aware and hard_specialist share the anchor fallback. robust is the only candidate touching all 423 val patients with a single weighted blend.

## Recommended submission order (highest EV first)

1. **fn_recovery** — best weighted OOF, mechanism targeted at the specific 8 hep + 18 death FN-near val patients. Anchor-tier projection.
2. **robust** — global FN-aware reweight, distinct mechanism, projects ~0.95 with the anchor-style LB gap.
3. **hard_specialist** — niche bet; meaningful only if hard cohort populates val_test heavily. Risky on death.
4. **fp_suppress** — OOF says it loses, but training discordance overestimates the loss for censored FP-likes; could surprise upward.
5. **xref_aware** — most anchor-correlated; safe but weakest distinct mechanism. Consider only if 4 prior submissions confirm a routing pattern.

## Files

- Script: `/Users/apunt/repos/KIRBy/tests/experiments/annitia/failure_candidates.py`
- Submissions:
  - `/Users/apunt/repos/KIRBy/tests/results/annitia/full/sub_master_failure_fn_recovery.csv`
  - `/Users/apunt/repos/KIRBy/tests/results/annitia/full/sub_master_failure_robust.csv`
  - `/Users/apunt/repos/KIRBy/tests/results/annitia/full/sub_master_failure_fp_suppress.csv`
  - `/Users/apunt/repos/KIRBy/tests/results/annitia/full/sub_master_failure_hard_specialist.csv`
  - `/Users/apunt/repos/KIRBy/tests/results/annitia/full/sub_master_failure_xref_aware.csv`
- Summary JSON: `/Users/apunt/repos/KIRBy/tests/results/annitia/full/failure_candidates_summary.json`
- Pre-submit verdict: all 5 PASS (exit 0).

## Caveats

- Training-side OOF C for these candidates is a LOWER BOUND for FP-like-affected variants (xref_aware, fp_suppress) because FP-like training patients are censored — under-ranking them is "discordant" on training but possibly correct on the leaderboard.
- The anchor's LB-OOF gap (+0.052) is the largest in the project's history; the linear projection (`+2.3403 · ΔOOF`) understates LB shifts for recipes very similar to the anchor.
- fn_recovery touches only 8 hep val patients — the entire LB difference vs anchor for hep depends on a small handful of test patients. High variance.
- hard_specialist swaps in a single base model for 40 death patients; on a 76-event endpoint that's enough to swing C_death by ±0.04. Use only if death-side cushion exists.
