# Final 10-submission plan — 2026-05-08

**Public best so far:** `sub_master_meta_avg_aware_weighted` = **0.9302** (10 base × 1 + 5 patient-aware × 2, weighted rank-mean).

**Strategy:** Each submission must yield distinct *information* — not a small perturbation. We're rank-limited by uncertainty in OOF→test gap, so prioritise submissions that *resolve* that uncertainty over submissions that maximise expected score.

**Constraint:** ≤10 submissions, ~24h to deadline. Submission scoring takes a few minutes per submit, so cost is real but not blocking.

---

## Score-vs-information matrix

| # | Recipe | Expected score | Information gained | Risk |
|---|--------|----------------|--------------------|----- |
| 1 | `meta_avg_aware_weighted_v2` (Round 4 expanded pool, same recipe) | 0.93+ | Did the new TabSurv-D2H/H2D + NHANES branches help, hurt, or wash out? | Low — recipe is proven |
| 2 | `meta_avg_grand_all_v2` (Round 4 expanded, equal weight) | 0.93+ | Whether *equal* weighting holds up at higher pool size, or aware-weighting is essential | Low |
| 3 | Best Round-5-tuned single-model recipe | 0.92-0.94 | Whether hparam tuning recovers the OOF→test gap | Low |
| 4 | Bootstrap-fusion (random-subset dropout, M=50) | 0.93+ | Whether explicit dropout-style robustness beats deterministic averaging | Medium |
| 5 | Isotonic-calibrated meta_avg | 0.93+ | Whether calibration helps despite concordance being rank-based | Medium |
| 6 | Hold-out-fusion-test winner (10% holdout selects fusion variant) | 0.93+ | Whether fusion-method selection on a held-out slice generalises | Medium |
| 7 | Multi-seed meta-avg (seeds 42/43/44/45 averaged) | 0.93+ | Whether seed-variance is the residual gap | Low |
| 8 | Hep ← rank_median, Death ← weighted_anomaly_per_patient (per-endpoint best from refuse_meta_aware) | 0.93+ | Per-endpoint specialist vs unified recipe | Medium |
| 9 | NNLS-stacked meta (constrained-positive Coxnet replacement) | 0.92-0.94 | Whether constrained learned-weights generalise where unconstrained Coxnet failed catastrophically | High |
|10 | Hierarchical fusion v2 (post-Round-4 with pretraining gate) | 0.91-0.93 | Whether per-source pre-fusion + subjective gating beats flat at the LARGER pool size | High |

**Total expected information:** all 10 yield distinct signals about *which generalisation lever works*, not whether 0.93 → 0.93 with noise.

---

## Submission ordering (priority order)

### Tier 1 — **Confirm Round 4 paid off** (submissions 1-2)
Goal: know whether the new branches (TabSurv-D2H/H2D, NHANES-T2.x) actually contribute, or whether the 0.9302 is already at the ceiling.

1. **`meta_avg_aware_weighted_v2`** — same recipe, expanded pool. Compare to 0.9302.
2. **`meta_avg_grand_all_v2`** — equal-weight expanded pool. Compare to 0.9294.

**Decision rule:**
- Both ≥ 0.9302 → expansion helps. Continue with v2 as new baseline.
- Both < 0.9302 → expansion hurt. Revert to original pool and pursue different generalisation levers.
- Split → submit (5) isotonic-calibrated to see if calibration realigns the new branches.

### Tier 2 — **Test residual generalisation levers** (submissions 3-6)
After Tier 1, pick 2-4 of these based on what we learned:

3. **Round-5-best-recipe**: take the Optuna-tuned models, plug into best variant.
4. **Bootstrap-fusion**: M=50 random-subset dropout averaging.
5. **Isotonic-calibrated**: post-process each variant via isotonic regression, then average.
6. **Hold-out-fusion-test winner**: 10% holdout selects which variant generalises best.

### Tier 3 — **Hedges & exploration** (submissions 7-10)
Spend remaining submissions on:

7. **Multi-seed meta-avg**: rerun base models with seeds 42-45 and average.
8. **Per-endpoint specialist**: hep best ≠ death best, allow that.
9. **NNLS-stacked**: constrained-positive learned weights.
10. **Hierarchical v2**: per-source pre-fusion at the larger pool.

---

## Decision flowchart for tomorrow morning

```
START → Run Tier 1 (1, 2)
  │
  ├── Both ≥ 0.9302 → go to Tier 2 with v2 as base
  │     │
  │     ├── 5/6 = win → Tier 3 hedges, ride v2
  │     └── 5/6 = wash → maybe ceiling — Tier 3 exploration
  │
  ├── Both < 0.9302 → REVERT to original pool
  │     │
  │     └── Run isotonic + bootstrap on original pool first
  │
  └── Split → run isotonic to disambiguate
```

---

## Sanity checks BEFORE every submission

1. CSV passes `build_journal.py` validation (correct cols, no NaN, monotone ranks)
2. trustii_id matches expected count (no row drops)
3. Spot-check 5 random predictions vs sub_master_meta_avg_aware_weighted (the 0.9302 anchor) — Spearman ρ should be > 0.6 (not identical, not anti-correlated)
4. Visual check: distribution histograms not pathological (single-mode, no NaN, plausible spread)
5. OOF C of the recipe must be ≥ 0.85 hep / 0.92 death (matched 0.9302's profile)

Encoded in `pre_submit_check.py` (TODO — write).

---

## What I do NOT recommend

- **Submitting tiny perturbations** (e.g., reweighting by 0.1) — wastes a submission, no info.
- **Submitting variants that scored < 0.92 OOF** — public leaderboard punishes overfit.
- **Submitting Round 3-style learned stack** — proven to overfit catastrophically (0.7181).
- **Submitting hierarchical v1 or refuse_reselection** — both underperformed in offline tests.

---

## Time budget tomorrow

| Activity | Estimated time |
|----------|----------------|
| Wake-up status check + read `auto_chain.log` + `journal_*` | 10 min |
| Build any final variant CSVs that didn't auto-build | 30 min |
| Tier 1 (submit 1+2) | 20 min |
| Wait for scores | 30 min |
| Tier 2 (submit 3-6) | 1.5 hrs |
| Tier 3 (submit 7-10) | 1.5 hrs |

If anything goes wrong overnight (auto_chain failure, Round 4 abort), `tier1_fallback.sh` rebuilds the original 0.9302 recipe from the existing pool — that's our floor.
