# Morning briefing — 2026-05-08 (REVISED 03:55 after sanity sweep)

## TL;DR — Go/No-Go table

```
Order Recipe                              Tier     Verdict
   1. meta_avg_aware_weighted_v2          Tier 1   WARN (ρ≈1.00 to anchor — confirmation)
   2. HIER                                Tier 2   PASS
   3. isotonic_avg                        Tier 2   PASS
   4. meta_avg_grand_all_v2               Tier 2   PASS
   5. meta_avg_new_variants_only          Tier 3   PASS
   6. bootstrap_dropout                   Tier 3   WARN (ρ≈1.00 — variance reduction)
   7. meta_avg_aware_weighted_no_nnls     Tier 4   WARN (ρ≈1.00 — no-nnls)
   8. holdout_pick                        Tier 4   PASS

DO NOT SUBMIT (would tank macro, despite looking "novel"):
   ❌ subgroup_routed     OOF hep 0.76 / death 0.71
   ❌ subgroup_hep_only   expected macro ~0.88
   ❌ subgroup_blend      expected macro ~0.89-0.90
   ❌ nnls_stack          death OOF C 0.85
```

WARN means "very high correlation to known-good anchor" — informational
only; safe to ship. FAIL would block submission. None of the candidates
fail.

## Pre-submit hard checks (run for every CSV)

A pre_submit_check sweep on all 50 sub_master_*.csv files caught **7 broken
submissions** that would have wasted submissions if shipped:

- `sub_master_subgroup_routed.csv` had GUIDs in the trustii_id column
  (would map every prediction to the wrong patient → catastrophic).
  **FIXED** + ZIP rebuilt.
- 6 older variants with the same GUID-leak (`anomaly_blend_meta`,
  `anomaly_router_meta`, `cluster_router_meta`,
  `coxnet_stack_with_uncertainty`, `disagreement_router_meta`,
  `weighted_anomaly_per_patient`). **FIXED**. None are in the recommended
  plan but they were in the contention pool — re-run any analysis that
  joined on trustii_id.

After fix: all 50 CSVs pass structural audit. All 12 candidate ZIPs match
their latest CSVs (md5 checked).

## ⚠️ Reality check on the subgroup variants

Last night's brief overstated subgroup_blend / subgroup_hep_only /
subgroup_routed. Real OOF C of the cluster-routed Cox (snapshot +
clinical_pro features, GMM k=6, per-cluster Cox PH):

| | hep OOF-C | death OOF-C |
|---|---|---|
| subgroup_routed (raw) | **0.7638** | **0.7150** |
| Global Cox same features | 0.8401 | 0.7439 |
| Existing ensemble pool best | 0.8980 | 0.9736 |

The "novel signal" is just **lower-quality predictions** that look
different from the anchor because they're worse, not because they carry
new information. Submitting any subgroup variant raw is expected-negative.

The journal files for `subgroup_blend` and `subgroup_hep_only` claim OOF
0.8980/0.9736 — **those are the anchor's numbers, auto-generated
boilerplate**. Do not trust journal text on these three submissions.

## Recommended 8-submission plan (revised)

### Tier 1: Anchor sanity (1 submit)
1. **`meta_avg_aware_weighted_v2`** — proven recipe rebuilt with the
   expanded post-Round-4 pool. ρ=0.99 to anchor, expected ~0.9302 ± noise.
   Pure confirmation; reveals whether the new variants helped.

### Tier 2: Architecture probes (3 submits)
2. **`HIER`** — hierarchical fusion. Most-different recipe in the
   *quality-controlled* pool (ρ=0.87 hep / 0.88 death to anchor). Real
   recipe shift, NOT a downgraded prediction.
3. **`isotonic_avg`** — calibration probe (ρ=0.97 / 0.98).
4. **`meta_avg_grand_all_v2`** — broad-pool average. Fully passes
   pre_submit_check, ρ=0.99 to anchor — incremental.

### Tier 3: New-variant test + variance reduction (2 submits)
5. **`meta_avg_new_variants_only`** — pure post-Round-4 variants (ρ=0.97
   hep / 0.96 death). If positive, post-Round-4 work added value.
6. **`bootstrap_dropout`** — variance reduction via random subset
   averaging. ρ=0.99 to anchor.

### Tier 4: Reserved (2 submits)
7. **`meta_avg_aware_weighted_no_nnls`** — no-nnls variant (ρ=0.99/1.00).
8. **`holdout_pick`** — stability-based picker (passes all checks).

### DO NOT SUBMIT (revised)
- ❌ **`subgroup_routed`** — raw cluster-routed Cox; OOF hep 0.76,
  death 0.71. Expected LB collapse.
- ❌ **`subgroup_hep_only`** — anchor death + raw routed hep. Expected hep
  LB ~0.83, drops macro ~0.05 below 0.9302.
- ❌ **`subgroup_blend`** — 50/50 anchor+routed rank mean. Best estimate
  macro ~0.89-0.90; below 0.9302 floor. Skip unless you accept negative EV.
- ❌ `nnls_stack` — death OOF C only 0.85.
- ❌ `tier1_fallback` — identical to `meta_avg_aware_weighted` (already
  used as anchor for the 0.9302 LB).

If you want the subgroup work to count for something, treat it as **a
diagnostic**, not a submission: 5 val patients are beyond train
Mahalanobis p99, and cluster 2 (97 train / 1 val) is a real distribution-
shift signal.

## Decision flow at submit time

```
After submit #1 (meta_avg_aware_weighted_v2):
├── ≥ 0.9302   → expanded pool helped. Continue Tier 2.
├── < 0.9302   → expanded pool noisy. Consider re-anchoring on
│                meta_avg_aware_weighted (the proven 0.9302 recipe).
└── ≈ 0.9302   → null. Tier 2 still informative.

After submit #2 (HIER):
├── ≥ 0.93     → hierarchical adds. Submit hier-flavoured recipes if any.
├── 0.85–0.92  → moderate signal loss. Don't escalate hier.
└── < 0.85     → hier broken. Don't pursue.

If 4 of 6 Tier 1-3 submissions land below 0.9302:
→ Stop probing. Submit `meta_avg_aware_weighted` exactly (proven 0.9302).
  Better to repeat than to bleed.
```

## Files

All recipes ready in `tests/results/annitia/submissions/`:

- `sub_master_meta_avg_aware_weighted_v2_submission.zip` ← Tier 1 #1
- `sub_master_HIER_submission.zip` ← Tier 2 #2
- `sub_master_isotonic_avg_submission.zip` ← Tier 2 #3
- `sub_master_meta_avg_grand_all_v2_submission.zip` ← Tier 2 #4
- `sub_master_meta_avg_new_variants_only_submission.zip` ← Tier 3 #5
- `sub_master_bootstrap_dropout_submission.zip` ← Tier 3 #6
- `sub_master_meta_avg_aware_weighted_no_nnls_submission.zip` ← Tier 4 #7
- `sub_master_holdout_pick_submission.zip` ← Tier 4 #8

Each has a paired `_journal.md` — note that the journals are
auto-generated from a shared template; trust the CSV + this briefing
over the journal narrative.

## Per-submission pre-flight (REQUIRED)

```
python tests/experiments/annitia/pre_submit_check.py \
    tests/results/annitia/full/sub_master_<recipe>.csv
```

Will exit non-zero if: column names wrong, NaN, trustii_id ordering off
(THIS would have caught last night's GUID bug), constant predictions,
inf, or row count mismatch with the proven 0.9302 anchor.

## What we learned overnight

1. **0.9302 is near the noise ceiling** for the current model class. SE_C
   ≈ 0.025 with 47 hep events.
2. **Hep (47 events) has no HIGH-trust single model.** Death (76 events)
   has 2 (`tabpfn-landmark` CI[0.906,0.956], `tabpfn25-disc-K20`
   CI[0.857,0.923]).
3. **Most variants are 95% identical** — the LB battle is over the 21
   contested hep patients (5%) and 1 contested death.
4. **38% of contested val_test patients have a near-neighbour in the
   training failure set** — same failure modes.
5. **Round 4 cross-endpoint pretraining added basically nothing** —
   variants are informationally redundant with the existing pool.
6. **Subgroup-routed Cox is worse, not different** — diagnostic value only;
   not a submission. (Corrected, 03:55.)
7. **7 CSVs had a silent GUID-vs-int trustii_id bug** — caught by
   pre_submit_check before submission. ALWAYS run the check.

## Round 5 hparam tuning skipped

`experiment_hparam.py` crashed in 3s with `top view set = []`. Bug in CV
summary read. Not worth debugging today — diagnostics show hparam tuning
won't break the 0.93 ceiling.
