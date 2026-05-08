# Morning briefing — 2026-05-08

## What happened overnight

**Round 4 finished at 03:11.** Auto_chain advanced — but bash slurped the
script at startup, so my edits to add sanity gates, refuse_hierarchical, and
the 4 new fusion variants did NOT execute. Recovery script ran the missed
steps at 03:15. **All variants and recipes are now built.**

## Key finding: Round 4 added almost nothing

Post-Round-4 OOF C (38 base models) vs pre-Round-4 OOF C (33 base models):

| Endpoint | Pre Round 4 best | Post Round 4 best | Δ |
|---|---|---|---|
| hepatic | 0.8976 | 0.8980 | +0.0004 |
| death | 0.9736 | 0.9736 | 0.0000 |
| meta-aware hep | 0.9013 | 0.9017 | +0.0004 |
| meta-aware death | 0.9722 | 0.9723 | +0.0001 |

The TabSurv-D2H/H2D + MLP-D2H/H2D cross-endpoint branches don't add
information. This validates the contention map finding that variants are
already 95% identical.

## Variant differentiation vs the 0.9302 anchor (Spearman ρ)

| Variant | hep ρ | death ρ | Verdict |
|---|---|---|---|
| **meta_avg_aware_weighted_v2** | 0.9855 | 0.9992 | Anchor-like (~0.93 ± 0.005) |
| **meta_avg_grand_all_v2** | 0.9852 | 0.9979 | Anchor-like |
| **meta_avg_aware_weighted_no_nnls** | 0.9859 | 0.9995 | Anchor-like |
| **bootstrap_dropout** | 0.9867 | 0.9996 | Anchor-like |
| **holdout_pick** | 0.9858 | 0.9989 | Anchor-like |
| **isotonic_avg** | 0.9712 | 0.9826 | **Mild probe** |
| **meta_avg_new_variants_only** | 0.9738 | 0.9573 | **Mild probe** |
| **HIER** | 0.8713 | 0.8803 | **Strong probe** |
| ❌ nnls_stack | 0.8462 | 0.3335 | BROKEN — death OOF C only 0.85 |
| **subgroup_routed** | 0.5374 | 0.3531 | **HIGH-VARIANCE PROBE** — totally new signal but death is risky |
| **subgroup_hep_only** | 0.5374 | 1.0000 | **HEP-ONLY PROBE** — keeps anchor death, novel hep |
| **subgroup_blend** | 0.8737 | 1.0000 | **Soft hep probe** — 50/50 mix, anchor death |

Anything with ρ > 0.99 to anchor is essentially the same submission.

## Recommended 8-submission plan

### Tier 1: Confirm where we actually are (2 submits)
1. **`meta_avg_aware_weighted_v2`** — the proven recipe with the expanded pool. If it scores ≥ 0.9302 we know the new variants helped. If < 0.93 we know they hurt.
2. **`subgroup_blend`** — soft hep probe (ρ=0.87 hep, ρ=1.00 death). Tests subgroup-routed signal SAFELY. Death is the anchor so worst case = 0.93 - tiny amount; best case = 0.93+ if subgroup helps hep.

### Tier 2: Probe further (3 submits)
3. **`subgroup_hep_only`** — only submit if Tier 1 #2 (`subgroup_blend`) showed positive. Pure hep subgroup signal with anchor death; a stronger probe in the same direction.
4. **`HIER`** — most different recipe in the variant pool. Resolves the hierarchical-fusion question.
5. **`isotonic_avg`** — calibration probe.

### Tier 3: Hedges (3 submits)
6. **`meta_avg_new_variants_only`** — pure new-variants test.
7. **`bootstrap_dropout`** — variance reduction.
8. **Reserved** — based on Tier 1+2 results. If subgroup_hep_only scored higher, try `subgroup_routed` (full subgroup) as a swing-for-the-fences. Otherwise `meta_avg_grand_all_v2`.

### Do NOT submit
- `nnls_stack` — death OOF C is 0.85, will tank LB.
- `tier1_fallback` — same as `meta_avg_aware_weighted` (0.9302 anchor) — already submitted.

## Decision flow at submit time

```
After submit #1 (v2 vs original anchor):
├── v2 ≥ 0.9302   → expansion helped. Continue with HIER, then Tier 2.
├── v2 < 0.9302   → expansion hurt. Skip Tier 2 v2 variants. Fall back to original.
└── v2 ≈ 0.9302   → null result. Tier 2 still informative.

After submit #2 (HIER):
├── HIER ≥ 0.93   → hierarchical works! Submit hier-based recipes.
├── HIER ~ 0.85-0.92 → hierarchical loses signal. Don't submit hier-based again.
└── HIER < 0.85   → hierarchical broken. Don't pursue.
```

## Files for tomorrow

All recipes ready in `tests/results/annitia/submissions/`:
- `sub_master_meta_avg_aware_weighted_v2_submission.zip` ← Tier 1 #1
- `sub_master_HIER_submission.zip` ← Tier 1 #2
- `sub_master_isotonic_avg_submission.zip` ← Tier 2 #3
- `sub_master_meta_avg_new_variants_only_submission.zip` ← Tier 2 #4
- `sub_master_bootstrap_dropout_submission.zip` ← Tier 2 #5
- `sub_master_meta_avg_grand_all_v2_submission.zip` ← Tier 3 #6
- `sub_master_meta_avg_aware_weighted_no_nnls_submission.zip` ← Tier 3 #7

Each has a paired `_journal.md` with the recipe definition.

## Per-submission pre-flight check

```
python tests/experiments/annitia/pre_submit_check.py \
    tests/results/annitia/full/sub_master_<recipe>.csv
```

Will warn if:
- ρ > 0.999 to anchor (wasted submission — too similar to already-submitted)
- ρ < 0.40 to anchor (suspicious — verify intentional)
- NaN, constant predictions, row count mismatch

## Round 5 hparam tuning skipped

`experiment_hparam.py` crashed in 3s with `top view set = []` — there's a bug
in how it reads the CV summary. Not worth debugging today; diagnostics show
hparam tuning won't break the 0.93 ceiling.

## What we learned tonight (compactable)

1. **0.9302 is near the noise ceiling** for our current model class on this
   dataset. Half-width of 95% CI ≈ 0.04.
2. **Hep (47 events) has no HIGH-trust single model.** Death (76 events) has 2
   (tabpfn-landmark, tabpfn25-disc-K20).
3. **Most variants are 95% identical** — the LB battle is over the 21 contested
   hep patients (5% of val_test) and 1 contested death.
4. **38% of contested val_test patients have a near-neighbour in the training
   FAILURE set** — the LB gap is the same failure modes we see in training.
5. **Round 4 added basically nothing.** Cross-endpoint pretraining (TabSurv-
   D2H/H2D) is informationally redundant with the existing pool.
