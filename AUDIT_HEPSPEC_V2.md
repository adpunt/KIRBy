# AUDIT — sub_master_hepspec_v2 candidates

Date: 2026-05-08  •  Author: hepspec_v2 build


## File paths
- Candidate 1 (tuned-blend): `/Users/apunt/repos/KIRBy/tests/results/annitia/full/sub_master_hepspec_v2_tuned.csv`
- Candidate 2 (pure-tuned): `/Users/apunt/repos/KIRBy/tests/results/annitia/full/sub_master_hepspec_v2_pure.csv`
- Anchor (death source): `/Users/apunt/repos/KIRBy/tests/results/annitia/full/sub_master_meta_avg_aware_weighted.csv`
- Tuned RSF OOF: `tests/results/annitia/branches/oof_hepatic_hparam_rsf__snapshot+clinical_pro+fpca+gp_smooth.npy`
- Tuned RSF val_test: `tests/results/annitia/hparam/val_hepatic_hparam-rsf__snapshot+clinical_pro+fpca+gp_smooth.npy`

### Provenance note (important)
The val_test prediction file matching `branches/oof_hepatic_hparam_rsf__...npy`
does NOT exist at `branches/val_hepatic_hparam_rsf__...npy` or under `hep_boost/`.
The closest available pair is in `tests/results/annitia/hparam/`
(`oof_hepatic_hparam-rsf__...` and `val_hepatic_hparam-rsf__...`). The two OOFs are
Spearman ρ ≈ 0.97 with each other and have OOF C = 0.8815 vs 0.8819 — essentially
the same fit on the same view set. We use `hparam/val_hepatic_hparam-rsf__...` as
the val_test predictions for both candidates. If you want a strictly matching pair,
the hparam-rsf OOF (C=0.8819) should be used instead of the branches OOF.

## OOF C per candidate (training side)
| metric | C1 (tuned-blend) | C2 (pure-tuned) | anchor |
|---|---|---|---|
| hep OOF C | ~0.8881* | 0.8815 | 0.9036 |
| death OOF C | 0.9722† | 0.9722† | 0.9722 |
| weighted (0.7/0.3) | ~0.9133* | 0.9087 | 0.9242 |

\* C1 hep OOF is a proxy: the rank-blended hep score has no training-row OOF on disk
(the anchor ensemble OOF predictions were not persisted). The proxy reported is the
linear combination 0.7·C(rsf)+0.3·C(anchor)=0.7·0.8815 + 0.3·0.9036, which **lower-bounds**
the actual blend C if rsf and anchor disagree on hard pairs (typical case → real C is
between proxy and anchor).

† Death predictions are identical to the anchor (anchor.risk_death copied verbatim),
so death OOF C inherits the anchor value.

### Anchor OOF reference (refuse_meta_aware weighted_anomaly_per_patient)
- hep OOF C = 0.9036
- death OOF C = 0.9722
- weighted (0.7/0.3) = 0.9242

## Pre_submit_check verdict
- C1: **WARN** (exit 1)
```
=== pre_submit_check: sub_master_hepspec_v2_tuned.csv ===
WARNINGS (review):
  [WARN] risk_death ~identical to anchor (rho=1.0000) — wasted submission?
  anchor: sub_master_meta_avg_aware_weighted
  rows: 423
```
- C2: **WARN** (exit 1)
```
=== pre_submit_check: sub_master_hepspec_v2_pure.csv ===
WARNINGS (review):
  [WARN] risk_death ~identical to anchor (rho=1.0000) — wasted submission?
  anchor: sub_master_meta_avg_aware_weighted
  rows: 423
```

## Spearman ρ vs anchor
| candidate | hep ρ | death ρ |
|---|---|---|
| C1 (tuned-blend) | 0.9178 | 1.0000 |
| C2 (pure-tuned)  | 0.8399 | 1.0000 |

## Honest verdict — would I submit this?

**No, not as a blind LB-burn replacement of the 0.9302 anchor.**

Reasoning:
1. The tuned RSF hep OOF C = **0.8815** is *higher than any single base hep model* but
   it is **lower** than the anchor ensemble's hep OOF C = **0.9036**.
   The user's memo framed the +0.013 improvement against the next-best *single base*
   (synaug-none-rsf 0.868); against the anchor it is **−0.022**.
2. Naive arithmetic at 0.7 weight gives ΔLB ≈ 0.7 × (−0.022) ≈ **−0.015** for C2 (pure).
   For C1 (0.7·tuned + 0.3·anchor), the rank-blend partially recovers, but per Pareto
   the best *expected* outcome is bounded above by max(C(rsf), C(anchor)) for the hep
   half — i.e., we are ≤ anchor performance on hep with 0.3 anchor mass.
3. The pre_submit_check is the safety net (block constants/anti-correlated/etc.), not
   a fitness verdict. Both pass the bar.
4. Hep has only 47 events on the LB; the OOF gap could reverse on the LB by ±0.02
   easily — but it could also widen. Submitting the pure-tuned version is a directional
   bet that the *anchor is overfit to OOF* and the tuned RSF is more general. Without
   evidence (e.g., bootstrap CI, holdout slice) that supports this, the prior favors the
   anchor.

Recommended use:
- Keep C1 (tuned-blend) as a low-variance hedge: +30% anchor mass on hep, anchor death
  unchanged. Worst-case downside is small (~−0.005), best case marginal upside if the
  anchor was over-blended.
- Keep C2 (pure-tuned) only as a diagnostic comparator. Do NOT submit unless the LB
  budget is large or there's an explicit "explore" submission this round.
- The user's own framing (+0.013 vs single-base → +0.005-0.008 expected after dilution)
  applies *only relative to the single-base baseline*, not relative to the 0.9302 anchor.
  Against the anchor, both candidates are expected to lose mean LB.

## Sanity numbers
- val_rsf range: [0.131, 8.106], n=423
- C1 hep range: [0.000, 0.997]; std=0.279
- C2 hep range: [0.000, 1.000]; std=0.289
- death range: [0.004, 0.991]; std=0.287
