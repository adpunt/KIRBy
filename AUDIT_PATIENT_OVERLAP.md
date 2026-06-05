# AUDIT — Patient Overlap (load-bearing × contested × FN/FP)

Date: 2026-05-08
Inputs: `AUDIT_SENSITIVITY` (5+5 load-bearing train indices), `valtest_contention_*.csv`
(top-21 hepatic + top-1 death contested val patients), `deep_failure_audit.json`
(5 FN + top-30 FP training patients).
Feature space: snapshot + clinical_pro views (58 dims after merge), median-imputed,
standardised on combined train ∪ val_test (n_train=1253, n_val=423).

## Headline

The three patient sets do NOT overlap. **None** of the 10 load-bearing training
patients appear in the 5-NN of any of the 22 contested val_test patients. They are
not in the FN list and not in the FP top-30 either. A naive "specialist sub-ensemble
targeted at LB-style patients" will not move the contested val_test patients,
because LB patients are not near them.

## Key numbers

| Quantity | Observed | Expected (random) | p |
|---|---|---|---|
| LB hits in 5-NN of contested val (110 draws) | 0 | ≈ 0.88 | 1.0 (binomial upper tail) |
| Contested val queries with ≥1 LB neighbour | 0 / 22 | — | — |
| LB ∩ FN (hepatic) | ∅ | — | — |
| LB ∩ FP top-30 (hepatic) | ∅ | — | — |

Stronger statement — median rank (out of 1253) of each LB patient as neighbour of
the 22 contested val_test patients (lower = closer; 626 = random):

```
LB (overall)            median rank 802   %top-5 = 0.0%
hepatic FN  (n=5)       median rank 531   %top-5 = 0.0%
hepatic FP top-30       median rank 738   %top-5 = 0.2%
```

Per LB patient:

```
77   (hep)    median 1207   min 1064   max 1245   → near boundary, FAR from contested
627  (hep)    median 1092   min 1036   max 1241   → FAR
851  (death)  median 1252   min 1248   max 1253   → near singleton, ESSENTIALLY MAX
650  (hep)    median  967   min  129   max 1105
591  (hep)    median  812
728  (death)  median  743
454  (death)  median  683
178  (hep)    median  535
293  (death)  median  360   min   52
129  (death)  median  353   min   57   → closest to contested, but still not top-5
```

LB patients are not just "not over-represented" near contested val — they are
systematically *under*-represented. Three of them (77, 627, 851) sit so far from
the contested val cohort that they look more like distant outliers in 58-D snapshot
space than nearby support.

## Profiles of the 10 load-bearing patients

```
idx 77   hep+ at t=3, death+ at t=4 (rapid double event), baseline 63, fu 1y
idx 178  hep+ at t=2 (long fu, censored death t=8), baseline 55, fu 8y
idx 591  hep+ at t=3, death- t=5, baseline 62, fu 5y
idx 627  hep+ at t=2, death- t=2 (very short), baseline 64, fu 2y
idx 650  hep+ at t=1 (very early hep), death- t=6, baseline 62, fu 6y
idx 129  hep- t=1, death+ at t=3, baseline 54, fu 1y (early death no hep)
idx 293  hep- t=2, death+ at t=4, baseline 72, fu 2y
idx 454  hep- t=3, death+ at t=4, baseline 79, fu 3y
idx 728  hep- t=2, death+ at t=3, baseline 73, fu 2y
idx 851  hep+ t=3, death+ t=4, baseline 70, fu 2y
```

Pattern: load-bearing = patients with **events at short follow-up**, often the
short-tenure event-positives the Cox/AFT objectives lean on hardest. None overlap
with the FN/FP profiles from the discriminator audit, which target *uneventful*
short-fu (FP) or *late-event long-fu* (FN) regions.

So the three sets describe three different failure modes:
1. **LB train patients** = early-event, short-fu (training leverage points).
2. **FN train** = late-event, long-fu, mild biomarkers (model misses positives).
3. **FP train** = no event, short-fu, but high stiffness/age (model false-alarms).
4. **Contested val_test** = whatever validation neighbourhood disagrees on; in
   feature space it sits in a different region than any of the train sets above.

## Files

- `tests/results/annitia/full/loadbearing_contested_overlap.csv` — per-contested-val
  patient: 5-NN train indices, # of LB hits, hypergeometric p.
- `tests/results/annitia/full/loadbearing_profiles.csv` — LB profile + bucket flags
  + n contested-NN hits.
- `tests/results/annitia/full/loadbearing_overlap_diag.json` — summary stats.
- `tests/results/annitia/full/patient_overlap.png` — PCA(snapshot+clinical_pro)
  scatter; LB (gold stars) sit at the periphery, contested val (green diamonds)
  cluster centrally; FN (red X) and FP (orange) populate yet another region.

## Implication for ensembling

The first-pass plan in the prompt — *target LB patient features in val_test with
a specialist sub-ensemble* — is **not supported** by this evidence. The val_test
patients that move on rep changes are not LB-like, so weighting up an LB-favouring
sub-ensemble on those val rows would not be on-distribution.

### Revised candidate construction (proposal only — not implemented)

Three more useful experiments, in order of expected payoff:

1. **Contested-region specialist (data-driven, not LB-driven).** Fit a kNN router
   on the 58-D snapshot+clinical_pro space. For each contested val_test patient,
   find its actual 5-NN training neighbours (which we now have in the CSV) and
   weight a sub-ensemble that scores highest on those specific train neighbours.
   This is a *retrieval-conditioned blend* rather than an LB-anchored one.

2. **Three-bucket model selection.** Use the discriminator's FP/FN buckets plus
   the LB cohort to build a 3-way router on val_test rows: (a) FP-like, (b) FN-like,
   (c) LB-like, (d) generic. Each bucket gets the variant that scored best on
   *that bucket's training rows in OOF* — not on the global C-index. The headline
   contention CSVs already give you the per-row IQR; combining row-bucket × variant-
   per-bucket is closer to the real signal.

3. **Stop optimising for LB removal.** AUDIT_SENSITIVITY tells us *which patients
   the model is leaning on*, not which patients matter at val_test time. The
   present analysis shows those are largely disjoint sets. Robustness to LB
   removal is interesting for stability, but it is not a leaderboard lever.

If we want to use LB at all, the most defensible move is to *down-weight*
LB-adjacent training patients during stacker fitting (a leverage-aware ridge),
not to *up-weight* an LB-style sub-ensemble at inference.

## Caveats

- Only 10 LB patients × 22 contested val patients — small sample, could miss
  weak signals. Binomial p≈1 with 110 draws is, however, decisive that there is
  no strong overlap to find.
- 58-D snapshot+clinical_pro is one feature view. The discriminator audit used a
  similar but not identical feature set; the contention CSVs are model-output
  IQR, not feature-space distances. Other reps (longitudinal, fpca, gp_smooth)
  could show different geometry — but we already know those reps disagree, which
  is exactly what makes those val patients contested.
- Median-imputation flattens missingness signal; LB and contested val both have
  high missingness in NIT columns, so the imputation noise is mutual.
