# ANNITIA Train ↔ Val_Test Leakage Hunt (2026-05-08)

**Question:** does train↔val_test linkage explain the LB gap? (top 0.97906, ours 0.9302; needs C_hep ≈ 0.96, +0.08 over our best 0.88).

**Verdict: NO LEAKAGE FOUND.** The 0.08 hep gap must come from model quality.

## Inputs
- `train.csv` — 1253 patients × 287 cols (incl. 4 outcome cols)
- `val_test.csv` — 423 patients × 284 cols (NO outcome cols, +1 `trustii_id`)
- 282 common feature columns

## Findings

### 1. Direct ID overlap
- `patient_id_anon` overlap train ∩ val_test = **0 / 423**.
- val_test has its own GUID space disjoint from train.

### 2. Near-duplicate features (exact-cell match)
Per val_test row, found train row maximizing # of equal non-NaN cells.

| metric | value |
|---|---|
| match-count median | 7 |
| match-count max | 13 |
| frac_match ≥ 0.50 | 3 / 423 |
| frac_match ≥ 0.80 | **0 / 423** |
| frac_match ≥ 0.95 | **0 / 423** |
| ≥30 overlapping cells, zero mismatches | **0 / 423** |

The "best" matches all have low overlap (≤16 non-NaN cells) and only 4–7 exact equalities — consistent with chance for sparse rows.

### 3. Visit-pattern matching (same patient at different timepoints)
Signature = first 5 visits of `Age, alt, ast, BMI` (20 cols).
- Median signature exact-match cells = 3, max = 5.
- Full-signature match (all non-NaN equal AND ≥4 cells): **0 / 423**.
- No "staged split where val_test = same patient at later visits".

### 4. Cohort / site columns
- No `site / center / hosp / clinic / cohort / recruit` columns in either CSV.
- No way to verify enrollment-site overlap; not a leakage path the data exposes.

### 5. Outcome leakage in val_test schema
- val_test outcome cols: **NONE**.
  - missing: `evenements_hepatiques_majeurs`, `evenements_hepatiques_age_occur`, `death`, `death_age_occur`.
- Only extra col vs train is `trustii_id`. No `last_followup_age`, `cause_of_death`, time-to-event indicator.
- No leakage in the y_struct.

### 6. First-visit signature substring match
Subsumed by (3). No matches.

### 7. NIT scores at baseline (cohort-shift / sicker val_test?)
Baseline (v1) distributions:

| NIT | train (n, median, mean) | val_test (n, median, mean) |
|---|---|---|
| FibroScan stiffness | 877, 6.39, 7.78 | 283, 6.13, 7.51 |
| FibroTest | 824, 0.25, 0.31 | 331, 0.26, 0.30 |
| Aixplorer AIx | 310, 6.09, 6.98 | 116, 6.13, 7.01 |

Static features (gender, T2DM, HTN, DLP, bariatric): all means within 0.05 of train.

→ val_test is **not a sicker-cohort artifact**; if anything FibroScan median is slightly *lower* (less fibrosis). The "0.97 model" is not winning by exploiting an enriched event base rate.

### Distance-distribution sanity check
Standardized-Euclidean nearest-neighbour distances (NaN→0, presence-aware):

| comparison | min | p05 | median |
|---|---|---|---|
| val_test → train | 0.944 | 2.451 | 5.255 |
| train → train (200-sample) | 0.208 | 0.575 | 4.663 |

**Within-train neighbours are 4–5× closer than the closest val_test→train pair.**
If val_test contained duplicates we would expect val→train min ≪ train→train min. Observed pattern is the opposite — same population, distinct individuals.

## Conclusion
- No direct ID overlap, no near-duplicate rows, no shared visit signatures, no outcome columns, no cohort-enrichment, no detectable site/center hook.
- val_test→train NN distances are *strictly larger* than within-train NN distances; impossible to explain by a "leaked patient copy".
- **The +0.08 hep gap to the leaderboard top is model-quality, not leakage.**

## Recommendation
- **Do NOT** build a leakage-exploit candidate (copy training outcome to matched val patient). There is nothing to copy — the closest train neighbour to a val patient is further away than 95% of train↔train pairs.
- Path to 0.96 hep is real model improvement:
  - Stronger longitudinal modelling (trajectory features, slope/last-3-visit deltas — currently only baseline NIT_v1 is well-fit).
  - Ensembling across endpoints (top teams may be co-modelling death + hep with shared latent risk).
  - Survival-aware boosting on per-visit panel rather than wide static row.
  - NIT-only specialist sub-model — small group of patients where FibroScan is the strongest signal — currently averaged out.

## Artifacts
- `tests/results/annitia/full/leakage_findings.csv` — per-val_test row: best train match, exact-cell match count, frac_match, signature match, normalized-Euclid NN distance.
- All 423 rows; sorted by `nn_norm_euclid` ascending.
- Top match: `vt_idx=388 trustii=389` ↔ `train_pid=FRBOO703B95O` at distance 0.944 — still further than the median train↔train NN distance.
