# Complexity (c01–c24) & Similarity (s01–s13) Experiment Results — Analysis

**Synced:** 2026-06-02 (JSON-only rsync from ARC `tests/results/`).
**Status:** First batch. WILL EXPAND — more datasets/experiments coming. This is the living record so the analysis is not lost to compaction.

All numbers below are read directly from `tests/results/complexity/<exp>/*.json` and
`tests/results/similarity/<exp>/*.json`. Source/docstrings: `tests/experiments/{complexity,similarity}/`.

---

## 0. VALIDITY GATE — dataset sizes (n training rows)

The single most important caveat: **the suite is dominated by tiny antibody datasets.**
5-fold CV on n<200 is essentially noise; treat sign-flips there as variance, not signal.

| n | dataset | type |
|---:|---|---|
| 94 | ginkgo_heparin | reg |
| 130 | flab_pk_jain_heprt | reg |
| 132 | flab_pk_jain_fcrnrt | reg |
| 137 | flab_polyreact_jain_cic | reg |
| 137 | flab_aggregation_jain_hicrt | reg |
| 137 | flab_thermo_jain | reg |
| 137 | flab_aggregation_jain | reg |
| 137 | flab_polyreact_jain_bvp | reg |
| 141 | abcov_kd | reg |
| 146 | abagym_HER2 | reg |
| 197 | ginkgo_pr_cho | reg |
| 217 | flab_immunogenicity_marks | reg |
| 239 | ginkgo_titer | reg |
| 242 | ginkgo_acsins | reg |
| 242 | ginkgo_colloidal | reg |
| 242 | ginkgo_hic | reg |
| **397** | **abagym_Ang2** | reg |
| **418** | **abagym_VEGF** | reg |
| **429** | **ddg_combined** | reg |
| **564** | **humanness_therapeutic** | clf (AUROC) |
| **627** | **abagym_lysozyme** | reg |
| **659** | **abagym_EGFR** | reg |
| **672** | **flab_thermo_tresanco** | reg |
| **3955** | **buchwald_hartwig** | reaction (R²/RMSE) |
| **5000** | **tz_her2_binary / multiclass** | clf |
| ~ | **tcga_luad_vs_lusc** | clf |
| ~ | **gb_human_nontata_promoters** | nucleotide clf |

**Bolded = reasonably powered (n≥~400).** Everything above the line is n<250 = unreliable.
Regression metric is Spearman ρ (higher better) unless noted; buchwald RMSE lower=better.

### ⚠️ Meta-finding: depth ran on the smallest data
- Broad coverage (c01, c02, c04, c05, s01–s05): ~22–28 datasets incl. the powered ones.
- The "deep/mechanistic" experiments (c09–c24, s06–s13) ran on only ~7–9 datasets,
  **mostly the tiny ones**. For c17–c24 the ONLY powered datasets are
  abagym_VEGF (418), abagym_Ang2 (397), humanness (564).
- Net: the experiments with the most theoretical ambition have the least statistical power.

---

## 1. CORE RESULT — Complexity (c02): does the hybrid beat the best single rep?

c02 = the "optimal pairing" core experiment. Within each model family, compare the best
hybrid `score_mean` against that model's own best single rep (`best_single_score`).
- **Δhon** = honest: best hybrid in the matched (no_aug/uncapped) condition − best single rep.
- **Δopt** = optimistic: best hybrid taking the MAX over 4 conditions (no_aug, uncapped,
  top_tier_only, per_model_top_rep_dim) — this is post-hoc cherry-picking, inflates gains.
- **winM** = # of model families (of 6) where the hybrid beats that family's best single rep.

| n | dataset | bSingle | hyb_hon | Δhon | hyb_opt | Δopt | winM |
|---:|---|---:|---:|---:|---:|---:|---:|
| ~ | tcga_luad_vs_lusc | 0.986 | 0.997 | +0.012 | 0.997 | +0.012 | 5/6 |
| 94 | ginkgo_heparin | 0.711 | 0.754 | +0.043 | 0.754 | +0.043 | 5/6 |
| 130 | flab_pk_jain_heprt | 0.782 | 0.838 | +0.056 | 0.848 | +0.066 | 6/6 |
| 132 | flab_pk_jain_fcrnrt | 0.595 | 0.513 | **−0.081** | 0.652 | +0.057 | 3/6 |
| 137 | flab_aggregation_jain | 0.425 | 0.443 | +0.018 | 0.520 | +0.095 | 5/6 |
| 137 | flab_aggregation_jain_hicrt | 0.504 | 0.525 | +0.021 | 0.601 | +0.096 | 5/6 |
| 137 | flab_polyreact_jain_bvp | 0.374 | 0.450 | +0.075 | 0.458 | +0.084 | 6/6 |
| 137 | flab_polyreact_jain_cic | 0.585 | 0.574 | −0.011 | 0.581 | −0.004 | 5/6 |
| 137 | flab_thermo_jain | 0.457 | 0.530 | +0.073 | 0.547 | +0.090 | 5/6 |
| 141 | abcov_kd | 0.460 | 0.454 | −0.007 | 0.474 | +0.013 | 3/6 |
| 146 | abagym_HER2 | 0.453 | 0.353 | **−0.099** | 0.433 | −0.020 | 4/6 |
| 197 | ginkgo_pr_cho | 0.506 | 0.503 | −0.003 | 0.520 | +0.014 | 4/6 |
| 217 | flab_immunogenicity_marks | 0.399 | 0.396 | −0.003 | 0.421 | +0.022 | 5/6 |
| 239 | ginkgo_titer | 0.399 | 0.430 | +0.032 | 0.470 | +0.072 | 4/6 |
| 242 | ginkgo_acsins | 0.560 | 0.589 | +0.029 | 0.602 | +0.042 | 5/6 |
| 242 | ginkgo_colloidal | 0.417 | 0.417 | +0.000 | 0.417 | +0.000 | 5/6 |
| 242 | ginkgo_hic | 0.582 | 0.589 | +0.008 | 0.602 | +0.020 | 5/6 |
| **397** | **abagym_Ang2** | 0.474 | 0.505 | **+0.031** | 0.609 | +0.135 | 5/6 |
| **418** | **abagym_VEGF** | 0.633 | 0.606 | **−0.027** | 0.638 | +0.005 | 3/6 |
| **429** | **ddg_combined** | 0.293 | 0.401 | **+0.107** | 0.427 | +0.134 | 5/6 |
| **564** | **humanness_therapeutic** | 0.966 | 0.970 | **+0.004** | 0.970 | +0.004 | 5/6 |
| **627** | **abagym_lysozyme** | 0.486 | 0.515 | **+0.029** | 0.515 | +0.029 | 5/6 |
| **659** | **abagym_EGFR** | 0.324 | 0.373 | **+0.048** | 0.374 | +0.050 | 6/6 |
| **672** | **flab_thermo_tresanco** | 0.777 | 0.805 | **+0.028** | 0.813 | +0.036 | 5/5 |
| **5000** | **tz_her2_binary** | 0.919 | 0.923 | **+0.004** | 0.923 | +0.004 | 3/6 |
| **5000** | **tz_her2_multiclass** | 0.758 | 0.766 | **+0.008** | 0.766 | +0.008 | 6/6 |

**Reading:**
- On powered datasets (n≥400) the honest hybrid gain is **positive and consistent but small**:
  ddg +0.107 and EGFR +0.048 are the standouts; lysozyme +0.029, tresanco +0.028,
  Ang2 +0.031; saturated tasks (tcga, humanness, tz) near ceiling so only +0.004–0.012.
  **One genuine powered reversal: abagym_VEGF −0.027.**
- All the LARGE honest losses (fcrnrt −0.081, HER2 −0.099) are tiny-n (≤146) = noise.
- `winM` is 5–6/6 on most datasets: directionally the hybrid beats the *same model's* best
  single rep most of the time, even where the cross-model best barely moves.
- **Do NOT report Δopt** — it's a max over 4 conditions (multiple-comparison inflation;
  e.g. Ang2 +0.135 opt vs +0.031 honest). Report the matched-condition number.

**Headline:** the hybrid helps on well-powered data, but the effect is modest
(typ. +0.01 to +0.05 ρ), largest where baselines are weak (ddg, EGFR), and vanishes
on saturated tasks. The big "wins" in the raw tables are small-n variance.

---

## 2. CORE RESULT — Similarity (s01→s02): property-aware distance vs standard distance

s01 = how well standard single-rep distances track activity (sac_rho = Spearman between
feature-space similarity and label similarity). s02 = the hybrid/property-weighted fix.
- **s01best** = best |sac_rho| over all single reps (unweighted).
- **sig** = # reps with significant SAC / total reps (s01).
- **unwt / wtd** = s02 best individual rep, unweighted vs property-weighted.
- **hybrid** = s02 `best_sac` (the constructed hybrid similarity).

| n | dataset | s01best | sig | unwt | wtd | hybrid | Δ(hyb−unwt) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 94 | ginkgo_heparin | 0.138 | 4/27 | 0.138 | 0.359 | 0.308 | +0.170 |
| 130 | flab_pk_jain_heprt | 0.101 | 1/20 | 0.101 | 0.325 | 0.316 | +0.215 |
| 132 | flab_pk_jain_fcrnrt | 0.085 | 2/11 | 0.085 | 0.206 | 0.268 | +0.183 |
| 137 | flab_aggregation_jain | 0.114 | 9/23 | 0.114 | 0.245 | 0.326 | +0.211 |
| 137 | flab_aggregation_jain_hicrt | 0.047 | 0/23 | 0.047 | 0.232 | 0.194 | +0.147 |
| 137 | flab_polyreact_jain_bvp | 0.118 | 2/23 | 0.073 | 0.148 | 0.189 | +0.116 |
| 137 | flab_polyreact_jain_cic | 0.095 | 2/21 | 0.095 | 0.223 | 0.203 | +0.109 |
| 137 | flab_thermo_jain | 0.106 | 10/22 | 0.106 | 0.180 | 0.106 | +0.000 |
| 141 | abcov_kd | 0.117 | 10/27 | 0.117 | 0.173 | 0.148 | +0.030 |
| 146 | abagym_HER2 | 0.080 | 3/13 | 0.080 | 0.294 | 0.351 | +0.271 |
| 197 | ginkgo_pr_cho | 0.124 | 19/23 | 0.124 | 0.262 | 0.263 | +0.139 |
| 217 | flab_immunogenicity_marks | 0.208 | 14/27 | 0.208 | 0.387 | 0.208 | +0.000 |
| 239 | ginkgo_titer | 0.106 | 11/23 | 0.106 | 0.208 | 0.140 | +0.034 |
| 242 | ginkgo_acsins | 0.120 | 9/26 | 0.120 | 0.212 | 0.239 | +0.119 |
| 242 | ginkgo_colloidal | 0.085 | 0/23 | 0.085 | 0.176 | 0.085 | +0.000 |
| 242 | ginkgo_hic | 0.047 | 0/21 | 0.047 | 0.176 | 0.229 | +0.182 |
| **397** | **abagym_Ang2** | 0.155 | 7/24 | 0.155 | 0.250 | 0.190 | +0.036 |
| **418** | **abagym_VEGF** | 0.098 | 15/24 | 0.098 | 0.192 | 0.226 | +0.129 |
| **564** | **humanness_therapeutic** | 0.289 | 14/26 | 0.289 | 0.496 | 0.515 | +0.226 |
| **659** | **abagym_EGFR** | 0.122 | 14/26 | 0.007 | 0.169 | 0.083 | +0.075 |
| **3955** | **buchwald_hartwig** | 0.143 | 8/8 | 0.143 | 0.231 | 0.239 | +0.097 |
| **5000** | **tz_her2_binary** | 0.124 | 18/29 | 0.124 | 0.289 | 0.194 | +0.070 |
| **5000** | **tz_her2_multiclass** | 0.077 | 14/16 | 0.077 | 0.131 | 0.189 | +0.112 |

**Reading:**
- Standard single-rep distances are weak everywhere (sac_rho ~0.05–0.29, often
  non-significant). **Property-aware weighting roughly DOUBLES the correlation**
  (`wtd` ≈ 2× `unwt` on nearly every dataset) and the effect **holds on powered data**:
  humanness 0.289→0.515, buchwald 0.143→0.239, VEGF 0.098→0.226, tz_multiclass 0.077→0.189.
- This is the most robust positive story in the whole batch — direction-consistent across
  tiny AND powered datasets.
- **Caveat:** absolute correlations stay modest (~0.2–0.5). Relative improvement is the claim,
  not absolute predictivity. Also `best_sac` (hybrid) is sometimes *below* the simple
  property-weighted best rep (`wtd`) — i.e. the weighted single rep is often the real winner,
  not the constructed hybrid (3 datasets fall back: thermo_jain, immunogenicity, colloidal).

---

## 3. Secondary complexity experiments (c01, c04–c08)

- **c01 rep_characterization** — "no single best rep" CONFIRMED, cleaner on powered data
  (13 distinct rep winners; abagym_EGFR best saprot 0.322 vs worst −0.139).
  ⚠️ `mutual_info` field is degenerate (513/567 entries exactly 0.0) — do not cite MI.
- **c04 importance_as_metric** — importance-weighted distance ≥ unweighted on 19/20 reg
  datasets; powered all positive (lysozyme +0.470, EGFR +0.245, ddg +0.282) BUT "best delta"
  is max-over-3-methods (positive-selection bias) and absolute weighted R² is still negative
  on several (EGFR −0.047, ddg −0.311). Direction holds; magnitude tiny where base model good.
- **c05 learning_curves** — best-hybrid > best-single in 9/14 datasets; powered 3/4
  (humanness +0.053, EGFR +0.038, buchwald wins on RMSE; **VEGF reverses −0.027**).
  ⚠️ all 4 abagym curves are all-NaN at their smallest sample sizes (hybrid min budget 50 >
  subsample) → only ~3 valid points each.
- **c06 within_family_gradient** — capacity↔budget coupling NOT supported / degenerate.
  Budget grid floor (50) exceeds small-data feature dim → optimal pinned at 50, no variance.
  Only 2 datasets computable (EGFR, buchwald); 6 family ρ values mean −0.147, only 2/6 positive.
- **c07 data_complexity_partitions** — optimal config DOES differ by subpopulation on the
  `target_variance` split (3/3 powered: VEGF/EGFR/buchwald config_differs=True). ⚠️ only 6/13
  datasets ran (7 smallest yield subpops <50); `scaffold_frequency` split half-degenerate
  on antibody (frequent-scaffold subpop <2 groups → easy=NONE).
- **c08 calibration** — claim that complexity-matching improves calibration **REVERSES on
  all 4 powered datasets**: underfit is best-calibrated (VEGF underfit 0.122 vs optimal 0.168;
  humanness 0.046 vs 0.052; EGFR 0.045 vs 0.129; buchwald 0.053 vs 0.109). The 3 "supporting"
  datasets are tiny-n ties where optimal_budget≤25 makes optimal≡underfit.
  ⚠️ `ue_corr` is nan for all non-RF conditions.

---

## 4. Secondary complexity experiments (c09–c16)

- **c09 formal_analysis** — ANOVA: claimed model/budget/interaction dominance NOT supported;
  **residual variance is 69–98%** on every dataset. eta²(model) 1.3–23.9%, budget 0.1–3.8%.
  ⚠️ mutual-info retention block all zeros (degenerate). Only humanness is well-fit
  (val R²≈0.90, small train-val gap); all regression sets have large bias-variance gaps.
- **c10 noise_robustness** — ❌ TOTAL LOSS. All 9 files are 87 bytes:
  `{"error": "noiseInject not installed"}`. Nothing computed. (Dependency missing on ARC.)
- **c11 ensemble_vs_hybrid** — hybrid > best single on 5/9; powered 2/3 (Ang2 +0.043,
  humanness +0.010; **VEGF −0.047**). Hybrid consistently beats PCA-matched control
  (VEGF 0.586 vs 0.422). ⚠️ NO actual model-averaging "ensemble" arm exists despite the name —
  conditions are best_single/pca_matched/hybrid only.
- **c13 forward_compatibility** — "new rep helps hybrid" (D−C) near-null and sign-mixed
  (+0.059 EGFR … −0.095 HER2). "hybrid helps the rep" (E−B) ⚠️ NaN on 7/9 (only EGFR, humanness
  computed; humanness E−B +0.038, win 87%). Controls clean (random-Gaussian → AUROC 0.5).
- **c15 ml_theory_depth** — strongest signal: **hybrid beats PCA on double-descent metric 9/9**
  (VEGF 0.586 vs 0.363; humanness 0.944 vs 0.919). Distance-concentration (Beyer) improved on
  6/9 but **fails on VEGF/immunogenicity/humanness** at small budget. Genuine interpolation dip
  only at budget=n on VEGF (−0.131), EGFR (−0.020).
- **c16 feature_selection_as_regularization** — directionally supported: reg-param sensitivity
  small at low/optimal budget, rises with budget (VEGF Ridge std 0.004@b10 → 0.089@b250).
  ⚠️ full-dim tuned baseline NaN on 7/9 → hybrid-vs-tuned only resolves on EGFR+humanness (4 wins/2 losses).

---

## 5. Secondary complexity experiments (c17–c24) — weakest power

(Only powered datasets here: VEGF 418, Ang2 397, humanness 564.)

- **c17 info_theoretic_decomposition** — MI thesis degenerate (KSG≈0 everywhere). On actual
  downstream metric, hybrid > PCA/mRMR/random consistently but **RFE matches/beats hybrid** on
  powered sets (VEGF b25 hybrid 0.574 vs RFE 0.666; humanness 0.966 vs RFE 0.973).
- **c18 diversity_vs_depth** — free allocation ("natural") wins at budget 50 but **edge
  collapses at budget 100** (VEGF b50 0.588 best, b100 0.388 < depth_1 0.452). humanness flat.
- **c19 importance_weighted_training** — no consistent winner; deltas ±0.01–0.04, mostly
  neutral/negative on powered sets (VEGF best=unweighted). ⚠️ intended raw-concat arm MISSING
  (only per_space.hybrid present) → core question unanswerable.
- **c21 single_rep_augmentation** — ❌ all powered runs NaN (VEGF/Ang2/abcov_kd: aug_mean=NaN,
  no aug features survived selection; humanness errored). Only tiny sets produced numbers,
  net negative-to-flat (HER2 −0.065).
- **c22 explainability** — descriptive only. Named human-readable features only populate when
  blueprint picks aac/structural (VEGF: 30 named, aac+structural; Ang2: 0 named, PLM-dominated).
- **c23 evaluator_model_matching** — matching effect real but tiny: matched wins 118/219 (54%),
  mean(mismatched−matched) = −0.0073 ρ. On powered sets ≈ noise (VEGF 16/33; Ang2 19/33).
- **c24 blueprint_transfer** — ❌ both powered datasets errored (`insufficient_folds`, 80-byte
  stubs for Ang2/VEGF). Of 3 that ran: selection very stable (Nogueira φ 0.993–0.997) but
  cross-fold transfer mixed (HER2 own−cross +0.142 overfit, abcov_kd −0.014 generalises).

---

## 6. Secondary similarity experiments (s03–s13)

> **Update 2026-06-03:** s05/s06/s08/s10 have since been **code-fixed** and s07/s12/s13
> verified — see the FIX STATUS table in `COMPLEXITY_SIMILARITY_BROKEN_2026_06_03.md`.
> Headlines: s05 now reports an honest pre-registered `fixed_ref` delta (not the post-hoc
> max); s06 uses a graded classification error + surfaces conformal undercoverage (the
> regression near-zero correlation is a genuine negative finding); s08 self-diagnoses
> dimension-collapse and suppresses `optima_differ` when <4 distinct dims; s10 now
> direction-orients every delta (the buchwald RMSE "+13.077" was a worsening, not a gain);
> s13 is a verified mixed finding (retrieval wins, clustering loses), no bug. The numbers
> below are the PRE-fix JSONs and will change on the ARC rerun.

- **s03 feature_concentration** — MI-density mechanism ⚠️ degenerate (mi_full=0 everywhere).
  Only usable signal: LightGBM-gain importance Gini ~0.84–0.98 (importance IS concentrated).
- **s04 activity_cliffs** — ❌ all 23 skipped (124-byte stubs), molecular-only metric on
  antibody/reaction data. No results.
- **s05 knn_prediction** — hybrid distance kNN > best per-rep on 17/22; powered weak
  (VEGF +0.049, tz_multiclass +0.066, humanness +0.004, **EGFR −0.004, Ang2 −0.007**).
  Big gains all tiny-n. ⚠️ "best hybrid" selected post-hoc over 6 budgets×4 conditions
  (optimistic); importance-WEIGHTED hybrid does NOT beat unweighted (gain is the compact
  hybrid space, not the weights).
- **s06 applicability_domain** — distance↔error correlation near zero, sign-inconsistent on
  powered data (VEGF concat +0.144 / hybrid −0.006; tz +0.200/+0.246). Hybrid gives tighter
  intervals but lower coverage. Does not hold.
- **s07 noise_bridge** — ❌ degenerate. Cliff detection collapses: most datasets 0 cliffs;
  all abagym sets flag 100% of compounds as cliffs (contamination=1.0). delta never favors hybrid.
- **s08 optimal_dimension** — only 2 budgets swept (50, 500). optima_differ 7 True/9 False;
  powered VEGF/EGFR/Ang2 same optimum, humanness/tz differ. SAC ρ tiny (~0.1). No clean conclusion.
- **s10 importance_downstream** — importance weighting does NOT replicate for GP/SVR/qRASAR;
  neutral-to-negative on powered sets (VEGF GP −0.005, Ang2 −0.015). ⚠️ buchwald "+13.077" is
  an RMSE-scale artifact, not a gain.
- **s11 bayesian_optimization** — ❌ all 15 skipped (121-byte stubs, molecular-only). No results.
- **s12 cross_domain_cliffs** — ❌ degenerate. 13/16 found 0 cliff pairs; the 3 with pairs give
  nan or negative delta (Ang2 −0.022). No well-powered dataset produced enough cliffs.
- **s13 clustering_retrieval** — retrieval: hybrid wins 14/16 (small margins; VEGF/EGFR reverse).
  clustering coherence: hybrid wins only 5/16 (loses on most powered sets). Mixed.

### Cross-cutting tension (important for the paper)
In **s05 AND s10**, the importance-**weighting** step adds nothing (weighted ≈ or < unweighted).
Any downstream benefit comes from the **compact hybrid space**, not from the property-aware weights.
This is in tension with the s01/s02 framing, where weighting is what doubles sac_rho. Reconcile:
weighting helps the *similarity-activity correlation* metric but not *downstream kNN/GP prediction*.

---

## 7. Bottom lines

**What survives on well-powered data (defensible claims):**
1. No single best representation (c01).
2. Hybrid beats best single rep — modestly, +0.01–0.05 ρ, larger where baselines weak
   (c02: ddg +0.107, EGFR +0.048). One powered reversal (VEGF).
3. Property-aware similarity roughly doubles similarity-activity correlation (s01→s02),
   robust across tiny and powered data — the strongest positive result.
4. Hybrid > PCA on the high-dim/double-descent comparison (c15, 9/9).
5. Optimal config differs by data subpopulation, target_variance split (c07, 3/3 powered).

**What fails or reverses on well-powered data:**
- Calibration improves with complexity matching → REVERSES (c08, underfit best on all 4 powered).
- Capacity↔budget coupling → absent/degenerate (c06).
- Importance weighting helps downstream prediction → fails for kNN/GP/SVR/qRASAR (s05, s10).
- ANOVA model/budget dominance → residual dominates 69–98% (c09).
- Distance tracks error / applicability domain → near-zero, inconsistent (s06).

**Dead / not-produced (rerun or drop):**
- c10 (noiseInject not installed — total loss), c21 (NaN on all powered),
  c24 (errored on both powered), s04 & s11 (all skipped, molecular-only metric),
  s07 & s12 (cliff detection degenerate), s03/c09/c17 MI metrics (KSG≈0 everywhere),
  c13 E−B / c16 full-dim / c19 raw-concat arms (NaN/missing).

**Methodology flags to fix before writing up:**
- Report c02 matched-condition deltas, NOT the max-over-4-conditions (Δopt) number.
- "best hybrid over budgets×conditions" selection is optimistic; needs nested CV or a held-out
  condition choice.
- KSG mutual-information estimator returns ~0 across the board — replace or drop all MI claims.
- Most "deep" experiments need to be re-run on the powered datasets (EGFR, lysozyme, tresanco,
  ddg, tz, tcga, buchwald), not the n<150 antibody set, to be citable.

---

## 8. RERUNS / FIXES NEEDED (drafted from source)

Code locations verified by grep. Server ops are the user's to run (deploy via git pull on ARC).
Ordered by effort/impact.

### A. Dependency missing — c10 (total loss, trivial fix)
- **Cause:** `tests/experiments/complexity/c10_noise_robustness.py:27` does
  `from noiseInject import NoiseInjectorRegression`; package not installed on ARC →
  all 9 files are `{"error": "noiseInject not installed"}`.
- **Fix:** `pip install noiseInject` in the ARC env, add it to SETUP.md + requirements, rerun c10.
  Run on powered datasets too (currently only the tiny 9-dataset set).

### B. Broken MI estimator — c01, c09, c17, s03 (all MI claims dead)
- **Cause:** `estimate_mutual_information_knn(X_reduced, labels, k=5)` (KSG kNN estimator,
  called at `c01_rep_characterization.py:66`; same util used by c09 retention, c17, s03)
  returns ~0.0 almost everywhere (c01: 513/567 entries exactly 0.0; c09/s03 mi_full=0 everywhere).
- **Fix:** replace the KSG util with `sklearn.feature_selection.mutual_info_regression`
  (`mutual_info_classif` for clf) on the selected feature block, OR debug why X_reduced
  collapses MI (likely over-reduced dimensionality before the estimate). Then re-derive the
  MI-density / retention / decomposition sub-claims in c01, c09, c17, s03.
  Until fixed, **drop every MI-density / MI-retention statement** from the writeup.

### C. Degenerate cliff detection — s07, s12 (and the s07 abagym 100% bug)
- **Cause:** `s12_cross_domain_cliffs.py:42` `identify_cliff_pairs_euclidean(... sim_threshold=0.55,
  activity_k=2.0)` with `sim = 1/(1+d)` on standardized high-D concat. On antibody concat the
  pairwise Euclidean d is large → sim ≪ 0.55 → **0 cliff pairs** (13/16 datasets). s07 shows the
  opposite degeneracy on abagym (contamination=1.0, every compound flagged).
- **Fix:** make the similarity threshold **distribution-relative** (e.g. top-Nth-percentile of the
  pairwise similarity matrix) instead of an absolute 0.55, per domain. Re-validate that cliff
  counts land in a sane range (not 0%, not 100%) before trusting deltas. Then rerun s07/s12.

### D. NaN on powered datasets — c21, c24 (no usable powered result)
- **c21:** powered runs (VEGF/Ang2/abcov_kd) give `aug_mean=NaN` — no augmentation feature
  survived selection (`c21:245` nanmean over empty); humanness hits `no_augmentation_reps`
  (`c21:97` — no aug reps configured for that dataset). **Fix:** configure augmentation reps for
  humanness; investigate why aug features never survive selection on the regression powered sets
  (selection threshold likely too strict). Rerun.
- **c24:** Ang2/VEGF error `insufficient_folds` (`c24:237` — needs ≥2 successful folds; folds throw
  in "Own-fold eval failed" at `c24:~231`). **Fix:** capture/print the per-fold exception to find
  why blueprint discovery fails on those two; rerun once folds succeed.

### E. Missing experimental arms (code gaps, not just reruns)
- **c13:** "hybrid helps the rep" (E−B) is NaN on 7/9 — only computed for EGFR + humanness.
  The E−B path needs to populate for regression datasets.
- **c16:** full-dim tuned baseline NaN on 7/9 → hybrid-vs-tuned only resolves on 2 datasets.
- **c19:** the intended raw-concat comparison arm is absent (only `per_space.hybrid` written) →
  the central question is currently unanswerable. Add the raw/concat space.
- **c11:** named "ensemble" but has NO model-averaging arm (only best_single/pca_matched/hybrid).
  Either add a true model-level ensemble or rename the experiment.

### F. Domain-mismatch skips (decide: add molecular runs or mark N/A)
- **s04 (activity_cliffs), s11 (bayesian_optimization):** all runs skipped — both metrics are
  molecular-only and no molecular dataset was in this batch. **Decision needed:** add molecular
  datasets (caco2, herg, logd, qm9, drd2) to the s04/s11 run list, or formally mark them N/A
  for the antibody/reaction/TCGA scope.

### G. Methodology fixes (apply in analysis, no rerun)
- c02: report matched-condition Δ (Section 1 `Δhon`), never the max-over-4-conditions `Δopt`.
- "best hybrid over budgets×conditions" (c02, s05) is post-hoc optimistic — needs a held-out
  condition/budget choice or nested CV to be a clean estimate.
- Re-run the c09–c24 / s06–s13 "deep" experiments on the powered datasets, not the n<150 set.
