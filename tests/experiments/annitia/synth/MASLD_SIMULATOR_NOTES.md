# MASLD Simulator — clinical parameters & citations

This file documents every numerical parameter used in `masld_simulator.py`.
Each parameter cites the literature source. **Do not edit values without
updating the citation.** If you discover a parameter is wrong, bump the
simulator-version constant in `masld_simulator.py`.

## 1. Phenotype clusters (mixture weights)

Source: Tincopa & Loomba (Nat Med 2024) — MASLD subtype clustering on
multinational cohorts, plus Friedman & Cusi reviews.

| phenotype             | population fraction | rationale                                    |
|-----------------------|---------------------|----------------------------------------------|
| metabolic_obese       | 0.50                | dominant subtype in MASLD clinics            |
| lean_masld            | 0.18                | BMI <25 with MASLD criteria, often PNPLA3+   |
| pnpla3_driven         | 0.15                | rs738409 G/G; faster fibrosis                |
| diabetic_aggressive   | 0.12                | T2DM + obesity, fastest LSM rise             |
| regressor             | 0.05                | spontaneous regressors (e.g. weight loss)    |

## 2. Phenotype-conditional baseline distributions

Sources: NHANES MASLD subgroups, Pennisi 2023 cohort tables, AGILE 3+ paper
(Sanyal Lancet GH 2022), MASLD CPG 2024, Lassailly 2020.

Baseline (means ± SD, prevalences):

| variable           | metabolic | lean      | pnpla3    | diabetic  | regressor |
|--------------------|-----------|-----------|-----------|-----------|-----------|
| age (yr)           | 55±12     | 52±13     | 50±12     | 58±10     | 53±11     |
| BMI                | 33±5      | 23±2      | 28±3      | 35±5      | 30±4      |
| T2DM (P)           | 0.45      | 0.10      | 0.25      | 1.00      | 0.30      |
| Hypertension (P)   | 0.55      | 0.20      | 0.30      | 0.65      | 0.40      |
| Dyslipidaemia (P)  | 0.60      | 0.30      | 0.40      | 0.65      | 0.45      |
| ALT (U/L)          | 45±20     | 35±15     | 80±35     | 55±25     | 40±18     |
| AST (U/L)          | 35±15     | 28±12     | 60±25     | 42±18     | 32±14     |
| GGT (U/L)          | 60±35     | 35±20     | 70±40     | 90±55     | 50±28     |
| platelets (×10⁹/L) | 240±55    | 250±50    | 230±55    | 235±58    | 250±50    |
| chol (mmol/L)      | 5.0±1.0   | 4.7±0.9   | 4.8±1.0   | 5.0±1.1   | 4.8±0.9   |
| triglyc (mmol/L)   | 1.9±0.9   | 1.2±0.5   | 1.6±0.8   | 2.4±1.2   | 1.6±0.7   |
| gluc_fast (mmol/L) | 6.1±1.4   | 5.2±0.7   | 5.5±0.9   | 8.5±2.5   | 5.6±0.9   |
| bilirubin (μmol/L) | 12±5      | 11±4      | 13±6      | 12±5      | 11±4      |
| FibroScan (kPa)    | 8.5±4.0   | 7.0±3.5   | 10.0±5.0  | 12.0±6.0  | 6.5±2.5   |
| FibroTest          | 0.35±0.20 | 0.30±0.18 | 0.45±0.22 | 0.50±0.25 | 0.28±0.16 |
| Aixplorer SWE      | 9.0±4.0   | 7.5±3.5   | 10.5±5.0  | 12.5±6.5  | 7.0±2.8   |

## 3. Annual progression rates

| signal          | progressor | non-prog  | regressor | source                                      |
|-----------------|------------|-----------|-----------|---------------------------------------------|
| ΔLSM (kPa/yr)   | +1.5–2.5   | ±1.5      | −0.5      | Petta/Boursier IPDMA 2023; Lin J Hep 2024   |
| ΔFIB-4 /yr      | +0.15      | +0.05     | −0.15     | Hagström 2018; Lancet Reg Health Eur 2023   |
| ΔALT (U/L /yr)  | −2 to −4   | ±5        | −2        | "ALT paradox" near cirrhosis                |
| Δplatelets /yr  | −4 to −6   | −1        | 0         | Sandrin 2019; Vesterhus 2020                |
| ΔBMI (kg/m² /yr)| +0.2       | +0.1      | −0.5      | population drift; bariatric effect          |

Phenotype-specific LSM annual drift (kPa/yr, mean±SD):

| metabolic | lean   | pnpla3  | diabetic | regressor |
|-----------|--------|---------|----------|-----------|
| 0.30±0.6  | 0.50±0.8 | 0.80±1.0 | 1.20±1.2 | −0.20±0.5 |

## 4. Within-patient autocorrelation (visit-to-visit AR(1) coef)

Half-life (months) → AR(1) at 1-yr lag (Brownian-like):
- ALT/AST: ~5 mo → AR(1) ≈ 0.20 over 1 yr
- LSM, FibroTest: ~12 mo → AR(1) ≈ 0.50
- BMI: ~24 mo → AR(1) ≈ 0.71
- Platelets: ~12 mo → AR(1) ≈ 0.50
- Lipids/glucose: ~9 mo → AR(1) ≈ 0.41

We use AR(1) on the residual after the deterministic drift.

## 5. Joint correlations (residual covariance, pooled approximations)

Used as Σ for multivariate-normal noise on the residual after drift:

| pair                  | r     |
|-----------------------|-------|
| ALT–AST               | 0.80  |
| ALT–GGT               | 0.55  |
| AST–platelets         | −0.30 |
| FibroScan–FIB-4       | 0.60  |
| FibroScan–platelets   | −0.45 |
| FibroScan–FibroTest   | 0.55  |
| FibroScan–Aixplorer   | 0.78  |
| BMI–gluc_fast         | 0.30  |
| BMI–triglyc           | 0.30  |
| BMI–GGT               | 0.30  |
| triglyc–gluc_fast     | 0.35  |
| chol–triglyc          | 0.30  |

Sources: NHANES + UK Biobank pooled, see Petta 2024, Sandrin 2019.

## 6. AGILE 3+ logit (Sanyal Lancet GH 2022)

```
logit(AGILE_3+) = -3.92368
                  + 2.29714 * ln(LSM)
                  - 0.00902 * platelets
                  - 0.98633 * (AST/ALT)
                  + 1.08636 * diabetes
                  - 0.38581 * (sex == female)
                  + 0.03018 * age
```

Risk strata: <0.451 rule-out, ≥0.679 rule-in.

## 7. Hazard rates by AGILE 3+ stratum (per 100 patient-years)

Source: Pennisi CGH 2023 (n=2611, multinational), 5-yr LRE follow-up. Used as
baseline cumulative incidence to draw event times.

| stratum            | LRE incidence /100 PY | 5-yr cum-inc | source            |
|--------------------|-----------------------|--------------|-------------------|
| AGILE_3+ < 0.451   | 0.20                  | 1.0%         | Pennisi 2023      |
| 0.451 ≤ AGILE_3+ < 0.679 | 1.10            | 5.4%         | Pennisi 2023      |
| AGILE_3+ ≥ 0.679   | 5.50                  | 25.0%        | Pennisi 2023      |

We sample event time with exponential hazard at each visit interval and the
above instantaneous rate. If no event → censor at last observed visit.

## 8. Death hazard

Source: Sanyal NEJM 2021 — all-cause mortality by fibrosis stage:

| stratum   | mortality /100 PY |
|-----------|-------------------|
| F0–F2     | 0.32              |
| F3        | 0.89              |
| F4        | 1.76              |

We map AGILE 3+ ranges to fibrosis stage:
- <0.451 → F0–F2 (mortality 0.32)
- 0.451–0.679 → F3 (mortality 0.89)
- ≥0.679 → F4 (mortality 1.76)
plus age-related mortality at exp((age-50)*0.06) baseline scalar.

## 9. Bariatric surgery effect

Source: Lassailly 2020 (Gastroenterology); Sci Rep 2024 — LSM reduction
~30–35% over 6–12 mo, plateau by year 2. Modeled as a 0.7× multiplier on
LSM/FIB-4/ALT at visits ≥1 year post-op, plus a 0.4× multiplier on the LRE
hazard rate.

## 10. Visit timing

Source: ANNITIA train.csv summary — median 5 obs per patient, 21 visits max.
Modeled as Poisson(5) number of visits, gaps ~ Exp(0.6 yr) clipped [0.4, 3 yr].

## 11. Censoring

Censoring (loss to follow-up) modelled as exponential with median 8 years of
observed follow-up (matches ANNITIA distribution: median fu ~7 yr).

## What this simulator is NOT

- Not a causal model — biomarker→event is correlative
- Not learned from data — parameters are literature priors
- Not validated on held-out MASLD cohorts (no published LBs)
- Not a substitute for the 1106 real patients — pretrain only

The point is to give a model exposure to a wider distribution of MASLD
trajectories during pretraining, anchored to clinical priors.
