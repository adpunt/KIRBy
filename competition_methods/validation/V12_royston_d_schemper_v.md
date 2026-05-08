# V12 — Royston-Sauerbrei D and Schemper-Henderson V

**Type:** Validation metric (effect-size, second-opinion)
**Domain:** Survival prognostic models
**Cite:** Royston & Sauerbrei, "A new measure of prognostic separation in survival data", Stat Med 2004; Schemper & Henderson, "Predictive Accuracy and Explained Variation in Cox Regression", Biometrics 2000

## What it is
- **Royston-Sauerbrei D:** Quantifies how well a prognostic index separates patients into different survival distributions. Equivalent to the log-hazard-ratio between top and bottom risk groups, scaled. Higher = better separation.
- **Schemper-Henderson V:** Explained-variation analogue (R²-like) for survival, adjusted for censoring.

## Why it helps
- C-index alone is rank-only. D and V quantify *how much* the rank separation translates into actual survival difference. Two models with C=0.85 can have wildly different D — one separates the extremes cleanly, the other muddles the middle.
- When C-index ranking is ambiguous between candidates, D / V provide a second opinion grounded in effect size, not pair counting.
- D in particular is much more sensitive to the calibration of the prognostic index than C is.

## When to use
- Final candidate selection when top-k variants are within C-noise
- Reporting alongside C-index for prognostic model papers
- Sensitivity analysis: do C-rank and D-rank agree?

## When not to use
- Non-survival problems
- When a single risk-score output is unavailable (D is most natural for Cox-style PI)

## Implementation
```python
from sksurv.linear_model import CoxPHSurvivalAnalysis
# Royston D: fit Cox on the prognostic index, transform per Royston 2004
def royston_d(risk, y_struct):
    # standardize risk to N(0,1) ranks
    z = (rankdata(risk) - 0.5) / len(risk)
    z = norm.ppf(z) * np.sqrt(8/np.pi)  # Royston scaling
    cox = CoxPHSurvivalAnalysis().fit(z.reshape(-1,1), y_struct)
    return float(cox.coef_[0])  # = D

# Schemper-Henderson V: see survAUC R package or reimplement
```

R package `survAUC::AUC.sh` for V. Royston's `stpm2` (Stata) or `flexsurv` (R) handles D natively.

## Trade-offs
- D is sensitive to the prognostic-index calibration; transforming to standard normal ranks is mandatory
- V is computationally heavier (requires survival function predictions per patient)
- Not as widely supported in Python as C-index — partial reimplementation needed

## Empirical results (fill in as tested)
| Competition | Date | Primary metric | LB before → after | Notes |
|---|---|---|---|---|

---
[← back to index](../README.md)
