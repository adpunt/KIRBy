# V06 — Cumulative-dynamic AUC (Heagerty)

**Type:** Validation metric (time-dependent discrimination)
**Domain:** Survival
**Cite:** Heagerty et al., "Time-dependent ROC curves for censored survival data and a diagnostic marker", Biometrics 2000

## What it is
Time-dependent AUC at horizon t: the probability that a patient with event by time t has higher predicted risk than a patient still event-free at time t. Repeated over a grid of t → cumulative-dynamic AUC curve. Often summarized as a mean AUC over the grid (analogous to IBS for Brier).

## Why it helps
- C-index is a single scalar that averages over all event times. Models can be strong at short horizons but weak at long ones, or vice versa.
- Useful for fusion: weight base models by their horizon-specific AUC, not their global C.
- Reveals when in time a model adds discrimination — e.g. AGILE 3+ predicts 5y events but not 10y.

## When to use
- Survival problems with multiple clinically meaningful horizons
- Designing fusion weights that route long-horizon vs short-horizon predictions to specialist models
- Diagnosing why two C-equivalent models disagree on individual patients

## When not to use
- Single-horizon competition target
- Sparse events that prevent reliable per-horizon estimation

## Implementation
```python
from sksurv.metrics import cumulative_dynamic_auc
times = np.linspace(t_min, t_max, 20)
auc_t, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_pred, times)
```

## Trade-offs
- Requires sufficient events at each evaluated horizon
- Sensitive to censoring distribution near the upper end
- Often correlates strongly with C-index globally — diagnostic use, not always primary

## Empirical results (fill in as tested)
| Competition | Date | Primary metric | LB before → after | Notes |
|---|---|---|---|---|

---
[← back to index](../README.md)
