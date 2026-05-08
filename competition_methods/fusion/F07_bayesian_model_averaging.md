# F07 — Bayesian Model Averaging (BMA)

**Type:** Fusion / ensembling (posterior-weighted)
**Domain:** Well-defined model classes with computable likelihoods or BIC approximations
**Cite:** Hoeting et al., "Bayesian Model Averaging: A Tutorial", Statistical Science 1999

## What it is
For models {M_1, ..., M_K} with prior p(M_k) and posterior p(M_k | data),
```
p(y_new | data) = Σ p(y_new | M_k, data) * p(M_k | data)
```
In practice, BIC approximation:
```
w_k ∝ exp(-0.5 * BIC_k)
```
or with a reference prior:
```
w_k ∝ p(M_k) * exp(-0.5 * BIC_k)
```

## Why it helps
- Weights based on *posterior probability of the model*, not OOF metric performance
- Naturally penalizes overfitting via BIC's complexity term
- Theoretically robust to selection bias — no greedy max-of-many step
- Decision-theoretically optimal under squared-error loss for predictive distributions (when the prior is well-specified)

## When to use
- Bases are nested or comparable model classes (e.g. Cox with different feature subsets)
- Likelihoods or BIC are computable
- Predictions are probabilistic, not just point estimates

## When not to use
- Heterogeneous black-box bases with no comparable likelihood (random forest vs neural net BIC is hand-wavy)
- Survival models without a unified likelihood scale

## Implementation
```python
def bma_weights(bic_dict, prior=None):
    names = list(bic_dict)
    bic = np.array([bic_dict[n] for n in names])
    if prior is None:
        prior = np.ones(len(names)) / len(names)
    log_w = np.log(prior) - 0.5 * (bic - bic.min())
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    return dict(zip(names, w))
```

## Hyperparameter notes
- Use BIC, not AIC: BIC has the right complexity penalty for model averaging
- Reference prior (uniform) is conservative and usually fine
- Subtract `bic.min()` for numerical stability

## Trade-offs
- BIC requires *number of parameters* — ill-defined for nonparametric / deep bases
- Theoretical guarantees only kick in if the true model is in the library
- Often dominated empirically by F01 / F05 on heterogeneous libraries — but rarely catastrophically wrong

## Empirical results (fill in as tested)
| Competition | Date | Primary metric | LB before → after | Notes |
|---|---|---|---|---|

---
**See also:** [F08 Frequentist Model Averaging](F08_frequentist_model_averaging.md)
[← back to index](../README.md)
