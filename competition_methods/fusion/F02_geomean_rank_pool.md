# F02 — Geometric mean of rank-transformed predictions

**Type:** Fusion / ensembling (parameter-free)
**Domain:** Any ranking / discrimination problem (C-index, AUC, ranking competitions)

## What it is
1. For each candidate, transform predictions to ranks (or rank-percentiles in [0,1]).
2. Take the geometric mean across candidates element-wise.
3. (Optional) Restrict to a subset of decorrelated candidates picked by mutual rank-correlation, *not* by OOF metric.

## Why it helps
- **Zero parameters fit** → impossible to overfit OOF
- **Geometric mean is robust to outlier variants:** one terrible variant can sink an arithmetic mean (especially in probability space) but only multiplicatively dampens a geomean
- **Rank transformation** decouples from probability scale; immune to badly-calibrated bases
- Often outperforms arithmetic mean on heterogeneous bases (Kaggle folklore, also formalized for log-pool of probabilistic forecasts)

## When to use
- Ranking-only metrics (C-index, ROC AUC, NDCG)
- Bases with wildly different prediction scales / calibrations
- When you don't trust your OOF metric to pick fusion weights

## When not to use
- Probability-calibration metrics (Brier, log-loss) where rank transformation discards information
- Tiny pools (≤3 candidates) — averaging is more honest

## Implementation
```python
import numpy as np
from scipy.stats import rankdata

def geomean_rank_pool(oof_dict, decorrelate_top_k=None):
    names = list(oof_dict)
    ranks = {n: rankdata(oof_dict[n]) / len(oof_dict[n]) for n in names}
    if decorrelate_top_k:
        # pick top-k by mutual rank-correlation (NOT by OOF metric)
        names = greedy_decorrelate(ranks, k=decorrelate_top_k)
    log_r = np.mean([np.log(ranks[n] + 1e-9) for n in names], axis=0)
    return np.exp(log_r)

def greedy_decorrelate(ranks, k):
    names = list(ranks)
    chosen = [names[0]]
    while len(chosen) < k and len(chosen) < len(names):
        # add the one most decorrelated with the current set
        scores = {}
        for n in names:
            if n in chosen: continue
            scores[n] = -max(np.corrcoef(ranks[n], ranks[c])[0,1] for c in chosen)
        chosen.append(max(scores, key=scores.get))
    return chosen
```

## Hyperparameter notes
- `decorrelate_top_k = 10` is a good default for libraries of 30–100 candidates
- ε = 1e-9 inside the log avoids zero-rank singularities

## Trade-offs
- Discards probability information (only ranks remain)
- Requires no validation set — but also doesn't *use* one, so can't beat well-tuned weighted methods
- Excellent floor; rarely the absolute best; almost never the worst

## Empirical results (fill in as tested)
| Competition | Date | Primary metric | LB before → after | Notes |
|---|---|---|---|---|

---
[← back to index](../README.md)
