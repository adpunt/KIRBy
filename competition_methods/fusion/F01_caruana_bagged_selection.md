# F01 — Caruana bagged ensemble selection

**Type:** Fusion / ensembling
**Domain:** Any classification / regression / survival with a library of 20+ candidate models
**Cite:** Caruana, Niculescu-Mizil et al., "Ensemble Selection from Libraries of Models", ICML 2004

## What it is
1. Library = N candidate models (any heterogeneous mix).
2. **Sort-init:** seed the ensemble with the top-K models by hold-out metric.
3. **Forward greedy selection with replacement:** repeatedly add the model whose addition (averaged into the running ensemble, possibly with weight 1/(t+1)) most improves the hold-out metric. Same model can be added multiple times → de-facto weights.
4. **Bagging the selection process:** repeat steps 2–3 on B bootstrap samples of the hold-out set; average the resulting ensemble compositions.
5. Final ensemble = average prediction across the B bagged compositions.

## Why it helps
- Greedy selection over a large library overfits the selection metric. Without correction, expected lift over the best single model is illusory: 0.01–0.03 of the apparent gain is selection bias (Caruana 2004).
- Bagging the selection process reduces this overfit dramatically — the bag-averaged ensemble looks like the *expectation over selection variation*, not a single lucky pick.
- Forward-with-replacement allows weighting without explicitly fitting weights — robust to high-dimensional weight estimation.

## When to use
- 20+ candidate models, heterogeneous in family / hyperparameter / training data
- Selection metric is the same as evaluation metric (e.g. C-index)
- KDD-Cup-style competition with diverse model libraries (Niculescu-Mizil 2009 won this way)

## When not to use
- < 10 candidates (just average them)
- Models that are pre-calibrated and known to be exchangeable (simple averaging is a Bayes-optimal alternative under exchangeability)

## Implementation
Reference: [`dclambert/pyensemble`](https://github.com/dclambert/pyensemble) (sklearn-compat).
Sketch:
```python
def caruana_bagged(oof_dict, y, B=20, init_k=5, max_size=50, seed=0):
    rng = np.random.default_rng(seed)
    names = list(oof_dict.keys())
    bag_compositions = []
    for b in range(B):
        boot_idx = rng.integers(0, len(y), len(y))
        scores = {n: metric(oof_dict[n][boot_idx], y[boot_idx]) for n in names}
        ensemble = sorted(names, key=scores.get, reverse=True)[:init_k]
        running = np.mean([oof_dict[n][boot_idx] for n in ensemble], axis=0)
        for _ in range(max_size - init_k):
            best, best_score = None, -np.inf
            for n in names:
                cand = (running * len(ensemble) + oof_dict[n][boot_idx]) / (len(ensemble) + 1)
                s = metric(cand, y[boot_idx])
                if s > best_score:
                    best_score, best = s, n
            ensemble.append(best)
            running = (running * (len(ensemble) - 1) + oof_dict[best][boot_idx]) / len(ensemble)
        bag_compositions.append(ensemble)
    # Final ensemble: average over bag compositions on full data
    final = np.zeros(len(y))
    for comp in bag_compositions:
        final += np.mean([oof_dict[n] for n in comp], axis=0)
    return final / B
```

## Hyperparameter notes
- `init_k = 5`: Caruana's "sort init" guards against the greedy starting at a noisy single model
- `max_size`: 50 is typical; performance plateaus
- `B = 20`: more bags = lower variance, but plateau at ~20

## Trade-offs
- Compute: B × max_size × N metric evaluations on bootstrap samples — cheap if metrics are fast
- Doesn't *learn* feature interactions (it's pure prediction averaging)
- Works best when hold-out set is large enough for stable per-bag scoring

## Empirical results (fill in as tested)
| Competition | Date | Primary metric | LB before → after | Notes |
|---|---|---|---|---|
| ANNITIA MASLD | 2026-05-08 | combined hold-out C | not submitted | **Bagging did not prevent overfit at n=940/47 events.** Train LB 0.9141, hold LB 0.8999 (gap +0.014). With B=12 bags, init_k=5, max_size=40, library=206 candidates. F01 won DEATH hold-out (0.9797 ± 0.003) but HEP lost to anchor (0.866 vs 0.901). Lesson: bagged greedy selection still overfits when the bag-bootstrap is drawn from the same hold-out used for scoring. Need an OUTER hold-out separate from the bagging hold-out. |

---
**See also:** [F06 diversity-penalized](F06_diversity_penalized_selection.md), [V08 hold-out gate](../validation/V08_holdout_fusion_test_gate.md)
[← back to index](../README.md)
