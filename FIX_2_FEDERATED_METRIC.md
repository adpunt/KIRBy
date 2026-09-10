# Fix 2 — the federated calls that pass a model with no metric

Copy below the line into a fresh chat.

---

Nine calls to `create_hybrid` in the federated suite pass `model=` and no `metric_fn=`. Add the
metric.

The sites:

```
tests/experiments/federated/fed02_rep_diversity_silos.py:370
tests/experiments/federated/fed02_rep_diversity_silos.py:421
tests/experiments/federated/fed03_synthetic_private.py:462
tests/experiments/federated/fed03_synthetic_private.py:750
tests/experiments/federated/fed03_synthetic_private.py:813
tests/experiments/federated/fed03_synthetic_private.py:1091
tests/experiments/federated/fed03_synthetic_private.py:1237
tests/experiments/federated/federated_helpers.py:1100
tests/experiments/federated/federated_helpers.py:1875
```

Why it matters. `create_hybrid` scores each representation by fitting the model and calling
`metric_fn(y_val, y_pred)`. With no metric that call raises `TypeError`, the surrounding except
treats it as an infeasible allocation and returns minus infinity, and every representation then
scores the same. The budget split falls through to an equal split filed under whatever
allocation name the run requested. `src/kirby/hybrid.py:2172` now raises on the pair, so on this
machine these nine crash. `git show HEAD:src/kirby/hybrid.py` does not contain that raise, so on
the server they still equal-split.

Each site already has a metric in scope. `fed02:115` computes it from `get_domain_metric` and
never passes it. Find the equivalent in each file rather than importing a new one.

Rules:

- Read each site and its enclosing function before editing.
- Pass the dataset's own metric, not `r2` and not accuracy. If a site has no metric in scope,
  say so and stop rather than picking one.
- `lower_is_better` travels with the metric. Check whether the site needs it too.
- Do not change any allocation method, budget, representation or model.
- Where `groups` is also missing from the same call, report it and do not fix it — that is a
  separate sweep.
- When done, list which federated result files on disk were produced by the equal-split
  behaviour and are therefore not what their filename claims.
