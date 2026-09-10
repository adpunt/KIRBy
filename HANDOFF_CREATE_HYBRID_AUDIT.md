# Handoff prompt — audit every `create_hybrid` call

Copy everything below the line into a fresh chat.

---

Audit every call to `create_hybrid` in the KIRBy repo and tell me which ones are wrong.

`create_hybrid` lives in `src/kirby/hybrid.py`. It takes several representations of the same
molecules, scores each feature, splits a feature budget across the representations, and returns
the selected feature matrix plus a `feature_info` dict. It is the function every experiment's
hybrid arm goes through.

Find every call. Include `src/`, `tests/experiments/` and anything else. Include the wrappers
that call it on an experiment's behalf: `build_fold_hybrid`, `create_hybrid_cached`,
`get_fitted_hybrid`, `get_fitted_pool_hybrid` and `apply_blueprint`. Do not skip
`tests/experiments/framework/` this time — I want the whole picture.

For each call, check all of these:

**1. Is `groups` passed?** `hybrid.py:4440`: with groups, the internal accept/reject holdout is a
`GroupKFold` fold. With no groups it is `np.random.RandomState(42).permutation`, a random 20%
carve of the training rows. That carve decides which features get in. On a scaffold-split
dataset it puts the same scaffold on both sides. Report every call that omits it.

**2. Is `model` passed without `metric_fn`?** `hybrid.py:2172` now raises on that pair. I
confirmed this by running it. So any call passing `model=` and no `metric_fn=` crashes on the
next run — it does not produce a wrong number, it produces no number. `fed02:370`, `fed02:421`
and `fed03:462` are three I already know about. Find the rest.

**3. Is `internal_evaluator='match'` passed with `model=None`?** `hybrid.py:4405` resolves that
to `lightgbm`, and accept/reject is then scored with `r2_score` or accuracy instead of the
dataset's own metric. Silent.

**4. Is the recipe complete?** `build_fold_hybrid` in `tests/experiments/_shared.py` forwards the
internal evaluator, the augmentation strategy, the augmentation margin and fraction, the
candidate ranking rule, the augmentation budgets, and the per-representation scores the search
chose. A bare `create_hybrid` forwards none of them and takes defaults instead. For every bare
call, list which of those fields the blueprint carried and the call dropped.

**5. Is it fitted on training rows only?** The hybrid must be built inside the fold, on that
fold's training rows, and applied to the held-out rows with `apply_feature_selection`. Report
any call that builds on every row and is then scored by cross-validation over those same rows.

**6. Is it a duplicate?** `create_hybrid` re-scores every representation and recomputes feature
importance on every call — measured at 262 to 336 seconds. Report any place that calls it more
than once with arguments that produce the same matrix.

Rules:

- Read the code. Do not report anything from a description or a memory file.
- Quote `file:line` for every finding.
- Plain language. Every noun names a quantity or a step. Say what is computed. Say "features",
  never "columns". No sentence over 30 words.
- Do not fix anything. Report only.
- Do not propose reviving `greedy`, `dominant_greedy` or `SAC`. All three are closed.
- Do not propose cutting a dataset, budget, representation or allocation method.

Give me a table: file, line, which of the six checks it fails, and one sentence on what that
changes. Group by check, not by file. Then say which calls are correct, by name, so I know what
the right pattern looks like in this repo.

Two files already establish the correct pattern:
`tests/experiments/noise/n15_clean_vs_noisy_hybrid.py` and
`tests/experiments/similarity/s04_activity_cliffs.py`.
