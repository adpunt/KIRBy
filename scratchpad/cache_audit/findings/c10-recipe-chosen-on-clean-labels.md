# c10-recipe-chosen-on-clean-labels

**title:** c10 sweeps label noise but takes the hybrid's recipe from the clean-label precompute

**area:** tests/experiments/complexity/c01–c19 + complexity_helpers.py (cache behaviour audit)

**class:** A-wrong-answer

**severity as reported:** large

**location:** tests/experiments/complexity/c10_noise_robustness.py:156-163; tests/experiments/_shared.py:4869-4874

## scenario as reported
Run 1 is the suite precompute on logd with the true labels; it writes the winning allocation method, importance method, internal evaluator and augmentation condition per budget. Run 2 is c10 on logd: at c10:156 it calls get_best_search_for_downstream with data=data and no label_noise=, so it gets that clean-label recipe, then at c10:377-388 it injects noise into y_train per fold and rebuilds features under that same recipe at every level of every strategy. Phase 1 (c10:276-306) scores each single representation with no recipe selection at all. So at level 0.6 the hybrid arm is running a recipe that was picked using the clean labels of every row — including the rows each fold then validates on — while the representation it is compared against had no such access. auc_norm and retention_pct for 'hybrids' are inflated relative to 'individual_reps' by exactly that, and the gap grows with the level because the clean recipe keeps pointing at the signal the noise is destroying.

## evidence as reported
c10:156-163 — `search, meta = get_best_search_for_downstream(reps, labels, groups, downstream_model='rf', budget=budget, augmentations=..., dataset_name=dataset_name, task_type=task_type, data=data,)` — no label_noise argument; `grep -n label_noise c10_noise_robustness.py` returns nothing. _shared.py:4869-4874 — "label_noise: a kirby.noise_spec.NoiseSpec. When given, resolves the NOISY lineage — a blueprint searched on noisy training labels and scored on clean ones — instead of the clean precompute. Use this in any experiment that injects label noise, so the hybrid does not enter the comparison having been built with access to the very labels the experiment then corrupts."

## fix as proposed
Owner decision, because it costs one precompute per (strategy, level): either pass label_noise=NoiseSpec(strategy, level) into the c10:156 lookup so each cell resolves its own noisy lineage (find_best_hybrid.py:7325-7327 already namespaces the tier and baseline filenames by spec_id), or keep the clean recipe and record it in the output as a named limitation the way c15/c17/c19 record 'recipe_selected_on': 'all rows'. Do not leave it silent.
