# fed01-c0-hybrid-gets-no-groups

**title:** fed01's centralised condition builds its hybrid with no scaffold groups, so the budget split is decided on a scaffold-blind 20%

**area:** tests/experiments/federated/ (fed01-fed05 + federated_helpers.py + lhasa_herg_loader.py), and the cache layers they reach in src/kirby/find_best_hybrid.py, src/kirby/hybrid.py, src/kirby/result_cache.py, tests/experiments/_shared.py

**class:** A-wrong-answer

**severity as reported:** large

**location:** tests/experiments/federated/fed01_noise_heterogeneity.py:381-387; src/kirby/hybrid.py:4754-4780 (internal_holdout_split); compare tests/experiments/federated/federated_helpers.py:1231-1232, fed02:418-422, fed04:89-91, fed04:267-269

## scenario as reported
Not a cache collision -- a wrong number, reported here because it sits in the same call. fed01's condition 0 calls create_hybrid(train_reps_all, train_labels_noisy, allocation_method=..., importance_method=..., internal_evaluator=..., augmentations=..., augmentation_strategy=...) and passes no groups, while fed01 has `groups` in scope and slices it for every other condition (fed01:476, :613, :831). With groups=None, internal_holdout_split falls to `indices = np.random.RandomState(42).permutation(n_samples)` -- a fixed-seed but scaffold-blind carve. So on a scaffold-grouped dataset, near-duplicate molecules sit on both sides of the split that decides how the 100-feature budget is divided across representations and which block of extra features is kept, and condition 0's held-out MCC is then compared against conditions 5 and 6, whose per-silo splits ARE scaffold-grouped (federated_helpers.py:1197-1201, which refuses to score a silo at all rather than draw a random 20%). Condition 0 is the centralised arm the whole experiment is measured against.

## evidence as reported
fed01:381-387 -- the argument list runs `train_reps_all, train_labels_noisy, allocation_method=_alloc, importance_method=_imp, internal_evaluator=_int_eval, augmentations=train_augs, augmentation_strategy=aug_cond)` with no groups=. hybrid.py:4777-4780: "Without groups: a fixed-seed random carve of validation_split of the rows." -- `n_val = max(int(n_samples * validation_split), 1); indices = np.random.RandomState(42).permutation(n_samples); return indices[n_val:], indices[:n_val]`. federated_helpers.py:1186-1189 says the same thing in words: "create_hybrid needs them or it splits the silo at random to decide each representation's share of the budget."

## fix as proposed
Add `groups=(groups[train_idx] if groups is not None else None)` to the fed01:381 call, matching fed04:269 and fed02:420. Every stored fed01 condition-0 number was chosen on the scaffold-blind carve, so fed01 has to re-run.
