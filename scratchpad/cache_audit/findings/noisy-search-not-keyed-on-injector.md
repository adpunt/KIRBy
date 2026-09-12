# noisy-search-not-keyed-on-injector

**title:** A cached noisy search records nothing about the injector that produced its labels, and the hash built to catch that is never compared

**area:** tests/experiments/noise/ (n01–n15, noise_helpers.py, _similarity_hybrid_space.py, the _smoke_* files) plus src/kirby/noise_spec.py and src/kirby/utils/noise_estimation.py

**class:** C-stale-undetected

**severity as reported:** large

**location:** src/kirby/noise_spec.py:234-260, 466-477; tests/experiments/noise/n15_clean_vs_noisy_hybrid.py:53-55, 186-189; src/kirby/find_best_hybrid.py:592-653 (_make_search_meta)

## scenario as reported
Run 1 builds searches/logd__general__noise_gaussian_e0200_s0_spread__selnoisy__none.pkl. Its stored meta records dataset_name, tier, rep names/shapes, n_folds, seed, budgets, and labels_hash — and labels_hash is the hash of the CLEAN label column, because find_best_hybrid passes `labels` (not the noisy copy) into _make_search_meta. The only trace of the noise is the spec_id inside `tier`, which is built from the strategy NAME, the level, and the seeds. noiseInject is then upgraded in a way that keeps the strategy names (the module docstring at noise_spec.py:49-57 describes exactly such a change already having happened once, and noise_helpers.py:117-121 cites another, d39d611, that changed which molecules are selected). Run 2 injects the new realisation into n15's y_noisy_full at n15:188 and trains on it, but every meta field the pool checks is byte-identical, so the pool hands back run 1's blueprint — fitted to labels that no longer exist anywhere in run 2 — and n15 reports oracle_advantage against it. Nothing raises.

## evidence as reported
noise_spec.py:242-249 builds spec_id from strategy/level/seeds/dose_units/params only; :471-474 'Experiments assert this against the value the cached search was built under, so a blueprint fitted to a DIFFERENT noise draw fails loudly instead of silently'; n15:53-55 'The labels hash is asserted against the cached search'. grep for noisy_labels_hash across the repo returns three sites: the definition, n15:189 where it is written into the results dict, and a print in _smoke_noisy_search.py:81. There is no comparison anywhere. _make_search_meta (find_best_hybrid.py:610) stores 'labels_hash': _hash_bytes(_canonical_bytes(labels)) — the clean column — and has no noise field.

## fix as proposed
Either put the noisy realisation into the identity — hash the noisy column and fold that hash into spec_id (and so into tier and the filename) — or record the installed noiseInject version in the search meta so a version change is a clean miss. Then make n15 actually compare its noisy_labels_hash against the value the search stored and raise on a mismatch, or delete the two docstring sentences that claim it does. Until one of those lands, a noiseInject upgrade needs a documented wipe of searches/*__noise_*.pkl, the matching result_db rows, and the noise-namespaced baselines/, hybrid_evals/ and fold_scores/ entries.
