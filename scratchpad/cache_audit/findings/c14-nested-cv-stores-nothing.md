# c14-nested-cv-stores-nothing

**title:** c14's nested_cv_evaluate runs with cache=False, so no outer fold's search is ever stored or reused

**area:** tests/experiments/complexity/c01–c19 + complexity_helpers.py (cache behaviour audit)

**class:** B-never-hits

**severity as reported:** large

**location:** tests/experiments/complexity/c14_cross_domain_summary.py:257-269; src/kirby/find_best_hybrid.py:6766 (cache default), :7754 (result_db gated on cache), :9053-9061

## scenario as reported
c14 calls nested_cv_evaluate with no cache= and no tier=. nested_cv_evaluate forwards **search_kwargs untouched (it sets only cache_namespace), and _find_best_hybrid_inner declares `cache: bool = False`, so every inner search runs with cache off. find_best_hybrid.py:7754 makes result_db_dataset None when cache is false, so the per-configuration rows are never written. Run 1: c14 --tasks logd --mode full searches 5 outer folds' worth of recipes and dies at the wall on fold 4. Run 2: the resubmit re-searches folds 0-3 from scratch — the result DB has nothing to serve. The hybrid_eval dedup does not cover it either: find_best_hybrid.py:4041 requires `allocation_method.startswith('top_')`, and performance_weighted (create_hybrid's default since 2026-09-10) and the pairdist/entropy/rank_exp families all fail that test.

## evidence as reported
`grep -n cache tests/experiments/complexity/c14_cross_domain_summary.py` returns nothing. find_best_hybrid.py:9048-9061 builds inner_kwargs from search_kwargs and adds only verbose/augmentations/cache_namespace — no cache, no tier. find_best_hybrid.py:7754 — `result_db_dataset=(_ds_cache if (cache and cache_path is not None) else None)`. find_best_hybrid.py:7387-7391 — `if not tier: raise ValueError("find_best_hybrid(cache=True) requires tier=...")`.

## fix as proposed
Needs a name chosen, so it is the owner's call: pass cache=True plus a tier= from c14 (c11 uses tier='general' with the same nested_cv_cache_namespace, c14:257 would match it), or give nested_cv_evaluate its own tier default and turn cache on there. Note the docstring at find_best_hybrid.py:8781 already claims "Every inner search is cached under its own cache_namespace", which is false for every caller that does not pass cache=True — c14 is the one in this area.
