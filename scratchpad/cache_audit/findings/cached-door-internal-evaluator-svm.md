# cached-door-internal-evaluator-svm

**title:** create_hybrid_cached says an unspecified internal evaluator is svm; the engine says lightgbm

**area:** Experiment-side caches in tests/experiments/_shared.py (create_hybrid_cached / _hybrid_fold_cache_key, rep_cache, the search-reuse helpers, and the model/evaluator lists that ride in the search cache key)

**class:** A-wrong-answer

**severity as reported:** small

**location:** tests/experiments/_shared.py:113; src/kirby/hybrid.py:4819

## scenario as reported
A caller that gives create_hybrid_cached no internal_evaluator gets an SVM splitting the feature budget across representations; the same caller going straight to create_hybrid gets LightGBM. Under performance_weighted (the default allocation for both) the evaluator IS what scores each representation and therefore what decides its share of the budget, so the two doors return different hybrids for the same request. The comment eight lines above this default states the rule it breaks: it explains that allocation_method was moved to 'performance_weighted' on 2026-09-10 because 'three doors into one engine must not disagree about what "no allocation method given" means', and then leaves internal_evaluator at the pre-2026-07-16 value. Nothing on disk is mislabelled — internal_evaluator is in the fold cache key — but the two doors answer the same question differently.

## evidence as reported
tests/experiments/_shared.py:113   internal_evaluator='svm',
src/kirby/hybrid.py:4819           internal_evaluator='lightgbm',   (with the comment: 'Ridge was replaced by lightgbm as the internal-evaluator default across this file on 2026-07-16; that change swept every function whose default was ridge and missed this one, which said svm')
tests/experiments/_shared.py:105-112 is the comment asserting the three doors must agree.

## fix as proposed
Change line 113 to 'lightgbm'. The evaluator is in the fold cache key, so a changed default writes new keys and no stored file is reinterpreted. The one in-scope caller (c09_formal_analysis.py:366, :543) always passes int_eval explicitly, so no current number moves. Worth telling the owner separately that c09:539 falls back to 'ridge' when no blueprint is found, which is a forbidden internal evaluator.
