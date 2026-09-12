# aug-margin-fraction-dead-key-params

**title:** _hybrid_fold_cache_key accepts augmentation_margin and augmentation_fraction, and create_hybrid_cached never passes them

**area:** Experiment-side caches in tests/experiments/_shared.py (create_hybrid_cached / _hybrid_fold_cache_key, rep_cache, the search-reuse helpers, and the model/evaluator lists that ride in the search cache key)

**class:** A-wrong-answer

**severity as reported:** small

**location:** tests/experiments/_shared.py:75-76 (parameters), 97-98 (hashed), 143-150 (the call that omits them), 173 (they reach create_hybrid through **kwargs); src/kirby/hybrid.py:4932, 4939 (create_hybrid pops them)

## scenario as reported
The key function was written to cover these two — it hashes 'aug_margin:{float}' and 'aug_fraction:{value}' — but create_hybrid_cached's call at line 143 does not forward them, so both always hash their defaults 0.0 and None. Run 1: create_hybrid_cached(..., augmentation_strategy='uncapped') — a block of extras is accepted on any improvement, indices cached. Run 2: the same call plus augmentation_margin=0.05, meaning a block must beat keeping nothing by 0.05 before it is kept. kwargs carries the margin into create_hybrid (hybrid.py:4932 pops it), but the key still hashes aug_margin:0.0, so run 2 lands on run 1's file and is served the selection made with no margin at all. augmentation_fraction is the same: it replaces the 0/5/10/20/40 ladder with round(fraction*width) extras, and hashes as None either way.

## evidence as reported
tests/experiments/_shared.py:75-76
                           augmentation_margin=0.0,
                           augmentation_fraction=None,
tests/experiments/_shared.py:97-98
    h.update(f"aug_margin:{float(augmentation_margin)}".encode())
    h.update(f"aug_fraction:{augmentation_fraction}".encode())
tests/experiments/_shared.py:143-150 passes step_size, patience and greedy_tolerance out of kwargs and stops there.
src/kirby/hybrid.py:4932  _aug_margin = float(kwargs.pop('augmentation_margin', 0.0) or 0.0)
src/kirby/hybrid.py:4939  _aug_fraction = kwargs.pop('augmentation_fraction', None)

## fix as proposed
Add the two lines the others already have: augmentation_margin=kwargs.get('augmentation_margin', 0.0), augmentation_fraction=kwargs.get('augmentation_fraction', None). Nothing on disk changes meaning, because no caller passes either value today.
