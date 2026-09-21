"""Shrink the searched grid for a local run, from the environment.

Every variable is opt-in: with none of them set this module changes nothing,
which is what a cluster run gets. It exists so a laptop can run the whole
suite on whole datasets without editing twenty experiment files — the rows
stay complete, only the number of configurations searched per row falls.

⛔ NOT a pruning decision. Anything run under these settings is a smoke run:
the axes are cut for compute, so no cell it reports is evidence about which
allocation method, importance method or budget wins.

    KIRBY_LOCAL_GRID=1          print what was changed
    KIRBY_LOCAL_BUDGETS         feature budgets, e.g. 10,50,250,1000
    KIRBY_LOCAL_MODELS          downstream models auto-selected by a search
    KIRBY_LOCAL_EVALUATORS      internal evaluators inside a search
    KIRBY_LOCAL_IMPORTANCE      feature importance methods
    KIRBY_LOCAL_ALLOCATIONS     allocation methods
    KIRBY_LOCAL_REP_COUNTS      how many representations a recipe may use
                                ('none' means the whole pool)
"""
import os


def _names(var):
    raw = os.environ.get(var)
    if not raw:
        return None
    return [p.strip() for p in raw.split(',') if p.strip()]


def _ints(var):
    vals = _names(var)
    return None if vals is None else [int(v) for v in vals]


def _counts(var):
    vals = _names(var)
    if vals is None:
        return None
    return [None if v.lower() in ('none', 'all') else int(v) for v in vals]


def apply():
    """Patch the framework's default axes. Returns {name: value} of what changed."""
    import kirby.find_best_hybrid as fbh

    changed = {}

    def _set(attr, value):
        if value is None:
            return
        setattr(fbh, attr, value)
        changed[attr] = value

    _set('DEFAULT_ALLOCATION_METHODS', _names('KIRBY_LOCAL_ALLOCATIONS'))
    _set('FAST_ALLOCATION_METHODS', _names('KIRBY_LOCAL_ALLOCATIONS'))
    _set('DEFAULT_IMPORTANCE_METHODS', _names('KIRBY_LOCAL_IMPORTANCE'))
    _set('FAST_IMPORTANCE_METHODS', _names('KIRBY_LOCAL_IMPORTANCE'))
    _set('DEFAULT_INTERNAL_EVALUATORS', _names('KIRBY_LOCAL_EVALUATORS'))
    _set('FAST_INTERNAL_EVALUATORS', _names('KIRBY_LOCAL_EVALUATORS'))
    _set('DEFAULT_REP_COUNTS', _counts('KIRBY_LOCAL_REP_COUNTS'))
    _set('FAST_REP_COUNTS', _counts('KIRBY_LOCAL_REP_COUNTS'))

    # A search that is handed models=None reads the tier labels, so the model
    # list is cut by relabelling rather than by a second filter the cache key
    # would not see.
    keep = _names('KIRBY_LOCAL_MODELS')
    if keep:
        unknown = [m for m in keep if m not in fbh.DEFAULT_MODELS]
        if unknown:
            raise ValueError(f"KIRBY_LOCAL_MODELS names no such model: {unknown}")
        for name, spec in fbh.DEFAULT_MODELS.items():
            if spec['tier'] in fbh.AUTO_SELECTED_TIERS and name not in keep:
                spec['tier'] = 'off'
            elif name in keep and spec['tier'] not in fbh.AUTO_SELECTED_TIERS:
                spec['tier'] = 'full'
        changed['auto_selected_models'] = sorted(keep)

    if changed and os.environ.get('KIRBY_LOCAL_GRID'):
        print("  [local grid] " + "; ".join(
            f"{k}={v}" for k, v in sorted(changed.items())))
    return changed


def cut_pool(dataset_name, rep_names):
    """Cut the representation pool to KIRBY_LOCAL_REPS. Unset, returns it unchanged.

    ⛔ THE POOL IS THE EXPERIMENT'S SUBJECT, so this is only ever a compute cut
    for a local run and never a finding about which representations matter. A
    hybrid over five representations is not the hybrid the suite reports.

    Row count is NEVER cut here. Whole datasets only.

    A number keeps the first N in the order DATASET_REPS wrote, which is the
    order the rep-pruning step signed off; a comma-separated list names them
    outright. Either way the kept list is printed, so a result produced under a
    cut pool says so in its own log.

    ⛔ This function was lost once already: _shared.py kept calling it while
    _local_grid.py no longer defined it, so load_data's auto-filter path raised
    ImportError and run_all silently loaded all ten representations. Do not
    remove it without removing both call sites.
    """
    want = os.environ.get('KIRBY_LOCAL_REPS')
    if not want or not rep_names:
        return rep_names
    names = list(rep_names)
    if want.strip().isdigit():
        kept = names[:int(want)]
    else:
        asked = [p.strip() for p in want.split(',') if p.strip()]
        kept = [n for n in names if n in asked]
    if not kept:
        return names
    dropped = [n for n in names if n not in kept]
    print(f"  [local pool] {dataset_name}: {len(kept)} of {len(names)} "
          f"representations kept {kept}; dropped {dropped}")
    return kept
