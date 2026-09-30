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

    # ⛔ THE FOLD COUNT IS A CONDITION, NOT A SAMPLE SIZE. Every configuration
    # in a search is scored once per inner fold, so this multiplies the whole
    # grid. get_config only offers 5 (--mode full) or 1 (--mode intermediary),
    # and 1 is not usable: cv_split falls to a single GroupShuffleSplit and
    # c24 refuses outright with 'insufficient_folds'. The project's own rule is
    # 3-5 fold, group-aware, so 3 is inside it.
    #
    # NOTHING HERE TOUCHES THE ROW COUNT. Every fold still sees every molecule.
    _folds = os.environ.get('KIRBY_LOCAL_FOLDS')
    if _folds:
        n = int(_folds)
        if n < 2:
            raise ValueError(
                f"KIRBY_LOCAL_FOLDS={n}: a single fold is not a fold count. "
                f"cv_split degenerates to one GroupShuffleSplit and c24 "
                f"refuses with insufficient_folds. Use 3 or more.")
        _set('DEFAULT_N_FOLDS', n)

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


def cap_rep_width(reps):
    """Cut every representation to KIRBY_LOCAL_MAX_FEATURES_PER_REP columns.

    ⛔ A COMPUTE CUT FOR THE LOCAL RUN, NEVER A FINDING. The columns kept are
    the highest-variance ones, chosen WITHOUT the labels, so nothing here is
    supervised selection and no row is dropped: every molecule keeps every row.

    Why it exists. Measured 2026-09-24 on flab: importance computation is 62%
    of every create_hybrid call (93s of 150s), and it scales with the total
    width of the pool. The molecular pools are dominated by one representation
    -- grover is 4800 features, 57% of caco2's 8389 and 49% of herg_ki's 9761 --
    so a per-representation cap is the only cut that attacks the real cost
    while leaving every representation in the pool and every experiment's axes
    intact.

    Unset, returns the matrices unchanged.
    """
    cap = os.environ.get('KIRBY_LOCAL_MAX_FEATURES_PER_REP')
    if not cap or not reps:
        return reps
    import numpy as np
    n = int(cap)
    out, notes = {}, []
    for name, X in reps.items():
        if X is None or X.ndim != 2 or X.shape[1] <= n:
            out[name] = X
            continue
        v = np.nanvar(np.asarray(X, dtype=np.float64), axis=0)
        v[~np.isfinite(v)] = -np.inf
        keep = np.sort(np.argsort(v)[-n:])
        out[name] = X[:, keep]
        notes.append(f"{name} {X.shape[1]}->{n}")
    if notes and os.environ.get('KIRBY_LOCAL_GRID'):
        print(f"  [local width cap] top-{n} by variance, labels not used: "
              + ", ".join(notes))
    return out


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
