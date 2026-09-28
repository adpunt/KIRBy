#!/usr/bin/env python3
"""
How many molecules does each representation fail to tell apart?

THE QUESTION. A representation turns a molecule into a row of numbers. When two
different molecules get the identical row, every model downstream must give them
the same prediction. If their measured values differ, that is error no model can
remove. This script counts how often that happens, for every representation on
every molecular and antibody dataset, and turns the count into the best score any
model could reach.

WHY A RAW COUNT OF REPEATED ROWS IS NOT THE ANSWER. Measured on the cached
matrices 2026-09-16, three different things produce a repeated row and they need
different responses:

  1. The dataset holds the same input twice. On flab_thermo_tresanco, 232 of 672
     rows repeat another row under esm2, onehot, blosum and antiberty -- the same
     232 under all four. onehot writes one 1 per residue position, so it can only
     tie two sequences that agree residue for residue. Those rows are the same
     antibody twice. Nothing about esm2 caused it.
  2. Featurisation failed and wrote a row of zeros. On QM9, usrcat and
     coulomb_matrix each repeat 412 rows and all 412 are all-zero -- no conformer
     was generated. Five whole matrices on disk are a single constant vector:
     unimol_v2 on herg_ki and logd, aido_protein and xtrimopglm on
     flab_thermo_tresanco, chytorch_graphormer on buchwald_hartwig.
  3. Two genuinely different molecules the representation cannot separate. This
     is the only one that is a property of the representation.

So every count here is split three ways, and a fourth column says how many of (3)
are molecules that differ only in stereochemistry -- which on herg_ki is all of
them, and which no stereo-blind representation can fix.

WHAT `n_still_tied_if_concatenated` IS FOR. Two representations can each tie 56
molecules and still be useless to each other: on herg_ki, ecfp4, pdv, mordred,
smited and atom_pair tie the SAME 56 rows, so concatenating any of them leaves all
56 tied. The pairwise table computes that number directly rather than by set
overlap: it is the number of molecules still sharing a row in the two matrices
side by side.

NO MODEL FITS. This reads cached representation matrices and counts. It runs on a
laptop. The companion experiment that tries to FIX what this finds is
opig_shared_vectors_remedy.py.

Usage:
    # everything cached, molecular + antibody
    KIRBY_CACHE_DIR=/Volumes/seagate/kirby_cache \
        python tests/experiments/opig_shared_vectors_census.py

    # one dataset, named representations
    KIRBY_CACHE_DIR=/Volumes/seagate/kirby_cache \
        python tests/experiments/opig_shared_vectors_census.py \
        --datasets herg_ki --reps ecfp4,pdv,chemberta,maccs

    # include representations that are not cached yet (BUILDS them -- slow, GPU)
    python tests/experiments/opig_shared_vectors_census.py --build-missing
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import csv
import json
import time
import argparse
import warnings
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
warnings.filterwarnings('ignore')

from experiments._shared import (add_distance_metric_arg,          # noqa: E402
                                 apply_distance_metric_arg,
                                 convert_numpy)

DOMAINS = ('molecular', 'antibody')
DEFAULT_OUTPUT = os.path.join(HERE, '..', 'results', 'comparisons')


# ===== CONFIGURATION ======================================================

def roster(domains=DOMAINS):
    """Datasets that are (a) in one of the wanted domains and (b) have a pool."""
    import experiments.dataset_policy  # noqa: F401  (registers the datasets)
    from kirby.registry import all_datasets
    from experiments.complexity.complexity_helpers import DATASET_REPS
    reg = all_datasets()
    return [name for name in sorted(DATASET_REPS)
            if name in reg and reg[name].domain in domains]


def pool_for(dataset_name, include_donors=True):
    """The pruned pool, plus the donor list this dataset augments from."""
    from experiments.complexity.complexity_helpers import DATASET_REPS
    entry = DATASET_REPS[dataset_name]
    names = list(entry.get('general') or [])
    if include_donors:
        for r in (entry.get('augmentation') or []):
            if r not in names:
                names.append(r)
    return names


def cached_reps(dataset_name):
    """Representation names with a matrix already on disk for this dataset."""
    from experiments._shared import _rep_cache_dir
    d = _rep_cache_dir() / dataset_name
    if not d.is_dir():
        return set()
    return {p.stem for p in d.glob('*.npy') if not p.name.startswith('._')}


# ===== COUNTING ===========================================================

def row_partition(X):
    """Group ids for identical rows, compared byte for byte.

    Returns (group_id_per_row, size_of_each_row's_group). Two rows share a group
    id when their bytes are identical, which for a frozen encoder is exactly the
    condition under which no model can tell them apart.
    """
    A = np.ascontiguousarray(X)
    if A.ndim != 2:
        raise ValueError(f'expected a 2-D matrix, got shape {A.shape}')
    view = A.view([('', A.dtype)] * A.shape[1]).ravel()
    _, inv, counts = np.unique(view, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).ravel()
    return inv, counts[inv]


def joint_partition(inv_a, inv_b):
    """Group ids for the two matrices laid side by side.

    Two molecules share a row of the concatenation exactly when they share a row
    in BOTH halves, so the pair (id_a, id_b) is the concatenation's group id. No
    matrix is built.
    """
    pairs = np.stack([np.asarray(inv_a), np.asarray(inv_b)], axis=1)
    _, inv, counts = np.unique(pairs, axis=0, return_inverse=True,
                              return_counts=True)
    inv = np.asarray(inv).ravel()
    return inv, counts[inv]


def input_keys(data, domain):
    """One string per row that is the same for two rows holding the same input.

    Molecular: the InChIKey, so two spellings of one molecule match and two
    stereoisomers do not. Antibody: the heavy and light sequences joined.
    Returns None when the dataset gives nothing to key on.
    """
    if domain == 'antibody':
        heavy = data.get('heavy_seqs')
        light = data.get('light_seqs')
        if heavy is None:
            return None
        light = light if light is not None else [''] * len(heavy)
        return np.array([f'{h}|{l}' for h, l in zip(heavy, light)], dtype=object)
    smiles = data.get('smiles')
    if smiles is None:
        return None
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    keys = []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if s else None
        keys.append(Chem.MolToInchiKey(m) if m is not None else f'UNPARSED:{s}')
    return np.array(keys, dtype=object)


def stereo_stripped_keys(data, domain):
    """One string per molecule with stereochemistry removed.

    Two molecules that differ only in stereochemistry get the same string here,
    which is what a stereo-blind representation sees. Antibody datasets get None.
    """
    if domain != 'molecular':
        return None
    smiles = data.get('smiles')
    if smiles is None:
        return None
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    keys = []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if s else None
        keys.append(Chem.MolToSmiles(m, isomericSmiles=False)
                    if m is not None else f'UNPARSED:{s}')
    return np.array(keys, dtype=object)


def attribute_groups(inv, sizes, zero_row, keys, stereo_keys):
    """Split the molecules that share a row into the three causes.

    Each GROUP of tied rows is attributed once, in this order, and the molecule
    counts follow from that:
      all rows are all-zero            -> featurisation failed
      all rows hold the same input     -> the dataset holds it more than once
      otherwise                        -> the representation merged them
    A merged group is additionally marked stereo-only when every molecule in it
    canonicalises to one string once stereochemistry is stripped.
    """
    tied = sizes > 1
    out = {'n_molecules_sharing': int(tied.sum()),
           'n_groups': 0,
           'n_from_zero_rows': 0,
           'n_from_repeated_input': 0,
           'n_representation_merged': 0,
           'n_stereo_only': 0,
           'n_merged_groups': 0,
           'n_stereo_only_groups': 0}
    if not tied.any():
        return out
    order = np.argsort(inv, kind='stable')
    inv_sorted = inv[order]
    boundaries = np.flatnonzero(np.diff(inv_sorted)) + 1
    for members in np.split(order, boundaries):
        if members.size < 2:
            continue
        out['n_groups'] += 1
        if zero_row[members].all():
            out['n_from_zero_rows'] += members.size
            continue
        if keys is not None and len(set(keys[members])) == 1:
            out['n_from_repeated_input'] += members.size
            continue
        out['n_representation_merged'] += members.size
        out['n_merged_groups'] += 1
        if stereo_keys is not None and len(set(stereo_keys[members])) == 1:
            out['n_stereo_only'] += members.size
            out['n_stereo_only_groups'] += 1
    return out


# ===== THE CEILING ========================================================

def metric_ceiling(inv, labels, task_type, metric_name):
    """The best score any model can reach given that tied rows share a prediction.

    Regression: a model that sees only these features must give every molecule in
    a tied group one number, and the number that minimises squared error is the
    group's mean measured value. So the floor on RMSE is the root mean square of
    each molecule's distance from its own group's mean, in the dataset's label
    units, and the ceiling on R2 follows from it.

    Classification scored by AUC: every molecule in a tied group receives one
    identical score. The best AUC a group-constant score can reach is the AUC of
    scoring each group by its own fraction of actives (ordering by likelihood
    ratio), with a tie counting 0.5. That is 'auc_ceiling'.

    ⛔ Until 2026-09-28 'auc_ceiling' was 1 - 0.5 * (tied pairs / all pairs),
    which assumes every untied active-inactive pair can be ordered correctly at
    once. Two mixed groups make that impossible: A above B loses B's actives
    against A's inactives. That number is an upper bound above the real
    ceiling; it is kept as 'auc_bound_tied_pairs_only'.

    Classification scored by accuracy: the best a model can do in a tied group is
    the group's commonest class.
    """
    y = np.asarray(labels, dtype=float)
    n = y.size
    out = {'n_molecules': int(n)}
    order = np.argsort(inv, kind='stable')
    groups = np.split(order, np.flatnonzero(np.diff(inv[order])) + 1)

    if task_type == 'regression':
        resid = np.zeros(n, dtype=float)
        for members in groups:
            resid[members] = y[members] - y[members].mean()
        rmse_floor = float(np.sqrt(np.mean(resid ** 2)))
        var = float(np.var(y))
        out['rmse_floor_label_units'] = rmse_floor
        out['label_std_label_units'] = float(np.std(y))
        out['r2_ceiling'] = float(1.0 - (rmse_floor ** 2) / var) if var > 0 else None
        out['ceiling_metric'] = 'rmse_floor_label_units'
        return out

    classes = np.unique(y)
    out['n_classes'] = int(classes.size)
    if classes.size == 2:
        pos, neg = classes[1], classes[0]
        n_pos = int((y == pos).sum())
        n_neg = int((y == neg).sum())
        tied_pairs = 0
        for members in groups:
            if members.size < 2:
                continue
            gp = int((y[members] == pos).sum())
            tied_pairs += gp * (members.size - gp)
        total_pairs = n_pos * n_neg
        out['n_active_inactive_pairs'] = int(total_pairs)
        out['n_pairs_forced_to_tie'] = int(tied_pairs)
        out['auc_bound_tied_pairs_only'] = (
            float(1.0 - 0.5 * tied_pairs / total_pairs) if total_pairs else None)
        if total_pairs:
            from sklearn.metrics import roc_auc_score
            is_pos = (y == pos).astype(float)
            group_rate = np.zeros(n, dtype=float)
            for members in groups:
                group_rate[members] = is_pos[members].mean()
            out['auc_ceiling'] = float(roc_auc_score(is_pos, group_rate))
        else:
            out['auc_ceiling'] = None
    correct = 0
    for members in groups:
        vals, cnt = np.unique(y[members], return_counts=True)
        correct += int(cnt.max())
    out['accuracy_ceiling'] = float(correct / n) if n else None
    out['ceiling_metric'] = ('auc_ceiling' if metric_name == 'auc'
                             else 'accuracy_ceiling')
    return out


def label_disagreement(inv, sizes, labels):
    """How far apart the measured values are inside the groups that must tie."""
    y = np.asarray(labels, dtype=float)
    order = np.argsort(inv, kind='stable')
    spreads = []
    for members in np.split(order, np.flatnonzero(np.diff(inv[order])) + 1):
        if members.size < 2:
            continue
        spreads.append(float(np.std(y[members])))
    if not spreads:
        return {'n_groups_with_disagreement': 0,
                'mean_label_spread_in_groups': None,
                'max_label_spread_in_groups': None}
    spreads = np.asarray(spreads)
    return {'n_groups_with_disagreement': int((spreads > 0).sum()),
            'mean_label_spread_in_groups': float(spreads.mean()),
            'max_label_spread_in_groups': float(spreads.max())}


# ===== IS THE CACHED MATRIX WHAT THE CODE BUILDS TODAY? ===================

# Representations whose builder runs on a laptop in seconds and takes SMILES.
# A learned embedding is included only when loading its checkpoint is cheap.
_REBUILDABLE = {
    'ecfp4': 'create_ecfp4', 'maccs': 'create_maccs', 'avalon': 'create_avalon',
    'atom_pair': 'create_atom_pair', 'top_torsion': 'create_top_torsion',
    'rdkfp': 'create_rdkfp', 'pdv': 'create_pdv', 'chemberta': 'create_chemberta',
    # ⛔ chembert IS NOT chemberta, AND IT IS THE ONE THE POOLS USE.
    # Added 2026-09-18. chembert is 256 features and sits in the general pool of
    # adme_hlm, herg_ki, logd and qm9; chemberta is 384 and sits in qm9's alone.
    # Only chemberta was in this map, so the pools of three datasets held a
    # representation this check had never once looked at — and a stale-cache
    # note about "chemberta on herg_ki and logd" was chasing a representation
    # those two do not carry.
    # Its builder wants torch and transformers, same as chemberta, and the
    # check rebuilds only the sampled rows, so a CPU node is enough.
    'chembert': 'create_chembert',
}


def freshness_check(data, rep_names, n_rows=200, tol=1e-4):
    """Rebuild a sample of rows and see whether the cache still matches the code.

    A cached matrix carries a sha1 of the INPUT strings (_identity.json), which
    catches rows being reordered or a molecule being added. It carries nothing
    about the code that built the features, so a change to a featuriser leaves a
    matrix on disk that no longer matches what the same call returns today and
    nothing notices.

    MEASURED 2026-09-16, which is why this exists: the ChemBERTa reader stopped
    keeping stereochemistry markers, and the cached matrices for herg_ki and logd
    were built before that. 853 of herg_ki's 1,412 rows and 2,730 of logd's 5,028
    differ from what the code returns now. The cached matrix gives every
    stereoisomer its own row; the code gives each pair one shared row -- which is
    130 molecules on herg_ki and 709 on logd that the cache says are separable
    and the pipeline says are not.
    """
    smiles = data.get('smiles')
    if smiles is None:
        return {'note': 'no SMILES on this dataset, nothing to rebuild'}
    import kirby.representations.molecular as M
    n = len(smiles)
    idx = np.linspace(0, n - 1, min(n_rows, n)).astype(int)
    idx = np.unique(idx)
    sample = [smiles[i] for i in idx]
    out = {}
    for rep in rep_names:
        fn_name = _REBUILDABLE.get(rep)
        if fn_name is None or not hasattr(M, fn_name):
            continue
        cached = data['reps'].get(rep)
        if cached is None:
            continue
        try:
            fresh = np.asarray(getattr(M, fn_name)(sample), dtype=np.float32)
        except Exception as exc:
            out[rep] = {'error': f'{type(exc).__name__}: {exc}'}
            continue
        if fresh.shape != (len(idx), cached.shape[1]):
            out[rep] = {'rows_checked': int(len(idx)),
                        'cached_width': int(cached.shape[1]),
                        'rebuilt_width': int(fresh.shape[1]),
                        'matches': False,
                        'note': 'the rebuilt matrix is a different width'}
            continue
        diff = np.abs(fresh - cached[idx])
        n_bad = int((diff > tol).any(axis=1).sum())
        out[rep] = {'rows_checked': int(len(idx)),
                    'rows_that_differ': n_bad,
                    'largest_difference': float(diff.max()),
                    'matches': bool(n_bad == 0)}
    return out


# ===== ONE DATASET ========================================================

def hybrid_rows(dataset_name, reps, labels, groups, n_per_model=2, verbose=True,
                keys=None, stereo_keys=None, task_type=None, domain=None):
    """How many molecules the TOP HYBRIDS give one identical row.

    ⛔ top_base_reps IS DELIBERATELY NOT CALLED IN THIS FILE, AND A 2026-09-28
    AUDIT FLAGGED ITS ABSENCE AS A GAP. It is not one. Everywhere else the
    comparison is each downstream model's best two representations against its
    best two hybrids, both ranked by a score. This file ranks nothing by a
    score: it counts, for every representation, how many molecules get one
    identical row. The base side is therefore the whole per-representation table
    under 'per_representation', which covers every representation rather than a
    best two, and top_hybrids is used here only to choose WHICH hybrids to count
    ties for. Adding a top-two-by-score ranking would be adding a different
    measurement to a file that scores nothing.

    Same question the per-representation table answers, asked of a hybrid. A
    hybrid picks features out of the representations in its pool, so it can only
    tell two molecules apart when at least one representation in that pool
    already does -- the all-representations-concatenated line is the floor it
    cannot beat. What is worth knowing is how close a 100 or 500 feature
    selection gets to that floor while dropping most of the features.

    Where the recipes come from: the search the precompute already stored for
    this dataset. Nothing is searched here. The precompute saves the recipe --
    which representations, how many features from each, how to rank them -- and
    not the feature list, because a per-fold experiment has to rank on its own
    training rows and a whole-dataset list would have seen the held-out ones.
    So the recipe is applied here, over every molecule, which costs one
    importance ranking per representation and no search.

    Returns [] when no search is stored, rather than building one.
    """
    out = []
    try:
        from experiments._shared import (get_best_search_for_downstream,
                                         top_hybrids, build_fold_hybrid)
        from kirby.hybrid import create_hybrid
        from experiments.complexity.complexity_helpers import DATASET_REPS
    except Exception as exc:
        if verbose:
            print(f'  hybrid rows skipped: {type(exc).__name__}: {exc}', flush=True)
        return out

    # Refuse to search. If the precompute has not run for this dataset there is
    # nothing to read and the row is simply absent.
    #
    # ⛔ THE LOOKUP GETS THE GENERAL POOL ONLY. Fixed 2026-09-28. The stored
    # search is keyed on the exact set of representations it searched (the
    # 'rep_pool' column of the result database), and the precompute searches
    # DATASET_REPS[...]['general']. This used to pass `reps`, which is the
    # census set: the general pool PLUS the augmentation donors. On herg_ki that
    # is 15 representations against a stored search over 10, so the key never
    # matched, KIRBY_NO_COLD_SEARCH refused, and no hybrid row was produced.
    _general = list((DATASET_REPS.get(dataset_name) or {}).get('general') or [])
    search_reps = {k: reps[k] for k in _general if k in reps}
    if _general and len(search_reps) < len(_general) and verbose:
        print(f'  hybrid rows: {len(_general) - len(search_reps)} of the '
              f'{len(_general)} general-pool representations are not loaded '
              f'here, so the stored search will not match: '
              f'{sorted(set(_general) - set(search_reps))}', flush=True)
    if not search_reps:
        search_reps = dict(reps)
    _prev = os.environ.get('KIRBY_NO_COLD_SEARCH')
    os.environ['KIRBY_NO_COLD_SEARCH'] = '1'
    try:
        try:
            search, _meta = get_best_search_for_downstream(
                search_reps, labels, groups, downstream_model='rf', budget=None,
                dataset_name=dataset_name, task_type=task_type, domain=domain)
        except Exception as exc:
            if verbose:
                print(f'  no stored search for {dataset_name} '
                      f'({type(exc).__name__}) -- no hybrid rows', flush=True)
            return out
        if search is None:
            if verbose:
                print(f'  no stored search for {dataset_name} -- no hybrid rows',
                      flush=True)
            return out

        # The downstream models to loop are the ones this stored search
        # actually scored, read off the search itself. requested=None asks for
        # all of them. The search above was fetched for 'rf', which decides
        # which stored search is read and not which models it holds scores for.
        #
        # ⛔ What used to be here: `from kirby.find_best_hybrid import
        # DOWNSTREAM_MODELS`. That name is not defined anywhere under src/kirby/
        # -- checked 2026-09-28, zero occurrences -- so the import raised
        # ImportError, the bare `except Exception: return out` below caught it,
        # and this function returned an empty list on every dataset with nothing
        # printed. No hybrid row has ever reached the table.
        from experiments._shared import resolve_comparison_models
        _DM, _dropped_models = resolve_comparison_models(search, None)
    except Exception as exc:
        if verbose:
            print(f'  hybrid rows skipped: could not read which downstream '
                  f'models the stored search scored '
                  f'({type(exc).__name__}: {exc})', flush=True)
        return out
    finally:
        if _prev is None:
            os.environ.pop('KIRBY_NO_COLD_SEARCH', None)
        else:
            os.environ['KIRBY_NO_COLD_SEARCH'] = _prev

    if not _DM and verbose:
        print(f'  no downstream model has a score on the stored search for '
              f'{dataset_name} -- no hybrid rows', flush=True)
    for _name, _why in _dropped_models:
        if verbose:
            print(f'  hybrid rows for {_name}: none -- {_why}', flush=True)

    n = int(np.asarray(labels).size)
    y_all = np.asarray(labels)
    # Per-representation allocation scores over every molecule, one set per
    # (internal evaluator, representation set). A recipe with rep_count below
    # its pool size keeps only the rep_count representations these rank
    # highest, as the search did on each of its folds. The stored recipes carry
    # rep_scores=None, so build_fold_hybrid cannot rank without these.
    _alloc_scores = {}

    def _scores_for(bp, base):
        key = (bp.internal_evaluator, tuple(sorted(base)))
        if key not in _alloc_scores:
            _X, _info = create_hybrid(
                base, y_all, allocation_method='performance_weighted',
                budget=bp.budget, importance_method=bp.importance_method,
                internal_evaluator=bp.internal_evaluator, groups=groups,
                augmentation_strategy='none')
            _alloc_scores[key] = _info.get('allocation_baseline_scores')
        return _alloc_scores[key]

    for model_name in _DM:
        try:
            picked = top_hybrids(search, model_name, n_per_model,
                                 require_full=False,
                                 context=f'census/{dataset_name}/{model_name}')
        except Exception as exc:
            # Say which downstream model has no hybrid row and why. Dropping it
            # silently made a table of three models look like a table of five.
            if verbose:
                print(f'  hybrid rows for {model_name}: none -- top_hybrids '
                      f'raised {type(exc).__name__}: {exc}', flush=True)
            continue
        if picked.get('shortfall_reason') and verbose:
            print(f'  hybrid rows for {model_name}: '
                  f'{picked["shortfall_reason"]}', flush=True)
        for rank, entry in enumerate(picked.get('selected') or [], start=1):
            bp = entry.get('blueprint')
            if bp is None:
                continue
            # ⛔ REBUILT FROM THE RECIPE, NOT FROM create_hybrid's DEFAULTS.
            # Fixed 2026-09-28. This used to call create_hybrid on every
            # representation the census loaded, with four recipe fields. That
            # ignored the recipe's own representation list (bp.reps), its
            # representation count (bp.rep_count: the local default keeps 5 of
            # 10) and its augmentation and ranking fields. The row was then a
            # hybrid the search never chose, filed under the winner's name.
            base = {k: reps[k] for k in (bp.reps or []) if k in reps}
            _missing = sorted(set(bp.reps or []) - set(base))
            if _missing or not base:
                if verbose:
                    print(f'  hybrid {model_name} rank {rank}: the recipe needs '
                          f'{_missing or bp.reps}, not loaded here -- no row',
                          flush=True)
                continue
            try:
                _pre = None
                _rc = getattr(bp, 'rep_count', None)
                if _rc is not None and int(_rc) < len(base):
                    _pre = _scores_for(bp, base)
                X, _Xv, _info = build_fold_hybrid(
                    base, base, y_all, bp, groups=groups,
                    precomputed_alloc_scores=_pre)
            except Exception as exc:
                if verbose:
                    print(f'  hybrid {model_name} rank {rank}: could not build '
                          f'({type(exc).__name__}: {exc})', flush=True)
                continue
            Xh = np.asarray(X, dtype=np.float64)
            inv, sizes = row_partition(Xh)
            # Same attribution as the per-representation rows, so the column
            # 'n_representation_merged' means the same thing on both. It used
            # to be every molecule sharing a row, repeated inputs and zero rows
            # included, under the name of the narrower count -- and the row
            # carried no 'n_stereo_only', which print_worst reads, so the first
            # hybrid row to reach the table raised KeyError before the JSON
            # was written.
            groups_attr = attribute_groups(inv, sizes, ~np.any(Xh != 0, axis=1),
                                           keys, stereo_keys)
            tied = int(groups_attr['n_representation_merged'])
            out.append({
                'dataset': dataset_name,
                'representation': f'hybrid[{model_name}#{rank}]',
                'is_hybrid': True,
                'downstream_model': model_name,
                'rank_within_model': rank,
                'budget': bp.budget,
                'allocation_method': bp.allocation_method,
                'importance_method': bp.importance_method,
                'n_features': int(Xh.shape[1]),
                'n_molecules': n,
                'recipe_reps': list(bp.reps or []),
                'rep_count': getattr(bp, 'rep_count', None),
                'reps_used': sorted(_info.get('rep_count_kept') or base),
                'n_distinct_vectors': int(np.unique(inv).size),
                **groups_attr,
                'percent_molecules_sharing': (round(
                    100.0 * groups_attr['n_molecules_sharing'] / n, 3)
                    if n else None),
                'percent_merged': round(100.0 * tied / n, 4) if n else None,
            })
            if verbose:
                print(f'    hybrid[{model_name}#{rank}] '
                      f'{Xh.shape[1]:>5}f  '
                      f'{groups_attr["n_molecules_sharing"]:>5} of {n} '
                      f'share a row, {tied} merged by the representation '
                      f'({100.0*tied/max(n,1):5.2f}%)', flush=True)
    return out


def census_one(dataset_name, rep_names, build_missing=False, verbose=True,
               check_freshness=0, with_hybrids=2):
    """Every count for one dataset. Returns (rows, pairwise, pool_row, meta)."""
    from experiments._shared import load_data, _resolve_dataset_metric

    if not build_missing:
        have = cached_reps(dataset_name)
        missing = [r for r in rep_names if r not in have]
        rep_names = [r for r in rep_names if r in have]
        if missing and verbose:
            print(f'  not cached, skipped ({len(missing)}): '
                  f'{", ".join(missing)}', flush=True)
    if not rep_names:
        return [], [], None, {'error': 'no cached representations'}

    data = load_data(dataset_name, rep_preset='full', rep_names=rep_names)
    reps = {k: v for k, v in (data.get('reps') or {}).items() if k in rep_names}
    if not reps:
        return [], [], None, {'error': 'load_data returned no matrices'}

    labels = np.asarray(data['labels'])
    domain = data.get('domain')
    metric_name, task_type, _ = _resolve_dataset_metric(
        dataset_name=dataset_name, labels=labels,
        task_type=data.get('task_type'), domain=domain, metric='auto')

    keys = input_keys(data, domain)
    stereo_keys = stereo_stripped_keys(data, domain)
    n_repeated_inputs = (int(len(labels) - len(set(keys.tolist())))
                         if keys is not None else None)

    rows, partitions = [], {}
    for rep in sorted(reps):
        X = np.asarray(reps[rep])
        if X.ndim != 2 or X.shape[0] != labels.size:
            rows.append({'dataset': dataset_name, 'representation': rep,
                         'error': f'shape {X.shape} against {labels.size} labels'})
            continue
        inv, sizes = row_partition(X)
        partitions[rep] = inv
        zero_row = ~np.any(X != 0, axis=1)
        row = {'dataset': dataset_name, 'representation': rep,
               'domain': domain, 'metric': metric_name, 'task_type': task_type,
               'n_molecules': int(X.shape[0]), 'n_features': int(X.shape[1]),
               'n_distinct_vectors': int(np.unique(inv).size),
               'n_all_zero_rows': int(zero_row.sum()),
               'n_rows_with_nan': int(np.isnan(X).any(axis=1).sum())}
        row.update(attribute_groups(inv, sizes, zero_row, keys, stereo_keys))
        row['percent_molecules_sharing'] = round(
            100.0 * row['n_molecules_sharing'] / row['n_molecules'], 3)
        row.update(label_disagreement(inv, sizes, labels))
        row.update({f'ceiling_{k}': v for k, v in
                    metric_ceiling(inv, labels, task_type, metric_name).items()
                    if k != 'n_molecules'})
        rows.append(row)
        if verbose:
            print(f'    {rep:22s} {row["n_molecules_sharing"]:6d} of '
                  f'{row["n_molecules"]:6d} share a row '
                  f'({row["percent_molecules_sharing"]:6.2f}%)  '
                  f'[repeated input {row["n_from_repeated_input"]}, '
                  f'zero rows {row["n_from_zero_rows"]}, '
                  f'representation {row["n_representation_merged"]}'
                  f', of those stereo-only {row["n_stereo_only"]}]', flush=True)

    pairwise = []
    for a, b in combinations(sorted(partitions), 2):
        inv_j, sizes_j = joint_partition(partitions[a], partitions[b])
        ta = np.bincount(partitions[a])[partitions[a]] > 1
        tb = np.bincount(partitions[b])[partitions[b]] > 1
        pairwise.append({
            'dataset': dataset_name, 'rep_a': a, 'rep_b': b,
            'n_tied_a': int(ta.sum()), 'n_tied_b': int(tb.sum()),
            'n_tied_in_both': int((ta & tb).sum()),
            'n_still_tied_if_concatenated': int((sizes_j > 1).sum()),
        })

    pool_row = None
    if len(partitions) >= 2:
        names = sorted(partitions)
        inv_all = partitions[names[0]]
        for rep in names[1:]:
            inv_all, sizes_all = joint_partition(inv_all, partitions[rep])
        pool_row = {'dataset': dataset_name,
                    'n_representations_concatenated': len(names),
                    'representations': names,
                    'n_molecules': int(labels.size),
                    'n_still_tied_if_concatenated': int((sizes_all > 1).sum())}
        pool_row.update({f'ceiling_{k}': v for k, v in
                         metric_ceiling(inv_all, labels, task_type,
                                        metric_name).items()
                         if k != 'n_molecules'})

    meta = {'dataset': dataset_name, 'domain': domain, 'metric': metric_name,
            'task_type': task_type, 'n_molecules': int(labels.size),
            'n_rows_repeating_an_input_already_present': n_repeated_inputs,
            'representations_counted': sorted(partitions)}
    if check_freshness:
        meta['cache_matches_the_code'] = freshness_check(
            data, sorted(reps), n_rows=check_freshness)
        stale = [r for r, v in meta['cache_matches_the_code'].items()
                 if isinstance(v, dict) and v.get('matches') is False]
        if stale and verbose:
            print(f'  ⛔ CACHED MATRIX NO LONGER MATCHES THE CODE: '
                  f'{", ".join(stale)}', flush=True)
            for r in stale:
                v = meta['cache_matches_the_code'][r]
                print(f'     {r}: {v.get("rows_that_differ")} of '
                      f'{v.get("rows_checked")} sampled rows differ, largest '
                      f'difference {v.get("largest_difference")}', flush=True)
    # The hybrid rows, from the recipes the precompute already stored. Appended
    # to the same list the per-representation rows are in, so every table that
    # sorts `rows` picks them up and a hybrid sits beside the representations it
    # was built from. Absent when no search is stored for this dataset.
    if with_hybrids:
        try:
            rows.extend(hybrid_rows(dataset_name, reps, labels,
                                    data.get('groups'), n_per_model=with_hybrids,
                                    verbose=verbose, keys=keys,
                                    stereo_keys=stereo_keys,
                                    task_type=task_type, domain=domain))
        except Exception as exc:
            if verbose:
                print(f'  hybrid rows failed: {type(exc).__name__}: {exc}',
                      flush=True)
    return rows, pairwise, pool_row, meta


# ===== ANALYSIS ===========================================================

def print_worst(rows, top_n=25):
    real = [r for r in rows if r.get('n_representation_merged') is not None]
    real.sort(key=lambda r: -(r['n_representation_merged'] / max(r['n_molecules'], 1)))
    print('\n' + '=' * 96)
    print('WORST: different molecules the representation itself merges')
    print('One row is one (dataset, representation). The count excludes inputs the '
          'dataset holds\ntwice and rows the featuriser left at zero.')
    print('=' * 96)
    print(f'{"dataset":24s} {"representation":20s} {"merged":>8s} {"of":>8s} '
          f'{"%":>7s} {"stereo-only":>12s}')
    for r in real[:top_n]:
        pct = 100.0 * r['n_representation_merged'] / max(r['n_molecules'], 1)
        print(f'{r["dataset"]:24s} {r["representation"]:20s} '
              f'{r["n_representation_merged"]:8d} {r["n_molecules"]:8d} '
              f'{pct:6.2f}% {r["n_stereo_only"]:12d}')


def print_dead(rows):
    dead = [r for r in rows
            if r.get('n_distinct_vectors') is not None
            and r['n_distinct_vectors'] <= 5 < r['n_molecules']]
    if not dead:
        return
    print('\n' + '=' * 96)
    print('MATRICES THAT CARRY ALMOST NO INFORMATION')
    print('Five or fewer distinct rows for the whole dataset. These are broken '
          'caches, not results.')
    print('=' * 96)
    for r in sorted(dead, key=lambda r: (r['dataset'], r['representation'])):
        print(f'  {r["dataset"]:24s} {r["representation"]:20s} '
              f'{r["n_distinct_vectors"]:3d} distinct rows for '
              f'{r["n_molecules"]:6d} molecules, '
              f'{r["n_all_zero_rows"]} of them all zero')


def print_ceilings(rows, top_n=20):
    reg = [r for r in rows if r.get('ceiling_ceiling_metric') ==
           'rmse_floor_label_units' and r.get('ceiling_rmse_floor_label_units')]
    if reg:
        reg.sort(key=lambda r: -r['ceiling_rmse_floor_label_units'])
        print('\n' + '=' * 96)
        print('BEST SCORE ANY MODEL COULD REACH — regression datasets')
        print('rmse_floor is in the dataset\'s own label units and is the error '
              'left when every\ntied group is predicted at its own mean. '
              'label_std is that dataset\'s label spread.')
        print('=' * 96)
        print(f'{"dataset":24s} {"representation":20s} {"rmse floor":>11s} '
              f'{"label std":>11s} {"r2 ceiling":>11s}')
        for r in reg[:top_n]:
            r2 = r.get('ceiling_r2_ceiling')
            print(f'{r["dataset"]:24s} {r["representation"]:20s} '
                  f'{r["ceiling_rmse_floor_label_units"]:11.4f} '
                  f'{r["ceiling_label_std_label_units"]:11.4f} '
                  f'{"—" if r2 is None else f"{r2:11.4f}"}')
    clf = [r for r in rows if r.get('ceiling_auc_ceiling') is not None
           and r['ceiling_auc_ceiling'] < 1.0]
    if clf:
        clf.sort(key=lambda r: r['ceiling_auc_ceiling'])
        print('\n' + '=' * 96)
        print('BEST SCORE ANY MODEL COULD REACH — datasets scored by AUC')
        print('=' * 96)
        print(f'{"dataset":24s} {"representation":20s} {"auc ceiling":>12s} '
              f'{"pairs forced to tie":>21s} {"of all pairs":>14s}')
        for r in clf[:top_n]:
            print(f'{r["dataset"]:24s} {r["representation"]:20s} '
                  f'{r["ceiling_auc_ceiling"]:12.5f} '
                  f'{r["ceiling_n_pairs_forced_to_tie"]:21d} '
                  f'{r["ceiling_n_active_inactive_pairs"]:14d}')


def print_pairs_that_cannot_help(pairwise, top_n=25):
    useless = [p for p in pairwise
               if p['n_still_tied_if_concatenated'] > 0
               and p['n_still_tied_if_concatenated'] == min(p['n_tied_a'],
                                                            p['n_tied_b'])]
    if not useless:
        return
    print('\n' + '=' * 96)
    print('PAIRS WHERE THE BETTER OF THE TWO GAINS NOTHING')
    print('Concatenating these two leaves as many molecules tied as the '
          'better of the two had on its own. Whatever the second one adds, '
          'it does not separate the tied molecules of the first.')
    print('=' * 96)
    print(f'{"dataset":22s} {"rep a":18s} {"rep b":18s} {"a":>6s} {"b":>6s} '
          f'{"together":>9s}')
    for p in sorted(useless, key=lambda p: -p['n_still_tied_if_concatenated'])[:top_n]:
        print(f'{p["dataset"]:22s} {p["rep_a"]:18s} {p["rep_b"]:18s} '
              f'{p["n_tied_a"]:6d} {p["n_tied_b"]:6d} '
              f'{p["n_still_tied_if_concatenated"]:9d}')


def print_pool(pool_rows):
    if not pool_rows:
        return
    print('\n' + '=' * 96)
    print('THE WHOLE POOL SIDE BY SIDE')
    print('Molecules still sharing a row when every representation in the pool is '
          'concatenated.\nThis is the floor a hybrid cannot go below, because a '
          'hybrid selects from these features.')
    print('=' * 96)
    print(f'{"dataset":24s} {"reps":>5s} {"molecules":>10s} {"still tied":>11s} '
          f'{"%":>7s}')
    for p in sorted(pool_rows, key=lambda p: p['dataset']):
        pct = 100.0 * p['n_still_tied_if_concatenated'] / max(p['n_molecules'], 1)
        print(f'{p["dataset"]:24s} {p["n_representations_concatenated"]:5d} '
              f'{p["n_molecules"]:10d} {p["n_still_tied_if_concatenated"]:11d} '
              f'{pct:6.2f}%')


# ===== MAIN ===============================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=None,
                    help='default: every molecular and antibody dataset with a pool')
    ap.add_argument('--domains', nargs='+', default=list(DOMAINS),
                    choices=['molecular', 'antibody', 'nucleotide', 'dta',
                             'reaction', 'tcga'])
    ap.add_argument('--reps', default=None,
                    help='comma-separated representation names; default is each '
                         'dataset\'s pruned pool plus the list it augments from')
    ap.add_argument('--no-donors', action='store_true',
                    help='pruned pool only, leave out the augmentation donors')
    ap.add_argument('--build-missing', action='store_true',
                    help='build representations that are not cached (slow, GPU)')
    ap.add_argument('--no-pairwise', action='store_true',
                    help='skip the representation-by-representation table')
    ap.add_argument('--check-freshness', type=int, default=0, nargs='?',
                    const=200, dest='check_freshness',
                    help='rebuild this many evenly-spaced rows with the live '
                         'builder and report every representation whose cached '
                         'matrix no longer matches. Only the builders that run '
                         'on a laptop are checked. Default off.')
    # Default 2, not 0. The hybrid half of the comparison is part of what this
    # file measures, so it runs without a flag; passing --with-hybrids 0 turns
    # it off and any other number overrides how many hybrids per downstream
    # model are counted.
    ap.add_argument('--with-hybrids', type=int, default=2, nargs='?',
                    const=2, dest='with_hybrids',
                    help='count ties for the top N hybrids per '
                         'downstream model, read from the search the '
                         'precompute stored (default 2, 0 = off). Applies each '
                         'stored recipe over every molecule; it never '
                         'searches.')
    ap.add_argument('--output-dir', default=DEFAULT_OUTPUT)
    ap.add_argument('--force-rerun', action='store_true',
                    help='Recompute what an earlier attempt finished '
                         '(<output>.partial.jsonl).')
    ap = add_distance_metric_arg(ap)
    args = ap.parse_args()
    apply_distance_metric_arg(args)

    datasets = args.datasets or roster(tuple(args.domains))
    override = ([r.strip() for r in args.reps.split(',') if r.strip()]
                if args.reps else None)

    # Each dataset's census is written to
    # opig_shared_vectors_census.partial.jsonl when it finishes and read back
    # on the next attempt (2026-09-25; experiments/_checkpoint.py). The census
    # counts identical rows of stored matrices, so a dataset read back holds
    # what a fresh count gives. A dataset that raised is not stored, is retried
    # next time, and makes the script exit 1 after the JSON is written. A
    # dataset census_one itself skips (meta['error']) is stored: that skip is
    # decided by the data, not by a failure.
    from experiments._checkpoint import script_checkpoint, exit_if_failed
    _ck = script_checkpoint(
        'opig_shared_vectors_census',
        os.path.join(args.output_dir, 'opig_shared_vectors_census.json'),
        args, exclude=('datasets', 'domains', 'output_dir'))
    _failed = []

    all_rows, all_pairs, all_pool, all_meta = [], [], [], []
    t0 = time.time()
    for ds in datasets:
        print(f'\n=== {ds} ({time.time() - t0:.0f}s) ===', flush=True)
        names = override or pool_for(ds, include_donors=not args.no_donors)
        try:
            if _ck.done(dataset=ds, names=list(names)):
                rows, pairs, pool_row, meta = _ck.get(dataset=ds,
                                                      names=list(names))
                print('  finished in an earlier attempt, read back',
                      flush=True)
            else:
                rows, pairs, pool_row, meta = census_one(
                    ds, names, build_missing=args.build_missing,
                    check_freshness=args.check_freshness,
                    with_hybrids=args.with_hybrids)
                _ck.put((rows, pairs, pool_row, meta), dataset=ds,
                        names=list(names))
        except Exception as exc:                      # one dataset must not stop the rest
            print(f'  FAILED: {type(exc).__name__}: {exc}', flush=True)
            all_meta.append({'dataset': ds, 'error': f'{type(exc).__name__}: {exc}'})
            _failed.append(ds)
            continue
        if meta.get('error'):
            print(f'  skipped: {meta["error"]}', flush=True)
        all_rows.extend(rows)
        if not args.no_pairwise:
            all_pairs.extend(pairs)
        if pool_row:
            all_pool.append(pool_row)
        all_meta.append(meta)

    print_dead(all_rows)
    print_worst(all_rows)
    print_ceilings(all_rows)
    print_pairs_that_cannot_help(all_pairs)
    print_pool(all_pool)

    os.makedirs(args.output_dir, exist_ok=True)
    payload = {'measured_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'what_is_counted':
                   'A molecule counts as sharing a vector when its row of the '
                   'representation matrix is byte-for-byte identical to some '
                   'other molecule\'s row. The count is split by cause: the '
                   'dataset holding one input twice, a featuriser writing a row '
                   'of zeros, and the representation merging two genuinely '
                   'different molecules. Only the third is a property of the '
                   'representation.',
               'per_representation': all_rows,
               'pairwise': all_pairs,
               'whole_pool_concatenated': all_pool,
               'per_dataset': all_meta}
    jpath = os.path.join(args.output_dir, 'opig_shared_vectors_census.json')
    with open(jpath, 'w') as f:
        json.dump(convert_numpy(payload), f, indent=1)
    print(f'\nwrote {jpath}')

    if all_rows:
        fields = sorted({k for r in all_rows for k in r})
        fields = (['dataset', 'representation'] +
                  [k for k in fields if k not in ('dataset', 'representation')])
        cpath = os.path.join(args.output_dir, 'opig_shared_vectors_census.csv')
        with open(cpath, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)
        print(f'wrote {cpath}')
    exit_if_failed(_failed, 'dataset')


if __name__ == '__main__':
    main()
