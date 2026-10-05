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

THE PER-REPRESENTATION HALF FITS NO MODEL. It reads cached representation
matrices and counts, and runs on a laptop. THE HYBRID HALF (on by default, off
with --with-hybrids 0) does not: since 2026-10-04 it runs, or reads from the
cache, a hybrid search on each outer fold's training rows and fits each chosen
recipe on that fold. Since 2026-10-05 it counts three kinds of hybrid per fold:
the general search's best two per decided model, the distance search's best
per decided model, and the similarity search's best. See hybrid_rows. The
outer fold count is one setting, --n-outer-folds, defaulting to get_config's
n_splits. The companion experiment that tries to
FIX what this finds is opig_shared_vectors_remedy.py.

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

    Classification scored by AUC: a tied active and a tied inactive must receive
    one identical score, so that pair counts 0.5 instead of 1. The ceiling is
    1 - 0.5 * (tied active-inactive pairs / all active-inactive pairs).

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
        out['auc_ceiling'] = (float(1.0 - 0.5 * tied_pairs / total_pairs)
                              if total_pairs else None)
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

# Whether knn's own top hybrid of the per-fold distance search is counted too
# (slot 'distance1' under downstream model 'knn'). Added 2026-10-05, OFF.
# Adelaide's answer 1 (handoff 2.3) is "the top hybrid of the distance search
# and the top hybrid of the similarity search", which
# _shared.fold_neighbour_space_recipes builds per decided model: each of the
# dataset's two models gets its own top distance hybrid. knn is the third
# model of every distance search (decided_grid.json) but is not one of the
# dataset's two, so under that reading it gets no slot. Until 2026-10-05 this
# file counted knn's top 2 distance hybrids and nothing else from the distance
# search. True adds knn's top 1 beside the two models' own; it costs no search
# (the distance search already scores knn), one more hybrid build per fold.
COUNT_KNN_DISTANCE_HYBRID = False


def hybrid_pool(dataset_name):
    """The representations every hybrid of this dataset is searched over.

    (general names, donor names). The general list is DATASET_REPS[ds]
    ['general'], the representation-pruning result. The donor list
    (['augmentation']) is passed only when Step 8 keeps an augmentation
    condition other than 'none' on the general or the distance search
    (decided_aug_conditions: ginkgo_colloidal and ginkgo_hic, distance search
    only); everywhere else it is [] and no donor is loaded for the hybrids.

    Fixed 2026-10-05 (handoff 2.4 step 9). Until then the hybrids were built
    over whatever the per-representation table counted: --reps or --no-donors
    changed the hybrid pool, and a pool representation with no cached matrix
    was silently left out of the search. Both asked for a different search
    from the one every other experiment runs on this dataset.
    """
    from experiments.complexity.complexity_helpers import DATASET_REPS
    from experiments.decided_grid import decided_aug_conditions, is_decided
    entry = DATASET_REPS.get(dataset_name) or {}
    general = list(entry.get('general') or [])
    donors = [r for r in (entry.get('augmentation') or []) if r not in general]
    keeps_aug = False
    for tier in ('general', 'general_distance'):
        if is_decided(dataset_name, tier) and any(
                c != 'none' for c in decided_aug_conditions(dataset_name, tier)):
            keeps_aug = True
    return general, (donors if keeps_aug else [])


def hybrid_rows(dataset_name, base, augs, labels, groups, *, task_type,
                metric_name, n_outer_folds, seed=42, checkpoint=None,
                checkpoint_key=None, verbose=True):
    """How many molecules the TOP HYBRIDS give one identical row, per outer fold.

    ⛔ top_base_reps IS DELIBERATELY NOT CALLED IN THIS FILE, AND A 2026-09-28
    AUDIT FLAGGED ITS ABSENCE AS A GAP. It is not one. Everywhere else the
    comparison is each downstream model's best two representations against its
    best two hybrids, both ranked by a score. This file ranks nothing by a
    score: it counts, for every representation, how many molecules get one
    identical row. The base side is therefore the whole per-representation table
    under 'per_representation', which covers every representation rather than a
    best two. The 2 best base representations per model per fold are still
    RECORDED (under 'hybrid_folds', from the same per-fold search) so the
    result says which representations the other experiments would fit.

    Same question the per-representation table answers, asked of a hybrid. A
    hybrid picks features out of the representations in its pool, so it can only
    tell two molecules apart when at least one representation in that pool
    already does -- the all-representations-concatenated line is the floor it
    cannot beat. What is worth knowing is how close a 100 or 500 feature
    selection gets to that floor while dropping most of the features.

    WHERE THE RECIPES COME FROM (2026-10-04, owner's rules of 2026-10-03). For
    each outer fold of nested_cv_outer_folds (the same split and the same
    per-fold cache namespace the other experiments use):
      1. the fold's searches are run on that fold's TRAINING rows, on the
         dataset's decided grid, pruning off (fold_standard_selection,
         fold_searches -> search_on_fold). A search the cache does not hold
         is built here (R6); nothing is refused.
      2. each recipe is fitted on the training rows (build_fold_hybrid; a
         predictive recipe with its downstream model and the dataset's
         metric, a similarity recipe with neither), the selected features
         are taken from the held-out rows too, and ties are counted over
         every molecule -- the same rows the per-representation table counts
         over. The held-out rows' own count is written beside it.
    So one hybrid row is one (fold, downstream model, slot).

    WHICH HYBRIDS, changed 2026-10-05 to Adelaide's answer 1 (handoff 2.3):
    the census measures something with distances, so it counts both kinds of
    neighbour-space hybrid beside the general ones. Per outer fold, the same
    three searches and the same picks as _shared.fold_neighbour_space_recipes:
      'general1', 'general2'  each of the dataset's two decided models: its 2
                              best hybrids of the general search, the second
                              skipped and replaced when its selected features
                              equal the first's.
      'distance1'             each of the two models: its best hybrid of the
                              distance search (tier general_distance, budgets
                              up to 500 features), over every augmentation
                              condition that search runs. knn too only when
                              COUNT_KNN_DISTANCE_HYBRID is True.
      'similarity1'           one per fold: the best hybrid of the similarity
                              search (objective='similarity', features chosen
                              so that distance follows activity difference),
                              ranked over every model it scored. It fits no
                              model, so its downstream_model is None.
    Until 2026-10-05: 2 general hybrids per model and knn's 2 best distance
    hybrids; no similarity hybrid, no distance hybrid for the two models.

    A FAILED SEARCH OR BUILD IS SAVED, not only printed (handoff 2.13): it is
    a row with 'missing_reason' and an entry in the returned failures list
    with dataset, fold, model, search and error. A fold with a failure is not
    checkpointed, so the next attempt retries it; the folds that finished are
    read back.

    Returns (rows, fold_records, failures).
    """
    from kirby.find_best_hybrid import nested_cv_outer_folds

    y = np.asarray(labels)
    g = np.asarray(groups) if groups is not None else None
    outer_folds = nested_cv_outer_folds(base, y, g, n_outer_folds, seed,
                                        task_type)
    n_outer = len(outer_folds)
    if verbose and n_outer == 1:
        print('  one outer fold: a single grouped held-out split, no spread '
              'over folds', flush=True)

    rows, records, failures = [], [], []
    for fold_i, (tr, va) in enumerate(outer_folds):
        fkey = None
        if checkpoint is not None and checkpoint_key is not None:
            fkey = dict(checkpoint_key, unit='hybrid_fold', fold=int(fold_i),
                        n_outer_folds=int(n_outer))
            if checkpoint.done(**fkey):
                f_rows, f_rec = checkpoint.get(**fkey)
                rows.extend(f_rows)
                records.append(f_rec)
                if verbose:
                    print(f'  fold {fold_i}: hybrid rows finished in an '
                          f'earlier attempt, read back', flush=True)
                continue
        f_rows, f_rec, f_fail = _hybrid_rows_one_fold(
            dataset_name, base, augs, y, g, np.asarray(tr), np.asarray(va),
            fold_i=fold_i, n_outer=n_outer, task_type=task_type,
            metric_name=metric_name, seed=seed, verbose=verbose)
        rows.extend(f_rows)
        records.append(f_rec)
        failures.extend(f_fail)
        # Written per fold (dirty-run bug 12): a fold search can take hours,
        # and a kill after fold 3 used to throw away folds 0-2.
        if fkey is not None and not f_fail:
            checkpoint.put((f_rows, f_rec), **fkey)
    return rows, records, failures


def _hybrid_rows_one_fold(dataset_name, base, augs, y, g, tr, va, *, fold_i,
                          n_outer, task_type, metric_name, seed, verbose):
    """The hybrid rows of one outer fold. See hybrid_rows."""
    from experiments._shared import (fold_standard_selection, fold_searches,
                                     build_fold_hybrid,
                                     fold_hybrid_scoring_kwargs, top_hybrids,
                                     top_hybrids_across_models)
    from experiments.decided_grid import decided_models, is_decided
    from kirby.find_best_hybrid import get_model, BUILTIN_METRICS
    from kirby.result_cache import StaleResultDB

    n = int(y.size)
    metric_fn, lower_is_better = BUILTIN_METRICS[metric_name]
    rows, failures = [], []
    record = {'dataset': dataset_name, 'fold': int(fold_i),
              'n_outer_folds': int(n_outer), 'n_train': int(tr.size),
              'n_held_out': int(va.size),
              'selection_scope': 'fold_train_rows',
              'metric': metric_name, 'lower_is_better': bool(lower_is_better),
              'models': {}, 'unavailable': {}}

    # The same arguments _shared.fold_neighbour_space_recipes hands every
    # search, so all three are the per-fold store's entries the noise and
    # similarity experiments read, and a second call is a memo hit.
    common = dict(train_idx=tr, fold_i=fold_i, n_outer_folds=n_outer,
                  dataset_name=dataset_name, augmentations=augs,
                  task_type=task_type, metric=metric_name, seed=seed)

    def _missing(search, model_name, why, slot=None, failed=True):
        """A slot with no count. failed=True also records it as a failure."""
        if verbose:
            print(f'  fold {fold_i} {search} {model_name}'
                  f'{" " + slot if slot else ""}: {why}', flush=True)
        row = {'dataset': dataset_name, 'is_hybrid': True,
               'fold': int(fold_i), 'n_outer_folds': int(n_outer),
               'selection_scope': 'fold_train_rows', 'search': search,
               'downstream_model': model_name, 'missing_reason': why}
        if slot:
            row['slot'] = slot
        rows.append(row)
        if failed:
            failures.append({'dataset': dataset_name, 'fold': int(fold_i),
                             'n_outer_folds': int(n_outer), 'search': search,
                             'downstream_model': model_name, 'slot': slot,
                             'error': why})

    def _search_errors(searches):
        """The failure records search_on_fold kept, as one string."""
        fails = list(getattr(searches, 'failures', None) or [])
        if not fails:
            return 'no condition produced a search (no error recorded)'
        return '; '.join(f"{f.get('aug_condition')}: {f.get('error_type')}: "
                         f"{f.get('error')}" for f in fails)

    # (search, slot, downstream model or None, top_hybrids entry)
    to_build = []

    # --- general search: the 2 best per decided model -----------------------
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    try:
        sel = fold_standard_selection(base, y, g, **common)
    except StaleResultDB:
        raise
    except Exception as exc:
        sel = {}
        for m in models:
            _missing('general', m, f'general search raised '
                     f'{type(exc).__name__}: {exc}')
    if sel and any(v is None for v in sel.values()):
        # fold_standard_selection only prints why. Read the failure records
        # from the same memoised search (no second search is run).
        try:
            why = _search_errors(fold_searches(base, y, g, tier='general',
                                               **common))
        except StaleResultDB:
            raise
        except Exception as exc:
            why = f'{type(exc).__name__}: {exc}'
    for m, picked in sel.items():
        if picked is None:
            _missing('general', m, f'general search on this fold\'s training '
                     f'rows failed: {why}')
            continue
        record['models'].setdefault(m, {}).update({
            'base_representations': list(picked['base']),
            'base_training_row_scores': [
                {'rep': e['rep'], 'score': e.get('score')}
                for e in picked['base_detail']['selected']],
            'general_hybrids_skipped_identical_features': [
                e.get('label') for e in
                picked.get('hybrids_skipped_identical_features') or []]})
        hybrids = picked['hybrids']
        if len(hybrids) < 2:
            _missing('general', m, f'asked for 2 hybrids with distinct '
                     f'selected features, the fold search gives '
                     f'{len(hybrids)} for this model', failed=False)
        for rank, h in enumerate(hybrids, 1):
            to_build.append(('general', f'general{rank}', m, h))

    # --- distance search: each model's best ---------------------------------
    if not is_decided(dataset_name, 'general_distance'):
        record['unavailable']['distance'] = ('DECISIONS.md decides no '
                                             'distance search for this dataset')
        if verbose:
            print(f'  fold {fold_i}: {record["unavailable"]["distance"]}',
                  flush=True)
    else:
        dist_models = list(decided_models(dataset_name, 'general_distance'))
        wanted = [m for m in models if m in dist_models]
        if COUNT_KNN_DISTANCE_HYBRID and 'knn' in dist_models:
            wanted.append('knn')
        try:
            dsr = fold_searches(base, y, g, tier='general_distance', **common)
            conds = {c: s for c, s in dsr.items() if s is not None}
            dist_err = None if 'none' in conds else _search_errors(dsr)
        except StaleResultDB:
            raise
        except Exception as exc:
            conds, dist_err = {}, f'raised {type(exc).__name__}: {exc}'
        for m in models:
            if m not in dist_models:
                _missing('general_distance', m, f'{m} is not one of the '
                         f'distance search\'s decided models {dist_models}',
                         'distance1', failed=False)
        for m in wanted:
            if dist_err is not None:
                _missing('general_distance', m, f'distance search on this '
                         f'fold\'s training rows failed: {dist_err}',
                         'distance1')
                continue
            cands = []
            for cond, srch in conds.items():
                th = top_hybrids(srch, m, n=1, require_full=False,
                                 context=f'{dataset_name} distance fold '
                                         f'{fold_i} {m} {cond}')
                cands += [dict(e, aug_condition=cond) for e in th['selected']]
            lower_d = conds['none'].lower_is_better
            cands.sort(key=lambda e: (e['score_mean'] if lower_d
                                      else -e['score_mean'], e['label']))
            if not cands:
                _missing('general_distance', m, 'the distance search scored '
                         'no hybrid for this model', 'distance1',
                         failed=False)
                continue
            to_build.append(('general_distance', 'distance1', m, cands[0]))

    # --- similarity search: one per fold ------------------------------------
    # 2026-10-05: the owner cut objective='similarity' on some datasets
    # (dataset_policy.SIMILARITY_OBJECTIVE_SKIP_DATASETS: multiclass AUROC is
    # undefined; tz_her2_binary's edit-distance shells leave one group per
    # held-out block). There the slot is recorded as unavailable with the
    # policy's reason and no search is run -- not a failure, so the fold is
    # still checkpointed.
    from experiments.dataset_policy import similarity_objective_undefined
    sim_undefined = similarity_objective_undefined(
        dataset_name, labels=y, task_type=task_type)
    if sim_undefined is not None:
        record['unavailable']['similarity'] = sim_undefined
        _missing('similarity', None, sim_undefined, 'similarity1',
                 failed=False)
    else:
        try:
            ssr = fold_searches(base, y, g, tier='general',
                                objective='similarity', **common)
            sim = ssr.get('none')
            if sim is None:
                _missing('similarity', None, f'similarity search on this '
                         f'fold\'s training rows failed: '
                         f'{_search_errors(ssr)}', 'similarity1')
            else:
                # top_hybrids_across_models already passes require_full=False
                # to top_hybrids; passing it again raised TypeError on every
                # fold until 2026-10-05.
                th = top_hybrids_across_models(
                    sim, n=1,
                    context=f'{dataset_name} similarity fold {fold_i}')
                if th['selected']:
                    # The similarity score's own direction, written beside
                    # its training-row score (not the dataset metric's).
                    to_build.append(('similarity', 'similarity1', None,
                                     dict(th['selected'][0],
                                          aug_condition='none',
                                          search_lower_is_better=bool(getattr(
                                              sim, 'lower_is_better',
                                              False)))))
                else:
                    _missing('similarity', None, 'the similarity search '
                             'scored no hybrid', 'similarity1', failed=False)
        except StaleResultDB:
            raise
        except Exception as exc:
            _missing('similarity', None, f'similarity search raised '
                     f'{type(exc).__name__}: {exc}', 'similarity1')

    # --- build each on the training rows, count ties over every molecule ----
    for search, slot, model_name, h in to_build:
        bp = h['blueprint']
        aug_cond = h.get('aug_condition') or 'none'
        use_aug = aug_cond != 'none' and augs is not None
        model = None
        if model_name is not None:
            model = get_model(model_name, task_type, n_samples=int(tr.size))
            if model is None:
                _missing(search, model_name, f'{model_name} skips {tr.size} '
                         f'training rows (size guard)', slot, failed=False)
                continue
        try:
            X_tr_h, X_va_h, _fi = build_fold_hybrid(
                {k: v[tr] for k, v in base.items()},
                {k: v[va] for k, v in base.items()},
                y[tr], bp,
                augmentations_train=({k: v[tr] for k, v in augs.items()}
                                     if use_aug else None),
                augmentations_val=({k: v[va] for k, v in augs.items()}
                                   if use_aug else None),
                groups=g[tr] if g is not None else None,
                **fold_hybrid_scoring_kwargs(
                    bp, model=model,
                    metric_fn=metric_fn if model is not None else None,
                    lower_is_better=(lower_is_better if model is not None
                                     else None)))
        except StaleResultDB:
            raise
        except Exception as exc:
            _missing(search, model_name, f'build raised '
                     f'{type(exc).__name__}: {exc}', slot)
            continue
        # Back into dataset row order, so the count is over every molecule
        # like the per-representation rows. The dtype is the built hybrid's
        # own (float32 since answer 18); ties are compared byte for byte, so
        # widening it would change nothing but the memory.
        X = np.empty((n, X_tr_h.shape[1]), dtype=np.result_type(X_tr_h, X_va_h))
        X[tr] = X_tr_h
        X[va] = X_va_h
        inv, sizes = row_partition(X)
        tied = int((sizes > 1).sum())
        tied_va = int((sizes[va] > 1).sum())
        rows.append({
            'dataset': dataset_name,
            'representation': (f'hybrid[{model_name or "similarity"}#{slot}'
                               f'#fold{fold_i}]'),
            'is_hybrid': True,
            'fold': int(fold_i),
            'n_outer_folds': int(n_outer),
            'selection_scope': 'fold_train_rows',
            # None for the similarity hybrid, which fits no model.
            'downstream_model': model_name,
            # 'general1', 'general2', 'distance1' or 'similarity1'.
            'slot': slot,
            # 'general', 'general_distance' or 'similarity'.
            'search': search,
            'objective': getattr(bp, 'objective', None),
            'aug_condition': aug_cond,
            'budget': bp.budget,
            'allocation_method': bp.allocation_method,
            'importance_method': bp.importance_method,
            'internal_evaluator': bp.internal_evaluator,
            'rep_count': getattr(bp, 'rep_count', None),
            # The recipe's mean over the inner folds of THIS fold's training
            # rows, in the search's own metric (the dataset's metric for the
            # general and distance searches; the neighbour-prediction score
            # for the similarity search). No held-out score: the census fits
            # no model on the held-out rows, it counts tied rows.
            'training_row_score': h.get('score_mean'),
            'training_row_metric': (metric_name if search != 'similarity'
                                    else 'similarity_objective'),
            'metric': metric_name,
            # Direction of training_row_score (2026-10-05): the similarity
            # search's own for the similarity hybrid, the dataset metric's
            # otherwise. 'metric' above keeps the dataset metric's name.
            'lower_is_better': bool(h['search_lower_is_better']
                                    if search == 'similarity'
                                    else lower_is_better),
            'n_features': int(X.shape[1]),
            'n_molecules': n,
            'n_representation_merged': tied,
            'percent_merged': round(100.0 * tied / n, 4) if n else None,
            'n_held_out': int(va.size),
            'n_held_out_merged': tied_va,
        })
        if verbose:
            print(f'    fold {fold_i} hybrid[{model_name or "similarity"}#'
                  f'{slot}] {X.shape[1]:>5}f  {tied:>5} of {n} share a '
                  f'row ({100.0*tied/max(n,1):5.2f}%)', flush=True)
    return rows, record, failures


def census_one(dataset_name, rep_names, build_missing=False, verbose=True,
               check_freshness=0, with_hybrids=True, n_outer_folds=None,
               checkpoint=None, checkpoint_key=None):
    """Every count for one dataset. Returns (rows, pairwise, pool_row, meta).

    n_outer_folds: required when with_hybrids; main() reads it from one
    setting (--n-outer-folds, default get_config's n_splits).
    """
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

    # The hybrids are searched over the dataset's decided pool, whatever the
    # table counts (hybrid_pool). Load both; the per-representation loop below
    # counts only rep_names. A pool member with no cached matrix is BUILT by
    # load_data here rather than left out of the search.
    pool_general, pool_donors = (hybrid_pool(dataset_name) if with_hybrids
                                 else ([], []))
    to_load = list(dict.fromkeys(list(rep_names) + pool_general + pool_donors))
    if with_hybrids and verbose:
        _have = cached_reps(dataset_name)
        _build = [r for r in pool_general + pool_donors if r not in _have]
        if _build:
            print(f'  hybrid pool members not cached, built now '
                  f'({len(_build)}): {", ".join(_build)}', flush=True)

    data = load_data(dataset_name, rep_preset='full', rep_names=to_load)
    all_reps = dict(data.get('reps') or {})
    reps = {k: v for k, v in all_reps.items() if k in rep_names}
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
    # Reaction datasets: how many distinct values each component takes. A
    # representation of one component can have no more distinct rows than
    # that, which print_dead needs to tell a broken matrix from a small input
    # (2026-10-05: buchwald_hartwig's product representations have 5 distinct
    # rows because the dataset has 5 distinct products).
    comps = data.get('components')
    if comps is not None and len(comps) and isinstance(comps[0], dict):
        meta['n_distinct_per_component'] = {
            c: len({str(d.get(c)) for d in comps})
            for c in sorted({k for d in comps for k in d})}
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
    # The hybrid rows, one per (outer fold, downstream model, slot), from
    # searches on each fold's training rows. Appended to the same list the
    # per-representation rows are in, so every table that sorts `rows` picks
    # them up and a hybrid sits beside the representations it was built from.
    if with_hybrids:
        if n_outer_folds is None:
            raise ValueError('census_one: with_hybrids needs n_outer_folds '
                             '(main() passes --n-outer-folds or get_config\'s '
                             'n_splits)')
        meta['hybrid_selection_scope'] = 'fold_train_rows'
        meta['hybrid_n_outer_folds_asked'] = int(n_outer_folds)
        meta['hybrid_pool'] = {'general': pool_general,
                               'augmentation': pool_donors}
        meta['hybrid_count_knn_distance'] = COUNT_KNN_DISTANCE_HYBRID
        meta['hybrid_folds'] = []
        # Every failed search or build, with dataset, fold, model, search and
        # error (handoff 2.13: until 2026-10-05 a failed search was only
        # printed). main() does not checkpoint a dataset that has any.
        meta['hybrid_failures'] = []
        base = {k: all_reps[k] for k in pool_general if k in all_reps}
        augs = {k: all_reps[k] for k in pool_donors if k in all_reps} or None
        lost = [k for k in pool_general + pool_donors if k not in all_reps]
        if lost or not base:
            why = (f'hybrid pool members load_data returned no matrix for: '
                   f'{", ".join(lost) or "(the whole general list)"}; the '
                   f'search would run on a different pool, so no hybrid is '
                   f'counted')
            meta['hybrid_failures'].append({'dataset': dataset_name,
                                            'fold': None, 'search': None,
                                            'downstream_model': None,
                                            'error': why})
            if verbose:
                print(f'  hybrid rows skipped: {why}', flush=True)
        else:
            try:
                h_rows, h_records, h_fail = hybrid_rows(
                    dataset_name, base, augs, labels, data.get('groups'),
                    task_type=task_type, metric_name=metric_name,
                    n_outer_folds=n_outer_folds, checkpoint=checkpoint,
                    checkpoint_key=checkpoint_key, verbose=verbose)
                rows.extend(h_rows)
                meta['hybrid_folds'] = h_records
                meta['hybrid_failures'].extend(h_fail)
            except Exception as exc:
                from kirby.result_cache import StaleResultDB
                if isinstance(exc, StaleResultDB):
                    raise
                # Written into the result, not only printed.
                meta['hybrid_failures'].append({
                    'dataset': dataset_name, 'fold': None, 'search': None,
                    'downstream_model': None,
                    'error': f'{type(exc).__name__}: {exc}'})
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
              # Hybrid rows carry no stereo split (no cause attribution is
              # done for them), so r["n_stereo_only"] raised KeyError on the
              # first hybrid row and the JSON was never written.
              f'{pct:6.2f}% {str(r.get("n_stereo_only", "-")):>12s}')


def print_dead(rows, metas=()):
    """Matrices with one row for every input, and matrices with very few.

    Changed 2026-10-05. Until then every matrix with five or fewer distinct
    rows was printed as a broken cache. In the dirty run that put
    buchwald_hartwig's product_maccs and product_pharmacophore3d under that
    heading, and both are right: the dataset's 3,955 reactions use 15 aryl
    halides (5 aryl groups x 3 halogens) and one amine, so 5 distinct
    products, and each matrix has exactly 5 distinct rows (checked on the
    local cache 2026-10-05). A few
    distinct rows is broken only when the input has more distinct values than
    that, which this file cannot tell for a representation of one reaction
    component. So only a matrix that is ONE vector for every input is called
    broken; two to five distinct rows are listed separately, with each
    reaction component's count of distinct values beside them, for a person
    to check.
    """
    per_rep = [r for r in rows if r.get('n_distinct_vectors') is not None
               and not r.get('is_hybrid')]
    dead = [r for r in per_rep if r['n_distinct_vectors'] == 1 < r['n_molecules']]
    few = [r for r in per_rep if 1 < r['n_distinct_vectors'] <= 5 < r['n_molecules']]
    comp_counts = {m.get('dataset'): m.get('n_distinct_per_component')
                   for m in metas if m.get('n_distinct_per_component')}
    if dead:
        print('\n' + '=' * 96)
        print('MATRICES THAT ARE ONE VECTOR FOR EVERY INPUT')
        print('Every row identical. These are broken caches, not results.')
        print('=' * 96)
        for r in sorted(dead, key=lambda r: (r['dataset'], r['representation'])):
            print(f'  {r["dataset"]:24s} {r["representation"]:20s} '
                  f'1 distinct row for {r["n_molecules"]:6d} inputs, '
                  f'{r["n_all_zero_rows"]} of them all zero')
    if few:
        print('\n' + '=' * 96)
        print('MATRICES WITH TWO TO FIVE DISTINCT ROWS')
        print('Not necessarily broken. A representation of one reaction '
              'component (product_*, reactant_*)\nhas at most as many distinct '
              'rows as that component has distinct values. Check each\n'
              'against its input before calling it a broken cache.')
        print('=' * 96)
        for r in sorted(few, key=lambda r: (r['dataset'], r['representation'])):
            print(f'  {r["dataset"]:24s} {r["representation"]:20s} '
                  f'{r["n_distinct_vectors"]:3d} distinct rows for '
                  f'{r["n_molecules"]:6d} inputs, '
                  f'{r["n_all_zero_rows"]} of them all zero')
        for ds in sorted({r['dataset'] for r in few} & set(comp_counts)):
            print(f'  {ds}: distinct values per reaction component: '
                  + ', '.join(f'{c} {k}' for c, k in comp_counts[ds].items()))


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
    # On by default. The hybrid half of the comparison is part of what this
    # file measures, so it runs without a flag; --with-hybrids 0 turns it off.
    # Until 2026-10-05 any other number set how many hybrids per model were
    # counted; the count is now fixed by Adelaide's answer 1 (2 general + 1
    # distance per decided model, 1 similarity per fold, see hybrid_rows), so
    # the number only switches the half on or off.
    ap.add_argument('--with-hybrids', type=int, default=1, nargs='?',
                    const=1, dest='with_hybrids',
                    help='per outer fold, count ties for each decided '
                         'model\'s 2 best general hybrids and best distance '
                         'hybrid, and the best similarity hybrid, all searched '
                         'on that fold\'s training rows (default on, 0 = off). '
                         'A fold search the cache does not hold is RUN here.')
    # One setting for the outer fold count (handoff 2.4 step 14, 2026-10-05).
    # Default None = get_config's n_splits under --mode full, the count every
    # experiment built on get_config uses (DEFAULT_N_FOLDS, or
    # KIRBY_LOCAL_FOLDS when set). Until 2026-10-05 the default was a literal 5
    # here and in hybrid_rows and census_one. 1 is a single grouped held-out
    # split (GroupShuffleSplit, 20% held out).
    ap.add_argument('--n-outer-folds', type=int, default=None,
                    dest='n_outer_folds',
                    help='outer folds of nested_cv_outer_folds (seed 42), the '
                         'split whose training rows each hybrid is searched '
                         'and fitted on. Default: get_config\'s n_splits. '
                         'Must match the other experiments\' per-fold '
                         'searches or every fold search is built here.')
    ap.add_argument('--output-dir', default=DEFAULT_OUTPUT)
    ap.add_argument('--force-rerun', action='store_true',
                    help='Recompute what an earlier attempt finished '
                         '(<output>.partial.jsonl).')
    ap = add_distance_metric_arg(ap)
    args = ap.parse_args()
    apply_distance_metric_arg(args)
    if args.n_outer_folds is None:
        from experiments._shared import get_config
        args.n_outer_folds = int(get_config(argparse.Namespace(
            mode='full', distance_metric=args.distance_metric))['n_splits'])
    if args.n_outer_folds < 1:
        ap.error('--n-outer-folds must be 1 or more')
    print(f'outer folds for the hybrid rows: {args.n_outer_folds}'
          + (' (one grouped held-out split, no spread over folds)'
             if args.n_outer_folds == 1 else ''), flush=True)

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
    all_failures = []
    t0 = time.time()
    for ds in datasets:
        print(f'\n=== {ds} ({time.time() - t0:.0f}s) ===', flush=True)
        names = override or pool_for(ds, include_donors=not args.no_donors)
        # hybrid_set: bumped 2026-10-05 to Adelaide's answer 1 (2 general per
        # decided model, each model's best distance hybrid, the best
        # similarity hybrid; hybrids on the decided pool only), so an earlier
        # attempt's rows (2026-10-04: knn's 2 distance hybrids, the pool cut
        # to what was cached) are not read back. Each finished fold is stored
        # under the same key plus its fold number (hybrid_rows), so a killed
        # run keeps its finished folds.
        _ckkey = dict(dataset=ds, names=list(names),
                      # '_v2' 2026-10-05: similarity1 actually counted
                      # (require_full TypeError), its own direction, and the
                      # owner's similarity cut honoured.
                      hybrid_set=('2026-10-05_per_fold_general2_distance1_'
                                  'similarity1_decided_pool_v2'),
                      count_knn_distance=COUNT_KNN_DISTANCE_HYBRID,
                      n_outer_folds=int(args.n_outer_folds))
        try:
            if _ck.done(**_ckkey):
                rows, pairs, pool_row, meta = _ck.get(**_ckkey)
                print('  finished in an earlier attempt, read back',
                      flush=True)
            else:
                rows, pairs, pool_row, meta = census_one(
                    ds, names, build_missing=args.build_missing,
                    check_freshness=args.check_freshness,
                    with_hybrids=bool(args.with_hybrids),
                    n_outer_folds=args.n_outer_folds,
                    checkpoint=_ck, checkpoint_key=_ckkey)
                # A dataset with a failed search or build is not stored, so
                # the next attempt retries it (its finished folds are read
                # back). The failures are in this run's JSON either way.
                if not meta.get('hybrid_failures'):
                    _ck.put((rows, pairs, pool_row, meta), **_ckkey)
        except Exception as exc:                      # one dataset must not stop the rest
            print(f'  FAILED: {type(exc).__name__}: {exc}', flush=True)
            all_meta.append({'dataset': ds, 'error': f'{type(exc).__name__}: {exc}'})
            _failed.append(ds)
            continue
        for f in meta.get('hybrid_failures') or []:
            all_failures.append(f)
        if meta.get('error'):
            print(f'  skipped: {meta["error"]}', flush=True)
        all_rows.extend(rows)
        if not args.no_pairwise:
            all_pairs.extend(pairs)
        if pool_row:
            all_pool.append(pool_row)
        all_meta.append(meta)

    print_dead(all_rows, all_meta)
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
               'per_dataset': all_meta,
               # Every per-fold search or hybrid build that failed, with
               # dataset, fold, downstream model, search and error (handoff
               # 2.13). The same entries sit under each dataset's
               # per_dataset[...]['hybrid_failures'].
               'hybrid_search_failures': all_failures,
               'hybrid_n_outer_folds': args.n_outer_folds}
    if all_failures:
        print(f'\n{len(all_failures)} per-fold hybrid search(es) or build(s) '
              f'failed; each is in the JSON under hybrid_search_failures:',
              flush=True)
        for f in all_failures:
            print(f"  {f.get('dataset')} fold {f.get('fold')} "
                  f"{f.get('search')} {f.get('downstream_model')}: "
                  f"{str(f.get('error'))[:200]}", flush=True)
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
