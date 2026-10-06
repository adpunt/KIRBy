"""Shared utilities for complexity and similarity experiment scripts.

Extracted from complexity_theory_experiments.py and similarity_experiments.py.
Provides mode configs, CV helpers, model factories, metrics, data loading, and IO.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import tempfile
import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.special import digamma
from scipy.stats import spearmanr, f as f_dist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from kirby.scaling import FeatureScaler  # 0/1 features unscaled (2026-10-05)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors

# Add src and tests to path. THIS HAS TO COME BEFORE THE FIRST kirby IMPORT.
# It did not, and on 2026-09-05 all 16 flab_expression_koenig tasks of job
# 8718964 died at ModuleNotFoundError: kirby.similarity.distance_config. That
# module is committed (c01132d) and was in the checkout the job ran from. The
# import above this line resolved `kirby` against the venv's installed copy
# instead, which is shared between the stat-cadd and stat-ecr checkouts and
# was older. setup.sh reinstalls it editable but swallows the failure with
# `|| true`, so a failed reinstall is silent. Putting src on the path first
# makes the checkout the script lives in win, whatever the venv holds.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from kirby.similarity.distance_config import (
    DISTANCE_METRICS, DEFAULT_DISTANCE_METRIC,
    get_distance_metric, set_distance_metric,
    sklearn_kwargs as sklearn_distance_kwargs,
)
from kirby.hybrid import create_hybrid, apply_feature_selection, compute_feature_importance
from kirby.find_best_hybrid import find_best_hybrid, SearchResult
from kirby.registry import DATASET_REGISTRY, lookup as registry_lookup

from experiments import dataset_policy  # noqa: F401  (registers dataset skip policy + registry)

warnings.filterwarnings('ignore')


# =============================================================================
# FOLD-AWARE HYBRID CACHING
# =============================================================================

import hashlib
import pickle

def _hybrid_fold_cache_dir() -> Path:
    """Return cache directory for per-fold hybrid feature_info."""
    cache_root = Path(os.environ.get('KIRBY_CACHE_DIR',
                      str(Path.home() / '.cache' / 'kirby')))
    d = cache_root / 'hybrid_fold_cache'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hybrid_fold_cache_key(train_idx, rep_shapes, allocation_method,
                           importance_method, budget, internal_evaluator,
                           augmentation_names, augmentation_strategy,
                           step_size=5, patience=20, greedy_tolerance=0.0,
                           augmentation_margin=0.0,
                           augmentation_fraction=None,
                           dataset_name=None, labels=None, groups=None):
    """Hash fold + hybrid config into a cache key.

    dataset_name, labels and groups are part of the key: row indices and
    representation shapes alone can repeat across two datasets of the same
    size, and the same rows under different labels or a different grouping
    select different features.
    """
    h = hashlib.sha256()
    h.update(train_idx.tobytes())
    h.update(f"ds:{dataset_name}".encode())
    if labels is not None:
        h.update(np.ascontiguousarray(labels).tobytes())
    if groups is not None:
        h.update(np.ascontiguousarray(
            np.asarray(groups).astype('U')).tobytes())
    for name, shape in sorted(rep_shapes.items()):
        h.update(f"{name}:{shape}".encode())
    h.update(f"{allocation_method}|{importance_method}|{budget}|{internal_evaluator}|{step_size}|{patience}|{greedy_tolerance}".encode())
    h.update(f"aug_strategy:{augmentation_strategy}".encode())
    h.update(f"aug_margin:{float(augmentation_margin)}".encode())
    h.update(f"aug_fraction:{augmentation_fraction}".encode())
    if augmentation_names:
        h.update(f"aug:{'|'.join(sorted(augmentation_names))}".encode())
    return h.hexdigest()[:24]


# The only create_hybrid parameters create_hybrid_cached may forward through
# **kwargs: every one of them is hashed into _hybrid_fold_cache_key below.
# Anything else changes the selected features without changing the key.
_HYBRID_FOLD_KEY_KWARGS = frozenset({
    'step_size', 'patience', 'greedy_tolerance',
    'augmentation_margin', 'augmentation_fraction',
})


def create_hybrid_cached(train_reps, y_train, train_idx,
                         # Tracks create_hybrid's own default, which moved from
                         # 'top_3' on 2026-09-10 (FRAMEWORK_FINALIZATION.md 3.1).
                         # Three doors into one engine must not disagree about
                         # what "no allocation method given" means. The method is
                         # in this function's fold cache key below, so a changed
                         # default writes new keys rather than reusing old ones.
                         allocation_method='performance_weighted',
                         importance_method='lightgbm_gain',
                         budget=100,
                         # Tracks create_hybrid's own default too (hybrid.py:4912),
                         # which moved from 'svm' to 'lightgbm' on 2026-07-16. The
                         # 2026-09-10 pass tracked the allocation method through all
                         # three doors and left this one behind, so a caller that
                         # named no evaluator split the budget with an SVM here and
                         # with LightGBM through create_hybrid. The evaluator IS in
                         # the fold cache key below, so nothing stored was ever
                         # served wrongly; a changed default writes new keys.
                         internal_evaluator='lightgbm',
                         augmentations=None,
                         augmentation_strategy='uncapped',
                         groups=None, dataset_name=None,
                         **kwargs):
    """Fold-aware cached create_hybrid.

    Caches feature_info (selected indices) keyed by fold train indices +
    hybrid config. The actual X_hybrid is reconstructed from feature_info
    since it's just index selection on the rep arrays.

    Args:
        train_reps: {rep_name: train_features} for this fold.
        y_train: Training labels for this fold.
        train_idx: np.ndarray of training sample indices (used for cache key).
        groups: this fold's training groups. Without them create_hybrid cuts
            its internal split at random, so each representation's share of
            the budget is decided on rows that share scaffolds with their own
            training rows.
        dataset_name: part of the cache key, so two datasets of the same size
            cannot serve each other's features.
        Other args: same as create_hybrid.

    Returns:
        (X_hybrid, feature_info) — same as create_hybrid.
    """
    rep_shapes = {name: X.shape for name, X in train_reps.items()}
    aug_names = sorted(augmentations.keys()) if augmentations else None

    # Anything create_hybrid accepts that this key cannot represent is REFUSED
    # rather than forwarded. Measured 2026-09-10: eighteen create_hybrid
    # parameters changed which features were selected while leaving the key
    # byte-identical, so the second call was handed the first call's selection.
    # A margin of 0.0 kept all twenty extra features and a margin of 5.0 kept
    # none, and both wrote one file. Refusing is the only version of this that
    # stays right when create_hybrid gains a parameter later.
    _unkeyed = sorted(set(kwargs) - _HYBRID_FOLD_KEY_KWARGS)
    if _unkeyed:
        raise ValueError(
            f"create_hybrid_cached cannot key on {_unkeyed}. Each of these "
            f"changes which features create_hybrid selects, and none of them "
            f"reaches the per-fold cache key, so two calls differing only in "
            f"one of them share a file and the second is served the first's "
            f"selection. Call kirby.hybrid.create_hybrid directly, or add the "
            f"parameter to _hybrid_fold_cache_key and to "
            f"_HYBRID_FOLD_KEY_KWARGS.")

    key = _hybrid_fold_cache_key(
        train_idx, rep_shapes, allocation_method, importance_method,
        budget, internal_evaluator, aug_names, augmentation_strategy,
        step_size=kwargs.get('step_size', 5),
        patience=kwargs.get('patience', 20),
        greedy_tolerance=kwargs.get('greedy_tolerance', 0.0),
        # Both were parameters of the key function that nothing ever passed, so
        # they hashed their defaults on every call while the caller's real value
        # went to create_hybrid through **kwargs. float() mirrors create_hybrid's
        # own coercion so 0 and 0.0 cannot write two files for one selection.
        augmentation_margin=float(kwargs.get('augmentation_margin', 0.0) or 0.0),
        augmentation_fraction=(None if kwargs.get('augmentation_fraction') is None
                               else float(kwargs['augmentation_fraction'])),
        dataset_name=dataset_name, labels=y_train, groups=groups,
    )
    cache_dir = _hybrid_fold_cache_dir()
    cache_path = cache_dir / f'{key}.pkl'

    if cache_path.exists():
        try:
            feature_info = pickle.loads(cache_path.read_bytes())
            X_hybrid = apply_feature_selection(train_reps, feature_info)
            # Reconstruct augmentation features if present
            aug_info = feature_info.get('augmentation_info')
            if aug_info and augmentations:
                from kirby.hybrid import apply_augmentation_selection
                X_aug = apply_augmentation_selection(augmentations, aug_info)
                if X_aug is not None:
                    X_hybrid = np.hstack([X_hybrid, X_aug])
            return X_hybrid, feature_info
        except Exception:
            pass  # corrupted, recompute

    X_hybrid, feature_info = create_hybrid(
        train_reps, y_train, groups=groups,
        allocation_method=allocation_method,
        importance_method=importance_method,
        budget=budget,
        internal_evaluator=internal_evaluator,
        augmentations=augmentations,
        augmentation_strategy=augmentation_strategy,
        **kwargs,
    )

    try:
        fd, tmp = tempfile.mkstemp(dir=str(cache_dir), suffix='.tmp')
        fd_closed = False
        try:
            os.write(fd, pickle.dumps(feature_info))
            os.close(fd)
            fd_closed = True
            os.replace(tmp, str(cache_path))
        except Exception:
            if not fd_closed:
                os.close(fd)
            try:
                os.unlink(tmp)
            except OSError:
                pass
    except Exception:
        pass  # non-fatal

    return X_hybrid, feature_info


# =============================================================================
# REPRESENTATION CACHING
# =============================================================================

def _rep_cache_dir() -> Path:
    """Return cache directory for pre-computed representations."""
    cache_root = Path(os.environ.get('KIRBY_CACHE_DIR',
                      str(Path.home() / '.cache' / 'kirby')))
    d = cache_root / 'rep_cache'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ab_row_identity(heavy_seqs, light_seqs):
    """One string per antibody: its heavy and light chains."""
    light = light_seqs if light_seqs is not None else [''] * len(heavy_seqs)
    return [f'{h}|{l}' for h, l in zip(heavy_seqs, light)]


def _cache_rep(dataset_name: str, rep_name: str, compute_fn,
               identity=None) -> np.ndarray:
    """Load rep from cache if available, otherwise compute and save.

    `identity` is one string per row, in row order — see _cache_rep_dict. It is
    checked against `_identity.json` beside the arrays, so a cache built from a
    different set of rows raises instead of being paired with the wrong labels.
    """
    cache_dir = _rep_cache_dir() / dataset_name
    _check_or_write_rep_identity(cache_dir, identity)
    cache_file = cache_dir / f'{rep_name}.npy'

    if cache_file.exists():
        arr = as_rep_dtype(np.load(cache_file))
        print(f"    {rep_name}: cached ({arr.shape[1]}D)")
        return arr

    arr = as_rep_dtype(compute_fn())
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_rep_identity_now(cache_dir, identity)
    tmp_path = cache_file.with_suffix('.tmp.npy')
    try:
        np.save(str(tmp_path), arr)
        os.replace(str(tmp_path), str(cache_file))
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    print(f"    {rep_name}: computed + cached ({arr.shape[1]}D)")
    return arr


def _save_rep_npy(cache_dir: Path, rep_name: str, arr: np.ndarray) -> None:
    """Atomically write one rep array into the per-dataset rep cache."""
    cache_file = cache_dir / f'{rep_name}.npy'
    if cache_file.exists():
        return
    tmp_path = cache_file.with_suffix('.tmp.npy')
    try:
        np.save(str(tmp_path), as_rep_dtype(arr))
        os.replace(str(tmp_path), str(cache_file))
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _rep_identity_path(cache_dir: Path) -> Path:
    return cache_dir / '_identity.json'


def _rep_identity(items) -> dict:
    """What the arrays in a rep cache are paired with: rows, and a hash.

    `items` is one string per row, in row order — SMILES for a molecular set,
    the sequence for a nucleotide set, the sample id for TCGA.
    """
    import hashlib
    h = hashlib.sha1()
    for it in items:
        h.update(str(it).encode('utf-8', 'replace'))
        h.update(b'\x00')
    return {'n_rows': len(items), 'sha1': h.hexdigest()}


def _check_or_write_rep_identity(cache_dir: Path, items) -> None:
    """Refuse a rep cache that was built from DIFFERENT rows.

    Until 2026-09-10 nothing recorded which molecules a cached array belonged
    to. The only check was the row count, in `_finalize_load_data`, so two
    loads that differed in WHICH rows they held but not HOW MANY silently
    paired one load's features with another load's labels. qm9 is the case
    that showed it: `load_qm9` looked for its list of RDKit-valid molecules at
    a path relative to the working directory, so a job started outside the
    repo root sampled 10,000 molecules out of 130,831 instead of out of
    129,428 — a different set, 9,977 rows against 9,978.

    A cache with no identity file gets one stamped from the load that finds it
    missing, marked as such: it is evidence about THIS load, not about the run
    that built the arrays.
    """
    if items is None:
        return
    import json as _json
    ident = _rep_identity(items)
    path = _rep_identity_path(cache_dir)
    if path.exists():
        try:
            stored = _json.loads(path.read_text())
        except Exception:
            stored = None
        if stored and stored.get('sha1') and stored['sha1'] != ident['sha1']:
            raise ValueError(
                f"Representation cache {cache_dir} was built from a DIFFERENT "
                f"set of rows: it holds {stored.get('n_rows')} rows with hash "
                f"{str(stored.get('sha1'))[:12]}, this load has "
                f"{ident['n_rows']} rows with hash {ident['sha1'][:12]}. "
                f"Row counts alone would not have caught this. Either the "
                f"loader picked a different sample (check any warning it "
                f"printed) or the cache belongs to another version of the "
                f"dataset — purge it: python tests/purge_rep_cache.py "
                f"{cache_dir.name} --delete")
        return
    if cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        ident['stamped_by'] = 'a later load, not the run that built the arrays'
        path.write_text(_json.dumps(ident, indent=2))
        print(f"    [identity] recorded {ident['n_rows']} rows for "
              f"{cache_dir.name} (first load since identity checking existed)")


def _write_rep_identity_now(cache_dir: Path, items) -> None:
    """Record the rows the arrays being written were built from."""
    if items is None:
        return
    import json as _json
    ident = _rep_identity(items)
    ident['stamped_by'] = 'the run that built the arrays'
    _rep_identity_path(cache_dir).write_text(_json.dumps(ident, indent=2))


def _cache_rep_dict(dataset_name: str, compute_all_fn,
                    filter_names: Optional[List[str]] = None,
                    compute_subset_fn=None,
                    identity=None) -> Dict[str, np.ndarray]:
    """Cache a function that returns {rep_name: array} dict.

    Loads individual reps from cache. When some are missing:
      - if compute_subset_fn is given, only the MISSING reps are generated
        (compute_subset_fn(missing_names) -> {name: array});
      - otherwise compute_all_fn() regenerates the whole preset.

    The subset path matters because compute_all_fn re-enters every rep's
    generator on a single miss. Callers that can narrow their preset to a
    name list should pass compute_subset_fn.

    `identity` is one string per row, in row order (SMILES / sequence / sample
    id). It is hashed into `_identity.json` beside the arrays and checked on
    every later load, so a cache built from a different set of rows raises
    instead of being paired with the wrong labels.
    """
    cache_dir = _rep_cache_dir() / dataset_name
    _check_or_write_rep_identity(cache_dir, identity)
    reps = {}
    missing = []

    # First pass: load what's cached
    if filter_names:
        wanted = filter_names
    else:
        # If no filter, check what's cached and what compute_all_fn would produce
        # We don't know the keys without calling it, so just call it
        wanted = None

    if wanted is not None:
        for rn in wanted:
            cache_file = cache_dir / f'{rn}.npy'
            if cache_file.exists():
                reps[rn] = as_rep_dtype(np.load(cache_file))
                print(f"    {rn}: cached ({reps[rn].shape[1]}D)")
            else:
                missing.append(rn)

        if not missing:
            return reps

        if compute_subset_fn is not None:
            print(f"    Computing {len(missing)} missing rep(s): {missing}")
            # get_molecular_representations raises "No representations could be
            # generated!" when every rep in the subset it was handed gets
            # dropped. That is correct when it is building the whole pool and
            # nothing survives, but here it is building only the MISSING subset
            # while `reps` already holds everything that came from cache. A
            # single undrawable rep in the subset then kills a run that has, in
            # practice, all the representations it needs.
            #
            # Measured 2026-09-25: simopt on herg_ki and adme_mdr1 loaded 13 and
            # 14 representations from cache, needed only `grover`, could not
            # build it (GROVER repo not installed), and raised — while caco2,
            # which needed nothing, succeeded. Two of three datasets lost to a
            # rep none of them use.
            #
            # The drop-and-warn contract three lines below is what this path is
            # supposed to follow, so follow it here too: a subset that produces
            # nothing is a warning, not a failure, PROVIDED something was
            # cached. With no cached reps either, the original error stands.
            try:
                new_reps = compute_subset_fn(missing)
            except ValueError as _e:
                if not reps:
                    raise
                print(f"    WARNING: none of the {len(missing)} missing rep(s) "
                      f"{missing} could be generated ({_e}); continuing with "
                      f"the {len(reps)} already cached.")
                return reps
            cache_dir.mkdir(parents=True, exist_ok=True)
            _write_rep_identity_now(cache_dir, identity)
            for k, v in new_reps.items():
                _save_rep_npy(cache_dir, k, v)
                print(f"    {k}: computed + cached ({v.shape[1]}D)")
            # A rep that failed to generate is simply absent — same
            # drop-and-warn contract as the compute_all_fn path.
            reps.update({k: v for k, v in new_reps.items() if k in wanted})
            return reps

    # Compute all, cache each, return wanted
    print(f"    Computing representations...")
    all_reps = compute_all_fn()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_rep_identity_now(cache_dir, identity)
    for k, v in all_reps.items():
        _save_rep_npy(cache_dir, k, v)
        print(f"    {k}: computed + cached ({v.shape[1]}D)")

    if filter_names:
        return {k: v for k, v in all_reps.items() if k in filter_names}
    return all_reps


# =============================================================================
# MODE CONFIGURATIONS — derived from the DEFAULT_* constants in find_best_hybrid.py
# =============================================================================
# No standalone config dicts here. Experiments use the DEFAULT_* constants directly.
# The only experiment-level configs are: n_splits, models, budgets.
# These are set per-experiment or via get_config(args).

# Canonical budget lists — used by get_config() and all 4 helpers' run_all().
# Change here to change everywhere.
FULL_BUDGETS = [10, 25, 50, 100, 250, 500, 1000, 2000]
DISTANCE_BUDGETS = [10, 25, 50, 100, 250, 500]
# Internal evaluators for the DISTANCE-pool hybrid search (s04/s05/s06/
# s10/s13/c03). Ridge REMOVED everywhere (2026-07-16, hybrid rebuild Phase 0.1)
# — the earlier "keep ridge as a distance-pool evaluator" carve-out (2026-06-08)
# is superseded. Removing it changes what the distance-tier search computes, so
# distance-experiment search caches for any re-scored dataset must be deleted
# before re-running (reps/evaluator set are NOT in the cache key).
DISTANCE_EVALUATORS = ['knn', 'rf', 'dnn', 'lightgbm', 'svm']

# Downstream models for the DISTANCE-pool lineage. DOWNSTREAM_MODELS plus knn.
# s02/s04/s05/s06/s08/s10/s13 all ask for a knn-scored config: nearest-
# neighbour prediction is the thing those experiments measure, not a downstream
# evaluator choice. knn is tier 'extra' as of 2026-09-04, so models=None no
# longer scores it and every one of those lookups would return (None, None).
# Naming it here keeps it in the distance searches only.
#
# This list is part of the search cache key, so it MUST be passed on BOTH sides:
# the three distance precompute sites (populate_precomputed_searches,
# similarity_helpers.run_all, precompute_hybrids.py) and the read side
# (get_best_search_for_downstream's distance branch). Change one and the reads
# miss, cold-rebuilding a search that already exists on disk.
# Defined just below DOWNSTREAM_MODELS, which it builds on.

# The 250-row gate on `capped` was REMOVED 2026-08-27 by the owner.
#
# It existed because the OLD capped refitted a model and ran a
# permutation-importance pass for EVERY candidate feature, which is what
# 'has only ever completed for n<=242' measured. capped was rebuilt on
# 2026-08-25: one ranking fit, one base-ranking fit and five scoring fits
# per fold. The gate outlived its reason and was excluding capped from
# every molecular dataset -- logd 5,028 rows, herg_ki 1,412, buchwald
# 3,955 -- and capped is the only method where an extra feature has to be
# worth more than a base one.
#
# capped is now emitted for every dataset. If a precompute job overruns
# because of it, shorten that job -- do not put the gate back.

# Large-dataset experiment gating (2026-06-28). The uncached experiments
# (c05 learning-curves, c07 partitions, c13 forward-compat) re-search from
# scratch and blow the 7-day wall on big datasets. Above LARGE_DATASET_N the
# gate kicks in: c05 caps its learning curve at LARGE_SUBSAMPLE_K rows instead
# of going to 100% of n, c07 is skipped, c13 subsamples to LARGE_SUBSAMPLE_K
# (c17 already subsamples to 2000 internally). Chosen so only the genuine
# wall-blockers (nt_*/gb_*/DTA, all >5k) are gated; everything <=adme-size runs
# full. qm9 is loaded at 10000 (user decision 2026-09-02, reverting the 5000 that
# rode in on 6064089, an allocator commit) so it now trips this gate: c05, c07 and
# c13 take the gated path for qm9. That is the intended trade -- the qm9 rep pool
# was selected by rep pruning at ~10k, and its scaffold count nearly doubles
# (2,015 -> 3,457) against 5000. A 5000 slice was tried again on 2026-10-06 and
# reverted the same day: load_qm9 draws random.sample(range(total), n), so 5000
# is a DIFFERENT set of molecules, not the first half, and the qm9
# representation cache refuses rows it was not built from
# (_check_or_write_rep_identity).
LARGE_DATASET_N = 5000
LARGE_SUBSAMPLE_K = 2000

# NOTE: the large-n feasibility gate (limit 'uncapped'/distance cost on big
# datasets) was prototyped then REVERTED on 2026-06-09 — it was a methodology
# decision that needs a proper discussion with the user first (see
# REPRODUCIBILITY_LEDGER.md §K/§P/§R). No gate is active right now.

# Core downstream models (lit-validated across molecular, antibody, DTA, TCGA, reaction).
# RF + XGBoost (tree ensembles), SVM (kernel), DNN (neural), LightGBM (boosting).
# Ridge REMOVED 2026-07-16 (hybrid rebuild Phase 0.1) — never a downstream model again.
# MLP REPLACED BY DNN 2026-09-04 (owner). DNN settings live in kirby/models/dnn.py:
# (256,128,64), dropout 0.2, lr 1e-3, 200 epochs, batch 256, patience 20, Adam +
# ReduceLROnPlateau, 80/20 internal holdout, targets z-normalised.
#
# This list MUST stay equal to the models=None default set in find_best_hybrid.py
# (tiers 'fast' + 'full'). Every precomputed search runs models=None, so a name
# here that is not in that set gets no blueprint: get_best_search_for_downstream
# returns (None, None) and the experiment writes NO rows for it, silently. gp and
# knn are tier 'extra' and are therefore NOT in this list; name them directly in
# an experiment that wants them.
DOWNSTREAM_MODELS = ['rf', 'xgboost', 'svm', 'dnn', 'lightgbm']

# Distance-pool downstream models (see DISTANCE_EVALUATORS above for why knn).
DISTANCE_MODELS = DOWNSTREAM_MODELS + ['knn']

# Main-pool downstream models for an experiment that needs knn scored. Same five
# names plus knn, same shape as DISTANCE_MODELS, but for tier 'general'.
#
# s02 reads the general lineage, not the distance one, and asks it for a
# knn-scored recipe. knn is tier 'extra' as of 2026-09-04, so models=None scores
# no knn and the general searches on disk contain none: the s02 lookup finds no
# knn among the models the search scored and returns nothing.
#
# knn stays tier 'extra'. Naming this list here is what keeps models=None — which
# every other precompute uses — untouched. The model list is normalised into the
# search cache key (find_best_hybrid._make_search_meta), so a search run with
# this list is a SEPARATE cache entry from the models=None one: nothing already
# precomputed is invalidated, and the knn-scored general search is an extra file
# rather than a replacement. It MUST be passed on both sides, by whatever
# precomputes it and by whatever reads it, or the read misses and rebuilds a
# search that is already on disk.
MAIN_MODELS_WITH_KNN = DOWNSTREAM_MODELS + ['knn']

# Main-pool downstream models for an experiment that needs Ridge scored. Same
# five names plus 'ridge', the same shape as DISTANCE_MODELS and
# MAIN_MODELS_WITH_KNN above, and added for the same reason.
#
# c16 compares three model families and names all three: 'ridge' is the Ridge
# family (sklearn Ridge alpha=1.0 / RidgeClassifier alpha=1.0), 'svm' is SVR/SVC
# and 'rf' is RandomForest. 'ridge' is NOT auto-selected — in
# find_best_hybrid.DEFAULT_MODELS it carries tier 'off', which like 'extra' is
# outside AUTO_SELECTED_TIERS — so models=None, which every precompute uses,
# scores no ridge and c16's lookup for it returns nothing. (The decision text
# said tier 'extra'; the table on disk says 'off'. Both are excluded from
# AUTO_SELECTED_TIERS, so models=None behaves the same either way — but the two
# labels promise a reader different things, and this list does not change it.)
#
# Naming the list here is what keeps models=None untouched. The model list is
# normalised into the search cache key (find_best_hybrid._make_search_meta), so
# a search run with this list is a SEPARATE cache entry: nothing already
# precomputed is invalidated, and the ridge-scored search is an extra file
# rather than a replacement. It MUST be passed on BOTH sides — by whatever
# precomputes it (tests/precompute_hybrids.py --with-ridge) and by whatever
# reads it — or the read misses and cold-rebuilds a search that is on disk.
#
# One caveat that is NOT handled here: the classifier this name resolves to is
# RidgeClassifier, which has no predict_proba, so a binary-classification
# dataset whose metric is auc cannot score it.
RIDGE_MODELS = DOWNSTREAM_MODELS + ['ridge']


# Pruning modes — single source of truth for adaptive-pruning configuration.
# Used by both tests/precompute_hybrids.py (precompute) and the experiment
# helpers (run_all in complexity/similarity/noise).
#
# Each mode bundles:
#   - schedule: global multi-stage schedule applied to every model.
#   - model_overrides: per-model schedule overrides (or None).
#   - min_samples_for_overrides: dataset n_samples threshold below which
#     model_overrides are dropped. Mid-flight pruning on small datasets
#     risks dropping the actual winner because fold-0 noise dominates;
#     the gate prevents this.
#
# Calibration source: f16_multistage_pruning sweep on f15 data (2026-04-26).
# For caco2, herg_ki, logd, buchwald_hartwig (n>=2000), td=0.5 at ckpt=[50]
# preserves best_by_model[m] within 2% for SVM/MLP while saving ~48% of
# those models' fold-evals. For flab_pk_jain_fcrnrt (n=132) and abagym_EGFR,
# MLP/Ridge fail at td>=0.3 — hence the n_samples gate at 1000.
PRUNING_MODES = {
    # THE CALIBRATED SCHEDULE. Set on the f33 record 2026-09-16/17 -- five
    # datasets, 5100 configurations each, five folds -- and written down in
    # FRAMEWORK_FINALIZATION.md 7.3. SAFE there means every one of a dataset's
    # 25 (model, budget) conditions returned the IDENTICAL best five-fold score
    # as the fully unpruned grid, not a score within some tolerance. It saves
    # 13% to 44% of the time, 30% across the five.
    #
    # ⛔ THE THREE MODES BELOW PREDATE IT AND NONE OF THEM MATCHES IT.
    # `default_pruning` empties the checkpoints, which is most of the saving and
    # the only rule that is a floor under every dataset, and uses 0.15 where the
    # calibration says 0.20. `aggressive_pruning` drops at budgets 25 and 50 as
    # well as 100; only 100 was validated, and on that record the old
    # best-of-3-at-0.15 setting deletes rank_exp_d0.3 and wglobal_power2 on
    # tcga_brca_pam50 -- the two methods that win most of its conditions -- and
    # damages 17 of its 25. They are kept because 24 call sites name them and
    # because f15/f16 results were produced under them; they are not the setting
    # to run anything new with.
    #
    # It is a reference to DEFAULT_PRUNING_SCHEDULE, not a copy, so it cannot
    # drift from the engine the way those three did.
    'calibrated': {
        'description': (
            'The schedule calibrated on the f33 record, 2026-09-16/17: vote '
            'after folds 0 and 1, per-axis votes on representation count, '
            'importance method and internal evaluator, and a tail drop at '
            'budget 100. Measured to lose no winner on any of the five '
            'recorded datasets.'
        ),
        'schedule': None,   # filled in below from the engine
        'model_overrides': None,
        'min_samples_for_overrides': 0,
    },
    # Nothing is dropped: every allocation method in the list is searched on
    # every fold. This is the precompute default from 2026-09-08 (user
    # decision). Under any pruning mode, WHICH allocation methods survive is
    # decided by counting fold-0 wins across the models and budgets in that one
    # call -- so a model added in a later run searches a different, shorter
    # list than the models before it, and the two cannot be put in one table.
    # The experiments keep their own mode; only precompute defaults to this.
    'no_pruning': {
        'description': (
            'No pruning of any kind. Every allocation method is evaluated on '
            'every fold, so a model precomputed later covers exactly the same '
            'recipes as the models precomputed before it.'
        ),
        'schedule': {
            'checkpoints': [],
            'tail_drop_per_checkpoint': 0.0,
            'final_strategy_threshold': 0.0,
            'min_configs_per_model': 1,
        },
        'model_overrides': None,
        'min_samples_for_overrides': 0,
    },
    'default_pruning': {
        'description': (
            'Existing single-stage strategy pruning (adaptive_pruning=0.15) '
            'only. No mid-flight pruning on any model. Bit-identical to '
            'pre-Phase-2 behaviour and to f15-validated default.'
        ),
        'schedule': {
            'checkpoints': [],
            'tail_drop_per_checkpoint': 0.0,
            'final_strategy_threshold': 0.15,
            'min_configs_per_model': 1,
        },
        'model_overrides': None,
        'min_samples_for_overrides': 0,
    },
    'aggressive_pruning': {
        'description': (
            'F16-validated aggressive schedule applied globally to all '
            'models with no n_samples gate: checkpoints=[25, 50, 100], '
            'tail_drop=0.5. F16 calibration (caco2 n=2158, 2026-04-25): '
            '58.0% eval reduction, top-3 overlap=1.0, worst per-model '
            'relative loss=0%. Replaces prior conservative setting '
            '(checkpoints=[50], gated at n>=1000 for gp/svm/mlp only) '
            'because that gate produced zero pruning on antibody/small '
            'datasets where it was most needed.'
        ),
        'schedule': {
            'checkpoints': [25, 50, 100],
            'tail_drop_per_checkpoint': 0.5,
            'final_strategy_threshold': 0.15,
            'min_configs_per_model': 1,
        },
        'model_overrides': None,
        'min_samples_for_overrides': 0,
    },
}


# Filled from the engine rather than copied, so 'calibrated' cannot drift from
# DEFAULT_PRUNING_SCHEDULE the way the three older modes drifted.
def _fill_calibrated_schedule():
    from kirby.find_best_hybrid import DEFAULT_PRUNING_SCHEDULE
    PRUNING_MODES['calibrated']['schedule'] = dict(DEFAULT_PRUNING_SCHEDULE)


_fill_calibrated_schedule()


def resolve_pruning_mode(mode_name, override_checkpoints=None,
                         override_tail_drop=None, override_min_samples=None):
    """Return (schedule, model_overrides_template, min_samples_gate).

    `model_overrides_template` is the unconditional per-model dict; the
    n_samples gate is applied by the caller once data is loaded.
    """
    if mode_name not in PRUNING_MODES:
        raise ValueError(f"Unknown pruning mode '{mode_name}'. "
                         f"Choose from {list(PRUNING_MODES.keys())}")
    mode = PRUNING_MODES[mode_name]
    schedule = dict(mode['schedule'])
    model_overrides = (
        {k: dict(v) for k, v in mode['model_overrides'].items()}
        if mode['model_overrides'] else None
    )
    min_samples = int(mode['min_samples_for_overrides'])
    if model_overrides is not None:
        if override_checkpoints is not None:
            for m in model_overrides:
                model_overrides[m]['checkpoints'] = list(override_checkpoints)
        if override_tail_drop is not None:
            for m in model_overrides:
                model_overrides[m]['tail_drop_per_checkpoint'] = float(
                    override_tail_drop)
    if override_min_samples is not None:
        min_samples = int(override_min_samples)
    return schedule, model_overrides, min_samples


def apply_pruning_gate(model_overrides, n_samples, min_samples_gate):
    """Return model_overrides if n_samples >= gate else None.

    Centralized gate logic so precompute and experiment runners agree.
    """
    if model_overrides is None:
        return None
    if n_samples < min_samples_gate:
        return None
    return model_overrides


def own_search_kwargs(n_samples, models=None, dataset_label=None,
                      dataset_name=None, tier='general'):
    """find_best_hybrid kwargs for an experiment that builds its OWN search.

    c05 searches a subset of the rows, c33 searches inside each outer fold,
    c15c searches one budget at a time. Nothing a stored search holds can
    match those searches, and KIRBY_NO_COLD_SEARCH does not refuse them
    (own_search=True).

    Since 2026-10-05 the axes are the dataset's decided grid
    (decided_grid.apply_decided_grid, rules R1 and R3), and pruning is off
    (Adelaide 2026-10-05). Until then this named the engine's fast lists, the
    five DOWNSTREAM_MODELS and the PRUNING_MODE schedule. `dataset_name` is
    the real dataset name and is required: a decorated label
    (<ds>_n<rows>_s<seed>) goes in `dataset_label`, which only names the
    search. `models`, when given, is narrowed to the decided models; asking
    for none of them raises. `tier` containing 'distance' gets the distance
    grid. `n_samples` is kept for the callers' signature and not used.
    """
    if not dataset_name:
        raise ValueError(
            "own_search_kwargs needs dataset_name= (the real dataset name) to "
            "look up the decided grid; the fast grid and five models it used "
            "to return were replaced by DECISIONS.md.")
    from experiments.decided_grid import apply_decided_grid
    kw = {'own_search': True, 'tier': tier,
          'models': list(models) if models is not None else None}
    apply_decided_grid(kw, dataset_name)
    kw['adaptive_pruning_schedule'] = None
    kw['model_pruning_aggressiveness'] = None
    return kw


def resolve_pruning_kwargs(pruning_mode, n_samples, verbose=True,
                           dataset_label=None):
    """Convenience helper for scripts that call find_best_hybrid directly.

    Resolves the pruning_mode, applies the n_samples gate, and returns the
    kwargs dict ready to splat:

        kwargs = resolve_pruning_kwargs('aggressive_pruning', len(labels))
        result = find_best_hybrid(..., **kwargs)

    Returns a dict with two keys:
        adaptive_pruning_schedule: schedule dict
        model_pruning_aggressiveness: per-model overrides (or None after gate)

    When verbose=True, prints a single line indicating which mode is active
    and whether the gate fired.
    """
    # Pruning is off in every search since 2026-10-05 (Adelaide; handoff 2.1).
    # Scripts that call find_best_hybrid directly (trastuzumab, ginkgo, c10,
    # n20, ...) reach their search through here, so this returns no schedule
    # and no per-model aggressiveness whatever mode is named.
    if pruning_mode not in (None, '', 'no_pruning'):
        prefix = f"  [{dataset_label}] " if dataset_label else "  "
        print(f"{prefix}pruning_mode={pruning_mode} ignored: pruning is off "
              f"in every search")
    return {'adaptive_pruning_schedule': None,
            'model_pruning_aggressiveness': None}
    schedule, ovr_template, min_n = resolve_pruning_mode(pruning_mode)
    overrides = apply_pruning_gate(ovr_template, n_samples, min_n)
    if verbose:
        prefix = f"  [{dataset_label}] " if dataset_label else "  "
        if overrides is not None:
            print(f"{prefix}pruning_mode={pruning_mode}, "
                  f"overrides active for {sorted(overrides.keys())}")
        elif ovr_template is not None:
            print(f"{prefix}pruning_mode={pruning_mode}, "
                  f"overrides gated off (n={n_samples} < {min_n})")
        else:
            print(f"{prefix}pruning_mode={pruning_mode} (no per-model overrides)")
    return {
        'adaptive_pruning_schedule': schedule,
        'model_pruning_aggressiveness': overrides,
    }

# Downstream models for noise experiments. Mirrors DOWNSTREAM_MODELS — no
# tier='extra' additions, so iteration cache-hits the precompute model set.
#
# 2026-05-22: ngboost removed. It is tier='extra' and not in precompute's
# default model list, so including it here caused ngboost to be trained from
# scratch on every (rep × LNL method × noise type) combo during noise
# experiments — burning hours per dataset across the 23-dataset done set.
# n08 (estimation survey), n11 (estimation×remediation grid) and n13
# (alt distributions) still use ngboost as an *estimator* (literature
# methods, not through find_best_hybrid); n13 already gates via
# get_noise_estimator import (line 47-48). s11 uses ngboost only as a BO
# surrogate when explicitly requested. None of these were affected by this
# change — only the implicit downstream iteration was killed.
DOWNSTREAM_MODELS_NOISE = list(DOWNSTREAM_MODELS)

# Generic internal evaluators for hybrid construction.
# 'match' resolves to the downstream model type; the rest test robustness
# across inductive biases: tree-ensemble (rf), boosted (lightgbm), neural (dnn).
# Set by the owner 2026-09-04: mlp -> dnn, and svm dropped. This list is part of
# the search cache key, so changing it invalidates every cached search.
INTERNAL_EVALUATORS = ['match', 'rf', 'lightgbm', 'dnn']


def extract_top3_base_reps(search, model='rf'):
    """Extract top 3 base reps ranked by a single model from a SearchResult.

    Args:
        search: SearchResult from find_best_hybrid (must have .baselines).
        model: Model name to rank by (default 'rf').

    Returns:
        List of (rep_name, score) tuples, sorted best-first, length <= 3.

    ⛔ TIES BREAK ON REPRESENTATION NAME SINCE 2026-09-18, OWNER'S DECISION.
    Before that the sort was `scores.sort(key=lambda x: x[1], ...)` with no
    second key, so two representations on the same score kept whatever order
    `baselines` happened to iterate in. That order comes from the order the
    representations were built, which is arbitrary and is recorded nowhere, so
    the same stored rows could name a different top representation on two runs.
    Baseline scores are rounded to 4 decimals before they are stored
    (find_best_hybrid.py:5525), so exact ties are not rare.

    The score is unchanged by this. Only WHICH of two equal-scoring names is
    printed changes, and now it is the same name every run.

    `extract_top_base_reps_with_ties` below returns the same ranking plus, for
    each entry, the other representations it tied with. 38 call sites take the
    plain (name, score) tuples and are unaffected.
    """
    baselines = getattr(search, 'baselines', {})
    # Direction of the quantity BEING SORTED, which is not always
    # search.lower_is_better. find_best_hybrid keeps two flags: the dataset
    # metric's own direction, and a copy overwritten to False when the objective
    # is a blend containing prediction, because the winner is then scored on a
    # correlation. `baselines` stays in the dataset's own metric either way, so
    # under a blend on an rmse dataset the stored flag says False and this sort
    # would put the LARGEST rmse first. metric_name still holds the dataset's
    # metric, so resolve from that first and fall back to the flag only when the
    # name is not a builtin (a geometry or blend metric name, where the stored
    # flag does describe the baselines). Fixed 2026-09-10 (metric-direction audit).
    lower = object_direction(search, 'the SearchResult passed to '
                             'extract_top3_base_reps')
    scores = []
    for rep_name, model_scores in baselines.items():
        if rep_name.startswith('_'):
            continue  # skip _secondary etc.
        s = model_scores.get(model)
        if s is not None and not np.isnan(s):
            scores.append((rep_name, float(s)))
    # Score first, then representation name. `lower` flips the score half only:
    # the name half is always ascending, so the tie-break does not depend on
    # the metric's direction.
    scores.sort(key=lambda x: ((x[1] if lower else -x[1]), x[0]))
    return scores[:3]


def extract_top_base_reps_with_ties(search, model='rf', n=3, tol=0.0):
    """extract_top3_base_reps, plus which representations tied with each entry.

    Added 2026-09-18, owner's decision: break the tie by name AND report the
    tie. A tie is what makes "the best base representation" an unstable thing to
    measure against — if rank 1 and rank 2 hold the same score, an arm that
    beats rank 1 has beaten one of two that were level.

    tol: how close two scores must be to count as tied. 0.0 means exactly
    equal, which is the case worth reporting, because baseline scores are
    rounded to 4 decimals before they are stored (find_best_hybrid.py:5525) and
    so land on each other exactly. A caller that wants "within 0.001" passes it.

    Returns a list of dicts, best-first, length <= n:
        {'rep': str, 'score': float, 'rank': int, 'tied_with': [str, ...]}
    `tied_with` lists every OTHER representation within tol of this one,
    ordered by name, and is empty when nothing tied.

    Reads stored baseline rows. 0 new model fits.
    """
    baselines = getattr(search, 'baselines', {})
    lower = object_direction(search, 'the SearchResult passed to '
                             'extract_top_base_reps_with_ties')
    scores = []
    for rep_name, model_scores in baselines.items():
        if rep_name.startswith('_'):
            continue
        sc = model_scores.get(model)
        if sc is not None and not np.isnan(sc):
            scores.append((rep_name, float(sc)))
    scores.sort(key=lambda x: ((x[1] if lower else -x[1]), x[0]))
    out = []
    for i, (rep_name, sc) in enumerate(scores[:n]):
        tied = sorted(other for other, o_sc in scores
                      if other != rep_name and abs(o_sc - sc) <= tol)
        out.append({'rep': rep_name, 'score': sc, 'rank': i + 1,
                    'tied_with': tied})
    return out


# =============================================================================
# HONEST EVALUATION: NESTED CROSS-VALIDATION
# =============================================================================
# The honest leave-one-fold-out (LOFO) re-score that used to live here has been
# REMOVED (2026-07-13). It was a pure-arithmetic patch over a whole-dataset
# search: the hybrid recipe was chosen and scored on the SAME folds, and LOFO
# re-picked the winner per held-out fold to claw back some of that optimism.
#
# It was never equivalent to holding data out, because the per-fold scores it
# re-selected over came from models trained on data that INCLUDED the held-out
# fold, and the candidate pool had already been pruned using every fold.
#
# The replacement is real nested cross-validation: kirby.find_best_hybrid.
# nested_cv_evaluate. The outer fold is held out, the search only ever sees
# outer-train, and BOTH arms (hybrid and each individual base rep) are scored on
# the untouched outer fold. Experiments call run_nested_cv() below.
#
# Do NOT reintroduce a re-score over a whole-dataset search.


def outer_rep_builders_from_factories(rep_factories, n_samples, names=None):
    """Turn per-fold representation factories into nested-CV outer builders.

    A factory is called factory(train_idx, val_idx, train_labels) and returns
    (train_matrix, val_matrix), with everything it fits — column selection,
    z-score statistics — taken from train_idx alone. nested_cv_evaluate wants
    the other shape: builder(train_idx, y_train) returning the FULL-length
    matrix, which it then slices for the fold itself. Asking the factory to
    featurise every row as its validation half gives exactly that, and every
    row outside train_idx is featurised from training rows only.

    Use this rather than passing rep_factories straight through to
    nested_cv_evaluate. rep_factories reaches the INNER search only, so a
    representation supplied that way is invisible on the outer test fold: the
    hybrid silently loses it there, and the baseline arm never scores it at all.

    Args:
        rep_factories: {name: factory}.
        n_samples: rows in the dataset.
        names: restrict to these names (default: all of rep_factories).

    Returns: {name: builder}.
    """
    all_rows = np.arange(int(n_samples))
    wanted = list(names) if names is not None else sorted(rep_factories)

    def _make(fn):
        def _build(train_idx, y_train):
            _, full = fn(np.asarray(train_idx), all_rows, y_train)
            return np.asarray(full)
        return _build

    return {n: _make(rep_factories[n]) for n in wanted if n in rep_factories}


def _apply_dataset_search_speed(kwargs, dataset_name):
    """Put a find_best_hybrid call on the dataset's decided grid.

    Since 2026-10-04 this applies DECISIONS.md (through decided_grid.py) to
    every axis: the dataset's two models (plus knn on the distance search where
    Step 9 runs it), its budget range, importance methods, allocation methods,
    internal evaluators, representation counts, and its speed class. An axis
    the caller left as None gets the decided list; an axis the caller named is
    narrowed to the decided members and the dropped ones are printed. See
    decided_grid.apply_decided_grid.

    It used to set speed='fast' on a list of 14 datasets and leave every other
    axis at the engine default (five models, eight budgets). That list and the
    gp-on-small-datasets rule are both replaced by the record: the fast class
    is the record's ten fast-tier datasets, and gp runs exactly where Step 1
    names it.

    A dataset or search the record does not cover raises
    decided_grid.UndecidedSearch rather than running the old defaults.
    """
    from experiments.decided_grid import apply_decided_grid
    return apply_decided_grid(kwargs, dataset_name)


def run_nested_cv(reps, labels, groups, config, *,
                  dataset_name,
                  augmentations=None,
                  task_type='auto',
                  metric='auto',
                  domain=None,
                  tier='general',
                  budgets=None,
                  internal_evaluators=None,
                  n_outer_folds=None,
                  seed=42,
                  verbose=True,
                  label_noise=None,
                  augmentation_strategy=None,
                  **search_kwargs):
    """Honest hybrid-vs-baseline comparison for one dataset.

    THE standard entry point for experiments. Wraps nested_cv_evaluate with this
    repo's metric/task resolution and config conventions.

    The search runs on outer-train only, per outer fold; the hybrid AND every
    individual base representation are then scored on the untouched outer fold.
    Nothing here is selection-leaked, so the numbers can be reported as-is.

    Pass the FULL reps/labels/groups. Do NOT pre-slice — this function does the
    outer splitting.

    Returns:
        NestedCVResult. Call .to_dict() for the JSON block to persist.
    """
    from kirby.find_best_hybrid import nested_cv_evaluate

    if dataset_name is None:
        raise ValueError("run_nested_cv requires dataset_name= (cache key)")

    metric, task_type = _resolve_search_metric(
        labels, task_type=task_type, metric=metric, domain=domain,
        dataset_name=dataset_name)

    if n_outer_folds is None:
        n_outer_folds = config.get('n_splits', 5)

    # Every axis comes from the dataset's decided grid (DECISIONS.md) unless
    # the caller names it, in which case it is narrowed to the decided members.
    # config['budgets'], config['importance_methods'] and config['models'] are
    # NOT read: get_config builds them the same for every dataset, and the
    # record decides them per dataset. models used to be None here, which
    # scored five models and never gp.
    kwargs = dict(
        budgets=budgets,
        task_type=task_type,
        metric=metric,
        cache=True,
        tier=tier,
    )
    if internal_evaluators is not None:
        kwargs['internal_evaluators'] = internal_evaluators
    if label_noise is not None:
        kwargs['label_noise'] = label_noise
    kwargs.update(search_kwargs)
    _apply_dataset_search_speed(kwargs, dataset_name)

    # nested_cv_evaluate runs ONE augmentation condition per call. Step 8 cuts
    # augmentation on every dataset but ginkgo_colloidal and ginkgo_hic (distance
    # search only), so augmentation representations are dropped unless the
    # caller's condition is one the record keeps. The old default here was
    # 'uncapped' whenever augmentations were passed, never 'none'.
    from experiments.decided_grid import decided_aug_conditions
    _strategy = augmentation_strategy or 'none'
    if augmentations is None or len(augmentations) == 0:
        _strategy = 'none'
    if _strategy not in decided_aug_conditions(dataset_name, tier):
        if verbose and _strategy != 'none':
            print(f"  [decided grid] {dataset_name}: augmentation "
                  f"'{_strategy}' is cut (DECISIONS.md Step 8); running 'none'")
        _strategy = 'none'
    kwargs['augmentations'] = augmentations if _strategy != 'none' else None
    kwargs['augmentation_strategy'] = _strategy

    return nested_cv_evaluate(
        base_reps=reps,
        labels=labels,
        groups=groups,
        n_outer_folds=n_outer_folds,
        seed=seed,
        verbose=verbose,
        dataset_name=dataset_name,
        **kwargs,
    )


def precompute_nested_cv_searches(reps, labels, groups, config,
                                  augmentations=None, verbose=True,
                                  task_type='auto', metric='auto',
                                  domain=None,
                                  dataset_name=None, tier='general',
                                  adaptive_pruning_schedule=None,
                                  model_pruning_aggressiveness=None,
                                  internal_evaluators=None,
                                  budgets=None,
                                  models=None,
                                  n_outer_folds=None,
                                  seed=42,
                                  label_noise=None,
                                  objective=None):
    """Warm the per-outer-fold inner-search caches that run_nested_cv reads.

    objective: search objective forwarded to find_best_hybrid —
    'predictive' (the default when None) or 'similarity'. 'similarity_sac'
    was CUT on 2026-08-26 and now raises.
    Without it a geometry study could only ever warm predictive inner searches,
    which is why nothing in the similarity work has run through nested CV.

    Nested CV runs one search per OUTER FOLD, each on a different training
    subset. Precompute must therefore warm n_outer_folds searches per
    (dataset, aug_condition), not one.

    The outer splits and the per-fold cache namespace both come from the shared
    helpers in kirby.find_best_hybrid, so precompute and the experiment cannot
    drift apart — if they did, precompute would warm entries nothing ever reads
    and every experiment would recompute the whole search from cold.

    Returns:
        {aug_cond: {fold_idx: SearchResult}}
    """
    from kirby.find_best_hybrid import (find_best_hybrid,
                                        nested_cv_cache_namespace,
                                        nested_cv_outer_folds)

    if dataset_name is None:
        raise ValueError(
            "precompute_nested_cv_searches requires dataset_name= "
            "(used as part of the cache filename)")

    metric, task_type = _resolve_search_metric(
        labels, task_type=task_type, metric=metric, domain=domain,
        dataset_name=dataset_name)

    if n_outer_folds is None:
        n_outer_folds = config.get('n_splits', 5)

    # The decided grid (DECISIONS.md), applied once before the loop so an
    # undecided dataset raises instead of failing every fold quietly.
    _grid_kw = _apply_dataset_search_speed(
        dict(tier=tier, budgets=budgets, models=models,
             internal_evaluators=internal_evaluators), dataset_name)

    # Step 8's conditions: 'none', plus 'uncapped' only where it is kept.
    from experiments.decided_grid import decided_aug_conditions
    aug_conditions = ['none']
    if augmentations is not None and len(augmentations) > 0:
        aug_conditions += [c for c in decided_aug_conditions(dataset_name, tier)
                           if c != 'none']

    outer_folds = nested_cv_outer_folds(
        reps, labels, groups, n_outer_folds, seed, task_type)
    n_outer = len(outer_folds)

    from kirby.result_cache import StaleResultDB as _StaleResultDB

    results = PrecomputeResults()
    t_start = time.time()

    for idx, aug_cond in enumerate(aug_conditions, 1):
        augs = augmentations if aug_cond != 'none' else None
        aug_strat = aug_cond
        results[aug_cond] = {}

        for fold_i, (tr, _te) in enumerate(outer_folds):
            if verbose:
                print(f"  [{idx}/{len(aug_conditions)}] aug={aug_cond} "
                      f"outer-fold {fold_i + 1}/{n_outer} "
                      f"(train={len(tr)}) ...", end='', flush=True)
            t0 = time.time()
            try:
                _fbh_kwargs = dict(
                    base_reps={k: v[tr] for k, v in reps.items()},
                    labels=labels[tr],
                    groups=(groups[tr] if groups is not None else None),
                    augmentations=({k: v[tr] for k, v in augs.items()}
                                   if augs else None),
                    augmentation_strategy=aug_strat,
                    task_type=task_type,
                    metric=metric,
                    cache=True,
                    verbose=verbose,
                    dataset_name=dataset_name,
                    tier=tier,
                    cache_namespace=nested_cv_cache_namespace(
                        fold_i, n_outer, seed),
                    adaptive_pruning_schedule=adaptive_pruning_schedule,
                    model_pruning_aggressiveness=model_pruning_aggressiveness,
                )
                _fbh_kwargs.update(_grid_kw)
                if label_noise is not None:
                    _fbh_kwargs['label_noise'] = label_noise
                if objective is not None:
                    _fbh_kwargs['objective'] = objective
                search = find_best_hybrid(**_fbh_kwargs)
                results[aug_cond][fold_i] = search
                if verbose:
                    print(f" done ({time.time() - t0:.1f}s)")
            except _StaleResultDB:
                # Same reason as the other three: a store written under an
                # older column list fails every fold, and a None fold is a gap
                # the caller fills from the remaining folds rather than a stop.
                raise
            except Exception as e:
                # Fold value stays None; the exception goes into
                # results.failures (fold index in the record) so the caller
                # can read which folds failed instead of scraping stdout.
                results[aug_cond][fold_i] = None
                rec = results.record_failure(
                    e, tier=tier, dataset=dataset_name, aug_condition=aug_cond)
                rec['fold'] = fold_i
                print(f" FAILED ({type(e).__name__}: {e})")

    if verbose:
        print(f"  precompute (nested CV) total: {time.time() - t_start:.1f}s")
        if results.failures:
            _failed = ', '.join(f"{f['aug_condition']}/fold{f['fold']}"
                                f"={f['error_type']}" for f in results.failures)
            print(f"  Precompute failures recorded ({len(results.failures)}): "
                  f"{_failed}")

    return results


# =============================================================================
# PER-FOLD RECIPE SEARCH
# The whole-dataset pool (get_best_search_for_downstream / get_fitted_pool_hybrid)
# ranks recipes by a score measured over every row, so an experiment that then
# reports a held-out score has chosen its allocation method, importance method,
# internal evaluator and budget using the rows it is about to be scored on.
# These two functions run the same search on ONE fold's TRAINING rows and hand
# back the same (search, metadata) pair, so the recipe never sees the held-out
# fold. Split and cache namespace both come from kirby.find_best_hybrid, so a
# fold search here reads exactly what precompute_nested_cv_searches warms.
# =============================================================================

# In-process memo so one experiment asking for the same fold's search once per
# budget, per model and per condition pays for it once. Keyed by everything that
# changes which search is run; the disk cache still backs it across processes.
# (Restored 2026-10-05 by the similarity alignment chat: an edit at 18:41 by the
# complexity alignment chat removed this block together with run_nested_cv and
# precompute_nested_cv_searches, while fold_searches still reads both. The
# complexity chat restored those two from a copy read before the edit.)
_FOLD_SEARCH_MEMO: Dict[tuple, Any] = {}


def _memo_rows_id(train_idx) -> str:
    """Short hash of the exact training rows a fold holds.

    The memo used to key on len(train_idx) alone, so two outer folds of one
    dataset — the same size by construction — shared an entry and the second was
    handed the first's search.
    """
    a = np.ascontiguousarray(np.asarray(train_idx).ravel().astype(np.int64))
    return hashlib.sha1(a.tobytes()).hexdigest()[:16]


def _memo_labels_id(labels) -> str:
    """Short hash of a label vector (numeric or not), for the fold memo.
    Two label vectors on the same rows (federated two-class labels against the
    continuous ones, clean against noisy) shared one memo entry before
    2026-10-05."""
    a = np.asarray(labels)
    try:
        b = np.ascontiguousarray(a.astype(np.float64)).tobytes()
    except (TypeError, ValueError):
        b = '\x1f'.join(map(str, a.ravel().tolist())).encode()
    return hashlib.sha1(b).hexdigest()[:16]


def search_on_fold(reps, labels, groups, *, train_idx, fold_i, n_outer_folds,
                   dataset_name, augmentations=None, task_type='auto',
                   metric='auto', domain=None, tier='general',
                   budgets=None, internal_evaluators=None, models=None,
                   seed=42,
                   verbose=False, label_noise=None, objective=None,
                   adaptive_pruning_schedule=None,
                   model_pruning_aggressiveness=None,
                   cache_namespace=None, condition_suffix=None,
        adaptive_pruning=None, fixed_grid=None, train_labels=None,
        noise_groups=None):
    """Search hybrid recipes on ONE outer fold's TRAINING rows.

    train_labels: a full-length label column the search TRAINS on while it is
        scored on `labels` (find_best_hybrid's train_labels; added 2026-10-06
        for n20, whose cleaned searches train on cleaned labels and are scored
        on the labels as they are). Sliced to train_idx here.
    noise_groups: one scaffold family per molecule, full length, for a
        label_noise spec that needs it (grouped_shifted). Sliced to train_idx.

    Same per-augmentation-condition loop as precompute_hybrid_searches, with
    two differences: every array is sliced to `train_idx` first, and the search
    is filed under this fold's nested-CV cache namespace so it cannot collide
    with the whole-dataset search or with another fold's.

    Args:
        train_idx: row numbers of this fold's training rows.
        fold_i, n_outer_folds, seed: name this fold's cache namespace. They must
            match the split the caller scores on — build the folds with
            kirby.find_best_hybrid.nested_cv_outer_folds so they do.
        tier: 'general' for the main lineage, 'general_distance' for the
            distance lineage (pass DISTANCE_EVALUATORS, DISTANCE_BUDGETS and
            DISTANCE_MODELS alongside it).
        models: the downstream models the search scores. None takes
            find_best_hybrid's tier default, which is what the main lineage's
            precompute runs. The distance lineage must pass DISTANCE_MODELS or
            knn is not scored at all, and get_best_for_downstream(search, 'knn')
            then falls back to another model's blueprint.
        cache_namespace: overrides the nested-CV namespace. Pass one when the
            training subset is NOT a nested-CV outer fold — a Bayesian search's
            seed rows, say — so the search cannot be served from, or overwrite,
            a cache entry belonging to a different subset of the same size.

    Returns:
        PrecomputeResults, {aug_condition: SearchResult or None}, with one
        `.failures` record per condition that threw.
    """
    from kirby.find_best_hybrid import (find_best_hybrid,
                                        nested_cv_cache_namespace)

    if dataset_name is None:
        raise ValueError("search_on_fold requires dataset_name= (cache key)")

    train_idx = np.asarray(train_idx)
    labels = np.asarray(labels)

    metric, task_type = _resolve_search_metric(
        labels, task_type=task_type, metric=metric, domain=domain,
        dataset_name=dataset_name)

    # The decided grid (DECISIONS.md) for this dataset and search, applied
    # BEFORE the loop so an undecided dataset or a cut model raises here
    # instead of being recorded as one more failed condition. Budgets, models
    # and evaluators the caller left as None get the decided lists.
    # fixed_grid: a complete set of axes from decided_grid.
    # fast_model_search_kwargs, for an experiment that runs one model outside
    # the dataset's two on purpose (n09's dnn, Adelaide 2026-10-05). It is used
    # as given; everything else goes through the decided grid.
    if fixed_grid is not None:
        _grid_kw = dict(fixed_grid)
        tier = _grid_kw.get('tier', tier)
    else:
        _grid_kw = _apply_dataset_search_speed(
            dict(tier=tier, budgets=budgets, models=models,
                 internal_evaluators=internal_evaluators), dataset_name)

    # Augmentation conditions are Step 8's: 'none' everywhere, plus 'uncapped'
    # on the distance search of ginkgo_colloidal and ginkgo_hic. 'capped' is
    # cut on every dataset.
    from experiments.decided_grid import decided_aug_conditions
    aug_conditions = ['none']
    if augmentations is not None and len(augmentations) > 0:
        aug_conditions += [c for c in decided_aug_conditions(dataset_name, tier)
                           if c != 'none']

    namespace = (cache_namespace if cache_namespace is not None
                 else nested_cv_cache_namespace(fold_i, n_outer_folds, seed))
    from kirby.result_cache import StaleResultDB as _StaleResultDB
    results = PrecomputeResults()

    # Pruning is off in every per-fold search (Adelaide, 2026-10-05; handoff
    # 2.1). A caller that still passes a schedule, per-model aggressiveness or
    # a truthy adaptive_pruning is told, and the search runs unpruned.
    _pr = {k: v for k, v in (
        ('adaptive_pruning_schedule', adaptive_pruning_schedule),
        ('model_pruning_aggressiveness', model_pruning_aggressiveness),
        ('adaptive_pruning', adaptive_pruning)) if v}
    if _pr:
        print(f"  search_on_fold {dataset_name} fold {fold_i}: pruning "
              f"arguments {sorted(_pr)} ignored; per-fold searches run "
              f"unpruned.")
    adaptive_pruning_schedule = None
    model_pruning_aggressiveness = None
    adaptive_pruning = None

    for aug_cond in aug_conditions:
        augs = augmentations if aug_cond != 'none' else None
        try:
            _kw = dict(
                base_reps={k: v[train_idx] for k, v in reps.items()},
                labels=labels[train_idx],
                groups=(np.asarray(groups)[train_idx]
                        if groups is not None else None),
                augmentations=({k: v[train_idx] for k, v in augs.items()}
                               if augs else None),
                augmentation_strategy=aug_cond,
                task_type=task_type,
                metric=metric,
                cache=True,
                verbose=verbose,
                dataset_name=dataset_name,
                tier=tier,
                cache_namespace=namespace,
                adaptive_pruning_schedule=adaptive_pruning_schedule,
                model_pruning_aggressiveness=model_pruning_aggressiveness,
            )
            _kw.update(_grid_kw)
            if condition_suffix is not None:
                _kw['condition_suffix'] = condition_suffix
            if adaptive_pruning is not None:
                _kw['adaptive_pruning'] = adaptive_pruning
            if label_noise is not None:
                _kw['label_noise'] = label_noise
            if train_labels is not None:
                _kw['train_labels'] = np.asarray(train_labels)[train_idx]
            if noise_groups is not None:
                _kw['noise_groups'] = np.asarray(noise_groups)[train_idx]
            if objective is not None:
                _kw['objective'] = objective
            results[aug_cond] = find_best_hybrid(**_kw)
        except _StaleResultDB:
            # Same reason as the other three. See result_cache.StaleResultDB.
            raise
        except Exception as e:
            results[aug_cond] = None
            rec = results.record_failure(e, tier=tier, dataset=dataset_name,
                                         aug_condition=aug_cond)
            rec['fold'] = int(fold_i)
    return results


def fold_searches(reps, labels, groups, *, train_idx, fold_i, n_outer_folds,
                  dataset_name, augmentations=None, task_type='auto',
                  metric='auto', domain=None, tier='general', budgets=None,
                  internal_evaluators=None, models=None, seed=42,
                  verbose=False, label_noise=None, objective=None,
                  adaptive_pruning_schedule=None,
                  model_pruning_aggressiveness=None, cache_namespace=None, condition_suffix=None,
        adaptive_pruning=None, train_labels=None, noise_groups=None):
    """search_on_fold, memoised in-process: {aug_condition: SearchResult}.

    get_fold_best_search and fold_standard_selection both read through this,
    so one fold's search is run once per process however many models,
    budgets or selections ask for it.
    """
    memo_key = (
        dataset_name, tier, cache_namespace,
        int(fold_i), int(n_outer_folds), int(seed),
        tuple(budgets) if budgets is not None else None,
        tuple(internal_evaluators) if internal_evaluators is not None else None,
        tuple(models) if models is not None else None,
        objective, str(label_noise) if label_noise is not None else None,
        # Representation names AND widths: the same name can arrive at two
        # widths, and a representation's width decides how much of the budget it
        # can absorb.
        tuple(sorted((k, int(np.asarray(v).shape[1])) for k, v in reps.items())),
        tuple(sorted(augmentations.keys())) if augmentations else None,
        # The training ROWS, not how many there are. `int(len(train_idx))` said
        # only the count, so two folds of the same size -- which is every fold
        # of an evenly-split dataset -- shared one entry and the second was
        # handed the first fold's search. The metric and the task type were
        # missing too, and both change what the search optimises.
        _memo_rows_id(train_idx),
        # The labels themselves (2026-10-05): two label vectors on the same
        # rows -- the federated two-class labels against the continuous ones,
        # or clean against noisy -- shared one entry before this.
        _memo_labels_id(labels),
        metric, task_type, condition_suffix, adaptive_pruning,
        None if train_labels is None else _memo_labels_id(train_labels),
        None if noise_groups is None else _memo_labels_id(noise_groups),
    )
    searches = _FOLD_SEARCH_MEMO.get(memo_key)
    if searches is None:
        searches = search_on_fold(
            reps, labels, groups,
            train_idx=train_idx, fold_i=fold_i, n_outer_folds=n_outer_folds,
            dataset_name=dataset_name, augmentations=augmentations,
            task_type=task_type, metric=metric, domain=domain, tier=tier,
            budgets=budgets, internal_evaluators=internal_evaluators,
            models=models,
            seed=seed, verbose=verbose, label_noise=label_noise,
            objective=objective,
            adaptive_pruning_schedule=adaptive_pruning_schedule,
            model_pruning_aggressiveness=model_pruning_aggressiveness,
            cache_namespace=cache_namespace,
            condition_suffix=condition_suffix,
            adaptive_pruning=adaptive_pruning,
            train_labels=train_labels, noise_groups=noise_groups,
        )
        for _s in searches.values():
            if _s is not None:
                _attach_evaluator_rep_scores(_s)
        _FOLD_SEARCH_MEMO[memo_key] = searches

    return searches


def _attach_evaluator_rep_scores(search):
    """Give every recipe of a per-fold search the representation scores of
    ITS OWN internal evaluator (2026-10-06).

    The engine sets Blueprint.rep_scores to each representation's best score
    over every model the search ran. Rebuilds pass those to create_hybrid as
    the budget-split scores, so two recipes that differed only in their
    evaluator were rebuilt as the same hybrid (seen on c09, herg_ki). The
    search already holds each evaluator's scores for each of its inner folds
    (search.alloc_cache, keyed (inner fold, evaluator[, metric])); their mean
    over inner folds is what the search split budgets by, measured on this
    outer fold's training rows only. Higher is better, as rep_scores is.

    A recipe whose evaluator has no stored scores (a geometry objective, or a
    search read back without its alloc cache) keeps the engine's value. The
    engine's value is kept as `rep_scores_best_over_models` for provenance.
    """
    cache = getattr(search, 'alloc_cache', None) or {}
    if not cache:
        return
    metric = getattr(search, 'metric_name', None)
    by_ev = {}
    for key, scores in cache.items():
        if not isinstance(key, tuple) or len(key) < 2 or not scores:
            continue
        ev = key[1]
        if len(key) == 3 and key[2] != metric:
            continue
        slot = by_ev.setdefault(ev, {}).setdefault(
            'metric' if len(key) == 3 else 'default', {})
        for rep, v in scores.items():
            if v is not None and np.isfinite(v):
                slot.setdefault(rep, []).append(float(v))
    means = {}
    for ev, slots in by_ev.items():
        src = slots.get('metric') or slots.get('default') or {}
        means[ev] = {r: float(np.mean(v)) for r, v in src.items() if v}
    for r in getattr(search, 'all_results', None) or []:
        bp = r.get('blueprint') if isinstance(r, dict) else None
        if bp is None:
            continue
        ev_scores = means.get(getattr(bp, 'internal_evaluator', None))
        if not ev_scores:
            continue
        if getattr(bp, 'rep_scores_best_over_models', None) is None:
            try:
                bp.rep_scores_best_over_models = dict(bp.rep_scores or {})
            except Exception:
                pass
        bp.rep_scores = {rn: ev_scores[rn] for rn in bp.reps
                         if rn in ev_scores}


def get_fold_best_search(reps, labels, groups, *, train_idx, fold_i,
                         n_outer_folds, downstream_model, budget=None,
                         dataset_name, augmentations=None, distance=False,
                         task_type='auto', metric='auto', domain=None,
                         seed=42, verbose=False, objective=None,
                         label_noise=None, budgets=None,
                         internal_evaluators=None, models=None,
                         adaptive_pruning_schedule=None,
                         model_pruning_aggressiveness=None,
                         cache_namespace=None, condition_suffix=None,
        adaptive_pruning=None):
    """The recipe for one fold, chosen on that fold's TRAINING rows only.

    Drop-in replacement for get_best_search_for_downstream inside a fold loop:
    it returns the same (search, metadata) pair, with the same metadata keys —
    'tier', 'aug_condition', 'score_mean', 'model', 'budget', 'best_config' —
    so a caller keeps reading meta['best_config']['blueprint'] and
    meta['aug_condition'] unchanged.

    Returns (None, None) when no augmentation condition produced a usable
    config for (downstream_model, budget).
    """
    # Budgets, evaluators and models left as None take the dataset's decided
    # grid for this search (DECISIONS.md, through search_on_fold). The
    # distance search used to default to DISTANCE_BUDGETS, DISTANCE_EVALUATORS
    # and DISTANCE_MODELS on every dataset.
    tier = 'general_distance' if distance else 'general'

    # A model outside the dataset's decided models is refused here. Without
    # this, get_best_for_downstream falls back to another model's best recipe
    # and returns it under the requested model's name.
    from experiments.decided_grid import decided_models
    _decided = decided_models(dataset_name, tier)
    if downstream_model not in _decided:
        raise ValueError(
            f'{dataset_name} {tier}: downstream model {downstream_model!r} is '
            f'not one of the decided models {_decided} (DECISIONS.md Step 1/9).')

    searches = fold_searches(
        reps, labels, groups, train_idx=train_idx, fold_i=fold_i,
        n_outer_folds=n_outer_folds, dataset_name=dataset_name,
        augmentations=augmentations, task_type=task_type, metric=metric,
        domain=domain, tier=tier, budgets=budgets,
        internal_evaluators=internal_evaluators, models=models, seed=seed,
        verbose=verbose, label_noise=label_noise, objective=objective,
        adaptive_pruning_schedule=adaptive_pruning_schedule,
        model_pruning_aggressiveness=model_pruning_aggressiveness,
        cache_namespace=cache_namespace,
        condition_suffix=condition_suffix, adaptive_pruning=adaptive_pruning)

    candidates = []
    for aug_cond, search in searches.items():
        if search is None:
            continue
        best = get_best_for_downstream(search, downstream_model, budget=budget)
        if not best or best.get('score_mean') is None:
            continue
        if not np.isfinite(best['score_mean']):
            continue
        candidates.append((aug_cond, search, best))
    if not candidates:
        return None, None

    # Same direction guard and tie-break order as the whole-dataset pool.
    ref_lib = candidates[0][1].lower_is_better
    candidates = [c for c in candidates if c[1].lower_is_better == ref_lib]
    if not candidates:
        return None, None
    aug_pref = {'none': 0, 'uncapped': 1, 'capped': 2}
    candidates.sort(key=lambda item: (
        item[2]['score_mean'] if ref_lib else -item[2]['score_mean'],
        aug_pref.get(item[0], 99)))
    aug_cond, search, best = candidates[0]

    return search, {
        'tier': tier,
        'aug_condition': aug_cond,
        'score_mean': best['score_mean'],
        'model': downstream_model,
        'budget': budget,
        'best_config': best,
        'fold': int(fold_i),
        'n_outer_folds': int(n_outer_folds),
        # The counterpart of get_best_search_for_downstream's 'whole_dataset'.
        # Every part of the recipe here — allocation method, budget, importance
        # method, internal evaluator, augmentation condition — was scored on
        # this fold's training rows only, so a caller that also rebuilds the
        # features per fold can report the held-out score as held out.
        'selection_scope': 'fold_train_rows',
        'selection_note': ('recipe chosen by its mean over inner folds of this '
                           "outer fold's training rows only"),
    }


# The owner's standard selection (2026-10-03): per downstream model, per outer
# fold, the 2 best base representations and the 2 best hybrids, both ranked on
# that fold's training rows. For noise experiments with two models that is 8
# configurations per fold.
STANDARD_N_BASE = 2
STANDARD_N_HYBRIDS = 2


def fold_standard_selection(reps, labels, groups, *, train_idx, fold_i,
                            n_outer_folds, dataset_name, models=None,
                            distance=False, augmentations=None,
                            task_type='auto', metric='auto', domain=None,
                            seed=42, verbose=False, objective=None,
                            label_noise=None, n_base=STANDARD_N_BASE,
                            n_hybrids=STANDARD_N_HYBRIDS,
                            adaptive_pruning_schedule=None,
                            model_pruning_aggressiveness=None,
                            cache_namespace=None, condition_suffix=None,
        adaptive_pruning=None, distinct_features=True, budget=None,
        train_labels=None, noise_groups=None):
    """The 2 best base representations and 2 best hybrids per model, one fold.

    budget: when given, the hybrids are the best at that budget (features)
        only, from the same per-fold search. For experiments that sweep the
        decided budgets and need the top hybrids at each (federated suite).
        None ranks over every budget the search ran.

    Both rankings come from ONE search run on this outer fold's training rows
    (fold_searches -> search_on_fold, on the dataset's decided grid). The base
    representations are ranked by their score in that search
    (search.baselines, the mean over its inner folds); the hybrids are the top
    configurations in search.all_results for that model. Nothing is read from
    the whole-dataset search.

    models: the downstream models. None = the dataset's two decided models
    (DECISIONS.md Step 1). knn is added only when named, and only on the
    distance search where Step 9 runs it.

    distinct_features: a lower-ranked hybrid whose selected features are
        identical to a hybrid already chosen is skipped and the next one is
        taken (Adelaide, 2026-10-05, handoff 2.1 and 5b item 5). Compared on
        the selected feature numbers of every representation, and of every
        augmentation donor, after building each candidate on this fold's
        training rows -- not on the recipe's name. Skipped candidates are
        listed under 'hybrids_skipped_identical_features'.

    Returns {model: {'base': [rep names], 'base_detail': top_base_reps(...),
                     'hybrids': [top_hybrids entry, each with 'blueprint' and
                                 'aug_condition' (and 'feature_signature'
                                 when distinct_features)],
                     'search': SearchResult of the 'none' condition}}.
    A model whose search failed maps to None, and the failure is printed.
    """
    from experiments.decided_grid import decided_models
    tier = 'general_distance' if distance else 'general'
    if models is None:
        models = [m for m in decided_models(dataset_name, tier) if m != 'knn']
    searches = fold_searches(
        reps, labels, groups, train_idx=train_idx, fold_i=fold_i,
        n_outer_folds=n_outer_folds, dataset_name=dataset_name,
        augmentations=augmentations, task_type=task_type, metric=metric,
        domain=domain, tier=tier, seed=seed, verbose=verbose,
        label_noise=label_noise, objective=objective,
        adaptive_pruning_schedule=adaptive_pruning_schedule,
        model_pruning_aggressiveness=model_pruning_aggressiveness,
        cache_namespace=cache_namespace,
        condition_suffix=condition_suffix, adaptive_pruning=adaptive_pruning,
        train_labels=train_labels, noise_groups=noise_groups)
    base_search = searches.get('none')
    if base_search is None:
        fails = getattr(searches, 'failures', None) or []
        print(f"  fold {fold_i}: {dataset_name} {tier} search failed: "
              f"{[f.get('error') or f.get('error_type') for f in fails]}")
        return {m: None for m in models}

    out = {}
    for m in models:
        ctx = f'{dataset_name} {tier} fold {fold_i} model {m}'
        # require_full=False: a model short of n_base scored representations
        # reports its own shortfall_reason in base_detail instead of raising
        # and costing the other model its slots.
        base = top_base_reps(base_search, m, n=n_base, context=ctx,
                             require_full=False)
        pooled = []
        for cond, srch in searches.items():
            if srch is None:
                continue
            # With the identical-features check on, draw spare candidates so
            # a skipped duplicate can be replaced by the next distinct one.
            th = top_hybrids(srch, m,
                             n=n_hybrids + (4 if distinct_features else 0),
                             budget=budget, context=ctx, require_full=False)
            for e in th['selected']:
                pooled.append(dict(e, aug_condition=cond))
        lower = base_search.lower_is_better
        pooled.sort(key=lambda e: (e['score_mean'] if lower else -e['score_mean'],
                                   e['label']))
        skipped = []
        if distinct_features and len(pooled) > 1:
            chosen, skipped = _distinct_feature_hybrids(
                pooled, n_hybrids, reps=reps, labels=labels, groups=groups,
                train_idx=train_idx, augmentations=augmentations, model=m,
                dataset_name=dataset_name, task_type=task_type, metric=metric,
                domain=domain, lower_is_better=lower)
        else:
            chosen = pooled[:n_hybrids]
        out[m] = {
            'base': [e['rep'] for e in base['selected']],
            'base_detail': base,
            'hybrids': chosen,
            'hybrids_skipped_identical_features': skipped,
            'search': base_search,
            'selection_scope': 'fold_train_rows',
        }
    return out


class _SimilaritySkipped(Exception):
    """The similarity search is not defined on this dataset (skip list)."""


def fold_neighbour_space_recipes(reps, labels, groups, *, train_idx, fold_i,
                                 n_outer_folds, dataset_name,
                                 augmentations=None, task_type='auto',
                                 metric='auto', domain=None, seed=42,
                                 label_noise=None, verbose=False,
                                 condition_suffix=None, train_labels=None,
                                 noise_groups=None):
    """Per outer fold: each decided model's 2 base representations, 2 general
    hybrids, and the recipes for the hybrid neighbour spaces.

    The per-fold replacement for noise_helpers.pool_recipe_per_downstream_model
    + pool_recipes_by_family, which read the whole-dataset stored search (R2)
    and handed out 2 general + 1 distance hybrids + 2 base representations =
    10 configurations for two models. Here the configurations a model is
    FITTED on are 2 base + 2 general hybrids, so 8 for two models (R5), and
    the distance and similarity hybrids are neighbour spaces only (handoff 2.3
    item 1: "both kinds" where an experiment measures something with
    neighbours).

    Three per-fold searches, each on this fold's training rows and the
    dataset's decided grid, each memoised by fold_searches:
      general                 -> 'hybrid', 'hybrid__general2' (distinct
                                 selected features, fold_standard_selection)
      general_distance        -> 'hybrid__distance1': this model's top hybrid
                                 of the distance search
      general, objective=
        'similarity'          -> 'hybrid_similarity': the top hybrid of the
                                 similarity search, ranked over every model it
                                 scored (a similarity search fits no model)

    Returns {model: {'base': [2 names], 'hybrids': [2 top_hybrids entries],
                     'fit_configurations': base + hybrids labels,
                     'recipes': {space: (Blueprint, aug_condition)} in the
                                form n07.fold_similarity_spaces reads,
                     'unavailable': {space: why}}}.
    A model whose general search failed maps to None.

    condition_suffix: files the three searches apart from the dataset's
    standard per-fold searches. Pass one whenever the rows are not the
    dataset's own (an experiment that keeps one row per molecule of
    herg_ic50, a subsample) or the labels are not its clean labels, so two
    different row sets never share one stored search under one fold name.
    """
    from experiments.decided_grid import decided_models, is_decided
    sel = fold_standard_selection(
        reps, labels, groups, train_idx=train_idx, fold_i=fold_i,
        n_outer_folds=n_outer_folds, dataset_name=dataset_name,
        augmentations=augmentations, task_type=task_type, metric=metric,
        domain=domain, seed=seed, label_noise=label_noise, verbose=verbose,
        condition_suffix=condition_suffix, train_labels=train_labels,
        noise_groups=noise_groups)
    common = dict(train_idx=train_idx, fold_i=fold_i,
                  n_outer_folds=n_outer_folds, dataset_name=dataset_name,
                  augmentations=augmentations, task_type=task_type,
                  metric=metric, domain=domain, seed=seed,
                  label_noise=label_noise, verbose=verbose,
                  condition_suffix=condition_suffix, train_labels=train_labels,
                  noise_groups=noise_groups)

    dist_search, dist_why, dist_searches = None, None, {}
    if is_decided(dataset_name, 'general_distance'):
        try:
            ds = fold_searches(reps, labels, groups, tier='general_distance',
                               **common)
            # every condition the decided grid runs ('none', plus 'uncapped'
            # on ginkgo_colloidal and ginkgo_hic)
            dist_searches = {c: v for c, v in ds.items() if v is not None}
            dist_search = dist_searches.get('none')
            if dist_search is None:
                dist_why = 'the per-fold distance search failed'
        except Exception as exc:
            dist_why = (f'the per-fold distance search raised '
                        f'{type(exc).__name__}: {exc}')
    else:
        dist_why = 'DECISIONS.md decides no distance search for this dataset'
    dist_models = (set(decided_models(dataset_name, 'general_distance'))
                   if dist_search is not None else set())

    sim_entry, sim_why = None, None
    try:
        from experiments.dataset_policy import similarity_objective_undefined
        _undef = similarity_objective_undefined(
            dataset_name, labels, None if task_type == 'auto' else task_type)
    except Exception:
        _undef = None
    try:
        if _undef:
            # The skip list (dataset_policy.SIMILARITY_UNDEFINED_OTHER and the
            # multiclass sets): the search could not score, so it is not run.
            raise _SimilaritySkipped(_undef)
        ss_ = fold_searches(reps, labels, groups, tier='general',
                            objective='similarity', **common)
        sim_search = ss_.get('none')
        if sim_search is None:
            sim_why = 'the per-fold similarity search failed'
        else:
            th = top_hybrids_across_models(
                sim_search, n=1, require_full=False,
                context=f'{dataset_name} similarity fold {fold_i}')
            if th['selected']:
                sim_entry = dict(th['selected'][0], aug_condition='none')
            else:
                sim_why = 'the per-fold similarity search scored no hybrid'
    except _SimilaritySkipped as exc:
        sim_why = f'not run on this dataset: {exc}'
    except Exception as exc:
        sim_why = (f'the per-fold similarity search raised '
                   f'{type(exc).__name__}: {exc}')

    out = {}
    for m, s_m in sel.items():
        if s_m is None:
            out[m] = None
            continue
        recipes, unavailable = {}, {}
        for space, i in (('hybrid', 0), ('hybrid__general2', 1)):
            if len(s_m['hybrids']) > i:
                h = s_m['hybrids'][i]
                recipes[space] = (h['blueprint'], h.get('aug_condition', 'none'))
            else:
                unavailable[space] = (f'the per-fold general search gave this '
                                      f'model fewer than {i + 1} hybrids with '
                                      f'distinct features')
        if dist_search is None:
            unavailable['hybrid__distance1'] = dist_why
        elif m not in dist_models:
            unavailable['hybrid__distance1'] = (
                f'{m} is not one of the distance search\'s decided models '
                f'{sorted(dist_models)}')
        else:
            cands = []
            for cond, srch in dist_searches.items():
                th = top_hybrids(srch, m, n=1, require_full=False,
                                 context=f'{dataset_name} distance fold '
                                         f'{fold_i} {m} {cond}')
                cands += [dict(e, aug_condition=cond) for e in th['selected']]
            lower_d = dist_search.lower_is_better
            cands.sort(key=lambda e: (e['score_mean'] if lower_d
                                      else -e['score_mean'], e['label']))
            if cands:
                recipes['hybrid__distance1'] = (cands[0]['blueprint'],
                                                cands[0]['aug_condition'])
            else:
                unavailable['hybrid__distance1'] = (
                    'the per-fold distance search scored no hybrid for ' + m)
        if sim_entry is not None:
            recipes['hybrid_similarity'] = (sim_entry['blueprint'], 'none')
        else:
            unavailable['hybrid_similarity'] = sim_why
        out[m] = {
            'base': s_m['base'],
            'base_detail': s_m.get('base_detail'),
            'hybrids': s_m['hybrids'],
            'hybrids_skipped_identical_features':
                s_m.get('hybrids_skipped_identical_features', []),
            'fit_configurations': (list(s_m['base'])
                                   + [h['label'] for h in s_m['hybrids']]),
            'recipes': recipes,
            'unavailable': unavailable,
            'selection_scope': 'fold_train_rows',
        }
    return out


def hybrid_feature_signature(feat_info):
    """The selected features of one built hybrid, as a hashable value:
    ((representation or donor, (feature numbers, sorted)), ...). Read from the
    feat_info build_fold_hybrid returns; every entry that carries
    'selected_indices' counts, at any depth one level down (augmentation
    donors sit under their own key)."""
    out = []

    def _walk(prefix, d):
        for k, v in sorted(d.items(), key=lambda kv: str(kv[0])):
            if not isinstance(v, dict):
                continue
            if 'selected_indices' in v:
                idx = np.asarray(v['selected_indices']).ravel()
                out.append((f'{prefix}{k}', tuple(sorted(int(i) for i in idx))))
            elif not prefix:
                _walk(f'{k}/', v)
    _walk('', feat_info or {})
    return tuple(out)


def _distinct_feature_hybrids(pooled, n_hybrids, *, reps, labels, groups,
                              train_idx, augmentations, model, dataset_name,
                              task_type, metric, domain, lower_is_better):
    """Walk the ranked candidates, building each on the fold's training rows,
    and keep the first n_hybrids whose selected features differ. A candidate
    whose build raises is kept with feature_signature None (it cannot be shown
    to duplicate anything) and the error is recorded on it."""
    from kirby.find_best_hybrid import BUILTIN_METRICS
    tr = np.asarray(train_idx)
    y = np.asarray(labels)
    metric_name, task = _resolve_search_metric(
        y, task_type=task_type, metric=metric, domain=domain,
        dataset_name=dataset_name)
    metric_fn = BUILTIN_METRICS.get(metric_name)
    if isinstance(metric_fn, tuple):
        metric_fn = metric_fn[0]
    factory = get_model_factories([model], task_type=task,
                                  n_samples=len(tr)).get(model)
    train_reps = {k: np.asarray(v)[tr] for k, v in reps.items()}
    probe = {k: v[:1] for k, v in train_reps.items()}
    g_tr = None if groups is None else np.asarray(groups)[tr]
    chosen, skipped, seen = [], [], set()
    for e in pooled:
        if len(chosen) >= n_hybrids:
            break
        bp = e.get('blueprint')
        augs = (augmentations if (e.get('aug_condition') or 'none') != 'none'
                else None)
        sig, err = None, None
        try:
            _, _, fi = build_fold_hybrid(
                train_reps, probe, y[tr], bp,
                augmentations_train=(None if augs is None else
                                     {a: np.asarray(X)[tr] for a, X in augs.items()}),
                augmentations_val=(None if augs is None else
                                   {a: np.asarray(X)[tr[:1]] for a, X in augs.items()}),
                groups=g_tr,
                # The same budget-split scores every rebuild of this recipe is
                # given (its evaluator's, _attach_evaluator_rep_scores), so the
                # check compares the hybrids that will actually be scored.
                precomputed_alloc_scores=getattr(bp, 'rep_scores', None),
                **fold_hybrid_scoring_kwargs(
                    bp, model=(factory() if factory else None),
                    metric_fn=metric_fn, lower_is_better=lower_is_better))
            sig = hybrid_feature_signature(fi)
        except Exception as exc:
            err = f'{type(exc).__name__}: {exc}'
        if sig is not None and sig in seen:
            skipped.append({'label': e.get('label'),
                            'aug_condition': e.get('aug_condition'),
                            'score_mean': e.get('score_mean'),
                            'same_features_as': next(
                                c['label'] for c in chosen
                                if c.get('feature_signature') == sig)})
            print(f"  {dataset_name} {model}: hybrid {e.get('label')} selects "
                  f"the same features as one already chosen; taking the next")
            continue
        if sig is not None:
            seen.add(sig)
        chosen.append(dict(e, feature_signature=sig, signature_error=err))
    return chosen, skipped


# =============================================================================
# HONEST (leave-one-fold-out) RE-SCORING
# Restored 2026-08-24: dropped from _shared as collateral in e921e22 (a
# 'remove ridge' commit); recovered verbatim from d471d37, the last commit
# that had it (paired per-fold deltas + hard metric-direction guards).
# ~30 experiments import fold_reselection / build_fold_reselection_input here.
# =============================================================================


def _hr_is_bad(x) -> bool:
    """True if x is None or NaN (or non-numeric). Used to skip bad fold scores."""
    if x is None:
        return True
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return True
    return xf != xf  # NaN != NaN


def _hr_lofo_score(items, lower_is_better, exclude_folds=()):
    """Leave-one-fold-out honest score for one arm (pure arithmetic).

    Args:
        items: list of (name, fold_scores_list). fold_scores may be ragged,
            contain NaN/None, or be empty.
        lower_is_better: metric direction.
        exclude_folds: fold indices that contribute NO held-out term. They are
            still read when averaging the OTHER folds to pick each held-out
            fold's winner — the exclusion is only of the score reported on
            them. Used for fold 0 of a search that filtered its own candidate
            list on fold 0's scores.

    Returns:
        (reselected_score, winners_per_fold, n_folds_used, single_fold_flag,
         per_fold)

        reselected_score: mean over held-out folds of the LOFO-selected item's
            held-out score; NaN if the arm is empty or nothing scorable.
        winners_per_fold: name selected for each held-out fold (skips folds
            where no item has usable other-fold data, so may be shorter than F).
        n_folds_used: number of held-out folds that contributed a finite score.
        single_fold_flag: True when only one fold is present, so no held-out
            selection is possible and the plain best single-fold score is
            returned instead (the caller should treat it as optimistic).
        per_fold: length-F list, fold-INDEX-aligned, of this arm's held-out
            score on each fold (None where the arm produced none). Aligned so
            the caller can pair hybrid against baseline fold-by-fold; the
            compacted `winners_per_fold` above cannot be used for that.
    """
    # Keep only items with at least one fold score present.
    clean = [(n, [s for s in fs]) for n, fs in items
             if fs is not None and len(fs) > 0]
    if not clean:
        return float('nan'), [], 0, False, []
    excluded = {int(f) for f in (exclude_folds or ())}

    F = max(len(fs) for _, fs in clean)
    # Strictly-better comparison so ties keep the first-seen item (stable).
    if lower_is_better:
        strictly_better = lambda a, b: a < b
    else:
        strictly_better = lambda a, b: a > b

    if F == 1:
        # Only one fold: no held-out fold available. Fall back to the plain
        # best single-fold score and flag it as not-honest.
        if 0 in excluded:
            # The one fold there is, is the fold the candidate list was
            # filtered on. Nothing honest can be read off it.
            return float('nan'), [], 0, True, [None]
        best_name, best_val = None, None
        for n, fs in clean:
            v = fs[0]
            if _hr_is_bad(v):
                continue
            v = float(v)
            if best_val is None or strictly_better(v, best_val):
                best_name, best_val = n, v
        if best_val is None:
            return float('nan'), [], 0, True, [None]
        return best_val, [best_name], 1, True, [best_val]

    held_out = []
    winners = []
    per_fold = [None] * F
    for f in range(F):
        if f in excluded:
            continue
        best_name, best_mean, best_heldout = None, None, None
        for n, fs in clean:
            others = [float(fs[k]) for k in range(len(fs))
                      if k != f and k < len(fs) and not _hr_is_bad(fs[k])]
            if not others:
                continue
            m = sum(others) / len(others)
            if best_mean is None or strictly_better(m, best_mean):
                best_name, best_mean = n, m
                # This item's held-out score on fold f (may be missing/NaN).
                best_heldout = (float(fs[f])
                                if f < len(fs) and not _hr_is_bad(fs[f])
                                else None)
        if best_name is None:
            continue  # no item had usable other-fold data for this fold
        winners.append(best_name)
        if best_heldout is not None:
            held_out.append(best_heldout)
            per_fold[f] = best_heldout

    if not held_out:
        return float('nan'), winners, 0, False, per_fold
    return (sum(held_out) / len(held_out), winners, len(held_out), False,
            per_fold)


def hr_exclude_folds(block):
    """Fold indices a stored fold_reselection block says to drop from both arms.

    A search that filters its own candidate list on fold 0 — every default
    schedule with more than three allocation methods does, and the cluster
    submitters run a schedule that also drops half of each allocation/importance
    pair at three checkpoints — leaves fold 0's hybrid score picked from the
    recipes fold 0 kept. The single representations pass through no such filter.
    build_fold_reselection_input writes [0] into the block in that case.

    A block written before this key existed carries neither 'exclude_folds' nor
    'fold0_pruned'. That reads as [0], not []: the searches those blocks came
    from used the default five allocation methods against a strategy_top_n of 3,
    which prunes. Pass exclude_folds=[] explicitly to override.
    """
    if not isinstance(block, dict):
        return [0]
    if 'exclude_folds' in block:
        return [int(f) for f in (block.get('exclude_folds') or [])]
    if block.get('fold0_pruned') is False:
        return []
    return [0]


def hr_direction(block):
    """Resolve lower_is_better from a stored fold_reselection block — no silent default.

    The historic sign bug came from readers doing bool(block.get('lower_is_better')),
    which turns a MISSING key into False = "higher is better" and silently inverts
    every RMSE/MAE conclusion. Direction is never guessed here:

      - if the block carries 'metric_name' and it is a BUILTIN_METRIC, that table
        is authoritative (and we raise if the stored flag contradicts it);
      - else the stored flag is used, but only if actually present;
      - else raise. A block with no direction is unscoreable, not higher-is-better.
    """
    from kirby.find_best_hybrid import BUILTIN_METRICS

    if not isinstance(block, dict):
        raise TypeError(f"fold_reselection block must be a dict, got {type(block).__name__}")

    metric = block.get('metric_name')
    stored = block.get('lower_is_better')

    if metric in BUILTIN_METRICS:
        canonical = BUILTIN_METRICS[metric][1]
        if stored is not None and bool(stored) != canonical:
            raise ValueError(
                f"metric direction contradiction: metric_name={metric!r} is "
                f"lower_is_better={canonical} per BUILTIN_METRICS, but the block "
                f"stores lower_is_better={bool(stored)}. Refusing to guess — one "
                f"of the two is wrong and the delta sign depends on it.")
        return canonical

    if stored is None:
        raise ValueError(
            "fold_reselection block has neither 'metric_name' nor 'lower_is_better'. "
            "Refusing to default to higher-is-better: that silently inverts the "
            "delta sign on rmse/mae. Persist the direction.")
    return bool(stored)


def object_direction(obj, what='object'):
    """Direction of the scores `obj` carries — metric name first, then the flag.

    Same contract as hr_direction, for a SearchResult or a Blueprint instead of a
    stored fold-reselection block: the metric NAME is authoritative when it is one
    of BUILTIN_METRICS, the stored flag is used only when it is not, and a thing
    carrying neither raises rather than defaulting to higher-is-better. A default
    of False is not a direction — it is a guess that happens to be right on 46 of
    the 70 registry entries, and it is the shape of the historic sign bug.
    Added 2026-09-10 (metric-direction audit).
    """
    from kirby.find_best_hybrid import BUILTIN_METRICS

    name = getattr(obj, 'metric_name', None) or getattr(obj, 'metric', None)
    if name in BUILTIN_METRICS:
        return BUILTIN_METRICS[name][1]
    stored = getattr(obj, 'lower_is_better', None)
    if stored is None:
        raise ValueError(
            f"{what} carries neither a known metric name nor 'lower_is_better', "
            f"so the direction of its scores is unknown. Refusing to default to "
            f"higher-is-better: that silently inverts every rmse/mae comparison.")
    return bool(stored)


def fold_reselection(candidates_by_model, baselines_by_model, lower_is_better,
                   metric_name=None, exclude_folds=None,
                   recipe_selection=None):
    """Leave-one-fold-out per-fold reselection of hybrid vs best-single-rep.

    Pure arithmetic on per-fold scores already computed — NO model fitting, NO
    I/O. Applies the identical held-out-fold tax to BOTH arms.

    Args:
        candidates_by_model: {model: [ {"name": str, "fold_scores": [..]}, .. ]}
            The hybrid CANDIDATE recipes for each model (all recipes searched,
            NOT just the same-fold winner — the winner is re-selected per
            held-out fold). Only include models actually evaluated.
        baselines_by_model: {model: {rep_name: [fold_scores]}}. The baseline
            single reps for each model.
        lower_is_better: metric direction (True for rmse/mae/etc.).
        exclude_folds: fold indices that contribute no term to either arm.
            Pass [0] for a search that pruned its candidate list on fold 0:
            the hybrid arm's fold-0 score is then measured over the recipes
            fold 0 helped keep, while the single representations were never
            filtered, so the two arms would not be paying the same tax.
            build_fold_reselection_input writes this into the persisted block as
            'exclude_folds'; read it back with hr_exclude_folds(block) and pass
            it here. Both arms drop the same folds, so the comparison stays
            paired.
        recipe_selection: what rows chose the hybrid recipe, copied onto every
            model entry as 'recipe_selection'. Pass 'whole_dataset' (the
            'selection_scope' get_best_search_for_downstream returns) whenever
            the candidates rebuild a recipe the search picked over every row:
            the per-fold rebuild fixes the feature weights, not the choice of
            allocation method, budget, importance method or evaluator, so the
            delta is not fully held out and a reader must be told. None when
            the candidates were not chosen on any of these rows.

    Returns:
        {model: {
            'hybrid_reselected':          float (NaN if no candidates),
            'best_single_reselected':     float (NaN if no baselines),
            'delta':                  float, difference of the two means above,
                                      direction-corrected so + means the hybrid
                                      is better (NaN if either side NaN),

            # --- the paired per-fold view: report these ---
            'hybrid_fold_scores':     [held-out score per fold, fold-aligned,
                                      None where absent],
            'baseline_fold_scores':   [same, for the best single rep],
            'per_fold_delta':         [hybrid - baseline on each fold,
                                      direction-corrected, None where either
                                      arm is missing that fold],
            'delta_mean_of_folds':    float, mean of per_fold_delta over folds
                                      where BOTH arms scored. This is the
                                      paired number to report; it equals
                                      'delta' only when both arms cover the
                                      same folds.
            'n_paired_folds':         int (folds where both arms scored),

            'hybrid_winners_per_fold':   [recipe name per held-out fold],
            'baseline_winners_per_fold': [rep name per held-out fold],
            'n_folds_used':           int (held-out folds that contributed),
            'excluded_folds':         [fold indices dropped from both arms],
            'recipe_selection':       str|None, what rows chose the recipe,
            'single_fold':            bool (True => n_folds==1, scores are the
                                      plain best, NOT honest — no held-out tax),
        }}
        One entry per model appearing in EITHER input dict. Empty candidate or
        baseline sets yield NaN for that arm (and NaN delta), never an error.
    """
    # Direction guard. When the caller names the metric, BUILTIN_METRICS is
    # authoritative and a contradicting flag is a hard error — the sign of every
    # delta below depends on getting this right, and it has been wrong before.
    if metric_name is not None:
        from kirby.find_best_hybrid import BUILTIN_METRICS
        if metric_name in BUILTIN_METRICS:
            canonical = BUILTIN_METRICS[metric_name][1]
            if bool(lower_is_better) != canonical:
                raise ValueError(
                    f"metric direction contradiction: metric_name={metric_name!r} "
                    f"is lower_is_better={canonical} per BUILTIN_METRICS, but "
                    f"lower_is_better={bool(lower_is_better)} was passed.")
            lower_is_better = canonical

    # Both arms drop the same folds. Dropping them from one side only would
    # compare a four-fold mean against a five-fold mean.
    _excluded = sorted({int(f) for f in (exclude_folds or ())})

    models = set(candidates_by_model or {}) | set(baselines_by_model or {})
    out = {}
    for model in sorted(models):
        cand_list = (candidates_by_model or {}).get(model) or []
        cand_items = [
            (c.get('name', f'cand{i}'), c.get('fold_scores'))
            for i, c in enumerate(cand_list)
        ]
        base_items = list(((baselines_by_model or {}).get(model) or {}).items())

        h_score, h_win, h_nf, h_single, h_folds = _hr_lofo_score(
            cand_items, lower_is_better, exclude_folds=_excluded)
        b_score, b_win, b_nf, b_single, b_folds = _hr_lofo_score(
            base_items, lower_is_better, exclude_folds=_excluded)

        if _hr_is_bad(h_score) or _hr_is_bad(b_score):
            delta = float('nan')
        elif lower_is_better:
            delta = b_score - h_score  # lower hybrid == positive delta
        else:
            delta = h_score - b_score  # higher hybrid == positive delta

        # PAIRED per-fold comparison: on each held-out fold, the hybrid picked
        # on the other folds vs the single rep picked on the other folds. Both
        # arms paid the identical held-out tax, so the difference is meaningful
        # fold by fold. `delta` above is a difference of two MEANS, which only
        # equals the mean of these differences when both arms scored the same
        # fold set — it silently diverges when either arm drops a fold.
        # Report `delta_mean_of_folds`; it is the paired quantity.
        n_f = max(len(h_folds), len(b_folds))
        per_fold_delta = [None] * n_f
        paired = []
        for f in range(n_f):
            hv = h_folds[f] if f < len(h_folds) else None
            bv = b_folds[f] if f < len(b_folds) else None
            if hv is None or bv is None:
                continue
            d = (bv - hv) if lower_is_better else (hv - bv)
            per_fold_delta[f] = d
            paired.append(d)

        out[model] = {
            'hybrid_reselected': h_score,
            'best_single_reselected': b_score,
            'delta': delta,
            # Fold-index-aligned; None = that arm produced no held-out score.
            'hybrid_fold_scores': h_folds,
            'baseline_fold_scores': b_folds,
            'per_fold_delta': per_fold_delta,
            'delta_mean_of_folds': (sum(paired) / len(paired) if paired
                                    else float('nan')),
            'n_paired_folds': len(paired),
            'hybrid_winners_per_fold': h_win,
            'baseline_winners_per_fold': b_win,
            'n_folds_used': max(h_nf, b_nf),
            'single_fold': bool(h_single or b_single),
            'excluded_folds': list(_excluded),
            'recipe_selection': recipe_selection,
        }
    return out


def build_fold_reselection_input(search, *, skip_secondary=True):
    """Build the standard fold_reselection JSON block from a SearchResult.

    Groups hybrid candidates from search.all_results by model (each carries its
    own per-fold 'fold_scores'), and reads per-fold baseline scores from
    search.baseline_fold_scores (additive field; empty on old cache files).
    Only models actually present are included — the model set is never assumed.

    Returns a dict with the exact keys fold_reselection() expects, ready to
    json.dump under the top-level "fold_reselection" key:

        {"metric_name": str,
         "lower_is_better": bool,
         "fold0_pruned": bool | None,
         "exclude_folds": [int, ...],
         "candidates_by_model": {model: [{"name", "fold_scores"}, ...]},
         "baselines_by_model":  {model: {rep: [fold_scores]}}}

    'exclude_folds' is [0] whenever the search filtered its candidate list using
    fold 0's scores, and [] when it provably did not. Fold 0's hybrid score is
    otherwise read off recipes fold 0 helped keep, while the single
    representations were never filtered — the two arms would not be paying the
    same held-out tax. A SearchResult unpickled from a file written before
    fold0_pruned existed reports None, and None is treated as pruned: the
    default schedule with five allocation methods and strategy_top_n=3 prunes.

    metric_name is persisted alongside the flag so a reader can cross-check the
    direction against BUILTIN_METRICS (see hr_direction) instead of trusting a
    bare bool that may be missing from an old file — a missing flag reads as
    False = higher-is-better and silently inverts every rmse/mae delta.
    """
    metric = getattr(search, 'metric_name', None)
    # Direction is READ from the search, never defaulted. If the search carries a
    # known metric, BUILTIN_METRICS wins and a contradiction is a hard error.
    lower = getattr(search, 'lower_is_better', None)
    from kirby.find_best_hybrid import BUILTIN_METRICS
    if metric in BUILTIN_METRICS:
        canonical = BUILTIN_METRICS[metric][1]
        if lower is not None and bool(lower) != canonical:
            raise ValueError(
                f"SearchResult direction contradiction: metric_name={metric!r} is "
                f"lower_is_better={canonical} per BUILTIN_METRICS, but the search "
                f"carries lower_is_better={bool(lower)}.")
        lower = canonical
    elif lower is None:
        raise ValueError(
            "SearchResult carries neither a known metric_name nor lower_is_better; "
            "refusing to default to higher-is-better (inverts rmse/mae deltas).")
    lower = bool(lower)

    candidates_by_model: Dict[str, list] = {}
    for r in (getattr(search, 'all_results', None) or []):
        model = r.get('model')
        folds = r.get('fold_scores')
        if model is None or not folds:
            continue
        # Stable recipe id from the swept config (winner label only).
        name = "{}|{}|b{}|{}|s{}".format(
            r.get('allocation_method'), r.get('importance_method'),
            r.get('budget'), r.get('internal_evaluator'), r.get('step_size'))
        candidates_by_model.setdefault(model, []).append(
            {'name': name, 'fold_scores': list(folds)})

    # search.baseline_fold_scores is {rep: {model: [folds]}} — invert to
    # {model: {rep: [folds]}}.
    baselines_by_model: Dict[str, dict] = {}
    bfs = getattr(search, 'baseline_fold_scores', None) or {}
    for rep_name, per_model in bfs.items():
        if rep_name.startswith('_'):
            continue  # skip reserved keys, mirrors extract_top3_base_reps
        for model, folds in (per_model or {}).items():
            if not folds:
                continue
            baselines_by_model.setdefault(model, {})[rep_name] = list(folds)

    # None when the SearchResult predates the field; hr_exclude_folds reads
    # None as pruned, because the default schedule prunes.
    _f0 = getattr(search, 'fold0_pruned', None)
    return {
        'metric_name': metric,
        'lower_is_better': lower,
        'fold0_pruned': (None if _f0 is None else bool(_f0)),
        'exclude_folds': ([] if _f0 is False else [0]),
        'candidates_by_model': candidates_by_model,
        'baselines_by_model': baselines_by_model,
    }


# =============================================================================
# REP TIER CLASSIFICATION
# =============================================================================

# Per-rep tier overrides (exceptions to pure dimensionality classification)
REP_TIER_OVERRIDES = {
    'aac': 2,  # 40D but bumped to tier 2 per user decision
}


def classify_rep_tier(dim: int, rep_name: str = None) -> int:
    """Classify a representation into a tier based on dimensionality.

    Tier 1 (<50D):   augmentation-only, NEVER standalone baselines
    Tier 2 (50-200D): both standalone baselines and augmentation candidates
    Tier 3 (>200D):  base reps, standalone baselines only

    Per-rep exceptions are handled via REP_TIER_OVERRIDES.
    Matches both exact names and domain-stripped suffixes (e.g. protein_aac → aac).
    """
    if rep_name:
        if rep_name in REP_TIER_OVERRIDES:
            return REP_TIER_OVERRIDES[rep_name]
        # Strip domain prefix (drug_, protein_) for DTA reps
        for prefix in ('drug_', 'protein_'):
            if rep_name.startswith(prefix):
                suffix = rep_name[len(prefix):]
                if suffix in REP_TIER_OVERRIDES:
                    return REP_TIER_OVERRIDES[suffix]
                break
    if dim < 50:
        return 1
    elif dim <= 200:
        return 2
    else:
        return 3


# =============================================================================
# CV SPLITTING
# =============================================================================

def _should_stratify(task_type, labels):
    """Decide whether a k-fold split should be class-stratified.

    Two-tier rule (ISS-84):
      1. If `task_type` is given, it is authoritative — 'classification'
         stratifies, 'regression' never does. This is the correct signal:
         the loader / registry already knows the task, and the label *dtype*
         is a poor proxy (a regression target can be integer-typed, e.g.
         counts/ordinals, and a classification target can be float-typed,
         e.g. 0.0/1.0). Passing task_type removes both failure modes.
      2. If `task_type` is None (the many existing positional callers that
         predate this argument), fall back to the historical integer-dtype
         heuristic so their behaviour is unchanged.
    """
    if task_type == 'classification':
        return True
    if task_type == 'regression':
        return False
    # Heuristic fallback (task_type not supplied by caller).
    return labels is not None and np.issubdtype(
        np.asarray(labels).dtype, np.integer)


def cv_split(groups, n_splits, X, labels, seed=42, mode=None, *,
             task_type=None):
    """Yield (train_idx, val_idx) using a group-aware CV strategy.

    - n_splits=1: single 80/20 GroupShuffleSplit (scaffold/site-grouped).
    - n_splits>1: StratifiedGroupKFold for classification (class-balanced
      *and* group-respecting), GroupKFold otherwise.

    `groups` is REQUIRED and must have at least `max(2, n_splits)` distinct
    values. There is NO random-split fallback: per project rule
    (feedback_no_random_splits.md) random KFold/ShuffleSplit is forbidden for
    any production result, so an absent or under-resolved `groups` is a loud
    error rather than a silent degradation to random splitting. If a caller
    genuinely has no grouping it must be fixed upstream (assign scaffold/site
    groups in the loader) — not papered over here.

    `task_type` (keyword-only, ISS-84) selects the k-fold strategy
    authoritatively: 'classification' -> StratifiedGroupKFold (classes stay
    balanced across folds while groups are preserved — required for TCGA
    classification e.g. tcga_brca_pam50 / tcga_luad_vs_lusc and any other
    grouped classification task), 'regression' -> plain GroupKFold. When it is
    left None (existing positional callers), the strategy falls back to the
    historical integer-label heuristic, so no existing caller changes
    behaviour. Pass task_type explicitly for classification datasets whose
    labels are not integer-typed, which the heuristic would otherwise miss.
    StratifiedGroupKFold itself remains group-respecting (no sample's group
    spans train+test) — stratification only rebalances *which* groups go to
    each fold.

    The historical `mode` argument is accepted only for backward-compatibility
    with existing callers and is ignored: the split strategy is fully
    determined by `n_splits` (it always was — `mode` never altered behaviour).
    Passing it now emits a DeprecationWarning so callers can drop it.
    """
    if mode is not None:
        warnings.warn(
            "cv_split(mode=...) is vestigial and ignored; the split strategy "
            "is determined entirely by n_splits. Drop the `mode` argument.",
            DeprecationWarning, stacklevel=2,
        )

    n_groups_seen = 0 if groups is None else len(np.unique(groups))
    required = max(2, n_splits)
    if groups is None or n_groups_seen < required:
        raise ValueError(
            f"cv_split: group-aware splitting requires groups with >= "
            f"{required} distinct values for n_splits={n_splits}, but got "
            f"{'groups=None' if groups is None else f'{n_groups_seen} distinct group(s)'}. "
            "Random splitting is forbidden (feedback_no_random_splits.md) — "
            "fix the caller to supply scaffold/site/identity groups instead of "
            "falling back to a random split."
        )

    if n_splits == 1:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        yield from gss.split(X, labels, groups=groups)
        return

    # Use StratifiedGroupKFold for classification (preserves class balance
    # within grouped folds). Falls back to GroupKFold if unavailable.
    if _should_stratify(task_type, labels):
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            yield from StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            ).split(X, labels, groups=groups)
            return
        except ImportError:
            pass
    yield from GroupKFold(n_splits=n_splits).split(X, labels, groups=groups)


def _cv_split(groups, n_folds, X, labels, seed=42, *, task_type=None):
    """Legacy wrapper — delegates to cv_split (group-aware, no random fallback).

    Forwards the keyword-only `task_type` (ISS-84) so callers using the
    `_cv_split` alias can also request classification stratification.
    """
    yield from cv_split(groups, n_folds, X, labels, seed=seed,
                         task_type=task_type)


def make_cv_splitter(n_folds, groups=None, seed=42, *, task_type=None):
    """Return (splitter, use_groups) for direct sklearn CV loops.

    `groups` is REQUIRED and must have >= max(2, n_folds) distinct values.
    There is NO random-split fallback (feedback_no_random_splits.md): an
    absent or under-resolved `groups` raises rather than silently degrading
    to a random KFold/ShuffleSplit. `use_groups` is always True on success;
    callers must pass `groups` into `splitter.split(X, y, groups=groups)`.

    `task_type='classification'` (keyword-only, ISS-84) returns a
    StratifiedGroupKFold so classes stay balanced across grouped folds; this
    is compatible with the existing `splitter.split(X, y, groups=groups)`
    call convention (StratifiedGroupKFold also consumes `y`). Anything else
    (including the default None used by all current callers, which are
    regression splits) returns plain GroupKFold — unchanged behaviour.
    """
    n_groups_seen = 0 if groups is None else len(np.unique(groups))
    required = max(2, n_folds)
    if groups is None or n_groups_seen < required:
        raise ValueError(
            f"make_cv_splitter: group-aware splitting requires groups with "
            f">= {required} distinct values for n_folds={n_folds}, but got "
            f"{'groups=None' if groups is None else f'{n_groups_seen} distinct group(s)'}. "
            "Random splitting is forbidden (feedback_no_random_splits.md) — "
            "fix the caller to supply scaffold/site/identity groups."
        )
    if n_folds == 1:
        return GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed), True
    if task_type == 'classification':
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            return StratifiedGroupKFold(
                n_splits=n_folds, shuffle=True, random_state=seed), True
        except ImportError:
            pass
    return GroupKFold(n_splits=n_folds), True


# =============================================================================
# MOLECULAR GROUPING ALTERNATIVES (Butina clustering — ISS-85)
# =============================================================================

def butina_clusters(smiles_list, cutoff: float = 0.6,
                    radius: int = 2, n_bits: int = 2048):
    """Group molecules by RDKit Butina (Taylor-Butina) Tanimoto clustering.

    A lab-standard, group-aware alternative/complement to Murcko-scaffold
    splitting for the molecular domain (ISS-85). Two molecules with Tanimoto
    similarity >= `cutoff` (on Morgan/ECFP fingerprints) tend to land in the
    same cluster, so using the returned cluster ids as CV `groups` keeps
    near-duplicate / analog molecules together in a fold — tighter than a
    pure scaffold split when many molecules share a scaffold but differ in
    substituents, and it catches analog pairs that have *different* Murcko
    scaffolds.

    Butina is an exclusion-sphere clustering: it repeatedly picks the molecule
    with the most neighbours within the distance threshold (1 - cutoff) as a
    new cluster centroid and removes its neighbourhood. Singletons (molecules
    with no neighbour within the threshold) each become their own one-member
    cluster — handled naturally (every input row gets a cluster id, so the
    returned array is exactly len(smiles_list); no row is dropped).

    Args:
        smiles_list: list of SMILES strings. Invalid SMILES are tolerated:
            each invalid entry is given its own singleton cluster id (it has
            no fingerprint, so it cannot share a cluster — this keeps the
            output length aligned and avoids silently merging parse failures).
        cutoff: Tanimoto similarity threshold in [0, 1]. Higher cutoff =>
            stricter => more, smaller clusters (Butina distance = 1 - cutoff).
            0.6 is the common ECFP default (Butina 1999; RDKit cookbook).
        radius: Morgan fingerprint radius (2 == ECFP4).
        n_bits: fingerprint length.

    Returns:
        (groups, n_clusters): groups is an int np.ndarray of length
        len(smiles_list) with a cluster id per molecule (0..n_clusters-1,
        contiguous, assignment order = Butina cluster order with invalid /
        singleton molecules appended); n_clusters is the number of distinct
        clusters. Suitable to pass straight into cv_split / GroupKFold as
        `groups`.

    Wiring (how to select this instead of the default scaffold split):
        This is provided as an OPTION and deliberately does NOT change the
        default split of any existing molecular dataset (the default remains
        Murcko scaffold via assign_scaffold_groups in
        tests/find_best_hybrid_molecular.py — changing a dataset's split
        silently would change every published result for it). To use Butina
        for a molecular run, an experiment/loader that has the SMILES list can
        override the groups after load_data, e.g.:

            data = load_data('herg_ki', rep_names=[...])
            from experiments._shared import butina_clusters
            data['groups'], data['n_groups'] = butina_clusters(
                data['smiles'], cutoff=0.6)
            data['n_scaffolds'] = data['n_groups']

        (find_best_hybrid_molecular.assign_scaffold_groups is the function to
        mirror if/when a CLI flag for split-type is added there — that file is
        owned elsewhere, so the flag wiring is left as a documented follow-up.)
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.ML.Cluster import Butina

    smiles_list = list(smiles_list)
    n = len(smiles_list)

    # Build fingerprints; track which rows are valid (parseable).
    fps = []
    valid_rows = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=n_bits))
        valid_rows.append(i)

    groups = np.full(n, -1, dtype=int)

    if len(fps) >= 2:
        # Condensed lower-triangle distance matrix (1 - Tanimoto), as RDKit
        # Butina.ClusterData expects.
        dists = []
        for i in range(1, len(fps)):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend(1.0 - s for s in sims)
        clusters = Butina.ClusterData(
            dists, len(fps), 1.0 - cutoff, isDistData=True)
        # clusters is a tuple of tuples of *local* indices into `fps`.
        for cid, member_local_idxs in enumerate(clusters):
            for local_i in member_local_idxs:
                groups[valid_rows[local_i]] = cid
    elif len(fps) == 1:
        groups[valid_rows[0]] = 0

    # Any still-unassigned row (invalid SMILES, or the degenerate <2-fp case)
    # gets its own singleton cluster so the output stays length-n and no two
    # unparseable molecules are silently merged.
    next_id = int(groups.max()) + 1 if (groups >= 0).any() else 0
    for i in range(n):
        if groups[i] < 0:
            groups[i] = next_id
            next_id += 1

    return groups, int(len(np.unique(groups)))


def sequence_identity_clusters(seqs, identity_threshold: float = 0.95,
                               min_groups: int = 3):
    """Group protein/antibody sequences by near-identity (ISS-29/30).

    Exact-identity grouping (the current antibody `_identity_groups`) keeps
    *byte-identical* sequences together but puts two sequences that differ by
    a single residue into DIFFERENT groups, so analog / near-duplicate pairs
    leak across a GroupKFold train/test boundary. Measured on
    flab_thermo_tresanco (672 VHH rows, 556 unique): 899 row-level near-dup
    pairs (heavy chain Levenshtein distance <= 2), of which 784 (87%) span a
    train/test split under 5-fold exact-identity grouping. This helper closes
    that gap: any two sequences with normalised identity >= `identity_threshold`
    are placed in the same cluster (single-linkage), so near-duplicates stay
    on the same side of every fold.

    Normalised identity = 1 - levenshtein(a, b) / max(len(a), len(b)). Greedy
    single-linkage with length-band bucketing keeps it tractable for the
    antibody dataset sizes here (hundreds to a few thousand sequences); it is
    O(n^2) within a length band in the worst case but bands are small.

    Args:
        seqs: list of sequence strings (e.g. heavy chains; for paired data,
            concatenate heavy+light before calling so both chains count).
        identity_threshold: in (0, 1]. 1.0 reproduces exact-identity grouping
            (no near-dup merging). 0.95 merges sequences <=5% different — a
            sensible default that collapses single/double point-mutant analogs
            without over-merging genuinely distinct antibodies. Lower it to be
            stricter about leakage (bigger clusters), raise it to keep clusters
            tighter.
        min_groups: if the resulting cluster count is below this (degenerate,
            e.g. everything merged on a tiny near-clonal panel), fall back to a
            deterministic hash bucketing so GroupKFold still has enough groups —
            mirrors the existing `_identity_groups` min_groups guard.

    Returns:
        (groups, n_clusters): int np.ndarray of length len(seqs) with a cluster
        id per sequence, contiguous 0..n_clusters-1, plus the cluster count.
        Drop-in for use as CV `groups`.

    Wiring (ISS-29/30 fix, APPLIED — this helper is on the live path):
        tests/antibody_baseline_survey.py::_identity_groups is the single
        function that produces antibody CV groups for the `identity` strategy
        (FLAb panels, abcov, humanness) and it delegates to this helper. Paired
        datasets are handled there by concatenating heavy+light before calling;
        unpaired datasets pass heavy-only (empty light), which reduces to
        heavy with no behaviour change. tests/experiments/opig_humanness.py also
        routes its grouping through _identity_groups, so every antibody
        identity-strategy path uses this near-duplicate clustering. Strategies
        that fall back to identity when their metadata is missing
        (pre_split / edit_distance / experiment / position_modulo, see
        _assign_groups) reach it too.
    """
    seqs = list(seqs)
    n = len(seqs)
    if n == 0:
        return np.zeros(0, dtype=int), 0

    # MANDATORY Levenshtein -- no difflib fallback. difflib.SequenceMatcher
    # computes 2*matching_blocks/(len_a+len_b), NOT edit distance, so it yields
    # a DIFFERENT near-duplicate clustering -> different groups_hash/CV folds
    # than a machine that has Levenshtein. The ARC server was MISSING the
    # package and silently used difflib, so its antibody folds (and leakage
    # control) diverged from local and the cache self-drifted. Fail loud rather
    # than silently change the science. Pinned in setup.py / requirements.
    from Levenshtein import distance as _lev

    def _norm_identity(a, b):
        m = max(len(a), len(b))
        if m == 0:
            return 1.0
        return 1.0 - _lev(a, b) / m

    # Bucket by length band: two sequences can only reach `identity_threshold`
    # if their lengths are within (1 - threshold) * max_len of each other, so
    # we only need to compare within a band. Use exact length as the key and
    # also probe neighbouring lengths within the allowed gap.
    from collections import defaultdict
    by_len = defaultdict(list)
    for i, s in enumerate(seqs):
        by_len[len(s)].append(i)
    lengths = sorted(by_len.keys())

    parent = list(range(n))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    # First merge exact duplicates cheaply (within each length band).
    for L in lengths:
        idxs = by_len[L]
        seen = {}
        for i in idxs:
            s = seqs[i]
            if s in seen:
                _union(seen[s], i)
            else:
                seen[s] = i

    # Then near-duplicate single-linkage across compatible length bands.
    for li, L in enumerate(lengths):
        idxs_L = by_len[L]
        # candidate partner lengths: those within the max allowed length gap
        for L2 in lengths[li:]:
            if L2 == L:
                partners = idxs_L
                same_band = True
            else:
                if (L2 - L) / max(L2, 1) > (1.0 - identity_threshold):
                    break  # lengths sorted: further ones only get more distant
                partners = by_len[L2]
                same_band = False
            for ai, i in enumerate(idxs_L):
                start = ai + 1 if same_band else 0
                for j in partners[start:]:
                    if _find(i) == _find(j):
                        continue
                    if _norm_identity(seqs[i], seqs[j]) >= identity_threshold:
                        _union(i, j)

    # Relabel roots to contiguous ids. Order clusters CANONICALLY by their
    # lexicographically-smallest member sequence (unique per cluster), so the
    # integer labels depend only on the PARTITION CONTENT -- not on input row
    # order. This makes groups_hash reproducible across runs/machines and
    # invariant to row permutation. It changes only label VALUES, never which
    # samples co-cluster, so the CV folds are identical (science-neutral).
    members = defaultdict(list)
    for i in range(n):
        members[_find(i)].append(i)
    ordered_roots = sorted(members, key=lambda r: min(seqs[j] for j in members[r]))
    root_to_id = {r: new_id for new_id, r in enumerate(ordered_roots)}
    groups = np.array([root_to_id[_find(i)] for i in range(n)], dtype=int)
    n_clusters = len(root_to_id)

    # Degenerate-collapse guard (mirror _identity_groups' min_groups behaviour).
    if n_clusters < min_groups:
        import hashlib
        n_buckets = max(min_groups, n // 5)
        groups = np.array([
            int(hashlib.sha256(s.encode()).hexdigest(), 16) % n_buckets
            for s in seqs
        ])
        n_clusters = len(np.unique(groups))

    return groups, int(n_clusters)


# =============================================================================
# DOMAIN-AWARE METRIC DISPATCH
# =============================================================================

def _heuristic_metric_name(task_type: str, domain: Optional[str],
                           labels: Optional[np.ndarray]) -> str:
    """Fallback metric dispatcher for datasets not in DATASET_REGISTRY.

    Matches the historical behaviour of get_domain_metric, _resolve_search_metric,
    and precompute_hybrids._get_metric. Prefer the registry (see
    _resolve_dataset_metric); this exists only for datasets that haven't been
    registered yet.
    """
    if task_type == 'classification':
        n_classes = len(np.unique(labels)) if labels is not None else 2
        return 'f1_weighted' if n_classes > 2 else 'auc'
    if domain == 'dta':
        return 'ci'
    if domain == 'antibody':
        return 'spearman'
    if domain == 'nucleotide':
        return 'mcc'
    # molecular, reaction, tcga-regression (none currently), other
    return 'rmse'


def _resolve_dataset_metric(dataset_name: Optional[str] = None,
                            labels: Optional[np.ndarray] = None,
                            task_type: Optional[str] = None,
                            domain: Optional[str] = None,
                            metric: str = 'auto',
                            ) -> Tuple[str, str, bool]:
    """Canonical (metric, task_type, lower_is_better) resolver.

    Resolution order:
    1. If `metric` is an explicit string (not 'auto'), honour it and look up
       direction from BUILTIN_METRICS.
    2. If `dataset_name` is registered in DATASET_REGISTRY, use that entry —
       this is the preferred path and the one experiments should hit.
    3. Fallback: derive metric from task_type + domain via the heuristic
       (historical behaviour).

    Every return path is consistent with BUILTIN_METRICS for direction.
    """
    from kirby.find_best_hybrid import BUILTIN_METRICS

    # 1. Explicit metric wins. Still need task_type.
    if metric != 'auto':
        if task_type not in ('classification', 'regression') and dataset_name:
            info = registry_lookup(dataset_name)
            if info is not None:
                task_type = info.task_type
        if task_type not in ('classification', 'regression'):
            raise ValueError(
                f"_resolve_dataset_metric: explicit metric={metric!r} requires "
                f"task_type. Got task_type={task_type!r}, "
                f"dataset_name={dataset_name!r}."
            )
        if metric not in BUILTIN_METRICS:
            raise ValueError(
                f"_resolve_dataset_metric: metric={metric!r} is not in "
                f"BUILTIN_METRICS. Available: {sorted(BUILTIN_METRICS.keys())}"
            )
        _, lower_is_better = BUILTIN_METRICS[metric]
        return metric, task_type, lower_is_better

    # 2. Registry lookup by dataset_name.
    if dataset_name:
        info = registry_lookup(dataset_name)
        if info is not None:
            # If caller also passed task_type, cross-check for safety.
            if task_type is not None and task_type != info.task_type:
                raise ValueError(
                    f"_resolve_dataset_metric: caller passed "
                    f"task_type={task_type!r} for '{dataset_name}' but "
                    f"registry says {info.task_type!r}. Mismatch likely "
                    f"indicates a loader bug — fix one side."
                )
            return info.metric, info.task_type, info.lower_is_better

    # 3. Heuristic fallback.
    if task_type not in ('classification', 'regression'):
        raise ValueError(
            f"_resolve_dataset_metric: task_type={task_type!r} invalid and "
            f"dataset_name={dataset_name!r} is not in DATASET_REGISTRY. "
            f"Register the dataset or pass task_type explicitly."
        )
    chosen = _heuristic_metric_name(task_type, domain, labels)
    _, lower_is_better = BUILTIN_METRICS[chosen]
    return chosen, task_type, lower_is_better


def get_domain_metric(data: dict) -> Tuple[str, Any, bool]:
    """Return (metric_name, metric_fn, lower_is_better) for a loaded dataset.

    Prefers DATASET_REGISTRY when data['dataset_name'] is set (load_data sets
    it as of 2026-04-17). Falls back to the task_type + domain heuristic for
    datasets not yet registered.
    """
    from kirby.find_best_hybrid import BUILTIN_METRICS

    dataset_name = data.get('dataset_name')
    task_type = data.get('task_type')
    if task_type not in ('classification', 'regression'):
        raise ValueError(
            f"get_domain_metric: data['task_type']={task_type!r}. "
            f"Must be 'classification' or 'regression'. Check the loader."
        )
    domain = data.get('domain', 'molecular')
    labels = data.get('labels')

    metric, _, lower_is_better = _resolve_dataset_metric(
        dataset_name=dataset_name, labels=labels, task_type=task_type,
        domain=domain, metric='auto')
    fn, _ = BUILTIN_METRICS[metric]
    return metric, fn, lower_is_better


# =============================================================================
# EVALUATION HELPERS
# =============================================================================

def evaluate_cv(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    model_factory,
    n_folds: int = 5,
    # create_hybrid's default, tracked — see create_hybrid_cached above.
    allocation_method: str = 'performance_weighted',
    importance_method: str = 'lightgbm_gain',
    budget: int = 100,
    seed: int = 42,
    mode: str = 'full',
    metric_fn=None,
) -> Tuple[float, float, int]:
    """Proper per-fold CV: create_hybrid on train, apply_feature_selection on val.

    Returns (mean_score, std_score, n_features).
    """
    if metric_fn is None:
        metric_fn = r2_score

    sample_X = next(iter(base_reps.values()))
    scores = []
    n_feats = []

    # `mode` is vestigial in cv_split (split strategy is set by n_folds); do
    # not forward it, to avoid the cv_split deprecation warning per fold.
    for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels, seed):
        train_reps = {k: v[train_idx] for k, v in base_reps.items()}
        val_reps = {k: v[val_idx] for k, v in base_reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]

        try:
            X_train, feature_info = create_hybrid(
                train_reps, y_train,
                allocation_method=allocation_method,
                budget=budget,
                importance_method=importance_method,
                # Outer CV is group-aware (cv_split above); the internal
                # train/val split must be too, else it falls back to a random
                # 20% carve and ranks features under scaffold leakage.
                groups=groups[train_idx] if groups is not None else None,
            )
            X_val = apply_feature_selection(val_reps, feature_info)
        except Exception:
            continue

        scaler = FeatureScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        model = model_factory()
        model.fit(X_train_s, y_train)
        scores.append(metric_fn(y_val, model.predict(X_val_s)))
        n_feats.append(X_train.shape[1])

    if not scores:
        return float('nan'), float('nan'), 0
    return float(np.mean(scores)), float(np.std(scores)), int(np.mean(n_feats))


def evaluate_cv_with_train(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    model_factory,
    n_folds: int = 5,
    # create_hybrid's default, tracked — see create_hybrid_cached above.
    allocation_method: str = 'performance_weighted',
    importance_method: str = 'lightgbm_gain',
    budget: int = 100,
    seed: int = 42,
    mode: str = 'full',
    metric_fn=None,
    internal_evaluator: str = 'match',
    augmentations: Optional[Dict[str, np.ndarray]] = None,
    augmentation_strategy: str = 'uncapped',
) -> Tuple[float, float, float, float, int]:
    """Like evaluate_cv but also returns train score.

    Returns (val_mean, val_std, train_mean, train_std, n_features).

    `internal_evaluator` defaults to 'match' (the canonical default used
    across the hybrid search). It was previously 'ridge'; Ridge is forbidden
    as an internal hybrid-search evaluator (feedback_no_ridge.md), so the
    default no longer silently selects it. Callers that want a specific
    evaluator (e.g. c09 forwarding a blueprint's evaluator) still pass it
    explicitly.
    """
    from kirby.hybrid import apply_augmentation_selection

    if metric_fn is None:
        metric_fn = r2_score

    sample_X = next(iter(base_reps.values()))
    val_scores = []
    train_scores = []
    n_feats = []

    # `mode` is vestigial in cv_split (split strategy is set by n_folds); do
    # not forward it, to avoid the cv_split deprecation warning per fold.
    for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels, seed):
        train_reps = {k: v[train_idx] for k, v in base_reps.items()}
        val_reps = {k: v[val_idx] for k, v in base_reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]
        train_augs = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None

        try:
            X_train, feature_info = create_hybrid(
                train_reps, y_train,
                allocation_method=allocation_method,
                budget=budget,
                importance_method=importance_method,
                internal_evaluator=internal_evaluator,
                augmentations=train_augs,
                augmentation_strategy=augmentation_strategy,
                # Same as evaluate_cv: keep create_hybrid's internal split
                # group-aware instead of the random 20% fallback.
                groups=groups[train_idx] if groups is not None else None,
            )
            X_val = apply_feature_selection(val_reps, feature_info)
            aug_info = feature_info.get('augmentation_info')
            if aug_info and augmentations is not None:
                val_augs = {k: v[val_idx] for k, v in augmentations.items()}
                X_val_aug = apply_augmentation_selection(val_augs, aug_info)
                if X_val_aug is not None:
                    X_val = np.hstack([X_val, X_val_aug])
        except Exception:
            continue

        scaler = FeatureScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        model = model_factory()
        model.fit(X_train_s, y_train)
        # ⛔ GRADED SCORES FOR A RANKING METRIC. model.predict returns hard 0/1
        # on a classifier, and an AUROC read off hard labels collapses to the
        # mean of sensitivity and specificity (hybrid.py:2661). This is the
        # same defect that was fixed in c05's all-rows arm and in c22.
        # _predict_for_metric returns class-1 probabilities when metric_fn is
        # the built-in auc and the model's own predictions otherwise.
        from kirby.find_best_hybrid import _predict_for_metric as _pfm
        val_scores.append(metric_fn(y_val, _pfm(model, X_val_s, metric_fn)))
        train_scores.append(metric_fn(y_train, _pfm(model, X_train_s, metric_fn)))
        n_feats.append(X_train.shape[1])

    if not val_scores:
        return float('nan'), float('nan'), float('nan'), float('nan'), 0
    return (
        float(np.mean(val_scores)), float(np.std(val_scores)),
        float(np.mean(train_scores)), float(np.std(train_scores)),
        int(np.mean(n_feats)),
    )


def cap_dimensions(X_train, X_val=None, max_dim=None):
    """PCA-reduce if features exceed max_dim. Returns (X_train, X_val) or (X_train,).

    If max_dim is None, defaults to n_train (X_train.shape[0]).
    Only applies PCA when X_train.shape[1] > max_dim.
    """
    if max_dim is None:
        max_dim = X_train.shape[0]
    if X_train.shape[1] <= max_dim:
        return (X_train, X_val) if X_val is not None else (X_train,)
    n_components = max(1, min(max_dim, X_train.shape[0] - 1))
    pca = PCA(n_components=n_components)
    X_train_r = pca.fit_transform(X_train)
    if X_val is not None:
        return X_train_r, pca.transform(X_val)
    return (X_train_r,)


def evaluate_single_rep_cv(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    model_factory,
    n_folds: int = 5,
    mode: str = 'full',
    metric_fn=None,
    return_width: bool = False,
    cap: bool = True,
):
    """Simple CV for a single representation matrix.

    Returns (mean_score, std_score), or (mean_score, std_score, widths) when
    return_width is set. `widths` is the number of features actually fitted in
    each fold.

    Two things this function does that a caller cannot see from the outside, and
    both change the number it returns:

    Predictions go through _predict_for_metric. On a dataset whose registered
    metric is AUROC, model.predict returns hard 0/1 class labels and the AUROC
    computed from them is not the AUROC of the model — it is the AUROC of a
    single threshold. _predict_for_metric hands back predicted probabilities
    for that one metric and class labels for every other.

    cap_dimensions PCA-reduces the matrix whenever a representation has more
    features than the fold has training rows, so a 4,800-feature representation
    on 1,100 training rows is scored at 1,100 components, not 4,800. Nothing
    recorded the width that produced the score. return_width does.

    cap=False turns that rotation off and fits every feature. Measured on
    flab_thermo_tresanco, random forest, spearman, the same five folds: saprot
    scores 0.3775 rotated and 0.5896 at its own 1,280 features, esmc 0.2405 and
    0.5454, esm3 0.3714 and 0.5302. The unrotated numbers are the ones
    find_best_hybrid's own full-dimension baselines report, so with cap=True a
    caller's baseline and the search's baseline disagree about the same
    representation on the same folds by 0.16 to 0.30 spearman. A random forest
    splits on one feature at a time, so rotating the axes is not free for it.
    c11 took the same rotation off its baselines for the same reason
    (c11_ensemble_vs_hybrid.py _prep).
    """
    if metric_fn is None:
        metric_fn = r2_score

    from kirby.find_best_hybrid import _predict_for_metric

    scores = []
    widths = []
    # `mode` is vestigial in cv_split; do not forward it.
    for train_idx, val_idx in cv_split(groups, n_folds, X, labels):
        scaler = FeatureScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        # A held-out value beyond anything seen in training is cut to the
        # training range before predicting (owner, 2026-10-05). On adme_hlm
        # pdv fold 0 one held-out molecule sat 83 million training standard
        # deviations out on one feature and the dnn's rmse ran to thousands.
        X_val = np.clip(X_val, X_train.min(axis=0), X_train.max(axis=0))

        # PCA-reduce if features exceed training samples
        if cap:
            X_train, X_val = cap_dimensions(X_train, X_val)
        widths.append(int(X_train.shape[1]))

        model = model_factory()
        model.fit(X_train, labels[train_idx])
        scores.append(metric_fn(labels[val_idx],
                                _predict_for_metric(model, X_val, metric_fn)))

    mean, std = float(np.mean(scores)), float(np.std(scores))
    if return_width:
        return mean, std, widths
    return mean, std


def evaluate_weighted_knn(
    X: np.ndarray,
    y: np.ndarray,
    importance_scores: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 3,
    k_values: List[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate importance scores as feature weights for k-NN regression.

    Compares unweighted vs importance-weighted k-NN.
    Returns {k: {'unweighted_r2', 'weighted_r2', 'delta'}}.
    """
    from sklearn.neighbors import KNeighborsRegressor as _KNR

    if k_values is None:
        k_values = [5, 10, 20]

    weights = np.sqrt(np.maximum(importance_scores, 0))
    weights = weights / (weights.sum() + 1e-10)

    results = {}
    for k in k_values:
        unweighted_scores = []
        weighted_scores = []

        for train_idx, val_idx in cv_split(groups, n_folds, X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = FeatureScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            knn = _KNR(n_neighbors=min(k, len(y_train) - 1))
            knn.fit(X_train_s, y_train)
            unweighted_scores.append(r2_score(y_val, knn.predict(X_val_s)))

            X_train_w = X_train_s * weights
            X_val_w = X_val_s * weights
            knn_w = _KNR(n_neighbors=min(k, len(y_train) - 1))
            knn_w.fit(X_train_w, y_train)
            weighted_scores.append(r2_score(y_val, knn_w.predict(X_val_w)))

        uw = float(np.mean(unweighted_scores))
        w = float(np.mean(weighted_scores))
        results[k] = {'unweighted_r2': uw, 'weighted_r2': w, 'delta': w - uw}

    return results


def importance_feature_weights(importance_scores, exponent=None):
    """Importance scores turned into the multiplier for standardised features.

    s13 and five other similarity experiments wrote `np.sqrt(np.maximum(imp, 0))`
    inline: the features are multiplied by the square root of the importance,
    whatever distance is in force. compute_distance_matrix does something else
    for the same quantity -- it raises the importance to the power its metric's
    family implies, which is 1 under manhattan and 0.5 under euclidean. Both
    were called "the sqrt weighting"; they are the same number only under
    euclidean and cosine.

    `exponent` is the power the clipped importance is raised to. None reads
    KIRBY_IMPORTANCE_EXPONENT, and an unset environment gives 0.5, which is what
    the inline expression computed. 0 returns an all-ones vector -- every
    feature weighs the same, so the geometry is the unweighted one.

    Normalising is left to the caller, which is where it already is.
    """
    from kirby.similarity.preprocessing import importance_exponent_override
    alpha = exponent
    if alpha is None:
        alpha = importance_exponent_override()
    if alpha is None:
        alpha = 0.5
    w = np.maximum(np.nan_to_num(np.asarray(importance_scores, dtype=np.float64),
                                 nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    return np.power(w, float(alpha))


def make_nearest_neighbors(n_neighbors, metric=None, **kwargs):
    """A sklearn NearestNeighbors that obeys --distance-metric.

    Use this instead of NearestNeighbors(...) anywhere the neighbours are an
    ANSWER — a noise score, an applicability-domain distance, a retrieval set, a
    reliability weight, a hubness count. Those are the quantities the metric
    sweep is about.

    Do NOT use it for a KNN that is being fitted as a MODEL. Those tune their own
    metric per fold through _HP_SEARCH_SPACES['knn'], and forcing this one on
    them would overwrite the tuned choice with the sweep's.

    `metric` still wins when passed; None reads the process-wide setting, which
    is manhattan by default. Before 2026-09-04 every one of these took sklearn's
    own default of Minkowski p=2, which is Euclidean, at 36 places across
    complexity/, similarity/, noise/, opig_* and federated/.
    """
    kw = sklearn_distance_kwargs(metric)
    kw.update(kwargs)
    return NearestNeighbors(n_neighbors=n_neighbors, **kw)


def evaluate_similarity_retrieval(
    X: np.ndarray,
    y: np.ndarray,
    importance_scores: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 3,
    k_neighbors: int = 10,
) -> Dict[str, float]:
    """Evaluate importance scores as weights for similarity-based retrieval.

    For each test molecule, find k nearest neighbors in training set.
    Measure how well neighbor target values predict the test target.
    Returns {'unweighted_mae', 'weighted_mae', 'delta_mae',
             'unweighted_r2', 'weighted_r2', 'delta_r2'}.
    """
    from sklearn.metrics import mean_absolute_error

    weights = np.sqrt(np.maximum(importance_scores, 0))
    weights = weights / (weights.sum() + 1e-10)

    all_unweighted_preds = []
    all_weighted_preds = []
    all_true = []

    for train_idx, val_idx in cv_split(groups, n_folds, X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = FeatureScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        nn_uw = make_nearest_neighbors(min(k_neighbors, len(y_train) - 1))
        nn_uw.fit(X_train_s)
        dists_uw, idx_uw = nn_uw.kneighbors(X_val_s)
        w_uw = 1.0 / (dists_uw + 1e-10)
        preds_uw = np.average(y_train[idx_uw], axis=1, weights=w_uw)

        X_train_w = X_train_s * weights
        X_val_w = X_val_s * weights
        nn_w = make_nearest_neighbors(min(k_neighbors, len(y_train) - 1))
        nn_w.fit(X_train_w)
        dists_w, idx_w = nn_w.kneighbors(X_val_w)
        w_w = 1.0 / (dists_w + 1e-10)
        preds_w = np.average(y_train[idx_w], axis=1, weights=w_w)

        all_unweighted_preds.extend(preds_uw)
        all_weighted_preds.extend(preds_w)
        all_true.extend(y_val)

    all_true = np.array(all_true)
    all_unweighted_preds = np.array(all_unweighted_preds)
    all_weighted_preds = np.array(all_weighted_preds)

    uw_mae = mean_absolute_error(all_true, all_unweighted_preds)
    w_mae = mean_absolute_error(all_true, all_weighted_preds)
    uw_r2 = r2_score(all_true, all_unweighted_preds)
    w_r2 = r2_score(all_true, all_weighted_preds)

    return {
        'unweighted_mae': float(uw_mae),
        'weighted_mae': float(w_mae),
        'delta_mae': float(uw_mae - w_mae),
        'unweighted_r2': float(uw_r2),
        'weighted_r2': float(w_r2),
        'delta_r2': float(w_r2 - uw_r2),
    }


# =============================================================================
# MODEL FACTORIES
# =============================================================================

def get_model_factories(model_names: List[str],
                        task_type: str = 'regression',
                        n_samples: Optional[int] = None) -> Dict[str, callable]:
    """Return {name: factory_fn} for requested models.

    Delegates to find_best_hybrid.DEFAULT_MODELS for authoritative model
    definitions — every name defined there, with size guards and clf/reg
    dispatch. It does not filter by tier: a caller asking for a name labelled
    'extra' or 'off' still gets its factory.

    Args:
        model_names: List of model names (e.g. ['rf', 'ridge', 'gp']).
        task_type: 'regression' or 'classification'.
        n_samples: Dataset size — models with _max_samples auto-skip.

    Returns:
        {name: callable} where each callable returns a fresh model instance.
    """
    from kirby.find_best_hybrid import get_model as _get_model, DEFAULT_MODELS

    factories = {}
    for name in model_names:
        if name not in DEFAULT_MODELS:
            print(f"  WARNING: model '{name}' not in DEFAULT_MODELS, skipping")
            continue
        spec = DEFAULT_MODELS[name]
        max_s = spec.get('_max_samples')
        if max_s is not None and n_samples is not None and n_samples > max_s:
            print(f"  Skipping {name}: n_samples={n_samples} > _max_samples={max_s}")
            continue

        def _make_factory(_name, _task_type):
            return lambda: _get_model(_name, _task_type)

        factories[name] = _make_factory(name, task_type)
    return factories


def get_model_family_variants(
    task_type: str = 'regression',
) -> Dict[str, List[Tuple[str, callable, float]]]:
    """Return model families with complexity gradients, per task_type.

    Each family: list of (variant_name, factory_fn, capacity_index), where a
    larger capacity_index means a higher-capacity / less-regularised variant.

    `task_type` selects regressor vs classifier estimators so callers (c06,
    etc.) study a capacity gradient with the right model class — previously
    this hardcoded regressors, so on classification data the variants were
    trained on integer labels and scored with classification metrics, which
    is silently wrong (audit CXA-6).

    The linear family uses Lasso (alpha sweep) for regression and
    LogisticRegression (C sweep) for classification — both give a capacity
    gradient over a downstream linear model. Ridge was replaced by Lasso here
    (2026-07-18) so ridge is gone everywhere; LinearRegression could not be used
    because it has no regularisation knob and so cannot produce the capacity
    gradient this study needs.
    """
    is_clf = (task_type == 'classification')

    if is_clf:
        from sklearn.ensemble import RandomForestClassifier as _RF
        from sklearn.linear_model import LogisticRegression as _Logit
        from sklearn.neighbors import KNeighborsClassifier as _KNN
        from kirby.models.dnn import DNNClassifier as _NN
    else:
        from sklearn.ensemble import RandomForestRegressor as _RF
        from sklearn.linear_model import Lasso
        from sklearn.neighbors import KNeighborsRegressor as _KNN
        from kirby.models.dnn import DNNRegressor as _NN

    families = {
        'RF': [
            ('depth=3', lambda: _RF(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1), 1),
            ('depth=5', lambda: _RF(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1), 2),
            ('depth=10', lambda: _RF(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1), 3),
            ('depth=None', lambda: _RF(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1), 4),
        ],
        'kNN': [
            ('k=20', lambda: _KNN(n_neighbors=20, n_jobs=-1), 1),
            ('k=10', lambda: _KNN(n_neighbors=10, n_jobs=-1), 2),
            ('k=5', lambda: _KNN(n_neighbors=5, n_jobs=-1), 3),
            ('k=1', lambda: _KNN(n_neighbors=1, n_jobs=-1), 4),
        ],
        # The neural family. It was sklearn's MLP at four widths until
        # 2026-09-18, when the owner cut mlp from the suite: it did not perform
        # and it was slow. The same four widths are now built from KIRBy's own
        # dnn (src/kirby/models/dnn.py), which is the neural model every search
        # scores, so this family and the search agree on what a neural model is.
        # Everything but the width is dnn's default: dropout 0.2, lr 1e-3, 200
        # epochs, batch 256, patience 20.
        #
        # ⛔ THE FAMILY KEY CHANGED FROM 'MLP' TO 'DNN'. c06 keys its results by
        # family name, so a stored c06 result under 'MLP' was produced by a
        # different estimator and is not comparable to one under 'DNN'.
        'DNN': [
            ('16', lambda: _NN(hidden_sizes=(16,), random_state=42), 1),
            ('64', lambda: _NN(hidden_sizes=(64,), random_state=42), 2),
            ('128,64', lambda: _NN(hidden_sizes=(128, 64), random_state=42), 3),
            ('256,128,64', lambda: _NN(hidden_sizes=(256, 128, 64), random_state=42), 4),
        ],
    }

    # Linear family: capacity rises with weaker regularisation. For regression
    # this is Lasso (small alpha = high capacity); for classification it is
    # LogisticRegression (large C = high capacity). No Ridge anywhere.
    if is_clf:
        families['Logit'] = [
            ('C=0.01', lambda: _Logit(C=0.01, max_iter=1000), 1),
            ('C=0.1', lambda: _Logit(C=0.1, max_iter=1000), 2),
            ('C=1.0', lambda: _Logit(C=1.0, max_iter=1000), 3),
            ('C=100', lambda: _Logit(C=100.0, max_iter=1000), 4),
        ]
    else:
        families['Lasso'] = [
            ('alpha=100', lambda: Lasso(alpha=100.0, max_iter=5000), 1),
            ('alpha=10', lambda: Lasso(alpha=10.0, max_iter=5000), 2),
            ('alpha=1', lambda: Lasso(alpha=1.0, max_iter=5000), 3),
            ('alpha=0.01', lambda: Lasso(alpha=0.01, max_iter=5000), 4),
        ]

    try:
        if is_clf:
            from xgboost import XGBClassifier as _XGB
        else:
            from xgboost import XGBRegressor as _XGB
        families['XGBoost'] = [
            ('depth=2', lambda: _XGB(n_estimators=100, max_depth=2, random_state=42, n_jobs=-1, verbosity=0), 1),
            ('depth=4', lambda: _XGB(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1, verbosity=0), 2),
            ('depth=6', lambda: _XGB(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1, verbosity=0), 3),
            ('depth=None', lambda: _XGB(n_estimators=100, max_depth=0, random_state=42, n_jobs=-1, verbosity=0, tree_method='hist'), 4),
        ]
    except ImportError:
        pass

    # SVM: C controls capacity (low C = high regularisation = low capacity)
    if is_clf:
        from sklearn.svm import SVC as _SVC
        families['SVM'] = [
            ('C=0.01', lambda: _SVC(kernel='rbf', C=0.01, probability=False), 1),
            ('C=0.1', lambda: _SVC(kernel='rbf', C=0.1, probability=False), 2),
            ('C=1.0', lambda: _SVC(kernel='rbf', C=1.0, probability=False), 3),
            ('C=10.0', lambda: _SVC(kernel='rbf', C=10.0, probability=False), 4),
        ]
    else:
        # Labels standardised inside each fit, as in the search since fb1df54
        # (dirty-run bug 4, 2026-10-05). A plain SVR has epsilon 0.1, which on
        # labels with a spread far from 1 fits nothing.
        from kirby.hybrid import SVRScaledTarget as SVR
        families['SVM'] = [
            ('C=0.01', lambda: SVR(kernel='rbf', C=0.01), 1),
            ('C=0.1', lambda: SVR(kernel='rbf', C=0.1), 2),
            ('C=1.0', lambda: SVR(kernel='rbf', C=1.0), 3),
            ('C=10.0', lambda: SVR(kernel='rbf', C=10.0), 4),
        ]

    # GP: length_scale controls capacity (large = smooth/low capacity, small = wiggly/high)
    # Size guard: GP is O(n³), only viable for n ≤ 2000
    try:
        from sklearn.gaussian_process.kernels import RBF as _RBF, WhiteKernel as _WK
        if is_clf:
            from sklearn.gaussian_process import GaussianProcessClassifier as _GPC
            families['GP'] = [
                ('ls=10', lambda: _GPC(kernel=_RBF(length_scale=10.0), random_state=42), 1),
                ('ls=1', lambda: _GPC(kernel=_RBF(length_scale=1.0), random_state=42), 2),
                ('ls=0.1', lambda: _GPC(kernel=_RBF(length_scale=0.1), random_state=42), 3),
                ('ls=0.01', lambda: _GPC(kernel=_RBF(length_scale=0.01), random_state=42), 4),
            ]
        else:
            from sklearn.gaussian_process import GaussianProcessRegressor as _GPR
            families['GP'] = [
                ('ls=10', lambda: _GPR(
                    kernel=_RBF(length_scale=10.0) + _WK(), n_restarts_optimizer=2, random_state=42), 1),
                ('ls=1', lambda: _GPR(
                    kernel=_RBF(length_scale=1.0) + _WK(), n_restarts_optimizer=2, random_state=42), 2),
                ('ls=0.1', lambda: _GPR(
                    kernel=_RBF(length_scale=0.1) + _WK(), n_restarts_optimizer=2, random_state=42), 3),
                ('ls=0.01', lambda: _GPR(
                    kernel=_RBF(length_scale=0.01) + _WK(), n_restarts_optimizer=2, random_state=42), 4),
            ]
    except ImportError:
        pass

    return families


# =============================================================================
# METRICS
# =============================================================================

def compute_intrinsic_dimensionality(X: np.ndarray) -> Dict[str, Any]:
    """Compute intrinsic dimensionality using PCA and MLE methods."""
    results = {'nominal_dim': X.shape[1]}

    scaler = FeatureScaler()
    X_scaled = scaler.fit_transform(X)
    n, d = X_scaled.shape

    # ⛔ TWO ESTIMATORS, AND THE ROW NOW SAYS WHICH ONE. A representation of at
    # most 2000 features gets an exact PCA over the whole standardised matrix
    # and an MLE over that same matrix. A wider one gets TruncatedSVD capped at
    # 500 components and an MLE over a 200-component projection. Both land in
    # the same columns, so a table comparing representations compares two
    # estimators wherever the pool mixes widths -- on the flab protein pool
    # exactly one of ten (onehot, 5600 features) crosses the line. And a
    # threshold never reached inside those 500 components was reported AS 500,
    # which reads as a measurement.
    results['estimator'] = 'truncated_svd_500' if d > 2000 else 'exact_pca'
    results['n_features'] = int(d)
    if d > 2000:
        from sklearn.decomposition import TruncatedSVD
        max_comp = min(500, n - 1)
        results['max_components'] = int(max_comp)
        svd = TruncatedSVD(n_components=max_comp, random_state=42)
        svd.fit(X_scaled)
        cumvar = np.cumsum(svd.explained_variance_ratio_)
        for threshold in [0.90, 0.95, 0.99]:
            key = f'pca_{int(threshold * 100)}'
            idx = np.searchsorted(cumvar, threshold)
            results[key] = int(min(idx + 1, max_comp)) if idx < len(cumvar) else max_comp
            # True where the threshold was NOT reached inside the components
            # computed, so the number is the ceiling, not the dimension.
            results[f'{key}_at_ceiling'] = bool(idx >= len(cumvar))
    else:
        n_components = min(n, d)
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        for threshold in [0.90, 0.95, 0.99]:
            key = f'pca_{int(threshold * 100)}'
            intrinsic_dim = np.searchsorted(cumvar, threshold) + 1
            results[key] = int(min(intrinsic_dim, d))

    X_for_mle = X_scaled
    results['mle_on'] = 'full matrix'
    if d > 2000:
        from sklearn.decomposition import TruncatedSVD
        _mle_comp = min(200, n - 1)
        results['mle_on'] = f'truncated_svd {_mle_comp} components'
        X_for_mle = TruncatedSVD(n_components=_mle_comp, random_state=42).fit_transform(X_scaled)
    try:
        results['mle'] = float(estimate_intrinsic_dim_mle(X_for_mle, k=10))
    except Exception:
        results['mle'] = None

    if results.get('pca_95'):
        results['compression_ratio'] = X.shape[1] / results['pca_95']

    return results


def estimate_intrinsic_dim_mle(X: np.ndarray, k: int = 10) -> float:
    """MLE estimator of intrinsic dimensionality (Levina-Bickel 2004)."""
    n_samples = X.shape[0]
    k = min(k, n_samples - 1)

    nn = make_nearest_neighbors(k + 1)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    distances = distances[:, 1:]

    estimates = []
    for i in range(n_samples):
        dists = distances[i]
        if dists[-1] > 0:
            log_ratios = np.log(dists[-1] / dists[:-1])
            if np.sum(log_ratios) > 0:
                estimates.append((k - 1) / np.sum(log_ratios))

    return np.mean(estimates) if estimates else np.nan


def estimate_mutual_information_knn(X: np.ndarray, y: np.ndarray, k: int = 5,
                                    max_dim: int = 50) -> float:
    """Estimate I(X; Y) with a robust kNN estimator.

    History: this previously ran a hand-rolled joint KSG estimator (Kraskov
    et al. 2004) on PCA-reduced features. With ``max_dim`` up to 50 it operated
    far outside the KSG validity regime (joint dimension >> the ~6 dims KSG
    tolerates for n in the low hundreds), so the estimate was systematically
    negative and clamped to 0. The result was that *every* MI value across
    c01/c09/c17/s03 came out 0 — the metric was dead. See
    COMPLEXITY_SIMILARITY_BROKEN_2026_06_03.md.

    This replacement uses scikit-learn's per-feature Kraskov estimator
    (``mutual_info_regression`` / ``mutual_info_classif``), summed over a set of
    PCA-decorrelated components. It (a) does not collapse to 0, (b) auto-detects
    a discrete target and uses the classification estimator (the old code scaled
    class labels as if continuous, which was wrong for humanness/tcga/tz), and
    (c) keeps the n >> d guard so larger feature blocks can still carry more
    total information. Returns total estimated information in nats (>= 0).

    Args:
        k: neighbours for the Kraskov estimator (sklearn ``n_neighbors``).
        max_dim: cap on PCA components, further capped at n_samples // 5.
    """
    from sklearn.feature_selection import (
        mutual_info_regression, mutual_info_classif)

    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y).ravel()
    n_samples, d = X.shape
    if n_samples < 10:
        return 0.0  # too few samples for any stable MI estimate
    k = max(1, min(k, n_samples - 1))

    X_scaled = FeatureScaler().fit_transform(X)

    # Keep the n >> d guard, but PCA-decorrelate so per-feature MI can be summed
    # without double-counting linearly-shared information.
    max_components = max(1, min(max_dim, n_samples // 5))
    if d > max_components:
        from sklearn.decomposition import PCA as _PCA
        X_scaled = _PCA(n_components=max_components,
                        random_state=42).fit_transform(X_scaled)

    # Detect a discrete target: few unique values and (near-)integer encoded.
    y_unique = np.unique(y)
    is_discrete = (y_unique.size <= max(20, int(0.05 * n_samples))
                   and np.allclose(y, np.round(y)))

    def _mi(target):
        if is_discrete:
            return float(np.sum(mutual_info_classif(
                X_scaled, target.astype(int), n_neighbors=k, random_state=42)))
        return float(np.sum(mutual_info_regression(
            X_scaled, target, n_neighbors=k, random_state=42)))

    mi_raw = _mi(y)

    # Permutation bias correction. The summed per-feature kNN estimator has a
    # positive finite-sample bias that grows with the number of components, so
    # it does NOT cancel when comparing feature blocks of different sizes
    # (e.g. s03 full-vs-selected) and can swamp weak signal at small n. Subtract
    # the mean MI under label permutation (the null) to de-bias. Deterministic
    # seeds keep the estimate reproducible across runs.
    n_perm = 3
    null = np.mean([_mi(np.random.default_rng(1000 + s).permutation(y))
                    for s in range(n_perm)])

    return float(max(0.0, mi_raw - null))


def fit_learning_curve(n_samples: np.ndarray, scores: np.ndarray,
                       lower_is_better: bool = False) -> Dict[str, float]:
    """Fit score = asymptote - a/n^b and return the asymptote at infinite n.

    The curve is fitted on the score turned so that it RISES with n, then the
    asymptote is turned back into the dataset's own metric. Fixed 2026-09-10
    (metric-direction audit): the fit used to hardcode a rising curve with the
    asymptote bounded to [0, 1], so on an rmse dataset whose error exceeds 1 --
    logd sits near 1.17 -- the starting guess was outside its own bound,
    curve_fit raised on every call, and the except branch returned max(scores),
    which is the WORST score in the sweep, under the name of the asymptote.

    'r2_max' keeps its name so older result files stay readable; it holds the
    asymptote in the dataset's own metric, whatever that metric is.
    """
    from scipy.optimize import curve_fit

    scores = np.asarray(scores, dtype=float)
    # Higher-is-always-better while fitting, so the same rising form works for
    # rmse and for spearman.
    y = -scores if lower_is_better else scores

    def model(n, asym, a, b):
        return asym - a / np.power(n, b)

    try:
        span = float(np.max(y) - np.min(y))
        if not np.isfinite(span) or span <= 0:
            span = max(abs(float(np.max(y))), 1.0)
        # Bound the asymptote by the observed range rather than by [0, 1]:
        # an rmse of 1.17 is not out of range, it is Tuesday.
        lo, hi = float(np.min(y)) - span, float(np.max(y)) + span
        p0 = [float(np.max(y)), 1.0, 0.5]
        bounds = ([lo, 0, 0.1], [hi, 100, 2])
        popt, _ = curve_fit(model, n_samples, y, p0=p0, bounds=bounds, maxfev=5000)
        asym, a, b = popt

        predicted = model(n_samples, *popt)
        residuals = y - predicted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        fit_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {'r2_max': float(-asym if lower_is_better else asym),
                'a': float(a), 'b': float(b), 'fit_r2': float(fit_r2),
                'lower_is_better': bool(lower_is_better)}
    except Exception as e:
        # The best score actually seen, not the largest one.
        best = float(np.min(scores) if lower_is_better else np.max(scores))
        return {'error': str(e), 'r2_max': best,
                'lower_is_better': bool(lower_is_better)}


# How many of the outer folds the experiments that run SEARCHES OF THEIR OWN
# actually run, e.g. {'c06': 3, 'c07': 3, 'c13': 3, 'c23': 3} = the FIRST 3 of
# the 5-fold layout. The layout itself is unchanged -- the folds are the same
# rows as every other experiment's folds 0, 1 and 2, the stored searches keep
# their 5-fold names (ocv5f0s42 ...), and folds 3 and 4 can be added later
# without redoing the first three. An experiment absent from this table runs
# every fold. KIRBY_FIRST_FOLDS=<K> overrides the table for one process.
# EMPTY on 2026-10-07 (Adelaide: "hold off on the fold change for now"); wired
# into c06, c07, c13 and c23, switched on by filling this table.
OUTER_FOLDS_RUN = {}


def first_outer_folds(folds, exp_id):
    """The outer folds `exp_id` runs: the first OUTER_FOLDS_RUN[exp_id] of
    `folds` (or KIRBY_FIRST_FOLDS), else all of them. Prints what it kept.
    The caller keeps passing the FULL count as n_outer_folds to every search
    and namespace, so nothing stored changes its name."""
    env = os.environ.get('KIRBY_FIRST_FOLDS')
    k = int(env) if env else OUTER_FOLDS_RUN.get(exp_id)
    if not k or k >= len(folds):
        return list(folds)
    print(f"  {exp_id}: running outer folds 0..{k - 1} of the {len(folds)}-fold "
          f"layout ({'KIRBY_FIRST_FOLDS' if env else 'OUTER_FOLDS_RUN'}); "
          f"fold names and stored searches keep the {len(folds)}-fold layout")
    return list(folds)[:k]


def _top_singular_values(X_scaled: np.ndarray, n_comp: int) -> np.ndarray:
    """The n_comp largest singular values, randomised SVD in float64.

    2026-10-07: on buchwald_hartwig cgr_morgan (3955 x 2048, 0/1, every value
    finite) TruncatedSVD on the float32 array produced a non-finite
    intermediate and raised "array must not contain infs or NaNs"; the same
    call on 2000 of its rows did not. In float64 first; if it still raises,
    the top n_comp singular values of a full gesvd are the same quantity,
    computed the slow way.
    """
    from sklearn.decomposition import TruncatedSVD
    X64 = np.asarray(X_scaled, dtype=np.float64)
    try:
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd.fit(X64)
        return svd.singular_values_
    except (ValueError, np.linalg.LinAlgError):
        from scipy.linalg import svd as scipy_svd
        return scipy_svd(X64, full_matrices=False, compute_uv=False,
                         lapack_driver='gesvd')[:n_comp]


def compute_effective_rank(X: np.ndarray) -> float:
    """Effective rank: exp(entropy of normalised singular values)."""
    scaler = FeatureScaler()
    X_scaled = scaler.fit_transform(X)
    n, d = X_scaled.shape

    if d > 2000:
        from sklearn.decomposition import TruncatedSVD
        n_comp = min(500, n - 1)
        s = _top_singular_values(X_scaled, n_comp)
    else:
        try:
            s = np.linalg.svd(X_scaled, full_matrices=False, compute_uv=False)
        except np.linalg.LinAlgError:
            # gesdd LAPACK driver fails on some near-singular reps (observed
            # on abagym_Ang2). gesvd is slower but more robust.
            from scipy.linalg import svd as scipy_svd
            s = scipy_svd(X_scaled, full_matrices=False, compute_uv=False,
                          lapack_driver='gesvd')

    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def compute_dataset_complexity(reps: Dict[str, np.ndarray], labels: np.ndarray) -> Dict[str, float]:
    """Compute dataset complexity metrics (Lorena et al. 2018 inspired)."""
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    n, d = X_concat.shape
    y = labels

    metrics = {'n_samples': n, 'n_features': d}

    erank = compute_effective_rank(X_concat)
    metrics['erank'] = erank
    metrics['n_over_d'] = n / d
    metrics['n_over_erank'] = n / erank

    X_scaled = FeatureScaler().fit_transform(X_concat)
    if d > 2000:
        from sklearn.decomposition import TruncatedSVD
        n_comp = min(500, n - 1)
        s = _top_singular_values(X_scaled, n_comp)
    else:
        _, s, _ = np.linalg.svd(X_scaled, full_matrices=False)
    eigvals = s ** 2 / n
    eigvals = eigvals[eigvals > 1e-10]
    metrics['participation_ratio'] = float((eigvals.sum()) ** 2 / (eigvals ** 2).sum())

    max_features = min(d, 1000)
    if d > max_features:
        rng = np.random.RandomState(42)
        feat_idx = rng.choice(d, max_features, replace=False)
        X_sub = X_concat[:, feat_idx]
    else:
        X_sub = X_concat

    abs_corrs = []
    for j in range(X_sub.shape[1]):
        col = X_sub[:, j]
        if col.std() < 1e-10:
            abs_corrs.append(0.0)
            continue
        rho, _ = spearmanr(col, y)
        abs_corrs.append(abs(rho) if not np.isnan(rho) else 0.0)
    abs_corrs = np.array(abs_corrs)
    metrics['C1'] = float(np.max(abs_corrs))
    metrics['C2'] = float(np.mean(abs_corrs))

    X_for_model = X_scaled
    if d > 2000:
        from sklearn.decomposition import TruncatedSVD
        X_for_model = TruncatedSVD(n_components=min(500, n - 1), random_state=42).fit_transform(X_scaled)
    # L1 linearity descriptor: how far a plain linear fit is from the target.
    # LinearRegression (OLS) replaces Ridge here (2026-07-18) so ridge is gone
    # everywhere — this is a measurement of linear predictability, not a model
    # in the study.
    from sklearn.linear_model import LinearRegression
    linear = LinearRegression()
    linear.fit(X_for_model, y)
    y_pred_linear = linear.predict(X_for_model)
    y_range = y.max() - y.min()
    metrics['L1'] = float(np.mean(np.abs(y - y_pred_linear)) / y_range) if y_range > 0 else 0.0

    nn = make_nearest_neighbors(2, n_jobs=-1)
    nn.fit(X_for_model)
    _, indices = nn.kneighbors(X_for_model)
    y_nn = y[indices[:, 1]]
    metrics['S3'] = float(np.mean(np.abs(y - y_nn)) / y_range) if y_range > 0 else 0.0

    metrics['label_std'] = float(np.std(y))
    counts, _ = np.histogram(y, bins=20)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    metrics['target_entropy'] = float(-np.sum(probs * np.log(probs)))

    X_for_mle = X_scaled
    if d > 2000:
        from sklearn.decomposition import TruncatedSVD
        X_for_mle = TruncatedSVD(n_components=min(200, n - 1), random_state=42).fit_transform(X_scaled)
    try:
        metrics['mle_id'] = float(estimate_intrinsic_dim_mle(X_for_mle, k=10))
    except Exception:
        metrics['mle_id'] = None

    return metrics


# =============================================================================
# DATA LOADING (domain-aware)
# =============================================================================

_ANTIBODY_PREFIXES = ('flab_', 'abcov_', 'sabdab_', 'abagym_', 'abbind', 'ddg_',
                      'ginkgo_', 'tz_her2', 'humanness_', 'graphinity_')
_TCGA_PREFIXES = ('tcga_',)
_NUCLEOTIDE_PREFIXES = ('nt_', 'gb_')
_REACTION_DATASETS = ('buchwald_hartwig', 'suzuki', 'bh_doyle_plates')
_DTA_DATASETS = ('davis', 'kiba', 'bindingdb_kd')
_DTA_PREFIXES = ('pdbbind',)


# =============================================================================
# FRAMEWORK EXPERIMENT TASK & REP CONFIGURATION
# =============================================================================

FRAMEWORK_TASKS = {
    # Molecular (RMSE, scaffold-split)
    'logd': [
        'mhg_gnn', 'grover', 'mordred', 'mol2vec', 'ecfp4', 'molformer',
        'pdv', 'smited', 'avalon', 'chemberta', 'graph_kernel', 'maccs',
    ],
    'herg_ki': [
        'mhg_gnn', 'graph_kernel', 'avalon', 'grover', 'ecfp4', 'mordred',
        'pdv', 'molformer', 'smited', 'chemberta', 'maccs',
    ],
    # Antibody (Spearman, identity-grouped)
    'flab_thermo_tresanco': [
        'esm3', 'saprot', 'blosum', 'esmc', 'prot_t5', 'igt5', 'onehot',
        'esm2_early', 'esm2', 'mutation_features', 'ablang2', 'antiberty',
        'aac', 'structural',
    ],
    'flab_pk_jain_fcrnrt': [
        'ablang2', 'igt5', 'esm2_early', 'prot_t5', 'saprot',
        'mutation_features', 'antiberty', 'esmc', 'esm2', 'aac',
        'abroberta', 'onehot', 'structural',
    ],
    # TCGA (AUROC / F1w, TSS-grouped)
    'tcga_luad_vs_lusc': [
        'gene_geneformer_v2', 'gene_bulkrnabert', 'methyl_hvg',
        'gene_geneformer', 'gene_chromosome_arm', 'wsi_titan',
        'methyl_pca50', 'cnv_pca50', 'mirna_top200', 'wsi_gigapath',
        'rppa', 'immune_xcell', 'mut_driver', 'stemness_mrnasi',
    ],
    'tcga_brca_pam50': [
        'gene_geneformer_v2', 'gene_bulkrnabert', 'methyl_hvg',
        'gene_geneformer', 'gene_chromosome_arm', 'wsi_titan',
        'methyl_pca50', 'cnv_pca50', 'mirna_top200', 'wsi_gigapath',
        'rppa', 'immune_xcell', 'mut_driver', 'stemness_mrnasi',
    ],
    # Small antibody (Spearman, identity-grouped) — same family as flab_pk_jain_fcrnrt
    'flab_pk_jain_heprt': [
        'ablang2', 'igt5', 'esm2_early', 'prot_t5', 'saprot',
        'mutation_features', 'antiberty', 'esmc', 'esm2', 'aac',
        'abroberta', 'onehot', 'structural',
    ],
    # AbAgym DMS (Spearman, position_modulo split)
    'abagym_EGFR': [
        'esm3', 'saprot', 'blosum', 'esmc', 'prot_t5', 'igt5', 'onehot',
        'esm2_early', 'esm2', 'ablang2', 'antiberty', 'aac', 'structural',
    ],
    # Trastuzumab binding classification (AUROC/F1w, edit_distance grouped)
    'tz_her2_binary': [
        'esm3', 'saprot', 'blosum', 'esmc', 'prot_t5', 'igt5', 'onehot',
        'esm2_early', 'esm2', 'ablang2', 'antiberty', 'aac', 'structural',
    ],
    # Ginkgo developability (Spearman, pre-computed 5-fold splits, n=242)
    'ginkgo_hic': [
        'esm3', 'saprot', 'blosum', 'esmc', 'prot_t5', 'igt5', 'onehot',
        'esm2_early', 'esm2', 'ablang2', 'antiberty', 'aac', 'structural',
    ],
    # Reaction (R², leave-additive-out GroupKFold)
    'buchwald_hartwig': [
        'reactant_ecfp4', 'drfp_2048', 'onehot_components', 'rxnfp',
        'product_ecfp4', 'reactant_maccs', 'product_maccs', 'reactant_pdv',
        'product_pdv', 'diff_ecfp4', 'diff_maccs', 'component_descriptors',
        'ligand_descriptors', 'reactiont5', 'concat_ecfp4', 'concat_maccs',
        'concat_pdv',
    ],
}


def _is_antibody_dataset(name: str) -> bool:
    return any(name.lower().startswith(p) for p in _ANTIBODY_PREFIXES)


def _is_tcga_dataset(name: str) -> bool:
    return any(name.lower().startswith(p) for p in _TCGA_PREFIXES)


def _is_nucleotide_dataset(name: str) -> bool:
    return any(name.lower().startswith(p) for p in _NUCLEOTIDE_PREFIXES)


def _is_reaction_dataset(name: str) -> bool:
    return name.lower() in _REACTION_DATASETS


def _is_dta_dataset(name: str) -> bool:
    n = name.lower()
    return n in _DTA_DATASETS or any(n.startswith(p) for p in _DTA_PREFIXES)


# ── Large-n / domain experiment-scope policy (2026-06-09; user-approved) ──
# The similarity (s*) and neighbor-based noise (n*) experiments rest on the
# premise that "points close together in representation space have similar
# labels". Two rules decide where they are out of scope. The similarity/noise
# drivers (run_all) call should_skip_experiment() before each experiment and,
# on a skip, write the returned {'skipped':...} dict as the result marker so the
# wave-audit counts it as INTENTIONALLY-SKIPPED, not failed/missing. Full
# rationale + the noise neighbor-vs-model classification: REPRODUCIBILITY_LEDGER.md §E.
#
# Threshold matches s01's existing S01_MAX_DENSE_N so the family uses one cutoff;
# 15000 keeps qm9 (<=10k) and herg (7772) in, catches DTA (~30k) + graphinity (50k).
LARGE_N_SKIP_THRESHOLD = 15000

# Rule 1 (science): on NUCLEOTIDE datasets the similarity premise is unreliable
# -> skip ALL similarity + the neighbor/similarity-based noise experiments.
# (s09 is the cross-dataset aggregator, not a per-dataset experiment, so omitted.)
# KEPT on nucleotide (model/label-based, premise-independent): n06 (bindingdb-only),
# n07, n09, n13.
NUCLEOTIDE_SKIP_EXPERIMENTS = frozenset({
    's01', 's02', 's03', 's04', 's05', 's06', 's08', 's10', 's11', 's12', 's13',
    'n01', 'n02', 'n03', 'n04', 'n05', 'n08', 'n10', 'n11', 'n12', 'n14',
})

# Rule 2 (memory): these build an UNCAPPED n x n distance matrix -> OOM at large
# n. Skip on ANY dataset with n > LARGE_N_SKIP_THRESHOLD (covers non-nucleotide
# big sets — DTA, graphinity — which Rule 1 does not touch).
# n25 joins them because its exempt set is n01's cliff reference: one similarity
# value for every pair of molecules in the dataset, held as a dense matrix
# (n01._compute_ecfp4_tanimoto_sim / _compute_euclidean_sim). Same matrix, same
# ceiling. Added 2026-09-23.
LARGE_N_OOM_SKIP_EXPERIMENTS = frozenset({'s02', 's08', 'n01', 'n05', 'n10',
                                          'n14', 'n25'})


def should_skip_experiment(exp_id, data):
    """Return a {'skipped':...} marker dict if (exp_id, data) is out of scope per
    the large-n/domain policy above, else None.

    exp_id is the short id ('s02' / 'n01'); data is the loaded dataset dict
    (must carry 'domain' and 'labels'). The marker is JSON-saved by the caller so
    monitoring sees a present, intentionally-skipped result.
    """
    exp = str(exp_id).lower()
    domain = data.get('domain', 'molecular')
    try:
        n = int(len(data.get('labels', [])))
    except Exception:
        n = 0

    if domain == 'nucleotide' and exp in NUCLEOTIDE_SKIP_EXPERIMENTS:
        return {
            'skipped': True,
            'reason': (f'nucleotide domain: similarity premise unreliable for '
                       f'sequence embeddings (exp={exp})'),
            'policy': 'nucleotide_premise',
            'domain': domain,
            'n_samples': n,
        }

    if exp in LARGE_N_OOM_SKIP_EXPERIMENTS and n > LARGE_N_SKIP_THRESHOLD:
        return {
            'skipped': True,
            'reason': (f'large-n OOM guard: n={n} > {LARGE_N_SKIP_THRESHOLD} '
                       f'builds an uncapped n x n matrix (exp={exp})'),
            'policy': 'large_n_oom',
            'domain': domain,
            'n_samples': n,
        }

    return None


class NoApprovedRepPool(KeyError):
    """Raised when a dataset has no DATASET_REPS entry.

    The rep-pruning step writes DATASET_REPS. A dataset absent from it was
    never pruned, so there is no approved representation list to run against.
    Falling back to a hand-written list is what this exception exists to stop.
    """


def approved_rep_pool(dataset_name, extra=(), override=None):
    """The representations approved for `dataset_name`, plus named exceptions.

    Returns DATASET_REPS[dataset_name]['general'] + its 'augmentation' entry —
    the list the rep-pruning step signed off and the list precompute builds.
    Augmentation reps are included because the callers partition them back out
    themselves; passing only 'general' would silently drop the augmentation arm.

    `extra` names representations the experiment needs for a reason stated at
    the call site: a method the replicated paper used, or a rep a second
    evaluation arm scores directly. Each one is printed, so an addition with no
    reason behind it shows up in the log.

    Raises NoApprovedRepPool if the dataset has no DATASET_REPS entry.
    """
    from experiments.complexity.complexity_helpers import DATASET_REPS
    entry = DATASET_REPS.get(dataset_name)
    if not entry or not entry.get('general'):
        raise NoApprovedRepPool(
            f"{dataset_name} has no DATASET_REPS entry, so no approved "
            f"representation pool exists for it. Add one, or pass --reps "
            f"explicitly and say in the results why the pool was bypassed.")
    if override:
        print(f"  [reps] {dataset_name}: --reps given, "
              f"approved pool of {len(entry['general'])} bypassed")
        return list(override)
    pool = list(entry['general']) + list(entry.get('augmentation', []))
    out = list(pool)
    for rep in extra:
        if rep not in out:
            out.append(rep)
    print(f"  [reps] {dataset_name}: {len(pool)} approved "
          f"({len(entry['general'])} general + "
          f"{len(entry.get('augmentation', []))} augmentation)"
          + (f" + {len(out) - len(pool)} stated exception(s): "
             f"{[r for r in out if r not in pool]}" if len(out) > len(pool) else ""))
    return out


def _build_rep_factories(dataset_name, data):
    """Construct FT factory callables from DATASET_REPS ft_reps config.

    Returns dict mapping rep_name -> factory callable, or empty dict.
    Each factory has signature: (train_idx, val_idx, train_labels) -> (train_embs, val_embs).
    """
    try:
        from experiments.complexity.complexity_helpers import DATASET_REPS
    except ImportError:
        return {}

    ds_config = DATASET_REPS.get(dataset_name, {})
    ft_config = ds_config.get('ft_reps', {})
    if not ft_config:
        return {}

    factories = {}
    for rep_name, spec in ft_config.items():
        factory_type = spec['factory']
        base_model = spec['base_model']
        try:
            if factory_type == 'molecular_lora':
                from kirby.finetuning import make_molecular_lora_factory
                factories[rep_name] = make_molecular_lora_factory(
                    smiles_list=data['smiles'], dataset_name=dataset_name,
                    base_model=base_model)
            elif factory_type == 'molecular_contrastive':
                from kirby.finetuning import make_molecular_contrastive_factory
                factories[rep_name] = make_molecular_contrastive_factory(
                    smiles_list=data['smiles'], dataset_name=dataset_name,
                    base_model=base_model)
            elif factory_type == 'molecular_gnn':
                from kirby.finetuning import make_molecular_gnn_factory
                factories[rep_name] = make_molecular_gnn_factory(
                    smiles_list=data['smiles'], dataset_name=dataset_name,
                    base_model=base_model)
            elif factory_type == 'antibody_lora':
                from kirby.finetuning import make_antibody_lora_factory
                factories[rep_name] = make_antibody_lora_factory(
                    heavy_seqs=data['heavy_seqs'], light_seqs=data['light_seqs'],
                    dataset_name=dataset_name, base_model=base_model)
            else:
                print(f"  WARNING: unknown FT factory type '{factory_type}'")
        except Exception as e:
            print(f"  WARNING: FT factory {rep_name} failed: {e}")

    if factories:
        print(f"  [ft] {len(factories)} FT factories: {', '.join(factories)}")

    return factories


def _require_task_type(data: dict, source: str) -> str:
    """Assert data['task_type'] is set to a valid value. Raise if not.

    Every dataset loader MUST set 'task_type' to 'classification' or
    'regression'. Silent defaulting to 'regression' has caused classification
    datasets to be scored with regression metrics in the past — this guard
    turns that into a loud failure.
    """
    tt = data.get('task_type')
    if tt not in ('classification', 'regression'):
        raise ValueError(
            f"Loader for '{source}' returned task_type={tt!r}. "
            f"Every loader MUST set 'task_type' to 'classification' or "
            f"'regression' — silent defaults are not allowed."
        )
    return tt


def smoke_budgets(default=(25,)) -> list:
    """The feature budgets a smoke run searches over, from KIRBY_SMOKE_BUDGETS.

    Nine call sites across the opig files wrote the literal [25]. A single
    budget of 25 features is a real handicap against a base representation
    scored on all of its own: antiberty_cdr is 512 features wide, so a hybrid
    capped at 25 loses on width before it loses on method, and the delta cannot
    be read. This makes that one number a knob without touching nine files.

    KIRBY_SMOKE_BUDGETS is a comma-separated list, e.g. '100,200,500'. Unset or
    unparseable gives 25, which is what every one of those call sites did.
    """
    import os as _os
    raw = _os.environ.get('KIRBY_SMOKE_BUDGETS', '').strip()
    if not raw:
        return list(default)
    try:
        vals = [int(v) for v in raw.split(',') if v.strip()]
    except ValueError:
        return list(default)
    return vals or list(default)


def _smoke_subsample(data: dict) -> dict:
    """Test-only row cap, gated behind the KIRBY_SMOKE_N env var (default off).

    When KIRBY_SMOKE_N is set to a positive integer N and the dataset has more
    than N rows, deterministically keep N evenly-spaced rows (preserving class /
    scaffold diversity across the original order — NOT a random split). This
    exists purely so the experiment suite can be smoke-tested locally on tiny
    subsets; it is a no-op in any normal/server run where the env var is unset.
    """
    import os as _os
    raw = _os.environ.get('KIRBY_SMOKE_N', '').strip()
    if not raw:
        return data
    try:
        n_keep = int(raw)
    except ValueError:
        return data
    if n_keep <= 0:
        return data
    labels = data.get('labels')
    if labels is None:
        return data
    n = len(labels)
    if n <= n_keep:
        return data
    idx = np.linspace(0, n - 1, n_keep).round().astype(int)
    idx = np.unique(idx)

    def _take(v):
        if isinstance(v, np.ndarray) and len(v) == n:
            return v[idx]
        if isinstance(v, list) and len(v) == n:
            return [v[i] for i in idx]
        return v

    # Per-row top-level keys
    # 'mutations' belongs here: opig_trastuzumab_enrichment passes it into
    # compute_or_load_reps straight after this call, so under KIRBY_SMOKE_N the
    # sequence lists shrank and the mutation list did not, and the two arms were
    # built from rows that no longer lined up.
    for key in ('labels', 'groups', 'heavy_seqs', 'light_seqs', 'sequences',
                'seqs', 'smiles', 'sample_ids', 'sample_weights', 'mutations'):
        if key in data and data[key] is not None:
            data[key] = _take(data[key])
    # Rep dicts (and augmentation dict)
    for key in ('reps', 'augmentations'):
        v = data.get(key)
        if isinstance(v, dict):
            data[key] = {r: (arr[idx] if isinstance(arr, np.ndarray)
                             and len(arr) == n else arr)
                         for r, arr in v.items()}
    # Recompute group counts if present
    if 'groups' in data and isinstance(data['groups'], np.ndarray):
        n_uniq = int(len(np.unique(data['groups'])))
        if 'n_groups' in data:
            data['n_groups'] = n_uniq
        if 'n_scaffolds' in data:
            data['n_scaffolds'] = n_uniq
    print(f"  [SMOKE] KIRBY_SMOKE_N={n_keep}: subsampled {n} -> {len(idx)} rows")
    return data


# One number precision for every representation (Adelaide, 2026-10-05): float32,
# the precision the search already used and the one scikit-learn's trees and
# lightgbm work in. The noise scripts n32-n41 converted to float64, which moved
# lightgbm on mhg_gnn (herg_ki) from 0.6015 to 0.6079 rmse against every other
# experiment. Set once here, on load, and on every cache write.
REP_DTYPE = np.float32


def as_rep_dtype(arr):
    """A representation matrix in REP_DTYPE; non-float arrays pass through."""
    a = np.asarray(arr)
    if a.dtype.kind == 'f' and a.dtype != REP_DTYPE:
        return a.astype(REP_DTYPE)
    return a


def _finalize_load_data(data: dict, dataset_name: str) -> dict:
    """Post-process a load_data result: inject dataset_name, cross-check registry.

    - Adds `data['dataset_name']` for downstream consumers.
    - If the dataset is in DATASET_REGISTRY, verifies that the loader's
      task_type matches the registry entry, and backfills `data['domain']`
      if the loader didn't set it. Mismatches raise loudly — a silent drift
      between loader and registry is exactly the class of bug this registry
      exists to prevent.
    """
    data['dataset_name'] = dataset_name
    for _k in ('reps', 'augmentations'):
        if isinstance(data.get(_k), dict):
            data[_k] = {r: as_rep_dtype(a) for r, a in data[_k].items()}
    info = registry_lookup(dataset_name)
    if info is not None:
        loader_task = data.get('task_type')
        if loader_task != info.task_type:
            raise ValueError(
                f"Registry/loader mismatch for '{dataset_name}': "
                f"loader returned task_type={loader_task!r}, but "
                f"DATASET_REGISTRY says {info.task_type!r}. "
                f"Fix one side — they must agree."
            )
        if 'domain' not in data or data['domain'] is None:
            data['domain'] = info.domain
        elif data['domain'] != info.domain:
            # Loader set a domain that disagrees — warn once, registry wins.
            import warnings
            warnings.warn(
                f"Registry/loader domain mismatch for '{dataset_name}': "
                f"loader set domain={data['domain']!r}, registry says "
                f"{info.domain!r}. Using registry value."
            )
            data['domain'] = info.domain

    # Row-count guard: every representation must have exactly one row per label.
    # A stale on-disk rep cache with the wrong length (e.g. from an older loader
    # version) would otherwise be silently reused and misalign features from
    # their labels — the exact silent-corruption class that has cost real time
    # here. Raise loudly (non-destructive) naming the offending rep(s) and the
    # purge command, rather than guessing or recomputing automatically. Checked
    # BEFORE _smoke_subsample so it sees the full, pre-subsample arrays.
    _reps = data.get('reps')
    _labels = data.get('labels')
    if _reps and _labels is not None:
        _n_labels = len(_labels)
        _bad = {rn: int(arr.shape[0]) for rn, arr in _reps.items()
                if hasattr(arr, 'shape') and arr.ndim >= 1
                and arr.shape[0] != _n_labels}
        if _bad:
            raise ValueError(
                f"Row-count mismatch in '{dataset_name}': {_n_labels} labels but "
                f"representation row counts {_bad}. This is almost always a stale "
                f"rep cache from an older loader version being silently reused. "
                f"Purge it and re-run: "
                f"python tests/purge_rep_cache.py {dataset_name} --delete"
            )

    return _smoke_subsample(data)


def load_data(dataset_name: str, rep_preset: str = 'fast',
              delta: bool = False, augment: bool = False,
              rep_names: Optional[List[str]] = None,
              finetune: bool = False) -> dict:
    """Public entry point — loads data and stamps it with registry metadata.

    Thin wrapper around `_load_data_impl` that:
    - Injects `data['dataset_name']`.
    - Cross-checks `data['task_type']` against DATASET_REGISTRY.
    - Backfills `data['domain']` from registry when missing.
    """
    data = _load_data_impl(
        dataset_name, rep_preset=rep_preset, delta=delta, augment=augment,
        rep_names=rep_names, finetune=finetune)
    # KIRBY_LOCAL_MAX_FEATURES_PER_REP: one place, so every experiment that
    # loads through here sees the same matrices. Unset, this is a no-op. The
    # columns kept are the highest-variance ones, chosen without the labels, so
    # it is not supervised selection; no row is dropped.
    try:
        from experiments._local_grid import cap_rep_width
        if data.get('reps'):
            data['reps'] = cap_rep_width(data['reps'])
    except ImportError:
        pass
    return _finalize_load_data(data, dataset_name)


def DATASET_REPS_FOR_LHASA():
    """lhasa_herg's representation pool, from DATASET_REPS and nowhere else."""
    from experiments.complexity.complexity_helpers import DATASET_REPS
    return list(DATASET_REPS['lhasa_herg']['general'])


def _load_data_impl(dataset_name: str, rep_preset: str = 'fast',
                    delta: bool = False, augment: bool = False,
                    rep_names: Optional[List[str]] = None,
                    finetune: bool = False) -> dict:
    """Load dataset, generate representations, assign groups.

    Domain-aware: detects TCGA, antibody, or molecular dataset names and
    dispatches accordingly.
    Returns dict with keys: labels, reps, groups, n_groups, task_type,
    augmentations, plus domain-specific keys.

    When rep_names is None, auto-filters to the 'general' reps from
    DATASET_REPS (if the dataset has an entry).  Pass rep_names explicitly
    to override.

    **Do not call directly** — use `load_data()` so registry cross-checks
    and `dataset_name` injection happen.
    """
    # An explicit comma-separated --reps IS an explicit list. Before 2026-09-28
    # it went into rep_preset and the auto-filter below replaced it with the
    # DATASET_REPS general list, so `--reps a,b,c` was silently ignored on every
    # dataset DATASET_REPS names. Named presets ('fast', 'all', ...) carry no
    # comma and are unaffected.
    if rep_names is None and isinstance(rep_preset, str) and ',' in rep_preset:
        rep_names = [r.strip() for r in rep_preset.split(',') if r.strip()]

    # Auto-filter to DATASET_REPS general reps when no explicit list given
    if rep_names is None:
        try:
            from experiments.complexity.complexity_helpers import DATASET_REPS
            ds_reps = DATASET_REPS.get(dataset_name)
            if ds_reps is not None and ds_reps.get('general'):
                rep_names = ds_reps['general']
                from experiments._local_grid import cut_pool
                rep_names = cut_pool(dataset_name, rep_names)
                print(f"  [auto] {dataset_name}: filtering to "
                      f"{len(rep_names)} general reps from DATASET_REPS")
        except ImportError:
            pass

    # lhasa_herg is loaded by the federated suite, which built its own path to
    # it before load_data had one. Owner's decision 2026-09-17: it uses the same
    # representations as the other hERG sets, so it should be precomputable like
    # them -- and precompute_hybrids.py reaches every dataset through load_data.
    #
    # federated_helpers.load_lhasa_herg_data already returns a load_data-shaped
    # dict (reps, labels, groups, task_type, domain), so this delegates rather
    # than duplicating the SDF reading and the scaffold grouping.
    #
    # ⛔ IT HAS NO AUGMENTATION REPRESENTATIONS. That helper returns
    # augmentations=None, so only the 'none' condition exists for it -- which is
    # also all _pool_conditions asks for when a caller passes no augmentations,
    # so the two sides agree. Adding the augmented conditions means teaching that
    # helper to build them, not adding names to DATASET_REPS.
    if dataset_name == 'lhasa_herg':
        from experiments.federated.lhasa_herg_loader import load_lhasa_herg_training
        from find_best_hybrid_molecular import (
            get_representations as _get_mol_reps,
            assign_scaffold_groups as _assign_scaffold_groups,
        )
        smiles, labels, _inchikeys, _ = load_lhasa_herg_training()
        _names = rep_names or DATASET_REPS_FOR_LHASA()
        # ⛔ THROUGH THE REPRESENTATION CACHE, unlike
        # federated_helpers.load_lhasa_herg_data, which calls the builder
        # directly and so recomputes all ten every run. That is affordable once
        # inside a federated experiment and not affordable at all in a
        # precompute, where five conditions would each rebuild mordred, grover
        # and molformer over the whole set. The federated path is left alone.
        reps = _cache_rep_dict(
            dataset_name,
            lambda: _get_mol_reps(smiles, ','.join(_names)),
            filter_names=_names,
            compute_subset_fn=lambda names: _get_mol_reps(smiles, ','.join(names)),
            identity=list(smiles),
        )
        try:
            groups, _ = _assign_scaffold_groups(smiles)
        except Exception:
            groups = None
        data = {
            'reps': reps, 'labels': labels, 'smiles': smiles, 'groups': groups,
            'task_type': 'classification', 'domain': 'molecular',
            'is_binary': True, 'augmentations': None,
        }
        return _finalize_load_data(data, dataset_name)

    if _is_tcga_dataset(dataset_name):
        from find_best_hybrid_tcga import (
            load_dataset as load_tcga_dataset,
            get_representations as get_tcga_representations,
            assign_tss_groups,
        )

        expression, labels, task_type, metadata = load_tcga_dataset(dataset_name)
        sample_ids = metadata['sample_ids']
        gene_names = metadata['gene_names']
        wsi_dir = metadata.get('wsi_dir')
        cohort = metadata.get('cohort', '')

        reps = _cache_rep_dict(
            dataset_name,
            lambda: get_tcga_representations(
                expression, gene_names, sample_ids, wsi_dir,
                rep_preset, cohort=cohort,
            ),
            filter_names=rep_names,
            compute_subset_fn=lambda names: get_tcga_representations(
                expression, gene_names, sample_ids, wsi_dir,
                ','.join(names), cohort=cohort,
            ),
            identity=list(sample_ids),
        )

        tss_groups, n_groups = assign_tss_groups(sample_ids)

        print(f"  {dataset_name}: {len(sample_ids)} samples, {n_groups} TSS sites, "
              f"{len(reps)} reps ({', '.join(f'{k}={v.shape[1]}' for k, v in reps.items())})")

        return {
            'expression': expression,
            'labels': labels,
            'reps': reps,
            'groups': tss_groups,
            'n_groups': n_groups,
            'n_scaffolds': n_groups,
            'task_type': task_type,
            'domain': 'tcga',
            'augmentations': None,
            'metadata': metadata,
            'rep_factories': {},
        }

    elif _is_antibody_dataset(dataset_name):
        from antibody_baseline_survey import load_dataset as load_ab_dataset
        from kirby.representations.antibody import ANTIBODY_REPS, ANTIBODY_FAST_REPS
        from kirby.representations.mutation import MUTATION_REPS

        data = load_ab_dataset(dataset_name)
        heavy_seqs = data['heavy_seqs']
        light_seqs = data['light_seqs']
        labels = data['labels']
        groups = data['groups']
        n_groups = data['n_groups']
        mutations = data.get('mutations')

        # Generate reps based on preset or custom list
        if rep_preset in ('fast', 'skip_lm'):
            _rep_names_from_preset = sorted(ANTIBODY_FAST_REPS)
        else:
            _rep_names_from_preset = sorted(
                set(ANTIBODY_REPS.keys()) | set(MUTATION_REPS.keys())
            )

        _wanted_reps = rep_names if rep_names else _rep_names_from_preset
        reps = {}
        _failed_reps = []  # (rep, error) for REQUESTED reps that raised on load

        # Pass 1: standard antibody reps (heavy_seqs, light_seqs)
        for rn in _wanted_reps:
            if rn in ANTIBODY_REPS:
                try:
                    fn = ANTIBODY_REPS[rn]
                    reps[rn] = _cache_rep(dataset_name, rn,
                                          lambda _fn=fn: _fn(heavy_seqs, light_seqs),
                                          identity=_ab_row_identity(heavy_seqs, light_seqs))
                except Exception as e:
                    print(f"  WARNING: {rn} failed: {e}")
                    _failed_reps.append((rn, str(e)))
            elif rn not in MUTATION_REPS:
                print(f"  WARNING: antibody rep '{rn}' not found, skipping")

        # Pass 2: mutation-aware reps (require mutations metadata)
        _has_mutations = mutations and any(len(m) > 0 for m in mutations)
        if _has_mutations:
            for rn in _wanted_reps:
                if rn in MUTATION_REPS and rn not in reps:
                    try:
                        fn = MUTATION_REPS[rn]
                        reps[rn] = _cache_rep(
                            dataset_name, rn,
                            lambda _fn=fn: _fn(heavy_seqs, light_seqs, mutations),
                            identity=_ab_row_identity(heavy_seqs, light_seqs))
                    except Exception as e:
                        print(f"  WARNING: {rn} failed: {e}")
                        _failed_reps.append((rn, str(e)))

        # A2: a requested rep that failed to load is dropped — WARN, do not halt
        # (user preference 2026-06-09). NOTE: a dropped rep changes rep_names in
        # the cache key and can trigger a cache rebuild for this dataset; pin
        # network/HF reps offline to avoid intermittent drops.
        if _failed_reps:
            print(f"  WARNING: {dataset_name}: DROPPED {len(_failed_reps)} "
                  f"requested rep(s) that failed to load: "
                  f"{[r for r, _ in _failed_reps]} — rep_names changed, may "
                  f"rebuild this dataset's cache. First error: {_failed_reps[0][1]}")

        print(f"  {dataset_name}: {len(heavy_seqs)} antibodies, {n_groups} groups, "
              f"{len(reps)} reps ({', '.join(f'{k}={v.shape[1]}' for k, v in reps.items())})")

        result = {
            'heavy_seqs': heavy_seqs,
            'light_seqs': light_seqs,
            'labels': labels,
            'reps': reps,
            'groups': groups,
            'n_scaffolds': n_groups,
            'n_groups': n_groups,
            'task_type': _require_task_type(data, dataset_name),
            'domain': 'antibody',
            'augmentations': None,
            # What the folds hold out, in words, carried from the survey
            # loader. On the Jain panel, tresanco and marks the 95%-identity
            # clustering leaves one antibody per group, so the folds hold out
            # antibodies rather than homology families — a reader of the result
            # file cannot tell that from the strategy name alone.
            'group_strategy': data.get('group_strategy'),
            'group_scheme': data.get('group_scheme'),
        }
        result['rep_factories'] = _build_rep_factories(dataset_name, result) if finetune else {}
        return result
    elif _is_nucleotide_dataset(dataset_name):
        from find_best_hybrid_nucleotide import (
            load_dataset as load_nuc_dataset,
            get_representations as get_nuc_representations,
        )

        data = load_nuc_dataset(dataset_name)
        sequences = data['sequences']
        labels = data['labels']
        groups = data['groups']
        n_groups = data['n_groups']
        task_type = data['task_type']

        # Nucleotide rep generation: when explicit rep_names are given
        # (precompute / experiments), pass them as a comma-joined list so
        # foundation-model reps (mistral_dna, hyenadna, nt_v2_*) and
        # classical augmentation reps both get generated. Without this,
        # the default 'fast' preset skips foundation models entirely and
        # the precompute never sees the reps DATASET_REPS asks for.
        if rep_names:
            _nuc_preset = ','.join(rep_names)
        else:
            _nuc_preset = rep_preset
        reps = _cache_rep_dict(
            dataset_name,
            lambda: get_nuc_representations(sequences, _nuc_preset),
            filter_names=rep_names,
            compute_subset_fn=lambda names: get_nuc_representations(
                sequences, ','.join(names)),
            identity=list(sequences),
        )

        print(f"  {dataset_name}: {len(sequences)} sequences, {n_groups} groups, "
              f"{len(reps)} reps ({', '.join(f'{k}={v.shape[1]}' for k, v in reps.items())})")

        return {
            'sequences': sequences,
            'labels': labels,
            'reps': reps,
            'groups': groups,
            'n_groups': n_groups,
            'n_scaffolds': n_groups,
            'task_type': task_type,
            'domain': 'nucleotide',
            'augmentations': None,
            'rep_factories': {},
        }

    elif _is_reaction_dataset(dataset_name):
        from kirby.datasets.reaction_yields import (
            load_buchwald_hartwig, load_suzuki, load_bh_doyle_plates,
        )
        from kirby.representations.reaction import (
            create_drfp, create_concat_ecfp4, create_concat_maccs,
            create_concat_pdv, create_product_ecfp4, create_reactant_ecfp4,
            create_product_maccs, create_reactant_maccs,
            create_product_pdv, create_reactant_pdv,
            create_rxnfp, create_onehot_components,
            create_diff_ecfp4, create_diff_maccs,
            create_component_descriptors, create_ligand_descriptors,
            create_reactiont5,
            create_cgr_morgan,
            create_chytorch_graphormer,
            create_cgr_smiles_lm,
            create_reaction_rep_variant,
        )
        from kirby.representations.forward_compat_reaction import (
            create_rxngraphormer, create_reamvp, create_egret,
            create_unimol3, create_diff_chemeleon, create_concat_chemeleon,
        )
        from kirby.representations.reaction import _MOL_REP_VARIANTS

        loaders = {
            'buchwald_hartwig': load_buchwald_hartwig,
            'suzuki': load_suzuki,
            'bh_doyle_plates': load_bh_doyle_plates,
        }
        data = loaders[dataset_name]()
        rxn_smiles = data['rxn_smiles']
        labels = np.array(data['labels'], dtype=np.float32)
        groups = np.array(data['groups'])
        components = data.get('components')

        # Registry of all reaction rep generators
        _rxn_generators = {
            # c13 forward-compatibility candidates.
            # ⛔ egret: the released checkpoint is the Buchwald-Hartwig-FINETUNED
            # yield head, so embedding these datasets with it is training on the
            # labels. create_egret refuses unless allow_finetuned_checkpoint=True.
            'rxngraphormer': lambda: create_rxngraphormer(rxn_smiles),
            'reamvp': lambda: create_reamvp(rxn_smiles),
            'egret': lambda: create_egret(rxn_smiles),
            'unimol3': lambda: create_unimol3(rxn_smiles),
            'diff_chemeleon': lambda: create_diff_chemeleon(rxn_smiles),
            'concat_chemeleon': lambda: create_concat_chemeleon(rxn_smiles),
            'drfp_2048': lambda: create_drfp(rxn_smiles),
            'concat_ecfp4': lambda: create_concat_ecfp4(components),
            'concat_maccs': lambda: create_concat_maccs(components),
            'concat_pdv': lambda: create_concat_pdv(components),
            'product_ecfp4': lambda: create_product_ecfp4(rxn_smiles),
            'reactant_ecfp4': lambda: create_reactant_ecfp4(rxn_smiles),
            'product_maccs': lambda: create_product_maccs(rxn_smiles),
            'reactant_maccs': lambda: create_reactant_maccs(rxn_smiles),
            'product_pdv': lambda: create_product_pdv(rxn_smiles),
            'reactant_pdv': lambda: create_reactant_pdv(rxn_smiles),
            'rxnfp': lambda: create_rxnfp(rxn_smiles),
            'onehot_components': lambda: create_onehot_components(components),
            'diff_ecfp4': lambda: create_diff_ecfp4(rxn_smiles),
            'diff_maccs': lambda: create_diff_maccs(rxn_smiles),
            'component_descriptors': lambda: create_component_descriptors(components),
            'ligand_descriptors': lambda: create_ligand_descriptors(components),
            'reactiont5': lambda: create_reactiont5(rxn_smiles),
            'cgr_morgan': lambda: create_cgr_morgan(rxn_smiles),
            'chytorch_graphormer': lambda: create_chytorch_graphormer(rxn_smiles),
            'cgr_smiles_lm': lambda: create_cgr_smiles_lm(rxn_smiles),
        }

        # Determine which reps to generate
        if rep_names:
            wanted = rep_names
        elif rep_preset == 'all':
            wanted = list(_rxn_generators.keys())
        else:
            # fast preset: the original 8
            wanted = ['drfp_2048', 'concat_ecfp4', 'concat_maccs', 'concat_pdv',
                       'product_ecfp4', 'reactant_ecfp4', 'rxnfp', 'onehot_components']

        def _variant_generator(rep_name):
            """Fallback for reactant_/product_/diff_<molrep> reps not in the
            explicit registry. Returns a zero-arg generator or None. The
            molecular-rep variants (reactant_avalon, diff_mhg_gnn, …) are
            applicable to any reaction dataset with rxn_smiles; the ecfp4/maccs/
            pdv variants already have dedicated entries above and never reach
            here. Matrices for these are typically precomputed + cached."""
            for _role in ('reactant', 'product', 'diff'):
                _prefix = _role + '_'
                if rep_name.startswith(_prefix):
                    _molrep = rep_name[len(_prefix):]
                    if _molrep in _MOL_REP_VARIANTS:
                        return lambda: create_reaction_rep_variant(
                            rxn_smiles, _molrep, _role)
            return None

        reps = {}
        _failed_reps = []  # (rep, error) for REQUESTED reps that raised on load
        for name in wanted:
            gen = _rxn_generators.get(name) or _variant_generator(name)
            if gen is None:
                print(f"  WARNING: unknown reaction rep '{name}', skipping")
                continue
            try:
                reps[name] = _cache_rep(dataset_name, name,
                                        lambda _g=gen: _g(),
                                        identity=list(rxn_smiles))
            except Exception as e:
                print(f"  WARNING: {name} failed: {e}")
                _failed_reps.append((name, str(e)))

        # A2: dropped reps -> WARN, do not halt (user preference). A dropped rep
        # changes rep_names and may rebuild this dataset's cache.
        if _failed_reps:
            print(f"  WARNING: {dataset_name}: DROPPED {len(_failed_reps)} "
                  f"requested reaction rep(s): {[r for r, _ in _failed_reps]} — "
                  f"rep_names changed, may rebuild. First error: {_failed_reps[0][1]}")

        n_groups = len(np.unique(groups))
        print(f"  {dataset_name}: {len(labels)} reactions, {n_groups} groups, "
              f"{len(reps)} reps ({', '.join(f'{k}={v.shape[1]}' for k, v in reps.items())})")

        return {
            'labels': labels,
            'reps': reps,
            'groups': groups,
            'n_groups': n_groups,
            'n_scaffolds': n_groups,
            'task_type': 'regression',
            'domain': 'reaction',
            'augmentations': None,
            'rep_factories': {},
            # The reaction SMILES and the parsed components the reps were built
            # from. Carried through so an experiment that needs a structural
            # reference can compute one from the chemistry instead of from the
            # representations it is grading — n01's cliff reference did the
            # latter, which made the cliffs a function of whichever rep
            # contributed the most features.
            'rxn_smiles': rxn_smiles,
            'components': components,
        }

    elif _is_dta_dataset(dataset_name):
        # Lazy import to avoid pulling DTA deps for non-DTA runs.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from find_best_hybrid_dta import (
            load_dataset as load_dta_dataset,
            get_representations as get_dta_representations,
        )
        from kirby.datasets.dta import (
            assign_cold_target_groups, assign_cold_drug_groups,
        )

        data = load_dta_dataset(dataset_name)
        drug_smiles = data['drug_smiles']
        protein_seqs = data['protein_seqs']
        labels = np.asarray(data['labels'], dtype=np.float64)
        metadata = data['metadata']

        # Subsample to 30K pairs to match the DATASET_REPS comment
        # ("n=30,000") and the dta_baseline_survey convention. Cold-protein
        # split is the default — same as dta_baseline_survey.
        #
        # WHOLE PAIRS. The draw picks (drug, protein) PAIRS and takes every row
        # of the ones it picks. Until 2026-09-12 it drew ROWS, and BindingDB
        # holds the same pair measured by more than one lab: a pair measured
        # three times survived only if all three of its rows happened to be
        # drawn, and one left holding a single row became, to anything reading
        # the data, a pair measured once. Of the 3,494 pairs measured more than
        # once in the cached file, the row draw left 1,505 whole; drawing pairs
        # leaves 2,017 over 5,480 rows. n06 reads the spread of those repeat
        # measurements as its ground truth, so the row draw was deleting half
        # of what it measures. Nothing else changes: every pair is still drawn
        # at random, so the mix of repeated and once-measured pairs is the
        # file's own.
        DTA_MAX_PAIRS = 30_000
        if len(labels) > DTA_MAX_PAIRS:
            rng = np.random.RandomState(42)
            _rows_of_pair = {}
            for _i, (_d, _p) in enumerate(zip(metadata['drug_ids'],
                                              metadata['protein_ids'])):
                _rows_of_pair.setdefault(f'{_d}|{_p}', []).append(_i)
            _pairs = sorted(_rows_of_pair)
            _idx, _n = [], 0
            for _k in rng.permutation(len(_pairs)):
                _rows = _rows_of_pair[_pairs[_k]]
                if _n + len(_rows) > DTA_MAX_PAIRS:
                    continue
                _idx.extend(_rows)
                _n += len(_rows)
            idx = np.sort(np.array(_idx, dtype=int))
            drug_smiles = [drug_smiles[i] for i in idx]
            protein_seqs = [protein_seqs[i] for i in idx]
            labels = labels[idx]
            metadata['drug_ids'] = [metadata['drug_ids'][i] for i in idx]
            metadata['protein_ids'] = [metadata['protein_ids'][i] for i in idx]
            metadata['n_drugs'] = len(set(metadata['drug_ids']))
            metadata['n_proteins'] = len(set(metadata['protein_ids']))
            _n_repeated = sum(1 for v in _rows_of_pair.values() if len(v) > 1
                              and v[0] in set(idx.tolist()))
            print(f"  Subsampled {dataset_name} to {len(labels)} rows, drawn "
                  f"as whole (drug, protein) pairs: {len(set(zip(metadata['drug_ids'], metadata['protein_ids'])))} "
                  f"pairs, {_n_repeated} of them measured more than once")

        groups, n_groups = assign_cold_target_groups(metadata['protein_ids'])
        print(f"  {dataset_name}: {len(labels)} pairs, "
              f"{metadata['n_drugs']} drugs × {metadata['n_proteins']} proteins, "
              f"{n_groups} cold-target groups")

        # DTA reps: drug + protein. get_representations handles the dispatch.
        if rep_names:
            preset_or_list = ','.join(rep_names)
        else:
            preset_or_list = rep_preset
        reps = get_dta_representations(
            drug_smiles, protein_seqs, preset_or_list,
            cache=True, protein_ids=metadata.get('protein_ids'))

        if rep_names:
            reps = {k: v for k, v in reps.items() if k in rep_names}

        print(f"  {dataset_name}: {len(labels)} samples, {n_groups} groups, "
              f"{len(reps)} reps "
              f"({', '.join(f'{k}={v.shape[1]}' for k, v in reps.items())})")

        return {
            'drug_smiles': drug_smiles,
            'protein_seqs': protein_seqs,
            'labels': labels,
            'reps': reps,
            'groups': groups,
            'n_groups': n_groups,
            'n_scaffolds': n_groups,
            'task_type': data.get('task_type', 'regression'),
            'domain': 'dta',
            'augmentations': None,
            'rep_factories': {},
            'metadata': metadata,
        }

    elif dataset_name == 'cdr_flexibility':
        # ITsFlexible CDR-flexibility (OPIG). Has its own raw loader (not the
        # antibody survey loader). SPLIT = option A (user 2026-07-19): cluster on
        # the CDR3 LOOP sequences at 80% identity, NOT the whole Fv — antibody
        # frameworks are ~98% conserved, so whole-Fv clustering collapses to a
        # near-random singleton split (see opig_cdr_flexibility.py:229-246).
        from kirby.datasets.itsflexible import load_itsflexible
        from kirby.representations.antibody import ANTIBODY_REPS, ANTIBODY_FAST_REPS
        from kirby.representations.mutation import MUTATION_REPS

        data = load_itsflexible(mode='cdr', cdr_types=['CDRH3', 'CDRL3'])
        heavy_seqs = data['heavy_seqs']
        light_seqs = data['light_seqs']
        labels = np.asarray(data['labels'])
        cdr_seqs = data['cdr_seqs']

        # Option A: group by CDR3-loop sequence identity (80%). The min_groups=3
        # hash-fallback guarantees a valid GroupKFold even if too few loops merge.
        groups, n_groups = sequence_identity_clusters(
            cdr_seqs, identity_threshold=0.80, min_groups=3)

        if rep_preset in ('fast', 'skip_lm'):
            _rep_names_from_preset = sorted(ANTIBODY_FAST_REPS)
        else:
            _rep_names_from_preset = sorted(ANTIBODY_REPS.keys())
        _wanted_reps = rep_names if rep_names else _rep_names_from_preset

        reps = {}
        _failed_reps = []
        for rn in _wanted_reps:
            if rn in ANTIBODY_REPS:
                try:
                    fn = ANTIBODY_REPS[rn]
                    reps[rn] = _cache_rep(dataset_name, rn,
                                          lambda _fn=fn: _fn(heavy_seqs, light_seqs),
                                          identity=_ab_row_identity(heavy_seqs, light_seqs))
                except Exception as e:
                    print(f"  WARNING: {rn} failed: {e}")
                    _failed_reps.append((rn, str(e)))
            elif rn not in MUTATION_REPS:
                print(f"  WARNING: antibody rep '{rn}' not found, skipping")
        if _failed_reps:
            print(f"  WARNING: {dataset_name}: DROPPED {len(_failed_reps)} "
                  f"requested rep(s): {[r for r, _ in _failed_reps]} — rep_names "
                  f"changed, may rebuild this dataset's cache.")

        # Split-quality guard: a near-all-singleton split is a random split in
        # disguise. Report the structure so the validity number can be trusted
        # (or rejected). NOT a hard gate — the user judges the number.
        _sizes = np.unique(groups, return_counts=True)[1]
        print(f"  {dataset_name}: {len(heavy_seqs)} CDRs, {n_groups} identity "
              f"groups (largest={int(_sizes.max())}, "
              f"singletons={int((_sizes == 1).sum())}/{n_groups}), {len(reps)} reps")

        result = {
            'heavy_seqs': heavy_seqs,
            'light_seqs': light_seqs,
            'labels': labels,
            'reps': reps,
            'groups': groups,
            'n_scaffolds': n_groups,
            'n_groups': n_groups,
            'task_type': _require_task_type(data, dataset_name),
            'domain': 'antibody',
            'augmentations': None,
        }
        result['rep_factories'] = _build_rep_factories(dataset_name, result) if finetune else {}
        return result

    else:
        from find_best_hybrid_molecular import (
            load_dataset as load_molecular_dataset,
            get_representations as get_molecular_representations,
            assign_scaffold_groups,
        )

        from find_best_hybrid_molecular import (
            FACTORY_REPS as _MOL_FACTORY_REPS,
            get_rep_factories as _get_mol_rep_factories,
        )

        smiles, labels, task_type = load_molecular_dataset(dataset_name)

        # A dataset that keeps one row per MEASUREMENT hands back a grouping that
        # says which rows are the same molecule. herg_ic50 is the only one today.
        # Two things have to happen when it is present, and both are here rather
        # than left to each experiment to remember.
        import find_best_hybrid_molecular as _fbhm
        _rep_groups = getattr(_fbhm, '_LAST_REPLICATE_GROUPS', None)
        _fbhm._LAST_REPLICATE_GROUPS = None   # never leak into the next load
        # Which ChEMBL document and assay each row came from (herg_ic50 only).
        # Same stash, same reset, same length check as the replicate groups.
        _rec_docs = getattr(_fbhm, '_LAST_RECORD_DOCUMENTS', None)
        _rec_assays = getattr(_fbhm, '_LAST_RECORD_ASSAYS', None)
        _fbhm._LAST_RECORD_DOCUMENTS = None
        _fbhm._LAST_RECORD_ASSAYS = None
        for _nm, _arr in (('document', _rec_docs), ('assay', _rec_assays)):
            if _arr is not None and len(_arr) != len(labels):
                raise RuntimeError(
                    f"{dataset_name}: the loader returned {len(_arr)} {_nm} ids "
                    f"for {len(labels)} labels, so the ids are not on the rows "
                    f"of the measurements they name.")
        if _rep_groups is not None and len(_rep_groups) != len(labels):
            raise RuntimeError(
                f"{dataset_name}: the loader returned {len(_rep_groups)} replicate "
                f"group ids for {len(labels)} labels. They index the same rows, so "
                f"a mismatch means one of the two was filtered and the other was "
                f"not, and every row after the first gap would be mislabelled.")

        groups, n_scaffolds = assign_scaffold_groups(smiles)
        if _rep_groups is not None:
            # The fold geometry STAYS the scaffold one, and the molecule grouping
            # is handed out beside it rather than in place of it.
            #
            # This block used to replace `groups` with the molecule ids, on the
            # reasoning that a scaffold split would leave one measurement of a
            # molecule in training and another held out. That does not happen.
            # assign_scaffold_groups maps a SMILES to its Murcko scaffold and then
            # to an id (find_best_hybrid_molecular.py:78-84), so every row of one
            # molecule carries one SMILES, one scaffold and one id. Scaffold
            # grouping is COARSER than molecule grouping, so it already keeps a
            # molecule's repeats together and additionally keeps its whole
            # scaffold family together. Measured on 7 rows over 4 molecules: 4
            # molecule groups, 2 scaffold groups, every molecule inside one of
            # them.
            #
            # Swapping the two therefore dropped the scaffold rule and bought
            # nothing. The assertion below is what makes that a checked property
            # of the data rather than a paragraph.
            _rg = np.asarray(_rep_groups)
            for _m in np.unique(_rg):
                _sc = np.unique(groups[_rg == _m])
                if len(_sc) != 1:
                    raise RuntimeError(
                        f"{dataset_name}: molecule {_m!r} spans {len(_sc)} "
                        f"scaffold groups, so a scaffold-grouped fold would put "
                        f"two of its measurements on opposite sides of a split. "
                        f"Every row of one molecule must carry the same SMILES.")
            print(f"  {dataset_name}: {len(labels)} measurements over "
                  f"{len(np.unique(_rg))} molecules in {n_scaffolds} scaffold "
                  f"groups; folds are scaffold-grouped and every molecule's "
                  f"repeats sit in one fold")

        # The eleven sim_* similarity recipes, PCA and the pdv/sns reps are
        # never returned by get_representations: each one fits on the training
        # rows of a fold, so it has no whole-dataset matrix to cache. Asking for
        # one used to leave it silently absent — a run over the recipes came
        # back holding only the general representations, under the recipes'
        # name. Build them as per-fold factories instead and hand them to the
        # caller separately.
        _requested = list(rep_names) if rep_names else None
        if _requested is not None:
            _factory_names = [r for r in _requested if r in _MOL_FACTORY_REPS]
            _matrix_names = [r for r in _requested if r not in _MOL_FACTORY_REPS]
        else:
            _factory_names, _matrix_names = [], None

        reps = {}
        if _matrix_names is None or _matrix_names:
            reps = _cache_rep_dict(
                dataset_name,
                lambda: get_molecular_representations(smiles, rep_preset),
                filter_names=_matrix_names,
                compute_subset_fn=lambda names: get_molecular_representations(
                    smiles, ','.join(names)),
                identity=list(smiles),
            )

        fold_factories = {}
        if _factory_names:
            fold_factories = _get_mol_rep_factories(
                smiles, ','.join(_factory_names), dataset_name=dataset_name)
            _missing = [r for r in _factory_names if r not in fold_factories]
            if _missing:
                # get_rep_factories prints and swallows a factory that raised.
                # Refuse rather than hand back a short list under the name the
                # caller asked for.
                raise RuntimeError(
                    f"{dataset_name}: {len(_missing)} per-fold representation(s) "
                    f"{_missing} were requested but could not be built. They "
                    f"have no whole-dataset matrix, so a run without them is "
                    f"not the run that was asked for.")

        print(f"  {dataset_name}: {len(smiles)} samples, {n_scaffolds} scaffolds, "
              f"{len(reps)} reps ({', '.join(f'{k}={v.shape[1]}' for k, v in reps.items())})")
        if fold_factories:
            print(f"    + {len(fold_factories)} per-fold representation(s): "
                  f"{', '.join(sorted(fold_factories))}")

        result = {
            'smiles': smiles,
            'labels': labels,
            'reps': reps,
            'groups': groups,
            'n_scaffolds': n_scaffolds,
            'task_type': task_type,
            'domain': 'molecular',
            'augmentations': None,
        }
        if _rep_groups is not None:
            # Which rows are the same molecule. This is NOT the fold geometry —
            # `groups` above is, and it is the scaffold one. n17 and n19 read
            # this to collapse the measurement rows into one row per molecule.
            result['replicate_groups'] = np.asarray(_rep_groups)
        if _rec_docs is not None:
            # Which ChEMBL document each measurement row came from. n29 reads it.
            result['record_documents'] = np.asarray(_rec_docs)
        if _rec_assays is not None:
            result['record_assays'] = np.asarray(_rec_assays)
        result['rep_factories'] = _build_rep_factories(dataset_name, result) if finetune else {}
        _clash = sorted(set(fold_factories) & set(result['rep_factories']))
        if _clash:
            raise RuntimeError(
                f"{dataset_name}: {_clash} is both a per-fold representation and "
                f"a fine-tuning factory; one would overwrite the other.")
        result['rep_factories'].update(fold_factories)
        # Named separately from the fine-tuning factories so a caller running
        # nested CV can turn exactly these into outer_rep_builders.
        result['fold_fitted_reps'] = sorted(fold_factories)
        return result


# =============================================================================
# SIMILARITY-SPECIFIC HELPERS
# =============================================================================

def get_hybrid_features_and_weights(
    reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    weight_method: str = 'lightgbm_gain',
    budget: Optional[int] = None,
    augmentations: Optional[Dict[str, np.ndarray]] = None,
    # No default since 2026-09-10. 'rmse' silently searched every antibody and
    # classification dataset under the wrong metric, and the metric is in the
    # cache filename and the result database's identity, so it was not
    # recoverable from a stored row. Resolve it with _resolve_dataset_metric.
    metric: str = None,
    dataset_name: Optional[str] = None,
    tier: str = 'general',
    task_type: Optional[str] = None,
    augmentation_strategy: Optional[str] = None,
    objective: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, SearchResult]:
    """Run find_best_hybrid and return hybrid features + flat importance weights.

    `dataset_name`/`tier` are forwarded so `find_best_hybrid(cache=True)` can
    locate its cache file. Required (the cache layer raises ValueError without
    them); pass `dataset_name=None` only if the caller can tolerate raising.
    `augmentation_strategy` should be set explicitly when `augmentations` is
    None to avoid the cache-key collision documented in
    `project_cache_collision_fix.md`.
    """
    kwargs = dict(
        base_reps=reps,
        labels=labels,
        groups=groups,
        return_fitted=True,
        weight_method=weight_method,
        verbose=False,
        cache=True,
        metric=metric,
        dataset_name=dataset_name,
        tier=tier,
    )
    if task_type is not None:
        kwargs['task_type'] = task_type
    if objective is not None:
        kwargs['objective'] = objective
    if budget is not None:
        kwargs['budgets'] = [budget]
    if augmentations is not None:
        kwargs['augmentations'] = augmentations
    if augmentation_strategy is not None:
        kwargs['augmentation_strategy'] = augmentation_strategy
    elif augmentations is None:
        # Explicit 'none' avoids colliding with real uncapped-augmentation
        # cache files when augmentations are absent.
        kwargs['augmentation_strategy'] = 'none'

    _apply_dataset_search_speed(kwargs, dataset_name)
    search = find_best_hybrid(**kwargs)
    # Use RF as default downstream model for weight extraction
    fitted = (search.fitted_results or {}).get('rf')

    if fitted is None:
        raise RuntimeError("find_best_hybrid returned no fitted result for 'rf'")

    X_hybrid = apply_feature_selection(reps, _fitted_to_feature_info(fitted))
    # Append augmentation features if present (must match flatten_weights layout)
    if fitted.augmentation_info and augmentations is not None:
        from kirby.hybrid import apply_augmentation_selection
        X_aug = apply_augmentation_selection(augmentations, fitted.augmentation_info)
        if X_aug is not None:
            X_hybrid = np.hstack([X_hybrid, X_aug])
    weights_flat = flatten_weights(fitted)

    return X_hybrid, weights_flat, search


def _fitted_to_feature_info(fitted) -> dict:
    """Convert FittedResult to feature_info dict for apply_feature_selection."""
    feature_info = {}
    for rep_name, indices in fitted.selected_indices.items():
        feature_info[rep_name] = {'selected_indices': indices}
    return feature_info


def flatten_weights(fitted) -> np.ndarray:
    """Flatten per-rep feature_weights into a single array matching X_hybrid columns.

    Includes augmentation feature weights when augmentation_info is present.
    The returned array has one weight per column of the full hybrid matrix
    (base features + augmentation features).
    """
    parts = []
    # Base rep weights
    for rep_name in fitted.selected_indices:
        if fitted.feature_weights and rep_name in fitted.feature_weights:
            parts.append(fitted.feature_weights[rep_name])
        else:
            parts.append(np.ones(len(fitted.selected_indices[rep_name])))
    # Augmentation weights (appended after base features, matching X_hybrid layout)
    # NOTE: iteration order must match apply_augmentation_selection (insertion order)
    if fitted.augmentation_info:
        for aug_name in fitted.augmentation_info:
            aug = fitted.augmentation_info[aug_name]
            if 'importance_scores' in aug:
                parts.append(np.asarray(aug['importance_scores']))
            else:
                n_feat = aug.get('n_features', len(aug.get('selected_indices', [])))
                parts.append(np.ones(n_feat))
    return np.concatenate(parts) if parts else np.array([])


# The single-representation comparison of every noise experiment: the top
# NOISE_BASE_REP_COUNT base representations FOR EACH DOWNSTREAM MODEL, ranked by
# that model's own score in the stored search -- the same rule c02 and the rest
# of the study use (top_base_reps). Owner's decision 2026-09-25.
#
# ⛔ It replaced a fixed preference list of unlearned encodings
# (ecfp4, blosum, onehot, ...) taken in list order regardless of score. An
# earlier session introduced it without the owner's sign-off. On
# flab_thermo_tresanco it made the comparison onehot -- 0.3471 spearman with rf,
# the worst of the 10 representations -- while the hybrid drew from the best
# three, so every hybrid-versus-base number on that dataset was strong
# representations against the weakest one. Do not reintroduce a fixed list.
#
# The ranking is top_base_reps' default, rule='whole_dataset': each
# representation's mean over the stored search's own folds, as everywhere else
# in the study. That search saw every row, so which two representations are
# picked was decided with the held-out molecules in view; the scores reported
# for them are measured fresh on each fold.
NOISE_BASE_REP_COUNT = 2


def noise_base_reps_by_model(searches_by_model, models, reps=None, n=None,
                             dataset_name=None, verbose=True):
    """The top-n single representations for each downstream model.

    Args:
        searches_by_model: {model: SearchResult or None}, the stored search the
            pool returned for each model (noise_helpers.
            pool_recipe_per_downstream_model gives it as by_model[m]['search']).
        models: downstream model names, in the caller's order.
        reps: the representations loaded for this run. A representation the
            search ranked but this run did not load is skipped, and the next one
            down is taken.
        n: how many per model. Default NOISE_BASE_REP_COUNT.

    Returns:
        (by_model, missing)
        by_model: {model: [rep at rank 1, rep at rank 2, ...]}, possibly shorter
            than n when fewer representations carry a score for that model.
        missing: {model: one sentence} for every model that got none -- it has
            no single-representation comparison in this run, and says why.
    Structure-tuned `sim_*` recipes are study treatments and never selected.
    """
    n = NOISE_BASE_REP_COUNT if n is None else int(n)
    by_model, missing = {}, {}
    for m in models:
        search = (searches_by_model or {}).get(m)
        if search is None:
            missing[m] = ('the pool returned no stored search for this model, '
                          'so its single representations cannot be ranked')
            continue
        excl = [r for r in (getattr(search, 'baselines', {}) or {})
                if str(r).startswith('sim_')]
        try:
            got = top_base_reps(search, m, n, only=(list(reps) if reps is not None
                                                    else None),
                                exclude_reps=excl, require_full=False,
                                context={'dataset': dataset_name,
                                         'purpose': 'noise single-rep comparison'})
        except Exception as exc:
            missing[m] = f'ranking failed: {type(exc).__name__}: {exc}'
            continue
        names = [e['rep'] for e in got.get('selected', [])]
        if not names:
            missing[m] = ('no representation carries a score for this model in '
                          'the stored search')
            continue
        by_model[m] = names
    if verbose:
        for m in models:
            print(f"  single-representation comparison for {m}: "
                  + (', '.join(f'rank {i + 1} {r}' for i, r in
                               enumerate(by_model[m]))
                     if m in by_model else f'none -- {missing[m]}'))
    return by_model, missing


def base_rank_groups(base_reps_by_model, rank, models):
    """{rep: [models]} for the representations at `rank` (1-based).

    Models whose rank-`rank` representation is the same share one feature
    matrix. A model with fewer than `rank` representations is absent.
    """
    out = {}
    for m in models:
        names = base_reps_by_model.get(m) or []
        if len(names) >= rank:
            out.setdefault(names[rank - 1], []).append(m)
    return out


def flatten_fold_weights(feat_info) -> np.ndarray:
    """Flatten per-rep importance scores from per-fold feature_info.

    Unlike flatten_weights (which uses FittedResult from global search),
    this uses the feature_info dict returned by create_hybrid / build_fold_hybrid.
    Weights match the columns of the per-fold hybrid matrix exactly.
    """
    parts = []
    for rep_name, info in feat_info.items():
        if not isinstance(info, dict) or 'selected_indices' not in info:
            continue
        if 'importance_scores' in info:
            parts.append(np.asarray(info['importance_scores']))
        else:
            parts.append(np.ones(info.get('n_features', len(info['selected_indices']))))
    # Augmentation weights
    aug_info = feat_info.get('augmentation_info')
    if aug_info:
        for aug_name, aug in aug_info.items():
            if 'importance_scores' in aug:
                parts.append(np.asarray(aug['importance_scores']))
            else:
                n_feat = aug.get('n_features', len(aug.get('selected_indices', [])))
                parts.append(np.ones(n_feat))
    return np.concatenate(parts) if parts else np.array([])


# =============================================================================
# PLANTED DECOY DONORS
#
# Two donors that a working admission rule must take NOTHING from. Both were
# written for scratchpad/aug_decoy_check.py and are moved here so c21 and the
# framework augmentation experiments share one implementation instead of three.
#
# They are offered alongside the real donors in the same pool, so they cost no
# extra conditions. What they buy is a count per fold: how many pure-noise
# features and how many near-copies of base features survived selection. On
# logd and herg_ki the shipping default took up to 5 of each per fold.
#
# The matched-shape RANDOM donor is a different thing and is not here: it is a
# separate arm (real donors replaced by random ones of the same shape), which
# isolates the selection from the extra model capacity. c21 builds that in
# _build_conditions.
# =============================================================================

NOISE_DONOR = '__gaussian__'
ECHO_DONOR = '__echo__'


def make_noise_donor(n_samples, n_features, rng):
    """A donor of pure Gaussian noise, `n_features` wide.

    Width is normally matched to the total width of the real donors, so the
    noise competes on equal terms in the pooled ranking rather than being
    drowned out. Anything taken from this donor is a defect.
    """
    return rng.standard_normal((n_samples, n_features))


def make_echo_donor(base_X, n_cols, jitter, rng):
    """Features copied out of the base representation and roughed up slightly.

    Each copied feature gets Gaussian noise added at `jitter` times that
    feature's own standard deviation, so a duplicate-detector cannot find it by
    an exact match. This is the only decoy that catches a rule rewarding
    features the base already carries; no amount of pure noise reveals that.

    Features with no spread are skipped: a constant feature is not a duplicate
    of anything, it is dead, and letting one in makes the decoy easier than it
    should be.

    Returns None when the base has no varying feature at all.
    """
    sd = base_X.std(axis=0)
    live = np.flatnonzero(sd > 0)
    if live.size == 0:
        return None
    take = rng.choice(live, size=min(n_cols, live.size), replace=False)
    echo = base_X[:, take].astype(float, copy=True)
    echo += rng.standard_normal(echo.shape) * (sd[take] * jitter)
    return echo


def plant_decoys(donors, base_X, rng, echo_cols=64, echo_jitter=0.05):
    """Add both decoy donors to a {name: features} pool. Returns a new dict.

    The noise donor is matched in width to the real donors combined. Build the
    decoys full length, once per fold with a fresh seed, so the same features
    can be sliced for the training rows, the admission holdout and the outer
    fold.
    """
    out = dict(donors)
    n_real = sum(v.shape[1] for v in donors.values())
    n_samples = base_X.shape[0]
    if n_real > 0:
        out[NOISE_DONOR] = make_noise_donor(n_samples, n_real, rng)
    echo = make_echo_donor(base_X, echo_cols, echo_jitter, rng)
    if echo is not None:
        out[ECHO_DONOR] = echo
    return out


def decoy_counts(augmentation_info):
    """How many features this fold kept from each decoy donor.

    Returns {'noise': int, 'echo': int}. Both should be 0.
    """
    info = augmentation_info or {}

    def _n(name):
        d = info.get(name) or {}
        return int(len(np.asarray(d.get('selected_indices', ()))))

    return {'noise': _n(NOISE_DONOR), 'echo': _n(ECHO_DONOR)}


def fold_hybrid_scoring_kwargs(blueprint, *, model=None, metric_fn=None,
                               lower_is_better=None):
    """The model/metric arguments build_fold_hybrid will accept for this recipe.

    A recipe searched under a geometry objective fits no downstream model, and
    build_fold_hybrid raises when model or metric_fn arrives with one. Every
    caller that rebuilds recipes from both search families hands its arguments
    through here: for a predictive recipe they come back unchanged, for a
    geometry recipe an empty dict comes back instead.

    Without this the caller passes a model to a geometry recipe, the raise is
    swallowed by the caller's per-fold except, and every similarity-family arm
    is written out as not computed on every fold.
    """
    from kirby.find_best_hybrid import _is_geometry_objective
    if _is_geometry_objective(getattr(blueprint, 'objective', None)):
        return {}
    return {'model': model, 'metric_fn': metric_fn,
            'lower_is_better': lower_is_better}


def build_fold_hybrid(train_reps, val_reps, y_train, blueprint,
                      augmentations_train=None, augmentations_val=None,
                      budget_override=None, aug_per_source_budget=None,
                      *, groups=None, model=None, metric_fn=None,
                      lower_is_better=None, objective=None, is_clf=None,
                      candidate_ranking=None, augmentation_threshold=None,
                      augmentation_budget=None, aug_max_candidates=None,
                      precomputed_importance=None, precomputed_alloc_scores=None):
    """Per-fold hybrid construction using a search blueprint.

    Properly passes internal_evaluator and augmentations from the blueprint,
    unlike bare create_hybrid calls that only pass allocation/budget/importance.

    Args:
        train_reps: {rep_name: X_train} for this fold.
        val_reps: {rep_name: X_val} for this fold.
        y_train: Training labels for this fold.
        blueprint: Blueprint from search result (e.g. best_by_model[model]['blueprint']).
        augmentations_train: {aug_name: X_train} sliced augmentation reps, or None.
        augmentations_val: {aug_name: X_val} sliced augmentation reps, or None.
        budget_override: If set, apply the blueprint's recipe at this budget
            instead of `blueprint.budget`. Used by c15's fine-grained sweep
            (double-descent detection at non-canonical budgets like 75, 150,
            300, 750, etc.) where a single fixed recipe is applied across
            many budgets to isolate budget as the only variable.
        aug_per_source_budget: int or {aug_name: N}, forwarded to the uncapped
            augmentation path. Defaults to the blueprint's own value, so a
            rebuild reproduces the configuration the search chose. Pass it
            explicitly only to override. getattr guards blueprints unpickled
            from caches written before the field existed.
        groups: Group labels ALREADY SLICED to this fold's training rows
            (i.e. `groups[train_idx]`), one entry per row of y_train. Forwarded
            to create_hybrid, which uses it for the internal admission holdout:
            with groups it takes the first fold of a GroupKFold, without it it
            falls back to a RandomState(42) 20% RANDOM split. On a
            scaffold-split dataset that fallback judges keep/drop decisions on
            a holdout sharing scaffolds with its own training rows, so leaving
            this None reintroduces the leak the outer split exists to prevent.
        model: The real downstream estimator INSTANCE (not a factory) that this
            fold's hybrid will be fed to. Forwarded to create_hybrid, which
            hands it to _evaluate_allocation: with a model, accept/reject is
            scored by fitting that model and calling metric_fn; without one it
            falls back to the internal evaluator's r2_score / accuracy_score.
            Also resolves internal_evaluator='match'. Requires metric_fn.
        metric_fn: The dataset's real metric, called as metric_fn(y_true,
            y_pred). Only used by create_hybrid when `model` is also given.
        lower_is_better: Direction for metric_fn. create_hybrid negates the
            score when True (hybrid.py:1374), and it applies that ONLY on the
            model branch — so the direction must match metric_fn, not the
            internal evaluator. None (default) takes the direction from the
            blueprint, which is the metric the search itself used; pass it
            explicitly if metric_fn is not the blueprint's metric, because a
            wrong direction silently inverts every accept decision.
        candidate_ranking: 'residual_correlation' ranks each candidate feature
            by how well it explains what the base representation could NOT;
            'rf_importance' ranks it by how well it predicts the property on
            its own. None (default) takes the blueprint's value. Before
            2026-08-26 the blueprint carried no such field, so this always fell
            back to 'residual_correlation' even for a search that ran the other
            rule — and the two are separate rows of the stored result, so the
            row said one thing and the rebuild did another.
        augmentation_threshold: how much a whole donor block must improve the
            score for greedy_ablation to keep it. None takes the blueprint's.
        augmentation_budget: features taken from EACH donor by the 'all' and
            'greedy_ablation' paths. Not aug_per_source_budget, which caps
            donors before pooling for 'capped'/'uncapped'. None takes the
            blueprint's.
        aug_max_candidates: length of the ranked candidate list a block size is
            chosen from. None takes the blueprint's, which is 50. At 50 an
            augmentation_fraction of 0.05, 0.10 or 0.25 of a 1000 budget all
            clamp to the same 50 features, so the size setting means nothing
            until this moves. Swept by its own experiment; it is NOT part of a
            stored result's key, so two values would collide on one row if a
            SEARCH ever varied it.

    Returns:
        (X_tr_h, X_va_h, feat_info)
    """
    from kirby.hybrid import apply_augmentation_selection

    if aug_per_source_budget is None:
        aug_per_source_budget = getattr(blueprint, 'aug_per_source_budget', None)

    # Direction defaults to the blueprint's own metric direction. Only meaningful
    # alongside metric_fn — create_hybrid applies it on the model branch only.
    if lower_is_better is None:
        # No silent default — a blueprint with neither a known metric name nor
        # a stored flag raises. Fixed 2026-09-10 (metric-direction audit).
        lower_is_better = object_direction(blueprint, 'the blueprint passed to '
                                           'build_fold_hybrid')

    # Objective defaults to the blueprint's own, so a rebuild reproduces the
    # quantity the search selected under. Blueprints pickled before the
    # objective field existed report 'predictive', which is what they were.
    if objective is None:
        objective = getattr(blueprint, 'objective', None)

    # Under a geometry objective create_hybrid fits no downstream model —
    # model/metric_fn are ignored and, worse, metric_fn IS the raising guard.
    # Refuse the contradiction rather than let a caller believe it passed one.
    from kirby.find_best_hybrid import (
        OBJECTIVE_PREDICTIVE as _OBJ_PRED, _is_geometry_objective,
    )
    if objective is not None and _is_geometry_objective(objective):
        if model is not None or metric_fn is not None:
            raise ValueError(
                f"build_fold_hybrid: objective={objective!r} scores held-out "
                f"geometry and fits no downstream model, but model/metric_fn "
                f"were passed. Under a geometry objective the dataset metric "
                f"is a raising guard — pass neither.")
        if is_clf is None:
            # The recipe records 'classification' or 'regression' in
            # `task_type`; there is no `is_clf` field on Blueprint, so the
            # getattr this line used to do returned False for every recipe.
            # Under a geometry objective that made the rebuild score each
            # molecule's neighbours by Spearman on the class codes, while the
            # search itself scored 2*(AUROC - 0.5) on them.
            is_clf = (getattr(blueprint, 'task_type', 'regression')
                      == 'classification')
    elif is_clf is None:
        is_clf = False

    # A model with no metric_fn is worse than no model at all: create_hybrid's
    # model branch calls metric_fn(y_val, y_pred) and its except clause swallows
    # TypeError (hybrid.py:1369), so metric_fn=None would score EVERY allocation
    # as -inf silently instead of raising.
    if model is not None and metric_fn is None:
        raise ValueError(
            "build_fold_hybrid: model was given without metric_fn. "
            "create_hybrid scores the model branch with metric_fn(y_val, y_pred) "
            "and silently treats the resulting TypeError as an infeasible "
            "allocation, so pass both or neither.")

    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(y_train):
            raise ValueError(
                f"build_fold_hybrid: groups has {len(groups)} entries but "
                f"y_train has {len(y_train)}. Pass this fold's TRAIN slice "
                f"(groups[train_idx]), not the whole-dataset groups array.")

    # ── THE RECIPE'S REPRESENTATION COUNT. Fixed 2026-09-18. ──────────────
    #
    # DEFAULT_REP_COUNTS = [2, 3, 5, None] is a SEARCHED axis: the whole grid
    # runs once per entry, each time restricted to the N representations that
    # fold's allocation scores rank highest (find_best_hybrid.py:4706-4721).
    # A configuration that won on 2 of 10 is stored with rep_count=2.
    #
    # This function ignored that. It handed create_hybrid whatever `train_reps`
    # its caller passed — the whole pool — so a recipe that won on 2
    # representations was rebuilt over all 10. Different features, different
    # budget split, and the score read back under the winning recipe's name.
    # Same class as the ranking/threshold fields the comment below records:
    # a recipe field that existed in the search and was never threaded into the
    # rebuild, with nothing raising.
    #
    # The search ranks by that fold's allocation scores. The nearest thing here
    # is `precomputed_alloc_scores` when the caller passed this fold's, and
    # `Blueprint.rep_scores` otherwise — the search's own per-rep allocation
    # scores, carried on the recipe for exactly this. Higher is better in both,
    # whatever the dataset's metric (find_best_hybrid.py:4700-4701).
    #
    # ⛔ NO SILENT FALLBACK, same rule the search states at :4703-4705. A recipe
    # that asks for N and cannot be ranked raises here rather than quietly
    # rebuilding on the whole pool and being read as N.
    _rep_count = getattr(blueprint, 'rep_count', None)
    _rep_count_source = None
    if _rep_count is not None and int(_rep_count) < len(train_reps):
        _rep_count = int(_rep_count)
        _scores = precomputed_alloc_scores or getattr(blueprint, 'rep_scores', None)
        _rep_count_source = ('fold_alloc_scores' if precomputed_alloc_scores
                             else 'blueprint.rep_scores')
        if not _scores:
            raise ValueError(
                f"build_fold_hybrid: this recipe was searched over "
                f"{_rep_count} of the pool's representations "
                f"(blueprint.rep_count={_rep_count}) and neither "
                f"precomputed_alloc_scores nor blueprint.rep_scores is "
                f"available to say WHICH {_rep_count}. Rebuilding on all "
                f"{len(train_reps)} would give a different feature set and a "
                f"different budget split from the configuration that won, and "
                f"the score would be filed under the winner's name.")
        _missing = [r for r in train_reps if r not in _scores]
        if len(train_reps) - len(_missing) < _rep_count:
            raise ValueError(
                f"build_fold_hybrid: the recipe asks for {_rep_count} "
                f"representations and only {len(train_reps) - len(_missing)} of "
                f"the {len(train_reps)} given carry an allocation score "
                f"({_rep_count_source}). Missing: {sorted(_missing)}.")
        # Score first, then name, so two representations on the same score
        # resolve the same way on every run.
        _ranked = sorted(train_reps,
                         key=lambda r: (-_scores.get(r, float('-inf')), r))
        _keep = set(_ranked[:_rep_count])
        train_reps = {k: v for k, v in train_reps.items() if k in _keep}
        val_reps = {k: v for k, v in val_reps.items() if k in _keep}

    _budget = budget_override if budget_override is not None else blueprint.budget
    X_tr_h, feat_info = create_hybrid(
        train_reps, y_train,
        allocation_method=blueprint.allocation_method,
        budget=_budget,
        importance_method=blueprint.importance_method,
        internal_evaluator=blueprint.internal_evaluator,
        augmentations=augmentations_train,
        augmentation_strategy=blueprint.augmentation_strategy,
        aug_per_source_budget=aug_per_source_budget,
        # Part of the recipe. Omitting it accepts blocks of extras the search
        # would have rejected. getattr guards blueprints pickled before the
        # field existed.
        augmentation_margin=float(
            getattr(blueprint, 'augmentation_margin', 0.0) or 0.0),
        augmentation_fraction=getattr(blueprint, 'augmentation_fraction', None),
        # Added 2026-08-26. Three of these four are columns of the stored
        # result's KEY, so the search files two values as two separate rows —
        # yet none of them was on the recipe, so every rebuild here took
        # create_hybrid's defaults ('residual_correlation', None, 20) whatever
        # the search ran under. A run under 'rf_importance' was rebuilt under
        # the other ranking rule and nothing raised. The fourth,
        # aug_max_candidates, is the length of the ranked candidate list; it is
        # NOT a key column and is swept by its own experiment.
        candidate_ranking=(candidate_ranking if candidate_ranking is not None
                           else getattr(blueprint, 'candidate_ranking',
                                        'residual_correlation')),
        augmentation_threshold=(
            augmentation_threshold if augmentation_threshold is not None
            else getattr(blueprint, 'augmentation_threshold', None)),
        augmentation_budget=int(
            augmentation_budget if augmentation_budget is not None
            else getattr(blueprint, 'augmentation_budget', 20)),
        aug_max_candidates=int(
            aug_max_candidates if aug_max_candidates is not None
            else getattr(blueprint, 'aug_max_candidates', 50)),
        groups=groups,
        # The per-representation scores the SEARCH chose. The engine's own
        # rebuild (find_best_hybrid.apply_blueprint) forwards these; this one
        # did not, so every rebuild here re-scored the representations from the
        # fold's own data instead of reusing what the search decided.
        #
        # That is not only slower. A top_k or performance_weighted allocation
        # turns these scores into WHICH representations get budget, so a rebuild
        # that recomputes them can select a different set than the search did --
        # the same shape of defect as the recipe fields fixed on 2026-08-26.
        # getattr guards recipes pickled before the field existed.
        # Per-feature importance over the base representations. It depends on
        # the fold and the importance method and on NOTHING else -- not the
        # allocation, not the downstream model, not the augmentation settings.
        # create_hybrid recomputes it on every call, so a caller sweeping those
        # settings pays for it every time. Measured 2026-08-27 on
        # flab_thermo_tresanco: building the base hybrid for ONE fold with
        # augmentation OFF took 261.9s, and the grid rebuilds it for every
        # configuration.
        **({'precomputed_importance': precomputed_importance}
           if precomputed_importance is not None else {}),
        # Per-representation baseline scores. These depend on the fold and the
        # downstream model, not on the allocation method -- the allocation only
        # turns them into a budget split.
        #
        # Only an explicit argument is used. The recipe's own rep_scores were
        # measured over the WHOLE dataset by the search that produced it, so
        # falling back to them lets this fold's validation rows decide how much
        # budget each representation gets. A caller that has fold-local scores
        # passes them; otherwise create_hybrid re-scores on this fold's
        # training rows, which costs a rebuild per fold and is the honest one.
        **({'precomputed_alloc_scores': precomputed_alloc_scores}
           if precomputed_alloc_scores is not None else {}),
        model=model,
        metric_fn=metric_fn,
        lower_is_better=lower_is_better,
        **({'objective': objective, 'is_clf': is_clf}
           if objective is not None else {}),
    )
    # base_only=True: the augmentation block is appended just below from
    # augmentations_val (and any demoted reps). Without it
    # apply_feature_selection printed "what comes back is the base block
    # ALONE" on every augmented recipe although the rows were completed a few
    # lines later -- a false alarm seen in n02 on 2026-09-27.
    X_va_h = apply_feature_selection(val_reps, feat_info, base_only=True)

    # Apply augmentation selection to val if the hybrid included augmentations.
    #
    # A top_k allocation auto-demotes every base rep it gave zero budget into
    # the augmentation pool, INSIDE create_hybrid. Those reps then appear in
    # augmentation_info but not in the caller's val-side dict, and
    # apply_augmentation_selection raises. top_3 WAS the default allocation
    # until 2026-09-01 and a pool is typically ~10 reps, so on a searched
    # blueprint this fired on most folds; the raise then surfaced as a failed
    # fold wherever the caller wrapped this in try/except. No allocation in the
    # current default set zeroes a representation, so this now only fires on a
    # blueprint stored before that date or an explicit top_k request. The
    # demoted column indices are recorded by create_hybrid so the same columns
    # can be taken from val_reps here.
    aug_info = feat_info.get('augmentation_info')
    if aug_info:
        val_pool = dict(augmentations_val) if augmentations_val else {}
        demoted = feat_info.get('demoted_reps') or {}
        for _name, _cols in demoted.items():
            if _name not in val_pool and _name in val_reps:
                val_pool[_name] = val_reps[_name][:, _cols]
        if val_pool:
            X_va_aug = apply_augmentation_selection(val_pool, aug_info)
            if X_va_aug is not None:
                X_va_h = np.hstack([X_va_h, X_va_aug])

    # Which representations this rebuild was actually restricted to, and what
    # ranked them. Without it a reader cannot tell a recipe rebuilt on its own
    # N from one rebuilt on the whole pool — which is the state every rebuild
    # was in before 2026-09-18.
    if _rep_count_source is not None:
        feat_info['rep_count_applied'] = _rep_count
        feat_info['rep_count_ranked_by'] = _rep_count_source
        feat_info['rep_count_kept'] = sorted(train_reps)

    return X_tr_h, X_va_h, feat_info


# =============================================================================
# HYBRID SEARCH PRECOMPUTE
# =============================================================================

class PrecomputeResults(dict):
    """{aug_cond: SearchResult|None} plus a readable record of what failed.

    A plain dict in every respect the callers use (keyed lookup, `.get`,
    membership, len), so existing consumers are unaffected. The extra
    `failures` list holds one dict per caught exception:
      {'tier', 'dataset', 'aug_condition', 'error_type', 'error', 'traceback'}
    Before this existed a caught precompute exception left only `None` under
    the aug_condition key and a printed line, so a caller could not tell a
    failure from a condition that was never requested.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failures = []

    def record_failure(self, exc, *, tier=None, dataset=None,
                       aug_condition=None):
        """Append one failure record and return it."""
        import traceback as _tb
        rec = {
            'tier': tier,
            'dataset': dataset,
            'aug_condition': aug_condition,
            'error_type': type(exc).__name__,
            'error': str(exc),
            'traceback': _tb.format_exc(),
        }
        self.failures.append(rec)
        return rec


def precompute_failures(obj):
    """Failure records carried by a precompute dict; [] for a plain dict.

    Lets a caller read the failures without an isinstance check, and without
    breaking on precompute dicts built before PrecomputeResults existed.
    """
    return list(getattr(obj, 'failures', []) or [])


def precompute_hybrid_searches(reps, labels, groups, config,
                               augmentations=None, verbose=True,
                               task_type='auto', metric='auto',
                               domain=None,
                               dataset_name=None, tier='general',
                               adaptive_pruning_schedule=None,
                               model_pruning_aggressiveness=None,
                               internal_evaluators=None,
                               budgets=None,
                               models=None,
                               label_noise=None,
                               objective=None):
    """Find the best hybrid across all budgets for each augmentation condition.

    objective: search objective forwarded to find_best_hybrid —
    'predictive' (the default when None) or 'similarity'. 'similarity_sac'
    was CUT on 2026-08-26 and now raises.
    find_best_hybrid namespaces its cache tier by objective, so a geometry
    precompute lives beside the predictive one and never overwrites it.

    label_noise: a kirby.noise_spec.NoiseSpec. When given, the search TRAINS on a
    deterministic noisy copy of the labels and SCORES on the clean ones, so the
    hybrid gets no more access to the true labels than a base representation does.
    find_best_hybrid derives its own cache tier from the spec, so the noisy lineage
    lives beside the clean one and never overwrites it.

    ONE find_best_hybrid call per aug_condition with ALL budgets. Baselines
    and importance scores are computed once and reused across budgets.
    best_by_model from each search gives per-model best configs across all budgets.

    Args:
        reps: Base representations {name: array}.
        labels: Target values.
        groups: Group labels for CV (or None).
        config: Dict with keys: budgets.
        augmentations: {aug_name: array} or None.
        verbose: Print progress.
        dataset_name: REQUIRED. Used as part of the cache filename
            (e.g. '{dataset_name}__general__uncapped__full.pkl').
        tier: Cache-condition label for the filename. 'general' (default);
            for distance lineage pass 'general_distance' (must be combined
            with internal_evaluators=DISTANCE_EVALUATORS and
            budgets=DISTANCE_BUDGETS to populate the correct cache key).
        internal_evaluators: Override evaluator list. Pass DISTANCE_EVALUATORS
            for distance lineage. None = find_best_hybrid uses its defaults.
        budgets: Override budget list. Pass DISTANCE_BUDGETS for distance lineage.
            None = use FULL_BUDGETS.

    Returns:
        Dict mapping aug_cond -> SearchResult. Keys: 'none', and 'uncapped'
        if augmentations are available. Each SearchResult contains results
        across ALL budgets.
    """
    from kirby.find_best_hybrid import find_best_hybrid

    if dataset_name is None:
        raise ValueError(
            "precompute_hybrid_searches requires dataset_name= "
            "(used as part of the cache filename)"
        )

    # Always use FULL_BUDGETS + models=None for precompute
    # unless overridden (distance lineage uses DISTANCE_BUDGETS+EVALUATORS).
    # This matches the SLURM precompute cache key regardless of the
    # experiment's own --mode setting. When SLURM cache exists on disk,
    # these calls hit it instantly (no compute).
    if budgets is None:
        budgets = FULL_BUDGETS
    # `models` used to be overwritten with None unconditionally on this line,
    # which threw away both the argument and config['models']. The distance
    # lineage asks for six models including knn (this function's own docstring
    # says so, and get_fold_best_search and the SLURM precompute both pass
    # them), so the distance search scored five and never knn: s10 asked the
    # pool for a knn recipe, did not find one, and wrote hybrid_evaluated=False.
    # Worse than a missing arm — the allocation methods pruned at the end of
    # fold 0 are voted on per (importance, evaluator, model, budget) group, so
    # five models and six models prune differently, and the winners this search
    # reports for the other five models are not the winners the SLURM
    # precompute reports for the same dataset. The signature default is already
    # None, so a caller that passes nothing behaves exactly as before and the
    # line is simply gone.

    # Resolve metric using shared logic (matches get_precomputed_search fallback)
    metric, task_type = _resolve_search_metric(
        labels, task_type=task_type, metric=metric, domain=domain,
        dataset_name=dataset_name)

    # Precompute aug conditions. The pool is one 10-rep set (tier 'general');
    # the 'top' tier is retired (2026-08-24). It emits {none, uncapped, capped}.
    # capped used to live on 'top' only because 'general' was a big expensive
    # pool — now 'general' IS that 10-rep set, so capped folds back onto it.
    aug_conditions = ['none']
    if augmentations is not None and len(augmentations) > 0:
        aug_conditions.append('uncapped')
        # capped runs on the standard (non-distance) lineage only. Distance
        # precompute doesn't emit capped (no consumer needs it). Above
        aug_conditions.append('capped')   # gate removed 2026-08-27
    aug_conditions = _local_aug_conditions(aug_conditions)

    from kirby.result_cache import StaleResultDB as _StaleResultDB

    results = PrecomputeResults()
    t_start = time.time()

    for idx, aug_cond in enumerate(aug_conditions, 1):
        augs = augmentations if aug_cond != 'none' else None
        # Pass aug_strat verbatim so 'none'/'uncapped'/'capped' each get
        # their own cache filename. Mapping 'none'→'uncapped' collided
        # the no-aug results with real uncapped files on the same path.
        aug_strat = aug_cond

        if verbose:
            print(f"  [{idx}/{len(aug_conditions)}] aug={aug_cond} "
                  f"budgets={budgets} ...", end='', flush=True)

        t0 = time.time()
        try:
            _fbh_kwargs = dict(
                base_reps=reps,
                labels=labels,
                groups=groups,
                budgets=budgets,
                models=models,
                augmentations=augs,
                augmentation_strategy=aug_strat,
                task_type=task_type,
                metric=metric,
                cache=True,
                verbose=verbose,
                dataset_name=dataset_name,
                tier=tier,
                adaptive_pruning_schedule=adaptive_pruning_schedule,
                model_pruning_aggressiveness=model_pruning_aggressiveness,
            )
            if internal_evaluators is not None:
                _fbh_kwargs['internal_evaluators'] = internal_evaluators
            if label_noise is not None:
                _fbh_kwargs['label_noise'] = label_noise
            if objective is not None:
                _fbh_kwargs['objective'] = objective
            _apply_dataset_search_speed(_fbh_kwargs, dataset_name)
            search = find_best_hybrid(**_fbh_kwargs)
            results[aug_cond] = search
            elapsed = time.time() - t0

            if verbose:
                if search.best_by_model:
                    # Show summary from best across all models
                    all_bests = list(search.best_by_model.values())
                    _cmp = min if search.lower_is_better else max
                    top = _cmp(all_bests, key=lambda r: r.get('score_mean', 0))
                    score = top.get('score_mean', 0)
                    print(f" done {elapsed:.1f}s "
                          f"({len(all_bests)} models, "
                          f"top={top.get('model', '?')} score={score:.4f})")
                else:
                    print(f" done {elapsed:.1f}s (no results)")

        except _StaleResultDB:
            # NOT recorded as a per-condition failure. Every other failure here
            # is one condition's problem and the remaining conditions are still
            # worth computing; a result_db written under an older column list
            # fails EVERY condition and every later read of the same store, so
            # the run's own summary line reads "0/1 searches" and it carries on
            # to produce numbers from nothing. Stop instead.
            raise
        except Exception as e:
            if verbose:
                print(f" FAILED: {e}")
            # Value stays None (what every consumer already tests for); the
            # exception itself goes into results.failures so the caller can
            # read it instead of scraping stdout.
            results[aug_cond] = None
            results.record_failure(e, tier=tier, dataset=dataset_name,
                                   aug_condition=aug_cond)

    if verbose:
        total_time = time.time() - t_start
        n_ok = sum(1 for v in results.values() if v is not None)
        print(f"  Precompute complete: {n_ok}/{len(aug_conditions)} "
              f"searches in {total_time:.0f}s")
        if results.failures:
            _failed = ', '.join(f"{f['aug_condition']}={f['error_type']}"
                                for f in results.failures)
            print(f"  Precompute failures recorded ({len(results.failures)}): "
                  f"{_failed}")

    return results


def populate_precompute_dicts(data, dataset_name, *, domain=None,
                              task_type=None, metric='auto',
                              top_reps=None, verbose=True,
                              adaptive_pruning_schedule=None,
                              model_pruning_aggressiveness=None,
                              include_distance=True):
    """Populate the precompute dicts on `data` for the pool helper.

    Replaces the boilerplate that each category helper (complexity, similarity,
    noise, federated) was duplicating. After this call, data has keys:
      data['precomputed_searches']            (the 10-rep pool)
      data['precomputed_searches_distance']   (general_distance tier)
      data['precompute_errors']               {tier: [failure record, ...]}

    Both precompute dicts are PrecomputeResults, so `.failures` (or
    precompute_failures(d)) lists every caught exception. The same records are
    mirrored onto data['precompute_errors'] keyed by tier, including the case
    where the whole general_distance pass raised.

    The pool helper looks for exactly these keys, so categories that call
    this get cache-hit performance on distance-pool consumers (s04/s05/s06/
    s10/s13/c03) without paying disk-load per pool query.

    `top_reps` is accepted for backward compatibility and ignored: the 'top'
    tier is retired (2026-08-24) — there is one 10-rep pool per dataset.

    Args:
        data: dict with 'reps', 'labels', 'groups' (modified in place).
        dataset_name: cache file key.
        include_distance: set False if no distance-pool experiment will run
            (saves the distance precompute pass).
    """
    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    augs = data.get('augmentations')
    errors = data.setdefault('precompute_errors', {})

    # Main-pool precompute over the one 10-rep pool (tier 'general').
    if len(reps) >= 2:
        if verbose:
            print(f"  Precomputing main-pool (general, {len(reps)} reps)...")
        data['precomputed_searches'] = precompute_hybrid_searches(
            reps=reps, labels=labels, groups=groups,
            config={'budgets': FULL_BUDGETS},
            augmentations=augs, verbose=verbose,
            task_type=task_type, metric=metric, domain=domain,
            dataset_name=dataset_name, tier='general',
            adaptive_pruning_schedule=adaptive_pruning_schedule,
            model_pruning_aggressiveness=model_pruning_aggressiveness,
        )
        errors['general'] = precompute_failures(data['precomputed_searches'])
    else:
        data['precomputed_searches'] = PrecomputeResults()

    # Distance-tier precompute. Populates the dict the pool helper consumes
    # when distance=True. Uses DISTANCE_EVALUATORS + DISTANCE_BUDGETS.
    if not include_distance:
        data['precomputed_searches_distance'] = PrecomputeResults()
        return data

    if len(reps) >= 2:
        if verbose:
            print(f"  Precomputing distance-pool (general_distance)...")
        try:
            data['precomputed_searches_distance'] = precompute_hybrid_searches(
                reps=reps, labels=labels, groups=groups,
                config={'budgets': DISTANCE_BUDGETS},
                augmentations=augs, verbose=verbose,
                task_type=task_type, metric=metric, domain=domain,
                dataset_name=dataset_name, tier='general_distance',
                internal_evaluators=DISTANCE_EVALUATORS,
                budgets=DISTANCE_BUDGETS,
                models=DISTANCE_MODELS,
                adaptive_pruning_schedule=adaptive_pruning_schedule,
                model_pruning_aggressiveness=model_pruning_aggressiveness,
            )
            errors['general_distance'] = precompute_failures(
                data['precomputed_searches_distance'])
        except Exception as e:
            if verbose:
                print(f"  general_distance precompute failed: {e}")
            # The whole pass raised, so there is no per-aug_condition value to
            # key on: record one failure with aug_condition None on the empty
            # dict, and mirror it onto data['precompute_errors'].
            _empty = PrecomputeResults()
            _empty.record_failure(e, tier='general_distance',
                                  dataset=dataset_name, aug_condition=None)
            data['precomputed_searches_distance'] = _empty
            errors['general_distance'] = precompute_failures(_empty)
    else:
        data['precomputed_searches_distance'] = PrecomputeResults()

    return data


def _resolve_search_metric(labels, task_type, metric='auto', domain=None,
                           dataset_name=None):
    """Resolve (metric_name, task_type) for cache alignment.

    Delegates to _resolve_dataset_metric — prefers DATASET_REGISTRY when
    dataset_name is given, falls back to heuristic. Kept as a thin wrapper
    for existing callers that don't yet pass dataset_name.
    """
    metric_name, resolved_task, _ = _resolve_dataset_metric(
        dataset_name=dataset_name, labels=labels, task_type=task_type,
        domain=domain, metric=metric)
    return metric_name, resolved_task



def no_startup_searches(data, pruning_mode=None, suite=''):
    """What each suite's run_all does instead of whole-dataset searches.

    Until 2026-10-05 every run_all ran precompute_hybrid_searches over all of
    the dataset's rows before its first experiment. A search on all rows
    scores every hybrid on molecules that are later held out, so every stored
    hybrid score leaked the held-out fold (HANDOFF_PER_FOLD_ALIGNMENT.md;
    rule R2 of HANDOFF_EXPERIMENT_ALIGNMENT.md 2.1). Each experiment now
    builds the per-fold searches it uses (R6, fold_searches /
    fold_standard_selection). The in-memory stores are left empty, so an
    experiment not yet aligned that still reads them falls through to
    get_precomputed_search's on-disk lookup, which KIRBY_NO_COLD_SEARCH=1
    refuses for a whole-dataset search.

    Pruning is off in every per-fold search (Adelaide 2026-10-05), so the
    pruning mode a launcher passes is printed and not used.
    """
    data['precomputed_searches'] = {}
    data['precomputed_searches_distance'] = {}
    data.setdefault('precompute_errors', {})
    print(f"\n  {suite + ': ' if suite else ''}no whole-dataset searches at "
          f"start-up; each experiment builds its own per-fold searches.")
    if pruning_mode not in (None, '', 'no_pruning'):
        print(f"  pruning mode {pruning_mode!r} ignored: per-fold searches run "
              f"unpruned on the decided grid.")
    return data['precompute_errors']

def get_precomputed_search(reps, labels, groups,
                           aug_condition='uncapped', augmentations=None,
                           weight_method=None, precomputed=None,
                           metric='auto', task_type=None, domain=None,
                           dataset_name=None, tier='general',
                           return_fitted=False,
                           objective=None,
                           **kwargs):
    """Retrieve a hybrid search result for a given aug_condition.

    If precomputed dict is passed (from data['precomputed_searches']),
    returns the cached SearchResult directly — instant, no computation.
    Otherwise calls find_best_hybrid with all budgets and cache=True.

    The returned SearchResult contains results across ALL budgets.
    Use search.best_by_model[model] or search.best_by_model_budget[model][budget]
    for per-model best configs.

    Args:
        reps: Base representations {name: array}.
        labels: Target values.
        groups: Group labels for CV (or None).
        aug_condition: 'none', 'uncapped', or 'capped'.
        augmentations: {aug_name: array} or None.
        speed: Search speed tier ('fast' or 'full').
        return_fitted: If True, also fit best config on full data.
        weight_method: Importance method for weights (needed if return_fitted).
        precomputed: Dict from data['precomputed_searches'] (optional).
        metric: Metric name ('auto' resolves from domain/task_type).
            Must match what was used during precomputation for cache alignment.
        task_type: REQUIRED. 'classification' or 'regression'. Pass
            `data['task_type']` from load_data — no auto-detect.
        domain: Domain hint for metric resolution ('molecular', 'antibody', etc.).
        dataset_name: REQUIRED when precomputed is None (falls back to
            find_best_hybrid with cache=True). Used as part of the cache
            filename.
        tier: Cache-condition label for the store ('general').
        return_fitted: If True, also fit the best config on full data.
        objective: Search objective — 'predictive' (default when None),
            'similarity' ('similarity_sac' was CUT). Forwarded to find_best_hybrid.
            A non-predictive objective is REFUSED against the in-memory
            `precomputed` store: that store is keyed by aug_condition alone and
            is filled by a predictive precompute, so serving from it would have
            answered a geometry question with a predictive search — silently,
            with no error and a plausible-looking number.
        **kwargs: Additional args passed to find_best_hybrid (NOT `budget`;
            per-budget selection is done downstream via best_by_model_budget).

    Returns:
        SearchResult from find_best_hybrid.
    """
    from kirby.find_best_hybrid import (
        OBJECTIVE_PREDICTIVE as _OBJ_PRED, _VALID_OBJECTIVES as _OBJS,
    )
    if objective is not None and objective not in _OBJS:
        raise ValueError(
            f"get_precomputed_search: objective={objective!r}. "
            f"Must be one of {_OBJS}."
        )
    _geo_request = objective is not None and objective != _OBJ_PRED
    # Guard against kwargs leakage: aug_condition drives augmentation_strategy
    # per pool iteration. If a caller forwards augmentation_strategy via
    # **kwargs, it would silently override the per-iteration value below.
    if 'augmentation_strategy' in kwargs:
        import warnings as _warnings
        _warnings.warn(
            "augmentation_strategy passed via kwargs is ignored; "
            "get_precomputed_search sets it per aug_condition iteration",
            stacklevel=2,
        )
        kwargs = {k: v for k, v in kwargs.items() if k != 'augmentation_strategy'}

    # Fast path: return from in-memory precomputed results.
    # The store's keys are aug_conditions only — nothing in them records which
    # objective the search optimised. Handing a geometry request whatever sits
    # under its aug_condition is how a similarity experiment ended up reporting
    # a predictive search's answer. Refuse instead: either the caller has a
    # geometry precompute of its own, or it should fall through to the
    # objective-namespaced on-disk cache below.
    if precomputed is not None:
        if _geo_request:
            raise ValueError(
                f"get_precomputed_search: objective={objective!r} cannot be "
                f"served from the in-memory `precomputed` store — that store "
                f"is keyed by aug_condition alone and holds predictive "
                f"searches, so it would answer a geometry question with a "
                f"predictive result. Pass precomputed=None so the call falls "
                f"through to the objective-namespaced on-disk cache."
            )
        result = precomputed.get(aug_condition)
        if result is not None:
            return result

    # On-demand: call find_best_hybrid with all budgets and cache=True
    from kirby.find_best_hybrid import find_best_hybrid

    if dataset_name is None:
        raise ValueError(
            "get_precomputed_search requires dataset_name= when "
            "precomputed is None (cache filenames are dataset-keyed)"
        )

    # Resolve metric and task_type so cache keys match precomputation
    resolved_metric, resolved_task_type = _resolve_search_metric(
        labels, task_type=task_type, metric=metric, domain=domain,
        dataset_name=dataset_name)

    augs = augmentations if aug_condition != 'none' else None
    aug_strat = aug_condition

    # Allow kwargs to override defaults (e.g., budgets, internal_evaluators)
    call_kwargs = dict(
        base_reps=reps,
        labels=labels,
        groups=groups,
        budgets=FULL_BUDGETS,
        augmentations=augs,
        augmentation_strategy=aug_strat,
        metric=resolved_metric,
        task_type=resolved_task_type,
        return_fitted=return_fitted,
        weight_method=weight_method,
        cache=True,
        verbose=False,
        dataset_name=dataset_name,
        tier=tier,
        # Matches what precompute writes. Both sides are off by default since
        # 2026-09-13, so this is belt and braces rather than a correction — but
        # it is what made the precomputed searches readable again on 2026-09-10,
        # when the engine still defaulted to on and this reader passed nothing,
        # so every expected metadata said 0.15, every stored file said 0.0, the
        # check rejected on that one field, and every experiment rebuilt the
        # search from scratch beside a file it could not read.
        #
        # An experiment that wants pruning passes adaptive_pruning in kwargs
        # below; it is then asking for a different search and should not be
        # served the precomputed one.
        adaptive_pruning=False,
    )
    if objective is not None:
        call_kwargs['objective'] = objective
    call_kwargs.update(kwargs)

    _apply_dataset_search_speed(call_kwargs, dataset_name)
    return find_best_hybrid(**call_kwargs)


# `get_best_precomputed_search` used to live here. Deleted 2026-09-10
# (metric-direction audit). It had zero callers anywhere in the repository —
# superseded by get_best_search_for_downstream below — and it carried the one
# defaulted-direction read in this file that genuinely inverted: its empty-search
# sentinel was `float('inf') if getattr(search, 'lower_is_better', False) else
# float('-inf')`, and when `search` was None the getattr fell back to False, so
# the sentinel was -inf. On a lower-is-better metric -inf is the BEST possible
# value, which handed the comparison to whichever condition was missing.


# =============================================================================
# POOL SUBSETS — which family of stored searches a caller may draw from
# =============================================================================
# The pool a caller picks a recipe from is (tier x augmentation condition).
# Every candidate it could reach before 2026-09-12 was searched under the
# predictive objective. find_best_hybrid also stores searches run under
# objective='similarity' — features chosen so that each molecule's activity is
# predicted from its five nearest neighbours in the distance the features
# define — and those are stored SEPARATELY: the objective rides on the tier in
# the search filename and sits in the search metadata, so a similarity search
# is an extra file beside the predictive one, never a replacement.
#
# A subset names which objectives are eligible, IN PREFERENCE ORDER. The
# augmentation conditions are the same in every subset: 'none' (the base
# representations alone), 'uncapped' and 'capped' (with the augmentation
# representations). 'predictive_only' is the default and is exactly the
# behaviour every existing caller already gets.
#
# ⛔ THE TWO FAMILIES ARE SCORED IN DIFFERENT UNITS. A predictive search's
# stored score is the dataset's own metric (rmse on herg_ki, spearman on
# flab_thermo_tresanco). A similarity search's stored score is the Spearman
# correlation between each held-out molecule's true activity and the activity
# its five nearest neighbours predict — find_best_hybrid overrides the metric
# name to 'readacross' for it, and fits no downstream model at all. Nothing
# stored puts the two on one ruler, so the pool does NOT rank one against the
# other: see the 'objective_choice' key in the returned metadata.
POOL_SUBSETS = {
    'predictive_only': ('predictive',),
    'predictive_and_similarity': ('predictive', 'similarity'),
    'similarity_only': ('similarity',),
}
DEFAULT_POOL_SUBSET = 'predictive_only'


def resolve_pool_subset(pool_subset):
    """(objectives, subset_name) for a named pool subset. None = the default.

    Returns the objectives in PREFERENCE ORDER — first one wins when the
    families cannot be compared (see POOL_SUBSETS above).
    """
    from kirby.find_best_hybrid import _VALID_OBJECTIVES as _OBJS
    name = DEFAULT_POOL_SUBSET if pool_subset is None else pool_subset
    # KIRBY_POOL_SUBSET overrides every caller's choice, for local runs that
    # must read only the families already on disk (added 2026-09-28: logd has
    # no similarity search locally, so 'predictive_and_similarity' built one
    # live). Unset, nothing changes.
    name = os.environ.get('KIRBY_POOL_SUBSET') or name
    if name not in POOL_SUBSETS:
        raise ValueError(
            f"pool_subset={pool_subset!r} is not a known subset. "
            f"Known: {sorted(POOL_SUBSETS)}.")
    objectives = POOL_SUBSETS[name]
    bad = [o for o in objectives if o not in _OBJS]
    if bad:
        raise ValueError(
            f"pool subset {name!r} names objective(s) {bad} that "
            f"find_best_hybrid does not accept ({_OBJS}).")
    return objectives, name


# ⛔ THE TWO DATASETS THE PRECOMPUTE REFUSES TO BUILD 'uncapped' FOR.
#
# This list MUST equal UNCAPPED_SKIP_DATASETS in tests/precompute_hybrids.py.
# That file drops every 'uncapped' condition for them — a cost decision of
# 2026-06-21, because both are drug-target sets near 50,000 rows and
# `general/none` alone measured about 4 days on bindingdb_kd.
#
# _pool_conditions asked for 'uncapped' on every dataset with no exception, so
# on these two every experiment asked the pool for a search nobody had built.
# It either found nothing, or cold-built inside the experiment's own job the
# very thing the cost decision existed to avoid. The two halves of one decision
# disagreed and neither said so.
#
# The owner's call, 2026-09-18: stop asking rather than pay the days. An
# experiment on kiba or bindingdb_kd now runs on 'none' and 'capped', and the
# returned metadata says which conditions the pool held.
#
# Two demo sets are in precompute_hybrids.py's list as well and are NOT here:
# gb_demo_coding_vs_intergenomic_seqs and gb_demo_human_or_worm are demos, not
# part of the suite, so nothing reads a pool for them.
UNCAPPED_SKIP_DATASETS = frozenset({'bindingdb_kd', 'kiba'})


def _local_aug_conditions(conditions):
    """Cut the augmentation conditions to KIRBY_LOCAL_AUG_CONDITIONS when it is set.

    Unset — the cluster — this returns `conditions` unchanged. It exists for a
    laptop run: an 'uncapped' or 'capped' configuration-fold costs about eight
    model fits where 'none' costs one, so the two augmented conditions are what
    makes a local precompute take hours per dataset.

    ⛔ This is a COST cut for a smoke run, not a finding about augmentation.
    Anything run with it set cannot say which augmentation condition wins.

    Both places that name the conditions call this, so what the precompute
    BUILDS and what an experiment ASKS the pool for stay the same list — they
    have disagreed before and neither side said so.
    """
    want = os.environ.get('KIRBY_LOCAL_AUG_CONDITIONS')
    if not want:
        return conditions
    keep = {p.strip() for p in want.split(',') if p.strip()}
    return [c for c in conditions if c in keep] or ['none']


def _pool_conditions(distance, augmentations, dataset_name=None, n_samples=None):
    """The (tier, aug_condition, augmentations) members of the pool.

    One 10-rep pool ('general', or 'general_distance' for the distance
    lineage), 'none' always, plus 'uncapped' and — on the main lineage only —
    'capped' when augmentation representations were passed. This has to match
    what the precompute EMITS or the lookup misses and the experiment
    cold-builds the search live.

    dataset_name is what lets it match: a dataset in UNCAPPED_SKIP_DATASETS gets
    no 'uncapped' member, because the precompute builds none for it. Passing
    None keeps every condition, which is the old behaviour for any caller that
    does not know its dataset.
    """
    has_augs = bool(augmentations)
    _no_uncapped = dataset_name in UNCAPPED_SKIP_DATASETS
    if distance:
        pool = [('general_distance', 'none', None)]
        if has_augs and not _no_uncapped:
            pool.append(('general_distance', 'uncapped', augmentations))
        # The capped distance search, which precompute_hybrids.py builds under
        # 10,000 molecules since 2026-09-29. It was built and never read.
        if has_augs and n_samples is not None and int(n_samples) < 10000:
            pool.append(('general_distance', 'capped', augmentations))
        _keep = set(_local_aug_conditions([c for _t, c, _a in pool]))
        return [m for m in pool if m[1] in _keep]
    pool = [('general', 'none', None)]
    if has_augs:
        if not _no_uncapped:
            pool.append(('general', 'uncapped', augmentations))
        pool.append(('general', 'capped', augmentations))
    _keep = set(_local_aug_conditions([c for _t, c, _a in pool]))
    return [m for m in pool if m[1] in _keep]


def _resolve_pool_objectives(pool_subset, objective):
    """(objectives, subset_name) from the two ways a caller can name a family.

    `objective=` (one search family, the older parameter) and `pool_subset=`
    (a named set of families) mean overlapping things, so passing both is
    refused rather than one silently winning.
    """
    if pool_subset is not None and objective is not None:
        raise ValueError(
            f"pass pool_subset= or objective=, not both "
            f"(pool_subset={pool_subset!r}, objective={objective!r}). "
            f"pool_subset names a set of search families; objective names one.")
    if pool_subset is not None:
        return resolve_pool_subset(pool_subset)
    if objective is not None:
        return (objective,), f'objective={objective}'
    return resolve_pool_subset(None)


def _iter_pool_searches(reps, labels, groups, *, pool, dataset_name,
                        main_dict, weight_method, metric, task_type, domain,
                        return_fitted, label_noise, extra, verbose_pool):
    """Load every pool member's SearchResult. Yields (tier, aug, objective, search).

    One entry per (tier, aug_condition, objective) that produced a search with
    per-model results. A member that raised prints a warning and is skipped —
    a swallowed error here silently drops the hybrid from an experiment.
    """
    from kirby.find_best_hybrid import OBJECTIVE_PREDICTIVE as _OBJ_PRED
    from kirby.result_cache import StaleResultDB as _StaleResultDB
    for tier, aug_cond, augs, obj in pool:
        _geo = obj != _OBJ_PRED
        # The in-memory dicts are filled by a predictive precompute and record
        # no objective, so a geometry member must resolve from the
        # objective-namespaced files on disk.
        precomputed = (main_dict if (tier in ('general', 'general_distance')
                                     and not _geo) else None)
        try:
            search = get_precomputed_search(
                reps, labels, groups,
                aug_condition=aug_cond, augmentations=augs,
                weight_method=weight_method, precomputed=precomputed,
                metric=metric, task_type=task_type, domain=domain,
                dataset_name=dataset_name, tier=tier,
                return_fitted=return_fitted,
                **({'label_noise': label_noise} if label_noise is not None else {}),
                **({'objective': obj} if _geo else {}),
                **extra,
            )
        except _StaleResultDB:
            # NOT swallowed, and not turned into a skipped pool member. A
            # result_db written under an older column list means every member
            # of this pool fails to open, so the loop yields nothing and the
            # caller builds its recipe from an EMPTY pool -- and then reports a
            # number that looks like every other run's. Measured 2026-09-20 on
            # herg_ki, caco2 and adme_mdr1: the warning printed, the run
            # continued, and its recipe came from 0 rows while the cells
            # already on disk had taken theirs from 168, 168 and 150.
            raise
        except Exception as e:
            print(f"[pool] WARNING: {tier}/{aug_cond}/{obj} search failed: "
                  f"{type(e).__name__}: {e}")
            if verbose_pool:
                import traceback
                traceback.print_exc()
            continue
        # Memoize: if we have an in-memory dict and just computed/loaded a
        # result, write it back so subsequent pool calls in this process hit
        # memory instead of paying disk-load (or retriggering find_best_hybrid).
        if (precomputed is not None and search is not None
                and aug_cond not in precomputed):
            precomputed[aug_cond] = search
        if search is None or not search.best_by_model:
            continue
        yield tier, aug_cond, obj, search


def get_best_search_for_downstream(
    reps, labels, groups, *, downstream_model, budget,
    augmentations=None, 
    metric='auto', task_type=None, domain=None,
    dataset_name=None, distance=False,
    data=None, return_fitted=False, weight_method=None,
    label_noise=None,
    objective=None,
    pool_subset=None,
    **kwargs,
):
    """Pick the best precomputed SearchResult from the (tier x aug_condition) pool.

    objective: search objective — 'predictive' (default when None),
    'similarity' ('similarity_sac' was CUT). Under a geometry objective the in-memory
    `data['precomputed_searches']` dicts are bypassed entirely (they hold
    predictive searches and carry no objective marker), so the pool resolves
    from the objective-namespaced on-disk cache instead.

    pool_subset: which FAMILIES of stored search the pool may draw from, by name
    (see POOL_SUBSETS). 'predictive_only' is the default and is today's
    behaviour exactly. 'predictive_and_similarity' also admits the searches run
    under objective='similarity'. 'similarity_only' admits those alone. Passing
    both pool_subset= and objective= is refused. The two families are scored in
    different units and the pool does not rank one against the other — the
    returned metadata says which family the winner came from ('objective') and
    how it was chosen ('objective_choice'), and carries each family's own best
    under 'by_objective'.

    label_noise: a kirby.noise_spec.NoiseSpec. When given, resolves the NOISY
    lineage — a blueprint searched on noisy training labels and scored on clean
    ones — instead of the clean precompute. Use this in any experiment that injects
    label noise, so the hybrid does not enter the comparison having been built with
    access to the very labels the experiment then corrupts.

    Selection is per-(downstream_model, budget). For each candidate SearchResult
    in the pool, looks up the best config for (downstream_model, budget) via
    get_best_for_downstream and compares by score_mean (respecting
    search.lower_is_better). Returns the winning SearchResult and metadata about
    which (tier, aug_condition) won.

    Main pool (distance=False):
      tier='general'  x aug=['none', 'uncapped']
      tier='top'      x aug=['none', 'uncapped', 'capped']
      (general/capped intentionally absent -- dropped 2026-05-22)

    Distance pool (distance=True):
      tier='general_distance' x aug=['none', 'uncapped']
      tier='top_distance'     x aug=['none', 'uncapped']
      (DISTANCE_EVALUATORS + DISTANCE_BUDGETS lineage; no capped)

    Args:
        reps: Base representations {name: array}.
        labels: Target values.
        groups: Group labels for CV (or None).
        downstream_model: Model name to optimise for (e.g. 'rf', 'svm', 'knn').
        budget: Specific budget the caller cares about (used for selection).
            Pass None to compare across all budgets for the model.
        augmentations: {aug_name: array} or None. Determines which aug
            conditions are eligible; with no augs, only 'none' is considered.
        speed: 'fast' or 'full'.
        metric: Metric name ('auto' = resolve from domain/task_type).
        task_type: 'classification' or 'regression' (required by underlying
            get_precomputed_search).
        domain: Domain hint for metric resolution.
        dataset_name: REQUIRED. Cache files are dataset-keyed.
        distance: If True, iterate the distance-tier cache lineage instead
            of the main pool. Auto-passes internal_evaluators=DISTANCE_EVALUATORS
            and budgets=DISTANCE_BUDGETS to the underlying loaders.
        data: Optional. If provided, prefers the data['precomputed_searches']
            dict (in-memory hit). Falls back to disk-cached find_best_hybrid
            when missing.
        return_fitted: If True, request fitted=True from underlying searches.
        weight_method: Importance method for fitted results.
        **kwargs: Extra kwargs forwarded to get_precomputed_search.

    Returns:
        (search, metadata) where metadata is either:
            {
                'tier': str,
                'aug_condition': str,
                'score_mean': float,
                'model': str,
                'budget': int or None,
                'best_config': dict,  # from get_best_for_downstream
            }
        or (None, None) if no candidate had a valid result for the requested
        (downstream_model, budget).
    """
    if dataset_name is None:
        raise ValueError(
            "get_best_search_for_downstream requires dataset_name= "
            "(cache files are dataset-keyed)"
        )

    # Guard against kwargs leakage (same as get_precomputed_search): the pool
    # below iterates aug conditions and we must not let a caller-supplied
    # augmentation_strategy override the per-iteration value via **extra.
    if 'augmentation_strategy' in kwargs:
        import warnings as _warnings
        _warnings.warn(
            "augmentation_strategy passed via kwargs is ignored; "
            "get_best_search_for_downstream sets it per pool iteration",
            stacklevel=2,
        )
        kwargs = {k: v for k, v in kwargs.items() if k != 'augmentation_strategy'}

    # Which search families are eligible, in preference order.
    objectives, pool_subset_name = _resolve_pool_objectives(
        pool_subset, objective)

    # One 10-rep pool (tier 'general'); the 'top' tier is retired (2026-08-24).
    # The same (tier, aug_condition) members are asked of every eligible family.
    base_pool = _pool_conditions(distance, augmentations, dataset_name,
                                 n_samples=len(labels))
    pool = [(tier, aug_cond, augs, obj)
            for obj in objectives
            for tier, aug_cond, augs in base_pool]
    extra = dict(kwargs)
    if distance:
        # Distance lineage uses its own evaluators/budgets.
        extra.setdefault('internal_evaluators', DISTANCE_EVALUATORS)
        extra.setdefault('budgets', DISTANCE_BUDGETS)
        # Must match the models= passed by every distance precompute site, or
        # the disk-fallback key misses and cold-rebuilds an existing search.
        extra.setdefault('models', DISTANCE_MODELS)

    # Resolve in-memory precomputed dicts if data was passed.
    # Distance lineage now ALSO uses the in-memory precomputed_searches_distance
    # dict when populated by category helpers.
    # A geometry objective must never read the in-memory dicts: they are filled
    # by a predictive precompute and record no objective, so a hit there answers
    # a geometry question with a predictive search. Force the disk path, whose
    # cache filenames ARE objective-namespaced.
    from kirby.find_best_hybrid import OBJECTIVE_PREDICTIVE as _OBJ_PRED
    _any_predictive = any(o == _OBJ_PRED for o in objectives)
    # Same rule for a search asked to pick its winner on the noisy labels: the
    # in-memory dicts hold searches that picked theirs on the clean ones and
    # record nothing to tell them apart, so a hit there answers the question
    # with the search the caller is trying to avoid.
    _noisy_selection = bool(kwargs.get('score_on_noisy_labels'))
    # Same rule again for a caller that names the downstream models itself. The
    # in-memory dicts are keyed by aug_condition and record nothing about which
    # models their search scored, so a request for a knn-scored search would be
    # handed the models=None search, which scored no knn. Force the disk path,
    # whose cache key does name the model list. The distance branch below sets
    # its models on `extra`, not on kwargs, so it still reads its own dict.
    _explicit_models = kwargs.get('models') is not None

    main_dict = None
    if (data is not None and _any_predictive and not _noisy_selection
            and not _explicit_models):
        if label_noise is not None:
            # Noisy lineage: one {aug_cond: SearchResult} dict per noise spec, keyed
            # by spec_id. Never falls back to the clean dicts — a clean blueprint
            # served under a noisy request is exactly the silent bug this parameter
            # exists to kill.
            _sid = label_noise.spec_id
            main_dict = (data.get('precomputed_searches_noisy') or {}).get(_sid)
        elif distance:
            main_dict = data.get('precomputed_searches_distance')
        else:
            main_dict = data.get('precomputed_searches')

    _verbose_pool = os.environ.get('KIRBY_POOL_VERBOSE') == '1'

    candidates = []  # list of (tier, aug_cond, objective, search, best_config)
    # The load loop (including its per-member warning and the write-back into
    # the in-memory dict) lives in _iter_pool_searches, which the per-fold pool
    # reader uses as well.
    for tier, aug_cond, obj, search in _iter_pool_searches(
            reps, labels, groups, pool=pool, dataset_name=dataset_name,
            main_dict=main_dict, weight_method=weight_method, metric=metric,
            task_type=task_type, domain=domain, return_fitted=return_fitted,
            label_noise=label_noise, extra=extra, verbose_pool=_verbose_pool):
        # Reject cross-model fallback. get_best_for_downstream's level-5
        # fallback returns the "best blueprint from any model" with that
        # model's score — comparing those scores across pool candidates
        # would let the pool pick whichever (tier, aug) has the highest
        # score in ANY model, not the best for downstream_model. Require
        # downstream_model is actually represented in this search.
        if (downstream_model not in (search.best_by_model_budget or {})
                and downstream_model not in (search.best_by_model or {})):
            if _verbose_pool:
                print(f"[pool] {tier}/{aug_cond}: model={downstream_model} absent, skipping")
            continue
        best = get_best_for_downstream(search, downstream_model, budget=budget)
        if not best:
            continue
        score = best.get('score_mean')
        if score is None or not np.isfinite(score):
            continue
        candidates.append((tier, aug_cond, obj, search, best))

    if not candidates:
        # No pool member produced a usable hybrid for this (model, budget).
        # The caller will fall back to a non-hybrid path — make that loud
        # rather than returning None silently (this is how a hybrid can
        # silently vanish from an experiment's results).
        print(f"[pool] WARNING: no usable hybrid for model={downstream_model} "
              f"budget={budget} dataset={dataset_name} — all {len(pool)} pool "
              f"conditions failed or were empty; caller will use a fallback.")
        return None, None

    # Sort: best score first (respecting lower_is_better), then by tie-break
    # preference: none > uncapped > capped.
    aug_pref = {'none': 0, 'uncapped': 1, 'capped': 2}
    tier_pref = {'general': 0, 'general_distance': 0}

    def _key(item, lower_is_better):
        tier, aug_cond, obj, search, best = item
        s = best['score_mean']
        primary = s if lower_is_better else -s
        return (primary, aug_pref.get(aug_cond, 99), tier_pref.get(tier, 99))

    # Rank INSIDE each search family first. Two families are not comparable by
    # score (see POOL_SUBSETS): one is the dataset's own metric, the other a
    # correlation between each held-out molecule's activity and what its five
    # nearest neighbours predict.
    by_objective = {}
    for obj in objectives:
        items = [c for c in candidates if c[2] == obj]
        if not items:
            continue
        # All members of one family for one (dataset, metric) should agree on
        # lower_is_better. If they don't, the same dataset was searched under
        # different metrics — that's a bug elsewhere, but to avoid silent
        # score-direction confusion we drop any that disagrees with the first.
        ref_lib = items[0][3].lower_is_better
        items = [c for c in items if c[3].lower_is_better == ref_lib]
        if not items:
            continue
        items.sort(key=lambda it: _key(it, ref_lib))
        by_objective[obj] = (items[0], bool(ref_lib))

    present = [o for o in objectives if o in by_objective]
    if not present:
        return None, None

    if len(present) == 1:
        chosen_obj = present[0]
        objective_choice = 'only_family'
    else:
        _metrics = {getattr(by_objective[o][0][3], 'metric_name', None)
                    for o in present}
        _dirs = {by_objective[o][1] for o in present}
        if len(_metrics) == 1 and len(_dirs) == 1:
            # Same metric on both — a real comparison is defined.
            _lower = next(iter(_dirs))
            chosen_obj = sorted(
                present,
                key=lambda o: (by_objective[o][0][4]['score_mean'] if _lower
                               else -by_objective[o][0][4]['score_mean']))[0]
            objective_choice = 'score'
        else:
            # Two different metrics. Nothing stored puts them on one ruler, so
            # the winner is the first family the subset names, and the other
            # family's best is returned beside it under 'by_objective'.
            chosen_obj = present[0]
            objective_choice = 'subset_order'
            print(f"[pool] {dataset_name}: pool subset {pool_subset_name!r} "
                  f"spans metrics {sorted(str(m) for m in _metrics)} — these "
                  f"are not comparable, so the winner is the first family the "
                  f"subset names ({chosen_obj!r}). Every family's own best is "
                  f"in metadata['by_objective'].")

    (tier, aug_cond, chosen_obj_, search, best), lower_is_better = \
        by_objective[chosen_obj]

    metadata = {
        'tier': tier,
        'aug_condition': aug_cond,
        'score_mean': best['score_mean'],
        'model': downstream_model,
        'budget': budget,
        'best_config': best,
        # Which search family this recipe came from, so a result file records
        # it. 'predictive' whenever no other family was asked for.
        'objective': chosen_obj,
        'metric_name': getattr(search, 'metric_name', None),
        'lower_is_better': lower_is_better,
        'pool_subset': pool_subset_name,
        'objective_choice': objective_choice,
        # Best of EVERY eligible family, not just the returned one.
        'by_objective': {
            o: {
                'tier': by_objective[o][0][0],
                'aug_condition': by_objective[o][0][1],
                'score_mean': by_objective[o][0][4]['score_mean'],
                'metric_name': getattr(by_objective[o][0][3], 'metric_name',
                                       None),
                'lower_is_better': by_objective[o][1],
                'best_config': by_objective[o][0][4],
            }
            for o in present
        },
        # What rows chose this recipe. 'score_mean' is the mean over the
        # SEARCH's own inner folds, run over every row of the dataset, so the
        # allocation method, budget, importance method, internal evaluator and
        # augmentation condition were all picked with the held-out rows of any
        # later split among them. An experiment that rebuilds this recipe per
        # fold fixes the feature WEIGHTS, not the choice of recipe. Carry this
        # into whatever the experiment reports so the number is never read as a
        # held-out one; nested_cv_evaluate is the path that does not need it.
        'selection_scope': 'whole_dataset',
        'selection_note': ('recipe chosen by its mean over the search\'s inner '
                           'folds on all rows; only the features are refit'),
    }
    return search, metadata


def pool_recipes_by_family(meta, space_for_family=None, eligible_families=None,
                           prefix='hybrid_'):
    """One recipe per search family the pool holds, and a reason for each family
    that produced none.

    This is n11's loop (n11_estimation_remediation_grid.py:731-762) lifted into
    one place on 2026-09-19, when thirteen more experiments needed the same
    thing and thirteen copies of it would have drifted. n11, n13, n18 and n21
    keep their own copies and are untouched, so no number they have already
    reported moves.

    WHY A LOOP AND NOT `meta['best_config']`. POOL_SUBSETS names the families in
    PREFERENCE order and get_best_search_for_downstream returns the FIRST family
    that has an entry, because the two families' stored scores are in different
    units — a predictive search's score is the dataset's own metric, a
    similarity search's is the correlation between each held-out molecule's
    activity and what its five nearest neighbours predict. A predictive recipe
    almost always exists, so reading `meta['best_config']` alone gives the
    predictive recipe and records that the other one existed. Every eligible
    family's own best is in `meta['by_objective']`, which is what this reads.

    ⛔ The two families are NOT ranked against each other here. Each becomes its
    own feature space and each is scored by the experiment in the dataset's own
    metric, which is the only comparison that is defined.

    Args:
        meta: the metadata dict from get_best_search_for_downstream, or None.
        space_for_family: {family name: the result-file name of its space}.
            None gives f'{prefix}{family}'. Pass the file's existing hybrid
            space name for 'predictive' so nothing already reported is renamed.
        eligible_families: the families the caller's pool subset named (i.e.
            POOL_SUBSETS[subset]). A family in here that the pool returned
            nothing for gets its own reason; a family NOT in here gets a
            different one. None skips both.
        prefix: used only when space_for_family has no entry for a family.

    Returns:
        (recipes, unavailable)

        recipes: one dict per family that carries a usable recipe, in the
        order the pool reported them (subset preference order):
          {'space': str, 'objective': str, 'blueprint': Blueprint,
           'tier': str, 'aug_condition': str, 'budget': int,
           'allocation_method': str, 'importance_method': str,
           'internal_evaluator': str, 'augmentation_strategy': str,
           'search_score_mean': float, 'search_metric_name': str,
           'search_lower_is_better': bool, 'is_pool_winner': bool}

        unavailable: {space name: why it was not built}. Written to the result
        file by the caller, so a space that is absent always carries a reason
        rather than leaving a reader to work out whether it failed or was never
        asked for.
    """
    def _name(fam):
        if space_for_family and fam in space_for_family:
            return space_for_family[fam]
        return f'{prefix}{fam}'

    recipes, unavailable = [], {}

    if not meta:
        for fam in (eligible_families or ()):
            unavailable[_name(fam)] = (
                'the precomputed pool returned no recipe for this dataset, so '
                'no feature space could be built from any search family')
        return recipes, unavailable

    # THE OWNER'S SET (get_rule_hybrids, 2026-10-02): when the caller attached
    # it, every slot becomes its own space -- general rank 1 keeps the
    # family's existing space name so nothing already reported is renamed;
    # general rank 2 and distance rank 1 get '__general2' / '__distance1'.
    rule = meta.get('rule_hybrids')
    if rule:
        for fam, fo in (rule.get('by_objective') or {}).items():
            base = _name(fam)
            for h in fo.get('hybrids') or []:
                bp = h.get('blueprint')
                slot = h.get('slot') or 'general1'
                name = base if slot == 'general1' else f'{base}__{slot}'
                if bp is None:
                    unavailable[name] = (f'the {fam} {slot} hybrid carried no '
                                         f'blueprint, so it could not be built')
                    continue
                recipes.append({
                    'space': name, 'objective': fam, 'blueprint': bp,
                    'tier': h.get('tier'),
                    'aug_condition': h.get('aug_condition'),
                    'lineage': h.get('lineage'), 'slot': slot,
                    'budget': int(getattr(bp, 'budget', 0) or 0),
                    'allocation_method': getattr(bp, 'allocation_method', None),
                    'importance_method': getattr(bp, 'importance_method', None),
                    'internal_evaluator': getattr(bp, 'internal_evaluator', None),
                    'augmentation_strategy': h.get('aug_condition'),
                    'search_score_mean': h.get('score_other_folds'),
                    'search_metric_name': h.get('metric_name'),
                    'search_lower_is_better': h.get('lower_is_better'),
                    'budget_relaxed': h.get('budget_relaxed'),
                    'is_pool_winner': slot == 'general1',
                })
            if fo.get('missing'):
                _why = ' | '.join(fo['missing'])
                if not fo.get('hybrids'):
                    unavailable[base] = _why
                else:
                    unavailable[f'{base}__missing_slots'] = _why
        for fam in (eligible_families or ()):
            if fam not in (rule.get('by_objective') or {}):
                unavailable[_name(fam)] = (
                    f'the pool held no search under objective={fam} for this '
                    f'dataset in either the general or the distance searches')
        return recipes, unavailable

    winner = meta.get('objective')
    by_obj = meta.get('by_objective') or {}
    if not by_obj:
        # A metadata dict from before 'by_objective' existed, or a pool that
        # returned one family and recorded nothing beside it. Its top-level
        # fields ARE that one family's entry.
        by_obj = {winner or 'predictive': {
            'tier': meta.get('tier'),
            'aug_condition': meta.get('aug_condition'),
            'score_mean': meta.get('score_mean'),
            'metric_name': meta.get('metric_name'),
            'lower_is_better': meta.get('lower_is_better'),
            'best_config': meta.get('best_config'),
        }}

    for fam, entry in by_obj.items():
        bp = ((entry or {}).get('best_config') or {}).get('blueprint')
        name = _name(fam)
        if bp is None:
            unavailable[name] = (
                f'the {fam} family was in the pool but its winning '
                f'configuration carried no blueprint, so its feature space '
                f'could not be built')
            continue
        recipes.append({
            'space': name,
            'objective': fam,
            'blueprint': bp,
            'tier': entry.get('tier'),
            'aug_condition': entry.get('aug_condition'),
            'budget': int(getattr(bp, 'budget', 0) or 0),
            'allocation_method': getattr(bp, 'allocation_method', None),
            'importance_method': getattr(bp, 'importance_method', None),
            'internal_evaluator': getattr(bp, 'internal_evaluator', None),
            'augmentation_strategy': getattr(bp, 'augmentation_strategy', None),
            'search_score_mean': entry.get('score_mean'),
            'search_metric_name': entry.get('metric_name'),
            'search_lower_is_better': entry.get('lower_is_better'),
            'is_pool_winner': fam == winner,
        })

    for fam in (eligible_families or ()):
        name = _name(fam)
        if name in unavailable or any(r['space'] == name for r in recipes):
            continue
        unavailable[name] = (
            f'the pool held no search under objective={fam} for this dataset, '
            f'so the space whose features that search chose could not be built')

    if eligible_families is not None and space_for_family:
        for fam, name in space_for_family.items():
            if fam in eligible_families:
                continue
            if name in unavailable or any(r['space'] == name for r in recipes):
                continue
            unavailable[name] = (
                f'the {fam} family is not in the pool subset this run asked '
                f'for, so its feature space was never requested')

    return recipes, unavailable


def describe_pool_family(family):
    """One sentence saying what a search family chose its features for.

    The result files say this in words rather than leaving 'similarity' to be
    decoded. Same wording as n06's FEATURES_CHOSEN_BY.
    """
    return {
        'predictive': 'prediction of the label',
        'similarity': ("the distance between molecules: the features were "
                       "chosen so that each molecule's activity is predicted "
                       "by its five nearest neighbours in that distance, with "
                       "no downstream model fitted at all"),
    }.get(family, family)


def _other_fold_mean(fold_scores, held_out_fold, drop_folds=()):
    """(mean over every OTHER fold, how many folds that was).

    The held-out fold's own score never enters the mean. `drop_folds` removes
    further fold numbers from it. Returns (None, 0) when no other fold carries
    a finite score.
    """
    drop = {int(f) for f in (drop_folds or ())}
    vals = [float(s) for k, s in enumerate(fold_scores or [])
            if k != held_out_fold and k not in drop and not _hr_is_bad(s)]
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)


def get_fold_pool_recipes(
    reps, labels, groups, *, fold_i, n_outer_folds,
    downstream_model, budget=None,
    dataset_name=None, augmentations=None, distance=False,
    pool_subset=None, objective=None, n_hybrids=3,
    metric='auto', task_type=None, domain=None,
    data=None, weight_method=None, label_noise=None,
    drop_folds=None, **kwargs,
):
    """Read the pool for ONE held-out fold: best base representation + top hybrids.

    Reads searches that already exist. It runs no search of its own — that is
    what separates it from get_fold_best_search, which searches each fold's
    training rows from scratch.

    For the held-out fold `fold_i` it returns, for `downstream_model`:
      - the single base representation with the best mean over the OTHER folds;
      - the `n_hybrids` hybrid configurations with the best mean over the OTHER
        folds, per eligible search family.
    Fold `fold_i`'s own score never enters either ranking. Each returned entry
    also carries that fold's score, so a caller can report what the pool scored
    there, but nothing is selected on it.

    WHICH FOLDS THESE ARE. The per-fold numbers come from the stored search's
    own cross-validation over every row of the dataset: 5 folds, seed 42,
    group-aware (GroupKFold, or StratifiedGroupKFold on a classification set).
    nested_cv_outer_folds(reps, labels, groups, 5, 42, task_type) builds the
    identical partition, so an experiment splitting that way has its fold i and
    the pool's fold i over the same rows. An experiment that splits any other
    way does NOT, and the returned 'n_folds_in_pool' is there to be checked
    against n_outer_folds before the result is used.

    ⛔ WHAT IS AND IS NOT HELD OUT. The candidate list itself was filtered using
    fold 0's scores whenever the search pruned mid-flight ('fold0_pruned'),
    and every representation and recipe in the pool was computed over all rows.
    So this is a recipe chosen without fold_i's SCORES, not a recipe chosen by a
    search that never saw fold_i's rows. 'selection_scope' says exactly that.

    Args:
        fold_i: the held-out fold number.
        n_outer_folds: how many folds the CALLER is splitting into. Compared
            against the number of fold scores the pool holds; a mismatch is
            reported in 'warnings' rather than silently ranked over.
        downstream_model: the model the base representations and hybrids are
            ranked for ('knn', 'rf', ...).
        budget: a specific feature budget, or None for every budget the search
            scored for this model. When a budget is named and no configuration
            was scored at it, the filter is relaxed to every budget and that is
            recorded in 'warnings'.
        pool_subset: which search families are eligible (see POOL_SUBSETS).
            Default 'predictive_only'. Pass 'predictive_and_similarity' to
            include the hybrids built under objective='similarity'.
        objective: one family instead of a named subset. Passing both is
            refused.
        n_hybrids: how many hybrid configurations to return PER FAMILY.
        drop_folds: further fold numbers to leave out of the other-folds mean.
            Empty by default: only fold_i is left out.

    Returns:
        {
          'fold': int,
          'n_outer_folds': int,
          'n_folds_in_pool': int or None,
          'model': str,
          'budget': int or None,
          'pool_subset': str,
          'objectives': [str, ...],
          'base_rep': {...} or None,     # first family in subset order
          'hybrids': [{...}, ...],       # every family, subset order
          'by_objective': {objective: {'base_rep': {...} or None,
                                       'hybrids': [{...}, ...],
                                       'metric_name': str,
                                       'lower_is_better': bool,
                                       'n_hybrid_candidates': int,
                                       'n_base_reps': int}},
          'selection_scope': 'other_folds_of_pool_search',
          'selection_note': str,
          'fold0_pruned': bool or None,
          'budget_relaxed': bool,
          'budgets_ranked_over': [int, ...],
          'warnings': [str, ...],
        }

        A hybrid entry ('recipe' is the swept configuration alone; 'name' adds
        the family and the condition, which the same recipe repeats across):
          {'rank': int, 'objective': str, 'tier': str, 'aug_condition': str,
           'recipe': str, 'name': str,
           'blueprint': Blueprint, 'allocation_method': str,
           'importance_method': str, 'internal_evaluator': str, 'budget': int,
           'metric_name': str, 'lower_is_better': bool,
           'score_other_folds': float, 'n_other_folds': int,
           'score_this_fold': float or None, 'fold_scores': [float, ...]}

        A base_rep entry:
          {'rep': str, 'objective': str, 'tier': str, 'aug_condition': str,
           'metric_name': str, 'lower_is_better': bool,
           'score_other_folds': float, 'n_other_folds': int,
           'score_this_fold': float or None, 'fold_scores': [float, ...]}
    """
    import copy as _copy

    if dataset_name is None:
        raise ValueError(
            "get_fold_pool_recipes requires dataset_name= "
            "(cache files are dataset-keyed)")
    # fold_i=None: nothing held out, every fold enters the mean. That is the
    # whole-dataset ranking get_rule_hybrids uses for experiments that read one
    # recipe set for the whole dataset rather than one per held-out fold.
    if fold_i is not None:
        fold_i = int(fold_i)
        if fold_i < 0:
            raise ValueError(f"fold_i must be >= 0, got {fold_i}")
    n_hybrids = int(n_hybrids)
    if n_hybrids < 1:
        raise ValueError(f"n_hybrids must be >= 1, got {n_hybrids}")

    if 'augmentation_strategy' in kwargs:
        import warnings as _warnings
        _warnings.warn(
            "augmentation_strategy passed via kwargs is ignored; "
            "get_fold_pool_recipes sets it per pool iteration", stacklevel=2)
        kwargs = {k: v for k, v in kwargs.items()
                  if k != 'augmentation_strategy'}

    objectives, pool_subset_name = _resolve_pool_objectives(
        pool_subset, objective)

    base_pool = _pool_conditions(distance, augmentations, dataset_name,
                                 n_samples=len(labels))
    pool = [(tier, aug_cond, augs, obj)
            for obj in objectives
            for tier, aug_cond, augs in base_pool]
    extra = dict(kwargs)
    if distance:
        extra.setdefault('internal_evaluators', DISTANCE_EVALUATORS)
        extra.setdefault('budgets', DISTANCE_BUDGETS)
        extra.setdefault('models', DISTANCE_MODELS)

    from kirby.find_best_hybrid import OBJECTIVE_PREDICTIVE as _OBJ_PRED
    _any_predictive = any(o == _OBJ_PRED for o in objectives)
    _noisy_selection = bool(kwargs.get('score_on_noisy_labels'))
    _explicit_models = kwargs.get('models') is not None
    main_dict = None
    if (data is not None and _any_predictive and not _noisy_selection
            and not _explicit_models):
        if label_noise is not None:
            main_dict = (data.get('precomputed_searches_noisy')
                         or {}).get(label_noise.spec_id)
        elif distance:
            main_dict = data.get('precomputed_searches_distance')
        else:
            main_dict = data.get('precomputed_searches')

    _verbose_pool = os.environ.get('KIRBY_POOL_VERBOSE') == '1'
    warnings_out = []

    # {objective: {'hybrids': [row...], 'baselines': {rep: entry},
    #              'metric_name':, 'lower_is_better':}}
    fam = {}
    n_folds_in_pool = None
    fold0_pruned = None
    # Set when the named budget matched no stored configuration and the ranking
    # fell back to every budget the search holds. A caller labelling its rows by
    # the budget it asked for must read this, or a row labelled 10 features can
    # hold a hybrid built at 500.
    budget_relaxed = False
    budgets_ranked_over = []

    for tier, aug_cond, obj, search in _iter_pool_searches(
            reps, labels, groups, pool=pool, dataset_name=dataset_name,
            main_dict=main_dict, weight_method=weight_method, metric=metric,
            task_type=task_type, domain=domain, return_fitted=False,
            label_noise=label_noise, extra=extra,
            verbose_pool=_verbose_pool):

        # None (a pickle written before the flag existed) counts as pruned —
        # the default schedule prunes — the same reading hr_exclude_folds
        # takes. True as soon as ANY pool member filtered on fold 0.
        _f0 = getattr(search, 'fold0_pruned', None)
        _pruned = True if _f0 is None else bool(_f0)
        fold0_pruned = (_pruned if fold0_pruned is None
                        else (fold0_pruned or _pruned))

        entry = fam.setdefault(obj, {
            'hybrids': [], 'baselines': {},
            'metric_name': getattr(search, 'metric_name', None),
            'lower_is_better': None,
        })
        try:
            lower = object_direction(
                search, f'the {tier}/{aug_cond} SearchResult in the pool')
        except ValueError as e:
            warnings_out.append(
                f"{tier}/{aug_cond}/{obj}: no usable score direction ({e}) — "
                f"this search was not read")
            continue
        if entry['lower_is_better'] is None:
            entry['lower_is_better'] = bool(lower)
        elif entry['lower_is_better'] != bool(lower):
            warnings_out.append(
                f"{tier}/{aug_cond}/{obj}: score direction "
                f"lower_is_better={bool(lower)} disagrees with the family's "
                f"{entry['lower_is_better']} — this search was not read")
            continue

        # ── hybrid configurations ──
        rows = []
        for r in (getattr(search, 'all_results', None) or []):
            if r.get('model') != downstream_model:
                continue
            folds = r.get('fold_scores')
            if not folds:
                continue
            rows.append(r)
        if budget is not None:
            at_budget = [r for r in rows if r.get('budget') == budget]
            if at_budget:
                rows = at_budget
            elif rows:
                # The named budget was not scored, so the ranking falls back to
                # every budget this search holds. A caller that labels its rows
                # by the budget it asked for would then label a hybrid built at
                # some other budget with that number. Say so in the return, not
                # only in the text warning, so the caller can mark the cell.
                budget_relaxed = True
                budgets_ranked_over = sorted(
                    {r.get('budget') for r in rows if r.get('budget') is not None})
                warnings_out.append(
                    f"{tier}/{aug_cond}/{obj}: no configuration was scored at "
                    f"budget={budget} features for model={downstream_model}; "
                    f"ranked over all {len(budgets_ranked_over)} "
                    f"budgets it does hold")
        for r in rows:
            folds = list(r.get('fold_scores') or [])
            n_folds_in_pool = (len(folds) if n_folds_in_pool is None
                               else max(n_folds_in_pool, len(folds)))
            m, n_used = _other_fold_mean(folds, fold_i, drop_folds)
            if m is None:
                continue
            # 'recipe' is the swept configuration alone, the same label
            # build_fold_reselection_input writes. 'name' adds the family and the
            # condition, because the same recipe appears under several of them
            # and they are different feature sets.
            _recipe = "{}|{}|b{}|{}|s{}".format(
                r.get('allocation_method'), r.get('importance_method'),
                r.get('budget'), r.get('internal_evaluator'),
                r.get('step_size'))
            entry['hybrids'].append({
                'objective': obj,
                'tier': tier,
                'aug_condition': aug_cond,
                'recipe': _recipe,
                'name': f"{obj}|{tier}/{aug_cond}|{_recipe}",
                'blueprint': r.get('blueprint'),
                'allocation_method': r.get('allocation_method'),
                'importance_method': r.get('importance_method'),
                'internal_evaluator': r.get('internal_evaluator'),
                'budget': r.get('budget'),
                # How many representations this configuration used (None = the
                # whole pool). Two entries with the same recipe string differ
                # here; the rebuild honours it (restrict_to_rep_count).
                'rep_count': r.get('rep_count'),
                'metric_name': getattr(search, 'metric_name', None),
                'lower_is_better': bool(lower),
                'score_other_folds': float(m),
                'n_other_folds': int(n_used),
                'score_this_fold': (float(folds[fold_i])
                                    if fold_i is not None
                                    and fold_i < len(folds)
                                    and not _hr_is_bad(folds[fold_i])
                                    else None),
                'fold_scores': [(None if _hr_is_bad(s) else float(s))
                                for s in folds],
            })

        # ── single base representations ──
        # search.baseline_fold_scores is {rep: {model: [f0, f1, ...]}}. It is
        # empty on a search pickled before the field existed (2026-07), and
        # without it no base representation can be ranked per fold.
        bfs = getattr(search, 'baseline_fold_scores', None) or {}
        if not bfs:
            warnings_out.append(
                f"{tier}/{aug_cond}/{obj}: the stored search carries no "
                f"per-fold baseline scores, so no base representation could "
                f"be ranked from it")
        for rep_name, per_model in bfs.items():
            if rep_name.startswith('_'):
                continue   # reserved keys, mirrors extract_top3_base_reps
            folds = (per_model or {}).get(downstream_model)
            if not folds:
                continue
            folds = list(folds)
            n_folds_in_pool = (len(folds) if n_folds_in_pool is None
                               else max(n_folds_in_pool, len(folds)))
            m, n_used = _other_fold_mean(folds, fold_i, drop_folds)
            if m is None:
                continue
            cand = {
                'rep': rep_name,
                'objective': obj,
                'tier': tier,
                'aug_condition': aug_cond,
                'metric_name': getattr(search, 'metric_name', None),
                'lower_is_better': bool(lower),
                'score_other_folds': float(m),
                'n_other_folds': int(n_used),
                'score_this_fold': (float(folds[fold_i])
                                    if fold_i is not None
                                    and fold_i < len(folds)
                                    and not _hr_is_bad(folds[fold_i])
                                    else None),
                'fold_scores': [(None if _hr_is_bad(s) else float(s))
                                for s in folds],
            }
            prev = entry['baselines'].get(rep_name)
            # One representation can appear in several pool members with the
            # same scores; keep the better one under the family's direction so
            # the choice is defined rather than order-dependent.
            if prev is None or (
                    (cand['score_other_folds'] < prev['score_other_folds'])
                    if lower else
                    (cand['score_other_folds'] > prev['score_other_folds'])):
                entry['baselines'][rep_name] = cand

    if (fold_i is not None and n_folds_in_pool is not None
            and fold_i >= n_folds_in_pool):
        warnings_out.append(
            f"fold_i={fold_i} but the pool holds {n_folds_in_pool} fold "
            f"scores per configuration — nothing was held out of the ranking")
    if (n_outer_folds is not None and n_folds_in_pool is not None
            and int(n_outer_folds) != n_folds_in_pool):
        warnings_out.append(
            f"the caller is splitting into {int(n_outer_folds)} folds but the "
            f"pool holds {n_folds_in_pool} fold scores per configuration, so "
            f"fold {fold_i} is not the same rows on both sides")

    by_objective = {}
    for obj in objectives:
        entry = fam.get(obj)
        if entry is None:
            continue
        lower = bool(entry['lower_is_better'])
        hyb = sorted(entry['hybrids'],
                     key=lambda h: (h['score_other_folds'] if lower
                                    else -h['score_other_folds']))
        top = []
        for rank, h in enumerate(hyb[:n_hybrids], 1):
            h = dict(h)
            h['rank'] = rank
            bp = h.get('blueprint')
            if bp is not None:
                # Trust the condition that loaded this search over the
                # blueprint's own stored field (2026-04-21 cache-collision
                # fix), on a copy so the cached search is not mutated.
                bp = _copy.deepcopy(bp)
                bp.augmentation_strategy = h['aug_condition']
                h['blueprint'] = bp
            top.append(h)
        base_list = sorted(entry['baselines'].values(),
                           key=lambda b: (b['score_other_folds'] if lower
                                          else -b['score_other_folds']))
        by_objective[obj] = {
            'base_rep': (base_list[0] if base_list else None),
            'base_reps_ranked': base_list,
            'hybrids': top,
            'metric_name': entry['metric_name'],
            'lower_is_better': lower,
            'n_hybrid_candidates': len(entry['hybrids']),
            'n_base_reps': len(base_list),
        }

    present = [o for o in objectives if o in by_objective]
    flat_hybrids = [h for o in present for h in by_objective[o]['hybrids']]
    base_rep = None
    for o in present:
        if by_objective[o]['base_rep'] is not None:
            base_rep = by_objective[o]['base_rep']
            break

    if not flat_hybrids:
        print(f"[pool] WARNING: no hybrid configuration for "
              f"model={downstream_model} budget={budget} fold={fold_i} "
              f"dataset={dataset_name} — the pool held none for subset "
              f"{pool_subset_name!r}; caller will use a fallback.")

    return {
        'fold': fold_i,
        'n_outer_folds': (None if n_outer_folds is None else int(n_outer_folds)),
        'n_folds_in_pool': n_folds_in_pool,
        'model': downstream_model,
        'budget': budget,
        'pool_subset': pool_subset_name,
        'objectives': list(objectives),
        'base_rep': base_rep,
        'hybrids': flat_hybrids,
        'by_objective': by_objective,
        'selection_scope': 'other_folds_of_pool_search',
        'selection_note': (
            'best single representation and top hybrids ranked by their mean '
            "over the pool search's other folds; fold {} contributed nothing "
            'to the ranking. The search itself ran over every row, so this '
            'holds out the fold\'s SCORES, not its rows.'.format(fold_i)),
        'fold0_pruned': fold0_pruned,
        'budget_relaxed': budget_relaxed,
        'budgets_ranked_over': budgets_ranked_over,
        'warnings': warnings_out,
    }


# =============================================================================
# THE OWNER'S HYBRID SET: top 2 general + top 1 distance, per model, per family
# =============================================================================
# Owner, repeatedly and again on 2026-10-02: an experiment that compares
# hybrids compares, for each downstream model, the TOP 2 hybrids from the
# general searches PLUS the TOP 1 hybrid from the distance searches -- and,
# where it reads both search families (features chosen for prediction, and
# features chosen under objective='similarity'), that set for each family.
# The base side is each model's top 2 single representations.
#
# An audit that day found no experiment doing it: a pool read was either the
# general searches or the distance searches (_pool_conditions takes one), and
# most experiments built only the single winner. This is the one reader every
# experiment now goes through.
RULE_N_GENERAL = 2
RULE_N_DISTANCE = 1
RULE_N_BASE = 2


def get_rule_hybrids(
    reps, labels, groups, *, downstream_model, budget=None,
    dataset_name=None, augmentations=None, pool_subset=None, objective=None,
    fold_i=None, n_outer_folds=None,
    n_general=RULE_N_GENERAL, n_distance=RULE_N_DISTANCE, n_base=RULE_N_BASE,
    metric='auto', task_type=None, domain=None, data=None,
    weight_method=None, label_noise=None, drop_folds=None, **kwargs,
):
    """Per family: the top `n_general` general hybrids + top `n_distance`
    distance hybrids for ONE downstream model, and its top `n_base` single
    representations. Reads stored searches; runs none beyond what
    KIRBY_COLD_SEARCH_MAX_CONFIGS lets a missing small search do.

    fold_i=None ranks on the mean over every fold (one set for the whole
    dataset). fold_i=k ranks on the other folds' mean, as get_fold_pool_recipes
    does, so fold k's score never chooses fold k's hybrids.

    knn is scored on the general searches only by the knn pass, which runs
    under objective='similarity' with MAIN_MODELS_WITH_KNN. So for knn the
    general read names that model list; there is no predictive general search
    for knn, and that slot is reported missing rather than filled from another
    model.

    A distance search's budgets stop at max(DISTANCE_BUDGETS). Asked for a
    budget it never scored, get_fold_pool_recipes ranks over every budget it
    holds and sets 'budget_relaxed'; that is carried per entry here.

    Returns:
        {
          'model', 'budget', 'fold', 'pool_subset', 'objectives',
          'by_objective': {family: {
              'hybrids': [entry, ...],   # general ranks 1..n, then distance
              'general': [entry, ...], 'distance': [entry, ...],
              'base_reps': [base entry, ...],     # top n_base, general pool
              'metric_name', 'lower_is_better',
              'missing': [sentence, ...]}},
          'hybrids': [every entry, family order],
          'base_reps': [top n_base of the first family that has any],
          'missing': [sentence, ...],   # every slot the rule asked for and
                                        # did not get, and why
          'warnings': [...],
        }
        Each hybrid entry is a get_fold_pool_recipes entry plus 'lineage'
        ('general' | 'distance'), 'lineage_rank' (1-based), 'slot'
        (e.g. 'general1', 'distance1') and 'budget_relaxed'.
    """
    common = dict(
        fold_i=fold_i, n_outer_folds=n_outer_folds,
        downstream_model=downstream_model, budget=budget,
        dataset_name=dataset_name, augmentations=augmentations,
        pool_subset=pool_subset, objective=objective,
        metric=metric, task_type=task_type, domain=domain, data=data,
        weight_method=weight_method, label_noise=label_noise,
        drop_folds=drop_folds)
    gen_kw = dict(kwargs)
    if (downstream_model not in DOWNSTREAM_MODELS
            and downstream_model in MAIN_MODELS_WITH_KNN
            and 'models' not in gen_kw):
        gen_kw['models'] = list(MAIN_MODELS_WITH_KNN)
    out_parts = {}
    errors = {}
    for lineage, dist, n in (('general', False, n_general),
                             ('distance', True, n_distance)):
        if n <= 0:
            continue
        try:
            out_parts[lineage] = get_fold_pool_recipes(
                reps, labels, groups, distance=dist, n_hybrids=n,
                **common, **(gen_kw if not dist else dict(kwargs)))
        except Exception as exc:
            from kirby.result_cache import StaleResultDB as _Stale
            if isinstance(exc, _Stale):
                raise
            errors[lineage] = f'{type(exc).__name__}: {exc}'

    objectives, subset_name = _resolve_pool_objectives(pool_subset, objective)
    by_objective, missing_all, warnings_all = {}, [], []
    for lineage, part in out_parts.items():
        warnings_all += [f'{lineage}: {w}' for w in part.get('warnings', [])]
    for fam in objectives:
        fam_out = {'hybrids': [], 'general': [], 'distance': [],
                   'base_reps': [], 'metric_name': None,
                   'lower_is_better': None, 'missing': []}
        for lineage, n in (('general', n_general), ('distance', n_distance)):
            if n <= 0:
                continue
            part = out_parts.get(lineage)
            if part is None:
                fam_out['missing'].append(
                    f'{lineage} {fam}: the pool read raised '
                    f'{errors.get(lineage)}, so none of its {n} slot(s) '
                    f'was filled')
                continue
            e = (part.get('by_objective') or {}).get(fam)
            hy = list((e or {}).get('hybrids') or [])
            for h in hy:
                h = dict(h)
                h['lineage'] = lineage
                h['lineage_rank'] = int(h.get('rank', 0) or 0)
                h['slot'] = f"{lineage}{h['lineage_rank']}"
                h['budget_relaxed'] = bool(part.get('budget_relaxed'))
                fam_out[lineage].append(h)
            if len(hy) < n:
                why = ('the pool held no search under this family for this '
                       'tier' if e is None else
                       f'only {len(hy)} configuration(s) were scored for '
                       f'{downstream_model}')
                fam_out['missing'].append(
                    f'{lineage} {fam} for {downstream_model}: asked for {n}, '
                    f'got {len(hy)} -- {why}')
            if e is not None:
                if fam_out['metric_name'] is None:
                    fam_out['metric_name'] = e.get('metric_name')
                    fam_out['lower_is_better'] = e.get('lower_is_better')
                if lineage == 'general':
                    fam_out['base_reps'] = list(
                        e.get('base_reps_ranked') or [])[:n_base]
        fam_out['hybrids'] = fam_out['general'] + fam_out['distance']
        by_objective[fam] = fam_out
        missing_all += fam_out['missing']

    base_reps = []
    for fam in objectives:
        if by_objective.get(fam, {}).get('base_reps'):
            base_reps = by_objective[fam]['base_reps']
            break
    if len(base_reps) < n_base:
        missing_all.append(
            f'base representations for {downstream_model}: asked for '
            f'{n_base}, got {len(base_reps)}')
    if missing_all:
        print(f"[rule] {dataset_name} model={downstream_model} "
              f"budget={budget} fold={fold_i}: "
              + ' | '.join(missing_all))
    # Carried from the two reads so a caller can tell whether fold 0 filtered
    # the candidates and how many fold scores the pool holds, as it could
    # from get_fold_pool_recipes.
    _f0 = [p.get('fold0_pruned') for p in out_parts.values()]
    _nf = [p.get('n_folds_in_pool') for p in out_parts.values()
           if p.get('n_folds_in_pool') is not None]
    return {
        'fold0_pruned': (None if not _f0 or all(v is None for v in _f0)
                         else any(bool(v) for v in _f0 if v is not None)),
        'fold0_pruned_by_lineage': {k: p.get('fold0_pruned')
                                    for k, p in out_parts.items()},
        'n_folds_in_pool': (max(_nf) if _nf else None),
        'model': downstream_model, 'budget': budget, 'fold': fold_i,
        'pool_subset': subset_name, 'objectives': list(objectives),
        'by_objective': by_objective,
        'hybrids': [h for f in objectives
                    for h in by_objective.get(f, {}).get('hybrids', [])],
        'base_reps': base_reps,
        'missing': missing_all,
        'warnings': warnings_all,
        'rule': (f'top {n_general} general + top {n_distance} distance '
                 f'hybrids per family, top {n_base} single representations'),
    }


def build_hybrid_from_entry(reps, labels, groups, entry, *,
                            augmentations=None):
    """Fit one hybrid from a get_rule_hybrids / get_fold_pool_recipes entry on
    the rows given. Returns (X_hybrid, feature_info).

    Same create_hybrid arguments get_fitted_pool_hybrid passes (every recipe
    setting, the grouping, and the family's own objective), so a recipe is
    rebuilt the way the search scored it. To build on a training fold, pass
    that fold's rows; apply the returned feature_info to the held-out rows with
    hybrid.apply_feature_selection.
    """
    bp = entry['blueprint']
    aug_strat = entry.get('aug_condition') or getattr(
        bp, 'augmentation_strategy', 'none')
    return _create_hybrid_from_blueprint(
        reps, labels, groups, bp, aug_strat,
        augmentations if aug_strat != 'none' else None,
        entry.get('objective'))


def _create_hybrid_from_blueprint(reps, labels, groups, bp, aug_strat, augs,
                                  objective):
    from kirby.hybrid import create_hybrid
    from kirby.find_best_hybrid import restrict_to_rep_count
    # Only the representations the recipe was searched over, and only as many
    # as it used (rep_count). Both were ignored by the whole-data builders.
    _bp_reps = getattr(bp, 'reps', None)
    if _bp_reps:
        _sub = {k: reps[k] for k in _bp_reps if k in reps}
        if _sub:
            reps = _sub
    reps, _rc_note = restrict_to_rep_count(
        reps, bp, context='build_hybrid_from_entry')
    X_h, fi = create_hybrid(
        reps, labels,
        allocation_method=bp.allocation_method,
        budget=bp.budget,
        step_size=getattr(bp, 'step_size', 5),
        importance_method=bp.importance_method,
        internal_evaluator=bp.internal_evaluator,
        augmentations=augs,
        augmentation_strategy=aug_strat,
        augmentation_margin=float(
            getattr(bp, 'augmentation_margin', 0.0) or 0.0),
        augmentation_fraction=getattr(bp, 'augmentation_fraction', None),
        candidate_ranking=getattr(bp, 'candidate_ranking',
                                  'residual_correlation'),
        augmentation_threshold=getattr(bp, 'augmentation_threshold', None),
        augmentation_budget=int(getattr(bp, 'augmentation_budget', 20) or 20),
        aug_max_candidates=int(getattr(bp, 'aug_max_candidates', 50) or 50),
        aug_per_source_budget=getattr(bp, 'aug_per_source_budget', None),
        groups=groups,
        **({'objective': objective} if objective is not None else {}),
    )
    if _rc_note and isinstance(fi, dict):
        fi.update(_rc_note)
    return X_h, fi


def get_fitted_pool_hybrid(
    reps, labels, groups, *, downstream_model, budget,
    augmentations=None, data=None,
    metric='auto', task_type=None, domain=None,
    dataset_name=None, distance=False, objective=None, **kwargs,
):
    """Pool-best variant of get_fitted_hybrid.

    Picks the best precomputed SearchResult from the pool for
    (downstream_model, budget) via get_best_search_for_downstream, then
    builds X_hybrid on full data using the winning blueprint.

    Returns:
        (X_hybrid, feature_info, search, metadata) where metadata is the
        dict returned by get_best_search_for_downstream (tier/aug_condition
        of the winner). Raises RuntimeError if no candidate produced a
        usable result.
    """
    from kirby.hybrid import create_hybrid

    search, meta = get_best_search_for_downstream(
        reps, labels, groups,
        downstream_model=downstream_model, budget=budget,
        augmentations=augmentations,
        dataset_name=dataset_name, distance=distance,
        data=data, return_fitted=True, objective=objective, **kwargs,
    )
    if not meta:
        raise RuntimeError(
            f"get_fitted_pool_hybrid: no pool candidate produced a result for "
            f"model={downstream_model} budget={budget} dataset={dataset_name}"
        )

    bp = meta['best_config']['blueprint']
    # Trust meta['aug_condition'] (the condition that loaded this search), not
    # the blueprint's stored strategy (2026-04-21 cache-collision bug). The
    # rebuild forwards every recipe setting, the grouping, the winner's own
    # objective, and -- since 2026-10-02 -- only the representations the
    # recipe was searched over (rep_count), via _create_hybrid_from_blueprint.
    aug_strat = meta['aug_condition']
    augs = augmentations if aug_strat != 'none' else None
    X_hybrid, fi = _create_hybrid_from_blueprint(
        reps, labels, groups, bp, aug_strat, augs,
        meta.get('objective', objective))
    return X_hybrid, fi, search, meta


def get_fitted_hybrid(reps, labels, groups, *, downstream_model,
                      budget=None, aug_condition='uncapped',
                      augmentations=None, precomputed=None,
                      internal_evaluators=None, budgets=None,
                      metric='auto', task_type=None, domain=None,
                      dataset_name=None, tier='general',
                      adaptive_pruning_schedule=None,
                      model_pruning_aggressiveness=None,
                      objective=None,
                      **extra_kwargs):
    """Get X_hybrid and feature_info from a cached search (return_fitted=True).

    For full-dataset experiments that need the hybrid matrix without per-fold splitting.
    Each downstream model gets its own best hybrid — downstream_model is required.

    Args:
        reps: Base representations {name: array}.
        labels: Target values.
        groups: Group labels for CV (or None).
        downstream_model: Model name (e.g., 'rf', 'knn', 'gp'). Required.
            Selects from best_by_model_budget[downstream_model].
        budget: Specific budget to select (None = best across budgets for this model).
        aug_condition: 'none', 'uncapped', or 'capped'.
        augmentations: {aug_name: array} or None.
        speed: Search speed tier ('fast' or 'full').
        precomputed: Dict from data['precomputed_searches'] (optional).
        internal_evaluators: Override evaluator list (e.g., DISTANCE_EVALUATORS).
            When set, bypasses precomputed dict and runs on-demand (disk-cached).
        budgets: Override budget list (e.g., DISTANCE_BUDGETS).
            When set, bypasses precomputed dict and runs on-demand (disk-cached).
        metric: Metric name ('auto' resolves from domain/task_type).
        task_type: REQUIRED. 'classification' or 'regression'. Pass
            `data['task_type']` from load_data — no auto-detect.
        domain: Domain hint for metric resolution.

    Returns:
        (X_hybrid, feature_info, SearchResult)
    """
    # If custom evaluators, budgets, or pruning schedule requested, bypass
    # the in-memory precomputed dict (it was computed with different params).
    # Disk cache handles dedup based on full meta.
    kwargs = dict(extra_kwargs)
    use_precomputed = precomputed
    if internal_evaluators is not None:
        kwargs['internal_evaluators'] = internal_evaluators
        use_precomputed = None  # force on-demand search
    if budgets is not None:
        kwargs['budgets'] = budgets
        use_precomputed = None
    if adaptive_pruning_schedule is not None:
        kwargs['adaptive_pruning_schedule'] = adaptive_pruning_schedule
        use_precomputed = None  # different schedule = different cache key
    if model_pruning_aggressiveness is not None:
        kwargs['model_pruning_aggressiveness'] = model_pruning_aggressiveness
        use_precomputed = None
    if objective is not None:
        kwargs['objective'] = objective
        use_precomputed = None  # geometry/predictive both resolve from disk here

    # NOTE: `use_precomputed` is computed above but deliberately NOT passed —
    # this call has never forwarded it, so get_precomputed_search always takes
    # its precomputed=None default and resolves from the on-disk cache. Left as
    # found: wiring it up now would change which search every existing caller
    # gets. Flagged, not fixed.
    search = get_precomputed_search(
        reps, labels, groups,
        aug_condition=aug_condition, augmentations=augmentations,
        metric=metric, task_type=task_type, domain=domain,
        dataset_name=dataset_name, tier=tier,
        **kwargs)

    best_config = get_best_for_downstream(search, downstream_model, budget=budget)

    if best_config is not None and 'blueprint' in best_config:
        # Build hybrid from the selected blueprint on full data
        from kirby.hybrid import create_hybrid
        bp = best_config['blueprint']
        augs = augmentations if augmentations is not None else None
        X_hybrid, fi = create_hybrid(
            reps, labels,
            allocation_method=bp.allocation_method,
            budget=bp.budget,
            step_size=getattr(bp, 'step_size', 5),
            importance_method=bp.importance_method,
            internal_evaluator=bp.internal_evaluator,
            augmentations=augs,
            augmentation_strategy=getattr(bp, 'augmentation_strategy', 'uncapped'),
            # And the rest of the recipe, for the reason written out in
            # get_fitted_pool_hybrid: five of these decide whether a block of
            # extra features is kept at all, so leaving them at create_hybrid's
            # defaults can rebuild a different feature set from the one that
            # won. Both doors now forward what build_fold_hybrid forwards.
            augmentation_margin=float(
                getattr(bp, 'augmentation_margin', 0.0) or 0.0),
            augmentation_fraction=getattr(bp, 'augmentation_fraction', None),
            candidate_ranking=getattr(bp, 'candidate_ranking',
                                      'residual_correlation'),
            augmentation_threshold=getattr(bp, 'augmentation_threshold', None),
            augmentation_budget=int(
                getattr(bp, 'augmentation_budget', 20) or 20),
            aug_max_candidates=int(
                getattr(bp, 'aug_max_candidates', 50) or 50),
            aug_per_source_budget=getattr(bp, 'aug_per_source_budget', None),
            # Same as get_fitted_pool_hybrid: keep the full-data rebuild on the
            # same grouping the search used for its internal split.
            groups=groups,
            # And, same as get_fitted_pool_hybrid, rebuild under the objective
            # the search selected under. This function accepted `objective`,
            # used it to pick the search, and then dropped it here, so a
            # geometry-selected recipe was rebuilt with a predictively split
            # budget. Its sibling fifty lines above has forwarded it since the
            # geometry objectives went in; this copy was missed. No stored
            # result moves: `get_fitted_hybrid` is imported by nine experiment
            # files and called by none of them.
            **({'objective': objective} if objective is not None else {}),
        )
        # NOTE: create_hybrid already includes augmentation features in X_hybrid
        # when augmentations are provided — do NOT append them again.
        return X_hybrid, fi, search

    # Fallback: use per-model fitted_results from search
    fitted = search.fitted_results.get(downstream_model)
    if fitted is None:
        available = list(search.fitted_results.keys()) if search.fitted_results else []
        raise RuntimeError(
            f"No fitted result for model={downstream_model}, aug={aug_condition}. "
            f"Available models: {available}")
    fi = {}
    for rep_name, indices in fitted.selected_indices.items():
        fi[rep_name] = {
            'selected_indices': indices,
            'n_selected': len(indices),
        }
        if fitted.feature_weights and rep_name in fitted.feature_weights:
            fi[rep_name]['feature_weights'] = fitted.feature_weights[rep_name]
    X_hybrid = apply_feature_selection(reps, fi)

    # Include augmentation features if present
    if fitted.augmentation_info and augmentations is not None:
        from kirby.hybrid import apply_augmentation_selection
        fi['augmentation_info'] = fitted.augmentation_info
        X_aug = apply_augmentation_selection(augmentations, fitted.augmentation_info)
        if X_aug is not None:
            X_hybrid = np.hstack([X_hybrid, X_aug])

    return X_hybrid, fi, search


def get_best_for_downstream(search, downstream_model, budget=None):
    """Select the best config from a SearchResult for a specific downstream model.

    Lookup chain:
    1. best_by_model_budget[model][budget] (exact match)
    2. best across all budgets for this model in best_by_model_budget
    3. best_by_budget[budget] (model-agnostic, budget-specific)
    4. best_by_model[model] (model-specific, budget-agnostic)
    5. None — caller must handle.

    Args:
        search: SearchResult from find_best_hybrid.
        downstream_model: Model name (e.g., 'knn', 'gp', 'svm').
        budget: Specific budget to select (None = best across budgets for this model).

    Returns:
        best_config dict with 'blueprint', 'score_mean', etc. or None.
    """
    # Filter entries to those with valid (finite, non-None) score_mean.
    # Previously the key= used `r.get('score_mean', 0)` which silently
    # treated missing scores as 0 — for lower-is-better metrics (RMSE etc.)
    # 0 is always the min, so a broken/missing entry would win. That's the
    # 8-week silent-wrong-result bug.
    def _valid(entries):
        return [r for r in entries
                if r.get('score_mean') is not None
                and np.isfinite(r['score_mean'])]

    # Try model-specific budget lookup
    if search.best_by_model_budget and downstream_model in search.best_by_model_budget:
        model_budgets = search.best_by_model_budget[downstream_model]
        if budget is not None and budget in model_budgets:
            entry = model_budgets[budget]
            if entry.get('score_mean') is not None and np.isfinite(entry['score_mean']):
                return entry
            # Fall through to "best across budgets" if the requested one is broken.
        # Best across all budgets for this model
        valid = _valid(model_budgets.values())
        if valid:
            lower = object_direction(search, 'the SearchResult passed to '
                                     'get_best_for_downstream')
            return (min if lower else max)(valid, key=lambda r: r['score_mean'])

    # Fallback: budget-specific (model-agnostic)
    if budget is not None and search.best_by_budget and budget in search.best_by_budget:
        entry = search.best_by_budget[budget]
        if entry.get('score_mean') is not None and np.isfinite(entry['score_mean']):
            return entry

    # Fallback: model-specific (budget-agnostic)
    if search.best_by_model and downstream_model in search.best_by_model:
        entry = search.best_by_model[downstream_model]
        if entry.get('score_mean') is not None and np.isfinite(entry['score_mean']):
            return entry

    # Cross-model fallback: use overall best blueprint from any model.
    # The blueprint (allocation/importance/budget) is largely model-agnostic —
    # create_hybrid builds the same hybrid regardless of downstream model.
    if search.best_by_model:
        valid = _valid(search.best_by_model.values())
        if valid:
            lower = object_direction(search, 'the SearchResult passed to '
                                     'get_best_for_downstream')
            return (min if lower else max)(valid, key=lambda r: r['score_mean'])

    return None


# =============================================================================
# SHARED TOP-N SELECTION
#
# The shape the owner asked for: for each downstream model, the top 2 base
# representations and the top 2 hybrids, so 5 downstream models * (2 + 2) = 20
# model-and-representation pairs. The 2026-09-18 compute audit read all 77
# experiment files and found no file producing that shape, and found 27 distinct
# ranking rules in use, 22 of them written inside one experiment file.
#
# All four functions below are pure lookups over a SearchResult that is already
# in hand. They cost 0 new model fits: every number they read was written by the
# search that produced the SearchResult. They never call find_best_hybrid, never
# build an estimator, never open the result database. If a ranking cannot be
# answered from the stored rows the function raises (require_full=True, the
# default) or reports the shortfall with its counts.
#
# The one cost that is NOT zero sits one step earlier and belongs to the caller:
# there is no SearchResult on disk (find_best_hybrid.py:9150-9153), so a caller
# holding none has to call find_best_hybrid(cache=True). On a fully warm dataset
# that is 0 fits; on a cold one it is the whole search.
# =============================================================================

# What makes two hybrid rows the same configuration. 'tune' is in here on
# purpose: it is stamped on a freshly computed row (find_best_hybrid.py:5175)
# and is NOT restored by _db_row_to_result (:5839-5883), so leaving it out lets
# a tuned and an untuned run of the same 7 other fields collapse into one entry
# with whichever score sorted better.
_HYBRID_IDENTITY_FIELDS = ('allocation_method', 'importance_method', 'budget',
                           'internal_evaluator', 'step_size', 'rep_count',
                           'rep_subset', 'tune')


def _sel_context(context):
    """Trailing ' [<context>]' for an error message, or ''."""
    return f" [{context}]" if context else ""


def _sel_base_direction(search, context=None):
    """Direction of the numbers in search.baselines, plus where it came from.

    Base representation scores stay in the DATASET's own metric, so the metric
    name is authoritative: object_direction reads BUILTIN_METRICS by name first
    and falls back to the stored flag only when the name is not a builtin
    (_shared.py:1531). find_best_hybrid.py:8717 keeps the baselines at the
    dataset metric's direction while :8726 overwrites SearchResult
    .lower_is_better to False under a blend objective containing prediction,
    because the winner is then scored on a correlation -- so on an rmse dataset
    under such a blend the stored flag reads False while baselines still hold
    rmse, and sorting by the stored flag puts the WORST representation first.
    c02_complexity_grid.py:257 has exactly that shape today.

    'disagrees_with_stored_flag' is the only visible signature of a blend
    search: SearchResult carries no `objective` field, so nothing else in the
    stored object says which case it is in. UNVERIFIED whether any submit script
    in this repo currently passes a blend objective.
    """
    lower = object_direction(search, f'the SearchResult passed to a top-N '
                                     f'selector{_sel_context(context)}')
    name = getattr(search, 'metric_name', None)
    from kirby.find_best_hybrid import BUILTIN_METRICS
    source = 'metric_name' if name in BUILTIN_METRICS else 'stored_flag'
    stored = getattr(search, 'lower_is_better', None)
    return {
        'lower_is_better': bool(lower),
        'source': source,
        'disagrees_with_stored_flag': (stored is not None
                                       and bool(stored) != bool(lower)),
    }


def _sel_hybrid_direction(search, context=None):
    """Direction of the score_mean values in search.all_results.

    NOT object_direction, and this is the one place in these four functions
    where the metric name must not win. _aggregate_results ranks all_results
    with _score_to_comparable(r['score_mean'], lower_is_better) -- so the stored
    flag IS the direction of those numbers by construction, whatever the metric
    name says. Under a blend objective containing prediction, find_best_hybrid
    overwrites lower_is_better to False (:8726) while metric_name stays the
    dataset metric string, so resolving from the name here would hand back the
    worst hybrid. That is the mirror image of the defect the 2026-09-10
    metric-direction audit fixed on the baselines side.

    There is no "raise when the flag is None" branch, because the dataclass
    default is `lower_is_better: bool = False` (find_best_hybrid.py:1537): a
    SearchResult that never set it reads False rather than None, so such a
    branch would be dead code. The disagreement with the metric name is
    recorded instead.
    """
    stored = getattr(search, 'lower_is_better', None)
    if stored is None:
        raise ValueError(
            f"the SearchResult carries lower_is_better=None, so the direction "
            f"of its all_results scores is unknown{_sel_context(context)}")
    try:
        by_name = object_direction(search, 'a SearchResult')
    except ValueError:
        by_name = None
    return {
        'lower_is_better': bool(stored),
        'source': 'stored_flag',
        'disagrees_with_metric_name': (by_name is not None
                                       and bool(by_name) != bool(stored)),
    }


def _sel_tie_at_cut(ranked, n, name_of):
    """{'score', 'kept', 'dropped'} when the cut at n fell inside a tie, else None.

    A tie at the cut means the last slot was decided by the name tie-break and
    not by a measurement. Baseline scores are rounded to 4 decimals at
    find_best_hybrid.py:5524, which manufactures exact ties, so this is not a
    theoretical case.
    """
    if n <= 0 or len(ranked) <= n:
        return None
    edge = ranked[n - 1][0]
    if ranked[n][0] != edge:
        return None
    return {
        'score': edge,
        'kept': [name_of(e) for e in ranked[:n] if e[0] == edge],
        'dropped': [name_of(e) for e in ranked[n:] if e[0] == edge],
    }


def top_base_reps(search, downstream_model, n=2, *,
                  rule='whole_dataset', held_out_fold=None, drop_folds=(),
                  only=None, exclude_reps=(), max_features=None,
                  require_full=True, context=None):
    """Top `n` base representations for ONE downstream model. 0 new model fits.

    `downstream_model` is required and has no default. extract_top3_base_reps
    (_shared.py:809) defaults to 'rf' and 24 of its 26 caller files take that
    default, so those files report one ranking by one downstream model and then
    compare it against hybrids chosen for other downstream models. This one
    makes the caller name the model. It is a DOWNSTREAM model -- one of
    DOWNSTREAM_MODELS (_shared.py:539: rf, xgboost, svm, dnn, lightgbm), the
    model scored at the end -- never an internal evaluator, which is the model
    used inside the search to pick features (INTERNAL_EVALUATORS, :806).

    rule='whole_dataset' reads search.baselines[rep][downstream_model]: one
    number per representation, the search's own mean over its folds
    (find_best_hybrid.py:5524-5525). A representation picked this way was picked
    with the held-out molecules in view, so reporting that same number as a
    result is selection-optimistic.
    rule='other_folds' reads search.baseline_fold_scores[rep][downstream_model]
    (find_best_hybrid.py:1561) and means over every fold except `held_out_fold`,
    which removes that optimism. The return says which rule produced the numbers
    in 'score_basis'; no results file in the repo records that today.

    max_features bars a representation wider than the limit from selection while
    still measuring and reporting it, with its feature count written into the
    return. That is n07_remediation.py:494-499's rule made shared.

    require_full=True (the default) raises ValueError when fewer than `n`
    representations carry a usable score. The error names the dataset, the
    downstream model, n asked, n returned, how many representations were seen,
    how many were scored, and the per-reason drop counts.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}{_sel_context(context)}")
    if rule not in ('whole_dataset', 'other_folds'):
        raise ValueError(f"rule must be 'whole_dataset' or 'other_folds', got "
                         f"{rule!r}{_sel_context(context)}")
    if rule == 'other_folds' and held_out_fold is None:
        raise ValueError(f"rule='other_folds' needs held_out_fold: the fold "
                         f"being predicted, which is left out of the mean"
                         f"{_sel_context(context)}")

    # Imported rather than re-listed, so a fourth non-score key added at
    # find_best_hybrid.py:2370 is excluded here with one edit and not two.
    # Currently ('dimension', 'dimension_scored', '_secondary'); the first two
    # hold FEATURE COUNTS, so a caller naming one of them as the downstream
    # model would rank representations by width.
    from kirby.find_best_hybrid import _BASELINE_NON_SCORE_KEYS
    if downstream_model in _BASELINE_NON_SCORE_KEYS or \
            str(downstream_model).startswith('_'):
        raise ValueError(
            f"{downstream_model!r} is not a downstream model: it is one of the "
            f"non-score keys stored inside baselines[rep] "
            f"({', '.join(_BASELINE_NON_SCORE_KEYS)}), and two of those are "
            f"feature counts{_sel_context(context)}")

    direction = _sel_base_direction(search, context)
    lower = direction['lower_is_better']
    baselines = getattr(search, 'baselines', {}) or {}
    fold_scores_all = getattr(search, 'baseline_fold_scores', {}) or {}
    only_set = None if only is None else {str(r) for r in only}
    excl_set = {str(r) for r in (exclude_reps or ())}

    dropped = {'no_score_for_model': [], 'nan': [], 'underscore_key': [],
               'excluded_by_caller': [], 'not_in_only': [],
               'too_wide': [], 'no_fold_scores': []}
    ranked = []          # list of (sort_score, rep_name, entry_dict)
    n_scored = 0

    for rep_name, model_scores in baselines.items():
        if str(rep_name).startswith('_'):
            # '_secondary' and friends are not representations.
            dropped['underscore_key'].append(rep_name)
            continue
        if not isinstance(model_scores, dict):
            dropped['no_score_for_model'].append(rep_name)
            continue

        # The three non-score keys inside baselines[rep] -- 'dimension',
        # 'dimension_scored', '_secondary' (_BASELINE_NON_SCORE_KEYS,
        # find_best_hybrid.py:2370) -- can never enter a ranking here, because
        # this reads one named key and never iterates the dict. 'dimension' and
        # 'dimension_scored' are FEATURE COUNTS: iterating picks up 500 features
        # as if it were a score, which beats every r2 under max().
        dim = model_scores.get('dimension')
        dim_scored = model_scores.get('dimension_scored')

        if rule == 'whole_dataset':
            raw = model_scores.get(downstream_model)
            n_folds_used = None
        else:
            per_fold = (fold_scores_all.get(rep_name) or {}).get(downstream_model)
            if per_fold is None:
                dropped['no_fold_scores'].append(rep_name)
                continue
            raw, n_folds_used = _other_fold_mean(per_fold, held_out_fold,
                                                 drop_folds)

        if raw is None:
            # A downstream model over its _max_samples limit
            # (find_best_hybrid.py:2969-2978) or in a registered skip policy
            # (:2841-2848) has no key under ANY representation, so this list
            # comes back holding every representation and n_reps_scored is 0.
            dropped['no_score_for_model'].append(rep_name)
            continue
        try:
            score = float(raw)
        except (TypeError, ValueError):
            dropped['nan'].append(rep_name)
            continue
        if not np.isfinite(score):
            # A NaN compares False against everything and wins max() outright
            # (find_best_hybrid.py:7412-7416), so it never reaches the sort.
            dropped['nan'].append(rep_name)
            continue

        n_scored += 1  # it HAS a usable score; the bars below are selection bars

        if only_set is not None and rep_name not in only_set:
            dropped['not_in_only'].append(rep_name)
            continue
        if rep_name in excl_set:
            dropped['excluded_by_caller'].append(rep_name)
            continue
        if max_features is not None and dim is not None and dim > max_features:
            # Measured and reported, barred from selection, feature count kept.
            dropped['too_wide'].append((rep_name, int(dim)))
            continue

        ranked.append((score, str(rep_name), {
            'rep': rep_name,
            'score': score,
            'dimension': (int(dim) if dim is not None else None),
            'dimension_scored': (int(dim_scored) if dim_scored is not None
                                 else None),
            # A representation wider than HIGH_DIM_FILTER_THRESHOLD is scored on
            # 500 features (200 under 1000 molecules) by a per-fold LightGBM
            # filter, and the width actually scored is recorded beside it
            # (find_best_hybrid.py:5355-5365). The score itself carries no mark.
            'scored_at_full_width': (dim_scored is None),
            'n_folds_used': n_folds_used,
        }))

    # Sort key is a total order: the signed score, then the representation name
    # ascending. No dict insertion order anywhere -- for baselines that order is
    # the representation BUILD order, which differs between a fresh search and
    # one whose per-representation cache files were warm, so it is not
    # reproducible across two runs. The name is.
    ranked.sort(key=lambda e: ((e[0] if lower else -e[0]), e[1]))

    tie_at_cut = _sel_tie_at_cut(ranked, n, lambda e: e[1])
    selected = []
    for i, (score, rep_name, entry) in enumerate(ranked[:n]):
        entry = dict(entry)
        entry['rank'] = i + 1
        entry['tied_with'] = [o[1] for o in ranked
                              if o[0] == score and o[1] != rep_name]
        selected.append(entry)

    basis = ('whole_dataset_mean' if rule == 'whole_dataset'
             else f'other_folds_mean(held_out={held_out_fold})')
    out = {
        'kind': 'base_rep',
        'downstream_model': downstream_model,
        'dataset_name': getattr(search, 'dataset_name', None),
        'metric_name': getattr(search, 'metric_name', None),
        'direction': direction,
        'score_basis': basis,
        'ranked_on': ("the search's own mean over its folds, one number per "
                      "representation -- the same folds the selection is then "
                      "compared on"
                      if rule == 'whole_dataset' else
                      "the mean over every fold except the held-out one"),
        'n_requested': n,
        'n_returned': len(selected),
        'complete': len(selected) >= n,
        'selected': selected,
        'n_reps_in_baselines': len(baselines),
        'n_reps_scored_for_model': n_scored,
        'dropped': dropped,
        'tie_at_cut': tie_at_cut,
        'shortfall_reason': None,
        'context': context,
    }
    if not out['complete']:
        out['shortfall_reason'] = (
            f"dataset {out['dataset_name']}: asked for {n} base "
            f"representations for downstream model {downstream_model}, got "
            f"{len(selected)}; {len(baselines)} representations in baselines, "
            f"{n_scored} carry a usable score for that model; dropped "
            f"{ {k: len(v) for k, v in dropped.items() if v} }"
            f"{_sel_context(context)}")
        if require_full:
            raise ValueError(out['shortfall_reason'])
    return out


def top_hybrids(search, downstream_model, n=2, *, budget=None,
                rule='whole_dataset', held_out_fold=None, drop_folds=(),
                require_n_folds=None, require_full=True, context=None):
    """Top `n` hybrid configurations for ONE downstream model. 0 new model fits.

    Reads search.all_results (find_best_hybrid.py:1536) and nothing else. It
    does NOT read best_by_model, best_by_model_budget, best_by_budget,
    best_by_dimension or performance_matrix: each of those holds exactly one row
    per key, so none of them can answer a top-2 question.

    There is no fallback to another downstream model, at any point.
    get_best_for_downstream (_shared.py:6475) has a five-level chain whose last
    level returns the best blueprint from ANY downstream model (:6532-6539), so
    a top-2 built out of two calls to it can put another model's configuration
    inside a list labelled for this one, without saying so.

    What this cannot see: all_results holds only the configurations the search
    actually evaluated. Adaptive pruning drops configurations at each checkpoint
    budget and again at the end of fold 0 (find_best_hybrid.py:5975-5990,
    :6990-7000), and the cut-after-fold-0 rows are filed under a rep_subset
    prefix chosen so no search matches them (PRUNED_FOLD0_SUBSET, :5708, :7030).
    "Top 2 hybrids" is top 2 of the survivors, not top 2 of the grid.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}{_sel_context(context)}")
    if rule not in ('whole_dataset', 'other_folds'):
        raise ValueError(f"rule must be 'whole_dataset' or 'other_folds', got "
                         f"{rule!r}{_sel_context(context)}")
    if rule == 'other_folds' and held_out_fold is None:
        raise ValueError(f"rule='other_folds' needs held_out_fold"
                         f"{_sel_context(context)}")

    from kirby.find_best_hybrid import PRUNED_FOLD0_SUBSET

    direction = _sel_hybrid_direction(search, context)
    lower = direction['lower_is_better']
    rows = list(getattr(search, 'all_results', None) or [])

    counts = {'n_rows_total': len(rows), 'n_rows_for_model': 0,
              'n_rows_other_model': 0, 'n_rows_dropped_nan': 0,
              'n_rows_dropped_pruned_fold0': 0,
              'n_rows_dropped_wrong_budget': 0,
              'n_rows_dropped_wrong_fold_count': 0,
              'n_rows_missing_tune': 0,
              'n_duplicate_identities_collapsed': 0}

    by_identity = {}   # identity tuple -> (score, ident_str, entry, position)
    for pos, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        if row.get('model') != downstream_model:
            counts['n_rows_other_model'] += 1
            continue
        sub = row.get('rep_subset')
        if isinstance(sub, str) and sub.startswith(PRUNED_FOLD0_SUBSET):
            # These carry ONE fold score and are deliberately filed so no search
            # can match them (find_best_hybrid.py:7031-7043).
            counts['n_rows_dropped_pruned_fold0'] += 1
            continue
        counts['n_rows_for_model'] += 1
        if budget is not None and row.get('budget') != budget:
            counts['n_rows_dropped_wrong_budget'] += 1
            continue

        if rule == 'whole_dataset':
            raw = row.get('score_mean')
            n_folds_used = row.get('n_folds_scored')
        else:
            raw, n_folds_used = _other_fold_mean(row.get('fold_scores'),
                                                 held_out_fold, drop_folds)
        try:
            score = float(raw)
        except (TypeError, ValueError):
            counts['n_rows_dropped_nan'] += 1
            continue
        if not np.isfinite(score):
            # A NaN row STAYS in all_results: the _finite filter at
            # find_best_hybrid.py:7404-7421 restricts only the argmax views, so
            # this function has to drop it itself.
            counts['n_rows_dropped_nan'] += 1
            continue

        n_scored = row.get('n_folds_scored')
        if require_n_folds is not None and n_scored != require_n_folds:
            # Two configurations compared on different fold counts are not
            # comparable; c14_cross_domain_summary.py:157 does exactly that.
            counts['n_rows_dropped_wrong_fold_count'] += 1
            continue

        # rep_count is NOT restored to the top level by _db_row_to_result
        # (find_best_hybrid.py:5839-5883); it survives only inside the pickled
        # Blueprint (Blueprint.rep_count, :1436). A freshly computed row carries
        # it (:7033). rep_count_source makes a mixed-provenance ranking visible.
        bp = row.get('blueprint')
        if 'rep_count' in row:
            rep_count, rc_source = row.get('rep_count'), 'row'
        elif getattr(bp, 'rep_count', None) is not None:
            rep_count, rc_source = getattr(bp, 'rep_count'), 'blueprint'
        else:
            rep_count, rc_source = None, 'unknown'

        # 'tune' is stamped on a freshly computed row (:5175) and absent from a
        # database-read one. It is part of the identity, so a tuned and an
        # untuned run of the same 7 other fields do not collapse into one.
        if 'tune' in row:
            tune, tune_source = bool(row.get('tune')), 'row'
        else:
            tune, tune_source = None, 'missing'
            counts['n_rows_missing_tune'] += 1

        ident = (row.get('allocation_method'), row.get('importance_method'),
                 row.get('budget'), row.get('internal_evaluator'),
                 row.get('step_size'), rep_count,
                 row.get('rep_subset', 'all'), tune)
        ident_str = '/'.join('None' if v is None else str(v) for v in ident)

        entry = {
            'score_mean': score,
            'score_std': row.get('score_std'),
            'fold_scores': row.get('fold_scores'),
            'n_folds_scored': n_scored,
            'n_folds_asked': row.get('n_folds_asked'),
            'n_folds_used': n_folds_used,
            'n_features': row.get('n_features'),
            'model': row.get('model'),
            'allocation_method': row.get('allocation_method'),
            'importance_method': row.get('importance_method'),
            'budget': row.get('budget'),
            'internal_evaluator': row.get('internal_evaluator'),
            'step_size': row.get('step_size'),
            'rep_count': rep_count,
            'rep_count_source': rc_source,
            'rep_subset': row.get('rep_subset', 'all'),
            'tune': tune,
            'tune_source': tune_source,
            'time_seconds': row.get('time_seconds'),
            'blueprint': bp,
            'identity': dict(zip(_HYBRID_IDENTITY_FIELDS, ident)),
            'label': ident_str,
        }

        # Collapse duplicate identities BEFORE the cut. Without this a
        # refinement row and a checkpoint-reload row describing the same
        # configuration can fill both hybrid slots, and the grid reads as two
        # hybrids when it is one. Ties inside a collapse keep the smaller
        # time_seconds, then the earlier position in all_results.
        prev = by_identity.get(ident)
        if prev is None:
            by_identity[ident] = (score, ident_str, entry, pos)
        else:
            counts['n_duplicate_identities_collapsed'] += 1
            p_score, _, p_entry, p_pos = prev
            better = (score < p_score) if lower else (score > p_score)
            if not better and score == p_score:
                p_t = p_entry.get('time_seconds')
                t = entry.get('time_seconds')
                if t is not None and p_t is not None and t < p_t:
                    better = True
            if better:
                by_identity[ident] = (score, ident_str, entry, pos)

    # A search mixing rows that carry 'tune' with rows that do not cannot be
    # de-duplicated honestly: the same configuration appears under two
    # identities, one keyed True/False and one keyed None. Raise rather than
    # key on None. A search where NO row carries 'tune' (every row read back
    # from the result database) is unambiguous within itself, so it is reported
    # in ranked_over and in notes instead of raising.
    notes = []
    if counts['n_rows_missing_tune']:
        _with_tune = {k[:-1] for k in by_identity if k[-1] is not None}
        _without = {k[:-1] for k in by_identity if k[-1] is None}
        _clash = sorted(str(k) for k in (_with_tune & _without))
        if _clash:
            raise ValueError(
                f"all_results mixes rows that carry 'tune' with rows that do "
                f"not, for {len(_clash)} configuration(s) that are otherwise "
                f"identical (downstream model {downstream_model}). "
                f"_db_row_to_result (find_best_hybrid.py:5839-5883) does not "
                f"restore 'tune', so keying on None would merge a tuned and an "
                f"untuned run. First clash: {_clash[0]}{_sel_context(context)}")
    if counts['n_rows_missing_tune']:
        notes.append(
            f"{counts['n_rows_missing_tune']} of "
            f"{counts['n_rows_for_model']} configurations for downstream model "
            f"{downstream_model} carry no 'tune' field, so they were read back "
            f"from the result database; whether they were tuned is not recorded")

    ranked = sorted(by_identity.values(),
                    key=lambda e: ((e[0] if lower else -e[0]), e[1]))
    tie_at_cut = _sel_tie_at_cut([(e[0], e[1]) for e in ranked], n,
                                 lambda e: e[1])
    selected = []
    for i, (score, ident_str, entry, _pos) in enumerate(ranked[:n]):
        entry = dict(entry)
        entry['rank'] = i + 1
        entry['tied_with'] = [o[1] for o in ranked
                              if o[0] == score and o[1] != ident_str]
        selected.append(entry)

    if selected:
        _folds = {e.get('n_folds_scored') for e in selected}
        if len(_folds) > 1:
            notes.append(f"the selected configurations were scored on "
                         f"different fold counts: {sorted(str(f) for f in _folds)}")

    basis = ('whole_dataset_mean' if rule == 'whole_dataset'
             else f'other_folds_mean(held_out={held_out_fold})')
    counts['fold0_pruned'] = getattr(search, 'fold0_pruned', None)
    counts['note'] = ("all_results holds only the configurations the search "
                      "evaluated -- adaptive pruning drops configurations at "
                      "each checkpoint budget and again at the end of fold 0")
    out = {
        'kind': 'hybrid',
        'downstream_model': downstream_model,
        'dataset_name': getattr(search, 'dataset_name', None),
        'metric_name': getattr(search, 'metric_name', None),
        'direction': direction,
        'score_basis': basis,
        'budget': budget,
        'n_requested': n,
        'n_returned': len(selected),
        'complete': len(selected) >= n,
        'selected': selected,
        'ranked_over': counts,
        'tie_at_cut': tie_at_cut,
        'notes': notes,
        'shortfall_reason': None,
        'context': context,
    }
    if not out['complete']:
        out['shortfall_reason'] = (
            f"dataset {out['dataset_name']}: asked for {n} hybrid "
            f"configurations for downstream model {downstream_model}, got "
            f"{len(selected)}; all_results holds "
            f"{counts['n_rows_for_model']} configuration(s) scored with that "
            f"downstream model and {counts['n_rows_other_model']} scored with "
            f"other downstream models; dropped "
            f"{ {k: v for k, v in counts.items() if k.startswith('n_rows_dropped') and v} }"
            f"{_sel_context(context)}")
        if require_full:
            raise ValueError(out['shortfall_reason'])
    return out


def rank_downstream_models_by_base_reps(search, n_models=2, *, top_k=3,
                                        models=None, min_reps=None,
                                        rule='whole_dataset',
                                        held_out_fold=None, drop_folds=(),
                                        require_full=True, context=None):
    """Top `n_models` downstream models by the MEAN of their top `top_k` base
    representation scores. 0 new model fits -- it is `len(models)` passes over
    search.baselines, through top_base_reps, so the direction rule, the
    missing-score rule and the tie-break cannot diverge from the base selector.

    The owner's rule is the mean of the top 3. n07_remediation.py:487-488 ranks
    downstream models by each model's SINGLE best representation (_best at
    :484-485). On a dataset where one model has one very strong representation
    and a weak rest, the two rules name different models. top_k=1 reproduces
    n07's rule; the default 3 implements the owner's.

    It does not perform the cut. 'selected' holds the models kept, 'not_kept'
    holds every other eligible model WITH its mean so the size of the gap being
    discarded is on the page, and 'ineligible' holds models with fewer than
    `min_reps` scored representations, each with its reason. A mean over 2
    representation scores and a mean over 3 are different quantities, so a model
    short of `min_reps` is not ranked on a shorter mean.

    The means are in the dataset's own metric (SearchResult.metric_name,
    find_best_hybrid.py:1537), so they are comparable only within one dataset.
    """
    models = list(models) if models is not None else list(DOWNSTREAM_MODELS)
    if min_reps is None:
        min_reps = top_k
    direction = _sel_base_direction(search, context)
    lower = direction['lower_is_better']

    entries = []
    ineligible = []
    per_model = {}
    for m in models:
        sel = top_base_reps(search, m, top_k, rule=rule,
                            held_out_fold=held_out_fold, drop_folds=drop_folds,
                            require_full=False, context=context)
        per_model[m] = sel
        reps = [(e['rep'], e['score']) for e in sel['selected']]
        if len(reps) < min_reps:
            ineligible.append({
                'downstream_model': m,
                'n_reps_scored': sel['n_reps_scored_for_model'],
                'n_reps_ranked': len(reps),
                # A downstream model over its max-samples limit
                # (find_best_hybrid.py:2969-2978) or in a registered skip policy
                # (:2841-2848) has no key in baselines[rep] for ANY
                # representation, so this is 0. Ranking an empty mean last would
                # be a claim that the model lost, and it never ran.
                'reason': (f"{sel['n_reps_scored_for_model']} representation(s) "
                           f"carry a score for downstream model {m}; "
                           f"{min_reps} were needed for the mean"),
            })
            continue
        mean_score = float(np.mean([s for _, s in reps]))
        entries.append((mean_score, str(m), {
            'downstream_model': m,
            'mean_score': mean_score,
            'n_reps_used': len(reps),
            'reps': reps,
        }))

    entries.sort(key=lambda e: ((e[0] if lower else -e[0]), e[1]))
    tie_at_cut = _sel_tie_at_cut([(e[0], e[1]) for e in entries], n_models,
                                 lambda e: e[1])
    selected = []
    for i, (_score, _name, entry) in enumerate(entries[:n_models]):
        entry = dict(entry)
        entry['rank'] = i + 1
        selected.append(entry)
    not_kept = [dict(e[2]) for e in entries[n_models:]]

    # Under a geometry objective find_best_hybrid computes ONE score per
    # representation and writes the same number under every downstream model key
    # (find_best_hybrid.py:5465-5489), so every model has the identical ranked
    # list and the identical mean, and the order is decided entirely by the
    # model-name tie-break. Detected, not hidden.
    _lists = [tuple(e[2]['reps']) for e in entries]
    all_identical = bool(len(_lists) > 1 and len(set(_lists)) == 1)

    out = {
        'kind': 'downstream_model',
        'dataset_name': getattr(search, 'dataset_name', None),
        'metric_name': getattr(search, 'metric_name', None),
        'direction': direction,
        'score_basis': (per_model[models[0]]['score_basis'] if models else None),
        'mean_is_over': (f"the mean of each downstream model's top {top_k} "
                         f"base-representation scores, in "
                         f"{getattr(search, 'metric_name', None)}"),
        'n_requested': n_models,
        'n_returned': len(selected),
        'complete': len(selected) >= n_models,
        'selected': selected,
        'not_kept': not_kept,
        'ineligible': ineligible,
        'all_models_identical': all_identical,
        'per_model': per_model,
        'tie_at_cut': tie_at_cut,
        'shortfall_reason': None,
        'context': context,
    }
    if not out['complete']:
        out['shortfall_reason'] = (
            f"dataset {out['dataset_name']}: asked for {n_models} downstream "
            f"models ranked by the mean of their top {top_k} base "
            f"representations, ranked {len(entries)} of {len(models)}; "
            f"{len(ineligible)} had fewer than {min_reps} scored "
            f"representations{_sel_context(context)}")
        if require_full:
            raise ValueError(out['shortfall_reason'])
    return out


def top_hybrids_across_models(search, n=2, **kwargs):
    """Top `n` hybrid configurations over EVERY model the search scored.

    Owner's decision 2026-09-28: when a stored search holds no configuration for
    the model about to be scored, use the best hybrids anyone found rather than
    one named model's.

    ⛔ WHAT THIS IS FOR, AND WHAT IT IS NOT FOR. It exists to replace a
    substitution. Several files used to reach for the random forest's recipe when
    a model had none of its own, which silently reported one model's choice under
    another model's name. Ranking every model's rows together at least says what
    it is: the best configuration found for this dataset, by whichever model
    found it. It is still not that model's own recipe, and every entry carries
    'recipe_chosen_by' naming the model whose score put it at that rank.

    ⛔ THE SCORES BEING RANKED WERE PRODUCED BY DIFFERENT MODELS. They are in the
    dataset's own metric on the same folds, so they are the same quantity, but a
    configuration ranks high here because it suited the model that scored it.
    Never present a number from this as the scored model's own.

    Takes and forwards every keyword top_hybrids takes. The return adds
    'models_ranked_over' and 'recipe_chosen_by' on each entry.
    """
    fitted = searched_models(search)
    if not fitted:
        return {'selected': [], 'ranked_over': {}, 'tie_at_cut': None,
                'notes': ['the search names no model it scored, so there is '
                          'nothing to rank'],
                'shortfall_reason': 'no model on the search',
                'kind': 'hybrid_across_models', 'models_ranked_over': []}
    pooled = []
    # require_full is always False per model here (a model short of n must not
    # cost the others their entries); a caller passing it too raised TypeError
    # (duplicate keyword) until 2026-10-05, which lost every per-fold
    # similarity hybrid in fold_neighbour_space_recipes.
    kwargs.pop('require_full', None)
    for m in fitted:
        got = top_hybrids(search, m, n, require_full=False, **kwargs)
        for e in got['selected']:
            e = dict(e)
            e['recipe_chosen_by'] = m
            pooled.append(e)
    lower = bool(getattr(search, 'lower_is_better', False))
    pooled.sort(key=lambda e: e['score_mean'], reverse=not lower)
    # One configuration can be the best for two models; keep its better rank.
    seen, out = set(), []
    for e in pooled:
        key = e.get('label')
        if key in seen:
            continue
        seen.add(key)
        e['rank'] = len(out) + 1
        out.append(e)
        if len(out) == n:
            break
    return {'selected': out,
            'ranked_over': {'n_models': len(fitted),
                            'n_configurations_pooled': len(pooled)},
            'tie_at_cut': None,
            'notes': [],
            'shortfall_reason': (
                None if len(out) == n else
                f'asked for {n} configurations and {len(out)} were stored '
                f'across the {len(fitted)} models the search scored'),
            'kind': 'hybrid_across_models',
            'models_ranked_over': list(fitted)}


def searched_models(search):
    """The downstream models this SearchResult actually scored, sorted.

    ⛔ WHY THIS EXISTS. Every comparison in this repo is meant to be, per
    downstream model, that model's top base representations against its top
    hybrids. Nineteen files used to loop DOWNSTREAM_MODELS -- a fixed list of
    five names in this file -- rather than asking the search which models it
    scored. The two disagree whenever the search named a shorter list: a
    caller that passes its own speed='fast' with no models= gets rf, lightgbm
    and dnn only, a dataset's skip policy can drop svm or gp, and svm drops
    itself above 10,000 molecules. The models in the list that the search never
    scored then fell out of the loop, mostly with nothing written down, so a
    table of five models was really a table of three and said so nowhere.

    Read from best_by_model_budget first (the primary selection unit,
    find_best_hybrid.py:1596), then best_by_model, then the distinct
    downstream_model values in all_results. Returns [] when none of the three
    is populated, and the caller must then say so rather than guess.
    """
    for attr in ('best_by_model_budget', 'best_by_model'):
        d = getattr(search, attr, None) or {}
        names = [m for m, v in d.items() if v]
        if names:
            return sorted(names)
    rows = getattr(search, 'all_results', None) or []
    names = {r.get('downstream_model') or r.get('model') for r in rows}
    return sorted(n for n in names if n)


def resolve_comparison_models(search, requested=None):
    """(models to compare, [(dropped name, why)]).

    `requested` is what the experiment wants. Anything in it the search did not
    score is dropped and named, never substituted: there is no fallback to
    another model's rows anywhere in this function, because a row labelled for
    one model that holds another model's configuration is the defect this
    whole helper exists to stop.

    requested=None means every model the search scored, which is the right
    default for a comparison table -- it widens on its own when the search ran
    a longer list, so a distance search that scored knn as well as the five
    puts knn in the table without the caller naming it.
    """
    fitted = searched_models(search)
    if requested is None:
        return list(fitted), []
    requested = list(requested)
    if not fitted:
        # Nothing readable on the search. Keep the request and let each
        # top_base_reps / top_hybrids call report its own shortfall, rather
        # than silently returning an empty comparison.
        return requested, []
    keep = [m for m in requested if m in fitted]
    dropped = [(m, f'the search scored {fitted} and not {m!r}')
               for m in requested if m not in fitted]
    return keep, dropped


def select_model_rep_pairs(search, models=None, n_base=2, n_hybrids=2, *,
                           budget=None, rule='whole_dataset',
                           held_out_fold=None, drop_folds=(),
                           max_features=None, require_full=True, context=None):
    """The owner's grid: for each downstream model, the top `n_base` base
    representations and the top `n_hybrids` hybrids. 0 new model fits.

    With the default DOWNSTREAM_MODELS (_shared.py:539, five names) and 2 and 2,
    that is 5 * (2 + 2) = 20 model-and-representation pairs. 'n_pairs' counts
    the rows of the comparison, one per (downstream model, slot).
    'n_distinct_units' counts the separate things that have to be BUILT: one
    representation chosen by three downstream models is three rows and one unit.
    'collapsed' names every unit that two or more downstream models picked.

    Both halves come from ONE SearchResult, so the base representations and the
    hybrids are in the same metric. Calling top_base_reps and top_hybrids
    separately on two different searches is possible and nothing checks it.
    """
    models, _dropped = resolve_comparison_models(search, models)
    per_model = {}
    pairs = []
    shortfalls = []
    distinct = {}
    for _name, _why in _dropped:
        shortfalls.append(f"{_name}: not in the comparison -- {_why}")

    for m in models:
        base = top_base_reps(search, m, n_base, rule=rule,
                             held_out_fold=held_out_fold, drop_folds=drop_folds,
                             max_features=max_features,
                             require_full=False, context=context)
        hyb = top_hybrids(search, m, n_hybrids, budget=budget, rule=rule,
                          held_out_fold=held_out_fold, drop_folds=drop_folds,
                          require_full=False, context=context)
        per_model[m] = {'base': base, 'hybrid': hyb}
        for sel in (base, hyb):
            if sel['shortfall_reason']:
                shortfalls.append(sel['shortfall_reason'])
            if sel['tie_at_cut']:
                shortfalls.append(
                    f"{m}/{sel['kind']}: the cut fell inside a tie at "
                    f"{sel['tie_at_cut']['score']} -- kept "
                    f"{sel['tie_at_cut']['kept']}, dropped "
                    f"{sel['tie_at_cut']['dropped']}, decided by name and not "
                    f"by a measurement")
        for e in base['selected']:
            unit = ('base', e['rep'])
            row = {'downstream_model': m, 'kind': 'base',
                   'rank_within_model': e['rank'], 'unit_id': unit,
                   'label': e['rep'], 'score': e['score']}
            pairs.append(row)
            distinct.setdefault(unit, []).append(row)
        for e in hyb['selected']:
            unit = ('hybrid', tuple(e['identity'][k]
                                    for k in _HYBRID_IDENTITY_FIELDS))
            row = {'downstream_model': m, 'kind': 'hybrid',
                   'rank_within_model': e['rank'], 'unit_id': unit,
                   'label': e['label'], 'score': e['score_mean']}
            pairs.append(row)
            distinct.setdefault(unit, []).append(row)

    n_possible = len(models) * (n_base + n_hybrids)
    out = {
        'dataset_name': getattr(search, 'dataset_name', None),
        'metric_name': getattr(search, 'metric_name', None),
        'models': models,
        'models_dropped': [{'model': n, 'reason': w} for n, w in _dropped],
        'n_base': n_base,
        'n_hybrids': n_hybrids,
        'n_pairs_possible': n_possible,
        'n_pairs': len(pairs),
        'n_missing': n_possible - len(pairs),
        'complete': len(pairs) == n_possible,
        'pairs': pairs,
        'distinct_units': distinct,
        'n_distinct_units': len(distinct),
        'collapsed': [{'unit_id': u, 'label': rows[0]['label'],
                       'chosen_by': [(r['downstream_model'], r['kind'],
                                      r['rank_within_model']) for r in rows]}
                      for u, rows in distinct.items() if len(rows) > 1],
        'per_model': per_model,
        'shortfalls': shortfalls,
        'context': context,
    }
    if require_full and not out['complete']:
        raise ValueError(
            f"dataset {out['dataset_name']}: asked for {n_possible} "
            f"model-and-representation pairs ({len(models)} downstream models "
            f"* ({n_base} base representations + {n_hybrids} hybrids)), got "
            f"{len(pairs)}. " + ' | '.join(shortfalls) + _sel_context(context))
    return out


def hybrid_deltas(search, downstream_model, hybrid=None, *,
                  budget=None, context=None):
    """The two deltas the owner asked for, for one hybrid and one model.

    Owner's spec, 2026-09-18: report a hybrid against TWO things, not one.

      delta_vs_source_mean -- against the MEAN of the base representations that
        actually fed this hybrid. Which ones those are is not fixed: the
        representation-count axis restricts each configuration to the best N on
        that fold, so two hybrids in the same search draw from different pools.
        Reading the pool off the recipe is the only way to know.

      delta_vs_best_rep -- against the single best base representation for this
        downstream model. The head-to-head.

    They answer different questions. The first asks whether mixing beat the
    average of what went in. The second asks whether it beat the best thing
    available. A hybrid can win the first and lose the second, and that is worth
    seeing rather than hiding behind one number.

    WHERE THE SOURCE POOL COMES FROM. `Blueprint.allocation_per_fold` is
    {fold: {representation: n_features}} (find_best_hybrid.py:1425) -- how the
    budget was actually split. A representation listed in `Blueprint.reps` but
    given 0 features on every fold did NOT feed the hybrid and is excluded. When
    allocation_per_fold is absent -- an older stored recipe -- this falls back to
    Blueprint.reps and says so in the returned dict, because the two are not the
    same set and a caller comparing across datasets needs to know which it got.

    Direction is taken from the stored search, the same rule top_hybrids uses:
    the hybrid scores in all_results were ranked with `lower_is_better` by
    construction, so under a blend objective the metric name would give the
    wrong sign. See _sel_hybrid_direction.

    Returns a dict:
        {'delta_vs_source_mean', 'delta_vs_best_rep',
         'hybrid_score', 'source_mean', 'best_rep', 'best_rep_score',
         'source_reps', 'source_from', 'n_source_reps',
         'lower_is_better', 'notes'}
    A delta is POSITIVE when the hybrid is better, on either metric direction.

    Reads stored rows. 0 new model fits.
    """
    ctx = _sel_context(context)
    lower = _sel_hybrid_direction(search, context=context)['lower_is_better']
    notes = []

    if hybrid is None:
        _picked = top_hybrids(search, downstream_model, 1, budget=budget,
                              require_full=False, context=context)
        if not _picked['selected']:
            raise ValueError(
                f"hybrid_deltas{ctx}: no hybrid scored for downstream model "
                f"{downstream_model!r}"
                + (f" at budget {budget}" if budget is not None else "")
                + ". The search stored no configuration under that name — it is "
                  "absent, not beaten. Pass hybrid= to name one yourself.")
        # ⛔ WAS ['row'], WHICH IS NOT A KEY, FIXED 2026-09-28. Two things
        # disagreed. top_hybrids builds each selected entry as a flat dict of
        # score_mean, blueprint, budget, n_features and the rest (_shared.py:7693
        # onwards) and never puts the underlying row under a 'row' key. This line
        # asked for that key, so every call that left hybrid= unset raised
        # KeyError before either delta was computed.
        #
        # The two things this function then reads off `hybrid` are score_mean and
        # blueprint, and the entry carries both, so the entry is what it wants.
        # opig_ginkgo_developability.py:367 already passes the entry itself.
        #
        # ⛔ SOME CALLERS PASS hybrid= BY HAND SPECIFICALLY TO AVOID THIS, and
        # say so in a comment beside the call (opig_humanness.py:365,
        # opig_kingsmith_component_splits.py:580,
        # opig_trastuzumab_enrichment.py:762). Those calls keep working
        # unchanged; what changes is that the default path works too.
        hybrid = _picked['selected'][0]

    h_score = hybrid.get('score_mean')
    if h_score is None or not np.isfinite(h_score):
        raise ValueError(f"hybrid_deltas{ctx}: the hybrid row carries no finite "
                         f"score_mean, so neither delta is defined.")
    h_score = float(h_score)

    # --- which representations fed it -------------------------------------
    bp = hybrid.get('blueprint')
    source_reps, source_from = [], 'allocation_per_fold'
    _alloc = getattr(bp, 'allocation_per_fold', None) if bp is not None else None
    if _alloc:
        _got = set()
        for _fold_split in _alloc.values():
            for _rep, _n in (_fold_split or {}).items():
                if _n:
                    _got.add(_rep)
        source_reps = sorted(_got)
    if not source_reps:
        source_from = 'blueprint.reps'
        source_reps = sorted(getattr(bp, 'reps', None) or [])
        notes.append(
            "this recipe stores no per-fold budget split, so the source pool is "
            "Blueprint.reps — every representation the configuration was allowed "
            "to draw from, which can be more than it actually used")
    if not source_reps:
        raise ValueError(f"hybrid_deltas{ctx}: the recipe names no source "
                         f"representations, so delta_vs_source_mean is undefined.")

    # --- the two reference numbers ----------------------------------------
    baselines = getattr(search, 'baselines', {}) or {}
    src_scores, missing = [], []
    for rep in source_reps:
        sc = (baselines.get(rep) or {}).get(downstream_model)
        if sc is None or not np.isfinite(sc):
            missing.append(rep)
            continue
        src_scores.append(float(sc))
    if missing:
        notes.append(f"no {downstream_model} score for {len(missing)} of "
                     f"{len(source_reps)} source representations "
                     f"({', '.join(missing)}); the mean is over the rest")
    if not src_scores:
        raise ValueError(
            f"hybrid_deltas{ctx}: none of the {len(source_reps)} source "
            f"representations has a {downstream_model} baseline score, so "
            f"delta_vs_source_mean is undefined.")
    source_mean = float(np.mean(src_scores))

    _best = top_base_reps(search, downstream_model, 1, require_full=False,
                          context=context)
    best_rep = best_rep_score = None
    if _best['selected']:
        best_rep = _best['selected'][0]['rep']
        best_rep_score = float(_best['selected'][0]['score'])
    else:
        notes.append(f"no base representation carries a {downstream_model} "
                     f"score, so delta_vs_best_rep is None")

    # Positive = the hybrid is better, whichever way the metric runs.
    def _delta(ref):
        if ref is None:
            return None
        return float(ref - h_score) if lower else float(h_score - ref)

    return {
        'delta_vs_source_mean': _delta(source_mean),
        'delta_vs_best_rep': _delta(best_rep_score),
        'hybrid_score': h_score,
        'source_mean': source_mean,
        'best_rep': best_rep,
        'best_rep_score': best_rep_score,
        'source_reps': source_reps,
        'source_from': source_from,
        'n_source_reps': len(src_scores),
        'lower_is_better': lower,
        'notes': notes,
    }


def selection_diff(selection, previous):
    """One line: what a file's old rule picked against what a selector picks.

    `previous` is the list of names the experiment selected before. For a base
    representation selection the names are representation names; for a hybrid
    selection they are the identity labels. Written so a migration shows up in
    review as a stated reordering instead of a silent one.
    """
    if selection.get('kind') == 'base_rep':
        new = [e['rep'] for e in selection['selected']]
    elif selection.get('kind') == 'hybrid':
        new = [e['label'] for e in selection['selected']]
    else:
        new = [e['downstream_model'] for e in selection['selected']]
    prev = list(previous or [])
    replaced = [x for x in prev if x not in new]
    added = [x for x in new if x not in prev]
    if not replaced and not added:
        moved = sum(1 for i, x in enumerate(new)
                    if i < len(prev) and prev[i] != x)
        return (f"same {len(new)} selected, {moved} moved position: "
                f"{prev} -> {new}")
    return (f"was {prev} -> now {new}; {len(replaced)} replaced "
            f"({replaced}), {len(added)} new ({added})")



# =============================================================================
# OPTUNA HP TUNING
# =============================================================================

def tune_xgboost_optuna(X_train, y_train, groups=None, n_trials=50,
                        n_cv_folds=3, task_type='regression', seed=42,
                        verbose=False):
    """Tune XGBoost hyperparameters with Optuna. Returns best model instance.

    `groups` are this training set's scaffold/identity groups. Each trial is
    scored on a group-aware split, so a hyperparameter cannot be chosen because
    it memorised a scaffold that sits on both sides of the trial's own split.
    Required whenever the caller has groups; passing None raises rather than
    scoring trials on a random split.

    Requires: pip install optuna xgboost
    """
    import optuna
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    from xgboost import XGBRegressor, XGBClassifier
    from sklearn.model_selection import cross_val_score

    is_clf = task_type == 'classification'
    scoring = 'accuracy' if is_clf else 'r2'
    ModelClass = XGBClassifier if is_clf else XGBRegressor
    trial_splitter, _ = make_cv_splitter(
        n_cv_folds, groups=groups, seed=seed,
        task_type='classification' if is_clf else None)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': seed,
            'n_jobs': -1,
            'verbosity': 0,
        }
        model = ModelClass(**params)
        scores = cross_val_score(model, X_train, y_train, groups=groups,
                                 cv=trial_splitter, scoring=scoring)
        return scores.mean()

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    best = ModelClass(**study.best_params, random_state=seed, n_jobs=-1, verbosity=0)
    best.fit(X_train, y_train)
    return best


# =============================================================================
# IO HELPERS
# =============================================================================

def convert_numpy(obj):
    """Recursively convert numpy types to JSON-serialisable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif hasattr(obj, 'to_dict'):
        return convert_numpy(obj.to_dict())
    elif hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
        return convert_numpy(asdict(obj))
    elif isinstance(obj, dict):
        return {str(k): convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(v) for v in obj]
    return obj


# Top-level keys that describe the run rather than carry a computed quantity.
# classify_result_payload ignores them when deciding whether a saved result is
# empty; everything else in the dict counts as a payload section.
RESULT_META_KEYS = frozenset({
    'timestamp', 'experiment', 'experiment_id', 'dataset', 'domain',
    'config', 'meta', 'primary_metric', 'metric_name', 'metric',
    'lower_is_better', 'task_type', 'n_samples', 'n_reps', 'seed', 'mode',
    'notes', 'skipped', 'reason', 'policy',
    'result_status', 'has_computed_results',
})


# --- Status records vs measurements -------------------------------------------
# Exact key names that carry a status note or a status count rather than a
# measured quantity. A value stored under one of these never makes a payload
# non-empty, at any nesting depth.
MEASUREMENT_STATUS_KEYS = frozenset({
    'error', 'errors', 'error_type', 'error_source', 'error_source_note',
    'reason', 'reasons', 'traceback', 'message', 'status',
    'note', 'notes', 'warning', 'warnings', 'legend', 'arm_legend',
    'computed', 'ok', 'success', 'succeeded',
    'result_status', 'has_computed_results',
    'skipped', 'failed', 'failures',
})

# Suffixes that mark a key as a status count. '_error'/'_errors' are NOT here:
# 'cal_error', 'mean_error', 'prediction_error', 'cv_error' and 'decile_errors'
# are measurements in this repository. The substring 'skipped' is handled
# separately because status counts spell it mid-name ('cliff_n_skipped_clf',
# 'n_skipped_cells').
MEASUREMENT_STATUS_SUFFIXES = ('_skipped', '_failed', '_failure', '_failures',
                               '_reason', '_reasons')

# Keys whose False value declares the enclosing dict a status record.
STATUS_FLAG_KEYS = ('computed', 'ok', 'success', 'succeeded',
                    'has_computed_results')


def _is_status_key(key) -> bool:
    """True when `key` names a status note or a status count, not a measurement.

    Matches the 25 exact names in MEASUREMENT_STATUS_KEYS, the 6 suffixes in
    MEASUREMENT_STATUS_SUFFIXES, and any name containing 'skipped'.
    """
    if not isinstance(key, str):
        return False
    k = key.lower()
    if k in MEASUREMENT_STATUS_KEYS:
        return True
    if 'skipped' in k:
        return True
    return k.endswith(MEASUREMENT_STATUS_SUFFIXES)


def _is_status_record(item) -> bool:
    """True when `item` is a dict that declares itself a non-result.

    Two triggers: a status flag set False (computed=False, ok=False,
    success=False, succeeded=False, has_computed_results=False), or an 'error'
    or 'reason' key with no measurement anywhere else in the dict.
    """
    if not isinstance(item, dict):
        return False
    for flag in STATUS_FLAG_KEYS:
        if flag in item and item[flag] is False:
            return True
    if 'error' in item or 'reason' in item:
        rest = {k: v for k, v in item.items()
                if k not in ('error', 'reason')}
        return not _contains_measurement(rest)
    return False


def _is_measurement(item) -> bool:
    """True when `item` is a finite number that is not a bool.

    bool is excluded (True/False are flags, and bool subclasses int). NaN and
    +/-inf are excluded: they record that a quantity was not obtained.
    """
    if isinstance(item, (bool, np.bool_)):
        return False
    if isinstance(item, int):        # a Python int is always finite
        return True
    if isinstance(item, (float, np.floating, np.integer)):
        try:
            return bool(np.isfinite(item))
        except (TypeError, ValueError):
            return False
    return False


def _contains_measurement(value) -> bool:
    """True when a finite number sits somewhere in `value` under a non-status key.

    Iterative walk (no recursion limit), returning on the first measurement
    found. Skipped on the way down: values under a status key (_is_status_key),
    dicts that declare themselves status records (_is_status_record), strings,
    bools, None. So a payload built only from error strings, refusal reasons,
    skip counts and status flags contains no measurement.
    """
    stack = [value]
    seen = set()
    while stack:
        item = stack.pop()
        if item is None:
            continue
        if isinstance(item, (dict, list, tuple, set)):
            marker = id(item)
            if marker in seen:      # guard against a self-referencing payload
                continue
            seen.add(marker)
        if isinstance(item, dict):
            if _is_status_record(item):
                continue
            for k, v in item.items():
                if _is_status_key(k):
                    continue
                stack.append(v)
        elif isinstance(item, (list, tuple, set)):
            stack.extend(item)
        elif isinstance(item, str):
            continue
        elif _is_measurement(item):
            return True
        elif isinstance(item, np.ndarray):
            # A string or object array casts badly; walk its elements instead.
            try:
                if item.size and np.isfinite(item.astype(float)).any():
                    return True
            except (TypeError, ValueError):
                stack.extend(item.ravel().tolist())
    return False


def _payload_is_empty(value) -> bool:
    """True when `value` records no numeric measurement.

    A finite number under a non-status key is the only thing that makes a
    payload non-empty. Strings, bools, None and empty containers never do, so a
    payload holding only failure records, refusal records or status records is
    empty. See _contains_measurement for the walk.
    """
    return not _contains_measurement(value)


def classify_result_payload(results) -> str:
    """Return 'skipped', 'empty' or 'ok' for a result dict about to be saved.

    'skipped' — carries the {'skipped': True} marker should_skip_experiment
                writes, i.e. the experiment was never meant to run here.
    'empty'   — no finite number outside RESULT_META_KEYS and outside status
                keys, so the file records no measurement. All-failed runs,
                refusal payloads and status-only payloads land here.
    'ok'      — at least one finite numeric measurement is present.
    """
    if not isinstance(results, dict):
        return 'empty' if _payload_is_empty(results) else 'ok'
    if results.get('skipped'):
        return 'skipped'
    payload = {k: v for k, v in results.items() if k not in RESULT_META_KEYS}
    return 'empty' if _payload_is_empty(payload) else 'ok'


def experiment_result_is_empty(loaded, classify_unmarked: bool = False) -> bool:
    """True when a loaded result JSON records no computed quantity.

    Reads the 'result_status' / 'has_computed_results' keys save_experiment_
    results stamps. Files written before those keys existed carry neither; they
    return False unless classify_unmarked=True, which re-derives the status from
    the file's own sections via classify_result_payload.

    A resume check that tests only for the file existing treats an empty result
    as finished; this is the test that separates the two.
    """
    if not isinstance(loaded, dict):
        return False
    status = loaded.get('result_status')
    if status is not None:
        return status == 'empty'
    if 'has_computed_results' in loaded:
        return loaded['has_computed_results'] is False
    if classify_unmarked:
        return classify_result_payload(loaded) == 'empty'
    return False


def input_fingerprint(data, config=None):
    """What a result file was computed FROM, small enough to store in it.

    A resumed run decides whether to recompute an experiment. Asking only
    whether the output file exists cannot see that the fold grouping changed,
    that a representation was added or dropped, or that the labels moved — the
    stale file is kept and the run reports a number produced under a different
    dataset. This is what a resume compares.

    Contents: row count, split count, a content hash of the labels and of the
    group vector, the sorted representation names with their widths, the
    process-wide distance coefficient, the metric and task type the scores are
    in, and the noise grid a noise experiment swept. The
    hashes are the same canonical-bytes hashes find_best_hybrid's cache uses, so
    a string group vector hashes stably across processes.

    Returns a dict, or {} when `data` carries no labels (nothing to compare).
    """
    from kirby.find_best_hybrid import _hash_bytes, _canonical_bytes

    if not isinstance(data, dict):
        return {}
    labels = data.get('labels')
    if labels is None:
        return {}
    labels = np.asarray(labels)
    groups = data.get('groups')
    reps = data.get('reps') or {}
    fp = {
        'n_samples': int(len(labels)),
        'labels_hash': int(_hash_bytes(_canonical_bytes(labels))),
        'groups_hash': (int(_hash_bytes(_canonical_bytes(groups)))
                        if groups is not None else None),
        'n_groups': (int(len(np.unique(np.asarray(groups))))
                     if groups is not None else None),
        'reps': sorted(
            [str(k), int(np.shape(v)[1])] for k, v in reps.items()),
        # A representation with no whole-dataset matrix, listed by name only.
        'fold_fitted_reps': sorted(data.get('fold_fitted_reps') or []),
    }
    # The pairwise distance every neighbour, cliff, applicability-domain,
    # noise-score and retrieval quantity is computed under. Without it, a run
    # under a second coefficient reads the first coefficient's file, sees no
    # change and skips — so the numbers stay on the old coefficient with
    # nothing said. The distance sweep sidesteps this by writing each
    # coefficient to its own output tree; this covers the case where it does
    # not, which is every plain --distance-metric run into the normal tree.
    try:
        from kirby.similarity.distance_config import get_distance_metric
        fp['distance_metric'] = str(get_distance_metric())
    except Exception:
        pass
    if config:
        fp['n_splits'] = int(config.get('n_splits') or 0)
        # Which of those folds were run (noise suite: fold 0 only for a
        # one-fold screen, 2026-10-06). A 5-fold promotion must not read the
        # one-fold file back as finished.
        if config.get('only_folds') is not None:
            fp['only_folds'] = sorted(int(f) for f in config['only_folds'])
        # The metric the stored scores are IN. Without it, a dataset whose
        # registry metric moves -- r2 to rmse, auc to mcc -- keeps its finished
        # result files, the resume finds nothing changed, and the run reports
        # numbers in the old metric under the new one's name.
        _m = config.get('metric')
        if _m is not None and not callable(_m):
            fp['metric'] = str(_m)
        _tt = config.get('task_type')
        if _tt is not None:
            fp['task_type'] = str(_tt)
        # The noise grid a noise experiment swept. Changing the levels, the
        # strategies or the seeds produces a different table from the same
        # rows, labels, groups and representations -- so every field above is
        # unchanged and the finished file is kept. Stored as a canonical string
        # so a reordered list is not a change.
        _levels = config.get('noise_levels')
        _strats = config.get('strategies') or config.get('noise_strategies')
        _nseeds = config.get('noise_seeds')
        if _levels is not None or _strats is not None or _nseeds is not None:
            def _canon(v):
                if v is None:
                    return None
                if isinstance(v, dict):
                    return sorted((str(k), _canon(x)) for k, x in v.items())
                if isinstance(v, (list, tuple, set)):
                    return sorted(str(x) for x in v)
                return str(v)
            fp['noise_grid'] = repr((_canon(_strats), _canon(_levels),
                                     _canon(_nseeds)))
    return fp


def fingerprint_mismatch(stored, current):
    """Name the first field of an input_fingerprint that moved, or None.

    A stored fingerprint of None or {} means the file predates the stamp: it
    cannot be compared, so it is treated as stale and the reason says so.
    """
    if not stored:
        return 'the file records nothing about the data it was computed from'
    if not current:
        return None  # nothing to compare against; leave the file alone
    for key in ('n_samples', 'labels_hash', 'groups_hash', 'n_groups',
                'n_splits', 'distance_metric', 'metric', 'task_type',
                'noise_grid'):
        if key in current and stored.get(key) != current.get(key):
            return f'{key} changed: {stored.get(key)!r} -> {current.get(key)!r}'
    _old = {tuple(r) for r in (stored.get('reps') or [])}
    _new = {tuple(r) for r in (current.get('reps') or [])}
    if _old != _new:
        _gone = sorted(n for n, _ in _old - _new)
        _added = sorted(n for n, _ in _new - _old)
        return (f'representations changed: dropped {_gone or "none"}, '
                f'added {_added or "none"}')
    if (sorted(stored.get('fold_fitted_reps') or [])
            != sorted(current.get('fold_fitted_reps') or [])):
        return 'the per-fold representation list changed'
    return None


def save_experiment_results(results: dict, output_dir: str, experiment_id: str,
                            dataset: str = None,
                            metric_name: str = None,
                            lower_is_better: bool = None,
                            fingerprint: dict = None,
                            filename_suffix: str = None):
    """Save results JSON to output_dir/experiment_id/{dataset}[__suffix].json.

    If results contains a 'meta' key, the output uses the structured schema.
    Otherwise, wraps results with a timestamp for traceability.

    Always stamps the metric DIRECTION (`metric_name` + `lower_is_better`) at the
    top level so analysis/aggregation never has to re-guess it — the root cause of
    a recurring sign bug where lower-is-better (RMSE/MAE) molecular results were
    read as regressions. Callers may pass the authoritative values (e.g. from
    get_domain_metric(data)); otherwise they are derived from the registered
    `dataset` name. Values already present in `results` are never overwritten.

    Also stamps `result_status` ('ok' | 'empty' | 'skipped') and the boolean
    `has_computed_results`, so a reader can tell a file whose sections are all
    empty from one that finished with numbers in it. Existing keys and values
    are untouched: this only adds two keys.
    """
    import datetime

    out = Path(output_dir) / experiment_id
    out.mkdir(parents=True, exist_ok=True)
    # filename_suffix carries anything that changes what the run MEASURED while
    # leaving the dataset and the experiment the same -- today only the noise
    # condition. Without it a second condition overwrote the first, or was
    # skipped by the resume check as already done, and nothing in either file
    # said which condition had produced it. Added 2026-09-12 (user decision).
    _sfx = ""
    if filename_suffix:
        import re as _re
        _clean = _re.sub(r'[^A-Za-z0-9._-]+', '_', str(filename_suffix)).strip('_')
        if _clean:
            _sfx = f"__{_clean}"
    filename = f"{dataset}{_sfx}.json" if dataset else f"results{_sfx}.json"
    filepath = out / filename

    output = convert_numpy(results)
    if 'meta' not in output:
        output = {'timestamp': datetime.datetime.now().isoformat(), **output}
    else:
        output.setdefault('timestamp', datetime.datetime.now().isoformat())
        output.setdefault('experiment', experiment_id)

    # --- Persist metric direction (see docstring) ---
    if (metric_name is None or lower_is_better is None) and dataset:
        try:
            info = registry_lookup(dataset)
        except Exception:
            info = None
        if info is not None:
            if metric_name is None:
                metric_name = info.metric
            if lower_is_better is None:
                lower_is_better = info.lower_is_better
    # Fill in only when missing/None; never clobber an explicit value (incl. a
    # legitimate False) already set by make_structured_output or the caller.
    # Which distance produced these numbers. Stamped unconditionally so a file
    # is self-describing wherever it is copied to, and so the comparison script
    # never has to infer it from the directory it was found in.
    output.setdefault('distance_metric', get_distance_metric())

    if metric_name is not None and output.get('metric_name') is None:
        output['metric_name'] = metric_name
    if lower_is_better is not None and output.get('lower_is_better') is None:
        output['lower_is_better'] = lower_is_better

    # --- Stamp whether this file records any computed quantity ---
    # The resume checks in complexity/similarity/noise treat the file existing
    # as proof the experiment finished, so a run that returned only empty
    # sections was skipped on every resubmission. 'result_status' is 'ok',
    # 'empty' or 'skipped'; experiment_result_is_empty() reads it back.
    # setdefault: a caller that set either key keeps its own value.
    if isinstance(output, dict):
        _status = classify_result_payload(output)
        output.setdefault('result_status', _status)
        output.setdefault('has_computed_results', _status == 'ok')

    # --- Stamp what this file was computed from ---
    # A resumed run compares this against the data it just loaded. Without it a
    # file written before a fold-grouping, label or representation change is
    # kept and never recomputed. See input_fingerprint / fingerprint_mismatch.
    if isinstance(output, dict) and fingerprint:
        output.setdefault('input_fingerprint', convert_numpy(fingerprint))

    # Atomic write: write to temp file then rename, so parallel jobs
    # can't corrupt each other's output by truncating mid-write.
    import tempfile
    fd, tmp_path = tempfile.mkstemp(dir=str(out), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(output, f, indent=2)
        os.replace(tmp_path, str(filepath))  # atomic on same filesystem
    except BaseException:
        os.unlink(tmp_path)
        raise
    _mark = ''
    if isinstance(output, dict) and output.get('result_status') == 'empty':
        _mark = ' (result_status=empty: no computed quantity in the payload)'
    print(f"  [saved {filepath}]{_mark}")
    return filepath


def load_experiment_results(output_dir: str, experiment_id: str,
                            dataset: str = None) -> Optional[dict]:
    """Load results JSON from output_dir/experiment_id/{dataset}.json."""
    out = Path(output_dir) / experiment_id
    filename = f"{dataset}.json" if dataset else "results.json"
    filepath = out / filename
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return None


def make_structured_output(experiment_id: str, dataset: str, data: dict,
                           config: dict, results: Any,
                           primary_metric: str = None,
                           lower_is_better: bool = None) -> dict:
    """Build the common structured JSON schema for framework experiments.

    The top-level `lower_is_better` describes the DATASET's own metric, and is
    resolved from the loaded data rather than taken on trust from the caller.
    An experiment whose headline quantity runs the other way -- n01 reports a
    cliff-versus-non-cliff gap, where smaller is better on a dataset scored by
    spearman -- gets that recorded separately as
    `primary_metric_lower_is_better`, so the two are never confused.

    Before 2026-09-10 the caller's literal went straight into the top-level key
    and save_experiment_results fills that key from the registry only when it is
    None, so one hardcoded True hid the registry's answer for the whole file.
    Fixed 2026-09-10 (metric-direction audit).

    Returns a dict ready for save_experiment_results().
    """
    import datetime
    reps = data.get('reps', {})
    _primary_lib = lower_is_better
    try:
        _dataset_lib = get_domain_metric(data)[2]
    except Exception:
        # Unregistered dataset or a loader that set no task_type: fall back to
        # whatever the caller said, and let save_experiment_results try the
        # registry by name.
        _dataset_lib = lower_is_better
    return {
        'experiment': experiment_id,
        'dataset': dataset,
        'domain': data.get('domain', 'molecular'),
        'timestamp': datetime.datetime.now().isoformat(),
        'primary_metric': primary_metric,
        'lower_is_better': _dataset_lib,
        'primary_metric_lower_is_better': _primary_lib,
        'config': convert_numpy(config),
        'results': convert_numpy(results),
        'meta': {
            'n_samples': len(data.get('labels', [])),
            'n_reps': len(reps),
            'rep_dims': {k: v.shape[1] for k, v in reps.items()},
            'task_type': data['task_type'],
        },
    }


def run_across_tasks(run_fn, tasks: List[dict], output_dir: str,
                     experiment_id: str):
    """Run an experiment function across multiple (dataset, rep_preset) tasks.

    Args:
        run_fn: callable(data, config, output_dir, dataset_name) -> results
        tasks: list of dicts with keys: dataset, reps, mode (optional).
        output_dir: base output directory.
        experiment_id: experiment ID for structured output.

    Returns:
        list of (dataset_name, results) tuples.
    """
    all_results = []
    for task in tasks:
        dataset = task['dataset']
        rep_preset = task.get('reps', 'all')
        mode = task.get('mode', 'full')

        print(f"\n{'='*70}")
        print(f"  TASK: {dataset} (reps={rep_preset}, mode={mode})")
        print(f"{'='*70}")

        # Use curated rep list from FRAMEWORK_TASKS if available
        custom_reps = task.get('rep_names') or FRAMEWORK_TASKS.get(dataset)
        data = load_data(dataset, rep_preset, rep_names=custom_reps)
        # Build config from mode
        # Build config from the DEFAULT_* constants
        from kirby.find_best_hybrid import (
            DEFAULT_N_FOLDS, DEFAULT_ALLOCATION_METHODS,
            DEFAULT_IMPORTANCE_METHODS, DEFAULT_INTERNAL_EVALUATORS)
        config = {
            'n_splits': DEFAULT_N_FOLDS,
            'allocations': DEFAULT_ALLOCATION_METHODS,
            'importance_methods': DEFAULT_IMPORTANCE_METHODS,
            'internal_evaluators': DEFAULT_INTERNAL_EVALUATORS,
            # The default set from the model tiers, nothing else. This list
            # used to be hardcoded with knn, mlp, gp, catboost and cnn in it,
            # which contradicted the tiers — owner decision 2026-09-09:
            # catboost, cnn, bnn and ridge are tier 'off' and are not run in
            # the suite, and mlp, gp, knn are tier 'extra' and are not
            # auto-selected.
            'models': list(DOWNSTREAM_MODELS),
        }
        results = run_fn(data, config, output_dir, dataset)
        save_experiment_results(results, output_dir, experiment_id, dataset)
        all_results.append((dataset, results))

    return all_results


def add_distance_metric_arg(parser):
    """Give a parser --distance-metric. For scripts that build their own.

    create_experiment_parser already includes it and get_config() applies it.
    The eleven opig_* scripts build their own parsers, so they call this and
    then apply_distance_metric_arg(args) themselves.
    """
    parser.add_argument('--distance-metric', type=str,
                        default=DEFAULT_DISTANCE_METRIC,
                        choices=list(DISTANCE_METRICS),
                        help=('Pairwise distance for every neighbour, cliff, '
                              'applicability-domain, noise-score and retrieval '
                              'quantity this experiment computes. Default '
                              'manhattan. KIRBY_DISTANCE_METRIC in the '
                              'environment does the same thing.'))
    return parser


def apply_distance_metric_arg(args):
    """Set the process-wide distance from a parsed --distance-metric."""
    return set_distance_metric(
        getattr(args, 'distance_metric', None) or DEFAULT_DISTANCE_METRIC)


def create_experiment_parser(description: str, has_delta: bool = False,
                             has_augment: bool = False) -> argparse.ArgumentParser:
    """Create a standard argument parser for experiment scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--mode', choices=['intermediary', 'full'], default='intermediary',
                        help='intermediary (1 split, fast) or full (5-fold CV)')
    parser.add_argument('--dataset', type=str, default='logd',
                        help='Dataset name (logd, caco2, herg, qm9, flab_*, etc.)')
    parser.add_argument('--reps', type=str, default='fast',
                        help='Representation preset (fast, fast+hf, all) or comma-separated list')
    parser.add_argument('--output', type=str, default='tests/results',
                        help='Output directory for results JSON')
    parser.add_argument('--framework', action='store_true',
                        help='Use curated FRAMEWORK_TASKS rep lists (overrides --reps)')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model list (overrides mode default)')
    parser.add_argument('--budgets', type=str, default=None,
                        help='Comma-separated budget list (overrides script default)')
    parser.add_argument('--max-folds', type=int, default=None,
                        help='Max folds to evaluate (uses same splitter, stops early)')
    parser.add_argument('--distance-metric', type=str,
                        default=DEFAULT_DISTANCE_METRIC,
                        choices=list(DISTANCE_METRICS),
                        help=('Pairwise distance for every neighbour, cliff, '
                              'applicability-domain, noise-score and retrieval '
                              'quantity this experiment computes. Default '
                              'manhattan. k-means, the RBF kernels and the '
                              'ECFP4-Tanimoto cliff reference are fixed by what '
                              'they are and ignore this.'))
    if has_delta:
        parser.add_argument('--delta', action='store_true',
                            help='Generate delta representations for antibody datasets')
    if has_augment:
        parser.add_argument('--augment', action='store_true',
                            help='Include augmented features for antibody datasets')
    return parser


def parse_rep_args(args) -> Tuple[str, Optional[List[str]]]:
    """Parse --reps and --framework args into (rep_preset, rep_names).

    Returns:
        (rep_preset, rep_names) where rep_names is None if using a preset,
        or a list of rep names if using --framework or comma-separated --reps.
    """
    if getattr(args, 'framework', False):
        dataset = args.dataset
        if dataset in FRAMEWORK_TASKS:
            return 'all', FRAMEWORK_TASKS[dataset]
        else:
            print(f"  WARNING: {dataset} not in FRAMEWORK_TASKS, falling back to --reps")

    reps_arg = args.reps
    if ',' in reps_arg:
        return 'all', [r.strip() for r in reps_arg.split(',')]
    return reps_arg, None



def decided_config(config, dataset_name, verbose=True):
    """Set an experiment config's search axes to the dataset's decided grid.

    get_config and each suite's run_all used to hand every experiment the five
    DOWNSTREAM_MODELS and the eight FULL_BUDGETS, with the engine's default
    importance methods, allocation methods and internal evaluators. Those are
    the grid DECISIONS.md replaced (rules R1 and R3). This overwrites them
    with decided_grid.decided_search(dataset_name, 'general') and adds the
    distance search's models and budgets under 'distance_models' and
    'distance_budgets' (None when the dataset has no distance search).

    An undecided dataset raises decided_grid.UndecidedSearch; there is no
    fallback to the old lists.
    """
    from experiments.decided_grid import decided_search, is_decided
    g = decided_search(dataset_name, 'general')
    config['models'] = list(g['models'])
    config['budgets'] = list(g['budgets'])
    config['importance_methods'] = list(g['importance_methods'])
    config['allocations'] = list(g['allocation_methods'])
    config['internal_evaluators'] = list(g['internal_evaluators'])
    config['rep_counts'] = g['rep_counts']
    config['aug_conditions'] = list(g['aug_conditions'])
    if is_decided(dataset_name, 'general_distance'):
        d = decided_search(dataset_name, 'general_distance')
        config['distance_models'] = list(d['models'])
        config['distance_budgets'] = list(d['budgets'])
    else:
        config['distance_models'] = None
        config['distance_budgets'] = None
    config['decided_grid_dataset'] = dataset_name
    if verbose:
        print(f"  decided grid for {dataset_name}: models {config['models']}, "
              f"budgets {config['budgets']} features; distance search models "
              f"{config['distance_models']}, budgets {config['distance_budgets']}")
    return config

def get_config(args) -> dict:
    """Return config dict based on args.mode (fast or full).

    Pulls allocations/importance/evaluators from the DEFAULT_* constants.
    Models and budgets are experiment-controlled (set via CLI or experiment code).
    """
    import os as _os
    from kirby.find_best_hybrid import (
        DEFAULT_N_FOLDS, DEFAULT_ALLOCATION_METHODS,
        DEFAULT_IMPORTANCE_METHODS, DEFAULT_INTERNAL_EVALUATORS)

    # Use FULL_BUDGETS for hybrid search cache alignment with
    # SLURM precompute. --mode only affects the experiment's own CV
    # evaluation (n_splits), not the hybrid search speed.
    default_budgets = FULL_BUDGETS

    # n_splits for experiment CV: 1 for intermediary (quick check), 5 for full
    exp_folds = DEFAULT_N_FOLDS if args.mode == 'full' else 1
    # KIRBY_LOCAL_FOLDS: the experiment's OWN outer CV, the same knob
    # _local_grid applies to the search's inner folds. A condition, not a
    # sample size -- every fold still sees every molecule. Only honoured under
    # --mode full, so an intermediary run still means one split.
    _lf = os.environ.get('KIRBY_LOCAL_FOLDS')
    if _lf and args.mode == 'full':
        exp_folds = int(_lf)

    config = {
        'n_splits': exp_folds,
        'allocations': DEFAULT_ALLOCATION_METHODS,
        'importance_methods': DEFAULT_IMPORTANCE_METHODS,
        'internal_evaluators': DEFAULT_INTERNAL_EVALUATORS,
        'budgets': default_budgets,
        # Models for experiment use (baselines, evaluation, etc.)
        # NOTE: precompute_hybrid_searches ignores this — uses models=None
        # so the cache key matches the SLURM precompute.
        # Same correction as above — the default set from the model tiers.
        'models': list(DOWNSTREAM_MODELS),
    }

    # Smoke shrink, gated on KIRBY_SMOKE_N, applied BEFORE the CLI overrides so
    # an explicit --budgets still wins. Five opig files carry this same shrink
    # in their own _smoke_search_kwargs; every experiment that reaches its
    # search through get_config instead had none, so a local run searched the
    # full 8,160 configurations with deeplift. Measured 2026-09-19:
    # opig_scaffold_hybrid ran 8,160 per outer fold where its siblings ran 5.
    # deeplift is also the recorded local segfault trigger, so dropping it here
    # removes that from every get_config caller at once. No-op when the
    # variable is unset, so a cluster run is unchanged.
    if _os.environ.get('KIRBY_SMOKE_N', '').strip():
        config['internal_evaluators'] = ['rf']
        config['importance_methods'] = ['lightgbm_gain']
        config['budgets'] = smoke_budgets()

    # CLI overrides
    if getattr(args, 'models', None):
        config['models'] = [m.strip() for m in args.models.split(',')]
    if getattr(args, 'budgets', None):
        config['budgets'] = [int(b.strip()) for b in args.budgets.split(',')]
    if getattr(args, 'max_folds', None):
        config['max_folds'] = args.max_folds

    # Applied process-wide, not just returned: the neighbour searches, the
    # square distance matrices and the noise estimators read it through
    # kirby.similarity.distance_config rather than being handed it one argument
    # at a time. Recorded in the config so it reaches the results JSON.
    config['distance_metric'] = set_distance_metric(
        getattr(args, 'distance_metric', None) or DEFAULT_DISTANCE_METRIC)
    # The decided grid replaces the five models and eight budgets above when
    # the run names its dataset (R1, R3). A --models or --budgets given on the
    # command line is narrowed to the decided members, never widened.
    _ds = getattr(args, 'dataset', None)
    if isinstance(_ds, str) and _ds:
        _cli_models = config['models'] if getattr(args, 'models', None) else None
        _cli_budgets = config['budgets'] if getattr(args, 'budgets', None) else None
        decided_config(config, _ds)
        if _cli_models:
            kept = [m for m in _cli_models if m in config['models']]
            if not kept:
                raise ValueError(
                    f"--models {_cli_models}: none is one of {_ds}'s decided "
                    f"models {config['models']}")
            config['models'] = kept
        if _cli_budgets:
            kept = [b for b in _cli_budgets if b in config['budgets']]
            if not kept:
                raise ValueError(
                    f"--budgets {_cli_budgets}: none is one of {_ds}'s decided "
                    f"budgets {config['budgets']}")
            config['budgets'] = kept
        # The local smoke shrink (KIRBY_SMOKE_N) is re-applied on top: it is a
        # local-only override, and deeplift is the recorded local segfault.
        if _os.environ.get('KIRBY_SMOKE_N', '').strip():
            config['internal_evaluators'] = ['rf']
            config['importance_methods'] = ['lightgbm_gain']
            config['budgets'] = smoke_budgets()
    return config


# =============================================================================
# LOCAL-RUN GRID OVERRIDES
# =============================================================================
# Last thing this module does, so the names below are the ones every experiment
# imports. With no KIRBY_LOCAL_* variable set this is a no-op and a cluster run
# is unchanged. See tests/experiments/_local_grid.py.
# ⛔ OPTIONAL BY NECESSITY. _local_grid.py is untracked working-in-progress; a
# bare import of it was committed on 2026-09-18 while the module itself was not,
# so every experiment on the server died at import time with
# "cannot import name '_local_grid'". The overrides below are a local-run
# convenience gated on KIRBY_LOCAL_*, and a cluster run never wants them, so an
# absent module is the normal cluster case rather than an error.
try:
    from experiments import _local_grid as _lg  # noqa: E402
except ImportError:
    _lg = None

_LOCAL_GRID = _lg.apply() if _lg is not None else None
if _LOCAL_GRID:
    from kirby.find_best_hybrid import (  # noqa: F811
        DEFAULT_INTERNAL_EVALUATORS as _DIE)
    _b = _lg._ints('KIRBY_LOCAL_BUDGETS')
    if _b:
        FULL_BUDGETS = _b
        DISTANCE_BUDGETS = [x for x in _b if x <= max(DISTANCE_BUDGETS)] or _b
    INTERNAL_EVALUATORS = list(_DIE)
    _m = _LOCAL_GRID.get('auto_selected_models')
    if _m:
        DOWNSTREAM_MODELS = list(_m)
        # Four lists are built from DOWNSTREAM_MODELS higher up this file,
        # before this block runs, so each still named all six. Every one of
        # them is passed to find_best_hybrid BY NAME, and a named model is not
        # filtered by tier — so the six would be scored where the local grid
        # asked for one, and the search that is filed under those six names is
        # a different one from the search the precompute wrote.
        #
        # Measured: with MAIN_MODELS_WITH_KNN left at six, s02's pool read at
        # s02:306 missed the two-model search entirely and cold-built a
        # six-model one inside the experiment, including dnn, whose TensorFlow
        # import is broken in this environment. All three runs sat in state S
        # at 0% CPU for an hour on s02's first budget cell.
        DISTANCE_MODELS = list(DOWNSTREAM_MODELS) + ['knn']
        MAIN_MODELS_WITH_KNN = list(DOWNSTREAM_MODELS) + ['knn']
        RIDGE_MODELS = list(DOWNSTREAM_MODELS) + ['ridge']
        DOWNSTREAM_MODELS_NOISE = list(DOWNSTREAM_MODELS)
    # The distance tier's whole point is that a recipe is scored by predicting
    # each molecule from its neighbours, so knn stays as its internal evaluator
    # even when the downstream list is cut to two.
    DISTANCE_EVALUATORS = ['knn']
