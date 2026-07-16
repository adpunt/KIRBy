"""
Domain-agnostic hybrid representation search.

Accepts pre-computed numpy feature matrices and finds the best hybrid
configuration across models, budgets, allocation methods, and importance methods.

Works for any domain (molecular, antibody, fMRI, gene expression, etc.) — the
caller is responsible for generating representations and group labels.

Supports fine-tuned representations via `rep_factories`: per-fold callables that
produce train/val embeddings. These are pre-computed once and merged with frozen
base_reps during evaluation.

Usage:
    from kirby.find_best_hybrid import find_best_hybrid

    result = find_best_hybrid(
        base_reps={'ecfp4': X_ecfp, 'maccs': X_maccs, 'pdv': X_pdv},
        labels=y,
        groups=scaffold_groups,
        speed='full',
    )
    print(result.best_by_model)  # best hybrid per downstream model

    # With fine-tuned reps:
    from kirby.finetuning import make_antibody_lora_factory
    result = find_best_hybrid(
        base_reps={'kmer': X_kmer, 'aac': X_aac},
        labels=y,
        rep_factories={'lora_esm2': make_antibody_lora_factory(seqs, ...)},
    )
"""

import hashlib
import inspect
import os
import pickle
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold,
                                     StratifiedGroupKFold)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


# =============================================================================
# CACHE (filename-based, content-validated, no version numbers)
# =============================================================================
#
# Design rules:
#   1. Filename IS the cache key. No hashing. No version prefix.
#      A cache file's name tells you exactly what it holds.
#   2. Every cache file carries a `_cache_meta` header dict. Load validates
#      the header against the caller's expected metadata. Mismatch =
#      silently recompute, never crash.
#   3. Three layers, all under $KIRBY_CACHE_DIR (default ~/.cache/kirby):
#        searches/   — full SearchResult per (dataset, tier, aug_strategy, speed)
#        baselines/  — per-(dataset, rep, model, metric, folds, seed) scores
#        folds/      — per-(dataset, folds, seed) CV split indices
#   4. Atomic writes via tempfile + rename. Parallel writers to the same
#      filename collide on os.replace, not on partial content.
#   5. Failure to load is never fatal — the caller recomputes and rewrites.

def _cache_base() -> Path:
    return Path(os.environ.get('KIRBY_CACHE_DIR',
                                str(Path.home() / '.cache' / 'kirby')))


def _sanitize_filename_part(s: str) -> str:
    return str(s).replace('+', '_').replace('/', '_').replace(' ', '_')


def _search_cache_path(dataset_name: str, tier: str,
                       aug_strategy: str, speed: str,
                       condition_suffix: Optional[str] = None) -> Path:
    t = _sanitize_filename_part(tier)
    a = _sanitize_filename_part(aug_strategy)
    s = _sanitize_filename_part(speed)
    stem = f'{dataset_name}__{t}__{a}__{s}'
    if condition_suffix:
        stem = f'{stem}__{_sanitize_filename_part(condition_suffix)}'
    return _cache_base() / 'searches' / f'{stem}.pkl'


def _search_checkpoint_path(dataset_name: str, tier: str,
                            aug_strategy: str, speed: str,
                            condition_suffix: Optional[str] = None) -> Path:
    p = _search_cache_path(dataset_name, tier, aug_strategy, speed,
                           condition_suffix)
    return p.with_suffix('.ckpt.pkl')


def _canon_per_source_budget(v):
    """Canonical, hashable form of aug_per_source_budget for cache keys.

    Accepts None / int / {aug_name: N}. A dict becomes a sorted tuple of
    (name, N) pairs with falsy budgets dropped, so {'a':5,'b':0} and {'a':5}
    key identically. Returns None when nothing is capped, which keeps the key
    out of the meta entirely (see _make_search_meta).
    """
    if not v:
        return None
    if isinstance(v, dict):
        items = tuple(sorted((str(k), int(b)) for k, b in v.items() if b))
        return items or None
    return int(v)


def _baseline_cache_path(dataset_name: str, rep_name: str, model_name: str,
                         metric_name: str, n_folds: int, seed: int,
                         secondary_names: Optional[Tuple[str, ...]] = None,
                         noise_id: Optional[str] = None) -> Path:
    # Secondary metrics change the cached payload (a per-model secondary block
    # is stored next to the score), so they are part of the cache identity.
    # Omitted from the filename when absent, so baseline files written before
    # secondary metrics were cacheable still hit.
    sec = f"__sec-{'-'.join(secondary_names)}" if secondary_names else ''
    # noise_id MUST be in the FILENAME, not just the meta. Label noise is injected
    # per-fold on TRAIN rows only, so the global label array — and therefore
    # `labels_hash` in the meta — is IDENTICAL to the clean run. A noisy baseline
    # would otherwise hit the clean baseline file and silently return clean scores.
    # Omitted when absent, so every pre-existing clean baseline file still hits.
    nz = f'__{_sanitize_filename_part(noise_id)}' if noise_id else ''
    return (_cache_base() / 'baselines' /
            f'{dataset_name}__{rep_name}__{model_name}__'
            f'{metric_name}__{n_folds}f_{seed}{sec}{nz}.pkl')


def _folds_cache_path(dataset_name: str, n_folds: int, seed: int,
                      has_groups: bool) -> Path:
    g = 'grouped' if has_groups else 'ungrouped'
    return (_cache_base() / 'folds' /
            f'{dataset_name}__{n_folds}f_{seed}_{g}.pkl')


def _atomic_write_pickle(path: Path, obj: Any) -> bool:
    import tempfile
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(pickle.dumps(obj))
                f.flush()
                os.fsync(f.fileno())  # force bytes to disk BEFORE the rename, so
                # a node crash / NFS hiccup can't leave a renamed-but-truncated
                # cache file that loads as junk.
            os.replace(tmp, str(path))
            return True
        except BaseException:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except Exception:
                    pass
            raise
    except Exception:
        return False


_CACHE_STATS = {
    'search_hit': 0, 'search_miss': 0, 'search_stale': 0,
    'baseline_hit': 0, 'baseline_miss': 0, 'baseline_stale': 0,
    'hybrid_eval_hit': 0, 'hybrid_eval_miss': 0, 'hybrid_eval_stale': 0,
}


# Bump when augmentation default params or logic changes so old hybrid_eval
# cache entries (which don't include aug_max_candidates / aug_patience in
# their meta) get invalidated automatically. Old entries lack this key →
# meta-mismatch → silently recomputed on demand. No deletion needed.
# NOT bumped for aug_per_source_budget: that param defaults to None, which is
# byte-for-byte the old behaviour, and its meta key is only emitted when set.
# Bumping would cold-rebuild every existing augmented hybrid_eval entry.
AUG_LOGIC_VERSION = 2


def _reset_cache_stats() -> None:
    for k in _CACHE_STATS:
        _CACHE_STATS[k] = 0


def _cache_stats_summary() -> str:
    s = _CACHE_STATS
    return (
        f"cache stats — search: {s['search_hit']} hit / "
        f"{s['search_miss']} miss / {s['search_stale']} stale; "
        f"baseline: {s['baseline_hit']} hit / "
        f"{s['baseline_miss']} miss / {s['baseline_stale']} stale; "
        f"hybrid_eval: {s['hybrid_eval_hit']} hit / "
        f"{s['hybrid_eval_miss']} miss / {s['hybrid_eval_stale']} stale"
    )


def _load_cache_with_meta(
    path: Path,
    expected_meta: Dict[str, Any],
    # rep_hashes/aug_hashes/rep_hash excluded: neural/transformer reps
    # (ESM/prot_t5/geneformer/DNA-LMs/chembert etc.) are NOT bit-reproducible
    # across CPU thread counts / GPU / library versions, so their value-hash drifts
    # run-to-run and wrongly rejects valid caches (infinite-recompute churn on
    # molecular jobs). The SEARCH meta uses the plural rep_hashes/aug_hashes; the
    # per-(rep,model) BASELINE meta uses the singular rep_hash — all three are the
    # same non-reproducible content hash, so all three are ignored here or the
    # baseline layer would keep churning even after the search layer stopped. Cache
    # identity stays fully guarded by rep_names + rep_shape(s) + n_samples + seed +
    # labels_hash + groups_hash + budgets + tier + strategy + model/metric/fold
    # config (see _make_search_meta / _make_baseline_meta). A genuine rep change that
    # alters dimensionality is still caught by rep_shape(s); a target change is still
    # caught by labels_hash + n_samples.
    ignore_keys: Tuple[str, ...] = (
        'code_timestamp', 'rep_hashes', 'aug_hashes', 'rep_hash',
        'subset_hashes'),  # neural-rep value hash is not reproducible
    *,
    kind: str = 'search',
    verbose: bool = False,
) -> Optional[Any]:
    """Load a cache wrapper and validate its _cache_meta against expected.

    Returns payload on success, None on any failure (missing file,
    unpickle error, missing header, metadata mismatch). Failure is
    never fatal — caller should recompute and rewrite.

    Increments _CACHE_STATS[f'{kind}_hit' | f'{kind}_miss' | f'{kind}_stale']
    depending on outcome:
    - hit: file exists and meta matches.
    - miss: file doesn't exist.
    - stale: file exists but meta mismatches (or is corrupt).

    When verbose=True and a stale file is found, prints the first key that
    drifted — helps diagnose silent drift before it causes a wrong-answer.
    """
    # Diagnostic: KIRBY_CACHE_VERBOSE=1 forces the drift/miss reason to print on
    # EVERY load path (the main search load hardcodes verbose=False), so a single
    # run names the exact meta field that's rejecting a supposedly-DONE cache.
    if os.environ.get('KIRBY_CACHE_VERBOSE'):
        verbose = True
    if not path.exists():
        _CACHE_STATS[f'{kind}_miss'] = _CACHE_STATS.get(f'{kind}_miss', 0) + 1
        return None
    try:
        wrapper = pickle.loads(path.read_bytes())
    except Exception:
        _CACHE_STATS[f'{kind}_stale'] = _CACHE_STATS.get(f'{kind}_stale', 0) + 1
        if verbose:
            print(f"  Cache {path.name}: unpickle failed — will recompute.")
        return None
    if not isinstance(wrapper, dict):
        _CACHE_STATS[f'{kind}_stale'] = _CACHE_STATS.get(f'{kind}_stale', 0) + 1
        if verbose:
            print(f"  Cache {path.name}: not a dict wrapper — will recompute.")
        return None
    if '_cache_meta' not in wrapper or 'payload' not in wrapper:
        _CACHE_STATS[f'{kind}_stale'] = _CACHE_STATS.get(f'{kind}_stale', 0) + 1
        if verbose:
            print(f"  Cache {path.name}: missing envelope — will recompute.")
        return None
    stored = wrapper.get('_cache_meta') or {}
    for k, v in expected_meta.items():
        if k in ignore_keys:
            continue
        if k not in stored or stored[k] != v:
            _CACHE_STATS[f'{kind}_stale'] = _CACHE_STATS.get(f'{kind}_stale', 0) + 1
            if verbose:
                print(f"  Cache {path.name}: meta drift on '{k}' "
                      f"(stored={stored.get(k)!r}, expected={v!r}) "
                      f"— will recompute.")
            return None
    _CACHE_STATS[f'{kind}_hit'] = _CACHE_STATS.get(f'{kind}_hit', 0) + 1
    return wrapper['payload']


def _save_cache_with_meta(path: Path, meta: Dict[str, Any],
                          payload: Any) -> bool:
    return _atomic_write_pickle(path, {'_cache_meta': meta, 'payload': payload})


def _hash_bytes(b: bytes) -> int:
    return int.from_bytes(hashlib.sha1(b).digest()[:8], 'big', signed=False)


def _canonical_bytes(arr) -> bytes:
    """Reproducible bytes of a label/group array for content hashing.

    np.ascontiguousarray(x).tobytes() on an object/string-dtype array
    serialises Python object *pointers* (different in every process), so an
    "identical" string-group array hashes differently every run -> the cache
    self-drifts on groups_hash/labels_hash and cold-rebuilds forever. For
    object/string dtypes we factorize to canonical int codes (by sorted-unique
    value) first, which is process- and machine-stable.

    Numeric arrays pass through UNCHANGED (same bytes as the old
    np.ascontiguousarray(x).tobytes()), so existing integer-group/-label caches
    keep their exact hash and are NOT forced to rebuild.
    """
    a = np.asarray(arr).ravel()
    if a.dtype == object or a.dtype.kind in ('U', 'S'):
        _, inv = np.unique(a, return_inverse=True)
        a = inv.astype(np.int64)
    return np.ascontiguousarray(a).tobytes()


def _hash_array(arr: np.ndarray) -> int:
    """Content hash of a numpy array. SHA1 of tobytes + shape + dtype."""
    h = hashlib.sha1()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return int.from_bytes(h.digest()[:8], 'big', signed=False)


def _hash_rep_dict(reps: Dict[str, np.ndarray]) -> Tuple[Tuple[str, int], ...]:
    """Sorted tuple of (rep_name, content_hash) for cache metadata."""
    return tuple(sorted((k, _hash_array(v)) for k, v in reps.items()))


def _make_search_meta(
    dataset_name: str, tier: str, aug_strategy: str, speed: str,
    base_reps: Dict[str, np.ndarray], labels: np.ndarray,
    groups: Optional[np.ndarray], model_names: Optional[List[str]],
    metric_name: str, task_type: str, n_folds: int, seed: int,
    budgets: List[int],
    augmentations: Optional[Dict[str, np.ndarray]] = None,
    augmentation_strategy: str = 'uncapped',
    internal_evaluators: Optional[List[str]] = None,
    allocation_methods: Optional[List[str]] = None,
    importance_methods: Optional[List[str]] = None,
    adaptive_pruning: float = 0.15,
    augmentation_threshold: Optional[float] = None,
    augmentation_budget: int = 20,
    aug_per_source_budget: Optional[int] = None,
    # Behaviour-affecting params. Any change in any of these produces a different
    # search; they must be in the cache key or two non-equivalent runs collide on
    # the same file.
    candidate_ranking: str = 'rf_importance',
    greedy_tolerance: float = 0.0,
    screen_threshold: Optional[float] = None,
    two_phase: bool = False,
    refinement_top_n: int = 5,
    refinement_step_sizes: Optional[List[int]] = None,
    test_rep_subsets: bool = False,
    dimension_tiers: Optional[List[int]] = None,
    weight_method: Optional[str] = None,
    step_size: int = 5,
    secondary_metrics: Optional[List[str]] = None,
    # Explicit identifiers for objects that can't be hashed (cv_splitter,
    # rep_factories). Caller-supplied — see find_best_hybrid docstring.
    cv_splitter_id: Optional[str] = None,
    rep_factories_id: Optional[str] = None,
    # Multi-stage adaptive pruning (Phase 2). Schedule fields are added to
    # the cache meta only when non-default, so legacy callers (no schedule)
    # produce byte-identical cache keys to pre-Phase-2 runs.
    pruning_schedule: Optional[Dict[str, Any]] = None,
    # Per-model schedule overrides (Phase 3). Same compatibility rule:
    # absent / empty dict = no contribution to cache key.
    model_pruning_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    # Hyperparameter tuning. A tuned run and an untuned run are DIFFERENT
    # searches (different estimators) and must not share a cache file. Emitted
    # only when tuning is on, so every pre-existing untuned cache file still
    # matches (_load_cache_with_meta iterates the EXPECTED meta).
    tune: bool = False,
    tune_n_iter: int = 20,
) -> Dict[str, Any]:
    meta_extra: Dict[str, Any] = {}
    if tune:
        meta_extra['tune'] = True
        meta_extra['tune_n_iter'] = int(tune_n_iter)
    if pruning_schedule is not None:
        ckpts = tuple(sorted(int(b) for b in pruning_schedule.get('checkpoints', [])))
        td = float(pruning_schedule.get('tail_drop_per_checkpoint', 0.0))
        mc = int(pruning_schedule.get('min_configs_per_model', 1))
        if ckpts:
            meta_extra['pruning_checkpoints'] = ckpts
        if td > 0:
            meta_extra['pruning_tail_drop'] = td
        if mc != 1:
            meta_extra['pruning_min_configs'] = mc
    if model_pruning_overrides:
        # Canonicalize: sort by model name; freeze each override into a
        # tuple of sorted (k, v) pairs so the meta is hashable + stable.
        canonical = []
        for mname in sorted(model_pruning_overrides.keys()):
            ov = model_pruning_overrides[mname]
            ov_items = []
            for k in sorted(ov.keys()):
                v = ov[k]
                if isinstance(v, list):
                    v = tuple(sorted(v) if k == 'checkpoints' else v)
                ov_items.append((k, v))
            canonical.append((mname, tuple(ov_items)))
        if canonical:
            meta_extra['model_pruning_overrides'] = tuple(canonical)
    return {
        'kind': 'search',
        'dataset_name': dataset_name,
        'tier': tier,
        'aug_strategy': aug_strategy,
        'speed': speed,
        'n_samples': int(len(labels)),
        'rep_names': tuple(sorted(base_reps.keys())),
        'rep_shapes': tuple(sorted(
            (k, tuple(v.shape)) for k, v in base_reps.items())),
        'rep_hashes': _hash_rep_dict(base_reps),
        'model_names': tuple(sorted(model_names)) if model_names else None,
        'metric_name': metric_name,
        'task_type': task_type,
        'n_folds': n_folds,
        'seed': seed,
        'budgets': tuple(budgets),
        'labels_hash': _hash_bytes(_canonical_bytes(labels)),
        'groups_hash': (_hash_bytes(_canonical_bytes(groups))
                        if groups is not None else None),
        'aug_names': tuple(sorted(augmentations.keys())) if augmentations else (),
        'aug_hashes': (_hash_rep_dict(augmentations) if augmentations else ()),
        'augmentation_strategy': augmentation_strategy,
        'internal_evaluators': (tuple(sorted(internal_evaluators))
                                if internal_evaluators else None),
        'allocation_methods': (tuple(sorted(allocation_methods))
                               if allocation_methods else None),
        'importance_methods': (tuple(sorted(importance_methods))
                               if importance_methods else None),
        'adaptive_pruning': float(adaptive_pruning),
        'augmentation_threshold': (float(augmentation_threshold)
                                   if augmentation_threshold is not None else None),
        'augmentation_budget': int(augmentation_budget),
        # Emitted only when set. _load_cache_with_meta iterates the EXPECTED
        # meta, so omitting the key leaves every pre-existing search cache file
        # matching. Emitting it unconditionally would cold-rebuild all of them.
        **({'aug_per_source_budget': _canon_per_source_budget(aug_per_source_budget)}
           if _canon_per_source_budget(aug_per_source_budget) else {}),
        'candidate_ranking': candidate_ranking,
        'greedy_tolerance': float(greedy_tolerance),
        'screen_threshold': (float(screen_threshold)
                             if screen_threshold is not None else None),
        'two_phase': bool(two_phase),
        'refinement_top_n': int(refinement_top_n),
        'refinement_step_sizes': (tuple(refinement_step_sizes)
                                  if refinement_step_sizes else None),
        'test_rep_subsets': bool(test_rep_subsets),
        'dimension_tiers': (tuple(dimension_tiers)
                            if dimension_tiers else None),
        'weight_method': weight_method,
        'step_size': int(step_size),
        'secondary_metrics': (tuple(sorted(secondary_metrics))
                              if secondary_metrics else None),
        'cv_splitter_id': cv_splitter_id,
        'rep_factories_id': rep_factories_id,
        'code_timestamp': time.time(),
        **meta_extra,
    }


def _make_baseline_meta(
    dataset_name: str, rep_name: str, rep_shape: tuple,
    model_name: str, metric_name: str, n_folds: int, seed: int,
    labels: np.ndarray, groups: Optional[np.ndarray],
    secondary_names: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Any]:
    meta = {
        'kind': 'baseline',
        'dataset_name': dataset_name,
        'rep_name': rep_name,
        'rep_shape': tuple(rep_shape),
        'rep_hash': None,  # set by caller with _hash_array(X)
        'model_name': model_name,
        'metric_name': metric_name,
        'n_folds': n_folds,
        'seed': seed,
        'n_samples': int(len(labels)),
        'labels_hash': _hash_bytes(_canonical_bytes(labels)),
        'groups_hash': (_hash_bytes(_canonical_bytes(groups))
                        if groups is not None else None),
        # Bump when _evaluate_single's filter policy changes so old
        # unfiltered baseline scores don't get served stale.
        'hd_filter_version': HIGH_DIM_FILTER_VERSION,
        'hd_filter_threshold': HIGH_DIM_FILTER_THRESHOLD,
        'code_timestamp': time.time(),
    }
    # Only added when present. _load_cache_with_meta iterates over the EXPECTED
    # meta, so an absent key leaves pre-existing no-secondary files matching.
    if secondary_names:
        meta['secondary_names'] = tuple(secondary_names)
    return meta


def _make_folds_meta(
    dataset_name: str, n_folds: int, seed: int,
    n_samples: int, groups: Optional[np.ndarray],
    cv_splitter_id: Optional[str] = None,
    labels: Optional[np.ndarray] = None,
    task_type: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        'kind': 'folds',
        'dataset_name': dataset_name,
        'n_folds': n_folds,
        'seed': seed,
        'n_samples': n_samples,
        'groups_hash': (_hash_bytes(_canonical_bytes(groups))
                        if groups is not None else None),
        # Distinguishes default KFold/GroupKFold from custom splitters so a
        # different splitter never silently serves cached splits.
        'cv_splitter_id': cv_splitter_id,
        # task_type picks the splitter (StratifiedGroupKFold for classification,
        # GroupKFold otherwise) and StratifiedGroupKFold's splits depend on the
        # LABELS. Without both in the key, the same dataset/n_folds/seed/groups
        # would serve cached splits computed for a different label vector or a
        # different splitter.
        'task_type': task_type,
        'labels_hash': (_hash_bytes(_canonical_bytes(labels))
                        if labels is not None else None),
        'code_timestamp': time.time(),
    }


def _hybrid_eval_cache_path(dataset_name: str, subset_hash: str,
                            alloc: str, imp: str, evaluator: str,
                            budget: int, fold_idx: int, n_folds: int,
                            seed: int, model_name: str,
                            aug_strategy: str = 'none',
                            aug_names: Tuple[str, ...] = (),
                            aug_per_source_budget: Optional[int] = None,
                            noise_id: Optional[str] = None) -> Path:
    """Per-(rep_subset, config, fold, model) hybrid-eval cache path.

    Layout: hybrid_evals/{dataset}/{...}.pkl so one subdir per dataset
    stays under ~5k-10k files for typical speed=full precompute
    (budgets x allocs x imps x evals x folds × models with adaptive
    pruning). Flat-dir alternative would put ~130k files in one
    directory.

    subset_hash = short hex hash of the sorted rep names the allocation
    picked. For top_k allocations this is predictable from baselines
    before calling create_hybrid, so cache can be probed pre-compute.

    Augmentation pool disambiguation: the base filename does NOT encode
    aug_names, so two runs sharing a base subset + strategy but using DIFFERENT
    augmentation pools map to one path. The meta check rejects the mismatch (so
    answers are always correct) but each run evicts the other and nothing is
    ever reused. That only happens when one process sweeps several pools over
    one base — i.e. c21's per-condition pools. A normal precompute fixes one
    pool per dataset, so it never collides.

    The aug-name suffix is therefore emitted ONLY when aug_per_source_budget is
    set, which is exactly the sweeping caller. Emitting it unconditionally would
    rename — and so orphan — every existing augmented entry in the precompute.
    """
    a = _sanitize_filename_part(alloc)
    i = _sanitize_filename_part(imp)
    e = _sanitize_filename_part(evaluator)
    ags = _sanitize_filename_part(aug_strategy)
    m = _sanitize_filename_part(model_name)
    aug_sfx = ''
    _psb = _canon_per_source_budget(aug_per_source_budget)
    if _psb and aug_names:
        _h = hashlib.sha1(
            ('|'.join(sorted(aug_names)) + '#' + repr(_psb)).encode()
        ).hexdigest()[:12]
        aug_sfx = f'__ps{_h}'
    # Same reasoning as _baseline_cache_path: with train-only label noise the global
    # labels array is unchanged, so the meta's labels_hash cannot tell a noisy run
    # from a clean one. The noise id must be in the FILENAME or the two conditions
    # share a file and evict each other forever. Omitted when absent, so every
    # existing clean entry keeps its path.
    nz = f'__{_sanitize_filename_part(noise_id)}' if noise_id else ''
    fname = (f'{subset_hash}__{a}__{i}__{e}__b{budget}__'
             f'f{fold_idx}of{n_folds}_{seed}__{m}__{ags}{aug_sfx}{nz}.pkl')
    return _cache_base() / 'hybrid_evals' / dataset_name / fname


def _ft_embeddings_cache_path(dataset_name: str, factory_id: str,
                              rep_name: str, fold_idx: int,
                              n_folds: int, seed: int) -> Path:
    """Per-(dataset, factory_id, rep, fold) FT embedding cache path.

    One file per fold × rep. Factory_id is caller-supplied and carries
    version information so a new factory invalidates the cache.
    """
    fid = _sanitize_filename_part(factory_id)
    return (_cache_base() / 'ft_embeddings' /
            f'{dataset_name}__{fid}__{rep_name}__'
            f'f{fold_idx}of{n_folds}_{seed}.pkl')


def _predict_top_k_subset(
    alloc: str,
    base_rep_names: List[str],
    fold_alloc_scores: Optional[Dict[str, float]],
    baselines: Optional[Dict[str, Dict[str, float]]],
    evaluator_for_baseline: str,
    lower_is_better: bool,
) -> Optional[List[str]]:
    """If `alloc` is a 'top_N' pattern, return the N rep names that the
    allocator will select. Returns None for any other allocation style,
    because non-top_k methods use all input reps with varying weights —
    the rep subset isn't predictable pre-compute and dedup doesn't apply.

    Score sources (in priority order):
      1. fold_alloc_scores: alloc_cache entries are pre-shifted so
         "higher-is-better" regardless of the underlying metric
         direction (see alloc cache population in _run_search).
         Sort descending.
      2. baselines: raw per-rep per-model scores. For these,
         lower_is_better matters — we pick by direction.

    The prediction must match hybrid.allocate_performance_weighted's
    top_k scheme, which sorts baseline_scores descending (higher-is-
    better semantics on comparable scores).
    """
    if not alloc.startswith('top_'):
        return None
    try:
        k = int(alloc.split('_', 1)[1])
    except (ValueError, IndexError):
        return None
    if k <= 0 or k >= len(base_rep_names):
        return None

    used_alloc_scores = False
    scored: List[Tuple[str, float]] = []
    if fold_alloc_scores:
        for name in base_rep_names:
            v = fold_alloc_scores.get(name)
            if v is not None and np.isfinite(v):
                scored.append((name, float(v)))
        used_alloc_scores = len(scored) >= k

    if not used_alloc_scores and baselines:
        # Fallback: best-model-per-rep from the baseline dict.
        scored = []
        for name in base_rep_names:
            model_scores = baselines.get(name, {})
            vals = [v for mk, v in model_scores.items()
                    if mk not in ('dimension', '_secondary')
                    and isinstance(v, (int, float)) and np.isfinite(v)]
            if vals:
                best = min(vals) if lower_is_better else max(vals)
                scored.append((name, float(best)))
    if len(scored) < k:
        return None  # not enough scored reps to predict

    if used_alloc_scores:
        # alloc_cache is pre-shifted: always sort descending
        scored.sort(key=lambda t: t[1], reverse=True)
    else:
        # baselines are raw: respect direction
        scored.sort(key=lambda t: t[1], reverse=not lower_is_better)
    return sorted(n for n, _ in scored[:k])


def _make_hybrid_eval_meta(
    dataset_name: str, subset_names: List[str], subset_hashes: Tuple[Tuple[str, int], ...],
    alloc: str, imp: str, evaluator: str, budget: int,
    fold_idx: int, n_folds: int, seed: int, model_name: str,
    train_idx: np.ndarray, val_idx: np.ndarray,
    labels_hash: int, aug_strategy: str,
    aug_names: Tuple[str, ...], aug_hashes: Tuple[Tuple[str, int], ...],
    step_size: int, candidate_ranking: str, greedy_tolerance: float,
    augmentation_threshold: Optional[float], augmentation_budget: int,
    metric_name: str, task_type: str,
    aug_per_source_budget: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        'kind': 'hybrid_eval',
        'dataset_name': dataset_name,
        'subset_names': tuple(sorted(subset_names)),
        'subset_hashes': subset_hashes,
        'alloc': alloc,
        'imp': imp,
        'evaluator': evaluator,
        'budget': int(budget),
        'fold_idx': int(fold_idx),
        'n_folds': int(n_folds),
        'seed': int(seed),
        'model_name': model_name,
        'train_idx_hash': _hash_bytes(np.ascontiguousarray(train_idx).tobytes()),
        'val_idx_hash': _hash_bytes(np.ascontiguousarray(val_idx).tobytes()),
        'labels_hash': labels_hash,
        'aug_strategy': aug_strategy,
        'aug_names': aug_names,
        'aug_hashes': aug_hashes,
        'step_size': int(step_size),
        'candidate_ranking': candidate_ranking,
        'greedy_tolerance': float(greedy_tolerance),
        'augmentation_threshold': (float(augmentation_threshold)
                                   if augmentation_threshold is not None else None),
        'augmentation_budget': int(augmentation_budget),
        # Emitted only when set — see AUG_LOGIC_VERSION note above.
        **({'aug_per_source_budget': _canon_per_source_budget(aug_per_source_budget)}
           if _canon_per_source_budget(aug_per_source_budget) else {}),
        'aug_logic_version': AUG_LOGIC_VERSION,
        'metric_name': metric_name,
        'task_type': task_type,
        'code_timestamp': time.time(),
    }


def _subset_hash_str(subset_hashes: Tuple[Tuple[str, int], ...]) -> str:
    """Short hex hash of the (rep_name, content_hash) tuples, for filenames."""
    h = hashlib.sha1()
    for name, content_hash in subset_hashes:
        h.update(name.encode())
        h.update(str(content_hash).encode())
    return h.hexdigest()[:16]


def _subset_name_hex(subset_names) -> str:
    """Filename-safe hash of the sorted rep NAMES in a subset.

    The old hybrid-eval filename hashed rep VALUES (_hash_array of the
    embeddings). Neural/transformer embeddings are not bit-reproducible across
    runs, so the filename changed every run and the per-fold hybrid-eval cache
    NEVER hit -> silent wasted recompute on every neural rep. Rep names are a
    stable identity, so key the filename on the sorted names instead.
    """
    h = hashlib.sha1()
    for name in sorted(subset_names):
        h.update(name.encode())
        h.update(b'\x00')
    return h.hexdigest()[:16]


def _make_ft_embedding_meta(
    dataset_name: str, factory_id: str, rep_name: str,
    fold_idx: int, n_folds: int, seed: int,
    train_idx: np.ndarray, val_idx: np.ndarray,
    labels_hash: int, groups_hash: Optional[int],
) -> Dict[str, Any]:
    return {
        'kind': 'ft_embedding',
        'dataset_name': dataset_name,
        'factory_id': factory_id,
        'rep_name': rep_name,
        'fold_idx': int(fold_idx),
        'n_folds': int(n_folds),
        'seed': int(seed),
        # Fold identity: hash of the actual index tuples so we can detect
        # splitter drift even if the rest of the key matches.
        'train_idx_hash': _hash_bytes(np.ascontiguousarray(train_idx).tobytes()),
        'val_idx_hash': _hash_bytes(np.ascontiguousarray(val_idx).tobytes()),
        'labels_hash': labels_hash,
        'groups_hash': groups_hash,
        'code_timestamp': time.time(),
    }


def _load_or_compute_folds(
    dataset_name: Optional[str],
    splitter,
    use_groups: bool,
    labels: np.ndarray,
    groups: Optional[np.ndarray],
    n_folds: int,
    seed: int,
    cv_splitter_id: Optional[str] = None,
    verbose: bool = False,
    task_type: Optional[str] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Materialise CV fold indices, caching them per dataset.

    When dataset_name is None, caching is skipped (caller is not yet
    registry-aware) — the splits are still computed correctly, just not
    persisted. groups_hash, labels_hash, task_type and cv_splitter_id are the
    axes that invalidate (labels + task_type because classification uses
    StratifiedGroupKFold, whose splits are label-dependent).
    """
    # Always compute splits so the return type is stable.
    dummy_X = np.zeros((len(labels), 1))
    split_args = (dummy_X, labels, groups) if use_groups else (dummy_X, labels)

    if not dataset_name:
        return list(splitter.split(*split_args))

    path = _folds_cache_path(dataset_name, n_folds, seed,
                             has_groups=groups is not None)
    meta = _make_folds_meta(dataset_name, n_folds, seed, len(labels),
                            groups, cv_splitter_id=cv_splitter_id,
                            labels=labels, task_type=task_type)
    # Not counted in _CACHE_STATS today — folds are cheap; their cache is
    # a determinism guarantee more than a speed gain.
    cached = _load_cache_with_meta(path, meta, kind='search',
                                    verbose=False)
    if cached is not None:
        return cached

    folds = list(splitter.split(*split_args))
    _save_cache_with_meta(path, meta, folds)
    if verbose:
        print(f"  Saved folds cache to {path.name}")
    return folds


def _fit_model(m, X, y, groups=None):
    """Fit `m`, passing `groups` only to models whose fit() accepts it.

    The neural models (DNN/CNN/BNN/pretrained) carve an internal early-stopping
    holdout out of whatever they are handed. Left group-blind, that carve puts
    near-duplicate scaffolds on both sides, so early stopping sees an
    optimistically easy validation set and stops too late. Passing the fold's
    groups makes the carve group-aware.

    sklearn estimators do not accept `groups` in fit(), so this probes the
    signature rather than try/except-ing (a TypeError raised from INSIDE a
    model's own fit would otherwise be silently swallowed and retried blind).
    """
    if groups is not None:
        try:
            _params = inspect.signature(m.fit).parameters
        except (TypeError, ValueError):
            _params = {}
        if 'groups' in _params:
            return m.fit(X, y, groups=groups)
    return m.fit(X, y)


def _impute_nan(X_train, X_val):
    """Impute NaN with training median, drop all-NaN columns, clip inf.

    Training median is computed from X_train and applied to both
    X_train and X_val (no leakage from val into imputation).
    Columns that are entirely NaN in training are dropped from both.

    Returns:
        (X_train, X_val, had_imputation) — had_imputation is True if
        any NaN values were imputed or all-NaN columns were dropped.
    """
    had_imputation = False

    # Clip inf first
    X_train = np.clip(X_train, -1e10, 1e10)
    X_val = np.clip(X_val, -1e10, 1e10)

    # Drop columns that are entirely NaN in training (no signal)
    all_nan_cols = np.isnan(X_train).all(axis=0)
    if all_nan_cols.any():
        keep = ~all_nan_cols
        X_train = X_train[:, keep]
        X_val = X_val[:, keep]
        had_imputation = True

    # Per-feature median from training data only
    nan_mask_train = np.isnan(X_train)
    if nan_mask_train.any():
        had_imputation = True
        medians = np.nanmedian(X_train, axis=0)
        # Apply to train
        inds_train = np.where(nan_mask_train)
        X_train[inds_train] = medians[inds_train[1]]
        # Apply same medians to val
        nan_mask_val = np.isnan(X_val)
        if nan_mask_val.any():
            inds_val = np.where(nan_mask_val)
            X_val[inds_val] = medians[inds_val[1]]
    else:
        # No NaN in train — still check val
        nan_mask_val = np.isnan(X_val)
        if nan_mask_val.any():
            had_imputation = True
            medians = np.nanmedian(X_train, axis=0)
            inds_val = np.where(nan_mask_val)
            X_val[inds_val] = medians[inds_val[1]]

    return X_train, X_val, had_imputation


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Blueprint:
    """Minimal transferable recipe for recreating a hybrid.

    Carries metric + lower_is_better so that every downstream consumer
    knows the direction without guessing.  Published blueprints should
    always be round-tripped through to_dict() / from_dict().
    """
    reps: List[str]
    allocation_method: str
    importance_method: str
    budget: int
    internal_evaluator: str = 'match'
    step_size: int = 5
    speed: str = ''
    n_folds: int = 0
    seed: int = 42
    metric: str = 'r2'
    task_type: str = 'regression'
    augmentation_strategy: str = 'uncapped'
    lower_is_better: bool = False
    # Per-source cap applied to augmentation candidates before pooling: int, or
    # {aug_name: N}, or None for "pool every column". Part of the recipe, not a
    # search dimension — a rebuild that omits it pools every column and stops
    # reproducing the configuration the search actually chose. Carried here so
    # build_fold_hybrid / apply-time callers do not have to thread it by hand.
    aug_per_source_budget: Optional[Union[int, Dict[str, int]]] = None
    # Which objective the search optimized: 'predictive' (downstream model
    # performance) or 'similarity_sac' (held-out similarity preservation).
    # A sim-optimized blueprint is a DIFFERENT recipe than a predictive one and
    # consumers should not mix them; this field makes the provenance explicit.
    objective: str = 'predictive'
    # Per-rep allocation scores from the search (enables label-free apply).
    # {rep_name: float} — higher = more budget share.
    rep_scores: Optional[Dict[str, float]] = None

    def to_dict(self) -> dict:
        d = {
            'reps': self.reps,
            'allocation_method': self.allocation_method,
            'importance_method': self.importance_method,
            'internal_evaluator': self.internal_evaluator,
            'budget': self.budget,
            'step_size': self.step_size,
            'speed': self.speed,
            'n_folds': self.n_folds,
            'seed': self.seed,
            'metric': self.metric,
            'task_type': self.task_type,
            'augmentation_strategy': self.augmentation_strategy,
            'lower_is_better': self.lower_is_better,
            'objective': self.objective,
        }
        if self.rep_scores is not None:
            d['rep_scores'] = self.rep_scores
        # Omitted when unset so existing serialised blueprints round-trip
        # byte-identically. Dicts survive JSON as {name: N}.
        if self.aug_per_source_budget is not None:
            d['aug_per_source_budget'] = self.aug_per_source_budget
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'Blueprint':
        """Reconstruct a Blueprint from a dict (e.g. loaded from JSON).

        Ignores unknown keys so older/newer serialisations stay compatible.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


@dataclass
class FittedResult:
    """Concrete fitted result with feature indices and optional weights."""
    blueprint: Blueprint
    selected_indices: Dict[str, np.ndarray]
    feature_weights: Optional[Dict[str, np.ndarray]] = None
    n_features: int = 0
    augmentation_info: Optional[dict] = None


@dataclass
class SearchResult:
    """Full output of find_best_hybrid().

    Primary selection unit: best_by_model_budget[model][budget].
    Each downstream model gets its own best hybrid configuration.

    Every SearchResult carries metric_name + lower_is_better so that
    downstream code NEVER has to guess comparison direction.
    """
    best_by_model: Dict[str, dict]
    best_by_dimension: Dict[int, dict]
    performance_matrix: Dict[str, Dict[str, float]]
    baselines: Dict[str, Dict[str, float]]
    all_results: List[dict]
    metric_name: str = 'r2'
    lower_is_better: bool = False
    # Per-model fitted results: {model_name: FittedResult}
    fitted_results: Dict[str, FittedResult] = field(default_factory=dict)
    finetuned_embeddings: Optional[Dict[str, np.ndarray]] = None
    best_by_budget: Dict[int, dict] = field(default_factory=dict)
    # Primary selection unit: best features per (model, budget) pair.
    # {model_name: {budget: best_config_dict}}
    best_by_model_budget: Dict[str, Dict[int, dict]] = field(default_factory=dict)
    # Dataset provenance (set by callers who know the dataset name).
    dataset_name: Optional[str] = None
    # Per-fold importance scores keyed (fold_idx, importance_method) -> {rep_name: scores_array}.
    # Lets downstream experiments (c06 Option B, c15, c17, c19) reuse precompute's
    # importance instead of recomputing. Empty by default for back-compat with
    # older cache files.
    importance_cache: Dict = field(default_factory=dict)
    # Per-fold allocation scores keyed (fold_idx, evaluator[, metric_name]) -> {rep_name: float}.
    alloc_cache: Dict = field(default_factory=dict)
    # Per-(rep, model) per-fold BASELINE scores: {rep_name: {model_name: [f0, f1, ...]}}.
    # Parallels the scalar mean at baselines[rep][model] WITHOUT changing it —
    # additive so the leave-one-fold-out honest re-score can apply the same
    # held-out-fold tax to baseline reps as to hybrid candidates. Empty by
    # default for back-compat with older cache files / pickles. Only populated
    # for the models actually evaluated (same gate as baselines).
    baseline_fold_scores: Dict = field(default_factory=dict)

    def __setstate__(self, state):
        """Backward-compat: old pickled SearchResult files do not contain
        importance_cache / alloc_cache / baseline_fold_scores. Python's
        @dataclass does NOT invoke default_factory on unpickle (only on
        __init__), so accessing those fields on an old pickle would
        AttributeError. Fill in defaults."""
        self.__dict__.update(state)
        for _fname in ('importance_cache', 'alloc_cache', 'baseline_fold_scores'):
            if _fname not in self.__dict__:
                self.__dict__[_fname] = {}


# =============================================================================
# METRIC ABSTRACTION
# =============================================================================

def _concordance_index(y_true, y_pred):
    """Concordance index (CI) — standard DTA metric. O(n²), fine for CV folds."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = len(y_true)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            concordant += (y_true[i] - y_true[j]) * (y_pred[i] - y_pred[j]) > 0
            discordant += (y_true[i] - y_true[j]) * (y_pred[i] - y_pred[j]) < 0
    total = concordant + discordant
    return concordant / total if total > 0 else 0.5


def _safe_roc_auc(y_true, y_pred):
    """AUROC that handles both binary and multi-class safely.

    Binary: expects 1D scores/probabilities.
    Multi-class: expects (n, n_classes) probability matrix, uses OVR macro.
    Falls back to 0.5 if computation fails.
    """
    n_classes = len(np.unique(y_true))
    if n_classes <= 2:
        return roc_auc_score(y_true, y_pred)
    # Multi-class: need probability matrix
    if np.asarray(y_pred).ndim == 2:
        return roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')
    # Got class labels instead of probabilities — can't compute AUC
    raise ValueError(
        f"Multi-class AUROC requires probability matrix, got 1D predictions. "
        f"Use 'f1_weighted' metric for multi-class classification."
    )


def _sac_not_a_prediction_metric(y_true, y_pred):
    """Guard: 'sac' is a similarity-preservation objective computed on the
    representation geometry (X, y), NOT on model predictions. It is scored
    directly by the objective='similarity_sac' path, so metric_fn must never be
    invoked with (y_true, y_pred). If it is, something bypassed the sim path."""
    raise RuntimeError(
        "SAC is a geometry objective, not a prediction metric — it must be "
        "scored via objective='similarity_sac', not metric_fn(y_true, y_pred)."
    )


BUILTIN_METRICS = {
    'r2':          (r2_score, False),
    'rmse':        (lambda y, p: np.sqrt(mean_squared_error(y, p)), True),
    'mae':         (mean_absolute_error, True),
    'accuracy':    (accuracy_score, False),
    'auc':         (_safe_roc_auc, False),
    'spearman':    (lambda y, p: float(spearmanr(y, p)[0]), False),
    'pearson':     (lambda y, p: float(pearsonr(y, p)[0]), False),
    'ci':          (_concordance_index, False),
    'f1_weighted': (lambda y, p: f1_score(y, p, average='weighted'), False),
    'mcc':         (matthews_corrcoef, False),
    # SAC: higher is better. Scorer is a guard (see objective='similarity_sac').
    'sac':         (_sac_not_a_prediction_metric, False),
}

# Objective modes for the hybrid search. 'predictive' = the classic
# fit-model-then-score-predictions objective. 'similarity_sac' = select the
# hybrid whose geometry best preserves "similar molecules -> similar
# properties" (held-out SAC), fitting no downstream model.
OBJECTIVE_PREDICTIVE = 'predictive'
OBJECTIVE_SIMILARITY_SAC = 'similarity_sac'
_VALID_OBJECTIVES = (OBJECTIVE_PREDICTIVE, OBJECTIVE_SIMILARITY_SAC)


def _similarity_objective_score(X_val_s, y_val, is_clf, seed=42) -> float:
    """Similarity-preservation objective (higher-is-better) for the sim search.

    Scores the held-out fold geometry: Spearman correlation between pairwise
    rep-space distance and pairwise activity difference (SAC). No model is fit —
    this is a pure function of the (already-scaled) embedding and the labels.

    Reuses the canonical scalable SAC in
    kirby.utils.noise_mitigation_methods (single source of truth). Returns a
    finite float; degenerate folds (constant distances/labels) score 0.0.
    """
    from kirby.utils.noise_mitigation_methods import (
        similarity_activity_correlation_scalable,
    )
    y_arr = np.asarray(y_val)
    if len(y_arr) < 3:
        return 0.0
    try:
        return similarity_activity_correlation_scalable(
            np.asarray(X_val_s), y_arr, is_clf=is_clf, seed=seed)
    except Exception:
        return 0.0


# The full classic metric set stored per result. Ranking still uses
# the single designated metric (score_mean); these are stored ALONGSIDE so a
# result is self-describing. The designated metric is added on top of this set
# by the caller (so a non-classic ranking metric like spearman/ci/mcc is always
# present in the row too).
REGRESSION_METRIC_SET = ('r2', 'mae', 'mse', 'rmse')
CLASSIFICATION_METRIC_SET = ('accuracy', 'balanced_accuracy', 'precision',
                             'recall', 'f1', 'roc_auc')


def _compute_classic_metrics(model, X_val_s, y_val, y_pred, task_type) -> Dict[str, float]:
    """Compute the full classic metric set for one fold.

    Regression uses the continuous `y_pred`. Classification re-predicts labels
    (and probabilities for roc_auc) from the fitted `model`. Each metric is
    independently guarded — a metric that can't be computed on this fold (e.g.
    roc_auc with one class present) is omitted rather than failing the fold.
    """
    out: Dict[str, float] = {}
    try:
        if task_type == 'classification':
            y_lab = model.predict(X_val_s)
            for name, fn in (
                ('accuracy', lambda: accuracy_score(y_val, y_lab)),
                ('balanced_accuracy', lambda: balanced_accuracy_score(y_val, y_lab)),
                ('precision', lambda: precision_score(y_val, y_lab, average='weighted', zero_division=0)),
                ('recall', lambda: recall_score(y_val, y_lab, average='weighted', zero_division=0)),
                ('f1', lambda: f1_score(y_val, y_lab, average='weighted', zero_division=0)),
            ):
                try:
                    out[name] = float(fn())
                except Exception:
                    pass
            # roc_auc needs scores/probabilities, and at least 2 classes present.
            try:
                if hasattr(model, 'predict_proba'):
                    y_score = model.predict_proba(X_val_s)
                    if y_score.shape[1] == 2:
                        y_score = y_score[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_score = model.decision_function(X_val_s)
                else:
                    y_score = None
                if y_score is not None and len(np.unique(y_val)) >= 2:
                    out['roc_auc'] = float(_safe_roc_auc(y_val, y_score))
            except Exception:
                pass
        else:
            for name, fn in (
                ('r2', lambda: r2_score(y_val, y_pred)),
                ('mae', lambda: mean_absolute_error(y_val, y_pred)),
                ('mse', lambda: mean_squared_error(y_val, y_pred)),
                ('rmse', lambda: float(np.sqrt(mean_squared_error(y_val, y_pred)))),
            ):
                try:
                    out[name] = float(fn())
                except Exception:
                    pass
    except Exception:
        pass
    return out


def _aggregate_fold_metrics(fold_metric_dicts: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Mean/std each metric across folds → {metric: {'mean':.., 'std':..}}.

    A metric is aggregated over only the folds where it was computable, so a
    metric missing on some folds still reports a mean over the rest.
    """
    if not fold_metric_dicts:
        return {}
    names = set()
    for d in fold_metric_dicts:
        names.update(d.keys())
    agg: Dict[str, Dict[str, float]] = {}
    for name in sorted(names):
        vals = [d[name] for d in fold_metric_dicts
                if name in d and d[name] is not None and np.isfinite(d[name])]
        if vals:
            agg[name] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    return agg


def _resolve_metric(metric) -> Tuple[Callable, bool]:
    """Return (scorer_fn, lower_is_better).

    Accepts:
    - str: lookup in BUILTIN_METRICS
    - callable: assume metric(y_true, y_pred), higher is better
    - tuple (callable, lower_is_better): explicit specification
    """
    if isinstance(metric, str):
        if metric not in BUILTIN_METRICS:
            raise ValueError(
                f"Unknown metric '{metric}'. "
                f"Available: {list(BUILTIN_METRICS.keys())}"
            )
        return BUILTIN_METRICS[metric]

    if isinstance(metric, tuple) and len(metric) == 2:
        return metric[0], metric[1]

    if callable(metric):
        return metric, False

    raise TypeError(f"metric must be str, callable, or (callable, bool). Got {type(metric)}")


def _score_to_comparable(raw_score: float, lower_is_better: bool) -> float:
    """Convert a raw score so that higher is always better (for comparisons)."""
    return -raw_score if lower_is_better else raw_score



def _screen_reps(
    base_reps: Dict[str, np.ndarray],
    baselines: Dict[str, Dict[str, float]],
    lower_is_better: bool,
    threshold: float,
    *,
    fold_reps=None,
    verbose: bool = True,
) -> tuple:
    """Screen out weak reps before hybrid search.

    For each rep, takes the best score across all models from baselines.
    Converts to comparable (higher=better) and applies a gap-based cutoff:
    reps more than ``threshold`` fraction away from the best are dropped.

    Returns (filtered_base_reps, filtered_fold_reps).
    """
    # Best single-rep score across models
    best_per_rep = {}
    for rep_name, scores in baselines.items():
        model_scores = [v for k, v in scores.items()
                        if k not in ('dimension', '_secondary')
                        and isinstance(v, (int, float)) and np.isfinite(v)]
        if model_scores:
            if lower_is_better:
                best_per_rep[rep_name] = min(model_scores)
            else:
                best_per_rep[rep_name] = max(model_scores)

    if not best_per_rep:
        return base_reps, fold_reps

    comparable = {k: _score_to_comparable(v, lower_is_better)
                  for k, v in best_per_rep.items()}
    best_comparable = max(comparable.values())

    # Gap-based cutoff: drop reps more than `threshold` away from best
    cutoff = best_comparable - threshold * abs(best_comparable)

    surviving = set()
    for rep_name, comp in comparable.items():
        if comp >= cutoff:
            surviving.add(rep_name)

    # Always keep at least 2 reps
    if len(surviving) < 2:
        top2 = sorted(comparable, key=comparable.get, reverse=True)[:2]
        surviving = set(top2)

    screened = set(base_reps.keys()) - surviving
    if verbose and screened:
        print(f"\nRep screening (threshold={threshold}): "
              f"kept {len(surviving)}/{len(base_reps)} reps, "
              f"screened out: {sorted(screened)}")

    filtered_reps = {k: v for k, v in base_reps.items() if k in surviving}

    filtered_fold_reps = fold_reps
    if fold_reps is not None:
        filtered_fold_reps = {}
        for fold_idx, fold_data in fold_reps.items():
            filtered_fold_reps[fold_idx] = {
                k: v for k, v in fold_data.items() if k in surviving
            }

    return filtered_reps, filtered_fold_reps


# =============================================================================
# MODEL RESOLUTION
# =============================================================================

def _get_xgb_regressor():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=100, colsample_bytree=0.5,
                            random_state=42, n_jobs=-1, verbosity=0)
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, max_features='sqrt',
                                     random_state=42, n_jobs=-1)


def _get_xgb_classifier():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, colsample_bytree=0.5,
                             random_state=42, n_jobs=-1, verbosity=0)
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, max_features='sqrt',
                                      random_state=42, n_jobs=-1)


def _get_lgbm_regressor():
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, max_features='sqrt',
                                     random_state=42, n_jobs=-1)


def _get_lgbm_classifier():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, max_features='sqrt',
                                      random_state=42, n_jobs=-1)


def _get_catboost_regressor():
    try:
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=500, learning_rate=0.1, depth=6, verbose=0,
            random_seed=42)
    except ImportError:
        return _get_lgbm_regressor()


def _get_catboost_classifier():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=500, learning_rate=0.1, depth=6, verbose=0,
            random_seed=42)
    except ImportError:
        return _get_lgbm_classifier()


def _get_gp_regressor():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    # normalize_y=True centers targets so LBFGS starts near the optimum;
    # alpha=1e-6 regularizes Cholesky on ill-conditioned kernels.
    # Widened kernel bounds let the optimizer escape poor initial length_scale
    # on high-dimensional rep inputs without being clipped at the defaults.
    return GaussianProcessRegressor(
        kernel=RBF(length_scale_bounds=(1e-3, 1e5))
             + WhiteKernel(noise_level_bounds=(1e-5, 1e2)),
        n_restarts_optimizer=3,
        normalize_y=True,
        alpha=1e-6,
        random_state=42)


def _get_gp_classifier():
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    return GaussianProcessClassifier(kernel=RBF(), random_state=42)


class TanimotoKernel(__import__('sklearn.gaussian_process.kernels',
                                fromlist=['Kernel']).Kernel):
    """Tanimoto (Jaccard) kernel for binary fingerprint vectors.

    k(x, x') = <x, x'> / (||x||^2 + ||x'||^2 - <x, x'>)

    Returns 1 for identical non-zero vectors, 0 for zero vectors.
    No learnable hyperparameters.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X, dtype=np.float64)
        if Y is None:
            Y = X
        else:
            Y = np.asarray(Y, dtype=np.float64)

        XY = X @ Y.T
        X_sq = np.sum(X ** 2, axis=1)
        Y_sq = np.sum(Y ** 2, axis=1)

        # denom[i,j] = ||x_i||^2 + ||y_j||^2 - <x_i, y_j>
        denom = X_sq[:, None] + Y_sq[None, :] - XY
        K = np.where(denom > 0, XY / denom, 0.0)

        if eval_gradient:
            return K, np.empty((X.shape[0], Y.shape[0], 0))
        return K

    def diag(self, X):
        X = np.asarray(X, dtype=np.float64)
        norms = np.sum(X ** 2, axis=1)
        return np.where(norms > 0, 1.0, 0.0)

    def is_stationary(self):
        return False

    @property
    def hyperparameter_names(self):
        return []

    @property
    def theta(self):
        return np.empty(0)

    @theta.setter
    def theta(self, value):
        pass

    @property
    def bounds(self):
        return np.empty((0, 2))

    def clone_with_theta(self, theta):
        return TanimotoKernel()


def _get_dnn_regressor():
    from kirby.models.dnn import DNNRegressor
    return DNNRegressor()


def _get_dnn_classifier():
    from kirby.models.dnn import DNNClassifier
    return DNNClassifier()


def _get_cnn_regressor():
    from kirby.models.cnn import CNNRegressor
    return CNNRegressor()


def _get_cnn_classifier():
    from kirby.models.cnn import CNNClassifier
    return CNNClassifier()


DEFAULT_MODELS = {
    'rf': {
        'regressor': lambda: __import__('sklearn.ensemble', fromlist=['RandomForestRegressor']).RandomForestRegressor(
            n_estimators=100, max_features='sqrt', random_state=42, n_jobs=-1),
        'classifier': lambda: __import__('sklearn.ensemble', fromlist=['RandomForestClassifier']).RandomForestClassifier(
            n_estimators=100, max_features='sqrt', random_state=42, n_jobs=-1),
        'tier': 'fast',
    },
    'xgboost': {
        'regressor': _get_xgb_regressor,
        'classifier': _get_xgb_classifier,
        'tier': 'fast',
    },
    'knn': {
        'regressor': lambda: __import__('sklearn.neighbors', fromlist=['KNeighborsRegressor']).KNeighborsRegressor(
            n_neighbors=5, n_jobs=-1),
        'classifier': lambda: __import__('sklearn.neighbors', fromlist=['KNeighborsClassifier']).KNeighborsClassifier(
            n_neighbors=5, n_jobs=-1),
        'tier': 'full',
        '_adaptive_k': True,
    },
    'lightgbm': {
        'regressor': _get_lgbm_regressor,
        'classifier': _get_lgbm_classifier,
        'tier': 'full',
    },
    'mlp': {
        'regressor': lambda: __import__('sklearn.neural_network', fromlist=['MLPRegressor']).MLPRegressor(
            hidden_layer_sizes=(256,), random_state=42, max_iter=500,
            early_stopping=True, validation_fraction=0.1),
        'classifier': lambda: __import__('sklearn.neural_network', fromlist=['MLPClassifier']).MLPClassifier(
            hidden_layer_sizes=(256,), random_state=42, max_iter=500,
            early_stopping=True, validation_fraction=0.1),
        'tier': 'full',
    },
    'svm': {
        'regressor': lambda: __import__('sklearn.svm', fromlist=['SVR']).SVR(),
        'classifier': lambda: __import__('sklearn.svm', fromlist=['SVC']).SVC(probability=True),
        'tier': 'full',
        '_max_samples': 10000,
    },
    'gp': {
        'regressor': _get_gp_regressor,
        'classifier': _get_gp_classifier,
        'tier': 'full',
        '_max_samples': 1000,
    },
    'catboost': {
        'regressor': _get_catboost_regressor,
        'classifier': _get_catboost_classifier,
        'tier': 'extra',
    },
    'dnn': {
        'regressor': _get_dnn_regressor,
        'classifier': _get_dnn_classifier,
        'tier': 'extra',
    },
    'cnn': {
        'regressor': _get_cnn_regressor,
        'classifier': _get_cnn_classifier,
        'tier': 'extra',
    },
    'bnn': {
        'regressor': lambda: __import__('kirby.models.bnn', fromlist=['BNNRegressor']).BNNRegressor(),
        'classifier': lambda: __import__('kirby.models.bnn', fromlist=['BNNClassifier']).BNNClassifier(),
        'tier': 'extra',
    },
    'ngboost': {
        'regressor': lambda: __import__('ngboost', fromlist=['NGBRegressor']).NGBRegressor(
            n_estimators=500, learning_rate=0.05, random_state=42, verbose=False),
        'classifier': lambda: __import__('ngboost', fromlist=['NGBClassifier']).NGBClassifier(
            n_estimators=500, learning_rate=0.05, random_state=42, verbose=False),
        'tier': 'extra',
    },
}

# Tier ordering for filtering.
# fast/full: auto-selected by _resolve_models(None, speed=...).
# extra: only used when explicitly requested by name (e.g. DOWNSTREAM_MODELS_NOISE).
_TIER_ORDER = {'full': 1, 'extra': 2}

# SVM (SVR/SVC) is O(n^2)-O(n^3) in the number of training rows, and under the
# 'uncapped' augmentation strategy the training matrix is expanded by full
# augmentation, so the effective row count SVM fits is well above the raw n.
# This map lowers SVM's effective _max_samples for that strategy so the uncapped
# condition is skipped above the cutoff while capped/none keep the blanket guard
# (DEFAULT_MODELS['svm']['_max_samples']). _check_max_samples takes the stricter
# of the two limits. Changing which models are present for ('svm','uncapped') at
# 5000<n<=10000 invalidates cached hybrid metadata for those cells; smaller-n and
# capped/none cells, and all non-svm cache keys, are untouched.
_STRATEGY_MAX_SAMPLES = {
    ('svm', 'uncapped'): 5000,
}


# Dataset-specific model-skip policy is injected by the caller (the experiment
# layer), so the framework ships with none. fn(dataset_name, strategy) -> list[str].
_MODEL_SKIP_POLICY = None


def register_model_skip_policy(fn):
    """Register a callable fn(dataset_name, strategy) -> list[str] of model
    names to drop for that dataset/strategy. None means no skips."""
    global _MODEL_SKIP_POLICY
    _MODEL_SKIP_POLICY = fn


def _applied_skips(dataset_name, strategy):
    if _MODEL_SKIP_POLICY is None or not dataset_name:
        return set()
    try:
        return set(_MODEL_SKIP_POLICY(dataset_name, strategy) or [])
    except Exception:
        return set()


# Per-dataset budget policy, injected by the experiment layer (mirrors the
# model-skip policy above). fn(dataset_name, budgets) -> budgets. Applied INSIDE
# find_best_hybrid so BOTH precompute and experiment-time calls cap the budget
# list identically — the budget is part of the cache key, so a shared cap keeps
# precompute and experiments requesting the same rows (no on-the-fly recompute).
_BUDGET_POLICY = None


def register_budget_policy(fn):
    """Register a callable fn(dataset_name, budgets) -> budgets that caps or
    transforms the budget list for a dataset. None means no transform."""
    global _BUDGET_POLICY
    _BUDGET_POLICY = fn


def _apply_budget_policy(dataset_name, budgets):
    if _BUDGET_POLICY is None or not dataset_name or budgets is None:
        return budgets
    try:
        out = _BUDGET_POLICY(dataset_name, list(budgets))
        # Fall back to the original list if the policy filtered everything out
        # (e.g. a single native-dim budget above the cap) — never zero budgets.
        return list(out) if out else budgets
    except Exception:
        return budgets


def _resolve_models(models, task_type: str, speed: str,
                     n_samples: Optional[int] = None,
                     dataset_name: Optional[str] = None,
                     strategy: Optional[str] = None) -> Dict[str, Any]:
    """Resolve model spec to dict of {name: sklearn_model_instance}.

    models can be:
    - None: use defaults filtered by speed tier
    - List[str]: lookup by name
    - Dict[str, model]: custom models (passthrough)
    - Single model instance: wrap as {'custom': model}

    Models with _max_samples are auto-skipped when n_samples exceeds the limit.
    """
    key = 'classifier' if task_type == 'classification' else 'regressor'
    max_tier = _TIER_ORDER.get(speed, 1)
    skip_set = _applied_skips(dataset_name, strategy)

    def _check_max_samples(name, spec):
        max_s = spec.get('_max_samples')
        # A (model, strategy) override may lower the effective cap below the
        # blanket _max_samples (e.g. SVM uncapped at n>5000). Take the stricter
        # limit.
        strat_cap = _STRATEGY_MAX_SAMPLES.get((name, strategy))
        if strat_cap is not None:
            max_s = strat_cap if max_s is None else min(max_s, strat_cap)
        if max_s is not None and n_samples is not None and n_samples > max_s:
            print(f"  Skipping {name}: n_samples={n_samples} > _max_samples={max_s}"
                  f" (strategy={strategy!r})")
            return False
        return True

    def _check_skip(name):
        if name in skip_set:
            print(f"  Skipping {name}: empirical skip for "
                  f"dataset={dataset_name!r} strategy={strategy!r}")
            return False
        return True

    if models is None:
        resolved = {}
        for name, spec in DEFAULT_MODELS.items():
            if _TIER_ORDER.get(spec['tier'], 0) <= max_tier:
                if _check_max_samples(name, spec) and _check_skip(name):
                    resolved[name] = spec[key]()
        return resolved

    if isinstance(models, list):
        resolved = {}
        for name in models:
            if name not in DEFAULT_MODELS:
                raise ValueError(
                    f"Unknown model '{name}'. "
                    f"Available: {list(DEFAULT_MODELS.keys())}"
                )
            if _check_max_samples(name, DEFAULT_MODELS[name]) and _check_skip(name):
                resolved[name] = DEFAULT_MODELS[name][key]()
        return resolved

    if isinstance(models, dict):
        return {name: deepcopy(m) for name, m in models.items()}

    # Single model instance
    if hasattr(models, 'fit') and hasattr(models, 'predict'):
        return {'custom': deepcopy(models)}

    raise TypeError(
        f"models must be None, list of names, dict of models, or a single model. "
        f"Got {type(models)}"
    )


def get_model(name: str, task_type: str = 'regression',
              n_samples: Optional[int] = None) -> Any:
    """Return a fresh model instance from DEFAULT_MODELS.

    Returns None (with warning) if n_samples exceeds the model's _max_samples.
    Useful for survey scripts to avoid duplicating model definitions.

    Args:
        name: Model name (e.g. 'lightgbm', 'gp', 'dnn').
        task_type: 'regression' or 'classification'.
        n_samples: Dataset size — models with _max_samples auto-skip if exceeded.

    Returns:
        Model instance, or None if skipped due to size guard.
    """
    if name not in DEFAULT_MODELS:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(DEFAULT_MODELS.keys())}")

    spec = DEFAULT_MODELS[name]
    max_s = spec.get('_max_samples')
    if max_s is not None and n_samples is not None and n_samples > max_s:
        print(f"  Skipping {name}: n_samples={n_samples} > _max_samples={max_s}")
        return None

    key = 'classifier' if task_type == 'classification' else 'regressor'
    return spec[key]()


def _predict_for_metric(model, X, metric_fn):
    """Predict using predict_proba for binary AUC, else predict (class labels).

    Only returns probabilities when metric_fn is the built-in AUC metric.
    Other binary classification metrics (accuracy, f1, mcc) need class labels.
    """
    _auc_fn = BUILTIN_METRICS.get('auc', (None,))[0]
    needs_proba = (metric_fn is _auc_fn)

    if needs_proba and hasattr(model, 'predict_proba'):
        try:
            y_prob = model.predict_proba(X)
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                return y_prob[:, 1]
        except Exception:
            pass
    return model.predict(X)


def _detect_task_type(labels: np.ndarray) -> str:
    """Auto-detect regression vs classification from labels.

    LEGACY HEURISTIC — do not use in new code. find_best_hybrid and the
    experiment entry points now require task_type explicitly (see
    DATASET_REGISTRY / load_data). This function remains only for two
    internal callers:

    1. tune_models (callers: find_best_hybrid_{molecular,tcga,dta}.py) —
       passes task_type=None by default, still auto-detects here.
    2. _evaluate_single / _evaluate_baselines high-dim feature filter —
       used as a fallback when task_type is not threaded through (older
       internal call sites). The explicit task_type is preferred.

    Silent misclassification here would mean the wrong type of LightGBM
    (Regressor vs Classifier) running the HD filter — it would still
    produce *a* filter, but with the wrong inductive bias. Registered
    datasets never hit this path now.
    """
    unique = np.unique(labels)
    if len(unique) <= 20 and np.allclose(labels, labels.astype(int)):
        return 'classification'
    return 'regression'


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

_HP_SEARCH_SPACES = {
    'rf': {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [5, 10, 20, None],
        'max_features': ['sqrt', 'log2', 0.3, 0.5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    'xgboost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.3],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.3, 0.5, 0.7, 1.0],
    },
    'lightgbm': {
        'n_estimators': [50, 100, 200],
        'max_depth': [-1, 5, 10, 20],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 63, 127],
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 11, 15],
        'weights': ['uniform', 'distance'],
    },
    'mlp': {
        'hidden_layer_sizes': [(128,), (256,), (256, 128), (512,)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01],
    },
    'svm': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto'],
    },
    'gp': {
        'kernel': [
            __import__('sklearn.gaussian_process.kernels', fromlist=['RBF']).RBF(),
            __import__('sklearn.gaussian_process.kernels', fromlist=['RBF', 'WhiteKernel']).RBF()
            + __import__('sklearn.gaussian_process.kernels', fromlist=['WhiteKernel']).WhiteKernel(),
            __import__('sklearn.gaussian_process.kernels', fromlist=['Matern']).Matern(nu=1.5),
            __import__('sklearn.gaussian_process.kernels', fromlist=['Matern']).Matern(nu=2.5),
            TanimotoKernel(),
        ],
        'alpha': [1e-10, 1e-5, 1e-2, 0.1],
    },
    'catboost': {
        'iterations': [200, 500, 1000],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 10],
    },
    # dnn: skip — too slow for RandomizedSearchCV
    # cnn: skip — too slow for RandomizedSearchCV
}


def _count_grid_size(param_grid: dict) -> int:
    """Count total combinations in a parameter grid."""
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size


def tune_models(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: Optional[np.ndarray] = None,
    task_type: Optional[str] = None,
    speed: str = 'fast',
    n_iter: int = 20,
    n_folds: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Tune model hyperparameters via RandomizedSearchCV on concatenated reps.

    Runs RandomizedSearchCV for each model in the speed tier
    using the concatenation of all base_reps as features. Returns a dict of
    tuned model instances ready to pass to find_best_hybrid(models=...).

    ★ LEAKAGE WARNING — read before calling this directly.
    This tunes on EVERY row of whatever you hand it. If you tune on the full
    dataset and then cross-validate that same dataset and report the fold
    scores, the hyperparameters were chosen using the folds you are reporting.
    That is tuning on the test fold, and it is exactly what the CLI drivers used
    to do with --tune.

    The supported way to tune is `find_best_hybrid(tune=True)`, which calls this
    on the data find_best_hybrid was handed. Under nested_cv_evaluate that data
    is outer-train ONLY, so tuning never sees the outer test fold and the
    reported score stays honest. Do not hand-roll a pre-tune.

    Args:
        base_reps: Dict of {name: ndarray} base representations.
        labels: Target values.
        groups: Group labels for GroupKFold (optional).
        task_type: 'regression' or 'classification' (auto-detected if None).
        speed: Speed tier — determines which models to tune.
        n_iter: Max RandomizedSearchCV iterations per model.
        n_folds: Number of CV folds for tuning.
        seed: Random seed.

    Returns:
        Dict[str, model] — tuned model instances keyed by model name.
    """
    from sklearn.model_selection import (
        RandomizedSearchCV, GroupKFold, KFold, StratifiedKFold,
    )

    if task_type is None:
        task_type = _detect_task_type(labels)

    # Concatenate all base reps into a single feature matrix
    # NOTE: scaling is done inside Pipeline per CV fold (no pre-scaling leakage)
    X = np.hstack([base_reps[name] for name in sorted(base_reps.keys())])
    y = labels

    # Build CV splitter
    if groups is not None:
        n_unique_groups = len(np.unique(groups))
        actual_folds = min(n_folds, n_unique_groups)
        cv = GroupKFold(n_splits=actual_folds)
        cv_args = {'groups': groups}
    elif task_type == 'classification':
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        cv_args = {}
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        cv_args = {}

    # Scoring metric
    if task_type == 'classification':
        scoring = 'accuracy'
    else:
        scoring = 'r2'

    # Resolve which models to tune (pass n_samples for size guards)
    default_models = _resolve_models(None, task_type, speed,
                                      n_samples=len(y))

    tuned = {}
    for model_name, model in default_models.items():
        if model_name in ('dnn', 'cnn'):
            # DNN/CNN too slow for RandomizedSearchCV — uses internal early stopping
            tuned[model_name] = model
            print(f"  {model_name}: skip (uses internal early stopping)")
            continue

        param_grid = _HP_SEARCH_SPACES.get(model_name)
        if param_grid is not None:
            # Filter out HP params that don't exist on the actual model
            # (e.g., catboost fallback to LightGBM won't have 'iterations')
            valid_params = model.get_params().keys() if hasattr(model, 'get_params') else set()
            filtered_grid = {k: v for k, v in param_grid.items() if k in valid_params}
            if not filtered_grid:
                param_grid = None  # all params invalid → skip tuning
            elif len(filtered_grid) < len(param_grid):
                param_grid = filtered_grid
        if param_grid is None:
            tuned[model_name] = model
            print(f"  {model_name}: skip (no search space defined)")
            continue

        # Cap n_iter at total grid size to avoid sampling duplicates
        grid_size = _count_grid_size(param_grid)
        actual_iter = min(n_iter, grid_size)

        try:
            # Wrap model in Pipeline so StandardScaler fits per CV fold
            from sklearn.pipeline import Pipeline as _Pipeline
            pipe = _Pipeline([('scaler', StandardScaler()), ('model', deepcopy(model))])
            pipe_grid = {f'model__{k}': v for k, v in param_grid.items()}

            search = RandomizedSearchCV(
                pipe, pipe_grid,
                n_iter=actual_iter,
                cv=cv,
                scoring=scoring,
                random_state=seed,
                n_jobs=1,
                error_score=np.nan,
            )
            search.fit(X, y, **cv_args)

            # Extract the tuned model from the Pipeline (caller's _evaluate_single
            # does its own per-fold scaling, so we return the raw model)
            best_model = search.best_estimator_.named_steps['model']

            # Silence verbosity on tuned estimators
            # LightGBM: verbose=-1 (silent), RF/MLP: verbose=0 (silent)
            if 'LGBM' in type(best_model).__name__:
                best_model.set_params(verbose=-1)
            elif hasattr(best_model, 'verbose'):
                best_model.set_params(verbose=0)
            if hasattr(best_model, 'verbosity'):
                best_model.set_params(verbosity=0)

            tuned[model_name] = best_model
            # Report best params without 'model__' prefix for readability
            clean_params = {k.replace('model__', ''): v
                           for k, v in search.best_params_.items()}
            print(f"  {model_name}: {search.best_score_:.4f} "
                  f"({actual_iter} iters, best: {clean_params})")

        except Exception as e:
            print(f"  {model_name}: tuning FAILED ({e}), using defaults")
            tuned[model_name] = model

    return tuned


# =============================================================================
# SPEED TIER CONFIGURATION
# =============================================================================

SPEED_TIERS = {
    # Single tier. Faster searches go through the multi-stage
    # adaptive_pruning_schedule (see PRUNING_MODES), which keeps the cache key
    # identical to production rather than diverging it via a separate tier.
    'full': {
        'importance_methods': ['deeplift', 'lightgbm_gain', 'treeshap'],
        'allocation_methods': [
            'top_2', 'top_3', 'top_5',
            'performance_weighted',
            'entropy_1.0',
            'power_2',
            'rank_linear',
            'entropy_2.0',
            'rank_exp_d0.5',
        ],
        'internal_evaluators': ['match', 'rf', 'svm', 'lightgbm', 'mlp'],
        'n_folds': 5,
    },
}

# Per-model maximum budget. Models sensitive to high dimensionality
# (curse of dimensionality, kernel explosion, parameter count) are capped.
# Tree-based models handle high-d natively via feature splitting. Slow-model
# budgets above ~250 contributed dominant wall time without contributing wins.
MODEL_MAX_BUDGET = {
    'gp': 1000,
    'knn': 2000,
    'svm': 500,
    'mlp': 1000,
    'bnn': 1000,
    'dnn': 1000,
    'cnn': 1000,
    # No cap: rf, xgboost, lightgbm, catboost, ngboost
}


# Default adaptive-pruning schedule. Empty checkpoints + tail_drop=0 makes the
# fold-0 mid-flight pruning a no-op, so behavior is identical to the legacy
# single-stage adaptive_pruning=0.15 path. Callers wanting more aggressive
# pruning override this with an explicit schedule (see f16_multistage_pruning
# for calibrated schedules).
DEFAULT_PRUNING_SCHEDULE: Dict[str, Any] = {
    'checkpoints': [],                  # budgets at which to mid-flight prune
    'tail_drop_per_checkpoint': 0.0,    # bottom X% of (alloc, imp) pairs to drop
    'final_strategy_threshold': 0.15,   # end-of-fold-0 strategy prune (existing)
    'min_configs_per_model': 1,         # floor — best_by_model[m] always exists
}


def _normalize_pruning_schedule(
    adaptive_pruning: Union[bool, float, None] = True,
    adaptive_pruning_schedule: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Translate legacy adaptive_pruning + new schedule into a canonical dict.

    Precedence: explicit schedule wins. If only legacy adaptive_pruning is
    given, build a schedule with empty checkpoints (current behavior).

    Returns a dict with all four schedule fields present.
    """
    def _coerce(out: Dict[str, Any]) -> Dict[str, Any]:
        out['checkpoints'] = sorted(int(float(b)) for b in out.get('checkpoints', []))
        out['tail_drop_per_checkpoint'] = float(
            out.get('tail_drop_per_checkpoint', 0.0))
        out['final_strategy_threshold'] = float(
            out.get('final_strategy_threshold', 0.15))
        out['min_configs_per_model'] = int(
            out.get('min_configs_per_model', 1))
        return out

    if adaptive_pruning_schedule is not None:
        out = dict(DEFAULT_PRUNING_SCHEDULE)
        out.update(adaptive_pruning_schedule)
        return _coerce(out)

    # Legacy path: translate adaptive_pruning to schedule.final_strategy_threshold
    out = dict(DEFAULT_PRUNING_SCHEDULE)
    if adaptive_pruning is True:
        out['final_strategy_threshold'] = 0.15
    elif not adaptive_pruning:
        out['final_strategy_threshold'] = 0.0
    elif isinstance(adaptive_pruning, (int, float)):
        out['final_strategy_threshold'] = float(adaptive_pruning)
    return out


def _resolve_model_schedule(
    model_name: str,
    global_schedule: Dict[str, Any],
    model_overrides: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Merge per-model override on top of the global schedule for one model.

    Models not listed in `model_overrides` use the global schedule unchanged.
    Override fields are shallow-merged (e.g. an override can specify only
    `checkpoints` and inherit the rest from the global schedule).
    """
    if not model_overrides or model_name not in model_overrides:
        return global_schedule
    merged = dict(global_schedule)
    merged.update(model_overrides[model_name])
    # Re-coerce in case override supplied wrongly-typed values
    merged['checkpoints'] = sorted(int(float(b)) for b in merged.get('checkpoints', []))
    merged['tail_drop_per_checkpoint'] = float(
        merged.get('tail_drop_per_checkpoint', 0.0))
    merged['final_strategy_threshold'] = float(
        merged.get('final_strategy_threshold', 0.15))
    merged['min_configs_per_model'] = int(
        merged.get('min_configs_per_model', 1))
    return merged


def _compute_tail_prunes(
    fold0_results: List[Tuple[Tuple, Dict[str, Any]]],
    checkpoint_budget: int,
    tail_drop: float,
    min_configs: int,
    lower_is_better: bool,
    model_filter: Optional[set] = None,
) -> Dict[Tuple[str, str], set]:
    """At a checkpoint budget, compute (alloc, imp) pairs to drop per group.

    Within each (model_name, evaluator) group restricted to fold-0 results
    at budget <= checkpoint_budget:
      1. Rank by fold-0 score_mean (NaN sinks to bottom).
      2. Keep top max(min_configs, round(n * (1 - tail_drop))).
      3. Identify dropped (alloc, imp) pairs.

    Args:
        model_filter: if set, only consider groups whose model_name is in this
            set. Used by per-model schedules so e.g. an aggressive GP-specific
            checkpoint at b=10 only fires on GP groups.

    Returns: {(model_name, evaluator): {(alloc, imp), ...}} for dropped pairs.
    Configs at higher budgets in the same group with these (alloc, imp) pairs
    will never be evaluated.
    """
    from collections import defaultdict
    if tail_drop <= 0.0:
        return {}
    by_group: Dict[Tuple[str, str], List] = defaultdict(list)
    for ck, result in fold0_results:
        # ck = (rep_set_name, alloc, bgt, imp, model_name, eval_name)
        _, alloc, bgt, imp, model_name, eval_name = ck
        if bgt > checkpoint_budget:
            continue
        if model_filter is not None and model_name not in model_filter:
            continue
        by_group[(model_name, eval_name)].append((alloc, imp, result))

    dropped: Dict[Tuple[str, str], set] = {}
    sentinel = float('inf') if lower_is_better else float('-inf')
    for grp, entries in by_group.items():
        if not entries:
            continue
        def _score(e):
            s = e[2].get('score_mean')
            if s is None or (isinstance(s, float) and np.isnan(s)):
                return sentinel
            return s
        entries_sorted = sorted(entries, key=_score, reverse=not lower_is_better)
        n = len(entries_sorted)
        n_keep = max(min_configs, int(round(n * (1 - tail_drop))))
        kept = {(a, i) for a, i, _ in entries_sorted[:n_keep]}
        all_pairs = {(a, i) for a, i, _ in entries_sorted}
        dropped_for_grp = all_pairs - kept
        if dropped_for_grp:
            dropped[grp] = dropped_for_grp
    return dropped


# =============================================================================
# INTERNAL EVALUATION FUNCTIONS
# =============================================================================

def _make_cv_splitter(groups: Optional[np.ndarray], n_folds: int, seed: int,
                      cv_splitter=None, task_type: Optional[str] = None):
    """Return (splitter, use_groups).

    Group-aware always. For classification the full-fold case uses
    StratifiedGroupKFold so classes stay balanced across folds WHILE groups are
    respected (TCGA clf folds were previously unstratified → imbalanced folds →
    noisy clf metrics). NEVER falls back to a random k-fold split (the
    non-negotiable no-random-splits rule): a groupless / single-group request
    for k-fold RAISES so the caller fixes the upstream grouping instead of
    silently random-splitting. task_type defaults to None → GroupKFold
    (backward-compatible for the ~callers that don't thread it).
    """
    if cv_splitter is not None:
        return cv_splitter, False
    _is_clf = (task_type == 'classification')
    if n_folds == 1:
        # Single 80/20 split (intermediary quick-check mode, NOT production CV).
        if groups is not None:
            return GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed), True
        # No groups → a leak-free split is impossible. NO random ShuffleSplit
        # fallback (forbidden); mirror the production branch below and raise
        # instead.
        raise ValueError(
            "_make_cv_splitter: cannot build a group-aware single 80/20 "
            "quick-check split — groups is None. Supply scaffold/site/identity/"
            "complex groups. No random-split fallback."
        )
    if groups is not None:
        n_unique = len(np.unique(groups))
        if n_unique >= n_folds:
            if _is_clf:
                return StratifiedGroupKFold(
                    n_splits=n_folds, shuffle=True, random_state=seed), True
            return GroupKFold(n_splits=n_folds), True
        # Not enough groups for requested n_folds — reduce folds to match groups
        # (group-aware; stratification with very few groups is unreliable, so the
        # reduced case stays GroupKFold for both task types).
        if n_unique >= 2:
            import warnings
            warnings.warn(
                f"Only {n_unique} unique groups for {n_folds}-fold CV. "
                f"Reducing to {n_unique}-fold GroupKFold to preserve group structure.",
                stacklevel=2,
            )
            return GroupKFold(n_splits=n_unique), True
    # groups is None, or only 1 unique group → a leak-free split is impossible.
    # NO random fallback (forbidden). Raise so the upstream grouping is fixed.
    raise ValueError(
        "_make_cv_splitter: cannot build a group-aware "
        f"{n_folds}-fold split — "
        f"{'groups is None' if groups is None else 'only 1 unique group'}. "
        "Supply scaffold/site/identity/complex groups. No random-split "
        "fallback."
    )


def _precompute_fold_reps(
    rep_factories: Dict[str, Callable],
    labels: np.ndarray,
    groups: Optional[np.ndarray],
    n_folds: int,
    seed: int,
    cv_splitter=None,
    verbose: bool = True,
    dataset_name: Optional[str] = None,
    factory_ids: Optional[Dict[str, str]] = None,
    cv_splitter_id: Optional[str] = None,
    task_type: Optional[str] = None,
) -> Tuple[Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]], List[Tuple[np.ndarray, np.ndarray]]]:
    """Pre-compute fine-tuned representations for each fold.

    Calls each factory once per fold. This happens BEFORE the search grid,
    so fine-tuning cost is n_folds × n_factories (NOT per search config).

    When dataset_name and factory_ids are supplied, per-fold embeddings are
    cached to ~/.cache/kirby/ft_embeddings/ so subsequent calls with the
    same factory_id, fold indices, and labels hit the cache. Different
    factory versions (different id) don't collide.

    Args:
        rep_factories: {rep_name: factory_callable}
        labels: Target values.
        groups: Group labels for GroupKFold.
        n_folds: Number of CV folds.
        seed: Random seed.
        verbose: Print progress.
        dataset_name: Optional — required for FT cache.
        factory_ids: Optional {rep_name: factory_id_string} — required for
            FT cache. Caller-supplied identifier, typically a version
            string. Mismatched id = cache miss; no id = cache skipped.
        cv_splitter_id: Optional identifier for a custom splitter (used
            by the folds cache layer; not the FT cache directly).

    Returns:
        (fold_reps, precomputed_folds)
        fold_reps: {fold_idx: {rep_name: (train_embs, val_embs)}}
        precomputed_folds: [(train_idx, val_idx), ...] — materialized splits
    """
    if not rep_factories:
        return {}, []

    if verbose:
        print(f"\nPre-computing fine-tuned representations "
              f"({len(rep_factories)} factories × {n_folds} folds)...")

    # Materialize fold splits (cached when dataset_name is supplied)
    splitter, use_groups = _make_cv_splitter(groups, n_folds, seed,
                                             cv_splitter=cv_splitter, task_type=task_type)
    precomputed_folds = _load_or_compute_folds(
        dataset_name=dataset_name, splitter=splitter, use_groups=use_groups,
        labels=labels, groups=groups, n_folds=n_folds, seed=seed,
        cv_splitter_id=cv_splitter_id, verbose=verbose, task_type=task_type)

    _labels_hash = _hash_bytes(_canonical_bytes(labels))
    _groups_hash = (_hash_bytes(_canonical_bytes(groups))
                    if groups is not None else None)

    fold_reps = {}
    n_cache_hits = 0
    n_cache_misses = 0
    for fold_idx, (train_idx, val_idx) in enumerate(precomputed_folds):
        fold_reps[fold_idx] = {}
        train_labels = labels[train_idx]

        for rep_name, factory in rep_factories.items():
            # Attempt cache hit first. Gated on (dataset_name, factory_id).
            fid = (factory_ids or {}).get(rep_name) if factory_ids else None
            ft_cache_path = None
            if dataset_name and fid:
                ft_cache_path = _ft_embeddings_cache_path(
                    dataset_name, fid, rep_name, fold_idx, n_folds, seed)
                ft_meta = _make_ft_embedding_meta(
                    dataset_name, fid, rep_name, fold_idx, n_folds, seed,
                    train_idx, val_idx, _labels_hash, _groups_hash)
                cached = _load_cache_with_meta(
                    ft_cache_path, ft_meta, kind='search', verbose=False)
                if cached is not None:
                    fold_reps[fold_idx][rep_name] = cached
                    n_cache_hits += 1
                    if verbose:
                        shape_msg = (f"{cached[0].shape[1]}D"
                                     if isinstance(cached, tuple) else '?')
                        print(f"  Fold {fold_idx+1}/{n_folds}, {rep_name}: "
                              f"loaded from FT cache ({shape_msg}).")
                    continue

            start = time.time()
            if verbose:
                print(f"  Fold {fold_idx+1}/{n_folds}, {rep_name}...", end=' ', flush=True)

            try:
                train_embs, val_embs = factory(train_idx, val_idx, train_labels)
                fold_reps[fold_idx][rep_name] = (train_embs, val_embs)
                elapsed = time.time() - start
                if verbose:
                    print(f"done ({train_embs.shape[1]}D, {elapsed:.1f}s)")
                # Persist to cache when we have an id for it.
                if ft_cache_path is not None:
                    _save_cache_with_meta(
                        ft_cache_path, ft_meta, (train_embs, val_embs))
                    n_cache_misses += 1
            except Exception as e:
                if verbose:
                    print(f"FAILED: {e}")

    if verbose and (n_cache_hits or n_cache_misses):
        print(f"  FT embeddings: {n_cache_hits} cache hits, "
              f"{n_cache_misses} computed + cached")

    return fold_reps, precomputed_folds


def _extract_full_embeddings(
    rep_factories: Dict[str, Callable],
    labels: np.ndarray,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Fine-tune each factory on ALL data and return full-dataset embeddings.

    Each factory is called with train_idx=val_idx=all_idx so it trains on the
    full dataset and extracts embeddings for every sample.

    ★ NEVER feed these back in as base_reps and then cross-validate.
    These embeddings were fine-tuned on EVERY row's labels, including whatever
    you later hold out. Scoring them under CV — or under nested CV — reports a
    number the model already saw the answers for. It is a total leak, and no
    outer fold loop can undo it, because the leak is baked into the features.

    (An earlier version of this docstring recommended exactly that use. It was
    wrong. If you need fine-tuned reps inside an evaluation, pass `rep_factories`
    to find_best_hybrid so they are re-fit per fold on training rows only —
    see _precompute_fold_reps.)

    This exists ONLY to build the full-data product artifact (FittedResult /
    SearchResult.finetuned_embeddings) that ships a trained model to a user.
    """
    if not rep_factories:
        return {}

    if verbose:
        print(f"\nExtracting full-dataset embeddings "
              f"({len(rep_factories)} factories)...")

    all_idx = np.arange(len(labels))
    full_embeddings = {}

    for rep_name, factory in rep_factories.items():
        start = time.time()
        if verbose:
            print(f"  {rep_name}...", end=" ", flush=True)
        try:
            # Train on all data, extract for all data
            train_embs, _ = factory(all_idx, all_idx, labels)
            full_embeddings[rep_name] = train_embs
            elapsed = time.time() - start
            if verbose:
                print(f"done ({train_embs.shape[1]}D, {elapsed:.1f}s)")
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")

    return full_embeddings


#: High-dim rep filter — applied only in _evaluate_single standalone baseline scoring.
#: Reps with more than HIGH_DIM_THRESHOLD features get cut down per-fold via
#: LightGBM feature importance fit on the train fold only (no label leakage
#: into val). Does NOT affect hybrid search feature selection — hybrid still
#: sees the full-dim rep. Only the allocation-input baseline score is filtered.
#: Version is baked into the baseline cache key so old unfiltered scores are
#: automatically invalidated.
HIGH_DIM_FILTER_THRESHOLD = 5000
HIGH_DIM_FILTER_VERSION = 1


def _select_top_features_per_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_n: int,
    task_type: str,
) -> np.ndarray:
    """Fit LightGBM on train fold, return indices of top-N features.

    Per-fold: no val-fold labels leak into feature selection. Falls back
    to sklearn RandomForest if lightgbm is unavailable.
    """
    X_clean = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    try:
        if task_type == 'classification':
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(n_estimators=50, max_depth=5,
                                    random_state=42, n_jobs=-1, verbose=-1)
        else:
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(n_estimators=50, max_depth=5,
                                   random_state=42, n_jobs=-1, verbose=-1)
    except ImportError:
        if task_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier as _M
        else:
            from sklearn.ensemble import RandomForestRegressor as _M
        model = _M(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_clean, y_train)
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-target_n:]
    top_idx.sort()
    return top_idx


def _evaluate_single(
    X: np.ndarray,
    y: np.ndarray,
    model,
    groups: Optional[np.ndarray],
    n_folds: int,
    metric_fn: Callable,
    seed: int = 42,
    cv_splitter=None,
    secondary_metric_fns: Optional[Dict[str, Callable]] = None,
    task_type: Optional[str] = None,
    return_fold_scores: bool = False,
    train_labels: Optional[np.ndarray] = None,
) -> Union[Tuple[float, float], Tuple[float, float, Dict[str, Tuple[float, float]]]]:
    """Evaluate one (X, model) with CV.

    train_labels: optional noisy full-length label column. When given, the model
    TRAINS on train_labels[train_idx] but is always SCORED on the clean y[val_idx].
    None (the default) means train_labels is y — byte-identical to the old behaviour.

    Returns (mean_score, std_score) if no secondary metrics.
    Returns (mean_score, std_score, {name: (mean, std)}) if secondary metrics provided.

    When return_fold_scores=True, the raw per-fold primary-metric score list is
    appended as the FINAL tuple element (i.e. (mean, std, folds) with no
    secondary metrics, or (mean, std, sec_results, folds) with them). Additive:
    the sole caller (_evaluate_baselines) opts in; the default keeps the
    historical 2-/3-tuple return for any other reader.

    When X has more than HIGH_DIM_FILTER_THRESHOLD features, a per-fold
    LightGBM feature importance filter is applied (train fold only → top N,
    applied to both train and val for that fold). Target N is 500 if the
    dataset has >= 1000 samples else 200. This is a standalone-eval-only
    safeguard — hybrid search itself is unaffected.
    """
    splitter, use_groups = _make_cv_splitter(groups, n_folds, seed,
                                             cv_splitter=cv_splitter, task_type=task_type)
    split_args = (X, y, groups) if use_groups else (X, y)
    scores = []
    sec_scores = {name: [] for name in (secondary_metric_fns or {})}

    # Decide filter params up-front (const per rep, reused across folds)
    _apply_hd_filter = X.shape[1] > HIGH_DIM_FILTER_THRESHOLD
    if _apply_hd_filter:
        _hd_target_n = 500 if len(y) >= 1000 else 200
        if _hd_target_n >= X.shape[1]:
            _apply_hd_filter = False
        else:
            # Prefer explicit task_type (from find_best_hybrid / caller) over
            # the label-count heuristic. Heuristic remains as a fallback for
            # the one legacy caller (tune_models) that still auto-detects.
            _hd_task_type = (task_type
                             if task_type in ('classification', 'regression')
                             else _detect_task_type(y))

    # Only scale continuous features; leave binary (ECFP, MACCS, etc.) raw
    is_binary = (X.dtype == np.uint8 or
                 (len(np.unique(X)) <= 2 and
                  set(np.unique(X).tolist()).issubset({0, 0.0, 1, 1.0})))

    # Label-noise contract: TRAIN labels may be noisy, VALIDATION labels never are.
    # `train_labels` is the noisy full column (or None / `y` itself when clean), so
    # y_train comes from it while y_val always comes from the clean `y`.
    _y_tr_src = y if train_labels is None else np.asarray(train_labels).ravel()

    for train_idx, val_idx in splitter.split(*split_args):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = _y_tr_src[train_idx], y[val_idx]

        # Per-fold high-dim filter — fit LGBM on train only, pick top N
        # features, apply to this fold's train and val.
        if _apply_hd_filter:
            top_idx = _select_top_features_per_fold(
                X_train, y_train, _hd_target_n, _hd_task_type)
            X_train = X_train[:, top_idx]
            X_val = X_val[:, top_idx]

        if is_binary:
            X_train_s = X_train.astype(np.float32)
            X_val_s = X_val.astype(np.float32)
            X_train_s, X_val_s, _ = _impute_nan(X_train_s, X_val_s)
        else:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            X_train_s, X_val_s, _ = _impute_nan(X_train_s, X_val_s)

        m = deepcopy(model)
        # Adaptive k for KNN: scale to sqrt(N_train), capped at 5
        if hasattr(m, 'n_neighbors') and hasattr(m, 'set_params'):
            k = min(5, max(1, int(np.sqrt(len(train_idx)))))
            m.set_params(n_neighbors=k)
        _fg = groups[train_idx] if groups is not None else None
        _fit_model(m, X_train_s, y_train, groups=_fg)
        y_pred = _predict_for_metric(m, X_val_s, metric_fn)
        scores.append(metric_fn(y_val, y_pred))

        if secondary_metric_fns:
            y_pred_labels = m.predict(X_val_s)
            for sec_name, sec_fn in secondary_metric_fns.items():
                try:
                    sec_scores[sec_name].append(sec_fn(y_val, y_pred_labels))
                except Exception:
                    pass

    mean, std = float(np.mean(scores)), float(np.std(scores))
    _folds_out = [float(s) for s in scores]
    if secondary_metric_fns is None:
        return (mean, std, _folds_out) if return_fold_scores else (mean, std)

    sec_results = {}
    for sec_name, vals in sec_scores.items():
        if vals:
            sec_results[sec_name] = (float(np.mean(vals)), float(np.std(vals)))
    if return_fold_scores:
        return mean, std, sec_results, _folds_out
    return mean, std, sec_results


def _hash_feature_selection(feature_info: dict) -> str:
    """Hash selected indices from feature_info for dedup.

    Two configs that select the exact same features from each rep
    produce identical hybrid matrices, so downstream model evaluation
    can be reused.
    """
    parts = []
    for rep_name in sorted(feature_info.keys()):
        info = feature_info[rep_name]
        if isinstance(info, dict) and 'selected_indices' in info:
            idx = tuple(sorted(info['selected_indices']))
            parts.append(f"{rep_name}:{idx}")
    # Include augmentation info if present
    aug_info = feature_info.get('augmentation_info')
    if aug_info and isinstance(aug_info, dict):
        for aname in sorted(aug_info.keys()):
            ainfo = aug_info[aname]
            if isinstance(ainfo, dict) and 'selected_indices' in ainfo:
                parts.append(f"aug_{aname}:{tuple(sorted(ainfo['selected_indices']))}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _evaluate_hybrid_config(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    allocation_method: str,
    budget: int,
    importance_method: str,
    model,
    model_name: str,
    internal_evaluator: str = 'match',
    groups: Optional[np.ndarray] = None,
    n_folds: int = 5,
    metric_fn: Callable = None,
    metric_name: Optional[str] = None,
    seed: int = 42,
    step_size: int = 5,
    augmentations: Optional[Dict[str, np.ndarray]] = None,
    augmentation_strategy: str = 'uncapped',
    augmentation_threshold: Optional[float] = None,
    augmentation_budget: int = 20,
    aug_per_source_budget: Optional[int] = None,
    candidate_ranking: str = 'rf_importance',
    lower_is_better: bool = False,
    fold_reps: Optional[Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = None,
    precomputed_folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    cv_splitter=None,
    importance_cache: Optional[Dict] = None,
    alloc_cache: Optional[Dict] = None,
    selection_dedup: Optional[Dict] = None,
    greedy_tolerance: float = 0.0,
    objective: str = OBJECTIVE_PREDICTIVE,
    fold_indices: Optional[List[int]] = None,
    # Persistent hybrid_eval dedup cache. Active only for top_k
    # allocations where the selected rep subset is predictable from
    # per-fold alloc scores. `dataset_name` and `task_type` are required
    # for the cache to activate; if either is missing the cache silently
    # no-ops and behaviour is identical to pre-D.2.
    dataset_name: Optional[str] = None,
    task_type: Optional[str] = None,
    baselines: Optional[Dict[str, Dict[str, float]]] = None,
    use_hybrid_eval_cache: bool = True,
    train_labels: Optional[np.ndarray] = None,
    noise_id: Optional[str] = None,
) -> dict:
    """Evaluate a single hybrid configuration. Returns result dict.

    train_labels / noise_id: label-noise support. The hybrid is constructed and the
    model fitted on train_labels[train_idx] (noisy when a NoiseSpec is active), while
    scoring always uses the clean labels[val_idx]. `noise_id` namespaces the
    hybrid_eval dedup cache, whose path carries no labels hash and would otherwise
    return clean-label results for a noisy run.

    Args:
        fold_indices: If set, only evaluate these fold indices (0-based).
            None (default) evaluates all folds.
    """
    from kirby.hybrid import create_hybrid, apply_feature_selection, apply_augmentation_selection

    start_time = time.time()

    # Use precomputed folds if available (ensures identical splits with fine-tuned reps)
    if precomputed_folds is not None:
        fold_splits = precomputed_folds
    else:
        # task_type MUST be threaded: for classification it selects
        # StratifiedGroupKFold. The baseline arm (_evaluate_single) already
        # passes it, so omitting it here gave the two arms DIFFERENT fold
        # partitions on every classification dataset — making hybrid and
        # baseline fold_scores non-comparable fold-for-fold.
        splitter, use_groups = _make_cv_splitter(groups, n_folds, seed,
                                                 cv_splitter=cv_splitter,
                                                 task_type=task_type)
        sample_X = next(iter(base_reps.values()))
        split_args = (sample_X, labels, groups) if use_groups else (sample_X, labels)
        fold_splits = list(splitter.split(*split_args))

    fold_scores = []
    fold_n_features = []
    fold_metric_dicts = []   # full classic metric set per fold
    any_imputation = False

    _eval_fold_indices = range(len(fold_splits)) if fold_indices is None else fold_indices

    # Precompute shared cache-key components for this call.
    # Only built when the hybrid_eval cache is actually active — every
    # other call path short-circuits _hybrid_cache_enabled below.
    _sim_mode = (objective == OBJECTIVE_SIMILARITY_SAC)
    _is_clf_task = (task_type == 'classification')
    _hybrid_cache_enabled = (
        use_hybrid_eval_cache
        and not _sim_mode  # sim scores are geometry-based, not model-based;
                           # the hybrid_eval cache keys on metric/model only.
        and dataset_name is not None
        and task_type in ('classification', 'regression')
        and metric_name is not None
        and allocation_method.startswith('top_')
    )
    if _hybrid_cache_enabled:
        # Hash the TRAIN-side label column. Under train-only noise the clean `labels`
        # array is identical between a clean and a noisy run, so hashing it would make
        # the meta blind to the very thing that changed.
        _labels_hash = _hash_bytes(_canonical_bytes(
            labels if train_labels is None else train_labels))
        _aug_names_t = (tuple(sorted(augmentations.keys()))
                        if augmentations else ())
        _aug_hashes_t = (_hash_rep_dict(augmentations)
                         if augmentations else ())

    for fold_idx in _eval_fold_indices:
        train_idx, val_idx = fold_splits[fold_idx]
        train_reps = {k: v[train_idx] for k, v in base_reps.items()}
        val_reps = {k: v[val_idx] for k, v in base_reps.items()}
        # Label-noise contract: the hybrid is BUILT and the model TRAINED on
        # (possibly noisy) train labels; scoring at the bottom of this loop always
        # uses the clean val labels.
        y_train = (labels[train_idx] if train_labels is None
                   else np.asarray(train_labels).ravel()[train_idx])
        y_val = labels[val_idx]

        # Merge fine-tuned reps into train/val reps for this fold
        if fold_reps and fold_idx in fold_reps:
            for rep_name, (ft_train, ft_val) in fold_reps[fold_idx].items():
                train_reps[rep_name] = ft_train
                val_reps[rep_name] = ft_val

        # Split augmentations per fold
        train_augs = None
        val_augs = None
        if augmentations is not None:
            train_augs = {name: X[train_idx] for name, X in augmentations.items()}
            val_augs = {name: X[val_idx] for name, X in augmentations.items()}

        # Look up precomputed importance for this (fold, importance_method)
        fold_importance = None
        if importance_cache is not None:
            fold_importance = importance_cache.get((fold_idx, importance_method))

        # Look up precomputed allocation scores for this (fold, evaluator).
        # Prefer metric-specific scores when available.
        # In similarity mode the precomputed alloc scores are model-prediction
        # based (and the 'sac' entry is a no-op 0.0 from the metric guard), so
        # they must NOT be reused — force the allocators to compute live SAC
        # per rep instead. This is what makes performance_weighted/top_* pick a
        # DIFFERENT rep composition under the similarity objective.
        fold_alloc_scores = None
        if alloc_cache is not None and not _sim_mode:
            if metric_name:
                fold_alloc_scores = alloc_cache.get((fold_idx, internal_evaluator, metric_name))
            if fold_alloc_scores is None:
                fold_alloc_scores = alloc_cache.get((fold_idx, internal_evaluator))

        # Persistent hybrid_eval cache — probe before compute.
        # Dedup applies when `alloc` is top_k and the predicted subset
        # equals another run's subset (e.g. general top_3 and top_tier
        # top_3 both pick the same 3 reps). Miss cases pay only the
        # cost of predicting + hashing + one file stat.
        _he_cache_path = None
        _he_expected_meta = None
        if _hybrid_cache_enabled:
            _predicted_subset = _predict_top_k_subset(
                allocation_method, sorted(base_reps.keys()),
                fold_alloc_scores=fold_alloc_scores, baselines=baselines,
                evaluator_for_baseline=internal_evaluator,
                lower_is_better=lower_is_better,
            )
            if _predicted_subset is not None:
                # Identify the subset by NAME, not by neural-rep VALUE hash
                # (embeddings aren't bit-reproducible -> filename churned every
                # run -> cache never hit). subset_hashes is kept for the meta
                # record only and is now in ignore_keys, so value drift no longer
                # rejects the cache either.
                _subset_hex = _subset_name_hex(_predicted_subset)
                _subset_hashes = tuple((name, 0) for name in sorted(_predicted_subset))
                _he_cache_path = _hybrid_eval_cache_path(
                    dataset_name=dataset_name,
                    subset_hash=_subset_hex,
                    alloc=allocation_method, imp=importance_method,
                    evaluator=internal_evaluator, budget=budget,
                    fold_idx=fold_idx, n_folds=len(fold_splits),
                    seed=seed, model_name=model_name,
                    aug_strategy=augmentation_strategy,
                    aug_names=_aug_names_t,
                    aug_per_source_budget=aug_per_source_budget,
                    noise_id=noise_id,
                )
                _he_expected_meta = _make_hybrid_eval_meta(
                    dataset_name=dataset_name,
                    subset_names=_predicted_subset,
                    subset_hashes=_subset_hashes,
                    alloc=allocation_method, imp=importance_method,
                    evaluator=internal_evaluator, budget=budget,
                    fold_idx=fold_idx, n_folds=len(fold_splits), seed=seed,
                    model_name=model_name,
                    train_idx=train_idx, val_idx=val_idx,
                    labels_hash=_labels_hash,
                    aug_strategy=augmentation_strategy,
                    aug_names=_aug_names_t, aug_hashes=_aug_hashes_t,
                    step_size=step_size,
                    candidate_ranking=candidate_ranking,
                    greedy_tolerance=greedy_tolerance,
                    augmentation_threshold=augmentation_threshold,
                    augmentation_budget=augmentation_budget,
                    aug_per_source_budget=aug_per_source_budget,
                    metric_name=metric_name, task_type=task_type,
                )
                _cached_eval = _load_cache_with_meta(
                    _he_cache_path, _he_expected_meta,
                    kind='hybrid_eval', verbose=False)
                # A disk entry WITHOUT a stored metric set (an older file) is
                # treated as a MISS so the fold recomputes the full set — never
                # mix metric-bearing and metric-less folds in one config.
                if _cached_eval is not None and _cached_eval.get('metrics') is not None:
                    fold_scores.append(_cached_eval['score'])
                    fold_n_features.append(_cached_eval['n_features'])
                    fold_metric_dicts.append(_cached_eval['metrics'])
                    if _cached_eval.get('had_imputation'):
                        any_imputation = True
                    # Also feed the in-memory selection_dedup so sibling
                    # configs within this call benefit.
                    if (selection_dedup is not None
                            and _cached_eval.get('selection_hash')):
                        selection_dedup[(fold_idx, model_name,
                                         _cached_eval['selection_hash'])] = (
                            _cached_eval['score'], _cached_eval['n_features'],
                            _cached_eval['metrics'])
                    continue

        try:
            # Pass fold-level groups for group-aware internal split
            fold_groups = groups[train_idx] if groups is not None else None
            X_train, feature_info = create_hybrid(
                train_reps, y_train,
                allocation_method=allocation_method,
                budget=budget,
                step_size=step_size,
                importance_method=importance_method,
                internal_evaluator=internal_evaluator,
                augmentations=train_augs,
                augmentation_strategy=augmentation_strategy,
                augmentation_threshold=augmentation_threshold,
                augmentation_budget=augmentation_budget,
                aug_per_source_budget=aug_per_source_budget,
                candidate_ranking=candidate_ranking,
                model=model,
                metric_fn=metric_fn,
                lower_is_better=lower_is_better,
                precomputed_importance=fold_importance,
                precomputed_alloc_scores=fold_alloc_scores,
                greedy_tolerance=greedy_tolerance,
                groups=fold_groups,
                objective=objective,
                is_clf=_is_clf_task,
            )
            X_val = apply_feature_selection(val_reps, feature_info)

            # Apply augmentation selection to val set.
            # Auto-demote (top_k) may have added demoted base reps to the
            # augmentation pool inside create_hybrid. train_augs was mutated
            # to include full-data slices (n_samples, n_top_features) for
            # demoted reps, so val indexing works directly.
            aug_info = feature_info.get('augmentation_info')
            if aug_info and train_augs is not None:
                # Rebuild val_augs to pick up any demoted reps added by auto-demote.
                # train_augs[name] for demoted reps is (n_full, top_k_features)
                # from create_hybrid, so val_idx slicing is safe.
                val_augs = {}
                for name in train_augs:
                    arr = train_augs[name]
                    if arr.shape[0] == len(labels):
                        # Full-data array (from auto-demote mutation)
                        val_augs[name] = arr[val_idx]
                    elif augmentations is not None and name in augmentations:
                        val_augs[name] = augmentations[name][val_idx]
                    else:
                        val_augs[name] = base_reps[name][val_idx]
                aug_val_features = apply_augmentation_selection(val_augs, aug_info)
                if aug_val_features is not None:
                    X_val = np.hstack([X_val, aug_val_features])
        except Exception as _fold_err:
            import traceback
            traceback.print_exc()
            continue

        # Dedup: if identical feature selection was already evaluated with
        # the same downstream model on this fold, reuse the score.
        sel_hash = None
        if selection_dedup is not None:
            sel_hash = _hash_feature_selection(feature_info)
            dedup_key = (fold_idx, model_name, sel_hash)
            cached = selection_dedup.get(dedup_key)
            if cached is not None:
                fold_scores.append(cached[0])
                fold_n_features.append(cached[1])
                # cached is (score, n_features, metric_dict) post-D6.
                fold_metric_dicts.append(cached[2] if len(cached) > 2 else {})
                selection_dedup['_hits'] = selection_dedup.get('_hits', 0) + 1
                continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_train_s, X_val_s, _fold_imputed = _impute_nan(X_train_s, X_val_s)
        if _fold_imputed:
            any_imputation = True

        if _sim_mode:
            # Similarity objective: score the held-out fold GEOMETRY directly
            # (SAC = higher-is-better). No downstream model is fit, so
            # _predict_for_metric / _compute_classic_metrics are skipped —
            # there is no y_pred. Selecting features on train (inside
            # create_hybrid) and scoring SAC on val keeps this honest (not the
            # in-sample, definitional SAC of the older s02 path).
            score = _similarity_objective_score(X_val_s, y_val, _is_clf_task, seed=seed)
            fold_scores.append(score)
            fold_n_features.append(X_train.shape[1])
            _fold_metrics = {'sac': float(score)} if np.isfinite(score) else {}
            fold_metric_dicts.append(_fold_metrics)
            if selection_dedup is not None and sel_hash is not None:
                selection_dedup[(fold_idx, model_name, sel_hash)] = (
                    score, X_train.shape[1], _fold_metrics)
            continue

        m = deepcopy(model)
        # Adaptive k for KNN: scale to sqrt(N_train), capped at 5
        if hasattr(m, 'n_neighbors') and hasattr(m, 'set_params'):
            k = min(5, max(1, int(np.sqrt(len(train_idx)))))
            m.set_params(n_neighbors=k)
        _fg = groups[train_idx] if groups is not None else None
        _fit_model(m, X_train_s, y_train, groups=_fg)
        y_pred = _predict_for_metric(m, X_val_s, metric_fn)
        score = metric_fn(y_val, y_pred)
        fold_scores.append(score)
        fold_n_features.append(X_train.shape[1])
        # Full classic metric set for this fold, plus the designated
        # ranking metric itself (so the row is self-describing even when the
        # ranking metric is non-classic — spearman/ci/mcc).
        _fold_metrics = _compute_classic_metrics(m, X_val_s, y_val, y_pred, task_type)
        if metric_name and np.isfinite(score):
            _fold_metrics.setdefault(metric_name, float(score))
        fold_metric_dicts.append(_fold_metrics)

        # Store in dedup cache (in-memory, per-call)
        if selection_dedup is not None and sel_hash is not None:
            selection_dedup[(fold_idx, model_name, sel_hash)] = (
                score, X_train.shape[1], _fold_metrics)

        # Persist to disk for cross-invocation dedup. Skipped
        # if the predicted subset disagreed with the actual picked
        # subset (rare — only happens if baselines/alloc_scores lied
        # about top-k ordering).
        if _he_cache_path is not None and _he_expected_meta is not None:
            # Cross-check: the reps the allocator actually picked match
            # what we predicted. feature_info has per-rep sub-dicts
            # (with 'selected_indices') plus bookkeeping entries like
            # 'allocation' and 'augmentation_info' that we must exclude.
            _nonrep_keys = {'allocation', 'augmentation_info'}
            _actual_picked = sorted(
                name for name, info in feature_info.items()
                if name not in _nonrep_keys
                and isinstance(info, dict)
                and 'selected_indices' in info
                and len(info['selected_indices']) > 0
            )
            _predicted_sorted = sorted(_predicted_subset)
            if _actual_picked == _predicted_sorted:
                _payload = {
                    'score': float(score),
                    'n_features': int(X_train.shape[1]),
                    'selection_hash': sel_hash,
                    'had_imputation': bool(_fold_imputed),
                    'metrics': _fold_metrics,   # full classic set for this fold
                }
                _save_cache_with_meta(
                    _he_cache_path, _he_expected_meta, _payload)

    if not fold_scores:
        raise ValueError("All folds failed")

    elapsed = time.time() - start_time
    # Include both frozen and fine-tuned rep names
    rep_names = list(base_reps.keys())
    if fold_reps:
        # Add factory rep names from first fold
        first_fold = next(iter(fold_reps.values()), {})
        for name in first_fold:
            if name not in rep_names:
                rep_names.append(name)

    blueprint = Blueprint(
        reps=rep_names,
        allocation_method=allocation_method,
        importance_method=importance_method,
        internal_evaluator=internal_evaluator,
        budget=budget,
        step_size=step_size,
        augmentation_strategy=augmentation_strategy,
        aug_per_source_budget=aug_per_source_budget,
        metric=metric_name or 'r2',
        lower_is_better=lower_is_better,
        objective=objective,
    )

    return {
        'blueprint': blueprint,
        'model': model_name,
        'score_mean': float(np.mean(fold_scores)),
        'score_std': float(np.std(fold_scores)),
        'fold_scores': [float(s) for s in fold_scores],
        # The full classic metric set (regression r2/mae/mse/rmse OR
        # classification accuracy/balanced_accuracy/precision/recall/f1/roc_auc),
        # plus the designated ranking metric, each {'mean','std'} over folds.
        # Ranking still uses score_mean; this is stored alongside.
        'metrics': _aggregate_fold_metrics(fold_metric_dicts),
        'n_features': int(np.mean(fold_n_features)),
        'time_seconds': elapsed,
        'allocation_method': allocation_method,
        'importance_method': importance_method,
        'internal_evaluator': internal_evaluator,
        'budget': budget,
        'step_size': step_size,
        'had_imputation': any_imputation,
    }


def _evaluate_baselines(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    models: Dict[str, Any],
    groups: Optional[np.ndarray],
    n_folds: int,
    metric_fn: Callable,
    seed: int = 42,
    verbose: bool = True,
    fold_reps: Optional[Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = None,
    precomputed_folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    secondary_metric_fns: Optional[Dict[str, Callable]] = None,
    cv_splitter=None,
    dataset_name: Optional[str] = None,
    metric_name: Optional[str] = None,
    use_baseline_cache: bool = True,
    task_type: Optional[str] = None,
    objective: str = OBJECTIVE_PREDICTIVE,
    train_labels: Optional[np.ndarray] = None,
    noise_id: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, list]]]:
    """Evaluate full-dimension baselines for each rep x model.

    train_labels / noise_id: label-noise support. Baselines MUST be trained on the
    same noisy labels as the hybrid candidates — otherwise a noisy hybrid search
    would be compared against CLEAN-label baselines, which does not remove the
    clean-label advantage, it hands it to the base representations instead.
    `noise_id` namespaces the per-rep baseline cache; without it a noisy run would
    HIT the clean baseline cache (its path carries no labels hash) and silently
    return clean scores.

    Returns (baselines, baseline_fold_scores):
      - baselines: the historical {rep: {model: mean_score, 'dimension': d,
        ['_secondary': ...]}} dict (unchanged shape/types).
      - baseline_fold_scores: additive parallel {rep: {model: [f0, f1, ...]}}
        holding the raw per-fold primary-metric scores, so a leave-one-fold-out
        honest re-score can tax baseline reps the same way as hybrid candidates.
        Only carries entries for the models actually evaluated. Entries served
        from an OLD per-rep baseline cache (written before per-fold caching) are
        simply absent — readers must tolerate a missing (rep, model).

    Factory reps (from fold_reps) are evaluated per-fold since they have
    different train/val embeddings per fold.

    If secondary_metric_fns is provided, stores secondary metrics under
    '_secondary' key per rep: {f'{model}_{metric}': {'mean': ..., 'std': ...}}.

    Per-rep baseline caching:
        When dataset_name and metric_name are provided, individual (rep, model)
        baseline scores are cached to ~/.cache/kirby/baselines/ and reused across
        find_best_hybrid calls that share the same dataset. This is the main
        "dedup across conditions" win — 6 precompute conditions per dataset all
        hit the same baseline cache.

        Secondary metrics are cached too: their names go into the cache key
        (filename + meta) and their per-model values ride in the payload under
        'secondary'. A run requesting secondary metrics will not accept an entry
        written without them.

        Cache invalidates when rep content, labels, groups, n_folds, or seed
        changes (content-validated on load).
    """
    if verbose:
        print("\nEvaluating full-dimension baselines...")

    baselines = {}
    # Additive parallel structure: per-(rep, model) raw per-fold scores.
    # Kept OUT of the `baselines` dict (which many downstream readers iterate)
    # so their value-type assumptions are untouched.
    baseline_fold_scores: Dict[str, Dict[str, list]] = {}

    def _record_folds(_rep, _model, _folds):
        if _folds is not None and len(_folds) > 0:
            baseline_fold_scores.setdefault(_rep, {})[_model] = [
                float(s) for s in _folds]

    # Only enable baseline caching when it's safe:
    #   - dataset_name + metric_name both known
    # Secondary metrics used to disable this cache entirely, which meant every
    # CLI wrapper (all four pass secondary_metrics) recomputed the full CV
    # baseline for every rep on every run. They are now part of the cache
    # identity (filename + meta) and their values ride along in the payload.
    _sec_names = (tuple(sorted(secondary_metric_fns))
                  if secondary_metric_fns else None)
    _baseline_cache_enabled = (
        use_baseline_cache
        and dataset_name is not None
        and metric_name is not None
    )

    # Frozen reps: standard global-array evaluation
    for rep_name, X in base_reps.items():
        baselines[rep_name] = {'dimension': X.shape[1]}
        sec_all = {}
        # Hash rep once per rep (reused for every model)
        _rep_hash = _hash_array(X) if _baseline_cache_enabled else None

        for model_name, model in models.items():
            # --- Per-(rep, model) baseline cache lookup ---
            _cached_score = None
            _cached_sec = {}
            _cached_fold = None
            _cache_path = None
            _baseline_meta = None
            if _baseline_cache_enabled:
                _cache_path = _baseline_cache_path(
                    dataset_name=dataset_name,
                    rep_name=rep_name,
                    model_name=model_name,
                    metric_name=metric_name,
                    n_folds=n_folds,
                    seed=seed,
                    secondary_names=_sec_names,
                    noise_id=noise_id,
                )
                _baseline_meta = _make_baseline_meta(
                    dataset_name=dataset_name,
                    rep_name=rep_name,
                    rep_shape=X.shape,
                    model_name=model_name,
                    metric_name=metric_name,
                    n_folds=n_folds,
                    seed=seed,
                    # Hash the TRAIN-side label column: under noise this is the
                    # noisy one, so the meta actually distinguishes the conditions
                    # (the clean `labels` array would be identical in both).
                    labels=(labels if train_labels is None else train_labels),
                    groups=groups,
                    secondary_names=_sec_names,
                )
                if noise_id:
                    _baseline_meta['noise_id'] = noise_id
                _baseline_meta['rep_hash'] = _rep_hash
                _cached = _load_cache_with_meta(
                    _cache_path, _baseline_meta,
                    kind='baseline', verbose=verbose)
                if _cached is not None and isinstance(_cached, dict):
                    # A secondary-metric run must not be served an entry that
                    # predates secondary caching — it would silently drop the
                    # secondary block. Treat as a miss and rewrite.
                    if _sec_names is not None and 'secondary' not in _cached:
                        _cached_score = None
                    elif not _cached.get('fold_scores'):
                        # Entries written before per-fold baseline caching carry
                        # only the mean. Serving them would leave the held-out
                        # baseline arm with no folds to select on, so force a
                        # miss and rewrite rather than silently degrade.
                        _cached_score = None
                    else:
                        _cached_score = _cached.get('score')
                        _cached_sec = _cached.get('secondary') or {}
                        _cached_fold = _cached.get('fold_scores')

            if _cached_score is not None:
                baselines[rep_name][model_name] = _cached_score
                for _sec_name, _entry in _cached_sec.items():
                    sec_all[f'{model_name}_{_sec_name}'] = _entry
                _record_folds(rep_name, model_name, _cached_fold)
                continue

            # --- Similarity objective: per-rep SAC baseline (no model fit) ---
            # SAC is model-independent — score the rep's own held-out geometry
            # per fold and average. This is also the "best single rep by SAC"
            # comparator. Avoids invoking the 'sac' metric guard.
            if objective == OBJECTIVE_SIMILARITY_SAC:
                _is_clf_b = (task_type == 'classification')
                _spl, _use_g = _make_cv_splitter(groups, n_folds, seed,
                                                 cv_splitter=cv_splitter,
                                                 task_type=task_type)
                _sargs = (X, labels, groups) if _use_g else (X, labels)
                _sac_folds = []
                for _tr, _va in _spl.split(*_sargs):
                    _sc = StandardScaler().fit(X[_tr])
                    _sac_folds.append(_similarity_objective_score(
                        _sc.transform(X[_va]), labels[_va], _is_clf_b, seed=seed))
                rounded_score = round(float(np.mean(_sac_folds)) if _sac_folds else 0.0, 4)
                baselines[rep_name][model_name] = rounded_score
                _record_folds(rep_name, model_name, _sac_folds)
                if _baseline_cache_enabled and _cache_path is not None:
                    _baseline_meta['code_timestamp'] = time.time()
                    # SAC scores never populate secondary metrics; store an
                    # empty block so a secondary-metric run reads a hit rather
                    # than looping on the 'secondary' missing-key miss above.
                    _save_cache_with_meta(
                        _cache_path, _baseline_meta,
                        {'score': rounded_score, 'dimension': X.shape[1],
                         'secondary': {},
                         'fold_scores': [float(s) for s in _sac_folds]})
                continue

            # --- Compute (cache miss) ---
            result = _evaluate_single(
                X, labels, model, groups, n_folds, metric_fn, seed,
                cv_splitter=cv_splitter,
                secondary_metric_fns=secondary_metric_fns,
                task_type=task_type,
                return_fold_scores=True,
                train_labels=train_labels,
            )
            _sec_for_model = {}
            if secondary_metric_fns is not None:
                mean_score, _, sec_dict, _fold_scores = result
                for sec_name, (sec_mean, sec_std) in sec_dict.items():
                    _entry = {'mean': round(sec_mean, 4),
                              'std': round(sec_std, 4)}
                    sec_all[f'{model_name}_{sec_name}'] = _entry
                    _sec_for_model[sec_name] = _entry
            else:
                mean_score, _, _fold_scores = result
            rounded_score = round(mean_score, 4)
            baselines[rep_name][model_name] = rounded_score
            _record_folds(rep_name, model_name, _fold_scores)

            # --- Write to per-rep baseline cache (on miss only) ---
            if _baseline_cache_enabled and _cache_path is not None:
                _baseline_meta['code_timestamp'] = time.time()
                _save_cache_with_meta(
                    _cache_path, _baseline_meta,
                    {'score': rounded_score, 'dimension': X.shape[1],
                     'secondary': _sec_for_model,
                     'fold_scores': [float(s) for s in _fold_scores]},
                )

        if sec_all:
            baselines[rep_name]['_secondary'] = sec_all

        if verbose:
            scores_str = ", ".join(
                f"{m}={baselines[rep_name][m]:.3f}" for m in models
            )
            print(f"  {rep_name} (dim={X.shape[1]}): {scores_str}")

    # Factory reps: per-fold evaluation
    if fold_reps and precomputed_folds:
        factory_rep_names = set()
        for fold_data in fold_reps.values():
            factory_rep_names.update(fold_data.keys())

        for rep_name in sorted(factory_rep_names):
            # Get dimension from first fold
            first_fold = next(iter(fold_reps.values()))
            if rep_name not in first_fold:
                continue
            dim = first_fold[rep_name][0].shape[1]
            baselines[rep_name] = {'dimension': dim}
            sec_all = {}

            # Per-fold high-dim filter for factory reps — same policy as
            # _evaluate_single. Currently no factory rep exceeds 5000D so
            # this is future-proofing; turns on automatically if that changes.
            _apply_hd_filter = dim > HIGH_DIM_FILTER_THRESHOLD
            if _apply_hd_filter:
                _hd_target_n = 500 if len(labels) >= 1000 else 200
                if _hd_target_n >= dim:
                    _apply_hd_filter = False
                else:
                    _hd_task_type = (task_type
                                     if task_type in ('classification', 'regression')
                                     else _detect_task_type(labels))

            for model_name, model in models.items():
                fold_scores = []
                fold_sec_scores = {name: [] for name in (secondary_metric_fns or {})}
                for fold_idx, (train_idx, val_idx) in enumerate(precomputed_folds):
                    if fold_idx not in fold_reps or rep_name not in fold_reps[fold_idx]:
                        continue
                    ft_train, ft_val = fold_reps[fold_idx][rep_name]
                    # Train on (possibly noisy) train labels; ALWAYS score on clean val.
                    y_train = (labels[train_idx] if train_labels is None
                               else np.asarray(train_labels).ravel()[train_idx])
                    y_val = labels[val_idx]

                    if _apply_hd_filter:
                        top_idx = _select_top_features_per_fold(
                            ft_train, y_train, _hd_target_n, _hd_task_type)
                        ft_train = ft_train[:, top_idx]
                        ft_val = ft_val[:, top_idx]

                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(ft_train)
                    X_val_s = scaler.transform(ft_val)

                    m = deepcopy(model)
                    _fg = groups[train_idx] if groups is not None else None
                    _fit_model(m, X_train_s, y_train, groups=_fg)
                    y_pred = _predict_for_metric(m, X_val_s, metric_fn)
                    fold_scores.append(metric_fn(y_val, y_pred))

                    if secondary_metric_fns:
                        y_pred_labels = m.predict(X_val_s)
                        for sec_name, sec_fn in secondary_metric_fns.items():
                            try:
                                fold_sec_scores[sec_name].append(sec_fn(y_val, y_pred_labels))
                            except Exception:
                                pass

                if fold_scores:
                    baselines[rep_name][model_name] = round(float(np.mean(fold_scores)), 4)
                    _record_folds(rep_name, model_name, fold_scores)

                if secondary_metric_fns:
                    for sec_name, vals in fold_sec_scores.items():
                        if vals:
                            sec_all[f'{model_name}_{sec_name}'] = {
                                'mean': round(float(np.mean(vals)), 4),
                                'std': round(float(np.std(vals)), 4),
                            }

            if sec_all:
                baselines[rep_name]['_secondary'] = sec_all

            if verbose:
                scores_str = ", ".join(
                    f"{m}={baselines[rep_name].get(m, 'N/A')}" for m in models
                )
                print(f"  {rep_name} (dim={dim}, fine-tuned): {scores_str}")

    return baselines, baseline_fold_scores


# =============================================================================
# SEARCH LOOP
# =============================================================================

# How many freshly-computed configs to buffer before flushing to the per-dataset
# SQLite store. Small enough that a killed job loses little; large enough to
# avoid a commit per config. Each flush is its own transaction (crash-safe).
_DB_FLUSH_INTERVAL = 25

# The step_size the main search loop uses (it does not pass step_size into
# _evaluate_hybrid_config, so the default applies). MUST match that default;
# two-phase refinement is the only path that varies step_size per result.
_DEFAULT_STEP_SIZE = 5


def _result_to_db_row(identity: Dict[str, Any], result: dict,
                      ranking_metric: Optional[str],
                      config_const: Optional[Dict[str, Any]] = None,
                      rep_pool: Optional[str] = None) -> Dict[str, Any]:
    """Build a result_cache row dict from an `identity` + a result dict.

    The SWEPT config columns are read from the RESULT (which carries every one
    of them), so this works for both the main loop and two-phase refinement
    (which varies step_size). `config_const` supplies the invocation-constant
    config columns (the aug recipe + greedy/candidate knobs); `rep_pool` is the
    recorded reps-used data. `rep_subset` defaults to 'all' (the engine only
    sets result['rep_subset'] for leave-one-out subsets).
    """
    row = dict(identity)
    if config_const:
        row.update(config_const)
    row['rep_pool'] = rep_pool
    row['rep_subset'] = result.get('rep_subset', 'all')
    row['allocation_method'] = result.get('allocation_method')
    row['budget'] = result.get('budget')
    row['importance_method'] = result.get('importance_method')
    row['model'] = result.get('model')
    row['internal_evaluator'] = result.get('internal_evaluator')
    row['step_size'] = result.get('step_size')
    row['score_mean'] = result.get('score_mean')
    row['score_std'] = result.get('score_std')
    row['ranking_metric'] = ranking_metric
    row['fold_scores'] = result.get('fold_scores')
    row['metrics'] = result.get('metrics')  # full metric set (None on older rows)
    # NOTE: per-fold HYBRID-candidate scores ride in 'fold_scores' above (they
    # survive the result_cache DB via fold_scores_json). Per-fold BASELINE
    # scores are NOT per-config rows — they persist via the per-rep baseline
    # cache payload ('fold_scores' key) and SearchResult.baseline_fold_scores.
    # No baseline column is added here: RESULT_COLS in result_cache.py (owned
    # elsewhere) is fixed, so an extra key would be silently dropped on upsert.
    row['n_features'] = result.get('n_features')
    row['time_seconds'] = result.get('time_seconds')
    row['had_imputation'] = result.get('had_imputation')
    row['blueprint'] = result.get('blueprint')
    return row


def _db_row_to_result(row: Dict[str, Any]) -> dict:
    """Reconstruct an `all_results`-shaped dict from a stored result_cache row.

    Key columns come back as canonical TEXT, so the numeric config fields
    (budget, step_size) are coerced back to int to match what `_eval_cfg`
    produces and what `_aggregate_results` / `best_by_model_budget` compare on.
    """
    def _as_int(v):
        s = '' if v is None else str(v).strip()
        return int(s) if s not in ('', 'None') else None

    r = {
        'blueprint': row.get('blueprint'),
        'model': row.get('model'),
        'score_mean': row.get('score_mean'),
        'score_std': row.get('score_std'),
        'fold_scores': row.get('fold_scores'),
        'n_features': _as_int(row.get('n_features')),
        'time_seconds': row.get('time_seconds'),
        'allocation_method': row.get('allocation_method'),
        'importance_method': row.get('importance_method'),
        'internal_evaluator': row.get('internal_evaluator'),
        'budget': _as_int(row.get('budget')),
        'step_size': _as_int(row.get('step_size')),
        'had_imputation': bool(row.get('had_imputation')),
        'metrics': row.get('metrics'),
    }
    # The engine attaches 'rep_subset' only for non-'all' rep sets.
    if row.get('rep_subset') not in ('all', '', None):
        r['rep_subset'] = row['rep_subset']
    return r


def _run_search(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: Optional[np.ndarray],
    speed: str,
    models: Dict[str, Any],
    metric_fn: Callable,
    budgets: List[int],
    allocation_methods: Optional[List[str]] = None,
    importance_methods: Optional[List[str]] = None,
    internal_evaluators: Optional[List[str]] = None,
    objective: str = OBJECTIVE_PREDICTIVE,
    n_folds: Optional[int] = None,
    test_rep_subsets: bool = False,
    seed: int = 42,
    verbose: bool = True,
    augmentations: Optional[Dict[str, np.ndarray]] = None,
    augmentation_strategy: str = 'uncapped',
    augmentation_threshold: Optional[float] = None,
    augmentation_budget: int = 20,
    aug_per_source_budget: Optional[int] = None,
    candidate_ranking: str = 'rf_importance',
    lower_is_better: bool = False,
    fold_reps: Optional[Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = None,
    precomputed_folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    cv_splitter=None,
    greedy_tolerance: float = 0.0,
    metric_name: Optional[str] = None,
    adaptive_pruning: Union[bool, float] = True,
    adaptive_pruning_schedule: Optional[Dict[str, Any]] = None,
    model_pruning_aggressiveness: Optional[Dict[str, Dict[str, Any]]] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_meta: Optional[Dict[str, Any]] = None,
    # hybrid_eval cache context.
    dataset_name: Optional[str] = None,
    task_type: Optional[str] = None,
    baselines: Optional[Dict[str, Dict[str, float]]] = None,
    # Caller-supplied importance_cache (skips the expensive per-fold/method
    # rebuild when non-empty). Used by experiments that reuse precompute's
    # importance scores.
    importance_cache_override: Optional[Dict] = None,
    # Per-dataset SQLite result store. When identity+dataset are
    # given, prior configs are loaded from the DB and skipped, and newly
    # computed configs are written back. Replaces the searches/*.pkl bundle.
    result_db_identity: Optional[Dict[str, Any]] = None,
    result_db_dataset: Optional[str] = None,
    result_db_cache_dir: Optional[Path] = None,
    result_db_config_const: Optional[Dict[str, Any]] = None,
    result_db_rep_pool: Optional[str] = None,
    # Label noise. `train_labels` is the noisy FULL label column (fold-independent,
    # deterministic — see kirby.noise_spec). Everything the model is allowed to see
    # (importance, allocation, hybrid construction, model fit) reads train-fold rows
    # from it; scoring always reads the clean `labels`.
    train_labels: Optional[np.ndarray] = None,
    noise_id: Optional[str] = None,
) -> Tuple[List[dict], Dict, Dict]:
    """Main search loop over all configurations.

    Adaptive pruning (Phase 2 multi-stage version):

    - Default schedule = empty checkpoints + final_strategy_threshold=0.15
      (matches legacy adaptive_pruning=True behavior bit-for-bit).
    - When schedule['checkpoints'] is non-empty AND tail_drop_per_checkpoint>0,
      mid-flight pruning fires after fold 0 has finished evaluating all configs
      at each checkpoint budget. Within each (model, evaluator) group, configs
      with fold-0 scores in the bottom tail_drop fraction are dropped — their
      (alloc, importance) pair is excluded from higher budgets in the same
      group AND from full-eval. min_configs_per_model is the floor: every
      model retains at least N surviving configs so best_by_model[m] is
      always populated.
    - End-of-fold-0 strategy pruning at final_strategy_threshold (top-3 in
      <X% of groups) runs after mid-flight pruning, on whatever survives.

    Backwards compat: legacy `adaptive_pruning` param is translated into a
    schedule with empty checkpoints. Callers passing the new
    `adaptive_pruning_schedule` override the legacy param.

    If checkpoint_path is set, periodically saves progress so the search
    can resume after a SLURM timeout without losing work.
    """
    tier = SPEED_TIERS[speed]
    search_importance = importance_methods if importance_methods is not None else tier['importance_methods']
    search_allocation = allocation_methods if allocation_methods is not None else tier['allocation_methods']
    search_evaluators = internal_evaluators if internal_evaluators is not None else tier['internal_evaluators']
    search_budgets = budgets
    search_folds = n_folds if n_folds is not None else tier['n_folds']

    # Resolve 'match' evaluators: for each model, the evaluator IS that model
    use_matched = 'match' in search_evaluators
    explicit_evaluators = [e for e in search_evaluators if e != 'match']

    # Build list of rep sets to test
    rep_sets = [base_reps]
    rep_set_names = ['all']

    if test_rep_subsets and len(base_reps) >= 3:
        for leave_out in base_reps:
            subset = {k: v for k, v in base_reps.items() if k != leave_out}
            rep_sets.append(subset)
            rep_set_names.append(f'no_{leave_out}')

    # --- top_k guard: skip top_k strategies where k >= n_reps ---
    n_reps = len(base_reps) + (len(fold_reps.get(0, {})) if fold_reps else 0)
    _filtered_alloc = []
    for alloc in search_allocation:
        if alloc.startswith('top_'):
            try:
                k = int(alloc.split('_')[1])
                if k >= n_reps:
                    if verbose:
                        print(f"  Skipping {alloc}: k={k} >= n_reps={n_reps} "
                              f"(equivalent to performance_weighted)")
                    continue
            except (IndexError, ValueError):
                pass
        _filtered_alloc.append(alloc)
    search_allocation = _filtered_alloc

    # Build effective evaluator list per model
    def _evaluators_for_model(model_name):
        """Return list of evaluator names to test for a given downstream model."""
        evs = list(explicit_evaluators)
        if use_matched:
            # Map model name to evaluator name
            match_map = {
                'rf': 'rf', 'lightgbm': 'lightgbm', 'xgboost': 'xgboost',
                'mlp': 'mlp', 'svm': 'svm', 'gp': 'gp',
                'knn': 'knn', 'catboost': 'catboost', 'dnn': 'dnn', 'cnn': 'cnn',
                'bnn': 'bnn', 'ngboost': 'ngboost',
            }
            matched_ev = match_map.get(model_name)
            if matched_ev and matched_ev not in evs:
                evs.append(matched_ev)
        return evs if evs else ['rf']

    def _budgets_for_model(model_name):
        """Return budgets filtered by per-model cap from MODEL_MAX_BUDGET."""
        cap = MODEL_MAX_BUDGET.get(model_name)
        if cap is None:
            return search_budgets
        return [b for b in search_budgets if b <= cap]

    # Compute effective config count (respects per-model budget caps)
    total = 0
    for alloc in search_allocation:
        n_imp = len(search_importance)
        for model_name in models:
            n_ev = len(_evaluators_for_model(model_name))
            n_bgt = len(_budgets_for_model(model_name))
            total += len(rep_sets) * n_bgt * n_imp * n_ev

    if verbose:
        print(f"\nSearching {total} unique configurations ({speed} mode)...")

    # ── Label-noise source for every TRAIN-side quantity ──
    # When no noise is active this IS `labels`, so behaviour is byte-identical to
    # before. Note the fold BOUNDARIES above are always computed from the clean
    # labels — they must not move when the noise draw changes.
    _train_label_src = (labels if train_labels is None
                        else np.asarray(train_labels).ravel())

    # ── Pre-compute importance scores for deduplication ──
    # Importance for a given (rep, fold_train_data, method) is identical across
    # all (alloc, budget, evaluator, model) configs. Compute once, reuse everywhere.
    from kirby.hybrid import compute_feature_importance

    if precomputed_folds is not None:
        _fold_splits_for_cache = precomputed_folds
    else:
        _splitter, _use_groups = _make_cv_splitter(groups, search_folds, seed,
                                                   cv_splitter=cv_splitter,
                                                   task_type=task_type)
        _sample_X = next(iter(base_reps.values()))
        _split_args = (_sample_X, labels, groups) if _use_groups else (_sample_X, labels)
        _fold_splits_for_cache = list(_splitter.split(*_split_args))

    # Use caller-supplied importance_cache if provided and non-empty;
    # otherwise build from scratch.
    if importance_cache_override:
        importance_cache = importance_cache_override
        if verbose:
            print(f"  Using caller-supplied importance_cache "
                  f"({len(importance_cache)} entries) — skipping rebuild")
    else:
        importance_cache = {}  # (fold_idx, imp_method) -> {rep_name: scores_array}
        n_imp_total = len(_fold_splits_for_cache) * len(search_importance)
        if verbose:
            print(f"  Pre-computing importance: {len(search_importance)} methods × "
                  f"{len(_fold_splits_for_cache)} folds × {len(base_reps)} reps...")
        _imp_t0 = time.time()
        for fold_idx, (train_idx, _) in enumerate(_fold_splits_for_cache):
            fold_train_reps = {k: v[train_idx] for k, v in base_reps.items()}
            # Feature importance is a TRAIN-side quantity: under label noise it must
            # be computed from the noisy train labels, like everything else the model
            # is allowed to see.
            fold_train_labels = _train_label_src[train_idx]
            # Merge fine-tuned reps for this fold
            if fold_reps and fold_idx in fold_reps:
                for rep_name, (ft_train, _) in fold_reps[fold_idx].items():
                    fold_train_reps[rep_name] = ft_train
            for imp_method in search_importance:
                try:
                    importance_cache[(fold_idx, imp_method)] = compute_feature_importance(
                        fold_train_reps, fold_train_labels, method=imp_method)
                except Exception as e:
                    if verbose:
                        print(f"    importance {imp_method} fold {fold_idx}: FAILED ({e})")
        if verbose:
            print(f"  Importance pre-computed in {time.time() - _imp_t0:.1f}s "
                  f"({len(importance_cache)}/{n_imp_total} successful)")

    # ── Pre-compute allocation baseline scores for deduplication ──
    # allocate_performance_weighted trains the internal evaluator on each rep
    # to score them. For a given (fold, evaluator), these scores are identical
    # across all (budget, importance, model) configs. Compute once, reuse.
    from kirby.hybrid import _evaluate_allocation

    alloc_cache = {}  # (fold_idx, evaluator) -> {rep_name: score}  (R²/accuracy, backward compat)
                      # (fold_idx, evaluator, metric_name) -> {rep_name: comparable_score}
    _alloc_t0 = time.time()

    # Collect ALL unique evaluator names that will actually be used.
    # This includes explicit evaluators AND matched evaluators (resolved per model).
    _all_evaluator_names = set(explicit_evaluators)
    if use_matched:
        for model_name in models:
            for ev in _evaluators_for_model(model_name):
                _all_evaluator_names.add(ev)

    # Metrics to pre-compute from each evaluator's predictions
    _alloc_metrics = {}
    for mname, (mfn, mlib) in BUILTIN_METRICS.items():
        _alloc_metrics[mname] = (mfn, mlib)

    # Similarity mode does not use model-prediction alloc scores — the
    # allocators compute live SAC per rep instead. Skip the precompute (it would
    # only fit models whose 'sac' metric is a no-op guard) and leave alloc_cache
    # empty; _evaluate_hybrid_config forces fold_alloc_scores=None in sim mode.
    _skip_alloc_precompute = (objective == OBJECTIVE_SIMILARITY_SAC)

    for fold_idx, (train_idx, _) in (
            [] if _skip_alloc_precompute else enumerate(_fold_splits_for_cache)):
        fold_train_reps = {k: v[train_idx] for k, v in base_reps.items()}
        # Allocation scores are fitted inside the train fold (they get split again
        # internally), so under label noise they too must see only noisy labels.
        fold_train_labels = _train_label_src[train_idx]
        if fold_reps and fold_idx in fold_reps:
            for rep_name, (ft_train, _) in fold_reps[fold_idx].items():
                fold_train_reps[rep_name] = ft_train

        # Replicate create_hybrid's internal train/val split (group-aware)
        n_fold = len(fold_train_labels)
        fold_train_groups = groups[train_idx] if groups is not None else None
        if fold_train_groups is not None and len(np.unique(fold_train_groups)) >= 2:
            from sklearn.model_selection import GroupShuffleSplit
            _gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            _tr_idx, _va_idx = next(_gss.split(
                np.zeros(n_fold), fold_train_labels, fold_train_groups))
        else:
            n_val_internal = max(1, int(n_fold * 0.2))
            _indices = np.random.RandomState(42).permutation(n_fold)
            _tr_idx = _indices[n_val_internal:]
            _va_idx = _indices[:n_val_internal]
        _train_labels = fold_train_labels[_tr_idx]
        _val_labels = fold_train_labels[_va_idx]

        for ev in _all_evaluator_names:
            default_scores = {}   # R²/accuracy (backward compat)
            metric_scores = {mn: {} for mn in _alloc_metrics}

            for rep_name, X_full in fold_train_reps.items():
                X_tr = X_full[_tr_idx].copy()
                X_va = X_full[_va_idx].copy()
                X_tr, X_va, _ = _impute_nan(X_tr, X_va)
                try:
                    default_score, y_pred = _evaluate_allocation(
                        X_tr, X_va, _train_labels, _val_labels,
                        internal_evaluator=ev, return_predictions=True)
                    default_scores[rep_name] = max(default_score, 0.0)

                    # Compute all metrics from same predictions
                    if y_pred is not None:
                        for mn, (mfn, mlib) in _alloc_metrics.items():
                            try:
                                raw = mfn(_val_labels, y_pred)
                                # Convert to "higher = better" comparable form.
                                # No max(,0) clipping: negated RMSE=-0.5 means
                                # "RMSE of 0.5" and must stay negative so reps
                                # can be ranked correctly.
                                metric_scores[mn][rep_name] = -raw if mlib else raw
                            except Exception:
                                metric_scores[mn][rep_name] = 0.0
                    else:
                        for mn in _alloc_metrics:
                            metric_scores[mn][rep_name] = 0.0
                except Exception:
                    default_scores[rep_name] = 0.0
                    for mn in _alloc_metrics:
                        metric_scores[mn][rep_name] = 0.0

            alloc_cache[(fold_idx, ev)] = default_scores
            for mn, rep_scores in metric_scores.items():
                # Shift scores to be non-negative (allocation weighting
                # requires >= 0). Preserves relative ordering.
                if rep_scores:
                    min_val = min(rep_scores.values())
                    if min_val < 0:
                        rep_scores = {rn: s - min_val for rn, s in rep_scores.items()}
                alloc_cache[(fold_idx, ev, mn)] = rep_scores

    if verbose:
        print(f"  Allocation scores pre-computed in {time.time() - _alloc_t0:.1f}s "
              f"({len(_all_evaluator_names)} evaluators × {len(BUILTIN_METRICS)} metrics, "
              f"evaluators: {sorted(_all_evaluator_names)})")

    # ── Build flat list of all configs to evaluate ──
    configs = []
    for rep_set, rep_set_name in zip(rep_sets, rep_set_names):
        for alloc in search_allocation:
            effective_importance = search_importance

            for model_name, model in models.items():
                model_budgets = _budgets_for_model(model_name)
                model_evaluators = _evaluators_for_model(model_name)
                for bgt in model_budgets:
                    for imp in effective_importance:
                        for eval_name in model_evaluators:
                            configs.append((
                                rep_set, rep_set_name, alloc, bgt, imp,
                                model_name, model, eval_name,
                            ))

    # ── Optional expensive-first ordering (opt-in; default OFF) ──
    # When KIRBY_EXPENSIVE_FIRST / KIRBY_PRECOMPUTE_EXPENSIVE_FIRST is set, sort
    # the flat config list so the costliest cells (slow models, largest budgets)
    # run FIRST. This is the inner-loop complement to the precompute reverse flag
    # (which only reorders the OUTER condition list). On a wall-clock TIMEOUT it
    # makes the cheap, fast-to-recompute cells the casualties instead of the
    # gp/svm/large-budget cells you most want to keep. RESULT-NEUTRAL on a
    # complete run: the best config is chosen by max score regardless of
    # iteration order, and checkpoint/resume keys on config identity not index.
    # Default (no env var) leaves `configs` in its original build order.
    if os.environ.get('KIRBY_EXPENSIVE_FIRST') or \
            os.environ.get('KIRBY_PRECOMPUTE_EXPENSIVE_FIRST'):
        # Lower rank == more expensive == sorted earlier. Budget DESC within a model.
        _MODEL_COST_RANK = {
            'gp': 0, 'svm': 1, 'cnn': 2, 'dnn': 3, 'bnn': 3, 'mlp': 4,
            'catboost': 5, 'lightgbm': 6, 'xgboost': 7, 'rf': 8,
            'knn': 9,
        }
        # config tuple = (rep_set, rep_set_name, alloc, bgt, imp, model_name, model, eval_name)
        configs.sort(key=lambda c: (_MODEL_COST_RANK.get(c[5], 5), -c[3]))

    # ── Evaluate configs sequentially ──
    # sklearn models use n_jobs=-1 internally for C-level parallelism.
    # Python-level threading on top causes CPU oversubscription and GIL contention.
    selection_dedup = {'_hits': 0}

    _common_kwargs = dict(
        labels=labels, groups=groups, n_folds=search_folds,
        metric_fn=metric_fn, metric_name=metric_name, seed=seed,
        augmentations=augmentations, augmentation_strategy=augmentation_strategy,
        augmentation_threshold=augmentation_threshold,
        augmentation_budget=augmentation_budget,
        aug_per_source_budget=aug_per_source_budget,
        candidate_ranking=candidate_ranking, lower_is_better=lower_is_better,
        fold_reps=fold_reps, precomputed_folds=precomputed_folds,
        cv_splitter=cv_splitter, importance_cache=importance_cache,
        alloc_cache=alloc_cache, selection_dedup=selection_dedup,
        greedy_tolerance=greedy_tolerance,
        objective=objective,
        # Cross-invocation dedup cache activates only when the caller
        # passes dataset_name + task_type. Otherwise no-ops.
        dataset_name=dataset_name,
        task_type=task_type,
        baselines=baselines,
        train_labels=train_labels,
        noise_id=noise_id,
    )

    def _eval_cfg(cfg, fold_indices=None):
        (rep_set, rep_set_name, alloc, bgt, imp,
         model_name, model, eval_name) = cfg
        result = _evaluate_hybrid_config(
            base_reps=rep_set, allocation_method=alloc, budget=bgt,
            importance_method=imp, model=model, model_name=model_name,
            internal_evaluator=eval_name, fold_indices=fold_indices,
            **_common_kwargs,
        )
        if rep_set_name != 'all':
            result['rep_subset'] = rep_set_name
        return result

    # ── Checkpoint helpers ──
    _CHECKPOINT_INTERVAL = 100  # save every N configs

    def _save_checkpoint(data):
        if checkpoint_path is None:
            return
        try:
            import tempfile
            # Wrap payload with meta (same envelope as _save_cache_with_meta)
            # so _load_checkpoint can validate parameter alignment.
            wrapper = {
                '_cache_meta': checkpoint_meta or {},
                'payload': data,
            }
            fd, tmp = tempfile.mkstemp(dir=str(checkpoint_path.parent), suffix='.tmp')
            with os.fdopen(fd, 'wb') as f:
                f.write(pickle.dumps(wrapper))
            os.replace(tmp, str(checkpoint_path))
        except Exception:
            pass

    def _load_checkpoint():
        """Load checkpoint, validating that its meta matches the current run.

        Returns None on missing file, unpickle error, missing envelope, or
        meta mismatch. A mismatched checkpoint is deleted so a crashed
        resume can't pick up an incompatible partial result.
        """
        if checkpoint_path is None or not checkpoint_path.exists():
            return None
        try:
            wrapper = pickle.loads(checkpoint_path.read_bytes())
        except Exception:
            return None
        # Legacy un-wrapped checkpoints: treat as invalid (can't verify).
        if not isinstance(wrapper, dict) or 'payload' not in wrapper:
            if verbose:
                print(f"  Checkpoint {checkpoint_path.name}: un-validated "
                      f"legacy format — discarding.")
            try:
                checkpoint_path.unlink()
            except Exception:
                pass
            return None
        stored = wrapper.get('_cache_meta') or {}
        if checkpoint_meta is not None:
            for k, v in checkpoint_meta.items():
                if k in ('code_timestamp',):
                    continue
                if k not in stored or stored[k] != v:
                    if verbose:
                        print(f"  Checkpoint {checkpoint_path.name} stale "
                              f"({k} drifted) — discarding and recomputing.")
                    try:
                        checkpoint_path.unlink()
                    except Exception:
                        pass
                    return None
        return wrapper['payload']

    def _cfg_key(cfg):
        """Hashable key for a config tuple (excludes model callable)."""
        return (cfg[1], cfg[2], cfg[3], cfg[4], cfg[5], cfg[7])  # rep_set_name, alloc, bgt, imp, model_name, eval_name

    ckpt = _load_checkpoint()

    # ── Adaptive pruning (multi-stage): mid-flight + end-of-fold-0 ──
    pruned_strategies = set()
    dropped_pairs: Dict[Tuple[str, str], set] = {}  # (model, eval) -> {(alloc, imp)}

    # Resolve the global schedule once. Default = legacy single-stage behavior.
    _global_schedule = _normalize_pruning_schedule(
        adaptive_pruning=adaptive_pruning,
        adaptive_pruning_schedule=adaptive_pruning_schedule,
    )
    _prune_frac = _global_schedule['final_strategy_threshold']

    # Build per-model effective schedules. Models with no override use the
    # global schedule; overridden models (e.g. {'gp': {'checkpoints': [10,25]}})
    # have their schedule fields shallow-merged.
    _model_schedules: Dict[str, Dict[str, Any]] = {}
    for mname in models.keys():
        _model_schedules[mname] = _resolve_model_schedule(
            mname, _global_schedule, model_pruning_aggressiveness)

    # Union of all checkpoints across global + per-model schedules. The fold-0
    # loop fires the pruning step at each checkpoint; per-model filters inside
    # _compute_tail_prunes ensure each schedule only prunes its own models.
    _all_checkpoints = sorted(set(_global_schedule['checkpoints']) |
                              set(b for s in _model_schedules.values()
                                  for b in s['checkpoints']))
    # Per-checkpoint: which models is this checkpoint active for?
    _models_at_ckpt: Dict[int, set] = {b: set() for b in _all_checkpoints}
    for mname, s in _model_schedules.items():
        for b in s['checkpoints']:
            if s['tail_drop_per_checkpoint'] > 0:
                _models_at_ckpt.setdefault(b, set()).add(mname)
    _mid_flight_active = bool(_all_checkpoints) and any(
        _models_at_ckpt[b] for b in _all_checkpoints)

    # Adaptive pruning runs only when there are enough strategies AND folds > 1.
    # The legacy 'truthy adaptive_pruning' check is preserved via threshold > 0
    # OR mid-flight active.
    _pruning_enabled = (search_folds > 1 and len(search_allocation) > 3
                        and (_prune_frac > 0 or _mid_flight_active))

    if _pruning_enabled:
        # Check if fold 0 is already done (from checkpoint)
        if ckpt and ckpt.get('fold0_done'):
            fold0_results_saved = ckpt['fold0_results']  # list of (cfg_key, result)
            pruned_strategies = ckpt.get('pruned_strategies', set())
            # Restore mid-flight pruning state (Phase 2). Lists from JSON-style
            # serialization are converted back to sets for fast membership.
            saved_drops = ckpt.get('dropped_pairs') or {}
            dropped_pairs = {tuple(k) if isinstance(k, list) else k:
                             set(tuple(p) for p in v)
                             for k, v in saved_drops.items()}
            if verbose:
                print(f"\n  Resumed from checkpoint: fold 0 complete "
                      f"({len(fold0_results_saved)} results, "
                      f"{len(pruned_strategies)} strategies pruned, "
                      f"{sum(len(v) for v in dropped_pairs.values())} mid-flight pairs dropped)")
        else:
            if verbose:
                print(f"\n  Adaptive pruning: evaluating fold 0 for {total} configs...")
                if _mid_flight_active:
                    print(f"    Schedule: global checkpoints={_global_schedule['checkpoints']} "
                          f"global tail_drop={_global_schedule['tail_drop_per_checkpoint']:.0%}")
                    overrides_active = [m for m, s in _model_schedules.items()
                                        if s != _global_schedule]
                    if overrides_active:
                        print(f"    Per-model overrides active for: {sorted(overrides_active)}")

            # Resume fold 0 from partial checkpoint
            fold0_done_keys = set()
            fold0_results = []
            if ckpt and 'fold0_results' in ckpt:
                fold0_results = ckpt['fold0_results']
                fold0_done_keys = {r[0] for r in fold0_results}
                # Resume mid-flight drops too
                saved_drops = ckpt.get('dropped_pairs') or {}
                dropped_pairs = {tuple(k) if isinstance(k, list) else k:
                                 set(tuple(p) for p in v)
                                 for k, v in saved_drops.items()}
                if verbose:
                    print(f"    Resuming fold 0: {len(fold0_done_keys)}/{total} already done")

            # When mid-flight pruning is active, iterate configs by ascending
            # budget so we can fire the checkpoint pruning step after the LAST
            # config at each checkpoint budget completes. Order within the same
            # budget is preserved (model, evaluator, alloc, importance).
            if _mid_flight_active:
                _iter_configs = sorted(
                    configs,
                    key=lambda c: (c[3], c[5], c[7], c[2], c[4]))
            else:
                _iter_configs = configs

            def _is_dropped(cfg):
                grp = (cfg[5], cfg[7])
                if grp not in dropped_pairs:
                    return False
                return (cfg[2], cfg[4]) in dropped_pairs[grp]

            n_iter = len(_iter_configs)
            for idx, cfg in enumerate(_iter_configs, 1):
                ck = _cfg_key(cfg)
                if ck in fold0_done_keys:
                    continue
                if _is_dropped(cfg):
                    continue
                try:
                    result = _eval_cfg(cfg, fold_indices=[0])
                    fold0_results.append((ck, result))
                except Exception:
                    pass

                # Mid-flight pruning: fires on transition out of a checkpoint budget.
                # Each per-model schedule with tail_drop>0 can fire at its OWN
                # checkpoints; the union is _all_checkpoints. _models_at_ckpt[b]
                # tells us which models to prune at budget b.
                if _mid_flight_active:
                    cur_b = cfg[3]
                    is_last = idx == n_iter
                    next_b = _iter_configs[idx][3] if not is_last else None
                    crossing = is_last or next_b != cur_b
                    if cur_b in _all_checkpoints and crossing:
                        active_models = _models_at_ckpt.get(cur_b, set())
                        if active_models:
                            # Group active models by their (tail_drop, min_configs)
                            # so we make one _compute_tail_prunes call per unique
                            # (frac, floor, model_subset).
                            from collections import defaultdict as _dd
                            buckets = _dd(set)
                            for mname in active_models:
                                s = _model_schedules[mname]
                                if cur_b not in s['checkpoints']:
                                    continue
                                bucket_key = (s['tail_drop_per_checkpoint'],
                                              s['min_configs_per_model'])
                                buckets[bucket_key].add(mname)
                            new_drops_total = {}
                            for (td, mc), mset in buckets.items():
                                drops = _compute_tail_prunes(
                                    fold0_results, cur_b, td, mc,
                                    lower_is_better, model_filter=mset)
                                for grp, pairs in drops.items():
                                    new_drops_total.setdefault(grp, set()).update(pairs)
                            for grp, pairs in new_drops_total.items():
                                dropped_pairs.setdefault(grp, set()).update(pairs)
                            if verbose and new_drops_total:
                                n_new = sum(len(v) for v in new_drops_total.values())
                                print(f"    checkpoint b={cur_b}: "
                                      f"pruned {n_new} (alloc, imp) pairs across "
                                      f"{len(new_drops_total)} (model, eval) groups "
                                      f"[active models: {sorted(active_models)}]")

                if verbose and idx % 200 == 0:
                    print(f"    fold 0: {idx}/{total}")
                if idx % _CHECKPOINT_INTERVAL == 0:
                    _save_checkpoint({
                        'fold0_done': False,
                        'fold0_results': fold0_results,
                        # Keep tuple keys (pickle-safe, hashable). Load side
                        # accepts either tuple or list keys.
                        'dropped_pairs': {k: list(v)
                                          for k, v in dropped_pairs.items()},
                    })

            # End-of-fold-0 strategy pruning (top-3 in <X% of groups).
            # Runs on whatever survives mid-flight pruning. Skipped if
            # _prune_frac == 0 (caller explicitly disabled strategy prune).
            from collections import defaultdict
            if _prune_frac > 0:
                groups_f0 = defaultdict(list)
                for ck, result in fold0_results:
                    _, alloc, bgt, imp, model_name, eval_name = ck
                    if (model_name, eval_name) in dropped_pairs and \
                       (alloc, imp) in dropped_pairs[(model_name, eval_name)]:
                        continue  # don't count tail-pruned configs
                    key = (imp, eval_name, model_name, bgt)
                    groups_f0[key].append((alloc, result['score_mean']))

                n_groups_f0 = len(groups_f0)
                threshold = n_groups_f0 * _prune_frac

                top3_counts = defaultdict(int)
                for key, entries in groups_f0.items():
                    ranked = sorted(entries, key=lambda x: x[1],
                                    reverse=not lower_is_better)
                    for alloc, _ in ranked[:3]:
                        top3_counts[alloc] += 1

                pruned_strategies = {s for s in search_allocation
                                     if top3_counts.get(s, 0) < threshold}

            # Save completed fold 0 + pruning decisions
            _save_checkpoint({
                'fold0_done': True,
                'fold0_results': fold0_results,
                'pruned_strategies': pruned_strategies,
                # Keep tuple keys (pickle-safe, hashable). Load side
                # accepts either tuple or list keys.
                'dropped_pairs': {k: list(v)
                                  for k, v in dropped_pairs.items()},
                'full_results': [],
                'full_done_keys': set(),
            })

        if pruned_strategies and verbose:
            print(f"  Pruned {len(pruned_strategies)}/{len(search_allocation)} strategies "
                  f"(top-3 in <{_prune_frac:.0%} of groups): {sorted(pruned_strategies)}")

        # Rebuild config list without pruned strategies AND tail-pruned pairs
        def _survives_pruning(cfg):
            if cfg[2] in pruned_strategies:
                return False
            grp = (cfg[5], cfg[7])
            if grp in dropped_pairs and (cfg[2], cfg[4]) in dropped_pairs[grp]:
                return False
            return True

        if pruned_strategies or dropped_pairs:
            configs = [cfg for cfg in configs if _survives_pruning(cfg)]
            total = len(configs)
            if verbose:
                print(f"  Remaining: {total} configs")

    # ── Full evaluation (all folds) ──
    # Resume from checkpoint if available
    results = []
    full_done_keys = set()

    # ── Durable cross-run reuse: per-dataset SQLite result store ──
    # The DB is the persistence layer. When active it SUPERSEDES the file
    # checkpoint's full-results resume (which would otherwise double-count): we
    # seed `results` from the DB instead. Only the per-config model fits are
    # skipped — the upfront per-fold importance/alloc caches still compute, so
    # the returned importance_cache/alloc_cache are always populated.
    _rc = None
    _rc_done = set()           # canonical config keys already in the DB
    _rc_new_rows = []          # freshly computed rows pending flush
    _rc_ranking_metric = metric_name
    _rc_cfg_const = result_db_config_const or {}

    def _rc_cfg_dict(rep_subset, alloc, bgt, imp, model_name, eval_name, step):
        # The full per-row config: the 7 swept fields + the invocation-constant
        # aug/candidate/greedy fields. cfg_key() reads it by column NAME.
        d = {'rep_subset': rep_subset, 'allocation_method': alloc, 'budget': bgt,
             'importance_method': imp, 'model': model_name,
             'internal_evaluator': eval_name, 'step_size': step}
        d.update(_rc_cfg_const)
        return d

    if result_db_identity is not None and result_db_dataset is not None:
        from .result_cache import ResultCache as _ResultCache, cfg_key as _rc_key
        _rc = _ResultCache(result_db_dataset, cache_dir=result_db_cache_dir)
        for _row in _rc.fetch_rows(result_db_identity):
            results.append(_db_row_to_result(_row))
            _rc_done.add(_row['_cfg_key'])
        if _rc_done and verbose:
            print(f"  ResultCache: {len(_rc_done)} stored configs reused; "
                  f"computing only the remainder of {total}")
    elif ckpt and 'full_results' in ckpt:
        # File-checkpoint resume only when the DB is NOT the store.
        results = ckpt['full_results']
        full_done_keys = ckpt.get('full_done_keys', set())
        if full_done_keys and verbose:
            print(f"  Resuming full eval: {len(full_done_keys)}/{total} already done")

    if verbose:
        print(f"\n  Evaluating {total} configs (all {search_folds} folds)...")

    for idx, cfg in enumerate(configs, 1):
        (_, rep_set_name, alloc, bgt, imp,
         model_name, _, eval_name) = cfg
        ck = _cfg_key(cfg)
        if ck in full_done_keys:
            continue
        if _rc is not None:
            # The main loop never varies step_size (it uses _evaluate_hybrid_config's
            # default, _DEFAULT_STEP_SIZE); only two-phase refinement does, and
            # those rows carry their own step_size, so they never match here.
            _ck_canon = _rc_key(_rc_cfg_dict(cfg[1], cfg[2], cfg[3], cfg[4],
                                             cfg[5], cfg[7], _DEFAULT_STEP_SIZE))
            if _ck_canon in _rc_done:
                continue
        try:
            result = _eval_cfg(cfg)
            results.append(result)
            full_done_keys.add(ck)
            # Persist to the DB ONLY when the score is finite. A NaN score (a
            # metric undefined on some/all folds — not an exception) is treated
            # like a transient failure: not stored, so it RE-ATTEMPTS next run
            # rather than being frozen as a permanent skip (matches the old
            # path's refusal to cache empty searches: missing = recompute).
            # It is still in `results` for THIS run's aggregation.
            _sm = result.get('score_mean')
            if _rc is not None and _sm is not None and np.isfinite(_sm):
                _rc_new_rows.append(
                    _result_to_db_row(result_db_identity, result,
                                      _rc_ranking_metric, _rc_cfg_const,
                                      result_db_rep_pool))
                _rc_done.add(_rc_key(_rc_cfg_dict(
                    cfg[1], cfg[2], cfg[3], cfg[4], cfg[5], cfg[7],
                    result.get('step_size', _DEFAULT_STEP_SIZE))))
                if len(_rc_new_rows) >= _DB_FLUSH_INTERVAL:
                    _rc.upsert_rows(_rc_new_rows)
                    _rc_new_rows.clear()

            if verbose:
                print(
                    f"  [{idx}/{total}] {model_name}/{alloc}/{imp}/b={bgt}/ev={eval_name}: "
                    f"score={result['score_mean']:.4f}+/-{result['score_std']:.3f}, "
                    f"dim={result['n_features']}"
                )

        except Exception as e:
            if verbose:
                print(
                    f"  [{idx}/{total}] {model_name}/{alloc}/{imp}/b={bgt}/ev={eval_name}: "
                    f"FAILED - {e}"
                )

        if idx % _CHECKPOINT_INTERVAL == 0 and _rc is None:
            _save_checkpoint({
                'fold0_done': True,
                'fold0_results': ckpt.get('fold0_results', []) if ckpt else [],
                'pruned_strategies': pruned_strategies,
                'full_results': results,
                'full_done_keys': full_done_keys,
            })

    # Final flush of any buffered rows + ALWAYS close the DB handle (a flush
    # failure must not leak the connection / -journal).
    if _rc is not None:
        try:
            if _rc_new_rows:
                _rc.upsert_rows(_rc_new_rows)
                _rc_new_rows.clear()
        finally:
            _rc.close()

    if verbose and selection_dedup:
        dedup_hits = selection_dedup.get('_hits', 0)
        dedup_entries = len(selection_dedup) - 1
        if dedup_hits > 0:
            print(f"  Feature selection dedup: {dedup_hits} fold evaluations skipped "
                  f"({dedup_entries} unique selections cached)")

    return results, importance_cache, alloc_cache


# =============================================================================
# TWO-PHASE REFINEMENT
# =============================================================================

# Allocation methods where step_size has an effect (greedy-based methods)
_STEP_SIZE_SENSITIVE = {'greedy', 'dominant_greedy'}



def _run_refinement_phase(
    phase1_results: List[dict],
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: Optional[np.ndarray],
    models: Dict[str, Any],
    metric_fn: Callable,
    n_folds: int,
    seed: int = 42,
    top_n: int = 5,
    step_sizes: Optional[List[int]] = None,
    verbose: bool = True,
    augmentations: Optional[Dict[str, np.ndarray]] = None,
    augmentation_strategy: str = 'uncapped',
    augmentation_threshold: Optional[float] = None,
    augmentation_budget: int = 20,
    aug_per_source_budget: Optional[int] = None,
    candidate_ranking: str = 'rf_importance',
    lower_is_better: bool = False,
    fold_reps: Optional[Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = None,
    precomputed_folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    cv_splitter=None,
    metric_name: Optional[str] = None,
) -> List[dict]:
    """Phase 2: re-evaluate top configs from Phase 1 with alternative step_sizes.

    Filters to greedy-based allocations only (step_size irrelevant for
    performance_weighted/fixed). Deduplicates by (model, alloc, budget, importance)
    before picking top N.

    Returns list of result dicts (same shape as _run_search output).
    """
    from kirby.hybrid import NAMED_ALLOCATIONS

    if step_sizes is None:
        step_sizes = [1, 2, 5, 10, 25]

    # Resolve named allocations to find which are step-size sensitive
    def _is_step_sensitive(alloc_name):
        base = alloc_name
        if alloc_name in NAMED_ALLOCATIONS:
            base = NAMED_ALLOCATIONS[alloc_name][0]
        return base in _STEP_SIZE_SENSITIVE

    # Filter to step-size sensitive configs
    sensitive = [r for r in phase1_results if _is_step_sensitive(r['allocation_method'])]
    if not sensitive:
        if verbose:
            print("  Refinement: no step-size-sensitive allocations to refine.")
        return []

    # Sort by best score first, deduplicate by (model, alloc, budget, importance).
    # Normalize importance for insensitive allocations to avoid duplicate entries.
    sensitive.sort(
        key=lambda r: _score_to_comparable(r['score_mean'], lower_is_better),
        reverse=True)
    seen = set()
    unique = []
    for r in sensitive:
        dedup_imp = r['importance_method']
        key = (r['model'], r['allocation_method'], r['budget'], dedup_imp)
        if key not in seen:
            seen.add(key)
            unique.append(r)
            if len(unique) >= top_n:
                break

    # Phase 1 default step_size
    phase1_step = 10

    # Evaluate each top config with alternative step_sizes
    refinement_results = []
    total = len(unique) * len([s for s in step_sizes if s != phase1_step])

    if verbose:
        print(f"\nRefinement phase: {len(unique)} configs x "
              f"{len(step_sizes) - (1 if phase1_step in step_sizes else 0)} step_sizes "
              f"= {total} evaluations")

    idx = 0
    for r in unique:
        model_name = r['model']
        model = models[model_name]
        for ss in step_sizes:
            if ss == r.get('step_size', phase1_step):
                continue  # Already evaluated in phase 1
            idx += 1
            try:
                result = _evaluate_hybrid_config(
                    base_reps=base_reps,
                    labels=labels,
                    allocation_method=r['allocation_method'],
                    budget=r['budget'],
                    importance_method=r['importance_method'],
                    model=model,
                    model_name=model_name,
                    internal_evaluator=r.get('internal_evaluator', 'match'),
                    groups=groups,
                    n_folds=n_folds,
                    metric_fn=metric_fn,
                    metric_name=metric_name,
                    seed=seed,
                    step_size=ss,
                    augmentations=augmentations,
                    augmentation_strategy=augmentation_strategy,
                    augmentation_threshold=augmentation_threshold,
                    augmentation_budget=augmentation_budget,
                    aug_per_source_budget=aug_per_source_budget,
                    candidate_ranking=candidate_ranking,
                    lower_is_better=lower_is_better,
                    fold_reps=fold_reps,
                    precomputed_folds=precomputed_folds,
                    cv_splitter=cv_splitter,
                )
                result['refinement_phase'] = True
                refinement_results.append(result)

                if verbose:
                    print(
                        f"  [refine {idx}/{total}] {model_name}/"
                        f"{r['allocation_method']}/b={r['budget']}/step={ss}: "
                        f"score={result['score_mean']:.4f}+/-{result['score_std']:.3f}"
                    )
            except Exception as e:
                if verbose:
                    print(
                        f"  [refine {idx}/{total}] {model_name}/"
                        f"{r['allocation_method']}/b={r['budget']}/step={ss}: "
                        f"FAILED - {e}"
                    )

    return refinement_results


# =============================================================================
# RESULT AGGREGATION
# =============================================================================

def _aggregate_results(
    results: List[dict],
    baselines: Dict[str, Dict[str, float]],
    lower_is_better: bool,
    dimension_tiers: List[int],
) -> dict:
    """Aggregate results into best_by_model, best_by_dimension, performance_matrix."""
    if not results:
        return {
            'best_by_model': {},
            'best_by_dimension': {},
            'best_by_budget': {},
            'best_by_model_budget': {},
            'performance_matrix': {},
        }

    def _key(r):
        return _score_to_comparable(r['score_mean'], lower_is_better)

    # Best by model
    model_names = sorted(set(r['model'] for r in results))
    best_by_model = {}
    for mname in model_names:
        model_results = [r for r in results if r['model'] == mname]
        if model_results:
            best_by_model[mname] = max(model_results, key=_key)

    # Best by dimension tier
    best_by_dimension = {}
    for tier in dimension_tiers:
        eligible = [r for r in results if r['n_features'] <= tier]
        if eligible:
            best_by_dimension[tier] = max(eligible, key=_key)

    # Best by budget
    budget_values = sorted(set(r['budget'] for r in results))
    best_by_budget = {}
    for bgt in budget_values:
        budget_results = [r for r in results if r['budget'] == bgt]
        if budget_results:
            best_by_budget[bgt] = max(budget_results, key=_key)

    # Best by (model, budget) — primary selection unit.
    # For each downstream model, find the best feature config per budget.
    # Avoids selection overfitting by not picking across models.
    best_by_model_budget = {}
    for mname in model_names:
        best_by_model_budget[mname] = {}
        for bgt in budget_values:
            eligible = [r for r in results
                        if r['model'] == mname and r['budget'] == bgt]
            if eligible:
                best_by_model_budget[mname][bgt] = max(eligible, key=_key)

    # Performance matrix: model x dimension
    performance_matrix = {}
    for mname in model_names:
        performance_matrix[mname] = {}
        for tier in dimension_tiers:
            eligible = [
                r for r in results
                if r['model'] == mname and r['n_features'] <= tier
            ]
            if eligible:
                best = max(eligible, key=_key)
                performance_matrix[mname][str(tier)] = round(best['score_mean'], 4)

    return {
        'best_by_model': best_by_model,
        'best_by_dimension': best_by_dimension,
        'best_by_budget': best_by_budget,
        'best_by_model_budget': best_by_model_budget,
        'performance_matrix': performance_matrix,
    }


# =============================================================================
# FITTED RESULT
# =============================================================================

def _compute_fitted_result(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    blueprint: Blueprint,
    weight_method: Optional[str] = 'lightgbm_gain',
    augmentations: Optional[Dict[str, np.ndarray]] = None,
    augmentation_strategy: str = 'uncapped',
    augmentation_threshold: Optional[float] = None,
    augmentation_budget: int = 20,
    aug_per_source_budget: Optional[int] = None,
    candidate_ranking: str = 'rf_importance',
    finetuned_embeddings: Optional[Dict[str, np.ndarray]] = None,
    lower_is_better: bool = False,
    objective: str = OBJECTIVE_PREDICTIVE,
    is_clf: bool = False,
) -> FittedResult:
    """Fit on full data and return concrete feature indices + optional weights.

    In similarity mode (objective='similarity_sac') the allocation is rebuilt
    with the SAC objective so the fitted feature composition matches what the
    search actually selected (otherwise the full-data refit would silently fall
    back to predictive allocation)."""
    from kirby.hybrid import create_hybrid, compute_feature_importance

    # Use only the reps in the blueprint (frozen + factory)
    reps_to_use = {k: v for k, v in base_reps.items() if k in blueprint.reps}
    if finetuned_embeddings:
        for k, v in finetuned_embeddings.items():
            if k in blueprint.reps:
                reps_to_use[k] = v

    _, feature_info = create_hybrid(
        reps_to_use, labels,
        allocation_method=blueprint.allocation_method,
        budget=blueprint.budget,
        step_size=blueprint.step_size,
        importance_method=blueprint.importance_method,
        internal_evaluator=blueprint.internal_evaluator,
        augmentations=augmentations,
        augmentation_strategy=augmentation_strategy,
        augmentation_threshold=augmentation_threshold,
        augmentation_budget=augmentation_budget,
        aug_per_source_budget=aug_per_source_budget,
        candidate_ranking=candidate_ranking,
        lower_is_better=lower_is_better,
        objective=objective,
        is_clf=is_clf,
    )

    selected_indices = {}
    total_features = 0
    for rep_name, info in feature_info.items():
        if not isinstance(info, dict) or 'selected_indices' not in info:
            continue
        selected_indices[rep_name] = info['selected_indices']
        total_features += len(info['selected_indices'])

    feature_weights = None
    if weight_method is not None:
        try:
            importance = compute_feature_importance(
                reps_to_use, labels, method=weight_method
            )
            feature_weights = {}
            for rep_name, indices in selected_indices.items():
                if rep_name in importance:
                    feature_weights[rep_name] = importance[rep_name][indices]
        except Exception:
            pass

    # Preserve augmentation info if present
    aug_info = feature_info.get('augmentation_info')

    return FittedResult(
        blueprint=blueprint,
        selected_indices=selected_indices,
        feature_weights=feature_weights,
        n_features=total_features,
        augmentation_info=aug_info,
    )


def find_best_single_rep_augmented(
    primary_rep: str,
    primary_features: np.ndarray,
    augmentation_pool: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: Optional[np.ndarray] = None,
    **kwargs,
) -> 'SearchResult':
    """Find the best augmented version of a single representation.

    Wraps find_best_hybrid for the single-rep augmentation pattern:
    one primary rep gets 100% of the budget, and all other reps are
    tested as augmentations (their best features added if they help).

    Args:
        primary_rep: Name of the primary representation.
        primary_features: Feature matrix for the primary rep (n_samples, n_features).
        augmentation_pool: Dict of {rep_name: features} to test as augmentations.
            The primary rep is automatically excluded if present.
        labels: Target values.
        groups: Group labels for CV.
        **kwargs: Passed to find_best_hybrid. Defaults:
            budgets=[primary_features.shape[1]] (use full base dimension),
            augmentation_strategy='uncapped'.

    Returns:
        SearchResult from find_best_hybrid.
    """
    base_reps = {primary_rep: primary_features}
    augmentations = {k: v for k, v in augmentation_pool.items() if k != primary_rep}

    kwargs.setdefault('budgets', [primary_features.shape[1]])
    kwargs.setdefault('augmentation_strategy', 'uncapped')

    return find_best_hybrid(
        base_reps=base_reps,
        labels=labels,
        groups=groups,
        augmentations=augmentations,
        **kwargs,
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def find_best_hybrid(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: Optional[np.ndarray] = None,
    *,
    budgets: Optional[List[int]] = None,
    models: Optional[Any] = None,
    metric: Union[str, Callable, Tuple[Callable, bool]] = 'r2',
    task_type: str = '',
    speed: str = 'full',
    allocation_methods: Optional[List[str]] = None,
    importance_methods: Optional[List[str]] = None,
    internal_evaluators: Optional[List[str]] = None,
    # Objective: 'predictive' (downstream model performance, the default) or
    # 'similarity_sac' (held-out similarity preservation — finds a DIFFERENT
    # hybrid whose geometry makes similar samples share properties). In sim mode
    # no downstream model is fit; scoring is SAC on the held-out fold.
    objective: str = OBJECTIVE_PREDICTIVE,
    n_folds: Optional[int] = None,
    test_rep_subsets: bool = False,
    return_fitted: bool = False,
    weight_method: Optional[str] = None,
    dimension_tiers: Optional[List[int]] = None,
    augmentations: Optional[Dict[str, np.ndarray]] = None,
    augmentation_strategy: str = 'uncapped',
    augmentation_threshold: Optional[float] = None,
    augmentation_budget: int = 20,
    aug_per_source_budget: Optional[int] = None,
    candidate_ranking: str = 'rf_importance',
    two_phase: bool = False,
    refinement_top_n: int = 5,
    refinement_step_sizes: Optional[List[int]] = None,
    rep_factories: Optional[Dict[str, Callable]] = None,
    secondary_metrics: Optional[List[str]] = None,
    cv_splitter=None,
    screen_threshold: Optional[float] = None,
    seed: int = 42,
    verbose: bool = True,
    cache: bool = False,
    cache_dir: Optional[str] = None,
    greedy_tolerance: float = 0.0,
    adaptive_pruning: Union[bool, float] = True,
    adaptive_pruning_schedule: Optional[Dict[str, Any]] = None,
    model_pruning_aggressiveness: Optional[Dict[str, Dict[str, Any]]] = None,
    dataset_name: Optional[str] = None,
    tier: Optional[str] = None,
    # Caller-supplied identifiers for objects that can't be hashed. Required
    # when cache=True AND the corresponding object is non-None.
    cv_splitter_id: Optional[str] = None,
    rep_factories_id: Optional[str] = None,
    condition_suffix: Optional[str] = None,
    # Namespaces every CACHE PATH (search pickle, per-rep baselines, folds,
    # hybrid-eval dedup, fine-tuned embeddings, result DB) WITHOUT touching
    # `dataset_name`, which stays the true registry name so the per-dataset
    # budget and model-skip policies still resolve.
    #
    # Required by nested_cv_evaluate: each outer fold searches a DIFFERENT
    # training subset under the same dataset name. The metas are content-checked
    # (labels_hash / rep_hashes), so a wrong result can never be served — but
    # without a namespace all outer folds map to the SAME filenames and evict
    # each other, so nothing is ever reused.
    #
    # Do NOT namespace by mangling dataset_name instead: that silently bypasses
    # _apply_budget_policy / _applied_skips, changing the budget list and model
    # set for the run.
    cache_namespace: Optional[str] = None,
    # Hyperparameter tuning. When True, tune_models() runs on the data THIS call
    # was handed, and the tuned estimators are used for the search.
    #
    # This is the ONLY sanctioned way to tune. Under nested_cv_evaluate the data
    # handed here is outer-train only, so tuning never sees the outer test fold.
    # The old pattern — tune once on the full dataset in a CLI driver, then pass
    # models=<tuned> — chose hyperparameters using the folds it then reported.
    #
    # `tune` is part of the cache key: a tuned and an untuned run are different
    # searches and must not share a cache file.
    tune: bool = False,
    tune_n_iter: int = 20,
    # Caller opt-in: keep budgets > n_samples instead of dropping them.
    # Used by c02's "hybrid at top-rep native dim" pass where we explicitly
    # want to evaluate the hybrid at a budget that matches the best single
    # rep's full dim (often > n_samples on small antibody datasets).
    # Precompute leaves this False so its cache key is unaffected.
    allow_budget_exceeds_n: bool = False,
    # Caller-supplied importance_cache (skips _run_search's per-fold importance
    # rebuild when non-empty). Pass `search.importance_cache` from a
    # previously-cached SearchResult to reuse precompute's importance.
    importance_cache: Optional[Dict[Tuple[int, str], Dict[str, np.ndarray]]] = None,
    # Label noise (kirby.noise_spec.NoiseSpec). When set, the search TRAINS on a
    # deterministic noisy copy of the labels and SCORES on the clean ones — the
    # honest setting, in which the hybrid has no more access to ground truth than
    # a base representation does. See the `tier` note below.
    label_noise: Optional[Any] = None,
) -> SearchResult:
    """
    Find the best hybrid representation configuration.

    Domain-agnostic: works with any pre-computed feature matrices.

    Args:
        base_reps: Dict of {rep_name: np.ndarray} feature matrices (all same n_samples).
        labels: Target values (n_samples,).
        groups: Group labels for GroupKFold (n_samples,). If None, uses random KFold.
        models: Model specification. None=defaults for speed tier.
            List[str] for named models. Dict[str, model] for custom models.
        metric: Evaluation metric. String ('r2','rmse','mae','accuracy','auc',
            'spearman','pearson','ci','f1_weighted')
            or callable(y_true, y_pred)->float, or (callable, lower_is_better).
        task_type: 'regression' or 'classification'. Required — no auto-detect.
        speed: Search comprehensiveness: 'fast' or 'full'.
        budgets: List of feature budgets to search over (required).
        allocation_methods: Override allocation methods for the speed tier.
        importance_methods: Override importance methods for the speed tier.
        internal_evaluators: Override internal evaluator list for the speed tier.
            Fast tier default is ['match'] (matched to downstream model).
            Medium/long tiers default to multiple evaluators.
        n_folds: Override number of CV folds for the speed tier.
        test_rep_subsets: Also test leave-one-rep-out subsets (when >=3 reps).
        return_fitted: If True, fit best config on full data and return FittedResult.
        weight_method: Importance method for feature weights (only if return_fitted=True).
        dimension_tiers: Dimension breakpoints for best_by_dimension.
            Defaults to [50, 100, 200, 500].
        augmentations: Dict of {aug_name: np.ndarray} supplementary feature arrays.
            These are small feature sets (e.g. developability, humanness) that augment
            the hybrid after main feature selection. Each is independently evaluated
            and only kept if it improves validation performance.
        augmentation_strategy: How to handle augmentations. 'uncapped' (default)
            adds features beyond budget if they help. 'capped' fills then replaces within
            budget. 'greedy_ablation' tests each rep-level. 'all' includes all. 'none' ignores.
        augmentation_threshold: Minimum improvement to keep an augmentation (default: any).
        augmentation_budget: Max features per augmentation to select (default: 20).
        rep_factories: Dict of {rep_name: factory_callable} for per-fold fine-tuned
            representations. Each factory is called with (train_idx, val_idx, train_labels)
            and returns (train_embeddings, val_embeddings). Fine-tuning happens once per
            fold (NOT per search config). The resulting embeddings are merged with frozen
            base_reps during evaluation. Requires peft + torch for actual fine-tuning
            (but the interface itself is dependency-free).
        secondary_metrics: List of metric names (e.g. ['spearman', 'rmse']) to compute
            alongside the primary metric. Only computed for baselines (not during search
            to avoid overhead). Results stored in baselines[rep]['_secondary'].
        screen_threshold: Optional pre-search rep screening threshold (0.0–1.0).
            After baselines, drops reps whose best single-rep score is more than
            this fraction away from the best rep. E.g. 0.2 means "drop reps more
            than 20% worse than the best". None (default) disables screening.
        seed: Random seed.
        verbose: Print progress.
        cache: If True, cache results to disk (filename-based, content-validated).
            Requires dataset_name and tier to be set. Loading an existing cache
            validates that the stored metadata matches the current call; a
            mismatch triggers silent recompute + rewrite.
        cache_dir: Optional override root for cache files. Defaults to
            $KIRBY_CACHE_DIR or ~/.cache/kirby. Cache subdirectories
            (searches/, baselines/, folds/) are created under this root.
        dataset_name: Required when cache=True. Part of the cache filename.
        tier: Required when cache=True. Condition name (e.g. 'general',
            'top', 'general_distance'). Part of the cache filename.
        cv_splitter_id: Caller-supplied string that identifies the
            `cv_splitter` object. Required when cache=True AND
            cv_splitter is not None. Arbitrary splitter objects cannot
            be compared for cache validation; the caller must supply a
            stable identifier (e.g., "ginkgo_hierarchical_5fold_v1"). A
            different id invalidates the cache.
        rep_factories_id: Same contract as cv_splitter_id but for
            `rep_factories`. Required when cache=True AND rep_factories
            is not None. Use a per-factory version string
            (e.g., "lora_esm2_v3_2026_04_10").
        condition_suffix: Optional extra label appended to the cache
            filename. Use for synthetic variations on a registered
            dataset (e.g. "n500" for a subsample, "loo_additive" for a
            leave-one-out split, "asymmetric" for a synthetic silo).
            Registry-backed metric resolution still uses dataset_name;
            only the cache key differs. Distinct suffixes never
            collide.

    Returns:
        SearchResult with best configs, baselines, and optional fitted result.
    """
    # --- Validate inputs ---
    if budgets is None:
        import warnings
        warnings.warn(
            "budgets not specified — using default [50, 150, 300]. "
            "Pass budgets= explicitly for reproducible results.",
            stacklevel=2,
        )
        budgets = [50, 150, 300]

    # Per-dataset budget cap (e.g. drop 1000/2000 on large datasets). Applied
    # here so precompute and experiment-time calls share the capped list and
    # cache keys stay aligned.
    budgets = _apply_budget_policy(dataset_name, budgets)

    labels = np.asarray(labels).ravel()

    if not base_reps:
        raise ValueError("base_reps cannot be empty")
    total_reps = len(base_reps) + (len(rep_factories) if rep_factories else 0)
    n_aug = len(augmentations) if augmentations else 0
    if total_reps < 2 and n_aug == 0:
        raise ValueError("Need at least 2 representations for a hybrid "
                         "(or 1 base rep + augmentations)")

    n_samples_list = [X.shape[0] for X in base_reps.values()]
    if len(set(n_samples_list)) > 1:
        raise ValueError(f"All representations must have same n_samples. Got: {n_samples_list}")
    n_samples = n_samples_list[0]

    if len(labels) != n_samples:
        raise ValueError(f"labels length ({len(labels)}) != n_samples ({n_samples})")

    # --- Normalize augmentation_strategy when there are no augmentations ---
    # With augmentations=None the aug logic in hybrid.create_hybrid never fires,
    # so 'uncapped' and 'none' are identical. But the strategy is in the cache
    # key, so normalize to 'none' to avoid a meta mismatch — otherwise a no-aug
    # caller that left the 'uncapped' default would write a no-aug result into
    # the same filename space as a real uncapped run with different aug_names.
    # Behaviour-preserving: only the cache key changes; scores are unaffected
    # because no augmentation is ever applied when augmentations is falsy.
    if not augmentations and augmentation_strategy != 'none':
        if verbose:
            print(f"  No augmentations supplied — normalizing "
                  f"augmentation_strategy={augmentation_strategy!r} -> 'none' "
                  f"for cache key.")
        augmentation_strategy = 'none'

    # --- Budget filter (before cache key) ---
    # Rule: drop any caller-supplied budget > n_samples. If anything was
    # dropped, also append n_samples as an at-n boundary test point so
    # the hybrid is evaluated at the largest budget that still fits the
    # dataset. Without this, _evaluate_single's baselines use full native
    # rep dim while the hybrid stops at the largest pre-existing budget
    # below n — unfair on small-n datasets.
    #
    # When allow_budget_exceeds_n=True, the caller has explicitly asked
    # for budgets that may exceed n_samples (e.g. evaluating a hybrid at
    # a base rep's full native dim on a small-n dataset). Pass through
    # unchanged.
    if allow_budget_exceeds_n:
        filtered = sorted(set(budgets))
    else:
        filtered = [b for b in budgets if b <= n_samples]
        added_at_n = False
        if any(b > n_samples for b in budgets) and n_samples not in filtered:
            filtered.append(n_samples)
            added_at_n = True
        if not filtered:
            filtered = [n_samples]
        filtered = sorted(set(filtered))
        if filtered != list(budgets) and verbose:
            dropped = [b for b in budgets if b > n_samples]
            parts = []
            if dropped:
                parts.append(f"Dropped budgets > n={n_samples}: {dropped}")
            if added_at_n:
                parts.append(f"added n={n_samples} as at-n boundary")
            if parts:
                print("; ".join(parts))
    budgets = filtered

    # --- Normalize parameters for cache key alignment ---
    # Resolve defaults from SPEED_TIERS so that None and explicit tier
    # values produce identical cache keys.
    _tier = SPEED_TIERS[speed]
    _norm_alloc = allocation_methods if allocation_methods is not None else _tier['allocation_methods']
    _norm_imp = importance_methods if importance_methods is not None else _tier['importance_methods']
    _norm_eval = internal_evaluators if internal_evaluators is not None else _tier['internal_evaluators']
    _norm_folds = n_folds if n_folds is not None else _tier['n_folds']

    # Normalize models to sorted list of names. Apply the same _max_samples
    # and empirical skips that _resolve_models uses, so cached metadata
    # matches the actual model list this run will produce.
    _skip_set = _applied_skips(dataset_name, augmentation_strategy)

    # Cache-path dataset key. Every policy lookup above this line uses the TRUE
    # dataset_name; every cache path below uses _ds_cache. See cache_namespace.
    _ds_cache = dataset_name
    if dataset_name and cache_namespace:
        _ds_cache = f'{dataset_name}__{_sanitize_filename_part(cache_namespace)}'

    def _passes_max_samples(name, spec):
        max_s = spec.get('_max_samples')
        # Keep this in lockstep with _check_max_samples so the cache-key model
        # list matches the models actually run (otherwise a stale cache built
        # with SVM present could be reused for an uncapped large-n cell that now
        # skips SVM).
        strat_cap = _STRATEGY_MAX_SAMPLES.get((name, augmentation_strategy))
        if strat_cap is not None:
            max_s = strat_cap if max_s is None else min(max_s, strat_cap)
        return not (max_s is not None and n_samples is not None
                    and n_samples > max_s)

    if models is None:
        _max_tier = _TIER_ORDER.get(speed, 1)
        _norm_models = sorted(
            name for name, spec in DEFAULT_MODELS.items()
            if _TIER_ORDER.get(spec['tier'], 0) <= _max_tier
            and _passes_max_samples(name, spec)
            and name not in _skip_set
        )
    elif isinstance(models, dict):
        _norm_models = sorted(
            n for n in models.keys() if n not in _skip_set
        )
    elif isinstance(models, list):
        _norm_models = sorted(
            n for n in models
            if n in DEFAULT_MODELS
            and _passes_max_samples(n, DEFAULT_MODELS[n])
            and n not in _skip_set
        )
    else:
        _norm_models = None  # single model instance

    # Normalize adaptive_pruning to float threshold
    if adaptive_pruning is True:
        _norm_pruning = 0.15
    elif not adaptive_pruning:
        _norm_pruning = 0.0
    else:
        _norm_pruning = float(adaptive_pruning)

    # --- Validate task_type (required, no auto-detect) ---
    if task_type not in ('classification', 'regression'):
        raise ValueError(
            f"find_best_hybrid: task_type={task_type!r}. "
            f"Must be 'classification' or 'regression' — explicit, no auto."
        )
    _task_type_for_meta = task_type

    # --- Validate objective; in similarity mode force the SAC metric so the
    # stored metric_name and comparison direction are always consistent
    # (SAC is higher-is-better; no downstream model is fit). ---
    if objective not in _VALID_OBJECTIVES:
        raise ValueError(
            f"find_best_hybrid: objective={objective!r}. "
            f"Must be one of {_VALID_OBJECTIVES}."
        )
    if objective == OBJECTIVE_SIMILARITY_SAC and metric != 'sac':
        if verbose:
            print(f"[objective=similarity_sac] overriding metric={metric!r} -> 'sac' "
                  f"(geometry objective, no downstream model fit).")
        metric = 'sac'

    # Metric name as it appears in cached metadata. Callables can't be
    # filename-encoded, so when metric is a callable the caller is expected
    # to also use cache=False (caching a custom metric has no safe key).
    _metric_name_for_meta = metric if isinstance(metric, str) else 'custom'

    # --- Label noise -------------------------------------------------------
    # ONE deterministic noisy copy of the label column, drawn here and reused by
    # every fold (kirby.noise_spec guarantees reproducibility — a fresh injector
    # per call, because NoiseInject's RNG advances on each inject()).
    #
    # The noise condition is carried by `tier`, and it MUST be: label noise is
    # applied to TRAINING rows only, so the global label array is byte-identical
    # to the clean run and `labels_hash` — which is in the SQLite identity — cannot
    # tell the two apart. Without a distinct tier, a "noisy" search would read the
    # CLEAN rows straight back out of the result DB and never recompute. `tier` is
    # IDENTITY_COLS[0] and is already used exactly this way by general_distance /
    # top_distance, so this needs no schema change.
    _noise_id: Optional[str] = None
    _train_labels: Optional[np.ndarray] = None
    if label_noise is not None:
        if rep_factories:
            # _precompute_fold_reps fine-tunes embeddings on labels[train_idx] and
            # does NOT take the noisy column. Running it under label noise would
            # fine-tune the representations on CLEAN labels and hand the hybrid
            # exactly the ground-truth advantage this parameter exists to remove.
            # Refuse rather than leak silently.
            raise ValueError(
                "find_best_hybrid: label_noise is not supported together with "
                "rep_factories — fine-tuned embeddings would still be trained on "
                "the clean labels, reintroducing the leak. Precompute the "
                "fine-tuned reps separately and pass them as base_reps.")
        from kirby.noise_spec import make_noisy_labels as _mk_noisy
        _is_clf_noise = (task_type == 'classification'
                         if task_type in ('classification', 'regression')
                         else _detect_task_type(labels) == 'classification')
        _train_labels = _mk_noisy(labels, label_noise, is_clf=_is_clf_noise)
        _noise_id = label_noise.spec_id
        if tier:
            tier = label_noise.tier_name(tier)
        if verbose:
            print(f"[label_noise] {_noise_id} — training on noisy labels, "
                  f"scoring on clean. tier={tier!r}")

    # --- Check cache (filename-based, content-validated) ---
    cache_path = None
    expected_search_meta: Optional[Dict[str, Any]] = None
    result_db_identity: Optional[Dict[str, Any]] = None
    result_db_config_const: Optional[Dict[str, Any]] = None
    result_db_rep_pool: Optional[str] = None
    if cache:
        # Required when cache=True. Raise loudly instead of silently
        # falling back to a hash-based key — we don't have one anymore.
        if not dataset_name:
            raise ValueError(
                "find_best_hybrid(cache=True) requires dataset_name=... "
                "and tier=... (filename-based cache has no fallback key)"
            )
        if not tier:
            raise ValueError(
                "find_best_hybrid(cache=True) requires tier=... "
                "(e.g. 'general', 'top', 'general_distance')"
            )
        # Custom metric callables cannot be cache-validated safely.
        if not isinstance(metric, str):
            raise ValueError(
                "find_best_hybrid(cache=True) requires a string metric "
                "(callable metrics cannot be filename-encoded)"
            )
        # Identity must be caller-supplied for objects we can't hash.
        if cv_splitter is not None and not cv_splitter_id:
            raise ValueError(
                "find_best_hybrid(cache=True) with a custom cv_splitter "
                "requires cv_splitter_id=... A stable identifier is the "
                "caller's responsibility — arbitrary splitter objects "
                "cannot be compared for cache validation."
            )
        if rep_factories and not rep_factories_id:
            raise ValueError(
                "find_best_hybrid(cache=True) with rep_factories requires "
                "rep_factories_id=... A stable per-version identifier is "
                "the caller's responsibility — factory callables cannot "
                "be introspected safely."
            )

        cache_path = _search_cache_path(
            _ds_cache, tier, augmentation_strategy, speed,
            condition_suffix=condition_suffix)
        if cache_dir is not None:
            # Honor legacy cache_dir override by rerooting.
            cache_path = Path(cache_dir) / cache_path.name

        expected_search_meta = _make_search_meta(
            tune=tune,
            tune_n_iter=tune_n_iter,
            dataset_name=_ds_cache,
            tier=tier,
            aug_strategy=augmentation_strategy,
            speed=speed,
            base_reps=base_reps,
            labels=labels,
            groups=groups,
            model_names=_norm_models,
            metric_name=_metric_name_for_meta,
            task_type=_task_type_for_meta,
            n_folds=_norm_folds,
            seed=seed,
            budgets=budgets,
            augmentations=augmentations,
            augmentation_strategy=augmentation_strategy,
            internal_evaluators=_norm_eval,
            allocation_methods=_norm_alloc,
            importance_methods=_norm_imp,
            adaptive_pruning=_norm_pruning,
            augmentation_threshold=augmentation_threshold,
            augmentation_budget=augmentation_budget,
            aug_per_source_budget=aug_per_source_budget,
            candidate_ranking=candidate_ranking,
            greedy_tolerance=greedy_tolerance,
            screen_threshold=screen_threshold,
            two_phase=two_phase,
            refinement_top_n=refinement_top_n,
            refinement_step_sizes=refinement_step_sizes,
            test_rep_subsets=test_rep_subsets,
            dimension_tiers=dimension_tiers,
            weight_method=weight_method,
            step_size=5,  # currently hardcoded in _evaluate_hybrid_config
            secondary_metrics=secondary_metrics,
            cv_splitter_id=cv_splitter_id,
            rep_factories_id=rep_factories_id,
            pruning_schedule=adaptive_pruning_schedule,
            model_pruning_overrides=model_pruning_aggressiveness,
        )
        # condition_suffix is in the filename but also goes in meta so
        # validation catches accidental collisions. Explicit None stored.
        expected_search_meta['condition_suffix'] = condition_suffix

        # --- Per-dataset SQLite result store --------------------------------
        # The DB is the only search-result store. Values come from
        # expected_search_meta so they match its deterministic, type-coerced
        # form; aug_names/rep_pool use repr(tuple(...)) so two name sets can't
        # collide via a comma join.
        _esm = expected_search_meta
        # KEY: data + fold split + tier/speed (the pool + grouping handle). Never
        # the sweep (models/rankers/budgets) nor the aug recipe — those are
        # per-row config, so reuse on a changed list is a SELECT, not a rebuild.
        result_db_identity = {
            'tier': tier,
            'speed': speed,
            'labels_hash': _esm.get('labels_hash'),
            'groups_hash': _esm.get('groups_hash'),
            'n_folds': _esm.get('n_folds'),
            'seed': _esm.get('seed'),
            'cv_splitter_id': _esm.get('cv_splitter_id'),
        }
        # Per-row config that is CONSTANT for this invocation: the augmentation
        # recipe + the greedy/candidate knobs. Constant across rows here, but
        # part of each row's config key, so a different aug strategy (or
        # candidate ranking) coexists in the same DB instead of overwriting.
        result_db_config_const = {
            'aug_strategy': augmentation_strategy,
            'aug_names': repr(tuple(_esm.get('aug_names') or ())),
            'aug_threshold': _esm.get('augmentation_threshold'),
            'aug_budget': _esm.get('augmentation_budget'),
            'candidate_ranking': _esm.get('candidate_ranking'),
            'greedy_tolerance': _esm.get('greedy_tolerance'),
        }
        # Recorded on every row as data (NOT a key): the rep pool this search ran
        # over, so a stored row is self-describing. repr(tuple) avoids the comma
        # collision of a join.
        result_db_rep_pool = repr(tuple(_esm.get('rep_names') or ()))

    if groups is not None:
        groups = np.asarray(groups).ravel()
        if len(groups) != n_samples:
            raise ValueError(f"groups length ({len(groups)}) != n_samples ({n_samples})")

    if speed not in SPEED_TIERS:
        raise ValueError(f"speed must be one of {list(SPEED_TIERS.keys())}. Got '{speed}'")

    if dimension_tiers is None:
        dimension_tiers = [50, 100, 200, 500]

    # --- Validate augmentations ---
    if augmentations is not None:
        for aug_name, X_aug in augmentations.items():
            if X_aug.shape[0] != n_samples:
                raise ValueError(
                    f"Augmentation '{aug_name}' has {X_aug.shape[0]} samples, "
                    f"expected {n_samples}"
                )
        if verbose:
            aug_dims = {k: v.shape[1] for k, v in augmentations.items()}
            print(f"Augmentations: {aug_dims} (strategy={augmentation_strategy})")

    # --- Infer n_folds from cv_splitter ---
    if cv_splitter is not None and n_folds is None:
        n_folds = cv_splitter.get_n_splits()

    # --- Auto-detect task type ---
        if verbose:
            print(f"Auto-detected task type: {task_type}")

    # --- Resolve metric ---
    metric_fn, lower_is_better = _resolve_metric(metric)

    # --- Resolve secondary metrics ---
    resolved_secondary = None
    if secondary_metrics:
        resolved_secondary = {name: _resolve_metric(name)[0] for name in secondary_metrics}
        if verbose:
            print(f"Secondary metrics: {secondary_metrics}")

    # --- Resolve models ---
    resolved_models = _resolve_models(models, task_type, speed,
                                       n_samples=n_samples,
                                       dataset_name=dataset_name,
                                       strategy=augmentation_strategy)
    if verbose:
        print(f"Models: {list(resolved_models.keys())}")
        print(f"Representations: {list(base_reps.keys())} "
              f"(dims: {[X.shape[1] for X in base_reps.values()]})")
        if rep_factories:
            print(f"Fine-tuning factories: {list(rep_factories.keys())}")

    # --- Hyperparameter tuning (on THIS call's data only) ---
    # Under nested_cv_evaluate, base_reps/labels here are outer-train only, so
    # the tuning CV never touches the outer test fold. Tuning at the CLI-driver
    # level on the full dataset (the old --tune path) chose hyperparameters
    # using the folds it then reported — that is why this lives here now.
    if tune:
        if verbose:
            print("Tuning hyperparameters (RandomizedSearchCV) on this call's "
                  f"data only (n={n_samples})...")
        _tuned = tune_models(
            base_reps=base_reps,
            labels=labels,
            groups=groups,
            task_type=task_type,
            speed=speed,
            n_iter=tune_n_iter,
            seed=seed,
        )
        # Keep only the models this run actually resolved, so the tuned set
        # cannot silently widen or narrow the model list behind the cache key.
        for _mn, _mdl in (_tuned or {}).items():
            if _mn in resolved_models:
                resolved_models[_mn] = _mdl
        if verbose:
            print(f"  Tuned: {sorted(set(_tuned or {}) & set(resolved_models))}")

    # --- Pre-compute fine-tuned reps ---
    fold_reps = {}
    precomputed_folds = []
    if rep_factories:
        actual_folds_for_ft = n_folds if n_folds is not None else SPEED_TIERS[speed]['n_folds']
        # Per-factory ids for the FT embeddings cache: compose the single
        # caller-supplied rep_factories_id with each factory's name.
        # Different rep_factories_id values invalidate independently.
        _factory_ids = None
        if rep_factories_id:
            _factory_ids = {
                name: f'{rep_factories_id}__{name}'
                for name in rep_factories.keys()
            }
        fold_reps, precomputed_folds = _precompute_fold_reps(
            rep_factories, labels, groups, actual_folds_for_ft, seed,
            cv_splitter=cv_splitter, verbose=verbose,
            dataset_name=_ds_cache,
            factory_ids=_factory_ids,
            cv_splitter_id=cv_splitter_id,
            task_type=task_type,
        )
    else:
        fold_reps = None
        precomputed_folds = None

    # --- Evaluate baselines ---
    baseline_folds = n_folds if n_folds is not None else SPEED_TIERS[speed]['n_folds']
    baselines, baseline_fold_scores = _evaluate_baselines(
        base_reps, labels, resolved_models, groups,
        baseline_folds, metric_fn, seed, verbose,
        fold_reps=fold_reps,
        precomputed_folds=precomputed_folds,
        secondary_metric_fns=resolved_secondary,
        cv_splitter=cv_splitter,
        # Pass through for per-rep baseline caching. _evaluate_baselines
        # will only cache if these are non-None and no secondary metrics
        # are requested (safe fallback to uncached otherwise).
        dataset_name=_ds_cache,
        metric_name=_metric_name_for_meta,
        task_type=task_type,
        objective=objective,
        # Baselines MUST train on the same noisy labels as the hybrid candidates.
        # Leaving them clean would not remove the ground-truth advantage — it would
        # hand it to the base representations instead.
        train_labels=_train_labels,
        noise_id=_noise_id,
    )

    # --- Pre-search rep screening ---
    if screen_threshold is not None:
        base_reps, fold_reps = _screen_reps(
            base_reps, baselines, lower_is_better, screen_threshold,
            fold_reps=fold_reps, verbose=verbose,
        )

    # --- Run search ---
    # Resolve metric name string for alloc cache lookup
    _metric_name_str = metric if isinstance(metric, str) else None

    # Derive checkpoint path from cache path (if caching enabled)
    _ckpt_path = None
    if cache and cache_path is not None:
        _ckpt_path = cache_path.with_suffix('.ckpt.pkl')

    all_results, _imp_cache, _alloc_cache = _run_search(
        base_reps, labels, groups, speed, resolved_models,
        metric_fn,
        budgets=budgets,
        allocation_methods=allocation_methods,
        importance_methods=importance_methods,
        internal_evaluators=internal_evaluators,
        objective=objective,
        n_folds=n_folds,
        test_rep_subsets=test_rep_subsets,
        seed=seed,
        verbose=verbose,
        augmentations=augmentations,
        augmentation_strategy=augmentation_strategy,
        augmentation_threshold=augmentation_threshold,
        augmentation_budget=augmentation_budget,
        aug_per_source_budget=aug_per_source_budget,
        candidate_ranking=candidate_ranking,
        lower_is_better=lower_is_better,
        fold_reps=fold_reps,
        precomputed_folds=precomputed_folds,
        cv_splitter=cv_splitter,
        greedy_tolerance=greedy_tolerance,
        metric_name=_metric_name_str,
        # Pass the NORMALISED float, not the raw param. The inline
        # normaliser inside _run_search tripped on bool-is-int for True
        # (would prune 100% of strategies after fold 0, so only fold-0
        # scores reached best_by_model).
        adaptive_pruning=_norm_pruning,
        adaptive_pruning_schedule=adaptive_pruning_schedule,
        model_pruning_aggressiveness=model_pruning_aggressiveness,
        checkpoint_path=_ckpt_path,
        # Validate checkpoint payload against the same meta used for the
        # final cache file. Resume only when parameters match.
        checkpoint_meta=expected_search_meta,
        # hybrid_eval cache context. Activates the persistent
        # per-(subset, config, fold, model) cache for top_k allocations
        # so general-tier and top-tier runs share the identical-subset
        # computation. Gated on dataset_name being set (i.e. caller
        # wants persistent caching at all).
        dataset_name=_ds_cache,
        task_type=_task_type_for_meta,
        baselines=baselines,
        importance_cache_override=importance_cache,
        # Per-dataset SQLite result store. None when caching is off
        # or KIRBY_RESULT_DB=0 (then _run_search uses no durable store).
        result_db_identity=(result_db_identity
                            if (cache and cache_path is not None) else None),
        result_db_dataset=(_ds_cache
                           if (cache and cache_path is not None) else None),
        result_db_cache_dir=cache_dir,
        result_db_config_const=(result_db_config_const
                                if (cache and cache_path is not None) else None),
        result_db_rep_pool=(result_db_rep_pool
                            if (cache and cache_path is not None) else None),
        train_labels=_train_labels,
        noise_id=_noise_id,
    )

    # --- Two-phase refinement ---
    if two_phase and all_results:
        actual_folds_for_refine = n_folds if n_folds is not None else SPEED_TIERS[speed]['n_folds']
        refinement_results = _run_refinement_phase(
            phase1_results=all_results,
            base_reps=base_reps,
            labels=labels,
            groups=groups,
            models=resolved_models,
            metric_fn=metric_fn,
            n_folds=actual_folds_for_refine,
            seed=seed,
            top_n=refinement_top_n,
            step_sizes=refinement_step_sizes,
            verbose=verbose,
            augmentations=augmentations,
            augmentation_strategy=augmentation_strategy,
            augmentation_threshold=augmentation_threshold,
            augmentation_budget=augmentation_budget,
            aug_per_source_budget=aug_per_source_budget,
            candidate_ranking=candidate_ranking,
            lower_is_better=lower_is_better,
            fold_reps=fold_reps,
            precomputed_folds=precomputed_folds,
            cv_splitter=cv_splitter,
            metric_name=_metric_name_str,
        )
        all_results.extend(refinement_results)
        # Persist refinement rows to the DB too (they run AFTER _run_search has
        # closed its handle). Each carries its own step_size — a config column —
        # so they never collide with their phase-1 siblings. Finite-score guard
        # mirrors the main loop (NaN configs re-attempt next run).
        if result_db_identity is not None and refinement_results:
            from .result_cache import ResultCache as _ResultCache
            _ref_rows = [
                _result_to_db_row(result_db_identity, _r, _metric_name_str,
                                  result_db_config_const, result_db_rep_pool)
                for _r in refinement_results
                if _r.get('score_mean') is not None
                and np.isfinite(_r.get('score_mean'))
            ]
            if _ref_rows:
                with _ResultCache(_ds_cache, cache_dir=cache_dir) as _ref_db:
                    _ref_db.upsert_rows(_ref_rows)

    # --- Aggregate ---
    agg = _aggregate_results(all_results, baselines, lower_is_better, dimension_tiers)

    # --- Extract full-data factory embeddings ---
    finetuned_embeddings = None
    if rep_factories:
        finetuned_embeddings = _extract_full_embeddings(
            rep_factories, labels, verbose
        )

    # --- Per-model fitted results ---
    fitted_results = {}
    if return_fitted and agg['best_by_model']:
        if verbose:
            print(f"\nFitting best configs on full data ({len(agg['best_by_model'])} models)...")
        for model_name, best_r in agg['best_by_model'].items():
            try:
                fitted_results[model_name] = _compute_fitted_result(
                    base_reps, labels, best_r['blueprint'], weight_method,
                    augmentations=augmentations,
                    augmentation_strategy=augmentation_strategy,
                    augmentation_threshold=augmentation_threshold,
                    augmentation_budget=augmentation_budget,
                    aug_per_source_budget=aug_per_source_budget,
                    candidate_ranking=candidate_ranking,
                    finetuned_embeddings=finetuned_embeddings,
                    lower_is_better=lower_is_better,
                    objective=objective,
                    is_clf=(task_type == 'classification'),
                )
            except Exception as e:
                if verbose:
                    print(f"  WARNING: fitted result for {model_name} failed: {e}")


    # --- Resolve metric name for display ---
    metric_name = metric if isinstance(metric, str) else 'custom'
    actual_folds = n_folds if n_folds is not None else SPEED_TIERS[speed]['n_folds']

    # --- Build per-rep best baseline scores (for Blueprint.rep_scores) ---
    # For each rep, take its best score across models (direction-aware).
    # Stored in "higher is better" convention (negated for lower-is-better
    # metrics) to match what _evaluate_allocation returns.  This ensures
    # that apply_blueprint → precomputed_alloc_scores → top_k sorts
    # correctly regardless of metric direction.
    _rep_best = {}
    for rep_name, model_scores in baselines.items():
        vals = [v for k, v in model_scores.items()
                if k not in ('dimension', '_secondary')
                and isinstance(v, (int, float)) and np.isfinite(v)]
        if vals:
            raw_best = float(min(vals) if lower_is_better else max(vals))
            _rep_best[rep_name] = -raw_best if lower_is_better else raw_best

    # --- Populate Blueprint reproducibility fields ---
    for r in all_results:
        bp = r['blueprint']
        bp.speed = speed
        bp.n_folds = actual_folds
        bp.seed = seed
        bp.metric = metric_name
        bp.task_type = task_type
        bp.lower_is_better = lower_is_better
        # Attach per-rep scores for the reps in this blueprint
        bp.rep_scores = {rn: _rep_best[rn] for rn in bp.reps if rn in _rep_best}

    search_result = SearchResult(
        best_by_model=agg['best_by_model'],
        best_by_dimension=agg['best_by_dimension'],
        best_by_budget=agg['best_by_budget'],
        best_by_model_budget=agg['best_by_model_budget'],
        performance_matrix=agg['performance_matrix'],
        baselines=baselines,
        all_results=all_results,
        metric_name=metric_name,
        lower_is_better=lower_is_better,
        fitted_results=fitted_results,
        finetuned_embeddings=finetuned_embeddings,
        dataset_name=dataset_name,
        importance_cache=_imp_cache,
        alloc_cache=_alloc_cache,
        baseline_fold_scores=baseline_fold_scores,
    )

    # --- Print per-model-budget summary (primary selection unit) ---
    if verbose and agg['best_by_model_budget']:
        model_names = sorted(agg['best_by_model_budget'].keys())
        budget_values = sorted(set(
            bgt for mb in agg['best_by_model_budget'].values() for bgt in mb.keys()))
        print(f"\nBest features per model x budget ({metric_name}):")
        header = f"  {'model':<12s}" + "".join(f"  b={b:<5d}" for b in budget_values)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for mname in model_names:
            parts = [f"  {mname:<12s}"]
            for bgt in budget_values:
                r = agg['best_by_model_budget'][mname].get(bgt)
                if r:
                    parts.append(f"  {r['score_mean']:.4f}")
                else:
                    parts.append(f"  {'—':>6s}")
            print("".join(parts))

    # --- Persistence ---
    # The per-config rows were already written to the per-dataset SQLite store
    # inside _run_search (including the two-phase refinement rows below). There
    # is no monolithic-pickle save.

    # End-of-run cache observability. Always print so cache hit/miss is visible
    # in SLURM logs.
    _s = _CACHE_STATS
    if any(_s.values()):
        print(f"[cache] {_cache_stats_summary()}")

    return search_result


# =============================================================================
# PUBLIC API: APPLY A PUBLISHED BLUEPRINT
# =============================================================================

def apply_blueprint(
    base_reps: Dict[str, np.ndarray],
    blueprint: Union[Blueprint, dict],
    labels: Optional[np.ndarray] = None,
    augmentations: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, dict]:
    """Create a hybrid representation from a published blueprint.

    This is the user-facing "product" entry point.  Given a blueprint
    (from a search, JSON lookup, or the registry), produce the hybrid
    feature matrix.  No search is performed — the blueprint specifies
    exactly which allocation, importance method, evaluator, and budget
    to use.

    For top_k blueprints that carry rep_scores, the per-rep evaluation
    is skipped entirely (scores are injected from the blueprint).

    Args:
        base_reps: {rep_name: np.ndarray} feature matrices.
            Must contain at least the reps listed in blueprint.reps.
        blueprint: A Blueprint instance or dict (from Blueprint.to_dict() / JSON).
        labels: Target values.  Required for importance-based feature selection.
            If None AND the blueprint carries rep_scores, a variance-based
            fallback is used (suitable for inference on unlabelled data).
        augmentations: Optional augmentation reps.

    Returns:
        (X_hybrid, feature_info) — same contract as create_hybrid().
    """
    from kirby.hybrid import create_hybrid

    if isinstance(blueprint, dict):
        blueprint = Blueprint.from_dict(blueprint)

    # Filter base_reps to only the reps in the blueprint
    bp_reps = {k: base_reps[k] for k in blueprint.reps if k in base_reps}
    missing = set(blueprint.reps) - set(bp_reps)
    if missing:
        raise ValueError(
            f"base_reps is missing representations required by the blueprint: "
            f"{sorted(missing)}")

    # If labels are not provided, use variance-based feature selection.
    # create_hybrid needs labels for its internal evaluator, so we synthesise
    # dummy labels (variance selection doesn't depend on them when
    # precomputed_alloc_scores are injected).
    n_samples = next(iter(bp_reps.values())).shape[0]
    if labels is None:
        labels = np.zeros(n_samples)
        # Force variance-based importance (no model fitting)
        importance_method = 'lightgbm_gain'  # will be overridden by precomputed
    else:
        importance_method = blueprint.importance_method

    kwargs = dict(
        base_reps=bp_reps,
        labels=labels,
        allocation_method=blueprint.allocation_method,
        budget=blueprint.budget,
        step_size=blueprint.step_size,
        importance_method=importance_method,
        internal_evaluator=blueprint.internal_evaluator,
        augmentations=augmentations,
        augmentation_strategy=blueprint.augmentation_strategy,
    )

    # Inject stored rep_scores to skip per-rep model evaluation
    if blueprint.rep_scores is not None:
        kwargs['precomputed_alloc_scores'] = blueprint.rep_scores

    return create_hybrid(**kwargs)


def get_best_hybrid(
    dataset_name: str,
    model: str = 'rf',
    budget: Optional[int] = None,
    tier: str = 'general',
    aug_condition: str = 'uncapped',
    registry_path: Optional[str] = None,
) -> dict:
    """Look up the best published hybrid config for a dataset.

    Reads from the hybrid registry (populated by a precompute pass)
    and returns a self-contained dict with the blueprint, scores, and
    metric direction.

    This is the "product" lookup — one line to get the recommended hybrid
    configuration for any dataset KIRBy has been run on.

    Args:
        dataset_name: Dataset name as used in DATASET_REPS (e.g. 'caco2').
        model: Downstream model name (e.g. 'rf', 'lightgbm', 'knn').
        budget: Specific budget (None = best across all budgets for this model).
        tier: Rep tier ('general' or 'top').
        aug_condition: Augmentation condition ('none', 'uncapped', 'capped').
        registry_path: Override path to hybrid_registry.json.

    Returns:
        Dict with keys:
          - 'blueprint': Blueprint instance (ready for apply_blueprint())
          - 'score_mean': float
          - 'score_std': float
          - 'metric': str (e.g. 'rmse', 'r2', 'ci')
          - 'lower_is_better': bool
          - 'budget': int
          - 'model': str
          - 'dataset': str
          - 'n_features': int

    Raises:
        KeyError if dataset/model/budget not found in registry.
    """
    import json as _json

    if registry_path is None:
        # Use the single _cache_base() helper instead of re-reading
        # KIRBY_CACHE_DIR inline, so this site can never drift from the rest
        # of the module's cache-root resolution. Behaviour-identical (same
        # env var, same ~/.cache/kirby default).
        reg_path = _cache_base() / 'hybrid_registry.json'
    else:
        reg_path = Path(registry_path)

    if not reg_path.exists():
        raise FileNotFoundError(
            f"Hybrid registry not found at {reg_path}. "
            f"Build the hybrid registry first.")

    registry = _json.loads(reg_path.read_text())

    if dataset_name not in registry:
        available = sorted(registry.keys())
        raise KeyError(
            f"Dataset '{dataset_name}' not in registry. "
            f"Available: {available}")

    ds_entry = registry[dataset_name]
    metric_name = ds_entry['metric']
    lower_is_better = ds_entry['lower_is_better']

    config_key = f"{tier}/{aug_condition}"
    if config_key not in ds_entry.get('configs', {}):
        available = sorted(ds_entry.get('configs', {}).keys())
        raise KeyError(
            f"Config '{config_key}' not found for '{dataset_name}'. "
            f"Available: {available}")

    config_entry = ds_entry['configs'][config_key]

    if model not in config_entry:
        available = sorted(config_entry.keys())
        raise KeyError(
            f"Model '{model}' not found for '{dataset_name}/{config_key}'. "
            f"Available: {available}")

    model_entry = config_entry[model]

    if budget is not None:
        bgt_key = str(budget)
        if bgt_key not in model_entry:
            available = sorted(model_entry.keys(), key=int)
            raise KeyError(
                f"Budget {budget} not found for "
                f"'{dataset_name}/{config_key}/{model}'. "
                f"Available: {available}")
        best = model_entry[bgt_key]
    else:
        # Pick the best budget (direction-aware)
        candidates = list(model_entry.items())
        if lower_is_better:
            bgt_key, best = min(candidates, key=lambda kv: kv[1]['score_mean'])
        else:
            bgt_key, best = max(candidates, key=lambda kv: kv[1]['score_mean'])

    bp_dict = best.get('blueprint')
    if bp_dict is None:
        raise ValueError(
            f"No blueprint stored for '{dataset_name}/{config_key}/{model}/"
            f"{bgt_key}'. Cache may need rebuilding.")

    bp = Blueprint.from_dict(bp_dict)

    return {
        'blueprint': bp,
        'score_mean': best['score_mean'],
        'score_std': best.get('score_std', 0.0),
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'budget': int(bgt_key) if budget is None else budget,
        'model': model,
        'dataset': dataset_name,
        'n_features': best.get('n_features', 0),
    }


# =============================================================================
# NESTED CROSS-VALIDATION (HONEST EVALUATION)
# =============================================================================

@dataclass
class NestedCVResult:
    """Honest evaluation via nested cross-validation.

    The ONLY sanctioned way to compare a hybrid against base representations.

    Structure (this is the whole point — read it):

        outer group-aware CV over the dataset
        for each outer fold i:
            outer_test  = fold i                (the search NEVER sees this)
            outer_train = every other fold
            inner: find_best_hybrid(outer_train)   <- its own inner CV lives here
            hybrid arm:   blueprint picked on outer_train -> rebuilt on
                          outer_train -> scored on outer_test
            baseline arm: best single rep picked on outer_train (by the inner
                          search's own baselines) -> refit on outer_train ->
                          scored on outer_test
            every individual rep is ALSO scored on outer_test, for per-rep
            reporting (baselines are individual reps; never a mean of reps)
        report the mean across outer folds

    Both arms pay the identical held-out tax: each picks its winner using only
    outer_train, and is judged on an outer_test it never touched. That is what
    makes the delta meaningful. `bias` (search_mean - honest_mean) is the
    selection-overfitting the old whole-dataset path was silently reporting.
    """
    # {model: {'hybrid_fold_scores', 'best_single_fold_scores', 'per_fold_delta',
    #          'hybrid_mean', 'hybrid_std', 'best_single_mean', 'best_single_std',
    #          'delta_mean_of_folds', 'n_paired_folds',
    #          'blueprints_per_fold', 'best_single_reps_per_fold',
    #          'search_mean', 'bias'}}
    by_model: Dict[str, dict]
    # EVERY rep scored individually on the outer test folds:
    # {model: {rep: {'fold_scores': [...], 'mean': float, 'std': float}}}
    baselines_by_model: Dict[str, Dict[str, dict]]
    metric_name: str
    lower_is_better: bool
    n_outer_folds: int
    # Per-(model, budget) honest scores, for experiments that report by budget.
    # {model: {budget: {'honest_mean','honest_std','fold_scores','search_mean','bias'}}}
    by_model_budget: Dict[str, Dict[int, dict]] = field(default_factory=dict)
    # The inner SearchResult from each outer fold, for inspection / blueprint
    # reporting (rep_sparsity, chosen reps, ...). One per outer fold.
    inner_searches: List[Optional[SearchResult]] = field(default_factory=list)

    # ---- SearchResult-compatible views -------------------------------------
    # These deliberately mirror SearchResult's field names so existing reporting
    # code keeps working — but every number they return is the HONEST outer-fold
    # score, not the selection-leaked inner one. Swapping find_best_hybrid for
    # nested_cv_evaluate therefore upgrades a report in place.

    @property
    def best_by_model(self) -> Dict[str, dict]:
        """{model: {...}} with score_mean = the honest outer-fold hybrid score.

        `blueprint` is the winner from the FIRST outer fold that produced one —
        a representative, NOT "the" blueprint: nested CV legitimately picks a
        different recipe per outer fold. Use `blueprints_per_fold` when the
        per-fold identity matters (rep_sparsity, transfer, stability).

        The flat recipe fields (allocation_method / importance_method / budget /
        internal_evaluator / n_features) are ALSO surfaced, read off that
        representative blueprint, because every existing report reads them
        straight off a SearchResult row. They describe fold 0's winner only —
        the SCORE is the honest cross-fold mean, the RECIPE is representative.
        """
        out: Dict[str, dict] = {}
        for m, e in (self.by_model or {}).items():
            bps = [b for b in (e.get('blueprints_per_fold') or []) if b]
            bp = bps[0] if bps else None
            bpd = (bp.to_dict() if hasattr(bp, 'to_dict')
                   else (bp if isinstance(bp, dict) else {})) or {}
            out[m] = {
                'model': m,
                'score_mean': e['hybrid_mean'],
                'score_std': e['hybrid_std'],
                'fold_scores': e['hybrid_fold_scores'],
                'blueprint': bp,
                'blueprints_per_fold': e.get('blueprints_per_fold'),
                'search_mean': e.get('search_mean'),
                'bias': e.get('bias'),
                'allocation_method': bpd.get('allocation_method'),
                'importance_method': bpd.get('importance_method'),
                'internal_evaluator': bpd.get('internal_evaluator'),
                'budget': bpd.get('budget'),
                'n_features': e.get('n_features', bpd.get('budget')),
            }
        return out

    @property
    def baselines(self) -> Dict[str, Dict[str, float]]:
        """{rep: {model: honest mean}} — SearchResult.baselines shape."""
        out: Dict[str, Dict[str, float]] = {}
        for m, reps in (self.baselines_by_model or {}).items():
            for r, e in reps.items():
                out.setdefault(r, {})[m] = e['mean']
        return out

    @property
    def baseline_fold_scores(self) -> Dict[str, Dict[str, list]]:
        """{rep: {model: [per-outer-fold score]}} — SearchResult shape."""
        out: Dict[str, Dict[str, list]] = {}
        for m, reps in (self.baselines_by_model or {}).items():
            for r, e in reps.items():
                out.setdefault(r, {})[m] = e['fold_scores']
        return out

    @property
    def all_results(self) -> List[dict]:
        """Every config the INNER searches evaluated, across all outer folds.

        WARNING: these carry the inner search's own CV scores, which are
        selection-side diagnostics — NOT honest held-out scores. Use them for
        search-behaviour questions (which importance method wins, grid ANOVA),
        never as a reported hybrid performance number. For that, use
        best_by_model / by_model, which are scored on the untouched outer fold.

        Each row is tagged with 'outer_fold'.
        """
        rows: List[dict] = []
        for fold_i, s in enumerate(self.inner_searches or []):
            if s is None:
                continue
            for r in (s.all_results or []):
                rr = dict(r)
                rr['outer_fold'] = fold_i
                rows.append(rr)
        return rows

    @property
    def best_by_model_budget(self) -> Dict[str, Dict[int, dict]]:
        """{model: {budget: {'score_mean': honest, ...}}} — SearchResult shape."""
        out: Dict[str, Dict[int, dict]] = {}
        for m, bd in (self.by_model_budget or {}).items():
            out[m] = {}
            for b, e in bd.items():
                bp = e.get('blueprint')
                bpd = (bp.to_dict() if hasattr(bp, 'to_dict')
                       else (bp if isinstance(bp, dict) else {})) or {}
                out[m][b] = {
                    'model': m,
                    'score_mean': e['honest_mean'],
                    'score_std': e['honest_std'],
                    'fold_scores': e['fold_scores'],
                    'search_mean': e.get('search_mean'),
                    'bias': e.get('bias'),
                    # Recipe fields, mirroring a SearchResult row so per-budget
                    # reporting code (c06's capacity gradient) reads the same
                    # keys. Representative of one outer fold — see the note in
                    # nested_cv_evaluate.
                    'blueprint': bp,
                    'blueprints_per_fold': e.get('blueprints_per_fold'),
                    'allocation_method': bpd.get('allocation_method'),
                    'importance_method': bpd.get('importance_method'),
                    'internal_evaluator': bpd.get('internal_evaluator'),
                    'budget': bpd.get('budget', b),
                    'n_features': bpd.get('budget', b),
                }
        return out

    def _bp_json(self, bp):
        if bp is None:
            return None
        return bp.to_dict() if hasattr(bp, 'to_dict') else bp

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary. Experiments dump this directly under the
        top-level key "nested_cv"."""
        by_model = {}
        for m, e in (self.by_model or {}).items():
            d = dict(e)
            d['blueprints_per_fold'] = [
                self._bp_json(b) for b in (e.get('blueprints_per_fold') or [])]
            by_model[m] = d
        return {
            'metric_name': self.metric_name,
            'lower_is_better': bool(self.lower_is_better),
            'n_outer_folds': int(self.n_outer_folds),
            'by_model': by_model,
            'baselines_by_model': self.baselines_by_model,
            'by_model_budget': {
                m: {str(b): self._budget_json(e) for b, e in bd.items()}
                for m, bd in (self.by_model_budget or {}).items()
            },
        }

    def _budget_json(self, e: dict) -> dict:
        """Serialise one by_model_budget entry (Blueprints -> dicts)."""
        d = dict(e)
        d['blueprint'] = self._bp_json(e.get('blueprint'))
        d['blueprints_per_fold'] = [
            self._bp_json(b) for b in (e.get('blueprints_per_fold') or [])]
        return d


def nested_cv_cache_namespace(fold_i: int, n_outer_folds: int, seed: int) -> str:
    """The cache namespace for one outer fold's inner search.

    SINGLE SOURCE OF TRUTH. nested_cv_evaluate and the precompute layer must
    agree on this string exactly, or precompute warms cache entries the
    experiment never reads and every fold recomputes from scratch.
    """
    return f'ocv{int(n_outer_folds)}f{int(fold_i)}s{int(seed)}'


def nested_cv_outer_folds(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: Optional[np.ndarray],
    n_outer_folds: int,
    seed: int,
    task_type: str,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """The outer (train_idx, test_idx) splits used by nested_cv_evaluate.

    SINGLE SOURCE OF TRUTH — precompute calls this so it searches exactly the
    training subsets the experiment will later ask for.
    """
    splitter, use_groups = _make_cv_splitter(
        groups, n_outer_folds, seed, task_type=task_type)
    sample_X = next(iter(base_reps.values()))
    split_args = (sample_X, labels, groups) if use_groups else (sample_X, labels)
    return list(splitter.split(*split_args))


def _outer_fit_score(X_train, y_train, X_test, y_test, model_name, task_type,
                     metric_fn, seed=42, objective: str = OBJECTIVE_PREDICTIVE,
                     train_groups=None):
    """Score a feature matrix on the held-out outer fold.

    Scaler and imputation are fit on outer_train ONLY.

    objective='predictive' (default): fit `model_name` on outer_train, predict
    outer_test, score with metric_fn.

    objective='similarity_sac': no downstream model is fit. The features were
    selected on outer_train (inside the blueprint rebuild); we score the
    GEOMETRY of the untouched outer fold directly (SAC, higher is better). This
    is what makes nested CV usable for the similarity experiments — without it
    they could only be evaluated by fitting a predictor, which is not the
    quantity they report.

    Returns (score, metric_dict) — (nan, {}) when the model is unavailable or
    the fit fails.
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    X_tr, X_te, _ = _impute_nan(X_tr, X_te)

    if objective == OBJECTIVE_SIMILARITY_SAC:
        s = _similarity_objective_score(
            X_te, y_test, task_type == 'classification', seed=seed)
        return float(s), ({'sac': float(s)} if np.isfinite(s) else {})

    model = get_model(model_name, task_type, n_samples=len(y_train))
    if model is None:
        return float('nan'), {}
    m = deepcopy(model)
    if hasattr(m, 'n_neighbors') and hasattr(m, 'set_params'):
        m.set_params(n_neighbors=min(5, max(1, int(np.sqrt(len(y_train))))))
    # train_groups keeps the neural models' internal early-stopping carve
    # group-aware inside the outer training set.
    _fit_model(m, X_tr, y_train, groups=train_groups)
    y_pred = _predict_for_metric(m, X_te, metric_fn)
    score = float(metric_fn(y_test, y_pred))
    metrics = _compute_classic_metrics(m, X_te, y_test, y_pred, task_type)
    return score, metrics


def nested_cv_evaluate(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: Optional[np.ndarray] = None,
    *,
    n_outer_folds: int = 5,
    seed: int = 42,
    verbose: bool = True,
    dataset_name: Optional[str] = None,
    **search_kwargs,
) -> NestedCVResult:
    """Nested CV: the honest hybrid-vs-baseline comparison.

    For each outer fold: search on outer_train only, then score BOTH the winning
    hybrid AND the base representations on the untouched outer_test fold.

    Every inner search is cached under its own `cache_namespace` (one per outer
    fold). Without that, all outer folds map to the same cache filenames and
    evict each other — the metas are content-checked so nothing WRONG is served,
    but nothing is reused either. `dataset_name` itself is passed through intact
    so the per-dataset budget / model-skip policies still resolve.

    Args:
        base_reps: {rep_name: matrix}. Full dataset — this function does the
            slicing. Do NOT pre-slice.
        labels: Full label vector.
        groups: Group labels for group-aware splitting. Required (no random
            splits).
        n_outer_folds: Number of outer held-out folds.
        seed: Seed for the outer splitter.
        dataset_name: Registry name. Needed for caching.
        **search_kwargs: Forwarded to find_best_hybrid (budgets, speed, metric,
            task_type, augmentations, ...).

    Returns:
        NestedCVResult.
    """
    from kirby.hybrid import apply_feature_selection, apply_augmentation_selection

    objective = search_kwargs.get('objective', OBJECTIVE_PREDICTIVE)
    _sim_mode = (objective == OBJECTIVE_SIMILARITY_SAC)

    if _sim_mode:
        # SAC is the score; there is no predictor and no metric_fn. Force the
        # direction explicitly rather than letting _resolve_metric guess.
        metric_fn, lower_is_better = None, False
        metric_name = 'sac'
    else:
        metric = search_kwargs.get('metric', 'r2')
        metric_fn, lower_is_better = _resolve_metric(metric)
        metric_name = metric if isinstance(metric, str) else 'custom'

    task_type = search_kwargs.get('task_type')
    if task_type not in ('classification', 'regression'):
        raise ValueError(
            f"nested_cv_evaluate: task_type={task_type!r}. Must be "
            f"'classification' or 'regression'. Pass it explicitly.")

    full_augs = search_kwargs.get('augmentations', None)

    # Outer splits via the shared helper, so precompute warms exactly the
    # training subsets we ask for here.
    outer_folds = nested_cv_outer_folds(
        base_reps, labels, groups, n_outer_folds, seed, task_type)
    n_outer = len(outer_folds)

    # {model: [per-outer-fold score]}
    hybrid_folds: Dict[str, List[Optional[float]]] = {}
    single_folds: Dict[str, List[Optional[float]]] = {}
    single_reps: Dict[str, List[Optional[str]]] = {}
    blueprints: Dict[str, List[Optional[dict]]] = {}
    search_means: Dict[str, List[float]] = {}
    # {model: {rep: [per-outer-fold score]}}
    per_rep: Dict[str, Dict[str, List[Optional[float]]]] = {}
    # {model: {budget: [per-outer-fold score]}} and inner counterpart
    budget_folds: Dict[str, Dict[int, List[Optional[float]]]] = {}
    budget_search: Dict[str, Dict[int, List[float]]] = {}
    # {model: {budget: [per-outer-fold Blueprint]}} — the recipe each outer fold
    # picked AT that budget. Reporting code needs the recipe fields
    # (importance/allocation) alongside the honest score.
    budget_bps: Dict[str, Dict[int, List[Optional[Blueprint]]]] = {}
    inner_searches: List[Optional[SearchResult]] = []
    hybrid_failures: List[Tuple[str, int, str]] = []

    def _slot(d, model, default):
        if model not in d:
            d[model] = [default] * n_outer
        return d[model]

    for fold_i, (tr, te) in enumerate(outer_folds):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Nested CV outer fold {fold_i + 1}/{n_outer} "
                  f"(train={len(tr)}, test={len(te)})")
            print(f"{'=' * 60}")

        train_reps = {k: v[tr] for k, v in base_reps.items()}
        test_reps = {k: v[te] for k, v in base_reps.items()}
        y_train, y_test = labels[tr], labels[te]
        train_groups = groups[tr] if groups is not None else None
        train_augs = ({k: v[tr] for k, v in full_augs.items()}
                      if full_augs else None)
        test_augs = ({k: v[te] for k, v in full_augs.items()}
                     if full_augs else None)

        inner_kwargs = dict(search_kwargs)
        inner_kwargs['verbose'] = verbose
        inner_kwargs['augmentations'] = train_augs
        # Each outer fold searches a DIFFERENT training subset. Give each its
        # own cache namespace or they collide on one file and evict each other.
        _ns = nested_cv_cache_namespace(fold_i, n_outer, seed)
        inner_kwargs['cache_namespace'] = (
            f"{_ns}__{inner_kwargs['cache_namespace']}"
            if inner_kwargs.get('cache_namespace') else _ns)

        # --- rep_factories: remap indices into the ORIGINAL dataset ---
        # A factory is called factory(train_idx, val_idx, train_labels) and
        # closes over the FULL-dataset SMILES / sequence list, doing
        # full_data[train_idx]. But the inner search only sees outer-train rows,
        # so the indices it hands the factory are subset-local (0..len(tr)-1).
        # Passed through raw they would address the WRONG molecules — silently,
        # since the shapes still line up. Wrap each factory so subset-local
        # indices are lifted back to global dataset positions via `tr`.
        if inner_kwargs.get('rep_factories'):
            _tr_global = np.asarray(tr)

            def _lift(_fn):
                def _wrapped(train_idx, val_idx, train_labels):
                    return _fn(_tr_global[np.asarray(train_idx)],
                               _tr_global[np.asarray(val_idx)],
                               train_labels)
                return _wrapped

            inner_kwargs['rep_factories'] = {
                _name: _lift(_fn)
                for _name, _fn in inner_kwargs['rep_factories'].items()
            }
            # The factory outputs now depend on WHICH outer fold we are in, so
            # the fine-tuned-embedding cache must not be shared across folds.
            # cache_namespace above already covers the FT embedding path.
            if inner_kwargs.get('rep_factories_id'):
                inner_kwargs['rep_factories_id'] = (
                    f"{inner_kwargs['rep_factories_id']}__{_ns}")

        try:
            inner = find_best_hybrid(
                base_reps=train_reps,
                labels=y_train,
                groups=train_groups,
                dataset_name=dataset_name,
                **inner_kwargs,
            )
        except Exception as e:
            print(f"  Outer fold {fold_i}: inner search FAILED ({type(e).__name__}: {e})")
            inner_searches.append(None)
            continue
        inner_searches.append(inner)

        models_here = sorted(inner.best_by_model.keys())

        for model_name in models_here:
            _slot(hybrid_folds, model_name, None)
            _slot(single_folds, model_name, None)
            _slot(single_reps, model_name, None)
            _slot(blueprints, model_name, None)
            search_means.setdefault(model_name, [])
            per_rep.setdefault(model_name, {})

            # ---- HYBRID ARM: blueprint chosen on outer_train only ----
            best = inner.best_by_model[model_name]
            bp = best['blueprint']
            search_means[model_name].append(float(best['score_mean']))
            try:
                X_tr_h, finfo = apply_blueprint(
                    train_reps, bp, labels=y_train, augmentations=train_augs)
                bp_obj = bp if isinstance(bp, Blueprint) else Blueprint.from_dict(bp)
                te_bp_reps = {k: test_reps[k] for k in bp_obj.reps if k in test_reps}
                X_te_h = apply_feature_selection(te_bp_reps, finfo)

                aug_info = finfo.get('augmentation_info')
                if test_augs is not None and aug_info is not None:
                    aug_te = apply_augmentation_selection(test_augs, aug_info)
                    if aug_te is not None:
                        X_te_h = np.hstack([X_te_h, aug_te])

                score, _ = _outer_fit_score(
                    X_tr_h, y_train, X_te_h, y_test, model_name, task_type,
                    metric_fn, seed=seed, objective=objective,
                    train_groups=train_groups)
                hybrid_folds[model_name][fold_i] = score
                # Store the Blueprint OBJECT (downstream reads .reps/.budget);
                # to_dict() serialises it at persist time.
                blueprints[model_name][fold_i] = bp_obj
            except Exception as e:
                # ALWAYS loud, never verbose-gated. A silently-swallowed hybrid
                # arm leaves hybrid_fold_scores all-None and delta = NaN, which
                # reads as "no effect" rather than "the arm never ran" — the
                # experiment would look like it succeeded with nothing in it.
                hybrid_failures.append((model_name, fold_i, f'{type(e).__name__}: {e}'))
                print(f"  !! hybrid outer-eval FAILED [{model_name}, outer fold "
                      f"{fold_i}]: {type(e).__name__}: {e}")

            # ---- BASELINE ARM: every rep scored on outer_test ----
            for rep_name, Xr in base_reps.items():
                try:
                    s, _ = _outer_fit_score(
                        Xr[tr], y_train, Xr[te], y_test, model_name, task_type,
                        metric_fn, seed=seed, objective=objective,
                        train_groups=train_groups)
                except Exception:
                    s = float('nan')
                per_rep[model_name].setdefault(rep_name, [None] * n_outer)
                per_rep[model_name][rep_name][fold_i] = (
                    s if np.isfinite(s) else None)

            # The BEST SINGLE REP is picked using the inner search's own
            # baselines — i.e. outer_train only. Identical tax to the hybrid.
            # It is NOT the argmax over outer_test scores (that would be an
            # oracle, and would make the baseline arm unbeatable-by-construction).
            cand = []
            for rep_name, per_model in (inner.baselines or {}).items():
                v = (per_model or {}).get(model_name)
                if v is None or not np.isfinite(v):
                    continue
                cand.append((rep_name, float(v)))
            if cand:
                pick = (min(cand, key=lambda t: t[1]) if lower_is_better
                        else max(cand, key=lambda t: t[1]))[0]
                single_reps[model_name][fold_i] = pick
                sv = per_rep[model_name].get(pick, [None] * n_outer)[fold_i]
                single_folds[model_name][fold_i] = sv

            # ---- per-budget honest scores ----
            bud = (inner.best_by_model_budget or {}).get(model_name, {}) or {}
            for budget, cfg in bud.items():
                budget_folds.setdefault(model_name, {}).setdefault(
                    int(budget), [None] * n_outer)
                budget_search.setdefault(model_name, {}).setdefault(
                    int(budget), [])
                budget_bps.setdefault(model_name, {}).setdefault(
                    int(budget), [None] * n_outer)
                budget_search[model_name][int(budget)].append(
                    float(cfg['score_mean']))
                try:
                    bbp = cfg['blueprint']
                    Xb_tr, bfinfo = apply_blueprint(
                        train_reps, bbp, labels=y_train, augmentations=train_augs)
                    bbp_obj = (bbp if isinstance(bbp, Blueprint)
                               else Blueprint.from_dict(bbp))
                    budget_bps[model_name][int(budget)][fold_i] = bbp_obj
                    te_b = {k: test_reps[k] for k in bbp_obj.reps if k in test_reps}
                    Xb_te = apply_feature_selection(te_b, bfinfo)
                    b_aug = bfinfo.get('augmentation_info')
                    if test_augs is not None and b_aug is not None:
                        a_te = apply_augmentation_selection(test_augs, b_aug)
                        if a_te is not None:
                            Xb_te = np.hstack([Xb_te, a_te])
                    bs, _ = _outer_fit_score(
                        Xb_tr, y_train, Xb_te, y_test, model_name, task_type,
                        metric_fn, seed=seed, objective=objective,
                        train_groups=train_groups)
                    if np.isfinite(bs):
                        budget_folds[model_name][int(budget)][fold_i] = bs
                except Exception:
                    pass

    # A hybrid arm that never scored ANYTHING is a broken run, not a null
    # result. Left alone it produces hybrid_mean=NaN and delta=NaN, which reads
    # downstream as "the hybrid didn't help" — a scientific claim we would be
    # inventing out of a crash. Refuse to return it.
    if hybrid_failures and not any(
            v is not None
            for folds_ in hybrid_folds.values() for v in folds_):
        _uniq = sorted({msg for _, _, msg in hybrid_failures})
        raise RuntimeError(
            "nested_cv_evaluate: the hybrid arm failed on EVERY (model, outer "
            f"fold) — {len(hybrid_failures)} failures, no hybrid score was ever "
            "produced. Refusing to report NaN deltas as a null result. "
            f"Distinct errors: {_uniq}")

    # ---------------- aggregate ----------------
    def _fin(xs):
        return [float(x) for x in xs if x is not None and np.isfinite(x)]

    by_model: Dict[str, dict] = {}
    for model_name in sorted(set(hybrid_folds) | set(single_folds)):
        hf = hybrid_folds.get(model_name, [None] * n_outer)
        sf = single_folds.get(model_name, [None] * n_outer)

        per_fold_delta: List[Optional[float]] = [None] * n_outer
        paired: List[float] = []
        for f in range(n_outer):
            hv, bv = hf[f], sf[f]
            if hv is None or bv is None or not np.isfinite(hv) or not np.isfinite(bv):
                continue
            d = (bv - hv) if lower_is_better else (hv - bv)
            per_fold_delta[f] = float(d)
            paired.append(float(d))

        hv_f, sv_f = _fin(hf), _fin(sf)
        sm = search_means.get(model_name) or []
        search_mean = float(np.mean(sm)) if sm else float('nan')
        hybrid_mean = float(np.mean(hv_f)) if hv_f else float('nan')
        # bias = how much the whole-dataset search score flattered itself,
        # direction-corrected so positive = optimistic.
        if np.isfinite(search_mean) and np.isfinite(hybrid_mean):
            bias = ((hybrid_mean - search_mean) if lower_is_better
                    else (search_mean - hybrid_mean))
        else:
            bias = float('nan')

        by_model[model_name] = {
            'hybrid_fold_scores': hf,
            'best_single_fold_scores': sf,
            'best_single_reps_per_fold': single_reps.get(model_name,
                                                         [None] * n_outer),
            'blueprints_per_fold': blueprints.get(model_name, [None] * n_outer),
            'hybrid_mean': hybrid_mean,
            'hybrid_std': float(np.std(hv_f)) if hv_f else float('nan'),
            'best_single_mean': float(np.mean(sv_f)) if sv_f else float('nan'),
            'best_single_std': float(np.std(sv_f)) if sv_f else float('nan'),
            'per_fold_delta': per_fold_delta,
            # THE number to report: mean of the paired per-fold differences.
            'delta_mean_of_folds': (float(np.mean(paired)) if paired
                                    else float('nan')),
            'n_paired_folds': len(paired),
            'search_mean': search_mean,
            'bias': bias,
        }

    baselines_by_model: Dict[str, Dict[str, dict]] = {}
    for model_name, reps_d in per_rep.items():
        baselines_by_model[model_name] = {}
        for rep_name, fs in reps_d.items():
            f = _fin(fs)
            baselines_by_model[model_name][rep_name] = {
                'fold_scores': fs,
                'mean': float(np.mean(f)) if f else float('nan'),
                'std': float(np.std(f)) if f else float('nan'),
            }

    by_model_budget: Dict[str, Dict[int, dict]] = {}
    for model_name, bd in budget_folds.items():
        by_model_budget[model_name] = {}
        for budget, fs in sorted(bd.items()):
            f = _fin(fs)
            sm = budget_search.get(model_name, {}).get(budget, [])
            s_mean = float(np.mean(sm)) if sm else float('nan')
            h_mean = float(np.mean(f)) if f else float('nan')
            if np.isfinite(s_mean) and np.isfinite(h_mean):
                b = ((h_mean - s_mean) if lower_is_better else (s_mean - h_mean))
            else:
                b = float('nan')
            bps_b = [x for x in
                     (budget_bps.get(model_name, {}).get(budget) or []) if x]
            by_model_budget[model_name][budget] = {
                'honest_mean': h_mean,
                'honest_std': float(np.std(f)) if f else float('nan'),
                'fold_scores': fs,
                'search_mean': s_mean,
                'bias': b,
                # Representative recipe (first outer fold that produced one).
                # The SCORE is the honest cross-fold mean; the RECIPE is one
                # fold's pick — nested CV may legitimately choose differently
                # per fold. blueprints_per_fold has them all.
                'blueprint': bps_b[0] if bps_b else None,
                'blueprints_per_fold': budget_bps.get(model_name, {}).get(budget),
            }

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"NESTED CV SUMMARY ({n_outer} outer folds) — metric={metric_name}")
        print(f"{'=' * 60}")
        for model_name in sorted(by_model):
            e = by_model[model_name]
            print(f"\n  {model_name}:")
            print(f"    hybrid (honest):      {e['hybrid_mean']:.4f} "
                  f"+/-{e['hybrid_std']:.3f}")
            print(f"    best single (honest): {e['best_single_mean']:.4f} "
                  f"+/-{e['best_single_std']:.3f}  "
                  f"reps={e['best_single_reps_per_fold']}")
            print(f"    paired delta:         {e['delta_mean_of_folds']:+.4f} "
                  f"(+ = hybrid better, over {e['n_paired_folds']} folds)")
            print(f"    selection bias:       {e['bias']:+.4f} "
                  f"(inner search said {e['search_mean']:.4f})")

    return NestedCVResult(
        by_model=by_model,
        baselines_by_model=baselines_by_model,
        metric_name=metric_name,
        lower_is_better=lower_is_better,
        n_outer_folds=n_outer,
        by_model_budget=by_model_budget,
        inner_searches=inner_searches,
    )


def find_best_hybrid_with_aggregation(**kwargs):
    """TODO: Test aggregating importance from non-correlated cluster reps.
    Hypothesis: borda/avg of 6 cluster reps may improve feature selection.
    """
    raise NotImplementedError("Aggregation experiment — see benchmark_importance_aggregation.py")


def find_best_hybrid_low_dim(**kwargs):
    """TODO: Optimize for dim 25-50 for downstream similarity/clustering.
    Hypothesis: different methods work best at very low budgets.
    """
    raise NotImplementedError("Low-dim experiment")
