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
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

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
#        searches/   — full SearchResult per (dataset, tier, aug_strategy)
#        baselines/  — per-(dataset, rep, model, metric, folds, seed) scores
#        folds/      — per-(dataset, folds, seed) CV split indices
#   4. Atomic writes via tempfile + rename. Parallel writers to the same
#      filename collide on os.replace, not on partial content.
#   5. Failure to load is never fatal — the caller recomputes and rewrites.

# Set for the duration of ONE find_best_hybrid call when the caller passes
# cache_dir=. Until 2026-09-03 cache_dir rerooted only the search pickle and the
# result database: per-representation baselines, fold splits, per-fold hybrid
# evaluations and fine-tuned embeddings kept reading and writing the default
# root, so a run pointed at a second cache silently mixed two caches. Every path
# builder goes through _cache_base(), so overriding it here reroots all of them
# with no new argument on six signatures. Saved and restored by
# _cache_root_override(); KIRBY_CACHE_DIR is unaffected and still wins when no
# cache_dir is passed.
_CACHE_ROOT_OVERRIDE: Optional[Path] = None


def _cache_base() -> Path:
    if _CACHE_ROOT_OVERRIDE is not None:
        return Path(_CACHE_ROOT_OVERRIDE)
    return Path(os.environ.get('KIRBY_CACHE_DIR',
                                str(Path.home() / '.cache' / 'kirby')))


@contextmanager
def _cache_root_override(cache_dir):
    """Point every cache path at `cache_dir` for the duration of the block."""
    global _CACHE_ROOT_OVERRIDE
    _prev = _CACHE_ROOT_OVERRIDE
    if cache_dir is not None:
        _CACHE_ROOT_OVERRIDE = Path(cache_dir)
    try:
        yield
    finally:
        _CACHE_ROOT_OVERRIDE = _prev


def _sanitize_filename_part(s: str) -> str:
    return str(s).replace('+', '_').replace('/', '_').replace(' ', '_')


def _objective_cache_token(objective) -> Optional[str]:
    """The objective as it appears in a cache FILENAME.

    A plain name is itself, so every cache entry written before blends existed
    keeps its path. A blend is a dict, and str() of a dict puts braces, quotes
    and commas into the filename and depends on insertion order — so a blend
    becomes its stable sorted name instead, e.g.
    'blend_predictive0.5_similarity0.25_struct_fid0.25'.
    """
    if objective is None:
        return None
    if isinstance(objective, dict):
        parts = [f'{k}{float(v):g}' for k, v in sorted(objective.items())]
        return 'blend_' + '_'.join(parts)
    return str(objective)


def _search_cache_path(dataset_name: str, tier: str,
                       aug_strategy: str,
                       condition_suffix: Optional[str] = None) -> Path:
    t = _sanitize_filename_part(tier)
    a = _sanitize_filename_part(aug_strategy)
    stem = f'{dataset_name}__{t}__{a}'
    if condition_suffix:
        stem = f'{stem}__{_sanitize_filename_part(condition_suffix)}'
    return _cache_base() / 'searches' / f'{stem}.pkl'


def _search_checkpoint_path(dataset_name: str, tier: str,
                            aug_strategy: str,
                            condition_suffix: Optional[str] = None) -> Path:
    p = _search_cache_path(dataset_name, tier, aug_strategy,
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
                         noise_id: Optional[str] = None,
                         objective: Optional[str] = None) -> Path:
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
    # objective namespaces the baseline just like noise_id: a geometry/similarity
    # search stores a structural score under (rep, model, metric), which a
    # predictive run stores a model CV score under. Same reasoning as tier
    # namespacing — but this path carries no tier, so the marker rides here.
    # Predictive leaves the filename byte-identical (no suffix), so existing
    # clean baselines keep hitting.
    _ob_tok = _objective_cache_token(objective)
    ob = (f'__obj-{_sanitize_filename_part(_ob_tok)}'
          if _ob_tok and _ob_tok != OBJECTIVE_PREDICTIVE else '')
    return (_cache_base() / 'baselines' /
            f'{dataset_name}__{rep_name}__{model_name}__'
            f'{metric_name}__{n_folds}f_{seed}{sec}{nz}{ob}.pkl')


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


# NB: search-level hit/miss/stale counters were removed — search results now
# live in the SQLite result_db (result_cache.py), not a searches/*.pkl with
# hit/miss accounting. Baseline + hybrid_eval caches are still .pkl and counted.
_CACHE_STATS = {
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
        f"cache stats — baseline: {s['baseline_hit']} hit / "
        f"{s['baseline_miss']} miss / {s['baseline_stale']} stale; "
        f"hybrid_eval: {s['hybrid_eval_hit']} hit / "
        f"{s['hybrid_eval_miss']} miss / {s['hybrid_eval_stale']} stale"
    )


# Meta keys _make_search_meta writes ONLY when the caller turned the feature
# on, so that files written before the feature existed keep matching. That
# one-sidedness is why they need the extra check in _load_cache_with_meta: a
# caller who asks for none of them would otherwise be served a file written by
# a caller who asked for one.
_OPT_IN_META_KEYS = (
    'tune',
    'tune_n_iter',
    'tune_knn_k',
    'struct_pairs_id',
    # Added 2026-09-03. Same one-sidedness, same consequence, and none of these
    # is in the search pickle's FILENAME either — so a run that caps augmentation
    # columns per source, or prunes on a schedule, or overrides a model's
    # pruning, wrote its result where a run doing none of those then read it.
    'aug_per_source_budget',
    'pruning_checkpoints',
    'pruning_tail_drop',
    'pruning_min_configs',
    'pruning_strategy_top_n',
    'model_pruning_overrides',
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
    # The loop above only checks keys the CALLER expects, and the opt-in keys
    # below are written only when their feature is on. So a plain run, whose
    # expected meta has none of them, matched a file a tuned run had written
    # and was served the tuned result. `tune` and `tune_knn_k` are not in the
    # FILENAME either, so both arms of a tuned-vs-untuned comparison land on
    # one path — the untuned arm silently read the tuned arm's answer. Reject
    # a stored opt-in the caller did not ask for. Files written before any of
    # these existed carry none of them and still match.
    for k in _OPT_IN_META_KEYS:
        if k in stored and k not in expected_meta:
            _CACHE_STATS[f'{kind}_stale'] = _CACHE_STATS.get(f'{kind}_stale', 0) + 1
            if verbose:
                print(f"  Cache {path.name}: stored opt-in '{k}'="
                      f"{stored.get(k)!r} was not requested — will recompute.")
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
    dataset_name: str, tier: str, aug_strategy: str,
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
    candidate_ranking: str = 'residual_correlation',
    greedy_tolerance: float = 0.0,
    augmentation_margin: float = 0.0,
    augmentation_fraction: Optional[float] = None,
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
    # Whether the tuned n_neighbors was kept instead of being forced to 5.
    # Two runs that differ only here fit DIFFERENT KNN estimators, so they are
    # different searches. Emitted only when on, same rule as `tune`.
    tune_knn_k: bool = False,
    # Identity of the MCS pair set the search recorded structural fidelity
    # against, or None when it recorded none. Emitted only when set, so every
    # pre-existing cache file still matches (_load_cache_with_meta iterates the
    # EXPECTED meta). A search WITH pairs stores an extra 'struct_fidelity' per
    # fold, so it must not be served a cached search that stored none.
    struct_pairs_id: Optional[str] = None,
) -> Dict[str, Any]:
    meta_extra: Dict[str, Any] = {}
    if struct_pairs_id:
        meta_extra['struct_pairs_id'] = struct_pairs_id
    if tune:
        meta_extra['tune'] = True
        meta_extra['tune_n_iter'] = int(tune_n_iter)
    if tune_knn_k:
        meta_extra['tune_knn_k'] = True
    if pruning_schedule is not None:
        ckpts = tuple(sorted(int(b) for b in pruning_schedule.get('checkpoints', [])))
        td = float(pruning_schedule.get('tail_drop_per_checkpoint', 0.0))
        mc = int(pruning_schedule.get('min_configs_per_model', 1))
        stn = int(pruning_schedule.get('strategy_top_n', 3))
        if stn != 3:
            meta_extra['pruning_strategy_top_n'] = stn
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
        'augmentation_margin': float(augmentation_margin),
        'augmentation_fraction': (float(augmentation_fraction)
                                  if augmentation_fraction is not None else None),
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
                            noise_id: Optional[str] = None,
                            objective: Optional[str] = None) -> Path:
    """Per-(rep_subset, config, fold, model) hybrid-eval cache path.

    Layout: hybrid_evals/{dataset}/{...}.pkl so one subdir per dataset
    stays under ~5k-10k files for a typical precompute
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
    # objective namespaces the hybrid-eval file for the same reason as noise_id
    # (see above): a similarity/geometry search evaluates a different quantity
    # than a predictive one for the same subset+config+fold. Predictive omits the
    # suffix, so existing entries keep their path.
    _ob_tok = _objective_cache_token(objective)
    ob = (f'__obj-{_sanitize_filename_part(_ob_tok)}'
          if _ob_tok and _ob_tok != OBJECTIVE_PREDICTIVE else '')
    fname = (f'{subset_hash}__{a}__{i}__{e}__b{budget}__'
             f'f{fold_idx}of{n_folds}_{seed}__{m}__{ags}{aug_sfx}{nz}{ob}.pkl')
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
    augmentation_margin: float,
    augmentation_fraction: Optional[float],
    augmentation_threshold: Optional[float], augmentation_budget: int,
    metric_name: str, task_type: str,
    aug_per_source_budget: Optional[int] = None,
    subset_widths: Optional[Dict[str, int]] = None,
    struct_pairs_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        'kind': 'hybrid_eval',
        'dataset_name': dataset_name,
        # Which MCS pair set (if any) the stored fold metrics were computed
        # against. A run with structural pairs stores an extra 'struct_fidelity'
        # metric per fold; a run without does not. Same key would serve one to
        # the other, so the pair set is part of the identity. None = no pairs.
        'struct_pairs_id': struct_pairs_id,
        'subset_names': tuple(sorted(subset_names)),
        'subset_hashes': subset_hashes,
        # Feature width per rep. subset_hashes is in ignore_keys (neural
        # embeddings are not bit-reproducible), so before this the meta carried
        # NO record of how wide each rep was — a width-reduced run and a
        # full-width run over the same rep names produced identical metadata and
        # each was served the other's answer. Widths are reproducible integers,
        # so they can be checked. Emitted only when supplied.
        **({'subset_widths': tuple(sorted((str(k), int(v))
                                          for k, v in subset_widths.items()))}
           if subset_widths else {}),
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
        'augmentation_margin': float(augmentation_margin),
        'augmentation_fraction': (float(augmentation_fraction)
                                  if augmentation_fraction is not None else None),
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


def _subset_name_hex(subset_names, subset_widths=None) -> str:
    """Filename-safe hash of the sorted rep NAMES in a subset, plus widths.

    The old hybrid-eval filename hashed rep VALUES (_hash_array of the
    embeddings). Neural/transformer embeddings are not bit-reproducible across
    runs, so the filename changed every run and the per-fold hybrid-eval cache
    NEVER hit -> silent wasted recompute on every neural rep. Rep names are a
    stable identity, so key the filename on the sorted names instead.

    Names ALONE, however, are not a full identity: the same rep name can arrive
    at different feature widths (a width-reduced pool vs the full-width one),
    and those two runs evaluate different matrices. Keying on names only made
    them share a filename AND an accepted meta, so whichever ran first was
    served to the other with no error. `subset_widths` is a {name: n_features}
    mapping; when given, the width joins the name in the hash so the two get
    separate files instead of colliding. Omitted -> the historical name-only
    hash, unchanged.
    """
    h = hashlib.sha1()
    for name in sorted(subset_names):
        h.update(name.encode())
        if subset_widths is not None:
            h.update(f':{int(subset_widths.get(name, -1))}'.encode())
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
    # How much better than keeping NO extras a block of extras must score to be
    # accepted. Part of the recipe: a rebuild that omits it accepts blocks the
    # search would have rejected. 0.0 = accept any improvement, however small.
    augmentation_margin: float = 0.0
    # Size the extras as a FRACTION of the budget instead of the fixed ladder
    # of 0/5/10/20/40. None = use the ladder. Set = take exactly
    # round(fraction * width) top-ranked extras, with no internal comparison,
    # which is the 2026-08-20 decision: rank once, cut at the cap, no inner
    # holdout. The value is swept by the grid, not chosen in the engine.
    augmentation_fraction: Optional[float] = None
    # Which rule ranks the pooled candidates: 'residual_correlation' (against
    # the base representation's out-of-fold residuals) or 'rf_importance'
    # (base-blind). This is a column of the stored result's KEY
    # (result_cache.CONFIG_COLS), so the search files two rankers as two rows —
    # but until 2026-08-26 it was not on the recipe, so every per-fold rebuild
    # took create_hybrid's 'residual_correlation' default whatever the search
    # ran under. A search under 'rf_importance' was therefore rebuilt under the
    # other rule, with nothing raising. Same shape as the 24 August defect.
    candidate_ranking: str = 'residual_correlation'
    # Minimum improvement a whole donor block must show for greedy_ablation to
    # keep it. None = keep any block that improves at all. Also a key column,
    # also missing from the recipe until 2026-08-26.
    augmentation_threshold: Optional[float] = None
    # Features taken from EACH donor by the 'all' and 'greedy_ablation' paths.
    # Not the same setting as aug_per_source_budget, which caps donors before
    # pooling for 'capped'/'uncapped'. Also a key column, also missing.
    augmentation_budget: int = 20
    # Length of the ranked candidate list that a block size is chosen from.
    # NOT a key column and deliberately not made one: it is swept by its own
    # experiment, which builds recipes directly and never writes to result_db,
    # so adding a key column would force every result_db/*.db to be deleted for
    # nothing. Recorded here so a rebuild reproduces the list the run used.
    # 50 clamps every augmentation_fraction at a budget of 1000 to the same 50
    # extras, which is why the fraction axis needs this to move.
    aug_max_candidates: int = 50
    # Which objective the search optimized: 'predictive' (downstream model
    # performance) or 'similarity' (held-out activity read-across).
    # A sim-optimized blueprint is a DIFFERENT recipe than a predictive one and
    # consumers should not mix them; this field makes the provenance explicit.
    objective: str = 'predictive'
    # Per-rep allocation scores from the search (enables label-free apply).
    # {rep_name: float} — higher = more budget share.
    rep_scores: Optional[Dict[str, float]] = None
    # Which extra features each fold of the search actually kept:
    # {fold_index: {donor_name: [feature indices into that donor]}}.
    # DATA, not part of the recipe and not part of any cache key — recording it
    # changes no stored result's identity. Written so that a winning run can be
    # explained ("which features won it") and so a rebuild can be checked
    # against what the search chose. Before this existed the stored recipe named
    # the strategy and the per-donor cap but never the surviving features.
    kept_augmentations: Optional[Dict[int, Dict[str, List[int]]]] = None
    # How the budget was actually split, per fold: {fold: {rep_name: n_features}}.
    # DATA, like kept_augmentations above, and for the same reason: without it no
    # stored result says which representations fed the hybrid or how many
    # features each got. That is the question the whole rebuild roadmap turns on
    # — on a large, near-tied pool a proportional split gives the best
    # representation a handful of features out of the budget, and there was no
    # way to see it in a real run. Not part of any cache key.
    allocation_per_fold: Optional[Dict[int, Dict[str, int]]] = None

    def to_dict(self) -> dict:
        d = {
            'reps': self.reps,
            'allocation_method': self.allocation_method,
            'importance_method': self.importance_method,
            'internal_evaluator': self.internal_evaluator,
            'budget': self.budget,
            'step_size': self.step_size,
            'n_folds': self.n_folds,
            'seed': self.seed,
            'metric': self.metric,
            'task_type': self.task_type,
            'augmentation_strategy': self.augmentation_strategy,
            'augmentation_margin': float(self.augmentation_margin),
            'augmentation_fraction': self.augmentation_fraction,
            'candidate_ranking': self.candidate_ranking,
            'augmentation_threshold': self.augmentation_threshold,
            'augmentation_budget': int(self.augmentation_budget),
            'aug_max_candidates': int(self.aug_max_candidates),
            'lower_is_better': self.lower_is_better,
            'objective': self.objective,
        }
        if self.rep_scores is not None:
            d['rep_scores'] = self.rep_scores
        # Omitted when unset so existing serialised blueprints round-trip
        # byte-identically. Dicts survive JSON as {name: N}.
        if self.aug_per_source_budget is not None:
            d['aug_per_source_budget'] = self.aug_per_source_budget
        # Same rule. JSON turns the integer fold keys into strings, so
        # from_dict puts them back.
        if self.kept_augmentations:
            d['kept_augmentations'] = {
                int(f): {str(n): [int(i) for i in idx] for n, idx in per.items()}
                for f, per in self.kept_augmentations.items()
            }
        # Same rule again: omitted when unset, integer fold keys restored by
        # from_dict after a JSON round trip.
        if self.allocation_per_fold:
            d['allocation_per_fold'] = {
                int(f): {str(n): int(c) for n, c in per.items()}
                for f, per in self.allocation_per_fold.items()
            }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'Blueprint':
        """Reconstruct a Blueprint from a dict (e.g. loaded from JSON).

        Ignores unknown keys so older/newer serialisations stay compatible.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        # A JSON round trip turns the integer fold keys into strings. Put them
        # back so bp.kept_augmentations[0] works whether the blueprint came
        # from a pickle or from a file.
        ka = filtered.get('kept_augmentations')
        if isinstance(ka, dict):
            filtered['kept_augmentations'] = {
                int(f): {str(n): [int(i) for i in idx] for n, idx in per.items()}
                for f, per in ka.items()
            }
        ap = filtered.get('allocation_per_fold')
        if isinstance(ap, dict):
            filtered['allocation_per_fold'] = {
                int(f): {str(n): int(c) for n, c in per.items()}
                for f, per in ap.items()
            }
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
    # The estimators find_best_hybrid(tune=True) actually fitted with, keyed by
    # model name. Empty on every untuned search and on every older pickle.
    # nested_cv_evaluate reads this so the held-out score can be produced by
    # the same estimator the inner search chose, instead of a fresh default.
    tuned_models: Dict = field(default_factory=dict)

    def __setstate__(self, state):
        """Backward-compat: old pickled SearchResult files do not contain
        importance_cache / alloc_cache / baseline_fold_scores / tuned_models.
        Python's @dataclass does NOT invoke default_factory on unpickle (only
        on __init__), so accessing those fields on an old pickle would
        AttributeError. Fill in defaults."""
        self.__dict__.update(state)
        for _fname in ('importance_cache', 'alloc_cache', 'baseline_fold_scores',
                       'tuned_models'):
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
    a geometry quantity, never a prediction metric, so metric_fn must never be
    invoked with (y_true, y_pred). If it is, something bypassed the sim path."""
    raise RuntimeError(
        "This is a geometry score, not a prediction metric — it must be "
        "scored via objective='similarity', not metric_fn(y_true, y_pred)."
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
    # SAC: higher is better. CUT as an objective 2026-08-26; the name stays
    # so an old cached metric_name still resolves. Scorer is a guard.
    'sac':         (_sac_not_a_prediction_metric, False),
    # Each molecule's property predicted from its five nearest neighbours:
    # higher is better. Guard — scored on geometry, never on predictions
    # (see objective='similarity').
    'readacross':  (_sac_not_a_prediction_metric, False),
    # MCS structural fidelity: higher is better. Same guard, same reason
    # (see objective='struct_fid').
    'struct_fidelity': (_sac_not_a_prediction_metric, False),
}

# Objective modes for the hybrid search. 'predictive' = the classic
# fit-model-then-score-predictions objective. 'similarity' = select the hybrid
# whose geometry gives the best held-out ACTIVITY READ-ACROSS (scaffold-grouped
# kNN Spearman); the honest outer fold additionally reports MCS STRUCTURAL
# FIDELITY when structural pairs are supplied. 'similarity' fits NO downstream
# model.
#
# ---------------------------------------------------------------------------
# SAC IS CUT AS A SEARCH OBJECTIVE. CLOSED 2026-08-26 by the owner.
#
# SAC = Spearman between how far apart two molecules sit and how different
# their measured property is. The name still resolves so old configs raise a
# readable error instead of NameError; nothing selects it.
#
# The evidence on BOTH sides, so nobody reports either half as new:
#
#   AGAINST (f19, tests/results/f19_objective_scoring/, herg_ki + logd).
#   Picking by SAC scored worst of four on all three yardsticks on both
#   datasets. herg_ki held-out predictive 0.4659 where picking by read-across
#   got 0.7408 and picking by prediction got 0.7198.
#
#   FOR (SIM_OPTIMIZED_HYBRID_ANALYSIS_2026_07_10.md, 4 datasets, 3 seeds).
#   A SAC-picked hybrid had a lower neighbour-lie rate than the best single
#   representation on 4 of 4, and higher noise-detection precision by
#   neighbourhood consensus on 8 of 8 (+0.059 to +0.114 on the two ADME sets).
#   Dimension and budget confounds were tested and refuted.
#
# The owner weighed both and cut it. Do not re-derive this.
# ---------------------------------------------------------------------------
OBJECTIVE_PREDICTIVE = 'predictive'
OBJECTIVE_SIMILARITY_SAC = 'similarity_sac'   # CUT — see the block above
OBJECTIVE_SIMILARITY = 'similarity'
# Added 2026-08-27. Picks the feature subset whose distances best track MCS
# distance: how much two molecules the space calls close really do share atoms
# and bonds. Needs `structural_pairs` and `row_index`; raises without them.
OBJECTIVE_STRUCT_FID = 'struct_fid'
# Added 2026-09-03. Prediction scored as a CORRELATION -- Spearman for
# regression, 2*(AUROC - 0.5) for classification -- instead of the dataset's own
# metric. Identical in every other respect to OBJECTIVE_PREDICTIVE: a model is
# fitted and scored on held-out rows.
#
# WHY IT EXISTS. Inside a blend, prediction is already scored this way, because
# an RMSE cannot be averaged with two Spearman correlations. The plain
# 'predictive' objective is not. So in an objective-weight grid the prediction
# corner was measured in one quantity and every mixture containing prediction in
# another, and the corner was not the limit of its own edge: 'predict 1.00' and
# 'predict 0.75 + neighbour 0.25' differed both in the weights AND in what
# prediction meant. This name is that corner, on the mixtures' scale.
#
# It is scored through the blend machinery as {OBJECTIVE_PREDICTIVE: 1.0}, so it
# is that limit exactly -- same fold score, same budget split by ranking the
# representations, same cache identity rules. Mirrored in hybrid.py.
OBJECTIVE_PREDICTIVE_CORR = 'predictive_corr'
_VALID_OBJECTIVES = (OBJECTIVE_PREDICTIVE, OBJECTIVE_PREDICTIVE_CORR,
                     OBJECTIVE_SIMILARITY, OBJECTIVE_STRUCT_FID)

# ---------------------------------------------------------------------------
# BLENDED OBJECTIVES — all three components, on ONE scale
#
# An objective may be a name, or a dict of names to weights:
#
#     objective={'predictive': 0.5, 'similarity': 0.25, 'struct_fid': 0.25}
#
# Each fold's blend score is the WEIGHTED MEAN of its components on that fold.
# For a mean to mean anything the components have to share a scale, and the
# dataset's own metric does not: RMSE on seven of the eight molecular sets
# against two Spearman correlations. So inside a blend ONLY, prediction is
# scored as a correlation between predicted and true on the same held-out rows —
# Spearman for regression, 2*(AUROC - 0.5) for classification, which puts chance
# at 0 the way a correlation does. All three components are then bounded in
# [-1, 1], higher-is-better, on the same rows.
#
# The plain objective='predictive' is UNTOUCHED and still scores the dataset's
# own metric. Nothing outside a blend changes.
#
# Why a weighted mean and not a rank across configurations: a rank would make a
# hybrid's score depend on which OTHER hybrids happened to be in the grid, so
# adding a configuration could change the winner among the others. The budget
# split does rank, because there it is comparing representations against each
# other and has all of them in hand. Two different sums, both deliberate.
# ---------------------------------------------------------------------------

# Which recorded number each blendable component SELECTS on. Every run records
# all of them; this says which one each component reads.
#   'predictive_corr' is the correlation form described above, NOT the dataset's
#   own metric (which is recorded separately under 'predictive').
_BLEND_COMPONENT_KEY = {
    OBJECTIVE_PREDICTIVE: 'predictive_corr',
    OBJECTIVE_SIMILARITY: 'readacross',
    OBJECTIVE_STRUCT_FID: 'struct_fidelity',
}
# The subset scored with NO downstream model.
_GEOMETRY_PRIMARY_KEY = {
    OBJECTIVE_SIMILARITY: 'readacross',
    OBJECTIVE_STRUCT_FID: 'struct_fidelity',
}
# Which recorded number a PLAIN (non-dict) objective selects on. The two
# geometry names plus 'predictive_corr', which fits a model and then reads the
# correlation. 'predictive' is absent on purpose: it is scored by the dataset's
# own metric_fn, not by a key out of the fold's metric dict.
_PLAIN_PRIMARY_KEY = dict(_GEOMETRY_PRIMARY_KEY)
_PLAIN_PRIMARY_KEY[OBJECTIVE_PREDICTIVE_CORR] = 'predictive_corr'


# Every number recorded for a winner on each outer test fold, whichever
# objective picked it. 'predictive' is the dataset's own metric;
# 'predictive_corr' is the correlation form a blend uses, kept alongside so a
# blend's own score can be read back as the weighted mean of its components.
_SIDE_METRIC_KEYS = ('predictive', 'predictive_corr', 'readacross',
                     'struct_fidelity')


def _is_geometry_objective(obj) -> bool:
    """True for the objectives scored on rep-space GEOMETRY (no model fit).

    'similarity' (nearest-neighbour activity) and 'struct_fid' (MCS) qualify,
    and so does a dict blending ONLY those two. A blend containing 'predictive'
    does not: something has to fit the model that produces its correlation.
    Centralises the 'no downstream model' gates."""
    if isinstance(obj, dict):
        return bool(obj) and all(k in _GEOMETRY_PRIMARY_KEY for k in obj)
    return obj in _GEOMETRY_PRIMARY_KEY


def _is_blend_objective(obj) -> bool:
    """True when the objective is scored by the BLEND machinery.

    Every dict, plus 'predictive_corr', which is the one-component blend
    {predictive: 1.0} under a name. What this gates is the budget split: these
    rank the representations against each other, so a precomputed per-
    representation score under some other quantity must not be used.
    """
    return isinstance(obj, dict) or obj == OBJECTIVE_PREDICTIVE_CORR


def _blend_needs_model(obj) -> bool:
    """True when a blend has a prediction component, so a model must be fitted.

    Distinct from `not _is_geometry_objective`: a plain 'predictive' objective
    also fits a model, but it is scored by the dataset's own metric, not by the
    correlation form a blend uses.
    """
    return ((isinstance(obj, dict) and OBJECTIVE_PREDICTIVE in obj)
            or obj == OBJECTIVE_PREDICTIVE_CORR)


def _prediction_correlation(y_true, y_pred, task_type) -> float:
    """Prediction quality as a correlation, so it can be blended.

    Regression: Spearman between predicted and true.
    Classification: 2 * (AUROC - 0.5), which is 0 at chance and 1 at perfect,
    matching a correlation's zero point and upper bound.

    NEVER used for a plain objective='predictive' — that keeps the dataset's own
    metric. This exists only so a blend has three numbers on one scale.
    """
    try:
        if task_type == 'classification':
            auc = _safe_roc_auc(y_true, y_pred)
            return float(2.0 * (float(auc) - 0.5)) if np.isfinite(auc) else float('nan')
        rho, _ = spearmanr(y_true, y_pred)
        return float(rho) if np.isfinite(rho) else float('nan')
    except Exception:
        return float('nan')


# What a one-component blend collapses to. Each component alone, under the name
# the engine scores that component by.
_SINGLE_COMPONENT_OBJECTIVE = {
    OBJECTIVE_PREDICTIVE: OBJECTIVE_PREDICTIVE_CORR,
    OBJECTIVE_SIMILARITY: OBJECTIVE_SIMILARITY,
    OBJECTIVE_STRUCT_FID: OBJECTIVE_STRUCT_FID,
}


def _validate_blend_objective(objective: dict) -> dict:
    """Check a dict objective and return it with weights summing to 1.

    Mirrors kirby.hybrid.normalise_objective, which hybrid.py keeps its own copy
    of so it need not import this module. Change one, change both.

    Normalising here means {'similarity': 3, 'struct_fid': 1} and
    {'similarity': 0.75, 'struct_fid': 0.25} are the SAME objective, so they
    produce one cache entry rather than two that disagree.
    """
    if not objective:
        raise ValueError(
            "find_best_hybrid: objective={} is empty. A blend needs at least "
            f"one of {tuple(_BLEND_COMPONENT_KEY)} with a positive weight.")
    bad = [k for k in objective if k not in _BLEND_COMPONENT_KEY]
    if bad:
        raise ValueError(
            f"find_best_hybrid: objective blends {bad}, which cannot be "
            f"blended. Blendable components are {tuple(_BLEND_COMPONENT_KEY)}.")
    nonpos = {k: v for k, v in objective.items() if not float(v) > 0}
    if nonpos:
        raise ValueError(
            f"find_best_hybrid: objective has non-positive weights {nonpos}. "
            "Drop a component instead of weighting it zero, so the stored "
            "metric name says which components actually scored the run.")
    if len(objective) == 1:
        # A one-component "blend" is that component. Returning the plain name
        # keeps it on ONE cache entry instead of two names for identical work.
        # {'predictive': 1.0} lands on 'predictive_corr', NOT on 'predictive':
        # a weight of 1.0 on prediction is the limit of the mixtures, and every
        # mixture scores prediction as a correlation. Sending it to the
        # dataset's own metric instead made the weight 1.0 point discontinuous
        # with 0.99. Nothing in the repo passed a one-component dict when this
        # changed on 2026-09-03.
        return _SINGLE_COMPONENT_OBJECTIVE[next(iter(objective))]
    total = float(sum(float(v) for v in objective.values()))
    return {k: float(v) / total for k, v in sorted(objective.items())}


def _objective_needs_pairs(objective) -> bool:
    """True when scoring this objective requires MCS pairs and row numbers."""
    if isinstance(objective, dict):
        return OBJECTIVE_STRUCT_FID in objective
    return objective == OBJECTIVE_STRUCT_FID


def _objective_metric_name(objective) -> str:
    """Metric name stored in cached metadata and used in cache filenames.

    Two different blends must never share a name — same filename, different
    quantity, and the second run silently reads the first one's answer.
    """
    if isinstance(objective, dict):
        parts = [f'{k}{v:g}' for k, v in sorted(objective.items())]
        return 'blend_' + '_'.join(parts)
    return _PLAIN_PRIMARY_KEY[objective]


def _blend_score(objective, metrics) -> float:
    """The score an objective SELECTS on, out of the numbers recorded on a fold.

    `metrics` is the dict built by the fold scorer. A missing key means the
    quantity was not computable on this fold (too few MCS pairs, no groups), and
    returns NaN — the same thing the caller saw before any of this was recorded.

    A dict `objective` is a blend: the weighted mean of its components. Any
    missing component makes the whole blend NaN rather than silently scoring the
    run on whichever part survived.
    """
    if isinstance(objective, dict):
        total = float(sum(objective.values()))
        acc = 0.0
        for name, w in objective.items():
            v = metrics.get(_BLEND_COMPONENT_KEY[name])
            if v is None or not np.isfinite(v):
                return float('nan')
            acc += float(w) * float(v)
        return acc / total if total else float('nan')
    v = metrics.get(_PLAIN_PRIMARY_KEY.get(objective))
    return float('nan') if v is None else float(v)


# Old name, kept so nothing that imported it breaks mid-refactor.
_geometry_primary = _blend_score


def _similarity_objective_score(X_val_s, y_val, is_clf, seed=42) -> float:
    """SAC (higher-is-better). NO LONGER CALLED — kept as the record.

    SAC was cut as a search objective on 2026-08-26; see the block above
    OBJECTIVE_PREDICTIVE for the evidence on both sides. Nothing in the search
    calls this. It stays so the quantity remains readable and so a future study
    that wants to REPORT SAC (rather than select by it) has one definition.

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


def _geometry_fold_mean(fold_scores):
    """Mean over the folds where the geometry score was DEFINED.

    A geometry score is NaN when the correlation could not be formed on that
    fold. np.mean would turn one such fold into a NaN for the whole
    representation, and NaN never loses a Python max(), so that representation
    would then win the budget split. Returns (mean, n_defined, n_total); the
    mean is NaN only when NO fold was defined.
    """
    vals = np.asarray([float(v) for v in (fold_scores or [])], dtype=np.float64)
    n_total = int(vals.size)
    finite = np.isfinite(vals)
    n_def = int(finite.sum())
    if n_def == 0:
        return float('nan'), 0, n_total
    return float(vals[finite].mean()), n_def, n_total


def _readacross_objective_score(X_val_s, y_val, groups_val, is_clf=False,
                                seed=42) -> float:
    """Activity read-across objective (higher-is-better) for objective='similarity'.

    Builds the held-out fold's pairwise Euclidean distance matrix from the
    (already-scaled) embedding and scores how activity-coherent its
    neighbourhoods are via scaffold-grouped, distance-weighted kNN read-across
    (continuous MODI in its regression form): the Spearman correlation between
    the kNN-predicted and true labels. No downstream model is fit.

    Re-implemented here (src must not import from tests/) from the reference
    tests/experiments/similarity_metrics/simeval.eval_knn_readacross — keep the
    two in sync.

    Returns NaN when the correlation is UNDEFINED — fewer than 10 held-out rows,
    fewer than 2 scaffold groups, constant labels, constant predictions, no
    groups supplied, or an exception. Until 2026-08-27 every one of those
    returned 0.0, which is also what a computed correlation of zero looks like,
    so a fold that could not be scored was indistinguishable from a
    representation with no neighbourhood structure at all.

    NaN NEVER LOSES A PYTHON max(). Every caller that averages or ranks these
    must use np.nanmean and must skip non-finite entries.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.neighbors import KNeighborsRegressor
    X = np.asarray(X_val_s, dtype=np.float64)
    y = np.asarray(y_val, dtype=np.float64).ravel()
    n = len(y)
    if n < 10 or X.shape[0] != n:
        return float('nan')
    if groups_val is None:
        return float('nan')
    g = np.asarray(groups_val).ravel()
    n_groups = len(np.unique(g))
    if n_groups < 2:
        return float('nan')
    try:
        # Pairwise Euclidean distance (matches simeval.dist_euclidean geometry).
        G = X @ X.T
        sq = np.diag(G)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * G, 0.0)
        D = np.sqrt(d2)
        n_splits = min(5, n_groups)
        gkf = GroupKFold(n_splits=n_splits)
        preds = np.full(n, np.nan)
        for tr, te in gkf.split(np.arange(n), y, g):
            kk = min(5, len(tr))
            if kk < 1:
                continue
            m = KNeighborsRegressor(n_neighbors=kk, weights='distance',
                                    metric='precomputed')
            m.fit(D[np.ix_(tr, tr)], y[tr])
            preds[te] = m.predict(D[np.ix_(te, tr)])
        ok = np.isfinite(preds)
        if ok.sum() < 3 or np.std(y[ok]) == 0 or np.std(preds[ok]) == 0:
            return float('nan')
        rho, _ = spearmanr(preds[ok], y[ok])
        return float(rho) if np.isfinite(rho) else float('nan')
    except Exception:
        return float('nan')


def _partial_spearman(a, b, control):
    """Spearman(a, b) with the rank-linear effect of `control` removed from both.

    Ranks all three, regresses the ranks of a and of b on the ranks of control,
    and correlates the residuals. Returns NaN when any input is constant.
    """
    from scipy.stats import rankdata
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(control)
    if np.std(ra) == 0 or np.std(rb) == 0 or np.std(rc) == 0:
        return float('nan')
    A = np.column_stack([np.ones_like(rc), rc])
    try:
        resid_a = ra - A @ np.linalg.lstsq(A, ra, rcond=None)[0]
        resid_b = rb - A @ np.linalg.lstsq(A, rb, rcond=None)[0]
    except np.linalg.LinAlgError:
        return float('nan')
    if np.std(resid_a) == 0 or np.std(resid_b) == 0:
        return float('nan')
    rho, _ = spearmanr(resid_a, resid_b)
    return float(rho) if np.isfinite(rho) else float('nan')


def _structural_fidelity_scores(X_sub_s, sub_idx_global, structural_pairs) -> dict:
    """Both structural-fidelity numbers for one fold.

    Returns {'struct_fidelity': float, 'struct_fidelity_size_controlled': float,
             'struct_fidelity_n_pairs': int}.

    'struct_fidelity' is the original quantity and its value is unchanged:
    Spearman between the hybrid's row-pair Euclidean distance and MCS distance.

    'struct_fidelity_size_controlled' is the same correlation with the pair's
    COMBINED HEAVY-ATOM COUNT partialled out of both sides. MCS distance is
    1 - shared/(a + b - shared), so it rises with how big the two molecules are
    together, and so does a Euclidean distance over count and descriptor blocks.
    Measured 2026-08-26 on the study's own cached pairs: Spearman(MCS distance,
    atoms_i + atoms_j) is 0.53 on herg_ki and 0.44 on logd, and removing it
    moves ecfp4:euclidean from 0.507 to 0.218 on herg_ki while
    ecfp4_count:tanimoto moves from 0.107 to 0.169 — they change places.
    Removing the size DIFFERENCE instead changes almost nothing (0.507 -> 0.506).

    NaN for the controlled number when the pair set carries no 'n_atoms' array
    (written before 2026-08-27) — unavailable, never silently uncontrolled.

    It is REPORTED ONLY. Nothing selects or ranks on it.
    """
    nan = float('nan')
    out = {'struct_fidelity': nan, 'struct_fidelity_size_controlled': nan,
           'struct_fidelity_n_pairs': 0}
    if structural_pairs is None or sub_idx_global is None:
        return out
    try:
        I = np.asarray(structural_pairs['I']).astype(int)
        J = np.asarray(structural_pairs['J']).astype(int)
        mcs = np.asarray(structural_pairs['mcs'], dtype=np.float64)
    except Exception:
        return out
    g = np.asarray(sub_idx_global).ravel().astype(np.int64)
    if g.size == 0:
        return out
    pos = np.full(int(g.max()) + 1, -1, dtype=np.int64)
    pos[g] = np.arange(g.size, dtype=np.int64)
    in_range_i = (I >= 0) & (I < pos.size)
    in_range_j = (J >= 0) & (J < pos.size)
    li = np.where(in_range_i, pos[np.clip(I, 0, pos.size - 1)], -1)
    lj = np.where(in_range_j, pos[np.clip(J, 0, pos.size - 1)], -1)
    mask = (li >= 0) & (lj >= 0)
    n_pairs = int(mask.sum())
    out['struct_fidelity_n_pairs'] = n_pairs
    if n_pairs < 10:
        return out
    X = np.asarray(X_sub_s, dtype=np.float64)
    dist = np.linalg.norm(X[li[mask]] - X[lj[mask]], axis=1)
    m = mcs[mask]
    if np.std(dist) == 0 or np.std(m) == 0:
        return out
    rho, _ = spearmanr(dist, m)
    out['struct_fidelity'] = float(rho) if np.isfinite(rho) else nan

    na = structural_pairs.get('n_atoms') if hasattr(structural_pairs, 'get') \
        else (structural_pairs['n_atoms'] if 'n_atoms' in structural_pairs else None)
    if na is None:
        return out
    na = np.asarray(na, dtype=np.float64)
    ii, jj = I[mask], J[mask]
    if ii.size == 0 or int(max(ii.max(), jj.max())) >= na.size:
        return out
    size_sum = na[ii] + na[jj]
    finite = np.isfinite(size_sum)
    if int(finite.sum()) < 10:
        return out
    out['struct_fidelity_size_controlled'] = _partial_spearman(
        dist[finite], m[finite], size_sum[finite])
    return out


def _structural_fidelity_score(X_sub_s, sub_idx_global, structural_pairs) -> float:
    """MCS structural-fidelity metric for objective='similarity' (higher-better).

    Spearman correlation between the hybrid's row-pair rep-space Euclidean
    distance and the precomputed MCS distance, over the structural pairs whose
    BOTH endpoints fall inside this fold. `structural_pairs` is a dict-like with
    integer arrays 'I','J' (indices into the FULL dataset row order) and 'mcs'.
    `sub_idx_global` are the GLOBAL indices of the rows in `X_sub_s` (in row
    order), so pairs can be remapped to fold-local positions.

    Re-implemented from tests/experiments/similarity_metrics/structural_yardstick
    (d_euclidean row-pair form + spearmanr vs mcs). Returns NaN when pairs are
    absent or fewer than ~10 survive the fold (not statistically meaningful),
    signalling 'skip structure'.
    """
    if structural_pairs is None or sub_idx_global is None:
        return float('nan')
    try:
        I = np.asarray(structural_pairs['I']).astype(int)
        J = np.asarray(structural_pairs['J']).astype(int)
        mcs = np.asarray(structural_pairs['mcs'], dtype=np.float64)
    except Exception:
        return float('nan')
    # Remap whole-dataset row numbers to positions in this fold, through an
    # inverse-index array rather than a dict and two comprehensions. Under
    # objective='struct_fid' this runs once per representation in the budget
    # split, once per allocation trial, and once per configuration per fold, so
    # a Python loop over every pair lands in the innermost path of the search.
    g = np.asarray(sub_idx_global).ravel().astype(np.int64)
    if g.size == 0:
        return float('nan')
    pos = np.full(int(g.max()) + 1, -1, dtype=np.int64)
    pos[g] = np.arange(g.size, dtype=np.int64)
    # A pair endpoint numbered beyond this fold's highest row simply is not in
    # the fold; clip the lookup and let the -1 mask drop it.
    in_range_i = (I >= 0) & (I < pos.size)
    in_range_j = (J >= 0) & (J < pos.size)
    li = np.where(in_range_i, pos[np.clip(I, 0, pos.size - 1)], -1)
    lj = np.where(in_range_j, pos[np.clip(J, 0, pos.size - 1)], -1)
    mask = (li >= 0) & (lj >= 0)
    if int(mask.sum()) < 10:
        return float('nan')
    X = np.asarray(X_sub_s, dtype=np.float64)
    a = X[li[mask]]
    b = X[lj[mask]]
    dist = np.linalg.norm(a - b, axis=1)
    m = mcs[mask]
    if np.std(dist) == 0 or np.std(m) == 0:
        return float('nan')
    rho, _ = spearmanr(dist, m)
    return float(rho) if np.isfinite(rho) else float('nan')


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
        # 'blend_similarity0.75_struct_fid0.25' and friends are minted per run
        # by _objective_metric_name, so they cannot be table entries. Same
        # guard, same direction as the two geometry names they blend.
        if metric.startswith('blend_'):
            return _sac_not_a_prediction_metric, False
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
        from kirby.hybrid import lgbm_n_jobs
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=100, random_state=42, n_jobs=lgbm_n_jobs(), verbose=-1)
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, max_features='sqrt',
                                     random_state=42, n_jobs=-1)


def _get_lgbm_classifier():
    try:
        from kirby.hybrid import lgbm_n_jobs
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=100, random_state=42, n_jobs=lgbm_n_jobs(), verbose=-1)
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
    'ridge': {
        # Linear baseline retained ONLY for nucleotide datasets (nt_*/gb_*),
        # where classical linear models are competitive with trees on one-hot/
        # k-mer reps. tier 'extra' keeps it out of every models=None default;
        # _resolve_models injects it for nucleotide only. Do NOT use elsewhere.
        'regressor': lambda: __import__('sklearn.linear_model', fromlist=['Ridge']).Ridge(alpha=1.0),
        'classifier': lambda: __import__('sklearn.linear_model', fromlist=['RidgeClassifier']).RidgeClassifier(alpha=1.0),
        'tier': 'extra',
    },
}

# Model-inclusion categories (NOT a speed tier). 'full' models are the default
# set auto-selected by _resolve_models(None); 'extra' models (e.g. ridge) are
# only used when explicitly requested by name or by the nucleotide ridge inject.
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
    except Exception as e:
        # Was a bare `return budgets`. A policy that raised looked exactly like
        # a policy that changed nothing, so a dataset ran at the wrong budget
        # list with no sign of it.
        print(f"  !! budget policy raised for {dataset_name!r}: "
              f"{type(e).__name__}: {e} — running at the caller's budgets "
              f"{sorted(budgets)}.")
        return budgets
    # Fall back to the original list if the policy filtered everything out
    # (e.g. a single native-dim budget above the cap) — never zero budgets.
    out = list(out) if out else budgets
    _dropped = [b for b in budgets if b not in out]
    _added = [b for b in out if b not in budgets]
    if _dropped or _added:
        print(f"  budget policy for {dataset_name!r}: "
              + (f"dropped {sorted(_dropped)}; " if _dropped else "")
              + (f"added {sorted(_added)}; " if _added else "")
              + f"searching {sorted(out)}.")
    return out


def _resolve_models(models, task_type: str,
                     n_samples: Optional[int] = None,
                     dataset_name: Optional[str] = None,
                     strategy: Optional[str] = None) -> Dict[str, Any]:
    """Resolve model spec to dict of {name: sklearn_model_instance}.

    models can be:
    - None: use the default ('full'-category) model set
    - List[str]: lookup by name
    - Dict[str, model]: custom models (passthrough)
    - Single model instance: wrap as {'custom': model}

    Models with _max_samples are auto-skipped when n_samples exceeds the limit.
    """
    key = 'classifier' if task_type == 'classification' else 'regressor'
    max_tier = _TIER_ORDER['full']  # default set = non-'extra' models
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
        # Ridge is a linear baseline kept ONLY for nucleotide datasets (it sits
        # at tier 'extra' so it is never auto-selected anywhere else). Inject it
        # here for nt_*/gb_* only; _check_skip still lets the skip policy veto it.
        if dataset_name and dataset_name.startswith(('nt_', 'gb_')):
            if 'ridge' not in resolved and _check_skip('ridge'):
                resolved['ridge'] = DEFAULT_MODELS['ridge'][key]()
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
    n_iter: int = 20,
    n_folds: int = 3,
    seed: int = 42,
    model_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Tune model hyperparameters via RandomizedSearchCV on concatenated reps.

    Runs RandomizedSearchCV for each requested model
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
        n_iter: Max RandomizedSearchCV iterations per model.
        model_names: Tune only these models. find_best_hybrid passes the list
            it actually resolved. None tunes every default, which is what a
            direct caller used to get whether it wanted it or not.
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

    # Resolve which models to tune (pass n_samples for size guards).
    # model_names restricts it to the models the CALLER will actually search.
    # Without that this tuned all ten defaults even for models=['knn'], and
    # RandomizedSearchCV on svm/mlp/catboost over a 15k-wide concatenation cost
    # far more than the search it was preparing. None keeps the old full list.
    default_models = _resolve_models(None, task_type,
                                      n_samples=len(y))
    if model_names is not None:
        _want = set(model_names)
        default_models = {k: v for k, v in default_models.items() if k in _want}

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
# DEFAULT SEARCH CONFIGURATION
# =============================================================================

# Search defaults as flat module constants. There is ONE search configuration;
# cheaper searches go through the multi-stage adaptive_pruning_schedule (see
# PRUNING_MODES), which keeps the cache key identical to production rather than
# diverging it via a separate configuration.
# DEFAULT_ALLOCATION_METHODS is where the allocation-strategy set is configured.
DEFAULT_IMPORTANCE_METHODS = ['deeplift', 'lightgbm_gain', 'treeshap']
# The five surviving allocations, chosen 2026-09-01 from the c32 cross sweep
# (weighting x mechanism x N; see CHANGES_BEFORE_RERUN.md "Allocation set cut to
# five"). Each is one (weighting x mechanism) pair; the number-of-reps knob is a
# SEPARATE axis, swept elsewhere, and is deliberately NOT encoded here (that is
# why top_2/top_3/top_5 are gone — they welded a rep-count into the weighting
# slot). wglobal_power2 is the one global-pooling arm that stayed competitive;
# the four others are per-rep quota. See the EXTRA_ALLOCATION_METHODS note below
# for the cache caveat wglobal_power2 carries.
DEFAULT_ALLOCATION_METHODS = [
    'performance_weighted',   # linear      | quota
    'power_2',                # power_law^2 | quota
    'softmax_T1',             # softmax T=1 | quota
    'rank_exp_d0.3',          # rank_exp d=0.3 | quota
    'wglobal_power2',         # power_law^2 | global (weighted_global)
]
DEFAULT_INTERNAL_EVALUATORS = ['match', 'rf', 'svm', 'lightgbm', 'mlp']
DEFAULT_N_FOLDS = 5

# Remaining global-pooling variants, IMPLEMENTED in the engine (create_hybrid /
# hybrid.py) and callable through the search via `allocation_methods=[...]`, but
# NOT in the default set above. They are opt-in: adding one enlarges EVERY
# default search's config space, so experiments request them explicitly, e.g.
#     find_best_hybrid(..., allocation_methods=['weighted_global'])
# or _run_search(..., allocation_methods=['weighted_global']).
# (The c32-chosen global arm, 'wglobal_power2', was promoted INTO the default
# set above on 2026-09-01; these are the ones that did NOT survive that cut.)
#
# 'weighted_global' = global feature-pooling allocator (hybrid.py
# allocate_weighted_global). It sizes each representation's slice of a feature
# candidate pool by that rep's performance weight (same weighting families as
# the quota allocators), glues the per-rep candidates into ONE pool, then runs a
# SINGLE global importance ranking over the whole pool to pick the budget —
# replacing the per-rep quota that gives each rep a fixed slice. Extra params
# (weighting_scheme, weighting_params, pool_total_mult) ride in through the
# NAMED_ALLOCATIONS extra dict in hybrid.py; this file needs no signature change
# to thread them. NOTE: no weighted_global name starts with 'top_', so it
# bypasses the hybrid_eval persistent cache (find_best_hybrid.py:2787) by design
# — safe (no mis-cache), but every fold recomputes (no cross-invocation dedup).
# ★ This caveat now applies to a DEFAULT method too ('wglobal_power2'): its
# global arm recomputes each fold rather than reusing the persistent cache.
EXTRA_ALLOCATION_METHODS = ['weighted_global']


def _validate_allocation_names(names: Sequence[str]) -> None:
    """Raise unless every name dispatches in create_hybrid.

    Valid = a key of hybrid.NAMED_ALLOCATIONS, or one of the five base names
    create_hybrid branches on directly.
    """
    from kirby.hybrid import NAMED_ALLOCATIONS
    valid = set(NAMED_ALLOCATIONS) | {
        'greedy', 'performance_weighted', 'fixed', 'dominant_greedy',
        'weighted_global',
    }
    unknown = [n for n in names if n not in valid]
    if unknown:
        raise ValueError(
            f"Unknown allocation_methods {unknown}. The default set is "
            f"{DEFAULT_ALLOCATION_METHODS}; every other dispatchable name is a "
            f"key of hybrid.NAMED_ALLOCATIONS "
            f"({', '.join(sorted(NAMED_ALLOCATIONS))}).")


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
    # How many allocation methods score a point per group in the end-of-fold-0
    # strategy prune. 3 = the historical hardcoded value. Lower it to prune a
    # short allocation list: with 3 methods searched and this at 3, every method
    # places every time and nothing is ever dropped.
    'strategy_top_n': 3,
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
        out['strategy_top_n'] = max(1, int(out.get('strategy_top_n', 3)))
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
    merged['strategy_top_n'] = max(1, int(merged.get('strategy_top_n', 3)))
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
    _factory_failures = []
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
                _factory_failures.append(
                    (rep_name, fold_idx, f'{type(e).__name__}: {e}'))

    if verbose and (n_cache_hits or n_cache_misses):
        print(f"  FT embeddings: {n_cache_hits} cache hits, "
              f"{n_cache_misses} computed + cached")

    # A factory that raised used to leave that ONE fold without that
    # representation, silently under verbose=False. The search then averaged
    # folds built from different representation pools and reported the mean as
    # one number. Refuse instead: a per-fold representation either exists on
    # every fold or the run is not comparable.
    if _factory_failures:
        _lines = '; '.join(f'{r} on fold {f}: {m}'
                           for r, f, m in _factory_failures[:8])
        raise RuntimeError(
            f"{len(_factory_failures)} per-fold representation build(s) failed "
            f"across {n_folds} folds. A fold missing a representation makes the "
            f"fold mean an average over different representation pools, so this "
            f"is refused rather than skipped. Failures: {_lines}"
            + (' ...' if len(_factory_failures) > 8 else ''))

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
        from kirby.hybrid import lgbm_n_jobs
        if task_type == 'classification':
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(n_estimators=50, max_depth=5,
                                    random_state=42, n_jobs=lgbm_n_jobs(), verbose=-1)
        else:
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(n_estimators=50, max_depth=5,
                                   random_state=42, n_jobs=lgbm_n_jobs(), verbose=-1)
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


def _tune_on_training_fold(model, X_train, y_train, task_type,
                           groups=None, n_iter: int = 20, seed: int = 42,
                           model_name: Optional[str] = None):
    """Tune one estimator on the rows and FEATURES it is about to be fitted on.

    Every score KIRBy reports comes from a model fitted on ONE feature matrix:
    a single representation for a baseline, or the hybrid's selected features
    for a hybrid cell. tune_models tunes on the hstack of every base
    representation instead, so before 2026-09-03 a tuned hyperparameter was
    chosen on a 15k-wide matrix the model is never fitted on. This tunes on
    the matrix that is actually used.

    X_train / y_train are the TRAINING rows of the current fold, so the
    validation rows this estimator is scored on are not seen here.

    Returns a new estimator with the chosen parameters, or the original when
    there is no search space, too few rows, or the search raises. Never
    fatal — an untuned estimator is a worse estimator, not a wrong one.
    """
    from sklearn.model_selection import RandomizedSearchCV, GroupKFold, KFold
    from sklearn.pipeline import Pipeline as _Pipeline

    name = model_name or _model_name_for_search_space(model)
    grid = _HP_SEARCH_SPACES.get(name)
    if grid is None or model is None:
        return model
    valid = model.get_params().keys() if hasattr(model, 'get_params') else set()
    grid = {k: v for k, v in grid.items() if k in valid}
    if not grid:
        return model

    n_tune_folds = 3
    if len(y_train) < n_tune_folds * 2:
        return model

    cv_args = {}
    if groups is not None and len(np.unique(groups)) >= n_tune_folds:
        cv = GroupKFold(n_splits=n_tune_folds)
        cv_args['groups'] = groups
    else:
        cv = KFold(n_splits=n_tune_folds, shuffle=True, random_state=seed)

    scoring = 'accuracy' if task_type == 'classification' else 'r2'
    try:
        pipe = _Pipeline([('scaler', StandardScaler()),
                          ('model', deepcopy(model))])
        search = RandomizedSearchCV(
            pipe, {f'model__{k}': v for k, v in grid.items()},
            n_iter=min(n_iter, _count_grid_size(grid)), cv=cv, scoring=scoring,
            random_state=seed, n_jobs=1, error_score=np.nan)
        search.fit(X_train, y_train, **cv_args)
        best = search.best_estimator_.named_steps['model']
    except Exception:
        return model
    if 'LGBM' in type(best).__name__:
        best.set_params(verbose=-1)
    elif hasattr(best, 'verbose'):
        try:
            best.set_params(verbose=0)
        except Exception:
            pass
    # Carry the stamp so _set_knn_k keeps the k this search chose. Without it
    # the k is overwritten with 5 two lines later and the tuning is discarded.
    for _attr in (KNN_TUNED_K_ATTR,):
        if getattr(model, _attr, False):
            setattr(best, _attr, True)
    return best


def _model_name_for_search_space(model) -> Optional[str]:
    """Map an estimator to its _HP_SEARCH_SPACES key by class name."""
    if model is None:
        return None
    c = type(model).__name__.lower()
    if 'randomforest' in c:
        return 'rf'
    if 'xgb' in c:
        return 'xgboost'
    if 'lgbm' in c or 'lightgbm' in c:
        return 'lightgbm'
    if 'kneighbors' in c:
        return 'knn'
    if 'mlp' in c:
        return 'mlp'
    if c.startswith('sv'):
        return 'svm'
    if 'gaussianprocess' in c:
        return 'gp'
    if 'catboost' in c:
        return 'catboost'
    return None


# Attribute stamped on a KNN estimator whose n_neighbors was chosen by
# tune_models and must therefore survive to the fit. It rides on the object,
# so it reaches every scorer that deep-copies the model — including the 21
# _evaluate_allocation sites in hybrid.py — without threading a flag through
# any of them. deepcopy preserves it; sklearn's clone would not, and nothing
# on this path clones.
#
# MIRRORED in hybrid.py, which by convention imports nothing from this file.
# The two strings must stay equal or hybrid.py's own KNN evaluator falls back
# to k=5 and a tuned run reads as an untuned one.
KNN_TUNED_K_ATTR = '_kirby_tuned_k'


def _set_knn_k(m, n_train: int) -> None:
    """Choose n_neighbors on a KNN estimator immediately before it is fitted.

    Default: k = min(5, sqrt(n_train)). That is exactly 5 for any training
    fold of 25 rows or more, so it is the constant behind every KNN number
    KIRBy has reported. Unchanged unless the estimator is stamped.

    Stamped (find_best_hybrid(tune=True, tune_knn_k=True)): the n_neighbors
    tune_models chose is kept, clamped to n_train - 1 so the fit cannot raise.

    No-op on any estimator without n_neighbors.
    """
    if not (hasattr(m, 'n_neighbors') and hasattr(m, 'set_params')):
        return
    if getattr(m, KNN_TUNED_K_ATTR, False):
        k = int(getattr(m, 'n_neighbors', 5) or 5)
    else:
        k = min(5, max(1, int(np.sqrt(n_train))))
    # Clamp is a no-op on the default path: k is at most 5 and at most
    # sqrt(n_train), so k <= n_train - 1 already for every n_train >= 2.
    k = max(1, min(k, max(1, n_train - 1)))
    m.set_params(n_neighbors=k)


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
    tune_per_fold: bool = False,
    tune_n_iter: int = 20,
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
        _fg = groups[train_idx] if groups is not None else None
        if tune_per_fold:
            # Tune on THIS representation's training rows, not on the hstack
            # of every representation. This is the matrix m is fitted on.
            m = _tune_on_training_fold(m, X_train_s, y_train, task_type,
                                       groups=_fg, n_iter=tune_n_iter,
                                       seed=seed)
        _set_knn_k(m, len(train_idx))
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
    tune_per_fold: bool = False,
    tune_n_iter: int = 20,
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
    candidate_ranking: str = 'residual_correlation',
    lower_is_better: bool = False,
    fold_reps: Optional[Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = None,
    precomputed_folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    cv_splitter=None,
    importance_cache: Optional[Dict] = None,
    alloc_cache: Optional[Dict] = None,
    selection_dedup: Optional[Dict] = None,
    greedy_tolerance: float = 0.0,
    augmentation_margin: float = 0.0,
    augmentation_fraction: Optional[float] = None,
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
    row_index: Optional[np.ndarray] = None,
    structural_pairs: Optional[dict] = None,
) -> dict:
    """Evaluate a single hybrid configuration. Returns result dict.

    row_index / structural_pairs: MCS structural fidelity inside the search.
    `structural_pairs` is {'I','J','mcs'}, where I and J are row numbers into
    the WHOLE dataset. `row_index` gives the whole-dataset row number of each
    row of `base_reps`, in order. Under nested CV the search only sees the
    outer-training rows and renumbers them from 0, so without `row_index` a
    stored pair "molecule 4012 against molecule 88" cannot be located and MCS
    fidelity was computable only at the outer test fold. With it, each fold's
    validation rows are mapped back to whole-dataset numbers and structural
    fidelity rides along as a per-fold secondary metric named
    'struct_fidelity'. Same trick nested_cv_evaluate already uses to give the
    per-fold representation factories whole-dataset row numbers.

    Supplying them does NOT change which configuration wins -- the returned
    score is still the objective's own statistic. Structure is recorded, not
    optimised.

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

    # Identity of the MCS pair set, so a run that records structural fidelity
    # and a run that does not cannot be served each other's cached folds.
    _struct_pairs_id = None
    if structural_pairs is not None and row_index is not None:
        try:
            _struct_pairs_id = _hash_bytes(
                np.ascontiguousarray(
                    np.asarray(structural_pairs['I'], dtype=np.int64)).tobytes()
                + np.ascontiguousarray(
                    np.asarray(structural_pairs['J'], dtype=np.int64)).tobytes()
                + np.ascontiguousarray(
                    np.asarray(row_index, dtype=np.int64)).tobytes())
        except Exception:
            _struct_pairs_id = 'unhashable'

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
    # Which extra features each fold actually kept: {fold: {donor: [indices]}}.
    # Nothing recorded this before, so a winning recipe could not be explained
    # or checked afterwards -- the stored recipe named the strategy and the
    # per-donor cap, never the features that survived. Recorded as DATA on the
    # blueprint, not as part of the cache key, so no stored result changes
    # identity and no result database has to be rebuilt.
    fold_kept_aug = {}
    # {fold: {rep_name: n_features}} — the base budget split this fold used.
    fold_allocation = {}
    any_imputation = False

    _eval_fold_indices = range(len(fold_splits)) if fold_indices is None else fold_indices

    # Precompute shared cache-key components for this call.
    # Only built when the hybrid_eval cache is actually active — every
    # other call path short-circuits _hybrid_cache_enabled below.
    _sim_mode = _is_geometry_objective(objective)
    # A blend that contains 'predictive' is neither: a model IS fitted, and its
    # prediction is then combined with the geometry numbers on the same rows.
    _blend_with_model = _blend_needs_model(objective)
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
        # A BLEND must not use them either, for the same reason and with a
        # sharper failure: allocate_performance_weighted prefers
        # precomputed_scores over its own blend branch, so a blend handed
        # precomputed scores would split its budget by PREDICTION ALONE while
        # reporting itself under the blend's name. Nothing would raise.
        fold_alloc_scores = None
        if alloc_cache is not None and not _sim_mode and not _blend_with_model:
            if metric_name:
                fold_alloc_scores = alloc_cache.get((fold_idx, internal_evaluator, metric_name))
            if fold_alloc_scores is None:
                fold_alloc_scores = alloc_cache.get((fold_idx, internal_evaluator))

        # Persistent hybrid_eval cache — probe before compute.
        # Dedup applies when `alloc` is top_k and the predicted subset
        # equals another run's subset (e.g. two conditions whose top_3
        # both pick the same 3 reps). Miss cases pay only the
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
                # Feature width is part of the subset's identity — see
                # _subset_name_hex. Taken from the live matrices, so it reflects
                # what this run will actually evaluate.
                _subset_widths = {
                    n: int(base_reps[n].shape[1])
                    for n in _predicted_subset if n in base_reps
                }
                _subset_hex = _subset_name_hex(_predicted_subset, _subset_widths)
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
                    objective=objective,
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
                    augmentation_margin=augmentation_margin,
                    augmentation_fraction=augmentation_fraction,
                    augmentation_threshold=augmentation_threshold,
                    augmentation_budget=augmentation_budget,
                    aug_per_source_budget=aug_per_source_budget,
                    metric_name=metric_name, task_type=task_type,
                    subset_widths=_subset_widths,
                    struct_pairs_id=_struct_pairs_id,
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
                augmentation_margin=augmentation_margin,
                augmentation_fraction=augmentation_fraction,
                groups=fold_groups,
                objective=objective,
                is_clf=_is_clf_task,
                # Whole-dataset row numbers of THIS fold's training rows, so a
                # struct_fid objective can split the budget by MCS fidelity
                # instead of by whatever else was to hand.
                row_index=(np.asarray(row_index)[train_idx]
                           if row_index is not None else None),
                structural_pairs=structural_pairs,
            )

            # How many features each representation actually got, this fold.
            # create_hybrid computes this and puts it on feature_info; nothing
            # read it, so no stored result could answer the question the whole
            # rebuild roadmap is about — which representations contributed, how
            # many each, and whether the best one was truncated to dust by the
            # per-representation quota. The augmentation side has been recorded
            # since 2026-08-26 (kept_augmentations); this is the same treatment
            # for the base split. DATA, not part of any cache key: recording it
            # changes no stored result's identity.
            _alloc_here = feature_info.get('allocation') or {}
            if not _alloc_here:
                # Not every allocator writes 'allocation'; count the columns
                # each representation actually contributed instead.
                _alloc_here = {
                    _rn: len(np.asarray(_ri.get('selected_indices', [])))
                    for _rn, _ri in feature_info.items()
                    if isinstance(_ri, dict) and 'selected_indices' in _ri
                }
            if _alloc_here:
                fold_allocation[int(fold_idx)] = {
                    str(_rn): int(_c) for _rn, _c in _alloc_here.items()}

            # Record which extra features this fold kept, per donor, before
            # anything downstream can overwrite feature_info.
            _kept_here = feature_info.get('augmentation_info') or {}
            if _kept_here:
                fold_kept_aug[int(fold_idx)] = {
                    str(_dn): [int(i) for i in
                               np.asarray(_di.get('selected_indices', []))]
                    for _dn, _di in _kept_here.items()
                }
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
                        # No honest source for this augmentation's validation
                        # rows. Falling back to the full base representation was
                        # silently WRONG: an auto-demoted rep was trained on its
                        # top-importance columns and would then be validated on
                        # the FIRST columns of the same rep, because the stored
                        # indices are positions into the demoted block and the
                        # bounds check passes on the wider matrix.
                        raise KeyError(
                            f"augmentation {name!r} has no validation-side "
                            f"source: it is not in the augmentation dict and "
                            f"carries no full-length array. Its training "
                            f"columns cannot be located on the held-out rows.")
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
            # Geometry objective: score the held-out fold GEOMETRY directly
            # (higher-is-better). No downstream model is fit, so
            # _predict_for_metric / _compute_classic_metrics are skipped —
            # there is no y_pred. Selecting features on train (inside
            # create_hybrid) and scoring on val keeps this honest (not the
            # in-sample, definitional score of the older s02 path).
            #   similarity  -> each held-out molecule's property predicted from
            #                  its five nearest neighbours, Spearman'd against
            #                  truth, scaffold-grouped.
            #   struct_fid  -> Spearman between pairwise distance and MCS
            #                  distance.
            #   a dict      -> weighted mean of the two.
            # BOTH numbers are recorded on every geometry fold; _geometry_primary
            # says which one the objective selects on. MCS used to be impossible
            # here: the pairs store row numbers into the whole dataset, the
            # search renumbers each fold's rows from 0, and nothing carried the
            # original numbers in, so a pair could not be located. `row_index`
            # carries them (2026-08-26).
            _g_val = groups[val_idx] if groups is not None else None
            _fold_metrics = {}
            _ra = _readacross_objective_score(
                X_val_s, y_val, _g_val, _is_clf_task, seed=seed)
            if np.isfinite(_ra):
                _fold_metrics['readacross'] = float(_ra)
            if structural_pairs is not None and row_index is not None:
                _sfd = _structural_fidelity_scores(
                    X_val_s, np.asarray(row_index)[val_idx], structural_pairs)
                _sf = _sfd['struct_fidelity']
                if np.isfinite(_sf):
                    _fold_metrics['struct_fidelity'] = float(_sf)
                # Reported beside it, never selected on: the same correlation
                # with the pair's combined heavy-atom count partialled out.
                _sfc = _sfd['struct_fidelity_size_controlled']
                if np.isfinite(_sfc):
                    _fold_metrics['struct_fidelity_size_controlled'] = float(_sfc)
            score = _geometry_primary(objective, _fold_metrics)
            fold_scores.append(score)
            fold_n_features.append(X_train.shape[1])
            fold_metric_dicts.append(_fold_metrics)
            if selection_dedup is not None and sel_hash is not None:
                selection_dedup[(fold_idx, model_name, sel_hash)] = (
                    score, X_train.shape[1], _fold_metrics)
            continue

        m = deepcopy(model)
        _fg = groups[train_idx] if groups is not None else None
        if tune_per_fold:
            # Tune on the hybrid's SELECTED features for this fold — the
            # budget-many columns m is about to be fitted on.
            m = _tune_on_training_fold(m, X_train_s, y_train, task_type,
                                       groups=_fg, n_iter=tune_n_iter,
                                       seed=seed, model_name=model_name)
        _set_knn_k(m, len(train_idx))
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

        if _blend_with_model:
            # A blend containing 'predictive'. The model above supplied the
            # prediction; the other two components are read off the SAME
            # held-out rows, and the fold's score is the weighted mean. The
            # dataset's own metric cannot go into that mean — RMSE against two
            # correlations — so the prediction component is its correlation
            # form. The raw metric stays in _fold_metrics either way.
            _fold_metrics['predictive_corr'] = _prediction_correlation(
                y_val, y_pred, task_type)
            _g_val = groups[val_idx] if groups is not None else None
            _ra = _readacross_objective_score(
                X_val_s, y_val, _g_val, _is_clf_task, seed=seed)
            if np.isfinite(_ra):
                _fold_metrics['readacross'] = float(_ra)
            if structural_pairs is not None and row_index is not None:
                _sfd = _structural_fidelity_scores(
                    X_val_s, np.asarray(row_index)[val_idx], structural_pairs)
                _sf = _sfd['struct_fidelity']
                if np.isfinite(_sf):
                    _fold_metrics['struct_fidelity'] = float(_sf)
                # Reported beside it, never selected on: the same correlation
                # with the pair's combined heavy-atom count partialled out.
                _sfc = _sfd['struct_fidelity_size_controlled']
                if np.isfinite(_sfc):
                    _fold_metrics['struct_fidelity_size_controlled'] = float(_sfc)
            score = _blend_score(objective, _fold_metrics)
            fold_scores[-1] = score

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

    # How many folds this configuration actually scored, and how many it was
    # asked for. A fold that raised is skipped above, so score_mean can be a
    # mean over 3 folds sitting in the same ranking as a mean over 5. Nothing
    # used to record the difference. Recorded here and carried onto the stored
    # row, so a reader can exclude a short configuration; the search itself
    # still ranks them together (changing that is a policy decision, written up
    # in FRAMEWORK_FINALIZATION.md).
    _n_folds_scored = len(fold_scores)
    # fold_indices restricts the pass (the fold-0 pruning sweep asks for one
    # fold on purpose); "asked" is what THIS call requested, not the split count.
    _n_folds_asked = (len(fold_splits) if fold_indices is None
                      else len(list(fold_indices)))
    if _n_folds_scored < _n_folds_asked:
        print(f"  !! {model_name}/{allocation_method}/{importance_method}/"
              f"b={budget}/ev={internal_evaluator}: scored "
              f"{_n_folds_scored}/{_n_folds_asked} folds — the mean below is "
              f"over those {_n_folds_scored}, and is ranked against "
              f"{_n_folds_asked}-fold means.")

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
        augmentation_margin=augmentation_margin,
        augmentation_fraction=augmentation_fraction,
        # All three are columns of the stored result's KEY, so the search files
        # two values as two rows. Recording them here is what makes a rebuild
        # reproduce the row: without them build_fold_hybrid fell back to
        # create_hybrid's defaults ('residual_correlation', None, 20) whatever
        # the search ran under.
        candidate_ranking=candidate_ranking,
        augmentation_threshold=augmentation_threshold,
        augmentation_budget=augmentation_budget,
        # The search always ranks against a 50-long candidate list — it has no
        # parameter for this and deliberately gains none, because the length is
        # NOT in the key and two searches at different lengths would overwrite
        # each other's row. Recorded so the value is on the recipe rather than
        # implied.
        aug_max_candidates=50,
        metric=metric_name or 'r2',
        lower_is_better=lower_is_better,
        objective=objective,
        kept_augmentations=(fold_kept_aug or None),
        allocation_per_fold=(fold_allocation or None),
    )

    return {
        'blueprint': blueprint,
        'model': model_name,
        # Under a geometry objective a fold's correlation can be UNDEFINED
        # (NaN) — too few held-out rows, one scaffold group, constant labels.
        # np.mean turns one such fold into a NaN for the whole configuration,
        # while the baseline arm has always used _geometry_fold_mean and kept
        # the folds that WERE defined. The two arms were averaging differently,
        # so a hybrid lost a comparison it never actually lost. Same rule on
        # both sides now. Predictive runs are unaffected: their fold scores are
        # finite, and _geometry_fold_mean over finite values IS np.mean.
        'score_mean': (float(np.mean(fold_scores)) if not _sim_mode
                       else _geometry_fold_mean(fold_scores)[0]),
        'score_std': float(np.std(fold_scores)),
        'fold_scores': [float(s) for s in fold_scores],
        'n_folds_scored': _n_folds_scored,
        'n_folds_asked': _n_folds_asked,
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
    tune_per_fold: bool = False,
    tune_n_iter: int = 20,
    # The MCS pair set and the whole-dataset row number of each row. Needed
    # ONLY under objective='struct_fid' (or a blend containing it): without
    # them this function scored every representation by nearest-neighbour
    # activity and filed the number under structural fidelity, so the baseline
    # arm and the hybrid arm were two different quantities.
    structural_pairs: Optional[dict] = None,
    row_index: Optional[np.ndarray] = None,
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

    # Which geometry quantity a baseline is scored on. Both branches below used
    # to call the read-across scorer unconditionally, so under
    # objective='struct_fid' every per-representation baseline held a
    # nearest-neighbour activity correlation while the hybrid arm held an MCS
    # correlation -- and the two were subtracted from each other.
    _geo_components = None
    if _is_geometry_objective(objective):
        _geo_components = (dict(objective) if isinstance(objective, dict)
                           else {objective: 1.0})
        if OBJECTIVE_STRUCT_FID in _geo_components and (
                structural_pairs is None or row_index is None):
            raise ValueError(
                "_evaluate_baselines: objective %r scores MCS structural "
                "fidelity, which needs structural_pairs ({'I','J','mcs'}) and "
                "row_index (whole-dataset row number of each row). Without "
                "them the baselines silently held nearest-neighbour activity "
                "scores instead, which is a different quantity from the one "
                "the hybrid arm was scored on." % (objective,))

    def _geo_fold_score(X_val_scaled, val_idx, _is_clf):
        """One fold's geometry score under the objective actually requested."""
        _parts = {}
        if OBJECTIVE_SIMILARITY in _geo_components:
            _g_va = groups[val_idx] if groups is not None else None
            _parts[OBJECTIVE_SIMILARITY] = _readacross_objective_score(
                X_val_scaled, labels[val_idx], _g_va, _is_clf, seed=seed)
        if OBJECTIVE_STRUCT_FID in _geo_components:
            _parts[OBJECTIVE_STRUCT_FID] = _structural_fidelity_score(
                X_val_scaled, np.asarray(row_index)[val_idx], structural_pairs)
        if len(_parts) == 1:
            return float(next(iter(_parts.values())))
        # A blend of the two geometry components. Any component undefined on
        # this fold makes the fold undefined, the same rule _blend_score uses.
        _tot = float(sum(_geo_components.values()))
        _acc = 0.0
        for _k, _w in _geo_components.items():
            _v = _parts.get(_k, float('nan'))
            if not np.isfinite(_v):
                return float('nan')
            _acc += float(_w) * float(_v)
        return _acc / _tot if _tot else float('nan')

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
        # A representation wider than HIGH_DIM_FILTER_THRESHOLD is not scored at
        # its full width: _evaluate_single fits a per-fold LightGBM importance
        # filter and keeps 500 features (200 under 1000 rows). 'dimension' still
        # reported the full width, so a table read "esm3, 5120 features" for a
        # score measured on 500. The width actually scored is recorded beside it
        # -- absent when no filter fired, so a full-width baseline is unchanged.
        if X.shape[1] > HIGH_DIM_FILTER_THRESHOLD:
            _scored_w = 500 if len(labels) >= 1000 else 200
            if _scored_w < X.shape[1]:
                baselines[rep_name]['dimension_scored'] = int(_scored_w)
                if verbose:
                    print(f"  {rep_name}: {X.shape[1]} features, scored on the "
                          f"top {_scored_w} per fold (high-dimension filter). "
                          f"This is NOT a full-dimension baseline.")
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
                    objective=objective,
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

            # --- Geometry objective: per-rep baseline (no model fit) ---
            # Model-independent — score the rep's own held-out geometry per fold
            # and average. This is also the "best single rep by geometry"
            # comparator. Avoids invoking the geometry metric guard.
            #   similarity -> activity read-across.
            if _is_geometry_objective(objective):
                _is_clf_b = (task_type == 'classification')
                _spl, _use_g = _make_cv_splitter(groups, n_folds, seed,
                                                 cv_splitter=cv_splitter,
                                                 task_type=task_type)
                _sargs = (X, labels, groups) if _use_g else (X, labels)
                _sac_folds = []
                for _tr, _va in _spl.split(*_sargs):
                    _sc = StandardScaler().fit(X[_tr])
                    _sac_folds.append(_geo_fold_score(
                        _sc.transform(X[_va]), _va, _is_clf_b))
                _m, _nd, _nt = _geometry_fold_mean(_sac_folds)
                rounded_score = round(_m, 4) if np.isfinite(_m) else float('nan')
                if _nd < _nt and verbose:
                    print(f"    {rep_name}: geometry score defined on {_nd}/{_nt} "
                          f"folds; the mean is over those {_nd}")
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
                tune_per_fold=tune_per_fold,
                tune_n_iter=tune_n_iter,
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

            # --- Geometry objective: per-fold factory baseline (no model fit) ---
            # The base-rep loop above has this branch; without the same branch
            # here every factory rep fell through to _fit_model + metric_fn, and
            # under a geometry objective metric_fn is the deliberate raising
            # guard (_sac_not_a_prediction_metric). That made every factory rep
            # — including all nine structure-tuned sim_* recipes, which ARE
            # factory reps — crash the run under objective='similarity'. Scores the fold's own val embedding geometry and
            # averages, exactly as the base-rep branch does. Model-independent,
            # so it is computed once and written under every model key.
            if _is_geometry_objective(objective):
                _is_clf_f = (task_type == 'classification')
                _geo_folds = []
                for fold_idx, (train_idx, val_idx) in enumerate(precomputed_folds):
                    if fold_idx not in fold_reps or rep_name not in fold_reps[fold_idx]:
                        continue
                    ft_train, ft_val = fold_reps[fold_idx][rep_name]
                    _sc = StandardScaler().fit(ft_train)
                    _Xt, _Xv, _ = _impute_nan(_sc.transform(ft_train),
                                              _sc.transform(ft_val))
                    _geo_folds.append(_geo_fold_score(_Xv, val_idx, _is_clf_f))
                if _geo_folds:
                    _gm, _gnd, _gnt = _geometry_fold_mean(_geo_folds)
                    _geo_mean = round(_gm, 4) if np.isfinite(_gm) else float('nan')
                    if _gnd < _gnt and verbose:
                        print(f"    {rep_name}: geometry score defined on "
                              f"{_gnd}/{_gnt} folds; the mean is over those {_gnd}")
                    for model_name in models:
                        baselines[rep_name][model_name] = _geo_mean
                        _record_folds(rep_name, model_name, _geo_folds)
                if verbose:
                    print(f"  {rep_name} (dim={dim}, geometry): "
                          f"{baselines[rep_name].get(next(iter(models)), float('nan'))}")
                continue

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
                    # Same missing-value fill the base-rep path gets in
                    # _evaluate_single: training median per feature, applied to
                    # both sides. Without it a per-fold rep carrying NaN reached
                    # the model untouched — lightgbm absorbs that, RF/SVR/GP
                    # raise, so the same rep scored or crashed depending only on
                    # which model it was paired with.
                    X_train_s, X_val_s, _ = _impute_nan(X_train_s, X_val_s)

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
    # A freshly computed result carries these; a stored row predates the column,
    # so they are recovered from the fold scores it does store. Without them a
    # reader that checks the fold count would KeyError on a cached row and not
    # on a computed one.
    _fs = row.get('fold_scores')
    r['n_folds_scored'] = len(_fs) if _fs else None
    r['n_folds_asked'] = None
    # The engine attaches 'rep_subset' only for non-'all' rep sets.
    if row.get('rep_subset') not in ('all', '', None):
        r['rep_subset'] = row['rep_subset']
    return r


def _run_search(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: Optional[np.ndarray],
    models: Dict[str, Any],
    metric_fn: Callable,
    budgets: List[int],
    allocation_methods: Optional[List[str]] = None,
    importance_methods: Optional[List[str]] = None,
    internal_evaluators: Optional[List[str]] = None,
    objective: str = OBJECTIVE_PREDICTIVE,
    n_folds: Optional[int] = None,
    tune_per_fold: bool = False,
    tune_n_iter: int = 20,
    test_rep_subsets: bool = False,
    seed: int = 42,
    verbose: bool = True,
    augmentations: Optional[Dict[str, np.ndarray]] = None,
    augmentation_strategy: str = 'uncapped',
    augmentation_threshold: Optional[float] = None,
    augmentation_budget: int = 20,
    aug_per_source_budget: Optional[int] = None,
    candidate_ranking: str = 'residual_correlation',
    lower_is_better: bool = False,
    fold_reps: Optional[Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = None,
    precomputed_folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    cv_splitter=None,
    greedy_tolerance: float = 0.0,
    augmentation_margin: float = 0.0,
    augmentation_fraction: Optional[float] = None,
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
    row_index: Optional[np.ndarray] = None,
    structural_pairs: Optional[dict] = None,
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
    search_importance = importance_methods if importance_methods is not None else DEFAULT_IMPORTANCE_METHODS
    search_allocation = allocation_methods if allocation_methods is not None else DEFAULT_ALLOCATION_METHODS
    _validate_allocation_names(search_allocation)
    search_evaluators = internal_evaluators if internal_evaluators is not None else DEFAULT_INTERNAL_EVALUATORS
    search_budgets = budgets
    search_folds = n_folds if n_folds is not None else DEFAULT_N_FOLDS

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

    _budget_cap_notes = []

    def _budgets_for_model(model_name):
        """Return budgets filtered by per-model cap from MODEL_MAX_BUDGET."""
        cap = MODEL_MAX_BUDGET.get(model_name)
        if cap is None:
            return search_budgets
        kept = [b for b in search_budgets if b <= cap]
        dropped = [b for b in search_budgets if b > cap]
        if dropped:
            # The cap used to remove budgets in silence, so a budget column
            # compared a different set of models in each row, and a model whose
            # every budget exceeded its cap simply had no entry anywhere with
            # nothing said. MODEL_MAX_BUDGET is a hardcoded per-model list, not
            # a measured result -- see FRAMEWORK_FINALIZATION.md.
            _budget_cap_notes.append((model_name, cap, dropped, len(kept)))
        if not kept:
            raise ValueError(
                f"MODEL_MAX_BUDGET caps {model_name!r} at {cap} features, and "
                f"every requested budget {sorted(search_budgets)} is above it, "
                f"so {model_name!r} would be searched at zero configurations "
                f"and silently absent from every result. Drop the model, lower "
                f"a budget, or raise the cap in "
                f"find_best_hybrid.MODEL_MAX_BUDGET.")
        return kept

    # Compute effective config count (respects per-model budget caps)
    total = 0
    for alloc in search_allocation:
        n_imp = len(search_importance)
        for model_name in models:
            n_ev = len(_evaluators_for_model(model_name))
            n_bgt = len(_budgets_for_model(model_name))
            total += len(rep_sets) * n_bgt * n_imp * n_ev

    if _budget_cap_notes:
        for _mn, _cap, _dropped, _n_kept in sorted(set(
                (m, c, tuple(d), k) for m, c, d, k in _budget_cap_notes)):
            print(f"  MODEL_MAX_BUDGET: {_mn} capped at {_cap} features — "
                  f"budgets {list(_dropped)} not searched for it, "
                  f"{_n_kept} budget(s) remain. Any budget column below "
                  f"therefore compares a different set of models per row.")

    if verbose:
        print(f"\nSearching {total} unique configurations...")

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

    # Geometry mode does not use model-prediction alloc scores — the
    # allocators compute live geometry per rep instead. Skip the precompute (it
    # would only fit models whose geometry metric is a no-op guard) and leave
    # alloc_cache empty; _evaluate_hybrid_config forces fold_alloc_scores=None
    # in geometry mode.
    # A BLEND is skipped too: its budget split ranks representations under every
    # component and combines the ranks, so a per-rep prediction score is not the
    # quantity it needs. Precomputing them would be wasted model fits at best,
    # and a silently wrong budget split at worst.
    _skip_alloc_precompute = (_is_geometry_objective(objective)
                              or _is_blend_objective(objective))

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
        tune_per_fold=tune_per_fold, tune_n_iter=tune_n_iter,
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
        augmentation_margin=augmentation_margin,
        augmentation_fraction=augmentation_fraction,
        objective=objective,
        # Cross-invocation dedup cache activates only when the caller
        # passes dataset_name + task_type. Otherwise no-ops.
        dataset_name=dataset_name,
        task_type=task_type,
        baselines=baselines,
        train_labels=train_labels,
        noise_id=noise_id,
        row_index=row_index,
        structural_pairs=structural_pairs,
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
    _strategy_top_n = _global_schedule['strategy_top_n']

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
    # More methods than get a point per group, or the prune can never drop one.
    _pruning_enabled = (search_folds > 1
                        and len(search_allocation) > _strategy_top_n
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
            _f0_failures = []
            for idx, cfg in enumerate(_iter_configs, 1):
                ck = _cfg_key(cfg)
                # A config already scored (resume) or already dropped is not
                # re-evaluated -- but the checkpoint-crossing test below MUST
                # still run for it. It used to sit after two `continue`s, so a
                # resumed run whose last config at a checkpoint budget was
                # already done never fired that checkpoint's tail prune, and
                # went on to evaluate a DIFFERENT set of configurations than the
                # same run uninterrupted.
                _skip = (ck in fold0_done_keys) or _is_dropped(cfg)
                if not _skip:
                    try:
                        result = _eval_cfg(cfg, fold_indices=[0])
                        fold0_results.append((ck, result))
                    except Exception as _f0_err:
                        # Was `pass`. A config that raises on fold 0 contributes
                        # no score, so its allocation method loses a placement
                        # in the end-of-fold-0 vote below and can be pruned out
                        # of the whole run on the strength of a crash nobody
                        # saw. Always printed, never verbose-gated.
                        _f0_failures.append((ck, f'{type(_f0_err).__name__}: {_f0_err}'))
                        print(f"  !! fold-0 eval FAILED [{cfg[5]}/{cfg[2]}/"
                              f"{cfg[4]}/b={cfg[3]}/ev={cfg[7]}]: "
                              f"{type(_f0_err).__name__}: {_f0_err}")

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

            if _f0_failures:
                print(f"  !! fold 0: {len(_f0_failures)} of {n_iter} configs "
                      f"raised and scored nothing. The allocation-method vote "
                      f"below counts only configs that scored, so a method "
                      f"whose configs crashed can be pruned for crashing.")

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

                place_counts = defaultdict(int)
                for key, entries in groups_f0.items():
                    ranked = sorted(entries, key=lambda x: x[1],
                                    reverse=not lower_is_better)
                    for alloc, _ in ranked[:_strategy_top_n]:
                        place_counts[alloc] += 1

                pruned_strategies = {s for s in search_allocation
                                     if place_counts.get(s, 0) < threshold}

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
                  f"(top-{_strategy_top_n} in <{_prune_frac:.0%} of groups): "
                  f"{sorted(pruned_strategies)}")

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

        # Which recipes THIS call asked for, ignoring step_size. Every stored
        # row for the identity used to be appended to `results` wholesale, so a
        # call that asked for one budget was ranked over -- and reported --
        # every budget a previous call had ever stored under the same data and
        # folds. best_by_model could name a recipe this invocation never
        # requested. Rows all stay in the database; the run now reports what it
        # asked for. step_size is blanked on both sides of the match so a
        # two-phase refinement row (same recipe, different step) is still reused.
        def _blank_step(k):
            return tuple(k)[:-1] + ('',)

        _asked = {
            _blank_step(_rc_key(_rc_cfg_dict(c[1], c[2], c[3], c[4], c[5],
                                             c[7], _DEFAULT_STEP_SIZE)))
            for c in configs
        }
        _n_stored = 0
        _n_other = 0
        for _row in _rc.fetch_rows(result_db_identity):
            _n_stored += 1
            _rk = _row['_cfg_key']
            if _blank_step(_rk) not in _asked:
                _n_other += 1
                continue
            results.append(_db_row_to_result(_row))
            _rc_done.add(_rk)
        if verbose and _n_stored:
            print(f"  ResultCache: {len(_rc_done)} stored configs reused; "
                  f"computing only the remainder of {total}"
                  + (f" ({_n_other} further stored rows are other "
                     f"configurations and are not reported here)"
                     if _n_other else ""))
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
    candidate_ranking: str = 'residual_correlation',
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

    # EVERY configuration scoring NaN is a broken run, not a close contest.
    # max() over an all-NaN list returns the first element, so the search
    # reported a "best" recipe picked by nothing at all and every number
    # downstream was arithmetic on an arbitrary choice.
    # How it happens under a geometry objective: MCS fidelity is NaN when fewer
    # than 10 sampled pairs have BOTH molecules inside the fold being scored, so
    # too few pairs for the fold size silently voids every score. Found
    # 2026-08-27 running a similarity/struct_fid blend on 400 pairs, where an
    # inner validation fold holds about 10.
    _finite = [r for r in results
               if r.get('score_mean') is not None
               and np.isfinite(r['score_mean'])]
    if not _finite:
        raise RuntimeError(
            f"find_best_hybrid: all {len(results)} configurations scored NaN, "
            "so the winner would be whichever one came first. Refusing to "
            "report it. Under a geometry objective the usual cause is too few "
            "MCS pairs for the fold size: structural fidelity needs at least "
            "10 pairs with BOTH molecules inside the scored fold, and a fold "
            "holds roughly n_pairs * (fold_fraction ** 2) of them. Raise "
            "n_pairs, or check that structural_pairs and row_index index the "
            "same row order.")

    # From here on EVERY winner is picked out of `_finite`, never out of
    # `results`. A NaN score compares False against everything, so a NaN sitting
    # first in a list wins max() outright and nothing after it can displace it.
    # The all-NaN case above is a crash; a PARTLY-NaN list used to hand back a
    # winner chosen by list order, which is the same defect one config at a
    # time. `results` is still what gets stored and returned — only the argmax
    # is restricted.
    if len(_finite) < len(results):
        _lost = sorted(set(r['model'] for r in results)
                       - set(r['model'] for r in _finite))
        print(f"  !! {len(results) - len(_finite)} of {len(results)} "
              f"configurations scored NaN and cannot win anything."
              + (f" No configuration scored for {_lost}, so "
                 f"{'they are' if len(_lost) > 1 else 'it is'} absent from "
                 f"every table below." if _lost else ""))

    # Best by model
    model_names = sorted(set(r['model'] for r in _finite))
    best_by_model = {}
    for mname in model_names:
        model_results = [r for r in _finite if r['model'] == mname]
        if model_results:
            best_by_model[mname] = max(model_results, key=_key)

    # Best by dimension tier
    best_by_dimension = {}
    for tier in dimension_tiers:
        eligible = [r for r in _finite if r['n_features'] <= tier]
        if eligible:
            best_by_dimension[tier] = max(eligible, key=_key)

    # Best by budget
    budget_values = sorted(set(r['budget'] for r in _finite))
    best_by_budget = {}
    for bgt in budget_values:
        budget_results = [r for r in _finite if r['budget'] == bgt]
        if budget_results:
            best_by_budget[bgt] = max(budget_results, key=_key)

    # Best by (model, budget) — primary selection unit.
    # For each downstream model, find the best feature config per budget.
    # Avoids selection overfitting by not picking across models.
    best_by_model_budget = {}
    for mname in model_names:
        best_by_model_budget[mname] = {}
        for bgt in budget_values:
            eligible = [r for r in _finite
                        if r['model'] == mname and r['budget'] == bgt]
            if eligible:
                best_by_model_budget[mname][bgt] = max(eligible, key=_key)

    # Performance matrix: model x dimension
    performance_matrix = {}
    for mname in model_names:
        performance_matrix[mname] = {}
        for tier in dimension_tiers:
            eligible = [
                r for r in _finite
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
    candidate_ranking: str = 'residual_correlation',
    finetuned_embeddings: Optional[Dict[str, np.ndarray]] = None,
    lower_is_better: bool = False,
    objective: str = OBJECTIVE_PREDICTIVE,
    is_clf: bool = False,
    groups: Optional[np.ndarray] = None,
    # The downstream estimator INSTANCE this recipe was searched for, and the
    # dataset's own metric. Both were missing until 2026-09-03, so the full-data
    # refit took _evaluate_allocation's OTHER branch: it scored allocations with
    # the internal evaluator on r2/accuracy instead of the real model on the
    # real metric, and internal_evaluator='match' (the Blueprint default)
    # resolved to lightgbm whatever the search had matched it to. The published
    # feature list was therefore a different recipe from the one that won.
    model=None,
    metric_fn: Optional[Callable] = None,
) -> FittedResult:
    """Fit on full data and return concrete feature indices + optional weights.

    In similarity mode (objective='similarity') the allocation is rebuilt
    with the SAC objective so the fitted feature composition matches what the
    search actually selected (otherwise the full-data refit would silently fall
    back to predictive allocation).

    `groups` feeds create_hybrid's INTERNAL train/validation split, matching
    what _run_search already does per fold (groups=fold_groups). Without it
    that split falls back to np.random.RandomState(42).permutation — a random
    20% carve. Scope of the effect: the internal split drives the greedy /
    dominant_greedy allocators and every augmentation strategy other than
    'none'. It does NOT change a performance_weighted / top_k blueprint whose
    rep_scores are injected as precomputed_alloc_scores and whose augmentation
    strategy is 'none' — that path never consults the split (feature
    importance is computed on all rows). Caveat: create_hybrid only takes the
    group-aware branch when the array holds >= 2 distinct groups; otherwise it
    silently uses the random fallback."""
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
        groups=groups,
        model=model,
        metric_fn=metric_fn,
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
    **kwargs,
) -> 'SearchResult':
    """Search entry point. See `_find_best_hybrid_inner` for the full contract.

    This wrapper exists for ONE reason: `cache_dir` used to reroot only the
    search pickle and the result database, while per-representation baselines,
    fold splits, per-fold hybrid evaluations and fine-tuned embeddings kept
    reading the default cache root. A run pointed at a second cache therefore
    read half its inputs from the first one. Every cache path resolves through
    _cache_base(), so pointing that at cache_dir for the duration of the call
    reroots all of them at once.
    """
    with _cache_root_override(kwargs.get('cache_dir')):
        return _find_best_hybrid_inner(base_reps, labels, groups, **kwargs)


def _find_best_hybrid_inner(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: Optional[np.ndarray] = None,
    *,
    budgets: Optional[List[int]] = None,
    models: Optional[Any] = None,
    metric: Union[str, Callable, Tuple[Callable, bool]] = 'r2',
    task_type: str = '',
    allocation_methods: Optional[List[str]] = None,
    importance_methods: Optional[List[str]] = None,
    internal_evaluators: Optional[List[str]] = None,
    # Objective: 'predictive' (downstream model performance, the default) or
    # 'similarity' (held-out activity read-across — finds a DIFFERENT hybrid
    # whose geometry makes near neighbours share properties). In sim mode no
    # downstream model is fit; scoring is read-across on the held-out fold.
    objective: str = OBJECTIVE_PREDICTIVE,
    # Whole-dataset row number of each row of base_reps, in order, plus the
    # cached MCS pairs {'I','J','mcs'} whose I/J are whole-dataset row
    # numbers. Supply both to record MCS structural fidelity per fold
    # ('struct_fidelity' in each result's metrics). Neither changes which
    # configuration wins. Under nested CV, nested_cv_evaluate fills
    # row_index with the outer-training row numbers automatically.
    row_index: Optional[np.ndarray] = None,
    structural_pairs: Optional[dict] = None,
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
    candidate_ranking: str = 'residual_correlation',
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
    augmentation_margin: float = 0.0,
    augmentation_fraction: Optional[float] = None,
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
    # Let a tuned n_neighbors survive to the fit. OFF by default, and OFF is
    # exactly the old behaviour: k = min(5, sqrt(N_train)), i.e. 5 for any
    # training fold of 25 rows or more.
    #
    # Four scorers here and one in hybrid.py used to overwrite n_neighbors
    # AFTER the model was built, so tune=True moved `weights` and nothing
    # else. With this on, find_best_hybrid stamps the tuned KNN and every
    # scorer keeps its k. Needs tune=True to do anything — on its own there is
    # no tuned value to keep.
    #
    # In the cache key: a run that keeps its tuned k is a different search.
    tune_knn_k: bool = False,
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
    # Feature blacklist: {rep_or_aug_name: iterable of ORIGINAL column indices
    # to DROP}. Applied ONCE to base_reps AND augmentations before anything
    # consumes them, so blacklisted columns can never be ranked or selected in
    # any allocation/augmentation path. Index-based (the matrices carry no
    # names — the CALLER resolves names -> indices). Default None = no-op, zero
    # behaviour change. NOTE: rep NAMES are unchanged by a column drop and
    # rep/aug hashes are in the cache ignore-keys, so a blacklisted run shares
    # cache paths with a full-width run — pass a distinct cache_namespace /
    # condition_suffix (or wipe the cache) so the two do not collide.
    feature_blacklist: Optional[Dict[str, Sequence[int]]] = None,
) -> SearchResult:
    """
    Find the best hybrid representation configuration.

    Domain-agnostic: works with any pre-computed feature matrices.

    Args:
        base_reps: Dict of {rep_name: np.ndarray} feature matrices (all same n_samples).
        labels: Target values (n_samples,).
        groups: Group labels for GroupKFold (n_samples,). If None, uses random KFold.
        models: Model specification. None=the default model set.
            List[str] for named models. Dict[str, model] for custom models.
        metric: Evaluation metric. String ('r2','rmse','mae','accuracy','auc',
            'spearman','pearson','ci','f1_weighted')
            or callable(y_true, y_pred)->float, or (callable, lower_is_better).
        task_type: 'regression' or 'classification'. Required — no auto-detect.
        objective: 'predictive' (default; fit model, score predictions), or
            'similarity' (select by held-out ACTIVITY READ-ACROSS — scaffold-
            grouped kNN Spearman on the hybrid's distance matrix, no model fit;
            groups are required). 'similarity_sac' was CUT on 2026-08-26 and
            now raises. For 'similarity' the honest outer fold
            (nested_cv_evaluate) additionally reports MCS structural fidelity
            when structural_pairs are supplied there. Inside this inner search
            structural fidelity is recorded per configuration only when BOTH
            row_index and structural_pairs are passed; row_index carries the
            whole-dataset row number of each row, which is what a stored pair
            needs to be located.
        feature_blacklist: Optional {rep_or_aug_name: [ORIGINAL col idx to
            DROP]}. Applied once to base_reps AND augmentations before the hybrid
            is built, so blacklisted columns never enter ranking/selection in any
            path. Index-based; the caller resolves names -> indices. Default
            None = no-op. NOTE (cache): rep names are unchanged by a drop, so
            pass a distinct cache_namespace/condition_suffix (or wipe the cache)
            to keep blacklisted and full-width runs from colliding.
        budgets: List of feature budgets to search over (required).
        allocation_methods: Override the default allocation methods
            (DEFAULT_ALLOCATION_METHODS: the per-rep quota family —
            top_*/performance_weighted/power_2/rank_*/entropy_*). Opt-in methods
            that are implemented but NOT in the default set are listed in
            EXTRA_ALLOCATION_METHODS — e.g. pass allocation_methods=['weighted_global']
            to run the global feature-pooling allocator (hybrid.py
            allocate_weighted_global) instead of / alongside the default. Names
            flow through generically; the only gatekeeper is create_hybrid's dispatch.
        importance_methods: Override the default importance methods (DEFAULT_IMPORTANCE_METHODS).
        internal_evaluators: Override the default internal evaluator list
            (default ['match'], matched to the downstream model).
        n_folds: Override the number of CV folds (default DEFAULT_N_FOLDS).
        adaptive_pruning_schedule: Override any field of
            DEFAULT_PRUNING_SCHEDULE; unnamed fields keep their default. All
            five, and what each one changes:
              strategy_top_n (3) — after fold 0 every configuration has a
                score. They are grouped, one group per (importance method,
                evaluator, model, budget). Inside a group the allocation
                methods are ranked and the best `strategy_top_n` each get a
                point. Set it BELOW the number of allocation methods searched
                or no method can ever lose a point, and nothing is dropped.
              final_strategy_threshold (0.15) — an allocation method is dropped
                from the remaining folds when it scored a point in fewer than
                this fraction of the groups.
              checkpoints ([]) and tail_drop_per_checkpoint (0.0) — with both
                set, fold 0 stops at each listed budget and drops the worst
                `tail_drop_per_checkpoint` of (allocation, importance) pairs
                per (model, evaluator) before continuing.
              min_configs_per_model (1) — floor on how many configurations
                survive per model, so best_by_model always has an entry.
            Every field that differs from its default goes into the search
            cache key, so a pruned run and an unpruned one do not collide.
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

    # --- Feature blacklist: drop named columns before ANYTHING consumes the
    # matrices. Applied once here (after the n_samples check, before the
    # aug-strategy normalization / budget filter / any cache-meta hash / the
    # baselines and importance precompute) so blacklisted columns can never be
    # ranked or selected in any allocation or augmentation path. The reduced
    # matrices flow unchanged everywhere downstream; selected_indices returned
    # by create_hybrid are already in reduced-column space and are re-applied to
    # the same reduced val_reps, so no index remapping is needed. Guarded by
    # `if feature_blacklist:` so the default path leaves both dicts untouched. ---
    if feature_blacklist:
        def _drop_blacklisted(dct):
            out = {}
            for name, X in dct.items():
                drop = feature_blacklist.get(name)
                if not drop:
                    out[name] = X
                    continue
                drop_idx = np.asarray(sorted({int(i) for i in drop}), dtype=int)
                if drop_idx.size and (drop_idx.min() < 0
                                      or drop_idx.max() >= X.shape[1]):
                    raise ValueError(
                        f"feature_blacklist index out of range for {name!r} "
                        f"(width {X.shape[1]}).")
                keep = np.setdiff1d(np.arange(X.shape[1]), drop_idx)
                if keep.size == 0:
                    raise ValueError(
                        f"feature_blacklist removes ALL columns of {name!r}.")
                out[name] = X[:, keep]
            return out
        base_reps = _drop_blacklisted(base_reps)
        if augmentations:
            augmentations = _drop_blacklisted(augmentations)
        if verbose:
            print(f"  feature_blacklist applied to "
                  f"{sorted(feature_blacklist.keys())} — columns dropped "
                  f"before hybrid build.")

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
    # Resolve defaults from the module DEFAULT_* constants so that None and
    # explicit default values produce identical cache keys.
    _norm_alloc = allocation_methods if allocation_methods is not None else DEFAULT_ALLOCATION_METHODS
    # Reject an unknown allocation name HERE. Without this the name survives
    # until create_hybrid's dispatch, which is inside the fold loop — after the
    # reps are loaded and the importance scores are computed. On a cluster that
    # is a wasted job for a typo.
    _validate_allocation_names(_norm_alloc)
    _norm_imp = importance_methods if importance_methods is not None else DEFAULT_IMPORTANCE_METHODS
    _norm_eval = internal_evaluators if internal_evaluators is not None else DEFAULT_INTERNAL_EVALUATORS
    _norm_folds = n_folds if n_folds is not None else DEFAULT_N_FOLDS

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
        _max_tier = _TIER_ORDER['full']  # default set = non-'extra' models
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

    # --- Validate objective; in geometry mode force the matching geometry
    # metric so the stored metric_name and comparison direction are always
    # consistent (all of them are higher-is-better; no downstream model is fit).
    #   similarity -> 'readacross', struct_fid -> 'struct_fidelity',
    #   a dict     -> 'blend_<component><weight>_...' ---
    if isinstance(objective, dict):
        objective = _validate_blend_objective(objective)
    elif objective not in _VALID_OBJECTIVES:
        if objective == OBJECTIVE_SIMILARITY_SAC:
            raise ValueError(
                "find_best_hybrid: objective='similarity_sac' was CUT on "
                "2026-08-26. Nothing selects by SAC any more. Use "
                f"objective='{OBJECTIVE_SIMILARITY}' (each molecule's property "
                "predicted from its five nearest neighbours). See the block "
                "above OBJECTIVE_PREDICTIVE for the numbers on both sides."
            )
        raise ValueError(
            f"find_best_hybrid: objective={objective!r}. "
            f"Must be one of {_VALID_OBJECTIVES}, or a dict blending "
            f"{tuple(_BLEND_COMPONENT_KEY)}."
        )
    # MCS fidelity cannot be computed without the pairs and the whole-dataset
    # row numbers. Without them _structural_fidelity_score returns NaN on every
    # fold, every configuration ties at NaN, and the search reports a winner
    # picked by nothing. Refuse instead. Checked OUTSIDE the geometry branch: a
    # blend containing both 'predictive' and 'struct_fid' is not a geometry
    # objective and would otherwise skip the check.
    if _objective_needs_pairs(objective) and (
            structural_pairs is None or row_index is None):
        raise ValueError(
            f"find_best_hybrid: objective={objective!r} scores MCS "
            "structural fidelity, which needs BOTH structural_pairs "
            "({'I','J','mcs'}) and row_index (whole-dataset row number of "
            "each row, in order). "
            f"structural_pairs={'given' if structural_pairs is not None else 'None'}, "
            f"row_index={'given' if row_index is not None else 'None'}. "
            "nested_cv_evaluate fills row_index itself; pass "
            "structural_pairs to it.")
    if _is_geometry_objective(objective):
        _geo_metric = _objective_metric_name(objective)
        if metric != _geo_metric:
            if verbose:
                print(f"[objective={objective}] overriding metric={metric!r} -> "
                      f"{_geo_metric!r} (geometry objective, no downstream "
                      f"model fit).")
            metric = _geo_metric

    # Metric name as it appears in cached metadata. Callables can't be
    # filename-encoded, so when metric is a callable the caller is expected
    # to also use cache=False (caching a custom metric has no safe key).
    _metric_name_for_meta = metric if isinstance(metric, str) else 'custom'
    if _blend_needs_model(objective):
        # `metric` stays the dataset's own metric because the model still has
        # to be fitted and scored with it. What the run SELECTS on is the
        # blend, so the meta has to say so — otherwise a blend and a plain
        # predictive run on the same dataset record the same metric_name and a
        # cached search for one satisfies the meta check for the other.
        _metric_name_for_meta = _objective_metric_name(objective)

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

    # objective namespaces the SAME data+folds so a similarity/geometry search and
    # a predictive one never share a result-DB identity or a searches/*.pkl file
    # (both key on tier). Predictive leaves tier byte-identical, so existing rows
    # keep matching. Same tier-riding trick as label_noise above and as
    # general_distance / top_distance. The baseline and hybrid-eval caches carry no
    # tier, so they get the objective marker threaded into their own filenames.
    if tier and objective and objective != OBJECTIVE_PREDICTIVE:
        tier = f'{tier}__obj-{_objective_cache_token(objective)}'

    # condition_suffix reached the search-pickle FILENAME and the search meta,
    # but neither the result database's filename nor its row identity — so two
    # arms of one experiment that differ ONLY by condition_suffix shared rows,
    # and whichever ran first answered for both. Riding it on `tier` fixes that
    # with no schema change and no wipe: `tier` is IDENTITY_COLS[0] and already
    # carries the noise condition and the objective the same way. A run that
    # passes no condition_suffix leaves `tier` byte-identical, so every stored
    # row keeps matching.
    if tier and condition_suffix:
        tier = f'{tier}__cond-{_sanitize_filename_part(condition_suffix)}'

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
            _ds_cache, tier, augmentation_strategy,
            condition_suffix=condition_suffix)
        # cache_dir is now handled by _cache_base() for EVERY cache path (see
        # the find_best_hybrid wrapper), so the search pickle lands at
        # <cache_dir>/searches/<name>.pkl like every other layer. It used to be
        # rerooted flat to <cache_dir>/<name>.pkl while the result database
        # already used <cache_dir>/result_db/, so the two halves of one cache
        # sat in different shapes. A pickle written flat under an old cache_dir
        # is now a MISS, which recomputes; the result database is untouched.

        # Hash of the MCS pair set + the whole-dataset row numbers, so a search
        # that recorded structural fidelity cannot be served one that did not.
        _search_struct_pairs_id = None
        if structural_pairs is not None and row_index is not None:
            try:
                _search_struct_pairs_id = _hash_bytes(
                    np.ascontiguousarray(
                        np.asarray(structural_pairs['I'], dtype=np.int64)).tobytes()
                    + np.ascontiguousarray(
                        np.asarray(structural_pairs['J'], dtype=np.int64)).tobytes()
                    + np.ascontiguousarray(
                        np.asarray(row_index, dtype=np.int64)).tobytes())
            except Exception:
                _search_struct_pairs_id = 'unhashable'

        expected_search_meta = _make_search_meta(
            tune=tune,
            tune_n_iter=tune_n_iter,
            tune_knn_k=tune_knn_k,
            struct_pairs_id=_search_struct_pairs_id,
            dataset_name=_ds_cache,
            tier=tier,
            aug_strategy=augmentation_strategy,
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
            augmentation_margin=augmentation_margin,
            augmentation_fraction=augmentation_fraction,
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
        # KEY: data + fold split + tier (the pool + grouping handle). Never
        # the sweep (models/rankers/budgets) nor the aug recipe — those are
        # per-row config, so reuse on a changed list is a SELECT, not a rebuild.
        result_db_identity = {
            'tier': tier,
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
        # The per-source cap changes WHICH augmentation columns are candidates,
        # so two runs differing only in it are different recipes. It has no
        # column of its own (the DB has no migration path — adding a key column
        # would force wiping every result_db), so it rides in aug_names as a
        # suffix, and ONLY when set. Unset -> the string is byte-identical to
        # what previous runs wrote, so existing rows keep matching. Same
        # convention as the hybrid-eval filename suffix (_hybrid_eval_cache_path).
        _psb_key = _canon_per_source_budget(aug_per_source_budget)
        result_db_config_const = {
            'aug_strategy': augmentation_strategy,
            'aug_names': (repr(tuple(_esm.get('aug_names') or ()))
                          + (f'|ps={_psb_key!r}' if _psb_key else '')),
            'aug_threshold': _esm.get('augmentation_threshold'),
            'aug_budget': _esm.get('augmentation_budget'),
            'candidate_ranking': _esm.get('candidate_ranking'),
            'greedy_tolerance': _esm.get('greedy_tolerance'),
            'augmentation_margin': _esm.get('augmentation_margin'),
            'augmentation_fraction': _esm.get('augmentation_fraction'),
        }
        # Recorded on every row as data (NOT a key): the rep pool this search ran
        # over, so a stored row is self-describing. repr(tuple) avoids the comma
        # collision of a join.
        result_db_rep_pool = repr(tuple(_esm.get('rep_names') or ()))

    if groups is not None:
        groups = np.asarray(groups).ravel()
        if len(groups) != n_samples:
            raise ValueError(f"groups length ({len(groups)}) != n_samples ({n_samples})")

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
    # The registry is the single source of truth for a dataset's metric, but
    # find_best_hybrid has never consulted it: `metric` defaults to 'r2', so a
    # caller that forgot to pass one ran an RMSE dataset under r2 and cached the
    # result under a metric name that was never asked for. registry.lookup's
    # docstring claims a guard here that does not exist. Said out loud rather
    # than overridden, because f18 sweeps the metric on purpose and an explicit
    # metric= must always win. Whether the registry should become the DEFAULT is
    # written up in FRAMEWORK_FINALIZATION.md.
    if dataset_name and isinstance(metric, str) and metric == 'r2':
        try:
            from kirby import registry as _reg
            _reg_info = _reg.lookup(dataset_name)
        except Exception:
            _reg_info = None
        if _reg_info is not None and _reg_info.metric != 'r2':
            print(f"  !! metric: running {dataset_name!r} under 'r2', but the "
                  f"registry says its metric is {_reg_info.metric!r}. 'r2' is "
                  f"find_best_hybrid's DEFAULT, not a registry lookup — pass "
                  f"metric={_reg_info.metric!r} if this was not deliberate. "
                  f"The metric is in the cache filename, so the two are "
                  f"stored separately.")
    metric_fn, lower_is_better = _resolve_metric(metric)
    if _blend_needs_model(objective):
        # A blend containing 'predictive' still needs a real metric function:
        # something has to fit the model and turn it into predictions. But the
        # WINNER is chosen on the blend, which is a weighted mean of three
        # correlations and is always higher-is-better. Leaving lower_is_better
        # at the dataset metric's value (True on every RMSE set here) would make
        # the search take the max of the NEGATED blend and pick the worst
        # configuration in the grid, with nothing raising.
        lower_is_better = False

    # --- Resolve secondary metrics ---
    resolved_secondary = None
    if secondary_metrics:
        resolved_secondary = {name: _resolve_metric(name)[0] for name in secondary_metrics}
        if verbose:
            print(f"Secondary metrics: {secondary_metrics}")

    # --- Resolve models ---
    resolved_models = _resolve_models(models, task_type,
                                       n_samples=n_samples,
                                       dataset_name=dataset_name,
                                       strategy=augmentation_strategy)
    if verbose:
        print(f"Models: {list(resolved_models.keys())}")
        print(f"Representations: {list(base_reps.keys())} "
              f"(dims: {[X.shape[1] for X in base_reps.values()]})")
        if rep_factories:
            print(f"Fine-tuning factories: {list(rep_factories.keys())}")

    # --- Hyperparameter tuning ---
    # Tuning happens PER FOLD, inside the scorers, on the feature matrix the
    # model is actually fitted on: one representation for a baseline, the
    # budget-many selected features for a hybrid cell. See
    # _tune_on_training_fold.
    #
    # It used to happen once here, over np.hstack of every base representation
    # — a 15k-wide matrix no reported score is ever produced on. A
    # hyperparameter chosen there is chosen for the wrong space, and for KNN
    # the wrong space is the whole point: the hybrid exists to change the
    # distance. Removed 2026-09-03.
    #
    # All this block does now is mark the KNN estimator, so the k each fold's
    # search chooses survives _set_knn_k instead of being overwritten with 5.
    if tune:
        if tune_knn_k:
            for _mdl in resolved_models.values():
                if hasattr(_mdl, 'n_neighbors'):
                    setattr(_mdl, KNN_TUNED_K_ATTR, True)
        if verbose:
            _tuning = sorted(
                _mn for _mn, _m in resolved_models.items()
                if _HP_SEARCH_SPACES.get(_model_name_for_search_space(_m)))
            print(f"Tuning per fold on the fitted features: {_tuning}"
                  f"{' (keeping tuned k)' if tune_knn_k else ' (k stays 5)'}")

    # --- Pre-compute fine-tuned reps ---
    fold_reps = {}
    precomputed_folds = []
    if rep_factories:
        actual_folds_for_ft = n_folds if n_folds is not None else DEFAULT_N_FOLDS
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
    baseline_folds = n_folds if n_folds is not None else DEFAULT_N_FOLDS
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
        tune_per_fold=tune,
        tune_n_iter=tune_n_iter,
        # Only read under objective='struct_fid' (or a blend containing it),
        # where the per-representation baseline has to be an MCS correlation
        # like the hybrid arm's, not a nearest-neighbour one.
        structural_pairs=structural_pairs,
        row_index=row_index,
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
        base_reps, labels, groups, resolved_models,
        metric_fn,
        budgets=budgets,
        allocation_methods=allocation_methods,
        importance_methods=importance_methods,
        internal_evaluators=internal_evaluators,
        objective=objective,
        n_folds=n_folds,
        tune_per_fold=tune,
        tune_n_iter=tune_n_iter,
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
        augmentation_margin=augmentation_margin,
        augmentation_fraction=augmentation_fraction,
        row_index=row_index,
        structural_pairs=structural_pairs,
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
        actual_folds_for_refine = n_folds if n_folds is not None else DEFAULT_N_FOLDS
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
                    # Full-data refit: `groups` is find_best_hybrid's own
                    # parameter, already length-checked against n_samples
                    # above, and base_reps here is the full dataset — so the
                    # rows line up.
                    groups=groups,
                    # The real estimator and the real metric, so the refit
                    # reproduces the recipe the search selected instead of an
                    # internal-evaluator r2 stand-in. Under a geometry
                    # objective no model is fitted at all, so pass neither —
                    # a model without a metric marks every allocation
                    # infeasible.
                    model=(None if _is_geometry_objective(objective)
                           else resolved_models.get(model_name)),
                    metric_fn=(None if _is_geometry_objective(objective)
                               else metric_fn),
                )
            except Exception as e:
                if verbose:
                    print(f"  WARNING: fitted result for {model_name} failed: {e}")


    # --- Resolve metric name for display ---
    metric_name = metric if isinstance(metric, str) else 'custom'
    actual_folds = n_folds if n_folds is not None else DEFAULT_N_FOLDS

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
        tuned_models=(dict(resolved_models) if tune else {}),
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

    # --- How the winner actually split the budget -------------------------
    # The question the whole hybrid rebuild turns on, and until now unanswerable
    # from a real run: which representations fed the hybrid, how many features
    # each got, and whether the best one was cut to dust. On a large pool of
    # near-tied representations a proportional split hands the best one a
    # handful of features out of the whole budget and everything else two or
    # three — nothing raises, nothing warns, and the recipe reads normal.
    # `feature_info['allocation']` has always held this and was always thrown
    # away; it now rides on the Blueprint as `allocation_per_fold`. Printed for
    # each model's winner on fold 0.
    if verbose and agg['best_by_model']:
        for mname in sorted(agg['best_by_model']):
            _bp = agg['best_by_model'][mname].get('blueprint')
            _apf = getattr(_bp, 'allocation_per_fold', None) or {}
            _f0 = _apf.get(min(_apf)) if _apf else None
            if not _f0:
                continue
            _used = {k: v for k, v in _f0.items() if v > 0}
            _tot = sum(_f0.values())
            _rank = sorted(_used.items(), key=lambda kv: -kv[1])
            _top = ', '.join(f'{k}={v}' for k, v in _rank[:4])
            _share = (100.0 * _rank[0][1] / _tot) if _tot else 0.0
            print(f"  budget split [{mname}, {_bp.allocation_method}, "
                  f"budget {_bp.budget}]: {_tot} features across "
                  f"{len(_used)} of {len(_f0)} representations; the largest "
                  f"share is {_rank[0][1]} ({_share:.0f}%). {_top}"
                  + (' ...' if len(_rank) > 4 else ''))

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
    groups: Optional[np.ndarray] = None,
    model=None,
    metric_fn=None,
    lower_is_better: Optional[bool] = None,
    row_index: Optional[np.ndarray] = None,
    structural_pairs: Optional[dict] = None,
    is_clf: bool = False,
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
        groups: Group labels (n_samples,) for create_hybrid's INTERNAL
            train/validation split.  Must be aligned with the rows of
            base_reps.  When omitted, create_hybrid falls back to a random
            20% split, so a scaffold-grouped dataset is rebuilt under a
            different split than _run_search used (it passes groups[train_idx]
            per fold).  Pass the same groups slice the search saw; None only
            when the dataset genuinely has none.  Note the effect is confined
            to blueprints whose recipe actually consults the split — the
            greedy / dominant_greedy allocators and any augmentation strategy
            other than 'none'.  A performance_weighted / top_k blueprint with
            rep_scores injected and augmentation 'none' is unaffected either
            way.  Also note create_hybrid only takes its group-aware branch
            when the array holds >= 2 distinct groups.
        model: The downstream estimator INSTANCE this rebuild is for (not a
            factory).  _run_search passes the real model into create_hybrid, so
            every allocation trial and every augmentation keep/drop during the
            SEARCH was scored by fitting that model and calling metric_fn.
            Omitting it here does not lose a little precision — it takes a
            different branch of _evaluate_allocation, scoring with
            internal_evaluator on r2_score / accuracy_score instead, and it
            silently resolves internal_evaluator='match' (the Blueprint
            default) to 'lightgbm'.  Requires metric_fn.
        metric_fn: The dataset's real metric, called as metric_fn(y_true,
            y_pred).  Only consulted when `model` is also given.
        lower_is_better: Direction for metric_fn.  None (default) takes the
            direction off the blueprint, which is the metric the search used.
            Pass explicitly only when metric_fn is not the blueprint's metric —
            a wrong direction silently inverts every accept decision.

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

    # A misaligned groups array would silently produce the WRONG internal
    # split (e.g. whole-dataset groups handed to a per-fold train slice).
    # Fail loudly instead.
    if groups is not None:
        groups = np.asarray(groups).ravel()
        if len(groups) != n_samples:
            raise ValueError(
                f"apply_blueprint: groups length ({len(groups)}) != number of "
                f"rows in base_reps ({n_samples}). Pass the groups slice that "
                f"matches the rows you are rebuilding on.")

    # create_hybrid's model branch scores with metric_fn(y_val, y_pred) and its
    # except clause swallows TypeError, so a model with no metric_fn would mark
    # EVERY allocation infeasible (-inf) in silence rather than raising.
    if model is not None and metric_fn is None:
        raise ValueError(
            "apply_blueprint: model was given without metric_fn. create_hybrid "
            "scores the model branch with metric_fn(y_val, y_pred) and treats "
            "the resulting TypeError as an infeasible allocation, so pass both "
            "or neither.")

    # Direction defaults to the blueprint's own metric direction — the metric
    # the search itself used. Pass it explicitly only when metric_fn is not the
    # blueprint's metric: a wrong direction silently inverts every accept.
    if lower_is_better is None:
        lower_is_better = bool(getattr(blueprint, 'lower_is_better', False))

    if labels is None:
        labels = np.zeros(n_samples)
        # Force variance-based importance (no model fitting)
        importance_method = 'lightgbm_gain'  # will be overridden by precomputed
    else:
        importance_method = blueprint.importance_method

    # A top_k recipe that zeroes a representation moves its top columns into
    # the augmentation pool INSIDE create_hybrid, by writing them into this very
    # dict — and never removes them. Callers that hand the SAME augmentation
    # dict to several rebuilds (nested_cv_evaluate builds train_augs once per
    # outer fold and rebuilds every model against it) had model 2 rebuilt with
    # model 1's demoted representation already in the pool, so the result
    # depended on model ordering. Rebuild against a copy.
    if augmentations is not None:
        augmentations = dict(augmentations)

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
        # Without this the rebuild pools EVERY column of every augmentation
        # source instead of the top-N the search selected, so the honest
        # nested-CV arms scored a different feature set than the one chosen.
        # getattr guards blueprints unpickled from caches written before the
        # field existed (same guard as _shared.build_fold_hybrid).
        aug_per_source_budget=getattr(blueprint, 'aug_per_source_budget', None),
        # Same guard: blueprints pickled before the field existed have no
        # attribute, and 0.0 reproduces the old accept-any-improvement rule.
        augmentation_margin=float(getattr(blueprint, 'augmentation_margin', 0.0) or 0.0),
        augmentation_fraction=getattr(blueprint, 'augmentation_fraction', None),
        # These four were missing until 2026-08-26 and each one silently
        # substituted create_hybrid's own default. Three of them
        # (candidate_ranking, augmentation_threshold, augmentation_budget) are
        # columns of the stored result's KEY, so a search that filed a row under
        # 'rf_importance' was rebuilt here under 'residual_correlation' with
        # nothing raising. Same shape as the 24 August grouping/model/metric
        # defect. Recipes pickled before the fields existed report the old
        # defaults, which is exactly what they ran under.
        candidate_ranking=getattr(blueprint, 'candidate_ranking',
                                  'residual_correlation'),
        augmentation_threshold=getattr(blueprint, 'augmentation_threshold', None),
        augmentation_budget=int(getattr(blueprint, 'augmentation_budget', 20)),
        aug_max_candidates=int(getattr(blueprint, 'aug_max_candidates', 50)),
        # Without this create_hybrid's internal train/val split falls back to
        # np.random.RandomState(42).permutation — a random 20% carve — while
        # _run_search built the same recipe under groups[train_idx]. Matters
        # for greedy/dominant_greedy allocation and for augmentation
        # strategies other than 'none'; a no-op for the precomputed-scores
        # top_k path below (see the `groups` docstring entry).
        groups=groups,
        # The search scores every allocation trial and every augmentation
        # keep/drop with the REAL downstream model and the REAL metric
        # (_evaluate_hybrid_config passes model/metric_fn/lower_is_better into
        # create_hybrid). Omitting them here does not merely lose precision —
        # _evaluate_allocation takes an entirely different branch, scoring with
        # internal_evaluator on r2_score/accuracy_score instead. It also
        # changes which evaluator that is: internal_evaluator='match' is the
        # Blueprint default and resolves off `model`, so with model=None it
        # silently becomes 'lightgbm' whatever the search matched it to. So a
        # rebuild without these reproduces a DIFFERENT recipe than the one the
        # search selected, with nothing raising.
        model=model,
        metric_fn=metric_fn,
        lower_is_better=lower_is_better,
        # Carried on the Blueprint since the geometry-objective work: a
        # blueprint searched under a geometry objective ranks allocations by
        # held-out SAC and fits no model at all. Rebuilding it under the
        # predictive default silently re-selects features by prediction.
        objective=getattr(blueprint, 'objective', 'predictive'),
        # A blueprint searched under 'struct_fid' (or a blend containing it)
        # splits its budget by MCS fidelity. Rebuilding without the pairs and
        # the whole-dataset row numbers raises instead of quietly splitting the
        # budget by something else.
        row_index=row_index,
        structural_pairs=structural_pairs,
        # Never passed until 2026-09-03, so a classification recipe was rebuilt
        # under regression scoring: the geometry components score class codes
        # as a continuous target and the augmentation accept/reject test picks
        # a regressor. No effect on a predictive rebuild that supplies model +
        # metric_fn, which never consults it.
        is_clf=is_clf,
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
    # ---- the other two numbers, recorded for every winner since 2026-08-27 ---
    # Under a geometry objective no model decides the winner, so a prediction
    # score has to be fitted separately to put that hybrid on the same table as
    # a predictive-picked one. This names the metric that was used and its
    # direction. Under objective='predictive' both mirror metric_name /
    # lower_is_better, and side_predictive_* repeats the primary.
    side_metric_name: Optional[str] = None
    side_metric_lower_is_better: bool = False
    # {side_metric_name: {model: {budget: {'mean','fold_scores'}}}}
    by_model_budget_side: Dict[str, Dict[str, Dict[int, dict]]] = field(
        default_factory=dict)

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
                # The width recorded on the outer folds. None when no fold
                # scored — never silently substitute the requested budget,
                # which is what this used to do.
                'n_features': e.get('n_features'),
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
                    # The per-budget arm records no rebuilt width, so there is
                    # nothing honest to put here. It used to repeat the budget,
                    # which reads as a measured column count.
                    'n_features': None,
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
                     train_groups=None, structural_pairs=None, test_idx=None,
                     test_groups=None, side_metric_fn=None, tuned_model=None,
                     tune_per_fold=False, tune_n_iter=20, train_groups_for_tune=None):
    """Score a feature matrix on the held-out outer fold.

    Scaler and imputation are fit on outer_train ONLY.

    tuned_model: the estimator this outer fold's INNER search tuned, when the
    caller ran find_best_hybrid(tune=True) and kept it. None (the default)
    builds the untuned default from model_name, which is what every KIRBy
    number before 2026-09-03 used. Passing it is the only way a tuned
    hyperparameter reaches the honest held-out score.

    THREE NUMBERS ARE RECORDED FOR EVERY WINNER, WHICHEVER OBJECTIVE PICKED IT.
    Only one of them is the RETURNED score — the one the objective selects on,
    so delta_mean_of_folds stays single-valued and well-defined. The other two
    ride along in the returned metric dict:

      'predictive'      the dataset's own metric from a fitted model
      'readacross'      each held-out molecule's property predicted from its
                        five nearest neighbours, Spearman'd against truth,
                        scaffold-grouped, higher is better
      'struct_fidelity' Spearman between pairwise distance and MCS distance,
                        higher is better

    Before 2026-08-27 a predictive run recorded none of the geometry numbers and
    a geometry run fitted no model, so two objectives could never be compared on
    one table. That is what this records.

    objective='predictive' (default): fit `model_name` on outer_train, predict
    outer_test, score with metric_fn. RETURNS that score. The two geometry
    numbers are computed on the same held-out features and stored.

    objective='similarity': no downstream model decides the winner. The features
    were selected on outer_train (inside the blueprint rebuild); the GEOMETRY of
    the untouched outer fold is scored directly and RETURNED. `side_metric_fn`,
    when given, fits `model_name` on the same features so the dataset's own
    metric is stored alongside.

    'struct_fidelity' needs `structural_pairs` and `test_idx`, and is only
    present when statistically meaningful — >=10 pairs with both molecules in
    this fold. 'readacross' needs `test_groups`.

    Returns (score, metric_dict) — (nan, {}) when the model is unavailable or
    the fit fails. A side number that fails is omitted; it never fails the fold.
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    X_tr, X_te, _ = _impute_nan(X_tr, X_te)
    _is_clf = (task_type == 'classification')

    def _geometry_metrics():
        """The two geometry numbers on the held-out features. Never raises."""
        out = {}
        try:
            ra = _readacross_objective_score(
                X_te, y_test, test_groups, _is_clf, seed=seed)
            if np.isfinite(ra):
                out['readacross'] = float(ra)
        except Exception:
            pass
        try:
            if structural_pairs is not None and test_idx is not None:
                sfd = _structural_fidelity_scores(
                    X_te, test_idx, structural_pairs)
                sf = sfd['struct_fidelity']
                if np.isfinite(sf):
                    out['struct_fidelity'] = float(sf)
                # Reported beside it, never selected on. NaN rather than absent
                # would read as a computed value downstream, so an unavailable
                # size control (a pair file written before 2026-08-27 carries no
                # heavy-atom counts) leaves the key out entirely.
                sfc = sfd['struct_fidelity_size_controlled']
                if np.isfinite(sfc):
                    out['struct_fidelity_size_controlled'] = float(sfc)
        except Exception:
            pass
        return out

    if _is_geometry_objective(objective):
        metrics = _geometry_metrics()
        score = _geometry_primary(objective, metrics)
        # The dataset's own metric, so a geometry-picked hybrid can be put on
        # the same table as a predictive-picked one. Stored, never selected on.
        if side_metric_fn is not None:
            try:
                _sm = tuned_model if tuned_model is not None else get_model(
                    model_name, task_type, n_samples=len(y_train))
                if _sm is not None:
                    _sm = deepcopy(_sm)
                    if tune_per_fold:
                        _sm = _tune_on_training_fold(
                            _sm, X_tr, y_train, task_type,
                            groups=train_groups, n_iter=tune_n_iter, seed=seed)
                    _set_knn_k(_sm, len(y_train))
                    _fit_model(_sm, X_tr, y_train, groups=train_groups)
                    _yp = _predict_for_metric(_sm, X_te, side_metric_fn)
                    _sv = float(side_metric_fn(y_test, _yp))
                    if np.isfinite(_sv):
                        metrics['predictive'] = _sv
                    # Same fit, read as a correlation, so a geometry-picked
                    # hybrid can be compared with a blend on the blend's terms.
                    _sc = _prediction_correlation(y_test, _yp, task_type)
                    if np.isfinite(_sc):
                        metrics['predictive_corr'] = float(_sc)
            except Exception:
                pass
        return float(score), metrics

    model = tuned_model if tuned_model is not None else get_model(
        model_name, task_type, n_samples=len(y_train))
    if model is None:
        return float('nan'), {}
    m = deepcopy(model)
    if tune_per_fold:
        # The honest number: tune on the outer-TRAIN rows of the very matrix
        # scored on the outer test fold. The test fold is not seen here.
        m = _tune_on_training_fold(m, X_tr, y_train, task_type,
                                   groups=train_groups, n_iter=tune_n_iter,
                                   seed=seed, model_name=model_name)
    _set_knn_k(m, len(y_train))
    # train_groups keeps the neural models' internal early-stopping carve
    # group-aware inside the outer training set.
    _fit_model(m, X_tr, y_train, groups=train_groups)
    y_pred = _predict_for_metric(m, X_te, metric_fn)
    score = float(metric_fn(y_test, y_pred))
    metrics = _compute_classic_metrics(m, X_te, y_test, y_pred, task_type)
    metrics['predictive'] = score
    # Recorded for EVERY winner, not only for blends: without it the pure
    # 'predictive' objective has no entry in the correlation column and cannot
    # be compared with a blend on the column the blend was scored in.
    _pc = _prediction_correlation(y_test, y_pred, task_type)
    if np.isfinite(_pc):
        metrics['predictive_corr'] = float(_pc)
    metrics.update(_geometry_metrics())
    if _blend_needs_model(objective):
        # The winner was chosen on the blend, so the honest outer fold has to
        # report the blend. `score` above is the dataset's own metric and stays
        # in metrics['predictive'] as one of the four recorded numbers.
        score = _blend_score(objective, metrics)
    return float(score), metrics


def nested_cv_evaluate(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: Optional[np.ndarray] = None,
    *,
    n_outer_folds: int = 5,
    seed: int = 42,
    verbose: bool = True,
    dataset_name: Optional[str] = None,
    structural_pairs: Optional[dict] = None,
    outer_rep_builders: Optional[Dict[str, Callable]] = None,
    outer_rep_builders_id: Optional[str] = None,
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
        structural_pairs: Optional {'I','J','mcs'} arrays for objective=
            'similarity'. I/J are integer indices into the SAME row order as
            `base_reps`/`labels` (built from the identical prepared arrays);
            'mcs' is the paired MCS distance. When supplied, MCS structural
            fidelity is reported (as secondary metric 'struct_fidelity') on each
            outer test fold ALONGSIDE the primary read-across score. Passing
            pairs built from a DIFFERENT row order silently mis-scores — an
            assert on max index below guards the gross case. Ignored unless
            objective='similarity'.
        outer_rep_builders: {rep_name: builder} for representations that are FIT
            ON THE TRAINING LABELS and so cannot be precomputed once. Each is
            called ONCE PER OUTER FOLD as builder(train_idx, y_train) where
            train_idx are GLOBAL row numbers into labels/base_reps and y_train is
            labels[train_idx]. It must return the full-length (n_samples, width)
            matrix: rows outside train_idx are featurised USING ONLY what was fit
            on train_idx, exactly as a fitted transform is applied to held-out
            rows. This function then slices it for the fold like any other rep,
            so the outer test fold never informs its own features.
            Name collisions with base_reps raise. Width may differ between folds.
            `base_reps` may be empty when builders supply every representation.
            Nothing here reaches the INNER folds — a builder fit on outer-train is
            reused across all inner folds of that outer fold, which is honest for
            the outer score but leaks within the inner search. For a rep that must
            also be rebuilt per inner fold, use `rep_factories` instead.
        outer_rep_builders_id: Stable identifier for what `outer_rep_builders`
            computes, in the same spirit as `rep_factories_id`. Required whenever
            builders are passed and caching is on: builder output changes per fold
            and per code version, while the cache key sees only rep_names and
            rep_shapes, so without it a changed builder would silently serve a
            stale search. Folded into each fold's cache namespace. Bump it
            whenever the builder's computation changes.
        **search_kwargs: Forwarded to find_best_hybrid (budgets, metric,
            task_type, augmentations, ...).

    Returns:
        NestedCVResult.
    """
    from kirby.hybrid import apply_feature_selection, apply_augmentation_selection

    objective = search_kwargs.get('objective', OBJECTIVE_PREDICTIVE)
    if isinstance(objective, dict):
        objective = _validate_blend_objective(objective)
        search_kwargs = dict(search_kwargs, objective=objective)
    _sim_mode = _is_geometry_objective(objective)
    # A blend that contains 'predictive' is neither: a model IS fitted, and its
    # prediction is then combined with the geometry numbers on the same rows.
    _blend_with_model = _blend_needs_model(objective)

    # Alignment contract for structural pairs (see docstring): I/J must index
    # the full base_reps/labels row order. Guard the gross mis-alignment case.
    # Checked whichever objective is running: since 2026-08-27 a predictive run
    # records MCS fidelity too, so mis-aligned pairs would mis-score it as well.
    if structural_pairs is not None:
        _n = len(labels)
        _maxij = max(int(np.max(structural_pairs['I'])),
                     int(np.max(structural_pairs['J'])))
        if _maxij >= _n:
            raise ValueError(
                f"structural_pairs index {_maxij} >= n_samples {_n}: pairs are "
                f"not aligned to base_reps/labels row order.")

    # Label-fitted representations (see docstring). Validated here rather than in
    # the fold loop so a bad call fails before any search runs.
    if outer_rep_builders:
        _clash = sorted(set(outer_rep_builders) & set(base_reps))
        if _clash:
            raise ValueError(
                f"outer_rep_builders name(s) {_clash} also in base_reps. A "
                f"builder REPLACES nothing — give it its own name or drop the "
                f"precomputed copy.")
        if search_kwargs.get('cache', True) and not outer_rep_builders_id:
            raise ValueError(
                "nested_cv_evaluate(outer_rep_builders=...) with cache=True "
                "requires outer_rep_builders_id=... The cache key sees only "
                "rep_names and rep_shapes, so a changed builder would serve a "
                "stale search. Pass a stable id and bump it when the builder's "
                "computation changes.")

    # The dataset's own metric. Under a geometry objective it does not decide
    # any winner; it is fitted and recorded so a geometry-picked hybrid lands on
    # the same table as a predictive-picked one. Under 'predictive' it IS the
    # winner's score and metric_fn already carries it.
    _side_metric_fn = None
    _side_metric_name = None
    _side_lower_is_better = False
    if _sim_mode:
        # Geometry objective: no predictor and no metric_fn. Force the direction
        # explicitly rather than letting _resolve_metric guess.
        metric_fn, lower_is_better = None, False
        metric_name = _objective_metric_name(objective)
        _raw_metric = search_kwargs.get('metric')
        if isinstance(_raw_metric, str) and (
                _raw_metric in BUILTIN_METRICS
                and _raw_metric not in ('sac', 'readacross', 'struct_fidelity')):
            _side_metric_name = _raw_metric
        else:
            _side_metric_name = (
                'accuracy' if search_kwargs.get('task_type') == 'classification'
                else 'spearman')
        _side_metric_fn, _side_lower_is_better = _resolve_metric(
            _side_metric_name)
    else:
        metric = search_kwargs.get('metric', 'r2')
        metric_fn, lower_is_better = _resolve_metric(metric)
        metric_name = metric if isinstance(metric, str) else 'custom'
        # Under 'predictive' the side prediction number IS the primary, so it
        # carries the primary's direction. rmse is lower-is-better.
        _side_metric_name = metric_name
        _side_lower_is_better = lower_is_better
        if _blend_with_model:
            # A blend containing prediction. metric_fn stays the dataset's own
            # metric — the model has to be fitted and turned into predictions —
            # but the WINNER is the blend, which is a weighted mean of three
            # correlations and always higher-is-better. Leaving lower_is_better
            # at the dataset metric's value would orient every paired delta
            # backwards on an RMSE dataset.
            lower_is_better = False
            metric_name = _objective_metric_name(objective)

    # Direction of the quantity SearchResult.baselines holds, which is NOT
    # always the direction the winner is scored in. Under a blend containing
    # prediction the baselines are still the dataset's own metric (RMSE, lower
    # is better) while `lower_is_better` was just flipped to False for the
    # blend. The best-single-representation comparator reads the baselines, so
    # using `lower_is_better` there took the max of a set of RMSEs and handed
    # the hybrid the WORST representation to beat. Geometry baselines are
    # read-across / MCS correlations, higher-is-better, and _sim_mode already
    # sets lower_is_better False — so this differs from `lower_is_better` only
    # under a model-bearing blend.
    _baselines_lower_is_better = (_side_lower_is_better if _blend_with_model
                                  else lower_is_better)

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
    # Every OTHER number the hybrid scored on each outer test fold, whichever
    # objective picked it: {side_metric_name: {model: [per-outer-fold score]}}.
    # Keys are 'predictive', 'readacross', 'struct_fidelity'. A key is absent
    # when that quantity was not computable (no MCS pairs, no groups). This is
    # what lets two objectives be compared on one table.
    hybrid_side_folds: Dict[str, Dict[str, List[Optional[float]]]] = {}
    single_folds: Dict[str, List[Optional[float]]] = {}
    single_reps: Dict[str, List[Optional[str]]] = {}
    blueprints: Dict[str, List[Optional[dict]]] = {}
    search_means: Dict[str, List[float]] = {}
    # {model: {rep: [per-outer-fold score]}}
    per_rep: Dict[str, Dict[str, List[Optional[float]]]] = {}
    # One entry per outer fold whose inner search raised. All folds failing is
    # a crash, not a null result — see the guard after the fold loop.
    inner_failures: List[str] = []
    # {side_metric_name: {model: {rep: [per-outer-fold score]}}} — the same three
    # numbers as hybrid_side_folds, for every individual representation.
    per_rep_side: Dict[str, Dict[str, Dict[str, List[Optional[float]]]]] = {}
    # {model: {budget: [per-outer-fold score]}} and inner counterpart
    budget_folds: Dict[str, Dict[int, List[Optional[float]]]] = {}
    # {side_metric_name: {model: {budget: [per-outer-fold score]}}}
    budget_side: Dict[str, Dict[str, Dict[int, List[Optional[float]]]]] = {}
    budget_search: Dict[str, Dict[int, List[float]]] = {}
    # {model: {budget: [per-outer-fold Blueprint]}} — the recipe each outer fold
    # picked AT that budget. Reporting code needs the recipe fields
    # (importance/allocation) alongside the honest score.
    budget_bps: Dict[str, Dict[int, List[Optional[Blueprint]]]] = {}
    inner_searches: List[Optional[SearchResult]] = []
    hybrid_failures: List[Tuple[str, int, str]] = []
    # (model, rep, outer fold, message) for every baseline arm that raised.
    baseline_failures: List[Tuple[str, str, int, str]] = []
    # {model: [per-outer-fold hybrid width]} — the real column count.
    hybrid_widths: Dict[str, List[Optional[int]]] = {}
    # {model: [per-outer-fold width of the representation it was compared to]}.
    single_widths: Dict[str, List[Optional[int]]] = {}

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

        y_train, y_test = labels[tr], labels[te]

        # A label-fitted rep cannot be precomputed once for the whole dataset —
        # doing so would fit on rows it is later scored on. Build it here, from
        # THIS fold's training rows only, then slice it like any other rep. The
        # builder sees global row numbers because it closes over full-length
        # sequence/SMILES lists, the same contract rep_factories uses.
        fold_reps = dict(base_reps)
        if outer_rep_builders:
            _n_rows = len(labels)
            for _name, _build in outer_rep_builders.items():
                _mat = np.asarray(_build(np.asarray(tr), y_train))
                if _mat.ndim != 2 or _mat.shape[0] != _n_rows:
                    raise ValueError(
                        f"outer_rep_builders[{_name!r}] returned shape "
                        f"{_mat.shape} on outer fold {fold_i}; expected "
                        f"({_n_rows}, width) — the FULL dataset, not the "
                        f"training subset. nested_cv_evaluate does the slicing.")
                fold_reps[_name] = _mat
            if verbose:
                print(f"  built {len(outer_rep_builders)} label-fitted rep(s) on "
                      f"{len(tr)} training rows: "
                      + ', '.join(f'{k}={fold_reps[k].shape[1]}d'
                                  for k in outer_rep_builders))

        train_reps = {k: v[tr] for k, v in fold_reps.items()}
        test_reps = {k: v[te] for k, v in fold_reps.items()}
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
        # A label-fitted rep's VALUES change with the builder's code, but the
        # cache key sees only rep_names and rep_shapes (content hashes are
        # ignored — see _cache_lookup's ignore_keys). The id carries that change.
        if outer_rep_builders_id:
            _ns = f"{_ns}__orb_{outer_rep_builders_id}"
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

        # --- structural pairs: give the inner search the ORIGINAL row numbers ---
        # The MCS pairs store row numbers into the whole dataset. The inner
        # search only sees the outer-training rows and renumbers them from 0, so
        # without `tr` a stored pair cannot be located and MCS fidelity was only
        # computable out here, on the outer test fold. Same lift already applied
        # to rep_factories a few lines above.
        if structural_pairs is not None:
            inner_kwargs['row_index'] = np.asarray(tr)
            inner_kwargs['structural_pairs'] = structural_pairs

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
            inner_failures.append(f'outer fold {fold_i}: {type(e).__name__}: {e}')
            inner_searches.append(None)
            continue
        inner_searches.append(inner)

        models_here = sorted(inner.best_by_model.keys())
        if not models_here:
            # The search RAN and returned, but produced no winning recipe for
            # any model — every configuration it tried failed to build. It does
            # not raise, because _evaluate_hybrid_config catches per-config
            # failures and _finalize returns empty dicts when nothing survived.
            # Recorded so the all-folds guard below can tell an empty result
            # from a working one. Found 2026-08-27 by asking for a blended
            # objective on the greedy allocator, which refuses blends: every
            # configuration raised, every raise was swallowed, and the run came
            # back with no models and no error.
            inner_failures.append(
                f'outer fold {fold_i}: the search produced no recipe for any '
                f'model — every configuration failed to build')

        # The estimators THIS outer fold's inner search tuned, if any. Empty on
        # every untuned run, which keeps the held-out score identical to before.
        # Tuning happened on outer-train rows only, so re-using the estimator
        # here does not leak the outer test fold.
        _fold_tuned = getattr(inner, 'tuned_models', None) or {}
        _tune_outer = bool(search_kwargs.get('tune', False))
        _tune_iter_outer = int(search_kwargs.get('tune_n_iter', 20))

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
                # Rebuild the recipe under the SAME conditions the inner search
                # selected it under: group-aware internal split, real model,
                # real metric. get_model returns None for an unavailable model
                # (optional dependency); metric_fn is None only when the
                # objective is a geometry one, where _evaluate_allocation
                # ignores the model anyway — pass neither in that case, since a
                # model without a metric is worse than no model at all.
                _rb_model = get_model(model_name, task_type,
                                      n_samples=len(y_train))
                if metric_fn is None:
                    _rb_model = None
                X_tr_h, finfo = apply_blueprint(
                    train_reps, bp, labels=y_train, augmentations=train_augs,
                    groups=train_groups, model=_rb_model,
                    metric_fn=(metric_fn if _rb_model is not None else None),
                    lower_is_better=lower_is_better,
                    row_index=np.asarray(tr),
                    structural_pairs=structural_pairs,
                    is_clf=(task_type == 'classification'))
                bp_obj = bp if isinstance(bp, Blueprint) else Blueprint.from_dict(bp)
                te_bp_reps = {k: test_reps[k] for k in bp_obj.reps if k in test_reps}
                X_te_h = apply_feature_selection(te_bp_reps, finfo)

                aug_info = finfo.get('augmentation_info')
                if aug_info:
                    # A top_k allocation that zeroes a representation moves that
                    # representation's top features into the augmentation pool
                    # INSIDE create_hybrid ("auto-demote"), so its name appears
                    # in augmentation_info while `test_augs` — built from the
                    # caller's augmentation dict — has no entry for it.
                    # apply_augmentation_selection then raised, the whole outer
                    # fold was dropped from the HYBRID arm, and the BASELINE arm
                    # kept it. finfo['demoted_reps'] records exactly which
                    # columns were taken, so the test side takes the same ones.
                    _te_augs = dict(test_augs) if test_augs else {}
                    for _dn, _di in (finfo.get('demoted_reps') or {}).items():
                        if _dn in test_reps:
                            _te_augs[_dn] = test_reps[_dn][:, _di]
                    if _te_augs:
                        aug_te = apply_augmentation_selection(_te_augs, aug_info)
                        if aug_te is not None:
                            X_te_h = np.hstack([X_te_h, aug_te])
                # The width the hybrid ACTUALLY had on this fold. Reporting used
                # to fall back to the requested budget, which is neither an
                # upper nor a lower bound: a representation narrower than its
                # quota gives fewer, and augmentation extras are appended
                # OUTSIDE the budget and give more.
                hybrid_widths.setdefault(model_name, [None] * n_outer)
                hybrid_widths[model_name][fold_i] = int(X_tr_h.shape[1])

                score, _hyb_metrics = _outer_fit_score(
                    X_tr_h, y_train, X_te_h, y_test, model_name, task_type,
                    metric_fn, seed=seed, objective=objective,
                    train_groups=train_groups,
                    structural_pairs=structural_pairs,
                    test_idx=te,
                    test_groups=(groups[te] if groups is not None else None),
                    side_metric_fn=_side_metric_fn,
                    tuned_model=_fold_tuned.get(model_name),
                    tune_per_fold=_tune_outer, tune_n_iter=_tune_iter_outer)
                hybrid_folds[model_name][fold_i] = score
                for _side_key in _SIDE_METRIC_KEYS:
                    _side_val = _hyb_metrics.get(_side_key)
                    if _side_val is None or not np.isfinite(_side_val):
                        continue
                    _per_model = hybrid_side_folds.setdefault(_side_key, {})
                    _slot(_per_model, model_name, None)
                    _per_model[model_name][fold_i] = float(_side_val)
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
            # Each rep gets the same three numbers the hybrid gets. Without the
            # side numbers here, a geometry-picked hybrid's prediction score has
            # nothing to be compared against and cannot be read.
            # fold_reps, not base_reps: a label-fitted rep exists only for THIS
            # fold, and skipping it here would let it into the hybrid while
            # leaving it out of the comparison it is supposed to beat.
            for rep_name, Xr in fold_reps.items():
                _rep_metrics = {}
                try:
                    s, _rep_metrics = _outer_fit_score(
                        Xr[tr], y_train, Xr[te], y_test, model_name, task_type,
                        metric_fn, seed=seed, objective=objective,
                        train_groups=train_groups,
                        structural_pairs=structural_pairs,
                        test_idx=te,
                        test_groups=(groups[te] if groups is not None else None),
                        side_metric_fn=_side_metric_fn,
                        tuned_model=_fold_tuned.get(model_name),
                    tune_per_fold=_tune_outer, tune_n_iter=_tune_iter_outer)
                except Exception as _b_err:
                    # Was a bare `s = nan`. The hybrid arm has printed loudly
                    # since it was written; the baseline arm swallowed the same
                    # failure in silence, so a dead comparator read as
                    # delta = NaN — "no effect" rather than "the arm never ran".
                    s = float('nan')
                    baseline_failures.append(
                        (model_name, rep_name, fold_i,
                         f'{type(_b_err).__name__}: {_b_err}'))
                    print(f"  !! baseline outer-eval FAILED [{model_name}, "
                          f"{rep_name}, outer fold {fold_i}]: "
                          f"{type(_b_err).__name__}: {_b_err}")
                per_rep[model_name].setdefault(rep_name, [None] * n_outer)
                per_rep[model_name][rep_name][fold_i] = (
                    s if np.isfinite(s) else None)
                for _side_key in _SIDE_METRIC_KEYS:
                    _side_val = _rep_metrics.get(_side_key)
                    if _side_val is None or not np.isfinite(_side_val):
                        continue
                    (per_rep_side
                     .setdefault(_side_key, {})
                     .setdefault(model_name, {})
                     .setdefault(rep_name, [None] * n_outer))[fold_i] = (
                        float(_side_val))

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
                pick = (min(cand, key=lambda t: t[1])
                        if _baselines_lower_is_better
                        else max(cand, key=lambda t: t[1]))[0]
                single_reps[model_name][fold_i] = pick
                sv = per_rep[model_name].get(pick, [None] * n_outer)[fold_i]
                single_folds[model_name][fold_i] = sv
                # The comparator's width. The hybrid is held to `budget`
                # features; the representation it is compared against is scored
                # at its FULL native width, which on these datasets is routinely
                # ten times wider. Nothing recorded the two side by side, so a
                # delta could be read as a method result when it was partly a
                # width difference. Recorded, never acted on.
                if pick in fold_reps:
                    single_widths.setdefault(model_name, [None] * n_outer)
                    single_widths[model_name][fold_i] = int(
                        fold_reps[pick].shape[1])

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
                    # Same rebuild conditions as the headline arm above.
                    _bb_model = get_model(model_name, task_type,
                                          n_samples=len(y_train))
                    if metric_fn is None:
                        _bb_model = None
                    Xb_tr, bfinfo = apply_blueprint(
                        train_reps, bbp, labels=y_train,
                        augmentations=train_augs, groups=train_groups,
                        model=_bb_model,
                        metric_fn=(metric_fn if _bb_model is not None else None),
                        lower_is_better=lower_is_better,
                        row_index=np.asarray(tr),
                        structural_pairs=structural_pairs,
                        is_clf=(task_type == 'classification'))
                    bbp_obj = (bbp if isinstance(bbp, Blueprint)
                               else Blueprint.from_dict(bbp))
                    budget_bps[model_name][int(budget)][fold_i] = bbp_obj
                    te_b = {k: test_reps[k] for k in bbp_obj.reps if k in test_reps}
                    Xb_te = apply_feature_selection(te_b, bfinfo)
                    b_aug = bfinfo.get('augmentation_info')
                    if b_aug:
                        # Same auto-demote reconstruction as the headline arm.
                        _b_augs = dict(test_augs) if test_augs else {}
                        for _dn, _di in (bfinfo.get('demoted_reps') or {}).items():
                            if _dn in test_reps:
                                _b_augs[_dn] = test_reps[_dn][:, _di]
                        if _b_augs:
                            a_te = apply_augmentation_selection(_b_augs, b_aug)
                            if a_te is not None:
                                Xb_te = np.hstack([Xb_te, a_te])
                    bs, _bud_metrics = _outer_fit_score(
                        Xb_tr, y_train, Xb_te, y_test, model_name, task_type,
                        metric_fn, seed=seed, objective=objective,
                        train_groups=train_groups,
                        structural_pairs=structural_pairs,
                        test_idx=te,
                        test_groups=(groups[te] if groups is not None else None),
                        side_metric_fn=_side_metric_fn,
                        tuned_model=_fold_tuned.get(model_name),
                    tune_per_fold=_tune_outer, tune_n_iter=_tune_iter_outer)
                    if np.isfinite(bs):
                        budget_folds[model_name][int(budget)][fold_i] = bs
                    for _side_key in _SIDE_METRIC_KEYS:
                        _side_val = _bud_metrics.get(_side_key)
                        if _side_val is None or not np.isfinite(_side_val):
                            continue
                        (budget_side
                         .setdefault(_side_key, {})
                         .setdefault(model_name, {})
                         .setdefault(int(budget), [None] * n_outer))[fold_i] = (
                            float(_side_val))
                except Exception as _bud_err:
                    # Was a bare `pass`. The per-budget arm is a whole reported
                    # column; a budget that never rebuilt showed up as a blank
                    # cell indistinguishable from one that was never requested.
                    print(f"  !! per-budget rebuild FAILED [{model_name}, "
                          f"budget {budget}, outer fold {fold_i}]: "
                          f"{type(_bud_err).__name__}: {_bud_err}")

    # An inner search that failed on EVERY outer fold is a broken run too, and
    # it fails EARLIER than the hybrid arm below: no search means no blueprint,
    # so the hybrid loop never runs and hybrid_failures stays empty. Left alone
    # this returned an empty NestedCVResult with no models in it, which reads
    # downstream as a dataset that produced nothing rather than as a crash.
    # Found 2026-08-27 by asking for objective='struct_fid' with no MCS pairs.
    if inner_failures and len(inner_failures) == n_outer:
        _uniq_inner = sorted(set(inner_failures))
        raise RuntimeError(
            "nested_cv_evaluate: the inner search produced nothing on EVERY "
            f"outer fold ({n_outer} of {n_outer}), so no hybrid was ever "
            "built. Refusing to return an empty result. Distinct causes: "
            f"{_uniq_inner}")

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

    if baseline_failures:
        _bf_by_rep = {}
        for _m, _r, _f, _msg in baseline_failures:
            _bf_by_rep.setdefault((_m, _r), []).append(_f)
        _dead = [(m, r) for (m, r), fs in _bf_by_rep.items()
                 if len(set(fs)) == n_outer]
        print(f"  !! baseline arm: {len(baseline_failures)} "
              f"(model, representation, outer fold) evaluations raised.")
        if _dead:
            print(f"  !! these never scored on ANY outer fold: "
                  f"{sorted(_dead)} — they cannot be the comparator, so the "
                  f"delta is against whatever else survived.")

    # A budget that exists on fewer outer folds than the rest. find_best_hybrid
    # appends n_samples as an at-n boundary budget when a requested budget
    # exceeds the row count, and each outer fold has a DIFFERENT training row
    # count, so each fold can add a budget of its own. Its "honest mean" is then
    # a mean over one fold, printed in the same table as five-fold means.
    for _m, _b_map in budget_folds.items():
        for _b, _folds in _b_map.items():
            _n_scored = sum(1 for v in _folds
                            if v is not None and np.isfinite(v))
            if 0 < _n_scored < n_outer:
                print(f"  !! budget {_b} for {_m} scored on {_n_scored}/"
                      f"{n_outer} outer folds. Its mean is over those "
                      f"{_n_scored} and is NOT comparable with the "
                      f"{n_outer}-fold budgets beside it.")

    # The hybrid is held to `budget` features; the representation it is measured
    # against gets its full native width. On these datasets that is routinely a
    # factor of ten, and no output has ever put the two numbers next to each
    # other — so a delta reads as a method result when part of it is a width
    # difference. Said once per model, from fold 0.
    for _m, _hw in hybrid_widths.items():
        _sw = single_widths.get(_m) or []
        _pairs = [(h, sv) for h, sv in zip(_hw, _sw)
                  if h is not None and sv is not None]
        if not _pairs:
            continue
        _h0, _s0 = _pairs[0]
        if _s0 >= 2 * _h0:
            print(f"  !! {_m}: the hybrid has {_h0} features; the single "
                  f"representation it is compared against has {_s0}. The "
                  f"comparison is not at equal width.")

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
        # direction-corrected so positive = optimistic. The two means must cover
        # the SAME outer folds: search_means gains an entry for every fold that
        # produced a recipe, while hv_f only keeps folds whose honest rebuild
        # scored finitely. When a rebuild fails on some folds and not others the
        # difference subtracts means over two different fold sets, so it is not
        # the paired quantity it is printed as.
        if (np.isfinite(search_mean) and np.isfinite(hybrid_mean)
                and len(sm) == len(hv_f)):
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
            # NaN, not 0.0, when one fold contributed: np.std of a single value
            # is exactly zero, which reads downstream as a measured spread of
            # zero rather than as no spread having been measured.
            'hybrid_std': (float(np.std(hv_f)) if len(hv_f) > 1
                           else float('nan')),
            'best_single_mean': float(np.mean(sv_f)) if sv_f else float('nan'),
            'best_single_std': (float(np.std(sv_f)) if len(sv_f) > 1
                                else float('nan')),
            'per_fold_delta': per_fold_delta,
            # THE number to report: mean of the paired per-fold differences.
            'delta_mean_of_folds': (float(np.mean(paired)) if paired
                                    else float('nan')),
            'n_paired_folds': len(paired),
            'search_mean': search_mean,
            'bias': bias,
            # The hybrid's real column count per outer fold, and its mean.
            # Reporting used to substitute the requested budget, which the real
            # width matches only by coincidence.
            'n_features_per_fold': hybrid_widths.get(model_name,
                                                     [None] * n_outer),
            # The comparator's own width, per fold, beside the hybrid's.
            'best_single_n_features_per_fold': single_widths.get(
                model_name, [None] * n_outer),
            'n_features': (int(round(float(np.mean(
                [w for w in hybrid_widths.get(model_name, []) if w is not None]))))
                if any(w is not None
                       for w in hybrid_widths.get(model_name, [])) else None),
        }
        # Every OTHER number this hybrid scored on the outer test folds, so a
        # hybrid picked by one objective can be read on the others' terms. The
        # objective's own number stays THE primary/delta metric above; these are
        # recorded, never selected on. A key is absent when the quantity was not
        # computable on any fold.
        #   'struct_fidelity_*' keep their old names so existing readers of a
        #   similarity run still find them.
        for _side_key, _per_model in hybrid_side_folds.items():
            _folds = _per_model.get(model_name)
            if _folds is None:
                continue
            _vals = _fin(_folds)
            _prefix = ('struct_fidelity' if _side_key == 'struct_fidelity'
                       else f'side_{_side_key}')
            by_model[model_name][f'{_prefix}_fold_scores'] = _folds
            by_model[model_name][f'{_prefix}_mean'] = (
                float(np.mean(_vals)) if _vals else float('nan'))
            # Paired against the SAME best single representation the delta above
            # uses, so a side number is a comparison and not a bare figure.
            _rep_side = per_rep_side.get(_side_key, {}).get(model_name, {})
            # Orient the difference so positive always means the hybrid is
            # better, exactly as delta_mean_of_folds above does. The two
            # geometry numbers are Spearman correlations and are always
            # higher-is-better; the prediction number carries the dataset's own
            # metric, so on rmse a SMALLER number is the better one. Without
            # this the prediction side-delta reported the opposite sign to the
            # primary delta on every lower-is-better dataset.
            _side_lower = (_side_lower_is_better
                           if _side_key == 'predictive' else False)
            _paired_side = []
            for _fi in range(n_outer):
                _pick = single_reps.get(model_name, [None] * n_outer)[_fi]
                if _pick is None:
                    continue
                _h = _folds[_fi]
                _b = _rep_side.get(_pick, [None] * n_outer)[_fi]
                if _h is None or _b is None:
                    continue
                _paired_side.append(float(_b) - float(_h) if _side_lower
                                    else float(_h) - float(_b))
            by_model[model_name][f'{_prefix}_delta_mean_of_folds'] = (
                float(np.mean(_paired_side)) if _paired_side else float('nan'))
            by_model[model_name][f'{_prefix}_n_paired_folds'] = len(_paired_side)

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
            # The other two numbers for this representation alone, so the
            # hybrid's side numbers have something to be compared against.
            for _side_key, _pm in per_rep_side.items():
                _sfs = _pm.get(model_name, {}).get(rep_name)
                if _sfs is None:
                    continue
                _sv = _fin(_sfs)
                _prefix = ('struct_fidelity' if _side_key == 'struct_fidelity'
                           else f'side_{_side_key}')
                baselines_by_model[model_name][rep_name][
                    f'{_prefix}_fold_scores'] = _sfs
                baselines_by_model[model_name][rep_name][f'{_prefix}_mean'] = (
                    float(np.mean(_sv)) if _sv else float('nan'))

    by_model_budget: Dict[str, Dict[int, dict]] = {}
    for model_name, bd in budget_folds.items():
        by_model_budget[model_name] = {}
        for budget, fs in sorted(bd.items()):
            f = _fin(fs)
            sm = budget_search.get(model_name, {}).get(budget, [])
            s_mean = float(np.mean(sm)) if sm else float('nan')
            h_mean = float(np.mean(f)) if f else float('nan')
            # Same fold-set requirement as the headline bias above.
            if (np.isfinite(s_mean) and np.isfinite(h_mean)
                    and len(sm) == len(f)):
                b = ((h_mean - s_mean) if lower_is_better else (s_mean - h_mean))
            else:
                b = float('nan')
            bps_b = [x for x in
                     (budget_bps.get(model_name, {}).get(budget) or []) if x]
            by_model_budget[model_name][budget] = {
                'honest_mean': h_mean,
                'honest_std': float(np.std(f)) if len(f) > 1 else float('nan'),
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

    # The other two numbers per budget, same shape, so the budget question can
    # be asked of any of the three and not only of the one that picked.
    by_model_budget_side: Dict[str, Dict[str, Dict[int, dict]]] = {}
    for _side_key, _per_model in budget_side.items():
        by_model_budget_side[_side_key] = {}
        for model_name, bd in _per_model.items():
            by_model_budget_side[_side_key][model_name] = {}
            for budget, fs in sorted(bd.items()):
                f = _fin(fs)
                by_model_budget_side[_side_key][model_name][budget] = {
                    'mean': float(np.mean(f)) if f else float('nan'),
                    'std': float(np.std(f)) if f else float('nan'),
                    'fold_scores': fs,
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
        side_metric_name=_side_metric_name,
        side_metric_lower_is_better=_side_lower_is_better,
        by_model_budget_side=by_model_budget_side,
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
