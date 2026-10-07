"""Disk caches for work the noise suite repeats with identical inputs.

Plan section 3. Every wrapper below computes something whose result is fixed
by its inputs, stores it, and hands back the stored copy the next time the same
inputs arrive. None of them removes a sweep axis.

THE KEY IDEA: noisy-label work is keyed on a HASH OF THE LABEL VECTOR, never on
the noise condition's name. At level 0 every condition hands over the clean
labels, so all of them share one entry automatically, and so does every
experiment that injects the same draw.

WHERE: $KIRBY_CACHE_DIR/noise_cache/<dataset>/<kind>/<key>.{npz,pkl}, through
kirby.paths.cache_dir. `dataset` is used for the path only; it is never in a
key, so it cannot make two different inputs collide or two identical ones
miss.

WHAT A KEY HOLDS. Content hashes of every array the computation reads (dtype,
shape and bytes, so a float32 and a float64 copy of the same numbers are two
entries -- n07, n09 and n11-n13 do not cast to float64 and the others do), the
settings the computation reads, and three process settings that change the
answer without appearing in any argument:
  lgbm_n_jobs       kirby.hybrid.lgbm_n_jobs(), what _shared and hybrid.py fit
                    LightGBM with (KIRBY_LGBM_THREADS, default -1)
  oof_lgb_threads   kirby.utils.noisy_labels.core._lgb_threads(), what
                    oof_predict fits with (KIRBY_LGBM_THREADS, default 1)
  distance_metric   the process-wide distance (--distance-metric /
                    KIRBY_DISTANCE_METRIC), which every detector with
                    metric=None and make_nearest_neighbors read
No representation names and no version field, per the locked rules: a code
change means wipe the directory and rerun. The one place a name enters is
cached_hybrid_for_fold with want_feature_info=True, because the returned
feature info is keyed by representation name; see its docstring.

KIRBY_NOISE_CACHE=0 disables every cache: each wrapper computes and never
reads or writes. Counted as 'bypass'.

Hits, misses and bypasses are counted per kind. report() returns them. At exit
one line per kind is printed, and when KIRBY_NOISE_CACHE_REPORT names a path
the same dict is written there as JSON.

A failed computation (a detector returning None, a hybrid build returning
(None, None)) is returned and NOT stored, so a failure is never replayed as a
result.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import pickle
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def array_hash(*arrays) -> str:
    """blake2b over each array's dtype, shape and bytes, in order.

    None hashes as its own token, so (None, x) and (x,) differ. Object arrays
    (SMILES lists) are hashed through their str() values. Lists and tuples are
    turned into arrays first.
    """
    h = hashlib.blake2b(digest_size=20)
    for a in arrays:
        if a is None:
            h.update(b'<None>')
            continue
        arr = np.asarray(a)
        if arr.dtype == object or arr.dtype.kind in ('U', 'S'):
            h.update(b'<str>' + str(arr.shape).encode())
            for v in arr.ravel():
                s = str(v).encode()
                h.update(len(s).to_bytes(8, 'little') + s)
            continue
        arr = np.ascontiguousarray(arr)
        h.update(str(arr.dtype.str).encode() + b'|' + str(arr.shape).encode()
                 + b'|')
        h.update(arr.tobytes())
    return h.hexdigest()


def _canon(v):
    """A JSON-able canonical form of a key value. Arrays are refused: hash
    them with array_hash first, so a key never silently holds a repr."""
    if isinstance(v, np.ndarray):
        raise TypeError('put array_hash(arr) in a key, not the array')
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in sorted(v.items(),
                                                     key=lambda kv: str(kv[0]))}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    raise TypeError(f'cannot put {type(v).__name__} in a cache key')


def key_hash(key: dict) -> str:
    """blake2b of the canonical JSON of `key` (sorted keys)."""
    blob = json.dumps(_canon(key), sort_keys=True, separators=(',', ':'))
    return hashlib.blake2b(blob.encode(), digest_size=20).hexdigest()


def process_settings() -> dict:
    """The process settings every key carries. See the module docstring.

    Added in review 2026-09-25, three more that change a cached value without
    appearing in any argument:
      fill_zero_gain       KIRBY_FILL_ZERO_GAIN=1 changes how many features a
                           hybrid build takes from a representation whose
                           importance is mostly zero (hybrid.py:5444)
      importance_exponent  KIRBY_IMPORTANCE_EXPONENT, the power fold_weights
                           raises importance weights to (preprocessing.py:431)
      omp_threads          OMP_NUM_THREADS, which sets torch's thread count
                           for the deeplift importance fits and the dnn model;
                           a float sum over a different thread count can land
                           on a different last bit. When unset, torch uses the
                           machine's cores, so the core count is recorded.
    """
    from kirby.hybrid import lgbm_n_jobs
    from kirby.utils.noisy_labels.core import _lgb_threads
    from kirby.similarity.distance_config import get_distance_metric
    from kirby.similarity.preprocessing import importance_exponent_override
    omp = os.environ.get('OMP_NUM_THREADS')
    return {'lgbm_n_jobs': int(lgbm_n_jobs()),
            'oof_lgb_threads': int(_lgb_threads()),
            'distance_metric': get_distance_metric(),
            'fill_zero_gain': os.environ.get('KIRBY_FILL_ZERO_GAIN') == '1',
            'importance_exponent': importance_exponent_override(),
            'omp_threads': (str(omp).strip() if omp and str(omp).strip()
                            else f'unset:{os.cpu_count()}')}


def model_signature(model) -> dict:
    """What a downstream model instance is, beyond the name a caller gives it.

    A name alone does not fix the model: get_model_factories hands back a
    classifier or a regressor under the same name depending on task_type, and
    two callers can build 'rf' with different settings. The class and every
    get_params() value (as repr, so nested estimators and kernels are
    included) go in the key. A model with no get_params falls back to its
    repr; if that repr carries a memory address the entry simply never hits
    again, which costs a recompute and never hands back a wrong value.
    """
    if model is None:
        return None
    cls = type(model)
    sig = {'class': f'{cls.__module__}.{cls.__qualname__}'}
    get_params = getattr(model, 'get_params', None)
    if callable(get_params):
        try:
            params = get_params(deep=True)
        except TypeError:
            params = get_params()
        sig['params'] = {str(k): repr(v) for k, v in sorted(
            params.items(), key=lambda kv: str(kv[0]))}
    else:
        sig['repr'] = repr(model)
    return sig


# ---------------------------------------------------------------------------
# Store, counters, report
# ---------------------------------------------------------------------------

_COUNTS = defaultdict(lambda: {'hits': 0, 'misses': 0, 'bypass': 0, 'unreadable': 0})


def enabled() -> bool:
    return os.environ.get('KIRBY_NOISE_CACHE', '1').strip() != '0'


def cache_root() -> Path:
    from kirby.paths import cache_dir
    return cache_dir('noise_cache', create=False)


def _path(dataset, kind, digest, ext):
    return cache_root() / str(dataset) / kind / f'{digest}.{ext}'


def _is_plain_array(v):
    return isinstance(v, np.ndarray) and v.dtype != object


def _atomic_write(path: Path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix='.tmp_')
    try:
        with os.fdopen(fd, 'wb') as fh:
            writer(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_once(dataset, kind, digest):
    p = _path(dataset, kind, digest, 'npz')
    if p.exists():
        with np.load(p, allow_pickle=False) as z:
            return True, z['value']
    p = _path(dataset, kind, digest, 'pkl')
    if p.exists():
        with open(p, 'rb') as fh:
            return True, pickle.load(fh)
    return False, None


def _load(dataset, kind, digest, attempts=3):
    """A stored entry, or (False, None). Never raises on an unreadable file.

    On the shared filesystem another job can replace an entry while this one
    reads it (_atomic_write's os.replace), and the reader then gets
    "OSError: [Errno 116] Stale file handle" or "File is not a zip file".
    That crashed n07 on the server (traceback through
    cached_detector_scores, 2026-09-28). Re-open a few times; if it still cannot be read, treat
    it as a miss -- the value is recomputed and the entry rewritten.
    """
    import time
    import zipfile
    last = None
    for i in range(attempts):
        try:
            return _load_once(dataset, kind, digest)
        except (OSError, EOFError, zipfile.BadZipFile, ValueError,
                pickle.UnpicklingError, KeyError) as exc:
            last = exc
            time.sleep(0.5 * (i + 1))
    _COUNTS[kind]['unreadable'] += 1
    print(f"  [noise_cache] {kind}/{digest}: unreadable after {attempts} tries "
          f"({type(last).__name__}: {last}); recomputing")
    return False, None


def _store(dataset, kind, digest, value):
    if _is_plain_array(value):
        _atomic_write(_path(dataset, kind, digest, 'npz'),
                      lambda fh: np.savez(fh, value=value))
    else:
        _atomic_write(_path(dataset, kind, digest, 'pkl'),
                      lambda fh: pickle.dump(value, fh,
                                             protocol=pickle.HIGHEST_PROTOCOL))


def cached(dataset, kind, key, compute, *, store_if=None):
    """The generic wrapper. Returns compute() or the stored copy of it.

    `key` is a dict of canonical values; the process settings are added here.
    `store_if(value)` False means return without storing (a failure).
    """
    if not enabled():
        _COUNTS[kind]['bypass'] += 1
        return compute()
    full = dict(key)
    full['_kind'] = kind
    full['_process'] = process_settings()
    digest = key_hash(full)
    hit, value = _load(dataset, kind, digest)
    if hit:
        _COUNTS[kind]['hits'] += 1
        return value
    _COUNTS[kind]['misses'] += 1
    value = compute()
    if store_if is None or store_if(value):
        _store(dataset, kind, digest, value)
    return value


def report() -> dict:
    """{kind: {'hits', 'misses', 'bypass'}} for this process so far."""
    return {k: dict(v) for k, v in sorted(_COUNTS.items())}


def reset_counts():
    _COUNTS.clear()


def _at_exit():
    rep = report()
    if not rep:
        return
    for kind, c in rep.items():
        print(f"noise_cache {kind}: {c['hits']} hits, {c['misses']} misses, "
              f"{c['bypass']} bypassed")
    path = os.environ.get('KIRBY_NOISE_CACHE_REPORT')
    if path:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as fh:
                json.dump(rep, fh, indent=1, sort_keys=True)
        except OSError as exc:
            print(f"noise_cache: could not write {path}: {exc}")


atexit.register(_at_exit)


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

def cached_scaled(dataset, X, train_idx, val_idx, *, dtype=None):
    """(X_tr_s, X_va_s): kirby.scaling.FeatureScaler fitted on X[train_idx],
    applied to both: 0/1 features left as they are, the rest standardised
    (Adelaide 2026-10-05). Until then every feature was standardised; the key
    carries 'scaler' so an entry written under the old rule is never served.

    The pattern every noise experiment writes inline (n07:2489-2491:
    `sc.fit_transform(X_rep[train_idx])`, `sc.transform(X_rep[val_idx])`).
    `dtype` casts X first; None keeps X's own dtype, and the key records the
    dtype either way, so a float32 and a float64 caller never share an entry.

    Deterministic: the scaler has no randomness. Keyed on the content of X
    and the two index vectors, never on a representation name.
    """
    from kirby.scaling import FeatureScaler
    A = np.asarray(X) if dtype is None else np.asarray(X, dtype=dtype)
    tr = np.asarray(train_idx)
    va = np.asarray(val_idx)
    key = {'X': array_hash(A), 'train_idx': array_hash(tr),
           'val_idx': array_hash(va), 'scaler': 'binary_unscaled_v1'}

    def _compute():
        sc = FeatureScaler()
        return (sc.fit_transform(A[tr]), sc.transform(A[va]))
    return cached(dataset, 'scaled', key, _compute)


def predict_mode_for(metric_name):
    """'proba' for auc, 'predict' otherwise -- n07._predict_for_metric's rule
    (n07:359-372)."""
    return 'proba' if metric_name == 'auc' else 'predict'


def _predict(model, X, predict_mode):
    if predict_mode == 'proba' and hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        if getattr(proba, 'ndim', 1) == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba
    if predict_mode not in ('predict', 'proba'):
        raise ValueError(f"predict_mode must be 'predict' or 'proba', got "
                         f"{predict_mode!r}")
    return model.predict(X)


# ── Models whose fit() takes no per-molecule weight ───────────────────────
# Added 2026-10-07. sklearn's GaussianProcessRegressor.fit(X, y) has no
# sample_weight. A cleanup that hands back weights instead of new labels
# (n07 downweight, huber_weight, invvar_weight; n02's reliability weights)
# cannot be applied to it. That cell is recorded as not run, with this reason,
# the rule n02, n11, n12, n13, n20, n21, n31, n42 and c19 already followed.
# Before this, the fit raised: uncaught on the D-versus-D' rows (n02 and n07
# units failed on the five datasets where gp is a decided model), caught and
# left unexplained elsewhere (n18, n20, n24, n25, n35, n39).
WEIGHTS_NOT_RUN = ("not run: this model's fit() takes no per-molecule weight "
                   "(sklearn GaussianProcessRegressor), so a cleanup that "
                   "weights molecules cannot be applied to it")


class WeightsNotAccepted(TypeError):
    """A per-molecule weight was handed to a model whose fit() takes none."""


def fit_takes_sample_weight(model_or_factory):
    """True when the model's fit() accepts sample_weight. Takes a built model
    or a zero-argument factory. A fit(**kwargs) counts as accepting it."""
    import inspect
    m = model_or_factory
    if not hasattr(m, 'fit') and callable(m):
        m = m()
    try:
        params = inspect.signature(m.fit).parameters
    except (TypeError, ValueError):
        return False
    return ('sample_weight' in params
            or any(p.kind == p.VAR_KEYWORD for p in params.values()))


def require_weights_accepted(model_or_factory, sample_weight):
    """Raise WeightsNotAccepted (message WEIGHTS_NOT_RUN) when sample_weight is
    given and the model cannot take it. Nothing happens otherwise."""
    if sample_weight is not None and not fit_takes_sample_weight(
            model_or_factory):
        raise WeightsNotAccepted(WEIGHTS_NOT_RUN)


def cached_fit_predict(dataset, model_name, make_model, X_tr, y_tr, X_va,
                       sample_weight=None, predict_mode='predict'):
    """Fit make_model() on (X_tr, y_tr[, sample_weight]) and predict X_va.

    Deterministic only when the factory seeds its model. The downstream
    factories (_shared.get_model_factories -> find_best_hybrid.get_model) do
    for rf, svm, lightgbm and xgboost. `model_name` and the built instance's
    class and parameters (model_signature) are in the key, so a classifier and
    a regressor filed under one name never share an entry. A caller must not
    pass a factory whose
    fit is not reproducible (an unseeded one, or a torch model on a
    multithreaded run) under a name that says it is.

    `predict_mode` 'proba' returns the positive-class probability for a binary
    classifier, as n07._predict_for_metric does for auc; predict_mode_for
    derives it from the metric name.
    """
    # The instance the factory builds is in the key, not only its name
    # (model_signature). Building one is cheap; fitting is what is cached.
    probe = make_model()
    require_weights_accepted(probe, sample_weight)
    use_kernel = svm_kernel_applies(probe, X_tr)
    key = {'model': str(model_name), 'model_signature': model_signature(probe),
           'X_tr': array_hash(X_tr),
           'y_tr': array_hash(y_tr), 'X_va': array_hash(X_va),
           'sample_weight': array_hash(sample_weight),
           'predict_mode': str(predict_mode)}
    if use_kernel:
        # A different computation from the usual fit (see
        # svm_kernel_fit_predict), so it never shares an entry with one.
        key['svm_kernel'] = 'precomputed_rbf_scale'

    def _compute():
        if use_kernel:
            return svm_kernel_fit_predict(probe, X_tr, y_tr, X_va,
                                          sample_weight=sample_weight)
        m = probe
        if sample_weight is None:
            m.fit(X_tr, y_tr)
        else:
            m.fit(X_tr, y_tr, sample_weight=np.asarray(sample_weight))
        return np.asarray(_predict(m, X_va, predict_mode))
    return cached(dataset, 'fit_predict', key, _compute)


# ── The SVM's kernel, computed once per feature matrix ────────────────────
# Added 2026-10-04. Within one fold an experiment fits the regression SVM many
# times on the SAME features with DIFFERENT labels (one fit per cleanup,
# setting and control). The RBF kernel depends on the features only, so it is
# computed once per (training features, held-out features) and every fit is
# handed kernel='precomputed'.
#
# MEASURED on logd fold 0, mhg_gnn, 4,022 training molecules x 1,024 features:
# 9.5 s a fit the usual way, 0.6 s for the kernel once, then 0.5 s a fit.
# Predictions within 0.00099 logD units of the usual fit, the size of the
# solver's own stopping tolerance (tol=1e-3); on a toy set the scores agree to
# 2e-16. The same model: the repo's SVRScaledTarget (labels standardised), the
# gamma that gamma='scale' would pick, every other parameter unchanged.
#
# Only a regression SVM with kernel='rbf' and gamma='scale'; a classifier (SVC)
# keeps the usual fit. Above SVM_KERNEL_MAX_ROWS training molecules the usual
# fit is used: the kernel is n x n floats (1.15 GB at 12,000).
SVM_KERNEL_MAX_ROWS = 12000
_KERNELS = {}            # (hash X_tr, hash X_va, gamma) -> (K_tr, K_va)
_KERNELS_KEEP = 2


def svm_kernel_applies(model, X_tr):
    from sklearn.svm import SVR
    if not isinstance(model, SVR):
        return False
    p = model.get_params()
    return (p.get('kernel') == 'rbf' and p.get('gamma') == 'scale'
            and np.asarray(X_tr).shape[0] <= SVM_KERNEL_MAX_ROWS)


def svm_kernel_fit_predict(model, X_tr, y_tr, X_va, sample_weight=None):
    """The model's own fit and predict, on a kernel computed once per
    (X_tr, X_va) in this process. Returns the predictions on X_va."""
    from sklearn.metrics.pairwise import rbf_kernel
    A = np.asarray(X_tr, dtype=np.float64)
    B = np.asarray(X_va, dtype=np.float64)
    var = float(A.var())
    gamma = 1.0 / (A.shape[1] * var) if var > 0 else 1.0
    k = (array_hash(A), array_hash(B), gamma)
    if k not in _KERNELS:
        while len(_KERNELS) >= _KERNELS_KEEP:
            _KERNELS.pop(next(iter(_KERNELS)))
        _KERNELS[k] = (rbf_kernel(A, A, gamma=gamma),
                       rbf_kernel(B, A, gamma=gamma))
    K_tr, K_va = _KERNELS[k]
    params = {p: v for p, v in model.get_params().items()
              if p not in ('kernel', 'gamma', 'degree', 'coef0')}
    m = type(model)(kernel='precomputed', **params)
    if sample_weight is None:
        m.fit(K_tr, y_tr)
    else:
        m.fit(K_tr, y_tr, sample_weight=np.asarray(sample_weight))
    return np.asarray(m.predict(K_va))


def cached_oof_predict(dataset, X, y, groups, n_splits=5, seed=0,
                       sample_weight=None, params=None):
    """kirby.utils.noisy_labels.oof_predict (core.py:151), the out-of-fold
    LightGBM target n07's oof_smooth and js_shrink correct toward and that
    n24, n25, n30-n34, n38, n40 and n41 compute for the same draw.

    Deterministic: GroupKFold has no randomness and the LightGBM parameters
    are fixed (core.py:30-40); the thread count it reads is in every key.
    The `splitter=` argument is not offered: a callable cannot be keyed.
    """
    from kirby.utils.noisy_labels import oof_predict
    key = {'X': array_hash(X), 'y': array_hash(y), 'groups': array_hash(groups),
           'n_splits': int(n_splits), 'seed': int(seed),
           'sample_weight': array_hash(sample_weight),
           'params': params}

    def _compute():
        return np.asarray(oof_predict(X, y, groups, n_splits=n_splits,
                                      seed=seed, sample_weight=sample_weight,
                                      params=params))
    return cached(dataset, 'oof_predict', key, _compute)


def _neighbour_lists_hash(nbrs):
    """Content hash of every neighbour array in a neighbour_lists dict: the
    exact per-k lists MODEL_INPUT carries, and idx_train / dist_train."""
    exact = nbrs.get('exact_by_k')
    return {'setting': nbrs.get('setting_name'),
            'train': array_hash(nbrs.get('idx_train'), nbrs.get('dist_train')),
            'exact_by_k': (None if exact is None else
                           [[int(k), array_hash(exact[k][0], exact[k][1])]
                            for k in sorted(exact)])}


# The four estimators that take precomputed_neighbors
# (src/kirby/utils/noise_estimation.py, _neighbours_for, added 2026-09-25).
NEIGHBOUR_DETECTORS = ('neighbor_residual', 'local_variance_ratio',
                       'distance_weighted', 'distance_weighted_residual',
                       'activity_cliff_aware')


def _detector_scores_from_lists(X_tr_s, y_tr, is_clf, name, groups, nbrs, k):
    """One neighbour detector's scores with its neighbours read from `nbrs`.

    Builds the estimator with the arguments n07._compute_noise_scores offers
    it (n07:452-453: k=10, metric=None, is_clf, exclude_groups=groups) plus
    precomputed_neighbors=nbrs. With k=None the estimator's own count is used:
    min(10, n-1) (noise_estimation.py, every neighbour estimator's
    estimate()). A failed fit prints and returns None, as n07 does
    (n07:476-481).
    """
    import inspect
    from kirby.utils.noise_estimation import ESTIMATOR_REGISTRY
    cls = ESTIMATOR_REGISTRY[name]
    offered = {'k': 10 if k is None else int(k), 'metric': None,
               'is_clf': is_clf, 'exclude_groups': groups,
               'precomputed_neighbors': nbrs}
    accepts = set(inspect.signature(cls.__init__).parameters)
    # A k the MODEL_INPUT lists do not hold is a configuration error, not a
    # failed fit: raised here, outside the try, so it is not turned into None.
    exact = nbrs.get('exact_by_k')
    k_use = int(min(offered['k'], len(y_tr) - 1))
    if exact is not None and k_use not in exact:
        raise KeyError(f"MODEL_INPUT lists hold k in {sorted(exact)}, the "
                       f"estimator asks for {k_use}")
    try:
        est = cls(**{a: v for a, v in offered.items() if a in accepts})
        est.fit(X_tr_s, y_tr)
        return est.noise_scores_
    except Exception as exc:
        print(f"      noise_scores estimator failed: "
              f"{type(exc).__name__}: {exc}")
        return None


def cached_detector_scores(dataset, X_tr_s, y_tr, is_clf=False, detector=None,
                           groups=None, similarity_neighbours=None,
                           similarity_k=None):
    """n07._compute_noise_scores (n07:399-480): one detector's suspicion
    scores on (X_tr_s, y_tr).

    Deterministic: every estimator it builds is seeded (random_state=42) and
    takes k=10 and the process-wide metric, which is in the key. A None
    (failed fit) is returned and not stored.

    similarity_neighbours, added 2026-09-25 (plan section 1). None, the
    default, is n07._compute_noise_scores unchanged. Given the dict
    similarity_space.neighbour_lists (or noise_cache.cached_neighbours)
    returns, the estimator is built HERE, not through n07, with
    precomputed_neighbors set to it, so the detector judges each molecule
    against its neighbours in the similarity setting's features while
    anything else it computes still reads X_tr_s. similarity_k is how many
    neighbours; None keeps the estimator's own min(10, n-1).

    Rules, each checked:
      - Only the four neighbour estimators (NEIGHBOUR_DETECTORS) read a
        neighbour list. Any other detector -- cross_val, n07's default --
        would ignore it and write today's numbers under the setting's name, so
        it is refused with ValueError, the same rule n07.cleaned_training_set
        applies (n07:703-722).
      - The lists are taken as given: exclude_groups is NOT reapplied to them
        (noise_estimation._neighbours_for). Until 2026-10-05 a list in which a
        molecule had a neighbour from its own scaffold group was refused when
        `groups` was given. That check is gone: the owner decided on 2026-10-05
        that a molecule's scaffold-mates stay in its neighbour list, because
        removing them made finding corrupted labels WORSE on all four f38
        datasets -- worse in every one of herg's 942 paired cells, and by a
        median of 0.0021 to 0.0407 detection AUROC on the other three
        (tests/experiments/framework/KNN_SEARCH_SPACE_RECORD.md, axis 6).
        `groups` is still part of the cache key, so lists built either way keep
        separate entries.
      - The MODEL_INPUT setting's lists, built with exclude_groups=groups,
        give the same scores as similarity_neighbours=None bit for bit
        (tests/smoke/smoke_noise_similarity_foundation.py, check b).
    The key adds the content hash of every neighbour array and similarity_k.
    """
    from experiments.noise import n07_remediation as n07
    name = n07.DETECTOR if detector is None else str(detector)
    key = {'X': array_hash(X_tr_s), 'y': array_hash(y_tr),
           'is_clf': bool(is_clf), 'detector': name,
           'groups': array_hash(groups)}

    if similarity_neighbours is None:
        def _compute():
            return n07._compute_noise_scores(X_tr_s, y_tr, is_clf=is_clf,
                                             detector=name, groups=groups)
    else:
        if name not in NEIGHBOUR_DETECTORS:
            raise ValueError(
                f"cached_detector_scores: detector {name!r} does not read a "
                f"neighbour list, so similarity setting "
                f"{similarity_neighbours.get('setting_name')!r} would be "
                f"ignored. Neighbour detectors: {NEIGHBOUR_DETECTORS}.")
        if not similarity_neighbours.get('available', False):
            raise ValueError(f"similarity_neighbours is unavailable: "
                             f"{similarity_neighbours.get('reason')}")
        key['similarity_neighbours'] = _neighbour_lists_hash(
            similarity_neighbours)
        key['similarity_k'] = None if similarity_k is None else int(similarity_k)
        # How much each neighbour counts reaches the neighbour detectors since
        # 2026-10-06 (kirby.utils.noise_estimation._scheme_weights).
        key['scheme'] = similarity_neighbours.get('scheme')

        def _compute():
            return _detector_scores_from_lists(X_tr_s, y_tr, is_clf, name,
                                               groups, similarity_neighbours,
                                               similarity_k)
    out = cached(dataset, 'detector_scores', key, _compute,
                 store_if=lambda v: v is not None)
    return None if out is None else np.asarray(out)


def cached_neighbours(dataset, setting, *, spaces_train, spaces_val, y_train,
                      groups_train=None, exclude_groups=None, smiles_train=None,
                      smiles_val=None, is_clf=False):
    """similarity_space.neighbour_lists, same arguments.

    The key holds the setting, the content of only the matrices that setting
    reads (its own space, or 'model_input' for MODEL_INPUT and
    forest_proximity), the SMILES for Tanimoto, and the labels only when they
    are read -- a weighting other than 'none', or the forest. So an unweighted
    feature space shares one entry across every noise condition, level and
    draw of a fold. KIRBY_IMPORTANCE_EXPONENT is in the key when weighted.

    Deterministic: cdist and a stable sort; the importance fits and the forest
    are seeded. An unavailable space ('available': False) is returned and not
    stored.
    """
    from experiments.noise import similarity_space as ss
    sp = setting.space
    read_key = 'model_input' if sp in (ss.MODEL_INPUT_SPACE,
                                       ss.FOREST_SPACE) else sp
    reads_labels = (setting.weighting != 'none') or sp == ss.FOREST_SPACE
    key = {'setting': setting.to_dict(),
           'train': (array_hash((spaces_train or {}).get(read_key))
                     if sp != ss.TANIMOTO_SPACE else None),
           'val': (None if spaces_val is None else
                   array_hash(spaces_val.get(read_key))
                   if sp != ss.TANIMOTO_SPACE else None),
           'want_val': spaces_val is not None,
           'smiles_train': (array_hash(np.asarray(list(smiles_train), dtype=object))
                            if sp == ss.TANIMOTO_SPACE and smiles_train is not None
                            else None),
           'smiles_val': (array_hash(np.asarray(list(smiles_val), dtype=object))
                          if sp == ss.TANIMOTO_SPACE and smiles_val is not None
                          else None),
           'y': array_hash(y_train) if reads_labels else None,
           'is_clf': bool(is_clf) if sp == ss.FOREST_SPACE else None,
           'exclude_groups': array_hash(exclude_groups),
           'n_train': int(len(y_train)),
           'importance_exponent': None}
    if setting.weighting != 'none':
        from kirby.similarity.preprocessing import importance_exponent_override
        key['importance_exponent'] = importance_exponent_override()

    def _compute():
        return ss.neighbour_lists(setting, spaces_train=spaces_train,
                                  spaces_val=spaces_val, y_train=y_train,
                                  groups_train=groups_train,
                                  exclude_groups=exclude_groups,
                                  smiles_train=smiles_train,
                                  smiles_val=smiles_val, is_clf=is_clf)
    return cached(dataset, 'neighbours', key, _compute,
                  store_if=lambda v: bool(v and v.get('available')))


def cached_forest_leaf_ids(dataset, X_train, y_train, is_clf, X_apply=None):
    """(train leaf ids, X_apply leaf ids or None) under the forest
    noise_estimation.forest_leaf_ids fits (noise_estimation.py:2119), through
    similarity_space._forest_leaves, which builds the same forest and also
    applies it to X_apply. The training ids equal forest_leaf_ids's.

    Deterministic: FOREST_SEED is fixed and n_jobs does not change the trees.
    n25's prox_smooth refits this about 9 times per pass.
    """
    from experiments.noise.similarity_space import _forest_leaves
    from kirby.utils.noise_estimation import (
        FOREST_TREES, FOREST_MIN_SAMPLES_LEAF, FOREST_SEED)
    key = {'X': array_hash(X_train), 'y': array_hash(y_train),
           'is_clf': bool(is_clf), 'X_apply': array_hash(X_apply),
           'forest': [int(FOREST_TREES), int(FOREST_MIN_SAMPLES_LEAF),
                      int(FOREST_SEED)]}
    return cached(dataset, 'forest_leaf_ids', key,
                  lambda: _forest_leaves(X_train, y_train, is_clf, X_apply))


# Every Blueprint field build_fold_hybrid or hybrid_for_fold reads
# (_shared.py:5083-5263, noise_helpers.py:995-1026), except `reps`, whose
# content goes in the key as matrix hashes instead of names.
BLUEPRINT_FIELDS = (
    'allocation_method', 'importance_method', 'budget', 'internal_evaluator',
    'metric', 'lower_is_better', 'task_type', 'objective',
    'augmentation_strategy', 'aug_per_source_budget', 'augmentation_margin',
    'augmentation_fraction', 'candidate_ranking', 'augmentation_threshold',
    'augmentation_budget', 'aug_max_candidates', 'rep_count')


def blueprint_fingerprint(bp, rep_order):
    """The recipe fields a per-fold build reads, canonical.

    rep_scores go in as values in `rep_order` (the order the pool dict is in),
    so no name enters. `reps` is the order-free set the build is cut down to;
    its content enters through the matrix hashes the caller of this function
    computes.
    """
    fp = {f: getattr(bp, f, None) for f in BLUEPRINT_FIELDS}
    scores = getattr(bp, 'rep_scores', None)
    fp['rep_scores'] = (None if not scores else
                        [None if scores.get(r) is None else float(scores[r])
                         for r in rep_order])
    return fp


def cached_hybrid_for_fold(dataset, train_reps, val_reps, y_train, bp,
                           groups_train, model, metric_fn, lower_is_better, *,
                           model_name, metric_name, train_idx,
                           augmentations_train=None, augmentations_val=None,
                           want_feature_info=False):
    """noise_helpers.hybrid_for_fold (noise_helpers.py:939), cached.

    Returns exactly what it returns: (X_tr_h, X_va_h), or with
    want_feature_info the third value too. (None, None[, None]) -- a failed
    build -- is returned and not stored.

    The key: every recipe field the build reads (blueprint_fingerprint), the
    content hash of each representation matrix hybrid_for_fold will use (the
    pool cut to bp.reps, in pool order, as noise_helpers.py:995-1002 cuts it),
    the augmentation names and content hashes, the train index hash, the label
    hash, the groups hash, model_name (None under a geometry recipe, where no
    model is fitted), metric_name and lower_is_better.

    ⛔ REPRESENTATION NAMES enter the key when want_feature_info is True,
    because that return value is {rep_name: ...}: a cached copy built under
    other names would hand back the wrong names. They also enter when the
    recipe has a rep_count, because that cut breaks a tie on score by name
    (_shared.py:5184-5185). Otherwise the key is content only, in the recipe's
    order. The model enters as its name plus model_signature(model).

    Deterministic given the key: create_hybrid's importance fits and
    allocation are seeded; a caller passing a model whose fit is not
    reproducible must not cache.
    """
    from experiments.noise.noise_helpers import hybrid_for_fold
    # The representations hybrid_for_fold builds from, IN THE ORDER IT BUILDS
    # THEM (noise_helpers.py:995-1000): the recipe's order, not the pool's.
    # The order decides the column order of the returned matrices, so two
    # recipes holding the same set in a different order must not share a key.
    # Fixed in review 2026-09-25; this used the pool's order before.
    bp_reps = getattr(bp, 'reps', None)
    if bp_reps:
        used = [r for r in bp_reps if r in train_reps] or list(train_reps)
    else:
        used = list(train_reps)
    # A rep_count cut breaks a tie on score by representation name
    # (_shared.py:5184-5185), so under a rep_count the names can change which
    # representations are built from and they go in the key.
    names_matter = bool(want_feature_info) or (
        getattr(bp, 'rep_count', None) is not None)
    key = {'blueprint': blueprint_fingerprint(bp, used),
           'train_reps': [array_hash(train_reps[r]) for r in used],
           'val_reps': [array_hash(val_reps[r]) if r in val_reps else None
                        for r in used],
           'augmentations': (None if augmentations_train is None else
                             [[a, array_hash(augmentations_train[a]),
                               array_hash((augmentations_val or {}).get(a))]
                              for a in augmentations_train]),
           'train_idx': array_hash(np.asarray(train_idx)),
           'y': array_hash(y_train), 'groups': array_hash(groups_train),
           'model': None if model_name is None else str(model_name),
           'model_signature': model_signature(model),
           'metric': str(metric_name), 'lower_is_better': bool(lower_is_better),
           'want_feature_info': bool(want_feature_info),
           'rep_names': list(used) if names_matter else None}

    def _compute():
        return hybrid_for_fold(train_reps, val_reps, y_train, bp, groups_train,
                               model, metric_fn, lower_is_better,
                               augmentations_train=augmentations_train,
                               augmentations_val=augmentations_val,
                               want_feature_info=want_feature_info)
    return cached(dataset, 'hybrid_for_fold', key, _compute,
                  store_if=lambda v: v is not None and v[0] is not None)
