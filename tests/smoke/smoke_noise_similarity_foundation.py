#!/usr/bin/env python3
"""The shared noise-suite code of plan sections 1-3, on synthetic rows.

Checks, each printed PASS / FAIL / SKIP with the numbers behind it:

  (a) n07.cleaned_training_set with similarity_neighbours=None, and with the
      MODEL_INPUT setting's lists, is bit-identical to the untouched call for
      label_smooth and knn_smooth (regression and classification), and
      prox_smooth with the forest_proximity lists is bit-identical to its own
      forest. Run on continuous features and on 0/1 features, where
      neighbour ties are everywhere.
  (b) each of the four neighbour estimators given precomputed_neighbors equal
      to what _knn_indices returns gives identical scores.
  (c) every cache wrapper: the second call is a hit and returns bit-identical
      arrays; KIRBY_NOISE_CACHE=0 gives the same numbers.
  (d) enumerate_settings gives 50 settings when every space is available.
  (e) d_vs_dprime's gain is positive when D' helped, for an rmse dataset and
      an auc dataset, with the direction read from get_domain_metric.
  (f) resume: done_keys skips rows already written, and a row cut off by a
      kill is skipped rather than raised on.
  (g) the controls delete, downweight or move the same number of molecules as
      the method.

Synthetic: 200 rows, 3 fake representations, SMILES cycled from a short list
of real ones (so there are exact duplicates and Tanimoto ties). The cache is
written under a temporary KIRBY_CACHE_DIR, never the real one.

    KIRBY_LGBM_THREADS=1 OMP_NUM_THREADS=1 \\
        python tests/smoke/smoke_noise_similarity_foundation.py
"""
import os
import sys
import tempfile
import time

os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('KIRBY_RF_JOBS', '1')
_TMP = tempfile.mkdtemp(prefix='kirby_noise_cache_smoke_')
os.environ['KIRBY_CACHE_DIR'] = _TMP
os.environ.pop('KIRBY_NOISE_CACHE', None)
os.environ.pop('KIRBY_IMPORTANCE_EXPONENT', None)

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.noise import similarity_space as ss          # noqa: E402
from experiments.noise import noise_cache as nc               # noqa: E402
from experiments.noise import d_vs_dprime as dd               # noqa: E402
from experiments.noise import controls as ctl                 # noqa: E402

T0 = time.time()
RESULTS = []


def report(tag, ok, detail=''):
    status = {True: 'PASS', False: 'FAIL', None: 'SKIP'}[ok]
    RESULTS.append((tag, status))
    print(f"[{status}] {tag}: {detail}", flush=True)


def same(a, b):
    """Bit-identical, recursively through dicts, tuples and lists."""
    if isinstance(a, dict):
        return (isinstance(b, dict) and a.keys() == b.keys()
                and all(same(a[k], b[k]) for k in a))
    if isinstance(a, (tuple, list)):
        return (isinstance(b, (tuple, list)) and len(a) == len(b)
                and all(same(x, y) for x, y in zip(a, b)))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a_, b_ = np.asarray(a), np.asarray(b)
        return (a_.dtype == b_.dtype and a_.shape == b_.shape
                and a_.tobytes() == b_.tobytes())
    if isinstance(a, float) and isinstance(b, float):
        return (a == b) or (np.isnan(a) and np.isnan(b))
    return a == b


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
N, N_TR = 200, 160
rng = np.random.RandomState(0)
SMILES_POOL = ['CCO', 'CCN', 'c1ccccc1', 'c1ccccc1O', 'c1ccccc1N', 'CC(=O)O',
               'CC(=O)N', 'CCCC', 'CCCCO', 'c1ccncc1', 'C1CCCCC1', 'CC(C)O',
               'OCC(O)CO', 'c1ccc2ccccc2c1', 'CN(C)C', 'CCOC(=O)C', 'Clc1ccccc1',
               'Brc1ccccc1', 'NC(=O)c1ccccc1', 'CC#N', 'not_a_smiles']
smiles = np.array([SMILES_POOL[i % len(SMILES_POOL)] for i in range(N)],
                  dtype=object)
latent = rng.randn(N, 5)
reps = {
    'repA': latent @ rng.randn(5, 30) + 0.3 * rng.randn(N, 30),   # continuous
    'repB': (rng.rand(N, 64) < 0.15).astype(np.float64),          # 0/1, ties
    'repC': latent[:, :3] @ rng.randn(3, 10) + rng.randn(N, 10),
}
y_clean = latent @ rng.randn(5) + 0.1 * rng.randn(N)
y_noisy = y_clean + np.where(rng.rand(N) < 0.15, 3.0 * rng.randn(N), 0.0)
y_clf = (y_noisy > np.median(y_noisy)).astype(int)
groups = np.arange(N) // 5          # 40 groups; the split below is group-disjoint
tr, va = np.arange(N_TR), np.arange(N_TR, N)


def zs(X):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    return sc.fit_transform(X[tr]), sc.transform(X[va])


print(f"imports done in {time.time() - T0:.1f}s", flush=True)
from experiments.noise import n07_remediation as n07          # noqa: E402
from kirby.utils.noise_estimation import (                    # noqa: E402
    _knn_indices, forest_leaf_ids, NeighborResidualEstimator,
    LocalVarianceRatioEstimator, DistanceWeightedResidualEstimator,
    ActivityCliffAwareEstimator)
print(f"n07 imported at {time.time() - T0:.1f}s", flush=True)

# ---------------------------------------------------------------------------
# (a) bit-identity through cleaned_training_set
# ---------------------------------------------------------------------------
scores = np.abs(rng.randn(N_TR))
for rep in ('repA', 'repB'):
    X_tr_s, X_va_s = zs(reps[rep])
    mi = ss.neighbour_lists(ss.MODEL_INPUT, spaces_train={'model_input': X_tr_s},
                            spaces_val={'model_input': X_va_s},
                            y_train=y_noisy[tr])
    for label_kind, y_tr, is_clf in (('regression', y_noisy[tr], False),
                                     ('classification', y_clf[tr], True)):
        for strat in ('label_smooth', 'knn_smooth'):
            base = n07.cleaned_training_set(strat, X_tr_s, y_tr, scores, 0.1,
                                            is_clf=is_clf)
            none = n07.cleaned_training_set(strat, X_tr_s, y_tr, scores, 0.1,
                                            is_clf=is_clf,
                                            similarity_neighbours=None)
            given = n07.cleaned_training_set(strat, X_tr_s, y_tr, scores, 0.1,
                                             is_clf=is_clf,
                                             similarity_neighbours=mi)
            ok = same(base, none) and same(base, given)
            n_moved = int(np.sum(base['y'] != y_tr))
            report(f"a {rep} {label_kind} {strat}", ok,
                   f"None and MODEL_INPUT bit-identical to untouched; "
                   f"{n_moved} of {N_TR} labels moved")
    # prox_smooth through the forest_proximity setting's lists.
    fp = ss.neighbour_lists(ss.SimilaritySetting('forest_proximity', 'none',
                                                 'forest'),
                            spaces_train={'model_input': X_tr_s},
                            spaces_val={'model_input': X_va_s},
                            y_train=y_noisy[tr])
    base = n07.cleaned_training_set('prox_smooth', X_tr_s, y_noisy[tr], scores,
                                    0.1)
    given = n07.cleaned_training_set('prox_smooth', X_tr_s, y_noisy[tr], scores,
                                     0.1, similarity_neighbours=fp)
    L_ref = forest_leaf_ids(X_tr_s, y_noisy[tr], False)
    L_new, _ = ss._forest_leaves(X_tr_s, y_noisy[tr], False)
    report(f"a {rep} prox_smooth forest_proximity", same(base, given)
           and same(L_ref, L_new),
           "prox_smooth with the lists bit-identical to its own forest; "
           "training leaf ids identical to forest_leaf_ids")
    # Informational: why MODEL_INPUT carries one list per k.
    cut = ss.neighbour_lists(ss.SimilaritySetting('model_input', 'none',
                                                  'process'),
                             spaces_train={'model_input': X_tr_s},
                             spaces_val=None, y_train=y_noisy[tr])
    q9 = cut['exact_by_k'][9][0]
    q20_cut = cut['exact_by_k'][20][0][:, :9]
    print(f"      info {rep}: the 9 neighbours of a query for 10 differ from "
          f"the first 9 of a query for 21 on "
          f"{int((q9 != q20_cut).any(axis=1).sum())} of {N_TR} training rows",
          flush=True)

# Stable-sort prefix property for a non-MODEL_INPUT setting.
X_tr_s, X_va_s = zs(reps['repB'])
s_b = ss.neighbour_lists(ss.SimilaritySetting('base_1', 'none', 'manhattan'),
                         spaces_train={'base_1': X_tr_s},
                         spaces_val={'base_1': X_va_s}, y_train=y_noisy[tr],
                         exclude_groups=groups[tr])
s_b5 = ss.neighbour_lists(ss.SimilaritySetting('base_1', 'none', 'manhattan',
                                               k_max=5),
                          spaces_train={'base_1': X_tr_s},
                          spaces_val={'base_1': X_va_s}, y_train=y_noisy[tr],
                          exclude_groups=groups[tr])
g_tr = groups[tr]
no_self = not np.any(s_b['idx_train'] == np.arange(N_TR)[:, None])
no_group = not np.any(g_tr[s_b['idx_train']] == g_tr[:, None])
report("a stable top-k prefix", same(s_b['idx_train'][:, :5], s_b5['idx_train'])
       and no_self and no_group and np.all(np.diff(s_b['dist_train'], axis=1) >= 0),
       f"k_max=20 list cut to 5 equals a k_max=5 list on 0/1 features; no row "
       f"is its own neighbour or shares its group; k_eff={s_b['k_eff']}")

# ---------------------------------------------------------------------------
# (b) estimators with precomputed neighbours
# ---------------------------------------------------------------------------
import inspect  # noqa: E402
X_tr_s, _ = zs(reps['repA'])
for cls in (NeighborResidualEstimator, LocalVarianceRatioEstimator,
            DistanceWeightedResidualEstimator, ActivityCliffAwareEstimator):
    if 'precomputed_neighbors' not in inspect.signature(cls.__init__).parameters:
        report(f"b {cls.__name__}", None,
               "precomputed_neighbors is not an argument: "
               "src/kirby/utils/noise_estimation.py had uncommitted edits "
               "from another session, so it was not edited")
        continue
    for eg in (None, g_tr):
        d, i = _knn_indices(X_tr_s, 10, None, exclude_groups=eg,
                            return_distance=True)
        a = cls(k=10, exclude_groups=eg).fit(X_tr_s, y_noisy[tr]).noise_scores_
        b = cls(k=10, exclude_groups=eg, precomputed_neighbors=(i, d)).fit(
            X_tr_s, y_noisy[tr]).noise_scores_
        report(f"b {cls.__name__} exclude_groups={'yes' if eg is not None else 'no'}",
               same(a, b), "identical scores")

# ---------------------------------------------------------------------------
# (c) cache wrappers
# ---------------------------------------------------------------------------
from kirby.find_best_hybrid import Blueprint  # noqa: E402
from experiments._shared import get_model_factories, get_domain_metric  # noqa: E402

DS = 'smoke_toy'
data_reg = {'dataset_name': None, 'task_type': 'regression', 'labels': y_noisy}
metric_reg = get_domain_metric(data_reg)
rf_factory = get_model_factories(['rf'], 'regression', N_TR)['rf']
XA_tr, XA_va = zs(reps['repA'])
XC_tr, XC_va = zs(reps['repC'])
bp = Blueprint(reps=['repA', 'repC'], allocation_method='performance_weighted',
               importance_method='lightgbm_gain', budget=20,
               internal_evaluator='lightgbm', metric=metric_reg[0],
               lower_is_better=metric_reg[2], augmentation_strategy='none')
spaces_tr = {'model_input': XA_tr, 'base_1': XA_tr, 'hybrid': XC_tr}
spaces_va = {'model_input': XA_va, 'base_1': XA_va, 'hybrid': XC_va}

CALLS = {
    'scaled': lambda: nc.cached_scaled(DS, reps['repA'], tr, va),
    'fit_predict': lambda: nc.cached_fit_predict(
        DS, 'rf', rf_factory, XA_tr, y_noisy[tr], XA_va),
    'oof_predict': lambda: nc.cached_oof_predict(DS, XA_tr, y_noisy[tr],
                                                 groups[tr], n_splits=3),
    'detector_scores': lambda: nc.cached_detector_scores(
        DS, XA_tr, y_noisy[tr], detector='neighbor_residual', groups=groups[tr]),
    'neighbours': lambda: nc.cached_neighbours(
        DS, ss.SimilaritySetting('base_1', 'lightgbm_gain', 'euclidean'),
        spaces_train=spaces_tr, spaces_val=spaces_va, y_train=y_noisy[tr]),
    'forest_leaf_ids': lambda: nc.cached_forest_leaf_ids(
        DS, XA_tr, y_noisy[tr], False, X_apply=XA_va),
    'hybrid_for_fold': lambda: nc.cached_hybrid_for_fold(
        DS, {'repA': reps['repA'][tr], 'repB': reps['repB'][tr],
             'repC': reps['repC'][tr]},
        {'repA': reps['repA'][va], 'repB': reps['repB'][va],
         'repC': reps['repC'][va]},
        y_noisy[tr], bp, groups[tr], rf_factory(), metric_reg[1], metric_reg[2],
        model_name='rf', metric_name=metric_reg[0], train_idx=tr),
}
try:
    import rdkit  # noqa: F401
    CALLS['neighbours_tanimoto'] = lambda: nc.cached_neighbours(
        DS, ss.SimilaritySetting('tanimoto_ecfp4', 'none', 'tanimoto'),
        spaces_train=spaces_tr, spaces_val=spaces_va, y_train=y_noisy[tr],
        smiles_train=smiles[tr], smiles_val=smiles[va])
except ImportError:
    report("c neighbours_tanimoto", None, "RDKit not importable")

for name, call in CALLS.items():
    kind = 'neighbours' if name.startswith('neighbours') else name
    t = time.time()
    before = dict(nc.report().get(kind, {'hits': 0, 'misses': 0, 'bypass': 0}))
    first = call()
    mid = dict(nc.report()[kind])
    second = call()
    after = dict(nc.report()[kind])
    os.environ['KIRBY_NOISE_CACHE'] = '0'
    off = call()
    os.environ.pop('KIRBY_NOISE_CACHE')
    miss_then_hit = (mid['misses'] == before['misses'] + 1
                     and after['hits'] == mid['hits'] + 1)
    if name == 'fit_predict':
        # The downstream rf factory fits with n_jobs=-1, and its prediction
        # moves by ~1e-16 between two identical fits in one process (the
        # trees' outputs are summed in thread order). That is the forest, not
        # the cache: the stored copy is bit-identical to the first fit. A
        # recompute is held to the suite's same-versus-same tolerance, 1e-12
        # (NOISE_AUDIT_2026_09_10.md section 17, quoted in
        # smoke_n12_two_spaces.py).
        gap = float(np.max(np.abs(np.asarray(first) - np.asarray(off))))
        off_ok, off_txt = gap <= 1e-12, f"recompute within {gap:.1e} of it"
    else:
        off_ok, off_txt = same(first, off), "KIRBY_NOISE_CACHE=0 identical"
    report(f"c {name}", miss_then_hit and same(first, second) and off_ok,
           f"miss then hit, stored copy bit-identical, {off_txt} "
           f"({time.time() - t:.1f}s)")
tan = CALLS.get('neighbours_tanimoto')
if tan is not None:
    tl = tan()
    print(f"      info tanimoto: k_eff={tl['k_eff']}, nearest distance range "
          f"{tl['dist_train'][:, 0].min():.3f}-{tl['dist_train'][:, 0].max():.3f}",
          flush=True)

# An unavailable space is reported, not faked, and not stored.
un = nc.cached_neighbours(DS, ss.SimilaritySetting('hybrid_similarity', 'none',
                                                   'manhattan'),
                          spaces_train=spaces_tr, spaces_val=spaces_va,
                          y_train=y_noisy[tr])
report("c unavailable space", (not un['available']) and un['idx_train'] is None,
       f"reason: {un['reason']}")

# ---------------------------------------------------------------------------
# (d) the count
# ---------------------------------------------------------------------------
settings = ss.enumerate_settings()
names = [s.name for s in settings]
report("d enumerate_settings", len(settings) == 50 and len(set(names)) == 50,
       f"{len(settings)} settings, {len(set(names))} distinct names; without "
       f"hybrid_similarity: "
       f"{len(ss.enumerate_settings([s for s in ss.ALL_SPACES if s != 'hybrid_similarity']))}")

# ---------------------------------------------------------------------------
# (e) gain sign, direction from get_domain_metric
# ---------------------------------------------------------------------------
data_clf = {'dataset_name': None, 'task_type': 'classification',
            'labels': y_clf}
metric_clf = get_domain_metric(data_clf)
ident = dict(dataset='toy', experiment='smoke', method='m', method_setting='s',
             similarity_setting='model_input|none|process', model_input='repA',
             noise_condition='gaussian', level=0.1, draw=0, fold=0)
r_rmse = dd.make_row(**ident, model='rf', metric=metric_reg[0],
                     lower_is_better=metric_reg[2], score_D=1.0,
                     score_Dprime=0.8, score_clean=0.5)
r_auc = dd.make_row(**ident, model='rf', metric=metric_clf[0],
                    lower_is_better=metric_clf[2], score_D=0.70,
                    score_Dprime=0.80, score_clean=0.9)
report("e gain sign", metric_reg[0] == 'rmse' and metric_reg[2] is True
       and metric_clf[0] == 'auc' and metric_clf[2] is False
       and r_rmse['gain'] > 0 and r_auc['gain'] > 0,
       f"rmse 1.0 -> 0.8 gives gain {r_rmse['gain']:+.3f}; auc 0.70 -> 0.80 "
       f"gives gain {r_auc['gain']:+.3f}")
# End to end through the fits, with a weighted D'.
cleaned = n07.cleaned_training_set('downweight', XA_tr, y_noisy[tr], scores, 0.1)
sc = dd.score_d_dprime(DS, cleaned, XA_tr, y_noisy[tr], XA_va, y_clean[va],
                       metric=metric_reg, task_type='regression',
                       y_tr_clean=y_clean[tr])
rows = dd.rows_for(sc, metric=metric_reg, **ident)
report("e score_d_dprime", len(rows) == 2 and all(
    np.isfinite([r['score_D'], r['score_Dprime'], r['score_clean']]).all()
    for r in rows),
    '; '.join(f"{r['model']} rmse D {r['score_D']:.4f} D' {r['score_Dprime']:.4f}"
              f" clean {r['score_clean']:.4f}" for r in rows))

# ---------------------------------------------------------------------------
# (f) resume
# ---------------------------------------------------------------------------
path = os.path.join(_TMP, 'rows.jsonl')
all_rows = [dd.make_row(**{**ident, 'fold': f}, model='rf', metric='rmse',
                        lower_is_better=True, score_D=1.0, score_Dprime=0.9,
                        score_clean=0.8) for f in range(4)]
dd.append_rows(path, all_rows[:2])
with open(path, 'a') as fh:
    fh.write('{"dataset": "toy", "experi')          # a kill mid-line
done = dd.done_keys(path)
todo = [r for r in all_rows if dd.row_key(r) not in done]
report("f resume", len(done) == 2 and [r['fold'] for r in todo] == [2, 3],
       f"{len(done)} rows on disk, {len(todo)} left to run (folds "
       f"{[r['fold'] for r in todo]}); a half-written line was skipped")

# ---------------------------------------------------------------------------
# (g) controls
# ---------------------------------------------------------------------------
rem = n07.cleaned_training_set('remove', XA_tr, y_noisy[tr], scores, 0.1)
n_del = int((~rem['keep']).sum())
tw, n_t = ctl.random_twin(rem, XA_tr, y_noisy[tr], (0, 0))
sh = ctl.shuffled_twin(rem, XA_tr, y_noisy[tr], (0, 0))
rr, n_r = ctl.random_twin(
    rem, XA_tr, y_noisy[tr], (1, 0),
    rerun=lambda s, rate: n07.cleaned_training_set('remove', XA_tr, y_noisy[tr],
                                                   s, rate))
ks = n07.cleaned_training_set('knn_smooth', XA_tr, y_noisy[tr], scores, 0.1)
n_mv = int(np.sum(ks['label_column'] != y_noisy[tr]))
tk, _ = ctl.random_twin(ks, XA_tr, y_noisy[tr], (0, 0))
n_mv_twin = int(np.sum(tk['label_column'] != y_noisy[tr]))
same_set = bool(np.all(tw['keep'] == rem['keep']))
report("g controls", n_t == n_del and int((~tw['keep']).sum()) == n_del
       and int((~sh['keep']).sum()) == n_del
       and int((~rr['keep']).sum()) == n_del and n_mv_twin == n_mv
       and not same_set,
       f"remove deleted {n_del} of {N_TR}; random twin {int((~tw['keep']).sum())}, "
       f"shuffled twin {int((~sh['keep']).sum())}, n24-style rerun "
       f"{int((~rr['keep']).sum())}; knn_smooth moved {n_mv}, its random twin "
       f"{n_mv_twin}; controls_enabled()={ctl.controls_enabled()}")

# ---------------------------------------------------------------------------
# (h) review checks added 2026-09-25: keys that must differ, and the guard
# ---------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
sig_r = nc.model_signature(RandomForestRegressor(random_state=0))
sig_c = nc.model_signature(RandomForestClassifier(random_state=0))
sig_r2 = nc.model_signature(RandomForestRegressor(random_state=0))
report("h model_signature", sig_r != sig_c and sig_r == sig_r2,
       "an rf regressor and an rf classifier give different keys; two equal "
       "regressors give one")

# Two recipes holding the same representations in opposite order: the build
# follows the recipe's order, so the cached copy of one must not serve the
# other.
from dataclasses import replace as _dc_replace  # noqa: E402
bp_rev = _dc_replace(bp, reps=['repC', 'repA'])
_trr = {'repA': reps['repA'][tr], 'repB': reps['repB'][tr],
        'repC': reps['repC'][tr]}
_var = {'repA': reps['repA'][va], 'repB': reps['repB'][va],
        'repC': reps['repC'][va]}
h_fwd = CALLS['hybrid_for_fold']()
h_rev = nc.cached_hybrid_for_fold(
    DS, _trr, _var, y_noisy[tr], bp_rev, groups[tr], rf_factory(),
    metric_reg[1], metric_reg[2], model_name='rf', metric_name=metric_reg[0],
    train_idx=tr)
os.environ['KIRBY_NOISE_CACHE'] = '0'
h_rev_off = nc.cached_hybrid_for_fold(
    DS, _trr, _var, y_noisy[tr], bp_rev, groups[tr], rf_factory(),
    metric_reg[1], metric_reg[2], model_name='rf', metric_name=metric_reg[0],
    train_idx=tr)
os.environ.pop('KIRBY_NOISE_CACHE')
report("h hybrid recipe order", same(h_rev, h_rev_off),
       f"recipe ['repC','repA'] after ['repA','repC'] was cached: served copy "
       f"equals a fresh build; the two orders' training matrices are "
       f"{'equal' if same(h_fwd[0], h_rev[0]) else 'different'}")

# The guard: a setting the cleanup would ignore is refused.
s_base = ss.neighbour_lists(ss.SimilaritySetting('base_1', 'none', 'manhattan'),
                            spaces_train={'base_1': XA_tr}, spaces_val=None,
                            y_train=y_noisy[tr])
refused = []
for strat in ('remove', 'downweight', 'prox_smooth'):
    try:
        n07.cleaned_training_set(strat, XA_tr, y_noisy[tr], scores, 0.1,
                                 similarity_neighbours=s_base)
    except ValueError:
        refused.append(strat)
mi_a = ss.neighbour_lists(ss.MODEL_INPUT, spaces_train={'model_input': XA_tr},
                          spaces_val=None, y_train=y_noisy[tr])
ok_mi = same(n07.cleaned_training_set('remove', XA_tr, y_noisy[tr], scores, 0.1),
             n07.cleaned_training_set('remove', XA_tr, y_noisy[tr], scores, 0.1,
                                      similarity_neighbours=mi_a))
report("h guard", refused == ['remove', 'downweight', 'prox_smooth'] and ok_mi,
       f"base_1 lists refused by {refused}; MODEL_INPUT accepted by remove and "
       f"bit-identical")

# ---------------------------------------------------------------------------
print()
print(f"cache report: {nc.report()}")
n_fail = sum(1 for _, s in RESULTS if s == 'FAIL')
n_skip = sum(1 for _, s in RESULTS if s == 'SKIP')
print(f"{len(RESULTS)} checks: {len(RESULTS) - n_fail - n_skip} pass, "
      f"{n_fail} fail, {n_skip} skip, in {time.time() - T0:.0f}s")
sys.exit(1 if n_fail else 0)
