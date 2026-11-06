#!/usr/bin/env python3
"""Smoke for n44_detect_and_clean.run_unit: the detector x cleanup cross on
one unit of random data, no per-fold search, no dataset.

What is checked: on 160 random molecules (20 features, 16 labels moved by
4 label SD) with rf as the model, a unit with two detector spaces (own and a
random 'hybrid') writes rows for every (detector, cleanup, setting) pair the
pairing rules allow, every row key is distinct, the score-free cleanups run
once, the rounds axis runs, and the detection record holds an AUROC per score
detector. The in-loop network detectors are included (each trains a small DNN
on 160 rows).

Run:  python tests/experiments/noise/_smoke_n44.py
"""
import os
import sys
import tempfile
import time
import warnings

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'KIRBY_LGBM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
_tmp = tempfile.mkdtemp(prefix='n44_smoke_')
os.environ['KIRBY_CACHE_DIR'] = os.path.join(_tmp, 'cache')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
warnings.filterwarnings('ignore')

import numpy as np

from experiments.noise import n44_detect_and_clean as dc
from experiments.noise import n07_remediation as n07
from experiments.noise import n42_similarity_noise_sweep as n42
from experiments.noise.d_vs_dprime import row_key

rng = np.random.RandomState(0)
N, P = 160, 20
X = rng.normal(size=(N, P)).astype(np.float32)
Xh = np.hstack([X[:, :10], rng.normal(size=(N, 10)).astype(np.float32)])
beta = rng.normal(size=P)
y_clean = X @ beta + 0.3 * rng.normal(size=N)
cmask = np.zeros(N, dtype=bool)
cmask[rng.choice(N, 16, replace=False)] = True
y0 = y_clean.copy()
y0[cmask] += 4.0 * y_clean.std() * rng.choice([-1, 1], size=16)
groups = np.arange(N) // 4
tr = np.arange(0, 128)
va = np.arange(128, N)
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
metric = ('rmse', rmse, True)

sink = n07.RowSink(_tmp, dc.EXP_ID, 'smoke', 'gaussian__smoke')
errors = n42._Errors()
counts, unavailable = {}, {}
u = n42._Unit(ds='smoke', sink=sink, metric=metric, task_type='regression',
              is_clf=False, condition='outlier_p05', level=1.0, draw=0, fold=0,
              model='rf', mi='rf:base_1', X_tr=X[tr], X_va=X[va], y0=y0[tr],
              y_clean=y_clean[tr], y_val=y_clean[va], g_tr=groups[tr],
              train_idx=tr, val_idx=va, prov={}, cliff_train=None,
              rand_train=None, measurement_counts=None, row_models=('rf',),
              takes_weight=True, learner_factories={}, errors=errors,
              unavailable=unavailable, counts=counts, fold_similarity=None)
u.cmask = cmask[tr]
u.detection = {}
u.twins = False
spaces = {'own': dc._Space('own', X[tr], X[va], u),
          'hybrid': dc._Space('hybrid', Xh[tr], Xh[va], u)}

from experiments.noise import noise_timing
from experiments.noise import noise_helpers as nh
timing_file = noise_timing.timing_path(_tmp, dc.EXP_ID, 'smoke', 'outlier_p05')
t0 = time.time()
with noise_timing.log_to(timing_file, experiment=dc.EXP_ID, dataset='smoke'):
    dc.run_unit(u, X[tr], X[va], spaces, None, {})
dt = time.time() - t0

failures = []


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg)
    if not cond:
        failures.append(msg)


rows = []
with open(sink.path) as fh:
    import json
    for line in fh:
        rows.append(json.loads(line))
keys = [row_key(r) for r in rows]
methods = sorted({r['method'] for r in rows})
dets = sorted({r['method'].split('/')[1] for r in rows})
cls = sorted({r['method'].split('/')[2] for r in rows})
print(f'{len(rows)} rows in {dt:.0f}s; {len(dets)} detectors, {len(cls)} cleanups')
check(len(keys) == len(set(keys)), 'every row key distinct')
check(len(rows) > 500, f'the cross wrote {len(rows)} rows (over 500)')
check('none' in dets and 'random' in dets, 'score-free (none) and random rows present')
check(any(d.startswith('oracle@precision') for d in dets), 'oracle-precision flag rows present')
check(all(r['method_setting'].startswith('dspace=own|') for r in rows
          if r['method'].split('/')[1].startswith('oracle@')),
      'oracle-precision rows in one detector space only (own)')
check(any(d.startswith('conformal_bh') for d in dets) or counts.get('n41_no_flags'),
      'conformal flag rows present (or none flagged, counted)')
check(all(any(r['method_setting'].startswith(f'dspace={sp}|') for r in rows)
          for sp in ('own', 'hybrid')), 'rows in both detector spaces')
scored = [r for r in rows if r['score_Dprime'] is not None and np.isfinite(r['score_Dprime'])]
check(len(scored) > 0.5 * len(rows), f'{len(scored)} of {len(rows)} rows carry a D\' score')
check(not any(r['method'].endswith('/rounds_delete') for r in rows),
      'rounds_delete not run (cut 2026-10-31)')
check(any('/score_weight' in r['method'] and np.isfinite(r['score_Dprime'] or np.nan)
          for r in rows), 'score_weight scored under a unit-interval detector')
check(any('/spread_invvar_weight' in r['method'] and np.isfinite(r['score_Dprime'] or np.nan)
          for r in rows), 'spread_invvar_weight scored under a spread detector')
for c in ('network_rewrite', 'ema_rewrite', 'rewrite', 'winsorise', 'exp_downweight',
          'move_confidence', 'bayes_shrink', 'shrink_flagged_per_row'):
    check(any(r['method'].endswith('/' + c) and np.isfinite(r['score_Dprime'] or np.nan)
              for r in rows), f'{c} scored')
det_rec = u.detection.get('own', {})
check(len(det_rec) >= 10 and all('detection_auroc' in v for v in det_rec.values()),
      f'detection graded for {len(det_rec)} detectors in the own space')
best = max(det_rec.items(), key=lambda kv: kv[1].get('detection_auroc', 0))
print(f'       best detector on the moved labels: {best[0]} auroc {best[1]["detection_auroc"]:.3f}')
if errors.d:
    print('  errors:')
    for k, v in errors.d.items():
        print(f'    {k}: {v["count"]}x {v["first"][:160]}')
check(len(errors.d) <= 3, f'{len(errors.d)} distinct errors (3 or fewer)')

# Adelaide's decisions of 2026-11-04 (PAPER_3_NOISE_EXPERIMENTS.md section 1)
check(dc.MODEL_INPUTS == ('base_1', 'hybrid'), 'model inputs are base_1 and hybrid')
_reg = {'task_type': 'regression', 'labels': list(range(100)), 'reps': {},
        'smiles': [str(i) for i in range(100)]}
os.environ['KIRBY_NOISE_PROMOTE'] = 'n44'
check(dc.default_conditions(_reg) == ['gaussian', 'outlier_p05', 'grouped_shifted',
                                      'censoring', 'shift_only_p10', 'hetero_seen'],
      'noise kinds after the cuts')
check('sparse_dprime_up_p10' in nh.DECIDED_REGRESSION_CONDITIONS,
      'sparse_dprime_up_p10 still readable by other experiments')
_called = []
_idx = nh.sweep_conditions(lambda *a, **k: _called.append(k), 'x', _reg, {}, _tmp, 'smoke',
                           default_conditions=[], kwargs={'noise_conditions':
                           ['dense_oofmse', 'hetero_hidden', 'sparse_dprime_up_p10']},
                           cut_conditions=nh.PAPER3_CUT_CONDITIONS)
check(not _called and len(_idx['conditions_skipped_and_why']) == 3,
      'a job naming a cut kind runs nothing and records why')
check(not any('__random_twin' in m or '__shuffled_twin' in m for m in methods),
      'first run: no twins')
check(not any('arm=' in r['method_setting'] for r in rows), 'first run: no put-back')
# twins stay off even if the environment asks for them
os.environ['KIRBY_NOISE_CONTROLS'] = '1'
_s2 = n07.RowSink(_tmp, dc.EXP_ID, 'smoke', 'twins_off')
n07.dprime_rows(_s2, dataset_name='smoke', cleaned={'X': X[tr], 'y': y0[tr] + 0.1, 'sample_weight': None},
                X_tr=X[tr], y_D=y0[tr], X_va=X[va], y_va=y_clean[va], metric=metric,
                task_type='regression', y_clean=y_clean[tr], models=('rf',),
                identity=n07.row_identity(dataset_name='smoke', experiment='n44', method='a/b',
                                          method_setting='s', similarity_setting='none',
                                          model_input='rf:base_1', noise_condition='outlier_p05',
                                          level=1.0, draw=0, fold=0), twins=False)
check(_s2.n_written == 1, 'twins=False writes one row with KIRBY_NOISE_CONTROLS=1')
del os.environ['KIRBY_NOISE_CONTROLS']

# no-noise results run once
_big = dict(_reg)
check(nh.runs_level0(_big, 'gaussian') and not nh.runs_level0(_big, 'outlier_p05'),
      'level 0 runs only under gaussian')
check(nh.level0_rows_path('o', 'n44_detect_and_clean', 'x', _big, 'L').endswith('x__gaussian__L.rows.jsonl'),
      'every kind can find the level-0 rows')
# reliability_weight once, under no detector, needs neighbour lists: checked with a fake list below
check(not any('/reliability_weight' in r['method'] for r in rows),
      'reliability_weight writes nothing without neighbour lists')

# the second run: only named results, its own file, plain rows keep the first run's label
_sec = n07.RowSink(_tmp, dc.EXP_ID, 'smoke', 'outlier_p05__second_run')
_mask = np.zeros(len(tr), dtype=bool); _mask[:10] = True
_rand = np.zeros(len(tr), dtype=bool); _rand[10:20] = True
u2 = n42._Unit(ds='smoke', sink=_sec, metric=metric, task_type='regression',
               is_clf=False, condition='outlier_p05', level=1.0, draw=0, fold=0,
               model='rf', mi='rf:base_1', X_tr=X[tr], X_va=X[va], y0=y0[tr],
               y_clean=y_clean[tr], y_val=y_clean[va], g_tr=groups[tr],
               train_idx=tr, val_idx=va, prov={}, cliff_train=_mask,
               rand_train=_rand, measurement_counts=None, row_models=('rf',),
               takes_weight=True, learner_factories={}, errors=n42._Errors(),
               unavailable={}, counts={}, fold_similarity=None, twins=True,
               plain_unlabelled=True,
               second_run={'cliff_definition': 'n25_existing',
                           'results': [{'method': 'random/remove'}]})
u2.cmask = cmask[tr]; u2.detection = {}
dc.run_unit(u2, X[tr], X[va], {'own': dc._Space('own', X[tr], X[va], u2)}, None, {})
_r2 = [json.loads(l) for l in open(_sec.path)]
_m2 = {r['method'] for r in _r2}
check(_m2 == {f'n44/random/remove{t}' for t in ('', '__random_twin', '__shuffled_twin')},
      f'second run wrote only the named result and its twins: {sorted(_m2)}')
_arms = {r['method_setting'].split('arm=')[1] if 'arm=' in r['method_setting'] else 'plain' for r in _r2}
check(_arms == {'plain', 'cliff_exempt', 'random_exempt'}, f'second run arms {sorted(_arms)}')
_p2 = {row_key(r) for r in _r2 if 'arm=' not in r['method_setting'] and 'twin' not in r['method']}
_p1 = {row_key(r) for r in rows if r['method'] == 'n44/random/remove' and r['method_setting'].startswith('dspace=own|')}
check(_p2 and _p2 <= _p1, 'plain rows carry the same key in both runs')
_bad = os.path.join(_tmp, 'bad.json')
for _content, _why in (({'cliff_definition': None, 'results': [{'method': 'a/b'}]}, 'no cliff definition'),
                       ({'cliff_definition': 'n25_existing', 'results': []}, 'no results')):
    json.dump(_content, open(_bad, 'w'))
    os.environ['KIRBY_N44_SECOND_RUN'] = _bad
    try:
        dc.second_run_spec(); check(False, f'refuses with {_why}')
    except ValueError:
        check(True, f'refuses with {_why}')
del os.environ['KIRBY_N44_SECOND_RUN']

# reliability_weight once per neighbour setting, whatever the detectors
_calls = []
_orig = dc.apply_cleanup
_stub = lambda c, *a, **k: (_calls.append(c), {'X': X[tr], 'y': y0[tr], 'sample_weight': None})[1]
dc.apply_cleanup = _stub
_S = type('S', (), {'name': 's0', 'space': 'other'})()
_sim = type('Sim', (), {'kept': [(_S, 5, 'rank')], 'nb': {'s0': {}}, 'with_scheme': lambda self, s, sch: {}})()
_ds, _nc = dc.detector_scores, dc.nc.cached_detector_scores
dc.detector_scores = lambda sp, name: np.linspace(0, 1, len(tr))
dc.nc.cached_detector_scores = lambda *a, **k: np.linspace(0, 1, len(tr))
_u3 = n42._Unit(ds='smoke', sink=n07.RowSink(None, dc.EXP_ID, 'smoke'), metric=metric, task_type='regression',
                is_clf=False, condition='outlier_p05', level=1.0, draw=0, fold=0, model='rf', mi='rf:base_1',
                X_tr=X[tr], X_va=X[va], y0=y0[tr], y_clean=y_clean[tr], y_val=y_clean[va], g_tr=groups[tr],
                train_idx=tr, val_idx=va, prov={}, cliff_train=None, rand_train=None, measurement_counts=None,
                row_models=('rf',), takes_weight=True, learner_factories={}, errors=n42._Errors(),
                unavailable={}, counts={}, fold_similarity=None, twins=False)
_u3.cmask = None; _u3.detection = {}
_ff = dc.flag_detectors; dc.flag_detectors = lambda sp: []
dc.run_unit(_u3, X[tr], X[va], {'own': dc._Space('own', X[tr], X[va], _u3)}, _sim, {})
dc.apply_cleanup, dc.detector_scores, dc.nc.cached_detector_scores, dc.flag_detectors = _orig, _ds, _nc, _ff
_nr = _calls.count('reliability_weight')
check(_nr == len(dc.cleanup_settings('reliability_weight')),
      f'reliability_weight ran {_nr} times for one neighbour setting and {len(dc.cleanup_settings("reliability_weight"))} strengths')

# time recording
lines = noise_timing.read_log(timing_file)
whats = {l['what'] for l in lines}
print(f'       {len(lines)} time lines; {sorted(whats)[:8]}')
for w in ('detector', 'cleanup', 'result', 'cache:fit_predict'):
    check(w in whats, f'time recorded for {w}')
check(len({l['detector'] for l in lines if l['what'] == 'detector'}) >= 10,
      'time recorded per detector')
check(len({l['cleanup'] for l in lines if l['what'] == 'cleanup'}) >= 10,
      'time recorded per cleanup')
_res = [l for l in lines if l['what'] == 'result']
check(len(_res) >= len(rows), f'{len(_res)} result time lines for {len(rows)} rows')
check({l['status'] for l in _res} >= {'computed'}, 'fresh fits are marked computed')
check(any(l['what'] == 'cleanup' and l['status'] == 'refused' for l in lines)
      or all(r['score_Dprime'] is not None for r in rows),
      'a cleanup that returned nothing is marked refused')
noise_timing.summarise(lines)

print()
if failures:
    print(f'{len(failures)} FAILED'); sys.exit(1)
print('all passed')
