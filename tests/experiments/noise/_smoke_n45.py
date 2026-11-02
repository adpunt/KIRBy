#!/usr/bin/env python3
"""Smoke for n45_detector_grading on random data, no per-fold search, no
dataset.

What is checked:
  1. grade_unit on 128 random training molecules (20 features, 16 labels
     moved up by 3 label SDs, kind shift_only_p10, rf): every detector n45
     grades (n44's lists) gets a grade or a recorded reason; every grade has
     its chance level beside it; the neighbour detectors are scored under two
     neighbour settings; the ceiling record carries n31's predicted AUROC;
     the set-precision rows are written (3 flag shares x 5 precisions x 2
     seeds x 4 operations) with every metric; the realised precisions are in
     the record.
  2. A level-0 unit: every detector records "level 0" and no row is written.
  3. summarise_condition gives a break-even entry and a hybrid-minus-single
     entry from two units.
  4. The real-noise code paths: all_scores with the repeat exclusion;
     run_real_noise on a dataset with no repeat structure and no curator
     flag records both reasons; grade_database_disagreement and
     grade_curator_flag on a made-up ground truth and flag.

KIRBY_N45_SMOKE_DETECTORS=all runs every detector in check 1 (the default);
a comma list narrows it, for the smoke only.

Run:  python tests/experiments/noise/_smoke_n45.py
"""
import json
import os
import sys
import tempfile
import time
import warnings

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'KIRBY_LGBM_THREADS'):
    os.environ[_v] = '1'
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
_root = ('/Volumes/seagate' if os.path.isdir('/Volumes/seagate')
         else '/private/tmp/claude-501/-Users-apunt-repos-KIRBy/'
              '62112545-c7bc-4a8e-9acd-c99b58998fb5/scratchpad')
_tmp = tempfile.mkdtemp(prefix='n45_smoke_', dir=_root)
os.environ['KIRBY_CACHE_DIR'] = os.path.join(_tmp, 'cache')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
warnings.filterwarnings('ignore')

import numpy as np

from experiments.noise import n45_detector_grading as dg
from experiments.noise import n07_remediation as n07
from experiments.noise import n42_similarity_noise_sweep as n42
from experiments.noise import similarity_space as ss
from experiments.noise.d_vs_dprime import row_key, ALL_METRICS
from _ground_truth_providers import GroundTruth

print(f'temp dir {_tmp}')
failures = []


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg)
    if not cond:
        failures.append(msg)


rng = np.random.RandomState(0)
N, P = 160, 20
X = rng.normal(size=(N, P)).astype(np.float32)
Xh = np.hstack([X[:, :10], rng.normal(size=(N, 10)).astype(np.float32)])
beta = rng.normal(size=P)
y_clean = X @ beta + 0.3 * rng.normal(size=N)
tr = np.arange(0, 128)
va = np.arange(128, N)
cmask = np.zeros(N, dtype=bool)
cmask[rng.choice(tr, 16, replace=False)] = True
y0 = y_clean.copy()
y0[cmask] += 3.0 * y_clean[tr].std()
groups = np.arange(N) // 4
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
metric = ('rmse', rmse, True)

env = os.environ.get('KIRBY_N45_SMOKE_DETECTORS', 'all')
subset = None if env == 'all' else tuple(env.split(','))
print(f'detectors in check 1: {"all" if subset is None else subset}')

COND = 'shift_only_p10'
sink = n07.RowSink(_tmp, dg.EXP_ID, 'smoke', f'{COND}__smoke')
units = n07.FoldCheckpoint(_tmp, dg.EXP_ID, 'smoke', f'{COND}__smoke',
                           tag='units', fingerprint={'smoke': 1})
unavailable = {}


def make_unit(mi, level, Xtr):
    return dg._GradeUnit(
        ds='smoke', is_clf=False, condition=COND, level=float(level), draw=0,
        fold=0, model='rf', mi=f'rf:{mi}', y0=(y0[tr] if level > 0
                                               else y_clean[tr]),
        g_tr=groups[tr], cmask=(cmask[tr] if level > 0 else None),
        takes_weight=True, errors=n42._Errors(), unavailable=unavailable)


settings = [ss.SimilaritySetting('base_1', 'none', 'manhattan'),
            ss.SimilaritySetting(ss.FOREST_SPACE, 'none', 'forest')]


def lists_for(u, Xtr, Xva):
    sim = dg._AllSettings(u, settings, n07.sweep_ks(),
                          {'base_1': Xtr, 'model_input': Xtr},
                          {'base_1': Xva, 'model_input': Xva}, {}, {},
                          None, None)
    return dict(sim.nb)


# ---- 1. one unit, injected noise ----
print('\n1. grade_unit, base_1, level 1')
u = make_unit('base_1', 1.0, X[tr])
t0 = time.time()
lists = lists_for(u, X[tr], X[va])
rec = dg.grade_unit(u, X[tr], X[va], lists, sink=sink, y_clean=y_clean[tr],
                    y_val=y_clean[va], metric=metric, task_type='regression',
                    noise_groups_tr=groups[tr], detectors=subset)
dt1 = time.time() - t0
units.put([0, 1.0, 0, 'rf', 'base_1'], rec)
print(f'  {len(rec["detection"])} score grades, {len(rec["flag_detection"])} '
      f'flag grades, {len(rec["not_graded"])} not graded, {dt1:.0f}s')
check(all(v is not None for v in lists.values()),
      f'neighbour lists built for both settings ({list(lists)})')
expected = [d for d in dg.SCORE_DETECTORS + dg.NEIGHBOUR_DETECTORS
            + ('conformal_bh',) if subset is None or d in subset]
missing = []
for d in expected:
    has_grade = any(dg.detector_base(k) == d for k in
                    list(rec['detection']) + list(rec['flag_detection']))
    has_reason = any(dg.detector_base(k) == d for k in rec['not_graded'])
    if not (has_grade or has_reason):
        missing.append(d)
check(not missing, f'every one of {len(expected)} detectors has a grade or a '
                   f'reason (missing: {missing})')
check(all(all(c in g for c in ('chance_auroc', 'chance_average_precision',
                               'chance_precision_at_k'))
          for g in rec['detection'].values()),
      'chance level beside every score grade')
check(all('chance_flag_precision' in g for g in rec['flag_detection'].values()),
      'chance level beside every flag grade')
check(all(set(g['precision_at_flag_share']) == {f'{s:g}' for s in dg.FLAG_SHARES}
          for g in rec['detection'].values()),
      'precision at each flag share in every score grade')
nb_keys = [k for k in rec['detection'] if '|k=' in k]
check(len(nb_keys) > 0 and any(ss.FOREST_SPACE in k for k in nb_keys),
      f'{len(nb_keys)} neighbour-detector grades, forest setting among them')
check('predicted_auroc' in rec['ceiling'] and 'detection_abs_residual'
      in rec['ceiling'], f"ceiling: predicted AUROC "
      f"{rec['ceiling'].get('predicted_auroc')}, measured "
      f"{(rec['ceiling'].get('detection_abs_residual') or {}).get('detection_auroc')}")
n_or = len(dg.FLAG_SHARES) * len(dg.PRECISIONS) * len(dg.FLAG_SEEDS)
check(len(rec['oracle']['realised_precision']) == n_or,
      f'{len(rec["oracle"]["realised_precision"])} realised precisions '
      f'(expected {n_or})')
rows = [json.loads(l) for l in open(sink.path)]
orows = [r for r in rows if r['method'].startswith('oracle@')]
check(len(orows) == n_or * len(dg.OPERATIONS),
      f'{len(orows)} set-precision rows (expected {n_or * len(dg.OPERATIONS)})')
check(all(r['experiment'] == dg.EXP_ID for r in rows), 'rows say experiment n45')
check(len({row_key(r) for r in rows}) == len(rows), 'every row key distinct')
mets = set(ALL_METRICS['regression'])
check(all(set((r.get('all_Dprime') or {})) >= mets for r in orows
          if r.get('score_Dprime') is not None),
      'every scored row carries all regression metrics')
for d, why in sorted(rec['not_graded'].items()):
    print(f'       not graded: {d}: {str(why)[:120]}')

# ---- 1b. the hybrid input, so the summary has a hybrid-minus-single entry --
print('\n1b. grade_unit, hybrid, level 1 (residual and random detectors only)')
uh = make_unit('hybrid', 1.0, Xh[tr])
rech = dg.grade_unit(uh, Xh[tr], Xh[va], None, sink=sink, y_clean=y_clean[tr],
                     y_val=y_clean[va], metric=metric, task_type='regression',
                     noise_groups_tr=groups[tr],
                     detectors=('oof_residual', 'random', 'cross_val'))
units.put([0, 1.0, 0, 'rf', 'hybrid'], rech)
check(set(rech['detection']) >= {'oof_residual', 'random'},
      f'hybrid unit graded {sorted(rech["detection"])}')

# ---- 2. level 0 ----
print('\n2. grade_unit, level 0')
n_before = sink.n_written
u0 = make_unit('base_1', 0.0, X[tr])
rec0 = dg.grade_unit(u0, X[tr], X[va], None, sink=sink, y_clean=y_clean[tr],
                     y_val=y_clean[va], metric=metric, task_type='regression',
                     noise_groups_tr=groups[tr], detectors=subset)
check(not rec0['detection'] and all(v == dg.REASON_LEVEL_0
                                    for v in rec0['not_graded'].values()),
      f'level 0: {len(rec0["not_graded"])} detectors carry the level-0 reason')
check(sink.n_written == n_before and 'not_run_because' in rec0['oracle'],
      'level 0: no set-precision row, reason recorded')

# ---- 3. summary ----
print('\n3. summarise_condition')
summ = dg.summarise_condition(units, sink, COND, True)
be = summ['break_even'].get('1', {}).get('rf:base_1', {})
check(len(be) == len(dg.FLAG_SHARES) and all(
    set(v) == set(dg.OPERATIONS) for v in be.values()),
      f'break-even per flag share and operation ({len(be)} shares)')
for sh, by_op in be.items():
    print(f'       {sh}: ' + ', '.join(
        f"{op} {e['break_even_precision'] if e['break_even_precision'] is None else round(e['break_even_precision'], 3)}"
        for op, e in by_op.items()))
hm = summ['hybrid_minus_single'].get('1', {}).get('rf', {})
check('hybrid_minus_base_1' in hm and 'oof_residual' in hm['hybrid_minus_base_1'],
      'hybrid-minus-single entry for oof_residual')
check('oof_residual' in summ['detector_median_auroc'],
      'detector_median_auroc written for the transfer headline')

# ---- 4. real-noise code paths ----
print('\n4. real-noise code paths')
ur = dg._GradeUnit(ds='smoke', is_clf=False, condition='real_noise',
                   level=0.0, draw=0, fold=0, model='rf', mi='base_1@rf',
                   y0=y_clean, g_tr=groups, cmask=None, errors=n42._Errors(),
                   unavailable={})
from experiments.noise import n44_detect_and_clean as n44
excl = np.arange(N) // 2
spr = n44._Space('own', X, None, ur)
sc, fl, why = dg.all_scores(ur, spr, None, detectors=dg.EXCLUSION_TAKERS,
                            exclusion=excl)
check(set(sc) | set(why) >= set(dg.EXCLUSION_TAKERS),
      f'exclusion takers {dg.EXCLUSION_TAKERS}: scored {sorted(sc)}, '
      f'reasons {sorted(why)}')
data = {'labels': y_clean, 'task_type': 'regression', 'reps': {'r': X},
        'groups': groups}
out = dg.run_real_noise(data, {'n_splits': 5}, _tmp, 'smoke')
check(out['database_disagreement'].get('skipped')
      and out['database_disagreement'].get('reason'),
      f"database disagreement skipped: "
      f"{str(out['database_disagreement'].get('reason'))[:150]}")
check(out['curator_flag'].get('skipped') and out['curator_flag'].get('reason'),
      f"curator flag skipped: {str(out['curator_flag'].get('reason'))[:150]}")
gid = np.arange(N) // 2
spread = {int(g): float(np.std(y0[gid == g])) for g in np.unique(gid)}
gt = GroundTruth(provider='made_up', statistic='spearman', units='label',
                 group_ids=gid, group_value=spread)
noisy_rows = np.abs(y0 - y_clean) + 0.01 * rng.normal(size=N)
scores = {'base_1@rf': {'oof_residual': noisy_rows,
                        'random': rng.uniform(size=N)},
          'hybrid@rf': {'oof_residual': noisy_rows + rng.normal(size=N),
                        'random': rng.uniform(size=N)}}
folds5 = [np.arange(i, N, 5) for i in range(5)]
gd = dg.grade_database_disagreement([gt], scores, sorted(scores), folds5, N,
                                    _tmp, 'smoke', None)
a = gd['ground_truth']['made_up']['per_space']['base_1@rf']['oof_residual']
check(a.get('value') is not None and a.get('chance') == 0.0,
      f"agreement graded: spearman {a.get('value')}, chance {a.get('chance')}")
check('base_1@rf' in gd['ground_truth']['made_up']['hybrid_minus_single'],
      'database hybrid-minus-single table written')
check('transfer_vs_n45_injected' in gd, 'transfer entry written '
      f"({gd['transfer_vs_n45_injected']['per_space'].get('base_1@rf', {}).get('refused')})")
flag = cmask.copy()
texts = ['Outside typical range' if f else None for f in flag]
texts[int(np.flatnonzero(flag)[0])] = 'Potential transcription error'
gc = dg.grade_curator_flag(flag, texts, np.arange(N) // 2, 'made up', scores,
                           sorted(scores), folds5)
r = gc['per_space']['base_1@rf']['oof_residual']
check(r['records'] is not None and 'chance_average_precision' in r['records']
      and r['molecules'] is not None and len(r['per_flag_text']) == 2,
      f"curator grading: AP {r['records']['detection_average_precision']:.3f} "
      f"(chance {r['records']['chance_average_precision']:.3f}), two texts")

print()
if any(e for e in u.errors.d):
    print('  unit errors:')
    for k, v in u.errors.d.items():
        print(f'    {k}: {v["count"]}x {v["first"][:160]}')
if failures:
    print(f'{len(failures)} FAILED')
    sys.exit(1)
print(f'all passed ({dt1:.0f}s in check 1)')
