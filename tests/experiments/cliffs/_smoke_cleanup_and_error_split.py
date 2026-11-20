#!/usr/bin/env python3
"""cliff_aware_cleanup and error_by_cliff_status end to end on a made-up
dataset, with the per-fold search replaced by fixed inputs.

WHAT IT CHECKS (the changes of 2026-11-16):
  - Both units refuse to start, naming the setting, while their model inputs
    have no value, and fit only the inputs named once they do.
  - cliff_aware_cleanup takes detectors and cleanups only n44 had before
    (oof_residual, rewrite), a flag detector (conformal_bh), a neighbour
    detector and a neighbour cleanup, all through n44's own functions.
  - The fit count it prints is at least what it built, L's own settings
    included.
  - Both write a timing log.

WHAT IT DOES NOT CHECK. Real per-fold searches: fold_configurations and
fold_model_inputs are replaced, so no hybrid is searched and the model inputs
are fixed slices of the made-up representations.

Run:  python tests/experiments/cliffs/_smoke_cleanup_and_error_split.py
"""
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'KIRBY_LGBM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
HERE = Path(__file__).resolve().parent
for _p in (HERE.parent.parent.parent / 'src', HERE.parent.parent,
           HERE.parent.parent / 'experiments'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
warnings.filterwarnings('ignore')
BASE = '/Volumes/seagate' if os.path.ismount('/Volumes/seagate') else None
TMP = tempfile.mkdtemp(prefix='cliff_cleanup_smoke_', dir=BASE)
os.environ['KIRBY_CACHE_DIR'] = os.path.join(TMP, 'cache')

import numpy as np  # noqa: E402

from experiments.cliffs import _smoke_units as SU  # noqa: E402
from experiments.noise import noise_helpers as nh  # noqa: E402
from experiments import decided_grid  # noqa: E402

failures = []


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg, flush=True)
    if not cond:
        failures.append(msg)


def settings(**over):
    s = dict(SU.SETTINGS)
    s.update(over)
    p = Path(TMP) / 'CLIFF_SETTINGS.json'
    p.write_text(json.dumps(s, indent=1))
    os.environ['KIRBY_CLIFF_SETTINGS'] = str(p)


data = SU.made_up_dataset(n=90)
config = {'n_splits': 3}
os.environ['KIRBY_NOISE_ONLY_FOLDS'] = '0'

# The per-fold search, replaced: base_1 is the first representation, hybrid
# the two glued together, base_2 the second.
decided_grid.decided_models = lambda ds, tier='general': ['rf', 'svm']
nh.fold_configurations = lambda *a, **k: {
    m: {'base': ['cont_a', 'cont_b'], 'recipes': {}, 'hybrids': []}
    for m in ('rf', 'svm')}


def _inputs(data, dataset_name, model, entry, *, train_idx, val_idx, spaces,
            **k):
    a, b = data['reps']['cont_a'], data['reps']['cont_b']
    full = {'base_1': a, 'base_2': b, 'hybrid': np.hstack([a, b])}
    got = {s: (full[s][train_idx], full[s][val_idx]) for s in spaces
           if s in full}
    return got, {s: 'not in this smoke' for s in spaces if s not in full}, \
        {s: s for s in got}


nh.fold_model_inputs = _inputs

from experiments.cliffs import cliff_aware_cleanup as C  # noqa: E402
from experiments.cliffs import error_by_cliff_status as E  # noqa: E402

out = Path(TMP) / 'results'
t_all = time.time()

print('\ncliff_aware_cleanup with no model inputs written')
settings(outlier_detection_methods=['oof_residual'],
         label_handling_methods=['rewrite'])
r = C.run(data, config, str(out), 'made_up')
check(r.get('skipped') and 'cliff_aware_cleanup_model_inputs'
      in r['meta']['waiting_for'], 'refuses and names the model input setting')

print('\ncliff_aware_cleanup with n44-only, flag and neighbour methods')
M = ['oof_residual', 'cross_val', 'conformal_bh', 'neighbor_residual']
L = ['rewrite', 'remove', 'exp_downweight', 'label_smooth',
     'reliability_weight']
check(all(m in C.known_detectors(False) for m in M),
      f'every M named is accepted: {M}')
check(all(c in C.known_cleanups(False) for c in L),
      f'every L named is accepted: {L}')
settings(outlier_detection_methods=M, label_handling_methods=L,
         cliff_aware_cleanup_model_inputs=['base_1'])
t0 = time.time()
r = C.run(data, config, str(out), 'made_up')
rows = [json.loads(x) for x in Path(r['meta']['rows_file']).read_text().splitlines()]
print(f'  {len(rows)} rows in {time.time() - t0:.0f}s')
check(len(rows) > 0, 'rows written')
check({x['model_input'] for x in rows} == {'base_1'}, 'only base_1 fitted')
dets = {x['outlier_detection_method'].split('@')[0] for x in rows}
cls = {x['label_handling_method'] for x in rows}
check({'oof_residual', 'cross_val', 'conformal_bh', 'neighbor_residual'} <= dets
      or print('  detectors with rows:', dets, '\n  errors:',
               list(r['errors'].items())[:8]),
      f'rows under each M: {sorted(dets)}')
check(set(L) <= cls, f'rows under each L: {sorted(cls)}')
check(any(x['neighbour_setting'] != 'none' for x in rows),
      'neighbour settings reached the rows')
built = r['meta']['per_fold'][0]['n_cleaned_training_sets']
n_sets, how = C.cleaned_set_count(M, L, False, n_models=2)
check(all(v <= n_sets for v in built.values()),
      f'built {built} cleaned sets per unit; the printed count allows {n_sets}')
settings_in_rows = {(x['label_handling_method'], x['label_handling_setting'])
                    for x in rows if x['label_handling_method'] == 'exp_downweight'}
check(len({s.split('|')[-1] for _c, s in settings_in_rows})
      == len(C.n44.cleanup_settings('exp_downweight')),
      f"exp_downweight ran at its {len(C.n44.cleanup_settings('exp_downweight'))} "
      f"n44 settings")
check(all(np.isfinite(x['score_Dprime']) for x in rows
          if x['label_handling_method'] == 'rewrite'),
      "rewrite's D' scored")
check(any(x['cross_check_by_outlier_share'] for x in rows),
      'cross-check written')
tl = list((out / C.EXP_ID).glob('*.timing.jsonl.gz'))
check(bool(tl), f'timing log written: {[p.name for p in tl]}')

print('\nerror_by_cliff_status with no model inputs written')
settings()
r = E.run(data, config, str(out), 'made_up')
check(r.get('skipped') and 'error_by_cliff_status_model_inputs'
      in r['meta']['waiting_for'], 'refuses and names the model input setting')
print('\nerror_by_cliff_status with hybrid only')
settings(error_by_cliff_status_model_inputs=['hybrid'])
r = E.run(data, config, str(out), 'made_up')
fits = r['meta']['per_fold'][0]['fits'] if r['meta'].get('per_fold') else []
check(fits and all(f.endswith('|hybrid') for f in fits),
      f'fits only the named input: {fits}')
tl = list((out / E.EXP_ID).glob('*.timing.jsonl.gz'))
check(bool(tl), f'timing log written: {[p.name for p in tl]}')

print(f'\n{time.time() - t_all:.0f}s in all; files under {TMP}')
print('all passed' if not failures else f'{len(failures)} FAILED')
sys.exit(1 if failures else 0)
