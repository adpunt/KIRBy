#!/usr/bin/env python3
"""Checks on hybrid_metric_scores.py that need no search and no server.

What it checks:
  1. The two inputs that are not decided have no default: --n-folds and
     --repetitions are required, and so is --output-dir.
  2. Resume: a row already in scores.jsonl is recognised by its key, a
     truncated last line is not, and append_rows round-trips.
  3. The metric list the script computes is simeval's own nine, split into the
     seven that read no injected noise and the two that do, with nothing added
     and nothing dropped.
  4. The coefficients offered for a 0/1, a count and a continuous matrix are
     the ones simeval defines for that kind, and the kind is read from the
     matrix.
  5. Noise injection: the level is a fraction of the standard deviation of the
     clean TRAINING labels, no held-out label moves, and two repetitions give
     two different realisations.
  6. Under a condition that corrupts every training molecule,
     noise_detect_auroc comes back NaN and is marked not defined, while
     noise_recovery is defined.
  7. The seven activity numbers come back on a real distance matrix.

Run:  python tests/experiments/cliffs/_smoke_hybrid_metric_scores.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent.parent
REPO = TESTS.parent
for _p in (REPO / 'src', TESTS, TESTS / 'experiments'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.cliffs import hybrid_metric_scores as H          # noqa: E402
from experiments.similarity_metrics import simeval as SE          # noqa: E402

FAILED = []


def check(label, cond, detail=''):
    print(f"  {'ok  ' if cond else 'FAIL'} {label}"
          f"{(' -- ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILED.append(label)


# ── 1. the inputs with no default ─────────────────────────────────────────
print('\n1. the inputs that are not decided have no default')
for missing, flags in (
        ('--n-folds', ['--datasets', 'herg_ki', '--repetitions', '1',
                       '--output-dir', '/x']),
        ('--repetitions', ['--datasets', 'herg_ki', '--n-folds', '5',
                           '--output-dir', '/x']),
        ('--output-dir', ['--datasets', 'herg_ki', '--n-folds', '5',
                          '--repetitions', '1'])):
    r = subprocess.run([sys.executable, str(HERE / 'hybrid_metric_scores.py')]
                       + flags, capture_output=True, text=True)
    check(f'{missing} is required',
          r.returncode != 0 and missing in (r.stderr + r.stdout),
          r.stderr[-300:])

r = subprocess.run([sys.executable, str(HERE / 'hybrid_metric_scores.py'),
                    '--datasets', 'herg_ki', '--n-folds', '5',
                    '--repetitions', '1', '--output-dir', '/x',
                    '--noise-conditions', 'not_a_condition'],
                   capture_output=True, text=True)
check('an unknown noise condition is refused by name',
      r.returncode != 0 and 'not_a_condition' in (r.stderr + r.stdout))


# ── 2. resume ──────────────────────────────────────────────────────────────
print('\n2. resume reads the rows already on disk')
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / 'scores.jsonl'
    rows = [
        dict(dataset='herg_ki', fold=0, hybrid='hybrid__svm',
             coefficient='cosine', noise_condition=None, noise_level=None,
             repetition=None, modi=0.5),
        dict(dataset='herg_ki', fold=0, hybrid='hybrid__svm',
             coefficient='cosine', noise_condition='outliers',
             noise_level=1.0, repetition=0, noise_detect_auroc=0.7),
    ]
    H.append_rows(path, rows)
    done = H.read_done(path)
    check('both rows come back', len(done) == 2, str(done))
    check('the activity row and the noise row are different keys',
          H._row_key(rows[0]) != H._row_key(rows[1]))
    check('a key that was written is recognised', H._row_key(rows[1]) in done)
    not_written = dict(rows[1], repetition=1)
    check('a key that was not written is not recognised',
          H._row_key(not_written) not in done)
    with open(path, 'a') as fh:
        fh.write('{"dataset": "herg_ki", "fold": 0, "hyb')
    check('a truncated last line is not counted as a finished row',
          len(H.read_done(path)) == 2)
    check('read_done on a file that is not there is empty',
          H.read_done(Path(td) / 'nope.jsonl') == set())


# ── 3. the metric list is simeval's own ───────────────────────────────────
print("\n3. the metrics are simeval's nine, split by whether they read noise")
check('seven activity metrics', len(H.ACTIVITY_EVALS) == 7,
      str(H.ACTIVITY_EVALS))
check('two noise metrics', len(H.NOISE_EVALS) == 2)
check('together they are exactly simeval.EVALS',
      sorted(H.ACTIVITY_EVALS + H.NOISE_EVALS) == sorted(SE.EVALS),
      f'{sorted(H.ACTIVITY_EVALS + H.NOISE_EVALS)} vs {sorted(SE.EVALS)}')
check('cliff_auroc is the one simeval does not call activity-grounded',
      [k for k, v in SE.EVAL_IS_ACTIVITY_GROUNDED.items() if not v]
      == ['cliff_auroc'])


# ── 4. coefficients by the assembled matrix's kind ────────────────────────
print("\n4. the coefficients offered come from the assembled matrix's kind")
rng = np.random.RandomState(0)
X_bits = (rng.rand(40, 12) > 0.5).astype(float)
X_count = rng.randint(0, 5, (40, 12)).astype(float)
X_cont = rng.randn(40, 12)
for X, want_kind, want_table in ((X_bits, 'binary', SE.BIT_COEFFS),
                                 (X_count, 'count', SE.COUNT_COEFFS),
                                 (X_cont, 'continuous', SE.CONT_COEFFS)):
    kind, coeffs = H.coefficients_for(X, 'smoke', 'x')
    check(f'{want_kind} matrix -> kind {want_kind}', kind == want_kind, kind)
    check(f'{want_kind} matrix -> {sorted(want_table)}',
          sorted(coeffs) == sorted(want_table), str(sorted(coeffs)))


# ── 5. injection: the level is a fraction of the training label SD ────────
print('\n5. the level is a fraction of the clean TRAINING labels'
      ' standard deviation')
n = 200
y = rng.randn(n) * 2.0 + 7.0
groups = np.repeat(np.arange(20), 10)
train_idx = np.arange(0, 160)
sd_train = float(np.std(y[train_idx]))
sd_all = float(np.std(y))
y_noisy, mask, eps, prov = H.inject(y, groups, train_idx, 'outliers', 1.0, 0,
                                    'regression')
check('the noisy labels cover the training rows only',
      len(y_noisy) == len(train_idx), f'{len(y_noisy)} vs {len(train_idx)}')
check('the standard deviation recorded is the training one',
      abs(prov['clean_training_label_sd'] - sd_train) < 1e-9,
      f"{prov['clean_training_label_sd']} vs train {sd_train} "
      f"(all rows {sd_all})")
check('level 1 asks for noise of one training standard deviation',
      abs(prov['target_noise_label_units'] - sd_train) < 1e-6,
      str(prov['target_noise_label_units']))
check('the realised noise is about the level',
      0.7 <= prov['realised_noise_fraction_of_sd'] <= 1.4,
      str(prov['realised_noise_fraction_of_sd']))
check('outliers corrupt some but not all training molecules',
      0 < int(mask.sum()) < len(train_idx), str(int(mask.sum())))
_, _, eps2, prov2 = H.inject(y, groups, train_idx, 'outliers', 1.0, 1,
                             'regression')
check('two repetitions are two different realisations',
      not np.allclose(eps, eps2))
check('two repetitions are two different seeds',
      prov['noise_seed'] != prov2['noise_seed'])
check('the level is recorded as a fraction, not in label units',
      prov['noise_level_fraction_of_training_label_sd'] == 1.0)


# ── 6. gaussian corrupts every training molecule ──────────────────────────
print('\n6. a condition that corrupts every training molecule')
D_cont = SE.dist_euclidean(np.asarray(X_cont))
y_g, mask_g, eps_g, prov_g = H.inject(y, groups, train_idx, 'gaussian', 1.0, 0,
                                      'regression')
check('gaussian corrupts every training molecule', bool(mask_g.all()),
      str(int(mask_g.sum())))
D_tr = SE.dist_euclidean(rng.randn(len(train_idx), 12))
sc, failed = H.noise_scores(D_tr, y_g, mask_g, eps_g)
check('noise_detect_auroc is NaN with no uncorrupted molecules',
      not np.isfinite(sc['noise_detect_auroc']),
      str(sc['noise_detect_auroc']))
check('and is marked not defined', sc['noise_detect_defined'] is False)
check('the row says every training molecule was corrupted',
      sc['all_training_molecules_corrupted'] is True)
check('noise_recovery is still defined', sc['noise_recovery_defined'] is True,
      str(sc['noise_recovery']))
check('no metric raised', not failed, str(failed))

y_o, mask_o, eps_o, _ = H.inject(y, groups, train_idx, 'outliers', 1.0, 0,
                                 'regression')
sc_o, failed_o = H.noise_scores(D_tr, y_o, mask_o, eps_o)
check('under outliers noise_detect_auroc is a number',
      np.isfinite(sc_o['noise_detect_auroc']),
      str(sc_o['noise_detect_auroc']))


# ── 7. the seven activity numbers on a real distance matrix ───────────────
print('\n7. the seven activity numbers come back')
m = len(train_idx)
Xq = rng.randn(m, 16)
yq = Xq[:, 0] * 3 + rng.randn(m) * 0.3
Dq = SE.dist_euclidean(Xq)
tan_ref = 1.0 - SE.dist_tanimoto((rng.rand(m, 32) > 0.5).astype(float))
sc_a, failed_a = H.activity_scores(Dq, yq, groups[train_idx], tan_ref)
check('no activity metric raised', not failed_a, str(failed_a))
for name in H.ACTIVITY_EVALS:
    check(f'{name} is a finite number', np.isfinite(sc_a[name]),
          str(sc_a.get(name)))
check('the cliff count is reported', isinstance(sc_a['n_cliffs'], int))
check('diag_ecfp_agreement is reported alongside',
      'diag_ecfp_agreement' in sc_a)


# ── 8. the two crashes of 2026-11-26, on herg_ki fold 0 ──────────────────
# Both killed a server run after the searches had finished, so neither was
# reachable from sections 1-7. These two checks need no search.
print('\n8. the two things that crashed the first server run')

import inspect                                                    # noqa: E402
from experiments import _shared as SH                             # noqa: E402
from experiments.cliffs import definitions as DEF                 # noqa: E402
from experiments.decided_grid import AXES                         # noqa: E402

fs_params = inspect.signature(SH.fold_searches).parameters
sof_params = inspect.signature(SH.search_on_fold).parameters
check('fold_searches has no rep_counts argument (it is a grid axis)',
      'rep_counts' not in fs_params and 'rep_counts' not in sof_params)
check('fold_searches passes a complete grid through as fixed_grid',
      'fixed_grid' in fs_params and 'fixed_grid' in sof_params)
src = inspect.getsource(SH.fold_searches)
check('and forwards it to search_on_fold', 'fixed_grid=fixed_grid' in src)
check('and keeps it in the memo key, so two grids are two searches',
      'json.dumps(\n            fixed_grid' in src or 'fixed_grid,' in src)

grid = DEF._whole_pool_grid('herg_ki')
# find_best_hybrid takes **kwargs, so it cannot say which names are real. The
# reference is the shape search_on_fold builds when no grid is passed in: that
# one is known to run. 'tier' is allowed on top, because the fixed_grid branch
# of search_on_fold reads it by name.
reference = set(SH._apply_dataset_search_speed(
    dict(tier='general', budgets=None, models=None,
         internal_evaluators=None), 'herg_ki')) | {'tier'}
check('the whole-pool grid holds only names the ordinary path also sets',
      set(grid) <= reference, str(sorted(set(grid) - reference)))
check('and aug_conditions is not one of them', 'aug_conditions' not in grid)
check('every searched axis is in it', set(AXES) <= set(grid),
      str(sorted(set(AXES) - set(grid))))
check('rep_counts on it is the whole pool', grid['rep_counts'] == [None],
      str(grid['rep_counts']))
for axis in AXES:
    if axis != 'rep_counts':
        check(f'{axis} is not empty', bool(grid[axis]), str(grid[axis]))

# The noise row: inject's provenance carries noise_condition and repetition of
# its own, and the row also needs them as its own columns.
_, mask_r, eps_r, nprov = H.inject(y, groups, train_idx, 'outliers', 1.0, 2,
                                   'regression')
check('inject provenance carries noise_condition and repetition',
      'noise_condition' in nprov and 'repetition' in nprov)
sc_r, failed_r = H.noise_scores(D_tr, y_o, mask_r, eps_r)
common = dict(dataset='herg_ki', fold=0, hybrid='hybrid__svm',
              coefficient='cosine', feature_kind='continuous')
row = dict(common)
row.update(nprov)
row.update(sc_r)
row.update(noise_condition='outliers', noise_level=1.0, repetition=2,
           metrics_that_raised=failed_r or None)
check('a noise row builds with no duplicate column', row['repetition'] == 2)
check('its resume key is the one the columns name',
      H._row_key(row) == ('herg_ki', 0, 'hybrid__svm', 'cosine',
                          'outliers', 1.0, 2), str(H._row_key(row)))
check('the row is JSON, so it can be appended', isinstance(
    json.loads(json.dumps(row, default=str)), dict))


print()
if FAILED:
    print(f'{len(FAILED)} CHECK(S) FAILED:')
    for f in FAILED:
        print(f'  - {f}')
    raise SystemExit(1)
print('all checks passed')
