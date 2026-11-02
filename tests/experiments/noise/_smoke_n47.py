#!/usr/bin/env python3
"""Smoke for n47_noise_measurements: each unit-level function on random data,
no per-fold search, no dataset.

120 random molecules with real SMILES (small alcohols, ethers and amines),
one 0/1 representation of 64 features and one continuous one of 16, rf as
the model, KIRBY_NOISE_CONTROLS=1 so the relabelled part's twins run. Checked:
each measurement returns a value or a recorded reason; every row carries
all_D, and every row with a fitted D' carries all_Dprime; every row key is
distinct.

Not run here, and why: n01._run_cliff_stratified, n14's per-fold geometry,
n37's neighbour-setting sweep, c10's recipe movement and feature rebuild, and
n16 whole all need the per-fold hybrid searches of a registered dataset.
The noise in the Gamma check is added by hand (normal draws), because
inject_condition needs a registered dataset name.

Run:  python tests/experiments/noise/_smoke_n47.py
"""
import json
import os
import sys
import tempfile
import time
import warnings

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'KIRBY_LGBM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ['KIRBY_NOISE_CONTROLS'] = '1'
_base = ('/Volumes/seagate' if os.path.ismount('/Volumes/seagate') else
         '/private/tmp/claude-501/-Users-apunt-repos-KIRBy/'
         '62112545-c7bc-4a8e-9acd-c99b58998fb5/scratchpad')
_tmp = tempfile.mkdtemp(prefix='n47_smoke_', dir=_base)
os.environ['KIRBY_CACHE_DIR'] = os.path.join(_tmp, 'cache')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
warnings.filterwarnings('ignore')

import numpy as np

from experiments.noise import n47_noise_measurements as ms
from experiments.noise import n07_remediation as n07
from experiments.noise import n27_removal_cost_and_propagation as n27
from experiments.noise.d_vs_dprime import row_key
from kirby.scaling import FeatureScaler

t_all = time.time()
failures = []


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg, flush=True)
    if not cond:
        failures.append(msg)


rng = np.random.RandomState(0)
smiles = []
for a in range(1, 9):
    for b in range(0, 5):
        for het in ('O', 'N', 'S'):
            smiles.append('C' * a + het + 'C' * b if b else 'C' * a + het)
smiles = smiles[:120]
N = len(smiles)
bits = (rng.rand(N, 64) < 0.2).astype(np.float32)
cont = rng.normal(size=(N, 16)).astype(np.float32)
beta = rng.normal(size=16)
y = cont @ beta + 0.3 * rng.normal(size=N)
y[rng.choice(N, 6, replace=False)] += 4.0 * y.std()
groups = np.arange(N) // 4
data = {'reps': {'bits': bits, 'cont': cont}, 'labels': y, 'groups': groups,
        'smiles': smiles, 'task_type': 'regression', 'domain': 'molecular'}
tr, va = np.arange(0, 96), np.arange(96, N)
y_tr, y_va = y[tr], y[va]
sc = FeatureScaler().fit(cont[tr])
Xc_tr, Xc_va = sc.transform(cont[tr]), sc.transform(cont[va])
Xh_tr = np.hstack([Xc_tr, bits[tr]])
Xh_va = np.hstack([Xc_va, bits[va]])
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
metric = ('rmse', rmse, True)
sink = n07.RowSink(_tmp, ms.EXP_ID, 'smoke', 'smoke')
print(f'{N} molecules, temp dir {_tmp}')

# ── n01: the cliff reference, independent and circular ──
print('\nn01 cliff reference')
cr = ms.cliff_reference(data, y, False)
check(cr['cliff_reference_independent'] is True,
      f"with SMILES: reference independent, {cr['n_cliff_molecules']} of "
      f"{cr['n_molecules']} molecules in a cliff pair "
      f"({cr['info'].get('structure_reference')})")
no_smi = dict(data, smiles=None)
cr2 = ms.cliff_reference(no_smi, y, False)
check(cr2['cliff_reference_independent'] is False,
      f"without SMILES: circularity flag set, structure "
      f"{cr2['info'].get('structure_reference')}")

# ── n14: landscape without the per-fold geometry ──
print('\nn14 landscape')
t0 = time.time()
ls = ms.n14_landscape(data, 'smoke', {'n_splits': 3}, None, with_similarity=False)
check('cliff_geometry' in ls and ls['cliff_geometry']['n_compounds'] == N,
      f"cliff list: {ls['cliff_geometry']['n_cliff_compounds']} cliff molecules, "
      f"{ls['cliff_geometry']['n_cliff_pairs']} pairs, lean "
      f"{ls['cliff_geometry']['geometry_lean']} "
      f"({ls['ac_reference_provenance']['similarity_criterion']})")
check(set(ls['feasibility']) >= {'pu_calibration_feasible', 'has_replicates',
                                 'has_multi_assay'},
      f"feasibility flags: {ls['feasibility']}")
check(ls['stage0_diagnostics'] is not None or ls['stage0_error'],
      'stage-0: ' + (f"resid_sd {ls['stage0_diagnostics']['resid_sd_label_units']:.3f} "
                     f"label units, kurtosis "
                     f"{ls['stage0_diagnostics']['excess_kurtosis']:.2f}"
                     if ls['stage0_diagnostics'] else
                     f"reason recorded: {ls['stage0_error']}"))
mmp = dict(data, mmp_pairs=[(0, 1, 3.0), (2, 3, 0.1), (4, 5, -2.5)])
ls2 = ms.n14_landscape(mmp, 'smoke', {'n_splits': 3}, None, with_similarity=False)
check(ls2['ac_reference_provenance']['similarity_criterion']
      == 'matched_molecular_pair' and ls2['cliff_geometry']['n_cliff_pairs'] == 2,
      f"mmp_pairs read: {ls2['cliff_geometry']['n_cliff_pairs']} cliff pairs "
      f"from 3 pairs given (2 over 2 log units)")
print(f'  ({time.time() - t0:.0f}s)')

# ── n37: Gamma per representation, the check, error, SIMEX ──
print('\nn37 Gamma test, check, error share, SIMEX')
t0 = time.time()
by_rep, tables, errs = ms.gamma_by_representation('smoke', data['reps'],
                                                  ['bits', 'cont'], tr, va, y_tr)
check(not errs and all('intercept' in by_rep[r] and 'negative' in by_rep[r]
                       for r in ('bits', 'cont')),
      'intercept per representation: ' + ', '.join(
          f"{r} {by_rep[r]['intercept']:.3f} ({by_rep[r]['distance']}, "
          f"negative={by_rep[r]['negative']})" for r in by_rep))
passes = [(0.0, 0, y_tr)] + [
    (lv, d, y_tr + lv * y_tr.std() * np.random.RandomState(10 + d).normal(size=len(y_tr)))
    for lv in (0.5, 1.0) for d in (0, 1, 2)]
chk = ms.gamma_check(tables, y_tr, passes)
check(len(chk) == len(passes) * len(tables)
      and all(r['intercept'] is not None for r in chk),
      f'check: {len(chk)} intercepts on damaged labels')
estimates = {rn: t[2] for rn, t in tables.items()}
usable = [v for v in estimates.values() if v >= 0]
s2_grid = float(np.median(usable)) if usable else float('nan')
row, tails, simex, res, pred = ms.model_error_unit(
    sink, dataset_name='smoke', fold=0, model='rf', space='base_1',
    X_tr=Xc_tr, X_va=Xc_va, y_tr=y_tr, y_va=y_va, metric=metric,
    task_type='regression', estimates=estimates, own=estimates.get('cont'),
    s2_grid=s2_grid)
check(np.isfinite(row['mse']) and set(row['share_by_representation']) == {'bits', 'cont'},
      f"error: rmse {row['held_out_score']:.3f}, shares "
      f"{ {k: (None if v is None else round(v, 3)) for k, v in row['share_by_representation'].items()} }")
check('excess_kurtosis' in tails, f"tails: kurtosis {tails.get('excess_kurtosis', float('nan')):.2f}")
check(simex is not None and ('linear' in simex or 'not_run' in simex),
      'SIMEX: ' + (f"linear read at {list(simex['linear']['at'])}" if 'linear' in simex
                   else f"reason recorded: {simex.get('not_run')}"))
frec = {'fold': 0, 'noise_estimate_by_representation': by_rep, 'check': chk,
        'model_error': {'rf': {'base_1': row}}, 'simex': {'rf': {'base_1': simex}}}
summ = ms.noise_budget_summary([frec], {'rf': {'base_1': {
    'res': [res.tolist()], 'pred': [pred.tolist()], 'rows': [va.tolist()]}}},
    y, None)
check('noise_estimate_by_representation' in summ['summary']
      and 'rf' in summ['out_of_fold'] and summ['check'],
      'n37 summary, out-of-fold tails and check summary built')
print(f'  ({time.time() - t0:.0f}s)')

# ── n27: removal cost curve and relabelled ──
print('\nn27 removal cost and relabelled')
t0 = time.time()
ref, removal = ms.removal_cost_unit(
    sink, dataset_name='smoke', fold=0, model='rf', space='base_1',
    X_tr=Xc_tr, X_va=Xc_va, y_tr=y_tr, y_va=y_va, metric=metric,
    task_type='regression')
check(sorted(removal) == ['0.25', '0.5', '0.7', '0.8', '0.9']
      and all(len(v) == n27.REPEATS for v in removal.values()),
      f"removal: {len(removal)} shares x {n27.REPEATS} draws; share 1.0 rmse "
      f"{ref['score']:.3f}, share 0.25 draw 0 rmse {removal['0.25']['r0']['score']:.3f}")
rel = ms.relabel_unit(sink, dataset_name='smoke', fold=0, model='rf',
                      space='base_1', X_tr=Xc_tr, X_va=Xc_va, y_tr=y_tr,
                      y_va=y_va, g_tr=groups[tr], metric=metric,
                      task_type='regression')
check(sorted(rel) == ['0.25', '0.5', '1'] and all(np.isfinite(v['score']) for v in rel.values()),
      'relabelled: ' + ', '.join(f"alpha {a} rmse {v['score']:.3f} moved var "
                                 f"{v['moved_variance']:.3f}" for a, v in rel.items()))
s27 = n27._summarise({'base_1': {'rf': {'f0': ref}}},
                     {'base_1': {sh: {'rf': {r: {'f0': v} for r, v in by_r.items()}}
                                 for sh, by_r in removal.items()}},
                     {}, {'base_1': {a: {'rf': {'f0': v}} for a, v in rel.items()}},
                     True, 'rmse')
check(bool(s27['removal_cost']) and bool(s27['propagation']),
      'n27._summarise reads the unit outputs')
print(f'  ({time.time() - t0:.0f}s)')

# ── c10: score drop, level 0 and a damaged level ──
print('\nc10 score drop')
t0 = time.time()
records = []
y_dam = y_tr + 1.0 * y_tr.std() * np.random.RandomState(3).normal(size=len(y_tr))
for lv, d, yD in ((0.0, 0, y_tr), (1.0, 0, y_dam)):
    recs = ms.score_drop_unit(
        sink, dataset_name='smoke', condition='gaussian', level=lv, draw=d,
        fold=0, model='rf', inputs={'base_1': (Xc_tr, Xc_va),
                                    'hybrid': (Xh_tr, Xh_va)},
        names={'base_1': 'cont', 'hybrid': 'cont+bits'}, y_D=yD, y_clean=y_tr,
        y_va=y_va, metric=metric, task_type='regression',
        n_corrupted=0 if lv == 0 else len(y_tr))
    records.extend(recs.values())
check(len(records) == 4 and all(r['heldout_score'] is not None and
                                r['train_score'] is not None for r in records),
      f'{len(records)} configuration fits with held-out and training scores')
rob = ms.robustness_summary(records, 'gaussian', (0.0, 1.0), (0,), True)
d_h = rob['rf']['hybrid']['gaussian']['difference_damaged_minus_clean']['1.0']['mean']
d_b = rob['rf']['base_1']['gaussian']['difference_damaged_minus_clean']['1.0']['mean']
check(d_h is not None and d_b is not None,
      f'damaged minus clean rmse at level 1.0: base_1 {d_b:+.3f}, hybrid {d_h:+.3f}')
stab = [{'fold': f, 'model': 'rf', 'slot': 'hybrid', 'level': lv, 'draw': 0,
         'features': [['cont', list(range(8 - int(lv * 2)))], ['bits', [f, 5]]]}
        for f in (0, 1, 2) for lv in (0.0, 0.5, 1.0)]
fs = ms.feature_stability_summary(stab, 80, 'gaussian')
check('nogueira_phi_level0' in fs['rf']['hybrid'] and 'feature_nds' in fs['rf']['hybrid'],
      f"feature stability: nogueira {fs['rf']['hybrid']['nogueira_phi_level0']:.3f}, "
      f"slope of Jaccard on level {fs['rf']['hybrid']['feature_nds']:.3f}")
print(f'  ({time.time() - t0:.0f}s)')

# ── n16: rank rows against level 0 with the floor ──
print('\nn16 rank agreement')
t0 = time.time()
cells = ms.importance_rank_unit(Xc_tr, {0.0: y_tr, 1.0: y_dam}, 'lightgbm_gain',
                                fold=0, dataset_name='smoke')
check(set(cells) == {'0', '1', 'floor_seed'},
      f"cells {sorted(cells)}: level 1 spearman (either scored) "
      f"{cells['1']['spearman_either_scored']['mean']:.3f}, floor "
      f"{cells['floor_seed']['spearman_either_scored']['mean']:.3f}")
print(f'  ({time.time() - t0:.0f}s)')

# ── the rows ──
print('\nrows')
rows = []
with open(sink.path) as fh:
    for line in fh:
        rows.append(json.loads(line))
keys = [row_key(r) for r in rows]
by_method = {}
for r in rows:
    by_method[r['method']] = by_method.get(r['method'], 0) + 1
print(f'  {len(rows)} rows: {by_method}')
check(len(keys) == len(set(keys)), 'every row key distinct')
check(all(isinstance(r.get('all_D'), dict) and 'spearman' in r['all_D'] for r in rows),
      'every row carries all_D with every regression metric')
fitted = [r for r in rows if r['score_Dprime'] is not None
          and np.isfinite(r['score_Dprime'])]
check(all(isinstance(r.get('all_Dprime'), dict) for r in fitted),
      f'all {len(fitted)} rows with a fitted D\' carry all_Dprime')
want = {'c10_score_drop': 4, 'n27_removal_cost': 5 * n27.REPEATS,
        'n27_move_toward_oof': 3, 'n37_model_error': 1, 'n37_simex': 9}
check(all(by_method.get(k) == v for k, v in want.items()),
      f'row counts per method as expected {want}')
check(by_method.get('n27_move_toward_oof__random_twin') == 3
      and by_method.get('n27_move_toward_oof__shuffled_twin') == 3,
      'relabelled twins written under KIRBY_NOISE_CONTROLS=1')

print(f'\n{time.time() - t_all:.0f}s in all')
if failures:
    print(f'{len(failures)} FAILED')
    sys.exit(1)
print('all passed')
