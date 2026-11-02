#!/usr/bin/env python3
"""Smoke for n46_structure_specific_fixes: every family's unit function on
small random data, no per-fold search, no dataset.

What is checked: on random molecules (12 features) with injected censoring
(top quarter clipped), a real floor (bottom quarter set to the 25th
percentile), injected group offsets, simulated sources and fake repeat
measurements with fake documents, each family writes D-against-D' rows or a
recorded "not applicable"; under a kind that lacks the structure, the family
records "not applicable" and writes no row; every row key is distinct; every
row carries all_D, and all_Dprime wherever score_Dprime is a number.

Run:  python tests/experiments/noise/_smoke_n46.py
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
_base = ('/Volumes/seagate' if os.path.isdir('/Volumes/seagate') else
         '/private/tmp/claude-501/-Users-apunt-repos-KIRBy/'
         '62112545-c7bc-4a8e-9acd-c99b58998fb5/scratchpad')
_tmp = tempfile.mkdtemp(prefix='n46_smoke_', dir=_base)
os.environ['KIRBY_CACHE_DIR'] = os.path.join(_tmp, 'cache')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
warnings.filterwarnings('ignore')

import numpy as np

from experiments._shared import get_model_factories
from experiments.noise import n46_structure_specific_fixes as st
from experiments.noise import n07_remediation as n07
from experiments.noise import n17_replicate_removal as n17
from experiments.noise import n19_measurement_count as n19
from experiments.noise import n29_document_bias as n29
from experiments.noise import n39_correlated_sources as n39
from experiments.noise import n42_similarity_noise_sweep as n42
from experiments.noise.d_vs_dprime import row_key
from sklearn.metrics.pairwise import euclidean_distances

t_all = time.time()
rng = np.random.RandomState(0)
N, P = 160, 12
X = rng.normal(size=(N, P)).astype(np.float32)
beta = rng.normal(size=P)
y_clean = X @ beta + 0.3 * rng.normal(size=N)
tr, va = np.arange(128), np.arange(128, N)
groups = np.arange(N) // 4
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
metric = ('rmse', rmse, True)
fac = get_model_factories(['rf', 'xgboost', 'rf_l1', 'lightgbm_huber'],
                          task_type='regression', n_samples=len(tr))
swap_f = {k: v for k, v in fac.items() if k in ('rf_l1', 'lightgbm_huber')}

sink = n07.RowSink(_tmp, st.EXP_ID, 'smoke', 'smoke')
errors = n42._Errors()
na_log, unavailable = {}, {}
units = []


def unit(condition, level, y0, model='rf', *, Xtr=None, Xva=None, yv=None,
         ycl='default', g=None, mi='base_1', draw=0, fold=0, prov=None):
    u = st._Unit(ds='smoke', sink=sink, metric=metric, task_type='regression',
                 condition=condition, level=level, draw=draw, fold=fold,
                 model=model, mi=f'{model}:{mi}',
                 X_tr=X[tr] if Xtr is None else Xtr,
                 X_va=X[va] if Xva is None else Xva, y0=y0,
                 y_clean=y_clean[tr] if isinstance(ycl, str) else ycl,
                 y_val=y_clean[va] if yv is None else yv,
                 g_tr=groups[tr] if g is None else g, factory=fac[model],
                 takes_weight=True, errors=errors, na_log=na_log,
                 unavailable=unavailable, swap_factories=swap_f, prov=prov)
    units.append(u)
    return u


timings = {}

# 1. injected censoring: top quarter clipped, rf (rf_l1 swap) and xgboost (AFT)
t0 = time.time()
lim = float(np.quantile(y_clean, 0.75))
y_cens = np.minimum(y_clean[tr], lim)
for m in ('rf', 'xgboost'):
    st.censoring_fixes(unit('censoring', 0.25, y_cens, m,
                            prov={'censoring_limit': lim}))
u_gauss = unit('gaussian', 1.0, y_clean[tr] + rng.normal(size=len(tr)))
st.censoring_fixes(u_gauss)
timings['censoring'] = time.time() - t0

# 2. real censoring: a floor at the 25th percentile, training and held out
t0 = time.time()
flo = float(np.quantile(y_clean, 0.25))
st.censoring_fixes(unit('real_censoring', None, np.maximum(y_clean[tr], flo),
                        ycl=None, yv=np.maximum(y_clean[va], flo)),
                   side='lower')
timings['real_censoring'] = time.time() - t0

# 3. grouped shifts: one offset per group of 4, and a kind with no offset
t0 = time.time()
off = rng.normal(size=groups.max() + 1) * y_clean.std()
st.grouped_fixes(unit('grouped_shifted', 1.0, y_clean[tr] + off[groups[tr]]),
                 groups[tr], groups[va])
st.grouped_fixes(unit('outlier_p05', 1.0, y_clean[tr]), groups[tr])
timings['grouped'] = time.time() - t0

# 4. close pairs, in one stand-in setting (Euclidean on the features)
t0 = time.time()
yg = y_clean[tr] + 0.5 * y_clean.std() * rng.normal(size=len(tr))
st.pair_fixes(unit('gaussian', 0.5, yg),
              [('euclidean_smoke', lambda: (euclidean_distances(X[tr]).astype(np.float32), None))])
timings['close_pairs'] = time.time() - t0

# 5. sources: n39's structure, uniform 1-4 sources per molecule
t0 = time.time()
shares, _src = n39.count_shares('uniform_1_to_4')
stc = n39.draw_structure(N, shares, seed=7)
sigma_e = 1.0 * float(np.std(y_clean))
rho, tau = 0.3, 0.5 * sigma_e
val, y_mean_src, b_true = n39.simulate(y_clean, stc, sigma_e, rho, tau)
est = st.source_estimates(stc, val, tr, b_true, sigma_e)
st.source_fixes(unit('sources|uniform_1_to_4|rho0.3|tau0.5', 1.0, y_mean_src[tr]),
                est, sigma_e, rho, tau)
timings['sources'] = time.time() - t0

# 6. repeat measurements: 100 molecules measured 1-5 times by 15 documents
t0 = time.time()
n_mol = 100
Xm = rng.normal(size=(n_mol, P)).astype(np.float32)
ym = Xm @ beta[:P]
counts = rng.choice([1, 2, 3, 4, 5], size=n_mol, p=[0.3, 0.15, 0.25, 0.15, 0.15])
rg = np.repeat(np.arange(n_mol), counts)
rng.shuffle(rg)
doc_off = rng.normal(scale=0.5, size=15)
docs = rng.randint(0, 15, size=rg.size)
labels = ym[rg] + doc_off[docs] + 0.4 * rng.normal(size=rg.size)
stats = n19.measurement_stats(labels, rg)
order = stats['row_index']
Xmol = np.zeros((n_mol, P), dtype=np.float32)
Xmol[:] = Xm[rg[order]]
mol_groups = np.arange(n_mol) // 2
mtr, mva = np.arange(70), np.arange(70, n_mol)
ent = n17.build_entities(labels, rg, seed=0)
va_g = mva[ent['gradeable'][mva]]
assert len(va_g) >= 5, f'only {len(va_g)} gradeable held-out molecules'
arm_f = dict(get_model_factories(['ngboost'], task_type='regression', n_samples=70))
from kirby.find_best_hybrid import _get_gp_regressor
arm_f['gp'] = _get_gp_regressor
arm_f['xgboost'] = n19.fit_interval_aft_predict
for build in n19.LABEL_BUILDS:
    y_all, n_behind = n19.training_labels(ent, stats, build)
    lo, hi = n19.interval_bounds(ent, stats, build, y_all)
    var_own, _a, _b = n19.label_variance(stats, n_behind)
    var_cal, nfit, _c = n19.calibrated_variance(stats, n_behind)
    ctx = {'build': build, 'n_behind_tr': n_behind[mtr], 'var_own_tr': var_own[mtr],
           'var_cal_tr': var_cal[mtr], 'var_all_tr': stats['var_all'][mtr],
           'n_meas_tr': stats['n_measurements'][mtr],
           'n_meas_va': stats['n_measurements'][va_g], 'lo_tr': lo[mtr],
           'hi_tr': hi[mtr], 'arm_factories': arm_f, 'gp_rows': np.arange(len(mtr)),
           'run_model_arms': True}
    u = unit(st.REPEATS_ROW_CONDITION, None, y_all[mtr], Xtr=Xmol[mtr],
             Xva=Xmol[va_g], yv=ent['grade_label'][va_g], ycl=None,
             g=mol_groups[mtr])
    st.n19_fixes(u, ctx)
rec_mol = n29.record_to_molecule(rg)
doc_names, rec_doc = np.unique(docs.astype(str), return_inverse=True)
y_mean = stats['mean_all']
c29 = st.n29_fold_context(labels, rec_mol, rec_doc, y_mean, mtr, mva, mol_groups, n_mol)
u29 = unit(st.REPEATS_ROW_CONDITION, None, y_mean[mtr], Xtr=Xmol[mtr],
           Xva=Xmol[mva], yv=y_mean[mva], ycl=None, g=mol_groups[mtr])
resid = st.n26_error_by_count(u29, stats['n_measurements'][mva])
st.n29_fixes(u29, c29, known_bias={'labels': labels, 'rec_mol': rec_mol,
                                   'rec_doc': rec_doc, 'doc_names': doc_names,
                                   'tr': mtr, 'va': mva, 'y_mean': y_mean})
timings['repeats'] = time.time() - t0

# ---------------------------------------------------------------------------
failures = []


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg)
    if not cond:
        failures.append(msg)


rows = []
with open(sink.path) as fh:
    for line in fh:
        rows.append(json.loads(line))
keys = [row_key(r) for r in rows]
fams = {}
for r in rows:
    f, meth = r['method'].split('/', 1)
    fams.setdefault(f, set()).add(meth)
print(f'{len(rows)} rows; seconds per family: '
      + ', '.join(f'{k} {v:.0f}' for k, v in timings.items()))
for f in sorted(fams):
    print(f'   {f}: {sorted(fams[f])}')
check(len(keys) == len(set(keys)), 'every row key distinct')
check(all(r.get('all_D') for r in rows if r['score_D'] is not None
          and np.isfinite(r['score_D'])), 'every row with a D score carries all_D')
fin = [r for r in rows if r['score_Dprime'] is not None and np.isfinite(r['score_Dprime'])]
check(all(r.get('all_Dprime') for r in fin),
      f"all_Dprime on every row with a D' score ({len(fin)} of {len(rows)})")
check(all(set(r['all_D']) == {'rmse', 'mae', 'r2', 'spearman', 'pearson'}
          for r in rows if r.get('all_D')), 'all_D holds rmse, mae, r2, spearman, pearson')


def has(fam, meth=None, scored=True):
    for r in rows:
        f, m = r['method'].split('/', 1)
        if f == fam and (meth is None or m == meth):
            if not scored or (r['score_Dprime'] is not None
                              and np.isfinite(r['score_Dprime'])):
                return True
    return False


for meth in ('winsorise', 'constant_lift', 'per_molecule_lift',
             'per_molecule_lift_shuffled', 'truncated_normal_mean', 'censored_em',
             'drop_clipped', 'not_at_floor_control', 'xgboost_aft', 'rf_l1'):
    check(has('censoring', meth), f'censoring/{meth} scored')
check('censoring' in u_gauss.na and not any(
    r['noise_condition'] == 'gaussian' and r['method'].startswith('censoring/')
    for r in rows), 'censoring under gaussian: not applicable, no row')
check(any(f.startswith('censoring/lightgbm_huber') for f in na_log),
      'censoring/lightgbm_huber not applicable in rf and xgboost units')
for meth in ('winsorise', 'constant_lift', 'censored_em', 'drop_clipped',
             'not_at_floor_control'):
    check(has('real_censoring', meth), f'real_censoring/{meth} scored')
rc = [u for u in units if u.condition == 'real_censoring'][0]
check(len(rc.record['real_censoring'].get('three_scores', {})) > 5,
      "real censoring: n30b's three scores recorded per arm")
for meth in ('group_center', 'group_center_3plus', 'random_intercept'):
    check(has('grouped', meth), f'grouped/{meth} scored')
check('offset_recovery' in units[4].record.get('grouped', {}),
      'grouped: offset recovery recorded under grouped_shifted')
check('grouped' in units[5].na, 'grouped under outlier_p05: not applicable')
check(has('close_pairs', 'downweight_pairs'), 'close_pairs/downweight_pairs scored')
pr = units[6].record['close_pairs']['per_setting'].get('euclidean_smoke', {})
check(np.isfinite(pr.get('intercept_estimate', np.nan)),
      f"close pairs: noise estimate with no model, intercept "
      f"{pr.get('intercept_estimate', float('nan')):.3f} (injected variance "
      f"{units[6].record['close_pairs'].get('realised_injected_variance', float('nan')):.3f})")
for meth in ('offset_removed', 'weights_n19_fitted', 'weights_within_var',
             'offset_removed_within_var_weights', 'weights_rho0_oracle',
             'weights_true_rho_oracle'):
    check(has('sources', meth) or f'sources/{meth}' in na_log,
          f'sources/{meth} scored or not applicable')
for meth in [c for c in n19.CONDITIONS if c != 'untouched']:
    check(has('repeats_n19', meth) or f'repeats_n19/{meth}' in na_log,
          f'repeats_n19/{meth} scored or not applicable')
check(any(r['method_setting'] == 'label_build=single' for r in rows)
      and any(r['method_setting'] == 'label_build=averaged' for r in rows),
      'n19 single and averaged label builds both written')
for meth in ('ngboost_count_feature', 'gp_per_molecule_noise', 'xgboost_interval_aft'):
    check(has('repeats_n19', meth, scored=False), f'repeats_n19/{meth} row written')
for meth in ('per_record', 'per_record_bias', 'mean_label_bias_corrected',
             'per_record_random_effect', 'document_quantile_map'):
    check(has('repeats_n29', meth), f'repeats_n29/{meth} scored')
check('repeats_n29/within_document_rank' in u29.na,
      'within_document_rank recorded as never run (rf unit)')
check('known_bias_check' in u29.record.get('repeats_n29', {}),
      'n29 check with known biases recorded')
check('error_by_count' in u29.record.get('repeats_n26', {}),
      'n26 error by measurement count recorded')
if errors.d:
    print('  errors:')
    for k, v in errors.d.items():
        print(f'    {k}: {v["count"]}x {v["first"][:160]}')
check(len(errors.d) == 0, f'{len(errors.d)} distinct errors')
print(f'\nnot applicable, recorded: {sorted(na_log)}')
print(f'total {time.time() - t_all:.0f}s; files under {_tmp}')
if failures:
    print(f'{len(failures)} FAILED')
    sys.exit(1)
print('all passed')
