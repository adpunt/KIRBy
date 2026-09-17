#!/usr/bin/env python3
"""Smoke for noise_ceiling_analyse: does the extrapolation find a ceiling it is shown?

PART 1. SCORES BUILT FROM A KNOWN CEILING AND A KNOWN BASELINE ERROR.

Every case here is generated, not measured. A ceiling c and a baseline error b are
chosen, scores are written at the six levels of the shared grid by evaluating a known
curve at total error sqrt(b**2 + L**2), and the script is asked to get c back. Nothing
about a real dataset is being claimed.

  exact_variance_linear    the scores ARE ceiling + slope * t**2, to machine precision.
                           The variance-linear fit must return the ceiling exactly.
  exact_error_linear       the scores are ceiling + slope * t. The error-linear fit must
                           return the ceiling exactly, and the variance-linear one must
                           not: reading a straight line in t as a straight line in t**2
                           misses, and by how much is printed.
  exact_rmse_quadrature    the scores are sqrt(ceiling**2 + k * t**2), which is how two
                           independent spreads combine in an rmse. The squared-score fit
                           must return the ceiling exactly.
  noisy_variance_linear    the same as the first case with fold-sized scatter added to
                           each score. The ceiling must come back close, and how close
                           is printed rather than asserted tight.
  no_baseline_error        b is zero, so the ceiling IS the level-0 score and the two
                           must agree.
  b_recovered_from_curve   the scores are ceiling + slope * sqrt(b**2 + L**2) with b
                           hidden. The three-parameter fit must find b without being
                           told it.
  wrong_shape_is_visible   the scores decay exponentially, which none of the fitted
                           forms is. The ceiling that comes back is wrong, and the
                           check is that the fit quality SAYS so -- a low fit R-squared
                           or a large residual -- rather than the number looking clean.

PART 2. A DATASET WHOSE PUBLISHED ERROR IS REGISTERED.

herg_ki has a published assay error in dataset_policy.REFERENCE_ERROR_LABEL_UNITS:
0.54 pKi units, Kramer et al. 2012. This part writes a result file in the shape
n09_lnl_survey writes, carrying that dataset's real provenance fields, and runs the
whole reader over it. The scores in it are GENERATED, so the extrapolated ceiling is a
property of the generator and not of hERG.

What it checks is that both numbers arrive and are printed beside each other: the
published error, and the one the curve gives. Neither is used to check the other. They
are different quantities measured different ways -- one is a figure from a paper, the
other is the third parameter of a three-parameter fit on six points -- so a smoke that
asserted they matched would be asserting something nobody has established.
"""

import os
import sys
import json
import math
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import numpy as np

import experiments.noise.noise_ceiling_analyse as nca
from experiments.dataset_policy import (REFERENCE_ERROR_LABEL_UNITS,
                                        noise_levels as dataset_noise_levels)

FAIL = []


def check(name, ok, detail=''):
    print("  [%s] %s%s" % ('PASS' if ok else 'FAIL', name,
                           (' -- ' + detail) if detail else ''))
    if not ok:
        FAIL.append(name)


GRID = list(dataset_noise_levels('herg_ki'))


def _curve(levels, scores, b, metric='rmse', lower_is_better=True,
           published_units=None, condition='gaussian', dataset='synthetic'):
    c = nca.Curve(dataset=dataset, experiment='smoke', condition=condition,
                  what='generated scores', metric=metric,
                  lower_is_better=lower_is_better, levels=levels, scores=scores,
                  source_file='(generated in the smoke)')
    c.b = b
    c.published_error_units = published_units
    c.b_source = '(set by the smoke)'
    return c


def _totals(levels, b):
    return np.sqrt(float(b) ** 2 + np.asarray(levels, dtype=float) ** 2)


# ---------------------------------------------------------------------------
# PART 1
# ---------------------------------------------------------------------------

def part1():
    print("\nPART 1: scores generated from a known ceiling and a known baseline error")
    print("  The grid is %s, in fractions of the clean label spread." % (GRID,))

    # --- exact_variance_linear -------------------------------------------
    c_true, slope, b = 0.6000, 0.8000, 0.5910
    t = _totals(GRID, b)
    scores = c_true + slope * t ** 2
    cur = _curve(GRID, scores.tolist(), b)
    f = nca.fit_var_linear(cur)
    err = abs(f.ceiling - c_true)
    check('exact_variance_linear: the ceiling comes back', err < 1e-8,
          'true ceiling %.6f rmse, extrapolated %.6f rmse, out by %.2e'
          % (c_true, f.ceiling, err))
    # The level-0 score is NOT the ceiling. It sits at total error b, above it.
    measured_at_level_zero = float(scores[0])
    gap = measured_at_level_zero - c_true
    check('exact_variance_linear: level 0 and the ceiling are different numbers',
          abs(gap - slope * b * b) < 1e-8,
          'measured at level 0 %.6f rmse, ceiling %.6f rmse, the labels\' own error '
          'costing %.6f rmse' % (measured_at_level_zero, c_true, gap))
    check('exact_variance_linear: the cost of the baseline error is reported',
          abs(f.cost_of_baseline_error - gap) < 1e-8,
          'reported %.6f rmse' % f.cost_of_baseline_error)

    # --- exact_error_linear ----------------------------------------------
    c_true, slope, b = 0.5500, 0.4000, 0.4030
    t = _totals(GRID, b)
    scores = c_true + slope * t
    cur = _curve(GRID, scores.tolist(), b)
    f_t = nca.fit_t_linear(cur)
    f_v = nca.fit_var_linear(cur)
    check('exact_error_linear: the error-linear fit comes back exactly',
          abs(f_t.ceiling - c_true) < 1e-8,
          'true %.6f rmse, extrapolated %.6f rmse' % (c_true, f_t.ceiling))
    check('exact_error_linear: the variance-linear fit does NOT, and the miss shows',
          abs(f_v.ceiling - c_true) > 1e-3 and f_v.max_residual > 1e-4,
          'variance-linear says %.6f rmse against a true %.6f, out by %.4f; its largest '
          'residual is %.4f rmse' % (f_v.ceiling, c_true, abs(f_v.ceiling - c_true),
                                     f_v.max_residual))

    # --- exact_rmse_quadrature -------------------------------------------
    c_true, k, b = 0.5000, 1.2000, 0.5910
    t = _totals(GRID, b)
    scores = np.sqrt(c_true ** 2 + k * t ** 2)
    cur = _curve(GRID, scores.tolist(), b)
    f = nca.fit_sq_quadrature(cur)
    check('exact_rmse_quadrature: the squared-score fit comes back exactly',
          f.refused is None and abs(f.ceiling - c_true) < 1e-8,
          'true %.6f rmse, extrapolated %.6f rmse' % (c_true, f.ceiling)
          if f.refused is None else f.refused)

    # --- noisy_variance_linear -------------------------------------------
    # The scatter is 0.01 rmse units on a curve spanning about 0.8, which is the size
    # of a fold-to-fold difference rather than a rounding error.
    c_true, slope, b = 0.6000, 0.8000, 0.5910
    rng = np.random.RandomState(0)
    t = _totals(GRID, b)
    errs = []
    for seed in range(20):
        rng = np.random.RandomState(seed)
        scores = c_true + slope * t ** 2 + rng.normal(0, 0.01, size=len(GRID))
        cur = _curve(GRID, scores.tolist(), b)
        errs.append(abs(nca.fit_var_linear(cur).ceiling - c_true))
    errs = np.asarray(errs)
    check('noisy_variance_linear: the ceiling survives fold-sized scatter',
          float(errs.max()) < 0.05,
          'scores scattered by 0.01 rmse; over 20 draws the ceiling misses by %.4f to '
          '%.4f rmse, median %.4f' % (errs.min(), errs.max(), float(np.median(errs))))

    # --- no_baseline_error ------------------------------------------------
    c_true, slope, b = 0.7000, 0.5000, 0.0
    t = _totals(GRID, b)
    scores = c_true + slope * t ** 2
    cur = _curve(GRID, scores.tolist(), b)
    f = nca.fit_var_linear(cur)
    check('no_baseline_error: with b zero the ceiling IS the level-0 score',
          abs(f.ceiling - float(scores[0])) < 1e-8
          and abs(f.cost_of_baseline_error) < 1e-8,
          'ceiling %.6f rmse, measured at level 0 %.6f rmse, the labels\' own error '
          'costing %.2e' % (f.ceiling, float(scores[0]), f.cost_of_baseline_error))

    # --- b_recovered_from_curve -------------------------------------------
    c_true, slope, b_true = 0.5500, 0.4000, 0.5910
    t = _totals(GRID, b_true)
    scores = c_true + slope * t
    cur = _curve(GRID, scores.tolist(), None)   # b deliberately NOT supplied
    f = nca.fit_free_baseline_error(cur)
    bhat = None if f.refused else f.extra.get(
        'baseline_error_fraction_of_spread_from_the_curve')
    check('b_recovered_from_curve: the fit finds the baseline error it was not told',
          bhat is not None and abs(bhat - b_true) < 1e-3,
          'hidden b %.4f of the label spread, recovered %.4f' % (b_true, bhat)
          if bhat is not None else (f.refused or 'no estimate'))
    check('b_recovered_from_curve: and the ceiling with it',
          f.refused is None and abs(f.ceiling - c_true) < 1e-3,
          'true ceiling %.6f rmse, extrapolated %.6f rmse'
          % (c_true, f.ceiling) if f.refused is None else f.refused)

    # --- wrong_shape_is_visible -------------------------------------------
    # An exponential decay is none of the fitted forms. The ceiling that comes back is
    # wrong; what has to be true is that the fit quality says so.
    c_true, b = 0.4000, 0.5910
    t = _totals(GRID, b)
    scores = c_true + 2.0 * (np.exp(3.0 * t) - 1.0)
    cur = _curve(GRID, scores.tolist(), b)
    f = nca.fit_var_linear(cur)
    span = float(np.max(scores) - np.min(scores))
    check('wrong_shape_is_visible: the residual flags a curve of the wrong shape',
          f.max_residual > 0.02 * span,
          'largest residual %.4f rmse against a curve spanning %.4f rmse; the '
          'extrapolated ceiling %.4f is not to be read' % (f.max_residual, span, f.ceiling))
    # And the three forms disagree with each other, which is the other tell.
    ceilings = [nca.fit_var_linear(cur).ceiling, nca.fit_t_linear(cur).ceiling,
                nca.fit_var_quadratic(cur).ceiling]
    spread = max(ceilings) - min(ceilings)
    check('wrong_shape_is_visible: the three forms disagree with each other',
          spread > 0.02 * span,
          'variance-linear %.4f, error-linear %.4f, variance-quadratic %.4f rmse; '
          'they span %.4f' % (ceilings[0], ceilings[1], ceilings[2], spread))

    # --- the fraction-of-labels refusal -----------------------------------
    cur = _curve(GRID, (0.6 + 0.8 * np.asarray(GRID) ** 2).tolist(), 0.5910,
                 condition='censoring')
    f = nca.fit_free_baseline_error(cur)
    check('censoring gets no quadrature',
          f.refused is not None and 'labels changed' in f.refused,
          f.refused or 'it fitted anyway')
    check('censoring: the total error IS the level, nothing added in quadrature',
          np.allclose(cur.totals(), np.asarray(GRID)),
          'totals %s' % np.round(cur.totals(), 4).tolist())


# ---------------------------------------------------------------------------
# PART 2
# ---------------------------------------------------------------------------

def _herg_provenance(level, b, tau):
    """A provenance row in the shape noise_helpers._already_there writes one."""
    return {
        'noise_spec_id': 'gaussian_e%04d_s0_spread' % int(round(level * 1000)),
        'noise_level': float(level),
        'dose_units': 'spread',
        'baseline_error_fraction_of_spread': float(b),
        'baseline_error_label_units': float(tau),
        'baseline_error_source': 'herg_ki',
        'total_noise_fraction_of_spread': float(math.sqrt(b * b + level * level)),
        'r2_ceiling_from_baseline_error': float(1.0 - b * b),
    }


def part2():
    print("\nPART 2: a dataset whose published assay error is registered")
    tau = REFERENCE_ERROR_LABEL_UNITS['herg_ki']
    # hERG's label spread, measured on chembl_herg_ki.csv 2026-09-12 and recorded in
    # noise_helpers._already_there: 0.9143 pKi units over 1415 molecules. Used here to
    # put b on the same axis the grid uses, exactly as _already_there does.
    label_spread = 0.9143
    b = tau / label_spread
    print("  Published: %.4f pKi units (Kramer et al. 2012), over a label spread of "
          "%.4f pKi" % (tau, label_spread))
    print("  so b = %.4f of the label spread." % b)
    print("  The scores below are GENERATED. Nothing here is a measurement of hERG.")

    c_true, slope = 0.5000, 0.4500
    t = _totals(GRID, b)
    scores = c_true + slope * t

    per_sigma = {}
    provenance = {}
    for L, s in zip(GRID, scores):
        per_sigma[str(L)] = {
            'base_no_lnl': {'mean': float(s), 'std': 0.0, 'n_folds': 3,
                            'folds': [1, 1, 1], 'fold_scores': [float(s)] * 3},
            'hybrid_no_lnl': {'mean': float(s) - 0.01, 'std': 0.0, 'n_folds': 3,
                              'folds': [1, 1, 1], 'fold_scores': [float(s) - 0.01] * 3},
            'weighted_hybrid_no_lnl': {'mean': float('nan'), 'n_folds': 0,
                                       'folds': [], 'fold_scores': []},
        }
        row = _herg_provenance(L, b, tau)
        provenance[row['noise_spec_id']] = row

    payload = {
        'timestamp': '(generated in the smoke)',
        'metric': 'rmse',
        'metric_name': 'rmse',
        'lower_is_better': True,
        'base_rep': 'ecfp4',
        'sigmas': GRID,
        'noise_condition': 'gaussian',
        'noise_provenance': provenance,
        'methods': [],
        'task_type': 'regression',
        'is_clf': False,
        'per_sigma': per_sigma,
        'result_status': 'ok',
        'has_computed_results': True,
    }

    out = tempfile.mkdtemp(prefix='noise_ceiling_smoke_')
    d = os.path.join(out, 'n09_lnl_survey')
    os.makedirs(d)
    with open(os.path.join(d, 'herg_ki.json'), 'w') as fh:
        json.dump(payload, fh, indent=2)

    curves, notes = nca.read_curves(out)
    check('part2: the reader found the generated file', len(curves) >= 1,
          '%d curve(s); notes: %s' % (len(curves), '; '.join(notes) or 'none'))
    if not curves:
        return

    cur = [c for c in curves if 'base_no_lnl' in c.what]
    check('part2: the base condition is among them', len(cur) == 1,
          'found %d' % len(cur))
    if not cur:
        return
    cur = cur[0]

    check('part2: b was read out of the file\'s own provenance, not recomputed',
          cur.b is not None and abs(cur.b - b) < 1e-9,
          'b = %s, source: %s' % (nca._fmt(cur.b), cur.b_source))
    check('part2: the published figure in the label\'s own units came through',
          cur.published_error_units is not None
          and abs(cur.published_error_units - tau) < 1e-9,
          '%.4f pKi units' % cur.published_error_units)
    check('part2: the label spread is recovered from the two provenance fields',
          cur.label_spread is not None and abs(cur.label_spread - label_spread) < 1e-6,
          '%.4f pKi' % cur.label_spread)

    tt = cur.totals()
    check('part2: total error at level 0 is b, not zero',
          abs(tt[0] - b) < 1e-9,
          'level 0.0 sits at total error %.4f of the label spread' % tt[0])
    check('part2: total error at the top of the grid is sqrt(b^2 + 1)',
          abs(tt[-1] - math.sqrt(b * b + 1.0)) < 1e-9,
          'level 1.0 sits at total error %.4f of the label spread' % tt[-1])

    rows = nca.published_side_by_side(curves)
    check('part2: a side-by-side row was produced', len(rows) >= 1,
          '%d row(s)' % len(rows))
    row = [r for r in rows if 'base_no_lnl' in r['curve']]
    row = row[0] if row else None
    check('part2: the row carries BOTH the published b and the curve\'s own',
          row is not None and row['published_b_fraction_of_spread'] is not None
          and row['b_from_the_curve'] is not None,
          ('published %.4f, from the curve %.4f, both as fractions of the label spread'
           % (row['published_b_fraction_of_spread'], row['b_from_the_curve']))
          if row and row['b_from_the_curve'] is not None else (row or {}).get('refused'))
    check('part2: the row carries BOTH ceilings',
          row is not None and row['ceiling_from_published_b'] is not None
          and row['ceiling_from_the_curve'] is not None,
          ('ceiling using the published figure %.4f rmse, ceiling using the curve\'s own '
           '%.4f rmse' % (row['ceiling_from_published_b'], row['ceiling_from_the_curve']))
          if row and row['ceiling_from_the_curve'] is not None else 'missing')

    print("\n  The whole read-out, as the script prints it:")
    nca.print_curve(cur)
    nca.print_published_side_by_side(rows)


# ---------------------------------------------------------------------------
# PART 3
# ---------------------------------------------------------------------------

def part3():
    """Nothing in a result file is dropped without the reader saying so.

    n13 writes two shapes of level key in ONE file. Nine of its conditions key
    per_strength by the level alone -- "0.3". The tenth, `partial`, keys it by the
    level AND the share of molecules the level reached -- "0.3_frac0.1" -- because that
    condition sweeps the share at a fixed strength. A reader that takes float() of the
    key loses the whole condition and says nothing, which is how this file first read
    as four conditions when it holds five.
    """
    print("\nPART 3: both shapes of level key in one file")

    def _cell(score):
        return {'noisy_score': float(score), 'clean_score': 10.0,
                'noisy_score_by_model': {'rf': float(score)},
                'clean_score_by_model': {'rf': 10.0}}

    payload = {
        'metric': 'rmse', 'metric_name': 'rmse', 'lower_is_better': True,
        'noise_types': ['gaussian', 'partial'],
        'task_type': 'regression', 'is_clf': False,
        'result_status': 'ok',
        'per_noise_type': {
            'gaussian': {'per_strength': {
                str(L): _cell(10.0 + 2.0 * L ** 2) for L in GRID}},
            'partial': {'per_strength': {
                '0.3_frac%g' % f: _cell(10.0 + 3.0 * f)
                for f in (0.1, 0.3, 0.5)}},
        },
    }
    curves = nca._read_n13(payload, 'synthetic', 'n13', '(generated in the smoke)')
    conds = sorted(set(c.condition for c in curves))
    check('part3: both conditions came through', len(conds) == 2,
          'conditions read: %s' % conds)

    g = [c for c in curves if c.condition == 'gaussian']
    check('part3: the plainly-keyed condition kept all six levels',
          len(g) == 1 and g[0].levels == [float(x) for x in GRID],
          'levels %s' % (g[0].levels if g else None))

    pt = [c for c in curves if c.condition.startswith('partial')]
    check('part3: the compound-keyed condition became a curve over the share',
          len(pt) == 1 and pt[0].levels == [0.1, 0.3, 0.5],
          'condition %s, x values %s'
          % (pt[0].condition if pt else None, pt[0].levels if pt else None))
    check('part3: and is marked as counting labels, so it gets no quadrature',
          bool(pt) and pt[0].x_counts_labels
          and list(pt[0].totals()) == [0.1, 0.3, 0.5],
          'totals %s' % (list(pt[0].totals()) if pt else None))
    if pt:
        f = nca.fit_free_baseline_error(pt[0])
        check('part3: the free-baseline fit refuses a share axis',
              f.refused is not None, f.refused or 'it fitted anyway')

    # The bug that hid it: setdefault('', []) ran before float() was allowed to fail,
    # so a block of compound keys looked as though it had a plain level axis.
    groups, unparsed = nca._level_groups(payload['per_noise_type']['partial']['per_strength'])
    check('part3: a compound-key block has NO plain level group',
          not groups.get('') and not unparsed,
          'groups %s, unparsed %s' % (sorted(groups), unparsed))


# ---------------------------------------------------------------------------
# PART 4
# ---------------------------------------------------------------------------

def _write(root, exp, name, payload):
    d = os.path.join(root, exp)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), 'w') as fh:
        if isinstance(payload, str):
            fh.write(payload)
        else:
            json.dump(payload, fh)


def part4():
    """The read-out survives every broken file, and writes something either way.

    This runs at the end of a noise job that may have taken a day. It must not be able
    to take that job down, whatever it finds on disk. Each case below is a shape a
    result directory has actually held or could hold: a half-written file, a run that
    wrote a skip, a condition where every fold failed and left NaN, a level axis with
    one value on it.
    """
    print("\nPART 4: broken files, and the run carries on")

    def _cell(score):
        return {'noisy_score': score, 'noisy_score_by_model': {'rf': score}}

    def _n13(per_strength, **extra):
        d = {'metric': 'rmse', 'metric_name': 'rmse', 'lower_is_better': True,
             'task_type': 'regression', 'is_clf': False, 'result_status': 'ok',
             'per_noise_type': {'gaussian': {'per_strength': per_strength}}}
        d.update(extra)
        return d

    root = tempfile.mkdtemp(prefix='noise_ceiling_broken_')
    cases = {
        'truncated_json': ('n13_noise_distributions', 'd01.json',
                           '{"metric": "rmse", "per_noise'),
        'not_an_object': ('n13_noise_distributions', 'd02.json', '[1, 2, 3]'),
        'wrote_a_skip': ('n09_lnl_survey', 'd03.json',
                         {'result_status': 'skipped', 'reason': 'n too small'}),
        'every_score_nan': ('n13_noise_distributions', 'd04.json',
                            _n13({str(L): _cell(float('nan')) for L in GRID})),
        'every_score_identical': ('n13_noise_distributions', 'd05.json',
                                  _n13({str(L): _cell(1.0) for L in GRID})),
        'one_level_only': ('n13_noise_distributions', 'd06.json',
                           _n13({'0.3': _cell(1.0)})),
        'per_strength_is_a_list': ('n13_noise_distributions', 'd07.json',
                                   _n13([1, 2, 3])),
        'a_cell_is_a_string': ('n13_noise_distributions', 'd08.json',
                               _n13({str(L): 'oops' for L in GRID})),
        'scores_but_no_levels': ('n13_noise_distributions', 'd09.json',
                                 _n13({'low': _cell(1.0), 'high': _cell(2.0)})),
        'baseline_error_zero': ('n13_noise_distributions', 'd10.json',
                                _n13({str(L): _cell(1.0 + L) for L in GRID},
                                     noise_provenance={'g': {
                                         'baseline_error_fraction_of_spread': 0.0,
                                         'baseline_error_label_units': 0.54}})),
        'negative_rmse': ('n13_noise_distributions', 'd11.json',
                          _n13({str(L): _cell(-1.0 - L) for L in GRID},
                               noise_provenance={'g': {
                                   'baseline_error_fraction_of_spread': 0.59,
                                   'baseline_error_label_units': 0.54}})),
        'provenance_is_a_list': ('n13_noise_distributions', 'd12.json',
                                 _n13({str(L): _cell(1.0 + L) for L in GRID},
                                      noise_provenance=['not', 'a', 'dict'])),
        'no_metric_recorded': ('n13_noise_distributions', 'd13.json',
                               {'per_noise_type': {'gaussian': {'per_strength': {
                                   str(L): _cell(1.0 + L) for L in GRID}}}}),
    }
    for name, (exp, fname, payload) in cases.items():
        _write(root, exp, fname, payload)

    for name, (_exp, fname, _p) in sorted(cases.items()):
        ds = fname.split('.')[0]
        try:
            out = nca.write_readout(root, ds, verbose=False)
            wrote = out is not None and os.path.exists(str(out))
            failed_marker = os.path.exists(
                os.path.join(root, 'noise_ceiling', '%s.FAILED.txt' % ds))
            check('part4: %s' % name, wrote and not failed_marker,
                  'wrote %s' % (os.path.basename(str(out)) if out else 'nothing'))
        except Exception as e:
            check('part4: %s' % name, False,
                  'it RAISED, which would take the job down: %s: %s'
                  % (type(e).__name__, e))

    # A directory with nothing in it, and one that does not exist at all.
    empty = tempfile.mkdtemp(prefix='noise_ceiling_empty_')
    try:
        out = nca.write_readout(empty, 'nothing_here', verbose=False)
        check('part4: an empty results directory', out is not None
              and os.path.exists(str(out)), 'wrote %s' % out)
    except Exception as e:
        check('part4: an empty results directory', False, '%s: %s' % (type(e).__name__, e))

    missing = os.path.join(empty, 'does_not_exist')
    try:
        out = nca.write_readout(missing, 'nothing_here', verbose=False)
        check('part4: a results directory that is not there',
              out is not None and os.path.exists(str(out)), 'wrote %s' % out)
    except Exception as e:
        check('part4: a results directory that is not there', False,
              '%s: %s' % (type(e).__name__, e))

    # And the whole directory at once, which is what a job with several datasets meets.
    try:
        out = nca.write_readout(root, None, verbose=False)
        check('part4: every broken file in one pass', out is not None
              and os.path.exists(str(out)), 'wrote %s' % out)
    except Exception as e:
        check('part4: every broken file in one pass', False,
              '%s: %s' % (type(e).__name__, e))

    # The hook itself: run_all must call it, or none of the above ever fires.
    import inspect
    import experiments.noise.noise_helpers as nh
    src = inspect.getsource(nh.run_all)
    check('part4: noise_helpers.run_all calls the read-out',
          'write_readout' in src and 'noise_ceiling_analyse' in src,
          'found in run_all' if 'write_readout' in src else 'NOT WIRED IN')
    # The call must sit between a `try:` and an `except`, or an ImportError on the
    # module would take down a job that had already produced every result.
    _lines = src.splitlines()
    _i = next((i for i, ln in enumerate(_lines) if 'write_readout(' in ln), None)
    _before = '\n'.join(_lines[max(0, _i - 6):_i]) if _i is not None else ''
    _after = '\n'.join(_lines[_i:_i + 6]) if _i is not None else ''
    check('part4: and the call is wrapped so it cannot fail the job',
          _i is not None and 'try:' in _before and 'except Exception' in _after,
          'try: above and except Exception below the call'
          if _i is not None else 'no call found')


# ---------------------------------------------------------------------------
# PART 5
# ---------------------------------------------------------------------------

def part5():
    """A new experiment cannot be dropped without somebody being told.

    Two ways a curve goes missing and nothing says so. The first: an experiment is
    added to the noise suite and this script has no reader for it. The second, worse
    one: it sits in NOT_A_SWEEP, so the note reads like a decision rather than an
    oversight. n21 was in that table until 2026-09-17 while writing a score at every
    level, and the log said "scores how well corrupted molecules are found, not a
    prediction" as though somebody had meant it.
    """
    print("\nPART 5: nothing is dropped quietly")
    import experiments.noise.noise_helpers as nh

    known = set()
    for dirname, (short, _fn) in nca.READERS.items():
        known.add(short)
    for dirname in nca.NOT_A_SWEEP:
        known.add(dirname.split('_')[0])

    suite = set(nh.SINGLE_DATASET_EXPERIMENTS)
    unclassified = sorted(suite - known)
    check('part5: every experiment in the noise suite is classified',
          not unclassified,
          'unclassified: %s -- each needs a reader in READERS or an entry in '
          'NOT_A_SWEEP saying why it has no curve' % unclassified
          if unclassified else '%d experiments, all accounted for' % len(suite))

    # n21 specifically, because it is the one that was wrong.
    check('part5: n21 has a reader, not a skip',
          'n21_neighbour_distance' in nca.READERS
          and 'n21_neighbour_distance' not in nca.NOT_A_SWEEP,
          'READERS entry present' if 'n21_neighbour_distance' in nca.READERS
          else 'still skipped')

    # The guard itself: a directory labelled "not a sweep" that holds a sweep.
    root = tempfile.mkdtemp(prefix='noise_ceiling_mislabel_')
    name = sorted(nca.NOT_A_SWEEP)[0]
    d = os.path.join(root, name)
    os.makedirs(d)
    with open(os.path.join(d, 'ds.json'), 'w') as fh:
        json.dump({'metric': 'rmse', 'lower_is_better': True, 'result_status': 'ok',
                   'per_level': {str(L): {'noisy_score_by_model': {'rf': 1.0 + L}}
                                 for L in GRID}}, fh)
    _curves, notes = nca.read_curves(root)
    flagged = [n for n in notes if 'MARKED' in n and 'label is wrong' in n]
    check('part5: a mislabelled directory is called out, not skipped quietly',
          len(flagged) == 1, flagged[0][:150] if flagged else 'not flagged: %s' % notes)

    # And a directory nobody has written a reader for says what to do about it.
    root2 = tempfile.mkdtemp(prefix='noise_ceiling_unknown_')
    d2 = os.path.join(root2, 'n99_brand_new')
    os.makedirs(d2)
    with open(os.path.join(d2, 'ds.json'), 'w') as fh:
        json.dump({'metric': 'rmse', 'per_sigma': {}}, fh)
    _c2, notes2 = nca.read_curves(root2)
    check('part5: an unknown experiment says a reader is needed',
          any('needs one adding to READERS' in n for n in notes2),
          notes2[0][:140] if notes2 else 'no note at all')


def main():
    print("=" * 78)
    print("SMOKE: noise_ceiling_analyse")
    print("=" * 78)
    part1()
    part2()
    part3()
    part4()
    part5()
    print("\n" + "=" * 78)
    if FAIL:
        print("FAILED: %d" % len(FAIL))
        for f in FAIL:
            print("  - %s" % f)
        return 1
    print("ALL PASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())
