#!/usr/bin/env python3
"""N45, detector grading: how well each detector finds wrong labels, graded
against four things.

HANDOFF_NOISE_RESTRUCTURE.md, section 2 (working name DG). Written 2026-11-02.
Replaces n06, n22, n31 and track B of n13. The old files stay where they are
until the author says the new ones replace them.

THE DETECTORS. Every detector n44 uses, imported from n44 so the two lists
cannot drift: n44.REGISTRY_DETECTORS (the registry with its aliases dropped),
n44.RESIDUAL_DETECTORS (n32's six), n44.N41_DETECTORS (n41's three), n41's
conformal flags (n44.flag_detectors), and the random control. The neighbour
detectors (n44.NEIGHBOUR_DETECTORS) are scored under every neighbour setting,
every neighbour count in n07.sweep_ks() and, for the ones that read it, every
counting scheme in n07.sweep_schemes(). Grading fits no model, so no setting
is screened out (the rule n06, n13 track B and n22 already followed). On a
classification dataset the regression-only detectors
(n44.REGRESSION_ONLY_DETECTORS) are not scored and say why.

THE DETECTOR SPACES. n44's: per decided model and outer fold, the model's
best single representation (base_1) and its first general hybrid
(hybrid), from the per-fold search. n44's second chain (a base model with the
detector in the hybrid) grades the same matrix with the same labels as the
hybrid input's own space, so the detector is graded once, under model input
hybrid. The hybrid-minus-single tables below answer that chain's question.

WHAT EACH DETECTOR IS GRADED AGAINST.

  (a) Injected noise. Per noise kind, level, draw, fold, model and model
      input: noise_helpers.detection_scores against the molecules the
      injector moved (AUROC, average precision, precision and recall at k,
      k = the number moved), with the chance level written beside each
      number (AUROC 0.5; average precision and precision at k = the share
      moved). Also the precision of each detector's top 5, 10 and 20 percent
      (n44.FLAG_SHARES), beside the share moved. Source: n13 track B, n44's
      unit-level detection record. Kinds: noise_helpers.decided_conditions
      plus n31's outlier_only_p05 (see THE NOISE KINDS). Levels
      decided_levels, level 0 included. Draws draws_for(kind, (0, 1, 2)).
      Folds noise_outer_folds, all 5. At level 0, and under a kind where
      every label moves (gaussian), no molecule is told apart from another,
      so there is nothing to grade: each detector records that reason and is
      not scored.

  (b) Real database disagreement. Source: n06. _ground_truth_providers says
      which rows are repeat measurements of one thing and how much they
      disagree: BindingDB's same drug on the same protein, the four Lhasa
      hERG collections calling one molecule differently, ChEMBL compounds
      measured in two papers. Each detector's score per row is correlated
      with its repeat group's disagreement (agreement(): Spearman for a
      spread, AUROC for a yes/no; chance 0 and 0.5). Kept from n06: the
      transfer headline (does the ordering of detectors under injected noise
      survive on real disagreement: rank correlation between (a)'s median
      AUROC per detector and (b)'s agreement per detector; also n06's own
      comparison with the stored n11 and n13 files) and the hybrid-minus-
      single table.

  (c) Curator flags. Source: n22. ChEMBL's data_validity_comment on hERG
      IC50 records (301 records on 215 molecules, counted 2026-09-17).
      n22.grade: average precision, precision and recall at k, chance level
      beside them, over records and over molecules, and per flag text.
      Kept from n22: 293 of the 301 say "Outside typical range", which
      ChEMBL assigns because the value is extreme, so a detector that reads
      the label's distance from its neighbours is graded partly against a
      property of the label itself. The two texts are graded apart. The
      hybrid-minus-single table.

  (d) The ceiling. Source: n31. Per unit of (a), for a regression set: the
      model's own error on clean labels in label-SD units (s_m), the AUROC
      n31's formulas predict a residual detector can reach from it (shift
      kinds: Phi(delta / (sqrt(2) s_m)); outlier_only kinds:
      (2/pi) arctan(sqrt(1 + rho^2))), and the AUROC the absolute and the
      signed out-of-fold residual measured. The residual's shape (excess
      kurtosis, scaffold-group ICC, correlation with the prediction) on
      noisy and on clean labels. The predicted number sits beside every
      detector's measured AUROC in the summary.

  (e) The set-precision flags. Source: n31 (synthetic_flags, PRECISIONS,
      FLAG_SEEDS, OPERATIONS). Flag sets built from the injector's mask with
      a set share of true hits (0, 0.25, 0.5, 0.75, 1.0), at each flag share
      of the training molecules, over n31's two seeds. Each is acted on by
      n31's four operations (delete, weight 0.25, replace by the out-of-fold
      prediction, shrink toward it by the estimated weight), and the model
      is refitted. These are downstream fits, so every one is written as a
      d_vs_dprime row through n07.dprime_rows (section 11: every metric
      saved). The break-even precision (where the mean gain over the plain
      fit crosses zero, n31.break_even_precision) is computed per (model,
      model input, level, flag share, operation), with every detector's own
      precision at the same flag share beside it. Section 9.6: these are
      reported FIRST in Paper 3. Seed 0 is drawn with n44's oracle seed, so
      those flag sets equal n44's first pass and their fits are read from the
      noise cache when n44 ran first.

THE NOISE KINDS. noise_helpers.decided_conditions(data). On a regression set
also n31's outlier_only_p05: n31 kept it (DECIDED 2026-10-06) because its
wider-error formula assumes the molecules outside the widened share are
untouched, and the outlier_only kinds are the only ones where that holds.
Leaving it out would leave the wider-error formula with no kind to run on.
The set-precision flags run where n44 builds its oracle flags (a regression
kind with a moved subset); n31 skipped classification and n44 builds none
there, so on a classification set they record that reason.

QM9 (handoff section 10). Grading needs a known set of wrong labels. Only
injected noise gives one, and level 0 grades nothing, so QM9 is one dataset
among the others here.

WHAT ONE ROW IS (the rows file, part (e) only). d_vs_dprime's row: the
model fitted on D (training labels with the injected noise) and on D' (D
after one set-precision flag set went through one operation), both scored on
the held-out fold, beside the fit on clean labels. method is
"oracle@precision=<p>/<operation>"; method_setting is
"flag_share=<s>|seed=<k>".

WHERE THE GRADES ARE. (a), (d) and the realised precisions of (e): one
checkpoint line per unit (<dataset>__<kind>__<layout>.units.jsonl), and the
per-kind result file's summary. (b) and (c): <dataset>__real_noise.json,
with per-row scores in an .npz beside it.

NOT CARRIED, AND WHY.
  - n31's tail_scores (error on the top and bottom tenth of held-out labels,
    calibration slope) per fit: n07.dprime_rows saves d_vs_dprime's
    ALL_METRICS and has no field for them.
  - n31's four operations on the real detector's own flag set: n44 fits them
    (oof_residual x remove, top_share_downweight, rewrite, bayes_shrink at
    every flag share), so they are read from n44's rows, not fitted twice.
  - n06 and n22 capped gaussian_process at 1,000 rows. n44 does not cap it,
    and n45 scores through n44's code, so it does not either.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import time
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments._shared import (
    get_domain_metric, get_model_factories, input_fingerprint, convert_numpy,
)
from experiments.decided_grid import decided_models
from experiments.dataset_policy import noise_groups_for
from experiments.noise.noise_helpers import (
    sweep_conditions, decided_conditions, noise_outer_folds, fold_spread,
    fold_configurations, fold_model_inputs, configuration_record,
    decided_levels, inject_condition, level_unit, draws_for,
    detection_scores, detection_is_defined, DETECTION_IS_DEFINED,
)
from kirby.noise_spec import GROUPED_STRATEGIES
from kirby.utils.noise_estimation import ESTIMATOR_REGISTRY

from experiments.noise import noise_cache as nc
from experiments.noise import n07_remediation as n07
from experiments.noise import n31_regime_and_breakeven as n31
from experiments.noise import n32_iterative_and_dual_selection as n32
from experiments.noise import n42_similarity_noise_sweep as n42
from experiments.noise import n44_detect_and_clean as n44
from experiments.noise import _sim_axis_n15_n22 as sx
from experiments.noise.controls import controls_enabled
from experiments.noise import noise_timing
from experiments.noise.noise_helpers import (
    PAPER3_CUT_CONDITIONS, without_paper3_cuts)

# n06 and n22 import each other and n11 by bare module name, so they are
# imported the way they import themselves.
import n06_bindingdb_ground_truth as n06  # noqa: E402
import n22_curator_flag as n22  # noqa: E402
from _ground_truth_providers import (  # noqa: E402
    collect_ground_truths, agreement, rows_by_group, GROUP_SPREAD_DDOF,
)

EXP_ID = 'n45_detector_grading'
RUN_ID = 'n45'
RESULT_LAYOUT = 'detector_grading_2026-11-02'
REAL_NOISE_SUFFIX = 'real_noise'
REAL_NOISE_IDENTITY = 'n45_real_noise_v1_2026-11-02'

# From n44, so n44 and n45 grade the same detectors in the same spaces.
NOISE_DRAWS = tuple(n44.NOISE_DRAWS)
MODEL_INPUTS = tuple(n44.MODEL_INPUTS)
HYBRID_INPUT = n44.HYBRID_INPUT
BASE_INPUTS = tuple(m for m in MODEL_INPUTS if m != HYBRID_INPUT)
FLAG_SHARES = tuple(n44.FLAG_SHARES)
OOF_SPLITS = int(n44.OOF_SPLITS)
SCORE_DETECTORS = tuple(
    d for d in (n44.REGISTRY_DETECTORS + n44.RESIDUAL_DETECTORS
                + n44.N41_DETECTORS + n44.CONTROL_DETECTORS)
    if d not in n44.NEIGHBOUR_DETECTORS)
NEIGHBOUR_DETECTORS = tuple(n44.NEIGHBOUR_DETECTORS)
REGRESSION_ONLY_DETECTORS = tuple(n44.REGRESSION_ONLY_DETECTORS)

# From n31: the set-precision flags.
PRECISIONS = tuple(float(p) for p in n31.PRECISIONS)
FLAG_SEEDS = tuple(int(s) for s in n31.FLAG_SEEDS)
OPERATIONS = tuple(n31.OPERATIONS)
# Operations that read the out-of-fold prediction of a regression label.
OPERATIONS_NEEDING_P = ('rewrite', 'bayes_shrink')

REASON_REGRESSION_ONLY = ('regression only: it reads a numeric residual or a '
                          'spread (n44.REGRESSION_ONLY_DETECTORS)')
REASON_LEVEL_0 = 'level 0: no training label moved, so there is nothing to find'
REASON_EVERY_LABEL = ('every training label moved under this kind, so no '
                      'molecule is untouched to tell apart')
REASON_NO_ORACLE_CLF = (
    'not built on classification: n31 skipped classification and n44 builds '
    'its oracle flags only under DETECTION_IS_DEFINED, which holds no flip '
    'kind. Extending them is the author\'s decision.')

# Detectors that cannot be told to leave a row's own repeat measurements out
# of what they read, so on the real-noise grading (b, c) they partly read the
# disagreement they are graded against. knn_shapley (n32) values each row by
# its nearest neighbours' labels with no exclusion; lof (n41) measures local
# density in [features, residual], and a repeat measurement has identical
# features. Graded anyway, and marked in the result file.
READS_OWN_REPEATS = ('knn_shapley', 'lof')

# Registry detectors that take exclude_groups but are not among
# n44.NEIGHBOUR_DETECTORS: n44 runs them with no exclusion (scaffold-mates
# kept). On the real-noise grading they are given the row's own repeats as
# the exclusion, as n06 and n22 gave every estimator that takes it.
EXCLUSION_TAKERS = tuple(
    d for d in SCORE_DETECTORS if d in ESTIMATOR_REGISTRY
    and 'exclude_groups' in inspect.signature(
        ESTIMATOR_REGISTRY[d].__init__).parameters)


def default_conditions(data):
    """decided_conditions plus n31's outlier_only_p05 on a regression set."""
    conds = without_paper3_cuts(decided_conditions(data))
    if data.get('task_type') != 'classification':
        conds += [c for c in n31.VARIANCE_CONDITIONS if c not in conds]
    return conds


_READS_SCHEME = {}


def reads_scheme(det):
    if det not in _READS_SCHEME:
        _READS_SCHEME[det] = bool(n22.reads_scheme(det))
    return _READS_SCHEME[det]


class _GradeUnit:
    """What n44.detector_scores, n44.flag_detectors and the neighbour lists
    read off a unit: ds, y0, g_tr, is_clf, errors, condition, level, draw,
    fold, mi, model, unavailable."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class _AllSettings(n42._Similarity):
    """n42's neighbour lists for one unit, every setting, with no screen.
    The screen picks settings for refits; grading refits nothing."""

    def _screen(self):
        self.kept = []
        self.table = []
        self.kept_sk = []
        self.kept_settings = [s for s in self.settings if self.nb.get(s.name)]


# ---------------------------------------------------------------------------
# grading one score or one flag set
# ---------------------------------------------------------------------------

def grade_scores(cmask, scores, condition):
    """detection_scores plus the chance level beside each number and the
    precision of the top FLAG_SHARES. (grade, None) or (None, reason)."""
    g = detection_scores(cmask, scores, condition)
    if g is None:
        return None, ('detection_scores returned nothing: no moved molecule, '
                      'every molecule moved, or the lengths differ')
    g = dict(g)
    share = g['detection_n_corrupted'] / float(g['detection_n_training'])
    g['chance_auroc'] = 0.5
    g['chance_average_precision'] = share
    g['chance_precision_at_k'] = share
    m = np.asarray(cmask, dtype=bool)
    s = np.where(np.isfinite(scores), scores, -np.inf)
    by = {}
    for sh in FLAG_SHARES:
        fl = n44.flags_from(s, sh)
        nf = int(fl.sum())
        by[f'{sh:g}'] = {'precision': (float(m[fl].mean()) if nf else None),
                         'n_flagged': nf, 'chance_precision': share}
    g['precision_at_flag_share'] = by
    return g, None


def grade_flags(cmask, flags, condition):
    """A flag detector: precision and recall of its flags, chance beside."""
    m = np.asarray(cmask, dtype=bool)
    f = np.asarray(flags, dtype=bool)
    if not f.any():
        return None, 'the detector flagged no molecule'
    g, why = grade_scores(m, f.astype(float), condition)
    if g is None:
        return None, why
    g['flag_precision'] = float(m[f].mean())
    g['flag_recall'] = float(m[f].sum()) / float(m.sum())
    g['n_flagged'] = int(f.sum())
    g['chance_flag_precision'] = float(m.mean())
    return g, None


# ---------------------------------------------------------------------------
# every detector's scores in one space
# ---------------------------------------------------------------------------

def _first_error(errors, prefix):
    for k, v in errors.d.items():
        if k.startswith(prefix):
            return v['first']
    return None


def _excluded_registry_scores(u, X, name, exclusion):
    """A registry detector that takes exclude_groups, given the rows' own
    repeat groups (real-noise grading only). Arguments as
    n07._compute_noise_scores offers them, with exclude_groups set."""
    cls = ESTIMATOR_REGISTRY[name]
    accepts = set(inspect.signature(cls.__init__).parameters)
    offered = {'k': 10, 'metric': None, 'is_clf': u.is_clf,
               'groups': u.g_tr, 'exclude_groups': exclusion}

    def _compute():
        est = cls(**{k: v for k, v in offered.items() if k in accepts})
        est.fit(X, u.y0)
        return np.asarray(est.noise_scores_, dtype=float)
    key = {'detector': name, 'X': nc.array_hash(X), 'y': nc.array_hash(u.y0),
           'groups': nc.array_hash(u.g_tr),
           'exclude': nc.array_hash(exclusion), 'is_clf': bool(u.is_clf)}
    return nc.cached(u.ds, 'n45_excluded_detector', key, _compute,
                     store_if=lambda v: v is not None)


def all_scores(u, sp, lists_by_setting, *, ks=None, schemes=None,
               detectors=None, exclusion=None):
    """Every detector's output in space sp (an n44._Space on unit u).

    lists_by_setting  {setting name: neighbour lists or None}, or None when
                      no neighbour setting could be built (then each
                      neighbour detector records that reason).
    detectors         None = every detector. A smoke may pass a subset.
    exclusion         the rows' own repeat groups (real-noise grading); None
                      under injected noise.

    Returns (scores {key: array}, flags {key: bool array}, reasons {key:
    why not scored}).
    """
    ks = tuple(n07.sweep_ks() if ks is None else ks)
    schemes = tuple(n07.sweep_schemes() if schemes is None else schemes)
    scores, flags, reasons = {}, {}, {}
    want = None if detectors is None else set(detectors)

    for det in SCORE_DETECTORS:
        if want is not None and det not in want:
            continue
        if u.is_clf and det in REGRESSION_ONLY_DETECTORS:
            reasons[det] = REASON_REGRESSION_ONLY
            continue
        if exclusion is not None and det in EXCLUSION_TAKERS:
            try:
                sc = _excluded_registry_scores(u, sp.X_tr, det, exclusion)
            except Exception as exc:
                u.errors.add(f'detector={det}|{sp.name}', exc)
                sc = None
        else:
            sc = n44.detector_scores(sp, det)
        if sc is None:
            reasons[det] = (_first_error(u.errors, f'detector={det}|')
                            or _first_error(u.errors, 'oof_prediction')
                            or 'the detector returned no scores')
            continue
        sc = np.asarray(sc, dtype=float)
        if not np.isfinite(sc).any():
            reasons[det] = 'every score is nan or infinite'
            continue
        scores[det] = sc

    if want is None or 'conformal_bh' in want:
        if u.is_clf:
            reasons['conformal_bh'] = REASON_REGRESSION_ONLY
        else:
            got = n44.flag_detectors(sp)
            for name, fl in got:
                flags[name] = np.asarray(fl, dtype=bool)
            if not got:
                reasons['conformal_bh'] = (
                    _first_error(u.errors, 'detector=conformal_bh|')
                    or _first_error(u.errors, 'oof_prediction')
                    or 'no calibration share left enough molecules '
                       f'(n41.MIN_CALIBRATION = {n44.n41.MIN_CALIBRATION})')

    for det in NEIGHBOUR_DETECTORS:
        if want is not None and det not in want:
            continue
        if not lists_by_setting:
            reasons[det] = 'no neighbour setting was built for this unit'
            continue
        any_scored = False
        for sname, L in lists_by_setting.items():
            if L is None:
                reasons[f'{det}|{sname}'] = ('the neighbour lists of this '
                                            'setting were not built')
                continue
            for k in sx.ks_held(L, ks):
                for sch in (schemes if reads_scheme(det) else (None,)):
                    key = f'{det}|{sname}|k={k}|scheme={sch or "own"}'
                    Ls = L if sch is None else dict(L, scheme=sch)
                    try:
                        sc = nc.cached_detector_scores(
                            u.ds, sp.X_tr, u.y0, is_clf=u.is_clf, detector=det,
                            groups=u.g_tr, similarity_neighbours=Ls,
                            similarity_k=k)
                    except Exception as exc:
                        reasons[key] = f'{type(exc).__name__}: {exc}'
                        continue
                    if sc is None:
                        reasons[key] = 'the estimator failed inside its fit'
                        continue
                    scores[key] = np.asarray(sc, dtype=float)
                    any_scored = True
        if not any_scored and det not in reasons:
            reasons[det] = 'no setting and neighbour count gave a score'
    return scores, flags, reasons


# ---------------------------------------------------------------------------
# (d) the ceiling, n31's per-pass record
# ---------------------------------------------------------------------------

def ceiling_record(u, sp, y_clean, noise_groups_tr):
    """n31's formulas and residual shape for one unit, or the reason none."""
    if u.is_clf:
        return {'not_written_because': ('n31\'s formulas are for a regression '
                                        'residual')}
    y0 = np.asarray(u.y0, dtype=float)
    y_clean = np.asarray(y_clean, dtype=float)
    p = sp.p()
    if p is None:
        return {'not_written_because': (
            _first_error(u.errors, 'oof_prediction')
            or 'the out-of-fold prediction failed')}
    try:
        p_clean = nc.cached_oof_predict(u.ds, sp.X_tr, y_clean, u.g_tr,
                                        n_splits=min(OOF_SPLITS, len(y0)))
    except Exception as exc:
        return {'not_written_because': f'clean out-of-fold prediction: '
                                       f'{type(exc).__name__}: {exc}'}
    label_sd = float(np.std(y_clean))
    clean_mse = float(np.mean((y_clean - p_clean) ** 2))
    s_m = np.sqrt(clean_mse) / label_sd if label_sd > 0 else float('nan')
    resid = y0 - p
    rec = {'training_molecules': int(len(y0)), 'label_sd': label_sd,
           's_m_label_sd_units': float(s_m),
           'clean_oof_rmse_label_units': float(np.sqrt(clean_mse)),
           'residual_shape_noisy': n31.residual_shape(y0, p, noise_groups_tr),
           'residual_shape_clean': n31.residual_shape(y_clean, p_clean,
                                                      noise_groups_tr)}
    if float(u.level) <= 0:
        rec['detection_not_defined_because'] = REASON_LEVEL_0
        return rec
    if u.cmask is None or not detection_is_defined(u.condition):
        rec['detection_not_defined_because'] = REASON_EVERY_LABEL
        return rec
    mask = np.asarray(u.cmask, dtype=bool)
    disp = y0 - y_clean
    rec['molecules_moved'] = int(mask.sum())
    if mask.any() and label_sd > 0:
        if u.condition in n31.SHIFT_CONDITIONS:
            delta = float(np.mean(disp[mask])) / label_sd
            rec['realised_shift_label_sd_units'] = delta
            rec['predicted_auroc'] = n31.predicted_auroc_shift(delta, s_m)
            rec['predicted_auroc_formula'] = ('Phi(delta/(sqrt(2) s_m)), '
                                              'signed score')
        elif u.condition.startswith('outlier_only'):
            sig = float(np.sqrt(np.mean(disp[mask] ** 2))) / label_sd
            rec['realised_rms_injected_error_label_sd_units'] = sig
            rec['rho'] = float(sig / s_m) if s_m > 0 else float('nan')
            rec['predicted_auroc'] = n31.predicted_auroc_wider(sig, s_m)
            rec['predicted_auroc_formula'] = ('(2/pi) arctan(sqrt(1+rho^2)), '
                                              'absolute score')
        else:
            rec['predicted_auroc_not_written_because'] = (
                'n31 writes its formulas for the shift kinds (shift_only, '
                'sparse_dprime) and the outlier_only kinds only; the wider-'
                'error formula assumes the other labels untouched')
    rec['detection_abs_residual'] = detection_scores(mask, np.abs(resid),
                                                     u.condition)
    rec['detection_signed_residual'] = detection_scores(mask, resid,
                                                        u.condition)
    return rec


# ---------------------------------------------------------------------------
# (e) the set-precision flags, refitted, rows through n07.dprime_rows
# ---------------------------------------------------------------------------

def oracle_seed(u, share, prec, seed):
    """Seed 0 is n44.oracle_flags' own seed, so the flag set is n44's."""
    parts = ['n44_oracle', u.condition, f'{float(u.level):g}', u.draw,
             u.fold, share, prec]
    if int(seed) != 0:
        parts.append(int(seed))
    return n32.replicate_seed(*parts)


def emit_row(u, sink, method, method_setting, cleaned, *, X_tr, X_va, y_clean,
             y_val, metric, task_type):
    """One D' through n07.dprime_rows (d_vs_dprime's every metric). A
    weighted D' on a model whose fit takes no weight is recorded as not run,
    as n42 does."""
    if (cleaned is not None and cleaned.get('sample_weight') is not None
            and not u.takes_weight):
        u.unavailable.setdefault(f'{u.model}|{method}', n42.NO_WEIGHT_REASON)
        return None
    ident = n07.row_identity(
        dataset_name=u.ds, experiment=EXP_ID, method=method,
        method_setting=method_setting, similarity_setting=n07.NO_NEIGHBOURS,
        model_input=u.mi, noise_condition=u.condition, level=u.level,
        draw=u.draw, fold=u.fold)
    try:
        return n07.dprime_rows(
            sink, dataset_name=u.ds, cleaned=cleaned, X_tr=X_tr, y_D=u.y0,
            X_va=X_va, y_va=y_val, metric=metric, task_type=task_type,
            y_clean=y_clean, models=(u.model,), identity=ident,
            is_clf=u.is_clf)
    except Exception as exc:
        u.errors.add(f'{method}|fit', exc)
        return None


def oracle_rows(u, sp, sink, *, X_va, y_clean, y_val, metric, task_type):
    """Every (flag share, precision, seed, operation): the flag set from
    n31.synthetic_flags, the operation from n31.apply_operation, one row
    each. Returns {'realised_precision': {...}, 'not_run': {...}}."""
    out = {'realised_precision': {}, 'not_run': {}}
    if u.is_clf:
        out['not_run_because'] = REASON_NO_ORACLE_CLF
        return out
    if float(u.level) <= 0:
        out['not_run_because'] = REASON_LEVEL_0
        return out
    if (u.cmask is None or u.condition not in DETECTION_IS_DEFINED
            or not np.asarray(u.cmask).any()):
        out['not_run_because'] = REASON_EVERY_LABEL
        return out
    y = np.asarray(u.y0, dtype=float)
    X = sp.X_tr
    n = len(y)
    p = sp.p()
    for share in FLAG_SHARES:
        k = int(round(share * n))
        if k <= 0:
            out['not_run'][f'flag_share={share:g}'] = (
                f'{share:g} of {n} molecules rounds to no molecule')
            continue
        for prec in PRECISIONS:
            for seed in FLAG_SEEDS:
                flags, got = n31.synthetic_flags(
                    u.cmask, prec, oracle_seed(u, share, prec, seed), k)
                out['realised_precision'][
                    f'{share:g}|{prec:g}|{seed}'] = got
                for op in OPERATIONS:
                    method = f'oracle@precision={prec:g}/{op}'
                    setting = f'flag_share={share:g}|seed={seed}'
                    if op in OPERATIONS_NEEDING_P and p is None:
                        out['not_run'][f'{method}|{setting}'] = (
                            'the out-of-fold prediction failed')
                        continue
                    lab, keep, w = n31.apply_operation(
                        op, y, flags, p if op in OPERATIONS_NEEDING_P else None)
                    if keep is not None:
                        cleaned = n42._keep_dict(X, y, keep)
                    elif w is not None:
                        cleaned = n07.cleaned_from_weights(X, y, w)
                    else:
                        cleaned = n07.cleaned_from_labels(X, lab)
                    emit_row(u, sink, method, setting, cleaned, X_tr=X,
                             X_va=X_va, y_clean=y_clean, y_val=y_val,
                             metric=metric, task_type=task_type)
    return out


# ---------------------------------------------------------------------------
# one unit of (a), (d), (e)
# ---------------------------------------------------------------------------

def grade_unit(u, X_tr, X_va, lists_by_setting, *, sink, y_clean, y_val,
               metric, task_type, noise_groups_tr=None, ks=None, schemes=None,
               detectors=None):
    """Every detector graded against the injected noise of one (fold, level,
    draw, model, model input), the ceiling record, and the set-precision
    rows. `detectors` narrows the list for a smoke only; a run passes None.

    Returns the record that goes into the unit's checkpoint line:
      detection        {detector key: grade_scores}
      flag_detection   {flag detector key: grade_flags}
      not_graded       {detector key: why}
      ceiling          ceiling_record
      oracle           realised precision per (share, precision, seed)
    """
    t0 = time.time()
    sp = n44._Space('own', X_tr, X_va, u)
    rec = {'detection': {}, 'flag_detection': {}, 'not_graded': {}}
    defined = (float(u.level) > 0 and u.cmask is not None
               and detection_is_defined(u.condition)
               and 0 < int(np.sum(u.cmask)) < len(u.y0))
    if defined:
        scores, flags, reasons = all_scores(
            u, sp, lists_by_setting, ks=ks, schemes=schemes,
            detectors=detectors)
        rec['not_graded'].update(reasons)
        for key, sc in scores.items():
            g, why = grade_scores(u.cmask, sc, u.condition)
            if g is None:
                rec['not_graded'][key] = why
            else:
                rec['detection'][key] = g
        for key, fl in flags.items():
            g, why = grade_flags(u.cmask, fl, u.condition)
            if g is None:
                rec['not_graded'][key] = why
            else:
                rec['flag_detection'][key] = g
    else:
        why = REASON_LEVEL_0 if float(u.level) <= 0 else REASON_EVERY_LABEL
        names = [d for d in SCORE_DETECTORS + NEIGHBOUR_DETECTORS
                 + ('conformal_bh',)
                 if detectors is None or d in detectors]
        rec['not_graded'] = {d: why for d in names}
    rec['ceiling'] = ceiling_record(u, sp, y_clean, noise_groups_tr)
    n_before = sink.n_written
    rec['oracle'] = oracle_rows(u, sp, sink, X_va=X_va, y_clean=y_clean,
                                y_val=y_val, metric=metric,
                                task_type=task_type)
    rec['rows_written'] = int(sink.n_written - n_before)
    rec['seconds'] = round(time.time() - t0, 1)
    return rec


# ---------------------------------------------------------------------------
# the injected-noise run
# ---------------------------------------------------------------------------

def run(data, config, output_dir, dataset_name, **kwargs):
    """(a), (d), (e) for every noise kind, then (b) and (c) once."""
    force = bool(kwargs.get('force_rerun', False))
    models_asked = kwargs.get('models')
    index = sweep_conditions(
        _run_one_condition, EXP_ID, data, config, output_dir, dataset_name,
        default_conditions=default_conditions(data), kwargs=kwargs,
        groups_needed=GROUPED_STRATEGIES, cut_conditions=PAPER3_CUT_CONDITIONS)
    try:
        real = run_real_noise(data, config, output_dir, dataset_name,
                              models=models_asked, force_rerun=force,
                              injected_files=index.get('conditions_written'))
        from experiments._shared import save_experiment_results
        save_experiment_results(real, output_dir, EXP_ID, dataset_name,
                                fingerprint=input_fingerprint(data, config),
                                filename_suffix=REAL_NOISE_SUFFIX)
        index['real_noise_file'] = os.path.join(
            str(output_dir), EXP_ID, f'{dataset_name}__{REAL_NOISE_SUFFIX}.json')
        index['real_noise_skipped'] = {
            k: (v or {}).get('reason') for k, v in real.items()
            if isinstance(v, dict) and v.get('skipped')}
    except Exception as exc:
        index['real_noise_error'] = f'{type(exc).__name__}: {exc}'
        print(f"  real-noise grading failed: {index['real_noise_error']}")
    return index


def _models_for(dataset_name, task_type, n, asked=None):
    decided = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    models = [m for m in decided if m in asked] if asked else list(decided)
    factories = get_model_factories(models, task_type=task_type, n_samples=n)
    return decided, [m for m in models if m in factories], factories


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    print("\n" + "=" * 70)
    print("EXPERIMENT N45, DETECTOR GRADING: INJECTED NOISE, CEILING, "
          "SET-PRECISION FLAGS")
    print("=" * 70)
    t_start = time.time()

    task_type = data['task_type']
    is_clf = task_type == 'classification'
    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    augmentations = data.get('augmentations')
    metric = get_domain_metric(data)
    metric_name, _fn, lower_is_better = metric
    condition = kwargs.get('noise_condition') or default_conditions(data)[0]
    levels = tuple(float(x) for x in decided_levels(data, dataset_name, condition))
    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    n_molecules = len(set(smiles)) if has_smiles else len(labels)

    decided, models, factories = _models_for(dataset_name, task_type,
                                             len(labels), kwargs.get('models'))
    takes_weight = {m: bool(nc.fit_takes_sample_weight(factories[m]))
                    for m in models}

    folds, n_folds = noise_outer_folds(data, config)
    settings = n07.sweep_settings(dataset_name, n_molecules, has_smiles)
    ks = n07.sweep_ks()
    schemes = n07.sweep_schemes()
    draws = tuple(int(d) for d in draws_for(condition, NOISE_DRAWS))
    passes = [(lv, int(dr)) for lv in levels if lv != 0.0 for dr in draws]
    passes += [(lv, int(NOISE_DRAWS[0])) for lv in levels if lv == 0.0]
    noise_groups = noise_groups_for(data)[0]

    file_cond = f'{condition}__{RESULT_LAYOUT}'
    sink = n07.RowSink(output_dir, EXP_ID, dataset_name, file_cond)
    fingerprint = {'layout': RESULT_LAYOUT,
                   'input': input_fingerprint(data, config),
                   'settings': [s.name for s in settings], 'ks': list(ks),
                   'schemes': list(schemes), 'levels': list(levels),
                   'draws': list(draws), 'n_folds': int(n_folds),
                   'models': list(models), 'model_inputs': list(MODEL_INPUTS),
                   'detectors': list(SCORE_DETECTORS + NEIGHBOUR_DETECTORS),
                   'flag_shares': list(FLAG_SHARES),
                   'precisions': list(PRECISIONS),
                   'flag_seeds': list(FLAG_SEEDS),
                   'controls': bool(controls_enabled()), 'condition': condition}
    units = n07.FoldCheckpoint(output_dir, EXP_ID, dataset_name, file_cond,
                               tag='units', fingerprint=fingerprint)
    fold_ckpt = n07.FoldCheckpoint(output_dir, EXP_ID, dataset_name, file_cond,
                                   tag='folds', fingerprint=fingerprint)
    errors = n42._Errors()
    unavailable, provenance, fold_records = {}, {}, {}
    n_done = n_resumed = 0

    def _done(key):
        return units._k(key) in units.entries

    print(f"  condition {condition}; levels {list(levels)} "
          f"({level_unit(condition, is_clf)}); {len(passes)} (level, draw) "
          f"passes; {n_folds} outer folds; models {models}; inputs "
          f"{list(MODEL_INPUTS)}; rows -> {sink.path}")

    for fold_i, train_idx, val_idx in folds:
        train_idx = np.asarray(train_idx)
        val_idx = np.asarray(val_idx)
        all_keys = [[int(fold_i), float(lv), int(dr), m, mi]
                    for lv, dr in passes for m in models for mi in MODEL_INPUTS]
        frec_old = fold_ckpt.get([int(fold_i)])
        if frec_old is not None and all(_done(k) for k in all_keys):
            fold_records[str(fold_i)] = frec_old
            n_resumed += len(all_keys)
            print(f"  fold {fold_i}: every unit read back from {units.path}")
            continue
        print(f"\n  Fold {fold_i} of {n_folds}: {len(train_idx)} training, "
              f"{len(val_idx)} held-out rows")
        try:
            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=train_idx, n_folds=n_folds)
        except Exception as exc:
            print(f"    per-fold searches failed: {type(exc).__name__}: {exc}")
            fold_records[str(fold_i)] = {'error': f'per-fold searches failed: '
                                                  f'{type(exc).__name__}: {exc}'}
            continue
        frec = {'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)),
                'models': {}}
        for m in models:
            entry = entries.get(m)
            names = {}
            if entry is not None:
                for i, mi in enumerate(BASE_INPUTS):
                    if len(entry.get('base') or []) > i:
                        names[mi] = entry['base'][i]
                if entry.get('hybrids'):
                    names[HYBRID_INPUT] = entry['hybrids'][0].get('label')
            frec['models'][m] = configuration_record(entry, names)
        fold_records[str(fold_i)] = frec
        fold_ckpt.put([int(fold_i)], frec)

        y_clean = labels[train_idx]
        y_val = labels[val_idx]
        g_tr = None if groups is None else np.asarray(groups)[train_idx]
        ng_tr = (None if noise_groups is None
                 else np.asarray(noise_groups)[train_idx])
        smi_tr = None if smiles is None else [smiles[i] for i in train_idx]
        smi_va = None if smiles is None else [smiles[i] for i in val_idx]

        for level, draw in passes:
            todo = [(m, mi) for m in models for mi in MODEL_INPUTS
                    if not _done([int(fold_i), float(level), int(draw), m, mi])]
            n_resumed += len(models) * len(MODEL_INPUTS) - len(todo)
            if not todo:
                continue
            try:
                y0, cmask, prov = inject_condition(
                    data, dataset_name, condition, level, draw,
                    train_idx=train_idx, groups=noise_groups)
                provenance.setdefault(f'level={level:g}|draw={draw}',
                                      convert_numpy(prov))
            except Exception as exc:
                errors.add(f'inject|level={level:g}', exc)
                continue
            y0 = np.asarray(y0)
            cmask = None if cmask is None else np.asarray(cmask, dtype=bool)

            for m in models:
                mis = [mi for (mm, mi) in todo if mm == m]
                if not mis:
                    continue
                entry = entries.get(m)
                if entry is None:
                    for mi in mis:
                        unavailable.setdefault(f'fold{fold_i}|{m}:{mi}',
                                               'the per-fold search failed')
                    continue
                try:
                    ins, ins_unavail, _names = fold_model_inputs(
                        data, dataset_name, m, entry, train_idx=train_idx,
                        val_idx=val_idx, y_train=y0, factory=factories[m],
                        metric=metric, spaces=MODEL_INPUTS)
                    sp_tr, sp_va, sp_unavail = n07.fold_similarity_spaces(
                        dataset_name, list(settings), model_input_tr=None,
                        model_input_va=None, reps=reps, train_idx=train_idx,
                        val_idx=val_idx, y_train=y0, groups_train=g_tr,
                        base_reps=entry.get('base') or [],
                        recipes=entry.get('recipes') or {},
                        augmentations=augmentations, model_name=m,
                        model_factory=factories.get(m), metric=metric,
                        task_type=task_type)
                except Exception as exc:
                    errors.add(f'spaces|{m}', exc)
                    continue
                shared_lists = {}
                for mi in mis:
                    t0 = time.time()
                    ukey = [int(fold_i), float(level), int(draw), m, mi]
                    if mi not in ins:
                        unavailable.setdefault(f'fold{fold_i}|{m}:{mi}',
                                               ins_unavail.get(mi) or 'not built')
                        continue
                    X_tr, X_va = ins[mi]
                    u_err = n42._Errors()
                    u = _GradeUnit(
                        ds=dataset_name, is_clf=is_clf, condition=condition,
                        level=float(level), draw=int(draw), fold=int(fold_i),
                        model=m, mi=f'{m}:{mi}', y0=y0, g_tr=g_tr,
                        cmask=cmask, takes_weight=takes_weight[m],
                        errors=u_err, unavailable=unavailable)
                    lists = None
                    if settings:
                        try:
                            sim = _AllSettings(
                                u, settings, ks, dict(sp_tr, model_input=X_tr),
                                dict(sp_va, model_input=X_va), shared_lists,
                                sp_unavail, smi_tr, smi_va)
                            lists = dict(sim.nb)
                        except Exception as exc:
                            u_err.add('neighbour_lists', exc)
                    try:
                        noise_timing.set_context(
                            fold=int(fold_i), level=float(level), draw=int(draw),
                            model=m, model_input=mi)
                        with noise_timing.timed('piece'):
                            rec = grade_unit(u, X_tr, X_va, lists, sink=sink,
                                             y_clean=y_clean, y_val=y_val,
                                             metric=metric, task_type=task_type,
                                             noise_groups_tr=ng_tr, ks=ks,
                                             schemes=schemes)
                    except Exception as exc:
                        u_err.add(f'unit|{m}:{mi}', exc)
                        rec = {'error': f'{type(exc).__name__}: {exc}'}
                    rec['n_features'] = int(np.shape(X_tr)[1])
                    rec['errors'] = dict(u_err.d)
                    for k_, v_ in u_err.d.items():
                        e = errors.d.setdefault(k_, {'count': 0,
                                                     'first': v_['first']})
                        e['count'] += v_['count']
                    units.put(ukey, convert_numpy(rec))
                    n_done += 1
                    print(f"    fold {fold_i} level {level:g} draw {draw} "
                          f"{m}:{mi}: {len(rec.get('detection') or {})} graded, "
                          f"{rec.get('rows_written', 0)} rows, "
                          f"{time.time() - t0:.0f}s")

    summary = summarise_condition(units, sink, condition, lower_is_better)
    results = {
        'experiment': EXP_ID, 'run_id': RUN_ID, 'result_layout': RESULT_LAYOUT,
        'metric': metric_name, 'lower_is_better': bool(lower_is_better),
        'detection_metrics': ('detection_auroc (chance 0.5), '
                              'detection_average_precision and '
                              'detection_precision_at_k (chance = the share '
                              'of training molecules moved); higher better'),
        'noise_condition': condition, 'task_type': task_type,
        'levels': list(levels), 'level_unit': level_unit(condition, is_clf),
        'draws_per_nonzero_level': len(draws),
        'passes': [[float(lv), int(dr)] for lv, dr in passes],
        'n_folds': int(n_folds), 'folds_run': [int(f) for f, _t, _v in folds],
        'models': models, 'models_decided': decided,
        'model_inputs': list(MODEL_INPUTS),
        'detectors': {'score': list(SCORE_DETECTORS),
                      'neighbour': list(NEIGHBOUR_DETECTORS),
                      'flag': ['conformal_bh@cal|q'],
                      'regression_only': list(REGRESSION_ONLY_DETECTORS),
                      'source': 'n44_detect_and_clean, imported'},
        'neighbour_settings': [s.name for s in settings],
        'neighbour_ks': list(ks), 'neighbour_schemes': list(schemes),
        'neighbour_screen': ('none: grading fits no model, so every '
                             '(setting, k, scheme) is scored'),
        'set_precision_flags': {'precisions': list(PRECISIONS),
                                'flag_shares': list(FLAG_SHARES),
                                'seeds': list(FLAG_SEEDS),
                                'operations': list(OPERATIONS),
                                'downweight_to': n31.DOWNWEIGHT_TO,
                                'seed_0_is_n44s_oracle_seed': True},
        'folds': fold_records, 'summary': summary,
        'units_computed_this_run': int(n_done), 'units_resumed': int(n_resumed),
        'errors': errors.d, 'unavailable': unavailable,
        'noise_provenance': provenance,
        'rows': sink.summary(),
        'checkpoint': {'units': units.path, 'folds': fold_ckpt.path},
        'seconds': round(time.time() - t_start, 1),
    }
    print(f"\n  N45 {condition}: {n_done} units computed, {n_resumed} resumed, "
          f"{sink.n_written} rows written, {len(errors.d)} distinct errors, "
          f"{time.time() - t_start:.0f}s")
    return results


# ---------------------------------------------------------------------------
# the per-kind summary
# ---------------------------------------------------------------------------

DETECTION_KEYS = ('detection_auroc', 'detection_average_precision',
                  'detection_precision_at_k')
CHANCE_OF = {'detection_auroc': 'chance_auroc',
             'detection_average_precision': 'chance_average_precision',
             'detection_precision_at_k': 'chance_precision_at_k'}


def detector_base(key):
    """'neighbor_residual|base_1|none|cosine|k=5|scheme=rank' ->
    'neighbor_residual'; 'conformal_bh@cal=...' -> 'conformal_bh'."""
    return key.split('|', 1)[0].split('@', 1)[0]


def _unit_rows(units):
    out = []
    for kjson, v in units.entries.items():
        try:
            f_, lv_, dr_, m_, mi_ = json.loads(kjson)
        except Exception:
            continue
        out.append((int(f_), float(lv_), int(dr_), m_, mi_, v or {}))
    return out


def _per_fold_spread(items):
    """[(fold, value)] -> fold_spread of each fold's mean over draws."""
    by = {}
    for f_, x in items:
        if x is not None and np.isfinite(x):
            by.setdefault(f_, []).append(float(x))
    return fold_spread([float(np.mean(by[f_])) for f_ in sorted(by)])


def summarise_condition(units, sink, condition, lower_is_better):
    rows = _unit_rows(units)
    det = {}       # (level, model:mi) -> det key -> metric -> [(fold, value)]
    ceil = {}      # (level, model:mi) -> name -> [(fold, value)]
    prec = {}      # (level, model:mi, share) -> det key -> [(fold, value)]
    realised = {}  # (level, model:mi, share, precision) -> [values]
    for f_, lv_, _dr, m_, mi_, v in rows:
        cell = (lv_, f'{m_}:{mi_}')
        for key, g in list((v.get('detection') or {}).items()) + list(
                (v.get('flag_detection') or {}).items()):
            d = det.setdefault(cell, {}).setdefault(key, {})
            for mk in DETECTION_KEYS:
                d.setdefault(mk, []).append((f_, g.get(mk)))
                d.setdefault(CHANCE_OF[mk], []).append((f_, g.get(CHANCE_OF[mk])))
            for sh, pr in (g.get('precision_at_flag_share') or {}).items():
                prec.setdefault((lv_, f'{m_}:{mi_}', sh), {}).setdefault(
                    key, []).append((f_, pr.get('precision')))
        c = v.get('ceiling') or {}
        cc = ceil.setdefault(cell, {})
        cc.setdefault('predicted_auroc', []).append((f_, c.get('predicted_auroc')))
        cc.setdefault('s_m_label_sd_units', []).append(
            (f_, c.get('s_m_label_sd_units')))
        cc.setdefault('measured_auroc_abs_residual', []).append(
            (f_, (c.get('detection_abs_residual') or {}).get('detection_auroc')))
        cc.setdefault('measured_auroc_signed_residual', []).append(
            (f_, (c.get('detection_signed_residual') or {}).get('detection_auroc')))
        cc.setdefault('excess_kurtosis_noisy', []).append(
            (f_, (c.get('residual_shape_noisy') or {}).get('excess_kurtosis')))
        for kk, rp in ((v.get('oracle') or {}).get('realised_precision')
                       or {}).items():
            sh, pq, _sd = kk.split('|')
            realised.setdefault((lv_, f'{m_}:{mi_}', sh, pq), []).append(rp)

    detection = {}
    for (lv_, cell), by_det in det.items():
        pred = _per_fold_spread(ceil.get((lv_, cell), {}).get(
            'predicted_auroc', []))
        out = {}
        for key, d in by_det.items():
            out[key] = {mk: _per_fold_spread(xs) for mk, xs in d.items()}
            out[key]['predicted_ceiling_auroc'] = pred
        detection.setdefault(f'{lv_:g}', {})[cell] = out

    ceiling = {f'{lv_:g}': {} for lv_, _c in ceil}
    for (lv_, cell), d in ceil.items():
        ceiling[f'{lv_:g}'][cell] = {k: _per_fold_spread(xs)
                                     for k, xs in d.items()}

    # hybrid minus each single representation, per model and detector
    hyb = {}
    for (lv_, cell), by_det in det.items():
        m_, mi_ = cell.split(':', 1)
        if mi_ != HYBRID_INPUT:
            continue
        for base in BASE_INPUTS:
            other = det.get((lv_, f'{m_}:{base}'), {})
            for key, d in by_det.items():
                if key not in other:
                    continue
                pa = {}
                for f_, x in d['detection_auroc']:
                    pa.setdefault(f_, []).append(x)
                pb = {}
                for f_, x in other[key]['detection_auroc']:
                    pb.setdefault(f_, []).append(x)
                diffs = []
                for f_ in sorted(set(pa) & set(pb)):
                    xa = [x for x in pa[f_] if x is not None]
                    xb = [x for x in pb[f_] if x is not None]
                    if xa and xb:
                        diffs.append((f_, float(np.mean(xa) - np.mean(xb))))
                (hyb.setdefault(f'{lv_:g}', {}).setdefault(m_, {})
                 .setdefault(f'hybrid_minus_{base}', {}))[key] = {
                    'auroc_in_hybrid': _per_fold_spread(d['detection_auroc']),
                    f'auroc_in_{base}': _per_fold_spread(
                        other[key]['detection_auroc']),
                    'difference': _per_fold_spread(diffs)}

    # break-even precision from the rows file
    gains = {}
    if sink.path and os.path.exists(sink.path):
        with open(sink.path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                meth = str(r.get('method') or '')
                if (r.get('experiment') != EXP_ID
                        or not meth.startswith('oracle@precision=')
                        or '__' in meth):
                    continue
                pq, op = meth[len('oracle@precision='):].split('/', 1)
                sh = str(r.get('method_setting')).split('|')[0].split('=')[1]
                g = r.get('gain')
                if g is None or not np.isfinite(g):
                    continue
                key = (float(r['level']), str(r['model_input']), sh, op)
                gains.setdefault(key, {}).setdefault(pq, []).append(
                    (int(r['fold']), float(g)))
    break_even = {}
    for (lv_, cell, sh, op), by_p in gains.items():
        curve = {}
        for pq, xs in by_p.items():
            rp = [x for x in realised.get((lv_, cell, sh, pq), [])
                  if x is not None and np.isfinite(x)]
            curve[pq] = {'mean_gain': float(np.mean([g for _f, g in xs])),
                         'gain_by_fold': _per_fold_spread(xs),
                         'n_rows': len(xs),
                         'realised_precision_mean': (float(np.mean(rp))
                                                     if rp else None)}
        be, why = n31.break_even_precision(
            [curve[k]['realised_precision_mean'] for k in curve],
            [curve[k]['mean_gain'] for k in curve])
        dets = {k: _per_fold_spread(xs)
                for k, xs in prec.get((lv_, cell, sh), {}).items()}
        (break_even.setdefault(f'{lv_:g}', {}).setdefault(cell, {})
         .setdefault(f'flag_share={sh}', {}))[op] = {
            'curve': curve, 'break_even_precision': be,
            'break_even_refused_because': why,
            'detector_precision_at_this_flag_share': dets}

    medians = {}
    for (_lv, _cell), by_det in det.items():
        for key, d in by_det.items():
            medians.setdefault(detector_base(key), []).extend(
                x for _f, x in d['detection_auroc']
                if x is not None and np.isfinite(x))
    return {
        'detection': detection,
        'detection_note': ('per level, per "<model>:<model input>", per '
                           'detector key: each number is fold_spread over '
                           'folds of the fold\'s mean over draws; the chance '
                           'level of the same number is beside it; '
                           'predicted_ceiling_auroc is n31\'s formula for '
                           'that unit, where it is written'),
        'ceiling': ceiling,
        'hybrid_minus_single': hyb,
        'break_even': break_even,
        'break_even_note': ('gain = the score of D\' minus D in the dataset\'s '
                            'own direction (positive = the operation helped), '
                            'pooled over draws, folds and the two seeds; the '
                            'break-even is interpolated on the mean realised '
                            'precision (n31.break_even_precision)'),
        'detector_median_auroc': {k: (float(np.median(v)) if v else None)
                                  for k, v in medians.items()},
        'detector_median_auroc_note': ('median over every graded cell of '
                                       'this kind; read by the transfer '
                                       'headline of the real-noise grading'),
    }


# ---------------------------------------------------------------------------
# (b) and (c): real noise
# ---------------------------------------------------------------------------

def _merge_groups(n, arrays):
    """One label per row: rows that share a label in ANY of the arrays share
    one here. Negative labels are rows in no group. Union-find through
    n06.union_exclusion_groups."""
    objs = [types.SimpleNamespace(rows=rows_by_group(np.asarray(a)))
            for a in arrays if a is not None]
    return n06.union_exclusion_groups(objs, n)


def _injected_ranking(files):
    """{detector: median AUROC} from n45's own per-kind files: the median of
    each kind's detector_median_auroc."""
    per = {}
    used = []
    for _cond, path in (files or {}).items():
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except Exception:
            continue
        med = (doc.get('summary') or {}).get('detector_median_auroc') or {}
        used.append(path)
        for k, v in med.items():
            if v is not None:
                per.setdefault(k, []).append(float(v))
    return {k: float(np.median(v)) for k, v in per.items()}, used


def run_real_noise(data, config, output_dir, dataset_name, *, models=None,
                   force_rerun=False, injected_files=None, detectors=None):
    """(b) database disagreement and (c) curator flags. Each row is scored in
    its own fold's space (n06 and n22: the fold's training rows choose the
    space; every row of the fold is handed to the detector; the held-out
    rows keep their scores). Each detector is told the row's own repeat
    measurements wherever it can be (neighbour lists built with them
    excluded; registry detectors that take exclude_groups; every other
    detector's inner folds keep a repeat group on one side)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N45, DETECTOR GRADING: REAL DATABASE DISAGREEMENT AND "
          "CURATOR FLAGS")
    print("=" * 70)
    t_start = time.time()
    labels = np.asarray(data['labels'], dtype=np.float64)
    n = len(labels)
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric = get_domain_metric(data)
    out = {'experiment': EXP_ID, 'run_id': RUN_ID, 'metric': metric[0],
           'lower_is_better': bool(metric[2]), 'task_type': task_type,
           'n_samples': int(n), 'group_spread_ddof': GROUP_SPREAD_DDOF,
           'detectors_reading_own_repeats': {
               d: 'cannot be told to leave a row\'s own repeat measurements '
                  'out; partly reads the disagreement it is graded against'
               for d in READS_OWN_REPEATS},
           'detectors_given_the_repeat_exclusion': list(EXCLUSION_TAKERS)
           + list(NEIGHBOUR_DETECTORS)}

    gts, report = collect_ground_truths(data, dataset_name)
    out['database_disagreement'] = {'providers': convert_numpy(report)}
    if not gts:
        out['database_disagreement'].update({
            'skipped': True, 'reason': ('no ground-truth provider found a '
                                        f'repeat structure in {dataset_name}: '
                                        + '; '.join(f"{k}: {v.get('reason')}"
                                                    for k, v in report.items()))})
    flag, flag_text, flag_note = n22._flag_for_rows(data, dataset_name)
    out['curator_flag'] = {}
    molecule_of_row = None
    if flag is None:
        out['curator_flag'] = {'skipped': True, 'reason': flag_note}
    elif int(flag.sum()) in (0, n):
        out['curator_flag'] = {'skipped': True, 'reason': (
            f'{int(flag.sum())} of {n} records carry a flag, so precision '
            f'is fixed by construction')}
        flag = None
    else:
        rep = data.get('replicate_groups')
        if rep is None:
            from kirby.datasets.herg import load_herg_ic50_measurements
            rep = load_herg_ic50_measurements()['replicate_groups']
        molecule_of_row = np.asarray(rep, dtype=np.int64)
    if not gts and flag is None:
        print(f"  SKIP: {out['database_disagreement'].get('reason')}")
        print(f"  SKIP: {out['curator_flag'].get('reason')}")
        return out

    exclusion = _merge_groups(n, [gt.group_ids for gt in gts]
                              + ([molecule_of_row] if molecule_of_row is not None
                                 else []))
    sg = data.get('groups')
    fold_groups_all = _merge_groups(n, [exclusion] + (
        [np.asarray(_codes(sg))] if sg is not None else []))

    decided, models_, factories = _models_for(dataset_name, task_type, n, models)
    folds, n_folds = noise_outer_folds(data, config)
    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    n_molecules = len(set(smiles)) if has_smiles else n
    settings = n07.sweep_settings(dataset_name, n_molecules, has_smiles)
    ks, schemes = n07.sweep_ks(), n07.sweep_schemes()
    fingerprint = {'identity': REAL_NOISE_IDENTITY, 'dataset': dataset_name,
                   'n': int(n), 'n_folds': int(n_folds),
                   'labels': nc.array_hash(labels),
                   'exclusion': nc.array_hash(exclusion),
                   'models': list(models_), 'inputs': list(MODEL_INPUTS),
                   'detectors': list(SCORE_DETECTORS + NEIGHBOUR_DETECTORS),
                   'settings': [s.name for s in settings]}
    fp_tag = nc.array_hash(np.frombuffer(json.dumps(
        fingerprint, sort_keys=True).encode(), dtype=np.uint8))[:12]
    ckpt = n07.FoldCheckpoint(output_dir, EXP_ID, dataset_name,
                              REAL_NOISE_SUFFIX, tag='folds',
                              fingerprint=fingerprint)
    scores = {}     # space '<input>@<model>' -> key -> length-n array
    per_fold = {}
    errors = n42._Errors()

    def _put(space, key, vi, v):
        scores.setdefault(space, {}).setdefault(
            key, np.full(n, np.nan))[vi] = v

    for fold_i, train_idx, val_idx in folds:
        train_idx, val_idx = np.asarray(train_idx), np.asarray(val_idx)
        done = None if force_rerun else ckpt.get(['fold', int(fold_i)])
        if done is not None and os.path.exists(str(done.get('npz') or '')):
            z = np.load(done['npz'])
            for jk, (space, key) in (done.get('npz_keys') or {}).items():
                _put(space, key, z['val_idx'], z[jk])
            per_fold[str(fold_i)] = done.get('meta')
            print(f"  fold {fold_i}: read back from {done['npz']}")
            continue
        print(f"\n  === fold {fold_i}: {len(train_idx)} training rows, "
              f"{len(val_idx)} held-out rows ===")
        meta = {'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)),
                'configurations': {}, 'unavailable': {}, 'not_scored': {}}
        try:
            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=train_idx, n_folds=n_folds)
        except Exception as exc:
            meta['error'] = f'per-fold searches failed: {type(exc).__name__}: {exc}'
            per_fold[str(fold_i)] = meta
            continue
        rows = np.concatenate([train_idx, val_idx])
        is_val = np.zeros(len(rows), dtype=bool)
        is_val[len(train_idx):] = True
        y_rows = labels[rows]
        excl_rows = exclusion[rows]
        g_rows = fold_groups_all[rows]
        smi_rows = sx.smiles_for(data, rows)
        fold_scores = {}
        for m in models_:
            entry = entries.get(m)
            try:
                ins, unav, names = fold_model_inputs(
                    data, dataset_name, m, entry, train_idx=train_idx,
                    val_idx=val_idx, y_train=labels[train_idx],
                    factory=factories[m], metric=metric, spaces=MODEL_INPUTS)
            except Exception as exc:
                meta['unavailable'][m] = f'{type(exc).__name__}: {exc}'
                continue
            meta['configurations'][m] = configuration_record(entry, names)
            for mi, why in (unav or {}).items():
                meta['unavailable'][f'{mi}@{m}'] = why
            spaces_rows, nb_unavail = {}, {}
            if settings and entry is not None:
                try:
                    sp_tr, sp_va, nb_unavail = n07.fold_similarity_spaces(
                        dataset_name, list(settings), model_input_tr=None,
                        model_input_va=None, reps=data['reps'],
                        train_idx=train_idx, val_idx=val_idx,
                        y_train=labels[train_idx],
                        groups_train=(None if sg is None
                                      else np.asarray(sg)[train_idx]),
                        base_reps=entry.get('base') or [],
                        recipes=entry.get('recipes') or {},
                        augmentations=data.get('augmentations'), model_name=m,
                        model_factory=factories.get(m), metric=metric,
                        task_type=task_type)
                    spaces_rows = {k: np.vstack([sp_tr[k], sp_va[k]])
                                   for k in sp_tr
                                   if sp_tr.get(k) is not None
                                   and sp_va.get(k) is not None}
                except Exception as exc:
                    errors.add(f'neighbour_spaces|{m}', exc)
            for mi in MODEL_INPUTS:
                if mi not in ins:
                    continue
                space = f'{mi}@{m}'
                X_rows = n06._clean_matrix(np.vstack(ins[mi]))
                u_err = n42._Errors()
                u = _GradeUnit(ds=dataset_name, is_clf=is_clf,
                               condition='real_noise', level=0.0, draw=0,
                               fold=int(fold_i), model=m, mi=space, y0=y_rows,
                               g_tr=g_rows, cmask=None, errors=u_err,
                               unavailable=meta['unavailable'])
                sp = n44._Space('own', X_rows, None, u)
                lists = None
                if settings:
                    nbrs = n07.fold_neighbours(
                        dataset_name, settings,
                        dict(spaces_rows, model_input=X_rows), None, y_rows,
                        exclude_groups=excl_rows, smiles_train=smi_rows,
                        smiles_val=None, is_clf=is_clf)
                    lists = {s: (L if L and L.get('available') else None)
                             for s, L in nbrs.items()}
                    for s, L in nbrs.items():
                        if not (L and L.get('available')):
                            meta['unavailable'][f'{space}|{s}'] = (
                                (L or {}).get('reason')
                                or nb_unavail.get(s.split('|')[0]))
                sc, fl, why = all_scores(u, sp, lists, ks=ks, schemes=schemes,
                                         detectors=detectors,
                                         exclusion=excl_rows)
                for key, v in fl.items():
                    sc[key] = v.astype(float)
                fold_scores[space] = {k: np.asarray(v, dtype=float)[is_val]
                                      for k, v in sc.items()}
                meta['not_scored'][space] = why
                for k_, v_ in u_err.d.items():
                    e = errors.d.setdefault(k_, {'count': 0,
                                                 'first': v_['first']})
                    e['count'] += v_['count']
                print(f"    {space}: {len(sc)} detector keys scored, "
                      f"{len(why)} not")
        payload, npz_keys = {'val_idx': val_idx}, {}
        j = 0
        for space, d in fold_scores.items():
            for key, v in d.items():
                payload[f's{j}'] = v
                npz_keys[f's{j}'] = [space, key]
                _put(space, key, val_idx, v)
                j += 1
        meta = convert_numpy(meta)
        per_fold[str(fold_i)] = meta
        if output_dir is not None:
            npz = os.path.join(str(output_dir), EXP_ID,
                               f'{dataset_name}__{REAL_NOISE_SUFFIX}.fold'
                               f'{fold_i}.{fp_tag}.npz')
            os.makedirs(os.path.dirname(npz), exist_ok=True)
            np.savez_compressed(npz, **payload)
            ckpt.put(['fold', int(fold_i)], {'meta': meta, 'npz': npz,
                                             'npz_keys': npz_keys})

    out['per_fold'] = per_fold
    out['models'] = models_
    out['models_decided'] = decided
    out['errors'] = errors.d
    out['how_a_row_was_scored'] = (
        'per outer fold, the spaces are chosen by the fold\'s searches on its '
        'training rows; every row of the fold is handed to the detector in '
        'that space; only the held-out rows keep a score. Each row is scored '
        'once, in its own fold\'s space.')
    fold_rows = [np.asarray(va) for _f, _tr, va in folds]
    spaces = sorted(scores)
    if gts:
        out['database_disagreement'].update(
            grade_database_disagreement(gts, scores, spaces, fold_rows, n,
                                        output_dir, dataset_name,
                                        injected_files))
    if flag is not None:
        out['curator_flag'] = grade_curator_flag(
            flag, flag_text, molecule_of_row, flag_note, scores, spaces,
            fold_rows)
    if output_dir is not None and scores:
        path = os.path.join(str(output_dir), EXP_ID,
                            f'{dataset_name}__{REAL_NOISE_SUFFIX}_per_row.npz')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pay = {'labels': labels, 'exclusion_group': exclusion}
        if molecule_of_row is not None:
            pay['molecule_id'] = molecule_of_row
            pay['curator_flag'] = np.asarray(flag, dtype=int)
        keys = {}
        for i, sp in enumerate(spaces):
            for j, key in enumerate(sorted(scores[sp])):
                pay[f's{i}_{j}'] = scores[sp][key]
                keys[f's{i}_{j}'] = [sp, key]
        np.savez_compressed(path, **pay)
        out['per_row_scores_file'] = path
        out['per_row_scores_keys'] = keys
    out['seconds'] = round(time.time() - t_start, 1)
    return out


def _codes(values):
    """Integer codes for any label array (scaffold strings or ints)."""
    _u, inv = np.unique(np.asarray(values).astype(str), return_inverse=True)
    return inv.astype(np.int64)


def _hybrid_minus_single(spaces, value_of):
    """{base space: {hybrid space: {key: (hybrid, base, difference)}}} for the
    same model. value_of(space, key) -> number or None."""
    out = {}
    for sp in spaces:
        mi, m = sp.split('@', 1)
        if mi not in BASE_INPUTS:
            continue
        hs = f'{HYBRID_INPUT}@{m}'
        if hs not in spaces:
            continue
        row = {}
        for key in sorted(set(value_of.keys_of(hs)) & set(value_of.keys_of(sp))):
            a, b = value_of(hs, key), value_of(sp, key)
            row[key] = {'in_the_hybrid': a, f'in_{mi}': b,
                        'difference': (None if a is None or b is None
                                       else float(a - b))}
        out[sp] = {'model': m, 'hybrid_space': hs, 'per_detector': row}
    return out


class _Lookup:
    def __init__(self, table):
        self.t = table

    def keys_of(self, sp):
        return list(self.t.get(sp, {}))

    def __call__(self, sp, key):
        return self.t.get(sp, {}).get(key)


def grade_database_disagreement(gts, scores, spaces, fold_rows, n, output_dir,
                                dataset_name, injected_files):
    """n06's grading: agreement per (provider, space, detector key), pooled
    and per fold, chance beside it; hybrid minus single; transfer."""
    res = {'ground_truth': {}}
    headline = {}
    for gt in gts:
        chance = 0.0 if gt.statistic == 'spearman' else 0.5
        per_space, table = {}, {}
        for sp in spaces:
            per_key = {}
            for key, sc in scores[sp].items():
                a = agreement(gt, sc) or {}
                a.pop('p', None)
                a['chance'] = chance
                a['n_molecules_scored'] = int(np.isfinite(sc).sum())
                vals, refused = [], {}
                for i, va in enumerate(fold_rows):
                    only = np.full(n, np.nan)
                    only[va] = sc[va]
                    af = agreement(gt, only) or {}
                    vals.append(af.get('value'))
                    if af.get('refused'):
                        refused[str(i)] = af['refused']
                a['by_fold'] = dict(fold_spread(vals), refused_by_fold=refused)
                per_key[key] = a
                table.setdefault(sp, {})[key] = a.get('value')
            per_space[sp] = per_key
        res['ground_truth'][gt.provider] = {
            'statistic': gt.statistic, 'units': gt.units,
            'chance': chance, 'n_groups': gt.n_groups,
            'n_rows_covered': gt.n_rows_covered,
            'counts': convert_numpy(gt.counts), 'notes': gt.notes,
            'per_space': convert_numpy(per_space),
            'hybrid_minus_single': convert_numpy(
                _hybrid_minus_single(spaces, _Lookup(table))),
        }
        if not headline:
            for sp in spaces:
                per_det = {}
                for key, v in table.get(sp, {}).items():
                    if v is not None and np.isfinite(v):
                        per_det.setdefault(detector_base(key), []).append(v)
                headline[sp] = {d: float(np.median(v)) for d, v in per_det.items()}
            res['transfer_measured_against'] = gt.provider
    res['transfer_detector_number'] = (
        'per detector, the median of its agreement over its neighbour '
        'settings, counts and schemes (n06\'s rule); for every other '
        'detector, its agreement')
    inj, used = _injected_ranking(injected_files)
    tr = {'read_from': used, 'injected_ranking': inj, 'per_space': {}}
    for sp, per in headline.items():
        common = sorted(d for d in set(inj) & set(per))
        row = {'n_detectors_in_common': len(common),
               'detectors_in_common': common}
        if len(common) < n06.MIN_ESTIMATORS_FOR_TRANSFER:
            row['rank_correlation'] = None
            row['refused'] = (f'{len(common)} detectors in both orderings, '
                              f'below {n06.MIN_ESTIMATORS_FOR_TRANSFER}')
        else:
            from scipy.stats import spearmanr
            rho, _p = spearmanr([per[d] for d in common],
                                [inj[d] for d in common])
            row['rank_correlation'] = float(rho)
        tr['per_space'][sp] = row
    res['transfer_vs_n45_injected'] = tr
    res['transfer_vs_n45_injected_note'] = (
        'rank correlation between the detectors\' median detection AUROC '
        'under n45\'s injected kinds and their agreement with the real '
        'disagreement; positive = the two orderings agree. No p-value.')
    try:
        res['transfer_vs_stored_n11_n13'] = convert_numpy(
            n06.transfer_to_injected_noise(headline, output_dir, dataset_name))
    except Exception as exc:
        res['transfer_vs_stored_n11_n13'] = {
            'refused': f'{type(exc).__name__}: {exc}'}
    return res


def grade_curator_flag(flag, flag_text, molecule_of_row, flag_note, scores,
                       spaces, fold_rows):
    """n22's grading: records and molecules, per flag text, per fold;
    hybrid minus single on average precision over records."""
    n = len(flag)
    n_mol = int(molecule_of_row.max()) + 1
    flag_mol = n22._by_molecule(flag, molecule_of_row, n_mol, 'any')
    texts = {}
    for t in flag_text:
        if t:
            texts[t] = texts.get(t, 0) + 1
    res = {'how_the_flag_was_obtained': flag_note,
           'n_records': int(n), 'n_records_flagged': int(flag.sum()),
           'n_molecules': n_mol,
           'n_molecules_with_a_flagged_record': int(flag_mol.sum()),
           'flag_texts': dict(sorted(texts.items(), key=lambda kv: -kv[1])),
           'what_the_flag_is_and_is_not': {
               'outside_typical_range_is_about_the_value': (
                   f"{texts.get('Outside typical range', 0)} of the "
                   f"{int(flag.sum())} flagged records carry 'Outside typical "
                   f"range', which ChEMBL assigns because the measured value "
                   f"is extreme. A detector that scores a record by the "
                   f"distance between its label and its neighbours' labels is "
                   f"graded partly against a property of the label itself, "
                   f"not only against an independent verdict."),
               'the_two_texts_are_graded_apart': (
                   'per_flag_text holds each text\'s own numbers; a record '
                   'with the other text is dropped, not counted clean'),
               'this_is_one_kind_of_noise': (
                   "Owner's ruling, 2026-09-17: the flag is outlier noise. A "
                   "detector that wins here has not been shown to win on any "
                   "other kind."),
               'what_is_not_claimed': (
                   'an unflagged record is not a certified correct one')}}
    per_space, table = {}, {}
    for sp in spaces:
        per_key = {}
        for key, sus in scores[sp].items():
            row = {'records': n22.grade(flag, sus, label='records')}
            sus_mol = n22._by_molecule(sus, molecule_of_row, n_mol, 'max')
            row['molecules'] = n22.grade(flag_mol, sus_mol, label='molecules')
            row['records_per_fold_average_precision'] = fold_spread([
                (n22.grade(flag[vi], sus[vi], label='records') or {}).get(
                    'detection_average_precision') for vi in fold_rows])
            per_text = {}
            for t, c in texts.items():
                this = np.array([ft == t for ft in flag_text], dtype=bool)
                keep = ~(flag & ~this)
                g = n22.grade(this[keep], sus[keep], label='records')
                if g is not None:
                    g['flag_text'] = t
                    g['n_records_with_another_flag_text_dropped'] = int(
                        (~keep).sum())
                    if c < 30:
                        g['too_few_to_read'] = (
                            f'{c} records carry this text; a number over {c} '
                            f'records is not worth reading on its own')
                per_text[t] = g
            row['per_flag_text'] = per_text
            per_key[key] = row
            table.setdefault(sp, {})[key] = (row['records'] or {}).get(
                'detection_average_precision')
        per_space[sp] = per_key
    res['per_space'] = convert_numpy(per_space)
    res['hybrid_minus_single'] = convert_numpy(
        _hybrid_minus_single(spaces, _Lookup(table)))
    res['hybrid_minus_single_note'] = (
        'average precision against the curator flag over records, in the '
        'hybrid and in the single representation, both printed beside the '
        'difference')
    return res


def main():
    import argparse
    from experiments.noise.noise_helpers import run_all
    ap = argparse.ArgumentParser(description='N45 on one dataset through '
                                             'noise_helpers.run_all')
    ap.add_argument('dataset')
    ap.add_argument('--output', required=True)
    ap.add_argument('--noise-condition', default=None)
    args = ap.parse_args()
    run_all(dataset_name=args.dataset, output_dir=args.output,
            experiments=[RUN_ID], noise_condition=args.noise_condition)


if __name__ == '__main__':
    main()
