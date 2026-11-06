#!/usr/bin/env python3
"""N44, detect and clean: every detector x every cleanup it can pair with,
scored as D against D'.

HANDOFF_NOISE_RESTRUCTURE.md, section 2 (the detector-by-cleanup experiment,
working name DC) and section 9. Written 2026-10-30. Named by the author
2026-10-31: the restructure's five experiments are n43 to n47 and replace the
old n-files, one copy of which is archived.

WHAT ONE ROW IS. d_vs_dprime's row: the configuration's own model (one of the
dataset's two decided models) fitted on D, the training fold with the injected
labels, and on D', D after one detector's scores went through one cleanup,
both scored on the same held-out fold in the dataset's own metric, beside the
fit on the clean training labels. `method` is "<detector>/<cleanup>";
`method_setting` carries the detector space (dspace=own|hybrid), the cleanup's
own setting (flag share, strength, ...) and, for a neighbour detector or
cleanup, the kept neighbour setting.

THE THREE CHAINS (author's decision 2026-10-30, handoff section 8). For each
model and fold:
  1. model on a base representation, detector in that same representation:
     the reference.
  2. model on the base representation, detector in the hybrid: against 1,
     does the hybrid make a better detector.
  3. model on the hybrid, detector in the hybrid: the best achievable;
     against 1, does the hybrid improve prediction under noise, cleaning
     included.
A cleanup hands back labels, weights or a keep mask over the training
molecules, none of which carries features, so a cleaned set made in the hybrid
space trains the base-representation model as it is (_transfer). The base
representation is the model's best single representation from the per-fold
search (base_1; Adelaide, 2026-11-04: "base_1 and hybrid only"); the hybrid is
that search's first general
hybrid, rebuilt from the damaged labels (noise_helpers.fold_model_inputs).
hybrid__general2 is not fitted here: the decision names one hybrid.

THE LOOPS, outermost first: condition (noise_helpers.sweep_conditions), fold
(noise_outer_folds, all 5: author 2026-10-30), level, draw (3 per non-zero
level, 1 under censoring, level 0 once), model, model input (base_1, hybrid),
detector space (own; hybrid for a base input), detector, cleanup,
cleanup setting, neighbour setting where read.

THE DETECTORS (list B, handoff 4.2). One score per training molecule, higher
= more suspect, or a flag set. Each is computed once per (features, labels,
groups) through the noise cache and shared by every cleanup.
  registry      every name in kirby.utils.noise_estimation.ESTIMATOR_REGISTRY
                with its aliases dropped: the neighbour family (scored under
                each kept neighbour setting), the model family, the spread
                family, the in-loop network tests (net_*), and the three moved
                from noise_mitigation_methods.py.
  residual      n32: oof_residual (|y - p|, p the out-of-fold LightGBM
                prediction), ngboost_rescaled, quantile_forest_rescaled,
                conformal_rescaled, knn_shapley, cleanlab.
  n41           isolation_forest and lof on [features, residual],
                residual_low_disagreement; conformal flags with a false-flag
                rate (random calibration shares x q), a FLAG detector.
  controls      random (uniform scores, one draw per unit) and, where the
                corrupted subset is defined, oracle@precision=<p>, a flag set
                whose share of true hits is set (n31.synthetic_flags).
Not here and why: close-pair disagreement (n35) weights pairs, not molecules,
and stays its own family in a later commit; the curator and database flags
are grading inputs (DG).

THE CLEANUPS (list C, handoff 4.3). Each reads scores or flags and returns the
dict n07.cleaned_training_set returns.
  n07           remove, downweight, huber_weight, js_shrink, label_smooth,
                knn_smooth, prox_smooth, oof_smooth, gp_smooth, and the two
                score-free ones, invvar_weight and constant_lift (once, no
                detector). quantile_clip stays cut (f36, 2026-10-05).
  split         score_weight, spread_invvar_weight, network_rewrite,
                ema_rewrite (n07.SPLIT_CLEANUPS, kirby.utils.lnl_parts).
  n31           rewrite (flagged labels replaced by p), top-share downweight
                (flagged weight DOWNWEIGHT_TO), bayes_shrink (flagged labels
                shrunk toward p by the weight estimated from the residuals).
  n32           shrink_flagged, one weight and per row.
  n30 / n41     winsorise (every residual clipped at c scales, no detector)
                and winsorise_flagged (the flagged ones only).
  n11 / n02     exponential down-weight (weights_from_scores 'exponential')
                at n11's strengths.
  n23           move_confidence: every label toward p by the 'confidence'
                rule of smooth_by_scores (its 'proportional' rule is
                oof_smooth).
  rounds        CUT, author's decision 2026-10-31. rounds_delete (the
                detector re-scored on the kept molecules and the flagged
                share deleted, three times; n32's trimmed refit generalised
                to any detector) is not run. Reason, from the stored n32
                rows: the trimmed refit raised the top score in no more
                cells than deleting the same number of molecules at random
                (10 to 18 of 37 against 14 to 22), and under outliers and
                shifts it sat below deleting once (4 of 6 and 2 of 6 against
                6 of 6 and 6 of 6). Multistage cleaning, the other repeating
                method, runs whole in the noise-aware experiment. The branch
                stays in apply_cleanup and runs only when named.
Pairing rules (PAIRS_WITH): a weighting cleanup needs a model whose fit takes
a weight; spread_invvar_weight needs a spread detector; score_weight needs a
score in [0, 1]; a flag detector feeds the flag-reading cleanups directly and
the score cleanups as a 0/1 score; the regression-only cleanups are not run
on a classification set; a neighbour cleanup reads the kept neighbour
settings. A cleanup that declines writes a row with score_Dprime nan, so the
cell is on disk.

CONTROLS. The random and oracle detectors above. THE FIRST RUN HAS NO PUT-BACK
AND NO TWINS (Adelaide, 2026-11-04: "First the results, then the checks for
what I need"): KIRBY_NOISE_CONTROLS is not read here and no cliff reference is
computed. THE SECOND RUN (KIRBY_N44_SECOND_RUN names a JSON file, see
second_run_spec) writes put-back and twins for the results named in that file,
to its own rows and checkpoint files. Detection AUROC of every score detector against the
corrupted subset (noise_helpers.detection_scores) is written into each unit's
checkpoint, for the grading experiment to read.

THE NOISE UNIT (n38, added 2026-11-02). hetero_seen sizes its noise by
lightgbm's out-of-fold residual on the clean training labels
(noise_helpers.oof_residual_sd) rather than by the spread of the labels: each
molecule gets a noise SD that grows with its heavy-atom count. Levels are 0,
0.1, 0.3 and 1.0 times the out-of-fold mean squared error
(noise_helpers.relative_levels). The other kinds sized this way were cut
(noise_helpers.PAPER3_CUT_CONDITIONS).

NOT YET IN THIS FILE (handoff 3.1, 3.2), each a later commit: the one-lab
label regime (n17 entities), the not-at-floor control (n30b), the removal cost
curve (n27, a measurement), n30's lifts under censoring (ST), n35's pair
weights.

RESUME. n07.RowSink skips every row already on disk; each finished unit is
in the .units checkpoint under the run's fingerprint.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments._shared import (
    get_domain_metric, get_model_factories, input_fingerprint, convert_numpy,
)
from experiments.decided_grid import decided_models
from experiments.dataset_policy import noise_groups_for
from experiments.noise.noise_helpers import (
    sweep_conditions, conditions_for_task, noise_outer_folds, fold_spread,
    fold_configurations, fold_model_inputs, configuration_record,
    decided_levels, inject_condition, dataset_knn_group, level_unit,
    heavy_atom_counts,
    draws_for, detection_scores, DETECTION_IS_DEFINED,
)
from kirby.noise_spec import GROUPED_STRATEGIES
from kirby.utils.noise_estimation import (
    ESTIMATOR_REGISTRY, IN_LOOP_DETECTORS, weights_from_scores, smooth_by_scores,
)

from experiments.noise import noise_cache as nc
from experiments.noise import noise_timing
from experiments.noise.noise_helpers import (
    PAPER3_CUT_CONDITIONS, without_paper3_cuts, runs_level0, level0_rows_path)
from experiments.noise import similarity_space as ss
from experiments.noise import n07_remediation as n07
from experiments.noise import n02_sample_weighting as n02
from experiments.noise import n11_estimation_remediation_grid as n11
from experiments.noise import n23_uncertainty_cleaning as n23
from experiments.noise import n25_cliff_exemption as n25
from experiments.noise import n30_censored_likelihood as n30
from experiments.noise import n31_regime_and_breakeven as n31
from experiments.noise import n32_iterative_and_dual_selection as n32
from experiments.noise import n40_model_side_regularisation as n40
from experiments.noise import n33_robust_training as n33
from experiments.noise import n41_conformal_fdr_detection as n41
from experiments.noise import n42_similarity_noise_sweep as n42
from experiments.noise.controls import controls_enabled

EXP_ID = 'n44_detect_and_clean'
RESULT_LAYOUT = 'three_chains_2026-10-30'

NOISE_DRAWS = (0, 1, 2)

# The model inputs fitted: the two base representations and one hybrid.
MODEL_INPUTS = ('base_1', 'hybrid')
# Which cliff definitions the second run can use today. Which one it uses is
# Adelaide's and has no default (HANDOFF_AC_STUDY.md; the cliff grid chat).
SECOND_RUN_CLIFF_DEFINITIONS = ('n25_existing',)


class _NoCliffReference(Exception):
    """The first run computes no cliff reference."""


def second_run_spec():
    """None on the first run, or the second run's JSON file:
    {"cliff_definition": "n25_existing",
     "results": [{"method": "<detector>/<cleanup>",
                  "condition": "<optional noise kind>",
                  "dataset": "<optional dataset>"}]}
    Nothing is chosen for her: an empty list or a missing cliff definition
    refuses to start."""
    path = os.environ.get('KIRBY_N44_SECOND_RUN')
    if not path:
        return None
    with open(path) as fh:
        spec = json.load(fh)
    if not spec.get('results'):
        raise ValueError(f'{path}: no results named for the second run')
    if spec.get('cliff_definition') not in SECOND_RUN_CLIFF_DEFINITIONS:
        raise ValueError(
            f"{path}: cliff_definition is {spec.get('cliff_definition')!r}; it must be "
            f"one of {SECOND_RUN_CLIFF_DEFINITIONS}. It is not chosen by default.")
    return spec


def second_run_wants(spec, dataset, condition, detector, cleanup):
    """True when this (detector, cleanup) is named for this dataset and kind."""
    method = f"{str(detector).split('@')[0]}/{cleanup}"
    for r in spec['results']:
        if (r.get('method') == method
                and r.get('condition', condition) == condition
                and r.get('dataset', dataset) == dataset):
            return True
    return False
HYBRID_INPUT = 'hybrid'

# The cleanup settings. One grid for every flag-reading cleanup (n07.FLAG_SHARES,
# n31, n32, n41 all sweep it); strengths per cleanup from n25.STRENGTHS and the
# files the cleanups come from.
FLAG_SHARES = tuple(float(s) for s in n07.FLAG_SHARES)
STRENGTHS = dict(n25.STRENGTHS)
STRENGTHS['exp_downweight'] = tuple(float(s) for s in n11.WEIGHT_STRENGTHS)
STRENGTHS['move_confidence'] = tuple(STRENGTHS.get('oof_smooth', (0.25, 0.5, 1.0)))
# n02's own strength for the neighbour-agreement weights (m27), beside n11's
# weight grid, so the strength n02 ran is in the sweep and is not the only one.
STRENGTHS['reliability_weight'] = tuple(sorted(
    {float(n02.DEFAULT_STRENGTH)} | {float(x) for x in n11.WEIGHT_STRENGTHS}))
WINSOR_C = tuple(float(c) for c in n30.WINSOR_C)
# rounds_delete: the flagged share deleted per round, ROUNDS rounds. n32's
# itlm keeps 0.8, 0.9, 0.95 (deletes 0.2, 0.1, 0.05) for 4 rounds; n09's
# multistage deletes 0.1 for 3 rounds.
ROUNDS = (3,)
# oracle@precision: n31's grid.
PRECISIONS = tuple(float(p) for p in n31.PRECISIONS)
OOF_SPLITS = int(n32.OOF_SPLITS)

# Registry aliases, so a detector is scored once under one name.
REGISTRY_ALIASES = ('distance_weighted_residual', 'ensemble_disagreement',
                    'cross_val_residual', 'hybrid_noise', 'heteroscedastic_gp')
REGISTRY_DETECTORS = tuple(k for k in ESTIMATOR_REGISTRY
                           if k not in REGISTRY_ALIASES)
NEIGHBOUR_DETECTORS = tuple(d for d in REGISTRY_DETECTORS
                            if d in nc.NEIGHBOUR_DETECTORS)
# The detectors that read the out-of-fold LightGBM prediction p.
RESIDUAL_DETECTORS = ('oof_residual', 'ngboost_rescaled',
                      'quantile_forest_rescaled', 'conformal_rescaled',
                      'knn_shapley', 'cleanlab')
N41_DETECTORS = ('isolation_forest', 'lof', 'residual_low_disagreement')
# Flag detectors: a boolean per molecule instead of a score.
CONFORMAL_FLAG_SETTINGS = tuple((float(sh), float(q))
                                for sh in n41.CALIBRATION_SHARES
                                for q in n41.BH_QS)
CONTROL_DETECTORS = ('random',)
# Co-teaching (audit m69), as n32 split it: two learners each pick the
# lowest-loss share for the OTHER to train on, for n32.EXCHANGE_ROUNDS rounds,
# and a molecule is kept only when both keep it. A FLAG detector, not a score:
# the exchange hands back a keep mask, not one number per molecule. Added
# 2026-11-02; handoff 4.1 calls it a DC cell and n43 defers it here. It runs
# only where the dataset has a second decided model to be the other learner,
# and the unit's own model is the first.
TWO_LEARNER_DETECTOR = 'two_learner_small_loss'
# Detectors whose score is a per-molecule spread (variance or SD), for the
# inverse-variance weight: n23's sources less the neighbour one, plus the
# evidential network's variance.
SPREAD_DETECTORS = tuple(s for s in n23.SOURCES
                         if s not in nc.NEIGHBOUR_DETECTORS) + (
                             'net_evidential_spread',)
# Regression-only detectors: they read a numeric residual or a spread.
REGRESSION_ONLY_DETECTORS = (RESIDUAL_DETECTORS + N41_DETECTORS
                             + ('net_evidential_spread', 'label_zscore'))

# The cleanups and what each needs.
#   reads   'score' | 'flag' | 'none'
#   shape   'delete' | 'weight' | 'label'
#   axis    'flag_share' | 'strength' | 'winsor_c' | 'rounds' | 'none'
#   reg_only, needs_p (the out-of-fold prediction), neighbours (reads a
#   neighbour list), spread (needs a spread detector), unit (score in [0, 1])
def _c(reads, shape, axis, **kw):
    d = {'reads': reads, 'shape': shape, 'axis': axis, 'reg_only': False,
         'needs_p': False, 'neighbours': False, 'spread': False,
         'unit': False}
    d.update(kw)
    return d


CLEANUPS = {
    # n07
    'remove': _c('flag', 'delete', 'flag_share'),
    'downweight': _c('score', 'weight', 'strength'),
    'huber_weight': _c('score', 'weight', 'strength'),
    'js_shrink': _c('score', 'label', 'strength', reg_only=True, needs_p=True),
    'label_smooth': _c('score', 'label', 'strength', neighbours=True),
    'knn_smooth': _c('flag', 'label', 'flag_share', neighbours=True),
    'prox_smooth': _c('score', 'label', 'strength', neighbours=True),
    'oof_smooth': _c('score', 'label', 'strength', reg_only=True, needs_p=True),
    'gp_smooth': _c('score', 'label', 'strength', reg_only=True),
    'invvar_weight': _c('none', 'weight', 'none'),
    # n02 (m27): each molecule weighted by how far its label sits from its
    # neighbours' mean label, in label SDs, under the setting's counting
    # scheme. It computes that agreement itself from the neighbour list, so it
    # reads no detector's score, but it does need a neighbour list.
    'reliability_weight': _c('none', 'weight', 'strength', neighbours=True),
    'constant_lift': _c('none', 'label', 'none', reg_only=True, needs_p=True),
    # split (kirby.utils.lnl_parts)
    'score_weight': _c('score', 'weight', 'none', unit=True),
    'spread_invvar_weight': _c('score', 'weight', 'none', spread=True),
    'network_rewrite': _c('flag', 'label', 'flag_share'),
    'ema_rewrite': _c('flag', 'label', 'flag_share'),
    # n31
    'rewrite': _c('flag', 'label', 'flag_share', reg_only=True, needs_p=True),
    'top_share_downweight': _c('flag', 'weight', 'flag_share'),
    'bayes_shrink': _c('flag', 'label', 'flag_share', reg_only=True, needs_p=True),
    # n32
    'shrink_flagged_one_weight': _c('flag', 'label', 'flag_share', reg_only=True, needs_p=True),
    'shrink_flagged_per_row': _c('flag', 'label', 'flag_share', reg_only=True, needs_p=True),
    # n30 / n41
    'winsorise': _c('none', 'label', 'winsor_c', reg_only=True, needs_p=True),
    'winsorise_flagged': _c('flag', 'label', 'flag_share', reg_only=True, needs_p=True),
    # n33 (m40): every label moved halfway toward an out-of-fold LightGBM
    # teacher fitted with a Huber loss; reads no detector.
    'huber_teacher_blend': _c('none', 'label', 'none', reg_only=True, needs_p=True),
    # n11
    'exp_downweight': _c('score', 'weight', 'strength'),
    # n23
    'move_confidence': _c('score', 'label', 'strength', reg_only=True, needs_p=True),
}
# Cut 2026-10-31 (author): see the docstring. Named here so a caller can
# still ask for it; cleanups_for never returns it.
CUT_CLEANUPS = {'rounds_delete': _c('flag', 'delete', 'rounds')}
WEIGHT_CLEANUPS = tuple(c for c, d in CLEANUPS.items() if d['shape'] == 'weight')
SCORE_FREE_CLEANUPS = tuple(c for c, d in CLEANUPS.items() if d['reads'] == 'none')
NEIGHBOUR_CLEANUPS = tuple(c for c, d in CLEANUPS.items() if d['neighbours'])
CLF_NOT_RUN_CLEANUPS = tuple(n42.CLF_NOT_RUN_CLEANUPS)


# ---------------------------------------------------------------------------
# cleanup settings
# ---------------------------------------------------------------------------

def cleanup_settings(cleanup):
    """[(label, kwargs)] for one cleanup's own axis."""
    ax = CLEANUPS[cleanup]['axis']
    if ax == 'flag_share':
        return [(f'flag_share={s:g}', {'share': s}) for s in FLAG_SHARES]
    if ax == 'strength':
        vals = STRENGTHS.get(cleanup)
        if not vals:
            return [('default', {})]
        return [(f'strength={float(v):g}', {'strength': float(v)})
                for v in vals]
    if ax == 'winsor_c':
        return [(f'c={c:g}', {'c': c}) for c in WINSOR_C]
    if ax == 'rounds':
        return [(f'flag_share={s:g}|rounds={r}', {'share': s, 'rounds': r})
                for s in FLAG_SHARES for r in ROUNDS]
    return [('default', {})]


def cleanup_settings_of(cleanup):
    d = CLEANUPS.get(cleanup) or CUT_CLEANUPS[cleanup]
    return cleanup_settings(cleanup) if cleanup in CLEANUPS else [
        (f'flag_share={s:g}|rounds={r}', {'share': s, 'rounds': r})
        for s in FLAG_SHARES for r in ROUNDS]


def cleanups_for(is_clf, takes_weight):
    out = []
    for c, d in CLEANUPS.items():
        if is_clf and (d['reg_only'] or c in CLF_NOT_RUN_CLEANUPS):
            continue
        if d['shape'] == 'weight' and not takes_weight:
            continue
        out.append(c)
    return tuple(out)


def flags_from(scores, share):
    """The top `share` of molecules by score, as a flag set."""
    s = np.asarray(scores, dtype=float)
    k = int(round(float(share) * s.size))
    if k <= 0:
        return np.zeros(s.size, dtype=bool)
    return n31.detector_flags(s, k)


# ---------------------------------------------------------------------------
# the detectors, each a function of (unit, space) -> scores or flags
# ---------------------------------------------------------------------------

class _Space:
    """One feature space a detector runs in: the matrices and the
    out-of-fold prediction p (computed once, cached)."""

    def __init__(self, name, X_tr, X_va, u):
        self.name = name
        self.X_tr = X_tr
        self.X_va = X_va
        self.u = u
        self._p = None
        self._p_done = False

    def p(self):
        if not self._p_done:
            self._p_done = True
            try:
                self._p = nc.cached_oof_predict(
                    self.u.ds, self.X_tr, self.u.y0, self.u.g_tr,
                    n_splits=min(OOF_SPLITS, len(self.u.y0)))
            except Exception as exc:
                self.u.errors.add(f'oof_prediction|{self.name}', exc)
        return self._p


def _cached_scores(sp, name, compute, extra=None):
    key = {'detector': name, 'X': nc.array_hash(sp.X_tr),
           'y': nc.array_hash(sp.u.y0), 'groups': nc.array_hash(sp.u.g_tr),
           'is_clf': bool(sp.u.is_clf)}
    if extra:
        key.update(extra)
    return nc.cached(sp.u.ds, 'n44_detector', key,
                     lambda: np.asarray(compute(), dtype=float),
                     store_if=lambda v: v is not None)


def detector_scores(sp, name):
    """Scores of one non-neighbour, non-flag detector in space `sp`, or None."""
    u = sp.u
    y0 = u.y0
    try:
        if name in REGISTRY_DETECTORS:
            return nc.cached_detector_scores(u.ds, sp.X_tr, y0, is_clf=u.is_clf,
                                             detector=name, groups=u.g_tr)
        if name in RESIDUAL_DETECTORS:
            p = sp.p()
            if p is None:
                return None
            n32_name = 'conformal' if name == 'conformal_rescaled' else name
            return _cached_scores(sp, name, lambda: n32.detector_scores(
                n32_name, sp.X_tr, np.asarray(y0, dtype=float), u.g_tr, p,
                OOF_SPLITS), {'p': nc.array_hash(p)})
        if name in N41_DETECTORS:
            p = sp.p()
            if p is None:
                return None
            return _cached_scores(sp, name, lambda: n41.detector_scores(
                name, sp.X_tr, np.asarray(y0, dtype=float), u.g_tr, p, None,
                OOF_SPLITS, dataset_name=u.ds), {'p': nc.array_hash(p)})
        if name == 'random':
            rng = np.random.default_rng(n32.replicate_seed(
                'n44_random', u.condition, f'{float(u.level):g}', u.draw,
                u.fold, u.mi, sp.name))
            return rng.uniform(size=len(y0))
    except Exception as exc:
        u.errors.add(f'detector={name}|{sp.name}', exc)
        return None
    raise ValueError(f'n44: unknown detector {name!r}')


def flag_detectors(sp):
    """[(name, flags)] for the flag detectors in space `sp`."""
    u = sp.u
    out = []
    if not u.is_clf:
        p = sp.p()
        if p is not None:
            y0 = np.asarray(u.y0, dtype=float)
            n = len(y0)
            try:
                score, _sigma = n41.rescaled_residual(
                    sp.X_tr, y0, u.g_tr, p, OOF_SPLITS, dataset_name=u.ds)
            except Exception as exc:
                u.errors.add(f'detector=conformal_bh|{sp.name}', exc)
                score = None
            if score is not None:
                for sh, q in CONFORMAL_FLAG_SETTINGS:
                    cal = n41.random_calibration(n, sh, n32.replicate_seed(
                        'cal', f'{float(u.level):g}', u.draw, u.fold, n, sh))
                    if int(cal.sum()) < n41.MIN_CALIBRATION or not (~cal).any():
                        continue
                    pv = n41.conformal_pvalues(score[cal], score[~cal])
                    flags = np.zeros(n, dtype=bool)
                    flags[np.flatnonzero(~cal)[n41.benjamini_hochberg(pv, q)]] = True
                    out.append((f'conformal_bh@cal={n41.random_cal_name(sh)}|q={q:g}',
                                flags))
    # co-teaching: the two-learner exchange at each kept share (audit m69)
    lf = getattr(u, 'learner_factories', None) or {}
    other = [m for m in lf if m != u.model]
    if not u.is_clf and other:
        for share in n32.ITLM_ALPHAS:
            try:
                keep, _rec, _loss = n32.two_learner_exchange(
                    sp.X_tr, np.asarray(u.y0, dtype=float), u.g_tr,
                    lf[u.model], lf[other[0]], float(share))
            except Exception as exc:
                u.errors.add(f'detector={TWO_LEARNER_DETECTOR}|{sp.name}', exc)
                continue
            u.counts['two_learner_rounds'] = int(n32.EXCHANGE_ROUNDS)
            out.append((f'{TWO_LEARNER_DETECTOR}@other={other[0]}'
                        f'|share_kept={float(share):g}',
                        ~np.asarray(keep, dtype=bool)))
    elif not u.is_clf:
        u.unavailable.setdefault(
            TWO_LEARNER_DETECTOR,
            'the exchange needs a second learner and this dataset has one '
            'decided model')
    return out


def oracle_flags(u):
    """[(name, flags)]: flag sets built from the injected noise at a set
    precision (n31.synthetic_flags), where the corrupted subset is defined.
    They read no features, so they run in one detector space, in their own
    pass before every detector (Adelaide, 2026-10-31, handoff section 9.6)."""
    out = []
    if u.cmask is not None and u.condition in DETECTION_IS_DEFINED \
            and float(u.level) > 0 and u.cmask.any():
        for share in FLAG_SHARES:
            k = int(round(share * len(u.y0)))
            if k <= 0:
                continue
            for prec in PRECISIONS:
                flags, got = n31.synthetic_flags(
                    u.cmask, prec, n32.replicate_seed(
                        'n44_oracle', u.condition, f'{float(u.level):g}',
                        u.draw, u.fold, share, prec), k)
                if flags.any():
                    out.append((f'oracle@precision={prec:g}|flag_share={share:g}',
                                flags))
    return out


# ---------------------------------------------------------------------------
# the cleanups, each (space, scores, flags, setting) -> cleaned dict or None
# ---------------------------------------------------------------------------

def _transfer(cleaned, X_model):
    """The cleaned set re-expressed on the model's own features: the keep
    mask, labels and weights carry over; the features do not."""
    if cleaned is None:
        return None
    keep = cleaned.get('keep')
    if keep is not None:
        return {'X': X_model[keep], 'y': cleaned['y'], 'keep': keep,
                'sample_weight': None, 'label_column': None}
    if len(cleaned['y']) != X_model.shape[0]:
        return None
    return {'X': X_model, 'y': cleaned['y'], 'keep': None,
            'sample_weight': cleaned.get('sample_weight'),
            'label_column': cleaned.get('label_column')}


def apply_cleanup(cleanup, sp, scores, flags, kw, *, lists=None, k=None,
                  targets=None, refit=None):
    """One cleanup on one detector's output. `refit(keep) -> scores` re-runs
    the detector on the kept molecules (rounds_delete, cut; runs when named)."""
    u = sp.u
    X, y0 = sp.X_tr, u.y0
    d = CLEANUPS.get(cleanup) or CUT_CLEANUPS[cleanup]
    share = kw.get('share')
    strength = kw.get('strength')
    if d['reads'] == 'flag' and flags is None:
        flags = flags_from(scores, share)
    if d['reads'] == 'score' and scores is None:
        scores = np.asarray(flags, dtype=float)
    if d['needs_p'] and cleanup not in ('winsorise',):
        p = sp.p()
        if p is None:
            return None

    # n07's own branches, including the split cleanups
    if cleanup in n07.CLEANUPS or cleanup in n07.SPLIT_CLEANUPS:
        nbkw = {} if lists is None else {'similarity_neighbours': lists,
                                         'similarity_k': k}
        if cleanup in ('network_rewrite', 'ema_rewrite', 'knn_smooth', 'remove'):
            # n07 thresholds the score itself; a flag detector hands a 0/1
            # score whose top share is exactly its flags
            sc = (np.asarray(flags, dtype=float) if scores is None
                  else np.asarray(scores, dtype=float))
            fs = (float(np.mean(flags)) if scores is None and flags is not None
                  else share)
        else:
            sc, fs = scores, (share if share is not None else n42.UNREAD_FLAG_RATE)
        return n07.cleaned_training_set(
            cleanup, X, y0, sc, fs if fs is not None else n42.UNREAD_FLAG_RATE,
            is_clf=u.is_clf, groups=u.g_tr, strength=strength, targets=targets,
            measurement_counts=u.measurement_counts, dataset_name=u.ds, **nbkw)

    y = np.asarray(y0, dtype=float)
    if cleanup in ('rewrite', 'top_share_downweight', 'bayes_shrink'):
        op = 'downweight' if cleanup == 'top_share_downweight' else cleanup
        lab, keep, w = n31.apply_operation(op, y, flags, sp.p() if op != 'downweight' else None)
        if keep is not None:
            return n42._keep_dict(X, y, keep)
        if w is not None:
            return n07.cleaned_from_weights(X, y, w)
        return n07.cleaned_from_labels(X, lab)
    if cleanup in ('shrink_flagged_one_weight', 'shrink_flagged_per_row'):
        one, per, _rec = n32.shrink_flagged(y, sp.p(), float(np.mean(flags)))
        col = one if cleanup.endswith('one_weight') else per
        return None if col is None else n07.cleaned_from_labels(X, col)
    if cleanup == 'winsorise':
        p = sp.p()
        if p is None:
            return None
        col, _s, _moved = n30.winsorise(y, p, float(kw['c']))
        return n07.cleaned_from_labels(X, col)
    if cleanup == 'reliability_weight':
        if lists is None or k is None:
            return None
        idx, dist = ss.neighbours_at(lists, k)
        w = n02.reliability_from_neighbours(
            y, idx, is_clf=u.is_clf,
            strength=float(strength) if strength is not None else None,
            dist=dist, scheme=lists.get('scheme'))
        return n07.cleaned_from_weights(X, y, np.asarray(w, dtype=float))
    if cleanup == 'huber_teacher_blend':
        # n33's arm as n33 builds it: threshold TEACHER_HUBER_C x s, with
        # s = 1.4826 x median |y - p|; label (1 - w) y + w teacher.
        s_rob = n33.robust_scale(y, sp.p())
        if not (np.isfinite(s_rob) and s_rob > 0):
            return None
        teacher = nc.cached_oof_predict(
            u.ds, X, y, u.g_tr, n_splits=min(OOF_SPLITS, len(y)),
            params={'objective': 'huber', 'alpha': n33.TEACHER_HUBER_C * s_rob})
        w = float(n33.TEACHER_WEIGHT)
        return n07.cleaned_from_labels(X, (1.0 - w) * y + w * np.asarray(teacher, dtype=float))
    if cleanup == 'winsorise_flagged':
        return n07.cleaned_from_labels(X, n41.winsorise_flagged(y, sp.p(), flags))
    if cleanup == 'exp_downweight':
        w = weights_from_scores(np.asarray(scores, dtype=float),
                                strength=float(strength), method='exponential')
        return n07.cleaned_from_weights(X, y, w)
    if cleanup == 'move_confidence':
        col = smooth_by_scores(y, np.asarray(scores, dtype=float), sp.p(),
                               alpha=float(strength), method='confidence')
        return n07.cleaned_from_labels(X, col)
    if cleanup == 'rounds_delete':
        keep = np.ones(len(y0), dtype=bool)
        sc = np.asarray(scores if scores is not None else flags, dtype=float)
        for r in range(int(kw['rounds'])):
            if r > 0:
                if refit is None:
                    return None
                sc_kept = refit(keep)
                if sc_kept is None:
                    return None
                sc = np.full(len(y0), -np.inf)
                sc[keep] = np.asarray(sc_kept, dtype=float)
            fl = flags_from(np.where(keep, sc, -np.inf), share)
            keep = keep & ~fl
            if keep.sum() < 10:
                return None
        return n42._keep_dict(X, y0, keep)
    raise ValueError(f'n44: unknown cleanup {cleanup!r}')


# ---------------------------------------------------------------------------
# one unit
# ---------------------------------------------------------------------------

# Detectors whose score is in [0, 1] by construction, for score_weight
# (one minus the score as the weight).
UNIT_DETECTORS = ('net_meta_weight', 'net_josrc_test', 'net_loss_mixture',
                  'net_trusted_weight', 'cleanlab', 'random')


def _pairs_with(cleanup, detector, is_flag):
    """Does this cleanup read this detector's output? (PAIRS_WITH rules.)"""
    d = CLEANUPS[cleanup]
    if d['reads'] == 'none':
        return False
    if d['spread']:
        return detector in SPREAD_DETECTORS and not is_flag
    if d['unit']:
        return is_flag or detector in UNIT_DETECTORS
    if cleanup == 'rounds_delete':
        return not is_flag and detector != 'random'
    return True


def run_unit(u, X_model, X_model_va, spaces, sim, targets_by_space):
    """Every detector x every cleanup for one (fold, level, draw, model,
    model input), in each detector space."""
    cleanups = cleanups_for(u.is_clf, u.takes_weight)
    is_hyb_input = u.mi.endswith(':' + HYBRID_INPUT)

    spec = getattr(u, 'second_run', None)

    def wanted(detector, cleanup):
        return spec is None or second_run_wants(spec, u.ds, u.condition,
                                                detector, cleanup)

    def emit(detector, cleanup, setting, sim_name, cleaned, dspace):
        if not wanted(detector, cleanup):
            return
        ms = f'dspace={dspace}|{setting}'
        u.emit('n44', f'{detector}/{cleanup}', ms, sim_name,
               _transfer(cleaned, X_model), exempt=u.cliff_train is not None)

    def clean(c, det, dspace, setting, *args, **kw):
        # one cleaned training set, timed; None is a refusal
        if not wanted(det, c):
            return None
        with noise_timing.timed('cleanup', detector=det, cleanup=c,
                                dspace=dspace, setting=setting) as step:
            cl = apply_cleanup(c, *args, **kw)
            if cl is None:
                step.status = 'refused'
        return cl

    # score-free cleanups: once per unit, in the model's own space
    own = spaces['own']
    tg_own = targets_by_space.get('own')
    for c in cleanups:
        # a score-free cleanup that reads neighbours runs per neighbour setting
        if CLEANUPS[c]['reads'] != 'none' or CLEANUPS[c]['neighbours']:
            continue
        for lab, kw in cleanup_settings(c):
            try:
                cl = clean(c, 'none', 'own', lab, own, None, None, kw,
                           targets=tg_own)
            except Exception as exc:
                u.errors.add(f'none/{c}', exc)
                cl = None
            emit('none', c, lab, n07.NO_NEIGHBOURS, cl, 'own')

    def clean_with(scored, sp, tg, dspace):
        for det, sc, fl in scored:
            is_flag = sc is None
            base_det = det.split('@')[0]

            def _refit(keep, _det=det, _sp=sp):
                # the detector re-scored on the kept molecules
                sub = _Space(_sp.name, _sp.X_tr[keep], _sp.X_va, _SubUnit(u, keep))
                return detector_scores(sub, _det)

            for c in cleanups:
                d = CLEANUPS[c]
                if d['reads'] == 'none' or d['neighbours']:
                    continue
                if not _pairs_with(c, base_det, is_flag):
                    continue
                for lab, kw in cleanup_settings(c):
                    if is_flag and d['reads'] == 'flag' and d['axis'] == 'flag_share':
                        # a flag detector sets its own share
                        if kw.get('share') != FLAG_SHARES[0]:
                            continue
                        lab = 'flags=detector'
                    try:
                        cl = clean(c, det, dspace, lab, sp, sc, fl, kw, targets=tg,
                                   refit=_refit if c == 'rounds_delete' else None)
                    except Exception as exc:
                        u.errors.add(f'{det}/{c}', exc)
                        cl = None
                    emit(det, c, lab, n07.NO_NEIGHBOURS, cl, dspace)

    # First pass: the oracle flags, once, in the model's own space. A cleanup
    # that needs a prediction or neighbours computes it there.
    clean_with([(name, None, fl) for name, fl in oracle_flags(u)],
               own, tg_own, 'own')

    for dspace, sp in spaces.items():
        tg = targets_by_space.get(dspace)
        dets = [d for d in REGISTRY_DETECTORS + RESIDUAL_DETECTORS
                + N41_DETECTORS + CONTROL_DETECTORS
                if d not in NEIGHBOUR_DETECTORS
                and not (u.is_clf and d in REGRESSION_ONLY_DETECTORS)]
        scored = []
        for det in dets:
            with noise_timing.timed('detector', detector=det,
                                    dspace=dspace) as step:
                sc = detector_scores(sp, det)
                if sc is None:
                    step.status = 'refused'
            if sc is None:
                u.counts['detector_none'] = u.counts.get('detector_none', 0) + 1
                continue
            sc = np.asarray(sc, dtype=float)
            if not np.all(np.isfinite(sc)):
                sc = np.where(np.isfinite(sc), sc, np.nanmax(sc[np.isfinite(sc)]) if np.isfinite(sc).any() else 0.0)
            scored.append((det, sc, None))
            if u.cmask is not None and float(u.level) > 0:
                g = detection_scores(u.cmask, sc, u.condition)
                if g is not None:
                    u.detection.setdefault(dspace, {})[det] = g
        with noise_timing.timed('detector', detector='flag_detectors',
                                dspace=dspace):
            flagged = flag_detectors(sp)
        for name, fl in flagged:
            scored.append((name, None, fl))
        clean_with(scored, sp, tg, dspace)

        # neighbour detectors and neighbour cleanups, under the kept settings
        if sim is None or dspace != 'own':
            # the neighbour lists are built in the model input's spaces
            # (n42._Similarity); a hybrid detector space reads the hybrid
            # through its own 'hybrid' setting in those lists
            continue
        sk_done = set()
        for s, k, scheme in sim.kept:
            L = sim.nb.get(s.name)
            if L is None:
                continue
            Ls = sim.with_scheme(s, scheme)
            first_sk = (s.name, k) not in sk_done
            sk_done.add((s.name, k))
            # A neighbour cleanup that reads no score gives the same training
            # set under every detector, so it runs once per neighbour setting.
            # Adelaide, 2026-11-04: "the same thing doesn't get run multiple
            # times". This was reliability_weight, once per scored detector.
            for c in cleanups:
                if not (CLEANUPS[c]['neighbours'] and CLEANUPS[c]['reads'] == 'none'):
                    continue
                for lab, kw in cleanup_settings(c):
                    try:
                        cl = clean(c, 'none', dspace, f'{s.name}|k={k}|{scheme}|{lab}',
                                   sp, None, None, kw, lists=Ls, k=k, targets=tg)
                    except Exception as exc:
                        u.errors.add(f'none/{c}|{s.name}', exc)
                        cl = None
                    emit('none', c, f'k={k}|scheme={scheme}|{lab}', s.name, cl, dspace)
            # non-neighbour detectors' scores through the neighbour cleanups
            for det, sc, fl in scored:
                if sc is None:
                    continue
                for c in cleanups:
                    if not CLEANUPS[c]['neighbours'] or CLEANUPS[c]['reads'] == 'none':
                        continue
                    if c == 'prox_smooth' and s.space != ss.FOREST_SPACE:
                        continue
                    for lab, kw in cleanup_settings(c):
                        try:
                            cl = clean(c, det, dspace, f'{s.name}|k={k}|{scheme}|{lab}',
                                       sp, sc, None, kw, lists=Ls, k=k, targets=tg)
                        except Exception as exc:
                            u.errors.add(f'{det}/{c}|{s.name}', exc)
                            cl = None
                        emit(det, c, f'k={k}|scheme={scheme}|{lab}', s.name,
                             cl, dspace)
            # the neighbour detectors scored in this setting
            for det in NEIGHBOUR_DETECTORS:
                try:
                    with noise_timing.timed('detector', detector=det, dspace=dspace,
                                            similarity=s.name, k=k) as step:
                        sc = nc.cached_detector_scores(
                            u.ds, sp.X_tr, u.y0, is_clf=u.is_clf, detector=det,
                            groups=u.g_tr, similarity_neighbours=L, similarity_k=k)
                        if sc is None:
                            step.status = 'refused'
                except Exception as exc:
                    u.errors.add(f'detector={det}|{s.name}', exc)
                    sc = None
                if sc is None:
                    continue
                sc = np.asarray(sc, dtype=float)
                if u.cmask is not None and float(u.level) > 0 and first_sk:
                    g = detection_scores(u.cmask, sc, u.condition)
                    if g is not None:
                        u.detection.setdefault(dspace, {})[f'{det}|{s.name}|k={k}'] = g
                for c in cleanups:
                    d = CLEANUPS[c]
                    if d['reads'] == 'none':
                        continue
                    reads = d['neighbours']
                    if reads and c == 'prox_smooth' and s.space != ss.FOREST_SPACE:
                        continue
                    if not reads and not first_sk:
                        continue
                    if not _pairs_with(c, det, False) or c == 'rounds_delete':
                        continue
                    for lab, kw in cleanup_settings(c):
                        try:
                            cl = clean(c, det, dspace, f'{s.name}|k={k}|{scheme}|{lab}',
                                       sp, sc, None, kw,
                                       lists=Ls if reads else None,
                                       k=k if reads else None, targets=tg)
                        except Exception as exc:
                            u.errors.add(f'{det}/{c}|{s.name}', exc)
                            cl = None
                        ms = (f'k={k}|scheme={scheme}|{lab}' if reads
                              else f'k={k}|{lab}')
                        emit(det, c, ms, s.name, cl, dspace)


class _SubUnit:
    """The unit restricted to the kept molecules, for a detector refit."""

    def __init__(self, u, keep):
        self.ds, self.is_clf, self.errors = u.ds, u.is_clf, u.errors
        self.condition, self.level, self.draw, self.fold, self.mi = (
            u.condition, u.level, u.draw, u.fold, u.mi)
        self.y0 = np.asarray(u.y0)[keep]
        self.g_tr = None if u.g_tr is None else np.asarray(u.g_tr)[keep]


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

# The kind whose level is a share of the model's own out-of-fold error, not of
# the label spread (see THE NOISE UNIT above). Adelaide, 2026-11-04: "Keep
# hetero_seen"; the other two were cut.
MODEL_ERROR_CONDITIONS = ('hetero_seen',)


def default_conditions(data):
    """Every injected kind this dataset can be given: the decided ones left
    after the 2026-11-04 cuts, and on a regression set hetero_seen."""
    conds = without_paper3_cuts(conditions_for_task(data))
    if data.get('task_type') != 'classification':
        conds += [c for c in MODEL_ERROR_CONDITIONS if c not in conds]
    return conds


def _condition_levels(data, dataset_name, condition):
    """The levels one kind is swept at here. A kind sized by the model's own
    error reads n40's rule, which is noise_helpers.relative_levels with the
    large-dataset grid for dense_oofmse; every other kind reads the decided
    noise grid."""
    if condition in MODEL_ERROR_CONDITIONS:
        return tuple(n40._condition_levels(data, dataset_name, condition))
    return tuple(decided_levels(data, dataset_name, condition))


def run(data, config, output_dir, dataset_name, **kwargs):
    """DC across every injected condition this dataset can be given."""
    return sweep_conditions(
        _run_one_condition, EXP_ID, data, config, output_dir, dataset_name,
        default_conditions=default_conditions(data), kwargs=kwargs,
        groups_needed=GROUPED_STRATEGIES, cut_conditions=PAPER3_CUT_CONDITIONS)


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    print("\n" + "=" * 70)
    print("EXPERIMENT N44, DETECT AND CLEAN: EVERY DETECTOR x EVERY CLEANUP, THREE CHAINS")
    print("=" * 70)
    t_start = time.time()

    task_type = data['task_type']
    is_clf = task_type == 'classification'
    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    augmentations = data.get('augmentations')
    metric = get_domain_metric(data)
    metric_name, _metric_fn, lower_is_better = metric
    condition = kwargs.get('noise_condition') or default_conditions(data)[0]
    levels = tuple(float(x) for x in _condition_levels(data, dataset_name,
                                                       condition))
    knn_grp = dataset_knn_group(data, dataset_name)
    # hetero_seen's noise SD follows each molecule's heavy-atom count, so that
    # count has to reach the injector; every other kind ignores it. A dataset
    # with no SMILES has no count, and the kind is skipped with the reason
    # rather than raising inside every fold (as n38 did).
    heavy_atoms = None
    if condition == 'hetero_seen':
        if data.get('smiles') is None:
            print(f"  SKIPPED: {dataset_name} loads no SMILES, so there is no "
                  f"heavy-atom count for hetero_seen to follow.")
            return {'skipped': 'no_smiles', 'noise_condition': condition}
        heavy_atoms = heavy_atom_counts(data['smiles'])
    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    n_molecules = len(set(smiles)) if has_smiles else len(labels)

    decided = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    asked = kwargs.get('models')
    models = [m for m in decided if m in asked] if asked else list(decided)
    factories = get_model_factories(models, task_type=task_type,
                                    n_samples=len(labels))
    models = [m for m in models if m in factories]
    takes_weight = {m: bool(nc.fit_takes_sample_weight(factories[m]))
                    for m in models}

    folds, n_folds = noise_outer_folds(data, config)
    settings = n07.sweep_settings(dataset_name, n_molecules, has_smiles)
    ks = n07.sweep_ks()
    schemes = n07.sweep_schemes()
    draws = tuple(int(d) for d in draws_for(condition, NOISE_DRAWS))
    passes = [(lv, int(dr)) for lv in levels if lv != 0.0 for dr in draws]
    # Level 0 runs once, in level0_owner's file; every kind's comparison reads it
    # there (noise_helpers.level0_rows_path).
    passes += [(lv, int(NOISE_DRAWS[0])) for lv in levels
               if lv == 0.0 and runs_level0(data, condition)]
    noise_groups = noise_groups_for(data)[0]
    try:
        mcounts = n07.measurement_counts_for_rows(data)
    except Exception:
        mcounts = None
    cliff_structure = cliff_structure_info = cliff_error = None
    spec = second_run_spec()
    if spec is None:
        cliff_error = 'first run: no put-back (Adelaide, 2026-11-04)'
    try:
        if spec is None:
            raise _NoCliffReference
        _m, _info = n25.cliff_mask_for_dataset(data, labels, is_clf)
        cliff_structure = _info.pop('similarity_matrix', None)
        # How that matrix was cut, so every per-fold call cuts it the same way
        # (2026-11-04). Without it n25 falls back to the top
        # n01.SIMILAR_PERCENTILE of pairs, 2.0, while the cache key records
        # 0.55: two cutoffs, one of them written down. See n42._cliff_masks.
        cliff_structure_info = {
            'structure_reference': _info.get('structure_reference'),
            'similar_percentile': _info.get('similar_percentile'),
            'reference_independent': _info.get('reference_independent', True)}
    except _NoCliffReference:
        pass
    except Exception as exc:
        cliff_error = f'{type(exc).__name__}: {exc}'

    file_cond = f'{condition}__{RESULT_LAYOUT}'
    if spec is not None:
        # its own rows and checkpoint: the first run's stay as they are
        file_cond += '__second_run'
    sink = n07.RowSink(output_dir, EXP_ID, dataset_name, file_cond)
    fingerprint = {'layout': RESULT_LAYOUT,
                   'input': input_fingerprint(data, config),
                   'settings': [s.name for s in settings], 'ks': list(ks),
                   'schemes': list(schemes), 'screen_top': n07.screen_top(),
                   'levels': list(levels), 'draws': list(draws),
                   'n_folds': int(n_folds), 'models': list(models),
                   'model_inputs': list(MODEL_INPUTS),
                   'cleanups': sorted(CLEANUPS), 'flag_shares': list(FLAG_SHARES),
                   'second_run': spec, 'condition': condition,
                   'passes': [[float(lv), int(dr)] for lv, dr in passes]}
    units = n07.FoldCheckpoint(output_dir, EXP_ID, dataset_name, file_cond,
                               tag='units', fingerprint=fingerprint)
    fold_ckpt = n07.FoldCheckpoint(output_dir, EXP_ID, dataset_name, file_cond,
                                   tag='folds', fingerprint=fingerprint)
    errors = n42._Errors()
    unavailable, counts, provenance, fold_records = {}, {}, {}, {}
    n_units_done = n_units_resumed = 0

    def _done(key):
        return units._k(key) in units.entries

    print(f"  condition {condition}; levels {list(levels)} "
          f"({level_unit(condition, is_clf)}); {len(passes)} (level, draw) "
          f"passes; {n_folds} outer folds; models {models}; inputs "
          f"{list(MODEL_INPUTS)}; {len(CLEANUPS)} cleanups; rows -> {sink.path}")

    for fold_i, train_idx, val_idx in folds:
        train_idx = np.asarray(train_idx)
        val_idx = np.asarray(val_idx)
        all_keys = [[int(fold_i), float(lv), int(dr), m, sp]
                    for lv, dr in passes for m in models for sp in MODEL_INPUTS]
        frec_old = fold_ckpt.get([int(fold_i)])
        if frec_old is not None and all(_done(k) for k in all_keys):
            fold_records[str(fold_i)] = frec_old
            n_units_resumed += len(all_keys)
            print(f"  fold {fold_i}: every unit read back from {units.path}")
            continue
        print(f"\n  Fold {fold_i} of {n_folds}: {len(train_idx)} training, "
              f"{len(val_idx)} held-out rows")
        noise_timing.set_context(fold=int(fold_i), level=None, draw=None,
                                 model=None, model_input=None)
        try:
            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=train_idx, n_folds=n_folds)
        except Exception as exc:
            print(f"    per-fold searches failed: {type(exc).__name__}: {exc}")
            fold_records[str(fold_i)] = {'error': f'per-fold searches failed: '
                                                  f'{type(exc).__name__}: {exc}'}
            continue
        frec = {'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)),
                'metric': metric_name, 'lower_is_better': bool(lower_is_better),
                'selection_scope': 'fold_train_rows', 'models': {}}
        for m in models:
            entry = entries.get(m)
            names = {}
            if entry is not None:
                if entry.get('base'):
                    names['base_1'] = entry['base'][0]
                if entry.get('hybrids'):
                    names['hybrid'] = entry['hybrids'][0].get('label')
            frec['models'][m] = configuration_record(entry, names)
        fold_records[str(fold_i)] = frec
        fold_ckpt.put([int(fold_i)], frec)

        y_clean = labels[train_idx]
        y_val = labels[val_idx]
        g_tr = None if groups is None else np.asarray(groups)[train_idx]
        smi_tr = None if smiles is None else [smiles[i] for i in train_idx]
        smi_va = None if smiles is None else [smiles[i] for i in val_idx]
        mc_tr = None if mcounts is None else np.asarray(mcounts)[train_idx]

        for level, draw in passes:
            todo = [(m, sp) for m in models for sp in MODEL_INPUTS
                    if not _done([int(fold_i), float(level), int(draw), m, sp])]
            n_units_resumed += len(models) * len(MODEL_INPUTS) - len(todo)
            if not todo:
                continue
            noise_timing.set_context(level=float(level), draw=int(draw),
                                     model=None, model_input=None)
            try:
                y0, cmask, prov = inject_condition(
                    data, dataset_name, condition, level, draw,
                    train_idx=train_idx, groups=noise_groups,
                    heavy_atoms=heavy_atoms)
                provenance.setdefault(f'level={level:g}|draw={draw}',
                                      convert_numpy(prov))
            except Exception as exc:
                errors.add(f'inject|level={level:g}', exc)
                continue
            y0 = np.asarray(y0)
            cmask = None if cmask is None else np.asarray(cmask, dtype=bool)
            cliff_train = rand_train = None
            if cliff_error is None:
                try:
                    cliff_train, rand_train, n_random_drawn = \
                        n42._cliff_masks(
                            data, dataset_name, y0, train_idx, is_clf, fold_i,
                            int(draw), cliff_structure,
                            structure_info=cliff_structure_info)
                    if n_random_drawn < int(cliff_train.sum()):
                        # The control put back fewer molecules than the cliff
                        # arm, so the two differ in two ways at once on this
                        # pass: which molecules and how many.
                        errors.add(
                            f'random_put_back_short|fold={fold_i}|'
                            f'draw={draw}',
                            ValueError(
                                f'{n_random_drawn} non-cliff molecules drawn '
                                f'against {int(cliff_train.sum())} cliff '
                                f'molecules'))
                except Exception as exc:
                    errors.add('n25/cliff_mask', exc)

            for m in models:
                sps = [sp for (mm, sp) in todo if mm == m]
                if not sps:
                    continue
                entry = entries.get(m)
                if entry is None:
                    for sp in sps:
                        unavailable.setdefault(f'fold{fold_i}|{m}:{sp}',
                                               'the per-fold search failed')
                    continue
                t0 = time.time()
                noise_timing.set_context(model=m, model_input=None)
                try:
                    ins, ins_unavail, _names = fold_model_inputs(
                        data, dataset_name, m, entry, train_idx=train_idx,
                        val_idx=val_idx, y_train=y0, factory=factories[m],
                        metric=metric, spaces=MODEL_INPUTS)
                    with noise_timing.timed('neighbour_spaces'):
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
                for mi in sps:
                    ukey = [int(fold_i), float(level), int(draw), m, mi]
                    if mi not in ins:
                        unavailable.setdefault(f'fold{fold_i}|{m}:{mi}',
                                               ins_unavail.get(mi) or 'not built')
                        continue
                    X_tr, X_va = ins[mi]
                    noise_timing.set_context(model_input=mi)
                    u = n42._Unit(
                        exp_id=EXP_ID, twins=spec is not None,
                        second_run=spec, plain_unlabelled=True,
                        ds=dataset_name, sink=sink, metric=metric,
                        task_type=task_type, is_clf=is_clf,
                        condition=condition, level=float(level),
                        draw=int(draw), fold=int(fold_i), model=m,
                        mi=f'{m}:{mi}', X_tr=X_tr, X_va=X_va, y0=y0,
                        y_clean=y_clean, y_val=y_val, g_tr=g_tr,
                        train_idx=train_idx, val_idx=val_idx, prov=prov,
                        cliff_train=cliff_train, rand_train=rand_train,
                        measurement_counts=mc_tr, row_models=(m,),
                        takes_weight=takes_weight[m],
                        # the other decided model is the second learner of
                        # the co-teaching exchange (TWO_LEARNER_DETECTOR)
                        learner_factories=factories,
                        errors=errors, unavailable=unavailable, counts=counts,
                        fold_similarity=None)
                    u.cmask = cmask
                    u.detection = {}
                    # the detector spaces: the model's own, and the hybrid
                    # for a base input
                    spaces = {'own': _Space('own', X_tr, X_va, u)}
                    if mi != HYBRID_INPUT and HYBRID_INPUT in ins:
                        spaces['hybrid'] = _Space('hybrid', ins[HYBRID_INPUT][0],
                                                  ins[HYBRID_INPUT][1], u)
                    targets_by_space = {}
                    for name, sp in spaces.items():
                        try:
                            with noise_timing.timed('correction_targets', dspace=name):
                                targets_by_space[name] = n23.correction_targets_cached(
                                    dataset_name, sp.X_tr, y0, g_tr,
                                    n07.targets_needed(list(n07.CLEANUPS)))
                        except Exception as exc:
                            errors.add(f'targets|{name}', exc)
                    spaces_tr = dict(sp_tr, model_input=X_tr)
                    spaces_va = dict(sp_va, model_input=X_va)
                    try:
                        with noise_timing.timed('neighbour_lists_and_screen'):
                            sim = n42._Similarity(u, settings, ks, spaces_tr,
                                                  spaces_va, shared_lists,
                                                  sp_unavail, smi_tr, smi_va)
                    except Exception as exc:
                        errors.add(f'similarity|{m}:{mi}', exc)
                        sim = None
                    try:
                        with noise_timing.timed('piece'):
                            run_unit(u, X_tr, X_va, spaces, sim, targets_by_space)
                    except Exception as exc:
                        errors.add(f'unit|{m}:{mi}', exc)
                    n_rows = u.rows_written()
                    units.put(ukey, {'rows_written': int(n_rows),
                                     'seconds': round(time.time() - t0, 1),
                                     'n_features': int(np.shape(X_tr)[1]),
                                     'detector_spaces': list(spaces),
                                     'base_scores': u.base_scores,
                                     'detection': convert_numpy(u.detection),
                                     'screen': (sim.screen_record()
                                                if sim is not None else None)})
                    n_units_done += 1
                    print(f"    fold {fold_i} level {level:g} draw {draw} "
                          f"{m}:{mi}: {n_rows} rows, {time.time() - t0:.0f}s")
                    t0 = time.time()

    # summary: the plain model on D at each level, per fold (as n42)
    unit_vals = []
    for kjson, v in units.entries.items():
        try:
            f_, lv_, dr_, m_, sp_ = json.loads(kjson)
        except Exception:
            continue
        unit_vals.append((int(f_), float(lv_), int(dr_), m_, sp_, v or {}))

    def _by_fold(pick):
        out = {}
        for f_, lv_, _dr, m_, sp_, v in unit_vals:
            x = pick(v)
            if x is None or not np.isfinite(x):
                continue
            out.setdefault((m_, sp_, lv_), {}).setdefault(f_, []).append(float(x))
        return {key: {f_: float(np.mean(xs)) for f_, xs in byf.items()}
                for key, byf in out.items()}

    def _bs(v, k):
        return (v.get('base_scores') or {}).get(k)

    score_D = _by_fold(lambda v: _bs(v, 'score_D'))
    minus_clean = _by_fold(lambda v: (None if _bs(v, 'score_D') is None
                                      or _bs(v, 'score_clean') is None
                                      else _bs(v, 'score_D') - _bs(v, 'score_clean')))
    summary = {}
    for (m_, sp_, lv_), byf in score_D.items():
        summary.setdefault(m_, {}).setdefault(sp_, {})[f'{lv_:g}'] = {
            'score_D': fold_spread([byf[f_] for f_ in sorted(byf)]),
            'damaged_minus_clean_labels': fold_spread(
                list((minus_clean.get((m_, sp_, lv_)) or {}).values()))}

    results = {
        'experiment': EXP_ID, 'result_layout': RESULT_LAYOUT,
        'metric': metric_name, 'lower_is_better': bool(lower_is_better),
        'noise_condition': condition, 'task_type': task_type,
        'levels': list(levels), 'level_unit': level_unit(condition, is_clf),
        'level0_rows_file': (None if output_dir is None else level0_rows_path(
            output_dir, EXP_ID, dataset_name, data, RESULT_LAYOUT)),
        'draws_per_nonzero_level': len(draws),
        'passes': [[float(lv), int(dr)] for lv, dr in passes],
        'knn_group': knn_grp, 'n_folds': int(n_folds),
        'folds_run': [int(f) for f, _t, _v in folds],
        'models': models, 'models_decided': decided,
        'models_without_sample_weight': [m for m in models if not takes_weight[m]],
        'model_inputs': list(MODEL_INPUTS),
        'chains': {'1': 'model on base_1, detector in the same (dspace=own)',
                   '2': 'model on base_1, detector in the hybrid (dspace=hybrid)',
                   '3': 'model on the hybrid, detector in the hybrid (model_input hybrid, dspace=own)'},
        'detectors': {'registry': list(REGISTRY_DETECTORS),
                      'residual': list(RESIDUAL_DETECTORS),
                      'n41': list(N41_DETECTORS),
                      'flag': ['conformal_bh@cal|q', 'oracle@precision|flag_share'],
                      'controls': list(CONTROL_DETECTORS),
                      'spread': list(SPREAD_DETECTORS),
                      'regression_only': list(REGRESSION_ONLY_DETECTORS)},
        'cleanups': {c: {k: v for k, v in d.items()} for c, d in CLEANUPS.items()},
        'cleanups_cut': {c: 'rounds_delete cut 2026-10-31 (author): the trimmed '
                            'refit raised the top in no more cells than random '
                            'deletion and fewer than deleting once'
                         for c in CUT_CLEANUPS},
        'cleanup_settings': {c: [lab for lab, _kw in cleanup_settings(c)]
                             for c in CLEANUPS},
        'flag_shares': list(FLAG_SHARES), 'rounds': list(ROUNDS),
        'precisions': list(PRECISIONS),
        'folds': fold_records, 'summary_by_level': summary,
        'n25_cliff_reference_error': cliff_error,
        'units_computed_this_run': int(n_units_done),
        'units_resumed': int(n_units_resumed),
        'counts': counts, 'errors': errors.d, 'unavailable': unavailable,
        'noise_provenance': provenance,
        'similarity': n07.similarity_record(
            settings, sink, schemes_swept=list(schemes),
            screen_top=n07.screen_top(), screen_scored_on=n07.SCREEN_SCORED_ON,
            knn_group=knn_grp, scaffold_mates_excluded=False),
        'checkpoint': {'units': units.path, 'folds': fold_ckpt.path},
        'seconds': round(time.time() - t_start, 1),
    }
    print(f"\n  N44 {condition}: {n_units_done} units computed, "
          f"{n_units_resumed} resumed, {sink.n_written} rows written, "
          f"{len(errors.d)} distinct errors, {time.time() - t_start:.0f}s")
    return results


def main():
    import argparse
    from experiments.noise.noise_helpers import run_all
    ap = argparse.ArgumentParser(description='N44 on one dataset through '
                                             'noise_helpers.run_all')
    ap.add_argument('dataset')
    ap.add_argument('--output', required=True)
    ap.add_argument('--noise-condition', default=None)
    args = ap.parse_args()
    run_all(dataset_name=args.dataset, output_dir=args.output,
            experiments=['n44'], noise_condition=args.noise_condition)


if __name__ == '__main__':
    main()
