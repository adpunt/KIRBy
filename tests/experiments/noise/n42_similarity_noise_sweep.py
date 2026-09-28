#!/usr/bin/env python3
"""N42: every data-changing mitigation method x every similarity setting x
every model input, scored as D against D'.

Plan: ~/.claude/plans/pasted-content-id-0428-noise-experiment-fizzy-ladybug.md,
section 6 ("One new experiment"). Written 2026-09-25.

WHAT ONE ROW IS. d_vs_dprime's row: one downstream model (rf, svm, and the
pass's own model when its fit is seeded) fitted on D -- the training fold with
the injected labels -- and on D' -- D after one mitigation method -- and both
scored on the same held-out fold in the dataset's own metric, beside the fit on
the clean training labels. Rows go to
<output>/n42_similarity_noise_sweep/<dataset>__<condition>.rows.jsonl as each
(fold, level, draw, model input) unit finishes.

THE LOOPS, outermost first:
  condition   noise_helpers.sweep_conditions over conditions_for_task(data), one
              result file and one rows file per condition. It sits outside the
              fold loop because sweep_conditions writes and resumes per
              condition; the rows are the same whichever loop is outermost.
  fold        _shared.cv_split, the split every noise experiment uses
  level       dataset_policy.noise_levels for the condition
  draw        NOISE_DRAWS; one draw at level 0 on a regression set
              (n23._levels_and_draws)
  model       every downstream model the config names
  model input that model's base_1, base_2 (its rank-1 and rank-2 single
              representation in its own stored search,
              _shared.noise_base_reps_by_model) and hybrid (its prediction-built
              recipe from the precompute, pool_recipe_per_downstream_model +
              pool_recipes_by_family). The hybrid's features are chosen from the
              NOISY training labels, as n23 and n32 build theirs. Built by
              n07.fold_similarity_spaces, so the model input and the similarity
              space of the same name are one cache entry.
  method      below, each imported from the experiment that owns it
  similarity  for a method that reads neighbours: every setting
              n07.sweep_settings() returns (KIRBY_SIMILARITY_SETTINGS; model_input
              first) x every k in SWEPT_KS that the lists hold

THE METHODS, by the experiment they are imported from. Nothing below re-implements
a method; the functions named are called.

  n07  the ten-cleanup menu, n07.cleaned_training_set
         detector 'cross_val' (n07's DETECTOR): similarity 'none' for every
             cleanup that reads no neighbours; label_smooth and knn_smooth under
             every setting and k; prox_smooth under forest_proximity only
         detectors 'neighbor_residual' and 'activity_cliff_aware' (n25.DETECTORS
             that read neighbours): every score-driven cleanup under every
             setting and k, the lists built with the scaffold groups excluded,
             as n07._compute_noise_scores passes them
         score-free cleanups (quantile_clip, invvar_weight): once
       Each cleanup at every value of its own axis, n25.setting_axis: the flag
       rates of n24 and n25 together for a cleanup that thresholds, n25.STRENGTHS
       for one that leans on the score.
  n24  its arms are n07 cleanups under neighbor_residual and cross_val at its
       FLAG_RATES, all inside the n07 block above. Its delete_at_random control
       is the random twin controls.random_twin writes under
       KIRBY_NOISE_CONTROLS=1.
  n25  every n07 D' also as 'arm=cliff_exempt' and 'arm=random_exempt'
       (n25.put_back over n25.cliff_mask_for_dataset / random_exempt_mask). The
       cliff set is ECFP4 Tanimoto, not on the similarity axis (n25's own TODO).
  n02  reliability_weight (n02.reliability_from_neighbours) and
       cliff_aware_weight (ActivityCliffAwareEstimator.get_weights on the
       cached scores), every setting and k, no group exclusion, as n02's sweep.
  n23  drop and move arms (n23._arms, n23.drop_keep_mask,
       n23.cleaned_training_set) for every source in n23.SOURCES. A model-based
       source is scored once; its move arms are swept only when its target is
       the neighbours' mean (n23's rule). activity_cliff_aware is scored under
       every setting and k.
  n30  winsorise at every WINSOR_C; under censoring the three lift columns
       (n30.lift_columns without its shuffled controls).
  n31  remove / downweight / rewrite on the top-residual molecules
       (n31.detector_flags, n31.apply_operation), at every share in
       FLAG_SHARES of the training molecules.
  n32  itlm at every alpha, two_learner_exchange and shrink_flagged (both
       columns) at every share in FLAG_SHARES.
  n33  invvar_ngboost weights and c_mixup (D' holds 2n rows).
  n35  downweight_pairs in the model input's own distance and in every
       similarity setting's full training distance (FoldSimilarity.full_distance).
  n38  its three data-changing arms are n30's winsorise at c=1.5, n32's itlm at
       alpha 0.9 and n33's invvar_ngboost, all already above.
  n41  conformal flags (n41.rescaled_residual, random_calibration,
       conformal_pvalues, benjamini_hochberg) at every calibration share and q,
       acted on by winsorise_flagged and by removal.

  NOT IMPORTED, and why, is METHODS_NOT_RUN below; it is written into every
  result file.

The n30-n41 families read an out-of-fold residual on a numeric label, as their
own files do (each runs regression only), so they are not run on a
classification dataset; the result file says so.

CHECK RUNS. KIRBY_N42_FAMILIES=n07,n02 (a comma list of FAMILIES) runs only
those families; unset runs all. KIRBY_SIMILARITY_SETTINGS narrows the settings.
Neither is a cut of the sweep: both default to everything.

RESUME. Rows: n07.RowSink skips every row key already on disk. Work: each
finished (fold, level, draw, model, model input) unit is recorded in
<dataset>__<condition>.units.jsonl (n07.FoldCheckpoint) under the run's input
fingerprint, the settings, the ks, the families, the models and the controls
flag; a unit recorded there is not recomputed. --force-rerun does NOT reach
either file (sweep_conditions consumes it): delete them to redo.
"""
from __future__ import annotations

import copy
import functools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments._shared import (
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS_NOISE,
    get_best_search_for_downstream, noise_base_reps_by_model,
    pool_recipes_by_family, POOL_SUBSETS, input_fingerprint, convert_numpy,
)
from experiments.dataset_policy import noise_groups_for
from experiments.noise.noise_helpers import (
    noise_spec_for, inject_train_labels, sweep_conditions, conditions_for_task,
    pool_recipe_per_downstream_model,
)
from kirby.noise_spec import GROUPED_STRATEGIES
from kirby.utils.noise_estimation import ActivityCliffAwareEstimator

from experiments.noise import noise_cache as nc
from experiments.noise import similarity_space as ss
from experiments.noise.controls import controls_enabled
from experiments.noise import n07_remediation as n07
from experiments.noise import n02_sample_weighting as n02
from experiments.noise import n23_uncertainty_cleaning as n23
from experiments.noise import n24_delete_or_keep as n24
from experiments.noise import n25_cliff_exemption as n25
from experiments.noise import n30_censored_likelihood as n30
from experiments.noise import n31_regime_and_breakeven as n31
from experiments.noise import n32_iterative_and_dual_selection as n32
from experiments.noise import n33_robust_training as n33
from experiments.noise import n35_near_duplicate_disagreement as n35
from experiments.noise import n41_conformal_fdr_detection as n41


EXP_ID = 'n42_similarity_noise_sweep'

# The replicate index of the injection; the same three n07, n23, n24 and n32
# sweep. One is kept at level 0 on a regression set (n23._levels_and_draws).
NOISE_DRAWS = (0, 1, 2)

# Same pool subset and one space per search family as n07, n23 and n24.
POOL_SUBSET = 'predictive_and_similarity'
SPACE_FOR_FAMILY = {'predictive': 'hybrid', 'similarity': 'hybrid_similarity'}

# Owner decision 2026-09-25: the model is fitted on each model's top-2 single
# representations and its prediction-built hybrid, unchanged.
MODEL_INPUTS = ('base_1', 'base_2', 'hybrid')

# The families this file runs, in the order they run in each unit.
FAMILIES = ('n07', 'n02', 'n23', 'n30', 'n31', 'n32', 'n33', 'n35', 'n41')

# Families that read an out-of-fold residual on a numeric label. Each of their
# own files runs regression only (n30:412-417; n31, n32, n33, n35, n41 inject
# with is_clf=False throughout).
REGRESSION_ONLY_FAMILIES = ('n30', 'n31', 'n32', 'n33', 'n35', 'n41')

# The flag rates a thresholding n07 cleanup is scored at: n24's and n25's
# grids together, (0.01, 0.05, 0.1, 0.2).
FLAG_RATES = tuple(sorted(set(float(r) for r in n24.FLAG_RATES)
                          | set(float(r) for r in n25.FLAG_RATES)))

# The shares of training molecules the flag-count methods of n31 and n32 flag:
# n24.FLAG_RATES, the grid n31.FLAG_SHARES and n32.FLAG_SHARES also sweep. The
# same at every condition and level. Until 2026-09-28 this was one share per
# condition, the condition's nominal share; owner's decision 2026-09-28 is that
# no method may read what only exists because the noise was injected.
FLAG_SHARES = tuple(float(r) for r in n24.FLAG_RATES)

# The flag rate handed to a cleanup that does not read one (n25.reads_flag_rate
# says which). n07's own default noise_fraction (n07:2303). It changes nothing
# for those cleanups; it is passed because the argument is positional.
UNREAD_FLAG_RATE = 0.1

# The detectors the n07 cleanups are driven by. n25's three: cross_val reads
# no neighbours, the other two do.
N07_DETECTORS = tuple(n25.DETECTORS)

# Models whose fit noise_cache may store (n23.CACHEABLE_MODELS). A pass's own
# model is added to rf and svm only when it is one of these; see TODO below.
CACHEABLE_MODELS = tuple(n23.CACHEABLE_MODELS)

# Every data-changing method named in the plan that this file does not run,
# and why. Written into every result file.
METHODS_NOT_RUN = {
    'n31/bayes_shrink': (
        'its shrink weight is sigma_n^2 / (sigma_n^2 + clean MSE), where '
        'sigma_n^2 is the injected error actually delivered (n31:719-721). '
        'That reads the corruption, which no mitigation method can see.'),
    'n31/synthetic_flags': (
        'flags built from the corruption mask at a chosen precision '
        '(n31.synthetic_flags). A measurement of break-even precision, not a '
        'method a user could run.'),
    'n33/huber_teacher_blend': (
        'the blended label column is built inside n33.label_arm_predict '
        '(n33:474-476), which also fits and predicts. No function returns the '
        'column, so it cannot be imported, and copying the line is not '
        'allowed.'),
    'n33/yeo_johnson': (
        'a target transform whose predictions are transformed back '
        '(n33.label_arm_predict). A D-prime label column cannot carry the '
        'inverse transform, so the fit is a different model, not different '
        'data.'),
    'n33/target_noise_bagging': (
        'an average of several fits on relabelled copies (n33.label_arm_predict). '
        'One D-prime cannot express an ensemble.'),
    'n33/robust_loss_swaps': (
        'Huber, Tukey and expectile objectives change the model, not the '
        'training data.'),
    'n30/aft_and_comparators': (
        'xgboost_aft, rf_l1 and lightgbm_huber change the model, not the data.'),
    'n30/per_molecule_lift_shuffled': (
        "n30's own control. The shared random and shuffled twins cover it "
        'under KIRBY_NOISE_CONTROLS=1.'),
    'n32/random_keep, n41/random_flags, n24/delete_at_random, n38/random': (
        'each experiment\'s own random control. The shared random twin '
        '(controls.random_twin) covers them under KIRBY_NOISE_CONTROLS=1.'),
    'n41/oracle_clean and trusted calibration': (
        'oracle_clean calibrates on molecules the injector did not move, which '
        'reads the corruption; trusted needs repeat measurements. Only the '
        'random calibration shares are run.'),
    'n41/isolation_forest, lof, residual_low_disagreement; n32/knn_shapley, '
    'ngboost_rescaled, quantile_forest_rescaled, conformal, cleanlab': (
        'detectors that are only graded in their own files; no arm there '
        'changes the training data from them.'),
    'n38/robust_loss': 'changes the model, not the data.',
    'n09, n11, n12, n13, n21 methods': (
        'not in the list of files this experiment was asked to import from '
        '(plan section 6, n42 brief).'),
}

# Families whose arms are already run by another family here.
COVERED_BY_ANOTHER_FAMILY = {
    'n24': 'n07 block: cleanups under neighbor_residual and cross_val at '
           'n24.FLAG_RATES, which are inside FLAG_RATES here',
    'n25': 'n07 block: every n07 D-prime also written with arm=cliff_exempt '
           'and arm=random_exempt',
    'n38': "n30 winsorise at c=1.5, n32 itlm at alpha 0.9 and n33 "
           'invvar_ngboost',
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def families_asked(kwargs=None):
    """FAMILIES, or the subset named by kwargs['families'] or
    KIRBY_N42_FAMILIES (a comma list). An unknown name raises."""
    raw = (kwargs or {}).get('families') or os.environ.get('KIRBY_N42_FAMILIES')
    if not raw:
        return FAMILIES
    names = ([str(x).strip() for x in raw] if isinstance(raw, (list, tuple))
             else [t.strip() for t in str(raw).split(',')])
    names = [n for n in names if n]
    bad = [n for n in names if n not in FAMILIES]
    if bad:
        raise ValueError(f'n42: unknown families {bad}; known {FAMILIES}')
    return tuple(f for f in FAMILIES if f in names)


def _keep_dict(X_tr, y_tr, keep):
    """A deletion D' in the dict shape n07.cleaned_training_set returns."""
    keep = np.asarray(keep, dtype=bool)
    y_tr = np.asarray(y_tr)
    return {'X': X_tr[keep], 'y': y_tr[keep], 'sample_weight': None,
            'keep': keep, 'label_column': None}


def _rows_added_dict(X_new, y_new):
    """A D' with rows added (c_mixup). No keep, weight or label column, so
    controls.random_twin and shuffled_twin write no twin for it."""
    return {'X': np.asarray(X_new), 'y': np.asarray(y_new),
            'sample_weight': None, 'keep': None, 'label_column': None}


def _n07_cleanups(is_clf):
    return tuple(c for c in n07.CLEANUPS
                 if not (is_clf and c in n07.REGRESSION_ONLY_CLEANUPS))


def _n07_slots(cleanup, is_clf):
    """[(flag rate, strength, label)] for one n07 cleanup, from
    n25.setting_axis: every flag rate for a cleanup that thresholds, every
    strength for one that leans on the score, one default otherwise."""
    axis, vals = n25.setting_axis(cleanup, is_clf)
    if axis == 'flag_rate':
        return [(r, None, f'flag_rate={r:g}') for r in FLAG_RATES]
    if axis == 'strength':
        return [(UNREAD_FLAG_RATE, float(v), f'strength={float(v):g}')
                for v in vals]
    return [(UNREAD_FLAG_RATE, None, 'default')]


class _Errors:
    """{where: {'count': n, 'first': message}}, so one method failing on every
    unit is one entry, not thousands."""

    def __init__(self):
        self.d = {}

    def add(self, where, exc):
        msg = exc if isinstance(exc, str) else f'{type(exc).__name__}: {exc}'
        e = self.d.setdefault(where, {'count': 0, 'first': msg})
        e['count'] += 1


# ---------------------------------------------------------------------------
# One unit: (fold, level, draw, model, model input)
# ---------------------------------------------------------------------------

class _Unit:
    """Everything one unit's methods read, and the one place a row is written."""

    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.n_rows_before = kw['sink'].n_written
        self._p = None
        self._p_done = False

    # -- the out-of-fold LightGBM prediction, shared by n30-n41 --
    def p(self):
        if not self._p_done:
            self._p_done = True
            try:
                self._p = nc.cached_oof_predict(
                    self.ds, self.X_tr, self.y0, self.g_tr,
                    n_splits=min(int(n30.OOF_SPLITS), len(self.y0)))
            except Exception as exc:
                self.errors.add('oof_prediction', exc)
                self._p = None
        return self._p

    def emit(self, family, method, method_setting, similarity_setting,
             cleaned, exempt=False):
        """Write the rows of one D' (n07.dprime_rows: RF, SVM and the pass's
        own model when cacheable, plus the control twins when enabled). With
        exempt=True the n25 arms are written beside it."""
        ms = (f'{method_setting}|arm=plain' if exempt else method_setting)
        ident = n07.row_identity(
            dataset_name=self.ds, experiment=EXP_ID,
            method=f'{family}/{method}', method_setting=ms,
            similarity_setting=similarity_setting, model_input=self.mi,
            noise_condition=self.condition, level=self.level, draw=self.draw,
            fold=self.fold)
        try:
            n07.dprime_rows(self.sink, dataset_name=self.ds, cleaned=cleaned,
                            X_tr=self.X_tr, y_D=self.y0, X_va=self.X_va,
                            y_va=self.y_val, metric=self.metric,
                            task_type=self.task_type, y_clean=self.y_clean,
                            models=self.row_models, identity=ident,
                            is_clf=self.is_clf)
        except Exception as exc:
            self.errors.add(f'{family}/{method}|fit', exc)
            return
        if not exempt or cleaned is None or self.cliff_train is None:
            return
        for arm, mask in (('cliff_exempt', self.cliff_train),
                          ('random_exempt', self.rand_train)):
            try:
                back = n25.put_back(cleaned, mask, self.X_tr, self.y0)
            except Exception as exc:
                self.errors.add(f'{family}/{method}|{arm}', exc)
                continue
            self.emit(family, method, f'{method_setting}|arm={arm}',
                      similarity_setting, back, exempt=False)

    def rows_written(self):
        return self.sink.n_written - self.n_rows_before


class _Similarity:
    """The neighbour lists of one unit, per setting, built on first use.

    Lists of a feature space or Tanimoto are shared by every model input of a
    model (`shared`, one dict per (fold, level, draw, model)); MODEL_INPUT and
    forest_proximity read the model-input matrix and are this unit's own. Every
    list goes through n07.fold_neighbours -> noise_cache.cached_neighbours.
    `grouped` lists exclude a molecule's scaffold group (for detectors), plain
    ones do not (for smoothing targets and n02), as n07's sweep builds them.
    """

    PER_INPUT = (ss.MODEL_INPUT_SPACE, ss.FOREST_SPACE)

    def __init__(self, u, settings, ks, spaces_tr, spaces_va, shared,
                 unavailable_spaces, smiles_tr, smiles_va):
        self.u = u
        self.settings = settings
        self.ks = tuple(ks)
        self.sp_tr = spaces_tr
        self.sp_va = spaces_va
        self.shared = shared
        self.own = {}
        self.unavailable_spaces = unavailable_spaces
        self.smi_tr = smiles_tr
        self.smi_va = smiles_va

    def lists(self, s, grouped=False):
        grouped = bool(grouped) and self.u.g_tr is not None
        memo = self.own if s.space in self.PER_INPUT else self.shared
        key = (s.name, grouped)
        if key not in memo:
            got = n07.fold_neighbours(
                self.u.ds, [s], self.sp_tr, self.sp_va, self.u.y0,
                exclude_groups=self.u.g_tr if grouped else None,
                smiles_train=self.smi_tr, smiles_val=self.smi_va,
                is_clf=self.u.is_clf).get(s.name)
            if not got or not got.get('available'):
                why = (self.unavailable_spaces.get(s.space)
                       or (got or {}).get('reason') or 'not built')
                self.u.unavailable.setdefault(
                    f'{self.u.model}|{s.name}', why)
                got = None
            memo[key] = got
        return memo[key]

    def ks_for(self, lists):
        return tuple(k for k in n07.usable_ks(lists, self.ks)) if lists else ()


# ---------------------------------------------------------------------------
# The families
# ---------------------------------------------------------------------------

def _fam_n07(u, sim):
    """n07's menu under n25's detectors and n24/n25's settings, with n25's
    exemption arms."""
    NB = n07.NEIGHBOUR_CLEANUPS
    cleanups = _n07_cleanups(u.is_clf)
    try:
        targets = n23.correction_targets_cached(
            u.ds, u.X_tr, u.y0, u.g_tr, n07.targets_needed(cleanups))
    except Exception as exc:
        u.errors.add('n07/correction_targets', exc)
        targets = None

    def clean(st, scores, rate, strength, lists=None, k=None):
        kw = ({} if lists is None else
              {'similarity_neighbours': lists, 'similarity_k': k})
        try:
            return n07.cleaned_training_set(
                st, u.X_tr, u.y0, scores, rate, is_clf=u.is_clf,
                groups=u.g_tr, strength=strength, targets=targets,
                measurement_counts=u.measurement_counts, dataset_name=u.ds,
                **kw)
        except Exception as exc:
            u.errors.add(f'n07/{st}', exc)
            return None

    # Score-free cleanups: no detector, no neighbours.
    for st in cleanups:
        if st not in n07.SCORE_FREE_CLEANUPS:
            continue
        for rate, strength, lab in _n07_slots(st, u.is_clf):
            u.emit('n07', st, f'detector=none|{lab}', n07.NO_NEIGHBOURS,
                   clean(st, None, rate, strength), exempt=True)

    # cross_val: reads no neighbours.
    cv = None
    if 'cross_val' in N07_DETECTORS:
        try:
            cv = nc.cached_detector_scores(u.ds, u.X_tr, u.y0, is_clf=u.is_clf,
                                           detector='cross_val', groups=u.g_tr)
        except Exception as exc:
            u.errors.add('n07/detector=cross_val', exc)
        for st in cleanups:
            if st in n07.SCORE_FREE_CLEANUPS or st in NB:
                continue
            for rate, strength, lab in _n07_slots(st, u.is_clf):
                u.emit('n07', st, f'detector=cross_val|{lab}', n07.NO_NEIGHBOURS,
                       None if cv is None else clean(st, cv, rate, strength),
                       exempt=True)

    nb_detectors = [d for d in N07_DETECTORS if d in nc.NEIGHBOUR_DETECTORS]
    for s in sim.settings:
        L = sim.lists(s, grouped=False)
        Lg = sim.lists(s, grouped=True) if nb_detectors else None
        ks = sorted(set(sim.ks_for(L)) | set(sim.ks_for(Lg)))
        for k in ks:
            L_ok = L is not None and k in sim.ks_for(L)
            # cross_val scores, the neighbour cleanups reading this setting.
            if 'cross_val' in N07_DETECTORS and L_ok:
                for st in cleanups:
                    if st not in NB:
                        continue
                    if st == 'prox_smooth' and s.space != ss.FOREST_SPACE:
                        continue
                    for rate, strength, lab in _n07_slots(st, u.is_clf):
                        u.emit('n07', st, f'k={k}|detector=cross_val|{lab}',
                               s.name,
                               None if cv is None else
                               clean(st, cv, rate, strength, L, k),
                               exempt=True)
            # The neighbour detectors, scored in this setting.
            if Lg is None or k not in sim.ks_for(Lg):
                continue
            for d in nb_detectors:
                try:
                    sc = nc.cached_detector_scores(
                        u.ds, u.X_tr, u.y0, is_clf=u.is_clf, detector=d,
                        groups=u.g_tr, similarity_neighbours=Lg,
                        similarity_k=k)
                except Exception as exc:
                    u.errors.add(f'n07/detector={d}', exc)
                    sc = None
                for st in cleanups:
                    if st in n07.SCORE_FREE_CLEANUPS:
                        continue
                    reads = st in NB
                    if reads and (not L_ok or (st == 'prox_smooth'
                                               and s.space != ss.FOREST_SPACE)):
                        continue
                    for rate, strength, lab in _n07_slots(st, u.is_clf):
                        u.emit('n07', st, f'k={k}|detector={d}|{lab}', s.name,
                               None if sc is None else
                               clean(st, sc, rate, strength,
                                     L if reads else None,
                                     k if reads else None),
                               exempt=True)


def _fam_n02(u, sim):
    """n02's two weights from each setting's neighbours (n02's sweep:
    no group exclusion, groups=None to the cliff estimator)."""
    for s in sim.settings:
        L = sim.lists(s, grouped=False)
        for k in sim.ks_for(L):
            try:
                idx = ss.neighbours_at(L, k)[0]
                w = n02.reliability_from_neighbours(u.y0, idx, is_clf=u.is_clf)
                d = n07.cleaned_from_weights(u.X_tr, u.y0, w)
            except Exception as exc:
                u.errors.add('n02/reliability_weight', exc)
                d = None
            u.emit('n02', 'reliability_weight',
                   f'k={k}|strength={float(n02.DEFAULT_STRENGTH):g}', s.name, d)
            d2 = None
            try:
                sc = n07.detector_scores_with(u.ds, 'activity_cliff_aware',
                                              u.X_tr, u.y0, L, k,
                                              is_clf=u.is_clf)
                if sc is not None:
                    est = ActivityCliffAwareEstimator(k=k, metric=None,
                                                      is_clf=u.is_clf)
                    est.noise_scores_ = np.asarray(sc, dtype=float)
                    d2 = n07.cleaned_from_weights(
                        u.X_tr, u.y0, est.get_weights(strength=1.0))
            except Exception as exc:
                u.errors.add('n02/cliff_aware_weight', exc)
            u.emit('n02', 'cliff_aware_weight', f'k={k}|strength=1', s.name, d2)


def _n23_dprime(u, arm, scores, target):
    """One n23 arm's D', or None when it declined. 'skip' for a move on a
    class label (see TODO in _fam_n23)."""
    name, shape, setting, method = arm
    if shape == 'drop':
        keep = n23.drop_keep_mask(setting, scores)
        return None if int(keep.sum()) < 10 else _keep_dict(u.X_tr, u.y0, keep)
    if shape == 'move':
        if u.is_clf:
            return 'skip'
        if target is None:
            return None
        got = n23.cleaned_training_set('move', setting, u.X_tr, u.y0, scores,
                                       target, method=method)
        return None if got is None else n07.cleaned_from_labels(u.X_tr, got[1])
    return None


def _fam_n23(u, sim):
    """n23's drop and move arms for every uncertainty source."""
    arms = [a for a in n23._arms(n23.DROP_SHARES, n23.MOVE_STRENGTHS,
                                 n23.MOVE_METHODS) if a[1] != 'none']
    # TODO(similarity-sweep 2026-09-25): n23 runs its move arms on a class
    # label too (smooth_by_scores on 0/1 values). They are skipped here on a
    # classification set; whether n23 should do the same is the owner's call.

    def emit_arm(source, arm, scores, target, sim_name, k=None):
        try:
            d = _n23_dprime(u, arm, scores, target)
        except Exception as exc:
            u.errors.add(f'n23/{arm[0]}', exc)
            d = None
        if isinstance(d, str):
            return
        pre = '' if k is None else f'k={k}|'
        u.emit('n23', arm[0], f'{pre}source={source}', sim_name, d)

    model_sources = [s_ for s_ in n23.SOURCES
                     if s_ not in nc.NEIGHBOUR_DETECTORS]
    nb_sources = [s_ for s_ in n23.SOURCES if s_ in nc.NEIGHBOUR_DETECTORS]

    def nb_target(L, k):
        idx = ss.neighbours_at(L, k)[0]
        return np.mean(np.asarray(u.y0, dtype=float)[idx], axis=1)

    for src in model_sources:
        try:
            scores, target, why = n23._uncertainty_cached(
                u.ds, src, u.X_tr, u.y0, u.g_tr, u.is_clf,
                n23.ESTIMATOR_FOLDS, neighbour_k=n23.NEIGHBOUR_K)
        except Exception as exc:
            u.errors.add(f'n23/source={src}', exc)
            continue
        if scores is None:
            u.errors.add(f'n23/source={src}', str(why))
            continue
        scores = np.asarray(scores, dtype=float)
        own = (why == n23.OWN_TARGET)
        for arm in arms:
            if arm[1] == 'drop' or own:
                emit_arm(src, arm, scores, target, n07.NO_NEIGHBOURS)
        if own:
            continue
        # The move target is the neighbours' mean: swept (n23's rule).
        for s in sim.settings:
            L = sim.lists(s, grouped=False)
            for k in sim.ks_for(L):
                try:
                    tgt = nb_target(L, k)
                except Exception as exc:
                    u.errors.add(f'n23/target|{src}', exc)
                    continue
                for arm in arms:
                    if arm[1] == 'move':
                        emit_arm(src, arm, scores, tgt, s.name, k)

    for src in nb_sources:
        for s in sim.settings:
            L = sim.lists(s, grouped=False)
            Lg = sim.lists(s, grouped=True)
            for k in sim.ks_for(Lg):
                if k not in sim.ks_for(L):
                    continue
                try:
                    sc = nc.cached_detector_scores(
                        u.ds, u.X_tr, u.y0, is_clf=u.is_clf, detector=src,
                        groups=u.g_tr, similarity_neighbours=Lg,
                        similarity_k=k)
                    tgt = nb_target(L, k)
                except Exception as exc:
                    u.errors.add(f'n23/source={src}', exc)
                    continue
                if sc is None:
                    continue
                sc = np.asarray(sc, dtype=float)
                for arm in arms:
                    emit_arm(src, arm, sc, tgt, s.name, k)


def _fam_n30(u, sim):
    p = u.p()
    if p is None:
        return
    for c in n30.WINSOR_C:
        try:
            col, _s, _moved = n30.winsorise(u.y0, p, float(c))
            d = n07.cleaned_from_labels(u.X_tr, col)
        except Exception as exc:
            u.errors.add('n30/winsorise', exc)
            d = None
        u.emit('n30', 'winsorise', f'c={float(c):g}', n07.NO_NEIGHBOURS, d)
    if u.condition != 'censoring' or float(u.level) <= 0:
        return
    limit = (u.prov or {}).get('censoring_limit')
    if limit is None:
        u.errors.add('n30/lift', 'no censoring_limit on the provenance row')
        return
    try:
        at = n30.rows_at_limit(u.y0, float(limit))
        sd = n30.residual_sd(u.y0, p, at)
        if not at.any() or not np.isfinite(sd):
            u.errors.add('n30/lift', 'no molecule at the limit, or no residual '
                                     'SD off it')
            return
        cols, _info = n30.lift_columns(u.y0, p, at, float(limit), sd, (), ())
    except Exception as exc:
        u.errors.add('n30/lift', exc)
        return
    for arm, col in cols.items():
        u.emit('n30', arm, 'default', n07.NO_NEIGHBOURS,
               n07.cleaned_from_labels(u.X_tr, col))


def _fam_n31(u, sim):
    """n31's detector flag set at every share in FLAG_SHARES, the same at every
    condition and level (owner's decision 2026-09-28), as n31 now does."""
    p = u.p()
    if p is None:
        return
    for share in FLAG_SHARES:
        k = int(round(share * len(u.y0)))
        if k <= 0:
            continue
        flag = n31.detector_flags(np.abs(np.asarray(u.y0, dtype=float) - p), k)
        for op in ('remove', 'downweight', 'rewrite'):
            try:
                lab, keep, w = n31.apply_operation(op, u.y0, flag, p, 0.0)
                if keep is not None:
                    d = _keep_dict(u.X_tr, u.y0, keep)
                elif w is not None:
                    d = n07.cleaned_from_weights(u.X_tr, u.y0, w)
                else:
                    d = n07.cleaned_from_labels(u.X_tr, lab)
            except Exception as exc:
                u.errors.add(f'n31/{op}', exc)
                d = None
            u.emit('n31', f'detector/{op}',
                   f'flag_share={share:g}|downweight_to={n31.DOWNWEIGHT_TO:g}'
                   if op == 'downweight' else f'flag_share={share:g}',
                   n07.NO_NEIGHBOURS, d)


def _fam_n32(u, sim):
    p = u.p()
    if p is None:
        return
    y0 = np.asarray(u.y0, dtype=float)
    for a in n32.ITLM_ALPHAS:
        key = {'X': nc.array_hash(u.X_tr), 'y': nc.array_hash(y0),
               'groups': nc.array_hash(u.g_tr), 'alpha': float(a),
               'rounds': int(n32.ITLM_ROUNDS),
               'oof_splits': int(n32.OOF_SPLITS), 'p': nc.array_hash(p)}
        try:
            keep = nc.cached(u.ds, 'n42_itlm_keep', key, lambda a=a: np.asarray(
                n32.itlm(u.X_tr, y0, u.g_tr, a, rounds=n32.ITLM_ROUNDS,
                         oof_splits=n32.OOF_SPLITS,
                         first_loss=(y0 - p) ** 2)[0], dtype=bool))
            d = _keep_dict(u.X_tr, y0, keep)
        except Exception as exc:
            u.errors.add('n32/itlm', exc)
            d = None
        u.emit('n32', 'itlm', f'alpha={float(a):g}|rounds={n32.ITLM_ROUNDS}',
               n07.NO_NEIGHBOURS, d)

    # two_learner and the flagged-only shrinks at every share in FLAG_SHARES,
    # the same at every condition and level (owner's decision 2026-09-28).
    for share in FLAG_SHARES:
        d = None
        try:
            lf = u.learner_factories
            la, lb = n32.EXCHANGE_LEARNERS[0], n32.EXCHANGE_LEARNERS[1]
            key = {'X': nc.array_hash(u.X_tr), 'y': nc.array_hash(y0),
                   'groups': nc.array_hash(u.g_tr), 'share_kept': 1.0 - share,
                   'rounds': int(n32.EXCHANGE_ROUNDS),
                   'oof_splits': int(n32.OOF_SPLITS),
                   'a': nc.model_signature(lf[la]()),
                   'b': nc.model_signature(lf[lb]())}
            both = nc.cached(
                u.ds, 'n42_two_learner_keep', key,
                lambda share=share: np.asarray(n32.two_learner_exchange(
                    u.X_tr, y0, u.g_tr, lf[la], lf[lb], 1.0 - share,
                    oof_splits=n32.OOF_SPLITS)[0], dtype=bool))
            d = _keep_dict(u.X_tr, y0, both)
        except Exception as exc:
            u.errors.add('n32/two_learner', exc)
        u.emit('n32', 'two_learner', f'share_kept={1.0 - share:g}',
               n07.NO_NEIGHBOURS, d)

        try:
            one, per, _rec = n32.shrink_flagged(y0, p, share)
        except Exception as exc:
            u.errors.add('n32/shrink_flagged', exc)
            one = per = None
        for name, col in (('shrink_flagged_one_weight', one),
                          ('shrink_flagged_per_row', per)):
            u.emit('n32', name, f'share={share:g}', n07.NO_NEIGHBOURS,
                   None if col is None
                   else n07.cleaned_from_labels(u.X_tr, col))


def _fam_n33(u, sim):
    p = u.p()
    if p is None:
        return
    y0 = np.asarray(u.y0, dtype=float)
    d = None
    try:
        s = n33.robust_scale(y0, p)
        if np.isfinite(s) and s > 0:
            # TODO(similarity-sweep 2026-09-25): unverified that NGBoost's fit
            # is seeded; a cached sigma is then one draw of it.
            key = {'X': nc.array_hash(u.X_tr), 'y': nc.array_hash(y0),
                   'groups': nc.array_hash(u.g_tr),
                   'n_splits': int(n33.OOF_SPLITS)}
            sig = nc.cached(u.ds, 'n42_ngboost_sigma', key,
                            lambda: np.asarray(n33.oof_ngboost_sigma(
                                u.X_tr, y0, u.g_tr, n33.OOF_SPLITS)))
            d = n07.cleaned_from_weights(u.X_tr, y0,
                                         n33.invvar_weights(sig, s))
        else:
            u.errors.add('n33/invvar_ngboost', 'robust scale is 0 or not finite')
    except Exception as exc:
        u.errors.add('n33/invvar_ngboost', exc)
    u.emit('n33', 'invvar_ngboost', f'sigma_floor={n33.SIGMA_FLOOR_S:g}',
           n07.NO_NEIGHBOURS, d)

    d = None
    try:
        rng = np.random.default_rng(n32.replicate_seed(
            'n42_cmixup', u.condition, f'{float(u.level):g}', u.draw, u.fold,
            u.mi))
        Xa, ya = n33.c_mixup(u.X_tr, y0, n33.CMIXUP_BW_SD * float(np.std(y0)),
                             rng)
        d = _rows_added_dict(Xa, ya)
    except Exception as exc:
        u.errors.add('n33/c_mixup', exc)
    u.emit('n33', 'c_mixup',
           f'bw_sd={n33.CMIXUP_BW_SD:g}|beta={n33.CMIXUP_BETA:g}',
           n07.NO_NEIGHBOURS, d)


def _fam_n35(u, sim):
    n_tr = len(u.y0)
    if n_tr > n35.MAX_PAIR_ROWS:
        u.unavailable.setdefault(
            'n35', f'{n_tr} training molecules is over n35.MAX_PAIR_ROWS='
                   f'{n35.MAX_PAIR_ROWS}')
        return
    y0 = np.asarray(u.y0, dtype=float)
    ms = (f'close_percentile={n35.CLOSE_PERCENTILE:g}|'
          f'weight={n35.DOWNWEIGHT:g}')

    def weights(D):
        rec, (pi, pj, pdy) = n35.pair_record(D, y0, None)
        flagged = n35.flag_pairs(pi, pj, pdy, rec['closest_bin_estimate'])
        return n35.pair_weights(n_tr, pi, pj, flagged)

    # The model input's own distance, as n35 computes it for its base cells
    # (n35:770-772): Tanimoto on a 0/1 representation's raw rows, Euclidean on
    # the standardised matrix otherwise. The hybrid is not 0/1.
    d = None
    try:
        rn = u.base_rep_name
        if rn is not None and n35.is_binary(u.reps[rn]):
            D = n35.tanimoto_distance_matrix(
                np.asarray(u.reps[rn])[np.asarray(u.train_idx)])
        else:
            D = n35.euclidean_distance_matrix(u.X_tr)
        d = n07.cleaned_from_weights(u.X_tr, y0, weights(D))
        del D
    except Exception as exc:
        u.errors.add('n35/downweight_pairs|model_input', exc)
    u.emit('n35', 'downweight_pairs', ms, ss.MODEL_INPUT.name, d)

    fsim = u.fold_similarity()
    for s in sim.settings:
        if s.space == ss.MODEL_INPUT_SPACE:
            continue
        d = None
        try:
            D, why = fsim.full_distance(s, u.model, u.X_tr, dtype=np.float32)
            if D is None:
                u.unavailable.setdefault(f'{u.model}|{s.name}|n35', why)
                continue
            d = n07.cleaned_from_weights(u.X_tr, y0, weights(D))
            del D
        except Exception as exc:
            u.errors.add('n35/downweight_pairs', exc)
        u.emit('n35', 'downweight_pairs', ms, s.name, d)


def _fam_n41(u, sim):
    p = u.p()
    if p is None:
        return
    y0 = np.asarray(u.y0, dtype=float)
    n = len(y0)
    try:
        score, _sigma = n41.rescaled_residual(u.X_tr, y0, u.g_tr, p,
                                              n41.OOF_SPLITS, dataset_name=u.ds)
    except Exception as exc:
        u.errors.add('n41/rescaled_residual', exc)
        return
    for sh in n41.CALIBRATION_SHARES:
        cal = n41.random_calibration(n, sh, n32.replicate_seed(
            'cal', f'{float(u.level):g}', u.draw, u.fold, n, sh))
        if int(cal.sum()) < n41.MIN_CALIBRATION or not (~cal).any():
            u.unavailable.setdefault(
                f'n41|{n41.random_cal_name(sh)}',
                f'calibration set of {int(cal.sum())} molecules is under '
                f'MIN_CALIBRATION={n41.MIN_CALIBRATION}')
            continue
        pv = n41.conformal_pvalues(score[cal], score[~cal])
        for q in n41.BH_QS:
            flags = np.zeros(n, dtype=bool)
            flags[np.flatnonzero(~cal)[n41.benjamini_hochberg(pv, q)]] = True
            if not flags.any():
                # n41 writes no arm when nothing is flagged (no_flag_arms);
                # counted here instead of a row identical to D.
                u.counts['n41_no_flags'] = u.counts.get('n41_no_flags', 0) + 1
                continue
            base = f'cal={n41.random_cal_name(sh)}|q={float(q):g}'
            u.emit('n41', 'winsorise', f'{base}|c={n41.WINSOR_C:g}',
                   n07.NO_NEIGHBOURS,
                   n07.cleaned_from_labels(
                       u.X_tr, n41.winsorise_flagged(y0, p, flags)))
            u.emit('n41', 'remove', base, n07.NO_NEIGHBOURS,
                   _keep_dict(u.X_tr, y0, ~flags))


FAMILY_FNS = {'n07': _fam_n07, 'n02': _fam_n02, 'n23': _fam_n23,
              'n30': _fam_n30, 'n31': _fam_n31, 'n32': _fam_n32,
              'n33': _fam_n33, 'n35': _fam_n35, 'n41': _fam_n41}


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run(data, config, output_dir, dataset_name, **kwargs):
    """N42 across every condition this dataset can be given
    (conditions_for_task), one result file and one rows file per condition,
    through noise_helpers.sweep_conditions. Called by noise_helpers.run_all as
    run(data, config, output_dir, dataset_name[, noise_condition=...]
    [, force_rerun=...])."""
    return sweep_conditions(
        _run_one_condition, EXP_ID, data, config, output_dir, dataset_name,
        default_conditions=conditions_for_task(data), kwargs=kwargs,
        groups_needed=GROUPED_STRATEGIES)


def _recipes_for(by_model, models):
    """{model: {space: (Blueprint, aug_condition)}} and {model: {space: why}},
    as n07 builds them (n07's per-family loop): the recipe copied before its
    augmentation strategy is set, because the object is shared with every other
    reader of the pool in this process."""
    recipes, unavail = {}, {}
    for m in models:
        meta = (by_model.get(m) or {}).get('meta')
        try:
            rs, un = pool_recipes_by_family(
                meta, SPACE_FOR_FAMILY,
                eligible_families=POOL_SUBSETS[POOL_SUBSET])
        except Exception as exc:
            recipes[m] = {}
            unavail[m] = {'_all': f'{type(exc).__name__}: {exc}'}
            continue
        per = {}
        for r in rs:
            bp = r['blueprint']
            aug = r['aug_condition'] or 'uncapped'
            if getattr(bp, 'augmentation_strategy', None) != aug:
                try:
                    bp = copy.copy(bp)
                    bp.augmentation_strategy = aug
                except Exception:
                    pass
            per[r['space']] = (bp, aug)
        recipes[m] = per
        if un:
            unavail[m] = dict(un)
    return recipes, unavail


def _cliff_masks(data, dataset_name, y0, train_idx, is_clf, fold_i, draw,
                 structure):
    """n25's exempt sets for one pass: the cliff molecules among the training
    rows, from the labels as they look (n25:1407-1420), and as many random
    non-cliff ones. The mask is cached on the labels and the rows."""
    smiles = data.get('smiles')
    key = {'y': nc.array_hash(np.asarray(y0)),
           'train_idx': nc.array_hash(np.asarray(train_idx)),
           'smiles': (None if smiles is None else nc.array_hash(
               np.asarray(list(smiles), dtype=object))),
           'is_clf': bool(is_clf),
           'sim_threshold': float(n25.CLIFF_SIM_THRESHOLD),
           'activity_k': float(n25.CLIFF_ACTIVITY_K), 'mode': 'mad'}
    cliff = nc.cached(dataset_name, 'n42_cliff_mask', key, lambda: np.asarray(
        n25.cliff_mask_for_dataset(data, np.asarray(y0), is_clf,
                                   subset=train_idx, structure=structure)[0],
        dtype=bool))
    rand, _n = n25.random_exempt_mask(cliff, int(cliff.sum()), fold_i, draw)
    return cliff, rand


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    print("\n" + "=" * 70)
    print("EXPERIMENT N42: EVERY MITIGATION METHOD x SIMILARITY SETTING x "
          "MODEL INPUT")
    print("=" * 70)
    t_start = time.time()

    task_type = data['task_type']
    is_clf = task_type == 'classification'
    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    augmentations = data.get('augmentations')
    n_folds = int(config['n_splits'])
    metric = get_domain_metric(data)
    metric_name, _metric_fn, lower_is_better = metric
    condition = kwargs.get('noise_condition') or conditions_for_task(data)[0]

    models = list(kwargs.get('models') or config.get('models')
                  or DOWNSTREAM_MODELS_NOISE)
    factories = get_model_factories(models, task_type=task_type,
                                    n_samples=len(labels))
    models = [m for m in models if m in factories]
    families = families_asked(kwargs)
    skipped_families = {}
    if is_clf:
        for f in families:
            if f in REGRESSION_ONLY_FAMILIES:
                skipped_families[f] = ('reads an out-of-fold residual on a '
                                       'numeric label; its own file runs '
                                       'regression only')
        families = tuple(f for f in families if f not in skipped_families)
    learner_factories = {}
    if 'n32' in families:
        learner_factories = get_model_factories(
            list(n32.EXCHANGE_LEARNERS), task_type=task_type,
            n_samples=len(labels))

    # ── The precompute: each model's recipes and its top-2 single reps ──
    lookup = functools.partial(
        get_best_search_for_downstream, reps, labels, groups,
        budget=kwargs.get('budget'), augmentations=augmentations,
        dataset_name=dataset_name, task_type=task_type, data=data,
        pool_subset=POOL_SUBSET)
    by_model, no_recipe = pool_recipe_per_downstream_model(lookup, models)
    recipes, recipes_unavailable = _recipes_for(by_model, models)
    try:
        base_by_model, base_missing = noise_base_reps_by_model(
            {m: by_model[m]['search'] for m in by_model}, models, reps=reps,
            dataset_name=dataset_name, verbose=False)
    except Exception as exc:
        base_by_model = {}
        base_missing = {'_all': f'{type(exc).__name__}: {exc}'}
    for m in models:
        print(f"  {m}: base {base_by_model.get(m)}, recipe spaces "
              f"{sorted(recipes.get(m, {}))}")

    settings = n07.sweep_settings()
    ks = n07.sweep_ks()
    passes = n23._levels_and_draws(dataset_name, condition, is_clf,
                                   draws=NOISE_DRAWS)
    noise_groups = noise_groups_for(data)[0]
    mcounts = None
    if 'n07' in families:
        try:
            mcounts = n07.measurement_counts_for_rows(data)
        except Exception:
            mcounts = None
    smiles = data.get('smiles')

    # n25's structural reference over every molecule; it reads no label.
    cliff_structure = None
    cliff_error = None
    if 'n07' in families:
        try:
            _m, _info = n25.cliff_mask_for_dataset(data, labels, is_clf)
            cliff_structure = _info.pop('similarity_matrix', None)
        except Exception as exc:
            cliff_error = f'{type(exc).__name__}: {exc}'

    sink = n07.RowSink(output_dir, EXP_ID, dataset_name, condition)
    fingerprint = {'input': input_fingerprint(data, config),
                   'settings': [s.name for s in settings], 'ks': list(ks),
                   'families': list(families), 'models': list(models),
                   'controls': bool(controls_enabled()),
                   'condition': condition}
    units = n07.FoldCheckpoint(output_dir, EXP_ID, dataset_name, condition,
                               tag='units', fingerprint=fingerprint)
    errors = _Errors()
    unavailable = {}
    # prox_smooth reads forest-proximity neighbours and nothing else here
    # (_fam_n07 skips it in every other space). Under a settings list without
    # forest_proximity, KIRBY_SIMILARITY_SETTINGS=model_input included, it
    # writes no row at all; say so rather than leave it silently absent.
    if 'n07' in families and not any(s.space == ss.FOREST_SPACE
                                     for s in settings):
        unavailable['n07/prox_smooth'] = (
            'runs only under the forest_proximity similarity setting, which '
            'is not among the settings this run sweeps')
    counts = {}
    provenance = {}
    n_units_done = n_units_resumed = 0

    print(f"  condition {condition}; {len(passes)} (level, draw) passes; "
          f"{len(settings)} similarity settings x ks {list(ks)}; families "
          f"{list(families)}; models {models}; rows -> {sink.path}")
    if cliff_error:
        print(f"  n25 cliff reference unavailable: {cliff_error}")

    sample_X = next(iter(reps.values()))
    mi_settings = [ss.SimilaritySetting(sp, 'none', 'manhattan')
                   for sp in MODEL_INPUTS]

    for fold_i, (train_idx, val_idx) in enumerate(
            cv_split(groups, n_folds, sample_X, labels)):
        train_idx = np.asarray(train_idx)
        val_idx = np.asarray(val_idx)
        y_clean = labels[train_idx]
        y_val = labels[val_idx]
        g_tr = None if groups is None else np.asarray(groups)[train_idx]
        smi_tr = None if smiles is None else [smiles[i] for i in train_idx]
        smi_va = None if smiles is None else [smiles[i] for i in val_idx]
        mc_tr = None if mcounts is None else np.asarray(mcounts)[train_idx]

        for level, draw in passes:
            todo = [(m, mi) for m in models for mi in MODEL_INPUTS
                    if units.get([fold_i, float(level), int(draw), m, mi])
                    is None]
            n_units_resumed += len(models) * len(MODEL_INPUTS) - len(todo)
            if not todo:
                continue
            try:
                spec = noise_spec_for(dataset_name, condition, level,
                                      noise_seed=draw, is_clf=is_clf)
                y0, _cmask, prov = inject_train_labels(
                    labels, spec, train_idx=train_idx, is_clf=is_clf,
                    groups=noise_groups, dataset_name=dataset_name)
                provenance.setdefault(spec.spec_id, convert_numpy(prov))
            except Exception as exc:
                errors.add(f'inject|level={level:g}', exc)
                continue
            y0 = np.asarray(y0)

            cliff_train = rand_train = None
            if 'n07' in families and cliff_error is None:
                try:
                    cliff_train, rand_train = _cliff_masks(
                        data, dataset_name, y0, train_idx, is_clf, fold_i,
                        int(draw), cliff_structure)
                except Exception as exc:
                    errors.add('n25/cliff_mask', exc)

            for m in models:
                mis = [mi for (mm, mi) in todo if mm == m]
                if not mis:
                    continue
                t0 = time.time()
                # TODO(similarity-sweep 2026-09-25): n07.fold_similarity_spaces
                # hands the recipe's augmentation donors to the hybrid build
                # when its aug_condition is not 'none'; n23 (cached_hybrid_cell)
                # and n32 (FoldSimilarity._hyb) build the same recipe with no
                # donors. So n42's hybrid is not the same matrix as n23's and
                # n32's under an augmented recipe. Which one is right is the
                # owner's call; n42 follows n07.
                try:
                    sp_tr, sp_va, sp_unavail = n07.fold_similarity_spaces(
                        dataset_name, list(settings) + mi_settings,
                        model_input_tr=None, model_input_va=None, reps=reps,
                        train_idx=train_idx, val_idx=val_idx, y_train=y0,
                        groups_train=g_tr, base_reps=base_by_model.get(m, []),
                        recipes=recipes.get(m, {}),
                        augmentations=augmentations, model_name=m,
                        model_factory=factories.get(m), metric=metric,
                        task_type=task_type)
                except Exception as exc:
                    errors.add(f'spaces|{m}', exc)
                    continue
                shared_lists = {}
                fsim_box = {}
                base_names = list(base_by_model.get(m, []))

                for mi in mis:
                    # Each unit's own time; the first also carries the spaces
                    # build above. Until 2026-09-28 t0 was set once per model,
                    # so every later unit's seconds included the earlier ones.
                    if mi != mis[0]:
                        t0 = time.time()
                    ukey = [fold_i, float(level), int(draw), m, mi]
                    X_tr = sp_tr.get(mi)
                    if X_tr is None:
                        unavailable.setdefault(
                            f'{m}:{mi}', sp_unavail.get(mi)
                            or (base_missing.get(m) if mi.startswith('base')
                                else (recipes_unavailable.get(m) or {})
                                .get(mi) or (no_recipe.get(m) or {}).get('why'))
                            or 'not built')
                        continue
                    X_va = sp_va[mi]
                    mi_label = f'{m}:{mi}'
                    spaces_tr = dict(sp_tr, model_input=X_tr)
                    spaces_va = dict(sp_va, model_input=X_va)

                    def _fsim(_m=m, _tr=spaces_tr, _va=spaces_va,
                              _un=sp_unavail, _y=y0, _tri=train_idx,
                              _vai=val_idx, _g=g_tr):
                        # n32.FoldSimilarity with its feature spaces set to
                        # the matrices n07.fold_similarity_spaces built, so
                        # n35's full distances are taken in exactly the
                        # spaces the neighbour lists above were found in.
                        if 'f' not in fsim_box:
                            f = n32.FoldSimilarity(
                                dataset_name, data, reps=reps, train_idx=_tri,
                                val_idx=_vai, y_train=_y, groups_train=_g,
                                base_by_model={_m: base_names},
                                recipe_for_space_by_model={
                                    _m: {k: v[0] for k, v in
                                         recipes.get(_m, {}).items()}},
                                model_factories=factories, metric=metric,
                                task_type=task_type,
                                augmentations=augmentations, smiles=smiles,
                                is_clf=is_clf)
                            for sp in ss.FEATURE_SPACES:
                                if sp in _tr:
                                    f._spaces[(_m, sp)] = (_tr[sp], _va[sp],
                                                           None)
                                else:
                                    why = _un.get(sp, 'not built')
                                    f._spaces[(_m, sp)] = (None, None, why)
                                    f.unavailable[f'{_m}|{sp}'] = why
                            fsim_box['f'] = f
                        return fsim_box['f']

                    u = _Unit(
                        ds=dataset_name, sink=sink, metric=metric,
                        task_type=task_type, is_clf=is_clf,
                        condition=condition, level=float(level),
                        draw=int(draw), fold=int(fold_i), model=m, mi=mi_label,
                        X_tr=X_tr, X_va=X_va, y0=y0, y_clean=y_clean,
                        y_val=y_val, g_tr=g_tr, train_idx=train_idx,
                        val_idx=val_idx, reps=reps, prov=prov,
                        cliff_train=cliff_train, rand_train=rand_train,
                        measurement_counts=mc_tr,
                        row_models=n07.row_models(
                            [m] if m in CACHEABLE_MODELS else []),
                        learner_factories=learner_factories,
                        base_rep_name=(base_names[int(mi[-1]) - 1]
                                       if mi.startswith('base_')
                                       and len(base_names) >= int(mi[-1])
                                       else None),
                        errors=errors, unavailable=unavailable, counts=counts,
                        fold_similarity=_fsim)
                    sim = _Similarity(u, settings, ks, spaces_tr, spaces_va,
                                      shared_lists, sp_unavail, smi_tr, smi_va)
                    for fam in families:
                        try:
                            FAMILY_FNS[fam](u, sim)
                        except Exception as exc:
                            errors.add(f'{fam}|unit', exc)
                    n_rows = u.rows_written()
                    units.put(ukey, {'rows_written': int(n_rows),
                                     'seconds': round(time.time() - t0, 1)})
                    n_units_done += 1
                    print(f"    fold {fold_i} level {level:g} draw {draw} "
                          f"{mi_label}: {n_rows} rows, "
                          f"{time.time() - t0:.0f}s")

    # TODO(similarity-sweep 2026-09-25): the pass's own model is added to the
    # row models only when its fit is seeded (CACHEABLE_MODELS). dnn rows are
    # therefore rf and svm only; fitting dnn through cached_fit_predict would
    # store one draw of an unseeded fit.
    results = {
        'metric': metric_name, 'lower_is_better': bool(lower_is_better),
        'noise_condition': condition, 'task_type': task_type,
        'passes': [[float(lv), int(dr)] for lv, dr in passes],
        'n_folds': n_folds, 'models': models,
        'model_inputs': list(MODEL_INPUTS),
        'base_reps_by_model': base_by_model, 'base_reps_missing': base_missing,
        'recipe_spaces_by_model': {m: sorted(v) for m, v in recipes.items()},
        'recipe_spaces_unavailable': recipes_unavailable,
        'models_without_recipe': no_recipe,
        'families_run': list(families),
        'families_skipped': skipped_families,
        'flag_rates': list(FLAG_RATES),
        'flag_shares_n31_n32': list(FLAG_SHARES),
        'n07_detectors': list(N07_DETECTORS),
        'methods_not_run': METHODS_NOT_RUN,
        'covered_by_another_family': COVERED_BY_ANOTHER_FAMILY,
        'n25_cliff_reference_error': cliff_error,
        'units_computed_this_run': int(n_units_done),
        'units_resumed': int(n_units_resumed),
        'counts': counts,
        'errors': errors.d,
        'unavailable': unavailable,
        'noise_provenance': provenance,
        'similarity': n07.similarity_record(settings, sink),
        'seconds': round(time.time() - t_start, 1),
    }
    print(f"\n  N42 {condition}: {n_units_done} units computed, "
          f"{n_units_resumed} resumed, {sink.n_written} rows written, "
          f"{len(errors.d)} distinct errors, {time.time() - t_start:.0f}s")
    return results


def main():
    import argparse
    from experiments.noise.noise_helpers import run_all
    ap = argparse.ArgumentParser(description='N42 on one dataset through '
                                             'noise_helpers.run_all')
    ap.add_argument('dataset')
    ap.add_argument('--output', required=True)
    ap.add_argument('--noise-condition', default=None)
    ap.add_argument('--pruning-mode', default='calibrated')
    args = ap.parse_args()
    run_all(dataset_name=args.dataset, output_dir=args.output,
            experiments=['n42'], pruning_mode=args.pruning_mode,
            noise_condition=args.noise_condition)


if __name__ == '__main__':
    main()
