#!/usr/bin/env python3
"""N46, structure-specific fixes: the fixes that need something only one
noise kind or one dataset has.

HANDOFF_NOISE_RESTRUCTURE.md, section 2 (working name ST), sections 4.1 (its
ST rows), 4.4 and 5. Written 2026-11-02. Replaces n19, n26, n29, n30, n30b,
n34, n35 and n39; those files stay in place and every function below that
does the arithmetic is imported from them, not rewritten.

WHAT ONE ROW IS. d_vs_dprime's row, written through n07.dprime_rows, so every
row carries the dataset's own metric and all_D / all_Dprime / all_clean
(handoff section 11). D is the configuration's own model fitted on the
training labels the fix starts from; D' is the same model on the training
set after the fix; both are scored on the same held-out rows. Where a fix
changes the model or the features instead of the training set (the censored
likelihood, the robust-loss swaps, the count feature, NGBoost, the GP noise
term, the interval fit, the ranking within each paper), the row is built from
the two predictions with d_vs_dprime.make_row and d_vs_dprime._all_scores, so
it carries the same metrics. `method` is "<family>/<fix>".

THE FAMILIES. Each fix runs only where the structure it reads exists.
Everywhere else it writes "not applicable" with the reason, into the unit's
checkpoint line and the result file (`not_applicable`), never a silent skip.

  censoring       n30. Needs labels piled on a top value. Runs under the
                  injected `censoring` kind, at every decided level including
                  0 (where a pile already in the labels as delivered is read
                  as a limit, n30's rule). Fixes: winsorise at n30.WINSOR_C;
                  constant_lift, per_molecule_lift, its three shuffled
                  controls, truncated_normal_mean (n30.lift_columns);
                  censored_em (n30.censored_em_column, the model's own
                  out-of-fold fit, any model); drop_clipped; n30b's
                  not_at_floor_control (the same number of labels NOT at the
                  limit moved up by constant_lift's amount); and the model
                  swaps xgboost_aft (n30.fit_aft_predict, the censored
                  likelihood), rf_l1, lightgbm_huber, each only in its parent
                  model's units (n30's rule, 2026-10-06).
                  The injected-censoring check of n30b (--inject-censoring,
                  where the truth is known) is this family under `censoring`:
                  the limit is read off the clipped training labels and the
                  clean fit is score_clean. It runs on every regression
                  dataset, so QM9's three targets (CLEAN_BASE_DATASETS) are
                  the comparison point; the result file says so.
  real_censoring  n30b on labels as delivered with a floor. The same fixes
                  through n30b's negated-label functions (a floor becomes a
                  pile at the top), the not-at-floor control moving labels
                  down, and n30b's three scores (Spearman on every held-out
                  row, above the floor, Harrell's C) in the checkpoint.
                  Applies only to a dataset whose floor the author has
                  accepted as a limit (REAL_CENSORING_LIMITS). On any other
                  dataset a pile on the lowest or highest label is counted
                  and written as not applicable with the reason.
  grouped         n34. Needs scaffold groups (noise_groups_for). Runs under
                  grouped_shifted, which draws one offset per group, and under
                  gaussian, the negative control with no group offset; level 0
                  of both is the real labels. Fixes: group_center,
                  group_center_3plus, random_intercept (MERF-style EM). Offset
                  recovery against the recomputed injected offsets is in the
                  checkpoint under grouped_shifted.
  close_pairs     n35. Needs pairwise distances among the training rows (up
                  to n35.MAX_PAIR_ROWS). Runs under every kind and level. Per
                  neighbour setting kept by the screen (the three best
                  distinct settings, n42._Similarity.kept_settings): the pair
                  record with the noise estimate that reads no model
                  (intercept_estimate, closest_bin_estimate), the pair test
                  (|dy| over 3 sqrt(2) sigma, sigma^2 the intercept) and the
                  pair weights (both molecules of a flagged pair at 0.25). The
                  realised injected variance is written beside the estimate.
  sources         n39. Its own injected structure, run as the pass
                  'sources': N_SOURCES simulated sources, a count
                  distribution, sigma_e on the decided gaussian levels, rho in
                  n39.RHOS, tau in n39.TAU_FRACS, three draws. Fixes:
                  offset_removed (mixed model, n39.fit_offsets),
                  weights_n19_fitted, weights_within_var (the 1/(...) weight
                  from the labels and the model's error),
                  offset_removed_within_var_weights, and the two oracle
                  weights. On a dataset with repeat measurements the fold's
                  real-data block (pair residual correlation, an upper bound
                  on rho) is in the checkpoint.
  repeats         n19, n26 and n29, the pass 'repeat_measurements'. Needs
                  replicate groups (herg_ic50, bindingdb_kd, kiba); n29's
                  fixes also need a document per measurement
                  (record_documents, herg_ic50 only).
                    n19: the six training-label fixes (count, spread and
                    precision weights, shrunk, calibrated) and the four added
                    2026-09-24 (invvar_weighted, count_feature,
                    invvar_and_count_feature, spread_pooled_weighted), under
                    both label builds, single and averaged, three draws of
                    which measurement is the single label; the model pairs
                    ngboost_count_feature against ngboost_plain,
                    gp_per_molecule_noise against gp_plain and
                    xgboost_interval_aft against xgboost_aft_point, in the
                    first decided model's units.
                    n26: error by measurement count of the plain model's
                    held-out residuals, per unit and pooled over folds, with
                    the three 1/count fits.
                    n29: per_record (one row per measurement),
                    per_record_bias, mean_label_bias_corrected,
                    per_record_random_effect, document_quantile_map (the
                    per-paper quantile map); the offsets' spread, sign flips
                    and level correlation; the check with known biases on the
                    first fold run, base_1, first model. within_document_rank
                    (ranking within each paper) runs only in an xgboost unit;
                    herg_ic50's models are svm and dnn, so it has never run,
                    and every unit writes that as not applicable.

THE LOOPS. run() sweeps, through noise_helpers.sweep_conditions, every decided
kind (noise_helpers.decided_conditions) and then the three structure passes
STRUCTURE_PASSES, one result file and one rows file each. Inside an injected
kind: fold (noise_outer_folds, all 5), level (decided_levels, 0 first), draw
(draws_for: 3 per non-zero level, 1 under censoring, level 0 once), model (the
dataset's two decided models), model input (base_1, hybrid). The
hybrid is the per-fold search's first general hybrid rebuilt from the labels
the fix starts from (fold_model_inputs), as in n44.

ONE ROW PER MOLECULE. On a dataset whose loader keeps repeat measurements
(herg_ic50), every pass runs on noise_helpers.molecule_mean_data (one row per
molecule, labelled with its mean) and files its per-fold searches under
MOLECULE_MEAN_SEARCH_SUFFIX, as n19, n26, n29 and n39 did; n29's per-record
fixes expand back to the measurements of the training molecules.

RESUME. n07.RowSink skips every row already on disk; each finished unit is in
the .units checkpoint under the run's fingerprint.

NOT CARRIED, and why. n35's count of activity cliffs among flagged pairs and
n29's 'isolated' held-out view are measurements, not fixes; the isolated
view's scores are in the checkpoint. n34's and n39's tail scores (worst tenth
of held-out molecules) are not written: every row's all_* metrics replace the
one-metric summaries those files kept.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments._shared import (
    get_domain_metric, get_model_factories, input_fingerprint, convert_numpy,
    fold_standard_selection,
)
from experiments.decided_grid import decided_models
from experiments.dataset_policy import noise_groups_for
from experiments.noise.noise_helpers import (
    sweep_conditions, decided_conditions, noise_outer_folds,
    fold_configurations, fold_model_inputs, configuration_record,
    decided_levels, inject_condition, dataset_knn_group, level_unit,
    draws_for, MOLECULE_MEAN_SEARCH_SUFFIX,
    CLEAN_BASE_DATASETS,
)
from experiments.noise._ground_truth_providers import build_replicate_groups
from kirby.noise_spec import GROUPED_STRATEGIES

from experiments.noise import noise_cache as nc
from experiments.noise import similarity_space as ss
from experiments.noise import d_vs_dprime as dvd
from experiments.noise import n07_remediation as n07
from experiments.noise import n17_replicate_removal as n17
from experiments.noise import n19_measurement_count as n19
from experiments.noise import n26_noise_variance_from_repeats as n26
from experiments.noise import n29_document_bias as n29
from experiments.noise import n30_censored_likelihood as n30
from experiments.noise import n30b_real_censoring as n30b
from experiments.noise import n32_iterative_and_dual_selection as n32
from experiments.noise import n34_group_random_effects as n34
from experiments.noise import n35_near_duplicate_disagreement as n35
from experiments.noise import n39_correlated_sources as n39
from experiments.noise import n42_similarity_noise_sweep as n42
from experiments.noise.controls import controls_enabled
from experiments.noise import noise_timing
from experiments.noise.noise_helpers import (
    PAPER3_CUT_CONDITIONS, without_paper3_cuts)

EXP_ID = 'n46_structure_specific_fixes'
RESULT_LAYOUT = 'structure_fixes_2026-11-02'

NOISE_DRAWS = (0, 1, 2)
MODEL_INPUTS = ('base_1', 'hybrid')  # Adelaide, 2026-11-04: "okay then the cut to all"

# The passes swept after the decided kinds. Each injects its own structure or
# reads one the loader carries.
SOURCES_PASS = 'sources'
REPEATS_PASS = 'repeat_measurements'
REAL_CENSORING_PASS = 'real_censoring'
STRUCTURE_PASSES = (SOURCES_PASS, REPEATS_PASS, REAL_CENSORING_PASS)

# n34's conditions: grouped_shifted carries a group offset; gaussian is the
# negative control (DECIDED 2026-10-06, Adelaide delegated).
GROUPED_CONDITIONS = tuple(n34.CONDITIONS)

# Floors the author has accepted as a limit in the labels as delivered.
# ginkgo_pr_cho: 2026-10-09 (HANDOFF_CENSORING_PRCHO.md). It is out of Papers
# 2 and 3 (plan.py, 2026-11-01), so on the decided datasets this pass writes
# not applicable. adme_hlm and adme_rlm: accepted 2026-11-02 (Adelaide). They
# pile on their minimum: 958 of 3,087 molecules at log CLint 0.676, and 346
# of 3,054 at 1.03. Neither the Biogen repo README nor the Polaris page states
# a limit of quantitation, and the paper's methods were not read, so the
# floor is taken from the data, not from a published assay range.
REAL_CENSORING_LIMITS = {'ginkgo_pr_cho': 'lower', 'adme_hlm': 'lower',
                         'adme_rlm': 'lower'}

# n30's model swaps and the model each replaces.
CENSORING_SWAPS = (('xgboost_aft', n30.AFT_PARENT),
                   ('rf_l1', 'rf'), ('lightgbm_huber', 'lightgbm'))

# The noise_condition column of the repeat-measurement rows: nothing
# injected, the laboratories disagree (n19.ROW_CONDITION).
REPEATS_ROW_CONDITION = n19.ROW_CONDITION
N29_SETTING = f'ridge_alpha={n29.RIDGE_ALPHA:g}|n_iter={n29.N_ITER}'
N35_SETTING = (f'close_percentile={n35.CLOSE_PERCENTILE:g}|'
               f'weight={n35.DOWNWEIGHT:g}|sigma2=intercept_estimate')
NO_NB = n07.NO_NEIGHBOURS


# ---------------------------------------------------------------------------
# one unit: the place every row is written
# ---------------------------------------------------------------------------

class _Unit:
    """Everything one unit's fixes read. The attributes n42._Similarity reads
    (ds, y0, is_clf, metric, y_val, g_tr, unavailable, model, errors) are
    here under the same names."""

    def __init__(self, *, ds, sink, metric, task_type, condition, level,
                 draw, fold, model, mi, X_tr, X_va, y0, y_clean, y_val, g_tr,
                 factory, takes_weight, errors, na_log, unavailable,
                 swap_factories=None, prov=None):
        self.ds, self.sink, self.metric, self.task_type = ds, sink, metric, task_type
        self.is_clf = task_type == 'classification'
        self.condition, self.level, self.draw, self.fold = condition, level, draw, fold
        self.model, self.mi = model, mi
        self.X_tr, self.X_va = X_tr, X_va
        self.y0 = np.asarray(y0, dtype=float)
        self.y_clean = None if y_clean is None else np.asarray(y_clean, dtype=float)
        self.y_val = np.asarray(y_val, dtype=float)
        self.g_tr = g_tr
        self.factory, self.takes_weight = factory, takes_weight
        self.errors, self.na_log, self.unavailable = errors, na_log, unavailable
        self.swap_factories = swap_factories or {}
        self.prov = prov or {}
        self.na = {}
        self.record = {}
        self.n_rows_before = sink.n_written
        self._p, self._p_done = None, False

    # the out-of-fold LightGBM prediction p every n30 arm reads
    def p(self):
        if not self._p_done:
            self._p_done = True
            try:
                self._p = nc.cached_oof_predict(
                    self.ds, self.X_tr, self.y0, self.g_tr,
                    n_splits=min(int(n30.OOF_SPLITS), len(self.y0)))
            except Exception as exc:
                self.errors.add('oof_prediction', exc)
        return self._p

    def pred(self, X, y, X_va=None, w=None):
        """The unit's model fitted on (X, y[, w]), predicting X_va (default
        the unit's held-out rows), through the noise cache."""
        return np.asarray(nc.cached_fit_predict(
            self.ds, self.model, self.factory, X, np.asarray(y),
            self.X_va if X_va is None else X_va, sample_weight=w,
            predict_mode=nc.predict_mode_for(self.metric[0])), dtype=float)

    def _ident(self, family, method, setting, sim):
        return n07.row_identity(
            dataset_name=self.ds, experiment=EXP_ID,
            method=f'{family}/{method}', method_setting=setting,
            similarity_setting=sim, model_input=self.mi,
            noise_condition=self.condition, level=self.level, draw=self.draw,
            fold=self.fold)

    def emit(self, family, method, setting, cleaned, sim=NO_NB):
        """One D' that changes the training set: n07.dprime_rows. None writes
        the row with score_Dprime nan, so the cell is on disk."""
        try:
            n07.dprime_rows(
                self.sink, dataset_name=self.ds, cleaned=cleaned,
                X_tr=self.X_tr, y_D=self.y0, X_va=self.X_va, y_va=self.y_val,
                metric=self.metric, task_type=self.task_type,
                y_clean=self.y_clean, models=(self.model,),
                identity=self._ident(family, method, setting, sim),
                is_clf=self.is_clf)
        except Exception as exc:
            self.errors.add(f'{family}/{method}|fit', exc)

    def emit_pred(self, family, method, setting, model_label, compute,
                  sim=NO_NB):
        """One fix that changes the model or the features. compute() returns
        (pred_D, pred_Dprime, pred_clean[, why not run]); any may be None.
        The row is d_vs_dprime.make_row with every metric of ALL_METRICS."""
        ident = self._ident(family, method, setting, sim)
        if self.sink.is_done(ident, (model_label,)):
            return
        why = None
        try:
            got = compute()
        except Exception as exc:
            self.errors.add(f'{family}/{method}|fit', exc)
            got, why = (None, None, None), f'{type(exc).__name__}: {exc}'
        pD, pP, pC = got[:3]
        if len(got) > 3 and got[3]:
            why = got[3]
        name, fn, lib = self.metric

        def sc(p):
            if p is None:
                return float('nan'), None
            p = np.asarray(p, dtype=float)
            return (float(fn(self.y_val, p)),
                    dvd._all_scores(self.y_val, p, name, self.task_type))
        (sD, aD), (sP, aP), (sC, aC) = sc(pD), sc(pP), sc(pC)
        try:
            row = dvd.make_row(**ident, model=model_label, metric=name,
                               lower_is_better=lib, score_D=sD,
                               score_Dprime=sP, score_clean=sC, all_D=aD,
                               all_Dprime=aP, all_clean=aC,
                               dprime_not_run=why)
            self.sink.write([row])
        except Exception as exc:
            self.errors.add(f'{family}/{method}|row', exc)

    def not_applicable(self, family, reason):
        self.na[family] = reason
        d = self.na_log.setdefault(family, {})
        d[reason] = d.get(reason, 0) + 1

    def rows_written(self):
        return self.sink.n_written - self.n_rows_before


def _label_dict(X, y):
    return n07.cleaned_from_labels(X, y)


def _rows_dict(X, y, w=None):
    """A D' whose rows are not D's rows (one row per measurement): no keep,
    no label column, so controls.py writes no twin for it (n42's rule)."""
    return {'X': np.asarray(X), 'y': np.asarray(y, dtype=float),
            'sample_weight': None if w is None else np.asarray(w, dtype=float),
            'keep': None, 'label_column': None}


# ---------------------------------------------------------------------------
# censoring (n30) and real censoring (n30b)
# ---------------------------------------------------------------------------

def censoring_fixes(u, side='upper'):
    """n30's fixes on labels clipped at the top (side 'upper', the injected
    kind) or n30b's on labels with a floor (side 'lower', the labels as
    delivered). The limit is read off the training labels, never the
    injector (n30, 2026-09-28)."""
    fam = 'censoring' if side == 'upper' else 'real_censoring'
    if u.is_clf:
        u.not_applicable(fam, 'a class label has no value beyond a limit')
        return
    if side == 'upper' and u.condition != 'censoring':
        u.not_applicable(fam, f'{u.condition} clips no label; every censoring '
                              f'fix reads the labels sitting at a limit')
        return
    y0 = u.y0
    if side == 'upper':
        limit = n30.limit_from_labels(y0)
        at = n30.rows_at_limit(y0, limit)
        direction = 1.0
    else:
        limit = n30b.floor_from_labels(y0)
        at = n30.rows_at_limit(-y0, None if limit is None else -limit)
        direction = -1.0
    rec = {'side': side, 'limit': limit,
           'limit_read_from': 'the training labels (n30.limit_from_labels'
                              + (', on the negated labels)' if side == 'lower'
                                 else ')'),
           'molecules_at_the_limit': int(at.sum()),
           'training_molecules': int(len(y0))}
    if side == 'upper':
        inj = u.prov.get('censoring_limit')
        rec['limit_from_injector_for_evaluation'] = inj
        if u.y_clean is not None and at.any() and limit is not None:
            rec['true_labels_above_the_limit_label_units'] = float(
                np.mean(u.y_clean[at]) - limit)
    u.record[fam] = rec
    p = u.p()
    if p is None:
        u.not_applicable(fam, 'the out-of-fold prediction every fix reads failed')
        return
    arms = {}
    # winsorise reads every label and no limit
    for c in n30.WINSOR_C:
        col, s, moved = n30.winsorise(y0, p, float(c))
        rec[f'winsor_c{c:g}'] = {'threshold_label_units': float(c * s),
                                 'labels_moved': int(moved)}
        arms[('winsorise', f'c={c:g}')] = _label_dict(u.X_tr, col)
    sd = n30.residual_sd(y0, p, at)
    rec['residual_sd_rows_not_at_limit'] = sd
    if limit is None or not at.any() or not np.isfinite(sd):
        u.not_applicable(f'{fam}/lift_drop_em_control',
                         'no limit in the training labels: n30.limit_from_labels '
                         'needs 1% of the training molecules, and at least 2, on '
                         'the extreme value')
        for sw, _p in CENSORING_SWAPS:
            if side == 'upper':
                u.not_applicable(f'{fam}/{sw}', 'no limit in the training labels')
    else:
        seed_key = (int(u.fold), int(round(float(u.level or 0.0) * 1000)),
                    int(u.draw or 0))
        if side == 'upper':
            cols, linfo = n30.lift_columns(y0, p, at, limit, sd,
                                           n30.SHUFFLE_SEEDS, seed_key)
        else:
            cols, linfo = n30b.lower_label_columns(y0, p, at, limit, sd,
                                                   n30.SHUFFLE_SEEDS, seed_key)
        rec.update(linfo)
        for arm, col in cols.items():
            meth, _, seed = arm.partition('/')
            arms[(meth, seed.replace('seed', 'seed=') or 'default')] = \
                _label_dict(u.X_tr, col)
        arms[('drop_clipped', 'default')] = n42._keep_dict(u.X_tr, y0, ~at)
        if u.g_tr is None:
            u.not_applicable(f'{fam}/censored_em',
                             'no fold groups for the out-of-fold refits')
        else:
            try:
                if side == 'upper':
                    f_ = u.factory
                    em, einfo = n30.censored_em_column(
                        lambda yy: n30._own_model_oof(f_, u.X_tr, yy, u.g_tr,
                                                      n30.OOF_SPLITS),
                        y0, at, limit)
                else:
                    em, einfo = n30b.lower_censored_em(
                        u.factory, u.X_tr, y0, at, limit, u.g_tr,
                        n30.OOF_SPLITS)
                rec.update(einfo)
            except Exception as exc:
                u.errors.add(f'{fam}/censored_em', exc)
                em = None
            arms[('censored_em', f'rounds={n30.CENSORED_EM_ROUNDS}')] = (
                None if em is None else _label_dict(u.X_tr, em))
        amount = linfo['mean_lift_label_units']
        for s in n30b.CONTROL_SEEDS:
            col, k = n30b.not_at_floor_control(y0, at, amount, s, seed_key,
                                               direction=direction)
            rec['not_at_floor_control_labels_moved'] = int(k)
            arms[('not_at_floor_control', f'seed={s}')] = _label_dict(u.X_tr, col)
        # the model swaps: upper limit only, each in its parent's units
        if side == 'lower':
            u.not_applicable(f'{fam}/model_swaps',
                             'n30.fit_aft_predict takes an upper limit only '
                             '(the interval [limit, inf)); n30b ran no swap')
        for sw, parent in (CENSORING_SWAPS if side == 'upper' else ()):
            if u.model != parent:
                u.not_applicable(f'{fam}/{sw}',
                                 f'replaces {parent}, which this unit does not '
                                 f'fit (n30: a swap runs in its parent model\'s '
                                 f'units only, 2026-10-06)')
                continue

            def _swap(_sw=sw):
                pD = u.pred(u.X_tr, y0)
                pC = None if u.y_clean is None else u.pred(u.X_tr, u.y_clean)
                if _sw == 'xgboost_aft':
                    pP = n30.fit_aft_predict(u.X_tr, y0, at, u.X_va, sd)
                else:
                    f = u.swap_factories.get(_sw)
                    if f is None:
                        return pD, None, pC, 'no model factory for this swap'
                    m = f()
                    m.fit(u.X_tr, y0)
                    pP = m.predict(u.X_va)
                return pD, pP, pC
            u.emit_pred(fam, sw, 'default', sw, _swap)

    scores3 = {}
    for (meth, setting), cleaned in arms.items():
        u.emit(fam, meth, setting, cleaned)
        if side == 'lower' and cleaned is not None:
            # n30b's three scores: the held-out labels carry the floor too.
            # The fit is the one dprime_rows just made (noise cache).
            try:
                pr = u.pred(cleaned['X'], cleaned['y'],
                            w=cleaned.get('sample_weight'))
                scores3[f'{meth}|{setting}'] = n30b.three_scores(
                    u.y_val, pr, limit)
            except Exception as exc:
                u.errors.add(f'{fam}/{meth}|three_scores', exc)
    if side == 'lower':
        try:
            scores3['as_delivered'] = n30b.three_scores(
                u.y_val, u.pred(u.X_tr, y0), limit)
        except Exception as exc:
            u.errors.add(f'{fam}/as_delivered|three_scores', exc)
        rec['three_scores'] = scores3


# ---------------------------------------------------------------------------
# grouped shifts (n34)
# ---------------------------------------------------------------------------

def grouped_fixes(u, ng_tr, ng_va=None):
    """n34's offset per scaffold group: the group mean residual, the same for
    groups of 3 or more, and a shrunk random intercept. The residuals come
    from one out-of-fold LightGBM (n34's arrangement)."""
    fam = 'grouped'
    if u.is_clf:
        u.not_applicable(fam, 'a class label has no offset to subtract')
        return
    if u.condition not in GROUPED_CONDITIONS:
        u.not_applicable(fam, f'{u.condition} draws no offset per scaffold '
                              f'group; n34 runs under grouped_shifted and '
                              f'under gaussian, its negative control')
        return
    if ng_tr is None:
        u.not_applicable(fam, 'no SMILES, so no scaffold groups')
        return
    ng_tr = np.asarray(ng_tr)
    g = u.g_tr if u.g_tr is not None else np.arange(len(u.y0))

    def oof_fn(X, y):
        return nc.cached_oof_predict(u.ds, X, y, g, n_splits=n34.OOF_SPLITS)
    y0 = u.y0
    _, inv, counts = n34._group_index(ng_tr)
    rec = {'training_groups': int(counts.size),
           'groups_of_one': int((counts == 1).sum()),
           'median_group_size': float(np.median(counts)),
           'groups_of_3_plus': int((counts >= n34.MIN_GROUP_SIZE_FOR_CENTER).sum())}
    if ng_va is not None:
        tg = set(np.unique(ng_tr).tolist())
        rec['held_out_molecules_in_a_training_group'] = int(
            sum(1 for x in np.asarray(ng_va) if x in tg))
    u.record[fam] = rec
    try:
        r = y0 - oof_fn(u.X_tr, y0)
        off_c = n34.group_center_offsets(r, ng_tr)
        off_c3 = n34.group_center_offsets(
            r, ng_tr, min_size=n34.MIN_GROUP_SIZE_FOR_CENTER)
        off_ri, info = n34.random_intercept_offsets(u.X_tr, y0, ng_tr, oof_fn)
        rec['random_intercept'] = info
    except Exception as exc:
        u.errors.add(f'{fam}/offsets', exc)
        for arm in ('group_center', 'group_center_3plus', 'random_intercept'):
            u.emit(fam, arm, 'default', None)
        return
    if u.condition == 'grouped_shifted' and float(u.level or 0) > 0 \
            and u.y_clean is not None:
        rec['offset_recovery'] = {
            'group_center': n34.offset_recovery(off_c, y0, u.y_clean, ng_tr),
            'random_intercept': n34.offset_recovery(off_ri, y0, u.y_clean, ng_tr)}
    u.emit(fam, 'group_center', 'default', _label_dict(u.X_tr, y0 - off_c))
    u.emit(fam, 'group_center_3plus',
           f'min_group_size={n34.MIN_GROUP_SIZE_FOR_CENTER}',
           _label_dict(u.X_tr, y0 - off_c3))
    u.emit(fam, 'random_intercept', f'em_max_rounds={n34.EM_MAX_ROUNDS}',
           _label_dict(u.X_tr, y0 - off_ri))


# ---------------------------------------------------------------------------
# close pairs (n35)
# ---------------------------------------------------------------------------

def pair_fixes(u, distances, same_mol=None):
    """n35 in each kept neighbour setting. distances: [(setting name,
    fn() -> (D, why))], D the square training-by-training distance."""
    fam = 'close_pairs'
    if u.is_clf:
        u.not_applicable(fam, 'the pair estimate reads a numeric label difference')
        return
    n_tr = len(u.y0)
    if n_tr > n35.MAX_PAIR_ROWS:
        u.not_applicable(fam, f'{n_tr} training rows is over n35.MAX_PAIR_ROWS='
                              f'{n35.MAX_PAIR_ROWS}')
        return
    if not distances:
        u.not_applicable(fam, 'no neighbour setting was kept by the screen')
        return
    per = {}
    rec = {'per_setting': per}
    if u.y_clean is not None:
        rec['realised_injected_variance'] = float(np.mean((u.y0 - u.y_clean) ** 2))
    u.record[fam] = rec
    for name, fn in distances:
        try:
            D, why = fn()
        except Exception as exc:
            u.errors.add(f'{fam}@{name}|distance', exc)
            D, why = None, f'{type(exc).__name__}: {exc}'
        if D is None:
            u.not_applicable(f'{fam}@{name}', str(why))
            continue
        try:
            pr, (pi, pj, pdy) = n35.pair_record(D, u.y0, same_mol)
            del D
            sigma2 = pr['intercept_estimate']
            flagged = n35.flag_pairs(pi, pj, pdy, sigma2)
            w = n35.pair_weights(n_tr, pi, pj, flagged)
            pr.update({'flag_estimate': 'intercept_estimate',
                       'flag_estimate_usable': bool(np.isfinite(sigma2)
                                                    and sigma2 >= 0),
                       'flagged_pairs': int(flagged.sum()),
                       'flagged_molecules': int(np.sum(w < 1.0))})
            per[name] = pr
            cleaned = n07.cleaned_from_weights(u.X_tr, u.y0, w)
        except Exception as exc:
            u.errors.add(f'{fam}@{name}', exc)
            cleaned = None
        u.emit(fam, 'downweight_pairs', N35_SETTING, cleaned, sim=name)


def run_unit(u, distances, ng_tr, ng_va=None, same_mol=None):
    """Every family for one unit of an injected kind."""
    for fn, args in ((censoring_fixes, ()), (grouped_fixes, (ng_tr, ng_va)),
                     (pair_fixes, (distances, same_mol))):
        try:
            fn(u, *args)
        except Exception as exc:
            u.errors.add(f'{fn.__name__}|unit', exc)


# ---------------------------------------------------------------------------
# sources (n39)
# ---------------------------------------------------------------------------

def source_estimates(st, val, train_idx, b_true, sigma_e,
                     offset_method='mixedlm'):
    """What n39 fits once per pass on the training molecules' measurements:
    the source offsets, the within-molecule variance, the offset-removed
    labels. Independent of the model and the features."""
    k_tr = st['k'][train_idx]
    mol_t, src_t, val_t = n39.restrict(st, val, train_idx)
    erec = {'training_molecules_with_2_plus_sources': int((k_tr >= 2).sum())}
    out = {'k_tr': k_tr, 'y_off': None, 's_w2': None, 'tau2_hat': 0.0,
           'record': erec}
    try:
        b_hat, s_w2, oinfo = n39.fit_offsets(mol_t, src_t, val_t,
                                             st['n_sources'], method=offset_method)
        used = np.unique(src_t)
        out['y_off'] = (np.bincount(mol_t, weights=val_t - b_hat[src_t],
                                    minlength=len(train_idx)) / k_tr)
        out['s_w2'] = s_w2
        out['tau2_hat'] = float(np.var(b_hat[used])) if used.size > 1 else 0.0
        erec.update({'offset_method_used': oinfo['method'], 'offset_info': oinfo,
                     'within_variance_hat': s_w2,
                     # reads the injected sigma_e: a diagnostic, no fix uses it
                     'rho_hat_diagnostic': n39.rho_hat_from(s_w2, sigma_e),
                     'tau2_hat': out['tau2_hat'],
                     'offset_recovery': n39.offset_recovery(b_hat, b_true, used)})
    except Exception as exc:
        erec['offsets_error'] = f'{type(exc).__name__}: {exc}'
    return out


def source_fixes(u, est, sigma_e, rho, tau):
    """n39's arms for one unit. u.y0 is the plain mean of each molecule's
    sources (D); u.y_clean the clean labels."""
    fam = 'sources'
    k_tr = np.asarray(est['k_tr'], dtype=float)
    y_pl, y_cl = u.y0, u.y_clean
    g = u.g_tr if u.g_tr is not None else np.arange(len(y_pl))

    def oof(X, y):
        return nc.cached_oof_predict(u.ds, X, y, g, n_splits=n39.OOF_SPLITS)
    s_w2, tau2_hat, y_off = est['s_w2'], est['tau2_hat'], est['y_off']
    s_ok = s_w2 is not None and np.isfinite(s_w2)
    arms = {}
    try:
        sd2_clean = float(np.mean((y_cl - oof(u.X_tr, y_cl)) ** 2))
        p_pl = oof(u.X_tr, y_pl)
        su2, se2, finfo = n19.fit_noise_variance(y_pl, p_pl, k_tr)
        u.record.setdefault(fam, {})['n19_fit'] = finfo
        if su2 is not None:
            arms['weights_n19_fitted'] = n07.cleaned_from_weights(
                u.X_tr, y_pl, n19.inverse_variance_weights(su2, se2 / k_tr))
        else:
            arms['weights_n19_fitted'] = None
            u.not_applicable(f'{fam}/weights_n19_fitted', finfo.get('reason'))
        if s_ok:
            v = (s_w2 + tau2_hat) / k_tr
            sd2, _ = n39.subtracted_model_error(y_pl, p_pl, v)
            arms['weights_within_var'] = n07.cleaned_from_weights(
                u.X_tr, y_pl, n39.weights_from_variance(sd2, v))
        if y_off is not None:
            arms['offset_removed'] = _label_dict(u.X_tr, y_off)
            if s_ok:
                p_off = oof(u.X_tr, y_off)
                v0 = s_w2 / k_tr
                sd2o, _ = n39.subtracted_model_error(y_off, p_off, v0)
                arms['offset_removed_within_var_weights'] = {
                    'X': u.X_tr, 'y': np.asarray(y_off), 'keep': None,
                    'sample_weight': n39.weights_from_variance(sd2o, v0),
                    'label_column': np.asarray(y_off)}
        arms['weights_rho0_oracle'] = n07.cleaned_from_weights(
            u.X_tr, y_pl, n39.weights_from_variance(
                sd2_clean, n39.rho_aware_variance(sigma_e, 0.0, k_tr, tau ** 2)))
        arms['weights_true_rho_oracle'] = n07.cleaned_from_weights(
            u.X_tr, y_pl, n39.weights_from_variance(
                sd2_clean, n39.rho_aware_variance(sigma_e, rho, k_tr, tau ** 2)))
    except Exception as exc:
        u.errors.add(f'{fam}/arms', exc)
    for arm in n39.ARMS:
        if arm in ('clean', n39.ARM_TO_BEAT):
            continue
        if arm not in arms:
            why = ('the source offsets could not be fitted' if y_off is None
                   else 'no within-molecule variance (no molecule has 2 '
                        'sources among the training molecules)')
            u.not_applicable(f'{fam}/{arm}', why)
            continue
        u.emit(fam, arm, f'n_sources={n39.N_SOURCES}', arms[arm])


# ---------------------------------------------------------------------------
# repeat measurements (n19, n26, n29)
# ---------------------------------------------------------------------------

def n19_fixes(u, ctx):
    """n19's conditions and model pairs for one (draw, label build, fold,
    model, model input). u.y0 is the label build's training labels; u.y_val
    the held-out consensus labels (n17.build_entities); u.y_clean None."""
    fam = 'repeats_n19'
    setting = f"label_build={ctx['build']}"
    y_tr = u.y0
    nb, var_own, var_cal = ctx['n_behind_tr'], ctx['var_own_tr'], ctx['var_cal_tr']
    oof = y_sh = y_ca = None
    s2u = s2e = w_inv = w_pool = None
    rec = u.record.setdefault(fam, {})
    try:
        oof = nc.cached_oof_predict(u.ds, u.X_tr, y_tr, u.g_tr,
                                    n_splits=n19.OOF_SPLITS)
        y_sh, s_own, tau2 = n19.shrink_toward(y_tr, oof, var_own)
        y_ca, _s, _t = n19.shrink_toward(y_tr, oof, var_cal)
        rec.update({'tau2': float(tau2), 'mean_shrinkage': float(np.mean(s_own))})
        s2u, s2e, nf = n19.fit_noise_variance(y_tr, oof, nb)
        rec['fitted_noise_variance'] = nf
        if s2u is not None:
            w_inv = n19.inverse_variance_weights(s2u, s2e / np.maximum(nb, 1.0))
            pulled, pooled = n19.pooled_spread_variance(ctx['var_all_tr'],
                                                        ctx['n_meas_tr'])
            rec['pooled_measurement_variance'] = pooled
            w_pool = n19.inverse_variance_weights(s2u, pulled / np.maximum(nb, 1.0))
    except Exception as exc:
        u.errors.add(f'{fam}/oof_or_variance', exc)
    X_tr_c, X_va_c = n19.with_count_feature(u.X_tr, u.X_va, ctx['n_meas_tr'],
                                            ctx['n_meas_va'])
    for cond in n19.CONDITIONS:
        if cond == 'untouched':
            continue                       # D
        if cond in n19.WEIGHTED_CONDITIONS:
            w = n19.weights_for(cond, ctx['n_meas_tr'], var_own, nb)
            u.emit(fam, cond, setting, n07.cleaned_from_weights(u.X_tr, y_tr, w))
        elif cond in n19.RELABEL_CONDITIONS:
            y_use = y_sh if cond == 'shrunk' else y_ca
            u.emit(fam, cond, setting,
                   None if y_use is None else _label_dict(u.X_tr, y_use))
        elif cond in n19.COUNT_FEATURE_CONDITIONS:
            w = w_inv if cond == 'invvar_and_count_feature' else None
            if cond == 'invvar_and_count_feature' and w is None:
                u.not_applicable(f'{fam}/{cond}', 'sigma_u^2 and sigma_e^2 '
                                 'could not be fitted (no 1/count spread or no '
                                 'out-of-fold prediction)')

            def _cf(_w=w, _c=cond):
                pD = u.pred(u.X_tr, y_tr)
                if _c == 'invvar_and_count_feature' and _w is None:
                    return pD, None, None, 'no fitted noise variance'
                if _w is not None and not u.takes_weight:
                    return pD, None, None, nc.WEIGHTS_NOT_RUN
                return pD, u.pred(X_tr_c, y_tr, X_va=X_va_c, w=_w), None
            u.emit_pred(fam, cond, setting, u.model, _cf)
        else:                                # the fitted-variance weights
            w = w_pool if cond == 'spread_pooled_weighted' else w_inv
            if w is None:
                u.not_applicable(f'{fam}/{cond}', 'sigma_u^2 and sigma_e^2 '
                                 'could not be fitted')
            u.emit(fam, cond, setting,
                   None if w is None else n07.cleaned_from_weights(u.X_tr, y_tr, w))

    if not ctx.get('run_model_arms'):
        u.not_applicable(f'{fam}/model_arms', 'n19 fits the model pairs in the '
                         "first decided model's units only (2026-10-06)")
        return
    af = ctx.get('arm_factories') or {}
    gp_rows = ctx['gp_rows']

    def _ngb():
        m0, m1 = af['ngboost'](), af['ngboost']()
        m0.fit(u.X_tr, y_tr)
        m1.fit(X_tr_c, y_tr)
        return m0.predict(u.X_va), m1.predict(X_va_c), None

    def _gp():
        m0 = af['gp']()
        m0.fit(u.X_tr[gp_rows], y_tr[gp_rows])
        p0 = m0.predict(u.X_va)
        if s2e is None:
            return p0, None, None, 'no fitted sigma_e^2'
        y_sub = y_tr[gp_rows]
        var_y = float(np.var(y_sub))
        if var_y <= 0:
            return p0, None, None, 'the GP draw has labels of zero variance'
        m1 = af['gp']()
        m1.set_params(alpha=(s2e / np.maximum(nb[gp_rows], 1.0) / var_y) + 1e-6)
        m1.fit(u.X_tr[gp_rows], y_sub)
        return p0, m1.predict(u.X_va), None

    def _aft():
        if oof is None:
            return None, None, None, 'no out-of-fold prediction for the AFT scale'
        sd = float(np.std(y_tr - oof))
        p0 = n19.fit_interval_aft_predict(u.X_tr, y_tr, y_tr, u.X_va, sd)
        lo, hi = ctx['lo_tr'], ctx['hi_tr']
        if not np.any(hi > lo):
            return p0, None, None, ('every training molecule has a zero-width '
                                    'interval, so the fit would repeat '
                                    'xgboost_aft_point')
        return p0, n19.fit_interval_aft_predict(u.X_tr, lo, hi, u.X_va, sd), None

    for variant, fn in (('ngboost_count_feature', _ngb),
                        ('gp_per_molecule_noise', _gp),
                        ('xgboost_interval_aft', _aft)):
        learner = n19.MODEL_ARM_LEARNER[variant]
        key = 'xgboost' if learner == 'xgboost_aft' else learner
        if key not in af:
            u.not_applicable(f'{fam}/{variant}', f'{key} could not be built')
            continue
        u.emit_pred(fam, variant, f'{setting}|against={n19.MODEL_ARM_PAIRS[variant]}',
                    learner, fn)


def n29_fold_context(labels, rec_mol, rec_doc, y_mean, tr, va, ent_groups, n_mol):
    """n29's per-fold document structure over the training measurements."""
    tr_set = np.zeros(n_mol, dtype=bool)
    tr_set[tr] = True
    rec_tr = np.where(tr_set[rec_mol])[0]
    local = np.full(n_mol, -1, dtype=np.int64)
    local[tr] = np.arange(len(tr))
    rml = local[rec_mol[rec_tr]]
    rd_tr = rec_doc[rec_tr]
    component, comp_info = n29.document_groups(rec_mol[rec_tr], rd_tr)
    eligible = n29.eligible_documents(component, comp_info)
    train_docs = set(np.unique(rd_tr).tolist())
    va_docs = {}
    for i in np.where(~tr_set[rec_mol])[0]:
        va_docs.setdefault(int(rec_mol[i]), set()).add(int(rec_doc[i]))
    isolated = np.array([not (va_docs.get(int(m), set()) & train_docs)
                         for m in va], dtype=bool)
    y_qmap, q_docs, q_meas = n29.quantile_map_labels(labels[rec_tr], rd_tr)
    big = [c for c in comp_info.values()
           if c['documents'] >= n29.MIN_DOCUMENTS_IN_GROUP]
    return {'rec_tr': rec_tr, 'rec_mol_local': rml, 'rec_doc_tr': rd_tr,
            'y_rec_tr': labels[rec_tr], 'g_rec_tr': ent_groups[rec_mol[rec_tr]],
            'component': component, 'eligible': eligible, 'isolated': isolated,
            'y_rec_qmap': y_qmap, 'n_tr': len(tr),
            'record': {'training_measurements': int(len(rec_tr)),
                       'training_documents': len(train_docs),
                       'documents_given_an_offset': len(eligible),
                       'document_groups_of_2_or_more': len(big),
                       'held_out_molecules_isolated': int(isolated.sum()),
                       'quantile_map_documents': int(q_docs),
                       'quantile_map_measurements': int(q_meas)}}


def n29_fixes(u, ctx, known_bias=None):
    """n29's arms for one (fold, model, model input). u.X_tr is one row per
    training molecule, u.y0 its mean label (D), u.y_val the held-out
    molecules' mean measurement. known_bias: None, or the arguments the
    check with known biases needs."""
    fam = 'repeats_n29'
    rec = u.record.setdefault(fam, dict(ctx['record']))
    X_rec = u.X_tr[ctx['rec_mol_local']]
    yr, rd, gr = ctx['y_rec_tr'], ctx['rec_doc_tr'], ctx['g_rec_tr']
    comp, elig = ctx['component'], ctx['eligible']
    try:
        offsets, resid, trace = n29.fit_document_offsets(
            X_rec, yr, rd, gr, elig, comp, dataset_name=u.ds)
    except Exception as exc:
        u.errors.add(f'{fam}/offsets', exc)
        for arm in n29.ARMS[1:]:
            if arm != 'within_document_rank':
                u.emit(fam, arm, N29_SETTING, None)
        return
    flips = n29.sign_flip_shares(resid, rd, ctx['rec_mol_local'], ctx['n_tr'],
                                 elig, comp, offsets, n29.N_BOOTSTRAP,
                                 n29.RIDGE_ALPHA, seed=u.fold)
    vals = np.array([offsets[d] for d in elig], dtype=float)
    fl = np.array([v for v in flips.values() if np.isfinite(v)])
    rec.update({
        'largest_offset_change_per_pass': trace,
        'offset_spread_sd': float(vals.std()) if vals.size else float('nan'),
        'documents_with_an_offset': int(vals.size),
        f'documents_flipping_sign_in_over_{n29.FLIP_SHARE_REPORTED:g}_of_resamples':
            int((fl > n29.FLIP_SHARE_REPORTED).sum()),
        'offset_against_document_level': n29.offset_level_correlation(
            offsets, yr - resid, rd, elig, comp)})
    b_rec = n29.offsets_per_record(offsets, rd)
    b_re = None
    try:
        vt = []
        off_re, _r, trace_re = n29.fit_document_offsets(
            X_rec, yr, rd, gr, elig, comp, alpha='fitted', variance_trace=vt,
            dataset_name=u.ds)
        b_re = n29.offsets_per_record(off_re, rd)
        rec['random_effect'] = {'variances_per_pass': vt,
                                'largest_offset_change_per_pass': trace_re}
    except Exception as exc:
        u.errors.add(f'{fam}/random_effect', exc)
    sets = n29.arm_training_sets(u.X_tr, u.y0, X_rec, yr, ctx['rec_mol_local'],
                                 b_rec, b_rec_re=b_re, y_rec_qmap=ctx['y_rec_qmap'])
    iso = ctx['isolated']
    rec['isolated_view'] = {}
    for arm in n29.ARMS:
        if arm in (n29.ARM_TO_BEAT, 'within_document_rank'):
            continue
        if arm not in sets:
            u.emit(fam, arm, N29_SETTING, None)
            continue
        Xa, ya = sets[arm]
        cleaned = (_label_dict(Xa, ya) if Xa.shape[0] == len(u.y0)
                   else _rows_dict(Xa, ya))
        u.emit(fam, arm, N29_SETTING, cleaned)
        if iso.sum() >= 5:
            try:
                pr = u.pred(Xa, ya)
                rec['isolated_view'][arm] = float(u.metric[1](u.y_val[iso], pr[iso]))
            except Exception as exc:
                u.errors.add(f'{fam}/{arm}|isolated', exc)
    if u.model == 'xgboost':
        def _rank():
            pD = u.pred(u.X_tr, u.y0)
            pP, cal = n29.rank_within_documents(X_rec, yr, rd, gr, u.X_va)
            rec['within_document_rank'] = cal
            return pD, pP, None
        u.emit_pred(fam, 'within_document_rank', N29_SETTING, 'xgboost', _rank)
    else:
        u.not_applicable(f'{fam}/within_document_rank',
                         'never ran: an xgboost ranker (rank:pairwise) paired '
                         'against mean_label fitted by xgboost; xgboost is not '
                         'one of this dataset\'s decided models (n29, decided '
                         '2026-10-06, Adelaide delegated)')
    if known_bias is not None and vals.size >= 2:
        try:
            kb = n29._known_bias_check(
                known_bias['labels'], known_bias['rec_mol'], known_bias['rec_doc'],
                known_bias['doc_names'], known_bias['tr'], known_bias['va'],
                ctx['rec_tr'], ctx['rec_mol_local'], u.X_tr, u.X_va, gr, elig,
                comp, float(vals.var()), offsets, known_bias['y_mean'],
                u.factory, u.model, u.metric[1], n29.N_ITER, n29.RIDGE_ALPHA,
                n29.OOF_SPLITS)
            rec['known_bias_check'] = convert_numpy(kb)
        except Exception as exc:
            u.errors.add(f'{fam}/known_bias_check', exc)


def n26_error_by_count(u, k_va):
    """n26: the plain model's held-out residual (mean label minus prediction)
    binned by measurement count. Returns the residuals for pooling."""
    pr = u.pred(u.X_tr, u.y0)
    r = u.y_val - pr
    u.record['repeats_n26'] = {'error_by_count': n26.error_by_count(
        r, np.asarray(k_va))}
    return r


# ---------------------------------------------------------------------------
# shared pieces of the drivers
# ---------------------------------------------------------------------------

def _models(dataset_name, kwargs, n_samples):
    decided = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    asked = kwargs.get('models')
    models = [m for m in decided if m in asked] if asked else list(decided)
    factories = get_model_factories(models, task_type='regression',
                                    n_samples=n_samples)
    models = [m for m in models if m in factories]
    return decided, models, factories


def _work_data(data, dataset_name):
    """(rows the pass runs on, search suffix, replicate groups, stats)."""
    rg = build_replicate_groups(data)
    if rg is None:
        return data, None, None, None
    stats = n19.measurement_stats(np.asarray(data['labels'], dtype=float), rg)
    ent = n19._molecule_rows_data(data, stats, stats['mean_all'], dataset_name)
    same = (len(stats['n_measurements']) == len(data['labels'])
            and np.array_equal(stats['row_index'],
                               np.arange(len(data['labels']))))
    return ent, (None if same else MOLECULE_MEAN_SEARCH_SUFFIX), rg, stats


def _selection_entries(work, dataset_name, fold_i, tr, n_folds, models, suffix):
    """n19's per-fold selection (no neighbour searches): {model: entry}."""
    sel = fold_standard_selection(
        work['reps'], np.asarray(work['labels']), work.get('groups'),
        train_idx=tr, fold_i=int(fold_i), n_outer_folds=n_folds,
        dataset_name=dataset_name, models=list(models),
        augmentations=work.get('augmentations'), task_type='regression',
        condition_suffix=suffix)
    return {m: n19._entry_from_selection(sel.get(m)) for m in models}


def _fold_record(entries):
    out = {}
    for m, e in entries.items():
        names = {}
        if e is not None:
            if e.get('base'):
                names['base_1'] = e['base'][0]
            if e.get('hybrids'):
                names['hybrid'] = e['hybrids'][0].get('label')
        out[m] = configuration_record(e, names)
    return out


def _na_result(dataset_name, family, reason, **extra):
    return dict({'experiment': EXP_ID, 'result_layout': RESULT_LAYOUT,
                 'dataset': dataset_name,
                 'not_applicable': {family: {reason: 1}}}, **extra)


class _Ckpt:
    """The rows sink, the unit checkpoint and the run's error log of one
    pass, as n44 keeps them."""

    def __init__(self, output_dir, dataset_name, file_cond, fingerprint):
        self.sink = n07.RowSink(output_dir, EXP_ID, dataset_name, file_cond)
        self.units = n07.FoldCheckpoint(output_dir, EXP_ID, dataset_name,
                                        file_cond, tag='units',
                                        fingerprint=fingerprint)
        self.errors = n42._Errors()
        self.na_log, self.unavailable = {}, {}
        self.n_done = self.n_resumed = 0

    def done(self, key):
        return self.units._k(key) in self.units.entries

    def unit(self, **kw):
        return _Unit(sink=self.sink, errors=self.errors, na_log=self.na_log,
                     unavailable=self.unavailable, **kw)

    def put(self, key, u, t0, extra=None):
        v = {'rows_written': int(u.rows_written()),
             'seconds': round(time.time() - t0, 1),
             'n_features': int(np.shape(u.X_tr)[1]),
             'not_applicable': dict(u.na), 'record': convert_numpy(u.record)}
        if extra:
            v.update(extra)
        self.units.put(key, v)
        self.n_done += 1

    def summary(self):
        return {'rows': self.sink.summary(), 'units_computed_this_run': self.n_done,
                'units_resumed': self.n_resumed, 'errors': self.errors.d,
                'unavailable': self.unavailable,
                'not_applicable': self.na_log, 'checkpoint': self.units.path,
                'controls_run': bool(controls_enabled())}


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def run(data, config, output_dir, dataset_name, **kwargs):
    """Every decided kind, then the structure passes, one file each."""
    if data.get('task_type') != 'regression':
        why = (f"{dataset_name} is {data.get('task_type')}: every fix here reads "
               f"a numeric label (a limit, an offset, a label difference, a "
               f"source mean, a measurement spread)")
        return {'experiment': EXP_ID, 'result_layout': RESULT_LAYOUT,
                'not_applicable': {f: {why: 1} for f in (
                    'censoring', 'real_censoring', 'grouped', 'close_pairs',
                    'sources', 'repeats_n19', 'repeats_n26', 'repeats_n29')}}
    default = (without_paper3_cuts(decided_conditions(data))
               + list(STRUCTURE_PASSES))
    return sweep_conditions(
        _run_one_condition, EXP_ID, data, config, output_dir, dataset_name,
        default_conditions=default, kwargs=kwargs,
        groups_needed=GROUPED_STRATEGIES, cut_conditions=PAPER3_CUT_CONDITIONS)


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    cond = kwargs.pop('noise_condition')
    if cond == SOURCES_PASS:
        return _run_sources(data, config, output_dir, dataset_name, **kwargs)
    if cond == REPEATS_PASS:
        return _run_repeats(data, config, output_dir, dataset_name, **kwargs)
    if cond == REAL_CENSORING_PASS:
        return _run_real_censoring(data, config, output_dir, dataset_name, **kwargs)
    return _run_injected(data, config, output_dir, dataset_name, cond, **kwargs)


def _fsim_factory(dataset_name, work, m, entry, train_idx, val_idx, y0, g_tr,
                  factories, metric, sp_tr, sp_va, sp_unavail, box):
    """n32.FoldSimilarity with its feature spaces set to the matrices
    n07.fold_similarity_spaces built (n42's arrangement), so n35's full
    distances are taken in the spaces the neighbour lists were found in."""
    def _f():
        if 'f' not in box:
            f = n32.FoldSimilarity(
                dataset_name, work, reps=work['reps'], train_idx=train_idx,
                val_idx=val_idx, y_train=y0, groups_train=g_tr,
                base_by_model={m: list(entry.get('base') or [])},
                recipe_for_space_by_model={m: {k: v[0] for k, v in
                                               (entry.get('recipes') or {}).items()}},
                model_factories=factories, metric=metric,
                task_type='regression',
                augmentations=work.get('augmentations'),
                smiles=work.get('smiles'), is_clf=False)
            for sp in ss.FEATURE_SPACES:
                if sp in sp_tr:
                    f._spaces[(m, sp)] = (sp_tr[sp], sp_va[sp], None)
                else:
                    why = sp_unavail.get(sp, 'not built')
                    f._spaces[(m, sp)] = (None, None, why)
                    f.unavailable[f'{m}|{sp}'] = why
            box['f'] = f
        return box['f']
    return _f


def _run_injected(data, config, output_dir, dataset_name, condition, **kwargs):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT N46, STRUCTURE-SPECIFIC FIXES UNDER {condition}")
    print("=" * 70)
    t_start = time.time()
    work, suffix, _rg, _st = _work_data(data, dataset_name)
    labels = np.asarray(work['labels'], dtype=float)
    groups = work.get('groups')
    metric = get_domain_metric(data)
    levels = tuple(float(x) for x in decided_levels(work, dataset_name, condition))
    draws = tuple(int(d) for d in draws_for(condition, NOISE_DRAWS))
    passes = [(lv, d) for lv in levels if lv != 0.0 for d in draws]
    passes += [(lv, int(NOISE_DRAWS[0])) for lv in levels if lv == 0.0]
    decided, models, factories = _models(dataset_name, kwargs, len(labels))
    takes_weight = {m: bool(nc.fit_takes_sample_weight(factories[m])) for m in models}
    swap_f = get_model_factories([s for s, p in CENSORING_SWAPS if s != 'xgboost_aft'],
                                 task_type='regression', n_samples=len(labels))
    smiles = work.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    n_mol = len(set(smiles)) if has_smiles else len(labels)
    settings = n07.sweep_settings(dataset_name, n_mol, has_smiles)
    ks = n07.sweep_ks()
    noise_groups = noise_groups_for(work)[0]
    folds, n_folds = noise_outer_folds(work, config)
    file_cond = f'{condition}__{RESULT_LAYOUT}'
    fp = {'layout': RESULT_LAYOUT, 'input': input_fingerprint(data, config),
          'condition': condition, 'levels': list(levels), 'draws': list(draws),
          'n_folds': int(n_folds), 'models': list(models),
          'model_inputs': list(MODEL_INPUTS), 'search_suffix': suffix,
          'settings': [s.name for s in settings], 'ks': list(ks),
          'screen_top': n07.screen_top(), 'controls': bool(controls_enabled())}
    ck = _Ckpt(output_dir, dataset_name, file_cond, fp)
    fold_records = {}
    print(f"  {condition}: levels {list(levels)} ({level_unit(condition)}); "
          f"{len(passes)} (level, draw) passes; {n_folds} folds; models {models}; "
          f"rows -> {ck.sink.path}")

    for fold_i, tr, va in folds:
        tr, va = np.asarray(tr), np.asarray(va)
        keys = [[int(fold_i), float(lv), int(dr), m, mi]
                for lv, dr in passes for m in models for mi in MODEL_INPUTS]
        if all(ck.done(k) for k in keys):
            ck.n_resumed += len(keys)
            continue
        try:
            entries = fold_configurations(work, dataset_name, fold_i=fold_i,
                                          train_idx=tr, n_folds=n_folds,
                                          condition_suffix=suffix)
        except Exception as exc:
            fold_records[str(fold_i)] = {'error': f'{type(exc).__name__}: {exc}'}
            continue
        fold_records[str(fold_i)] = _fold_record(entries)
        y_clean, y_val = labels[tr], labels[va]
        g_tr = None if groups is None else np.asarray(groups)[tr]
        ng_tr = None if noise_groups is None else np.asarray(noise_groups)[tr]
        ng_va = None if noise_groups is None else np.asarray(noise_groups)[va]
        smi_tr = None if smiles is None else [smiles[i] for i in tr]
        smi_va = None if smiles is None else [smiles[i] for i in va]
        for level, draw in passes:
            todo = [(m, mi) for m in models for mi in MODEL_INPUTS
                    if not ck.done([int(fold_i), level, draw, m, mi])]
            ck.n_resumed += len(models) * len(MODEL_INPUTS) - len(todo)
            if not todo:
                continue
            try:
                y0, _cm, prov = inject_condition(work, dataset_name, condition,
                                                 level, draw, train_idx=tr,
                                                 groups=noise_groups)
            except Exception as exc:
                ck.errors.add(f'inject|level={level:g}', exc)
                continue
            y0 = np.asarray(y0, dtype=float)
            for m in models:
                mis = [mi for mm, mi in todo if mm == m]
                entry = entries.get(m)
                if not mis:
                    continue
                if entry is None:
                    for mi in mis:
                        ck.unavailable[f'fold{fold_i}|{m}:{mi}'] = 'the per-fold search failed'
                    continue
                try:
                    ins, unav, _n = fold_model_inputs(
                        work, dataset_name, m, entry, train_idx=tr, val_idx=va,
                        y_train=y0, factory=factories[m], metric=metric,
                        spaces=MODEL_INPUTS)
                    sp_tr, sp_va, sp_unavail = n07.fold_similarity_spaces(
                        dataset_name, list(settings), model_input_tr=None,
                        model_input_va=None, reps=work['reps'], train_idx=tr,
                        val_idx=va, y_train=y0, groups_train=g_tr,
                        base_reps=entry.get('base') or [],
                        recipes=entry.get('recipes') or {},
                        augmentations=work.get('augmentations'), model_name=m,
                        model_factory=factories[m], metric=metric,
                        task_type='regression')
                except Exception as exc:
                    ck.errors.add(f'spaces|{m}', exc)
                    continue
                shared, box = {}, {}
                fsim = _fsim_factory(dataset_name, work, m, entry, tr, va, y0,
                                     g_tr, factories, metric, sp_tr, sp_va,
                                     sp_unavail, box)
                for mi in mis:
                    t0 = time.time()
                    if mi not in ins:
                        ck.unavailable[f'fold{fold_i}|{m}:{mi}'] = unav.get(mi) or 'not built'
                        continue
                    X_tr, X_va = ins[mi]
                    u = ck.unit(ds=dataset_name, metric=metric,
                                task_type='regression', condition=condition,
                                level=level, draw=draw, fold=int(fold_i),
                                model=m, mi=f'{m}:{mi}', X_tr=X_tr, X_va=X_va,
                                y0=y0, y_clean=y_clean, y_val=y_val, g_tr=g_tr,
                                factory=factories[m], takes_weight=takes_weight[m],
                                swap_factories=swap_f, prov=prov)
                    distances = []
                    try:
                        sim = n42._Similarity(
                            u, settings, ks, dict(sp_tr, model_input=X_tr),
                            dict(sp_va, model_input=X_va), shared, sp_unavail,
                            smi_tr, smi_va)
                        for s in sim.kept_settings:
                            if s.space == ss.MODEL_INPUT_SPACE:
                                continue
                            distances.append((s.name, (
                                lambda _s=s, _X=X_tr, _m=m: fsim().full_distance(
                                    _s, _m, _X, dtype=np.float32))))
                        screen = sim.screen_record()
                    except Exception as exc:
                        ck.errors.add(f'similarity|{m}:{mi}', exc)
                        screen = None
                    noise_timing.set_context(fold=int(fold_i), level=float(level),
                                             draw=int(draw), model=m, model_input=mi)
                    with noise_timing.timed('piece'):
                        run_unit(u, distances, ng_tr, ng_va)
                    ck.put([int(fold_i), level, draw, m, mi], u, t0,
                           {'screen_kept_settings': (screen or {}).get(
                               'kept_settings_n35')})
                    print(f"    fold {fold_i} level {level:g} draw {draw} {m}:{mi}: "
                          f"{u.rows_written()} rows, {time.time() - t0:.0f}s")

    res = {'experiment': EXP_ID, 'result_layout': RESULT_LAYOUT,
           'noise_condition': condition, 'metric': metric[0],
           'lower_is_better': bool(metric[2]), 'levels': list(levels),
           'level_unit': level_unit(condition), 'draws_per_nonzero_level': len(draws),
           'n_folds': int(n_folds), 'folds_run': [int(f) for f, _t, _v in folds],
           'models': models, 'models_decided': decided,
           'model_inputs': list(MODEL_INPUTS), 'search_condition_suffix': suffix,
           'rows_are': ('one row per molecule (molecule_mean_data)' if suffix
                        else 'the dataset as loaded'),
           'knn_group': dataset_knn_group(work, dataset_name),
           'clean_base_dataset': dataset_name in CLEAN_BASE_DATASETS,
           'clean_base_note': (
               'QM9 labels are computed, so under censoring this dataset is '
               "n30b's injected-censoring check, where the truth is known; at "
               'level 0 it is the clean comparison point (handoff section 10).'
               if dataset_name in CLEAN_BASE_DATASETS else None),
           'families': ['censoring', 'grouped', 'close_pairs'],
           'folds': fold_records, 'seconds': round(time.time() - t_start, 1)}
    res.update(ck.summary())
    print(f"  N46 {condition}: {ck.n_done} units, {ck.sink.n_written} rows, "
          f"{len(ck.errors.d)} distinct errors")
    return res


def _run_sources(data, config, output_dir, dataset_name, **kwargs):
    """n39: simulated sources with correlated errors and offsets."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N46, SOURCES (n39's injected structure)")
    print("=" * 70)
    t_start = time.time()
    work, suffix, rg, stats = _work_data(data, dataset_name)
    y_ent = np.asarray(work['labels'], dtype=float)
    groups = work.get('groups')
    n_ent, label_sd = len(y_ent), float(np.std(y_ent))
    metric = get_domain_metric(data)
    decided, models, factories = _models(dataset_name, kwargs, n_ent)
    takes_weight = {m: bool(nc.fit_takes_sample_weight(factories[m])) for m in models}
    levels = [float(x) for x in decided_levels(work, dataset_name, 'gaussian')
              if float(x) > 0]
    shares, shares_src = {}, {}
    for cd in n39.COUNT_DISTRIBUTIONS:
        shares[cd], shares_src[cd] = n39.count_shares(cd)
    passes = [(cd, lv, float(r), float(t), int(d))
              for cd in n39.COUNT_DISTRIBUTIONS for lv in levels
              for r in n39.RHOS for t in n39.TAU_FRACS for d in n39.NOISE_DRAWS]
    folds, n_folds = noise_outer_folds(work, config)
    rec_mol = rec_doc = None
    if rg is not None:
        rec_mol = n29.record_to_molecule(rg)
        rd = data.get('record_documents')
        rec_doc = (None if rd is None else
                   np.unique(np.asarray(rd).astype(str), return_inverse=True)[1])
    fp = {'layout': RESULT_LAYOUT, 'input': input_fingerprint(data, config),
          'passes': [list(p) for p in passes], 'n_folds': int(n_folds),
          'models': list(models), 'model_inputs': list(MODEL_INPUTS),
          'search_suffix': suffix, 'controls': bool(controls_enabled())}
    ck = _Ckpt(output_dir, dataset_name, f'{SOURCES_PASS}__{RESULT_LAYOUT}', fp)
    structures, fold_records, real_block = {}, {}, []

    def _structure(cd, draw):
        if (cd, draw) not in structures:
            structures[(cd, draw)] = n39.draw_structure(
                n_ent, shares[cd], n39.pass_seed(dataset_name, cd, draw))
        return structures[(cd, draw)]

    for fold_i, tr, va in folds:
        tr, va = np.asarray(tr), np.asarray(va)
        keys = [[int(fold_i), cd, lv, r, t, d, m, mi]
                for cd, lv, r, t, d in passes for m in models for mi in MODEL_INPUTS]
        if all(ck.done(k) for k in keys) and (rec_mol is None or ck.done(['real', int(fold_i)])):
            ck.n_resumed += len(keys)
            continue
        try:
            entries = _selection_entries(work, dataset_name, fold_i, tr, n_folds,
                                         models, suffix)
        except Exception as exc:
            fold_records[str(fold_i)] = {'error': f'{type(exc).__name__}: {exc}'}
            continue
        fold_records[str(fold_i)] = _fold_record(entries)
        y_cl, y_va = y_ent[tr], y_ent[va]
        g_tr = None if groups is None else np.asarray(groups)[tr]
        # n39's hybrid_labels='clean': built once per fold from the clean
        # training labels and shared by every pass and arm
        cfgs = []
        for m in models:
            ins, unav, _n = fold_model_inputs(
                work, dataset_name, m, entries.get(m), train_idx=tr, val_idx=va,
                y_train=y_cl, factory=factories[m], metric=metric,
                spaces=MODEL_INPUTS)
            for mi in MODEL_INPUTS:
                if mi in ins:
                    cfgs.append((m, mi, ins[mi][0], ins[mi][1]))
                else:
                    ck.unavailable[f'fold{fold_i}|{m}:{mi}'] = unav.get(mi) or 'not built'
        # n39's real-data block, on a dataset with repeat measurements
        if rec_mol is not None and not ck.done(['real', int(fold_i)]):
            labels_rec = np.asarray(data['labels'], dtype=float)
            mols, a, b, cnt = n39.real_pairs(labels_rec, rec_mol, rec_doc, tr,
                                             seed=fold_i)
            s_w2 = n39.pooled_within_variance(labels_rec, rec_mol, tr)
            fr = {'fold': int(fold_i), 'pairs': cnt, 'pooled_within_variance': s_w2,
                  'per_configuration': {}}
            pos = {int(x): j for j, x in enumerate(tr)}
            for m, mi, X_tr, _X_va in cfgs:
                try:
                    p = nc.cached_oof_predict(dataset_name, X_tr, y_cl,
                                              g_tr if g_tr is not None else np.arange(len(tr)),
                                              n_splits=n39.OOF_SPLITS)
                    pm = np.array([p[pos[int(x)]] for x in mols])
                    su2, se2, finfo = n19.fit_noise_variance(
                        y_cl, p, stats['n_measurements'][tr])
                    fr['per_configuration'][f'{m}:{mi}'] = {
                        'pair_residual_correlation_upper_bound_on_rho':
                            n39.pair_residual_correlation(a - pm, b - pm),
                        'n19_intercept_a': su2, 'n19_slope_b': se2}
                except Exception as exc:
                    ck.errors.add(f'real|{m}:{mi}', exc)
            real_block.append(convert_numpy(fr))
            ck.units.put(['real', int(fold_i)], convert_numpy(fr))
        for cd, level, rho, tf, draw in passes:
            todo = [(m, mi, X_tr, X_va) for m, mi, X_tr, X_va in cfgs
                    if not ck.done([int(fold_i), cd, level, rho, tf, draw, m, mi])]
            if not todo:
                continue
            st = _structure(cd, draw)
            sigma_e = level * label_sd
            tau = tf * sigma_e
            val, y_mean, b_true = n39.simulate(y_ent, st, sigma_e, rho, tau)
            est = source_estimates(st, val, tr, b_true, sigma_e)
            est['record'].update({'sigma_e': sigma_e, 'tau': tau, 'rho_true': rho,
                                  'true_within_variance': (1.0 - rho) * sigma_e ** 2})
            ckey = n39.cond_key(cd, rho, tf)
            for m, mi, X_tr, X_va in todo:
                t0 = time.time()
                u = ck.unit(ds=dataset_name, metric=metric, task_type='regression',
                            condition=f'{SOURCES_PASS}|{ckey}', level=level,
                            draw=draw, fold=int(fold_i), model=m, mi=f'{m}:{mi}',
                            X_tr=X_tr, X_va=X_va, y0=y_mean[tr], y_clean=y_cl,
                            y_val=y_va, g_tr=g_tr, factory=factories[m],
                            takes_weight=takes_weight[m])
                u.record['sources_pass'] = est['record']
                try:
                    source_fixes(u, est, sigma_e, rho, tau)
                except Exception as exc:
                    ck.errors.add('sources|unit', exc)
                ck.put([int(fold_i), cd, level, rho, tf, draw, m, mi], u, t0)
            print(f"    fold {fold_i} {ckey} level {level:g} d{draw}: "
                  f"{len(todo)} units")

    res = {'experiment': EXP_ID, 'result_layout': RESULT_LAYOUT,
           'noise_condition': SOURCES_PASS, 'metric': metric[0],
           'lower_is_better': bool(metric[2]), 'models': models,
           'models_decided': decided, 'model_inputs': list(MODEL_INPUTS),
           'n_folds': int(n_folds), 'folds_run': [int(f) for f, _t, _v in folds],
           'n_sources': n39.N_SOURCES, 'rhos': list(n39.RHOS),
           'tau_fracs_of_sigma_e': list(n39.TAU_FRACS),
           'count_distributions': list(n39.COUNT_DISTRIBUTIONS),
           'count_distribution_source': shares_src,
           'sigma_e_levels': levels,
           'sigma_e_units': 'level x the SD of the clean label column '
                            '(decided_levels for gaussian, 0 dropped)',
           'hybrid_labels': 'clean (n39 default): built once per fold',
           'rows_note': ("D is plain_mean (each molecule's source mean); "
                         'noise_condition is sources|<count distribution>|rho|tau'),
           'rho_identifiability_note': (
               'rho is not identifiable from the labels; rho_hat_diagnostic '
               'reads the injected sigma_e and no fix uses it (n39, 2026-09-28)'),
           'search_condition_suffix': suffix, 'real_data_block': real_block,
           'folds': fold_records, 'seconds': round(time.time() - t_start, 1)}
    res.update(ck.summary())
    return res


def _run_repeats(data, config, output_dir, dataset_name, **kwargs):
    """n19, n26 and n29 on the repeat measurements the loader keeps."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N46, REPEAT MEASUREMENTS (n19, n26, n29)")
    print("=" * 70)
    t_start = time.time()
    work, suffix, rg, stats = _work_data(data, dataset_name)
    if rg is None:
        return _na_result(dataset_name, 'repeats',
                          f"{dataset_name}'s loader keeps no repeat measurements "
                          f"(build_replicate_groups returns None), so no molecule "
                          f"has a measurement count, a spread or a document")
    labels = np.asarray(data['labels'], dtype=float)
    metric = get_domain_metric(data)
    ent_groups = np.asarray(work['groups'])
    n_mol = len(stats['n_measurements'])
    y_mean = stats['mean_all']
    probe = n17.build_entities(labels, rg, seed=0)
    if not np.array_equal(probe['row_index'], stats['row_index']):
        raise RuntimeError('n46: build_entities and measurement_stats disagree '
                           'about the molecule order')
    rec_mol = n29.record_to_molecule(rg)
    rd_raw = data.get('record_documents')
    if rd_raw is not None:
        doc_names, rec_doc = np.unique(np.asarray(rd_raw).astype(str),
                                       return_inverse=True)
    else:
        doc_names = rec_doc = None
    folds, n_folds = noise_outer_folds(work, config)
    largest = max((int(np.isin(rec_mol, tr).sum()) for _, tr, _ in folds), default=n_mol)
    decided, models, factories = _models(dataset_name, kwargs, largest)
    takes_weight = {m: bool(nc.fit_takes_sample_weight(factories[m])) for m in models}
    arm_f = {}
    try:
        arm_f.update(get_model_factories(['ngboost'], task_type='regression',
                                         n_samples=n_mol))
    except Exception:
        pass
    try:
        from kirby.find_best_hybrid import _get_gp_regressor
        arm_f['gp'] = _get_gp_regressor
    except Exception:
        pass
    try:
        import xgboost  # noqa: F401
        arm_f['xgboost'] = n19.fit_interval_aft_predict
    except ImportError:
        pass
    fp = {'layout': RESULT_LAYOUT, 'input': input_fingerprint(data, config),
          'n_folds': int(n_folds), 'models': list(models),
          'model_inputs': list(MODEL_INPUTS), 'n_draws': n19.N_DRAWS,
          'label_builds': list(n19.LABEL_BUILDS), 'search_suffix': suffix,
          'controls': bool(controls_enabled())}
    ck = _Ckpt(output_dir, dataset_name, f'{REPEATS_PASS}__{RESULT_LAYOUT}', fp)
    fold_records = {}
    pooled = {}
    known_bias_fold = folds[0][0] if folds else None
    entities = {d: n17.build_entities(labels, rg, seed=d) for d in range(n19.N_DRAWS)}
    n_grade = int(probe['gradeable'].sum())

    for fold_i, tr, va in folds:
        tr, va = np.asarray(tr), np.asarray(va)
        try:
            entries = _selection_entries(work, dataset_name, fold_i, tr, n_folds,
                                         models, suffix)
        except Exception as exc:
            fold_records[str(fold_i)] = {'error': f'{type(exc).__name__}: {exc}'}
            continue
        fold_records[str(fold_i)] = _fold_record(entries)
        g_tr = ent_groups[tr]
        # ── n19: three draws x two label builds ──
        for draw in range(n19.N_DRAWS):
            ent = entities[draw]
            va_g = va[ent['gradeable'][va]]
            if len(va_g) < 5:
                why = (f'fold {fold_i} draw {draw}: fewer than 5 held-out '
                       f'molecules measured {n17.MIN_MEASUREMENTS}+ times, so '
                       f'nothing to grade against (n19\'s rule)')
                d = ck.na_log.setdefault('repeats_n19', {})
                d[why] = d.get(why, 0) + 1
                continue
            for build in n19.LABEL_BUILDS:
                y_all, n_behind = n19.training_labels(ent, stats, build)
                lo_all, hi_all = n19.interval_bounds(ent, stats, build, y_all)
                var_own, _nf, _fl = n19.label_variance(stats, n_behind)
                var_cal, noise_fit, _nt = n19.calibrated_variance(stats, n_behind)
                rng = np.random.default_rng(1000 * draw + int(fold_i))
                gp_rows = (np.arange(len(tr)) if len(tr) <= n19.GP_MAX_TRAIN else
                           np.sort(rng.choice(len(tr), n19.GP_MAX_TRAIN, replace=False)))
                ctx = {'build': build, 'n_behind_tr': n_behind[tr],
                       'var_own_tr': var_own[tr], 'var_cal_tr': var_cal[tr],
                       'var_all_tr': stats['var_all'][tr],
                       'n_meas_tr': stats['n_measurements'][tr],
                       'n_meas_va': stats['n_measurements'][va_g],
                       'lo_tr': lo_all[tr], 'hi_tr': hi_all[tr],
                       'arm_factories': arm_f, 'gp_rows': gp_rows}
                for m in models:
                    keys = [['n19', int(fold_i), draw, build, m, mi] for mi in MODEL_INPUTS]
                    if all(ck.done(k) for k in keys):
                        ck.n_resumed += len(keys)
                        continue
                    ins, unav, _n = fold_model_inputs(
                        work, dataset_name, m, entries.get(m), train_idx=tr,
                        val_idx=va_g, y_train=y_all[tr], factory=factories[m],
                        metric=metric, spaces=MODEL_INPUTS)
                    for mi in MODEL_INPUTS:
                        key = ['n19', int(fold_i), draw, build, m, mi]
                        if ck.done(key):
                            continue
                        if mi not in ins:
                            ck.unavailable[f'fold{fold_i}|{m}:{mi}'] = unav.get(mi) or 'not built'
                            continue
                        t0 = time.time()
                        u = ck.unit(ds=dataset_name, metric=metric,
                                    task_type='regression',
                                    condition=REPEATS_ROW_CONDITION, level=None,
                                    draw=draw, fold=int(fold_i), model=m,
                                    mi=f'{m}:{mi}', X_tr=ins[mi][0],
                                    X_va=ins[mi][1], y0=y_all[tr], y_clean=None,
                                    y_val=ent['grade_label'][va_g], g_tr=g_tr,
                                    factory=factories[m],
                                    takes_weight=takes_weight[m])
                        ctx['run_model_arms'] = (m == models[0])
                        u.record['fitted_variance_by_measurement_count'] = noise_fit
                        try:
                            n19_fixes(u, ctx)
                        except Exception as exc:
                            ck.errors.add('repeats_n19|unit', exc)
                        ck.put(key, u, t0)
        # ── n29 and n26: one draw, every held-out molecule ──
        n29_ctx = (None if rec_doc is None else n29_fold_context(
            labels, rec_mol, rec_doc, y_mean, tr, va, ent_groups, n_mol))
        for m in models:
            ins, unav, _n = fold_model_inputs(
                work, dataset_name, m, entries.get(m), train_idx=tr, val_idx=va,
                y_train=y_mean[tr], factory=factories[m], metric=metric,
                spaces=MODEL_INPUTS)
            for mi in MODEL_INPUTS:
                key = ['n29', int(fold_i), m, mi]
                if ck.done(key):
                    ck.n_resumed += 1
                    v = ck.units.get(key) or {}
                    if v.get('residuals') is not None:
                        pooled.setdefault(f'{m}:{mi}', ([], []))
                        pooled[f'{m}:{mi}'][0].extend(v['residuals'])
                        pooled[f'{m}:{mi}'][1].extend(v['counts'])
                    continue
                if mi not in ins:
                    ck.unavailable[f'fold{fold_i}|{m}:{mi}'] = unav.get(mi) or 'not built'
                    continue
                t0 = time.time()
                u = ck.unit(ds=dataset_name, metric=metric, task_type='regression',
                            condition=REPEATS_ROW_CONDITION, level=None, draw=0,
                            fold=int(fold_i), model=m, mi=f'{m}:{mi}',
                            X_tr=ins[mi][0], X_va=ins[mi][1], y0=y_mean[tr],
                            y_clean=None, y_val=y_mean[va], g_tr=g_tr,
                            factory=factories[m], takes_weight=takes_weight[m])
                extra = {}
                try:
                    r = n26_error_by_count(u, stats['n_measurements'][va])
                    extra = {'residuals': [float(x) for x in r],
                             'counts': [int(x) for x in stats['n_measurements'][va]]}
                    pooled.setdefault(f'{m}:{mi}', ([], []))
                    pooled[f'{m}:{mi}'][0].extend(extra['residuals'])
                    pooled[f'{m}:{mi}'][1].extend(extra['counts'])
                except Exception as exc:
                    ck.errors.add('repeats_n26|unit', exc)
                if n29_ctx is None:
                    u.not_applicable('repeats_n29', 'no document per measurement '
                                     '(record_documents); only herg_ic50 carries one')
                else:
                    kb = None
                    if (fold_i == known_bias_fold and mi == 'base_1'
                            and m == models[0]):
                        kb = {'labels': labels, 'rec_mol': rec_mol, 'rec_doc': rec_doc,
                              'doc_names': doc_names, 'tr': tr, 'va': va,
                              'y_mean': y_mean}
                    try:
                        n29_fixes(u, n29_ctx, known_bias=kb)
                    except Exception as exc:
                        ck.errors.add('repeats_n29|unit', exc)
                ck.put(key, u, t0, extra)

    pooled_fits = {}
    for key, (r, k) in pooled.items():
        r, k = np.asarray(r, dtype=float), np.asarray(k, dtype=float)
        r2, ik = r ** 2, 1.0 / np.maximum(k, 1.0)
        pooled_fits[key] = {'error_by_count': n26.error_by_count(r, k),
                            'ols': n26.fit_ols(r2, ik),
                            'binned': n26.fit_binned(r2, k),
                            'gamma': n26.fit_gamma(r2, ik)}
    values = n26.measurements_by_molecule(labels, rg)
    res = {'experiment': EXP_ID, 'result_layout': RESULT_LAYOUT,
           'noise_condition': REPEATS_PASS, 'row_noise_condition': REPEATS_ROW_CONDITION,
           'metric': metric[0], 'lower_is_better': bool(metric[2]),
           'models': models, 'models_decided': decided,
           'model_inputs': list(MODEL_INPUTS), 'n_folds': int(n_folds),
           'folds_run': [int(f) for f, _t, _v in folds],
           'n_measurements': int(len(labels)), 'n_molecules': int(n_mol),
           'n_molecules_gradeable': n_grade,
           'label_builds': list(n19.LABEL_BUILDS), 'n19_draws': n19.N_DRAWS,
           'n19_grading': ('held-out molecules measured '
                           f'{n17.MIN_MEASUREMENTS}+ times, against the mean of '
                           'the measurements not used for training '
                           '(n17.build_entities)'),
           'n29_grading': "held-out molecules against the mean of all their measurements",
           'documents_available': rec_doc is not None,
           'within_document_rank_ran': 'xgboost' in models,
           'weights_reach_feature_selection': False,
           'n26_pooled_over_folds': pooled_fits,
           'n26_direct_noise_variance': n26.direct_noise_variance(values),
           'search_condition_suffix': suffix, 'folds': fold_records,
           'seconds': round(time.time() - t_start, 1)}
    res.update(ck.summary())
    return res


def _run_real_censoring(data, config, output_dir, dataset_name, **kwargs):
    """n30b on labels as delivered, where the author has accepted a floor."""
    labels = np.asarray(data['labels'], dtype=float)
    floor = n30b.floor_from_labels(labels)
    top = n30.limit_from_labels(labels)
    piles = {'lowest_value': floor,
             'labels_on_lowest_value': (int(np.sum(labels <= floor + 1e-12))
                                        if floor is not None else 0),
             'highest_value': top,
             'labels_on_highest_value': (int(np.sum(labels >= top - 1e-12))
                                         if top is not None else 0),
             'n_labels': int(len(labels))}
    side = REAL_CENSORING_LIMITS.get(dataset_name)
    if side is None:
        why = ('the author has accepted no limit in this dataset\'s labels as '
               'delivered (REAL_CENSORING_LIMITS)')
        if floor is not None or top is not None:
            why += ('; a pile was found (piles), and whether it is an assay '
                    'limit is the author\'s decision')
        return _na_result(dataset_name, 'real_censoring', why, piles=piles)
    print("\n" + "=" * 70)
    print(f"EXPERIMENT N46, REAL CENSORING ({side} limit, labels as delivered)")
    print("=" * 70)
    t_start = time.time()
    metric = get_domain_metric(data)
    decided, models, factories = _models(dataset_name, kwargs, len(labels))
    takes_weight = {m: bool(nc.fit_takes_sample_weight(factories[m])) for m in models}
    groups = data.get('groups')
    folds, n_folds = noise_outer_folds(data, config)
    fp = {'layout': RESULT_LAYOUT, 'input': input_fingerprint(data, config),
          'side': side, 'n_folds': int(n_folds), 'models': list(models),
          'model_inputs': list(MODEL_INPUTS), 'controls': bool(controls_enabled())}
    ck = _Ckpt(output_dir, dataset_name, f'{REAL_CENSORING_PASS}__{RESULT_LAYOUT}', fp)
    fold_records = {}
    for fold_i, tr, va in folds:
        tr, va = np.asarray(tr), np.asarray(va)
        try:
            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=tr, n_folds=n_folds)
        except Exception as exc:
            fold_records[str(fold_i)] = {'error': f'{type(exc).__name__}: {exc}'}
            continue
        fold_records[str(fold_i)] = _fold_record(entries)
        g_tr = None if groups is None else np.asarray(groups)[tr]
        for m in models:
            ins, unav, _n = fold_model_inputs(
                data, dataset_name, m, entries.get(m), train_idx=tr, val_idx=va,
                y_train=labels[tr], factory=factories[m], metric=metric,
                spaces=MODEL_INPUTS)
            for mi in MODEL_INPUTS:
                key = [int(fold_i), m, mi]
                if ck.done(key):
                    ck.n_resumed += 1
                    continue
                if mi not in ins:
                    ck.unavailable[f'fold{fold_i}|{m}:{mi}'] = unav.get(mi) or 'not built'
                    continue
                t0 = time.time()
                u = ck.unit(ds=dataset_name, metric=metric, task_type='regression',
                            condition=REAL_CENSORING_PASS, level=None, draw=0,
                            fold=int(fold_i), model=m, mi=f'{m}:{mi}',
                            X_tr=ins[mi][0], X_va=ins[mi][1], y0=labels[tr],
                            y_clean=None, y_val=labels[va], g_tr=g_tr,
                            factory=factories[m], takes_weight=takes_weight[m])
                try:
                    censoring_fixes(u, side=side)
                except Exception as exc:
                    ck.errors.add('real_censoring|unit', exc)
                ck.put(key, u, t0)
    res = {'experiment': EXP_ID, 'result_layout': RESULT_LAYOUT,
           'noise_condition': REAL_CENSORING_PASS, 'side': side, 'piles': piles,
           'metric': metric[0], 'lower_is_better': bool(metric[2]),
           'models': models, 'models_decided': decided,
           'model_inputs': list(MODEL_INPUTS), 'n_folds': int(n_folds),
           'three_scores_where': 'each unit\'s record.real_censoring.three_scores',
           'folds': fold_records, 'seconds': round(time.time() - t_start, 1)}
    res.update(ck.summary())
    return res


def main():
    import argparse
    from experiments.noise.noise_helpers import run_all
    ap = argparse.ArgumentParser(description='N46 on one dataset through '
                                             'noise_helpers.run_all')
    ap.add_argument('dataset')
    ap.add_argument('--output', required=True)
    ap.add_argument('--noise-condition', default=None,
                    help='one decided kind, or sources / repeat_measurements / '
                         'real_censoring')
    args = ap.parse_args()
    run_all(dataset_name=args.dataset, output_dir=args.output,
            experiments=['n46'], noise_condition=args.noise_condition)


if __name__ == '__main__':
    main()
