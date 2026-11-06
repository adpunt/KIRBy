#!/usr/bin/env python3
"""N43, noise-aware training: every published method that trains a model
differently under noisy labels, scored as D against D'.

HANDOFF_NOISE_RESTRUCTURE.md, section 2 (the noise-aware experiment, working
name NA), section 4.1 (list A, the rows whose "Runs as" says NA) and section 8
(the feature sets). Written 2026-11-02. It replaces the method parts of n09,
n33 and n40, whose code it imports; those files stay as they are.

WHAT ONE ROW IS. d_vs_dprime's row. D is the plain model trained on the
training fold's labels as this pass holds them (damaged above level 0, as
delivered at level 0). D' is one method trained on the same labels, the same
features and the same molecules. Both are scored on the same held-out fold
against its untouched labels, beside the plain model trained on the clean
training labels (score_clean). score_D, score_Dprime and score_clean are in
the dataset's own metric; all_D, all_Dprime and all_clean hold every metric
of d_vs_dprime.ALL_METRICS from the same predictions (section 11).
`method` is '<source>/<name>', the source being the file the method's code
comes from: lnl (kirby.utils.lnl_methods, as n09 ran it), n33 or n40.
`model` is the model D is. `model_input` is the feature space alone.

THE METHODS (list A rows marked NA; n09, n33 and n40 "Do not lose").
  lnl/<name>   the network methods n09 ran, each trained as published by
               lnl_methods.get_lnl_method(name).train_and_predict, with n09's
               settings (seed 42 + fold, selection rows kept scaffold-disjoint
               through selection_groups). D is n09's plain network,
               n09._run_baseline. model 'dnn'. The nine: early_learning,
               self_paced_learning, dividemix, meta_weight_net, mentornet
               (MentorNet PD), mentornet_dd, heteroscedastic_gp,
               evidential_kernel (the last two regression only) and
               multistage_cleaning (whole here, author 2026-10-31, handoff
               9.6). MentorNet DD and Meta-Weight-Net take their trusted rows
               from the method's own selection carve, which carries the
               training labels' noise (handoff section 8, open).
  n33/<name>   n33's arms on the dataset's two decided models (knn left out,
               as n33 does), each paired against the same model on the same
               labels:
                 yeo_johnson, c_mixup, target_noise_bagging (n33
                 label_arm_predict, n33's settings);
                 every loss swap of n33.build_swaps at n33.HUBER_C (lightgbm
                 Huber, fair, median, Tukey, the two expectiles; xgboost
                 pseudo-Huber and absolute; rf_l1; the DNN under a Huber loss),
                 each run only where the model it replaces is decided, as in
                 n33, with every threshold a multiple of
                 s = 1.4826 * median |y - p| (p the out-of-fold LightGBM
                 prediction on the pass's labels, n33.robust_scale);
                 ngboost_t, NGBoost with a Student-t likelihood (3 degrees of
                 freedom), paired against NGBoost with a normal likelihood
                 (model 'ngboost').
  n40/<name>   n40's arms, each paired against the decided model at its fixed
               default settings on the same labels (n40's `fixed`):
                 retuned and tuned_clean_reused (n40.tune, tuned_model; 20
                 settings plus the default, 3 scaffold-grouped inner folds of
                 the training molecules only); seed_one and seed_avg (10
                 seeds, n40.seeded_model); rf_1000_trees;
                 stack, gp_ml_nugget, gp_nugget_cv, gp_nugget_from_gamma,
                 gp_nugget_cp, lightgbm_plus_gp_residual (and lightgbm when
                 the host model is not lightgbm), gp_means_ml_nugget,
                 gp_means_per_molecule_alpha, knn_mean: models outside the
                 decided pair, each written once against every decided
                 model's fixed fit. n40's own pairs (GP_PAIRS, for example
                 gp_nugget_from_gamma against gp_ml_nugget) are read from two
                 rows that share a key and a D. The GP arms run on folds of
                 1,000 training molecules or fewer (find_best_hybrid's GP
                 cap), the two molecule-mean arms only where repeat
                 measurements exist. The Gamma noise estimate and the kNN
                 means are computed under every neighbour setting of the
                 dataset's group (n07.sweep_settings), k in 5, 10, 20 (the
                 neighbour count), both counting schemes, read through the
                 host model's neighbour spaces as n40 reads them; the host's
                 own features are not a neighbour setting (n40, 2026-10-06).
               n33 and n40 are regression only, as they were: on a
               classification set only the lnl methods run, and the reason is
               written under `unavailable`.

NOT RUN HERE, and why (each name is in the result file, `not_run_here`).
  small_loss, confident_learning, distance_selection, josrc_weighting and
  co-teaching are DC cells only (handoff 4.1): n44 produces them.
  Jo-SRC as published and the two Jo-SRC + DivideMix combinations are cut
  (author 2026-10-30, handoff 9.4).
  invvar_ngboost and huber_teacher_blend from n33 are list C cleanups (handoff
  4.3: inverse-variance weight, Huber blend of label and prediction), not
  list A methods.
  The censored likelihood, one row per measurement, the log-count feature and
  ranking within each paper are ST (n46).

THE FEATURE SPACES (author 2026-10-30, handoff section 8: "NA keeps n09's
feature sets, searched on each condition's damaged labels"; base_2 cut by the
author 2026-11-04). Per (fold, level, draw): base_1, the network's best single
representation, and hybrid and hybrid__general2, its two best hybrids with
distinct features, from one search per pass on that pass's damaged training
labels (n09._dnn_fold_entry; at level 0 the labels as delivered), built by
noise_helpers.fold_model_inputs with n09's plain network in the
accept-or-reject step; and weighted_hybrid, hybrid 1 with each feature scaled
by its importance (n09._weighted_copy). Every model and method of the pass is
trained on these same matrices, so every row of one (pass, space) shares its
features. This is a choice made in this file: n33 and n40 each fitted the
decided models on their own per-fold spaces searched on the clean labels,
which section 8 rules out (a search on clean labels used under noise leaks
them). Fitting the decided models on searches of their own, run on the damaged
labels, is the other option; it is not built.

THE LOOPS, outermost first: condition (noise_helpers.sweep_conditions: the
decided kinds of conditions_for_task left after the 2026-11-04 cuts, plus
hetero_seen, whose level is a share of the model's own out-of-fold error
(MODEL_ERROR_CONDITIONS) on a regression set, and whose levels are n40's), fold (noise_outer_folds, all 5:
author 2026-10-30), level (0 first, always run), draw (3 per non-zero level,
1 under censoring; level 0 once), feature space, method. One unit is one
(fold, level, draw); each finished unit is written to the .units checkpoint
under the run's fingerprint, and n07.RowSink skips every row already on disk.

RESIDUAL KURTOSIS (n33, "Do not lose"). For every regression unit and feature
space the unit's checkpoint carries n33._space_context's record: the robust
scale s, the residual SD and the Pearson kurtosis of y - p (3.0 for normal
residuals).

DETECTION. A method that hands out one suspicion score per training molecule
(its metadata 'suspicion_scores') is graded against the molecules whose labels
moved (noise_helpers.detection_scores), into the unit's checkpoint.

THE CLEAN BASE. qm9, qm9_mu and qm9_alpha (noise_helpers.CLEAN_BASE_DATASETS)
have computed labels and run here like any other dataset; no code path differs.
The analysis reads their level-0 rows beside the measured datasets' level-0
rows (handoff section 10): a method that raises the score on a measured
dataset at level 0 found noise that was there; one that raises QM9's is
improving training, not removing noise.

n38's NOISE-UNIT AXIS IS HERE (2026-11-02). Every kind in
MODEL_ERROR_CONDITIONS sizes its noise by lightgbm's out-of-fold residual
on the clean training labels (noise_helpers.oof_residual_sd), where every
other kind sizes it by the spread of the labels. So the two units run side
by side here, which is what n38 measured.
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
    PAPER3_CUT_CONDITIONS, without_paper3_cuts, runs_level0,
    fold_model_inputs, configuration_record, decided_levels,
    heavy_atom_counts,
    inject_condition, level_unit, draws_for, detection_scores,
    dataset_knn_group, CLEAN_BASE_DATASETS,
)
from kirby.noise_spec import GROUPED_STRATEGIES
from kirby.utils.lnl_methods import get_lnl_method, REGRESSION_ONLY_LNL

from experiments.noise import noise_cache as nc
from experiments.noise import n07_remediation as n07
from experiments.noise import noise_timing
from experiments.noise import n09_lnl_survey as n09
from experiments.noise import n32_iterative_and_dual_selection as n32
from experiments.noise import n33_robust_training as n33
from experiments.noise import n40_model_side_regularisation as n40
from experiments.noise import n42_similarity_noise_sweep as n42
from experiments.noise.d_vs_dprime import make_row, _all_scores

EXP_ID = 'n43_noise_aware_training'
RESULT_LAYOUT = 'na_2026-11-02'

NOISE_DRAWS = (0, 1, 2)

# n09's feature spaces (handoff section 8) without base_2. Adelaide, 2026-11-04:
# "okay then the cut to all". hybrid__general2 and weighted_hybrid are open
# (PAPER_3_NOISE_EXPERIMENTS.md section 3).
FIT_SPACES = tuple(s for s in n09.FIT_SPACES if s != 'base_2')
WEIGHTED_SPACE = n09.WEIGHTED_SPACE
FEATURE_SPACES = tuple(s for s in n09.ALL_SPACES if s != 'base_2')

# The lnl methods trained as published here (handoff 4.1, "Runs as" NA; 9.6
# for multistage_cleaning). n09 trains every one with the network.
LNL_MODEL = n09.BASE_RANK_MODEL
LNL_METHODS = ('early_learning', 'self_paced_learning', 'dividemix',
               'meta_weight_net', 'mentornet', 'mentornet_dd',
               'heteroscedastic_gp', 'evidential_kernel',
               'multistage_cleaning')

# n33's arms that are list A methods. invvar_ngboost and huber_teacher_blend
# are list C cleanups (NOT_RUN_HERE).
N33_LABEL_ARMS = ('yeo_johnson', 'c_mixup', 'target_noise_bagging')
N33_SETTINGS = {
    'yeo_johnson': 'standardize=True',
    'c_mixup': f'bw_sd={n33.CMIXUP_BW_SD:g}|beta={n33.CMIXUP_BETA:g}',
    'target_noise_bagging': f'bags={n33.TNB_BAGS}|scale={n33.TNB_SCALE:g}',
}
HUBER_C = tuple(float(c) for c in n33.HUBER_C)
NGBOOST_MODEL = 'ngboost'

# n40's arms on the decided models, and the ones fitted in the host model's
# features (stored under `fixed` in n40, written here against every decided
# model's fixed fit).
N40_MODEL_ARMS = ('retuned', 'tuned_clean_reused', 'seed_one', 'seed_avg',
                  'rf_1000_trees')
N40_HOST_ARMS = ('stack', 'gp_ml_nugget', 'gp_nugget_cv',
                 'gp_nugget_from_gamma', 'gp_nugget_cp',
                 'lightgbm_plus_gp_residual', 'lightgbm',
                 'gp_means_ml_nugget', 'gp_means_per_molecule_alpha',
                 'knn_mean')

NOT_RUN_HERE = {
    'lnl/small_loss': 'DC cell only (handoff 4.1): small-loss rank x delete',
    'lnl/confident_learning': 'DC cell only (handoff 4.1): cleanlab score x delete',
    'lnl/distance_selection': 'DC cell only (handoff 4.1): neighbour-label gap x delete',
    'lnl/josrc_weighting': ('DC cell (handoff 4.1): Jo-SRC test x down-weight; '
                            'not a published method, does not run in NA'),
    'n32/two_learner_exchange': 'co-teaching, DC cell (handoff 4.1)',
    'josrc': 'Jo-SRC as published: cut, author 2026-10-30 (handoff 9.4)',
    'josrc_dividemix_1': 'cut with Jo-SRC, author 2026-10-30 (handoff 9.4)',
    'josrc_dividemix_2': 'cut with Jo-SRC, author 2026-10-30 (handoff 9.4)',
    'n33/invvar_ngboost': 'list C, inverse-variance weight (handoff 4.3)',
    'n33/huber_teacher_blend': 'list C, Huber blend of label and prediction (handoff 4.3)',
    'n30/fit_aft_predict': 'censored likelihood: ST (handoff 4.1)',
    'n29/per_record': 'one row per measurement: ST (handoff 4.1)',
    'n19/log_count': 'log-count feature, plain and NGBoost: ST (handoff 4.1)',
    'n29/rank_within_paper': 'ranking within each paper: ST, never ran (handoff 4.1)',
}

REGRESSION_ONLY_REASON = {
    'n33': ('n33 is regression only: every arm changes a loss on a number or '
            'the number itself'),
    'n40': ('n40 is regression only: the Gamma estimate, Cp and the GP noise '
            'term are defined on a number'),
}


# ---------------------------------------------------------------------------
# one unit: (fold, level, draw)
# ---------------------------------------------------------------------------

class _Unit:
    """Everything one (fold, level, draw) reads, and the one place a row is
    written."""

    def __init__(self, *, ds, sink, metric, task_type, condition, level,
                 draw, fold, y0, y_clean, y_val, g_tr, cmask, errors,
                 unavailable, classes_=None):
        self.ds, self.sink, self.metric = ds, sink, metric
        self.task_type = task_type
        self.is_clf = task_type == 'classification'
        self.condition, self.level, self.draw, self.fold = (
            condition, float(level), int(draw), int(fold))
        self.y0 = np.asarray(y0)
        self.y_clean = np.asarray(y_clean)
        self.y_val = np.asarray(y_val)
        self.g_tr = None if g_tr is None else np.asarray(g_tr)
        self.cmask = None if cmask is None else np.asarray(cmask, dtype=bool)
        self.errors, self.unavailable = errors, unavailable
        self.classes_ = classes_
        self.n_rows_before = sink.n_written
        self.counts = {}
        self.record = {'per_space': {}, 'detection': {}, 'base_scores': {},
                       'n40': {}}
        self._memo = {}

    # -- a value computed once per unit, on first use --
    def ref(self, key, fn):
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]

    def groups_or_rows(self):
        return (self.g_tr if self.g_tr is not None
                else np.arange(len(self.y0)))

    def identity(self, method, setting, model_input, sim):
        return n07.row_identity(
            dataset_name=self.ds, experiment='n43', method=method,
            method_setting=setting, similarity_setting=sim,
            model_input=model_input, noise_condition=self.condition,
            level=self.level, draw=self.draw, fold=self.fold)

    def _score(self, pred):
        metric_name, metric_fn, _lib = self.metric
        pred = np.asarray(pred)
        return (float(metric_fn(self.y_val, pred)),
                _all_scores(self.y_val, pred, metric_name, self.task_type))

    def emit(self, method, setting, model, space, make_P, ref_D, ref_C,
             sim=n07.NO_NEIGHBOURS, not_run=None):
        """Write one row. make_P, ref_D and ref_C are callables returning
        held-out predictions; nothing is computed when the row is on disk.
        not_run (a reason) writes the row with score_Dprime nan and the
        reason, without calling make_P, so the cell is on disk."""
        ident = self.identity(method, setting, space, sim)
        if self.sink.is_done(ident, (model,)):
            self.counts['rows_on_disk'] = self.counts.get('rows_on_disk', 0) + 1
            return
        try:
            pD = ref_D()
            pC = ref_C()
        except Exception as exc:
            self.errors.add(f'{model}|{space}|D_or_clean', exc)
            return
        s_D, a_D = self._score(pD)
        s_C, a_C = self._score(pC)
        # the source is in the key: lnl's D is n09's network, n33's and
        # n40's D on dnn is the registry DNN, a different fit
        key = f'{method.split("/")[0]}|{model}|{space}'
        if key not in self.record['base_scores']:
            self.record['base_scores'][key] = {'score_D': s_D,
                                               'score_clean': s_C}
        s_P, a_P = float('nan'), None
        if not_run is None:
            try:
                pP = make_P()
                if pP is None:
                    not_run = 'the method gave no prediction'
                else:
                    s_P, a_P = self._score(pP)
            except Exception as exc:
                self.errors.add(f'{method}|{setting}|{model}', exc)
                not_run = f'raised {type(exc).__name__}'
        metric_name, _fn, lower_is_better = self.metric
        row = make_row(**ident, model=model, metric=metric_name,
                       lower_is_better=lower_is_better, score_D=s_D,
                       score_Dprime=s_P, score_clean=s_C, all_D=a_D,
                       all_Dprime=a_P, all_clean=a_C, dprime_not_run=not_run)
        self.sink.write([row])

    def rows_written(self):
        return self.sink.n_written - self.n_rows_before


def _fit_pred(u, model, factory, X_tr, y, X_va):
    """A decided model's held-out prediction through noise_cache when the
    model is seeded (n32.fit_predict_cached)."""
    return np.asarray(n32.fit_predict_cached(
        u.ds, model, factory, X_tr, y, X_va, u.metric[0]), dtype=float)


def _refs(u, model, factory, space, X_tr, X_va):
    """(D, clean) prediction callables for one decided model in one space,
    shared by every n33 and n40 arm of the unit."""
    D = lambda: u.ref((model, space, 'D'),
                      lambda: _fit_pred(u, model, factory, X_tr, u.y0, X_va))
    C = lambda: u.ref((model, space, 'C'),
                      lambda: _fit_pred(u, model, factory, X_tr, u.y_clean, X_va))
    return D, C


def _rng(*parts):
    return np.random.default_rng(n32.replicate_seed('n43', *parts))


# ---------------------------------------------------------------------------
# lnl methods (n09)
# ---------------------------------------------------------------------------

def lnl_space(u, space, X_tr, X_va, methods):
    seed = 42 + u.fold

    def _plain(y):
        return np.asarray(n09._run_baseline(
            X_tr, y, X_va, u.y_val, random_state=seed, is_clf=u.is_clf,
            classes_=u.classes_))

    D = lambda: u.ref((LNL_MODEL, space, 'lnl_D'), lambda: _plain(u.y0))
    C = lambda: u.ref((LNL_MODEL, space, 'lnl_C'), lambda: _plain(u.y_clean))
    for name in methods:
        if u.is_clf and name in REGRESSION_ONLY_LNL:
            u.unavailable.setdefault(
                f'lnl/{name}', 'regression only (lnl_methods.REGRESSION_ONLY_LNL)')
            continue

        def make(name=name):
            m = get_lnl_method(name, random_state=seed, is_clf=u.is_clf,
                               classes_=u.classes_)
            # Keeps the method's checkpoint-selection rows scaffold-disjoint
            # inside the training fold, as n09 does.
            m.selection_groups = u.g_tr
            res = m.train_and_predict(X_tr, u.y0, X_va, u.y_val)
            sus = (res.get('metadata') or {}).get('suspicion_scores')
            if sus is not None and u.cmask is not None and u.level > 0:
                det = detection_scores(u.cmask, sus, u.condition)
                if det is not None:
                    u.record['detection'].setdefault(space, {})[name] = det
            return res['y_pred']

        u.emit(f'lnl/{name}', 'as_published', LNL_MODEL, space, make, D, C)


# ---------------------------------------------------------------------------
# n33's arms
# ---------------------------------------------------------------------------

def n33_space(u, space, X_tr, X_va, models, factories, swaps, ngb_pair, ctx, s):
    metric_name = u.metric[0]
    have_s = bool(np.isfinite(s) and s > 0)
    y0 = np.asarray(u.y0, dtype=float)
    for m in models:
        fac = factories[m]
        D, C = _refs(u, m, fac, space, X_tr, X_va)
        for arm in N33_LABEL_ARMS:
            not_run = None
            if arm == 'target_noise_bagging' and 'resid_sd' not in ctx:
                not_run = 'no out-of-fold residual SD on this pass'

            def make(arm=arm, fac=fac, m=m):
                return n33.label_arm_predict(
                    arm, fac, X_tr, y0, X_va, metric_name, ctx,
                    _rng(arm, m, space, u.condition, f'{u.level:g}', u.draw,
                         u.fold))
            u.emit(f'n33/{arm}', N33_SETTINGS[arm], m, space, make, D, C,
                   not_run=not_run)
        for swap, parent in swaps.items():
            if parent != m:
                continue
            not_run = (None if have_s or swap == 'rf_l1' else
                       'the robust scale s is zero or not finite on this pass')

            def make(swap=swap):
                return n33._fit_predict(n33.make_swap(swap, s, HUBER_C),
                                        X_tr, y0, X_va, metric_name)
            u.emit(f'n33/{swap}', f'replaces={parent}', m, space, make, D, C,
                   not_run=not_run)
    if ngb_pair:
        def _ngb(dist, y):
            return np.asarray(n33.make_ngboost(dist).fit(X_tr, y).predict(X_va),
                              dtype=float)
        D = lambda: u.ref((NGBOOST_MODEL, space, 'D'),
                          lambda: _ngb('normal', y0))
        C = lambda: u.ref((NGBOOST_MODEL, space, 'C'),
                          lambda: _ngb('normal', np.asarray(u.y_clean, dtype=float)))
        u.emit('n33/ngboost_t',
               f'paired_against=ngboost_normal|df={n33.NGB_T_DF}',
               NGBOOST_MODEL, space, lambda: _ngb('t', y0), D, C)


# ---------------------------------------------------------------------------
# n40's arms
# ---------------------------------------------------------------------------

_CLEAN_TUNED = {}


def n40_space(u, space, X_tr, X_va, models, factories, host, neighbours,
              sim_ks, schemes, rep_ids_tr):
    """n40's arms in one feature space. `neighbours(space, X_tr, X_va)` gives
    [{'name', 'blocks', 'idx_val', 'dist_val', 'why'}], one per neighbour
    setting (blocks: n32.FoldSimilarity.distance_blocks' function; idx_val,
    dist_val: each held-out molecule's nearest training molecules)."""
    metric_name, metric_fn, lower = u.metric
    y0 = np.asarray(u.y0, dtype=float)
    g = u.groups_or_rows()
    rec = u.record['n40'].setdefault(space, {'tuned': {}, 'seed_avg': {}})
    refs = {}
    for m in models:
        fac = factories[m]
        D, C = _refs(u, m, fac, space, X_tr, X_va)
        refs[m] = (D, C)
        # retuned and tuned_clean_reused
        if m in n40.TUNE_GRIDS:
            def _tuned_noisy(m=m, fac=fac):
                return u.ref((m, space, 'tune_noisy'), lambda: n40.tune(
                    m, fac, X_tr, y0, g, metric_fn, lower,
                    n40.TUNE_N_ITER, n40.TUNE_INNER_FOLDS))

            def _tuned_clean(m=m, fac=fac):
                # The clean tuning depends on the matrix and the clean labels
                # only, so it is reused across the passes of a fold that hand
                # the same matrix (n40 reused it for single representations).
                ck = (u.ds, u.fold, m, nc.array_hash(X_tr))
                if ck not in _CLEAN_TUNED:
                    _CLEAN_TUNED[ck] = n40.tune(
                        m, fac, X_tr, np.asarray(u.y_clean, dtype=float), g,
                        metric_fn, lower, n40.TUNE_N_ITER,
                        n40.TUNE_INNER_FOLDS)
                return _CLEAN_TUNED[ck]

            for arm, tuner in (('retuned', _tuned_noisy),
                               ('tuned_clean_reused', _tuned_clean)):
                def make(m=m, fac=fac, tuner=tuner, arm=arm):
                    params, info = tuner()
                    rec['tuned'].setdefault(m, {})[arm] = {
                        'params': n40._params_for_json(params),
                        'chose_default': info['chose_default']}
                    return n40._fit_predict(n40.tuned_model(m, fac, params, y0),
                                            X_tr, y0, X_va, metric_name)
                u.emit(f'n40/{arm}',
                       f'n_iter={n40.TUNE_N_ITER}|inner_folds={n40.TUNE_INNER_FOLDS}',
                       m, space, make, D, C)
        else:
            u.unavailable.setdefault(f'n40/retuned|{m}',
                                     f'{m} has no grid in n40.TUNE_GRIDS')
        # seed averaging, or the 1,000-tree forest
        if m in n40.SEED_SUBSAMPLE:
            def _seeds(m=m, fac=fac):
                return u.ref((m, space, 'seeds'), lambda: [
                    n40._fit_predict(n40.seeded_model(m, fac, sd), X_tr, y0,
                                     X_va, metric_name)
                    for sd in range(n40.SEED_AVG_N)])

            def make_one(_s=_seeds):
                return _s()[0]

            def make_avg(_s=_seeds, m=m):
                preds = _s()
                rec['seed_avg'][m] = {'n_seeds': n40.SEED_AVG_N,
                                      'mean_sd_across_seeds': float(
                                          np.mean(np.std(preds, axis=0)))}
                return np.mean(preds, axis=0)
            u.emit('n40/seed_one', 'seeds=1', m, space, make_one, D, C)
            u.emit('n40/seed_avg', f'seeds={n40.SEED_AVG_N}', m, space,
                   make_avg, D, C)
        elif m == 'rf':
            u.emit('n40/rf_1000_trees', f'trees={n40.RF_BIG_TREES}', m, space,
                   lambda fac=fac: n40._fit_predict(
                       fac().set_params(n_estimators=n40.RF_BIG_TREES),
                       X_tr, y0, X_va, metric_name), D, C)
        else:
            u.unavailable.setdefault(
                f'n40/seed_avg|{m}',
                f'{m} has no seed or tree setting in n40 (SEED_SUBSAMPLE, rf)')

    def host_rows(arm, setting, make, sim=n07.NO_NEIGHBOURS, not_run=None):
        """One model outside the decided pair, written against every decided
        model's fixed fit (n40's stack is paired the same way)."""
        P = lambda: u.ref(('host', space, arm, setting, sim), make)
        for m in models:
            D, C = refs[m]
            u.emit(f'n40/{arm}', setting, m, space, P, D, C, sim=sim,
                   not_run=not_run)

    # the stack: non-negative least squares on the decided models'
    # out-of-fold predictions, applied to their fixed held-out predictions
    if len(models) > 1:
        def make_stack():
            from scipy.optimize import nnls
            P = np.column_stack([n40._oof_model(factories[m], X_tr, y0, g,
                                                n40.OOF_SPLITS)
                                 for m in models])
            w, _ = nnls(P, y0)
            rec['stack_weights'] = dict(zip(models, map(float, w)))
            if w.sum() <= 0:
                return None
            return np.column_stack([refs[m][0]() for m in models]) @ w
        host_rows('stack', f'models={"+".join(models)}', make_stack)
    else:
        u.unavailable.setdefault('n40/stack', 'one decided model: nothing to stack')

    # the neighbour settings: the Gamma estimate per (setting, k) and the
    # neighbour lists of the held-out molecules
    settings = list(neighbours(space, X_tr, X_va)) if neighbours else []
    if not settings:
        u.unavailable.setdefault('n40/knn_mean', 'no neighbour setting on this pass')
    sigma2 = {}
    rec['gamma_by_similarity'] = {}
    for st in settings:
        if st.get('blocks') is None:
            rec['gamma_by_similarity'][st['name']] = {'unavailable': st.get('why')}
            continue
        for k in sim_ks:
            try:
                s2, gk = u.ref(('gamma', space, st['name'], k),
                               lambda st=st, k=k: n40.gamma_sigma2(
                                   X_tr, y0, seed=u.fold, k_max=k,
                                   dist_blocks=st['blocks']))
            except Exception as exc:
                u.errors.add(f'n40/gamma@{st["name"]}', exc)
                continue
            rec['gamma_by_similarity'][f'{st["name"]}|k{k}'] = (
                None if gk is None else
                {kk: gk[kk] for kk in ('intercept', 'k_max', 'n_used')})
            if s2 is not None:
                sigma2[(st['name'], k)] = s2
    for st in settings:
        iv, dv = st.get('idx_val'), st.get('dist_val')
        if iv is None:
            continue
        for sch in schemes:
            for k in sim_ks:
                if k > np.shape(iv)[1]:
                    continue
                host_rows('knn_mean', f'k={k}|scheme={sch}',
                          lambda k=k, sch=sch, iv=iv, dv=dv:
                          n40.knn_scheme_predict(y0, iv, dv, k, sch),
                          sim=st['name'])

    # the GP arms and the kriging, under find_best_hybrid's GP cap
    cap = n40.gp_max_samples()
    n_tr = len(y0)
    if cap is not None and n_tr > cap:
        why = (f'{n_tr} training molecules, above the GP cap of {cap} '
               f'(DEFAULT_MODELS["gp"]["_max_samples"])')
        for a in ('gp_ml_nugget', 'gp_nugget_cv', 'gp_nugget_from_gamma',
                  'gp_nugget_cp', 'lightgbm_plus_gp_residual',
                  'gp_means_ml_nugget', 'gp_means_per_molecule_alpha'):
            u.unavailable.setdefault(f'n40/{a}|{space}', why)
        return

    def _gp():
        def fit():
            gp = n40.make_gp().fit(X_tr, y0)
            egp = n40._EigenGP(gp.kernel_.k1, X_tr, y0)
            white = float(gp.kernel_.k2.noise_level)
            grid = [white * mm for mm in n40.GP_NUGGET_MULT]
            rec['gp'] = {'ml_noise_normalised': white,
                         'ml_noise_label_units': white * egp.sd ** 2}
            return gp, egp, grid
        return u.ref(('gp', space), fit)

    host_rows('gp_ml_nugget', 'noise_term=marginal_likelihood',
              lambda: _gp()[0].predict(X_va))

    def make_cv():
        gp, egp, grid = _gp()
        a_cv, _sc = n40.gp_cv_choice(gp.kernel_.k1, X_tr, y0, g, grid,
                                     metric_fn, lower,
                                     inner_folds=n40.TUNE_INNER_FOLDS)
        rec['gp']['cv_noise_label_units'] = a_cv * egp.sd ** 2
        return egp.predict(X_va, a_cv)
    host_rows('gp_nugget_cv', f'grid_mult={len(n40.GP_NUGGET_MULT)}', make_cv)

    for (sname, k), s2 in sigma2.items():
        def make_gamma(s2=s2):
            _gp_, egp, _grid = _gp()
            return egp.predict(X_va, s2 / egp.sd ** 2)

        def make_cp(s2=s2):
            _gp_, egp, grid = _gp()
            a_g = s2 / egp.sd ** 2
            a_cp, _c = n40.gp_cp_choice(egp, y0, s2, grid + [a_g])
            return egp.predict(X_va, a_cp)
        host_rows('gp_nugget_from_gamma', f'k={k}', make_gamma, sim=sname)
        host_rows('gp_nugget_cp', f'k={k}', make_cp, sim=sname)

    # regression kriging on lightgbm's out-of-fold residuals
    def _lgbm_fixed():
        if host == 'lightgbm' and 'lightgbm' in refs:
            return refs['lightgbm'][0]()
        from kirby.find_best_hybrid import get_model
        return u.ref(('lightgbm_host', space), lambda: n40._fit_predict(
            get_model('lightgbm', 'regression'), X_tr, y0, X_va, metric_name))

    def make_krig():
        p_oof = nc.cached_oof_predict(u.ds, X_tr, y0, u.g_tr,
                                      n_splits=n40.OOF_SPLITS)
        gp_r = n40.make_gp().fit(X_tr, y0 - p_oof)
        return _lgbm_fixed() + gp_r.predict(X_va)
    host_rows('lightgbm_plus_gp_residual', 'residual_gp', make_krig)
    if host != 'lightgbm':
        host_rows('lightgbm', 'fixed_defaults', _lgbm_fixed)

    # molecule means, with and without a per-molecule noise term
    if rep_ids_tr is None or len(np.unique(rep_ids_tr)) == len(rep_ids_tr):
        u.unavailable.setdefault(
            'n40/gp_means', 'no repeat measurements on this dataset')
        return
    first, ym, kcount, s2e, dof = n40.molecule_means(y0, np.asarray(rep_ids_tr))
    if cap is not None and len(ym) > cap:
        u.unavailable.setdefault(f'n40/gp_means|{space}',
                                 f'{len(ym)} training molecules, above the GP cap of {cap}')
        return
    if not np.isfinite(s2e):
        u.unavailable.setdefault(f'n40/gp_means|{space}',
                                 'no training molecule has two measurements')
        return
    Xm = X_tr[first]
    rec['gp_means'] = {'n_molecules': int(len(ym)),
                       'within_variance_label_units': float(s2e),
                       'within_dof': int(dof), 'max_count': int(kcount.max())}
    host_rows('gp_means_ml_nugget', 'molecule_means',
              lambda: n40.make_gp().fit(Xm, ym).predict(X_va))
    var_m = float(np.var(ym)) or 1.0
    host_rows('gp_means_per_molecule_alpha', 'molecule_means|alpha=s2e/k',
              lambda: n40.make_gp((s2e / kcount) / var_m).fit(Xm, ym).predict(X_va))


# ---------------------------------------------------------------------------
# the unit
# ---------------------------------------------------------------------------

def run_unit(u, spaces, *, lnl_methods=LNL_METHODS, n33_models=(),
             n40_models=(), factories=None, swaps=None, ngb_pair=False,
             host=None, neighbours=None, sim_ks=(5, 10, 20),
             schemes=('rank', 'inverse_square'), rep_ids_tr=None):
    """Every method of n43 on one (fold, level, draw). `spaces` is
    {feature space: (X_train, X_held_out)}."""
    factories = factories or {}
    swaps = swaps or {}
    if u.is_clf:
        for src, why in REGRESSION_ONLY_REASON.items():
            u.unavailable.setdefault(f'{src}/*', why)
    for space in FEATURE_SPACES:
        if space not in spaces:
            continue
        X_tr, X_va = spaces[space]
        t0 = time.time()
        noise_timing.set_context(model_input=space)
        try:
            with noise_timing.timed('network_methods'):
                lnl_space(u, space, X_tr, X_va, lnl_methods)
        except Exception as exc:
            u.errors.add(f'lnl|{space}', exc)
        if u.is_clf:
            continue
        # p, s, the residual SD and the residual kurtosis (n33), once per
        # space. invvar_ngboost and huber_teacher_blend are not asked for, so
        # no NGBoost spread or teacher is fitted here.
        sc_err = {}
        ctx, sp_rec, s = n33._space_context(
            u.ds, X_tr, np.asarray(u.y0, dtype=float), u.g_tr,
            n33.OOF_SPLITS, list(N33_LABEL_ARMS), sc_err, space)
        for k, v in sc_err.items():
            u.errors.add(k, v)
        u.record['per_space'][space] = dict(
            sp_rec, n_features=int(np.shape(X_tr)[1]))
        if n33_models:
            try:
                with noise_timing.timed('robust_training_methods'):
                    n33_space(u, space, X_tr, X_va, n33_models, factories, swaps,
                              ngb_pair, ctx, s)
            except Exception as exc:
                u.errors.add(f'n33|{space}', exc)
        if n40_models:
            try:
                with noise_timing.timed('neighbour_model_methods'):
                    n40_space(u, space, X_tr, X_va, n40_models, factories, host,
                              neighbours, sim_ks, schemes, rep_ids_tr)
            except Exception as exc:
                u.errors.add(f'n40|{space}', exc)
        u.record['per_space'][space]['seconds'] = round(time.time() - t0, 1)


def _n40_neighbours(fsim, host, sim_settings, sim_ks):
    """The neighbour-setting provider n40_space reads, from one
    n32.FoldSimilarity, as n40 reads it: distance blocks for the Gamma
    estimate, held-out neighbour lists for the kNN means. The host model's own
    features are skipped (n40, 2026-10-06)."""
    from experiments.noise import similarity_space as ss
    kmax = max(max(n40.KNN_K_GRID), max(sim_ks))
    memo = {}

    def provide(space, X_tr, X_va):
        out = []
        for st in sim_settings:
            if st.space == ss.MODEL_INPUT_SPACE:
                out.append({'name': st.name, 'blocks': None, 'idx_val': None,
                            'dist_val': None,
                            'why': "model_input is the host model's own "
                                   "features, cut as a neighbour space "
                                   "(2026-10-06)"})
                continue
            key = (st.name, space) if st.space == ss.FOREST_SPACE else st.name
            if key not in memo:
                blocks, _n, why = fsim.distance_blocks(st, host, X_tr)
                wide = ss.SimilaritySetting(st.space, st.weighting, st.metric,
                                            k_max=kmax)
                nb = fsim.neighbours(wide, host, X_tr, X_va)
                ok = bool(nb.get('available'))
                memo[key] = {'name': st.name, 'blocks': blocks,
                             'idx_val': nb['idx_val'] if ok else None,
                             'dist_val': nb['dist_val'] if ok else None,
                             'why': why or (None if ok else nb.get('reason'))}
            out.append(memo[key])
        return out
    return provide


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

# The kind whose level is a share of the model's own out-of-fold error rather
# than a share of the label spread: hetero_seen gives each molecule a noise SD
# that grows with its heavy-atom count. Adelaide, 2026-11-04: "Keep hetero_seen";
# the other kinds sized this way were cut. Levels come from
# n40._condition_levels, which reads noise_helpers.relative_levels: 0, 0.1, 0.3
# and 1.0 times the out-of-fold mean squared error.
MODEL_ERROR_CONDITIONS = ('hetero_seen',)


def default_conditions(data):
    """The decided kinds left after the 2026-11-04 cuts; on a regression set
    hetero_seen (MODEL_ERROR_CONDITIONS) as well."""
    conds = without_paper3_cuts(conditions_for_task(data))
    if data.get('task_type') != 'classification':
        conds += [c for c in MODEL_ERROR_CONDITIONS if c not in conds]
    return conds


def run(data, config, output_dir, dataset_name, **kwargs):
    """NA across every condition this dataset can be given."""
    return sweep_conditions(
        _run_one_condition, EXP_ID, data, config, output_dir, dataset_name,
        default_conditions=default_conditions(data), kwargs=kwargs,
        groups_needed=GROUPED_STRATEGIES, cut_conditions=PAPER3_CUT_CONDITIONS)


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    print("\n" + "=" * 70)
    print("EXPERIMENT N43, NOISE-AWARE TRAINING: LIST A METHODS AS PUBLISHED")
    print("=" * 70)
    t_start = time.time()

    task_type = data['task_type']
    is_clf = task_type == 'classification'
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    metric = get_domain_metric(data)
    metric_name, _fn, lower_is_better = metric
    condition = kwargs.get('noise_condition') or default_conditions(data)[0]
    levels = (tuple(float(x) for x in decided_levels(data, dataset_name, condition))
              if is_clf else
              tuple(float(x) for x in n40._condition_levels(data, dataset_name,
                                                            condition)))
    classes_ = np.unique(labels) if is_clf else None
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

    # The models of n33 and n40 (regression only): the dataset's two decided
    # models; n33 leaves knn out. A caller's `models` can only narrow them.
    decided = list(decided_models(dataset_name, 'general'))
    asked = kwargs.get('models')
    narrowed = [m for m in decided if (not asked or m in asked)]
    n33_models = [] if is_clf else [m for m in narrowed if m != 'knn']
    n40_models = [] if is_clf else list(narrowed)
    factories = get_model_factories(sorted(set(n33_models + n40_models)),
                                    task_type=task_type, n_samples=len(labels))
    no_factory = [m for m in set(n33_models + n40_models) if m not in factories]
    n33_models = [m for m in n33_models if m in factories]
    n40_models = [m for m in n40_models if m in factories]
    swaps = {k: v for k, v in n33.build_swaps(HUBER_C).items() if v in n33_models}
    swaps_not_run = {}
    if 'rf_l1' in swaps and not get_model_factories(
            ['rf_l1'], task_type=task_type, n_samples=len(labels)):
        swaps_not_run['rf_l1'] = (f'rf_l1 is capped at 5,000 molecules in '
                                  f'DEFAULT_MODELS and this dataset has {len(labels)}')
        swaps.pop('rf_l1')
    ngb_pair = bool(n33_models) and n33.ngboost_host(n33_models) is not None
    host = (None if not n40_models else
            n40.HOST_MODEL if n40.HOST_MODEL in n40_models else n40_models[0])
    lnl_methods = list(kwargs.get('lnl_methods') or LNL_METHODS)

    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    n_molecules = len(set(smiles)) if has_smiles else len(labels)
    sim_settings = list(n07.sweep_settings(dataset_name, n_molecules, has_smiles))
    sim_ks = tuple(int(k) for k in n07.sweep_ks())
    schemes = tuple(n07.sweep_schemes())
    rep_ids = None
    if not is_clf:
        from experiments.noise._ground_truth_providers import build_replicate_groups
        try:
            rep_ids = build_replicate_groups(data)
        except Exception:
            rep_ids = None

    folds, n_folds = noise_outer_folds(data, config)
    draws = tuple(int(d) for d in draws_for(condition, NOISE_DRAWS))
    # level 0 runs once, in level0_owner's file (noise_helpers.level0_rows_path)
    passes = [(lv, int(NOISE_DRAWS[0])) for lv in levels
              if lv == 0.0 and runs_level0(data, condition)]
    passes += [(lv, int(dr)) for lv in levels if lv != 0.0 for dr in draws]
    noise_groups = noise_groups_for(data)[0]

    file_cond = f'{condition}__{RESULT_LAYOUT}'
    sink = n07.RowSink(output_dir, EXP_ID, dataset_name, file_cond)
    fingerprint = {'layout': RESULT_LAYOUT,
                   'input': input_fingerprint(data, config),
                   'levels': list(levels), 'draws': list(draws),
                   'n_folds': int(n_folds), 'lnl_methods': lnl_methods,
                   'n33_models': n33_models, 'n40_models': n40_models,
                   'swaps': sorted(swaps), 'settings': [s.name for s in sim_settings],
                   'ks': list(sim_ks), 'schemes': list(schemes),
                   'condition': condition}
    units = n07.FoldCheckpoint(output_dir, EXP_ID, dataset_name, file_cond,
                               tag='units', fingerprint=fingerprint)
    errors = n42._Errors()
    unavailable, provenance = {}, {}
    n_done = n_resumed = 0

    print(f"  condition {condition}; levels {list(levels)} "
          f"({level_unit(condition, is_clf)}); {len(passes)} (level, draw) "
          f"passes; {n_folds} outer folds; lnl methods {lnl_methods} on "
          f"{LNL_MODEL}; n33 models {n33_models}; n40 models {n40_models} "
          f"(host {host}); rows -> {sink.path}")

    for fold_i, train_idx, val_idx in folds:
        train_idx = np.asarray(train_idx)
        val_idx = np.asarray(val_idx)
        y_clean = labels[train_idx]
        y_val = labels[val_idx]
        g_tr = None if groups is None else np.asarray(groups)[train_idx]
        print(f"\n  Fold {fold_i} of {n_folds}: {len(train_idx)} training, "
              f"{len(val_idx)} held-out rows")
        for level, draw in passes:
            ukey = [int(fold_i), float(level), int(draw)]
            if units._k(ukey) in units.entries:
                n_resumed += 1
                continue
            t0 = time.time()
            noise_timing.set_context(fold=int(fold_i), level=float(level),
                                     draw=int(draw), model_input=None)
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

            # n09's five feature spaces, searched on this pass's labels
            try:
                with noise_timing.timed('search'):
                    entry = n09._dnn_fold_entry(
                        data, dataset_name, fold_i=fold_i, train_idx=train_idx,
                        val_idx=val_idx, n_folds=n_folds, y_search_train=y0,
                        is_clf=is_clf, classes_=classes_, task_type=task_type,
                        metric=metric)
            except Exception as exc:
                entry = {'failed': f'the per-fold dnn search raised '
                                   f'{type(exc).__name__}: {exc}'}
            if entry.get('failed'):
                errors.add(f'search|fold={fold_i}', entry['failed'])
                units.put(ukey, {'failed': entry['failed']})
                continue

            def _factory(fold_i=fold_i):
                return n09._BaselineDNN(random_state=42 + int(fold_i),
                                        is_clf=is_clf, classes_=classes_)
            spaces, unav, names = fold_model_inputs(
                data, dataset_name, LNL_MODEL, entry, train_idx=train_idx,
                val_idx=val_idx, y_train=y0, factory=_factory, metric=metric,
                spaces=FIT_SPACES)
            unav = dict(unav)
            if 'hybrid' in spaces:
                try:
                    Xw_tr, Xw_va, why = n09._weighted_copy(
                        data, dataset_name, entry, train_idx=train_idx,
                        val_idx=val_idx, y_train=y0, factory=_factory,
                        metric=metric, X_hyb_tr=spaces['hybrid'][0],
                        X_hyb_va=spaces['hybrid'][1])
                except Exception as exc:
                    Xw_tr, why = None, f'{type(exc).__name__}: {exc}'
                if Xw_tr is not None:
                    spaces[WEIGHTED_SPACE] = (Xw_tr, Xw_va)
                else:
                    unav[WEIGHTED_SPACE] = why
            else:
                unav.setdefault(WEIGHTED_SPACE, 'no hybrid 1 on this pass')
            for sp, why in unav.items():
                unavailable.setdefault(f'fold{fold_i}|level={level:g}|draw={draw}|{sp}',
                                       str(why))

            # n40's neighbour spaces: the network's own recipes plus the
            # fold's distance and similarity hybrids (n09._neighbour_recipes),
            # read through an n32.FoldSimilarity for the host model as n40
            # reads them, with this pass's matrices handed over
            neighbours = None
            if n40_models and sim_settings:
                try:
                    recipes = dict(entry.get('recipes') or {})
                    nb_rec, _nb_unav, _nb_src = n09._neighbour_recipes(
                        data, dataset_name, fold_i=fold_i,
                        train_idx=train_idx, n_folds=n_folds)
                    recipes.update(nb_rec)
                    fsim = n32.fold_similarity(
                        dataset_name, data, reps=data['reps'],
                        train_idx=train_idx, val_idx=val_idx, y_train=y0,
                        groups_train=(g_tr if g_tr is not None
                                      else np.arange(len(train_idx))),
                        base_by_model={host: list(entry.get('base') or [])},
                        recipe_for_space_by_model={host: recipes},
                        model_factories=factories, metric=metric,
                        task_type=task_type)
                    for sp in FIT_SPACES:
                        if sp in spaces:
                            fsim._spaces[(host, sp)] = (spaces[sp][0],
                                                        spaces[sp][1], None)
                    neighbours = _n40_neighbours(fsim, host, sim_settings, sim_ks)
                except Exception as exc:
                    errors.add('n40/neighbour_spaces', exc)

            u = _Unit(ds=dataset_name, sink=sink, metric=metric,
                      task_type=task_type, condition=condition, level=level,
                      draw=draw, fold=fold_i, y0=y0, y_clean=y_clean,
                      y_val=y_val, g_tr=g_tr, cmask=cmask, errors=errors,
                      unavailable=unavailable, classes_=classes_)
            try:
                run_unit(u, spaces, lnl_methods=lnl_methods,
                         n33_models=n33_models, n40_models=n40_models,
                         factories=factories, swaps=swaps, ngb_pair=ngb_pair,
                         host=host, neighbours=neighbours, sim_ks=sim_ks,
                         schemes=schemes,
                         rep_ids_tr=(None if rep_ids is None
                                     else np.asarray(rep_ids)[train_idx]))
            except Exception as exc:
                errors.add('unit', exc)
            units.put(ukey, {
                'rows_written': int(u.rows_written()),
                'seconds': round(time.time() - t0, 1),
                'configuration': convert_numpy(configuration_record(entry, names)),
                'search_grid': entry.get('search_grid'),
                'condition_suffix': entry.get('condition_suffix'),
                'spaces': sorted(spaces),
                'per_space': convert_numpy(u.record['per_space']),
                'detection': convert_numpy(u.record['detection']),
                'base_scores': convert_numpy(u.record['base_scores']),
                'n40': convert_numpy(u.record['n40']),
                'counts': u.counts})
            n_done += 1
            print(f"    fold {fold_i} level {level:g} draw {draw}: "
                  f"{u.rows_written()} rows, {time.time() - t0:.0f}s")

    # summary: each D model's score at each level, per space, over folds
    by = {}
    for kjson, v in units.entries.items():
        try:
            f_, lv_, _dr = json.loads(kjson)
        except Exception:
            continue
        for key, bs in ((v or {}).get('base_scores') or {}).items():
            sD, sC = bs.get('score_D'), bs.get('score_clean')
            if sD is None or not np.isfinite(sD):
                continue
            d = by.setdefault(key, {}).setdefault(f'{float(lv_):g}', {})
            d.setdefault('D', {}).setdefault(int(f_), []).append(float(sD))
            if sC is not None and np.isfinite(sC):
                d.setdefault('D_minus_clean', {}).setdefault(int(f_), []).append(
                    float(sD) - float(sC))
    summary = {key: {lv: {k: fold_spread([float(np.mean(x))
                                          for _f, x in sorted(byf.items())])
                          for k, byf in d.items()}
                     for lv, d in per.items()}
               for key, per in by.items()}

    results = {
        'experiment': EXP_ID, 'result_layout': RESULT_LAYOUT,
        'metric': metric_name, 'lower_is_better': bool(lower_is_better),
        'noise_condition': condition, 'task_type': task_type,
        'levels': list(levels), 'level_unit': level_unit(condition, is_clf),
        'draws_per_nonzero_level': len(draws),
        'passes': [[float(lv), int(dr)] for lv, dr in passes],
        'n_folds': int(n_folds), 'folds_run': [int(f) for f, _t, _v in folds],
        'knn_group': dataset_knn_group(data, dataset_name),
        'is_clean_base_dataset': dataset_name in CLEAN_BASE_DATASETS,
        'feature_spaces': list(FEATURE_SPACES),
        'feature_spaces_note': (
            "n09's five spaces, from one network search per (fold, level, "
            "draw) on that pass's training labels; every model and method of "
            "the pass is trained on the same matrices."),
        'lnl_methods': lnl_methods, 'lnl_model': LNL_MODEL,
        'n33_models': n33_models, 'n33_label_arms': list(N33_LABEL_ARMS),
        'n33_swaps': swaps, 'n33_swaps_not_run': swaps_not_run,
        'n33_ngboost_pair': ['ngboost_normal', 'ngboost_t'] if ngb_pair else [],
        'huber_c': list(HUBER_C),
        'n40_models': n40_models, 'n40_host_model': host,
        'n40_model_arms': list(N40_MODEL_ARMS),
        'n40_host_arms': list(N40_HOST_ARMS),
        'n40_pairs_read_from_rows': [list(p) for p in n40.GP_PAIRS],
        'models_decided': decided, 'models_without_factory': no_factory,
        'similarity_settings': [s.name for s in sim_settings],
        'similarity_ks': list(sim_ks), 'similarity_schemes': list(schemes),
        'not_run_here': NOT_RUN_HERE,
        'summary_by_level': summary,
        'units_computed_this_run': int(n_done), 'units_resumed': int(n_resumed),
        'rows': sink.summary(),
        'errors': errors.d, 'unavailable': unavailable,
        'noise_provenance': provenance,
        'checkpoint': {'units': units.path},
        'seconds': round(time.time() - t_start, 1),
    }
    print(f"\n  N43 {condition}: {n_done} units computed, {n_resumed} resumed, "
          f"{sink.n_written} rows written, {len(errors.d)} distinct errors, "
          f"{time.time() - t_start:.0f}s")
    return results


def main():
    import argparse
    from experiments.noise.noise_helpers import run_all
    ap = argparse.ArgumentParser(description='N43 on one dataset through '
                                             'noise_helpers.run_all')
    ap.add_argument('dataset')
    ap.add_argument('--output', required=True)
    ap.add_argument('--noise-condition', default=None)
    args = ap.parse_args()
    run_all(dataset_name=args.dataset, output_dir=args.output,
            experiments=['n43'], noise_condition=args.noise_condition)


if __name__ == '__main__':
    main()
