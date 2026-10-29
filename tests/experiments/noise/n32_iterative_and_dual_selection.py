#!/usr/bin/env python3
"""N32: trimmed refits, two learners picking for each other, and better detectors.

Paper: Noise.

WHERE THIS CAME FROM. The second research report (2026-09-24, "Label-Noise
Mitigation ... Across Signal-to-Noise Regimes") proposed, for datasets where the
model's out-of-fold error is small:

  H2  ITLM (Shen & Sanghavi 2019): fit, keep the lowest-loss share alpha of the
      training molecules, refit on them, repeat 3-5 rounds.
  H3  a cleanlab-style score: the residual divided by the model's own predicted
      spread, instead of the raw residual.
  H4  shrink the FLAGGED labels only toward the out-of-fold prediction, by
      w = sn^2 / (sn^2 + sm^2).
  H6  two different learners, each choosing the low-loss molecules the OTHER
      one trains on.
  E   conformal scores as a detector.
  O   data valuation (Data Shapley) as a detector.

What already existed, read 2026-09-24: n09 `multistage_cleaning` is 3 rounds at
a fixed 10% drop on IN-SAMPLE loss; n09 `small_loss` is one learner;
`CoTeachingFilter` in src/kirby/utils/noise_mitigation_methods.py is not used by
this directory; n07 `js_shrink` shrinks EVERY row. None divides a residual by a
predicted spread.

THE DETECTORS. Each gives one suspicion score per training molecule, and each is
graded by `detection_scores` against the molecules the injector moved (AUROC,
average precision, precision at the true count). No detector reads the mask.

  oof_residual              |y - p|, p the group-aware out-of-fold LightGBM
                            prediction (kirby.utils.noisy_labels.oof_predict).
  ngboost_rescaled          |y - mu| / sigma from ESTIMATOR_REGISTRY['ngboost'],
                            which keeps its out-of-fold mu and sigma on the
                            fitted object (predictions_, sigma_). The registry's
                            own score is sigma alone, which reads no label; the
                            division is done here.
  quantile_forest_rescaled  |y - median| / spread from
                            ESTIMATOR_REGISTRY['quantile_forest']. Its range is
                            the 10th to 90th percentile of the labels in the
                            leaves, so spread = width / (2 * 1.2816), the SD a
                            normal with that range would have. The registry's own
                            score is the distance OUTSIDE the range, which is 0
                            for most molecules; the division is done here.
  conformal                 |y - pred| / uncertainty from
                            ESTIMATOR_REGISTRY['conformal'] (a forest's per-tree
                            spread). This is the conformity score the report's
                            conformal outlier p-values are computed from; the
                            p-value is decreasing in it within one calibration
                            fold. The registry's own score is the interval
                            WIDTH, which reads no label.
  knn_shapley               minus each molecule's Shapley value to a k-nearest-
                            neighbour predictor, summed with every other training
                            molecule as the test point. See knn_shapley_values.
                            Scored under every neighbour setting and k (5, 10,
                            20); see THE NEIGHBOUR SPACES below.
  cleanlab                  1 - cleanlab's regression label quality
                            (noise_helpers.cleanlab_suspicion, lightgbm inside).
                            Added 2026-09-25 from the third research report: the
                            rescaled detectors above are cleanlab-STYLE scores;
                            this is cleanlab's own, with its bootstrap estimate
                            of the model's uncertainty.

THE METHODS. Each of the dataset's two models is fitted on what each method
leaves, in each of its 4 configurations, and scored on held-out molecules
against their UNTOUCHED labels.

  clean                     the training labels before injection. The most any
                            method can recover.
  no_cleaning               the corrupted labels as they are. The arm to beat.
  itlm_a<alpha>             4 rounds. Each round: a LightGBM (oof_predict's
                            settings) is fitted on the kept molecules; a kept
                            molecule's loss is its OUT-OF-FOLD squared error, a
                            dropped molecule's is its error under the model
                            fitted on all kept molecules (the report's snippet);
                            the lowest-loss share alpha is kept. The overlap of
                            the kept sets of consecutive rounds is written.
  itlm_a<alpha>_random/seed<k>
                            the same NUMBER of molecules removed at random. The
                            control, for n24's reason: a smaller training set is
                            a different fit, and sometimes a better one.
  two_learner_share<s>      lightgbm and svm (the suite's own factories). Each
                            scores every molecule out of fold; each hands the
                            other its lowest-loss share; each refits on what it
                            was handed and rescores; 2 exchange rounds. These two
                            are part of the method, not downstream models. Each
                            downstream model is then fitted on the INTERSECTION
                            of the two learners' final kept sets -- the molecules
                            both would keep. The union was the other option; with
                            the two learners agreeing on most molecules it would
                            remove almost nothing. How far the two flagged sets
                            agree is written as counts. Each learner flags the
                            share s of the training molecules and keeps 1 - s.
  two_learner_share<s>_random/seed<k>
                            the same number removed at random.
  shrink_flagged_one_weight_share<s>
                            the share s (the top of |y - p|) moved toward p
                            by ONE weight w = sn^2 / (sn^2 + sm^2). sm^2 is the
                            mean square of y - p over the UNFLAGGED molecules;
                            sn^2 is how far the flagged molecules' mean square
                            exceeds it, floored at 0. Mean squares, not
                            variances: a one-direction shift has a small
                            variance and a large mean square. Both are estimated from the
                            corrupted labels; the injector's size is NOT read.
  shrink_flagged_per_row_share<s>
                            the same flagged share, each molecule with its own
                            w_i = max(r_i^2 - sm^2, 0) / (max(r_i^2 - sm^2, 0)
                            + sm^2).

THE FLAGGED SHARE s is swept over FLAG_SHARES = 0.05, 0.10, 0.20 of the
training molecules, the grid n24 uses (its FLAG_RATES), the same at every
condition and level. Owner's decision 2026-09-28: no method may read what only
exists because the noise was injected -- how many molecules moved, the
condition's nominal share, or the injected variance. A real user knows none of
them. Until then the share was the condition's nominal share (FLAG_SHARE
below); FLAG_SHARE is now only written beside the results, for reading, and
never reaches a method. ITLM's alphas were already a fixed grid.

TAIL SCORES. Every scored fit also writes `tail_scores`: rmse on the held-out
molecules whose clean label is in the top and the bottom tenth, and the
calibration slope. The report warns that trimming and relabelling at high R^2
remove genuine extremes; the overall rmse can hide that.

WHERE THE FEATURES COME FROM (2026-10-05, chat 5, HANDOFF_EXPERIMENT_ALIGNMENT
2.4). The models are the dataset's two decided models (DECISIONS.md Step 1,
decided_grid.decided_models). For each outer fold
(noise_helpers.noise_outer_folds), searches on that fold's training rows with
the clean training labels (noise_helpers.fold_configurations) give each model
its 2 best base representations (`base_1`, `base_2`) and its 2 best general
hybrids (`hybrid`, `hybrid__general2`): 2 models x 4 = 8 fitted
configurations. Each pass rebuilds the hybrids on the fold's training rows from
that pass's CORRUPTED training labels (noise_helpers.fold_model_inputs), and
every detector and method shares them. `clean` is the clean labels in
corrupted-label features. Until 2026-10-05 the recipes and the base
representations were read once from the whole-dataset stored searches, for
five models, with up to six hybrids per model.

THE NEIGHBOUR SPACES. knn_shapley is the one detector here that finds
neighbours. It is scored under every decided neighbour setting
(KNN_SEARCH_SPACE_RECORD: the dataset group's spaces and distances, 5, 10 and
20 neighbours, scaffold-mates not excluded). Its spaces include the top hybrid
of the per-fold distance search (`hybrid__distance1`) and the top hybrid of the
per-fold similarity search (`hybrid_similarity`); those two are neighbour
spaces only and no model is fitted on them.

WHAT THIS CANNOT SAY. The noise is injected, so every number is confined to the
named condition. Nothing here measures real label error.

THE NOISE GRID (KNN_SEARCH_SPACE_RECORD, "The noise grid", Adelaide
2026-10-05). The decided kinds where detection is defined (some molecules
move, some do not): outlier_p05, censoring, shift_only_p10 and
sparse_dprime_up_p10. gaussian and grouped_shifted move every training label,
so there is nothing for a detector to find and n32 skips them. Levels by
dataset group (noise_helpers.decided_levels), 3 draws per non-zero level
(one under censoring, noise_helpers.draws_for), level 0 once and run last in
each fold.

Usage:
    python n32_iterative_and_dual_selection.py --dataset logd --mode full
    python n32_iterative_and_dual_selection.py --dataset herg_ki --mode full \
        --noise-condition shift_only_p10
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import time
import zlib

import numpy as np
from scipy.stats import norm

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories, DOWNSTREAM_MODELS,
    base_rank_groups,
)
from experiments.dataset_policy import noise_groups_for
from experiments.decided_grid import decided_models
from experiments.noise.noise_helpers import (
    detection_scores, detection_is_defined, tail_scores, sweep_conditions,
    DETECTION_IS_DEFINED, SHIFT_ONLY_CONDITIONS, RELATIVE_CONDITIONS,
    draws_for,
)
# CHAT 5 ALIGNMENT (2026-10-05, HANDOFF_EXPERIMENT_ALIGNMENT 2.4): folds, the
# per-fold configurations, the decided noise grid and one injection call. They
# replace cv_split, get_best_search_for_downstream,
# pool_recipe_per_downstream_model, pool_recipes_by_family,
# noise_base_reps_by_model, group_models_by_recipe, noise_levels and the three
# injectors this file called itself.
from experiments.noise.noise_helpers import (
    noise_outer_folds, fold_spread, fold_configurations, fold_model_inputs,
    configuration_record, inject_condition, decided_conditions,
    decided_levels, dataset_knn_group, level_unit, CONFIGURATION_SPACES,
)
from experiments.noise.n07_remediation import _predict_for_metric
from kirby.noise_spec import GROUPED_STRATEGIES
from kirby.scaling import FeatureScaler
from kirby.utils.noisy_labels.core import (
    make_model as make_oof_lgbm, group_kfold_splitter, _safe_n_splits,
)

# outlier_only_p10 until 2026-10-05; that kind was cut by the noise grid
# decision. outlier_p05 is the decided kind that moves a subset.
DEFAULT_NOISE_CONDITION = 'outlier_p05'

# Bumped 2026-10-05 (chat 5): the configurations are chosen per outer fold by
# searches on that fold's training rows, for the dataset's two decided models.
# It is written into the result header, which ledger_signature hashes, so a
# rows file or checkpoint written by the whole-dataset design is set aside
# and never read back as done.
# Bumped again 2026-10-06: censoring runs one draw (noise_helpers.draws_for)
# and the header records knn_shapley as outside the counting-scheme axis.
RESULT_LAYOUT = 'fold_aligned_2026-10-06'

# The nominal share of molecules each condition damages. Written into the
# results for reading only; no method reads it (owner's decision 2026-09-28).
FLAG_SHARE = {
    'outlier_p01': 0.01, 'outlier_p05': 0.05, 'outlier_p10': 0.10,
    'outlier_only_p01': 0.01, 'outlier_only_p05': 0.05,
    'outlier_only_p10': 0.10,
    'grouped_wider': 0.20,
    **SHIFT_ONLY_CONDITIONS,
    # 2026-09-27. The six sparse_dprime_* conditions were added to
    # DETECTION_IS_DEFINED after this table was written, so a run naming no
    # condition wrote 10 files and then raised on the first of them. Each moves
    # the share of molecules carried in its own definition in noiseInject.
    **{c: float(v['share']) for c, v in RELATIVE_CONDITIONS.items()
       if c.startswith('sparse_dprime')},
}
# censoring's share is its level, the fraction clipped.

# The share of training molecules two_learner and the flagged-only shrinks
# flag, swept. n24's FLAG_RATES; the same at every condition and level, so no
# method reads how many molecules the injector moved (owner's decision
# 2026-09-28).
FLAG_SHARES = (0.05, 0.10, 0.20)

# 3 draws per non-zero level, the noise grid rule.
# DECIDED 2026-10-06 (Adelaide delegated): one draw under censoring. Its clip
# reads no seed, so draws 0, 1 and 2 gave bit-identical training labels
# (checked 2026-10-05 at 0.25 clipped, 400 labels). noise_helpers.draws_for
# cuts the list to its first draw for censoring and leaves every other kind
# at these three.
NOISE_DRAWS = (0, 1, 2)
ITLM_ALPHAS = (0.8, 0.9, 0.95)
ITLM_ROUNDS = 4
EXCHANGE_ROUNDS = 2
EXCHANGE_LEARNERS = ('lightgbm', 'svm')
RANDOM_SEEDS = (0, 1, 2)
OOF_SPLITS = 5
KNN_SHAPLEY_K = 5
DETECTORS = ('oof_residual', 'ngboost_rescaled', 'quantile_forest_rescaled',
             'conformal', 'knn_shapley', 'cleanlab')
GAP_MIN_FRAC = 0.02

# The 4 fitted configurations of one model (noise_helpers.CONFIGURATION_SPACES):
# base_1 and base_2 are the model's rank-1 and rank-2 single representation,
# hybrid and hybrid__general2 its two general hybrids, all from the per-fold
# search. Until 2026-10-05 the pool also gave hybrid__distance1,
# hybrid_similarity and their rank-2 and distance slots as fitted spaces.
BASE_SPACES = ('base_1', 'base_2')

# The spread of a normal whose 10th-to-90th percentile range is 1.
_Q10_Q90_WIDTH_IN_SD = 2.0 * float(norm.ppf(0.9))


# ═════════════════════════════════════════════════════════════════════════════
# SHARED BY n32-n41, added 2026-09-25: the D-versus-D' rows, the per-pass
# checkpoint, the cached fits and the similarity axis. Plan sections 1-3
# (~/.claude/plans/pasted-content-id-0428-noise-experiment-fizzy-ladybug.md).
#
# It lives in this file because n38 and n41 already import from it and the
# shared modules (noise_helpers.py, similarity_space.py, noise_cache.py) were
# not in the 2026-09-25 change's file list. Every n33-n41 imports it from here.
# TODO(similarity-sweep 2026-09-25): move this block into a shared module. Which
# module owns it is the owner's call, not made here.
# ═════════════════════════════════════════════════════════════════════════════

import hashlib as _hashlib
import json as _json

from experiments.noise import d_vs_dprime as _dvd
from experiments.noise import noise_cache as _nc
from experiments.noise import similarity_space as _ss
from experiments.noise.controls import (
    controls_enabled, random_twin, shuffled_twin,
)

# The D-versus-D' fits are made with at least these downstream models
# (d_vs_dprime.DEFAULT_MODELS, owner decision 2026-09-25: "RF and SVM at least").
ROW_MODELS = tuple(_dvd.DEFAULT_MODELS)

# Downstream models whose fit is seeded, so a cached prediction is the one a
# refit would give (noise_cache.cached_fit_predict's rule). dnn is left out: a
# torch fit on a multithreaded run need not repeat to the last bit.
CACHEABLE_FIT_MODELS = ('rf', 'svm', 'lightgbm', 'xgboost')


def downstream_model_list(kwargs, config):
    """The downstream models an n32-n41 run fits, with rf and svm added.

    n32 stopped calling this 2026-10-05 (chat 5): it fits the dataset's two
    decided models only (handoff R1), and this list appends rf and svm to the
    config's five. Left as it was for n33-n41, which still import it.

    An explicit `models=` (a smoke or a direct caller) is taken as it is. With
    none, the config's or DOWNSTREAM_MODELS list gets any of ROW_MODELS it
    lacks appended, so the D-versus-D' rows always carry rf and svm.
    DOWNSTREAM_MODELS (_shared.py:565) already holds both, so a default run
    fits exactly what it fitted before 2026-09-25.
    """
    explicit = kwargs.get('models')
    models = list(explicit or config.get('models') or DOWNSTREAM_MODELS)
    if not explicit:
        models += [m for m in ROW_MODELS if m not in models]
    return models


def method_setting(**kw):
    """The d_vs_dprime `method_setting` string: 'key=value;...', sorted by key,
    or 'default' when the method has no setting."""
    parts = [f'{k}={v}' for k, v in sorted(kw.items()) if v is not None]
    return ';'.join(parts) if parts else 'default'


def fit_predict_cached(dataset_name, model_name, factory, X_tr, y_tr, X_va,
                       metric_name, keep=None, sample_weight=None):
    """Held-out predictions of one fit, through noise_cache when the model is
    seeded (CACHEABLE_FIT_MODELS) and a dataset name is given.

    The same prediction n07._predict_for_metric gives: noise_cache's
    predict_mode_for is 'proba' for auc and 'predict' otherwise, the rule
    _predict_for_metric applies (n07:361-374). `keep` cuts the rows first, as
    every _fit_score here does.
    """
    X_f = X_tr
    y_f = np.asarray(y_tr)
    w_f = sample_weight
    if keep is not None:
        keep = np.asarray(keep, dtype=bool)
        X_f, y_f = X_f[keep], y_f[keep]
        w_f = None if w_f is None else np.asarray(w_f)[keep]
    if dataset_name is not None and model_name in CACHEABLE_FIT_MODELS:
        return np.asarray(_nc.cached_fit_predict(
            dataset_name, model_name, factory, X_f, y_f, X_va,
            sample_weight=w_f,
            predict_mode=_nc.predict_mode_for(metric_name)))
    m = factory()
    _nc.require_weights_accepted(m, w_f)  # gp: noise_cache.WEIGHTS_NOT_RUN
    if w_f is None:
        m.fit(X_f, y_f)
    else:
        m.fit(X_f, y_f, sample_weight=np.asarray(w_f))
    return np.asarray(_predict_for_metric(m, X_va, metric_name))


def scaled_base_cells(dataset_name, reps, train_idx, val_idx, base_by_model,
                      rank, models):
    """[(X_tr_s, X_va_s, [models])] for one base rank, through
    noise_cache.cached_scaled so a fold's matrix is scaled once across levels,
    draws, conditions and experiments (kirby.scaling.FeatureScaler: 0/1
    features left as they are, the rest standardised).

    2026-10-05 (chat 5, handoff 2.3 item 18, dirty-run bug 5): the matrix keeps
    its own dtype, float32 on load. Until then it was cast to float64 first,
    which the search path never did."""
    out = []
    for rn, ms in base_rank_groups(base_by_model, rank, models).items():
        X_tr_s, X_va_s = _nc.cached_scaled(dataset_name, reps[rn], train_idx,
                                           val_idx)
        out.append((X_tr_s, X_va_s, list(ms)))
    return out


def cached_hybrid(dataset_name, train_reps, val_reps, y_train, bp, groups_train,
                  model_name, model_factories, metric, train_idx, shareable):
    """noise_helpers.hybrid_for_fold through noise_cache.cached_hybrid_for_fold,
    with the arguments every n32-n41 fold loop passes: no model under a
    geometry recipe (`shareable`), else a fresh instance of the first model of
    the group. Returns (X_tr_h, X_va_h), or (None, None) on a failed build."""
    metric_name, metric_fn, lower_is_better = metric
    model = None if shareable else model_factories[model_name]()
    return _nc.cached_hybrid_for_fold(
        dataset_name, train_reps, val_reps, y_train, bp, groups_train, model,
        metric_fn, lower_is_better,
        model_name=None if shareable else model_name,
        metric_name=metric_name, train_idx=train_idx)


# ── The rows file and the per-pass checkpoint ──

def ledger_signature(data, config, header, splits, extra=None):
    """What a rows file and a checkpoint were computed from, as one hash.

    The input fingerprint run_all and sweep_conditions stamp their files with
    (_shared.input_fingerprint), the experiment's own settings as written into
    its result header before the fold loop, the fold indices, the similarity
    settings and ks swept, and whether controls run. Any change means the rows
    and the checkpoint on disk describe another run and are set aside.
    """
    from experiments._shared import input_fingerprint, convert_numpy
    blob = {
        'fingerprint': input_fingerprint(data, config),
        'header': convert_numpy(header),
        'splits': [[_nc.array_hash(np.asarray(tr)), _nc.array_hash(np.asarray(va))]
                   for tr, va in splits],
        'similarity_settings': os.environ.get('KIRBY_SIMILARITY_SETTINGS', ''),
        'swept_ks': list(_ss.SWEPT_KS),
        'controls': controls_enabled(),
        'extra': convert_numpy(extra),
    }
    text = _json.dumps(blob, sort_keys=True, default=lambda o: type(o).__name__)
    return _hashlib.blake2b(text.encode('utf-8'), digest_size=16).hexdigest()


class PassLedger:
    """The D-versus-D' rows of one (experiment, dataset, condition), and a
    checkpoint of the experiment's own accumulators after every finished pass.

    Files:
      <output>/<exp_id>/<dataset>[__<condition>].rows.jsonl
          d_vs_dprime rows, appended as each pass finishes (append_rows).
      <output>/<exp_id>/_checkpoints/<dataset>[__<condition>].json
          {'signature', 'passes_done', 'state'}: the experiment's accumulators
          as they stood after the last finished pass. In a subdirectory so no
          reader globbing <exp_id>/<dataset>*.json takes it for a result.

    Resume: a pass whose key is in passes_done is skipped and the accumulators
    are restored from `state`. A row already in the rows file is never written
    twice (done_keys). Rows are written BEFORE the checkpoint, so a kill between
    the two re-runs the pass and the rows it wrote are skipped.

    A checkpoint whose signature differs from this run's (ledger_signature), or
    a rows file with no checkpoint beside it, describes another run: both are
    renamed with a '.set_aside_<time>' suffix, never deleted, and the run
    starts clean. restart=True (run_dataset.py --force-rerun) does the same.

    output_dir None (a caller with nowhere to write) keeps the rows in
    `memory_rows` and checkpoints nothing.
    """

    def __init__(self, output_dir, exp_id, dataset_name, condition=None, *,
                 signature, restart=False):
        self.enabled = output_dir is not None
        self.signature = signature
        self.memory_rows = []
        self.passes_done = set()
        self.state = None
        self.done = set()
        self.set_aside = []
        self.rows_written = 0
        if not self.enabled:
            self.rows_path = self.ckpt_path = None
            return
        stem = dataset_name if not condition else f'{dataset_name}__{condition}'
        d = os.path.join(output_dir, exp_id)
        self.rows_path = os.path.join(d, f'{stem}.rows.jsonl')
        self.ckpt_path = os.path.join(d, '_checkpoints', f'{stem}.json')
        if restart:
            self._set_aside('force_rerun')
        else:
            self._load()
        if not os.path.exists(self.ckpt_path):
            self._write({'signature': signature, 'passes_done': [],
                         'state': None})
        self.done = _dvd.done_keys(self.rows_path)

    def _set_aside(self, why):
        stamp = time.strftime('%Y%m%d_%H%M%S')
        for p in (self.rows_path, self.ckpt_path):
            if p and os.path.exists(p):
                dest = f'{p}.set_aside_{stamp}_{why}'
                os.replace(p, dest)
                self.set_aside.append(dest)
                print(f"  {os.path.basename(p)}: set aside ({why}) -> {dest}")
        self.passes_done, self.state = set(), None

    def _load(self):
        ck = None
        if os.path.exists(self.ckpt_path):
            try:
                with open(self.ckpt_path) as fh:
                    ck = _json.load(fh)
            except (ValueError, OSError):
                ck = None
        if ck is None:
            if os.path.exists(self.rows_path) or os.path.exists(self.ckpt_path):
                self._set_aside('no_readable_checkpoint')
            return
        if ck.get('signature') != self.signature:
            self._set_aside('inputs_changed')
            return
        self.passes_done = set(ck.get('passes_done') or [])
        self.state = ck.get('state')

    def _write(self, blob):
        os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)
        tmp = f'{self.ckpt_path}.tmp{os.getpid()}'
        with open(tmp, 'w') as fh:
            _json.dump(blob, fh, allow_nan=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.ckpt_path)

    def pass_done(self, key):
        return str(key) in self.passes_done

    def restored(self):
        """The accumulators as the last finished pass left them, or None."""
        return self.state

    def add_rows(self, rows):
        """Append the rows not already on disk. Returns how many were new."""
        new = []
        for r in rows:
            k = _dvd.row_key(r)
            if k in self.done:
                continue
            self.done.add(k)
            new.append(r)
        if not new:
            return 0
        if self.enabled:
            _dvd.append_rows(self.rows_path, new)
        else:
            self.memory_rows.extend(new)
        self.rows_written += len(new)
        return len(new)

    def finish_pass(self, key, state):
        """Record `key` as finished, with the accumulators as they stand."""
        from experiments._shared import convert_numpy
        self.passes_done.add(str(key))
        if not self.enabled:
            return
        self._write({'signature': self.signature,
                     'passes_done': sorted(self.passes_done),
                     'state': convert_numpy(state)})

    def describe(self):
        return {'rows_file': self.rows_path, 'checkpoint_file': self.ckpt_path,
                'rows_written_this_run': self.rows_written,
                'passes_resumed_from_checkpoint': len(self.passes_done),
                'set_aside': self.set_aside}


def restart_flag(kwargs):
    """True when run_dataset.py --force-rerun reached this experiment. run()
    copies force_rerun to 'restart_passes' before sweep_conditions pops it."""
    return bool(kwargs.get('restart_passes') or kwargs.get('force_rerun'))


def arm_rows(*, dataset_name, exp_id, metric, condition, level, draw, fold,
             model_input, scores_by_arm, reference='no_cleaning', clean='clean',
             skip=(), similarity_setting='model_input|none|process',
             setting_of=None):
    """d_vs_dprime rows from scores an experiment already holds for one pass.

    scores_by_arm: {arm: {model: score}} for ONE (level, draw, fold, feature
    space). D is `reference` (the noisy labels as received), D' each other arm,
    score_clean the `clean` arm, all fitted with the same model on the same
    features and scored on the same held-out fold. No fit is made here.
    setting_of(arm) returns (method, method_setting, similarity_setting) for an
    arm, or None to use (arm, 'default', similarity_setting).
    """
    ref = scores_by_arm.get(reference) or {}
    cl = scores_by_arm.get(clean) or {}
    rows = []
    for arm, by_model in scores_by_arm.items():
        if arm in (reference, clean) or arm in skip:
            continue
        meth, mset, sim = ((setting_of(arm) if setting_of else None)
                           or (arm, 'default', similarity_setting))
        for m, s in by_model.items():
            if m not in ref:
                continue
            rows.append(_dvd.make_row(
                dataset=dataset_name, experiment=exp_id, method=meth,
                method_setting=mset, similarity_setting=sim,
                model_input=model_input, model=m, noise_condition=condition,
                level=level, draw=draw, fold=fold, metric=metric[0],
                lower_is_better=metric[2], score_D=ref[m], score_Dprime=s,
                score_clean=cl.get(m, float('nan'))))
    return rows


# ── Controls (KIRBY_NOISE_CONTROLS=1 only) ──

def arm_result(y_D, *, y=None, keep=None, weight=None):
    """The dict controls.random_twin / shuffled_twin read, for one arm that
    changed the training data: keep (a deletion), weight (a weighting) or y (a
    label move). Returns None for an arm that changes two of them at once --
    controls._shape reads only the first it finds, so a twin of it would
    silently drop the other change."""
    changes = [keep is not None, weight is not None,
               y is not None and not np.array_equal(np.asarray(y),
                                                      np.asarray(y_D))]
    if sum(changes) > 1:
        return None
    return {'keep': None if keep is None else np.asarray(keep, dtype=bool),
            'sample_weight': None if weight is None else np.asarray(weight,
                                                                    dtype=float),
            'label_column': (None if y is None or not changes[2]
                             else np.asarray(y, dtype=float))}


def control_twins(result, X_tr, y_D, seed):
    """{'random_twin': dict or None, 'shuffled_twin': dict or None} for one arm
    (controls.py), or {} when controls are off or the arm cannot be twinned."""
    if not controls_enabled() or result is None:
        return {}
    rt, _n = random_twin(result, X_tr, y_D, seed)
    st = shuffled_twin(result, X_tr, y_D, seed)
    return {'random_twin': rt, 'shuffled_twin': st}


def twin_scores(dataset_name, twins, factories, models, X_va, y_va, metric):
    """{twin: {model: score}} for the twins control_twins built."""
    metric_name, metric_fn, _lib = metric
    out = {}
    for tname, t in twins.items():
        if t is None:
            continue
        for m in models:
            try:
                pred = fit_predict_cached(
                    dataset_name, m, factories[m], t['X'], t['y'], X_va,
                    metric_name, sample_weight=t.get('sample_weight'))
                out.setdefault(tname, {})[m] = float(metric_fn(y_va, pred))
            except Exception as exc:
                print(f"      control {tname}/{m} failed: "
                      f"{type(exc).__name__}: {exc}")
    return out


# ── The similarity axis ──

def sweep_settings(dataset_name=None, n_molecules=None, has_smiles=True,
                   available_spaces=None):
    """The neighbour settings a run sweeps, for this dataset's group.

    2026-10-05 (chat 5): n07.sweep_settings, the decided list
    (KNN_SEARCH_SPACE_RECORD): the dataset group (more than 5000 molecules is
    fast_knn) decides the spaces and the distances, no importance weighting,
    and no SMILES removes tanimoto_ecfp4. KIRBY_SIMILARITY_SETTINGS still
    narrows it. Until 2026-10-05 this put MODEL_INPUT (the model's own matrix)
    first; that space is cut in both groups (axis 1, number 10).

    Pass the dataset name and its molecule count. Called with neither (n35,
    n36, n37, n40 and n41 still call it so), n07.sweep_settings returns
    full_knn's list and prints that it did.
    """
    from experiments.noise.n07_remediation import sweep_settings as _n07_ss
    return list(_n07_ss(dataset_name, n_molecules, has_smiles=has_smiles,
                        available_spaces=available_spaces))


def sweep_ks(today_k):
    """similarity_space.SWEPT_KS plus the k the method uses today.

    n32 reads n07.sweep_ks() since 2026-10-05. Every caller's own k today (5,
    10 or 20) is already in SWEPT_KS (5, 10, 20), so this returns SWEPT_KS for
    each of them."""
    return tuple(sorted(set(int(k) for k in _ss.SWEPT_KS) | {int(today_k)}))


def is_model_input(setting):
    return setting.space == _ss.MODEL_INPUT_SPACE


class FoldSimilarity:
    """The similarity spaces of one training fold, built on demand and cached.

    One per (fold, training labels): the importance weighting and the forest
    read the labels the caller holds (the noisy ones, owner decision
    2026-09-25), and the augmented and hybrid spaces are built from them.

    Spaces, per downstream model (similarity_space's names):
      base_1, base_2          that model's rank-1 / rank-2 single
                              representation, standardised on the fold
                              (noise_cache.cached_scaled)
      base_k_aug_uncapped     base_k plus augmentation features (c21's
                              uncapped method; the capped spaces were dropped
                              2026-10-03), built with
                              similarity_space.base_aug_blueprint through
                              noise_cache.cached_hybrid_for_fold; the donors
                              are data['augmentations'] named in
                              DATASET_REPS[ds]['augmentation'], minus the base
                              itself (c21:429)
      hybrid, hybrid__general2, hybrid_similarity
                              the model's stored recipe for that slot
                              (predictive general rank 1 and 2, similarity
                              general rank 1; similarity_space.HYBRID_SPACES),
                              built exactly as the model-input cell is, so the
                              build is one cache entry for both
      tanimoto_ecfp4          SMILES, if the data carries them
      forest_proximity, model_input
                              the model-input matrix the caller passes
    A space that cannot be built is recorded in `unavailable` with its reason
    and never faked.
    """

    def __init__(self, dataset_name, data, *, reps, train_idx, val_idx, y_train,
                 groups_train, base_by_model, recipe_for_space_by_model,
                 model_factories, metric, task_type, augmentations=None,
                 smiles=None, is_clf=False):
        self.ds = dataset_name
        self.data = data
        self.reps = reps
        self.train_idx = np.asarray(train_idx)
        self.val_idx = np.asarray(val_idx)
        self.y = np.asarray(y_train)
        self.g = groups_train
        self.base_by_model = base_by_model or {}
        self.recipes = recipe_for_space_by_model or {}
        self.factories = model_factories
        self.metric = metric
        self.task_type = task_type
        self.aug = augmentations
        self.smiles = None if smiles is None else list(smiles)
        self.is_clf = bool(is_clf)
        self._spaces = {}
        self.unavailable = {}

    # -- building the feature spaces --

    # 2026-10-05 (chat 5): every matrix below keeps its own dtype (float32 on
    # load, handoff 2.3 item 18) and is scaled with kirby.scaling.FeatureScaler
    # (0/1 features unscaled, item 17). Until then each was cast to float64
    # and every feature standardised.

    def _base(self, model, k):
        names = self.base_by_model.get(model) or []
        if len(names) < k:
            return None, None, f'{model} has no rank-{k} single representation'
        X_tr, X_va = _nc.cached_scaled(self.ds, self.reps[names[k - 1]],
                                       self.train_idx, self.val_idx)
        return X_tr, X_va, None

    def _aug(self, model, k, strategy):
        from experiments.complexity.complexity_helpers import DATASET_REPS
        names = self.base_by_model.get(model) or []
        if len(names) < k:
            return None, None, f'{model} has no rank-{k} single representation'
        base = names[k - 1]
        wanted = (DATASET_REPS.get(self.ds) or {}).get('augmentation') or []
        pool = {a: v for a, v in (self.aug or {}).items()
                if a in wanted and a != base}
        if not pool:
            return None, None, (f'no augmentation representation is loaded for '
                                f'{self.ds} besides {base}')
        metric_name, metric_fn, lib = self.metric
        bp = _ss.base_aug_blueprint(
            base, np.shape(self.reps[base])[1], metric_name=metric_name,
            task_type=self.task_type, lower_is_better=lib, strategy=strategy)
        tr = self.train_idx
        va = self.val_idx
        X_tr_h, X_va_h = _nc.cached_hybrid_for_fold(
            self.ds, {base: np.asarray(self.reps[base][tr])},
            {base: np.asarray(self.reps[base][va])},
            self.y, bp, self.g, self.factories[model](), metric_fn, lib,
            model_name=model, metric_name=metric_name, train_idx=tr,
            augmentations_train={a: np.asarray(v[tr])
                                 for a, v in pool.items()},
            augmentations_val={a: np.asarray(v[va])
                               for a, v in pool.items()})
        if X_tr_h is None:
            return None, None, 'the augmented build failed'
        sc = FeatureScaler()
        return sc.fit_transform(X_tr_h), sc.transform(X_va_h), None

    def _hyb(self, model, space):
        """The recipe is a Blueprint, or since 2026-10-05 a (Blueprint,
        augmentation condition) pair as noise_helpers.fold_configurations
        hands it out. A pair whose condition is not 'none' is rebuilt WITH the
        augmentation representations, as it was searched; a bare Blueprint is
        rebuilt without them, as before."""
        from experiments.noise.noise_helpers import recipe_is_geometry
        rec = (self.recipes.get(model) or {}).get(space)
        bp, aug_cond = (rec if isinstance(rec, tuple) else (rec, 'none'))
        if bp is None:
            return None, None, f'{model} holds no {space} recipe'
        shareable = recipe_is_geometry(bp)
        train_reps = {k: np.asarray(v[self.train_idx])
                      for k, v in self.reps.items()}
        val_reps = {k: np.asarray(v[self.val_idx])
                    for k, v in self.reps.items()}
        if (aug_cond or 'none') != 'none' and self.aug:
            metric_name, metric_fn, lib = self.metric
            X_tr_h, X_va_h = _nc.cached_hybrid_for_fold(
                self.ds, train_reps, val_reps, self.y, bp, self.g,
                None if shareable else self.factories[model](), metric_fn,
                lib, model_name=None if shareable else model,
                metric_name=metric_name, train_idx=self.train_idx,
                augmentations_train={a: np.asarray(v[self.train_idx])
                                     for a, v in self.aug.items()},
                augmentations_val={a: np.asarray(v[self.val_idx])
                                   for a, v in self.aug.items()})
        else:
            X_tr_h, X_va_h = cached_hybrid(
                self.ds, train_reps, val_reps, self.y, bp, self.g, model,
                self.factories, self.metric, self.train_idx, shareable)
        if X_tr_h is None:
            return None, None, 'the hybrid build failed'
        sc = FeatureScaler()
        return sc.fit_transform(X_tr_h), sc.transform(X_va_h), None

    def feature_space(self, model, space):
        """(Z_tr, Z_va) standardised, or (None, None) with the reason in
        self.unavailable[(model, space)]."""
        key = (model, space)
        if key not in self._spaces:
            try:
                if space in ('base_1', 'base_2'):
                    got = self._base(model, int(space[-1]))
                elif space in _ss.AUG_SPACES:
                    _, k, _, st = space.split('_')
                    got = self._aug(model, int(k), st)
                elif space in _ss.HYBRID_SPACES:
                    got = self._hyb(model, space)
                else:
                    got = (None, None, f'{space} is not a feature space')
            except Exception as exc:
                got = (None, None, f'{type(exc).__name__}: {exc}')
            self._spaces[key] = got
            if got[0] is None:
                self.unavailable[f'{model}|{space}'] = got[2]
        Z_tr, Z_va, _ = self._spaces[key]
        return Z_tr, Z_va

    def reason(self, model, space):
        return self.unavailable.get(f'{model}|{space}')

    def _weighted(self, setting, Z_tr, Z_va=None):
        """The setting's importance weighting folded into Z, exactly as
        similarity_space.neighbour_lists does it (similarity_space.py:560-574),
        with the importance itself cached (noise_cache kind
        'importance_weights', keyed on the matrix and label content).

        Since 2026-10-05 every setting's weighting is 'none' (axis 3 cut), so
        Z comes back as it went in, in its own dtype."""
        if setting.weighting == 'none':
            return Z_tr, Z_va
        Z_tr = np.asarray(Z_tr)
        Z_va = None if Z_va is None else np.asarray(Z_va)
        from kirby.similarity.preprocessing import fold_weights
        w = _nc.cached(self.ds, 'importance_weights',
                       {'Z': _nc.array_hash(Z_tr), 'y': _nc.array_hash(self.y),
                        'method': setting.weighting},
                       lambda: _ss._importance_weights(Z_tr, self.y,
                                                       setting.weighting))
        Z_tr, _ = fold_weights(Z_tr, w, setting.metric, convention='mean1')
        if Z_va is not None:
            Z_va, _ = fold_weights(Z_va, w, setting.metric, convention='mean1')
        return Z_tr, Z_va

    # -- what consumers read --

    def neighbours(self, setting, model, X_model_tr, X_model_va=None,
                   exclude_groups=None):
        """noise_cache.cached_neighbours for one setting and one model, or a
        dict with available False and the reason."""
        sp = setting.space
        tr, va = {'model_input': X_model_tr}, (
            None if X_model_va is None else {'model_input': X_model_va})
        if sp in _ss.FEATURE_SPACES:
            Z_tr, Z_va = self.feature_space(model, sp)
            if Z_tr is None:
                return _ss._unavailable(setting, self.reason(model, sp))
            tr[sp] = Z_tr
            if va is not None:
                va[sp] = Z_va
        if sp == _ss.TANIMOTO_SPACE and self.smiles is None:
            return _ss._unavailable(setting, f'{self.ds} carries no SMILES')
        sm_tr = (None if self.smiles is None
                 else [self.smiles[i] for i in self.train_idx])
        sm_va = (None if self.smiles is None or va is None
                 else [self.smiles[i] for i in self.val_idx])
        return _nc.cached_neighbours(
            self.ds, setting, spaces_train=tr, spaces_val=va, y_train=self.y,
            groups_train=self.g, exclude_groups=exclude_groups,
            smiles_train=sm_tr, smiles_val=sm_va, is_clf=self.is_clf)

    def space_matrix(self, setting, model, X_model_tr, X_model_va=None):
        """(Z_tr, Z_va, None) the setting's weighted feature matrix, for a
        consumer that needs coordinates and not only distances (n41's joined
        matrix). (None, None, reason) for Tanimoto and forest proximity, which
        are distances with no coordinates, and for MODEL_INPUT, whose caller
        keeps its own matrix."""
        sp = setting.space
        if sp not in _ss.FEATURE_SPACES:
            return None, None, f'{sp} is a distance with no feature matrix'
        Z_tr, Z_va = self.feature_space(model, sp)
        if Z_tr is None:
            return None, None, self.reason(model, sp)
        Z_tr, Z_va = self._weighted(setting, Z_tr, Z_va if X_model_va is not None
                                    else None)
        return Z_tr, Z_va, None

    def distance_blocks(self, setting, model, X_model_tr):
        """(dist(a, b), n, None): a function returning the distances of
        training rows a..b to every training row under the setting, float64;
        or (None, 0, reason). Self-distances are left as computed (0), as a
        square matrix has them; the consumer masks the diagonal. MODEL_INPUT is
        refused: its caller keeps the distance it computes today.

        The same arithmetic as similarity_space.neighbour_lists: scipy jaccard
        on the ECFP4 bits for Tanimoto (similarity_space.py:505-507), the
        forest share for forest proximity (_forest_distance), cdist under the
        setting's metric on the weighted standardised features.

        2026-10-05 (chat 5): the feature-space distance is
        similarity_space.feature_distance, which computes all three decided
        distances (manhattan, cosine, minkowski p=0.5) and sets cosine's
        undefined pairs to +inf. Until then the scipy name was read from
        similarity_space._CDIST, which holds no minkowski entry, so every
        minkowski_p0.5 setting raised KeyError."""
        from scipy.spatial.distance import cdist
        sp = setting.space
        n = len(self.y)
        if sp == _ss.MODEL_INPUT_SPACE:
            return None, 0, 'model_input: the caller computes it'
        if sp == _ss.TANIMOTO_SPACE:
            if self.smiles is None:
                return None, 0, f'{self.ds} carries no SMILES'
            bits, bad = _ss._ecfp4_bits(
                [self.smiles[i] for i in self.train_idx])

            def _tanimoto(a, b):
                # an unparseable SMILES is skipped, as in similarity_space
                D = cdist(bits[a:b], bits, metric='jaccard')
                D[:, bad] = np.inf
                D[bad[a:b], :] = np.inf
                return D
            return _tanimoto, n, None
        if sp == _ss.FOREST_SPACE:
            L_tr, _, O_tr = _nc.cached_forest_leaf_ids(
                self.ds, X_model_tr, self.y, self.is_clf)
            return (lambda a, b: np.asarray(
                _ss._forest_distance(L_tr[a:b], L_tr, O_tr[a:b], O_tr),
                dtype=np.float64), n, None)
        Z_tr, _ = self.feature_space(model, sp)
        if Z_tr is None:
            return None, 0, self.reason(model, sp)
        Z, _ = self._weighted(setting, Z_tr)
        metric = setting.metric
        return (lambda a, b: _ss.feature_distance(Z[a:b], Z, metric), n, None)

    def full_distance(self, setting, model, X_model_tr, dtype=np.float64):
        """The square training-by-training distance under the setting, or
        (None, reason)."""
        fn, n, why = self.distance_blocks(setting, model, X_model_tr)
        if fn is None:
            return None, why
        D = np.empty((n, n), dtype=dtype)
        step = _ss._chunk_rows(n)
        for a in range(0, n, step):
            b = min(n, a + step)
            D[a:b] = fn(a, b)
        return D, None


def fold_similarity(dataset_name, data, *, reps, train_idx, val_idx, y_train,
                    groups_train, base_by_model, recipe_for_space_by_model,
                    model_factories, metric, task_type, is_clf=False):
    """FoldSimilarity with the data's own augmentations and SMILES."""
    return FoldSimilarity(
        dataset_name, data, reps=reps, train_idx=train_idx, val_idx=val_idx,
        y_train=y_train, groups_train=groups_train, base_by_model=base_by_model,
        recipe_for_space_by_model=recipe_for_space_by_model,
        model_factories=model_factories, metric=metric, task_type=task_type,
        augmentations=data.get('augmentations'), smiles=data.get('smiles'),
        is_clf=is_clf)


def setting_models(setting, cell_models):
    """Which models a setting is computed for in one cell. A per-model space
    (base_*, aug, hybrid) differs by model; model_input, Tanimoto and forest
    proximity read only the cell's own matrix, so they are computed once, for
    the cell's first model."""
    if setting.space in _ss.FEATURE_SPACES:
        return list(cell_models)
    return list(cell_models[:1])


def similarity_key(setting, k, model=None, today_k=None, per_model=False):
    """The name a similarity result is stored under. MODEL_INPUT at today's k
    keeps today's bare name (''), so today's keys are unchanged."""
    if is_model_input(setting) and today_k is not None and int(k) == int(today_k):
        return ''
    tail = f'@{setting.name}|k{int(k)}'
    if per_model and model is not None and setting.space in _ss.FEATURE_SPACES:
        tail += f'|{model}'
    return tail


# ── Detectors ────────────────────────────────────────────────────────────────

def knn_shapley_values(X, y, k=KNN_SHAPLEY_K, groups=None, block=512,
                       dist_rows=None, ks=None):
    """Each training molecule's Shapley value to a k-nearest-neighbour predictor.

    SIMILARITY AXIS, added 2026-09-25. `dist_rows(a, b)`, when given, returns
    the distances of training rows a..b to every training row in a similarity
    setting (FoldSimilarity.distance_blocks), and the neighbours are ordered by
    it instead of by squared Euclidean distance in X. Only the ORDER is read,
    so any monotone distance gives the same values. None is today's path,
    unchanged. `ks`, when given, computes every k in it from ONE ordering per
    test molecule and returns {k: values}; the ordering does not depend on k.

    THE UTILITY. For one test molecule t with label y_t, a subset S of the other
    training molecules is worth
        U(S) = (1/k) * sum over the min(k, |S|) members of S nearest to t
                       of  -(y_i - y_t)^2
    i.e. minus the mean squared label gap to the neighbours, one term per
    neighbour. That is NOT minus the squared error of the neighbours' MEAN: the
    mean's squared error is not a sum over neighbours, and Jia et al.'s exact
    recursion needs a sum. The per-neighbour sum is the regression analogue of
    their classification utility (1/k) * sum 1[y_i = y_t], with the indicator
    replaced by -(y_i - y_t)^2.

    THE RECURSION (Jia et al. 2019, "Efficient task-specific data valuation for
    nearest neighbor algorithms", Theorem 1), with u_i = -(y_(i) - y_t)^2 and the
    others sorted nearest first, i = 1..N:
        s_N = u_N / N
        s_i = s_(i+1) + (u_i - u_(i+1)) / k * min(k, i) / i
    It is exact for this utility; _smoke_n32 checks it against a brute-force
    Shapley value over every subset on 7 molecules.

    EVERY training molecule is the test point once, with itself left out of
    the others. The values are summed over the test points and divided by
    their number.

    groups: when given, every molecule of the test point's own scaffold group
    is also left out. n32 passes None since 2026-10-05: the kNN record keeps
    one neighbour list, the full one, with scaffold-mates in it
    (KNN_SEARCH_SPACE_RECORD axis 6, Adelaide 2026-10-05). The argument is
    kept for a caller that names it.

    Returns one value per molecule; a LOW value is a molecule that hurts.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    g = None if groups is None else np.asarray(groups)
    k_list = [int(k)] if ks is None else [int(x) for x in ks]
    if dist_rows is None:
        # The feature matrix in its own dtype (float32 rule, 2026-10-05).
        X = np.asarray(X)
        sq = np.einsum('ij,ij->i', X, X)
    values = {kk: np.zeros(n) for kk in k_list}
    counted = 0
    for start in range(0, n, block):
        stop = min(n, start + block)
        if dist_rows is None:
            d2 = sq[start:stop, None] + sq[None, :] - 2.0 * X[start:stop] @ X.T
        else:
            d2 = np.asarray(dist_rows(start, stop), dtype=float)
        for r, t in enumerate(range(start, stop)):
            dist = d2[r].copy()
            dist[t] = np.inf
            if g is not None:
                dist[g == g[t]] = np.inf
            others = np.flatnonzero(np.isfinite(dist))
            N = others.size
            if N == 0:
                continue
            order = others[np.argsort(dist[others], kind='stable')]
            u = -(y[order] - y[t]) ** 2
            for kk in k_list:
                s = np.empty(N)
                s[-1] = u[-1] / N
                if N > 1:
                    i = np.arange(1, N)               # 1-indexed i = 1..N-1
                    terms = (u[:-1] - u[1:]) / kk * np.minimum(kk, i) / i
                    s[:-1] = s[-1] + np.cumsum(terms[::-1])[::-1]
                values[kk][order] += s
            counted += 1
    out = {kk: v / max(counted, 1) for kk, v in values.items()}
    return out[k_list[0]] if ks is None else out


def _rescaled(y, centre, spread):
    spread = np.asarray(spread, dtype=float)
    return np.abs(np.asarray(y, dtype=float) - np.asarray(centre, dtype=float)) / \
        np.clip(spread, 1e-9, None)


def detector_scores(name, X, y, groups, p_oof, oof_splits):
    """One suspicion score per training molecule; higher = more suspect."""
    from kirby.utils.noise_estimation import ESTIMATOR_REGISTRY
    y = np.asarray(y, dtype=float)
    if name == 'oof_residual':
        return np.abs(y - p_oof)
    if name == 'ngboost_rescaled':
        est = ESTIMATOR_REGISTRY['ngboost'](n_folds=oof_splits, groups=groups)
        est.estimate(X, y)
        return _rescaled(y, est.predictions_, est.sigma_)
    if name == 'quantile_forest_rescaled':
        est = ESTIMATOR_REGISTRY['quantile_forest'](n_folds=oof_splits,
                                                    groups=groups)
        est.estimate(X, y)
        spread = np.asarray(est.interval_widths_) / _Q10_Q90_WIDTH_IN_SD
        return _rescaled(y, est.predictions_, spread)
    if name == 'conformal':
        est = ESTIMATOR_REGISTRY['conformal'](n_folds=oof_splits, groups=groups)
        est.estimate(X, y)
        return _rescaled(y, est.predictions_, est.uncertainties_)
    if name == 'knn_shapley':
        # Scaffold-mates stay in the neighbour list (axis 6, 2026-10-05), so
        # the groups are not passed. They still decide the out-of-fold splits
        # of every other detector.
        return -knn_shapley_values(X, y, groups=None)
    if name == 'cleanlab':
        from experiments.noise.noise_helpers import cleanlab_suspicion
        return cleanlab_suspicion(X, y, cv_n_folds=oof_splits)
    raise ValueError(f'n32: unknown detector {name!r}')


# DECIDED 2026-10-06 (Adelaide delegated): knn_shapley keeps equal neighbour
# shares and is recorded as outside the counting-scheme axis
# (KNN_SEARCH_SPACE_RECORD axis 5, 'rank' and 'inverse_square'). Its utility is
# the mean over the k nearest of -(y_i - y_test)^2, each neighbour counting
# 1/k, and Jia et al.'s exact recursion needs that equal share; a weighted
# utility has no recursion in this code.
KNN_SHAPLEY_SCHEME_NOTE = (
    'knn_shapley counts each of the k neighbours equally (1/k). It is outside '
    'the counting-scheme axis: the schemes rank and inverse_square are not '
    'applied to it, because the exact Shapley recursion needs equal shares '
    '(decided 2026-10-06)')


def _knn_shapley_sweep(fsim, settings, ks, X_cell_tr, y0, cmask, condition,
                       model, store_key, lvl_key, fold_i, det_acc, errors,
                       unavailable=None):
    """knn_shapley under each neighbour setting and every k, graded against
    the mask. Rewritten 2026-10-05 (chat 5).

    One model's neighbour spaces (fsim, its base_1, base_2, augmented,
    hybrid, hybrid__general2, hybrid_similarity and hybrid__distance1 spaces),
    or for forest_proximity the configuration matrix X_cell_tr the forest is
    grown on. Stored under det_acc[level][store_key] as
    'knn_shapley@<setting>|k<k>'. No model is fitted per setting, so every
    setting is scored (no screen). One ordering per setting serves every k
    (knn_shapley_values(ks=...)). Scaffold-mates are not left out (axis 6).

    Until 2026-10-05 the sweep also ran on the configuration's own matrix
    (MODEL_INPUT, the bare key 'knn_shapley'); that space is cut (axis 1,
    number 10).
    """
    for setting in settings:
        try:
            fn, _n, why = fsim.distance_blocks(setting, model, X_cell_tr)
            if fn is None:
                if unavailable is not None:
                    unavailable[f'{setting.name}|{model}'] = why
                continue
            by_k = knn_shapley_values(None, y0, groups=None, ks=ks,
                                      dist_rows=fn)
        except Exception as exc:
            errors[f'{lvl_key}|{store_key}|knn_shapley@{setting.name}|{model}|'
                   f'f{fold_i}'] = f'{type(exc).__name__}: {exc}'
            continue
        for k, vals in by_k.items():
            name = f'knn_shapley@{setting.name}|k{int(k)}'
            d = detection_scores(cmask, -vals, condition)
            if d is not None:
                d = dict(d, similarity_setting=setting.name, k=int(k),
                         similarity_model=model)
                (det_acc.setdefault(lvl_key, {}).setdefault(store_key, {})
                 .setdefault(name, []).append(d))


# ── Methods ─────────────────────────────────────────────────────────────────

def subset_loss(make, X, y, groups, keep, oof_splits):
    """Squared error of every molecule under a model trained on the kept ones.

    A kept molecule's error is OUT OF FOLD within the kept set (group-aware); a
    dropped molecule's is under one model fitted on every kept molecule. This is
    the report's H2 snippet with the model left as an argument.
    """
    y = np.asarray(y, dtype=float)
    keep = np.asarray(keep, dtype=bool)
    ki = np.flatnonzero(keep)
    g_k = np.asarray(groups)[ki]
    loss = np.empty(len(y))
    k = _safe_n_splits(g_k, oof_splits)
    oof = np.full(ki.size, np.nan)
    for tr, te in group_kfold_splitter(g_k, k, X[ki], y[ki]):
        m = make()
        m.fit(X[ki][tr], y[ki][tr])
        oof[te] = m.predict(X[ki][te])
    loss[ki] = (y[ki] - oof) ** 2
    di = np.flatnonzero(~keep)
    if di.size:
        m = make()
        m.fit(X[ki], y[ki])
        loss[di] = (y[di] - m.predict(X[di])) ** 2
    return loss


def lowest_share(loss, share_kept):
    """Boolean mask of the lowest-loss share of rows, ties broken by position."""
    loss = np.asarray(loss, dtype=float)
    n_keep = int(round(share_kept * loss.size))
    n_keep = min(max(n_keep, 2), loss.size)
    keep = np.zeros(loss.size, dtype=bool)
    keep[np.argsort(loss, kind='stable')[:n_keep]] = True
    return keep


def jaccard(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    union = int(np.sum(a | b))
    return float(np.sum(a & b)) / union if union else 1.0


def itlm(X, y, groups, alpha, rounds=ITLM_ROUNDS, oof_splits=OOF_SPLITS,
         make=None, first_loss=None):
    """ITLM: returns (final keep mask, last round's loss, per-round records).

    first_loss, added 2026-09-25: round 1 keeps every molecule, so its loss is
    (y - p)^2 with p the plain group-aware out-of-fold LightGBM prediction
    (oof_predict: the same make_oof_lgbm, the same group_kfold_splitter at the
    same fold count). A caller holding that p passes (y - p)^2 and round 1 is
    not refitted -- once per alpha before this. Only read when `make` is the
    default and oof_splits is the count p was computed with.
    """
    use_first = first_loss is not None and make is None
    make = make or make_oof_lgbm
    keep = np.ones(len(y), dtype=bool)
    records = []
    loss = None
    for t in range(int(rounds)):
        if t == 0 and use_first:
            loss = np.asarray(first_loss, dtype=float)
        else:
            loss = subset_loss(make, X, y, groups, keep, oof_splits)
        new_keep = lowest_share(loss, alpha)
        records.append({'round': t + 1,
                        'kept': int(new_keep.sum()),
                        'overlap_with_previous_round_kept_set': (
                            jaccard(keep, new_keep) if t > 0 else None)})
        keep = new_keep
    return keep, loss, records


def two_learner_exchange(X, y, groups, make_a, make_b, share_kept,
                         rounds=EXCHANGE_ROUNDS, oof_splits=OOF_SPLITS):
    """Each learner picks the lowest-loss share for the OTHER to train on.

    Returns (intersection keep mask, record). The record holds the counts of
    molecules each learner flags and how many both flag.
    """
    all_rows = np.ones(len(y), dtype=bool)
    loss_a = subset_loss(make_a, X, y, groups, all_rows, oof_splits)
    loss_b = subset_loss(make_b, X, y, groups, all_rows, oof_splits)
    for _ in range(int(rounds)):
        keep_for_a = lowest_share(loss_b, share_kept)
        keep_for_b = lowest_share(loss_a, share_kept)
        loss_a = subset_loss(make_a, X, y, groups, keep_for_a, oof_splits)
        loss_b = subset_loss(make_b, X, y, groups, keep_for_b, oof_splits)
    keep_a = lowest_share(loss_a, share_kept)
    keep_b = lowest_share(loss_b, share_kept)
    both = keep_a & keep_b
    flag_a, flag_b = ~keep_a, ~keep_b
    return both, {
        'flagged_by_first_learner': int(flag_a.sum()),
        'flagged_by_second_learner': int(flag_b.sum()),
        'flagged_by_both': int(np.sum(flag_a & flag_b)),
        'flagged_by_either': int(np.sum(flag_a | flag_b)),
        'kept_by_both_and_trained_on': int(both.sum()),
        'training_molecules': int(len(y)),
    }, np.maximum(loss_a, loss_b)


def shrink_flagged(y, p, share):
    """The two flagged-only shrink columns and what they estimated."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    r = y - p
    flag = ~lowest_share(np.abs(r), 1.0 - share)
    if flag.sum() < 2 or (~flag).sum() < 2:
        return None, None, {'why_none': 'the flagged or unflagged set holds '
                                        'fewer than 2 molecules'}
    # MEAN SQUARES, not variances. Under a one-direction shift every flagged
    # residual sits near +delta, so their VARIANCE is small and a variance-based
    # sn^2 came out 0 on the smoke's toy data -- the shrink moved nothing. p is
    # an out-of-fold prediction of the labels, so a residual's expected value is
    # 0 and its mean square is its spread around the truth.
    sm2 = float(np.mean(r[~flag] ** 2))
    sn2 = max(float(np.mean(r[flag] ** 2)) - sm2, 0.0)
    w = sn2 / (sn2 + sm2) if (sn2 + sm2) > 0 else 0.0
    one = y.copy()
    one[flag] = (1.0 - w) * y[flag] + w * p[flag]
    excess = np.maximum(r[flag] ** 2 - sm2, 0.0)
    w_i = excess / (excess + sm2) if sm2 > 0 else np.zeros_like(excess)
    per = y.copy()
    per[flag] = (1.0 - w_i) * y[flag] + w_i * p[flag]
    return one, per, {
        'molecules_flagged': int(flag.sum()),
        'mean_square_residual_unflagged_label_units2': sm2,
        'extra_mean_square_on_flagged_label_units2': sn2,
        'one_weight': w,
        'per_row_weight_mean': float(np.mean(w_i)),
    }


def replicate_seed(*parts):
    """zlib.crc32, not hash(): stable across processes (n24:308-322)."""
    key = '|'.join(str(p) for p in parts)
    return int(zlib.crc32(key.encode('utf-8')) & 0x7FFFFFFF)


def random_keep(n, n_remove, seed):
    keep = np.ones(n, dtype=bool)
    if n_remove > 0:
        rng = np.random.default_rng(int(seed))
        keep[rng.choice(n, int(n_remove), replace=False)] = False
    return keep


def build_arm_names(alphas, seeds, shares=FLAG_SHARES):
    out = ['clean', 'no_cleaning']
    for a in alphas:
        out.append(f'itlm_a{a:g}')
        out += [f'itlm_a{a:g}_random/seed{s}' for s in seeds]
    for sh in shares:
        out.append(f'two_learner_share{sh:g}')
        out += [f'two_learner_share{sh:g}_random/seed{s}' for s in seeds]
    for sh in shares:
        out += [f'shrink_flagged_one_weight_share{sh:g}',
                f'shrink_flagged_per_row_share{sh:g}']
    return out


def _fit_score(factory, X_tr, y_tr, X_va, y_va, metric_fn, metric_name,
               keep=None, dataset_name=None, model_name=None):
    """One fit; (score, tail dict, None) or (None, None, why).

    dataset_name and model_name, added 2026-09-25, route a seeded model's fit
    through noise_cache (fit_predict_cached); omitted, the fit is made here as
    before. The prediction is the same either way."""
    try:
        pred = fit_predict_cached(dataset_name, model_name, factory, X_tr,
                                  y_tr, X_va, metric_name, keep=keep)
        return (float(metric_fn(y_va, pred)),
                tail_scores(y_va, np.asarray(pred, dtype=float)), None)
    except Exception as exc:
        return None, None, f'{type(exc).__name__}: {exc}'


def _levels_and_draws(data, dataset_name, condition, levels=None, draws=None):
    """(level, draw) pairs for one fold, level 0 LAST.

    Levels: noise_helpers.decided_levels (the noise grid decision, Adelaide
    2026-10-05) for this dataset's group, each kind in its own unit
    (level_unit). A `levels` kwarg (a smoke, or --levels) overrides. 3 draws
    per non-zero level, one under censoring (noise_helpers.draws_for,
    2026-10-06); level 0 is the labels as delivered and runs once.

    Level 0 runs last in each fold since 2026-10-05 (chat 5): in the dirty run
    (28 Sep to 2 Oct) n32 finished level 0 only inside its time, so no noise
    was ever injected. Every pass fits the `clean` arm itself, so no pass
    needs level 0 first. Until 2026-10-05 the levels came from
    dataset_policy.noise_levels, shift_only from SHIFT_ONLY_DELTAS_SD and the
    sparse_dprime kinds from relative_levels (1, 2, 4).
    """
    if levels is None:
        levels = decided_levels(data, dataset_name, condition)
    draws = tuple(draws_for(condition,
                            NOISE_DRAWS if draws is None else draws))
    out = []
    for lvl in sorted({float(l) for l in levels if float(l) > 0}):
        for d in draws:
            out.append((lvl, int(d)))
    if any(float(l) == 0.0 for l in levels):
        out.append((0.0, int(draws[0])))
    return out


def _per_fold_means(by_pass):
    """{pass key 'l<level>d<draw>f<fold>': score} -> the mean over draws in
    each fold, in fold order, for noise_helpers.fold_spread."""
    by_fold = {}
    for k, v in (by_pass or {}).items():
        if v is None or not np.isfinite(float(v)):
            continue
        by_fold.setdefault(int(str(k).rsplit('f', 1)[-1]), []).append(float(v))
    return [float(np.mean(v)) for _, v in sorted(by_fold.items())]


EXP_ID = 'n32_iterative_and_dual_selection'


def run(data, config, output_dir, dataset_name, **kwargs):
    """N32 across every decided condition where detection is defined, one
    file each.

    Since 2026-10-05 (chat 5) the default is the decided noise grid's kinds
    that move a subset of the training labels (noise_helpers
    .decided_conditions, filtered by detection_is_defined): outlier_p05,
    censoring, shift_only_p10, sparse_dprime_up_p10. A named condition that
    the grid cut, or one where every label moves, gets a skip carrying the
    reason. Until then the default was every condition in
    DETECTION_IS_DEFINED, sixteen of them cut since.
    """
    if data['task_type'] != 'regression':
        return {'skipped': 'not_regression', 'task_type': data['task_type']}
    decided = list(decided_conditions(data))
    default = [c for c in decided if detection_is_defined(c)]
    named = kwargs.get('noise_condition')
    if named and not detection_is_defined(named):
        print(f"  SKIPPED: {named} moves every training label, so there is no "
              f"damaged subset to find.")
        return {'skipped': 'detection_undefined', 'noise_condition': named}
    if named and named not in decided:
        print(f"  SKIPPED: {named} is not one of the decided kinds of damage "
              f"(KNN_SEARCH_SPACE_RECORD, noise grid, 2026-10-05): "
              f"{', '.join(default)}.")
        return {'skipped': 'condition_cut_by_noise_grid',
                'noise_condition': named, 'decided_conditions': default}
    if kwargs.get('noise_conditions'):
        asked = [str(c) for c in kwargs['noise_conditions']]
        cut = [c for c in asked if c not in default]
        if cut:
            print(f"  dropped {cut}: cut by the noise grid, or every label "
                  f"moves. Decided here: {default}")
        kwargs['noise_conditions'] = [c for c in asked if c in default]
    # sweep_conditions pops force_rerun; the per-pass checkpoint and rows file
    # read it as restart_passes (PassLedger), 2026-09-25.
    kwargs['restart_passes'] = bool(kwargs.get('force_rerun'))
    # 2026-10-05 (chat 5): the condition files are fingerprinted with the
    # decided levels, the draws, the outer fold count, the folds run and
    # RESULT_LAYOUT, as n07 does. sweep_conditions skips a condition whose
    # file is on disk and whose input_fingerprint matches, and until now that
    # fingerprint held nothing n32 changed (n32's config carries no noise
    # grid). So a file written by the whole-dataset five-model design was
    # read back as 'already on disk'. Files without this field now mismatch
    # and are recomputed.
    from experiments.noise.noise_helpers import noise_n_folds
    sweep_conds = kwargs.get('noise_conditions') or default
    fp_config = dict(config)
    fp_config['n_splits'] = noise_n_folds(config)
    fp_config['noise_levels'] = {
        c: [f'{l:g}/d{d}' for l, d in _levels_and_draws(
            data, dataset_name, c, levels=kwargs.get('levels'),
            draws=kwargs.get('draws'))]
        for c in sweep_conds}
    fp_config['noise_seeds'] = {
        c: list(draws_for(c, NOISE_DRAWS if kwargs.get('draws') is None
                          else kwargs['draws']))
        for c in sweep_conds}
    fp_config['noise_strategies'] = [
        RESULT_LAYOUT,
        f"only_folds={os.environ.get('KIRBY_NOISE_ONLY_FOLDS') or 'all'}"]
    return sweep_conditions(
        _run_one_condition, EXP_ID, data, fp_config, output_dir, dataset_name,
        default_conditions=default, kwargs=kwargs,
        groups_needed=GROUPED_STRATEGIES)


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    """N32: ITLM, two-learner exchange, flagged-only shrink, six detectors.

    Per outer fold (noise_helpers.noise_outer_folds):
      1. noise_helpers.fold_configurations runs this fold's searches on its
         training rows (clean labels) and gives each decided model 2 base
         representations, 2 general hybrids and the neighbour-space recipes.
      2. Per (level, draw), noise_helpers.inject_condition damages the
         training labels, and fold_model_inputs builds each model's 4
         configurations, the hybrids rebuilt from the damaged labels.
      3. Configurations that two models share (the same matrix) are one cell,
         so each detector and method runs once on it.
      4. knn_shapley is scored under every neighbour setting of each model.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT N32: TRIMMED REFITS, TWO LEARNERS, BETTER DETECTORS")
    print("=" * 70)

    condition = kwargs.get('noise_condition') or DEFAULT_NOISE_CONDITION
    labels = np.asarray(data['labels'], dtype=float)
    groups = data.get('groups')
    task_type = data['task_type']
    if task_type != 'regression':
        print(f"  SKIPPED: {dataset_name} is {task_type}. Every method here "
              f"ranks molecules by a squared residual, so n32 is regression "
              f"only.")
        return {'skipped': 'not_regression', 'task_type': task_type}
    # Only decided kinds where detection is defined. run() refuses the rest
    # before it gets here; the --tag path calls this function directly, so
    # the same refusal is repeated (2026-10-05, chat 5). Until then a tagged
    # run could inject a kind the noise grid cut.
    allowed = [c for c in decided_conditions(data) if detection_is_defined(c)]
    if not detection_is_defined(condition):
        raise ValueError(
            f"n32: condition {condition!r} moves every training label, so there "
            f"is no damaged subset to find. Run it under one of: "
            f"{', '.join(allowed)}.")
    if condition not in allowed:
        raise ValueError(
            f"n32: condition {condition!r} is not one of the decided kinds of "
            f"damage (KNN_SEARCH_SPACE_RECORD, noise grid, 2026-10-05). Run it "
            f"under one of: {', '.join(allowed)}.")

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)
    # Outer folds from config['n_splits'] (KIRBY_NOISE_N_FOLDS overrides), the
    # splits the per-fold searches are filed under. Until 2026-10-05:
    # cv_split with mode 'intermediary' at 1 fold and 'full' otherwise.
    folds, n_folds = noise_outer_folds(data, config)

    alphas = [float(a) for a in (kwargs.get('alphas') or ITLM_ALPHAS)]
    seeds = [int(s) for s in (kwargs.get('random_seeds') or RANDOM_SEEDS)]
    detectors = list(kwargs.get('detectors') or DETECTORS)
    oof_splits = int(kwargs.get('oof_splits', OOF_SPLITS))
    itlm_rounds = int(kwargs.get('itlm_rounds', ITLM_ROUNDS))
    flag_shares = [float(x) for x in (kwargs.get('flag_shares') or FLAG_SHARES)]
    arm_names = build_arm_names(alphas, seeds, flag_shares)

    # The dataset's two decided models (DECISIONS.md Step 1, handoff R1).
    # Until 2026-10-05: the config's five plus rf and svm
    # (downstream_model_list). A `models` kwarg (a smoke) can only narrow them.
    decided = list(decided_models(dataset_name, 'general'))
    asked = kwargs.get('models')
    downstream_models = ([m for m in decided if m in asked] if asked
                         else list(decided))
    if asked and [m for m in asked if m not in decided]:
        print(f"  models {[m for m in asked if m not in decided]} dropped: "
              f"not decided for {dataset_name} ({decided})")
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    models_unavailable = {
        m: 'get_model_factories gave no factory (size limit or unknown name)'
        for m in downstream_models if m not in model_factories}
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError("n32: no downstream model could be built.")

    # The neighbour settings (KNN_SEARCH_SPACE_RECORD, Adelaide 2026-10-05):
    # the dataset group's spaces and distances, no weighting, ks 5, 10, 20.
    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    knn_group = dataset_knn_group(data, dataset_name)
    n_molecules = len(set(smiles)) if has_smiles else len(labels)
    sim_settings = sweep_settings(dataset_name, n_molecules,
                                  has_smiles=has_smiles)
    from experiments.noise.n07_remediation import sweep_ks as _n07_sweep_ks
    sim_ks = tuple(_n07_sweep_ks())
    # A per-model space (base_*, augmented, hybrid) is scored per model; the
    # Tanimoto bits once per pass; forest proximity per configuration, since
    # its forest is grown on the configuration's matrix.
    per_model_settings = [s for s in sim_settings
                          if s.space in _ss.FEATURE_SPACES]
    forest_settings = [s for s in sim_settings
                       if s.space == _ss.FOREST_SPACE]
    shared_settings = [s for s in sim_settings
                       if s.space not in _ss.FEATURE_SPACES
                       and s.space != _ss.FOREST_SPACE]

    learner_factories = get_model_factories(
        list(EXCHANGE_LEARNERS), task_type=task_type, n_samples=len(labels))
    missing_learners = [m for m in EXCHANGE_LEARNERS
                        if m not in learner_factories]
    if missing_learners:
        raise RuntimeError(f"n32: the exchange learners {missing_learners} "
                           f"could not be built.")

    noise_groups = noise_groups_for(data)[0]
    if condition in GROUPED_STRATEGIES and noise_groups is None:
        raise RuntimeError(
            f"n32: condition {condition!r} needs scaffold groups and "
            f"{dataset_name} has none.")

    results = {
        'result_layout': RESULT_LAYOUT,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_condition': condition,
        'dose_units': level_unit(condition),
        'knn_group': knn_group,
        'nominal_flag_share': FLAG_SHARE.get(condition,
                                             'the level (fraction clipped)'),
        'flag_shares': flag_shares,
        'flag_share_note': (
            'two_learner and the flagged-only shrinks flag each share in '
            'flag_shares of the training molecules, the same at every '
            'condition and level (owner\'s decision 2026-09-28). '
            'nominal_flag_share is the condition\'s own share, written for '
            'reading only; no method reads it.'),
        'noise_draws': list(draws_for(
            condition, NOISE_DRAWS if kwargs.get('draws') is None
            else kwargs['draws'])),
        'models': downstream_models,
        'models_decided': decided,
        'models_unavailable': models_unavailable,
        'configurations_per_model': list(CONFIGURATION_SPACES),
        'selection_scope': 'fold_train_rows',
        'selection_note': (
            'per outer fold and decided model: the 2 best base '
            'representations and the 2 best general hybrids from searches on '
            'that fold\'s training rows with the clean training labels '
            '(noise_helpers.fold_configurations); which ones, per fold, is '
            'under folds. Each hybrid\'s training-row score is its score_mean '
            'there; held_out_score is the configuration fitted on the clean '
            'training labels at level 0.'),
        'n_folds': n_folds,
        'folds_run': [int(i) for i, _, _ in folds],
        'exchange_learners': list(EXCHANGE_LEARNERS),
        'exchange_rule': ('every downstream model is fitted on the '
                          'INTERSECTION of the two learners\' final kept sets'),
        'arms': arm_names,
        'arm_to_beat': 'no_cleaning',
        'detectors': detectors,
        'itlm_alphas': alphas,
        'itlm_rounds': itlm_rounds,
        'exchange_rounds': EXCHANGE_ROUNDS,
        'random_seeds': seeds,
        'oof_splits': oof_splits,
        'knn_shapley_utility': (
            'mean over the k nearest of -(y_i - y_test)^2, one term per '
            'neighbour; not the squared error of the neighbours\' mean'),
        'knn_shapley_neighbour_weighting': KNN_SHAPLEY_SCHEME_NOTE,
        'knn_shapley_counting_scheme': 'outside_axis_equal_shares',
        'similarity_settings': [s.name for s in sim_settings],
        'similarity_ks': list(sim_ks),
        'similarity_note': (
            'knn_shapley is scored under every neighbour setting and k, with '
            'scaffold-mates kept in the list. Keys are '
            'knn_shapley@<setting>|k<k>. A per-model space is under '
            'detection[level][neighbour_spaces/<model>], the Tanimoto bits '
            'under detection[level][neighbour_spaces], forest proximity under '
            'the configuration its forest was grown on. hybrid__distance1 and '
            'hybrid_similarity are neighbour spaces only; no model is fitted '
            'on them. Only the ordering of the neighbours is read.'),
        'neighbour_screen': (
            'not run: no method here refits a model per neighbour setting, '
            'so every setting is scored'),
        'detection_at_level_0': (
            'not computed: no molecule moved, so there is nothing to find'),
        'controls_run': controls_enabled(),
        'shrink_note': (
            'sm^2 and sn^2 are estimated from the corrupted labels: sm^2 is '
            'the mean square of y - p over the unflagged molecules, sn^2 how '
            'far the flagged molecules\' mean square exceeds it. The injector\'s own '
            'size is not read.'),
        'features_are_fixed_note': (
            'Each configuration is built once per (level, draw, fold); the '
            'hybrids choose their features from the CORRUPTED training labels. '
            'Every detector and method shares it, clean included.'),
        'robustness_note': (
            'damaged_minus_clean in each summary row is the no_cleaning score '
            'minus the clean score in the dataset\'s own metric (direction: '
            'lower_is_better). auc_norm is not computed here.'),
        'claim_scope': (
            'The noise is injected, so every number is confined to this '
            'condition on this dataset. Held-out molecules are graded against '
            'untouched labels in every arm.'),
        'per_level': {},
        'tails': {},
        'detection': {},
        'method_records': {},
        'folds': {},
    }

    passes = _levels_and_draws(data, dataset_name, condition,
                               levels=kwargs.get('levels'),
                               draws=kwargs.get('draws'))
    results['noise_levels'] = sorted({l for l, _ in passes})
    print(f"  {len(labels)} molecules, {len(folds)} of {n_folds} folds, "
          f"models {downstream_models}, {knn_group}")
    print(f"  condition {condition}, levels {results['noise_levels']}, "
          f"{len(passes)} (level, draw) passes per fold, {len(arm_names)} "
          f"arms, detectors {detectors}, {len(sim_settings)} neighbour "
          f"settings x {len(sim_ks)} ks")

    acc, tail_acc, det_acc, rec_acc, errors = {}, {}, {}, {}, {}
    fold_records, prov_acc = {}, {}
    t0 = time.time()

    # Rows and the per-pass checkpoint (PassLedger), 2026-09-25. A pass is one
    # (level, draw, fold); a finished one is skipped on a rerun and its
    # accumulators restored. The signature hashes the header above, which
    # carries RESULT_LAYOUT.
    ledger = PassLedger(output_dir, EXP_ID, dataset_name, condition,
                        signature=ledger_signature(
                            data, config, results,
                            [(tr, va) for _, tr, va in folds]),
                        restart=restart_flag(kwargs))
    if ledger.restored():
        st = ledger.restored()
        acc, tail_acc = st['acc'], st['tail_acc']
        det_acc, rec_acc, errors = st['det_acc'], st['rec_acc'], st['errors']
        fold_records = st.get('fold_records') or {}
        prov_acc = st.get('prov_acc') or {}
        print(f"  resumed {len(ledger.passes_done)} finished passes from "
              f"{ledger.ckpt_path}")

    def _state():
        return {'acc': acc, 'tail_acc': tail_acc, 'det_acc': det_acc,
                'rec_acc': rec_acc, 'errors': errors,
                'fold_records': fold_records, 'prov_acc': prov_acc}

    for fold_i, train_idx, val_idx in folds:
        fold_keys = [f'l{l:g}d{d}f{fold_i}' for l, d in passes]
        if all(ledger.pass_done(k) for k in fold_keys):
            print(f"\n  fold {fold_i}: every pass on disk in the checkpoint")
            continue
        print(f"\n  === fold {fold_i} of {n_folds}: {len(train_idx)} training, "
              f"{len(val_idx)} held-out rows ===")
        # The per-fold searches, on this fold's training rows. Stored per fold
        # (ocv{n}f{i}s{seed}) and shared with every experiment on the dataset.
        try:
            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=train_idx, n_folds=n_folds)
        except Exception as exc:
            print(f"    per-fold searches failed: {type(exc).__name__}: {exc}")
            errors[f'f{fold_i}|fold_configurations'] = (
                f'{type(exc).__name__}: {exc}')
            fold_records[str(fold_i)] = {
                'error': f'per-fold searches failed: {type(exc).__name__}: '
                         f'{exc}'}
            continue
        frec = fold_records.get(str(fold_i))
        if not frec or 'models' not in frec:
            # New, or an earlier run recorded only that this fold's searches
            # failed.
            frec = fold_records[str(fold_i)] = {
                'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)),
                'models': {}}
        for m in downstream_models:
            rec = configuration_record(entries.get(m))
            rec['metric'] = metric_name
            rec['lower_is_better'] = bool(lower_is_better)
            frec['models'].setdefault(m, rec)
            print(f"    {m}: base {rec.get('base')}, hybrids "
                  f"{[h.get('label') for h in rec.get('hybrids') or []]}")

        y_val = labels[val_idx]
        y_clean_tr = labels[train_idx]
        g_tr = (np.asarray(groups)[train_idx] if groups is not None
                else np.arange(len(train_idx)))

        for level, draw in passes:
            lvl_key = f'{level:g}'
            pass_key = f'l{lvl_key}d{draw}f{fold_i}'
            if ledger.pass_done(pass_key):
                continue
            print(f"\n    --- {condition} / level {lvl_key} / draw {draw} / "
                  f"fold {fold_i} ---")
            pass_rows = []
            try:
                y0, cmask, prov = inject_condition(
                    data, dataset_name, condition, level, draw,
                    train_idx=train_idx, groups=noise_groups)
            except Exception as exc:
                errors[f'{lvl_key}|d{draw}|f{fold_i}|inject'] = (
                    f'{type(exc).__name__}: {exc}')
                continue
            prov_acc[pass_key] = convert_numpy(prov)
            y0 = np.asarray(y0, dtype=float)
            cmask = np.asarray(cmask, dtype=bool)
            n_tr = int(len(y0))
            moved = int(cmask.sum())

            # Each model's 4 configurations, the hybrids rebuilt on this
            # fold's training rows from THIS pass's damaged labels with the
            # per-fold recipe (its own representation scores: dirty-run bug
            # 1), 0/1 features unscaled.
            inputs = {}
            for m in downstream_models:
                ins, unav, names = fold_model_inputs(
                    data, dataset_name, m, entries.get(m),
                    train_idx=train_idx, val_idx=val_idx, y_train=y0,
                    factory=model_factories[m], metric=metric)
                inputs[m] = ins
                mrec = frec['models'][m]
                if names:
                    mrec.setdefault('configuration_names', dict(names))
                mrec.setdefault('n_features', {})
                for sp, (Xa, _Xb) in ins.items():
                    mrec['n_features'].setdefault(sp, int(np.shape(Xa)[1]))
                for sp in CONFIGURATION_SPACES:
                    if sp not in ins:
                        # The per-fold search's own reason first; the
                        # builder's text still names the old stored pool.
                        why = (((entries.get(m) or {}).get('unavailable')
                                or {}).get(sp) or unav.get(sp) or 'not built')
                        (mrec.setdefault('configurations_unavailable', {})
                         .setdefault(sp, why))
                        errors[f'{lvl_key}|d{draw}|f{fold_i}|{sp}/{m}|build'] = (
                            why)

            # A configuration two models share (the same matrix) is one cell:
            # every detector and method below reads only the features and the
            # labels, so it runs once and both models are fitted on it.
            cells = []
            for space in CONFIGURATION_SPACES:
                by_hash = {}
                for m in downstream_models:
                    if space not in inputs[m]:
                        continue
                    X_tr_s, X_va_s = inputs[m][space]
                    h = (_nc.array_hash(np.asarray(X_tr_s)),
                         _nc.array_hash(np.asarray(X_va_s)))
                    if h in by_hash:
                        by_hash[h][2].append(m)
                    else:
                        by_hash[h] = [X_tr_s, X_va_s, [m]]
                for X_tr_s, X_va_s, ms in by_hash.values():
                    cells.append((space, X_tr_s, X_va_s, ms))

            # knn_shapley under every neighbour setting of each model. Not at
            # level 0, where no molecule moved (the dirty run computed every
            # detector there and threw the scores away).
            run_knn = 'knn_shapley' in detectors and moved > 0
            fsims = {}
            sim_unavail = {}
            for mi, m in enumerate(downstream_models):
                entry = entries.get(m)
                if entry is None:
                    continue
                fsim = fold_similarity(
                    dataset_name, data, reps=data['reps'],
                    train_idx=train_idx, val_idx=val_idx, y_train=y0,
                    groups_train=g_tr,
                    base_by_model={m: list(entry.get('base') or [])},
                    recipe_for_space_by_model={
                        m: dict(entry.get('recipes') or {})},
                    model_factories=model_factories, metric=metric,
                    task_type=task_type)
                # The 4 configurations are the same matrices as the base and
                # general-hybrid neighbour spaces: handed over, not rebuilt.
                for sp, (Xa, Xb) in inputs[m].items():
                    fsim._spaces[(m, sp)] = (Xa, Xb, None)
                for sp, why in (entry.get('unavailable') or {}).items():
                    if sp not in inputs[m]:
                        fsim._spaces[(m, sp)] = (None, None, why)
                        fsim.unavailable[f'{m}|{sp}'] = why
                fsims[m] = fsim
                if not run_knn:
                    continue
                _knn_shapley_sweep(
                    fsim, per_model_settings, sim_ks, None, y0, cmask,
                    condition, m, f'neighbour_spaces/{m}', lvl_key, fold_i,
                    det_acc, errors, unavailable=sim_unavail)
                if mi == 0 and shared_settings:
                    _knn_shapley_sweep(
                        fsim, shared_settings, sim_ks, None, y0, cmask,
                        condition, m, 'neighbour_spaces', lvl_key, fold_i,
                        det_acc, errors, unavailable=sim_unavail)
            if sim_unavail:
                (frec.setdefault('neighbour_spaces_unavailable', {})
                 .update(sim_unavail))

            for space, X_tr_s, X_va_s, cell_models in cells:
                _ekey = (space if cell_models == list(downstream_models)
                         else f'{space}/{cell_models[0]}')
                rec = {'pass': pass_key, 'training_molecules': n_tr,
                       'molecules_moved': moved, 'models': list(cell_models)}

                try:
                    # The out-of-fold LightGBM target, cached on the label
                    # hash (identical across n24, n25, n30-n34, n38, n40
                    # and n41 for the same draw).
                    p = _nc.cached_oof_predict(dataset_name, X_tr_s, y0,
                                               g_tr, n_splits=oof_splits)
                except Exception as exc:
                    errors[f'{lvl_key}|{_ekey}|f{fold_i}|oof'] = (
                        f'{type(exc).__name__}: {exc}')
                    p = None

                # ── Detectors, graded against the mask (not at level 0) ──
                if run_knn and forest_settings and cell_models[0] in fsims:
                    _knn_shapley_sweep(
                        fsims[cell_models[0]], forest_settings, sim_ks,
                        X_tr_s, y0, cmask, condition, cell_models[0], _ekey,
                        lvl_key, fold_i, det_acc, errors,
                        unavailable=rec.setdefault('similarity_unavailable',
                                                   {}))
                for det in (detectors if moved > 0 else ()):
                    if det == 'knn_shapley':
                        continue
                    if det == 'oof_residual' and p is None:
                        continue
                    try:
                        sus = detector_scores(det, X_tr_s, y0, g_tr, p,
                                              oof_splits)
                    except Exception as exc:
                        errors[f'{lvl_key}|{_ekey}|{det}|f{fold_i}'] = (
                            f'{type(exc).__name__}: {exc}')
                        continue
                    d = detection_scores(cmask, sus, condition)
                    if d is not None:
                        (det_acc.setdefault(lvl_key, {})
                         .setdefault(_ekey, {})
                         .setdefault(det, []).append(d))

                # ── The training labels and keep masks ──
                cols = {'clean': (y_clean_tr, None),
                        'no_cleaning': (y0, None)}
                for a in alphas:
                    try:
                        keep, last_loss, rounds = itlm(
                            X_tr_s, y0, g_tr, a, rounds=itlm_rounds,
                            oof_splits=oof_splits,
                            first_loss=(None if p is None
                                        else (y0 - p) ** 2))
                    except Exception as exc:
                        errors[f'{lvl_key}|{_ekey}|itlm_a{a:g}|'
                               f'f{fold_i}'] = f'{type(exc).__name__}: {exc}'
                        continue
                    cols[f'itlm_a{a:g}'] = (y0, keep)
                    n_rm = int(n_tr - keep.sum())
                    rec[f'itlm_a{a:g}'] = {
                        'rounds': rounds, 'molecules_removed': n_rm,
                        'moved_molecules_among_the_removed':
                            int(np.sum(cmask & ~keep)),
                        'detection_by_last_round_loss':
                            (detection_scores(cmask, last_loss, condition)
                             if moved > 0 else None)}
                    for s in seeds:
                        rk = random_keep(n_tr, n_rm, replicate_seed(
                            s, lvl_key, draw, fold_i, n_tr, n_rm))
                        cols[f'itlm_a{a:g}_random/seed{s}'] = (y0, rk)
                # Each swept share of the training molecules, not the
                # condition's own share (owner's decision 2026-09-28).
                for sh in flag_shares:
                    tl_arm = f'two_learner_share{sh:g}'
                    try:
                        both, xrec, xloss = two_learner_exchange(
                            X_tr_s, y0, g_tr,
                            learner_factories[EXCHANGE_LEARNERS[0]],
                            learner_factories[EXCHANGE_LEARNERS[1]],
                            1.0 - sh, oof_splits=oof_splits)
                        cols[tl_arm] = (y0, both)
                        n_rm = int(n_tr - both.sum())
                        xrec['moved_molecules_among_the_removed'] = int(
                            np.sum(cmask & ~both))
                        xrec['detection_by_larger_of_the_two_losses'] = (
                            detection_scores(cmask, xloss, condition)
                            if moved > 0 else None)
                        rec[tl_arm] = xrec
                        for s in seeds:
                            rk = random_keep(n_tr, n_rm, replicate_seed(
                                'x', s, lvl_key, draw, fold_i, n_tr, n_rm))
                            cols[f'{tl_arm}_random/seed{s}'] = (y0, rk)
                    except Exception as exc:
                        errors[f'{lvl_key}|{_ekey}|{tl_arm}|f{fold_i}'] = (
                            f'{type(exc).__name__}: {exc}')
                    if p is not None:
                        one, per, srec = shrink_flagged(y0, p, sh)
                        rec[f'shrink_flagged_share{sh:g}'] = srec
                        if one is not None:
                            cols[f'shrink_flagged_one_weight_share{sh:g}'] = (
                                one, None)
                            cols[f'shrink_flagged_per_row_share{sh:g}'] = (
                                per, None)

                rec_acc.setdefault(lvl_key, {}).setdefault(
                    _ekey, []).append(rec)

                space_cell = acc.setdefault(lvl_key, {}).setdefault(
                    space, {})
                tail_cell = tail_acc.setdefault(lvl_key, {}).setdefault(
                    space, {})
                pass_scores = {}
                for arm in arm_names:
                    if arm not in cols:
                        continue
                    y_use, keep = cols[arm]
                    for mname in cell_models:
                        s_, tl, err = _fit_score(
                            model_factories[mname], X_tr_s, y_use, X_va_s,
                            y_val, metric_fn, metric_name, keep=keep,
                            dataset_name=dataset_name, model_name=mname)
                        if s_ is None:
                            errors[f'{lvl_key}|{_ekey}|{arm}|{mname}|'
                                   f'f{fold_i}'] = err
                            continue
                        (space_cell.setdefault(arm, {})
                         .setdefault(mname, {})[pass_key]) = s_
                        pass_scores.setdefault(arm, {})[mname] = s_
                        if tl is not None:
                            (tail_cell.setdefault(arm, {})
                             .setdefault(mname, {})[pass_key]) = tl
                if level == 0.0:
                    # The configuration's held-out score: fitted on the clean
                    # training labels, scored on the held-out rows.
                    for mname, s_ in (pass_scores.get('clean') or {}).items():
                        (frec['models'][mname]
                         .setdefault('held_out_score', {})[space]) = s_

                # ── Controls (KIRBY_NOISE_CONTROLS=1): each arm that
                # changed the data, twinned by controls.py. Written as
                # rows only; the summary is unchanged. ──
                for arm in arm_names:
                    if (arm not in cols or arm in ('clean', 'no_cleaning')
                            or '_random/' in arm):
                        continue
                    y_use, keep = cols[arm]
                    twins = control_twins(
                        arm_result(y0, y=None if keep is not None else y_use,
                                   keep=keep),
                        X_tr_s, y0, (replicate_seed(arm), draw, fold_i,
                                     int(round(level * 1000))))
                    for tw, by_m in twin_scores(
                            dataset_name, twins, model_factories,
                            cell_models, X_va_s, y_val, metric).items():
                        pass_scores[f'{arm}|{tw}'] = by_m

                # ── D against D', one row per (arm, model). model_input is
                # the configuration name ('base_1'), the model its own
                # field: until 2026-10-05 it was 'base_1/<first model of the
                # cell>', even on the other model's rows, so a join with
                # n30 and n31's 'base_1' missed rows (dirty-run bug 13). ──
                pass_rows += arm_rows(
                    dataset_name=dataset_name, exp_id=EXP_ID,
                    metric=metric, condition=condition, level=level,
                    draw=draw, fold=fold_i, model_input=space,
                    scores_by_arm=pass_scores)
            ledger.add_rows(pass_rows)
            ledger.finish_pass(pass_key, _state())
            print(f"    level {lvl_key} draw {draw} fold {fold_i}: "
                  f"{moved} of {n_tr} training molecules moved, "
                  f"{len(cells)} configuration cells, {len(pass_rows)} "
                  f"D-vs-D' rows [{time.time() - t0:.0f}s]")

    results['folds'] = convert_numpy(fold_records)
    results['injection_provenance'] = convert_numpy(prov_acc)
    results['per_level'] = convert_numpy(acc)
    results['tails'] = convert_numpy(tail_acc)
    results['detection'] = convert_numpy({
        lvl: {sp: {d: _mean_detection(rows) for d, rows in by_det.items()}
              for sp, by_det in by_sp.items()}
        for lvl, by_sp in det_acc.items()})
    results['method_records'] = convert_numpy(rec_acc)
    results['d_vs_dprime'] = ledger.describe()
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy(_summarise(
        acc, tail_acc, arm_names, downstream_models, alphas, seeds,
        metric_name, lower_is_better, shares=flag_shares))
    _print_summary(results, metric_name)
    return results


def _mean_detection(rows):
    """Average the detection readings over the passes of one level."""
    if not rows:
        return None
    keys = ('detection_auroc', 'detection_average_precision',
            'detection_precision_at_k', 'detection_n_corrupted',
            'detection_n_training')
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows
                if r.get(k) is not None and np.isfinite(float(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float('nan')
    out['detection_n_passes'] = len(rows)
    return out


def _paired(base, got, lower_is_better):
    """n24:981-1006. Positive gain means the second beat the first."""
    shared = sorted(set(base) & set(got))
    gains = np.array([(base[k] - got[k]) if lower_is_better
                      else (got[k] - base[k]) for k in shared], dtype=float)
    gains = gains[np.isfinite(gains)]
    scale = max(abs(float(np.mean(list(base.values())))) if base else 1.0, 1.0)
    tol = 1e-9 * scale
    return {
        'n_passes_paired': int(gains.size),
        'passes_won': int((gains > tol).sum()),
        'passes_lost': int((gains < -tol).sum()),
        'passes_tied': int((np.abs(gains) <= tol).sum()),
        'mean_gain': float(gains.mean()) if gains.size else float('nan'),
        'spread_over_pass_gains': (float(gains.std()) if gains.size
                                   else float('nan')),
        'pass_gains': [float(x) for x in gains],
    }


def recovery_share(clean, noisy, arm, lower_is_better,
                   gap_min_frac=GAP_MIN_FRAC):
    """Share of the clean-to-corrupted gap an arm recovers, or (None, why)."""
    if not all(np.isfinite(v) for v in (clean, noisy, arm)):
        return None, 'a score is missing'
    gap = (noisy - clean) if lower_is_better else (clean - noisy)
    floor = gap_min_frac * abs(clean)
    if gap < floor:
        return None, (f'the clean-to-corrupted gap is {gap:+.6f}, under '
                      f'{gap_min_frac:g} of the clean score ({floor:.6f})')
    gain = (noisy - arm) if lower_is_better else (arm - noisy)
    return float(gain / gap), None


def _mean_on(d, keys):
    vals = [d[k] for k in keys if k in d and np.isfinite(d[k])]
    return float(np.mean(vals)) if vals else float('nan')


def _tail_row(tails_arm, tails_base):
    """Mean of each tail number for the arm, and its mean change from
    no_cleaning on the passes both scored."""
    keys = sorted(set(tails_arm) & set(tails_base))
    out = {'n_passes': len(keys)}
    for f in ('tail_rmse_top', 'tail_rmse_bottom', 'calibration_slope'):
        a = [tails_arm[k][f] for k in keys]
        b = [tails_base[k][f] for k in keys]
        out[f] = float(np.mean(a)) if a else float('nan')
        out[f'{f}_change_from_no_cleaning'] = (
            float(np.mean(np.subtract(a, b))) if a else float('nan'))
    return out


def _summarise(acc, tail_acc, arm_names, models, alphas, seeds, metric_name,
               lower_is_better, shares=FLAG_SHARES):
    """Every arm against no_cleaning; each removal arm against its randoms.

    2026-10-05 (chat 5): each row also carries its spread over folds
    (noise_helpers.fold_spread over the per-fold means, 'one fold, no spread'
    on one fold) and damaged_minus_clean, the no_cleaning score minus the clean
    score on the passes both scored, in the dataset's own metric."""
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'headline': ('passes won against no_cleaning, paired per pass; a '
                        'pass is one (draw, fold)'),
           'per_level': {}}
    removal = {f'itlm_a{a:g}': [f'itlm_a{a:g}_random/seed{s}' for s in seeds]
               for a in alphas}
    for sh in shares:
        removal[f'two_learner_share{sh:g}'] = [
            f'two_learner_share{sh:g}_random/seed{s}' for s in seeds]
    for lvl, by_space in acc.items():
        lrow = {}
        for space, by_arm in by_space.items():
            base = by_arm.get('no_cleaning') or {}
            clean = by_arm.get('clean') or {}
            tails = (tail_acc.get(lvl) or {}).get(space) or {}
            srow = {}
            for m in models:
                b = base.get(m) or {}
                if not b:
                    continue
                row = {'no_cleaning_mean': _mean_on(b, list(b)),
                       'no_cleaning_over_folds': fold_spread(
                           _per_fold_means(b)),
                       'per_arm': {}}
                c_m = clean.get(m) or {}
                both = sorted(set(c_m) & set(b))
                diffs = {k: float(b[k]) - float(c_m[k]) for k in both}
                row['damaged_minus_clean'] = (
                    _mean_on(diffs, both) if both else float('nan'))
                row['damaged_minus_clean_over_folds'] = fold_spread(
                    _per_fold_means(diffs))
                for arm in arm_names:
                    if arm == 'no_cleaning':
                        continue
                    g = (by_arm.get(arm) or {}).get(m) or {}
                    if not g:
                        continue
                    e = {'vs_no_cleaning': _paired(b, g, lower_is_better),
                         'over_folds': fold_spread(_per_fold_means(g))}
                    if arm != 'clean':
                        keys = sorted(set(clean.get(m) or {}) & set(b) & set(g))
                        sh, why = (recovery_share(
                            _mean_on(clean[m], keys), _mean_on(b, keys),
                            _mean_on(g, keys), lower_is_better)
                            if keys else (None, 'no pass scored all three'))
                        e['recovery_share'] = sh
                        e['recovery_share_refused_because'] = why
                    if arm in removal:
                        vs = {}
                        for r_arm in removal[arm]:
                            rg = (by_arm.get(r_arm) or {}).get(m) or {}
                            if rg:
                                vs[r_arm] = _paired(rg, g, lower_is_better)
                        e['vs_random_removal'] = vs
                        e['beat_every_random_seed'] = bool(vs and all(
                            v['passes_won'] * 2 > v['n_passes_paired']
                            for v in vs.values()))
                    ta = (tails.get(arm) or {}).get(m) or {}
                    tb = (tails.get('no_cleaning') or {}).get(m) or {}
                    if ta and tb:
                        e['tails'] = _tail_row(ta, tb)
                    row['per_arm'][arm] = e
                srow[m] = row
            lrow[space] = srow
        out['per_level'][lvl] = lrow
    return out


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric_name}; paired per pass against no_cleaning)")
    for lvl, by_sp in (results.get('detection') or {}).items():
        for sp, by_det in by_sp.items():
            for det, row in by_det.items():
                if row:
                    print(f"  level {lvl} {sp}/{det}: AUROC "
                          f"{row['detection_auroc']:.3f}, average precision "
                          f"{row['detection_average_precision']:.3f}, over "
                          f"{row['detection_n_corrupted']:.0f} moved molecules "
                          f"of {row['detection_n_training']:.0f}")
    for lvl, lrow in s['per_level'].items():
        print(f"\n  level {lvl}:")
        for space, srow in lrow.items():
            for m, row in srow.items():
                print(f"    {space}/{m}: no_cleaning "
                      f"{row['no_cleaning_mean']:.4f}")
                for arm, e in row['per_arm'].items():
                    if '_random/' in arm:
                        continue
                    v = e['vs_no_cleaning']
                    tail = ''
                    if 'beat_every_random_seed' in e:
                        tail = (f"; beat every random seed: "
                                f"{e['beat_every_random_seed']}")
                    print(f"      {arm:28s} won {v['passes_won']} of "
                          f"{v['n_passes_paired']} passes, mean "
                          f"{v['mean_gain']:+.4f} (spread "
                          f"{v['spread_over_pass_gains']:.4f}){tail}")
    print("\n  The count of passes won is the headline. A mean gain no larger "
          "than the spread of the per-pass gains behind it is not a gain.")


def main():
    parser = create_experiment_parser(
        'N32: trimmed refits, two learners, better detectors')
    parser.add_argument('--noise-condition', type=str, default=None,
                        help=f'Injected condition (default '
                             f'{DEFAULT_NOISE_CONDITION}). One of the decided '
                             f'kinds where detection is defined: '
                             f'outlier_p05, censoring, shift_only_p10, '
                             f'sparse_dprime_up_p10 (regression)')
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated levels. Default: the shared grid; '
                             'for shift_only_* the shift in label SDs')
    parser.add_argument('--draws', type=str, default=None,
                        help='Comma-separated injection replicates')
    parser.add_argument('--alphas', type=str, default=None,
                        help='Comma-separated ITLM kept shares')
    parser.add_argument('--detectors', type=str, default=None,
                        help=f'Comma-separated detectors (default '
                             f'{",".join(DETECTORS)})')
    parser.add_argument('--random-seeds', type=str, default=None,
                        help='Comma-separated random-removal replicates')
    parser.add_argument('--flag-shares', type=str, default=None,
                        help='Comma-separated shares of training molecules '
                             'two_learner and the shrinks flag (default '
                             f'{",".join(f"{x:g}" for x in FLAG_SHARES)})')
    parser.add_argument('--exp-folds', type=int, default=None,
                        help='Outer folds. Overrides the 1 or 5 --mode gives')
    parser.add_argument('--tag', type=str, default=None,
                        help='Extra text on the result filename')
    args = parser.parse_args()
    config = get_config(args)
    if args.exp_folds:
        config['n_splits'] = int(args.exp_folds)
    data = load_data(args.dataset, rep_preset=args.reps)

    kw = {}
    if args.noise_condition:
        kw['noise_condition'] = args.noise_condition
    if args.levels:
        kw['levels'] = [float(x) for x in args.levels.split(',')]
    if args.draws:
        kw['draws'] = [int(x) for x in args.draws.split(',')]
    if args.alphas:
        kw['alphas'] = [float(x) for x in args.alphas.split(',')]
    if args.detectors:
        kw['detectors'] = [x.strip() for x in args.detectors.split(',')]
    if args.random_seeds:
        kw['random_seeds'] = [int(x) for x in args.random_seeds.split(',')]
    if args.flag_shares:
        kw['flag_shares'] = [float(x) for x in args.flag_shares.split(',')]

    if args.tag:
        # A tagged run is a one-off: one condition, written under the tag.
        kw.setdefault('noise_condition', DEFAULT_NOISE_CONDITION)
        res = _run_one_condition(data, config, args.output, args.dataset, **kw)
        save_experiment_results(res, args.output, EXP_ID, args.dataset,
                                filename_suffix=f"{kw['noise_condition']}_{args.tag}")
        return
    # run() writes one file per condition itself; this is the index.
    res = run(data, config, args.output, args.dataset, **kw)
    save_experiment_results(res, args.output, EXP_ID, args.dataset)


if __name__ == '__main__':
    main()
