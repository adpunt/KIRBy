#!/usr/bin/env python3
"""N23: can a model's own per-molecule uncertainty clean labels?

Paper: Noise.

THE QUESTION. Every detector in this suite scores a molecule by comparing its
label to the labels of its nearest neighbours. A model that predicts a SPREAD
per molecule is a different quantity: it is what the model thinks that
molecule's label noise is, learned from the features rather than from the
neighbours. Nothing in the suite has used one to clean anything.

TWO SHAPES, and the point is that both are tried.

  drop   remove the training molecules whose predicted spread is largest. The
         share removed is swept over DROP_SHARES.
  move   move each training label toward the model's own out-of-fold prediction,
         further the wider that molecule's predicted spread. The strength is
         swept over MOVE_STRENGTHS and the rule over MOVE_METHODS.

  none   the score to beat: train on the labels as they are.

THE TWO MOVE RULES ARE NOT THE SAME KNOB, and both run because the difference
matters here. 'proportional' rescales the spreads to run from zero to one and
moves each label by `strength` times its rescaled value -- so it keeps the ORDER
of the molecules and throws away the SCALE. 'confidence' uses the spread as it
comes, in the label's own units, and moves each label by
`strength * spread / (1 + strength * spread)`. A predicted spread is in label
units, so 'confidence' is the rule that uses what makes it different from a
neighbour score; 'proportional' is the rule `n07.label_smooth` already uses, so
it is what makes the two comparable. Both are recorded on every row.

WHICH UNCERTAINTY NUMBERS, AND WHAT IS KNOWN ABOUT THEM.

  hetero_gp           a Gaussian process for the mean plus a network
                      predicting each molecule's noise, trained together. Ported
                      into KIRBy on 2026-09-18 from qsar_qm_models, which is the
                      only model in either roster whose two uncertainty terms
                      come apart. Out of fold.
  ngboost             the spread NGBoost predicts per molecule, out of fold.
  heteroscedastic_nn  a network with a second output predicting each molecule's
                      variance, out of fold.
  gaussian_process    KIRBy's, and it is NOT the same quantity -- see below.
  activity_cliff_aware  the neighbour-based score n07, n18 and n20 already use.
                      It reads the molecule's own LABEL against its neighbours'
                      rather than predicting anything from the features, which
                      is why it is the only one of the five that can answer
                      under a condition that picks its corrupted molecules at
                      random. The thing to beat.

⛔ THREE OF THE FIVE CAN ONLY ANSWER WHEN THE NOISE IS A PROPERTY OF THE
STRUCTURE -- hetero_gp, ngboost and heteroscedastic_nn. They predict a
molecule's noise from its features and they are scored out of fold, so a
held-out molecule can only be ranked when its own features say how noisy it is.
`outlier_p01`, `outlier_p05` and `outlier_p10` choose the molecules that get the
wider error AT RANDOM, with no reference to the features
(NOISE_AUDIT_2026_09_10.md 6.7). `grouped_wider` gives the wider error to whole
scaffold families, which a representation encodes.

The other two read the molecule's OWN LABEL, so they answer under any condition
that moves a subset of labels. activity_cliff_aware compares the label to its
neighbours'. gaussian_process adds the out-of-fold residual to the posterior
spread, and that residual is the same kind of quantity -- see the note below,
which said so while LEARNS_NOISE_FROM_FEATURES listed it anyway until
2026-09-21.

⛔ `grouped_wider` CANNOT RUN ON EVERY DATASET, AND IT IS THE ONLY ENTRY IN
NOISE_KEYED_TO_STRUCTURE. It needs one scaffold-group id per molecule, built by
`kirby.noise_spec.assign_noise_groups(smiles)`. `dataset_policy.noise_groups_for`
returns None for any set that loads no SMILES -- antibody, nucleotide or
expression -- and the injector then raises. So on flab_thermo_tresanco there is
no condition at all under which the three feature-based sources are defined, and
the guard in run() says so rather than letting every fold raise and writing a
file with no score in it.

MEASURED 2026-09-18 on a 300-molecule fixture, out of fold over three folds, the
rank correlation between each source's number and the true noise width:

    which molecules are loud   hetero_gp   ngboost   activity_cliff_aware
    drawn at random               +0.149    -0.027                 +0.057
    decided by the features       +0.383    +0.511                 +0.078

So a chance-level reading under an outlier condition is the condition, not the
method. `sources_defined_under_this_condition` is written onto every result file
so that cannot be misread.

⛔ A MODEL'S UNCERTAINTY IS TWO NUMBERS ADDED TOGETHER, AND ONLY ONE OF THEM IS
A CLEANING SIGNAL. One is how noisy that molecule's label looks. The other is
how unsure the model is about that region of chemistry. Dropping molecules by
the second is dropping the molecules the model finds hard, which is n18's own
caveat at :23-26.

Which models tell the two apart was measured in the sister study, not here:
`~/repos/qsar_qm_models/RERUN_PLAN.md` sections 2.30 and 2.32, three seeds,
3,000 real QM9 molecules, three folds out of fold on Murcko scaffold groups,
plain Gaussian noise at level 0 and level 1. Read those numbers as that study's,
dated 2026-08-30 and 2026-08-31, and re-check before repeating them:

  Gaussian process       label-noise term rises x17.2, model-unsure term x1.5.
                         Separation 11.2 to 18.7 over three seeds. Its two terms
                         are the same SIZE as each other, which is why the
                         separation means something. The only PASS.
  NGBoost                label-noise term rises x9.1, the second best measured,
                         and it stays on that study's uncertainty list. It has
                         NO model-unsure term: that term came out 3,581 times
                         smaller than the other, and it RISES with the noise,
                         which disqualifies it. So nothing there can check that
                         NGBoost's number is not tracking model ignorance.
  variance-head network  separation 1.5 to 2.3. Real but small.
  random forest          separation 1.1, written down as FAIL: "one bootstrap
                         causes both".

⛔ KIRBY'S `gaussian_process` ESTIMATOR DOES NOT RETURN THE NUMBER THAT PASSED.
Read out of `src/kirby/utils/noise_estimation.py:726-800` this session. Its
per-molecule score is the posterior standard deviation PLUS the out-of-fold
residual. Its label-noise term is `k2__noise_level`, one number for the whole
fold, identical for every molecule, so it cannot rank molecules at all. It is
swept here anyway, because the difference between it and the two that do predict
a spread per molecule is then measured on this data rather than carried over
from another repository's table.

AND THE RESIDUAL HALF OF THAT SUM IS WHY IT IS THE BEST OF THE FIVE UNDER THE
DEFAULT CONDITION. `np.abs(y - y_pred)` reads the molecule's own label against
what a model that did not see it predicted, which finds a label that moved
however the condition chose it. MEASURED 2026-09-21, flab_thermo_tresanco, 672
antibodies, base representation esm2_cdr at 1280 features, two inner folds, one
outer fold of two, draw 0. Average precision at ranking the training molecules
whose label actually moved, each against the share that moved:

  outlier_p10 at level 0.5, 29 of the 336 training molecules moved,
  so a coin flip scores 0.0863
      gaussian_process      0.3016
      activity_cliff_aware  0.2664
      heteroscedastic_nn    0.0878

  censoring at level 0.25, 79 and 86 of the 336 training molecules moved on the
  two folds, so a coin flip scores 0.2351 and 0.2560
      gaussian_process      0.2663 and 0.2764
      heteroscedastic_nn    0.1947 and 0.2776
      activity_cliff_aware  0.1881 and 0.2340

Clipping every label above a cut-point does NOT give the three a condition they
can see: all three sit at the coin flip, and two of them below it. A clipped
label is not far from its neighbours -- clipping pulls the top of the range
together -- so it is the one condition in DETECTION_IS_DEFINED that makes the
neighbour comparison worse rather than better.

THE CHECK THAT DOES NOT NEED A MODEL FIT AT ALL. Under injection there is a list
of the molecules whose label actually moved. `noise_helpers.detection_scores`
grades each uncertainty number against that list before any cleaning runs. That
is the direct answer to "is this number a noise signal", and it costs nothing:
the spreads and the list are both already in memory. It is defined only under
the conditions where a corrupted SUBSET exists -- `outlier_p10` is one.

THE ZERO-NOISE CONTROL. Level 0.0 in the injected regime corrupts nothing, so
there is nothing to clean. A source whose cleaning still wins folds there is
removing or moving the molecules the model finds hard. That is n18's QM9
argument put on a level axis, and it applies to every dataset rather than to one.

BOTH REGIMES.

  injected      each decided kind in which a corrupted subset exists
                (SWEPT_CONDITIONS), at the decided levels for the dataset's
                group (noise_helpers.decided_levels), three draws per level
                above 0 and one at level 0, which is the labels as delivered.
                Detection is written here. (Until 2026-10-05: outlier_p10 and
                the other DETECTION_IS_DEFINED kinds on the old shared grid.)
  as_they_come  nothing is corrupted. n18's regime. A win is a win on the data
                as it arrived, and there is no list to be right about.

WHAT THIS DOES NOT ASK. The features are fixed. Since 2026-10-05 (chat 5,
HANDOFF_EXPERIMENT_ALIGNMENT section 2.4) each of the dataset's two decided
models is fitted on four configurations per outer fold -- its 2 best single
representations (`base_1`, `base_2`) and its 2 best general hybrids
(`hybrid`, `hybrid__general2`) -- all ranked by searches on that fold's
training rows (noise_helpers.fold_configurations), so 8 configurations for
two models. The hybrids are rebuilt on the fold's training molecules at the
labels the pass holds, before any cleaning. What each model was fitted on, per
fold, is in `fold_configurations` in the result file.

The similarity hybrid (features chosen so that each molecule's activity is
predicted by its five nearest neighbours in the distance they define) and the
distance hybrid (the top hybrid of the distance search) are NOT fitted
configurations. They are neighbour spaces in the similarity sweep (handoff 2.3
item 1). Until 2026-10-05 the similarity hybrid was a fitted space, and every
recipe and base representation came from a whole-dataset stored search, for
five models. No cleaning reaches the feature ranking or the budget split in
any configuration. Whether it should is n20's question, not this one.

WHAT THE HELD-OUT MOLECULES ARE GRADED AGAINST. `labels[val_idx]`, untouched, in
every arm and both regimes. No cleaned label and no corrupted label is ever
scored against.

THE HEADLINE IS A COUNT OF FOLDS, not a mean, for n18's reason at :33-37. Every
comparison is PAIRED: the same fold, the same held-out molecules, two arms.

SIMILARITY AXIS, ROWS, CACHES, CHECKPOINT (2026-09-25, plan sections 1-3;
revised 2026-10-05, chat 5).
Since 2026-10-05 (chat 5) the main table reads no neighbours. Until then
activity_cliff_aware, and the move target of a source with no prediction of
its own, found 10 neighbours in the configuration's own features by the
process-wide distance, every neighbour counting equally; the kNN record cut
that space (axis 1) and that counting (axis 5). Both now run only in the
similarity sweep: the neighbour settings of the dataset's kNN group
(n07.sweep_settings), a molecule's scaffold-mates kept in its list (axis 6),
every (setting, k) graded for detection, and only the screen's top 3
(setting, k, counting scheme) refitted, the move target being the neighbours'
label weighted by that scheme. So activity_cliff_aware has no row in the
summary's folds-won table; its numbers are in sim_detection and the rows
file. Every (source, arm, model) writes one D-versus-D' row per pass to
<dataset>__<condition>__fold_aligned_swept_neighbours_2026-10-05.rows.jsonl
beside the
condition file; the accumulators are checkpointed per (regime, level, draw,
fold); the hybrid builds, the scaled representations, the no-cleaning and
clean-label fits of seeded models and the uncertainty estimates go through
noise_cache. Controls run with KIRBY_NOISE_CONTROLS=1.

Usage:
    python n23_uncertainty_cleaning.py --dataset herg_ki --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import time
import zlib
from pathlib import Path
import numpy as np

# 2026-10-05 (chat 5): the whole-dataset readers (get_best_search_for_downstream,
# noise_base_reps_by_model, pool_recipe_per_downstream_model,
# pool_recipes_by_family), cv_split, DOWNSTREAM_MODELS and StandardScaler are no
# longer imported. Every configuration now comes from a search on the outer
# fold's training rows (noise_helpers.fold_configurations). n23 scales no
# features itself: noise_helpers.fold_model_inputs and
# n07.fold_similarity_spaces hand over the scaled matrices.
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories,
)
from experiments.dataset_policy import (
    noise_levels, noise_groups_for, noise_dose_units,
)
from experiments.noise.noise_helpers import (
    detection_scores, detection_is_defined, DETECTION_IS_DEFINED,
    sweep_conditions,
)
from experiments.noise.n07_remediation import _predict_for_metric
from kirby.noise_spec import GROUPED_STRATEGIES
from kirby.utils.noise_estimation import get_noise_estimator, smooth_by_scores
# Plan section 2 and 3 (2026-09-25): the D-versus-D' rows, the disk caches,
# the similarity axis and the random / shuffled controls. Read their module
# docstrings before changing anything below that calls them.
from experiments.noise import d_vs_dprime as dvd
from experiments.noise import noise_cache as nc
from experiments.noise import similarity_space as ss
from experiments.noise.controls import (
    controls_enabled, random_twin, shuffled_twin)


# ===========================================================================
# SHARED BY n23-n31, added 2026-09-25 (plan sections 1-3).
#
# WHY IT LIVES IN THIS FILE. The change that added it was limited to the
# files n23-n31, so it could not go into d_vs_dprime.py or similarity_space.py,
# where it belongs. n24, n25, n26, n27, n29, n30 and n31 import it from here.
# TODO(similarity-sweep 2026-09-25): move RowSink and PassCheckpoint into
# d_vs_dprime.py.
#
# 2026-10-05 (chat 5): ensure_dprime_models, swept_similarity,
# model_similarity_spaces, setting_neighbours, cached_base_cell and
# cached_hybrid_cell were deleted from this file. A grep of tests/ that day
# found no importer of any of them outside n23, and n23 no longer called
# them. What the other files still import from here: RowSink, PassCheckpoint,
# fit_score_cached, control_twins, as_cleanup_dict, fit_cleaned,
# CACHEABLE_MODELS, NO_NEIGHBOURS, TODAY_SIMILARITY, cleanup_reads_setting,
# TODAY_NEIGHBOUR_CLEANUPS and correction_targets_cached (n24, n26, n27,
# n29, n30, n31), and _uncertainty_cached, OWN_TARGET, SOURCES and the arm
# helpers (n42).
#
# What is here:
#   RowSink                 the per-fold D-versus-D' rows (d_vs_dprime),
#                           resumed from the keys already on disk
#   PassCheckpoint          the experiment's own accumulators, saved after
#                           every pass, so a killed run resumes where it
#                           stopped instead of from the start
#   fit_score_cached        one downstream fit and its held-out score, through
#                           noise_cache when the model is seeded
#   control_twins           random_twin / shuffled_twin, when
#                           KIRBY_NOISE_CONTROLS=1
# ===========================================================================

# The downstream models whose fit noise_cache may store: they are seeded, so
# a stored prediction is the prediction a refit would give. Taken from
# noise_cache.cached_fit_predict's docstring ("rf, svm, lightgbm and
# xgboost"). dnn is left out: its fit goes through torch, and a stored copy of
# a fit that is not reproducible would freeze one draw of it.
CACHEABLE_MODELS = ('rf', 'svm', 'lightgbm', 'xgboost')

# The similarity_setting value on a row whose method read no neighbours at
# all, and the value on a row computed exactly as the experiment computed it
# before 2026-09-25: neighbours in the model's own matrix, the process-wide
# distance, each consumer's own neighbour count (label_smooth and knn_smooth 9,
# the neighbour detectors and prox_smooth 10; similarity_space docstring).
NO_NEIGHBOURS = 'none'
# 2026-10-05 (chat 5): n23 no longer writes this value; the kNN record cut the
# model_input space it names. It stays because n24 imports it.
TODAY_SIMILARITY = f'{ss.MODEL_INPUT.name}|k=today'


def _nan(v):
    return float('nan') if v is None else float(v)


def rows_path(output_dir, exp_full_name, dataset_name, condition=None):
    """<output>/<exp_full_name>/<dataset>[__condition].rows.jsonl -- beside the
    result file run_all writes (<dataset>[__condition].json)."""
    sfx = f'__{condition}' if condition else ''
    return os.path.join(str(output_dir), exp_full_name,
                        f'{dataset_name}{sfx}.rows.jsonl')


class RowSink:
    """The D-versus-D' rows of one run (plan section 2), written per fold.

    add() builds one row with d_vs_dprime.make_row, so every row carries the
    metric and lower_is_better from get_domain_metric and a signed `gain`.
    flush() appends the rows added since the last flush with
    d_vs_dprime.append_rows (flushed and fsynced), so a killed run keeps every
    fold it finished. A row whose key (d_vs_dprime.KEY_FIELDS) is already on
    disk is not written again, so a resumed run adds only what is missing.

    A rows file written from other data is moved aside, not appended to: the
    file carries a sidecar <rows>.fingerprint.json holding
    _shared.input_fingerprint(data, config), and a mismatch renames both to
    <name>.stale-<unix time>. Without it a row key computed on old labels
    would block the new row with the same key.

    output_dir None (a smoke, a direct caller with nowhere to write) keeps the
    rows in memory, in `self.rows`.
    """

    def __init__(self, output_dir, exp_full_name, dataset_name, *, experiment,
                 metric, condition=None, data=None, config=None):
        self.dataset = str(dataset_name)
        self.experiment = str(experiment)
        self.metric_name = metric[0]
        self.lower_is_better = metric[2]
        self.path = (None if output_dir is None else
                     rows_path(output_dir, exp_full_name, dataset_name,
                               condition))
        self.rows = []
        self._pending = []
        self._pending_keys = set()
        self.n_written = 0
        self.n_already_on_disk = 0
        self.moved_aside = None
        if self.path is not None:
            self._check_fingerprint(data, config)
        self.done = dvd.done_keys(self.path) if self.path else set()

    def _check_fingerprint(self, data, config):
        import json
        from experiments._shared import input_fingerprint, fingerprint_mismatch
        fp = convert_numpy(input_fingerprint(data, config)) if data else {}
        side = self.path + '.fingerprint.json'
        if os.path.exists(self.path):
            why = None
            if os.path.exists(side):
                try:
                    with open(side) as fh:
                        why = fingerprint_mismatch(json.load(fh), fp)
                except (OSError, ValueError) as exc:
                    why = f'its fingerprint could not be read ({exc})'
            elif fp:
                why = 'it carries no fingerprint'
            if why:
                stamp = int(time.time())
                self.moved_aside = f'{self.path}.stale-{stamp}'
                os.replace(self.path, self.moved_aside)
                if os.path.exists(side):
                    os.replace(side, f'{side}.stale-{stamp}')
                print(f"  rows file {self.path}: {why}; moved to "
                      f"{self.moved_aside} and started again")
        if fp:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(side, 'w') as fh:
                json.dump(fp, fh, sort_keys=True)

    def add(self, *, method, method_setting, similarity_setting, model_input,
            model, noise_condition, level, draw, fold, score_D, score_Dprime,
            score_clean):
        row = dvd.make_row(
            dataset=self.dataset, experiment=self.experiment,
            method=str(method), method_setting=str(method_setting),
            similarity_setting=str(similarity_setting),
            model_input=str(model_input), model=str(model),
            noise_condition=noise_condition, level=level, draw=draw,
            fold=int(fold), metric=self.metric_name,
            lower_is_better=self.lower_is_better,
            score_D=_nan(score_D), score_Dprime=_nan(score_Dprime),
            score_clean=_nan(score_clean))
        key = dvd.row_key(row)
        if key in self.done or key in self._pending_keys:
            self.n_already_on_disk += int(key in self.done)
            return
        self._pending.append(row)
        self._pending_keys.add(key)

    def flush(self):
        if not self._pending:
            return 0
        if self.path is not None:
            dvd.append_rows(self.path, self._pending)
        else:
            self.rows.extend(self._pending)
        self.done |= self._pending_keys
        n = len(self._pending)
        self.n_written += n
        self._pending, self._pending_keys = [], set()
        return n

    def summary(self):
        return {'rows_file': self.path,
                'rows_written_this_run': int(self.n_written),
                'rows_already_on_disk_not_rewritten': int(
                    self.n_already_on_disk),
                'rows_file_moved_aside_as_stale': self.moved_aside,
                'row_fields': list(dvd.ROW_FIELDS),
                'what_a_row_is': (
                    'one (method, method setting, similarity setting, model '
                    'input, model, condition, level, draw, fold): the model '
                    "fitted on D and on D', scored on the same held-out fold "
                    "in the dataset's own metric, and the model fitted on the "
                    'clean training labels beside them. gain is positive when '
                    "D' beat D, read with lower_is_better. The similarity "
                    'setting ends in |k=<neighbour count>; k=today is each '
                    "consumer's own count before 2026-09-25, and 'none' means "
                    'the method read no neighbours.')}


class PassCheckpoint:
    """The experiment's accumulators, written after every finished pass.

    An experiment that saved only at the end lost everything on a kill. This
    pickles a dict of the accumulators plus the set of finished pass keys to
    <output>/<exp>/_checkpoints/<dataset>[__condition].pkl, atomically, after
    each pass. On the next run with the SAME key the dict comes back and the
    finished passes are skipped; any other key starts from nothing.

    The key is a hash of: _shared.input_fingerprint(data, config), the run's
    own settings (the caller passes them), noise_cache.process_settings(),
    KIRBY_SIMILARITY_SETTINGS, KIRBY_NOISE_CONTROLS, and the bytes of the
    experiment's own source file, so an edit to the experiment invalidates a
    half-finished run instead of mixing old and new passes. ⛔ An edit to a
    module it imports (n07, noise_helpers, _shared) is NOT in the key; wipe
    the _checkpoints directory after one, the locked rule for every cache.

    output_dir None: nothing is written and load() returns None.
    clear() is called once the result is complete.
    """

    def __init__(self, output_dir, exp_full_name, dataset_name, *, key,
                 source_file, condition=None, data=None, config=None):
        import hashlib
        import json
        self.path = None
        if output_dir is None:
            return
        sfx = f'__{condition}' if condition else ''
        self.path = os.path.join(str(output_dir), exp_full_name,
                                 '_checkpoints', f'{dataset_name}{sfx}.pkl')
        from experiments._shared import input_fingerprint
        with open(source_file, 'rb') as fh:
            src = hashlib.blake2b(fh.read(), digest_size=16).hexdigest()
        full = {'run': key,
                'fingerprint': (convert_numpy(input_fingerprint(data, config))
                                if data else None),
                'process': nc.process_settings(),
                'similarity_env': os.environ.get('KIRBY_SIMILARITY_SETTINGS'),
                'controls': controls_enabled(),
                'source': src}
        blob = json.dumps(full, sort_keys=True, default=repr)
        self.digest = hashlib.blake2b(blob.encode(), digest_size=20).hexdigest()

    def load(self):
        import pickle
        if self.path is None or not os.path.exists(self.path):
            return None
        try:
            with open(self.path, 'rb') as fh:
                got = pickle.load(fh)
        except Exception as exc:
            print(f"  checkpoint {self.path} could not be read ({exc}); "
                  f"starting from nothing")
            return None
        if got.get('digest') != self.digest:
            print(f"  checkpoint {self.path} was written by a different run "
                  f"(data, settings or code changed); starting from nothing")
            return None
        return got.get('state')

    def save(self, state):
        import pickle
        import tempfile
        if self.path is None:
            return
        d = os.path.dirname(self.path)
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, prefix='.tmp_')
        try:
            with os.fdopen(fd, 'wb') as fh:
                pickle.dump({'digest': self.digest, 'state': state}, fh,
                            protocol=pickle.HIGHEST_PROTOCOL)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def clear(self):
        if self.path is not None and os.path.exists(self.path):
            os.unlink(self.path)


def fit_score_cached(dataset_name, model_name, factory, X_tr, y_tr, X_va, y_va,
                     metric_fn, metric_name, sample_weight=None):
    """(held-out score, None) or (None, why) for one downstream fit.

    A model in CACHEABLE_MODELS goes through noise_cache.cached_fit_predict,
    which fits factory() on the same arrays and predicts with the same rule as
    n07._predict_for_metric (noise_cache.predict_mode_for), so a stored value
    is bit-identical to a refit. Any other model is fitted here, every time,
    as before. Used for the fits plan section 3 lists as repeated: the
    no-cleaning fit on D and the clean-label fit."""
    try:
        if model_name in CACHEABLE_MODELS:
            pred = nc.cached_fit_predict(
                dataset_name, model_name, factory, X_tr, y_tr, X_va,
                sample_weight=sample_weight,
                predict_mode=nc.predict_mode_for(metric_name))
        else:
            m = factory()
            if sample_weight is None:
                m.fit(X_tr, y_tr)
            else:
                m.fit(X_tr, y_tr, sample_weight=np.asarray(sample_weight))
            pred = _predict_for_metric(m, X_va, metric_name)
        return float(metric_fn(y_va, pred)), None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def smiles_for(data, idx):
    """The fold's SMILES for tanimoto_ecfp4, or None when the dataset has
    none (similarity_space then reports the space unavailable)."""
    smi = data.get('smiles')
    if smi is None:
        return None
    return [smi[int(i)] for i in np.asarray(idx)]


def as_cleanup_dict(X_tr_s, y_tr, *, keep=None, labels=None, weight=None):
    """A cleanup's output in the shape controls.py and n07 use (X, y,
    sample_weight, keep, label_column), for the experiments whose cleanups
    return something else."""
    y_tr = np.asarray(y_tr)
    if keep is not None:
        keep = np.asarray(keep, dtype=bool)
        return {'X': X_tr_s[keep], 'y': y_tr[keep], 'sample_weight': None,
                'keep': keep, 'label_column': None}
    if weight is not None:
        return {'X': X_tr_s, 'y': y_tr, 'sample_weight': np.asarray(weight),
                'keep': None, 'label_column': None}
    lab = y_tr if labels is None else np.asarray(labels)
    return {'X': X_tr_s, 'y': lab, 'sample_weight': None, 'keep': None,
            'label_column': lab}


def control_twins(cleaned, X_tr_s, y_tr, seed, *, is_clf=False, rerun=None):
    """[(control name, twin dict or None, why None)] for one D', or [] when
    KIRBY_NOISE_CONTROLS is not 1 (controls.controls_enabled).

    random_twin: the same number of molecules touched, chosen at random; with
    `rerun` it is the method's own rule on randomly flagged molecules
    (controls.random_twin). shuffled_twin: the corrections permuted. On a
    CLASS label a permuted label change need not be a class
    (controls.shuffled_twin docstring), so shuffled_twin is refused there and
    random_twin needs `rerun`; the refusal is returned, not hidden.
    """
    if not controls_enabled() or cleaned is None:
        return []
    out = []
    label_move = (cleaned.get('keep') is None
                  and cleaned.get('sample_weight') is None
                  and cleaned.get('label_column') is not None)
    if is_clf and label_move and rerun is None:
        out.append(('random_twin', None,
                    'a permuted class-label change need not be a class; this '
                    'cleanup offers no rerun'))
    else:
        try:
            tw, _n = random_twin(cleaned, X_tr_s, y_tr, seed, rerun=rerun)
            out.append(('random_twin', tw,
                        None if tw is not None else 'the method touched no molecule'))
        except Exception as exc:
            out.append(('random_twin', None, f'{type(exc).__name__}: {exc}'))
    if is_clf and label_move:
        out.append(('shuffled_twin', None,
                    'a permuted class-label change need not be a class'))
    else:
        tw = shuffled_twin(cleaned, X_tr_s, y_tr, seed)
        out.append(('shuffled_twin', tw,
                    None if tw is not None else 'the method touched no molecule'))
    return out


def fit_cleaned(factory, cleaned, X_va, y_va, metric_fn, metric_name):
    """(score, None) or (None, why): one fit on a cleanup dict, passing the
    weight only when there is one -- n07._evaluate_fold's rule."""
    try:
        m = factory()
        if cleaned.get('sample_weight') is None:
            m.fit(cleaned['X'], cleaned['y'])
        else:
            m.fit(cleaned['X'], cleaned['y'],
                  sample_weight=cleaned['sample_weight'])
        return float(metric_fn(y_va, _predict_for_metric(m, X_va,
                                                         metric_name))), None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def cleanup_reads_setting(cleanup, setting):
    """Whether n07.cleaned_training_set reads `setting`'s neighbour list for
    this cleanup: label_smooth and knn_smooth read any setting; prox_smooth
    reads only forest_proximity (n07:722-733, which refuses the rest)."""
    if cleanup in ('label_smooth', 'knn_smooth'):
        return True
    if cleanup == 'prox_smooth':
        return setting.space == ss.FOREST_SPACE
    return False


# The cleanups whose TODAY path reads neighbours in the model's own matrix
# (n07:826-917): label_smooth and knn_smooth through make_nearest_neighbors,
# prox_smooth through its own forest proximity.
TODAY_NEIGHBOUR_CLEANUPS = ('label_smooth', 'knn_smooth', 'prox_smooth')


def correction_targets_cached(dataset_name, X_tr_s, y_tr, groups, want):
    """n07.correction_targets with its two cacheable halves cached.

    'oof' goes through noise_cache.cached_oof_predict with the call n07 makes
    (n07:529: oof_predict(X, y, groups, n_splits=min(REPAIR_OOF_SPLITS, n)),
    seed 0), so it is shared with every experiment that computes the same
    out-of-fold LightGBM target for the same draw (plan section 3). 'gp' and
    'quantile' go through noise_cache.cached under kind 'correction_targets':
    both estimators are seeded (random_state=42), and a result with a failed
    half is returned and not stored. An empty `want` is refused rather than
    handed to correction_targets, which reads an empty want as all three
    (n24's note at the same call).
    """
    from experiments.noise.n07_remediation import (
        correction_targets, REPAIR_OOF_SPLITS)
    want = set(want or ())
    if not want:
        return {}
    out = {}
    if 'oof' in want:
        try:
            out['oof'] = nc.cached_oof_predict(
                dataset_name, X_tr_s, y_tr, groups,
                n_splits=min(REPAIR_OOF_SPLITS, len(y_tr)))
        except Exception as exc:
            print(f"      oof target failed: {type(exc).__name__}: {exc}")
            out['oof'] = None
    rest = sorted(want - {'oof'})
    if rest:
        key = {'X': nc.array_hash(X_tr_s), 'y': nc.array_hash(y_tr),
               'groups': nc.array_hash(groups), 'want': rest}
        out.update(nc.cached(
            dataset_name, 'correction_targets', key,
            lambda: correction_targets(X_tr_s, y_tr, groups=groups,
                                       want=set(rest)),
            store_if=lambda v: all(v.get(k) is not None for k in rest)))
    return out


# Which per-molecule number the cleaning is keyed on. The first two predict a
# spread per molecule. The third does not -- see the docstring. The fourth is
# the neighbour-based score the rest of the suite already uses, and is here to
# be beaten rather than to be tested.
SOURCES = ('hetero_gp', 'ngboost', 'heteroscedastic_nn', 'gaussian_process',
           'activity_cliff_aware')

# Which sources predict a molecule's noise FROM ITS FEATURES ALONE. Every one of
# these is scored out of fold, so it can only rank a held-out molecule when that
# molecule's own features say how noisy it is. A source NOT in here reads the
# molecule's own label as well, which is what lets it find a label that moved
# whatever the condition picked.
#
# ⛔ gaussian_process WAS IN THIS TUPLE AND DOES NOT BELONG IN IT. Read out of
# noise_estimation.py:911-916 on 2026-09-21: its score is `y_std + residuals`,
# where `residuals` is `np.abs(y - y_pred)` against its own out-of-fold
# prediction. Half that number is the molecule's own label compared to what a
# model that did not see it predicted, which is a residual detector and has
# nothing to do with whether the features say the molecule is noisy. This file's
# own docstring already said so -- "the posterior standard deviation PLUS the
# out-of-fold residual" -- and the tuple disagreed with it.
#
# MEASURED, flab_thermo_tresanco, 672 antibodies, base representation esm2_cdr
# at 1280 features, outlier_p10 at level 0.5, draw 0, fold 0 of two, two inner
# folds, 29 of the 336 training molecules moved so a coin flip scores 0.0863
# average precision:
#
#     gaussian_process      0.3016   marked undefined here, and the best of all five
#     activity_cliff_aware  0.2664   marked defined
#     heteroscedastic_nn    0.0878   marked undefined, and at chance as recorded
#
# So the file was writing `sources_defined_under_this_condition['gaussian_process']
# = False` onto the best detector of the five under its own default condition.
LEARNS_NOISE_FROM_FEATURES = ('hetero_gp', 'ngboost', 'heteroscedastic_nn')

# The conditions whose noise is decided by something the features encode. Both
# grouped conditions are: `grouped_wider` gives whole scaffold families the
# wider error, and `grouped_shifted` gives whole scaffold families a common
# OFFSET, `eps_i = sqrt(rho)*tau*b_g(i) + sqrt(1-rho)*tau*e_i` with rho 0.62
# from Bentz et al. 2013 Table 7. The three outlier conditions draw the
# corrupted molecules at RANDOM with no reference to the features
# (NOISE_AUDIT_2026_09_10.md 6.7), so a feature-based source is being asked
# something undefined under them -- it is not that the method failed. The five
# constant-width conditions give every molecule the same width, so nothing can
# rank anything under those.
#
# ⛔ KEYED TO STRUCTURE AND DETECTION-GRADEABLE ARE TWO DIFFERENT THINGS, and
# grouped_shifted is the one condition that separates them. Detection needs
# some molecules corrupted and some not, which is what `DETECTION_IS_DEFINED`
# in noise_helpers lists; under grouped_shifted EVERY training molecule moves,
# so there is no corrupted subset to rank and `detection_scores` returns None.
# Its noise is still predictable from the features, because a molecule's offset
# is its group's offset. So a feature-reading source is DEFINED under it and
# the only thing that can be read off it is whether cleaning wins folds.
# `grouped_shifted` was added here on 2026-09-21: until then it was named in
# four files and run by nothing but n13's sweep-everything mode.
NOISE_KEYED_TO_STRUCTURE = ('grouped_wider', 'grouped_shifted')

# The share of training molecules `drop` removes. n18's, for n18's reason:
# nobody knows the right share here, so it is swept rather than looked up.
DROP_SHARES = (0.05, 0.10, 0.20)

# How far `move` pushes a label toward the model's own out-of-fold prediction.
MOVE_STRENGTHS = (0.25, 0.5, 1.0)

# The two rules, both run. See the docstring: one keeps the order of the
# molecules and drops the scale, the other uses the spread in label units.
MOVE_METHODS = ('proportional', 'confidence')

# How many inner folds each uncertainty estimate is computed over. Every source
# here scores a molecule from a model that did not see it, and the inner split
# is group-aware, so a molecule's own scaffold family never sets its own score.
ESTIMATOR_FOLDS = 5

# The k _uncertainty hands an estimator that takes one (n07's and n18's).
# n42 passes it too. Since 2026-10-05 n23's neighbour sources read the swept
# k (5, 10, 20) instead, in the similarity sweep.
NEIGHBOUR_K = 10

# The condition the injected regime runs under when a caller names one and
# nothing else. outlier_p05 corrupts a SUBSET, which is what makes
# detection_scores defined -- under gaussian and grouped_shifted every training
# molecule moves. 2026-10-05 (chat 5): outlier_p10 until then; the noise grid
# decision (KNN_SEARCH_SPACE_RECORD, "The noise grid", Adelaide 2026-10-05)
# cut outlier_p10 and kept outlier_p05.
DEFAULT_NOISE_CONDITION = 'outlier_p05'

# EVERY CONDITION THIS FILE SWEEPS, and why it is this list and not all
# fourteen. n23 grades a per-molecule uncertainty against the molecules whose
# label actually moved, so it needs a condition where SOME moved and some did
# not -- under a constant-width condition every training label moves, the
# suspect set is every molecule, precision is 1.0 by construction and the
# number says nothing. That is noise_helpers.DETECTION_IS_DEFINED, which is
# also where the three outlier_only_* conditions were added on 2026-09-23.
#
# It is a LIST and the run loops it. Passing a single `noise_condition` still
# works and becomes a one-element list.
#
# 2026-10-05 (chat 5): the list is the decided kinds in which a corrupted
# subset exists (noise_helpers.conditions_for_task with
# detection_is_the_headline=True, computed per dataset in run()): on a
# regression dataset outlier_p05, censoring, shift_only_p10 and
# sparse_dprime_up_p10; on a classification dataset the flip kinds. Until then
# it was all of DETECTION_IS_DEFINED, which holds kinds the noise grid cut
# (outlier_p01/p10, outlier_only_*, grouped_wider, shift_only_p05,
# sparse_dprime_p*, sparse_dprime_up_p02/p05). This tuple is what a caller
# that names nothing on a regression dataset gets; run() recomputes it.
#
# OPEN (chat 5): grouped_wider was the only condition swept here under which
# the three feature-based sources (hetero_gp, ngboost, heteroscedastic_nn) are
# defined AND a corrupted subset exists. The noise grid cut it. grouped_shifted
# is kept by the grid and is keyed to structure too, but every molecule moves
# under it, so it has no detection reading; n23 has never swept it. As it
# stands no swept condition makes those three sources defined. Adelaide's to
# decide: add grouped_shifted, keep grouped_wider for n23 only, or neither.
SWEPT_CONDITIONS = ('outlier_p05', 'censoring', 'shift_only_p10',
                    'sparse_dprime_up_p10')
NOISE_DRAWS = (0, 1, 2)
REGIMES = ('injected', 'as_they_come')

# What this file writes since chat 5 (2026-10-05). It goes into the
# checkpoint key, the rows file name and the condition files' fingerprint, so
# nothing written by the whole-dataset design is read back.
# Changed the same day when the main table stopped reading neighbours in the
# fitted matrix (activity_cliff_aware and the fallback move target moved to
# the similarity sweep), so checkpoints from the first version are not read.
RESULT_LAYOUT = 'fold_aligned_swept_neighbours_2026-10-05'

# Which searches the recipe came from before 2026-10-05. Kept for the record;
# nothing reads it since the per-fold searches replaced the stored ones.
POOL_SUBSET = 'predictive_and_similarity'

# ⛔ ONE SPACE PER FAMILY, NOT ONE RECIPE. Added 2026-09-19.
#
# POOL_SUBSETS names the two families in PREFERENCE order and the pool returns
# the first that has an entry, because their stored scores are in different
# units -- a predictive search's score is the dataset's own metric, a
# similarity search's is the correlation between each held-out molecule's
# activity and what its five nearest neighbours predict. A predictive recipe
# always exists, so until 2026-09-19 this file asked for both families and
# built one.
#
# It matters here, though less than in n12 or n17. One of the five uncertainty
# sources, activity_cliff_aware, scores a molecule against its nearest
# neighbours, and the similarity search chooses features so that exactly that
# distance predicts activity from the five nearest neighbours. The other four
# -- hetero_gp, ngboost, heteroscedastic_nn and gaussian_process -- fit a model
# on the features instead, so for them the space changes what the model is
# given rather than which neighbours are found.
#
# 'hybrid' is still the predictive recipe, so every number reported before
# 2026-09-19 still means what it meant.
#
# 2026-10-05 (chat 5): the four configurations a model is FITTED on are
# noise_helpers.CONFIGURATION_SPACES -- base_1, base_2, hybrid,
# hybrid__general2 -- all from the outer fold's own searches. The similarity
# hybrid and the distance hybrid are neighbour spaces only (handoff 2.3 item
# 1), reached through the similarity sweep. SPACE_FOR_FAMILY below is no
# longer read by this file; a smoke script still names it.
HYBRID_SPACE = 'hybrid'
# base_1 and base_2 are each downstream model's rank-1 and rank-2 single
# representation, ranked by the search on the fold's training rows.
BASE_SPACES = ('base_1', 'base_2')
SPACE_FOR_FAMILY = {
    'predictive': HYBRID_SPACE,
    'similarity': 'hybrid_similarity',
}


def _estimator_kwargs(source, groups_train, is_clf, n_folds, k):
    """Only the constructor arguments this estimator actually takes.

    The two families take different ones and there is no shared signature. The
    model-based sources take `n_folds` and `groups`, and score each molecule
    from a model fitted on the OTHER inner folds. The neighbour-based ones take
    `k` and `metric`, and score a molecule against its neighbours inside the
    same training block with no inner split at all.

    Passing one family's arguments to the other raises, and until 2026-09-17
    this file did exactly that: every neighbour-based source failed at
    construction, wrote an error row, and produced no arm. Read the signature
    instead of guessing.
    """
    import inspect
    cls = get_noise_estimator.__globals__['ESTIMATOR_REGISTRY'][source]
    accepted = set(inspect.signature(cls.__init__).parameters)
    # ⛔ `exclude_groups` ADDED 2026-09-25 (plan section 4). The neighbour
    # family names the scaffold groups `exclude_groups`, not `groups`
    # (n07._compute_noise_scores, n07:421-432), so this filter dropped them and
    # activity_cliff_aware here scored a molecule against neighbours from its
    # own scaffold family, while n07, n24 and n25 excluded them. The same
    # detector gave different scores in different experiments. It is offered
    # under both names now, as n07 does. THIS CHANGES n23's
    # activity_cliff_aware numbers from before 2026-09-25 on every dataset
    # with fold groups.
    #
    # 2026-10-05 (chat 5): `exclude_groups` is no longer passed. Axis 6 of
    # KNN_SEARCH_SPACE_RECORD (Adelaide, 2026-10-05): a molecule's
    # scaffold-mates stay in its neighbour list. `groups` is still passed: the
    # model-based sources use it for their group-aware inner folds, which is
    # not a neighbour list. n42 reaches this through _uncertainty_cached, so
    # its activity_cliff_aware numbers move with it.
    want = {'n_folds': n_folds, 'groups': groups_train, 'k': k,
            'metric': None}
    if is_clf:
        want['is_clf'] = True
    return {p: v for p, v in want.items() if p in accepted}


def _scored_out_of_fold(source):
    """Whether this source scores a molecule from a model that did not see it."""
    import inspect
    cls = get_noise_estimator.__globals__['ESTIMATOR_REGISTRY'][source]
    return 'n_folds' in set(inspect.signature(cls.__init__).parameters)


OWN_TARGET = 'the source own oof prediction'
# target_source for a source with no prediction of its own (2026-10-05, chat
# 5): its move target is the neighbours' label under each kept setting of the
# similarity sweep, so its move arms exist only in the rows file.
NEIGHBOUR_TARGET = ("the neighbours' label under each kept neighbour setting "
                    "(similarity sweep only)")


def _uncertainty_cached(dataset_name, source, X_tr_s, y_tr, groups_train,
                        is_clf, n_folds, neighbour_k=10):
    """_uncertainty through noise_cache (plan section 3: "GP, quantile forest,
    NGBoost -- the same class and arguments in n18, n23, n25 and n32").

    Keyed on the content of X_tr_s, y_tr and the groups, the source, is_clf,
    the inner fold count and k; the process settings (threads, distance) are
    added by noise_cache.cached. Every estimator in SOURCES is seeded
    (random_state=42; the torch ones call torch.manual_seed), so a stored
    result is what a refit gives. A failed estimate (scores None) is returned
    and not stored. dataset_name None bypasses the cache. The as_they_come
    regime and the injected level-0 pass hand over the same labels, so they
    share one entry, which is the repeat plan section 3 names for n23.
    """
    def _compute():
        return _uncertainty(source, X_tr_s, y_tr, groups_train, is_clf,
                            n_folds, neighbour_k=neighbour_k)
    if dataset_name is None:
        return _compute()
    key = {'source': str(source), 'X': nc.array_hash(X_tr_s),
           'y': nc.array_hash(y_tr), 'groups': nc.array_hash(groups_train),
           'is_clf': bool(is_clf), 'n_folds': int(n_folds),
           'k': int(neighbour_k),
           # 2026-10-05 (chat 5): a stored result from before held the
           # unweighted neighbour-mean target; this key never matches one.
           'fallback_target': 'none_2026-10-05'}
    return tuple(nc.cached(dataset_name, 'n23_uncertainty', key, _compute,
                           store_if=lambda v: v[0] is not None))


def _uncertainty(source, X_tr_s, y_tr, groups_train, is_clf, n_folds,
                 neighbour_k=10):
    """One spread per training molecule, and the denoised value to move toward.

    Returns (scores, target, target_source) or (None, None, why).

    `target` is the out-of-fold prediction the `move` shape pushes labels
    toward. The model-based sources produce one themselves and set
    `predictions_`. A source that does not gets target None and
    target_source NEIGHBOUR_TARGET; its move target is built in the
    similarity sweep from the decided neighbour settings (2026-10-05, chat 5;
    until then the plain mean of its ten nearest neighbours in the fitted
    matrix). `target_source` is written onto
    every row, because a reader comparing two sources has to know whether they
    differ in the score alone or in the score and the target.
    """
    try:
        est = get_noise_estimator(
            source, **_estimator_kwargs(source, groups_train, is_clf,
                                        n_folds, neighbour_k))
    except Exception as exc:
        return None, None, f'{type(exc).__name__}: {exc}'

    try:
        scores = np.asarray(est.estimate(X_tr_s, y_tr), dtype=float)
    except Exception as exc:
        return None, None, f'{type(exc).__name__}: {exc}'
    if scores.shape[0] != y_tr.shape[0] or not np.all(np.isfinite(scores)):
        return None, None, 'the estimate was the wrong length or not finite'

    target = getattr(est, 'predictions_', None)
    if target is not None and np.shape(target) == np.shape(y_tr):
        return scores, np.asarray(target, dtype=float), OWN_TARGET

    # No denoised value of its own. 2026-10-05 (chat 5): no target is
    # returned here any more. Until then the target was the plain mean of the
    # molecule's ten nearest neighbours' labels, found in the matrix the model
    # is fitted on by the process-wide distance. The kNN record cut that space
    # (axis 1, model_input) and that counting (axis 5 keeps 'rank' and
    # 'inverse_square'). The move target for such a source is now built only
    # in the similarity sweep (_score_configuration), from the screen's top 3
    # (setting, k, scheme) through similarity_space.weighted_neighbour_label.
    # Those lists leave the molecule's own row out (similarity_space), which
    # is what _drop_self did here for dirty-run bug 7.
    return scores, None, NEIGHBOUR_TARGET


def drop_keep_mask(setting, scores):
    """True for the molecules a `drop@setting` arm keeps. One rule, one place.

    ⛔ THE ARM'S NAME IS A SHARE AND THE COUNT IT REMOVES IS NOT EXACTLY THAT
    SHARE. Measured on flab_thermo_tresanco on 2026-09-21, 448 training
    molecules, onehot at 5600 features, outlier_p10 at level 0.5, fold 0:

      drop@0.05  removes 23 where 5 percent of 448 is 22.4, for four of the
                 five uncertainty sources. `np.percentile` interpolates
                 between the two order statistics that straddle the 95th
                 percentile and `scores < cut` keeps only what is strictly
                 below it, so a value sitting ON the interpolated cut is
                 removed.
      drop@0.1   removes 45 for activity_cliff_aware, gaussian_process,
                 heteroscedastic_nn -- exactly the 45 the name promises -- and
                 47 for ngboost, whose 448 molecules carry only 251 distinct
                 spreads, so molecules TIED at the cut all go.

    Two molecules of 448 changes no conclusion, which is why the rule is left
    as it is rather than swapped for a rank cut: swapping it would move every
    stored n23 number by a hair and settle nothing. What was missing was the
    COUNT, so `run` now records how many molecules each drop arm actually
    removed and how many of them were molecules whose label the injector moved.
    """
    cut = np.percentile(scores, 100.0 * (1.0 - float(setting)))
    return np.asarray(scores) < cut


def cleaned_training_set(shape, setting, X_tr_s, y_tr, scores, target,
                         method=None):
    """What one shape hands the downstream fit. None when it cannot be applied.

    Returns (X, y) with the rows and labels the fit sees.

    `drop` removes the `setting` share of molecules with the widest predicted
    spread. `move` keeps every molecule and shifts its label toward `target`.
    Neither sets a per-row weight, and that is deliberate: a weight cannot be
    read against `drop` on one axis, and `n20` records that nothing in the
    feature selection reads one either.
    """
    if shape == 'none':
        return X_tr_s, y_tr
    if shape == 'drop':
        keep = drop_keep_mask(setting, scores)
        if keep.sum() < 10:
            return None
        return X_tr_s[keep], y_tr[keep]
    if shape == 'move':
        if target is None:
            return None
        return X_tr_s, smooth_by_scores(y_tr, scores, target,
                                        alpha=float(setting), method=method)
    raise ValueError(f"cleaned_training_set: unknown shape {shape!r}")


def _arms(drop_shares, move_strengths, move_methods):
    """Every (arm name, shape, setting, method). 'none' is the score to beat."""
    out = [('none', 'none', None, None)]
    for s in drop_shares:
        out.append((f'drop@{s:g}', 'drop', s, None))
    for m in move_methods:
        for s in move_strengths:
            out.append((f'move_{m}@{s:g}', 'move', s, m))
    return out


def _fit_score(factory, X_tr, y_tr, X_va, y_va, metric_fn, metric_name):
    """One fit, one held-out score against the untouched labels, or (None, why)."""
    try:
        m = factory()
        m.fit(X_tr, y_tr)
        return float(metric_fn(y_va, _predict_for_metric(m, X_va, metric_name))), None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def _levels_and_draws(dataset_name, condition, is_clf, levels=None, draws=None):
    """(level, draw) pairs for the injected regime.

    Level 0.0 is the labels as delivered, so three draws are three identical
    runs and one is kept. 2026-10-05 (chat 5): on a classification set too.
    Until then a classification set ran three draws at level 0, because the
    flip calibrator returned a floor rather than zero there; Adelaide decided
    2026-10-05 that level 0 is the labels as delivered (handoff 2.3 item 10,
    dirty-run bug 6), and noise_spec and noise_helpers.inject_condition now
    return the clean labels at level 0. n42 calls this too.
    """
    levels = (noise_levels(dataset_name, strategy=condition, is_clf=is_clf)
              if levels is None else levels)
    draws = tuple(NOISE_DRAWS if draws is None else draws)
    out = []
    for lvl in levels:
        here = draws if float(lvl) > 0 else draws[:1]
        for d in here:
            out.append((float(lvl), int(d)))
    return out


def run(data, config, output_dir, dataset_name, **kwargs):
    """N23 across every condition where detection is defined.

    The sweep, the per-condition checkpointing, the resume and the stride slice
    all live in noise_helpers.sweep_conditions -- seven experiments need them
    and three copies of one loop is how n17, n18 and n23 each ended up carrying
    the same per-model build defect. See that function for why each condition is
    written as it lands.

    It sweeps the conditions in which a corrupted subset exists, not all of
    them: n23 grades a per-molecule uncertainty against the molecules whose
    label actually moved, and under a constant-width condition every training
    label moves, so the suspect set is every molecule and precision is 1.0 by
    construction.

    2026-10-05 (chat 5): those conditions are now the decided kinds
    (noise_helpers.conditions_for_task(detection_is_the_headline=True); see
    SWEPT_CONDITIONS for the OPEN item about grouped_wider). The condition
    files are fingerprinted with the decided levels, the draws, the outer fold
    count (KIRBY_NOISE_N_FOLDS included), the folds run
    (KIRBY_NOISE_ONLY_FOLDS) and RESULT_LAYOUT, so a file written by the
    whole-dataset design is recomputed, never read back.
    """
    from experiments.noise.noise_helpers import (
        conditions_for_task, decided_levels, noise_n_folds)
    conditions = list(conditions_for_task(data, detection_is_the_headline=True))
    fp_config = dict(config)
    fp_config['n_splits'] = noise_n_folds(config)
    fp_config['noise_levels'] = {
        c: [float(x) for x in decided_levels(data, dataset_name, c)]
        for c in conditions}
    fp_config['noise_seeds'] = list(NOISE_DRAWS)
    fp_config['noise_strategies'] = [
        RESULT_LAYOUT,
        f"only_folds={os.environ.get('KIRBY_NOISE_ONLY_FOLDS') or 'all'}"]
    return sweep_conditions(
        _run_one_condition, 'n23_uncertainty_cleaning',
        data, fp_config, output_dir, dataset_name,
        default_conditions=conditions, kwargs=kwargs,
        groups_needed=GROUPED_STRATEGIES)


# The two of a model's four fitted configurations whose features are chosen
# from the labels. They are rebuilt on every (level, draw) from the labels the
# pass holds; base_1 and base_2 choose no features and are built once per fold.
HYBRID_CONFIGURATIONS = ('hybrid', 'hybrid__general2')


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    """N23: cleaning by a model's own per-molecule uncertainty, ONE condition.

    CHAT 5 ALIGNMENT, 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT section 2.4).
    What each outer fold does, in order:
      1. Folds are noise_helpers.noise_outer_folds: the splits the per-fold
         searches are filed under, the count from config['n_splits'] or
         KIRBY_NOISE_N_FOLDS. Until 2026-10-05 they were _shared.cv_split.
      2. noise_helpers.fold_configurations runs this fold's searches on its
         training rows (clean labels, decided grid, pruning off) and gives each
         of the dataset's two decided models its 2 best base representations,
         its 2 best general hybrids, and the recipes of the neighbour spaces
         (the distance hybrid and the similarity hybrid). Nothing is read from a
         whole-dataset search. Until 2026-10-05 the recipes and the base
         representations came from whole-dataset stored searches, ranked on
         every row, for five models.
      3. Each model is fitted on its 4 configurations (2 models x 4 = 8, R5):
         base_1 and base_2 built once per fold, the two hybrids rebuilt on the
         fold's training rows from the labels the pass holds (damaged under
         injection), through noise_helpers.fold_model_inputs, which passes the
         fold search's own representation scores (dirty-run bug 1).
      4. Every uncertainty source, shape and arm runs in each configuration,
         fitted with that configuration's model only.
      5. The similarity sweep: the neighbour settings of the dataset's group
         (n07.sweep_settings: no model input, no importance weighting,
         distances per group), lists with scaffold-mates kept, every
         (setting, k) graded for detection, and only the screen's top 3
         (setting, k, counting scheme) refitted (n07.screen_settings).
    Each (regime, level, draw, fold) pass is checkpointed as it finishes, and
    its rows go to disk first.
    """
    from experiments.decided_grid import decided_models
    from experiments.noise.noise_helpers import (
        noise_outer_folds, fold_configurations, fold_model_inputs,
        configuration_record, inject_condition, decided_levels,
        dataset_knn_group, level_unit, CONFIGURATION_SPACES)
    from experiments.noise import n07_remediation as n07

    print("\n" + "=" * 70)
    print("EXPERIMENT N23: CAN A PER-MOLECULE UNCERTAINTY CLEAN LABELS")
    print("=" * 70)

    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    groups_arr = None if groups is None else np.asarray(groups)
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)

    condition = kwargs.get('noise_condition') or DEFAULT_NOISE_CONDITION
    regimes = tuple(kwargs.get('regimes') or REGIMES)
    sources = tuple(kwargs.get('sources') or SOURCES)
    drop_shares = tuple(float(x) for x in
                        (kwargs.get('drop_shares') or DROP_SHARES))
    move_strengths = tuple(float(x) for x in
                           (kwargs.get('move_strengths') or MOVE_STRENGTHS))
    move_methods = tuple(kwargs.get('move_methods') or MOVE_METHODS)
    est_folds = int(kwargs.get('estimator_folds', ESTIMATOR_FOLDS))
    # The k handed to an estimator that takes one. Since 2026-10-05 (chat 5)
    # no source that reads neighbours runs in the main table, so in n23 this
    # only enters _uncertainty_cached's key; the swept k (5, 10, 20) is what
    # the neighbour sources use.
    neighbour_k = int(kwargs.get('neighbour_k', NEIGHBOUR_K))

    # R1 (chat 5, 2026-10-05): the dataset's two decided models
    # (DECISIONS.md Step 1), nothing else; knn runs only inside the distance
    # search. Replaces config['models'] (run_all hands every experiment five)
    # and ensure_dprime_models, which added rf and svm to every list. A
    # `models` kwarg (a smoke, a direct caller) can only narrow the decided
    # pair; a name outside it is dropped and written down.
    decided = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    asked = kwargs.get('models')
    if asked:
        models = [m for m in asked if m in decided]
        models_not_decided = [m for m in asked if m not in decided]
    else:
        models = list(decided)
        models_not_decided = []
    if config.get('models'):
        print(f"  config['models'] {list(config['models'])} is not read: "
              f"n23 fits the decided models {decided}")
    if kwargs.get('base_reps') is not None:
        print("  base_reps given by the caller are not read: each fold's base "
              "representations come from that fold's search")
    model_factories = get_model_factories(
        models, task_type=task_type, n_samples=len(labels))
    models_without_factory = [m for m in models if m not in model_factories]
    models = [m for m in models if m in model_factories]
    if not models:
        raise RuntimeError(
            f"n23: no decided model could be built for {dataset_name}. "
            f"Decided {decided}, asked {list(asked or [])}, factories "
            f"{sorted(model_factories)}.")

    noise_groups = noise_groups_for(data)[0]

    # ⛔ A GROUPED CONDITION NEEDS SCAFFOLD GROUPS, AND AN ANTIBODY SET HAS NONE.
    #
    # The two entries of NOISE_KEYED_TO_STRUCTURE are the two conditions this
    # file records as making its three feature-based sources defined.
    # `grouped_wider` gives a three-times wider error to whole scaffold families
    # and `grouped_shifted` gives whole scaffold families a common offset, and
    # `kirby.noise_spec.make_noisy_labels_verbose` refuses to run either without
    # one scaffold-group id per molecule built by `assign_noise_groups(smiles)`.
    # `noise_groups_for` returns None for any dataset that loads no SMILES -- an
    # antibody, nucleotide or expression set -- and says so in its own docstring:
    # the grouped conditions are dropped rather than run over invented groups.
    #
    # Measured 2026-09-21 on flab_thermo_tresanco, 672 antibodies: the injector
    # raises. Without this check every fold raised inside the loop below, was
    # caught, wrote one row into `_errors`, and continued -- so the run finished
    # and wrote a result file with no score in it at all. Say it here instead,
    # where a reader sees it.
    if str(condition) in GROUPED_STRATEGIES and noise_groups is None:
        raise RuntimeError(
            f"n23: condition {condition!r} gives whole scaffold families a wider "
            f"error, and {dataset_name} loads no SMILES, so there are no "
            f"scaffold groups to give it to. noise_groups_for returned None. "
            f"Every fold would raise inside the injector and the run would write "
            f"a file with no score in it. Datasets with SMILES -- caco2, herg_ki "
            f"-- can run this condition; an antibody set cannot. Pick a "
            f"condition whose corrupted molecules are chosen without groups: "
            f"{', '.join(c for c in SWEPT_CONDITIONS if c not in GROUPED_STRATEGIES)}.")

    arms = _arms(drop_shares, move_strengths, move_methods)
    # Dirty-run bug (DIRTY_RUN_RESULTS Part 2, n23: 720 "Unknown label type:
    # continuous" on class labels), fixed 2026-10-05 (chat 5). A `move` arm
    # shifts each label part of the way toward a target, which on a class
    # label gives a number that is not a class, and every classifier refused
    # it. On a classification dataset the move arms are not run and the
    # result file says so; the drop arms and the no-cleaning fit run as
    # before.
    # OPEN (chat 5): the other choice is to round a moved class label to the
    # nearer class (EXPERIMENT_DEBUGGING.md n11 P1 names both). Adelaide's to
    # decide; skipping is the one that invents no new arm.
    if is_clf:
        arms_not_run = {a[0]: ('a moved class label is not a class, so no '
                               'classifier can be fitted on it')
                        for a in arms if a[1] == 'move'}
        arms = [a for a in arms if a[1] != 'move']
    else:
        arms_not_run = {}

    # The dataset's kNN group and the neighbour settings it sweeps
    # (KNN_SEARCH_SPACE_RECORD, Adelaide 2026-10-05).
    knn_grp = dataset_knn_group(data, dataset_name)
    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    n_molecules = len(set(smiles)) if has_smiles else len(labels)
    sim_settings = n07.sweep_settings(dataset_name, n_molecules, has_smiles)

    # The decided levels for this kind on this dataset's group (noise grid
    # decision, Adelaide 2026-10-05). A `levels` kwarg (a smoke) overrides.
    levels = (tuple(float(x) for x in kwargs['levels'])
              if kwargs.get('levels') is not None else
              tuple(float(x) for x in decided_levels(data, dataset_name,
                                                     condition)))

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'result_layout': RESULT_LAYOUT,
        'selection_scope': 'fold_train_rows',
        'regimes': list(regimes),
        'noise_condition': condition,
        'level_unit': level_unit(condition, is_clf),
        'knn_group': knn_grp,
        'dose_units': noise_dose_units(dataset_name),
        'sources': list(sources),
        'shapes': ['none', 'drop'] + ([] if is_clf else ['move']),
        'drop_shares': list(drop_shares),
        'move_strengths': list(move_strengths),
        'move_methods': list(move_methods),
        'arms': [a[0] for a in arms],
        'arms_not_run_and_why': arms_not_run,
        'arm_to_beat': 'none',
        'models': models,
        'models_decided': decided,
        'models_asked_but_not_decided': models_not_decided,
        'models_without_factory': models_without_factory,
        'configurations_per_model': list(CONFIGURATION_SPACES),
        'estimator_folds': est_folds,
        'neighbour_k': neighbour_k,
        'main_table_sources': [s_ for s_ in sources
                               if s_ not in nc.NEIGHBOUR_DETECTORS],
        'main_table_scope': (
            'The folds-won table and the detection block hold the sources '
            'whose score reads no neighbours, with their drop arms, and their '
            'move arms only when the source predicts a target of its own. '
            'activity_cliff_aware, and the move arms of a source with no '
            "target of its own, read neighbours: they are in sim_detection "
            'and the rows file, under the screened neighbour settings '
            '(2026-10-05, kNN record axes 1 and 5).'),
        'claim_scope': (
            'Measures whether a per-molecule uncertainty estimate can clean '
            'labels. Every score is against held-out labels that were never '
            'cleaned and never corrupted. A win says the cleaning helped the '
            'model; under injection the detection block says separately '
            'whether the number found the molecules that actually moved.'),
        'two_terms_note': (
            "A model's uncertainty is how noisy the label looks plus how "
            'unsure the model is about that region. Only the first is a '
            'cleaning signal. Which models tell the two apart was measured in '
            '~/repos/qsar_qm_models/RERUN_PLAN.md sections 2.30 and 2.32, '
            'dated 2026-08-30 and 2026-08-31, on QM9 and not on this data: '
            'Gaussian process separation 11.2 to 18.7 over three seeds; '
            'NGBoost has no model-unsure term at all; variance-head network '
            '1.5 to 2.3; random forest 1.1, recorded as FAIL.'),
        'gaussian_process_is_not_that_number': (
            "KIRBy's gaussian_process estimator returns the posterior standard "
            'deviation plus the out-of-fold residual, which is the '
            'MODEL-UNSURE term, and its label-noise term is one number for the '
            'whole fold. Read out of noise_estimation.py:726-800 on '
            '2026-09-17. It is swept so the difference is measured here.'),
        'features_are_fixed_note': (
            "Each decided model has four configurations per outer fold: its "
            "2 best single representations and its 2 best general hybrids, "
            "all ranked by searches on that fold's training rows "
            "(fold_configurations). The hybrids are rebuilt on the fold's "
            "training molecules at the labels the pass holds, before any "
            "cleaning, and every source, shape and setting shares them. No "
            "cleaning reaches the feature ranking or the budget split. Whether "
            "it should is n20's question."),
        'zero_noise_control_note': (
            'Level 0.0 under injection corrupts nothing, so there is nothing '
            'to clean. A source that still wins folds there is removing or '
            'moving the molecules the model finds hard.'),
        # The two families do not score a molecule the same way, and the
        # difference is not a setting anyone chose -- it is what each estimator
        # is. A model-based source fits on the other inner folds and scores the
        # held-out ones. A neighbour-based source compares a molecule to its
        # neighbours inside the same training block, with no inner split. So two
        # sources can differ in the score AND in whether the molecule's own
        # label helped produce it.
        'scored_out_of_fold_by_source': {
            s: bool(_scored_out_of_fold(s)) for s in sources},
        # Whether each source's number can mean anything under the condition
        # this run used. A source that predicts noise from the features needs a
        # condition that decides which molecules are noisy from something the
        # features encode; the outlier conditions do not.
        'sources_defined_under_this_condition': {
            s: bool(s not in LEARNS_NOISE_FROM_FEATURES
                    or condition in NOISE_KEYED_TO_STRUCTURE)
            for s in sources},
        'condition_keys_noise_to_structure': bool(
            condition in NOISE_KEYED_TO_STRUCTURE),
        'sources_defined_note': (
            'False means this source predicts a molecule\'s noise from its '
            'features and the condition in use picked its corrupted molecules '
            'without reference to the features, so there is nothing in the '
            'features to generalise from and a chance-level reading is the '
            'condition rather than the method. The two grouped conditions are '
            'the ones that do not: grouped_wider gives whole scaffold families '
            'the wider error and grouped_shifted gives whole scaffold families '
            'a common offset. True under grouped_shifted does NOT mean a '
            'detection number exists: every training molecule moves under it, '
            'so there is no corrupted subset to rank and the only reading is '
            'whether cleaning wins folds. NOISE_AUDIT_2026_09_10.md 6.7.'),
        'scored_out_of_fold_note': (
            'True means the molecule was scored by a model fitted on the other '
            'inner folds, group-aware, so its own scaffold family did not set '
            'its own score. False means the score is a comparison against '
            'neighbours inside the same training block.'),
        'per_regime': {},
        'detection': {},
        'target_source_by_source': {},
        'estimator_seconds_by_source': {},
    }

    folds, n_folds = noise_outer_folds(data, config)
    results['n_folds'] = int(n_folds)
    results['folds_run'] = [int(f) for f, _, _ in folds]

    passes = {'as_they_come': [(None, 0)]}
    if 'injected' in regimes:
        passes['injected'] = _levels_and_draws(
            dataset_name, condition, is_clf, levels=levels,
            draws=kwargs.get('draws'))
        results['noise_levels'] = sorted({l for l, _ in passes['injected']})
    for regime in regimes:
        if regime not in passes:
            raise ValueError(f"n23: unknown regime {regime!r}")

    n_passes = sum(len(passes[r]) for r in regimes)
    n_cfg = len(models) * len(CONFIGURATION_SPACES)
    print(f"  {len(labels)} molecules, {n_folds} outer folds "
          f"(running {results['folds_run']}), decided models {models}, "
          f"group {knn_grp}")
    print(f"  condition {condition}, levels {list(levels)} "
          f"({level_unit(condition, is_clf)})")
    print(f"  sources {list(sources)}")
    print(f"  {len(arms)} arms: 1 to beat, {len(drop_shares)} drop, "
          f"{0 if is_clf else len(move_methods) * len(move_strengths)} move")
    print(f"  {n_passes} (level, draw) passes over {len(regimes)} regimes; "
          f"{n_cfg} fitted configurations per fold ({len(models)} models x "
          f"{len(CONFIGURATION_SPACES)}); at most "
          f"{n_passes * len(folds) * len(sources) * n_cfg} uncertainty "
          f"estimates")

    # acc[regime][level][space][source][arm][model][fold_key] = score
    acc = {}
    errors = {}
    detection_acc = {}
    est_seconds = {}
    target_src = {}
    # How many molecules each drop arm REMOVED, and how many of them were
    # molecules whose label the injector moved. Without this the arm is known
    # only by its name: on 2026-09-21 `drop@0.1` with heteroscedastic_nn won
    # every fold while removing 45 of the 448 training molecules, of which 5 of
    # the 42 that moved -- removing 45 of 448 at random takes 4.2 of the 42 by
    # arithmetic. The count is what said the winning arm was not removing the
    # molecules whose labels moved, and it is cheaper than one fit.
    removed_acc = {}
    # The similarity sweep's detection readings:
    # sim_detection[level][space][model][source]['<setting>|k=<k>'] = [reading]
    sim_detection = {}
    # Why a neighbour space or setting could not be built, per (space, model);
    # the first reason seen is kept.
    sim_unavailable = {}
    # Per outer fold: what each model was fitted on (item 11 of handoff 2.4).
    fold_records = {}
    # The screen's tables, one per (fold, pass, model, configuration).
    screen_acc = []
    # Configurations that could not be built, per (level, model, space).
    config_unavailable = {}
    t0 = time.time()

    # ── Rows, checkpoints ──
    # The rows file and the checkpoint carry RESULT_LAYOUT, so rows and
    # passes written by the whole-dataset design (same model_input names,
    # 'base_1', 'hybrid') are never read back as done.
    cond_tag = f'{condition}__{RESULT_LAYOUT}'
    rows = RowSink(output_dir, 'n23_uncertainty_cleaning', dataset_name,
                   experiment='n23', metric=metric,
                   condition=cond_tag, data=data, config=config)
    ckpt = PassCheckpoint(
        output_dir, 'n23_uncertainty_cleaning', dataset_name,
        condition=cond_tag, data=data, config=config,
        source_file=__file__,
        key={'layout': RESULT_LAYOUT, 'regimes': list(regimes),
             'sources': list(sources),
             'drop_shares': list(drop_shares),
             'move_strengths': list(move_strengths),
             'move_methods': list(move_methods), 'est_folds': est_folds,
             'neighbour_k': neighbour_k, 'models': list(models),
             'levels': list(levels), 'draws': kwargs.get('draws'),
             'n_splits': int(n_folds),
             'folds_run': results['folds_run'],
             'settings': [s.name for s in sim_settings],
             'screen_top': n07.screen_top()})
    done_passes = set()
    _state = ckpt.load()
    if _state:
        acc, errors = _state['acc'], _state['errors']
        detection_acc, est_seconds = _state['detection_acc'], _state['est_seconds']
        target_src, removed_acc = _state['target_src'], _state['removed_acc']
        sim_detection = _state['sim_detection']
        sim_unavailable = _state['sim_unavailable']
        fold_records = _state.get('fold_records', {})
        screen_acc = _state.get('screen', [])
        config_unavailable = _state.get('config_unavailable', {})
        done_passes = set(_state['done'])
        print(f"  resumed from {ckpt.path}: {len(done_passes)} (regime, level, "
              f"draw, fold) passes already done")

    # The sources whose SCORE reads a neighbour list (only the neighbour
    # estimators take precomputed_neighbors: noise_cache.NEIGHBOUR_DETECTORS).
    nbr_sources = [s for s in sources if s in nc.NEIGHBOUR_DETECTORS]
    results['similarity_sweep'] = {
        'settings_env': os.environ.get('KIRBY_SIMILARITY_SETTINGS'),
        'knn_group': knn_grp,
        'settings_swept': [s.name for s in sim_settings],
        'ks': list(n07.sweep_ks()),
        'schemes': list(n07.sweep_schemes()),
        'screen_top': n07.screen_top(),
        'screen_scored_on': n07.SCREEN_SCORED_ON,
        'sources_whose_score_reads_neighbours': nbr_sources,
        'what_is_swept': (
            "Nothing in the main table reads neighbours (its rows carry "
            'similarity_setting "none"). activity_cliff_aware, and the move '
            'arms of a source with no prediction of its own, exist only here. '
            "The neighbour settings of this dataset's group "
            '(KNN_SEARCH_SPACE_RECORD, Adelaide 2026-10-05) are built on '
            "each configuration's fold, with a molecule's scaffold-mates kept "
            'in its list. Every (setting, k) grades the neighbour sources '
            '(activity_cliff_aware) for detection, which fits nothing. The '
            'screen then scores every (setting, k, counting scheme) by how '
            "well a molecule's neighbours predict its training label and "
            'keeps the best 3; only those are refitted: the neighbour '
            'sources with all arms, their move target the neighbours\' label '
            'weighted by the scheme; and a source with no prediction of its '
            "own, whose move target is the neighbours' label under the same "
            "kept (setting, k, scheme), its "
            'move arms only. A row\'s similarity_setting is the setting, its '
            'method_setting starts k=<k>|scheme=<scheme>. The sources that '
            'read no neighbours are not swept: their rows carry '
            'similarity_setting "none".'),
    }

    def _save_state():
        ckpt.save({'acc': acc, 'errors': errors,
                   'detection_acc': detection_acc, 'est_seconds': est_seconds,
                   'target_src': target_src, 'removed_acc': removed_acc,
                   'sim_detection': sim_detection,
                   'sim_unavailable': sim_unavailable,
                   'fold_records': fold_records, 'screen': screen_acc,
                   'config_unavailable': config_unavailable,
                   'done': sorted(done_passes)})

    for fold_i, train_idx, val_idx in folds:
        _fold_ids = [f"{r}|l{'none' if l is None else f'{l:g}'}d{d}f{fold_i}"
                     for r in regimes for l, d in passes[r]]
        if (str(fold_i) in fold_records
                and all(p in done_passes for p in _fold_ids)):
            print(f"\n  Fold {fold_i}: every pass is on disk in the checkpoint")
            continue
        print(f"\n  Fold {fold_i} of {n_folds}: {len(train_idx)} training, "
              f"{len(val_idx)} held-out rows")
        try:
            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=train_idx, n_folds=n_folds)
        except Exception as exc:
            why = (f'the per-fold searches failed: {type(exc).__name__}: '
                   f'{exc}')
            print(f"    {why}")
            errors[f'f{fold_i}|fold_configurations'] = why
            fold_records[str(fold_i)] = {'error': why}
            _save_state()
            continue
        y_val = labels[val_idx]
        y_clean_tr = (np.asarray(labels[train_idx]) if is_clf
                      else np.asarray(labels[train_idx], dtype=float))
        g_tr = None if groups_arr is None else groups_arr[train_idx]
        smi_tr = smiles_for(data, train_idx)
        smi_va = smiles_for(data, val_idx)

        # base_1 and base_2 of each model: no features are chosen from the
        # labels, so one matrix per fold serves every pass.
        base_inputs = {}
        frec = {'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)),
                'models': {}}
        for m in models:
            entry = entries.get(m)
            ins, unav, names = fold_model_inputs(
                data, dataset_name, m, entry, train_idx=train_idx,
                val_idx=val_idx, y_train=y_clean_tr,
                factory=model_factories[m], metric=metric, spaces=BASE_SPACES)
            base_inputs[m] = ins
            rec = configuration_record(entry, names)
            rec['base_unavailable'] = {sp: why for sp, why in unav.items()
                                       if sp in BASE_SPACES}
            rec['n_features'] = {sp: int(np.shape(X[0])[1])
                                 for sp, X in ins.items()}
            rec['metric'] = metric_name
            rec['lower_is_better'] = bool(lower_is_better)
            rec['training_row_score_note'] = (
                "hybrids[*].score_mean is the fold search's score of that "
                "hybrid on this fold's training rows (its own inner folds), "
                'in the search metric. The held-out scores of every '
                'configuration are in per_regime, keyed by fold, and '
                'held_out_score_no_cleaning below.')
            rec['held_out_score_no_cleaning'] = {}
            frec['models'][m] = rec
            print(f"    {m}: base {rec.get('base')}, hybrids "
                  f"{[h.get('label') for h in rec.get('hybrids') or []]}")
        # A resumed fold keeps the held-out scores its finished passes wrote.
        _prev = (fold_records.get(str(fold_i)) or {}).get('models') or {}
        for m, rec in frec['models'].items():
            rec['held_out_score_no_cleaning'].update(
                (_prev.get(m) or {}).get('held_out_score_no_cleaning') or {})
        fold_records[str(fold_i)] = frec

        for regime in regimes:
            acc.setdefault(regime, {})
            for level, draw in passes[regime]:
                lvl_key = 'none' if level is None else f'{level:g}'
                fold_key = f'l{lvl_key}d{draw}f{fold_i}'
                pass_id = f'{regime}|{fold_key}'
                if pass_id in done_passes:
                    print(f"    {pass_id}: on disk in the checkpoint")
                    continue
                tag = (regime if level is None
                       else f'{regime}/{condition}/level {level:g}/draw {draw}')
                print(f"\n    --- fold {fold_i}, {tag} ---")
                cell = acc[regime].setdefault(lvl_key, {})
                # What a row calls this pass.
                row_cond = condition if regime == 'injected' else 'as_they_come'
                row_level = None if level is None else float(level)

                cmask = None
                if regime != 'injected':
                    y0 = y_clean_tr
                else:
                    try:
                        y0, cmask, _prov = inject_condition(
                            data, dataset_name, condition, float(level),
                            int(draw), train_idx=train_idx,
                            groups=noise_groups)
                    except Exception as exc:
                        errors[f'{regime}|{lvl_key}|d{draw}|f{fold_i}|inject'] = (
                            f'{type(exc).__name__}: {exc}')
                        continue
                    y0 = np.asarray(y0) if is_clf else np.asarray(y0, dtype=float)
                    cmask = np.asarray(cmask, dtype=bool)

                # Each model's 4 configurations: base_1 and base_2 from the
                # fold, the two hybrids rebuilt from the labels this pass
                # holds, before any cleaning.
                configs = []
                for m in models:
                    entry = entries.get(m)
                    hyb, unav_h, _names = fold_model_inputs(
                        data, dataset_name, m, entry, train_idx=train_idx,
                        val_idx=val_idx, y_train=y0,
                        factory=model_factories[m], metric=metric,
                        spaces=HYBRID_CONFIGURATIONS)
                    for sp in CONFIGURATION_SPACES:
                        src_ = (hyb if sp in HYBRID_CONFIGURATIONS
                                else base_inputs.get(m) or {})
                        pair = src_.get(sp)
                        if pair is None:
                            why = (unav_h.get(sp) if sp in HYBRID_CONFIGURATIONS
                                   else frec['models'][m]['base_unavailable']
                                   .get(sp)) or 'not built'
                            (config_unavailable.setdefault(lvl_key, {})
                             .setdefault(m, {}).setdefault(sp, why))
                            continue
                        configs.append((m, entry, sp, pair[0], pair[1]))

                for m, entry, space, X_tr_s, X_va_s in configs:
                    _score_configuration(
                        m=m, entry=entry, space=space, X_tr_s=X_tr_s,
                        X_va_s=X_va_s, y0=y0, y_clean_tr=y_clean_tr,
                        y_val=y_val, cmask=cmask, g_tr=g_tr,
                        train_idx=train_idx, val_idx=val_idx,
                        smi_tr=smi_tr, smi_va=smi_va, regime=regime,
                        level=level, lvl_key=lvl_key, draw=draw,
                        fold_i=fold_i, fold_key=fold_key, row_cond=row_cond,
                        row_level=row_level, cell=cell, frec=frec,
                        ctx=dict(
                            dataset_name=dataset_name, data=data,
                            condition=condition, is_clf=is_clf,
                            task_type=task_type, metric=metric,
                            sources=sources, arms=arms, est_folds=est_folds,
                            neighbour_k=neighbour_k,
                            model_factories=model_factories,
                            sim_settings=sim_settings, n07=n07, rows=rows,
                            errors=errors, est_seconds=est_seconds,
                            target_src=target_src,
                            detection_acc=detection_acc,
                            removed_acc=removed_acc,
                            sim_detection=sim_detection,
                            sim_unavailable=sim_unavailable,
                            screen_acc=screen_acc))

                # The pass is finished: its rows go to disk and the
                # accumulators to the checkpoint, in that order, so a kill
                # between the two re-runs the pass and adds no duplicate row.
                rows.flush()
                done_passes.add(pass_id)
                _save_state()
                print(f"    fold {fold_i}, {tag} done [{time.time() - t0:.0f}s]")

    results['per_regime'] = convert_numpy(acc)
    results['fold_configurations'] = convert_numpy(fold_records)
    results['configurations_unavailable'] = convert_numpy(config_unavailable)
    # {level: {space: {downstream model: {source: reading}}}}. In a hybrid
    # space each model's uncertainty number was computed in ITS OWN features,
    # so one reading per space would average two different feature spaces.
    results['detection'] = convert_numpy({
        lvl: {sp: {m: {src: _mean_detection(rows_)
                       for src, rows_ in by_src.items()}
                   for m, by_src in by_model.items()}
              for sp, by_model in by_sp.items()}
        for lvl, by_sp in detection_acc.items()})
    results['detection_note'] = (
        'One reading per (noise level, configuration, decided model, '
        'uncertainty source), averaged over the folds and draws of that level. '
        "Each model's configurations are its own (its 2 single "
        'representations and 2 hybrids from the fold search), so two models '
        'that share a single representation give the same reading there. '
        'Absent under a classification flip kind: '
        'noise_helpers.DETECTION_IS_DEFINED does not list them, so '
        'detection_scores returns None.')
    results['target_source_by_source'] = {k: sorted(v)
                                          for k, v in target_src.items()}
    # Summed over the folds of one (level, space, source, arm), so every count
    # below is molecules, not a share and not a per-fold average.
    results['rows_removed_by_drop_arm'] = convert_numpy({
        lvl: {sp: {m: {src: {arm: {
            'n_folds': len(rr),
            'molecules_removed_summed_over_the_folds':
                int(sum(r['n_removed'] for r in rr)),
            'training_molecules_summed_over_the_folds':
                int(sum(r['n_training'] for r in rr)),
            'moved_molecules_removed_summed_over_the_folds':
                (None if any(r['n_moved_among_the_removed'] is None for r in rr)
                 else int(sum(r['n_moved_among_the_removed'] for r in rr))),
            'moved_molecules_summed_over_the_folds':
                (None if any(r['n_moved'] is None for r in rr)
                 else int(sum(r['n_moved'] for r in rr))),
        } for arm, rr in by_arm.items()}
            for src, by_arm in by_src.items()}
            for m, by_src in by_model.items()}
        for sp, by_model in by_sp.items()}
        for lvl, by_sp in removed_acc.items()})
    results['rows_removed_note'] = (
        'One entry per (configuration, decided model, source, drop arm). '
        'Every number is a count of molecules summed over the outer folds and '
        'draws of that level, so the share the arm removed is '
        'molecules_removed divided by training_molecules. What the ranking did '
        'is moved_molecules_removed against the count chance would take, which '
        'is molecules_removed times moved_molecules divided by '
        'training_molecules. An arm whose two are equal ranked no better than '
        'removing that many molecules at random.')
    results['estimator_seconds_by_source'] = {
        k: {'mean': float(np.mean(v)), 'n_estimates': len(v),
            'total': float(np.sum(v))}
        for k, v in est_seconds.items()}
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy(
        _summarise(acc, [a[0] for a in arms], models, metric_name,
                   lower_is_better, results['detection']))
    # The similarity sweep's detection readings, averaged over the folds and
    # draws of a level as `detection` is, the spaces and settings that could
    # not be built with the reason, and the screen's tables.
    results['similarity_sweep']['detection'] = convert_numpy({
        lvl: {sp: {m: {src: {lab: _mean_detection(rs)
                             for lab, rs in by_lab.items()}
                       for src, by_lab in by_src.items()}
                   for m, by_src in by_m.items()}
              for sp, by_m in by_sp.items()}
        for lvl, by_sp in sim_detection.items()})
    results['similarity_sweep']['spaces_unavailable'] = sim_unavailable
    results['screen'] = convert_numpy(screen_acc)
    results['rows'] = rows.summary()
    results['controls_run'] = bool(controls_enabled())
    _print_summary(results, metric_name)
    print(f"  D-versus-D' rows: {rows.n_written} written this run to "
          f"{rows.path or 'memory'}")
    # The result is complete; the per-pass checkpoint is not needed again.
    ckpt.clear()
    return results


def _score_configuration(*, m, entry, space, X_tr_s, X_va_s, y0, y_clean_tr,
                         y_val, cmask, g_tr, train_idx, val_idx, smi_tr,
                         smi_va, regime, level, lvl_key, draw, fold_i,
                         fold_key, row_cond, row_level, cell, frec, ctx):
    """Every source, shape and arm in ONE configuration (one decided model,
    one of its four feature matrices) on one (regime, level, draw, fold)
    pass, then the similarity sweep for it. Writes into the accumulators in
    `ctx` and `cell`; returns nothing.

    Split out of _run_one_condition on 2026-10-05 (chat 5) when the loop
    became per decided model and per configuration. The body below is the
    loop body that was there, with one model per matrix instead of the
    models sharing a single representation.
    """
    ds = ctx['dataset_name']
    data = ctx['data']
    condition = ctx['condition']
    is_clf = ctx['is_clf']
    metric_name, metric_fn, lower_is_better = ctx['metric']
    arms = ctx['arms']
    errors = ctx['errors']
    rows = ctx['rows']
    n07 = ctx['n07']
    factory = ctx['model_factories'][m]
    space_cell = cell.setdefault(space, {})
    _ekey = f'{space}/{m}'

    # score_clean: the model fitted on the clean training labels (plan
    # section 2). In the as_they_come regime the clean labels are D, so the D
    # score below is used instead of a second identical fit.
    clean_score = None
    if regime == 'injected':
        clean_score, _e = fit_score_cached(
            ds, m, factory, X_tr_s, y_clean_tr, X_va_s, y_val, metric_fn,
            metric_name)
    # The no-cleaning score on this matrix: score_D of the sweep's rows.
    # 2026-10-05 (chat 5): fitted here once, before the sources, so it exists
    # even when every source in the main table is skipped or fails. The 'none'
    # arm below is the same call and reads it back from noise_cache.
    d_score, _e = fit_score_cached(ds, m, factory, X_tr_s, y0, X_va_s, y_val,
                                   metric_fn, metric_name)
    if d_score is not None and regime == 'as_they_come':
        frec['models'][m]['held_out_score_no_cleaning'][space] = d_score
    # {source: (scores, target, target source)} from the main table, reused
    # by the sweep for a source whose score reads no neighbours.
    today_by_source = {}

    for source in ctx['sources']:
        # 2026-10-05 (chat 5), kNN record axis 1: a source whose score reads
        # neighbours (activity_cliff_aware) is no longer scored in the main
        # table. There its neighbours were found in the matrix the model is
        # fitted on, by the process-wide distance, at k=10 -- the model_input
        # space, which the record cut. It is scored, graded for detection and
        # refitted only in the similarity sweep below, under the dataset's
        # decided settings.
        if source in nc.NEIGHBOUR_DETECTORS:
            today_by_source[source] = (None, None, NEIGHBOUR_TARGET)
            ctx['target_src'].setdefault(source, set()).add(NEIGHBOUR_TARGET)
            continue
        t_est = time.time()
        scores, target, tsrc = _uncertainty_cached(
            ds, source, X_tr_s, y0, g_tr, is_clf, ctx['est_folds'],
            neighbour_k=ctx['neighbour_k'])
        ctx['est_seconds'].setdefault(source, []).append(time.time() - t_est)
        if scores is None:
            errors[f'{regime}|{lvl_key}|{_ekey}|{source}|f{fold_i}|estimate'] = tsrc
            print(f"      {_ekey}/{source}: {tsrc}")
            continue
        ctx['target_src'].setdefault(source, set()).add(tsrc)
        today_by_source[source] = (scores, target, tsrc)

        # Did the number find the molecules that really moved, before any
        # cleaning ran? It costs no model fit.
        if cmask is not None and detection_is_defined(condition):
            det = detection_scores(cmask, scores, condition)
            if det is not None:
                (ctx['detection_acc'].setdefault(lvl_key, {})
                 .setdefault(space, {}).setdefault(m, {})
                 .setdefault(source, []).append(det))

        src_cell = space_cell.setdefault(source, {})
        for arm, shape, setting, method in arms:
            if shape == 'move' and tsrc != OWN_TARGET:
                # No prediction of its own: its move target is the neighbours'
                # label, built only in the similarity sweep (2026-10-05).
                continue
            try:
                got = cleaned_training_set(shape, setting, X_tr_s, y0, scores,
                                           target, method=method)
            except Exception as exc:
                errors[f'{regime}|{lvl_key}|{_ekey}|{source}|{arm}|f{fold_i}|'
                       f'clean'] = f'{type(exc).__name__}: {exc}'
                continue
            if got is None:
                errors[f'{regime}|{lvl_key}|{_ekey}|{source}|{arm}|f{fold_i}|'
                       f'clean'] = 'the shape could not be applied on this fold'
                continue
            Xc, yc = got
            if shape == 'drop':
                rm = ~drop_keep_mask(setting, scores)
                (ctx['removed_acc'].setdefault(lvl_key, {})
                 .setdefault(space, {}).setdefault(m, {})
                 .setdefault(source, {}).setdefault(arm, []).append({
                     'n_removed': int(rm.sum()),
                     'n_training': int(len(scores)),
                     'n_moved_among_the_removed': (
                         None if cmask is None else
                         int(np.sum(np.asarray(cmask, dtype=bool) & rm))),
                     'n_moved': (None if cmask is None
                                 else int(np.sum(cmask)))}))
            if shape == 'none':
                # The no-cleaning fit is the same fit for every source;
                # through noise_cache for a seeded model.
                s, err = fit_score_cached(ds, m, factory, Xc, yc, X_va_s,
                                          y_val, metric_fn, metric_name)
            else:
                s, err = _fit_score(factory, Xc, yc, X_va_s, y_val, metric_fn,
                                    metric_name)
            if s is None:
                errors[f'{regime}|{lvl_key}|{_ekey}|{source}|{arm}|{m}|'
                       f'f{fold_i}'] = err
                continue
            src_cell.setdefault(arm, {}).setdefault(m, {})[fold_key] = s

        # ── The D-versus-D' rows of the main table ──
        # 2026-10-05 (chat 5): nothing in the main table reads neighbours any
        # more, so every row here carries similarity_setting 'none'. Until
        # then a neighbour-reading row carried TODAY_SIMILARITY
        # ('model_input|none|process|k=today'), a setting the kNN record cut.
        sD = (src_cell.get('none') or {}).get(m, {}).get(fold_key)
        for arm, shape, setting, method in arms:
            if shape == 'none':
                continue
            if shape == 'move' and tsrc != OWN_TARGET:
                continue
            sP = (src_cell.get(arm) or {}).get(m, {}).get(fold_key)
            if sD is None and sP is None:
                continue
            rows.add(method=source, method_setting=arm,
                     similarity_setting=NO_NEIGHBOURS,
                     model_input=space, model=m, noise_condition=row_cond,
                     level=row_level, draw=draw, fold=fold_i, score_D=sD,
                     score_Dprime=sP,
                     score_clean=sD if clean_score is None else clean_score)

        # ── Controls (KIRBY_NOISE_CONTROLS=1 only) ──
        if controls_enabled():
            for arm, shape, setting, method in arms:
                if shape == 'none':
                    continue
                if shape == 'move' and tsrc != OWN_TARGET:
                    continue
                try:
                    if shape == 'drop':
                        _cd = as_cleanup_dict(
                            X_tr_s, y0, keep=drop_keep_mask(setting, scores))
                    else:
                        _g = cleaned_training_set(shape, setting, X_tr_s, y0,
                                                  scores, target, method=method)
                        if _g is None:
                            continue
                        _cd = as_cleanup_dict(X_tr_s, y0, labels=_g[1])
                except Exception:
                    continue
                _seed = (int(draw), int(fold_i),
                         int(zlib.crc32(f'{source}|{arm}|{space}|{m}'.encode())
                             & 0x7FFFFFFF))
                for cname, tw, why in control_twins(_cd, X_tr_s, y0, _seed,
                                                    is_clf=is_clf):
                    if tw is None:
                        errors[f'{regime}|{lvl_key}|{_ekey}|{source}|{arm}|'
                               f'{cname}|f{fold_i}'] = why
                        continue
                    sPc, _e = fit_cleaned(factory, tw, X_va_s, y_val,
                                          metric_fn, metric_name)
                    rows.add(method=source,
                             method_setting=f'{arm}|control={cname}',
                             similarity_setting=NO_NEIGHBOURS,
                             model_input=space, model=m,
                             noise_condition=row_cond, level=row_level,
                             draw=draw, fold=fold_i, score_D=sD,
                             score_Dprime=sPc,
                             score_clean=sD if clean_score is None
                             else clean_score)

    # ── The similarity sweep (KNN_SEARCH_SPACE_RECORD, Adelaide 2026-10-05) ──
    # Run for the sources whose score reads neighbours, and for a source with
    # no prediction of its own, whose move target is the neighbours' label.
    # Since 2026-10-05 (chat 5) this is the ONLY place either is computed:
    # the main table no longer reads neighbours in the fitted matrix. The
    # model is still
    # fitted on THIS configuration's matrix: the neighbour space and the
    # fitted configuration are separate axes.
    sim_settings = ctx['sim_settings']
    sweep_sources = [
        s_ for s_ in ctx['sources'] if s_ in today_by_source and (
            s_ in nc.NEIGHBOUR_DETECTORS
            or (today_by_source[s_][2] != OWN_TARGET and not is_clf))]
    if not (sim_settings and sweep_sources):
        return
    # The neighbour spaces: this model's base_1/base_2 (and their augmented
    # copies where the group keeps them), its hybrids, the similarity hybrid
    # and the distance hybrid from this fold's searches (entry['recipes']),
    # Tanimoto bits, forest leaves -- whatever the group's settings name.
    # Built from the labels the pass holds.
    sp_tr, sp_va, unav = n07.fold_similarity_spaces(
        ds, sim_settings, model_input_tr=X_tr_s, model_input_va=X_va_s,
        reps=data['reps'], train_idx=train_idx, val_idx=val_idx, y_train=y0,
        groups_train=g_tr, base_reps=(entry or {}).get('base') or [],
        recipes=(entry or {}).get('recipes') or {},
        augmentations=data.get('augmentations'), model_name=m,
        model_factory=ctx['model_factories'][m], metric=ctx['metric'],
        task_type=ctx['task_type'])
    for sp_u, w in unav.items():
        ctx['sim_unavailable'].setdefault(sp_u, {}).setdefault(m, w)
    # Axis 6: one list per setting, scaffold-mates kept (no exclude_groups).
    nbrs = n07.fold_neighbours(ds, sim_settings, sp_tr, sp_va, y0,
                               smiles_train=smi_tr, smiles_val=smi_va,
                               is_clf=is_clf)
    for s in sim_settings:
        L = nbrs.get(s.name)
        if L is not None and not L.get('available'):
            ctx['sim_unavailable'].setdefault(s.name, {}).setdefault(
                m, L.get('reason'))
    nb_sources = [s_ for s_ in sweep_sources if s_ in nc.NEIGHBOUR_DETECTORS]

    # Detection grading fits no model, so every (setting, k) is graded.
    scored = {}

    def _nb_scores(source, s, kk):
        key = (source, s.name, int(kk))
        if key not in scored:
            try:
                sc = nc.cached_detector_scores(
                    ds, X_tr_s, y0, is_clf=is_clf, detector=source,
                    groups=g_tr, similarity_neighbours=nbrs[s.name],
                    similarity_k=int(kk))
            except Exception as exc:
                errors[f'{regime}|{lvl_key}|{_ekey}|{source}|{s.name}|k={kk}|'
                       f'f{fold_i}|estimate'] = f'{type(exc).__name__}: {exc}'
                sc = None
            scored[key] = None if sc is None else np.asarray(sc, dtype=float)
        return scored[key]

    if cmask is not None and detection_is_defined(condition) and nb_sources:
        for s in sim_settings:
            L = nbrs.get(s.name)
            for kk in n07.usable_ks(L):
                for source in nb_sources:
                    sc = _nb_scores(source, s, kk)
                    if sc is None:
                        continue
                    det = detection_scores(cmask, sc, condition)
                    if det is not None:
                        (ctx['sim_detection'].setdefault(lvl_key, {})
                         .setdefault(space, {}).setdefault(m, {})
                         .setdefault(source, {})
                         .setdefault(f'{s.name}|k={int(kk)}', [])
                         .append(det))

    # The per-dataset screen: refit only the best 3 (setting, k, scheme).
    kept, table = n07.screen_settings(sim_settings, nbrs, y0,
                                      metric=ctx['metric'], is_clf=is_clf,
                                      y_val=y_val)
    ctx['screen_acc'].append({
        'model': m, 'model_input': space, 'fold': int(fold_i),
        'regime': regime, 'level': row_level, 'draw': int(draw),
        'condition': row_cond,
        'kept': [f'{s_.name}|k={k_}|{sch}' for s_, k_, sch in kept],
        'table': table})

    for s, kk, scheme in kept:
        try:
            _idx, _dist = ss.neighbours_at(nbrs[s.name], kk)
            # The move target: the neighbours' label weighted by the counting
            # scheme (axis 5). Not used on class labels, where no move arm
            # runs.
            s_target = (None if is_clf else ss.weighted_neighbour_label(
                y0, _idx, _dist, scheme, is_clf=False))
        except Exception as exc:
            errors[f'{regime}|{lvl_key}|{_ekey}|{s.name}|k={kk}|{scheme}|'
                   f'f{fold_i}|target'] = f'{type(exc).__name__}: {exc}'
            continue
        ms_prefix = f'k={int(kk)}|scheme={scheme}'
        for source in sweep_sources:
            t_scores = today_by_source[source][0]
            if source in nc.NEIGHBOUR_DETECTORS:
                # The detector reads the neighbours but not the counting
                # scheme, so its score is the same under both schemes.
                s_scores = _nb_scores(source, s, kk)
                if s_scores is None:
                    continue
                arms_here = [a for a in arms if a[1] != 'none']
            else:
                # Its score reads no neighbours; only its fallback move
                # target did.
                s_scores = t_scores
                arms_here = [a for a in arms if a[1] == 'move']
            for arm, shape, setting, method in arms_here:
                try:
                    got = cleaned_training_set(shape, setting, X_tr_s, y0,
                                               s_scores, s_target,
                                               method=method)
                except Exception as exc:
                    errors[f'{regime}|{lvl_key}|{_ekey}|{source}|{s.name}|'
                           f'{ms_prefix}|{arm}|f{fold_i}|clean'] = (
                        f'{type(exc).__name__}: {exc}')
                    continue
                if got is None:
                    continue
                sP, err = _fit_score(factory, got[0], got[1], X_va_s, y_val,
                                     metric_fn, metric_name)
                if sP is None:
                    errors[f'{regime}|{lvl_key}|{_ekey}|{source}|{s.name}|'
                           f'{ms_prefix}|{arm}|f{fold_i}'] = err
                rows.add(method=source, method_setting=f'{ms_prefix}|{arm}',
                         similarity_setting=s.name, model_input=space,
                         model=m, noise_condition=row_cond, level=row_level,
                         draw=draw, fold=fold_i, score_D=d_score,
                         score_Dprime=sP,
                         score_clean=(d_score if clean_score is None
                                      else clean_score))
                if not controls_enabled():
                    continue
                _cd = (as_cleanup_dict(X_tr_s, y0,
                                       keep=drop_keep_mask(setting, s_scores))
                       if shape == 'drop' else
                       as_cleanup_dict(X_tr_s, y0, labels=got[1]))
                _seed = (int(draw), int(fold_i), int(
                    zlib.crc32(f'{source}|{arm}|{space}|{m}|{s.name}|'
                               f'{ms_prefix}'.encode()) & 0x7FFFFFFF))
                for cname, tw, why in control_twins(_cd, X_tr_s, y0, _seed,
                                                    is_clf=is_clf):
                    if tw is None:
                        continue
                    sPc, _e = fit_cleaned(factory, tw, X_va_s, y_val,
                                          metric_fn, metric_name)
                    rows.add(method=source,
                             method_setting=f'{ms_prefix}|{arm}|control={cname}',
                             similarity_setting=s.name, model_input=space,
                             model=m, noise_condition=row_cond,
                             level=row_level, draw=draw, fold=fold_i,
                             score_D=d_score, score_Dprime=sPc,
                             score_clean=(d_score if clean_score is None
                                          else clean_score))


def _mean_detection(rows):
    """Average the detection readings over the folds and draws."""
    if not rows:
        return None
    keys = ('detection_average_precision', 'detection_precision_at_k',
            'detection_recall_at_k', 'detection_n_corrupted',
            'detection_n_training')
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows
                if r.get(k) is not None and np.isfinite(float(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float('nan')
    out['detection_condition'] = rows[0].get('detection_condition')
    out['detection_n_passes'] = len(rows)
    return out


def _outer_fold_of(fold_key):
    """The outer fold number of a fold key 'l<level>d<draw>f<fold>'."""
    return int(str(fold_key).rsplit('f', 1)[1])


def _paired(base, got, lower_is_better):
    """One arm against `none`, on the (draw, fold) passes both scored.
    Positive means better.

    2026-10-05 (chat 5): the spread is now over OUTER FOLDS
    (noise_helpers.fold_spread): each fold's gain is the mean over its draws,
    and the spread is the standard deviation of those, or None with 'one
    fold, no spread' when one fold ran. Until then it was the standard
    deviation over every (draw, fold) pass, which is 0 on one pass, so a
    single positive gain was written as clearing its own spread.
    n_folds_paired, folds_won, folds_lost and folds_tied still count
    (draw, fold) passes, as before.
    """
    from experiments.noise.noise_helpers import fold_spread
    shared = sorted(set(base) & set(got))
    pairs = [(k, (base[k] - got[k]) if lower_is_better else (got[k] - base[k]))
             for k in shared]
    pairs = [(k, float(g)) for k, g in pairs if np.isfinite(g)]
    gains = np.array([g for _k, g in pairs], dtype=float)
    scale = max(abs(float(np.mean(list(base.values())))) if base else 1.0, 1.0)
    tol = 1e-9 * scale
    by_fold = {}
    for k, g in pairs:
        by_fold.setdefault(_outer_fold_of(k), []).append(g)
    fs = fold_spread([float(np.mean(v)) for _f, v in sorted(by_fold.items())])
    return {
        'n_folds_paired': int(gains.size),
        'n_outer_folds': len(by_fold),
        'folds_won': int((gains > tol).sum()),
        'folds_lost': int((gains < -tol).sum()),
        # A large tied count means the arm is the uncleaned fit under a
        # different name -- a move at a strength that shifts nothing, say.
        'folds_tied': int((np.abs(gains) <= tol).sum()),
        'tie_tolerance': float(tol),
        'mean_gain': float(gains.mean()) if gains.size else float('nan'),
        'gain_by_outer_fold': fs,
        'spread_over_fold_gains': fs.get('sd'),
        'fold_gains': [float(x) for x in gains],
        # A gain no larger than the spread of the per-fold gains behind it is
        # not a gain. Two measured numbers compared, not a test: the suite's
        # standing rule at this replicate count is raw spread only. None when
        # one outer fold ran: there is no spread to compare with.
        'gain_exceeds_its_own_spread': (
            None if fs.get('sd') is None
            else bool(fs['mean'] > fs['sd'])),
    }


def _summarise(acc, arm_names, models, metric_name, lower_is_better, detection):
    """Every arm against `none`, paired per fold, plus what the zero-noise row says.

    THE HEADLINE IS A COUNT OF FOLDS, for n18's reason at :33-37. A mean over
    folds hides an arm that wins big on one fold and loses on the rest.

    Two things are rolled up beside it.

      settings_winning_most_folds  how many (arm, model) settings a source won a
                                   majority of folds at. A source that only wins
                                   at one setting has shown less than one that
                                   wins at several.
      zero_noise_wins              the same count at level 0.0, where nothing
                                   was corrupted. Anything above zero there is
                                   the cleaning removing or moving the molecules
                                   the model finds hard, so a source's real
                                   count has to be read against this one.
    """
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'headline': 'folds won against no cleaning, paired per fold',
           'per_regime': {}}

    for regime, by_level in acc.items():
        reg = {'per_level': {}}
        for lvl, by_space in by_level.items():
            lrow = {'per_space': {}}
            for space, by_source in by_space.items():
                srow = {'per_source': {}}
                for source, by_arm in by_source.items():
                    base_by_model = by_arm.get('none') or {}
                    rows, per_arm = [], {}
                    for arm in arm_names:
                        if arm == 'none':
                            continue
                        per_model = {}
                        for m in models:
                            b = base_by_model.get(m) or {}
                            g = (by_arm.get(arm) or {}).get(m) or {}
                            if not b or not g:
                                continue
                            cellv = _paired(b, g, lower_is_better)
                            per_model[m] = cellv
                            rows.append((arm, m, cellv))
                        if per_model:
                            per_arm[arm] = {'per_model': per_model}
                    majority = [r for r in rows
                                if r[2]['folds_won'] * 2 > r[2]['n_folds_paired']]
                    swept = [r for r in rows
                             if r[2]['n_folds_paired'] > 0
                             and r[2]['folds_won'] == r[2]['n_folds_paired']]
                    clears = [r for r in rows
                              if r[2]['gain_exceeds_its_own_spread']]
                    best = (max(rows, key=lambda r: (
                        r[2]['folds_won'] / max(r[2]['n_folds_paired'], 1),
                        r[2]['mean_gain'])) if rows else None)
                    srow['per_source'][source] = {
                        'per_arm': per_arm,
                        'uncleaned_mean_by_model': {
                            m: (float(np.mean(list(v.values()))) if v
                                else float('nan'))
                            for m, v in base_by_model.items()},
                        'settings_scored': len(rows),
                        'settings_winning_most_folds': len(majority),
                        'settings_winning_every_fold': len(swept),
                        'settings_whose_gain_clears_its_own_spread': len(clears),
                        'best_setting': (None if best is None else {
                            'arm': best[0], 'model': best[1],
                            'folds_won': best[2]['folds_won'],
                            'n_folds_paired': best[2]['n_folds_paired'],
                            'mean_gain': best[2]['mean_gain'],
                            'spread_over_fold_gains':
                                best[2]['spread_over_fold_gains']}),
                        # How well this number found the molecules that moved,
                        # before any cleaning. Absent where nothing was
                        # corrupted or where every molecule moved.
                        # One reading per downstream model: in a hybrid
                        # space the uncertainty number was computed in that
                        # model's own features.
                        'detection_by_model': {
                            m: (detection.get(lvl, {}).get(space, {})
                                .get(m, {}).get(source))
                            for m in models
                            if (detection.get(lvl, {}).get(space, {})
                                .get(m, {}).get(source)) is not None},
                    }
                lrow['per_space'][space] = srow
            reg['per_level'][lvl] = lrow
        out['per_regime'][regime] = reg

    # The zero-noise control, rolled up per source so it sits beside the counts.
    zero = (out['per_regime'].get('injected', {}).get('per_level', {})
            .get('0', {}).get('per_space', {}))
    out['zero_noise_control'] = {
        'what_it_says': (
            'Level 0.0 corrupts nothing, so there is nothing to clean. A '
            'source winning folds here is removing or moving the molecules the '
            'model finds hard, and its count at every other level has to be '
            'read against this one.'),
        'settings_winning_most_folds_at_level_zero': {
            f'{space}/{source}': blk['settings_winning_most_folds']
            for space, srow in zero.items()
            for source, blk in srow['per_source'].items()},
    } if zero else {'what_it_says': 'level 0.0 did not run'}
    return out


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric_name}; paired per fold against no cleaning)")
    if not results.get('hybrid_available', True):
        print("  no hybrid recipe was available; single representations only")
    secs = results.get('estimator_seconds_by_source') or {}
    if secs:
        print("  seconds per uncertainty estimate: "
              + ', '.join(f"{k} {v['mean']:.1f}" for k, v in secs.items()))
    tgt = results.get('target_source_by_source') or {}
    for src, where in tgt.items():
        print(f"    {src}: move pushes labels toward {', '.join(where)}")

    for regime, reg in s['per_regime'].items():
        print(f"\n  regime {regime}:")
        for lvl, lrow in reg['per_level'].items():
            print(f"    level {lvl}:")
            for space, srow in lrow['per_space'].items():
                for source, blk in srow['per_source'].items():
                    b = blk['best_setting']
                    # One detection reading per downstream model since
                    # 2026-09-23. Printed for the first model that has one and
                    # named, rather than averaged over models whose features
                    # differ; every model's is in the result file.
                    _dets = blk.get('detection_by_model') or {}
                    det, det_model = next(
                        ((d, m) for m, d in _dets.items() if d), (None, None))
                    detstr = ''
                    if det and np.isfinite(det.get('detection_average_precision',
                                                   float('nan'))):
                        detstr = (f"; in {det_model}'s features it found the "
                                  f"moved molecules at average "
                                  f"precision {det['detection_average_precision']:.3f} "
                                  f"over {det['detection_n_corrupted']:.0f} "
                                  f"of {det['detection_n_training']:.0f} "
                                  f"training molecules")
                    print(f"      {space:6s} {source:20s} won most folds in "
                          f"{blk['settings_winning_most_folds']} of "
                          f"{blk['settings_scored']} settings, cleared its own "
                          f"spread in "
                          f"{blk['settings_whose_gain_clears_its_own_spread']}"
                          f"{detstr}")
                    if b:
                        _sp = b['spread_over_fold_gains']
                        _sps = ('one fold, no spread' if _sp is None
                                else f'spread over folds {_sp:.4f}')
                        print(f"             best: {b['arm']}/{b['model']} — "
                              f"{b['folds_won']} of {b['n_folds_paired']} "
                              f"(draw, fold) passes, mean {b['mean_gain']:+.4f} "
                              f"({_sps})")
    z = s.get('zero_noise_control', {}).get(
        'settings_winning_most_folds_at_level_zero')
    if z:
        print("\n  the zero-noise control — nothing was corrupted, so nothing "
              "here is cleaning:")
        for k, v in z.items():
            print(f"    {k}: won most folds in {v} settings")
    print("\n  The count of folds won is the headline. A mean gain no larger "
          "than the\n  spread of the per-fold gains behind it is not a gain.")


def main():
    parser = create_experiment_parser(
        'N23: can a per-molecule uncertainty clean labels')
    parser.add_argument('--sources', type=str, default=None,
                        help='Comma-separated uncertainty sources')
    parser.add_argument('--noise-condition', type=str, default=None,
                        help=f'Injected condition (default {DEFAULT_NOISE_CONDITION})')
    parser.add_argument('--regimes', type=str, default=None,
                        help='Comma-separated: injected, as_they_come')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, rep_preset=args.reps)
    kw = {}
    if args.sources:
        kw['sources'] = [x.strip() for x in args.sources.split(',')]
    if args.noise_condition:
        kw['noise_condition'] = args.noise_condition
    if args.regimes:
        kw['regimes'] = [x.strip() for x in args.regimes.split(',')]
    res = run(data, config, args.output, args.dataset, **kw)
    save_experiment_results(res, args.output, 'n23_uncertainty_cleaning',
                            args.dataset,
                            filename_suffix=kw.get('noise_condition'))


if __name__ == '__main__':
    main()
