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

  injected      `outlier_p10` on the shared grid (0.0, 0.15, 0.3, 0.5, 0.75,
                1.0 of the clean label spread), three draws per level, one at
                level 0.0 on a regression set because the draw delivers exactly
                zero there. Detection is written here.
  as_they_come  nothing is corrupted. n18's regime. A win is a win on the data
                as it arrived, and there is no list to be right about.

WHAT THIS DOES NOT ASK. The features are fixed. Two single-representation
spaces and one hybrid per family of stored search the pool holds, each built
once per fold on the fold's training molecules at their UNCLEANED labels.

THE SINGLE-REPRESENTATION COMPARISON is each downstream model's top 2 base
representations by that model's own score in the stored search -- spaces
`base_1` and `base_2` -- the same rule as the rest of the study
(_shared.noise_base_reps_by_model, owner's decision 2026-09-25). Which
representations those are is written to `base_reps_by_model`. A model the pool
holds no stored search for has no single-representation space either, and
`base_reps_missing` says why.

There are two hybrid
families: a search that chose its features to predict the label ('hybrid') and
one that chose them so that each molecule's activity is predicted by its five
nearest neighbours in the distance those features define
('hybrid_similarity'). Before 2026-09-19 only the first was built and the
second was written into the result file as a score nothing had measured in. No
cleaning reaches the feature ranking or the budget split in any space. Whether
it should is n20's question, not this one.

WHAT THE HELD-OUT MOLECULES ARE GRADED AGAINST. `labels[val_idx]`, untouched, in
every arm and both regimes. No cleaned label and no corrupted label is ever
scored against.

THE HEADLINE IS A COUNT OF FOLDS, not a mean, for n18's reason at :33-37. Every
comparison is PAIRED: the same fold, the same held-out molecules, two arms.

SIMILARITY AXIS, ROWS, CACHES, CHECKPOINT (2026-09-25, plan sections 1-3).
The loop is today's path and computes what it computed before, with one
exception: activity_cliff_aware now excludes a molecule's scaffold group from
its neighbours, as n07, n24 and n25 do (_estimator_kwargs). On top of it, each
similarity setting x k of similarity_space.settings_from_env() is run for the
sources whose score reads neighbours, and for a source whose move target fell
back to the neighbours' mean; KIRBY_SIMILARITY_SETTINGS=model_input runs
today's path alone. Every (source, arm, model) writes one D-versus-D' row per
fold to <dataset>__<condition>.rows.jsonl beside the condition file; the
accumulators are checkpointed per fold; the hybrid builds, the scaled
representations, the no-cleaning and clean-label fits of seeded models and the
uncertainty estimates go through noise_cache. Controls run with
KIRBY_NOISE_CONTROLS=1.

Usage:
    python n23_uncertainty_cleaning.py --dataset herg_ki --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import functools
import time
import zlib
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS,
    get_best_search_for_downstream, make_nearest_neighbors,
    noise_base_reps_by_model, base_rank_groups, NOISE_BASE_REP_COUNT,
    # One feature space per search family the pool holds, added 2026-09-19.
    # n11's loop, in one place. See SPACE_FOR_FAMILY below.
    pool_recipes_by_family, describe_pool_family, POOL_SUBSETS,
)
from experiments.dataset_policy import (
    noise_levels, noise_groups_for, noise_dose_units,
)
from experiments.noise.noise_helpers import (
    noise_spec_for, inject_train_labels, detection_scores,
    detection_is_defined, hybrid_for_fold, DETECTION_IS_DEFINED,
    sweep_conditions,
    pool_recipe_per_downstream_model, group_models_by_recipe,
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
# d_vs_dprime.py and model_similarity_spaces into similarity_space.py.
#
# What is here:
#   RowSink                 the per-fold D-versus-D' rows (d_vs_dprime),
#                           resumed from the keys already on disk
#   PassCheckpoint          the experiment's own accumulators, saved after
#                           every pass, so a killed run resumes where it
#                           stopped instead of from the start
#   fit_score_cached        one downstream fit and its held-out score, through
#                           noise_cache when the model is seeded
#   swept_similarity        the similarity settings x ks beyond today's
#   model_similarity_spaces the features each similarity setting reads, for
#                           one downstream model on one fold
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
TODAY_SIMILARITY = f'{ss.MODEL_INPUT.name}|k=today'


def ensure_dprime_models(models):
    """`models` plus rf and svm, the models every D-versus-D' row needs
    (d_vs_dprime.DEFAULT_MODELS; owner decision 2026-09-25: "RF and SVM at
    least"). The default list, _shared.DOWNSTREAM_MODELS, already holds both,
    so a default run is unchanged; a caller that asked for fewer models gets
    these two added, and the result file's `models` says so."""
    out = list(models)
    for m in dvd.DEFAULT_MODELS:
        if m not in out:
            out.append(m)
    return out


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


def swept_similarity(available_spaces=None):
    """[(row label, SimilaritySetting, k)] for every setting beyond today's.

    Today's path -- neighbours in the model's own matrix, each consumer's own
    k -- is the experiment's existing loop and always runs; its rows carry
    TODAY_SIMILARITY. This is what is swept ON TOP of it:
    similarity_space.settings_from_env() (KIRBY_SIMILARITY_SETTINGS; unset or
    'all' is every setting, owner decision 2026-09-25) crossed with
    SWEPT_KS = (5, 9, 10, 20). 'model_input' alone returns [], so a run under
    KIRBY_SIMILARITY_SETTINGS=model_input computes exactly what it computed
    before 2026-09-25. A MODEL_INPUT entry in a comma list is today's path
    and adds nothing here.
    """
    out = []
    for s in ss.settings_from_env(available_spaces):
        if s.space == ss.MODEL_INPUT_SPACE:
            continue
        for k in ss.SWEPT_KS:
            out.append((f'{s.name}|k={int(k)}', s, int(k)))
    return out


def model_similarity_spaces(dataset_name, data, *, model_name, factory,
                            base_reps, hybrid_train, train_idx, val_idx,
                            y_train, groups_train, metric, task_type,
                            model_input_train):
    """({space: standardised training matrix}, {space: why it is missing})
    for ONE downstream model on ONE fold: the spaces similarity_space reads.

      model_input        the matrix the model is fitted on in this cell,
                         read by forest_proximity
      base_1, base_2     this model's rank-1 and rank-2 single
                         representations (_shared.noise_base_reps_by_model),
                         standardised on the fold's training rows
                         (noise_cache.cached_scaled, float64 as every
                         experiment here casts)
      base_k_aug_<st>    each base plus augmented features, built the way c21
                         builds B_aug (similarity_space.base_aug_blueprint),
                         for st in capped and uncapped, through
                         noise_cache.cached_hybrid_for_fold with the model in
                         the accept test, as c21 passes it (c21:925-929). The
                         augmentation pool is data['augmentations'] minus the
                         base itself (c21:429). The labels are `y_train` --
                         the ones the experiment holds, noisy under injection
                         (owner decision: importance and selection from the
                         noisy labels).
      hybrid, hybrid_similarity
                         the experiment's own per-model build for this fold,
                         handed in as `hybrid_train` {space: matrix}; never
                         rebuilt here.

    A space that cannot be built is left out and its reason returned, never
    faked (similarity_space.neighbour_lists then reports it unavailable).
    Only training rows are returned: every consumer here reads training
    neighbours only.
    """
    from sklearn.preprocessing import StandardScaler as _SS
    metric_name, metric_fn, lower_is_better = metric
    reps = data['reps']
    augs = data.get('augmentations') or {}
    spaces = {'model_input': model_input_train}
    why = {}
    for i in (1, 2):
        rn = base_reps[i - 1] if base_reps and len(base_reps) >= i else None
        if rn is None or rn not in reps:
            why[f'base_{i}'] = (f'{model_name} has no rank-{i} single '
                                f'representation')
            for st in ss.AUG_STRATEGIES:
                why[ss.aug_space_name(i, st)] = why[f'base_{i}']
            continue
        X_tr_b, _ = nc.cached_scaled(dataset_name, reps[rn], train_idx,
                                     val_idx, dtype=np.float64)
        spaces[f'base_{i}'] = X_tr_b
        aug_names = [a for a in augs if a != rn]
        for st in ss.AUG_STRATEGIES:
            an = ss.aug_space_name(i, st)
            if not aug_names:
                why[an] = (f'{dataset_name} loads no augmentation '
                           f'representation other than {rn}')
                continue
            try:
                bp = ss.base_aug_blueprint(
                    rn, int(np.shape(reps[rn])[1]), metric_name=metric_name,
                    task_type=task_type, lower_is_better=lower_is_better,
                    strategy=st)
                got = nc.cached_hybrid_for_fold(
                    dataset_name,
                    {rn: np.asarray(reps[rn][train_idx], dtype=np.float64)},
                    {rn: np.asarray(reps[rn][val_idx], dtype=np.float64)},
                    y_train, bp, groups_train, factory(), metric_fn,
                    lower_is_better, model_name=model_name,
                    metric_name=metric_name, train_idx=train_idx,
                    augmentations_train={a: np.asarray(augs[a])[train_idx]
                                         for a in aug_names},
                    augmentations_val={a: np.asarray(augs[a])[val_idx]
                                       for a in aug_names})
            except Exception as exc:
                why[an] = f'the augmented build raised {type(exc).__name__}: {exc}'
                continue
            if got is None or got[0] is None:
                why[an] = 'the augmented build returned nothing on this fold'
                continue
            spaces[an] = _SS().fit_transform(np.asarray(got[0]))
    for sp in ('hybrid', 'hybrid_similarity'):
        X = (hybrid_train or {}).get(sp)
        if X is None:
            why[sp] = f'no {sp} matrix was built for {model_name} on this fold'
        else:
            spaces[sp] = X
    return spaces, why


def smiles_for(data, idx):
    """The fold's SMILES for tanimoto_ecfp4, or None when the dataset has
    none (similarity_space then reports the space unavailable)."""
    smi = data.get('smiles')
    if smi is None:
        return None
    return [smi[int(i)] for i in np.asarray(idx)]


def setting_neighbours(dataset_name, setting, spaces_train, y_train, *,
                       exclude_groups, smiles_train, is_clf):
    """noise_cache.cached_neighbours for one setting, training rows only."""
    return nc.cached_neighbours(
        dataset_name, setting, spaces_train=spaces_train, spaces_val=None,
        y_train=y_train, exclude_groups=exclude_groups,
        smiles_train=smiles_train, is_clf=is_clf)


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


def cached_base_cell(dataset_name, X_full, train_idx, val_idx):
    """(X_tr_s, X_va_s) for a single-representation space: StandardScaler on
    the fold's training rows, float64, through noise_cache.cached_scaled.
    Bit-identical to StandardScaler().fit_transform(np.asarray(X[train],
    float64)) -- casting then indexing and indexing then casting give the same
    values."""
    return nc.cached_scaled(dataset_name, X_full, train_idx, val_idx,
                            dtype=np.float64)


def cached_hybrid_cell(dataset_name, train_reps, val_reps, y_train, bp,
                       groups_train, model_name, model, metric, train_idx,
                       augmentations_train=None, augmentations_val=None):
    """noise_helpers.hybrid_for_fold through noise_cache.cached_hybrid_for_fold.
    model_name is None under a shareable recipe, where no model is passed."""
    metric_name, metric_fn, lower_is_better = metric
    return nc.cached_hybrid_for_fold(
        dataset_name, train_reps, val_reps, y_train, bp, groups_train, model,
        metric_fn, lower_is_better, model_name=model_name,
        metric_name=metric_name, train_idx=train_idx,
        augmentations_train=augmentations_train,
        augmentations_val=augmentations_val)

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

# How many nearest neighbours the neighbour-based sources compare a molecule
# against, and how many the move target falls back to. n07's and n18's.
NEIGHBOUR_K = 10

# The condition the injected regime runs under. outlier_p10 corrupts a SUBSET,
# which is what makes detection_scores defined -- under gaussian, the student_t
# settings, laplace and grouped_shifted every training molecule moves.
DEFAULT_NOISE_CONDITION = 'outlier_p10'

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
# dirty-local 2026-09-28: only the names NoiseSpec accepts. DETECTION_IS_DEFINED
# also holds shift_only_p05/p10 and the six sparse_dprime_* conditions, which
# are not in the NoiseInject registry; noise_spec_for raises ValueError on them
# outside any try, so a sweep with no --noise-condition wrote the eight real
# conditions and then crashed on shift_only_p05 before the index was saved.
from kirby.noise_spec import REGRESSION_STRATEGIES as _REG_STRATEGIES
SWEPT_CONDITIONS = tuple(c for c in DETECTION_IS_DEFINED
                         if c in _REG_STRATEGIES)
NOISE_DRAWS = (0, 1, 2)
REGIMES = ('injected', 'as_they_come')

# Which searches the recipe may come from. Same as n02, n07, n11, n12, n13, n17,
# n18 and n19 since 2026-09-12.
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
HYBRID_SPACE = 'hybrid'
# base_1 and base_2 are each downstream model's rank-1 and rank-2 single
# representation (see the module docstring).
BASE_SPACES = tuple(f'base_{_k}' for _k in range(1, NOISE_BASE_REP_COUNT + 1))
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
    want = {'n_folds': n_folds, 'groups': groups_train, 'k': k,
            'metric': None, 'exclude_groups': groups_train}
    if is_clf:
        want['is_clf'] = True
    return {p: v for p, v in want.items() if p in accepted}


def _scored_out_of_fold(source):
    """Whether this source scores a molecule from a model that did not see it."""
    import inspect
    cls = get_noise_estimator.__globals__['ESTIMATOR_REGISTRY'][source]
    return 'n_folds' in set(inspect.signature(cls.__init__).parameters)


OWN_TARGET = 'the source own oof prediction'


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
           'k': int(neighbour_k)}
    return tuple(nc.cached(dataset_name, 'n23_uncertainty', key, _compute,
                           store_if=lambda v: v[0] is not None))


def _uncertainty(source, X_tr_s, y_tr, groups_train, is_clf, n_folds,
                 neighbour_k=10):
    """One spread per training molecule, and the denoised value to move toward.

    Returns (scores, target, target_source) or (None, None, why).

    `target` is the out-of-fold prediction the `move` shape pushes labels
    toward. The model-based sources produce one themselves and set
    `predictions_`; the neighbour-based ones do not, so their target is the mean
    of a molecule's ten nearest neighbours' labels -- which is what
    `n07.label_smooth` already moves toward. `target_source` is written onto
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

    # No denoised value of its own: move toward the neighbours' mean instead,
    # the same target n07.label_smooth uses.
    try:
        k = min(neighbour_k, len(y_tr) - 1)
        nn = make_nearest_neighbors(n_neighbors=k)
        nn.fit(X_tr_s)
        _, nbr_idx = nn.kneighbors(X_tr_s)
        return (scores, np.mean(y_tr[nbr_idx[:, 1:]], axis=1),
                'the mean of the ten nearest neighbours')
    except Exception as exc:
        return scores, None, f'no target: {type(exc).__name__}: {exc}'


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

    At level 0.0 on a regression dataset the draw delivers exactly zero, so
    three draws are three identical runs and one is kept. NOT on a
    classification set: the flip calibrator returns a floor rather than zero at
    a target of 0.0. Same rule as noise_helpers.noisy_search_specs.
    """
    levels = (noise_levels(dataset_name, strategy=condition, is_clf=is_clf)
              if levels is None else levels)
    draws = tuple(NOISE_DRAWS if draws is None else draws)
    out = []
    for lvl in levels:
        here = draws if (float(lvl) > 0 or is_clf) else draws[:1]
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

    It sweeps the eight in DETECTION_IS_DEFINED, not all fourteen: n23 grades a
    per-molecule uncertainty against the molecules whose label actually moved,
    and under a constant-width condition every training label moves, so the
    suspect set is every molecule and precision is 1.0 by construction.
    """
    return sweep_conditions(
        _run_one_condition, 'n23_uncertainty_cleaning',
        data, config, output_dir, dataset_name,
        default_conditions=SWEPT_CONDITIONS, kwargs=kwargs,
        groups_needed=GROUPED_STRATEGIES)


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    """N23: cleaning by a model's own per-molecule uncertainty, ONE condition."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N23: CAN A PER-MOLECULE UNCERTAINTY CLEAN LABELS")
    print("=" * 70)

    labels = np.asarray(data['labels'])
    reps = data['reps']
    groups = data.get('groups')
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    condition = kwargs.get('noise_condition') or DEFAULT_NOISE_CONDITION
    regimes = tuple(kwargs.get('regimes') or REGIMES)
    sources = tuple(kwargs.get('sources') or SOURCES)
    drop_shares = tuple(float(x) for x in
                        (kwargs.get('drop_shares') or DROP_SHARES))
    move_strengths = tuple(float(x) for x in
                           (kwargs.get('move_strengths') or MOVE_STRENGTHS))
    move_methods = tuple(kwargs.get('move_methods') or MOVE_METHODS)
    est_folds = int(kwargs.get('estimator_folds', ESTIMATOR_FOLDS))
    # The neighbour count the neighbour-based sources use, and the one
    # the move target falls back to. n07's and n18's.
    neighbour_k = int(kwargs.get('neighbour_k', NEIGHBOUR_K))

    # ⛔ config FIRST, then the kwarg. run_all passes the caller's model list in
    # `config['models']` and never as a `models` kwarg -- the only kwarg it ever
    # forwards is noise_condition (noise_helpers.py:1053). This file read
    # kwargs alone, so the list the runner was asked for was silently dropped
    # and every run fell back to the module default. Eight files in this
    # directory had it; the same files read config['n_splits'] correctly two
    # lines away. Fixed 2026-09-20. An explicit kwarg still wins, which is what
    # a smoke or a direct caller uses.
    # rf and svm are added when the caller left them out: every D-versus-D'
    # row needs them (ensure_dprime_models, owner decision 2026-09-25).
    downstream_models = ensure_dprime_models(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n23: no downstream model could be built. Asked for "
            f"{list(kwargs.get('models') or DOWNSTREAM_MODELS)}, got "
            f"{sorted(model_factories)}.")

    # {model: [rank-1 rep, rank-2 rep]} given directly (a smoke or a direct
    # caller). None means: rank them from each model's stored search, below.
    base_reps_override = kwargs.get('base_reps')
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
            f"{', '.join(c for c in DETECTION_IS_DEFINED if c not in GROUPED_STRATEGIES)}.")

    arms = _arms(drop_shares, move_strengths, move_methods)

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'regimes': list(regimes),
        'noise_condition': condition,
        'dose_units': noise_dose_units(dataset_name),
        'sources': list(sources),
        'shapes': ['none', 'drop', 'move'],
        'drop_shares': list(drop_shares),
        'move_strengths': list(move_strengths),
        'move_methods': list(move_methods),
        'arms': [a[0] for a in arms],
        'arm_to_beat': 'none',
        'models': downstream_models,
        'estimator_folds': est_folds,
        'neighbour_k': neighbour_k,
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
            "The hybrid is built once per fold per downstream model on that "
            "fold's training molecules at their UNCLEANED labels, from the "
            'recipe the search picked for that model, and every source, shape '
            'and setting shares it. No cleaning reaches the feature '
            "ranking or the budget split. Whether it should is n20's question."),
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
        # features encode; the three outlier conditions do not.
        'sources_defined_under_this_condition': {
            s: bool(s not in LEARNS_NOISE_FROM_FEATURES
                    or condition in NOISE_KEYED_TO_STRUCTURE)
            for s in sources},
        'condition_keys_noise_to_structure': bool(
            condition in NOISE_KEYED_TO_STRUCTURE),
        'sources_defined_note': (
            'False means this source predicts a molecule\'s noise from its '
            'features and the condition in use picked its corrupted molecules '
            'at random, so there is nothing in the features to generalise from '
            'and a chance-level reading is the condition rather than the '
            'method. The two grouped conditions are the ones that do not: '
            'grouped_wider gives whole scaffold families the wider error and '
            'grouped_shifted gives whole scaffold families a common offset. '
            'True under grouped_shifted does NOT mean a detection number '
            'exists: every training molecule moves under it, so there is no '
            'corrupted subset to rank and the only reading is whether cleaning '
            'wins folds. NOISE_AUDIT_2026_09_10.md 6.7.'),
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

    sample_X = next(iter(reps.values()))
    splits = list(cv_split(groups, n_folds, sample_X, labels, mode=mode))

    # ── The hybrid recipe, fetched once PER DOWNSTREAM MODEL. n18's block,
    # for n18's reason. ──
    #
    # ⛔ IT WAS ONE RECIPE, ASKED FOR downstream_models[0], AND EVERY MODEL WAS
    # SCORED ON IT. downstream_models[0] is 'rf' in every run this file has
    # made, so a run over the five models in DOWNSTREAM_MODELS -- rf, xgboost,
    # svm, dnn, lightgbm -- wrote five numbers measured in the features the
    # random forest's recipe chose. Each model now gets the recipe the search
    # picked FOR it, and a model the pool holds no search for gets no hybrid
    # space at all rather than the forest's. Changed 2026-09-23.
    by_model, no_recipe = pool_recipe_per_downstream_model(
        functools.partial(
            get_best_search_for_downstream, reps, labels, groups,
            budget=None, augmentations=data.get('augmentations'),
            dataset_name=dataset_name, task_type=task_type, data=data,
            pool_subset=POOL_SUBSET),
        downstream_models)
    # ⛔ WHAT WENT WRONG IS WRITTEN DOWN, NOT JUST PRINTED. Without this the
    # only trace in the result file was `hybrid_unavailable_reason`, which
    # says "no recipe came back from the pool subset" -- and a reader takes
    # that to mean the pool holds nothing for this dataset. On 2026-09-20 it
    # meant something else: every result_db file on the machine was written
    # before `distance_metric` joined the search identity, so opening one
    # raises StaleResultDB, and `_shared._iter_pool_searches` re-raises it
    # rather than skipping the pool member. The lookup helper catches it per
    # model and says which of the two happened.
    # Same shape of loss the three re-raise sites in result_cache.py were
    # added to stop. The exception text goes on the file so the two cases
    # can be told apart.
    _raised = [d['lookup_error'] for d in no_recipe.values()
               if d['lookup_error']]
    results_hybrid_lookup_error = _raised[0] if _raised else None
    results['hybrid_unavailable_by_model'] = {
        m: d['why'] for m, d in no_recipe.items()}

    # One recipe per (downstream model, search family the pool holds), and a
    # written reason for each family that produced none.
    recipes_by_model, recipe_for_space_by_model = {}, {}
    hybrid_unavailable = {}
    for _m in downstream_models:
        _m_recipes, _m_unavail = pool_recipes_by_family(
            by_model[_m]['meta'] if _m in by_model else None,
            SPACE_FOR_FAMILY,
            eligible_families=POOL_SUBSETS.get(POOL_SUBSET))
        recipes_by_model[_m] = _m_recipes
        recipe_for_space_by_model[_m] = {r['space']: r['blueprint']
                                         for r in _m_recipes}
        for _sp, _why in _m_unavail.items():
            hybrid_unavailable.setdefault(_sp, {})[_m] = _why
    # {space: {model: why that model has no recipe in it}}. One reason per
    # space until 2026-09-23, when the recipe became per downstream model: a
    # space can now be missing for one model and present for another.
    results['hybrid_spaces_unavailable'] = hybrid_unavailable
    results['hybrid_lookup_error'] = results_hybrid_lookup_error

    # ── The single-representation comparison, per downstream model ──
    if base_reps_override is not None:
        base_by_model = {m: list(base_reps_override[m])
                         for m in downstream_models if base_reps_override.get(m)}
        base_missing = {m: 'not given by the caller' for m in downstream_models
                        if m not in base_by_model}
    else:
        base_by_model, base_missing = noise_base_reps_by_model(
            {m: d['search'] for m, d in by_model.items()}, downstream_models,
            reps=reps, dataset_name=dataset_name)
    results['base_reps_by_model'] = base_by_model
    results['base_reps_missing'] = base_missing
    base_spaces_run = tuple(
        f'base_{k}' for k in range(1, NOISE_BASE_REP_COUNT + 1)
        if base_rank_groups(base_by_model, k, downstream_models))

    _first = next((m for m in downstream_models
                   if recipe_for_space_by_model.get(m)), None)
    bp = None
    if _first is None:
        spaces = base_spaces_run
        results['hybrid_available'] = False
        results['hybrid_unavailable_reason'] = (
            (f'the pool lookup RAISED {results_hybrid_lookup_error} — the pool '
             f'was never read, so this is not a statement about what it holds. '
             f'Every arm ran in the single representations only.')
            if results_hybrid_lookup_error else
            (f'no recipe came back from the pool subset {POOL_SUBSET!r} for '
             'any downstream model, so every arm ran in the single '
             'representations only. Dropping the run '
             'would lose real answers because one of several spaces is missing.'))
        print(f"  no hybrid recipe from pool subset {POOL_SUBSET}; "
              f"single representations only")
    else:
        # A space runs when at least one downstream model has a recipe for it,
        # in the order the pool reported them.
        _ordered = []
        for _m in downstream_models:
            for r in recipes_by_model.get(_m, ()):
                if r['space'] not in _ordered:
                    _ordered.append(r['space'])
        spaces = base_spaces_run + tuple(_ordered)
        hybrid_recipes = recipes_by_model[_first]
        _meta = by_model[_first]['meta']
        bp = by_model[_first]['blueprint']
        results['hybrid_available'] = True
        results['hybrid_models_with_a_recipe'] = [
            m for m in downstream_models if recipe_for_space_by_model.get(m)]
        results['hybrid_models_without_one'] = [
            m for m in downstream_models
            if not recipe_for_space_by_model.get(m)]
        results['hybrid_recipe_keys_name_one_model'] = _first
        results['hybrid_recipe_keys_note'] = (
            'hybrid_family, hybrid_budget, hybrid_reps, '
            'hybrid_space_of_pool_winner and hybrid_spaces all describe the '
            f'recipes the search picked for {_first}. Every other downstream '
            'model was scored in the features of ITS OWN recipe, and all of '
            'them are in hybrid_recipe_by_model.')
        results['hybrid_family'] = _meta.get('objective')
        results['hybrid_budget'] = getattr(bp, 'budget', None)
        results['hybrid_reps'] = sorted(getattr(bp, 'reps', []) or [])
        results['hybrid_space_of_pool_winner'] = SPACE_FOR_FAMILY.get(
            _meta.get('objective'), f"hybrid_{_meta.get('objective')}")
        results['hybrid_spaces'] = convert_numpy({
            r['space']: {
                'built_under_objective': r['objective'],
                'features_chosen_by': describe_pool_family(r['objective']),
                'is_the_recipe_the_pool_returned': r['is_pool_winner'],
                'tier': r['tier'], 'aug_condition': r['aug_condition'],
                'budget': r['budget'],
                'allocation_method': r['allocation_method'],
                'importance_method': r['importance_method'],
                'reps': sorted(getattr(r['blueprint'], 'reps', []) or []),
                'search_score_mean': r['search_score_mean'],
                'search_metric_name': r['search_metric_name'],
            } for r in hybrid_recipes})
        # One entry per (downstream model, space): the recipe that model's
        # features were rebuilt from in that space.
        results['hybrid_recipe_by_model'] = convert_numpy({
            m: {r['space']: {
                'built_under_objective': r['objective'],
                'tier': r['tier'], 'aug_condition': r['aug_condition'],
                'budget': r['budget'],
                'allocation_method': r['allocation_method'],
                'importance_method': r['importance_method'],
                'reps': sorted(getattr(r['blueprint'], 'reps', []) or []),
                'search_score_mean': r['search_score_mean'],
                'search_metric_name': r['search_metric_name'],
            } for r in rs}
            for m, rs in recipes_by_model.items() if rs})
        results['hybrid_spaces_note'] = (
            'One hybrid space per family of stored search, and inside each '
            'space one feature matrix per downstream model. A predictive '
            'search chose its features to predict the label and its '
            "search_score_mean is in the dataset's own metric. A similarity "
            "search chose its features so that each molecule's activity is "
            'predicted by its five nearest neighbours in the distance those '
            'features define, fits no downstream model, and its '
            'search_score_mean is that correlation. The two search scores are '
            'in different units and nothing here ranks one against the other. '
            "Every arm below is scored in the dataset's own metric, on the "
            'same folds, in every space.')
        for m in results['hybrid_models_with_a_recipe']:
            for r in recipes_by_model[m]:
                print(f"  {m} / hybrid space {r['space']}: family "
                      f"{r['objective']!r}, budget {r['budget']} features, "
                      f"allocation {r['allocation_method']}, search score "
                      f"{r['search_score_mean']} in "
                      f"{r['search_metric_name']!r}")
        for m in results['hybrid_models_without_one']:
            print(f"  {m}: no recipe — "
                  f"{(no_recipe.get(m) or {}).get('why', 'no recipe')}")
    for _sp, _why_by_model in hybrid_unavailable.items():
        for _m, _why in _why_by_model.items():
            print(f"  no {_sp} space for {_m}: {_why}")
    results['spaces_run'] = list(spaces)

    passes = {'as_they_come': [(None, 0)]}
    if 'injected' in regimes:
        passes['injected'] = _levels_and_draws(
            dataset_name, condition, is_clf,
            levels=kwargs.get('levels'), draws=kwargs.get('draws'))
        results['noise_levels'] = sorted({l for l, _ in passes['injected']})

    n_passes = sum(len(passes[r]) for r in regimes)
    print(f"  {len(labels)} molecules, {len(splits)} folds, "
          f"models {downstream_models}")
    print(f"  sources {list(sources)}")
    print(f"  {len(arms)} arms: 1 to beat, {len(drop_shares)} drop, "
          f"{len(move_methods) * len(move_strengths)} move")
    # How many feature matrices one fold holds: one per distinct representation
    # at each single-representation rank, plus one per (hybrid space,
    # downstream model with a recipe for it). The uncertainty estimate is
    # computed on each of them, so this is what the estimate count is a
    # multiple of.
    _matrices_per_fold = sum(
        len(base_rank_groups(base_by_model, _k, downstream_models))
        for _k in range(1, NOISE_BASE_REP_COUNT + 1)) + sum(
        1 for _sp in spaces if _sp not in BASE_SPACES
        for _m in downstream_models
        if _sp in recipe_for_space_by_model.get(_m, {}))
    print(f"  {n_passes} (level, draw) passes over {len(regimes)} regimes; "
          f"{_matrices_per_fold} feature "
          f"{'matrix' if _matrices_per_fold == 1 else 'matrices'} per fold "
          f"({len(spaces)} spaces, one matrix per distinct representation in "
          f"each single-representation space and one per downstream model in "
          f"each hybrid space); "
          f"{n_passes * len(splits) * len(sources) * _matrices_per_fold} "
          f"uncertainty estimates")

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
    # The similarity sweep's detection readings (2026-09-25):
    # sim_detection[level][space][model][source][similarity label] = [reading]
    sim_detection = {}
    # Why a similarity space could not be built, per (space, model); the
    # first reason seen is kept.
    sim_unavailable = {}
    t0 = time.time()

    # ── Rows, checkpoints, the similarity axis (2026-09-25) ──
    # The D-versus-D' rows of plan section 2, one file per condition next to
    # the condition's result file (sweep_conditions writes
    # <exp>/<dataset>__<condition>.json).
    rows = RowSink(output_dir, 'n23_uncertainty_cleaning', dataset_name,
                   experiment='n23', metric=(metric_name, metric_fn,
                                             lower_is_better),
                   condition=condition, data=data, config=config)
    # n23 saved nothing until a condition finished: one condition measured
    # 1,991 to 7,095 seconds locally (sweep_conditions docstring). The
    # accumulators are now checkpointed after every (regime, level, draw,
    # fold) and a rerun skips the ones already done.
    ckpt = PassCheckpoint(
        output_dir, 'n23_uncertainty_cleaning', dataset_name,
        condition=condition, data=data, config=config,
        source_file=__file__,
        key={'regimes': list(regimes), 'sources': list(sources),
             'drop_shares': list(drop_shares),
             'move_strengths': list(move_strengths),
             'move_methods': list(move_methods), 'est_folds': est_folds,
             'neighbour_k': neighbour_k, 'models': list(downstream_models),
             'base_reps': base_by_model, 'levels': kwargs.get('levels'),
             'draws': kwargs.get('draws'), 'n_splits': n_folds})
    done_passes = set()
    _state = ckpt.load()
    if _state:
        acc, errors = _state['acc'], _state['errors']
        detection_acc, est_seconds = _state['detection_acc'], _state['est_seconds']
        target_src, removed_acc = _state['target_src'], _state['removed_acc']
        sim_detection = _state['sim_detection']
        sim_unavailable = _state['sim_unavailable']
        done_passes = set(_state['done'])
        print(f"  resumed from {ckpt.path}: {len(done_passes)} (regime, level, "
              f"draw, fold) passes already done")
    sim_axis = swept_similarity()
    # The sources whose SCORE reads a neighbour list (only the neighbour
    # estimators take precomputed_neighbors: noise_cache.NEIGHBOUR_DETECTORS).
    nbr_sources = [s for s in sources if s in nc.NEIGHBOUR_DETECTORS]
    results['similarity_sweep'] = {
        'settings_env': os.environ.get('KIRBY_SIMILARITY_SETTINGS'),
        'n_settings_times_k': len(sim_axis),
        'ks': list(ss.SWEPT_KS),
        'sources_whose_score_reads_neighbours': nbr_sources,
        'what_is_swept': (
            'Beyond today\'s path (TODAY_SIMILARITY on a row), every '
            'similarity setting x k is run for: the sources whose score reads '
            'neighbours (activity_cliff_aware), all arms, their move target '
            'being the mean label of the k nearest neighbours in the '
            'setting; and a source with no prediction of its own whose move '
            'target fell back to the neighbours\' mean, its move arms only. '
            'Neighbours for a SCORE exclude the molecule\'s scaffold group, '
            'as the score does; neighbours for a TARGET do not, as today\'s '
            'target does not (n23 _uncertainty). The sources that read no '
            'neighbours are not swept: their rows carry similarity_setting '
            '"none".'),
    }

    def _save_state():
        ckpt.save({'acc': acc, 'errors': errors,
                   'detection_acc': detection_acc, 'est_seconds': est_seconds,
                   'target_src': target_src, 'removed_acc': removed_acc,
                   'sim_detection': sim_detection,
                   'sim_unavailable': sim_unavailable,
                   'done': sorted(done_passes)})

    for regime in regimes:
        if regime not in passes:
            raise ValueError(f"n23: unknown regime {regime!r}")
        acc.setdefault(regime, {})
        for level, draw in passes[regime]:
            lvl_key = 'none' if level is None else f'{level:g}'
            tag = (regime if level is None
                   else f'{regime}/{condition}/level {level:g}/draw {draw}')
            print(f"\n  --- {tag} ---")
            cell = acc[regime].setdefault(lvl_key, {})
            spec = (None if regime != 'injected'
                    else noise_spec_for(dataset_name, condition, level,
                                        noise_seed=draw, is_clf=is_clf))
            # What a row calls this pass.
            row_cond = condition if regime == 'injected' else 'as_they_come'
            row_level = None if level is None else float(level)

            for fold_i, (train_idx, val_idx) in enumerate(splits):
                fold_key = f'l{lvl_key}d{draw}f{fold_i}'
                pass_id = f'{regime}|{fold_key}'
                if pass_id in done_passes:
                    print(f"    fold {fold_i}: on disk in the checkpoint")
                    continue
                y_val = labels[val_idx]
                g_tr = groups[train_idx] if groups is not None else None

                cmask = None
                if spec is None:
                    y0 = np.asarray(labels[train_idx], dtype=float)
                else:
                    try:
                        y0, cmask, _prov = inject_train_labels(
                            labels, spec, train_idx=train_idx, is_clf=is_clf,
                            groups=noise_groups, dataset_name=dataset_name)
                    except Exception as exc:
                        errors[f'{regime}|{lvl_key}|d{draw}|f{fold_i}|inject'] = (
                            f'{type(exc).__name__}: {exc}')
                        continue
                    y0 = np.asarray(y0, dtype=float)
                # The clean training labels, for the score_clean column. In
                # the as_they_come regime they ARE y0.
                y_clean_tr = np.asarray(labels[train_idx], dtype=float)

                train_reps = {k: np.asarray(v[train_idx], dtype=np.float64)
                              for k, v in reps.items()}
                val_reps = {k: np.asarray(v[val_idx], dtype=np.float64)
                            for k, v in reps.items()}

                # The feature spaces, all built from this fold's training
                # molecules at their UNCLEANED labels and shared by every arm.
                # A single-representation space (base_1, base_2) holds one
                # matrix per distinct representation at that rank, shared by
                # the downstream models that rank it there. A hybrid space is
                # one matrix PER
                # downstream model: that model's own recipe, rebuilt with that
                # model in the accept/reject test. Until 2026-09-23 it was one
                # matrix built from downstream_models[0]'s recipe, and every
                # model was scored on it.
                #
                # {space: [(training matrix, held-out matrix, the models
                # scored on it)]}.
                fold_spaces = {}
                for _k, _bsp in enumerate(BASE_SPACES, start=1):
                    if _bsp not in spaces:
                        continue
                    _bcells = []
                    for _rn, _ms in base_rank_groups(
                            base_by_model, _k, downstream_models).items():
                        # noise_cache.cached_scaled since 2026-09-25: the
                        # scaled representation depends on the fold only.
                        _Xt, _Xv = cached_base_cell(dataset_name, reps[_rn],
                                                    train_idx, val_idx)
                        _bcells.append((_Xt, _Xv, list(_ms)))
                    fold_spaces[_bsp] = _bcells
                # One build per (family the pool holds, downstream model).
                # hybrid_for_fold reads the objective off the recipe and drops
                # the model, the metric and its direction under a geometry
                # recipe, where build_fold_hybrid refuses the pair, so one call
                # serves both families.
                for _sp in spaces:
                    if _sp in BASE_SPACES:
                        continue
                    # ONE BUILD PER DISTINCT RECIPE, not per model. A model
                    # with no recipe in this family still gets no matrix rather
                    # than another model's; models sharing one geometry recipe
                    # share the build, because no downstream model is fitted
                    # inside it. See noise_helpers.group_models_by_recipe.
                    # A failed build is still dropped without a record here,
                    # which is what this did before and is not changed by the
                    # grouping.
                    _cells = []
                    for _bp_m, _mnames, _shareable in group_models_by_recipe(
                            {_m: recipe_for_space_by_model.get(_m, {}).get(_sp)
                             for _m in downstream_models}, downstream_models):
                        # Through noise_cache.cached_hybrid_for_fold since
                        # 2026-09-25: the same build for the same draw is
                        # shared with n24, n25, n30-n35, n38, n40 and n41.
                        X_tr_h, X_va_h = cached_hybrid_cell(
                            dataset_name, train_reps, val_reps, y0, _bp_m,
                            g_tr, None if _shareable else _mnames[0],
                            None if _shareable
                            else model_factories[_mnames[0]](),
                            (metric_name, metric_fn, lower_is_better),
                            train_idx)
                        if X_tr_h is not None:
                            sh = StandardScaler()
                            _cells.append((sh.fit_transform(X_tr_h),
                                           sh.transform(X_va_h),
                                           list(_mnames)))
                    if _cells:
                        fold_spaces[_sp] = _cells
                # {model: {hybrid space: its training matrix}}, for the
                # similarity spaces of the sweep below.
                hybrid_by_model = {}
                for _sp, _cs in fold_spaces.items():
                    if _sp in BASE_SPACES:
                        continue
                    for _Xt, _Xv, _ms in _cs:
                        for _m in _ms:
                            hybrid_by_model.setdefault(_m, {})[_sp] = _Xt

                # One pass per (space, feature matrix): once per distinct
                # representation in each single-representation space, once per
                # downstream model in each hybrid space.
                for space, (X_tr_s, X_va_s, cell_models) in (
                        (_sp, _cell) for _sp in spaces
                        for _cell in fold_spaces.get(_sp, ())):
                    space_cell = cell.setdefault(space, {})
                    # What an error is filed under. Two downstream models in
                    # one hybrid space are two feature spaces, so their
                    # failures must not overwrite each other under one key.
                    _ekey = (space if cell_models == list(downstream_models)
                             else f'{space}/{cell_models[0]}')
                    # score_clean per model on this matrix: the model fitted
                    # on the clean training labels (plan section 2). In the
                    # as_they_come regime the clean labels are D, so the D
                    # score below is used instead of a second identical fit.
                    clean_score = {}
                    if regime == 'injected':
                        for mname in cell_models:
                            s_c, _e = fit_score_cached(
                                dataset_name, mname, model_factories[mname],
                                X_tr_s, y_clean_tr, X_va_s, y_val, metric_fn,
                                metric_name)
                            clean_score[mname] = s_c
                    # {model: the first no-cleaning score on this matrix},
                    # the score_D of the similarity sweep's rows.
                    d_score = {}
                    # {source: (scores, target, target source)} from today's
                    # path, reused by the sweep for a source whose score
                    # reads no neighbours.
                    today_by_source = {}

                    for source in sources:
                        t_est = time.time()
                        scores, target, tsrc = _uncertainty_cached(
                            dataset_name, source, X_tr_s, y0, g_tr, is_clf,
                            est_folds, neighbour_k=neighbour_k)
                        est_seconds.setdefault(source, []).append(
                            time.time() - t_est)
                        if scores is None:
                            errors[f'{regime}|{lvl_key}|{_ekey}|{source}|'
                                   f'f{fold_i}|estimate'] = tsrc
                            print(f"      {_ekey}/{source}: {tsrc}")
                            continue
                        target_src.setdefault(source, set()).add(tsrc)
                        today_by_source[source] = (scores, target, tsrc)

                        # Did the number find the molecules that really moved,
                        # before any cleaning ran? It costs no model fit.
                        if cmask is not None and detection_is_defined(
                                spec.strategy):
                            det = detection_scores(cmask, scores,
                                                   spec.strategy)
                            if det is not None:
                                # Under each downstream model this matrix was
                                # built for. Models sharing one representation
                                # in a single-representation space share one
                                # matrix, so the same reading is written under
                                # each of them; otherwise they are different
                                # feature spaces and different readings.
                                for _dm in cell_models:
                                    (detection_acc
                                     .setdefault(lvl_key, {})
                                     .setdefault(space, {})
                                     .setdefault(_dm, {})
                                     .setdefault(source, []).append(det))

                        src_cell = space_cell.setdefault(source, {})
                        for arm, shape, setting, method in arms:
                            try:
                                got = cleaned_training_set(
                                    shape, setting, X_tr_s, y0, scores,
                                    target, method=method)
                            except Exception as exc:
                                errors[f'{regime}|{lvl_key}|{_ekey}|{source}|'
                                       f'{arm}|f{fold_i}|clean'] = (
                                    f'{type(exc).__name__}: {exc}')
                                continue
                            if got is None:
                                errors[f'{regime}|{lvl_key}|{_ekey}|{source}|'
                                       f'{arm}|f{fold_i}|clean'] = (
                                    'the shape could not be applied on this fold')
                                continue
                            Xc, yc = got
                            if shape == 'drop':
                                rm = ~drop_keep_mask(setting, scores)
                                _rm_row = {
                                    'n_removed': int(rm.sum()),
                                    'n_training': int(len(scores)),
                                    'n_moved_among_the_removed': (
                                        None if cmask is None else
                                        int(np.sum(np.asarray(cmask, dtype=bool) & rm))),
                                    'n_moved': (None if cmask is None
                                                else int(np.sum(cmask)))}
                                # Under each model this matrix was built for,
                                # for the reason the detection block gives.
                                for _dm in cell_models:
                                    (removed_acc
                                     .setdefault(lvl_key, {})
                                     .setdefault(space, {})
                                     .setdefault(_dm, {})
                                     .setdefault(source, {})
                                     .setdefault(arm, []).append(dict(_rm_row)))
                            for mname in cell_models:
                                if shape == 'none':
                                    # The no-cleaning fit is the same fit for
                                    # every source; through noise_cache for a
                                    # seeded model (plan section 3: "n23
                                    # refits it 5 times per pass").
                                    s, err = fit_score_cached(
                                        dataset_name, mname,
                                        model_factories[mname], Xc, yc,
                                        X_va_s, y_val, metric_fn, metric_name)
                                else:
                                    s, err = _fit_score(
                                        model_factories[mname], Xc, yc, X_va_s,
                                        y_val, metric_fn, metric_name)
                                if s is None:
                                    errors[f'{regime}|{lvl_key}|{_ekey}|'
                                           f'{source}|{arm}|{mname}|'
                                           f'f{fold_i}'] = err
                                    continue
                                (src_cell.setdefault(arm, {})
                                         .setdefault(mname, {})[fold_key]) = s
                                if shape == 'none':
                                    d_score.setdefault(mname, s)

                        # ── The D-versus-D' rows of today's path ──
                        # One per (arm, model); score_D is this source's own
                        # no-cleaning fit on the same matrix and fold.
                        _reads_nbrs_score = source in nc.NEIGHBOUR_DETECTORS
                        for arm, shape, setting, method in arms:
                            if shape == 'none':
                                continue
                            _nb = (_reads_nbrs_score or
                                   (shape == 'move' and tsrc != OWN_TARGET))
                            for mname in cell_models:
                                sD = (src_cell.get('none') or {}).get(
                                    mname, {}).get(fold_key)
                                sP = (src_cell.get(arm) or {}).get(
                                    mname, {}).get(fold_key)
                                if sD is None and sP is None:
                                    continue
                                rows.add(method=source, method_setting=arm,
                                         similarity_setting=(
                                             TODAY_SIMILARITY if _nb
                                             else NO_NEIGHBOURS),
                                         model_input=space, model=mname,
                                         noise_condition=row_cond,
                                         level=row_level, draw=draw,
                                         fold=fold_i, score_D=sD,
                                         score_Dprime=sP,
                                         score_clean=clean_score.get(mname, sD))

                        # ── Controls (KIRBY_NOISE_CONTROLS=1 only) ──
                        # random_twin and shuffled_twin of every drop and move
                        # arm, fitted per model and written as rows whose
                        # method_setting ends |control=<twin>.
                        if controls_enabled():
                            for arm, shape, setting, method in arms:
                                if shape == 'none':
                                    continue
                                try:
                                    if shape == 'drop':
                                        _cd = as_cleanup_dict(
                                            X_tr_s, y0,
                                            keep=drop_keep_mask(setting, scores))
                                    else:
                                        _g = cleaned_training_set(
                                            shape, setting, X_tr_s, y0, scores,
                                            target, method=method)
                                        if _g is None:
                                            continue
                                        _cd = as_cleanup_dict(X_tr_s, y0,
                                                              labels=_g[1])
                                except Exception:
                                    continue
                                _seed = (int(draw), int(fold_i),
                                         int(zlib.crc32(f'{source}|{arm}|{space}'
                                                        .encode()) & 0x7FFFFFFF))
                                for cname, tw, why in control_twins(
                                        _cd, X_tr_s, y0, _seed, is_clf=is_clf):
                                    for mname in cell_models:
                                        if tw is None:
                                            errors[f'{regime}|{lvl_key}|{_ekey}|'
                                                   f'{source}|{arm}|{cname}|'
                                                   f'f{fold_i}'] = why
                                            continue
                                        sP, err = fit_cleaned(
                                            model_factories[mname], tw, X_va_s,
                                            y_val, metric_fn, metric_name)
                                        sD = (src_cell.get('none') or {}).get(
                                            mname, {}).get(fold_key)
                                        rows.add(
                                            method=source,
                                            method_setting=f'{arm}|control={cname}',
                                            similarity_setting=(
                                                TODAY_SIMILARITY
                                                if (_reads_nbrs_score or (
                                                    shape == 'move'
                                                    and tsrc != OWN_TARGET))
                                                else NO_NEIGHBOURS),
                                            model_input=space, model=mname,
                                            noise_condition=row_cond,
                                            level=row_level, draw=draw,
                                            fold=fold_i, score_D=sD,
                                            score_Dprime=sP,
                                            score_clean=clean_score.get(mname, sD))

                    # ── The similarity sweep (2026-09-25, plan section 1) ──
                    # Per downstream model, because each model's similarity
                    # spaces are its own (its base_1/base_2, its hybrids).
                    # The model is still fitted on THIS matrix: the model
                    # input and the similarity space are separate axes.
                    _sweep_sources = [
                        s_ for s_ in sources if s_ in today_by_source and (
                            s_ in nc.NEIGHBOUR_DETECTORS
                            or today_by_source[s_][2] != OWN_TARGET)]
                    if sim_axis and _sweep_sources:
                        for mname in cell_models:
                            _spaces_tr, _why = model_similarity_spaces(
                                dataset_name, data, model_name=mname,
                                factory=model_factories[mname],
                                base_reps=base_by_model.get(mname),
                                hybrid_train=hybrid_by_model.get(mname),
                                train_idx=train_idx, val_idx=val_idx,
                                y_train=y0, groups_train=g_tr,
                                metric=(metric_name, metric_fn,
                                        lower_is_better),
                                task_type=task_type,
                                model_input_train=X_tr_s)
                            for _sp_u, _w in _why.items():
                                sim_unavailable.setdefault(
                                    _sp_u, {}).setdefault(mname, _w)
                            _smi = smiles_for(data, train_idx)
                            _lists = {}
                            for label, sset, kk in sim_axis:
                                if sset.name not in _lists:
                                    _ex = setting_neighbours(
                                        dataset_name, sset, _spaces_tr, y0,
                                        exclude_groups=g_tr,
                                        smiles_train=_smi, is_clf=is_clf)
                                    _pl = (_ex if g_tr is None else
                                           setting_neighbours(
                                               dataset_name, sset, _spaces_tr,
                                               y0, exclude_groups=None,
                                               smiles_train=_smi,
                                               is_clf=is_clf))
                                    _lists[sset.name] = (_ex, _pl)
                                nb_ex, nb_pl = _lists[sset.name]
                                if not (nb_ex.get('available')
                                        and nb_pl.get('available')):
                                    sim_unavailable.setdefault(
                                        sset.space, {}).setdefault(
                                            mname, nb_ex.get('reason')
                                            or nb_pl.get('reason'))
                                    continue
                                try:
                                    _nidx, _ = ss.neighbours_at(nb_pl, kk)
                                    s_target = np.mean(y0[_nidx], axis=1)
                                except Exception as exc:
                                    errors[f'{regime}|{lvl_key}|{_ekey}|{mname}|'
                                           f'{label}|f{fold_i}|target'] = (
                                        f'{type(exc).__name__}: {exc}')
                                    continue
                                for source in _sweep_sources:
                                    t_scores, t_target, t_src = today_by_source[source]
                                    if source in nc.NEIGHBOUR_DETECTORS:
                                        try:
                                            s_scores = nc.cached_detector_scores(
                                                dataset_name, X_tr_s, y0,
                                                is_clf=is_clf, detector=source,
                                                groups=g_tr,
                                                similarity_neighbours=nb_ex,
                                                similarity_k=kk)
                                        except Exception as exc:
                                            errors[f'{regime}|{lvl_key}|{_ekey}|'
                                                   f'{mname}|{source}|{label}|'
                                                   f'f{fold_i}|estimate'] = (
                                                f'{type(exc).__name__}: {exc}')
                                            continue
                                        if s_scores is None:
                                            continue
                                        s_scores = np.asarray(s_scores, dtype=float)
                                        # A neighbour source has no prediction
                                        # of its own: its target is the
                                        # neighbours' mean, here in the setting.
                                        s_tgt = s_target
                                        _arms_here = [a for a in arms
                                                      if a[1] != 'none']
                                        if cmask is not None and \
                                                detection_is_defined(spec.strategy):
                                            det = detection_scores(
                                                cmask, s_scores, spec.strategy)
                                            if det is not None:
                                                (sim_detection
                                                 .setdefault(lvl_key, {})
                                                 .setdefault(space, {})
                                                 .setdefault(mname, {})
                                                 .setdefault(source, {})
                                                 .setdefault(label, [])
                                                 .append(det))
                                    else:
                                        # Its score reads no neighbours; only
                                        # its fallback move target did.
                                        s_scores = t_scores
                                        s_tgt = s_target
                                        _arms_here = [a for a in arms
                                                      if a[1] == 'move']
                                    for arm, shape, setting, method in _arms_here:
                                        try:
                                            got = cleaned_training_set(
                                                shape, setting, X_tr_s, y0,
                                                s_scores, s_tgt, method=method)
                                        except Exception as exc:
                                            errors[f'{regime}|{lvl_key}|{_ekey}|'
                                                   f'{mname}|{source}|{label}|'
                                                   f'{arm}|f{fold_i}|clean'] = (
                                                f'{type(exc).__name__}: {exc}')
                                            continue
                                        if got is None:
                                            continue
                                        sP, err = _fit_score(
                                            model_factories[mname], got[0],
                                            got[1], X_va_s, y_val, metric_fn,
                                            metric_name)
                                        if sP is None:
                                            errors[f'{regime}|{lvl_key}|{_ekey}|'
                                                   f'{mname}|{source}|{label}|'
                                                   f'{arm}|f{fold_i}'] = err
                                        sD = d_score.get(mname)
                                        rows.add(method=source,
                                                 method_setting=arm,
                                                 similarity_setting=label,
                                                 model_input=space,
                                                 model=mname,
                                                 noise_condition=row_cond,
                                                 level=row_level, draw=draw,
                                                 fold=fold_i, score_D=sD,
                                                 score_Dprime=sP,
                                                 score_clean=clean_score.get(
                                                     mname, sD))
                                        if not controls_enabled():
                                            continue
                                        _cd = (as_cleanup_dict(
                                            X_tr_s, y0,
                                            keep=drop_keep_mask(setting,
                                                                s_scores))
                                               if shape == 'drop' else
                                               as_cleanup_dict(X_tr_s, y0,
                                                               labels=got[1]))
                                        _seed = (int(draw), int(fold_i), int(
                                            zlib.crc32(f'{source}|{arm}|{space}|'
                                                       f'{mname}|{label}'.encode())
                                            & 0x7FFFFFFF))
                                        for cname, tw, why in control_twins(
                                                _cd, X_tr_s, y0, _seed,
                                                is_clf=is_clf):
                                            if tw is None:
                                                continue
                                            sPc, _e = fit_cleaned(
                                                model_factories[mname], tw,
                                                X_va_s, y_val, metric_fn,
                                                metric_name)
                                            rows.add(
                                                method=source,
                                                method_setting=(
                                                    f'{arm}|control={cname}'),
                                                similarity_setting=label,
                                                model_input=space, model=mname,
                                                noise_condition=row_cond,
                                                level=row_level, draw=draw,
                                                fold=fold_i, score_D=sD,
                                                score_Dprime=sPc,
                                                score_clean=clean_score.get(
                                                    mname, sD))

                # The fold is finished: its rows go to disk and the
                # accumulators to the checkpoint, in that order, so a kill
                # between the two re-runs the fold and adds no duplicate row.
                rows.flush()
                done_passes.add(pass_id)
                _save_state()
                print(f"    fold {fold_i} done [{time.time() - t0:.0f}s]")

    results['per_regime'] = convert_numpy(acc)
    # {level: {space: {downstream model: {source: reading}}}}. The model level
    # arrived 2026-09-23 with the per-model recipe: in a hybrid space each
    # model's uncertainty number was computed in ITS OWN features, so one
    # reading per space would have averaged two different feature spaces.
    results['detection'] = convert_numpy({
        lvl: {sp: {m: {src: _mean_detection(rows)
                       for src, rows in by_src.items()}
                   for m, by_src in by_model.items()}
              for sp, by_model in by_sp.items()}
        for lvl, by_sp in detection_acc.items()})
    results['detection_note'] = (
        'One reading per (noise level, feature space, downstream model, '
        'uncertainty source). In base_1 and base_2 the downstream models that '
        'rank the same representation there share one feature matrix, so '
        'their readings are identical; in a hybrid space each model has its '
        'own features and its own reading.')
    results['target_source_by_source'] = {k: sorted(v)
                                          for k, v in target_src.items()}
    # Summed over the folds of one (level, space, source, arm), so every count
    # below is molecules, not a share and not a per-fold average.
    results['rows_removed_by_drop_arm'] = convert_numpy({
        lvl: {sp: {m: {src: {arm: {
            'n_folds': len(rows),
            'molecules_removed_summed_over_the_folds':
                int(sum(r['n_removed'] for r in rows)),
            'training_molecules_summed_over_the_folds':
                int(sum(r['n_training'] for r in rows)),
            'moved_molecules_removed_summed_over_the_folds':
                (None if any(r['n_moved_among_the_removed'] is None for r in rows)
                 else int(sum(r['n_moved_among_the_removed'] for r in rows))),
            'moved_molecules_summed_over_the_folds':
                (None if any(r['n_moved'] is None for r in rows)
                 else int(sum(r['n_moved'] for r in rows))),
        } for arm, rows in by_arm.items()}
            for src, by_arm in by_src.items()}
            for m, by_src in by_model.items()}
        for sp, by_model in by_sp.items()}
        for lvl, by_sp in removed_acc.items()})
    results['rows_removed_note'] = (
        'One entry per (feature space, downstream model, source, drop arm) -- '
        'the model level because a hybrid space is one feature matrix per '
        'downstream model since 2026-09-23, so the molecules a drop arm '
        'removes differ between them. Every number is a count of molecules summed '
        'over the outer folds of that level, so the share the arm removed is '
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
        _summarise(acc, [a[0] for a in arms], downstream_models, metric_name,
                   lower_is_better, results['detection']))
    # The similarity sweep's detection readings, averaged over the folds and
    # draws of a level as `detection` is, and the spaces that could not be
    # built with the reason (2026-09-25).
    results['similarity_sweep']['detection'] = convert_numpy({
        lvl: {sp: {m: {src: {lab: _mean_detection(rs)
                             for lab, rs in by_lab.items()}
                       for src, by_lab in by_src.items()}
                   for m, by_src in by_m.items()}
              for sp, by_m in by_sp.items()}
        for lvl, by_sp in sim_detection.items()})
    results['similarity_sweep']['spaces_unavailable'] = sim_unavailable
    results['rows'] = rows.summary()
    results['controls_run'] = bool(controls_enabled())
    _print_summary(results, metric_name)
    print(f"  D-versus-D' rows: {rows.n_written} written this run to "
          f"{rows.path or 'memory'}")
    # The result is complete; the per-pass checkpoint is not needed again.
    ckpt.clear()
    return results


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


def _paired(base, got, lower_is_better):
    """One arm against `none`, on the folds both scored. Positive means better."""
    shared = sorted(set(base) & set(got))
    gains = np.array([(base[k] - got[k]) if lower_is_better
                      else (got[k] - base[k]) for k in shared], dtype=float)
    gains = gains[np.isfinite(gains)]
    scale = max(abs(float(np.mean(list(base.values())))) if base else 1.0, 1.0)
    tol = 1e-9 * scale
    return {
        'n_folds_paired': int(gains.size),
        'folds_won': int((gains > tol).sum()),
        'folds_lost': int((gains < -tol).sum()),
        # A large tied count means the arm is the uncleaned fit under a
        # different name -- a move at a strength that shifts nothing, say.
        'folds_tied': int((np.abs(gains) <= tol).sum()),
        'tie_tolerance': float(tol),
        'mean_gain': float(gains.mean()) if gains.size else float('nan'),
        'spread_over_fold_gains': (float(gains.std()) if gains.size
                                   else float('nan')),
        'fold_gains': [float(x) for x in gains],
        # A gain no larger than the spread of the per-fold gains behind it is
        # not a gain. Two measured numbers compared, not a test: the suite's
        # standing rule at this replicate count is raw spread only.
        'gain_exceeds_its_own_spread': bool(
            gains.size > 0 and gains.mean() > gains.std()),
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
                        print(f"             best: {b['arm']}/{b['model']} — "
                              f"{b['folds_won']} of {b['n_folds_paired']} "
                              f"folds, mean {b['mean_gain']:+.4f} "
                              f"(spread {b['spread_over_fold_gains']:.4f})")
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
    # dirty-local 2026-09-28: run() goes through sweep_conditions, which has
    # ALREADY written each condition's result to <dataset>__<condition>.json.
    # `res` is the sweep's index; with --noise-condition it was saved under
    # that same suffix and overwrote the result. The index goes to
    # <dataset>.json.
    save_experiment_results(res, args.output, 'n23_uncertainty_cleaning',
                            args.dataset)


if __name__ == '__main__':
    main()
