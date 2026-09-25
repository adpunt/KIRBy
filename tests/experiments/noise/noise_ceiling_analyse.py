#!/usr/bin/env python3
"""Extrapolate each noise sweep back to zero error, and read off the ceiling.

WHAT THIS IS. A reading of numbers the noise suite has already written. It runs no
model, injects nothing and loads no dataset. It opens the result JSON files under
tests/results/noise, pulls out every score-against-level curve it can find, fits each
one, and extrapolates.

WHY. `noise_helpers._already_there` records what the labels were already carrying
before anything was injected. It gets that from a published assay error, and
`dataset_policy.REFERENCE_ERROR_LABEL_UNITS` holds one for three datasets out of the
forty-two the suite runs -- herg_ki, caco2 and logd. The other thirty-nine have no
ceiling recorded at all. Extrapolating the sweep gives a version measured on this
repository's own runs instead of read off someone else's figure.

THE ARITHMETIC, and the thing every number here has to be labelled with.

Each injecting experiment sweeps six levels of ADDED error, in fractions of the clean
label spread: 0.0, 0.15, 0.3, 0.5, 0.75, 1.0 (dataset_policy._DEFAULT_NOISE_LEVELS).
Call one of those L. The labels were not exact before L was added: they carry b of
their own, also as a fraction of their own spread. Error adds in quadrature, so the
TOTAL error at level L is

    t(L) = sqrt(b**2 + L**2)

and the six scores are a curve against t, not against L. At L = 0 the total is not
zero -- it is b. So three different quantities live near the bottom of every curve and
they are not interchangeable:

    score at L = 0        what the experiment MEASURED with nothing added. Total
                          error b. On herg_ki b is 0.591 of the label spread.
    score at t = b        the same point, as the FIT places it. The gap between this
                          and the measured one says how well the fit describes the
                          low end, which is where the extrapolation starts.
    score at t = 0        the CEILING. Extrapolated, never measured. What the score
                          would be if the labels carried no error of their own.

The difference between the second and the third is the thing being estimated: what the
assay's own error is costing. Every printed number below says which of the three it is.

WHAT THE EXTRAPOLATION ASSUMES, stated rather than buried. The injected error goes into
the TRAINING labels only; every experiment here scores against held-out labels as they
come (n09_lnl_survey.py:397-405 injects on train_idx, scores on labels[val_idx]).
The labels' own error b is in both halves. So running the curve back past L = 0 to
t = 0 treats b as though it behaved like injected error, which removes it from the
training side only. The held-out labels still carry it and no extrapolation of this
curve takes it out of them. The ceiling here is therefore "what the score would be with
the training labels exact", not "with the assay exact everywhere".

WHERE b COMES FROM. Each result file carries a `noise_provenance` block, one row per
noise spec, and on the three registered datasets those rows carry
`baseline_error_fraction_of_spread` (b) and `baseline_error_label_units` (the published
figure in the label's own units). This script reads b from there and never recomputes
it. Where no row carries it -- the other thirty-nine datasets -- the ceiling cannot be
placed at a single number, so the fit is reported as a slope plus a short table of what
the ceiling would be over a range of b.

TWO CONDITIONS THAT GET NO QUADRATURE. Censoring's level is the fraction of labels
clipped, and a classification level is the fraction of labels flipped. Neither is an
amount of error and neither adds in quadrature with anything -- `_already_there` already
refuses to write a total for censoring for exactly this reason. Those curves are fitted
against L and the extrapolation stops at L = 0, which is a different quantity from the
ceiling and is labelled as such.

WHY NOT THE EXISTING CURVE CODE. `noiseInject.metrics._curve_robustness` wants four
levels and the shared grid has six, so it runs. But every scalar it returns is computed
from ret(x) = score(x) / score(0), which is 1 at x = 0 by construction. A quantity
pinned to the level-0 score cannot estimate something above it. Its numbers are printed
beside the fits where the metric is one it is defined on, as a cross-reference.

Usage:
    python tests/experiments/noise/noise_ceiling_analyse.py
    python tests/experiments/noise/noise_ceiling_analyse.py --dataset herg_ki
    python tests/experiments/noise/noise_ceiling_analyse.py --input tests/results/noise
    python tests/experiments/noise/noise_ceiling_analyse.py --json out.json
"""

import os
import sys
import json
import math
import re
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'tests'))

import numpy as np

DEFAULT_INPUT = os.path.join(
    os.environ.get('KIRBY_RESULTS_DIR', 'tests/results'), 'noise')

# A two-parameter fit on two points passes through both and says nothing about the
# shape, so it cannot be extrapolated anywhere. Three is the fewest that leaves a
# residual to read.
MIN_POINTS_2P = 3
MIN_POINTS_3P = 4

# Conditions whose level is a FRACTION OF LABELS, not an amount of error. They get no
# total and no quadrature. Kept as prefixes because a file records the condition it ran
# under as a plain string.
FRACTION_OF_LABELS_CONDITIONS = ('censoring',)


# ---------------------------------------------------------------------------
# One curve: the scores one experiment wrote for one thing, at each level.
# ---------------------------------------------------------------------------

class Curve(object):
    """Scores at each added-error level, for ONE thing, on ONE dataset.

    `what` names the thing the scores are of, in the experiment's own words -- a
    representation name, a model name, a condition name, or a combination. It is
    carried through unchanged so a row here can be found again in the file it came
    from.
    """

    def __init__(self, dataset, experiment, condition, what, metric,
                 lower_is_better, levels, scores, source_file,
                 is_clf=False):
        order = np.argsort(np.asarray(levels, dtype=float))
        self.dataset = dataset
        self.experiment = experiment
        self.condition = condition
        self.what = what
        self.metric = metric
        self.lower_is_better = bool(lower_is_better)
        self.levels = [float(levels[i]) for i in order]
        self.scores = [float(scores[i]) for i in order]
        self.source_file = str(source_file)
        self.is_clf = bool(is_clf)
        # Set where the x axis counts LABELS changed rather than measuring an amount
        # of error, whatever the condition is called. Nothing adds in quadrature with
        # a count of labels.
        self.x_counts_labels = False
        self.b = None                  # fraction of the clean label spread
        self.published_error_units = None   # the label's own units
        self.b_source = 'not recorded in this file'

    @property
    def level_is_a_fraction_of_labels(self):
        """True where the level counts LABELS changed, not an amount of error."""
        if self.is_clf or self.x_counts_labels:
            return True
        c = (self.condition or '').lower()
        return any(c.startswith(p) for p in FRACTION_OF_LABELS_CONDITIONS)

    @property
    def label_spread(self):
        """The clean label SD in the label's own units, or None.

        Recovered from the two provenance fields rather than by loading the dataset:
        `baseline_error_label_units` is the published error in label units and
        `baseline_error_fraction_of_spread` is that same error divided by the spread.
        """
        if self.b is None or self.published_error_units is None or self.b <= 0:
            return None
        return float(self.published_error_units) / float(self.b)

    def totals(self, b=None):
        """Total error at each level: sqrt(b**2 + L**2), or L itself where b is unknown."""
        bb = self.b if b is None else b
        if bb is None or self.level_is_a_fraction_of_labels:
            return np.asarray(self.levels, dtype=float)
        return np.sqrt(float(bb) ** 2 + np.asarray(self.levels, dtype=float) ** 2)

    def label(self):
        return "%s / %s / %s / %s" % (self.dataset, self.experiment,
                                      self.condition, self.what)


# ---------------------------------------------------------------------------
# Readers. One adapter per experiment, because each writes its own shape.
# ---------------------------------------------------------------------------

def _finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# A level key that carries a SECOND axis in it. n13's `partial` condition writes
# "0.3_frac0.1" -- strength 0.3 delivered to a tenth of the molecules -- so the level
# and the share of molecules it reached are one string. Reading the whole string as a
# number fails, and dropping it loses the condition outright.
_SECOND_AXIS_KEY = re.compile(r'^(?P<level>-?[0-9.eE+]+)_(?P<axis>[A-Za-z_]+)(?P<value>[0-9.eE+-]+)$')


def _level_groups(mapping):
    """Split a level-keyed dict into one group per second axis value.

    Returns (groups, unparsed). `groups` maps a label for the second axis -- the empty
    string where there is none -- to a sorted list of (level, original key). `unparsed`
    lists any key that is neither a number nor a number with a second axis stuck to it,
    so a caller can say what it could not read instead of dropping it.
    """
    if not isinstance(mapping, dict):
        return {}, []
    groups, unparsed = {}, []
    for k in mapping:
        # float() FIRST. Calling it inside setdefault(...).append(...) creates the
        # empty-string group before the conversion is allowed to fail, which left
        # every compound-key block looking as though it had a plain level axis.
        plain = None
        try:
            plain = float(k)
        except (TypeError, ValueError):
            pass
        if plain is not None:
            groups.setdefault('', []).append((plain, k))
            continue
        m = _SECOND_AXIS_KEY.match(str(k))
        if m:
            label = '%s%s' % (m.group('axis'), m.group('value'))
            groups.setdefault(label, []).append((float(m.group('level')), k))
        else:
            unparsed.append(str(k))
    for label in groups:
        groups[label].sort()
    return groups, unparsed


def _levels_of(mapping):
    """The single group of a mapping with no second axis. Empty when it has one."""
    groups, _unparsed = _level_groups(mapping)
    return groups.get('', [])


def _provenance_blocks(payload):
    """Every noise_provenance dict in a file, top level or inside a level block.

    n09, n11, n12 and n13 write one at the top. n21 writes one inside each level
    block (n21_neighbour_distance.py:989), so a reader that looked only at the top
    found no baseline error on it and reported the dataset as having none.
    """
    out = []
    top = payload.get('noise_provenance')
    if isinstance(top, dict):
        out.append(top)

    def _walk(node, depth):
        if depth > 3 or not isinstance(node, dict):
            return
        for k, v in node.items():
            if k == 'noise_provenance' and isinstance(v, dict):
                out.append(v)
            elif isinstance(v, dict):
                _walk(v, depth + 1)

    _walk(payload, 0)
    return out


def _attach_baseline_error(curve, payload):
    """Fill in b and the published figure from the file's own provenance rows."""
    rows = {}
    for block in _provenance_blocks(payload):
        rows.update(block)
    for spec_id, row in rows.items():
        if not isinstance(row, dict):
            continue
        b = _finite(row.get('baseline_error_fraction_of_spread'))
        if b is None:
            continue
        curve.b = b
        curve.published_error_units = _finite(row.get('baseline_error_label_units'))
        curve.b_source = ("noise_provenance[%s], from "
                          "dataset_policy.REFERENCE_ERROR_LABEL_UNITS[%s]"
                          % (spec_id, row.get('baseline_error_source', curve.dataset)))
        return


def _common(payload, dataset, experiment, source_file):
    metric = payload.get('metric') or payload.get('metric_name')
    lower = payload.get('lower_is_better')
    is_clf = bool(payload.get('is_clf') or payload.get('task_type') == 'classification')
    return dict(dataset=dataset, experiment=experiment, metric=metric,
                lower_is_better=bool(lower), source_file=source_file, is_clf=is_clf)


def _model_series(cells):
    """Which score series a list of level cells holds: (model name or None, its label).

    n11, n12 and n13 all write `noisy_score_by_model`, one score per downstream model,
    beside a bare `noisy_score` which is the reference model's. A file written before
    the per-model block existed has only the bare one, and dropping it would lose the
    whole file rather than one of its series.

    Takes the CELLS, not the dict they came out of, because the caller has already
    decided which keys belong to this curve -- `partial` writes "0.3_frac0.1" and the
    level and the share it reached come apart there, not here.
    """
    cells = [c for c in cells if isinstance(c, dict)]
    models = set()
    for c in cells:
        models.update((c.get('noisy_score_by_model') or {}).keys())
    if models:
        return [(m, 'noisy score, model %s' % m) for m in sorted(models)]
    if any(_finite(c.get('noisy_score')) is not None for c in cells):
        return [(None, 'noisy score, reference model (this file predates the '
                       'per-model block)')]
    return []


def _model_score(cell, model):
    if model is None:
        return _finite((cell or {}).get('noisy_score'))
    return _finite(((cell or {}).get('noisy_score_by_model') or {}).get(model))


def _read_n09(payload, dataset, experiment, source_file):
    """per_sigma[level][condition]['mean'], and per_sigma[level][condition][method]['mean'].

    Since 2026-09-25 n09 runs its single-representation conditions once per
    rank of the DNN's top 2: 'base_1_no_lnl' / 'base_2_no_lnl' and
    'base_1_lnl' / 'base_2_lnl', with the representations in
    payload['base_reps_by_model']['dnn']. A file written before that carries
    the one 'base_no_lnl' / 'base_lnl' pair and 'base_rep', and is still read.
    """
    per = payload.get('per_sigma') or {}
    if not isinstance(per, dict) or not per:
        return []
    cond = payload.get('noise_condition', 'unknown')
    kw = _common(payload, dataset, experiment, source_file)
    out = []

    # {condition prefix: the representation it was run in}, one entry per rank.
    _rank_model = payload.get('base_reps_ranked_by', 'dnn')
    _ranked = (payload.get('base_reps_by_model') or {}).get(_rank_model)
    if _ranked is not None:
        base_arms = {'base_%d' % (i + 1): r for i, r in enumerate(_ranked)}
    else:
        base_arms = {'base': payload.get('base_rep', '?')}

    plain = (['%s_no_lnl' % b for b in base_arms]
             + ['hybrid_no_lnl', 'weighted_hybrid_no_lnl'])
    for arm in plain:
        levels, scores = [], []
        for lvl, key in _levels_of(per):
            v = _finite((per[key].get(arm) or {}).get('mean'))
            if v is not None:
                levels.append(lvl)
                scores.append(v)
        if len(levels) >= MIN_POINTS_2P:
            # A single-representation arm names its representation. The
            # hybrid arms are not run in one, so they carry none.
            _b = arm[:-len('_no_lnl')]
            if _b == 'base':
                what = '%s (base rep %s)' % (arm, base_arms[_b])
            elif _b in base_arms:
                what = '%s (single representation %s)' % (arm, base_arms[_b])
            else:
                what = arm
            out.append(Curve(condition=cond, what=what,
                             levels=levels, scores=scores, **kw))

    # The level-0 slot holds no method scores at all (n09 only fills them where
    # sigma > 0), so a method curve is missing its bottom point by construction. Kept,
    # and the missing level is what `n_points` and `lowest level fitted` report.
    methods = payload.get('methods') or []
    for arm in (['%s_lnl' % b for b in base_arms]
                + ['hybrid_lnl', 'weighted_hybrid_lnl']):
        for m in methods:
            levels, scores = [], []
            for lvl, key in _levels_of(per):
                cell = (per[key].get(arm) or {}).get(m) or {}
                v = _finite(cell.get('mean'))
                if v is not None:
                    levels.append(lvl)
                    scores.append(v)
            if len(levels) >= MIN_POINTS_2P:
                out.append(Curve(condition=cond, what='%s / %s' % (arm, m),
                                 levels=levels, scores=scores, **kw))
    return out


def _read_by_model(payload, dataset, experiment, source_file, per_key='per_sigma'):
    """per_sigma[level]['noisy_score_by_model'][model] -- n11 and n12 both write this."""
    per = payload.get(per_key) or {}
    if not isinstance(per, dict) or not per:
        return []
    cond = payload.get('noise_condition', 'unknown')
    kw = _common(payload, dataset, experiment, source_file)
    out = []
    for m, what in _model_series([per[key] for _l, key in _levels_of(per)]):
        levels, scores = [], []
        for lvl, key in _levels_of(per):
            v = _model_score(per[key], m)
            if v is not None:
                levels.append(lvl)
                scores.append(v)
        if len(levels) >= MIN_POINTS_2P:
            out.append(Curve(condition=cond, what=what,
                             levels=levels, scores=scores, **kw))
    return out


def _read_n21(payload, dataset, experiment, source_file):
    """n21 keys its levels 'per_level'; the block inside is n11's shape."""
    return _read_by_model(payload, dataset, experiment, source_file,
                          per_key='per_level')


def _read_n13(payload, dataset, experiment, source_file):
    """per_noise_type[condition]['per_strength'][level]['noisy_score_by_model'][model]."""
    per_nt = payload.get('per_noise_type') or {}
    if not isinstance(per_nt, dict) or not per_nt:
        return []
    kw = _common(payload, dataset, experiment, source_file)
    out = []
    for nt in sorted(per_nt):
        _blk = per_nt[nt]
        per = (_blk.get('per_strength') if isinstance(_blk, dict) else None) or {}
        if not isinstance(per, dict):
            print('  %s / %s: per_strength holds a %s, not an object'
                  % (dataset, nt, type(per).__name__))
            continue
        groups, unparsed = _level_groups(per)
        if unparsed:
            print('  %s / %s: level keys not read: %s'
                  % (dataset, nt, ', '.join(sorted(unparsed))))
        n_before = len(out)
        for axis_label in sorted(groups):
            # One sub-dict per value of the second axis, so `_model_series` sees only
            # the levels that belong to this curve.
            for m, what in _model_series([per[key] for _l, key in groups[axis_label]]):
                levels, scores = [], []
                for lvl, key in groups[axis_label]:
                    v = _model_score(per[key], m)
                    if v is not None:
                        levels.append(lvl)
                        scores.append(v)
                if len(levels) >= MIN_POINTS_2P:
                    cond = nt if not axis_label else '%s (%s)' % (nt, axis_label)
                    out.append(Curve(condition=cond, what=what,
                                     levels=levels, scores=scores, **kw))
        if len(out) > n_before or groups.get(''):
            continue
        # Nothing was swept over the STRENGTH here: every second-axis value holds one
        # strength only. `partial` is run this way -- one strength, delivered to a
        # different share of the molecules each time -- so the axis that moves is the
        # SHARE, and the curve is over that instead. A share of molecules is a count of
        # labels, not an amount of error, so it adds in quadrature with nothing and
        # `level_is_a_fraction_of_labels` says so.
        out.extend(_read_second_axis(per, groups, nt, kw))
    return out


def _read_second_axis(per, groups, nt, kw):
    """Curves over the SECOND axis, where the level axis holds one value per group.

    The share the level was delivered to is the x here. Marked as counting labels so
    no quadrature is applied to it and the extrapolation stops at zero share, which is
    the score on untouched labels -- not a ceiling.
    """
    by_level = {}
    for axis_label, entries in groups.items():
        m = re.match(r'^([A-Za-z_]+)([0-9.eE+-]+)$', axis_label)
        if not m or len(entries) != 1:
            return []
        lvl, key = entries[0]
        by_level.setdefault((lvl, m.group(1)), []).append((float(m.group(2)), key))
    out = []
    for (lvl, axis_name), entries in sorted(by_level.items()):
        entries.sort()
        for model, what in _model_series([per[key] for _x, key in entries]):
            xs, scores = [], []
            for x, key in entries:
                v = _model_score(per[key], model)
                if v is not None:
                    xs.append(x)
                    scores.append(v)
            if len(xs) < MIN_POINTS_2P:
                continue
            c = Curve(condition='%s at strength %g, swept over %s' % (nt, lvl, axis_name),
                      what=what, levels=xs, scores=scores, **kw)
            c.x_counts_labels = True
            out.append(c)
    return out


def _read_n23(payload, dataset, experiment, source_file):
    """n23 writes per_regime[regime][level][space][source][arm][model][fold] = score.

    It is a sweep, and it was unclassified until 2026-09-21. It runs the injected
    regime over dataset_policy's level grid and writes a held-out score at every
    one of them, so leaving it out of both tables dropped every curve it produces
    without a note. It is NOT in n11's shape: there is no `noisy_score_by_model`
    cell. The innermost mapping is one score per fold key, so the point on the
    curve is the MEAN over the folds that scored. n23's own headline is a count of
    folds won, and a count of folds is not a point on an error curve.

    What each curve is. One per (feature space, uncertainty source, arm, model).
    `arm` is what was done to the training labels -- 'none' is the uncleaned fit,
    'drop@0.1' removes the tenth of training molecules with the widest predicted
    spread, 'move_confidence@0.5' shifts every training label toward the model's
    own out-of-fold prediction. The 'none' arm is the same fit under all five
    sources, so it is emitted once per space rather than five times.

    The 'as_they_come' regime is skipped: it corrupts nothing and its only key is
    'none', which is not a level.
    """
    per_regime = payload.get('per_regime') or {}
    inj = per_regime.get('injected')
    if not isinstance(inj, dict) or not inj:
        return []
    kw = _common(payload, dataset, experiment, source_file)
    cond = payload.get('noise_condition', 'unknown')

    # (space, source, arm, model) -> {level: mean over the folds that scored}
    series = {}
    for lvl, key in _levels_of(inj):
        for space, by_source in (inj[key] or {}).items():
            if not isinstance(by_source, dict):
                continue
            for source, by_arm in by_source.items():
                if not isinstance(by_arm, dict):
                    continue
                for arm, by_model in by_arm.items():
                    if not isinstance(by_model, dict):
                        continue
                    for model, by_fold in by_model.items():
                        if not isinstance(by_fold, dict):
                            continue
                        vals = [_finite(v) for v in by_fold.values()]
                        vals = [v for v in vals if v is not None]
                        if not vals:
                            continue
                        src = None if arm == 'none' else source
                        series.setdefault((space, src, arm, model), {})[lvl] = (
                            float(np.mean(vals)))

    out = []
    for (space, src, arm, model), by_level in sorted(
            series.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        if len(by_level) < MIN_POINTS_2P:
            continue
        levels = sorted(by_level)
        what = ('no cleaning, %s space, model %s' % (space, model)
                if src is None else
                'cleaning %s keyed on %s, %s space, model %s'
                % (arm, src, space, model))
        out.append(Curve(condition=cond, what=what, levels=levels,
                         scores=[by_level[l] for l in levels], **kw))
    return out


def _read_c10(payload, dataset, experiment, source_file):
    """individual_reps[rep][condition]['scores_by_level'], plus best_single and hybrids."""
    kw = _common(payload, dataset, experiment, source_file)
    out = []

    def _add(block, what_fmt):
        for cond, cell in (block or {}).items():
            if not isinstance(cell, dict):
                continue
            sbl = cell.get('scores_by_level') or {}
            levels, scores = [], []
            for lvl, key in _levels_of(sbl):
                v = _finite(sbl[key])
                if v is not None:
                    levels.append(lvl)
                    scores.append(v)
            if len(levels) >= MIN_POINTS_2P:
                out.append(Curve(condition=cond, what=what_fmt,
                                 levels=levels, scores=scores, **kw))

    for rep, per_cond in (payload.get('individual_reps') or {}).items():
        _add(per_cond, 'representation %s on its own' % rep)
    _add(payload.get('best_single') or {}, 'best single representation, re-picked per fold')
    for budget, per_cond in (payload.get('hybrids') or {}).items():
        _add(per_cond, 'hybrid at budget %s' % budget)
    return out


def _read_n15(payload, dataset, experiment, source_file):
    """per_level_per_model[level][model]['noisy_bp_score_mean_over_draws']."""
    per = payload.get('per_level_per_model') or {}
    if not isinstance(per, dict) or not per:
        return []
    kw = _common(payload, dataset, experiment, source_file)
    cond = payload.get('noise_condition', 'unknown')
    models = set()
    for _lvl, key in _levels_of(per):
        models.update((per[key] or {}).keys())
    out = []
    for m in sorted(models):
        for field, name in [('noisy_bp_score_mean_over_draws', 'noise-searched recipe'),
                            ('clean_bp_score_mean_over_draws', 'clean-searched recipe')]:
            levels, scores = [], []
            for lvl, key in _levels_of(per):
                v = _finite(((per[key] or {}).get(m) or {}).get(field))
                if v is not None:
                    levels.append(lvl)
                    scores.append(v)
            if len(levels) >= MIN_POINTS_2P:
                out.append(Curve(condition=cond, what='%s, model %s' % (name, m),
                                 levels=levels, scores=scores, **kw))
    return out


# Directory name under tests/results/noise -> (short id, reader).
READERS = {
    'n09_lnl_survey':                  ('n09', _read_n09),
    'n11_estimation_remediation_grid': ('n11', _read_by_model),
    'n12_importance_weighted_distance': ('n12', _read_by_model),
    'n13_noise_distributions':         ('n13', _read_n13),
    'n15_clean_vs_noisy_hybrid':       ('n15', _read_n15),
    # n21 writes the same by-model block n11 and n12 do, keyed 'per_level' rather
    # than 'per_sigma'. It was in NOT_A_SWEEP until 2026-09-17 on the grounds that
    # average precision is its headline; it also scores predictions at every level,
    # and skipping it dropped every one of those curves without a note.
    'n21_neighbour_distance':          ('n21', _read_n21),
    # n23 sweeps the injected level and writes a held-out score at each one, so it
    # belongs here rather than in NOT_A_SWEEP. Its shape is its own -- one score
    # per fold key, not n11's `noisy_score_by_model` -- so it gets its own reader.
    # Added 2026-09-21, after _smoke_noise_ceiling part5 reported it unclassified.
    'n23_uncertainty_cleaning':        ('n23', _read_n23),
    'c10_noise_robustness':            ('c10', _read_c10),
}

# Read and reported as NOT a curve producer, with the reason, rather than left out
# silently. n07 runs at ONE level (dataset_policy.reference_noise_level, 0.5 of the
# label spread) so it has no curve to fit; n14 and n16 train no downstream model and
# write no score; n17, n18, n19 and n20 inject nothing at all.
NOT_A_SWEEP = {
    'noise_ceiling': 'this script\'s own output',
    'n01_activity_cliff_framework': 'injects nothing (noise_helpers.py:758)',
    'n02_sample_weighting': 'injects nothing (noise_helpers.py:758)',
    'n03_sas_landscape': 'cut 2026-07-07, irredeemable by design (AUDIT_NOISE.md)',
    'n04_gp_loo_detection': 'cut 2026-07-07, irredeemable by design (AUDIT_NOISE.md)',
    'n05_noise_injection_benchmark': 'cut 2026-07-07, irredeemable by design (AUDIT_NOISE.md)',
    'n08_estimation_survey': 'cut 2026-07-07, irredeemable by design (AUDIT_NOISE.md)',
    'n10_hybrid_neighborhood_lnl': 'cut 2026-07-07, irredeemable by design (AUDIT_NOISE.md)',
    'n22_curator_flag': 'injects nothing; its suspect records are the ones ChEMBL\'s '
                        'own curators marked, and it sweeps no level',
    'n07_remediation': 'runs at one level only (reference_noise_level = 0.5), no curve',
    'n14_activity_cliff_landscape': 'a gate; trains no downstream model, writes no score',
    'n16_importance_under_noise': 'a gate; scores feature rankings, not predictions',
    'n17_replicate_removal': 'injects nothing; its noise is disagreement between laboratories',
    'n18_clean_the_real_labels': 'injects nothing; runs on the labels as they come',
    'n19_measurement_count': 'injects nothing; its axis is the measurement count',
    'n20_clean_before_search': 'injects nothing',
    'n06_bindingdb_ground_truth': 'injects nothing; real measurement disagreement',
}


# The blocks an experiment writes its level sweep into. A directory claiming not to
# sweep, whose files hold one of these keyed by numbers, is mislabelled.
_LEVEL_BLOCK_KEYS = ('per_sigma', 'per_level', 'per_strength', 'per_level_per_model',
                     'scores_by_level')


def _looks_like_a_sweep(directory):
    """The first file in `directory` that holds a numeric level-keyed block, or None.

    A guard against this script's own NOT_A_SWEEP table going stale. n21 sat in that
    table until 2026-09-17 while writing `per_level` with a score at every level, and
    nothing said so -- the note read like a decision rather than a mistake.
    """

    def _numeric_level_block(node, depth=0):
        if depth > 5 or not isinstance(node, dict):
            return False
        for k, v in node.items():
            if k in _LEVEL_BLOCK_KEYS and isinstance(v, dict) and v:
                if any(_level_groups(v)[0].values()):
                    return True
            if isinstance(v, dict) and _numeric_level_block(v, depth + 1):
                return True
        return False

    for f in sorted(directory.glob('*.json'))[:5]:
        try:
            payload = json.loads(f.read_text())
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get('result_status') != 'skipped':
            if _numeric_level_block(payload):
                return f.name
    return None


def read_curves(input_dir, dataset_filter=None):
    """Every curve under `input_dir`, plus one note per directory that produced none."""
    root = Path(input_dir)
    curves, notes = [], []
    if not root.exists():
        return curves, ['%s does not exist' % root]
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if sub.name in NOT_A_SWEEP:
            contradicted = _looks_like_a_sweep(sub)
            if contradicted:
                notes.append(
                    '%s: MARKED "not a sweep" (%s) BUT %s holds a level-keyed block '
                    'of scores. The label is wrong and its curves are being dropped; '
                    'move it into READERS.'
                    % (sub.name, NOT_A_SWEEP[sub.name], contradicted))
            else:
                notes.append('%s: skipped -- %s' % (sub.name, NOT_A_SWEEP[sub.name]))
            continue
        if sub.name.startswith('_'):
            notes.append('%s: skipped -- a probe, not an experiment' % sub.name)
            continue
        if sub.name not in READERS:
            notes.append('%s: skipped -- no reader written for this experiment\'s '
                         'result shape. If it sweeps a level and writes a score, it '
                         'needs one adding to READERS.' % sub.name)
            continue
        exp_id, reader = READERS[sub.name]
        n_here = 0
        for f in sorted(sub.glob('*.json')):
            # A file is <dataset>.json or <dataset>__<condition>.json.
            ds = f.stem.split('__', 1)[0]
            if dataset_filter and ds != dataset_filter:
                continue
            try:
                payload = json.loads(f.read_text())
            except Exception as e:
                notes.append('%s: unreadable -- %s' % (f, e))
                continue
            if not isinstance(payload, dict):
                notes.append('%s: holds a %s, not an object; no reader can use it'
                             % (f, type(payload).__name__))
                continue
            if payload.get('result_status') == 'skipped':
                notes.append('%s: the run wrote a skip, not scores' % f)
                continue
            # One malformed file costs one file. Before this, a reader that raised
            # on a half-written result took the whole dataset's read-out with it and
            # the job produced nothing at all.
            try:
                got = reader(payload, ds, exp_id, f)
                for c in got:
                    _attach_baseline_error(c, payload)
            except Exception as e:
                notes.append('%s: could not be read -- %s: %s'
                             % (f, type(e).__name__, e))
                continue
            curves.extend(got)
            n_here += len(got)
        if n_here == 0:
            notes.append('%s: read, no curve with %d or more levels in it'
                         % (sub.name, MIN_POINTS_2P))
    return curves, notes


# ---------------------------------------------------------------------------
# The fits.
# ---------------------------------------------------------------------------

MIN_DISTINCT_X = 2


def _unfittable(levels, scores, min_points):
    """Why this curve cannot be fitted, or None.

    Checked before every fit. Least squares on a design whose x column is constant
    returns a number without complaining, and that number is not an extrapolation of
    anything -- it is whatever the solver's pseudo-inverse produced.
    """
    if len(levels) < min_points:
        return 'fewer than %d levels' % min_points
    y = np.asarray(scores, dtype=float)
    if not np.all(np.isfinite(y)):
        return 'a score that is not a finite number'
    if len(set(float(v) for v in levels)) < MIN_DISTINCT_X:
        return 'every level is the same value, so there is no curve'
    if float(np.max(y) - np.min(y)) <= 0:
        return 'every score is the same value, so there is nothing to extrapolate'
    return None


class Fit(object):
    """One extrapolation of one curve back to zero total error."""

    def __init__(self, form, ceiling, at_level_zero, slope, r_squared,
                 max_residual, n_points, reach, extra=None, refused=None):
        self.form = form
        self.ceiling = ceiling              # score at total error 0
        self.at_level_zero = at_level_zero  # the FITTED score at total error b
        self.slope = slope
        self.r_squared = r_squared
        self.max_residual = max_residual    # largest |residual|, in the score's units
        self.n_points = n_points
        self.reach = reach                  # how far past the data the fit runs
        self.extra = extra or {}
        self.refused = refused

    @property
    def cost_of_baseline_error(self):
        """Fitted score at total error b, minus the ceiling. What b is costing."""
        if self.ceiling is None or self.at_level_zero is None:
            return None
        return float(self.at_level_zero - self.ceiling)


def _ols(design, y):
    """Least squares, plus R-squared and the largest residual, in y's own units."""
    design = np.asarray(design, dtype=float)
    y = np.asarray(y, dtype=float)
    coef, _res, _rank, _sv = np.linalg.lstsq(design, y, rcond=None)
    pred = design.dot(coef)
    resid = y - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return coef, r2, float(np.max(np.abs(resid)))


def _reach(t):
    """How far past the lowest fitted point the extrapolation runs, as a fraction.

    The fit is built on total errors between min(t) and max(t) and then read at 0.
    min(t) / (max(t) - min(t)) says how far outside its own range that is.
    """
    t = np.asarray(t, dtype=float)
    span = float(t.max() - t.min())
    return float(t.min() / span) if span > 0 else float('nan')


def fit_var_linear(curve, b=None):
    """score = ceiling + slope * (total error)**2. Two parameters.

    Variances add, so this is the form the quadrature argument points at: the score
    moves with the added VARIANCE, to first order. Note what that costs -- with
    t**2 = b**2 + L**2 the b**2 term is a constant, so b only shifts the intercept and
    the curve carries no information about b at all. The published figure is doing all
    the work of placing the ceiling under this form.
    """
    bb = curve.b if b is None else b
    _why = _unfittable(curve.levels, curve.scores, MIN_POINTS_2P)
    if _why:
        return Fit('var_linear', None, None, None, None, None, len(curve.levels), None,
                   refused=_why)
    if bb is None and not curve.level_is_a_fraction_of_labels:
        return Fit('var_linear', None, None, None, None, None, len(curve.levels), None,
                   refused='no baseline error recorded for this dataset, so the '
                           'ceiling cannot be placed at one number')
    t = curve.totals(bb)
    y = np.asarray(curve.scores, dtype=float)
    design = np.column_stack([np.ones_like(t), t ** 2])
    coef, r2, mx = _ols(design, y)
    ceiling = float(coef[0])
    at_zero_level = float(coef[0] + coef[1] * (t.min() ** 2))
    return Fit('var_linear', ceiling, at_zero_level, float(coef[1]), r2, mx,
               len(curve.levels), _reach(t))


def fit_t_linear(curve, b=None):
    """score = ceiling + slope * (total error). Two parameters.

    No quadrature assumption about how the score responds -- it is here so a
    disagreement with the variance-linear fit is visible. If the two ceilings are far
    apart the curve is not well described by either and the extrapolation is soft.
    """
    bb = curve.b if b is None else b
    _why = _unfittable(curve.levels, curve.scores, MIN_POINTS_2P)
    if _why:
        return Fit('t_linear', None, None, None, None, None, len(curve.levels), None,
                   refused=_why)
    if bb is None and not curve.level_is_a_fraction_of_labels:
        return Fit('t_linear', None, None, None, None, None, len(curve.levels), None,
                   refused='no baseline error recorded for this dataset')
    t = curve.totals(bb)
    y = np.asarray(curve.scores, dtype=float)
    design = np.column_stack([np.ones_like(t), t])
    coef, r2, mx = _ols(design, y)
    return Fit('t_linear', float(coef[0]), float(coef[0] + coef[1] * t.min()),
               float(coef[1]), r2, mx, len(curve.levels), _reach(t))


def fit_var_quadratic(curve, b=None):
    """score = ceiling + a * t**2 + c * t**4. Three parameters on six points.

    Buys curvature, which the two-parameter fits cannot express, at the cost of one
    degree of freedom. Where it disagrees with var_linear the curve is bending and the
    straight extrapolation is reading through the bend.
    """
    bb = curve.b if b is None else b
    _why = _unfittable(curve.levels, curve.scores, MIN_POINTS_3P)
    if _why:
        return Fit('var_quadratic', None, None, None, None, None, len(curve.levels),
                   None, refused=_why)
    if bb is None and not curve.level_is_a_fraction_of_labels:
        return Fit('var_quadratic', None, None, None, None, None, len(curve.levels),
                   None, refused='no baseline error recorded for this dataset')
    t = curve.totals(bb)
    y = np.asarray(curve.scores, dtype=float)
    design = np.column_stack([np.ones_like(t), t ** 2, t ** 4])
    coef, r2, mx = _ols(design, y)
    ceiling = float(coef[0])
    at_zero_level = float(coef[0] + coef[1] * t.min() ** 2 + coef[2] * t.min() ** 4)
    return Fit('var_quadratic', ceiling, at_zero_level, float(coef[1]), r2, mx,
               len(curve.levels), _reach(t), extra={'t4_coefficient': float(coef[2])})


def fit_sq_quadrature(curve, b=None):
    """score**2 = ceiling**2 + slope * (total error)**2. Two parameters.

    For an error metric -- rmse, mae -- the metric is itself a spread, so two
    independent sources of error combine in it the same way they combine anywhere else.
    The fit is a straight line in the SQUARED score, and the ceiling is the square root
    of its intercept. Defined only where every score is positive, and undefined where
    the fitted intercept comes out negative, which says the curve does not have this
    shape rather than that the ceiling is imaginary.
    """
    bb = curve.b if b is None else b
    if not curve.lower_is_better:
        return Fit('sq_quadrature', None, None, None, None, None, len(curve.levels),
                   None, refused='only defined on an error metric (lower is better)')
    _why = _unfittable(curve.levels, curve.scores, MIN_POINTS_2P)
    if _why:
        return Fit('sq_quadrature', None, None, None, None, None, len(curve.levels),
                   None, refused=_why)
    if bb is None and not curve.level_is_a_fraction_of_labels:
        return Fit('sq_quadrature', None, None, None, None, None, len(curve.levels),
                   None, refused='no baseline error recorded for this dataset')
    y = np.asarray(curve.scores, dtype=float)
    if np.any(y <= 0):
        return Fit('sq_quadrature', None, None, None, None, None, len(curve.levels),
                   None, refused='a score of zero or below; squaring loses its sign')
    t = curve.totals(bb)
    design = np.column_stack([np.ones_like(t), t ** 2])
    coef, r2, mx_sq = _ols(design, y ** 2)
    if coef[0] <= 0:
        return Fit('sq_quadrature', None, None, float(coef[1]), r2, None,
                   len(curve.levels), _reach(t),
                   refused='the fitted intercept is %.4g, at or below zero, so the '
                           'curve does not have this shape' % float(coef[0]))
    ceiling = float(math.sqrt(coef[0]))
    at_zero_level = float(math.sqrt(max(coef[0] + coef[1] * t.min() ** 2, 0.0)))
    # Residuals were fitted on the squared score; report them back in the score's own
    # units so they can be read against the ceiling.
    pred = np.sqrt(np.clip(design.dot(coef), 0.0, None))
    mx = float(np.max(np.abs(y - pred)))
    return Fit('sq_quadrature', ceiling, at_zero_level, float(coef[1]), r2, mx,
               len(curve.levels), _reach(t),
               extra={'max_residual_on_squared_score': mx_sq})


def fit_free_baseline_error(curve):
    """score = ceiling + slope * sqrt(bhat**2 + L**2). Three parameters.

    This is the one fit that asks the curve what b is instead of being told. It works
    only because the square root has curvature near L = 0: how rounded the bottom of
    the curve is says how much error was already there. Under the variance-linear form
    it would not work at all -- there b**2 is absorbed into the intercept and vanishes.

    bhat comes back in fractions of the clean label spread, the same units as the
    published b, so the two can be printed beside each other. Three parameters on six
    points, so bhat is the softest number this script produces.
    """
    if curve.level_is_a_fraction_of_labels:
        return Fit('free_b', None, None, None, None, None, len(curve.levels), None,
                   refused='the level counts labels changed, not an amount of error')
    _why = _unfittable(curve.levels, curve.scores, MIN_POINTS_3P)
    if _why:
        return Fit('free_b', None, None, None, None, None, len(curve.levels), None,
                   refused=_why)
    try:
        from scipy.optimize import curve_fit as _curve_fit
    except ImportError as e:
        return Fit('free_b', None, None, None, None, None, len(curve.levels), None,
                   refused='scipy is not installed here: %s' % e)
    L = np.asarray(curve.levels, dtype=float)
    y = np.asarray(curve.scores, dtype=float)

    def _model(x, ceiling, slope, bhat):
        return ceiling + slope * np.sqrt(bhat ** 2 + x ** 2)

    span = float(np.max(y) - np.min(y))
    slope0 = span / max(float(L.max() - L.min()), 1e-9)
    if not curve.lower_is_better:
        slope0 = -abs(slope0)
    try:
        p, _cov = _curve_fit(_model, L, y, p0=[float(y[0]), slope0, 0.3],
                             maxfev=20000,
                             bounds=([-np.inf, -np.inf, 1e-4],
                                     [np.inf, np.inf, 5.0]))
    except Exception as e:
        return Fit('free_b', None, None, None, None, None, len(curve.levels), None,
                   refused='the fit did not converge: %s' % e)
    ceiling, slope, bhat = float(p[0]), float(p[1]), float(p[2])
    pred = _model(L, *p)
    resid = y - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    t = np.sqrt(bhat ** 2 + L ** 2)
    return Fit('free_b', ceiling, float(ceiling + slope * bhat), slope, r2,
               float(np.max(np.abs(resid))), len(curve.levels), _reach(t),
               extra={'baseline_error_fraction_of_spread_from_the_curve': bhat})


ALL_FORMS = ['var_linear', 't_linear', 'var_quadratic', 'sq_quadrature', 'free_b']


def fit_all(curve):
    return {
        'var_linear': fit_var_linear(curve),
        't_linear': fit_t_linear(curve),
        'var_quadratic': fit_var_quadratic(curve),
        'sq_quadrature': fit_sq_quadrature(curve),
        'free_b': fit_free_baseline_error(curve),
    }


def ceiling_over_a_range_of_b(curve, b_values):
    """What the variance-linear ceiling would be for each candidate b.

    For the thirty-nine datasets with no published assay error. The slope does not
    depend on b at all -- shifting b only shifts where the fit is read -- so this is
    one fit read at several points, not several fits.
    """
    if curve.level_is_a_fraction_of_labels:
        return None
    if _unfittable(curve.levels, curve.scores, MIN_POINTS_2P):
        return None
    L = np.asarray(curve.levels, dtype=float)
    y = np.asarray(curve.scores, dtype=float)
    design = np.column_stack([np.ones_like(L), L ** 2])
    coef, _r2, _mx = _ols(design, y)
    intercept_at_level_zero, slope = float(coef[0]), float(coef[1])
    return {
        'fitted_score_at_level_zero': intercept_at_level_zero,
        'slope_per_unit_variance': slope,
        'ceiling_by_b': {float(b): float(intercept_at_level_zero - slope * b * b)
                         for b in b_values},
    }


def curve_robustness_cross_reference(curve):
    """What noiseInject's own curve scalars say, where its metric rules allow them.

    Printed as a cross-reference, not as a ceiling. Every one of them is computed from
    score(level) / score(0), so all of them are 1 at level 0 by construction and none
    of them can reach above the level-0 score.
    """
    try:
        from noiseInject.metrics import (_curve_robustness,
                                         _CURVE_METRICS_REGRESSION,
                                         _CURVE_METRICS_CLASSIFICATION)
    except Exception as e:
        return {'unavailable': str(e)}
    allowed = set(_CURVE_METRICS_REGRESSION) | set(_CURVE_METRICS_CLASSIFICATION)
    if (curve.metric or '') not in allowed:
        return {'not_defined_on_this_metric': curve.metric,
                'defined_on': sorted(allowed)}
    x = np.asarray(curve.levels, dtype=float)
    y = np.asarray(curve.scores, dtype=float)
    if 0.0 not in set(x.tolist()):
        return {'no_level_zero_score': 'the retention curve divides by it'}
    base = float(y[int(np.argmin(np.abs(x)))])
    return _curve_robustness(x, y, base, None)


# ---------------------------------------------------------------------------
# Printing.
# ---------------------------------------------------------------------------

def _fmt(v, nd=4):
    if v is None:
        return '--'
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return 'nan' if not math.isfinite(f) else ('%.*f' % (nd, f))


def print_curve(curve, forms=ALL_FORMS, b_grid=(0.2, 0.3, 0.4, 0.5, 0.6)):
    print('')
    print('=' * 78)
    print(curve.label())
    print('=' * 78)
    direction = ('lower is better, so the ceiling is the SMALLEST value'
                 if curve.lower_is_better else
                 'higher is better, so the ceiling is the LARGEST value')
    print('  Score is %s on %s; %s.' % (curve.metric, curve.dataset, direction))
    print('  Read from %s' % curve.source_file)

    if curve.level_is_a_fraction_of_labels:
        print('  The level here counts LABELS changed, not an amount of error, so it')
        print('  adds in quadrature with nothing. Every fit below runs back to level 0,')
        print('  which is the score on the labels as they come -- not a ceiling.')
    elif curve.b is None:
        print('  No published assay error for this dataset, so how much error the')
        print('  labels were already carrying is unknown and the ceiling cannot be put')
        print('  at one number. See the table of candidate values below.')
    else:
        sd = curve.label_spread
        print('  The labels were already carrying %s of their own spread before'
              % _fmt(curve.b))
        print('  anything was added. That is %s in the label\'s own units, over a label'
              % _fmt(curve.published_error_units))
        print('  spread of %s. Source: %s' % (_fmt(sd), curve.b_source))

    print('')
    if curve.x_counts_labels:
        print('  One row is one share of the molecules the level was delivered to. The')
        print('  share is a fraction of the training molecules; the score is in %s.'
              % curve.metric)
        print('    %-14s %-16s %s' % ('share reached', '(no total)',
                                      'score (%s)' % curve.metric))
    else:
        print('  One row is one added-error level. Levels and total error are fractions of')
        print('  the clean label spread; the score is in %s.' % curve.metric)
        print('    %-14s %-16s %s' % ('added error', 'total error',
                                      'score (%s)' % curve.metric))
    t = curve.totals()
    for L, tt, s in zip(curve.levels, t, curve.scores):
        print('    %-14s %-16s %s' % (_fmt(L, 3), _fmt(tt, 4), _fmt(s)))

    print('')
    if curve.x_counts_labels:
        print('  Each fit runs the curve back to a share of zero. Both columns are the')
        print('  score on labels nothing reached; neither is a ceiling, because no')
        print('  amount of error has been taken out of them.')
        col_a, col_b = 'at share 0', 'at share 0'
    else:
        print('  Each fit runs the curve back to zero total error. "at level 0" is the')
        print('  fitted score where nothing was added; "ceiling" is the extrapolation.')
        col_a, col_b = 'at level 0', 'ceiling'
    print('    %-15s %-11s %-11s %-11s %-8s %-9s %s'
          % ('form', col_a, col_b, 'difference', 'fit R2', 'max resid', 'note'))
    fits = fit_all(curve)
    for name in forms:
        f = fits.get(name)
        if f is None:
            continue
        if f.refused:
            print('    %-15s %-11s %-11s %-11s %-8s %-9s %s'
                  % (name, '--', '--', '--', '--', '--', f.refused))
            continue
        note = ''
        if f.reach is not None and math.isfinite(f.reach):
            note = 'reads %s of its own range past the lowest point' % _fmt(f.reach, 2)
        if name == 'free_b':
            bhat = f.extra.get('baseline_error_fraction_of_spread_from_the_curve')
            note = 'baseline error from the curve: %s of the label spread' % _fmt(bhat, 4)
        print('    %-15s %-11s %-11s %-11s %-8s %-9s %s'
              % (name, _fmt(f.at_level_zero), _fmt(f.ceiling),
                 _fmt(f.cost_of_baseline_error), _fmt(f.r_squared, 4),
                 _fmt(f.max_residual), note))

    measured0 = None
    if curve.x_counts_labels:
        print('')
        print('  The two score columns above differ only because the fits disagree with')
        print('  each other about the shape near zero; there is no baseline error being')
        print('  subtracted here.')
    elif curve.levels and abs(curve.levels[0]) < 1e-12:
        measured0 = curve.scores[0]
        print('')
        print('  Measured at level 0: %s %s. Compare it with the "at level 0" column'
              % (_fmt(measured0), curve.metric))
        print('  above -- that gap is how well each fit describes the bottom of the')
        print('  curve, which is where the extrapolation starts.')
    else:
        print('')
        print('  This curve has no level-0 score, so there is nothing measured to')
        print('  compare the fitted "at level 0" column against.')
        print('  Lowest level fitted: %s of the label spread.' % _fmt(curve.levels[0], 3))

    if curve.b is None and not curve.level_is_a_fraction_of_labels:
        table = ceiling_over_a_range_of_b(curve, b_grid)
        if table:
            print('')
            print('  With no published figure, the ceiling depends on how much error')
            print('  the labels carry. One row is one candidate value, in fractions of')
            print('  the clean label spread, under the variance-linear fit.')
            print('  The fit puts the score at added error 0 at %s %s, and every row'
                  % (_fmt(table['fitted_score_at_level_zero']), curve.metric))
            print('  below subtracts from it.')
            print('    %-28s %s' % ('labels already carrying', 'ceiling (%s)' % curve.metric))
            for b, c in sorted(table['ceiling_by_b'].items()):
                print('    %-28s %s' % (_fmt(b, 2), _fmt(c)))

    xref = curve_robustness_cross_reference(curve)
    print('')
    if 'auc_norm' in xref:
        print('  noiseInject\'s own curve scalars, for cross-reference only: '
              'auc_norm %s,' % _fmt(xref.get('auc_norm')))
        print('  weibull_beta %s, weibull_tau %s. All of them divide by the level-0'
              % (_fmt(xref.get('weibull_beta')), _fmt(xref.get('weibull_tau'))))
        print('  score, so none of them can reach above it.')
    elif 'not_defined_on_this_metric' in xref:
        print('  noiseInject\'s curve scalars are not defined on %s (they need a '
              'bounded,' % curve.metric)
        print('  higher-is-better metric: %s), so none is printed.'
              % ', '.join(xref['defined_on']))
    else:
        print('  noiseInject\'s curve scalars: %s' % xref)


def published_side_by_side(curves):
    """The published assay error and the one the curve gives, printed beside each other.

    Neither checks the other. One is a figure from a paper, in the label's own units
    and converted to a fraction of this repository's label spread. The other is the
    third parameter of a three-parameter fit on six points. They are different
    quantities measured different ways and the table shows both.
    """
    rows = []
    for c in curves:
        if c.b is None:
            continue
        f = fit_free_baseline_error(c)
        rows.append({
            'curve': c.label(),
            'metric': c.metric,
            'published_error_label_units': c.published_error_units,
            'published_b_fraction_of_spread': c.b,
            'b_from_the_curve': (None if f.refused else
                                 f.extra.get('baseline_error_fraction_of_spread_from_the_curve')),
            'ceiling_from_published_b': fit_var_linear(c).ceiling,
            'ceiling_from_the_curve': (None if f.refused else f.ceiling),
            'refused': f.refused,
        })
    return rows


def print_published_side_by_side(rows):
    if not rows:
        print('\nNo curve in this run is on a dataset with a published assay error, so')
        print('there is no side-by-side table. dataset_policy.REFERENCE_ERROR_LABEL_UNITS')
        print('holds a figure for herg_ki, caco2 and logd and for no other dataset.')
        return
    print('')
    print('=' * 78)
    print('PUBLISHED FIGURE BESIDE THE CURVE\'S OWN')
    print('=' * 78)
    print('  One row is one curve. Neither column checks the other: the published')
    print('  column is a number from a paper, the curve column is the third parameter')
    print('  of a three-parameter fit on six points. Both are fractions of the clean')
    print('  label spread. The two ceilings are in the curve\'s own metric.')
    print('    %-44s %-12s %-12s %-12s %s'
          % ('curve', 'published b', 'b from curve', 'ceiling(pub)', 'ceiling(curve)'))
    for r in rows:
        print('    %-44s %-12s %-12s %-12s %s'
              % (r['curve'][:44], _fmt(r['published_b_fraction_of_spread']),
                 _fmt(r['b_from_the_curve']), _fmt(r['ceiling_from_published_b']),
                 _fmt(r['ceiling_from_the_curve'])))
        if r['refused']:
            print('      %s' % r['refused'])


def as_json(curves):
    out = []
    for c in curves:
        fits = fit_all(c)
        out.append({
            'dataset': c.dataset,
            'experiment': c.experiment,
            'condition': c.condition,
            'what': c.what,
            'metric': c.metric,
            'lower_is_better': c.lower_is_better,
            'source_file': c.source_file,
            'levels_added_error_fraction_of_spread': c.levels,
            'total_error_fraction_of_spread': [float(v) for v in c.totals()],
            'scores': c.scores,
            'baseline_error_fraction_of_spread': c.b,
            'baseline_error_label_units': c.published_error_units,
            'baseline_error_source': c.b_source,
            'level_counts_labels_not_error': c.level_is_a_fraction_of_labels,
            'fits': {k: ({'refused': f.refused} if f.refused else {
                'score_at_total_error_zero_the_ceiling': f.ceiling,
                'fitted_score_at_added_error_zero': f.at_level_zero,
                'cost_of_the_baseline_error': f.cost_of_baseline_error,
                'slope': f.slope,
                'fit_r_squared': f.r_squared,
                'max_residual_in_score_units': f.max_residual,
                'n_levels': f.n_points,
                'extrapolation_reach': f.reach,
                'extra': f.extra,
            }) for k, f in fits.items()},
        })
    return out


# ---------------------------------------------------------------------------
# The entry point the noise run calls. It must never fail the job.
# ---------------------------------------------------------------------------

def write_readout(input_dir, dataset=None, output_dir=None, verbose=True):
    """Read every curve, write the numbers and the printed read-out, return the path.

    Called at the end of `noise_helpers.run_all`, so the ceiling arrives with the
    experiments instead of being a step somebody has to remember. It is a reading of
    files already on disk -- it trains nothing and loads no dataset, so it costs
    seconds at the end of a job that ran for hours.

    NOTHING IN HERE MAY RAISE. A job that has just spent a day on the noise suite must
    not lose it to an arithmetic error in a read-out. Every failure is caught, printed
    and written into the output, and the function returns None.

    Two files per dataset, under <input_dir>/noise_ceiling/:
      <dataset>.json  every number, including the refusals and why
      <dataset>.txt   the same thing as the script prints it
    """
    try:
        out_root = Path(output_dir or input_dir) / 'noise_ceiling'
        out_root.mkdir(parents=True, exist_ok=True)
        stem = dataset or 'all_datasets'
        json_path = out_root / ('%s.json' % stem)
        txt_path = out_root / ('%s.txt' % stem)

        curves, notes = read_curves(input_dir, dataset_filter=dataset)

        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print('Noise ceiling read-out')
            print('Read %d curve(s) from %s' % (len(curves), input_dir))
            for n in notes:
                print('  %s' % n)
            for c in curves:
                try:
                    print_curve(c)
                except Exception as e:
                    print('\n  %s: could not be printed -- %s: %s'
                          % (c.label(), type(e).__name__, e))
            try:
                print_published_side_by_side(published_side_by_side(curves))
            except Exception as e:
                print('\n  the side-by-side table could not be built -- %s: %s'
                      % (type(e).__name__, e))
        text = buf.getvalue()
        txt_path.write_text(text)

        payload = {
            'dataset': dataset,
            'input_dir': str(input_dir),
            'n_curves': len(curves),
            'notes': notes,
            'level_grid_is': 'fractions of the clean label spread',
            'curves': as_json(curves),
        }
        json_path.write_text(json.dumps(payload, indent=2, default=str))
        if verbose:
            print('  [noise ceiling] %d curve(s) -> %s and %s'
                  % (len(curves), json_path, txt_path))
            for n in notes:
                print('  [noise ceiling] %s' % n)
        return json_path
    except Exception as e:
        import traceback
        print('  [noise ceiling] the read-out failed and the run continues -- '
              '%s: %s' % (type(e).__name__, e))
        traceback.print_exc()
        try:
            out_root = Path(output_dir or input_dir) / 'noise_ceiling'
            out_root.mkdir(parents=True, exist_ok=True)
            (out_root / ('%s.FAILED.txt' % (dataset or 'all_datasets'))).write_text(
                '%s: %s\n\n%s' % (type(e).__name__, e, traceback.format_exc()))
        except Exception:
            pass
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--input', default=DEFAULT_INPUT,
                    help='directory of noise result files (default: %s)' % DEFAULT_INPUT)
    ap.add_argument('--dataset', default=None, help='one dataset only')
    ap.add_argument('--experiment', default=None,
                    help='one experiment only, by its short id (c10, n09, n11, n12, n13, n15)')
    ap.add_argument('--json', default=None, help='also write every number to this file')
    ap.add_argument('--max-curves', type=int, default=None,
                    help='print at most this many curves (the JSON always holds all)')
    args = ap.parse_args(argv)

    curves, notes = read_curves(args.input, dataset_filter=args.dataset)
    if args.experiment:
        curves = [c for c in curves if c.experiment == args.experiment]

    print('Read %d curve(s) from %s' % (len(curves), args.input))
    for n in notes:
        print('  %s' % n)

    shown = curves if args.max_curves is None else curves[:args.max_curves]
    for c in shown:
        print_curve(c)
    if len(shown) < len(curves):
        print('\n%d further curve(s) not printed (--max-curves). They are in the JSON.'
              % (len(curves) - len(shown)))

    print_published_side_by_side(published_side_by_side(curves))

    if args.json:
        Path(args.json).write_text(json.dumps(as_json(curves), indent=2))
        print('\n[wrote %s]' % args.json)
    return 0


if __name__ == '__main__':
    sys.exit(main())
