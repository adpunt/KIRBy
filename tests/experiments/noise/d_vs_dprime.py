"""The main measurement of every noise experiment: a model fitted on D and on D'.

Plan section 2. D is the training fold as the experiment receives it -- noisy
labels, every molecule. D' is D after one mitigation method. Both are fitted
with the same downstream model and scored on the same held-out fold, and the
model fitted on the CLEAN training labels is scored there too, so every row
carries the three numbers in the dataset's own metric:

  score_D       fitted on D
  score_Dprime  fitted on D'
  score_clean   fitted on the clean labels of the same training molecules

`gain` is score_Dprime - score_D when higher is better and score_D -
score_Dprime when lower is better, so a positive gain always means the method
helped. The direction comes from get_domain_metric's lower_is_better
(_shared.py:2510) and is never assumed; make_row refuses a row without one.

Rows are appended to a JSONL file as each fold finishes, flushed and fsynced,
so a killed run keeps every fold it finished. done_keys reads the file back and
a rerun skips any row whose key is already there.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# The columns of one output row, in order. Plan section 2, plus `gain`.
ROW_FIELDS = ('dataset', 'experiment', 'method', 'method_setting',
              'similarity_setting', 'model_input', 'model', 'noise_condition',
              'level', 'draw', 'fold', 'metric', 'lower_is_better',
              'score_D', 'score_Dprime', 'score_clean', 'gain')

# Written only when it applies. dprime_not_run: why score_Dprime is nan when
# the method did run but its output could not be fitted (2026-10-07: a
# weighting cleanup on a model whose fit takes no weight, noise_cache
# WEIGHTS_NOT_RUN).
OPTIONAL_ROW_FIELDS = ('dprime_not_run',)

# The columns that identify a row. A rerun skips a row whose key is on disk.
KEY_FIELDS = ('dataset', 'experiment', 'method', 'method_setting',
              'similarity_setting', 'model_input', 'model', 'noise_condition',
              'level', 'draw', 'fold')

# The downstream models the D-versus-D' fits use. Names in DOWNSTREAM_MODELS
# (_shared.py:565); factories from get_model_factories.
DEFAULT_MODELS = ('rf', 'svm')


def gain(score_D, score_Dprime, lower_is_better):
    """Positive when D' beat D, in the metric's own units."""
    if lower_is_better is None:
        raise ValueError('lower_is_better is None: the direction of the metric '
                         'must come from get_domain_metric, never be assumed')
    a, b = float(score_D), float(score_Dprime)
    if not (math.isfinite(a) and math.isfinite(b)):
        return float('nan')
    return (a - b) if bool(lower_is_better) else (b - a)


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def make_row(**fields):
    """One output row. Every field of ROW_FIELDS except `gain` is required;
    gain is computed here. Extra fields are refused, so a typo cannot create a
    column nobody reads."""
    missing = [f for f in ROW_FIELDS if f != 'gain' and f not in fields]
    extra = [f for f in fields
             if f not in ROW_FIELDS and f not in OPTIONAL_ROW_FIELDS]
    if missing or extra:
        raise ValueError(f'make_row: missing {missing}, unknown {extra}')
    if fields['lower_is_better'] is None:
        raise ValueError('make_row: lower_is_better is None')
    row = {f: _jsonable(fields.get(f)) for f in ROW_FIELDS if f != 'gain'}
    row['lower_is_better'] = bool(row['lower_is_better'])
    row['gain'] = gain(row['score_D'], row['score_Dprime'],
                       row['lower_is_better'])
    for f in OPTIONAL_ROW_FIELDS:
        if fields.get(f) is not None:
            row[f] = _jsonable(fields[f])
    return row


def row_key(row):
    """The tuple that identifies a row. Each value is JSON-encoded, so a level
    of 0.1 written and read back is the same key."""
    return tuple(json.dumps(_jsonable(row.get(f)), sort_keys=True)
                 for f in KEY_FIELDS)


def append_rows(path, rows):
    """Append rows to a JSONL file, flushed and fsynced before returning."""
    rows = list(rows)
    if not rows:
        return 0
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, 'a') as fh:
        for r in rows:
            fh.write(json.dumps({k: _jsonable(v) for k, v in r.items()},
                                allow_nan=True) + '\n')
        fh.flush()
        os.fsync(fh.fileno())
    return len(rows)


def done_keys(path):
    """The keys of every complete row already in `path`. A last line cut off by
    a kill is skipped, not raised on."""
    out = set()
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.add(row_key(json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
    return out


def _fit_score(dataset, model_name, factory, X_tr, y_tr, X_va, y_va, metric_name,
               metric_fn, sample_weight=None):
    from experiments.noise.noise_cache import cached_fit_predict, predict_mode_for
    pred = cached_fit_predict(dataset, model_name, factory, X_tr, y_tr, X_va,
                              sample_weight=sample_weight,
                              predict_mode=predict_mode_for(metric_name))
    return float(metric_fn(y_va, pred))


def score_d_dprime(dataset, cleaned, X_tr, y_tr_D, X_va, y_va, *, metric,
                   task_type, y_tr_clean=None, models=DEFAULT_MODELS):
    """score_D, score_Dprime and score_clean per downstream model.

    Args:
        dataset: the dataset name, for the cache path only.
        cleaned: what n07.cleaned_training_set returned for D' -- keys X, y,
            sample_weight -- or None when the method declined, which gives
            score_Dprime nan.
        X_tr, y_tr_D: D, the training matrix and the labels as the experiment
            received them.
        X_va, y_va: the held-out fold. y_va are the clean held-out labels.
        metric: the (metric_name, metric_fn, lower_is_better) triple
            get_domain_metric(data) returns.
        task_type: 'regression' or 'classification', for the model factories.
        y_tr_clean: the clean training labels; None gives score_clean nan.
        models: downstream model names; factories come from
            _shared.get_model_factories.

    Every fit goes through noise_cache.cached_fit_predict, so the D fit and the
    clean fit are computed once per (fold, features, labels, model) however
    many methods are scored against them.

    Returns {model_name: {'score_D', 'score_Dprime', 'score_clean'}}.
    """
    from experiments._shared import get_model_factories
    from experiments.noise.noise_cache import (fit_takes_sample_weight,
                                               WEIGHTS_NOT_RUN)
    metric_name, metric_fn, _lib = metric
    factories = get_model_factories(list(models), task_type,
                                    n_samples=len(y_tr_D))
    out = {}
    for name in models:
        f = factories.get(name)
        if f is None:
            continue
        s_D = _fit_score(dataset, name, f, X_tr, y_tr_D, X_va, y_va,
                         metric_name, metric_fn)
        not_run = None
        if cleaned is None:
            s_P = float('nan')
        elif (cleaned.get('sample_weight') is not None
              and not fit_takes_sample_weight(f)):
            # 2026-10-07: gp takes no per-molecule weight. Recorded, not
            # raised; before this the unit failed here.
            s_P, not_run = float('nan'), WEIGHTS_NOT_RUN
        else:
            s_P = _fit_score(dataset, name, f, cleaned['X'], cleaned['y'], X_va,
                             y_va, metric_name, metric_fn,
                             sample_weight=cleaned.get('sample_weight'))
        s_C = (float('nan') if y_tr_clean is None else
               _fit_score(dataset, name, f, X_tr, y_tr_clean, X_va, y_va,
                          metric_name, metric_fn))
        out[name] = {'score_D': s_D, 'score_Dprime': s_P, 'score_clean': s_C}
        if not_run:
            out[name]['dprime_not_run'] = not_run
    return out


def rows_for(scores_by_model, *, metric, **identity):
    """Output rows from score_d_dprime's result. `identity` holds every
    KEY_FIELDS entry but `model`; `metric` is the get_domain_metric triple."""
    metric_name, _fn, lower_is_better = metric
    return [make_row(**identity, model=m, metric=metric_name,
                     lower_is_better=lower_is_better, **s)
            for m, s in scores_by_model.items()]
