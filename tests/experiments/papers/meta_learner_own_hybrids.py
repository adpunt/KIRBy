#!/usr/bin/env python3
"""A meta-learner over models that each have their own hybrid. One outer fold.

    python tests/experiments/papers/meta_learner_own_hybrids.py DATASET --fold 2 \
        --candidates rf,dnn,lightgbm,svm,gp,xgboost,ngboost --inner-folds K \
        --min-rows N --output <run>/meta_learner

Adelaide's design, 2026-11-05. Stacking (Wolpert 1992), with one search per
model, so each model has its own recipe and its own feature set.

On one outer fold:

1.  Search every candidate model on the fold's training rows. A candidate that
    is one of the dataset's two recorded models reads the ledger's own search.
    Any other candidate has a search of its own, filed under its own name, so it
    cannot overwrite the two-model search the other experiments read.
2.  Take the top 5 by the score each search gives its best recipe, on those same
    training rows.
3.  Run each model's best recipe once, on all the training folds. That gives one
    fixed feature set per model (Adelaide, 2026-11-05), used by every fit below.
4.  Split the training folds into inner folds. On each, fit the model on the fixed
    features of the inner training folds and predict the inner held-out fold.
    Every sample in the training folds ends with one prediction per model.
5.  Refit each model on the same fixed features over all the training folds and
    predict the outer held-out fold.
6.  Fit every meta-learner on the predictions from step 4 and score it on the
    predictions from step 5.

WHAT THIS MEANS, and the Methods section has to say so: the recipe, the fixed
features and the top-5 pick all saw the labels of the inner held-out folds whose
predictions the meta-learner trains on. The outer held-out fold is seen by nothing
until the final score.

Every meta-learner's test score is written. The one that counts has the best
cross-validated score on the training folds' predictions (Adelaide, 2026-11-06),
so it is never picked by its test score.

Compared with, on the same outer fold (Adelaide, 2026-11-06):
  best_model_hybrid     the best candidate's hybrid, by its search score
  plain_average_of_5    the 5 models' predictions averaged
  best_model_base_pair  the best model and single representation over every
                        candidate, by the search score, fitted at full width
  c11's ensembles       written by c11, joined by meta_learner_summary.py

The candidate list, the inner fold count and the minimum row count are required
arguments; the ledger fills them from RUN_SETTINGS.json.

The search space and pruning come from the unit's environment, the same as the
ledger's search steps (KIRBY_SEARCH_SPACE, KIRBY_SEARCH_PRUNING). The ledger runs
this in the cache of the search the experiments read.

Each model's predictions are saved after each fit, in
<output>/<dataset>__meta_f<k>.predictions/, so a job killed at its time limit
resumes, and adding a meta-learner later fits no downstream model again.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent.parent
REPO = TESTS.parent
for p in (REPO / 'src', TESTS, TESTS / 'experiments'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np   # noqa: E402

SEED = 42
#: Adelaide, 2026-11-05: "take N=5 models"; "the top 5".
TOP_N_MODELS = 5
RESULT_SCHEMA = 'meta_learner_own_hybrids_2026_11_05'


# ── The meta-learners ──────────────────────────────────────────────────────
#
# Input: X, one row per sample, one column per model (regression) or per model
# and class (classification, K columns for each model, in `classes` order).
# Output of predict: n values (regression) or an n x K probability matrix.
#
# The families are Adelaide's list of 2026-11-05: linear regression with
# constraints or weighted averages, ridge regression, gradient boosted trees,
# random forests, small nets. The settings inside each family are NOT hers. They
# are in META_LEARNERS below so they can be read and changed, and every one runs.


def _uniform_weights(n_models):
    return np.full(n_models, 1.0 / n_models)


class EqualWeights:
    """The five predictions averaged."""

    def __init__(self, n_classes=None):
        self.k = n_classes

    def fit(self, X, y):
        return self

    def predict(self, X):
        if self.k is None:
            return X.mean(axis=1)
        return X.reshape(len(X), -1, self.k).mean(axis=1)


class SimplexWeights:
    """One weight per model, each at least 0, summing to 1. Fitted to squared
    error (regression) or log loss (classification)."""

    def __init__(self, n_classes=None):
        self.k = n_classes
        self.w = None

    def _blend(self, X, w):
        if self.k is None:
            return X @ w
        return np.einsum('nmk,m->nk', X.reshape(len(X), -1, self.k), w)

    def fit(self, X, y):
        from scipy.optimize import minimize
        n_models = X.shape[1] if self.k is None else X.shape[1] // self.k

        def loss(w):
            p = self._blend(X, w)
            if self.k is None:
                return float(np.mean((p - y) ** 2))
            return -float(np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))))

        res = minimize(loss, _uniform_weights(n_models), method='SLSQP',
                       bounds=[(0, 1)] * n_models,
                       constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1})
        w = np.clip(res.x, 0, None)
        self.w = w / w.sum() if w.sum() > 0 else _uniform_weights(n_models)
        return self

    def predict(self, X):
        return self._blend(X, self.w)


class NonNegativeLinear:
    """Linear regression with every coefficient at least 0, no sum constraint
    (non-negative least squares; Breiman 1996). For classification, one such fit
    per class on that class's probabilities, the output renormalised."""

    def __init__(self, n_classes=None):
        self.k = n_classes
        self.coef = None
        self.intercept = 0.0

    def fit(self, X, y):
        from scipy.optimize import nnls
        if self.k is None:
            mx, my = X.mean(axis=0), y.mean()
            self.coef, _ = nnls(X - mx, y - my)
            self.intercept = my - mx @ self.coef
            return self
        cube = X.reshape(len(X), -1, self.k)
        self.coef = np.stack([nnls(cube[:, :, c], (y == c).astype(float))[0]
                              for c in range(self.k)])
        return self

    def predict(self, X):
        if self.k is None:
            return X @ self.coef + self.intercept
        cube = X.reshape(len(X), -1, self.k)
        p = np.stack([cube[:, :, c] @ self.coef[c] for c in range(self.k)], axis=1)
        p = np.clip(p, 0, None)
        total = p.sum(axis=1, keepdims=True)
        return np.where(total > 0, p / np.where(total > 0, total, 1), 1.0 / self.k)


class _Sklearn:
    """A sklearn-shaped estimator, optionally standardising its input first."""

    def __init__(self, make, n_classes=None, scale=False):
        self.make, self.k, self.scale = make, n_classes, scale

    def fit(self, X, y):
        self.m = self.make()
        if self.scale:
            from sklearn.preprocessing import StandardScaler
            self.s = StandardScaler().fit(X)
            X = self.s.transform(X)
        self.m.fit(X, y)
        return self

    def predict(self, X):
        if self.scale:
            X = self.s.transform(X)
        if self.k is None:
            return np.asarray(self.m.predict(X))
        p = np.asarray(self.m.predict_proba(X))
        out = np.zeros((len(X), self.k))
        seen = getattr(self.m, 'classes_', np.arange(self.k))
        out[:, np.asarray(seen, dtype=int)] = p
        return out


def _ridge(alpha, k):
    from sklearn.linear_model import LogisticRegression, Ridge
    if k is None:
        return _Sklearn(lambda: Ridge(alpha=alpha), k, scale=True)
    return _Sklearn(lambda: LogisticRegression(C=1.0 / alpha, max_iter=2000), k, scale=True)


def _gbt(n_estimators, depth, k):
    from lightgbm import LGBMClassifier, LGBMRegressor
    from kirby.hybrid import lgbm_n_jobs   # n_jobs=-1 segfaults once torch is loaded
    cls = LGBMRegressor if k is None else LGBMClassifier
    return _Sklearn(lambda: cls(n_estimators=n_estimators, max_depth=depth,
                                learning_rate=0.05, min_child_samples=5,
                                random_state=SEED, n_jobs=lgbm_n_jobs(), verbose=-1), k)


def _rf(depth, k):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    cls = RandomForestRegressor if k is None else RandomForestClassifier
    return _Sklearn(lambda: cls(n_estimators=300, max_depth=depth,
                                random_state=SEED, n_jobs=-1), k)


def _net(hidden, k):
    from kirby.models.dnn import DNNClassifier, DNNRegressor
    cls = DNNRegressor if k is None else DNNClassifier
    return _Sklearn(lambda: cls(hidden_sizes=hidden, dropout=0.0), k, scale=True)


META_LEARNERS = {
    'equal_weights': lambda k: EqualWeights(k),
    'simplex_weights': lambda k: SimplexWeights(k),
    'nonnegative_linear': lambda k: NonNegativeLinear(k),
    'ridge_a0.01': lambda k: _ridge(0.01, k),
    'ridge_a0.1': lambda k: _ridge(0.1, k),
    'ridge_a1': lambda k: _ridge(1.0, k),
    'ridge_a10': lambda k: _ridge(10.0, k),
    'ridge_a100': lambda k: _ridge(100.0, k),
    'gbt_n50_d2': lambda k: _gbt(50, 2, k),
    'gbt_n200_d2': lambda k: _gbt(200, 2, k),
    'gbt_n50_d3': lambda k: _gbt(50, 3, k),
    'gbt_n200_d3': lambda k: _gbt(200, 3, k),
    'rf_d3': lambda k: _rf(3, k),
    'rf_d5': lambda k: _rf(5, k),
    'rf_full': lambda k: _rf(None, k),
    'net_8': lambda k: _net((8,), k),
    'net_16_8': lambda k: _net((16, 8), k),
}


def to_metric_space(pred, classes, needs_proba):
    """A meta-learner's output as the metric wants it. Regression passes through;
    classification gives the positive-class probability (or the whole matrix
    with more than two classes) to a probability metric, and hard labels to the
    others."""
    pred = np.asarray(pred)
    if pred.ndim == 1:
        return pred
    if needs_proba:
        return pred[:, 1] if pred.shape[1] == 2 else pred
    return np.asarray(classes)[np.argmax(pred, axis=1)]


def score_meta_learners(X_tr, y_tr, X_te, y_te, inner_splits, *, metric_fn,
                        classes=None, needs_proba=False, learners=None):
    """{name: {'heldout_score', 'train_rows_cv_score', 'fit_s', 'error'}}.

    y_tr and y_te are the labels as they are. For classification the fit uses
    each label's position in `classes`. inner_splits is positions into X_tr.
    """
    k = None if classes is None else len(classes)
    y_fit = y_tr if classes is None else np.searchsorted(classes, y_tr)
    out = {}
    for name, make in (learners or META_LEARNERS).items():
        rec = {'heldout_score': None, 'train_rows_cv_score': None, 'fit_s': None,
               'error': None}
        try:
            t0 = time.time()
            m = make(k).fit(X_tr, y_fit)
            rec['heldout_score'] = float(metric_fn(
                y_te, to_metric_space(m.predict(X_te), classes, needs_proba)))
            rec['fit_s'] = round(time.time() - t0, 3)
            cv = []
            for itr, iva in inner_splits:
                mi = make(k).fit(X_tr[itr], y_fit[itr])
                cv.append(float(metric_fn(
                    y_tr[iva], to_metric_space(mi.predict(X_tr[iva]), classes, needs_proba))))
            rec['train_rows_cv_score'] = float(np.mean(cv)) if cv else None
        except Exception as exc:
            rec['error'] = f'{type(exc).__name__}: {exc}'
        out[name] = rec
    return out


# ── The candidates' searches ───────────────────────────────────────────────

def candidate_grid(model):
    """The run's search space for one model named by hand. The dataset's own
    two models go through the ledger's search instead."""
    from experiments.decided_grid import (fast_space_lists, full_space_lists,
                                          search_space_mode)
    mode = search_space_mode()
    if mode == 'decided':
        raise ValueError('KIRBY_SEARCH_SPACE is not set to full or fast: this runs '
                         'under the ledger, which sets it from RUN_SETTINGS.json')
    lists = full_space_lists('general') if mode == 'full' else fast_space_lists('general')
    lists.pop('aug_conditions', None)
    return dict(lists, models=[model], tier='general')


def candidate_suffix(model, base_suffix=None):
    """Its own search file and its own rows, under its own name."""
    return f'{base_suffix}__meta_{model}' if base_suffix else f'meta_{model}'


def search_candidates(data, labels, tr, fold, n_outer, dataset, candidates,
                      recorded_models, base_suffix=None, log=print):
    """{model: {'entry': best recipe or None, 'score', 'seconds', 'why_none'}}."""
    from experiments._shared import (_attach_evaluator_rep_scores, search_on_fold,
                                     top_base_reps, top_hybrids)
    out = {}
    common = dict(train_idx=np.asarray(tr), fold_i=fold, n_outer_folds=n_outer,
                  dataset_name=dataset, augmentations=data.get('augmentations'),
                  task_type=data.get('task_type') or 'auto', seed=SEED, tier='general')
    recorded_search = None
    for model in candidates:
        t0 = time.time()
        rec = {'entry': None, 'score': None, 'seconds': None, 'why_none': None,
               'searched': 'own', 'base_rep': None, 'base_score': None}
        try:
            if model in recorded_models:
                rec['searched'] = 'the ledger search of the two recorded models'
                if recorded_search is None:
                    recorded_search = search_on_fold(
                        data['reps'], labels, data.get('groups'),
                        condition_suffix=base_suffix, **common)
                srch = recorded_search
            else:
                srch = search_on_fold(
                    data['reps'], labels, data.get('groups'),
                    fixed_grid=candidate_grid(model), run_pruning=True,
                    condition_suffix=candidate_suffix(model, base_suffix), **common)
            s = srch.get('none')
            if s is None:
                fails = getattr(srch, 'failures', None) or []
                rec['why_none'] = '; '.join(str(f.get('error')) for f in fails) or 'search returned nothing'
            else:
                _attach_evaluator_rep_scores(s)
                got = top_hybrids(s, model, n=1, require_full=False,
                                  context=f'{dataset} fold {fold} model {model}')
                if got['selected']:
                    rec['entry'] = got['selected'][0]
                    rec['score'] = float(rec['entry']['score_mean'])
                    rec['lower_is_better'] = bool(s.lower_is_better)
                base = top_base_reps(s, model, n=1, require_full=False,
                                     context=f'{dataset} fold {fold} model {model}')
                if base['selected']:
                    rec['base_rep'] = base['selected'][0]['rep']
                    rec['base_score'] = float(base['selected'][0]['score'])
                else:
                    rec['why_none'] = 'the search scored no configuration for this model'
        except Exception as exc:
            if type(exc).__name__ == 'StaleResultDB':
                raise
            rec['why_none'] = f'{type(exc).__name__}: {exc}'
        rec['seconds'] = round(time.time() - t0, 1)
        log(f'  {dataset} fold {fold} {model}: score {rec["score"]} in {rec["seconds"]} s'
            f'{" (" + rec["why_none"] + ")" if rec["why_none"] else ""}', flush=True)
        out[model] = rec
    return out


def top_models(searches, n=TOP_N_MODELS):
    """The n best candidates by their search score. Fewer when fewer have one;
    the caller records the shortfall."""
    have = [(m, r) for m, r in searches.items() if r['entry'] is not None]
    lower = any(r.get('lower_is_better') for _, r in have)
    have.sort(key=lambda x: (x[1]['score'] if lower else -x[1]['score'], x[0]))
    return [m for m, _ in have[:n]]


def best_base_pair(searches, lower):
    """The best (model, representation) by search score over every candidate."""
    have = [(r['base_score'], m, r['base_rep']) for m, r in searches.items()
            if r.get('base_rep') is not None]
    if not have:
        return None
    have.sort(key=lambda x: (x[0] if lower else -x[0], x[1], x[2]))
    score, model, rep = have[0]
    return {'model': model, 'representation': rep, 'train_folds_score': score}


def meta_learner_that_counts(meta, lower):
    """The best cross-validated score on the training folds' predictions. The
    test score is only used to skip a meta-learner that produced none."""
    have = [(r['train_rows_cv_score'], name) for name, r in meta.items()
            if r.get('heldout_score') is not None
            and r.get('train_rows_cv_score') is not None
            and np.isfinite(r['train_rows_cv_score'])]
    if not have:
        return None
    have.sort(key=lambda x: (x[0] if lower else -x[0], x[1]))
    return have[0][1]


# ── One model's predictions ────────────────────────────────────────────────

def as_matrix(raw, hard, classes, seen):
    """One model's output as n x K in `classes` order. A model with no
    probabilities gives 1 for the class it predicted."""
    n, k = len(hard), len(classes)
    out = np.zeros((n, k))
    if raw is not None:
        raw = np.asarray(raw)
        out[:, np.searchsorted(classes, np.asarray(seen))] = raw
        return out
    out[np.arange(n), np.searchsorted(classes, np.asarray(hard))] = 1.0
    return out


def fixed_feature_set(reps, labels, groups, tr, te, blueprint, factory, *,
                      metric_fn, lower_is_better):
    """Run the recipe once on the training folds. Returns the fixed features as
    (training folds matrix, outer held-out fold matrix), same features in both."""
    from experiments._shared import build_fold_hybrid, fold_hybrid_scoring_kwargs
    X_tr, X_te, _info = build_fold_hybrid(
        {k: v[tr] for k, v in reps.items()}, {k: v[te] for k, v in reps.items()},
        labels[tr], blueprint, groups=(groups[tr] if groups is not None else None),
        precomputed_alloc_scores=getattr(blueprint, 'rep_scores', None),
        **fold_hybrid_scoring_kwargs(blueprint, model=factory(), metric_fn=metric_fn,
                                     lower_is_better=lower_is_better))
    return np.asarray(X_tr), np.asarray(X_te)


def fit_predict_fixed(X_fit, y_fit, X_pred, factory, classes):
    """Fit on these fixed features, scaled on the fitting folds only, and predict.
    Returns n values, or n x K probabilities."""
    from kirby.scaling import FeatureScaler
    sc = FeatureScaler()
    X_fit_s = sc.fit_transform(X_fit)
    X_pred_s = sc.transform(X_pred)
    model = factory()
    model.fit(X_fit_s, y_fit)
    hard = np.asarray(model.predict(X_pred_s))
    if classes is None:
        return hard.astype(float)
    raw = None
    if hasattr(model, 'predict_proba'):
        try:
            raw = model.predict_proba(X_pred_s)
        except Exception:
            raw = None
    return as_matrix(raw, hard, classes, getattr(model, 'classes_', classes))


def recipe_tag(model, label):
    """A saved prediction's name, carrying the recipe it came from, so a reset unit
    whose search now picks another recipe fits again instead of reading the old one."""
    import hashlib
    return f"{model}__{hashlib.sha1(str(label).encode()).hexdigest()[:10]}"


def cached_predict(path, compute):
    """A saved prediction, or compute and save it. np.savez through a temporary
    name, so a job killed mid-write leaves nothing half-read."""
    path = Path(path)
    if path.exists():
        with np.load(path) as z:
            return z['pred'], float(z['seconds'])
    t0 = time.time()
    pred = compute()
    secs = time.time() - t0
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp.npz')
    np.savez(tmp, pred=pred, seconds=secs)
    os.replace(tmp, path)
    return pred, secs


# ── The unit ───────────────────────────────────────────────────────────────

def write_result(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.json.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(result, indent=1, default=float) + '\n')
    os.replace(tmp, path)


def run(a):
    t_start = time.time()
    from kirby.find_best_hybrid import (BUILTIN_METRICS, DEFAULT_N_FOLDS,
                                        nested_cv_outer_folds)
    from experiments import dataset_policy  # noqa: F401  fills the registry
    from experiments._shared import cv_split, get_domain_metric, get_model_factories
    from experiments.decided_grid import decided_models
    from experiments.noise.noise_helpers import (MOLECULE_MEAN_SEARCH_SUFFIX,
                                                 load_noise_data, molecule_mean_data)
    from experiments.orchestrate.plan import MOLECULE_MEAN_SEARCH_DATASETS

    ds, fold = a.dataset, a.fold
    n_outer = int(a.n_folds or DEFAULT_N_FOLDS)
    unit = f'meta_f{fold}'
    out_file = Path(a.output) / f'{ds}__{unit}.json'
    base = dict(result_schema=RESULT_SCHEMA, dataset=ds, fold=fold, n_outer_folds=n_outer,
                candidates=a.candidates, inner_folds=a.inner_folds, min_rows=a.min_rows,
                top_n_models=TOP_N_MODELS, seed=SEED,
                search_space=os.environ.get('KIRBY_SEARCH_SPACE'),
                search_pruning=os.environ.get('KIRBY_SEARCH_PRUNING'))

    data, _rep_names = load_noise_data(ds)
    suffix = None
    if ds in MOLECULE_MEAN_SEARCH_DATASETS:
        data = molecule_mean_data(data, ds)
        if data is None:
            sys.exit(f'{ds}: the loader exposes no repeat structure')
        suffix = MOLECULE_MEAN_SEARCH_SUFFIX
    labels = np.asarray(data['labels'])
    data.setdefault('dataset_name', ds)

    if len(labels) < a.min_rows:
        reason = f'{len(labels)} rows, below the minimum of {a.min_rows} in RUN_SETTINGS.json'
        print(f'{ds}: {reason}; skipped', flush=True)
        write_result(out_file, dict(base, result_status='skipped', reason=reason,
                                    n_rows=len(labels), seconds=time.time() - t_start,
                                    timestamp=time.strftime('%Y-%m-%dT%H:%M:%S')))
        return

    groups = data.get('groups')
    groups = np.asarray(groups) if groups is not None else None
    reps = data['reps']
    task_type = data.get('task_type') or 'regression'
    metric_name, metric_fn, lower = get_domain_metric(data)
    needs_proba = metric_fn is BUILTIN_METRICS.get('auc', (None,))[0]
    classes = np.unique(labels) if task_type == 'classification' else None

    folds = nested_cv_outer_folds(reps, labels, groups, n_outer, SEED, task_type)
    tr, te = (np.asarray(x) for x in folds[fold])
    print(f'{ds} fold {fold}: {len(tr)} training rows, {len(te)} held out, metric '
          f'{metric_name}, candidates {a.candidates}', flush=True)

    # Steps 1 and 2: every candidate searched, the top 5 taken on training rows.
    factories = get_model_factories(a.candidates, task_type=task_type, n_samples=len(tr))
    unavailable = {m: 'no factory: above its row limit, or an unknown name'
                   for m in a.candidates if m not in factories}
    recorded = [m for m in decided_models(ds, 'general') if m != 'knn']
    searches = search_candidates(data, labels, tr, fold, n_outer, ds, list(factories),
                                 recorded, base_suffix=suffix)
    chosen = top_models(searches)
    print(f'{ds} fold {fold}: top models {chosen}', flush=True)

    candidates_out = {
        m: {k: r.get(k) for k in ('score', 'seconds', 'why_none', 'searched')}
        | {'recipe': (r['entry'] or {}).get('label')}
        for m, r in searches.items()}
    base.update(metric=metric_name, lower_is_better=bool(lower), n_rows=len(labels),
                n_train=len(tr), n_heldout=len(te), candidates_unavailable=unavailable,
                candidate_searches=candidates_out, top_models=chosen)
    if len(chosen) < 2:
        write_result(out_file, dict(base, result_status='skipped',
                                    reason=f'{len(chosen)} candidate(s) have a recipe; '
                                           f'a meta-learner needs at least 2',
                                    seconds=time.time() - t_start,
                                    timestamp=time.strftime('%Y-%m-%dT%H:%M:%S')))
        return

    # Step 4's inner folds. Positions into tr.
    sample_X = next(iter(reps.values()))[tr]
    inner_groups = groups[tr] if groups is not None else None
    inner = [(np.asarray(i), np.asarray(j)) for i, j in
             cv_split(inner_groups, a.inner_folds, sample_X, labels[tr], mode='full',
                      task_type=task_type)]
    ck = Path(a.output) / f'{ds}__{unit}.predictions'
    ck.mkdir(parents=True, exist_ok=True)

    train_pred, test_pred, per_model = {}, {}, {}
    for m in chosen:
        entry = searches[m]['entry']
        tag = recipe_tag(m, entry.get('label'))
        bp, fac = entry['blueprint'], factories[m]
        shape = (len(tr),) if classes is None else (len(tr), len(classes))
        oof = np.full(shape, np.nan)
        t_build = time.time()
        X_fixed_tr, X_fixed_te = fixed_feature_set(
            reps, labels, groups, tr, te, bp, fac, metric_fn=metric_fn, lower_is_better=lower)
        build_s = time.time() - t_build
        y_tr = labels[tr]
        secs = 0.0
        for j, (itr, iva) in enumerate(inner):
            pred, s = cached_predict(
                ck / f'{tag}__inner{j}of{len(inner)}.npz',
                lambda: fit_predict_fixed(X_fixed_tr[itr], y_tr[itr], X_fixed_tr[iva], fac, classes))
            oof[iva] = pred
            secs += s
        # Step 5: the same fixed features, every training fold.
        pred, s = cached_predict(
            ck / f'{tag}__heldout.npz',
            lambda: fit_predict_fixed(X_fixed_tr, y_tr, X_fixed_te, fac, classes))
        train_pred[m], test_pred[m] = oof, pred
        per_model[m] = {
            'heldout_score': float(metric_fn(labels[te], to_metric_space(pred, classes, needs_proba))),
            'train_rows_score_from_search': searches[m]['score'],
            'n_features': int(X_fixed_tr.shape[1]),
            'feature_set_seconds': round(build_s, 1),
            'inner_and_refit_seconds': round(secs + s, 1)}
        print(f"  {ds} fold {fold} {m}: held-out {per_model[m]['heldout_score']:.4f}, "
              f"{secs + s:.0f} s for inner folds and refit", flush=True)

    # Step 6.
    def stack(store):
        return np.column_stack([store[m].reshape(len(store[m]), -1) for m in chosen])

    X_tr, X_te = stack(train_pred), stack(test_pred)
    bad = ~np.isfinite(X_tr).all(axis=1)
    if bad.any():
        raise RuntimeError(f'{int(bad.sum())} training rows have no prediction from a model')
    meta = score_meta_learners(X_tr, labels[tr], X_te, labels[te], inner, metric_fn=metric_fn,
                               classes=classes, needs_proba=needs_proba)
    pair = best_base_pair(searches, lower)
    if pair is not None:
        rep = pair['representation']
        pred, _s = cached_predict(
            ck / f"base_pair__{recipe_tag(pair['model'], rep)}__heldout.npz",
            lambda: fit_predict_fixed(reps[rep][tr], labels[tr], reps[rep][te],
                                      factories[pair['model']], classes))
        pair['heldout_score'] = float(metric_fn(labels[te], to_metric_space(pred, classes, needs_proba)))
    counts = meta_learner_that_counts(meta, lower)
    comparison = {
        'meta_learner_that_counts': {'name': counts, **meta[counts]} if counts else None,
        'best_model_hybrid': {'model': chosen[0], **per_model[chosen[0]]},
        'plain_average_of_5': meta.get('equal_weights'),
        'best_model_base_pair': pair,
    }
    write_result(out_file, dict(
        base, result_status='ok', models=per_model, meta_learners=meta, comparison=comparison,
        leak_note=('The recipe, the fixed features and the top-5 pick saw the labels of '
                   'the inner held-out folds the meta-learner trains on. The outer '
                   'held-out fold was seen only for the final score.'),
        seconds=round(time.time() - t_start, 1),
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S')))
    print(f'{ds} fold {fold}: wrote {out_file} in {time.time() - t_start:.0f} s', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('dataset')
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--candidates', required=True,
                    type=lambda s: [x for x in s.split(',') if x],
                    help='comma-separated model names; no default')
    ap.add_argument('--inner-folds', type=int, required=True, help='no default')
    ap.add_argument('--min-rows', type=int, required=True, help='no default')
    ap.add_argument('--output', required=True)
    ap.add_argument('--n-folds', type=int, default=None, help='outer folds (default DEFAULT_N_FOLDS)')
    run(ap.parse_args(argv))


if __name__ == '__main__':
    main()
