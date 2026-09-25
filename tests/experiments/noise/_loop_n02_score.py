#!/usr/bin/env python3
"""n02's reliability score, swept over the estimators that score the same thing.

WHAT n02 DOES NOW. It weights every training row by how far that row's label
sits from the mean of its ten nearest neighbours, in the space the feature
importances scale. That deviation goes through weights_from_scores at
strength 3.0, which is a 4-to-1 row weight between the most and the least
suspect molecule.

WHAT THIS SWEEPS. Three axes, one at a time, all through the SAME path -- a
per-row suspicion score, min-max normalised by weights_from_scores, handed to
.fit(sample_weight=...):

  --methods    which suspicion score. 'n02_current' is the neighbour-mean
               deviation n02 uses today. Everything else is a name in
               ESTIMATOR_REGISTRY (src/kirby/utils/noise_estimation.py).
  --strengths  how hard the weights lean. 1.0 is 2-to-1 between the most and
               least suspect row, 3.0 is 4-to-1, 9.0 is 10-to-1, 0.0 is no
               weighting at all.
  --weight-shapes
               how the suspicion score becomes a weight. 'inverse' is what n02
               uses: 1/(1 + strength * s), which moves EVERY training row --
               on herg_ki 556 of 941 rows end up more than a quarter of the
               way down from the heaviest weight to the lightest. 'soft' and
               'exponential' are the other two shapes weights_from_scores
               already holds. 'stepR' leaves every row at a weight of 1 except
               the R most suspect share, which drop to 1/(1 + strength) --
               step0.01 is 9 of 941 molecules on herg_ki. n18's two cleanups
               that read a flag rate, `remove` and `knn_smooth`, touch only
               that share and they are the ones that win.
  --space      which feature space the model is FITTED in: the control base
               representation, or the per-fold hybrid.
  --score-space  which feature space the neighbours are found in. It defaults
               to --space, which is what n02 does today: the rows are weighted
               in whatever space they are fitted in. Naming a different one
               separates the two, which nothing in n02 has ever done.
  --shuffle-seeds
               hands the SAME weight vector to the training rows in a permuted
               order, one arm per named seed, beside the arm that keeps the
               pairing. Both arms hold the identical multiset of weights, so
               the top-to-bottom row weight ratio is identical and the only
               difference is which row got which weight. Nothing in n02 has
               ever measured that.
  --importance-exponents
               the power the feature importance is raised to before it scales
               the axes of that neighbour search. n02 uses the square root,
               0.5, and has never run any other value. 0.0 leaves every
               feature at the width the scaler gave it, which is the space
               n18's `downweight` finds its neighbours in -- and n18's
               downweight WINS on herg_ki at +0.0047 rmse where n02's weights
               lose on the same dataset.

⛔ FULL ROWS. Every molecule. What is cut is the search space: 3 outer folds,
one base representation (the control), one importance scorer.

Every number it writes is in the dataset's own metric, measured on held-out
rows of the same folds for every arm, so an arm-minus-arm difference is over
one fold set.
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('KIRBY_RF_JOBS', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('KIRBY_CACHE_DIR', '/Volumes/seagate/kirby_cache')
if os.environ.get('KIRBY_SMOKE_N', '').strip():
    raise SystemExit('KIRBY_SMOKE_N is a ROW CAP inside load_data. Unset it.')

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'tests'))
sys.path.insert(0, os.path.join(REPO, 'tests', 'experiments'))

try:
    import torch  # noqa: F401  (before LightGBM; hybrid.py:3410 records why)
except ImportError:
    pass

import argparse
import inspect
import json
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_domain_metric, get_model_factories, cv_split,
    compute_feature_importance,
    get_best_search_for_downstream, build_fold_hybrid,
    fold_hybrid_scoring_kwargs, pool_recipes_by_family, POOL_SUBSETS,
)
from experiments.complexity.complexity_helpers import DATASET_REPS
from experiments.noise.n02_sample_weighting import (
    WEIGHTED_MODELS, SPACE_FOR_FAMILY, POOL_SUBSET, compute_reliability_scores,
)
from kirby.utils.noise_estimation import ESTIMATOR_REGISTRY, weights_from_scores


def _suspicion(method, X, y, k, is_clf):
    """One per-row suspicion score, higher = the label looks less trustworthy.

    'n02_current' reproduces what n02 computes today: the deviation of a row's
    label from its neighbours' mean, before weights_from_scores sees it. Every
    other name builds the registry estimator and calls estimate().
    """
    if method == 'n02_current':
        # compute_reliability_scores at strength 0 would flatten the weights,
        # so the deviation is recomputed here exactly as that function does.
        from experiments._shared import make_nearest_neighbors
        k_actual = min(k, len(y) - 1)
        nn = make_nearest_neighbors(n_neighbors=k_actual + 1)
        nn.fit(X)
        _, idx = nn.kneighbors(X)
        nb = idx[:, 1:]
        if is_clf:
            return 1.0 - np.mean((y[nb] == y[:, None]).astype(float), axis=1)
        return np.abs(y - np.mean(y[nb], axis=1))

    cls = ESTIMATOR_REGISTRY[method]
    params = inspect.signature(cls.__init__).parameters
    kw = {}
    if 'k' in params:
        kw['k'] = k
    if 'is_clf' in params:
        kw['is_clf'] = is_clf
    if 'random_state' in params:
        kw['random_state'] = 42
    est = cls(**kw)
    return np.asarray(est.estimate(X, y), dtype=np.float64)


def _weights(scores, strength, shape='inverse'):
    """One weight per training row, mean-normalised to 1.

    'inverse', 'soft' and 'exponential' are the three shapes
    weights_from_scores already holds and they all move every row. 'stepR'
    leaves every row at 1 except the R most suspect share of them, which drop
    to 1/(1 + strength) -- the same ratio 'inverse' puts between its extreme
    rows, applied to a named share of the molecules instead of to two
    individuals.
    """
    scores = np.asarray(scores, dtype=np.float64)
    ok = np.isfinite(scores)
    if not ok.all():
        scores = np.where(ok, scores, np.nanmedian(scores[ok]) if ok.any() else 0.0)
    if np.ptp(scores) < 1e-12:
        return np.ones(len(scores)), 'flat_score'
    if shape.startswith('step'):
        rate = float(shape[4:])
        n_flag = max(1, int(round(rate * len(scores))))
        w = np.ones(len(scores), dtype=np.float64)
        w[np.argsort(-scores)[:n_flag]] = 1.0 / (1.0 + float(strength))
        return w / w.mean(), None
    return weights_from_scores(scores, strength=float(strength),
                               method=shape), None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--methods', default='n02_current')
    p.add_argument('--strengths', default='3.0')
    p.add_argument('--models', default=','.join(WEIGHTED_MODELS))
    p.add_argument('--folds', type=int, default=3)
    p.add_argument('--k-neighbors', type=int, default=10)
    p.add_argument('--weight-shapes', default='inverse',
                   help="comma separated: inverse (what n02 ships), soft, "
                        "exponential, or stepR for a flag rate, e.g. step0.05")
    p.add_argument('--shuffle-seeds', default='',
                   help='comma separated integers. For every arm this run '
                        'builds, also fit one arm per seed whose weights are '
                        'that arm\'s weights reordered under that seed. The '
                        'seed is written into the arm name and into the '
                        'result file, so a shuffled arm can be re-read.')
    p.add_argument('--importance-method', default='lightgbm_gain')
    p.add_argument('--importance-exponents', default='0.5',
                   help='the power the feature importance is raised to before '
                        'it scales the axes the neighbour search runs in. 0.5 '
                        'is the square root n02 uses today; 0.0 leaves the '
                        'features unscaled, which is the space n18 finds its '
                        'neighbours in; 1.0 scales by the importance itself')
    p.add_argument('--space', default='base',
                   help="'base' = the control representation; a name in "
                        "n02's SPACE_FOR_FAMILY = the per-fold hybrid built "
                        "from that family's stored search")
    p.add_argument('--score-space', default=None,
                   help='where the suspicion score is computed; defaults to '
                        '--space')
    p.add_argument('--hybrid-model', default='rf',
                   help='whose stored recipe the hybrid space is built from')
    p.add_argument('--tag', default='sweep')
    p.add_argument('--out-dir', default='/Volumes/seagate/local_noise_run/loop_n02')
    p.add_argument('--base-rep', required=True,
                   help=('the single representation to compare against. '
                         'Required: the fixed preference list that used to '
                         'choose it was deleted 2026-09-25; the noise '
                         'experiments now compare against each downstream '
                         "model's top 2 single representations by its "
                         'stored score'))
    a = p.parse_args()

    a.score_space = a.score_space or a.space
    methods = [m.strip() for m in a.methods.split(',') if m.strip()]
    strengths = [float(s) for s in a.strengths.split(',') if s.strip()]
    exponents = [float(e) for e in a.importance_exponents.split(',') if e.strip()]
    shapes = [w.strip() for w in a.weight_shapes.split(',') if w.strip()]
    shuffle_seeds = [int(x) for x in a.shuffle_seeds.split(',') if x.strip()]
    if len(set(shuffle_seeds)) != len(shuffle_seeds):
        raise SystemExit(f'repeated shuffle seed in {shuffle_seeds}')
    for _sh in shapes:
        if _sh not in ('inverse', 'soft', 'exponential'):
            if not _sh.startswith('step'):
                raise SystemExit(f'unknown weight shape {_sh!r}')
            float(_sh[4:])
    models = [m.strip() for m in a.models.split(',') if m.strip()]
    for m in methods:
        if m != 'n02_current' and m not in ESTIMATOR_REGISTRY:
            raise SystemExit(f'unknown score {m!r}; registry holds '
                             f'{sorted(ESTIMATOR_REGISTRY)}')

    _sp_tag = (a.space if a.score_space == a.space
               else f'{a.space}_scored_in_{a.score_space}')
    out_path = os.path.join(a.out_dir,
                            f'score_sweep_{a.dataset}_{_sp_tag}_{a.tag}.json')
    if os.path.exists(out_path):
        print(f'RESULT ALREADY EXISTS, nothing to do: {out_path}')
        return

    rep_names = DATASET_REPS[a.dataset]['general']
    data = load_data(a.dataset, rep_preset='all', rep_names=rep_names)
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data['groups']
    augmentations = data.get('augmentations')
    base_rep = a.base_rep
    if base_rep not in reps:
        raise SystemExit(f'--base-rep {base_rep!r} is not among the loaded '
                         f'representations {sorted(reps)}')
    sample_X = next(iter(reps.values()))
    factories = get_model_factories(models, task_type=task_type,
                                    n_samples=len(labels))

    print(f'{a.dataset}: {len(labels)} rows, metric {metric_name} '
          f"({'lower' if lower_is_better else 'higher'} wins), control "
          f'representation {base_rep}, space {a.space}, '
          f'{a.folds} outer folds, models {list(factories)}')
    print(f'scores swept: {methods}')
    print(f'strengths swept: {strengths}')
    print(f'importance exponents swept: {exponents} '
          f'(0.5 is the square root n02 ships, 0.0 is no scaling at all)')
    print(f'weight shapes swept: {shapes} '
          f"('inverse' is what n02 ships and it moves every training row)")
    print(f'shuffle seeds: {shuffle_seeds or "none"} '
          f'(each one reorders the same weights among the training rows)',
          flush=True)

    # One stored recipe per hybrid space this run needs -- the space the model
    # is fitted in, and the space the suspicion score is computed in, which are
    # the same space unless --score-space says otherwise.
    recipe_for_space = {}
    _needed = {sp for sp in (a.space, a.score_space) if sp != 'base'}
    if _needed:
        _s, _meta = get_best_search_for_downstream(
            reps, labels, groups, downstream_model=a.hybrid_model,
            budget=None, augmentations=augmentations,
            dataset_name=a.dataset, task_type=task_type, data=data,
            pool_subset=POOL_SUBSET)
        _recipes, _unavail = pool_recipes_by_family(
            _meta, SPACE_FOR_FAMILY,
            eligible_families=POOL_SUBSETS[POOL_SUBSET])
        _per_space = {r['space']: r for r in _recipes}
        for _sp in sorted(_needed):
            if _sp not in _per_space:
                raise SystemExit(f'no {_sp} recipe for {a.hybrid_model} on '
                                 f'{a.dataset}: {_unavail}')
            _r = _per_space[_sp]
            _bp = _r['blueprint']
            _aug = _r['aug_condition'] or 'uncapped'
            if getattr(_bp, 'augmentation_strategy', None) != _aug:
                try:
                    _bp.augmentation_strategy = _aug
                except Exception:
                    pass
            recipe_for_space[_sp] = (_bp, _aug, _r['objective'])
            print(f'  {_sp} recipe: family {_r["objective"]}, aug {_aug}',
                  flush=True)

    # {model: {arm: [per-fold score]}}; arm 'unweighted' plus one per
    # (score, strength). Every list holds the SAME folds -- a fold enters all
    # of them or none.
    per_model = {m: {} for m in factories}
    score_cost = {}
    score_spread = {}
    notes = []
    n_folds_used = 0

    for fold_i, (tr, va) in enumerate(cv_split(groups, a.folds, sample_X,
                                               labels, mode='full')):
        t_fold = time.time()
        y_tr, y_va = labels[tr], labels[va]

        def _matrices(space):
            """Scaled training and held-out matrices for one feature space."""
            if space == 'base':
                return (reps[base_rep][tr], reps[base_rep][va], base_rep)
            _bp, _aug, _obj = recipe_for_space[space]
            _mA = augmentations if _aug != 'none' else None
            _a_tr = {k: v[tr] for k, v in _mA.items()} if _mA else None
            _a_va = {k: v[va] for k, v in _mA.items()} if _mA else None
            _h_tr, _h_va, _ = build_fold_hybrid(
                {k: v[tr] for k, v in reps.items()},
                {k: v[va] for k, v in reps.items()},
                y_tr, _bp,
                augmentations_train=_a_tr, augmentations_val=_a_va,
                groups=(groups[tr] if groups is not None else None),
                **fold_hybrid_scoring_kwargs(
                    _bp, model=factories[a.hybrid_model](),
                    metric_fn=metric_fn, lower_is_better=lower_is_better))
            return _h_tr, _h_va, 'hybrid'

        def _importance(X_raw, key):
            """The scaled matrix and its per-feature importance."""
            _sc = StandardScaler()
            _Xs = _sc.fit_transform(X_raw)
            _imp = compute_feature_importance({key: _Xs}, y_tr,
                                              method=a.importance_method)[key]
            return _Xs, np.maximum(_imp, 0)

        def _scaled_by_importance(_Xs, _imp, exponent):
            """The space the neighbour search runs in, at one exponent.

            At exponent 0.0 every feature keeps the width the scaler gave it,
            which is the space n18 finds its neighbours in. At 0.5 each
            feature is multiplied by the square root of its importance, which
            is what n02 does today.
            """
            if exponent == 0.0:
                return _Xs
            _w = np.power(_imp, exponent)
            _w = _w / (_w.sum() + 1e-10)
            return _Xs * _w

        X_raw_tr, X_raw_va, imp_key = _matrices(a.space)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_raw_tr)
        X_va = scaler.transform(X_raw_va)

        if a.score_space == a.space:
            _Xs_sc, _imp_sc = _importance(X_raw_tr, imp_key)
        else:
            _s_tr, _s_va, _s_key = _matrices(a.score_space)
            _Xs_sc, _imp_sc = _importance(_s_tr, _s_key)
        X_w_by_exp = {e: _scaled_by_importance(_Xs_sc, _imp_sc, e)
                      for e in exponents}

        fold_rows = {m: {} for m in factories}
        fold_ok = True

        for mname, factory in factories.items():
            try:
                mdl = factory()
                mdl.fit(X_tr, y_tr)
                fold_rows[mname]['unweighted'] = float(
                    metric_fn(y_va, mdl.predict(X_va)))
            except Exception as exc:
                notes.append(f'fold {fold_i} {mname} unweighted failed: {exc}')
                fold_ok = False
        if not fold_ok:
            continue

        for method in methods:
          for exp in exponents:
            # An arm name carries the exponent only when more than one is
            # swept, so a single-exponent run keeps the arm names every
            # result file written before this axis existed used.
            _etag = '' if len(exponents) == 1 else f'|imp^{exp}'
            t0 = time.time()
            try:
                sc = _suspicion(method, X_w_by_exp[exp], y_tr, a.k_neighbors,
                                is_clf)
            except Exception as exc:
                notes.append(f'fold {fold_i} score {method}{_etag} failed: '
                             f'{type(exc).__name__}: {exc}')
                fold_ok = False
                break
            score_cost.setdefault(f'{method}{_etag}', []).append(
                time.time() - t0)
            for st in strengths:
              for shape in shapes:
                # A shape name is carried in the arm only when more than one
                # is swept, so a single-shape run keeps the arm names every
                # result file written before this axis existed used.
                _stag = '' if len(shapes) == 1 else f'|{shape}'
                w, flat = _weights(sc, st, shape)
                if flat:
                    notes.append(f'fold {fold_i} score {method}{_etag} is flat')
                arm = f'{method}@{st}{_etag}{_stag}'
                # The scored vector, then that SAME vector handed to the
                # training rows in a permuted order, one arm per named seed.
                # A shuffled arm holds exactly the weights the scored arm
                # holds, so its top-to-bottom row weight ratio is identical
                # and the only thing that differs is which row got which
                # weight. The draw is keyed on the seed and the fold index,
                # so every fold gets its own permutation and re-running the
                # same seed reproduces it.
                arms_here = [(arm, w)]
                for _seed in shuffle_seeds:
                    _rng = np.random.default_rng([int(_seed), int(fold_i)])
                    arms_here.append((f'{arm}|shuffled_seed{_seed}',
                                      w[_rng.permutation(len(w))]))
                for arm_name, w_use in arms_here:
                    score_spread.setdefault(arm_name, []).append(
                        float(w_use.max() / max(w_use.min(), 1e-12)))
                    for mname, factory in factories.items():
                        try:
                            mdl = factory()
                            mdl.fit(X_tr, y_tr, sample_weight=w_use)
                            fold_rows[mname][arm_name] = float(
                                metric_fn(y_va, mdl.predict(X_va)))
                        except Exception as exc:
                            notes.append(f'fold {fold_i} {mname} {arm_name} '
                                         f'failed: {type(exc).__name__}: '
                                         f'{exc}')
                            fold_ok = False
            if not fold_ok:
                break
          if not fold_ok:
              break

        if not fold_ok:
            notes.append(f'fold {fold_i} dropped from every arm')
            continue

        for mname in factories:
            for arm, v in fold_rows[mname].items():
                per_model[mname].setdefault(arm, []).append(v)
        n_folds_used += 1
        print(f'  fold {fold_i} done in {time.time() - t_fold:.0f}s '
              f'({len(tr)} training rows)', flush=True)

    summary = {}
    for mname, arms in per_model.items():
        base = arms.get('unweighted')
        if not base:
            continue
        uw = float(np.mean(base))
        rows = {}
        for arm, vals in arms.items():
            mean = float(np.mean(vals))
            delta = mean - uw
            rows[arm] = {
                'mean': mean,
                'per_fold': vals,
                'delta_vs_unweighted': delta,
                # positive always means the weights helped
                'gain': (-delta if lower_is_better else delta),
            }
        summary[mname] = rows

    out = {
        'dataset': a.dataset,
        'n_molecules': int(len(labels)),
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'control_representation': base_rep,
        'space': a.space,
        'score_space': a.score_space,
        'hybrid_model': a.hybrid_model if a.space != 'base' else None,
        'folds_requested': a.folds,
        'folds_scored': n_folds_used,
        'k_neighbors': a.k_neighbors,
        'importance_method': a.importance_method,
        'importance_exponents': exponents,
        'weight_shapes': shapes,
        'shuffle_seeds': shuffle_seeds,
        'shuffle_draw': ('numpy default_rng([seed, fold_index]).permutation '
                         'of the training rows, applied to the weight vector '
                         'the scored arm uses' if shuffle_seeds else None),
        'methods': methods,
        'strengths': strengths,
        'models': list(factories),
        'per_model': summary,
        'score_seconds_per_fold': {k: float(np.mean(v))
                                   for k, v in score_cost.items()},
        'weight_ratio_top_to_bottom': {k: float(np.mean(v))
                                       for k, v in score_spread.items()},
        'notes': notes,
    }
    os.makedirs(a.out_dir, exist_ok=True)
    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {out_path}')

    for mname, rows in summary.items():
        print(f'\n{mname}  unweighted {rows["unweighted"]["mean"]:.4f} '
              f'{metric_name}')
        for arm, r in sorted(rows.items(), key=lambda kv: -kv[1]['gain']):
            if arm == 'unweighted':
                continue
            print(f'   {arm:<34} {r["mean"]:.4f} {metric_name}  '
                  f'gain {r["gain"]:+.4f}')


if __name__ == '__main__':
    main()
