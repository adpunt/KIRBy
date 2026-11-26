#!/usr/bin/env python3
"""Every candidate hybrid, scored by every metric in simeval.py, per fold.

WHAT THIS IS FOR. TBD2 asks which representations and which hybrids the
activity cliff study runs on. This produces the numbers that question waits on:
for each candidate hybrid, searched on one fold's training molecules, how
activity-coherent its neighbourhoods are under each of the metrics in
`tests/experiments/similarity_metrics/simeval.py`. It decides nothing.

ONE ROW IS one candidate hybrid on one fold under one distance coefficient,
plus, for the two noise metrics, one noise condition at one level on one
repetition. Rows are appended to `scores.jsonl` in the output directory as they
are produced, so a session that ends at four hours loses at most the hybrid it
was in the middle of, and a rerun of the same command skips every row already
there.

WHICH MOLECULES THE METRICS READ: that fold's training molecules only.
Adelaide, 2026-11-16: "Over that 80". Every distance matrix, every label and
every scaffold group below is sliced to `train_idx` before a metric sees it.

THE CANDIDATE HYBRIDS come from two places and are not written out here:

  experiments._shared.fold_neighbour_space_recipes
      per decided model: 'hybrid' and 'hybrid__general2' (the general tier's
      top two with distinct features)
      shared by every model: 'hybrid__distance1' (the distance search, which
      scores knn and nothing else) and 'hybrid_similarity' (the general tier
      under the similarity objective, which fits no model)

  experiments.cliffs.definitions.restricted_hybrids
      the restricted-input searches: the dataset's 0/1 representations, and
      those plus its count representations, each under the prediction
      objective (one per decided model) and under the similarity objective
      (one, since it fits no model), with the whole representation pool as the
      representation count

So the number of candidates per fold is not fixed at eight: the restricted
searches split four ways rather than two, because the objective and the
restricted representation list are two axes and neither is cut. Every candidate
that builds is scored and named; one that does not build is written to
`refused.json` with the reason, never skipped silently.

DISTANCE COEFFICIENTS. A metric in simeval.py scores a distance matrix, not a
feature matrix, so each hybrid is turned into one. Which coefficients are
defined depends on the assembled matrix's feature kind, and every one simeval
offers for that kind is run: simeval.BIT_COEFFS for a 0/1 matrix,
COUNT_COEFFS for a count matrix, CONT_COEFFS otherwise. The kind is read from
the assembled matrix.

THE TWO NOISE METRICS. `noise_detect_auroc` and `noise_recovery` need injected
label noise, and the other seven numbers do not, so the seven are computed once
per (hybrid, coefficient) and the two are computed once per (condition, level,
repetition). Injection goes through `kirby.noise_spec.make_noisy_train_labels`,
which is the definition the noise suite uses: the level is a fraction of the
standard deviation of the clean TRAINING labels, and no held-out label is
touched.

The two noise metrics read a mask of which molecules were corrupted, and that
mask is not one rule for every condition. Under outliers it is the share whose
noise scale is above the smallest, the badly-wrong subset. Under gaussian every
training label gets the same scale, so the mask is every training molecule --
and `noise_detect_auroc` is then an AUROC with no negative class, which
simeval.eval_noise_detect returns NaN for. Every row carries
`corrupted_mask_rule`, `noise_detect_defined` and `noise_recovery_defined`, so
a number that does not exist is not read as a bad score.

simeval.py is imported READ-ONLY. Nothing here changes it.

WHAT NO ROW CLAIMS. Not which hybrid is best, and not which metric to believe.
Comparing the metrics against each other is an analysis for after these numbers
exist.

Usage (no default for the fold count, the repetition count or the output
directory):

    python tests/experiments/cliffs/hybrid_metric_scores.py \\
        --datasets herg_ki,caco2,qm9,logd \\
        --n-folds 5 --repetitions 3 \\
        --output-dir $DATA/cliff_tbd2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent.parent
REPO = TESTS.parent
for _p in (REPO / 'src', TESTS, TESTS / 'experiments'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.similarity_metrics import simeval as SE            # noqa: E402
from experiments.noise import noise_timing as NT                    # noqa: E402

EXP_ID = 'hybrid_metric_scores'

# The seven numbers that do not read injected noise, in simeval's own order.
# simeval.EVALS carries all nine; these are that list without the two noise
# ones. Taken from it rather than retyped, so a change there is not silently
# missed here.
ACTIVITY_EVALS = [e for e in SE.EVALS
                  if e not in ('noise_detect_auroc', 'noise_recovery')]
NOISE_EVALS = ['noise_detect_auroc', 'noise_recovery']

# Her noise conditions, by the names she uses, mapped to the noiseInject
# condition each one is. Adelaide, 2026-11-18, on outliers and gaussian at
# level 1: "The noise conditions and levels you laid out are fine with me".
# Both are inputs below, so this maps names and chooses nothing.
CONDITION_STRATEGY = {
    'outliers': 'outlier_p05',
    'gaussian': 'gaussian',
    'laplace': 'laplace',
    'student_t': 'student_t_nu5',
    'grouped_wider': 'grouped_wider',
    'grouped_shifted': 'grouped_shifted',
    'censoring': 'censoring',
}


# ──────────────────────────────────────────────────────────────────────────
# the rows already on disk
# ──────────────────────────────────────────────────────────────────────────

def _row_key(row):
    """What makes a row unique, so a rerun can skip it."""
    return (row['dataset'], int(row['fold']), row['hybrid'],
            row['coefficient'], row.get('noise_condition'),
            row.get('noise_level'), row.get('repetition'))


def read_done(path):
    """Every row key already written. A truncated last line (a session killed
    mid-write) is dropped, not treated as a finished row."""
    done = set()
    if not Path(path).exists():
        return done
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(_row_key(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append_rows(path, rows):
    """Write and flush to the file descriptor, so a kill between hybrids
    cannot lose a row that was reported as written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str, separators=(',', ':')) + '\n')
        fh.flush()
        os.fsync(fh.fileno())


# ──────────────────────────────────────────────────────────────────────────
# candidates
# ──────────────────────────────────────────────────────────────────────────

def candidate_hybrids(data, dataset_name, *, train_idx, fold_i, n_folds,
                      verbose=True):
    """{name: (matrix over this fold's training molecules, provenance)} and
    {name: why it did not build}.

    The general, distance and similarity recipes come back from
    fold_neighbour_space_recipes as blueprints and are rebuilt here on this
    fold's training molecules. The restricted-input ones come back from
    restricted_hybrids already assembled over every molecule and are sliced.
    """
    from experiments._shared import (build_fold_hybrid,
                                     fold_hybrid_scoring_kwargs,
                                     fold_neighbour_space_recipes,
                                     get_domain_metric, get_model_factories)
    from experiments.cliffs import definitions as D

    reps = {k: np.asarray(v, dtype=np.float64)
            for k, v in (data.get('reps') or {}).items()}
    labels = np.asarray(data['labels']).ravel()
    groups = data.get('groups')
    train_idx = np.asarray(train_idx)
    y_train = labels[train_idx]
    g_train = None if groups is None else np.asarray(groups)[train_idx]
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data.get('task_type') or 'auto'

    built, refused = {}, {}

    with NT.timed('neighbour_space_recipes'):
        try:
            per_model = fold_neighbour_space_recipes(
                reps, labels, groups, train_idx=train_idx, fold_i=int(fold_i),
                n_outer_folds=int(n_folds), dataset_name=dataset_name,
                augmentations=data.get('augmentations'), task_type=task_type,
                domain=data.get('domain'), verbose=False)
        except Exception as exc:
            per_model = {}
            refused['general, distance and similarity searches'] = \
                f'{type(exc).__name__}: {exc}'

    tr_reps = {k: v[train_idx] for k, v in reps.items()}
    aug = data.get('augmentations') or None
    aug_tr = (None if not aug else
              {k: np.asarray(v)[train_idx] for k, v in aug.items()})

    # A recipe shared by both models ('hybrid__distance1' is knn's and the
    # same for every model, 'hybrid_similarity' fits no model) is built once.
    seen_recipe = {}
    for model, entry in sorted((per_model or {}).items()):
        if entry is None:
            refused[f'every space of {model}'] = \
                'the per-fold general search failed for this model'
            continue
        for space, why in (entry.get('unavailable') or {}).items():
            refused[f'{space} ({model})'] = why
        for space, (bp, aug_cond) in (entry.get('recipes') or {}).items():
            shared = space in ('hybrid__distance1', 'hybrid_similarity')
            name = space if shared else f'{space}__{model}'
            if name in built or name in refused:
                continue
            fp = (space, id(bp))
            if shared and fp in seen_recipe:
                continue
            seen_recipe[fp] = name
            factory = get_model_factories(
                [model], 'classification' if task_type == 'classification'
                else 'regression', len(labels)).get(model)
            score_kw = fold_hybrid_scoring_kwargs(
                bp, model=(factory() if factory else None),
                metric_fn=metric_fn, lower_is_better=lower_is_better)
            try:
                with NT.timed('build_hybrid', hybrid=name):
                    X_tr, _X_va, info = build_fold_hybrid(
                        tr_reps, tr_reps, y_train, bp,
                        augmentations_train=aug_tr, augmentations_val=aug_tr,
                        groups=g_train, **score_kw)
            except Exception as exc:
                refused[name] = f'{type(exc).__name__}: {exc}'
                continue
            X_tr = np.asarray(X_tr, dtype=np.float64)
            built[name] = (X_tr, {
                'space': space,
                'objective': ('similarity' if space == 'hybrid_similarity'
                              else 'predictive'),
                'search_tier': ('general_distance'
                                if space == 'hybrid__distance1' else 'general'),
                'downstream_model': None if shared else model,
                'shared_by_every_model': bool(shared),
                'aug_condition': aug_cond,
                'built_on': "this fold's training molecules",
                'n_training_molecules': int(len(train_idx)),
                'budget_features': (int(bp.budget)
                                    if getattr(bp, 'budget', None) else None),
                'allocation_method': getattr(bp, 'allocation_method', None),
                'importance_method': getattr(bp, 'importance_method', None),
                'internal_evaluator': getattr(bp, 'internal_evaluator', None),
                'representation_count': getattr(bp, 'n_reps', None),
                'search_metric': metric_name,
                'n_features_assembled': int(X_tr.shape[1]),
            })
            if verbose:
                print(f'    {name}: {X_tr.shape[1]} features', flush=True)

    with NT.timed('restricted_hybrids'):
        try:
            rb, rr = D.restricted_hybrids(
                data, dataset_name, train_idx=train_idx, fold_i=int(fold_i),
                n_folds=int(n_folds), verbose=verbose)
        except Exception as exc:
            rb, rr = {}, {'the restricted-input searches':
                          f'{type(exc).__name__}: {exc}'}
    refused.update(rr or {})
    for name, (X_all, prov) in (rb or {}).items():
        built[name] = (np.asarray(X_all, dtype=np.float64)[train_idx],
                       dict(prov, search_tier='general'))

    return built, refused


def coefficients_for(X, dataset_name, name):
    """{coefficient name: function} -- every one simeval defines for this
    matrix's feature kind, read from the ASSEMBLED matrix."""
    from kirby.similarity.preprocessing import feature_kind
    kind = feature_kind(X)
    table = {'binary': SE.BIT_COEFFS, 'count': SE.COUNT_COEFFS}.get(
        kind, SE.CONT_COEFFS)
    return kind, dict(table)


# ──────────────────────────────────────────────────────────────────────────
# the metrics
# ──────────────────────────────────────────────────────────────────────────

def tanimoto_similarity_reference(reps, train_idx, dataset_name):
    """ECFP4 Tanimoto SIMILARITY over this fold's training molecules.

    Two of simeval's numbers read it: cliff_ground_truth picks which molecules
    are cliffs with `>= 0.55`, and diag_ecfp_agreement correlates against
    `1 - it`. Both read it as a similarity, so it is 1 - the Tanimoto distance.
    It is the cliff TARGET, never the yardstick -- simeval's
    EVAL_IS_ACTIVITY_GROUNDED marks cliff_auroc False for that reason.
    """
    for key in ('ecfp4', 'ecfp', 'morgan'):
        if key in reps:
            X = np.asarray(reps[key], dtype=np.float64)[np.asarray(train_idx)]
            return key, 1.0 - SE.dist_tanimoto(X)
    raise KeyError(
        f'{dataset_name} has no ecfp4 representation loaded, so the cliff '
        f'target simeval.cliff_ground_truth reads cannot be built. '
        f'Representations present: {sorted(reps)}')


def activity_scores(D, y, groups, tanimoto_ref, seed=0):
    """The seven numbers that read no injected noise. Each metric is called
    on its own so one that raises does not take the other six with it."""
    cliffs = SE.cliff_ground_truth(tanimoto_ref, y)
    out, failed = {}, {}

    def one(name, fn):
        try:
            with NT.timed('metric', metric=name):
                out[name] = float(fn())
        except Exception as exc:
            out[name] = float('nan')
            failed[name] = f'{type(exc).__name__}: {exc}'

    try:
        with NT.timed('metric', metric='knn_readacross'):
            rho, negrmse = SE.eval_knn_readacross(D, y, groups)
        out['knn_rho'], out['knn_neg_rmse'] = float(rho), float(negrmse)
    except Exception as exc:
        out['knn_rho'] = out['knn_neg_rmse'] = float('nan')
        failed['knn_readacross'] = f'{type(exc).__name__}: {exc}'
    one('modi', lambda: SE.eval_modi_binary(D, y))
    one('patterson', lambda: SE.eval_patterson(D, y, seed=seed))
    one('cliff_auroc', lambda: SE.eval_cliff_auroc(D, y, cliffs))
    one('violation_neg', lambda: SE.eval_violation_rate(D, y))
    one('bedroc', lambda: SE.eval_vs_enrichment(D, y))
    one('diag_ecfp_agreement',
        lambda: SE.diag_ecfp_agreement(D, tanimoto_ref, seed=seed))
    out['n_cliffs'] = int(cliffs.sum())
    return out, failed


def inject(labels, groups, train_idx, condition, level, repetition,
           task_type):
    """(y_noisy over the training rows, corrupted mask, epsilon, provenance).

    `kirby.noise_spec.make_noisy_train_labels`, the same definition the noise
    suite injects with: the level is a fraction of the standard deviation of
    the clean TRAINING labels, and only the training rows are corrupted.
    """
    from kirby.noise_spec import NoiseSpec, make_noisy_train_labels
    strategy = CONDITION_STRATEGY.get(condition)
    if strategy is None:
        raise ValueError(
            f'{condition!r} is not a noise condition this names; they are '
            f'{sorted(CONDITION_STRATEGY)}.')
    spec = NoiseSpec(strategy=strategy, target_effective_noise=float(level),
                     noise_seed=10_000 + int(repetition), dose_units='spread')
    y_noisy, res = make_noisy_train_labels(
        labels, spec, np.asarray(train_idx),
        is_clf=(task_type == 'classification'),
        groups=None if groups is None else np.asarray(groups))
    if res is None:
        raise ValueError(
            f'{condition!r} on a classification dataset gives no epsilon and '
            f'no corrupted mask, so neither noise metric is defined.')
    # WHICH MOLECULES THE MASK MARKS, and why it is not one rule.
    #
    # Both noise metrics take a mask of corrupted molecules: eval_noise_detect
    # is an AUROC against it, eval_noise_recovery a correlation on it.
    #
    # Under outliers a share of the molecules get a much wider error than the
    # rest, so the badly-wrong subset is the molecules whose noise scale is
    # above the smallest one, and that is what the mask marks.
    #
    # Under gaussian every molecule gets the same scale. Reading the subset off
    # the scale there marks NO molecule, because none is above the minimum --
    # and that is the opposite of the truth, which is that every training label
    # was corrupted. kirby.noise_spec.CONSTANT_SCALE_STRATEGIES is the list of
    # conditions that behave that way, so for those the mask is every training
    # molecule.
    #
    # A full mask leaves eval_noise_detect with no negative class and it
    # returns NaN. That is recorded, not worked around: a condition that
    # corrupts everything has no detection problem to score.
    from kirby.noise_spec import CONSTANT_SCALE_STRATEGIES
    scale = np.asarray(res.noise_scale, dtype=float)
    if strategy in CONSTANT_SCALE_STRATEGIES:
        mask = np.ones(len(scale), dtype=bool)
        mask_rule = ('every training molecule: this condition gives them all '
                     'the same noise scale')
    else:
        mask = scale > (scale.min() + 1e-12)
        mask_rule = 'the molecules whose noise scale is above the smallest one'
    eps = np.asarray(res.epsilon, dtype=float)
    prov = {
        'noise_condition': condition,
        'noiseinject_strategy': strategy,
        'noise_level_fraction_of_training_label_sd': float(level),
        'repetition': int(repetition),
        'noise_seed': int(spec.noise_seed),
        'clean_training_label_sd': float(res.clean_label_sd),
        'target_noise_label_units': float(res.target_dose_label_units),
        'realised_noise_label_units': float(res.realised_dose_label_units),
        'realised_noise_fraction_of_sd': float(
            res.realised_dose_fraction_of_spread),
        'corrupted_molecules': int(mask.sum()),
        'corrupted_mask_rule': mask_rule,
        'affected_molecule_fraction': float(res.affected_molecule_fraction),
    }
    return np.asarray(y_noisy, dtype=float), mask, eps, prov


def noise_scores(D, y_noisy, mask, eps):
    """The two numbers that read injected noise, and whether each is defined.

    `noise_detect_auroc` is an AUROC against the corrupted mask, so a
    condition that corrupts every training molecule leaves it with no negative
    class and simeval returns NaN. `noise_recovery` is a correlation on the
    corrupted molecules and is defined in that case.
    """
    out, failed = {}, {}
    for name, fn in (('noise_detect_auroc',
                      lambda: SE.eval_noise_detect(D, y_noisy, mask)),
                     ('noise_recovery',
                      lambda: SE.eval_noise_recovery(D, y_noisy, mask, eps))):
        try:
            with NT.timed('metric', metric=name):
                out[name] = float(fn())
        except Exception as exc:
            out[name] = float('nan')
            failed[name] = f'{type(exc).__name__}: {exc}'
    out['noise_detect_defined'] = bool(np.isfinite(out['noise_detect_auroc']))
    out['noise_recovery_defined'] = bool(np.isfinite(out['noise_recovery']))
    out['all_training_molecules_corrupted'] = bool(mask.all())
    return out, failed


# ──────────────────────────────────────────────────────────────────────────
# one dataset
# ──────────────────────────────────────────────────────────────────────────

def check_environment(verbose=True):
    """Import everything the run reaches, and say which Python did it.

    Run before any dataset is touched. A stale NoiseInject checkout on the
    server, or a shell where setup.sh was never sourced, raised here after the
    representations had already been loaded; this costs seconds instead.
    Returns the modules run_dataset needs, so they are imported exactly once.
    """
    import platform
    from experiments._shared import load_data, fold_searches  # noqa: F401
    from experiments.noise.noise_helpers import noise_outer_folds
    from experiments.cliffs import definitions as _D           # noqa: F401
    from kirby.noise_spec import (NoiseSpec,                   # noqa: F401
                                  CONSTANT_SCALE_STRATEGIES)
    from kirby.find_best_hybrid import find_best_hybrid        # noqa: F401
    import noiseInject
    import experiments.dataset_policy                          # noqa: F401
    if verbose:
        print(f'  python {platform.python_version()} at {sys.executable}',
              flush=True)
        print(f'  noiseInject from {Path(noiseInject.__file__).parent}',
              flush=True)
        print(f'  {len(SE.EVALS)} metrics from {Path(SE.__file__).name}: '
              f'{", ".join(SE.EVALS)}', flush=True)
    return load_data, noise_outer_folds


def run_dataset(dataset_name, *, n_folds, repetitions, conditions, level,
                output_dir, only_folds=None, verbose=True,
                environment=None):
    """Every candidate hybrid on every requested fold, appended as it is
    scored. Returns (rows written, refusals)."""
    load_data, noise_outer_folds = environment or check_environment(verbose)

    out_dir = Path(output_dir)
    scores_path = out_dir / 'scores.jsonl'
    done = read_done(scores_path)
    print(f'\n{"=" * 70}\n{dataset_name}: {len(done)} rows already on disk\n'
          f'{"=" * 70}', flush=True)

    t0 = time.time()
    data = load_data(dataset_name)
    labels = np.asarray(data['labels']).ravel()
    groups = data.get('groups')
    reps = data.get('reps') or {}
    task_type = data.get('task_type') or 'regression'
    print(f'  loaded in {time.time() - t0:.0f}s: {len(labels)} molecules, '
          f'{len(reps)} representations, task {task_type}', flush=True)

    config = {'n_splits': int(n_folds)}
    if only_folds:
        config['only_folds'] = list(only_folds)
    folds, n_outer = noise_outer_folds(data, config)

    ecfp_name, tan_ref_full = None, None
    written, refusals = 0, {}

    for fold_i, train_idx, val_idx in folds:
        print(f'\n  fold {fold_i} of {n_outer}: {len(train_idx)} training '
              f'molecules, the ones the metrics read', flush=True)
        y_train = labels[np.asarray(train_idx)]
        g_train = (None if groups is None
                   else np.asarray(groups)[np.asarray(train_idx)])
        if g_train is None:
            raise ValueError(
                f'{dataset_name} has no scaffold groups, and '
                f'simeval.eval_knn_readacross splits on them.')
        ecfp_name, tan_ref = tanimoto_similarity_reference(
            reps, train_idx, dataset_name)

        with NT.timed('candidates', fold=int(fold_i)):
            built, refused = candidate_hybrids(
                data, dataset_name, train_idx=train_idx, fold_i=fold_i,
                n_folds=n_outer, verbose=verbose)
        for k, v in (refused or {}).items():
            refusals[f'fold {fold_i}: {k}'] = v
            print(f'    no {k}: {v}', flush=True)
        print(f'    {len(built)} candidate hybrids built', flush=True)

        for name, (X, prov) in sorted(built.items()):
            kind, coeffs = coefficients_for(X, dataset_name, name)
            rows = []
            for coeff_name, coeff_fn in sorted(coeffs.items()):
                base_key = (dataset_name, int(fold_i), name, coeff_name)
                wanted = [base_key + (None, None, None)] + [
                    base_key + (c, float(level), int(r))
                    for c in conditions for r in range(int(repetitions))]
                if all(k in done for k in wanted):
                    continue
                with NT.timed('distance_matrix', hybrid=name,
                              coefficient=coeff_name):
                    Dm = coeff_fn(X)
                Dm = np.asarray(Dm, dtype=np.float64)
                common = dict(
                    dataset=dataset_name, fold=int(fold_i), hybrid=name,
                    coefficient=coeff_name, feature_kind=kind,
                    n_training_molecules=int(len(train_idx)),
                    n_features=int(X.shape[1]),
                    ecfp_reference=ecfp_name,
                    label_metric=prov.get('search_metric'), **{
                        f'provenance_{k}': v for k, v in prov.items()})

                if base_key + (None, None, None) not in done:
                    sc, failed = activity_scores(Dm, y_train, g_train, tan_ref)
                    rows.append(dict(common, noise_condition=None,
                                     noise_level=None, repetition=None,
                                     metrics_that_raised=failed or None, **sc))

                for cond in conditions:
                    for r in range(int(repetitions)):
                        if base_key + (cond, float(level), int(r)) in done:
                            continue
                        try:
                            y_noisy, mask, eps, nprov = inject(
                                labels, groups, train_idx, cond, level, r,
                                task_type)
                        except Exception as exc:
                            refusals[f'fold {fold_i}: {cond} level {level} '
                                     f'repetition {r}'] = \
                                f'{type(exc).__name__}: {exc}'
                            continue
                        sc, failed = noise_scores(Dm, y_noisy, mask, eps)
                        # nprov carries its own noise_condition and repetition,
                        # so the row is built by update, not by keyword: the
                        # three resume-key columns are written last and are the
                        # ones _row_key reads.
                        row = dict(common)
                        row.update(nprov)
                        row.update(sc)
                        row.update(noise_condition=cond,
                                   noise_level=float(level),
                                   repetition=int(r),
                                   metrics_that_raised=failed or None)
                        rows.append(row)

            if rows:
                with NT.timed('result', hybrid=name, fold=int(fold_i)):
                    append_rows(scores_path, rows)
                written += len(rows)
                done.update(_row_key(r) for r in rows)
                print(f'      {name}: {len(rows)} rows written '
                      f'({written} this session)', flush=True)

    if refusals:
        path = out_dir / f'{dataset_name}__refused.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(refusals, indent=2, default=str))
        print(f'\n  {len(refusals)} refusals written to {path}', flush=True)
    return written, refusals


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', required=True,
                    help='comma separated. Adelaide, 2026-11-18, on herg: '
                         '"we\'re obviously not running herg or herg_blocking"')
    ap.add_argument('--n-folds', type=int, required=True,
                    help='the outer fold count the per-fold searches are '
                         'filed under. No default: not decided.')
    ap.add_argument('--repetitions', type=int, required=True,
                    help='how many times each noise condition is injected at '
                         'the level, each on its own seed. No default: not '
                         'decided.')
    ap.add_argument('--output-dir', required=True,
                    help='where scores.jsonl and the timing logs go')
    ap.add_argument('--only-folds', default=None,
                    help='comma separated fold numbers, to run some of the '
                         'folds now and the rest later')
    ap.add_argument('--noise-conditions', default='outliers,gaussian',
                    help='Adelaide, 2026-11-18: "The noise conditions and '
                         'levels you laid out are fine with me"')
    ap.add_argument('--noise-level', type=float, default=1.0,
                    help='a fraction of the standard deviation of the clean '
                         'training labels')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--check-environment-only', action='store_true',
                    help='import everything the run reaches, print which '
                         'Python and which noiseInject, and stop')
    a = ap.parse_args(argv)

    datasets = [d.strip() for d in a.datasets.split(',') if d.strip()]
    conditions = [c.strip() for c in a.noise_conditions.split(',') if c.strip()]
    only = (None if not a.only_folds else
            [int(x) for x in a.only_folds.split(',') if x.strip()])
    unknown = [c for c in conditions if c not in CONDITION_STRATEGY]
    if unknown:
        ap.error(f'unknown noise conditions {unknown}; '
                 f'they are {sorted(CONDITION_STRATEGY)}')

    # Before anything is loaded: every import the run reaches, once.
    print('\nchecking the environment', flush=True)
    environment = check_environment(not a.quiet)
    print('  every import the run reaches is in place', flush=True)
    if a.check_environment_only:
        return 0

    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for ds in datasets:
        with NT.log_to(NT.timing_path(out_dir, EXP_ID, ds, 'all'),
                       experiment=EXP_ID, dataset=ds, condition='all'):
            with NT.timed('run'):
                written, _ = run_dataset(
                    ds, n_folds=a.n_folds, repetitions=a.repetitions,
                    conditions=conditions, level=a.noise_level,
                    output_dir=out_dir, only_folds=only,
                    verbose=not a.quiet, environment=environment)
        total += written
    print(f'\n{total} rows written this session to {out_dir / "scores.jsonl"}',
          flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
