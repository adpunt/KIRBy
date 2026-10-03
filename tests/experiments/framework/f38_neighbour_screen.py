#!/usr/bin/env python3
"""f38 screen: how well each neighbour setting flags the corrupted training
labels, with no model refitted. Handoff: tests/experiments/framework/F38_HANDOFF.md.

WHAT ONE ROW IS. One (dataset, noise condition, fold, draw, downstream model,
neighbour setting, k, grouped). Its score is the AUROC of the injected-corruption
mask (noise_helpers.inject_train_labels) against each training molecule's gap
between its noisy label and its k nearest neighbours' noisy labels:
  regression      |y - mean(y of the k neighbours)|, and the same with the median
                  (the median is what f23's simeval.eval_noise_detect uses)
  classification  the share of the k neighbours whose label differs from its own
1.0 means every corrupted molecule has a larger gap than every clean one; 0.5 is
no better than chance.

THE NEIGHBOUR SETTINGS. A neighbour setting is one feature space, one feature
importance weighting and one distance.
  the noise suite's own  every setting n07.sweep_settings() returns: 7 feature
                         spaces x 4 weightings x {manhattan, euclidean}, plus
                         tanimoto_ecfp4, forest_proximity and model_input. Built
                         by n07.fold_similarity_spaces and n07.fold_neighbours,
                         the same calls n42 makes, so the lists are the ones the
                         noise suite reads (and share its disk cache).
  f38's additions        on the same 7 feature-space matrices: cosine,
                         correlation and minkowski p=0.5; and every weighted
                         setting at importance power 0.5 and 1 as well as the
                         power the distance implies (preprocessing.weight_exponent:
                         1 under manhattan, 0.5 under euclidean and cosine, 1/p = 2
                         under minkowski p=0.5). correlation has no power: the
                         weight goes inside its formula (fold_weights returns it
                         for scipy). Built here; nothing in noise/ is changed.
model_input and forest_proximity depend on which matrix the model is fitted on, so
they are scored once per model input (base_1, base_2, hybrid), as n42 does.

k = 5, 9, 10, 20 (similarity_space.SWEPT_KS). Each setting is scored on two
lists: 'grouped' excludes the molecule's own scaffold group (what n07's neighbour
detectors read), 'plain' excludes only the molecule itself (what label_smooth,
knn_smooth and n02 read).

WHERE THE SCORE IS UNDEFINED, and skipped with the reason written down:
  as_they_come and the curator flag   nothing was injected, so there is no mask
  a condition where every training    noise_helpers.DETECTION_IS_DEFINED lists the
  label moves (gaussian, the heavy    regression conditions with a corrupted
  tails, grouped_shifted)             subset; the others are written as skipped
  a draw whose mask is all one value  written as skipped with the counts

NEIGHBOUR-LIST OVERLAP. For each pair of settings that differ in exactly one of
(feature space, distance, weighting with its power), and model_input,
tanimoto_ecfp4 and forest_proximity against every other setting: the share of a
molecule's k neighbours the two lists have in common, averaged over the training
molecules. Plain lists only. 1.0 means the two settings pick the same neighbours.

LEVELS. Regression: one level, dataset_policy.reference_noise_level (0.5 of the
label SD), as n07. Classification: flip rate 0.1, as n07's noise_fraction. Draws
0, 1, 2 (n07.NOISE_SEEDS).

ONE TASK = one (dataset, condition, fold, draw, model). Each task writes
  <out>/<dataset>/<condition>__fold<i>__draw<d>__<model>.jsonl          detection
  <out>/<dataset>/<condition>__fold<i>__draw<d>__<model>.overlap.jsonl   overlap
A detection row already on disk is skipped on a rerun. The overlap file is written
in one go at the end of the task, so a task stopped before then recomputes the
lists it needs (the noise suite's are read back from its cache).

Usage on the server, inside one `srun ... --pty /bin/bash` session:
    python tests/experiments/framework/f38_neighbour_screen.py --list-tasks tasks.txt
    python tests/experiments/framework/f38_neighbour_screen.py --task caco2 outlier_p05 0 0 rf
"""
import os
import sys

if 'KIRBY_CACHE_DIR' not in os.environ:
    if os.path.isdir('/Volumes/seagate/kirby_simrun_cache'):
        os.environ['KIRBY_CACHE_DIR'] = '/Volumes/seagate/kirby_simrun_cache'
    else:
        sys.exit('KIRBY_CACHE_DIR is not set. Run `. setup.sh` first.')
os.environ.setdefault('KIRBY_NO_COLD_SEARCH', '1.0')

import json
import time
import argparse
import functools
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2] / 'src'))
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parent))

from experiments._shared import (  # noqa: E402
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS_NOISE,
    get_best_search_for_downstream, noise_base_reps_by_model)
from experiments.dataset_policy import (  # noqa: E402
    noise_groups_for, reference_noise_level)
from experiments.noise.noise_helpers import (  # noqa: E402
    noise_spec_for, inject_train_labels, conditions_for_task,
    pool_recipe_per_downstream_model, load_noise_data, DETECTION_IS_DEFINED)
from experiments.noise import similarity_space as ss  # noqa: E402
from experiments.noise import n07_remediation as n07  # noqa: E402
from experiments.noise import n42_similarity_noise_sweep as n42  # noqa: E402
from kirby.similarity.preprocessing import (  # noqa: E402
    fold_weights, weight_exponent, importance_exponent_override)

OUT_DEFAULT = _HERE.parents[1] / 'results' / 'framework' / 'f38_neighbour_screen'

KS = tuple(ss.SWEPT_KS)
K_MAX = ss.DEFAULT_K_MAX
DRAWS = tuple(n07.NOISE_SEEDS)
CLF_FLIP_RATE = 0.1
MODEL_INPUTS = n42.MODEL_INPUTS

# Every distance f38 builds on a feature space: (scipy name, scipy kwargs, the
# name preprocessing.fold_weights knows it by, minkowski p).
DISTANCES = {
    'manhattan': ('cityblock', {}, 'manhattan', None),
    'euclidean': ('euclidean', {}, 'euclidean', None),
    'cosine': ('cosine', {}, 'cosine', None),
    'correlation': ('correlation', {}, 'correlation', None),
    'minkowski_p0.5': ('minkowski', {'p': 0.5}, 'minkowski', 0.5),
}
POWERS = (0.5, 1.0)
WEIGHTED = tuple(w for w in ss.WEIGHTINGS if w != 'none')


def implied_power(distance):
    _, _, coeff, p = DISTANCES[distance]
    return weight_exponent(coeff, p)


def powers_for(distance):
    """The importance powers a weighted setting under `distance` is run at: the
    one the distance implies, plus 0.5 and 1. None where the weight cannot be
    folded into the features (correlation)."""
    implied = implied_power(distance)
    if implied is None:
        return (None,)
    return tuple(sorted({float(implied), *POWERS}))


# ---------------------------------------------------------------------------
# The settings
# ---------------------------------------------------------------------------

def describe(space, weighting, distance, power):
    """One setting as a dict, and its name. The noise suite's settings keep their
    own name (space|weighting|distance); f38's weighted ones add |power<p>."""
    if space in (ss.MODEL_INPUT_SPACE, ss.TANIMOTO_SPACE, ss.FOREST_SPACE) \
            or weighting == 'none':
        name = f'{space}|{weighting}|{distance}'
    else:
        name = f'{space}|{weighting}|{distance}|power{power:g}' \
            if power is not None else f'{space}|{weighting}|{distance}|in_formula'
    return {'name': name, 'space': space, 'weighting': weighting,
            'distance': distance, 'power': power}


def is_noise_suite_setting(weighting, distance, power):
    """Whether this feature-space setting is one n07.sweep_settings() returns,
    so its lists come from n07.fold_neighbours rather than from f38."""
    if distance not in ss.FEATURE_METRICS:
        return False
    return weighting == 'none' or power == implied_power(distance)


def feature_space_settings(spaces):
    out = []
    for sp in ss.FEATURE_SPACES:
        if sp not in spaces:
            continue
        for dist in DISTANCES:
            out.append(describe(sp, 'none', dist, None))
            for w in WEIGHTED:
                for pw in powers_for(dist):
                    out.append(describe(sp, w, dist, pw))
    return out


# ---------------------------------------------------------------------------
# f38's own lists
# ---------------------------------------------------------------------------

def own_lists(Z_tr, distance, w, power, exclude_groups):
    """(idx, n_rows_with_undefined_distance): each training row's K_MAX nearest
    other training rows under one f38 setting, nearest first.

    The matrices are already z-scored (n07.fold_similarity_spaces), which
    fold_weights requires. cosine and correlation are undefined for a row with
    no spread (all zero, or all equal); those distances are set to +inf, so such
    a row is nobody's neighbour, and the count is returned."""
    scipy_name, kw, coeff, p = DISTANCES[distance]
    kw = dict(kw)
    Z = np.asarray(Z_tr, dtype=np.float64)
    if w is not None:
        Z, scipy_kw = fold_weights(Z, w, coeff, p=p, convention='mean1',
                                   exponent=power)
        kw.update(scipy_kw)
    n = len(Z)
    k_eff = ss._k_eff(n, K_MAX, exclude_groups)
    bad = set()

    def _d(a, b):
        D = cdist(Z[a:b], Z, metric=scipy_name, **kw)
        nan = np.isnan(D)
        if nan.any():
            bad.update((np.where(nan.any(1))[0] + a).tolist())
            D[nan] = np.inf
        return D
    idx, _ = ss._lists_from_distance(_d, n, n, k_eff, self_rows=True,
                                     exclude_groups=exclude_groups)
    return idx, len(bad)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def detection_auroc(y, mask, idx, is_clf):
    """(AUROC on the mean gap, AUROC on the median gap). The second is None on
    classification, where the gap is the share of neighbours that disagree."""
    nb = y[idx]
    if is_clf:
        gap = (nb != y[:, None]).mean(axis=1)
        return float(roc_auc_score(mask, gap)), None
    mean_gap = np.abs(y - nb.mean(axis=1))
    med_gap = np.abs(y - np.median(nb, axis=1))
    return (float(roc_auc_score(mask, mean_gap)),
            float(roc_auc_score(mask, med_gap)))


def shared_share(a, b, k):
    """Mean over rows of |a's first k ∩ b's first k| / k."""
    A, B = a[:, :k], b[:, :k]
    return float((A[:, :, None] == B[:, None, :]).any(axis=2).sum(axis=1).mean()
                 / k)


def overlap_pairs(descs):
    """Pairs differing in exactly one of (space, distance, weighting+power), plus
    every pair with model_input, tanimoto_ecfp4 or forest_proximity in it."""
    single = (ss.MODEL_INPUT_SPACE, ss.TANIMOTO_SPACE, ss.FOREST_SPACE)
    pairs = []
    for i in range(len(descs)):
        for j in range(i + 1, len(descs)):
            a, b = descs[i], descs[j]
            if a['space'] in single or b['space'] in single:
                pairs.append((i, j))
                continue
            diff = ((a['space'] != b['space'])
                    + (a['distance'] != b['distance'])
                    + ((a['weighting'], a['power'])
                       != (b['weighting'], b['power'])))
            if diff == 1:
                pairs.append((i, j))
    return pairs


# ---------------------------------------------------------------------------
# One task
# ---------------------------------------------------------------------------

def task_path(out_dir, dataset, condition, fold_i, draw, model, overlap=False):
    sfx = '.overlap.jsonl' if overlap else '.jsonl'
    return (Path(out_dir) / dataset
            / f'{condition}__fold{fold_i}__draw{draw}__{model}{sfx}')


def done_keys(path):
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            keys.add((r['setting'], r.get('model_input'), r['grouped']))
    return keys


def _append(path, row):
    with open(path, 'a') as fh:
        fh.write(json.dumps(row) + '\n')
        fh.flush()
        os.fsync(fh.fileno())


def run_task(dataset, condition, fold_i, draw, model, out_dir):
    if importance_exponent_override() is not None:
        raise SystemExit('KIRBY_IMPORTANCE_EXPONENT is set; f38 sweeps the power '
                         'itself, so unset it.')
    t0 = time.time()
    out = task_path(out_dir, dataset, condition, fold_i, draw, model)
    out_ov = task_path(out_dir, dataset, condition, fold_i, draw, model, True)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(out)

    data, _rep_names = load_noise_data(dataset)
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    smiles = data.get('smiles')
    base = {'dataset': dataset, 'condition': condition, 'fold': fold_i,
            'draw': draw, 'model': model}

    def skipped(why):
        _append(out, dict(base, setting='_skipped', model_input=None,
                          grouped=None, reason=why))
        print(f'  skipped: {why}')

    if condition not in conditions_for_task(data):
        return skipped(f'{condition} is not a condition for a {task_type} set')
    if not is_clf and condition not in DETECTION_IS_DEFINED:
        return skipped('every training label moves under this condition, so '
                       'there is no clean set to tell apart')
    if model not in DOWNSTREAM_MODELS_NOISE:
        return skipped(f'{model} is not in DOWNSTREAM_MODELS_NOISE')

    # The precompute: this model's recipes and its top-2 single reps (n42).
    metric = get_domain_metric(data)
    factories = get_model_factories([model], task_type=task_type,
                                    n_samples=len(labels))
    lookup = functools.partial(
        get_best_search_for_downstream, reps, labels, groups, budget=None,
        augmentations=data.get('augmentations'), dataset_name=dataset,
        task_type=task_type, data=data, pool_subset=n42.POOL_SUBSET)
    by_model, _no_recipe = pool_recipe_per_downstream_model(lookup, [model])
    recipes, recipes_unavail = n42._recipes_for(by_model, [model])
    base_by_model, base_missing = noise_base_reps_by_model(
        {m: by_model[m]['search'] for m in by_model}, [model], reps=reps,
        dataset_name=dataset, verbose=False)

    folds = list(cv_split(groups, 5, next(iter(reps.values())), labels))
    train_idx, val_idx = (np.asarray(a) for a in folds[fold_i])
    level = CLF_FLIP_RATE if is_clf else reference_noise_level(dataset)
    spec = noise_spec_for(dataset, condition, level, noise_seed=draw,
                          is_clf=is_clf)
    y0, mask, _prov = inject_train_labels(
        labels, spec, train_idx=train_idx, is_clf=is_clf,
        groups=noise_groups_for(data)[0], dataset_name=dataset)
    y0 = np.asarray(y0)
    mask = np.asarray(mask, dtype=bool)
    n_corrupt = int(mask.sum())
    base.update(level=float(level), n_train=int(len(y0)),
                n_corrupted=n_corrupt)
    if n_corrupt in (0, len(mask)):
        return skipped(f'{n_corrupt} of {len(mask)} training molecules '
                       'corrupted; AUROC needs both kinds')
    g_tr = None if groups is None else np.asarray(groups)[train_idx]
    smi_tr = None if smiles is None else [smiles[i] for i in train_idx]

    settings = n07.sweep_settings()
    mi_settings = [ss.SimilaritySetting(sp, 'none', 'manhattan')
                   for sp in MODEL_INPUTS]
    sp_tr, sp_va, sp_unavail = n07.fold_similarity_spaces(
        dataset, list(settings) + mi_settings, model_input_tr=None,
        model_input_va=None, reps=reps, train_idx=train_idx,
        val_idx=val_idx, y_train=y0, groups_train=g_tr,
        base_reps=base_by_model.get(model, []),
        recipes=recipes.get(model, {}), augmentations=data.get('augmentations'),
        model_name=model, model_factory=factories.get(model), metric=metric,
        task_type=task_type)
    unavailable = dict(sp_unavail)
    for sp, why in (recipes_unavail.get(model) or {}).items():
        unavailable.setdefault(sp, why)
    if base_missing.get(model):
        unavailable.setdefault('base', base_missing[model])
    print(f'  {dataset} {condition} fold {fold_i} draw {draw} {model}: '
          f'{n_corrupt}/{len(y0)} training molecules corrupted; spaces '
          f'{sorted(sp_tr)}; unavailable {unavailable}')

    groupings = (('plain', None),) + (() if g_tr is None
                                      else (('grouped', g_tr),))
    plain_lists = []          # (desc, idx) for the overlap step

    def score_and_write(desc, mi, grouped, idx, extra=None):
        if (desc['name'], mi, grouped) in done:
            return
        row = dict(base, setting=desc['name'], space=desc['space'],
                   weighting=desc['weighting'], distance=desc['distance'],
                   power=desc['power'], model_input=mi, grouped=grouped,
                   **(extra or {}))
        for k in KS:
            if idx.shape[1] < k:
                row[f'auroc_mean_k{k}'] = row[f'auroc_median_k{k}'] = None
                continue
            a_mean, a_med = detection_auroc(y0, mask, idx[:, :k], is_clf)
            row[f'auroc_mean_k{k}'] = a_mean
            row[f'auroc_median_k{k}'] = a_med
        _append(out, row)

    # 1. The noise suite's own feature-space settings, and tanimoto_ecfp4.
    weights_by = {}
    for s in settings:
        if s.space in (ss.MODEL_INPUT_SPACE, ss.FOREST_SPACE):
            continue
        power = (None if s.weighting == 'none' or s.space == ss.TANIMOTO_SPACE
                 else implied_power(s.metric))
        desc = describe(s.space, s.weighting, s.metric, power)
        for grouped, eg in groupings:
            got = n07.fold_neighbours(dataset, [s], sp_tr, None, y0,
                                      exclude_groups=eg, smiles_train=smi_tr,
                                      is_clf=is_clf).get(s.name)
            if not got or not got.get('available'):
                unavailable.setdefault(desc['name'],
                                       (got or {}).get('reason', 'not built'))
                break
            if got.get('weights') is not None:
                weights_by[(s.space, s.weighting)] = np.asarray(got['weights'])
            idx = np.asarray(got['idx_train'])
            score_and_write(desc, None, grouped, idx)
            if grouped == 'plain':
                plain_lists.append((desc, idx))

    # 2. f38's additions on the same matrices.
    for desc in feature_space_settings(sp_tr):
        if is_noise_suite_setting(desc['weighting'], desc['distance'],
                                  desc['power']):
            continue
        Z = sp_tr[desc['space']]
        w = None
        if desc['weighting'] != 'none':
            key = (desc['space'], desc['weighting'])
            if key not in weights_by:
                weights_by[key] = ss._importance_weights(Z, y0,
                                                         desc['weighting'])
            w = weights_by[key]
        for grouped, eg in groupings:
            need = (desc['name'], None, grouped) not in done
            if not need and grouped != 'plain':
                continue
            idx, n_bad = own_lists(Z, desc['distance'], w, desc['power'], eg)
            score_and_write(desc, None, grouped, idx,
                            {'rows_with_undefined_distance': n_bad})
            if grouped == 'plain':
                plain_lists.append((desc, idx))

    # 3. model_input and forest_proximity, once per model input.
    for mi in MODEL_INPUTS:
        if mi not in sp_tr:
            continue
        spaces = dict(sp_tr, model_input=sp_tr[mi])
        for s in (ss.MODEL_INPUT,
                  ss.SimilaritySetting(ss.FOREST_SPACE, 'none', 'forest')):
            desc = describe(s.space, 'none', s.metric, None)
            desc = dict(desc, name=f"{desc['name']}@{mi}")
            for grouped, eg in groupings:
                got = n07.fold_neighbours(dataset, [s], spaces, None, y0,
                                          exclude_groups=eg, is_clf=is_clf
                                          ).get(s.name)
                if not got or not got.get('available'):
                    unavailable.setdefault(desc['name'],
                                           (got or {}).get('reason'))
                    break
                if s.space == ss.MODEL_INPUT_SPACE:
                    # One exact list per k (similarity_space.neighbours_at).
                    if (desc['name'], mi, grouped) in done:
                        idx20 = ss.neighbours_at(got, max(n07.usable_ks(got, KS)))[0]
                    else:
                        row = dict(base, setting=desc['name'], space=s.space,
                                   weighting='none', distance=s.metric,
                                   power=None, model_input=mi, grouped=grouped)
                        for k in KS:
                            if k not in n07.usable_ks(got, KS):
                                row[f'auroc_mean_k{k}'] = None
                                row[f'auroc_median_k{k}'] = None
                                continue
                            i_k = ss.neighbours_at(got, k)[0]
                            a, b = detection_auroc(y0, mask, i_k, is_clf)
                            row[f'auroc_mean_k{k}'] = a
                            row[f'auroc_median_k{k}'] = b
                        _append(out, row)
                        idx20 = ss.neighbours_at(got, max(n07.usable_ks(got, KS)))[0]
                    if grouped == 'plain':
                        plain_lists.append((desc, np.asarray(idx20)))
                else:
                    idx = np.asarray(got['idx_train'])
                    score_and_write(desc, mi, grouped, idx)
                    if grouped == 'plain':
                        plain_lists.append((desc, idx))

    # 4. Neighbour-list overlap, plain lists only.
    descs = [d for d, _ in plain_lists]
    idxs = [i for _, i in plain_lists]
    pairs = overlap_pairs(descs)
    tmp = out_ov.with_suffix('.tmp')
    with open(tmp, 'w') as fh:
        for i, j in pairs:
            row = dict(base, a=descs[i]['name'], b=descs[j]['name'])
            for k in KS:
                kk = min(k, idxs[i].shape[1], idxs[j].shape[1])
                row[f'shared_k{k}'] = (shared_share(idxs[i], idxs[j], kk)
                                       if kk == k else None)
            fh.write(json.dumps(row) + '\n')
        fh.write(json.dumps(dict(base, a='_unavailable', b=None,
                                 unavailable=unavailable)) + '\n')
    os.replace(tmp, out_ov)
    print(f'  {len(descs)} settings scored, {len(pairs)} overlap pairs, '
          f'{time.time() - t0:.0f}s -> {out}')
    return 0


# ---------------------------------------------------------------------------
# Task list
# ---------------------------------------------------------------------------

def all_noise_datasets():
    """The noise suite's full dataset list: stage1_prelim_noise.py --all-datasets."""
    sys.path.insert(0, str(_HERE.parents[1] / 'slurm_scripts' / 'runbook'))
    from experiments.complexity.complexity_helpers import DATASET_REPS
    from stage1_prelim_noise import EXCLUDED
    return sorted(set(DATASET_REPS) - set(EXCLUDED))


def list_tasks(datasets, models, path):
    """Every task, cheapest first: by training molecules, then condition."""
    tasks = []
    for ds in datasets:
        try:
            data, _ = load_noise_data(ds)
        except Exception as exc:
            print(f'  {ds}: not loaded ({type(exc).__name__}: {exc}); left out')
            continue
        is_clf = data['task_type'] == 'classification'
        conds = [c for c in conditions_for_task(data)
                 if is_clf or c in DETECTION_IS_DEFINED]
        n = len(data['labels'])
        for c in conds:
            for fi in range(5):
                for d in DRAWS:
                    for m in models:
                        tasks.append((n, f'{ds} {c} {fi} {d} {m}'))
    tasks.sort()
    Path(path).write_text('\n'.join(t for _, t in tasks) + '\n')
    print(f'{len(tasks)} tasks written to {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=None,
                    help='default: every noise-suite dataset')
    ap.add_argument('--models', nargs='+', default=list(DOWNSTREAM_MODELS_NOISE))
    ap.add_argument('--out', default=str(OUT_DEFAULT))
    ap.add_argument('--list-tasks', metavar='PATH')
    ap.add_argument('--task', nargs=5,
                    metavar=('DATASET', 'CONDITION', 'FOLD', 'DRAW', 'MODEL'))
    a = ap.parse_args()
    from experiments import dataset_policy  # noqa: F401  (fills the registry)
    if a.list_tasks:
        return list_tasks(a.datasets or all_noise_datasets(), a.models,
                          a.list_tasks)
    if a.task:
        ds, cond, fi, d, m = a.task
        return run_task(ds, cond, int(fi), int(d), m, a.out)
    ap.error('give --list-tasks or --task')


if __name__ == '__main__':
    sys.exit(main())
