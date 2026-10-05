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
# ONE THREAD PER TASK. Parallelism comes from running many tasks at once
# (--jobs). Measured 2026-10-04 on the laptop: with these unset, deeplift after
# a lightgbm fit segfaults (two OpenMP runtimes, torch's and LightGBM's); on the
# server 16 tasks each starting threads for every core on the node made no
# progress in 25 minutes. With them at 1, lightgbm_gain, treeshap and deeplift
# on herg_ki's pdv and ecfp4 take 0.3 to 7.4 s each. Set before numpy, torch
# and sklearn are imported.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'LOKY_MAX_CPU_COUNT',
           'KIRBY_LGBM_THREADS', 'KIRBY_RF_JOBS'):
    os.environ[_v] = '1'
# f38 never runs a hybrid search itself, not even the small top-up the guard
# otherwise allows (find_best_hybrid._refuse_cold_search). The first server run,
# herg_ki censoring 0 0 dnn on 2026-10-04, found 7824 of 8160 configurations
# stored, started on the 336 left and ran out of memory. A missing search now
# fails the task within seconds, and the log names the precompute command that
# fills it.
os.environ['KIRBY_COLD_SEARCH_MAX_CONFIGS'] = '0'

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
sys.path.insert(0, str(_HERE))   # f18_distance_metric_sweep

from experiments._shared import (  # noqa: E402
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS_NOISE)
from experiments.dataset_policy import (  # noqa: E402
    noise_groups_for, reference_noise_level)
from experiments.noise.noise_helpers import (  # noqa: E402
    noise_spec_for, inject_train_labels, conditions_for_task,
    load_noise_data, DETECTION_IS_DEFINED)
from experiments.noise import similarity_space as ss  # noqa: E402
import f18_distance_metric_sweep as f18  # noqa: E402
from experiments.noise import n07_remediation as n07  # noqa: E402
from experiments.noise import n42_similarity_noise_sweep as n42  # noqa: E402
from kirby.similarity import neighbour_weighting  # noqa: E402
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

def _correlation_square(Z, w=None):
    """Correlation distance between every pair of rows, optionally weighted, as
    one matrix product. Same formula as scipy's correlation(u, v, w): centre each
    row on its w-weighted mean, then 1 - the w-weighted cosine. scipy's cdist
    computes the weighted form one pair at a time in Python, about 1.3 million
    calls on 1129 molecules; this is the same numbers in one product. A row with
    no spread gives NaN, as scipy does."""
    v = np.ones(Z.shape[1]) if w is None else np.asarray(w, dtype=np.float64)
    v = v / v.sum()
    C = (Z - (Z @ v)[:, None]) * np.sqrt(v)[None, :]
    nrm = np.sqrt((C * C).sum(axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        Cn = C / nrm[:, None]
    D = 1.0 - Cn @ Cn.T
    np.clip(D, 0.0, 2.0, out=D)
    return D


def _minkowski_square(Z, p, chunk_elements=50_000_000):
    """Fractional Minkowski between every pair of rows, in torch row blocks
    (the same arithmetic as f34_full_sweep._fractional_minkowski)."""
    import torch
    T = torch.from_numpy(np.ascontiguousarray(Z, dtype=np.float64))
    n, d = T.shape
    ch = max(1, int(chunk_elements // max(1, n * d)))
    D = np.empty((n, n), dtype=np.float64)
    for s0 in range(0, n, ch):
        e = min(s0 + ch, n)
        D[s0:e] = ((T[s0:e, None, :] - T[None, :, :])
                   .abs_().pow_(p).sum(-1).pow_(1.0 / p)).numpy()
    return D


def minkowski_squares(Z, p, columns):
    """{key: D} for several weightings of one fractional Minkowski in one pass.

    Folding v**a into the features gives |v**a x - v**a y|**p = v**(a p) |x - y|**p,
    so every weighting is the same per-feature term |x - y|**p summed against a
    different column. `columns` is {key: per-feature multiplier of that term}
    (ones for no weighting). The pass over pairs is done once, in torch row
    blocks, instead of once per weighting. The caller passes as many columns as
    fit in memory at once."""
    import torch
    T = torch.from_numpy(np.ascontiguousarray(Z, dtype=np.float64))
    n, d = T.shape
    keys = list(columns)
    W = torch.from_numpy(np.stack([np.asarray(columns[k], dtype=np.float64)
                                   for k in keys], axis=1))
    out = np.empty((len(keys), n, n), dtype=np.float64)
    ch = max(1, int(50_000_000 // max(1, n * d)))
    for s0 in range(0, n, ch):
        e = min(s0 + ch, n)
        E = (T[s0:e, None, :] - T[None, :, :]).abs_().pow_(p)
        out[:, s0:e, :] = (E @ W).pow_(1.0 / p).permute(2, 0, 1).numpy()
    return {k: out[i] for i, k in enumerate(keys)}


def square_distance(Z_tr, distance, w, power):
    """(D, n_rows_with_undefined_distance): every training row against every
    other under one setting, computed once and shared by the plain and the
    group-excluding lists.

    The matrices are already z-scored (n07.fold_similarity_spaces), which
    fold_weights requires. Under manhattan, euclidean and cosine the importance
    weight is folded into the features at `power` (fold_weights); under
    minkowski p=0.5 likewise; under correlation it goes inside the formula.
    Undefined distances (cosine or correlation on a row with no spread) become
    +inf, so such a row is nobody's neighbour, and are counted."""
    scipy_name, kw, coeff, p = DISTANCES[distance]
    Z = np.asarray(Z_tr, dtype=np.float64)
    if distance == 'correlation':
        D = _correlation_square(Z, None if w is None else
                                fold_weights(Z, w, coeff, convention='mean1')[1]['w'])
    else:
        if w is not None:
            Z, _ = fold_weights(Z, w, coeff, p=p, convention='mean1',
                                exponent=power)
        if distance == 'minkowski_p0.5':
            D = _minkowski_square(Z, 0.5)
        else:
            D = cdist(Z, Z, metric=scipy_name, **kw)
    bad = np.isnan(D)
    n_bad = int(bad.any(axis=1).sum())
    D[bad] = np.inf
    return D, n_bad


def lists_from_square(D, exclude_groups):
    """(idx, dist): each row's K_MAX nearest other rows, nearest first, ties by
    column index (similarity_space._lists_from_distance, the noise suite's own
    list builder), the row itself and, if given, its group excluded."""
    n = len(D)
    k_eff = ss._k_eff(n, K_MAX, exclude_groups)
    return ss._lists_from_distance(lambda a, b: D[a:b].copy(), n, n, k_eff,
                                   self_rows=True,
                                   exclude_groups=exclude_groups)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

# How much each neighbour counts in the neighbours' label. f34's five schemes
# (neighbour_weighting.WEIGHTING_SCHEMES without inverse_sqrt, which f34 dropped
# with a measured reason: NEIGHBOUR_WEIGHTING_AND_DISTANCE_RESULTS.md).
SCHEMES = ('uniform', 'inverse', 'inverse_square', 'gaussian', 'rank')


def _weighted_value(y_noisy, idx, dist, scheme, is_clf):
    """Each molecule's neighbours' label under one scheme: the weighted mean on
    regression; on classification the weighted share of neighbours whose label
    differs from the molecule's own, and the weighted majority class."""
    nb = y_noisy[idx]
    w = neighbour_weighting.weight_array(dist, scheme)
    if is_clf:
        disagree = (w * (nb != y_noisy[:, None])).sum(1) / w.sum(1)
        classes = np.unique(y_noisy)
        votes = np.stack([(w * (nb == c)).sum(1) for c in classes], axis=1)
        return disagree, classes[votes.argmax(1)]
    return (w * nb).sum(1) / w.sum(1), None


def detection_auroc(y, mask, idx, is_clf, dist=None, scheme='uniform'):
    """(AUROC of the mask against the gap under `scheme`, AUROC against the gap
    to the unweighted median). The median is what f23's simeval uses; it is
    None on classification, where the gap is the (weighted) share of
    neighbours that disagree."""
    d = np.ones(idx.shape) if dist is None else dist
    v, _ = _weighted_value(y, idx, d, scheme, is_clf)
    if is_clf:
        return float(roc_auc_score(mask, v)), None
    med_gap = np.abs(y - np.median(y[idx], axis=1))
    return (float(roc_auc_score(mask, np.abs(y - v))),
            float(roc_auc_score(mask, med_gap)))


def recovery(y_noisy, y_clean, mask, idx, is_clf, dist=None, scheme='uniform'):
    """How close the neighbours' label is to the CLEAN label on the corrupted
    molecules -- what label_smooth, knn_smooth and n02's weights rely on.

    regression      (RMSE of the neighbours' weighted mean against the clean
                     label, RMSE of the noisy label against the clean label),
                     both in the label's own units, over the corrupted
                     molecules. The second is the same for every setting: it is
                     what the molecule carries before any cleaning.
    classification  (share of corrupted molecules whose neighbours' weighted
                     majority is the clean label, None)
    """
    m = np.asarray(mask, dtype=bool)
    d = np.ones(idx.shape) if dist is None else dist
    v, maj = _weighted_value(y_noisy, idx, d, scheme, is_clf)
    yc = y_clean[m]
    if is_clf:
        return float((maj[m] == yc).mean()), None
    return (float(np.sqrt(np.mean((v[m] - yc) ** 2))),
            float(np.sqrt(np.mean((y_noisy[m] - yc) ** 2))))


def scores_at(y_noisy, y_clean, mask, idx, dist, k, is_clf):
    """Every score of one setting at one k, as row fields: detection AUROC and
    recovery under each of SCHEMES, plus the unweighted-median AUROC."""
    out = {}
    for sc in SCHEMES:
        a, med = detection_auroc(y_noisy, mask, idx, is_clf, dist, sc)
        rec, before = recovery(y_noisy, y_clean, mask, idx, is_clf, dist, sc)
        out[f'auroc_{sc}_k{k}'] = a
        out[f'recovery_{sc}_k{k}'] = rec
        out['recovery_before_cleaning'] = before
    out[f'auroc_median_k{k}'] = med
    return out


def empty_at(k):
    out = {f'auroc_median_k{k}': None}
    for sc in SCHEMES:
        out[f'auroc_{sc}_k{k}'] = out[f'recovery_{sc}_k{k}'] = None
    return out


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
# Recipes and base representations, from what is stored only
# ---------------------------------------------------------------------------
#
# Owner, 2026-10-04: the precompute is to be redone per fold on a much smaller
# grid, but the stored results can be used for testing. So f38 reads the stored
# rows directly and never runs a search: a recipe that is not stored makes its
# space unavailable, recorded with the reason. These recipes come from the old
# whole-dataset searches, partly filled, so hybrid, hybrid__general2 and
# hybrid_similarity here are test spaces, not the ones the rebuilt precompute
# will give.
#
#   hybrid             the best stored row for this model: tier 'general',
#                      objective 'predictive', augmentation none or uncapped
#   hybrid__general2   the next-best stored row there with a different recipe.
#                      NOT _shared.get_rule_hybrids' rank-2 rule, which needs the
#                      full search
#   hybrid_similarity  the best stored row under tier 'general__obj-similarity'
#   base_1, base_2     the model's top two single representations by its cached
#                      5-fold baseline (cached_baseline_top2); where
#                      those files are missing, ranked once on this run's folds
#                      and kept in <out>/<dataset>/base_reps.json

def _labels_hash(labels):
    from kirby.find_best_hybrid import _hash_bytes, _canonical_bytes
    out = {_hash_bytes(_canonical_bytes(np.asarray(labels)))}
    try:
        out.add(_hash_bytes(_canonical_bytes(np.asarray(labels, dtype=float))))
    except (TypeError, ValueError):
        pass
    return {str(h) for h in out}


def stored_recipes(dataset, model, labels, rep_names):
    """({space: (Blueprint, aug_condition)}, {space: why unavailable})."""
    import pickle
    import sqlite3
    from kirby.paths import cache_root
    from kirby.find_best_hybrid import BUILTIN_METRICS
    path = cache_root() / 'result_db' / f'{dataset}.db'
    want = {'hybrid': ('general', 'predictive', 0),
            'hybrid__general2': ('general', 'predictive', 1),
            'hybrid_similarity': ('general__obj-similarity', 'similarity', 0)}
    if not path.exists():
        return {}, {sp: f'no result database at {path}' for sp in want}
    con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    rows = con.execute(
        "SELECT tier, objective, aug_strategy, metric, score_mean, labels_hash, "
        "blueprint FROM results WHERE model=? AND blueprint IS NOT NULL "
        "AND score_mean IS NOT NULL AND aug_strategy IN ('none', 'uncapped') "
        "AND tier IN ('general', 'general__obj-similarity')", (model,)).fetchall()
    con.close()
    hashes = _labels_hash(labels)
    recipes, why = {}, {}
    for sp, (tier, obj, rank) in want.items():
        cand = [r for r in rows if r[0] == tier and r[1] == obj
                and str(r[5]) in hashes]
        if not cand:
            why[sp] = (f'no stored {tier}/{obj} row for {model} whose labels '
                       f'match the loaded data')
            continue
        lower = BUILTIN_METRICS.get(cand[0][3], (None, False))[1]
        cand.sort(key=lambda r: r[4], reverse=not lower)
        # Only recipes whose representations are all loaded can be rebuilt; a
        # stored row from an older representation list is passed over.
        seen, picked, n_other_pool, n_unbuildable = [], None, 0, 0
        for r in cand:
            if r[6] in seen:
                continue
            bp_r = pickle.loads(r[6])
            if any(x not in rep_names for x in (getattr(bp_r, 'reps', None) or [])):
                n_other_pool += 1
                continue
            # A recipe searched over a subset of the pool (rep_count below the
            # pool size) can only be rebuilt if it carries the per-rep scores
            # that say WHICH subset; build_fold_hybrid refuses otherwise
            # (_shared.py:5339-5352). The first server task hit this on
            # herg_ki's dnn similarity recipe (rep_count=5 of 10).
            rc = getattr(bp_r, 'rep_count', None)
            n_pool = len(getattr(bp_r, 'reps', None) or rep_names)
            if (rc is not None and int(rc) < n_pool
                    and not getattr(bp_r, 'rep_scores', None)):
                n_unbuildable += 1
                continue
            seen.append(r[6])
            if len(seen) - 1 == rank:
                picked, bp = r, bp_r
                break
        if picked is None:
            why[sp] = (f'fewer than {rank + 1} distinct stored recipes that can '
                       f'be rebuilt ({n_other_pool} use representations not '
                       f'loaded; {n_unbuildable} used a subset of the pool '
                       f'without recording which)')
            continue
        aug = picked[2] or 'none'
        try:
            bp.augmentation_strategy = aug
        except Exception:
            pass
        recipes[sp] = (bp, aug)
        if n_other_pool or n_unbuildable:
            why[f'{sp}__passed_over'] = (
                f'{n_other_pool} better-ranked stored recipes use representations '
                f'not loaded; {n_unbuildable} used a subset of the pool without '
                f'recording which')
    return recipes, why


def cached_baseline_top2(dataset, reps, model, metric, lower, n_samples,
                         labels_hash):
    """Top two representations for one model from the cached 5-fold baselines,
    reading only files fitted on the whole dataset (same molecule count and
    labels). Copied from f34_loss_functions.top_base_reps, which is not in git."""
    import pickle
    from kirby.paths import cache_root
    scored = []
    for rep in reps:
        f = (cache_root() / 'baselines'
             / f'{dataset}__{rep}__{model}__{metric}__5f_42.pkl')
        try:
            d = pickle.load(open(f, 'rb'))
            meta, payload = d['_cache_meta'], d['payload']
        except Exception:
            continue
        if (meta.get('n_samples') != n_samples
                or str(meta.get('labels_hash')) != str(labels_hash)):
            continue
        scored.append((float(payload['score']), rep))
    scored.sort(reverse=not lower)
    return [r for _s, r in scored[:2]]


def base_reps_for(dataset, model, data, labels, groups, task_type, out_dir):
    """The model's top two single representations (see the block comment)."""
    from kirby.find_best_hybrid import BUILTIN_METRICS, get_model
    metric_name = get_domain_metric(data)[0]
    metric_fn, lower = BUILTIN_METRICS[metric_name]
    reps = data['reps']
    for h in _labels_hash(labels):
        base = cached_baseline_top2(dataset, list(reps), model, metric_name,
                                    lower, len(labels), h)
        if len(base) == 2:
            return base, 'cached 5-fold baselines'
    store = Path(out_dir) / dataset / 'base_reps.json'
    store.parent.mkdir(parents=True, exist_ok=True)
    # One process ranks; every other task on this (dataset, model) waits on the
    # lock and reads the file, instead of all of them fitting the same 50
    # models at once.
    import fcntl
    lock_fh = open(store.with_suffix('.lock'), 'w')
    fcntl.flock(lock_fh, fcntl.LOCK_EX)
    got = json.loads(store.read_text()) if store.exists() else {}
    if model not in got:
        from sklearn.preprocessing import StandardScaler
        splits = list(cv_split(groups, 5, next(iter(reps.values())), labels))
        means = {}
        for rep, X in reps.items():
            sc = []
            for tr, va in splits:
                z = StandardScaler().fit(X[tr])
                est = get_model(model, task_type)
                est.fit(np.nan_to_num(z.transform(X[tr])), labels[tr])
                sc.append(float(metric_fn(labels[va], est.predict(
                    np.nan_to_num(z.transform(X[va]))))))
            means[rep] = float(np.mean(sc))
        got[model] = sorted(means, key=means.get, reverse=not lower)[:2]
        store.parent.mkdir(parents=True, exist_ok=True)
        tmp = store.with_suffix('.tmp')
        tmp.write_text(json.dumps(got, indent=1))
        os.replace(tmp, store)
    fcntl.flock(lock_fh, fcntl.LOCK_UN)
    lock_fh.close()
    return got[model], 'ranked on this run\'s 5 folds (no cached baselines)'


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


_STAGES = {}
_TASK_LOCK = None


def _stage(name, t_prev):
    """Print and record how long the step that just ended took."""
    now = time.time()
    _STAGES[name] = round(now - t_prev, 1)
    print(f'  [stage] {name}: {now - t_prev:.1f}s', flush=True)
    return now


def run_task(dataset, condition, fold_i, draw, model, out_dir):
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    if importance_exponent_override() is not None:
        raise SystemExit('KIRBY_IMPORTANCE_EXPONENT is set; f38 sweeps the power '
                         'itself, so unset it.')
    t0 = time.time()
    out = task_path(out_dir, dataset, condition, fold_i, draw, model)
    out_ov = task_path(out_dir, dataset, condition, fold_i, draw, model, True)
    out.parent.mkdir(parents=True, exist_ok=True)
    # One process per task, across sruns: a task another process is running
    # is skipped here (that process finishes it). The lock is released when
    # the holder exits, however it exits.
    import fcntl
    global _TASK_LOCK
    _TASK_LOCK = open(out.with_suffix('.lock'), 'w')
    try:
        fcntl.flock(_TASK_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f'  running in another process, skipped here: {out}')
        return 0
    done = done_keys(out)
    if ('_done', None, None) in done or ('_skipped', None, None) in done:
        print(f'  already done: {out}')
        return 0

    data, _rep_names = load_noise_data(dataset)
    t_s = _stage('load data', t0)
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

    # This model's recipes and its top-2 single reps, from stored rows only.
    metric = get_domain_metric(data)
    factories = get_model_factories([model], task_type=task_type,
                                    n_samples=len(labels))
    recipes_m, recipes_unavail = stored_recipes(dataset, model, labels,
                                                set(reps))
    t_s = _stage('read stored recipes', t_s)
    base_names, base_source = base_reps_for(dataset, model, data, labels,
                                            groups, task_type, out_dir)
    t_s = _stage('base representations', t_s)
    base.update(base_reps=list(base_names), base_reps_source=base_source,
                recipes={sp: {'aug': aug, 'reps': list(getattr(bp, 'reps', []) or []),
                              'budget': getattr(bp, 'budget', None)}
                         for sp, (bp, aug) in recipes_m.items()})

    folds = list(cv_split(groups, 5, next(iter(reps.values())), labels))
    train_idx, val_idx = (np.asarray(a) for a in folds[fold_i])
    level = CLF_FLIP_RATE if is_clf else reference_noise_level(dataset)
    spec = noise_spec_for(dataset, condition, level, noise_seed=draw,
                          is_clf=is_clf)
    y0, mask, _prov = inject_train_labels(
        labels, spec, train_idx=train_idx, is_clf=is_clf,
        groups=noise_groups_for(data)[0], dataset_name=dataset)
    y0 = np.asarray(y0)
    y_clean = labels[train_idx]
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
        base_reps=list(base_names),
        recipes=recipes_m, augmentations=data.get('augmentations'),
        model_name=model, model_factory=factories.get(model), metric=metric,
        task_type=task_type)
    t_s = _stage('build feature spaces (hybrids on noisy labels)', t_s)
    unavailable = dict(sp_unavail)
    for sp, why in recipes_unavail.items():
        unavailable.setdefault(sp, why)
    print(f'  {dataset} {condition} fold {fold_i} draw {draw} {model}: '
          f'{n_corrupt}/{len(y0)} training molecules corrupted; spaces '
          f'{sorted(sp_tr)}; unavailable {unavailable}')

    groupings = (('plain', None),) + (() if g_tr is None
                                      else (('grouped', g_tr),))
    plain_lists = []          # (desc, idx) for the overlap step

    def score_and_write(desc, mi, grouped, idx, dist, extra=None):
        if (desc['name'], mi, grouped) in done:
            return
        row = dict(base, setting=desc['name'], space=desc['space'],
                   weighting=desc['weighting'], distance=desc['distance'],
                   power=desc['power'], model_input=mi, grouped=grouped,
                   **(extra or {}))
        for k in KS:
            row.update(empty_at(k) if idx.shape[1] < k else
                       scores_at(y0, y_clean, mask, idx[:, :k], dist[:, :k],
                                 k, is_clf))
        _append(out, row)

    # 1. tanimoto_ecfp4, through the noise suite's own code and cache.
    for s in settings:
        if s.space != ss.TANIMOTO_SPACE:
            continue
        desc = describe(s.space, s.weighting, s.metric, None)
        for grouped, eg in groupings:
            got = n07.fold_neighbours(dataset, [s], sp_tr, None, y0,
                                      exclude_groups=eg, smiles_train=smi_tr,
                                      is_clf=is_clf).get(s.name)
            if not got or not got.get('available'):
                unavailable.setdefault(desc['name'],
                                       (got or {}).get('reason', 'not built'))
                break
            idx = np.asarray(got['idx_train'])
            score_and_write(desc, None, grouped, idx,
                            np.asarray(got['dist_train']))
            if grouped == 'plain':
                plain_lists.append((desc, idx))
    t_s = _stage('1. tanimoto_ecfp4', t_s)

    # 2. Every feature-space setting: the noise suite's own (manhattan and
    # euclidean at the power the distance implies) and f38's additions. The
    # noise suite's are computed here rather than through n07.fold_neighbours
    # because that path recomputes the importance for every list; the
    # arithmetic is the same (fold_weights, cdist, _lists_from_distance), checked
    # list for list in tests/test_f38_neighbour_screen.py. Importance is
    # computed once per (space, method); each distance matrix once per setting.
    weights_by = {}
    mink = {}            # desc name -> D, filled one space at a time
    for desc in feature_space_settings(sp_tr):
        todo = [g for g in groupings
                if (desc['name'], None, g[0]) not in done or g[0] == 'plain']
        Z = sp_tr[desc['space']]
        w = None
        if desc['weighting'] != 'none':
            key = (desc['space'], desc['weighting'])
            if key not in weights_by:
                weights_by[key] = ss._importance_weights(Z, y0,
                                                         desc['weighting'])
            w = weights_by[key]
        if desc['distance'] == 'minkowski_p0.5':
            if desc['name'] not in mink:
                # Every weighting of this space's minkowski in one pass.
                from kirby.similarity.preprocessing import normalise_weights
                mink.clear()
                cols = {}
                for d2 in feature_space_settings({desc['space']: Z}):
                    if d2['distance'] != 'minkowski_p0.5':
                        continue
                    if d2['weighting'] == 'none':
                        cols[d2['name']] = np.ones(Z.shape[1])
                        continue
                    k2 = (desc['space'], d2['weighting'])
                    if k2 not in weights_by:
                        weights_by[k2] = ss._importance_weights(
                            Z, y0, d2['weighting'])
                    v = normalise_weights(weights_by[k2], 'mean1')
                    cols[d2['name']] = v ** (d2['power'] * 0.5)
                # Only as many matrices as fit in 1 GB at once, starting
                # from this setting (so a large dataset takes a few passes).
                names = list(cols)
                start = names.index(desc['name'])
                g = max(1, int(1e9 // (len(Z) ** 2 * 8)))
                cols = {k: cols[k] for k in names[start:start + g]}
                mink.update(minkowski_squares(np.asarray(Z, np.float64), 0.5,
                                              cols))
            D = mink.pop(desc['name'])
            n_bad = 0
        else:
            D, n_bad = square_distance(Z, desc['distance'], w, desc['power'])
        for grouped, eg in todo:
            idx, dist = lists_from_square(D, eg)
            score_and_write(desc, None, grouped, idx, dist,
                            {'rows_with_undefined_distance': n_bad})
            if grouped == 'plain':
                plain_lists.append((desc, idx))
        del D

    t_s = _stage('2. extra distances and powers', t_s)
    # 3. model_input and forest_proximity, once per model input.
    for mi in MODEL_INPUTS:
        if mi not in sp_tr:
            continue
        spaces = dict(sp_tr, model_input=sp_tr[mi])
        for s in (ss.MODEL_INPUT,
                  ss.SimilaritySetting(ss.FOREST_SPACE, 'none', 'forest')):
            desc = describe(s.space, 'none', s.metric, None)
            desc = dict(desc, name=f"{desc['name']}@{mi}")
            if s.space == ss.FOREST_SPACE:
                # One forest per model input, both list kinds read off one
                # distance matrix. Same forest, distance and tie order as
                # similarity_space.neighbour_lists (forest leaves, the share of
                # trees not shared, sorter='numpy'); that path refits the
                # forest for each list kind, about 60 s each on herg_ki.
                L_tr, _ = ss._forest_leaves(np.asarray(sp_tr[mi]), y0, is_clf)
                Df = ss._forest_distance(L_tr, L_tr)
                for grouped, eg in groupings:
                    k_eff = ss._k_eff(len(Df), K_MAX, eg)
                    idx, dist = ss._lists_from_distance(
                        lambda a, b: Df[a:b].copy(), len(Df), len(Df), k_eff,
                        self_rows=True, exclude_groups=eg, itemsize=4,
                        sorter='numpy')
                    score_and_write(desc, mi, grouped, idx, dist)
                    if grouped == 'plain':
                        plain_lists.append((desc, idx))
                del Df
                continue
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
                                row.update(empty_at(k))
                                continue
                            i_k, d_k = ss.neighbours_at(got, k)
                            row.update(scores_at(y0, y_clean, mask, i_k, d_k,
                                                 k, is_clf))
                        _append(out, row)
                        idx20 = ss.neighbours_at(got, max(n07.usable_ks(got, KS)))[0]
                    if grouped == 'plain':
                        plain_lists.append((desc, np.asarray(idx20)))
                else:
                    idx = np.asarray(got['idx_train'])
                    score_and_write(desc, mi, grouped, idx,
                                    np.asarray(got['dist_train']))
                    if grouped == 'plain':
                        plain_lists.append((desc, idx))

    t_s = _stage('3. model_input and forest_proximity', t_s)
    # 4. The raw representations behind base_1 and base_2, under every
    # coefficient f18 defines for what the array holds (binary or count;
    # f18.classify_representation). The noise suite only ever z-scores them,
    # which turns a fingerprint bit into a signed number; f34 found dice,
    # tanimoto and the other set-overlap coefficients win most fingerprint
    # representations for prediction. Unweighted only. A continuous base
    # representation is skipped: the z-scored base_k settings above cover it.
    for i_b, rep_name in enumerate(list(base_names)[:2], start=1):
        X = reps.get(rep_name)
        if X is None:
            continue
        X_tr = np.asarray(X)[train_idx].astype(np.float64)
        kind = f18.classify_representation(X_tr)
        if kind == 'continuous':
            continue
        space = f'base_{i_b}_raw'
        for coef, (fn, _needs_nonneg) in f18.coefficients_for(kind).items():
            desc = describe(space, 'none', coef, None)
            todo = [g for g in groupings
                    if (desc['name'], None, g[0]) not in done or g[0] == 'plain']
            try:
                D = fn(X_tr)
            except Exception as exc:
                unavailable.setdefault(desc['name'],
                                       f'{type(exc).__name__}: {exc}')
                continue
            if D is None:
                unavailable.setdefault(desc['name'], 'coefficient undefined')
                continue
            D = np.asarray(D, dtype=np.float64)
            D[np.isnan(D)] = np.inf
            for grouped, eg in todo:
                k_eff = ss._k_eff(len(X_tr), K_MAX, eg)
                idx, dist = ss._lists_from_distance(
                    lambda a, b: D[a:b].copy(), len(X_tr), len(X_tr), k_eff,
                    self_rows=True, exclude_groups=eg)
                score_and_write(desc, None, grouped, idx, dist,
                                {'representation': rep_name, 'kind': kind})
                if grouped == 'plain':
                    plain_lists.append((desc, idx))

    t_s = _stage('4. raw fingerprints', t_s)
    # 5. Neighbour-list overlap, plain lists only.
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
    t_s = _stage('5. overlap', t_s)
    _append(out, dict(base, setting='_done', model_input=None, grouped=None,
                      seconds=round(time.time() - t0, 1), stages=dict(_STAGES)))
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


# The two downstream models the owner decided per dataset on 2026-10-03
# (scratchpad/precompute_cuts/DECISIONS.md, step 1). Used when --models is not
# given.
DECIDED_MODELS = {
    'adme_mdr1': ('svm', 'dnn'),
    'caco2': ('dnn', 'svm'),
    'herg': ('lightgbm', 'xgboost'),
    'herg_ki': ('svm', 'dnn'),
}


def list_tasks(datasets, models, path, draws=DRAWS):
    """Every task, cheapest first: by training molecules, then condition.
    models None means each dataset's two decided models (DECIDED_MODELS)."""
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
            ms = models or DECIDED_MODELS.get(ds) or DOWNSTREAM_MODELS_NOISE
            for fi in range(5):
                for d in draws:
                    for m in ms:
                        tasks.append((n, f'{ds} {c} {fi} {d} {m}'))
    tasks.sort()
    Path(path).write_text('\n'.join(t for _, t in tasks) + '\n')
    print(f'{len(tasks)} tasks written to {path}')


def status(path, out_dir):
    """One table: per dataset, how many tasks are finished, skipped, in
    progress and not started, and how many setting rows the in-progress ones
    hold. Reads files only."""
    lines = [l.split() for l in Path(path).read_text().splitlines() if l.strip()]
    by = {}
    for t in lines:
        ds, cond, fi, d, m = t
        f = task_path(out_dir, ds, cond, int(fi), int(d), m)
        st = by.setdefault(ds, {'tasks': 0, 'finished': 0, 'skipped': 0,
                                'in progress': 0, 'not started': 0, 'rows': [],
                                'seconds': []})
        st['tasks'] += 1
        if not f.exists():
            st['not started'] += 1
            continue
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        kinds = {r.get('setting') for r in rows}
        if '_done' in kinds:
            st['finished'] += 1
            st['seconds'] += [r['seconds'] for r in rows if r.get('setting') == '_done']
        elif '_skipped' in kinds:
            st['skipped'] += 1
        else:
            st['in progress'] += 1
            st['rows'].append(len(rows))
    print(f"{'dataset':10s} {'tasks':>5s} {'finished':>8s} {'skipped':>7s} "
          f"{'running/stopped':>15s} {'not started':>11s}  rows in unfinished tasks "
          f"(about 500 when done)   minutes per finished task")
    for ds, st in by.items():
        r = sorted(st['rows'])
        rows = (f'{r[0]}-{r[-1]}, median {r[len(r) // 2]}' if r else '-')
        mins = (f"{np.median(st['seconds']) / 60:.0f}" if st['seconds'] else '-')
        print(f"{ds:10s} {st['tasks']:5d} {st['finished']:8d} {st['skipped']:7d} "
              f"{st['in progress']:15d} {st['not started']:11d}  {rows:40s}  {mins}")
    tot = {k: sum(v[k] for v in by.values())
           for k in ('tasks', 'finished', 'skipped', 'in progress', 'not started')}
    print(f"{'ALL':10s} {tot['tasks']:5d} {tot['finished']:8d} {tot['skipped']:7d} "
          f"{tot['in progress']:15d} {tot['not started']:11d}")


# f34's four datasets, so the screen's ranking can be set beside f34's.
F34_DATASETS = ('herg', 'herg_ki', 'caco2', 'adme_mdr1')


def run_tasks(path, jobs, out_dir, part=None, only=None, reverse=False):
    """Run every task in `path` (one per line, as --list-tasks writes them),
    `jobs` at a time, each in its own process with its own log under
    <out>/logs/. A finished task is skipped by run_task itself, so rerunning
    this after a stop picks up where it left off."""
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    lines = [l.split() for l in Path(path).read_text().splitlines() if l.strip()]
    if only:
        lines = [t for t in lines if t[0] in only]
    if reverse:
        lines = lines[::-1]
    if part:
        # --part K/N: this srun takes every N-th task starting at the K-th, so
        # N sruns started with 1/N .. N/N share the list without overlapping.
        k, n = (int(x) for x in part.split('/'))
        lines = lines[k - 1::n]
    logs = Path(out_dir) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)   # the thread limits set at the top are inherited

    import threading
    state = {'done': 0, 'running': set()}
    lock = threading.Lock()
    stop = threading.Event()
    t_start = time.time()

    def heartbeat():
        # A line every 60 s, so an interactive session never sits silent.
        while not stop.wait(60):
            with lock:
                print(f"  [{(time.time() - t_start) / 60:.0f} min] "
                      f"{state['done']} of {len(lines)} finished, running: "
                      f"{'; '.join(sorted(state['running']))}", flush=True)
    threading.Thread(target=heartbeat, daemon=True).start()

    def one(t):
        log = logs / ('__'.join(t) + '.log')
        t0 = time.time()
        with lock:
            state['running'].add(' '.join(t))
        print(f"start {' '.join(t)}  log {log}", flush=True)
        with open(log, 'a') as fh:
            rc = subprocess.call([sys.executable, __file__, '--out', str(out_dir),
                                  '--task', *t], stdout=fh,
                                 stderr=subprocess.STDOUT, env=env)
        with lock:
            state['running'].discard(' '.join(t))
            state['done'] += 1
        print(f"{'ok  ' if rc == 0 else 'FAIL'} {' '.join(t)}  "
              f"{time.time() - t0:.0f}s  log {log}", flush=True)
        if rc != 0:
            tail = log.read_text().strip().splitlines()[-25:]
            print('    ' + '\n    '.join(tail), flush=True)
        return rc
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        rcs = list(ex.map(one, lines))
    stop.set()
    n_fail = sum(rc != 0 for rc in rcs)
    print(f'{len(lines) - n_fail} of {len(lines)} tasks ok, {n_fail} failed')
    return 1 if n_fail else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=list(F34_DATASETS),
                    help="default: f34's four; 'all' for every noise dataset")
    ap.add_argument('--models', nargs='+', default=None,
                    help="default: each dataset's two decided models "
                         "(DECIDED_MODELS)")
    ap.add_argument('--draws', type=int, default=len(DRAWS),
                    help='how many noise draws per condition, from draw 0')
    ap.add_argument('--out', default=str(OUT_DEFAULT))
    ap.add_argument('--list-tasks', metavar='PATH')
    ap.add_argument('--task', nargs=5,
                    metavar=('DATASET', 'CONDITION', 'FOLD', 'DRAW', 'MODEL'))
    ap.add_argument('--status', metavar='PATH',
                    help='print progress of the tasks listed in PATH and exit')
    ap.add_argument('--run-tasks', metavar='PATH',
                    help='run every task listed in PATH')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--only', nargs='+', metavar='DATASET',
                    help='with --run-tasks: only these datasets')
    ap.add_argument('--reverse', action='store_true',
                    help='with --run-tasks: take the list from the end')
    ap.add_argument('--part', metavar='K/N',
                    help='with --run-tasks: run only part K of N, so N sruns '
                         'can share one task list')
    a = ap.parse_args()
    from experiments import dataset_policy  # noqa: F401  (fills the registry)
    if a.list_tasks:
        ds = all_noise_datasets() if a.datasets == ['all'] else a.datasets
        return list_tasks(ds, a.models, a.list_tasks, DRAWS[:a.draws])
    if a.status:
        return status(a.status, a.out)
    if a.run_tasks:
        return run_tasks(a.run_tasks, a.jobs, a.out, a.part, a.only, a.reverse)
    if a.task:
        ds, cond, fi, d, m = a.task
        return run_task(ds, cond, int(fi), int(d), m, a.out)
    ap.error('give --list-tasks, --run-tasks or --task')


if __name__ == '__main__':
    sys.exit(main())
