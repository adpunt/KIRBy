#!/usr/bin/env python3
"""N35: how far apart are the labels of two nearly identical molecules?

Paper: Noise.

THE QUESTION. Two molecules at almost zero distance should have almost the same
true label. What their RECORDED labels still disagree by is label noise, read
without any model. The second research report of 2026-09-24 proposed it as D5:

    E[(y_i - y_j)^2] / 2  ->  noise variance, as the distance -> 0

and, as A7, downweighting both molecules of a near-duplicate pair whose labels
disagree by more than 3 * sqrt(2) * sigma, where sigma is that noise estimate.

WHAT IS COMPUTED, per (level, fold), on the TRAINING molecules only:

  1. Every pairwise distance among the fold's training molecules, in these
     distance spaces:
       base_1, base_2    each downstream model's rank-1 and rank-2 single
                         representation (see THE SINGLE-REPRESENTATION
                         COMPARISON below), one matrix per distinct
                         representation. Tanimoto distance when every value of
                         that representation is 0 or 1 (a fingerprint);
                         otherwise Euclidean on the fold-standardised matrix.
       <hybrid space>    Euclidean on each standardised hybrid matrix the fold
                         builds, one per group of models sharing a recipe.
       forest_proximity  one minus the share of FOREST_TREES trees (100,
                         kirby.utils.noise_estimation) that put the two
                         molecules in one leaf, so distances are steps of 0.01.
                         One forest PER FEATURE MATRIX above, fitted on that
                         standardised matrix and the labels AS SEEN, and its
                         weights used only in that space -- the way n18
                         re-runs its detectors per space (owner's decision
                         2026-09-25). Written under
                         'forest_proximity:<space key>'. ⛔ REFUSED above n07.PROX_SMOOTH_MAX_ROWS
                         training molecules, for n07's reason: it is a square
                         matrix, and a proximity over a subsample is a different
                         quantity. ⛔ The forest reads the labels, so two
                         molecules it puts together already tend to have close
                         labels. Its estimate is biased low by construction and
                         is reported beside the other two, not instead of them.
  2. The close pairs: every pair at or below the 1st percentile of that
     distribution (ties at the cut-off all kept, and counted).
  3. The close pairs split into N_BINS bins of equal pair count by distance. Per
     bin: pairs, distance range, mean |dy|, and mean(dy^2)/2. mean(dy^2)/2 is
     Var(dy)/2 with each pair counted in both orders, so the arbitrary order of
     i and j cannot give dy a mean.
  4. The noise estimate: mean(dy^2)/2 in the CLOSEST bin. Also the intercept of
     a straight line through the per-bin values against mean bin distance, i.e.
     the same quantity carried to distance 0. The closest bin still holds some
     real activity difference, so both are upper bounds on the noise variance.

ON A DATASET WITH REPEAT MEASUREMENTS (herg_ic50, bindingdb_kd, kiba; any
dataset build_replicate_groups returns groups for) one row is one measurement,
and two rows of the same molecule are at distance 0 in every space. Those pairs
ARE repeat measurements. Every close-pair number is therefore also written split
into same-molecule pairs and different-molecule pairs, and beside it the pooled
within-molecule variance of the fold's training rows, sum((k-1) * var) / sum(k-1),
which is how n19.measurement_stats and n26 read the noise directly.

THE CHECK. `gaussian` at the dataset's level grid, injected into the training
labels only. Each base_1 / base_2 pair set depends on the features alone, so across levels
it is the same pairs, and the increase of the base estimate over level 0 should
be the injected variance itself -- each member of a pair carries its own draw,
and Var(e_i - e_j) / 2 = Var(e). The realised injected variance,
mean((y_seen - y_clean)^2) over the training molecules, is written beside it.
The hybrid is rebuilt from the noisy labels at every level, so its features and
therefore its pair set can move; each forest proximity is refitted on the noisy
labels, and its pairs move with them. Neither is expected to track exactly; the
base_1 and base_2 matrices are the check, one row per matrix.

THE METHOD ARM (A7). Per (level, fold, feature space, model), all scored on the
same held-out molecules against their CLEAN labels:

  untouched                         the training labels as seen, weight 1
  downweight_pairs_this_space       both molecules of every close pair in THIS
                                    feature space's own distance whose |dy| is
                                    over 3 * sqrt(2) * sigma get weight 0.25.
                                    sigma^2 = the closest-bin estimate, same
                                    space, same labels.
  downweight_pairs_forest_proximity the same rule with the pairs and sigma from
                                    the forest proximity fitted on THIS
                                    feature space's matrix. Absent where the
                                    proximity was refused.

The weight 0.25 is FINDINGS.md section 1's "downweight to a quarter", the only
operation there that was free to apply at level 0.

HOW MANY FLAGGED PAIRS ARE ACTIVITY CLIFFS. n25's cliff definition, which is
n01.identify_cliff_pairs: ECFP4 Tanimoto at or above 0.55 between the two
molecules, and |dy| over 2 median absolute deviations of the SEEN training
labels. On a molecular dataset with SMILES, the Tanimoto is computed for the
flagged pairs only, from the same 2048-bit radius-2 fingerprints n01 builds. On
any other dataset n25.cliff_mask_for_dataset is called on the training subset
and its pair matrix is used; where that returns no pair matrix (the antibody
path) the count is of flagged pairs whose two molecules are both cliff
molecules, and the result says so.

WHERE THE FEATURES COME FROM. As in n30: the recipe for each downstream model is
read from the precomputed pool, searched on clean labels; each fold rebuilds the
hybrid from that fold's SEEN training labels once per level, and every arm
shares it.

THE SINGLE-REPRESENTATION COMPARISON is each downstream model's top 2 base
representations by that model's own score in the stored search -- spaces
`base_1` and `base_2` -- the same rule as the rest of the study
(_shared.noise_base_reps_by_model, owner's decision 2026-09-25). Which
representations those are is written to `base_reps_by_model`; a model the pool
holds no stored search for has none, and `base_reps_missing` says why.

Usage:
    python n35_near_duplicate_disagreement.py --dataset herg_ki --mode full
    python n35_near_duplicate_disagreement.py --dataset herg_ic50 --mode full --levels 0,0.5
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import functools
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS,
    get_best_search_for_downstream,
    noise_base_reps_by_model, base_rank_groups, NOISE_BASE_REP_COUNT,
    pool_recipes_by_family, describe_pool_family, POOL_SUBSETS,
)
from experiments.dataset_policy import (
    noise_levels, noise_groups_for, noise_dose_units,
)
from experiments.noise.noise_helpers import (
    noise_spec_for, inject_train_labels, hybrid_for_fold,
    pool_recipe_per_downstream_model, group_models_by_recipe, tail_scores,
)
from experiments.noise.n07_remediation import (
    _predict_for_metric, PROX_SMOOTH_MAX_ROWS,
)
from experiments.noise._ground_truth_providers import build_replicate_groups
# The D-versus-D' rows, per-pass checkpoint, cached builds, controls and the
# similarity axis shared by n32-n41 (n32's shared block, 2026-09-25).
from experiments.noise.n32_iterative_and_dual_selection import (
    method_setting, downstream_model_list, fit_predict_cached, cached_hybrid,
    ledger_signature, PassLedger, restart_flag, arm_rows, arm_result,
    control_twins, twin_scores, controls_enabled, replicate_seed,
    sweep_settings, is_model_input, fold_similarity, setting_models, _nc,
    _ss,
)

EXP_ID = 'n35_near_duplicate_disagreement'

# The only injected condition. The check needs noise whose variance is the
# same on every molecule, so that the increase of the estimate is one number.
CHECK_CONDITION = 'gaussian'
NOISE_DRAWS = (0,)
# Close pairs: at or below this percentile of the pairwise distances.
CLOSE_PERCENTILE = 1.0
# Equal-count bins over the close pairs, by distance.
N_BINS = 5
# Flag rule, from the report: |dy| over this many sigma.
FLAG_MULTIPLIER = 3.0 * np.sqrt(2.0)
DOWNWEIGHT = 0.25
# Above this many training rows the pairwise distances are not computed. A
# float32 upper triangle at 12,000 rows is 288 MB, on top of the square matrix.
MAX_PAIR_ROWS = 12000
# n25's cliff definition (n01's numbers).
CLIFF_SIM_THRESHOLD = 0.55
CLIFF_ACTIVITY_K = 2.0

POOL_SUBSET = 'predictive_and_similarity'
SPACE_FOR_FAMILY = {
    'predictive': 'hybrid',
    'similarity': 'hybrid_similarity',
}
# base_1 and base_2 are each downstream model's rank-1 and rank-2 single
# representation (see the module docstring).
BASE_SPACES = tuple(f'base_{_k}' for _k in range(1, NOISE_BASE_REP_COUNT + 1))
PROXIMITY_SPACE = 'forest_proximity'
ARMS = ('untouched', 'downweight_pairs_this_space',
        'downweight_pairs_forest_proximity')


def search_fold_reps(bp, fold_idx, n_folds, pool):
    """The representations the stored search built this fold's hybrid on.

    DIRTY-LOCAL 2026-09-28, the same rule n02 took in 736ec93. A recipe with
    rep_count N below the pool size is built, in the search, on the N
    representations that fold's own training rows rank highest.
    build_fold_hybrid, handed the whole pool and no per-fold scores, ranks by
    Blueprint.rep_scores instead (_shared.py:5277-5307): each
    representation's best score over EVERY molecule, held-out ones included,
    on clean labels (find_best_hybrid.py:9492-9522). On herg_ki that built
    every fold's predictive hybrid on mhg_gnn, ecfp4, atom_pair, graph_kernel
    and smited, while the search used mordred in every fold and ecfp4 in none.

    Blueprint.allocation_per_fold records the representations each search
    fold gave features to. The search folds are this file's folds: both are
    GroupKFold on the same groups, and the stored single-representation
    baselines equal this file's level-0 scores (herg_ki rf on ecfp4, 0.6032
    RMSE in both). Returns (names, source); names is None when the recipe does
    not cut the pool or carries no per-fold record, and build_fold_hybrid
    then does what it did before.
    """
    rc = getattr(bp, 'rep_count', None)
    if rc is None or int(rc) >= len(pool):
        return None, 'whole_pool'
    apf = getattr(bp, 'allocation_per_fold', None)
    if not apf:
        return None, 'blueprint.rep_scores (no allocation_per_fold on the recipe)'
    bp_folds = getattr(bp, 'n_folds', None)
    if bp_folds is not None and int(bp_folds) != int(n_folds):
        return None, (f'blueprint.rep_scores (recipe searched on {bp_folds} '
                      f'folds, this run has {n_folds})')
    per = apf.get(fold_idx, apf.get(str(fold_idx)))
    if not per:
        return None, f'blueprint.rep_scores (no allocation for fold {fold_idx})'
    names = [r for r, n in per.items() if n and r in pool]
    if not names:
        return None, f'blueprint.rep_scores (fold {fold_idx} allocation empty)'
    return names, 'allocation_per_fold'


# ---------------------------------------------------------------------------
# Distances and close pairs
# ---------------------------------------------------------------------------

def is_binary(X):
    """Every value 0 or 1: a fingerprint, so Tanimoto is defined."""
    X = np.asarray(X)
    return X.size > 0 and bool(np.all((X == 0) | (X == 1)))


def tanimoto_distance_matrix(X):
    """One minus Tanimoto, for 0/1 rows. Two all-zero rows are at distance 0."""
    A = np.asarray(X, dtype=np.float32)
    inter = A @ A.T
    cnt = A.sum(axis=1)
    union = cnt[:, None] + cnt[None, :] - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        sim = np.where(union > 0, inter / union, 1.0)
    return (1.0 - sim).astype(np.float32)


def euclidean_distance_matrix(X):
    from sklearn.metrics.pairwise import euclidean_distances
    return euclidean_distances(np.asarray(X, dtype=np.float64)).astype(np.float32)


def close_pairs(D, percentile=CLOSE_PERCENTILE):
    """(i, j, d) for every i < j at or below `percentile` of the i < j distances.

    Returns (i, j, d, threshold, n_pairs_total).
    """
    D = np.asarray(D)
    n = D.shape[0]
    total = n * (n - 1) // 2
    if total == 0:
        return (np.zeros(0, int), np.zeros(0, int), np.zeros(0),
                float('nan'), 0)
    upper = np.empty(total, dtype=np.float32)
    pos = 0
    for i in range(n - 1):
        row = D[i, i + 1:]
        upper[pos:pos + row.size] = row
        pos += row.size
    k = max(0, int(np.ceil(percentile / 100.0 * total)) - 1)
    thr = float(np.partition(upper, k)[k])
    del upper
    ii, jj, dd = [], [], []
    for i in range(n - 1):
        row = D[i, i + 1:]
        hit = np.nonzero(row <= thr)[0]
        if hit.size:
            ii.append(np.full(hit.size, i))
            jj.append(hit + i + 1)
            dd.append(row[hit].astype(float))
    if not ii:
        return (np.zeros(0, int), np.zeros(0, int), np.zeros(0), thr, total)
    return (np.concatenate(ii), np.concatenate(jj), np.concatenate(dd),
            thr, total)


def pair_bins(d, dy, n_bins=N_BINS):
    """Equal-count bins by distance. Per bin: count, range, mean |dy|, E[dy^2]/2."""
    order = np.argsort(d, kind='stable')
    out = []
    for chunk in np.array_split(order, min(n_bins, max(1, d.size))):
        if chunk.size == 0:
            continue
        out.append({
            'n_pairs': int(chunk.size),
            'distance_min': float(d[chunk].min()),
            'distance_max': float(d[chunk].max()),
            'distance_mean': float(d[chunk].mean()),
            'mean_abs_label_difference': float(np.mean(np.abs(dy[chunk]))),
            'half_mean_squared_label_difference': float(
                np.mean(dy[chunk] ** 2) / 2.0),
        })
    return out


def noise_estimate(d, dy, n_bins=N_BINS):
    """Closest-bin E[dy^2]/2, and the straight-line intercept at distance 0."""
    if d.size == 0:
        return {'n_close_pairs': 0, 'bins': [],
                'closest_bin_estimate': float('nan'),
                'intercept_estimate': float('nan')}
    bins = pair_bins(d, dy, n_bins)
    est = bins[0]['half_mean_squared_label_difference']
    xs = np.array([b['distance_mean'] for b in bins])
    ys = np.array([b['half_mean_squared_label_difference'] for b in bins])
    if len(bins) >= 2 and np.ptp(xs) > 0:
        slope, icpt = np.polyfit(xs, ys, 1)
        icpt = float(icpt)
    else:
        icpt = float('nan')
    return {'n_close_pairs': int(d.size), 'bins': bins,
            'closest_bin_estimate': float(est),
            'intercept_estimate': icpt}


def pooled_replicate_variance(y, rep_groups):
    """sum((k-1) * var_ddof1) / sum(k-1) over molecules measured 2+ times."""
    y = np.asarray(y, dtype=float)
    by = {}
    for i, g in enumerate(np.asarray(rep_groups)):
        by.setdefault(g, []).append(i)
    num = den = 0.0
    n_mol = 0
    for rows in by.values():
        if len(rows) >= 2:
            v = y[rows]
            num += float(np.sum((v - v.mean()) ** 2))
            den += len(rows) - 1
            n_mol += 1
    return {'pooled_within_molecule_variance': (num / den) if den else float('nan'),
            'molecules_measured_twice_or_more': int(n_mol),
            'degrees_of_freedom': int(den)}


def pair_record(D, y_seen, same_mol=None):
    """Everything this file reports about one distance space on one fold."""
    i, j, d, thr, total = close_pairs(D)
    dy = y_seen[i] - y_seen[j]
    rec = {'pairs_total': int(total), 'distance_threshold': thr,
           'close_percentile': CLOSE_PERCENTILE}
    rec.update(noise_estimate(d, dy))
    if same_mol is not None and d.size:
        s = same_mol[i] == same_mol[j]
        rec['same_molecule_pairs'] = noise_estimate(d[s], dy[s])
        rec['different_molecule_pairs'] = noise_estimate(d[~s], dy[~s])
    return rec, (i, j, dy)


def flag_pairs(i, j, dy, sigma2):
    """Close pairs with |dy| over FLAG_MULTIPLIER * sigma. Returns pair mask."""
    if not np.isfinite(sigma2) or sigma2 < 0 or dy.size == 0:
        return np.zeros(dy.size, dtype=bool)
    return np.abs(dy) > FLAG_MULTIPLIER * np.sqrt(sigma2)


def pair_weights(n, i, j, flagged):
    w = np.ones(n, dtype=float)
    w[i[flagged]] = DOWNWEIGHT
    w[j[flagged]] = DOWNWEIGHT
    return w


# ---------------------------------------------------------------------------
# Activity cliffs among the flagged pairs
# ---------------------------------------------------------------------------

def ecfp4_fingerprints(smiles):
    """n01's fingerprints: Morgan radius 2, 2048 bits. None where RDKit fails."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem
        RDLogger.DisableLog('rdApp.*')
    except Exception:
        return None
    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if s is not None else None
        fps.append(None if m is None
                   else AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
    return fps


def count_cliff_pairs(fi, fj, dy_flagged, y_seen, fps_train=None,
                      pair_matrix=None, cliff_molecules=None):
    """How many flagged pairs meet n25's cliff definition.

    fi, fj  the flagged pairs' training-row indices
    """
    if fi.size == 0:
        return {'flagged_pairs': 0, 'flagged_cliff_pairs': 0,
                'how': 'no flagged pair'}
    mad = float(np.median(np.abs(y_seen - np.median(y_seen))))
    act = np.abs(dy_flagged) > CLIFF_ACTIVITY_K * mad
    if fps_train is not None:
        from rdkit import DataStructs
        sim = np.array([
            (DataStructs.TanimotoSimilarity(fps_train[a], fps_train[b])
             if fps_train[a] is not None and fps_train[b] is not None
             else 0.0) for a, b in zip(fi, fj)])
        n = int(np.sum((sim >= CLIFF_SIM_THRESHOLD) & act))
        how = ('ECFP4 Tanimoto >= 0.55 on the pair, and |dy| over 2 median '
               'absolute deviations of the seen training labels (n01)')
    elif pair_matrix is not None:
        n = int(np.sum(pair_matrix[fi, fj]))
        how = 'n25.cliff_mask_for_dataset on the training subset, pair matrix'
    elif cliff_molecules is not None:
        n = int(np.sum(cliff_molecules[fi] & cliff_molecules[fj]))
        how = ('APPROXIMATE: both molecules are cliff molecules under n25; no '
               'pair matrix exists on this dataset')
    else:
        return {'flagged_pairs': int(fi.size), 'flagged_cliff_pairs': None,
                'how': 'no cliff reference available'}
    return {'flagged_pairs': int(fi.size), 'flagged_cliff_pairs': n,
            'mad_of_seen_training_labels': mad, 'how': how}


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _fit_score(factory, X_tr, y_tr, X_va, y_va, metric_fn, metric_name,
               weights=None, dataset_name=None, model_name=None):
    """One fit, one held-out score and its tail scores, or (None, None, why).

    dataset_name / model_name, added 2026-09-25, route a seeded model through
    noise_cache (n32.fit_predict_cached); the prediction is the same."""
    try:
        pred = fit_predict_cached(dataset_name, model_name, factory, X_tr,
                                  y_tr, X_va, metric_name,
                                  sample_weight=weights)
        return float(metric_fn(y_va, pred)), tail_scores(y_va, pred), None
    except Exception as exc:
        return None, None, f'{type(exc).__name__}: {exc}'


def _levels_and_draws(dataset_name, levels=None, draws=None):
    levels = (noise_levels(dataset_name, strategy=CHECK_CONDITION,
                           is_clf=False) if levels is None else levels)
    draws = tuple(NOISE_DRAWS if draws is None else draws)
    return [(float(l), int(d)) for l in levels for d in draws]


def run(data, config, output_dir, dataset_name, force_rerun=False, **kwargs):
    """N35: label disagreement of near-duplicate pairs, and downweighting them.

    force_rerun (run_dataset.py --force-rerun, via run_all) sets aside the
    rows file and the per-pass checkpoint and recomputes every pass.

    THE SIMILARITY AXIS, added 2026-09-25 (plan section 1). The close pairs of
    today's two arms are found in the model's own matrix (this space's
    distance) and in a forest fitted on it: that is the model_input setting,
    unchanged. Every other similarity setting the run sweeps
    (KIRBY_SIMILARITY_SETTINGS, n32.sweep_settings) finds the close pairs in
    ITS distance instead -- the full training-by-training matrix, the same 1st
    percentile, the same flag rule -- and adds one arm,
    downweight_pairs@<setting>, fitted on the cell's own matrix. A per-model
    space (base_*, *_aug_*, hybrid*) is built for each model of the cell and
    that model is fitted on its own pairs. The rule reads no k, so k is not
    swept here."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N35: NEAR-DUPLICATE LABEL DISAGREEMENT")
    print("=" * 70)

    labels = np.asarray(data['labels'], dtype=float)
    reps = data['reps']
    groups = data.get('groups')
    task_type = data['task_type']
    if task_type != 'regression':
        print(f"  SKIPPED: {dataset_name} is {task_type}. A squared label "
              f"difference is not defined between two classes.")
        return {'skipped': 'not_regression', 'task_type': task_type}
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    run_proximity = not kwargs.get('skip_proximity', False)
    count_cliffs = not kwargs.get('skip_cliffs', False)

    downstream_models = downstream_model_list(kwargs, config)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n35: no downstream model could be built. Asked for "
            f"{list(kwargs.get('models') or DOWNSTREAM_MODELS)}, got "
            f"{sorted(model_factories)}.")

    # {model: [rank-1 rep, rank-2 rep]} given directly (a smoke or a direct
    # caller). None means: rank them from the stored search, below.
    base_reps_override = kwargs.get('base_reps')
    noise_groups = noise_groups_for(data)[0]
    rep_groups = build_replicate_groups(data)

    fps_all = None
    cliff_note = None
    if count_cliffs and data.get('smiles') is not None \
            and data.get('domain', 'molecular') == 'molecular':
        fps_all = ecfp4_fingerprints(data['smiles'])
        if fps_all is None:
            cliff_note = 'RDKit unavailable; cliffs counted through n25 instead'

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_condition': CHECK_CONDITION,
        'dose_units': noise_dose_units(dataset_name),
        'models': downstream_models,
        'arms': list(ARMS),
        'arm_to_beat': 'untouched',
        'close_percentile': CLOSE_PERCENTILE,
        'n_bins': N_BINS,
        'flag_rule': (f'|dy| > {FLAG_MULTIPLIER:.4f} * sigma, sigma^2 the '
                      f'closest-bin estimate in the same distance space'),
        'downweight': DOWNWEIGHT,
        'proximity_row_cap': PROX_SMOOTH_MAX_ROWS,
        'has_repeat_measurements': rep_groups is not None,
        'estimate_note': (
            'closest_bin_estimate is mean(dy^2)/2 over the closest fifth of the '
            'close pairs; intercept_estimate carries the per-bin values to '
            'distance 0 with a straight line. Both hold some real activity '
            'difference and are upper bounds on the noise variance.'),
        'proximity_note': (
            'One forest per feature matrix, fitted on that standardised matrix '
            'and the labels as seen; its weights are used only in that space. '
            'Its close pairs already tend to agree on the label, so its '
            'estimate is biased low by construction.'),
        'check_note': (
            'Under gaussian, each base_1/base_2 pair set does not depend on the '
            'labels, so its closest-bin estimate minus its level-0 value should '
            'equal the realised injected variance on the same fold. The hybrid '
            'is rebuilt from the noisy labels and each forest is refitted on '
            'them, so their pair sets move with the level.'),
        'cliff_note': cliff_note,
        'similarity_settings': [st.name for st in sweep_settings()],
        'similarity_note': (
            'downweight_pairs_this_space and downweight_pairs_forest_proximity '
            'are the model_input setting, unchanged. downweight_pairs@<setting> '
            'finds the close pairs in that setting\'s distance; its pair record '
            'is written under <space key>@<setting>[|<model>].'),
        'controls_run': controls_enabled(),
        # In the header, so the ledger signature changes and a checkpoint
        # written under the whole-pool rebuild is set aside, not read back.
        'hybrid_fold_rep_rule': 'allocation_per_fold (search_fold_reps)',
        'rows_note': ('D is untouched; score_clean is the same model fitted on '
                      'the clean training labels, fitted for the rows only and '
                      'not an arm of the summary.'),
        'per_level': {},
        'pairs_per_pass': {},
    }

    splits = list(cv_split(groups, n_folds, next(iter(reps.values())), labels,
                           mode=mode))

    by_model, no_recipe = pool_recipe_per_downstream_model(
        functools.partial(
            get_best_search_for_downstream, reps, labels, groups,
            budget=None, augmentations=data.get('augmentations'),
            dataset_name=dataset_name, task_type=task_type, data=data,
            pool_subset=POOL_SUBSET),
        downstream_models)
    _raised = [d['lookup_error'] for d in no_recipe.values()
               if d['lookup_error']]
    results['hybrid_lookup_error'] = _raised[0] if _raised else None
    results['hybrid_unavailable_by_model'] = {
        m: d['why'] for m, d in no_recipe.items()}

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
    results['hybrid_spaces_unavailable'] = hybrid_unavailable

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
    # Tanimoto for a representation whose every value is 0 or 1.
    base_binary = {r: is_binary(reps[r])
                   for names in base_by_model.values() for r in names}
    results['base_distance_by_rep'] = {
        r: ('tanimoto' if b else 'euclidean_on_standardised_features')
        for r, b in base_binary.items()}

    _ordered = []
    for _m in downstream_models:
        for r in recipes_by_model.get(_m, ()):
            if r['space'] not in _ordered:
                _ordered.append(r['space'])
    spaces = base_spaces_run + tuple(_ordered)
    results['hybrid_available'] = bool(_ordered)
    if _ordered:
        results['hybrid_recipe_by_model'] = convert_numpy({
            m: {r['space']: {
                'built_under_objective': r['objective'],
                'features_chosen_by': describe_pool_family(r['objective']),
                'budget': r['budget'],
                'allocation_method': r['allocation_method'],
                'importance_method': r['importance_method'],
                'reps': sorted(getattr(r['blueprint'], 'reps', []) or []),
                'search_score_mean': r['search_score_mean'],
                'search_metric_name': r['search_metric_name'],
            } for r in rs}
            for m, rs in recipes_by_model.items() if rs})
    else:
        results['hybrid_unavailable_reason'] = (
            (f"the pool lookup RAISED {results['hybrid_lookup_error']}; the "
             f"pool was never read. Every arm ran in the single "
             f"representations only.")
            if results['hybrid_lookup_error'] else
            (f'no recipe came back from pool subset {POOL_SUBSET!r} for any '
             f'downstream model. Every arm ran in the single '
             f'representations only.'))
        print("  no hybrid recipe; single representations only")
    results['spaces_run'] = list(spaces)

    passes = _levels_and_draws(dataset_name, levels=kwargs.get('levels'),
                               draws=kwargs.get('draws'))
    results['noise_levels'] = sorted({l for l, _ in passes})
    print(f"  {len(labels)} rows, {len(splits)} folds, models "
          f"{downstream_models}")
    print(f"  spaces {list(spaces)}; single-representation distances "
          f"{results['base_distance_by_rep']}; levels "
          f"{results['noise_levels']}")

    acc, tails, errors, pair_acc = {}, {}, {}, {}
    t0 = time.time()
    metric = (metric_name, metric_fn, lower_is_better)
    sim_settings = [st for st in sweep_settings() if not is_model_input(st)]
    # Rows and the per-pass checkpoint, 2026-09-25 (n32's PassLedger). The
    # file stem is the check condition, main()'s result suffix.
    ledger = PassLedger(output_dir, EXP_ID, dataset_name, CHECK_CONDITION,
                        signature=ledger_signature(data, config, results,
                                                   splits),
                        restart=bool(force_rerun) or restart_flag(kwargs))
    if ledger.restored():
        st_ = ledger.restored()
        acc, tails, errors, pair_acc = (st_['acc'], st_['tails'],
                                        st_['errors'], st_['pair_acc'])
        print(f"  resumed {len(ledger.passes_done)} finished passes")

    for level, draw in passes:
        lvl_key = f'{level:g}'
        spec = noise_spec_for(dataset_name, CHECK_CONDITION, level,
                              noise_seed=draw, is_clf=False)
        cell = acc.setdefault(lvl_key, {})
        tcell = tails.setdefault(lvl_key, {})
        print(f"\n  --- gaussian / level {lvl_key} / draw {draw} ---")

        for fold_i, (train_idx, val_idx) in enumerate(splits):
            pass_key = f'l{lvl_key}d{draw}f{fold_i}'
            if ledger.pass_done(pass_key):
                continue
            pass_rows = []
            y_val = labels[val_idx]
            y_clean_tr = labels[train_idx]
            g_tr = groups[train_idx] if groups is not None else None
            n_tr = len(train_idx)
            try:
                y0, _mask, _prov = inject_train_labels(
                    labels, spec, train_idx=train_idx, is_clf=False,
                    groups=noise_groups, dataset_name=dataset_name)
            except Exception as exc:
                errors[f'{lvl_key}|d{draw}|f{fold_i}|inject'] = (
                    f'{type(exc).__name__}: {exc}')
                continue
            y0 = np.asarray(y0, dtype=float)
            same_mol = (np.asarray(rep_groups)[train_idx]
                        if rep_groups is not None else None)
            prec = {
                'pass': pass_key, 'training_rows': int(n_tr),
                'realised_injected_variance': float(
                    np.mean((y0 - y_clean_tr) ** 2)),
                'per_space': {},
            }
            if rep_groups is not None:
                prec['replicate_variance_seen'] = pooled_replicate_variance(
                    y0, same_mol)
                prec['replicate_variance_clean'] = pooled_replicate_variance(
                    y_clean_tr, same_mol)
            if n_tr > MAX_PAIR_ROWS:
                prec['refused'] = (
                    f'{n_tr} training rows is over MAX_PAIR_ROWS='
                    f'{MAX_PAIR_ROWS}; the pairwise distances were not '
                    f'computed and no arm ran')
                pair_acc.setdefault(lvl_key, []).append(prec)
                # Recorded as finished, so a resumed run does not append this
                # refusal to pair_acc a second time.
                ledger.finish_pass(pass_key, {'acc': acc, 'tails': tails,
                                              'errors': errors,
                                              'pair_acc': pair_acc})
                print(f"    fold {fold_i}: {prec['refused']}")
                continue

            fps_train = ([fps_all[i] for i in train_idx]
                         if fps_all is not None else None)
            cliff_pairs_mat = cliff_mols = None
            if count_cliffs and fps_train is None:
                try:
                    from experiments.noise.n25_cliff_exemption import (
                        cliff_mask_for_dataset)
                    from experiments.noise.n01_activity_cliff_framework import (
                        identify_cliff_pairs)
                    cliff_mols, cinfo = cliff_mask_for_dataset(
                        data, y0, False, subset=train_idx)
                    sim = cinfo.pop('similarity_matrix', None)
                    if sim is not None:
                        cliff_pairs_mat, _ = identify_cliff_pairs(
                            sim, y0, CLIFF_SIM_THRESHOLD, CLIFF_ACTIVITY_K,
                            similar_percentile=cinfo.get('similar_percentile'))
                except Exception as exc:
                    errors[f'{lvl_key}|f{fold_i}|cliffs'] = (
                        f'{type(exc).__name__}: {exc}')

            def _describe(space_key, D):
                rec, (pi, pj, pdy) = pair_record(D, y0, same_mol)
                flagged = flag_pairs(pi, pj, pdy, rec['closest_bin_estimate'])
                rec['flagged_pairs'] = int(flagged.sum())
                w = pair_weights(n_tr, pi, pj, flagged)
                rec['flagged_molecules'] = int(np.sum(w < 1.0))
                if count_cliffs:
                    rec['cliffs'] = count_cliff_pairs(
                        pi[flagged], pj[flagged], pdy[flagged], y0,
                        fps_train=fps_train, pair_matrix=cliff_pairs_mat,
                        cliff_molecules=cliff_mols)
                prec['per_space'][space_key] = rec
                return w

            train_reps = {k: np.asarray(v[train_idx], dtype=np.float64)
                          for k, v in reps.items()}
            val_reps = {k: np.asarray(v[val_idx], dtype=np.float64)
                        for k, v in reps.items()}

            def _prox(cell_key, X_tr_cell):
                """Forest-proximity weights from a forest fitted on THIS
                cell's standardised matrix. None when switched off, refused
                or failed."""
                if not run_proximity:
                    return None
                pkey = f'{PROXIMITY_SPACE}:{cell_key}'
                if PROX_SMOOTH_MAX_ROWS is not None and n_tr > PROX_SMOOTH_MAX_ROWS:
                    prec['per_space'][pkey] = {
                        'refused': (f'{n_tr} training rows is over '
                                    f'PROX_SMOOTH_MAX_ROWS='
                                    f'{PROX_SMOOTH_MAX_ROWS}; refused rather '
                                    f'than subsampled, as n07 does')}
                    return None
                try:
                    from kirby.utils.noise_estimation import (
                        forest_proximity_distance)
                    # The forest's leaf ids through noise_cache (2026-09-25):
                    # the same forest forest_leaf_ids fits, field for field
                    # (similarity_space._forest_leaves), fitted once per
                    # (matrix, labels).
                    Dp = forest_proximity_distance(_nc.cached_forest_leaf_ids(
                        dataset_name, X_tr_cell, y0, False)[0])
                    w = _describe(pkey, Dp)
                    del Dp
                    return w
                except Exception as exc:
                    errors[f'{lvl_key}|{pkey}|f{fold_i}'] = (
                        f'{type(exc).__name__}: {exc}')
                    return None

            # {space: [(X_tr, X_va, models, weights from this space's pairs,
            #           weights from this space's forest proximity)]}
            # Each single-representation space holds one matrix per distinct
            # representation at that rank, shared by the models that rank it
            # there.
            fold_spaces = {}
            for _k, _bsp in enumerate(BASE_SPACES, start=1):
                if _bsp not in spaces:
                    continue
                _bcells = []
                for _rn, _ms in base_rank_groups(base_by_model, _k,
                                                 downstream_models).items():
                    # Standardised once per fold through noise_cache
                    # (2026-09-25); the same float64 values.
                    Xb_tr, Xb_va = _nc.cached_scaled(
                        dataset_name, reps[_rn], train_idx, val_idx,
                        dtype=np.float64)
                    _key = (_bsp if list(_ms) == list(downstream_models)
                            else f'{_bsp}/{_ms[0]}')
                    Db = (tanimoto_distance_matrix(train_reps[_rn])
                          if base_binary[_rn]
                          else euclidean_distance_matrix(Xb_tr))
                    w_b = _describe(_key, Db)
                    del Db
                    _bcells.append((Xb_tr, Xb_va, list(_ms), w_b,
                                    _prox(_key, Xb_tr)))
                fold_spaces[_bsp] = _bcells

            for _sp in spaces:
                if _sp in BASE_SPACES:
                    continue
                _cells = []
                for _bp_m, _mnames, _shareable in group_models_by_recipe(
                        {_m: recipe_for_space_by_model.get(_m, {}).get(_sp)
                         for _m in downstream_models}, downstream_models):
                    # DIRTY-LOCAL 2026-09-28: the representations the search
                    # built THIS fold's hybrid on (search_fold_reps). Before
                    # this, build_fold_hybrid ranked the whole pool by
                    # Blueprint.rep_scores, a whole-dataset clean-label score
                    # that saw this fold's held-out molecules.
                    _keep, _keep_src = search_fold_reps(
                        _bp_m, fold_i, n_folds, list(train_reps))
                    results.setdefault('hybrid_fold_reps', {}).setdefault(
                        _sp, {}).setdefault(_mnames[0], {})[str(fold_i)] = {
                            'reps': sorted(_keep) if _keep else None,
                            'source': _keep_src}
                    _tr_h = (train_reps if _keep is None else
                             {k: v for k, v in train_reps.items() if k in _keep})
                    _va_h = (val_reps if _keep is None else
                             {k: v for k, v in val_reps.items() if k in _keep})
                    # Cached per (fold, recipe, labels), 2026-09-25. The cache
                    # key holds each used matrix's content hash, so a cut pool
                    # never reads back a whole-pool build.
                    X_tr_h, X_va_h = cached_hybrid(
                        dataset_name, _tr_h, _va_h, y0, _bp_m, g_tr,
                        _mnames[0], model_factories, metric, train_idx,
                        _shareable)
                    if X_tr_h is None:
                        errors[f'{lvl_key}|{_sp}/{_mnames[0]}|f{fold_i}|'
                               f'build'] = 'the hybrid could not be built'
                        continue
                    sh = StandardScaler()
                    Xh_tr = sh.fit_transform(X_tr_h)
                    Xh_va = sh.transform(X_va_h)
                    _key = (_sp if list(_mnames) == list(downstream_models)
                            else f'{_sp}/{_mnames[0]}')
                    w_h = _describe(_key, euclidean_distance_matrix(Xh_tr))
                    _cells.append((Xh_tr, Xh_va, list(_mnames), w_h,
                                   _prox(_key, Xh_tr)))
                if _cells:
                    fold_spaces[_sp] = _cells

            # The similarity spaces of this fold, from the labels as seen
            # (plan section 1, 2026-09-25).
            fsim = fold_similarity(
                dataset_name, data, reps=reps, train_idx=train_idx,
                val_idx=val_idx, y_train=y0, groups_train=g_tr,
                base_by_model=base_by_model,
                recipe_for_space_by_model=recipe_for_space_by_model,
                model_factories=model_factories, metric=metric,
                task_type=task_type)

            for space in spaces:
                for (X_tr_s, X_va_s, cell_models, w_self,
                     w_prox) in fold_spaces.get(space, ()):
                    _ekey = (space if cell_models == list(downstream_models)
                             else f'{space}/{cell_models[0]}')
                    space_cell = cell.setdefault(space, {})
                    tspace = tcell.setdefault(space, {})
                    # {arm: {model: weights}}: today's arms hold one weight
                    # vector for every model of the cell.
                    arm_w = {'untouched': {m: None for m in cell_models},
                             'downweight_pairs_this_space':
                                 {m: w_self for m in cell_models}}
                    if w_prox is not None:
                        arm_w['downweight_pairs_forest_proximity'] = {
                            m: w_prox for m in cell_models}
                    # Every other similarity setting: its own close pairs.
                    for setting in sim_settings:
                        if setting.space == _ss.FOREST_SPACE and run_proximity:
                            # downweight_pairs_forest_proximity above IS this
                            # setting: the same forest on the same matrix and
                            # labels. Not computed twice.
                            continue
                        for model in setting_models(setting, cell_models):
                            skey = (f'{_ekey}@{setting.name}'
                                    + (f'|{model}' if len(cell_models) > 1
                                       and setting.space in _ss.FEATURE_SPACES
                                       else ''))
                            try:
                                D, why = fsim.full_distance(
                                    setting, model, X_tr_s, dtype=np.float32)
                                if D is None:
                                    prec['per_space'][skey] = {
                                        'unavailable': why}
                                    continue
                                w_s = _describe(skey, D)
                                del D
                            except Exception as exc:
                                errors[f'{lvl_key}|{skey}|f{fold_i}'] = (
                                    f'{type(exc).__name__}: {exc}')
                                continue
                            fit_for = (cell_models if setting.space
                                       not in _ss.FEATURE_SPACES else [model])
                            for m in fit_for:
                                arm_w.setdefault(
                                    f'downweight_pairs@{setting.name}',
                                    {})[m] = w_s
                    pass_scores = {}
                    for arm, w_by_m in arm_w.items():
                        for mname, w in w_by_m.items():
                            if arm != 'untouched' and w is not None \
                                    and np.all(w == 1.0):
                                # Nothing flagged: the arm IS the untouched
                                # fit. Scored anyway so the pairing has every
                                # pass, and the count of flagged molecules
                                # says why it tied.
                                w = None
                            s, tl, err = _fit_score(
                                model_factories[mname], X_tr_s, y0, X_va_s,
                                y_val, metric_fn, metric_name, weights=w,
                                dataset_name=dataset_name, model_name=mname)
                            if s is None:
                                errors[f'{lvl_key}|{_ekey}|{arm}|{mname}|'
                                       f'f{fold_i}'] = err
                                continue
                            (space_cell.setdefault(arm, {})
                             .setdefault(mname, {})[pass_key]) = s
                            (tspace.setdefault(arm, {})
                             .setdefault(mname, {})[pass_key]) = tl
                            pass_scores.setdefault(arm, {})[mname] = s
                    # score_clean for the rows: the same model on the clean
                    # training labels. Not an arm; not in the summary.
                    for mname in cell_models:
                        s, _tl, _err = _fit_score(
                            model_factories[mname], X_tr_s, y_clean_tr, X_va_s,
                            y_val, metric_fn, metric_name,
                            dataset_name=dataset_name, model_name=mname)
                        if s is not None:
                            pass_scores.setdefault('clean', {})[mname] = s
                    # ── Controls (KIRBY_NOISE_CONTROLS=1): every downweight
                    # arm is a weighting, twinned by controls.py. ──
                    for arm, w_by_m in arm_w.items():
                        if arm == 'untouched':
                            continue
                        for mname, w in w_by_m.items():
                            if w is None or np.all(w == 1.0):
                                continue
                            twins = control_twins(
                                arm_result(y0, weight=w), X_tr_s, y0,
                                (replicate_seed(arm, mname), draw, fold_i,
                                 int(round(level * 1000))))
                            for tw, by_m in twin_scores(
                                    dataset_name, twins, model_factories,
                                    [mname], X_va_s, y_val, metric).items():
                                pass_scores.setdefault(f'{arm}|{tw}',
                                                       {}).update(by_m)
                    pass_rows += arm_rows(
                        dataset_name=dataset_name, exp_id=EXP_ID,
                        metric=metric, condition=CHECK_CONDITION, level=level,
                        draw=draw, fold=fold_i, model_input=_ekey,
                        scores_by_arm=pass_scores, reference='untouched',
                        setting_of=_n35_row_setting)

            pair_acc.setdefault(lvl_key, []).append(prec)
            ledger.add_rows(pass_rows)
            ledger.finish_pass(pass_key, {'acc': acc, 'tails': tails,
                                          'errors': errors,
                                          'pair_acc': pair_acc})
            _b1 = next((k for k in prec['per_space']
                        if k == 'base_1' or k.startswith('base_1/')), None)
            b = prec['per_space'].get(_b1, {}) if _b1 else {}
            print(f"    level {lvl_key} fold {fold_i}: {n_tr} training rows, "
                  f"{b.get('n_close_pairs')} close pairs in {_b1}, closest-bin "
                  f"estimate {b.get('closest_bin_estimate', float('nan')):.4f}, "
                  f"injected {prec['realised_injected_variance']:.4f}, "
                  f"{b.get('flagged_pairs')} pairs flagged "
                  f"[{time.time() - t0:.0f}s]")

    results['per_level'] = convert_numpy(acc)
    results['tails_per_level'] = convert_numpy(tails)
    results['pairs_per_pass'] = convert_numpy(pair_acc)
    results['d_vs_dprime'] = ledger.describe()
    if errors:
        results['_errors'] = errors
    results['check'] = convert_numpy(_check(pair_acc))
    results['summary'] = convert_numpy(_summarise(
        acc, tails, downstream_models, metric_name, lower_is_better))
    _print_summary(results, metric_name)
    return results


def _n35_row_setting(arm):
    """(method, method_setting, similarity_setting) of an n35 arm for its row.
    Today's two arms are the model_input setting; downweight_pairs@<setting>
    is the same rule in that setting's distance."""
    base, _, twin = arm.partition('|')
    tail = f'|{twin}' if twin else ''
    rule = method_setting(close_percentile=CLOSE_PERCENTILE,
                          weight=DOWNWEIGHT)
    if base == 'downweight_pairs_this_space':
        return (f'downweight_pairs{tail}', rule, _ss.MODEL_INPUT.name)
    if base == 'downweight_pairs_forest_proximity':
        # A forest fitted on the model-input matrix and the labels as seen:
        # exactly the forest_proximity setting (similarity_space.py:531-541).
        return (f'downweight_pairs{tail}', rule, 'forest_proximity|none|forest')
    if base.startswith('downweight_pairs@'):
        return (f'downweight_pairs{tail}', rule, base.split('@', 1)[1])
    return (arm, 'default', _ss.MODEL_INPUT.name)


def _check(pair_acc):
    """Per space: the estimate's rise over level 0 beside the injected variance.

    Paired by fold: level L's fold f against level 0's fold f.
    """
    base_by_fold = {}
    for rec in pair_acc.get('0', []):
        f = rec['pass'].split('f')[-1]
        base_by_fold[f] = rec
    out = {}
    for lvl, recs in pair_acc.items():
        for rec in recs:
            f = rec['pass'].split('f')[-1]
            ref = base_by_fold.get(f)
            if ref is None:
                continue
            for sp, r in rec.get('per_space', {}).items():
                r0 = ref.get('per_space', {}).get(sp)
                if not r0 or 'closest_bin_estimate' not in r \
                        or 'closest_bin_estimate' not in r0:
                    continue
                row = out.setdefault(sp, {}).setdefault(lvl, {
                    'folds': [], 'estimate_rise': [],
                    'intercept_rise': [], 'realised_injected_variance': []})
                row['folds'].append(f)
                row['estimate_rise'].append(
                    r['closest_bin_estimate'] - r0['closest_bin_estimate'])
                row['intercept_rise'].append(
                    r['intercept_estimate'] - r0['intercept_estimate'])
                row['realised_injected_variance'].append(
                    rec['realised_injected_variance'])
    for sp, by_lvl in out.items():
        for lvl, row in by_lvl.items():
            for k in ('estimate_rise', 'intercept_rise',
                      'realised_injected_variance'):
                v = np.asarray(row[k], dtype=float)
                v = v[np.isfinite(v)]
                row[f'{k}_mean'] = float(v.mean()) if v.size else float('nan')
    return out


def _paired(base, got, lower_is_better):
    """n30's pairing: one arm against another on the passes both scored."""
    shared = sorted(set(base) & set(got))
    gains = np.array([(base[k] - got[k]) if lower_is_better
                      else (got[k] - base[k]) for k in shared], dtype=float)
    gains = gains[np.isfinite(gains)]
    # Finite scores only: one NaN untouched score made max(nan, 1.0) return
    # nan, so the tolerance was nan and every pass counted as neither won,
    # lost nor tied.
    _fin = np.asarray([v for v in base.values() if np.isfinite(v)], dtype=float)
    scale = max(abs(float(_fin.mean())) if _fin.size else 1.0, 1.0)
    tol = 1e-9 * scale
    return {
        'n_passes_paired': int(gains.size),
        'passes_won': int((gains > tol).sum()),
        'passes_lost': int((gains < -tol).sum()),
        'passes_tied': int((np.abs(gains) <= tol).sum()),
        'tie_tolerance': float(tol),
        'mean_gain': float(gains.mean()) if gains.size else float('nan'),
        'spread_over_pass_gains': (float(gains.std()) if gains.size
                                   else float('nan')),
        'pass_gains': [float(x) for x in gains],
    }


def _tail_means(d):
    keys = ('tail_rmse_top', 'tail_rmse_bottom', 'calibration_slope')
    out = {}
    for k in keys:
        v = [t[k] for t in d.values() if t and np.isfinite(t.get(k, np.nan))]
        out[f'{k}_mean'] = float(np.mean(v)) if v else float('nan')
    return out


def _summarise(acc, tails, models, metric_name, lower_is_better):
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'headline': ('passes won against untouched, paired per pass; a '
                        'pass is one (draw, fold)'),
           'per_level': {}}
    for lvl, by_space in acc.items():
        lrow = {}
        for space, by_arm in by_space.items():
            base = by_arm.get('untouched') or {}
            srow = {}
            for m in models:
                b = base.get(m) or {}
                if not b:
                    continue
                _bv = np.asarray(list(b.values()), dtype=float)
                row = {'untouched_mean': (float(_bv[np.isfinite(_bv)].mean())
                                          if np.isfinite(_bv).any()
                                          else float('nan')),
                       'untouched_nonfinite_passes': int(
                           (~np.isfinite(_bv)).sum()),
                       'untouched_tails': _tail_means(
                           tails[lvl][space]['untouched'][m]),
                       'per_arm': {}}
                # Today's arms first, then every downweight_pairs@<setting>
                # (2026-09-25), in the order they were scored.
                for arm in list(ARMS[1:]) + [a for a in by_arm
                                             if a not in ARMS]:
                    g = (by_arm.get(arm) or {}).get(m) or {}
                    if not g:
                        continue
                    row['per_arm'][arm] = {
                        'vs_untouched': _paired(b, g, lower_is_better),
                        'tails': _tail_means(tails[lvl][space][arm][m])}
                srow[m] = row
            lrow[space] = srow
        out['per_level'][lvl] = lrow
    return out


def _print_summary(results, metric_name):
    print("\n" + "=" * 70)
    print("CHECK: rise of the closest-bin estimate over level 0, beside the "
          "realised injected variance (label units squared)")
    for sp, by_lvl in results['check'].items():
        for lvl in sorted(by_lvl, key=float):
            r = by_lvl[lvl]
            print(f"  {sp:28s} level {lvl:>5s}: rise "
                  f"{r['estimate_rise_mean']:+.4f}, injected "
                  f"{r['realised_injected_variance_mean']:.4f} "
                  f"({len(r['folds'])} folds)")
    print(f"\nSUMMARY ({metric_name}; paired per pass against untouched)")
    for lvl, lrow in results['summary']['per_level'].items():
        print(f"\n  level {lvl}:")
        for space, srow in lrow.items():
            for m, row in srow.items():
                print(f"    {space}/{m}: untouched {row['untouched_mean']:.4f}")
                for arm, e in row['per_arm'].items():
                    v = e['vs_untouched']
                    print(f"      {arm:36s} won {v['passes_won']} of "
                          f"{v['n_passes_paired']} passes, mean "
                          f"{v['mean_gain']:+.4f} (spread "
                          f"{v['spread_over_pass_gains']:.4f})")
    print("\n  The count of passes won is the headline. A mean gain no larger "
          "than the spread of the per-pass gains behind it is not a gain.")


def main():
    parser = create_experiment_parser(
        'N35: near-duplicate label disagreement, and downweighting it')
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated gaussian levels. Default: the '
                             "dataset's level grid")
    parser.add_argument('--draws', type=str, default=None,
                        help='Comma-separated injection replicates (default 0)')
    parser.add_argument('--skip-proximity', action='store_true',
                        help='Do not compute the forest proximity')
    parser.add_argument('--skip-cliffs', action='store_true',
                        help='Do not count activity cliffs among flagged pairs')
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
    if args.levels:
        kw['levels'] = [float(x) for x in args.levels.split(',')]
    if args.draws:
        kw['draws'] = [int(x) for x in args.draws.split(',')]
    if args.skip_proximity:
        kw['skip_proximity'] = True
    if args.skip_cliffs:
        kw['skip_cliffs'] = True

    res = run(data, config, args.output, args.dataset, **kw)
    suffix = CHECK_CONDITION if not args.tag else f'{CHECK_CONDITION}_{args.tag}'
    save_experiment_results(res, args.output, EXP_ID, args.dataset,
                            filename_suffix=suffix)


if __name__ == '__main__':
    main()
