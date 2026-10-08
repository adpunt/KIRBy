#!/usr/bin/env python3
"""N35: how far apart are the labels of two nearly identical molecules?

Paper: Noise.

THE QUESTION. Two molecules at almost zero distance should have almost the same
true label. What their RECORDED labels still disagree by is label noise, read
without any model. The second research report of 2026-09-24 proposed it as D5:

    E[(y_i - y_j)^2] / 2  ->  noise variance, as the distance -> 0

and, as A7, downweighting both molecules of a near-duplicate pair whose labels
disagree by more than 3 * sqrt(2) * sigma, where sigma is that noise estimate.

CHAT 5 ALIGNMENT, 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT.md section 2.4,
KNN_SEARCH_SPACE_RECORD.md). What changed, in one place:
  - Models: the dataset's two decided models (DECISIONS.md Step 1), not the
    five of get_config plus rf and svm.
  - Folds: noise_helpers.noise_outer_folds, the splits the per-fold searches
    are filed under; the count is config['n_splits'] (KIRBY_NOISE_N_FOLDS
    overrides). Until 2026-10-05 the folds were _shared.cv_split.
  - Selection: per outer fold, per decided model, the 2 best base
    representations and the 2 best general hybrids from searches on that
    fold's training rows (noise_helpers.fold_configurations). 2 models x 4 =
    8 fitted configurations. Until 2026-10-05 the recipes and the base
    representations came from the whole-dataset stored searches, and each
    model was fitted on up to 6 hybrids.
  - Distances: the decided neighbour settings of the dataset's group
    (n07.sweep_settings). The distance hybrid and the similarity hybrid are
    distance spaces only, never fitted.
  - DECIDED 2026-10-06 (Adelaide delegated): the arm that found close pairs
    in the fitted configuration's own matrix (Euclidean, or Tanimoto on a 0/1
    base representation), downweight_pairs_this_space, is removed. The record
    cuts that space as a neighbour setting (axis 1, number 10) and cuts
    Euclidean (axis 2).
  - Refits: every setting gets a pair record (no model fit); only the screen's
    top 3 distinct settings (n07.screen_settings) get a downweighting refit.
    DECIDED 2026-10-06 (Adelaide delegated): top 3 means the 3 best distinct
    settings. The screen predicts a held-back fifth of the training
    molecules from lists built on the other four fifths (inner=).
  - Noise: gaussian at the decided levels of the dataset's group, 3 draws per
    non-zero level, level 0 once (noise_helpers.decided_levels,
    inject_condition).
  - float32 representations throughout. 0/1 features are left unscaled
    (kirby.scaling.FeatureScaler) in the base representations and in every
    distance space FoldSimilarity builds. NOT YET in the two fitted hybrids:
    fold_model_inputs builds them through n07.fold_similarity_spaces, which
    scales them with a plain StandardScaler (n07._scale_pair). That is shared
    code; the change is reported, not made here.

WHAT IS COMPUTED, per (level, draw, outer fold), on the TRAINING molecules only:

  1. Every pairwise distance among the fold's training molecules, in every
     neighbour setting of the dataset's group (n07.sweep_settings: the
     representation the distance is measured in, and the distance formula):
       base_1, base_2,        each decided model's rank-1 and rank-2 single
       base_k_aug_uncapped    representation from the fold's search, and the
                              same with augmentation features added
       hybrid,                that model's two general hybrids, its distance
       hybrid__general2,      hybrid and its similarity hybrid, rebuilt on the
       hybrid__distance1,     fold's training molecules from the labels AS
       hybrid_similarity      SEEN
       tanimoto_ecfp4         ECFP4 bits under Tanimoto (needs SMILES)
       forest_proximity       one minus the share of trees that put the two
                              molecules in one leaf, from a forest fitted on
                              THE FITTED CONFIGURATION'S matrix and the labels
                              as seen (one forest per configuration). ⛔ The
                              forest reads the labels, so two molecules it puts
                              together already tend to have close labels; its
                              estimate is biased low by construction.
     manhattan, cosine and minkowski p=0.5 on full_knn datasets; manhattan and
     cosine on fast_knn datasets. A feature-space setting is one matrix per
     model; Tanimoto is one per pass; forest proximity is one per fitted
     configuration. Each is written under its pair key (_pair_key).
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

THE CHECK. `gaussian` at the dataset's decided levels, injected into the
training labels only. A setting whose pair set depends on the features alone
(base_1, base_2, tanimoto_ecfp4; `pair_set_reads_labels` False) has the same
pairs at every level, so the increase of its estimate over level 0 should be
the injected variance itself -- each member of a pair carries its own draw, and
Var(e_i - e_j) / 2 = Var(e). The realised injected variance,
mean((y_seen - y_clean)^2) over the training molecules, is written beside it.
The hybrids and augmented spaces are rebuilt from the noisy labels at every
level and each forest is refitted on them, so their pair sets can move; they
are not expected to track exactly.

THE METHOD ARM (A7). Per (level, draw, fold, fitted configuration), all scored
on the same held-out molecules against their CLEAN labels:

  untouched                    the training labels as seen, weight 1
  downweight_pairs@<setting>   both molecules of every close pair in that
                               setting's distance whose |dy| is over
                               3 * sqrt(2) * sigma get weight 0.25. sigma^2 =
                               the closest-bin estimate, same setting, same
                               labels. Fitted for the screen's top 3 settings
                               only (below).
  clean                        the same model on the clean training labels;
                               for the rows and the damaged-minus-clean
                               difference, not an arm of the summary.

THE SCREEN (KNN_SEARCH_SPACE_RECORD, "Screening the kNN settings per dataset",
Adelaide 2026-10-05). Per fitted configuration, every setting is scored with no
model fit on how well its neighbours predict the labels as seen of a held-back
fifth of the training molecules (whole scaffold groups), from neighbour lists
built on the other four fifths, at 5, 10 and 20 neighbours under both counting
schemes (n07.screen_settings with inner=). n35's rule reads no neighbour count
and no counting scheme, so the unit refitted is a setting: the 3 distinct
settings whose best combination ranks highest (KIRBY_NOISE_SCREEN_TOP
overrides; 0 refits all). DECIDED 2026-10-06 (Adelaide delegated).

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

Usage:
    python n35_near_duplicate_disagreement.py --dataset herg_ki --mode full
    python n35_near_duplicate_disagreement.py --dataset herg_ic50 --mode full --levels 0,0.5
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import time

import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories,
)
from experiments.dataset_policy import noise_groups_for, noise_dose_units
from experiments.decided_grid import decided_models
from experiments.noise.noise_helpers import (
    tail_scores, noise_outer_folds, fold_spread, fold_configurations,
    fold_model_inputs, configuration_record, inject_condition,
    decided_levels, dataset_knn_group, level_unit, CONFIGURATION_SPACES,
    draws_for,
)
from experiments.noise.n07_remediation import (
    sweep_settings, sweep_ks, sweep_schemes, screen_settings, screen_top,
    SCREEN_SCORED_ON,
)
from experiments.noise._ground_truth_providers import build_replicate_groups
from experiments.noise import similarity_space as _ss
# The D-versus-D' rows, per-pass checkpoint, cached fits, controls and the
# fold's distance spaces shared by n32-n41 (n32's shared block, 2026-09-25).
from experiments.noise.n32_iterative_and_dual_selection import (
    method_setting, fit_predict_cached, ledger_signature, PassLedger,
    restart_flag, arm_rows, arm_result, control_twins, twin_scores,
    controls_enabled, replicate_seed, fold_similarity,
)

EXP_ID = 'n35_near_duplicate_disagreement'

# Bumped 2026-10-05 (chat 5): it is part of the checkpoint signature, so a
# rows file or checkpoint of the old layout (whole-dataset recipes, five
# models, cv_split folds) is set aside and never read back.
RESULT_LAYOUT = 'chat5_per_fold_2026_10_06'
# 2026-10-06: downweight_pairs_this_space removed and the screen scored on a
# held-back fifth of the training molecules, so a checkpoint of the
# 2026-10-05 layout is set aside.

# The only injected condition. The check needs noise whose variance is the
# same on every molecule, so that the increase of the estimate is one number.
CHECK_CONDITION = 'gaussian'
# 3 draws per non-zero level, level 0 once (chat 5, 2026-10-05). Was (0,).
NOISE_DRAWS = (0, 1, 2)
# Close pairs: at or below this percentile of the pairwise distances.
CLOSE_PERCENTILE = 1.0
# Equal-count bins over the close pairs, by distance.
N_BINS = 5
# Flag rule, from the report: |dy| over this many sigma.
FLAG_MULTIPLIER = 3.0 * np.sqrt(2.0)
DOWNWEIGHT = 0.25
# Above this many training rows the pairwise distances are not computed. A
# float32 square matrix at 12,000 rows is 576 MB.
MAX_PAIR_ROWS = 12000
# n25's cliff definition (n01's numbers).
CLIFF_SIM_THRESHOLD = 0.55
CLIFF_ACTIVITY_K = 2.0

# The 4 fitted configurations of one decided model
# (noise_helpers.CONFIGURATION_SPACES): base_1 and base_2 are fixed per fold;
# the two hybrids are rebuilt from each pass's labels as seen.
BASE_SPACES = ('base_1', 'base_2')
HYBRID_FIT_SPACES = tuple(sp for sp in CONFIGURATION_SPACES
                          if sp not in BASE_SPACES)
# Settings whose pair set does not read the labels: the check's reference.
LABEL_FREE_SPACES = ('base_1', 'base_2', _ss.TANIMOTO_SPACE)
ARM_TO_BEAT = 'untouched'
ARM_PREFIX = 'downweight_pairs@'
# DECIDED 2026-10-06 (Adelaide delegated): n35's own close-pair arm,
# downweight_pairs_this_space (close pairs in the fitted configuration's own
# matrix, Euclidean or Tanimoto on a 0/1 base), is removed.
# KNN_SEARCH_SPACE_RECORD cuts that space (axis 1, number 10) and Euclidean
# (axis 2).
CONTROL_TWINS = ('random_twin', 'shuffled_twin')


# ---------------------------------------------------------------------------
# Distances and close pairs
# ---------------------------------------------------------------------------

# is_binary, tanimoto_distance_matrix and euclidean_distance_matrix were the
# distance of the removed downweight_pairs_this_space arm. n37 and
# _smoke_n35 still import them, so they stay.

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
    # The matrix keeps its own dtype (float32 on load, handoff 2.3 item 18);
    # until 2026-10-05 it was cast to float64 first.
    return np.asarray(euclidean_distances(np.asarray(X)), dtype=np.float32)


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


def _levels_and_draws(data, dataset_name, levels=None, draws=None):
    """(level, draw) pairs. Levels are the decided gaussian levels of the
    dataset's group (noise_helpers.decided_levels, Adelaide 2026-10-05):
    0, 0.5, 1.0, 1.5 on full_knn and 0, 1.0 on fast_knn. Level 0 is the labels
    as delivered and runs once; every other level runs each draw. A `levels`
    kwarg (a smoke, --levels) overrides. Until 2026-10-05 the levels came from
    dataset_policy.noise_levels with one draw."""
    levels = (decided_levels(data, dataset_name, CHECK_CONDITION)
              if levels is None else levels)
    draws = draws_for(CHECK_CONDITION,
                      tuple(NOISE_DRAWS if draws is None else draws))
    out = []
    for lvl in levels:
        lvl = float(lvl)
        for d in (draws[:1] if lvl == 0.0 else draws):
            out.append((lvl, int(d)))
    return out


def _pass_key(level, draw, fold_i):
    return f'l{float(level):g}d{int(draw)}f{int(fold_i)}'


def _fold_of(pass_key):
    return pass_key.rsplit('f', 1)[1]


def _pair_key(setting, model, space):
    """Where a setting's pair record is written. A feature-space setting is
    one matrix per model; Tanimoto one per pass; forest proximity is fitted on
    the configuration's own matrix, so one per (model, configuration)."""
    if setting.space == _ss.FOREST_SPACE:
        return f'{setting.name}|{model}|{space}'
    if setting.space in _ss.FEATURE_SPACES:
        return f'{setting.name}|{model}'
    return setting.name


def _screen_distinct(settings, nbrs, y_seen, metric, inner):
    """The screen's ranking, cut to distinct settings.

    n07.screen_settings ranks every (setting, neighbour count, counting
    scheme) on how well the neighbours predict the labels as seen of a
    held-back fifth of the training molecules (inner=, lists rebuilt on the
    other four fifths).
    n35's flag rule reads no neighbour count and no counting scheme, so what
    is refitted here is a setting: the first screen_top() distinct settings of
    the ranking (each at its best combination). Returns (kept setting names,
    compact table)."""
    # DECIDED 2026-10-06 (Adelaide delegated): top 3 means the 3 best
    # distinct settings, so 3 downweighting refits per configuration and 3
    # distances tested. Screen calls pass inner=.
    top = screen_top()
    ranked, table = screen_settings(settings, nbrs, y_seen, metric=metric,
                                    is_clf=False, top=0, inner=inner)
    kept = []
    for s, _k, _sch in ranked:
        if s.name not in kept:
            kept.append(s.name)
        if top > 0 and len(kept) >= top:
            break
    cols = ('setting', 'k', 'scheme', 'score', 'error')
    return kept, {'columns': list(cols),
                  'rows': [[r.get(c) for c in cols] for r in table]}


def run(data, config, output_dir, dataset_name, force_rerun=False, **kwargs):
    """N35: label disagreement of near-duplicate pairs, and downweighting them.

    CHAT 5 ALIGNMENT, 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT section 2.4).
    What each outer fold does, in order:
      1. Folds are noise_helpers.noise_outer_folds. A fold with more than
         MAX_PAIR_ROWS training rows is refused before any search runs.
      2. noise_helpers.fold_configurations runs this fold's searches on its
         training rows (clean labels) and gives each decided model its 2 best
         base representations, its 2 best general hybrids, and the recipes of
         its distance hybrid and similarity hybrid.
      3. Per (level, draw): gaussian is injected into the training labels;
         the two fitted hybrids are rebuilt from those labels with the fold
         search's own representation scores (dirty-run bug 1).
      4. Per fitted configuration: a pair record in every neighbour setting
         (no fit), the screen (scored on a held-back fifth of the training
         molecules), then untouched, clean, and one downweighting refit per
         kept setting.
    Each (level, draw, fold) pass is checkpointed as it finishes, rows first.

    force_rerun (run_dataset.py --force-rerun, via run_all) sets aside the
    rows file and the per-pass checkpoint and recomputes every pass."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N35: NEAR-DUPLICATE LABEL DISAGREEMENT")
    print("=" * 70)

    labels = np.asarray(data['labels'], dtype=float)
    groups = data.get('groups')
    groups_arr = None if groups is None else np.asarray(groups)
    task_type = data['task_type']
    if task_type != 'regression':
        print(f"  SKIPPED: {dataset_name} is {task_type}. A squared label "
              f"difference is not defined between two classes.")
        return {'skipped': 'not_regression', 'task_type': task_type}
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)
    run_proximity = not kwargs.get('skip_proximity', False)
    count_cliffs = not kwargs.get('skip_cliffs', False)

    # R1 (chat 5, 2026-10-05): the dataset's two decided models
    # (DECISIONS.md Step 1); knn runs only inside the distance search. Was
    # n32.downstream_model_list: config['models'] (five) plus rf and svm. A
    # `models` kwarg (a smoke, a direct caller) can only narrow the pair.
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
              f"n35 fits the decided models {decided}")
    if kwargs.get('base_reps') is not None:
        print("  base_reps given by the caller are not read: each fold's base "
              "representations come from that fold's search")
    model_factories = get_model_factories(
        models, task_type=task_type, n_samples=len(labels))
    models_without_factory = [m for m in models if m not in model_factories]
    models = [m for m in models if m in model_factories]
    if not models:
        raise RuntimeError(
            f"n35: no decided model could be built for {dataset_name}. "
            f"Decided {decided}, asked {list(asked or [])}, factories "
            f"{sorted(model_factories)}.")

    noise_groups = noise_groups_for(data)[0]
    rep_groups = build_replicate_groups(data)

    # The neighbour settings (KNN_SEARCH_SPACE_RECORD, Adelaide 2026-10-05):
    # the dataset group's spaces and distances, no importance weighting, no
    # model_input. Until 2026-10-05 n35 called n32.sweep_settings() with no
    # dataset, which returned full_knn's list on every dataset.
    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    knn_grp = dataset_knn_group(data, dataset_name)
    n_molecules = len(set(smiles)) if has_smiles else len(labels)
    sim_settings = list(sweep_settings(dataset_name, n_molecules,
                                       has_smiles=has_smiles))
    if not run_proximity:
        sim_settings = [s for s in sim_settings
                        if s.space != _ss.FOREST_SPACE]

    fps_all = None
    cliff_note = None
    if count_cliffs and has_smiles \
            and data.get('domain', 'molecular') == 'molecular':
        fps_all = ecfp4_fingerprints(smiles)
        if fps_all is None:
            cliff_note = 'RDKit unavailable; cliffs counted through n25 instead'

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'result_layout': RESULT_LAYOUT,
        'selection_scope': 'fold_train_rows',
        'noise_condition': CHECK_CONDITION,
        'level_unit': level_unit(CHECK_CONDITION, False),
        'dose_units': noise_dose_units(dataset_name),
        'knn_group': knn_grp,
        'models': models,
        'models_decided': decided,
        'models_asked_but_not_decided': models_not_decided,
        'models_without_factory': models_without_factory,
        'configurations_per_model': list(CONFIGURATION_SPACES),
        'arm_to_beat': ARM_TO_BEAT,
        'arm_prefix': ARM_PREFIX,
        'close_percentile': CLOSE_PERCENTILE,
        'n_bins': N_BINS,
        'flag_rule': (f'|dy| > {FLAG_MULTIPLIER:.4f} * sigma, sigma^2 the '
                      f'closest-bin estimate in the same setting'),
        'downweight': DOWNWEIGHT,
        'max_pair_rows': MAX_PAIR_ROWS,
        'has_repeat_measurements': rep_groups is not None,
        'similarity_settings': [s.name for s in sim_settings],
        'proximity_run': run_proximity,
        'screen': {'top_distinct_settings': screen_top(),
                   'neighbour_counts': list(sweep_ks()),
                   'counting_schemes': list(sweep_schemes()),
                   'scored_on': SCREEN_SCORED_ON,
                   'note': (
                       'Per fitted configuration, every (setting, neighbour '
                       'count, counting scheme) is scored with no model fit '
                       'on a held-back fifth of the training molecules, '
                       'lists built on the other four fifths '
                       '(n07.screen_settings, inner=); the first top_distinct_'
                       'settings distinct settings of that ranking get a '
                       'downweighting refit. The tables are in '
                       'pairs_per_pass[level][i].screen.')},
        'estimate_note': (
            'closest_bin_estimate is mean(dy^2)/2 over the closest fifth of the '
            'close pairs; intercept_estimate carries the per-bin values to '
            'distance 0 with a straight line. Both hold some real activity '
            'difference and are upper bounds on the noise variance.'),
        'proximity_note': (
            'forest_proximity is one forest per fitted configuration, fitted '
            'on that configuration\'s matrix and the labels as seen. Its close '
            'pairs already tend to agree on the label, so its estimate is '
            'biased low by construction.'),
        'check_note': (
            'Under gaussian, a setting whose pair set does not read the labels '
            '(pair_set_reads_labels False: base_1, base_2, tanimoto_ecfp4) '
            'has the same pairs at every level, so its closest-bin estimate minus '
            'its level-0 value should equal the realised injected variance on '
            'the same fold. Per fold the draws are averaged; over folds, '
            'fold_spread.'),
        'robustness_note': (
            'damaged_minus_clean in the summary is the untouched score minus '
            'the clean-label score of the same model and configuration, in the '
            "dataset's own metric (read with lower_is_better), averaged over "
            'draws per fold, then fold_spread over folds. No auc_norm is '
            'computed here; on fast_knn datasets it is absent by decision.'),
        'model_input_note': (
            'downweight_pairs_this_space (close pairs in each fitted matrix, '
            'Euclidean or Tanimoto on a 0/1 base) was removed 2026-10-06 '
            '(Adelaide delegated): KNN_SEARCH_SPACE_RECORD cuts that space '
            '(axis 1, number 10) and euclidean (axis 2). Until 2026-10-05 a '
            'forest on each fitted matrix was a second fixed arm; it is now '
            'the forest_proximity setting, refitted only when the screen '
            'keeps it.'),
        'hybrid_scaling_note': (
            'The two fitted hybrids are standardised with a plain '
            'StandardScaler inside the shared n07.fold_similarity_spaces, so '
            'their 0/1 features are scaled; the same recipes as distance '
            'spaces use FeatureScaler, which leaves 0/1 features as they are.'),
        'cliff_note': cliff_note,
        'controls_run': controls_enabled(),
        'rows_note': ('D is untouched; score_clean is the same model fitted on '
                      'the clean training labels. model_input in a row is the '
                      'configuration name (base_1, base_2, hybrid, '
                      'hybrid__general2) and the model is its own column.'),
    }

    folds, n_folds = noise_outer_folds(data, config)
    results['n_folds'] = int(n_folds)
    results['folds_run'] = [int(f) for f, _, _ in folds]
    passes = _levels_and_draws(data, dataset_name, levels=kwargs.get('levels'),
                               draws=kwargs.get('draws'))
    results['noise_levels'] = sorted({l for l, _ in passes})
    results['passes'] = [f'{l:g}|d{d}' for l, d in passes]
    n_cfg = len(models) * len(CONFIGURATION_SPACES)
    results['fitted_configurations_per_fold'] = n_cfg
    print(f"  {len(labels)} rows, {n_folds} outer folds (running "
          f"{results['folds_run']}), decided models {models}, group {knn_grp}")
    print(f"  {len(passes)} (level, draw) passes at levels "
          f"{results['noise_levels']}; {n_cfg} fitted configurations per fold; "
          f"{len(sim_settings)} neighbour settings, top {screen_top()} "
          f"refitted")

    # acc[level][space][arm][model][pass_key] = held-out score. space is the
    # configuration name, keyed under the model whose configuration it is.
    acc, tails, errors, pair_acc = {}, {}, {}, {}
    fold_records = {}
    config_unavailable = {}
    t0 = time.time()
    ledger = PassLedger(
        output_dir, EXP_ID, dataset_name, CHECK_CONDITION,
        signature=ledger_signature(
            data, config, results, [(tr, va) for _, tr, va in folds],
            extra={'layout': RESULT_LAYOUT, 'passes': results['passes']}),
        restart=bool(force_rerun) or restart_flag(kwargs))
    if ledger.restored():
        st_ = ledger.restored()
        acc, tails, errors, pair_acc = (st_['acc'], st_['tails'],
                                        st_['errors'], st_['pair_acc'])
        fold_records = st_.get('fold_records', {})
        config_unavailable = st_.get('config_unavailable', {})
        print(f"  resumed {len(ledger.passes_done)} finished passes")

    def _state():
        return {'acc': acc, 'tails': tails, 'errors': errors,
                'pair_acc': pair_acc, 'fold_records': fold_records,
                'config_unavailable': config_unavailable}

    for fold_i, train_idx, val_idx in folds:
        _keys = [_pass_key(l, d, fold_i) for l, d in passes]
        if all(ledger.pass_done(k) for k in _keys):
            print(f"\n  Fold {fold_i}: every pass is on disk in the checkpoint")
            continue
        n_tr = len(train_idx)
        print(f"\n  Fold {fold_i} of {n_folds}: {n_tr} training, "
              f"{len(val_idx)} held-out rows")
        if n_tr > MAX_PAIR_ROWS:
            # Refused before the fold's searches run, so nothing is searched
            # for a fold that scores nothing (audit, "Other problems").
            why = (f'{n_tr} training rows is over MAX_PAIR_ROWS='
                   f'{MAX_PAIR_ROWS}; the pairwise distances were not '
                   f'computed and no arm ran')
            fold_records[str(fold_i)] = {'refused': why}
            for (level, draw), pk in zip(passes, _keys):
                if ledger.pass_done(pk):
                    continue
                pair_acc.setdefault(f'{level:g}', []).append(
                    {'pass': pk, 'fold': int(fold_i), 'draw': int(draw),
                     'training_rows': int(n_tr), 'refused': why})
                ledger.finish_pass(pk, _state())
            print(f"    {why}")
            continue
        try:
            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=train_idx, n_folds=n_folds)
        except Exception as exc:
            why = (f'the per-fold searches failed: {type(exc).__name__}: '
                   f'{exc}')
            print(f"    {why}")
            errors[f'f{fold_i}|fold_configurations'] = why
            fold_records[str(fold_i)] = {'error': why}
            continue

        y_val = labels[val_idx]
        y_clean_tr = labels[train_idx]
        g_tr = None if groups_arr is None else groups_arr[train_idx]
        same_mol = (np.asarray(rep_groups)[train_idx]
                    if rep_groups is not None else None)
        fps_train = ([fps_all[i] for i in train_idx]
                     if fps_all is not None else None)

        # base_1 and base_2 of each model read no labels, so one matrix per
        # fold serves every pass. float32 kept and 0/1 features unscaled
        # (handoff 2.3 items 17, 18): until 2026-10-05 every representation
        # was cast to float64 here and every feature standardised.
        base_inputs = {}
        frec = {'n_train': int(n_tr), 'n_val': int(len(val_idx)),
                'models': {}}
        for m in models:
            entry = entries.get(m)
            ins, unav, names = fold_model_inputs(
                data, dataset_name, m, entry, train_idx=train_idx,
                val_idx=val_idx, y_train=y_clean_tr,
                factory=model_factories[m], metric=metric, spaces=BASE_SPACES)
            base_inputs[m] = (ins, unav)
            rec = configuration_record(entry, names)
            rec['base_unavailable'] = dict(unav)
            rec['metric'] = metric_name
            rec['lower_is_better'] = bool(lower_is_better)
            rec['training_row_score_note'] = (
                "hybrids[*].score_mean is the fold search's score of that "
                "hybrid on this fold's training rows (its own inner folds), "
                "in the search metric. The base representations' search "
                "scores are not handed through fold_configurations. The "
                "held-out scores are in held_out_score, keyed by level|draw, "
                "then configuration, then arm.")
            rec['held_out_score'] = {}
            frec['models'][m] = rec
            print(f"    {m}: base {rec.get('base')}, hybrids "
                  f"{[h.get('label') for h in rec.get('hybrids') or []]}")
        _prev = (fold_records.get(str(fold_i)) or {}).get('models') or {}
        for m, rec in frec['models'].items():
            rec['held_out_score'].update(
                (_prev.get(m) or {}).get('held_out_score') or {})
        fold_records[str(fold_i)] = frec
        base_by_model = {m: list((entries.get(m) or {}).get('base') or [])
                         for m in models}
        recipes_by_model = {m: dict((entries.get(m) or {}).get('recipes')
                                    or {}) for m in models}

        for (level, draw), pass_key in zip(passes, _keys):
            if ledger.pass_done(pass_key):
                continue
            lvl_key = f'{level:g}'
            cell = acc.setdefault(lvl_key, {})
            tcell = tails.setdefault(lvl_key, {})
            print(f"\n    --- fold {fold_i}, gaussian / level {lvl_key} / "
                  f"draw {draw} ---")
            pass_rows = []
            try:
                y0, _mask, _prov = inject_condition(
                    data, dataset_name, CHECK_CONDITION, level, draw,
                    train_idx=train_idx, groups=noise_groups)
            except Exception as exc:
                errors[f'{lvl_key}|d{draw}|f{fold_i}|inject'] = (
                    f'{type(exc).__name__}: {exc}')
                continue
            y0 = np.asarray(y0, dtype=float)
            prec = {
                'pass': pass_key, 'fold': int(fold_i), 'draw': int(draw),
                'training_rows': int(n_tr),
                'realised_injected_variance': float(
                    np.mean((y0 - y_clean_tr) ** 2)),
                'per_space': {}, 'screen': {}, 'kept_settings': {},
            }
            if rep_groups is not None:
                prec['replicate_variance_seen'] = pooled_replicate_variance(
                    y0, same_mol)
                prec['replicate_variance_clean'] = pooled_replicate_variance(
                    y_clean_tr, same_mol)

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
                    errors[f'{lvl_key}|d{draw}|f{fold_i}|cliffs'] = (
                        f'{type(exc).__name__}: {exc}')

            def _describe(key, D, setting, model, space):
                rec, (pi, pj, pdy) = pair_record(D, y0, same_mol)
                flagged = flag_pairs(pi, pj, pdy, rec['closest_bin_estimate'])
                rec['flagged_pairs'] = int(flagged.sum())
                w = pair_weights(n_tr, pi, pj, flagged)
                rec['flagged_molecules'] = int(np.sum(w < 1.0))
                rec['setting'] = setting.name
                rec['model'] = (model if setting.space in _ss.FEATURE_SPACES
                                or setting.space == _ss.FOREST_SPACE
                                else None)
                rec['configuration'] = (space if setting.space
                                        == _ss.FOREST_SPACE else None)
                rec['pair_set_reads_labels'] = (setting.space
                                                not in LABEL_FREE_SPACES)
                if count_cliffs:
                    rec['cliffs'] = count_cliff_pairs(
                        pi[flagged], pj[flagged], pdy[flagged], y0,
                        fps_train=fps_train, pair_matrix=cliff_pairs_mat,
                        cliff_molecules=cliff_mols)
                prec['per_space'][key] = rec
                return w

            # The fold's distance spaces, built on demand from the labels as
            # seen: base_*, base_k_aug_uncapped and the four hybrid spaces per
            # model, from the per-fold recipes (each a (Blueprint, augmentation
            # condition) pair, rebuilt WITH augmentation where it was searched
            # with it).
            fsim = fold_similarity(
                dataset_name, data, reps=data['reps'], train_idx=train_idx,
                val_idx=val_idx, y_train=y0, groups_train=g_tr,
                base_by_model=base_by_model,
                recipe_for_space_by_model=recipes_by_model,
                model_factories=model_factories, metric=metric,
                task_type=task_type)
            w_cache = {}

            def _weights(setting, model, space, X_tr_s):
                """The setting's downweighting vector, its pair record written
                once per pair key. None when the setting is unavailable."""
                key = _pair_key(setting, model, space)
                if key in w_cache:
                    return w_cache[key]
                w = None
                try:
                    D, why = fsim.full_distance(setting, model, X_tr_s,
                                                dtype=np.float32)
                    if D is None:
                        prec['per_space'][key] = {'setting': setting.name,
                                                  'unavailable': why}
                    else:
                        w = _describe(key, D, setting, model, space)
                        del D
                except Exception as exc:
                    errors[f'{lvl_key}|{key}|d{draw}f{fold_i}'] = (
                        f'{type(exc).__name__}: {exc}')
                w_cache[key] = w
                return w

            # Each model's 4 fitted configurations: base_1 and base_2 from the
            # fold, the two hybrids rebuilt from this pass's labels as seen.
            configs = []
            for m in models:
                entry = entries.get(m)
                hyb, unav_h, _n = fold_model_inputs(
                    data, dataset_name, m, entry, train_idx=train_idx,
                    val_idx=val_idx, y_train=y0, factory=model_factories[m],
                    metric=metric, spaces=HYBRID_FIT_SPACES)
                b_ins, b_unav = base_inputs[m]
                for sp in CONFIGURATION_SPACES:
                    src_ = hyb if sp in HYBRID_FIT_SPACES else b_ins
                    unav_ = unav_h if sp in HYBRID_FIT_SPACES else b_unav
                    pair = src_.get(sp)
                    if pair is None:
                        (config_unavailable.setdefault(pass_key, {})
                         .setdefault(m, {})[sp]) = (unav_.get(sp)
                                                    or 'not built')
                        continue
                    configs.append((m, sp, pair[0], pair[1]))

            for mname, space, X_tr_s, X_va_s in configs:
                cfg_key = f'{space}/{mname}'
                space_cell = cell.setdefault(space, {})
                tspace = tcell.setdefault(space, {})

                # Scoring only: a pair record in every setting (no fit).
                for s in sim_settings:
                    _weights(s, mname, space, X_tr_s)

                # The screen: neighbours of every setting, scored on the
                # labels as seen. Scaffold-mates stay in the lists (record
                # axis 6), so no groups are excluded.
                nbrs = {}
                for s in sim_settings:
                    try:
                        nbrs[s.name] = fsim.neighbours(s, mname, X_tr_s)
                    except Exception as exc:
                        nbrs[s.name] = {'available': False,
                                        'reason': f'{type(exc).__name__}: '
                                                  f'{exc}'}
                # DECIDED 2026-10-06 (Adelaide delegated): the screen
                # predicts a held-back fifth of the training molecules, from
                # lists rebuilt on the other four fifths of the same spaces
                # (n07.screen_settings inner=). The spaces are the ones the
                # lists above were built from: the fitted matrix as
                # model_input (forest proximity) and each feature space of
                # this model.
                sp_tr = {'model_input': X_tr_s}
                for s in sim_settings:
                    if s.space in _ss.FEATURE_SPACES \
                            and s.space not in sp_tr:
                        Z_tr, _Z_va = fsim.feature_space(mname, s.space)
                        if Z_tr is not None:
                            sp_tr[s.space] = Z_tr
                inner = {'dataset_name': dataset_name, 'spaces_train': sp_tr,
                         'y_train': y0, 'groups_train': g_tr,
                         'smiles_train': (None if not has_smiles else
                                          [smiles[i] for i in train_idx]),
                         'is_clf': False}
                try:
                    kept, table = _screen_distinct(sim_settings, nbrs, y0,
                                                   metric, inner)
                except Exception as exc:
                    kept, table = [], {'error': f'{type(exc).__name__}: {exc}'}
                    errors[f'{lvl_key}|{cfg_key}|d{draw}f{fold_i}|screen'] = (
                        table['error'])
                del nbrs
                prec['screen'][cfg_key] = table
                prec['kept_settings'][cfg_key] = kept

                by_name = {s.name: s for s in sim_settings}
                arm_w = {ARM_TO_BEAT: None}
                # DECIDED 2026-10-06 (Adelaide delegated):
                # downweight_pairs_this_space is removed; only the screen's
                # kept settings are refitted.
                for name in kept:
                    w = _weights(by_name[name], mname, space, X_tr_s)
                    if w is None:
                        continue
                    arm_w[f'{ARM_PREFIX}{name}'] = w

                pass_scores = {}
                for arm, w in arm_w.items():
                    if arm != ARM_TO_BEAT and np.all(w == 1.0):
                        # Nothing flagged: the arm IS the untouched fit.
                        # Scored anyway so the pairing has every pass, and the
                        # count of flagged molecules says why it tied.
                        w = None
                    s_, tl, err = _fit_score(
                        model_factories[mname], X_tr_s, y0, X_va_s, y_val,
                        metric_fn, metric_name, weights=w,
                        dataset_name=dataset_name, model_name=mname)
                    if s_ is None:
                        errors[f'{lvl_key}|{cfg_key}|{arm}|d{draw}f{fold_i}'] = err
                        continue
                    (space_cell.setdefault(arm, {})
                     .setdefault(mname, {})[pass_key]) = s_
                    (tspace.setdefault(arm, {})
                     .setdefault(mname, {})[pass_key]) = tl
                    pass_scores.setdefault(arm, {})[mname] = s_
                # clean: the same model on the clean training labels. For the
                # rows and damaged_minus_clean; not an arm of the summary.
                s_, tl, err = _fit_score(
                    model_factories[mname], X_tr_s, y_clean_tr, X_va_s, y_val,
                    metric_fn, metric_name, dataset_name=dataset_name,
                    model_name=mname)
                if s_ is not None:
                    pass_scores.setdefault('clean', {})[mname] = s_
                    (space_cell.setdefault('clean', {})
                     .setdefault(mname, {})[pass_key]) = s_
                else:
                    errors[f'{lvl_key}|{cfg_key}|clean|d{draw}f{fold_i}'] = err
                (frec['models'][mname]['held_out_score']
                 .setdefault(f'{lvl_key}|d{draw}', {}))[space] = {
                    arm: (pass_scores.get(arm) or {}).get(mname)
                    for arm in list(arm_w) + ['clean']}

                # ── Controls (KIRBY_NOISE_CONTROLS=1): every downweight arm
                # is a weighting, twinned by controls.py. ──
                for arm, w in arm_w.items():
                    if arm == ARM_TO_BEAT or w is None or np.all(w == 1.0):
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
                # ── D against D'. model_input is the configuration name
                # alone and the model its own column (dirty-run bug 13: until
                # 2026-10-05 n32-n35 wrote 'base_1/rf', n30 and n31
                # 'base_1'). ──
                pass_rows += arm_rows(
                    dataset_name=dataset_name, exp_id=EXP_ID,
                    metric=metric, condition=CHECK_CONDITION, level=level,
                    draw=draw, fold=fold_i, model_input=space,
                    scores_by_arm=pass_scores, reference=ARM_TO_BEAT,
                    setting_of=_n35_row_setting)

            pair_acc.setdefault(lvl_key, []).append(prec)
            ledger.add_rows(pass_rows)
            ledger.finish_pass(pass_key, _state())
            _b1 = next((k for k in prec['per_space']
                        if k.startswith('base_1|')
                        and 'closest_bin_estimate' in prec['per_space'][k]),
                       None)
            b = prec['per_space'].get(_b1, {}) if _b1 else {}
            print(f"    level {lvl_key} d{draw} fold {fold_i}: {n_tr} training "
                  f"rows, {b.get('n_close_pairs')} close pairs in {_b1}, "
                  f"closest-bin estimate "
                  f"{b.get('closest_bin_estimate', float('nan')):.4f}, "
                  f"injected {prec['realised_injected_variance']:.4f}, "
                  f"{b.get('flagged_pairs')} pairs flagged "
                  f"[{time.time() - t0:.0f}s]")

    results['per_fold'] = convert_numpy(fold_records)
    results['configurations_unavailable'] = convert_numpy(config_unavailable)
    results['per_level'] = convert_numpy(acc)
    results['tails_per_level'] = convert_numpy(tails)
    results['pairs_per_pass'] = convert_numpy(pair_acc)
    results['d_vs_dprime'] = ledger.describe()
    if errors:
        results['_errors'] = errors
    results['check'] = convert_numpy(_check(pair_acc))
    results['summary'] = convert_numpy(_summarise(
        acc, tails, models, metric_name, lower_is_better))
    _print_summary(results, metric_name)
    return results


def _n35_row_setting(arm):
    """(method, method_setting, similarity_setting) of an n35 arm for its row.

    A setting name holds '|' itself ('base_1|none|manhattan'), so a control
    twin's suffix is split off by its name. Until 2026-10-05 the arm was split
    at the first '|', which cut every downweight_pairs@<setting> row's
    similarity_setting to its space alone."""
    base, twin = arm, ''
    for tw in CONTROL_TWINS:
        if arm.endswith(f'|{tw}'):
            base, twin = arm[:-len(tw) - 1], tw
            break
    tail = f'|{twin}' if twin else ''
    rule = method_setting(close_percentile=CLOSE_PERCENTILE,
                          weight=DOWNWEIGHT)
    if base.startswith(ARM_PREFIX):
        return (f'downweight_pairs{tail}', rule, base[len(ARM_PREFIX):])
    return (arm, 'default', _ss.MODEL_INPUT.name)


def _check(pair_acc):
    """Per pair key: the estimate's rise over level 0, beside the injected
    variance. Level L's (draw, fold) is paired with level 0's same fold; the
    draws of a fold are averaged, then fold_spread over folds."""
    ref = {}
    for rec in pair_acc.get('0', []):
        if 'per_space' in rec:
            ref[str(rec.get('fold', _fold_of(rec['pass'])))] = rec
    raw = {}
    for lvl, recs in pair_acc.items():
        for rec in recs:
            f = str(rec.get('fold', _fold_of(rec['pass'])))
            r0rec = ref.get(f)
            if r0rec is None or 'per_space' not in rec:
                continue
            for key, r in rec['per_space'].items():
                r0 = r0rec['per_space'].get(key)
                if not r0 or 'closest_bin_estimate' not in r \
                        or 'closest_bin_estimate' not in r0:
                    continue
                row = raw.setdefault(key, {}).setdefault(lvl, {
                    'pair_set_reads_labels': r.get('pair_set_reads_labels'),
                    'by_fold': {}})
                bf = row['by_fold'].setdefault(f, {
                    'estimate_rise': [], 'intercept_rise': [],
                    'realised_injected_variance': []})
                bf['estimate_rise'].append(
                    r['closest_bin_estimate'] - r0['closest_bin_estimate'])
                bf['intercept_rise'].append(
                    r['intercept_estimate'] - r0['intercept_estimate'])
                bf['realised_injected_variance'].append(
                    rec['realised_injected_variance'])
    out = {}
    for key, by_lvl in raw.items():
        for lvl, row in by_lvl.items():
            o = {'pair_set_reads_labels': row['pair_set_reads_labels'],
                 'folds': sorted(row['by_fold'])}
            for q in ('estimate_rise', 'intercept_rise',
                      'realised_injected_variance'):
                per_fold = []
                for f in o['folds']:
                    v = np.asarray(row['by_fold'][f][q], dtype=float)
                    v = v[np.isfinite(v)]
                    per_fold.append(float(v.mean()) if v.size else None)
                o[q] = fold_spread(per_fold)
            out.setdefault(key, {})[lvl] = o
    return out


def _paired(base, got, lower_is_better):
    """n30's pairing: one arm against another on the passes both scored."""
    shared = sorted(set(base) & set(got))
    gains = np.array([(base[k] - got[k]) if lower_is_better
                      else (got[k] - base[k]) for k in shared], dtype=float)
    keep = np.isfinite(gains)
    gains = gains[keep]
    shared = [k for k, ok in zip(shared, keep) if ok]
    scale = max(abs(float(np.mean(list(base.values())))) if base else 1.0, 1.0)
    tol = 1e-9 * scale
    by_fold = {}
    for k, g in zip(shared, gains):
        by_fold.setdefault(_fold_of(k), []).append(float(g))
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
        # Draws averaged within a fold, then mean and sd over folds.
        'gain_over_folds': fold_spread(
            [float(np.mean(v)) for _f, v in sorted(by_fold.items())]),
    }


def _tail_means(d):
    keys = ('tail_rmse_top', 'tail_rmse_bottom', 'calibration_slope')
    out = {}
    for k in keys:
        v = [t[k] for t in d.values() if t and np.isfinite(t.get(k, np.nan))]
        out[f'{k}_mean'] = float(np.mean(v)) if v else float('nan')
    return out


def _damaged_minus_clean(untouched, clean):
    """untouched minus clean per pass, averaged over draws per fold, then
    fold_spread. In the dataset's own metric; its sign reads with
    lower_is_better."""
    by_fold = {}
    for k in sorted(set(untouched) & set(clean)):
        d = untouched[k] - clean[k]
        if np.isfinite(d):
            by_fold.setdefault(_fold_of(k), []).append(float(d))
    return fold_spread([float(np.mean(v)) for _f, v in sorted(by_fold.items())])


def _summarise(acc, tails, models, metric_name, lower_is_better):
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'headline': ('passes won against untouched, paired per pass; a '
                        'pass is one (draw, fold). gain_over_folds averages '
                        'the draws of a fold, then mean and sd over folds.'),
           'per_level': {}}
    for lvl, by_space in acc.items():
        lrow = {}
        for space, by_arm in by_space.items():
            base = by_arm.get(ARM_TO_BEAT) or {}
            clean = by_arm.get('clean') or {}
            srow = {}
            for m in models:
                b = base.get(m) or {}
                if not b:
                    continue
                row = {'untouched_mean': float(np.mean(list(b.values()))),
                       'untouched_tails': _tail_means(
                           tails[lvl][space][ARM_TO_BEAT][m]),
                       'damaged_minus_clean': _damaged_minus_clean(
                           b, clean.get(m) or {}),
                       'per_arm': {}}
                for arm in by_arm:
                    if not arm.startswith(ARM_PREFIX):
                        continue
                    g = (by_arm.get(arm) or {}).get(m) or {}
                    if not g:
                        continue
                    # A18 (2026-10-08): untouched_mean averages every pass;
                    # these average only the passes this arm was refitted on,
                    # with the per-pass untouched scores beside the arm's.
                    _sh = sorted(k for k in set(b) & set(g)
                                 if np.isfinite(b[k]) and np.isfinite(g[k]))
                    row['per_arm'][arm] = {
                        'passes': _sh,
                        'untouched_per_pass': {k: b[k] for k in _sh},
                        'refit_per_pass': {k: g[k] for k in _sh},
                        'untouched_mean_same_passes': (
                            float(np.mean([b[k] for k in _sh])) if _sh
                            else float('nan')),
                        'refit_mean_same_passes': (
                            float(np.mean([g[k] for k in _sh])) if _sh
                            else float('nan')),
                        'passes_refitted': len(g),
                        'vs_untouched': _paired(b, g, lower_is_better),
                        'tails': _tail_means(tails[lvl][space][arm][m])}
                srow[m] = row
            lrow[space] = srow
        out['per_level'][lvl] = lrow
    return out


def _fmt_spread(fs):
    if not fs or fs.get('mean') is None:
        return 'none'
    if fs.get('sd') is None:
        return f"{fs['mean']:+.4f} ({fs.get('note')})"
    return f"{fs['mean']:+.4f} (sd {fs['sd']:.4f} over {fs['n_folds']} folds)"


def _print_summary(results, metric_name):
    print("\n" + "=" * 70)
    print("CHECK: rise of the closest-bin estimate over level 0, beside the "
          "realised injected variance (label units squared); settings whose "
          "pairs do not read the labels")
    for key, by_lvl in results['check'].items():
        for lvl in sorted(by_lvl, key=float):
            r = by_lvl[lvl]
            if r.get('pair_set_reads_labels'):
                continue
            print(f"  {key:44s} level {lvl:>5s}: rise "
                  f"{_fmt_spread(r['estimate_rise'])}, injected "
                  f"{_fmt_spread(r['realised_injected_variance'])}")
    print(f"\nSUMMARY ({metric_name}; paired per pass against untouched)")
    for lvl, lrow in results['summary']['per_level'].items():
        print(f"\n  level {lvl}:")
        for space, srow in lrow.items():
            for m, row in srow.items():
                print(f"    {space}/{m}: untouched {row['untouched_mean']:.4f}, "
                      f"damaged minus clean "
                      f"{_fmt_spread(row['damaged_minus_clean'])}")
                for arm, e in row['per_arm'].items():
                    v = e['vs_untouched']
                    print(f"      {arm:52s} won {v['passes_won']} of "
                          f"{v['n_passes_paired']} passes, gain over folds "
                          f"{_fmt_spread(v['gain_over_folds'])}")
    print("\n  The count of passes won is the headline. A setting is refitted "
          "only on the passes whose screen kept it, so passes differ by arm.")


def main():
    parser = create_experiment_parser(
        'N35: near-duplicate label disagreement, and downweighting it')
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated gaussian levels. Default: the '
                             "decided levels of the dataset's group")
    parser.add_argument('--draws', type=str, default=None,
                        help='Comma-separated injection replicates (default '
                             '0,1,2; level 0 runs the first only)')
    parser.add_argument('--skip-proximity', action='store_true',
                        help='Leave the forest_proximity setting out')
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
