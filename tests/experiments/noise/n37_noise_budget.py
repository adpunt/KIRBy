#!/usr/bin/env python3
"""N37: how much of the model's error is label noise, read without cleaning anything.

Paper: Noise.

THE QUESTION. The third research report (2026-09-25, "Dense Gaussian Noise vs
Sparse Gross Errors") says label noise can only be worth handling where it is a
large share of the model's error, and asks for that share to be measured before
any cleaning is tried. This file measures it on clean labels, per dataset, and
makes no change to any label. Nothing here is a pruning rule: the report's "stop
noise work where the share is under 0.1" is NOT applied; the share is written.

WHAT IS COMPUTED, per outer fold (noise_helpers.noise_outer_folds), on the
fold's training molecules as recorded:

  1. ONE NOISE ESTIMATE PER REPRESENTATION, side by side (Adelaide 2026-10-05,
     HANDOFF_EXPERIMENT_ALIGNMENT 2.3 item 19). The Gamma test (Stefansson,
     Koncar & Jones 1997): for each molecule, its k-th nearest neighbour,
     k = 1..K_MAX. delta_k is the mean squared distance to it, gamma_k the mean
     of half the squared label difference. A straight line through the K_MAX
     (delta, gamma) points; its intercept estimates the label-noise variance,
     in label units squared. It also holds real activity differences the
     features cannot resolve, so it is an upper bound. Computed in EVERY
     pruned representation of the dataset (DATASET_REPS[ds]['general'], or
     every loaded one where the dataset has no entry): Tanimoto when every
     value is 0 or 1, else Euclidean on the fold-scaled representation
     (kirby.scaling.FeatureScaler). Until 2026-10-05 there was one estimate,
     from the first model's rank-1 representation (rf's under the old model
     list); on herg_ki, mhg_gnn's would have halved every share (dirty-run
     report).
     A NEGATIVE intercept is not a variance. It is written as computed, with
     'negative': True, gets no share, and is left out of the summary mean
     (the fold is named under negative_folds). Until 2026-10-05 it was
     averaged in (flab_expression_koenig, 2 of 3 folds negative in the dirty
     run).

  2. THE CHECK. `gaussian` at the dataset's decided levels (noise grid,
     KNN_SEARCH_SPACE_RECORD 2026-10-05: 0, 0.5, 1.0, 1.5 on full_knn; 0, 1.0
     on fast_knn), 3 draws per non-zero level, level 0 once, injected into the
     training labels only. A representation's distances do not depend on the
     labels, so its intercept minus its level-0 intercept should equal the
     realised injected variance on that fold. Checked in every representation
     of item 1.

  3. THE MODEL'S ERROR. Per outer fold, the dataset's two decided models
     (DECISIONS.md Step 1), each on its 4 configurations (handoff R5: 2 base
     representations + 2 general hybrids, chosen by searches on that fold's
     training rows; noise_helpers.fold_configurations), so 8 fits per fold.
     Each fitted on the fold's training labels and scored on the held-out
     molecules: the domain metric and the mean squared error, both against
     the held-out labels as recorded. Written beside it: EACH representation's
     noise estimate divided by that MSE (share_by_representation), and the
     configuration's own space's intercept divided by it
     (share_this_configuration). On a dataset with repeat measurements, the
     same with n26's pooled within-molecule variance of the training rows
     (n26.direct_noise_variance).

  4. SIMEX (Cook & Stefanski 1994), as a sensitivity curve. Extra noise
     N(0, v) is added to the training labels at v = lambda * s2_grid, lambda in
     SIMEX_LAMBDAS, SIMEX_DRAWS draws each. Every configuration is refitted and
     scored on the ORIGINAL held-out labels. Straight-line and quadratic fits
     of MSE on v (v = 0 is the plain fit) are read at v = minus each noise
     estimate of item 1, and at minus the configuration's own: the MSE with
     that estimate of the noise gone. So one set of refits serves every
     estimate.
     DECIDED 2026-10-06 (Adelaide delegated): s2_grid, which sets how much
     noise the refits add, is the median of the fold's non-negative
     per-representation estimates; negative estimates are left out of it.
     Until 2026-10-05 it was the single reference estimate, which item 1 no
     longer has.
     ⛔ The hybrid features are built ONCE per fold from the clean training
     labels and held fixed across lambda. This curve therefore measures the
     model's sensitivity to label noise, not the feature choice's.

  5. RESIDUAL TAILS, on the out-of-fold residual of each (model,
     configuration): the held-out residuals of every fold put together, so
     every molecule appears once. SD divided by 1.4826 * MAD (about 1.0 for
     Gaussian residuals), excess kurtosis, the Student-t degrees of freedom by
     maximum likelihood (scipy.stats.t.fit), and the Hill tail index over the
     largest HILL_SHARE of |residual|. The first two are also written per
     fold. ⛔ base_1 is the model's rank-1 representation ON EACH FOLD, which
     can differ between folds; the names per fold are under per_fold.

  6. RESIDUAL SPREAD AGAINST THE PREDICTION. The squared out-of-fold residual
     regressed on the prediction: R^2. On a dataset with repeat measurements,
     a second fit with the molecule's measurement count and the SD of its
     measurements added. No test statistic and no p-value (2026-10-05: n*R^2,
     the Breusch-Pagan statistic, was removed with the other significance
     tests).

  7. THE NEIGHBOUR SETTINGS (KNN_SEARCH_SPACE_RECORD, Adelaide 2026-10-05).
     The Gamma intercept under every decided neighbour setting of the
     dataset's group (n07.sweep_settings: spaces, distances, no importance
     weighting) with K_MAX replaced by each swept neighbour count (5, 10, 20).
     The spaces include the top hybrid of the per-fold distance search
     (hybrid__distance1) and of the per-fold similarity search
     (hybrid_similarity); they are neighbour spaces only, no model is fitted
     on them. Scaffold-mates are kept in every neighbour list (axis 6).
     How much each neighbour counts (axis 5) does not apply: the Gamma test
     reads each neighbour's own distance and label gap and never averages
     neighbour labels. No screen: nothing is refitted per setting.

Usage:
    python n37_noise_budget.py --dataset logd --mode full
    python n37_noise_budget.py --dataset herg_ic50 --mode full --skip-simex
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import time
import zlib

import numpy as np
from scipy import stats as st

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories,
)
from experiments.dataset_policy import noise_groups_for
from experiments.decided_grid import decided_models
# CHAT 5 ALIGNMENT (2026-10-05, HANDOFF_EXPERIMENT_ALIGNMENT 2.4): folds, the
# per-fold configurations and the decided noise grid. They replace cv_split,
# get_best_search_for_downstream, pool_recipe_per_downstream_model,
# pool_recipes_by_family, noise_base_reps_by_model, base_rank_groups,
# group_models_by_recipe, noise_levels and noise_spec_for/inject_train_labels.
from experiments.noise.noise_helpers import (
    noise_outer_folds, fold_spread, fold_configurations, fold_model_inputs,
    configuration_record, inject_condition, decided_levels,
    dataset_knn_group, level_unit, CONFIGURATION_SPACES, draws_for,
)
from experiments.noise.n35_near_duplicate_disagreement import (
    is_binary, MAX_PAIR_ROWS,
)
from experiments.noise.n26_noise_variance_from_repeats import (
    measurements_by_molecule, direct_noise_variance,
)
from experiments.noise._ground_truth_providers import build_replicate_groups
# The per-fold checkpoint, cached fits and the neighbour spaces shared by
# n32-n41 (n32's shared block). n37 changes no label, so it writes no
# D-versus-D' rows: SIMEX adds noise to measure sensitivity, it mitigates
# nothing.
from experiments.noise.n32_iterative_and_dual_selection import (
    fit_predict_cached, ledger_signature, PassLedger, restart_flag,
    sweep_settings, fold_similarity, _nc, _ss,
)

EXP_ID = 'n37_noise_budget'
# Bumped 2026-10-05 (chat 5): it is in the result header, which the
# checkpoint signature hashes, so a checkpoint written by the whole-dataset
# version is set aside instead of read back.
RESULT_LAYOUT = 'fold_aligned_2026-10-05'
# The only injected condition, for the check. One of the six decided kinds.
CHECK_CONDITION = 'gaussian'
# 3 draws per non-zero level; level 0 is the labels as delivered, run once
# (noise grid decision, 2026-10-05). Was (0,) until then.
NOISE_DRAWS = (0, 1, 2)
# Neighbours in the per-representation Gamma test (item 1). The neighbour
# settings of item 7 use 5, 10 and 20 instead.
# DECIDED 2026-10-06 (Adelaide delegated): item 1's distance stays Tanimoto
# on 0/1 representations and Euclidean otherwise, as the Gamma test was
# published, although Euclidean is cut from the neighbour settings
# (KNN_SEARCH_SPACE_RECORD axis 2). The decided distances are all in item 7.
K_MAX = 10
# SIMEX: extra noise at lambda * s2_grid (module docstring, item 4).
SIMEX_LAMBDAS = (0.5, 1.0, 2.0)
# Draw ids, not a count. _local_run.py sets (0,) for the local debug run; that
# is where the dirty run's one SIMEX draw came from.
SIMEX_DRAWS = (0, 1, 2)
# Hill estimator over this share of the largest |residual|.
HILL_SHARE = 0.10
MAD_TO_SD = 1.4826


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def hill_index(r, share=HILL_SHARE):
    """Hill's tail index alpha over the largest `share` of |r|.

    H = mean over the top k of log(|r|_(i) / |r|_(k+1)); alpha = 1 / H. A
    Student-t with nu degrees of freedom has alpha = nu; a Gaussian's alpha
    grows with k and n rather than settling.
    """
    a = np.sort(np.abs(np.asarray(r, dtype=float)))[::-1]
    a = a[a > 0]
    k = int(np.ceil(share * a.size))
    if k < 2 or k >= a.size:
        return None
    h = float(np.mean(np.log(a[:k] / a[k])))
    return {'alpha': (1.0 / h) if h > 0 else None, 'k': k, 'n': int(a.size)}


def tail_diagnostics(r, fit_t=True):
    """SD / (1.4826 MAD), excess kurtosis, Student-t nu and Hill alpha of r."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 8:
        return {'n': int(r.size), 'too_few': True}
    mad = MAD_TO_SD * float(np.median(np.abs(r - np.median(r))))
    out = {
        'n': int(r.size),
        'sd': float(np.std(r)),
        'mad_sd': mad,
        'sd_over_mad_sd': (float(np.std(r)) / mad) if mad > 0 else None,
        'excess_kurtosis': float(st.kurtosis(r)),
        'hill': hill_index(r),
    }
    if fit_t:
        try:
            nu, loc, scale = st.t.fit(r)
            out.update({'student_t_nu': float(nu), 'student_t_loc': float(loc),
                        'student_t_scale': float(scale)})
        except Exception as exc:
            out['student_t_error'] = f'{type(exc).__name__}: {exc}'
    return out


def spread_regression(r, X_aux):
    """R^2 of r^2 on X_aux (with intercept). No test statistic, no p-value
    (n*R^2 removed 2026-10-05: it exists only to be read against a
    chi-squared table)."""
    r2 = np.asarray(r, dtype=float) ** 2
    A = np.column_stack([np.ones(r2.size), np.asarray(X_aux, dtype=float)])
    ok = np.all(np.isfinite(A), axis=1) & np.isfinite(r2)
    A, r2 = A[ok], r2[ok]
    if r2.size <= A.shape[1] + 1 or np.var(r2) == 0:
        return {'n': int(r2.size), 'too_few': True}
    coef, *_ = np.linalg.lstsq(A, r2, rcond=None)
    fit = A @ coef
    rsq = 1.0 - float(np.sum((r2 - fit) ** 2)) / float(np.sum((r2 - r2.mean()) ** 2))
    return {'n': int(r2.size), 'r_squared': rsq,
            'coefficients': [float(c) for c in coef]}


def simex_extrapolate(lams, mses):
    """Straight-line and quadratic fits of MSE on lambda, carried to lambda = -1.

    lams includes 0 (the plain fit). Returns the extrapolated MSE from each fit
    and the drop from the plain fit, in label units squared and as a share.
    Kept for _smoke_n37; run() reads the curve with simex_read_at.
    """
    x = np.asarray(lams, dtype=float)
    y = np.asarray(mses, dtype=float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    out = {'lambdas': x.tolist(), 'mse': y.tolist()}
    if x.size < 2:
        return out
    y0 = float(y[x == 0][0]) if np.any(x == 0) else float('nan')
    for name, deg in (('linear', 1), ('quadratic', 2)):
        if x.size <= deg:
            continue
        c = np.polyfit(x, y, deg)
        at = float(np.polyval(c, -1.0))
        out[name] = {'mse_at_minus_one': at, 'coefficients': c.tolist(),
                     'drop_from_plain': y0 - at,
                     'drop_share_of_plain': ((y0 - at) / y0) if y0 > 0 else None}
    return out


def simex_read_at(added_variance, mses, targets):
    """Straight-line and quadratic fits of MSE on the added noise variance,
    each read at v = -s2 for every s2 in `targets` ({name: s2}).

    added_variance includes 0 (the plain fit). For each fit and target: the
    MSE there, the drop from the plain fit (label units squared) and the drop
    as a share of the plain MSE."""
    x = np.asarray(added_variance, dtype=float)
    y = np.asarray(mses, dtype=float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    out = {'added_variance': x.tolist(), 'mse': y.tolist()}
    if x.size < 2:
        return out
    y0 = float(y[x == 0][0]) if np.any(x == 0) else float('nan')
    for name, deg in (('linear', 1), ('quadratic', 2)):
        if x.size <= deg:
            continue
        c = np.polyfit(x, y, deg)
        at = {}
        for tname, s2 in targets.items():
            m_at = float(np.polyval(c, -float(s2)))
            at[tname] = {'s2': float(s2), 'mse_with_noise_gone': m_at,
                         'drop_from_plain': y0 - m_at,
                         'drop_share_of_plain': ((y0 - m_at) / y0)
                         if y0 > 0 else None}
        out[name] = {'coefficients': c.tolist(), 'at': at}
    return out


def neighbour_table(dist_rows, n, k_max):
    """Each row's k_max nearest other rows: (idx, dist), both (n, k).

    dist_rows(a, b) returns the distances of rows a..b to every row, so no
    n-by-n matrix is held (similarity_space._chunk_rows sets the block). A
    row's own column is excluded; scaffold-mates and repeat measurements of
    the same molecule are not. Ties broken by row index
    (similarity_space._stable_topk)."""
    k = int(min(k_max, n - 1))
    idx = np.empty((n, max(k, 0)), dtype=np.int64)
    dist = np.empty((n, max(k, 0)), dtype=np.float64)
    if k < 1:
        return idx, dist
    step = _ss._chunk_rows(n)
    for a in range(0, n, step):
        b = min(n, a + step)
        # Distances, not features: float64 so the squared distances below
        # are summed at full precision.
        D = np.array(dist_rows(a, b), dtype=np.float64)
        D[np.arange(b - a), np.arange(a, b)] = np.inf
        D[np.isnan(D)] = np.inf
        idx[a:b], dist[a:b] = _ss._stable_topk(D, k,
                                                self_cols=np.arange(a, b))
    return idx, dist


def gamma_from_neighbours(idx, dist, y, k_max):
    """The Gamma test (noise_helpers.gamma_test's arithmetic) from a
    neighbour_table, at k = 1..k_max (k_max at most the table's width)."""
    y = np.asarray(y, dtype=float).ravel()
    k_max = int(min(k_max, idx.shape[1]))
    if k_max < 2:
        return {'unavailable': f'fewer than 2 neighbours per molecule '
                               f'({idx.shape[1]})'}
    d = dist[:, :k_max]
    if not np.all(np.isfinite(d)):
        return {'unavailable': (
            f'{int((~np.isfinite(d)).any(axis=1).sum())} molecules have fewer '
            f'than {k_max} neighbours at a defined distance')}
    dy2 = 0.5 * (y[idx[:, :k_max]] - y[:, None]) ** 2
    delta = np.mean(d ** 2, axis=0)
    gamma = np.mean(dy2, axis=0)
    B, A = np.polyfit(delta, gamma, 1)
    var_y = float(np.var(y))
    return {'intercept': float(A), 'slope': float(B), 'k_max': k_max,
            'delta': delta.tolist(), 'gamma': gamma.tolist(),
            'v_ratio': float(A / var_y) if var_y > 0 else None,
            'var_y': var_y, 'n': int(len(y)), 'negative': bool(A < 0)}


def _tanimoto_rows(B):
    """dist_rows for one minus Tanimoto on 0/1 rows (n35's
    tanimoto_distance_matrix, a block at a time). Two all-zero rows are at
    distance 0."""
    B = np.asarray(B, dtype=np.float32)
    cnt = B.sum(axis=1)

    def rows(a, b):
        inter = B[a:b] @ B.T
        union = cnt[a:b, None] + cnt[None, :] - inter
        with np.errstate(divide='ignore', invalid='ignore'):
            sim = np.where(union > 0, inter / union, 1.0)
        return 1.0 - sim
    return rows


def _euclidean_rows(X):
    from sklearn.metrics.pairwise import euclidean_distances
    X = np.asarray(X)
    return lambda a, b: euclidean_distances(X[a:b], X)


def _seed(*parts):
    return zlib.crc32('|'.join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def _molecule_count_and_spread(labels, rep_groups):
    """Per row: how many measurements its molecule has, and their SD (ddof 0)."""
    g = np.asarray(rep_groups)
    labels = np.asarray(labels, dtype=float)
    k = np.zeros(len(g))
    sd = np.zeros(len(g))
    by = {}
    for i, gi in enumerate(g):
        by.setdefault(gi, []).append(i)
    for rows in by.values():
        k[rows] = len(rows)
        sd[rows] = float(np.std(labels[rows]))
    return k, sd


def _check_passes(data, dataset_name, levels=None, draws=None):
    """[(level, draw)]: level 0 once, every other level once per draw."""
    levels = (decided_levels(data, dataset_name, CHECK_CONDITION)
              if levels is None else levels)
    # draws_for gives one draw under censoring; the check is gaussian, so
    # every draw runs.
    draws = tuple(draws_for(CHECK_CONDITION,
                            tuple(NOISE_DRAWS if draws is None else draws)))
    out = []
    for lv in levels:
        if float(lv) == 0.0:
            out.append((0.0, 0))
        else:
            out += [(float(lv), int(d)) for d in draws]
    return out


def estimate_representations(dataset_name, reps):
    """Every representation item 1 estimates the noise in: the dataset's
    pruned list (DATASET_REPS[ds]['general']) as loaded, or every loaded
    representation where the dataset has no entry (herg_ic50)."""
    from experiments.complexity.complexity_helpers import DATASET_REPS
    pruned = (DATASET_REPS.get(dataset_name) or {}).get('general')
    if not pruned:
        return list(reps)
    return [r for r in pruned if r in reps]


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run(data, config, output_dir, dataset_name, force_rerun=False, **kwargs):
    """N37: the label-noise share of the model's error, on clean labels.

    force_rerun (run_dataset.py --force-rerun, via run_all) sets aside the
    per-fold checkpoint and recomputes every fold.

    Per outer fold:
      1. noise_helpers.fold_configurations runs this fold's searches on its
         training rows (clean labels) and gives each decided model 2 base
         representations, 2 general hybrids and the neighbour-space recipes.
      2. The Gamma intercept in every pruned representation, and the check.
      3. fold_model_inputs builds each model's 4 configurations from the clean
         training labels; each is fitted, scored, and refitted for SIMEX.
      4. The Gamma intercept under every neighbour setting of each model.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT N37: NOISE BUDGET")
    print("=" * 70)

    # Labels may be float64; representations keep their own dtype (float32 on
    # load, handoff 2.3 item 18).
    labels = np.asarray(data['labels'], dtype=float)
    reps = data['reps']
    groups = data.get('groups')
    task_type = data['task_type']
    if task_type != 'regression':
        print(f"  SKIPPED: {dataset_name} is {task_type}. A noise variance in "
              f"label units is not defined for class labels.")
        return {'skipped': 'not_regression', 'task_type': task_type}
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)
    # Outer folds from config['n_splits'] (KIRBY_NOISE_N_FOLDS overrides), the
    # splits the per-fold searches are filed under. Until 2026-10-05:
    # cv_split with mode 'intermediary' at 1 fold and 'full' otherwise.
    folds, n_folds = noise_outer_folds(data, config)
    run_forest = not kwargs.get('skip_proximity', False)
    run_simex = not kwargs.get('skip_simex', False)
    simex_lambdas = tuple(float(x) for x in
                          (kwargs.get('simex_lambdas') or SIMEX_LAMBDAS))
    simex_draws = tuple(int(x) for x in
                        (SIMEX_DRAWS if kwargs.get('simex_draws') is None
                         else kwargs['simex_draws']))
    k_max = int(kwargs.get('k_max') or K_MAX)

    # The dataset's two decided models (DECISIONS.md Step 1, handoff R1).
    # Until 2026-10-05: the config's five plus rf and svm. A `models` kwarg
    # (a smoke) can only narrow them.
    decided = list(decided_models(dataset_name, 'general'))
    asked = kwargs.get('models')
    downstream_models = ([m for m in decided if m in asked] if asked
                         else list(decided))
    if asked and [m for m in asked if m not in decided]:
        print(f"  models {[m for m in asked if m not in decided]} dropped: "
              f"not decided for {dataset_name} ({decided})")
    # A model's size limit (_max_samples) is checked against the most
    # training rows any outer fold fits on, not the whole dataset's row count
    # (2026-10-05): on herg_ic50, 12,008 measurements dropped svm although
    # each fold trains on about 9,600 rows and the fold search runs svm.
    n_fit_rows = max((len(tr) for _, tr, _ in folds), default=len(labels))
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=n_fit_rows)
    models_unavailable = {
        m: (f'get_model_factories gave no factory at {n_fit_rows} training '
            f'rows (size limit or unknown name)')
        for m in downstream_models if m not in model_factories}
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n37: no decided model could be built. Decided {decided}, "
            f"unavailable {models_unavailable}.")

    # The neighbour settings (KNN_SEARCH_SPACE_RECORD, Adelaide 2026-10-05):
    # the dataset group's spaces and distances, no weighting, ks 5, 10, 20.
    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    knn_group = dataset_knn_group(data, dataset_name)
    n_molecules = len(set(smiles)) if has_smiles else len(labels)
    sim_settings = sweep_settings(dataset_name, n_molecules,
                                  has_smiles=has_smiles)
    from experiments.noise.n07_remediation import sweep_ks as _n07_sweep_ks
    sim_ks = tuple(_n07_sweep_ks())
    # A per-model space (base_*, augmented, hybrid) is scored per model; the
    # Tanimoto bits once per fold; forest proximity per configuration, since
    # its forest is grown on the configuration's matrix.
    per_model_settings = [s for s in sim_settings
                          if s.space in _ss.FEATURE_SPACES]
    forest_settings = ([s for s in sim_settings
                        if s.space == _ss.FOREST_SPACE] if run_forest else [])
    shared_settings = [s for s in sim_settings
                       if s.space not in _ss.FEATURE_SPACES
                       and s.space != _ss.FOREST_SPACE]

    est_reps = estimate_representations(dataset_name, reps)
    noise_groups = noise_groups_for(data)[0]
    rep_groups = build_replicate_groups(data)
    if rep_groups is not None:
        row_count, row_spread = _molecule_count_and_spread(labels, rep_groups)
    passes = _check_passes(data, dataset_name, levels=kwargs.get('levels'),
                           draws=kwargs.get('draws'))

    results = {
        'result_layout': RESULT_LAYOUT,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'check_condition': CHECK_CONDITION,
        'check_levels': sorted({l for l, _ in passes}),
        'check_passes': [[l, d] for l, d in passes],
        'dose_units': level_unit(CHECK_CONDITION),
        'knn_group': knn_group,
        'models': downstream_models,
        'models_decided': decided,
        'models_unavailable': models_unavailable,
        'configurations_per_model': list(CONFIGURATION_SPACES),
        'selection_scope': 'fold_train_rows',
        'selection_note': (
            'per outer fold and decided model: the 2 best base '
            'representations and the 2 best general hybrids from searches on '
            'that fold\'s training rows with the clean training labels '
            '(noise_helpers.fold_configurations); which ones, per fold, is '
            'under per_fold[i].models. A hybrid\'s training-row score is its '
            'score_mean there.'),
        'n_folds': n_folds,
        'folds_run': [int(i) for i, _, _ in folds],
        'estimate_representations': est_reps,
        'k_max': k_max,
        'simex_lambdas': list(simex_lambdas),
        'simex_draws': list(simex_draws),
        'hill_share': HILL_SHARE,
        'max_pair_rows': MAX_PAIR_ROWS,
        'has_repeat_measurements': rep_groups is not None,
        'gamma_note': (
            'intercept of gamma_k on delta_k, k = 1..k_max nearest neighbours; '
            'label units squared. An upper bound on the label-noise variance: '
            'it also holds activity differences the features cannot resolve. '
            'A negative intercept is not a variance: it is written with '
            'negative=True, gets no share, and is left out of the summary '
            'mean.'),
        'estimate_note': (
            'one noise estimate per representation (Adelaide 2026-10-05): '
            'noise_estimate_by_representation, Tanimoto on 0/1 '
            'representations, else Euclidean on the fold-scaled features. '
            'Every model and configuration has a share against each.'),
        'simex_note': (
            'extra noise N(0, v), v = lambda * s2_grid, s2_grid the median of '
            'the fold\'s non-negative per-representation intercepts; the MSE '
            'curve on v is read at v = -s2 for each representation\'s s2 and '
            'for the configuration\'s own. Scored on the original held-out '
            'labels; the hybrid features are built once per fold from the '
            'clean training labels and held fixed, so the curve is the model '
            'sensitivity, not the feature choice sensitivity.'),
        'spread_note': 'R^2 of r^2 on the inputs; no test statistic, no p-value.',
        'no_pruning_note': ('the report proposes stopping noise work where the '
                            'noise share is under 0.1; that rule is not applied '
                            'here, the share is written.'),
        'similarity_settings': [s.name for s in sim_settings],
        'similarity_ks': list(sim_ks),
        'similarity_note': (
            'gamma_sweep holds the Gamma intercept under each neighbour '
            'setting with k in place of k_max, keys <setting>|k<k>: per model '
            'for base, augmented and hybrid spaces (per_model/<model>), once '
            'per fold for the Tanimoto bits (shared), per configuration for '
            'forest proximity (forest/<model>/<configuration>). '
            'hybrid__distance1 and hybrid_similarity are neighbour spaces '
            'only. Scaffold-mates are kept in every list. How much each '
            'neighbour counts does not apply: the Gamma test averages no '
            'neighbour labels.'),
        'neighbour_screen': (
            'not run: no model is refitted per neighbour setting, so every '
            'setting is scored'),
    }
    if rep_groups is not None:
        results['repeat_noise_variance_whole_dataset'] = direct_noise_variance(
            measurements_by_molecule(labels, rep_groups))

    print(f"  {len(labels)} rows, {len(folds)} of {n_folds} folds, models "
          f"{downstream_models}, {knn_group}")
    print(f"  noise estimated in {len(est_reps)} representations; check "
          f"levels {results['check_levels']} ({len(passes)} passes); "
          f"{len(sim_settings)} neighbour settings x {len(sim_ks)} ks")

    per_fold = []
    # {model: {configuration: {'res': [...], 'pred': [...], 'rows': [...]}}}
    oof = {}
    errors = {}
    t0 = time.time()
    # The per-fold checkpoint (n32's PassLedger; no rows). The signature
    # hashes the header above, which carries RESULT_LAYOUT.
    ledger = PassLedger(output_dir, EXP_ID, dataset_name, kwargs.get('tag'),
                        signature=ledger_signature(
                            data, config, results,
                            [(tr, va) for _, tr, va in folds]),
                        restart=bool(force_rerun) or restart_flag(kwargs))
    if ledger.restored():
        st_ = ledger.restored()
        per_fold, oof, errors = st_['per_fold'], st_['oof'], st_['errors']
        print(f"  resumed {len(ledger.passes_done)} finished folds")

    def _sweep(dest, fsim, settings, model, X_model_tr, y_tr, n_tr, fold_i,
               tag):
        """Gamma intercepts of `settings` at every swept k into dest."""
        for setting in settings:
            try:
                fn, n, why = fsim.distance_blocks(setting, model, X_model_tr)
                if fn is None:
                    dest[setting.name] = {'unavailable': why}
                    continue
                idx, dist = neighbour_table(fn, n_tr, max(sim_ks))
                for k in sim_ks:
                    dest[f'{setting.name}|k{int(k)}'] = gamma_from_neighbours(
                        idx, dist, y_tr, int(k))
            except Exception as exc:
                errors[f'f{fold_i}|{tag}|{setting.name}'] = (
                    f'{type(exc).__name__}: {exc}')

    for fold_i, train_idx, val_idx in folds:
        if ledger.pass_done(f'f{fold_i}'):
            continue
        y_tr = labels[train_idx]
        y_va = labels[val_idx]
        g_tr = (np.asarray(groups)[train_idx] if groups is not None else None)
        n_tr = len(train_idx)
        frec = {'fold': int(fold_i), 'training_rows': int(n_tr),
                'held_out_rows': int(len(val_idx)), 'models': {},
                'noise_estimate_by_representation': {},
                'gamma_by_configuration': {}, 'gamma_sweep': {},
                'check': [], 'model_error': {}, 'simex': {},
                'tails_per_fold': {}}
        if rep_groups is not None:
            frec['repeat_noise_variance_training'] = direct_noise_variance(
                measurements_by_molecule(y_tr, np.asarray(rep_groups)[train_idx]))
        # DECIDED 2026-10-06 (Adelaide delegated): kiba and bindingdb_kd stay
        # refused. A fold over MAX_PAIR_ROWS training rows runs nothing, and
        # is checked before the per-fold searches so a refused fold runs
        # none. The distances below are built a block at a time, so memory
        # is not what limits them; the refusal is on run time only.
        if n_tr > MAX_PAIR_ROWS:
            frec['refused'] = (f'{n_tr} training rows is over MAX_PAIR_ROWS='
                               f'{MAX_PAIR_ROWS}; nothing was computed')
            per_fold.append(frec)
            ledger.finish_pass(f'f{fold_i}', {'per_fold': per_fold, 'oof': oof,
                                              'errors': errors})
            print(f"    fold {fold_i}: {frec['refused']}")
            continue

        # ── The per-fold searches (R2, R6), on this fold's training rows ──
        try:
            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=train_idx, n_folds=n_folds)
        except Exception as exc:
            errors[f'f{fold_i}|fold_configurations'] = (
                f'{type(exc).__name__}: {exc}')
            frec['error'] = (f'per-fold searches failed: '
                             f'{type(exc).__name__}: {exc}')
            per_fold.append(frec)
            print(f"    fold {fold_i}: {frec['error']}")
            continue

        # ── 1 and 2. One noise estimate per representation, and the check ──
        tables = {}
        for rn in est_reps:
            try:
                if is_binary(reps[rn]):
                    fn, dname = (_tanimoto_rows(np.asarray(reps[rn])[train_idx]),
                                 'tanimoto')
                else:
                    Xs_tr, _ = _nc.cached_scaled(dataset_name, reps[rn],
                                                 train_idx, val_idx)
                    fn, dname = (_euclidean_rows(Xs_tr),
                                 'euclidean_on_scaled_features')
                idx, dist = neighbour_table(fn, n_tr, k_max)
                g = gamma_from_neighbours(idx, dist, y_tr, k_max)
                g['distance'] = dname
                frec['noise_estimate_by_representation'][rn] = g
                if 'intercept' in g:
                    tables[rn] = (idx, dist, g['intercept'])
            except Exception as exc:
                errors[f'f{fold_i}|estimate|{rn}'] = (
                    f'{type(exc).__name__}: {exc}')
        for level, draw in passes:
            try:
                y_n, _m, _p = inject_condition(
                    data, dataset_name, CHECK_CONDITION, level, draw,
                    train_idx=train_idx, groups=noise_groups)
                y_n = np.asarray(y_n, dtype=float)
            except Exception as exc:
                errors[f'f{fold_i}|check|{level:g}|d{draw}'] = (
                    f'{type(exc).__name__}: {exc}')
                continue
            injected = float(np.mean((y_n - y_tr) ** 2))
            for rn, (idx, dist, _i0) in tables.items():
                gn = gamma_from_neighbours(idx, dist, y_n, k_max)
                frec['check'].append({
                    'representation': rn, 'level': level, 'draw': draw,
                    'intercept': gn.get('intercept'),
                    'realised_injected_variance': injected})
        estimates = {rn: t[2] for rn, t in tables.items()}
        usable = {rn: s2 for rn, s2 in estimates.items() if s2 >= 0}
        del tables

        # ── 3. The 8 configurations, from the clean training labels ──
        inputs = {}
        for m in downstream_models:
            entry = entries.get(m)
            rec = configuration_record(entry)
            rec['metric'] = metric_name
            rec['lower_is_better'] = bool(lower_is_better)
            ins, unav, names = fold_model_inputs(
                data, dataset_name, m, entry, train_idx=train_idx,
                val_idx=val_idx, y_train=y_tr, factory=model_factories[m],
                metric=metric)
            inputs[m] = ins
            if names:
                rec['configuration_names'] = dict(names)
            rec['n_features'] = {sp: int(np.shape(Xa)[1])
                                 for sp, (Xa, _Xb) in ins.items()}
            for sp in CONFIGURATION_SPACES:
                if sp not in ins:
                    why = unav.get(sp) or 'not built'
                    rec.setdefault('configurations_unavailable', {})[sp] = why
                    errors[f'f{fold_i}|{m}|{sp}|build'] = why
            frec['models'][m] = rec
            print(f"    {m}: base {rec.get('base')}, hybrids "
                  f"{[h.get('label') for h in rec.get('hybrids') or []]}")

        # The Gamma intercept in each configuration's own space: a base
        # configuration's is its representation's (item 1); a hybrid's is
        # Euclidean on the scaled hybrid matrix, computed once per distinct
        # matrix. Both biased low on a hybrid: it was built from these labels.
        _by_hash = {}
        for m in downstream_models:
            entry = entries.get(m) or {}
            for sp, (X_tr_s, _X_va_s) in inputs[m].items():
                if sp in ('base_1', 'base_2'):
                    i = int(sp[-1]) - 1
                    rn = (entry.get('base') or [None, None])[i]
                    g = frec['noise_estimate_by_representation'].get(rn)
                    frec['gamma_by_configuration'].setdefault(m, {})[sp] = (
                        {'same_as_representation': rn,
                         'intercept': (g or {}).get('intercept')})
                    continue
                h = _nc.array_hash(np.asarray(X_tr_s))
                if h not in _by_hash:
                    try:
                        idx, dist = neighbour_table(_euclidean_rows(X_tr_s),
                                                    n_tr, k_max)
                        _by_hash[h] = gamma_from_neighbours(idx, dist, y_tr,
                                                            k_max)
                    except Exception as exc:
                        _by_hash[h] = {'unavailable': (
                            f'{type(exc).__name__}: {exc}')}
                frec['gamma_by_configuration'].setdefault(m, {})[sp] = dict(
                    _by_hash[h], distance='euclidean_on_scaled_features')

        # ── 7. The neighbour settings: per model, shared, per configuration ──
        fsims = {}
        sweep = frec['gamma_sweep']
        # The shared settings (Tanimoto bits) do not depend on the model; they
        # are scored once per fold through the first model whose per-fold
        # search gave an entry (2026-10-05: before, only through the first
        # decided model, so a failed first search skipped them silently).
        shared_done = not shared_settings
        for m in downstream_models:
            entry = entries.get(m)
            if entry is None:
                sweep.setdefault('per_model', {})[m] = {
                    'unavailable': 'no per-fold search entry for this model'}
                continue
            fsim = fold_similarity(
                dataset_name, data, reps=reps, train_idx=train_idx,
                val_idx=val_idx, y_train=y_tr, groups_train=g_tr,
                base_by_model={m: list(entry.get('base') or [])},
                recipe_for_space_by_model={m: dict(entry.get('recipes') or {})},
                model_factories=model_factories, metric=metric,
                task_type=task_type)
            # The 4 configurations are the same matrices as the base and
            # general-hybrid neighbour spaces: handed over, not rebuilt.
            for sp, (Xa, Xb) in inputs[m].items():
                fsim._spaces[(m, sp)] = (Xa, Xb, None)
            for sp, why in (entry.get('unavailable') or {}).items():
                if sp not in inputs[m]:
                    fsim._spaces[(m, sp)] = (None, None, why)
                    fsim.unavailable[f'{m}|{sp}'] = why
            fsims[m] = fsim
            _sweep(sweep.setdefault('per_model', {}).setdefault(m, {}), fsim,
                   per_model_settings, m, None, y_tr, n_tr, fold_i,
                   f'per_model/{m}')
            if not shared_done:
                _sweep(sweep.setdefault('shared', {}), fsim, shared_settings,
                       m, None, y_tr, n_tr, fold_i, 'shared')
                sweep['shared_scored_through'] = m
                shared_done = True
            for sp, (X_tr_s, _X_va_s) in inputs[m].items():
                if forest_settings:
                    _sweep(sweep.setdefault('forest', {}).setdefault(m, {})
                           .setdefault(sp, {}), fsim, forest_settings, m,
                           X_tr_s, y_tr, n_tr, fold_i, f'forest/{m}/{sp}')
        if not shared_done:
            why = 'no decided model had a per-fold search entry in this fold'
            sweep['shared'] = {st.name: {'unavailable': why}
                               for st in shared_settings}
            errors[f'f{fold_i}|shared'] = why

        # ── 3 and 4. The model's error, the shares, SIMEX ──
        s2_grid = (float(np.median(list(usable.values()))) if usable
                   else float('nan'))
        frec['simex_s2_grid'] = s2_grid if np.isfinite(s2_grid) else None
        for m in downstream_models:
            fac = model_factories[m]
            for sp, (X_tr_s, X_va_s) in inputs[m].items():
                try:
                    pred = np.asarray(fit_predict_cached(
                        dataset_name, m, fac, X_tr_s, y_tr, X_va_s,
                        metric_name), dtype=float)
                except Exception as exc:
                    errors[f'f{fold_i}|{m}|{sp}|fit'] = (
                        f'{type(exc).__name__}: {exc}')
                    continue
                res = y_va - pred
                mse = float(np.mean(res ** 2))
                own = ((frec['gamma_by_configuration'].get(m) or {})
                       .get(sp) or {}).get('intercept')
                row = {
                    'held_out_score': float(metric_fn(y_va, pred)),
                    'mse': mse,
                    'share_by_representation': {
                        rn: (s2 / mse if s2 >= 0 and mse > 0 else None)
                        for rn, s2 in estimates.items()},
                    'share_this_configuration': (
                        own / mse if own is not None and own >= 0 and mse > 0
                        else None),
                }
                neg = sorted(rn for rn, s2 in estimates.items() if s2 < 0)
                if neg:
                    row['no_share_negative_intercept'] = neg
                if rep_groups is not None:
                    s2e = frec['repeat_noise_variance_training']['s2e']
                    row['repeat_variance_over_mse'] = (
                        s2e / mse if mse > 0 and np.isfinite(s2e) else None)
                frec['model_error'].setdefault(m, {})[sp] = row
                frec['tails_per_fold'].setdefault(m, {})[sp] = \
                    tail_diagnostics(res, fit_t=False)
                o = oof.setdefault(m, {}).setdefault(
                    sp, {'res': [], 'pred': [], 'rows': []})
                o['res'].append(res)
                o['pred'].append(pred)
                o['rows'].append(np.asarray(val_idx))

                if not run_simex:
                    continue
                if not (np.isfinite(s2_grid) and s2_grid > 0):
                    frec['simex'].setdefault(m, {})[sp] = {
                        'not_run': 'no representation gave a positive Gamma '
                                   'intercept on this fold'}
                    continue
                curve_v, curve_m, curve_metric = [0.0], [mse], [
                    row['held_out_score']]
                per_draw = {}
                for lam in simex_lambdas:
                    ms, mets = [], []
                    for d in simex_draws:
                        rng = np.random.RandomState(_seed(
                            dataset_name, fold_i, lam, d))
                        y_x = y_tr + rng.normal(size=n_tr) * np.sqrt(
                            lam * s2_grid)
                        try:
                            p_x = np.asarray(fit_predict_cached(
                                dataset_name, m, fac, X_tr_s, y_x, X_va_s,
                                metric_name), dtype=float)
                        except Exception as exc:
                            errors[f'f{fold_i}|{m}|{sp}|simex|{lam:g}|d{d}'] = (
                                f'{type(exc).__name__}: {exc}')
                            continue
                        ms.append(float(np.mean((y_va - p_x) ** 2)))
                        mets.append(float(metric_fn(y_va, p_x)))
                    per_draw[f'{lam:g}'] = {'mse': ms, 'held_out_score': mets}
                    if ms:
                        curve_v.append(lam * s2_grid)
                        curve_m.append(float(np.mean(ms)))
                        curve_metric.append(float(np.mean(mets)))
                targets = {f'representation:{rn}': s2
                           for rn, s2 in usable.items()}
                if own is not None and own >= 0:
                    targets['this_configuration'] = own
                ex = simex_read_at(curve_v, curve_m, targets)
                ex['held_out_score_curve'] = curve_metric
                ex['per_draw'] = per_draw
                ex['s2_grid'] = s2_grid
                frec['simex'].setdefault(m, {})[sp] = ex

        per_fold.append(frec)
        ledger.finish_pass(f'f{fold_i}', {'per_fold': per_fold, 'oof': oof,
                                          'errors': errors})
        _est = ', '.join(f"{rn} {s2:.4f}" for rn, s2 in estimates.items())
        print(f"    fold {fold_i}: {n_tr} training rows, Gamma intercepts "
              f"{_est} [{time.time() - t0:.0f}s]")

    results['per_fold'] = convert_numpy(per_fold)
    results['checkpoint'] = ledger.describe()

    # Out-of-fold residuals: every held-out molecule once.
    oof_block = {}
    for mname, by_sp in oof.items():
        for sp, o in by_sp.items():
            res = np.concatenate([np.asarray(x) for x in o['res']])
            pred = np.concatenate([np.asarray(x) for x in o['pred']])
            rows = np.concatenate([np.asarray(x) for x in o['rows']])
            rec = {'tails': tail_diagnostics(res),
                   'spread_on_prediction': spread_regression(res, pred[:, None]),
                   'mse': float(np.mean(res ** 2))}
            if rep_groups is not None:
                rec['spread_on_prediction_count_and_spread'] = spread_regression(
                    res, np.column_stack([pred, row_count[rows],
                                          row_spread[rows]]))
            oof_block.setdefault(mname, {})[sp] = rec
    results['out_of_fold'] = convert_numpy(oof_block)
    if errors:
        results['_errors'] = errors
    results['check'] = convert_numpy(_check(per_fold))
    results['summary'] = convert_numpy(_summarise(per_fold, oof_block))
    _print_summary(results, metric_name)
    return results


def _check(per_fold):
    """Per representation and level: the intercept's rise over level 0 beside
    the realised injected variance, per fold (mean over draws), then the
    spread over folds."""
    by = {}
    for f in per_fold:
        rows = f.get('check') or []
        ref = {r['representation']: r['intercept'] for r in rows
               if r['level'] == 0.0}
        for r in rows:
            if r['level'] == 0.0:
                continue
            r0 = ref.get(r['representation'])
            if r0 is None or r['intercept'] is None:
                continue
            c = (by.setdefault(r['representation'], {})
                 .setdefault(f"{r['level']:g}", {})
                 .setdefault(f['fold'], {'rise': [], 'injected': []}))
            c['rise'].append(r['intercept'] - r0)
            c['injected'].append(r['realised_injected_variance'])
    out = {}
    for rn, by_l in by.items():
        for lv, by_f in by_l.items():
            folds = sorted(by_f)
            rise = [float(np.mean(by_f[i]['rise'])) for i in folds]
            inj = [float(np.mean(by_f[i]['injected'])) for i in folds]
            out.setdefault(rn, {})[lv] = {
                'folds': folds,
                'draws_per_fold': [len(by_f[i]['rise']) for i in folds],
                'intercept_rise': fold_spread(rise),
                'realised_injected_variance': fold_spread(inj)}
    return out


def _estimate_spread(values_by_fold):
    """fold_spread over the non-negative intercepts, with the negative folds
    named and their values kept."""
    ok = {f: v for f, v in values_by_fold.items() if v is not None and v >= 0}
    neg = {f: v for f, v in values_by_fold.items() if v is not None and v < 0}
    out = fold_spread(list(ok.values()))
    out['folds'] = sorted(ok)
    out['negative_folds'] = sorted(neg)
    out['negative_values'] = [neg[f] for f in sorted(neg)]
    return out


def _summarise(per_fold, oof_block):
    """Per representation, model and configuration: the fold values side by
    side and their spread (fold_spread; no significance test)."""
    est, share, own, simex = {}, {}, {}, {}
    for f in per_fold:
        if f.get('refused') or f.get('error'):
            continue
        fi = f['fold']
        for rn, g in (f.get('noise_estimate_by_representation') or {}).items():
            if isinstance(g, dict) and 'intercept' in g:
                est.setdefault(rn, {})[fi] = g['intercept']
        for m, by_sp in (f.get('model_error') or {}).items():
            for sp, row in by_sp.items():
                for rn, v in (row.get('share_by_representation') or {}).items():
                    (share.setdefault(m, {}).setdefault(sp, {})
                     .setdefault(rn, []).append(v))
                own.setdefault(m, {}).setdefault(sp, []).append(
                    row.get('share_this_configuration'))
        for m, by_sp in (f.get('simex') or {}).items():
            for sp, ex in by_sp.items():
                for fit in ('linear', 'quadratic'):
                    for tname, at in ((ex.get(fit) or {}).get('at') or {}).items():
                        (simex.setdefault(m, {}).setdefault(sp, {})
                         .setdefault(tname, {}).setdefault(fit, [])
                         .append(at.get('drop_share_of_plain')))
    out = {
        'noise_estimate_by_representation': {
            rn: _estimate_spread(v) for rn, v in est.items()},
        'share_by_representation': {
            m: {sp: {rn: fold_spread(v) for rn, v in by_rn.items()}
                for sp, by_rn in by_sp.items()}
            for m, by_sp in share.items()},
        'share_this_configuration': {
            m: {sp: fold_spread(v) for sp, v in by_sp.items()}
            for m, by_sp in own.items()},
        'simex_drop_share': {
            m: {sp: {t: {fit: fold_spread(v) for fit, v in by_fit.items()}
                     for t, by_fit in by_t.items()}
                for sp, by_t in by_sp.items()}
            for m, by_sp in simex.items()},
        'tails': {},
    }
    for m, by_sp in oof_block.items():
        for sp, rec in by_sp.items():
            t = rec['tails']
            row = {k: t.get(k) for k in ('sd_over_mad_sd', 'excess_kurtosis',
                                         'student_t_nu')}
            row['hill_alpha'] = (t.get('hill') or {}).get('alpha')
            row['spread_r_squared'] = rec['spread_on_prediction'].get('r_squared')
            out['tails'].setdefault(m, {})[sp] = row
    return out


def _fmt(x, p=4):
    return 'n/a' if x is None or not np.isfinite(x) else f'{x:.{p}f}'


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print("NOISE ESTIMATE per representation: Gamma intercept, label units "
          "squared (upper bound on the noise variance). Mean over the folds "
          "with a non-negative intercept, then the fold SD:")
    for rn, g in s['noise_estimate_by_representation'].items():
        neg = (f", negative on folds {g['negative_folds']}"
               if g['negative_folds'] else '')
        print(f"  {rn:24s} {_fmt(g['mean'])} sd {_fmt(g['sd'])} "
              f"({g['n_folds']} folds{neg})")
    print("\nCHECK: rise of each representation's intercept over level 0, "
          "beside the realised injected variance (label units squared, mean "
          "over folds):")
    for rn, by_l in results['check'].items():
        for lvl in sorted(by_l, key=float):
            r = by_l[lvl]
            print(f"  {rn:24s} level {lvl:>4s}: rise "
                  f"{_fmt(r['intercept_rise']['mean'])}, injected "
                  f"{_fmt(r['realised_injected_variance']['mean'])}")
    reps = list(s['noise_estimate_by_representation'])
    print("\nSHARE of held-out MSE: one row per model and configuration, one "
          "number per representation's noise estimate (mean over folds):")
    print("  " + " " * 30 + " ".join(f"{rn[:10]:>10s}" for rn in reps))
    for m, by_sp in s['share_by_representation'].items():
        for sp, by_rn in by_sp.items():
            vals = " ".join(f"{_fmt((by_rn.get(rn) or {}).get('mean'), 3):>10s}"
                            for rn in reps)
            print(f"  {m + '/' + sp:30s}{vals}")
    if s['simex_drop_share']:
        print("\nSIMEX: share of held-out MSE gone with the configuration's own "
              "estimate removed (linear, quadratic), mean over folds:")
        for m, by_sp in s['simex_drop_share'].items():
            for sp, by_t in by_sp.items():
                v = by_t.get('this_configuration') or {}
                print(f"  {m}/{sp}: {_fmt((v.get('linear') or {}).get('mean'), 3)}, "
                      f"{_fmt((v.get('quadratic') or {}).get('mean'), 3)}")
    print("\nOUT-OF-FOLD RESIDUALS: SD/MAD-SD, excess kurtosis, t nu, Hill "
          "alpha, R^2 of r^2 on prediction")
    for m, by_sp in s['tails'].items():
        for sp, t in by_sp.items():
            print(f"  {m}/{sp}: {_fmt(t['sd_over_mad_sd'], 2)}, "
                  f"{_fmt(t['excess_kurtosis'], 2)}, "
                  f"{_fmt(t['student_t_nu'], 1)}, {_fmt(t['hill_alpha'], 2)}, "
                  f"{_fmt(t['spread_r_squared'], 3)}")


def main():
    parser = create_experiment_parser(
        'N37: the label-noise share of the model error, on clean labels')
    parser.add_argument('--levels', type=str, default=None,
                        help="Comma-separated gaussian levels for the check. "
                             "Default: the dataset's decided levels")
    parser.add_argument('--draws', type=str, default=None,
                        help='Comma-separated draw ids for each non-zero check '
                             'level (default 0,1,2; level 0 runs once)')
    parser.add_argument('--simex-lambdas', type=str, default=None,
                        help='Comma-separated SIMEX multipliers (default 0.5,1,2)')
    parser.add_argument('--simex-draws', type=str, default=None,
                        help='Comma-separated SIMEX draw IDS, not a count '
                             '(default 0,1,2). Use --skip-simex to run none')
    parser.add_argument('--k-max', type=int, default=None,
                        help=f'Neighbours in the per-representation Gamma test '
                             f'(default {K_MAX})')
    parser.add_argument('--skip-proximity', action='store_true',
                        help='Do not compute the forest-proximity neighbour '
                             'settings')
    parser.add_argument('--skip-simex', action='store_true',
                        help='Do not run the SIMEX curve')
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
    if args.simex_lambdas:
        kw['simex_lambdas'] = [float(x) for x in args.simex_lambdas.split(',')]
    if args.simex_draws:
        kw['simex_draws'] = [int(x) for x in args.simex_draws.split(',')]
    if args.k_max:
        kw['k_max'] = args.k_max
    if args.skip_proximity:
        kw['skip_proximity'] = True
    if args.skip_simex:
        kw['skip_simex'] = True
    if args.tag:
        kw['tag'] = args.tag

    res = run(data, config, args.output, args.dataset, **kw)
    save_experiment_results(res, args.output, EXP_ID, args.dataset,
                            filename_suffix=args.tag or None)


if __name__ == '__main__':
    main()
