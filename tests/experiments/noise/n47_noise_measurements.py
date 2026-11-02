#!/usr/bin/env python3
"""N47, noise measurements: numbers about the noise and the data that train no
cleanup.

HANDOFF_NOISE_RESTRUCTURE.md, section 2 (working name MS) and section 5.
Replaces n01, n14, n16, n27, n37, and c10 (c10 moved from the complexity suite
to the noise suite on 2026-09-12). Written 2026-11-02. The old files stay where
they are. Every computation here is imported from them; each piece names the
function it comes from. Where an old function was nested inside another and
could not be imported, it is copied here and says so.

TWO KINDS OF RESULT FILE PER DATASET.

  <dataset>__labels_as_delivered.json
      Measurements that add no noise, or add their own. Written once, before
      the condition sweep.
        n01  the cliff reference per domain, and held-out error on cliff
             against non-cliff molecules.
        n14  the cliff list, prevalence, geometry, stage-0 diagnostics and
             feasibility flags.
        n37  the Gamma test per representation, the check under gaussian,
             each configuration's error with the noise share, SIMEX, residual
             tails, the Gamma test under every neighbour setting.
        n27  the removal cost curve and the relabelled part.

  <dataset>__<condition>.json
      Measurements that read one damage kind. One file per kind, through
      noise_helpers.sweep_conditions.
        c10  each configuration's held-out score under damaged labels (the
             score drop), feature stability under the first kind swept, and
             whether the searched recipe moves under damage.
        n27  the injected part, under gaussian only: how much of the injected
             label variance reaches the held-out error. Read from the c10 fits
             under gaussian, which are the same fits (same labels, same
             features, same model, through noise_cache).
        n16  rank agreement of feature importance against level 0, the two
             level-0 floors, turnover in the top set.

ROWS (handoff section 11). Every part that scores a downstream fit writes
d_vs_dprime rows through n07.RowSink, so all of ALL_METRICS are saved:
  c10_score_drop      D = the damaged training labels, nothing cleaned, so
                      score_Dprime is nan by design. score_D is the fit on
                      the damaged labels, score_clean the fit on the correct
                      labels in the same features.
  n27_removal_cost    D = every training molecule at its correct label; D' =
                      a random share of them. score_D equals score_clean.
  n27_move_toward_oof D' = the correct labels moved toward the out-of-fold
                      prediction. Its random and shuffled twins run under
                      KIRBY_NOISE_CONTROLS=1 (n07.dprime_rows).
  n37_model_error     D = the labels as delivered, nothing cleaned.
  n37_simex           D = the labels as delivered plus N(0, v) extra noise,
                      nothing cleaned; score_clean is the fit on the labels as
                      delivered.
The pure measurements (cliff counts, Gamma intercepts, rank agreement, tails)
go in the result json per unit. Two fitted parts write no rows, and why:
  n01's cliff-stratified error is scored on the cliff and non-cliff halves of
  the held-out fold. d_vs_dprime scores the whole held-out fold, so a row
  cannot hold it. It stays in the dataset's own metric, as n01 wrote it.
  c10's recipe-movement part runs searches, not fits.

THE NOISE AXIS. noise_helpers.decided_conditions, decided_levels,
inject_condition and draws_for: level 0 first and once, 3 draws per non-zero
level, 1 under censoring. Folds: noise_outer_folds, all 5 (author,
2026-10-30). Per fold and decided model, the 4 configurations of
fold_configurations (base_1, base_2, hybrid, hybrid__general2).

WHAT EACH OLD FILE ASKED NOT TO LOSE (handoff section 5), and where it is:
  n01  the circularity flag: `cliff_reference_independent` is written by the
       cliff reference and by the stratified part. False means the cliffs were
       found in the representations being graded.
  n14  the feasibility flags: n14_landscape()['feasibility'].
  n16  the two level-0 floors: n16's own `floor_seed` cells (level 0 scored a
       second time with another scorer seed) and the across-fold Nogueira
       stability at level 0, both inside n16's record.
  n27  the removal cost curve: shares 0.25, 0.5, 0.7, 0.8, 0.9 of the training
       molecules, 20 draws each, every label correct, paired against all of
       them. It is also the control of handoff section 3.2: a deletion
       cleanup in n44 is read against the same share here, matched on
       dataset, model, configuration and fold.
  n37  negative Gamma intercepts are written with negative=True, get no share,
       and are left out of the summary mean (n37._summarise).
  c10  no treatment arm: c10 is a measurement, so its rows carry method
       c10_score_drop and method_setting no_treatment.

NOT CARRIED FROM c10, and why: its module-level check that the noiseInject
package is installed. c10 skipped the whole dataset without it, but c10's
damage comes from noise_helpers.inject_condition, which does not import
noiseInject.

RESUME. RowSink skips every row already on disk. Each finished unit is in a
FoldCheckpoint (.units.jsonl) under the run's fingerprint. n01, n14 and n16
keep their own checkpoints under <output>/n47_noise_measurements/parts/.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments._shared import (
    get_domain_metric, get_model_factories, input_fingerprint, convert_numpy,
    save_experiment_results, fingerprint_mismatch, experiment_result_is_empty,
    hybrid_feature_signature,
)
from experiments.decided_grid import decided_models
from experiments.dataset_policy import noise_groups_for
from experiments.noise.noise_helpers import (
    sweep_conditions, conditions_for_task, noise_outer_folds, fold_spread,
    fold_configurations, fold_model_inputs, configuration_record,
    decided_levels, inject_condition, draws_for, level_unit,
    dataset_knn_group, recipe_is_geometry, CONFIGURATION_SPACES,
)
from kirby.noise_spec import GROUPED_STRATEGIES

from experiments.noise import noise_cache as nc
from experiments.noise import n07_remediation as n07
from experiments.noise import n25_cliff_exemption as n25
from experiments.noise.d_vs_dprime import score_d_dprime, rows_for

EXP_ID = 'n47_noise_measurements'
RUN_ID = 'n47'
RESULT_LAYOUT = 'n47_2026-11-02'
CLEAN_SUFFIX = 'labels_as_delivered'

# c10's noise seed (c10.run noise_seed=42) and n27's (inject_condition's
# default base_seed=42) are the same, so the gaussian fits are shared.
NOISE_SEED = 42
NOISE_DRAWS = (0, 1, 2)


def _parts_dir(output_dir):
    """Where n01, n14 and n16 keep their own checkpoints and files."""
    return (None if output_dir is None
            else os.path.join(str(output_dir), EXP_ID, 'parts'))


def _identity(dataset_name, method, setting, space, condition, level, draw,
              fold):
    return n07.row_identity(
        dataset_name=dataset_name, experiment=RUN_ID, method=method,
        method_setting=setting, similarity_setting=n07.NO_NEIGHBOURS,
        model_input=space, noise_condition=condition, level=level, draw=draw,
        fold=fold)


def _mse(all_scores):
    """Held-out mean squared error from a row's all_* scores (rmse squared),
    in label units squared, or None."""
    r = (all_scores or {}).get('rmse')
    if r is None or not np.isfinite(r):
        return None
    return float(r) ** 2


def score_and_write(sink, *, dataset_name, cleaned, X_tr, y_D, X_va, y_va,
                    y_clean, model, metric, task_type, identity, twins=False):
    """score_D, score_Dprime and score_clean of one model, the row written.

    d_vs_dprime.score_d_dprime and rows_for, which n07.dprime_rows calls;
    called here directly so the scores come back on a resumed run too (every
    fit is in noise_cache). twins=True also runs n07.dprime_rows for the
    random and shuffled twins under KIRBY_NOISE_CONTROLS=1.
    """
    sc = score_d_dprime(dataset_name, cleaned, X_tr, y_D, X_va, y_va,
                        metric=metric, task_type=task_type,
                        y_tr_clean=y_clean, models=(model,))
    if sink is not None and sink.enabled:
        sink.write(rows_for(sc, metric=metric, **identity))
        if twins and cleaned is not None:
            n07.dprime_rows(sink, dataset_name=dataset_name, cleaned=cleaned,
                            X_tr=X_tr, y_D=y_D, X_va=X_va, y_va=y_va,
                            metric=metric, task_type=task_type,
                            y_clean=y_clean, models=(model,),
                            identity=identity)
    return sc.get(model)


# ===========================================================================
# n01: the cliff reference per domain
# ===========================================================================

def cliff_reference(data, labels, is_clf):
    """Which molecules sit in an activity-cliff pair, on every molecule at
    the labels as delivered.

    n25.cliff_mask_for_dataset, which is n01's reference rule: ECFP4 Tanimoto
    for a molecular set, heavy-chain edit distance for an antibody set,
    reactant and product bits for a reaction, the drug within one protein for
    a drug-target row; Euclidean on every representation glued together when
    none of those exists. `cliff_reference_independent` is False only in that
    last case, where the cliffs are a function of the representations being
    graded and any comparison between representations on them is circular.
    """
    mask, info = n25.cliff_mask_for_dataset(data, np.asarray(labels), is_clf)
    info = dict(info)
    info.pop('similarity_matrix', None)
    return {'cliff_reference_independent': bool(
                info.get('reference_independent', True)),
            'n_cliff_molecules': int(np.asarray(mask).sum()),
            'n_molecules': int(len(mask)),
            'info': convert_numpy(info),
            'source': 'n25.cliff_mask_for_dataset (n01 rule)'}


def n01_part(data, config, output_dir, dataset_name, *, sim_threshold=0.55,
             activity_k=2.0):
    """n01 whole: the reference above, then n01._run_cliff_stratified
    (per fold, the two decided models on their 4 configurations, held-out
    error on cliff against non-cliff under the reference cliffs and under
    every neighbour setting)."""
    from experiments.noise import n01_activity_cliff_framework as n01
    labels = np.asarray(data['labels'])
    is_clf = data['task_type'] == 'classification'
    out = {'sim_threshold': sim_threshold,
           'activity_k_in_median_absolute_deviations': activity_k,
           'similar_percentile': n01.SIMILAR_PERCENTILE}
    try:
        out['cliff_reference'] = cliff_reference(data, labels, is_clf)
    except Exception as exc:
        out['cliff_reference'] = {'error': f'{type(exc).__name__}: {exc}'}
    try:
        out['stratified'] = n01._run_cliff_stratified(
            data['reps'], labels, data.get('groups'), config, data,
            sim_threshold, activity_k, smiles=data.get('smiles'),
            is_clf=is_clf, dataset_name=dataset_name,
            output_dir=_parts_dir(output_dir))
    except Exception as exc:
        out['stratified'] = {'error': f'{type(exc).__name__}: {exc}'}
    return out


# ===========================================================================
# n14: cliff list, prevalence, geometry, stage-0 diagnostics, feasibility
# ===========================================================================

def n14_landscape(data, dataset_name, config, output_dir, *,
                  with_similarity=True):
    """n14.run's record, from n14's own functions.

    The cliff list reads data['mmp_pairs'] when the loader gives them
    (n14._build_ac_reference), else ECFP4 Tanimoto at 0.55 with a 2-log-unit
    potency gap. with_similarity=False leaves out the per-fold geometry under
    every neighbour setting (n14._cliff_geometry_by_similarity), which needs
    the per-fold searches; only the smoke passes it.
    """
    from experiments.noise import n14_activity_cliff_landscape as n14
    from experiments._shared import cv_split, LARGE_N_SKIP_THRESHOLD, REP_DTYPE
    from experiments.noise.noise_helpers import noise_n_folds
    from kirby.utils.noisy_labels import from_linear, stage0_diagnostics

    domain = data.get('domain', 'molecular')
    task_type = data['task_type']
    labels = np.asarray(data['labels'], dtype=float).ravel()
    groups = data.get('groups')
    n = len(labels)
    # n14's scope guards, unchanged.
    if task_type == 'classification':
        return {'skipped': True, 'reason': 'classification out of scope',
                'policy': 'regression_only', 'domain': domain, 'n_samples': n}
    if domain != 'molecular':
        return {'skipped': True, 'reason': f'non-molecular domain ({domain})',
                'policy': 'molecular_only', 'domain': domain, 'n_samples': n}
    if n > LARGE_N_SKIP_THRESHOLD:
        return {'skipped': True, 'reason': f'large-n OOM guard n={n}',
                'policy': 'large_n_oom', 'domain': domain, 'n_samples': n}

    molecule_of_row = n14._molecule_of_row(data, dataset_name, n)
    adj, provenance = n14._build_ac_reference(data, dataset_name, labels,
                                              molecule_of_row)
    geometry, _cliff_mask = n14._geometry(adj)
    n_cliffs = geometry['n_cliff_compounds']
    label_mad = float(np.median(np.abs(labels - np.median(labels))))
    label_context = {
        'label_mad': label_mad, 'label_std': float(labels.std()),
        'label_range': [float(labels.min()), float(labels.max())],
        'potency_threshold_in_mad_units': (
            float(n14.AC_POTENCY_THRESHOLD / label_mad) if label_mad > 0
            else None),
        'note': ('the 2-log threshold is literal 100-fold potency only for '
                 'genuine activity endpoints (pIC50/pKi/pKd); for other '
                 'regression targets it is a fixed 2-unit label gap.')}

    geometry_by_similarity, sim_meta = None, {}
    if with_similarity:
        try:
            geometry_by_similarity, sim_meta = n14._cliff_geometry_by_similarity(
                data, dataset_name, labels, output_dir, config,
                molecule_of_row=molecule_of_row)
        except Exception as exc:
            geometry_by_similarity = {'error': f'{type(exc).__name__}: {exc}'}
    else:
        geometry_by_similarity = {'not_run': 'with_similarity=False'}

    rep_groups = n14.build_replicate_groups(data)
    has_replicates, n_replicate_samples = False, 0
    if rep_groups is not None:
        _stds, has_rep = n14.compute_replicate_stds(labels, rep_groups)
        has_replicates = bool(has_rep.any())
        n_replicate_samples = int(has_rep.sum())
    meta = data.get('metadata') or {}
    has_multi_assay = bool(meta.get('assay_ids') is not None or
                           meta.get('n_assays', 0) and meta.get('n_assays', 0) > 1)
    feasibility = {
        'n_high_conf_cliffs': int(n_cliffs),
        'pu_calibration_feasible': bool(n_cliffs >= n14.PU_MIN_CLIFFS),
        'pu_min_cliffs': n14.PU_MIN_CLIFFS,
        'has_replicates': has_replicates,
        'n_replicate_samples': n_replicate_samples,
        'has_multi_assay': has_multi_assay}

    reps = data.get('reps') or {}
    stage0, stage0_error = None, None
    if reps and groups is not None:
        try:
            X_concat = np.nan_to_num(np.hstack([
                np.asarray(reps[k]).astype(REP_DTYPE, copy=False)
                for k in sorted(reps)]), nan=0.0)
            y_pos, params = from_linear(labels)
            n_splits = max(2, noise_n_folds(config))

            def _compute():
                return n14._rescale_stage0(stage0_diagnostics(
                    X_concat, y_pos, groups, n_splits=n_splits,
                    splitter=cv_split), params)
            if output_dir is None:
                stage0 = _compute()
            else:
                ck = n07.FoldCheckpoint(
                    output_dir, 'n14_activity_cliff_landscape', dataset_name,
                    tag='stage0', fingerprint={
                        'layout': n14.RESULT_LAYOUT,
                        'input': input_fingerprint(data, config),
                        'n_splits': int(n_splits),
                        'width': int(X_concat.shape[1])})
                stage0 = ck.cached(['stage0'], _compute)
            stage0['n_features'] = int(X_concat.shape[1])
        except Exception as exc:
            stage0_error = f'{type(exc).__name__}: {exc}'
    else:
        stage0_error = 'no reps or no groups available'

    decision_hint = {
        'cliffs_present': bool(n_cliffs > 0),
        'cliff_prevalence_pct': geometry['pct_cliff_compounds'],
        'geometry_lean': geometry['geometry_lean'],
        'opig_standalone_candidate': bool(
            geometry['pct_cliff_compounds'] >= 5.0
            and geometry['geometry_lean'] == 'clustered'
            and has_replicates and n_cliffs >= n14.PU_MIN_CLIFFS)}
    return convert_numpy({
        'source': 'n14_activity_cliff_landscape functions',
        'domain': domain, 'n_samples': n,
        'trains_downstream_models': False,
        'ac_reference_provenance': provenance,
        'label_context': label_context,
        'cliff_geometry': geometry,
        'feasibility': feasibility,
        'stage0_diagnostics': stage0, 'stage0_error': stage0_error,
        'decision_hint': decision_hint,
        'cliff_geometry_by_similarity': geometry_by_similarity,
        'similarity': sim_meta,
        'repeat_measurements_excluded': molecule_of_row is not None})


# ===========================================================================
# n37: the Gamma test per representation, the check, the noise share, SIMEX
# ===========================================================================

def gamma_by_representation(dataset_name, reps, rep_names, train_idx, val_idx,
                            y_tr, k_max=None):
    """n37 item 1 on one fold: the Gamma intercept in each representation.

    Tanimoto when every value is 0 or 1, else Euclidean on the fold-scaled
    representation (n37.neighbour_table, n37.gamma_from_neighbours). The
    intercept is in label units squared. A negative one is written as
    computed with negative=True.

    Returns ({representation: record}, {representation: (idx, dist,
    intercept)} for the check, {representation: error}).
    """
    from experiments.noise import n37_noise_budget as n37
    k_max = int(k_max or n37.K_MAX)
    train_idx = np.asarray(train_idx)
    n_tr = len(train_idx)
    by_rep, tables, errors = {}, {}, {}
    for rn in rep_names:
        try:
            if n37.is_binary(reps[rn]):
                fn = n37._tanimoto_rows(np.asarray(reps[rn])[train_idx])
                dname = 'tanimoto'
            else:
                Xs_tr, _ = n37._nc.cached_scaled(dataset_name, reps[rn],
                                                 train_idx, val_idx)
                fn = n37._euclidean_rows(Xs_tr)
                dname = 'euclidean_on_scaled_features'
            idx, dist = n37.neighbour_table(fn, n_tr, k_max)
            g = n37.gamma_from_neighbours(idx, dist, y_tr, k_max)
            g['distance'] = dname
            by_rep[rn] = g
            if 'intercept' in g:
                tables[rn] = (idx, dist, g['intercept'])
        except Exception as exc:
            errors[rn] = f'{type(exc).__name__}: {exc}'
    return by_rep, tables, errors


def gamma_check(tables, y_tr, labelled_passes, k_max=None):
    """n37 item 2: the intercept on damaged labels in each representation,
    beside the realised injected variance. labelled_passes is [(level, draw,
    damaged training labels)]. The distances do not change with the labels,
    so the intercept's rise over level 0 should equal the injected variance."""
    from experiments.noise import n37_noise_budget as n37
    k_max = int(k_max or n37.K_MAX)
    y_tr = np.asarray(y_tr, dtype=float)
    rows = []
    for level, draw, y_n in labelled_passes:
        y_n = np.asarray(y_n, dtype=float)
        injected = float(np.mean((y_n - y_tr) ** 2))
        for rn, (idx, dist, _i0) in tables.items():
            gn = n37.gamma_from_neighbours(idx, dist, y_n, k_max)
            rows.append({'representation': rn, 'level': float(level),
                         'draw': int(draw), 'intercept': gn.get('intercept'),
                         'realised_injected_variance': injected})
    return rows


def model_error_unit(sink, *, dataset_name, fold, model, space, X_tr, X_va,
                     y_tr, y_va, metric, task_type, estimates, own,
                     s2_grid, simex_lambdas=None, simex_draws=None,
                     repeat_s2e=None):
    """n37 items 3, 4 and the per-fold half of 5 for one configuration.

    The configuration fitted on the labels as delivered (row
    n37_model_error) and scored on the held-out labels; each
    representation's Gamma estimate divided by the held-out MSE; SIMEX refits
    on y + N(0, lambda * s2_grid) (rows n37_simex), read with
    n37.simex_read_at at v = minus each non-negative estimate.

    Returns (model_error row, per-fold tails, simex record or None,
    held-out residuals, held-out predictions), the shapes n37 writes.
    """
    from experiments.noise import n37_noise_budget as n37
    simex_lambdas = tuple(n37.SIMEX_LAMBDAS if simex_lambdas is None
                          else simex_lambdas)
    simex_draws = tuple(n37.SIMEX_DRAWS if simex_draws is None
                        else simex_draws)
    metric_name = metric[0]
    y_tr = np.asarray(y_tr, dtype=float)
    y_va = np.asarray(y_va, dtype=float)
    s = score_and_write(
        sink, dataset_name=dataset_name, cleaned=None, X_tr=X_tr, y_D=y_tr,
        X_va=X_va, y_va=y_va, y_clean=y_tr, model=model, metric=metric,
        task_type=task_type, identity=_identity(
            dataset_name, 'n37_model_error', CLEAN_SUFFIX, space, 'none',
            None, 0, fold))
    factory = get_model_factories([model], task_type=task_type,
                                  n_samples=len(y_tr))[model]
    # The same cached fit the row just scored.
    pred = np.asarray(nc.cached_fit_predict(
        dataset_name, model, factory, X_tr, y_tr, X_va,
        predict_mode=nc.predict_mode_for(metric_name)), dtype=float)
    res = y_va - pred
    mse = float(np.mean(res ** 2))
    row = {
        'held_out_score': s['score_D'],
        'mse': mse,
        'share_by_representation': {
            rn: (s2 / mse if s2 >= 0 and mse > 0 else None)
            for rn, s2 in estimates.items()},
        'share_this_configuration': (
            own / mse if own is not None and own >= 0 and mse > 0 else None)}
    neg = sorted(rn for rn, s2 in estimates.items() if s2 < 0)
    if neg:
        row['no_share_negative_intercept'] = neg
    if repeat_s2e is not None:
        row['repeat_variance_over_mse'] = (
            repeat_s2e / mse if mse > 0 and np.isfinite(repeat_s2e) else None)
    tails = n37.tail_diagnostics(res, fit_t=False)

    simex = None
    if not (s2_grid is not None and np.isfinite(s2_grid) and s2_grid > 0):
        simex = {'not_run': 'no representation gave a positive Gamma '
                            'intercept on this fold'}
    else:
        n_tr = len(y_tr)
        curve_v, curve_m, curve_metric = [0.0], [mse], [row['held_out_score']]
        per_draw = {}
        for lam in simex_lambdas:
            ms, mets = [], []
            for d in simex_draws:
                rng = np.random.RandomState(n37._seed(dataset_name, fold, lam, d))
                y_x = y_tr + rng.normal(size=n_tr) * np.sqrt(lam * s2_grid)
                sx = score_and_write(
                    sink, dataset_name=dataset_name, cleaned=None, X_tr=X_tr,
                    y_D=y_x, X_va=X_va, y_va=y_va, y_clean=y_tr, model=model,
                    metric=metric, task_type=task_type, identity=_identity(
                        dataset_name, 'n37_simex',
                        f'lambda={lam:g}|s2_grid={s2_grid:.6g}', space,
                        'simex_gaussian', float(lam), int(d), fold))
                m_x = _mse(sx.get('all_D'))
                if m_x is None:
                    continue
                ms.append(m_x)
                mets.append(float(sx['score_D']))
            per_draw[f'{lam:g}'] = {'mse': ms, 'held_out_score': mets}
            if ms:
                curve_v.append(lam * s2_grid)
                curve_m.append(float(np.mean(ms)))
                curve_metric.append(float(np.mean(mets)))
        targets = {f'representation:{rn}': s2 for rn, s2 in estimates.items()
                   if s2 >= 0}
        if own is not None and own >= 0:
            targets['this_configuration'] = own
        simex = n37.simex_read_at(curve_v, curve_m, targets)
        simex['held_out_score_curve'] = curve_metric
        simex['per_draw'] = per_draw
        simex['s2_grid'] = s2_grid
    return row, tails, simex, res, pred


def noise_budget_fold(data, dataset_name, sink, *, fold_i, train_idx, val_idx,
                      entries, models, factories, metric, task_type,
                      est_reps, check_passes, rep_groups, settings,
                      k_max=None, run_forest=True):
    """n37's body for one outer fold, with its fits written as rows.

    Returns (fold record in n37's per_fold shape, {model: {configuration:
    {'res', 'pred', 'rows'}}} for the out-of-fold tails, {key: error}).
    """
    from experiments.noise import n37_noise_budget as n37
    from experiments.noise.n07_remediation import sweep_ks
    k_max = int(k_max or n37.K_MAX)
    labels = np.asarray(data['labels'], dtype=float)
    reps = data['reps']
    groups = data.get('groups')
    metric_name, _fn, lower_is_better = metric
    y_tr, y_va = labels[train_idx], labels[val_idx]
    g_tr = None if groups is None else np.asarray(groups)[train_idx]
    n_tr = len(train_idx)
    errors, oof = {}, {}
    frec = {'fold': int(fold_i), 'training_rows': int(n_tr),
            'held_out_rows': int(len(val_idx)), 'models': {},
            'noise_estimate_by_representation': {},
            'gamma_by_configuration': {}, 'gamma_sweep': {},
            'check': [], 'model_error': {}, 'simex': {},
            'tails_per_fold': {}}
    if rep_groups is not None:
        frec['repeat_noise_variance_training'] = n37.direct_noise_variance(
            n37.measurements_by_molecule(y_tr, np.asarray(rep_groups)[train_idx]))
    # n37's refusal, unchanged (decided 2026-10-06).
    if n_tr > n37.MAX_PAIR_ROWS:
        frec['refused'] = (f'{n_tr} training rows is over MAX_PAIR_ROWS='
                           f'{n37.MAX_PAIR_ROWS}; nothing was computed')
        return frec, oof, errors

    # 1 and 2: one estimate per representation, and the check.
    by_rep, tables, errs = gamma_by_representation(
        dataset_name, reps, est_reps, train_idx, val_idx, y_tr, k_max)
    frec['noise_estimate_by_representation'] = by_rep
    for rn, e in errs.items():
        errors[f'f{fold_i}|estimate|{rn}'] = e
    noise_groups = noise_groups_for(data)[0]
    labelled = []
    for level, draw in check_passes:
        try:
            y_n, _m, _p = inject_condition(
                data, dataset_name, n37.CHECK_CONDITION, level, draw,
                train_idx=train_idx, groups=noise_groups)
            labelled.append((level, draw, np.asarray(y_n, dtype=float)))
        except Exception as exc:
            errors[f'f{fold_i}|check|{level:g}|d{draw}'] = (
                f'{type(exc).__name__}: {exc}')
    frec['check'] = gamma_check(tables, y_tr, labelled, k_max)
    estimates = {rn: t[2] for rn, t in tables.items()}
    usable = {rn: s2 for rn, s2 in estimates.items() if s2 >= 0}
    del tables

    # 3: the configurations, from the labels as delivered.
    inputs = {}
    for m in models:
        entry = entries.get(m)
        rec = configuration_record(entry)
        rec['metric'] = metric_name
        rec['lower_is_better'] = bool(lower_is_better)
        ins, unav, names = fold_model_inputs(
            data, dataset_name, m, entry, train_idx=train_idx,
            val_idx=val_idx, y_train=y_tr, factory=factories[m], metric=metric)
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

    # The Gamma intercept in each configuration's own space (n37, unchanged):
    # a base configuration's is its representation's; a hybrid's is Euclidean
    # on the scaled hybrid matrix, once per distinct matrix.
    by_hash = {}
    for m in models:
        entry = entries.get(m) or {}
        for sp, (X_tr_s, _X_va_s) in inputs[m].items():
            if sp in ('base_1', 'base_2'):
                rn = (entry.get('base') or [None, None])[int(sp[-1]) - 1]
                g = by_rep.get(rn)
                frec['gamma_by_configuration'].setdefault(m, {})[sp] = {
                    'same_as_representation': rn,
                    'intercept': (g or {}).get('intercept')}
                continue
            h = nc.array_hash(np.asarray(X_tr_s))
            if h not in by_hash:
                try:
                    idx, dist = n37.neighbour_table(n37._euclidean_rows(X_tr_s),
                                                    n_tr, k_max)
                    by_hash[h] = n37.gamma_from_neighbours(idx, dist, y_tr,
                                                           k_max)
                except Exception as exc:
                    by_hash[h] = {'unavailable': f'{type(exc).__name__}: {exc}'}
            frec['gamma_by_configuration'].setdefault(m, {})[sp] = dict(
                by_hash[h], distance='euclidean_on_scaled_features')

    # 7: the neighbour settings. Copied from n37.run's nested _sweep and its
    # caller (not importable): per model for the feature spaces, once per
    # fold for the Tanimoto bits, per configuration for forest proximity.
    sim_ks = tuple(sweep_ks())
    per_model_settings = [s for s in settings if s.space in n37._ss.FEATURE_SPACES]
    forest_settings = ([s for s in settings if s.space == n37._ss.FOREST_SPACE]
                       if run_forest else [])
    shared_settings = [s for s in settings
                       if s.space not in n37._ss.FEATURE_SPACES
                       and s.space != n37._ss.FOREST_SPACE]

    def _sweep(dest, fsim, sts, model, X_model_tr, tag):
        for setting in sts:
            try:
                fn, n, why = fsim.distance_blocks(setting, model, X_model_tr)
                if fn is None:
                    dest[setting.name] = {'unavailable': why}
                    continue
                idx, dist = n37.neighbour_table(fn, n_tr, max(sim_ks))
                for k in sim_ks:
                    dest[f'{setting.name}|k{int(k)}'] = n37.gamma_from_neighbours(
                        idx, dist, y_tr, int(k))
            except Exception as exc:
                errors[f'f{fold_i}|{tag}|{setting.name}'] = (
                    f'{type(exc).__name__}: {exc}')

    sweep = frec['gamma_sweep']
    shared_done = not shared_settings
    for m in models:
        entry = entries.get(m)
        if entry is None:
            sweep.setdefault('per_model', {})[m] = {
                'unavailable': 'no per-fold search entry for this model'}
            continue
        try:
            fsim = n37.fold_similarity(
                dataset_name, data, reps=reps, train_idx=train_idx,
                val_idx=val_idx, y_train=y_tr, groups_train=g_tr,
                base_by_model={m: list(entry.get('base') or [])},
                recipe_for_space_by_model={m: dict(entry.get('recipes') or {})},
                model_factories=factories, metric=metric, task_type=task_type)
        except Exception as exc:
            errors[f'f{fold_i}|fold_similarity|{m}'] = f'{type(exc).__name__}: {exc}'
            continue
        for sp, (Xa, Xb) in inputs[m].items():
            fsim._spaces[(m, sp)] = (Xa, Xb, None)
        for sp, why in (entry.get('unavailable') or {}).items():
            if sp not in inputs[m]:
                fsim._spaces[(m, sp)] = (None, None, why)
                fsim.unavailable[f'{m}|{sp}'] = why
        _sweep(sweep.setdefault('per_model', {}).setdefault(m, {}), fsim,
               per_model_settings, m, None, f'per_model/{m}')
        if not shared_done:
            _sweep(sweep.setdefault('shared', {}), fsim, shared_settings, m,
                   None, 'shared')
            sweep['shared_scored_through'] = m
            shared_done = True
        for sp, (X_tr_s, _X_va_s) in inputs[m].items():
            if forest_settings:
                _sweep(sweep.setdefault('forest', {}).setdefault(m, {})
                       .setdefault(sp, {}), fsim, forest_settings, m, X_tr_s,
                       f'forest/{m}/{sp}')
    if not shared_done:
        why = 'no decided model had a per-fold search entry in this fold'
        sweep['shared'] = {s.name: {'unavailable': why} for s in shared_settings}
        errors[f'f{fold_i}|shared'] = why

    # 3 and 4: the error, the shares, SIMEX. s2_grid is the median of the
    # fold's non-negative estimates (decided 2026-10-06).
    s2_grid = float(np.median(list(usable.values()))) if usable else float('nan')
    frec['simex_s2_grid'] = s2_grid if np.isfinite(s2_grid) else None
    s2e = (frec.get('repeat_noise_variance_training') or {}).get('s2e')
    for m in models:
        for sp, (X_tr_s, X_va_s) in inputs[m].items():
            own = ((frec['gamma_by_configuration'].get(m) or {})
                   .get(sp) or {}).get('intercept')
            try:
                row, tails, simex, res, pred = model_error_unit(
                    sink, dataset_name=dataset_name, fold=int(fold_i),
                    model=m, space=sp, X_tr=X_tr_s, X_va=X_va_s, y_tr=y_tr,
                    y_va=y_va, metric=metric, task_type=task_type,
                    estimates=estimates, own=own, s2_grid=s2_grid,
                    repeat_s2e=s2e)
            except Exception as exc:
                errors[f'f{fold_i}|{m}|{sp}|fit'] = f'{type(exc).__name__}: {exc}'
                continue
            frec['model_error'].setdefault(m, {})[sp] = row
            frec['tails_per_fold'].setdefault(m, {})[sp] = tails
            frec['simex'].setdefault(m, {})[sp] = simex
            o = oof.setdefault(m, {}).setdefault(sp, {'res': [], 'pred': [],
                                                       'rows': []})
            o['res'].append(np.asarray(res).tolist())
            o['pred'].append(np.asarray(pred).tolist())
            o['rows'].append(np.asarray(val_idx).tolist())
    return frec, oof, errors


def noise_budget_summary(per_fold, oof, labels, rep_groups):
    """n37's out-of-fold block, check and summary, from n37's own
    functions (tail_diagnostics, spread_regression, _check, _summarise)."""
    from experiments.noise import n37_noise_budget as n37
    oof_block = {}
    if rep_groups is not None:
        row_count, row_spread = n37._molecule_count_and_spread(labels, rep_groups)
    for m, by_sp in oof.items():
        for sp, o in by_sp.items():
            res = np.concatenate([np.asarray(x) for x in o['res']])
            pred = np.concatenate([np.asarray(x) for x in o['pred']])
            rows = np.concatenate([np.asarray(x, dtype=int) for x in o['rows']])
            rec = {'tails': n37.tail_diagnostics(res),
                   'spread_on_prediction': n37.spread_regression(res, pred[:, None]),
                   'mse': float(np.mean(res ** 2))}
            if rep_groups is not None:
                rec['spread_on_prediction_count_and_spread'] = \
                    n37.spread_regression(res, np.column_stack(
                        [pred, row_count[rows], row_spread[rows]]))
            oof_block.setdefault(m, {})[sp] = rec
    return {'out_of_fold': convert_numpy(oof_block),
            'check': convert_numpy(n37._check(per_fold)),
            'summary': convert_numpy(n37._summarise(per_fold, oof_block))}


# ===========================================================================
# n27: the removal cost curve and the relabelled part
# ===========================================================================

def removal_cost_unit(sink, *, dataset_name, fold, model, space, X_tr, X_va,
                      y_tr, y_va, metric, task_type, shares=None,
                      repeats=None):
    """n27's removal cost on one (fold, model, configuration).

    The configuration trained on a random share of the training molecules,
    every label correct (n27.kept_rows: the same molecules in every
    configuration), at each share below 1.0, `repeats` draws each, paired
    against training on all of them. One row per draw (n27_removal_cost).
    A random share is random by construction, so it gets no random twin
    (n27's rule).

    Returns (reference {'score', 'mse'}, {share: {'r<draw>': {'score',
    'mse', 'training_molecules_kept', 'training_molecules'}}}).
    """
    from experiments.noise import n27_removal_cost_and_propagation as n27
    shares = tuple(n27.SHARES if shares is None else shares)
    repeats = int(n27.REPEATS if repeats is None else repeats)
    y_tr = np.asarray(y_tr, dtype=float)
    n_tr = len(y_tr)
    ref_s = score_and_write(
        None, dataset_name=dataset_name, cleaned=None, X_tr=X_tr, y_D=y_tr,
        X_va=X_va, y_va=y_va, y_clean=y_tr, model=model, metric=metric,
        task_type=task_type, identity={})
    ref = {'score': ref_s['score_D'], 'mse': _mse(ref_s['all_D'])}
    removal = {}
    for share in shares:
        if float(share) >= 1.0:
            continue
        for rep_i in range(repeats):
            keep_idx = n27.kept_rows(share, rep_i, fold, n_tr)
            keep = np.zeros(n_tr, dtype=bool)
            keep[keep_idx] = True
            cleaned = {'X': X_tr[keep], 'y': y_tr[keep], 'keep': keep,
                       'sample_weight': None, 'label_column': None}
            s = score_and_write(
                sink, dataset_name=dataset_name, cleaned=cleaned, X_tr=X_tr,
                y_D=y_tr, X_va=X_va, y_va=y_va, y_clean=y_tr, model=model,
                metric=metric, task_type=task_type, identity=_identity(
                    dataset_name, 'n27_removal_cost',
                    f'share={float(share):g}|draw=r{rep_i}', space, 'none',
                    None, int(rep_i), fold))
            removal.setdefault(f'{float(share):g}', {})[f'r{rep_i}'] = {
                'score': s['score_Dprime'], 'mse': _mse(s.get('all_Dprime')),
                'training_molecules_kept': int(keep.sum()),
                'training_molecules': int(n_tr)}
    return ref, removal


def relabel_unit(sink, *, dataset_name, fold, model, space, X_tr, X_va, y_tr,
                 y_va, g_tr, metric, task_type, alphas=None, oof_splits=None):
    """n27's relabelled part on one (fold, model, configuration): the
    correct labels moved a fraction alpha toward the out-of-fold LightGBM
    prediction (noise_cache.cached_oof_predict, group-aware), fitted, and
    the moved variance written beside it. Rows n27_move_toward_oof, with
    their twins under KIRBY_NOISE_CONTROLS=1.

    Returns {alpha: {'score', 'mse', 'moved_variance'}}, or raises when the
    out-of-fold prediction fails."""
    from experiments.noise import n27_removal_cost_and_propagation as n27
    alphas = tuple(n27.ALPHAS if alphas is None else alphas)
    oof_splits = int(n27.OOF_SPLITS if oof_splits is None else oof_splits)
    y_tr = np.asarray(y_tr, dtype=float)
    oof = nc.cached_oof_predict(dataset_name, X_tr, y_tr, g_tr,
                                n_splits=oof_splits)
    out = {}
    for alpha in alphas:
        y_mv = n27.moved_toward(y_tr, oof, alpha)
        s = score_and_write(
            sink, dataset_name=dataset_name,
            cleaned=n07.cleaned_from_labels(X_tr, y_mv), X_tr=X_tr, y_D=y_tr,
            X_va=X_va, y_va=y_va, y_clean=y_tr, model=model, metric=metric,
            task_type=task_type, twins=True, identity=_identity(
                dataset_name, 'n27_move_toward_oof', f'alpha={float(alpha):g}',
                space, 'none', None, 0, fold))
        out[f'{float(alpha):g}'] = {
            'score': s['score_Dprime'], 'mse': _mse(s.get('all_Dprime')),
            'moved_variance': float(np.mean((y_mv - y_tr) ** 2))}
    return out


# ===========================================================================
# c10: the score drop under damage, hybrid against single
# ===========================================================================

def score_drop_unit(sink, *, dataset_name, condition, level, draw, fold,
                    model, inputs, names, y_D, y_clean, y_va, metric,
                    task_type, n_corrupted=0):
    """c10's Part 1 for one (fold, label set, model): each configuration
    fitted on y_D (damaged training labels; the labels as delivered at level
    0) and scored on the held-out labels as delivered. One row per
    configuration (c10_score_drop, no treatment).

    Also, as c10 wrote: the score on the training labels the model was
    trained on (a second fit through noise_cache, predicting the training
    rows), the width in features, and how many training labels moved.
    For n27's injected part: the held-out MSE and the realised injected
    variance, mean of (damaged - correct)^2 over the training molecules.

    Returns {space: record}.
    """
    metric_name, metric_fn, _lib = metric
    y_D = np.asarray(y_D, dtype=float)
    y_clean = np.asarray(y_clean, dtype=float)
    factory = get_model_factories([model], task_type=task_type,
                                  n_samples=len(y_D))[model]
    injected_var = float(np.mean((y_D - y_clean) ** 2))
    out = {}
    for space, (X_tr, X_va) in inputs.items():
        s = score_and_write(
            sink, dataset_name=dataset_name, cleaned=None, X_tr=X_tr, y_D=y_D,
            X_va=X_va, y_va=y_va, y_clean=y_clean, model=model, metric=metric,
            task_type=task_type, identity=_identity(
                dataset_name, 'c10_score_drop', 'no_treatment', space,
                condition, float(level), int(draw), fold))
        try:
            p_tr = nc.cached_fit_predict(
                dataset_name, model, factory, X_tr, y_D, X_tr,
                predict_mode=nc.predict_mode_for(metric_name))
            train = float(metric_fn(y_D, p_tr))
        except Exception as exc:
            train = None
            print(f"      train score {model}:{space}: {type(exc).__name__}: {exc}")
        out[space] = {
            'fold': int(fold), 'model': model, 'space': space,
            'configuration': names.get(space), 'condition': condition,
            'level': float(level), 'draw': int(draw),
            'n_features': int(np.shape(X_tr)[1]),
            'n_corrupted_train': int(n_corrupted),
            'train_score': train,
            'heldout_score': s['score_D'],
            'heldout_score_clean_labels_same_features': s['score_clean'],
            'heldout_mse': _mse(s.get('all_D')),
            'injected_variance': injected_var}
    return out


def robustness_summary(records, condition, levels, draws, lower_is_better):
    """c10's robustness block for one damage kind, copied from the nested
    _assemble in c10.run (not importable): per model and configuration, the
    held-out score by level, the plain difference damaged minus clean (fold
    mean over draws, in the metric's own units), and auc_norm with the
    Weibull shape where the kind has 4 levels (kirby.noise_robustness)."""
    from kirby.noise_robustness import robustness_metrics
    lvls = [float(x) for x in levels]
    nz = [x for x in lvls if x != 0.0]
    out = {}
    for model in sorted({r['model'] for r in records}):
        out[model] = {}
        for space in CONFIGURATION_SPACES:
            rs = [r for r in records if r['model'] == model and r['space'] == space]
            if not rs:
                continue
            by_fold = {}
            for r in rs:
                by_fold.setdefault(r['fold'], {})[(r['level'], r['draw'])] = \
                    r['heldout_score']
            score_by_level = {0.0: []}
            diff_by_level = {x: [] for x in nz}
            auc_fold, tau_fold, beta_fold = [], [], []
            for _f, sc in sorted(by_fold.items()):
                clean = sc.get((0.0, int(draws[0])))
                if clean is None:
                    clean = next((v for (lv, _d), v in sc.items() if lv == 0.0),
                                 None)
                if clean is not None:
                    score_by_level[0.0].append(clean)
                for x in nz:
                    vals = [sc[(x, d)] for d in draws if (x, d) in sc
                            and sc[(x, d)] is not None]
                    if vals:
                        score_by_level.setdefault(x, []).append(float(np.mean(vals)))
                    if vals and clean is not None:
                        diff_by_level[x].append(
                            float(np.mean([v - clean for v in vals])))
                pa, pt, pb = [], [], []
                if len(lvls) >= 4 and clean is not None:
                    for d in draws:
                        pts = [(x, sc.get((x, d))) for x in nz]
                        if any(v is None for _, v in pts):
                            continue
                        m = robustness_metrics(
                            np.array([0.0] + [x for x, _ in pts]),
                            np.array([clean] + [v for _, v in pts]),
                            baseline_threshold=None,
                            lower_is_better=lower_is_better)
                        pa.append(m['auc_norm'])
                        pt.append(m['weibull_tau'])
                        pb.append(m['weibull_beta'])

                def _fmean(v):
                    v = [x for x in v if x is not None and np.isfinite(x)]
                    return float(np.mean(v)) if v else None
                if pa:
                    auc_fold.append(_fmean(pa))
                    tau_fold.append(_fmean(pt))
                    beta_fold.append(_fmean(pb))
            blk = {'heldout_score_by_level': {
                       str(x): fold_spread(v) for x, v in sorted(score_by_level.items())},
                   'difference_damaged_minus_clean': {
                       str(x): fold_spread(v) for x, v in sorted(diff_by_level.items())}}
            if len(lvls) >= 4:
                blk['auc_norm'] = fold_spread(auc_fold)
                blk['weibull_tau'] = fold_spread(tau_fold)
                blk['weibull_beta'] = fold_spread(beta_fold)
            else:
                blk['auc_norm'] = None
                blk['auc_norm_absent'] = (f'{len(lvls)} levels; auc_norm needs 4 '
                                          f'including level 0 (decided 2026-10-05)')
            out[model][space] = {condition: blk}
    return out


def selected_features(data, dataset_name, model, factory, recipe, tr, va,
                      y_tr, metric):
    """The selected features of one hybrid recipe rebuilt on these training
    labels: [[representation, [feature numbers]], ...]. Copied from the
    nested _selected_features in c10.run (not importable)."""
    metric_name, metric_fn, lower_is_better = metric
    reps = data['reps']
    augmentations = data.get('augmentations') or {}
    bp, aug_cond = recipe
    geometry = recipe_is_geometry(bp)
    augs = augmentations if (aug_cond or 'none') != 'none' else None
    groups = data.get('groups')
    built = nc.cached_hybrid_for_fold(
        dataset_name, {k: v[tr] for k, v in reps.items()},
        {k: v[va] for k, v in reps.items()}, y_tr, bp,
        None if groups is None else np.asarray(groups)[tr],
        None if (geometry or factory is None) else factory(),
        metric_fn, lower_is_better,
        model_name=None if geometry else model,
        metric_name=metric_name, train_idx=tr,
        augmentations_train=(None if augs is None else
                             {a: X[tr] for a, X in augs.items()}),
        augmentations_val=(None if augs is None else
                           {a: X[va] for a, X in augs.items()}),
        want_feature_info=True)
    if built is None or built[0] is None or len(built) < 3:
        return None
    return [[name, [int(i) for i in idx]]
            for name, idx in hybrid_feature_signature(built[2])]


def feature_stability_summary(stab_rows, widths, condition):
    """c10's Part 2 summary, copied from the nested _assemble in c10.run:
    per model and hybrid slot, the across-fold Jaccard and Nogueira
    stability at level 0, the Jaccard of each damaged level against level 0
    on the same fold, its slope over level, and the mean selected features
    per representation."""
    from experiments.noise.c10_noise_robustness import (
        _nogueira_stability, HYBRID_SLOTS)
    out = {}
    for model in sorted({s['model'] for s in stab_rows}):
        out[model] = {}
        for slot in HYBRID_SLOTS:
            ss_ = [s for s in stab_rows if s['model'] == model and s['slot'] == slot]
            if not ss_:
                continue
            sets = {(s['fold'], s['level'], s['draw']):
                    {(name, int(i)) for name, idx in s['features'] for i in idx}
                    for s in ss_}
            lvl0 = {f: v for (f, lv, _d), v in sets.items() if lv == 0.0}
            blk = {'condition': condition,
                   'note': ('each fold has its own recipe in this slot, chosen '
                            'by that fold\'s search, so the across-fold '
                            'numbers include the recipe changing between folds')}
            if len(lvl0) >= 2:
                fs_ = [lvl0[f] for f in sorted(lvl0)]
                jac = []
                for i in range(len(fs_)):
                    for j in range(i + 1, len(fs_)):
                        u = len(fs_[i] | fs_[j])
                        jac.append(len(fs_[i] & fs_[j]) / u if u else 0.0)
                blk['fold_jaccard_level0'] = fold_spread(jac)
                blk['fold_jaccard_level0_unit'] = 'one value per pair of folds'
                blk['nogueira_phi_level0'] = _nogueira_stability(fs_, widths)[0]
            else:
                blk['fold_jaccard_level0'] = None
                blk['fold_jaccard_level0_absent'] = 'fewer than 2 folds'
            by_level = {}
            for (f, lv, _d), v in sets.items():
                if lv == 0.0 or f not in lvl0:
                    continue
                u = len(lvl0[f] | v)
                by_level.setdefault(lv, {}).setdefault(f, []).append(
                    len(lvl0[f] & v) / u if u else 0.0)
            blk['jaccard_vs_level0'] = {
                str(lv): fold_spread([float(np.mean(x)) for x in per_f.values()])
                for lv, per_f in sorted(by_level.items())}
            if len(by_level) >= 2:
                lv_arr = np.array(sorted(by_level))
                j_arr = np.array([blk['jaccard_vs_level0'][str(x)]['mean']
                                  for x in lv_arr])
                blk['feature_nds'] = float(np.polyfit(lv_arr, j_arr, 1)[0])
            counts, n_sets = {}, {}
            for (f, lv, _d), v in sets.items():
                c = counts.setdefault(str(lv), {})
                for name, _ in v:
                    c[name] = c.get(name, 0) + 1
                n_sets[str(lv)] = n_sets.get(str(lv), 0) + 1
            blk['mean_features_per_representation'] = {
                lv: {name: c / n_sets[lv] for name, c in sorted(cs.items())}
                for lv, cs in sorted(counts.items())}
            out[model][slot] = blk
    return out


# ===========================================================================
# n16: one representation's rank rows against level 0, for the smoke
# ===========================================================================

def importance_rank_unit(X, y_by_level, method, fold, top_k=None,
                         dataset_name=None):
    """n16's base arm for one representation and one fold: the importance
    vector at every level (n16._score_one, seed 42 + fold), level 0 scored a
    second time with another scorer seed (the floor), and the cells against
    level 0 (n16._base_cells: features scored, Spearman and Kendall over all,
    either-scored and both-scored features, top-k turnover).

    run() does not call this: it runs n16._run_one_condition whole, which
    loops this over representations, folds, draws and the hybrid arm. This
    is the same arithmetic on one cell, for the smoke."""
    from experiments.noise import n16_importance_under_noise as n16
    top_k = int(top_k or n16.TOP_K)
    block = {}
    for lv, y in y_by_level.items():
        runs = [('base', 42 + fold)]
        if float(lv) == 0.0:
            runs.append(('base_floor', 42 + fold + n16.FLOOR_SEED_OFFSET))
        for tag, seed in runs:
            v, _secs = n16._score_one(X, np.asarray(y, dtype=float), method,
                                      seed=seed, dataset=dataset_name)
            block[(tag, method, 'x', float(lv), int(fold))] = v
    return n16._base_cells(block, method, 'x', sorted(float(l) for l in y_by_level),
                           [int(fold)], int(np.shape(X)[1]), top_k)


# ===========================================================================
# the labels-as-delivered part
# ===========================================================================

def _decided(dataset_name, task_type, n_samples):
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    factories = get_model_factories(models, task_type=task_type,
                                    n_samples=n_samples)
    return [m for m in models if m in factories], factories, models


def run_labels_as_delivered(data, config, output_dir, dataset_name, **kwargs):
    """n01, n14, n37 and n27's removal and relabelled parts."""
    from experiments.noise import n37_noise_budget as n37
    from experiments.noise import n27_removal_cost_and_propagation as n27
    print("\n" + "=" * 70)
    print("EXPERIMENT N47, NOISE MEASUREMENTS: LABELS AS DELIVERED "
          "(n01, n14, n37, n27)")
    print("=" * 70)
    t_start = time.time()
    task_type = data['task_type']
    labels = np.asarray(data['labels'])
    metric = get_domain_metric(data)
    metric_name, _fn, lower_is_better = metric
    parts_dir = _parts_dir(output_dir)
    out = {'experiment': EXP_ID, 'result_layout': RESULT_LAYOUT,
           'part': CLEAN_SUFFIX, 'task_type': task_type,
           'metric': metric_name, 'lower_is_better': bool(lower_is_better)}

    print("\n  n01: cliff reference and cliff-stratified error")
    out['n01'] = n01_part(data, config, output_dir, dataset_name)
    print("\n  n14: cliff landscape")
    try:
        out['n14'] = n14_landscape(data, dataset_name, config, parts_dir)
    except Exception as exc:
        out['n14'] = {'error': f'{type(exc).__name__}: {exc}'}

    if task_type != 'regression':
        why = ('n37 and n27 are regression only: a noise variance in label '
               'units is not defined for a class label')
        out['n37'] = {'skipped': why}
        out['n27'] = {'skipped': why}
        out['seconds'] = round(time.time() - t_start, 1)
        return out

    labels = labels.astype(float)
    folds, n_folds = noise_outer_folds(data, config)
    n_fit_rows = max((len(tr) for _, tr, _ in folds), default=len(labels))
    models, factories, decided = _decided(dataset_name, task_type, n_fit_rows)
    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    n_molecules = len(set(smiles)) if has_smiles else len(labels)
    settings = n37.sweep_settings(dataset_name, n_molecules,
                                  has_smiles=has_smiles)
    est_reps = n37.estimate_representations(dataset_name, data['reps'])
    rep_groups = n37.build_replicate_groups(data)
    check_passes = n37._check_passes(data, dataset_name)
    run_forest = not kwargs.get('skip_proximity', False)

    sink = n07.RowSink(output_dir, EXP_ID, dataset_name,
                       f'{CLEAN_SUFFIX}__{RESULT_LAYOUT}')
    fingerprint = {'layout': RESULT_LAYOUT,
                   'input': input_fingerprint(data, config),
                   'models': list(models), 'n_folds': int(n_folds),
                   'settings': [s.name for s in settings],
                   'est_reps': list(est_reps),
                   'check_passes': [list(p) for p in check_passes],
                   'n27_shares': list(n27.SHARES), 'n27_repeats': n27.REPEATS,
                   'n27_alphas': list(n27.ALPHAS),
                   'simex_lambdas': list(n37.SIMEX_LAMBDAS),
                   'simex_draws': list(n37.SIMEX_DRAWS)}
    units = n07.FoldCheckpoint(output_dir, EXP_ID, dataset_name,
                               f'{CLEAN_SUFFIX}__{RESULT_LAYOUT}', tag='units',
                               fingerprint=fingerprint)
    print(f"\n  n37 and n27: {len(folds)} of {n_folds} folds, models {models}, "
          f"{len(est_reps)} representations for the Gamma test, "
          f"{len(check_passes)} check passes, rows -> {sink.path}")

    per_fold_37, oof_37, errors = [], {}, {}
    ref27, removal27, relabelled27, frec27 = {}, {}, {}, {}
    for fold_i, train_idx, val_idx in folds:
        train_idx, val_idx = np.asarray(train_idx), np.asarray(val_idx)
        fkey = f'f{fold_i}'
        got37 = units.get(['n37', int(fold_i)])
        got27 = units.get(['n27', int(fold_i)])
        if got37 is not None and got27 is not None:
            print(f"    fold {fold_i}: read back from {units.path}")
        else:
            print(f"\n    fold {fold_i}: {len(train_idx)} training, "
                  f"{len(val_idx)} held-out rows")
            entries = None
            if len(train_idx) <= n37.MAX_PAIR_ROWS or got27 is None:
                try:
                    entries = fold_configurations(
                        data, dataset_name, fold_i=fold_i, train_idx=train_idx,
                        n_folds=n_folds)
                except Exception as exc:
                    why = f'per-fold searches failed: {type(exc).__name__}: {exc}'
                    errors[f'{fkey}|fold_configurations'] = why
                    print(f"      {why}")
            if entries is not None and got37 is None:
                frec, oof, errs = noise_budget_fold(
                    data, dataset_name, sink, fold_i=fold_i,
                    train_idx=train_idx, val_idx=val_idx, entries=entries,
                    models=models, factories=factories, metric=metric,
                    task_type=task_type, est_reps=est_reps,
                    check_passes=check_passes, rep_groups=rep_groups,
                    settings=settings, run_forest=run_forest)
                got37 = {'frec': frec, 'oof': oof, 'errors': errs}
                units.put(['n37', int(fold_i)], got37)
            if entries is not None and got27 is None:
                got27 = _n27_fold(data, dataset_name, sink, fold_i=fold_i,
                                  train_idx=train_idx, val_idx=val_idx,
                                  entries=entries, models=models,
                                  factories=factories, metric=metric,
                                  task_type=task_type)
                units.put(['n27', int(fold_i)], got27)
        if got37 is not None:
            per_fold_37.append(got37['frec'])
            errors.update(got37.get('errors') or {})
            for m, by_sp in (got37.get('oof') or {}).items():
                for sp, o in by_sp.items():
                    d = oof_37.setdefault(m, {}).setdefault(
                        sp, {'res': [], 'pred': [], 'rows': []})
                    for k in ('res', 'pred', 'rows'):
                        d[k].extend(o[k])
        if got27 is not None:
            frec27[str(fold_i)] = got27['record']
            errors.update(got27.get('errors') or {})
            for sp, by_m in got27['ref'].items():
                for m, v in by_m.items():
                    ref27.setdefault(sp, {}).setdefault(m, {})[fkey] = v
            for sp, by_s in got27['removal'].items():
                for sh, by_m in by_s.items():
                    for m, by_r in by_m.items():
                        for r, v in by_r.items():
                            (removal27.setdefault(sp, {}).setdefault(sh, {})
                             .setdefault(m, {}).setdefault(r, {}))[fkey] = v
            for sp, by_a in got27['relabelled'].items():
                for a, by_m in by_a.items():
                    for m, v in by_m.items():
                        (relabelled27.setdefault(sp, {}).setdefault(a, {})
                         .setdefault(m, {}))[fkey] = v

    out['n37'] = {
        'source': 'n37_noise_budget functions; fits written as rows',
        'models': models, 'models_decided': decided,
        'estimate_representations': est_reps, 'k_max': n37.K_MAX,
        'check_condition': n37.CHECK_CONDITION,
        'check_passes': [list(p) for p in check_passes],
        'check_level_unit': level_unit(n37.CHECK_CONDITION),
        'simex_lambdas': list(n37.SIMEX_LAMBDAS),
        'simex_draws': list(n37.SIMEX_DRAWS),
        'similarity_settings': [s.name for s in settings],
        'has_repeat_measurements': rep_groups is not None,
        'gamma_note': ('intercept of gamma_k on delta_k, k = 1..k_max nearest '
                       'neighbours; label units squared; an upper bound on the '
                       'label-noise variance. A negative intercept is written '
                       'with negative=True, gets no share, and is left out of '
                       'the summary mean.'),
        'per_fold': convert_numpy(per_fold_37)}
    if rep_groups is not None:
        out['n37']['repeat_noise_variance_whole_dataset'] = \
            n37.direct_noise_variance(n37.measurements_by_molecule(
                labels, rep_groups))
    try:
        out['n37'].update(noise_budget_summary(per_fold_37, oof_37, labels,
                                               rep_groups))
    except Exception as exc:
        out['n37']['summary_error'] = f'{type(exc).__name__}: {exc}'

    out['n27'] = {
        'source': 'n27_removal_cost_and_propagation functions; fits written as rows',
        'shares': list(n27.SHARES), 'repeats': n27.REPEATS,
        'alphas': list(n27.ALPHAS), 'oof_splits': n27.OOF_SPLITS,
        'injected_part': ('in the gaussian condition file, read from the c10 '
                          'fits under gaussian'),
        'removal_note': ('one row per random draw of the kept molecules, '
                         'paired against the share 1.0 fit on the same fold '
                         'and features; positive loss means the smaller '
                         'training set scored worse. The control of handoff '
                         'section 3.2 for every deletion cleanup.'),
        'fold_configurations': convert_numpy(frec27),
        'per_fold': convert_numpy({'reference': ref27, 'removal_cost': removal27,
                                   'relabelled': relabelled27})}
    try:
        out['n27']['summary'] = convert_numpy(n27._summarise(
            ref27, removal27, {}, relabelled27, lower_is_better, metric_name))
    except Exception as exc:
        out['n27']['summary_error'] = f'{type(exc).__name__}: {exc}'

    out.update({'n_folds': int(n_folds),
                'folds_run': [int(f) for f, _t, _v in folds],
                'rows': sink.summary(), 'errors': errors,
                'checkpoint': units.path,
                'seconds': round(time.time() - t_start, 1)})
    print(f"\n  N47 labels as delivered: {sink.n_written} rows written, "
          f"{len(errors)} errors, {time.time() - t_start:.0f}s")
    return out


def _n27_fold(data, dataset_name, sink, *, fold_i, train_idx, val_idx, entries,
              models, factories, metric, task_type):
    """n27's removal cost and relabelled parts on one fold, every model and
    configuration built from the correct training labels."""
    labels = np.asarray(data['labels'], dtype=float)
    groups = data.get('groups')
    y_tr, y_va = labels[train_idx], labels[val_idx]
    g_tr = None if groups is None else np.asarray(groups)[train_idx]
    ref, removal, relabelled, errors, record = {}, {}, {}, {}, {}
    for m in models:
        entry = entries.get(m)
        ins, unav, names = fold_model_inputs(
            data, dataset_name, m, entry, train_idx=train_idx,
            val_idx=val_idx, y_train=y_tr, factory=factories[m], metric=metric)
        rec = configuration_record(entry, names)
        rec['configurations_unavailable'] = dict(unav)
        rec['n_features'] = {sp: int(np.shape(X[0])[1]) for sp, X in ins.items()}
        rec['held_out_score_correct_labels'] = {}
        record[m] = rec
        for sp, (X_tr, X_va) in ins.items():
            try:
                r, rem = removal_cost_unit(
                    sink, dataset_name=dataset_name, fold=int(fold_i), model=m,
                    space=sp, X_tr=X_tr, X_va=X_va, y_tr=y_tr, y_va=y_va,
                    metric=metric, task_type=task_type)
                ref.setdefault(sp, {})[m] = r
                rec['held_out_score_correct_labels'][sp] = r['score']
                for sh, by_r in rem.items():
                    removal.setdefault(sp, {}).setdefault(sh, {})[m] = by_r
            except Exception as exc:
                errors[f'{sp}|{m}|f{fold_i}|removal'] = f'{type(exc).__name__}: {exc}'
            try:
                rl = relabel_unit(
                    sink, dataset_name=dataset_name, fold=int(fold_i), model=m,
                    space=sp, X_tr=X_tr, X_va=X_va, y_tr=y_tr, y_va=y_va,
                    g_tr=g_tr, metric=metric, task_type=task_type)
                for a, v in rl.items():
                    relabelled.setdefault(sp, {}).setdefault(a, {})[m] = v
            except Exception as exc:
                errors[f'{sp}|{m}|f{fold_i}|relabelled'] = f'{type(exc).__name__}: {exc}'
    return {'ref': ref, 'removal': removal, 'relabelled': relabelled,
            'errors': errors, 'record': record}


# ===========================================================================
# one damage kind
# ===========================================================================

def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    """c10, n27's injected part (gaussian only) and n16, for one kind."""
    from experiments.noise import c10_noise_robustness as c10
    from experiments.noise import n16_importance_under_noise as n16
    from experiments.noise import n27_removal_cost_and_propagation as n27
    print("\n" + "=" * 70)
    print("EXPERIMENT N47, NOISE MEASUREMENTS: ONE DAMAGE KIND (c10, n27, n16)")
    print("=" * 70)
    t_start = time.time()
    task_type = data['task_type']
    condition = kwargs.get('noise_condition') or conditions_for_task(data)[0]
    if task_type != 'regression':
        # c10, n16 and n27 are each regression only, by their own guards.
        return {'experiment': EXP_ID, 'noise_condition': condition,
                'skipped': True, 'reason': 'regression_only',
                'task_type': task_type}
    labels = np.asarray(data['labels'], dtype=float)
    metric = get_domain_metric(data)
    metric_name, _fn, lower_is_better = metric
    levels = tuple(float(x) for x in decided_levels(data, dataset_name, condition))
    draws = tuple(int(d) for d in draws_for(condition, NOISE_DRAWS))
    cells = [c10.CLEAN_CELL] + [(condition, lv, d) for lv in levels
                                if lv != 0.0 for d in draws]
    folds, n_folds = noise_outer_folds(data, config)
    models, factories, decided = _decided(
        dataset_name, task_type,
        max((len(tr) for _, tr, _ in folds), default=len(labels)))
    noise_groups, n_noise_groups, largest_share = noise_groups_for(data)
    # c10's Part 2 runs under the first kind swept (c10.run: strategies[0],
    # grouped kinds dropped when there are no scaffold groups).
    kinds = [c for c in conditions_for_task(data)
             if not (noise_groups is None and c in GROUPED_STRATEGIES)]
    stability_here = bool(kinds) and condition == kinds[0]

    file_cond = f'{condition}__{RESULT_LAYOUT}'
    sink = n07.RowSink(output_dir, EXP_ID, dataset_name, file_cond)
    fingerprint = {'layout': RESULT_LAYOUT,
                   'input': input_fingerprint(data, config),
                   'condition': condition, 'levels': list(levels),
                   'draws': list(draws), 'n_folds': int(n_folds),
                   'models': list(models), 'noise_seed': NOISE_SEED}
    units = n07.FoldCheckpoint(output_dir, EXP_ID, dataset_name, file_cond,
                               tag='units', fingerprint=fingerprint)
    print(f"  condition {condition}; levels {list(levels)} "
          f"({level_unit(condition)}); {len(cells)} label sets per fold; "
          f"{n_folds} outer folds; models {models}; rows -> {sink.path}")

    records, stab_rows, recipe_rows = [], [], []
    recipe_skipped, fold_records, unavailable, errors = {}, {}, {}, {}
    entries_by_fold = {}
    widths = (sum(int(np.asarray(v).shape[1]) for v in data['reps'].values())
              + sum(int(np.asarray(v).shape[1])
                    for v in (data.get('augmentations') or {}).values()))

    for fold_i, tr, va in folds:
        tr, va = np.asarray(tr), np.asarray(va)
        print(f"\n  fold {fold_i}: {len(tr)} training, {len(va)} held out")
        try:
            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=tr, n_folds=n_folds)
        except Exception as exc:
            why = f'per-fold searches failed: {type(exc).__name__}: {exc}'
            fold_records[str(fold_i)] = {'error': why}
            print(f"    {why}")
            continue
        entries_by_fold[fold_i] = entries
        fold_records[str(fold_i)] = {
            'n_train': int(len(tr)), 'n_held_out': int(len(va)),
            'configurations': {m: configuration_record(entries.get(m))
                               for m in models}}
        y_clean = labels[tr]
        y_va = labels[va]
        for cond, lv, d in cells:
            try:
                y_tr, mask, _prov = inject_condition(
                    data, dataset_name, cond, lv, d, train_idx=tr,
                    groups=noise_groups, base_seed=NOISE_SEED)
            except Exception as exc:
                errors[f'f{fold_i}|inject|{lv:g}|d{d}'] = f'{type(exc).__name__}: {exc}'
                continue
            y_tr = np.asarray(y_tr, dtype=float)
            for m in models:
                key = ['score', int(fold_i), m, float(lv), int(d)]
                got = units.get(key)
                if got is None:
                    entry = entries.get(m)
                    if entry is None:
                        unavailable[f'f{fold_i}|{m}'] = 'the per-fold search failed'
                        continue
                    try:
                        ins, unav, names = fold_model_inputs(
                            data, dataset_name, m, entry, train_idx=tr,
                            val_idx=va, y_train=y_tr, factory=factories[m],
                            metric=metric)
                        recs = score_drop_unit(
                            sink, dataset_name=dataset_name, condition=condition,
                            level=lv, draw=d, fold=int(fold_i), model=m,
                            inputs=ins, names=names, y_D=y_tr, y_clean=y_clean,
                            y_va=y_va, metric=metric, task_type=task_type,
                            n_corrupted=int(np.asarray(mask).sum()))
                    except Exception as exc:
                        errors[f'f{fold_i}|{m}|{lv:g}|d{d}'] = f'{type(exc).__name__}: {exc}'
                        continue
                    got = {'records': list(recs.values()),
                           'unavailable': dict(unav)}
                    units.put(key, got)
                records.extend(got['records'])
                for sp, why in (got.get('unavailable') or {}).items():
                    unavailable.setdefault(f'f{fold_i}|{m}|{sp}', why)

            # c10 Part 2: selected features of each hybrid slot.
            if not stability_here:
                continue
            for m in models:
                entry = entries.get(m)
                if entry is None:
                    continue
                for slot in c10.HYBRID_SLOTS:
                    recipe = (entry.get('recipes') or {}).get(slot)
                    if recipe is None or recipe[0] is None:
                        continue
                    key = ['features', int(fold_i), m, slot, float(lv), int(d)]
                    sel = units.get(key)
                    if sel is None:
                        try:
                            feats = selected_features(
                                data, dataset_name, m, factories[m], recipe,
                                tr, va, y_tr, metric)
                        except Exception as exc:
                            errors[f'f{fold_i}|{m}|{slot}|{lv:g}|d{d}|features'] = (
                                f'{type(exc).__name__}: {exc}')
                            continue
                        if feats is None:
                            continue
                        sel = {'features': feats}
                        units.put(key, sel)
                    stab_rows.append({'fold': int(fold_i), 'model': m,
                                      'slot': slot, 'level': float(lv),
                                      'draw': int(d),
                                      'features': sel['features']})

    # c10 Part 3: does the recipe move under damage. One search per
    # (non-zero level, draw) and fold (c10.recipe_movement_fold).
    print(f"\n  c10 recipe movement: {len(cells) - 1} searches per fold on "
          f"damaged training labels")
    for fold_i, tr, _va in folds:
        if fold_i not in entries_by_fold:
            continue
        try:
            rows_, skipped_ = c10.recipe_movement_fold(
                data, dataset_name, fold_i=fold_i, tr=np.asarray(tr),
                n_folds=n_folds, entries=entries_by_fold[fold_i], cells=cells,
                noise_groups=noise_groups, noise_seed=NOISE_SEED, ckpt=units)
            recipe_rows.extend(rows_)
            recipe_skipped.update(skipped_)
        except Exception as exc:
            errors[f'f{fold_i}|recipe_movement'] = f'{type(exc).__name__}: {exc}'

    out = {
        'experiment': EXP_ID, 'result_layout': RESULT_LAYOUT,
        'noise_condition': condition, 'task_type': task_type,
        'metric': metric_name, 'lower_is_better': bool(lower_is_better),
        'levels': list(levels), 'level_unit': level_unit(condition),
        'draws_per_nonzero_level': len(draws), 'noise_seed': NOISE_SEED,
        'n_folds': int(n_folds), 'folds_run': [int(f) for f, _t, _v in folds],
        'models': models, 'models_decided': decided,
        'configuration_spaces': list(CONFIGURATION_SPACES),
        'knn_group': dataset_knn_group(data, dataset_name),
        'n_noise_groups': n_noise_groups,
        'largest_noise_group_share': largest_share,
        'folds': fold_records,
        'c10': {
            'source': 'c10_noise_robustness; fits written as rows '
                      '(c10_score_drop, no_treatment)',
            'design': ('per outer fold and decided model: 2 base '
                       'representations and 2 hybrids chosen by a search on the '
                       'fold\'s training rows (labels as delivered); each '
                       'fitted on damaged training labels, hybrids rebuilt on '
                       'them, scored on the held-out labels as delivered'),
            'records': records,
            'robustness': robustness_summary(records, condition, levels, draws,
                                             lower_is_better),
            'feature_stability': (
                feature_stability_summary(stab_rows, widths, condition)
                if stability_here else
                {'not_here': f'c10 Part 2 runs under the first kind swept '
                             f'({kinds[0] if kinds else None})'}),
            'feature_stability_rows': stab_rows,
            'recipe_movement': c10.recipe_movement_summary(recipe_rows),
            'recipe_movement_rows': recipe_rows,
            'recipe_movement_skipped': recipe_skipped,
            'recipe_movement_scoring': c10.RECIPE_SEARCH_SCORING,
            'recipe_movement_fields': list(c10.RECIPE_FIELDS)},
        'unavailable': unavailable,
    }

    # n27's injected part, gaussian only (decided 2026-10-06): the same fits
    # as c10's gaussian cells. The reference is each fold's level-0 fit.
    if condition == n27.CONDITION:
        ref, injected = {}, {}
        for r in records:
            fkey = f"f{r['fold']}"
            if r['heldout_mse'] is None:
                continue
            if r['level'] == 0.0:
                ref.setdefault(r['space'], {}).setdefault(r['model'], {})[fkey] = {
                    'score': r['heldout_score'], 'mse': r['heldout_mse']}
            else:
                lkey = f"{r['level']:g}"
                pkey = f"l{lkey}d{r['draw']}{fkey}"
                (injected.setdefault(r['space'], {}).setdefault(lkey, {})
                 .setdefault(r['model'], {}))[pkey] = {
                    'score': r['heldout_score'], 'mse': r['heldout_mse'],
                    'injected_variance': r['injected_variance'], 'fold': fkey}
        try:
            summ = n27._summarise(ref, {}, injected, {}, lower_is_better,
                                  metric_name)
        except Exception as exc:
            summ = {'error': f'{type(exc).__name__}: {exc}'}
        out['n27_injected'] = convert_numpy({
            'source': 'n27._summarise on the c10 gaussian fits',
            'ratio_note': ('propagation ratio = (held-out MSE of the fit on '
                           'injected labels - held-out MSE of the level-0 fit) '
                           '/ (mean over training molecules of (injected label '
                           '- correct label) squared); both in label units '
                           'squared, both written beside every ratio. The '
                           'hybrids are rebuilt on the injected labels, as '
                           'n27 did.'),
            'per_fold': {'reference': ref, 'injected': injected},
            'propagation': (summ or {}).get('propagation', summ)})

    # n16 whole: no fit, no rows; its result goes in this file.
    print("\n  n16: feature importance against level 0")
    try:
        out['n16'] = n16._run_one_condition(data, config, _parts_dir(output_dir),
                                            dataset_name,
                                            noise_condition=condition)
    except Exception as exc:
        out['n16'] = {'error': f'{type(exc).__name__}: {exc}'}

    out.update({'rows': sink.summary(), 'errors': errors,
                'checkpoint': units.path,
                'seconds': round(time.time() - t_start, 1)})
    print(f"\n  N47 {condition}: {len(records)} configuration fits read, "
          f"{sink.n_written} rows written, {len(errors)} errors, "
          f"{time.time() - t_start:.0f}s")
    return convert_numpy(out)


# ===========================================================================
# the run
# ===========================================================================

def _clean_file_on_disk(output_dir, dataset_name, fingerprint):
    """The labels-as-delivered file when it is finished and computed from
    these inputs (sweep_conditions' rule), else None."""
    dest = Path(output_dir) / EXP_ID / f'{dataset_name}__{CLEAN_SUFFIX}.json'
    if not (dest.exists() and dest.stat().st_size > 0):
        return None
    try:
        with open(dest) as fh:
            prev = json.load(fh)
    except (ValueError, OSError):
        return None
    if experiment_result_is_empty(prev, classify_unmarked=True):
        return None
    if fingerprint and fingerprint_mismatch(prev.get('input_fingerprint'),
                                            fingerprint):
        return None
    return str(dest)


def run(data, config, output_dir, dataset_name, **kwargs):
    """The labels-as-delivered part once, then every damage kind.

    A job named with one condition runs the labels-as-delivered part only
    when that condition is the first of the dataset's list, so parallel
    condition jobs do not compute it twice. `parts` ('labels_as_delivered',
    'conditions') picks either half.
    """
    kwargs = dict(kwargs)
    default = conditions_for_task(data)
    parts = kwargs.pop('parts', None)
    named = kwargs.get('noise_condition')
    do_clean = parts is None or CLEAN_SUFFIX in parts
    if (parts is None and named and not kwargs.get('noise_conditions')
            and default and named != default[0]):
        do_clean = False
    do_conditions = parts is None or 'conditions' in parts

    clean_file = None
    if do_clean:
        fp = input_fingerprint(data, config)
        clean_file = (None if kwargs.get('force_rerun') or output_dir is None
                      else _clean_file_on_disk(output_dir, dataset_name, fp))
        if clean_file:
            print(f"  {CLEAN_SUFFIX}: already on disk, skipping ({clean_file})")
        else:
            res = run_labels_as_delivered(data, config, output_dir,
                                          dataset_name, **kwargs)
            if output_dir is not None:
                save_experiment_results(res, output_dir, EXP_ID, dataset_name,
                                        fingerprint=fp,
                                        filename_suffix=CLEAN_SUFFIX)
                clean_file = str(Path(output_dir) / EXP_ID /
                                 f'{dataset_name}__{CLEAN_SUFFIX}.json')
    index = {}
    if do_conditions:
        index = sweep_conditions(
            _run_one_condition, EXP_ID, data, config, output_dir,
            dataset_name, default_conditions=default, kwargs=kwargs,
            groups_needed=GROUPED_STRATEGIES)
    index['labels_as_delivered_file'] = clean_file
    index['labels_as_delivered_run_here'] = bool(do_clean)
    return index


def main():
    import argparse
    from experiments.noise.noise_helpers import run_all
    ap = argparse.ArgumentParser(description='N47 on one dataset through '
                                             'noise_helpers.run_all')
    ap.add_argument('dataset')
    ap.add_argument('--output', required=True)
    ap.add_argument('--noise-condition', default=None)
    args = ap.parse_args()
    run_all(dataset_name=args.dataset, output_dir=args.output,
            experiments=['n47'], noise_condition=args.noise_condition)


if __name__ == '__main__':
    main()
