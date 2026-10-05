#!/usr/bin/env python3
"""C03: Cross-dataset comparison of optimal hybrid configurations.

Paper: Complexity
Conclusion: Optimal allocation/importance/budget choices CO-VARY across
datasets in interpretable ways (correlations with complexity metrics).
This does NOT directly demonstrate transfer — see c24 for true source→target
blueprint transfer.  c03 finds the native-optimal configuration on each
dataset and analyses cross-dataset similarities (which configs cluster
together; which complexity dimensions predict optimal budget).

Loads multiple datasets and computes complexity profiles (Lorena et al. 2018).
Per dataset, per held-out fold, per decided downstream model, it takes the 2
best base representations and the 2 best hybrids from a search on that fold's
training rows, fits each on the training rows and scores it on the held-out
rows. It then compares the winning budgets across datasets: one Spearman
coefficient per (complexity descriptor, downstream model), and a budget
variance decomposition over models and datasets.

2026-09-12, decided (EXPERIMENTS_FINALIZED, c03): the complexity-descriptor
correlation is computed per downstream model — one Spearman coefficient per
(descriptor, model), not one overall.  The winning budget for a
(dataset, model) is the budget whose best feature set scores highest for that
model, which is the definition the model-by-dataset table already used, so both
questions now use one definition.

2026-10-05, aligned with HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4 (chat 3):
  - Models: each dataset's two decided models (decided_grid.decided_models
    (ds, 'general')), nothing else. The five-model DOWNSTREAM_MODELS loop, the
    knn and gp rows the distance search added, and the SVR / ElasticNet fits
    are gone (R1).
  - Searches: one search per held-out fold, on that fold's training rows, on
    the dataset's decided grid, pruning off (fold_standard_selection ->
    search_on_fold). Nothing reads a search over every row: the stored
    distance search (get_best_search_for_downstream), the owner's-set reads
    (get_rule_hybrids) and the re-picks from the stored search's fold scores
    (fold_reselection, _winner_per_fold) are removed (R2, R3, R6).
  - Selection: per model, per fold, the 2 best base representations and the
    2 best hybrids; a second hybrid with the same selected features as the
    first is skipped (R4, handoff 2.1). c03 is not one of the experiments that
    keep a distance hybrid (handoff 2.3 item 1), so every hybrid comes from
    the general search and its decided budgets, not from the distance
    search's 10-500 features.
  - Each chosen hybrid is rebuilt on the fold's training rows with the
    representation scores that fold's search measured (dirty-run bug 1) and
    scored on the held-out rows. Base representations are fitted and scored
    the same way. Both go through the engine's held-out scorer, so features
    are scaled with the 0/1 rule (handoff 2.3 item 17) and the model is the
    one the search fitted.
  - Winning budget per (dataset, model): the budget of each fold's top
    hybrid. The correlation reads the median over folds; every fold's budget
    is in the result file.
  - Representations: DATASET_REPS[ds]['general'] only, float32. No
    augmentation: Step 8 keeps it only on the distance search of
    ginkgo_colloidal and ginkgo_hic, and c03 reads no distance search.
  - Outer folds: config['n_splits'] through nested_cv_outer_folds. 1 gives one
    held-out split, and every spread then says one fold.
  - No p-values (the owner excluded significance tests everywhere).
  - Each finished fold is written to
    <output>/c03_cross_dataset_transfer/multi__<dataset>.partial.jsonl and read
    back on a restart. The identity names this version, so a file written by
    the earlier code is moved aside, not read.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time

import numpy as np
from scipy.stats import spearmanr

from kirby.find_best_hybrid import nested_cv_outer_folds, _outer_fit_score

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, compute_dataset_complexity, get_domain_metric,
    get_model_factories, fold_standard_selection, build_fold_hybrid,
    fold_hybrid_scoring_kwargs, should_skip_experiment,
    STANDARD_N_BASE, STANDARD_N_HYBRIDS,
)
from experiments.decided_grid import decided_models, decided_search
from experiments._checkpoint import (
    Checkpoint, experiment_sidecar, run_identity)

EXPERIMENT_ID = 'c03_cross_dataset_transfer'
# The outer folds' seed. fold_standard_selection files each fold's search under
# nested_cv_cache_namespace(fold, n_folds, FOLD_SEED), the same name every other
# per-fold experiment on this dataset uses, so one fold's search is run once.
FOLD_SEED = 42
# Bumped 2026-10-05 (per-fold alignment). Part of the checkpoint identity, so a
# partial file written by the whole-dataset version is never read back.
RESULT_VERSION = '2026-10-05_fold_standard_selection'

HYBRID_FIELDS = ('label', 'allocation_method', 'importance_method', 'budget',
                 'internal_evaluator', 'rep_count', 'step_size', 'tune')


def _spread(vals):
    """Mean, standard deviation and count of the finite values. One value has
    no spread, and says so instead of reporting 0."""
    v = [float(x) for x in vals if x is not None and np.isfinite(float(x))]
    if not v:
        return {'mean': None, 'std': None, 'n_folds': 0}
    if len(v) == 1:
        return {'mean': v[0], 'std': None, 'n_folds': 1,
                'spread_note': 'one fold, no spread'}
    return {'mean': float(np.mean(v)), 'std': float(np.std(v)),
            'n_folds': len(v)}


def _baselines_for(search, models):
    """{representation: {model: training-row score}} from one fold's search,
    for the decided models only. Read, no fits."""
    out = {}
    for rep, by_model in (getattr(search, 'baselines', None) or {}).items():
        if not isinstance(by_model, dict):
            continue
        row = {}
        for m in models:
            v = by_model.get(m)
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                row[m] = v
        if row:
            out[rep] = row
    return out


def _fold_record(data, ds_name, k, tr, va, n_outer, models, metric_name,
                 metric_fn, lower_is_better):
    """One held-out fold: the standard selection and its held-out scores.

    Returns a plain dict (no blueprints), so it can be checkpointed.
    """
    task_type = data['task_type']
    labels = np.asarray(data['labels'])
    groups = np.asarray(data['groups'])
    reps = data['reps']
    print(f"    fold {k + 1}/{n_outer}: per-fold search on {len(tr)} "
          f"training rows, {len(va)} held-out rows")

    sel = fold_standard_selection(
        reps, labels, groups, train_idx=tr, fold_i=k,
        n_outer_folds=n_outer, dataset_name=ds_name, models=list(models),
        task_type=task_type, metric=metric_name, seed=FOLD_SEED,
        n_base=STANDARD_N_BASE, n_hybrids=STANDARD_N_HYBRIDS)

    tr_reps = {r: X[tr] for r, X in reps.items()}
    va_reps = {r: X[va] for r, X in reps.items()}
    y_tr, y_va = labels[tr], labels[va]
    g_tr = groups[tr]
    factories = get_model_factories(list(models), task_type=task_type,
                                    n_samples=len(tr))

    rec = {'fold': int(k), 'n_train': int(len(tr)), 'n_val': int(len(va)),
           'selection_scope': 'fold_train_rows', 'models': {},
           'failures': []}

    def _held_out(X_tr, X_va, m):
        score, _ = _outer_fit_score(
            X_tr, y_tr, X_va, y_va, m, task_type, metric_fn, seed=FOLD_SEED,
            train_groups=g_tr)
        return float(score) if np.isfinite(score) else None

    for m in models:
        s = sel.get(m)
        if s is None:
            rec['failures'].append(
                {'model': m, 'what': 'search',
                 'reason': "the search on this fold's training rows failed"})
            continue
        base_rows = []
        for i, e in enumerate(s['base_detail']['selected'], 1):
            rep = e['rep']
            row = {'slot': f'base{i}', 'rep': rep,
                   'n_features': int(reps[rep].shape[1]),
                   'score_train_rows': e.get('score')}
            try:
                row['score_held_out'] = _held_out(tr_reps[rep], va_reps[rep], m)
            except Exception as ex:
                row['score_held_out'] = None
                row['error'] = f'{type(ex).__name__}: {ex}'
            base_rows.append(row)
            print(f"      {m:<10} base{i}   {rep:<26} train-rows "
                  f"{metric_name}={e.get('score')}, held-out "
                  f"{metric_name}={row['score_held_out']}")
        if not s['base_detail'].get('complete', True):
            rec['failures'].append(
                {'model': m, 'what': 'base_reps',
                 'reason': s['base_detail'].get('shortfall_reason')
                 or f"fewer than {STANDARD_N_BASE} scored representations"})

        hyb_rows = []
        for i, e in enumerate(s['hybrids'], 1):
            bp = e['blueprint']
            row = {'slot': f'hybrid{i}',
                   **{f: e.get(f) for f in HYBRID_FIELDS},
                   'aug_condition': e.get('aug_condition', 'none'),
                   'score_train_rows': e.get('score_mean')}
            try:
                if row['aug_condition'] != 'none':
                    # c03 loads no augmentation representations, so the
                    # general search runs 'none' only (Step 8).
                    raise ValueError(
                        f"augmentation condition {row['aug_condition']!r} "
                        f"with no augmentation representations loaded")
                if factories.get(m) is None:
                    raise ValueError(f'no model factory for {m}')
                X_tr_h, X_va_h, fi = build_fold_hybrid(
                    tr_reps, va_reps, y_tr, bp, groups=g_tr,
                    # This fold's search measured these on this fold's
                    # training rows (dirty-run bug 1).
                    precomputed_alloc_scores=getattr(bp, 'rep_scores', None),
                    **fold_hybrid_scoring_kwargs(
                        bp, model=factories[m](), metric_fn=metric_fn,
                        lower_is_better=lower_is_better))
                row['features_by_rep'] = {
                    r: int(len(info['selected_indices']))
                    for r, info in fi.items()
                    if isinstance(info, dict) and 'selected_indices' in info}
                row['n_features'] = int(np.asarray(X_tr_h).shape[1])
                row['score_held_out'] = _held_out(X_tr_h, X_va_h, m)
            except Exception as ex:
                row['score_held_out'] = None
                row['error'] = f'{type(ex).__name__}: {ex}'
                print(f"      {m:<10} hybrid{i} build failed: {row['error']}")
                hyb_rows.append(row)
                continue
            hyb_rows.append(row)
            print(f"      {m:<10} hybrid{i} {row['n_features']:5d} features "
                  f"(budget {row['budget']}, {row['allocation_method']}, "
                  f"{row['importance_method']}, eval "
                  f"{row['internal_evaluator']}, rep_count "
                  f"{row['rep_count']}) train-rows "
                  f"{metric_name}={row['score_train_rows']}, held-out "
                  f"{metric_name}={row['score_held_out']}")
        if len(s['hybrids']) < STANDARD_N_HYBRIDS:
            rec['failures'].append(
                {'model': m, 'what': 'hybrids',
                 'reason': f"the search gave {len(s['hybrids'])} hybrids with "
                           f"distinct selected features, not "
                           f"{STANDARD_N_HYBRIDS}"})
        rec['models'][m] = {
            'base': base_rows,
            'hybrids': hyb_rows,
            'hybrids_skipped_identical_features': [
                h.get('label') for h in
                (s.get('hybrids_skipped_identical_features') or [])
                if isinstance(h, dict)],
            # Every representation's training-row score for this model, from
            # the same fold search.
            'baselines_train_rows': {
                r: v[m] for r, v in
                _baselines_for(s.get('search'), [m]).items()},
        }
    return rec


def _summarise_model(fold_recs, m, lower_is_better):
    """One downstream model on one dataset, over its held-out folds.

    The winning budget of a fold is the budget (features asked for) of that
    fold's top hybrid, ranked on the fold's training rows. 'budget' is the
    median over folds and is what the correlation reads; 'budget_per_fold'
    holds every fold's.
    """
    budgets, dims, h1_out, h1_tr, b1_out, deltas = [], [], [], [], [], []
    per_fold = []
    for rec in fold_recs:
        mr = (rec.get('models') or {}).get(m)
        if not mr:
            continue
        h1 = next((h for h in mr['hybrids'] if h['slot'] == 'hybrid1'), None)
        b1 = next((b for b in mr['base'] if b['slot'] == 'base1'), None)
        pf = {'fold': rec['fold']}
        if h1 is not None:
            pf.update({'budget': h1.get('budget'),
                       'n_features': h1.get('n_features'),
                       'allocation': h1.get('allocation_method'),
                       'importance': h1.get('importance_method'),
                       'internal_evaluator': h1.get('internal_evaluator'),
                       'rep_count': h1.get('rep_count'),
                       'aug_condition': h1.get('aug_condition'),
                       'hybrid1_train_rows': h1.get('score_train_rows'),
                       'hybrid1_held_out': h1.get('score_held_out')})
            if h1.get('budget') is not None:
                budgets.append(float(h1['budget']))
            if h1.get('n_features') is not None:
                dims.append(float(h1['n_features']))
            h1_out.append(h1.get('score_held_out'))
            h1_tr.append(h1.get('score_train_rows'))
        if b1 is not None:
            pf['base1_rep'] = b1.get('rep')
            pf['base1_held_out'] = b1.get('score_held_out')
            b1_out.append(b1.get('score_held_out'))
        if (h1 is not None and b1 is not None
                and h1.get('score_held_out') is not None
                and b1.get('score_held_out') is not None):
            # Hybrid minus best single representation, raw, in the dataset's
            # metric; read with 'lower_is_better'.
            pf['hybrid1_minus_base1'] = (float(h1['score_held_out'])
                                         - float(b1['score_held_out']))
            deltas.append(pf['hybrid1_minus_base1'])
        per_fold.append(pf)
    if not budgets:
        return None
    return {
        'budget': float(np.median(budgets)),
        'budget_per_fold': budgets,
        'dim': float(np.median(dims)) if dims else None,
        'dim_per_fold': dims,
        'score': _spread(h1_out)['mean'],
        'hybrid1_held_out': _spread(h1_out),
        'hybrid1_train_rows': _spread(h1_tr),
        'base1_held_out': _spread(b1_out),
        'hybrid1_minus_base1': dict(_spread(deltas),
                                    lower_is_better=lower_is_better),
        'per_fold': per_fold,
        'selection_scope': 'fold_train_rows',
    }


def run(datasets, config, output_dir, dataset_name=None, **kwargs):
    """Cross-context analysis across multiple datasets.

    Parameters
    ----------
    datasets : dict
        {dataset_name: data_dict} for each dataset.
    config : dict
        Config dict (fast or full). Only 'n_splits' and 'max_folds' are read;
        each dataset's search grid and models are its decided ones.
    output_dir : str
        Output directory path.
    dataset_name : str, optional
        Ignored (we use datasets dict keys instead).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT C03: CROSS-CONTEXT ANALYSIS")
    print("=" * 70)

    n_splits = int(config['n_splits'])
    max_folds = config.get('max_folds')

    results = {}

    for ds_name, data in datasets.items():
        print(f"\n  --- {ds_name} ---")
        try:
            _skip = should_skip_experiment('c03', data)
            if _skip:
                print(f"    skipped: {_skip['reason']}")
                results[ds_name] = _skip
                continue

            metric_name, metric_fn, lower_is_better = get_domain_metric(data)
            # Each dataset's two decided models (R1); raises when undecided.
            models = [m for m in decided_models(ds_name, 'general')
                      if m != 'knn']
            print(f"    metric {metric_name} "
                  f"({'lower' if lower_is_better else 'higher'} is better); "
                  f"decided models {models}")

            # Dataset complexity profile (a description of the data, fits no
            # model and picks nothing).
            print(f"  Computing complexity metrics...")
            complexity = compute_dataset_complexity(data['reps'], data['labels'])
            mle_str = f", MLE_ID={complexity['mle_id']:.1f}" if complexity.get('mle_id') else ""
            print(f"    erank={complexity['erank']:.1f}, PR={complexity['participation_ratio']:.1f}, "
                  f"n/d={complexity['n_over_d']:.2f}, n/erank={complexity['n_over_erank']:.1f}{mle_str}")
            print(f"    C1={complexity['C1']:.3f}, C2={complexity['C2']:.3f}, "
                  f"L1={complexity['L1']:.3f}, S3={complexity['S3']:.3f}")

            start = time.time()
            folds = nested_cv_outer_folds(
                data['reps'], np.asarray(data['labels']),
                np.asarray(data['groups']), n_splits, FOLD_SEED,
                data['task_type'])
            n_outer = len(folds)
            if n_outer != n_splits:
                print(f"    {n_splits} outer folds asked, {n_outer} built "
                      f"(fewer groups than folds)")
            run_folds = folds if not max_folds else folds[:int(max_folds)]

            ck = Checkpoint(
                experiment_sidecar(output_dir, EXPERIMENT_ID, 'multi',
                                   suffix=ds_name),
                run_identity(EXPERIMENT_ID, ds_name, data,
                             {'n_splits': n_splits},
                             result_version=RESULT_VERSION,
                             selection='fold_standard_selection',
                             models=models,
                             decided_grid=decided_search(ds_name, 'general'),
                             rep_names=sorted(data['reps'])))

            fold_recs = []
            for k, (tr, va) in enumerate(run_folds):
                if ck.done(fold=k):
                    fold_recs.append(ck.get(fold=k))
                    print(f"    fold {k + 1}/{n_outer}: read back from the "
                          f"partial file")
                    continue
                rec = _fold_record(data, ds_name, k, np.asarray(tr),
                                   np.asarray(va), n_outer, models,
                                   metric_name, metric_fn, lower_is_better)
                if not rec['failures']:
                    ck.put(rec, fold=k)
                fold_recs.append(rec)

            per_model = {}
            no_result = []
            for m in models:
                summ = _summarise_model(fold_recs, m, lower_is_better)
                if summ is None:
                    no_result.append(
                        {'downstream_model': m,
                         'reason': 'no fold produced a top hybrid for this '
                                   'model'})
                    print(f"    {m}: no fold produced a top hybrid")
                    continue
                per_model[m] = summ
                _h = summ['hybrid1_held_out']
                _d = summ['hybrid1_minus_base1']
                print(f"    {m:<10} winning budget per fold "
                      f"{summ['budget_per_fold']} features (median "
                      f"{summ['budget']:g}); hybrid1 held-out {metric_name} "
                      f"mean {_h['mean']} over {_h['n_folds']} fold(s); "
                      f"hybrid1 minus base1 {_d['mean']}")

            # Best model on this dataset: the decided model whose top hybrid
            # has the best training-row mean (the fold search's own score),
            # in the dataset's own direction. 2026-10-05: ranked on training
            # rows, not held-out rows, so the held-out mean reported below as
            # best_score is not the number the model was chosen on.
            def _train_mean(v):
                _t = (v.get('hybrid1_train_rows') or {}).get('mean')
                return (float(_t) if _t is not None and np.isfinite(_t)
                        else None)
            _scored = {m: _train_mean(v) for m, v in per_model.items()
                       if _train_mean(v) is not None}
            best_model_name = ((min if lower_is_better else max)(
                _scored, key=lambda m: _scored[m])
                if _scored else None)

            elapsed = time.time() - start
            _bm = per_model.get(best_model_name) or {}
            _bm_f0 = (_bm.get('per_fold') or [{}])[0]
            results[ds_name] = {
                'n_samples': len(data['labels']),
                'elapsed_seconds': elapsed,
                'complexity': complexity,
                'metric': metric_name,
                'lower_is_better': lower_is_better,
                'models': models,
                'representations': sorted(data['reps']),
                'n_outer_folds': n_outer,
                'n_folds_run': len(run_folds),
                'selection_scope': 'fold_train_rows',
                'selection': (f'{STANDARD_N_BASE} base representations + '
                              f'{STANDARD_N_HYBRIDS} general hybrids per '
                              f'decided model per held-out fold'),
                'best_model': best_model_name,
                'best_score': _bm.get('score'),
                # 2026-10-05: best_model is ranked on this, not on best_score.
                'best_selected_on': 'hybrid1_train_rows_mean',
                'best_train_rows_score': _scored.get(best_model_name),
                'best_budget': _bm.get('budget'),
                'best_dim': _bm.get('dim'),
                # The first fold's top-hybrid recipe for the best model; every
                # fold's is under best_by_model[model]['per_fold'].
                'best_allocation': _bm_f0.get('allocation'),
                'best_importance': _bm_f0.get('importance'),
                # One entry per decided model: what the correlation and the
                # model-by-dataset table read.
                'best_by_model': per_model,
                'models_dropped': no_result,
                # Every fold's base representations and hybrids, with
                # training-row and held-out scores.
                'per_fold': fold_recs,
            }

            if best_model_name is not None:
                print(f"    Best: {best_model_name}, "
                      f"{metric_name}={_bm['score'] if _bm['score'] is None else round(_bm['score'], 4)} held out "
                      f"(chosen on training-row mean "
                      f"{_scored[best_model_name]:.4f}), "
                      f"median winning budget={_bm['budget']:g} features")
            else:
                print(f"    No decided model produced a hybrid on {ds_name}.")

        except Exception as e:
            print(f"    {ds_name}: FAILED - {type(e).__name__}: {e}")
            results[ds_name] = {'error': f'{type(e).__name__}: {e}'}

    # Cross-dataset comparison table
    print("\n  CROSS-CONTEXT COMPARISON (median over folds of each fold's top hybrid):")
    print(f"  {'Dataset':<24} {'N':<6} {'Best Model':<12} {'Budget':<8} {'Score':<8}")
    print("  " + "-" * 65)
    for ds_name, r in results.items():
        if 'error' in r or r.get('skipped'):
            continue
        if r.get('best_model') is None:
            print(f"  {ds_name:<24} {r['n_samples']:<6} no decided model "
                  f"produced a hybrid")
            continue
        print(f"  {ds_name:<24} {r['n_samples']:<6} {r['best_model']:<12} "
              f"{r['best_budget']:<8g} "
              + (f"{r['best_score']:.4f}" if r['best_score'] is not None
                 else '-'))

    # --- Dataset complexity vs optimal hybrid dimension ---
    valid = {ds: r for ds, r in results.items()
             if 'error' not in r and not r.get('skipped')
             and r.get('best_budget') is not None}
    if len(valid) >= 2:
        print("\n  DATASET COMPLEXITY vs OPTIMAL BUDGET:")
        print(f"  {'Dataset':<24} {'erank':>7} {'PR':>7} {'n/erank':>8} {'C1':>6} {'C2':>6} "
              f"{'L1':>6} {'S3':>6} {'Budget':>7} {'Dim':>7}")
        print("  " + "-" * 96)

        complexity_keys = ['erank', 'participation_ratio', 'n_over_erank', 'C1', 'C2', 'L1', 'S3']
        ds_names = sorted(valid.keys())

        for ds in ds_names:
            c = valid[ds]['complexity']
            _dim = valid[ds]['best_dim']
            print(f"  {ds:<24} {c['erank']:>7.1f} {c['participation_ratio']:>7.1f} "
                  f"{c['n_over_erank']:>8.1f} {c['C1']:>6.3f} {c['C2']:>6.3f} "
                  f"{c['L1']:>6.3f} {c['S3']:>6.3f} "
                  f"{valid[ds]['best_budget']:>7g} "
                  f"{(f'{_dim:g}' if _dim is not None else '-'):>7}")

        # 2026-09-12 — DECIDED: one Spearman coefficient per (descriptor,
        # downstream model). A model enters only on the datasets where it is
        # one of the two decided models (2026-10-05).
        # 2026-10-05: no p-values (significance tests excluded everywhere).
        corr_models = sorted({m for ds in ds_names
                              for m in (valid[ds].get('best_by_model') or {})})
        correlations_by_model = {}
        print("\n  Spearman correlation, complexity descriptor -> winning budget.")
        print("  One coefficient per (descriptor, downstream model). One point is "
              "one dataset; n counts the datasets that have a winning budget for "
              "that model. The winning budget is the median over folds of each "
              "fold's top hybrid's budget, in features.")
        for model in corr_models:
            model_ds = [
                ds for ds in ds_names
                if (valid[ds].get('best_by_model') or {}).get(model, {})
                .get('budget') is not None
            ]
            if len(model_ds) < 2:
                print(f"    {model:<12} {len(model_ds)} dataset(s) with a winning "
                      f"budget — no coefficient")
                correlations_by_model[model] = {
                    'n_datasets': len(model_ds), 'datasets': model_ds,
                    'descriptors': {},
                }
                continue

            m_budgets = np.array(
                [float(valid[ds]['best_by_model'][model]['budget'])
                 for ds in model_ds])
            _dim_vals = [valid[ds]['best_by_model'][model].get('dim')
                         for ds in model_ds]
            m_dims = (np.array([float(d) for d in _dim_vals])
                      if all(d is not None for d in _dim_vals) else None)

            per_desc = {}
            print(f"    {model}  (n={len(model_ds)} datasets: "
                  f"{', '.join(model_ds)})")
            for key in complexity_keys:
                values = np.array([float(valid[ds]['complexity'][key])
                                   for ds in model_ds])
                if np.std(values) < 1e-10:
                    continue
                rho, _ = spearmanr(values, m_budgets)
                entry = {'rho_vs_budget': float(rho),
                         'n_datasets': len(model_ds)}
                # The same descriptor against the feature count the winning
                # hybrids actually reached (median over folds).
                if m_dims is not None:
                    rho_dim, _ = spearmanr(values, m_dims)
                    entry['rho_vs_dim'] = float(rho_dim)
                per_desc[key] = entry
                _dim_str = (f", features: rho={entry['rho_vs_dim']:+.3f}"
                            if 'rho_vs_dim' in entry else "")
                print(f"      {key:<20} -> budget: rho={rho:+.3f}{_dim_str}")
            correlations_by_model[model] = {
                'n_datasets': len(model_ds), 'datasets': model_ds,
                'descriptors': per_desc,
            }

        results['_complexity_correlation_by_model'] = correlations_by_model

        # erank against the winning budget, one coefficient per downstream model.
        print("\n  erank -> winning budget, one coefficient per downstream model:")
        for model in corr_models:
            entry = (correlations_by_model.get(model, {})
                     .get('descriptors', {}).get('erank'))
            n_ds = correlations_by_model.get(model, {}).get('n_datasets', 0)
            if entry is None:
                print(f"    {model:<12} no coefficient (n={n_ds} datasets)")
            else:
                print(f"    {model:<12} rho={entry['rho_vs_budget']:+.3f}, "
                      f"n={n_ds} datasets")

        # Per-model optimal budget comparison across datasets. A dash is a
        # model that is not one of that dataset's decided models, or that
        # produced no hybrid there.
        model_names = set()
        for ds in ds_names:
            model_names.update(valid[ds].get('best_by_model', {}).keys())
        if model_names:
            print("\n  PER-MODEL OPTIMAL BUDGET (cross-dataset; median "
                  "budget in features, held-out score):")
            print(f"  {'Model':<12}" + "".join(f"{ds:>24}" for ds in ds_names))
            print("  " + "-" * (12 + 24 * len(ds_names)))
            for model in sorted(model_names):
                row = f"  {model:<12}"
                for ds in ds_names:
                    bm = valid[ds].get('best_by_model', {}).get(model)
                    if bm and bm.get('score') is not None:
                        _cell = (f"b={bm['budget']:g} "
                                 f"score={bm['score']:.3f}")
                        row += f"{_cell:>24}"
                    else:
                        row += f"{'--':>24}"
                print(row)

            # MODEL x DATASET INTERACTION ANALYSIS
            # 2026-10-05: each dataset has its own two decided models, so a
            # full model x dataset table exists only for the models decided on
            # every dataset in the run. The decomposition uses those; the rest
            # are named.
            model_list = sorted(
                m for m in model_names
                if all(m in (valid[ds].get('best_by_model') or {})
                       for ds in ds_names))
            left_out = sorted(model_names - set(model_list))
            if left_out:
                print(f"\n  Budget variance decomposition uses the models with "
                      f"a winning budget on all {len(ds_names)} datasets: "
                      f"{model_list or 'none'}; left out {left_out}.")
            budget_matrix = np.array(
                [[valid[ds]['best_by_model'][m]['budget'] for ds in ds_names]
                 for m in model_list], dtype=float).reshape(len(model_list),
                                                            len(ds_names))
            score_matrix = np.array(
                [[valid[ds]['best_by_model'][m]['score'] if
                  valid[ds]['best_by_model'][m]['score'] is not None else np.nan
                  for ds in ds_names]
                 for m in model_list], dtype=float).reshape(len(model_list),
                                                           len(ds_names))

            # Budget variance decomposition
            if budget_matrix.shape[0] >= 2 and budget_matrix.shape[1] >= 2:
                grand_mean = np.mean(budget_matrix)
                row_means = np.mean(budget_matrix, axis=1)
                col_means = np.mean(budget_matrix, axis=0)
                ss_model = len(ds_names) * np.sum((row_means - grand_mean) ** 2)
                ss_dataset = len(model_list) * np.sum((col_means - grand_mean) ** 2)
                ss_total = np.sum((budget_matrix - grand_mean) ** 2)
                ss_interaction = ss_total - ss_model - ss_dataset

                if ss_total > 0:
                    print(f"\n  BUDGET VARIANCE DECOMPOSITION:")
                    print(f"    Model effect:       {ss_model/ss_total*100:5.1f}% of variance")
                    print(f"    Dataset effect:     {ss_dataset/ss_total*100:5.1f}% of variance")
                    print(f"    Interaction effect: {ss_interaction/ss_total*100:5.1f}% of variance")

                results['_budget_variance'] = {
                    'models': model_list,
                    'models_left_out': left_out,
                    'datasets': ds_names,
                    'ss_model': float(ss_model),
                    'ss_dataset': float(ss_dataset),
                    'ss_interaction': float(ss_interaction),
                    'ss_total': float(ss_total),
                    'pct_model': float(ss_model / ss_total * 100) if ss_total > 0 else 0,
                    'pct_dataset': float(ss_dataset / ss_total * 100) if ss_total > 0 else 0,
                    'pct_interaction': float(ss_interaction / ss_total * 100) if ss_total > 0 else 0,
                }

                # Does budget range differ across models?
                print(f"\n  MODEL DIMENSION SENSITIVITY:")
                print(f"  {'Model':<12} {'Budget Range':<15} {'Budget Std':<12} {'Score Range':<12}")
                print("  " + "-" * 51)
                for i, model in enumerate(model_list):
                    b_row = budget_matrix[i]
                    score_row = score_matrix[i]
                    print(f"  {model:<12} {int(b_row.min())}-{int(b_row.max()):<10} "
                          f"{np.std(b_row):>8.1f}    {np.nanmax(score_row) - np.nanmin(score_row):>8.4f}")
            else:
                print("\n  Budget variance decomposition: needs at least 2 models "
                      "with a winning budget on every dataset and 2 datasets; "
                      f"have {budget_matrix.shape[0]} model(s), "
                      f"{budget_matrix.shape[1]} dataset(s).")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C03: Cross-Dataset Transfer')
    parser.add_argument('--datasets', type=str, nargs='+', default=['logd', 'caco2', 'herg_ki'],
                        help='Dataset names to compare (at least 2)')
    args = parser.parse_args()
    config = get_config(args)
    if getattr(args, 'models', None) or getattr(args, 'budgets', None):
        print("  --models / --budgets ignored: each dataset uses its own "
              "decided models and budgets (DECISIONS.md).")
    if args.reps != 'fast':
        print("  --reps ignored: each dataset uses DATASET_REPS[ds]['general'].")

    print(f"Mode: {args.mode} ({config['n_splits']} outer fold(s))")
    print(f"Datasets: {args.datasets}")

    from experiments.complexity.complexity_helpers import DATASET_REPS
    from experiments._local_grid import cut_pool

    # Load all datasets, each with its pruned representations only (handoff
    # 2.4 step 9). No augmentation representations: c03 runs the general
    # search only, where Step 8 cut augmentation on every dataset.
    all_datasets = {}
    for ds_name in args.datasets:
        print(f"\nLoading {ds_name}...")
        try:
            if ds_name not in DATASET_REPS:
                raise KeyError(f'{ds_name} has no DATASET_REPS entry')
            rep_names = cut_pool(ds_name, list(DATASET_REPS[ds_name]['general']))
            data = load_data(ds_name, rep_preset='all', rep_names=rep_names)
            # Same NaN guard as complexity_helpers.run_all; keeps float32.
            for rname, X in list(data['reps'].items()):
                if np.isnan(X).any():
                    print(f"  WARNING: {rname} has {int(np.isnan(X).sum())} "
                          f"NaN values — imputing 0")
                    data['reps'][rname] = np.nan_to_num(X, nan=0.0)
            data['augmentations'] = None
            all_datasets[ds_name] = data
        except Exception as e:
            print(f"  FAILED: {e}")

    if len(all_datasets) < 2:
        print("Need at least 2 datasets for cross-context analysis.")
        sys.exit(1)

    results = run(all_datasets, config, args.output)
    save_experiment_results(results, args.output, EXPERIMENT_ID, 'multi')
