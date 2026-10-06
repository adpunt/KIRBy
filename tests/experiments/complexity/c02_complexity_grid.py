#!/usr/bin/env python3
"""C02: Model-budget complexity grid -- tests optimal pairing hypothesis (CORE).

Paper: Complexity
Conclusion: Different models have different optimal representation budgets.
Dependencies: None (standalone). c08 and c09 read this file's result JSON.

Aligned 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4). Per outer fold:

  * one hybrid search on that fold's TRAINING rows, on the dataset's decided
    grid (DECISIONS.md: its two models, budgets, importance methods,
    allocation methods, internal evaluators, representation counts), pruning
    off (_shared.fold_standard_selection -> fold_searches -> search_on_fold);
  * from that search, per decided model: the best training-row score at each
    decided budget (the model x budget heatmap), the best configuration, which
    importance method it used, and the training-row ranking of the base
    representations;
  * the standard selection: per decided model, the 2 best base
    representations and the 2 best hybrids (the second skipped when its
    selected features equal the first's), each fitted on the training rows
    and scored on the held-out rows; hybrids are rebuilt with the fold
    search's own per-representation scores;
  * PCA baselines: the concatenated representations reduced to each decided
    budget, fitted on the training rows, scored on the held-out rows.

Nothing is searched on the whole dataset. The number of outer folds is
config['n_splits'] (1 = one held-out split).

DECIDED 2026-10-06 (Adelaide delegated): the search with fine-tuned
representations (13 datasets) and the search at a single representation's own
width (e.g. 1,280 features on flab) stay dropped; see ARMS_NOT_RUN.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.decomposition import PCA

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, get_domain_metric,
    top_hybrids, hybrid_deltas,
    fold_standard_selection, fold_searches,
    build_fold_hybrid, fold_hybrid_scoring_kwargs,
    parse_rep_args,
)
from kirby.find_best_hybrid import (
    _predict_for_metric, nested_cv_outer_folds, nested_cv_cache_namespace)
# 0/1 features are never scaled, continuous ones are standardised (Adelaide,
# 2026-10-05; handoff 2.3 item 17). Replaces this file's StandardScaler.
from kirby.scaling import FeatureScaler

EXPERIMENT_DIR = 'c02_complexity_matching_grid'
SEED = 42
# The result identity. 2026-10-05: per-fold searches and the standard
# selection replaced the whole-dataset searches, so a partial file written
# under any other format is moved aside, never read back.
RESULT_FORMAT = 'per_fold_standard_selection_2026_10_05'

# Fields of a hybrid configuration written for every hybrid (2026-10-05).
HYBRID_FIELDS = ('label', 'allocation_method', 'importance_method', 'budget',
                 'internal_evaluator', 'rep_count', 'step_size', 'n_features')

# Arms the whole-dataset version ran that the decided rules do not allow.
# Each is recorded in the result file with its reason; none is run.
# DECIDED 2026-10-06 (Adelaide delegated): the search with fine-tuned
# representations and the search at a single representation's own width stay
# dropped.
ARMS_NOT_RUN = {
    'frozen_plus_ft': (
        'fine-tuned representations are not in DATASET_REPS[dataset]'
        "['general'] or ['augmentation'], the only representations an "
        'experiment may use (handoff 2.1); not run'),
    'per_model_top_rep_dim': (
        "a search at the width of a model's best base representation uses a "
        'budget outside the decided budgets (rule R3, handoff 2.1); not run'),
    'capped': ('augmentation condition capped is cut on every dataset '
               '(DECISIONS.md Step 8); not run'),
    'rule_hybrids_distance1': (
        'c02 measures nothing with neighbours or distances, so it uses the 2 '
        'general hybrids only (handoff 2.3 item 1); no distance hybrid'),
}


def _no_blueprint(obj):
    """Drop Blueprint objects (and anything else not plain) from a block
    before it is written; every recipe field is already a key beside it."""
    if isinstance(obj, dict):
        return {k: _no_blueprint(v) for k, v in obj.items()
                if k not in ('blueprint', 'search')}
    if isinstance(obj, (list, tuple)):
        return [_no_blueprint(v) for v in obj]
    return obj


def _plain(obj):
    """_no_blueprint, then numpy values to Python ones. Unlike convert_numpy
    it keeps dict keys as they are, so the integer budget keys this file
    reads back (heatmap, pca) stay integers in memory and in the per-fold
    partial file."""
    obj = _no_blueprint(obj)
    if isinstance(obj, dict):
        return {(k.item() if isinstance(k, np.generic) else k): _plain(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return convert_numpy(obj)


def _fit_score(X_tr, X_va, y_tr, y_va, factory, metric_fn):
    """Fit one model on a fold's training rows, score it on its held-out rows.

    Features: NaN -> 0 (as the PCA baseline always did), scaled with
    FeatureScaler fitted on the training rows (0/1 features untouched), and
    each held-out feature capped at the range it takes on the scaled training
    rows (owner delegated, 2026-10-05; on adme_hlm one held-out molecule sat
    83 million training standard deviations out). No float64 conversion.
    """
    X_tr = np.nan_to_num(np.asarray(X_tr), nan=0.0)
    X_va = np.nan_to_num(np.asarray(X_va), nan=0.0)
    scaler = FeatureScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_va = np.clip(X_va, X_tr.min(axis=0), X_tr.max(axis=0))
    m = factory()
    m.fit(X_tr, y_tr)
    return float(metric_fn(y_va, _predict_for_metric(m, X_va, metric_fn)))


def _over_folds(values):
    """Mean and raw spread over outer folds. No interval, no test."""
    vals = [float(v) for v in values
            if v is not None and np.isfinite(v)]
    if not vals:
        return {'mean': None, 'min': None, 'max': None, 'n_folds': 0}
    out = {'mean': float(np.mean(vals)), 'n_folds': len(vals)}
    if len(vals) == 1:
        out.update(min=None, max=None, spread='one fold, no spread')
    else:
        out.update(min=float(np.min(vals)), max=float(np.max(vals)))
    return out


def _better(a, b, lower):
    """+ when a is better than b, on either metric direction."""
    if a is None or b is None:
        return None
    return float(b - a) if lower else float(a - b)


def _hybrid_row(e):
    return {f: e.get(f) for f in HYBRID_FIELDS}


def _build_and_score(e, tr_reps, va_reps, y_tr, y_va, g_tr, augs, tr, va,
                     factory, metric_fn, lower_is_better):
    """Rebuild one hybrid from its per-fold recipe and score it held out."""
    bp = e['blueprint']
    cond = e.get('aug_condition') or 'none'
    aug_tr = aug_va = None
    if cond != 'none' and augs:
        aug_tr = {k: np.asarray(v)[tr] for k, v in augs.items()}
        aug_va = {k: np.asarray(v)[va] for k, v in augs.items()}
    X_tr, X_va, fi = build_fold_hybrid(
        tr_reps, va_reps, y_tr, bp,
        augmentations_train=aug_tr, augmentations_val=aug_va,
        groups=g_tr,
        # Dirty-run bug 1: the representation scores come from this fold's
        # own search (the blueprint was built on this fold's training rows),
        # never from a whole-dataset search.
        precomputed_alloc_scores=getattr(bp, 'rep_scores', None),
        **fold_hybrid_scoring_kwargs(bp, model=factory(), metric_fn=metric_fn,
                                     lower_is_better=lower_is_better))
    features_by_rep = {r: int(len(info['selected_indices']))
                       for r, info in fi.items()
                       if isinstance(info, dict) and 'selected_indices' in info}
    score = _fit_score(X_tr, X_va, y_tr, y_va, factory, metric_fn)
    return score, int(np.asarray(X_tr).shape[1]), features_by_rep


def _config_key(r):
    bp = r.get('blueprint')
    rc = r.get('rep_count', getattr(bp, 'rep_count', None))
    return (r.get('model'), r.get('allocation_method'),
            r.get('importance_method'), r.get('budget'),
            r.get('internal_evaluator'), rc, r.get('rep_subset', 'all'),
            r.get('step_size'))


def _run_fold(k, tr, va, *, reps, labels, groups, augs, dataset_name,
              n_outer, models, factories, budgets, general_names, metric_name,
              metric_fn, lower_is_better, task_type, X_concat, aug_tiers,
              dist_factories):
    """Everything c02 measures on one outer fold, as plain data."""
    ns = nested_cv_cache_namespace(k, n_outer, SEED)
    sel = fold_standard_selection(
        reps, labels, groups, train_idx=tr, fold_i=k, n_outer_folds=n_outer,
        dataset_name=dataset_name, models=list(models), augmentations=augs,
        task_type=task_type, metric=metric_name, seed=SEED)
    search = next((s['search'] for s in sel.values() if s is not None), None)
    rec = {'fold': k, 'n_train': int(len(tr)), 'n_val': int(len(va)),
           'cache_namespace': ns, 'selection_scope': 'fold_train_rows',
           'search_failed_models': sorted(m for m, s in sel.items()
                                          if s is None)}
    if search is None:
        rec['error_search'] = ("the search on this fold's training rows "
                               "failed for every model")
        print(f"  fold {k}: {rec['error_search']}")
        return rec

    # --- The grid, read off this fold's search (training rows only) --------
    configs = []
    for r in search.all_results:
        key = _config_key(r)
        configs.append({
            'model': key[0], 'allocation': key[1], 'importance': key[2],
            'budget': key[3], 'internal_evaluator': key[4], 'rep_count': key[5],
            'rep_subset': key[6], 'step_size': key[7],
            'score_mean': r.get('score_mean'), 'score_std': r.get('score_std'),
            'fold_scores': list(r.get('fold_scores') or []),
            'n_features': r.get('n_features'),
        })
    rec['configs'] = configs

    heatmap, best_by_model, baselines = {}, {}, {}
    for m in models:
        mc = [c for c in configs if c['model'] == m
              and c['score_mean'] is not None and np.isfinite(c['score_mean'])]
        heatmap[m] = {}
        for b in budgets:
            bc = [c for c in mc if c['budget'] == b]
            if bc:
                best = (min if lower_is_better else max)(
                    bc, key=lambda c: c['score_mean'])
                heatmap[m][b] = {'score': best['score_mean'],
                                 'n_features': best['n_features']}
        r = search.best_by_model.get(m)
        if r is not None:
            best_by_model[m] = {
                'model': m, 'allocation': r['allocation_method'],
                'importance': r['importance_method'], 'budget': r['budget'],
                'internal_evaluator': r.get('internal_evaluator'),
                'rep_count': _config_key(r)[5],
                'score_mean': r['score_mean'], 'score_std': r['score_std'],
                'n_features': r['n_features']}
        # Base representations ranked by this model's training-row score.
        # A per-representation loop, so it is filtered to the dataset's
        # general representations (handoff 2.4 step 9).
        ranked = [(rn, v.get(m)) for rn, v in search.baselines.items()
                  if rn in general_names]
        ranked = [(rn, float(s)) for rn, s in ranked
                  if s is not None and np.isfinite(s)]
        ranked.sort(key=lambda kv: ((kv[1] if lower_is_better else -kv[1]),
                                    kv[0]))
        baselines[m] = {'ranked_reps': [rn for rn, _ in ranked],
                        'ranked_scores': [s for _, s in ranked]}
    rec['heatmap'] = heatmap
    rec['best_by_model'] = best_by_model
    rec['baselines'] = baselines

    # --- Standard selection: 2 base + 2 hybrids per model, held out --------
    y = np.asarray(labels)
    y_tr, y_va = y[tr], y[va]
    g_tr = None if groups is None else np.asarray(groups)[tr]
    tr_reps = {r: np.asarray(X)[tr] for r, X in reps.items()}
    va_reps = {r: np.asarray(X)[va] for r, X in reps.items()}
    sel_out = {}
    for m, s in sel.items():
        if s is None:
            sel_out[m] = None
            continue
        factory = factories.get(m)
        base_rows = []
        for i, e in enumerate(s['base_detail']['selected'], 1):
            row = {'slot': f'base{i}', 'rep': e['rep'],
                   'n_features': int(np.asarray(reps[e['rep']]).shape[1]),
                   'train_score': e.get('score'), 'heldout_score': None}
            if factory is None:
                row['error'] = f'no model factory for {m}'
            else:
                try:
                    row['heldout_score'] = _fit_score(
                        tr_reps[e['rep']], va_reps[e['rep']], y_tr, y_va,
                        factory, metric_fn)
                except Exception as ex:
                    row['error'] = f'{type(ex).__name__}: {ex}'
            base_rows.append(row)
        hyb_rows = []
        for i, e in enumerate(s['hybrids'], 1):
            row = {'slot': f'hybrid{i}', **_hybrid_row(e),
                   'aug_condition': e.get('aug_condition') or 'none',
                   'train_score': e.get('score_mean'), 'heldout_score': None}
            if factory is None:
                row['error'] = f'no model factory for {m}'
            else:
                try:
                    (row['heldout_score'], row['n_features_built'],
                     row['features_by_rep']) = _build_and_score(
                        e, tr_reps, va_reps, y_tr, y_va, g_tr, augs, tr, va,
                        factory, metric_fn, lower_is_better)
                except Exception as ex:
                    row['error'] = f'{type(ex).__name__}: {ex}'
            hyb_rows.append(row)
        # The two reference numbers for hybrid 1, on this fold's training
        # rows (its search's own scores): against the mean of the
        # representations that fed it, and against the best one.
        deltas = None
        if s['hybrids']:
            try:
                deltas = _plain(hybrid_deltas(search, m,
                                              hybrid=s['hybrids'][0]))
            except Exception as ex:
                deltas = {'error': f'{type(ex).__name__}: {ex}'}
        b1 = base_rows[0]['heldout_score'] if base_rows else None
        h1 = hyb_rows[0]['heldout_score'] if hyb_rows else None
        sel_out[m] = {
            'base': base_rows, 'hybrids': hyb_rows,
            'base_shortfall': s['base_detail'].get('shortfall_reason'),
            'hybrids_skipped_identical_features': [
                x.get('label') for x in
                (s.get('hybrids_skipped_identical_features') or [])],
            'hybrid1_minus_base1_heldout': _better(h1, b1, lower_is_better),
            'hybrid1_train_row_deltas': deltas,
        }
        for row in base_rows + hyb_rows:
            sc = row['heldout_score']
            print(f"    fold {k} {m:8s} {row['slot']:8s} "
                  f"{(row.get('rep') or row.get('label') or ''):40.40s} "
                  f"train={row['train_score'] if row['train_score'] is None else round(row['train_score'], 4)} "
                  f"held-out={'--' if sc is None else round(sc, 4)}"
                  f"{'  ' + row['error'] if row.get('error') else ''}")
    rec['standard_selection'] = sel_out

    # --- PCA baselines at each decided budget, held out --------------------
    pca = {}
    for b in budgets:
        pca[b] = {}
        n_comp = min(int(b), X_concat.shape[1], len(tr) - 1)
        for m, factory in factories.items():
            ent = {'n_components': int(n_comp), 'heldout_score': None}
            if n_comp < 1:
                ent['error'] = 'fewer than 2 training rows'
                pca[b][m] = ent
                continue
            try:
                scaler = FeatureScaler()
                X_tr_s = scaler.fit_transform(X_concat[tr])
                X_va_s = scaler.transform(X_concat[va])
                # Owner delegated, 2026-10-05: held-out features capped at the
                # scaled training range before the projection.
                X_va_s = np.clip(X_va_s, X_tr_s.min(0), X_tr_s.max(0))
                p = PCA(n_components=n_comp)
                X_tr_p = p.fit_transform(X_tr_s)
                X_va_p = p.transform(X_va_s)
                mdl = factory()
                mdl.fit(X_tr_p, y_tr)
                ent['heldout_score'] = float(metric_fn(
                    y_va, _predict_for_metric(mdl, X_va_p, metric_fn)))
            except Exception as ex:
                ent['error'] = f'{type(ex).__name__}: {ex}'
            pca[b][m] = ent
    rec['pca'] = pca

    # --- Augmentation comparison, only where DECISIONS.md Step 8 keeps it ---
    aug_cmp = {}
    for tier in aug_tiers:
        try:
            srch = fold_searches(
                reps, labels, groups, train_idx=tr, fold_i=k,
                n_outer_folds=n_outer, dataset_name=dataset_name,
                augmentations=augs, task_type=task_type, metric=metric_name,
                tier=tier, seed=SEED)
        except Exception as ex:
            aug_cmp[tier] = {'error': f'{type(ex).__name__}: {ex}'}
            continue
        tier_factories = dist_factories if 'distance' in tier else factories
        by_model = {}
        for m, factory in tier_factories.items():
            by_model[m] = {}
            for cond, sr in srch.items():
                if sr is None:
                    by_model[m][cond] = {'error': 'search failed'}
                    continue
                th = top_hybrids(sr, m, n=1, require_full=False,
                                 context=f'{dataset_name} {tier} fold {k} '
                                         f'{m} {cond}')
                if not th['selected']:
                    by_model[m][cond] = {'error': th.get('shortfall_reason')}
                    continue
                e = dict(th['selected'][0], aug_condition=cond)
                row = {**_hybrid_row(e), 'aug_condition': cond,
                       'train_score': e.get('score_mean'),
                       'heldout_score': None}
                try:
                    (row['heldout_score'], row['n_features_built'],
                     row['features_by_rep']) = _build_and_score(
                        e, tr_reps, va_reps, y_tr, y_va, g_tr, augs, tr, va,
                        factory, metric_fn, lower_is_better)
                except Exception as ex:
                    row['error'] = f'{type(ex).__name__}: {ex}'
                by_model[m][cond] = row
        aug_cmp[tier] = {'conditions': sorted(srch.keys()),
                         'by_model': by_model,
                         'failures': [
                             {fk: (fv if isinstance(fv, (str, int, float,
                                                         type(None)))
                                   else str(fv)) for fk, fv in f.items()}
                             for f in (getattr(srch, 'failures', None) or [])]}
    rec['augmentation_comparison'] = aug_cmp
    return _plain(rec)


def run(data, config, output_dir, dataset_name, **kwargs):
    """Model x budget complexity grid, per outer fold."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C02: COMPLEXITY MATCHING GRID (per fold)")
    print("=" * 70)

    from experiments.decided_grid import (
        decided_models, decided_search, decided_aug_conditions, is_decided)
    from experiments.complexity.complexity_helpers import DATASET_REPS
    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data['task_type']
    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data['groups']
    augs = data.get('augmentations') or None

    # Models: the dataset's two decided models (DECISIONS.md Step 1). knn is
    # never a general model here.
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    factories = get_model_factories(models, task_type=task_type,
                                    n_samples=len(labels))
    models_dropped = [{'downstream_model': m,
                       'reason': 'no model factory at this dataset size'}
                      for m in models if m not in factories]
    budgets = list(decided_search(dataset_name, 'general')['budgets'])

    # Representations: only the dataset's general list (+ augmentation where
    # Step 8 keeps it). data['reps'] goes to the search unfiltered; the
    # per-representation loops (base ranking, PCA) are filtered.
    general_names = list((DATASET_REPS.get(dataset_name) or {}).get(
        'general') or reps.keys())
    extra = sorted(set(reps) - set(general_names))
    if extra:
        print(f"  representations outside DATASET_REPS['{dataset_name}']"
              f"['general'] left out of the base ranking and PCA: {extra}")
    pca_names = sorted(r for r in reps if r in general_names)
    X_concat = np.hstack([np.asarray(reps[r], dtype=np.float32)
                          for r in pca_names])
    X_concat = np.nan_to_num(np.clip(X_concat, -1e10, 1e10), nan=0.0)

    # Augmentation comparison (2.3 item 4): only in a search where Step 8
    # keeps a condition besides 'none' -- the distance search of
    # ginkgo_colloidal and ginkgo_hic today. Skipped elsewhere, with the reason.
    aug_tiers, aug_skip_reason = [], None
    if not augs:
        aug_skip_reason = 'the dataset has no augmentation representations'
    else:
        for tier in ('general', 'general_distance'):
            if (is_decided(dataset_name, tier) and
                    [c for c in decided_aug_conditions(dataset_name, tier)
                     if c != 'none']):
                aug_tiers.append(tier)
        if not aug_tiers:
            aug_skip_reason = ('DECISIONS.md Step 8 cuts augmentation on this '
                               "dataset's general and distance searches")
    dist_factories = {}
    if 'general_distance' in aug_tiers:
        dist_factories = get_model_factories(
            decided_models(dataset_name, 'general_distance'),
            task_type=task_type, n_samples=len(labels))
    if aug_skip_reason:
        print(f"  augmentation comparison skipped: {aug_skip_reason}")
    else:
        print(f"  augmentation comparison on: {aug_tiers}")
    for arm, why in ARMS_NOT_RUN.items():
        print(f"  {arm}: {why}")

    n_requested = int(config['n_splits'])
    folds = nested_cv_outer_folds(reps, labels, groups, n_requested, SEED,
                                  task_type)
    n_outer = len(folds)
    print(f"  models {models}; budgets {budgets} features; {len(reps)} "
          f"representations; {n_outer} outer fold(s) (asked {n_requested})")

    ck = Checkpoint(
        experiment_sidecar(output_dir, EXPERIMENT_DIR, dataset_name),
        run_identity(EXPERIMENT_DIR, dataset_name, data, config,
                     result_format=RESULT_FORMAT))

    per_fold = []
    for k, (tr, va) in enumerate(folds):
        if ck.done(fold=k):
            rec = ck.get(fold=k)
            print(f"  fold {k}: finished in an earlier attempt, read back")
        else:
            print(f"\n  --- outer fold {k + 1}/{n_outer}: {len(tr)} training "
                  f"rows, {len(va)} held out ---")
            rec = _run_fold(
                k, tr, va, reps=reps, labels=labels, groups=groups, augs=augs,
                dataset_name=dataset_name, n_outer=n_outer, models=models,
                factories=factories, budgets=budgets,
                general_names=general_names, metric_name=metric_name,
                metric_fn=metric_fn, lower_is_better=lower_is_better,
                task_type=task_type, X_concat=X_concat, aug_tiers=aug_tiers,
                dist_factories=dist_factories)
            if 'error_search' not in rec:
                ck.put(rec, fold=k)
        per_fold.append(rec)

    ok = [r for r in per_fold if 'error_search' not in r]

    # --- Summaries over outer folds ----------------------------------------
    # configs: one entry per configuration (as c09 reads it), its score the
    # mean over outer folds of each fold's training-row score;
    # 'fold_scores' are those per-outer-fold training-row scores.
    agg = {}
    for r in ok:
        for c in r.get('configs', []):
            key = (c['model'], c['allocation'], c['importance'], c['budget'],
                   c['internal_evaluator'], c['rep_count'], c['rep_subset'],
                   c['step_size'])
            a = agg.setdefault(key, {
                'model': c['model'], 'allocation': c['allocation'],
                'importance': c['importance'], 'budget': c['budget'],
                'internal_evaluator': c['internal_evaluator'],
                'rep_count': c['rep_count'], 'rep_subset': c['rep_subset'],
                'step_size': c['step_size'], 'fold_scores': [],
                'outer_folds': [], '_nf': []})
            if c['score_mean'] is not None and np.isfinite(c['score_mean']):
                a['fold_scores'].append(float(c['score_mean']))
                a['outer_folds'].append(r['fold'])
            if c['n_features'] is not None:
                a['_nf'].append(float(c['n_features']))
    configs = []
    for a in agg.values():
        fs = a['fold_scores']
        a['score_mean'] = float(np.mean(fs)) if fs else float('nan')
        a['score_std'] = float(np.std(fs)) if len(fs) > 1 else None
        a['n_features'] = float(np.mean(a['_nf'])) if a['_nf'] else None
        del a['_nf']
        configs.append(a)

    best_by_model = {}
    for m in models:
        mc = [c for c in configs if c['model'] == m
              and np.isfinite(c['score_mean'])
              and len(c['fold_scores']) == len(ok)]
        if mc:
            b = (min if lower_is_better else max)(
                mc, key=lambda c: c['score_mean'])
            best_by_model[m] = {**{k_: b[k_] for k_ in (
                'model', 'allocation', 'importance', 'budget',
                'internal_evaluator', 'rep_count', 'score_mean', 'score_std',
                'n_features')},
                'selection_scope': ('mean over outer folds of each fold\'s '
                                    'training-row score')}

    baselines = {}
    for m in models:
        per_rep = {}
        for r in ok:
            bl = r['baselines'].get(m) or {}
            for rn, sc in zip(bl.get('ranked_reps', []),
                              bl.get('ranked_scores', [])):
                per_rep.setdefault(rn, []).append(sc)
        ranked = sorted(((rn, float(np.mean(v))) for rn, v in per_rep.items()
                         if len(v) == len(ok)),
                        key=lambda kv: ((kv[1] if lower_is_better else -kv[1]),
                                        kv[0]))
        top = ranked[0][1] if ranked else None
        baselines[m] = {
            'tied_at_best_single': sorted(rn for rn, s in ranked if s == top),
            'best_single_rep': ranked[0][0] if ranked else None,
            'best_single_score': top,
            'all_reps_failed': not ranked,
            'ranked_reps': [rn for rn, _ in ranked],
            'score_basis': ('mean over outer folds of each fold\'s '
                            'training-row score'),
        }

    heatmap, heatmap_by_fold = {}, {}
    for m in models:
        heatmap[m], heatmap_by_fold[m] = {}, {}
        for b in budgets:
            vals = [(r['fold'], r['heatmap'].get(m, {}).get(b)) for r in ok]
            heatmap_by_fold[m][b] = {f: (v or {}).get('score') for f, v in vals}
            sc = [v['score'] for _, v in vals if v]
            if sc and len(sc) == len(ok):
                heatmap[m][b] = float(np.mean(sc))

    importance_methods = list(decided_search(dataset_name,
                                             'general')['importance_methods'])
    importance_wins = {imp: 0 for imp in importance_methods}
    for r in ok:
        for m, bb in r['best_by_model'].items():
            if bb['importance'] in importance_wins:
                importance_wins[bb['importance']] += 1

    pca_baselines, pca_widths, pca_fold_scores = {}, {}, {}
    for b in budgets:
        pca_baselines[b], pca_widths[b], pca_fold_scores[b] = {}, {}, {}
        for m in factories:
            ents = [r['pca'].get(b, {}).get(m) or {} for r in ok]
            sc = [e.get('heldout_score') for e in ents]
            pca_fold_scores[b][m] = sc
            pca_widths[b][m] = [e.get('n_components') for e in ents]
            good = [s for s in sc if s is not None]
            pca_baselines[b][m] = float(np.mean(good)) if good else float('nan')

    selection_summary, heldout_delta = {}, {}
    for m in models:
        slots = {}
        deltas = []
        for r in ok:
            s = (r.get('standard_selection') or {}).get(m)
            if not s:
                continue
            for row in s['base'] + s['hybrids']:
                d = slots.setdefault(row['slot'], {'per_fold': []})
                d['per_fold'].append({
                    'fold': r['fold'],
                    'name': row.get('rep') or row.get('label'),
                    'train_score': row.get('train_score'),
                    'heldout_score': row.get('heldout_score')})
            deltas.append(s.get('hybrid1_minus_base1_heldout'))
        for d in slots.values():
            d['heldout'] = _over_folds([p['heldout_score']
                                        for p in d['per_fold']])
        selection_summary[m] = slots
        heldout_delta[m] = {'per_fold': deltas, **_over_folds(deltas)}

    # --- Print ---------------------------------------------------------------
    print(f"\n  HEATMAP (model x budget -> best {metric_name} on each fold's "
          f"training rows, mean over {len(ok)} fold(s)):")
    print(f"  {'Model':<10}" + "".join(f"{b:>9}" for b in budgets))
    for m in models:
        print(f"  {m:<10}" + "".join(
            f"{heatmap[m][b]:>9.4f}" if b in heatmap[m] else f"{'--':>9}"
            for b in budgets))
    print(f"\n  PCA BASELINES (held-out {metric_name}, mean over folds):")
    for m in factories:
        print(f"  {m:<10}" + "".join(
            f"{pca_baselines[b][m]:>9.4f}" for b in budgets))
    print(f"\n  HYBRID 1 vs BASE 1, held out ({metric_name}, + means the "
          f"hybrid won; both picked on training rows):")
    for m in models:
        hd = heldout_delta[m]
        if hd['mean'] is None:
            print(f"    {m}: no fold with both scores")
        elif hd['n_folds'] == 1:
            print(f"    {m}: {hd['mean']:+.4f} (one fold, no spread)")
        else:
            print(f"    {m}: {hd['mean']:+.4f} over {hd['n_folds']} folds "
                  f"(min {hd['min']:+.4f}, max {hd['max']:+.4f})")

    return {
        'result_format': RESULT_FORMAT,
        'selection_scope': 'fold_train_rows',
        'metric_name': metric_name,
        'lower_is_better': bool(lower_is_better),
        'models': models,
        'models_dropped': models_dropped,
        'budgets': budgets,
        'n_outer_folds': n_outer,
        'n_outer_folds_asked': n_requested,
        'n_outer_folds_scored': len(ok),
        'per_fold': per_fold,
        # Over outer folds. configs/best_by_model/baselines/heatmap are
        # training-row scores of each fold's own search; pca_* and
        # standard_selection_summary/heldout_delta_by_model are held out.
        'configs': configs,
        'best_by_model': best_by_model,
        'baselines': baselines,
        'heatmap': heatmap,
        'heatmap_by_fold': heatmap_by_fold,
        'importance_wins': importance_wins,
        'importance_wins_counts': 'one per (model, outer fold)',
        'rep_dims': {n: int(np.asarray(v).shape[1]) for n, v in reps.items()},
        'pca_representations': pca_names,
        'pca_baselines': pca_baselines,
        'pca_baseline_widths': pca_widths,
        'pca_baseline_fold_scores': pca_fold_scores,
        'pca_validation_feature_cap':
            'clip to training-row min/max after FeatureScaler',
        'standard_selection_summary': selection_summary,
        'heldout_delta_by_model': heldout_delta,
        'has_augmentation': augs is not None,
        'augmentation_comparison_tiers': aug_tiers,
        'augmentation_comparison_skipped': aug_skip_reason,
        'arms_not_run': dict(ARMS_NOT_RUN),
    }


if __name__ == '__main__':
    parser = create_experiment_parser('C02: Complexity Matching Grid',
                                      has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    # A comma-separated --reps is a list of representation names, not a preset.
    rep_preset, rep_names = parse_rep_args(args)
    data = load_data(args.dataset, rep_preset, rep_names=rep_names,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    _mname, _, _lower = get_domain_metric(data)
    save_experiment_results(results, args.output, EXPERIMENT_DIR,
                            args.dataset,
                            metric_name=_mname, lower_is_better=_lower)
