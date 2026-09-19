#!/usr/bin/env python3
"""S04: Activity Cliffs -- RMSE_cliff improvement.

Paper: Similarity
Conclusion: Hybrid reduces RMSE on activity cliffs.
Dependencies: S02 (reads JSON, optional).

Identifies cliff compounds using ECFP4 Tanimoto (representation-independent
definition based on Bajorath standard). CV evaluation of kNN on cliff vs
non-cliff compounds for: per-rep kNN baselines, the pool's own single
representation, the pool's hybrid recipes (unweighted and importance-weighted),
RF baseline.

2026-09-12, owner's decision: this file runs NO hybrid recipe search. For each
held-out fold it reads the pool -- the best single representation plus the top
three hybrid configurations for its downstream model (kNN), ranked on the other
folds' scores. Its subset admits both families of stored search: the predictive
ones and the ones built under objective='similarity', where a configuration is
scored by predicting each held-out molecule's activity from its five nearest
neighbours. Three per family, so six hybrid arms per fold, not three.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, load_experiment_results,
    flatten_fold_weights, get_fold_pool_recipes,
    get_domain_metric, get_model_factories, build_fold_hybrid,
    fold_hybrid_scoring_kwargs,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
)
from experiments.similarity.similarity_helpers import identify_cliff_compounds
from kirby.find_best_hybrid import nested_cv_outer_folds

# 2026-09-12, owner's decision (EXPERIMENTS_FINALIZED.md, s04): s04 stops
# calling get_fold_best_search and reads the pool instead. So this seed no
# longer names a per-fold search cache -- it is the seed the folds are built
# with, and it has to be 42 because that is the seed the stored pool searches
# split on. Fold i here is fold i there only while both are 5 folds, seed 42,
# same groups and task type.
_FOLD_SEARCH_SEED = 42

# The subset of search families the pool may serve from: the hybrids built
# under the predictive objective, and the ones built under
# objective='similarity' (each held-out molecule's activity predicted from its
# five nearest neighbours). The owner put similarity hybrids in s04's subset.
_POOL_SUBSET = 'predictive_and_similarity'

# How many hybrid configurations the pool returns per family, per fold.
_N_POOL_HYBRIDS = 3

from kirby.hybrid import apply_feature_selection, compute_feature_importance
from experiments.similarity._summary_keys import (
    SELECTION_RULE as _SELECTION_RULE)

# Bit width and radius of the fingerprint the cliff rule is defined on. Radius
# 2 over 2048 bits is ECFP4, the same setting n01 builds its cliff similarity
# from.
_CLIFF_FP_BITS = 2048
_CLIFF_FP_RADIUS = 2


def _cliff_ecfp4(smiles_list):
    """ECFP4 bit matrix for the cliff rule, built from the dataset's SMILES.

    Returns (matrix, n_unparsed). A molecule rdkit cannot parse gets one
    private feature of its own rather than an all-zero row: scipy's jaccard
    distance calls two all-zero rows identical, which would make every
    unparsable molecule a cliff partner of every other one.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mols = [Chem.MolFromSmiles(smi) if smi else None for smi in smiles_list]
    unparsed = [i for i, m in enumerate(mols) if m is None]
    fp_array = np.zeros((len(mols), _CLIFF_FP_BITS + len(unparsed)),
                        dtype=np.float32)
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, _CLIFF_FP_RADIUS, nBits=_CLIFF_FP_BITS)
        for bit in fp.GetOnBits():
            fp_array[i, bit] = 1.0
    for slot, i in enumerate(unparsed):
        fp_array[i, _CLIFF_FP_BITS + slot] = 1.0
    return fp_array, len(unparsed)


def _cliff_fingerprint(data, reps):
    """The matrix the cliff rule is defined on, and where it came from.

    2026-09-12, owner's decision: build ECFP4 from the SMILES so the cliff rule
    stops depending on the representation pool. reps['ecfp4'] is the fallback
    for a dataset carrying no SMILES, or where rdkit cannot build them.

    Returns (matrix or None, source name or None, note).
    """
    smiles = data.get('smiles')
    n_rows = len(data['labels'])
    if smiles is not None and len(smiles) == n_rows:
        try:
            fp_array, n_unparsed = _cliff_ecfp4(smiles)
        except Exception as exc:
            note = (f'ECFP4 from SMILES failed ({type(exc).__name__}: {exc})')
        else:
            return fp_array, 'smiles_ecfp4', (
                f'ECFP4 radius {_CLIFF_FP_RADIUS} over {_CLIFF_FP_BITS} bits, '
                f'built from this dataset\'s {n_rows} SMILES; {n_unparsed} of '
                f'them rdkit could not parse and each got a private feature, '
                f'so it is a cliff partner of nothing')
    elif smiles is None:
        note = 'the dataset carries no SMILES'
    else:
        note = (f'the dataset carries {len(smiles)} SMILES for {n_rows} '
                f'molecules')

    if 'ecfp4' in reps:
        return reps['ecfp4'], 'pool_ecfp4', (
            f'fell back to the pool\'s ecfp4 representation: {note}')
    return None, None, (
        f'{note}, and ecfp4 is not in the representation pool either')


# =============================================================================
# EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Prediction performance on activity cliff compounds.

    Uses representation-independent cliff definition (Bajorath ECFP4 standard):
      - ECFP4 Tanimoto >= tanimoto_threshold
      - |delta_y| > activity_k * MAD(y)

    Reports RMSE_cliff, RMSE_noncliff for:
      - Per-rep kNN baselines
      - Hybrid kNN (unweighted and weighted)
      - RF baseline
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT S04: PREDICTION PERFORMANCE ON ACTIVITY CLIFFS")
    print("=" * 70)

    tanimoto_threshold = kwargs.get('tanimoto_threshold', 0.55)
    activity_k = kwargs.get('activity_k', 2.0)
    n_folds = config['n_splits']
    mode = 'intermediary' if config['n_splits'] == 1 else 'full'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    # --- Identify cliff compounds using ECFP4 Tanimoto ---
    # S04 is molecular-only (Bajorath ECFP4 standard). Non-molecular domains
    # should use S12 (cross-domain Euclidean cliffs) instead.
    domain = data.get('domain', 'molecular')
    if domain != 'molecular':
        print(f"  SKIP: S04 requires molecular ECFP4 (domain={domain}). Use S12 for cross-domain cliffs.")
        # F21: the S12 pointer is a CLAIM about another experiment. Nothing here
        # loads, runs or checks an S12 result, so the stub records the pointer as
        # unverified — "handled elsewhere" must stay distinguishable from
        # "answered nowhere" for a reader who only has this file's output.
        return {
            'skipped': True,
            'reason': f'molecular-only experiment, domain={domain}',
            'replacement_experiment': 's12',
            'replacement_verified': False,
            'replacement_note': (
                'S12 (cross-domain Euclidean cliffs) is named as the replacement; '
                'this stub does not read or check for any S12 cliff result for '
                'this dataset. Verify one exists before treating cliffs as covered.'),
        }
    cliff_fp, cliff_fp_source, cliff_fp_note = _cliff_fingerprint(data, reps)
    if cliff_fp is None:
        print(f"  ERROR: cannot identify cliffs -- {cliff_fp_note}")
        return {'error': 'ecfp4 not available',
                'cliff_fingerprint_source': None,
                'cliff_fingerprint_note': cliff_fp_note}
    print(f"  Cliff fingerprint: {cliff_fp_source} -- {cliff_fp_note}")

    # Derive task_type BEFORE cliff identification so classification cliffs
    # use the class-mismatch indicator, not the regression |delta_y| rule.
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data['task_type']
    is_clf = task_type == 'classification'

    cliff_mask, cliff_info = identify_cliff_compounds(
        cliff_fp, labels,
        tanimoto_threshold=tanimoto_threshold,
        activity_k=activity_k,
        is_clf=is_clf,
    )
    cliff_indices = set(np.where(cliff_mask)[0].tolist())

    if is_clf:
        print(f"  Cliff definition (clf): Tanimoto>={tanimoto_threshold}, "
              f"different class")
    else:
        print(f"  Cliff definition: Tanimoto>={tanimoto_threshold}, "
              f"|delta_y|>{activity_k}*MAD(y) = {cliff_info['activity_threshold']:.3f}")
    print(f"  Cliff compounds: {cliff_info['n_cliff_compounds']} / {len(labels)} "
          f"({cliff_info['pct_cliff_compounds']:.1f}%)")
    print(f"  Cliff pairs: {cliff_info['n_cliff_pairs']}")

    results = {
        # Which selection rule produced the numbers below. s09 reads this
        # and refuses to run its signed-rank test across a mixed set — see
        # _summary_keys.SELECTION_RULE. Added 2026-09-18.
        'selection_rule': _SELECTION_RULE,
        'cliff_definition': {
            'fingerprint_source': cliff_fp_source,
            'fingerprint_note': cliff_fp_note,
            'tanimoto_threshold': tanimoto_threshold,
            'activity_k': activity_k,
            'activity_threshold': cliff_info['activity_threshold'],
            'mad_y': cliff_info['mad_y'],
            'n_cliff_compounds': cliff_info['n_cliff_compounds'],
            'n_cliff_pairs': cliff_info['n_cliff_pairs'],
            'pct_cliff_compounds': cliff_info['pct_cliff_compounds'],
        },
        'per_rep': {},
        'hybrid': {},
    }

    # --- Per-fold reselection accumulators (this file's own cliff-CV folds) ---
    # The honest hybrid-vs-best-single comparison is done by the reader via the
    # shared fold_reselection() helper. We only persist the per-fold inputs here,
    # keyed by downstream MODEL, using the CONTRACT json_convention names. This
    # file runs its OWN CV (kNN + RF on the shared splitter), so we build the
    # block MANUALLY rather than build_fold_reselection_input(search) — the search's
    # internal baselines are on different folds/metric. Fold scores are POSITIONAL
    # (one slot per split, NaN where a fold was skipped) so both arms stay aligned
    # across the identical split order; the helper skips NaN slots.
    reselection_candidates = {}  # model -> [ {name, fold_scores}, ... ]
    reselected_baselines = {}   # model -> { rep_name: [fold_scores], ... }

    # --- CV evaluation of prediction on cliff vs non-cliff ---
    # The folds come from nested_cv_outer_folds, the helper each fold's recipe
    # search caches against, so fold i's recipe was searched on exactly fold i's
    # training rows. Same group-aware split as before: GroupKFold, or
    # StratifiedGroupKFold on a classification dataset.
    sample_X = next(iter(reps.values()))
    fold_splits = nested_cv_outer_folds(reps, labels, groups, n_folds,
                                        _FOLD_SEARCH_SEED, task_type)

    # --- Per-fold pool read (2026-09-12, owner's decision) -------------------
    # DECIDED: s04 stops calling get_fold_best_search. It reads the pool
    # instead -- the best single representation plus the top three hybrids for
    # its downstream model, chosen per fold from the other folds' scores.
    # Similarity hybrids are in its subset.
    #
    # What changed. Nothing here runs a hybrid recipe search any more. For each
    # held-out fold, get_fold_pool_recipes reads the stored searches' per-fold
    # scores and ranks both the single representations and the hybrid
    # configurations by their mean over the OTHER folds. Fold i's own score
    # never enters its own ranking. The searches themselves ran over every row,
    # so this holds out the fold's SCORES, not its rows -- and where a pool
    # search pruned mid-flight, fold 0's scores chose which configurations
    # survived at all ('fold0_pruned').
    budget = kwargs.get('budget', 100)
    augmentations = data.get('augmentations')
    _pool_cache = {}

    def _fold_pool(fold_i, model):
        """The pool read for held-out fold `fold_i`, ranked for `model`."""
        key = (int(fold_i), model)
        if key not in _pool_cache:
            _pool_cache[key] = get_fold_pool_recipes(
                reps, labels, groups,
                fold_i=fold_i, n_outer_folds=len(fold_splits),
                downstream_model=model, budget=budget,
                dataset_name=dataset_name, augmentations=augmentations,
                distance=True, data=data,
                metric=metric_name, task_type=task_type, domain=domain,
                pool_subset=_POOL_SUBSET, n_hybrids=_N_POOL_HYBRIDS)
        return _pool_cache[key]

    knn_pools = {}
    hybrid_arms = []          # [(objective, rank), ...] in subset order
    fold0_pruned = None
    try:
        for _fi in range(len(fold_splits)):
            knn_pools[_fi] = _fold_pool(_fi, 'knn')
        results['hybrid_search_status'] = {
            'ok': bool(any(p['hybrids'] for p in knn_pools.values())),
            'error': (None if any(p['hybrids'] for p in knn_pools.values())
                      else 'the pool held no kNN hybrid configuration for any fold'),
        }
    except Exception as e:
        print(f"  Pool read failed: {type(e).__name__}: {e}")
        # F17: persist the throw. Without this the 'hybrid' block stays the empty
        # dict it was initialised to while 'per_rep' fills normally, so the file
        # reads as a complete run with no hybrid arm.
        results['hybrid_search_status'] = {
            'ok': False,
            'error_type': type(e).__name__,
            'error': str(e),
        }

    if knn_pools:
        _p0 = knn_pools[0]
        # One arm per (family, rank) slot. The RECIPE inside a slot is re-picked
        # for every fold from that fold's other-folds ranking, so the arm is the
        # slot, not one fixed recipe. Under 'predictive_and_similarity' that is
        # three predictive slots plus three similarity slots, not three in total.
        _seen = []
        for _fi in sorted(knn_pools):
            for h in knn_pools[_fi]['hybrids']:
                if (h['objective'], h['rank']) not in _seen:
                    _seen.append((h['objective'], h['rank']))
        _obj_order = {o: i for i, o in enumerate(_p0['objectives'])}
        hybrid_arms = sorted(_seen, key=lambda k: (_obj_order.get(k[0], 99), k[1]))

        for _fi in sorted(knn_pools):
            for w in knn_pools[_fi]['warnings']:
                print(f"  [pool fold {_fi}] {w}")
        fold0_pruned = _p0.get('fold0_pruned')

        # What the pool chose, per fold, so the file records which single
        # representation and which recipes each fold's ranking picked. It is a
        # per-fold choice, not a whole-dataset ranking.
        results['pool_selection'] = {
            str(_fi): {
                'base_rep': (knn_pools[_fi]['base_rep'] or {}).get('rep'),
                'base_rep_score_other_folds':
                    (knn_pools[_fi]['base_rep'] or {}).get('score_other_folds'),
                'base_reps_ranked': [
                    (b['rep'], b['score_other_folds'])
                    for o in knn_pools[_fi]['by_objective']
                    for b in knn_pools[_fi]['by_objective'][o]['base_reps_ranked'][:3]
                ],
                'hybrids': [
                    {'name': h['name'], 'objective': h['objective'],
                     'rank': h['rank'], 'recipe': h['recipe'],
                     'metric_name': h['metric_name'],
                     'score_other_folds': h['score_other_folds']}
                    for h in knn_pools[_fi]['hybrids']],
                'selection_scope': knn_pools[_fi]['selection_scope'],
                'fold0_pruned': knn_pools[_fi]['fold0_pruned'],
                'n_folds_in_pool': knn_pools[_fi]['n_folds_in_pool'],
                'warnings': knn_pools[_fi]['warnings'],
            }
            for _fi in sorted(knn_pools)
        }
        results['pool_subset'] = _p0['pool_subset']
        results['pool_objectives'] = list(_p0['objectives'])
        results['pool_selection_note'] = _p0['selection_note']

        _b0 = _p0['base_rep']
        if _b0 is not None:
            print(f"  Pool base representation, fold 0 (by knn, other folds' "
                  f"{_b0['metric_name']}): {_b0['rep']}="
                  f"{_b0['score_other_folds']:.4f}")
        for _obj in _p0['by_objective']:
            _ranked = _p0['by_objective'][_obj]['base_reps_ranked'][:3]
            if _ranked:
                print(f"  Top-3 base reps, fold 0 ({_obj}, by knn): "
                      + ', '.join(f"{b['rep']}={b['score_other_folds']:.4f}"
                                  for b in _ranked))
        # Kept under its old key so the readers of s04's JSON still find it. It
        # is now the pool's kNN ranking on fold 0's other folds, not an RF
        # ranking out of a fold-0 search.
        results['top3_base_reps_first_fold'] = [
            (b['rep'], b['score_other_folds'])
            for o in _p0['by_objective']
            for b in _p0['by_objective'][o]['base_reps_ranked'][:3]]
        if hybrid_arms:
            print(f"  Pool hybrid arms: {len(hybrid_arms)} "
                  f"({', '.join(f'{o} rank {r}' for o, r in hybrid_arms)})")

    hybrid_blueprint_available = bool(hybrid_arms)

    def _make_knn(n_train):
        k = min(5, n_train - 1)
        if is_clf:
            return KNeighborsClassifier(n_neighbors=k)
        return KNeighborsRegressor(n_neighbors=k)

    def _evaluate_condition(get_Xy_fn, pre_scaled=False):
        """Run CV for a condition, return cliff/noncliff metrics."""
        fold_all, fold_cliff, fold_noncliff = [], [], []
        fold_rmse_all, fold_rmse_cliff, fold_rmse_noncliff = [], [], []
        # Positional per-fold metrics for fold_reselection, one slot per split so
        # every arm stays index-aligned. NaN where the fold was dropped, and on
        # the cliff/non-cliff lists also where the held-out fold carried 1 or
        # fewer rows of that stratum.
        fold_scores_pos = []
        fold_cliff_pos, fold_noncliff_pos = [], []
        # How many held-out rows of each stratum the fold carried. score_cliff
        # and score_noncliff can rest on different numbers of folds, and
        # neither count was written out before.
        n_cliff_rows_pos, n_noncliff_rows_pos = [], []

        for fold_i, (train_idx, val_idx) in enumerate(fold_splits):
            y_train, y_val = labels[train_idx], labels[val_idx]

            # Stratify by cliff membership. Depends only on the split, so it is
            # counted even for a fold whose features failed to build.
            val_cliff = np.array([i for i, vi in enumerate(val_idx)
                                  if vi in cliff_indices], dtype=int)
            val_noncliff = np.array([i for i, vi in enumerate(val_idx)
                                     if vi not in cliff_indices], dtype=int)
            n_cliff_rows_pos.append(int(len(val_cliff)))
            n_noncliff_rows_pos.append(int(len(val_noncliff)))

            def _drop_fold():
                fold_scores_pos.append(float('nan'))
                fold_cliff_pos.append(float('nan'))
                fold_noncliff_pos.append(float('nan'))

            try:
                X_tr, X_va = get_Xy_fn(train_idx, val_idx, y_train, fold_i)
            except Exception as exc:
                # N9: log the dropped fold so a thrown build is distinguishable
                # from a genuinely empty fold (X_tr is None below).
                print(f"      cliff CV fold build failed: "
                      f"{type(exc).__name__}: {exc}")
                _drop_fold()
                continue
            if X_tr is None:
                _drop_fold()
                continue

            if pre_scaled:
                X_tr_s, X_va_s = X_tr, X_va
            else:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_va_s = scaler.transform(X_va)

            model = _make_knn(len(y_train))
            model.fit(X_tr_s, y_train)
            y_pred = model.predict(X_va_s)

            # Overall
            _score_all = float(metric_fn(y_val, y_pred))
            fold_all.append(_score_all)
            fold_scores_pos.append(_score_all)
            if not is_clf:
                fold_rmse_all.append(float(np.sqrt(mean_squared_error(y_val, y_pred))))

            if len(val_cliff) > 1:
                _score_cliff = float(metric_fn(y_val[val_cliff], y_pred[val_cliff]))
                fold_cliff.append(_score_cliff)
                fold_cliff_pos.append(_score_cliff)
                if not is_clf:
                    fold_rmse_cliff.append(float(np.sqrt(mean_squared_error(
                        y_val[val_cliff], y_pred[val_cliff]))))
            else:
                fold_cliff_pos.append(float('nan'))
            if len(val_noncliff) > 1:
                _score_noncliff = float(metric_fn(y_val[val_noncliff],
                                                  y_pred[val_noncliff]))
                fold_noncliff.append(_score_noncliff)
                fold_noncliff_pos.append(_score_noncliff)
                if not is_clf:
                    fold_rmse_noncliff.append(float(np.sqrt(mean_squared_error(
                        y_val[val_noncliff], y_pred[val_noncliff]))))
            else:
                fold_noncliff_pos.append(float('nan'))

        if not fold_all:
            return None
        out = {
            'score_all': float(np.mean(fold_all)),
            'score_cliff': float(np.mean(fold_cliff)) if fold_cliff else float('nan'),
            'score_noncliff': float(np.mean(fold_noncliff)) if fold_noncliff else float('nan'),
            'metric': metric_name,
            # Positional overall per-fold scores for the fold_reselection block.
            'fold_scores': [float(s) for s in fold_scores_pos],
            # The same folds, scored on the cliff rows alone and on the
            # non-cliff rows alone. The cliff list is what the fold_reselection
            # block below is fed, so its held-out re-pick answers the cliff
            # question rather than the overall one.
            'fold_scores_cliff': [float(s) for s in fold_cliff_pos],
            'fold_scores_noncliff': [float(s) for s in fold_noncliff_pos],
            'n_cliff_rows_by_fold': n_cliff_rows_pos,
            'n_noncliff_rows_by_fold': n_noncliff_rows_pos,
            'n_folds_scored_cliff': int(len(fold_cliff)),
            'n_folds_scored_noncliff': int(len(fold_noncliff)),
        }
        if not is_clf:
            out['rmse_all'] = float(np.mean(fold_rmse_all))
            out['rmse_cliff'] = float(np.mean(fold_rmse_cliff)) if fold_rmse_cliff else float('nan')
            out['rmse_noncliff'] = float(np.mean(fold_rmse_noncliff)) if fold_rmse_noncliff else float('nan')
        return out

    # --- Precompute per-fold feature importance per rep (hoisted from lambda) ---
    # Cache fold train indices so we compute importance once per (fold, rep).
    def _fold_key(tr):
        return (int(tr[0]), int(tr[-1]), int(len(tr)))
    fold_importance = {}  # {rep_name: {fold_key: imp_array}}
    for rep_name, X_rep in reps.items():
        fold_importance[rep_name] = {}
        for tr, _va in fold_splits:
            try:
                imp = compute_feature_importance(
                    {rep_name: X_rep[tr]}, labels[tr], method='lightgbm_gain')
                fold_importance[rep_name][_fold_key(tr)] = imp[rep_name]
            except Exception as exc:
                # N9: log so a None-importance fold reads as "compute failed"
                # rather than silently weakening the weighted kNN.
                print(f"      importance fold failed (rep={rep_name}): "
                      f"{type(exc).__name__}: {exc}")
                fold_importance[rep_name][_fold_key(tr)] = None

    # --- Per-rep kNN baselines (unweighted + weighted) ---
    print(f"\n  Per-rep kNN baselines (k=5, metric={metric_name}):")
    for rep_name, X_rep in reps.items():
        # Unweighted
        def _get_rep(tr, va, yt, fold_i, _X=X_rep):
            return _X[tr], _X[va]
        res = _evaluate_condition(_get_rep)
        if res:
            results['per_rep'][rep_name] = res
            # Plain single-rep kNN is a baseline arm for fold_reselection (model=knn).
            # The cliff-row metric, not the overall one: this block's held-out
            # re-pick is the only place s04's cliff question is answered.
            reselected_baselines.setdefault('knn', {})[rep_name] = res['fold_scores_cliff']

            # Weighted: importance-weighted distance on this rep (precomputed per fold)
            def _get_rep_w(tr, va, yt, fold_i, _X=X_rep, _rname=rep_name):
                imp_arr = fold_importance[_rname].get(_fold_key(tr))
                if imp_arr is None:
                    return None, None
                w = np.sqrt(np.maximum(imp_arr, 0))
                scaler = StandardScaler()
                return scaler.fit_transform(_X[tr]) * w, scaler.transform(_X[va]) * w
            res_w = _evaluate_condition(_get_rep_w, pre_scaled=True)
            if res_w:
                results['per_rep'][rep_name]['score_cliff_weighted'] = res_w['score_cliff']
                results['per_rep'][rep_name]['score_all_weighted'] = res_w['score_all']
                results['per_rep'][rep_name]['fold_scores_cliff_weighted'] = \
                    res_w['fold_scores_cliff']
                results['per_rep'][rep_name]['fold_scores_noncliff_weighted'] = \
                    res_w['fold_scores_noncliff']
                if not is_clf:
                    results['per_rep'][rep_name]['rmse_cliff_weighted'] = res_w.get('rmse_cliff', float('nan'))
                # Importance-weighted single-rep kNN is still a single-rep baseline
                # (no cross-rep mixing) — record as a distinct baseline entry.
                reselected_baselines.setdefault('knn', {})[f'{rep_name}|weighted'] = res_w['fold_scores_cliff']

            rmse_part = f"  RMSE_cliff={res.get('rmse_cliff', float('nan')):.4f}" if not is_clf else ""
            w_part = ""
            if res_w and not is_clf:
                w_part = f"  | weighted: RMSE_cliff={res_w.get('rmse_cliff', float('nan')):.4f}"
            print(f"    {rep_name:<15} {metric_name}_all={res['score_all']:.4f}  "
                  f"{metric_name}_cliff={res['score_cliff']:.4f}{rmse_part}  "
                  f"{metric_name}_noncliff={res['score_noncliff']:.4f}"
                  + w_part)

    # --- The pool's best single representation, re-picked per fold -----------
    # 2026-09-12 decision: the pool supplies the best single representation as
    # well as the hybrids, chosen per fold from the other folds' scores. Which
    # representation that is can differ from fold to fold, so this is one arm
    # whose identity moves, not one of the fixed per-rep arms above.
    if knn_pools:
        _pool_base_by_fold = {
            _fi: (knn_pools[_fi]['base_rep'] or {}).get('rep')
            for _fi in knn_pools}

        def _get_pool_base(tr, va, yt, fold_i):
            _rname = _pool_base_by_fold.get(fold_i)
            if _rname is None or _rname not in reps:
                return None, None
            return reps[_rname][tr], reps[_rname][va]

        res_pb = _evaluate_condition(_get_pool_base)
        res_pb_w = None
        if res_pb:
            def _get_pool_base_w(tr, va, yt, fold_i):
                _rname = _pool_base_by_fold.get(fold_i)
                if _rname is None or _rname not in reps:
                    return None, None
                imp_arr = fold_importance[_rname].get(_fold_key(tr))
                if imp_arr is None:
                    return None, None
                w = np.sqrt(np.maximum(imp_arr, 0))
                _X = reps[_rname]
                scaler = StandardScaler()
                return (scaler.fit_transform(_X[tr]) * w,
                        scaler.transform(_X[va]) * w)

            res_pb_w = _evaluate_condition(_get_pool_base_w, pre_scaled=True)
            results['pool_base_rep'] = {
                'reps_by_fold': {str(_fi): _pool_base_by_fold.get(_fi)
                                 for _fi in sorted(_pool_base_by_fold)},
                'unweighted': res_pb,
                'weighted': res_pb_w,
                'selection_scope': 'other_folds_of_pool_search',
            }
            # A single representation, no cross-rep mixing, so it is a baseline
            # arm — one whose name says its identity was re-picked per fold.
            reselected_baselines.setdefault('knn', {})['pool_base_rep|per_fold'] = \
                res_pb['fold_scores_cliff']
            if res_pb_w:
                reselected_baselines['knn']['pool_base_rep|per_fold|weighted'] = \
                    res_pb_w['fold_scores_cliff']
            _names = sorted({v for v in _pool_base_by_fold.values() if v})
            rmse_part = (f"  RMSE_cliff={res_pb.get('rmse_cliff', float('nan')):.4f}"
                         if not is_clf else "")
            print(f"    pool_base({'/'.join(_names) if _names else 'none'}) "
                  f"{metric_name}_all={res_pb['score_all']:.4f}  "
                  f"{metric_name}_cliff={res_pb['score_cliff']:.4f}{rmse_part}")
        else:
            results['pool_base_rep'] = {
                'reps_by_fold': {str(_fi): _pool_base_by_fold.get(_fi)
                                 for _fi in sorted(_pool_base_by_fold)},
                'computed': False,
                'reason': ('the pool named no single representation for any '
                           'fold, or every cliff-CV fold was dropped'),
            }

    # --- Hybrid kNN, one arm per (family, rank) slot the pool returned ---
    if hybrid_blueprint_available:
        print(f"\n  Hybrid kNN (k=5), {len(hybrid_arms)} pool arms "
              f"(subset {_POOL_SUBSET}, top {_N_POOL_HYBRIDS} per family):")

        def _arm_entry(fold_i, objective, rank):
            """The pool's hybrid for this fold in the (family, rank) slot."""
            pool = knn_pools.get(fold_i)
            if pool is None:
                return None
            for h in pool['hybrids']:
                if h['objective'] == objective and h['rank'] == rank:
                    return h
            return None

        def _build_arm(tr, va, yt, fold_i, objective, rank):
            """(X_train, X_val, feature_info, blueprint) for one arm, one fold."""
            h = _arm_entry(fold_i, objective, rank)
            if h is None or h.get('blueprint') is None:
                return None, None, None, None
            bp_fold = h['blueprint']   # augmentation_strategy already set by the pool
            aug_cond = h['aug_condition']
            train_reps_h = {name: rep[tr] for name, rep in reps.items()}
            val_reps_h = {name: rep[va] for name, rep in reps.items()}
            _use_aug = augmentations if aug_cond != 'none' else None
            aug_tr = {k: v[tr] for k, v in _use_aug.items()} if _use_aug else None
            aug_va = {k: v[va] for k, v in _use_aug.items()} if _use_aug else None
            X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                train_reps_h, val_reps_h, yt, bp_fold,
                augmentations_train=aug_tr, augmentations_val=aug_va,
                groups=(groups[tr] if groups is not None else None),
                **fold_hybrid_scoring_kwargs(
                    bp_fold, model=_make_knn(len(yt)),
                    metric_fn=metric_fn, lower_is_better=lower_is_better))
            return X_tr_h, X_va_h, feat_info, bp_fold

        results['pool_hybrid_arms'] = {}
        for _obj, _rank in hybrid_arms:
            _arm = f'{_obj}|rank{_rank}'

            def _get_arm_uw(tr, va, yt, fold_i, _o=_obj, _r=_rank):
                X_tr_h, X_va_h, _fi, _bp = _build_arm(tr, va, yt, fold_i, _o, _r)
                if X_tr_h is None:
                    return None, None
                return X_tr_h, X_va_h

            def _get_arm_w(tr, va, yt, fold_i, _o=_obj, _r=_rank):
                X_tr_h, X_va_h, feat_info, bp_fold = _build_arm(
                    tr, va, yt, fold_i, _o, _r)
                if X_tr_h is None:
                    return None, None
                train_reps_h = {name: rep[tr] for name, rep in reps.items()}
                # Same importance method the search used to build this hybrid.
                imp = compute_feature_importance(
                    train_reps_h, yt,
                    method=getattr(bp_fold, 'importance_method', 'lightgbm_gain'))
                w_parts = []
                for rname in feat_info:
                    if not isinstance(feat_info[rname], dict):
                        continue
                    if 'selected_indices' not in feat_info[rname]:
                        continue
                    idx = feat_info[rname]['selected_indices']
                    if rname in imp:
                        w_parts.append(imp[rname][idx])
                    else:
                        w_parts.append(np.ones(len(idx)))
                w = np.sqrt(np.maximum(np.concatenate(w_parts), 0))
                # Scale-then-weight (matches Exp 5 and compute_distance_matrix
                # convention)
                scaler_w = StandardScaler()
                return (scaler_w.fit_transform(X_tr_h) * w,
                        scaler_w.transform(X_va_h) * w)

            res = _evaluate_condition(_get_arm_uw)
            res_w = _evaluate_condition(_get_arm_w, pre_scaled=True)

            _entry = {
                'objective': _obj,
                'rank': _rank,
                # The recipe in this slot is re-picked per fold, so store the
                # one each fold used rather than one name for the arm.
                'recipe_by_fold': {
                    str(_fi): ((_arm_entry(_fi, _obj, _rank) or {}).get('name'))
                    for _fi in sorted(knn_pools)},
                'score_other_folds_by_fold': {
                    str(_fi): ((_arm_entry(_fi, _obj, _rank) or {})
                               .get('score_other_folds'))
                    for _fi in sorted(knn_pools)},
                'pool_metric_name': next(
                    ((_arm_entry(_fi, _obj, _rank) or {}).get('metric_name')
                     for _fi in sorted(knn_pools)
                     if _arm_entry(_fi, _obj, _rank) is not None), None),
                'unweighted': res,
                'weighted': res_w,
            }
            if res is None:
                _entry['unweighted_status'] = {
                    'computed': False,
                    'reason': ('no cliff-CV fold produced a score (all folds '
                               'skipped or threw)'),
                }
            if res_w is None:
                _entry['weighted_status'] = {
                    'computed': False,
                    'reason': ('no cliff-CV fold produced a score (all folds '
                               'skipped or threw)'),
                }
            results['pool_hybrid_arms'][_arm] = _entry

            # Each arm is its own candidate for fold_reselection (model=knn).
            if res:
                reselection_candidates.setdefault('knn', []).append(
                    {'name': f'hybrid_{_obj}_rank{_rank}_unweighted',
                     'fold_scores': res['fold_scores_cliff']})
            if res_w:
                reselection_candidates.setdefault('knn', []).append(
                    {'name': f'hybrid_{_obj}_rank{_rank}_weighted',
                     'fold_scores': res_w['fold_scores_cliff']})

            for _label, _r in (('uw', res), ('w', res_w)):
                if not _r:
                    print(f"    {_arm}_{_label:<2} not computed: all cliff-CV "
                          f"folds dropped")
                    continue
                rmse_part = (f"  RMSE_cliff={_r.get('rmse_cliff', float('nan')):.4f}"
                             if not is_clf else "")
                print(f"    {_arm}_{_label:<2} {metric_name}_all={_r['score_all']:.4f}  "
                      f"{metric_name}_cliff={_r['score_cliff']:.4f}{rmse_part}  "
                      f"{metric_name}_noncliff={_r['score_noncliff']:.4f}")

        # results['hybrid']['unweighted'] / ['weighted'] are what s09 and the
        # first-pass reader read. They now hold the FIRST arm in subset order —
        # the predictive family's rank-1 hybrid where that family is present —
        # so those readers keep getting one hybrid number. Every arm is in
        # results['pool_hybrid_arms'].
        _lead = f'{hybrid_arms[0][0]}|rank{hybrid_arms[0][1]}' if hybrid_arms else None
        _lead_entry = results['pool_hybrid_arms'].get(_lead, {})
        results['hybrid']['lead_arm'] = _lead
        for _cond in ('unweighted', 'weighted'):
            if _lead_entry.get(_cond):
                results['hybrid'][_cond] = _lead_entry[_cond]
            else:
                # F17: every fold of this condition was dropped, so the key is
                # absent. Name that instead of leaving it simply missing.
                results['hybrid'][f'{_cond}_status'] = _lead_entry.get(
                    f'{_cond}_status',
                    {'computed': False,
                     'reason': 'no cliff-CV fold produced a score (all folds '
                               'skipped or threw)'})
    else:
        # F17: no recipe means NEITHER hybrid cliff score is attempted. Record
        # both absences and which of the two causes applies — the pool read
        # threw, or it returned no kNN hybrid configuration for any fold.
        _st = results.get('hybrid_search_status', {})
        if _st.get('error_type'):
            _reason = (f"pool read raised "
                       f"{_st.get('error_type')}: {_st.get('error', '')}")
        else:
            _reason = (_st.get('error')
                       or 'the pool held no kNN hybrid configuration')
        for _cond in ('unweighted', 'weighted'):
            results['hybrid'][f'{_cond}_status'] = {
                'computed': False,
                'reason': _reason,
            }
        print(f"\n  Hybrid kNN: not computed ({_reason})")

    # --- RF baseline (best single representation for RF, per fold) ----------
    # 2026-09-12: the representation came from fold 0's own search, which no
    # longer runs. It now comes from the same pool, read for downstream_model
    # 'rf' and ranked on each fold's OTHER folds, so it can differ per fold.
    # Where the pool names none, the fallback is unchanged: the best-scoring
    # representation under the kNN baselines above.
    print("\n  RF baseline (per-rep, best rep for RF):")
    fallback_rep_name = None
    if results['per_rep']:
        fallback_rep_name = (min if lower_is_better else max)(
            results['per_rep'], key=lambda r: results['per_rep'][r]['score_all']
        )
    rf_rep_by_fold = {}
    for _fi in range(len(fold_splits)):
        _rep = None
        try:
            _rep = (_fold_pool(_fi, 'rf')['base_rep'] or {}).get('rep')
        except Exception as exc:
            print(f"      rf pool read failed on fold {_fi}: "
                  f"{type(exc).__name__}: {exc}")
        if _rep is None or _rep not in reps:
            _rep = fallback_rep_name
        rf_rep_by_fold[_fi] = _rep
    _rf_reps = sorted({r for r in rf_rep_by_fold.values() if r})
    best_rep_name = (rf_rep_by_fold.get(0) or fallback_rep_name)
    if best_rep_name:
        rf_factory = get_model_factories(['rf'], task_type=task_type,
                                          n_samples=len(labels))['rf']

        fold_all, fold_cliff, fold_noncliff = [], [], []
        fold_rmse_cliff = []
        # Same positional convention as the kNN arms: one slot per split, NaN
        # where the held-out fold carried 1 or fewer rows of the stratum.
        fold_cliff_pos, fold_noncliff_pos = [], []
        n_cliff_rows_pos, n_noncliff_rows_pos = [], []
        for fold_i, (train_idx, val_idx) in enumerate(fold_splits):
            y_train, y_val = labels[train_idx], labels[val_idx]
            _rep = rf_rep_by_fold.get(fold_i) or best_rep_name
            X_best = reps[_rep]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_best[train_idx])
            X_va = scaler.transform(X_best[val_idx])
            rf = rf_factory()
            rf.fit(X_tr, y_train)
            y_pred = rf.predict(X_va)

            fold_all.append(metric_fn(y_val, y_pred))
            val_cliff = np.array([i for i, vi in enumerate(val_idx)
                                  if vi in cliff_indices], dtype=int)
            val_noncliff = np.array([i for i, vi in enumerate(val_idx)
                                     if vi not in cliff_indices], dtype=int)
            n_cliff_rows_pos.append(int(len(val_cliff)))
            n_noncliff_rows_pos.append(int(len(val_noncliff)))
            if len(val_cliff) > 1:
                _score_cliff = float(metric_fn(y_val[val_cliff], y_pred[val_cliff]))
                fold_cliff.append(_score_cliff)
                fold_cliff_pos.append(_score_cliff)
                if not is_clf:
                    fold_rmse_cliff.append(float(np.sqrt(mean_squared_error(
                        y_val[val_cliff], y_pred[val_cliff]))))
            else:
                fold_cliff_pos.append(float('nan'))
            if len(val_noncliff) > 1:
                _score_noncliff = float(metric_fn(y_val[val_noncliff],
                                                  y_pred[val_noncliff]))
                fold_noncliff.append(_score_noncliff)
                fold_noncliff_pos.append(_score_noncliff)
            else:
                fold_noncliff_pos.append(float('nan'))

        results['rf_baseline'] = {
            'rep': best_rep_name,
            # Which representation each fold actually fitted. 'rep' above is
            # fold 0's, kept as a plain name for the readers that print it.
            'reps_by_fold': {str(_fi): rf_rep_by_fold.get(_fi)
                             for _fi in sorted(rf_rep_by_fold)},
            'rep_selection': 'other_folds_of_pool_search',
            'score_all': float(np.mean(fold_all)),
            'score_cliff': float(np.mean(fold_cliff)) if fold_cliff else float('nan'),
            'score_noncliff': float(np.mean(fold_noncliff)) if fold_noncliff else float('nan'),
            'metric': metric_name,
            'fold_scores': [float(s) for s in fold_all],
            'fold_scores_cliff': [float(s) for s in fold_cliff_pos],
            'fold_scores_noncliff': [float(s) for s in fold_noncliff_pos],
            'n_cliff_rows_by_fold': n_cliff_rows_pos,
            'n_noncliff_rows_by_fold': n_noncliff_rows_pos,
            'n_folds_scored_cliff': int(len(fold_cliff)),
            'n_folds_scored_noncliff': int(len(fold_noncliff)),
        }
        # RF is a separate downstream model — record its per-fold single-rep
        # baseline under model=rf. No RF hybrid candidate exists here, so the
        # honest delta for 'rf' is NaN (hybrid arm empty); the helper tolerates it.
        # The name says the representation was re-picked per fold when the folds
        # did not all land on one. The cliff-row metric, same as the kNN arms.
        _rf_key = (best_rep_name if len(_rf_reps) <= 1
                   else 'pool_base_rep|per_fold')
        reselected_baselines.setdefault('rf', {})[_rf_key] = [
            float(s) for s in fold_cliff_pos]
        if not is_clf:
            results['rf_baseline']['rmse_cliff'] = float(np.mean(fold_rmse_cliff)) if fold_rmse_cliff else float('nan')
        rmse_part = f"  RMSE_cliff={results['rf_baseline'].get('rmse_cliff', float('nan')):.4f}" if not is_clf else ""
        print(f"    RF({'/'.join(_rf_reps) if _rf_reps else best_rep_name}) "
              f"{metric_name}_all={results['rf_baseline']['score_all']:.4f}  "
              f"{metric_name}_cliff={results['rf_baseline']['score_cliff']:.4f}"
              f"{rmse_part}")

    # --- Summary ---
    print(f"\n  --- Summary ({metric_name}_cliff comparison) ---")
    all_methods = []
    for rname, rv in results['per_rep'].items():
        all_methods.append((f"kNN({rname})", rv.get('score_cliff', float('nan'))))
        if 'score_cliff_weighted' in rv:
            all_methods.append((f"kNN({rname})_w", rv.get('score_cliff_weighted', float('nan'))))
    # The pool's own single representation, re-picked per fold.
    _pb = results.get('pool_base_rep') or {}
    if _pb.get('unweighted'):
        all_methods.append(("kNN(pool_base_rep)",
                            _pb['unweighted'].get('score_cliff', float('nan'))))
    if _pb.get('weighted'):
        all_methods.append(("kNN(pool_base_rep)_w",
                            _pb['weighted'].get('score_cliff', float('nan'))))
    # One line per pool arm, both weightings. The lead arm also appears under
    # results['hybrid'] for the readers that expect one hybrid there.
    for _arm, _entry in (results.get('pool_hybrid_arms') or {}).items():
        if _entry.get('unweighted'):
            all_methods.append((f"hybrid_kNN_uw[{_arm}]",
                                _entry['unweighted'].get('score_cliff', float('nan'))))
        if _entry.get('weighted'):
            all_methods.append((f"hybrid_kNN_w[{_arm}]",
                                _entry['weighted'].get('score_cliff', float('nan'))))
    if 'rf_baseline' in results:
        rv = results['rf_baseline']
        all_methods.append((f"RF({rv['rep']})", rv.get('score_cliff', float('nan'))))

    _nan_sentinel = np.inf if lower_is_better else -np.inf
    all_methods.sort(key=lambda x: x[1] if not np.isnan(x[1]) else _nan_sentinel,
                     reverse=not lower_is_better)
    print(f"  {'Method':<25} {metric_name + '_cliff':<12}")
    print("  " + "-" * 37)
    for name, score in all_methods:
        print(f"  {name:<25} {score:<12.4f}")

    # --- Per-fold reselection block (CONTRACT json_convention) ---
    # Reader: fold_reselection(block['candidates_by_model'],
    #                        block['baselines_by_model'], block['lower_is_better']).
    # 2026-09-12: the recipes are the pool's now, so both of these change.
    # exclude_folds: where a pool search pruned mid-flight it filtered its own
    # candidate list using fold 0's scores, so fold 0 contributes no held-out
    # term — the same reading hr_exclude_folds takes, and [] only when the pool
    # says outright that it did not prune.
    # recipe_selection: each fold's recipe was ranked on the other folds'
    # scores of a search that itself ran over every row. That is neither
    # 'whole_dataset' nor 'fold_train_rows'.
    # fold_scores: the metric over the fold's CLIFF rows alone, not over the
    # whole held-out fold, so the leave-one-fold-out re-pick answers the cliff
    # question. A fold whose held-out rows held 1 or fewer cliff molecules is
    # NaN in every arm and contributes nothing; the counts are on each arm as
    # 'n_cliff_rows_by_fold'.
    results['fold_reselection'] = {
        'lower_is_better': bool(lower_is_better),
        'fold_score_stratum': 'cliff_rows_only',
        'exclude_folds': ([] if fold0_pruned is False else [0]),
        'fold0_pruned': (None if fold0_pruned is None else bool(fold0_pruned)),
        'recipe_selection': 'other_folds_of_pool_search',
        'recipe_selection_note': (
            "recipe and single representation ranked by their mean over the "
            "pool search's other folds; the pool search ran over every row, so "
            "this holds out the fold's SCORES, not its rows"),
        'candidates_by_model': reselection_candidates,
        'baselines_by_model': reselected_baselines,
    }

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S04: Activity Cliffs',
                                     has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 's04_activity_cliffs', args.dataset)
