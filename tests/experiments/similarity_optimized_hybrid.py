#!/usr/bin/env python3
"""
Similarity-Optimized Hybrid — a hybrid found by optimizing SIMILARITY PRESERVATION
=================================================================================

KIRBy's hybrid search normally optimizes DOWNSTREAM PREDICTIVE PERFORMANCE. This
standalone experiment runs the SAME search machinery with a different objective —
``objective='similarity_sac'`` (find_best_hybrid) — which selects the hybrid whose
GEOMETRY best preserves "two similar molecules have similar properties" (held-out
SAC: Spearman between pairwise rep-space distance and |delta property|).

It answers two questions on small-molecule datasets, honestly (held-out, with
multi-seed spread — never the in-sample/definitional SAC of the older s02 path):

  Payoff 1 — SIMILARITY QUALITY: does the similarity-optimized hybrid preserve
             similarity on HELD-OUT folds better than the predictive-optimized
             hybrid and the best single rep? (held-out SAC + violation rate)

  Payoff 2 — NOISE MITIGATION: does its smoother geometry translate into better
             kNN label-denoising recovery under injected label noise?

The two hybrids are found by SEPARATE searches (different objective) — this is
NOT re-scoring the predictive hybrid; the similarity search discovers a
different rep composition (see the divergence check in the engine tests).

Datasets (sound, Murcko-scaffold split, precompute-built per DATASET_STATUS
2026-07-07): caco2, logd, herg (clf), herg_ki, qm9, adme_hlm, adme_rlm,
adme_mdr1. Avoid chembl_qsar (never run) and herg's stale graph_kernel rep.

Search space is kept small on purpose (budgets < 300, aggressive pruning) per
the project constraint.

Run:
    python tests/experiments/similarity_optimized_hybrid.py --datasets logd herg \
        --seeds 3 --output tests/results/sim_optimized_hybrid
"""
import argparse
import json
import os
import sys
import numpy as np

# --- Path bootstrap so `experiments...` and `kirby...` import cleanly ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS = os.path.dirname(_HERE)
for _p in (_TESTS, os.path.join(_TESTS, 'experiments')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from kirby.find_best_hybrid import (
    find_best_hybrid, OBJECTIVE_SIMILARITY_SAC, OBJECTIVE_PREDICTIVE,
)
from kirby.hybrid import create_hybrid, apply_feature_selection, compute_feature_importance
from kirby.utils.noise_mitigation_methods import (
    get_mitigation_method,
    # float-returning SAC (same one the engine optimizes) — the similarity_helpers
    # twin returns a 5-tuple, so use this to avoid confusion.
    similarity_activity_correlation_scalable,
)

from experiments._shared import load_data, cv_split
from experiments.similarity.similarity_helpers import violation_rate_scalable

# Small-molecule datasets that are methodologically sound (DATASET_STATUS §5/§9).
DEFAULT_DATASETS = ['caco2', 'logd', 'herg', 'herg_ki', 'qm9',
                    'adme_hlm', 'adme_rlm', 'adme_mdr1']

# Budgets kept < 300 (project constraint); aggressive pruning below.
SIM_BUDGETS = [10, 25, 50, 100, 200]
# Allocation methods that are objective-aware (route through _evaluate_allocation).
SIM_ALLOCATIONS = ['greedy', 'top_2', 'top_3', 'performance_weighted', 'dominant_greedy']
ADAPTIVE_PRUNING = 0.35   # aggressive: prune strategies >35% below the running best
N_FOLDS = 3
NOISE_FRACTIONS = [0.1, 0.2, 0.3]
DENOISE_METHOD = 'distance_weighted'   # kNN-in-rep-space label denoiser


def _model(is_clf):
    if is_clf:
        return RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    return RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)


def _score(model, X_tr, y_tr, X_va, y_va, is_clf):
    from sklearn.metrics import f1_score, r2_score
    m = _model(is_clf)
    m.fit(X_tr, y_tr)
    p = m.predict(X_va)
    return f1_score(y_va, p, average='weighted') if is_clf else r2_score(y_va, p)


def _standardize_pair(X_tr, X_va):
    sc = StandardScaler().fit(X_tr)
    return sc.transform(X_tr), sc.transform(X_va)


def _search(reps, labels, groups, is_clf, objective, tier, seed,
            dataset_name=None, cache_dir=None):
    """Run one objective-specific hybrid search; return (best Blueprint, baselines).

    Caching is ALWAYS on when a dataset_name is given (which it always is):
    reuses per-rep baseline caches, per-fold importance caches, and any matching
    result rows already present, and caches this run for cheap reruns. cache_dir
    is resolved by the caller to a concrete path (falls back to $KIRBY_CACHE_DIR
    or ~/.cache/kirby) so reuse of the existing precompute caches is guaranteed.
    """
    task_type = 'classification' if is_clf else 'regression'
    use_cache = dataset_name is not None
    kw = dict(
        base_reps=reps, labels=labels, groups=groups, task_type=task_type,
        speed='full', budgets=SIM_BUDGETS, n_folds=N_FOLDS,
        models=['rf'], internal_evaluators=['match'],
        allocation_methods=SIM_ALLOCATIONS, importance_methods=['lightgbm_gain'],
        augmentation_strategy='none', adaptive_pruning=ADAPTIVE_PRUNING,
        seed=seed, verbose=False, return_fitted=False,
        cache=use_cache, cache_dir=cache_dir, dataset_name=dataset_name,
        cv_splitter_id=(f'k{N_FOLDS}s{seed}' if use_cache else None),
    )
    if objective == OBJECTIVE_SIMILARITY_SAC:
        res = find_best_hybrid(objective=OBJECTIVE_SIMILARITY_SAC, tier=tier, **kw)
    else:
        res = find_best_hybrid(metric=('rmse' if not is_clf else 'f1_weighted'),
                               tier=tier, **kw)
    by_budget = res.best_by_model_budget.get('rf', {})
    best = max(by_budget.values(),
               key=lambda r: r['score_mean']) if by_budget else None
    return (best['blueprint'] if best else None), res.baselines


def _predictive_blueprint(reps, labels, groups, is_clf, seed, dataset_name,
                          data, cache_dir):
    """Predictive-optimized comparator blueprint.

    Deliberately runs the SAME controlled search as the similarity side (rf
    model, same small budget grid, tier 'general') and differs ONLY in the
    objective — an apples-to-apples A/B. This is both fair and safe.

    Note: we do NOT pull from the existing full precompute pool
    (get_best_search_for_downstream). On a cache miss that helper launches a
    full DEFAULT-model search (svm/mlp/…) with FULL_BUDGETS, which is heavy and
    was observed to segfault on large molecular sets — and it would also compare
    a blueprint tuned over a different model/budget space, muddying the A/B.
    Cache reuse still happens where it is safe: per-rep baseline caches, per-fold
    importance caches, and this run's own SQLite result store (tier 'general').
    """
    bp, _ = _search(reps, labels, groups, is_clf, OBJECTIVE_PREDICTIVE,
                    'general', seed, dataset_name=dataset_name, cache_dir=cache_dir)
    return bp, 'search'


def _build_hybrid_fold(reps_tr, reps_va, y_tr, bp, objective, is_clf):
    """Rebuild a blueprint's hybrid on the TRAIN fold, apply selection to VAL.
    Returns (X_tr, X_va) already standardized (scaler fit on train)."""
    model = None if objective == OBJECTIVE_SIMILARITY_SAC else _model(is_clf)
    metric_fn = None
    if objective == OBJECTIVE_PREDICTIVE:
        from sklearn.metrics import f1_score, mean_squared_error
        metric_fn = ((lambda y, p: f1_score(y, p, average='weighted')) if is_clf
                     else (lambda y, p: -np.sqrt(mean_squared_error(y, p))))
    # Within-rep feature selection stays importance-based (a cheap predictive
    # prior); the ALLOCATION across reps is what the objective drives. greedy/
    # dominant_greedy require importance to be supplied explicitly.
    imp = compute_feature_importance(reps_tr, y_tr, method=bp.importance_method)
    X_tr, fi = create_hybrid(
        reps_tr, y_tr, allocation_method=bp.allocation_method, budget=bp.budget,
        step_size=bp.step_size, importance_method=bp.importance_method,
        internal_evaluator=bp.internal_evaluator, augmentation_strategy='none',
        model=model, metric_fn=metric_fn, lower_is_better=False,
        objective=objective, is_clf=is_clf, precomputed_importance=imp,
    )
    X_va = apply_feature_selection(reps_va, fi)
    return _standardize_pair(X_tr, X_va)


def _heldout_similarity(reps, labels, groups, is_clf, bp_sim, bp_pred, best_rep, seed):
    """Payoff 1: per-fold held-out SAC + violation rate for sim-hybrid,
    predictive-hybrid, and best single rep. Returns per-space fold lists."""
    out = {k: {'sac': [], 'violation': []} for k in ('sim', 'pred', 'rep')}
    for tr, va in cv_split(groups, 3, next(iter(reps.values())), labels, seed=seed):
        reps_tr = {k: v[tr] for k, v in reps.items()}
        reps_va = {k: v[va] for k, v in reps.items()}
        y_va = labels[va]
        spaces = {}
        if bp_sim is not None:
            _, Xs = _build_hybrid_fold(reps_tr, reps_va, labels[tr], bp_sim,
                                       OBJECTIVE_SIMILARITY_SAC, is_clf)
            spaces['sim'] = Xs
        if bp_pred is not None:
            _, Xp = _build_hybrid_fold(reps_tr, reps_va, labels[tr], bp_pred,
                                       OBJECTIVE_PREDICTIVE, is_clf)
            spaces['pred'] = Xp
        # best single rep: standardize on train
        _, Xr = _standardize_pair(reps_tr[best_rep], reps_va[best_rep])
        spaces['rep'] = Xr
        for name, Xv in spaces.items():
            out[name]['sac'].append(
                similarity_activity_correlation_scalable(Xv, y_va, is_clf=is_clf))
            out[name]['violation'].append(
                violation_rate_scalable(Xv, y_va, is_clf=is_clf))
    return out


def _noise_recovery(reps, labels, groups, is_clf, bp_sim, bp_pred, best_rep,
                    noise_frac, seed):
    """Payoff 2: kNN-denoising recovery_rate under injected label noise, per
    fold, for each feature space. recovery = (cleaned-noisy)/(clean-noisy)."""
    from experiments.noise.n07_remediation import _inject_noise  # reuse
    out = {k: [] for k in ('sim', 'pred', 'rep')}
    for tr, va in cv_split(groups, 3, next(iter(reps.values())), labels, seed=seed):
        reps_tr = {k: v[tr] for k, v in reps.items()}
        reps_va = {k: v[va] for k, v in reps.items()}
        y_tr, y_va = labels[tr], labels[va]
        y_tr_noisy, _ = _inject_noise(y_tr, noise_frac, seed=seed, is_clf=is_clf)
        spaces = {}
        if bp_sim is not None:
            spaces['sim'] = _build_hybrid_fold(reps_tr, reps_va, y_tr, bp_sim,
                                               OBJECTIVE_SIMILARITY_SAC, is_clf)
        if bp_pred is not None:
            spaces['pred'] = _build_hybrid_fold(reps_tr, reps_va, y_tr, bp_pred,
                                                OBJECTIVE_PREDICTIVE, is_clf)
        spaces['rep'] = _standardize_pair(reps_tr[best_rep], reps_va[best_rep])
        for name, (X_tr, X_va) in spaces.items():
            clean = _score(None, X_tr, y_tr, X_va, y_va, is_clf)
            noisy = _score(None, X_tr, y_tr_noisy, X_va, y_va, is_clf)
            filt = get_mitigation_method(DENOISE_METHOD)
            try:
                Xc, yc, _rm = filt.clean_data(X_tr, y_tr_noisy)
                cleaned = _score(None, Xc, yc, X_va, y_va, is_clf)
            except Exception:
                cleaned = noisy
            denom = clean - noisy
            rec = (cleaned - noisy) / denom if abs(denom) > 1e-9 else np.nan
            out[name].append(rec)
    return out


def _ci(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return {'mean': None, 'lo': None, 'hi': None, 'n': 0}
    a = np.array(vals, float)
    return {'mean': float(a.mean()), 'lo': float(np.percentile(a, 2.5)),
            'hi': float(np.percentile(a, 97.5)), 'n': len(a),
            'std': float(a.std())}


def run_dataset(dataset_name, seeds, verbose=True, cache_dir=None, rep_names=None):
    if verbose:
        print(f"\n{'='*70}\n{dataset_name}\n{'='*70}")
    data = load_data(dataset_name, rep_names=rep_names)
    reps = {k: np.asarray(v) for k, v in data['reps'].items()}
    # Drop the known-stale herg graph_kernel rep (DATASET_STATUS §5).
    if dataset_name == 'herg':
        reps.pop('graph_kernel', None)
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    is_clf = data.get('task_type') == 'classification'

    sim_sac, pred_sac, rep_sac = [], [], []
    sim_vio, pred_vio, rep_vio = [], [], []
    noise = {f'{fr}': {'sim': [], 'pred': [], 'rep': []} for fr in NOISE_FRACTIONS}
    picks = []

    for seed in range(seeds):
        bp_sim, base_sim = _search(reps, labels, groups, is_clf,
                                   OBJECTIVE_SIMILARITY_SAC, 'general_sim', 42 + seed,
                                   dataset_name=dataset_name, cache_dir=cache_dir)
        # Predictive comparator: reuse the EXISTING precompute pool where possible.
        bp_pred, pred_src = _predictive_blueprint(
            reps, labels, groups, is_clf, 42 + seed, dataset_name, data, cache_dir)
        # Best single rep by held-out SAC (from the sim baseline).
        best_rep = max(reps.keys(),
                       key=lambda r: base_sim.get(r, {}).get('rf', -np.inf))
        picks.append({'seed': 42 + seed,
                      'sim_alloc': getattr(bp_sim, 'allocation_method', None),
                      'sim_budget': getattr(bp_sim, 'budget', None),
                      'pred_alloc': getattr(bp_pred, 'allocation_method', None),
                      'pred_budget': getattr(bp_pred, 'budget', None),
                      'pred_source': pred_src,
                      'best_rep': best_rep})

        p1 = _heldout_similarity(reps, labels, groups, is_clf,
                                 bp_sim, bp_pred, best_rep, 42 + seed)
        sim_sac += p1['sim']['sac']; pred_sac += p1['pred']['sac']; rep_sac += p1['rep']['sac']
        sim_vio += p1['sim']['violation']; pred_vio += p1['pred']['violation']; rep_vio += p1['rep']['violation']

        for fr in NOISE_FRACTIONS:
            r = _noise_recovery(reps, labels, groups, is_clf,
                                bp_sim, bp_pred, best_rep, fr, 42 + seed)
            for sp in ('sim', 'pred', 'rep'):
                noise[f'{fr}'][sp] += r[sp]

    result = {
        'dataset': dataset_name, 'task_type': 'classification' if is_clf else 'regression',
        'n_samples': int(len(labels)), 'n_reps': len(reps), 'seeds': seeds,
        'picks': picks,
        'payoff1_heldout_similarity': {
            'sac': {'sim': _ci(sim_sac), 'pred': _ci(pred_sac), 'rep': _ci(rep_sac)},
            'violation': {'sim': _ci(sim_vio), 'pred': _ci(pred_vio), 'rep': _ci(rep_vio)},
        },
        'payoff2_noise_recovery': {
            fr: {sp: _ci(noise[fr][sp]) for sp in ('sim', 'pred', 'rep')}
            for fr in noise
        },
    }
    if verbose:
        s = result['payoff1_heldout_similarity']['sac']
        print(f"  held-out SAC   sim={s['sim']['mean']:.3f}  "
              f"pred={s['pred']['mean']:.3f}  best_rep={s['rep']['mean']:.3f}")
        for fr in NOISE_FRACTIONS:
            n = result['payoff2_noise_recovery'][f'{fr}']
            def _m(d): return d['mean'] if d['mean'] is not None else float('nan')
            print(f"  recovery@{fr}   sim={_m(n['sim']):.3f}  "
                  f"pred={_m(n['pred']):.3f}  best_rep={_m(n['rep']):.3f}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--reps', nargs='+', default=None,
                    help='Explicit rep list override (default: DATASET_REPS general).')
    ap.add_argument('--output', default='tests/results/sim_optimized_hybrid')
    ap.add_argument('--cache-dir', default=os.environ.get('KIRBY_CACHE_DIR'),
                    help='Cache root. Reuses per-rep baseline/importance caches; '
                         'caches the sim search under tier general_sim. Defaults '
                         'to $KIRBY_CACHE_DIR or ~/.cache/kirby.')
    args = ap.parse_args()

    # Resolve to a concrete cache dir so caching is ALWAYS in place (an empty
    # --cache-dir / unset $KIRBY_CACHE_DIR still gets the standard cache root,
    # which is where the existing precompute baseline/importance caches live).
    cache_dir = (args.cache_dir or os.environ.get('KIRBY_CACHE_DIR')
                 or os.path.expanduser('~/.cache/kirby'))

    os.makedirs(args.output, exist_ok=True)
    print(f"[cache] caching ON, cache_dir={cache_dir}")
    all_results = {}
    for ds in args.datasets:
        try:
            all_results[ds] = run_dataset(ds, args.seeds, cache_dir=cache_dir,
                                          rep_names=args.reps)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results[ds] = {'error': str(e)}
        with open(os.path.join(args.output, f'{ds}.json'), 'w') as f:
            json.dump(all_results[ds], f, indent=2)
    with open(os.path.join(args.output, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote results to {args.output}")


if __name__ == '__main__':
    main()
