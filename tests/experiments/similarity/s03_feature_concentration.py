#!/usr/bin/env python3
"""S03: Feature Concentration Analysis -- WHY hybrid similarity works.

Paper: Similarity
Conclusion: Feature importance concentrates MI in selected dimensions.
Dependencies: S02 (reads JSON, optional).

For each rep: computes MI of full, selected, random, PCA-matched subsets.
Measures MI density ratio (selected/full) and Gini coefficient of importance scores.

DESCRIPTIVE-ONLY CAVEAT (2026-07-11): every quantity in this experiment
(mi_full, mi_selected, mi_random, mi_pca, mi_density_ratio, effective_rank,
Gini) is a FULL-DATA statistic — the mutual information is measured on the same
rows the feature selection was fit on, and there is NO held-out fold. These are
CONCENTRATION DIAGNOSTICS that explain WHY the hybrid selection keeps the
features it keeps; they are NOT a cross-validated performance comparison and
MUST NOT be reported as a hybrid "win" or an out-of-sample effect. A high
mi_density_ratio means selected features are individually more MI-dense on the
training data, nothing more. For the honest, cross-validated hybrid-vs-single
comparison see s02 (SAC) and s08 (SAC + prediction under CV). The output carries
`descriptive_only=True` and a `_caveat` string so downstream readers cannot
mistake it for a held-out result.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, load_experiment_results,
    get_rule_hybrids, build_hybrid_from_entry,
    estimate_mutual_information_knn, compute_effective_rank,
    FULL_BUDGETS,
)

from kirby.hybrid import compute_feature_importance


# COST / LARGE-N GATE (2026-06-08, RESULTS-AFFECTING, reversible).
#   The kNN mutual-information estimator (estimate_mutual_information_knn) does a
#   per-feature Kraskov neighbour search whose cost grows with the number of
#   rows. On the n=30k–100k DTA / nucleotide datasets this is the wall-buster
#   for s03 (every full/selected/random/PCA arm calls it). The KSG family
#   converges to a stable estimate well before ~2000 samples (Kraskov et al.
#   2004; the estimator is consistent and low-variance once n exceeds a few
#   hundred), so a fixed seeded row subsample gives a statistically equivalent
#   MI value while making s03 cost independent of n.
#   Notes:
#     - This is a MI-ESTIMATION subsample only. It does NOT touch any train/test
#       split (s03 has no CV — it reports MI/density/effective-rank summaries),
#       so it cannot introduce cross-group leakage. effective_rank is still
#       computed on the FULL matrix.
#     - Deterministic (RandomState(42)), so reruns are identical and the
#       subsample does not depend on group structure.
#     - REVERT: set MI_MAX_ROWS = None (or delete _mi_knn_capped and call
#       estimate_mutual_information_knn directly).
MI_MAX_ROWS = 2000


def _mi_knn_capped(X, y, k=5, max_rows=MI_MAX_ROWS, seed=42):
    """estimate_mutual_information_knn on a seeded row subsample (large-n gate).

    Returns (mi, n_components): the MI on a fixed random subset of <= max_rows
    rows, and the number of PCA components the estimator summed that MI over.
    Below the row cap every row is used (identical to calling the estimator
    directly).

    The component count is returned because the MI is NOT measured over the
    features handed in. estimate_mutual_information_knn PCA-reduces to
    max(1, min(max_dim=50, rows // 5)) components whenever the feature count
    exceeds that (_shared.py:2944-2948), then sums a per-component Kraskov
    estimate. Dividing that sum by a feature count of 2048 to get a density
    mixes two different denominators; dividing by the component count gives
    nats per component on both sides of mi_density_ratio.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if max_rows is not None and X.shape[0] > max_rows:
        idx = np.random.RandomState(seed).choice(X.shape[0], max_rows, replace=False)
        X, y = X[idx], y[idx]
    n_rows, n_features = X.shape[0], (X.shape[1] if X.ndim > 1 else 1)
    # Mirrors the estimator's own cap; it PCA-reduces only when the features
    # outnumber it, so the components actually used are the smaller of the two.
    n_components = min(n_features, max(1, min(50, n_rows // 5)))
    return estimate_mutual_information_knn(X, y, k=k), n_components


# The owner's set, in the order each cell builds them (2026-10-02).
_RULE_SLOTS = ('general1', 'general2', 'distance1')


def _cell_key(downstream_model, budget, slot='general1'):
    """The results['per_rep'] key for one (downstream model, budget) cell.

    A Python tuple used to be assigned here, which convert_numpy stringified on
    the way to JSON into keys like "('rf', 25)" — a reader had to parse a repr
    to get the model and the budget back. An explicit string is written instead.
    """
    if slot == 'general1':
        return f'{downstream_model}|{budget}'
    return f'{downstream_model}|{budget}|{slot}'


def _analyse_hybrid(reps, labels, fi, rng):
    """MI, effective rank and the random/PCA comparisons for every
    representation, against the features one hybrid kept from it. `fi` is the
    feature_info create_hybrid returned for that hybrid."""
    per_rep_results = {}
    for rep_name, X in reps.items():
        print(f"\n  {rep_name} (dim={X.shape[1]}):")
        rep_result = {}

        n_features = X.shape[1]
        n_samples = X.shape[0]

        # Full rep MI — let estimate_mutual_information_knn handle PCA
        # internally (max_dim=50, capped at n_samples//5). This ensures
        # consistent dimensionality treatment across full/selected/random.
        X_scaled_full = StandardScaler().fit_transform(X)
        try:
            mi_full, n_comp_full = _mi_knn_capped(
                X_scaled_full, labels, k=5)  # large-n row cap
        except Exception:
            mi_full, n_comp_full = float('nan'), 0
        rep_result['mi_full'] = mi_full
        rep_result['mi_components_full'] = n_comp_full
        rep_result['effective_rank_full'] = compute_effective_rank(X)

        # Selected features for this rep. Read off the feature_info the
        # full-data rebuild returned, not search.fitted_results — that
        # dict is empty here, so this whole comparison never ran.
        # feature_info also holds 'allocation' and 'augmentation_info',
        # hence the isinstance check.
        _fi_rep = fi.get(rep_name) if isinstance(fi, dict) else None
        if isinstance(_fi_rep, dict) and len(
                _fi_rep.get('selected_indices', [])) > 0:
            sel_idx = list(_fi_rep['selected_indices'])
            n_selected = len(sel_idx)

            if n_selected > 0:
                X_sel = X[:, sel_idx]

                # MI of selected
                try:
                    mi_sel, n_comp_sel = _mi_knn_capped(
                        StandardScaler().fit_transform(X_sel), labels, k=5)
                except Exception:
                    mi_sel, n_comp_sel = float('nan'), 0
                rep_result['mi_selected'] = mi_sel
                rep_result['mi_components_selected'] = n_comp_sel
                rep_result['n_selected'] = n_selected
                rep_result['effective_rank_selected'] = compute_effective_rank(X_sel)

                # Random subset of same size
                rand_idx = rng.choice(n_features, n_selected, replace=False)
                X_rand = X[:, rand_idx]
                try:
                    mi_rand, n_comp_rand = _mi_knn_capped(
                        StandardScaler().fit_transform(X_rand), labels, k=5)
                except Exception:
                    mi_rand, n_comp_rand = float('nan'), 0
                rep_result['mi_random'] = mi_rand
                rep_result['mi_components_random'] = n_comp_rand
                rep_result['effective_rank_random'] = compute_effective_rank(X_rand)

                # PCA to same dim — cap at min(50, n//5) so the KSG
                # estimator's internal PCA does not further reduce
                # this arm. Matches the implicit cap applied to the
                # full/selected/random arms inside the helper.
                ksg_cap = min(50, max(1, n_samples // 5))
                n_pca_comp = min(n_selected, n_samples - 1, ksg_cap)
                X_scaled_rep = StandardScaler().fit_transform(X)
                if n_features > 2000:
                    from sklearn.decomposition import TruncatedSVD
                    X_pca_match = TruncatedSVD(n_components=n_pca_comp, random_state=42).fit_transform(X_scaled_rep)
                else:
                    X_pca_match = PCA(n_components=n_pca_comp, random_state=42).fit_transform(X_scaled_rep)
                try:
                    mi_pca, n_comp_pca = _mi_knn_capped(
                        X_pca_match, labels, k=5)
                except Exception:
                    mi_pca, n_comp_pca = float('nan'), 0
                rep_result['mi_pca'] = mi_pca
                rep_result['mi_components_pca'] = n_comp_pca
                rep_result['effective_rank_pca'] = compute_effective_rank(X_pca_match)

                # MI density = nats per PCA COMPONENT, on both sides.
                # It used to be nats per feature: mi_full divided by
                # n_features, which can be 2048, while mi_full is a sum
                # over at most 50 components (_mi_knn_capped's
                # docstring). The two sides of the ratio below were
                # therefore divided by different things. DESCRIPTIVE
                # ONLY: this ratio is a full-data concentration
                # diagnostic (no held-out fold), NOT a hybrid win — see
                # module docstring.
                rep_result['mi_density_full'] = (
                    mi_full / n_comp_full if n_comp_full > 0 else 0)
                rep_result['mi_density_selected'] = (
                    mi_sel / n_comp_sel if n_comp_sel > 0 else 0)
                rep_result['mi_density_unit'] = 'nats_per_pca_component'
                # The ratio stays (secondary since 2026-10-05). It is NaN, never
                # inf or a huge number, when the full-set density is 0 or
                # not finite (the estimator clamps MI at 0).
                _d_full = rep_result['mi_density_full']
                rep_result['mi_density_ratio'] = (
                    rep_result['mi_density_selected'] / _d_full
                    if np.isfinite(_d_full) and _d_full > 0 else float('nan')
                )
                # HEADLINE (2026-10-05, owner delegated, 2026-10-05 — a
                # delegated default): MI of the selected features minus MI of
                # a random subset of the same number of features from the same
                # representation, in nats. Both come from _mi_knn_capped on
                # the same rows; no new estimate is made. Unlike the ratio it
                # is defined when the full-set MI is 0.
                rep_result['mi_selected_minus_random'] = (
                    float(mi_sel - mi_rand)
                    if np.isfinite(mi_sel) and np.isfinite(mi_rand)
                    else float('nan'))
                rep_result['mi_selected_minus_random_unit'] = 'nats'

                print(f"    MI selected - random: "
                      f"{rep_result['mi_selected_minus_random']:.4f} nats "
                      f"(headline)")
                print(f"    MI full={mi_full:.4f} ({n_features} features "
                      f"-> {n_comp_full} components), "
                      f"selected={mi_sel:.4f} ({n_selected} features "
                      f"-> {n_comp_sel} components), "
                      f"random={mi_rand:.4f}, pca={mi_pca:.4f}")
                print(f"    MI density ratio (selected/full): "
                      f"{rep_result['mi_density_ratio']:.2f}x")
            else:
                print(f"    No features selected from this rep")
        else:
            print(f"    Not in hybrid selection")

        per_rep_results[rep_name] = rep_result
    return per_rep_results


# =============================================================================
# EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Explain WHY hybrid similarity works: higher MI density, better effective
    rank distribution in selected features."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S03: FEATURE CONCENTRATION ANALYSIS")
    print("=" * 70)

    weight_methods = kwargs.get('weight_methods', ['lightgbm_gain'])
    task_type = data['task_type']

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    results = {'per_rep': {}, 'gini': {}, 'hybrid_recipe': {},
               'rule_missing': []}

    # DESCRIPTIVE-ONLY marker (2026-07-11, decision B). s03 has no held-out
    # fold: MI/density/rank/Gini are all measured on the FULL dataset with the
    # selection fit on the FULL dataset. These flags stop any reader from
    # treating mi_density_ratio (or Gini) as a cross-validated hybrid "win" —
    # it is a concentration diagnostic, not an out-of-sample performance number.
    results['descriptive_only'] = True
    # 2026-10-05, owner delegated, 2026-10-05 (delegated default): the headline
    # per representation is mi_selected_minus_random (nats); mi_density_ratio
    # is kept beside it as a secondary number.
    results['headline'] = 'mi_selected_minus_random'
    results['headline_unit'] = 'nats'
    results['_caveat'] = (
        "DESCRIPTIVE ONLY — no held-out fold. mi_full/mi_selected/mi_random/"
        "mi_pca/mi_density_ratio/effective_rank/Gini are all full-data statistics "
        "computed with feature selection fit on the full data. They explain WHY "
        "the hybrid keeps the features it keeps; they are NOT a cross-validated "
        "comparison and MUST NOT be reported as a hybrid win. See s02/s08 for the "
        "honest, held-out hybrid-vs-single comparison."
    )

    # Sweep budgets × downstream models. Scoped to ['rf', 'lightgbm', 'dnn'] —
    # the full DOWNSTREAM_MODELS set would explode result size with little gain.
    # One tree ensemble, one booster, one neural. dnn replaced mlp 2026-09-04
    # (owner): mlp is tier 'extra' now and no search scores it.
    budgets = kwargs.get('budgets', config.get('budgets', FULL_BUDGETS))
    downstream_models = kwargs.get('downstream_models', ['rf', 'lightgbm', 'dnn'])
    augmentations = data.get('augmentations')

    rng = np.random.RandomState(42)

    # Each (model, budget) cell is written to
    # s03_feature_concentration/<dataset>.partial.jsonl when it finishes and read
    # back on the next attempt (2026-09-25; _checkpoint.py). The random subsets
    # of every cell are drawn from the ONE rng above, so a cell read back also
    # restores the rng to the state it was left in after that cell; the next
    # cell computed draws exactly the subset an uninterrupted run draws. A cell
    # whose search failed is stored as {'error'} in results and is not saved to
    # the sidecar: it consumed no draw, and it is retried.
    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)
    _ck = Checkpoint(
        experiment_sidecar(output_dir, 's03_feature_concentration', dataset_name),
        run_identity('s03_feature_concentration', dataset_name, data, config,
                     kwargs=kwargs, budgets=list(budgets),
                     downstream_models=list(downstream_models),
                     hybrid_set='top2_general_top1_distance'))

    for downstream_model in downstream_models:
        for budget in budgets:
            print(f"\n  ── model={downstream_model}, budget={budget} ──")
            # THE OWNER'S SET (top 2 general + top 1 distance per model per
            # family, 2026-10-02). Until that date this cell read only the
            # single best general recipe (get_fitted_pool_hybrid). Each of the
            # three recipes get_rule_hybrids returns is now built on every row
            # and analysed. Keys: '<model>|<budget>' is general1, as before;
            # '<model>|<budget>|general2' and '<model>|<budget>|distance1' are
            # new. A slot the pool could not fill is in results['rule_missing'].
            try:
                rule = get_rule_hybrids(
                    reps, labels, groups,
                    downstream_model=downstream_model, budget=budget,
                    augmentations=augmentations,
                    dataset_name=dataset_name, task_type=task_type)
            except Exception as e:
                print(f"  pool read failed: {e}")
                results['per_rep'][_cell_key(downstream_model, budget)] = {
                    'error': str(e)}
                results['rule_missing'].append(
                    f'{downstream_model} budget={budget} features: pool read '
                    f'raised {type(e).__name__}: {e}')
                continue
            results['rule_missing'] += [
                f'{downstream_model} budget={budget} features: {m}'
                for m in rule['missing']]
            _by_slot = {h['slot']: h for h in rule['hybrids']}
            for _slot in _RULE_SLOTS:
                _key = _cell_key(downstream_model, budget, _slot)
                _ckey = dict(model=downstream_model, budget=budget, slot=_slot)
                if _ck.done(**_ckey):
                    _saved = _ck.get(**_ckey)
                    results['per_rep'][_key] = _saved['cell']
                    # A cell saved before 2026-10-05 has no headline field;
                    # it is the difference of two numbers the cell already
                    # holds (owner delegated, 2026-10-05).
                    for _rr in (_saved['cell'] or {}).values():
                        if (isinstance(_rr, dict) and 'mi_selected' in _rr
                                and 'mi_selected_minus_random' not in _rr):
                            _a, _b = _rr.get('mi_selected'), _rr.get('mi_random')
                            _ok = (_a is not None and _b is not None
                                   and np.isfinite(_a) and np.isfinite(_b))
                            _rr['mi_selected_minus_random'] = (
                                float(_a - _b) if _ok else float('nan'))
                            _rr['mi_selected_minus_random_unit'] = 'nats'
                    results['hybrid_recipe'][_key] = _saved.get('recipe')
                    rng.set_state(_saved['rng_state'])
                    continue
                entry = _by_slot.get(_slot)
                if entry is None:
                    results['per_rep'][_key] = {
                        'error': 'slot empty: ' + ('; '.join(rule['missing'])
                                                   or 'no recipe returned')}
                    continue
                print(f"\n  -- {_slot}: {entry.get('objective')} "
                      f"{entry.get('tier')}/{entry.get('aug_condition')} "
                      f"{entry.get('recipe')} --")
                try:
                    _X_hybrid, fi = build_hybrid_from_entry(
                        reps, labels, groups, entry,
                        augmentations=augmentations)
                except Exception as e:
                    print(f"  hybrid build failed: {e}")
                    results['per_rep'][_key] = {'error': str(e)}
                    continue
                _recipe = {k: entry.get(k) for k in (
                    'slot', 'objective', 'tier', 'aug_condition', 'recipe',
                    'budget', 'budget_relaxed', 'metric_name',
                    'score_other_folds')}
                results['hybrid_recipe'][_key] = _recipe
                per_rep_results = _analyse_hybrid(reps, labels, fi, rng)
                results['per_rep'][_key] = per_rep_results
                _ck.put({'cell': per_rep_results, 'recipe': _recipe,
                         'rng_state': rng.get_state()}, **_ckey)

    # Gini coefficient of importance scores per method
    for wm in weight_methods:
        print(f"\n  Gini coefficients ({wm}):")
        try:
            importance = compute_feature_importance(reps, labels, method=wm, task_type=task_type)
            for rep_name, scores in importance.items():
                scores_sorted = np.sort(np.abs(scores))
                n = len(scores_sorted)
                if n == 0 or scores_sorted.sum() == 0:
                    gini = 0.0
                else:
                    index = np.arange(1, n + 1)
                    gini = float((2 * np.sum(index * scores_sorted) /
                                 (n * np.sum(scores_sorted))) - (n + 1) / n)
                results['gini'].setdefault(wm, {})[rep_name] = gini
                print(f"    {rep_name}: Gini={gini:.3f}")
        except Exception as e:
            print(f"    {wm} failed: {e}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S03: Feature Concentration Analysis',
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
    save_experiment_results(results, args.output, 's03_feature_concentration', args.dataset)
