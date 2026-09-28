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
    get_fitted_hybrid, get_fitted_pool_hybrid,
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
    Below the row cap every row is used. Every block is PCA-rotated here to at
    most the component cap, keeping only components with non-zero variance
    (see the comment in the body), so every block is summed over decorrelated
    components and counted by its rank, not by the cap.

    The component count is returned because the MI is NOT measured over the
    features handed in. estimate_mutual_information_knn PCA-reduces to
    max(1, min(max_dim=50, rows // 5)) components whenever the feature count
    exceeds that (_shared.py:2944-2948), then sums a per-component Kraskov
    estimate. Dividing that sum by a feature count of 2048 to get a density
    mixes two different denominators; dividing by the component count gives
    nats per component on both sides of mi_density_ratio.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y)
    if max_rows is not None and X.shape[0] > max_rows:
        idx = np.random.RandomState(seed).choice(X.shape[0], max_rows, replace=False)
        X, y = X[idx], y[idx]
    n_rows, n_features = X.shape
    # Mirrors the estimator's own cap.
    cap = max(1, min(50, n_rows // 5))
    # Every block is rotated here, over the cap and under it, keeping only
    # components with non-zero variance, so the estimator (handed at most
    # `cap` columns) never runs its own PCA.
    #
    # Under the cap the estimator skips its PCA and sums per-feature MI over
    # the raw, correlated features, so information two features share is
    # counted twice; the full representation (over the cap) was always
    # decorrelated, so the two sides of mi_density_ratio were measured two
    # different ways. (dirty local 2026-09-28)
    #
    # Over the cap the estimator took exactly `cap` components whatever the
    # block's rank, and the count returned was `cap`. A block of 60 columns of
    # which 30 are constant -- a random draw from a sparse fingerprint -- gave
    # the same MI as its 30 live columns (0.1897 against 0.1895 nats on a
    # synthetic check) over 50 components instead of 30, so its MI per
    # component read 5/3 too low. Same rule on both sides now. Duplicate
    # fingerprint bits and constant columns give zero-variance components,
    # which the estimator would rescale to unit variance and count.
    # (dirty local 2026-09-28, second pass)
    #
    # fit_transform, not fit then transform: over the cap sklearn picks its
    # randomized solver, whose fit_transform (U * S) and transform (X @ V) are
    # not the same matrix when the spectrum is flat, and the estimator itself
    # used fit_transform. Measured on 300 standard-normal columns: 0.130 nats
    # through fit_transform, 0.042 through transform. fit_transform keeps the
    # full representation's number identical to what the estimator gave.
    Xs = StandardScaler().fit_transform(X)
    n_fit = min(cap, n_features, n_rows)
    pca = PCA(n_components=n_fit, random_state=42)
    Z = pca.fit_transform(Xs)
    ev = pca.explained_variance_
    keep = ev > 1e-10 * max(float(ev.max()), 1e-300)
    n_components = int(keep.sum())
    if n_components == 0:
        return 0.0, 0
    X = Z[:, keep]
    return estimate_mutual_information_knn(X, y, k=k), n_components


def _effective_rank_from_singular_values(s):
    """exp(entropy of normalised singular values), as compute_effective_rank.

    Used for the PCA-matched block. compute_effective_rank standardises its
    input first, and standardising PCA scores gives every component unit
    variance, so it returns the component count whatever the spectrum.
    """
    s = np.asarray(s, dtype=float)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


# How many random subsets of the selected size are drawn per representation.
# One draw was a single sample of the control; the median over draws is
# reported (a median across replicates), and every draw is written.
N_RANDOM_DRAWS = 5


def _cell_key(downstream_model, budget):
    """The results['per_rep'] key for one (downstream model, budget) cell.

    A Python tuple used to be assigned here, which convert_numpy stringified on
    the way to JSON into keys like "('rf', 25)" — a reader had to parse a repr
    to get the model and the budget back. An explicit string is written instead.
    """
    return f'{downstream_model}|{budget}'


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

    results = {'per_rep': {}, 'gini': {}}

    # DESCRIPTIVE-ONLY marker (2026-07-11, decision B). s03 has no held-out
    # fold: MI/density/rank/Gini are all measured on the FULL dataset with the
    # selection fit on the FULL dataset. These flags stop any reader from
    # treating mi_density_ratio (or Gini) as a cross-validated hybrid "win" —
    # it is a concentration diagnostic, not an out-of-sample performance number.
    results['descriptive_only'] = True
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
                     downstream_models=list(downstream_models)))

    for downstream_model in downstream_models:
        for budget in budgets:
            print(f"\n  ── model={downstream_model}, budget={budget} ──")
            _ckey = dict(model=downstream_model, budget=budget)
            if _ck.done(**_ckey):
                _saved = _ck.get(**_ckey)
                results['per_rep'][_cell_key(downstream_model, budget)] = \
                    _saved['cell']
                rng.set_state(_saved['rng_state'])
                continue
            try:
                X_hybrid, fi, search, _meta = get_fitted_pool_hybrid(
                    reps, labels, groups,
                    downstream_model=downstream_model, budget=budget,
                    augmentations=augmentations,
                    dataset_name=dataset_name, task_type=task_type)
            except Exception as e:
                print(f"  find_best_hybrid failed: {e}")
                results['per_rep'][_cell_key(downstream_model, budget)] = {
                    'error': str(e)}
                continue

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

                        # Random subsets of the same size, N_RANDOM_DRAWS of
                        # them; the median over draws is the reported value.
                        _mi_r, _nc_r, _er_r = [], [], []
                        for _ in range(N_RANDOM_DRAWS):
                            rand_idx = rng.choice(n_features, n_selected, replace=False)
                            X_rand = X[:, rand_idx]
                            try:
                                _m, _c = _mi_knn_capped(
                                    StandardScaler().fit_transform(X_rand), labels, k=5)
                            except Exception:
                                _m, _c = float('nan'), 0
                            _mi_r.append(_m)
                            _nc_r.append(_c)
                            _er_r.append(compute_effective_rank(X_rand))
                        _dens_r = [m / c if c > 0 else float('nan')
                                   for m, c in zip(_mi_r, _nc_r)]
                        # nanmedian of an all-NaN list warns and returns NaN.
                        _med = (lambda v: float(np.nanmedian(v))
                                if np.any(np.isfinite(v)) else float('nan'))
                        mi_rand = _med(_mi_r)
                        rep_result['mi_random'] = mi_rand
                        rep_result['mi_random_draws'] = [float(v) for v in _mi_r]
                        rep_result['mi_components_random'] = int(np.median(_nc_r))
                        rep_result['effective_rank_random'] = _med(_er_r)
                        rep_result['n_random_draws'] = N_RANDOM_DRAWS

                        # PCA to same dim — cap at min(50, n//5) so the KSG
                        # estimator's internal PCA does not further reduce
                        # this arm. Matches the implicit cap applied to the
                        # full/selected/random arms inside the helper.
                        ksg_cap = min(50, max(1, n_samples // 5))
                        n_pca_comp = min(n_selected, n_samples - 1, ksg_cap)
                        X_scaled_rep = StandardScaler().fit_transform(X)
                        if n_features > 2000:
                            from sklearn.decomposition import TruncatedSVD
                            _pca_model = TruncatedSVD(n_components=n_pca_comp, random_state=42)
                        else:
                            _pca_model = PCA(n_components=n_pca_comp, random_state=42)
                        X_pca_match = _pca_model.fit_transform(X_scaled_rep)
                        try:
                            mi_pca, n_comp_pca = _mi_knn_capped(
                                X_pca_match, labels, k=5)
                        except Exception:
                            mi_pca, n_comp_pca = float('nan'), 0
                        rep_result['mi_pca'] = mi_pca
                        rep_result['mi_components_pca'] = n_comp_pca
                        # From the singular values of the standardised
                        # representation, as the other blocks are. Standardising
                        # the PCA scores instead returned n_pca_comp always.
                        rep_result['effective_rank_pca'] = \
                            _effective_rank_from_singular_values(
                                _pca_model.singular_values_)

                        # MI density = nats per PCA COMPONENT, on both sides.
                        # It used to be nats per feature: mi_full divided by
                        # n_features, which can be 2048, while mi_full is a sum
                        # over at most 50 components (_mi_knn_capped's
                        # docstring). The two sides of the ratio below were
                        # therefore divided by different things. DESCRIPTIVE
                        # ONLY: this ratio is a full-data concentration
                        # diagnostic (no held-out fold), NOT a hybrid win — see
                        # module docstring.
                        # A failed estimate (NaN, 0 components) is NaN here,
                        # not 0: 0 read as "no concentration" and gave a ratio
                        # of 0 when only the selected block failed.
                        def _dens(m, c):
                            return m / c if c > 0 else float('nan')

                        def _ratio(a, b):
                            return (a / b if np.isfinite(a) and np.isfinite(b)
                                    and b > 0 else float('nan'))
                        rep_result['mi_density_full'] = _dens(mi_full, n_comp_full)
                        rep_result['mi_density_selected'] = _dens(mi_sel, n_comp_sel)
                        rep_result['mi_density_random'] = _med(_dens_r)
                        rep_result['mi_density_pca'] = _dens(mi_pca, n_comp_pca)
                        rep_result['mi_density_unit'] = 'nats_per_pca_component'
                        rep_result['mi_density_ratio'] = _ratio(
                            rep_result['mi_density_selected'],
                            rep_result['mi_density_full'])
                        # Selected against the two blocks of the same size.
                        # The ratio against the full representation divides a
                        # sum over up to 50 components by a sum over as few as
                        # the selected block has, so it moves with that count.
                        rep_result['mi_density_ratio_vs_random'] = _ratio(
                            rep_result['mi_density_selected'],
                            rep_result['mi_density_random'])
                        rep_result['mi_density_ratio_vs_pca'] = _ratio(
                            rep_result['mi_density_selected'],
                            rep_result['mi_density_pca'])
                        # The same comparisons as differences, in nats per
                        # component. The estimator clamps at 0 after its
                        # permutation correction, so a control block often
                        # scores exactly 0 and the ratio is then undefined.
                        rep_result['mi_density_minus_random'] = (
                            rep_result['mi_density_selected']
                            - rep_result['mi_density_random'])
                        rep_result['mi_density_minus_pca'] = (
                            rep_result['mi_density_selected']
                            - rep_result['mi_density_pca'])

                        print(f"    MI full={mi_full:.4f} ({n_features} features "
                              f"-> {n_comp_full} components), "
                              f"selected={mi_sel:.4f} ({n_selected} features "
                              f"-> {n_comp_sel} components), "
                              f"random={mi_rand:.4f}, pca={mi_pca:.4f}")
                        print(f"    MI per component, selected / full = "
                              f"{rep_result['mi_density_ratio']:.2f}, "
                              f"selected / random = "
                              f"{rep_result['mi_density_ratio_vs_random']:.2f}, "
                              f"selected / PCA = "
                              f"{rep_result['mi_density_ratio_vs_pca']:.2f}")
                    else:
                        print(f"    No features selected from this rep")
                else:
                    print(f"    Not in hybrid selection")

                per_rep_results[rep_name] = rep_result

            results['per_rep'][_cell_key(downstream_model, budget)] = per_rep_results
            _ck.put({'cell': per_rep_results, 'rng_state': rng.get_state()},
                    **_ckey)

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
    # --models used to be parsed into config['models'] and then ignored: run()
    # asked for its own three, rf/lightgbm/dnn, so a run given
    # --models rf,svm,lightgbm scored no svm and wrote an error cell for dnn,
    # which no local search scores. Passed through only when given, so a run
    # without the flag is unchanged. (dirty-local 2026-09-28)
    _kw = ({'downstream_models': list(config['models'])}
           if getattr(args, 'models', None) else {})
    results = run(data, config, args.output, args.dataset, **_kw)
    save_experiment_results(results, args.output, 's03_feature_concentration', args.dataset)
