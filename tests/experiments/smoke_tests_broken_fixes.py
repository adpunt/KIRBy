"""Smoke tests for the complexity/similarity experiment fixes.

Each fix listed in COMPLEXITY_SIMILARITY_BROKEN_2026_06_03.md gets a smoke test
here. Tests load a SMALL real-data subset locally (sequence-derived `aac`
features only, so no foundation-model embeddings / server data are needed) and
assert the fixed code behaves sanely. These are fast, local sanity checks — NOT
a substitute for the full re-run on ARC.

Run:  python tests/experiments/smoke_tests_broken_fixes.py
"""
import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
from experiments._shared import load_data  # noqa: E402


def _load(dataset, rep='aac'):
    d = load_data(dataset, rep_names=[rep], rep_preset='fast')
    return d['reps'][rep], np.asarray(d['labels']), d['heavy_seqs'], d['light_seqs']


# ---------------------------------------------------------------------------
# Fix 1: MI estimator (estimate_mutual_information_knn) no longer collapses to 0
#        and detects classification targets.  Used by c01/c09/c17/s03.
# ---------------------------------------------------------------------------
def smoke_mi_estimator():
    from experiments._shared import estimate_mutual_information_knn as mi

    rng = np.random.default_rng(0)

    # (a) Small-n robustness: on real n=137 flab features the estimator must NOT
    #     collapse to 0 and its de-biased null must be ~0 (the original bug was
    #     every value == 0).
    Xsmall, _, _, _ = _load('flab_polyreact_jain_bvp')
    Xs = (Xsmall - Xsmall.mean(0)) / (Xsmall.std(0) + 1e-9)
    y_sig_small = 2.0 * Xs[:, 0] + Xs[:, 1] + rng.normal(scale=0.3, size=len(Xsmall))
    mi_small = mi(Xsmall, y_sig_small)
    mi_small_null = mi(Xsmall, rng.permutation(y_sig_small))
    assert mi_small >= mi_small_null, f'small-n MI below null ({mi_small} < {mi_small_null})'
    assert mi_small_null < 0.1, f'de-biased null not ~0 at small n ({mi_small_null:.3f})'

    # (b) Clean separation on adequate n: real humanness aac features (n=564,
    #     d<cap so no PCA dilution), known injected signal vs permuted null.
    X, yc, _, _ = _load('humanness_therapeutic')
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y_signal = 2.0 * Xs[:, 0] + Xs[:, 1] + rng.normal(scale=0.3, size=len(X))
    mi_signal = mi(X, y_signal)
    mi_null = mi(X, rng.permutation(y_signal))
    assert mi_signal > mi_null + 0.5, (
        f'MI does not separate injected signal from null '
        f'(signal={mi_signal:.3f} null={mi_null:.3f})')

    # (c) Classification branch fires on real discrete humanness labels.
    assert np.unique(yc).size <= 5, 'expected discrete humanness label'
    mi_clf = mi(X, yc)
    mi_clf_null = mi(X, rng.permutation(yc))
    assert mi_clf > mi_clf_null, (
        f'MI(classif) no separation on real labels '
        f'(real={mi_clf:.3f} null={mi_clf_null:.3f})')

    return (f'small-n no-collapse (mi={mi_small:.3f}, null={mi_small_null:.3f}); '
            f'reg signal={mi_signal:.3f}>null={mi_null:.3f}; '
            f'clf real={mi_clf:.3f}>null={mi_clf_null:.3f}')


# ---------------------------------------------------------------------------
# Fix 2: cliff detection is distribution-relative — no longer 0% (s12) or
#        ~100% (s07) on real antibody data.
# ---------------------------------------------------------------------------
def smoke_cliff_editdist():
    from experiments.similarity.similarity_helpers import (
        identify_cliff_compounds_editdist)
    # Real antibody heavy chains + labels (deep-mutational-scan style data).
    _, y, heavy, _ = _load('flab_polyreact_jain_bvp')

    # Old behaviour: absolute max_edits=5 flags an excessive fraction.
    _, info_abs = identify_cliff_compounds_editdist(heavy, y, max_edits=5)
    # New behaviour: closest 5% of pairs are "structurally similar".
    _, info_pct = identify_cliff_compounds_editdist(
        heavy, y, similar_percentile=5.0)

    pct = info_pct['pct_cliff_compounds']
    # Cliff compounds should be a non-degenerate minority (not 0%, not ~all).
    assert 0.0 < pct < 80.0, f'cliff prevalence still degenerate: {pct:.1f}%'
    assert info_pct['edit_cutoff'] >= 1, 'edit cutoff collapsed below 1'
    return (f'editdist: abs(max_edits=5) -> {info_abs["pct_cliff_compounds"]:.0f}% '
            f"cliffs; percentile(5%) -> {pct:.0f}% cliffs "
            f"(cutoff={info_pct['edit_cutoff']:.0f} edits)")


def smoke_cliff_euclidean():
    import importlib
    s12 = importlib.import_module(
        'experiments.similarity.s12_cross_domain_cliffs')
    # Real antibody aac features stand in for the standardised high-D concat.
    X, y, _, _ = _load('flab_polyreact_jain_bvp')

    # Absolute 0.55 threshold -> typically 0 cliffs on standardised features.
    idx_abs, info_abs = s12.identify_cliff_pairs_euclidean(X, y, sim_threshold=0.55)
    # Distribution-relative: top 5% most-similar pairs.
    idx_pct, info_pct = s12.identify_cliff_pairs_euclidean(
        X, y, similar_percentile=5.0)

    assert info_pct['n_cliff_pairs'] >= info_abs['n_cliff_pairs'], (
        'percentile found fewer cliff pairs than absolute cutoff')
    assert info_pct['n_cliff_pairs'] > 0, (
        f'percentile still found 0 cliff pairs (cutoff={info_pct["sim_cutoff"]:.3f})')
    return (f'euclidean: abs(0.55) -> {info_abs["n_cliff_pairs"]} pairs; '
            f'percentile(5%) -> {info_pct["n_cliff_pairs"]} pairs '
            f'(cutoff sim={info_pct["sim_cutoff"]:.3f})')


# ---------------------------------------------------------------------------
# Fix 3: c10 noiseInject dependency. The package is private (adpunt/noise_inject)
#        and missing on ARC; setup.sh now clones+installs it. This test confirms
#        every condition c10 sweeps runs on real labels, that they all deliver the
#        SAME amount of noise (the check that the dose confound is gone), and that
#        injected noise degrades model fit.
# ---------------------------------------------------------------------------
def smoke_c10_noiseinject():
    try:
        import noiseInject  # noqa: F401
    except ImportError:
        # On a machine without the package this is expected; report, don't fail.
        return 'SKIPPED \u2014 noiseInject not importable here (install via setup.sh on ARC)'

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    from kirby.noise_spec import (
        DOSE_MATCHED_STRATEGIES, GROUPED_STRATEGIES, NoiseSpec,
        make_noisy_labels_verbose,
    )
    X, y, _, _ = _load('flab_thermo_jain')  # regression

    # Stand-in groups for the two grouped conditions. flab_thermo_jain is an
    # antibody set with no SMILES, so this smoke test cannot use real scaffolds;
    # it is checking the wiring, not the chemistry.
    groups = np.arange(len(y)) // 3

    # Every condition must run, perturb the labels, and deliver the dose it was
    # asked for. Before noiseInject 1.0.0 the conditions delivered between 0.49x
    # and 2.00x the same amount at one nominal setting, and their entire apparent
    # severity ordering was explained by that.
    target = 0.3 * float(np.std(y))
    delivered = {}
    for cond in DOSE_MATCHED_STRATEGIES:
        g = groups if cond in GROUPED_STRATEGIES else None
        res = make_noisy_labels_verbose(y.copy(), NoiseSpec(cond, 0.3, noise_seed=42),
                                        groups=g)
        assert res.y_noisy.shape == y.shape, f'{cond}: shape changed'
        assert not np.allclose(res.y_noisy, y), f'{cond}: labels unchanged'
        assert abs(res.unit_dose_g * res.solved_scale - target) < 1e-5 * target, \
            f'{cond}: solved scale does not deliver the target dose'
        delivered[cond] = res.realised_dose_label_units
    spread = max(delivered.values()) / min(delivered.values()) - 1.0
    assert spread < 0.25, f'dose is not flat across conditions: {spread:.1%}'

    # Noise degradation: fit RF on clean labels vs level 0.5; score should drop.
    tr, va = next(KFold(3, shuffle=True, random_state=0).split(X))
    def _r2(ytr):
        m = RandomForestRegressor(n_estimators=80, random_state=0, n_jobs=-1)
        m.fit(X[tr], ytr)
        from scipy.stats import spearmanr
        return spearmanr(y[va], m.predict(X[va])).correlation
    clean = _r2(y[tr])
    noisy = _r2(make_noisy_labels_verbose(
        y[tr].copy(), NoiseSpec('gaussian', 0.5, noise_seed=1)).y_noisy)
    assert noisy <= clean + 0.05, f'noise did not degrade fit (clean={clean:.3f} noisy={noisy:.3f})'
    return (f'{len(delivered)} conditions inject OK, dose flat to {spread:.1%}; '
            f'gaussian degrades \u03c1 {clean:.3f}->{noisy:.3f}')


# ---------------------------------------------------------------------------
# Fix 4 (s06): classification applicability-domain error is now graded
#        (1 - P(true class)) instead of a degenerate 0/1 misclassification
#        indicator, so Spearman(distance, error) has real power.
# ---------------------------------------------------------------------------
def smoke_s06_graded_clf_error():
    from experiments.similarity.s06_applicability_domain import _graded_clf_error
    from sklearn.ensemble import RandomForestClassifier
    X, y, _, _ = _load('humanness_therapeutic')  # binary clf labels
    assert np.unique(y).size >= 2, 'expected >=2 classes'
    split = int(0.7 * len(X))
    rf = RandomForestClassifier(n_estimators=60, random_state=0).fit(X[:split], y[:split])
    err = _graded_clf_error(rf, X[split:], y[split:])
    yv = y[split:]
    # Continuous in [0,1], NOT collapsed to two values like the old 0/1 indicator.
    assert err.min() >= 0.0 and err.max() <= 1.0, 'graded error out of [0,1]'
    n_distinct = len(np.unique(np.round(err, 3)))
    assert n_distinct > 2, f'graded error collapsed to {n_distinct} values (still 0/1-like)'
    # Misclassified points must carry higher error than correctly-classified.
    mis = rf.predict(X[split:]) != yv
    if mis.any() and (~mis).any():
        assert err[mis].mean() > err[~mis].mean(), 'graded error not higher on misclassified'
    return (f'graded clf error continuous ({n_distinct} distinct values in [0,1]); '
            f'mis={err[mis].mean():.3f} > correct={err[~mis].mean():.3f}')


# ---------------------------------------------------------------------------
# Fix 5 (s10): every stored delta is direction-oriented so positive == improvement
#        regardless of metric direction. The buchwald RMSE "+13.077" was a
#        worsening misread as a gain.
# ---------------------------------------------------------------------------
def smoke_s10_delta_orientation():
    from experiments.similarity.s10_importance_downstream import _orient_deltas
    # lower-is-better (RMSE): a positive raw delta means WORSE -> must become negative.
    rmse = {
        'weighted_gp': {
            'hybrid': {'standard': 14.21, 'weighted': 27.29, 'delta': 13.08},
            'per_rep': {'r1': {'delta': -2.0}},  # RMSE dropped -> improvement
        },
        'qrasar': {'hybrid': {'delta_uw': 1.0, 'delta_w': 2.0, 'delta_w_vs_uw': 0.5}},
    }
    for k in ('weighted_gp', 'qrasar'):
        _orient_deltas(rmse[k], True)
    assert rmse['weighted_gp']['hybrid']['delta'] == -13.08, 'RMSE +13.08 not flipped'
    assert rmse['weighted_gp']['per_rep']['r1']['delta'] == 2.0, 'RMSE -2.0 not flipped'
    assert rmse['qrasar']['hybrid']['delta_w'] == -2.0, 'qrasar delta_w not flipped'
    # Non-delta score fields must be untouched.
    assert rmse['weighted_gp']['hybrid']['weighted'] == 27.29, 'score field mutated'
    # higher-is-better (Spearman): orientation is a no-op.
    hib = {'t': {'hybrid': {'delta': 0.05}}}
    _orient_deltas(hib['t'], False)
    assert hib['t']['hybrid']['delta'] == 0.05, 'higher-better delta wrongly flipped'
    return 'RMSE delta +13.08 -> -13.08; higher-better no-op; score fields untouched'


SMOKE_TESTS = [
    ('Fix1 MI estimator (c01/c09/c17/s03)', smoke_mi_estimator),
    ('Fix2a cliff edit-distance (s07)', smoke_cliff_editdist),
    ('Fix2b cliff euclidean (s12)', smoke_cliff_euclidean),
    ('Fix3 c10 noiseInject dependency', smoke_c10_noiseinject),
    ('Fix4 s06 graded clf AD error', smoke_s06_graded_clf_error),
    ('Fix5 s10 delta orientation', smoke_s10_delta_orientation),
]


def main():
    print('=' * 70)
    print('SMOKE TESTS — complexity/similarity broken-fix verification')
    print('=' * 70)
    failed = 0
    for name, fn in SMOKE_TESTS:
        try:
            msg = fn()
            print(f'[PASS] {name}\n       {msg}')
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f'[FAIL] {name}\n       {type(e).__name__}: {e}')
            traceback.print_exc()
    print('-' * 70)
    print(f'{len(SMOKE_TESTS) - failed}/{len(SMOKE_TESTS)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
