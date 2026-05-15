"""Unit tests for kirby.uq.conformal_chem (T-UQ.2).

Acceptance criterion: the chemistry-aware (distance-binned Mondrian) conformal
regressor produces near-nominal coverage on a scaffold-split molecular
fixture, and the distance-binning machinery itself is exercised directly.

Run: pytest tests/test_conformal_chem.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))

pytest.importorskip("rdkit")

from sklearn.ensemble import RandomForestRegressor

from kirby.uq.conformal_chem import (
    ChemBinnedConformalRegressor, _morgan_fps, _distance_to_reference,
)
from kirby.uq.conformal import regression_coverage

ALPHA = 0.1
TARGET = 1.0 - ALPHA


# --------------------------------------------------------------------------
# Molecular fixture: core ring systems x prefix substituents, scaffold-grouped
# --------------------------------------------------------------------------

_CORES = [
    'c1ccccc1', 'c1ccncc1', 'c1ccc2ccccc2c1', 'C1CCCCC1', 'C1CCNCC1',
    'C1CCOCC1', 'c1ccsc1', 'c1cc[nH]c1', 'c1ccoc1', 'C1CCCC1',
    'C1CCCCCC1', 'c1ccc2[nH]ccc2c1', 'c1ccc2ncccc2c1', 'C1CCC2CCCCC2C1',
    'c1ccc2nccnc2c1',
]
_SUBS = ['', 'C', 'CC', 'CCC', 'CCCC', 'O', 'N', 'OC', 'NC', 'Cl', 'F',
         'Br', 'CO', 'CN', 'CCO', 'C(C)C', 'CCN', 'OCC', 'NCC', 'CCCCC']


def _make_mol_fixture():
    """Build a deterministic set of valid drug-like SMILES with a label that
    has real structure, so the wrapped estimator learns something."""
    from rdkit import Chem

    rng = np.random.RandomState(0)
    smiles, y = [], []
    seen = set()
    for ci, core in enumerate(_CORES):
        for si, sub in enumerate(_SUBS):
            mol = Chem.MolFromSmiles(sub + core)
            if mol is None:
                continue
            canon = Chem.MolToSmiles(mol)
            if canon in seen:
                continue
            seen.add(canon)
            smiles.append(canon)
            # label: structured in core + substituent length, plus noise
            y.append(2.0 + 0.5 * ci - 0.3 * len(sub) + rng.randn() * 0.4)
    return smiles, np.asarray(y, dtype=float)


def _scaffold_split(smiles, seed=0):
    """Train scaffolds disjoint from {calib, test}; calib and test are an
    exchangeable random partition of the held-out scaffolds' compounds (so the
    split-conformal coverage guarantee between calib and test is valid)."""
    from find_best_hybrid_molecular import assign_scaffold_groups

    groups, _ = assign_scaffold_groups(smiles)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n_train_scaf = max(1, int(0.6 * len(uniq)))
    train_scaf = set(uniq[:n_train_scaf].tolist())

    train_mask = np.isin(groups, list(train_scaf))
    train_idx = np.where(train_mask)[0]
    held = np.where(~train_mask)[0]
    rng.shuffle(held)
    half = len(held) // 2
    return train_idx, held[:half], held[half:]


@pytest.fixture(scope="module")
def mol_data():
    from kirby.representations.molecular import create_ecfp4

    smiles, y = _make_mol_fixture()
    assert len(smiles) >= 120, f"fixture too small: {len(smiles)}"
    X = np.asarray(create_ecfp4(smiles), dtype=np.float64)
    tr, ca, te = _scaffold_split(smiles)
    assert len(tr) >= 40 and len(ca) >= 20 and len(te) >= 20, \
        f"split too small: {len(tr)}/{len(ca)}/{len(te)}"
    return {
        'smiles': smiles, 'X': X, 'y': y,
        'tr': tr, 'ca': ca, 'te': te,
    }


# --------------------------------------------------------------------------
# Fingerprint + distance helpers
# --------------------------------------------------------------------------

def test_morgan_fps_invalid_smiles():
    fps = _morgan_fps(['c1ccccc1', 'not_a_smiles', None, 'CCO'])
    assert fps[0] is not None and fps[3] is not None
    assert fps[1] is None and fps[2] is None


def test_distance_to_reference():
    ref = _morgan_fps(['c1ccccc1', 'CCO'])
    # identical compound -> distance 0
    d_same = _distance_to_reference(_morgan_fps(['c1ccccc1']), ref)
    assert d_same[0] == pytest.approx(0.0, abs=1e-9)
    # a very different compound -> distance well above 0
    d_far = _distance_to_reference(_morgan_fps(['C1CCC2CCCCC2C1']), ref)
    assert 0.0 < d_far[0] <= 1.0
    # unparseable query -> maximally out-of-domain
    d_bad = _distance_to_reference(_morgan_fps(['not_a_smiles']), ref)
    assert d_bad[0] == 1.0
    # empty reference set -> distance 1.0
    d_empty = _distance_to_reference(_morgan_fps(['CCO']), [])
    assert d_empty[0] == 1.0


# --------------------------------------------------------------------------
# Mondrian binning / calibration machinery (direct, no RDKit spread needed)
# --------------------------------------------------------------------------

def test_calibrate_builds_requested_bins():
    cbcr = ChemBinnedConformalRegressor(estimator=None, alpha=ALPHA,
                                        n_bins=5, min_per_bin=20)
    # 200 calibration points with distances spanning [0, 1]
    distances = np.linspace(0.0, 1.0, 200)
    residuals = np.abs(np.random.RandomState(1).randn(200))
    cbcr._calibrate(distances, residuals)
    assert cbcr.n_bins_eff_ == 5
    # edges strictly increasing
    assert np.all(np.diff(cbcr.bin_edges_) > 0)
    # every bin populated, counts sum to n
    assert cbcr.bin_counts_.sum() == 200
    assert np.all(cbcr.bin_counts_ > 0)
    # per-bin half-widths finite and positive
    assert np.all(np.isfinite(cbcr.bin_quantiles_))
    assert np.all(cbcr.bin_quantiles_ > 0)


def test_calibrate_reduces_bins_for_small_calib():
    cbcr = ChemBinnedConformalRegressor(estimator=None, alpha=ALPHA,
                                        n_bins=5, min_per_bin=20)
    # only 30 calibration points -> at most 30 // 20 = 1 bin
    distances = np.linspace(0.0, 1.0, 30)
    residuals = np.abs(np.random.RandomState(2).randn(30))
    cbcr._calibrate(distances, residuals)
    assert cbcr.n_bins_eff_ == 1


def test_calibrate_collapses_tied_distances():
    cbcr = ChemBinnedConformalRegressor(estimator=None, alpha=ALPHA,
                                        n_bins=5, min_per_bin=10)
    # all identical distances -> quantile edges collapse -> single bin
    distances = np.full(100, 0.3)
    residuals = np.abs(np.random.RandomState(3).randn(100))
    cbcr._calibrate(distances, residuals)
    assert cbcr.n_bins_eff_ == 1
    assert np.isfinite(cbcr.bin_quantiles_[0])


def test_conformal_halfwidth_small_bin_is_inf():
    cbcr = ChemBinnedConformalRegressor(estimator=None, alpha=ALPHA)
    # n=3, alpha=0.1 -> k = ceil(4 * 0.9) = 4 > 3 -> +inf (honest, unbounded)
    assert cbcr._conformal_halfwidth(np.array([0.1, 0.2, 0.3])) == np.inf
    # a comfortably large bin -> finite
    big = np.abs(np.random.RandomState(4).randn(100))
    assert np.isfinite(cbcr._conformal_halfwidth(big))


def test_assign_bins_clips_out_of_range():
    cbcr = ChemBinnedConformalRegressor(estimator=None, alpha=ALPHA,
                                        n_bins=3, min_per_bin=10)
    cbcr._calibrate(np.linspace(0.0, 0.6, 90),
                    np.abs(np.random.RandomState(5).randn(90)))
    # below the first edge and above the last both clip into valid bins
    idx = cbcr._assign_bins(np.array([-5.0, 0.3, 99.0]))
    assert idx.min() >= 0 and idx.max() <= cbcr.n_bins_eff_ - 1
    assert idx[0] == 0 and idx[2] == cbcr.n_bins_eff_ - 1


# --------------------------------------------------------------------------
# Validation guards
# --------------------------------------------------------------------------

def test_init_validation():
    with pytest.raises(ValueError):
        ChemBinnedConformalRegressor(RandomForestRegressor(), alpha=0.0)
    with pytest.raises(ValueError):
        ChemBinnedConformalRegressor(RandomForestRegressor(), n_bins=0)
    with pytest.raises(ValueError):
        ChemBinnedConformalRegressor(RandomForestRegressor(), min_per_bin=0)


def test_predict_before_fit_raises():
    cbcr = ChemBinnedConformalRegressor(RandomForestRegressor(), alpha=ALPHA)
    with pytest.raises(RuntimeError):
        cbcr.predict(np.zeros((2, 4)), ['CCO', 'c1ccccc1'])


# --------------------------------------------------------------------------
# End-to-end on the scaffold-split molecular fixture
# --------------------------------------------------------------------------

def test_fit_predict_coverage(mol_data):
    d = mol_data
    cbcr = ChemBinnedConformalRegressor(
        RandomForestRegressor(n_estimators=80, random_state=0),
        alpha=ALPHA, n_bins=3, min_per_bin=8)
    cbcr.fit(d['X'][d['tr']], d['y'][d['tr']], [d['smiles'][i] for i in d['tr']],
             d['X'][d['ca']], d['y'][d['ca']], [d['smiles'][i] for i in d['ca']])

    # binning machinery produced a sane, monotone partition
    assert 1 <= cbcr.n_bins_eff_ <= 3
    assert np.all(np.diff(cbcr.bin_edges_) > 0)
    assert np.all(np.isfinite(cbcr.bin_quantiles_))
    assert np.all(cbcr.bin_quantiles_ > 0)

    te_smiles = [d['smiles'][i] for i in d['te']]
    iv = cbcr.predict(d['X'][d['te']], te_smiles)
    assert iv.point.shape == iv.lower.shape == iv.upper.shape == (len(d['te']),)
    assert np.all(iv.upper >= iv.lower)

    cov = regression_coverage(d['y'][d['te']], iv.lower, iv.upper)
    # split conformal is marginally valid + slightly conservative; allow a
    # generous two-sided band for the finite fixture.
    assert cov >= TARGET - 0.12, f"under-covering: {cov:.3f}"
    assert cov <= 1.0

    sc = cbcr.score(d['X'][d['te']], d['y'][d['te']], te_smiles)
    assert sc['target_coverage'] == pytest.approx(TARGET)
    assert sc['n_bins'] == cbcr.n_bins_eff_
    assert set(sc['per_bin'].keys()) == set(range(cbcr.n_bins_eff_))
    assert sum(b['n'] for b in sc['per_bin'].values()) == len(d['te'])


def test_fit_prefit_coverage(mol_data):
    d = mol_data
    est = RandomForestRegressor(n_estimators=80, random_state=0)
    est.fit(d['X'][d['tr']], d['y'][d['tr']])  # caller fits the estimator
    cbcr = ChemBinnedConformalRegressor(est, alpha=ALPHA, n_bins=3,
                                        min_per_bin=8)
    cbcr.fit_prefit([d['smiles'][i] for i in d['tr']],
                    d['X'][d['ca']], d['y'][d['ca']],
                    [d['smiles'][i] for i in d['ca']])
    te_smiles = [d['smiles'][i] for i in d['te']]
    iv = cbcr.predict(d['X'][d['te']], te_smiles)
    cov = regression_coverage(d['y'][d['te']], iv.lower, iv.upper)
    assert cov >= TARGET - 0.12, f"under-covering: {cov:.3f}"


def test_predict_handles_invalid_smiles(mol_data):
    d = mol_data
    cbcr = ChemBinnedConformalRegressor(
        RandomForestRegressor(n_estimators=40, random_state=0),
        alpha=ALPHA, n_bins=3, min_per_bin=8)
    cbcr.fit(d['X'][d['tr']], d['y'][d['tr']], [d['smiles'][i] for i in d['tr']],
             d['X'][d['ca']], d['y'][d['ca']], [d['smiles'][i] for i in d['ca']])
    # one unparseable SMILES in the predict batch must not crash; it is
    # treated as maximally out-of-domain (last/widest bin).
    X2 = d['X'][d['te']][:3]
    smi2 = ['not_a_smiles'] + [d['smiles'][i] for i in d['te'][1:3]]
    iv = cbcr.predict(X2, smi2)
    assert iv.lower.shape == (3,) and np.all(np.isfinite(iv.lower))
