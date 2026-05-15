"""Chemistry-aware conformal prediction (T-UQ.2).

`conformal_chem.py` — a scaffold-/chemistry-aware conformal variant for the
small vet ChEMBL regression tasks, where the test scaffolds are disjoint from
training (the standing no-random-splits rule).

Plain split conformal gives a single marginal interval width: it under-covers
compounds far from the training set and over-covers compounds near it.
`ChemBinnedConformalRegressor` bins compounds by their distance to the nearest
training compound (1 - max Tanimoto over Morgan fingerprints) and calibrates a
separate conformal quantile per bin — Mondrian conformal prediction with a
chemistry-derived partition. The result: intervals that widen out-of-domain,
with coverage held per distance bin rather than only on average.

Why distance-to-nearest-training-compound and not the scaffold-group ID:
under a scaffold split the test scaffolds are NOVEL — they appear in no
calibration bin — so a partition keyed on scaffold ID cannot be applied at
predict time. Distance to the training set IS computable for any molecule,
known or novel, so it is a partition that generalises. (User decision,
2026-05-15.)

This builds on `kirby.uq.conformal`: it reuses the `RegressionInterval` tuple
and the plain-numpy coverage/width metrics. Like that module, MAPIE is not
required here — the Mondrian split-conformal quantile is computed directly,
which keeps the interval construction fully transparent for a regulatory
write-up. RDKit IS required (Morgan fingerprints + Tanimoto).

Calibration modes mirror `kirby.uq.conformal`:
  - `fit(X_train, y_train, smiles_train, X_calib, y_calib, smiles_calib)` —
    fits the estimator on the train split, then bins + calibrates on the
    explicit (scaffold-disjoint) calibration split.
  - `fit_prefit(smiles_train, X_calib, y_calib, smiles_calib)` — the estimator
    is already fitted; `smiles_train` is the applicability-domain reference
    set the distance bins are measured against.

Scope: regression only — the vet eval targets are all regression. A
classification variant is deferred until a cross-species clf task needs it.

Reference:
    Vovk et al. — conformal prediction; Mondrian / label-conditional conformal.
    Norinder et al. — conformal prediction with applicability-domain binning
    for QSAR.
"""

import numpy as np

from kirby.uq.conformal import (
    RegressionInterval, regression_coverage, mean_interval_width,
)


# --------------------------------------------------------------------------
# Morgan fingerprints + distance-to-training-set
# --------------------------------------------------------------------------

def _morgan_fps(smiles, radius=2, n_bits=1024):
    """Morgan bit-vector fingerprints for a list of SMILES.

    Invalid SMILES yield None — the caller treats those as maximally
    out-of-domain (distance 1.0) rather than dropping rows, so the returned
    list always lines up 1:1 with the input.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if smi is not None else None
        if mol is None:
            fps.append(None)
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=n_bits))
    return fps


def _distance_to_reference(query_fps, reference_fps):
    """Per-query distance to the nearest reference compound.

    distance = 1 - max Tanimoto similarity to any reference fingerprint, in
    [0, 1]. A query with an unparseable SMILES (fp is None), or an empty
    reference set, gets distance 1.0 (maximally out-of-domain).
    """
    from rdkit import DataStructs

    ref = [f for f in reference_fps if f is not None]
    out = np.ones(len(query_fps), dtype=np.float64)
    if not ref:
        return out
    for i, qfp in enumerate(query_fps):
        if qfp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(qfp, ref)
        out[i] = 1.0 - (max(sims) if sims else 0.0)
    return out


# --------------------------------------------------------------------------
# Mondrian split-conformal regressor, binned by distance to the training set
# --------------------------------------------------------------------------

class ChemBinnedConformalRegressor:
    """Distance-binned (Mondrian) split-conformal prediction intervals.

    Args:
        estimator: an sklearn-compatible regressor. Unfitted for `fit`;
            already fitted for `fit_prefit`.
        alpha: miscoverage rate. alpha=0.1 -> 90% target coverage.
        n_bins: maximum number of distance bins. The effective count is
            reduced so every bin holds at least `min_per_bin` calibration
            compounds (and further if quantile bin edges collapse on ties);
            with too few calibration compounds it falls back to 1 bin, i.e.
            ordinary split conformal.
        min_per_bin: minimum calibration compounds per bin.
        fp_radius, fp_bits: Morgan fingerprint settings for the Tanimoto
            distance.

    After a fit, exposes:
        estimator_     the fitted estimator
        bin_edges_     (n_bins_eff_ + 1,) distance bin edges
        bin_quantiles_ (n_bins_eff_,) the conformal half-width per bin
        bin_counts_    (n_bins_eff_,) calibration compounds per bin
        n_bins_eff_    effective bin count
    """

    def __init__(self, estimator, alpha=0.1, n_bins=5, min_per_bin=20,
                 fp_radius=2, fp_bits=1024):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        if min_per_bin < 1:
            raise ValueError(f"min_per_bin must be >= 1, got {min_per_bin}")
        self.estimator = estimator
        self.alpha = alpha
        self.n_bins = n_bins
        self.min_per_bin = min_per_bin
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
        self.estimator_ = None
        self.train_fps_ = None
        self.bin_edges_ = None
        self.bin_quantiles_ = None
        self.bin_counts_ = None
        self.n_bins_eff_ = None

    @property
    def confidence_level(self):
        return 1.0 - self.alpha

    # -- internals ---------------------------------------------------------

    def _fp(self, smiles):
        return _morgan_fps(smiles, self.fp_radius, self.fp_bits)

    def _conformal_halfwidth(self, scores):
        """Finite-sample split-conformal quantile of |residual| scores.

        level = ceil((n+1)(1-alpha)) / n; if that exceeds 1 the bin is too
        small to guarantee coverage and the half-width is +inf (an honest,
        unbounded interval) rather than a silent under-covering value.
        """
        scores = np.asarray(scores, dtype=np.float64)
        n = len(scores)
        if n == 0:
            return np.inf
        k = np.ceil((n + 1) * self.confidence_level)
        if k > n:
            return np.inf
        return float(np.quantile(scores, k / n, method='higher'))

    def _bin_edges(self, calib_distances):
        """Quantile bin edges over the calibration distances, with the bin
        count reduced to honour `min_per_bin` and to drop edges that collapse
        on ties (e.g. many identical distances)."""
        n = len(calib_distances)
        n_bins_eff = max(1, min(self.n_bins, n // self.min_per_bin))
        edges = np.quantile(calib_distances,
                            np.linspace(0.0, 1.0, n_bins_eff + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            # all calibration distances identical -> a single global bin.
            edges = np.array([calib_distances.min(), calib_distances.max()],
                             dtype=np.float64)
        return edges

    def _assign_bins(self, distances):
        """Map distances to bin indices 0..n_bins_eff_-1. Values below the
        first edge land in bin 0, values above the last in the final bin."""
        # interior edges only; digitize then clip into valid range.
        interior = self.bin_edges_[1:-1]
        idx = np.digitize(distances, interior, right=False)
        return np.clip(idx, 0, self.n_bins_eff_ - 1)

    def _calibrate(self, calib_distances, calib_residuals):
        """Shared by fit / fit_prefit: build bins + per-bin half-widths."""
        calib_distances = np.asarray(calib_distances, dtype=np.float64)
        calib_residuals = np.asarray(calib_residuals, dtype=np.float64)
        self.bin_edges_ = self._bin_edges(calib_distances)
        self.n_bins_eff_ = len(self.bin_edges_) - 1

        bin_idx = self._assign_bins(calib_distances)
        quantiles = np.empty(self.n_bins_eff_, dtype=np.float64)
        counts = np.zeros(self.n_bins_eff_, dtype=np.int64)
        # global fallback for any bin left underpopulated by edge collapse
        global_hw = self._conformal_halfwidth(calib_residuals)
        for b in range(self.n_bins_eff_):
            mask = bin_idx == b
            counts[b] = int(mask.sum())
            if counts[b] == 0:
                quantiles[b] = global_hw
            else:
                quantiles[b] = self._conformal_halfwidth(calib_residuals[mask])
        self.bin_quantiles_ = quantiles
        self.bin_counts_ = counts

    # -- public API --------------------------------------------------------

    def fit(self, X_train, y_train, smiles_train,
            X_calib, y_calib, smiles_calib):
        """Fit the estimator on the train split, then bin + calibrate on the
        explicit (scaffold-disjoint) calibration split."""
        from sklearn.base import clone

        if not (len(X_train) == len(y_train) == len(smiles_train)):
            raise ValueError("X_train, y_train, smiles_train length mismatch")
        if not (len(X_calib) == len(y_calib) == len(smiles_calib)):
            raise ValueError("X_calib, y_calib, smiles_calib length mismatch")

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(np.asarray(X_train),
                            np.asarray(y_train).ravel())
        self.train_fps_ = self._fp(smiles_train)

        calib_pred = self.estimator_.predict(np.asarray(X_calib))
        residuals = np.abs(np.asarray(y_calib).ravel() - calib_pred)
        distances = _distance_to_reference(self._fp(smiles_calib),
                                           self.train_fps_)
        self._calibrate(distances, residuals)
        return self

    def fit_prefit(self, smiles_train, X_calib, y_calib, smiles_calib):
        """Calibrate an ALREADY-FITTED estimator. `smiles_train` is the
        applicability-domain reference set the distance bins are measured
        against (typically the SMILES the estimator was trained on)."""
        if not (len(X_calib) == len(y_calib) == len(smiles_calib)):
            raise ValueError("X_calib, y_calib, smiles_calib length mismatch")

        self.estimator_ = self.estimator
        self.train_fps_ = self._fp(smiles_train)

        calib_pred = self.estimator_.predict(np.asarray(X_calib))
        residuals = np.abs(np.asarray(y_calib).ravel() - calib_pred)
        distances = _distance_to_reference(self._fp(smiles_calib),
                                           self.train_fps_)
        self._calibrate(distances, residuals)
        return self

    def predict(self, X, smiles):
        """Return a RegressionInterval(point, lower, upper).

        Each compound's half-width is the conformal quantile of the distance
        bin it falls into (by distance to the nearest training compound)."""
        if self.estimator_ is None:
            raise RuntimeError(
                "ChemBinnedConformalRegressor.predict called before a fit")
        if len(X) != len(smiles):
            raise ValueError("X and smiles length mismatch")
        point = np.asarray(self.estimator_.predict(np.asarray(X))).ravel()
        distances = _distance_to_reference(self._fp(smiles), self.train_fps_)
        bin_idx = self._assign_bins(distances)
        half = self.bin_quantiles_[bin_idx]
        return RegressionInterval(point=point,
                                  lower=point - half, upper=point + half)

    def score(self, X, y_true, smiles):
        """Empirical coverage + mean interval width on (X, y_true, smiles),
        plus a per-bin breakdown (coverage, width, n) keyed by bin index."""
        iv = self.predict(X, smiles)
        y_true = np.asarray(y_true, dtype=float).ravel()
        distances = _distance_to_reference(self._fp(smiles), self.train_fps_)
        bin_idx = self._assign_bins(distances)

        per_bin = {}
        for b in range(self.n_bins_eff_):
            mask = bin_idx == b
            if not mask.any():
                per_bin[b] = {'n': 0, 'coverage': float('nan'),
                              'mean_interval_width': float('nan')}
                continue
            per_bin[b] = {
                'n': int(mask.sum()),
                'coverage': regression_coverage(
                    y_true[mask], iv.lower[mask], iv.upper[mask]),
                'mean_interval_width': mean_interval_width(
                    iv.lower[mask], iv.upper[mask]),
            }
        return {
            'coverage': regression_coverage(y_true, iv.lower, iv.upper),
            'mean_interval_width': mean_interval_width(iv.lower, iv.upper),
            'target_coverage': self.confidence_level,
            'n_bins': self.n_bins_eff_,
            'per_bin': per_bin,
        }
