#!/usr/bin/env python3
"""S11: Bayesian Optimization in Compact Hybrid Space.

Paper: Similarity
Conclusion: GP-BO in compact KIRBy hybrid space is more sample-efficient than
    in high-dimensional fingerprint space, because distance concentration makes
    the GP surrogate uninformative when dimensionality is high.

Core hypothesis: In high-D, pairwise distances concentrate (Beyer et al. 1999),
    so the RBF kernel values all converge to the same number. The GP posterior
    becomes flat (near-zero variance everywhere), Expected Improvement collapses,
    and BO degenerates to random search. Compact KIRBy hybrids avoid this.

Design: Simulate a library-screening BO loop.
    1. Hold out 80% as "unlabeled pool"
    2. Initialize with random 5% seed set
    3. Build representation (hybrid/PCA/raw) on seed set
    4. BO loop: fit GP -> compute EI -> select top-1 -> reveal label -> repeat
    5. Track best-found-so-far at each iteration

Conditions:
    1. ecfp4_raw       — ECFP4 (2048D) + RBF GP: shows RBF failure in high-D
    2. tanimoto_ecfp4  — ECFP4 (2048D) + Tanimoto GP: proper kernel for binary FPs
    3. best_single     — Best individual rep by seed-set GP + RBF GP
    4. pca_50d         — PCA to 50D from concat + RBF GP: unsupervised compression
    5. hybrid_50d      — KIRBy hybrid (budget=50) + RBF GP: MAIN CONDITION
    6. random_50d      — Random Gaussian projection to 50D + RBF GP: null compression
    7. random_acq      — Random acquisition baseline (no GP)
    8. multi_obj_hybrid — (optional, --multi-objective) scalarized multi-objective
       BO in hybrid space: geometric mean of normalized label + QED drug-likeness.
       Per Moesser (2023) Ch. 2. Molecular datasets with SMILES only.
    8. multi_obj_random — random baseline on combined objective (for comparison)

Metrics:
    - Primary: AUC of best-found-so-far curve (area under cumulative max)
    - Secondary: iterations to find top-1% compound, iterations to find top-5%

References:
    - Beyer et al. (ICDT 1999) "When Is Nearest Neighbor Meaningful?"
    - Snoek et al. (NIPS 2012) "Practical Bayesian Optimization"
    - Griffiths & Hernandez-Lobato (2020) "Constrained BO in Molecular Space"
    - Moesser (2023) PhD thesis, Ch. 2 (Oxford) — multi-objective BO

Usage:
    python s11_bayesian_optimization.py --dataset logd --reps all --mode intermediary
    python s11_bayesian_optimization.py --dataset logd --reps all --max-iterations 200 --n-seeds 5
    python s11_bayesian_optimization.py --dataset caco2 --reps all --rebuild-interval 50
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, KFold, cross_val_score

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy, get_domain_metric,
    make_structured_output, get_precomputed_search, get_best_search_for_downstream,
)

from sklearn.gaussian_process.kernels import Kernel, NormalizedKernelMixin
from kirby.hybrid import apply_feature_selection, apply_augmentation_selection
from kirby.hybrid import create_hybrid


class TanimotoKernel(NormalizedKernelMixin, Kernel):
    """Tanimoto (Jaccard) kernel for binary/count fingerprints.

    k(x, y) = <x, y> / (||x||² + ||y||² - <x, y>)

    Appropriate for molecular fingerprints where RBF assumes smooth
    continuous features. Bounded in [0, 1]. Parameter-free: it has NO
    learnable hyperparameters, so we deliberately do NOT define any
    ``hyperparameter_*`` attribute and do NOT override ``theta`` / ``bounds``
    / ``hyperparameters``. The base ``Kernel`` machinery then correctly yields
    an empty hyperparameter list (``hyperparameters == []``,
    ``theta == array([])``, ``bounds == array([]).``), which is what
    GaussianProcessRegressor.fit expects. (The previous override of
    ``hyperparameter_names`` was itself picked up by
    ``Kernel.hyperparameters`` — which collects every ``hyperparameter_*``
    attribute — and crashed fit with "'list' object has no attribute 'name'".)

    ``NormalizedKernelMixin`` supplies ``diag(X) -> ones`` (correct, since
    k(x, x) = 1 for any non-zero fingerprint). Tanimoto is NOT stationary, so
    ``is_stationary`` returns False (we do not use StationaryKernelMixin).
    A tunable variance/scale should be added by wrapping in
    ``ConstantKernel(...) * TanimotoKernel()`` (as fit_gp does), not by giving
    this kernel its own hyperparameter.
    """

    def __init__(self):
        # Explicit parameter-free __init__ is REQUIRED: sklearn's
        # Kernel.get_params introspects __init__'s signature and rejects any
        # *args/**kwargs (the inherited object.__init__ has them). A bare
        # `def __init__(self)` makes get_params return {} cleanly, which
        # clone()/clone_with_theta() rely on.
        pass

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X, dtype=np.float64)
        if Y is None:
            Y = X
        else:
            Y = np.asarray(Y, dtype=np.float64)
        XY = X @ Y.T
        X_sq = np.sum(X ** 2, axis=1)
        Y_sq = np.sum(Y ** 2, axis=1)
        denom = X_sq[:, None] + Y_sq[None, :] - XY
        K = np.where(denom > 0, XY / denom, 0.0)
        if eval_gradient:
            # Parameter-free kernel: zero-width gradient w.r.t. theta.
            return K, np.empty((X.shape[0], Y.shape[0], 0))
        return K

    def is_stationary(self):
        return False


# =============================================================================
# EXPECTED IMPROVEMENT
# =============================================================================

def expected_improvement(mu, sigma, y_best, xi=0.01):
    """Compute Expected Improvement.

    EI(x) = (mu(x) - y_best - xi) * Phi(z) + sigma(x) * phi(z)
    where z = (mu(x) - y_best - xi) / sigma(x)

    Args:
        mu: predicted means, shape (n,)
        sigma: predicted stds, shape (n,)
        y_best: best observed value so far (scalar)
        xi: exploration-exploitation tradeoff (small positive value)

    Returns:
        ei: expected improvement values, shape (n,)
    """
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    # Set EI to 0 where sigma is negligible (GP is certain)
    ei[sigma < 1e-8] = 0.0
    return ei


# =============================================================================
# SURROGATE MODEL FITTING (GP, NGBoost, BNN)
# =============================================================================

def fit_gp(X_train, y_train, optimize=True, kernel_type='rbf'):
    """Fit a GP with specified kernel. Returns (model, success).

    In high-D, the kernel matrix may become singular. We catch LinAlgError
    and return (None, False) — this IS the expected result showing GP failure.

    Args:
        kernel_type: 'rbf' (standard) or 'tanimoto' (for binary fingerprints).
    """
    if kernel_type == 'tanimoto':
        kernel = ConstantKernel(1.0) * TanimotoKernel() + WhiteKernel(noise_level=0.1)
    else:
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=2 if optimize else 0,
        alpha=1e-6,
        normalize_y=True,
        random_state=42,
    )
    try:
        gp.fit(X_train, y_train)
        return gp, True
    except np.linalg.LinAlgError:
        return None, False
    except ValueError:
        return None, False


def fit_ngboost(X_train, y_train, **kwargs):
    """Fit NGBoost for probabilistic prediction. Returns (model, success).

    NGBoost gives mu + sigma via natural gradient boosting over a Normal base.
    """
    try:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
        model = NGBRegressor(Dist=Normal, n_estimators=100, learning_rate=0.05,
                             random_state=42, verbose=False)
        model.fit(X_train, y_train)
        return model, True
    except ImportError:
        print("      WARNING: ngboost not installed, skipping")
        return None, False
    except Exception:
        return None, False


def fit_bnn(X_train, y_train, **kwargs):
    """Fit a Bayesian Neural Network for probabilistic prediction. Returns (model, success).

    Uses MC-Dropout BNN (sklearn-compatible wrapper). Predictions via
    multiple forward passes with dropout enabled give mean + std.
    """
    try:
        from kirby.models.bnn import BNNRegressor
        model = BNNRegressor(hidden_sizes=(64, 32), dropout=0.1, epochs=100,
                             mc_samples=20, random_state=42)
        model.fit(X_train, y_train)
        return model, True
    except ImportError:
        print("      WARNING: BNN model not available, skipping")
        return None, False
    except Exception:
        return None, False


def fit_gauche_gp(X_train, y_train, **kwargs):
    """Fit a Gauche GP for molecular BO. Returns (model, success).

    Gauche provides molecular-aware GP kernels (Tanimoto on fingerprints,
    graph kernels, etc.) via GPyTorch.
    """
    try:
        from gauche.dataloader import MolPropLoader
        from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel as GaucheTanimoto
        import gpytorch
        import torch

        # Lightweight Gauche GP wrapper with predict(X, return_std=True) interface
        class _GaucheGPModel:
            def __init__(self):
                self.model = None
                self.likelihood = None
                self.train_x = None

            def fit(self, X, y):
                self.train_x = torch.tensor(X, dtype=torch.float64)
                train_y = torch.tensor(y, dtype=torch.float64)
                self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

                class _ExactGP(gpytorch.models.ExactGP):
                    def __init__(self_, lh, tx, ty):
                        super().__init__(tx, ty, lh)
                        self_.mean_module = gpytorch.means.ConstantMean()
                        self_.covar_module = gpytorch.kernels.ScaleKernel(GaucheTanimoto())
                    def forward(self_, x):
                        return gpytorch.distributions.MultivariateNormal(
                            self_.mean_module(x), self_.covar_module(x))

                self.model = _ExactGP(self.likelihood, self.train_x, train_y)
                self.model.double()
                self.likelihood.double()
                self.model.train()
                self.likelihood.train()
                optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
                for _ in range(50):
                    optimizer.zero_grad()
                    output = self.model(self.train_x)
                    loss = -mll(output, train_y)
                    loss.backward()
                    optimizer.step()

            def predict(self, X, return_std=False):
                self.model.eval()
                self.likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    test_x = torch.tensor(X, dtype=torch.float64)
                    pred = self.likelihood(self.model(test_x))
                mu = pred.mean.numpy()
                if return_std:
                    return mu, pred.stddev.numpy()
                return mu

        model = _GaucheGPModel()
        model.fit(X_train, y_train)
        return model, True
    except ImportError:
        print("      WARNING: gauche not installed, skipping")
        return None, False
    except Exception as e:
        print(f"      Gauche GP failed: {e}")
        return None, False


# Map surrogate names to (fit_fn, supports_kernel_type)
SURROGATE_MODELS = {
    'gp': (fit_gp, True),         # sklearn GP, supports kernel_type arg
    'gauche_gp': (fit_gauche_gp, False),  # Gauche GP with Tanimoto
    'ngboost': (fit_ngboost, False),
    'bnn': (fit_bnn, False),
}

# Optional surrogate packages. If the import is missing, the corresponding
# hybrid_<surrogate> condition would have EVERY fit fail and silently degrade to
# pure random acquisition while still being reported under the surrogate's name
# (SIMA-11). We probe availability up front and skip+flag instead.
_OPTIONAL_SURROGATE_IMPORTS = {
    'ngboost': 'ngboost',
    'bnn': 'kirby.models.bnn',
    'gauche_gp': 'gauche',
}


def _surrogate_available(surrogate):
    """Return True if the optional surrogate's package can be imported.

    'gp' (sklearn) is always available. Anything in
    _OPTIONAL_SURROGATE_IMPORTS is checked via importlib so a missing package
    becomes an explicit skip rather than a random-acquisition curve mislabelled
    as a real method.
    """
    mod = _OPTIONAL_SURROGATE_IMPORTS.get(surrogate)
    if mod is None:
        return True  # 'gp' or unknown -> assume present (gp is sklearn)
    import importlib.util
    return importlib.util.find_spec(mod) is not None


def predict_with_uncertainty(model, X):
    """Unified predict interface: returns (mu, sigma).

    Works with sklearn GP, NGBoost, BNN, and Gauche GP — all support
    predict(X, return_std=True) or equivalent.
    """
    # NGBoost uses .pred_dist()
    if hasattr(model, 'pred_dist'):
        dist = model.pred_dist(X)
        return dist.loc, dist.scale
    # Everything else (GP, BNN) uses predict(X, return_std=True)
    return model.predict(X, return_std=True)


# =============================================================================
# MULTI-OBJECTIVE HELPERS
# =============================================================================

def compute_qed_scores(smiles_list):
    """Compute QED drug-likeness scores from SMILES.

    Returns np.array of QED values in [0, 1].
    """
    from rdkit import Chem
    from rdkit.Chem import QED as QEDModule

    scores = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None:
            scores.append(QEDModule.qed(mol))
        else:
            scores.append(0.5)  # neutral default
    return np.array(scores, dtype=np.float64)


def compute_combined_objective(y, qed_scores, lower_is_better=False):
    """Geometric mean of normalized label and QED for multi-objective BO.

    Mirrors Moesser (2023) Ch. 2: geometric mean of pIC50 across targets,
    adapted here as geometric mean of (normalized property, QED).

    Both components normalized to [0.01, 1] before combining.
    Always returns values to be MAXIMIZED.
    """
    y_work = -y if lower_is_better else y.copy()
    y_min, y_max = y_work.min(), y_work.max()
    y_range = y_max - y_min
    if y_range < 1e-10:
        y_norm = np.full_like(y_work, 0.5)
    else:
        y_norm = (y_work - y_min) / y_range

    y_norm = np.clip(y_norm, 0.01, 1.0)
    qed_clamped = np.clip(qed_scores, 0.01, 1.0)

    return np.sqrt(y_norm * qed_clamped)


# =============================================================================
# SINGLE BO RUN
# =============================================================================

def run_bo_loop(X_labeled, y_labeled, X_pool, y_pool,
                max_iterations, refit_interval=10, rng=None,
                kernel_type='rbf', surrogate='gp'):
    """Run a single BO loop in a given representation space.

    Args:
        X_labeled: initial labeled features, shape (n_seed, d)
        y_labeled: initial labeled targets, shape (n_seed,)
        X_pool: pool features, shape (n_pool, d)
        y_pool: pool targets (hidden, revealed one at a time)
        max_iterations: BO budget
        refit_interval: refit surrogate every N iterations
        kernel_type: 'rbf' or 'tanimoto' (for GP kernels only)
        surrogate: surrogate model name ('gp', 'gauche_gp', 'ngboost', 'bnn')

    Returns:
        best_so_far: list of best-found-so-far at each iteration
        surrogate_failures: number of surrogate fitting failures
    """
    if rng is None:
        rng = np.random.RandomState(42)

    fit_fn, supports_kt = SURROGATE_MODELS.get(surrogate, (fit_gp, True))

    X_lab = X_labeled.copy()
    y_lab = y_labeled.copy()
    pool_mask = np.ones(len(y_pool), dtype=bool)

    best_so_far = [float(np.max(y_lab))]
    surrogate_failures = 0
    last_optimize = -refit_interval

    for it in range(max_iterations):
        if pool_mask.sum() == 0:
            break

        optimize = (it - last_optimize) >= refit_interval
        if optimize:
            last_optimize = it

        # Fit surrogate on labeled set
        fit_kwargs = {}
        if supports_kt:
            fit_kwargs['optimize'] = optimize
            fit_kwargs['kernel_type'] = kernel_type
        model, success = fit_fn(X_lab, y_lab, **fit_kwargs)

        if not success:
            surrogate_failures += 1
            available = np.where(pool_mask)[0]
            selected_pool_idx = available[rng.randint(len(available))]
        else:
            X_pool_available = X_pool[pool_mask]
            mu, sigma = predict_with_uncertainty(model, X_pool_available)

            y_best = float(np.max(y_lab))
            ei = expected_improvement(mu, sigma, y_best)

            best_ei_idx = np.argmax(ei)
            available = np.where(pool_mask)[0]
            selected_pool_idx = available[best_ei_idx]

        x_new = X_pool[selected_pool_idx:selected_pool_idx + 1]
        y_new = y_pool[selected_pool_idx]

        X_lab = np.vstack([X_lab, x_new])
        y_lab = np.append(y_lab, y_new)
        pool_mask[selected_pool_idx] = False

        best_so_far.append(float(max(best_so_far[-1], y_new)))

    return best_so_far, surrogate_failures


def run_random_loop(y_labeled, y_pool, max_iterations, rng):
    """Run random acquisition baseline (no GP).

    Args:
        y_labeled: initial labeled targets
        y_pool: pool targets (hidden, revealed one at a time)
        max_iterations: budget
        rng: numpy RandomState

    Returns:
        best_so_far: list of best-found-so-far at each iteration
    """
    y_lab = y_labeled.copy()
    pool_mask = np.ones(len(y_pool), dtype=bool)

    best_so_far = [float(np.max(y_lab))]

    for it in range(max_iterations):
        if pool_mask.sum() == 0:
            break

        available = np.where(pool_mask)[0]
        selected = rng.choice(available)

        y_new = y_pool[selected]
        y_lab = np.append(y_lab, y_new)
        pool_mask[selected] = False

        best_so_far.append(float(max(best_so_far[-1], y_new)))

    return best_so_far


# =============================================================================
# REPRESENTATION BUILDERS
# =============================================================================

def build_ecfp4_features(reps, seed_idx, pool_idx):
    """Use ECFP4 raw (high-D) features.

    Falls back to the highest-dimensional rep if ecfp4 is not available.
    """
    if 'ecfp4' in reps:
        X_all = reps['ecfp4']
    else:
        # Use highest-dimensional individual rep as surrogate
        name = max(reps.keys(), key=lambda k: reps[k].shape[1])
        X_all = reps[name]
        print(f"    (ecfp4 not found, using {name} with dim={X_all.shape[1]})")

    scaler = StandardScaler()
    X_seed = scaler.fit_transform(X_all[seed_idx])
    X_pool = scaler.transform(X_all[pool_idx])
    return X_seed, X_pool, X_all.shape[1]


def build_best_single_features(reps, labels, seed_idx, pool_idx):
    """Use the best individual rep (highest CV R2 on seed set).

    Uses k-fold CV on the seed set instead of training-set R² so the
    selection is not pure overfit noise. Requires every fold to have at
    least 2 test samples — single-sample folds make R² undefined (NaN).
    When the seed is too small for any honest CV (n_seed < 4), refuses to
    pick a "best" and falls back deterministically to the highest-dim rep.
    """
    y_seed = labels[seed_idx]
    n_seed = len(y_seed)

    # Need at least 4 samples so that a 2-fold split has >=2 test samples.
    if n_seed < 4:
        fallback_name = max(reps.keys(), key=lambda k: reps[k].shape[1])
        print(f"    WARNING: seed too small (n_seed={n_seed} < 4) for honest "
              f"CV selection; falling back to highest-dim rep '{fallback_name}'")
        X_rep = reps[fallback_name]
        scaler = StandardScaler()
        X_seed = scaler.fit_transform(X_rep[seed_idx])
        X_pool = scaler.transform(X_rep[pool_idx])
        return X_seed, X_pool, X_rep.shape[1]

    best_name = None
    best_score = -np.inf

    # Require >=2 test samples per fold AND >=2 splits.
    # n_splits = max(2, min(5, n_seed // 2)) guarantees every fold has >=2
    # test rows (since n_seed // n_splits >= 2 when n_splits <= n_seed // 2).
    n_splits = max(2, min(5, n_seed // 2))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for name, X_rep in reps.items():
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_rep[seed_idx])
        try:
            # Use a fresh GP estimator per fold via cross_val_score with
            # scoring='r2'. Wrap in try/except since GP can fail on small folds.
            gp_est = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3),
                normalize_y=True, n_restarts_optimizer=0,
                random_state=42,
            )
            scores = cross_val_score(
                gp_est, X_s, y_seed, cv=kf, scoring='r2',
                error_score=np.nan,
            )
            score = float(np.nanmean(scores)) if np.any(~np.isnan(scores)) else -np.inf
        except Exception:
            score = -np.inf
        if score > best_score:
            best_score = score
            best_name = name

    if best_name is None:
        # CV failed on all reps -- fall back to highest-dim rep deterministically
        best_name = max(reps.keys(), key=lambda k: reps[k].shape[1])
        print(f"    WARNING: CV failed on all reps; falling back to highest-dim "
              f"rep '{best_name}'")

    print(f"    Best single rep: {best_name} (dim={reps[best_name].shape[1]}, "
          f"seed CV R2={best_score:.4f})")

    X_rep = reps[best_name]
    scaler = StandardScaler()
    X_seed = scaler.fit_transform(X_rep[seed_idx])
    X_pool = scaler.transform(X_rep[pool_idx])
    return X_seed, X_pool, X_rep.shape[1]


def build_pca_features(reps, seed_idx, pool_idx, n_components=50):
    """PCA to n_components from concatenated reps."""
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])

    scaler = StandardScaler()
    X_seed_s = scaler.fit_transform(X_concat[seed_idx])
    X_pool_s = scaler.transform(X_concat[pool_idx])

    n_comp = min(n_components, X_seed_s.shape[0] - 1, X_seed_s.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_seed_pca = pca.fit_transform(X_seed_s)
    X_pool_pca = pca.transform(X_pool_s)

    return X_seed_pca, X_pool_pca, n_comp


def build_hybrid_features(reps, labels, seed_idx, pool_idx, budget=50, bp=None,
                          augmentations=None):
    """Build KIRBy hybrid on seed set and transform pool.

    Args:
        bp: Blueprint from search. If provided, uses blueprint params instead
            of hardcoded defaults. The caller's `budget` arg is always honoured
            (so a hybrid_50d condition stays at 50 even if the blueprint was
            discovered at a different budget).
        augmentations: optional dict[name, array] of full-dataset augmentation
            features. Subsetting is done here so create_hybrid sees only seed
            rows.

    Returns (X_seed, X_pool, actual_dim, feature_info, scaler).
    """
    seed_reps = {k: v[seed_idx] for k, v in reps.items()}
    pool_reps = {k: v[pool_idx] for k, v in reps.items()}
    y_seed = labels[seed_idx]

    alloc = bp.allocation_method if bp else 'entropy_1.0'
    imp = bp.importance_method if bp else 'lightgbm_gain'
    int_eval = bp.internal_evaluator if bp else 'match'
    aug_strategy = bp.augmentation_strategy if bp else 'none'
    # SIMA-5: honour the blueprint's step_size (greedy allocation granularity).
    # Bare create_hybrid silently defaults step_size=5, so the hybrid here could
    # differ from the feature space the blueprint was scored against.
    step_sz = getattr(bp, 'step_size', 5) if bp else 5

    seed_augs = ({k: v[seed_idx] for k, v in augmentations.items()}
                 if augmentations else None)
    pool_augs = ({k: v[pool_idx] for k, v in augmentations.items()}
                 if augmentations else None)

    X_seed, feature_info = create_hybrid(
        seed_reps, y_seed,
        allocation_method=alloc,
        importance_method=imp,
        internal_evaluator=int_eval,
        budget=budget,
        step_size=step_sz,
        augmentations=seed_augs,
        augmentation_strategy=aug_strategy,
    )
    X_pool = apply_feature_selection(pool_reps, feature_info)
    aug_info = feature_info.get('augmentation_info') if isinstance(feature_info, dict) else None
    if aug_info and pool_augs:
        X_pool_aug = apply_augmentation_selection(pool_augs, aug_info)
        if X_pool_aug is not None:
            X_pool = np.hstack([X_pool, X_pool_aug])

    scaler = StandardScaler()
    X_seed_s = scaler.fit_transform(X_seed)
    X_pool_s = scaler.transform(X_pool)

    return X_seed_s, X_pool_s, X_seed.shape[1], feature_info, scaler


def build_random_features(reps, seed_idx, pool_idx, n_components=50, rng=None):
    """Random Gaussian projection to n_components."""
    if rng is None:
        rng = np.random.RandomState(42)

    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    d = X_concat.shape[1]

    scaler = StandardScaler()
    X_seed_s = scaler.fit_transform(X_concat[seed_idx])
    X_pool_s = scaler.transform(X_concat[pool_idx])

    # Random projection matrix (Gaussian)
    n_comp = min(n_components, d)
    projection = rng.randn(d, n_comp) / np.sqrt(n_comp)

    X_seed_rp = X_seed_s @ projection
    X_pool_rp = X_pool_s @ projection

    return X_seed_rp, X_pool_rp, n_comp


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_bo_metrics(best_so_far_curves, y_all, budgets):
    """Compute summary metrics from best-so-far curves across seeds.

    Args:
        best_so_far_curves: list of lists (one per seed)
        y_all: all target values (used for top-K thresholds)
        budgets: list of budget checkpoints to report

    Returns:
        dict with AUC, iterations-to-top-1%, iterations-to-top-5%
    """
    # NOTE: thresholds are computed on the BO-oriented `y_all`. When the
    # domain metric is lower_is_better, the caller passes y_bo = -labels, so
    # these thresholds live in NEGATED space. They are only used internally
    # for the iters_to_top1pct / iters_to_top5pct counters (which count
    # iterations, not metric values, so sign does not propagate).
    y_sorted = np.sort(y_all)[::-1]
    top_1pct_threshold = y_sorted[max(0, int(len(y_sorted) * 0.01) - 1)]
    top_5pct_threshold = y_sorted[max(0, int(len(y_sorted) * 0.05) - 1)]

    metrics = {}

    for budget in budgets:
        auc_values = []
        iters_to_top1 = []
        iters_to_top5 = []

        for curve in best_so_far_curves:
            # Truncate curve to budget (+1 for seed)
            c = curve[:budget + 1]

            # AUC of best-so-far curve (trapezoidal)
            if len(c) > 1:
                _trapezoid = getattr(np, 'trapezoid', np.trapz)
                auc = float(_trapezoid(c, dx=1.0))
            else:
                auc = float(c[0]) if c else 0.0
            auc_values.append(auc)

            # Iterations to find top-1%
            found_1 = None
            for i, val in enumerate(c):
                if val >= top_1pct_threshold:
                    found_1 = i
                    break
            iters_to_top1.append(found_1)

            # Iterations to find top-5%
            found_5 = None
            for i, val in enumerate(c):
                if val >= top_5pct_threshold:
                    found_5 = i
                    break
            iters_to_top5.append(found_5)

        # Aggregate across seeds
        metrics[f'budget_{budget}'] = {
            'auc_mean': float(np.mean(auc_values)),
            'auc_std': float(np.std(auc_values)),
            'iters_to_top1pct_mean': float(np.nanmean([
                x for x in iters_to_top1 if x is not None
            ])) if any(x is not None for x in iters_to_top1) else None,
            'iters_to_top1pct_found': sum(1 for x in iters_to_top1 if x is not None),
            'iters_to_top5pct_mean': float(np.nanmean([
                x for x in iters_to_top5 if x is not None
            ])) if any(x is not None for x in iters_to_top5) else None,
            'iters_to_top5pct_found': sum(1 for x in iters_to_top5 if x is not None),
        }

    return metrics


# =============================================================================
# EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """S11: Bayesian Optimization in Compact Hybrid Space."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S11: BAYESIAN OPTIMIZATION IN HYBRID SPACE")
    print("=" * 70)

    max_iterations = kwargs.get('max_iterations', 100)
    n_seeds = kwargs.get('n_seeds', 3)
    rebuild_interval = kwargs.get('rebuild_interval', 0)
    hybrid_budget = kwargs.get('hybrid_budget', 50)
    target_dim = kwargs.get('target_dim', 50)

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    task_type = data['task_type']
    n_samples = len(labels)

    # S11 is molecular-only: requires ECFP4 or molecular fingerprints for
    # Tanimoto kernel condition and library-screening framing.
    domain = data.get('domain', 'molecular')
    if domain not in ('molecular', 'reaction'):
        print(f"  Skipping S11: domain='{domain}' (molecular-only experiment)")
        return {'skipped': True, 'reason': f'domain={domain}', 'domain': domain}

    # S11 is regression-only. On binary classification labels the Bayesian-
    # optimization "best value found so far" saturates to 1.0 on the first
    # positive picked, so every condition ties at a degenerate score — it runs
    # without error but carries no signal (the library-screening / active-
    # learning framing assumes a continuous activity to optimise). Skip cleanly.
    if task_type == 'classification':
        print("  Skipping S11: task_type='classification' (Bayesian-optimization "
              "objective is degenerate on binary labels; regression-only).")
        return {'skipped': True, 'reason': 'classification_degenerate',
                'task_type': task_type}

    if n_samples < 200:
        print(f"  Skipping S11: n={n_samples} < 200 (BO seed set too small for greedy allocation)")
        return {'error': 'dataset_too_small', 'n_samples': n_samples}

    # SIMA-10: the BO seed/pool split MUST be group-aware. Random splits are
    # forbidden (feedback_no_random_splits). Check ONCE here (the requirement is
    # deterministic in `groups`, not per-seed) and skip the whole experiment
    # cleanly with a flag rather than raising mid-loop (which used to crash all
    # conditions) or silently shuffling. >=20 groups are needed so the 5% seed
    # split lands at least one group.
    _n_unique_groups = len(np.unique(groups)) if groups is not None else 0
    if groups is None:
        print("  Skipping S11: no groups provided; group-aware BO split required "
              "(random splits forbidden, see feedback_no_random_splits).")
        return {'skipped': True, 'reason': 'no_groups_for_grouped_bo_split',
                'n_samples': n_samples}
    if _n_unique_groups < 20:
        print(f"  Skipping S11: only {_n_unique_groups} unique groups (<20); "
              f"too few for a group-aware 5%/80% BO split.")
        return {'skipped': True, 'reason': f'too_few_groups:{_n_unique_groups}',
                'n_unique_groups': _n_unique_groups, 'n_samples': n_samples}

    # Get blueprint from pool helper (used by hybrid conditions)
    augmentations = data.get('augmentations')
    try:
        search, meta = get_best_search_for_downstream(
            reps, labels, groups,
            # RF blueprint reused across surrogates — see EXPERIMENT_DEBUGGING.md
            downstream_model='rf', budget=hybrid_budget,
            augmentations=augmentations,
            speed=config.get('search_speed', 'full'),
            dataset_name=dataset_name, task_type=task_type,
            data=data,
        )
        bp = meta['best_config']['blueprint'] if meta else None
    except Exception:
        bp = None

    # Domain-aware metric for determining "lower_is_better"
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    # BO maximizes -metric when lower_is_better, so curves & AUC are over the
    # negated objective. We sign-flip the AUC at STORAGE time below so that
    # JSON consumers read auc_mean in the natural metric direction without
    # needing to know about negation. display_sign stays 1.0 at display sites.
    display_sign = 1.0

    # For BO we always MAXIMIZE. If the task's primary metric is lower-is-better
    # (e.g., RMSE), negate targets so BO maximizes -RMSE.
    if lower_is_better:
        y_bo = -labels.copy()
        print(f"  Metric: {metric_name} (lower is better) -> negating for BO maximization")
    else:
        y_bo = labels.copy()
        print(f"  Metric: {metric_name} (higher is better) -> direct maximization")

    # Budget checkpoints for metric computation
    budget_checkpoints = [b for b in [25, 50, 100, 200] if b <= max_iterations]
    if max_iterations not in budget_checkpoints:
        budget_checkpoints.append(max_iterations)

    print(f"  n_samples: {n_samples}")
    print(f"  n_reps: {len(reps)} ({', '.join(f'{k}={v.shape[1]}' for k, v in reps.items())})")
    print(f"  max_iterations: {max_iterations}")
    print(f"  n_seeds: {n_seeds}")
    print(f"  rebuild_interval: {rebuild_interval}")
    print(f"  hybrid_budget: {hybrid_budget}")
    print(f"  budget_checkpoints: {budget_checkpoints}")

    # Top-K thresholds (on BO-oriented y)
    y_sorted = np.sort(y_bo)[::-1]
    top_1pct = y_sorted[max(0, int(len(y_sorted) * 0.01) - 1)]
    top_5pct = y_sorted[max(0, int(len(y_sorted) * 0.05) - 1)]
    print(f"  top-1% threshold: {top_1pct:.4f}")
    print(f"  top-5% threshold: {top_5pct:.4f}")

    # Multi-objective: combine primary label with QED (molecular only)
    multi_objective = kwargs.get('multi_objective', False)
    smiles = data.get('smiles', None)
    y_multi = None
    if multi_objective:
        if smiles is not None:
            try:
                qed_scores = compute_qed_scores(smiles)
                y_multi = compute_combined_objective(
                    labels, qed_scores, lower_is_better)
                print(f"  Multi-objective: label + QED geometric mean")
                print(f"    QED range: [{qed_scores.min():.3f}, "
                      f"{qed_scores.max():.3f}]")
                print(f"    Combined range: [{y_multi.min():.3f}, "
                      f"{y_multi.max():.3f}]")
            except Exception as e:
                print(f"  Multi-objective failed ({e}), disabled")
                multi_objective = False
        else:
            print(f"  Multi-objective skipped (no SMILES)")
            multi_objective = False

    conditions = [
        'ecfp4_raw',
        'tanimoto_ecfp4',
        'best_single',
        'pca_50d',
        'hybrid_50d',
        'hybrid_50d_refit',
        'random_50d',
        'random_acq',
        # Surrogate model comparison on hybrid space
        'hybrid_ngboost',
        'hybrid_bnn',
        'hybrid_gauche_gp',
    ]
    if multi_objective and y_multi is not None:
        conditions.extend(['multi_obj_hybrid', 'multi_obj_random'])

    # Per-rep BO: run BO in each individual rep's space (Moesser Ch.2 comparison)
    per_rep_bo = kwargs.get('per_rep_bo', False)
    if per_rep_bo:
        for rname in sorted(reps.keys()):
            conditions.append(f'single_{rname}')
        print(f"  Per-rep BO: {len(reps)} individual rep conditions added")

    results = {
        'conditions': {},
        'config': {
            'max_iterations': max_iterations,
            'n_seeds': n_seeds,
            'rebuild_interval': rebuild_interval,
            'hybrid_budget': hybrid_budget,
            'target_dim': target_dim,
            'budget_checkpoints': budget_checkpoints,
            'lower_is_better': lower_is_better,
            'metric_name': metric_name,
            'n_samples': n_samples,
        },
    }

    _has_ecfp4 = 'ecfp4' in reps

    for condition in conditions:
        print(f"\n  {'='*60}")
        print(f"  Condition: {condition}")
        print(f"  {'='*60}")

        # --- SIMA-11: skip conditions whose precondition is absent, with a flag,
        # instead of running a degenerate curve that looks like a real method. ---
        # (1) Optional-surrogate hybrid_<surr> conditions: if the package is not
        #     installed, every fit fails -> pure random acquisition mislabelled.
        if condition.startswith('hybrid_') and condition.split('_', 1)[1] in SURROGATE_MODELS:
            _surr = condition.split('_', 1)[1]
            if not _surrogate_available(_surr):
                print(f"    SKIP: surrogate '{_surr}' package not installed "
                      f"-> would be pure random acquisition mislabelled as a method.")
                results['conditions'][condition] = {
                    'skipped': True,
                    'reason': f'surrogate_unavailable:{_surr}',
                }
                continue
        # (2) tanimoto_ecfp4 needs real binary fingerprints. With no ecfp4 rep it
        #     falls back to a continuous max-dim rep, where the Tanimoto kernel is
        #     meaningless. Skip + flag rather than report a fake "tanimoto" result.
        if condition == 'tanimoto_ecfp4' and not _has_ecfp4:
            print("    SKIP: tanimoto_ecfp4 requires an 'ecfp4' rep "
                  "(Tanimoto kernel is meaningless on continuous features).")
            results['conditions'][condition] = {
                'skipped': True,
                'reason': 'no_ecfp4_rep_for_tanimoto',
            }
            continue

        all_curves = []
        total_gp_failures = 0
        cond_dim = target_dim  # default; overwritten by feature builders
        degenerate_split_seeds = 0  # SIMA-10b: count of skipped seed iterations

        for seed_i in range(n_seeds):
            seed = 42 + seed_i
            rng = np.random.RandomState(seed)
            print(f"\n    Seed {seed_i + 1}/{n_seeds} (random_state={seed})")

            # --- Split: 5% seed, 80% pool, 15% unused (to keep pool at ~80%) ---
            # Respect groups if available
            seed_frac = 0.05
            pool_frac = 0.80
            # Total used = seed + pool = 85%, held out = 15%

            all_idx = np.arange(n_samples)
            # groups is guaranteed non-None with >=20 unique groups by the
            # early skip check above (SIMA-10) — splitting is always group-aware.
            #
            # SIMA-10b: the group-aware seed/sub-pool split can be degenerate
            # even past the >=20-group gate, in TWO distinct ways, BOTH handled
            # here as a graceful per-seed skip+flag (never a random-split
            # fallback — feedback_no_random_splits — and never a silent empty
            # result):
            #   (a) GroupShuffleSplit RAISES ValueError when a whole-group
            #       allocation makes a side of the split empty (e.g. the
            #       sub-pool split over a reduced/coarse group set). We catch it.
            #   (b) GroupShuffleSplit returns a non-empty but too-small seed
            #       (<2 rows, unfittable) or an empty pool. We detect it below.
            try:
                # Use GroupShuffleSplit: first split off seed set
                gss_seed = GroupShuffleSplit(
                    n_splits=1, test_size=1.0 - seed_frac, random_state=seed)
                seed_idx, rest_idx = next(gss_seed.split(all_idx, y_bo, groups))

                # From rest, take pool_frac / (1 - seed_frac) as pool
                pool_from_rest = pool_frac / (1.0 - seed_frac)
                pool_from_rest = min(pool_from_rest, 0.95)  # safety
                gss_pool = GroupShuffleSplit(
                    n_splits=1, test_size=1.0 - pool_from_rest, random_state=seed)
                rest_groups = groups[rest_idx]
                pool_local_idx, _ = next(gss_pool.split(
                    rest_idx, y_bo[rest_idx], rest_groups))
                pool_idx = rest_idx[pool_local_idx]
            except ValueError as _split_err:
                # (a) split impossible for this seed/group layout.
                degenerate_split_seeds += 1
                print(f"      SKIP seed {seed_i + 1}: group-aware split raised "
                      f"({_split_err}); too few/uneven groups for a 5%/80% "
                      f"split.")
                continue

            # (b) split succeeded but is too small to fit / acquire from.
            if len(seed_idx) < 2 or len(pool_idx) < 1:
                degenerate_split_seeds += 1
                print(f"      SKIP seed {seed_i + 1}: degenerate group-aware "
                      f"split (seed={len(seed_idx)}, pool={len(pool_idx)}); "
                      f"too few/uneven groups for a 5%/80% split.")
                continue

            # Select target: combined objective for multi_obj, else standard
            y_cond = y_multi if condition.startswith('multi_obj_') else y_bo
            y_seed = y_cond[seed_idx]
            y_pool = y_cond[pool_idx]

            print(f"      seed: {len(seed_idx)}, pool: {len(pool_idx)}")

            # --- Build features for this condition ---
            try:
                if condition == 'ecfp4_raw':
                    X_seed, X_pool, dim = build_ecfp4_features(
                        reps, seed_idx, pool_idx)

                elif condition == 'tanimoto_ecfp4':
                    # Same features as ecfp4_raw but with Tanimoto kernel
                    # (proper kernel for binary fingerprints). No scaling —
                    # Tanimoto operates on raw binary/count features.
                    if 'ecfp4' in reps:
                        X_all = reps['ecfp4']
                    else:
                        name = max(reps.keys(), key=lambda k: reps[k].shape[1])
                        X_all = reps[name]
                    X_seed = X_all[seed_idx].astype(np.float64)
                    X_pool = X_all[pool_idx].astype(np.float64)
                    dim = X_all.shape[1]

                elif condition == 'best_single':
                    X_seed, X_pool, dim = build_best_single_features(
                        reps, y_bo, seed_idx, pool_idx)

                elif condition == 'pca_50d':
                    X_seed, X_pool, dim = build_pca_features(
                        reps, seed_idx, pool_idx, n_components=target_dim)

                elif condition == 'hybrid_50d':
                    if rebuild_interval > 0:
                        # BO with periodic hybrid rebuilding. refit_interval=10
                        # keeps the GP-refit cadence aligned with run_bo_loop
                        # (SIMA-12).
                        curve, gp_fail = _run_bo_with_rebuild(
                            reps, y_bo, seed_idx, pool_idx,
                            max_iterations, hybrid_budget, rebuild_interval,
                            rng=rng, bp=bp, augmentations=augmentations,
                            refit_interval=10)
                        all_curves.append(curve)
                        total_gp_failures += gp_fail
                        best_final = curve[-1] if curve else float('nan')
                        print(f"      final best: {best_final:.4f}, "
                              f"GP failures: {gp_fail}")
                        continue
                    else:
                        X_seed, X_pool, dim, _, _ = build_hybrid_features(
                            reps, y_bo, seed_idx, pool_idx,
                            budget=hybrid_budget, bp=bp,
                            augmentations=augmentations)

                elif condition == 'hybrid_50d_refit':
                    # Always rebuild hybrid every 25 iterations; GP refit stays
                    # at 10 to match run_bo_loop (SIMA-12: rebuild cadence and
                    # refit cadence are distinct).
                    curve, gp_fail = _run_bo_with_rebuild(
                        reps, y_bo, seed_idx, pool_idx,
                        max_iterations, hybrid_budget, 25,
                        rng=rng, bp=bp, augmentations=augmentations,
                        refit_interval=10)
                    all_curves.append(curve)
                    total_gp_failures += gp_fail
                    best_final = curve[-1] if curve else float('nan')
                    print(f"      final best: {best_final:.4f}, "
                          f"GP failures: {gp_fail}")
                    continue

                elif condition == 'random_50d':
                    X_seed, X_pool, dim = build_random_features(
                        reps, seed_idx, pool_idx,
                        n_components=target_dim, rng=rng)

                elif condition == 'random_acq':
                    curve = run_random_loop(y_seed, y_pool, max_iterations, rng)
                    all_curves.append(curve)
                    best_final = curve[-1] if curve else float('nan')
                    print(f"      final best: {best_final:.4f} (random)")
                    continue

                elif condition == 'multi_obj_hybrid':
                    X_seed, X_pool, dim, _, _ = build_hybrid_features(
                        reps, y_cond, seed_idx, pool_idx,
                        budget=hybrid_budget, bp=bp,
                        augmentations=augmentations)

                elif condition == 'multi_obj_random':
                    curve = run_random_loop(y_seed, y_pool, max_iterations, rng)
                    all_curves.append(curve)
                    best_final = curve[-1] if curve else float('nan')
                    print(f"      final best: {best_final:.4f} (random, multi-obj)")
                    continue

                elif condition.startswith('single_'):
                    # Per-rep BO: use a single rep's features (scaled)
                    rep_name = condition[len('single_'):]
                    if rep_name not in reps:
                        print(f"      rep '{rep_name}' not found, skipping")
                        continue
                    X_rep = reps[rep_name]
                    scaler = StandardScaler()
                    X_seed = scaler.fit_transform(X_rep[seed_idx])
                    X_pool = scaler.transform(X_rep[pool_idx])
                    dim = X_rep.shape[1]

                elif condition.startswith('hybrid_') and condition.split('_', 1)[1] in SURROGATE_MODELS:
                    # Surrogate model variant on hybrid features
                    X_seed, X_pool, dim, _, _ = build_hybrid_features(
                        reps, y_bo, seed_idx, pool_idx,
                        budget=hybrid_budget, bp=bp,
                        augmentations=augmentations)

                else:
                    print(f"      Unknown condition: {condition}")
                    continue

            except Exception as e:
                print(f"      Feature build FAILED: {e}")
                all_curves.append([float('nan')])
                continue

            cond_dim = dim
            print(f"      dim: {dim}")

            # --- Determine surrogate model ---
            if condition.startswith('hybrid_') and condition.split('_', 1)[1] in SURROGATE_MODELS:
                surr = condition.split('_', 1)[1]
            else:
                surr = 'gp'

            # --- Run BO loop ---
            kt = 'tanimoto' if condition == 'tanimoto_ecfp4' else 'rbf'
            curve, gp_fail = run_bo_loop(
                X_seed, y_seed, X_pool, y_pool,
                max_iterations=max_iterations,
                refit_interval=10, rng=rng,
                surrogate=surr,
                kernel_type=kt,
            )
            all_curves.append(curve)
            total_gp_failures += gp_fail

            best_final = curve[-1] if curve else float('nan')
            print(f"      final best: {best_final:.4f}, GP failures: {gp_fail}")

        # --- Aggregate metrics across seeds ---
        if not all_curves:
            # SIMA-10b: distinguish "all seeds had a degenerate group-aware
            # split" from a generic failure so the empty result is explicit and
            # flagged, not silently swallowed.
            if degenerate_split_seeds >= n_seeds:
                results['conditions'][condition] = {
                    'skipped': True,
                    'reason': 'all_seeds_degenerate_group_split',
                    'degenerate_split_seeds': degenerate_split_seeds,
                }
            else:
                results['conditions'][condition] = {'error': 'no successful runs'}
            continue

        y_for_metrics = y_multi if condition.startswith('multi_obj_') else y_bo
        metrics = compute_bo_metrics(all_curves, y_for_metrics, budget_checkpoints)

        # Sign-flip stored AUC back into the natural metric direction when the
        # underlying domain metric is lower_is_better. BO maximizes -metric so
        # `auc_mean` from compute_bo_metrics is the AUC of the -metric curve;
        # negate so consumers read auc_mean in the original metric direction.
        # auc_std stays untouched (std is sign-invariant). top-K iteration
        # counts are unaffected (they count iters, not metric values).
        # Multi-objective curves use y_multi which is constructed in
        # "higher is better" orientation (see compute_combined_objective),
        # so they are not sign-flipped.
        if lower_is_better and not condition.startswith('multi_obj_'):
            for bk_key, bk_metrics in metrics.items():
                if 'auc_mean' in bk_metrics:
                    bk_metrics['auc_mean'] = -bk_metrics['auc_mean']

        # Mean curve for plotting
        max_len = max(len(c) for c in all_curves)
        padded = []
        for c in all_curves:
            if len(c) < max_len:
                c_ext = c + [c[-1]] * (max_len - len(c))
            else:
                c_ext = c
            padded.append(c_ext)
        mean_curve = list(np.nanmean(padded, axis=0))
        std_curve = list(np.nanstd(padded, axis=0))

        # SIMA-11: when a surrogate-driven condition's fits ALL fail, the BO loop
        # falls back to random selection every iteration, so the reported "best
        # curve" is actually pure random acquisition. Flag it (informational --
        # for ecfp4_raw this degeneracy IS the hypothesis, but the flag stops a
        # silently-random curve being read as a successful method elsewhere).
        _is_random_only = condition in ('random_acq', 'random_50d', 'multi_obj_random')
        _max_possible_fails = n_seeds * max_iterations
        degenerate_random = (
            not _is_random_only
            and _max_possible_fails > 0
            and total_gp_failures >= 0.99 * _max_possible_fails
        )

        results['conditions'][condition] = {
            'metrics': metrics,
            'mean_curve': mean_curve,
            'std_curve': std_curve,
            'n_curves': len(all_curves),
            'total_gp_failures': total_gp_failures,
            'degenerate_random_fallback': bool(degenerate_random),
            'degenerate_split_seeds': degenerate_split_seeds,
            'dim': cond_dim,
        }
        if degenerate_random:
            print(f"    [flag] {condition}: every surrogate fit failed "
                  f"({total_gp_failures}/{_max_possible_fails}) -> this curve is "
                  f"pure random acquisition, not a real {condition} result.")

        # Print summary for this condition
        for bk in budget_checkpoints:
            key = f'budget_{bk}'
            if key in metrics:
                m = metrics[key]
                print(f"    Budget {bk}: AUC={display_sign*m['auc_mean']:.1f}"
                      f"+/-{m['auc_std']:.1f}, "
                      f"top1%: {m['iters_to_top1pct_mean']}"
                      f"/{m['iters_to_top1pct_found']} found, "
                      f"top5%: {m['iters_to_top5pct_mean']}"
                      f"/{m['iters_to_top5pct_found']} found")

    # =================================================================
    # FINAL SUMMARY
    # =================================================================
    print(f"\n{'='*70}")
    # Stored AUC is already sign-flipped at storage time when lower_is_better,
    # so the printed values are always in the natural metric direction.
    if lower_is_better:
        print("  SUMMARY: Best-found AUC at max budget "
              "(in original metric direction; lower is better)")
    else:
        print("  SUMMARY: Best-found AUC at max budget (higher is better)")
    print(f"{'='*70}")

    max_budget_key = f'budget_{max(budget_checkpoints)}'
    summary_rows = []
    for cond in conditions:
        if cond in results['conditions'] and 'metrics' in results['conditions'][cond]:
            m = results['conditions'][cond]['metrics'].get(max_budget_key, {})
            auc = m.get('auc_mean', float('nan'))
            auc_std = m.get('auc_std', float('nan'))
            gp_fail = results['conditions'][cond].get('total_gp_failures', 0)
            dim = results['conditions'][cond].get('dim', '?')
            summary_rows.append((cond, dim, auc, auc_std, gp_fail))

    print(f"  {'Condition':<16} {'Dim':>5} {'AUC':>10} {'Std':>8} {'GP Fail':>8}")
    print(f"  {'-'*16} {'-'*5} {'-'*10} {'-'*8} {'-'*8}")
    for cond, dim, auc, auc_std, gp_fail in summary_rows:
        print(f"  {cond:<16} {dim:>5} {display_sign*auc:>10.1f} "
              f"{auc_std:>8.1f} {gp_fail:>8}")

    # Highlight key comparison: hybrid vs ecfp4
    hybrid_res = results['conditions'].get('hybrid_50d', {}).get('metrics', {})
    ecfp4_res = results['conditions'].get('ecfp4_raw', {}).get('metrics', {})
    if hybrid_res and ecfp4_res:
        h_auc = hybrid_res.get(max_budget_key, {}).get('auc_mean', float('nan'))
        e_auc = ecfp4_res.get(max_budget_key, {}).get('auc_mean', float('nan'))
        if not (np.isnan(h_auc) or np.isnan(e_auc)):
            # Compare in original metric direction.
            h_disp = display_sign * h_auc
            e_disp = display_sign * e_auc
            pct_gain = 100.0 * (h_disp - e_disp) / abs(e_disp) if e_disp != 0 else 0
            print(f"\n  Hybrid vs ECFP4: AUC gain = {h_disp - e_disp:+.1f} "
                  f"({pct_gain:+.1f}%)")

    # Build structured output
    output = make_structured_output(
        's11_bayesian_optimization', dataset_name, data, config,
        results, primary_metric='bo_auc', lower_is_better=False,
    )
    return output


# =============================================================================
# HYBRID REBUILD BO
# =============================================================================

def _run_bo_with_rebuild(reps, y_bo, seed_idx, pool_idx,
                         max_iterations, hybrid_budget, rebuild_interval,
                         rng=None, bp=None, augmentations=None,
                         refit_interval=10):
    """BO loop with periodic hybrid rebuilding.

    Every rebuild_interval iterations, re-create the hybrid using all
    labeled data accumulated so far, and re-transform the remaining pool.

    Args:
        bp: Blueprint from search. If provided, uses blueprint params instead
            of hardcoded defaults.
        augmentations: optional dict[name, array] of full-dataset augmentation
            features. Subset per-call so create_hybrid sees only labeled rows.
        refit_interval: GP hyperparameter re-optimisation cadence (SIMA-12).
            Kept SEPARATE from rebuild_interval (the hybrid-rebuild cadence);
            both previously diverged because this path hard-coded 10 while
            run_bo_loop exposed refit_interval as a parameter. Defaults to 10
            to match run_bo_loop's default.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    y_seed = y_bo[seed_idx]
    y_pool = y_bo[pool_idx]

    alloc = bp.allocation_method if bp else 'entropy_1.0'
    imp = bp.importance_method if bp else 'lightgbm_gain'
    int_eval = bp.internal_evaluator if bp else 'match'
    # SIMA-5: thread the blueprint's step_size and augmentation_strategy so the
    # rebuilt hybrid matches the search blueprint's feature space (bare
    # create_hybrid silently defaults step_size=5 and would otherwise hardcode
    # 'uncapped' even when the blueprint used 'none'/'capped').
    step_sz = getattr(bp, 'step_size', 5) if bp else 5
    aug_strategy = (getattr(bp, 'augmentation_strategy', 'uncapped') if bp
                    else ('uncapped' if augmentations else 'none'))

    # Initial hybrid build — honour caller's hybrid_budget; blueprint may have
    # been discovered at a different budget but the BO condition fixes one.
    seed_reps = {k: v[seed_idx] for k, v in reps.items()}
    pool_reps_all = {k: v[pool_idx] for k, v in reps.items()}
    seed_augs = ({k: v[seed_idx] for k, v in augmentations.items()}
                 if augmentations else None)

    X_seed, feature_info = create_hybrid(
        seed_reps, y_seed,
        allocation_method=alloc,
        importance_method=imp,
        internal_evaluator=int_eval,
        budget=hybrid_budget,
        step_size=step_sz,
        augmentations=seed_augs,
        augmentation_strategy=aug_strategy,
    )
    X_pool_all = apply_feature_selection(pool_reps_all, feature_info)

    scaler = StandardScaler()
    X_lab = scaler.fit_transform(X_seed)
    X_pool_transformed = scaler.transform(X_pool_all)

    y_lab = y_seed.copy()
    pool_mask = np.ones(len(y_pool), dtype=bool)
    labeled_original_idx = list(seed_idx)  # track original indices

    best_so_far = [float(np.max(y_lab))]
    gp_failures = 0
    last_optimize = -refit_interval

    for it in range(max_iterations):
        if pool_mask.sum() == 0:
            break

        # Rebuild hybrid if interval is set
        if rebuild_interval > 0 and it > 0 and it % rebuild_interval == 0:
            try:
                lab_idx_arr = np.array(labeled_original_idx)
                lab_reps = {k: v[lab_idx_arr] for k, v in reps.items()}
                remaining_pool_orig = pool_idx[pool_mask]
                remaining_reps = {k: v[remaining_pool_orig] for k, v in reps.items()}
                lab_augs = ({k: v[lab_idx_arr] for k, v in augmentations.items()}
                            if augmentations else None)

                X_lab_new, feature_info = create_hybrid(
                    lab_reps, y_lab,
                    allocation_method=alloc,
                    importance_method=imp,
                    internal_evaluator=int_eval,
                    budget=hybrid_budget,
                    step_size=step_sz,
                    augmentations=lab_augs,
                    augmentation_strategy=aug_strategy,
                )
                X_remaining = apply_feature_selection(remaining_reps, feature_info)

                scaler = StandardScaler()
                X_lab = scaler.fit_transform(X_lab_new)

                # Rebuild pool transform for remaining points
                X_pool_transformed = np.full(
                    (len(y_pool), X_remaining.shape[1]), np.nan)
                X_pool_transformed[pool_mask] = scaler.transform(X_remaining)

                print(f"        [rebuild at iter {it}, "
                      f"labeled={len(y_lab)}, pool={pool_mask.sum()}]")
            except Exception as e:
                print(f"        [rebuild failed at iter {it}: {e}]")

        # Fit GP
        optimize = (it - last_optimize) >= refit_interval
        if optimize:
            last_optimize = it

        gp, success = fit_gp(X_lab, y_lab, optimize=optimize)

        if not success:
            gp_failures += 1
            available = np.where(pool_mask)[0]
            selected_pool_idx = available[rng.randint(len(available))]
        else:
            X_pool_available = X_pool_transformed[pool_mask]
            mu, sigma = gp.predict(X_pool_available, return_std=True)
            y_best = float(np.max(y_lab))
            ei = expected_improvement(mu, sigma, y_best)
            best_ei_idx = np.argmax(ei)
            available = np.where(pool_mask)[0]
            selected_pool_idx = available[best_ei_idx]

        # Reveal label
        x_new = X_pool_transformed[selected_pool_idx:selected_pool_idx + 1]
        y_new = y_pool[selected_pool_idx]

        X_lab = np.vstack([X_lab, x_new])
        y_lab = np.append(y_lab, y_new)
        pool_mask[selected_pool_idx] = False
        labeled_original_idx.append(pool_idx[selected_pool_idx])

        best_so_far.append(float(max(best_so_far[-1], y_new)))

    return best_so_far, gp_failures


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = create_experiment_parser(
        'S11: Bayesian Optimization in Compact Hybrid Space',
        has_delta=True, has_augment=True)
    parser.add_argument('--max-iterations', type=int, default=100,
                        help='Maximum BO iterations per run (default: 100)')
    parser.add_argument('--n-seeds', type=int, default=3,
                        help='Number of random seeds to average (default: 3)')
    parser.add_argument('--rebuild-interval', type=int, default=0,
                        help='Rebuild hybrid every N iterations (0=never, default: 0)')
    parser.add_argument('--hybrid-budget', type=int, default=50,
                        help='KIRBy hybrid budget (default: 50)')
    parser.add_argument('--target-dim', type=int, default=50,
                        help='Target dimensionality for PCA/random (default: 50)')
    parser.add_argument('--multi-objective', action='store_true', default=False,
                        help='Enable multi-objective BO: optimize label + QED '
                             '(molecular datasets with SMILES only, per Moesser 2023)')
    parser.add_argument('--per-rep-bo', action='store_true', default=False,
                        help='Run BO in each individual rep space (shows which '
                             'reps give GP the best landscape, per Moesser Ch.2)')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"N seeds: {args.n_seeds}")
    print(f"Rebuild interval: {args.rebuild_interval}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  max_iterations=args.max_iterations,
                  n_seeds=args.n_seeds,
                  rebuild_interval=args.rebuild_interval,
                  hybrid_budget=args.hybrid_budget,
                  target_dim=args.target_dim,
                  multi_objective=args.multi_objective,
                  per_rep_bo=args.per_rep_bo)
    save_experiment_results(results, args.output, 's11_bayesian_optimization',
                            args.dataset)
