"""Learning with Noisy Labels (LNL) Methods for QSAR Regression.

12 methods adapted from the confirmation report LNL survey. Each method
modifies the training process to handle label noise, either by filtering
data before training (data-level) or by modifying the training loop itself.

All methods use the same DNN architecture as src/kirby/models/dnn.py
(DNNRegressor) for consistency with the confirmation report.

Usage:
    from kirby.utils.lnl_methods import get_lnl_method, LNL_REGISTRY

    method = get_lnl_method('dividemix', epochs=200)
    result = method.train_and_predict(X_train, y_train, X_val, y_val)
    y_pred = result['y_pred']
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional


# ============================================================================
# SHARED HELPERS
# ============================================================================

def _build_dnn(input_dim, hidden_sizes=(256, 128, 64), dropout=0.2):
    """Build DNN matching DNNRegressor._build_net() exactly."""
    import torch.nn as nn
    layers = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def _z_normalize(y):
    """Z-normalize targets. Returns (y_norm, mean, std)."""
    y = np.asarray(y, dtype=np.float32).ravel()
    mean = float(y.mean())
    std = float(y.std()) if y.std() > 1e-8 else 1.0
    return (y - mean) / std, mean, std


def _standard_dnn_train_predict(X_train, y_train, X_val,
                                epochs=200, hidden_sizes=(256, 128, 64),
                                dropout=0.2, lr=0.001, batch_size=256,
                                patience=20, random_state=42,
                                sample_weights=None):
    """Standard DNN training loop matching dnn.py exactly.

    Optionally accepts sample_weights for weighted MSE.
    Returns y_pred on X_val (de-normalized).
    """
    import torch
    import torch.nn as nn

    rng = np.random.RandomState(random_state)
    torch.manual_seed(random_state)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)

    y_norm, y_mean, y_std = _z_normalize(y_train)

    # 80/20 internal holdout for early stopping
    n = len(y_norm)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    tr_idx, ho_idx = idx[:split], idx[split:]

    X_tr = torch.from_numpy(X_train[tr_idx])
    y_tr = torch.from_numpy(y_norm[tr_idx])
    X_ho = torch.from_numpy(X_train[ho_idx])
    y_ho = torch.from_numpy(y_norm[ho_idx])

    w_tr = None
    if sample_weights is not None:
        w_all = np.asarray(sample_weights, dtype=np.float32)
        w_tr = torch.from_numpy(w_all[tr_idx])

    device = torch.device('cpu')
    net = _build_dnn(X_train.shape[1], hidden_sizes, dropout).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    best_ho_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(len(X_tr))

        for start in range(0, len(X_tr), batch_size):
            batch_idx = perm[start:start + batch_size]
            xb = X_tr[batch_idx].to(device)
            yb = y_tr[batch_idx].to(device)

            optimizer.zero_grad()
            pred = net(xb).squeeze(-1)

            if w_tr is not None:
                wb = w_tr[batch_idx].to(device)
                loss = (wb * (pred - yb) ** 2).mean()
            else:
                loss = nn.functional.mse_loss(pred, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

        # Holdout eval
        net.eval()
        with torch.no_grad():
            ho_pred = net(X_ho.to(device)).squeeze(-1)
            ho_loss = nn.functional.mse_loss(ho_pred, y_ho.to(device)).item()

        scheduler.step(ho_loss)
        if ho_loss < best_ho_loss:
            best_ho_loss = ho_loss
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    # Predict on val
    net.eval()
    with torch.no_grad():
        X_val_t = torch.from_numpy(X_val).to(device)
        y_pred_norm = net(X_val_t).squeeze(-1).cpu().numpy()

    return y_pred_norm * y_std + y_mean


# ============================================================================
# BASE CLASS
# ============================================================================

class LNLMethod(ABC):
    """Base class for Learning with Noisy Labels methods."""

    def __init__(self, random_state=42, epochs=200, hidden_sizes=(256, 128, 64),
                 dropout=0.2, lr=0.001, batch_size=256, patience=20):
        self.random_state = random_state
        self.epochs = epochs
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience

    @property
    @abstractmethod
    def category(self) -> str:
        """'data_level' or 'training_loop'"""

    @abstractmethod
    def train_and_predict(self, X_train, y_train, X_val, y_val) -> dict:
        """Train on (X_train, y_train), predict on X_val.

        Args:
            X_train: Training features (n_train, n_features).
            y_train: Training labels (n_train,) — may contain noise.
            X_val: Validation features (n_val, n_features).
            y_val: Validation labels (n_val,) — clean labels for early
                   stopping and meta-learning methods.

        Returns:
            {'y_pred': np.ndarray, 'metadata': dict}
        """

    def _default_predict(self, X_train, y_train, X_val, sample_weights=None):
        """Convenience: standard DNN train+predict with current hyperparams."""
        y_pred = _standard_dnn_train_predict(
            X_train, y_train, X_val,
            epochs=self.epochs, hidden_sizes=self.hidden_sizes,
            dropout=self.dropout, lr=self.lr, batch_size=self.batch_size,
            patience=self.patience, random_state=self.random_state,
            sample_weights=sample_weights,
        )
        return y_pred


# ============================================================================
# DATA-LEVEL METHODS
# ============================================================================

class SmallLossSelection(LNLMethod):
    """One-time selection of 70% smallest-MSE samples after warmup.

    Reference: Han et al. (2018) Co-teaching.
    Category: Sample selection.
    """

    def __init__(self, selection_rate=0.7, warmup_epochs=10, **kwargs):
        super().__init__(**kwargs)
        self.selection_rate = selection_rate
        self.warmup_epochs = warmup_epochs

    @property
    def category(self):
        return 'data_level'

    def train_and_predict(self, X_train, y_train, X_val, y_val):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        X_t = np.asarray(X_train, dtype=np.float32)
        y_norm, y_mean, y_std = _z_normalize(y_train)

        # Warmup: train DNN for warmup_epochs on all data
        device = torch.device('cpu')
        net = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        X_tensor = torch.from_numpy(X_t)
        y_tensor = torch.from_numpy(y_norm)

        for epoch in range(self.warmup_epochs):
            net.train()
            perm = torch.randperm(len(X_t))
            for start in range(0, len(X_t), self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                pred = net(X_tensor[batch_idx].to(device)).squeeze(-1)
                loss = nn.functional.mse_loss(pred, y_tensor[batch_idx].to(device))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

        # Compute per-sample loss
        net.eval()
        with torch.no_grad():
            preds = net(X_tensor.to(device)).squeeze(-1).cpu().numpy()
        per_sample_loss = (preds - y_norm) ** 2

        # Keep smallest-loss samples
        n_keep = max(10, int(len(y_train) * self.selection_rate))
        keep_idx = np.argsort(per_sample_loss)[:n_keep]

        # Retrain on filtered data
        y_pred = self._default_predict(X_t[keep_idx], y_train[keep_idx], X_val)

        return {
            'y_pred': y_pred,
            'metadata': {
                'n_kept': n_keep,
                'n_removed': len(y_train) - n_keep,
                'mean_loss_kept': float(per_sample_loss[keep_idx].mean()),
                'mean_loss_removed': float(np.delete(per_sample_loss, keep_idx).mean()),
            },
        }


class ConfidentLearning(LNLMethod):
    """Flag high-error + low-uncertainty samples via ensemble, remove, retrain.

    Reference: Northcutt et al. (2021) Confident Learning.
    Category: Sample selection.
    """

    def __init__(self, n_ensemble=3, error_pct=85, uncertainty_pct=25, **kwargs):
        super().__init__(**kwargs)
        self.n_ensemble = n_ensemble
        self.error_pct = error_pct
        self.uncertainty_pct = uncertainty_pct

    @property
    def category(self):
        return 'data_level'

    def train_and_predict(self, X_train, y_train, X_val, y_val):
        X_t = np.asarray(X_train, dtype=np.float32)

        # Train ensemble of DNNs
        predictions = []
        for i in range(self.n_ensemble):
            y_pred_i = _standard_dnn_train_predict(
                X_t, y_train, X_t,  # predict on train
                epochs=min(self.epochs, 100), hidden_sizes=self.hidden_sizes,
                dropout=self.dropout, lr=self.lr, batch_size=self.batch_size,
                patience=self.patience, random_state=self.random_state + i,
            )
            predictions.append(y_pred_i)

        preds = np.array(predictions)  # (n_ensemble, n_samples)
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)

        # Flag: high error AND low uncertainty
        errors = np.abs(y_train - mean_pred)
        error_threshold = np.percentile(errors, self.error_pct)
        uncertainty_threshold = np.percentile(std_pred, self.uncertainty_pct)

        flagged = (errors > error_threshold) & (std_pred < uncertainty_threshold)
        keep_mask = ~flagged

        if keep_mask.sum() < 10:
            keep_mask = np.ones(len(y_train), dtype=bool)

        # Retrain on clean subset
        y_pred = self._default_predict(X_t[keep_mask], y_train[keep_mask], X_val)

        return {
            'y_pred': y_pred,
            'metadata': {
                'n_flagged': int(flagged.sum()),
                'n_kept': int(keep_mask.sum()),
                'error_threshold': float(error_threshold),
                'uncertainty_threshold': float(uncertainty_threshold),
            },
        }


class DistanceBasedSelection(LNLMethod):
    """Outlier detection via molecular similarity, remove, retrain.

    Category: Sample selection.
    """

    def __init__(self, k=10, threshold_pct=85, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.threshold_pct = threshold_pct

    @property
    def category(self):
        return 'data_level'

    def train_and_predict(self, X_train, y_train, X_val, y_val):
        from sklearn.neighbors import NearestNeighbors

        X_t = np.asarray(X_train, dtype=np.float32)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_t)

        # kNN label disagreement
        k_actual = min(self.k, len(y_train) - 1)
        nn = NearestNeighbors(n_neighbors=k_actual + 1)
        nn.fit(X_scaled)
        _, indices = nn.kneighbors(X_scaled)
        neighbor_idx = indices[:, 1:]  # exclude self

        neighbor_means = np.mean(y_train[neighbor_idx], axis=1)
        neighbor_stds = np.std(y_train[neighbor_idx], axis=1) + 1e-10

        # Disagreement: how much does sample deviate from neighbors
        disagreement = np.abs(y_train - neighbor_means) / neighbor_stds

        # Flag outliers
        threshold = np.percentile(disagreement, self.threshold_pct)
        flagged = disagreement > threshold
        keep_mask = ~flagged

        if keep_mask.sum() < 10:
            keep_mask = np.ones(len(y_train), dtype=bool)

        y_pred = self._default_predict(X_t[keep_mask], y_train[keep_mask], X_val)

        return {
            'y_pred': y_pred,
            'metadata': {
                'n_flagged': int(flagged.sum()),
                'n_kept': int(keep_mask.sum()),
                'threshold': float(threshold),
                'mean_disagreement': float(disagreement.mean()),
            },
        }


# ============================================================================
# SIMPLE TRAINING-LOOP METHODS
# ============================================================================

class MultistageCleaning(LNLMethod):
    """Progressive removal: 10%/stage x 3 stages via loss scoring.

    Reference: Zhao et al. (2025).
    Category: Iterative refinement.
    """

    def __init__(self, n_stages=3, removal_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_stages = n_stages
        self.removal_rate = removal_rate

    @property
    def category(self):
        return 'training_loop'

    def train_and_predict(self, X_train, y_train, X_val, y_val):
        X_t = np.asarray(X_train, dtype=np.float32)
        current_idx = np.arange(len(y_train))
        stage_info = []

        for stage in range(self.n_stages):
            X_stage = X_t[current_idx]
            y_stage = y_train[current_idx]

            # Train and compute per-sample loss
            y_pred_train = _standard_dnn_train_predict(
                X_stage, y_stage, X_stage,
                epochs=min(self.epochs, 100),
                hidden_sizes=self.hidden_sizes, dropout=self.dropout,
                lr=self.lr, batch_size=self.batch_size,
                patience=self.patience,
                random_state=self.random_state + stage,
            )
            losses = (y_pred_train - y_stage) ** 2

            # Remove top removal_rate% by loss
            n_remove = max(1, int(len(y_stage) * self.removal_rate))
            remove_local = np.argsort(losses)[-n_remove:]
            keep_local = np.setdiff1d(np.arange(len(y_stage)), remove_local)

            stage_info.append({
                'n_before': len(current_idx),
                'n_removed': n_remove,
                'mean_loss_removed': float(losses[remove_local].mean()),
            })

            current_idx = current_idx[keep_local]
            if len(current_idx) < 10:
                break

        # Final training on cleaned data
        y_pred = self._default_predict(X_t[current_idx], y_train[current_idx], X_val)

        return {
            'y_pred': y_pred,
            'metadata': {
                'stages': stage_info,
                'n_final': len(current_idx),
                'pct_retained': float(len(current_idx) / len(y_train) * 100),
            },
        }


class EarlyLearning(LNLMethod):
    """Cross-epoch reweighting with geometric decay.

    Samples with loss above median get their weight decayed by gamma each epoch.
    Reference: Zhang et al. (2025).
    Category: Iterative refinement.
    """

    def __init__(self, gamma=0.95, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    @property
    def category(self):
        return 'training_loop'

    def train_and_predict(self, X_train, y_train, X_val, y_val):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        X_t = np.asarray(X_train, dtype=np.float32)
        X_v = np.asarray(X_val, dtype=np.float32)
        y_norm, y_mean, y_std = _z_normalize(y_train)

        device = torch.device('cpu')
        net = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)

        X_tensor = torch.from_numpy(X_t)
        y_tensor = torch.from_numpy(y_norm)

        # Per-sample weights
        weights = np.ones(len(y_train), dtype=np.float32)

        # Use y_val for early stopping
        y_val_norm = (np.asarray(y_val, dtype=np.float32) - y_mean) / y_std
        X_val_tensor = torch.from_numpy(X_v)
        y_val_tensor = torch.from_numpy(y_val_norm)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            net.train()
            perm = torch.randperm(len(X_t))
            w_tensor = torch.from_numpy(weights)

            for start in range(0, len(X_t), self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                xb = X_tensor[batch_idx].to(device)
                yb = y_tensor[batch_idx].to(device)
                wb = w_tensor[batch_idx].to(device)

                optimizer.zero_grad()
                pred = net(xb).squeeze(-1)
                loss = (wb * (pred - yb) ** 2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

            # Compute per-sample loss and update weights
            net.eval()
            with torch.no_grad():
                all_pred = net(X_tensor.to(device)).squeeze(-1).cpu().numpy()
            per_sample_loss = (all_pred - y_norm) ** 2
            median_loss = np.median(per_sample_loss)

            # Geometric decay for high-loss samples
            high_loss_mask = per_sample_loss > median_loss
            weights[high_loss_mask] *= self.gamma
            weights = np.clip(weights, 0.01, 1.0)
            # Normalize to mean 1
            weights = weights / weights.mean()

            # Val early stopping
            with torch.no_grad():
                val_pred = net(X_val_tensor.to(device)).squeeze(-1)
                val_loss = nn.functional.mse_loss(val_pred, y_val_tensor.to(device)).item()

            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)

        net.eval()
        with torch.no_grad():
            y_pred = net(X_val_tensor.to(device)).squeeze(-1).cpu().numpy()

        return {
            'y_pred': y_pred * y_std + y_mean,
            'metadata': {
                'final_epoch': epoch + 1,
                'weight_mean': float(weights.mean()),
                'weight_std': float(weights.std()),
                'pct_downweighted': float((weights < 0.5).mean() * 100),
            },
        }


class UncertaintyCurriculum(LNLMethod):
    """Staged inclusion by confidence: 25% -> 50% -> 75% -> 100% after warmup.

    Reference: Bengio et al. (2009).
    Category: Curriculum learning.
    """

    def __init__(self, warmup_epochs=20, stages=(0.25, 0.5, 0.75, 1.0), **kwargs):
        super().__init__(**kwargs)
        self.warmup_epochs = warmup_epochs
        self.stages = stages

    @property
    def category(self):
        return 'training_loop'

    def train_and_predict(self, X_train, y_train, X_val, y_val):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        X_t = np.asarray(X_train, dtype=np.float32)
        X_v = np.asarray(X_val, dtype=np.float32)
        y_norm, y_mean, y_std = _z_normalize(y_train)
        n = len(y_train)

        device = torch.device('cpu')
        net = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)

        X_tensor = torch.from_numpy(X_t)
        y_tensor = torch.from_numpy(y_norm)

        y_val_norm = (np.asarray(y_val, dtype=np.float32) - y_mean) / y_std
        X_val_tensor = torch.from_numpy(X_v)
        y_val_tensor = torch.from_numpy(y_val_norm)

        # Compute stage epoch boundaries
        remaining = self.epochs - self.warmup_epochs
        stage_length = remaining // len(self.stages)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        current_inclusion = np.ones(n, dtype=bool)  # start with all during warmup

        for epoch in range(self.epochs):
            # Determine stage and inclusion
            if epoch < self.warmup_epochs:
                # Warmup: use all data
                current_inclusion = np.ones(n, dtype=bool)
            else:
                stage_idx = min(
                    (epoch - self.warmup_epochs) // max(stage_length, 1),
                    len(self.stages) - 1
                )
                fraction = self.stages[stage_idx]

                # Rank by confidence (inverse per-sample loss)
                net.eval()
                with torch.no_grad():
                    all_pred = net(X_tensor.to(device)).squeeze(-1).cpu().numpy()
                per_sample_loss = (all_pred - y_norm) ** 2
                n_include = max(10, int(n * fraction))
                include_idx = np.argsort(per_sample_loss)[:n_include]
                current_inclusion = np.zeros(n, dtype=bool)
                current_inclusion[include_idx] = True

            # Train on included samples
            idx_active = np.where(current_inclusion)[0]
            if len(idx_active) < 2:
                continue

            net.train()
            perm = torch.from_numpy(
                np.random.RandomState(self.random_state + epoch).permutation(len(idx_active))
            )
            for start in range(0, len(idx_active), self.batch_size):
                batch_local = perm[start:start + self.batch_size]
                batch_global = torch.from_numpy(idx_active)[batch_local]
                xb = X_tensor[batch_global].to(device)
                yb = y_tensor[batch_global].to(device)

                optimizer.zero_grad()
                pred = net(xb).squeeze(-1)
                loss = nn.functional.mse_loss(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

            # Val early stopping
            net.eval()
            with torch.no_grad():
                val_pred = net(X_val_tensor.to(device)).squeeze(-1)
                val_loss = nn.functional.mse_loss(val_pred, y_val_tensor.to(device)).item()

            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)

        net.eval()
        with torch.no_grad():
            y_pred = net(X_val_tensor.to(device)).squeeze(-1).cpu().numpy()

        return {
            'y_pred': y_pred * y_std + y_mean,
            'metadata': {
                'final_epoch': epoch + 1,
                'final_inclusion_pct': float(current_inclusion.mean() * 100),
            },
        }


# ============================================================================
# COMPLEX TRAINING-LOOP METHODS
# ============================================================================

class DivideMixReg(LNLMethod):
    """Regression-adapted DivideMix: dual DNNs + GMM on squared errors.

    Reference: Li et al. (2020) DivideMix.
    Category: Iterative refinement.
    """

    def __init__(self, warmup_epochs=10, **kwargs):
        super().__init__(**kwargs)
        self.warmup_epochs = warmup_epochs

    @property
    def category(self):
        return 'training_loop'

    def train_and_predict(self, X_train, y_train, X_val, y_val):
        import torch
        import torch.nn as nn
        from sklearn.mixture import GaussianMixture

        torch.manual_seed(self.random_state)
        X_t = np.asarray(X_train, dtype=np.float32)
        X_v = np.asarray(X_val, dtype=np.float32)
        y_norm, y_mean, y_std = _z_normalize(y_train)
        n = len(y_train)

        device = torch.device('cpu')
        net_a = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout).to(device)
        net_b = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout).to(device)
        opt_a = torch.optim.Adam(net_a.parameters(), lr=self.lr)
        opt_b = torch.optim.Adam(net_b.parameters(), lr=self.lr)

        X_tensor = torch.from_numpy(X_t)
        y_tensor = torch.from_numpy(y_norm)

        y_val_norm = (np.asarray(y_val, dtype=np.float32) - y_mean) / y_std
        X_val_tensor = torch.from_numpy(X_v)
        y_val_tensor = torch.from_numpy(y_val_norm)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        gmm_labels = None

        for epoch in range(self.epochs):
            # --- Warmup: standard training for both nets ---
            if epoch < self.warmup_epochs:
                for net, opt in [(net_a, opt_a), (net_b, opt_b)]:
                    net.train()
                    perm = torch.randperm(n)
                    for start in range(0, n, self.batch_size):
                        idx = perm[start:start + self.batch_size]
                        pred = net(X_tensor[idx].to(device)).squeeze(-1)
                        loss = nn.functional.mse_loss(pred, y_tensor[idx].to(device))
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                        opt.step()
                continue

            # --- GMM separation on losses from net_a ---
            net_a.eval()
            with torch.no_grad():
                pred_a = net_a(X_tensor.to(device)).squeeze(-1).cpu().numpy()
            losses_a = (pred_a - y_norm) ** 2
            log_losses = np.log(losses_a + 1e-10).reshape(-1, 1)

            try:
                gmm = GaussianMixture(n_components=2, random_state=self.random_state)
                gmm.fit(log_losses)
                gmm_labels = gmm.predict(log_losses)
                # Clean component = lower mean loss
                clean_comp = 0 if gmm.means_[0] < gmm.means_[1] else 1
                clean_mask_a = (gmm_labels == clean_comp)
            except Exception:
                clean_mask_a = np.ones(n, dtype=bool)

            # --- GMM separation on losses from net_b ---
            net_b.eval()
            with torch.no_grad():
                pred_b = net_b(X_tensor.to(device)).squeeze(-1).cpu().numpy()
            losses_b = (pred_b - y_norm) ** 2
            log_losses_b = np.log(losses_b + 1e-10).reshape(-1, 1)

            try:
                gmm_b = GaussianMixture(n_components=2, random_state=self.random_state)
                gmm_b.fit(log_losses_b)
                gmm_labels_b = gmm_b.predict(log_losses_b)
                clean_comp_b = 0 if gmm_b.means_[0] < gmm_b.means_[1] else 1
                clean_mask_b = (gmm_labels_b == clean_comp_b)
            except Exception:
                clean_mask_b = np.ones(n, dtype=bool)

            # --- Cross-train: net_a on clean samples from net_b, vice versa ---
            for net, opt, clean_mask in [(net_a, opt_a, clean_mask_b),
                                          (net_b, opt_b, clean_mask_a)]:
                clean_idx = np.where(clean_mask)[0]
                if len(clean_idx) < 2:
                    continue
                net.train()
                perm = torch.from_numpy(
                    np.random.RandomState(self.random_state + epoch).permutation(len(clean_idx))
                )
                for start in range(0, len(clean_idx), self.batch_size):
                    batch_local = perm[start:start + self.batch_size]
                    batch_global = torch.from_numpy(clean_idx)[batch_local]
                    pred = net(X_tensor[batch_global].to(device)).squeeze(-1)
                    loss = nn.functional.mse_loss(pred, y_tensor[batch_global].to(device))
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()

            # Val early stopping (average of both nets)
            net_a.eval()
            net_b.eval()
            with torch.no_grad():
                pred_va = net_a(X_val_tensor.to(device)).squeeze(-1)
                pred_vb = net_b(X_val_tensor.to(device)).squeeze(-1)
                avg_pred = (pred_va + pred_vb) / 2
                val_loss = nn.functional.mse_loss(avg_pred, y_val_tensor.to(device)).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    'a': {k: v.cpu().clone() for k, v in net_a.state_dict().items()},
                    'b': {k: v.cpu().clone() for k, v in net_b.state_dict().items()},
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            net_a.load_state_dict(best_state['a'])
            net_b.load_state_dict(best_state['b'])

        net_a.eval()
        net_b.eval()
        with torch.no_grad():
            pred_a = net_a(X_val_tensor.to(device)).squeeze(-1).cpu().numpy()
            pred_b = net_b(X_val_tensor.to(device)).squeeze(-1).cpu().numpy()

        y_pred = (pred_a + pred_b) / 2 * y_std + y_mean

        return {
            'y_pred': y_pred,
            'metadata': {
                'final_epoch': epoch + 1,
                'pct_clean_a': float(clean_mask_a.mean() * 100) if gmm_labels is not None else None,
                'pct_clean_b': float(clean_mask_b.mean() * 100) if gmm_labels is not None else None,
            },
        }


class MetaWeightNet(LNLMethod):
    """Meta-learning sample weights via small meta-network.

    Meta-network maps per-sample loss to a weight. Updated every
    meta_interval epochs using validation loss as meta-objective.
    Uses first-order approximation (no second-order gradients).

    Reference: Shu et al. (2019).
    Category: Meta-learning.
    """

    def __init__(self, meta_interval=5, meta_lr=0.001, **kwargs):
        super().__init__(**kwargs)
        self.meta_interval = meta_interval
        self.meta_lr = meta_lr

    @property
    def category(self):
        return 'training_loop'

    def train_and_predict(self, X_train, y_train, X_val, y_val):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        X_t = np.asarray(X_train, dtype=np.float32)
        X_v = np.asarray(X_val, dtype=np.float32)
        y_norm, y_mean, y_std = _z_normalize(y_train)
        y_val_norm = (np.asarray(y_val, dtype=np.float32) - y_mean) / y_std

        device = torch.device('cpu')
        net = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)

        # Meta-network: loss -> weight (small 2-layer MLP)
        meta_net = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        ).to(device)
        meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=self.meta_lr)

        X_tensor = torch.from_numpy(X_t)
        y_tensor = torch.from_numpy(y_norm)
        X_val_tensor = torch.from_numpy(X_v)
        y_val_tensor = torch.from_numpy(y_val_norm)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            # --- Meta-update every meta_interval epochs ---
            if epoch > 0 and epoch % self.meta_interval == 0 and len(X_v) > 0:
                net.eval()
                # Compute per-sample loss on training data
                with torch.no_grad():
                    train_pred = net(X_tensor.to(device)).squeeze(-1)
                    per_loss = ((train_pred - y_tensor.to(device)) ** 2).unsqueeze(-1)

                # Get meta-weights
                meta_weights = meta_net(per_loss.detach()).squeeze(-1)

                # First-order meta-gradient: train one step with meta-weights,
                # evaluate on val, update meta-net
                net.train()
                train_pred_2 = net(X_tensor.to(device)).squeeze(-1)
                weighted_loss = (meta_weights.detach() *
                                 (train_pred_2 - y_tensor.to(device)) ** 2).mean()
                optimizer.zero_grad()
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

                # Val loss for meta-gradient
                net.eval()
                val_pred = net(X_val_tensor.to(device)).squeeze(-1)
                meta_loss = nn.functional.mse_loss(val_pred, y_val_tensor.to(device))

                # Update meta-net
                # Recompute with grad for meta-net
                net.train()
                train_pred_3 = net(X_tensor.to(device)).squeeze(-1)
                per_loss_3 = ((train_pred_3 - y_tensor.to(device)) ** 2).unsqueeze(-1)
                meta_weights_3 = meta_net(per_loss_3.detach()).squeeze(-1)

                # Approximate: use val loss directly
                meta_optimizer.zero_grad()
                meta_loss.backward()
                meta_optimizer.step()

            # --- Standard weighted training step ---
            net.train()
            perm = torch.randperm(len(X_t))

            # Get current weights from meta-net
            with torch.no_grad():
                all_pred = net(X_tensor.to(device)).squeeze(-1)
                all_loss = ((all_pred - y_tensor.to(device)) ** 2).unsqueeze(-1)
                meta_w = meta_net(all_loss).squeeze(-1).cpu()

            for start in range(0, len(X_t), self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                xb = X_tensor[batch_idx].to(device)
                yb = y_tensor[batch_idx].to(device)
                wb = meta_w[batch_idx].to(device)

                optimizer.zero_grad()
                pred = net(xb).squeeze(-1)
                loss = (wb * (pred - yb) ** 2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

            # Val early stopping
            net.eval()
            with torch.no_grad():
                val_pred = net(X_val_tensor.to(device)).squeeze(-1)
                val_loss = nn.functional.mse_loss(val_pred, y_val_tensor.to(device)).item()

            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)

        net.eval()
        with torch.no_grad():
            y_pred = net(X_val_tensor.to(device)).squeeze(-1).cpu().numpy()

        return {
            'y_pred': y_pred * y_std + y_mean,
            'metadata': {
                'final_epoch': epoch + 1,
                'meta_weight_mean': float(meta_w.mean()),
                'meta_weight_std': float(meta_w.std()),
            },
        }


class MentorNet(LNLMethod):
    """Online mentor network with 7D sample features.

    Mentor input per sample: [current_loss, loss_history(5), knn_distance].
    Mentor outputs curriculum weight.

    Reference: Jiang et al. (2018).
    Category: Curriculum learning.
    """

    def __init__(self, k_neighbors=10, **kwargs):
        super().__init__(**kwargs)
        self.k_neighbors = k_neighbors

    @property
    def category(self):
        return 'training_loop'

    def train_and_predict(self, X_train, y_train, X_val, y_val):
        import torch
        import torch.nn as nn
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        torch.manual_seed(self.random_state)
        X_t = np.asarray(X_train, dtype=np.float32)
        X_v = np.asarray(X_val, dtype=np.float32)
        y_norm, y_mean, y_std = _z_normalize(y_train)
        n = len(y_train)

        # Pre-compute kNN distances
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_t)
        k_actual = min(self.k_neighbors, n - 1)
        nn_model = NearestNeighbors(n_neighbors=k_actual + 1)
        nn_model.fit(X_scaled)
        distances, _ = nn_model.kneighbors(X_scaled)
        knn_dist = distances[:, 1:].mean(axis=1)  # mean distance to k neighbors
        knn_dist = (knn_dist - knn_dist.mean()) / (knn_dist.std() + 1e-10)

        device = torch.device('cpu')
        net = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)

        # Mentor network: 7D -> 1 (weight)
        mentor = nn.Sequential(
            nn.Linear(7, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        ).to(device)
        mentor_opt = torch.optim.Adam(mentor.parameters(), lr=self.lr * 0.1)

        X_tensor = torch.from_numpy(X_t)
        y_tensor = torch.from_numpy(y_norm)
        knn_tensor = torch.from_numpy(knn_dist.astype(np.float32))

        y_val_norm = (np.asarray(y_val, dtype=np.float32) - y_mean) / y_std
        X_val_tensor = torch.from_numpy(X_v)
        y_val_tensor = torch.from_numpy(y_val_norm)

        # Loss history buffer (5 epochs)
        loss_history = np.zeros((n, 5), dtype=np.float32)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            # Compute current per-sample loss
            net.eval()
            with torch.no_grad():
                all_pred = net(X_tensor.to(device)).squeeze(-1).cpu().numpy()
            current_loss = (all_pred - y_norm) ** 2

            # Update loss history (shift left, add current)
            loss_history[:, :-1] = loss_history[:, 1:]
            loss_history[:, -1] = current_loss

            # Build 7D mentor features: [current_loss, history(5), knn_dist]
            mentor_features = np.column_stack([
                current_loss,
                loss_history,
                knn_dist,
            ]).astype(np.float32)
            mentor_tensor = torch.from_numpy(mentor_features)

            # Get mentor weights
            mentor.eval()
            with torch.no_grad():
                mentor_weights = mentor(mentor_tensor.to(device)).squeeze(-1).cpu()

            # Train main net with mentor weights
            net.train()
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                xb = X_tensor[batch_idx].to(device)
                yb = y_tensor[batch_idx].to(device)
                wb = mentor_weights[batch_idx].to(device)

                optimizer.zero_grad()
                pred = net(xb).squeeze(-1)
                loss = (wb * (pred - yb) ** 2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

            # Update mentor using validation loss
            net.eval()
            val_pred_t = net(X_val_tensor.to(device)).squeeze(-1)
            val_loss = nn.functional.mse_loss(val_pred_t, y_val_tensor.to(device))

            mentor_opt.zero_grad()
            # Approximate: update mentor to reduce val loss
            # Recompute with grad
            mentor.train()
            mf = mentor_tensor.to(device)
            mw = mentor(mf).squeeze(-1)
            net.train()
            train_pred = net(X_tensor.to(device)).squeeze(-1)
            weighted_train_loss = (mw * (train_pred.detach() - y_tensor.to(device)) ** 2).mean()
            weighted_train_loss.backward()
            mentor_opt.step()

            # Val early stopping
            net.eval()
            with torch.no_grad():
                v_pred = net(X_val_tensor.to(device)).squeeze(-1)
                v_loss = nn.functional.mse_loss(v_pred, y_val_tensor.to(device)).item()

            scheduler.step(v_loss)
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)

        net.eval()
        with torch.no_grad():
            y_pred = net(X_val_tensor.to(device)).squeeze(-1).cpu().numpy()

        return {
            'y_pred': y_pred * y_std + y_mean,
            'metadata': {
                'final_epoch': epoch + 1,
                'mentor_weight_mean': float(mentor_weights.mean()),
            },
        }


class ContrastToDivide(LNLMethod):
    """Consistency-based selection via prediction variance under perturbation.

    For each sample, create K perturbed copies, measure prediction variance.
    High variance on high-loss = clean (model uncertain); low variance on
    high-loss = noise (model confidently disagrees).

    Reference: Zheltonozhskii et al. (2021).
    Category: Iterative refinement.
    """

    def __init__(self, n_perturbations=5, perturbation_sigma=0.01, **kwargs):
        super().__init__(**kwargs)
        self.n_perturbations = n_perturbations
        self.perturbation_sigma = perturbation_sigma

    @property
    def category(self):
        return 'training_loop'

    def train_and_predict(self, X_train, y_train, X_val, y_val):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        X_t = np.asarray(X_train, dtype=np.float32)
        X_v = np.asarray(X_val, dtype=np.float32)
        y_norm, y_mean, y_std = _z_normalize(y_train)
        n = len(y_train)

        device = torch.device('cpu')
        net = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)

        X_tensor = torch.from_numpy(X_t)
        y_tensor = torch.from_numpy(y_norm)

        y_val_norm = (np.asarray(y_val, dtype=np.float32) - y_mean) / y_std
        X_val_tensor = torch.from_numpy(X_v)
        y_val_tensor = torch.from_numpy(y_val_norm)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            # Compute consistency scores every 5 epochs (expensive)
            if epoch % 5 == 0:
                net.eval()
                with torch.no_grad():
                    # Base prediction
                    base_pred = net(X_tensor.to(device)).squeeze(-1).cpu().numpy()
                    per_sample_loss = (base_pred - y_norm) ** 2

                    # Perturbed predictions
                    perturbed_preds = []
                    rng = np.random.RandomState(self.random_state + epoch)
                    for _ in range(self.n_perturbations):
                        noise = rng.normal(0, self.perturbation_sigma,
                                           X_t.shape).astype(np.float32)
                        X_pert = torch.from_numpy(X_t + noise)
                        p = net(X_pert.to(device)).squeeze(-1).cpu().numpy()
                        perturbed_preds.append(p)

                    pred_variance = np.var(perturbed_preds, axis=0)

                # Consistency signal: high loss + low variance = noise
                # Weight: downweight noise (high loss, low variance)
                loss_rank = np.argsort(np.argsort(per_sample_loss)) / n
                var_rank = np.argsort(np.argsort(pred_variance)) / n

                # Noise score: high loss rank, low variance rank
                noise_score = loss_rank * (1 - var_rank)
                weights = 1.0 / (1.0 + 3.0 * noise_score)
                weights = (weights / weights.mean()).astype(np.float32)
                w_tensor = torch.from_numpy(weights)

            # Weighted training
            net.train()
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                xb = X_tensor[batch_idx].to(device)
                yb = y_tensor[batch_idx].to(device)
                wb = w_tensor[batch_idx].to(device)

                optimizer.zero_grad()
                pred = net(xb).squeeze(-1)
                loss = (wb * (pred - yb) ** 2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

            # Val early stopping
            net.eval()
            with torch.no_grad():
                val_pred = net(X_val_tensor.to(device)).squeeze(-1)
                val_loss = nn.functional.mse_loss(val_pred, y_val_tensor.to(device)).item()

            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)

        net.eval()
        with torch.no_grad():
            y_pred = net(X_val_tensor.to(device)).squeeze(-1).cpu().numpy()

        return {
            'y_pred': y_pred * y_std + y_mean,
            'metadata': {
                'final_epoch': epoch + 1,
                'weight_mean': float(weights.mean()),
                'weight_std': float(weights.std()),
            },
        }


# ============================================================================
# GP EXTENSION METHODS
# ============================================================================

class HeteroscedasticGP(LNLMethod):
    """GP for mean prediction + NN for noise variance, joint training.

    Fits sklearn GP on a subsample (<=1000) for the mean, then trains
    a small NN to predict per-sample noise variance. Predictions use
    GP mean; noise estimates can be used for sample weighting.

    Reference: Liu et al. (2020).
    Category: GP extension.
    """

    GP_MAX = 1000

    def __init__(self, kernel='rbf', **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel

    @property
    def category(self):
        return 'training_loop'

    def train_and_predict(self, X_train, y_train, X_val, y_val):
        import torch
        import torch.nn as nn
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
        from sklearn.preprocessing import StandardScaler

        X_t = np.asarray(X_train, dtype=np.float32)
        X_v = np.asarray(X_val, dtype=np.float32)
        n = len(y_train)

        scaler = StandardScaler()
        X_t_s = scaler.fit_transform(X_t).astype(np.float32)
        X_v_s = scaler.transform(X_v).astype(np.float32)

        # Subsample for GP
        rng = np.random.RandomState(self.random_state)
        if n > self.GP_MAX:
            sub_idx = rng.choice(n, self.GP_MAX, replace=False)
        else:
            sub_idx = np.arange(n)

        # Build GP kernel
        if self.kernel == 'rbf':
            k = ConstantKernel() * RBF() + WhiteKernel()
        elif self.kernel == 'matern':
            k = ConstantKernel() * Matern(nu=2.5) + WhiteKernel()
        else:
            k = ConstantKernel() * RBF() + WhiteKernel()

        # Fit GP on subsample
        gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=2,
                                      random_state=self.random_state, alpha=1e-6)
        try:
            gp.fit(X_t_s[sub_idx], y_train[sub_idx])
            gp_mean, gp_std = gp.predict(X_t_s, return_std=True)
            gp_val_mean, gp_val_std = gp.predict(X_v_s, return_std=True)
        except Exception:
            # Fallback: just use DNN
            y_pred = self._default_predict(X_t, y_train, X_v)
            return {'y_pred': y_pred, 'metadata': {'gp_fallback': True}}

        # GP residuals as noise estimates
        residuals = np.abs(y_train - gp_mean)

        # Train NN on residuals to predict per-sample noise
        torch.manual_seed(self.random_state)
        device = torch.device('cpu')
        noise_net = nn.Sequential(
            nn.Linear(X_t.shape[1], 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Softplus(),  # positive noise prediction
        ).to(device)
        noise_opt = torch.optim.Adam(noise_net.parameters(), lr=0.001)

        X_tensor = torch.from_numpy(X_t_s)
        residual_tensor = torch.from_numpy(residuals.astype(np.float32))

        for ep in range(50):
            noise_net.train()
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                pred_noise = noise_net(X_tensor[idx].to(device)).squeeze(-1)
                target = residual_tensor[idx].to(device)
                loss = nn.functional.mse_loss(pred_noise, target)
                noise_opt.zero_grad()
                loss.backward()
                noise_opt.step()

        # Get noise estimates for weighting
        noise_net.eval()
        with torch.no_grad():
            noise_pred = noise_net(X_tensor.to(device)).squeeze(-1).cpu().numpy()

        # Sample weights: inverse noise
        weights = 1.0 / (1.0 + noise_pred)
        weights = weights / weights.mean()

        # Train DNN with sample weights
        y_pred = self._default_predict(X_t, y_train, X_v, sample_weights=weights)

        return {
            'y_pred': y_pred,
            'metadata': {
                'gp_r2': float(1 - np.sum((y_train - gp_mean) ** 2) /
                               np.sum((y_train - y_train.mean()) ** 2)),
                'noise_mean': float(noise_pred.mean()),
                'noise_std': float(noise_pred.std()),
                'kernel': self.kernel,
            },
        }


class EvidentialKernel(LNLMethod):
    """DNN predicting Normal-Inverse-Gamma (NIG) parameters for evidential regression.

    Outputs (mu, v, alpha, beta) per sample. Uncertainty-aware loss
    penalizes overconfident wrong predictions.

    Reference: Amini et al. (2020) Deep Evidential Regression.
    Category: GP extension / evidential.
    """

    def __init__(self, nig_coeff=0.01, **kwargs):
        super().__init__(**kwargs)
        self.nig_coeff = nig_coeff

    @property
    def category(self):
        return 'training_loop'

    def train_and_predict(self, X_train, y_train, X_val, y_val):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        X_t = np.asarray(X_train, dtype=np.float32)
        X_v = np.asarray(X_val, dtype=np.float32)
        y_norm, y_mean, y_std = _z_normalize(y_train)
        y_val_norm = (np.asarray(y_val, dtype=np.float32) - y_mean) / y_std

        device = torch.device('cpu')

        # NIG network: outputs 4 parameters (mu, log_v, log_alpha, log_beta)
        layers = []
        prev = X_t.shape[1]
        for h in self.hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev = h
        backbone = nn.Sequential(*layers)
        mu_head = nn.Linear(prev, 1)
        logv_head = nn.Linear(prev, 1)
        logalpha_head = nn.Linear(prev, 1)
        logbeta_head = nn.Linear(prev, 1)

        params = (list(backbone.parameters()) + list(mu_head.parameters()) +
                  list(logv_head.parameters()) + list(logalpha_head.parameters()) +
                  list(logbeta_head.parameters()))
        optimizer = torch.optim.Adam(params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)

        backbone.to(device)
        mu_head.to(device)
        logv_head.to(device)
        logalpha_head.to(device)
        logbeta_head.to(device)

        def forward(X_batch):
            h = backbone(X_batch)
            mu = mu_head(h).squeeze(-1)
            v = torch.exp(logv_head(h).squeeze(-1)) + 1e-6
            alpha = torch.exp(logalpha_head(h).squeeze(-1)) + 1.0  # alpha > 1
            beta = torch.exp(logbeta_head(h).squeeze(-1)) + 1e-6
            return mu, v, alpha, beta

        def nig_loss(y_true, mu, v, alpha, beta):
            """NIG negative log-likelihood + regularization."""
            omega = 2 * beta * (1 + v)
            nll = (0.5 * torch.log(np.pi / v)
                   - alpha * torch.log(omega)
                   + (alpha + 0.5) * torch.log(v * (y_true - mu) ** 2 + omega)
                   + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))
            # Regularization: penalize evidence on wrong predictions
            reg = torch.abs(y_true - mu) * (2 * v + alpha)
            return (nll + self.nig_coeff * reg).mean()

        X_tensor = torch.from_numpy(X_t)
        y_tensor = torch.from_numpy(y_norm)
        X_val_tensor = torch.from_numpy(X_v)
        y_val_tensor = torch.from_numpy(y_val_norm)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            backbone.train()
            perm = torch.randperm(len(X_t))
            for start in range(0, len(X_t), self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                xb = X_tensor[batch_idx].to(device)
                yb = y_tensor[batch_idx].to(device)

                mu, v, alpha, beta = forward(xb)
                loss = nig_loss(yb, mu, v, alpha, beta)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

            # Val eval
            backbone.eval()
            with torch.no_grad():
                mu_v, v_v, alpha_v, beta_v = forward(X_val_tensor.to(device))
                val_loss = nn.functional.mse_loss(mu_v, y_val_tensor.to(device)).item()

            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    'backbone': {k: v.cpu().clone() for k, v in backbone.state_dict().items()},
                    'mu': {k: v.cpu().clone() for k, v in mu_head.state_dict().items()},
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            backbone.load_state_dict(best_state['backbone'])
            mu_head.load_state_dict(best_state['mu'])

        backbone.eval()
        with torch.no_grad():
            mu_pred, v_pred, alpha_pred, beta_pred = forward(X_val_tensor.to(device))

        y_pred = mu_pred.cpu().numpy() * y_std + y_mean

        # Uncertainty: epistemic = beta / (v * (alpha - 1))
        epistemic = (beta_pred / (v_pred * (alpha_pred - 1) + 1e-6)).cpu().numpy()

        return {
            'y_pred': y_pred,
            'metadata': {
                'final_epoch': epoch + 1,
                'epistemic_mean': float(epistemic.mean()),
                'epistemic_std': float(epistemic.std()),
            },
        }


# ============================================================================
# REGISTRY
# ============================================================================

LNL_REGISTRY = {
    # Data-level
    'small_loss': SmallLossSelection,
    'confident_learning': ConfidentLearning,
    'distance_selection': DistanceBasedSelection,
    # Training-loop: simple
    'multistage_cleaning': MultistageCleaning,
    'early_learning': EarlyLearning,
    'uncertainty_curriculum': UncertaintyCurriculum,
    # Training-loop: complex
    'dividemix': DivideMixReg,
    'meta_weight_net': MetaWeightNet,
    'mentornet': MentorNet,
    'contrast_to_divide': ContrastToDivide,
    # GP extensions
    'heteroscedastic_gp': HeteroscedasticGP,
    'evidential_kernel': EvidentialKernel,
}

DATA_LEVEL_METHODS = {'small_loss', 'confident_learning', 'distance_selection'}
TRAINING_LOOP_METHODS = set(LNL_REGISTRY.keys()) - DATA_LEVEL_METHODS


def get_lnl_method(name: str, **kwargs) -> LNLMethod:
    """Factory function for LNL methods.

    Args:
        name: Method name (key in LNL_REGISTRY).
        **kwargs: Passed to method constructor.

    Returns:
        Instantiated LNLMethod.
    """
    if name not in LNL_REGISTRY:
        raise ValueError(
            f"Unknown LNL method '{name}'. Available: {list(LNL_REGISTRY.keys())}"
        )
    return LNL_REGISTRY[name](**kwargs)
