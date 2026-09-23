"""Sklearn-compatible MC-Dropout Bayesian Neural Network.

MC-Dropout BNN (Gal & Ghahramani 2016): standard DNN with dropout kept ON
at inference time. Multiple forward passes give a distribution over
predictions; mean = point estimate, std = epistemic uncertainty.

Critical design constraint: __init__ must NOT instantiate any PyTorch objects.
All torch objects are created in fit() to ensure deepcopy() works (required by
_evaluate_single in find_best_hybrid.py).
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


def _cap_torch_threads():
    """Hold torch to KIRBY_TORCH_THREADS cores, if it is set.

    torch takes every core by default. s11's BO loop refits this network once
    per iteration, so several concurrent runs each open a full-core thread pool
    per fit and the process is killed outright -- no traceback, because it is
    the kernel doing it, which is why nine s11 attempts over 2026-09-22 and
    09-23 died inside the hybrid_bnn condition with empty logs. Same shape as
    the lightgbm segfault that KIRBY_LGBM_THREADS=1 fixes.

    Unset leaves torch alone, so nothing on the cluster changes.
    """
    import os
    _t = os.environ.get('KIRBY_TORCH_THREADS')
    if not _t:
        return
    try:
        import torch
        torch.set_num_threads(max(1, int(_t)))
    except Exception:
        pass


class BNNRegressor(BaseEstimator, RegressorMixin):
    """MC-Dropout BNN regressor with sklearn interface.

    Trains identically to DNNRegressor. At predict time, runs mc_samples
    forward passes with dropout enabled and returns the mean.
    predict(X, return_std=True) returns (mean, std) for uncertainty.
    """

    def __init__(self, hidden_sizes=(128, 64), dropout=0.1, lr=0.001,
                 epochs=200, batch_size=256, patience=20, mc_samples=20,
                 random_state=42):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.mc_samples = mc_samples
        self.random_state = random_state

    def _build_net(self, input_dim):
        import torch.nn as nn
        layers = []
        prev = input_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y, groups=None):
        import torch
        import torch.nn as nn
        from ._es_split import early_stopping_split

        _cap_torch_threads()
        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()

        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) if y.std() > 1e-8 else 1.0
        y_norm = (y - self.y_mean_) / self.y_std_

        # Group-disjoint early-stopping holdout when groups are given.
        train_idx, ho_idx = early_stopping_split(
            y_norm, groups, rng, 'BNNRegressor')

        X_train = torch.from_numpy(X[train_idx])
        y_train = torch.from_numpy(y_norm[train_idx])
        X_ho = torch.from_numpy(X[ho_idx])
        y_ho = torch.from_numpy(y_norm[ho_idx])

        device = torch.device('cpu')
        net = self._build_net(X.shape[1]).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()

        best_ho_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            net.train()
            perm = torch.randperm(len(X_train))

            for start in range(0, len(X_train), self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                xb = X_train[batch_idx].to(device)
                yb = y_train[batch_idx].to(device)

                optimizer.zero_grad()
                pred = net(xb).squeeze(-1)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

            net.eval()
            with torch.no_grad():
                ho_pred = net(X_ho.to(device)).squeeze(-1)
                ho_loss = criterion(ho_pred, y_ho.to(device)).item()

            scheduler.step(ho_loss)

            if ho_loss < best_ho_loss:
                best_ho_loss = ho_loss
                best_state = {k: v.cpu().clone()
                              for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        self.model_ = net
        self.input_dim_ = X.shape[1]
        return self

    def predict(self, X, return_std=False):
        """Predict with MC-Dropout.

        If return_std=True, returns (mean, std) from mc_samples forward passes.
        """
        import torch
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.from_numpy(X)

        # Enable dropout at inference (train mode for dropout, but no grad)
        self.model_.train()
        preds = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                pred = self.model_(X_t).squeeze(-1).numpy()
                preds.append(pred * self.y_std_ + self.y_mean_)

        preds = np.array(preds)  # (mc_samples, n_samples)
        mu = preds.mean(axis=0)

        if return_std:
            sigma = preds.std(axis=0)
            return mu, sigma
        return mu


class BNNClassifier(BaseEstimator, ClassifierMixin):
    """MC-Dropout BNN classifier with sklearn interface.

    CrossEntropyLoss + softmax. predict_proba() uses MC-Dropout averaging.
    """

    def __init__(self, hidden_sizes=(128, 64), dropout=0.1, lr=0.001,
                 epochs=200, batch_size=256, patience=20, mc_samples=20,
                 random_state=42):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.mc_samples = mc_samples
        self.random_state = random_state

    def _build_net(self, input_dim, n_classes):
        import torch.nn as nn
        layers = []
        prev = input_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        return nn.Sequential(*layers)

    def fit(self, X, y, groups=None):
        import torch
        import torch.nn as nn
        from ._es_split import early_stopping_split

        _cap_torch_threads()
        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).ravel()

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_mapped = np.array([label_map[c] for c in y], dtype=np.int64)

        # Group-disjoint early-stopping holdout when groups are given.
        train_idx, ho_idx = early_stopping_split(
            y_mapped, groups, rng, 'BNNClassifier')

        X_train = torch.from_numpy(X[train_idx])
        y_train = torch.from_numpy(y_mapped[train_idx])
        X_ho = torch.from_numpy(X[ho_idx])
        y_ho = torch.from_numpy(y_mapped[ho_idx])

        device = torch.device('cpu')
        net = self._build_net(X.shape[1], n_classes).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        best_ho_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            net.train()
            perm = torch.randperm(len(X_train))
            for start in range(0, len(X_train), self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                xb = X_train[batch_idx].to(device)
                yb = y_train[batch_idx].to(device)

                optimizer.zero_grad()
                logits = net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

            net.eval()
            with torch.no_grad():
                ho_logits = net(X_ho.to(device))
                ho_loss = criterion(ho_logits, y_ho.to(device)).item()

            scheduler.step(ho_loss)

            if ho_loss < best_ho_loss:
                best_ho_loss = ho_loss
                best_state = {k: v.cpu().clone()
                              for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        self.model_ = net
        self.input_dim_ = X.shape[1]
        return self

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        """MC-Dropout averaged class probabilities."""
        import torch
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.from_numpy(X)

        self.model_.train()
        proba_samples = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                logits = self.model_(X_t)
                proba = torch.softmax(logits, dim=-1).numpy()
                proba_samples.append(proba)

        return np.mean(proba_samples, axis=0)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))
