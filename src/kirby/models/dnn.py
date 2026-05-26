"""Sklearn-compatible PyTorch DNN wrappers for regression and classification.

Critical design constraint: __init__ must NOT instantiate any PyTorch objects.
All torch objects are created in fit() to ensure deepcopy() works (required by
_evaluate_single in find_best_hybrid.py).

Template from src/kirby/finetuning/molecular_3d.py and qsar_qm_models.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


_GROUPS_WARNED = False


class DNNRegressor(BaseEstimator, RegressorMixin):
    """PyTorch DNN regressor with sklearn interface.

    Trains with Adam + ReduceLROnPlateau + early stopping.
    Internal 80/20 train/holdout split for early stopping.
    Target z-normalization for stable training.
    """

    def __init__(self, hidden_sizes=(256, 128, 64), dropout=0.2, lr=0.001,
                 epochs=200, batch_size=256, patience=20, random_state=42):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
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

        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()

        # Z-normalize targets
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) if y.std() > 1e-8 else 1.0
        y_norm = (y - self.y_mean_) / self.y_std_

        # 80/20 holdout for early stopping. Use group-aware split when groups
        # are provided so the internal ES holdout does not leak scaffolds from
        # the outer CV's train fold into its ES validation.
        n = len(y)
        if groups is not None:
            from sklearn.model_selection import GroupShuffleSplit
            groups_arr = np.asarray(groups)
            if len(groups_arr) != n:
                raise ValueError(
                    f"groups length {len(groups_arr)} != X length {n}")
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2,
                                    random_state=42)
            train_idx, ho_idx = next(gss.split(X, y_norm, groups=groups_arr))
        else:
            global _GROUPS_WARNED
            if not _GROUPS_WARNED:
                import warnings as _warnings
                _warnings.warn(
                    "DNNRegressor.fit called without groups — internal early-"
                    "stopping split is random and may leak scaffolds.",
                    stacklevel=2)
                _GROUPS_WARNED = True
            idx = rng.permutation(n)
            split = int(0.8 * n)
            train_idx, ho_idx = idx[:split], idx[split:]

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
            epoch_loss = 0.0
            n_batches = 0

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
                epoch_loss += loss.item()
                n_batches += 1

            # Holdout eval
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

    def predict(self, X):
        import torch
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(torch.from_numpy(X)).squeeze(-1).numpy()
        return pred * self.y_std_ + self.y_mean_


class DNNClassifier(BaseEstimator, ClassifierMixin):
    """PyTorch DNN classifier with sklearn interface.

    CrossEntropyLoss + softmax output. Supports predict_proba().
    """

    def __init__(self, hidden_sizes=(256, 128, 64), dropout=0.2, lr=0.001,
                 epochs=200, batch_size=256, patience=20, random_state=42):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
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

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).ravel()

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_mapped = np.array([label_map[c] for c in y], dtype=np.int64)

        # 80/20 holdout
        n = len(y)
        idx = rng.permutation(n)
        split = int(0.8 * n)
        train_idx, ho_idx = idx[:split], idx[split:]

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
        self.n_classes_ = n_classes
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X):
        import torch
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(X))
            proba = torch.softmax(logits, dim=1).numpy()
        return proba
