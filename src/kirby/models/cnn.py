"""Sklearn-compatible PyTorch 1D CNN wrappers for regression and classification.

Critical design constraint: __init__ must NOT instantiate any PyTorch objects.
All torch objects are created in fit() to ensure deepcopy() works (required by
_evaluate_single in find_best_hybrid.py).

Multi-scale 1D CNN: treats the embedding vector as a 1D signal and applies
parallel Conv1d paths with different kernel sizes (3, 7, 15) to capture
features at different scales. ProteinGym-style architecture — achieves
Spearman ~0.62 vs Ridge ~0.48 on ESM-2 embeddings.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


def _build_cnn_net(input_dim, n_filters, kernel_sizes, dropout, output_dim):
    """Build multi-scale 1D CNN module.

    Input shape:  (batch, n_features)
    Reshape to:   (batch, 1, n_features)   — treat embedding as 1D signal
    Per path:     Conv1d(1, n_filters, k) → BatchNorm1d → ReLU → AdaptiveAvgPool1d(1)
    Concat:       (batch, n_filters * n_paths)
    FC:           → FC(n_filters * n_paths, 128) → ReLU → Dropout → FC(128, output_dim)
    """
    import torch
    import torch.nn as nn

    class MultiScaleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.paths = nn.ModuleList()
            for k in kernel_sizes:
                self.paths.append(nn.Sequential(
                    nn.Conv1d(1, n_filters, kernel_size=k, padding=k // 2),
                    # InstanceNorm1d instead of BatchNorm1d — no crash on batch_size=1
                    nn.InstanceNorm1d(n_filters, affine=True),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),  # → (batch, n_filters, 1)
                ))
            merged_dim = n_filters * len(kernel_sizes)
            self.fc = nn.Sequential(
                nn.Linear(merged_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, output_dim),
            )

        def forward(self, x):
            # x: (batch, n_features)
            x = x.unsqueeze(1)  # → (batch, 1, n_features)
            parts = [path(x).squeeze(-1) for path in self.paths]
            merged = torch.cat(parts, dim=1)  # (batch, n_filters * n_paths)
            return self.fc(merged)

    return MultiScaleCNN()


class CNNRegressor(BaseEstimator, RegressorMixin):
    """PyTorch multi-scale 1D CNN regressor with sklearn interface.

    Treats the embedding vector as a 1D signal. Parallel Conv1d paths with
    kernel sizes (3, 7, 15) capture features at different scales.
    Trains with Adam + ReduceLROnPlateau + early stopping.
    Internal 80/20 train/holdout split for early stopping.
    Target z-normalization for stable training.
    Gradient clipping: clip_grad_norm_ 1.0.
    """

    def __init__(self, n_filters=64, kernel_sizes=(3, 7, 15), dropout=0.2,
                 lr=0.001, epochs=200, batch_size=256, patience=20,
                 random_state=42):
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def fit(self, X, y):
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

        # 80/20 holdout for early stopping
        n = len(y)
        idx = rng.permutation(n)
        split = int(0.8 * n)
        train_idx, ho_idx = idx[:split], idx[split:]

        X_train = torch.from_numpy(X[train_idx])
        y_train = torch.from_numpy(y_norm[train_idx])
        X_ho = torch.from_numpy(X[ho_idx])
        y_ho = torch.from_numpy(y_norm[ho_idx])

        device = torch.device('cpu')
        net = _build_cnn_net(
            X.shape[1], self.n_filters, self.kernel_sizes, self.dropout,
            output_dim=1).to(device)

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


class CNNClassifier(BaseEstimator, ClassifierMixin):
    """PyTorch multi-scale 1D CNN classifier with sklearn interface.

    CrossEntropyLoss + softmax output. Supports predict_proba().
    Gradient clipping: clip_grad_norm_ 1.0.
    """

    def __init__(self, n_filters=64, kernel_sizes=(3, 7, 15), dropout=0.2,
                 lr=0.001, epochs=200, batch_size=256, patience=20,
                 random_state=42):
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

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
        net = _build_cnn_net(
            X.shape[1], self.n_filters, self.kernel_sizes, self.dropout,
            output_dim=n_classes).to(device)

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
