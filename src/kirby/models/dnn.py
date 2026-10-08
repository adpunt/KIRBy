"""Sklearn-compatible PyTorch DNN wrappers for regression and classification.

Critical design constraint: __init__ must NOT instantiate any PyTorch objects.
All torch objects are created in fit() to ensure deepcopy() works (required by
_evaluate_single in find_best_hybrid.py).

Template from src/kirby/finetuning/molecular_3d.py and qsar_qm_models.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


_GROUPS_WARNED = False


def _apply_sample_weight(X, y, groups, sample_weight, model_name):
    """Validate a per-sample weight and drop the rows it sets to zero.

    A weight of exactly zero deletes the row from the fit. The row is removed
    here, before the target normalization, before the early-stopping carve and
    before the batching, so fitting with zeros on a block of rows is the same
    fit as passing the data with those rows already removed.

    Returns (X, y, groups, w). w comes back None when every surviving row
    carries the same weight, because a uniform weight cannot change a loss that
    is already divided by the weight sum; the unweighted branch of the training
    loop then runs unchanged.
    """
    if sample_weight is None:
        return X, y, groups, None

    w = np.asarray(sample_weight, dtype=np.float32).ravel()
    if len(w) != len(y):
        raise ValueError(
            f"{model_name}: sample_weight length {len(w)} != "
            f"n_samples {len(y)}")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{model_name}: sample_weight has non-finite values")
    if np.any(w < 0):
        raise ValueError(f"{model_name}: sample_weight has negative values")

    keep = w > 0
    n_keep = int(keep.sum())
    if n_keep < 2:
        raise ValueError(
            f"{model_name}: sample_weight leaves {n_keep} rows with a weight "
            f"above zero, out of {len(w)}")
    if n_keep < len(w):
        X = X[keep]
        y = y[keep]
        w = w[keep]
        if groups is not None:
            groups_arr = np.asarray(groups)
            if len(groups_arr) != len(keep):
                raise ValueError(
                    f"{model_name}: groups length {len(groups_arr)} != "
                    f"n_samples {len(keep)}")
            groups = groups_arr[keep]

    if np.all(w == w[0]):
        return X, y, groups, None
    return X, y, groups, w


class DNNRegressor(BaseEstimator, RegressorMixin):
    """PyTorch DNN regressor with sklearn interface.

    Trains with Adam + ReduceLROnPlateau + early stopping.
    Internal 80/20 train/holdout split for early stopping.
    Target z-normalization for stable training.
    """

    def __init__(self, hidden_sizes=(256, 128, 64), dropout=0.2, lr=0.001,
                 epochs=200, batch_size=256, patience=20, random_state=42,
                 loss='mse', huber_delta=1.0):
        # loss='huber' was added 2026-09-24 for n33 (robust losses under label
        # noise). huber_delta is in LABEL units; fit() divides it by the label
        # SD because the network trains on z-normalised labels. With the
        # default loss='mse' the training loop is the one it always was.
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.loss = loss
        self.huber_delta = huber_delta

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

    def fit(self, X, y, groups=None, sample_weight=None):
        import torch
        import torch.nn as nn

        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()

        # A per-sample weight multiplies that row's squared error. Rows with a
        # weight of zero are removed outright, so they do not reach the target
        # normalization or the early-stopping carve either.
        X, y, groups, w = _apply_sample_weight(
            X, y, groups, sample_weight, 'DNNRegressor')

        # Z-normalize targets
        self.y_mean_ = float(y.mean())
        # Prediction cap (2026-10-08, Adelaide): see predict().
        self.y_min_ = float(np.min(y))
        self.y_max_ = float(np.max(y))
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

        w_train = w_ho = None
        if w is not None:
            w_train = torch.from_numpy(w[train_idx])
            w_ho = torch.from_numpy(w[ho_idx])

        device = torch.device('cpu')
        net = self._build_net(X.shape[1]).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)
        if self.loss == 'mse':
            criterion = nn.MSELoss()
            per_row = lambda p, t: (p - t) ** 2
        elif self.loss == 'huber':
            criterion = nn.HuberLoss(delta=float(self.huber_delta) / self.y_std_)
            _row_huber = nn.HuberLoss(reduction='none',
                                      delta=float(self.huber_delta) / self.y_std_)
            per_row = lambda p, t: _row_huber(p, t)
        else:
            raise ValueError(f"DNNRegressor: loss must be 'mse' or 'huber', "
                             f"got {self.loss!r}")

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
                if w_train is not None:
                    # Divided by the weights in the batch, not by the number of
                    # rows in it, so the loss keeps the scale an unweighted MSE
                    # has and the gradient clip below still means the same
                    # thing. Every surviving weight is above zero, so the
                    # denominator cannot be zero.
                    wb = w_train[batch_idx].to(device)
                    loss = (wb * per_row(pred, yb)).sum() / wb.sum()
                else:
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
                y_ho_d = y_ho.to(device)
                if w_ho is not None:
                    # Early stopping tracks the objective being minimised, so
                    # the holdout loss carries the same weights.
                    w_ho_d = w_ho.to(device)
                    ho_loss = float(
                        ((w_ho_d * per_row(ho_pred, y_ho_d)).sum()
                         / w_ho_d.sum()).item())
                else:
                    ho_loss = criterion(ho_pred, y_ho_d).item()

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
        pred = pred * self.y_std_ + self.y_mean_
        # CAP (2026-10-08, Adelaide): predictions are clipped to the range of
        # the training labels. Before this, one adme_hlm molecule (row 824:
        # pdv Ipc 1.95e17 against a training maximum of 1.15e11; mordred
        # feature 13 2.57e12) passed unbounded through the network and gave
        # predictions ~5e5 off, so one fold's RMSE reached 1e3-1e5 in c06,
        # c11, c17, c21 and c22 on adme_hlm and adme_rlm. Experiments that
        # limit held-out features to the training range (c01, c07, c08) never
        # showed it. Recorded in RESULT_CAVEATS.md; every dnn result written
        # after this commit is capped.
        lo = getattr(self, 'y_min_', None)
        hi = getattr(self, 'y_max_', None)
        if lo is not None and hi is not None:
            pred = np.clip(pred, lo, hi)
        return pred


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

    def fit(self, X, y, groups=None, sample_weight=None):
        import torch
        import torch.nn as nn
        from ._es_split import early_stopping_split

        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).ravel()

        # A per-sample weight multiplies that row's cross-entropy. Rows with a
        # weight of zero are removed outright, before the class set is read off
        # y and before the early-stopping carve.
        X, y, groups, w = _apply_sample_weight(
            X, y, groups, sample_weight, 'DNNClassifier')

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_mapped = np.array([label_map[c] for c in y], dtype=np.int64)

        # 80/20 holdout for early stopping, group-disjoint when groups are given.
        train_idx, ho_idx = early_stopping_split(
            y_mapped, groups, rng, 'DNNClassifier')

        X_train = torch.from_numpy(X[train_idx])
        y_train = torch.from_numpy(y_mapped[train_idx])
        X_ho = torch.from_numpy(X[ho_idx])
        y_ho = torch.from_numpy(y_mapped[ho_idx])

        w_train = w_ho = None
        if w is not None:
            w_train = torch.from_numpy(w[train_idx])
            w_ho = torch.from_numpy(w[ho_idx])

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
                if w_train is not None:
                    # Divided by the weights in the batch, not by the number of
                    # rows in it, so the loss keeps the scale an unweighted
                    # cross-entropy has. Every surviving weight is above zero,
                    # so the denominator cannot be zero.
                    wb = w_train[batch_idx].to(device)
                    per_row = nn.functional.cross_entropy(logits, yb,
                                                          reduction='none')
                    loss = (wb * per_row).sum() / wb.sum()
                else:
                    loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

            net.eval()
            with torch.no_grad():
                ho_logits = net(X_ho.to(device))
                y_ho_d = y_ho.to(device)
                if w_ho is not None:
                    # Early stopping tracks the objective being minimised, so
                    # the holdout loss carries the same weights.
                    w_ho_d = w_ho.to(device)
                    per_row = nn.functional.cross_entropy(ho_logits, y_ho_d,
                                                          reduction='none')
                    ho_loss = float(
                        ((w_ho_d * per_row).sum() / w_ho_d.sum()).item())
                else:
                    ho_loss = criterion(ho_logits, y_ho_d).item()

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
