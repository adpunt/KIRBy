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

def _build_dnn(input_dim, hidden_sizes=(256, 128, 64), dropout=0.2, output_dim=1):
    """Build DNN matching DNNRegressor._build_net() exactly.

    For classification pass output_dim=n_classes; the final layer outputs
    logits (no softmax — pair with nn.CrossEntropyLoss).
    """
    import torch.nn as nn
    layers = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def _z_normalize(y):
    """Z-normalize targets. Returns (y_norm, mean, std)."""
    y = np.asarray(y, dtype=np.float32).ravel()
    mean = float(y.mean())
    std = float(y.std()) if y.std() > 1e-8 else 1.0
    return (y - mean) / std, mean, std


def _prep_clf_labels(y_train, y_val=None):
    """Map class labels to contiguous 0..n_classes-1 ints.

    Returns (y_train_int, classes_, y_val_int_or_None). The training labels
    define the class set; val labels are mapped through the same dict, and
    any val class missing from training is left at -1 (caller decides what
    to do — usually train-class predictions just won't match those rows).
    """
    y_train = np.asarray(y_train).ravel()
    classes_ = np.unique(y_train)
    label_map = {c: i for i, c in enumerate(classes_)}
    y_train_int = np.array([label_map[c] for c in y_train], dtype=np.int64)
    if y_val is None:
        return y_train_int, classes_, None
    y_val = np.asarray(y_val).ravel()
    y_val_int = np.array([label_map.get(c, -1) for c in y_val], dtype=np.int64)
    return y_train_int, classes_, y_val_int


def _clf_pred_from_logits(logits_np, classes_):
    """Convert raw logits (np.ndarray, shape (n, n_classes)) to y_pred.

    Returns P(class=positive) for binary classification (1-D float array),
    or argmax class labels for multi-class. This matches the convention
    expected by BUILTIN_METRICS: 'auc' wants probabilities for binary,
    'f1_weighted'/'accuracy'/'mcc' want class labels for multi-class.
    """
    import scipy.special
    proba = scipy.special.softmax(logits_np, axis=1)
    if proba.shape[1] == 2:
        # Binary — return P(y = positive class) so AUROC sees probabilities.
        return proba[:, 1]
    # Multi-class — return predicted class labels (caller maps via classes_).
    idx = np.argmax(proba, axis=1)
    return classes_[idx]


# (Module-level _per_sample_loss_torch was removed — each method writes its
# own inline _per_sample / _val_loss closures that hold a reference to
# self.is_clf, which is cleaner than a free function call. The classification
# rewrite for each LNL method follows this pattern.)


def _standard_dnn_train_predict(X_train, y_train, X_val,
                                epochs=200, hidden_sizes=(256, 128, 64),
                                dropout=0.2, lr=0.001, batch_size=256,
                                patience=20, random_state=42,
                                sample_weights=None, groups=None):
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
    y_for_split, _es_name = y_norm, 'lnl_dnn_regressor'
    # Early-stopping holdout, group-disjoint when the caller supplies the
    # fold's scaffold groups. A random carve puts the same scaffold on both
    # sides, so the early-stopping loss keeps falling on rows the net has
    # effectively seen and training stops too late.
    from kirby.models._es_split import early_stopping_split
    tr_idx, ho_idx = early_stopping_split(
        y_for_split, groups, rng, _es_name)

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


def _standard_dnn_train_predict_clf(X_train, y_train, X_val,
                                    epochs=200, hidden_sizes=(256, 128, 64),
                                    dropout=0.2, lr=0.001, batch_size=256,
                                    patience=20, random_state=42,
                                    sample_weights=None,
                                    classes_=None, groups=None):
    """Standard DNN classifier — clf mirror of _standard_dnn_train_predict.

    Cross-entropy loss, softmax output. Returns predictions in the format
    the noise-experiment metrics expect: P(y=positive) for binary, argmax
    class labels for multi-class. Optionally accepts per-sample weights
    for weighted cross-entropy.

    classes_: if provided, override the auto-detected class set (useful
    when y_train is a subsample that may not contain all classes — caller
    can pass the full class set so the network still has n_classes outputs).
    """
    import torch
    import torch.nn as nn

    rng = np.random.RandomState(random_state)
    torch.manual_seed(random_state)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)

    if classes_ is None:
        y_train_int, classes_, _ = _prep_clf_labels(y_train)
    else:
        label_map = {c: i for i, c in enumerate(classes_)}
        y_train_int = np.array([label_map[c] for c in np.asarray(y_train).ravel()],
                               dtype=np.int64)
    n_classes = len(classes_)

    n = len(y_train_int)
    y_for_split, _es_name = y_train_int, 'lnl_dnn_classifier'
    # Early-stopping holdout, group-disjoint when the caller supplies the
    # fold's scaffold groups. A random carve puts the same scaffold on both
    # sides, so the early-stopping loss keeps falling on rows the net has
    # effectively seen and training stops too late.
    from kirby.models._es_split import early_stopping_split
    tr_idx, ho_idx = early_stopping_split(
        y_for_split, groups, rng, _es_name)

    X_tr = torch.from_numpy(X_train[tr_idx])
    y_tr = torch.from_numpy(y_train_int[tr_idx])
    X_ho = torch.from_numpy(X_train[ho_idx])
    y_ho = torch.from_numpy(y_train_int[ho_idx])

    w_tr = None
    if sample_weights is not None:
        w_all = np.asarray(sample_weights, dtype=np.float32)
        w_tr = torch.from_numpy(w_all[tr_idx])

    device = torch.device('cpu')
    net = _build_dnn(X_train.shape[1], hidden_sizes, dropout,
                     output_dim=n_classes).to(device)
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
            logits = net(xb)

            if w_tr is not None:
                wb = w_tr[batch_idx].to(device)
                per_sample = nn.functional.cross_entropy(logits, yb,
                                                        reduction='none')
                loss = (wb * per_sample).mean()
            else:
                loss = nn.functional.cross_entropy(logits, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

        net.eval()
        with torch.no_grad():
            ho_logits = net(X_ho.to(device))
            ho_loss = nn.functional.cross_entropy(
                ho_logits, y_ho.to(device)).item()

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

    net.eval()
    with torch.no_grad():
        X_val_t = torch.from_numpy(X_val).to(device)
        val_logits = net(X_val_t).cpu().numpy()

    return _clf_pred_from_logits(val_logits, classes_)


def _train_clf_dnn_return_proba(X_train, y_train_int, X_predict, n_classes,
                                epochs=200, hidden_sizes=(256, 128, 64),
                                dropout=0.2, lr=0.001, batch_size=256,
                                patience=20, random_state=42, groups=None):
    """Train a clf DNN and return the full (n_predict, n_classes) proba matrix.

    Internal helper for ensemble/disagreement methods that need per-class
    probabilities rather than the user-facing 1D format from
    _standard_dnn_train_predict_clf. y_train_int must already be mapped to
    contiguous 0..n_classes-1 ints (use _prep_clf_labels).
    """
    import torch
    import torch.nn as nn
    import scipy.special

    rng = np.random.RandomState(random_state)
    torch.manual_seed(random_state)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_predict = np.asarray(X_predict, dtype=np.float32)

    n = len(y_train_int)
    y_for_split, _es_name = y_train_int, 'lnl_dnn_classifier'
    # Early-stopping holdout, group-disjoint when the caller supplies the
    # fold's scaffold groups. A random carve puts the same scaffold on both
    # sides, so the early-stopping loss keeps falling on rows the net has
    # effectively seen and training stops too late.
    from kirby.models._es_split import early_stopping_split
    tr_idx, ho_idx = early_stopping_split(
        y_for_split, groups, rng, _es_name)

    X_tr = torch.from_numpy(X_train[tr_idx])
    y_tr = torch.from_numpy(y_train_int[tr_idx])
    X_ho = torch.from_numpy(X_train[ho_idx])
    y_ho = torch.from_numpy(y_train_int[ho_idx])

    device = torch.device('cpu')
    net = _build_dnn(X_train.shape[1], hidden_sizes, dropout,
                     output_dim=n_classes).to(device)
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
            logits = net(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
        net.eval()
        with torch.no_grad():
            ho_logits = net(X_ho.to(device))
            ho_loss = nn.functional.cross_entropy(
                ho_logits, y_ho.to(device)).item()
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

    net.eval()
    with torch.no_grad():
        logits = net(torch.from_numpy(X_predict).to(device)).cpu().numpy()
    return scipy.special.softmax(logits, axis=1)


# ============================================================================
# BASE CLASS
# ============================================================================

class LNLMethod(ABC):
    """Base class for Learning with Noisy Labels methods.

    Set `is_clf=True` and `classes_=<full class set>` at construction to run
    the classification path. The 2 GP/evidential methods (HeteroscedasticGP,
    EvidentialKernel) raise NotImplementedError on is_clf=True — their
    underlying math (continuous noise variance, NIG distribution) does not
    have a loss-swap classification analog.
    """

    def __init__(self, random_state=42, epochs=200, hidden_sizes=(256, 128, 64),
                 dropout=0.2, lr=0.001, batch_size=256, patience=20,
                 is_clf=False, classes_=None):
        self.random_state = random_state
        self.epochs = epochs
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.is_clf = is_clf
        # classes_ is the *full* class set the experiment cares about,
        # passed in so per-fold y_train subsets that happen to miss a class
        # still build a network with the right number of output units.
        self.classes_ = (np.asarray(classes_) if classes_ is not None else None)

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

    # ---- helpers shared across methods ----

    def _resolve_classes(self, y_train):
        """Return the class set: explicit self.classes_ if set, else y_train uniques."""
        if self.classes_ is not None:
            return self.classes_
        return np.unique(np.asarray(y_train).ravel())

    def _default_predict(self, X_train, y_train, X_val, sample_weights=None):
        """Convenience: standard DNN train+predict with current hyperparams.

        Routes to the clf trainer when self.is_clf=True.
        """
        if self.is_clf:
            return _standard_dnn_train_predict_clf(
                X_train, y_train, X_val,
                epochs=self.epochs, hidden_sizes=self.hidden_sizes,
                dropout=self.dropout, lr=self.lr, batch_size=self.batch_size,
                patience=self.patience, random_state=self.random_state,
                sample_weights=sample_weights,
                classes_=self._resolve_classes(y_train),
            )
        return _standard_dnn_train_predict(
            X_train, y_train, X_val,
            epochs=self.epochs, hidden_sizes=self.hidden_sizes,
            dropout=self.dropout, lr=self.lr, batch_size=self.batch_size,
            patience=self.patience, random_state=self.random_state,
            sample_weights=sample_weights,
        )

    def _prep_y_for_torch(self, y_train):
        """Return (y_tensor, prep_meta) ready to feed into a torch loop.

        Regression: (y_norm_tensor[float32], (y_mean, y_std)) — caller
        de-normalizes outputs with `pred * y_std + y_mean`.

        Classification: (y_int_tensor[long], classes_) — caller predicts
        class probabilities/labels with _clf_pred_from_logits.
        """
        import torch
        if self.is_clf:
            classes_ = self._resolve_classes(y_train)
            label_map = {c: i for i, c in enumerate(classes_)}
            y_int = np.array([label_map[c] for c in np.asarray(y_train).ravel()],
                             dtype=np.int64)
            return torch.from_numpy(y_int), classes_
        y_norm, y_mean, y_std = _z_normalize(y_train)
        return torch.from_numpy(y_norm), (y_mean, y_std)


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
        device = torch.device('cpu')
        X_tensor = torch.from_numpy(X_t)

        if self.is_clf:
            classes_ = self._resolve_classes(y_train)
            label_map = {c: i for i, c in enumerate(classes_)}
            y_int = np.array([label_map[c] for c in np.asarray(y_train).ravel()],
                             dtype=np.int64)
            y_tensor = torch.from_numpy(y_int)
            net = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout,
                             output_dim=len(classes_)).to(device)
        else:
            y_norm, y_mean, y_std = _z_normalize(y_train)
            y_tensor = torch.from_numpy(y_norm)
            net = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)

        # Warmup: train DNN for warmup_epochs on all data
        for epoch in range(self.warmup_epochs):
            net.train()
            perm = torch.randperm(len(X_t))
            for start in range(0, len(X_t), self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                out = net(X_tensor[batch_idx].to(device))
                yb = y_tensor[batch_idx].to(device)
                if self.is_clf:
                    loss = nn.functional.cross_entropy(out, yb)
                else:
                    loss = nn.functional.mse_loss(out.squeeze(-1), yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

        # Per-sample loss
        net.eval()
        with torch.no_grad():
            out = net(X_tensor.to(device))
            if self.is_clf:
                per_sample_loss = nn.functional.cross_entropy(
                    out, y_tensor.to(device), reduction='none').cpu().numpy()
            else:
                preds = out.squeeze(-1).cpu().numpy()
                per_sample_loss = (preds - y_norm) ** 2

        # Keep smallest-loss samples
        n_keep = max(10, int(len(y_train) * self.selection_rate))
        keep_idx = np.argsort(per_sample_loss)[:n_keep]

        # Retrain on filtered data (routes to clf or reg trainer via _default_predict)
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

        if self.is_clf:
            classes_ = self._resolve_classes(y_train)
            label_map = {c: i for i, c in enumerate(classes_)}
            y_int = np.array([label_map[c] for c in np.asarray(y_train).ravel()],
                             dtype=np.int64)
            n_classes = len(classes_)
            # Ensemble of clf DNNs; collect (n_ensemble, n_samples, n_classes)
            proba_list = []
            for i in range(self.n_ensemble):
                proba_i = _train_clf_dnn_return_proba(
                    X_t, y_int, X_t, n_classes,
                    epochs=min(self.epochs, 100), hidden_sizes=self.hidden_sizes,
                    dropout=self.dropout, lr=self.lr, batch_size=self.batch_size,
                    patience=self.patience, random_state=self.random_state + i,
                )
                proba_list.append(proba_i)
            probs = np.array(proba_list)  # (E, n, K)
            mean_proba = probs.mean(axis=0)  # (n, K)
            # Per-sample error = 1 - P(true class) averaged across ensemble
            true_class_proba = mean_proba[np.arange(len(y_int)), y_int]
            errors = 1.0 - true_class_proba
            # Uncertainty = mean across classes of stddev across ensemble
            std_per_class = probs.std(axis=0)  # (n, K)
            std_pred = std_per_class.mean(axis=1)
        else:
            # Train ensemble of regression DNNs
            predictions = []
            for i in range(self.n_ensemble):
                y_pred_i = _standard_dnn_train_predict(
                    X_t, y_train, X_t,
                    epochs=min(self.epochs, 100), hidden_sizes=self.hidden_sizes,
                    dropout=self.dropout, lr=self.lr, batch_size=self.batch_size,
                    patience=self.patience, random_state=self.random_state + i,
                )
                predictions.append(y_pred_i)
            preds = np.array(predictions)  # (E, n)
            mean_pred = preds.mean(axis=0)
            std_pred = preds.std(axis=0)
            errors = np.abs(y_train - mean_pred)

        # Flag: high error AND low uncertainty (model confidently disagrees)
        error_threshold = np.percentile(errors, self.error_pct)
        uncertainty_threshold = np.percentile(std_pred, self.uncertainty_pct)

        flagged = (errors > error_threshold) & (std_pred < uncertainty_threshold)
        keep_mask = ~flagged

        if keep_mask.sum() < 10:
            keep_mask = np.ones(len(y_train), dtype=bool)

        # Retrain on clean subset (routes to clf or reg trainer)
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
        from sklearn.preprocessing import StandardScaler

        X_t = np.asarray(X_train, dtype=np.float32)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_t)

        # kNN structure
        k_actual = min(self.k, len(y_train) - 1)
        nn_model = NearestNeighbors(n_neighbors=k_actual + 1)
        nn_model.fit(X_scaled)
        _, indices = nn_model.kneighbors(X_scaled)
        neighbor_idx = indices[:, 1:]  # exclude self

        y_arr = np.asarray(y_train).ravel()
        if self.is_clf:
            # Disagreement = fraction of neighbors with a different class label.
            # Symmetric, scale-free, and meaningful for binary AND multi-class.
            neighbor_labels = y_arr[neighbor_idx]  # (n, k)
            disagreement = (neighbor_labels != y_arr[:, None]).mean(axis=1)
        else:
            neighbor_means = np.mean(y_arr[neighbor_idx], axis=1)
            neighbor_stds = np.std(y_arr[neighbor_idx], axis=1) + 1e-10
            disagreement = np.abs(y_arr - neighbor_means) / neighbor_stds

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

        if self.is_clf:
            classes_ = self._resolve_classes(y_train)
            label_map = {c: i for i, c in enumerate(classes_)}
            y_int_full = np.array(
                [label_map[c] for c in np.asarray(y_train).ravel()],
                dtype=np.int64,
            )
            n_classes = len(classes_)

        for stage in range(self.n_stages):
            X_stage = X_t[current_idx]
            y_stage = y_train[current_idx]

            if self.is_clf:
                y_stage_int = y_int_full[current_idx]
                # Train clf, then compute per-sample CE on training data.
                proba = _train_clf_dnn_return_proba(
                    X_stage, y_stage_int, X_stage, n_classes,
                    epochs=min(self.epochs, 100),
                    hidden_sizes=self.hidden_sizes, dropout=self.dropout,
                    lr=self.lr, batch_size=self.batch_size,
                    patience=self.patience,
                    random_state=self.random_state + stage,
                )
                # -log P(true class), shape (n_stage,)
                losses = -np.log(np.clip(
                    proba[np.arange(len(y_stage_int)), y_stage_int],
                    1e-10, 1.0))
            else:
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

        # Final training on cleaned data (routes through _default_predict)
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
        device = torch.device('cpu')

        if self.is_clf:
            classes_ = self._resolve_classes(y_train)
            label_map = {c: i for i, c in enumerate(classes_)}
            y_int = np.array([label_map[c] for c in np.asarray(y_train).ravel()],
                             dtype=np.int64)
            y_val_int = np.array(
                [label_map.get(c, 0) for c in np.asarray(y_val).ravel()],
                dtype=np.int64)
            y_tensor = torch.from_numpy(y_int)
            y_val_tensor = torch.from_numpy(y_val_int)
            net = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout,
                             output_dim=len(classes_)).to(device)
        else:
            y_norm, y_mean, y_std = _z_normalize(y_train)
            y_val_norm = (np.asarray(y_val, dtype=np.float32) - y_mean) / y_std
            y_tensor = torch.from_numpy(y_norm)
            y_val_tensor = torch.from_numpy(y_val_norm)
            net = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)

        X_tensor = torch.from_numpy(X_t)
        X_val_tensor = torch.from_numpy(X_v)

        # Per-sample weights
        weights = np.ones(len(y_train), dtype=np.float32)

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
                out = net(xb)
                if self.is_clf:
                    per = nn.functional.cross_entropy(out, yb, reduction='none')
                    loss = (wb * per).mean()
                else:
                    pred = out.squeeze(-1)
                    loss = (wb * (pred - yb) ** 2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

            # Compute per-sample loss and update weights
            net.eval()
            with torch.no_grad():
                out_all = net(X_tensor.to(device))
                if self.is_clf:
                    per_sample_loss = nn.functional.cross_entropy(
                        out_all, y_tensor.to(device),
                        reduction='none').cpu().numpy()
                else:
                    all_pred = out_all.squeeze(-1).cpu().numpy()
                    per_sample_loss = (all_pred - y_norm) ** 2
            median_loss = np.median(per_sample_loss)

            high_loss_mask = per_sample_loss > median_loss
            weights[high_loss_mask] *= self.gamma
            weights = np.clip(weights, 0.01, 1.0)
            weights = weights / weights.mean()

            # Val early stopping
            with torch.no_grad():
                val_out = net(X_val_tensor.to(device))
                if self.is_clf:
                    val_loss = nn.functional.cross_entropy(
                        val_out, y_val_tensor.to(device)).item()
                else:
                    val_pred = val_out.squeeze(-1)
                    val_loss = nn.functional.mse_loss(
                        val_pred, y_val_tensor.to(device)).item()

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
            out = net(X_val_tensor.to(device))
            if self.is_clf:
                y_pred = _clf_pred_from_logits(out.cpu().numpy(), classes_)
            else:
                y_pred = out.squeeze(-1).cpu().numpy() * y_std + y_mean

        return {
            'y_pred': y_pred,
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
        n = len(y_train)
        device = torch.device('cpu')

        if self.is_clf:
            classes_ = self._resolve_classes(y_train)
            label_map = {c: i for i, c in enumerate(classes_)}
            y_int = np.array([label_map[c] for c in np.asarray(y_train).ravel()],
                             dtype=np.int64)
            y_val_int = np.array(
                [label_map.get(c, 0) for c in np.asarray(y_val).ravel()],
                dtype=np.int64)
            y_tensor = torch.from_numpy(y_int)
            y_val_tensor = torch.from_numpy(y_val_int)
            net = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout,
                             output_dim=len(classes_)).to(device)
        else:
            y_norm, y_mean, y_std = _z_normalize(y_train)
            y_val_norm = (np.asarray(y_val, dtype=np.float32) - y_mean) / y_std
            y_tensor = torch.from_numpy(y_norm)
            y_val_tensor = torch.from_numpy(y_val_norm)
            net = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)

        X_tensor = torch.from_numpy(X_t)
        X_val_tensor = torch.from_numpy(X_v)

        # Compute stage epoch boundaries
        remaining = self.epochs - self.warmup_epochs
        stage_length = remaining // len(self.stages)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        current_inclusion = np.ones(n, dtype=bool)

        for epoch in range(self.epochs):
            if epoch < self.warmup_epochs:
                current_inclusion = np.ones(n, dtype=bool)
            else:
                stage_idx = min(
                    (epoch - self.warmup_epochs) // max(stage_length, 1),
                    len(self.stages) - 1
                )
                fraction = self.stages[stage_idx]

                net.eval()
                with torch.no_grad():
                    out_all = net(X_tensor.to(device))
                    if self.is_clf:
                        per_sample_loss = nn.functional.cross_entropy(
                            out_all, y_tensor.to(device),
                            reduction='none').cpu().numpy()
                    else:
                        all_pred = out_all.squeeze(-1).cpu().numpy()
                        per_sample_loss = (all_pred - y_norm) ** 2
                n_include = max(10, int(n * fraction))
                include_idx = np.argsort(per_sample_loss)[:n_include]
                current_inclusion = np.zeros(n, dtype=bool)
                current_inclusion[include_idx] = True

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
                out = net(xb)
                if self.is_clf:
                    loss = nn.functional.cross_entropy(out, yb)
                else:
                    loss = nn.functional.mse_loss(out.squeeze(-1), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

            # Val early stopping
            net.eval()
            with torch.no_grad():
                val_out = net(X_val_tensor.to(device))
                if self.is_clf:
                    val_loss = nn.functional.cross_entropy(
                        val_out, y_val_tensor.to(device)).item()
                else:
                    val_pred = val_out.squeeze(-1)
                    val_loss = nn.functional.mse_loss(
                        val_pred, y_val_tensor.to(device)).item()

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
            out = net(X_val_tensor.to(device))
            if self.is_clf:
                y_pred = _clf_pred_from_logits(out.cpu().numpy(), classes_)
            else:
                y_pred = out.squeeze(-1).cpu().numpy() * y_std + y_mean

        return {
            'y_pred': y_pred,
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
        import scipy.special
        from sklearn.mixture import GaussianMixture

        torch.manual_seed(self.random_state)
        X_t = np.asarray(X_train, dtype=np.float32)
        X_v = np.asarray(X_val, dtype=np.float32)
        n = len(y_train)
        device = torch.device('cpu')

        if self.is_clf:
            classes_ = self._resolve_classes(y_train)
            label_map = {c: i for i, c in enumerate(classes_)}
            y_int = np.array([label_map[c] for c in np.asarray(y_train).ravel()],
                             dtype=np.int64)
            y_val_int = np.array(
                [label_map.get(c, 0) for c in np.asarray(y_val).ravel()],
                dtype=np.int64)
            n_classes = len(classes_)
            y_tensor = torch.from_numpy(y_int)
            y_val_tensor = torch.from_numpy(y_val_int)
            net_a = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout,
                               output_dim=n_classes).to(device)
            net_b = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout,
                               output_dim=n_classes).to(device)
        else:
            y_norm, y_mean, y_std = _z_normalize(y_train)
            y_val_norm = (np.asarray(y_val, dtype=np.float32) - y_mean) / y_std
            y_tensor = torch.from_numpy(y_norm)
            y_val_tensor = torch.from_numpy(y_val_norm)
            net_a = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout).to(device)
            net_b = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout).to(device)

        opt_a = torch.optim.Adam(net_a.parameters(), lr=self.lr)
        opt_b = torch.optim.Adam(net_b.parameters(), lr=self.lr)

        X_tensor = torch.from_numpy(X_t)
        X_val_tensor = torch.from_numpy(X_v)

        def _train_step(net, opt, idx_subset):
            """One epoch over idx_subset for either reg or clf."""
            net.train()
            if len(idx_subset) < 2:
                return
            perm = torch.from_numpy(
                np.random.RandomState(self.random_state + epoch).permutation(len(idx_subset))
            )
            for start in range(0, len(idx_subset), self.batch_size):
                batch_local = perm[start:start + self.batch_size]
                batch_global = torch.from_numpy(idx_subset)[batch_local]
                out = net(X_tensor[batch_global].to(device))
                yb = y_tensor[batch_global].to(device)
                if self.is_clf:
                    loss = nn.functional.cross_entropy(out, yb)
                else:
                    loss = nn.functional.mse_loss(out.squeeze(-1), yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

        def _per_sample_loss_np(net):
            """Per-sample loss across all training samples (numpy 1D)."""
            net.eval()
            with torch.no_grad():
                out = net(X_tensor.to(device))
                if self.is_clf:
                    return nn.functional.cross_entropy(
                        out, y_tensor.to(device), reduction='none').cpu().numpy()
                pred = out.squeeze(-1).cpu().numpy()
                return (pred - y_norm) ** 2

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        gmm_labels = None
        clean_mask_a = clean_mask_b = np.ones(n, dtype=bool)

        for epoch in range(self.epochs):
            # Warmup: full-data training for both nets
            if epoch < self.warmup_epochs:
                _train_step(net_a, opt_a, np.arange(n))
                _train_step(net_b, opt_b, np.arange(n))
                continue

            # GMM separation on per-sample losses from net_a
            losses_a = _per_sample_loss_np(net_a)
            log_losses = np.log(losses_a + 1e-10).reshape(-1, 1)
            try:
                gmm = GaussianMixture(n_components=2, random_state=self.random_state)
                gmm.fit(log_losses)
                gmm_labels = gmm.predict(log_losses)
                clean_comp = 0 if gmm.means_[0] < gmm.means_[1] else 1
                clean_mask_a = (gmm_labels == clean_comp)
            except Exception:
                clean_mask_a = np.ones(n, dtype=bool)

            # GMM separation on per-sample losses from net_b
            losses_b = _per_sample_loss_np(net_b)
            log_losses_b = np.log(losses_b + 1e-10).reshape(-1, 1)
            try:
                gmm_b = GaussianMixture(n_components=2, random_state=self.random_state)
                gmm_b.fit(log_losses_b)
                gmm_labels_b = gmm_b.predict(log_losses_b)
                clean_comp_b = 0 if gmm_b.means_[0] < gmm_b.means_[1] else 1
                clean_mask_b = (gmm_labels_b == clean_comp_b)
            except Exception:
                clean_mask_b = np.ones(n, dtype=bool)

            # Cross-train each net on the other's clean set
            _train_step(net_a, opt_a, np.where(clean_mask_b)[0])
            _train_step(net_b, opt_b, np.where(clean_mask_a)[0])

            # Val early stopping (average loss across the two nets)
            net_a.eval()
            net_b.eval()
            with torch.no_grad():
                out_a = net_a(X_val_tensor.to(device))
                out_b = net_b(X_val_tensor.to(device))
                if self.is_clf:
                    val_loss = 0.5 * (
                        nn.functional.cross_entropy(out_a, y_val_tensor.to(device)).item() +
                        nn.functional.cross_entropy(out_b, y_val_tensor.to(device)).item()
                    )
                else:
                    pred_va = out_a.squeeze(-1)
                    pred_vb = out_b.squeeze(-1)
                    avg_pred = (pred_va + pred_vb) / 2
                    val_loss = nn.functional.mse_loss(
                        avg_pred, y_val_tensor.to(device)).item()

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
            out_a = net_a(X_val_tensor.to(device)).cpu().numpy()
            out_b = net_b(X_val_tensor.to(device)).cpu().numpy()

        if self.is_clf:
            # Average softmax probabilities, then format for the metric
            proba = 0.5 * (scipy.special.softmax(out_a, axis=1) +
                           scipy.special.softmax(out_b, axis=1))
            if n_classes == 2:
                y_pred = proba[:, 1]
            else:
                y_pred = classes_[np.argmax(proba, axis=1)]
        else:
            y_pred = (out_a.squeeze(-1) + out_b.squeeze(-1)) / 2 * y_std + y_mean

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
        device = torch.device('cpu')

        if self.is_clf:
            classes_ = self._resolve_classes(y_train)
            label_map = {c: i for i, c in enumerate(classes_)}
            y_int = np.array([label_map[c] for c in np.asarray(y_train).ravel()],
                             dtype=np.int64)
            y_val_int = np.array(
                [label_map.get(c, 0) for c in np.asarray(y_val).ravel()],
                dtype=np.int64)
            y_tensor = torch.from_numpy(y_int)
            y_val_tensor = torch.from_numpy(y_val_int)
            net = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout,
                             output_dim=len(classes_)).to(device)
        else:
            y_norm, y_mean, y_std = _z_normalize(y_train)
            y_val_norm = (np.asarray(y_val, dtype=np.float32) - y_mean) / y_std
            y_tensor = torch.from_numpy(y_norm)
            y_val_tensor = torch.from_numpy(y_val_norm)
            net = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)

        # Meta-network: per-sample loss (scalar) -> weight in (0, 1).
        # Input shape and semantics are identical for reg and clf — the loss
        # value is just MSE vs CE; the meta-net only sees the scalar magnitude.
        meta_net = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        ).to(device)
        meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=self.meta_lr)

        X_tensor = torch.from_numpy(X_t)
        X_val_tensor = torch.from_numpy(X_v)

        def _per_sample(net_out, y_target):
            if self.is_clf:
                return nn.functional.cross_entropy(net_out, y_target, reduction='none')
            return (net_out.squeeze(-1) - y_target) ** 2

        def _val_loss(net_out, y_target):
            if self.is_clf:
                return nn.functional.cross_entropy(net_out, y_target)
            return nn.functional.mse_loss(net_out.squeeze(-1), y_target)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            # Meta-update every meta_interval epochs.
            #
            # The meta-network learns per-sample weights with a first-order
            # bilevel step: take a VIRTUAL gradient step on the main net using
            # the meta-weighted train loss, then measure validation loss under
            # those virtual params and backprop it THROUGH the weights into the
            # meta-network. The previous version detached the weights before the
            # inner step and stepped the real optimiser, so the validation loss
            # had no gradient path to the meta-net — it never learned and just
            # applied its (fixed, near-random) initial weights.
            if epoch > 0 and epoch % self.meta_interval == 0 and len(X_v) > 0:
                try:
                    from torch.func import functional_call
                except ImportError:  # older torch
                    from torch.nn.utils.stateless import functional_call

                # eval() so dropout is off — the meta-gradient is deterministic
                # given the seed, not polluted by per-forward dropout masks.
                net.eval()
                out = net(X_tensor.to(device))
                per_loss = _per_sample(out, y_tensor.to(device))
                # weights carry a graph to meta_net params (the loss VALUE is
                # fed as a detached feature; the weight tensor itself does not
                # detach, so gradients can reach the meta-net).
                weights = meta_net(per_loss.detach().unsqueeze(-1)).squeeze(-1)
                weighted_loss = (weights * per_loss).mean()

                # Virtual one-step SGD update of the net params. create_graph
                # keeps the dependence on meta_net so meta_loss.backward() below
                # differentiates through weights -> virtual params -> val loss.
                param_items = list(net.named_parameters())
                grads = torch.autograd.grad(
                    weighted_loss, [p for _, p in param_items], create_graph=True)
                virtual_params = {name: p - self.lr * g
                                  for (name, p), g in zip(param_items, grads)}

                val_out = functional_call(
                    net, virtual_params, (X_val_tensor.to(device),))
                meta_loss = _val_loss(val_out, y_val_tensor.to(device))

                meta_optimizer.zero_grad()
                meta_loss.backward()
                meta_optimizer.step()
                net.train()

            # Standard weighted training
            net.train()
            perm = torch.randperm(len(X_t))

            with torch.no_grad():
                out_all = net(X_tensor.to(device))
                all_loss = _per_sample(out_all, y_tensor.to(device)).unsqueeze(-1)
                meta_w = meta_net(all_loss).squeeze(-1).cpu()

            for start in range(0, len(X_t), self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                xb = X_tensor[batch_idx].to(device)
                yb = y_tensor[batch_idx].to(device)
                wb = meta_w[batch_idx].to(device)

                optimizer.zero_grad()
                out = net(xb)
                per = _per_sample(out, yb)
                loss = (wb * per).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

            # Val early stopping
            net.eval()
            with torch.no_grad():
                val_out = net(X_val_tensor.to(device))
                val_loss = _val_loss(val_out, y_val_tensor.to(device)).item()

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
            out = net(X_val_tensor.to(device))
            if self.is_clf:
                y_pred = _clf_pred_from_logits(out.cpu().numpy(), classes_)
            else:
                y_pred = out.squeeze(-1).cpu().numpy() * y_std + y_mean

        return {
            'y_pred': y_pred,
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
        n = len(y_train)
        device = torch.device('cpu')

        if self.is_clf:
            classes_ = self._resolve_classes(y_train)
            label_map = {c: i for i, c in enumerate(classes_)}
            y_int = np.array([label_map[c] for c in np.asarray(y_train).ravel()],
                             dtype=np.int64)
            y_val_int = np.array(
                [label_map.get(c, 0) for c in np.asarray(y_val).ravel()],
                dtype=np.int64)
            y_tensor = torch.from_numpy(y_int)
            y_val_tensor = torch.from_numpy(y_val_int)
            net = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout,
                             output_dim=len(classes_)).to(device)
        else:
            y_norm, y_mean, y_std = _z_normalize(y_train)
            y_val_norm = (np.asarray(y_val, dtype=np.float32) - y_mean) / y_std
            y_tensor = torch.from_numpy(y_norm)
            y_val_tensor = torch.from_numpy(y_val_norm)
            net = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout).to(device)

        # Pre-compute kNN distances (feature-based, label-agnostic)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_t)
        k_actual = min(self.k_neighbors, n - 1)
        nn_model = NearestNeighbors(n_neighbors=k_actual + 1)
        nn_model.fit(X_scaled)
        distances, _ = nn_model.kneighbors(X_scaled)
        knn_dist = distances[:, 1:].mean(axis=1)
        knn_dist = (knn_dist - knn_dist.mean()) / (knn_dist.std() + 1e-10)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)

        # Mentor: 7D -> 1 (weight). Same architecture for reg and clf — the
        # mentor only consumes loss scalars + kNN distance, both of which are
        # well-defined regardless of task.
        mentor = nn.Sequential(
            nn.Linear(7, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        ).to(device)
        mentor_opt = torch.optim.Adam(mentor.parameters(), lr=self.lr * 0.1)

        X_tensor = torch.from_numpy(X_t)
        X_val_tensor = torch.from_numpy(X_v)

        def _per_sample(net_out, y_target):
            if self.is_clf:
                return nn.functional.cross_entropy(net_out, y_target, reduction='none')
            return (net_out.squeeze(-1) - y_target) ** 2

        def _val_loss(net_out, y_target):
            if self.is_clf:
                return nn.functional.cross_entropy(net_out, y_target)
            return nn.functional.mse_loss(net_out.squeeze(-1), y_target)

        loss_history = np.zeros((n, 5), dtype=np.float32)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        mentor_weights = None

        for epoch in range(self.epochs):
            # Current per-sample loss
            net.eval()
            with torch.no_grad():
                out_all = net(X_tensor.to(device))
                current_loss = _per_sample(out_all, y_tensor.to(device)).cpu().numpy()

            loss_history[:, :-1] = loss_history[:, 1:]
            loss_history[:, -1] = current_loss

            mentor_features = np.column_stack([
                current_loss,
                loss_history,
                knn_dist,
            ]).astype(np.float32)
            mentor_tensor = torch.from_numpy(mentor_features)

            mentor.eval()
            with torch.no_grad():
                mentor_weights = mentor(mentor_tensor.to(device)).squeeze(-1).cpu()

            # Weighted train step
            net.train()
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                xb = X_tensor[batch_idx].to(device)
                yb = y_tensor[batch_idx].to(device)
                wb = mentor_weights[batch_idx].to(device)

                optimizer.zero_grad()
                out = net(xb)
                per = _per_sample(out, yb)
                loss = (wb * per).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

            # Mentor update — learn a self-paced curriculum instead of
            # minimising (weight * loss). The old objective's trivial optimum
            # was all-zero weights (down-weight everything), so the mentor
            # collapsed to a useless constant. Here the mentor is trained to
            # REPRODUCE a self-paced target from its 7D features: emphasise
            # low-loss (likely-clean) samples, with the loss threshold annealed
            # upward over training so more samples enter the curriculum as the
            # net improves. This target has a non-degenerate optimum.
            frac = min(1.0, 0.5 + 0.5 * epoch / max(1, self.epochs - 1))
            thresh = np.quantile(current_loss, frac)
            spread = current_loss.std() + 1e-8
            target_w = 1.0 / (1.0 + np.exp((current_loss - thresh) / spread))
            target_tensor = torch.from_numpy(target_w.astype(np.float32)).to(device)

            mentor.train()
            mw = mentor(mentor_tensor.to(device)).squeeze(-1)
            mentor_loss = nn.functional.mse_loss(mw, target_tensor)
            mentor_opt.zero_grad()
            mentor_loss.backward()
            mentor_opt.step()

            # Val early stopping
            net.eval()
            with torch.no_grad():
                val_out = net(X_val_tensor.to(device))
                v_loss = _val_loss(val_out, y_val_tensor.to(device)).item()

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
            out = net(X_val_tensor.to(device))
            if self.is_clf:
                y_pred = _clf_pred_from_logits(out.cpu().numpy(), classes_)
            else:
                y_pred = out.squeeze(-1).cpu().numpy() * y_std + y_mean

        return {
            'y_pred': y_pred,
            'metadata': {
                'final_epoch': epoch + 1,
                'mentor_weight_mean': float(mentor_weights.mean()) if mentor_weights is not None else None,
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
        n = len(y_train)
        device = torch.device('cpu')

        if self.is_clf:
            classes_ = self._resolve_classes(y_train)
            label_map = {c: i for i, c in enumerate(classes_)}
            y_int = np.array([label_map[c] for c in np.asarray(y_train).ravel()],
                             dtype=np.int64)
            y_val_int = np.array(
                [label_map.get(c, 0) for c in np.asarray(y_val).ravel()],
                dtype=np.int64)
            y_tensor = torch.from_numpy(y_int)
            y_val_tensor = torch.from_numpy(y_val_int)
            n_classes = len(classes_)
            net = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout,
                             output_dim=n_classes).to(device)
        else:
            y_norm, y_mean, y_std = _z_normalize(y_train)
            y_val_norm = (np.asarray(y_val, dtype=np.float32) - y_mean) / y_std
            y_tensor = torch.from_numpy(y_norm)
            y_val_tensor = torch.from_numpy(y_val_norm)
            net = _build_dnn(X_t.shape[1], self.hidden_sizes, self.dropout).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)

        X_tensor = torch.from_numpy(X_t)
        X_val_tensor = torch.from_numpy(X_v)

        def _per_sample(net_out, y_target):
            if self.is_clf:
                return nn.functional.cross_entropy(net_out, y_target, reduction='none')
            return (net_out.squeeze(-1) - y_target) ** 2

        def _val_loss(net_out, y_target):
            if self.is_clf:
                return nn.functional.cross_entropy(net_out, y_target)
            return nn.functional.mse_loss(net_out.squeeze(-1), y_target)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        w_tensor = torch.ones(n, dtype=torch.float32)
        weights = np.ones(n, dtype=np.float32)

        for epoch in range(self.epochs):
            # Consistency scores every 5 epochs
            if epoch % 5 == 0:
                net.eval()
                with torch.no_grad():
                    base_out = net(X_tensor.to(device))
                    per_sample_loss = _per_sample(
                        base_out, y_tensor.to(device)).cpu().numpy()

                    # For consistency, measure prediction variance under
                    # perturbation. For reg: variance of scalar prediction.
                    # For clf: variance of P(true class) across perturbations.
                    rng = np.random.RandomState(self.random_state + epoch)
                    perturbed_signal = []
                    for _ in range(self.n_perturbations):
                        noise = rng.normal(0, self.perturbation_sigma,
                                           X_t.shape).astype(np.float32)
                        X_pert = torch.from_numpy(X_t + noise)
                        out_p = net(X_pert.to(device))
                        if self.is_clf:
                            proba = nn.functional.softmax(out_p, dim=1)
                            # P(true class) per sample, shape (n,)
                            true_idx = y_tensor.to(device)
                            sig = proba.gather(1, true_idx.unsqueeze(1)).squeeze(1)
                            perturbed_signal.append(sig.cpu().numpy())
                        else:
                            perturbed_signal.append(
                                out_p.squeeze(-1).cpu().numpy())

                    pred_variance = np.var(perturbed_signal, axis=0)

                # Noise score: high loss rank + low variance rank
                loss_rank = np.argsort(np.argsort(per_sample_loss)) / n
                var_rank = np.argsort(np.argsort(pred_variance)) / n
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
                out = net(xb)
                per = _per_sample(out, yb)
                loss = (wb * per).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

            # Val early stopping
            net.eval()
            with torch.no_grad():
                val_out = net(X_val_tensor.to(device))
                val_loss = _val_loss(val_out, y_val_tensor.to(device)).item()

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
            out = net(X_val_tensor.to(device))
            if self.is_clf:
                y_pred = _clf_pred_from_logits(out.cpu().numpy(), classes_)
            else:
                y_pred = out.squeeze(-1).cpu().numpy() * y_std + y_mean

        return {
            'y_pred': y_pred,
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
        if self.is_clf:
            raise NotImplementedError(
                "HeteroscedasticGP models continuous per-sample noise variance "
                "and has no loss-swap classification analog. The clf equivalent "
                "is a class-confusion / transition-matrix formulation, which is "
                "a different method entirely. n09 must gate this method to "
                "regression-only datasets (see REGRESSION_ONLY_LNL).")
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
        if self.is_clf:
            raise NotImplementedError(
                "EvidentialKernel uses Normal-Inverse-Gamma loss, which only "
                "makes sense for continuous targets. The classification analog "
                "is evidential classification (Sensoy et al. 2018, Dirichlet "
                "parameterization) — a different method entirely. n09 must "
                "gate this method to regression-only datasets (see "
                "REGRESSION_ONLY_LNL).")
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

# Methods that have no classification analog and will raise NotImplementedError
# on is_clf=True. The 10 other methods all support clf via MSE→CE swap.
REGRESSION_ONLY_LNL = {'heteroscedastic_gp', 'evidential_kernel'}


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
