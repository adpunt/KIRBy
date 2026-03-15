"""Sklearn-compatible DNN with optional pre-trained weight initialization.

Used for Imrie-style transfer learning: pre-train on broad ChEMBL data,
then fine-tune on specific downstream tasks.

Critical design constraint: __init__ must NOT instantiate any PyTorch objects.
All torch objects are created in fit() to ensure deepcopy() works (required by
_evaluate_single in find_best_hybrid.py).

Reference:
  Imrie et al. (2018). JCIM 58(11):2319-2330.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class PretrainedDNNRegressor(BaseEstimator, RegressorMixin):
    """DNN with optional pre-trained weight initialization.

    sklearn-compatible: __init__ stores only hyperparams (no torch objects).
    Weights loaded from disk inside fit(), so deepcopy() works safely.

    Args:
        hidden_sizes: Hidden layer sizes.
        dropout: Dropout rate.
        pretrained_path: Path to .pt file with pre-trained weights (str).
        finetune_lr: Learning rate for fine-tuning (10x lower than pre-training).
        freeze_layers: Number of hidden layers to freeze.
            0 = full fine-tuning (default, Imrie's approach).
            -1 = freeze ALL hidden layers, train only output layer (linear probing).
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        patience: Early stopping patience.
        random_state: Random seed.
    """

    def __init__(self, hidden_sizes=(256, 128, 64), dropout=0.2,
                 pretrained_path=None, finetune_lr=0.0001,
                 freeze_layers=0, epochs=200, batch_size=256,
                 patience=20, random_state=42):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.pretrained_path = pretrained_path
        self.finetune_lr = finetune_lr
        self.freeze_layers = freeze_layers
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

    def _load_pretrained(self, net, input_dim):
        """Load pre-trained weights from disk. Validates input_dim match."""
        import torch

        checkpoint = torch.load(self.pretrained_path, map_location='cpu',
                                weights_only=True)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            saved_input_dim = checkpoint.get('input_dim')
            if saved_input_dim is not None and saved_input_dim != input_dim:
                raise ValueError(
                    f"Pre-trained model expects input_dim={saved_input_dim}, "
                    f"but got {input_dim}. Rep mismatch?")
        else:
            state_dict = checkpoint

        # Load all weights (hidden + output) — output dim is (64→1) for both
        # pre-training (pIC50 regression) and downstream (regression)
        net.load_state_dict(state_dict)

    def _freeze_layers(self, net):
        """Freeze hidden layers based on freeze_layers setting."""
        if self.freeze_layers == 0:
            return  # Full fine-tuning

        # Count hidden layer blocks (Linear+ReLU+Dropout = 3 modules each)
        n_hidden = len(self.hidden_sizes)

        if self.freeze_layers == -1:
            # Freeze ALL hidden layers, train only output layer
            layers_to_freeze = n_hidden
        else:
            layers_to_freeze = min(self.freeze_layers, n_hidden)

        # Each hidden block = 3 modules (Linear, ReLU, Dropout)
        # Output layer is the last module in the Sequential
        n_modules_to_freeze = layers_to_freeze * 3
        for mod_idx, module in enumerate(net):
            if mod_idx < n_modules_to_freeze:
                for p in module.parameters():
                    p.requires_grad = False

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
        net = self._build_net(X.shape[1]).to(device)

        # Load pre-trained weights if available
        if self.pretrained_path is not None:
            self._load_pretrained(net, X.shape[1])

        # Freeze layers if requested
        self._freeze_layers(net)

        # Only optimize parameters that require gradients
        trainable_params = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=self.finetune_lr)
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
