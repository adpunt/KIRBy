"""Small MLP discrete-time hazard survival model.

Built so we can pretrain the trunk on 10K+ synthetic patients (P2 protocol)
and fine-tune the full network on 1106 real patients.

Architecture:
    trunk: Linear(d_in, 128) -> ReLU -> Dropout -> Linear(128, 64) -> ReLU
    head:  Linear(64, n_bins) — outputs hazard logits per time bin

Loss: discrete-time hazard NLL (Gensheimer 2019). For each patient with
(time, event), we identify bin b. For all bins j < b the hazard is 0
(survival contribution); at bin b the BCE term is event ∈ {0, 1}.

Risk score = sum of predicted hazards over all bins (cumulative log-risk).

Interface:
    .fit(X, y_struct)
    .predict_risk(X) -> ndarray
    .pretrain(X_synth, y_synth_struct, epochs=...)
    .save_trunk(path) / .load_trunk(path) — for cross-protocol weight reuse
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _y_to_bins(time, event, grid):
    """Discrete-time hazard target: per-patient bin index + binary event flag."""
    n_bins = len(grid) - 1
    idx = np.searchsorted(grid, time, side='right') - 1
    idx = np.clip(idx, 0, n_bins - 1).astype(np.int64)
    return idx, event.astype(np.float32)


class MLPDiscreteHazard:
    """Small PyTorch MLP, discrete-time hazard head, sklearn-style API."""

    def __init__(self, hidden=(128, 64), dropout: float = 0.3,
                 n_bins: int = 10, lr: float = 1e-3, weight_decay: float = 1e-4,
                 epochs: int = 80, finetune_epochs: int = 50,
                 batch_size: int = 64, device: str = 'cpu',
                 early_stop_patience: int = 15, seed: int = 42,
                 verbose: bool = False):
        self.hidden = tuple(hidden)
        self.dropout = float(dropout)
        self.n_bins = int(n_bins)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.finetune_epochs = int(finetune_epochs)
        self.batch_size = int(batch_size)
        self.device = device
        self.early_stop_patience = int(early_stop_patience)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self._scaler = None
        self._trunk = None
        self._head = None
        self._grid = None
        self._d_in = None

    # --- internals ---
    def _build(self, d_in: int):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        layers = []
        prev = d_in
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        trunk = nn.Sequential(*layers)
        head = nn.Linear(prev, self.n_bins)
        self._trunk = trunk.to(self.device)
        self._head = head.to(self.device)
        self._d_in = d_in

    def _make_grid(self, time, event):
        ev_t = time[(event > 0) & np.isfinite(time)]
        if ev_t.size < 5:
            ev_t = time[np.isfinite(time)]
        qs = np.linspace(0, 1, self.n_bins + 1)
        # Use event time quantiles, fall back to overall
        grid = np.quantile(ev_t, qs) if ev_t.size >= self.n_bins else np.quantile(time[np.isfinite(time)], qs)
        # Make strictly increasing
        for i in range(1, len(grid)):
            grid[i] = max(grid[i], grid[i - 1] + 1e-6)
        return grid

    def _train_epoch(self, X, bins, events, optimizer, criterion):
        import torch
        n = X.shape[0]
        idx = np.random.permutation(n)
        losses = []
        for i in range(0, n, self.batch_size):
            b = idx[i:i + self.batch_size]
            xb = torch.from_numpy(X[b]).float().to(self.device)
            bb = torch.from_numpy(bins[b]).long().to(self.device)
            eb = torch.from_numpy(events[b]).float().to(self.device)
            optimizer.zero_grad()
            logits = self._head(self._trunk(xb))
            # Build per-patient mask: bins <= bin_idx contribute
            B = logits.shape[0]; T = logits.shape[1]
            mask = (torch.arange(T, device=logits.device).unsqueeze(0) <= bb.unsqueeze(1)).float()
            event_at_bin = torch.zeros_like(logits)
            event_at_bin[torch.arange(B), bb] = eb
            haz_log = torch.nn.functional.logsigmoid(logits)
            haz_log_neg = torch.nn.functional.logsigmoid(-logits)
            # NLL = -[event_at_bin * log h - (mask - event_at_bin) * log(1-h)]
            ll = (event_at_bin * haz_log + (mask - event_at_bin) * haz_log_neg).sum(dim=1)
            loss = -ll.mean()
            loss.backward(); optimizer.step()
            losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else float('nan')

    def fit(self, X, y_struct, X_val=None, y_val_struct=None,
            epochs: Optional[int] = None,
            keep_grid: bool = False):
        import torch
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        X = np.asarray(X, dtype=np.float64)
        time = np.asarray(y_struct['time'], dtype=np.float64)
        event = np.asarray(y_struct['event']).astype(np.float32)
        valid = np.isfinite(time)
        Xv, tv, ev = X[valid], time[valid], event[valid]
        # Impute + scale (refit each call; for finetune we re-use scaler if loaded)
        if self._scaler is None:
            imp = SimpleImputer(strategy='median').fit(Xv)
            sc = StandardScaler().fit(imp.transform(Xv))
            self._imputer, self._scaler = imp, sc
        Xs = self._scaler.transform(self._imputer.transform(Xv))
        # Grid
        if (self._grid is None) or (not keep_grid):
            self._grid = self._make_grid(tv, ev)
        bins, ev_flag = _y_to_bins(tv, ev, self._grid)
        if self._trunk is None:
            self._build(Xs.shape[1])
        elif self._d_in != Xs.shape[1]:
            raise RuntimeError(f'feature dim {Xs.shape[1]} != pretrain dim {self._d_in}')
        optimizer = torch.optim.AdamW(
            list(self._trunk.parameters()) + list(self._head.parameters()),
            lr=self.lr, weight_decay=self.weight_decay)
        criterion = None
        eps = self.epochs if epochs is None else int(epochs)
        for ep in range(eps):
            loss = self._train_epoch(Xs, bins, ev_flag, optimizer, criterion)
            if self.verbose and (ep % 10 == 0 or ep == eps - 1):
                print(f'  epoch {ep}/{eps}: loss={loss:.4f}')
        return self

    def predict_risk(self, X):
        import torch
        if self._trunk is None:
            raise RuntimeError('model not fit')
        X = np.asarray(X, dtype=np.float64)
        Xs = self._scaler.transform(self._imputer.transform(X))
        with torch.no_grad():
            self._trunk.eval(); self._head.eval()
            xb = torch.from_numpy(Xs).float().to(self.device)
            logits = self._head(self._trunk(xb))
            haz = torch.sigmoid(logits)
            # Cumulative hazard ~ sum of bin hazards = risk score (higher = worse)
            risk = haz.sum(dim=1).cpu().numpy()
            self._trunk.train(); self._head.train()
        return np.asarray(risk, dtype=np.float64)

    def get_trunk_state(self):
        import torch
        return {k: v.detach().cpu().clone() for k, v in self._trunk.state_dict().items()}

    def load_trunk_state(self, state):
        import torch
        if self._trunk is None:
            self._build(self._d_in if self._d_in else next(iter(state.values())).shape[1])
        self._trunk.load_state_dict(state)
