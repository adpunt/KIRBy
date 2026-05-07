"""TabSurv — modern tabular NN (FT-Transformer-style) with survival head.

Minimal in-repo implementation since the arxiv-2605.03944 reference code
isn't reliably available. Architecture:

    Input X (n, d_in)
        -> Per-feature LayerNorm + Linear (d_emb)
        -> 2-layer Transformer encoder (8 heads, d_emb)
        -> [CLS] pooling
        -> Linear(d_emb, n_bins)
        -> Discrete-time hazard head, BCE-with-mask loss

Loss is the per-bin BCE used in pretrain_nhanes.py (looser than full
discrete-time NLL but lets us reuse the same SurvivalDiscretiser machinery
and converges fast on n=1253).

Inference: per-bin event-prob → 1 - prod(1 - p_k) risk score (cumulative
incidence by max horizon). Uses SurvivalDiscretiser for binning.

Falls back to ImportError if torch isn't available.
"""
from __future__ import annotations

import numpy as np

from ._survival_discretiser import SurvivalDiscretiser


class TabSurv:
    """FT-Transformer-style tabular survival model.

    Parameters
    ----------
    K : int
        Number of time bins.
    d_emb : int
    n_layers : int
    n_heads : int
    dropout : float
    epochs : int
    batch_size : int
    lr : float
    weight_decay : float
    patience : int
    seed : int
    device : 'cpu' | 'cuda'
    """

    def __init__(self, K: int = 10, d_emb: int = 64,
                 n_layers: int = 2, n_heads: int = 8,
                 dropout: float = 0.2, epochs: int = 60,
                 batch_size: int = 64, lr: float = 1e-3,
                 weight_decay: float = 1e-4, patience: int = 10,
                 seed: int = 42, device: str = 'cpu'):
        self.K = int(K); self.d_emb = int(d_emb)
        self.n_layers = int(n_layers); self.n_heads = int(n_heads)
        self.dropout = float(dropout); self.epochs = int(epochs)
        self.batch_size = int(batch_size); self.lr = float(lr)
        self.weight_decay = float(weight_decay); self.patience = int(patience)
        self.seed = int(seed); self.device = device
        self._net = None
        self._disc: SurvivalDiscretiser | None = None
        self._mean = None
        self._std = None

    def _build(self, d_in: int, n_bins: int):
        try:
            import torch.nn as nn
        except ImportError as e:
            raise ImportError('torch not installed') from e

        # FT-Transformer-lite: per-feature embedding via Linear, learnable CLS,
        # then standard transformer encoder.
        class _FTLite(nn.Module):
            def __init__(self, d_in, d_emb, n_layers, n_heads, dropout, n_bins):
                super().__init__()
                self.feat_proj = nn.Linear(1, d_emb)
                self.feat_bias = nn.Parameter((__import__('torch').randn(d_in, d_emb) * 0.02))
                self.cls = nn.Parameter((__import__('torch').randn(1, 1, d_emb) * 0.02))
                enc = nn.TransformerEncoderLayer(
                    d_model=d_emb, nhead=n_heads, dim_feedforward=d_emb * 2,
                    dropout=dropout, batch_first=True, activation='gelu')
                self.tr = nn.TransformerEncoder(enc, num_layers=n_layers)
                self.head = nn.Sequential(nn.LayerNorm(d_emb),
                                          nn.Linear(d_emb, n_bins))

            def forward(self, x):
                import torch
                # x: (B, d_in)
                t = self.feat_proj(x.unsqueeze(-1))   # (B, d_in, d_emb)
                t = t + self.feat_bias.unsqueeze(0)
                cls = self.cls.expand(t.size(0), -1, -1)
                t = torch.cat([cls, t], dim=1)         # (B, 1+d_in, d_emb)
                t = self.tr(t)
                return self.head(t[:, 0])               # (B, n_bins)

        return _FTLite(d_in, self.d_emb, self.n_layers, self.n_heads,
                       self.dropout, n_bins)

    def fit(self, X, y_struct):
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise ImportError('torch not installed') from e
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float64)
        time = np.asarray(y_struct['time'], dtype=np.float64)
        event = np.asarray(y_struct['event'], dtype=bool)
        # Standardise
        self._mean = X.mean(axis=0); self._std = X.std(axis=0) + 1e-6
        Xs = ((X - self._mean) / self._std).astype(np.float32)
        # Discretise + per-bin label matrix (NaN = skip)
        self._disc = SurvivalDiscretiser(K=self.K, strategy='equal_event').fit(time, event)
        Y = self._disc.labels(time, event).astype(np.float32)
        mask = (~np.isnan(Y)).astype(np.float32)
        Y0 = np.nan_to_num(Y, nan=0.0)

        n_bins = self._disc.n_bins
        net = self._build(Xs.shape[1], n_bins).to(self.device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        bce = nn.BCEWithLogitsLoss(reduction='none')
        Xt = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        Yt = torch.tensor(Y0, dtype=torch.float32, device=self.device)
        Mt = torch.tensor(mask, dtype=torch.float32, device=self.device)
        n = Xs.shape[0]; idx = np.arange(n)
        best, bad, best_state = float('inf'), 0, None
        for ep in range(self.epochs):
            np.random.RandomState(self.seed + ep).shuffle(idx)
            net.train()
            losses = []
            for i in range(0, n, self.batch_size):
                bs = idx[i:i + self.batch_size]
                bt = torch.tensor(bs, dtype=torch.long, device=self.device)
                opt.zero_grad()
                logits = net(Xt[bt])
                lpb = bce(logits, Yt[bt])
                m = Mt[bt]
                loss = (lpb * m).sum() / m.sum().clamp(min=1.0)
                loss.backward(); opt.step()
                losses.append(float(loss))
            ml = float(np.mean(losses))
            if ml < best - 1e-4:
                best, bad = ml, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self._net = net
        return self

    def predict_risk(self, X):
        if self._net is None or self._disc is None:
            raise RuntimeError('not fit')
        try:
            import torch
        except ImportError as e:
            raise ImportError('torch not installed') from e
        X = np.asarray(X, dtype=np.float64)
        Xs = ((X - self._mean) / self._std).astype(np.float32)
        with torch.no_grad():
            logits = self._net(torch.tensor(Xs, dtype=torch.float32,
                                            device=self.device))
            p = torch.sigmoid(logits).cpu().numpy()
        return self._disc.survival_to_risk(p)
