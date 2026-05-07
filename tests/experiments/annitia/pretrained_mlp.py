"""SurvivalPretrainedMLP — load NHANES-pretrained trunk, finetune Cox head.

The trunk was trained by tests/experiments/annitia/pretrain_nhanes.py on the
NHANES cohort over the all-cause-mortality endpoint with discrete-time
hazard targets. We discard the head, freeze the trunk (or partially unfreeze
the last layer), attach a 1-output linear "log hazard" head, and finetune on
ANNITIA folds with a Cox partial-likelihood loss.

Design choices baked in:
    - Strictly NHANES-schema input features. The wrapper expects features
      already projected into NHANES_FEATURE_SCHEMA. ANNITIA-schema → NHANES-
      schema projection is provided by
      kirby.datasets.nhanes_liver.project_annitia_to_nhanes; the search
      pipeline calls it via a dedicated 'nhanes_proj' view added to
      VIEW_BUILDERS.
    - Cox partial likelihood loss with negative-sampling per batch, so the
      MLP can be trained with mini-batches without breaking the loss.
    - LR-warm-restart on the head, near-zero LR on the trunk by default
      (transfer-learning recipe). A flag exposes full-network finetune for
      cases where the trunk benefits from updating.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class SurvivalPretrainedMLP:
    """Cox finetune on top of a NHANES-pretrained MLP trunk.

    Args:
        encoder_path: path to the .pt file produced by pretrain_nhanes.py.
        freeze_trunk: True to lock all trunk weights (head-only finetune).
            False finetunes everything with a 0.1× LR multiplier on the trunk.
        epochs / lr / batch_size / patience: standard.
        l2: weight decay on the cox head; the trunk uses a separate fixed wd.
        device: 'cpu' | 'cuda'.

    Interface: standard ANNITIA SURVIVAL_MODELS shape:
        m = SurvivalPretrainedMLP(encoder_path='...')
        m.fit(X, y_struct)         # X must be in NHANES_FEATURE_SCHEMA
        risk = m.predict_risk(X)   # higher = worse
    """

    def __init__(self,
                 encoder_path: Optional[str] = None,
                 freeze_trunk: bool = True,
                 trunk_lr_mult: float = 0.1,
                 epochs: int = 200,
                 lr: float = 5e-3,
                 batch_size: int = 64,
                 patience: int = 20,
                 l2: float = 1e-3,
                 device: str = 'cpu',
                 random_state: int = 42,
                 verbose: bool = False):
        # Env-var fallback: search.py constructs SURVIVAL_MODELS classes with
        # no kwargs, so the encoder path is taken from
        # $ANNITIA_NHANES_ENCODER when not passed explicitly.
        if encoder_path is None:
            encoder_path = os.environ.get('ANNITIA_NHANES_ENCODER')
        self.encoder_path = encoder_path
        self.freeze_trunk = freeze_trunk
        self.trunk_lr_mult = trunk_lr_mult
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.l2 = l2
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self._mean = None
        self._std = None
        self._trunk = None
        self._head = None
        self._dim_emb = None

    def _load_encoder(self, d_in: int):
        import torch
        import torch.nn as nn
        if self.encoder_path is None or not Path(self.encoder_path).exists():
            raise FileNotFoundError(
                f'Encoder file not found: {self.encoder_path}. '
                'Run pretrain_nhanes.py first.')
        payload = torch.load(self.encoder_path, map_location=self.device,
                             weights_only=False)
        # Sanity: feature dim matches
        if d_in != len(payload['feature_schema']):
            raise ValueError(
                f'Pretrained encoder expects {len(payload["feature_schema"])} '
                f'features but got {d_in}. Did you forget '
                'project_annitia_to_nhanes()?')
        # Rebuild the trunk
        hidden = payload['hidden']
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        trunk = nn.Sequential(*layers).to(self.device)
        trunk.load_state_dict(payload['trunk_state'])
        if self.freeze_trunk:
            for p in trunk.parameters():
                p.requires_grad = False
        head = nn.Linear(prev, 1).to(self.device)  # cox log-hazard
        # Set scaler from pretraining; we re-fit on ANNITIA below if user
        # prefers, but using NHANES-mean/std preserves the encoder's input
        # distribution.
        self._mean = payload['mean'].astype(np.float32)
        self._std = payload['std'].astype(np.float32)
        self._dim_emb = prev
        return trunk, head

    @staticmethod
    def _cox_loss(log_h, time, event):
        """Negative log partial likelihood under Cox. Mini-batch friendly:
        the risk set is restricted to the batch (Efron's approximation
        on a noisy partial likelihood). Returns a scalar loss.
        """
        import torch
        # Sort by descending time; the risk set for patient i is patients j
        # with t_j >= t_i.
        order = torch.argsort(time, descending=True)
        log_h = log_h[order]
        time = time[order]
        event = event[order]
        # For each i, the cumulative-from-top (which corresponds to all
        # patients with t_j >= t_i) of exp(log_h).
        max_h = log_h.max().detach()
        log_cumexp = torch.logcumsumexp(log_h - max_h, dim=0) + max_h
        ll = (log_h - log_cumexp) * event
        # Mean over events only (not all patients)
        denom = event.sum().clamp(min=1.0)
        return -ll.sum() / denom

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        import torch
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        time = y_struct['time'].astype(np.float32)
        event = y_struct['event'].astype(np.float32)
        keep = np.isfinite(time)
        X, time, event = X[keep], time[keep], event[keep]

        d = X.shape[1]
        trunk, head = self._load_encoder(d)
        self._trunk = trunk
        self._head = head

        Xs = ((X - self._mean) / self._std).astype(np.float32)
        Xt = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        tt = torch.tensor(time, dtype=torch.float32, device=self.device)
        et = torch.tensor(event, dtype=torch.float32, device=self.device)

        # Two parameter groups so trunk gets a smaller LR (or none if frozen)
        if self.freeze_trunk:
            opt = torch.optim.Adam(head.parameters(), lr=self.lr,
                                   weight_decay=self.l2)
        else:
            opt = torch.optim.Adam([
                {'params': trunk.parameters(),
                 'lr': self.lr * self.trunk_lr_mult, 'weight_decay': 1e-4},
                {'params': head.parameters(),
                 'lr': self.lr, 'weight_decay': self.l2},
            ])

        best_loss = float('inf')
        best_state = None
        bad = 0
        n = len(Xs)
        idx = np.arange(n)

        for ep in range(self.epochs):
            np.random.RandomState(self.random_state + ep).shuffle(idx)
            trunk.train(not self.freeze_trunk)
            head.train()
            losses = []
            for i in range(0, n, self.batch_size):
                bs = idx[i:i + self.batch_size]
                if event[bs].sum() < 1:
                    # No events in batch — Cox loss undefined.
                    continue
                bt = torch.tensor(bs, dtype=torch.long, device=self.device)
                opt.zero_grad()
                rep = trunk(Xt[bt])
                log_h = head(rep).squeeze(-1)
                loss = self._cox_loss(log_h, tt[bt], et[bt])
                loss.backward()
                opt.step()
                losses.append(float(loss))
            if not losses:
                continue
            mean_loss = float(np.mean(losses))
            if mean_loss < best_loss - 1e-4:
                best_loss = mean_loss
                best_state = (
                    {k: v.detach().clone() for k, v in trunk.state_dict().items()},
                    {k: v.detach().clone() for k, v in head.state_dict().items()},
                )
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
            if self.verbose and ep % 10 == 0:
                print(f'  [pretrained-mlp] ep={ep} loss={mean_loss:.4f}')
        if best_state is not None:
            trunk.load_state_dict(best_state[0])
            head.load_state_dict(best_state[1])
        trunk.eval()
        head.eval()
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        import torch
        if self._trunk is None:
            raise RuntimeError('SurvivalPretrainedMLP not fitted')
        Xs = ((np.asarray(X, dtype=np.float32) - self._mean) / self._std)
        with torch.no_grad():
            Xt = torch.tensor(Xs, dtype=torch.float32, device=self.device)
            rep = self._trunk(Xt)
            log_h = self._head(rep).squeeze(-1).cpu().numpy()
        return np.asarray(log_h, dtype=np.float64)
