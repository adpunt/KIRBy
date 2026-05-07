"""Cross-endpoint transfer learning: pretrain on one endpoint, fine-tune on another.

Motivation: hep has 47 events / 1253 patients (high censoring, low signal),
death has 76 events / 1253 patients. Pretraining a representation on the
endpoint with more events, then fine-tuning on the target, is the classical
transfer-learning answer to small-events data. This module provides three
sklearn-style survival wrappers that take the source endpoint's labels at
construction time and the target endpoint's labels at .fit() time.

Wrappers:
    TabSurvTransferEndpoint         — torch FT-Transformer encoder transfer
    CatBoostAFTTransferEndpoint     — CatBoost-AFT warm-start via init_model
    MLPTransferEndpoint             — small MLP encoder transfer

Usage example (in experiment_branches.py):

    src_y = train['death']['y_struct']
    src_valid = train['death']['valid_mask']
    _run('tabsurv-D2H',
         lambda: TabSurvTransferEndpoint(
             source_y=src_y, source_valid=src_valid, K=10),
         X_tr, X_va, 'hepatic', primary)

All wrappers are best-effort: if a heavy backend (torch / catboost) isn't
available, fit() raises ImportError so the runner skips the condition.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ._survival_discretiser import SurvivalDiscretiser


class TabSurvTransferEndpoint:
    """FT-Transformer encoder transfer: pretrain encoder on `source_y`,
    reset head, fine-tune on the target endpoint.

    Strategy:
        Phase 1 — pretrain full network (encoder + head) on source endpoint
                  for `pretrain_epochs` epochs.
        Phase 2 — reset head (target endpoint may have different bin edges),
                  freeze encoder, train head only for `head_epochs` epochs.
        Phase 3 — unfreeze encoder, joint fine-tune at `finetune_lr` for
                  `finetune_epochs` epochs.

    K (number of bins) is shared between source and target so the head
    output dim is consistent across phases.
    """

    def __init__(self, source_y: np.ndarray,
                 source_valid: Optional[np.ndarray] = None,
                 K: int = 10, d_emb: int = 64,
                 n_layers: int = 2, n_heads: int = 8, dropout: float = 0.2,
                 pretrain_epochs: int = 60, head_epochs: int = 20,
                 finetune_epochs: int = 40,
                 pretrain_lr: float = 1e-3, head_lr: float = 1e-3,
                 finetune_lr: float = 1e-4,
                 batch_size: int = 64, weight_decay: float = 1e-4,
                 patience: int = 10, seed: int = 42, device: str = 'cpu'):
        self.source_y = source_y
        self.source_valid = (np.ones(len(source_y), dtype=bool)
                             if source_valid is None
                             else np.asarray(source_valid, dtype=bool))
        self.K = int(K); self.d_emb = int(d_emb)
        self.n_layers = int(n_layers); self.n_heads = int(n_heads)
        self.dropout = float(dropout)
        self.pretrain_epochs = int(pretrain_epochs)
        self.head_epochs = int(head_epochs)
        self.finetune_epochs = int(finetune_epochs)
        self.pretrain_lr = float(pretrain_lr)
        self.head_lr = float(head_lr)
        self.finetune_lr = float(finetune_lr)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.seed = int(seed); self.device = device
        self._net = None
        self._target_disc: SurvivalDiscretiser | None = None
        self._mean = None; self._std = None

    def _build(self, d_in: int, n_bins: int):
        try:
            import torch.nn as nn
            import torch
        except ImportError as e:
            raise ImportError('torch not installed') from e

        class _FTLite(nn.Module):
            def __init__(self, d_in, d_emb, n_layers, n_heads, dropout, n_bins):
                super().__init__()
                self.feat_proj = nn.Linear(1, d_emb)
                self.feat_bias = nn.Parameter(torch.randn(d_in, d_emb) * 0.02)
                self.cls = nn.Parameter(torch.randn(1, 1, d_emb) * 0.02)
                enc = nn.TransformerEncoderLayer(
                    d_model=d_emb, nhead=n_heads, dim_feedforward=d_emb * 2,
                    dropout=dropout, batch_first=True, activation='gelu')
                self.tr = nn.TransformerEncoder(enc, num_layers=n_layers)
                self.head = nn.Sequential(nn.LayerNorm(d_emb),
                                          nn.Linear(d_emb, n_bins))

            def encode(self, x):
                t = self.feat_proj(x.unsqueeze(-1)) + self.feat_bias.unsqueeze(0)
                cls = self.cls.expand(t.size(0), -1, -1)
                t = torch.cat([cls, t], dim=1)
                t = self.tr(t)
                return t[:, 0]

            def forward(self, x):
                return self.head(self.encode(x))

        return _FTLite(d_in, self.d_emb, self.n_layers, self.n_heads,
                       self.dropout, n_bins)

    def _train_loop(self, net, opt, Xt, Yt, Mt, n_epochs, label, params=None):
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise ImportError('torch not installed') from e
        import sys, time as _time
        bce = nn.BCEWithLogitsLoss(reduction='none')
        n = Xt.shape[0]; idx = np.arange(n)
        best, bad, best_state = float('inf'), 0, None
        _t0 = _time.time()
        print(f'    [tabsurv-D2H] {label} n={n} epochs={n_epochs}', flush=True)
        for ep in range(n_epochs):
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
                    print(f'    [tabsurv-D2H] {label} early stop ep={ep+1}', flush=True)
                    break
            if (ep + 1) % 10 == 0 or ep == n_epochs - 1:
                print(f'    [tabsurv-D2H] {label} ep={ep+1}/{n_epochs} loss={ml:.4f} elapsed={_time.time()-_t0:.1f}s', flush=True)
        if best_state is not None:
            net.load_state_dict(best_state)

    def fit(self, X, y_target):
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise ImportError('torch not installed') from e
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float64)
        # Standardise on the union of source-valid and target rows since
        # this is a transductive feature transform (no test leakage in CV).
        self._mean = X.mean(axis=0); self._std = X.std(axis=0) + 1e-6
        Xs = ((X - self._mean) / self._std).astype(np.float32)

        # Source labels (death, when called from hep target)
        src_disc = SurvivalDiscretiser(K=self.K, strategy='equal_event').fit(
            self.source_y['time'], self.source_y['event'])
        Y_src = src_disc.labels(self.source_y['time'], self.source_y['event']).astype(np.float32)
        # Target labels
        time_t = np.asarray(y_target['time'], dtype=np.float64)
        event_t = np.asarray(y_target['event'], dtype=bool)
        self._target_disc = SurvivalDiscretiser(K=self.K, strategy='equal_event').fit(time_t, event_t)
        Y_tgt = self._target_disc.labels(time_t, event_t).astype(np.float32)

        # SurvivalDiscretiser may collapse to fewer than K bins on small folds
        # (equal_event with few events). The head needs to fit BOTH endpoints,
        # so use the minimum of the two bin counts and truncate both label
        # matrices to that width.
        n_bins = min(Y_src.shape[1], Y_tgt.shape[1])
        Y_src = Y_src[:, :n_bins]
        Y_tgt = Y_tgt[:, :n_bins]
        M_src = (~np.isnan(Y_src)).astype(np.float32)
        Y_src0 = np.nan_to_num(Y_src, nan=0.0)
        keep_src = self.source_valid & np.isfinite(self.source_y['time'])
        Xs_src = Xs[keep_src]; Y_src0 = Y_src0[keep_src]; M_src = M_src[keep_src]
        M_tgt = (~np.isnan(Y_tgt)).astype(np.float32)
        Y_tgt0 = np.nan_to_num(Y_tgt, nan=0.0)

        d_in = Xs.shape[1]
        net = self._build(d_in, n_bins).to(self.device)

        # ---- Phase 1: pretrain on source (death) ----
        Xt_src = torch.tensor(Xs_src, dtype=torch.float32, device=self.device)
        Yt_src = torch.tensor(Y_src0, dtype=torch.float32, device=self.device)
        Mt_src = torch.tensor(M_src, dtype=torch.float32, device=self.device)
        opt1 = torch.optim.AdamW(net.parameters(), lr=self.pretrain_lr,
                                 weight_decay=self.weight_decay)
        self._train_loop(net, opt1, Xt_src, Yt_src, Mt_src,
                         self.pretrain_epochs, 'phase1-pretrain')

        # ---- Phase 2: reset head, freeze encoder, train head on target ----
        # Reset head weights (Linear inside the Sequential)
        for m in net.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        for p in net.parameters():
            p.requires_grad = False
        for p in net.head.parameters():
            p.requires_grad = True
        Xt_tgt = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        Yt_tgt = torch.tensor(Y_tgt0, dtype=torch.float32, device=self.device)
        Mt_tgt = torch.tensor(M_tgt, dtype=torch.float32, device=self.device)
        opt2 = torch.optim.AdamW(net.head.parameters(), lr=self.head_lr,
                                 weight_decay=self.weight_decay)
        self._train_loop(net, opt2, Xt_tgt, Yt_tgt, Mt_tgt,
                         self.head_epochs, 'phase2-headonly')

        # ---- Phase 3: unfreeze, joint fine-tune at low LR ----
        for p in net.parameters(): p.requires_grad = True
        opt3 = torch.optim.AdamW(net.parameters(), lr=self.finetune_lr,
                                 weight_decay=self.weight_decay)
        self._train_loop(net, opt3, Xt_tgt, Yt_tgt, Mt_tgt,
                         self.finetune_epochs, 'phase3-finetune')

        net.eval()
        self._net = net
        return self

    def predict_risk(self, X):
        if self._net is None or self._target_disc is None:
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
        return self._target_disc.survival_to_risk(p)


class CatBoostAFTTransferEndpoint:
    """CatBoost-AFT warm-start: fit on source endpoint, then continue
    training on target using `init_model=`. Standard CatBoost API.

    The objective stays SurvivalAft for both phases; only the labels change.
    """

    def __init__(self, source_y: np.ndarray,
                 source_valid: Optional[np.ndarray] = None,
                 pretrain_iters: int = 500, finetune_iters: int = 200,
                 learning_rate: float = 0.05, depth: int = 6,
                 seed: int = 42):
        self.source_y = source_y
        self.source_valid = (np.ones(len(source_y), dtype=bool)
                             if source_valid is None
                             else np.asarray(source_valid, dtype=bool))
        self.pretrain_iters = int(pretrain_iters)
        self.finetune_iters = int(finetune_iters)
        self.learning_rate = float(learning_rate)
        self.depth = int(depth); self.seed = int(seed)
        self._model = None

    @staticmethod
    def _aft_targets(y_struct):
        # CatBoost SurvivalAft: lower=time for events, upper=time for events;
        # for censored: lower=time, upper=+inf.
        time = np.asarray(y_struct['time'], dtype=np.float64)
        event = np.asarray(y_struct['event'], dtype=bool)
        finite = np.isfinite(time) & (time > 0)
        # Replace non-finite with a sentinel; pool will be filtered by caller.
        t = np.where(finite, time, 1e-6)
        lo = t.copy()
        hi = np.where(event, t, np.inf)
        return lo, hi, finite

    def fit(self, X, y_target):
        try:
            from catboost import CatBoostRegressor, Pool
        except ImportError as e:
            raise ImportError('catboost not installed') from e
        X = np.asarray(X, dtype=np.float64)
        n_src = len(self.source_y)
        if X.shape[0] != n_src:
            raise ValueError(
                f'X rows ({X.shape[0]}) != source_y len ({n_src}); '
                f'pretrain expects same X for both endpoints')

        # ---- Phase 1: pretrain on source ----
        lo_s, hi_s, fin_s = self._aft_targets(self.source_y)
        keep_s = self.source_valid & fin_s
        pool_src = Pool(X[keep_s], label=np.column_stack([lo_s[keep_s], hi_s[keep_s]]))
        m_src = CatBoostRegressor(
            iterations=self.pretrain_iters,
            learning_rate=self.learning_rate, depth=self.depth,
            loss_function='SurvivalAft:dist=Normal',
            eval_metric='SurvivalAft', random_seed=self.seed,
            verbose=False, allow_writing_files=False)
        m_src.fit(pool_src)

        # ---- Phase 2: continue on target ----
        lo_t, hi_t, fin_t = self._aft_targets(y_target)
        keep_t = fin_t
        pool_tgt = Pool(X[keep_t], label=np.column_stack([lo_t[keep_t], hi_t[keep_t]]))
        m_tgt = CatBoostRegressor(
            iterations=self.finetune_iters,
            learning_rate=self.learning_rate / 2.0, depth=self.depth,
            loss_function='SurvivalAft:dist=Normal',
            eval_metric='SurvivalAft', random_seed=self.seed,
            verbose=False, allow_writing_files=False)
        m_tgt.fit(pool_tgt, init_model=m_src)
        self._model = m_tgt
        return self

    def predict_risk(self, X):
        if self._model is None:
            raise RuntimeError('not fit')
        # CatBoost AFT predicts log-time; lower → higher risk.
        return -np.asarray(self._model.predict(np.asarray(X, dtype=np.float64)),
                           dtype=np.float64)


class MLPTransferEndpoint:
    """Small MLP encoder transfer, cheap complement to TabSurvTransferEndpoint.

    Architecture:
        Linear(d_in, hidden) - ReLU - Dropout - Linear(hidden, hidden) - ReLU -
        Linear(hidden, K)        # discrete-time hazard head

    Training schedule mirrors TabSurvTransferEndpoint's three phases.
    """

    def __init__(self, source_y, source_valid=None, K: int = 10,
                 hidden: int = 128, dropout: float = 0.3,
                 pretrain_epochs: int = 80, head_epochs: int = 20,
                 finetune_epochs: int = 40,
                 pretrain_lr: float = 1e-3, head_lr: float = 1e-3,
                 finetune_lr: float = 1e-4, batch_size: int = 64,
                 weight_decay: float = 1e-4, patience: int = 12,
                 seed: int = 42, device: str = 'cpu'):
        self.source_y = source_y
        self.source_valid = (np.ones(len(source_y), dtype=bool)
                             if source_valid is None
                             else np.asarray(source_valid, dtype=bool))
        self.K = int(K); self.hidden = int(hidden); self.dropout = float(dropout)
        self.pretrain_epochs = int(pretrain_epochs)
        self.head_epochs = int(head_epochs)
        self.finetune_epochs = int(finetune_epochs)
        self.pretrain_lr = float(pretrain_lr)
        self.head_lr = float(head_lr)
        self.finetune_lr = float(finetune_lr)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.seed = int(seed); self.device = device
        self._net = None
        self._target_disc: SurvivalDiscretiser | None = None
        self._mean = None; self._std = None

    def _build(self, d_in, n_bins):
        try:
            import torch.nn as nn
        except ImportError as e:
            raise ImportError('torch not installed') from e
        return nn.Sequential(
            nn.Linear(d_in, self.hidden), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, n_bins),
        )

    def _train_loop(self, net, opt, Xt, Yt, Mt, n_epochs, label):
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise ImportError('torch not installed') from e
        import time as _time
        bce = nn.BCEWithLogitsLoss(reduction='none')
        n = Xt.shape[0]; idx = np.arange(n)
        best, bad, best_state = float('inf'), 0, None
        _t0 = _time.time()
        print(f'    [mlp-D2H] {label} n={n} epochs={n_epochs}', flush=True)
        for ep in range(n_epochs):
            np.random.RandomState(self.seed + ep).shuffle(idx)
            net.train()
            losses = []
            for i in range(0, n, self.batch_size):
                bs = idx[i:i + self.batch_size]
                bt = torch.tensor(bs, dtype=torch.long, device=self.device)
                opt.zero_grad()
                logits = net(Xt[bt])
                lpb = bce(logits, Yt[bt]); m = Mt[bt]
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
                    print(f'    [mlp-D2H] {label} early stop ep={ep+1}', flush=True)
                    break
            if (ep + 1) % 20 == 0 or ep == n_epochs - 1:
                print(f'    [mlp-D2H] {label} ep={ep+1}/{n_epochs} loss={ml:.4f} elapsed={_time.time()-_t0:.1f}s', flush=True)
        if best_state is not None:
            net.load_state_dict(best_state)

    def fit(self, X, y_target):
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise ImportError('torch not installed') from e
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float64)
        self._mean = X.mean(axis=0); self._std = X.std(axis=0) + 1e-6
        Xs = ((X - self._mean) / self._std).astype(np.float32)

        src_disc = SurvivalDiscretiser(K=self.K, strategy='equal_event').fit(
            self.source_y['time'], self.source_y['event'])
        Y_src = src_disc.labels(self.source_y['time'], self.source_y['event']).astype(np.float32)
        time_t = np.asarray(y_target['time'], dtype=np.float64)
        event_t = np.asarray(y_target['event'], dtype=bool)
        self._target_disc = SurvivalDiscretiser(K=self.K, strategy='equal_event').fit(time_t, event_t)
        Y_tgt = self._target_disc.labels(time_t, event_t).astype(np.float32)

        # Source/target may collapse to fewer than K bins on small folds.
        # Take the min and truncate so the shared head dim is consistent.
        n_bins = min(Y_src.shape[1], Y_tgt.shape[1])
        Y_src = Y_src[:, :n_bins]
        Y_tgt = Y_tgt[:, :n_bins]
        M_src = (~np.isnan(Y_src)).astype(np.float32); Y_src0 = np.nan_to_num(Y_src, nan=0.0)
        keep_src = self.source_valid & np.isfinite(self.source_y['time'])
        Xs_src = Xs[keep_src]; Y_src0 = Y_src0[keep_src]; M_src = M_src[keep_src]
        M_tgt = (~np.isnan(Y_tgt)).astype(np.float32); Y_tgt0 = np.nan_to_num(Y_tgt, nan=0.0)

        net = self._build(Xs.shape[1], n_bins).to(self.device)

        # Phase 1
        Xt_src = torch.tensor(Xs_src, dtype=torch.float32, device=self.device)
        Yt_src = torch.tensor(Y_src0, dtype=torch.float32, device=self.device)
        Mt_src = torch.tensor(M_src, dtype=torch.float32, device=self.device)
        opt1 = torch.optim.AdamW(net.parameters(), lr=self.pretrain_lr,
                                 weight_decay=self.weight_decay)
        self._train_loop(net, opt1, Xt_src, Yt_src, Mt_src,
                         self.pretrain_epochs, 'phase1-pretrain')

        # Phase 2: reset last Linear (head), freeze rest, train head only
        last_linear = None
        for m in net.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None:
            nn.init.xavier_uniform_(last_linear.weight)
            if last_linear.bias is not None: nn.init.zeros_(last_linear.bias)
        for p in net.parameters(): p.requires_grad = False
        if last_linear is not None:
            for p in last_linear.parameters(): p.requires_grad = True
        Xt_tgt = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        Yt_tgt = torch.tensor(Y_tgt0, dtype=torch.float32, device=self.device)
        Mt_tgt = torch.tensor(M_tgt, dtype=torch.float32, device=self.device)
        opt2 = torch.optim.AdamW(
            [p for p in net.parameters() if p.requires_grad],
            lr=self.head_lr, weight_decay=self.weight_decay)
        self._train_loop(net, opt2, Xt_tgt, Yt_tgt, Mt_tgt,
                         self.head_epochs, 'phase2-headonly')

        # Phase 3
        for p in net.parameters(): p.requires_grad = True
        opt3 = torch.optim.AdamW(net.parameters(), lr=self.finetune_lr,
                                 weight_decay=self.weight_decay)
        self._train_loop(net, opt3, Xt_tgt, Yt_tgt, Mt_tgt,
                         self.finetune_epochs, 'phase3-finetune')
        net.eval()
        self._net = net
        return self

    def predict_risk(self, X):
        if self._net is None or self._target_disc is None:
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
        return self._target_disc.survival_to_risk(p)
