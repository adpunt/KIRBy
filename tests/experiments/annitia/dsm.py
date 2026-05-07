"""Deep Survival Machines (DSM) for ANNITIA.

DSM is a parametric mixture-of-distributions survival NN (Nagpal et al. 2021,
auton-survival). Two flavours:

    SurvivalDSM      — single-risk, mirrors the SURVIVAL_MODELS interface
                       (fit(X, y_struct) / predict_risk(X)). Drop-in alongside
                       SurvivalCoxPH / SurvivalCoxnet / SurvivalRSF.

    DeepSurvivalMachinesCR — multi-event competing-risks variant. Joint head
                       over (hepatic, death). Mirrors deephit.DeepHitMT's
                       interface so run_dsm.py can swap in cleanly.

Both wrappers fall back to a small hand-rolled torch implementation if
auton-survival isn't installed on the server. The fallback is a Weibull
mixture-of-K head with shared trunk; it's a faithful but minimal DSM, NOT a
full reproduction of the auton-survival library.

Risk score convention (matches the rest of ANNITIA):
    higher = worse prognosis. We return the negative survival probability at
    the median pooled-duration horizon (= 1 - S(τ)).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers shared by both flavours
# ---------------------------------------------------------------------------
def _drop_invalid_single(X: np.ndarray, y_struct: np.ndarray):
    keep = ~np.isnan(y_struct['time']) & np.isfinite(y_struct['time'])
    return (np.asarray(X, dtype=np.float64)[keep],
            y_struct['event'][keep].astype(bool),
            y_struct['time'][keep].astype(np.float64),
            keep)


def _drop_invalid_joint(X, y_hep, y_death):
    hep_ok = ~np.isnan(y_hep['time']) & np.isfinite(y_hep['time'])
    death_ok = ~np.isnan(y_death['time']) & np.isfinite(y_death['time'])
    keep = hep_ok & death_ok
    return (np.asarray(X, dtype=np.float64)[keep],
            y_hep['event'][keep].astype(bool),
            y_hep['time'][keep].astype(np.float64),
            y_death['event'][keep].astype(bool),
            y_death['time'][keep].astype(np.float64),
            keep)


def _horizon_quantile(t: np.ndarray, q: float) -> float:
    pos = t[t > 0]
    if pos.size == 0:
        return 1.0
    return float(np.quantile(pos, q))


# ---------------------------------------------------------------------------
# Single-risk DSM
# ---------------------------------------------------------------------------
class SurvivalDSM:
    """Single-risk Deep Survival Machines wrapper.

    Tries auton-survival first. Falls back to a torch Weibull-mixture
    implementation if the library is missing on the host.
    """

    def __init__(self,
                 k: int = 3,
                 hidden: Tuple[int, ...] = (64, 64),
                 dropout: float = 0.3,
                 lr: float = 1e-3,
                 batch_size: int = 64,
                 epochs: int = 200,
                 patience: int = 20,
                 horizon_quantile: float = 0.5,
                 distribution: str = 'Weibull',
                 random_state: int = 42,
                 device: str = 'cpu',
                 verbose: bool = False):
        self.k = k
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.horizon_quantile = horizon_quantile
        self.distribution = distribution
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

        self._scaler = None
        self._horizon = None
        self._backend = None  # 'auton' or 'torch'
        self._model = None    # auton DeepSurvivalMachines instance OR torch Module

    # ----- auton-survival path -------------------------------------------
    def _try_fit_auton(self, X, e, t):
        try:
            from auton_survival.models.dsm import DeepSurvivalMachines
        except ImportError:
            return False
        m = DeepSurvivalMachines(
            k=self.k,
            distribution=self.distribution,
            layers=list(self.hidden),
        )
        m.fit(X, t, e,
              iters=self.epochs, learning_rate=self.lr,
              batch_size=self.batch_size, vsize=0.15)
        self._backend = 'auton'
        self._model = m
        return True

    # ----- torch fallback -------------------------------------------------
    def _fit_torch_fallback(self, X, e, t):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.random_state)

        n, d = X.shape
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        et = torch.tensor(e.astype(np.float32), device=self.device)
        tt = torch.tensor(np.maximum(t, 1e-3), dtype=torch.float32,
                          device=self.device)

        layers = []
        prev = d
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        trunk = nn.Sequential(*layers)
        # Mixture head: K shape, K scale, K weight logits.
        head = nn.Linear(prev, 3 * self.k)
        net = nn.Sequential(trunk, head).to(self.device)

        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)

        def _params(out):
            shape = torch.nn.functional.softplus(out[:, :self.k]) + 1e-3
            scale = torch.nn.functional.softplus(out[:, self.k:2*self.k]) + 1e-3
            logw = torch.log_softmax(out[:, 2*self.k:], dim=-1)
            return shape, scale, logw

        def _loglik(shape, scale, logw, time, event):
            # Weibull components: f = (k/lam) (t/lam)^(k-1) exp(-(t/lam)^k)
            #                    S = exp(-(t/lam)^k)
            # Per-component log-likelihoods, then logsumexp with mixture weights
            tt = time.unsqueeze(1)  # (n, 1)
            ratio = tt / scale       # (n, k)
            log_S = -ratio ** shape
            log_f = (torch.log(shape) - torch.log(scale)
                     + (shape - 1.0) * (torch.log(tt) - torch.log(scale))
                     + log_S)
            # Mix
            log_mix_f = torch.logsumexp(logw + log_f, dim=-1)
            log_mix_S = torch.logsumexp(logw + log_S, dim=-1)
            ll = event * log_mix_f + (1.0 - event) * log_mix_S
            return -ll.mean()

        best_loss = float('inf')
        best_state = None
        bad = 0
        idx = np.arange(n)
        for ep in range(self.epochs):
            np.random.RandomState(self.random_state + ep).shuffle(idx)
            net.train()
            losses = []
            for i in range(0, n, self.batch_size):
                bs = idx[i:i + self.batch_size]
                bt = torch.tensor(bs, dtype=torch.long, device=self.device)
                opt.zero_grad()
                out = net(Xt[bt])
                shape, scale, logw = _params(out)
                loss = _loglik(shape, scale, logw, tt[bt], et[bt])
                loss.backward()
                opt.step()
                losses.append(float(loss))
            mean_loss = float(np.mean(losses))
            if mean_loss < best_loss - 1e-4:
                best_loss = mean_loss
                best_state = {k_: v.detach().clone()
                              for k_, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
            if self.verbose and ep % 20 == 0:
                print(f'  [DSM-torch] ep={ep} loss={mean_loss:.4f}')
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self._backend = 'torch'
        self._model = (net, _params)

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        from sklearn.preprocessing import StandardScaler
        X, e, t, _ = _drop_invalid_single(X, y_struct)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._horizon = _horizon_quantile(t, self.horizon_quantile)

        if not self._try_fit_auton(Xs, e, t):
            self._fit_torch_fallback(Xs, e, t)
        return self

    def _predict_S_at_horizon(self, X: np.ndarray) -> np.ndarray:
        Xs = self._scaler.transform(np.asarray(X, dtype=np.float64))
        if self._backend == 'auton':
            # auton returns survival at requested time(s)
            S = self._model.predict_survival(Xs, [self._horizon])
            S = np.asarray(S, dtype=np.float64).reshape(-1)
            return S
        # torch fallback
        import torch
        net, _params = self._model
        with torch.no_grad():
            Xt = torch.tensor(Xs, dtype=torch.float32, device=self.device)
            out = net(Xt)
            shape, scale, logw = _params(out)
            tt = torch.tensor(np.array([[self._horizon]], dtype=np.float32),
                              device=self.device)
            ratio = tt / scale
            log_S = -ratio ** shape
            log_mix_S = torch.logsumexp(logw + log_S, dim=-1)
            S = torch.exp(log_mix_S).cpu().numpy()
        return np.asarray(S, dtype=np.float64).reshape(-1)

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        S = self._predict_S_at_horizon(X)
        return 1.0 - S


# ---------------------------------------------------------------------------
# Competing-risks DSM (joint head over hepatic + death)
# ---------------------------------------------------------------------------
class DeepSurvivalMachinesCR:
    """Joint multi-event DSM. Two cause-specific heads share a trunk, so
    the 47-event hepatic endpoint borrows representational strength from the
    250-event death endpoint.

    Interface mirrors deephit.DeepHitMT:
        m.fit(X, y_hep_struct, y_death_struct)
        rh, rd = m.predict_risk(X)

    Fallback only — auton-survival's DSM does single-risk natively. The CR
    extension is implemented from scratch in torch (Weibull mixture per
    cause + cause-specific weight head).
    """

    def __init__(self,
                 k: int = 3,
                 hidden: Tuple[int, ...] = (64, 64),
                 dropout: float = 0.3,
                 lr: float = 1e-3,
                 batch_size: int = 64,
                 epochs: int = 200,
                 patience: int = 20,
                 horizon_quantile: float = 0.5,
                 random_state: int = 42,
                 device: str = 'cpu',
                 verbose: bool = False):
        self.k = k
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.horizon_quantile = horizon_quantile
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

        self._scaler = None
        self._horizons = None  # per-cause horizon
        self._net = None
        self._n_causes = 2

    def _build_net(self, d):
        import torch.nn as nn
        layers = []
        prev = d
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        trunk = nn.Sequential(*layers)
        # 2 causes × (K shape, K scale, K weight logits) + 1 cause-selection
        # Cause-selection head softmaxes over (cause1, cause2, censored).
        head_per_cause = nn.Linear(prev, 3 * self.k)
        cause_head = nn.Linear(prev, self._n_causes + 1)
        return trunk, head_per_cause, cause_head

    def fit(self, X, y_hep, y_death):
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X, e_h, t_h, e_d, t_d, _ = _drop_invalid_joint(X, y_hep, y_death)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._horizons = (
            _horizon_quantile(t_h, self.horizon_quantile),
            _horizon_quantile(t_d, self.horizon_quantile),
        )

        n, d = Xs.shape
        Xt = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        # Per-cause time + event tensors
        t_h_t = torch.tensor(np.maximum(t_h, 1e-3), dtype=torch.float32,
                             device=self.device)
        t_d_t = torch.tensor(np.maximum(t_d, 1e-3), dtype=torch.float32,
                             device=self.device)
        e_h_t = torch.tensor(e_h.astype(np.float32), device=self.device)
        e_d_t = torch.tensor(e_d.astype(np.float32), device=self.device)

        trunk, head_h, head_d = self._build_net(d)
        # Replace head_d with a fresh-per-cause head; we already have two.
        _, head_h, head_d = self._build_net(d)
        # Two separate per-cause heads sharing the trunk:
        head_h = nn.Linear(self.hidden[-1], 3 * self.k)
        head_d = nn.Linear(self.hidden[-1], 3 * self.k)
        net = nn.ModuleDict(dict(trunk=trunk, head_h=head_h, head_d=head_d)).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)

        def _per_cause(out):
            K = self.k
            shape = torch.nn.functional.softplus(out[:, :K]) + 1e-3
            scale = torch.nn.functional.softplus(out[:, K:2*K]) + 1e-3
            logw = torch.log_softmax(out[:, 2*K:], dim=-1)
            return shape, scale, logw

        def _cause_logS_logf(out_cause, time):
            shape, scale, logw = _per_cause(out_cause)
            tt = time.unsqueeze(1)
            ratio = tt / scale
            log_S = -ratio ** shape
            log_f = (torch.log(shape) - torch.log(scale)
                     + (shape - 1.0) * (torch.log(tt) - torch.log(scale))
                     + log_S)
            log_mix_f = torch.logsumexp(logw + log_f, dim=-1)
            log_mix_S = torch.logsumexp(logw + log_S, dim=-1)
            return log_mix_S, log_mix_f

        best_loss = float('inf')
        best_state = None
        bad = 0
        idx = np.arange(n)

        for ep in range(self.epochs):
            np.random.RandomState(self.random_state + ep).shuffle(idx)
            net.train()
            losses = []
            for i in range(0, n, self.batch_size):
                bs = idx[i:i + self.batch_size]
                bt = torch.tensor(bs, dtype=torch.long, device=self.device)
                opt.zero_grad()
                rep = net['trunk'](Xt[bt])
                oh = net['head_h'](rep)
                od = net['head_d'](rep)
                logS_h, logf_h = _cause_logS_logf(oh, t_h_t[bt])
                logS_d, logf_d = _cause_logS_logf(od, t_d_t[bt])
                # Per-cause likelihood: borrowed-strength via shared trunk.
                # The two endpoints are NOT competing in the strict sense
                # (a patient can have hepatic event AND later die), so we sum
                # the two cause-specific log-likelihoods rather than enforcing
                # mutual exclusivity (DeepHit-MT-style).
                ll_h = e_h_t[bt] * logf_h + (1.0 - e_h_t[bt]) * logS_h
                ll_d = e_d_t[bt] * logf_d + (1.0 - e_d_t[bt]) * logS_d
                loss = -(ll_h.mean() + ll_d.mean()) * 0.5
                loss.backward()
                opt.step()
                losses.append(float(loss))
            mean_loss = float(np.mean(losses))
            if mean_loss < best_loss - 1e-4:
                best_loss = mean_loss
                best_state = {k_: v.detach().clone()
                              for k_, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
            if self.verbose and ep % 20 == 0:
                print(f'  [DSM-CR] ep={ep} loss={mean_loss:.4f}')
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self._net = (net, _per_cause)
        return self

    def predict_risk(self, X):
        import torch
        net, _per_cause = self._net
        Xs = self._scaler.transform(np.asarray(X, dtype=np.float64))
        Xt = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        h_h, h_d = self._horizons
        with torch.no_grad():
            rep = net['trunk'](Xt)
            oh = net['head_h'](rep)
            od = net['head_d'](rep)

            def _S_at(out, horizon):
                shape, scale, logw = _per_cause(out)
                tt = torch.full((out.size(0), 1), horizon,
                                dtype=torch.float32, device=self.device)
                ratio = tt / scale
                log_S = -ratio ** shape
                log_mix_S = torch.logsumexp(logw + log_S, dim=-1)
                return torch.exp(log_mix_S).cpu().numpy()

            S_h = _S_at(oh, h_h)
            S_d = _S_at(od, h_d)
        return 1.0 - S_h, 1.0 - S_d
