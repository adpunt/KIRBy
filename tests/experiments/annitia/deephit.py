"""DeepHit competing-risks multi-task survival model for ANNITIA.

Joint head over (hepatic, death) with a shared MLP trunk. The whole point of
this branch is borrowing strength across the death endpoint to compensate for
the 47 hepatic events. Per-endpoint subset enumeration would defeat the
shared-trunk purpose; this wrapper trains on the all-views concat directly.

Two backends:
    - pycox.models.DeepHit (preferred). Imported lazily.
    - hand-rolled torch competing-risks discrete-time net (fallback if pycox
      install fails on the server).

Interface:
    m = DeepHitMT(num_durations=20, hidden=(64, 64), dropout=0.3,
                  lr=1e-3, batch_size=64, epochs=200, patience=20)
    m.fit(X, y_hep_struct, y_death_struct)        # both structured arrays
    risk_hep, risk_death = m.predict_risk(X)      # CIF-integrated to horizon

`y_*_struct` follows sksurv's convention: dtype=[('event','?'),('time','<f8')].
`valid_mask` is encoded by NaN time (matching annitia_maslg._make_structured) —
those rows are dropped from training. For the joint head we train on the
intersection of (hepatic_valid AND death_valid) and predict for all rows.
"""

from typing import Optional, Tuple

import numpy as np


def _drop_invalid(X: np.ndarray, y_hep, y_death):
    """Keep rows valid (non-NaN time) for BOTH endpoints.

    Returns (X_kept, e_hep, t_hep, e_death, t_death, keep_mask).
    """
    hep_valid = ~np.isnan(y_hep['time'])
    death_valid = ~np.isnan(y_death['time'])
    keep = hep_valid & death_valid
    return (X[keep],
            y_hep['event'][keep].astype(bool),
            y_hep['time'][keep].astype(np.float64),
            y_death['event'][keep].astype(bool),
            y_death['time'][keep].astype(np.float64),
            keep)


def _build_time_grid(t_hep: np.ndarray, t_death: np.ndarray, num_durations: int) -> np.ndarray:
    """Quantile grid of pooled durations. Rightmost edge is the max followup."""
    pooled = np.concatenate([t_hep, t_death])
    pooled = pooled[np.isfinite(pooled) & (pooled > 0)]
    qs = np.linspace(0.0, 1.0, num_durations + 1)
    grid = np.quantile(pooled, qs)
    grid = np.unique(grid)
    if len(grid) < 3:
        grid = np.linspace(pooled.min(), pooled.max(), num_durations + 1)
    return grid.astype(np.float64)


def _bin_index(t: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Map a duration to its bin index in [0, len(grid)-2]."""
    idx = np.searchsorted(grid, t, side='right') - 1
    idx = np.clip(idx, 0, len(grid) - 2)
    return idx.astype(np.int64)


class DeepHitMT:
    """Discrete-time competing-risks NN with shared trunk.

    Two cause-specific heads (1=hepatic, 2=death). Causes are mutually
    exclusive in the discrete-time formulation: cause = argmax(time)
    among observed events, censored if neither.

    For the ANNITIA cohort the two endpoints are not literally competing in
    a clinical sense (a patient can experience hepatic decompensation AND
    later die), but the discrete-time competing-risks loss is the cleanest
    way to share a trunk while keeping per-cause hazards interpretable.
    """

    def __init__(self,
                 num_durations: int = 20,
                 hidden: Tuple[int, ...] = (64, 64),
                 dropout: float = 0.3,
                 lr: float = 1e-3,
                 batch_size: int = 64,
                 epochs: int = 200,
                 patience: int = 20,
                 horizon_quantile: float = 0.5,
                 device: str = 'cpu',
                 random_state: int = 42,
                 verbose: bool = False):
        self.num_durations = num_durations
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.horizon_quantile = horizon_quantile
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self._scaler = None
        self._grid = None
        self._horizon_bin = None
        self._net = None
        self._backend = None  # 'pycox' or 'torch'

    # --- Cause assignment for competing-risks discrete-time loss --------
    @staticmethod
    def _assign_cause(e_hep, t_hep, e_death, t_death):
        """Return (cause, time) per patient.
            cause: 0=censored, 1=hepatic, 2=death
            time:  duration to first observed event (or last followup)
        """
        n = len(e_hep)
        cause = np.zeros(n, dtype=np.int64)
        time = np.maximum(t_hep, t_death)  # default: censored at last followup
        # If hepatic event observed and earlier (or only) -> cause=1
        only_hep = e_hep & ~e_death
        only_death = e_death & ~e_hep
        both = e_hep & e_death
        cause[only_hep] = 1
        cause[only_death] = 2
        # If both observed, attribute to the earlier event (first to occur)
        if both.any():
            hep_first = both & (t_hep <= t_death)
            death_first = both & (t_hep > t_death)
            cause[hep_first] = 1
            cause[death_first] = 2
        # Time = the time of the assigned event (or last followup if censored)
        ev_time = np.where(cause == 1, t_hep, np.where(cause == 2, t_death, time))
        return cause, ev_time.astype(np.float64)

    # --- Backend init ---------------------------------------------------
    def _try_pycox(self, in_features: int, num_durations: int):
        try:
            import torchtuples as tt
            from pycox.models import DeepHit
        except Exception as e:
            if self.verbose:
                print(f'  [DeepHitMT] pycox unavailable ({e}); falling back to torch backend')
            return None
        # `pycox.models.DeepHit` is the multi-event variant (num_risks set
        # implicitly by net.out_features = num_risks * num_durations).
        net = tt.practical.MLPVanilla(
            in_features=in_features,
            num_nodes=list(self.hidden),
            out_features=2 * num_durations,
            batch_norm=True,
            dropout=self.dropout,
            output_bias=False,
        )
        return ('pycox', net, DeepHit)

    def _torch_fallback(self, in_features: int, num_durations: int):
        """Hand-rolled discrete-time competing-risks net.

        Output: (B, 2*K) logits reshaped to (B, 2, K). Softmax over the 2*K
        flat dimension gives a joint hazard distribution; censored patients
        contribute log(P(survive past their bin)) only.
        """
        try:
            import torch
            import torch.nn as nn
        except Exception as e:
            raise RuntimeError(f'torch unavailable: {e}')

        layers = []
        prev = in_features
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers += [nn.Linear(prev, 2 * num_durations)]
        net = nn.Sequential(*layers)
        return ('torch', net, None, None)

    # --- Fit ------------------------------------------------------------
    def fit(self, X: np.ndarray, y_hep_struct: np.ndarray, y_death_struct: np.ndarray):
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, dtype=np.float32)
        Xt, e_hep, t_hep, e_death, t_death, _ = _drop_invalid(X, y_hep_struct, y_death_struct)
        if len(Xt) < 50:
            raise ValueError(f'Too few jointly-valid patients to fit DeepHitMT ({len(Xt)})')

        self._scaler = StandardScaler().fit(Xt)
        Xs = self._scaler.transform(Xt).astype(np.float32)

        self._grid = _build_time_grid(t_hep, t_death, self.num_durations)
        K = len(self._grid) - 1
        cause, ev_time = self._assign_cause(e_hep, t_hep, e_death, t_death)
        bin_idx = _bin_index(ev_time, self._grid)

        # Horizon bin for risk integration at .predict_risk time
        horizon_t = float(np.quantile(np.concatenate([t_hep, t_death]),
                                      self.horizon_quantile))
        self._horizon_bin = int(_bin_index(np.array([horizon_t]), self._grid)[0])

        # Choose backend
        backend_info = self._try_pycox(in_features=Xs.shape[1], num_durations=K)
        if backend_info is None:
            backend_info = self._torch_fallback(in_features=Xs.shape[1], num_durations=K)
        self._backend, net, *rest = backend_info

        if self._backend == 'pycox':
            self._fit_pycox(Xs, bin_idx, cause, K, net, *rest)
        else:
            self._fit_torch(Xs, bin_idx, cause, K, net)
        return self

    def _fit_pycox(self, Xs, bin_idx, cause, K, net, DeepHit):
        import torch
        import torchtuples as tt
        # pycox DeepHit expects target = (durations, events) where events in {0,1,..,n_risks}
        durations = bin_idx.astype(np.int64)
        events = cause.astype(np.int64)
        # 80/20 split for early stopping
        rng = np.random.default_rng(self.random_state)
        n = len(Xs)
        perm = rng.permutation(n)
        n_val = max(int(0.2 * n), 32)
        idx_val, idx_tr = perm[:n_val], perm[n_val:]
        x_tr, x_val = Xs[idx_tr], Xs[idx_val]
        y_tr = (durations[idx_tr], events[idx_tr])
        y_val = (durations[idx_val], events[idx_val])
        model = DeepHit(net, tt.optim.Adam(self.lr),
                        alpha=0.2, sigma=0.1, duration_index=np.arange(K))
        callbacks = [tt.callbacks.EarlyStopping(patience=self.patience)]
        model.fit(x_tr, y_tr, batch_size=self.batch_size,
                  epochs=self.epochs, callbacks=callbacks,
                  val_data=(x_val, y_val), verbose=self.verbose)
        self._net = model

    def _fit_torch(self, Xs, bin_idx, cause, K, net):
        import torch
        import torch.nn.functional as F

        device = torch.device(self.device)
        net = net.to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        rng = np.random.default_rng(self.random_state)
        n = len(Xs)
        perm = rng.permutation(n)
        n_val = max(int(0.2 * n), 32)
        idx_val, idx_tr = perm[:n_val], perm[n_val:]
        Xt = torch.from_numpy(Xs).to(device)
        bin_t = torch.from_numpy(bin_idx).long().to(device)
        cause_t = torch.from_numpy(cause).long().to(device)

        K_arange = torch.arange(K, device=device).view(1, 1, K)

        def _loss(logits, b, c):
            # logits: (B, 2*K) -> joint softmax over (cause, bin); censored
            # patients contribute log P(no event in any cause by bin b).
            B = logits.shape[0]
            log_p = F.log_softmax(logits, dim=1)  # (B, 2*K)
            event_mask = c > 0
            # obs_idx: row index in the flat 2*K vector for the observed (cause,bin)
            obs_idx = (c.clamp(min=1) - 1) * K + b
            ll_event = log_p.gather(1, obs_idx.view(-1, 1)).squeeze(1)

            # Vectorised censored term: triangular mask over bins 0..b inclusive,
            # broadcast across both causes, summed in fp space then logged.
            p2 = log_p.exp().view(B, 2, K)
            mask = (K_arange <= b.view(B, 1, 1)).to(p2.dtype)  # (B,1,K)
            surv_prob = 1.0 - (p2 * mask).sum(dim=(1, 2))
            ll_cens = torch.log(surv_prob.clamp(min=1e-8))

            ll = torch.where(event_mask, ll_event, ll_cens)
            return -ll.mean()

        best_val = float('inf')
        best_state = None
        bad = 0
        for epoch in range(self.epochs):
            net.train()
            perm_ep = rng.permutation(len(idx_tr))
            for s in range(0, len(perm_ep), self.batch_size):
                bidx = idx_tr[perm_ep[s:s + self.batch_size]]
                opt.zero_grad()
                out = net(Xt[bidx])
                loss = _loss(out, bin_t[bidx], cause_t[bidx])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vloss = _loss(net(Xt[idx_val]), bin_t[idx_val], cause_t[idx_val]).item()
            if vloss < best_val - 1e-4:
                best_val = vloss
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
            if self.verbose and epoch % 20 == 0:
                print(f'  [DeepHitMT torch] epoch={epoch} val_loss={vloss:.4f}')
        if best_state is not None:
            net.load_state_dict(best_state)
        self._net = net

    # --- Predict --------------------------------------------------------
    def predict_risk(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (risk_hepatic, risk_death) — CIF integrated up to horizon bin."""
        Xs = self._scaler.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        K = len(self._grid) - 1
        if self._backend == 'pycox':
            # pycox `DeepHit.predict_cif` returns one of:
            #   (n_durations, n_samples, n_risks)   — older releases
            #   (n_risks, n_durations, n_samples)   — newer releases
            # Disambiguate by axis lengths (n_risks=2 is the smallest and unique).
            cif = np.asarray(self._net.predict_cif(Xs))
            n_samples = Xs.shape[0]
            if cif.shape[0] == 2:
                # (2, K, n)
                risk_hep = cif[0, self._horizon_bin, :]
                risk_death = cif[1, self._horizon_bin, :]
            elif cif.shape[2] == 2:
                # (K, n, 2)
                risk_hep = cif[self._horizon_bin, :, 0]
                risk_death = cif[self._horizon_bin, :, 1]
            elif cif.shape[1] == 2:
                # (K, 2, n) — unlikely but covered
                risk_hep = cif[self._horizon_bin, 0, :]
                risk_death = cif[self._horizon_bin, 1, :]
            else:
                raise RuntimeError(f'Unexpected pycox CIF shape: {cif.shape}')
            assert len(risk_hep) == n_samples and len(risk_death) == n_samples
        else:
            import torch
            import torch.nn.functional as F
            net = self._net
            net.eval()
            with torch.no_grad():
                logits = net(torch.from_numpy(Xs))
                B = logits.shape[0]
                p = F.softmax(logits.view(B, 2 * K), dim=1).view(B, 2, K).numpy()
            # CIF up to horizon = cumulative cause-specific prob
            risk_hep = p[:, 0, : self._horizon_bin + 1].sum(axis=1)
            risk_death = p[:, 1, : self._horizon_bin + 1].sum(axis=1)
        return np.asarray(risk_hep, dtype=np.float64), np.asarray(risk_death, dtype=np.float64)
