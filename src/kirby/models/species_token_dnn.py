"""Species-conditional DNN for cross-species pretraining (T-PT.5).

`SpeciesTokenDNNRegressor` extends the Imrie-style `PretrainedDNNRegressor`
with two conditioning tokens, concatenated to the molecular representation
before the first linear layer:

  - a learned `species_emb_dim`-D species embedding (default 16-D, locked by
    docs/cross_species_pretraining_design.md §6 Q3), indexed by SPECIES_VOCAB;
  - an `n_targets`-D target one-hot, indexed by a per-checkpoint target
    vocabulary (the top-N target_chembl_ids of the pretraining corpus, with
    the four vet eval target IDs force-included so the fine-tune rows always
    map to a real one-hot slot).

This is the model behind the `species_token_multispecies` variant: joint
training on all six mammalian species at all targets (corpus
`multispecies_chembl`), then fine-tuned on the vet eval targets. The "species
token earns its keep" test toggles the species index at predict time (real
species vs forced human) and measures the RMSE delta on held-out vet rows.

sklearn-compat constraint (same as the parent): __init__ stores only
hyperparameters — no torch objects — so clone()/deepcopy() work. Because the
sklearn fit(X, y) signature carries only X, the per-row species and target
indices ride as the LAST TWO COLUMNS of X:

    X[:, :-2]  molecular features
    X[:, -2]   species index   (int into SPECIES_VOCAB)
    X[:, -1]   target index    (int into the target vocab; -1 = out-of-vocab
               target -> all-zero one-hot)

`pack_token_columns()` builds that layout; the experiment script is the only
caller, so the convention stays local.

Pretraining entry point:

    python -m kirby.models.species_token_dnn --rep ecfp4,maccs --n-targets 50

mirrors pretrain_chembl_dnn.py but trains the token model on the
multispecies_chembl corpus and writes one checkpoint per rep into
{output_dir}/N{n_targets}/, plus a shared {output_dir}/N{n_targets}/
target_vocab.json the experiment reads to pack X.

Reference:
    Imrie et al. (2018). JCIM 58(11):2319-2330.
    TACTIC — species-conditional multi-task QSAR.
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # src/
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tests'))  # tests/

from kirby.models.dnn_pretrained import PretrainedDNNRegressor
from kirby.utils.species import SPECIES_REGISTRY

# Six mammalian species in the multispecies_chembl corpus, in a FIXED order —
# the species embedding table is indexed by this list, so it must not change
# between pretraining and fine-tuning.
SPECIES_VOCAB = ['human', 'rat', 'mouse', 'dog', 'cow', 'horse']
_SPECIES_IDX = {s: i for i, s in enumerate(SPECIES_VOCAB)}
HUMAN_IDX = _SPECIES_IDX['human']

HIDDEN_SIZES_DEFAULT = (256, 128, 64)


def species_to_idx(organism) -> int:
    """Map an organism string to its SPECIES_VOCAB index.

    Accepts the short key ('cow') or the ChEMBL Latin binomial
    ('Bos taurus'). Raises ValueError for anything outside SPECIES_VOCAB.
    """
    key = str(organism).strip().lower()
    if key in _SPECIES_IDX:
        return _SPECIES_IDX[key]
    for short, entry in SPECIES_REGISTRY.items():
        if str(entry['chembl_organism']).lower() == key and short in _SPECIES_IDX:
            return _SPECIES_IDX[short]
    raise ValueError(
        f"Organism '{organism}' not in SPECIES_VOCAB {SPECIES_VOCAB}")


def pack_token_columns(X_mol, organisms, target_chembl_ids, target_vocab):
    """Append [species_idx, target_idx] columns to a molecular feature matrix.

    Args:
        X_mol: (n, d) molecular features.
        organisms: length-n iterable of organism strings (short key or Latin).
        target_chembl_ids: length-n iterable of target_chembl_id strings.
        target_vocab: ordered list of target_chembl_ids defining the one-hot
            index. IDs not in the vocab get index -1 (all-zero one-hot).

    Returns:
        (n, d+2) float64 array — last two columns are species idx, target idx.
    """
    X_mol = np.asarray(X_mol, dtype=np.float64)
    n = X_mol.shape[0]
    if len(organisms) != n or len(target_chembl_ids) != n:
        raise ValueError(
            f"length mismatch: X_mol={n}, organisms={len(organisms)}, "
            f"target_chembl_ids={len(target_chembl_ids)}")
    sp = np.array([species_to_idx(o) for o in organisms], dtype=np.float64)
    tvocab = {t: i for i, t in enumerate(target_vocab)}
    tg = np.array([tvocab.get(t, -1) for t in target_chembl_ids],
                  dtype=np.float64)
    return np.column_stack([X_mol, sp, tg])


def _make_species_token_net(mol_dim, n_species, n_targets, emb_dim,
                            hidden_sizes, dropout):
    """Build the conditioned DNN. torch is imported here, never at __init__."""
    import torch
    import torch.nn as nn

    class _SpeciesTokenNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.species_emb = nn.Embedding(n_species, emb_dim)
            self.n_targets = n_targets
            layers = []
            prev = mol_dim + emb_dim + n_targets
            for h in hidden_sizes:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.backbone = nn.Sequential(*layers)

        def forward(self, x_mol, species_idx, target_idx):
            sp = self.species_emb(species_idx)
            tgt = torch.zeros(x_mol.shape[0], self.n_targets,
                              device=x_mol.device, dtype=x_mol.dtype)
            valid = target_idx >= 0
            if bool(valid.any()):
                oh = nn.functional.one_hot(
                    target_idx[valid].clamp(min=0), self.n_targets)
                tgt[valid] = oh.to(x_mol.dtype)
            h = torch.cat([x_mol, sp, tgt], dim=1)
            return self.backbone(h)

    return _SpeciesTokenNet()


class SpeciesTokenDNNRegressor(PretrainedDNNRegressor):
    """DNN conditioned on a learned species embedding + target one-hot.

    See module docstring for the X column layout. `n_targets` is required and
    must match the pretrained checkpoint's target vocab size.

    Args (beyond PretrainedDNNRegressor):
        n_targets: size of the target one-hot vocabulary (the N sweep value).
        species_emb_dim: width of the learned species embedding (16, locked).
        force_species_idx: if not None, every row is predicted as if it were
            this species index — used for the "human token" toggle. Has no
            effect during fit(); read only by predict().
    """

    def __init__(self, n_targets, hidden_sizes=(256, 128, 64), dropout=0.2,
                 species_emb_dim=16, pretrained_path=None, finetune_lr=0.0001,
                 freeze_layers=0, epochs=200, batch_size=256, patience=20,
                 random_state=42, zero_shot=False, force_species_idx=None):
        self.n_targets = n_targets
        self.species_emb_dim = species_emb_dim
        self.force_species_idx = force_species_idx
        super().__init__(
            hidden_sizes=hidden_sizes, dropout=dropout,
            pretrained_path=pretrained_path, finetune_lr=finetune_lr,
            freeze_layers=freeze_layers, epochs=epochs, batch_size=batch_size,
            patience=patience, random_state=random_state, zero_shot=zero_shot)

    @staticmethod
    def _split_tokens(X):
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] < 3:
            raise ValueError(
                f"SpeciesTokenDNNRegressor expects packed X with >=3 columns "
                f"(features + species idx + target idx), got {X.shape[1]}")
        X_mol = X[:, :-2].astype(np.float32)
        species_idx = X[:, -2].astype(np.int64)
        target_idx = X[:, -1].astype(np.int64)
        return X_mol, species_idx, target_idx

    def _build_token_net(self, mol_dim):
        return _make_species_token_net(
            mol_dim, len(SPECIES_VOCAB), self.n_targets, self.species_emb_dim,
            tuple(self.hidden_sizes), self.dropout)

    def _load_token_pretrained(self, net, mol_dim):
        """Load a species-token checkpoint into `net`; validate dims."""
        import torch
        ckpt = torch.load(self.pretrained_path, map_location='cpu',
                          weights_only=True)
        if not isinstance(ckpt, dict) or 'state_dict' not in ckpt:
            raise ValueError(
                f"'{self.pretrained_path}' is not a species-token checkpoint")
        saved_mol = ckpt.get('input_dim')
        if saved_mol is not None and saved_mol != mol_dim:
            raise ValueError(
                f"checkpoint mol_dim={saved_mol} != {mol_dim} (rep mismatch?)")
        saved_nt = ckpt.get('n_targets')
        if saved_nt is not None and saved_nt != self.n_targets:
            raise ValueError(
                f"checkpoint n_targets={saved_nt} != {self.n_targets}")
        saved_emb = ckpt.get('species_emb_dim')
        if saved_emb is not None and saved_emb != self.species_emb_dim:
            raise ValueError(
                f"checkpoint species_emb_dim={saved_emb} != "
                f"{self.species_emb_dim}")
        saved_hs = ckpt.get('hidden_sizes')
        if saved_hs is not None and tuple(saved_hs) != tuple(self.hidden_sizes):
            raise ValueError(
                f"checkpoint hidden_sizes={tuple(saved_hs)} != "
                f"{tuple(self.hidden_sizes)}")
        net.load_state_dict(ckpt['state_dict'])
        return ckpt

    def _freeze_token_layers(self, net):
        """freeze_layers=-1 -> linear probe (freeze species emb + all hidden
        backbone layers, train only the final linear head). 0 -> full
        fine-tune. Other values are treated as full fine-tune."""
        if self.freeze_layers == 0:
            return
        for p in net.species_emb.parameters():
            p.requires_grad = False
        backbone = net.backbone
        last = len(backbone) - 1
        for i, module in enumerate(backbone):
            if i < last:
                for p in module.parameters():
                    p.requires_grad = False

    def _fit_zero_shot_tokens(self, X_mol):
        net = self._build_token_net(X_mol.shape[1])
        ckpt = self._load_token_pretrained(net, X_mol.shape[1])
        if 'feat_mean' not in ckpt:
            raise ValueError(
                f"checkpoint '{self.pretrained_path}' has no 'feat_mean'/"
                f"'feat_scale' — re-run species-token pretraining to "
                f"regenerate it.")
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        sc.mean_ = np.asarray(ckpt['feat_mean'], dtype=np.float64)
        sc.scale_ = np.asarray(ckpt['feat_scale'], dtype=np.float64)
        sc.var_ = sc.scale_ ** 2
        sc.n_features_in_ = sc.mean_.shape[0]
        self.feat_scaler_ = sc
        self.y_mean_ = float(ckpt.get('y_mean', 0.0))
        self.y_std_ = float(ckpt.get('y_std', 1.0))
        net.eval()
        self.model_ = net
        self.input_dim_ = X_mol.shape[1]
        return self

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(self.random_state)

        X_mol, species_idx, target_idx = self._split_tokens(X)
        y = np.asarray(y, dtype=np.float32).ravel()

        if self.zero_shot:
            return self._fit_zero_shot_tokens(X_mol)

        # Scale molecular features only; token columns are indices.
        self.feat_scaler_ = StandardScaler()
        X_mol = self.feat_scaler_.fit_transform(X_mol).astype(np.float32)
        X_mol = np.nan_to_num(X_mol, nan=0.0, posinf=0.0, neginf=0.0)

        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) if y.std() > 1e-8 else 1.0
        y_norm = (y - self.y_mean_) / self.y_std_

        n = len(y)
        idx = rng.permutation(n)
        split = max(1, int(0.8 * n))
        tr_idx, ho_idx = idx[:split], idx[split:]
        if len(ho_idx) == 0:  # n too small for a holdout — reuse train
            ho_idx = tr_idx

        device = torch.device('cpu')
        net = self._build_token_net(X_mol.shape[1]).to(device)

        if self.pretrained_path is not None:
            self._load_token_pretrained(net, X_mol.shape[1])

        self._freeze_token_layers(net)

        trainable = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=self.finetune_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()

        Xm = torch.from_numpy(X_mol)
        sp = torch.from_numpy(species_idx)
        tg = torch.from_numpy(target_idx)
        yn = torch.from_numpy(y_norm)
        tr_t = torch.as_tensor(tr_idx, dtype=torch.long)
        ho_t = torch.as_tensor(ho_idx, dtype=torch.long)

        best_loss, best_state, patience_counter = float('inf'), None, 0
        for epoch in range(self.epochs):
            net.train()
            perm = tr_t[torch.randperm(len(tr_t))]
            for start in range(0, len(perm), self.batch_size):
                b = perm[start:start + self.batch_size]
                optimizer.zero_grad()
                pred = net(Xm[b], sp[b], tg[b]).squeeze(-1)
                loss = criterion(pred, yn[b])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

            net.eval()
            with torch.no_grad():
                ho_pred = net(Xm[ho_t], sp[ho_t], tg[ho_t]).squeeze(-1)
                ho_loss = criterion(ho_pred, yn[ho_t]).item()
            scheduler.step(ho_loss)

            if ho_loss < best_loss:
                best_loss = ho_loss
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
        self.input_dim_ = X_mol.shape[1]
        return self

    def predict(self, X):
        import torch
        X_mol, species_idx, target_idx = self._split_tokens(X)
        if hasattr(self, 'feat_scaler_'):
            X_mol = self.feat_scaler_.transform(X_mol).astype(np.float32)
            X_mol = np.nan_to_num(X_mol, nan=0.0, posinf=0.0, neginf=0.0)
        if self.force_species_idx is not None:
            species_idx = np.full_like(species_idx,
                                       int(self.force_species_idx))
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(
                torch.from_numpy(np.asarray(X_mol, dtype=np.float32)),
                torch.from_numpy(np.asarray(species_idx, dtype=np.int64)),
                torch.from_numpy(np.asarray(target_idx, dtype=np.int64)),
            ).squeeze(-1).numpy()
        return pred * self.y_std_ + self.y_mean_


# --------------------------------------------------------------------------
# Pretraining (run once per (rep, N) via SLURM; not per-fold).
# --------------------------------------------------------------------------

def _rep_funcs():
    from kirby.representations.molecular import (
        create_ecfp4, create_maccs, create_avalon, create_pdv,
        create_mordred, create_mol2vec,
    )
    return {
        'ecfp4': create_ecfp4,
        'maccs': create_maccs,
        'avalon': create_avalon,
        'pdv': create_pdv,
        'mordred': create_mordred,
        'mol2vec': create_mol2vec,
    }


def _vet_eval_target_ids():
    """The 4 vet eval target_chembl_ids — force-included in every N's vocab so
    fine-tune rows always map to a real one-hot slot. Read from the locked
    docs/cross_species_vet_eval_targets.json (source of truth)."""
    json_path = (Path(__file__).resolve().parents[3]
                 / 'docs' / 'cross_species_vet_eval_targets.json')
    data = json.loads(json_path.read_text())
    ids = []
    for k, v in data.items():
        if k.startswith('_'):
            continue
        tid = v.get('target_chembl_id')
        if tid and tid not in ids:
            ids.append(tid)
    return ids


def build_target_vocab(corpus_df, n_targets, vet_ids):
    """Ordered target one-hot vocabulary of size `n_targets`.

    The vet eval target IDs go first (force-included), then the most frequent
    remaining target_chembl_ids in the corpus fill up to n_targets.
    """
    vet_ids = list(dict.fromkeys(vet_ids))
    if n_targets <= len(vet_ids):
        return vet_ids[:n_targets]
    vet_set = set(vet_ids)
    counts = corpus_df['target_chembl_id'].value_counts()
    others = [t for t in counts.index if t not in vet_set]
    return vet_ids + others[:n_targets - len(vet_ids)]


def pretrain_one_rep_species_token(
    smiles, pactivities, organisms, target_chembl_ids, rep_name, n_targets,
    target_vocab, hidden_sizes, species_emb_dim, output_dir,
    epochs=200, batch_size=1024, lr=0.001, patience=30, seed=42,
):
    """Pretrain the species-token DNN for one rep. Saves
    {output_dir}/{rep}_{hs}.pt. Returns a summary dict (or None on failure)."""
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    rep_funcs = _rep_funcs()
    if rep_name not in rep_funcs:
        print(f"  ERROR: unknown rep '{rep_name}'. "
              f"Available: {', '.join(rep_funcs)}")
        return None

    print(f"\n{'='*60}\nPre-training (species token): {rep_name}\n{'='*60}")
    t0 = time.time()
    print(f"  Generating {rep_name} for {len(smiles)} compounds...", flush=True)
    try:
        X_mol = rep_funcs[rep_name](smiles)
    except Exception as e:
        print(f"  FAILED to generate {rep_name}: {e}")
        return None
    X_mol = np.asarray(X_mol, dtype=np.float32)
    y = np.asarray(pactivities, dtype=np.float32)
    sp_idx = np.array([species_to_idx(o) for o in organisms], dtype=np.int64)
    tvocab = {t: i for i, t in enumerate(target_vocab)}
    tg_idx = np.array([tvocab.get(t, -1) for t in target_chembl_ids],
                      dtype=np.int64)

    valid = np.all(np.isfinite(X_mol), axis=1) & np.isfinite(y)
    X_mol, y, sp_idx, tg_idx = X_mol[valid], y[valid], sp_idx[valid], tg_idx[valid]
    print(f"  Rep shape: {X_mol.shape} ({time.time()-t0:.1f}s), "
          f"{int((tg_idx >= 0).sum())}/{len(tg_idx)} rows have an in-vocab "
          f"target token")

    feat_scaler = StandardScaler()
    X_mol = feat_scaler.fit_transform(X_mol)
    X_mol = np.nan_to_num(X_mol, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    y_mean = float(y.mean())
    y_std = float(y.std()) if y.std() > 1e-8 else 1.0
    y_norm = (y - y_mean) / y_std

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    split = int(0.9 * len(y))
    tr_idx, ho_idx = idx[:split], idx[split:]

    device = torch.device('cpu')
    net = _make_species_token_net(
        X_mol.shape[1], len(SPECIES_VOCAB), n_targets, species_emb_dim,
        tuple(hidden_sizes), 0.2).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    Xm = torch.from_numpy(X_mol)
    sp = torch.from_numpy(sp_idx)
    tg = torch.from_numpy(tg_idx)
    yn = torch.from_numpy(y_norm)
    tr_t = torch.as_tensor(tr_idx, dtype=torch.long)
    ho_t = torch.as_tensor(ho_idx, dtype=torch.long)

    best_loss, best_state, patience_counter = float('inf'), None, 0
    t0_train = time.time()
    for epoch in range(epochs):
        net.train()
        perm = tr_t[torch.randperm(len(tr_t))]
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, len(perm), batch_size):
            b = perm[start:start + batch_size]
            optimizer.zero_grad()
            pred = net(Xm[b], sp[b], tg[b]).squeeze(-1)
            loss = criterion(pred, yn[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        net.eval()
        with torch.no_grad():
            ho_loss = criterion(
                net(Xm[ho_t], sp[ho_t], tg[ho_t]).squeeze(-1),
                yn[ho_t]).item()
        scheduler.step(ho_loss)

        if ho_loss < best_loss:
            best_loss = ho_loss
            best_state = {k: v.cpu().clone()
                          for k, v in net.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train={epoch_loss/max(n_batches,1):.4f}, "
                  f"ho={ho_loss:.4f} (RMSE={np.sqrt(ho_loss)*y_std:.3f}), "
                  f"patience={patience_counter}/{patience}", flush=True)
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        ho_pred = net(Xm[ho_t], sp[ho_t], tg[ho_t]).squeeze(-1).numpy()
    ho_pred_orig = ho_pred * y_std + y_mean
    ho_rmse = float(np.sqrt(np.mean((ho_pred_orig - y[ho_idx]) ** 2)))
    elapsed = time.time() - t0_train
    print(f"\n  Final holdout RMSE={ho_rmse:.3f} pIC50  ({elapsed:.1f}s)")

    os.makedirs(output_dir, exist_ok=True)
    hs_str = '_'.join(str(h) for h in hidden_sizes)
    save_path = os.path.join(output_dir, f'{rep_name}_{hs_str}.pt')
    checkpoint = {
        'state_dict': best_state if best_state is not None else net.state_dict(),
        'input_dim': X_mol.shape[1],
        'n_targets': n_targets,
        'species_emb_dim': species_emb_dim,
        'hidden_sizes': tuple(hidden_sizes),
        'species_vocab': list(SPECIES_VOCAB),
        'target_vocab': list(target_vocab),
        'n_compounds': int(len(y)),
        'holdout_rmse': ho_rmse,
        'y_mean': y_mean,
        'y_std': y_std,
        'rep_name': rep_name,
        # Pretrain feature scaler — required for faithful zero-shot transfer
        # (consumed by SpeciesTokenDNNRegressor(zero_shot=True)).
        'feat_mean': feat_scaler.mean_.tolist(),
        'feat_scale': feat_scaler.scale_.tolist(),
    }
    torch.save(checkpoint, save_path)
    print(f"  Saved: {save_path}")
    return {'rep_name': rep_name, 'input_dim': X_mol.shape[1],
            'n_compounds': int(len(y)), 'holdout_rmse': ho_rmse,
            'save_path': save_path, 'train_time': elapsed}


def main():
    parser = argparse.ArgumentParser(
        description='Pre-train the species-token DNN on a multispecies ChEMBL '
                    'corpus for the species_token_multispecies variant (T-PT.5).',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rep', type=str, required=True,
                        help='Comma-separated rep names '
                             '(ecfp4,maccs,pdv,mol2vec,mordred,avalon)')
    parser.add_argument('--n-targets', type=int, required=True,
                        help='Target one-hot vocab size N (sweep: 10, 50, 200)')
    parser.add_argument('--corpus', type=str, default='multispecies_chembl',
                        help='Named corpus from kirby.datasets.pretrain_corpora '
                             '(default: multispecies_chembl)')
    parser.add_argument('--n-compounds', type=int, default=500_000,
                        help='Max compounds per species (default: 500000)')
    parser.add_argument('--max-targets', type=int, default=200,
                        help='Top-N targets per species fetched into the '
                             'corpus (default: 200)')
    parser.add_argument('--hidden-sizes', type=str, default='256,128,64')
    parser.add_argument('--species-emb-dim', type=int, default=16,
                        help='Learned species embedding width (16, locked)')
    parser.add_argument('--output-dir', type=str,
                        default='/Volumes/seagate/kirby/chembl_pretrain/species_token',
                        help='Pretrain root; checkpoints go in '
                             '{output_dir}/N{n_targets}/')
    parser.add_argument('--data-dir', type=str,
                        default='/Volumes/seagate/kirby/chembl')
    parser.add_argument('--vet-cache-dir', type=str, default=None,
                        help='Vet eval ChEMBL CSV cache dir for the leakage '
                             'filter. Defaults to vet_eval_data.DEFAULT_CACHE_DIR.')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    warnings.filterwarnings('ignore')

    rep_names = [r.strip() for r in args.rep.split(',')]
    hidden_sizes = tuple(int(h) for h in args.hidden_sizes.split(','))

    from kirby.datasets.pretrain_corpora import list_corpora, compose
    if args.corpus not in list_corpora():
        parser.error(f"Unknown --corpus '{args.corpus}'. "
                     f"Known: {', '.join(list_corpora())}")

    out_dir = os.path.join(args.output_dir, f'N{args.n_targets}')
    print("=" * 70)
    print("Species-token DNN pre-training (T-PT.5)")
    print(f"  Reps:         {rep_names}")
    print(f"  N (targets):  {args.n_targets}")
    print(f"  Corpus:       {args.corpus}")
    print(f"  Hidden sizes: {hidden_sizes}")
    print(f"  Species emb:  {args.species_emb_dim}-D")
    print(f"  Output:       {out_dir}")
    print("=" * 70)

    # Leakage prevention: drop human downstream benchmarks AND every vet eval
    # InChIKey (the corpus is fine-tuned on the vet targets).
    from kirby.datasets.chembl_bioactivity import collect_downstream_inchikeys
    print("\nCollecting downstream InChIKeys for leakage prevention...")
    exclude_keys = collect_downstream_inchikeys()
    try:
        from experiments.cross_species.vet_eval_data import (
            collect_vet_eval_inchikeys,
            DEFAULT_CACHE_DIR as VET_DEFAULT_CACHE_DIR,
        )
        vet_dir = args.vet_cache_dir or VET_DEFAULT_CACHE_DIR
        print(f"Collecting vet eval InChIKeys for leakage prevention "
              f"(cache: {vet_dir})...")
        exclude_keys = exclude_keys | collect_vet_eval_inchikeys(vet_dir)
    except Exception as e:
        print(f"  WARNING: could not collect vet eval InChIKeys: {e}")
        print(f"  Corpus filtered against human downstream benchmarks only.")

    print(f"\nComposing corpus '{args.corpus}'...")
    df = compose(name=args.corpus, exclude_inchikeys=exclude_keys,
                 max_compounds_per_species=args.n_compounds,
                 max_targets=args.max_targets, data_dir=args.data_dir)
    if len(df) == 0:
        print("ERROR: corpus is empty after assembly/filtering. Aborting.")
        return
    print(f"  {len(df)} rows, {df['target_chembl_id'].nunique()} targets, "
          f"species: {sorted(df['organism'].unique())}")

    vet_ids = _vet_eval_target_ids()
    target_vocab = build_target_vocab(df, args.n_targets, vet_ids)
    print(f"  target vocab: {len(target_vocab)} ids "
          f"({len(vet_ids)} vet eval ids force-included)")

    os.makedirs(out_dir, exist_ok=True)
    vocab_path = os.path.join(out_dir, 'target_vocab.json')
    Path(vocab_path).write_text(json.dumps({
        'n_targets': args.n_targets,
        'corpus': args.corpus,
        'vet_eval_ids': vet_ids,
        'target_vocab': target_vocab,
    }, indent=2))
    print(f"  wrote {vocab_path}")

    smiles = df['smiles'].tolist()
    pactivities = df['pActivity'].values
    organisms = df['organism'].tolist()
    target_ids = df['target_chembl_id'].tolist()

    results = []
    for rep_name in rep_names:
        r = pretrain_one_rep_species_token(
            smiles, pactivities, organisms, target_ids, rep_name,
            args.n_targets, target_vocab, hidden_sizes, args.species_emb_dim,
            out_dir, epochs=args.epochs, batch_size=1024, lr=args.lr,
            patience=args.patience, seed=args.seed)
        if r:
            results.append(r)

    print(f"\n{'='*70}\nPRE-TRAINING SUMMARY (N={args.n_targets})\n{'='*70}")
    for r in results:
        print(f"  {r['rep_name']:12s}  dim={r['input_dim']:6d}  "
              f"n={r['n_compounds']:8d}  RMSE={r['holdout_rmse']:.3f}  "
              f"{r['train_time']:.1f}s")
    print(f"\nWeights saved to: {out_dir}/")


if __name__ == '__main__':
    main()
