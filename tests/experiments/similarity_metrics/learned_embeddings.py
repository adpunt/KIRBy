"""SERVER STEP 2 -- learned similarity embeddings (compass Stage 3).

Three learned representations, each turned into a similarity space and scored by
the SAME activity-grounded battery as the classical metrics:

  chemberta   -- ChemBERTa SMILES-transformer, mean-pooled hidden state.
  molformer   -- MoLFormer-XL SMILES-transformer, mean-pooled.
  triplet     -- a pure-torch deep-metric-learning encoder (the 'molecules as
                 triplets' idea, Coupry & Pogany, J Cheminform 2022, 14:11),
                 implemented as an MLP over a base fingerprint so it needs NO
                 torch-geometric / DGL (those are ABI-broken in this env). Two
                 triplet-definition modes:
                   structural -- positive = same Bemis-Murcko scaffold cluster
                                 (label-free; reproduces STRUCTURAL similarity).
                   activity   -- positive = smaller |dy| (SUPERVISED; the honest
                                 test of 'distance that reflects activity').
                 Triplets are drawn ONLY from the training scaffolds; the encoder
                 is applied to held-out scaffolds and scored there -> no leakage.

Why not a real message-passing GNN: torch_geometric fails to import here (torch-
scatter ABI mismatch). The MLP-over-fingerprint metric learner is the robust
stand-in; if a working PyG/DGL is available on the server, swapping the encoder
for a GIN is a drop-in change (noted in the README). The scientific question --
does a LEARNED distance beat the classical winners on activity coherence -- is
answered either way.

CPU-only fallback works; GPU (V100, per project notes) is much faster for the
transformers. Embeddings are cached to sim_data/emb_<dataset>_<name>.npy.
"""
import os, sys, argparse, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CHEMBERTA = os.environ.get('CHEMBERTA_ID', 'seyonec/ChemBERTa-zinc-base-v1')
MOLFORMER = os.environ.get('MOLFORMER_ID', 'ibm/MoLFormer-XL-both-10pct')


# --------------------------------------------------------- transformer embeds
def embed_transformer(smiles, model_id, batch=64, max_len=256):
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(model_id, trust_remote_code=True,
                                    deterministic_eval=True).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(smiles), batch):
            chunk = list(smiles[i:i + batch])
            enc = tok(chunk, padding=True, truncation=True, max_length=max_len,
                      return_tensors='pt').to(dev)
            h = mdl(**enc).last_hidden_state                     # (b, L, d)
            mask = enc['attention_mask'].unsqueeze(-1).float()
            emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1)   # mean-pool
            out.append(emb.cpu().numpy())
    return np.vstack(out).astype(np.float32)


# --------------------------------------------------------- triplet MLP encoder
def train_triplet(Xtr_base, ytr, groups_tr, mode='structural', dim=128,
                  epochs=40, margin=0.3, seed=0, n_trip=20000, lr=1e-3):
    """Return a callable f(X_base) -> embedding, trained on TRAIN scaffolds only."""
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xtr = torch.tensor(Xtr_base, dtype=torch.float32, device=dev)
    d = Xtr.shape[1]
    enc = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, 256),
                        nn.ReLU(), nn.Linear(256, dim)).to(dev)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    rs = np.random.RandomState(seed)
    n = len(ytr)

    def sample_triplets():
        a = rs.randint(0, n, n_trip); p = rs.randint(0, n, n_trip); q = rs.randint(0, n, n_trip)
        ok = (a != p) & (a != q) & (p != q)
        a, p, q = a[ok], p[ok], q[ok]
        if mode == 'structural':
            # positive = SAME scaffold cluster as anchor; negative = different
            same_p = groups_tr[a] == groups_tr[p]
            same_q = groups_tr[a] == groups_tr[q]
            # want p to be the same-cluster one, q the different
            swap = (~same_p) & same_q
            p2 = np.where(swap, q, p); q2 = np.where(swap, p, q)
            keep = (groups_tr[a] == groups_tr[p2]) & (groups_tr[a] != groups_tr[q2])
        else:  # activity: positive = smaller |dy|
            dyp = np.abs(ytr[a] - ytr[p]); dyq = np.abs(ytr[a] - ytr[q])
            swap = dyq < dyp
            p2 = np.where(swap, q, p); q2 = np.where(swap, p, q)
            keep = np.abs(ytr[a] - ytr[p2]) < np.abs(ytr[a] - ytr[q2])
        return a[keep], p2[keep], q2[keep]

    a, p, q = sample_triplets()
    a = torch.tensor(a, device=dev); p = torch.tensor(p, device=dev); q = torch.tensor(q, device=dev)
    trip = nn.TripletMarginLoss(margin=margin)
    bs = 2048
    for ep in range(epochs):
        perm = torch.randperm(len(a), device=dev)
        for s in range(0, len(a), bs):
            idx = perm[s:s + bs]
            ea, ep_, eq = enc(Xtr[a[idx]]), enc(Xtr[p[idx]]), enc(Xtr[q[idx]])
            loss = trip(ea, ep_, eq)
            opt.zero_grad(); loss.backward(); opt.step()

    def transform(Xb):
        with torch.no_grad():
            return enc(torch.tensor(Xb, dtype=torch.float32, device=dev)).cpu().numpy()
    return transform


# --------------------------------------------------------- driver: cache embeds
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=os.path.join(
        os.environ.get('KIRBY_CACHE_DIR', '.'), 'similarity_metrics', 'sim_data'))
    ap.add_argument('--which', nargs='+', default=['chemberta', 'molformer'],
                    help='transformer embeddings to precompute per dataset')
    args = ap.parse_args()
    import glob
    for npz in sorted(glob.glob(os.path.join(args.data_dir, 'sim_*.npz'))):
        d = np.load(npz, allow_pickle=True)
        if 'smiles' not in d:
            print(f'no smiles in {npz}; rebuild with build_moleculeace.py'); continue
        smi = [str(s) for s in d['smiles']]
        name = os.path.basename(npz)[4:-4]
        for which in args.which:
            out = os.path.join(args.data_dir, f'emb_{name}_{which}.npy')
            if os.path.exists(out):
                print(f'  cached {out}'); continue
            mid = CHEMBERTA if which == 'chemberta' else MOLFORMER
            try:
                emb = embed_transformer(smi, mid)
                np.save(out, emb)
                print(f'  {name} {which}: {emb.shape} -> {out}', flush=True)
            except Exception as e:
                print(f'  FAIL {name} {which}: {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
