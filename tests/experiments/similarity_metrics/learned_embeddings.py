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
import hashlib
import os, sys, argparse, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CHEMBERTA = os.environ.get('CHEMBERTA_ID', 'seyonec/ChemBERTa-zinc-base-v1')
MOLFORMER = os.environ.get('MOLFORMER_ID', 'ibm/MoLFormer-XL-both-10pct')


def emb_cache_name(dataset: str, which: str, model_id: str,
                   smiles=None) -> str:
    """File name for one dataset's cached transformer vectors.

    The name used to be `emb_<dataset>_<which>.npy`, where `which` is the alias
    'chemberta' or 'molformer'. Which MODEL that alias means is an environment
    variable, so a second run under a different checkpoint found a file of that
    name and scored the first checkpoint's vectors. The molecule list is not in
    it either, so rebuilding the dataset under the same name left every vector
    paired with another molecule's label.

    Both go in the name: a short hash of the model id, and a short hash of the
    molecule list when it is known.
    """
    mid = hashlib.sha1(str(model_id).encode()).hexdigest()[:8]
    part = f'emb_{dataset}_{which}__m{mid}'
    if smiles is not None:
        sm = hashlib.sha1('\n'.join(map(str, smiles)).encode()).hexdigest()[:8]
        part += f'__s{sm}'
    return part + '.npy'



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
    """Return a callable f(X_base) -> embedding, trained on TRAIN scaffolds only.

    A FRESH set of triplets is drawn on every one of the `epochs` passes. The
    draw used to happen once, before the pass loop, so all 40 passes fitted the
    encoder to one fixed set of a few thousand triples.

    In `structural` mode the positive is now drawn from INSIDE the anchor's
    Bemis-Murcko scaffold group and the negative from outside it. It used to
    draw two molecules uniformly over the training set and keep the triple only
    when exactly one of them happened to share the anchor's scaffold, which left
    about 140 triplets out of 20000 candidates. An anchor whose scaffold group
    holds only itself has no positive and is never drawn.

    The returned callable carries one surviving-triplet count per pass on its
    `n_kept_per_pass` attribute, `epochs` long, so a caller can record how many
    triplets the encoder was fitted on. It is an attribute rather than a second
    return value because run_battery_moleculeace.py calls this as
    `f = train_triplet(...)` and then `f(X)`.
    """
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

    if mode == 'structural':
        # scaffold-group membership as a flat index array: `order` lists the
        # molecules grouped together, `starts`/`sizes` bound each group's block,
        # and `rank` is a molecule's position inside its own block. That is what
        # lets a positive be drawn from inside the anchor's group directly.
        gi = np.unique(np.asarray(groups_tr), return_inverse=True)[1]
        sizes = np.bincount(gi)
        starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
        order = np.argsort(gi, kind='stable')
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(n) - starts[gi[order]]
        anchors_ok = np.flatnonzero(sizes[gi] >= 2)

    def sample_triplets():
        if mode == 'structural':
            # positive = SAME scaffold group as anchor; negative = different
            if len(anchors_ok) == 0:
                e = np.empty(0, dtype=np.int64)
                return e, e, e
            a = anchors_ok[rs.randint(0, len(anchors_ok), n_trip)]
            m1 = sizes[gi[a]] - 1                 # group members other than the anchor
            j = np.minimum((rs.random_sample(n_trip) * m1).astype(np.int64), m1 - 1)
            j = j + (j >= rank[a])                # skip over the anchor itself
            p2 = order[starts[gi[a]] + j]
            # negative = uniform over the training set, redrawn while it lands
            # in the anchor's own group; the few that still collide are dropped.
            q2 = rs.randint(0, n, n_trip)
            for _ in range(3):
                bad = gi[a] == gi[q2]
                if not bad.any():
                    break
                q2[bad] = rs.randint(0, n, int(bad.sum()))
            keep = gi[a] != gi[q2]
        else:  # activity: positive = smaller |dy|
            a = rs.randint(0, n, n_trip); p = rs.randint(0, n, n_trip); q = rs.randint(0, n, n_trip)
            ok = (a != p) & (a != q) & (p != q)
            a, p, q = a[ok], p[ok], q[ok]
            dyp = np.abs(ytr[a] - ytr[p]); dyq = np.abs(ytr[a] - ytr[q])
            swap = dyq < dyp
            p2 = np.where(swap, q, p); q2 = np.where(swap, p, q)
            keep = np.abs(ytr[a] - ytr[p2]) < np.abs(ytr[a] - ytr[q2])
        return a[keep], p2[keep], q2[keep]

    trip = nn.TripletMarginLoss(margin=margin)
    bs = 2048
    n_kept_per_pass = []
    for _ in range(epochs):
        a_np, p_np, q_np = sample_triplets()
        n_kept_per_pass.append(int(len(a_np)))
        if len(a_np) == 0:
            continue                              # nothing to learn from this draw
        a = torch.tensor(a_np, device=dev)
        p = torch.tensor(p_np, device=dev)
        q = torch.tensor(q_np, device=dev)
        perm = torch.randperm(len(a), device=dev)
        for s in range(0, len(a), bs):
            idx = perm[s:s + bs]
            ea, ep_, eq = enc(Xtr[a[idx]]), enc(Xtr[p[idx]]), enc(Xtr[q[idx]])
            loss = trip(ea, ep_, eq)
            opt.zero_grad(); loss.backward(); opt.step()

    def transform(Xb):
        with torch.no_grad():
            return enc(torch.tensor(Xb, dtype=torch.float32, device=dev)).cpu().numpy()
    transform.n_kept_per_pass = n_kept_per_pass
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
            mid = CHEMBERTA if which == 'chemberta' else MOLFORMER
            out = os.path.join(args.data_dir,
                               emb_cache_name(name, which, mid, smi))
            if os.path.exists(out):
                print(f'  cached {out}'); continue
            try:
                emb = embed_transformer(smi, mid)
                np.save(out, emb)
                print(f'  {name} {which}: {emb.shape} -> {out}', flush=True)
            except Exception as e:
                print(f'  FAIL {name} {which}: {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
