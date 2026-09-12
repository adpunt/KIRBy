"""Does the ranking network generalise at all? If not, its attributions cannot rank."""
import numpy as np, torch, torch.nn as nn, sys
sys.path.insert(0, 'src')
rng = np.random.RandomState(0)
n, n_signal, n_noise = 940, 40, 300      # the shape of herg_ki's curated pool
Xs = rng.randn(n, n_signal)
y = Xs[:, :8].sum(1) + 0.7*rng.randn(n)
X = np.hstack([Xs, rng.randn(n, n_noise)])
X = (X - X.mean(0)) / (X.std(0) + 1e-9)
Xt = torch.tensor(X, dtype=torch.float32)
yt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
perm = rng.permutation(n); nv = n//5
va, tr = torch.tensor(perm[:nv].copy()), torch.tensor(perm[nv:].copy())
torch.manual_seed(42)
net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(),
                    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
lf = nn.MSELoss()
base = float(((yt[va] - yt[tr].mean())**2).mean())
best, waited, bs = float('inf'), 0, 128
for ep in range(200):
    net.train(); order = torch.randperm(len(tr))
    for s in range(0, len(tr), bs):
        b = tr[order[s:s+bs]]
        opt.zero_grad(); lf(net(Xt[b]), yt[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        vl = float(lf(net(Xt[va]), yt[va]))
    if vl < best - 1e-6: best, waited = vl, 0
    else:
        waited += 1
        if waited >= 10: break
print(f"stopped at epoch {ep}")
print(f"held-out MSE {best:.4f}   predict-the-mean MSE {base:.4f}   "
      f"R2 {1-best/base:+.3f}")

from captum.attr import DeepLift
attr = DeepLift(net).attribute(Xt, baselines=torch.zeros_like(Xt))
score = np.abs(attr.detach().numpy()).mean(0)
order = np.argsort(score)[::-1]
for k in (14, 40, 100):
    hit = int((order[:k] < n_signal).sum())
    print(f"top {k:3d} by mean|attr|: {hit:3d} real, {k-hit:3d} noise")
# the true-signal columns are 0..7; how deep do they sit?
print("ranks of the 8 columns the label is built from:",
      sorted(int(np.where(order == j)[0][0]) for j in range(8)))
# compare with a plain correlation ranking on the same pool
corr = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) for j in range(X.shape[1])])
o2 = np.argsort(corr)[::-1]
for k in (14, 40):
    hit = int((o2[:k] < n_signal).sum())
    print(f"top {k:3d} by |corr| with y : {hit:3d} real, {k-hit:3d} noise")
