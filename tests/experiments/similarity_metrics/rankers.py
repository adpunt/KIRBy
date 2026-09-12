"""Feature scorers — the similarity-aware alternatives to the predictive
`compute_feature_importance`. Each scores every feature; the search keeps the
top-scoring ones (drop-in for `np.argsort(importances)`).

RReliefF and the Laplacian score read molecules. _spearman_cols scores each
feature by the Spearman between that feature's absolute difference across a
molecule pair and a target quantity on the same pairs, the target being either
the absolute activity gap (supervised) or a structural distance (label-free).
combine_ranks fuses several score vectors by averaging their ranks.

The five scorers built on that pair recipe — R1/R2 per-feature Spearman, R3
distance correlation, R4 mutual information and R7 kernel-target alignment —
had no caller anywhere in the repository and are deleted, together with the
RANKERS dict and the LABEL_FREE set that indexed them.

All return a (d,) array, higher = keep. Label-free scorers never see y.
"""
import numpy as np
from scipy.stats import rankdata


# ---------- pair sampling + per-feature |df| ----------
def _pairs(n, m, seed):
    r = np.random.RandomState(seed)
    i, j = r.randint(0, n, m), r.randint(0, n, m)
    ok = i != j
    return i[ok], j[ok]


def _spearman_cols(DF, t):
    """Spearman of each column of DF (m,d) against vector t (m,) -> (d,)."""
    tr = rankdata(t); tr = tr - tr.mean()
    Rr = np.apply_along_axis(rankdata, 0, DF)
    Rr = Rr - Rr.mean(0)
    num = Rr.T @ tr
    den = np.sqrt((Rr ** 2).sum(0) * (tr ** 2).sum())
    out = np.zeros(DF.shape[1]); nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


# ---------- distance correlation (nonlinear dependence between two vectors) ----------
def _dcor(a, b):
    a = a[:, None]; b = b[:, None]
    A = np.abs(a - a.T); B = np.abs(b - b.T)
    A = A - A.mean(0) - A.mean(1)[:, None] + A.mean()
    B = B - B.mean(0) - B.mean(1)[:, None] + B.mean()
    dcov = np.sqrt(max((A * B).mean(), 0))
    da = np.sqrt(max((A * A).mean(), 0)); db = np.sqrt(max((B * B).mean(), 0))
    return dcov / np.sqrt(da * db) if da > 0 and db > 0 else 0.0


# ---------- R5: RReliefF (NN pairs, resists collapse) ----------
def r5_rrelieff(X, y, struct=None, seed=0, max_n=800, m=250, k=30, sigma=20.0):
    from sklearn.neighbors import NearestNeighbors
    rs = np.random.RandomState(seed); n = len(X)
    if n > max_n:
        s = rs.choice(n, max_n, replace=False); X, y = X[s], y[s]; n = max_n
    d = X.shape[1]
    rng = np.ptp(X, 0); rng[rng == 0] = 1.0
    Xn = (X - X.min(0)) / rng; rng_y = float(np.ptp(y)) or 1.0
    m = min(m, n); k = min(k, n - 1)
    anchors = rs.choice(n, m, replace=False)
    nn = NearestNeighbors(n_neighbors=k + 1, metric='manhattan').fit(Xn)
    nb = nn.kneighbors(Xn[anchors])[1][:, 1:]
    w = np.exp(-(np.arange(1, k + 1) / sigma) ** 2); w /= w.sum()
    N_dC = 0.0; N_dA = np.zeros(d); N_dCdA = np.zeros(d)
    for s in range(0, m, 25):
        a = anchors[s:s + 25]; nbc = nb[s:s + 25]; c = len(a)
        dA = np.abs(Xn[a][:, None, :] - Xn[nbc])
        dC = np.abs(y[a][:, None] - y[nbc]) / rng_y
        wk = np.broadcast_to(w, (c, k))
        N_dC += float((dC * wk).sum())
        N_dA += np.einsum('ck,ckd->d', wk, dA)
        N_dCdA += np.einsum('ck,ckd->d', dC * wk, dA)
    den = m - N_dC
    if N_dC <= 0 or den <= 0:
        return np.zeros(d)
    return N_dCdA / N_dC - (N_dA - N_dCdA) / den


# ---------- R6: Laplacian score (unsupervised) ----------
def r6_laplacian(X, y=None, struct=None, seed=0, k=5, max_n=900):
    from sklearn.neighbors import NearestNeighbors
    n = len(X)
    if n > max_n:
        idx = np.random.RandomState(seed).choice(n, max_n, replace=False); X = X[idx]; n = max_n
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(X)
    dist, ind = nn.kneighbors(X)
    sig2 = max(np.mean(dist[:, 1:] ** 2), 1e-12)
    W = np.zeros((n, n))
    for a in range(n):
        for bi, b in enumerate(ind[a, 1:]):
            wv = np.exp(-dist[a, bi + 1] ** 2 / sig2); W[a, b] = wv; W[b, a] = wv
    D = W.sum(1); Dsum = D.sum()
    fbar = X - (X * D[:, None]).sum(0) / Dsum
    quad = (fbar * (D[:, None] * fbar)).sum(0) - (fbar * (W @ fbar)).sum(0)
    num = 2.0 * quad; den = (fbar ** 2 * D[:, None]).sum(0)
    score = np.where(den > 0, num / np.maximum(den, 1e-12), np.inf)
    return -score


# ---------- combinations (user's hedge: activity + structural) ----------
def combine_ranks(*scores, weights=None):
    """Fuse several ranker score-vectors by averaging their RANKS (higher=better).
    weights optional. Returns a fused score (higher=keep)."""
    weights = weights or [1.0] * len(scores)
    R = np.zeros(len(scores[0]))
    for s, w in zip(scores, weights):
        R += w * rankdata(s)
    return R
