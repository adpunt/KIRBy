"""Feature-importance rankers (R1-R9) — the similarity-aware alternatives to the
predictive `compute_feature_importance`. Each scores every feature; the search
keeps the top-scoring ones (drop-in for `np.argsort(importances)`).

Base recipe (R1-R4, R7): sample pairs (i,j); df = |f_i - f_j|; score feature f by
the DEPENDENCE between df and a TARGET-similarity quantity:
  target = |dy|                 -> activity-based (supervised; SAC-collapse risk)
  target = structural distance  -> label-free (MCS / seq-edit; the different one)
Combinations fuse an activity ranker with a structural one (user's hedge).

All return a (d,) array, higher = keep. Label-free rankers never see y.
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


# ---------- R1 / R2: per-feature Spearman(df, target) ----------
def r1_activity_spearman(X, y=None, struct=None, seed=0, m=8000):
    i, j = _pairs(len(X), m, seed)
    DF = np.abs(X[i] - X[j]); t = np.abs(y[i] - y[j])
    return _spearman_cols(DF, t)


def r2_structural_spearman(X, y=None, struct=None, seed=0, m=8000):
    """struct = precomputed (m,) structural distances for the SAME sampled pairs,
    or an (n,n) reference to index. Label-free."""
    i, j = _pairs(len(X), m, seed)
    DF = np.abs(X[i] - X[j])
    t = struct[i, j] if struct.ndim == 2 else struct[:len(i)]
    return _spearman_cols(DF, t)


# ---------- R3: distance correlation (nonlinear) ----------
def _dcor(a, b):
    a = a[:, None]; b = b[:, None]
    A = np.abs(a - a.T); B = np.abs(b - b.T)
    A = A - A.mean(0) - A.mean(1)[:, None] + A.mean()
    B = B - B.mean(0) - B.mean(1)[:, None] + B.mean()
    dcov = np.sqrt(max((A * B).mean(), 0))
    da = np.sqrt(max((A * A).mean(), 0)); db = np.sqrt(max((B * B).mean(), 0))
    return dcov / np.sqrt(da * db) if da > 0 and db > 0 else 0.0


def r3_distance_corr(X, y=None, struct=None, seed=0, m=1500, target='activity'):
    i, j = _pairs(len(X), m, seed)
    DF = np.abs(X[i] - X[j])
    t = np.abs(y[i] - y[j]) if target == 'activity' else (struct[i, j] if struct.ndim == 2 else struct[:len(i)])
    return np.array([_dcor(DF[:, k], t) for k in range(DF.shape[1])])


# ---------- R4: mutual information ----------
def r4_mutual_info(X, y=None, struct=None, seed=0, m=6000, target='activity'):
    from sklearn.feature_selection import mutual_info_regression
    i, j = _pairs(len(X), m, seed)
    DF = np.abs(X[i] - X[j])
    t = np.abs(y[i] - y[j]) if target == 'activity' else (struct[i, j] if struct.ndim == 2 else struct[:len(i)])
    return mutual_info_regression(DF, t, random_state=seed)


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


# ---------- R7: kernel-target alignment (per feature) ----------
def r7_kta(X, y=None, struct=None, seed=0, m=3000, target='activity'):
    """Alignment between each feature's pair-distance vector and the target
    pair-distance vector (cosine of centered vectors) — a marginal KTA."""
    i, j = _pairs(len(X), m, seed)
    DF = np.abs(X[i] - X[j])
    t = np.abs(y[i] - y[j]) if target == 'activity' else (struct[i, j] if struct.ndim == 2 else struct[:len(i)])
    t = t - t.mean(); DFc = DF - DF.mean(0)
    num = DFc.T @ t
    den = np.sqrt((DFc ** 2).sum(0) * (t ** 2).sum())
    out = np.zeros(DF.shape[1]); nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


# ---------- combinations (user's hedge: activity + structural) ----------
def combine_ranks(*scores, weights=None):
    """Fuse several ranker score-vectors by averaging their RANKS (higher=better).
    weights optional. Returns a fused score (higher=keep)."""
    weights = weights or [1.0] * len(scores)
    R = np.zeros(len(scores[0]))
    for s, w in zip(scores, weights):
        R += w * rankdata(s)
    return R


RANKERS = {
    'r1_activity': r1_activity_spearman,      # supervised, SAC-collapse risk
    'r2_structural': r2_structural_spearman,  # label-free (MCS/seq)
    'r3_dcor': r3_distance_corr,              # nonlinear dependence
    'r4_mi': r4_mutual_info,                  # any dependence
    'r5_rrelieff': r5_rrelieff,              # local NN, resists collapse
    'r6_laplacian': r6_laplacian,            # unsupervised locality
    'r7_kta': r7_kta,                        # kernel-target alignment
}
LABEL_FREE = {'r2_structural', 'r6_laplacian'}
# R8 HSIC-Lasso and R9 greedy-forward are SET selectors (reward complementarity),
# not per-feature scores — they wrap a greedy harness; added in the ranker experiment.
