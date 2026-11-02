"""Spearman correlation between two representations' pairwise distances, all pairs of reps.
Distance A: Tanimoto for 0/1 reps, Euclidean on standardised features otherwise.
Distance M: Manhattan on standardised features for every rep (the suite default,
src/kirby/similarity/distance_config.py)."""
import sys, os, warnings, itertools, json, numpy as np
warnings.filterwarnings("ignore"); sys.path.insert(0, "/Users/apunt/repos/KIRBy/tests")
from experiments._shared import load_data
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__))
ds = sys.argv[1]; cap = int(sys.argv[2]) if len(sys.argv) > 2 else 0
d = load_data(ds); reps = d["reps"]; n = len(d["labels"])
idx = np.arange(n)
if cap and n > cap:
    idx = np.sort(np.random.default_rng(0).choice(n, cap, replace=False))
out = {"dataset": ds, "n": n, "n_used": len(idx), "reps": {}}
R = {"A": {}, "M": {}}
def rk(D):
    v = rankdata(np.nan_to_num(D, nan=0.0)).astype(np.float32); v -= v.mean(); return v / np.linalg.norm(v)
for r in sorted(reps):
    X = np.asarray(reps[r], float)[idx]
    sd = X.std(0); keep = sd > 0
    Z = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]
    binary = bool(np.isin(X, [0, 1]).all())
    R["A"][r] = rk(pdist(X.astype(bool), "jaccard") if binary else pdist(Z, "euclidean"))
    R["M"][r] = rk(pdist(Z, "cityblock"))
    out["reps"][r] = {"width": X.shape[1], "A": "tanimoto" if binary else "euclid_std",
                      "all_zero_rows": int((np.abs(X).sum(1) == 0).sum())}
    print(r, X.shape[1], out["reps"][r], flush=True)
for tag in R:
    pairs = {f"{a}|{b}": float(np.dot(R[tag][a], R[tag][b])) for a, b in itertools.combinations(sorted(R[tag]), 2)}
    v = np.array(list(pairs.values()))
    out[tag] = {"pairs": pairs, "median": float(np.median(v))}
    s = sorted(pairs.items(), key=lambda t: t[1])
    print(tag, "MEDIAN", round(out[tag]["median"], 3), "LOW", s[:2], "HIGH", s[-2:], flush=True)
json.dump(out, open(f"{HERE}/distcorr_{ds}.json", "w"), indent=1)
