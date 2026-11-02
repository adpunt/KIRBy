import sys, json, itertools, numpy as np
sys.path.insert(0, "/Users/apunt/repos/KIRBy/tests")
from experiments.complexity.complexity_helpers import DATASET_REPS
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
ds, root = sys.argv[1], sys.argv[2]
names = sorted(DATASET_REPS[ds]["general"])
n = np.load(f"{root}/{ds}/{names[0]}.npy", mmap_mode="r").shape[0]
idx = np.arange(n) if n <= 2000 else np.sort(np.random.default_rng(0).choice(n, 2000, replace=False))
R = {}
for r in names:
    X = np.asarray(np.load(f"{root}/{ds}/{r}.npy", mmap_mode="r")[idx], float)
    sd = X.std(0); k = sd > 0
    v = rankdata(pdist((X[:, k] - X[:, k].mean(0)) / sd[k], "cityblock")).astype(np.float32)
    v -= v.mean(); R[r] = v / np.linalg.norm(v); del X
pairs = {f"{a}|{b}": float(R[a] @ R[b]) for a, b in itertools.combinations(names, 2)}
s = sorted(pairs.items(), key=lambda t: t[1])
print(ds, n, len(idx), "median", round(float(np.median(list(pairs.values()))), 3), "low", s[0], "high", s[-1])
json.dump({"n": n, "used": len(idx), "pairs": pairs}, open(f"safe_{ds}.json", "w"), indent=1)
