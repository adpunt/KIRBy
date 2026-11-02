import sys, warnings, numpy as np, json
warnings.filterwarnings("ignore"); sys.path.insert(0, "tests")
from experiments._shared import load_data

d = load_data("herg_ki")
reps, y, g = d["reps"], np.asarray(d["labels"], float), np.asarray(d["groups"])
n = len(y)

def tied(X):
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    _, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    return int((cnt[inv] > 1).sum())

print(f"RESULT herg_ki {n} molecules")
for r in sorted(reps):
    X = reps[r]; t = tied(X)
    print(f"RESULT  {r:<14}{X.shape[1]:>6}f  {t:>5}  {100*t/n:6.2f}%", flush=True)
names = sorted(reps)
cat = np.hstack([np.asarray(reps[r], float) for r in names])
t = tied(cat)
print(f"RESULT  {'ALL-10-concat':<14}{cat.shape[1]:>6}f  {t:>5}  {100*t/n:6.2f}%", flush=True)
np.save("/tmp/herg_cat.npy", cat)
json.dump({r: int(np.asarray(reps[r]).shape[1]) for r in names}, open("/tmp/herg_widths.json", "w"))
