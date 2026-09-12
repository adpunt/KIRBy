import numpy as np, time, sys
sys.path.insert(0, 'src')
from kirby.hybrid import _pool_and_rank_aug_features, CANDIDATE_RANKINGS
rng = np.random.RandomState(0)
n = 300
Xb = rng.randn(n, 40)
y = Xb[:, :5].sum(1) + 0.5 * rng.randn(n)
comp = np.column_stack([np.sin(3*Xb[:, 0]), np.cos(3*Xb[:, 1]), rng.randn(n)*0.1 + y])
aug = {'complement': comp, 'noise': rng.randn(n, 20), 'echo': Xb[:, :5]*1.05}
augv = {k: v.copy() for k, v in aug.items()}
groups = rng.randint(0, 30, size=n)
print(f"{'rule':24}{'seconds':>9}   top 5 donors of 28 candidates", flush=True)
for r in CANDIDATE_RANKINGS:
    t = time.time()
    tr, vl, prov = _pool_and_rank_aug_features(
        aug, augv, y, Xb, None, candidate_ranking=r,
        is_classification=False, max_candidates=10, groups_train=groups)
    print(f"{r:24}{time.time()-t:9.2f}   {[p[0] for p in prov[:5]]}", flush=True)
