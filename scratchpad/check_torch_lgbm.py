import numpy as np, sys
sys.path.insert(0, 'src')
from kirby.hybrid import _pool_and_rank_aug_features
rng = np.random.RandomState(0); n = 300
Xb = rng.randn(n, 40); y = Xb[:, :5].sum(1) + 0.5*rng.randn(n)
aug = {'a': rng.randn(n, 10)}; augv = {'a': aug['a'].copy()}
_pool_and_rank_aug_features(aug, augv, y, Xb, None,
                            candidate_ranking='deeplift',
                            is_classification=False, max_candidates=5)
print('deeplift ranking done', flush=True)
import lightgbm as lgb
m = lgb.LGBMRegressor(n_estimators=50, random_state=42,
                      n_jobs=int(__import__('os').environ.get('KIRBY_LGBM_THREADS', '-1')),
                      verbose=-1)
m.fit(Xb, y)
print('downstream lightgbm fit done', flush=True)
