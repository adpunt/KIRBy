"""How much two representations agree about which molecules are close.

For each dataset, takes the ten representations DATASET_REPS[ds]['general']
names, computes the Manhattan distance between every pair of molecules on
standardised features, and reports the Spearman correlation between every two
representations' distances over the same molecule pairs.

Reads the saved feature files in $KIRBY_CACHE_DIR/rep_cache/<dataset>/ and
nothing else: no model is loaded and no representation is computed. A
representation with no saved file is named and left out. Datasets with more
than 2,000 molecules use a fixed random sample of 2,000 (seed 0).

    python tests/tools/rep_distance_agreement.py herg_ki caco2 logd ...

Writes rep_distance_agreement_<dataset>.json in the current directory.
"""
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.complexity.complexity_helpers import DATASET_REPS

CAP = 2000
root = Path(os.environ.get('KIRBY_CACHE_DIR',
                           str(Path.home() / '.cache' / 'kirby'))) / 'rep_cache'

for ds in sys.argv[1:]:
    names = sorted(DATASET_REPS[ds]['general'])
    have = [r for r in names if (root / ds / f'{r}.npy').exists()]
    missing = [r for r in names if r not in have]
    if len(have) < 2:
        print(f'{ds}: only {len(have)} saved representations in {root / ds}; '
              f'missing {missing}', flush=True)
        continue
    n = np.load(root / ds / f'{have[0]}.npy', mmap_mode='r').shape[0]
    idx = (np.arange(n) if n <= CAP else
           np.sort(np.random.default_rng(0).choice(n, CAP, replace=False)))
    ranks = {}
    for r in have:
        X = np.load(root / ds / f'{r}.npy', mmap_mode='r')
        if X.shape[0] != n:
            raise ValueError(f'{ds}/{r}: {X.shape[0]} molecules, expected {n}')
        X = np.asarray(X[idx], dtype=float)
        sd = X.std(0)
        keep = sd > 0
        Z = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]
        v = rankdata(pdist(Z, 'cityblock')).astype(np.float32)
        v -= v.mean()
        ranks[r] = v / np.linalg.norm(v)
        del X, Z
    pairs = {f'{a}|{b}': float(ranks[a] @ ranks[b])
             for a, b in itertools.combinations(have, 2)}
    order = sorted(pairs.items(), key=lambda t: t[1])
    median = float(np.median(list(pairs.values())))
    print(f'{ds}: {n} molecules, {len(idx)} used, {len(have)} representations, '
          f'{len(pairs)} pairs of representations. median {median:.3f}. '
          f'least alike {order[0][0]} {order[0][1]:.3f}. '
          f'most alike {order[-1][0]} {order[-1][1]:.3f}. '
          f'missing: {missing or "none"}', flush=True)
    json.dump({'dataset': ds, 'n': n, 'n_used': len(idx), 'missing': missing,
               'median': median, 'pairs': pairs},
              open(f'rep_distance_agreement_{ds}.json', 'w'), indent=1)
