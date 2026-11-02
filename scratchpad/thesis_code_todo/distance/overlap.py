"""(c) overlap of the five-nearest-neighbour sets (neighbours from other scaffold
groups, the same sets as (b)) between ECFP4, a descriptor set and SMI-TED.
Reads knn5_<ds>_<rep>.npy written by compute.py. Mean fraction of the 5 shared, per molecule."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/apunt/repos/KIRBy/tests/experiments/complexity')
from complexity_helpers import DATASET_REPS
out = {}
for ds in ['herg_ki', 'caco2', 'logd', 'qm9', 'adme_mdr1', 'adme_hlm', 'adme_rlm']:
    pool = DATASET_REPS[ds]['general']
    desc = 'mordred' if 'mordred' in pool else 'pdv'
    emb = 'smited'
    trio = ['ecfp4', desc, emb]
    if not all(os.path.exists(f'{HERE}/knn5_{ds}_{r}.npy') for r in trio):
        continue
    K = {r: np.load(f'{HERE}/knn5_{ds}_{r}.npy') for r in trio}

    def ov(a, b):
        A, B = K[a], K[b]
        return float(np.mean([len(set(A[i]) & set(B[i])) / 5.0 for i in range(len(A))]))
    out[ds] = {'descriptor': desc, 'embedding': emb, 'ecfp4_in_pool': 'ecfp4' in pool,
               'ecfp4_vs_desc': ov('ecfp4', desc), 'ecfp4_vs_emb': ov('ecfp4', emb),
               'desc_vs_emb': ov(desc, emb), 'n_molecules': int(len(K['ecfp4']))}
    print(ds, out[ds])
json.dump(out, open(f'{HERE}/overlap.json', 'w'), indent=1)
