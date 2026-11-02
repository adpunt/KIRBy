"""(d) ECFP4 Tanimoto distance against maximum-common-substructure distance.

500 random molecule pairs per dataset (seed 0), rdFMCS with timeout=2 s,
ringMatchesRingOnly + completeRingsOnly (same settings as src/kirby/similarity/mcs.py),
single process. Two MCS distances from the same search:
  jaccard: 1 - n_mcs / (a + b - n_mcs)   (the repo's definition, mcs.py)
  max:     1 - n_mcs / max(a, b)
ECFP4 Tanimoto distance from the cached ecfp4.npy.
usage: python mcs.py <dataset>
"""
import json, os, sys, time
import numpy as np
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS
RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
ds = sys.argv[1]
N_PAIRS = 500
meta = np.load(f'/Users/apunt/repos/KIRBy/scratchpad/thesis_code_todo/meta_{ds}.npz', allow_pickle=True)
smiles = list(meta['smiles'])
n = len(smiles)
X = np.load(f'/Volumes/seagate/kirby_cache/rep_cache/{ds}/ecfp4.npy', mmap_mode='r')
rng = np.random.RandomState(0)
seen, pairs = set(), []
while len(pairs) < N_PAIRS:
    i, j = rng.randint(0, n, 2)
    if i == j:
        continue
    key = (min(i, j), max(i, j))
    if key in seen:
        continue
    seen.add(key); pairs.append(key)

rows = []
t0 = time.time()
for i, j in pairs:
    a, b = Chem.MolFromSmiles(smiles[i]), Chem.MolFromSmiles(smiles[j])
    if a is None or b is None:
        rows.append((i, j, np.nan, np.nan, np.nan, 0, 1)); continue
    r = rdFMCS.FindMCS([a, b], timeout=2, ringMatchesRingOnly=True, completeRingsOnly=True)
    k, na, nb = r.numAtoms, a.GetNumAtoms(), b.GetNumAtoms()
    jac = 1.0 - k / (na + nb - k) if (na + nb - k) > 0 else 1.0
    mx = 1.0 - k / max(na, nb)
    xa, xb = np.asarray(X[i]), np.asarray(X[j])
    inter = float(np.sum(xa * xb)); uni = float(xa.sum() + xb.sum() - inter)
    tan = 1.0 - (inter / uni if uni > 0 else 1.0)
    rows.append((i, j, tan, jac, mx, int(r.canceled), 0))
R = np.array(rows, dtype=float)
ok = R[:, 6] == 0
done = ok & (R[:, 5] == 0)
res = {
    'dataset': ds, 'n_pairs_sampled': N_PAIRS, 'seed': 0, 'timeout_s': 2,
    'n_unparsed': int((~ok).sum()), 'n_timed_out': int((ok & (R[:, 5] == 1)).sum()),
    'n_used_completed': int(done.sum()),
    'spearman_tanimoto_vs_mcs_jaccard_completed': float(spearmanr(R[done, 2], R[done, 3]).correlation),
    'spearman_tanimoto_vs_mcs_max_completed': float(spearmanr(R[done, 2], R[done, 4]).correlation),
    'spearman_tanimoto_vs_mcs_jaccard_all_parsed': float(spearmanr(R[ok, 2], R[ok, 3]).correlation),
    'median_tanimoto_distance': float(np.median(R[ok, 2])),
    'median_mcs_jaccard_distance': float(np.median(R[ok, 3])),
    'seconds': round(time.time() - t0, 1),
}
np.save(f'{HERE}/mcs_pairs_{ds}.npy', R)
json.dump(res, open(f'{HERE}/mcs_{ds}.json', 'w'), indent=1)
print(res, flush=True)
