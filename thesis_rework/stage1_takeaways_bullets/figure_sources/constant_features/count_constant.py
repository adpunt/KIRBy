"""Constant / near-constant feature counts per dataset and representation.
Reads only saved feature files (rep_cache/<ds>/<rep>.npy) and meta_<ds>.npz. Builds nothing."""
import json, os, sys
import numpy as np
sys.path.insert(0, '/Users/apunt/repos/KIRBy/tests/experiments/complexity')
from complexity_helpers import DATASET_REPS

R = '/Volumes/seagate/kirby_cache/rep_cache'
META = '/Users/apunt/repos/KIRBy/scratchpad/thesis_code_todo'
DS = ['herg_ki', 'caco2', 'logd', 'qm9', 'adme_mdr1', 'adme_hlm', 'adme_rlm', 'herg', 'lhasa_herg']
THR = 0.95
out = {'threshold_near_constant': THR, 'datasets': {}}

def count(X):
    n, p = X.shape
    const = near = 0
    const_on = const_off = 0
    nan_feats = 0
    top_share = np.empty(p)
    for j0 in range(0, p, 256):
        B = np.array(X[:, j0:j0 + 256], dtype=np.float64)
        nan_feats += int(np.isnan(B).any(axis=0).sum())
        for k in range(B.shape[1]):
            v, c = np.unique(B[:, k], return_counts=True, equal_nan=True)
            m = c.max()
            top_share[j0 + k] = m / n
            if len(v) == 1:
                const += 1
                if v[0] == 1: const_on += 1
                elif v[0] == 0: const_off += 1
            if m / n >= THR:
                near += 1
    Xs = np.asarray(X[:min(n, 2000)])
    binary = bool(np.isin(Xs, [0, 1]).all())
    if binary:
        binary = bool(np.isin(np.asarray(X), [0, 1]).all())
    return dict(n_molecules=n, total=p, constant=const, near_constant=near,
                constant_on=const_on, constant_off=const_off, binary=binary,
                features_with_nan=nan_feats, all_zero=bool(not np.asarray(X).any()))

for ds in DS:
    meta = np.load(f'{META}/meta_{ds}.npz', allow_pickle=True)
    n_meta = len(meta['smiles'])
    pool = list(DATASET_REPS.get(ds, {}).get('general') or [])
    aug = list(DATASET_REPS.get(ds, {}).get('augmentation') or [])
    d = {'n_molecules_meta': n_meta, 'general_pool': pool, 'augmentation': aug, 'reps': {}}
    for f in sorted(os.listdir(f'{R}/{ds}')):
        if not f.endswith('.npy'): continue
        rep = f[:-4]
        try:
            X = np.load(f'{R}/{ds}/{f}', mmap_mode='r')
        except ValueError as e:
            d['reps'][rep] = {'error': f'not a numeric array: {e}'}
            print(ds, rep, 'SKIP', e, flush=True); continue
        if X.ndim != 2 or X.shape[0] != n_meta:
            d['reps'][rep] = {'error': f'shape {X.shape} vs {n_meta} molecules'}
            print(ds, rep, 'SKIP shape', X.shape, flush=True); continue
        r = count(X)
        if rep == 'maccs':
            col0 = np.asarray(X[:, 0]); r['bit0_always_off'] = bool((col0 == 0).all())
        d['reps'][rep] = r
        print(ds, rep, r, flush=True)
    out['datasets'][ds] = d

a = np.load(f'{META}/meta_herg.npz', allow_pickle=True); b = np.load(f'{META}/meta_lhasa_herg.npz', allow_pickle=True)
out['herg_vs_lhasa_herg'] = {
    'same_smiles_in_order': bool(len(a['smiles']) == len(b['smiles']) and (a['smiles'] == b['smiles']).all()),
    'same_smiles_as_set': bool(set(a['smiles'].tolist()) == set(b['smiles'].tolist())),
    'same_labels': bool(len(a['labels']) == len(b['labels']) and (np.asarray(a['labels']) == np.asarray(b['labels'])).all()),
    'smiles_overlap_herg_ki': len(set(a['smiles'].tolist()) & set(np.load(f'{META}/meta_herg_ki.npz', allow_pickle=True)['smiles'].tolist())),
}
print(out['herg_vs_lhasa_herg'])
json.dump(out, open('/Users/apunt/repos/KIRBy/scratchpad/thesis_code_todo/constant/constant_counts.json', 'w'), indent=1)
