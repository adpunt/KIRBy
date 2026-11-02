"""Shared-vector census on herg_ki, logd, caco2 from cached feature files only.
Builds no representation. Writes ties_results.json next to this file."""
import sys, json, os, numpy as np
from collections import defaultdict
sys.path.insert(0, '/Users/apunt/repos/KIRBy/tests/experiments/complexity')
from complexity_helpers import DATASET_REPS
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

TD = '/Users/apunt/repos/KIRBy/scratchpad/thesis_code_todo'
CACHE = '/Volumes/seagate/kirby_cache/rep_cache'
OUT = os.path.join(TD, 'ties', 'ties_results.json')

def mol_keys(smiles):
    full, block1, flat = [], [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            full.append(None); block1.append(None); flat.append(None); continue
        k = Chem.MolToInchiKey(m)
        full.append(k); block1.append(k.split('-')[0])
        flat.append(Chem.MolToSmiles(m, isomericSmiles=False))
    return full, block1, flat

def census(inv, X, y, full, block1, flat):
    n = len(y)
    groups = defaultdict(list)
    for i, g in enumerate(inv):
        groups[g].append(i)
    tied = [g for g in groups.values() if len(g) > 1]
    zero_rows = (X == 0).all(axis=1) if X is not None else np.zeros(n, bool)
    n_tied = sum(len(g) for g in tied)
    same_mol = zero = distinct = stereo_only = stereo_only_flat = has_stereo_partner = 0
    sse = 0.0; spreads = []; n_diff = 0; n_stereo_groups = 0
    group_detail = []
    for g in tied:
        yg = y[g]; sse += float(((yg - yg.mean()) ** 2).sum())
        sp = float(yg.max() - yg.min()); spreads.append(sp)
        if sp > 0: n_diff += 1
        if len(set(block1[i] for i in g)) == 1: n_stereo_groups += 1
        for i in g:
            others = [j for j in g if j != i]
            if zero_rows[i]:
                zero += 1
            elif any(full[j] == full[i] for j in others):
                same_mol += 1
            else:
                distinct += 1
                if all(block1[j] == block1[i] for j in others): stereo_only += 1
                if all(flat[j] == flat[i] for j in others): stereo_only_flat += 1
                if any(block1[j] == block1[i] for j in others): has_stereo_partner += 1
        group_detail.append({'idx': [int(i) for i in g], 'labels': [float(v) for v in yg], 'spread': sp})
    var = float(np.var(y))
    rmse = float(np.sqrt(sse / n))
    return {
        'molecules': n, 'tied': n_tied, 'tied_pct': 100 * n_tied / n,
        'same_molecule': same_mol, 'zero_row': zero, 'distinct': distinct,
        'stereo_only': stereo_only, 'stereo_only_by_flat_smiles': stereo_only_flat,
        'with_a_stereo_partner': has_stereo_partner,
        'not_stereo_only': distinct - stereo_only,
        'rmse_min': rmse, 'best_r2': 1 - rmse ** 2 / var, 'label_var': var, 'label_sd': float(np.sqrt(var)),
        'n_groups': len(tied), 'n_groups_differing_labels': n_diff, 'n_groups_all_one_inchikey_block1': n_stereo_groups,
        'max_spread': max(spreads) if spreads else 0.0,
        'mean_spread': float(np.mean(spreads)) if spreads else 0.0,
        'n_groups_spread_gt_0p54': int(sum(s > 0.54 for s in spreads)),
        'largest_group': max((len(g) for g in tied), default=0),
        'groups': group_detail,
    }

res = {}
for ds in ['herg_ki', 'logd', 'caco2']:
    meta = np.load(f'{TD}/meta_{ds}.npz', allow_pickle=True)
    smiles = list(meta['smiles']); y = np.asarray(meta['labels'], float)
    full, block1, flat = mol_keys(smiles)
    general = DATASET_REPS[ds]['general']; aug = DATASET_REPS[ds]['augmentation']
    cached = sorted(f[:-4] for f in os.listdir(f'{CACHE}/{ds}') if f.endswith('.npy'))
    reps = list(dict.fromkeys(general + aug + [r for r in cached if r not in general + aug]))
    d = {'n': len(y), 'general': general, 'augmentation': aug,
         'n_unparsed_smiles': sum(k is None for k in full),
         'n_duplicate_inchikey': len(full) - len(set(full)),
         'label_sd': float(np.std(y)), 'reps': {}}
    invs = {}
    for r in reps:
        p = f'{CACHE}/{ds}/{r}.npy'
        if not os.path.exists(p):
            d['reps'][r] = {'missing': True}; continue
        X = np.ascontiguousarray(np.load(p, mmap_mode='r'))
        assert X.shape[0] == len(y), (ds, r, X.shape)
        info = {'features': int(X.shape[1]), 'dtype': str(X.dtype),
                'nan': int(np.isnan(X).sum()) if X.dtype.kind == 'f' else 0,
                'all_zero_rows': int((X == 0).all(axis=1).sum()),
                'in_general': r in general, 'in_augmentation': r in aug}
        if info['all_zero_rows'] == len(y):
            info['excluded'] = 'every row is zero; features are not real'
            d['reps'][r] = info; del X; continue
        _, inv = np.unique(X, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        invs[r] = inv
        info.update(census(inv, X, y, full, block1, flat))
        d['reps'][r] = info
        print(ds, r, X.shape[1], info['tied'], info['stereo_only'], round(info['rmse_min'], 4), flush=True)
        del X
    # concatenation of all general reps: two molecules tie iff they tie in every rep
    key = np.stack([invs[r] for r in general if r in invs], axis=1)
    _, cinv = np.unique(key, axis=0, return_inverse=True)
    c = census(np.asarray(cinv).ravel(), None, y, full, block1, flat)
    c['reps_joined'] = [r for r in general if r in invs]
    d['concat_general'] = c
    print(ds, 'concat_general', c['tied'], flush=True)
    d['smiles'] = smiles; d['inchikey'] = full
    res[ds] = d

json.dump(res, open(OUT, 'w'), indent=1)
print('wrote', OUT)
