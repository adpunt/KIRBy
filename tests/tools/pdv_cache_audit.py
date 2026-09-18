#!/usr/bin/env python
"""Locate, quarantine and rebuild the stale `pdv` matrices.

WHAT IS WRONG. `create_pdv` returns RAW descriptors today. It used to end with
`StandardScaler().fit_transform(X)` over every molecule handed in, and the
docstring at src/kirby/representations/molecular.py:985-996 records why that was
removed: each descriptor was centred and scaled using every molecule, held-out
ones included, so the held-out molecules' own mean and spread sat inside the
training features.

Every cached `pdv.npy` on the server still has it. Checked 2026-09-18 with
opig_shared_vectors_census --check-freshness 200: on caco2 and on herg_ki, 200
of 200 sampled rows differ from what the code builds now.

⛔ THERE ARE TWO CACHES FOR THE SAME MATRIX AND THE WRONG ONE WINS.

  rep_cache/<dataset>/pdv.npy        keyed by DATASET NAME, has _identity.json
  reps/<sha of the SMILES>/pdv.npy   keyed by CONTENT, no identity file at all

find_best_hybrid_molecular.py:665 consults the second one BEFORE building
anything, so deleting the first alone achieves nothing: the old matrix is read
out of `reps/` and written straight back into `rep_cache/`. That is exactly what
happened on the first attempt at this cleanup, and it is why `_identity.json`
never caught the staleness — it guards the layer that loses.

WHICH DATASETS. `pdv` is in the `general` pool of six: adme_hlm, adme_mdr1,
adme_rlm, caco2, chembl_qsar, herg_ki. It is cached on more than six, because
`reps/` is keyed by SMILES rather than by dataset, but a dataset that does not
carry it in a pool is unaffected whatever its cache holds.

USAGE
    python tests/tools/pdv_cache_audit.py                  # report only
    python tests/tools/pdv_cache_audit.py --quarantine     # move both layers aside
    python tests/tools/pdv_cache_audit.py --rebuild        # rebuild and check the means

Nothing is deleted. --quarantine MOVES files into a dated directory beside the
cache, so every one of them is recoverable.
"""
import argparse
import hashlib
import os
import shutil
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

warnings.filterwarnings('ignore')
import numpy as np

STAMP = '20260918'
POOL_DATASETS = ('adme_hlm', 'adme_mdr1', 'adme_rlm',
                 'caco2', 'chembl_qsar', 'herg_ki')


def cache_root():
    return Path(os.environ.get('KIRBY_CACHE_DIR',
                               str(Path.home() / '.cache' / 'kirby')))


def smiles_hash(smiles):
    """The key find_best_hybrid_molecular.py:554 builds for reps/<hash>/."""
    return hashlib.sha256('\n'.join(list(smiles)).encode()).hexdigest()[:16]


def looks_zscored(X):
    """Every non-constant column at mean 0 and spread 1 over ALL rows."""
    X = np.asarray(X, dtype=np.float64)
    keep = X.std(axis=0) > 1e-9
    if not keep.any():
        return None
    mu = np.abs(X[:, keep].mean(axis=0)).max()
    sd = np.abs(X[:, keep].std(axis=0) - 1.0).max()
    return bool(mu < 1e-3 and sd < 1e-3)


def find_for(dataset):
    """Both cache paths for one dataset, and whether each exists."""
    from experiments._shared import load_data
    root = cache_root()
    out = {'dataset': dataset,
           'rep_cache': root / 'rep_cache' / dataset / 'pdv.npy',
           'reps': None, 'error': None}
    try:
        data = load_data(dataset, rep_preset='full', rep_names=['ecfp4'])
    except Exception as exc:
        out['error'] = f'{type(exc).__name__}: {exc}'
        return out
    smiles = data.get('smiles')
    if smiles is None:
        out['error'] = f"load_data returned no 'smiles'; keys are {sorted(data)}"
        return out
    out['reps'] = root / 'reps' / smiles_hash(smiles) / 'pdv.npy'
    return out


def report(entries):
    print(f'\n{"dataset":14s} {"layer":10s} {"exists":7s} {"z-scored":9s} path')
    print('-' * 100)
    for e in entries:
        if e['error']:
            print(f'{e["dataset"]:14s} {"-":10s} {"-":7s} {"-":9s} {e["error"]}')
            continue
        for layer in ('reps', 'rep_cache'):     # reps first: it wins
            p = e[layer]
            if p is None:
                continue
            ex = p.exists()
            z = '-'
            if ex:
                try:
                    z = {True: 'YES', False: 'no', None: '?'}[looks_zscored(np.load(p))]
                except Exception as exc:
                    z = f'err {type(exc).__name__}'
            print(f'{e["dataset"]:14s} {layer:10s} {str(ex):7s} {z:9s} {p}')


def quarantine(entries):
    dest = cache_root() / f'pdv_stale_{STAMP}'
    dest.mkdir(parents=True, exist_ok=True)
    moved = 0
    for e in entries:
        for layer in ('reps', 'rep_cache'):
            p = e[layer]
            if p is None or not p.exists():
                continue
            target = dest / f'{layer}_{e["dataset"]}_{p.parent.name}_pdv.npy'
            shutil.move(str(p), str(target))
            print(f'  moved {p}  ->  {target.name}')
            moved += 1
    print(f'\n{moved} file(s) moved into {dest}. Nothing deleted.')
    return moved


def rebuild(datasets):
    from experiments._shared import load_data
    ok = True
    for ds in datasets:
        try:
            X = load_data(ds, rep_preset='full', rep_names=['pdv'])['reps']['pdv']
        except Exception as exc:
            print(f'{ds:14s} REBUILD FAILED {type(exc).__name__}: {exc}')
            ok = False
            continue
        z = looks_zscored(X)
        means = [round(float(v), 2) for v in np.asarray(X).mean(axis=0)[:4]]
        verdict = ('⛔ STILL Z-SCORED — another cache layer served it'
                   if z else 'raw, as the code builds it')
        print(f'{ds:14s} {tuple(X.shape)}  first column means {means}   {verdict}')
        ok = ok and not z
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=list(POOL_DATASETS))
    ap.add_argument('--quarantine', action='store_true',
                    help='move both cache layers aside (files are moved, never deleted)')
    ap.add_argument('--rebuild', action='store_true',
                    help='rebuild pdv and say whether it came back raw')
    args = ap.parse_args()

    print(f'cache root: {cache_root()}')
    entries = [find_for(ds) for ds in args.datasets]
    report(entries)

    if args.quarantine:
        print('\n--- quarantine ---')
        quarantine(entries)
    if args.rebuild:
        print('\n--- rebuild ---')
        good = rebuild(args.datasets)
        print('\n' + ('every rebuild came back raw — safe to re-check freshness '
                      'and then wipe the result databases'
                      if good else
                      '⛔ at least one came back z-scored. STOP. Do not delete any '
                      'result database: a cache layer is still serving the old '
                      'matrix and the rebuild has not happened.'))
