"""Which two downstream models for QM9 and the ChEMBL targets (Adelaide, 2026-11-04).

Each of dnn, lightgbm, svm, rf and xgboost is scored on three inputs: a hybrid
built from one fixed stored recipe, ECFP4 and PDV. Five scaffold folds, RMSE,
with the seconds each fit took and the seconds each fold's hybrid took to build.
A fixed recipe means only the model differs between rows of one input.

    python tests/experiments/model_choice_test.py --out DIR qm9 qm9_mu qm9_alpha
    python tests/experiments/model_choice_test.py --out DIR --table

Safe to repeat: a fit already in DIR/<dataset>.jsonl is not run again.

Recipes. QM9: dnn's best stored recipe on the gap with no added features
(slim_precompute/qm9.slim.db, 2026-10-03: power_2, 1,000 features, lightgbm_gain,
lightgbm evaluator, all ten representations). Its feature ranking is redone on
each target's own training labels. ChEMBL: fold 0's lightgbm recipe from the
October server run (opig/qsar.json), on the eight representations that run built.
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / 'src', REPO / 'tests', REPO / 'tests' / 'experiments'):
    sys.path.insert(0, str(p))

MODELS = ['dnn', 'lightgbm', 'svm', 'rf', 'xgboost']
INPUTS = ['hybrid', 'ecfp4', 'pdv']
QM9_RECIPE = dict(allocation_method='power_2', budget=1000,
                  importance_method='lightgbm_gain', internal_evaluator='lightgbm')
RECIPES = {
    'chembl_qsar_d2': dict(allocation_method='rank_exp_d0.3', budget=1000,
                           importance_method='lightgbm_gain', internal_evaluator='rf'),
    'chembl_qsar_fxa': dict(allocation_method='rank_exp_d0.3', budget=1000,
                            importance_method='lightgbm_gain', internal_evaluator='rf'),
    'chembl_qsar_mpro': dict(allocation_method='performance_weighted', budget=1000,
                             importance_method='lightgbm_gain', internal_evaluator='rf'),
}
CHEMBL_REPS = ['ecfp4', 'pdv', 'avalon', 'atom_pair', 'graph_kernel', 'mordred',
               'mhg_gnn', 'molformer']
FEATURE_TIMEOUT_S = 1800


def qm9_features(ds):
    """(smiles, labels, {rep: rows}) at 5,000 molecules. Taken from the stored
    10,000-molecule files when they match, built for these rows otherwise."""
    import numpy as np
    from rdkit import Chem, RDLogger
    import find_best_hybrid_molecular as F
    import kirby.datasets.qm9 as Q
    from experiments import _shared as S
    from experiments.complexity.complexity_helpers import DATASET_REPS
    RDLogger.DisableLog('rdApp.*')
    smiles, y, _ = F.load_dataset(ds)
    real = Q.load_qm9
    Q.load_qm9 = lambda n_samples=None, **k: real(n_samples=10000, **k)
    try:
        s10, _, _ = F.load_dataset('qm9')
    finally:
        Q.load_qm9 = real
    cache = S._rep_cache_dir() / 'qm9'
    ident = cache / '_identity.json'
    idx = None
    if ident.exists():
        if json.loads(ident.read_text()).get('sha1') == S._rep_identity(s10)['sha1']:
            key = lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s))
            pos = {key(s): i for i, s in enumerate(s10)}
            idx = np.array([pos[key(s)] for s in smiles])
        else:
            print(f'  {cache} holds other rows: building every feature set', flush=True)
    reps = {}
    for r in list(DATASET_REPS['qm9']['general']) + ['pdv']:
        f = cache / f'{r}.npy'
        if idx is not None and f.exists():
            reps[r] = np.load(f)[idx]
            print(f'  {r}: stored file', flush=True)
        else:
            t = time.time()
            reps[r] = F.get_representations(smiles, r)[r]
            print(f'  {r}: built in {time.time() - t:.0f}s', flush=True)
    return smiles, y, reps


def chembl_features(ds, out):
    """Build each feature set in its own process, capped at FEATURE_TIMEOUT_S,
    and record which ones exist so a repeat uses the same pool."""
    from experiments import _shared as S
    built_file = Path(out) / f'{ds}.feature_sets.json'
    if built_file.exists():
        built = json.loads(built_file.read_text())
    else:
        built = []
        for r in CHEMBL_REPS:
            t = time.time()
            try:
                rc = subprocess.run([sys.executable, '-u', __file__, '--build-one', ds, r],
                                    timeout=FEATURE_TIMEOUT_S).returncode
            except subprocess.TimeoutExpired:
                rc = 'timed out'
            print(f'  {ds} {r}: {"ready" if rc == 0 else f"FAILED ({rc})"} '
                  f'after {time.time() - t:.0f}s', flush=True)
            if rc == 0:
                built.append(r)
        built_file.write_text(json.dumps(built))
    missing = {'ecfp4', 'pdv'} - set(built)
    if missing:
        raise SystemExit(f'{ds}: {sorted(missing)} could not be built')
    d = S.load_data(ds, rep_names=built)
    return d['smiles'], d['labels'], d['reps']


def run(ds, out, models, inputs, n_run_folds):
    import torch  # noqa: F401  (before lightgbm: macOS OpenMP order)
    import numpy as np
    from find_best_hybrid_molecular import assign_scaffold_groups
    from kirby.find_best_hybrid import nested_cv_outer_folds, _outer_fit_score
    from kirby.hybrid import create_hybrid, apply_feature_selection

    recipe = QM9_RECIPE if ds.startswith('qm9') else RECIPES[ds]
    smiles, y, reps = (qm9_features(ds) if ds.startswith('qm9')
                       else chembl_features(ds, out))
    pdv, ecfp = (np.nan_to_num(np.asarray(reps[k], dtype=np.float32)) for k in ('pdv', 'ecfp4'))
    pool_names = [r for r in reps if not (ds.startswith('qm9') and r == 'pdv')]
    pool = {r: np.nan_to_num(np.asarray(reps[r], dtype=np.float32)) for r in pool_names}
    y = np.asarray(y, dtype=float)
    groups, n_groups = assign_scaffold_groups(smiles)
    groups = np.asarray(groups)
    print(f'{ds}: {len(y)} molecules, {n_groups} scaffolds, recipe {recipe}, '
          f'hybrid chooses from {pool_names}', flush=True)

    path = Path(out) / f'{ds}.jsonl'
    seen = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                seen.add((r['input'], r['model'], r['fold']))
            except ValueError:
                pass

    def rmse(a, b):
        return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

    folds = nested_cv_outer_folds({'x': np.zeros((len(y), 1))}, y, groups, 5, 42, 'regression')
    for fi, (tr, te) in enumerate(folds[:n_run_folds]):
        need = [(i, m) for i in inputs for m in models if (i, m, fi) not in seen]
        if not need:
            continue
        data = {'ecfp4': (ecfp[tr], ecfp[te], 0.0), 'pdv': (pdv[tr], pdv[te], 0.0)}
        if any(i == 'hybrid' for i, _ in need):
            t = time.time()
            Xtr, info = create_hybrid({r: a[tr] for r, a in pool.items()}, y[tr],
                                      groups=groups[tr], step_size=5, **recipe)
            Xte = apply_feature_selection({r: a[te] for r, a in pool.items()}, info)
            data['hybrid'] = (np.asarray(Xtr), np.asarray(Xte), time.time() - t)
            print(f'  fold {fi} hybrid: {data["hybrid"][0].shape[1]} features, '
                  f'{data["hybrid"][2]:.0f}s', flush=True)
        for i, m in need:
            a, b, build = data[i]
            t = time.time()
            err = None
            try:
                score, _ = _outer_fit_score(a, y[tr], b, y[te], m, 'regression', rmse, seed=42)
            except Exception as e:
                score, err = float('nan'), f'{type(e).__name__}: {e}'[:300]
            row = dict(dataset=ds, input=i, model=m, fold=fi, rmse=score, fit_s=time.time() - t,
                       build_s=build, n_features=int(a.shape[1]), n_molecules=len(y), error=err)
            with open(path, 'a') as f:
                f.write(json.dumps(row) + '\n')
            print(f'  fold {fi} {i:7s} {m:9s} rmse {score:.4f} {row["fit_s"]:.0f}s {err or ""}',
                  flush=True)
    print(f'FINISHED {ds}', flush=True)


def table(out):
    rows = collections.defaultdict(list)
    for p in sorted(Path(out).glob('*.jsonl')):
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            rows[(r['dataset'], r['input'], r['model'])].append(r)
    print('One row is one model on one input. RMSE is the mean over the folds shown, lower is')
    print('better. Fit is mean seconds per fold; add build, seconds per fold, for the hybrid.\n')
    print('| dataset | input | model | folds | RMSE | lowest fold | highest fold | fit s | build s | features |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for d in sorted({k[0] for k in rows}):
        for i in INPUTS:
            for m in MODELS:
                x = rows.get((d, i, m))
                if not x:
                    continue
                ok = [a for a in x if a['rmse'] == a['rmse']]
                s = sorted(a['rmse'] for a in ok)
                mean = lambda k: sum(a[k] for a in x) / len(x)
                if s:
                    print(f'| {d} | {i} | {m} | {len(ok)} | {sum(s) / len(s):.4f} | {s[0]:.4f} | {s[-1]:.4f} '
                          f'| {mean("fit_s"):.0f} | {mean("build_s"):.0f} | {x[0]["n_features"]} |')
                else:
                    print(f'| {d} | {i} | {m} | 0 | failed: {x[0]["error"]} | | | | | |')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('datasets', nargs='*')
    ap.add_argument('--out')
    ap.add_argument('--table', action='store_true')
    ap.add_argument('--build-one', nargs=2, metavar=('DATASET', 'REP'))
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--inputs', default=','.join(INPUTS))
    ap.add_argument('--folds', type=int, default=5)
    a = ap.parse_args()
    if a.build_one:
        from experiments import _shared as S
        d = S.load_data(a.build_one[0], rep_names=[a.build_one[1]])
        sys.exit(0 if a.build_one[1] in d['reps'] else 1)
    os.makedirs(a.out, exist_ok=True)
    if a.table:
        return table(a.out)
    for ds in a.datasets:
        run(ds, a.out, a.models.split(','), a.inputs.split(','), a.folds)


if __name__ == '__main__':
    main()
