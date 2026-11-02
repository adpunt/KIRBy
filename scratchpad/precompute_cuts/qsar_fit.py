"""Simple local fits to pick chembl_qsar's two models (Adelaide, 2026-10-05).
Each target: chembl_qsar's 10 pruned representations, five candidate models at
default settings, 5 scaffold folds (nested_cv_outer_folds, seed 42), mae."""
import sys, json, time, traceback
sys.path[:0] = ['tests', 'src']
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from kirby.datasets.chembl_qsar import load_chembl_qsar
from kirby.find_best_hybrid import nested_cv_outer_folds
from find_best_hybrid_molecular import get_representations, assign_scaffold_groups
from experiments.complexity.complexity_helpers import DATASET_REPS
from experiments._shared import get_model_factories

target = sys.argv[1]
OUT = f'/private/tmp/claude-501/-Users-apunt-repos-KIRBy/156cd673-22b0-4658-a80b-fa0f497d03ff/scratchpad/qsar_fits/{target}.json'
data = load_chembl_qsar(target)
smiles, y = data['smiles'], np.asarray(data['labels'], dtype=float)
groups, ng = assign_scaffold_groups(smiles)
print(target, len(y), 'molecules', ng, 'scaffolds', flush=True)
res = {'target': target, 'n': len(y), 'scores': {}, 'failed_reps': {}}
for rep in DATASET_REPS['chembl_qsar']['general']:
    t0 = time.time()
    try:
        X = get_representations(smiles, rep)[rep].astype(np.float32)
    except Exception as e:
        res['failed_reps'][rep] = f'{type(e).__name__}: {e}'[:200]
        print('  rep failed', rep, res['failed_reps'][rep], flush=True); continue
    X = np.nan_to_num(X)
    folds = nested_cv_outer_folds({rep: X}, y, groups, 5, 42, 'regression')
    facs = get_model_factories(['rf', 'svm', 'lightgbm', 'xgboost', 'dnn'], 'regression', n_samples=len(y))
    for m, fac in facs.items():
        maes = []
        try:
            for tr, te in folds:
                sc = StandardScaler().fit(X[tr])
                mdl = fac(); mdl.fit(sc.transform(X[tr]), y[tr])
                maes.append(float(mean_absolute_error(y[te], mdl.predict(sc.transform(X[te])))))
            res['scores'].setdefault(rep, {})[m] = maes
            print(f'  {rep:14s} {m:9s} mae {np.mean(maes):.4f}', flush=True)
        except Exception as e:
            res['scores'].setdefault(rep, {})[m] = f'{type(e).__name__}: {e}'[:200]
            print('  fit failed', rep, m, str(e)[:120], flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f'  {rep} done in {time.time()-t0:.0f}s', flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
