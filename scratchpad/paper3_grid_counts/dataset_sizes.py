"""Molecule (or sequence) count, task type and SMILES presence of every Paper 3
dataset, from the experiments' own loader (load_data) with one cheap
representation. Written to dataset_sizes.json beside this file.
Run: PYTHONPATH=src:tests:tests/experiments python scratchpad/paper3_grid_counts/dataset_sizes.py"""
import json, os, sys, time, traceback
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from experiments.papers import plan as P
from experiments._shared import load_data
import experiments.dataset_policy  # noqa
from experiments.complexity.complexity_helpers import DATASET_REPS
from experiments.noise.n36_remeasure_priority import build_replicate_groups


def main():
    out_p = os.path.join(os.path.dirname(__file__), 'dataset_sizes.json')
    out = json.load(open(out_p)) if os.path.exists(out_p) else {}
    for ds in P.PAPER3_DATASETS:
        if ds in out and 'error' not in out[ds] and 'has_replicates' in out[ds]:
            continue
        t0 = time.time()
        from experiments.noise.noise_helpers import REP_LIST_FROM
        pool = (DATASET_REPS.get(REP_LIST_FROM.get(ds, ds)) or {}).get('general') or ['ecfp4']
        rep = pool[:1] or None
        try:
            try:
                d = load_data(ds, rep_names=['ecfp4'])
                rep = ['ecfp4']
            except Exception:
                d = load_data(ds, rep_names=rep)
            smi = d.get('smiles')
            n_rows = len(d['labels'])
            n_mol = len(set(map(str, smi))) if smi is not None and len(smi) else n_rows
            out[ds] = {'rows': n_rows, 'molecules': n_mol, 'task_type': d.get('task_type'),
                       'domain': d.get('domain'), 'has_smiles': smi is not None and len(smi) > 0,
                       'has_groups': d.get('groups') is not None, 'rep_loaded': rep,
                       'has_replicates': build_replicate_groups(d) is not None,
                       'seconds': round(time.time() - t0, 1)}
        except Exception as exc:
            out[ds] = {'error': f'{type(exc).__name__}: {exc}'[:300]}
        print(ds, out[ds], flush=True)
        json.dump(out, open(out_p, 'w'), indent=1)


if __name__ == '__main__':
    main()
