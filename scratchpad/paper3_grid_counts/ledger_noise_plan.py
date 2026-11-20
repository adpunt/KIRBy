"""Print the Papers ledger's paper3 noise units and the folds and levels each
n43-n47 unit would run, for named datasets, from the code as it stands.
Run: PYTHONPATH=src:tests:tests/experiments python scratchpad/paper3_grid_counts/ledger_noise_plan.py caco2 logd qm9"""
import os, sys
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from experiments.papers import plan as P, worker as W
from experiments.orchestrate import plan as OP
from experiments.noise import noise_helpers as nh
from kirby.registry import lookup

def env_for(ds, unit):
    row = {'kind': 'dataset', 'dataset': ds}
    import experiments.papers.worker as w
    orig = w.read_settings
    w.read_settings = lambda r: {'experiments_search': 'pruned', 'cache_dir': '/x', 'searches': {'pruned': {'space': 'full'}}}
    try:
        _, env, _ = w.command_for(ds, row, 'paper3', unit, '/nonexistent_run')
    finally:
        w.read_settings = orig
    return env

COUNTS = {'caco2': 2158, 'logd': 5028, 'qm9': 4995}   # molecules, find_best_hybrid_molecular.load_dataset, 2026-11-16
for ds in sys.argv[1:]:
    info = lookup(ds)
    n = COUNTS.get(ds) or getattr(info, 'n_samples', None)
    print(f'\n### {ds}: registry n_samples={n}, task={getattr(info, "task_type", None)}')
    units = [k for k, c in P.paper_units(3, ds) if c == 'noise']
    noise5 = [u for u in units if u.split('__')[0] in ('n43', 'n44', 'n45', 'n46', 'n47')]
    print('paper3 noise units:', ' '.join(units))
    for u in noise5:
        env = env_for(ds, u)
        old = dict(os.environ)
        os.environ.clear(); os.environ.update(env)
        try:
            exp = u.split('__')[0]
            cfg = nh.one_fold_screen_config({'n_splits': 5}, exp, dataset_name=ds)
            folds = cfg.get('only_folds') or tuple(range(5))
            cond = u.split('__')[1] if '__' in u else None
            data = {'task_type': getattr(info, 'task_type', 'regression'), 'labels': [0] * int(n or 0)}
            from experiments.noise import n40_model_side_regularisation as n40
            lv = (None if not cond else n40._condition_levels(data, ds, cond) if cond == 'hetero_seen' else nh.decided_levels(data, ds, cond))
            print(f'  {u:32s} promote={env.get("KIRBY_NOISE_PROMOTE")!s:6s} folds={list(folds)} levels={lv}')
        finally:
            os.environ.clear(); os.environ.update(old)
