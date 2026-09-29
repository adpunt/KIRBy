"""Dirty local noise runner (2026-09-28). One experiment, one dataset per call.
Reuses tests/experiments/noise/_local_run.py (its sweep_cuts: 4 noise levels, 1 draw,
per-experiment cuts) but with the default-hybrid grid: evaluator rf, rep_count 5,
budget 250, models rf/svm/lightgbm, 3 search folds, 3 experiment folds.
usage: dirty_noise.py <dataset> <nXX> <outdir> [--condition C]"""
import sys, time
sys.argv, _args = sys.argv[:1], sys.argv[1:]
ds, exp, out = _args[0], _args[1], _args[2]
cond = _args[_args.index('--condition') + 1] if '--condition' in _args else None
from experiments.noise import _local_run as LR
LR.MODELS = ['rf', 'svm', 'lightgbm']
LR.BUDGETS = [250]
print('SHRINK:', LR.shrink(LR.MODELS, LR.BUDGETS, 3))
import kirby.find_best_hybrid as F
F.DEFAULT_INTERNAL_EVALUATORS = ['rf']; F.FAST_INTERNAL_EVALUATORS = ['rf']
F.DEFAULT_REP_COUNTS = [5]; F.FAST_REP_COUNTS = [5]
import experiments._shared as S
S.INTERNAL_EVALUATORS = ['rf']
print('SWEEP CUTS:', LR.sweep_cuts({exp}))
cfg = LR.config(LR.MODELS, LR.BUDGETS, 3)
cfg['internal_evaluators'] = ['rf']
from experiments.noise import noise_helpers as NH
kw = {'noise_condition': cond} if cond else {}
t0 = time.time()
NH.run_all(dataset_name=ds, output_dir=out, experiments=[exp], config=cfg, force_rerun=True, **kw)
print(f"\nTOTAL {time.time() - t0:.0f}s")
