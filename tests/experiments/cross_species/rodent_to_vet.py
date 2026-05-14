#!/usr/bin/env python3
"""rodent_to_vet (T-PT.4) — is "evolutionary closeness" real?

Cross-species pretraining variant `rodent_to_vet` from
`docs/cross_species_pretraining_design.md` §2. Identical machinery to T-PT.3
(`same_target_transfer`) — same vet eval targets, same conditions, same
`PretrainedDNNRegressor` — but the pretraining corpus is **rat + mouse**
ChEMBL (`pretrain_corpora` corpus `rodent_chembl`) instead of human. Question:
are rodents a better transfer source than humans for dog/cow endpoints?

Read against T-PT.3's JSON: per (rep × target), RMSE delta of rodent_to_vet
minus same_target_transfer. Positive lift for rodent = "evolutionary
closeness" supported (design doc §4 decision rule: ≥0.03 RMSE lift on ≥3 of
4 vet eval combos).

This script reuses `same_target_transfer.evaluate_target` /
`.print_analysis` verbatim — the ONLY difference is `--pretrain-dir` points
at the rodent_chembl checkpoint subdirectory.

Conditions per (vet target × rep): dnn_scratch, dnn_tl_full, dnn_tl_frozen,
zero_shot, rf, + a kirby_hybrid rep. Reps = C4 set. Splits = 5-fold scaffold
GroupKFold, 3-fold fallback at n < 150.

Usage:
    # Smoke test (local, fast)
    python tests/experiments/cross_species/rodent_to_vet.py \\
        --keys cow_DRD2 --reps ecfp4 --n-folds 2

    # Full run (ARC) — needs rodent_chembl checkpoints to exist:
    #   sbatch tests/slurm_scripts/pretrain_rodent.sh   (produces them)
    python tests/experiments/cross_species/rodent_to_vet.py

Reference:
    Imrie et al. (2018). JCIM 58(11):2319-2330.
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import argparse
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

warnings.filterwarnings('ignore')

from experiments._shared import convert_numpy
from experiments.cross_species.vet_eval_data import (
    list_vet_keys, DEFAULT_CACHE_DIR as VET_DEFAULT_CACHE_DIR,
)
from experiments.cross_species.same_target_transfer import (
    evaluate_target, print_analysis, DEFAULT_REPS, SMALL_N_THRESHOLD,
)

# Rodent checkpoints live in the rodent_chembl subdir produced by
# pretrain_chembl_dnn.py --corpus rodent_chembl.
DEFAULT_PRETRAIN_DIR = '/Volumes/seagate/kirby/chembl_pretrain/rodent_chembl'
DEFAULT_N_FOLDS = 5


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--keys', type=str, default=None,
                    help='Comma-separated vet target keys '
                         '(default: all in cross_species_vet_eval_targets.json)')
    ap.add_argument('--reps', type=str, default=None,
                    help=f'Comma-separated rep names (default: {",".join(DEFAULT_REPS)})')
    ap.add_argument('--n-folds', type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument('--pretrain-dir', type=str, default=DEFAULT_PRETRAIN_DIR,
                    help=f'Directory with rodent_chembl pretrained checkpoints '
                         f'(default: {DEFAULT_PRETRAIN_DIR})')
    ap.add_argument('--vet-cache-dir', type=str, default=VET_DEFAULT_CACHE_DIR,
                    help=f'ChEMBL activity CSV cache dir for vet targets '
                         f'(default: {VET_DEFAULT_CACHE_DIR}). Server-local on ARC.')
    ap.add_argument('--output', type=str, default=None,
                    help='Output JSON path (default: '
                         'tests/results/cross_species/rodent_to_vet.json)')
    args = ap.parse_args()

    keys = args.keys.split(',') if args.keys else list_vet_keys()
    reps = args.reps.split(',') if args.reps else DEFAULT_REPS

    print("=" * 70)
    print("rodent_to_vet  (T-PT.4)")
    print(f"  Vet targets:  {keys}")
    print(f"  Reps:         {reps}")
    print(f"  Folds:        {args.n_folds} (3-fold fallback at n<{SMALL_N_THRESHOLD})")
    print(f"  Pretrain dir: {args.pretrain_dir}  (corpus: rodent_chembl)")
    print(f"  Vet cache:    {args.vet_cache_dir}")
    print("=" * 70)

    if not os.path.isdir(args.pretrain_dir):
        print(f"\nWARNING: pretrain dir does not exist: {args.pretrain_dir}")
        print(f"  Produce rodent_chembl checkpoints first:")
        print(f"  python -m kirby.models.pretrain_chembl_dnn "
              f"--rep {','.join(reps)} --corpus rodent_chembl")
        print(f"  Continuing — TL + zero_shot will be skipped, "
              f"scratch/rf still run.\n")

    results = {}
    for key in keys:
        try:
            results[key] = evaluate_target(
                key, reps, args.n_folds, args.pretrain_dir,
                vet_cache_dir=args.vet_cache_dir)
        except Exception as e:
            print(f"  FAILED on {key}: {e}")
            import traceback
            traceback.print_exc()
            results[key] = {'error': str(e)[:200]}

    print_analysis(results, title="RODENT->VET TRANSFER ANALYSIS")

    out = Path(args.output or
               'tests/results/cross_species/rodent_to_vet.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(convert_numpy(results), indent=2))
    print(f"\nResults written: {out}")
    print("Compare against tests/results/cross_species/same_target_transfer.json "
          "for the evolutionary-closeness verdict.")


if __name__ == '__main__':
    main()
