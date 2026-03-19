#!/usr/bin/env python3
"""
Phase 5 Readiness Check

Quick sanity checks before submitting Phase 5 SLURM jobs. Verifies:
  1. All imports work (no missing packages)
  2. Dataset registries are consistent
  3. Model factories resolve correctly
  4. Rep presets match what the SLURM scripts expect
  5. Key dependencies are available (rdkit, torch, transformers, etc.)

Usage:
    python tests/verify_phase5_readiness.py              # all checks
    python tests/verify_phase5_readiness.py --quick       # imports only
    python tests/verify_phase5_readiness.py --domain dta  # one domain
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings('ignore')

PASS = 0
FAIL = 0
WARN = 0


def check(name, condition, msg=""):
    global PASS, FAIL
    if condition:
        print(f"  ✓ {name}")
        PASS += 1
    else:
        print(f"  ✗ {name}{f' — {msg}' if msg else ''}")
        FAIL += 1


def warn(name, msg):
    global WARN
    print(f"  ⚠ {name} — {msg}")
    WARN += 1


def check_imports():
    """Check that all key packages import."""
    print("\n--- Core Imports ---")

    packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scipy', 'scipy'),
        ('sklearn', 'scikit-learn'),
        ('xgboost', 'xgboost'),
        ('lightgbm', 'lightgbm'),
    ]
    for mod, name in packages:
        try:
            __import__(mod)
            check(name, True)
        except ImportError:
            check(name, False, "not installed")

    print("\n--- Optional Imports ---")
    optional = [
        ('torch', 'PyTorch'),
        ('transformers', 'HuggingFace transformers'),
        ('rdkit', 'RDKit'),
        ('catboost', 'CatBoost'),
        ('mordred', 'Mordred descriptors'),
        ('drfp', 'DRFP (reaction FPs)'),
        ('rxnfp', 'RXNFP (reaction FPs)'),
        ('polaris', 'Polaris Hub (ADME)'),
        ('esm', 'ESM (ESM-C/ESM3)'),
        ('antiberty', 'AntiBERTy'),
        ('ablang2', 'AbLang2'),
        ('minimol', 'MiniMol'),
        ('boltz', 'Boltz-2'),
    ]
    for mod, name in optional:
        try:
            __import__(mod)
            check(name, True)
        except ImportError:
            warn(name, "not installed (may cause some reps to skip)")


def check_kirby_core():
    """Check KIRBy core imports."""
    print("\n--- KIRBy Core ---")

    try:
        from kirby.find_best_hybrid import find_best_hybrid, _resolve_models, DEFAULT_MODELS
        check("find_best_hybrid import", True)
        check(f"DEFAULT_MODELS ({len(DEFAULT_MODELS)} models)", len(DEFAULT_MODELS) >= 10)
    except Exception as e:
        check("find_best_hybrid import", False, str(e))

    try:
        from kirby.find_best_hybrid import _resolve_models
        reg_models = _resolve_models(None, 'regression', 'long')
        clf_models = _resolve_models(None, 'classification', 'long')
        check(f"Regression models: {len(reg_models)} ({', '.join(reg_models.keys())})",
              len(reg_models) >= 9)
        check(f"Classification models: {len(clf_models)} ({', '.join(clf_models.keys())})",
              len(clf_models) >= 9)
    except Exception as e:
        check("Model resolution", False, str(e))


def check_molecular():
    """Check molecular domain readiness."""
    print("\n--- Molecular Domain ---")

    try:
        from find_best_hybrid_molecular import REP_PRESETS
        n_all = len(REP_PRESETS.get('all', []))
        check(f"REP_PRESETS['all']: {n_all} reps", n_all >= 25)
        # Check Phase 4 reps are present
        for rep in ['minimol', 'gp_molformer', 'mole']:
            present = rep in REP_PRESETS.get('all', [])
            check(f"  Phase 4: {rep}", present)
    except Exception as e:
        check("find_best_hybrid_molecular REP_PRESETS", False, str(e))

    try:
        from run_baseline_survey import DATASET_REGISTRY, DOMAIN_DATASETS
        mol_ds = DOMAIN_DATASETS.get('molecular', [])
        check(f"Molecular datasets: {len(mol_ds)} ({', '.join(mol_ds)})", len(mol_ds) >= 4)
    except Exception as e:
        check("DATASET_REGISTRY (molecular)", False, str(e))


def check_tcga():
    """Check TCGA domain readiness."""
    print("\n--- TCGA Domain ---")

    try:
        from find_best_hybrid_tcga import REP_PRESETS
        n_all = len(REP_PRESETS.get('all', []))
        check(f"REP_PRESETS['all']: {n_all} reps", n_all >= 50)
    except Exception as e:
        check("find_best_hybrid_tcga REP_PRESETS", False, str(e))


def check_dta():
    """Check DTA domain readiness."""
    print("\n--- DTA Domain ---")

    try:
        from find_best_hybrid_dta import REP_PRESETS
        n_full = len(REP_PRESETS.get('full', []))
        check(f"REP_PRESETS['full']: {n_full} reps", n_full >= 30)
        # Check Phase 4 reps
        full_reps = REP_PRESETS.get('full', [])
        for rep in ['protein_mint', 'protein_esm3', 'protein_saprot',
                     'drug_minimol', 'drug_gp_molformer', 'drug_mole']:
            check(f"  Phase 4: {rep}", rep in full_reps)
    except Exception as e:
        check("find_best_hybrid_dta REP_PRESETS", False, str(e))


def check_antibody():
    """Check antibody domain readiness."""
    print("\n--- Antibody Domain ---")

    try:
        from kirby.representations.antibody import ANTIBODY_REPS
        n_reps = len(ANTIBODY_REPS)
        check(f"ANTIBODY_REPS: {n_reps} reps", n_reps >= 30)
        # Check Phase 4 reps
        for rep in ['esm3', 'saprot', 'igt5', 'abroberta', 'mint']:
            check(f"  Phase 4: {rep}", rep in ANTIBODY_REPS)
    except Exception as e:
        check("ANTIBODY_REPS", False, str(e))

    try:
        from antibody_baseline_survey import DATASET_REGISTRY, DOMAIN_GROUPS
        n_all = len(DATASET_REGISTRY)
        n_dms = len(DOMAIN_GROUPS.get('all_dms', []))
        n_inter = len(DOMAIN_GROUPS.get('all_inter', []))
        check(f"Antibody datasets: {n_all} total ({n_dms} DMS + {n_inter} inter-family)",
              n_all >= 20)
    except Exception as e:
        check("antibody_baseline_survey DATASET_REGISTRY", False, str(e))


def check_reaction():
    """Check reaction domain readiness."""
    print("\n--- Reaction Domain ---")

    try:
        from kirby.representations.reaction import (
            create_drfp, create_concat_ecfp4, create_rxnfp, create_onehot_components)
        check("Reaction rep imports", True)
    except ImportError as e:
        warn("Reaction reps", f"some imports failed: {e}")

    try:
        from kirby.datasets.reaction_yields import load_buchwald_hartwig, load_suzuki
        check("Reaction dataset loaders", True)
    except ImportError as e:
        check("Reaction dataset loaders", False, str(e))


def check_metrics():
    """Check that domain-aware metrics are wired."""
    print("\n--- Metrics ---")

    try:
        from run_baseline_survey import _get_primary_metric
        checks = [
            ('logd', 'regression', 'rmse'),
            ('davis', 'regression', 'ci'),
            ('tcga_luad_vs_lusc', 'classification', 'auroc'),
            ('tcga_brca_pam50', 'classification', 'f1_weighted'),
        ]
        for ds, tt, expected_metric in checks:
            metric, _ = _get_primary_metric(ds, tt)
            check(f"{ds} primary metric = {metric}", metric == expected_metric,
                  f"expected {expected_metric}")
    except Exception as e:
        check("Primary metrics", False, str(e))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 5 readiness check")
    parser.add_argument('--quick', action='store_true', help='Imports only')
    parser.add_argument('--domain', type=str, default=None,
                       help='Check one domain: molecular, tcga, dta, antibody, reaction')
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 5 Readiness Check")
    print("=" * 60)

    check_imports()

    if args.quick:
        pass
    elif args.domain:
        dispatch = {
            'molecular': check_molecular,
            'tcga': check_tcga,
            'dta': check_dta,
            'antibody': check_antibody,
            'reaction': check_reaction,
        }
        if args.domain in dispatch:
            dispatch[args.domain]()
        else:
            print(f"Unknown domain: {args.domain}")
    else:
        check_kirby_core()
        check_molecular()
        check_tcga()
        check_dta()
        check_antibody()
        check_reaction()
        check_metrics()

    print(f"\n{'=' * 60}")
    print(f"Results: {PASS} passed, {FAIL} failed, {WARN} warnings")
    if FAIL == 0:
        print("Phase 5 is ready to go!")
    else:
        print("Fix failures before running Phase 5 SLURM jobs.")
    print(f"{'=' * 60}")
    return 1 if FAIL > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
