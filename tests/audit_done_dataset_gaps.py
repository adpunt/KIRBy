#!/usr/bin/env python
"""Per-dataset experiment-gap audit for precompute-DONE datasets.

For each dataset reported DONE by tests/check_precompute_status.py, list the
complexity / similarity / noise / OPIG experiments that have NOT written a
JSON to tests/results/, and that are NOT currently running or queued in
SLURM. Emits ready-to-run `sbatch` commands.

Usage on server:
    python tests/audit_done_dataset_gaps.py              # full audit + commands
    python tests/audit_done_dataset_gaps.py --no-commands  # gaps only, no sbatch
    python tests/audit_done_dataset_gaps.py --datasets a b c  # subset

Default expected experiment lists come from the *_helpers SINGLE_DATASET_EXPERIMENTS.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / 'tests' / 'results'

sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'tests'))


# ── Expected per-category experiment IDs ─────────────────────────────────────

COMPLEXITY_EXPECTED = [
    'c01', 'c02', 'c04', 'c05', 'c06', 'c07', 'c08', 'c09', 'c10', 'c11',
    'c13', 'c15', 'c16', 'c17', 'c18', 'c19', 'c21', 'c22', 'c23', 'c24',
]
SIMILARITY_EXPECTED = [
    's01', 's02', 's03', 's04', 's05', 's06', 's07', 's08',
    's10', 's11', 's12', 's13',
]
NOISE_EXPECTED = [
    'n01', 'n02', 'n03', 'n04', 'n05', 'n07', 'n08',
    'n09', 'n10', 'n11', 'n12', 'n13',
]
# c12 is TCGA-only; auto-added by the runner for tcga_* datasets.

# Maps c/s/n experiment ID → directory name under tests/results/<category>/
DIR_PREFIX_MAP = {
    'complexity': {
        'c01': 'c01_rep_characterization',
        'c02': 'c02_complexity_matching_grid',  # !! grid → matching_grid in results
        'c04': 'c04_importance_as_metric',
        'c05': 'c05_learning_curves',
        'c06': 'c06_within_family_gradient',
        'c07': 'c07_data_complexity_partitions',
        'c08': 'c08_calibration',
        'c09': 'c09_formal_analysis',
        'c10': 'c10_noise_robustness',
        'c11': 'c11_ensemble_vs_hybrid',
        'c13': 'c13_forward_compatibility',
        'c15': 'c15_ml_theory_depth',
        'c16': 'c16_feature_selection_as_regularization',
        'c17': 'c17_information_theoretic_decomposition',
        'c18': 'c18_diversity_vs_depth',
        'c19': 'c19_importance_weighted_training',
        'c21': 'c21_single_rep_augmentation',
        'c22': 'c22_explainability',
        'c23': 'c23_evaluator_model_matching',
        'c24': 'c24_blueprint_transfer',
    },
    'similarity': {
        's01': 's01_similarity_baseline',
        's02': 's02_hybrid_similarity',
        's03': 's03_feature_concentration',
        's04': 's04_activity_cliffs',
        's05': 's05_knn_prediction',
        's06': 's06_applicability_domain',
        's07': 's07_noise_bridge',
        's08': 's08_optimal_dimension',
        's10': 's10_importance_downstream',
        's11': 's11_bayesian_optimization',
        's12': 's12_cross_domain_cliffs',
        's13': 's13_clustering_retrieval',
    },
    'noise': {
        'n01': 'n01_activity_cliff_framework',
        'n02': 'n02_sample_weighting',
        'n03': 'n03_sas_landscape',
        'n04': 'n04_gp_loo_detection',
        'n05': 'n05_noise_injection_benchmark',
        'n07': 'n07_remediation',
        'n08': 'n08_estimation_survey',
        'n09': 'n09_lnl_survey',
        'n10': 'n10_hybrid_neighborhood_lnl',
        'n11': 'n11_estimation_remediation_grid',
        'n12': 'n12_importance_weighted_distance',
        'n13': 'n13_noise_distributions',
    },
}

# OPIG experiments → dataset-name patterns they apply to. Each is a single
# standalone script (no SINGLE_DATASET_EXPERIMENTS), launched via
# `python tests/experiments/<script>.py` not via run_dataset.py.
OPIG_APPLIES = {
    # script → list of dataset-name predicates
    'opig_ginkgo_developability':     lambda d: d.startswith('ginkgo_'),
    'opig_humanness':                 lambda d: d.startswith('humanness'),
    'opig_trastuzumab_enrichment':    lambda d: d.startswith('tz_her2'),
    'opig_kingsmith_component_splits': lambda d: d == 'buchwald_hartwig',
    'opig_cdr_flexibility':           lambda d: d.startswith('flab_'),
    'opig_activity_cliff_qsar':       lambda d: False,  # uses own ChEMBL data
    'opig_blueprint_transfer':        lambda d: False,  # multi-dataset script
    'opig_cgr_comparison':            lambda d: False,  # standalone reactions
    'opig_graphinity_ddg':            lambda d: False,  # uses own DDG data
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def get_done_datasets() -> List[str]:
    """Invoke check_precompute_status.py and parse out the DONE dataset names."""
    out = subprocess.run(
        [sys.executable, 'tests/check_precompute_status.py', '--done'],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        print('check_precompute_status.py failed:', file=sys.stderr)
        print(out.stdout, out.stderr, file=sys.stderr)
        sys.exit(1)
    done = []
    for line in out.stdout.splitlines():
        # Lines like "  abagym_VEGF                            DONE     2/2      6/6"
        parts = line.split()
        if len(parts) >= 2 and parts[1] == 'DONE':
            ds = parts[0]
            # Skip header underscore lines
            if not all(c == '-' for c in ds):
                done.append(ds)
    return sorted(set(done))


def existing_jsons(category: str, dataset: str) -> Set[str]:
    """Set of experiment IDs that already have a JSON under
    tests/results/<category>/<dir>/<dataset>.json."""
    have: Set[str] = set()
    dir_map = DIR_PREFIX_MAP[category]
    base = RESULTS / category
    for exp_id, dirname in dir_map.items():
        if (base / dirname / f'{dataset}.json').exists():
            have.add(exp_id)
    return have


def queued_jobs() -> List[Tuple[str, str, str]]:
    """List of (job_id, name, state) from squeue."""
    try:
        out = subprocess.run(
            ['squeue', '-u', os.environ.get('USER', ''), '-h', '-o', '%i %j %T'],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return []
    rows = []
    for line in out.stdout.splitlines():
        bits = line.split(None, 2)
        if len(bits) == 3:
            rows.append((bits[0], bits[1], bits[2]))
    return rows


# Prefixes match tests/slurm_scripts/submit_all_datasets.sh.
CATEGORY_PREFIX = {'complexity': 'cx', 'similarity': 'sim',
                   'noise': 'nz', 'federated': 'fed'}


def queued_categories_for_dataset(dataset: str,
                                   jobs: List[Tuple[str, str, str]]
                                   ) -> Set[str]:
    """Return {'complexity', 'similarity', 'noise'} for any active job whose
    8-char SLURM-truncated name matches '<prefix>_<dataset>'[:8].

    Conservative: a cx_abagym* job blocks cx_abagym_VEGF, cx_abagym_HER2 etc.
    because we can't disambiguate from the truncated name. The user can
    `scontrol show job <id> | grep Command=` to disambiguate before launching.
    """
    blocked: Set[str] = set()
    for cat, pfx in CATEGORY_PREFIX.items():
        if cat not in ('complexity', 'similarity', 'noise'):
            continue
        full = f'{pfx}_{dataset}'
        truncated = full[:8]
        for _, name, _ in jobs:
            if name.startswith(truncated):
                blocked.add(cat)
                break
    return blocked


def expected_for_dataset(category: str, dataset: str) -> List[str]:
    if category == 'complexity':
        exp = list(COMPLEXITY_EXPECTED)
        if dataset.startswith('tcga_'):
            exp.append('c12')
        return exp
    if category == 'similarity':
        return list(SIMILARITY_EXPECTED)
    if category == 'noise':
        return list(NOISE_EXPECTED)
    raise ValueError(category)


def applicable_opig(dataset: str) -> List[str]:
    return [name for name, pred in OPIG_APPLIES.items() if pred(dataset)]


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--datasets', nargs='*', default=None,
                    help='Override the DONE list with a manual subset.')
    ap.add_argument('--no-commands', action='store_true',
                    help='Print gap table only, no sbatch commands.')
    ap.add_argument('--threshold', type=int, default=1,
                    help='Only print sbatch for a category if missing >= N '
                         '(default 1 — list anything missing).')
    args = ap.parse_args()

    if args.datasets:
        datasets = sorted(args.datasets)
    else:
        datasets = get_done_datasets()
    print(f'# {len(datasets)} DONE datasets to audit\n')

    jobs = queued_jobs()
    print(f'# {len(jobs)} jobs currently in queue (R or PD)\n')

    sbatch_lines: List[str] = []

    for ds in datasets:
        blocked = queued_categories_for_dataset(ds, jobs)
        print(f'### {ds}')
        if blocked:
            print(f'    blocked categories (active SLURM jobs): {sorted(blocked)}')

        for category in ('complexity', 'similarity', 'noise'):
            have = existing_jsons(category, ds)
            need = [e for e in expected_for_dataset(category, ds) if e not in have]
            status = f'    {category:12s} {len(have):2d}/{len(have)+len(need):2d} done'
            if not need:
                print(f'{status}   ✓')
                continue
            print(f'{status}   missing: {",".join(need)}')

            if category in blocked:
                print('                  → skipping sbatch (job in queue)')
                continue
            if len(need) >= args.threshold:
                # The runner accepts `--experiments e1 e2 e3` as a single
                # space-quoted third positional arg to submit_dataset.sh.
                exp_arg = ' '.join(need)
                jobname = f'{CATEGORY_PREFIX[category]}_{ds}'
                cmd = (f'sbatch --job-name={jobname} '
                       f'tests/slurm_scripts/submit_dataset.sh '
                       f'{category} {ds} "{exp_arg}"')
                sbatch_lines.append(cmd)

        # OPIG: just list applicable scripts; user runs them as standalone.
        opig = applicable_opig(ds)
        if opig:
            print(f'    opig         applicable: {", ".join(opig)}')

        print()

    if not args.no_commands and sbatch_lines:
        print('# ─── Ready-to-run sbatch commands ─────────────────────────')
        print(f'# {len(sbatch_lines)} jobs total\n')
        for cmd in sbatch_lines:
            print(cmd)


if __name__ == '__main__':
    main()
