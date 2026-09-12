#!/usr/bin/env python3
"""Verify which (category, dataset) jobs from the 2026-05-08 wave actually
produced complete, error-free output, vs. silently failed inside run_all.

Background:
  - run_all in {complexity,similarity,noise,federated}_helpers.py wraps each
    experiment in try/except. On exception, it appends to in-memory results
    but does NOT call save_experiment_results, so NO FILE is written.
  - The job still exits 0. sacct shows COMPLETED. The failure is invisible
    unless you check files on disk.
  - Some experiments self-report errors by returning {'error': '...'} (these
    DO get written to disk). Those count as "errored" not "missing".

Run this on the server, where tests/results/ has the real data:
    cd /data/stat-ecr/scat9264/KIRBy
    python tests/verify_run_status.py

Output:
    - Per-category summary table (datasets x experiments)
    - List of (category, dataset) pairs to fully resubmit
    - List of (category, dataset, [missing exps]) for partial resubmits
    - Writes audit JSON to tests/results/_run_audit_2026_05_11.json
"""
import argparse
import datetime
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent

# Default cutoff: files older than this are treated as STALE (from a prior
# wave), not as evidence the current run succeeded. The 2026-05-08 wave
# (sbatch ID range 12103xxx) started submitting at ~2026-05-08 11:00 UTC.
DEFAULT_SINCE = '2026-05-08 00:00'

# ── Expected experiments per category (from *_helpers.py SINGLE_DATASET_EXPERIMENTS) ──

EXPERIMENTS = {
    'complexity': {
        'c01': 'c01_rep_characterization',
        'c02': 'c02_complexity_matching_grid',
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
        's08': 's08_optimal_dimension',
        's10': 's10_importance_downstream',
        's11': 's11_bayesian_optimization',
        's12': 's12_cross_domain_cliffs',
        's13': 's13_clustering_retrieval',
    },
    'noise': {
        'n01': 'n01_activity_cliff_framework',
        'n02': 'n02_sample_weighting',
        'n07': 'n07_remediation',
        'n09': 'n09_lnl_survey',
        'n11': 'n11_estimation_remediation_grid',
        'n12': 'n12_importance_weighted_distance',
        'n13': 'n13_noise_distributions',
    },
    'federated': {
        'fed01': 'fed01_noise_heterogeneity',
        'fed02': 'fed02_rep_diversity_silos',
        'fed03': 'fed03_synthetic_private',
        'fed04': 'fed04_cluster_aware_importance',
        'fed05': 'fed05_antibody_federated',
    },
}

# ── Job state from `sacct -u $USER -S 2026-05-08` (manually transcribed) ──
# Only complexity/similarity rows; opig and exp_%j are handled separately.

JOB_STATE = {
    'complexity': {
        'COMPLETED': [
            'herg_ki', 'flab_polyreact_jain_cic', 'flab_immunogenicity_marks',
            'flab_polyreact_jain_bvp', 'ddg_combined', 'flab_binding_koenig',
            'flab_expression_koenig', 'flab_binding_adams', 'flab_expression_adams',
            'flab_binding_phillips_h3', 'abagym_HER2', 'abagym_lysozyme',
            'abagym_VEGF',
        ],
        'RUNNING': [
            'caco2', 'logd', 'qm9', 'herg', 'kiba', 'bindingdb_kd',
            'adme_hlm', 'adme_rlm', 'adme_mdr1',
            'flab_thermo_tresanco', 'flab_pk_jain_heprt', 'flab_pk_jain_fcrnrt',
            'flab_aggregation_jain_hicrt', 'flab_thermo_jain', 'flab_aggregation_jain',
            'abcov_kd', 'abagym_Ang2', 'abagym_EGFR',
            'ginkgo_acsins', 'ginkgo_colloidal', 'ginkgo_heparin', 'ginkgo_hic',
            'ginkgo_pr_cho', 'ginkgo_titer',
            'buchwald_hartwig', 'bh_doyle_plates',
        ],
        'CANCELLED': [
            'tz_her2_binary', 'tz_her2_multiclass', 'humanness_therapeutic',
            'tcga_luad_vs_lusc', 'tcga_brca_pam50',
            'nt_H2AFZ', 'nt_H3K27ac', 'nt_H3K27me3', 'nt_H3K36me3',
            'nt_H3K4me1', 'nt_H3K4me2', 'nt_H3K4me3', 'nt_H3K9ac',
            'nt_H3K9me3', 'nt_H4K20me1', 'nt_enhancers', 'nt_enhancers_types',
            'nt_promoter_all', 'nt_promoter_no_tata', 'nt_promoter_tata',
            'nt_splice_sites_acceptors', 'nt_splice_sites_donors',
            'gb_demo_coding_vs_intergenomic_seqs', 'gb_demo_human_or_worm',
            'gb_human_enhancers_cohn', 'gb_human_nontata_promoters',
        ],
    },
    'similarity': {
        'COMPLETED': [
            'herg_ki', 'caco2', 'logd', 'qm9', 'adme_hlm', 'adme_rlm', 'adme_mdr1',
            'herg', 'kiba', 'bindingdb_kd', 'flab_thermo_tresanco',
            'flab_polyreact_jain_cic', 'flab_aggregation_jain_hicrt',
            'flab_thermo_jain', 'abcov_kd', 'flab_immunogenicity_marks',
            'flab_polyreact_jain_bvp', 'flab_binding_koenig', 'flab_binding_adams',
            'flab_expression_adams', 'abagym_Ang2', 'abagym_EGFR', 'abagym_HER2',
            'abagym_VEGF', 'ginkgo_acsins', 'ginkgo_heparin', 'ginkgo_hic',
            'ginkgo_titer', 'buchwald_hartwig', 'tcga_luad_vs_lusc',
            'tcga_brca_pam50',
            'nt_H2AFZ', 'nt_H3K27ac', 'nt_H3K27me3', 'nt_H3K36me3',
            'nt_H3K4me1', 'nt_H3K4me2', 'nt_H3K4me3', 'nt_H3K9ac',
            'nt_H3K9me3', 'nt_H4K20me1', 'nt_enhancers', 'nt_enhancers_types',
            'nt_promoter_all', 'nt_promoter_no_tata',
            'nt_splice_sites_acceptors', 'nt_splice_sites_donors',
            'gb_demo_coding_vs_intergenomic_seqs', 'gb_demo_human_or_worm',
            'gb_human_enhancers_cohn', 'gb_human_nontata_promoters',
        ],
        'RUNNING': [
            'flab_pk_jain_heprt', 'flab_pk_jain_fcrnrt', 'flab_aggregation_jain',
            'flab_expression_koenig', 'flab_binding_phillips_h3', 'ddg_combined',
            'abagym_lysozyme', 'ginkgo_colloidal', 'ginkgo_pr_cho',
            'bh_doyle_plates', 'nt_promoter_tata',
        ],
        'CANCELLED': [
            'tz_her2_binary', 'tz_her2_multiclass', 'humanness_therapeutic',
        ],
    },
}


def audit_job(category: str, dataset: str, expected_exps: dict,
              results_root: Path, since_ts: float):
    """Return per-job audit: which expected JSONs are present, errored, missing, stale."""
    cat_dir = results_root / category
    record = {
        'category': category,
        'dataset': dataset,
        'expected': len(expected_exps),
        'ok': [],
        'errored': [],          # JSON fresh + has {'error': ...}
        'missing': [],          # JSON expected but not on disk → silent crash
        'stale': [],             # JSON exists but mtime < since_ts (old wave leftover)
        'unreadable': [],       # JSON exists but parse failed
        'error_messages': {},   # exp_id -> error string
    }
    for short_id, dir_name in expected_exps.items():
        # c12 (modality ablation) is TCGA-only — only expected for tcga_*.
        if category == 'complexity' and short_id == 'c12' and not dataset.startswith('tcga_'):
            continue
        path = cat_dir / dir_name / f'{dataset}.json'
        if not path.exists():
            record['missing'].append(short_id)
            continue
        mtime = path.stat().st_mtime
        is_fresh = mtime >= since_ts
        try:
            d = json.loads(path.read_text())
        except Exception as e:
            record['unreadable'].append(short_id)
            record['error_messages'][short_id] = f'JSON parse: {e}'
            continue
        if not is_fresh:
            record['stale'].append(short_id)
            continue
        if isinstance(d, dict) and 'error' in d:
            record['errored'].append(short_id)
            record['error_messages'][short_id] = str(d['error'])[:200]
        else:
            record['ok'].append(short_id)
    record['ok_count'] = len(record['ok'])
    record['bad_count'] = (len(record['errored']) + len(record['missing'])
                           + len(record['unreadable']) + len(record['stale']))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--since', default=DEFAULT_SINCE,
                        help=f"Cutoff for 'fresh' files (default: {DEFAULT_SINCE}). "
                             "Files with mtime older than this are reported as STALE, "
                             "not as evidence of current-wave success.")
    parser.add_argument('--results-root', default=str(ROOT / 'tests' / 'results'),
                        help='Path to tests/results (default: repo-relative)')
    args = parser.parse_args()

    since_dt = datetime.datetime.strptime(args.since, '%Y-%m-%d %H:%M')
    since_ts = since_dt.timestamp()
    print(f"Fresh-file cutoff: {args.since} (epoch {since_ts:.0f})")

    results_root = Path(args.results_root)
    audit_rows = []

    for category in ('complexity', 'similarity'):
        states = JOB_STATE[category]
        expected_exps = EXPERIMENTS[category]
        for state in ('COMPLETED', 'RUNNING', 'CANCELLED'):
            for ds in states.get(state, []):
                rec = audit_job(category, ds, expected_exps, results_root, since_ts)
                rec['job_state'] = state
                audit_rows.append(rec)

    # ── Summary ──
    print(f"\n{'=' * 110}")
    print(f"{'CATEGORY':12} {'DATASET':35} {'STATE':10} {'OK':>4} {'ERR':>4} {'MISS':>5} {'STALE':>5} {'TOT':>4}  status")
    print('=' * 110)
    fully_bad = []           # nothing fresh on disk → full resubmit
    partial_bad = []         # some fresh files → partial resubmit
    fully_good = []
    for r in audit_rows:
        total = r['expected']
        ok = len(r['ok'])
        err = len(r['errored'])
        miss = len(r['missing'])
        stale = len(r['stale'])
        # FULL_OK requires ok == total (stale doesn't count as success).
        # EMPTY = no fresh output at all (everything either missing or stale).
        # PARTIAL = some fresh ok but not all.
        if ok == total:
            status = 'FULL_OK'
        elif ok == 0:
            status = 'EMPTY_STALE' if stale > 0 else 'EMPTY'
        else:
            status = 'PARTIAL'
        print(f"{r['category']:12} {r['dataset']:35} {r['job_state']:10} "
              f"{ok:>4} {err:>4} {miss:>5} {stale:>5} {total:>4}  {status}")
        if status in ('EMPTY', 'EMPTY_STALE'):
            fully_bad.append((r['category'], r['dataset'], r['job_state']))
        elif status == 'PARTIAL':
            # Missing + errored + stale all need re-run for THIS wave
            partial_bad.append((r['category'], r['dataset'], r['job_state'],
                                r['missing'] + r['errored'] + r['stale']))
        else:
            fully_good.append((r['category'], r['dataset']))

    # ── Error type breakdown ──
    err_counter = defaultdict(list)
    for r in audit_rows:
        for exp, msg in r['error_messages'].items():
            key = msg[:60]
            err_counter[key].append(f"{r['category']}/{exp}/{r['dataset']}")

    print(f"\n{'=' * 100}\nError messages observed:\n{'=' * 100}")
    for msg, where in sorted(err_counter.items(), key=lambda x: -len(x[1])):
        print(f"\n[{len(where):3}x] {msg}")
        for w in where[:5]:
            print(f"    - {w}")
        if len(where) > 5:
            print(f"    ... ({len(where) - 5} more)")

    # ── Resubmit list ──
    print(f"\n{'=' * 100}\nRESUBMIT — fully bad (no output at all):\n{'=' * 100}")
    for cat, ds, state in fully_bad:
        print(f"  sbatch --job-name={cat[:3]}_{ds} tests/slurm_scripts/submit_dataset.sh {cat} {ds}    # was {state}")

    print(f"\n{'=' * 100}\nRESUBMIT — partial (re-run specific experiments):\n{'=' * 100}")
    for cat, ds, state, exps in partial_bad:
        exp_arg = ' '.join(sorted(set(exps)))
        print(f"  sbatch --job-name={cat[:3]}_{ds}_partial tests/slurm_scripts/submit_dataset.sh "
              f"{cat} {ds} \"{exp_arg}\"    # was {state}; missing: {exp_arg}")

    print(f"\n{'=' * 100}\nFULLY GOOD ({len(fully_good)}):\n{'=' * 100}")
    for cat, ds in fully_good:
        print(f"  {cat}/{ds}")

    # ── Save audit JSON ──
    out_path = results_root / '_run_audit_2026_05_11.json'
    out_path.write_text(json.dumps({
        'rows': audit_rows,
        'fully_bad': [list(t) for t in fully_bad],
        'partial_bad': [(c, d, s, m) for c, d, s, m in partial_bad],
        'fully_good': [list(t) for t in fully_good],
        'error_messages': {k: v for k, v in err_counter.items()},
    }, indent=2, default=str))
    print(f"\nAudit JSON written to: {out_path}")


if __name__ == '__main__':
    main()
