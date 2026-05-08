"""Post-round sanity check — invoked between auto_chain stages.

Verifies the previous stage produced expected files and they are non-degenerate.

Usage:
    python post_round_sanity.py round4
    python post_round_sanity.py refuse_variants
    python post_round_sanity.py refuse_meta_aware
    python post_round_sanity.py refuse_hierarchical
    python post_round_sanity.py recipes_v2

Exit 0 = ok, 1 = warn (continue but log), 2 = fail (auto_chain should pause and log).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FULL_DIR = REPO / 'tests/results/annitia/full'

EXPECTED = {
    'round4': {
        # Round 4 should leave behind val arrays for the new branches
        # (silent failures here mean fusion silently uses fewer models)
        'patterns': [
            'tests/results/annitia/branches/tabsurv_*_val.npy',
            'tests/results/annitia/branches/catboost_*_val.npy',
            'tests/results/annitia/branches/mlp_*_val.npy',
        ],
        'min_files': 4,
    },
    'refuse_variants': {
        'csvs': [
            'sub_master_rank_mean.csv', 'sub_master_rank_median.csv',
            'sub_master_rank_trimmed.csv', 'sub_master_anomaly_blend.csv',
            'sub_master_specialist_override.csv',
        ],
    },
    'refuse_meta_aware': {
        'csvs': [
            'sub_master_anomaly_router_meta.csv',
            'sub_master_disagreement_router_meta.csv',
            'sub_master_anomaly_blend_meta.csv',
            'sub_master_weighted_anomaly_per_patient.csv',
            'sub_master_cluster_router_meta.csv',
        ],
    },
    'refuse_hierarchical': {
        'csvs': ['sub_master_HIER.csv'],
    },
    'recipes_v2': {
        'csvs': ['sub_master_meta_avg_grand_all_v2.csv',
                 'sub_master_meta_avg_aware_weighted_v2.csv'],
    },
}


def _validate_csv(p: Path) -> tuple[bool, str]:
    """Quick degeneracy check on a submission CSV. Returns (ok, msg)."""
    if not p.exists():
        return False, f"missing: {p.name}"
    try:
        df = pd.read_csv(p)
    except Exception as e:
        return False, f"unreadable {p.name}: {e}"
    if df.isna().sum().sum():
        return False, f"NaN in {p.name}"
    for col in ['risk_hepatic_event', 'risk_death']:
        if col not in df.columns:
            return False, f"{p.name} missing column {col}"
        if np.std(df[col].values) < 1e-10:
            return False, f"{p.name} {col} is constant"
    return True, ''


def _validate_round4(spec):
    issues = []
    files = []
    for pat in spec['patterns']:
        files += list(REPO.glob(pat))
    if len(files) < spec['min_files']:
        issues.append(f"only {len(files)} npy files found, expected ≥{spec['min_files']}")
        issues.append("  (fusion will silently exclude branches that didn't produce val.npy)")
    # check arrays are not all-zero / all-NaN
    bad = []
    for f in files:
        try:
            arr = np.load(f, allow_pickle=False)
            if np.isnan(arr).all() or np.std(arr) < 1e-10:
                bad.append(f.name)
        except Exception as e:
            bad.append(f"{f.name} ({e})")
    if bad:
        issues.append(f"{len(bad)} val.npy degenerate: {bad[:5]}")
    return issues, len(files)


def main(stage: str) -> int:
    if stage not in EXPECTED:
        print(f"unknown stage: {stage}")
        return 2
    spec = EXPECTED[stage]
    issues = []
    print(f"\n=== post_round_sanity: {stage} ===")

    if 'patterns' in spec:
        round_issues, n_files = _validate_round4(spec)
        issues += round_issues
        print(f"  {n_files} val.npy files found")

    if 'csvs' in spec:
        for csv_name in spec['csvs']:
            ok, msg = _validate_csv(FULL_DIR / csv_name)
            mark = 'OK' if ok else 'FAIL'
            print(f"  [{mark}] {csv_name}{(': ' + msg) if msg else ''}")
            if not ok:
                issues.append(msg)

    if issues:
        print("\nIssues found:")
        for i in issues:
            print(f"  - {i}")
        # Treat missing CSVs / degenerate arrays as warn (1) — auto_chain logs and continues
        # because we'd rather have a partial pool than no submission at all.
        return 1
    print("  [PASS] all checks passed")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'round4'))
