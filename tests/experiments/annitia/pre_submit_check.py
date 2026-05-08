"""Pre-submission sanity checks.

Validates a submission CSV before it goes to Trustii. Catches the kind of
silent regressions that previously cost us submissions (constant predictions,
NaN, row drops, anti-correlated to known-good anchor).

Usage:
    python pre_submit_check.py path/to/submission.csv
    python pre_submit_check.py path/to/submission.csv --anchor sub_master_meta_avg_aware_weighted

Exit codes: 0 = pass, 1 = warn (review manually), 2 = fail (do not submit).
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[3]
FULL_DIR = REPO / 'tests/results/annitia/full'
EXPECTED_COLS = ['trustii_id', 'risk_hepatic_event', 'risk_death']
ANCHOR_DEFAULT = 'sub_master_meta_avg_aware_weighted'


def check(csv_path: Path, anchor: str = ANCHOR_DEFAULT) -> int:
    issues = []
    warns = []
    df = pd.read_csv(csv_path)

    # 1. columns
    if list(df.columns) != EXPECTED_COLS:
        issues.append(f"columns are {list(df.columns)}, expected {EXPECTED_COLS}")

    # 2. NaN
    n_nan = df.isna().sum().sum()
    if n_nan:
        issues.append(f"{n_nan} NaN values present")

    # 3. row count vs anchor
    anchor_path = FULL_DIR / f'{anchor}.csv'
    if anchor_path.exists():
        anc = pd.read_csv(anchor_path)
        if len(df) != len(anc):
            issues.append(f"row count {len(df)} != anchor {len(anc)}")
        else:
            # 4. trustii_id alignment
            if not (df['trustii_id'].values == anc['trustii_id'].values).all():
                issues.append("trustii_id ordering does not match anchor")
            else:
                # 5. spearman vs anchor (sanity range, not identical)
                for col in ['risk_hepatic_event', 'risk_death']:
                    if col in df.columns and col in anc.columns:
                        rho, _ = spearmanr(df[col].values, anc[col].values)
                        if rho < 0:
                            issues.append(f"{col} anti-correlated with anchor (rho={rho:.3f})")
                        elif rho < 0.40:
                            warns.append(f"{col} weakly correlated with anchor (rho={rho:.3f}) — verify intentional")
                        elif rho > 0.999:
                            warns.append(f"{col} ~identical to anchor (rho={rho:.4f}) — wasted submission?")
    else:
        warns.append(f"anchor {anchor} not found at {anchor_path}; skipping correlation checks")

    # 6. constant predictions
    for col in ['risk_hepatic_event', 'risk_death']:
        if col in df.columns:
            vals = df[col].values
            if np.std(vals) < 1e-10:
                issues.append(f"{col} is constant (std={np.std(vals):.2e})")
            n_unique = len(np.unique(vals))
            if n_unique < 50:
                warns.append(f"{col} has only {n_unique} unique values — quantised?")

    # 7. range sanity
    for col in ['risk_hepatic_event', 'risk_death']:
        if col in df.columns:
            v = df[col].values
            if np.any(np.isinf(v)):
                issues.append(f"{col} has inf values")
            # Concordance is rank-based — don't enforce [0,1] but warn on extreme ranges
            if v.max() - v.min() > 1e6:
                warns.append(f"{col} has very large dynamic range ({v.min():.2e} to {v.max():.2e})")

    # report
    print(f"\n=== pre_submit_check: {csv_path.name} ===")
    if issues:
        print("ISSUES (block submission):")
        for i in issues:
            print(f"  [FAIL] {i}")
    if warns:
        print("WARNINGS (review):")
        for w in warns:
            print(f"  [WARN] {w}")
    if not issues and not warns:
        print("  [PASS] all checks passed")
    print(f"  anchor: {anchor}")
    print(f"  rows: {len(df)}")
    print()
    return 2 if issues else (1 if warns else 0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', type=Path)
    ap.add_argument('--anchor', default=ANCHOR_DEFAULT,
                    help='Stem of anchor CSV under tests/results/annitia/full/ (default: 0.9302 recipe)')
    args = ap.parse_args()
    sys.exit(check(args.csv, args.anchor))
