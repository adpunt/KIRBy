"""Build the final submission from competition_methods_eval results.

Strategy (post-cherry-pick lessons):
1. Load comparison.json from eval/.
2. Per endpoint, identify all methods within 1× SE of the top hold-out C
   (i.e. statistically tied for the lead).
3. Build the submission as the AVERAGE of those tied methods' val_test predictions.
   This is the only ensembling step — it's a final defense against picking the
   single OOF/hold-out winner (winner's curse).
4. Compare predicted-LB-by-bootstrap-on-anchor vs the proposed submission with
   honest "if this were just noise" calibration.
5. Write the submission CSV + a verdict markdown explaining the recipe.

Usage:
    python tests/experiments/annitia/build_submission_from_eval.py \\
        --eval-dir tests/results/annitia/eval --tag mse_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / 'src'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-dir', default='tests/results/annitia/eval')
    ap.add_argument('--tag', default='compmethods')
    ap.add_argument('--n-tie', type=int, default=3,
                    help='Always include top-N regardless of SE; can also use SE rule')
    ap.add_argument('--se-mult', type=float, default=1.0,
                    help='SE multiplier for tie-detection band')
    ap.add_argument('--exclude', nargs='+', default=[],
                    help='Methods to exclude (e.g. F02_geomean_rank if visibly bad)')
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    cmp_p = eval_dir / 'comparison.json'
    if not cmp_p.exists():
        sys.exit(f'comparison.json missing — run competition_methods_eval first')
    cmp = json.loads(cmp_p.read_text())

    summary_rows = cmp['summary_rows']
    # group by endpoint
    by_ep = {}
    for r in summary_rows:
        by_ep.setdefault(r['endpoint'], []).append(r)

    chosen = {}  # endpoint -> list of method names to ensemble
    decisions = []
    for endpoint, rows in by_ep.items():
        # filter exclude
        rows = [r for r in rows if r['method'] not in args.exclude]
        rows.sort(key=lambda r: -r['c_holdout'])
        top = rows[0]
        top_se = top.get('c_holdout_se', 0)
        # Tie band: anything within args.se_mult * top_se of top_holdout
        band = args.se_mult * top_se
        ties = [r for r in rows if r['c_holdout'] >= top['c_holdout'] - band]
        # Always include at least n_tie best
        if len(ties) < args.n_tie:
            ties = rows[:args.n_tie]
        chosen[endpoint] = [r['method'] for r in ties]
        decisions.append((endpoint, top, ties))
        print(f'\n=== {endpoint} ===')
        print(f'  top: {top["method"]:25s} hold C={top["c_holdout"]:.4f} ±{top_se:.4f}')
        print(f'  tied within {args.se_mult}*SE band ({band:.4f}):')
        for r in ties:
            print(f'    {r["method"]:25s} hold={r["c_holdout"]:.4f} '
                  f'±{r.get("c_holdout_se", 0):.4f} train={r["c_train"]:.4f} '
                  f'unoH={r["uno_holdout"]:.4f}')

    # Build ensemble val_test predictions per endpoint
    out_csv = eval_dir / f'sub_master_{args.tag}.csv'
    val_pred = {}
    for endpoint, methods in chosen.items():
        col = 'risk_hepatic_event' if endpoint == 'hepatic' else 'risk_death'
        ranks = []
        for m in methods:
            sub_p = eval_dir / f'sub_{m}.csv'
            if not sub_p.exists():
                print(f'WARN: {sub_p} missing')
                continue
            df = pd.read_csv(sub_p)
            if col not in df.columns:
                print(f'WARN: {sub_p} missing {col}')
                continue
            # Rank-percentile within method, then average
            from scipy.stats import rankdata
            r = (rankdata(df[col].values) - 0.5) / len(df)
            ranks.append(r)
        if not ranks:
            continue
        val_pred[endpoint] = np.mean(np.stack(ranks, axis=0), axis=0)
        # Use the first sub's trustii_id col
        ids = pd.read_csv(eval_dir / f'sub_{methods[0]}.csv')['trustii_id']
        val_pred[f'_ids_{endpoint}'] = ids

    # Write merged CSV
    if 'hepatic' in val_pred and 'death' in val_pred:
        df = pd.DataFrame({
            'trustii_id': val_pred['_ids_hepatic'],
            'risk_hepatic_event': val_pred['hepatic'],
            'risk_death': val_pred['death'],
        })
        df.to_csv(out_csv, index=False)
        print(f'\nSubmission written: {out_csv}')
        print(f'  rows: {len(df)}')

    # Write verdict markdown
    md_lines = ['# Submission verdict', '',
                f'Tag: `{args.tag}`', '',
                'Recipe: ensemble (rank-mean) of all hold-out-tied methods per endpoint.',
                'Tie band: `{}* SE`.'.format(args.se_mult), '']
    for endpoint, methods in chosen.items():
        md_lines.append(f'## {endpoint}\n')
        md_lines.append('Components: ' + ', '.join(f'`{m}`' for m in methods))
        md_lines.append('')
    md_lines.append('## Why this is safer than cherry-pick\n')
    md_lines.append('The cherry-pick failure on 2026-05-08 (`anomaly_blend_meta` × '
                    '`weighted_anomaly_per_patient`) had OOF 0.9323 / LB 0.898 '
                    '(gap +0.034) because both components fit weights on OOF. '
                    'Here, every method was fit on `train_idx` and evaluated on '
                    '`holdout_idx` — none saw the validation labels during fitting. '
                    'The ensemble of hold-out-tied methods further dampens any '
                    'remaining selection bias.')
    (eval_dir / f'verdict_{args.tag}.md').write_text('\n'.join(md_lines))
    print(f'  verdict: {eval_dir / f"verdict_{args.tag}.md"}')


if __name__ == '__main__':
    main()
