"""Cross-reference: match val_test contested patients to training failure patterns.

Workflow:
1. Load training failure analysis (FN-like and FP-like patient indices on training).
2. Load val_test contention table.
3. Compute features for both via build_views.
4. For each highly-contested val_test patient, find nearest training neighbour
   in feature space.
5. Check whether the nearest training neighbour is in the FN-like or FP-like
   set — if so, the val_test patient is likely a similar failure mode.
6. Report per-endpoint: which val_test contested patients map to which
   training failure types.

Output:
  tests/results/annitia/full/cross_reference.json
  tests/results/annitia/full/cross_reference.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia          # noqa: E402
from experiments.annitia.views import build_views              # noqa: E402

FULL = REPO / 'tests/results/annitia/full'


def features_matrix(d):
    views = build_views(d, view_names=['snapshot', 'clinical_pro'])
    X = np.concatenate([views[v][0] for v in ['snapshot', 'clinical_pro']], axis=1)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    train = load_annitia()
    val_path = '/Volumes/seagate/kirby/annitia/val_test.csv'
    val = load_annitia(data_dir=str(Path(val_path).parent),
                       filename=Path(val_path).name, has_outcomes=False)

    Xt = features_matrix(train)
    Xv = features_matrix(val)
    scaler = StandardScaler().fit(Xt)
    Xt_s = scaler.transform(Xt)
    Xv_s = scaler.transform(Xv)

    failure = json.loads((FULL / 'failure_analysis.json').read_text())

    out = {}
    for ep in ('hepatic', 'death'):
        f = failure[ep]
        if f is None:
            continue
        fn_idx = set(f['fn_like_idx'])
        fp_idx = set(f['fp_like_idx'])
        hard_idx = set(f['hardest_idx'])
        # All training failures (union)
        failure_set = fn_idx | fp_idx | hard_idx

        # Load val_test contention
        cont_csv = FULL / f'valtest_contention_{ep}.csv'
        if not cont_csv.exists():
            continue
        cont = pd.read_csv(cont_csv)
        # Top contested 10% or anything with IQR > 0.15
        cont_sorted = cont.sort_values('contention', ascending=False).head(50)
        contested_trustii = cont_sorted['trustii_id'].values

        # Map trustii_id back to row index
        v_trustii = val.get('trustii_id')
        # the load_annitia gives trustii_id as None for train. for val we expect
        # it to be a list/array
        if v_trustii is None:
            # Try patient_ids
            v_trustii = val['patient_ids']
        # Build mapping trustii_id -> val row
        id_to_idx = {int(t): i for i, t in enumerate(v_trustii)}

        # 5-NN on training
        nn = NearestNeighbors(n_neighbors=5).fit(Xt_s)
        result_rows = []
        for tid in contested_trustii:
            if tid not in id_to_idx:
                continue
            vidx = id_to_idx[tid]
            d, idx = nn.kneighbors(Xv_s[vidx:vidx+1])
            train_neighbours = idx[0].tolist()
            # Check if any neighbours are in failure sets
            n_fn = sum(1 for n in train_neighbours if n in fn_idx)
            n_fp = sum(1 for n in train_neighbours if n in fp_idx)
            n_hard = sum(1 for n in train_neighbours if n in hard_idx)
            n_failure = sum(1 for n in train_neighbours if n in failure_set)
            result_rows.append({
                'trustii_id': int(tid),
                'val_idx': int(vidx),
                'contention': float(cont_sorted[cont_sorted['trustii_id'] == tid]['contention'].values[0]),
                'mean_rank': float(cont_sorted[cont_sorted['trustii_id'] == tid]['mean_rank'].values[0]),
                'nn5_in_fn_like': n_fn,
                'nn5_in_fp_like': n_fp,
                'nn5_in_hard_set': n_hard,
                'nn5_in_failure_union': n_failure,
                'nearest_train_idx': int(train_neighbours[0]),
                'nn_distance': float(d[0][0]),
            })

        result_df = pd.DataFrame(result_rows)
        out_csv = FULL / f'cross_reference_{ep}.csv'
        result_df.to_csv(out_csv, index=False)

        # Aggregate
        total = len(result_df)
        with_fn = (result_df['nn5_in_fn_like'] > 0).sum()
        with_fp = (result_df['nn5_in_fp_like'] > 0).sum()
        with_hard = (result_df['nn5_in_hard_set'] > 0).sum()
        with_fail = (result_df['nn5_in_failure_union'] > 0).sum()

        out[ep] = {
            'n_contested_analysed': int(total),
            'pct_with_fn_neighbour': float(with_fn / max(total, 1)),
            'pct_with_fp_neighbour': float(with_fp / max(total, 1)),
            'pct_with_hard_neighbour': float(with_hard / max(total, 1)),
            'pct_with_any_failure_neighbour': float(with_fail / max(total, 1)),
        }
        print(f'\n=== {ep} ===')
        print(f'  Top {total} contested val_test patients analysed:')
        print(f'    {with_fn}/{total} ({with_fn/max(total,1):.1%}) have nearest-5-train-neighbour in FN-like set')
        print(f'    {with_fp}/{total} ({with_fp/max(total,1):.1%}) have nearest-5-train-neighbour in FP-like set')
        print(f'    {with_hard}/{total} ({with_hard/max(total,1):.1%}) have nearest in hardest-30 set')
        print(f'    {with_fail}/{total} ({with_fail/max(total,1):.1%}) have nearest in any-failure set')
        print(f'\n  Top 15 contested val_test {ep} patients with their training neighbours:')
        for _, r in result_df.head(15).iterrows():
            tags = []
            if r['nn5_in_fn_like'] > 0: tags.append(f"FN×{r['nn5_in_fn_like']}")
            if r['nn5_in_fp_like'] > 0: tags.append(f"FP×{r['nn5_in_fp_like']}")
            if r['nn5_in_hard_set'] > 0: tags.append(f"HARD×{r['nn5_in_hard_set']}")
            tag_str = ', '.join(tags) if tags else '(no failure neighbours)'
            print(f'    val_id={int(r["trustii_id"]):4d} cont={r["contention"]:.3f} '
                  f'mean_rank={r["mean_rank"]:.3f} → {tag_str}')

    (FULL / 'cross_reference.json').write_text(json.dumps(out, indent=2))
    print(f'\nwrote {FULL}/cross_reference.json')

    # Markdown
    md = ['# Cross-reference: val_test contested ↔ training failures',
          '',
          'For the most-contested val_test patients, find their 5 nearest training',
          'neighbours in feature space. Check whether those training patients are',
          'failure cases (FN-like, FP-like, or simply high-discordance).',
          '',
          'High overlap → val_test contested patients ARE the same failure modes',
          'we see in training. Specialist routing should focus there.',
          '',
          '## Per-endpoint summary',
          '']
    for ep, r in out.items():
        md.append(f'### {ep}')
        md.append('')
        md.append(f'- Top {r["n_contested_analysed"]} contested val_test patients analysed')
        md.append(f'- {r["pct_with_fn_neighbour"]:.1%} have a nearest-train-neighbour in FN-like set')
        md.append(f'- {r["pct_with_fp_neighbour"]:.1%} have a nearest-train-neighbour in FP-like set')
        md.append(f'- {r["pct_with_hard_neighbour"]:.1%} have a nearest-train-neighbour in hardest-30 set')
        md.append(f'- {r["pct_with_any_failure_neighbour"]:.1%} have a nearest in ANY failure category')
        md.append('')

    (FULL / 'cross_reference.md').write_text('\n'.join(md))
    print(f'wrote {FULL}/cross_reference.md')


if __name__ == '__main__':
    main()
