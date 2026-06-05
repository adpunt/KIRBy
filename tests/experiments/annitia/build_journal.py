"""Build a journal notebook + zip pairing for any annitia submission CSV.

User requirement: every submission needs a paired notebook (Trustii
convention). This is a small CLI that generates the .ipynb + .zip
without me having to hand-write it each time.

Usage:
    python tests/experiments/annitia/build_journal.py \
        --csv tests/results/annitia/full/sub_master_best_per_endpoint.csv \
        --variant best_per_endpoint \
        --hep-c 0.8976 --death-c 0.9730 \
        --hep-method rank_median --death-method rank_mean \
        --hep-members "..." --death-members "..."

If --hep-c / --death-c / --variant aren't passed, this looks them up from
`tests/results/annitia/full/refuse_variants_summary.json`.

Outputs (alongside the CSV):
    <csv-stem>_notebook.ipynb
    <csv-stem>_submission.zip   (CSV + notebook)
    <csv-stem>_journal.md       (plain markdown summary too, in case
                                 Trustii prefers .md)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
import zipfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]


def _load_summary():
    p = _REPO / 'tests' / 'results' / 'annitia' / 'full' / 'refuse_variants_summary.json'
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _build_notebook(csv_basename: str, variant: str,
                    hep_c: float, death_c: float,
                    hep_method: str, death_method: str,
                    hep_members: list, death_members: list,
                    notes: str = '') -> dict:
    overall_c = (0.7 * hep_c + 0.3 * death_c) if hep_c and death_c else float('nan')
    title_md = textwrap.dedent(f"""\
        # ANNITIA MASLD survival — `{variant}` submission

        **Author:** Adelaide Punt (St Catherine's College, Oxford)
        **Submission CSV:** `{csv_basename}`
        **Pipeline:** KIRBy `tests/experiments/annitia/` (multi-layered ensemble)

        ## Honest fusion picks

        | Endpoint | Method | OOF-C |
        |----------|--------|-------|
        | Hep      | `{hep_method}` | **{hep_c:.4f}** |
        | Death    | `{death_method}` | **{death_c:.4f}** |
        | **LB-weighted (0.7·hep + 0.3·death)** |  | **{overall_c:.4f}** |

        ## Methodology

        Five processing layers feed a single fusion stage:

        1. **BE**  algorithm (Cox / Coxnet / RSF / GBSA / LGBM-AFT / NGBoost
           / CatBoost-Cox / CatBoost-AFT) × view-set (`snapshot`,
           `+clinical_pro`, `+fpca`, `+gp_smooth`, `+visit_density+clinical_slope`,
           `+meld_na+visit_density`).
        2. **CD** Phase-C target framings (pseudo-observation, landmark binary,
           pchazard, logistic-hazard) and Phase-D training tricks
           (IPCW-weighted, bagged, robust-trimmed, snapshot, self-trained,
           XGB-pairwise, Fine-Gray, D.9 cross-pollination).
        3. **HB** hep-only boost models.
        4. **BRANCHES** DSM, TSFM-FFT, TabPFN landmark, TabPFN-2.5 disc, TabSurv,
           synthetic augmentation, NHANES-pretrained MLP, cross-endpoint
           pretrain/fine-tune.
        5. **EXTERNAL** NHANES T2.x recipes (deferred where applicable).

        Fusion strategy: **{variant}** — see ensemble members below.

        All model fits use 5-fold StratifiedKFold (random_state=42) on the
        per-endpoint event flag. Hep and death use independent fold partitions
        because their valid masks differ (hep n=1106, death n=966). Each
        member's OOF prediction is from a model that did not see that row
        during training. The fusion's OOF C-index is therefore honest
        out-of-sample.
    """)

    members_md = '\n## Ensemble members\n\n### Hep\n'
    for m in hep_members:
        members_md += f'- `{m}`\n'
    members_md += '\n### Death\n'
    for m in death_members:
        members_md += f'- `{m}`\n'

    notes_md = f'\n## Notes\n\n{notes}\n' if notes else ''

    cells = [
        {'cell_type': 'markdown', 'metadata': {},
         'source': title_md.splitlines(keepends=True)},
        {'cell_type': 'markdown', 'metadata': {},
         'source': members_md.splitlines(keepends=True)},
    ]
    if notes_md.strip():
        cells.append({'cell_type': 'markdown', 'metadata': {},
                      'source': notes_md.splitlines(keepends=True)})
    cells.extend([
        {'cell_type': 'code', 'execution_count': None, 'metadata': {},
         'outputs': [], 'source': [
             "import pandas as pd\n",
             f"submission = pd.read_csv('{csv_basename}')\n",
             "print(f'rows: {len(submission)}; cols: {list(submission.columns)}; "
             "NaNs: {int(submission.isna().sum().sum())}')\n",
             "submission.head(8)"
         ]},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {},
         'outputs': [], 'source': [
             "submission.iloc[:, 1:].describe()"
         ]},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {},
         'outputs': [], 'source': [
             "import matplotlib.pyplot as plt\n",
             "fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))\n",
             "for ax, col in zip(axes, ['risk_hepatic_event', 'risk_death']):\n",
             "    ax.hist(submission[col], bins=40, edgecolor='black', alpha=0.85)\n",
             "    ax.set_title(col); ax.set_xlabel('risk score')\n",
             "plt.tight_layout(); plt.show()"
         ]},
    ])
    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                           'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.11'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--variant', default=None,
                    help='Variant name; if not given, derived from CSV stem')
    ap.add_argument('--hep-c', type=float, default=None)
    ap.add_argument('--death-c', type=float, default=None)
    ap.add_argument('--hep-method', default=None)
    ap.add_argument('--death-method', default=None)
    ap.add_argument('--hep-members', default=None,
                    help='Comma-separated; if omitted, looks up summary')
    ap.add_argument('--death-members', default=None)
    ap.add_argument('--notes', default='')
    ap.add_argument('--output-dir', default=None,
                    help='Default: same dir as the CSV')
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        sys.exit(f'CSV not found: {csv_path}')
    out_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    variant = args.variant or csv_path.stem.replace('sub_master_', '')

    # Look up missing fields from variants summary if available
    summary = _load_summary()
    hep_c = args.hep_c
    death_c = args.death_c
    hep_method = args.hep_method or variant
    death_method = args.death_method or variant
    hep_members = (args.hep_members.split(',') if args.hep_members else
                   summary.get('hepatic', {}).get('best_members', []))
    death_members = (args.death_members.split(',') if args.death_members else
                     summary.get('death', {}).get('best_members', []))
    if hep_c is None and 'hepatic' in summary:
        hep_c = summary['hepatic'].get('best_oof_c', float('nan'))
    if death_c is None and 'death' in summary:
        death_c = summary['death'].get('best_oof_c', float('nan'))
    if hep_c is None: hep_c = float('nan')
    if death_c is None: death_c = float('nan')

    nb = _build_notebook(csv_path.name, variant, hep_c, death_c,
                         hep_method, death_method, hep_members, death_members,
                         notes=args.notes)
    nb_path = out_dir / f'{csv_path.stem}_notebook.ipynb'
    nb_path.write_text(json.dumps(nb, indent=1))

    # Plain-markdown journal too (just the markdown content of the notebook)
    md_path = out_dir / f'{csv_path.stem}_journal.md'
    md_path.write_text('\n'.join(
        ''.join(c['source']) if isinstance(c['source'], list) else c['source']
        for c in nb['cells'] if c['cell_type'] == 'markdown'))

    # Zip CSV + notebook
    zip_path = out_dir / f'{csv_path.stem}_submission.zip'
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, arcname=csv_path.name)
        z.write(nb_path, arcname=nb_path.name)

    print(f'Built journal for {csv_path.name}:')
    print(f'  notebook: {nb_path}')
    print(f'  markdown: {md_path}')
    print(f'  zip:      {zip_path}')


if __name__ == '__main__':
    main()
