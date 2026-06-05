"""Build a paired submission CSV + reproducibility notebook.

For ANNITIA Trustii submission. Each submission CSV gets:
- A .csv with trustii_id, risk_hepatic_event, risk_death
- A .ipynb showing the full recipe (which OOFs, fusion math)
- An optional 'include_synth_code' switch that embeds synth-data generation
  for synth-using submissions

Usage:
    python tests/experiments/annitia/build_submission_with_notebook.py \\
        --recipe-name S1_fusion_tied --include-synth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))


# ============================================================================
# Recipe definitions — each maps recipe-name to (hep_components, death_components)
# Components are pre-computed eval/sub_<method>.csv per-method val_test arrays
# ============================================================================
RECIPES = {
    'S1_fusion_tied_regular': dict(
        description='Ensemble of hold-out-tied methods, no synth bases.',
        hep_methods=['F09_trimmed_mean', 'F03_hodges_lehmann', 'F12_uncertainty_routed',
                     'anchor_rank_mean', 'F04_bates_granger'],
        death_methods=['F01_caruana_bagged', 'F09_trimmed_mean', 'F03_hodges_lehmann'],
        fusion='rank_mean',
        synth_in_pool=False,
    ),
    'S2_F03_hodges_regular': dict(
        description='Single Hodges-Lehmann pseudomedian — robust averaging, no synth.',
        hep_methods=['F03_hodges_lehmann'],
        death_methods=['F03_hodges_lehmann'],
        fusion='single',
        synth_in_pool=False,
    ),
    'S3_F05_stacking_regular': dict(
        description='L2-logistic stacked meta-learner — non-linear blend, no synth.',
        hep_methods=['F05_stacking_l2'],
        death_methods=['F05_stacking_l2'],
        fusion='single',
        synth_in_pool=False,
    ),
    'S4_specialists_regular': dict(
        description='Per-endpoint best — F09 trimmed hep + F01 caruana death.',
        hep_methods=['F09_trimmed_mean'],
        death_methods=['F01_caruana_bagged'],
        fusion='single',
        synth_in_pool=False,
    ),
    'S5_fusion_with_synth': dict(
        description='Hold-out-tied ensemble built from pool that INCLUDES synth bases.',
        hep_methods=['F09_trimmed_mean', 'F03_hodges_lehmann', 'F12_uncertainty_routed',
                     'anchor_rank_mean', 'F04_bates_granger'],
        death_methods=['F01_caruana_bagged', 'F09_trimmed_mean', 'F03_hodges_lehmann'],
        fusion='rank_mean',
        synth_in_pool=True,
    ),
    'S6_F09_with_synth': dict(
        description='F09 trimmed-mean fused over pool with synth bases.',
        hep_methods=['F09_trimmed_mean'],
        death_methods=['F09_trimmed_mean'],
        fusion='single',
        synth_in_pool=True,
    ),
    'S7_geomean_regular': dict(
        description='Geometric mean of rank-percentiles — different from arithmetic.',
        hep_methods=['F09_trimmed_mean', 'F03_hodges_lehmann', 'F12_uncertainty_routed',
                     'anchor_rank_mean', 'F04_bates_granger'],
        death_methods=['F01_caruana_bagged', 'F09_trimmed_mean', 'F03_hodges_lehmann'],
        fusion='geomean',
        synth_in_pool=False,
    ),
}


def build_csv(recipe_name: str, eval_dir: Path) -> pd.DataFrame:
    """Build the submission CSV for `recipe_name` from eval/sub_<method>.csv files."""
    rec = RECIPES[recipe_name]
    ids = pd.read_csv(eval_dir / 'sub_F09_trimmed_mean.csv')['trustii_id']

    def _load_rank(method: str, col: str) -> np.ndarray:
        df = pd.read_csv(eval_dir / f'sub_{method}.csv')
        x = df[col].values
        return (rankdata(x) - 0.5) / len(x)

    if rec['fusion'] == 'rank_mean':
        hep = np.mean([_load_rank(m, 'risk_hepatic_event') for m in rec['hep_methods']], axis=0)
        death = np.mean([_load_rank(m, 'risk_death') for m in rec['death_methods']], axis=0)
    elif rec['fusion'] == 'geomean':
        hep = np.exp(np.mean([np.log(_load_rank(m, 'risk_hepatic_event') + 1e-9)
                              for m in rec['hep_methods']], axis=0))
        death = np.exp(np.mean([np.log(_load_rank(m, 'risk_death') + 1e-9)
                                for m in rec['death_methods']], axis=0))
    elif rec['fusion'] == 'single':
        # Use raw method's val predictions
        hep = pd.read_csv(eval_dir / f'sub_{rec["hep_methods"][0]}.csv')['risk_hepatic_event'].values
        death = pd.read_csv(eval_dir / f'sub_{rec["death_methods"][0]}.csv')['risk_death'].values
    else:
        raise ValueError(f'unknown fusion: {rec["fusion"]}')

    return pd.DataFrame({'trustii_id': ids,
                         'risk_hepatic_event': hep,
                         'risk_death': death})


def build_notebook(recipe_name: str, eval_dir: Path, csv_path: Path) -> dict:
    """Build a Jupyter notebook with the recipe's reproducibility code."""
    rec = RECIPES[recipe_name]

    cells = []

    # 0. Title
    cells.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            f'# {recipe_name}\n',
            f'\n',
            f'{rec["description"]}\n',
            '\n',
            f'**Hep methods**: {", ".join(rec["hep_methods"])}\n',
            f'**Death methods**: {", ".join(rec["death_methods"])}\n',
            f'**Fusion**: {rec["fusion"]}\n',
            f'**Synth in pool**: {rec["synth_in_pool"]}\n',
        ]
    })

    # 1. Imports
    cells.append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            "import json\n",
            "from pathlib import Path\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "from scipy.stats import rankdata\n",
            "from sklearn.model_selection import StratifiedKFold\n",
            "from sksurv.metrics import concordance_index_censored\n",
        ]
    })

    # 2. Data loading
    cells.append({
        'cell_type': 'markdown', 'metadata': {},
        'source': ['## 1. Load train + val_test\n']
    })
    cells.append({
        'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
        'source': [
            "import sys; sys.path.insert(0, 'src')\n",
            "from kirby.datasets.annitia_maslg import load_annitia\n",
            "train = load_annitia()\n",
            "val = load_annitia(filename='val_test.csv', has_outcomes=False)\n",
            "print(f'train: {len(train[\"patient_ids\"])}, val: {len(val[\"trustii_id\"])}')\n",
            "print(f'hep events: {int(train[\"hepatic\"][\"y_struct\"][\"event\"][train[\"hepatic\"][\"valid_mask\"]].sum())}/{int(train[\"hepatic\"][\"valid_mask\"].sum())}')\n",
            "print(f'death events: {int(train[\"death\"][\"y_struct\"][\"event\"][train[\"death\"][\"valid_mask\"]].sum())}/{int(train[\"death\"][\"valid_mask\"].sum())}')\n",
        ]
    })

    # 3. Synth code (only if relevant)
    if rec['synth_in_pool']:
        cells.append({
            'cell_type': 'markdown', 'metadata': {},
            'source': ['## 2. Synthetic data generation\n',
                       '\n',
                       'We augment the candidate pool with models trained on synthetic '
                       'patients drawn from copulas / mixture mechanisms (M1, M2, M5). '
                       'Each synth model produces an OOF + val_test prediction that goes into the fusion pool.\n']
        })
        cells.append({
            'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': [
                "# Synth generation pipeline (high-level — actual generation in tests/experiments/annitia/synth/)\n",
                "# Mechanisms used:\n",
                "#  M1 = phenotype-conditional Gaussian copula (matches stratification by\n",
                "#       pnpla3_driven / metabolic_obese / diabetic_aggressive / regressor)\n",
                "#  M2 = Bayesian-network-sampled trajectories (preserves visit-to-visit\n",
                "#       conditional dependencies in lab values)\n",
                "#  M5 = mixture of M1+M2 with k=4 component weights\n",
                "#\n",
                "# Synth DataFrames stored in tests/results/annitia/synth/{M1_n10000_s42.csv,...}\n",
                "# Each is read by the standard kirby loader, fit by an RSF/GBSA/MLP-DH learner,\n",
                "# OOF predictions saved to branches/oof_<endpoint>_synth_<mech>_<model>_<P>__snapshot+summary.npy\n",
                "synth_dir = Path('tests/results/annitia/synth')\n",
                "synth_files = sorted(synth_dir.glob('*.csv'))\n",
                "print(f'Synth datasets on disk: {[f.name for f in synth_files]}')\n",
            ]
        })
        cells.append({
            'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': [
                "# Verify synth OOF files match the fusion pool registry\n",
                "synth_grid = json.load(open(synth_dir / 'grid_summary.json'))\n",
                "print('Synth conditions registered:')\n",
                "for c in synth_grid['conditions']:\n",
                "    print(f\"  {c.get('endpoint','?'):8s} {c.get('cid','?')[:55]:55s} C={c.get('mean_cindex','?'):.4f}\")\n",
            ]
        })

    # 4. Component assembly
    cells.append({
        'cell_type': 'markdown', 'metadata': {},
        'source': ['## 3. Load fusion components\n']
    })
    cells.append({
        'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
        'source': [
            "EVAL_DIR = Path('tests/results/annitia/eval')\n",
            "ids = pd.read_csv(EVAL_DIR / 'sub_F09_trimmed_mean.csv')['trustii_id']\n",
            "\n",
            "def _rank01(x):\n",
            "    return (rankdata(x) - 0.5) / len(x)\n",
            "\n",
            "def _load_rank(method, col):\n",
            "    df = pd.read_csv(EVAL_DIR / f'sub_{method}.csv')\n",
            "    return _rank01(df[col].values)\n",
            "\n",
            f"hep_methods = {rec['hep_methods']!r}\n",
            f"death_methods = {rec['death_methods']!r}\n",
        ]
    })

    # 5. Fusion math
    cells.append({
        'cell_type': 'markdown', 'metadata': {},
        'source': [f'## 4. Apply fusion (`{rec["fusion"]}`)\n']
    })
    if rec['fusion'] == 'rank_mean':
        fusion_src = [
            "# Rank-mean fusion: rank-transform each component, then arithmetic mean\n",
            "hep_pred = np.mean([_load_rank(m, 'risk_hepatic_event') for m in hep_methods], axis=0)\n",
            "death_pred = np.mean([_load_rank(m, 'risk_death') for m in death_methods], axis=0)\n",
        ]
    elif rec['fusion'] == 'geomean':
        fusion_src = [
            "# Geometric mean of rank-percentiles: more robust to outlier components\n",
            "hep_pred = np.exp(np.mean([np.log(_load_rank(m, 'risk_hepatic_event') + 1e-9)\n",
            "                            for m in hep_methods], axis=0))\n",
            "death_pred = np.exp(np.mean([np.log(_load_rank(m, 'risk_death') + 1e-9)\n",
            "                              for m in death_methods], axis=0))\n",
        ]
    else:  # single
        fusion_src = [
            "# Single-method submission: take that method's predictions directly\n",
            f"hep_pred = pd.read_csv(EVAL_DIR / f'sub_{{hep_methods[0]}}.csv')['risk_hepatic_event'].values\n",
            f"death_pred = pd.read_csv(EVAL_DIR / f'sub_{{death_methods[0]}}.csv')['risk_death'].values\n",
        ]
    cells.append({
        'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
        'source': fusion_src
    })

    # 6. Write CSV
    cells.append({
        'cell_type': 'markdown', 'metadata': {},
        'source': ['## 5. Build submission CSV\n']
    })
    cells.append({
        'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
        'source': [
            "submission = pd.DataFrame({\n",
            "    'trustii_id': ids,\n",
            "    'risk_hepatic_event': hep_pred,\n",
            "    'risk_death': death_pred,\n",
            "})\n",
            f"submission.to_csv('{recipe_name}.csv', index=False)\n",
            "print(f'Wrote {recipe_name}.csv  rows={len(submission)}')\n",
            "submission.head()\n"
        ]
    })

    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.11'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recipe-name', required=True, choices=list(RECIPES.keys()))
    ap.add_argument('--eval-dir', default='tests/results/annitia/eval')
    ap.add_argument('--output-dir', default='tests/results/annitia/submissions')
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_csv(args.recipe_name, eval_dir)
    csv_path = out_dir / f'{args.recipe_name}.csv'
    df.to_csv(csv_path, index=False)
    print(f'  CSV: {csv_path}  rows={len(df)}')

    nb = build_notebook(args.recipe_name, eval_dir, csv_path)
    nb_path = out_dir / f'{args.recipe_name}.ipynb'
    nb_path.write_text(json.dumps(nb, indent=2))
    print(f'  Notebook: {nb_path}')


if __name__ == '__main__':
    main()
