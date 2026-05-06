#!/usr/bin/env python3
"""Build Trustii submission zips for the R2 / stack / master variants.

Each zip contains:
  - annitia_submission.ipynb   self-contained notebook
  - the relevant model pickles
  - the submission CSV
  - train.csv, val_test.csv (for reproducibility)

Usage:
    python build_v2_zips.py r2     --base /Volumes/seagate/kirby/annitia
    python build_v2_zips.py stack  --base /Volumes/seagate/kirby/annitia
    python build_v2_zips.py master --base /Volumes/seagate/kirby/annitia
"""

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
SRC_LOADER = HERE.parent.parent.parent / 'src' / 'kirby' / 'datasets' / 'annitia_maslg.py'
SRC_VIEWS = HERE / 'views.py'
SRC_SURVIVAL = HERE / 'survival.py'
SRC_STACKING = HERE / 'stacking.py'


def read_module_body(path: Path) -> str:
    """Strip leading docstring + intra-package imports."""
    text = path.read_text()
    if text.startswith('"""'):
        end = text.find('"""', 3)
        if end != -1:
            text = text[end + 3:].lstrip('\n')
    out_lines = []
    skip_continuation = False
    for line in text.splitlines():
        if skip_continuation:
            if ')' in line:
                skip_continuation = False
            continue
        s = line.lstrip()
        if (s.startswith('from kirby.') or s.startswith('from experiments.')
                or s.startswith('import kirby.')):
            if '(' in line and ')' not in line:
                skip_continuation = True
            continue
        out_lines.append(line)
    return '\n'.join(out_lines).strip() + '\n'


def _abstract(label: str, summary: dict) -> str:
    if label == 'r2':
        return (
            "## Submission: R2 (random-forest hybrid search)\n\n"
            "**Strategy.** A KIRBy-style hybrid search enumerated all subsets of "
            "8 hand-crafted feature views (`snapshot`, `slope`, `summary`, `delta`, "
            "`missingness`, `nit`, `leakage`, `clinical`) crossed with 4 sksurv "
            "downstream models (CoxPH, Coxnet, RandomSurvivalForest, "
            "GradientBoostingSurvivalAnalysis), at two feature-pruning levels "
            "(none, top-50 by importance). Each candidate was scored by 5-fold "
            "stratified-on-event CV C-index. The top-3 candidates per endpoint "
            "were refit on all valid patients and rank-ensembled at predict time.\n\n"
            f"**Hepatic winner:** {summary['hep']}\n\n"
            f"**Death winner:** {summary['death']}\n"
        )
    if label == 'stack':
        return (
            "## Submission: Stacking (Coxnet meta-learner over survival bases)\n\n"
            "**Strategy.** Same KIRBy hybrid-search machinery as R2, but with a "
            "5th downstream model: a **stacking meta-learner**. The stacker "
            "generates 5-fold out-of-fold (OOF) risk predictions from each "
            "base model (CoxPH/Coxnet/RSF/GBSA), trains an outer Coxnet on the "
            "standardised OOF stack, and refits the bases on full training data "
            "for inference. This is the textbook stacking recipe adapted for "
            "censored survival outcomes.\n\n"
            f"**Hepatic winner:** {summary['hep']}\n\n"
            f"**Death winner:** {summary['death']}\n"
        )
    if label == 'master':
        return (
            "## Submission: Master ensemble (R2 + Stack)\n\n"
            "**Strategy.** Two independently fitted submissions are combined by "
            "**rank-averaging** their per-patient risk scores per endpoint, with "
            "equal weights. R2 uses random-forest survival predictions; Stack "
            "uses a Coxnet meta-learner over base-model OOF predictions. They "
            "found the same hepatic winner (so hepatic predictions are "
            "identical) but disagree on the death endpoint, where rank-averaging "
            "shaves variance.\n\n"
            "Each submission is itself already an internal top-3 rank-ensemble "
            "within its branch, so the master is effectively a 6-way ensemble "
            "of distinct (subset, model) configurations.\n\n"
            f"**Hepatic winner (both branches):** {summary['hep']}\n\n"
            f"**R2 death winner:** {summary['death_r2']}\n\n"
            f"**Stack death winner:** {summary['death_stack']}\n"
        )
    raise ValueError(label)


def _summary_line(search):
    b = search['best']
    return (f"`{'+'.join(b['views'])}` / **{b['model']}** "
            f"(C={b['mean_cindex']:.4f} ± {b['std_cindex']:.4f})")


def _build_results_table(search, label):
    rows = sorted(search['per_condition'], key=lambda r: -r['mean_cindex'])
    out = ['| Rank | Views | Model | Prune | Dim | C-index |',
           '|-----:|-------|-------|------:|----:|--------:|']
    for i, r in enumerate(rows[:10], 1):
        v = '+'.join(r['views'])
        out.append(f'| {i} | `{v}` | `{r["model"]}` | {r.get("prune", "—")} | '
                   f'{r["n_features"]} | {r["mean_cindex"]:.4f} ± {r["std_cindex"]:.4f} |')
    return '\n'.join(out)


def make_notebook_single(label: str, models_dir: Path,
                         submission_csv: Path) -> nbf.NotebookNode:
    """Build the notebook for r2 or stack — single set of pickles."""
    hep_search = json.load(open(models_dir / 'annitia_hepatic_search.json'))
    death_search = json.load(open(models_dir / 'annitia_death_search.json'))
    summary = {
        'hep': _summary_line(hep_search),
        'death': _summary_line(death_search),
    }

    nb = nbf.v4.new_notebook()
    cells = []

    # Title + abstract
    title = {'r2': 'R2', 'stack': 'Stack'}[label]
    cells.append(nbf.v4.new_markdown_cell(
        f"# ANNITIA MASLD Risk Stratification — {title} Submission\n\n"
        "**Trustii challenge:** [https://app.trustii.io/datasets/1551](https://app.trustii.io/datasets/1551)\n\n"
        + _abstract(label, summary)
    ))

    # Imports
    cells.append(nbf.v4.new_markdown_cell("## Imports & paths"))
    cells.append(nbf.v4.new_code_cell(
        "import os, sys, pickle, json, types\n"
        "from itertools import combinations\n"
        "from pathlib import Path\n"
        "from typing import Dict, List, Optional, Tuple\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sklearn.model_selection import StratifiedKFold\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.neighbors import NearestNeighbors\n"
        "\n"
        "from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis\n"
        "from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis\n"
        "from sksurv.metrics import concordance_index_censored\n"
        "\n"
        "TRAIN_CSV = 'train.csv'\n"
        "VAL_CSV   = 'val_test.csv'\n"
        "MODEL_HEPATIC = 'annitia_hepatic_model.pkl'\n"
        "MODEL_DEATH   = 'annitia_death_model.pkl'\n"
        "OUTPUT_CSV    = 'annitia_submission.csv'"
    ))

    # Inlined modules
    cells.append(nbf.v4.new_markdown_cell(
        "## Data loader\n\n"
        "Reads the wide-format CSV and produces per-patient records with "
        "static covariates, longitudinal `(age, value)` lists, and per-endpoint "
        "survival tuples (event flag, duration since first visit)."
    ))
    cells.append(nbf.v4.new_code_cell(read_module_body(SRC_LOADER)))

    cells.append(nbf.v4.new_markdown_cell(
        "## Feature view builders\n\n"
        "Eight deterministic transforms over each patient's record. The same "
        "builders run on training and validation patients alike."
    ))
    cells.append(nbf.v4.new_code_cell(read_module_body(SRC_VIEWS)))

    cells.append(nbf.v4.new_markdown_cell("## Survival downstream models"))
    cells.append(nbf.v4.new_code_cell(read_module_body(SRC_SURVIVAL)))

    if label == 'stack':
        cells.append(nbf.v4.new_markdown_cell(
            "## Stacking meta-learner\n\n"
            "Outer Coxnet trained on out-of-fold base-model predictions."
        ))
        cells.append(nbf.v4.new_code_cell(read_module_body(SRC_STACKING)))

    # Module aliasing for pickle compatibility
    alias_classes = ['SurvivalCoxPH', 'SurvivalCoxnet', 'SurvivalRSF',
                     'SurvivalGBSA', 'SURVIVAL_MODELS', 'concordance']
    if label == 'stack':
        alias_classes.append('SurvivalStacker')
    cells.append(nbf.v4.new_markdown_cell(
        "## Load pre-trained models\n\n"
        "Aliases `experiments.annitia.*` modules to this notebook's namespace "
        "so the saved pickles deserialize correctly."
    ))
    cells.append(nbf.v4.new_code_cell(
        "_alias_root = types.ModuleType('experiments')\n"
        "_alias_pkg  = types.ModuleType('experiments.annitia')\n"
        "_alias_surv = types.ModuleType('experiments.annitia.survival')\n"
        + ("_alias_stk = types.ModuleType('experiments.annitia.stacking')\n"
           "for _name in ('SurvivalStacker',):\n"
           "    setattr(_alias_stk, _name, globals()[_name])\n"
           "sys.modules['experiments.annitia.stacking'] = _alias_stk\n"
           if label == 'stack' else "")
        + f"for _name in {alias_classes!r}:\n"
        "    if _name in globals():\n"
        "        setattr(_alias_surv, _name, globals()[_name])\n"
        "sys.modules.setdefault('experiments', _alias_root)\n"
        "sys.modules.setdefault('experiments.annitia', _alias_pkg)\n"
        "sys.modules['experiments.annitia.survival'] = _alias_surv\n"
        "\n"
        "with open(MODEL_HEPATIC, 'rb') as f:\n"
        "    hep_payload = pickle.load(f)\n"
        "with open(MODEL_DEATH, 'rb') as f:\n"
        "    death_payload = pickle.load(f)\n"
        "\n"
        "for name, p in [('hepatic', hep_payload), ('death', death_payload)]:\n"
        "    print(f'{name:8s} -> views={\"+\".join(p[\"view_names\"])} '\n"
        "          f'model={p[\"model_name\"]} prune={p.get(\"prune\")}')"
    ))

    # Predict on validation
    cells.append(nbf.v4.new_markdown_cell("## Generate predictions"))
    cells.append(nbf.v4.new_code_cell(
        "val_data = load_annitia(data_dir='.', filename=VAL_CSV, has_outcomes=False)\n"
        "assert val_data['trustii_id'] is not None\n"
        "val_views = build_views(val_data)\n"
        "print(f'{len(val_data[\"patient_ids\"])} validation patients')\n"
        "\n"
        "def predict_risk(payload, views):\n"
        "    X = np.concatenate([views[v][0] for v in payload['view_names']], axis=1)\n"
        "    fi = payload.get('feature_indices')\n"
        "    if fi is not None:\n"
        "        X = X[:, fi]\n"
        "    return payload['model'].predict_risk(X)\n"
        "\n"
        "risk_hepatic = predict_risk(hep_payload, val_views)\n"
        "risk_death   = predict_risk(death_payload, val_views)\n"
        "\n"
        "submission = pd.DataFrame({\n"
        "    'trustii_id':         val_data['trustii_id'],\n"
        "    'risk_hepatic_event': risk_hepatic,\n"
        "    'risk_death':         risk_death,\n"
        "})\n"
        "submission.to_csv(OUTPUT_CSV, index=False)\n"
        "print(f'Wrote {len(submission)} rows -> {OUTPUT_CSV}')\n"
        "submission.head()"
    ))

    # Results
    cells.append(nbf.v4.new_markdown_cell(
        "## Cross-validation results — top 10 per endpoint\n\n"
        "**Hepatic:**\n\n" + _build_results_table(hep_search, label) + "\n\n"
        "**Death:**\n\n" + _build_results_table(death_search, label)
    ))

    # Limitations
    cells.append(nbf.v4.new_markdown_cell(
        "## Limitations\n\n"
        "- 47 hepatic events / 1106 valid patients — sparse positive class.\n"
        "- 21% of `death` cells blank → those patients are dropped from the death endpoint.\n"
        "- Time axis is patient age (`age_at_event - age_v1`).\n"
        "- The `missingness`/`leakage` views encode follow-up span; they generalise "
        "to the held-out set if its data-generating process matches training.\n"
        "- Held-out leaderboard performance is the trustworthy estimate; CV is an "
        "upper bound (the hybrid search picks the best of many candidates).\n"
    ))

    nb.cells = cells
    nb.metadata['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    }
    return nb


def make_notebook_master(r2_dir: Path, stack_dir: Path,
                          master_csv: Path) -> nbf.NotebookNode:
    """Build the master notebook — loads both R2 and stack, rank-averages."""
    hep_r2 = json.load(open(r2_dir / 'annitia_hepatic_search.json'))
    hep_st = json.load(open(stack_dir / 'annitia_hepatic_search.json'))
    d_r2 = json.load(open(r2_dir / 'annitia_death_search.json'))
    d_st = json.load(open(stack_dir / 'annitia_death_search.json'))
    summary = {
        'hep': _summary_line(hep_r2),
        'death_r2': _summary_line(d_r2),
        'death_stack': _summary_line(d_st),
    }

    nb = nbf.v4.new_notebook()
    cells = []
    cells.append(nbf.v4.new_markdown_cell(
        "# ANNITIA MASLD Risk Stratification — Master (R2 + Stack)\n\n"
        "**Trustii challenge:** [https://app.trustii.io/datasets/1551](https://app.trustii.io/datasets/1551)\n\n"
        + _abstract('master', summary)
    ))

    cells.append(nbf.v4.new_markdown_cell("## Imports"))
    cells.append(nbf.v4.new_code_cell(
        "import os, sys, pickle, json, types\n"
        "from itertools import combinations\n"
        "from pathlib import Path\n"
        "from typing import Dict, List, Optional, Tuple\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sklearn.model_selection import StratifiedKFold\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.neighbors import NearestNeighbors\n"
        "from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis\n"
        "from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis\n"
        "from sksurv.metrics import concordance_index_censored\n"
        "\n"
        "TRAIN_CSV = 'train.csv'\n"
        "VAL_CSV   = 'val_test.csv'\n"
        "OUTPUT_CSV = 'annitia_submission.csv'\n"
        "R2_DIR    = 'r2_artifacts'\n"
        "STACK_DIR = 'stack_artifacts'"
    ))

    cells.append(nbf.v4.new_markdown_cell("## Data loader, feature views, survival models, stacking"))
    cells.append(nbf.v4.new_code_cell(read_module_body(SRC_LOADER)))
    cells.append(nbf.v4.new_code_cell(read_module_body(SRC_VIEWS)))
    cells.append(nbf.v4.new_code_cell(read_module_body(SRC_SURVIVAL)))
    cells.append(nbf.v4.new_code_cell(read_module_body(SRC_STACKING)))

    cells.append(nbf.v4.new_markdown_cell("## Load both R2 and Stack pickles"))
    cells.append(nbf.v4.new_code_cell(
        "_alias_root = types.ModuleType('experiments')\n"
        "_alias_pkg  = types.ModuleType('experiments.annitia')\n"
        "_alias_surv = types.ModuleType('experiments.annitia.survival')\n"
        "_alias_stk  = types.ModuleType('experiments.annitia.stacking')\n"
        "for _n in ('SurvivalCoxPH','SurvivalCoxnet','SurvivalRSF',\n"
        "           'SurvivalGBSA','SURVIVAL_MODELS','concordance'):\n"
        "    if _n in globals(): setattr(_alias_surv, _n, globals()[_n])\n"
        "if 'SurvivalStacker' in globals():\n"
        "    setattr(_alias_stk, 'SurvivalStacker', globals()['SurvivalStacker'])\n"
        "sys.modules.setdefault('experiments', _alias_root)\n"
        "sys.modules.setdefault('experiments.annitia', _alias_pkg)\n"
        "sys.modules['experiments.annitia.survival'] = _alias_surv\n"
        "sys.modules['experiments.annitia.stacking'] = _alias_stk\n"
        "\n"
        "def _load(d, ep):\n"
        "    with open(f'{d}/annitia_{ep}_model.pkl', 'rb') as f:\n"
        "        return pickle.load(f)\n"
        "r2_hep    = _load(R2_DIR, 'hepatic')\n"
        "r2_death  = _load(R2_DIR, 'death')\n"
        "stk_hep   = _load(STACK_DIR, 'hepatic')\n"
        "stk_death = _load(STACK_DIR, 'death')\n"
        "for name, p in [('R2/hep', r2_hep), ('R2/death', r2_death),\n"
        "                ('Stack/hep', stk_hep), ('Stack/death', stk_death)]:\n"
        "    print(f'{name:12s} -> {\"+\".join(p[\"view_names\"])} / {p[\"model_name\"]}')"
    ))

    cells.append(nbf.v4.new_markdown_cell("## Predict and rank-average"))
    cells.append(nbf.v4.new_code_cell(
        "val_data = load_annitia(data_dir='.', filename=VAL_CSV, has_outcomes=False)\n"
        "val_views = build_views(val_data)\n"
        "n = len(val_data['trustii_id'])\n"
        "\n"
        "def predict_risk(payload, views):\n"
        "    X = np.concatenate([views[v][0] for v in payload['view_names']], axis=1)\n"
        "    fi = payload.get('feature_indices')\n"
        "    if fi is not None: X = X[:, fi]\n"
        "    return payload['model'].predict_risk(X)\n"
        "\n"
        "def rank_average(*risks):\n"
        "    out = np.zeros(len(risks[0]))\n"
        "    for r in risks:\n"
        "        ranks = np.empty(len(r))\n"
        "        ranks[np.argsort(r)] = np.arange(len(r))\n"
        "        out += ranks\n"
        "    return out / len(risks)\n"
        "\n"
        "risk_hep_master   = rank_average(predict_risk(r2_hep, val_views),\n"
        "                                  predict_risk(stk_hep, val_views))\n"
        "risk_death_master = rank_average(predict_risk(r2_death, val_views),\n"
        "                                  predict_risk(stk_death, val_views))\n"
        "\n"
        "submission = pd.DataFrame({\n"
        "    'trustii_id':         val_data['trustii_id'],\n"
        "    'risk_hepatic_event': risk_hep_master,\n"
        "    'risk_death':         risk_death_master,\n"
        "})\n"
        "submission.to_csv(OUTPUT_CSV, index=False)\n"
        "print(f'Wrote {len(submission)} rows -> {OUTPUT_CSV}')\n"
        "submission.head()"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Per-branch results\n\n"
        "**R2 — Hepatic top 10:**\n\n" + _build_results_table(hep_r2, 'r2') + "\n\n"
        "**R2 — Death top 10:**\n\n" + _build_results_table(d_r2, 'r2') + "\n\n"
        "**Stack — Hepatic top 10:**\n\n" + _build_results_table(hep_st, 'stack') + "\n\n"
        "**Stack — Death top 10:**\n\n" + _build_results_table(d_st, 'stack')
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Limitations\n\n"
        "- 47 hepatic events / 1106 valid patients — sparse positive class.\n"
        "- 21% of `death` cells blank.\n"
        "- Time axis is patient age.\n"
        "- The two branches agreed on the hepatic winner — master is identical "
        "to either branch on hepatic.\n"
        "- Death CV difference between R2 and Stack is small (~0.0005) — "
        "rank-averaging is conservative variance reduction, not a clear improvement.\n"
    ))

    nb.cells = cells
    nb.metadata['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    }
    return nb


def build(label: str, base: Path):
    if label == 'r2':
        models = base / 'r2_artifacts'
        csv = base / 'results_r2.csv'
        nb = make_notebook_single('r2', models, csv)
    elif label == 'stack':
        models = base / 'stack_artifacts'
        csv = base / 'results_stack.csv'
        nb = make_notebook_single('stack', models, csv)
    elif label == 'master':
        r2_dir = base / 'r2_artifacts'
        stack_dir = base / 'stack_artifacts'
        csv = base / 'master_2way.csv'
        nb = make_notebook_master(r2_dir, stack_dir, csv)
        models = None  # both dirs bundled below
    else:
        sys.exit(f'unknown label: {label}')

    out_nb = base / f'zip_{label}_notebook.ipynb'
    with open(out_nb, 'w') as f:
        nbf.write(nb, f)

    out_zip = base / f'zip_{label}.zip'
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(out_nb, arcname='annitia_submission.ipynb')
        if label == 'master':
            for sub, target in (('r2_artifacts', 'r2_artifacts'),
                                 ('stack_artifacts', 'stack_artifacts')):
                src = base / sub
                for fp in sorted(src.iterdir()):
                    if fp.is_file():
                        z.write(fp, arcname=f'{target}/{fp.name}')
            z.write(csv, arcname='annitia_submission.csv')
        else:
            for fp in sorted(models.iterdir()):
                if fp.is_file() and fp.suffix in ('.pkl',):
                    z.write(fp, arcname=fp.name)
            z.write(csv, arcname='annitia_submission.csv')
        if (base / 'train.csv').exists():
            z.write(base / 'train.csv', arcname='train.csv')
        if (base / 'val_test.csv').exists():
            z.write(base / 'val_test.csv', arcname='val_test.csv')
    print(f'wrote {out_zip}')
    with zipfile.ZipFile(out_zip, 'r') as z:
        for n in z.namelist():
            print(f'  {n:50s} {z.getinfo(n).file_size:>12,} bytes')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('label', choices=('r2', 'stack', 'master'))
    p.add_argument('--base', default='/Volumes/seagate/kirby/annitia')
    args = p.parse_args()
    build(args.label, Path(args.base))


if __name__ == '__main__':
    main()
