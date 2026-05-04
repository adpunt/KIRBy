#!/usr/bin/env python3
"""Build the Trustii submission package for the ANNITIA challenge.

Produces:
    <out_dir>/annitia_submission.ipynb     — self-contained notebook
    <out_dir>/annitia_submission.zip       — notebook + model pickles + CSV

The notebook:
- inlines every helper class so it has no kirby package dependency,
- loads pre-trained models from the bundled pickles,
- regenerates the submission CSV deterministically,
- documents the methodology and per-view results.

Usage:
    python tests/experiments/annitia/build_submission_zip.py \\
        --models-dir /Volumes/seagate/kirby/annitia/results \\
        --submission-csv /Volumes/seagate/kirby/annitia/annitia_submission.csv \\
        --hepatic-search /Volumes/seagate/kirby/annitia/annitia_hepatic_search.json \\
        --death-search /Volumes/seagate/kirby/annitia/annitia_death_search.json \\
        --out-dir /Volumes/seagate/kirby/annitia/
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
SRC_LOADER = HERE.parent.parent.parent / 'src' / 'kirby' / 'datasets' / 'annitia_maslg.py'
SRC_VIEWS = HERE / 'views.py'
SRC_SURVIVAL = HERE / 'survival.py'
SRC_SEARCH = HERE / 'search.py'


def read_module_body(path: Path) -> str:
    """Return module contents stripped of the leading docstring and any
    'from kirby...' imports (the loader pulls some shared constants from itself
    in the package layout — we inline the constants instead)."""
    text = path.read_text()
    # Strip the leading triple-quoted module docstring if present.
    if text.startswith('"""'):
        end = text.find('"""', 3)
        if end != -1:
            text = text[end + 3:].lstrip('\n')
    # Drop intra-package imports — we are inlining everything.
    # Handles single-line and multi-line `from kirby... import (...)` blocks.
    out_lines = []
    skipping_continuation = False
    for line in text.splitlines():
        if skipping_continuation:
            if ')' in line:
                skipping_continuation = False
            continue
        stripped = line.lstrip()
        is_intra_import = (
            stripped.startswith('from kirby.')
            or stripped.startswith('from experiments.')
            or stripped.startswith('import kirby.')
        )
        if is_intra_import:
            if '(' in line and ')' not in line:
                skipping_continuation = True
            continue
        out_lines.append(line)
    return '\n'.join(out_lines).strip() + '\n'


def make_notebook(models_dir: Path, hepatic_search: dict, death_search: dict) -> nbf.NotebookNode:
    cells = []
    nb = nbf.v4.new_notebook()

    # ---- Title + abstract ----
    cells.append(nbf.v4.new_markdown_cell(
        "# ANNITIA MASLD Risk Stratification — Submission\n\n"
        "**Trustii challenge:** "
        "[https://app.trustii.io/datasets/1551](https://app.trustii.io/datasets/1551)\n\n"
        "**Score (5-fold CV on train.csv):**\n\n"
        f"- C-index hepatic = **{hepatic_search['best']['mean_cindex']:.4f} "
        f"± {hepatic_search['best']['std_cindex']:.4f}** "
        f"(views = `{'+'.join(hepatic_search['best']['views'])}`, "
        f"model = `{hepatic_search['best']['model']}`)\n"
        f"- C-index death   = **{death_search['best']['mean_cindex']:.4f} "
        f"± {death_search['best']['std_cindex']:.4f}** "
        f"(views = `{'+'.join(death_search['best']['views'])}`, "
        f"model = `{death_search['best']['model']}`)\n"
        f"- Weighted score  = **"
        f"{0.7 * hepatic_search['best']['mean_cindex'] + 0.3 * death_search['best']['mean_cindex']:.4f}** "
        "(`0.7 × hepatic + 0.3 × death`)\n\n"
        "**Approach:** instead of using the wide-format CSV columns directly, we "
        "build six handcrafted *feature views* over each patient's longitudinal "
        "record (snapshot, slope, summary, delta, missingness, NIT-only), then run "
        "a hybrid search that enumerates every non-empty subset of views × "
        "{Cox PH, Coxnet, Random Survival Forest} and picks the winning "
        "(subset, model) by 5-fold stratified CV censored C-index. This is a "
        "survival-domain adaptation of the hybrid representation search pattern "
        "from the KIRBy framework (Punt et al., in preparation).\n\n"
        "**This notebook:** loads the pre-trained models from the bundled "
        "`*.pkl` files, applies the same view builders to `val_test.csv`, and "
        "writes the Trustii submission CSV. The full training procedure is "
        "documented in section 6; rerun it from scratch if you want to "
        "reproduce the model pickles."
    ))

    # ---- Methodology ----
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Methodology\n\n"
        "### 1.1 Pre-processing\n\n"
        "For every patient we extract:\n"
        "- a static-covariate vector (`gender`, `T2DM`, `Hypertension`, "
        "`Dyslipidaemia`, `bariatric_surgery`, `bariatric_surgery_age`, "
        "`baseline_age`, plus per-feature missingness flags),\n"
        "- a list of `(age, value)` observations per longitudinal variable, "
        "filtered to rows where both the age and the value are observed,\n"
        "- a per-endpoint survival tuple `(event, duration)` where "
        "`duration = age_at_event_or_last_visit − age_at_first_visit`.\n\n"
        "Patients with non-positive duration (event at baseline or no "
        "follow-up) are dropped from that endpoint. Patients with a blank "
        "`death` cell (~21% of the dataset) are dropped from the death "
        "endpoint only — treating \"status unknown\" as \"alive at last "
        "contact\" would bias the censoring distribution.\n\n"
        "### 1.2 Feature views\n\n"
        "Six deterministic transforms over the per-patient record:\n\n"
        "| View | What it captures | Dim |\n"
        "|------|------------------|----:|\n"
        "| `snapshot`    | static covariates + last observed value of each lab | 37 |\n"
        "| `slope`       | linear-fit slope of value vs age, per lab           | 12 |\n"
        "| `summary`     | mean / min / max / std per lab                      | 48 |\n"
        "| `delta`       | (last − first) per lab                              | 12 |\n"
        "| `missingness` | visit count, follow-up span, time-since-last-obs    | 25 |\n"
        "| `nit`         | snapshot+slope+delta restricted to FibroScan/FibroTest/Aixplorer | 18 |\n\n"
        "Missing slots are imputed with 0; explicit missingness flags are "
        "added where they carry signal. The `nit` view exists because the "
        "challenge requires the model to use at least one NIT.\n\n"
        "### 1.3 Survival downstreams\n\n"
        "Three estimators behind a uniform `.fit(X, y_struct) / .predict_risk(X)` API:\n\n"
        "- **`SurvivalCoxPH`** — `sksurv.linear_model.CoxPHSurvivalAnalysis` "
        "with α=1.0 ridge penalty and per-fold standardisation.\n"
        "- **`SurvivalCoxnet`** — `sksurv.linear_model.CoxnetSurvivalAnalysis` "
        "(elastic net, l1_ratio=0.9).\n"
        "- **`SurvivalRSF`** — `sksurv.ensemble.RandomSurvivalForest` "
        "(300 trees, sqrt features, min_samples_leaf=20).\n\n"
        "All three return scores where higher means worse prognosis, the "
        "convention `concordance_index_censored` expects.\n\n"
        "### 1.4 Hybrid search\n\n"
        "Enumerate every non-empty subset of the 6 views (63 subsets) × "
        "3 downstream models = **189 candidates per endpoint**. Score each "
        "by mean censored C-index over 5 stratified CV folds. Refit the "
        "winner on all valid patients."
    ))

    # ---- Setup / imports ----
    cells.append(nbf.v4.new_markdown_cell("## 2. Imports & paths"))
    cells.append(nbf.v4.new_code_cell(
        "import os, sys, pickle, json\n"
        "from itertools import combinations\n"
        "from pathlib import Path\n"
        "from typing import Dict, List, Optional, Tuple\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sklearn.model_selection import StratifiedKFold\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "\n"
        "from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis\n"
        "from sksurv.ensemble import RandomSurvivalForest\n"
        "from sksurv.metrics import concordance_index_censored\n"
        "\n"
        "# All paths assume the unzipped submission folder is the working dir.\n"
        "TRAIN_CSV = 'train.csv'                  # 1253-patient training data\n"
        "VAL_CSV   = 'val_test.csv'               # 423-patient validation data\n"
        "MODEL_HEPATIC = 'annitia_hepatic_model.pkl'\n"
        "MODEL_DEATH   = 'annitia_death_model.pkl'\n"
        "OUTPUT_CSV    = 'annitia_submission.csv'\n"
        "\n"
        "print('numpy', np.__version__, 'pandas', pd.__version__)"
    ))

    # ---- Loader ----
    cells.append(nbf.v4.new_markdown_cell("## 3. Data loader"))
    cells.append(nbf.v4.new_code_cell(read_module_body(SRC_LOADER)))

    # ---- Feature views ----
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Feature view builders\n\n"
        "Each builder is deterministic — the same transform runs on training "
        "and validation patients alike, with no fitting in between."
    ))
    cells.append(nbf.v4.new_code_cell(read_module_body(SRC_VIEWS)))

    # ---- Survival models ----
    cells.append(nbf.v4.new_markdown_cell("## 5. Survival downstream models"))
    cells.append(nbf.v4.new_code_cell(read_module_body(SRC_SURVIVAL)))

    # ---- Hybrid search ----
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Hybrid search\n\n"
        "The training procedure runs this search per endpoint (hepatic, "
        "death) over `train.csv`, then refits the winning (subset, model) "
        "and pickles it. The pickles are bundled with this notebook; the "
        "code below is for documentation and reproducibility — you do not "
        "need to rerun it to generate predictions."
    ))
    cells.append(nbf.v4.new_code_cell(read_module_body(SRC_SEARCH)))

    cells.append(nbf.v4.new_markdown_cell(
        "Reproducing the training pickles from scratch (≈4 min on the "
        "training data, slower under JupyterHub):\n\n"
        "```python\n"
        "data = load_annitia(data_dir='.', filename=TRAIN_CSV)\n"
        "views = build_views(data)\n"
        "for endpoint in ('hepatic', 'death'):\n"
        "    folds = get_cv_folds(data, endpoint=endpoint, n_folds=5, seed=42)\n"
        "    result = search_survival_hybrid(\n"
        "        views=views,\n"
        "        y_struct=data[endpoint]['y_struct'],\n"
        "        valid_mask=data[endpoint]['valid_mask'],\n"
        "        folds=folds,\n"
        "        endpoint=endpoint,\n"
        "        model_names=('cox', 'coxnet', 'rsf'),\n"
        "        verbose=False,\n"
        "    )\n"
        "    with open(f'annitia_{endpoint}_model.pkl', 'wb') as f:\n"
        "        pickle.dump({\n"
        "            'model': result.final_model,\n"
        "            'view_names': result.final_view_names,\n"
        "            'model_name': result.final_model_name,\n"
        "            'endpoint': endpoint,\n"
        "        }, f)\n"
        "```"
    ))

    # ---- Inference: load pickles ----
    cells.append(nbf.v4.new_markdown_cell(
        "## 7. Load pre-trained models\n\n"
        "The pickle files were originally produced inside the KIRBy package "
        "namespace. We alias `experiments.annitia.survival` to this notebook's "
        "globals before unpickling so the survival classes resolve correctly."
    ))
    cells.append(nbf.v4.new_code_cell(
        "import types\n"
        "_alias_root = types.ModuleType('experiments')\n"
        "_alias_pkg  = types.ModuleType('experiments.annitia')\n"
        "_alias_surv = types.ModuleType('experiments.annitia.survival')\n"
        "for _name in ('SurvivalCoxPH', 'SurvivalCoxnet', 'SurvivalRSF',\n"
        "              'SURVIVAL_MODELS', 'concordance'):\n"
        "    setattr(_alias_surv, _name, globals()[_name])\n"
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
        "          f'model={p[\"model_name\"]}')"
    ))

    # ---- Inference: predict on validation ----
    cells.append(nbf.v4.new_markdown_cell("## 8. Generate predictions on `val_test.csv`"))
    cells.append(nbf.v4.new_code_cell(
        "val_data = load_annitia(data_dir='.', filename=VAL_CSV, has_outcomes=False)\n"
        "assert val_data['trustii_id'] is not None, 'val_test.csv must contain trustii_id column'\n"
        "print(f'{len(val_data[\"patient_ids\"])} validation patients')\n"
        "\n"
        "val_views = build_views(val_data)\n"
        "for name, X_fnames in val_views.items():\n"
        "    print(f'  {name:13s} shape={X_fnames[0].shape}')"
    ))

    cells.append(nbf.v4.new_code_cell(
        "def predict_risk(payload, views):\n"
        "    \"\"\"Concat the selected view subset and call .predict_risk()\"\"\"\n"
        "    X = np.concatenate([views[v][0] for v in payload['view_names']], axis=1)\n"
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

    # ---- Results ----
    cells.append(nbf.v4.new_markdown_cell(
        "## 9. Cross-validation results\n\n"
        "Below is the per-view single-view C-index for each endpoint, plus "
        "the top-3 multi-view configurations selected by the search. Numbers "
        "are mean ± std across 5 stratified folds on `train.csv`.\n\n"
        + _build_results_section(hepatic_search, death_search)
    ))

    # ---- Limitations ----
    cells.append(nbf.v4.new_markdown_cell(
        "## 10. Limitations & caveats\n\n"
        "- **Few hepatic events (47 in 1253 patients).** Differences in "
        "C-index between candidate views for the hepatic endpoint are at "
        "the edge of the CV noise floor.\n"
        "- **~21% of `death` cells are blank.** Dropping those patients "
        "shrinks the death-endpoint training set to 966; if missingness is "
        "informative, the model's death predictions are biased.\n"
        "- **Time axis is patient age.** We use `age_at_event − age_v1` as "
        "the duration. Pre-baseline events are excluded by construction.\n"
        "- **`missingness` view encodes follow-up span.** It dominates "
        "single-view C-index for the death endpoint (0.944) — most likely "
        "because the synthetic data preserves the correlation between "
        "censoring time and event time. The model is valid on the held-out "
        "validation set if its data-generating process matches training, "
        "but the CV figure is an optimistic estimate of generalisation to "
        "a new cohort.\n"
        "- **Search-on-search overfit.** The hybrid search picks the best "
        "of 189 candidates per endpoint by CV. Held-out leaderboard "
        "performance is the trustworthy estimate; CV is an upper bound.\n"
    ))

    nb.cells = cells
    nb.metadata['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    }
    return nb


def _build_results_section(hepatic: dict, death: dict) -> str:
    def _per_view_singles(search):
        rows = sorted(search['per_condition'], key=lambda r: -r['mean_cindex'])
        singles = [r for r in rows if len(r['views']) == 1]
        by_view = {}
        for r in singles:
            v = r['views'][0]
            if v not in by_view or r['mean_cindex'] > by_view[v]['mean_cindex']:
                by_view[v] = r
        return by_view

    hep_singles = _per_view_singles(hepatic)
    death_singles = _per_view_singles(death)
    view_order = ['snapshot', 'slope', 'summary', 'delta', 'missingness', 'nit']

    lines = ['### Per-view single-view C-index (best model per cell)\n',
             '| View | Hepatic | Death |',
             '|------|--------:|------:|']
    for v in view_order:
        h = hep_singles.get(v)
        d = death_singles.get(v)
        h_txt = (f"{h['mean_cindex']:.4f} ({h['model']})" if h else '—')
        d_txt = (f"{d['mean_cindex']:.4f} ({d['model']})" if d else '—')
        lines.append(f'| `{v}` | {h_txt} | {d_txt} |')

    lines.append('\n### Top-3 hybrid configurations\n')
    for label, search in [('Hepatic', hepatic), ('Death', death)]:
        rows = sorted(search['per_condition'], key=lambda r: -r['mean_cindex'])
        lines.append(f'\n**{label} endpoint:**\n')
        lines.append('| Rank | Views | Model | Dim | C-index |')
        lines.append('|-----:|-------|-------|----:|--------:|')
        for i, r in enumerate(rows[:3], 1):
            v = '+'.join(r['views'])
            lines.append(f'| {i} | `{v}` | `{r["model"]}` | {r["n_features"]} | {r["mean_cindex"]:.4f} ± {r["std_cindex"]:.4f} |')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models-dir', required=True,
                   help='Directory with annitia_{hepatic,death}_model.pkl.')
    p.add_argument('--submission-csv', required=True,
                   help='Existing submission CSV (will be bundled and re-written by the notebook).')
    p.add_argument('--hepatic-search', required=True,
                   help='annitia_hepatic_search.json from the SLURM run.')
    p.add_argument('--death-search', required=True,
                   help='annitia_death_search.json from the SLURM run.')
    p.add_argument('--train-csv', default='/Volumes/seagate/kirby/annitia/train.csv',
                   help='Training CSV (bundled into zip for reproducibility).')
    p.add_argument('--val-csv', default='/Volumes/seagate/kirby/annitia/val_test.csv',
                   help='Validation CSV (bundled into zip).')
    p.add_argument('--out-dir', required=True,
                   help='Output directory for ipynb and zip.')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hepatic_search = json.load(open(args.hepatic_search))
    death_search = json.load(open(args.death_search))

    nb = make_notebook(Path(args.models_dir), hepatic_search, death_search)
    nb_path = out_dir / 'annitia_submission.ipynb'
    with open(nb_path, 'w') as f:
        nbf.write(nb, f)
    print(f'wrote notebook: {nb_path}')

    # Build zip with notebook + pickles + submission CSV + train + val
    zip_path = out_dir / 'annitia_submission.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(nb_path, arcname='annitia_submission.ipynb')
        z.write(Path(args.models_dir) / 'annitia_hepatic_model.pkl',
                arcname='annitia_hepatic_model.pkl')
        z.write(Path(args.models_dir) / 'annitia_death_model.pkl',
                arcname='annitia_death_model.pkl')
        z.write(args.submission_csv, arcname='annitia_submission.csv')
        if Path(args.train_csv).exists():
            z.write(args.train_csv, arcname='train.csv')
        if Path(args.val_csv).exists():
            z.write(args.val_csv, arcname='val_test.csv')
    print(f'wrote zip:      {zip_path}')
    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            info = z.getinfo(name)
            print(f'  {name:40s} {info.file_size:>12,} bytes')


if __name__ == '__main__':
    main()
