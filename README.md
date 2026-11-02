# KIRBy

## What it is

KIRBy builds molecular representations and combines features from several of them into one hybrid representation for property prediction.
The NoiseInject paper (Punt, Hanser, Werner and Morris, submitted to the Journal of Cheminformatics) uses one part of it.
That part is the runner for the three assay datasets, `tests/alternative_data_noise_robustness.py`.
The runner trains models on logD, Caco-2 efflux ratio and hERG K<sub>i</sub>, adds label noise with [NoiseInject](https://github.com/adpunt/noise_inject), and records how accuracy falls as the noise grows.
It uses KIRBy's code for six representations (ECFP4, PDV, Sort & Slice, MHG-GNN, Avalon, ChemBERTa) and its noise helpers in `src/kirby/noise_spec.py` and `src/kirby/noise_robustness.py`.
The hybrid representation search is not used in the paper.

The QM9 half of the paper was run by a different repository, [qsar_qm_models](https://github.com/adpunt/qsar_qm_models).
The assay runner also depends on that repository, as described below.

## Install

The runner needs three checkouts side by side and one conda environment.

```bash
mkdir study && cd study
git clone https://github.com/adpunt/KIRBy.git
git clone https://github.com/adpunt/qsar_qm_models.git
git clone https://github.com/adpunt/noise_inject.git NoiseInject
```

The environment is defined in `qsar_qm_models/env.yml` (Python 3.10.16, PyTorch 2.5.1 CPU build, RDKit from conda-forge).
Build it with that repository's `setup.sh`, not with `conda env create` alone.
`setup.sh` also installs NoiseInject and KIRBy into the environment from their checkouts, as editable installs.
See the qsar_qm_models README for the build steps.

The runner reads four things from the qsar_qm_models checkout:

- `models/model_defaults.py`, the model settings shared with the QM9 runs
- `results/master_tuned_hyperparameters_lab.json`, the tuned settings for the neural models
- `noise_conditions.json`, the list of noise conditions and levels
- `scripts/loss_functions.py`, `scripts/utils.py` and `scripts/uncertainty_decomposition.py`, helper code

It looks for that checkout in this order:

1. the directory named by the environment variable `QSAR_QM_MODELS_ROOT`
2. a directory called `qsar_qm_models` next to the KIRBy checkout
3. two fixed paths on the Oxford ARC cluster, which do not exist elsewhere

If none is found, the runner stops with an error that lists every place it looked.
With the layout above, nothing needs setting. Otherwise:

```bash
export QSAR_QM_MODELS_ROOT=/path/to/qsar_qm_models
```

### Data

All three datasets are read from `tests/data_cache/`.

- **logD and Caco-2** come from one file, `openadmet_train.csv`, the training set of the OpenADMET ExpansionRx challenge.
  If the file is missing, the runner downloads it from
  `https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data` on first use.
- **hERG K<sub>i</sub>** is `chembl_herg_ki.csv`, 1,415 compounds for target CHEMBL240 extracted from ChEMBL release 36.
  `chembl_herg_ki.provenance.json` beside it records the release, the date and an MD5 checksum.
  The runner will not fetch hERG from ChEMBL on its own, because ChEMBL changes between releases.
  To fetch it anyway, set `KIRBY_ALLOW_CHEMBL_FETCH=1`.

### Representations that need extra files

- **MHG-GNN** needs IBM's model code and weights, which are not installed by pip.
  Clone `https://huggingface.co/ibm-research/materials.mhg-ged` to `~/repos/materials`, and place
  `mhggnn_pretrained_model_0724_2023.pickle` at `~/repos/materials.mhg-ged/`.
  The other places searched are listed in `create_mhg_gnn` in `src/kirby/representations/molecular.py`.
- **ChemBERTa** downloads `DeepChem/ChemBERTa-77M-MTR` from Hugging Face on first use.

## Quick example

One dataset, one model, one representation and one noise condition: a random forest on ECFP4 for hERG K<sub>i</sub>, under Gaussian noise.

```bash
cd KIRBy/tests
python alternative_data_noise_robustness.py \
    --datasets herg_ki --models RF --reps ECFP4 --conditions gaussian \
    --results-root ../results/quick_example
```

This fits 35 models: 5 scaffold folds times 7 noise levels.
It took 1 to 3 minutes on a MacBook (2026-10-23), in the environment from `qsar_qm_models/env.yml`.
It writes three files to `results/quick_example/herg/`:

- `all_results.csv`, one row per fold and noise level, with R², RMSE, MAE and Spearman on the test fold
- `summary.csv`, one row per model, representation and noise condition, with clean R² and the area under the retention curve (`auc_norm`)
- `all_results_partial.csv`, a checkpoint written after each fold

On that run, clean R² was 0.511 (standard deviation 0.063 over the 5 folds) and `auc_norm` was 0.888.

## How to reproduce the paper's assay results

The paper's runs were one cluster job per model, representation and dataset.
The job scripts are written by `qsar_qm_models/slurm_scripts_validation_rerun/generate_scripts.py`.
That file holds the lists of models, representations and noise conditions, so they are not repeated here.
Each job runs this command from `KIRBy/tests`:

```bash
python -u alternative_data_noise_robustness.py \
    --datasets <logd | caco2 | herg_ki> \
    --models <model> \
    --reps <representation> \
    --conditions <noise conditions> \
    --results-root ../results/validation_rerun/<model>_<representation>_<dataset>
```

Gaussian process jobs add `--gp-kernel rbf --gp-reps <representation>`, or `--gp-kernel tanimoto` for the Tanimoto kernel on ECFP4.
The noise levels are not passed on the command line.
They are read from `noise_conditions.json` in qsar_qm_models.

The uncertainty results come from a second set of jobs, written by `qsar_qm_models/slurm_scripts_uncertainty_rerun/generate_scripts.py`.
Those jobs add `--unc-conditions all --oof-folds <n>` and write to `results/uncertainty_rerun/`.
For each model and representation they also write a file of per-molecule uncertainties, `<model>_<representation>_uncertainty_values.csv`.

**Runtime.** The job generator estimates about 3,900 job-hours for the three Gaussian-type conditions on all models, representations and datasets.
That is its planning estimate before a 1.5× safety margin, not a measured total.
The deep run and the censoring run, also written by the generator, add to it.

**Where the results are.** The result files are not in this repository.
The paper's runs are on the Oxford ARC cluster under `results/validation_rerun/` and `results/uncertainty_rerun/` in the KIRBy checkout there.

**How the paper's figures read them.** `qsar_qm_models/scripts/run_paper_analysis.py` is given the KIRBy result folders with `--validation-dir`.
It passes them to `load_assay_accuracy` in `qsar_qm_models/scripts/figlib_load.py`, which reads every `all_results.csv` beneath them.
The uncertainty results are read from `results/uncertainty_rerun/_merged/`, written by `qsar_qm_models/slurm_scripts_uncertainty_rerun/merge_results.py`.
`qsar_qm_models/scripts/regenerate_figures.sh` runs the merge steps and then submits the analysis that writes every figure and table.
The assay results feed these parts of the paper:

- the figure of `auc_norm` for the 13 base models with ECFP4 on logD, Caco-2 and hERG K<sub>i</sub>
- the figure of each base model's rank by `auc_norm` on every dataset (rank transfer)
- the table of `auc_norm` under Gaussian noise and its change under each other condition on the assay datasets
- panels b to d (logD, Caco-2, hERG K<sub>i</sub>) of the variance decomposition figure
- the assay rows of the table of how mean predicted uncertainty rises with noise, from the uncertainty jobs

## Tests

```bash
python -m pytest -q
```

`pytest.ini` limits collection to `tests/`, and skips `tests/archive/`, which holds retired scripts.

## Citation

If you use the assay runner, please cite the paper and the software:

- Punt A, Hanser T, Werner S, Morris G. NoiseInject: A Benchmarking Framework to Study Noise Robustness in QSAR Models. Submitted to the Journal of Cheminformatics, 2026.
- NoiseInject, archived at https://doi.org/10.5281/zenodo.20532710
- KIRBy, archived at DOI: TODO, to be filled in after the Zenodo release

`CITATION.cff` holds the same information in machine-readable form.

## Licence

TODO: the author chooses the licence. A `LICENSE` file goes at the repository root.
