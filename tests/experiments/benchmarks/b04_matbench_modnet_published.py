#!/usr/bin/env python3
"""Matbench v0.1 as published: the leaderboard's MODNet entry, v0.1.12.

Reruns the leaderboard entry matbench_v0.1_modnet_v0.1.12 on Matbench's own 5
test folds and compares the MAE per fold with the published MAE per fold.

Where the published scores come from. The leaderboard's info.json says they
were loaded from github.com/ml-evs/modnet-matbench release v0.4.0, archived
at Zenodo 10.5281/zenodo.5109941 (record 6368562, file
modnet-matbench-v0.4.0.zip, md5 d7c2b6e8eb35f09453824f6efc70911b, downloaded
2026-10-26 to /Volumes/seagate/benchmarks_protocol/matbench/
modnet_matbench_v0.4.0/). The GitHub tag no longer exists. That archive's
run_benchmark.py is what this file follows:
  features   MODNet's DeBreuck2020Featurizer(fast_oxid=True) on structure
             tasks, CompositionOnlyFeaturizer() on composition tasks
  folds      modnet.matbench.benchmark.matbench_kfold_splits: KFold(5,
             shuffle=True, random_state=18012019) over matminer's row order,
             which is Matbench's own split
  selection  per training fold, MODData.feature_selection(n=-1,
             use_precomputed_cross_nmi=True): every feature ranked by
             relevance to the target minus redundancy with features already
             ranked, both by normalised mutual information; the redundancy
             between features is read from MODNet's file precomputed on
             Materials Project, not from the fold
  model      EnsembleMODNetModel, n_models=5
  tuning     fit_preset with nested=5: MODNet's built-in grid of presets
             (layer sizes, number of features, learning rate, ...), each
             scored by 5-fold CV inside the training fold; hp_strategy is the
             default 'fit_preset' on every task
The leaderboard text says a genetic search was used on the four larger tasks
(perovskites among them). The archived code does not call it. The code is
followed here.

Version. The published run pinned modnet==0.1.12. That version is what runs
here, in its own environment: python 3.8, modnet 0.1.12, matminer 0.6.5,
pymatgen 2020.4.29, scikit-learn 0.23.2, tensorflow 2.4.4. The archive pins
only modnet; modnet 0.1.12 asks for pymatgen>=2020,<2020.9 and matminer>=0.6.2.
matminer 0.6.5 is the newest release that allows that pymatgen (0.7.0 needs
pymatgen>=2022). pymatgen 2020.4.29 is the version matminer 0.6.5 was tested
with. matminer 0.6.2 does not load its element data under these pymatgens:
pymatgen lists 118 elements, matminer's tables 103, and 0.6.2 raises where
0.6.5 fills elements 104-118 with missing values.
Which versions the published run actually had is not recorded.
MODNet's own featurizer computes 0 features under current matminer (0.9).
train_fold and matbench_kfold_splits are called from modnet 0.1.12 itself;
their arguments were read from its source (2026-10-26).

modnet_published_fold() is the protocol on any feature matrix, so a hybrid's
features can be scored the same way as MODNet's own features. For features
that are not MODNet's own, the redundancy is computed on the fold
(use_precomputed_cross_nmi=False): the precomputed file only covers MODNet's
feature names.

--fast is MODNet's own debug mode (fewer presets and epochs) with no nested
tuning, as in run_benchmark.py --fast. Use it locally to check the file runs.

Environment: /Volumes/seagate/matbench/env_modnet_0112, built with
    micromamba create -p <env> -c conda-forge python=3.8 pymatgen=2020.4.29 \
        scikit-learn=0.23 pandas=1.1 numpy=1.19
    <env>/bin/pip install tensorflow==2.4.4 tensorflow-probability==0.12.2 \
        modnet==0.1.12 "ruamel.yaml<0.18"
    <env>/bin/pip install --no-deps matminer==0.6.5 "monty<2022"
Featurised data is cached under
/Volumes/seagate/benchmarks_protocol/matbench/modnet_features/.

Output: /Volumes/seagate/benchmarks_protocol/results/b04_matbench_modnet.json,
rewritten after each fold.

    /Volumes/seagate/matbench/env_modnet_0112/bin/python \
        tests/experiments/benchmarks/b04_matbench_modnet_published.py \
        --tasks matbench_jdft2d --fast
"""
import argparse
import gzip
import json
import os
import time
from pathlib import Path

import numpy as np

# Every file lives under ROOT: /Volumes/seagate on the laptop; on the server
# set KIRBY_MATBENCH_ROOT to a folder on /data.
ROOT = Path(os.environ.get('KIRBY_MATBENCH_ROOT', '/Volumes/seagate'))
PROTOCOL_DIR = ROOT / 'benchmarks_protocol' / 'matbench'
PUBLISHED = PROTOCOL_DIR / 'modnet_v0.1.12' / 'results.json.gz'
FEATURES = PROTOCOL_DIR / 'modnet_features'
OUT = ROOT / 'benchmarks_protocol' / 'results' / 'b04_matbench_modnet.json'
TASKS = ['matbench_jdft2d', 'matbench_phonons', 'matbench_expt_gap',
         'matbench_dielectric', 'matbench_log_gvrh', 'matbench_perovskites']
TARGET = 'y'

# run_benchmark.py's best_settings. With tuning on, matbench_benchmark only
# reads n_feat and num_neurons from it (filled in below, as it does).
FIT_SETTINGS = {'increase_bs': False, 'lr': 0.005, 'epochs': 50, 'act': 'elu',
                'out_act': 'relu', 'batch_size': 32, 'loss': 'mae',
                'xscale': 'minmax'}


def published_mae(task):
    d = json.load(gzip.open(PUBLISHED))
    r = d['tasks'][task]['results']
    return [float(r[f'fold_{k}']['scores']['mae']) for k in range(5)]


def featurized(task, n_jobs):
    """MODData for the whole task with MODNet's own features, cached."""
    from modnet.preprocessing import MODData
    path = FEATURES / f'{task}_moddata.pkl.gz'
    if path.exists():
        return MODData.load(str(path))
    from matminer.datasets import load_dataset
    from modnet.featurizers.presets import (
        CompositionOnlyFeaturizer, DeBreuck2020Featurizer)
    from pymatgen.core import Composition
    df = load_dataset(task)
    target = [c for c in df.columns if c not in ('id', 'structure', 'composition')]
    if 'structure' in df.columns:
        materials, featurizer = df['structure'], DeBreuck2020Featurizer(fast_oxid=True)
    else:
        materials = df['composition'].map(Composition)
        featurizer = CompositionOnlyFeaturizer()
    data = MODData(materials=materials.tolist(),
                   targets=df[target].values, target_names=[TARGET],
                   featurizer=featurizer)
    data.featurize(n_jobs=n_jobs)
    # With a matminer it was not written for, the preset warns, computes
    # nothing and returns 0 features (seen with modnet 0.4.5 + matminer 0.9).
    if data.df_featurized.shape[1] == 0:
        raise RuntimeError(f'{task}: MODNet featurizer returned 0 features; '
                           f'it needs matminer 0.6.2 (see module docstring)')
    FEATURES.mkdir(parents=True, exist_ok=True)
    data.save(str(path))
    return data


def _train_fold(k, train_data, test_data, n_jobs, fast):
    from modnet.matbench.benchmark import train_fold
    from modnet.models import EnsembleMODNetModel
    fit_settings = dict(FIT_SETTINGS)
    fit_settings['n_feat'] = len(train_data.df_featurized.columns)
    fit_settings['num_neurons'] = [[4], [4], [4], [4]]
    ga_settings = {'size_pop': 20, 'num_generations': 10,
                   'early_stopping': 4, 'refit': False}
    res = train_fold((k, (train_data, test_data)), [[[TARGET]]], {TARGET: 1},
                     fit_settings, ga_settings,
                     model_type=EnsembleMODNetModel, hp_optimization=True,
                     hp_strategy='fit_preset', classification=False,
                     fast=fast, nested=0 if fast else 5, n_jobs=n_jobs,
                     n_models=5)
    if res['predictions'] is None:
        raise RuntimeError(f'fold {k}: MODNet train_fold failed (see log)')
    return np.asarray(res['predictions'].values).ravel()


def modnet_published_fold(X_tr, y_tr, X_te, feature_names=None, n_jobs=4,
                          fast=False, k=0):
    """The published protocol on any feature matrix. Returns test predictions."""
    import pandas as pd
    from modnet.preprocessing import MODData
    names = feature_names or [f'f{i}' for i in range(X_tr.shape[1])]
    tr_ids = [f'tr{i}' for i in range(len(X_tr))]
    te_ids = [f'te{i}' for i in range(len(X_te))]
    train = MODData(materials=[None] * len(tr_ids),
                    targets=np.asarray(y_tr).reshape(-1, 1),
                    target_names=[TARGET], structure_ids=tr_ids,
                    df_featurized=pd.DataFrame(X_tr, index=tr_ids, columns=names))
    test = MODData(materials=[None] * len(te_ids),
                   targets=np.zeros((len(te_ids), 1)), target_names=[TARGET],
                   structure_ids=te_ids,
                   df_featurized=pd.DataFrame(X_te, index=te_ids, columns=names))
    train.feature_selection(n=-1, use_precomputed_cross_nmi=False, n_jobs=n_jobs)
    return _train_fold(k, train, test, n_jobs, fast)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default=','.join(TASKS))
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--n-jobs', type=int, default=4)
    ap.add_argument('--fast', action='store_true',
                    help="MODNet's debug mode: fewer presets and epochs, no "
                         "nested tuning. For checking the file runs.")
    a = ap.parse_args()
    os.environ.setdefault('MATMINER_DATA', str(ROOT / 'matbench' / 'data'))
    from modnet.matbench.benchmark import matbench_kfold_splits

    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(OUT.read_text()) if OUT.exists() else []
    folds = [int(f) for f in a.folds.split(',')]
    for task in a.tasks.split(','):
        t0 = time.time()
        data = featurized(task, a.n_jobs)
        print(f'{task}: {data.df_featurized.shape[1]} MODNet features, '
              f'{time.time() - t0:.0f} s', flush=True)
        pub = published_mae(task)
        splits = list(matbench_kfold_splits(data))
        for k in folds:
            tr, te = splits[k]
            train_data, test_data = data.split((tr, te))
            t0 = time.time()
            train_data.feature_selection(n=-1, use_precomputed_cross_nmi=True,
                                         n_jobs=a.n_jobs)
            pred = _train_fold(k, train_data, test_data, a.n_jobs, a.fast)
            y_te = test_data.df_targets[TARGET].to_numpy()
            mae = float(np.mean(np.abs(pred - y_te)))
            row = dict(task=task, fold=k, fast=a.fast, published_mae=pub[k],
                       rerun_mae=mae, n_train=len(tr), n_test=len(te),
                       n_features=int(data.df_featurized.shape[1]),
                       seconds=round(time.time() - t0, 1))
            results = [r for r in results
                       if not (r['task'] == task and r['fold'] == k
                               and r['fast'] == a.fast)] + [row]
            OUT.write_text(json.dumps(results, indent=1))
            print(f'  fold {k}: published {pub[k]:.4f}, rerun {mae:.4f} MAE '
                  f'({len(te)} test compounds, {row["seconds"]:.0f} s'
                  f'{", --fast" if a.fast else ""})', flush=True)


if __name__ == '__main__':
    main()
