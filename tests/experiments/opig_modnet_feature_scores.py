#!/usr/bin/env python3
"""MODNet feature scores experiment (Matbench v0.1, follows b04)

Before MODNet trains, it ranks features with two scores, both normalised
mutual information:
  target score   how much a feature tells about the target
  pair score     how much two features tell about each other; a feature that
                 repeats one already ranked is pushed down
(modnet 0.1.12, preprocessing.get_features_dyn: target score / (pair score
with the most similar feature already ranked ** p + c), with p and c set for
scores between 0 and 1.)

Questions:
  1. Does MODNet's error change when both scores use a different measure?
  2. With a hybrid's features as input, MODNet computes its pair scores on
     those features. How does its error compare with its own features under
     the same measure?

Measures, each used for both scores, each giving 0 (unrelated) to 1:
  mutual_information    MODNet's own, computed by MODNet on the training fold
  pearson               size of the Pearson correlation (sign ignored)
  spearman              size of the Spearman correlation (sign ignored)
  distance_correlation  Szekely, Rizzo & Bakirov 2007; zero only when the two
                        are unrelated. Computed with the dcor package's
                        sorting method (method AVL, Huo & Szekely 2016), on
                        the same compounds as the other measures. Before the
                        first distance correlation run, its numbers are
                        checked against the direct formula (the one in
                        f30_scorers.s_dcor) on 300 compounds; the run stops if
                        they differ.
The published setting, mutual information with the pair scores read from
MODNet's file computed on Materials Project, is also run on MODNet's own
features ('mutual_information_file'). It is b04's setting; b04 has the
published scores to check it against.

Inputs:
  modnet   MODNet's own features, the cache b04 writes
  hybrid   c02's hybrids, built into features per fold by
           opig_modnet_hybrid_features.py in KIRBy's environment. --hybrids
           names which, as <model>_hybrid<i> (e.g. svm_hybrid1); each is a
           separate input. Files are read from --hybrid-dir (default that
           file's output folder). Their train and test indexes must equal
           Matbench's fold in matminer's row order (checked).

Swept with every measure, --low-information:
  drop   MODNet's own step before ranking: a feature whose mutual information
         with itself is below 0.2 is dropped (get_cross_nmi drop_thr=0.2,
         MODNet's default). For mutual_information this is MODNet unchanged;
         for the other measures the same rule is run first (sklearn
         mutual_info_regression, 3 neighbours, as MODNet's compute_mi, with a
         fixed seed where MODNet's is unseeded).
  keep   no feature dropped by that rule. For mutual_information,
         get_cross_nmi(drop_thr=0) is passed to MODNet's feature_selection
         (its cross_nmi argument); MODNet's code is not changed.
mutual_information_file is run once, recorded as low_information 'file'.

Kept the same as b04 in every run: Matbench's 5 folds, MODNet's network, its
settings search (fit_preset, 5 inner folds, 5-network ensemble), all features
ranked (n=-1). As in MODNet, features with one value on the whole training
fold are dropped, and at most 6000 training compounds are used for the
scores (MODNet's own n_samples).

Recorded per task, fold, input, measure and low-information setting: MAE on the test fold, number of
features ranked, the 100 top-ranked feature names (for comparing measures),
and seconds spent scoring and training.

Runs in b04's environment (/Volumes/seagate/matbench/env_modnet_0112), plus
dcor and the numba it needs, installed without other dependencies so
numpy stays at 1.19 (numba 0.55.2 is the version in the laptop copy):
    <env>/bin/pip install --no-deps dcor==0.6 numba==0.55.2 llvmlite==0.38.1
MODNet is server only: do not run this file on the laptop.

--check-dcor only checks distance correlation and estimates its time, then
stops: the direct formula against dcor on 300 compounds of fold 0, then 210
feature pairs timed on fold 0's scoring compounds, scaled to every pair, then
210 pairs timed on 6000 made-up compounds (the most any task scores on).

    /Volumes/seagate/matbench/env_modnet_0112/bin/python \\
        tests/experiments/opig_modnet_feature_scores.py --check-dcor \\
        --tasks matbench_jdft2d
    /Volumes/seagate/matbench/env_modnet_0112/bin/python \\
        tests/experiments/opig_modnet_feature_scores.py \\
        --tasks matbench_jdft2d --inputs modnet

Paths are under b04's ROOT (KIRBY_MATBENCH_ROOT, default /Volumes/seagate).
Output: <ROOT>/benchmarks_protocol/results/opig_modnet_feature_scores.json,
rewritten after each run; runs already in it are skipped.
"""
import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

# dcor needs numba's compiler. The server shell had NUMBA_DISABLE_JIT=1
# (2026-11-01), which makes dcor fail on import. Set before numba loads; an
# environment variable also overrides a .numba_config.yaml file.
os.environ['NUMBA_DISABLE_JIT'] = '0'

import numpy as np

_B04 = Path(__file__).resolve().parent / 'benchmarks' / 'b04_matbench_modnet_published.py'
_spec = importlib.util.spec_from_file_location('b04_matbench_modnet_published', _B04)
b04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b04)

OUT = b04.ROOT / 'benchmarks_protocol' / 'results' / 'opig_modnet_feature_scores.json'
MEASURES = ['mutual_information', 'pearson', 'spearman', 'distance_correlation']
N_SAMPLES = 6000   # MODNet's feature_selection n_samples
TOP = 100


def _prepare(df):
    """MODNet's handling before scoring (preprocessing.nmi_target): drop
    features with one value; if any value is missing, scale each feature to
    [-0.5, 0.5] and set missing values to -1."""
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    spread = df.max(axis=0) - df.min(axis=0)
    df = df.loc[:, spread[spread > 0].index]
    if df.isna().any().any():
        x = MinMaxScaler(feature_range=(-0.5, 0.5)).fit_transform(df.values)
        df = pd.DataFrame(np.nan_to_num(x, nan=-1), index=df.index,
                          columns=df.columns)
    return df


def _abs_corr(X, y):
    """Size of the Pearson correlation: every feature pair, and each feature
    with the target."""
    Z = (X - X.mean(0)) / X.std(0)
    yz = (y - y.mean()) / y.std()
    return np.abs(Z.T @ Z) / len(Z), np.abs(Z.T @ yz) / len(Z)


def _centred_distances(X):
    """Per feature: distances between every pair of compounds, double
    centred, flattened. (k features, m compounds) -> (k, m*m)."""
    D = np.abs(X.T[:, :, None] - X.T[:, None, :])
    D = (D - D.mean(2, keepdims=True) - D.mean(1, keepdims=True)
         + D.mean((1, 2), keepdims=True))
    return D.reshape(X.shape[1], -1)


def _dcor(X, y, chunk=64):
    """Distance correlation: every feature pair, and each feature with the
    target. The same formula as f30_scorers.s_dcor, for all pairs at once."""
    Xy = np.column_stack([X, y])
    F = Xy.shape[1]
    G = np.zeros((F, F))
    blocks = [(a, min(a + chunk, F)) for a in range(0, F, chunk)]
    for i, (a, b) in enumerate(blocks):
        A = _centred_distances(Xy[:, a:b])
        for c, d in blocks[i:]:
            B = A if c == a else _centred_distances(Xy[:, c:d])
            G[a:b, c:d] = A @ B.T / A.shape[1]
            G[c:d, a:b] = G[a:b, c:d].T
    v = np.sqrt(np.maximum(np.diag(G), 0))
    with np.errstate(divide='ignore', invalid='ignore'):
        R = np.sqrt(np.maximum(G, 0) / np.outer(v, v))
    R = np.nan_to_num(R, nan=0.0)
    return R[:-1, :-1], R[:-1, -1]


def _dcor_fast(X, y, batch=5000):
    """Distance correlation with dcor's sorting method: every feature pair,
    and each feature with the target."""
    import numba
    if numba.config.DISABLE_JIT:
        raise RuntimeError('numba compiling is off (NUMBA_DISABLE_JIT or '
                           '.numba_config.yaml); dcor cannot run')
    import dcor
    Xy = np.ascontiguousarray(np.column_stack([X, y]).T, dtype=float)
    i, j = np.triu_indices(len(Xy), k=1)
    R = np.eye(len(Xy))
    for s in range(0, len(i), batch):
        a, b = i[s:s + batch], j[s:s + batch]
        R[a, b] = R[b, a] = dcor.rowwise(
            dcor.distance_correlation, Xy[a], Xy[b],
            method=dcor.DistanceCovarianceMethod.AVL)
    R = np.nan_to_num(R, nan=0.0)
    return R[:-1, :-1], R[:-1, -1]


def _scoring_frame(train):
    """The training compounds MODNet scores on: at most N_SAMPLES, drawn
    as MODNet draws them for its pair scores."""
    df = train.df_featurized
    y = train.df_targets[b04.TARGET]
    if len(df) > N_SAMPLES:
        df = df.sample(n=N_SAMPLES, random_state=12)
        y = y.loc[df.index]
    return df, y


_DCOR_CHECKED = False


def check_dcor(df, y, n=300, n_feat=12, seed=0):
    """dcor's sorting method against the direct formula. Raises if they
    differ by more than 1e-6."""
    rs = np.random.RandomState(seed)
    rows = rs.choice(len(df), min(n, len(df)), replace=False)
    sub = _prepare(df.iloc[rows])
    cols = rs.choice(sub.shape[1], min(n_feat, sub.shape[1]), replace=False)
    X, yv = sub.iloc[:, cols].to_numpy(float), y.iloc[rows].to_numpy(float)
    exact = _dcor(X, yv)
    fast = _dcor_fast(X, yv)
    diff = max(float(np.abs(exact[0] - fast[0]).max()),
               float(np.abs(exact[1] - fast[1]).max()))
    print(f'distance correlation check: largest difference {diff:.2e} over '
          f'{len(cols)} features, {len(rows)} compounds', flush=True)
    if diff > 1e-6:
        raise RuntimeError('dcor and the direct formula disagree')
    return diff


def _low_information(df, seed):
    """MODNet's rule (get_cross_nmi, drop_thr=0.2): features whose mutual
    information with themselves is below 0.2."""
    from modnet.preprocessing import compute_mi
    return [c for c in df.columns
            if compute_mi(df[c].values, df[c].values, random_state=seed,
                          n_neighbors=3)[0] < 0.2]


def rank_features(train, measure, low_information, n_jobs, seed):
    """Sets train.optimal_features with MODNet's own ranking, using
    `measure` for both scores. Returns the number of features ranked."""
    global _DCOR_CHECKED
    import pandas as pd
    from modnet.preprocessing import get_cross_nmi, get_features_dyn
    if measure == 'mutual_information_file':
        train.feature_selection(n=-1, use_precomputed_cross_nmi=True,
                                n_jobs=n_jobs)
        return len(train.optimal_features)
    if measure == 'mutual_information':
        if low_information == 'drop':
            train.feature_selection(n=-1, n_jobs=n_jobs)
        else:
            df, _ = _scoring_frame(train)
            cross = get_cross_nmi(df, drop_thr=0, n_jobs=n_jobs)
            train.feature_selection(n=-1, cross_nmi=cross, n_jobs=n_jobs)
        return len(train.optimal_features)
    df, y = _scoring_frame(train)
    df = _prepare(df)
    if low_information == 'drop':
        df = df.drop(columns=_low_information(df, seed))
    X, yv = df.to_numpy(float), y.to_numpy(float)
    if measure == 'pearson':
        pair, target = _abs_corr(X, yv)
    elif measure == 'spearman':
        R = df.rank().to_numpy(float)
        pair, target = _abs_corr(R, y.rank().to_numpy(float))
    elif measure == 'distance_correlation':
        if not _DCOR_CHECKED:
            check_dcor(df, y, seed=seed)
            _DCOR_CHECKED = True
        pair, target = _dcor_fast(X, yv)
    else:
        raise ValueError(measure)
    names = list(df.columns)
    ranked = get_features_dyn(-1, pd.DataFrame(pair, index=names, columns=names),
                              pd.Series(target, index=names))
    train.optimal_features = ranked
    train.optimal_features_by_target = {b04.TARGET: ranked}
    return len(ranked)


def _moddata(X, y, names, prefix):
    import pandas as pd
    from modnet.preprocessing import MODData
    ids = [f'{prefix}{i}' for i in range(len(X))]
    return MODData(materials=[None] * len(ids),
                   targets=np.asarray(y, float).reshape(-1, 1),
                   target_names=[b04.TARGET], structure_ids=ids,
                   df_featurized=pd.DataFrame(X, index=ids, columns=names))


HYBRID_DIR = str(b04.ROOT / 'benchmarks_protocol' / 'matbench' / 'hybrid_features')


def hybrid_fold(hybrid_dir, task, k, name, tr, te, y):
    path = Path(hybrid_dir) / f'{task}_fold{k}_{name}.npz'
    z = np.load(path, allow_pickle=False)
    if not (np.array_equal(z['train_index'], tr)
            and np.array_equal(z['test_index'], te)):
        raise RuntimeError(f'{path}: compounds do not match Matbench fold {k}')
    names = [str(n) for n in z['feature_names']]
    return (_moddata(z['X_train'], y[tr], names, 'tr'),
            _moddata(z['X_test'], y[te], names, 'te'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default=','.join(b04.TASKS))
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--inputs', default='modnet',
                    help='comma list of modnet, hybrid')
    ap.add_argument('--measures',
                    default=','.join(['mutual_information_file'] + MEASURES))
    ap.add_argument('--hybrids', default='',
                    help='comma list of <model>_hybrid<i>, for --inputs hybrid')
    ap.add_argument('--hybrid-dir', default=HYBRID_DIR)
    ap.add_argument('--low-information', default='drop,keep',
                    help='comma list of drop, keep')
    ap.add_argument('--check-dcor', action='store_true',
                    help='check and time distance correlation only, then stop')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-jobs', type=int, default=4)
    ap.add_argument('--fast', action='store_true',
                    help="MODNet's debug mode, as in b04 --fast.")
    a = ap.parse_args()
    os.environ.setdefault('MATMINER_DATA', str(b04.ROOT / 'matbench' / 'data'))
    from modnet.matbench.benchmark import matbench_kfold_splits

    if a.check_dcor:
        for task in a.tasks.split(','):
            data = b04.featurized(task, a.n_jobs)
            tr, te = list(matbench_kfold_splits(data))[0]
            train, _ = data.split((tr, te))
            df, y = _scoring_frame(train)
            check_dcor(df, y, seed=a.seed)
            df = _prepare(df)
            X, yv = df.to_numpy(float), y.to_numpy(float)
            rs = np.random.RandomState(a.seed)
            cols = rs.choice(X.shape[1], min(20, X.shape[1]), replace=False)
            t0 = time.time()
            _dcor_fast(X[:, cols], yv)      # 20 features + target: 210 pairs
            per_pair = (time.time() - t0) / 210
            n_pairs = (X.shape[1] + 1) * X.shape[1] / 2
            print(f'{task}: {X.shape[1]} features, {len(X)} compounds; '
                  f'{per_pair * 1000:.2f} ms per pair, about '
                  f'{per_pair * n_pairs / 60:.0f} min for all {n_pairs:.0f} '
                  f'pairs on one core', flush=True)
        # The largest tasks score on N_SAMPLES compounds. Their features take
        # hours to compute, so this times made-up values at that size: the
        # time depends on the number of compounds, not on the values.
        rs = np.random.RandomState(a.seed)
        X = rs.randn(N_SAMPLES, 20)
        t0 = time.time()
        _dcor_fast(X, rs.randn(N_SAMPLES))
        per_pair = (time.time() - t0) / 210
        n_pairs = 1699 * 1698 / 2
        print(f'{N_SAMPLES} made-up compounds: {per_pair * 1000:.2f} ms per '
              f'pair; at 1698 features (jdft2d\'s MODNet count) about '
              f'{per_pair * n_pairs / 60:.0f} min on one core', flush=True)
        return

    inputs = [i for i in a.inputs.split(',') if i != 'hybrid']
    if 'hybrid' in a.inputs.split(','):
        if not a.hybrids:
            ap.error('--inputs hybrid needs --hybrids')
        inputs += [f'hybrid:{h}' for h in a.hybrids.split(',')]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {(r['task'], r['fold'], r['inputs'], r['measure'],
             r['low_information'], r['fast']) for r in results}
    settings = a.low_information.split(',')

    for task in a.tasks.split(','):
        data = b04.featurized(task, a.n_jobs)
        y = data.df_targets[b04.TARGET].to_numpy()
        splits = list(matbench_kfold_splits(data))
        for k in [int(f) for f in a.folds.split(',')]:
            tr, te = splits[k]
            for inp in inputs:
                for measure, low in [
                        (m, 'file' if m == 'mutual_information_file' else s)
                        for m in a.measures.split(',')
                        for s in (['file'] if m == 'mutual_information_file'
                                  else settings)]:
                    key = (task, k, inp, measure, low, a.fast)
                    if key in done:
                        continue
                    if inp != 'modnet' and measure == 'mutual_information_file':
                        continue   # the file only scores MODNet's own features
                    if inp == 'modnet':
                        train, test = data.split((tr, te))
                    else:
                        train, test = hybrid_fold(a.hybrid_dir, task, k,
                                                  inp.split(':', 1)[1], tr, te, y)
                    t0 = time.time()
                    n_ranked = rank_features(train, measure, low, a.n_jobs,
                                             a.seed)
                    t1 = time.time()
                    pred = b04._train_fold(k, train, test, a.n_jobs, a.fast)
                    y_te = test.df_targets[b04.TARGET].to_numpy()
                    mae = float(np.mean(np.abs(pred - y_te)))
                    row = dict(task=task, fold=k, inputs=inp, measure=measure,
                               low_information=low, fast=a.fast, mae=mae, n_train=len(tr),
                               n_test=len(te),
                               n_features=int(train.df_featurized.shape[1]),
                               n_ranked=n_ranked,
                               top_features=list(train.optimal_features[:TOP]),
                               seconds_scoring=round(t1 - t0, 1),
                               seconds_training=round(time.time() - t1, 1))
                    results.append(row)
                    done.add(key)
                    OUT.write_text(json.dumps(results, indent=1))
                    print(f'{task} fold {k} {inp} {measure} {low}: '
                          f'{mae:.4f} MAE, '
                          f'{n_ranked} features ranked, '
                          f'{row["seconds_scoring"]:.0f} s scoring, '
                          f'{row["seconds_training"]:.0f} s training', flush=True)


if __name__ == '__main__':
    main()
