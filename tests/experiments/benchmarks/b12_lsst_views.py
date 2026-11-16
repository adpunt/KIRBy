#!/usr/bin/env python3
"""LSST (UEA archive) view and model screen: every candidate view of one light
curve, each scored alone under every model a paper on this dataset used, on
the default train/test split, with test predictions kept and overlap between
views measured. Contract: tests/experiments/screen/datasets.py.

Object: one light curve, 6 passbands (u g r i z y) x 36 flux values. 2,459
training and 2,466 test curves (LSST_TRAIN.ts / LSST_TEST.ts), 14 classes, the
largest 31.5 % of the test file. The archive keeps flux only: no observation
times, no flux errors, no redshift. Index j of one band and index j of another
are not the same moment, so no view below differences two bands at one index.

Split. `default` is the archive's train/test files, the split every reference
number uses (Multiverse, arXiv 2603.20352, single run, seed 0). `r<k>` (k = 1
to 29) is resample k of Ruiz et al. 2021 (aeon `stratified_resample_data`,
random_state=k), whose Table 11 averages 30 resamples, resample 0 being the
default split. Which splits the screen runs is set in the ledger row.

Views (one vector per curve). Source: "code" = the named package's own
transform with the published classifier's settings; "paper" = written from a
paper's text; "none" = no paper. Views marked * are fitted on the split's
training curves only.

    raw                 216  flux values, bands concatenated (1NN-ED, Bagnall 2018)        none
    band_stats           54  per band max, min, mean, median, std, q10, q25, q75, q90     none
    band_colours        270  the 9 statistics above, every pair of the 6 bands (15):
                             difference and normalised difference (a-b)/(|a|+|b|).
                             Summaries first, then across bands, as Boone 2019's
                             flux ratios are taken between per-band summaries        paper
    index_periodogram   108  per band |rFFT|^2 of the mean-removed 36 values, bins
                             1-18; frequency in index units, not physical (DrCIF's
                             periodogram representation, Middlehurst 2021)            paper
    catch24             144  aeon Catch22(catch24=True, outlier_norm=False,
                             replace_nans=True): tsml-eval's Catch22 settings         code
    tsfresh          ~4,700  aeon TSFresh('comprehensive'), FreshPRINCE's transform   code
    rocket*          20,000  aeon Rocket(n_kernels=10000, random_state=0)             code
    minirocket*      ~9,996  aeon MiniRocket(random_state=0)                          code
    multirocket*    ~49,728  aeon MultiRocket(random_state=0)                         code
    hydra*            varies aeon HydraTransformer(random_state=0)                    code
    quant*            varies aeon QUANTTransformer() defaults                         code
    rdst*            30,000  aeon RandomDilatedShapeletTransform(max_shapelets=10000,
                             random_state=0); supervised, uses training labels        code
    mantis            1,536  Mantis-8M frozen (paris-noah/Mantis-8M), each band
                             interpolated 36 -> 512, per-band embeddings              code
    mantisv2          1,536  MantisV2 frozen, same handling (MantisV2 + RF: 66.0 %,
                             TS2Vec-FM paper Table 10)                                code
    moment            6,144  MOMENT-1-large frozen, each band interpolated to 512,
                             per-band embeddings concatenated                         code
    chronos2          4,608  Chronos-2 frozen encoder, each band's 36 values, mean
                             over tokens, bands concatenated                          code
    ts2vec*             320  TS2Vec (github.com/yuezhihan/ts2vec), trained on the
                             split's training curves, 6 input channels, full-series
                             encoding (TS2Vec paper, LSST 0.537 with its SVM)         code

Models. Each is the classifier a paper used on one of these views; every
model runs on every view.

    ridgecv     StandardScaler(with_mean=False) + RidgeClassifierCV(alphas=logspace(-3,3,10)):
                aeon's ROCKET, MiniRocket and MultiRocket classifiers
    rotf        aeon RotationForestClassifier(n_estimators=200, random_state=0): FreshPRINCE
    rf          RandomForestClassifier(n_estimators=200, random_state=0): aeon's Catch22
                classifier, and the 200-tree probe of the TS2Vec-FM paper
    extratrees  ExtraTreesClassifier(n_estimators=200, max_features=0.1, criterion='entropy',
                random_state=0): QUANT
    1nn         1-nearest neighbour, Euclidean, no scaling: Bagnall 2018's ED baseline
    logreg      StandardScaler + LogisticRegression(max_iter=1000): the TS2Vec-FM paper's
                linear probe. Its settings are not stated there; sklearn defaults
    svm         TS2Vec's protocol: SVC(rbf, gamma='scale'), C chosen by 5-fold grid
                search over 1e-4 ... 1e4 and inf. Written from TS2Vec's
                tasks/_eval_protocols.py as remembered, not re-read on 2026-11-09

Non-finite feature values are replaced by the column's median over the split's
training curves before any model sees them.

Metrics on the test curves: accuracy (the published metric), balanced accuracy
and macro F1.

    python tests/experiments/benchmarks/b12_lsst_views.py --stage data
    python tests/experiments/benchmarks/b12_lsst_views.py --stage build --views catch24 --splits default
    python tests/experiments/benchmarks/b12_lsst_views.py --stage evaluate --views catch24 --models rf --splits default
    python tests/experiments/benchmarks/b12_lsst_views.py --stage summarise

Paths: KIRBY_LSST_DIR (raw/ and views/; default
/Volumes/seagate/benchmarks_protocol/lsst), KIRBY_BENCH_RESULTS (results root;
results under b12_lsst_views/), KIRBY_TS2VEC_DIR (a clone of the TS2Vec repo).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.environ.get('KIRBY_LSST_DIR') or '/Volumes/seagate/benchmarks_protocol/lsst')
RESULTS_ROOT = Path(os.environ.get('KIRBY_BENCH_RESULTS') or '/Volumes/seagate/benchmarks_protocol/results')
RESULTS_DIR = RESULTS_ROOT / 'b12_lsst_views'
VIEWS_DIR = DATA_DIR / 'views'
TS2VEC_DIR = Path(os.environ.get('KIRBY_TS2VEC_DIR')
                  or '/Volumes/seagate/benchmarks_protocol/opssat_ad/ext/ts2vec')
SEED = 0
N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK') or os.environ.get('KIRBY_LSST_JOBS') or 1)
BANDS = 'ugrizy'

UNFITTED_VIEWS = ['raw', 'band_stats', 'band_colours', 'index_periodogram', 'catch24', 'tsfresh',
                  'mantis', 'mantisv2', 'moment', 'chronos2']
FITTED_VIEWS = ['rocket', 'minirocket', 'multirocket', 'hydra', 'quant', 'rdst', 'ts2vec']
VIEWS = UNFITTED_VIEWS + FITTED_VIEWS
TORCH_VIEWS = {'mantis', 'mantisv2', 'moment', 'chronos2', 'ts2vec'}
MODELS = ['ridgecv', 'rotf', 'rf', 'extratrees', '1nn', 'logreg', 'svm']


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg}', flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'b12_lsst.log', 'a') as f:
        f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}\n')


def _atomic_save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez(tmp, **arrays)
    tmp.rename(path)


def _atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.rename(path)


# ----------------------------------------------------------------------------
# Data and splits
# ----------------------------------------------------------------------------

def fetch():
    """Downloads the archive files once (aeon load_classification) and keeps
    them as one npz: X (4,925 x 6 x 36) in train-file then test-file order."""
    out = DATA_DIR / 'lsst.npz'
    if out.exists():
        return out
    from aeon.datasets import load_classification
    raw = DATA_DIR / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    X_tr, y_tr = load_classification('LSST', split='train', extract_path=str(raw))
    X_te, y_te = load_classification('LSST', split='test', extract_path=str(raw))
    X = np.concatenate([X_tr, X_te]).astype(np.float64)
    y = np.concatenate([y_tr, y_te]).astype(str)
    is_train = np.r_[np.ones(len(y_tr), bool), np.zeros(len(y_te), bool)]
    _atomic_save_npz(out, X=X, y=y, is_train=is_train)
    log(f'data: {X.shape} ({len(y_tr)} train, {len(y_te)} test), {len(np.unique(y))} classes')
    return out


def split_index(split):
    """(train indices, test indices) into lsst.npz rows."""
    d = np.load(DATA_DIR / 'lsst.npz')
    tr = np.where(d['is_train'])[0]
    te = np.where(~d['is_train'])[0]
    if split == 'default':
        return tr, te
    if not (split.startswith('r') and split[1:].isdigit() and 1 <= int(split[1:]) <= 29):
        raise SystemExit(f'unknown split {split!r}: default or r1 ... r29')
    from aeon.benchmarking.resampling import stratified_resample_data_indices
    y = d['y']
    new_tr, new_te = stratified_resample_data_indices(y[tr], y[te], random_state=int(split[1:]))
    allidx = np.r_[tr, te]
    return allidx[new_tr], allidx[new_te]


def load_split(split='default'):
    fetch()
    d = np.load(DATA_DIR / 'lsst.npz')
    tr, te = split_index(split)
    return d['X'][tr], d['y'][tr], d['X'][te], d['y'][te]


# ----------------------------------------------------------------------------
# Views
# ----------------------------------------------------------------------------

def _interp(X3, n=512):
    """(cases, bands, m) -> (cases, bands, n), linear in the index."""
    m = X3.shape[2]
    grid = np.linspace(0, m - 1, n)
    return np.stack([[np.interp(grid, np.arange(m), x) for x in case] for case in X3])


STATS = ['max', 'min', 'mean', 'median', 'std', 'q10', 'q25', 'q75', 'q90']


def _band_stats(X3):
    f = {'max': X3.max(2), 'min': X3.min(2), 'mean': X3.mean(2), 'median': np.median(X3, 2),
         'std': X3.std(2), 'q10': np.quantile(X3, .1, 2), 'q25': np.quantile(X3, .25, 2),
         'q75': np.quantile(X3, .75, 2), 'q90': np.quantile(X3, .9, 2)}
    return np.stack([f[s] for s in STATS], 2)          # cases x bands x stats


def unfitted_view(name, X3):
    """One matrix for every curve (no training rows involved)."""
    if name == 'raw':
        return X3.reshape(len(X3), -1), None
    if name == 'band_stats':
        S = _band_stats(X3)
        return S.reshape(len(X3), -1), [f'{b}_{s}' for b in BANDS for s in STATS]
    if name == 'band_colours':
        S = _band_stats(X3)
        cols, names = [], []
        for i in range(6):
            for j in range(i + 1, 6):
                a, b = S[:, i, :], S[:, j, :]
                cols += [a - b, (a - b) / (np.abs(a) + np.abs(b) + 1e-12)]
                names += [f'{BANDS[i]}-{BANDS[j]}_{s}_diff' for s in STATS]
                names += [f'{BANDS[i]}-{BANDS[j]}_{s}_normdiff' for s in STATS]
        return np.concatenate(cols, 1), names
    if name == 'index_periodogram':
        P = np.abs(np.fft.rfft(X3 - X3.mean(2, keepdims=True), axis=2)) ** 2
        return P[:, :, 1:].reshape(len(X3), -1), None
    if name == 'catch24':
        from aeon.transformations.collection.feature_based import Catch22
        T = Catch22(catch24=True, outlier_norm=False, replace_nans=True, n_jobs=N_JOBS)
        return np.asarray(T.fit_transform(X3)), None
    if name == 'tsfresh':
        from aeon.transformations.collection.feature_based import TSFresh
        T = TSFresh(default_fc_parameters='comprehensive', n_jobs=N_JOBS)
        return np.asarray(T.fit_transform(X3), np.float64), None
    if name in ('mantis', 'mantisv2'):
        import torch  # noqa: F401
        from mantis.architecture import Mantis8M, MantisV2
        from mantis.trainer import MantisTrainer
        if name == 'mantis':
            net = Mantis8M(device='cpu').from_pretrained('paris-noah/Mantis-8M')
        else:
            net = MantisV2(device='cpu').from_pretrained('paris-noah/MantisV2')
        tr = MantisTrainer(device='cpu', network=net)
        Z = np.asarray(tr.transform(_interp(X3).astype(np.float32)))
        return Z.reshape(len(X3), -1), None
    if name == 'moment':
        import torch
        from momentfm import MOMENTPipeline
        m = MOMENTPipeline.from_pretrained('AutonLab/MOMENT-1-large', model_kwargs={'task_name': 'embedding'})
        m.init()
        m.eval()
        R = _interp(X3)
        outs = []
        with torch.no_grad():
            for st in range(0, len(R), 32):
                xb = R[st:st + 32]
                per_band = [m(x_enc=torch.as_tensor(xb[:, [b], :], dtype=torch.float32)).embeddings.numpy()
                            for b in range(xb.shape[1])]
                outs.append(np.concatenate(per_band, 1))
        return np.concatenate(outs), None
    if name == 'chronos2':
        import torch
        from chronos import Chronos2Pipeline
        pipe = Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map='cpu', torch_dtype=torch.float32)
        outs = []
        for st in range(0, len(X3), 32):
            xb = X3[st:st + 32]
            per_band = []
            for b in range(xb.shape[1]):
                embs, _ = pipe.embed(torch.as_tensor(xb[:, [b], :], dtype=torch.float32))
                per_band.append(np.stack([e[0].detach().cpu().float().numpy().mean(0) for e in embs]))
            outs.append(np.concatenate(per_band, 1))
        return np.concatenate(outs), None
    raise ValueError(name)


def fitted_view(name, X_tr, y_tr, X_te):
    """Fitted on the split's training curves, applied to both sides."""
    if name in ('rocket', 'minirocket', 'multirocket', 'hydra'):
        from aeon.transformations.collection import convolution_based as cb
        cls = {'rocket': 'Rocket', 'minirocket': 'MiniRocket', 'multirocket': 'MultiRocket',
               'hydra': 'HydraTransformer'}[name]
        kw = {'n_kernels': 10000} if name == 'rocket' else {}
        T = getattr(cb, cls)(random_state=SEED, n_jobs=N_JOBS, **kw)
        T.fit(X_tr)
        f = (lambda Z: Z.numpy() if hasattr(Z, 'numpy') else np.asarray(Z))
        return f(T.transform(X_tr)), f(T.transform(X_te))
    if name == 'quant':
        from aeon.transformations.collection.interval_based import QUANTTransformer
        T = QUANTTransformer()
        T.fit(X_tr)
        return np.asarray(T.transform(X_tr)), np.asarray(T.transform(X_te))
    if name == 'rdst':
        from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
        T = RandomDilatedShapeletTransform(max_shapelets=10000, random_state=SEED, n_jobs=N_JOBS)
        T.fit(X_tr, y_tr)
        return np.asarray(T.transform(X_tr)), np.asarray(T.transform(X_te))
    if name == 'ts2vec':
        import torch
        if not (TS2VEC_DIR / 'ts2vec.py').exists():
            raise RuntimeError(f'ts2vec needs a clone of github.com/yuezhihan/ts2vec at {TS2VEC_DIR} '
                               '(set KIRBY_TS2VEC_DIR)')
        sys.path.insert(0, str(TS2VEC_DIR))
        from ts2vec import TS2Vec
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = TS2Vec(input_dims=X_tr.shape[1], output_dims=320, device='cpu')
        model.fit(np.transpose(X_tr, (0, 2, 1)).astype(np.float32), verbose=False)
        enc = (lambda X: model.encode(np.transpose(X, (0, 2, 1)).astype(np.float32),
                                      encoding_window='full_series'))
        return enc(X_tr), enc(X_te)
    raise ValueError(name)


def view_file(view, split):
    return VIEWS_DIR / split / f'{view}.npz'


def build(view, split):
    out = view_file(view, split)
    if out.exists():
        log(f'build {view} {split}: exists')
        return
    t0 = time.time()
    X_tr, y_tr, X_te, y_te = load_split(split)
    names = None
    if view in UNFITTED_VIEWS:
        # computed per curve, so the default split's file serves every split
        base = view_file(view, 'default_all')
        if not base.exists():
            d = np.load(DATA_DIR / 'lsst.npz')
            Z, names = unfitted_view(view, d['X'])
            _atomic_save_npz(base, Z=np.asarray(Z, np.float64))
        Z = np.load(base)['Z']
        tr, te = split_index(split)
        Z_tr, Z_te = Z[tr], Z[te]
    else:
        Z_tr, Z_te = fitted_view(view, X_tr, y_tr, X_te)
    _atomic_save_npz(out, X_tr=np.asarray(Z_tr, np.float64), X_te=np.asarray(Z_te, np.float64))
    info = {'view': view, 'split': split, 'n_features': int(Z_tr.shape[1]),
            'seconds': time.time() - t0, 'fitted_on': 'training curves' if view in FITTED_VIEWS else 'none',
            'names': names}
    _atomic_json(out.with_suffix('.json'), info)
    log(f'build {view} {split}: {Z_tr.shape[1]} features, {info["seconds"]:.0f} s')


# ----------------------------------------------------------------------------
# Models (never imports torch)
# ----------------------------------------------------------------------------

def make_model(name):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == 'ridgecv':
        from sklearn.linear_model import RidgeClassifierCV
        return make_pipeline(StandardScaler(with_mean=False), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))
    if name == 'rotf':
        from aeon.classification.sklearn import RotationForestClassifier
        return RotationForestClassifier(n_estimators=200, random_state=SEED, n_jobs=N_JOBS)
    if name == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=N_JOBS)
    if name == 'extratrees':
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=200, max_features=0.1, criterion='entropy',
                                    random_state=SEED, n_jobs=N_JOBS)
    if name == '1nn':
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=1, n_jobs=N_JOBS)
    if name == 'logreg':
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if name == 'svm':
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVC
        grid = {'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, np.inf]}
        return GridSearchCV(SVC(kernel='rbf', gamma='scale', max_iter=10000000), grid, cv=5, n_jobs=N_JOBS)
    raise ValueError(name)


def impute_train_median(A_tr, A_te):
    A_tr = np.array(A_tr, np.float64)
    A_te = np.array(A_te, np.float64)
    A_tr[~np.isfinite(A_tr)] = np.nan
    A_te[~np.isfinite(A_te)] = np.nan
    med = np.nanmedian(A_tr, 0) if np.isnan(A_tr).any() else np.zeros(A_tr.shape[1])
    med[~np.isfinite(med)] = 0.0
    for A in (A_tr, A_te):
        r, c = np.where(np.isnan(A))
        A[r, c] = med[c]
    return A_tr, A_te


def result_file(view, model, split):
    return RESULTS_DIR / split / f'{view}__{model}.json'


def evaluate(view, model, split):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    out = result_file(view, model, split)
    if out.exists():
        log(f'evaluate {view} {model} {split}: exists')
        return
    vf = view_file(view, split)
    if not vf.exists():
        raise SystemExit(f'{vf} missing: build {view} {split} first')
    v = np.load(vf)
    X_tr, X_te = impute_train_median(v['X_tr'], v['X_te'])
    _, y_tr, _, y_te = load_split(split)
    t0 = time.time()
    m = make_model(model).fit(X_tr, y_tr)
    pred = m.predict(X_te)
    sec = time.time() - t0
    res = {'view': view, 'model': model, 'split': split, 'n_features': int(X_tr.shape[1]),
           'n_train': int(len(y_tr)), 'n_test': int(len(y_te)),
           'accuracy': float(accuracy_score(y_te, pred)), 'n_correct': int((pred == y_te).sum()),
           'balanced_accuracy': float(balanced_accuracy_score(y_te, pred)),
           'macro_f1': float(f1_score(y_te, pred, average='macro')), 'seconds': sec,
           'n_jobs': N_JOBS, 'result_status': 'ok', 'debug': False}
    if model == 'svm':
        res['svm_best_C'] = float(m.best_params_['C'])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_name(f'{view}__{model}.pred.npy'), np.asarray(pred).astype(str))
    _atomic_json(out, res)
    log(f'evaluate {view} {model} {split}: accuracy {res["accuracy"]:.4f} '
        f'balanced {res["balanced_accuracy"]:.4f} ({sec:.0f} s)')


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------

REFERENCE = ('Reference on the default split (Multiverse, arXiv 2603.20352, accuracy_mean.csv, seed 0): '
             'FreshPRINCE 0.7028, HC2 0.6740, Catch22 0.6695, Arsenal 0.6577, RDST 0.6492, QUANT 0.6452, '
             'MR-Hydra 0.6427, ROCKET 0.6363, Dummy 0.3151. Fine-tuned Mantis 0.7109 (3 seeds, Mantis paper). '
             'Frozen MantisV2 + 200-tree RF 0.660 (TS2Vec-FM paper, Table 10).')


def _cka_matrix(Z):
    """Cheaper exact linear CKA through the curve x curve Gram matrix (wide views)."""
    Z = Z - Z.mean(0)
    return Z @ Z.T


def summarise():
    lines = ['# LSST view and model screen', '', REFERENCE, '']
    for split_dir in sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir()):
        split = split_dir.name
        results = [json.loads(p.read_text()) for p in sorted(split_dir.glob('*__*.json'))]
        if not results:
            continue
        views = [v for v in VIEWS if any(r['view'] == v for r in results)]
        models = [m for m in MODELS if any(r['model'] == m for r in results)]
        n_test = results[0]['n_test']
        cell = {(r['view'], r['model']): r for r in results}
        lines += [f'## split {split}', '',
                  f'One row is one view. A cell is test accuracy / balanced accuracy, {n_test} test curves.', '',
                  '| view | features | ' + ' | '.join(models) + ' |', '|---|---|' + '---|' * len(models)]
        for v in views:
            nf = next(r['n_features'] for r in results if r['view'] == v)
            cs = [f'{cell[v, m]["accuracy"]:.4f} / {cell[v, m]["balanced_accuracy"]:.4f}'
                  if (v, m) in cell else '' for m in models]
            lines.append(f'| {v} | {nf:,} | ' + ' | '.join(cs) + ' |')
        lines += ['', 'Seconds to fit and predict, same layout.', '',
                  '| view | ' + ' | '.join(models) + ' |', '|---|' + '---|' * len(models)]
        for v in views:
            cs = [f'{cell[v, m]["seconds"]:.0f}' if (v, m) in cell else '' for m in models]
            lines.append(f'| {v} | ' + ' | '.join(cs) + ' |')
        lines.append('')
        _, _, _, y_te = load_split(split)
        for m in models:
            preds = {}
            for v in views:
                p = split_dir / f'{v}__{m}.pred.npy'
                if p.exists():
                    preds[v] = np.load(p)
            if len(preds) < 2:
                continue
            vs = list(preds)
            err = {v: preds[v] != y_te for v in vs}
            lines += [f'### error overlap, {m}', '',
                      'Cell: of the test curves either view gets wrong, the percentage both get wrong. '
                      'Diagonal: test error rate in %.', '',
                      '| | ' + ' | '.join(vs) + ' |', '|---|' + '---|' * len(vs)]
            for a in vs:
                cs = []
                for b in vs:
                    if a == b:
                        cs.append(f'{100 * err[a].mean():.1f}')
                    else:
                        cs.append(f'{100 * (err[a] & err[b]).sum() / max((err[a] | err[b]).sum(), 1):.0f}')
                lines.append(f'| {a} | ' + ' | '.join(cs) + ' |')
            lines.append('')
        grams = {}
        for v in views:
            vf = view_file(v, split)
            if not vf.exists():
                continue
            Z = np.load(vf)['X_te'].astype(np.float64)
            Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
            Z[~np.isfinite(Z)] = 0.0
            grams[v] = _cka_matrix(Z)
        gv = list(grams)
        lines += ['### linear CKA between views', '',
                  f'Between standardised view matrices on the {n_test} test curves. '
                  '1 = the same information up to a linear map, 0 = unrelated.', '',
                  '| | ' + ' | '.join(gv) + ' |', '|---|' + '---|' * len(gv)]
        for a in gv:
            cs = []
            for b in gv:
                Ka, Kb = grams[a], grams[b]
                cs.append(f'{float((Ka * Kb).sum() / (np.linalg.norm(Ka) * np.linalg.norm(Kb) + 1e-30)):.2f}')
            lines.append(f'| {a} | ' + ' | '.join(cs) + ' |')
        lines.append('')
    tmp = RESULTS_DIR / 'summary.md.tmp'
    tmp.write_text('\n'.join(lines) + '\n')
    tmp.rename(RESULTS_DIR / 'summary.md')
    log(f'summary written: {RESULTS_DIR / "summary.md"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['data', 'build', 'evaluate', 'summarise'], required=True)
    ap.add_argument('--views', default=','.join(VIEWS))
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--splits', default='default', help='default and/or r1 ... r29, comma separated')
    a = ap.parse_args()
    if a.stage == 'data':
        fetch()
        return
    if a.stage == 'summarise':
        summarise()
        return
    fetch()
    for s in a.splits.split(','):
        for v in a.views.split(','):
            if a.stage == 'build':
                build(v, s)
            else:
                for m in a.models.split(','):
                    evaluate(v, m, s)


if __name__ == '__main__':
    main()
