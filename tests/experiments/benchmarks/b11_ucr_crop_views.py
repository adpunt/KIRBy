#!/usr/bin/env python3
"""UCR Crop: reproduction of the published classifiers, and the view x model
screen, on the archive's fixed split.

Dataset. UCR 2018 archive, Crop (donor F. Petitjean). One row is one pixel's
NDVI series from 46 FORMOSAT-2 images; 24 crop classes. Fixed split: 7,200
training and 16,800 test rows (the archive's TRAIN and TEST files; tsml-eval
calls this resample 0). Downloaded by aeon's load_classification from
timeseriesclassification.com.

Stages (one command per ledger unit; result file written last):

  data        download, check shapes, count test rows identical to a
              training row, write <data>/crop.npz
  published   one published classifier end to end, aeon 1.3 code
              (--models NAME); the score is put next to tsml-eval's
              resample-0 accuracy in the summary
  build       one view (--views NAME) -> <data>/views/<name>.npy
  evaluate    one view x one model (x one appended block with --plus)
  summarise   <results>/b11_views/summary.md

Views. "code" = called from the named package; "paper" = written from the
paper's text; "none" = no paper. Views fitted on data are fitted on the
7,200 training rows only.

    raw              46  the series as shipped                       none
    diff1            45  first difference                            none
    diff2            44  second difference                           none
    periodogram      24  |rfft| of the mean-removed series           none
    harmonic2         5  least-squares fit c + sum_k a_k cos + b_k sin,
                         k = 1, 2, period = 46 steps                 paper (Roerink 2000, plain OLS,
                                                                     no HANTS outlier iterations)
    harmonic3         7  the same, k = 1..3                          paper
    catch24          24  pycatch22, catch24=True                     code
    tsfresh        ~780  tsfresh ComprehensiveFCParameters           code
    rocket       20,000  aeon Rocket                                 code
    minirocket    9,996  aeon MiniRocket                             code
    multirocket ~49,728  aeon MultiRocket                            code
    hydra          var.  aeon HydraTransformer                       code
    quant          var.  aeon QUANTTransformer                       code
    rdst         ~30,000 aeon RandomDilatedShapeletTransform         code
    weasel2        var.  aeon WEASELTransformerV2 (supervised)       code
    random_intervals var. aeon RandomIntervals, defaults            code
    supervised_intervals var. aeon SupervisedIntervals (supervised) code
    ed_medoid        24  Euclidean distance to each class's training
                         medoid                                      none
    dtw_medoid       24  full-window DTW distance to the same medoids none
    moment        1,024  AutonLab/MOMENT-1-large, frozen, series
                         zero-padded to 512 with an input mask       code
    mantis          256  paris-noah/Mantis-8M, frozen, series linearly
                         interpolated to 512                         code
    chronos2        768  amazon/chronos-2 encoder, mean over tokens  code
    ts2vec          320  TS2Vec (github.com/yuezhihan/ts2vec, MIT),
                         trained on the training rows, full-series
                         encoding                                     code
    inceptiontime   640  InceptionTime (Ismail Fawaz 2020), PyTorch
                         port of github.com/hfawaz/InceptionTime: 5
                         networks trained on the training rows, each
                         network's 128 pooled values concatenated     code (ported)
    phenology        13  TIMESAT-style season metrics (Jonsson and
                         Eklundh 2004). Its smoothing and threshold
                         are inputs with no value: the view builds
                         only when all three are given               paper

Models (vector models the Crop papers used, with their code's settings):

    ridge_cv        StandardScaler(with_mean=False) + RidgeClassifierCV(
                    alphas=logspace(-3, 3, 10))       aeon MultiRocketClassifier
    extra_trees     ExtraTreesClassifier(200, max_features=0.1,
                    criterion='entropy')               aeon QUANTClassifier
    rf              RandomForestClassifier(200)        aeon Catch22Classifier
    rotation_forest aeon RotationForestClassifier(200) aeon FreshPRINCEClassifier
    logreg          StandardScaler + LogisticRegression(max_iter=1000)
                    (probe of arXiv 2610.28602; its settings not stated)
    svm             TS2Vec's fit_svm: SVC(rbf, gamma='scale'), C over
                    1e-4..1e4 and inf by 5-fold GridSearchCV, at most
                    10,000 training rows                 ts2vec/tasks/_eval_protocols.py
    knn1            1-nearest neighbour, Euclidean       UCR archive baseline

Seeds: 0 everywhere a seed is taken.

    python tests/experiments/benchmarks/b11_ucr_crop_views.py --stage data
    python tests/experiments/benchmarks/b11_ucr_crop_views.py --stage published --models multirocket
    python tests/experiments/benchmarks/b11_ucr_crop_views.py --stage build --views minirocket
    python tests/experiments/benchmarks/b11_ucr_crop_views.py --stage evaluate --views minirocket --models ridge_cv
    python tests/experiments/benchmarks/b11_ucr_crop_views.py --stage evaluate --views minirocket --models ridge_cv --plus harmonic3
    python tests/experiments/benchmarks/b11_ucr_crop_views.py --stage summarise

Paths: KIRBY_CROP_DIR (data and views; default
/Volumes/seagate/benchmarks_protocol/ucr_crop), KIRBY_BENCH_RESULTS (results;
default /Volumes/seagate/benchmarks_protocol/results). --debug fits on the
first 1,000 training rows and marks every file debug.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get('KIRBY_CROP_DIR') or '/Volumes/seagate/benchmarks_protocol/ucr_crop')
VIEWS_DIR = DATA / 'views'
RESULTS = Path(os.environ.get('KIRBY_BENCH_RESULTS')
               or '/Volumes/seagate/benchmarks_protocol/results') / 'b11_views'
SEED = 0
LENGTH = 46

VIEWS = ['raw', 'diff1', 'diff2', 'periodogram', 'harmonic2', 'harmonic3', 'catch24',
         'tsfresh', 'rocket', 'minirocket', 'multirocket', 'hydra', 'quant', 'rdst',
         'weasel2', 'random_intervals', 'supervised_intervals', 'ed_medoid', 'dtw_medoid',
         'moment', 'mantis', 'chronos2', 'ts2vec', 'inceptiontime']
# Built only when its three settings are given (none has a value yet).
SETTING_VIEWS = ['phenology']
# Views whose fit reads the training labels.
SUPERVISED_VIEWS = {'weasel2', 'supervised_intervals', 'ed_medoid', 'dtw_medoid',
                    'inceptiontime'}
MODELS = ['ridge_cv', 'extra_trees', 'rf', 'rotation_forest', 'logreg', 'svm', 'knn1']

# Published classifiers, run end to end; tsml-eval file name for resample 0.
PUBLISHED = {
    'multirocket': 'MR', 'minirocket': 'MiniROCKET', 'hydra': 'Hydra',
    'multirocket_hydra': 'MR-Hydra', 'quant': 'QUANT', 'rdst': 'RDST',
    'weasel2': 'WEASEL-2', 'catch22': 'Catch22', 'freshprince': 'FreshPRINCE',
    'drcif': 'DrCIF', 'rstsf': 'R-STSF', 'tsf': None, 'rocket': None,
    'knn1_ed': None, 'knn1_dtw': '1NN-DTW', 'inceptiontime': 'InceptionTime',
}
# Published numbers not in tsml-eval, with their source.
OTHER_PUBLISHED = {
    'knn1_ed': (0.7117, 'UCR 2018 briefing document, 1NN-ED error 0.2883'),
    'rocket': (None, 'tsml-eval has no ROCKET file; MultiRocket README gives only a '
               '30-resample mean, 0.7517'),
    'tsf': (None, 'tsml-eval has no TSF file; MultiRocket README gives only a 30-resample '
            'mean, 0.7456'),
}
TSML = ('https://raw.githubusercontent.com/time-series-machine-learning/tsml-eval/main/'
        'results/classification/Univariate/{}_accuracy.csv')


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg}', flush=True)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


# ── data ─────────────────────────────────────────────────────────────────────

def stage_data():
    from aeon.datasets import load_classification
    raw_dir = DATA / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    Xtr, ytr = load_classification('Crop', split='train', extract_path=str(raw_dir))
    Xte, yte = load_classification('Crop', split='test', extract_path=str(raw_dir))
    Xtr, Xte = Xtr[:, 0, :].astype(np.float64), Xte[:, 0, :].astype(np.float64)
    assert Xtr.shape == (7200, LENGTH) and Xte.shape == (16800, LENGTH), (Xtr.shape, Xte.shape)
    classes = sorted(set(ytr) | set(yte), key=lambda c: int(c))
    cmap = {c: i for i, c in enumerate(classes)}
    ytr = np.array([cmap[c] for c in ytr])
    yte = np.array([cmap[c] for c in yte])
    files = {}
    for p in sorted(raw_dir.rglob('*')):
        if p.is_file():
            files[str(p.relative_to(raw_dir))] = hashlib.md5(p.read_bytes()).hexdigest()
    train_rows = {r.tobytes() for r in Xtr}
    dup_test = int(sum(r.tobytes() in train_rows for r in Xte))
    info = {
        'n_train': len(Xtr), 'n_test': len(Xte), 'length': LENGTH, 'n_classes': len(classes),
        'train_per_class': np.bincount(ytr).tolist(), 'test_per_class': np.bincount(yte).tolist(),
        'test_rows_identical_to_a_training_row': dup_test,
        'train_rows_duplicated_within_train': int(len(Xtr) - len(train_rows)),
        'row_mean_range': [float(np.r_[Xtr, Xte].mean(1).min()), float(np.r_[Xtr, Xte].mean(1).max())],
        'row_std_range': [float(np.r_[Xtr, Xte].std(1).min()), float(np.r_[Xtr, Xte].std(1).max())],
        'value_range': [float(np.r_[Xtr, Xte].min()), float(np.r_[Xtr, Xte].max())],
        'nan_values': int(np.isnan(np.r_[Xtr, Xte]).sum()),
        'files_md5': files,
    }
    np.savez(DATA / 'crop.npz', X=np.r_[Xtr, Xte], y=np.r_[ytr, yte],
             train=np.r_[np.ones(len(Xtr), bool), np.zeros(len(Xte), bool)])
    write_json(DATA / 'crop.json', info)
    log(json.dumps({k: v for k, v in info.items() if k != 'files_md5'}))


def load(debug=False):
    d = np.load(DATA / 'crop.npz')
    X, y, train = d['X'], d['y'], d['train']
    if debug:
        keep = train.copy()
        keep[np.where(train)[0][1000:]] = False
        train = keep
    return X, y, train


# ── views ────────────────────────────────────────────────────────────────────

def harmonic(X, k):
    t = np.arange(X.shape[1])
    w = 2 * np.pi / X.shape[1]
    cols = [np.ones_like(t, float)]
    for j in range(1, k + 1):
        cols += [np.cos(j * w * t), np.sin(j * w * t)]
    A = np.stack(cols, 1)
    coef, *_ = np.linalg.lstsq(A, X.T, rcond=None)
    return coef.T


def medoids(X, y, train, dist):
    out = []
    for c in np.unique(y[train]):
        Xc = X[train & (y == c)]
        D = dist(Xc, Xc)
        out.append(Xc[np.argmin(D.sum(1))])
    return np.stack(out)


def ed(A, B):
    return np.sqrt(np.maximum(((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None]
                               - 2 * A @ B.T), 0))


def dtw(A, B):
    from aeon.distances import dtw_pairwise_distance
    return dtw_pairwise_distance(A[:, None, :], B[:, None, :], window=None)


def phenology(X, smoother, window, threshold):
    """TIMESAT-style metrics for one season per series (Jonsson and Eklundh
    2004): smoothing, then start and end of season where the smoothed curve
    crosses base + threshold x amplitude on the left and right of the peak."""
    from scipy.signal import savgol_filter
    if smoother != 'savgol':
        raise SystemExit(f'phenology smoother {smoother!r}: only savgol is written')
    S = savgol_filter(X, int(window), 2, axis=1, mode='interp')
    out = []
    t = np.arange(X.shape[1], dtype=float)
    for s in S:
        p = int(np.argmax(s))
        left_base, right_base = s[:p + 1].min(), s[p:].min()
        base = (left_base + right_base) / 2
        amp = s[p] - base
        lv = left_base + threshold * (s[p] - left_base)
        rv = right_base + threshold * (s[p] - right_base)
        li = np.where(s[:p + 1] <= lv)[0]
        sos = float(li[-1]) if len(li) else 0.0
        ri = np.where(s[p:] <= rv)[0]
        eos = float(p + ri[0]) if len(ri) else float(len(s) - 1)
        g = np.gradient(s)
        a, b = int(sos), int(np.ceil(eos))
        large = float(np.trapz(s[a:b + 1], t[a:b + 1])) if b > a else 0.0
        small = float(np.trapz(s[a:b + 1] - base, t[a:b + 1])) if b > a else 0.0
        out.append([sos, eos, eos - sos, base, float(p), s[p], amp,
                    float(g[a:p + 1].max()) if p >= a else 0.0,
                    float(-g[p:b + 1].min()) if b >= p else 0.0,
                    large, small, lv, rv])
    return np.array(out)


PHENOLOGY_NAMES = ['start', 'end', 'length', 'base', 'peak_time', 'peak_value', 'amplitude',
                   'left_derivative', 'right_derivative', 'large_integral', 'small_integral',
                   'start_value', 'end_value']


def inceptiontime_net(n_classes, n_filters=32, depth=6, kernel=40, bottleneck=32):
    import torch
    import torch.nn as nn

    class Module(nn.Module):
        def __init__(self, c_in):
            super().__init__()
            self.bott = (nn.Conv1d(c_in, bottleneck, 1, bias=False) if c_in > 1 else None)
            c = bottleneck if c_in > 1 else c_in
            self.convs = nn.ModuleList([nn.Conv1d(c, n_filters, k, padding='same', bias=False)
                                        for k in (kernel, kernel // 2, kernel // 4)])
            self.pool = nn.MaxPool1d(3, stride=1, padding=1)
            self.pconv = nn.Conv1d(c_in, n_filters, 1, bias=False)
            self.bn = nn.BatchNorm1d(4 * n_filters, eps=1e-3, momentum=0.01)

        def forward(self, x):
            z = self.bott(x) if self.bott is not None else x
            out = [cv(z) for cv in self.convs] + [self.pconv(self.pool(x))]
            return torch.relu(self.bn(torch.cat(out, 1)))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.mods = nn.ModuleList()
            self.shorts = nn.ModuleList()
            c = 1
            res_c = 1
            for d in range(depth):
                self.mods.append(Module(c))
                c = 4 * n_filters
                if d % 3 == 2:
                    self.shorts.append(nn.Sequential(
                        nn.Conv1d(res_c, c, 1, padding='same', bias=False),
                        nn.BatchNorm1d(c, eps=1e-3, momentum=0.01)))
                    res_c = c
            self.fc = nn.Linear(c, n_classes)

        def embed(self, x):
            res = x
            si = 0
            for d, m in enumerate(self.mods):
                x = m(x)
                if d % 3 == 2:
                    x = torch.relu(x + self.shorts[si](res))
                    res = x
                    si += 1
            return x.mean(-1)

        def forward(self, x):
            return self.fc(self.embed(x))

    net = Net()
    for mod in net.modules():
        if isinstance(mod, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(mod.weight)   # keras glorot_uniform
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
    return net


def train_inceptiontime(Xtr, ytr, n_classes, seed, epochs, ckpt):
    """One network as hfawaz/InceptionTime classifiers/inception.py trains it:
    Adam(1e-3), batch min(64, n // 10), ReduceLROnPlateau on training loss
    (factor 0.5, patience 50, min 1e-4), the weights of the lowest training
    loss kept (ModelCheckpoint save_best_only on 'loss'). Resumes from ckpt."""
    import torch
    torch.manual_seed(seed)
    net = inceptiontime_net(n_classes)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, eps=1e-7)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=50, min_lr=1e-4)
    start, best, best_state = 0, np.inf, None
    if ckpt.exists():
        st = torch.load(ckpt)
        net.load_state_dict(st['net'])
        opt.load_state_dict(st['opt'])
        sched.load_state_dict(st['sched'])
        start, best, best_state = st['epoch'], st['best'], st['best_state']
    Xt = torch.as_tensor(Xtr[:, None, :], dtype=torch.float32)
    yt = torch.as_tensor(ytr, dtype=torch.long)
    bs = int(min(64, len(Xt) // 10))
    g = torch.Generator().manual_seed(seed)
    for _ in range(start):
        torch.randperm(len(Xt), generator=g)
    loss_fn = torch.nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(start, epochs):
        net.train()
        perm = torch.randperm(len(Xt), generator=g)
        tot = 0.0
        for st_ in range(0, len(Xt), bs):
            idx = perm[st_:st_ + bs]
            opt.zero_grad()
            loss = loss_fn(net(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        ep_loss = tot / len(Xt)
        sched.step(ep_loss)
        if ep_loss < best:
            best = ep_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if (ep + 1) % 25 == 0 or ep + 1 == epochs:
            torch.save({'net': net.state_dict(), 'opt': opt.state_dict(),
                        'sched': sched.state_dict(), 'epoch': ep + 1, 'best': best,
                        'best_state': best_state}, ckpt)
            log(f'inceptiontime seed {seed}: epoch {ep + 1}/{epochs}, loss {ep_loss:.4f}, '
                f'{(time.time() - t0) / (ep + 1 - start):.1f} s per epoch')
    net.load_state_dict(best_state)
    net.eval()
    return net


def inceptiontime_ensemble(X, y, train, epochs, debug):
    """Five networks (the paper's ensemble). Returns (embedding of all rows,
    averaged class probabilities on all rows)."""
    import torch
    ck = DATA / 'inceptiontime_ckpt' / ('debug' if debug else 'full')
    ck.mkdir(parents=True, exist_ok=True)
    n_classes = int(y.max() + 1)
    embs, probs = [], []
    for seed in range(5):
        net = train_inceptiontime(X[train], y[train], n_classes, seed, epochs,
                                  ck / f'net{seed}.pt')
        with torch.no_grad():
            Xa = torch.as_tensor(X[:, None, :], dtype=torch.float32)
            e = torch.cat([net.embed(Xa[i:i + 2048]) for i in range(0, len(Xa), 2048)])
            embs.append(e.numpy())
            probs.append(torch.softmax(net.fc(e), 1).numpy())
    return np.concatenate(embs, 1), np.mean(probs, 0)


def build(view, debug, pheno=None, it_epochs=1500):
    X, y, train = load(debug)
    X3 = X[:, None, :]
    meta = {'view': view, 'fit_on': 'training rows', 'debug': debug}
    t0 = time.time()
    if view == 'raw':
        Z = X
    elif view == 'diff1':
        Z = np.diff(X, axis=1)
    elif view == 'diff2':
        Z = np.diff(X, 2, axis=1)
    elif view == 'periodogram':
        Z = np.abs(np.fft.rfft(X - X.mean(1, keepdims=True), axis=1))[:, 1:]
    elif view in ('harmonic2', 'harmonic3'):
        Z = harmonic(X, int(view[-1]))
    elif view == 'catch24':
        import pycatch22
        Z = np.array([pycatch22.catch22_all(list(map(float, x)), catch24=True)['values']
                      for x in X])
    elif view == 'tsfresh':
        import pandas as pd
        from tsfresh import extract_features
        from tsfresh.feature_extraction import ComprehensiveFCParameters
        long = pd.DataFrame({'id': np.repeat(np.arange(len(X)), LENGTH),
                             'time': np.tile(np.arange(LENGTH), len(X)), 'value': X.ravel()})
        F = extract_features(long, column_id='id', column_sort='time', column_value='value',
                             default_fc_parameters=ComprehensiveFCParameters(),
                             disable_progressbar=True, n_jobs=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
        Z = F.sort_index().to_numpy(np.float64)
        meta['names'] = list(map(str, F.columns))
    elif view in ('rocket', 'minirocket', 'multirocket', 'hydra'):
        from aeon.transformations.collection import convolution_based as cb
        cls = {'rocket': cb.Rocket, 'minirocket': cb.MiniRocket,
               'multirocket': cb.MultiRocket, 'hydra': cb.HydraTransformer}[view]
        T = cls(random_state=SEED).fit(X3[train])
        Z = T.transform(X3)
        Z = Z.numpy() if hasattr(Z, 'numpy') else np.asarray(Z)
    elif view == 'quant':
        from aeon.transformations.collection.interval_based import QUANTTransformer
        Z = np.asarray(QUANTTransformer().fit(X3[train]).transform(X3))
    elif view == 'rdst':
        from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
        T = RandomDilatedShapeletTransform(random_state=SEED).fit(X3[train], y[train])
        Z = np.asarray(T.transform(X3))
    elif view == 'weasel2':
        from aeon.classification.dictionary_based._weasel_v2 import WEASELTransformerV2
        T = WEASELTransformerV2(random_state=SEED).fit(X3[train], y[train])
        Z = T.transform(X3)
        Z = Z.toarray() if hasattr(Z, 'toarray') else np.asarray(Z)
    elif view == 'random_intervals':
        from aeon.transformations.collection.interval_based import RandomIntervals
        Z = np.asarray(RandomIntervals(random_state=SEED).fit(X3[train]).transform(X3))
    elif view == 'supervised_intervals':
        from aeon.transformations.collection.interval_based import SupervisedIntervals
        T = SupervisedIntervals(random_state=SEED).fit(X3[train], y[train])
        Z = np.asarray(T.transform(X3))
    elif view in ('ed_medoid', 'dtw_medoid'):
        dist = ed if view == 'ed_medoid' else dtw
        M = medoids(X, y, train, dist)
        Z = np.concatenate([dist(X[i:i + 2000], M) for i in range(0, len(X), 2000)])
    elif view == 'moment':
        import torch
        from momentfm import MOMENTPipeline
        m = MOMENTPipeline.from_pretrained('AutonLab/MOMENT-1-large',
                                           model_kwargs={'task_name': 'embedding'})
        m.init()
        m.eval()
        P = np.zeros((len(X), 512), np.float32)
        P[:, :LENGTH] = X
        mask = np.zeros((len(X), 512), np.float32)
        mask[:, :LENGTH] = 1
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), 64):
                outs.append(m(x_enc=torch.as_tensor(P[i:i + 64, None, :]),
                              input_mask=torch.as_tensor(mask[i:i + 64])).embeddings.numpy())
        Z = np.concatenate(outs)
        meta.update(model='AutonLab/MOMENT-1-large', input='zero-padded to 512, input mask')
    elif view == 'mantis':
        from mantis.architecture import Mantis8M
        from mantis.trainer import MantisTrainer
        net = Mantis8M(device='cpu').from_pretrained('paris-noah/Mantis-8M')
        R = np.stack([np.interp(np.linspace(0, LENGTH - 1, 512), np.arange(LENGTH), x) for x in X])
        Z = np.asarray(MantisTrainer(device='cpu', network=net).transform(R[:, None, :].astype(np.float32)))
        meta.update(model='paris-noah/Mantis-8M', input='linear interpolation to 512')
    elif view == 'chronos2':
        import torch
        from chronos import Chronos2Pipeline
        pipe = Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map='cpu',
                                                torch_dtype=torch.float32)
        outs = []
        for i in range(0, len(X), 64):
            embs, _ = pipe.embed(torch.as_tensor(X[i:i + 64, None, :], dtype=torch.float32))
            outs.append(np.stack([e[0].detach().cpu().float().numpy().mean(0) for e in embs]))
        Z = np.concatenate(outs)
        meta.update(model='amazon/chronos-2', pool='mean over encoder tokens')
    elif view == 'ts2vec':
        import torch
        d = os.environ.get('KIRBY_TS2VEC_DIR') or '/Volumes/seagate/benchmarks_protocol/opssat_ad/ext/ts2vec'
        sys.path.insert(0, d)
        from ts2vec import TS2Vec
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = TS2Vec(input_dims=1, output_dims=320, device='cpu')
        model.fit(X[train][:, :, None].astype(np.float32), verbose=False,
                  n_epochs=(5 if debug else None))
        Z = model.encode(X[:, :, None].astype(np.float32), encoding_window='full_series')
        meta.update(code=d, n_epochs='ts2vec default (n_iters by dataset size)')
    elif view == 'inceptiontime':
        Z, _ = inceptiontime_ensemble(X, y, train, 3 if debug else it_epochs, debug)
        meta.update(networks=5, epochs=3 if debug else it_epochs)
    elif view == 'phenology':
        if pheno is None or any(v is None for v in pheno):
            raise SystemExit('phenology needs --pheno-smoother, --pheno-window and '
                             '--pheno-threshold; none has a value yet')
        Z = phenology(X, *pheno)
        meta.update(names=PHENOLOGY_NAMES, smoother=pheno[0], window=pheno[1],
                    threshold=pheno[2])
    else:
        raise SystemExit(f'unknown view {view}')
    Z = np.asarray(Z, np.float64)
    Z[~np.isfinite(Z)] = np.nan
    if np.isnan(Z).any():   # training-row median, as b09 does
        med = np.nanmedian(Z[train], 0)
        med[~np.isfinite(med)] = 0.0
        idx = np.where(np.isnan(Z))
        meta['nan_filled_with_training_median'] = int(len(idx[0]))
        Z[idx] = med[idx[1]]
    meta.update(shape=list(Z.shape), seconds=time.time() - t0)
    VIEWS_DIR.mkdir(parents=True, exist_ok=True)
    tag = '.debug' if debug else ''
    np.save(VIEWS_DIR / f'{view}{tag}.npy', Z)
    write_json(VIEWS_DIR / f'{view}{tag}.json', meta)
    log(f'view {view}: {Z.shape} in {meta["seconds"]:.0f} s')


# ── models ───────────────────────────────────────────────────────────────────

def make_model(name):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == 'ridge_cv':
        from sklearn.linear_model import RidgeClassifierCV
        return make_pipeline(StandardScaler(with_mean=False),
                             RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))
    if name == 'extra_trees':
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=200, max_features=0.1, criterion='entropy',
                                    random_state=SEED, n_jobs=-1)
    if name == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    if name == 'rotation_forest':
        from aeon.classification.sklearn import RotationForestClassifier
        return RotationForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    if name == 'logreg':
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if name == 'knn1':
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=1)
    raise SystemExit(f'unknown model {name}')


def fit_svm(Xtr, ytr):
    """ts2vec/tasks/_eval_protocols.py fit_svm, as published (MIT)."""
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.svm import SVC
    nb = np.unique(ytr, return_counts=True)[1].shape[0]
    size = Xtr.shape[0]
    svm = SVC(C=np.inf, gamma='scale')
    if size // nb < 5 or size < 50:
        return svm.fit(Xtr, ytr)
    grid = GridSearchCV(svm, {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, np.inf],
                              'kernel': ['rbf'], 'degree': [3], 'gamma': ['scale'], 'coef0': [0],
                              'shrinking': [True], 'probability': [False], 'tol': [0.001],
                              'cache_size': [200], 'class_weight': [None], 'verbose': [False],
                              'max_iter': [10000000], 'decision_function_shape': ['ovr'],
                              'random_state': [None]}, cv=5, n_jobs=5)
    if size > 10000:
        split = train_test_split(Xtr, ytr, train_size=10000, random_state=0, stratify=ytr)
        Xtr, ytr = split[0], split[2]
    grid.fit(Xtr, ytr)
    return grid.best_estimator_


def eval_file(view, model, plus, debug):
    tail = f'__plus_{plus}' if plus else ''
    return RESULTS / ('debug' if debug else 'full') / f'{view}__{model}{tail}.json'


def view_matrix(view, debug):
    tag = '.debug' if debug else ''
    return np.load(VIEWS_DIR / f'{view}{tag}.npy')


def evaluate(view, model, plus, debug):
    from sklearn.metrics import accuracy_score, f1_score
    X, y, train = load(debug)
    test = np.zeros_like(train)
    test[len(train) - 16800:] = True
    Z = view_matrix(view, debug)
    if plus:
        Z = np.concatenate([Z, view_matrix(plus, debug)], 1)
    t0 = time.time()
    if model == 'svm':
        m = fit_svm(Z[train], y[train])
    else:
        m = make_model(model).fit(Z[train], y[train])
    pred = np.concatenate([m.predict(Z[test][i:i + 4000]) for i in range(0, test.sum(), 4000)])
    out = eval_file(view, model, plus, debug)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_suffix('.pred.npy'), pred.astype(np.int16))
    res = {'view': view, 'model': model, 'plus': plus, 'n_features': int(Z.shape[1]),
           'n_train': int(train.sum()), 'n_test': int(test.sum()),
           'accuracy': float(accuracy_score(y[test], pred)),
           'macro_f1': float(f1_score(y[test], pred, average='macro')),
           'test_rows_wrong': int((pred != y[test]).sum()),
           'seconds': time.time() - t0, 'debug': debug, 'result_status': 'ok'}
    write_json(out, res)
    log(f'{view}{"+" + plus if plus else ""} {model}: accuracy {res["accuracy"]:.4f} '
        f'({res["test_rows_wrong"]} of {res["n_test"]} wrong) in {res["seconds"]:.0f} s')


# ── published classifiers ────────────────────────────────────────────────────

def published(name, debug, it_epochs=1500):
    from sklearn.metrics import accuracy_score
    X, y, train = load(debug)
    test = np.zeros_like(train)
    test[len(train) - 16800:] = True
    X3 = X[:, None, :]
    nj = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    t0 = time.time()
    if name == 'inceptiontime':
        _, P = inceptiontime_ensemble(X, y, train, 3 if debug else it_epochs, debug)
        pred = P[test].argmax(1)
    else:
        from aeon.classification import convolution_based as cb, interval_based as ib
        from aeon.classification import dictionary_based as db, feature_based as fb
        from aeon.classification import shapelet_based as sb, distance_based as distb
        clf = {
            'multirocket': lambda: cb.MultiRocketClassifier(random_state=SEED, n_jobs=nj),
            'minirocket': lambda: cb.MiniRocketClassifier(random_state=SEED, n_jobs=nj),
            'rocket': lambda: cb.RocketClassifier(random_state=SEED, n_jobs=nj),
            'hydra': lambda: cb.HydraClassifier(random_state=SEED, n_jobs=nj),
            'multirocket_hydra': lambda: cb.MultiRocketHydraClassifier(random_state=SEED, n_jobs=nj),
            'quant': lambda: ib.QUANTClassifier(random_state=SEED),
            'rdst': lambda: sb.RDSTClassifier(random_state=SEED, n_jobs=nj),
            'weasel2': lambda: db.WEASEL_V2(random_state=SEED, n_jobs=nj),
            'catch22': lambda: fb.Catch22Classifier(random_state=SEED, n_jobs=nj),
            'freshprince': lambda: fb.FreshPRINCEClassifier(random_state=SEED, n_jobs=nj),
            'drcif': lambda: ib.DrCIFClassifier(random_state=SEED, n_jobs=nj),
            'rstsf': lambda: ib.RSTSF(random_state=SEED, n_jobs=nj),
            'tsf': lambda: ib.TimeSeriesForestClassifier(random_state=SEED, n_jobs=nj),
            'knn1_ed': lambda: distb.KNeighborsTimeSeriesClassifier(n_neighbors=1, distance='euclidean', n_jobs=nj),
            'knn1_dtw': lambda: distb.KNeighborsTimeSeriesClassifier(n_neighbors=1, distance='dtw', n_jobs=nj),
        }[name]()
        clf.fit(X3[train], y[train])
        pred = np.concatenate([clf.predict(X3[test][i:i + 4000])
                               for i in range(0, test.sum(), 4000)])
    acc = float(accuracy_score(y[test], pred))
    out = RESULTS / ('debug' if debug else 'full') / 'published' / f'{name}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_suffix('.pred.npy'), pred.astype(np.int16))
    write_json(out, {'classifier': name, 'accuracy': acc,
                     'test_rows_wrong': int((pred != y[test]).sum()), 'n_test': int(test.sum()),
                     'seconds': time.time() - t0, 'debug': debug, 'result_status': 'ok',
                     'tsml_eval_file': PUBLISHED.get(name)})
    log(f'published {name}: accuracy {acc:.4f} in {time.time() - t0:.0f} s')


def tsml_resample0(name):
    """(resample-0 accuracy, 30-resample mean) from tsml-eval, cached."""
    f = PUBLISHED.get(name)
    if not f:
        return None, None
    cache = DATA / 'tsml_eval' / f'{f}_accuracy.csv'
    if not cache.exists():
        import urllib.request
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(urllib.request.urlopen(TSML.format(f), timeout=60).read())
    for line in cache.read_text().splitlines():
        if line.startswith('Crop,'):
            v = [float(x) for x in line.split(',')[1:] if x]
            return v[0], float(np.mean(v))
    return None, None


# ── summary ──────────────────────────────────────────────────────────────────

def linear_cka(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    return float(np.linalg.norm(A.T @ B, 'fro') ** 2
                 / (np.linalg.norm(A.T @ A, 'fro') * np.linalg.norm(B.T @ B, 'fro') + 1e-30))


def summarise(debug):
    sub = RESULTS / ('debug' if debug else 'full')
    X, y, train = load(debug)
    yte = y[len(y) - 16800:]
    L = ['# UCR Crop: published classifiers and view screen', '',
         f'Fixed split: {int(train.sum())} training rows, 16,800 test rows, 24 classes. '
         'Accuracy on the test rows. One test row is 1/16,800 = 0.00006 accuracy.', '']
    pub = [json.loads(p.read_text()) for p in sorted((sub / 'published').glob('*.json'))]
    if pub:
        L += ['## Published classifiers', '',
              'One row is one classifier, run end to end with aeon 1.3 on the fixed split. '
              'tsml-eval resample 0 is the same split; its 30-resample mean is beside it.', '',
              '| classifier | accuracy here | tsml-eval resample 0 | tsml-eval 30-resample mean | hours |',
              '|---|---|---|---|---|']
        for r in pub:
            r0, mean = tsml_resample0(r['classifier'])
            other = OTHER_PUBLISHED.get(r['classifier'])
            if r0 is None and other and other[0] is not None:
                r0 = other[0]
            L.append(f"| {r['classifier']} | {r['accuracy']:.4f} | "
                     f"{'' if r0 is None else f'{r0:.4f}'} | {'' if mean is None else f'{mean:.4f}'} | "
                     f"{r['seconds'] / 3600:.2f} |")
        L.append('')
    res = [json.loads(p.read_text()) for p in sorted(sub.glob('*__*.json'))]
    if not res:
        (sub / 'summary.md').write_text('\n'.join(L) + '\n')
        return
    views = [v for v in VIEWS + SETTING_VIEWS if any(r['view'] == v for r in res)]
    models = [m for m in MODELS if any(r['model'] == m for r in res)]
    pluses = sorted({r['plus'] for r in res if r['plus']})

    def cell(v, m, p):
        r = [x for x in res if x['view'] == v and x['model'] == m and x['plus'] == p]
        return f"{r[0]['accuracy']:.4f}" if r else ''

    L += ['## Each view alone', '', 'One row is one view; a cell is test accuracy.', '',
          '| view | features | ' + ' | '.join(models) + ' |', '|---|---|' + '---|' * len(models)]
    for v in views:
        nf = next((x['n_features'] for x in res if x['view'] == v and not x['plus']), '')
        L.append(f'| {v} | {nf} | ' + ' | '.join(cell(v, m, None) for m in models) + ' |')
    for p in pluses:
        L += ['', f'## Each view with {p} appended', '',
              'One row is one view; a cell is test accuracy alone, then with the block appended.', '',
              '| view | ' + ' | '.join(models) + ' |', '|---|' + '---|' * len(models)]
        for v in views:
            if v == p:
                continue
            L.append(f'| {v} | ' + ' | '.join(
                f'{cell(v, m, None)} / {cell(v, m, p)}' if cell(v, m, p) else '' for m in models) + ' |')
    for m in models:
        preds = {v: np.load(eval_file(v, m, None, debug).with_suffix('.pred.npy'))
                 for v in views if eval_file(v, m, None, debug).with_suffix('.pred.npy').exists()}
        if len(preds) < 2:
            continue
        vs = list(preds)
        err = {v: preds[v] != yte for v in vs}
        L += ['', f'## Error overlap, {m}', '',
              'Cell: of the test rows either view gets wrong, the percentage both get wrong. '
              'Diagonal: the view\'s test error in %.', '',
              '| | ' + ' | '.join(vs) + ' |', '|---|' + '---|' * len(vs)]
        for a in vs:
            L.append(f'| {a} | ' + ' | '.join(
                f'{100 * err[a].mean():.1f}' if a == b else
                f'{100 * (err[a] & err[b]).sum() / max((err[a] | err[b]).sum(), 1):.0f}'
                for b in vs) + ' |')
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(np.where(~train)[0], 3000, replace=False))
    mats = {}
    for v in views:
        Z = view_matrix(v, debug)[idx].astype(np.float64)
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
        mats[v] = Z
    L += ['', '## Linear CKA between views', '',
          'Linear centred kernel alignment between standardised view matrices on 3,000 test rows '
          '(seed 0). 1 = the same information up to a linear map, 0 = unrelated.', '',
          '| | ' + ' | '.join(views) + ' |', '|---|' + '---|' * len(views)]
    for a in views:
        L.append(f'| {a} | ' + ' | '.join(f'{linear_cka(mats[a], mats[b]):.2f}' for b in views) + ' |')
    tmp = sub / 'summary.md.tmp'
    tmp.write_text('\n'.join(L) + '\n')
    tmp.replace(sub / 'summary.md')
    log(f'summary: {sub / "summary.md"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['data', 'published', 'build', 'evaluate', 'summarise'])
    ap.add_argument('--views', default='')
    ap.add_argument('--models', default='')
    ap.add_argument('--plus', default=None, help='a view appended to every view evaluated')
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--inceptiontime-epochs', type=int, default=1500,
                    help='the published code trains 1,500 epochs')
    ap.add_argument('--pheno-smoother', default=None)
    ap.add_argument('--pheno-window', type=int, default=None)
    ap.add_argument('--pheno-threshold', type=float, default=None)
    a = ap.parse_args()
    views = [v for v in a.views.split(',') if v]
    models = [m for m in a.models.split(',') if m]
    if a.stage == 'data':
        stage_data()
    elif a.stage == 'published':
        for m in models:
            published(m, a.debug, a.inceptiontime_epochs)
    elif a.stage == 'build':
        for v in views:
            build(v, a.debug, (a.pheno_smoother, a.pheno_window, a.pheno_threshold),
                  a.inceptiontime_epochs)
    elif a.stage == 'evaluate':
        for v in views:
            for m in models:
                evaluate(v, m, a.plus, a.debug)
    else:
        summarise(a.debug)


if __name__ == '__main__':
    main()
