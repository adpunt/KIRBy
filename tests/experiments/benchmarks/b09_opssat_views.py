#!/usr/bin/env python3
"""OPS-SAT-AD view and model screen: every candidate representation of one
telemetry segment, under the six supervised models of the paper's Table 3,
on the paper's fixed `train` split, with test-set metrics, test predictions
kept, and correlations between views.

Object: one segment of segments.csv (2,123; 8 to 1,040 points; 1 s or 5 s
sampling), label nominal (0) or anomalous (1). Split: the `train` column
(1,594 train, 529 test). One universal detector over all nine channels
(owner, 2026-10-30). Standardisation as in the authors' notebook: a
StandardScaler fit on the nominal training rows, applied to every view.

Views (one vector per segment). "Code" = ported or called from the named
package; "paper" = written from a paper's text; "none" = no paper.
Variable length is handled two ways where a view needs a fixed length:
`_512` = linear interpolation to 512 points (the length the pretrained
encoders take; 69 segments are longer than 512 and are shrunk); `_pad` =
padded to 1,040 with the segment's own mean (the bake-off's padding, without
its added noise).

    name              size    source
    feat18              18    Ruszczak 2025, shipped dataset.csv (code)
    catch24             24    Lubba 2019, pycatch22 (code); dfa needs 79 points
                              and is NaN below that, then imputed with the
                              training median
    tsfresh           ~794    Christ 2018, tsfresh ComprehensiveFCParameters
                              (code); NaN imputed with the training median
    minirocket_512/_pad 9,996 Dempster 2021, aeon (code); fit on training rows
    multirocket_512/_pad 49,728 Tan 2022, aeon (code); fit on training rows
    hydra_512/_pad    6,144    Dempster 2023, aeon (code); fit on training rows
    quant_512         length-proportional  Dempster 2023, aeon (code)
    moment_512        1,024    Goswami 2024, AutonLab/MOMENT-1-large frozen,
                              pooled embedding (code)
    mantis_512          256    Feofanov 2026, paris-noah/Mantis-8M frozen (code)
    chronos2_512        768    Ansari 2024, amazon/chronos-2 frozen encoder,
                              mean over its 34 tokens (code; the survey reports
                              this use only via MantisV2's paper)
    ts2vec_emb          320    Yue 2022, github.com/yuezhihan/ts2vec (MIT),
                              trained on the 1,594 training segments only,
                              NaN-padded to 1,040, full-series encoding
    resampled_512       512    none: the raw values interpolated to 512
    spectrum_512        257    none: rFFT magnitude of resampled_512
    gaf_resnet50      2,048    Goetze 2026 on ESA-ADB (paper): Gramian angular
                              summation field of resampled_512, 224 x 224,
                              ImageNet ResNet50 pooled output
    context20            20    none: the previous segment on the same channel
                              (by start time): its 18 shipped features, the
                              gap in seconds to it, and a has-previous flag
                              (zeros and flag 0 for a channel's first
                              segment). Its label is never used.

Models (Table 3's supervised six). The paper's FCNN has no shipped code;
the dense network here is this file's own (two hidden layers of 64, batch
norm, ReLU, dropout 0.2, Adam 1e-3, 200 epochs, batch 32, cross-entropy,
seed 2137) and is marked so.

    fcnn      see above (not the paper's architecture)
    xgbod     pyod XGBOD(random_state=2137), defaults (paper: PyOD defaults);
              fitted in a torch-free subprocess, see _xgb_worker.py
    rf        RandomForestClassifier(max_depth=25, random_state=2137)
              (ICCS 2023: depth 25; tree count not stated, default 100)
    adaboost  AdaBoostClassifier(random_state=2137) (the notebook)
    lsvc      LinearSVC(max_iter=10000), defaults
    logreg    LogisticRegression(max_iter=1000), defaults

Metrics on the 529 test rows, as Table 3: accuracy, MCC, F1, AUC-PR,
AUC-ROC. Reproduction check: feat18 + adaboost gives 0.934 / 0.797, the paper's
row, when the features are kept in float64 (float32 gives 0.940 / 0.814).

    python tests/experiments/benchmarks/b09_opssat_views.py --stage all

Output: /Volumes/seagate/benchmarks_protocol/opssat_ad/views/<view>.npy and
/Volumes/seagate/benchmarks_protocol/results/b09_opssat_views/ (one JSON
per view x model, summary.md, summary.json).
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.preprocessing import StandardScaler

OPSSAT_DIR = Path('/Volumes/seagate/benchmarks_protocol/opssat_ad')
RAW_DIR = OPSSAT_DIR / 'raw'
VIEWS_DIR = OPSSAT_DIR / 'views'
EXT_DIR = OPSSAT_DIR / 'ext'
RESULTS_DIR = Path('/Volumes/seagate/benchmarks_protocol/results/b09_opssat_views')
SEED = 2137
FIXED = 512
PAD = 1040

FEAT18 = ["mean", "var", "std", "len", "duration", "len_weighted", "gaps_squared",
          "var_div_duration", "var_div_len", "n_peaks", "smooth10_n_peaks",
          "smooth20_n_peaks", "diff_peaks", "diff2_peaks", "diff_var", "diff2_var",
          "kurtosis", "skew"]
VIEWS = ['feat18', 'catch24', 'tsfresh', 'minirocket_512', 'minirocket_pad',
         'multirocket_512', 'multirocket_pad', 'hydra_512', 'hydra_pad', 'quant_512',
         'moment_512', 'mantis_512', 'chronos2_512', 'ts2vec_emb', 'resampled_512',
         'spectrum_512', 'gaf_resnet50', 'context20']
MODELS = ['adaboost', 'logreg', 'lsvc', 'rf', 'xgbod', 'fcnn']


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg}', flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'b09_opssat_views.log', 'a') as f:
        f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}\n')


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load():
    """Returns meta (DataFrame, one row per segment in dataset.csv order) and
    series (list of float arrays in the same order)."""
    d = pd.read_csv(RAW_DIR / 'dataset.csv')
    s = pd.read_csv(RAW_DIR / 'segments.csv')
    s['timestamp'] = pd.to_datetime(s['timestamp'])
    series, starts, ends = [], [], []
    g = s.groupby('segment', sort=False)
    for seg in d['segment']:
        x = g.get_group(seg).sort_values('timestamp')
        series.append(x['value'].to_numpy(np.float64))
        starts.append(x['timestamp'].iloc[0])
        ends.append(x['timestamp'].iloc[-1])
    d['start'] = starts
    d['end'] = ends
    assert all(len(x) == n for x, n in zip(series, d['len']))
    return d, series


def resample(x, n=FIXED):
    if len(x) == 1:
        return np.full(n, x[0])
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def pad_mean(x, n=PAD):
    out = np.full(n, x.mean())
    out[:len(x)] = x
    return out


def save_view(name, X, meta_extra=None):
    VIEWS_DIR.mkdir(parents=True, exist_ok=True)
    # float64: rounding feat18 to float32 moves AdaBoost from 0.934 / 0.797 to
    # 0.940 / 0.814 (checked 2026-10-30), so no view is stored in float32
    X = np.asarray(X, np.float64)
    np.save(VIEWS_DIR / f'{name}.npy', X)
    info = {'view': name, 'shape': list(X.shape)}
    if meta_extra:
        info.update(meta_extra)
    with open(VIEWS_DIR / f'{name}.json', 'w') as f:
        json.dump(info, f, indent=1)
    log(f'view {name}: {X.shape}')


def have(name):
    return (VIEWS_DIR / f'{name}.npy').exists()


def impute_train_median(X, train):
    X = np.asarray(X, np.float64)
    X[~np.isfinite(X)] = np.nan
    med = np.nanmedian(X[train], axis=0)
    med[~np.isfinite(med)] = 0.0
    idx = np.where(np.isnan(X))
    X[idx] = med[idx[1]]
    return X


# ----------------------------------------------------------------------------
# Views
# ----------------------------------------------------------------------------

def build_views(which, d, series):
    train = d['train'].to_numpy() == 1
    R512 = np.stack([resample(x) for x in series])
    P1040 = np.stack([pad_mean(x) for x in series])

    if 'feat18' in which and not have('feat18'):
        save_view('feat18', d[FEAT18].to_numpy())

    if 'catch24' in which and not have('catch24'):
        import pycatch22
        rows, names = [], None
        for x in series:
            r = pycatch22.catch22_all(list(map(float, x)), catch24=True)
            names = r['names']
            rows.append(r['values'])
        X = impute_train_median(np.array(rows, np.float64), train)
        save_view('catch24', X, {'names': names})

    if 'tsfresh' in which and not have('tsfresh'):
        from tsfresh import extract_features
        from tsfresh.feature_extraction import ComprehensiveFCParameters
        long = pd.DataFrame({'id': np.repeat(np.arange(len(series)), [len(x) for x in series]),
                             'time': np.concatenate([np.arange(len(x)) for x in series]),
                             'value': np.concatenate(series)})
        F = extract_features(long, column_id='id', column_sort='time', column_value='value',
                             default_fc_parameters=ComprehensiveFCParameters(),
                             disable_progressbar=True, n_jobs=4)
        F = F.sort_index()
        X = impute_train_median(F.to_numpy(np.float64), train)
        save_view('tsfresh', X, {'names': list(map(str, F.columns))})

    kernel_specs = {'minirocket': 'MiniRocket', 'multirocket': 'MultiRocket', 'hydra': 'HydraTransformer'}
    for base, cls in kernel_specs.items():
        for suffix, X3 in (('_512', R512), ('_pad', P1040)):
            name = base + suffix
            if name in which and not have(name):
                from aeon.transformations.collection import convolution_based as cb
                T = getattr(cb, cls)(random_state=SEED)
                X3 = X3[:, None, :]
                T.fit(X3[train])
                Z = T.transform(X3)
                Z = Z.numpy() if hasattr(Z, 'numpy') else np.asarray(Z)
                save_view(name, Z, {'fit_on': 'training rows'})
    if 'quant_512' in which and not have('quant_512'):
        from aeon.transformations.collection.interval_based import QUANTTransformer
        T = QUANTTransformer()
        X3 = R512[:, None, :]
        T.fit(X3[train])
        save_view('quant_512', np.asarray(T.transform(X3)), {'fit_on': 'training rows'})

    if 'moment_512' in which and not have('moment_512'):
        import torch
        from momentfm import MOMENTPipeline
        m = MOMENTPipeline.from_pretrained('AutonLab/MOMENT-1-large',
                                           model_kwargs={'task_name': 'embedding'})
        m.init()
        m.eval()
        outs = []
        with torch.no_grad():
            for st in range(0, len(R512), 32):
                xb = torch.as_tensor(R512[st:st + 32, None, :], dtype=torch.float32)
                outs.append(m(x_enc=xb).embeddings.numpy())
        save_view('moment_512', np.concatenate(outs), {'model': 'AutonLab/MOMENT-1-large'})

    if 'mantis_512' in which and not have('mantis_512'):
        from mantis.architecture import Mantis8M
        from mantis.trainer import MantisTrainer
        net = Mantis8M(device='cpu').from_pretrained('paris-noah/Mantis-8M')
        tr = MantisTrainer(device='cpu', network=net)
        Z = tr.transform(R512[:, None, :].astype(np.float32))
        save_view('mantis_512', np.asarray(Z), {'model': 'paris-noah/Mantis-8M'})

    if 'chronos2_512' in which and not have('chronos2_512'):
        import torch
        from chronos import Chronos2Pipeline
        pipe = Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map='cpu',
                                                torch_dtype=torch.float32)
        outs = []
        for st in range(0, len(R512), 32):
            # embed returns (list of (1, tokens, 768) tensors, list of scales)
            embs, _ = pipe.embed(torch.as_tensor(R512[st:st + 32, None, :], dtype=torch.float32))
            outs.append(np.stack([e[0].detach().cpu().float().numpy().mean(0) for e in embs]))
        save_view('chronos2_512', np.concatenate(outs), {'model': 'amazon/chronos-2', 'pool': 'mean over the 34 encoder tokens'})

    if 'ts2vec_emb' in which and not have('ts2vec_emb'):
        import torch
        sys.path.insert(0, str(EXT_DIR / 'ts2vec'))
        from ts2vec import TS2Vec
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        Xn = np.full((len(series), PAD, 1), np.nan, np.float32)
        for i, x in enumerate(series):
            Xn[i, :len(x), 0] = x
        model = TS2Vec(input_dims=1, output_dims=320, device='cpu')
        t0 = time.time()
        model.fit(Xn[train], verbose=False)
        Z = model.encode(Xn, encoding_window='full_series')
        save_view('ts2vec_emb', Z, {'trained_on': 'training rows', 'seconds': time.time() - t0})

    if 'resampled_512' in which and not have('resampled_512'):
        save_view('resampled_512', R512)
    if 'spectrum_512' in which and not have('spectrum_512'):
        save_view('spectrum_512', np.abs(np.fft.rfft(R512 - R512.mean(1, keepdims=True), axis=1)))

    if 'gaf_resnet50' in which and not have('gaf_resnet50'):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import torch
        import torchvision
        from PIL import Image
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = torch.nn.Identity()
        m.eval()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        imgs = []
        for x in R512:
            s = 2 * (x - x.min()) / (x.max() - x.min() + 1e-12) - 1
            phi = np.arccos(np.clip(s, -1, 1))
            g = np.cos(phi[:, None] + phi[None, :])
            fig = plt.figure(figsize=(2.24, 2.24))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(g, cmap='viridis', vmin=-1, vmax=1)
            ax.axis('off')
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            buf.seek(0)
            imgs.append(np.asarray(Image.open(buf).convert('RGB').resize((224, 224))))
        imgs = np.stack(imgs)
        outs = []
        with torch.no_grad():
            for st in range(0, len(imgs), 32):
                xb = torch.as_tensor(imgs[st:st + 32]).permute(0, 3, 1, 2).float() / 255.0
                outs.append(m((xb - mean) / std).numpy())
        save_view('gaf_resnet50', np.concatenate(outs))

    if 'context20' in which and not have('context20'):
        F = d[FEAT18].to_numpy(np.float64)
        X = np.zeros((len(d), 20))
        order = d.sort_values(['channel', 'start']).index.to_numpy()
        prev = {}
        for i in order:
            ch = d.at[i, 'channel']
            if ch in prev:
                j = prev[ch]
                X[i, :18] = F[j]
                X[i, 18] = (d.at[i, 'start'] - d.at[j, 'end']).total_seconds()
                X[i, 19] = 1.0
            prev[ch] = i
        save_view('context20', X, {'note': 'previous segment on the same channel by start time; label never used'})


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

def run_xgb_worker(kind, X_tr, y_tr, X_te, seed, tmp_dir):
    """xgboost in a torch-free subprocess (see _xgb_worker.py for why)."""
    import subprocess
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    np.save(tmp_dir / 'X_tr.npy', np.asarray(X_tr))
    np.save(tmp_dir / 'y_tr.npy', np.asarray(y_tr))
    np.save(tmp_dir / 'X_te.npy', np.asarray(X_te))
    for f in ('pred.npy', 'score.npy'):
        (tmp_dir / f).unlink(missing_ok=True)
    worker = Path(__file__).resolve().parent / '_xgb_worker.py'
    r = subprocess.run([sys.executable, str(worker), kind, str(tmp_dir), str(seed)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not (tmp_dir / 'pred.npy').exists():
        raise RuntimeError(f'{kind} worker failed (exit {r.returncode}): {r.stderr[-800:]}')
    return np.load(tmp_dir / 'pred.npy'), np.load(tmp_dir / 'score.npy')


def fit_score(model, X_tr, y_tr, X_te):
    """Returns (predicted label, score for the anomalous class)."""
    if model == 'adaboost':
        from sklearn.ensemble import AdaBoostClassifier
        m = AdaBoostClassifier(random_state=SEED).fit(X_tr, y_tr)
        return m.predict(X_te), m.predict_proba(X_te)[:, 1]
    if model == 'logreg':
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
        return m.predict(X_te), m.predict_proba(X_te)[:, 1]
    if model == 'lsvc':
        from sklearn.svm import LinearSVC
        m = LinearSVC(max_iter=10000).fit(X_tr, y_tr)
        return m.predict(X_te), m.decision_function(X_te)
    if model == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(max_depth=25, random_state=SEED, n_jobs=4).fit(X_tr, y_tr)
        return m.predict(X_te), m.predict_proba(X_te)[:, 1]
    if model == 'xgbod':
        return run_xgb_worker('xgbod', X_tr, y_tr, X_te, SEED, OPSSAT_DIR / 'worker_tmp')
    if model == 'fcnn':
        import torch
        torch.manual_seed(SEED)
        d_in = X_tr.shape[1]
        net = torch.nn.Sequential(
            torch.nn.Linear(d_in, 64), torch.nn.BatchNorm1d(64), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 64), torch.nn.BatchNorm1d(64), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 2))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()
        Xt = torch.as_tensor(X_tr, dtype=torch.float32)
        yt = torch.as_tensor(y_tr, dtype=torch.long)
        g = torch.Generator().manual_seed(SEED)
        net.train()
        for _ in range(200):
            perm = torch.randperm(len(Xt), generator=g)
            for st in range(0, len(Xt), 32):
                idx = perm[st:st + 32]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                loss_fn(net(Xt[idx]), yt[idx]).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            p = torch.softmax(net(torch.as_tensor(X_te, dtype=torch.float32)), 1)[:, 1].numpy()
        return (p >= 0.5).astype(int), p
    raise ValueError(model)


def metrics(y, pred, score):
    return {'accuracy': float(accuracy_score(y, pred)), 'mcc': float(matthews_corrcoef(y, pred)),
            'f1': float(f1_score(y, pred)), 'auc_pr': float(average_precision_score(y, score)),
            'auc_roc': float(roc_auc_score(y, score))}


def evaluate(views, models, d):
    y = d['anomaly'].to_numpy()
    train = d['train'].to_numpy() == 1
    for view in views:
        if not have(view):
            log(f'evaluate: {view} missing, skipped')
            continue
        X = np.load(VIEWS_DIR / f'{view}.npy').astype(np.float64)
        sc = StandardScaler().fit(X[train & (y == 0)])  # the notebook: nominal training rows
        Xs = sc.transform(X)
        Xs[~np.isfinite(Xs)] = 0.0
        for model in models:
            out = RESULTS_DIR / f'{view}__{model}.json'
            if out.exists():
                continue
            t0 = time.time()
            try:
                pred, score = fit_score(model, Xs[train], y[train], Xs[~train])
            except Exception as e:  # noqa: BLE001
                log(f'{view} {model}: FAILED {type(e).__name__}: {e}')
                continue
            res = {'view': view, 'model': model, 'n_features': int(X.shape[1]),
                   'n_train': int(train.sum()), 'n_test': int((~train).sum()),
                   'seconds': time.time() - t0, **metrics(y[~train], pred, score)}
            np.save(RESULTS_DIR / f'{view}__{model}.pred.npy', np.stack([pred, score], 1))
            with open(out, 'w') as f:
                json.dump(res, f, indent=1)
            log(f'{view} {model}: acc {res["accuracy"]:.3f} mcc {res["mcc"]:.3f} '
                f'auc_roc {res["auc_roc"]:.3f} ({res["seconds"]:.0f} s)')


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------

def linear_cka(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    return float(np.linalg.norm(A.T @ B, 'fro') ** 2
                 / (np.linalg.norm(A.T @ A, 'fro') * np.linalg.norm(B.T @ B, 'fro') + 1e-30))


def summarise(d):
    y = d['anomaly'].to_numpy()
    train = d['train'].to_numpy() == 1
    yte = y[~train]
    results = [json.load(open(p)) for p in sorted(RESULTS_DIR.glob('*__*.json'))]
    views = [v for v in VIEWS if any(r['view'] == v for r in results)]
    models = [m for m in MODELS if any(r['model'] == m for r in results)]
    lines = ['# OPS-SAT-AD view and model screen', '',
             f'{len(results)} (view, model) runs on the fixed `train` split; metrics on the 529 test segments.',
             'The paper (Table 3, universal): FCNN 0.977 / 0.932, XGBOD 0.966 / 0.897, RF 0.955 / 0.862, '
             'AdaBoost 0.934 / 0.797, LSVC 0.926 / 0.771, LR 0.924 / 0.764 (accuracy / MCC), all on feat18.',
             'The fcnn here is not the paper\'s architecture (no code shipped).', '',
             'One row is one view; a cell is test accuracy / MCC.', '',
             '| view | features | ' + ' | '.join(models) + ' |', '|---|---|' + '---|' * len(models)]
    for v in views:
        cells = []
        for m in models:
            r = [x for x in results if x['view'] == v and x['model'] == m]
            cells.append(f'{r[0]["accuracy"]:.3f} / {r[0]["mcc"]:.3f}' if r else '')
        nf = next(x['n_features'] for x in results if x['view'] == v)
        lines.append(f'| {v} | {nf} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Same runs, test AUC-ROC / AUC-PR.', '',
              '| view | ' + ' | '.join(models) + ' |', '|---|' + '---|' * len(models)]
    for v in views:
        cells = []
        for m in models:
            r = [x for x in results if x['view'] == v and x['model'] == m]
            cells.append(f'{r[0]["auc_roc"]:.3f} / {r[0]["auc_pr"]:.3f}' if r else '')
        lines.append(f'| {v} | ' + ' | '.join(cells) + ' |')
    lines.append('')
    for m in models:
        preds = {v: np.load(RESULTS_DIR / f'{v}__{m}.pred.npy')[:, 0].astype(int)
                 for v in views if (RESULTS_DIR / f'{v}__{m}.pred.npy').exists()}
        if len(preds) < 2:
            continue
        vs = list(preds)
        err = {v: preds[v] != yte for v in vs}
        lines += [f'## error overlap, {m}', '',
                  'Cell: of the test segments that either view gets wrong, the fraction both get wrong '
                  '(Jaccard of the two error sets), in %. Diagonal: test error rate in %.', '',
                  '| | ' + ' | '.join(vs) + ' |', '|---|' + '---|' * len(vs)]
        for a in vs:
            cells = []
            for b in vs:
                if a == b:
                    cells.append(f'{100 * err[a].mean():.1f}')
                else:
                    u = (err[a] | err[b]).sum()
                    cells.append(f'{100 * (err[a] & err[b]).sum() / max(u, 1):.0f}')
            lines.append(f'| {a} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    mats = {}
    for v in views:
        X = np.load(VIEWS_DIR / f'{v}.npy').astype(np.float64)
        X = (X - X.mean(0)) / (X.std(0) + 1e-12)
        X[~np.isfinite(X)] = 0.0
        mats[v] = X
    lines += ['## linear CKA between views', '',
              'Linear centred kernel alignment between standardised view matrices on all 2,123 segments. '
              '1 = the same information up to a linear map, 0 = unrelated.', '',
              '| | ' + ' | '.join(views) + ' |', '|---|' + '---|' * len(views)]
    cka = {}
    for a in views:
        cells = []
        for b in views:
            key = tuple(sorted((a, b)))
            if key not in cka:
                cka[key] = linear_cka(mats[a], mats[b])
            cells.append(f'{cka[key]:.2f}')
        lines.append(f'| {a} | ' + ' | '.join(cells) + ' |')
    with open(RESULTS_DIR / 'summary.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(RESULTS_DIR / 'summary.json', 'w') as f:
        json.dump(results, f, indent=1)
    log(f'summary written: {RESULTS_DIR / "summary.md"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['views', 'evaluate', 'summarise', 'all'], default='all')
    ap.add_argument('--views', default=','.join(VIEWS))
    ap.add_argument('--models', default=','.join(MODELS))
    args = ap.parse_args()
    views = args.views.split(',')
    models = args.models.split(',')
    t0 = time.time()
    d, series = load()
    log(f'loaded {len(d)} segments in {time.time() - t0:.0f} s')
    if args.stage in ('views', 'all'):
        for v in views:
            try:
                build_views([v], d, series)
            except Exception as e:  # noqa: BLE001
                log(f'view {v}: FAILED {type(e).__name__}: {e}')
    if args.stage in ('evaluate', 'all'):
        evaluate(views, models, d)
    if args.stage in ('summarise', 'all'):
        summarise(d)


if __name__ == '__main__':
    main()
