#!/usr/bin/env python3
"""RoBo6 (rocket-body light curves, 6 classes): the five published baselines,
rerun under the authors' own code, on the fixed train.csv / test.csv split.

Source
    Data   huggingface.co/datasets/kyselica/RoBo6 (MIT). 5,676 training and
           1,404 test light curves; one row = one part of one MMT track.
    Code   github.com/kyselica12/RoBo6_Model_Comparison (no licence file),
           commit pinned below; Astroconformer from the submodule fork
           github.com/kyselica12/Astroconformer at 2ae06eb.
    Paper  Kyselica, Suppa, Silha, Durikovic, arXiv 2412.00544 (Table 3).

What is ported, and from where (read 2026-11-09):
    dataset.py      magnitudes z-scored with mean_std.csv, on non-zero values
                    only; zeros (no measurement / padding) stay 0. Phase
                    angle: (p - 90) / 90 on non-zero values.
    transforms.py   furfaro: first 500 grid points. allworth: first 9,600
                    points of mag and phase, mean of the non-zero values in
                    each run of 8 -> 1,200 points, 2 channels. yao: fold the
                    10,000 points at the stored period into 200 bins (zeros
                    counted in the bin count, as in their loop). resnet and
                    astroconformer: all 10,000 points.
    configs.py      50 epochs, Adam lr 0.001, batch 32 (furfaro 128,
                    allworth 256). Astroconformer: AdamW + their cyclic
                    LambdaLR, stepped every training step by their
                    LightningModule and once per epoch by Lightning itself.
    lightning_module.py  cross-entropy; the weights after the last epoch are
                    scored (no checkpoint is chosen); accuracy and the mean
                    of the six per-class F1 (= macro-F1).
    modules/*.py    the four networks are imported from the clone, unchanged.

Two facts about the published numbers, read from their results/*.csv:
    - For allworth, furfaro, resnet and yao every row is a whole number of
      702 test rows (half of 1,404); astroconformer rows are whole numbers of
      1,404 only. Those four ran on two GPUs (Lightning DDP) and every result
      row scores one half of the test set. The published mean +- sd is over
      20 rows = 10 runs x 2 halves. Under DDP each GPU takes batches of 32
      (so 64 per optimiser step) and its own batch-norm statistics.
      `--ddp-halves 2` emulates that on one device: each optimiser step
      averages the loss of two separately normalised sub-batches, running
      statistics come from the first (rank 0 broadcasts its buffers), and
      test scores are reported both on the full 1,404 rows and per half as
      DistributedSampler deals them (rank r gets rows r, r+2, ...).
    - The split CSVs on the hub changed on 2024-09-08 ("time bug fixed"),
      two days after their results were written (2024-09-06). Before it,
      every *_time.txt was a copy of *_mag.txt; after it, train.csv and
      test.csv carry one path column (the time file) under a 7-name header,
      so their dataset.py would now read the time file as magnitude. The
      mag and phase files are the same blobs in both revisions. This script
      reads the mag and phase files by name, so it is unaffected.

The stored `period` is in grid steps of 0.1 s, not seconds (checked on 9
curves by phase dispersion, 2026-11-09); their yao transform folds in grid
steps, which is consistent with that.

Seeds: their main.py passes seed=None. Here run k uses seed k.

    python tests/experiments/benchmarks/robo6_published.py --stage download
    python tests/experiments/benchmarks/robo6_published.py --stage reproduce \\
        --models yao furfaro --runs 1 --device cpu
    python tests/experiments/benchmarks/robo6_published.py --stage summarise

Data under $KIRBY_ROBO6_DIR (default /Volumes/seagate/benchmarks_protocol/robo6):
raw/ (hub files), clones/, arrays/ (mag.npy, phase.npy, meta.csv). Results
under $KIRBY_BENCH_RESULTS (default .../benchmarks_protocol/results)/robo6_published/:
one JSON per model and run, written last, and summary.md.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('KIRBY_ROBO6_DIR', '/Volumes/seagate/benchmarks_protocol/robo6'))
RESULTS = Path(os.environ.get('KIRBY_BENCH_RESULTS',
                              '/Volumes/seagate/benchmarks_protocol/results')) / 'robo6_published'
RAW, CLONES, ARRAYS = ROOT / 'raw', ROOT / 'clones', ROOT / 'arrays'

HUB = 'https://huggingface.co/datasets/kyselica/RoBo6/resolve'
HUB_REV = 'c8131442e3ff39e75362fda4f914a223fe2ac1d0'    # main on 2026-11-09
CSV_REV = 'acc63e93f1cd22d1a9c05028401df40798df2da1'    # split CSVs with mag/phase/time paths
CODE_REPO = 'https://github.com/kyselica12/RoBo6_Model_Comparison.git'
CODE_REV = 'main'                                       # pushed 2025-01-07; sha recorded per run
ASTRO_REPO = 'https://github.com/kyselica12/Astroconformer.git'
ASTRO_REV = '2ae06eb8e9367a02a480b87e77fb1069836c71bb'

LABELS = ['ARIANE 5 R/B', 'ATLAS 5 CENTAUR R/B', 'CZ-3B R/B', 'DELTA 4 R/B',
          'FALCON 9 R/B', 'H-2A R/B']                   # sorted, as their get_dataset
GRID = 10_000

# configs.py, per model. ddp: processes the published rows came from.
CONFIGS = {
    'allworth':       dict(input_size=1200, channels=['mag', 'phase'], batch=256, ddp=2),
    'furfaro':        dict(input_size=500, channels=['mag'], batch=128, ddp=2),
    'yao':            dict(input_size=200, channels=['mag'], batch=32, ddp=2),
    'resnet':         dict(input_size=GRID, channels=['mag'], batch=32, ddp=2),
    'astroconformer': dict(input_size=GRID, channels=['mag'], batch=32, ddp=1),
}
EPOCHS, LR = 50, 1e-3

# Table 3 of 2412.00544; the means below are recomputed from their results/*.csv.
PUBLISHED = {'astroconformer': (0.725, 0.684), 'resnet': (0.694, 0.600), 'yao': (0.672, 0.604),
             'furfaro': (0.628, 0.552), 'allworth': (0.559, 0.478)}


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg}', flush=True)


# ── download and pack ──────────────────────────────────────────────────────

def _get(url, dest, tries=5, rate_limit_waits=60):
    """The hub rate-limits anonymous downloads (HTTP 429 after about 2,000
    files on 2026-11-09). A 429 waits 60 s and does not use up a try."""
    import urllib.error
    import urllib.request
    import urllib.parse
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    k = waits = 0
    while True:
        try:
            with urllib.request.urlopen(urllib.parse.quote(url, safe=':/'), timeout=120) as r:
                tmp.write_bytes(r.read())
            tmp.rename(dest)
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and waits < rate_limit_waits:
                waits += 1
                time.sleep(60)
                continue
            k += 1
            if k >= tries:
                raise RuntimeError(f'{url}: {exc}')
            time.sleep(5 * k)
        except OSError as exc:
            # the Seagate drive drops for seconds at a time: wait for it,
            # without using up a try
            if not ROOT.exists() and waits < rate_limit_waits:
                waits += 1
                while not ROOT.exists():
                    time.sleep(10)
                dest.parent.mkdir(parents=True, exist_ok=True)
                continue
            k += 1
            if k >= tries:
                raise RuntimeError(f'{url}: {exc}')
            time.sleep(5 * k)


def _split_rows():
    out = []
    for split in ('train', 'test'):
        with open(RAW / f'{split}.csv') as f:
            for r in csv.DictReader(f):
                out.append(dict(split=split, label=r['label'], id=r['id'], part=r['part'],
                                period=r['period'], mag=r['mag'], phase=r['phase']))
    return out


def stage_download(workers):
    for name in ('train.csv', 'test.csv'):
        _get(f'{HUB}/{CSV_REV}/{name}', RAW / name)
    _get(f'{HUB}/{HUB_REV}/mean_std.csv', RAW / 'mean_std.csv')
    rows = _split_rows()
    files = sorted({r[k] for r in rows for k in ('mag', 'phase')})
    log(f'{len(rows)} curves, {len(files)} mag and phase files')
    done = [0]

    def one(rel):
        _get(f'{HUB}/{HUB_REV}/{rel}', RAW / rel)
        done[0] += 1
        if done[0] % 1000 == 0:
            log(f'{done[0]} / {len(files)} files')
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, files))
    for repo, rev, dest in ((CODE_REPO, CODE_REV, CLONES / 'RoBo6_Model_Comparison'),
                            (ASTRO_REPO, ASTRO_REV, CLONES / 'Astroconformer')):
        if not dest.exists():
            subprocess.run(['git', 'clone', '-q', repo, str(dest)], check=True)
        subprocess.run(['git', '-C', str(dest), 'checkout', '-q', rev], check=True)
    stage_pack(rows)


def stage_pack(rows=None):
    rows = rows or _split_rows()
    ARRAYS.mkdir(parents=True, exist_ok=True)
    if (ARRAYS / 'meta.csv').exists():
        log('arrays exist')
        return
    for ch in ('mag', 'phase'):
        X = np.zeros((len(rows), GRID), np.float64)
        for i, r in enumerate(rows):
            x = np.loadtxt(RAW / r[ch])
            if x.shape != (GRID,):
                raise ValueError(f'{r[ch]}: shape {x.shape}')
            X[i] = x
        np.save(ARRAYS / f'{ch}.npy', X)
        log(f'packed {ch}: {X.shape}')
    with open(ARRAYS / 'meta.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'label', 'id', 'part', 'period'])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})


def load_arrays():
    import pandas as pd
    meta = pd.read_csv(ARRAYS / 'meta.csv', dtype={'id': str})
    mag = np.load(ARRAYS / 'mag.npy', mmap_mode='r')
    phase = np.load(ARRAYS / 'phase.npy', mmap_mode='r')
    return meta, mag, phase


# ── their preprocessing ────────────────────────────────────────────────────

def normalise(mag, phase):
    """dataset.py: z-score non-zero values; zeros stay 0."""
    mean, std = np.loadtxt(RAW / 'mean_std.csv', delimiter=',', skiprows=1)
    m = np.array(mag, np.float64)
    nz = m != 0
    m[nz] = (m[nz] - mean) / std
    p = np.array(phase, np.float64)
    nz = p != 0
    p[nz] = (p[nz] - 90) / 90
    # datasets' torch formatter hands float32 to the transforms
    return m.astype(np.float32), p.astype(np.float32)


def allworth_view(m, p):
    out = []
    for x in (m, p):
        x = x[:, :1200 * 8].reshape(len(x), 1200, 8)
        cnt = (x != 0).sum(axis=2)
        cnt[cnt == 0] = 1
        out.append(x.sum(axis=2) / cnt)
    return np.stack(out, axis=1)


def yao_view(m, periods, size=200):
    """transforms.yao_transform, vectorised. Python's round and np.rint both
    round halves to even."""
    out = np.zeros((len(m), 1, size), np.float32)
    i = np.arange(m.shape[1], dtype=np.float64)
    for r in range(len(m)):
        step = periods[r] / size
        idx = np.rint((i - step / 2) / step).astype(np.int64) % size
        y = np.bincount(idx, weights=m[r].astype(np.float64), minlength=size)
        c = np.bincount(idx, minlength=size).astype(np.float64)
        nz = y != 0
        y[nz] = y[nz] / c[nz]
        out[r, 0] = y
    return out


def model_input(name, m, p, periods):
    if name == 'allworth':
        return allworth_view(m, p).astype(np.float32)
    if name == 'furfaro':
        return m[:, None, :500]
    if name == 'yao':
        return yao_view(m, periods)
    if name == 'astroconformer':
        return m                                       # (n, 10000); the net unsqueezes
    return m[:, None, :]                               # resnet


# ── their networks ─────────────────────────────────────────────────────────

def build_net(name, n_classes=6):
    code = CLONES / 'RoBo6_Model_Comparison'
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    cfg = CONFIGS[name]
    in_ch = len(cfg['channels'])
    if name == 'allworth':
        from modules.allworth import AllworthNet
        return AllworthNet(in_ch, cfg['input_size'], n_classes), None
    if name == 'furfaro':
        from modules.furfaro import FurfaroNet
        return FurfaroNet(in_ch, cfg['input_size'], n_classes), None
    if name == 'yao':
        from modules.yao import YaoNet
        return YaoNet(in_ch, cfg['input_size'], n_classes), None
    if name == 'resnet':
        from modules.resnet import resnet20
        return resnet20(n_classes, in_ch), None
    # astroconformer: their main.py's config loading, without torchinfo/nvitop
    import yaml
    astro = CLONES / 'Astroconformer'
    if str(astro) not in sys.path:
        sys.path.insert(0, str(astro))
    from Astroconformer.utils import Container
    from Astroconformer.Model.models import model_dict
    from Astroconformer.Model.utils import deepnorm_init
    from Astroconformer.Train.lr_scheduler import get_scheduler
    args = Container(**yaml.safe_load(open(astro / 'experiment_config.yaml')))
    args.load_dict(yaml.safe_load(open(astro / 'model_config.yaml'))[args.model])
    args.load_dict(yaml.safe_load(open(astro / 'default_config.yaml')))
    args.input_shape = (1, GRID)
    # init_train, minus the torchinfo summary
    args.stride = int(20 / args.sample_rates[0] ** 0.5)
    if args.deepnorm and args.num_layers >= 10:
        c = args.num_layers / 5.0
        args.alpha, args.beta = c ** 0.5, c ** -0.5
    net = model_dict[args.model](args)
    deepnorm_init(net, args)
    import torch
    opt = {'adamw': torch.optim.AdamW, 'adam': torch.optim.Adam}[args.optimizer](
        net.parameters(), lr=args.basic_lr, eps=args.eps, weight_decay=args.weight_decay)
    return net, (opt, get_scheduler(opt, args), args)


# ── training and scoring ───────────────────────────────────────────────────

def _metrics(y, pred, n_classes=6):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    labels = list(range(n_classes))
    f1 = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(np.mean(f1)),
            'macro_precision': float(np.mean(precision_score(y, pred, labels=labels, average=None, zero_division=0))),
            'macro_recall': float(np.mean(recall_score(y, pred, labels=labels, average=None, zero_division=0))),
            'f1_per_class': {LABELS[k]: float(f1[k]) for k in labels}}


def train_and_score(name, Xtr, ytr, Xte, yte, seed, device, ddp, epochs):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = CONFIGS[name]
    net, extra = build_net(name)
    net = net.to(device)
    if extra is None:
        opt, sched = torch.optim.Adam(net.parameters(), lr=LR), None
    else:
        opt, sched, _ = extra
    use_amp = name == 'astroconformer' and device.startswith('cuda')   # precision 16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    crit = nn.CrossEntropyLoss()
    bs = cfg['batch']
    n = len(Xtr)
    Xtr_t = torch.from_numpy(np.ascontiguousarray(Xtr))
    ytr_t = torch.from_numpy(ytr.astype(np.int64))
    bns = [m for m in net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        # DDP: each process shuffles its own half (DistributedSampler) and takes
        # batches of bs; one optimiser step consumes one batch from each.
        perm = torch.randperm(n, generator=gen)
        shards = [perm[r::ddp] for r in range(ddp)]
        steps = int(np.ceil(min(len(s) for s in shards) / bs))
        for b in range(steps):
            opt.zero_grad(set_to_none=True)
            for r, shard in enumerate(shards):
                idx = shard[b * bs:(b + 1) * bs]
                if len(idx) == 0:
                    continue
                if r > 0:                          # rank 0's running stats are kept
                    saved = [m.momentum for m in bns]
                    for m in bns:
                        m.momentum = 0.0
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    logits = net(Xtr_t[idx].to(device).float())
                    loss = crit(logits, ytr_t[idx].to(device)) / ddp
                scaler.scale(loss).backward()
                if r > 0:
                    for m, mo in zip(bns, saved):
                        m.momentum = mo
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()                       # their training_step
        if sched is not None:
            sched.step()                           # Lightning's per-epoch step
        if ep == 0 or (ep + 1) % 10 == 0:
            log(f'{name} seed {seed}: epoch {ep + 1}/{epochs}, {time.time() - t0:.0f} s')
    net.eval()
    preds = []
    with torch.no_grad():
        for b in range(0, len(Xte), 256):
            x = torch.from_numpy(np.ascontiguousarray(Xte[b:b + 256])).to(device).float()
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                preds.append(net(x).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    out = {'full_test': _metrics(yte, pred),
           'halves': [_metrics(yte[r::2], pred[r::2]) for r in range(2)] if ddp == 2 else None,
           'train_seconds': time.time() - t0, 'predictions': pred.tolist()}
    return out


def stage_reproduce(models, runs, device, epochs, debug):
    meta, mag, phase = load_arrays()
    m, p = normalise(mag, phase)
    y = meta['label'].map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    tr = (meta['split'] == 'train').to_numpy()
    te = ~tr
    periods = meta['period'].to_numpy(np.float64)
    sha = subprocess.run(['git', '-C', str(CLONES / 'RoBo6_Model_Comparison'), 'rev-parse', 'HEAD'],
                         capture_output=True, text=True).stdout.strip()
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name in models:
        X = model_input(name, m, p, periods)
        ddp = CONFIGS[name]['ddp']
        for k in runs:
            f = RESULTS / f'{name}__run{k}{"__debug" if debug else ""}.json'
            if f.exists():
                log(f'{f.name} exists')
                continue
            log(f'{name} run {k}: input {X.shape[1:]}, {tr.sum()} train, {te.sum()} test, ddp {ddp}')
            res = train_and_score(name, X[tr], y[tr], X[te], y[te], seed=k, device=device,
                                  ddp=ddp, epochs=epochs)
            res.update(model=name, run=k, seed=k, epochs=epochs, device=device, ddp_emulated=ddp,
                       code_sha=sha, astroconformer_sha=ASTRO_REV, hub_rev=HUB_REV,
                       csv_rev=CSV_REV, debug=debug, result_status='ok',
                       published_accuracy_f1=PUBLISHED[name])
            tmp = f.with_suffix('.tmp')
            tmp.write_text(json.dumps(res))
            tmp.rename(f)
            ft = res['full_test']
            log(f'{name} run {k}: accuracy {ft["accuracy"]:.4f}, macro-F1 {ft["macro_f1"]:.4f} '
                f'(published {PUBLISHED[name][0]} / {PUBLISHED[name][1]}), '
                f'{res["train_seconds"]:.0f} s')


def stage_summarise():
    rows = []
    for name in CONFIGS:
        res = [json.loads(f.read_text()) for f in sorted(RESULTS.glob(f'{name}__run*.json'))]
        res = [r for r in res if not r.get('debug')]
        if not res:
            continue
        acc = [r['full_test']['accuracy'] for r in res]
        f1 = [r['full_test']['macro_f1'] for r in res]
        if res[0]['halves']:
            ha = [h['accuracy'] for r in res for h in r['halves']]
            hf = [h['macro_f1'] for r in res for h in r['halves']]
            halves = f'{np.mean(ha):.4f} ± {np.std(ha):.4f} | {np.mean(hf):.4f} ± {np.std(hf):.4f}'
        else:
            halves = '—'
        rows.append(f'| {name} | {len(res)} | {PUBLISHED[name][0]} | {PUBLISHED[name][1]} | '
                    f'{np.mean(acc):.4f} ± {np.std(acc):.4f} | {np.mean(f1):.4f} ± {np.std(f1):.4f} | {halves} |')
    text = ('# RoBo6: published baselines rerun\n\nOne row is one model. Scores on the 1,404 test '
            'curves; mean ± population sd over runs. "Halves" scores each half of the test set '
            'as their two-GPU runs did (published rows are halves for all but astroconformer).\n\n'
            '| model | runs | published accuracy | published macro-F1 | accuracy | macro-F1 | '
            'halves: accuracy, macro-F1 |\n|---|---|---|---|---|---|---|\n' + '\n'.join(rows) + '\n')
    (RESULTS / 'summary.md').write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['download', 'pack', 'reproduce', 'summarise'])
    ap.add_argument('--models', nargs='+', default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument('--runs', type=int, default=10, help='runs 0..N-1 (their main.py: N = 10)')
    ap.add_argument('--first-run', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    if a.stage == 'download':
        stage_download(a.workers)
    elif a.stage == 'pack':
        stage_pack()
    elif a.stage == 'reproduce':
        stage_reproduce(a.models, range(a.first_run, a.first_run + a.runs), a.device,
                        a.epochs, debug=a.epochs != EPOCHS)
    else:
        stage_summarise()


if __name__ == '__main__':
    main()
