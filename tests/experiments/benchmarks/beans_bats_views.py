#!/usr/bin/env python3
"""BEANS bats view and model screen: every candidate representation of one
call, scored alone under the BEANS classical models on the BEANS split, with
test predictions kept and overlap between views measured.

Object: one call, 250 kHz float32 WAV (NEW_DATASETS.md, "BEANS bats"). Label:
which of 10 bats made it. Split: `published` = the BEANS 6,000 / 2,000 /
2,000 calls, rebuilt and md5-checked by beans_bats_published.beans_split.
A grouped split is not built: how to group is open (NEW_DATASETS.md).

Every waveform is first padded with zeros or cut to 5 s at 250 kHz, as the
BEANS loader does (`_get_waveform`, max_duration 5), then one of three input
modes feeds a frozen encoder:
    bb   baseband: resampled to the encoder's rate (torchaudio resample, as
         BEANS and every encoder paper do); everything above half that rate
         is lost
    te   time-expansion (Sarkar et al. 2026, arXiv 2604.27936, Sec. 3.2):
         the 250 kHz samples are read as if recorded at the encoder's rate,
         so 0-125 kHz lands in the encoder's band. 5 s becomes 78 s at 16 kHz
    mb   multi-band (same paper, Sec. 2; their package multiband-audio):
         MultibandTransform(sample_rate=250000, target_sr=<encoder rate>),
         every band through the encoder, the band embeddings concatenated.
         The paper fuses them with a trained module; concatenating leaves
         the choice of bands to the hybrid search
Every encoder output is averaged over time (final layer, time-mean: the
protocol of Miron et al. arXiv 2508.11845 and of Sarkar et al.).

Views
    mfcc_stats        80  BEANS code (beans_bats_published.mfcc_stats)
    <enc>_bb/_te/_mb      the nine encoders of Sarkar et al. Table 3, all
                          through avex (github.com/earthspecies/avex):
                          effnet_all, effnet_bio, effnet_as, beats_all,
                          beats_bio, beats_nlm, eat_all, eat_bio (16 kHz),
                          birdnet (48 kHz, avex's ONNX export of BirdNET 2.4)
    sl_eat_all_bb, sl_eat_bio_bb   Miron et al. Table 8 (avex)
    aves_bio_bb       AVES-bio (Hagiwara 2023), Earth Species' torchaudio port
    birdaves_bb       BirdAVES biox-base (Schwinger Table 10; avex's `aves_bio`)
    birdmae_large_bb  Bird-MAE-Large, 32 kHz (Schwinger et al. 2508.01277; avex)
    perch2_bb         Perch 2.0, 32 kHz (arXiv 2508.04665; avex's ONNX export
                      by a third party, justinchuby/Perch-onnx)
    biolingual_bb     BioLingual CLAP audio tower, 48 kHz (transformers)
    hubert_bb, wavlm_bb   HuBERT base, WavLM base+ (Cauzinille et al. 2025;
                          transformers; final layer, their best layer is not
                          reported per model)
    vggish_bb         VGGish 128-d, BEANS's own torch.hub port, mean over the
                      0.96 s examples (BEANS VGGishClassifier.forward, frozen)

Models: the BEANS five (beans_bats_published.GRIDS): fit on the training
calls per grid value, choose on the validation calls, score the test calls.

Stages
    extract    mfcc_stats (KIRBy's environment)
    build      --views V: one embedding view, chunks of 200 calls, resumes.
               Needs the environment in KIRBY_SCREEN_PY_BEANS_EMBED (avex,
               multiband-audio, transformers)
    evaluate   --views V --models M --splits published
    summarise  summary.md: test accuracy per view and model, error overlap
               between views per model, linear CKA between views

Paths: KIRBY_BEANS_DIR (audio/ or egyptian_fruit_bats.zip, calls.csv,
views/), KIRBY_BENCH_RESULTS (results under beans_bats_views/).
--limit N (debug) builds the first N calls only, into views/debug/.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import beans_bats_published as pub  # noqa: E402

DATA_DIR = pub.DATA_DIR
VIEWS_DIR = DATA_DIR / 'views'
RESULTS_DIR = Path(os.environ.get('KIRBY_BENCH_RESULTS',
                                  '/Volumes/seagate/benchmarks_protocol/results')) / 'beans_bats_views'
SPLITS = ['published']
MODELS = list(pub.GRIDS)
CHUNK = 200

# encoder key -> (how it is loaded, its sample rate)
AVEX = {
    'effnet_all': 'esp_aves2_effnetb0_all', 'effnet_bio': 'esp_aves2_effnetb0_bio',
    'effnet_as': 'esp_aves2_effnetb0_audioset', 'beats_all': 'esp_aves2_sl_beats_all',
    'beats_bio': 'esp_aves2_sl_beats_bio', 'beats_nlm': 'esp_aves2_naturelm_audio_v1_beats',
    'eat_all': 'esp_aves2_eat_all', 'eat_bio': 'esp_aves2_eat_bio',
    'sl_eat_all': 'esp_aves2_sl_eat_all_ssl_all', 'sl_eat_bio': 'esp_aves2_sl_eat_bio_ssl_all',
}
ENCODERS = {k: ('avex', v, 16_000) for k, v in AVEX.items()}
ENCODERS.update({
    'birdnet': ('avex_cls', 'birdnet', 48_000),
    'birdaves': ('avex_cls', 'aves_bio', 16_000),   # avex's `aves_bio` loads birdaves-biox-base
    'perch2': ('avex_cls', 'perch2', 32_000),
    'birdmae_large': ('avex_spec', {'name': 'birdmae', 'pretrained': True,
                                    'model_id': 'DBD-research-group/Bird-MAE-Large'}, 32_000),
    'aves_bio': ('aves_torchaudio',
                 'https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-bio', 16_000),
    'biolingual': ('hf_clap', 'davidrrobinson/BioLingual', 48_000),
    'hubert': ('hf_ssl', 'facebook/hubert-base-ls960', 16_000),
    'wavlm': ('hf_ssl', 'microsoft/wavlm-base-plus', 16_000),
    'vggish': ('vggish', 'harritaylor/torchvggish', 16_000),
})
SARKAR = ['effnet_all', 'effnet_bio', 'effnet_as', 'beats_all', 'beats_bio', 'beats_nlm',
          'eat_all', 'eat_bio', 'birdnet']
HAND_VIEWS = ['mfcc_stats']
EMBED_VIEWS = ([f'{e}_{m}' for e in SARKAR for m in ('bb', 'te', 'mb')]
               + [f'{e}_bb' for e in ENCODERS if e not in SARKAR])


def all_views():
    return HAND_VIEWS + EMBED_VIEWS


def log(msg):
    print('%s %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def view_path(view, debug=False):
    return (VIEWS_DIR / 'debug' if debug else VIEWS_DIR) / f'{view}.npy'


def result_path(view, model, split):
    return RESULTS_DIR / split / f'{view}__{model}.json'


# ── encoders ──────────────────────────────────────────────────────────────

class Encoder:
    """One frozen encoder: __call__(wave at its own rate, 1-D float32) -> 1-D
    time-mean embedding."""

    def __init__(self, key, device):
        import torch
        self.torch = torch
        self.kind, self.ref, self.rate = ENCODERS[key]
        self.device = device
        if self.kind in ('avex', 'avex_spec'):
            from avex import load_model
            if self.kind == 'avex':
                self.m = load_model(self.ref, return_features_only=True, device=device)
            else:
                from avex.configs import ModelSpec, AudioConfig
                spec = ModelSpec(device=device, audio_config=AudioConfig(sample_rate=self.rate), **self.ref)
                self.m = load_model(spec, return_features_only=True, device=device)
            self.m.eval()
        elif self.kind == 'avex_cls':
            if self.ref == 'birdnet':
                from avex.models.birdnet import Model
                self.m = Model(device=device, return_features_only=True)
            elif self.ref == 'perch2':
                from avex.models.perch2 import Model
                self.m = Model(device=device, return_features_only=True)
            else:
                from avex.models.aves_model import Model
                from avex.configs import AudioConfig
                self.m = Model(num_classes=len(pub.LABELS), device=device,
                               audio_config=AudioConfig(sample_rate=self.rate, representation='raw',
                                                        normalize=False))
            self.m.eval()
        elif self.kind == 'aves_torchaudio':
            # earthspecies/aves README: torchaudio wav2vec2_model from the ported config
            import urllib.request
            import torchaudio
            cfg = json.loads(urllib.request.urlopen(self.ref + '.torchaudio.model_config.json').read())
            self.m = torchaudio.models.wav2vec2_model(**cfg, aux_num_out=None)
            self.m.load_state_dict(torch.hub.load_state_dict_from_url(self.ref + '.torchaudio.pt',
                                                                      map_location='cpu'))
            self.m.feature_extractor.requires_grad_(False)
            self.m = self.m.to(device).eval()
        elif self.kind == 'hf_clap':
            from transformers import ClapModel, ClapProcessor
            self.m = ClapModel.from_pretrained(self.ref).to(device).eval()
            self.proc = ClapProcessor.from_pretrained(self.ref)
        elif self.kind == 'hf_ssl':
            from transformers import AutoModel, AutoFeatureExtractor
            self.m = AutoModel.from_pretrained(self.ref).to(device).eval()
            self.proc = AutoFeatureExtractor.from_pretrained(self.ref)
        elif self.kind == 'vggish':
            self.m = torch.hub.load(self.ref, 'vggish')
            self.m.postprocess = False      # BEANS VGGishClassifier: raw 128-d, no PCA
            self.m.eval()

    def __call__(self, w):
        torch = self.torch
        with torch.no_grad():
            if self.kind == 'avex_cls' and self.ref in ('birdnet', 'perch2'):
                x = torch.from_numpy(np.ascontiguousarray(w))[None].to(self.device)
                return self.m.extract_embeddings(x, aggregation='mean').float()[0].cpu().numpy()
            if self.kind == 'aves_torchaudio':
                x = torch.from_numpy(np.ascontiguousarray(w))[None].to(self.device)
                return self.m.extract_features(x)[0][-1].mean(dim=1)[0].cpu().numpy()
            if self.kind in ('avex', 'avex_spec', 'avex_cls'):
                out = self.m(torch.from_numpy(np.ascontiguousarray(w))[None].to(self.device))
                if isinstance(out, (tuple, list)):
                    out = out[0]
                out = out.float()
                if out.ndim == 3:            # (1, time, features)
                    out = out.mean(dim=1)
                elif out.ndim == 4:          # (1, channels, freq, time): average all but channels
                    out = out.mean(dim=(2, 3))
                return out[0].cpu().numpy()
            if self.kind == 'hf_clap':
                x = self.proc(audios=w, sampling_rate=self.rate, return_tensors='pt')
                return self.m.get_audio_features(**{k: v.to(self.device) for k, v in x.items()})[0].cpu().numpy()
            if self.kind == 'hf_ssl':
                x = self.proc(w, sampling_rate=self.rate, return_tensors='pt')
                h = self.m(**{k: v.to(self.device) for k, v in x.items()}).last_hidden_state
                return h.mean(dim=1)[0].cpu().numpy()
            if self.kind == 'vggish':
                # preprocess=True runs vggish_input.waveform_to_examples, the
                # function BEANS calls itself before its classifier
                return self.m(w.astype(np.float64), fs=self.rate).mean(dim=0).cpu().numpy()
        raise ValueError(self.kind)


def to_input(w250, mode, rate, mb=None):
    """The 5 s, 250 kHz waveform in the form the encoder reads."""
    import torch
    import torchaudio
    x = torch.from_numpy(np.ascontiguousarray(w250))
    if mode == 'bb':
        return [torchaudio.functional.resample(x, pub.SAMPLE_RATE, rate).numpy()]
    if mode == 'te':
        return [x.numpy()]                      # same samples, read at `rate`
    if mode == 'mb':
        bands = mb(x[None])[0]                  # (bands, samples at `rate`)
        return [b.numpy() for b in bands]
    raise ValueError(mode)


def pad_5s(w):
    n = pub.MAX_DURATION * pub.SAMPLE_RATE
    w = w[:n]
    return np.pad(w, (0, n - len(w))) if len(w) < n else w


def stage_build(view, limit=None, device=None):
    import torch
    debug = limit is not None
    out = view_path(view, debug)
    if out.exists():
        log(f'{view}: already built at {out}')
        return
    enc_key, mode = view.rsplit('_', 1)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK') or os.cpu_count() or 1))
    src = pub.AudioSource()
    calls = pub.calls_table(src)
    n = len(calls) if limit is None else min(limit, len(calls))
    enc = Encoder(enc_key, device)
    mb = None
    if mode == 'mb':
        import multiband_audio as mba
        mb = mba.MultibandTransform(sample_rate=pub.SAMPLE_RATE, target_sr=enc.rate)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(f'{view}.partial.npz')
    X, done = None, 0
    if part.exists():
        z = np.load(part)
        X, done = z['X'], int(z['done'])
        log(f'{view}: resuming after {done} calls')
    t0 = time.time()
    for i in range(done, n):
        parts = [enc(x) for x in to_input(pad_5s(src.read(calls.file[i])), mode, enc.rate, mb)]
        v = np.concatenate(parts).astype(np.float32)
        if X is None:
            X = np.full((n, len(v)), np.nan, np.float32)
        X[i] = v
        if (i + 1) % CHUNK == 0 or i + 1 == n:
            np.savez(part, X=X, done=i + 1)
            log(f'{view}: {i + 1} / {n} calls, {(time.time() - t0) / (i + 1 - done):.2f} s per call, '
                f'{X.shape[1]} features')
    np.save(out, X)
    meta = {'view': view, 'encoder': enc_key, 'mode': mode, 'rate': enc.rate, 'calls': n,
            'features': int(X.shape[1]), 'device': device, 'seconds': time.time() - t0,
            'bands': None if mb is None else len(mb.bands)}
    out.with_suffix('.json').write_text(json.dumps(meta, indent=1))
    if part.exists():
        part.unlink()
    log(f'{view}: written {out} ({n} calls, {X.shape[1]} features)')


def stage_extract():
    pub.stage_features(stream=False)       # writes <KIRBY_BEANS_DIR>/views/mfcc_stats.npy


# ── evaluate ──────────────────────────────────────────────────────────────

def stage_evaluate(view, model, split, seed=42):
    import pandas as pd
    from sklearn.metrics import f1_score
    if split not in SPLITS:
        raise SystemExit(f'split {split!r} is not built; only {SPLITS}')
    out = result_path(view, model, split)
    if out.exists():
        return
    vp = view_path(view)
    if not vp.exists():
        raise SystemExit(f'{view}: no view file at {vp}; build it first')
    X = np.load(vp).astype(np.float64)
    calls = pd.read_csv(DATA_DIR / 'calls.csv')
    if len(X) != len(calls) or np.isnan(X).any():
        raise SystemExit(f'{view}: {len(X)} rows (nan present: {np.isnan(X).any()}), {len(calls)} calls')
    y = calls.label.map({l: i for i, l in enumerate(pub.LABELS)}).values
    idx = {s: np.where(calls.split.values == s)[0] for s in ('train', 'valid', 'test')}
    t0 = time.time()
    pred, v, valid = pub.published_protocol(model, X[idx['train']], y[idx['train']], X[idx['valid']],
                                            y[idx['valid']], X[idx['test']], seed)
    acc = float(np.mean(pred == y[idx['test']]))
    res = {'view': view, 'model': model, 'split': split, 'n_features': int(X.shape[1]),
           'n_train': len(idx['train']), 'n_valid': len(idx['valid']), 'n_test': len(idx['test']),
           'test_accuracy': acc, 'test_macro_f1': float(f1_score(y[idx['test']], pred, average='macro')),
           'test_wrong': int(np.sum(pred != y[idx['test']])),
           'chosen': {pub.GRIDS[model][0]: v}, 'valid': valid, 'seed': seed,
           'seconds': time.time() - t0, 'debug': False, 'result_status': 'ok'}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out.with_suffix('')) + '.test_pred.npy', pred)
    out.write_text(json.dumps(res, indent=1))
    log(f'{view} {model} {split}: test accuracy {acc:.4f} ({res["seconds"]:.0f} s)')


# ── summarise ─────────────────────────────────────────────────────────────

def linear_cka(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    return float(np.linalg.norm(A.T @ B, 'fro') ** 2
                 / (np.linalg.norm(A.T @ A, 'fro') * np.linalg.norm(B.T @ B, 'fro') + 1e-30))


def stage_summarise():
    import pandas as pd
    calls = pd.read_csv(DATA_DIR / 'calls.csv')
    y = calls.label.map({l: i for i, l in enumerate(pub.LABELS)}).values
    te = np.where(calls.split.values == 'test')[0]
    lines = ['# BEANS bats view and model screen', '',
             'Test accuracy on the 2,000 BEANS test calls; each model\'s grid value chosen on the '
             '2,000 validation calls. Published (BEANS Table 2, mfcc_stats): '
             + ', '.join(f'{m} {v:.3f}' for m, v in pub.PUBLISHED.items()) + '.', '']
    for split in SPLITS:
        res = {}
        for p in sorted((RESULTS_DIR / split).glob('*__*.json')) if (RESULTS_DIR / split).is_dir() else []:
            r = json.loads(p.read_text())
            res[(r['view'], r['model'])] = r
        if not res:
            continue
        views = [v for v in all_views() if any(k[0] == v for k in res)]
        models = [m for m in MODELS if any(k[1] == m for k in res)]
        lines += [f'## {split} split', '', 'One row is one view; a cell is test accuracy.', '',
                  '| view | features | ' + ' | '.join(models) + ' |', '|---|---|' + '---|' * len(models)]
        for v in views:
            nf = next(r['n_features'] for k, r in res.items() if k[0] == v)
            lines.append(f'| {v} | {nf} | ' + ' | '.join(
                f"{res[(v, m)]['test_accuracy']:.4f}" if (v, m) in res else '' for m in models) + ' |')
        lines.append('')
        for m in models:
            errs = {}
            for v in views:
                f = Path(str(result_path(v, m, split).with_suffix('')) + '.test_pred.npy')
                if f.exists():
                    errs[v] = set(np.where(np.load(f) != y[te])[0])
            if len(errs) < 2:
                continue
            vs = list(errs)
            lines += [f'### error overlap, {m}, {split}', '',
                      'Shared wrong test calls over all wrong test calls of either view (Jaccard), in %. '
                      'Diagonal: error rate in %.', '',
                      '| | ' + ' | '.join(vs) + ' |', '|---|' + '---|' * len(vs)]
            for a in vs:
                cells = []
                for b in vs:
                    if a == b:
                        cells.append(f'{100 * len(errs[a]) / len(te):.1f}')
                    else:
                        u = errs[a] | errs[b]
                        cells.append(f'{100 * len(errs[a] & errs[b]) / max(len(u), 1):.1f}')
                lines.append(f'| {a} | ' + ' | '.join(cells) + ' |')
            lines.append('')
        mats = {}
        for v in views:
            X = np.load(view_path(v))[te].astype(float)
            mats[v] = (X - X.mean(0)) / (X.std(0) + 1e-12)
        lines += ['### linear CKA between views', '',
                  'Standardised view matrices on the 2,000 test calls. 1 = the same information up to a '
                  'linear map, 0 = unrelated.', '',
                  '| | ' + ' | '.join(views) + ' |', '|---|' + '---|' * len(views)]
        for a in views:
            lines.append(f'| {a} | ' + ' | '.join(f'{linear_cka(mats[a], mats[b]):.2f}' for b in views) + ' |')
        lines.append('')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / 'summary.md').write_text('\n'.join(lines) + '\n')
    log(f'summary written: {RESULTS_DIR / "summary.md"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['extract', 'build', 'evaluate', 'summarise'], required=True)
    ap.add_argument('--views', default=None)
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--splits', default='published')
    ap.add_argument('--limit', type=int, default=None, help='debug: first N calls, into views/debug/')
    ap.add_argument('--device', default=None)
    a = ap.parse_args()
    views = a.views.split(',') if a.views else None
    if a.stage == 'extract':
        stage_extract()
    elif a.stage == 'build':
        for v in views or EMBED_VIEWS:
            stage_build(v, a.limit, a.device)
    elif a.stage == 'evaluate':
        for v in views or all_views():
            for m in a.models.split(','):
                for s in a.splits.split(','):
                    stage_evaluate(v, m, s)
    else:
        stage_summarise()


if __name__ == '__main__':
    main()
