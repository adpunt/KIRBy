#!/usr/bin/env python3
"""Tranception Large embeddings for one ProteinGym assay's variant sequences.

NOT A RESULT. Writes a representation matrix; scoring is
b08_proteingym_score_representation.py.

The Tranception Large checkpoint on HuggingFace ships config.json and
pytorch_model.bin and no modeling file, so the architecture has to come from
code. It comes from ProteinNPT's vendored copy,
proteinnpt/utils/tranception/, which is what their own baseline imports
(baselines/model.py:13-14). Loading the checkpoint executes that code, which
the auto-mode sandbox refuses without a permission rule for this script.

Which hidden state to pool. Their two call sites disagree:
  baselines/model.py:157       output.hidden_states[0]   embedding layer
  utils/embedding_utils.py:300 output.hidden_states[-1]  layer 36
The second is the precompute path, which their README has you run before the
baselines so training reads embeddings from HDF5 rather than recomputing them.
This script follows that one: hidden_states[-1], 1,280 features.

Pooling is mean over tokens, because their head is AA_embeddings_mean_pooled
(Embeddings_Tranception_final.json) and nothing masks the pad positions. Every
sequence in a substitution assay is the same length, so there is no padding
here and the only tokens averaged in beyond the residues are [CLS] and [SEP],
which their tokenizer prepends and appends (model_pytorch.py:928-929).

Tokenization follows their encode_batch (model_pytorch.py:899-908): X, B, J
and Z are replaced before tokenizing. A substitution assay never contains
them; the replacement is kept so the code does not silently differ.

    python tests/experiments/benchmarks/b07_proteingym_tranception_embeddings.py \
        ESTA_BACSU_Nutschel_2020 --batch 4

    --limit N             embed only the first N variants, for a timed check
    --checkpoint-every N  rows per checkpoint, default 160
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('HF_HOME', '/Volumes/seagate/hf_cache')

PNPT = '/Volumes/seagate/ProteinNPT'
REPO_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', '..', 'src')
sys.path.insert(0, os.path.abspath(REPO_SRC))
sys.path.insert(0, PNPT)

import numpy as np
import torch

from kirby.datasets.proteingym import load_proteingym_dataset

OUT_ROOT = '/Volumes/seagate/proteingym_embeddings'
CKPT_ROOT = '/Volumes/seagate/benchmarks_protocol/proteingym/checkpoints'
CKPT_DIR = ('/Volumes/seagate/hf_cache/hub/'
            'models--OATML-Markslab--Tranception_Large/snapshots/'
            'd1ef627127760862bc0615ee74372ad4b23b2193')
N_LAYER = 36
HIDDEN = 1280


def save_matrix(out_dir, tag, X, mutants):
    os.makedirs(out_dir, exist_ok=True)
    # save and savez both append their own suffix to a path that lacks it, so
    # write through a handle or the temporary name is not what gets renamed.
    tmp = f'{out_dir}/{tag}.npy.partial'
    with open(tmp, 'wb') as fh:
        np.save(fh, X)
    os.replace(tmp, f'{out_dir}/{tag}.npy')
    tmp = f'{out_dir}/{tag}.mutants.txt.partial'
    with open(tmp, 'w') as fh:
        fh.write('\n'.join(mutants) + '\n')
    os.replace(tmp, f'{out_dir}/{tag}.mutants.txt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('assay', nargs='?', default='ESTA_BACSU_Nutschel_2020')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--checkpoint-every', type=int, default=160)
    ap.add_argument('--ckpt-dir', default=CKPT_ROOT)
    args = ap.parse_args()

    t0 = time.time()
    d = load_proteingym_dataset(f'pgym_{args.assay}')
    seqs = list(d['sequences'])
    mutants = [f"{m[0]['wt_aa']}{m[0]['pos']}{m[0]['mut_aa']}"
               for m in d['mutations']]
    if args.limit:
        seqs, mutants = seqs[:args.limit], mutants[:args.limit]
    print(f'{args.assay}: {len(seqs)} sequences, {len(seqs[0])} residues',
          flush=True)

    from proteinnpt.utils.tranception.config import TranceptionConfig
    from proteinnpt.utils.tranception.model_pytorch import (
        TranceptionLMHeadModel, get_tranception_tokenizer)

    cfg = json.load(open(f'{CKPT_DIR}/config.json'))
    cfg = TranceptionConfig(**cfg)
    cfg.tokenizer = get_tranception_tokenizer()
    cfg.inference_time_retrieval_type = None
    cfg.retrieval_aggregation_mode = None
    model = TranceptionLMHeadModel.from_pretrained(
        pretrained_model_name_or_path=CKPT_DIR, config=cfg)
    model.eval()
    if cfg.n_layer != N_LAYER or cfg.n_embd != HIDDEN:
        raise SystemExit(f'expected {N_LAYER} layers and {HIDDEN} features, '
                         f'config reports {cfg.n_layer} and {cfg.n_embd}')
    if len(seqs[0]) + 2 > cfg.n_ctx:
        raise SystemExit(f'{len(seqs[0])} residues plus two special tokens '
                         f'exceeds the {cfg.n_ctx} context, so their slicing '
                         f'would apply and this script does not do it')
    torch.set_num_threads(os.cpu_count() or 1)
    print(f'model loaded in {time.time()-t0:.0f}s, {cfg.n_layer} layers, '
          f'{cfg.n_embd} features, {torch.get_num_threads()} threads',
          flush=True)

    tag = ('tranception_mean_last'
           f'{"_first%d" % args.limit if args.limit else ""}')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt = f'{args.ckpt_dir}/{args.assay}__{tag}.npz'

    out = np.zeros((len(seqs), HIDDEN), dtype=np.float32)
    start = 0
    if os.path.exists(ckpt):
        z = np.load(ckpt)
        if [str(m) for m in z['mutants']] == mutants:
            start = int(z['done'])
            out[:start] = z['X'][:start]
            print(f'resuming from the checkpoint at {start}/{len(seqs)} '
                  f'variants, {ckpt}', flush=True)
        else:
            print('checkpoint is for a different variant list, ignoring it',
                  flush=True)

    def write_ckpt(done):
        # savez appends .npz to a path that lacks it, so write through a
        # handle and the temporary name stays the one that gets renamed.
        try:
            tmp = ckpt + '.partial'
            with open(tmp, 'wb') as fh:
                np.savez(fh, X=out, done=done, mutants=np.array(mutants))
            os.replace(tmp, ckpt)
        except OSError as exc:
            print(f'  checkpoint write failed ({exc}), continuing', flush=True)

    def encode(chunk):
        # their encode_batch, model_pytorch.py:899-908
        rep = {'X': 'ACDEFGHIKLMNPQRSTVWY', 'B': 'DN', 'J': 'IL', 'Z': 'EQ'}
        fixed = []
        for s in chunk:
            for ch, repl in rep.items():
                if ch in s:
                    s = ''.join(np.random.choice(list(repl)) if c == ch else c
                                for c in s)
            fixed.append(s)
        return cfg.tokenizer(fixed, add_special_tokens=True, truncation=True,
                             padding=True, max_length=cfg.n_ctx,
                             return_tensors='pt')

    t_embed = time.time()
    done = start
    with torch.no_grad():
        for i in range(start, len(seqs), args.batch):
            chunk = seqs[i:i + args.batch]
            enc = encode(chunk)
            o = model(input_ids=enc['input_ids'], return_dict=True,
                      output_hidden_states=True)
            hs = o.hidden_states[-1]
            out[i:i + len(chunk)] = hs.mean(dim=1).cpu().numpy()
            done += len(chunk)
            el = time.time() - t_embed
            n_new = done - start
            print(f'  {done}/{len(seqs)}  {el:.0f}s  '
                  f'{el/n_new:.2f}s per sequence  projected remaining '
                  f'{el/n_new*(len(seqs)-done)/60:.0f} min', flush=True)
            if done % args.checkpoint_every < args.batch or done == len(seqs):
                write_ckpt(done)
                print(f'  checkpoint at {done}', flush=True)

    write_ckpt(len(seqs))
    out_dir = f'{OUT_ROOT}/{args.assay}'
    try:
        save_matrix(out_dir, tag, out, mutants)
        print(f'wrote {out_dir}/{tag}.npy  shape {out.shape}')
    except OSError as exc:
        print(f'could not write the matrix ({exc}); it is in the checkpoint '
              f'at {ckpt} and re-running this command will copy it over',
              flush=True)
    print(f'TOTAL {time.time()-t0:.0f}s')
    print('DONE')


if __name__ == '__main__':
    main()
