#!/usr/bin/env python3
"""ESM-1v embeddings for one ProteinGym assay's variant sequences.

NOT A RESULT. This writes a representation matrix to disk; scoring is
b08_proteingym_score_representation.py.

What ProteinGym's "ESM-1v Embeddings" baseline reads (ProteinNPT,
proteinnpt/baselines/model.py, last_layer_index = 33 for ESM1v): the layer-33
hidden states of the mutated sequence. Their head then runs a convolution over
the residues before pooling, so this matrix is their input, not their protocol
-- see NEW_DATASETS.md, "Published protocol: the supervised baselines".

Their code pools over the full token axis including the BOS and EOS tokens,
because the mask is not applied. Every sequence in a substitution assay is the
same length, so the only thing pooling-with-no-mask changes here is that BOS
and EOS are averaged in. This reproduces that: pool over all tokens the
tokenizer emits. 1,280 features per variant.

Output: /Volumes/seagate/proteingym_embeddings/<assay>/esm1v_mean_l33.npy plus
a .mutants.txt in the same row order, so the matrix can be lined up with the
loader without re-running anything.

Checkpointing, added after a run died silently at 1,376 of 2,172 variants and
lost two hours of forward passes. Re-running the same command resumes from the
checkpoint and never recomputes a row it already has. Every write goes to a
temporary name and is renamed, so a seagate drop mid-write cannot leave a
half-written .npy in place of a good one. A checkpoint write that fails is
reported and the run continues.

    python tests/experiments/benchmarks/b05_proteingym_esm1v_embeddings.py \
        ESTA_BACSU_Nutschel_2020 --batch 16 --checkpoint-every 160

Measured on the laptop: 4.08 s per variant at 212 residues, 8,870 s for 2,172
variants.

    --limit N             embed only the first N variants, for a timed check
    --checkpoint-every N  rows per checkpoint, default 160
"""
import argparse
import os
import sys
import time

os.environ.setdefault('HF_HOME', '/Volumes/seagate/hf_cache')

REPO_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', '..', 'src')
sys.path.insert(0, os.path.abspath(REPO_SRC))

import numpy as np
import torch
from transformers import EsmModel, EsmTokenizer

from kirby.datasets.proteingym import load_proteingym_dataset

OUT_ROOT = '/Volumes/seagate/proteingym_embeddings'
CKPT_ROOT = '/Volumes/seagate/benchmarks_protocol/proteingym/checkpoints'
MODEL_ID = 'facebook/esm1v_t33_650M_UR90S_1'
LAYER = 33


def save_matrix(out_dir, tag, X, mutants):
    """Write the matrix and its row order, temporary name then rename."""
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

    tok = EsmTokenizer.from_pretrained(MODEL_ID)
    model = EsmModel.from_pretrained(MODEL_ID)
    model.eval()
    n_layers = model.config.num_hidden_layers
    if n_layers != LAYER:
        raise SystemExit(f'expected {LAYER} layers, model reports {n_layers}')
    torch.set_num_threads(os.cpu_count() or 1)
    print(f'model loaded in {time.time()-t0:.0f}s, {n_layers} layers, '
          f'{model.config.hidden_size} hidden size, {torch.get_num_threads()} '
          f'threads', flush=True)

    tag = f'esm1v_mean_l33{"_first%d" % args.limit if args.limit else ""}'
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt = f'{args.ckpt_dir}/{args.assay}__{tag}.npz'

    out = np.zeros((len(seqs), model.config.hidden_size), dtype=np.float32)
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

    t_embed = time.time()
    done = start
    with torch.no_grad():
        for i in range(start, len(seqs), args.batch):
            chunk = seqs[i:i + args.batch]
            enc = tok(chunk, return_tensors='pt', padding=True)
            # output_hidden_states gives n_layers+1 tensors, index 0 being the
            # embedding layer, so hidden_states[33] is layer 33.
            hs = model(**enc, output_hidden_states=True).hidden_states[LAYER]
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
