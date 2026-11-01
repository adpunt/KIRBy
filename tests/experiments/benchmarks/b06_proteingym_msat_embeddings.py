#!/usr/bin/env python3
"""MSA Transformer embeddings for one ProteinGym assay's variant sequences.

NOT A RESULT. Writes a representation matrix; scoring is
b08_proteingym_score_representation.py.

Every rule here was read out of ProteinNPT, with the line it came from. Nothing
from their repo is imported or executed; the model itself comes from fair-esm,
which is what their config points at (Embeddings_MSAT_final.json:
embedding_model_location "ESM/MSA_Transformer/esm_msa1b_t12_100M_UR50S.pt").

  the alignment     hhfilter -cov 75 -id 90 -qid 0 over the assay's a2m with
                    dots turned to dashes and everything upper-cased
                    (msa_utils.py:396-398). hhfilter reads at most 65,535
                    sequences, so on an alignment larger than that their
                    baseline sees only the first 65,535.
  preprocessing     drop the columns that are gaps in the first (reference)
                    sequence, drop sequences whose gap fraction exceeds 0.5,
                    drop sequences carrying a letter outside
                    ACDEFGHIKLMNPQRSTVWY and '-' (msa_utils.py:189-207,
                    150-172, defaults at lines 49-51).
  the sample        384 sequences per variant (embeddings_subs.sh:10,
                    num_MSA_sequences=384). The reference sequence first, then
                    383 drawn from the rest WITH REPLACEMENT by
                    random.choices weighted by the shipped weights
                    (msa_utils.py:485-487). Drawn ONCE and reused for every
                    variant, because their code only samples when
                    model.MSA_sample_sequences is None
                    (embedding_utils.py:98).
  the batch         each variant is its own MSA batch, the variant placed
                    first and the 384 sampled sequences after it
                    (embedding_utils.py:157). That makes 385 rows, not 384 --
                    their own assert at line 128 counts raw_batch_size +
                    num_MSA_sequences.
  the embedding     model(tokens, repr_layers=[12]), then row 0 of each batch
                    (embedding_utils.py:247, 263). 768 features after pooling.
  pooling           their head is AA_embeddings_mean_pooled
                    (Embeddings_MSAT_final.json), so mean over residues.

Their own precompute script runs batch_size=1 and does not pass
--fast_MSA_mode (scripts/embeddings/embeddings_subs.sh), so one variant per MSA
batch is their mechanism for substitution assays, not a choice. Row attention
makes the whole batch depend on the variant, so the 384 identical alignment
rows cannot be computed once and reused.

ESTA_BACSU verified 2026-10-26: 234,310 sequences in the a2m, 65,535 read by
hhfilter, 23,479 after filtering, 23,428 after preprocessing, against their
shipped weights file of 23,428. Zero difference.

The weights file is the MSA-Transformer-specific one. Their archive ships two
per assay at two different lengths and only this one matches the filtered
alignment:
  DMS_msa_weights/DMS_msa_weights_for_MSA_Transformer/<assay>_theta_0.2.npy

Measured on the laptop: 625 s per variant, so 377 hours for 2,172 variants.
This needs a GPU.

    --limit N             embed only the first N variants, for a timed check
    --seed                seed for the one weighted draw; in the output name
    --checkpoint-every N  variants per checkpoint, default 10
"""
import argparse
import os
import random
import sys
import time

REPO_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', '..', 'src')
sys.path.insert(0, os.path.abspath(REPO_SRC))

import numpy as np
import torch

from kirby.datasets.proteingym import load_proteingym_dataset

OUT_ROOT = '/Volumes/seagate/proteingym_embeddings'
CKPT_ROOT = '/Volumes/seagate/benchmarks_protocol/proteingym/checkpoints'
TORCH_HOME = '/Volumes/seagate/torch_hub'
ALPHABET = set('ACDEFGHIKLMNPQRSTVWY')
THRESHOLD_SEQUENCE_FRAC_GAPS = 0.5
NUM_MSA_SEQUENCES = 384
LAYER = 12


def read_a2m(path):
    names, seqs, name, buf = [], [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith('>'):
                if name is not None:
                    names.append(name)
                    seqs.append(''.join(buf))
                name, buf = line, []
            else:
                buf.append(line)
    if name is not None:
        names.append(name)
        seqs.append(''.join(buf))
    return names, seqs


def preprocess(names, seqs):
    """Their MSA_processing preprocessing, reimplemented. See the docstring."""
    seqs = [s.replace('.', '-').upper() for s in seqs]
    keep = [i for i, a in enumerate(seqs[0]) if a != '-']
    seqs = [''.join(s[i] for i in keep) for s in seqs]

    frac = np.array([[c == '-' for c in s] for s in seqs]).mean(axis=1)
    sel = frac <= THRESHOLD_SEQUENCE_FRAC_GAPS
    names = [n for n, k in zip(names, sel) if k]
    seqs = [s for s, k in zip(seqs, sel) if k]

    ok = [all(c in ALPHABET or c == '-' for c in s) for s in seqs]
    names = [n for n, k in zip(names, ok) if k]
    seqs = [s for s, k in zip(seqs, ok) if k]
    return names, seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('assay', nargs='?', default='ESTA_BACSU_Nutschel_2020')
    ap.add_argument('--msa', default='/Volumes/seagate/proteingym_raw/ex_msa/'
                    'hhfiltered/ESTA_BACSU_full_11-26-2021_b03_hhfiltered_'
                    'cov_75_maxid_90_minid_0.a2m')
    ap.add_argument('--weights', default='/Volumes/seagate/proteingym_raw/'
                    'ex_msa_weights_msat/ESTA_BACSU_theta_0.2.npy')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=2023)
    ap.add_argument('--checkpoint-every', type=int, default=10)
    ap.add_argument('--ckpt-dir', default=CKPT_ROOT)
    ap.add_argument('--check-alignment-only', action='store_true',
                    help='print the surviving-sequence count against their '
                         'weights file and stop, computing nothing')
    args = ap.parse_args()

    os.environ.setdefault('TORCH_HOME', TORCH_HOME)
    t0 = time.time()

    d = load_proteingym_dataset(f'pgym_{args.assay}')
    variants = list(d['sequences'])
    mutants = [f"{m[0]['wt_aa']}{m[0]['pos']}{m[0]['mut_aa']}"
               for m in d['mutations']]
    if args.limit:
        variants, mutants = variants[:args.limit], mutants[:args.limit]

    names, seqs = read_a2m(args.msa)
    print(f'alignment: {len(names)} sequences before preprocessing',
          flush=True)
    names, seqs = preprocess(names, seqs)
    w_all = np.load(args.weights)
    print(f'alignment: {len(names)} after preprocessing, weights file holds '
          f'{w_all.shape[0]}', flush=True)
    if len(names) != w_all.shape[0]:
        raise SystemExit('the filtered alignment and the weights do not line '
                         'up, so the weighted draw cannot follow their code')
    if len(seqs[0]) != len(variants[0]):
        raise SystemExit(f'alignment columns {len(seqs[0])} but variant '
                         f'sequences are {len(variants[0])} residues')
    print('the filtered alignment and their weights line up: one weight per '
          'surviving sequence, the reference sequence included. Their '
          'compute_sequence_weights deletes the reference weight from the '
          'loaded dictionary in memory (msa_utils.py:429-433), so the file on '
          'disk still carries it.', flush=True)
    if args.check_alignment_only:
        return

    # One weighted draw, reused for every variant (embedding_utils.py:98).
    # random.choices draws WITH replacement (msa_utils.py:486), and their
    # non-reference weights are the file with the reference's weight deleted
    # (msa_utils.py:429-433).
    random.seed(args.seed)
    pool = list(zip(names[1:], seqs[1:]))
    sample = [(names[0], seqs[0])]
    sample += random.choices(pool, weights=w_all[1:].tolist(),
                             k=NUM_MSA_SEQUENCES - 1)
    msa_rows = [s for _, s in sample]
    print(f'sampled {len(msa_rows)} alignment rows, seed {args.seed}, '
          f'{len(set(msa_rows))} of them distinct', flush=True)

    import esm
    model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    model.eval()
    bc = alphabet.get_batch_converter()
    torch.set_num_threads(os.cpu_count() or 1)
    print(f'model loaded in {time.time()-t0:.0f}s, {torch.get_num_threads()} '
          f'threads', flush=True)

    tag = (f'msat_mean_l12_seed{args.seed}'
           f'{"_first%d" % args.limit if args.limit else ""}')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt = f'{args.ckpt_dir}/{args.assay}__{tag}.npz'

    out = np.zeros((len(variants), 768), dtype=np.float32)
    start = 0
    if os.path.exists(ckpt):
        z = np.load(ckpt)
        if [str(m) for m in z['mutants']] == mutants:
            start = int(z['done'])
            out[:start] = z['X'][:start]
            print(f'resuming from the checkpoint at {start}/{len(variants)} '
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
    with torch.no_grad():
        for i, (mut, seq) in enumerate(zip(mutants, variants)):
            if i < start:
                continue
            rows = [(mut, seq)] + [(f'msa{j}', s)
                                   for j, s in enumerate(msa_rows)]
            _, _, tokens = bc([rows])
            rep = model(tokens, repr_layers=[LAYER])['representations'][LAYER]
            # row 0 of the batch is the variant (embedding_utils.py:263),
            # then mean over residues (AA_embeddings_mean_pooled)
            out[i] = rep[:, 0, :, :].mean(dim=1).cpu().numpy()
            done = i + 1
            el = time.time() - t_embed
            n_new = done - start
            print(f'  {done}/{len(variants)}  {el:.0f}s  '
                  f'{el/n_new:.2f}s per variant  projected remaining '
                  f'{el/n_new*(len(variants)-done)/3600:.1f} h', flush=True)
            if done % args.checkpoint_every == 0 or done == len(variants):
                write_ckpt(done)
                print(f'  checkpoint at {done}', flush=True)

    write_ckpt(len(variants))
    out_dir = f'{OUT_ROOT}/{args.assay}'
    try:
        os.makedirs(out_dir, exist_ok=True)
        # save appends .npy to a path that lacks it, so write through a handle
        # or the temporary name is not what gets renamed.
        tmp = f'{out_dir}/{tag}.npy.partial'
        with open(tmp, 'wb') as fh:
            np.save(fh, out)
        os.replace(tmp, f'{out_dir}/{tag}.npy')
        with open(f'{out_dir}/{tag}.mutants.txt', 'w') as fh:
            fh.write('\n'.join(mutants) + '\n')
        print(f'wrote {out_dir}/{tag}.npy  shape {out.shape}')
    except OSError as exc:
        print(f'could not write the matrix ({exc}); it is in the checkpoint '
              f'at {ckpt} and re-running this command will copy it over',
              flush=True)
    print(f'TOTAL {time.time()-t0:.0f}s')
    print('DONE')


if __name__ == '__main__':
    main()
