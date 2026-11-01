#!/usr/bin/env python3
"""Score one saved representation matrix on a ProteinGym assay's own folds.

NOT the published protocol. ProteinGym's three embedding baselines run a
convolution over the residues, then mean-pool, then a linear head, and add the
assay's zero-shot score outside the network with a learned scalar initialised
to 1.0 (ProteinNPT, baselines/model.py:80-120 and 165-230). This script fits a
tree or an SVM to features that are already pooled. Any write-up comparing a
number from here to their published table must say that the protocol differs:

  - no convolution over residues, because the features arrive pooled
  - a tree or an SVM instead of a trained linear head
  - --with-zero-shot adds the paired zero-shot score as one more feature, which
    is not their mechanism either; a tree cannot express "plus w times this
    score" with w learned and starting at 1.0

Their published scores for comparison are in
/Volumes/seagate/benchmarks_protocol/proteingym/
DMS_substitutions_Spearman_DMS_level_fold_<scheme>_5.csv, one number per assay
per method.

    python tests/experiments/benchmarks/b08_proteingym_score_representation.py \
        ESTA_BACSU_Nutschel_2020 esm1v_mean_l33 --models lightgbm,svm

The matrix is read from
/Volumes/seagate/proteingym_embeddings/<assay>/<tag>.npy and its row order from
<tag>.mutants.txt. Rows are re-ordered to the loader's order rather than
assumed to match, and an incomplete matrix is refused.
"""
import argparse
import csv
import json
import os
import sys
import time

REPO_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', '..', 'src')
sys.path.insert(0, os.path.abspath(REPO_SRC))

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVR

from kirby.datasets.proteingym import load_proteingym_dataset
from kirby.scaling import FeatureScaler

EMB_ROOT = '/Volumes/seagate/proteingym_embeddings'
ZS_DIR = '/Volumes/seagate/proteingym_raw/ex_zeroshot'
PUB_DIR = '/Volumes/seagate/benchmarks_protocol/proteingym'
RESULTS = '/Volumes/seagate/benchmarks_protocol/results'

# Which zero-shot score pairs with which representation, by the embedding
# model that produced it.
PAIRED_ZS = {'esm1v': 'ESM1v_ensemble',
             'msat': 'MSA_Transformer_ensemble',
             'tranception': 'Tranception_L'}

PUBLISHED_METHOD = {'esm1v': 'ESM-1v Embeddings',
                    'msat': 'MSA Transformer Embeddings',
                    'tranception': 'Tranception Embeddings'}


def model_for(name):
    if name == 'lightgbm':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(verbose=-1)
    if name == 'svm':
        return SVR()
    raise SystemExit(f'unknown model {name}')


def load_matrix(assay, tag, mutants):
    path = f'{EMB_ROOT}/{assay}/{tag}.npy'
    order_path = f'{EMB_ROOT}/{assay}/{tag}.mutants.txt'
    X = np.load(path)
    with open(order_path) as fh:
        saved = [ln.strip() for ln in fh if ln.strip()]
    if len(saved) != X.shape[0]:
        raise SystemExit(f'{tag}: {X.shape[0]} rows but {len(saved)} mutants '
                         f'listed, refusing to score it')
    if set(saved) != set(mutants):
        missing = len(set(mutants) - set(saved))
        raise SystemExit(f'{tag}: the matrix is incomplete, {missing} of '
                         f'{len(mutants)} variants are not in it, refusing '
                         f'to score it')
    pos = {m: i for i, m in enumerate(saved)}
    return X[[pos[m] for m in mutants]]


def load_zero_shot(assay, column, mutants):
    path = f'{ZS_DIR}/{assay}.csv'
    if not os.path.exists(path):
        raise SystemExit(f'no zero-shot file for {assay} at {path}')
    with open(path, newline='') as fh:
        rows = {r['mutant']: r for r in csv.DictReader(fh)}
    if column not in next(iter(rows.values())):
        raise SystemExit(f'{column} is not a column of {path}')
    missing = [m for m in mutants if m not in rows]
    if missing:
        raise SystemExit(f'{len(missing)} variants have no zero-shot score')
    return np.array([[float(rows[m][column])] for m in mutants],
                    dtype=np.float32)


def published(assay, tag, scheme):
    """Their number for the matching method, or None."""
    key = next((k for k in PUBLISHED_METHOD if tag.startswith(k)), None)
    if key is None:
        return None, None
    path = f'{PUB_DIR}/DMS_substitutions_Spearman_DMS_level_fold_{scheme}_5.csv'
    if not os.path.exists(path):
        return PUBLISHED_METHOD[key], None
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            if r['DMS_id'] == assay:
                return PUBLISHED_METHOD[key], r.get(PUBLISHED_METHOD[key])
    return PUBLISHED_METHOD[key], None


def fold_scores(X, y, g, model_name):
    out = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        sc = FeatureScaler()
        m = model_for(model_name)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        out.append(spearmanr(y[te], m.predict(sc.transform(X[te]))).correlation)
    return np.array(out, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('assay')
    ap.add_argument('tag', help='e.g. esm1v_mean_l33')
    ap.add_argument('--models', default='lightgbm,svm')
    ap.add_argument('--scheme', default='modulo',
                    choices=['modulo', 'contiguous'])
    ap.add_argument('--with-zero-shot', action='store_true',
                    help='also score the matrix plus the paired zero-shot '
                         'score as one more feature; not their mechanism')
    args = ap.parse_args()

    t0 = time.time()
    d = load_proteingym_dataset(f'pgym_{args.assay}', fold_scheme=args.scheme)
    mutants = [f"{m[0]['wt_aa']}{m[0]['pos']}{m[0]['mut_aa']}"
               for m in d['mutations']]
    y = np.asarray(d['labels'], dtype=float)
    g = np.asarray(d['groups'])
    X = load_matrix(args.assay, args.tag, mutants)
    print(f'{args.assay}  {args.tag}  {X.shape[0]} variants, {X.shape[1]} '
          f'features, fold scheme {args.scheme}', flush=True)

    held = [len(te) for _, te in GroupKFold(n_splits=5).split(X, y, g)]
    print(f'held-out sizes {held}', flush=True)

    runs = [('plain', X)]
    if args.with_zero_shot:
        key = next((k for k in PAIRED_ZS if args.tag.startswith(k)), None)
        if key is None:
            raise SystemExit(f'no paired zero-shot score known for {args.tag}')
        zs = load_zero_shot(args.assay, PAIRED_ZS[key], mutants)
        runs.append(('plus_zs', np.hstack([X, zs])))

    out = {'assay': args.assay, 'tag': args.tag, 'scheme': args.scheme,
           'n_variants': int(X.shape[0]), 'n_features': int(X.shape[1]),
           'held_out_sizes': held, 'scores': {}}
    for which, Xi in runs:
        for name in args.models.split(','):
            s = fold_scores(Xi, y, g, name)
            out['scores'][f'{name}__{which}'] = {
                'per_fold': [round(float(v), 4) for v in s],
                'mean': round(float(np.nanmean(s)), 4),
                'spread': round(float(np.nanmax(s) - np.nanmin(s)), 4),
                'n_features': int(Xi.shape[1]),
            }
            print(f'  {name:9} {which:8} {Xi.shape[1]:5} features  '
                  f'mean {np.nanmean(s):.4f}  spread '
                  f'{np.nanmax(s)-np.nanmin(s):.4f}  per fold '
                  + ' '.join(f'{v:.4f}' for v in s), flush=True)

    method, pub = published(args.assay, args.tag, args.scheme)
    if method:
        out['published_method'] = method
        out['published_score'] = pub
        print(f'their published {method} on this assay under '
              f'fold_{args.scheme}_5: {pub}', flush=True)
        print('their protocol differs: convolution over residues before '
              'pooling, a trained linear head, and the zero-shot score added '
              'outside the network. See this file\'s docstring.', flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    path = f'{RESULTS}/b08_{args.assay}__{args.tag}__{args.scheme}.json'
    tmp = path + '.partial'
    with open(tmp, 'w') as fh:
        json.dump(out, fh, indent=2)
    os.replace(tmp, path)
    print(f'wrote {path}')
    print(f'TOTAL {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
