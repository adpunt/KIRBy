"""
Multi-view Classification Benchmark Loaders

The datasets the multi-view feature selection literature reports on. Each
feature set (view) is one base representation, so `reps` goes straight into
find_best_hybrid as base_reps.

  mfeat           UCI Multiple Features, 2,000 handwritten digits, 10 classes,
                  200 per class. Views: fou 76, fac 216, kar 64, pix 240,
                  zer 47, mor 6 features.
  caltech101_7    1,474 images, 7 classes (34 to 798 per class).
  caltech101_20   2,386 images, 20 classes (33 to 798 per class).
                  Both Caltech views: gabor 48, wavelet_moments 40,
                  centrist 254, hog 1984, gist 512, lbp 928 features.
  nuswide_object  30,000 images, 31 classes (139 to 3,573 per class).
                  Views: color_histogram 64, color_moments 225,
                  color_correlogram 144, edge_direction 73,
                  wavelet_texture 128 features.
  reuters         18,758 documents first written in English, 6 classes
                  (1,207 to 5,102 per class). Views: the English original and
                  its machine translations into French, German, Italian and
                  Spanish; 21,531 / 24,892 / 34,251 / 15,506 / 11,547 word
                  features. SPARSE: returned as scipy.sparse CSR matrices,
                  about 0.2-0.7% nonzero. Densified, the five views are about
                  8 GB in float32.

WHERE EACH FILE CAME FROM (downloaded 2026-10-23 to
/Volumes/seagate/multiview_benchmarks/):

  mfeat_raw/mfeat-{fou,fac,kar,pix,zer,mor}
      UCI Machine Learning Repository, dataset 72, "Multiple Features",
      https://archive.ics.uci.edu/static/public/72/multiple+features.zip
      Donated by R.P.W. Duin. Rows are in class order, 200 per class.
      mvdata_li2015/handwritten.mat holds the same matrices (fou checked
      identical); the UCI files are read here because they are the original.

  mvdata_li2015/{Caltech101-7,Caltech101-20,NUSWIDEOBJ}.mat
      The .mat files released with Li, Nie, Huang, Huang, "Large-Scale
      Multi-View Spectral Clustering via Bipartite Graph", AAAI 2015.
      README: https://github.com/yeqinglee/mvdata ; files on Google Drive,
      folder 1O_3YmthAZGiq1ZPSdE74R7Nwos2PmnHH. Not an official release of
      Caltech-101 or NUS-WIDE: the features were computed or reformatted by
      those authors, and how is not documented there.
      NUS-WIDE-Object: every view carries one trailing all-zero feature
      (65/226/145/74/129 in the file, 64/225/144/73/128 in the NUS-WIDE
      paper). It is dropped here. The values are not the raw nonnegative
      histograms of the NUS release (they contain negatives), so the file
      was rescaled by whoever made it. The file does not record which image
      is in the official NUS-WIDE train or test half.

  reuters_raw/rcv1rcv2aminigoutte/ (not read by this loader)
      UCI dataset 259, "Reuters RCV1 RCV2 Multilingual, Multiview Text
      Categorization Test collection", Amini, Usunier, Goutte, NIPS 2009.
      111,740 documents over five source languages; ships no train/test split.
  mvdata_li2015/Reuters.mat (read by this loader)
      The English-source subset of the UCI collection as released by Li et
      al. 2015: 18,758 documents, matching EN/Index_EN-EN in row count.

None of these loaders assigns fold groups: `groups` is None. Which split to
use is the owner's decision (random stratified splits are what the published
results used; KIRBy's own rule is no random splits).

All loaders return a standard dict:
    reps: Dict[str, np.ndarray | scipy.sparse.csr_matrix]   one per view
    labels: np.ndarray        integer class labels, 0 .. n_classes-1
    task_type: str            'classification'
    groups: None
    n_classes: int
    class_names: List[str]
    domain: str               'multiview'
    dataset_name: str
    source: str               where the file came from
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

MULTIVIEW_DATASETS = ['mfeat', 'caltech101_7', 'caltech101_20',
                      'nuswide_object', 'reuters']

_DEFAULT_ROOT = Path('/Volumes/seagate/multiview_benchmarks')

_MFEAT_VIEWS = ['fou', 'fac', 'kar', 'pix', 'zer', 'mor']

# View order inside the Li 2015 .mat files, read off their README and the
# 'feanames' / 'langs' variables where present.
_CALTECH_VIEWS = ['gabor', 'wavelet_moments', 'centrist', 'hog', 'gist', 'lbp']
_NUS_VIEWS = ['color_histogram', 'color_moments', 'color_correlogram',
              'edge_direction', 'wavelet_texture']
_REUTERS_VIEWS = ['english', 'french', 'german', 'italian', 'spanish']

_LI2015_SOURCE = ('Li, Nie, Huang, Huang, AAAI 2015 release '
                  '(github.com/yeqinglee/mvdata, Google Drive)')


def _root(data_dir: Optional[str]) -> Path:
    """Env var > explicit kwarg > the seagate default."""
    env = os.environ.get('KIRBY_MULTIVIEW_DIR')
    if env:
        return Path(env)
    if data_dir is not None:
        return Path(data_dir)
    return _DEFAULT_ROOT


def _encode_labels(y) -> np.ndarray:
    y = np.asarray(y).ravel()
    _, codes = np.unique(y, return_inverse=True)
    return codes.astype(np.int64)


def _result(name, reps, labels, class_names, source) -> dict:
    return {
        'reps': reps,
        'labels': labels,
        'task_type': 'classification',
        'groups': None,
        'n_classes': int(len(np.unique(labels))),
        'class_names': list(class_names),
        'domain': 'multiview',
        'dataset_name': name,
        'source': source,
    }


def load_mfeat(data_dir: Optional[str] = None) -> dict:
    """UCI Multiple Features. Rows are in class order, 200 per digit."""
    d = _root(data_dir) / 'mfeat_raw'
    reps = {}
    for v in _MFEAT_VIEWS:
        path = d / f'mfeat-{v}'
        if not path.exists():
            raise FileNotFoundError(
                f'{path} missing. Download '
                f'https://archive.ics.uci.edu/static/public/72/multiple+features.zip '
                f'and unzip it into {d}.')
        reps[v] = np.loadtxt(path).astype(np.float32)
    n = {x.shape[0] for x in reps.values()}
    if n != {2000}:
        raise ValueError(f'mfeat views have row counts {n}, expected 2000')
    labels = np.repeat(np.arange(10), 200)
    return _result('mfeat', reps, labels, [str(i) for i in range(10)],
                   'UCI dataset 72, Multiple Features')


def _load_li2015(fname: str, view_names: List[str], drop_zero_last=False):
    import scipy.io as sio
    import scipy.sparse as sp

    path = _root(None) / 'mvdata_li2015' / fname
    if not path.exists():
        raise FileNotFoundError(
            f'{path} missing. Download the Google Drive folder linked from '
            f'github.com/yeqinglee/mvdata into {path.parent}.')
    m = sio.loadmat(str(path))
    views = list(m['X'].ravel())
    if len(views) != len(view_names):
        raise ValueError(f'{fname}: {len(views)} views, expected '
                         f'{len(view_names)}')
    reps = {}
    for name, x in zip(view_names, views):
        if sp.issparse(x):
            x = sp.csr_matrix(x, dtype=np.float32)
        else:
            x = np.asarray(x, dtype=np.float32)
            if drop_zero_last:
                if np.any(x[:, -1] != 0):
                    raise ValueError(f'{fname} {name}: last feature is not '
                                     f'all zero; the file is not the one '
                                     f'this loader was written for')
                x = x[:, :-1]
        reps[name] = x
    raw_y = np.asarray(m['Y']).ravel()
    labels = _encode_labels(raw_y)
    names = None
    for key in ('categories', 'category'):
        if key in m:
            names = [str(np.asarray(c).ravel()[0]) for c in m[key].ravel()]
    if names is None or len(names) != len(np.unique(raw_y)):
        names = [str(c) for c in np.unique(raw_y)]
    return reps, labels, names


def load_caltech101(n_classes: int) -> dict:
    if n_classes not in (7, 20):
        raise ValueError('Caltech101 multi-view exists as 7 or 20 classes')
    # Class names come from 'categories', which is in label order: its
    # 'lenSmp' counts equal the per-label row counts (checked 2026-10-23).
    # 'cateset' in the same file is in a different order and is not used.
    reps, labels, names = _load_li2015(f'Caltech101-{n_classes}.mat',
                                       _CALTECH_VIEWS)
    return _result(f'caltech101_{n_classes}', reps, labels, names,
                   _LI2015_SOURCE)


def load_nuswide_object() -> dict:
    reps, labels, names = _load_li2015('NUSWIDEOBJ.mat', _NUS_VIEWS,
                                       drop_zero_last=True)
    return _result('nuswide_object', reps, labels, names, _LI2015_SOURCE)


def load_reuters() -> dict:
    """English-source subset. Views are scipy.sparse CSR, not dense."""
    reps, labels, names = _load_li2015('Reuters.mat', _REUTERS_VIEWS)
    return _result('reuters', reps, labels, names,
                   _LI2015_SOURCE + '; subset of UCI dataset 259 '
                   '(Amini, Usunier, Goutte, NIPS 2009)')


def load_multiview(name: str, data_dir: Optional[str] = None) -> dict:
    if name == 'mfeat':
        return load_mfeat(data_dir)
    if name == 'caltech101_7':
        return load_caltech101(7)
    if name == 'caltech101_20':
        return load_caltech101(20)
    if name == 'nuswide_object':
        return load_nuswide_object()
    if name == 'reuters':
        return load_reuters()
    raise ValueError(f"Unknown multi-view dataset '{name}'. "
                     f'Available: {MULTIVIEW_DATASETS}')
