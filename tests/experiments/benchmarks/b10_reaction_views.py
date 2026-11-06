#!/usr/bin/env python3
"""Reaction view screen: every candidate representation of a reaction, for
the three yield datasets, scored alone under each dataset's decided models on
KIRBy's own outer folds; each view also scored with the cgr_bond_changes
counts added; linear CKA and error overlap between views.

Object: one reaction. Target: yield (%). Metric: the dataset's metric from
_shared.get_domain_metric (RMSE on all three; lower is better).
Datasets (tasks): buchwald_hartwig, bh_doyle_plates, suzuki.
Folds: kirby.find_best_hybrid.nested_cv_outer_folds, 5 folds, seed 42, the
loader's groups (additive on the two Buchwald-Hartwig sets, ligand on
suzuki): the folds every KIRBy experiment uses. A loader without groups is
refused (no random splits).
Models: experiments.decided_grid.decided_models(dataset, 'general'), built
by _shared.get_model_factories. Each fold: FeatureScaler fitted on the
training rows (0/1 features pass through), model fitted on the training
rows, scored on the held-out rows.

Views (decided by the owner, 2026-11-01):

  the dataset's pool, complexity_helpers.DATASET_REPS[ds]['general'] (10;
      cgr_morgan is in all three). Read from, or built into, the KIRBy rep
      cache (<KIRBY_CACHE_DIR>/rep_cache/<ds>/<view>.npy) through
      _shared._load_data_impl, the path every experiment uses.
  cgr_atompair, cgr_torsion, cgr_bond_changes: hand-written CGR views in
      src/kirby/representations/reaction.py, registered in
      _shared._rxn_generators; same cache.
  chytorch_graphormer, cgr_smiles_lm: built by this script into
      <results>/b10_views/views/<ds>/<view>.npy, never read from the rep
      cache: the cached buchwald_hartwig chytorch_graphormer of 2026-05-01 is
      all zeros, and a cached cgr_smiles_lm does not record which checkpoint
      made it. cgr_smiles_lm refuses to run without the CGR-pretrained
      checkpoint (create_cgr_smiles_lm would otherwise fall back to base
      ChemBERTa without saying so in the matrix).
  chemprop_cgr: chemprop 1.6.1 in reaction mode (Heid and Green 2022,
      --reaction, reaction_mode reac_diff, every other setting chemprop's
      default), trained per outer fold on that fold's training rows. Its
      validation rows (early stopping) are one group-aware inner fold of the
      training rows (nested_cv_outer_folds on the training rows, 5 folds,
      seed 42, the first fold); the held-out rows are chemprop's test set,
      used only for the score recorded beside the view. The view is the
      learned MPN encoder output (chemprop's fingerprint_type 'MPN', the FFN
      input), for the fold's training and held-out rows.
  DOPtools CircuS and ChyLine (cgr_circus, cgr_chyline) were dropped by the
      owner, 2026-11-02: doptools needs chython>=1.78 (3.x on PyPI) and
      chytorch 1.53 needs the installed chython 1.68, so they cannot share an
      environment. Three hand-written CGR fingerprints are screened already.

The fold-trained view (chemprop_cgr) is built per
(dataset, fold); every other view once per dataset. A build refuses (exits
non-zero, writes no result file) a matrix that fails
opig_cgr_comparison.refusal_reason: wrong shape, a non-finite value, or every
feature constant (all zero included).

Every view other than cgr_bond_changes is scored twice: alone, and with the
30 cgr_bond_changes counts appended (same folds, same model). The owner's
rule: a small set of counts is an augmentation, so the question is whether it
helps a standard view. cgr_bond_changes is also scored alone.

Stages (one command per ledger unit):
    build      --tasks DS --views V                 a fold-independent view
    build      --tasks DS --views V --folds K       a fold-trained view
    evaluate   --tasks DS --views V --models M [--plus-bond-changes]
    summarise  summary.md

Output under KIRBY_BENCH_RESULTS (the ledger sets <run>/reactions):
    b10_views/built/<ds>/<view>.json                  build record (fold-independent)
    b10_views/views/<ds>/<view>.npy                   chytorch_graphormer, cgr_smiles_lm
    b10_views/fold_views/<ds>/<view>_fold<k>.npz      fold-trained views
    b10_views/fold_views/<ds>/<view>_fold<k>.json     their build record
    b10_views/<ds>/<view>__<model>[__plus_bond_changes].json   + .oof.npy
    b10_views/summary.md

    python tests/experiments/benchmarks/b10_reaction_views.py --stage build \\
        --tasks buchwald_hartwig --views cgr_atompair
    python tests/experiments/benchmarks/b10_reaction_views.py --stage evaluate \\
        --tasks buchwald_hartwig --views cgr_atompair --models rf --plus-bond-changes
"""
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# LightGBM's OpenMP runtime before torch/rdkit (macOS duplicate-libomp crash),
# as opig_cgr_comparison.py does.
try:
    import lightgbm as _lightgbm  # noqa: F401
except Exception:
    pass

import argparse
import json
import sys
import time
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT / 'src', ROOT / 'tests', ROOT / 'tests' / 'experiments'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings('ignore')

RESULTS_DIR = Path(os.environ.get('KIRBY_BENCH_RESULTS',
                                  '/Volumes/seagate/benchmarks_protocol/results')) / 'b10_views'
TASKS = ['buchwald_hartwig', 'bh_doyle_plates', 'suzuki']
# Fold-independent CGR views added to every dataset's pool. cgr_morgan is in
# every pool already.
CGR_VIEWS = ['cgr_smiles_lm', 'chytorch_graphormer', 'cgr_atompair', 'cgr_torsion',
             'cgr_bond_changes']
# Built here, not read from the rep cache (see the docstring).
OWN_BUILD_VIEWS = ('chytorch_graphormer', 'cgr_smiles_lm')
FOLD_VIEWS = ['chemprop_cgr']
AUG_VIEW = 'cgr_bond_changes'
PLUS = 'plus_bond_changes'
N_FOLDS = 5
SEED = 42
INNER_FOLDS = 5


def log(msg):
    print('%s %s' % (time.strftime('%H:%M:%S'), msg), flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'b10_views.log', 'a') as f:
        f.write('%s %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg))


def pool(ds):
    from experiments.complexity.complexity_helpers import DATASET_REPS
    return list(DATASET_REPS[ds]['general'])


def views_for(ds):
    """Pool first, then the CGR views not already in it, then the
    fold-trained views."""
    p = pool(ds)
    return p + [v for v in CGR_VIEWS if v not in p] + FOLD_VIEWS


def models_for(ds):
    from experiments.decided_grid import decided_models
    return list(decided_models(ds, 'general'))


# ----------------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------------

def built_path(ds, view):
    return RESULTS_DIR / 'built' / ds / ('%s.json' % view)


def own_view_path(ds, view):
    return RESULTS_DIR / 'views' / ds / ('%s.npy' % view)


def fold_view_path(ds, view, k):
    return RESULTS_DIR / 'fold_views' / ds / ('%s_fold%d.npz' % (view, k))


def result_path(ds, view, model, plus=False):
    """The file the ledger reads for one evaluate unit."""
    return RESULTS_DIR / ds / ('%s__%s%s.json' % (view, model, ('__' + PLUS) if plus else ''))


def _write_json_last(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.%d' % os.getpid())
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


# ----------------------------------------------------------------------------
# data and folds
# ----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load(ds):
    """(rxn_smiles, labels float64, groups) from the loader _shared uses."""
    from kirby.datasets import reaction_yields as RY
    fn = {'buchwald_hartwig': RY.load_buchwald_hartwig,
          'bh_doyle_plates': RY.load_bh_doyle_plates,
          'suzuki': RY.load_suzuki}[ds]
    d = fn()
    if d.get('groups') is None:
        raise SystemExit('%s: the loader returned no groups (suzuki falls back to the TSVs '
                         'without the Excel file); refused, no random splits' % ds)
    return list(d['rxn_smiles']), np.asarray(d['labels'], np.float64), np.asarray(d['groups'])


@lru_cache(maxsize=None)
def folds(ds):
    from kirby.find_best_hybrid import nested_cv_outer_folds
    rx, y, g = load(ds)
    return nested_cv_outer_folds({}, y, g, N_FOLDS, SEED, 'regression')


def inner_val(ds, k):
    """One group-aware inner fold of outer fold k's training rows: chemprop's
    validation rows. Returns (inner_train_idx, inner_val_idx) as indices into
    the whole dataset."""
    from kirby.find_best_hybrid import nested_cv_outer_folds
    rx, y, g = load(ds)
    tr, _ = folds(ds)[k]
    itr, iva = nested_cv_outer_folds({}, y[tr], g[tr], INNER_FOLDS, SEED, 'regression')[0]
    return tr[itr], tr[iva]


def metric(ds):
    from experiments._shared import get_domain_metric
    rx, y, g = load(ds)
    name, fn, lower = get_domain_metric({'dataset_name': ds, 'task_type': 'regression',
                                         'domain': 'reaction', 'labels': y})
    return name, fn, lower


def refusal(X, n):
    from opig_cgr_comparison import refusal_reason
    return refusal_reason(X, n)


# ----------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------

def _cgr_checkpoint_or_exit():
    from kirby.representations.reaction import _find_cgr_smiles_checkpoint
    ck = _find_cgr_smiles_checkpoint()
    if not Path(ck).is_dir():
        raise SystemExit('cgr_smiles_lm: no CGR-pretrained checkpoint found (set KIRBY_CGR_CHECKPOINT; '
                         'on the server /data/stat-cadd/scat9264/kirby_data/cgr_smiles_pretrain/'
                         'checkpoint). create_cgr_smiles_lm would fall back to base ChemBERTa; '
                         'refused.')
    return ck


def build_fixed(ds, view):
    """One fold-independent view: check it, write its build record last."""
    out = built_path(ds, view)
    if out.exists():
        log('%s %s: already built' % (ds, view))
        return
    rx, y, g = load(ds)
    t0 = time.time()
    info = {}
    if view in OWN_BUILD_VIEWS:
        path = own_view_path(ds, view)
        if path.exists():
            X = np.load(path)
            info['source'] = 'this screen, an earlier attempt'
        else:
            from kirby.representations import reaction as R
            if view == 'cgr_smiles_lm':
                ck = _cgr_checkpoint_or_exit()
                info['checkpoint'] = ck
                X = R.create_cgr_smiles_lm(rx, checkpoint_dir=ck)
            else:
                X = R.create_chytorch_graphormer(rx)
            why = refusal(X, len(y))
            if why:
                raise SystemExit('%s %s: refused, %s' % (ds, view, why))
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.stem + '.tmp.%d.npy' % os.getpid())
            np.save(tmp, np.asarray(X, np.float32))
            os.replace(tmp, path)
            info['source'] = 'built by this screen'
    else:
        from experiments._shared import _load_data_impl, _rep_cache_dir
        cached = (_rep_cache_dir() / ds / ('%s.npy' % view)).exists()
        d = _load_data_impl(ds, rep_names=[view])
        if view not in d['reps']:
            raise SystemExit('%s %s: _load_data_impl could not produce it (see the warning above)'
                             % (ds, view))
        X = d['reps'][view]
        if len(d['labels']) != len(y) or not np.allclose(np.asarray(d['labels'], float), y):
            raise SystemExit('%s %s: the rep-cache loader and this script disagree on the rows'
                             % (ds, view))
        path = _rep_cache_dir() / ds / ('%s.npy' % view)
        info['source'] = 'KIRBy rep cache, existing' if cached else 'KIRBy rep cache, built now'
    why = refusal(X, len(y))
    if why:
        raise SystemExit('%s %s: refused, %s (file %s; move it aside to rebuild)'
                         % (ds, view, why, path))
    rec = dict(info, dataset=ds, view=view, path=str(path), n_rows=int(X.shape[0]),
               n_features=int(X.shape[1]), seconds=time.time() - t0,
               zero_rows=int((np.abs(np.asarray(X, float)).sum(1) == 0).sum()),
               result_status='ok')
    _write_json_last(out, rec)
    log('%s %s: %d x %d, %.0f s (%s)' % (ds, view, X.shape[0], X.shape[1], rec['seconds'],
                                         info['source']))


def _mapped(ds):
    from kirby.representations.reaction import _map_reactions_cached
    rx, y, g = load(ds)
    return _map_reactions_cached(rx)


def _build_chemprop(ds, k, work):
    """chemprop_cgr for outer fold k. Returns (X_train, X_test, info)."""
    import csv
    # RDKit 2023.09 patches pandas.io.formats.format.get_adjustment on import
    # (chemprop imports rdkit.Chem.PandasTools); pandas 2.2 moved it to
    # pandas.io.formats.printing, so the import fails on the server's
    # RDKit 2023.09.6 + pandas 2.2.0. Point the old name at the new one.
    import pandas.io.formats.format as _pd_format
    if not hasattr(_pd_format, 'get_adjustment'):
        from pandas.io.formats.printing import get_adjustment
        _pd_format.get_adjustment = get_adjustment
    import chemprop
    # chemprop 1.6.1 computes rmse with mean_squared_error(squared=False), and
    # scikit-learn 1.6 removed that argument: training died at the first validation
    # score. Same quantity, computed the way scikit-learn now asks.
    import numpy as _np
    from sklearn.metrics import mean_squared_error as _mse
    chemprop.train.metrics.rmse = lambda targets, preds: float(_np.sqrt(_mse(targets, preds)))
    rx, y, g = load(ds)
    tr, te = folds(ds)[k]
    itr, iva = inner_val(ds, k)
    mapped = _mapped(ds)
    work.mkdir(parents=True, exist_ok=True)

    def write(name, idx, with_y=True):
        p = work / name
        with open(p, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['rxn', 'yield'] if with_y else ['rxn'])
            for i in idx:
                w.writerow([mapped[i], float(y[i])] if with_y else [mapped[i]])
        return str(p)

    p_tr, p_va, p_te = write('train.csv', itr), write('val.csv', iva), write('test.csv', te)
    save = work / 'model'
    targs = chemprop.args.TrainArgs().parse_args([
        '--data_path', p_tr, '--separate_val_path', p_va, '--separate_test_path', p_te,
        '--dataset_type', 'regression', '--reaction', '--smiles_columns', 'rxn',
        '--target_columns', 'yield', '--save_dir', str(save), '--seed', str(SEED),
        '--pytorch_seed', str(SEED), '--num_workers', '0'])
    t0 = time.time()
    chemprop.train.cross_validate(args=targs, train_func=chemprop.train.run_training)
    t_train = time.time() - t0
    scores = {}
    sc = save / 'test_scores.csv'
    if sc.exists():
        rows = list(csv.DictReader(open(sc)))
        for r in rows:
            for key, val in r.items():
                if key != 'Task':
                    try:
                        scores[key] = float(val)
                    except (TypeError, ValueError):
                        pass
    order = np.concatenate([tr, te])
    p_all = write('fingerprint_rows.csv', order, with_y=False)
    fargs = chemprop.args.FingerprintArgs().parse_args([
        '--test_path', p_all, '--checkpoint_dir', str(save), '--preds_path',
        str(work / 'fingerprints.csv'), '--fingerprint_type', 'MPN', '--num_workers', '0',
        '--smiles_columns', 'rxn'])
    fp = chemprop.train.molecule_fingerprint.molecule_fingerprint(args=fargs)
    fp = np.asarray(fp)
    if fp.dtype == object:
        bad = int(np.sum(fp == 'Invalid SMILES'))
        if bad:
            raise SystemExit('%s chemprop_cgr fold %d: chemprop could not read %d values '
                             '(Invalid SMILES)' % (ds, k, bad))
        fp = fp.astype(np.float64)
    fp = fp[:, :, 0].astype(np.float32)
    info = {'chemprop_version': getattr(chemprop, '__version__', '1.6.1'),
            'settings': 'chemprop defaults; --reaction (reaction_mode reac_diff), seed %d' % SEED,
            'epochs': int(targs.epochs), 'n_train_rows': int(len(itr)), 'n_val_rows': int(len(iva)),
            'chemprop_heldout_scores': scores, 'train_seconds': t_train}
    return fp[:len(tr)], fp[len(tr):], info


def build_fold(ds, view, k):
    out = fold_view_path(ds, view, k)
    if out.exists():
        log('%s %s fold %d: already built' % (ds, view, k))
        return
    rx, y, g = load(ds)
    tr, te = folds(ds)[k]
    t0 = time.time()
    if view == 'chemprop_cgr':
        Xtr, Xte, info = _build_chemprop(ds, k, out.parent / ('chemprop_fold%d' % k))
    else:
        raise SystemExit('unknown fold-trained view %r' % view)
    X = np.zeros((len(y), Xtr.shape[1]), np.float32)
    X[tr], X[te] = Xtr, Xte
    why = refusal(X, len(y))
    if why:
        raise SystemExit('%s %s fold %d: refused, %s' % (ds, view, k, why))
    rec = dict(info, dataset=ds, view=view, fold=k, n_features=int(Xtr.shape[1]),
               n_train=int(len(tr)), n_test=int(len(te)), seconds=time.time() - t0,
               result_status='ok')
    _write_json_last(out.with_suffix('.json'), rec)
    tmp = out.with_name(out.stem + '.tmp.%d.npz' % os.getpid())
    np.savez(tmp, train_idx=tr, test_idx=te, X_train=Xtr, X_test=Xte)
    os.replace(tmp, out)          # the file the ledger checks: last
    log('%s %s fold %d: %d features, %.0f s' % (ds, view, k, Xtr.shape[1], rec['seconds']))


# ----------------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------------

def fixed_matrix(ds, view):
    rec = built_path(ds, view)
    if not rec.exists():
        raise SystemExit('%s %s: not built (%s missing)' % (ds, view, rec))
    return np.load(json.loads(rec.read_text())['path'])


def fold_matrices(ds, view, k):
    """(X_train, X_test) of one view on outer fold k."""
    tr, te = folds(ds)[k]
    if view in FOLD_VIEWS:
        p = fold_view_path(ds, view, k)
        if not p.exists():
            raise SystemExit('%s %s fold %d: not built (%s missing)' % (ds, view, k, p))
        z = np.load(p)
        if not (np.array_equal(z['train_idx'], tr) and np.array_equal(z['test_idx'], te)):
            raise SystemExit('%s %s fold %d: built on other rows than the outer fold' % (ds, view, k))
        return z['X_train'], z['X_test']
    X = _fixed_cached(ds, view)
    return X[tr], X[te]


@lru_cache(maxsize=None)
def _fixed_cached(ds, view):
    return fixed_matrix(ds, view)


def evaluate_one(ds, view, model, plus=False):
    out = result_path(ds, view, model, plus)
    if out.exists():
        return
    if plus and view == AUG_VIEW:
        raise SystemExit('%s is the added set; it is not added to itself' % AUG_VIEW)
    from experiments._shared import get_model_factories
    from kirby.scaling import FeatureScaler
    rx, y, g = load(ds)
    mname, mfn, lower = metric(ds)
    factory = get_model_factories([model], 'regression', n_samples=len(y))[model]
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.stem + '.partial.json')
    rows, oof = [], np.full(len(y), np.nan)
    if part.exists():
        pj = json.loads(part.read_text())
        rows, oof = pj['rows'], np.asarray([np.nan if v is None else v for v in pj['oof']], float)
        log('%s %s %s%s: resuming after %d fold(s)' % (ds, view, model, ' +bond' if plus else '',
                                                        len(rows)))
    t_all = time.time()
    n_feat = None
    for k, (tr, te) in enumerate(folds(ds)):
        if k < len(rows):
            continue
        Xtr, Xte = fold_matrices(ds, view, k)
        if plus:
            Btr, Bte = fold_matrices(ds, AUG_VIEW, k)
            Xtr, Xte = np.hstack([Xtr, Btr]), np.hstack([Xte, Bte])
        n_feat = int(Xtr.shape[1])
        t0 = time.time()
        sc = FeatureScaler()
        Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)
        m = factory()
        m.fit(Xtr_s, y[tr])
        pred = np.asarray(m.predict(Xte_s), float).ravel()
        oof[te] = pred
        rows.append({'fold': k, mname: float(mfn(y[te], pred)), 'n_train': int(len(tr)),
                     'n_test': int(len(te)), 'n_features': n_feat, 'seconds': time.time() - t0})
        log('%s %s %s%s fold %d: %s %.4f (%.0f s)' % (ds, view, model, ' +bond' if plus else '',
                                                      k, mname, rows[-1][mname], rows[-1]['seconds']))
        part.write_text(json.dumps({'rows': rows,
                                    'oof': [None if np.isnan(v) else float(v) for v in oof]}))
    vals = [r[mname] for r in rows]
    res = {'dataset': ds, 'view': view, 'model': model, 'plus_bond_changes': bool(plus),
           'metric': mname, 'lower_is_better': bool(lower), 'n_reactions': int(len(y)),
           'n_features_per_fold': [r['n_features'] for r in rows], 'folds': rows,
           'mean': float(np.mean(vals)), 'sd_over_folds': float(np.std(vals, ddof=1)),
           'seconds': time.time() - t_all, 'result_status': 'ok'}
    np.save(str(out.with_suffix('')) + '.oof.npy', oof)
    _write_json_last(out, res)
    if part.exists():
        part.unlink()


# ----------------------------------------------------------------------------
# summarise
# ----------------------------------------------------------------------------

def linear_cka(A, B):
    from opig_cgr_comparison import linear_cka as _cka
    return _cka(A, B)


def _std(X):
    X = np.asarray(X, float)
    return (X - X.mean(0)) / (X.std(0) + 1e-12)


def stage_summarise(tasks):
    from scipy.stats import spearmanr
    L = ['# Reaction view screen', '',
         'Each view scored alone, and with the 30 cgr_bond_changes counts appended, under each '
         'dataset\'s decided models on KIRBy\'s 5 outer folds (group-aware, seed 42). '
         'Scores are the dataset\'s metric on each fold\'s held-out reactions. '
         'SD is the standard deviation over the 5 folds. Script: '
         '`tests/experiments/benchmarks/b10_reaction_views.py`.', '']
    for ds in tasks:
        res = {}
        d = RESULTS_DIR / ds
        for p in sorted(d.glob('*__*.json')) if d.is_dir() else []:
            if p.name.endswith('.partial.json'):
                continue
            r = json.loads(p.read_text())
            res[(r['view'], r['model'], r['plus_bond_changes'])] = r
        if not res:
            continue
        rx, y, g = load(ds)
        views = [v for v in views_for(ds) if any(k[0] == v for k in res)]
        models = [m for m in models_for(ds) if any(k[1] == m for k in res)]
        mname = next(iter(res.values()))['metric']
        lower = next(iter(res.values()))['lower_is_better']
        L += ['## %s' % ds, '',
              '%d reactions, %d groups, metric %s (%s is better).'
              % (len(y), len(np.unique(g)), mname, 'lower' if lower else 'higher'), '']
        for m in models:
            L += ['### %s, %s: each view alone' % (ds, m), '',
                  'One row is one view. Fold columns are %s on that fold\'s held-out reactions.' % mname,
                  '', '| view | features | ' + ' | '.join('fold %d' % k for k in range(N_FOLDS))
                  + ' | mean | SD |', '|---|---|' + '---|' * (N_FOLDS + 2)]
            for v in views:
                r = res.get((v, m, False))
                if r is None:
                    continue
                nf = r['n_features_per_fold']
                nf = str(nf[0]) if len(set(nf)) == 1 else '%d to %d' % (min(nf), max(nf))
                L.append('| %s | %s | %s | %.4f | %.4f |' % (
                    v, nf, ' | '.join('%.4f' % f[mname] for f in r['folds']), r['mean'],
                    r['sd_over_folds']))
            L += ['', '### %s, %s: with cgr_bond_changes appended' % (ds, m), '',
                  'One row is one view. Each fold cell is "alone / with the 30 counts appended", '
                  'both %s on the same held-out reactions.' % mname, '',
                  '| view | ' + ' | '.join('fold %d' % k for k in range(N_FOLDS))
                  + ' | mean alone | mean with |', '|---|' + '---|' * (N_FOLDS + 2)]
            for v in views:
                a, b = res.get((v, m, False)), res.get((v, m, True))
                if a is None or b is None:
                    continue
                L.append('| %s | %s | %.4f | %.4f |' % (v, ' | '.join(
                    '%.4f / %.4f' % (fa[mname], fb[mname]) for fa, fb in zip(a['folds'], b['folds'])),
                    a['mean'], b['mean']))
            L.append('')
            errs = {}
            for v in views:
                f = Path(str(result_path(ds, v, m).with_suffix('')) + '.oof.npy')
                if f.exists():
                    errs[v] = np.abs(np.load(f) - y)
            if len(errs) >= 2:
                vs = list(errs)
                L += ['### %s, %s: error overlap' % (ds, m), '',
                      'Spearman correlation between two views\' absolute errors over every reaction '
                      '(out-of-fold, views alone). 1 = the same reactions are hard for both.', '',
                      '| | ' + ' | '.join(vs) + ' |', '|---|' + '---|' * len(vs)]
                for a in vs:
                    L.append('| %s | %s |' % (a, ' | '.join(
                        '%.2f' % (1.0 if a == b else spearmanr(errs[a], errs[b])[0]) for b in vs)))
                L.append('')
        # Linear CKA on each fold's held-out reactions (fold-trained views
        # exist only per fold), then the mean over the folds.
        cka = {}
        ok_views = []
        for v in views:
            try:
                fold_matrices(ds, v, 0)
                ok_views.append(v)
            except SystemExit:
                pass
        for k in range(N_FOLDS):
            mats = {}
            for v in ok_views:
                try:
                    mats[v] = _std(fold_matrices(ds, v, k)[1])
                except SystemExit:
                    pass
            for a in mats:
                for b in mats:
                    cka.setdefault((a, b), []).append(linear_cka(mats[a], mats[b]))
        if ok_views:
            L += ['### %s: linear CKA' % ds, '',
                  'Linear centred kernel alignment between standardised view matrices on each '
                  'fold\'s held-out reactions, mean over the folds a pair has. 1 = the same '
                  'information up to a linear map, 0 = unrelated.', '',
                  '| | ' + ' | '.join(ok_views) + ' |', '|---|' + '---|' * len(ok_views)]
            for a in ok_views:
                L.append('| %s | %s |' % (a, ' | '.join(
                    '%.2f' % np.mean(cka[(a, b)]) if (a, b) in cka else '' for b in ok_views)))
            L.append('')
        # chemprop's own held-out score, for reference.
        cp = []
        for k in range(N_FOLDS):
            j = fold_view_path(ds, 'chemprop_cgr', k).with_suffix('.json')
            if j.exists():
                r = json.loads(j.read_text())
                cp.append((k, r.get('chemprop_heldout_scores', {}), r.get('train_seconds')))
        if cp:
            L += ['### %s: chemprop\'s own held-out score' % ds, '',
                  'The chemprop model that chemprop_cgr\'s features come from, scored by chemprop on '
                  'the same held-out reactions (its test_scores.csv). Reference only.', '',
                  '| fold | scores | training seconds |', '|---|---|---|']
            for k, s, t in cp:
                L.append('| %d | %s | %s |' % (k, ', '.join('%s %.4f' % kv for kv in s.items()),
                                              '%.0f' % t if t else ''))
            L.append('')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_DIR / ('summary.md.tmp.%d' % os.getpid())
    tmp.write_text('\n'.join(L) + '\n')
    os.replace(tmp, RESULTS_DIR / 'summary.md')
    log('summary written: %s' % (RESULTS_DIR / 'summary.md'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=['build', 'evaluate', 'summarise'], required=True)
    ap.add_argument('--tasks', default=','.join(TASKS))
    ap.add_argument('--views', default=None)
    ap.add_argument('--folds', default=None, help='fold-trained views: comma list of outer folds')
    ap.add_argument('--models', default=None, help='default: the dataset\'s decided models')
    ap.add_argument('--plus-bond-changes', action='store_true')
    a = ap.parse_args()
    tasks = a.tasks.split(',')
    if a.stage == 'summarise':
        stage_summarise(tasks)
        return
    for ds in tasks:
        views = a.views.split(',') if a.views else views_for(ds)
        if a.stage == 'build':
            for v in views:
                if v in FOLD_VIEWS:
                    ks = [int(x) for x in a.folds.split(',')] if a.folds else range(N_FOLDS)
                    for k in ks:
                        build_fold(ds, v, k)
                else:
                    build_fixed(ds, v)
        else:
            models = a.models.split(',') if a.models else models_for(ds)
            for v in views:
                for m in models:
                    evaluate_one(ds, v, m, plus=a.plus_bond_changes)


if __name__ == '__main__':
    main()
