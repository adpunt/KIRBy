"""Test the 6 split fixes with one cheap rep (aac).

For each dataset: cluster sequences by identity (CDRH3 for tz_her2, full Fv for the
rest), check the split is non-degenerate (>= 5 real clusters, no hash-fallback), then
score amino-acid-composition (aac, 20-dim, computed inline) under group-CV. Answers:
(1) does a valid grouped split exist? (2) what's the honest number under it?
"""
import os, sys
import numpy as np
os.environ.setdefault('KIRBY_CACHE_DIR', '/Volumes/seagate/kirby_cache')
sys.path.insert(0, 'src'); sys.path.insert(0, 'tests'); sys.path.insert(0, 'tests/experiments')
from experiments._shared import sequence_identity_clusters
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from lightgbm import LGBMClassifier, LGBMRegressor

AA = 'ACDEFGHIKLMNPQRSTVWY'


def aac(seqs):
    X = np.zeros((len(seqs), 20))
    for i, s in enumerate(seqs):
        s = str(s).upper(); L = max(len(s), 1)
        for j, a in enumerate(AA):
            X[i, j] = s.count(a) / L
    return X


def cv(X, y, groups, is_clf):
    k = min(5, len(np.unique(groups)))
    if k < 2:
        return None
    oof = np.full(len(y), np.nan)
    sp = (StratifiedGroupKFold(k, shuffle=True, random_state=42).split(X, y, groups)
          if is_clf else GroupKFold(k).split(X, y, groups))
    for tr, va in sp:
        s = StandardScaler(); Xtr = s.fit_transform(X[tr]); Xva = s.transform(X[va])
        m = (LGBMClassifier if is_clf else LGBMRegressor)(
            n_estimators=150, num_leaves=31, learning_rate=0.05,
            random_state=42, n_jobs=-1, verbosity=-1)
        m.fit(Xtr, y[tr])
        oof[va] = m.predict_proba(Xva)[:, 1] if is_clf else m.predict(Xva)
    mask = ~np.isnan(oof)
    if is_clf:
        try:
            return ('auroc', float(roc_auc_score(y[mask], oof[mask])))
        except Exception:
            return ('auroc', float('nan'))
    return ('spearman', float(spearmanr(y[mask], oof[mask]).correlation))


def load(ds):
    """-> (cluster_seqs, feat_seqs, labels)"""
    if ds.startswith('tz_her2'):
        from kirby.datasets.trastuzumab import load_trastuzumab_dataset, VH_PRE
        d = load_trastuzumab_dataset(preset='binary' if ds.endswith('binary') else 'multiclass')
        heavy = d['heavy_seqs']; off = len(VH_PRE)
        cdrh3 = [h[off:off + 10] for h in heavy]   # the varying 10-mer
        return cdrh3, heavy, np.asarray(d['labels'])
    if ds == 'cdr_flexibility':
        from kirby.datasets.itsflexible import load_itsflexible
        d = load_itsflexible(mode='cdr')
        heavy = d['heavy_seqs']; light = d.get('light_seqs') or [''] * len(heavy)
        conc = [str(h) + str(l) for h, l in zip(heavy, light)]
        return conc, conc, np.asarray(d['labels'])
    # flab
    from antibody_baseline_survey import load_dataset
    d = load_dataset(ds)
    heavy = d['heavy_seqs']; light = d.get('light_seqs') or [''] * len(heavy)
    conc = [str(h) + str(l) for h, l in zip(heavy, light)]
    return conc, conc, np.asarray(d['labels'])


def run(ds):
    print(f"\n{'='*60}\n{ds}\n{'='*60}", flush=True)
    try:
        cl, feat, y = load(ds)
    except Exception as e:
        print(f"  LOAD FAILED: {e}", flush=True); return
    # subsample for tractable clustering (O(n^2) within length bands)
    if len(y) > 2500:
        rng = np.random.RandomState(42); idx = np.sort(rng.choice(len(y), 2500, replace=False))
        cl = [cl[i] for i in idx]; feat = [feat[i] for i in idx]; y = y[idx]
    uniq = np.unique(y)
    is_clf = (len(uniq) <= 5 and np.allclose(y, y.astype(int)))
    print(f"  n={len(y)}  task={'clf' if is_clf else 'reg'}  labels~{uniq[:5]}", flush=True)
    X = aac(feat)
    # Sweep from lenient (leaky ceiling) to strict (larger clusters = honest holdout).
    # As threshold drops, clusters merge; the honest number is where clusters are
    # substantial (not near-singletons) but still >= folds.
    for thr in (0.9, 0.7, 0.5, 0.3):
        groups, nclust = sequence_identity_clusters(cl, identity_threshold=thr, min_groups=5)
        sizes = np.unique(groups, return_counts=True)[1]
        line = (f"  identity>{thr}: {nclust:4d} clusters, largest={sizes.max():4d}, "
                f"median={int(np.median(sizes))}, singletons={(sizes==1).sum():4d}")
        if nclust >= 5:
            res = cv(X, y.astype(float), groups, is_clf)
            line += f"  -> aac {res[0]}={res[1]:+.3f}" if res else "  (CV skipped)"
        print(line, flush=True)


if __name__ == '__main__':
    for ds in (sys.argv[1:] or ['tz_her2_binary', 'cdr_flexibility',
                                 'flab_binding_phillips_h3', 'flab_binding_adams',
                                 'flab_expression_adams']):
        try:
            run(ds)
        except Exception:
            import traceback; traceback.print_exc()
