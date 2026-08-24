"""Rep-pruning EVIDENCE — greedy correlation elimination, swept over thresholds.

Produces, per dataset, the numbers the USER reads to pick the final <=10 reps. It
does NOT pick the pool: it shows, for a sweep of correlation thresholds, what a
greedy "drop reps too correlated with a better-scoring rep" filter would keep, and
why each drop happened. Different thresholds -> different pools; the user compares
and decides per dataset.

Method (per dataset):
  A. per-rep out-of-fold (OOF) prediction vectors under the dataset's group CV
     (lightgbm) + solo score.                                    [n CVs]
  B. full rep x rep prediction-correlation matrix (Spearman of OOF vectors). [free]
  C. GREEDY ELIMINATION, best-solo-first, for each threshold T: keep a rep only if
     its correlation to every already-kept rep is < T; otherwise drop it and record
     the kept rep it was redundant with. Dropped reps are never used as anchors
     again (user: "stop using it once filtered").                [free, uses B]
  D. one bounded concat check per rep: concat(rep, nearest higher-solo partner) -
     solo(partner) -> was dropping that rep actually free?        [n-1 CVs]

Compute is ~2n CVs per dataset (NOT the O(n^2) all-pairs concat). lightgbm only.
Cache path from KIRBY_CACHE_DIR; no hardcoded local path.

Tunables via env (light defaults are for a LOCAL smoke test; the server run should
override to the real values):
  KIRBY_PRUNE_FOLDS=3            group-CV folds
  KIRBY_PRUNE_CAP=512           per-rep top-variance feature cap (0 = no cap)
  KIRBY_PRUNE_THRESHOLDS=0.75,0.80,0.85,0.90,0.95   correlation sweep

Writes $KIRBY_CACHE_DIR/rep_prune/<ds>/:
  preds.npz     : {rep__<name>: oof_vector} + 'labels'
  scores.json   : {rep: {solo, rmse, n, dim}} best-first
  pred_corr.csv : rep x rep prediction-correlation matrix
  greedy.json   : {thresholds: {T: {kept, n_kept, dropped:[{rep,redundant_with,
                  corr,solo,anchor_solo}]}}, nearest:[{rep,partner,corr,concat,gain}]}

Run one dataset locally (import/paths verified; the real sweep is a server job):
  python tests/rep_prune_evidence.py herg_ki
"""
import os, sys, json, time, csv, itertools
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, 'experiments'))

from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMRegressor, LGBMClassifier

from experiments._shared import load_data

CACHE = os.environ.get('KIRBY_CACHE_DIR', os.path.expanduser('~/.cache/kirby'))
OUT_ROOT = os.path.join(CACHE, 'rep_prune')

N_FOLDS = int(os.environ.get('KIRBY_PRUNE_FOLDS', '3'))
CAP = int(os.environ.get('KIRBY_PRUNE_CAP', '512'))          # 0 disables the cap
THRESHOLDS = [float(t) for t in
              os.environ.get('KIRBY_PRUNE_THRESHOLDS', '0.75,0.80,0.85,0.90,0.95').split(',')]
# The concat check (does adding a rep to a stronger one gain signal?) is part of
# the COMBINATION work, not rep-pool cleanup — off by default. Set =1 to include it.
RUN_CONCAT = os.environ.get('KIRBY_PRUNE_CONCAT', '0') == '1'
# Score the nine similarity recipes alongside the dataset's normal reps. Off by
# default so an ordinary rep-prune run is byte-for-byte what it was before.
RUN_SIM = os.environ.get('KIRBY_PRUNE_SIM', '0') == '1'
N_EST = 150


def _model(is_clf):
    return (LGBMClassifier if is_clf else LGBMRegressor)(
        n_estimators=N_EST, num_leaves=31, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=-1)


def cap_cols(X):
    if not CAP or X.shape[1] <= CAP:
        return X
    idx = np.argsort(X.var(0))[::-1][:CAP]
    return X[:, np.sort(idx)]


def oof_folds(get_fold, n, y, groups, is_clf, classes=None):
    """OOF predictions under the group CV -> (corr_vector, score).

    `get_fold(train_idx, val_idx) -> (X_train, X_val)` so this works both for a
    fixed matrix and for a per-fold rep whose columns are chosen inside the fold.

      regression : prediction vector       + Spearman(y, pred)
      binary clf : class-1 proba vector     + AUROC
      multiclass : flattened per-class proba (n*C,) + weighted one-vs-rest AUROC.
    The multiclass correlation vector is the full proba matrix ravelled so two reps
    that predict the same class distributions read as highly correlated (uniform with
    the binary/reg path downstream — greedy just spearman-correlates these vectors)."""
    k = min(N_FOLDS, len(np.unique(groups)))
    if k < 2:
        return None, None
    multiclass = is_clf and classes is not None and len(classes) > 2
    # The splitters read only the row count off this argument, never its values.
    rows = np.zeros((n, 1))
    splitter = (StratifiedGroupKFold(k, shuffle=True, random_state=42).split(rows, y, groups)
                if is_clf else GroupKFold(k).split(rows, y, groups))
    if multiclass:
        cls_idx = {c: j for j, c in enumerate(classes)}
        proba = np.full((n, len(classes)), np.nan)
        for tr, va in splitter:
            Xtr_raw, Xva_raw = get_fold(tr, va)
            s = StandardScaler(); Xtr = s.fit_transform(Xtr_raw); Xva = s.transform(Xva_raw)
            m = _model(is_clf); m.fit(Xtr, y[tr])
            p = m.predict_proba(Xva)
            for jj, c in enumerate(m.classes_):      # a fold's train may miss a class
                proba[va, cls_idx[c]] = p[:, jj]
        mask = ~np.isnan(proba).any(1)
        pm = np.nan_to_num(proba[mask])
        sc = float(roc_auc_score(y[mask], pm, multi_class='ovr',
                                 average='weighted', labels=list(classes)))
        return pm.ravel(), sc
    v = np.full(n, np.nan)
    for tr, va in splitter:
        Xtr_raw, Xva_raw = get_fold(tr, va)
        s = StandardScaler(); Xtr = s.fit_transform(Xtr_raw); Xva = s.transform(Xva_raw)
        m = _model(is_clf); m.fit(Xtr, y[tr])
        if is_clf:
            p = m.predict_proba(Xva)
            v[va] = p[:, 1] if p.shape[1] > 1 else p[:, 0]
        else:
            v[va] = m.predict(Xva)
    mask = ~np.isnan(v)
    sc = (float(roc_auc_score(y[mask], v[mask])) if is_clf
          else float(spearmanr(y[mask], v[mask]).correlation))
    return v, sc


def oof(X, y, groups, is_clf, classes=None):
    """OOF for a rep that is one fixed matrix (columns already chosen)."""
    return oof_folds(lambda tr, va: (X[tr], X[va]), len(y), y, groups,
                     is_clf, classes)


def factory_folds(factory, y):
    """Wrap a per-fold rep factory into the `get_fold` contract used above.

    The factory picks its own columns from the training rows only; whatever it
    returns is then capped to the same top-variance width as the fixed-matrix
    reps, using training-fold variance so the val fold never influences the cap.
    """
    def get_fold(tr, va):
        Xtr, Xva = factory(tr, va, y[tr])
        Xtr = np.nan_to_num(np.asarray(Xtr, np.float64), nan=0.0)
        Xva = np.nan_to_num(np.asarray(Xva, np.float64), nan=0.0)
        if CAP and Xtr.shape[1] > CAP:
            idx = np.sort(np.argsort(Xtr.var(0))[::-1][:CAP])
            Xtr, Xva = Xtr[:, idx], Xva[:, idx]
        return Xtr, Xva
    return get_fold


def sim_recipe_factories(ds, smiles):
    """The nine similarity recipes (+2 single-block controls) as per-fold reps.

    Returns {} when the dataset has no SMILES or the machinery can't be built, so
    a failure here can never silently change a normal rep-prune run.
    """
    if not smiles:
        return {}
    from kirby.similarity.recipes import make_recipe_factory, RECIPE_ARCHITECTURES
    from kirby.similarity.selection import (
        make_saboteur_drop_factory, make_fine2d_factory)
    out = {}
    for name, (blocks, weights) in RECIPE_ARCHITECTURES.items():
        out[name] = make_recipe_factory(smiles, ds, blocks, weights=weights)
    out['sim_saboteur_drop'] = make_saboteur_drop_factory(smiles, ds)
    out['sim_fine2d'] = make_fine2d_factory(smiles, ds)
    return out


def greedy_eliminate(names, solo, corr, T):
    """Walk reps best-solo-first; keep a rep only if its correlation to every
    already-kept rep is < T, else drop it (recording the anchor it duplicates).
    Dropped reps never anchor a later comparison."""
    kept, dropped = [], []
    for r in names:                     # names already sorted best-solo-first
        anchor, cmax = None, -2.0
        for k in kept:
            c = corr[r][k]
            if c > cmax:
                cmax, anchor = c, k
        if anchor is not None and cmax >= T:
            dropped.append({'rep': r, 'redundant_with': anchor,
                            'corr': round(cmax, 3), 'solo': round(solo[r], 3),
                            'anchor_solo': round(solo[anchor], 3)})
        else:
            kept.append(r)
    return kept, dropped


def run(ds):
    t0 = time.time()
    print(f"\n{'='*70}\n{ds}\n{'='*70}", flush=True)
    # KIRBY_PRUNE_REPS overrides the dataset's normal rep list. Used to skip reps
    # whose builder is broken in a given environment (e.g. no working torch on a
    # laptop) so the rest of the table can still be produced.
    _override = os.environ.get('KIRBY_PRUNE_REPS', '').strip()
    rep_names = [r.strip() for r in _override.split(',') if r.strip()] or None
    if rep_names:
        print(f"  [override] rep list = {rep_names}", flush=True)
    data = load_data(ds, rep_names=rep_names)
    reps = data['reps']; y = np.asarray(data['labels'], float)
    groups = np.asarray(data['groups'])
    is_clf = data.get('task_type') == 'classification'
    classes = np.unique(y) if is_clf else None
    multiclass = is_clf and len(classes) > 2
    primary = ('auroc_ovr' if multiclass else 'auroc') if is_clf else 'spearman'
    out_dir = os.path.join(OUT_ROOT, ds); os.makedirs(out_dir, exist_ok=True)

    # ---- A: OOF vectors + solo scores --------------------------------------
    Xc, vec, solo, meta = {}, {}, {}, {}
    for rep in sorted(reps):
        X = cap_cols(np.nan_to_num(np.asarray(reps[rep], np.float64), nan=0.0))
        try:
            v, sc = oof(X, y, groups, is_clf, classes)
        except Exception as e:
            print(f"  {rep:24s} FAILED: {e}", flush=True); continue
        if v is None:
            print(f"  {rep:24s} skipped (too few groups)", flush=True); continue
        Xc[rep] = X; vec[rep] = v; solo[rep] = sc
        if multiclass:                              # v is (n*C,) proba, not per-sample
            meta[rep] = {'solo': round(sc, 4), 'rmse': None,
                         'n': int(len(v) // len(classes)), 'dim': int(X.shape[1])}
        else:
            m = ~np.isnan(v)
            meta[rep] = {'solo': round(sc, 4),
                         'rmse': float(np.sqrt(np.mean((y[m] - v[m]) ** 2))),
                         'n': int(m.sum()), 'dim': int(X.shape[1])}
        print(f"  {rep:24s} dim={X.shape[1]:5d}  {primary}={sc:+.4f}", flush=True)

    # ---- A2: per-fold similarity recipes, scored identically (opt-in) --------
    if RUN_SIM:
        try:
            sim_facs = sim_recipe_factories(ds, data.get('smiles'))
        except Exception as e:
            print(f"  [sim] could not build recipe factories: {e}", flush=True)
            sim_facs = {}
        for rep, fac in sim_facs.items():
            try:
                v, sc = oof_folds(factory_folds(fac, y), len(y), y, groups,
                                  is_clf, classes)
            except Exception as e:
                print(f"  {rep:24s} FAILED: {e}", flush=True); continue
            if v is None:
                print(f"  {rep:24s} skipped (too few groups)", flush=True); continue
            vec[rep] = v; solo[rep] = sc
            if multiclass:
                meta[rep] = {'solo': round(sc, 4), 'rmse': None,
                             'n': int(len(v) // len(classes)), 'dim': None}
            else:
                m = ~np.isnan(v)
                meta[rep] = {'solo': round(sc, 4),
                             'rmse': float(np.sqrt(np.mean((y[m] - v[m]) ** 2))),
                             'n': int(m.sum()), 'dim': None}
            print(f"  {rep:24s} dim= per-fold  {primary}={sc:+.4f}", flush=True)

    names = sorted(solo, key=lambda r: -solo[r])   # best-solo-first
    if len(names) < 2:
        print(f"  only {len(names)} scorable rep(s); nothing to prune", flush=True)
        return
    np.savez_compressed(os.path.join(out_dir, 'preds.npz'),
                        labels=y, **{f'rep__{k}': vec[k] for k in names})
    json.dump({'dataset': ds, 'is_clf': is_clf, 'primary': primary,
               'reps': {r: meta[r] for r in names}},
              open(os.path.join(out_dir, 'scores.json'), 'w'), indent=2)

    # ---- B: prediction-correlation matrix (free) ---------------------------
    corr = {a: {} for a in names}
    for a, b in itertools.combinations(names, 2):
        c = float(spearmanr(vec[a], vec[b]).correlation)
        corr[a][b] = corr[b][a] = c
    for a in names:
        corr[a][a] = 1.0
    with open(os.path.join(out_dir, 'pred_corr.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow([''] + names)
        for a in names:
            w.writerow([a] + [f"{corr[a][b]:.3f}" for b in names])

    # ---- C: greedy elimination over the threshold sweep --------------------
    thresholds = {}
    for T in THRESHOLDS:
        kept, dropped = greedy_eliminate(names, solo, corr, T)
        thresholds[f"{T:.2f}"] = {'kept': kept, 'n_kept': len(kept), 'dropped': dropped}
        print(f"  T={T:.2f}: keep {len(kept):2d} / {len(names)}  "
              f"(drop {len(dropped)})", flush=True)

    # ---- D: OPTIONAL concat check (combination work, off by default) --------
    nearest = []
    if RUN_CONCAT:
        for r in names[1:]:             # top-solo rep has no higher partner
            better = [p for p in names if solo[p] >= solo[r] and p != r]
            if not better:
                continue
            p = max(better, key=lambda p: corr[r][p])  # most-correlated anchor
            if r not in Xc or p not in Xc:
                continue                # per-fold reps have no single matrix
            try:
                _, concat = oof(np.hstack([Xc[r], Xc[p]]), y, groups, is_clf)
            except Exception as e:
                print(f"    concat {r}+{p} FAILED: {e}", flush=True); continue
            nearest.append({'rep': r, 'partner': p, 'corr': round(corr[r][p], 3),
                            'concat': round(concat, 3),
                            'gain': round(concat - solo[p], 4)})

    json.dump({'dataset': ds, 'primary': primary, 'n_reps': len(names),
               'solo': {r: round(solo[r], 4) for r in names},
               'thresholds': thresholds, 'nearest': nearest},
              open(os.path.join(out_dir, 'greedy.json'), 'w'), indent=2)
    print(f"  -> saved rep_prune/{ds}/  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
    print(f"[rep_prune] EVIDENCE ONLY — selects nothing. folds={N_FOLDS} cap={CAP} "
          f"thresholds={THRESHOLDS} concat={'on' if RUN_CONCAT else 'off'} "
          f"sim_recipes={'on' if RUN_SIM else 'off'}", flush=True)
    for ds in sys.argv[1:]:
        try:
            run(ds)
        except Exception:
            import traceback; tb = traceback.format_exc()
            print(tb, flush=True)
            print(f"DATASET {ds} FAILED", flush=True)
            # Leave a marker in the cache so a load-time failure (which happens
            # before run() makes the dir) is diagnosable without the slurm log.
            try:
                edir = os.path.join(OUT_ROOT, ds); os.makedirs(edir, exist_ok=True)
                with open(os.path.join(edir, 'error.txt'), 'w') as f:
                    f.write(tb)
            except Exception:
                pass
