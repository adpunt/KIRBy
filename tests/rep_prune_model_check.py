"""Would the rep pool have been the same reps under rf or svm instead of lightgbm?

Re-runs the WHOLE pruning procedure — solo score, out-of-fold prediction
correlation, greedy elimination best-solo-first at each correlation threshold —
once per model, on a dataset's ORIGINAL (pre-prune) rep list. Everything else is
identical to `rep_prune_evidence.py`: same group CV, same fold seed, same
top-variance feature cap, same StandardScaler, same `greedy_eliminate`. Only the
model changes.

Greedy is driven by the same solo metric the pruning used (Spearman for
regression, AUROC for classification), never by `--metric`, which only sets the
score column in the printed table.

Reports, per model: what greedy keeps at each threshold, and how that set
compares to the signed-off pool in `DATASET_REPS[ds]['general']`.

The original rep list is read from `$KIRBY_CACHE_DIR/rep_prune/<ds>/scores.json`
(the reps lightgbm actually scored before the pool was cut to 10), so the three
model rankings cover exactly the same reps. A rep whose `.npy` is not in the
local rep_cache is reported as MISSING, never silently dropped.

Model specs are the pipeline's own (`DEFAULT_MODELS` in find_best_hybrid.py):
  rf   RandomForestRegressor/Classifier(n_estimators=100, max_features='sqrt')
  svm  SVR() / SVC(probability=True)
lightgbm is re-run here (not read from the old files) so all three share folds.

Scores per rep:
  regression      spearman + rmse
  classification  f1_weighted + auroc (ovr-weighted for multiclass)

Writes, under $KIRBY_CACHE_DIR/rep_prune_models/<ds>/:
  scores.json   per model, per rep: both metrics + seconds
  greedy.json   per model: solo, threshold sweep, kept/dropped, pool comparison
  preds_<model>.npz   the out-of-fold vectors the correlations came from

  python tests/rep_prune_model_check.py caco2 --metric rmse
  python tests/rep_prune_model_check.py tcga_brca_pam50 --metric f1_weighted
"""
import os, sys, json, time, argparse
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, 'experiments'))

from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

from experiments._shared import load_data

CACHE = os.environ.get('KIRBY_CACHE_DIR', os.path.expanduser('~/.cache/kirby'))
PRUNE_ROOT = os.path.join(CACHE, 'rep_prune')
OUT_ROOT = os.path.join(CACHE, 'rep_prune_models')

N_FOLDS = int(os.environ.get('KIRBY_PRUNE_FOLDS', '3'))
CAP = int(os.environ.get('KIRBY_PRUNE_CAP', '512'))
MODELS = [m.strip() for m in
          os.environ.get('KIRBY_CHECK_MODELS', 'lightgbm,rf,svm').split(',') if m.strip()]

# Metric the dataset is ranked by. Lower-is-better metrics listed here.
LOWER_IS_BETTER = {'rmse'}


def make_model(name, is_clf):
    """The pipeline's own model specs; lightgbm matches rep_prune_evidence.py."""
    if name == 'lightgbm':
        from lightgbm import LGBMRegressor, LGBMClassifier
        return (LGBMClassifier if is_clf else LGBMRegressor)(
            n_estimators=150, num_leaves=31, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            n_jobs=int(os.environ.get('KIRBY_LGBM_THREADS', '-1')), verbosity=-1)
    if name == 'rf':
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        return (RandomForestClassifier if is_clf else RandomForestRegressor)(
            n_estimators=100, max_features='sqrt', random_state=42, n_jobs=-1)
    if name == 'svm':
        from sklearn.svm import SVR, SVC
        return SVC(probability=True, random_state=42) if is_clf else SVR()
    raise ValueError(f'unknown model {name}')


def cap_cols(X):
    if not CAP or X.shape[1] <= CAP:
        return X
    idx = np.argsort(X.var(0))[::-1][:CAP]
    return X[:, np.sort(idx)]


def score_rep(X, y, groups, is_clf, classes, model_name):
    """OOF under the dataset's group CV -> (oof_vector, {metric: value}).

    The vector is what the prediction-correlation matrix is built from, exactly as
    in rep_prune_evidence.py: predictions for regression, the flattened per-class
    probability matrix for multiclass, the class-1 probability for binary.
    Same folds for all models."""
    k = min(N_FOLDS, len(np.unique(groups)))
    if k < 2:
        return None
    rows = np.zeros((len(y), 1))
    splits = list(StratifiedGroupKFold(k, shuffle=True, random_state=42).split(rows, y, groups)
                  if is_clf else GroupKFold(k).split(rows, y, groups))
    if is_clf:
        cls_idx = {c: j for j, c in enumerate(classes)}
        proba = np.full((len(y), len(classes)), np.nan)
        for tr, va in splits:
            s = StandardScaler(); Xtr = s.fit_transform(X[tr]); Xva = s.transform(X[va])
            m = make_model(model_name, True); m.fit(Xtr, y[tr])
            p = m.predict_proba(Xva)
            for jj, c in enumerate(m.classes_):     # a fold's train may miss a class
                proba[va, cls_idx[c]] = p[:, jj]
        mask = ~np.isnan(proba).any(1)
        pm = np.nan_to_num(proba[mask])
        pred = np.asarray(classes)[pm.argmax(1)]
        multiclass = len(classes) > 2
        auroc = float(roc_auc_score(
            y[mask], pm if multiclass else pm[:, 1],
            **({'multi_class': 'ovr', 'average': 'weighted',
                'labels': list(classes)} if multiclass else {})))
        vec = pm.ravel() if multiclass else pm[:, 1]
        return vec, {'f1_weighted': float(f1_score(y[mask], pred, average='weighted')),
                     'auroc': auroc, 'n': int(mask.sum())}
    v = np.full(len(y), np.nan)
    for tr, va in splits:
        s = StandardScaler(); Xtr = s.fit_transform(X[tr]); Xva = s.transform(X[va])
        m = make_model(model_name, False); m.fit(Xtr, y[tr])
        v[va] = m.predict(Xva)
    mask = ~np.isnan(v)
    return v, {'spearman': float(spearmanr(y[mask], v[mask]).correlation),
               'rmse': float(np.sqrt(np.mean((y[mask] - v[mask]) ** 2))),
               'n': int(mask.sum())}


def original_reps(ds):
    """The pre-prune rep list — exactly what lightgbm scored in rep_prune."""
    p = os.path.join(PRUNE_ROOT, ds, 'scores.json')
    with open(p) as f:
        return list(json.load(f)['reps'])


def run(ds, metric):
    t0 = time.time()
    wanted = original_reps(ds)
    print(f"\n{'='*72}\n{ds}   original rep list: {len(wanted)} reps\n{'='*72}", flush=True)
    data = load_data(ds, rep_names=wanted)
    reps = data['reps']
    y = np.asarray(data['labels'], float)
    groups = np.asarray(data['groups'])
    is_clf = data.get('task_type') == 'classification'
    classes = np.unique(y) if is_clf else None
    missing = [r for r in wanted if r not in reps]
    if missing:
        print(f"  MISSING from local rep_cache ({len(missing)}): {', '.join(missing)}",
              flush=True)

    out = {m: {} for m in MODELS}
    vecs = {m: {} for m in MODELS}
    for rep in wanted:
        if rep not in reps:
            continue
        X = cap_cols(np.nan_to_num(np.asarray(reps[rep], np.float64), nan=0.0))
        line = f"  {rep:24s} dim={X.shape[1]:5d}"
        for mn in MODELS:
            t = time.time()
            try:
                v, sc = score_rep(X, y, groups, is_clf, classes, mn)
            except Exception as e:
                print(f"  {rep:24s} {mn} FAILED: {e}", flush=True); continue
            if sc is None:
                print(f"  {rep:24s} skipped (too few groups)", flush=True); break
            sc['seconds'] = round(time.time() - t, 1)
            out[mn][rep] = sc; vecs[mn][rep] = v
            line += f"   {mn}={sc[metric]:+.4f} ({sc['seconds']:.0f}s)"
        print(line, flush=True)

    out_dir = os.path.join(OUT_ROOT, ds); os.makedirs(out_dir, exist_ok=True)
    json.dump({'dataset': ds, 'metric': metric, 'is_clf': is_clf,
               'n_folds': N_FOLDS, 'cap': CAP, 'requested_reps': wanted,
               'missing_reps': missing, 'models': out},
              open(os.path.join(out_dir, 'scores.json'), 'w'), indent=2)
    report(ds, metric, out)
    prune(ds, out, vecs, is_clf, classes, out_dir)
    print(f"  -> saved rep_prune_models/{ds}/  ({time.time()-t0:.0f}s)", flush=True)


def prune(ds, out, vecs, is_clf, classes, out_dir):
    """Run the pruning procedure itself, once per model, and compare the pools.

    Same solo metric as the pruning (Spearman / AUROC), same Spearman correlation
    of OOF vectors, same `greedy_eliminate` from rep_prune_evidence.py, same
    threshold sweep. The comparison target is the signed-off pool in
    `DATASET_REPS[ds]['general']`.
    """
    from rep_prune_evidence import greedy_eliminate
    solo_key = ('auroc' if is_clf else 'spearman')
    try:
        from experiments.complexity.complexity_helpers import DATASET_REPS
        pool = list(DATASET_REPS.get(ds, {}).get('general') or [])
    except ImportError:
        pool = []

    saved = {}
    for mn in MODELS:
        if not out.get(mn):
            continue
        solo = {r: out[mn][r][solo_key] for r in out[mn]}
        names = sorted(solo, key=lambda r: -solo[r])          # best-solo-first
        np.savez_compressed(os.path.join(out_dir, f'preds_{mn}.npz'),
                            **{f'rep__{r}': vecs[mn][r] for r in names})
        corr = {a: {b: 1.0 if a == b else float(
            spearmanr(vecs[mn][a], vecs[mn][b]).correlation)
            for b in names} for a in names}
        ths = {}
        for T in [float(t) for t in os.environ.get(
                'KIRBY_PRUNE_THRESHOLDS', '0.75,0.80,0.85,0.90,0.95').split(',')]:
            kept, dropped = greedy_eliminate(names, solo, corr, T)
            ths[f"{T:.2f}"] = {'kept': kept, 'n_kept': len(kept), 'dropped': dropped}
        saved[mn] = {'solo': {r: round(solo[r], 4) for r in names},
                     'solo_metric': solo_key, 'thresholds': ths}

    json.dump({'dataset': ds, 'signed_off_pool': pool, 'models': saved},
              open(os.path.join(out_dir, 'greedy.json'), 'w'), indent=2)

    models = list(saved)
    if not models:
        return
    print(f"\n  greedy elimination — reps kept per correlation threshold, per model")
    for T in sorted(saved[models[0]]['thresholds'], key=float):
        for mn in models:
            b = saved[mn]['thresholds'][T]
            print(f"    T={T} {mn:9s} keep {b['n_kept']:2d}   {', '.join(b['kept'])}")
        print()

    if not pool:
        return
    print(f"  the signed-off pool ({len(pool)} reps) under each model — solo rank, "
          f"and whether greedy keeps it")
    hdr = ' '.join(f"{m:>22s}" for m in models)
    print(f"    {'rep':24s} {hdr}")
    for r in pool:
        cells = []
        for mn in models:
            s = saved[mn]
            if r not in s['solo']:
                cells.append(f"{'not scored':>22s}"); continue
            rank = sorted(s['solo'], key=lambda x: -s['solo'][x]).index(r) + 1
            # Greedy is not monotone in T: a rep kept at a low threshold can be
            # dropped at a higher one, because the set of anchors changes. So
            # list the thresholds that keep it, not a range.
            keeps = [T for T in sorted(s['thresholds'], key=float)
                     if r in s['thresholds'][T]['kept']]
            span = ('kept at ' + ','.join(keeps)) if keeps else 'dropped at every T'
            cells.append(f"{'#%d %s' % (rank, span):>28s}")
        print(f"    {r:24s} " + ' '.join(cells))
    print(f"\n  reps OUTSIDE the pool that greedy keeps at T=0.95:")
    for mn in models:
        extra = [r for r in saved[mn]['thresholds']['0.95']['kept'] if r not in pool]
        print(f"    {mn:9s} {len(extra):2d}: {', '.join(extra) or '-'}")


def report(ds, metric, out):
    lower = metric in LOWER_IS_BETTER
    models = [m for m in MODELS if out.get(m)]
    if not models:
        return
    common = sorted(set.intersection(*[set(out[m]) for m in models]),
                    key=lambda r: (1 if lower else -1) * out[models[0]][r][metric])
    ranks = {m: {r: i + 1 for i, r in enumerate(
        sorted(common, key=lambda r: (1 if lower else -1) * out[m][r][metric]))}
        for m in models}
    print(f"\n  {ds} — {metric} ({'lower' if lower else 'higher'} is better), "
          f"{len(common)} reps")
    print(f"    {'rep':24s} " + ' '.join(f"{m:>18s}" for m in models))
    for r in common:
        cells = ' '.join(f"{out[m][r][metric]:+10.4f} (#{ranks[m][r]:2d})" for m in models)
        print(f"    {r:24s} {cells}")
    base = models[0]
    print(f"\n    rank agreement with {base} (Spearman of the {len(common)}-rep ranking):")
    for m in models[1:]:
        a = [ranks[base][r] for r in common]; b = [ranks[m][r] for r in common]
        rho = spearmanr(a, b).correlation
        top1 = min(common, key=lambda r: ranks[m][r])
        print(f"      {m:9s} rho={rho:+.3f}   its best rep: {top1} "
              f"(lightgbm rank #{ranks[base][top1]})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='+')
    ap.add_argument('--metric', required=True,
                    help='rmse | spearman | f1_weighted | auroc')
    a = ap.parse_args()
    print(f"[rep_prune_model_check] folds={N_FOLDS} cap={CAP} models={MODELS} "
          f"cache={CACHE}", flush=True)
    for ds in a.datasets:
        run(ds, a.metric)
