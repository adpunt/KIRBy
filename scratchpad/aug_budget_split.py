"""Augmentation stage, take 2 — supplement competes for a FIXED budget.

WHY THIS REPLACES THE APPEND EXPERIMENT
    aug_stage_sweep.py appended supplementary columns on top of the base rep at
    its FULL native width. On flab_thermo_tresanco that is 1280 columns against
    ~538 training rows -- already past saturation -- so nothing could help, and
    randomly chosen columns scored as well as selected ones. Confirmed against
    c31: its best cell on that dataset was budget 500 (i.e. SMALLER than the
    1280 baseline) beating the baseline by +0.0505 Spearman. The useful
    direction is DOWN, not up.

WHAT THIS DOES INSTEAD
    Hold the finished feature count fixed at B. Split B between the base rep's
    own top columns and supplementary columns:
        n_sup  = min(round(p * B), pool_width)      <- the KNOB FIX
        n_base = B - n_sup
    So the supplement has to EARN its slots by displacing base columns. p=0 is
    the pure base-cut arm (top-B of the base rep, no supplement), which is the
    honest control for "is the supplement worth any budget at all".

KNOB FIX (the reported bug)
    Previously cap = round(p * base_width) with no pool bound, so on tresanco
    5/10/25/50/100% all resolved to the same 48 columns and returned identical
    numbers. Now the cap is bounded by the pool, the EFFECTIVE count is stored
    on every row, and duplicate (n_base, n_sup) shapes are computed once.

Arms per (fold, budget):
    BASECUT      top-B of the base rep alone                 (p = 0)
    SPLIT        (B-n_sup) base + n_sup ranked supplement
    SPLITRAND    same shape, supplement chosen at random     (control)
    plus TARGET (base at full native width) and DIRGATE, once per fold.

The supplementary ranking is recomputed per budget against the base columns
ACTUALLY in use, because a supplementary column that duplicates a base column
dropped by the cut becomes valuable again.

    python scratchpad/aug_budget_split.py flab_thermo_tresanco
"""
import os
import sys
import json
import time

import numpy as np

os.environ.setdefault('KIRBY_CACHE_DIR', '/Volumes/seagate/kirby_cache')
sys.path.insert(0, 'src')
sys.path.insert(0, 'tests')
sys.path.insert(0, 'tests/experiments')

from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from experiments._shared import load_data
from experiments.complexity.complexity_helpers import DATASET_REPS

OUT_ROOT = os.path.join(os.environ['KIRBY_CACHE_DIR'], 'aug_budget_split')

N_FOLDS = 5
N_ESTIMATORS = 150
SEED = 42
# Averages out lightgbm's own column sampling. 2 rather than 3: with 5 folds
# that is still 10 measurements per arm, and folds are worth far more than
# seeds -- measured on the completed append run, dropping to 2 FOLDS flips the
# sign of 41% of arm settings, so fold count is the last thing to cut.
N_SEEDS = 2
PRETRIM_PER_SOURCE = 64          # equal per donor; NOT performance-weighted
BUDGETS = [100, 250, 500, 1000]
FRACTIONS = [0.0, 0.05, 0.10, 0.25, 0.50]
MODEL = os.environ.get('KIRBY_AUG_MODEL', 'lightgbm')


def _make_model(is_clf, seed=SEED):
    if MODEL == 'rf':
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        cls = RandomForestClassifier if is_clf else RandomForestRegressor
        return cls(n_estimators=300, n_jobs=-1, random_state=seed)
    from lightgbm import LGBMClassifier, LGBMRegressor
    cls = LGBMClassifier if is_clf else LGBMRegressor
    return cls(n_estimators=N_ESTIMATORS, max_depth=-1, num_leaves=31,
               learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
               random_state=seed, n_jobs=-1, verbosity=-1)


def _clean(X):
    return np.nan_to_num(np.clip(np.asarray(X, dtype=np.float64), -1e10, 1e10),
                         nan=0.0, posinf=1e10, neginf=-1e10)


def _score(Xtr, ytr, Xva, yva, is_clf):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xva_s = sc.transform(Xva)
    ss_tot = float(np.sum((yva - np.mean(yva)) ** 2))
    acc = []
    for s in range(N_SEEDS):
        m = _make_model(is_clf, SEED + s)
        m.fit(Xtr_s, ytr)
        pred = m.predict_proba(Xva_s)[:, 1] if is_clf else m.predict(Xva_s)
        rho = spearmanr(yva, pred).correlation
        acc.append({'rmse': float(np.sqrt(np.mean((yva - pred) ** 2))),
                    'spearman': float(rho) if rho == rho else float('nan'),
                    'r2': 1.0 - float(np.sum((yva - pred) ** 2)) / ss_tot
                          if ss_tot > 0 else float('nan')})
    return {k: float(np.nanmean([d[k] for d in acc])) for k in acc[0]}


def _order(X, y, is_clf):
    """Column ordering, best first, by lightgbm gain. Train rows only."""
    from lightgbm import LGBMClassifier, LGBMRegressor
    cls = LGBMClassifier if is_clf else LGBMRegressor
    m = cls(n_estimators=N_ESTIMATORS, max_depth=-1, num_leaves=31,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, verbosity=-1)
    sc = StandardScaler()
    m.fit(sc.fit_transform(X), y)
    imp = np.asarray(m.booster_.feature_importance(importance_type='gain'),
                     dtype=float)
    return np.argsort(imp)[::-1]


def run(ds):
    t0 = time.time()
    print(f"\n{'='*74}\n{ds}  (model={MODEL}, {N_FOLDS} folds, {N_SEEDS} seeds)\n{'='*74}",
          flush=True)

    cfg = DATASET_REPS.get(ds, {})
    want_aug = list(cfg.get('augmentation', []))
    data = load_data(ds)
    reps = {k: _clean(v) for k, v in data['reps'].items()}
    base_pool = [r for r in cfg.get('general', []) if r in reps]

    missing = [r for r in want_aug if r not in reps]
    if missing:
        ad = load_data(ds, rep_preset='all', rep_names=missing)
        for r in missing:
            if r in ad.get('reps', {}):
                reps[r] = _clean(ad['reps'][r])
    aug_names = [r for r in want_aug if r in reps and r not in base_pool]

    y = np.asarray(data['labels'], dtype=float)
    groups = np.asarray(data['groups'])
    is_clf = data.get('task_type') == 'classification'

    ng = len(np.unique(groups))
    k = min(N_FOLDS, ng)
    if is_clf:
        splits = list(StratifiedGroupKFold(n_splits=k, shuffle=True,
                                           random_state=SEED)
                      .split(y.reshape(-1, 1), y, groups))
    else:
        splits = list(GroupKFold(n_splits=k).split(y.reshape(-1, 1), y, groups))

    # ---- best rep -------------------------------------------------------
    # Shortlist from a prior local ranking (the aug_stage sweep, else the
    # validity gate), then re-score the shortlist UNDER THIS PROTOCOL and take
    # the winner. Prior rankings used 3 folds / 1 seed and disagree with the
    # 5-fold answer on near-ties (tresanco: saprot_cdr at 3 folds, saprot at 5),
    # so they are used only to shortlist, never to name the target.
    shortlist = []
    prev = os.path.join(os.environ['KIRBY_CACHE_DIR'], 'aug_stage',
                        f'{ds}__{MODEL}.json')
    vpath = os.path.join(os.environ['KIRBY_CACHE_DIR'], 'validity', f'{ds}.json')
    try:
        if os.path.exists(prev):
            shortlist = [e['rep'] for e in json.load(open(prev))['rep_ranking'][:5]]
        elif os.path.exists(vpath):
            vj = json.load(open(vpath))['reps']
            shortlist = [k for k, _ in sorted(
                ((k, v.get('score')) for k, v in vj.items() if v.get('score')),
                key=lambda kv: -kv[1])[:5]]
    except Exception:
        shortlist = []
    shortlist = [r for r in shortlist if r in base_pool] or base_pool

    print(f"  scoring {len(shortlist)} candidate base reps under this protocol: "
          f"{shortlist}", flush=True)
    cand = {}
    for r in shortlist:
        X = reps[r]
        cand[r] = float(np.mean([
            _score(X[tr], y[tr], X[va], y[va], is_clf)['spearman']
            for tr, va in splits]))
        print(f"    {r:<22} {reps[r].shape[1]:>5}d  rho={cand[r]:+.4f}", flush=True)
    best = max(cand, key=cand.get)
    W = reps[best].shape[1]

    donors = {'AUG': aug_names,
              'BORROW': [r for r in base_pool if r != best]}
    donors['BOTH'] = donors['AUG'] + donors['BORROW']
    donors = {k_: v for k_, v in donors.items() if v}
    # Budgets above the training-row count are not worth spending time on: the
    # feature set would be wider than the data it is fitted on, which is the
    # regime the full-width baseline is already stuck in. On tresanco
    # (~538 train rows) this drops B=1000, which fold 1 measured as the worst
    # cell (-0.025 vs the full-width target). Larger datasets keep all four.
    n_train = int(len(y) * (1.0 - 1.0 / len(splits)))
    budgets = [b for b in BUDGETS if b <= W and b <= n_train] or [min(W, n_train)]
    print(f"  base={best} ({W} cols) | budgets={budgets} | "
          f"donors: " + ', '.join(f'{k_}={len(v)}' for k_, v in donors.items()),
          flush=True)

    rows = []
    rng = np.random.RandomState(SEED)

    for fi, (tr, va) in enumerate(splits):
        Xb = reps[best]
        ob = _order(Xb[tr], y[tr], is_clf)

        rows.append({'arm': 'TARGET', 'source': '-', 'budget': W, 'frac': 0.0,
                     'n_base': W, 'n_sup': 0, 'fold': fi,
                     **_score(Xb[tr], y[tr], Xb[va], y[va], is_clf)})

        worst = min(base_pool, key=lambda r: reps[r].shape[1]) if base_pool else best

        # per-donor orderings (one fit each), reused across budgets
        dord = {}
        for d in sorted({d for v in donors.values() for d in v}):
            dord[d] = _order(reps[d][tr], y[tr], is_clf)

        for B in budgets:
            bsel = ob[:B]
            rows.append({'arm': 'BASECUT', 'source': '-', 'budget': B, 'frac': 0.0,
                         'n_base': int(B), 'n_sup': 0, 'fold': fi,
                         **_score(Xb[tr][:, bsel], y[tr], Xb[va][:, bsel], y[va],
                                  is_clf)})

            for sname, dlist in donors.items():
                ptr, pva, origins = [], [], []
                for d in dlist:
                    kd = min(PRETRIM_PER_SOURCE, reps[d].shape[1])
                    sel = dord[d][:kd]
                    ptr.append(reps[d][tr][:, sel])
                    pva.append(reps[d][va][:, sel])
                    origins.extend((d, int(c)) for c in sel)
                P_tr, P_va = np.hstack(ptr), np.hstack(pva)

                # rank supplement against the base columns ACTUALLY in use
                nb0 = len(bsel)
                g = _order(np.hstack([Xb[tr][:, bsel], P_tr]), y[tr], is_clf)
                g = np.array([i - nb0 for i in g if i >= nb0])

                seen_shapes = set()
                for p in FRACTIONS:
                    if p == 0:
                        continue
                    n_sup = min(int(round(p * B)), P_tr.shape[1])   # KNOB FIX
                    n_base = B - n_sup
                    if n_sup < 1 or n_base < 1:
                        continue
                    if (n_base, n_sup) in seen_shapes:
                        continue          # duplicate shape -> compute once
                    seen_shapes.add((n_base, n_sup))

                    bs = ob[:n_base]
                    keep = g[:n_sup]
                    prov = {}
                    for i in keep:
                        prov[origins[i][0]] = prov.get(origins[i][0], 0) + 1
                    rows.append({
                        'arm': 'SPLIT', 'source': sname, 'budget': B,
                        'frac': p, 'frac_effective': n_sup / float(B),
                        'n_base': int(n_base), 'n_sup': int(n_sup), 'fold': fi,
                        'donors': dict(sorted(prov.items(), key=lambda kv: -kv[1])),
                        **_score(np.hstack([Xb[tr][:, bs], P_tr[:, keep]]), y[tr],
                                 np.hstack([Xb[va][:, bs], P_va[:, keep]]), y[va],
                                 is_clf)})

                    ridx = rng.choice(P_tr.shape[1], size=n_sup, replace=False)
                    rows.append({
                        'arm': 'SPLITRAND', 'source': sname, 'budget': B,
                        'frac': p, 'frac_effective': n_sup / float(B),
                        'n_base': int(n_base), 'n_sup': int(n_sup), 'fold': fi,
                        **_score(np.hstack([Xb[tr][:, bs], P_tr[:, ridx]]), y[tr],
                                 np.hstack([Xb[va][:, bs], P_va[:, ridx]]), y[va],
                                 is_clf)})

        Xw = reps[worst]
        rows.append({'arm': 'DIRGATE', 'source': '-', 'budget': Xw.shape[1],
                     'frac': 0.0, 'n_base': Xw.shape[1], 'n_sup': 0, 'fold': fi,
                     **_score(Xw[tr], y[tr], Xw[va], y[va], is_clf)})

        os.makedirs(OUT_ROOT, exist_ok=True)
        with open(os.path.join(OUT_ROOT, f'{ds}__{MODEL}.json'), 'w') as f:
            json.dump({'dataset': ds, 'model': MODEL, 'best_rep': best,
                       'base_width': W, 'budgets': budgets,
                       'fractions': FRACTIONS, 'n_folds': len(splits),
                       'n_seeds': N_SEEDS, 'aug_reps': {a: int(reps[a].shape[1])
                                                        for a in aug_names},
                       'donors': donors, 'rows': rows}, f, indent=2)
        print(f"    fold {fi+1}/{len(splits)} done ({time.time()-t0:.0f}s)",
              flush=True)

    summarise(ds)


def summarise(ds):
    path = os.path.join(OUT_ROOT, f'{ds}__{MODEL}.json')
    o = json.load(open(path))
    rows = o['rows']
    prim = 'spearman'
    tgt = {r['fold']: r for r in rows if r['arm'] == 'TARGET'}
    tm = float(np.nanmean([tgt[f][prim] for f in tgt]))

    def d(r):
        return r[prim] - tgt[r['fold']][prim]

    print(f"\n  TARGET {o['best_rep']} @ full {o['base_width']} cols: "
          f"{prim}={tm:+.4f}")
    dg = [d(r) for r in rows if r['arm'] == 'DIRGATE']
    if dg:
        print(f"  direction gate: {np.mean(dg):+.4f} "
              f"{'PASS' if np.mean(dg) < 0 else '*** FAIL ***'}")

    g = {}
    for r in rows:
        if r['arm'] in ('TARGET', 'DIRGATE'):
            continue
        g.setdefault((r['budget'], r['source'], r['arm'],
                      r['n_sup']), []).append(r)

    print(f"\n  {'budget':>6} {'source':<7} {'arm':<10} {'n_base':>7} {'n_sup':>6} "
          f"{'delta':>9} {'SE':>7} {'won':>6}")
    print('  ' + '-' * 64)
    for kk in sorted(g):
        rs = g[kk]
        arr = np.array([d(r) for r in rs])
        se = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else float('nan')
        print(f"  {kk[0]:>6} {kk[1]:<7} {kk[2]:<10} {rs[0]['n_base']:>7} "
              f"{kk[3]:>6} {arr.mean():>+9.4f} {se:>7.4f} "
              f"{sum(1 for v in arr if v > 0):>3}/{len(arr)}")


if __name__ == '__main__':
    for ds in (sys.argv[1:] or ['flab_thermo_tresanco']):
        try:
            run(ds)
        except Exception:
            import traceback
            traceback.print_exc()
