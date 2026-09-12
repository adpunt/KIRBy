"""Augmentation stage — supplement-percentage sweep (roadmap Phase 4.4 / Chat 6).

DECIDED BY USER (2026-08-20):
  * supplement size = a PERCENTAGE OF THE BASE REP'S WIDTH, value found by sweep
  * keep rule      = RANK ONCE, cut at the cap (no greedy A/B, no inner holdout)

Structure deliberately mirrors hybrid.allocate_weighted_global (the c31 engine
port) so the evidence transfers into the engine unchanged:
    per-source pre-trim -> pool -> ONE global ranking -> keep top-B.

The one difference, and it is the point of the augmentation category: the
supplement's share is NOT sized by standalone performance. paratope/structural
score rho 0.040 alone; performance-weighting would round them to zero. Sources
inside the supplement are pre-trimmed EQUALLY and then compete on the global
ranking.

Arms, per dataset (downstream + ranker = lightgbm):
  TARGET        best rep alone, full native width          <- the number to beat
  RANKONCE      base + top-(p% of base width) supplementary cols, global rank
  RANDOM        base + the SAME NUMBER of random cols       <- capacity control
  GREEDY        base + supplement via the CURRENT uncapped path (status quo)
  NOISE         base + p% pure-noise cols                   <- direction gate

Source sets: AUG (curated augmentation reps), BORROW (top cols of every OTHER
base rep), BOTH.

The global ranking is computed ONCE per (fold, source set) on [base | pool], so
every percentage in the sweep is a prefix of it — the whole sweep costs about
what a single arm costs. Ranking on [base | pool] (rather than pool alone) is
what makes the criterion MARGINAL value on top of the base: lightgbm will not
spend gain on a column whose information the base already carries.

Everything is a FIXED configuration: no search, no argmax inside a fold, so each
fold mean is unbiased and needs no nested CV to read.

Usage:
    python scratchpad/aug_stage_sweep.py flab_thermo_tresanco
    python scratchpad/aug_stage_sweep.py herg_ki logd buchwald_hartwig
    KIRBY_AUG_MODEL=rf python scratchpad/aug_stage_sweep.py herg_ki
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

OUT_ROOT = os.path.join(os.environ['KIRBY_CACHE_DIR'], 'aug_stage')

N_FOLDS = 5
N_ESTIMATORS = 150
SEED = 42

# lightgbm's own stochasticity (subsample=0.8 + colsample_bytree=0.8) means
# ADDING COLUMNS changes which features each tree sees, which moves the score on
# its own. A first run at 1 seed showed 128 columns of pure Gaussian noise
# "improving" tresanco by +0.007 — i.e. the model-level noise was larger than
# the effect being measured. Every arm is therefore averaged over N_SEEDS model
# seeds on the same fold before any delta is taken.
N_SEEDS = 3

# Percentages of the BASE rep's width to admit as supplement. 0 == the target.
PERCENTS = [0, 2, 5, 10, 25, 50, 100]

# Equal per-source pre-trim before pooling. Equal, NOT performance-weighted:
# see the module docstring. Sized so the pooled candidate set stays comfortably
# larger than the biggest cap we test (100% of the base width) -- with ~27
# donors, 64 each gives a 1728-column pool against a 1280-column top cap, so
# the global ranking is never pool-limited except at that extreme. Bounded so
# one wide donor cannot swamp the pool by column count alone.
PRETRIM_PER_SOURCE = 64

MODEL = os.environ.get('KIRBY_AUG_MODEL', 'lightgbm')


# ---------------------------------------------------------------- models ---
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


def _fit_score(Xtr, ytr, Xva, yva, is_clf, want_order=False):
    """Fit on train, score on val. Returns (metrics, order_or_None).

    When the downstream model IS lightgbm, the gain ordering is read off the
    SAME fitted booster — one fit, not two. That halves pass 1, which is the
    dominant cost on wide protein-embedding pools.
    """
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xva_s = sc.transform(Xva)
    ss_tot = float(np.sum((yva - np.mean(yva)) ** 2))

    per_seed, first = [], None
    for s in range(N_SEEDS):
        m = _make_model(is_clf, seed=SEED + s)
        m.fit(Xtr_s, ytr)
        if first is None:
            first = m
        pred = m.predict_proba(Xva_s)[:, 1] if is_clf else m.predict(Xva_s)
        rho = spearmanr(yva, pred).correlation
        ss_res = float(np.sum((yva - pred) ** 2))
        per_seed.append({
            'rmse': float(np.sqrt(np.mean((yva - pred) ** 2))),
            'spearman': float(rho) if rho == rho else float('nan'),
            'r2': 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')})
    mets = {k: float(np.nanmean([d[k] for d in per_seed])) for k in per_seed[0]}
    # Spread across model seeds on this fold = the model-level noise floor.
    mets['seed_sd'] = float(np.nanstd([d['spearman'] for d in per_seed]))

    order = None
    if want_order:
        if MODEL == 'lightgbm':
            imp = np.asarray(
                first.booster_.feature_importance(importance_type='gain'),
                dtype=float)
            order = np.argsort(imp)[::-1]
        else:
            order, _ = _gain_order(Xtr, ytr, is_clf)
    return mets, order


def _gain_order(X, y, is_clf):
    """Column ordering (best first) by lightgbm gain, fit on TRAIN rows only.

    The ranker is ALWAYS lightgbm_gain (the decided importance method), even
    when the downstream model is something else.
    """
    from lightgbm import LGBMClassifier, LGBMRegressor
    cls = LGBMClassifier if is_clf else LGBMRegressor
    m = cls(n_estimators=N_ESTIMATORS, max_depth=-1, num_leaves=31,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, verbosity=-1)
    sc = StandardScaler()
    m.fit(sc.fit_transform(X), y)
    imp = np.asarray(m.booster_.feature_importance(importance_type='gain'),
                     dtype=float)
    return np.argsort(imp)[::-1], imp


# ------------------------------------------------------------------ core ---
def run_dataset(ds):
    t0 = time.time()
    print(f"\n{'='*72}\n{ds}   (model={MODEL}, {N_FOLDS}-fold, seed={SEED})\n{'='*72}",
          flush=True)

    cfg = DATASET_REPS.get(ds, {})
    want_aug = list(cfg.get('augmentation', []))

    # load_data's default filters to the GENERAL pool only, so the augmentation
    # reps must be asked for BY NAME — this is precisely the "loosely wired"
    # step every helper re-implements (complexity_helpers.py:971 et al).
    data = load_data(ds)
    reps = {k: _clean(v) for k, v in data['reps'].items()}
    base_pool = [r for r in cfg.get('general', []) if r in reps]

    missing = [r for r in want_aug if r not in reps]
    if missing:
        print(f"  loading {len(missing)} augmentation reps by name: {missing}",
              flush=True)
        aug_data = load_data(ds, rep_preset='all', rep_names=missing)
        for r in missing:
            if r in aug_data.get('reps', {}):
                reps[r] = _clean(aug_data['reps'][r])

    y = np.asarray(data['labels'], dtype=float)
    groups = np.asarray(data['groups'])
    is_clf = data.get('task_type') == 'classification'

    # Tier invariant: augmentation n general == empty.
    aug_names = [r for r in want_aug if r in reps and r not in base_pool]

    print(f"  base pool: {len(base_pool)} reps | augmentation: {len(aug_names)} "
          f"({', '.join(f'{a}={reps[a].shape[1]}' for a in aug_names) or 'none'})",
          flush=True)
    if len(base_pool) < 2:
        print("  SKIP: need >=2 base reps")
        return None

    ng = len(np.unique(groups))
    k = min(N_FOLDS, ng)
    if is_clf:
        splits = list(StratifiedGroupKFold(n_splits=k, shuffle=True,
                                           random_state=SEED).split(y.reshape(-1, 1), y, groups))
    else:
        splits = list(GroupKFold(n_splits=k).split(y.reshape(-1, 1), y, groups))

    all_names = base_pool + aug_names

    # ---- PASS 1: per-rep fold scores + per-rep column orderings ------------
    # The score is a free by-product of the fit we need for the ordering, so
    # this pass yields the freshly-computed target ranking at no extra cost.
    rep_scores = {r: [] for r in all_names}
    orders = [dict() for _ in splits]
    print("\n  [pass 1] scoring every rep + computing its column ordering", flush=True)
    for fi, (tr, va) in enumerate(splits):
        for r in all_names:
            tr0 = time.time()
            X = reps[r]
            mets, order = _fit_score(X[tr], y[tr], X[va], y[va], is_clf,
                                     want_order=True)
            rep_scores[r].append(mets)
            orders[fi][r] = order
            print(f"      f{fi} {r:<22} {X.shape[1]:>5}d "
                  f"rho={mets['spearman']:+.3f} ({time.time()-tr0:.0f}s)", flush=True)
        print(f"    fold {fi+1}/{len(splits)} done ({time.time()-t0:.0f}s)", flush=True)

    prim = 'spearman'
    ranking = sorted(base_pool,
                     key=lambda r: -np.nanmean([m[prim] for m in rep_scores[r]]))
    best = ranking[0]
    W = reps[best].shape[1]
    print(f"\n  TARGET: {best} ({W} cols)  "
          f"rho={np.nanmean([m['spearman'] for m in rep_scores[best]]):+.4f} "
          f"rmse={np.nanmean([m['rmse'] for m in rep_scores[best]]):.4f} "
          f"r2={np.nanmean([m['r2'] for m in rep_scores[best]]):+.4f}")
    print(f"  next 4: " + ' | '.join(
        f"{r} {np.nanmean([m['spearman'] for m in rep_scores[r]]):+.3f}"
        for r in ranking[1:5]), flush=True)

    donors = {
        'AUG': aug_names,
        'BORROW': [r for r in base_pool if r != best],
    }
    donors['BOTH'] = donors['AUG'] + donors['BORROW']
    donors = {k_: v for k_, v in donors.items() if v}

    caps = [(p, int(round(p / 100.0 * W))) for p in PERCENTS]
    caps = [(p, c) for p, c in caps if p == 0 or c >= 1]
    print(f"\n  supplement caps (% of {W}): "
          + ', '.join(f'{p}%={c}' for p, c in caps), flush=True)

    # ---- PASS 2: arms -----------------------------------------------------
    results = []          # rows: {arm, source, percent, cap, fold, metrics, n_feat}

    def _record(arm, source, percent, cap, fi, mets, n_feat, added, prov=None):
        row = {'arm': arm, 'source': source, 'percent': percent,
               'cap': cap, 'fold': fi, 'n_features': int(n_feat),
               'n_added': int(added), **mets}
        if prov is not None:
            row['donors'] = prov      # {donor_rep: n_columns_contributed}
        results.append(row)

    def _build_out():
        return {
            'dataset': ds, 'model': MODEL, 'n_folds': len(splits), 'seed': SEED,
            'is_clf': is_clf, 'best_rep': best, 'base_width': W,
            'primary_metric': prim,
            'pretrim_per_source': PRETRIM_PER_SOURCE,
            'rep_ranking': [
                {'rep': r, 'dim': int(reps[r].shape[1]),
                 'spearman': float(np.nanmean([m['spearman'] for m in rep_scores[r]])),
                 'rmse': float(np.nanmean([m['rmse'] for m in rep_scores[r]])),
                 'r2': float(np.nanmean([m['r2'] for m in rep_scores[r]]))}
                for r in ranking],
            'aug_reps': {a: int(reps[a].shape[1]) for a in aug_names},
            'donors': {k_: v for k_, v in donors.items()},
            'rows': results,
        }

    def _write(o):
        os.makedirs(OUT_ROOT, exist_ok=True)
        with open(os.path.join(OUT_ROOT, f'{ds}__{MODEL}.json'), 'w') as f:
            json.dump(o, f, indent=2)

    rng = np.random.RandomState(SEED)
    print("\n  [pass 2] arms", flush=True)
    for fi, (tr, va) in enumerate(splits):
        Xb_tr, Xb_va = reps[best][tr], reps[best][va]

        # TARGET (percent 0) — the base alone.
        mets, _ = _fit_score(Xb_tr, y[tr], Xb_va, y[va], is_clf)
        _record('TARGET', '-', 0, 0, fi, mets, W, 0)

        # DIRECTION GATE: the WORST-ranked base rep, scored the same way. Its
        # delta must come out clearly negative. This is an unambiguous sign
        # test (unlike the noise arm, which a tree model can simply ignore).
        Xw = reps[ranking[-1]]
        mets, _ = _fit_score(Xw[tr], y[tr], Xw[va], y[va], is_clf)
        _record('DIRGATE', '-', 0, 0, fi, mets, Xw.shape[1], 0)

        for sname, dlist in donors.items():
            # -- build the candidate pool: EQUAL pre-trim per source --------
            cols_tr, cols_va, origins = [], [], []
            for d in dlist:
                Xd = reps[d]
                kd = min(PRETRIM_PER_SOURCE, Xd.shape[1])
                sel = orders[fi][d][:kd]
                cols_tr.append(Xd[tr][:, sel])
                cols_va.append(Xd[va][:, sel])
                origins.extend((d, int(c)) for c in sel)
            P_tr = np.hstack(cols_tr)
            P_va = np.hstack(cols_va)

            # -- ONE global ranking, computed on [base | pool] --------------
            # Ranking on the concatenation makes the criterion MARGINAL: a
            # column the base already explains earns little gain.
            gorder, _ = _gain_order(np.hstack([Xb_tr, P_tr]), y[tr], is_clf)
            gorder = np.asarray([i - W for i in gorder if i >= W])  # pool only

            for p, cap in caps:
                if p == 0:
                    continue
                cap_eff = min(cap, P_tr.shape[1])

                keep = gorder[:cap_eff]
                mets, _ = _fit_score(np.hstack([Xb_tr, P_tr[:, keep]]), y[tr],
                                     np.hstack([Xb_va, P_va[:, keep]]), y[va], is_clf)
                prov = {}
                for i in keep:
                    prov[origins[i][0]] = prov.get(origins[i][0], 0) + 1
                _record('RANKONCE', sname, p, cap_eff, fi, mets,
                        W + cap_eff, cap_eff,
                        prov=dict(sorted(prov.items(), key=lambda kv: -kv[1])))

                ridx = rng.choice(P_tr.shape[1], size=cap_eff, replace=False)
                mets, _ = _fit_score(np.hstack([Xb_tr, P_tr[:, ridx]]), y[tr],
                                     np.hstack([Xb_va, P_va[:, ridx]]), y[va], is_clf)
                _record('RANDOM', sname, p, cap_eff, fi, mets, W + cap_eff, cap_eff)

        # -- direction gate: pure noise MUST lose (source-independent) ------
        ncap = max(1, int(round(0.10 * W)))
        mets, _ = _fit_score(
            np.hstack([Xb_tr, rng.normal(size=(len(tr), ncap))]), y[tr],
            np.hstack([Xb_va, rng.normal(size=(len(va), ncap))]), y[va], is_clf)
        _record('NOISE', '-', 10, ncap, fi, mets, W + ncap, ncap)

        # -- status quo: the CURRENT uncapped greedy path -------------------
        # Expensive: up to ~50 sequential model fits per source set per fold,
        # each on [base | growing]. Off by default; it is the comparison arm,
        # not the result. KIRBY_AUG_GREEDY=1 to include it.
        for sname, dlist in (donors.items()
                             if os.environ.get('KIRBY_AUG_GREEDY') == '1' else []):
            try:
                from kirby.find_best_hybrid import Blueprint
                from experiments._shared import build_fold_hybrid
                bp = Blueprint(
                    reps=[best], allocation_method='performance_weighted',
                    importance_method='lightgbm_gain', internal_evaluator='match',
                    budget=W, step_size=5, augmentation_strategy='uncapped',
                    metric=prim, task_type='classification' if is_clf else 'regression',
                    lower_is_better=False)
                a_tr = {d: reps[d][tr] for d in dlist}
                a_va = {d: reps[d][va] for d in dlist}
                Xh_tr, Xh_va, _fi = build_fold_hybrid(
                    {best: Xb_tr}, {best: Xb_va}, y[tr], bp,
                    augmentations_train=a_tr, augmentations_val=a_va,
                    groups=(groups[tr] if groups is not None else None))
                mets, _ = _fit_score(Xh_tr, y[tr], Xh_va, y[va], is_clf)
                _record('GREEDY', sname, -1, -1, fi, mets, Xh_tr.shape[1],
                        Xh_tr.shape[1] - W)
            except Exception as e:
                print(f"      GREEDY/{sname} fold {fi} failed: {e}", flush=True)

        print(f"    fold {fi+1}/{len(splits)} arms done ({time.time()-t0:.0f}s)",
              flush=True)
        _write(_build_out())   # incremental: partial results survive a kill

    out = _build_out()
    _write(out)
    path = os.path.join(OUT_ROOT, f'{ds}__{MODEL}.json')
    print(f"\n  -> {path}   ({time.time()-t0:.0f}s)", flush=True)
    summarise(out)
    return out


# ------------------------------------------------------------- reporting ---
def summarise(out):
    """Paired per-fold deltas vs TARGET, direction-corrected."""
    rows = out['rows']
    prim = out['primary_metric']
    lower_better = prim in ('rmse', 'mae')
    nf = out['n_folds']

    tgt = {r['fold']: r for r in rows if r['arm'] == 'TARGET'}

    def delta(r):
        d = r[prim] - tgt[r['fold']][prim]
        return -d if lower_better else d

    print(f"\n  paired delta vs TARGET ({out['best_rep']}, {out['base_width']} cols) "
          f"— metric {prim}, positive = better")
    print(f"  {'arm':<10} {'source':<7} {'%':>4} {'added':>6} "
          f"{'mean d':>9} {'folds won':>10}")
    print('  ' + '-' * 54)

    seen = {}
    for r in rows:
        if r['arm'] == 'TARGET':
            continue
        seen.setdefault((r['arm'], r['source'], r['percent']), []).append(r)
    for (arm, src, pct), rs in sorted(
            seen.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        ds_ = [delta(r) for r in rs]
        won = sum(1 for d in ds_ if d > 0)
        pct_s = 'greedy' if pct < 0 else f'{pct}%'
        print(f"  {arm:<10} {src:<7} {pct_s:>4} {rs[0]['n_added']:>6} "
              f"{np.mean(ds_):>+9.4f} {won:>7}/{len(ds_)}")

    dg = [delta(r) for r in rows if r['arm'] == 'DIRGATE']
    if dg:
        v = float(np.mean(dg))
        print(f"\n  DIRECTION GATE (worst rep must LOSE to best): {v:+.4f} "
              f"{'PASS' if v < 0 else '*** FAIL — metric direction is wrong ***'}")

    noise = [delta(r) for r in rows if r['arm'] == 'NOISE']
    if noise:
        v, sd = float(np.mean(noise)), float(np.std(noise))
        seed_sd = float(np.nanmean([r.get('seed_sd', np.nan) for r in rows
                                    if r['arm'] == 'TARGET']))
        print(f"  NOISE FLOOR (adding meaningless columns): {v:+.4f} "
              f"(fold sd {sd:.4f}) | model-seed sd on target {seed_sd:.4f}")
        print("  -> any real gain must clearly exceed this floor.")


if __name__ == '__main__':
    targets = sys.argv[1:] or ['flab_thermo_tresanco']
    for ds in targets:
        try:
            run_dataset(ds)
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"DATASET {ds} FAILED", flush=True)
