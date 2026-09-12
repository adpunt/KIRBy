#!/usr/bin/env python3
"""Autopsy of the augmentation ADMISSION mechanism.

The arms measured on herg_ki kept 3-6 columns out of 45-338 offered, and kept
random columns at the SAME rate as importance-ranked ones. That says the
accept/reject test is not responding to signal. This script instruments the
real loop to find out which part is broken, measuring five things per fold:

  1. NOISE FLOOR. Refit the base alone with n different model seeds and with
     the columns permuted. That is the spread of the quantity the accept test
     compares against zero. If one candidate column moves the score by less
     than this, every accept/reject decision is a coin flip.

  2. THE LOOP, INSTRUMENTED. Per candidate: score before, score after, delta,
     accepted?, and the running bar. The bar is `current_score`, which only
     ever rises, so the question is whether it ratchets above the noise ceiling
     after a few lucky early accepts and then rejects everything.

  3. INNER GAIN vs OUTER GAIN. The loop maximises the internal holdout. Score
     the resulting column set on the untouched OUTER fold. If inner says +0.02
     and outer says -0.003, the mechanism is fitting its own holdout.

  4. RANDOM POOL. The identical loop fed pure gaussian columns. If it admits as
     many as the real pool, admission is noise, full stop.

  5. RANKING QUALITY. For the pooled candidates, compare the rf_importance
     order actually used against (a) each column's own correlation with the
     target and (b) its correlation with the BASE MODEL'S RESIDUALS, which is
     what a complementarity ranker should be using.

Read-only w.r.t. the repo. Writes one JSON.

Usage:
    KIRBY_CACHE_DIR=/Volumes/seagate/kirby_cache python3 -u \
      scratchpad/aug_mechanism_autopsy.py --dataset herg_ki --base-rep mhg_gnn
"""
import argparse
import json
import os
import sys
import time

# LOCAL macOS ONLY — must precede the experiments._shared import. See
# aug_pass1_target.py for the full note; without it the first lightgbm fit
# after load_data segfaults (exit 139, no traceback).
try:
    import numpy as _np
    import lightgbm as _lgb
    _lgb.LGBMRegressor(n_estimators=2, verbose=-1).fit(
        _np.zeros((8, 2)), _np.arange(8.0))
except Exception as _e:  # pragma: no cover
    print(f"  [warmup] lightgbm warm-up skipped: {_e}")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_model_factories, cv_split, get_domain_metric,
)
from experiments.complexity.complexity_helpers import DATASET_REPS

from kirby.hybrid import _pool_and_rank_aug_features, _evaluate_allocation


def _seeded(factory, seed):
    m = factory()
    try:
        if 'random_state' in m.get_params():
            m.set_params(random_state=seed)
    except Exception:
        pass
    return m


def _score(Xtr, Xvl, ytr, yvl, model, metric_fn, lower_is_better):
    """Exactly what the loop uses to decide: higher = better."""
    return _evaluate_allocation(Xtr, Xvl, ytr, yvl, model=model,
                                metric_fn=metric_fn,
                                lower_is_better=lower_is_better)


def _inner_split(groups, labels, n_samples, validation_split=0.2):
    """Replica of create_hybrid's internal holdout (hybrid.py:3642-3665)."""
    if groups is not None and len(np.unique(groups)) >= 2:
        n_groups = len(np.unique(groups))
        n_splits = max(2, min(int(round(1.0 / validation_split)), n_groups))
        gkf = GroupKFold(n_splits=n_splits)
        return next(gkf.split(np.zeros(n_samples), labels, groups))
    n_val = max(int(n_samples * validation_split), 1)
    idx = np.random.RandomState(42).permutation(n_samples)
    return idx[n_val:], idx[:n_val]


def _greedy(ranked_tr, ranked_vl, X_tr, X_vl, ytr, yvl, model, metric_fn,
            lower_is_better, prov, patience=10, tag=''):
    """The real augment_uncapped loop, instrumented. Same accept rule."""
    base = _score(X_tr, X_vl, ytr, yvl, model, metric_fn, lower_is_better)
    cur_tr, cur_vl = X_tr.copy(), X_vl.copy()
    cur = base
    trace, kept_cols, consec = [], [], 0
    n_examined = 0
    for i in range(ranked_tr.shape[1]):
        if consec >= patience:
            break
        n_examined += 1
        tt = np.hstack([cur_tr, ranked_tr[:, i:i + 1]])
        vv = np.hstack([cur_vl, ranked_vl[:, i:i + 1]])
        new = _score(tt, vv, ytr, yvl, model, metric_fn, lower_is_better)
        acc = bool(new > cur)
        trace.append({'rank': i, 'src': prov[i][0] if prov else None,
                      'bar': float(cur), 'new': float(new),
                      'delta': float(new - cur), 'accepted': acc})
        if acc:
            cur_tr, cur_vl, cur = tt, vv, new
            kept_cols.append(i)
            consec = 0
        else:
            consec += 1
    print(f"      [{tag}] examined {n_examined}/{ranked_tr.shape[1]}, "
          f"kept {len(kept_cols)}, inner {base:+.4f} -> {cur:+.4f} "
          f"({cur - base:+.4f})", flush=True)
    return {'baseline': float(base), 'final': float(cur),
            'inner_gain': float(cur - base), 'kept': kept_cols,
            'n_examined': n_examined, 'n_offered': int(ranked_tr.shape[1]),
            'trace': trace}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='herg_ki')
    ap.add_argument('--base-rep', required=True)
    ap.add_argument('--model', default='lightgbm')
    ap.add_argument('--n-folds', type=int, default=3)
    ap.add_argument('--noise-seeds', type=int, default=8)
    ap.add_argument('--max-candidates', type=int, default=50)
    ap.add_argument('--top-n', type=int, default=5,
                    help='per-source budget, matching the C_topreps arm')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cache = os.environ.get('KIRBY_CACHE_DIR')
    if not cache or 'seagate' not in cache:
        print(f"REFUSING TO RUN: KIRBY_CACHE_DIR={cache!r}")
        return 2

    ds = args.dataset
    pool = DATASET_REPS[ds]['general']
    aug_names = DATASET_REPS[ds].get('augmentation') or []
    data = load_data(ds, rep_names=pool + aug_names)
    reps, labels = data['reps'], data['labels']
    groups = data.get('groups')
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    factory = get_model_factories([args.model], task_type=data['task_type'],
                                  n_samples=len(labels))[args.model]
    model = factory()

    base_X = reps[args.base_rep]
    aug_full = {a: reps[a] for a in aug_names}
    print(f"{ds}: base={args.base_rep} {base_X.shape}, metric={metric_name} "
          f"lower_is_better={lower_is_better}")
    print(f"  aug pool: {[(a, v.shape[1]) for a, v in aug_full.items()]}")

    splits = list(cv_split(groups, args.n_folds, reps[pool[0]], labels))
    sign = -1.0 if lower_is_better else 1.0
    out_folds = []
    t0 = time.time()

    for fi, (otr, ova) in enumerate(splits):
        print(f"\n  === outer fold {fi} (train {len(otr)}, test {len(ova)}) "
              f"[{time.time()-t0:.0f}s]", flush=True)
        y_tr_all = labels[otr]
        g_tr = groups[otr] if groups is not None else None

        # Inner holdout, exactly as create_hybrid draws it.
        itr, ivl = _inner_split(g_tr, y_tr_all, len(otr))
        itr_g, ivl_g = otr[itr], otr[ivl]
        ytr, yvl = labels[itr_g], labels[ivl_g]

        Xb_tr, Xb_vl = base_X[itr_g], base_X[ivl_g]

        # --- 1. NOISE FLOOR on the inner holdout ---------------------------
        seed_scores, perm_scores = [], []
        for s in range(args.noise_seeds):
            m = _seeded(factory, 42 + s)
            seed_scores.append(_score(Xb_tr, Xb_vl, ytr, yvl, m, metric_fn,
                                      lower_is_better))
            rs = np.random.RandomState(1000 + s)
            order = rs.permutation(Xb_tr.shape[1])
            perm_scores.append(_score(Xb_tr[:, order], Xb_vl[:, order], ytr,
                                      yvl, model, metric_fn, lower_is_better))
        noise = {
            'seed_sd': float(np.std(seed_scores)),
            'seed_range': float(np.ptp(seed_scores)),
            'colperm_sd': float(np.std(perm_scores)),
            'colperm_range': float(np.ptp(perm_scores)),
            'seed_scores': [float(x) for x in seed_scores],
        }
        print(f"    noise floor: seed sd {noise['seed_sd']:.5f} "
              f"(range {noise['seed_range']:.5f}), colperm sd "
              f"{noise['colperm_sd']:.5f} (range {noise['colperm_range']:.5f})",
              flush=True)

        # --- 2/3. THE REAL LOOP, and what it delivers outside -------------
        aug_tr = {a: v[itr_g] for a, v in aug_full.items()}
        aug_vl = {a: v[ivl_g] for a, v in aug_full.items()}
        ranked_tr, ranked_vl, prov = _pool_and_rank_aug_features(
            aug_tr, aug_vl, ytr, Xb_tr, model, 'rf_importance',
            False, args.max_candidates, per_source_budget=args.top_n)

        real = _greedy(ranked_tr, ranked_vl, Xb_tr, Xb_vl, ytr, yvl, model,
                       metric_fn, lower_is_better, prov, tag='real')

        # Rebuild the chosen columns on FULL data, score on the outer fold.
        # prov indices refer to the ORIGINAL source arrays.
        def _outer_gain(kept_ranks):
            base_o = _score(base_X[otr], base_X[ova], labels[otr], labels[ova],
                            model, metric_fn, lower_is_better)
            if not kept_ranks:
                return base_o, base_o, 0.0
            cols = [aug_full[prov[r][0]][:, prov[r][1]:prov[r][1] + 1]
                    for r in kept_ranks]
            extra = np.hstack(cols)
            aug_o = _score(np.hstack([base_X[otr], extra[otr]]),
                           np.hstack([base_X[ova], extra[ova]]),
                           labels[otr], labels[ova], model, metric_fn,
                           lower_is_better)
            return base_o, aug_o, float(aug_o - base_o)

        b_o, a_o, gain_o = _outer_gain(real['kept'])
        print(f"    OUTER: base {b_o:+.4f} -> aug {a_o:+.4f} "
              f"({gain_o:+.4f})   [inner claimed {real['inner_gain']:+.4f}]",
              flush=True)

        # --- 4. RANDOM POOL through the identical loop --------------------
        rs = np.random.RandomState(7 + fi)
        n_rand = ranked_tr.shape[1]
        rand_tr = rs.standard_normal((Xb_tr.shape[0], n_rand))
        rand_vl = rs.standard_normal((Xb_vl.shape[0], n_rand))
        rand = _greedy(rand_tr, rand_vl, Xb_tr, Xb_vl, ytr, yvl, model,
                       metric_fn, lower_is_better,
                       [('__gaussian__', j) for j in range(n_rand)],
                       tag='pure-noise')

        # --- 5. RANKING QUALITY -------------------------------------------
        # What the used ranker (rf_importance on the aug pool ALONE) picks,
        # versus target correlation and versus BASE-RESIDUAL correlation.
        sc = StandardScaler()
        Xb_s = sc.fit_transform(Xb_tr)
        mb = _seeded(factory, 42)
        mb.fit(Xb_s, ytr)
        resid = ytr - mb.predict(Xb_s)
        tgt_r, res_r = [], []
        for (an, ci) in prov:
            col = aug_full[an][itr_g, ci]
            if np.std(col) > 1e-12:
                tgt_r.append(abs(float(np.corrcoef(col, ytr)[0, 1])))
                res_r.append(abs(float(np.corrcoef(col, resid)[0, 1])))
            else:
                tgt_r.append(0.0)
                res_r.append(0.0)
        tgt_r, res_r = np.array(tgt_r), np.array(res_r)
        from scipy.stats import spearmanr
        used_order = np.arange(len(prov))  # prov is already in ranked order
        rank_diag = {
            'rho_usedrank_vs_target': float(
                spearmanr(-used_order, tgt_r).correlation),
            'rho_usedrank_vs_residual': float(
                spearmanr(-used_order, res_r).correlation),
            'rho_target_vs_residual': float(
                spearmanr(tgt_r, res_r).correlation),
            'top5_used_srcs': [prov[i][0] for i in range(min(5, len(prov)))],
            'mean_target_r': float(np.mean(tgt_r)),
            'mean_residual_r': float(np.mean(res_r)),
        }
        print(f"    ranker: rho(used rank, target r)={rank_diag['rho_usedrank_vs_target']:+.3f}  "
              f"rho(used rank, residual r)={rank_diag['rho_usedrank_vs_residual']:+.3f}",
              flush=True)

        out_folds.append({
            'fold': fi, 'noise': noise, 'real': real, 'random': rand,
            'outer_base': float(b_o), 'outer_aug': float(a_o),
            'outer_gain': float(gain_o), 'ranking': rank_diag,
            'n_prov': len(prov),
        })

    # ---- summary ---------------------------------------------------------
    print("\n" + "=" * 72)
    kr = [f['real']['kept'] for f in out_folds]
    kn = [f['random']['kept'] for f in out_folds]
    print(f"  columns admitted  REAL pool: {[len(k) for k in kr]}  "
          f"mean {np.mean([len(k) for k in kr]):.1f}")
    print(f"  columns admitted  PURE NOISE: {[len(k) for k in kn]}  "
          f"mean {np.mean([len(k) for k in kn]):.1f}")
    ig = [f['real']['inner_gain'] for f in out_folds]
    og = [f['outer_gain'] for f in out_folds]
    print(f"  inner gain claimed: {np.mean(ig):+.4f}   "
          f"outer gain delivered: {np.mean(og):+.4f}")
    ns = [f['noise']['seed_sd'] for f in out_folds]
    ad = [abs(t['delta']) for f in out_folds for t in f['real']['trace']]
    print(f"  |per-candidate delta| median {np.median(ad):.5f}  vs "
          f"model-seed sd {np.mean(ns):.5f}  -> ratio "
          f"{np.median(ad)/max(np.mean(ns),1e-12):.2f}x")
    ex = [f['real']['n_examined'] for f in out_folds]
    of = [f['real']['n_offered'] for f in out_folds]
    print(f"  candidates EXAMINED before patience quit: {ex} of {of}")
    print("=" * 72)

    path = args.out or os.path.join(
        os.environ.get('KIRBY_SCRATCH', '/tmp'), f'aug_autopsy_{ds}.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'dataset': ds, 'base_rep': args.base_rep,
                   'model': args.model, 'metric': metric_name,
                   'lower_is_better': lower_is_better,
                   'folds': out_folds}, f, indent=2)
    print(f"  -> {path}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
