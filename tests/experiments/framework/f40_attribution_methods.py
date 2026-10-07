#!/usr/bin/env python3
"""f40 -- three ways of measuring which representations a hybrid's predictions
rely on, compared on the same fitted hybrids (Adelaide, 2026-10-07).

A framework experiment (a method bake-off). Debug runs only: what runs, how
long it takes, and whether the methods agree. Nothing here is a result.

Per dataset, decided model, hybrid and outer fold, on the held-out molecules
of that fold, with the model fitted on the fold's training rows. Every
explainer is c22's own function, at c22 commit 8e515d0:

  1. Grouped permutation (shuffle). c22._grouped_permutation_drop: each
     representation's features shuffled together across held-out molecules,
     5 repeats, drop in the dataset's own metric.
     share = drop / sum of POSITIVE drops (can be negative).
  2. SHAP as c22 runs it per model. c22._shap_mean_abs:
       xgboost, lightgbm, rf, catboost -> TreeExplainer, every held-out molecule;
       svm, gp -> KernelExplainer on c22's 50 fixed held-out molecules
         (c22.kernel_explained_rows), background shap.kmeans of the training
         rows with 10 centres, nsamples min(2 * features + 2048, 4096),
         l1_reg off.
  3. Gradient-based SHAP for dnn, two ways, both on every held-out molecule
     and the same 100 training molecules as background
     (c22.deep_background_rows):
       DeepExplainer, through c22._shap_mean_abs (what c22 runs);
       GradientExplainer, written here (Adelaide's choice, 2026-10-07).
     share = mean |SHAP| summed per representation / total, for 2 and 3
     (aggregate_importance_by_rep).

The permutation share and the SHAP share are DIFFERENT QUANTITIES. The first
is score lost in the metric when a representation is scrambled; the second is
how far the prediction moves, summed over a representation's features. Only
their ranking of the representations is compared: the Spearman correlation
over the representations in the hybrid, and whether they name the same top
representation.

KernelExplainer time-box. c22._shap_mean_abs is called on a few held-out
molecules at a time (c22.kernel_explained_rows patched to "every row given"
for the call; the background is KMeans with random_state 0, so it is the same
10 centres every call). c22's 50 molecules come first. Each hybrid stops at
its time cap and records how many of the 50 it reached. The cap is the time
left divided evenly over the KernelExplainer hybrids still to come, at most
KERNEL_CAP_S.

--kernel-check (one small dataset): on fold 0, hybrid1 of each svm or gp
model, KernelExplainer continues past c22's 50 molecules to every held-out
molecule, capped at KERNEL_CHECK_CAP_S. The share from c22's 50 is compared
with the share from every molecule reached.

Hybrids: fold_standard_selection on each fold's training rows, the dataset's
two decided models, 2 hybrids per model, rebuilt with build_fold_hybrid and
fitted with c22._fit_score -- the same hybrids c22 explains.

Resumable: each finished fold is written to
tests/results/framework/f40_attribution_methods/<dataset>/fold<k>.json and
skipped on a restart of the same command. The per-fold searches are cached by
find_best_hybrid as they run.

Usage (one dataset per process):
  python tests/experiments/framework/f40_attribution_methods.py --dataset herg_ki
  python tests/experiments/framework/f40_attribution_methods.py --summarise
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..', '..', '..', 'src'),
           os.path.join(_HERE, '..', '..'), os.path.join(_HERE, '..')):
    sys.path.insert(0, _p)

OUT = os.path.join(_HERE, '..', '..', 'results', 'framework',
                   'f40_attribution_methods')
SEED = 42
N_FOLDS = 5
N_HYBRIDS = 2                # per model per fold (fold_standard_selection)
KERNEL_CAP_S = 1800          # most seconds one hybrid's KernelExplainer gets
KERNEL_MIN_CAP_S = 300
KERNEL_CHECK_CAP_S = 3600    # fold 0, hybrid1, with --kernel-check
KERNEL_CHUNK = 5             # held-out molecules per c22._shap_mean_abs call
STOP_MARGIN_S = 600          # no new fold starts with less than this left


def _now():
    return time.strftime('%H:%M:%S')


def _heartbeat(t_start, every_s=180):
    """A line every few minutes so an srun terminal is never silent while
    a per-fold search runs (that search prints nothing on its own)."""
    import threading

    def beat():
        while True:
            time.sleep(every_s)
            print(f"  [{_now()}] still running, "
                  f"{(time.time() - t_start) / 60:.0f} min since start",
                  flush=True)
    threading.Thread(target=beat, daemon=True).start()


def _gradient_mean_abs(model, X_train_s, X_val_s, fold):
    """Mean |SHAP| per feature from shap.GradientExplainer on dnn's network,
    every held-out molecule, background = c22.deep_background_rows (the rows
    c22's DeepExplainer uses). Returns (mean_abs or None, error or None)."""
    from experiments.complexity import c22_explainability as c22
    # c22._shap_mean_abs's sys.modules lift (see the comment there): shap
    # probes transformers.TFPreTrainedModel, which imports TensorFlow, which
    # dies on this machine.
    lifted = {k: v for k, v in list(sys.modules.items())
              if k == 'transformers' or k.startswith('transformers.')}
    for k in lifted:
        del sys.modules[k]
    try:
        import shap
        import torch
        net = model.model_
        net.eval()
        bg_idx = c22.deep_background_rows(fold, len(X_train_s))
        bg = torch.from_numpy(np.asarray(X_train_s, dtype=np.float32)[bg_idx])
        xv = torch.from_numpy(np.asarray(X_val_s, dtype=np.float32))
        sv = shap.GradientExplainer(net, bg).shap_values(xv, rseed=SEED)
        out = c22._reduce_shap_to_per_feature(sv, X_val_s.shape[1])
        if out is None:
            return None, f'SHAP shape {np.shape(sv)} not aligned'
        return out, None
    except Exception as e:
        return None, f'{type(e).__name__}: {e}'
    finally:
        sys.modules.update(lifted)


def _kernel_timeboxed(m, model, X_train_s, X_val_s, fold, n_target, cap_s,
                      label):
    """c22._shap_mean_abs (KernelExplainer, c22's settings) a few held-out
    molecules per call. c22's 50 fixed molecules first, then the rest in a
    seeded order, up to n_target molecules or cap_s seconds.

    Returns the mean |SHAP| per feature over c22's 50 (or over as many of
    them as were reached) and over every molecule reached."""
    from experiments.complexity import c22_explainability as c22
    n = len(X_val_s)
    first = list(c22.kernel_explained_rows(fold, n))
    rest_set = set(range(n)) - set(first)
    rest = [i for i in np.random.default_rng(SEED).permutation(n)
            if i in rest_set]
    order = (first + rest)[:n_target]
    total = np.zeros(X_val_s.shape[1])
    at_first = None
    done, err, t0, last = 0, None, time.time(), time.time()
    orig = c22.kernel_explained_rows
    c22.kernel_explained_rows = lambda fold, n_val, n=None: np.arange(n_val)
    try:
        for s in range(0, len(order), KERNEL_CHUNK):
            if time.time() - t0 >= cap_s:
                break
            chunk = np.asarray(order[s:s + KERNEL_CHUNK])
            r = c22._shap_mean_abs(m, model, X_train_s, X_val_s[chunk],
                                   fold=fold)
            if r['shap_mean_abs'] is None:
                err = r['shap_error']
                break
            total += r['shap_mean_abs'] * len(chunk)
            done += len(chunk)
            if done == len(first):
                at_first = total / done
            if time.time() - last > 120:
                last = time.time()
                print(f"  [{_now()}] {label} KernelExplainer {done}/"
                      f"{len(order)} molecules, "
                      f"{(last - t0) / done:.1f} s per molecule", flush=True)
    finally:
        c22.kernel_explained_rows = orig
    secs = time.time() - t0
    if at_first is None and done and done < len(first):
        at_first = total / done
    return {'mean_abs_c22_50': at_first,
            'mean_abs_reached': total / done if done else None,
            'n_explained': done, 'n_target': len(order),
            'n_c22_rows': len(first), 'n_val': n,
            'cap_s': cap_s, 'reached_target': done == len(order),
            'seconds': secs,
            'seconds_per_molecule': secs / done if done else None,
            'error': err}


def _shares(fi, mean_abs, blocks):
    from kirby.importance.hybrid import aggregate_importance_by_rep
    if mean_abs is None:
        return None
    return {r: v for r, v in aggregate_importance_by_rep(fi, mean_abs).items()
            if r in blocks}


def _ranking_agreement(a, b, reps):
    """Spearman over `reps` and whether the top representation is the same."""
    from scipy.stats import spearmanr
    if not a or not b:
        return {'spearman': None, 'same_top': None, 'n_reps': len(reps)}
    va = [a.get(r) for r in reps]
    vb = [b.get(r) for r in reps]
    if any(x is None for x in va + vb):
        return {'spearman': None, 'same_top': None, 'n_reps': len(reps),
                'note': 'a share is missing (no positive permutation drop)'}
    top_a = max(reps, key=lambda r: a[r])
    top_b = max(reps, key=lambda r: b[r])
    out = {'top_a': top_a, 'top_b': top_b, 'same_top': top_a == top_b,
           'n_reps': len(reps)}
    if len(reps) < 3:
        out['spearman'] = None
        out['note'] = f'{len(reps)} representations, Spearman not computed'
    else:
        rho = spearmanr(va, vb).correlation
        out['spearman'] = None if not np.isfinite(rho) else float(rho)
    return out


def _run_fold(k, tr, va, *, data, dataset, models, factories, metric_name,
              metric_fn, lower_is_better, kernel_check, deadline, n_folds_left):
    from experiments._shared import (
        fold_standard_selection, build_fold_hybrid, fold_hybrid_scoring_kwargs)
    from experiments.complexity import c22_explainability as c22

    reps, labels = data['reps'], np.asarray(data['labels'])
    groups, augs = data.get('groups'), data.get('augmentations')
    y_tr, y_va = labels[tr], labels[va]
    g_tr = np.asarray(groups)[tr] if groups is not None else None
    rec = {'fold': k, 'n_train': len(tr), 'n_val': len(va), 'models': {}}

    t = time.time()
    sel = fold_standard_selection(
        reps, labels, groups, train_idx=tr, fold_i=k, n_outer_folds=N_FOLDS,
        dataset_name=dataset, models=list(models), augmentations=augs,
        task_type=data['task_type'], metric=metric_name, seed=SEED)
    rec['search_seconds'] = time.time() - t
    print(f"  [{_now()}] fold {k}: search and selection "
          f"{rec['search_seconds']:.0f} s", flush=True)

    tr_reps = {r: X[tr] for r, X in reps.items()}
    va_reps = {r: X[va] for r, X in reps.items()}
    tr_augs = {a: X[tr] for a, X in augs.items()} if augs else None
    va_augs = {a: X[va] for a, X in augs.items()} if augs else None

    n_kernel_models = sum(m in c22.KERNEL_MODELS for m in models)
    kernel_left = n_kernel_models * N_HYBRIDS * n_folds_left

    for m in models:
        s = sel.get(m)
        if s is None:
            rec['models'][m] = {'error': 'search failed'}
            continue
        factory = factories[m]
        rows = []
        for i, e in enumerate(s['hybrids'], 1):
            label = f'fold {k} {m} hybrid{i}'
            bp = e['blueprint']
            row = {'slot': f'hybrid{i}',
                   **{f: e.get(f) for f in c22.HYBRID_FIELDS}}
            try:
                t = time.time()
                X_tr_h, X_va_h, fi = build_fold_hybrid(
                    tr_reps, va_reps, y_tr, bp,
                    augmentations_train=tr_augs, augmentations_val=va_augs,
                    groups=g_tr,
                    precomputed_alloc_scores=getattr(bp, 'rep_scores', None),
                    **fold_hybrid_scoring_kwargs(
                        bp, model=factory(), metric_fn=metric_fn,
                        lower_is_better=lower_is_better))
                sc, fitted, X_tr_s, X_va_s = c22._fit_score(
                    X_tr_h, X_va_h, y_tr, y_va, factory, metric_fn)
                row['build_fit_seconds'] = time.time() - t
            except Exception as ex:
                row['error'] = f'{type(ex).__name__}: {ex}'
                print(f"  [{_now()}] {label} build/fit failed: {row['error']}")
                rows.append(row)
                if m in c22.KERNEL_MODELS:
                    kernel_left -= 1
                continue
            row['heldout_score'] = sc
            row['n_features'] = int(X_tr_s.shape[1])
            blocks = c22.rep_feature_blocks(fi)
            row['features_by_rep'] = {r: b - a for r, (a, b) in blocks.items()}
            rep_list = list(blocks)

            # 1. grouped permutation
            t = time.time()
            try:
                p = c22._grouped_permutation_drop(
                    fitted, X_va_s, y_va, fi, metric_fn, lower_is_better,
                    fold=k)
                row['perm_drop_by_rep'] = p['perm_drop_by_rep']
                row['perm_share_by_rep'] = p['perm_share_by_rep']
                row['perm_baseline_score'] = p['perm_baseline_score']
            except Exception as ex:
                row['perm_error'] = f'{type(ex).__name__}: {ex}'
            row['perm_seconds'] = time.time() - t

            # 2/3. SHAP as c22 runs it for this model
            t = time.time()
            if m in c22.KERNEL_MODELS:
                check = kernel_check and k == 0 and i == 1
                even = (deadline - time.time() - STOP_MARGIN_S) / max(
                    kernel_left, 1)
                cap = (KERNEL_CHECK_CAP_S if check else
                       max(KERNEL_MIN_CAP_S, min(KERNEL_CAP_S, even)))
                kr = _kernel_timeboxed(
                    m, fitted, X_tr_s, X_va_s, k,
                    len(X_va_s) if check else c22.KERNEL_N_EXPLAINED,
                    cap, label)
                kernel_left -= 1
                row['shap_method'] = 'KernelExplainer'
                row['shap_n_explained'] = min(kr['n_explained'],
                                              kr['n_c22_rows'])
                row['shap_n_target'] = kr['n_c22_rows']
                row['kernel'] = {x: kr[x] for x in
                                 ('n_explained', 'n_target', 'n_val', 'cap_s',
                                  'reached_target', 'seconds',
                                  'seconds_per_molecule', 'error')}
                row['shap_share_by_rep'] = _shares(
                    fi, kr['mean_abs_c22_50'], blocks)
                if check:
                    row['kernel_check_share_reached_by_rep'] = _shares(
                        fi, kr['mean_abs_reached'], blocks)
                    row['agree_kernel_c22_50_vs_reached'] = _ranking_agreement(
                        row['shap_share_by_rep'],
                        row['kernel_check_share_reached_by_rep'], rep_list)
                if kr['error']:
                    row['shap_error'] = kr['error']
            else:
                r = c22._shap_mean_abs(m, fitted, X_tr_s, X_va_s, fold=k)
                row['shap_method'] = r['shap_method']
                row['shap_n_explained'] = r['shap_n_explained']
                row['shap_n_target'] = len(X_va_s)
                row['shap_share_by_rep'] = _shares(fi, r['shap_mean_abs'],
                                                   blocks)
                if r['shap_error']:
                    row['shap_error'] = r['shap_error']
            row['shap_seconds'] = time.time() - t
            row['agree_perm_vs_shap'] = _ranking_agreement(
                row.get('perm_share_by_rep'), row.get('shap_share_by_rep'),
                rep_list)

            if m in c22.GRADIENT_MODELS:
                t = time.time()
                v, err = _gradient_mean_abs(fitted, X_tr_s, X_va_s, k)
                row['gradient_seconds'] = time.time() - t
                row['gradient_share_by_rep'] = _shares(fi, v, blocks)
                if err:
                    row['gradient_error'] = err
                row['agree_perm_vs_gradient'] = _ranking_agreement(
                    row.get('perm_share_by_rep'),
                    row['gradient_share_by_rep'], rep_list)
                row['agree_deep_vs_gradient'] = _ranking_agreement(
                    row.get('shap_share_by_rep'),
                    row['gradient_share_by_rep'], rep_list)

            a = row['agree_perm_vs_shap']
            extra = (f", GradientExplainer {row['gradient_seconds']:.1f} s"
                     if 'gradient_seconds' in row else '')
            print(f"  [{_now()}] {label}: {row['n_features']} features, "
                  f"{len(rep_list)} representations, {metric_name}={sc:.4f}; "
                  f"permutation {row['perm_seconds']:.1f} s, "
                  f"{row['shap_method']} {row['shap_seconds']:.1f} s "
                  f"({row.get('shap_n_explained')}/{row['shap_n_target']} "
                  f"molecules){extra}; permutation and SHAP same top "
                  f"representation: {a.get('same_top')}, Spearman "
                  f"{a.get('spearman')}"
                  + (f"; SHAP error {row['shap_error']}"
                     if row.get('shap_error') else ''), flush=True)
            rows.append(row)
        rec['models'][m] = {'hybrids': rows}
    return rec


def run(dataset, hours, kernel_check):
    from experiments._shared import (
        load_data, get_model_factories, get_domain_metric, convert_numpy)
    from experiments.decided_grid import decided_models
    from kirby.find_best_hybrid import nested_cv_outer_folds
    from kirby.paths import cache_root

    t_start = time.time()
    deadline = t_start + hours * 3600
    _heartbeat(t_start)
    out_dir = os.path.join(OUT, dataset)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{_now()}] f40 {dataset}: loading data", flush=True)
    data = load_data(dataset, 'all')
    labels = np.asarray(data['labels'])
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    models = [m for m in decided_models(dataset, 'general') if m != 'knn']
    factories = get_model_factories(models, task_type=data['task_type'],
                                    n_samples=len(labels))
    folds = nested_cv_outer_folds(data['reps'], labels, data.get('groups'),
                                  N_FOLDS, SEED, data['task_type'])
    print(f"[{_now()}] {len(labels)} molecules, {len(data['reps'])} "
          f"representations, models {models}, metric {metric_name}, "
          f"{len(folds)} folds, load {time.time() - t_start:.0f} s", flush=True)
    for k in range(len(folds)):
        db = (cache_root() / 'result_db'
              / f'{dataset}__ocv{N_FOLDS}f{k}s{SEED}.db')
        print(f"  fold {k} stored search: "
              f"{'found' if db.exists() else 'NOT found, will run'} ({db})")

    for k, (tr, va) in enumerate(folds):
        path = os.path.join(out_dir, f'fold{k}.json')
        if os.path.exists(path):
            print(f"[{_now()}] fold {k}: finished earlier, skipped")
            continue
        left = deadline - time.time()
        if left < STOP_MARGIN_S:
            print(f"[{_now()}] fold {k}: {left / 60:.0f} min left of "
                  f"{hours} h, not started")
            break
        print(f"\n[{_now()}] fold {k}: {len(tr)} training molecules, "
              f"{len(va)} held out, {left / 60:.0f} min left", flush=True)
        t = time.time()
        rec = _run_fold(k, tr, va, data=data, dataset=dataset, models=models,
                        factories=factories, metric_name=metric_name,
                        metric_fn=metric_fn, lower_is_better=lower_is_better,
                        kernel_check=kernel_check, deadline=deadline,
                        n_folds_left=len(folds) - k)
        rec.update(dataset=dataset, metric=metric_name,
                   fold_seconds=time.time() - t)
        with open(path, 'w') as f:
            json.dump(convert_numpy(rec), f, indent=1)
        print(f"[{_now()}] fold {k} written, {rec['fold_seconds']:.0f} s",
              flush=True)
    summarise()


def _fmt(vals, f='{:.2f}'):
    vals = [v for v in vals if v is not None]
    if not vals:
        return '-'
    if len(vals) == 1:
        return f.format(vals[0])
    return (f'{f.format(np.median(vals))} '
            f'({f.format(min(vals))} to {f.format(max(vals))})')


def _agree_cells(ag):
    st = [a['same_top'] for a in ag if a and a.get('same_top') is not None]
    return (f"{sum(st)}/{len(st)}" if st else '-',
            _fmt([a.get('spearman') for a in ag if a]))


def summarise():
    """One row per (dataset, model, comparison), over every hybrid and fold
    written so far."""
    head = ['One row = one dataset, model and pair of methods, over every '
            'hybrid on every fold written so far.',
            'Same top = hybrids where both methods name the same top '
            'representation, out of hybrids where both produced shares.',
            'Spearman = between the two methods\' per-representation shares, '
            'median (min to max) over hybrids.',
            'Seconds = per hybrid per fold, median (min to max), for the '
            'first method then the second.',
            'Molecules = held-out molecules the SHAP method explained, out of '
            'its target, summed over hybrids.', '',
            '| dataset | model | folds | hybrids | methods compared | same top '
            '| Spearman | seconds, first | seconds, second | molecules |',
            '|' + '---|' * 10]
    lines = list(head)
    if not os.path.isdir(OUT):
        print('nothing written')
        return
    for ds in sorted(os.listdir(OUT)):
        d = os.path.join(OUT, ds)
        if not os.path.isdir(d):
            continue
        recs = [json.load(open(os.path.join(d, f)))
                for f in sorted(os.listdir(d))
                if f.startswith('fold') and f.endswith('.json')]
        for m in sorted({m for r in recs for m in r['models']}):
            rows = [h for r in recs for h in
                    (r['models'].get(m) or {}).get('hybrids', [])
                    if 'error' not in h]
            if not rows:
                continue
            meth = rows[0].get('shap_method') or 'SHAP'
            mol = (f"{sum(h.get('shap_n_explained') or 0 for h in rows)}/"
                   f"{sum(h.get('shap_n_target') or 0 for h in rows)}")
            comps = [('permutation vs ' + meth, 'agree_perm_vs_shap',
                      'perm_seconds', 'shap_seconds', mol)]
            if any('agree_perm_vs_gradient' in h for h in rows):
                comps += [('permutation vs GradientExplainer',
                           'agree_perm_vs_gradient', 'perm_seconds',
                           'gradient_seconds', '-'),
                          (meth + ' vs GradientExplainer',
                           'agree_deep_vs_gradient', 'shap_seconds',
                           'gradient_seconds', '-')]
            chk = [h for h in rows if 'agree_kernel_c22_50_vs_reached' in h]
            if chk:
                comps.append((
                    'KernelExplainer c22\'s 50 vs every molecule reached',
                    'agree_kernel_c22_50_vs_reached', None, 'shap_seconds',
                    ', '.join(f"{h['kernel']['n_explained']}/"
                              f"{h['kernel']['n_val']}" for h in chk)))
            for name, key, s1, s2, molc in comps:
                use = chk if key == 'agree_kernel_c22_50_vs_reached' else rows
                top, rho = _agree_cells([h.get(key) for h in use])
                lines.append(
                    f"| {ds} | {m} | {len(recs)} | {len(use)} | {name} | "
                    f"{top} | {rho} | "
                    f"{_fmt([h.get(s1) for h in use], '{:.1f}') if s1 else '-'}"
                    f" | {_fmt([h.get(s2) for h in use], '{:.1f}')} | {molc} |")
    txt = '\n'.join(lines)
    with open(os.path.join(OUT, 'summary.md'), 'w') as f:
        f.write(txt + '\n')
    print('\n' + txt)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset')
    ap.add_argument('--hours', type=float, default=3.75,
                    help='wall time; no new fold starts with <10 min left')
    ap.add_argument('--kernel-check', action='store_true',
                    help='fold 0 hybrid1 of svm/gp: KernelExplainer on every '
                         'held-out molecule as well as c22\'s 50')
    ap.add_argument('--folds', type=int, default=N_FOLDS,
                    help='outer folds in the layout; 5 for every real run')
    ap.add_argument('--out', default=OUT,
                    help='output directory (local checks write elsewhere)')
    ap.add_argument('--summarise', action='store_true')
    a = ap.parse_args()
    N_FOLDS, OUT = a.folds, a.out
    if a.summarise:
        summarise()
    else:
        run(a.dataset, a.hours, a.kernel_check)
