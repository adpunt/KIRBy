#!/usr/bin/env python3
"""S03: Feature Concentration Analysis -- WHY hybrid similarity works.

Paper: Similarity
Conclusion: Feature importance concentrates label information in selected
dimensions.
Dependencies: none.

For each outer fold, for each of the dataset's two decided models, the hybrids
chosen on that fold's training rows (fold_similarity_selection: the 2 best
hybrids; plus the 2 best at each decided budget, the budget sweep s03 has
always run). For every representation a hybrid draws features from, it measures
how much label information sits in the features the hybrid kept, against all of
the representation's features, a random subset of the same number of features,
and a PCA projection to the same number of components. Plus the Gini
coefficient of each decided importance method's scores per representation.

WHERE EACH NUMBER IS MEASURED (2026-10-05, handoff Part 2 chat 4).
  train   the fold's training rows: the rows the search and the hybrid's
          feature selection were fit on. This is the quantity s03 always
          reported (it used every row before, with the selection fit on every
          row), so it is a concentration diagnostic, not an out-of-sample
          result.
  held_out the fold's held-out rows, with the features, the scaling and the
          PCA all fitted on the training rows. The selection never saw these
          rows.

WHAT THE "MI" NUMBER IS (dirty-run bug, DIRTY_RUN_RESULTS.md "s03"; renamed
2026-10-05). It is NOT the mutual information between the feature block and
the label. Each block is scaled and PCA-reduced to at most 50 components, a
one-feature Kraskov estimate (sklearn mutual_info_regression /
mutual_info_classif) is made for each component, the mean of the same sum under
3 label permutations is subtracted, and the components' estimates are SUMMED.
Information two components share about the label is counted twice, so the sum
can exceed the label's own entropy: the dirty run reached 2.13 nats on a
two-class label, where no true mutual information can exceed ln 2 = 0.693
nats. Every key therefore says 'summed_component_mi', and on a class label the
label's entropy in nats is stored beside it ('label_entropy_nats_*'), the
ceiling a true estimate could not pass. The selected-minus-random difference
(the headline) compares two sums over the same number of components, from the
same representation, on the same rows.
"""
import sys
import os
import zlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.decomposition import PCA

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, compute_effective_rank,
    fold_standard_selection, get_domain_metric, get_model_factories,
)
from experiments.decided_grid import decided_search
from experiments.similarity.similarity_helpers import (
    SIM_FOLD_SEED, sim_outer_folds, fold_spread_note,
    fold_similarity_selection, build_fold_entry, entry_record,
)

from kirby.hybrid import compute_feature_importance
from kirby.scaling import FeatureScaler


# COST / LARGE-N GATE (2026-06-08, RESULTS-AFFECTING, reversible).
#   The per-component Kraskov estimate does a neighbour search whose cost grows
#   with the number of rows. On the 30k-100k-row datasets this dominated s03.
#   A fixed seeded subsample of at most MI_MAX_ROWS of the rows being measured
#   is used. It does not touch any train/held-out split: the subsample is drawn
#   inside the rows already chosen (training or held-out), and the scaling and
#   PCA are fitted on the full training rows before it.
#   REVERT: set MI_MAX_ROWS = None.
MI_MAX_ROWS = 2000
# The component cap of the estimate: at most 50, and at most one component
# per 5 measured rows (the old _shared.estimate_mutual_information_knn rule).
MI_MAX_COMPONENTS = 50
MI_K = 5
MI_N_PERMUTATIONS = 3

# Identity of the partial file. Bumped 2026-10-05: the units are now per outer
# fold and per selected hybrid, and the MI keys were renamed, so a partial
# file written before cannot be read back into this format.
S03_FORMAT = 's03_per_fold_2026_10_05'


def _label_entropy_nats(y):
    """Entropy of a class label in nats: the ceiling of any true I(X; y)."""
    _, counts = np.unique(np.asarray(y), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def _seed_for(*parts):
    """A fixed seed from the unit's name, so a resumed run draws the same
    random subset as an uninterrupted one without carrying an rng state."""
    return zlib.crc32('|'.join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def _summed_component_mi(X_train, y_train, X_eval=None, y_eval=None, *,
                         is_clf, max_rows=MI_MAX_ROWS, k=MI_K, seed=42,
                         reduce=True):
    """Summed one-component Kraskov estimates, permutation-corrected, in nats.

    Scaling (FeatureScaler: 0/1 features untouched, the rest standardised) and
    the PCA are fitted on X_train. The estimate is made on X_eval / y_eval when
    given (held-out rows, clipped to the training range first), else on the
    training rows. Same arithmetic as _shared.estimate_mutual_information_knn
    except that (a) the transforms are fitted on the training rows, not on the
    rows measured, (b) the label is treated as classes from the dataset's
    task_type, not guessed from how many distinct values it has.

    reduce=False skips the PCA (for an input that already is PCA components).

    Returns (value, n_components, n_rows_measured).
    """
    A = np.asarray(X_train, dtype=np.float64)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    sc = FeatureScaler().fit(A)
    A_s = sc.transform(A)
    if X_eval is None:
        B_s, y = A_s, np.asarray(y_train).ravel()
    else:
        B = np.asarray(X_eval, dtype=np.float64)
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        B_s = np.clip(sc.transform(B), A_s.min(axis=0), A_s.max(axis=0))
        y = np.asarray(y_eval).ravel()

    rows = np.arange(len(B_s))
    if max_rows is not None and len(rows) > max_rows:
        rows = np.random.RandomState(seed).choice(len(rows), max_rows,
                                                  replace=False)
    n_rows = len(rows)
    if n_rows < 10:
        return float('nan'), 0, n_rows
    n_comp = max(1, min(MI_MAX_COMPONENTS, n_rows // 5))
    if reduce and A_s.shape[1] > n_comp:
        pca = PCA(n_components=n_comp, random_state=42).fit(A_s)
        Z = pca.transform(B_s[rows])
    else:
        Z = B_s[rows]
    y = y[rows]
    kk = max(1, min(k, n_rows - 1))

    from sklearn.feature_selection import (
        mutual_info_regression, mutual_info_classif)

    def _mi(target):
        if is_clf:
            # class labels as integer codes, whatever their original values
            _, codes = np.unique(target, return_inverse=True)
            return float(np.sum(mutual_info_classif(
                Z, codes, n_neighbors=kk, random_state=42)))
        return float(np.sum(mutual_info_regression(
            Z, target.astype(np.float64), n_neighbors=kk, random_state=42)))

    raw = _mi(y)
    null = np.mean([_mi(np.random.default_rng(1000 + s).permutation(y))
                    for s in range(MI_N_PERMUTATIONS)])
    return float(max(0.0, raw - null)), int(Z.shape[1]), n_rows


def _both_sides(X, tr, va, y, *, is_clf, seed, reduce=True):
    """The summed estimate on the training rows and on the held-out rows."""
    out = {}
    for side, (Xe, ye) in (('train', (None, None)),
                           ('held_out', (X[va], y[va]))):
        try:
            v, nc, nr = _summed_component_mi(
                X[tr], y[tr], Xe, ye, is_clf=is_clf, seed=seed, reduce=reduce)
        except Exception as e:
            v, nc, nr = float('nan'), 0, 0
            out[f'error_{side}'] = f'{type(e).__name__}: {e}'
        out[side] = v
        out[f'n_components_{side}'] = nc
        out[f'n_rows_{side}'] = nr
    return out


def _rep_full_stats(X, tr, va, y, *, is_clf):
    """What does not depend on the hybrid: the whole representation's summed
    estimate on both sides and its effective rank on the training rows.
    Computed once per representation per fold."""
    full = _both_sides(X, tr, va, y, is_clf=is_clf, seed=42)
    return {
        'n_features': int(X.shape[1]),
        'summed_component_mi_full': full,
        'effective_rank_full_train': float(compute_effective_rank(X[tr])),
    }


def _analyse_hybrid(reps, rep_names, labels, tr, va, fi, full_stats, *,
                    is_clf, unit_seed):
    """For every representation the hybrid draws features from: the summed
    estimate of the kept features, of a random subset of the same size, and of
    a PCA projection of the whole representation, on training and held-out
    rows. `fi` is the feature_info of the hybrid rebuilt on this fold's
    training rows; the indices it names are the representation's own columns.
    """
    per_rep = {}
    for rep_name in rep_names:
        X = np.asarray(reps[rep_name])
        n_features = X.shape[1]
        n_train = len(tr)
        rr = {'n_features': int(n_features)}
        _fi_rep = fi.get(rep_name) if isinstance(fi, dict) else None
        sel = (list(_fi_rep.get('selected_indices', []))
               if isinstance(_fi_rep, dict) else [])
        if not sel:
            rr['n_selected'] = 0
            rr['note'] = 'the hybrid kept no feature of this representation'
            per_rep[rep_name] = rr
            continue
        n_sel = len(sel)
        rr['n_selected'] = n_sel
        print(f"\n    {rep_name}: {n_sel} of {n_features} features kept")

        X_sel = X[:, sel]
        s_sel = _both_sides(X_sel, tr, va, labels, is_clf=is_clf, seed=42)
        rr['summed_component_mi_selected'] = s_sel
        rr['effective_rank_selected_train'] = float(
            compute_effective_rank(X_sel[tr]))

        rng = np.random.RandomState(_seed_for(unit_seed, rep_name))
        rand_idx = rng.choice(n_features, n_sel, replace=False)
        X_rand = X[:, rand_idx]
        s_rand = _both_sides(X_rand, tr, va, labels, is_clf=is_clf, seed=42)
        rr['summed_component_mi_random'] = s_rand
        rr['effective_rank_random_train'] = float(
            compute_effective_rank(X_rand[tr]))

        # PCA of the whole representation to the same number of components
        # (capped at the estimate's own component cap, so the estimate does
        # not reduce it again), fitted on the training rows.
        ksg_cap = max(1, min(MI_MAX_COMPONENTS,
                             min(n_train, MI_MAX_ROWS or n_train) // 5))
        n_pca = int(min(n_sel, n_train - 1, ksg_cap))
        try:
            sc = FeatureScaler().fit(np.asarray(X[tr], dtype=np.float64))
            A = sc.transform(np.asarray(X[tr], dtype=np.float64))
            B = np.clip(sc.transform(np.asarray(X[va], dtype=np.float64)),
                        A.min(axis=0), A.max(axis=0))
            if n_features > 2000:
                from sklearn.decomposition import TruncatedSVD
                proj = TruncatedSVD(n_components=n_pca, random_state=42).fit(A)
            else:
                proj = PCA(n_components=n_pca, random_state=42).fit(A)
            X_pca = np.zeros((X.shape[0], n_pca))
            X_pca[tr] = proj.transform(A)
            X_pca[va] = proj.transform(B)
            # The projection is already fitted on the training rows; the
            # estimate re-scales it (fit on training rows again) and does not
            # reduce it further.
            s_pca = _both_sides(X_pca, tr, va, labels, is_clf=is_clf, seed=42,
                                reduce=False)
            rr['effective_rank_pca_train'] = float(
                compute_effective_rank(X_pca[tr]))
        except Exception as e:
            s_pca = {'error': f'{type(e).__name__}: {e}'}
        rr['summed_component_mi_pca'] = s_pca
        rr['n_pca_components'] = n_pca

        full = full_stats[rep_name]['summed_component_mi_full']
        for side in ('train', 'held_out'):
            a, b = s_sel.get(side), s_rand.get(side)
            ok = (a is not None and b is not None
                  and np.isfinite(a) and np.isfinite(b))
            # HEADLINE (owner delegated, 2026-10-05): kept minus random, in
            # nats, same number of components on both sides.
            rr[f'summed_component_mi_selected_minus_random_{side}'] = (
                float(a - b) if ok else float('nan'))
            # Secondary: nats per component, kept over whole representation.
            # NaN when the whole-representation value is 0 or not finite.
            d_full = (full.get(side) / full.get(f'n_components_{side}')
                      if full.get(f'n_components_{side}') else float('nan'))
            d_sel = (a / s_sel.get(f'n_components_{side}')
                     if s_sel.get(f'n_components_{side}') else float('nan'))
            rr[f'summed_component_mi_density_ratio_{side}'] = (
                float(d_sel / d_full)
                if np.isfinite(d_full) and d_full > 0 and np.isfinite(d_sel)
                else float('nan'))
        rr['unit'] = 'nats (summed one-component estimates)'
        print(f"      kept - random: train "
              f"{rr['summed_component_mi_selected_minus_random_train']:.4f}, "
              f"held-out "
              f"{rr['summed_component_mi_selected_minus_random_held_out']:.4f}"
              f" nats")
        per_rep[rep_name] = rr
    return per_rep


def _gini(scores):
    s = np.sort(np.abs(np.asarray(scores, dtype=np.float64)))
    n = len(s)
    if n == 0 or s.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * s) / (n * np.sum(s))) - (n + 1) / n)


def _general_rep_names(dataset_name, reps):
    """The representations s03 measures one by one: representation pruning's
    list for this dataset (DATASET_REPS[ds]['general']), present in the data.
    The hybrid builder still gets the full data['reps']."""
    from experiments.complexity.complexity_helpers import DATASET_REPS
    general = (DATASET_REPS.get(dataset_name) or {}).get('general')
    if not general:
        return list(reps), ('no DATASET_REPS general list for this dataset; '
                            'every loaded representation measured')
    names = [r for r in general if r in reps]
    missing = [r for r in general if r not in reps]
    return names, (f'not loaded: {missing}' if missing else None)


# =============================================================================
# EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Per outer fold: how much label information the hybrid's kept features
    carry against all, random and PCA features of the same representation."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S03: FEATURE CONCENTRATION ANALYSIS")
    print("=" * 70)

    task_type = data['task_type']
    is_clf = task_type == 'classification'
    reps = data['reps']
    labels = np.asarray(data['labels'])
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    grid = decided_search(dataset_name, 'general')
    models = [m for m in grid['models'] if m != 'knn']
    budgets = list(grid['budgets'])
    gini_methods = list(grid['importance_methods'])

    # Models, budgets and importance methods come from the decided grid
    # (DECISIONS.md). run_all still hands s03 'downstream_models' (rf,
    # lightgbm, dnn); it is not used.
    ignored = {}
    for kw in ('downstream_models', 'budgets', 'weight_methods'):
        if kw in kwargs:
            ignored[kw] = kwargs[kw]
            print(f"  s03: ignoring {kw}={kwargs[kw]!r}; the decided grid "
                  f"gives models {models}, budgets {budgets}, importance "
                  f"methods {gini_methods}")

    rep_names, rep_note = _general_rep_names(dataset_name, reps)
    if rep_note:
        print(f"  s03: {rep_note}")

    factories = get_model_factories(models, task_type=task_type,
                                    n_samples=len(labels))
    folds = sim_outer_folds(data, config)
    n_folds = len(folds)

    results = {
        'experiment': 's03_feature_concentration',
        'format': S03_FORMAT,
        'dataset': dataset_name,
        'selection_scope': 'fold_train_rows',
        'n_outer_folds': n_folds,
        'models': models,
        'budgets': budgets,
        'gini_importance_methods': gini_methods,
        'representations_measured': rep_names,
        'representations_note': rep_note,
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'metric_note': ('the metric and its direction belong to the hybrids\' '
                        'training-row search scores (train_rows_score); s03 '
                        'itself measures information in nats'),
        'headline': 'summed_component_mi_selected_minus_random',
        'headline_unit': 'nats (summed one-component estimates)',
        'estimator': (
            'per block: FeatureScaler and PCA (at most 50 components, at most '
            'one per 5 rows) fitted on the fold\'s training rows; per '
            'component a one-feature Kraskov estimate (k=5, sklearn), minus '
            'the mean under 3 label permutations, clamped at 0 and summed over '
            'components. NOT the joint mutual information: shared information '
            'is counted once per component that carries it, so the sum can '
            'exceed the label entropy.'),
        'mi_max_rows': MI_MAX_ROWS,
        'ignored_kwargs': ignored,
        'folds': [],
    }

    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)
    _ck = Checkpoint(
        experiment_sidecar(output_dir, 's03_feature_concentration', dataset_name),
        run_identity('s03_feature_concentration', dataset_name, data, config,
                     format=S03_FORMAT, models=models, budgets=budgets,
                     gini_methods=gini_methods, rep_names=rep_names,
                     n_outer_folds=n_folds, mi_max_rows=MI_MAX_ROWS))

    for fold_i, (tr, va) in enumerate(folds):
        tr, va = np.asarray(tr), np.asarray(va)
        print(f"\n  ── fold {fold_i + 1}/{n_folds}: {len(tr)} training rows, "
              f"{len(va)} held-out rows ──")
        fold_out = {'fold': fold_i, 'n_train_rows': int(len(tr)),
                    'n_held_out_rows': int(len(va)),
                    'selection_scope': 'fold_train_rows',
                    'models': {}, 'unavailable': {}}
        if is_clf:
            fold_out['label_entropy_nats_train'] = _label_entropy_nats(labels[tr])
            fold_out['label_entropy_nats_held_out'] = _label_entropy_nats(labels[va])

        # The fold's standard selection: per decided model, 2 base
        # representations and the 2 best hybrids, from the search on this
        # fold's training rows.
        sel = fold_similarity_selection(
            data, dataset_name, train_idx=tr, fold_i=fold_i,
            n_outer_folds=n_folds, neighbour_spaces=False)
        fold_out['unavailable'].update(sel.get('unavailable') or {})

        # The budget sweep: the 2 best hybrids at each decided budget, from the
        # same per-fold search (fold_searches memoises it; no new search).
        by_budget = {}
        for b in budgets:
            try:
                by_budget[b] = fold_standard_selection(
                    reps, labels, data.get('groups'), train_idx=tr,
                    fold_i=fold_i, n_outer_folds=n_folds,
                    dataset_name=dataset_name,
                    augmentations=data.get('augmentations'),
                    task_type=task_type, seed=SIM_FOLD_SEED, budget=b)
            except Exception as e:
                by_budget[b] = {}
                fold_out['unavailable'][f'budget {b}'] = (
                    f'{type(e).__name__}: {e}')

        # Whole-representation numbers, once per fold.
        if _ck.done(fold=fold_i, unit='rep_full'):
            full_stats = _ck.get(fold=fold_i, unit='rep_full')
        else:
            full_stats = {r: _rep_full_stats(np.asarray(reps[r]), tr, va,
                                             labels, is_clf=is_clf)
                          for r in rep_names}
            _ck.put(full_stats, fold=fold_i, unit='rep_full')
        fold_out['rep_full'] = full_stats

        for m in models:
            s_m = (sel.get('models') or {}).get(m)
            if s_m is None:
                fold_out['models'][m] = None
                fold_out['unavailable'].setdefault(
                    f'general/{m}', 'the per-fold general search failed')
                continue
            # (aug_condition, label) -> entry plus where it was chosen.
            chosen = {}
            for rank, e in enumerate(s_m.get('hybrids') or [], 1):
                key = (e.get('aug_condition') or 'none', e['label'])
                chosen.setdefault(key, {'entry': e, 'slots': []})
                chosen[key]['slots'].append(f'top{rank}')
            for b in budgets:
                s_b = (by_budget.get(b) or {}).get(m)
                if not s_b:
                    continue
                for rank, e in enumerate(s_b.get('hybrids') or [], 1):
                    key = (e.get('aug_condition') or 'none', e['label'])
                    chosen.setdefault(key, {'entry': e, 'slots': []})
                    chosen[key]['slots'].append(f'budget{b}_top{rank}')

            m_out = {'base': list(s_m.get('base') or []),
                     'base_detail': s_m.get('base_detail'),
                     'hybrids_skipped_identical_features': [
                         x.get('label') if isinstance(x, dict) else x
                         for x in (s_m.get('hybrids_skipped_identical_features')
                                   or [])],
                     'hybrids': []}
            for (aug, label), c in chosen.items():
                entry = c['entry']
                ckey = dict(fold=fold_i, model=m, aug=aug, label=label)
                rec = dict(entry_record(entry), model=m, slots=c['slots'],
                           metric=metric_name,
                           lower_is_better=bool(lower_is_better),
                           selection_scope='fold_train_rows')
                if _ck.done(**ckey):
                    saved = _ck.get(**ckey)
                    rec.update(saved)
                    m_out['hybrids'].append(rec)
                    continue
                print(f"\n  -- fold {fold_i} {m} {c['slots']}: {label} --")
                try:
                    _Xtr, _Xva, fi = build_fold_entry(
                        data, entry, train_idx=tr, val_idx=va,
                        model_factory=factories.get(m), metric_fn=metric_fn,
                        lower_is_better=lower_is_better)
                except Exception as e:
                    print(f"  hybrid build failed: {e}")
                    rec['error'] = f'{type(e).__name__}: {e}'
                    m_out['hybrids'].append(rec)
                    continue
                rec['n_features_built'] = int(_Xtr.shape[1])
                rec['per_rep'] = _analyse_hybrid(
                    reps, rep_names, labels, tr, va, fi, full_stats,
                    is_clf=is_clf,
                    unit_seed=_seed_for(dataset_name, fold_i, m, aug, label))
                _ck.put({'n_features_built': rec['n_features_built'],
                         'per_rep': rec['per_rep']}, **ckey)
                m_out['hybrids'].append(rec)
            fold_out['models'][m] = m_out

        # Gini coefficient of each decided importance method's scores per
        # representation, on this fold's training rows.
        if _ck.done(fold=fold_i, unit='gini'):
            fold_out['gini'] = _ck.get(fold=fold_i, unit='gini')
        else:
            gini = {}
            for wm in gini_methods:
                print(f"\n  Gini coefficients ({wm}, fold {fold_i}):")
                try:
                    imp = compute_feature_importance(
                        {r: np.asarray(reps[r])[tr] for r in rep_names},
                        labels[tr], method=wm, task_type=task_type)
                    gini[wm] = {r: _gini(s) for r, s in imp.items()}
                    for r, g in gini[wm].items():
                        print(f"    {r}: Gini={g:.3f}")
                except Exception as e:
                    print(f"    {wm} failed: {e}")
                    gini[wm] = {'error': f'{type(e).__name__}: {e}'}
            fold_out['gini'] = gini
            if not any('error' in v for v in gini.values()):
                _ck.put(gini, fold=fold_i, unit='gini')

        results['folds'].append(fold_out)

    results['summary'] = _summarise(results, models, rep_names)
    return results


def _summarise(results, models, rep_names):
    """Per model, per slot ('top1', 'budget250_top1', ...), per representation:
    the headline on each fold, training and held-out. The recipe in a slot can
    differ between folds; each fold's recipe is in its fold record. Per-fold
    values and their spread only."""
    out = {}
    for m in models:
        for f in results['folds']:
            m_out = (f.get('models') or {}).get(m)
            if not m_out:
                continue
            for h in m_out.get('hybrids') or []:
                for slot in h.get('slots') or []:
                    for r in rep_names:
                        rr = (h.get('per_rep') or {}).get(r) or {}
                        for side in ('train', 'held_out'):
                            v = rr.get(
                                f'summed_component_mi_selected_minus_random_{side}')
                            if v is None:
                                continue
                            d = (out.setdefault(m, {}).setdefault(slot, {})
                                 .setdefault(r, {}).setdefault(
                                     side, {'per_fold': {}}))
                            d['per_fold'][f['fold']] = v
    for m in out.values():
        for slot in m.values():
            for r in slot.values():
                for d in r.values():
                    vals = [v for v in d['per_fold'].values()
                            if v is not None and np.isfinite(v)]
                    note = fold_spread_note(vals)
                    if note:
                        d['spread'] = note
                    else:
                        d['min'] = float(np.min(vals))
                        d['max'] = float(np.max(vals))
                        d['std'] = float(np.std(vals, ddof=1))
    return out


if __name__ == '__main__':
    parser = create_experiment_parser('S03: Feature Concentration Analysis',
                                     has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 's03_feature_concentration', args.dataset)
