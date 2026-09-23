#!/usr/bin/env python3
"""n21's whole table, at more folds and more corruption draws.

THE QUESTION, AND WHY IT IS THE HONEST NEXT STEP. n21 ranks fifteen distances
by detection average precision. Every reading of that table on
flab_thermo_tresanco so far stands on 2 folds and 1 corruption draw, which with
two detectors and one neighbour count is 4 paired readings per comparison.
Exactly one comparison in it had a mean gain larger than the spread of the
per-cell gains behind it. Every other ordering in the table -- including which
distance is first -- sits inside its own noise. This raises the folds from 2 to
5, the draws from 1 to 3, and the neighbour counts from 1 to all three n21
sweeps, which is 90 paired readings per comparison instead of 4, and asks
whether the top of the table separates.

WHAT IT CUTS TO PAY FOR THAT, AND NOTHING ELSE. The repair. n21's second
reading fits a downstream model once per space, detector, neighbour count,
draw and fold; on this dataset's control representation one fit is about 8
seconds and the fits are the whole cost of the file. The detection figure costs
no fit at all. Nothing about the distances is cut: all five families, both
metrics, both importance scorers, both exponents and all three neighbour counts
run. A distance that wins here is re-measured through n21 itself, with the
repair, before it is believed.

⛔ THE CONTROL CHANGED UNDER THIS FILE. `pick_noise_base_rep` no longer falls
back alphabetically, so flab_thermo_tresanco's plain representation is `onehot`
at 5600 features, not `esm2_cdr` at 1280. The recorded table -- forest
proximity 0.2455, importance-weighted 0.2388, similarity hybrid 0.2182, plain
0.2101, prediction hybrid 0.2045 -- was read in esm2_cdr. Those numbers are on
a different footing and this run does not compare against them; it re-reads
every row in the representation the control is now.

⛔ ROWS ARE NEVER CUT. All 672 antibodies, every fold. KIRBY_SMOKE_N is
refused.

IT PICKS UP WHERE IT WAS KILLED. One pass -- one fold under one corruption
draw -- is about 11 minutes here, and the machine went down on 2026-09-21 with
8 of the 15 finished. The result file is written after every pass now, not
after every fold, and a run that starts with one on disk scores only the
passes missing from it. A pass is taken from disk only when it was read in the
same representation, at the same level and condition, over the same distances,
detectors and neighbour counts; if any of those differ the file is ignored and
every pass runs again. N21_FORCE=1 starts from nothing.

THE SPACES ARE n21's OWN. `space_specs` and `_spaces_for_pass` are imported and
called, not copied, so the fifteen distances here are the fifteen distances n21
scores and cannot drift from them.

WHAT IT DOES NOT TOUCH. No cache file is deleted, moved or renamed. The stored
search is opened read-only, once, for the two hybrid recipes; if it holds
neither, the run says so and scores the other eleven distances.
"""
import os

for _k, _v in {
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1', 'KMP_DUPLICATE_LIB_OK': 'TRUE',
        'KIRBY_LGBM_THREADS': '1', 'KIRBY_RF_JOBS': '1',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'KIRBY_CACHE_DIR': '/Volumes/seagate/kirby_cache',
}.items():
    os.environ.setdefault(_k, _v)

if os.environ.get('KIRBY_SMOKE_N', '').strip():
    raise SystemExit(
        "KIRBY_SMOKE_N is set. It is a ROW CAP inside load_data, not a speed "
        "flag, and a detector graded against a corrupted subset has nothing to "
        "find once the rows are cut. Unset it.")

import json
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'tests'))
sys.path.insert(0, os.path.join(REPO, 'tests', 'experiments'))
sys.path.insert(0, os.path.join(REPO, 'src'))

import numpy as np
from sklearn.preprocessing import StandardScaler

DATASET = 'flab_thermo_tresanco'
CONDITION = 'outlier_p10'
LEVEL = 0.5
FOLDS = int(os.environ.get('N21_FOLDS', '5'))
DRAWS = tuple(int(x) for x in os.environ.get('N21_DRAWS', '0,1,2').split(','))
DOWNSTREAM_MODEL = 'rf'     # only used to fetch the recipe and test a build
OUT = os.environ.get('N21_OUT', '/Volumes/seagate/local_noise_run/loop_n21')


def _claim(out_dir, name):
    """Refuse to start while another copy of this script is already running.

    The supervisor tracks a job by a pid file it writes itself, so a run
    started by hand is invisible to it and it would start a second one. This
    is the lock both can see: the pid of whoever is running, in the output
    directory, removed on the way out. A stale file whose pid is gone is
    ignored.
    """
    import atexit
    os.makedirs(out_dir, exist_ok=True)
    lock = os.path.join(out_dir, f'.{name}.running')
    if os.path.exists(lock):
        try:
            other = int(open(lock).read().strip())
            os.kill(other, 0)
            print(f"already running as pid {other}; this copy exits")
            return False
        except (ValueError, ProcessLookupError, PermissionError, OSError):
            pass
    with open(lock, 'w') as fh:
        fh.write(str(os.getpid()))

    def _release():
        try:
            if int(open(lock).read().strip()) == os.getpid():
                os.remove(lock)
        except Exception:
            pass
    atexit.register(_release)
    return True


def _restore(prior, ctx):
    """Fill the tables from a result file this run wrote earlier.

    Returns the set of (fold, draw) passes that are already on disk and must
    not be run again, and a record of what was recovered for the result file.
    A pass is trusted only when the run that wrote it scored the same spaces,
    in the same representation, at the same level and condition, with the same
    detectors and the same neighbour counts. If any of those differ the whole
    file is ignored and every pass runs again, because a table half-read in
    one representation and half in another is not a table.
    """
    note = {'passes_from_disk': [], 'ignored_because': None}
    if not prior:
        return set(), note
    same = []
    for field, here in (('plain_representation', ctx['base_rep']),
                        ('noise_condition', ctx['CONDITION']),
                        ('level', ctx['LEVEL']),
                        ('detectors', ctx['detectors']),
                        ('k_values', ctx['k_values']),
                        # n21's forest constants are read at import, so a run
                        # restarted after one of them moves would put passes
                        # from two different forests in one column.
                        ('forest', dict(ctx['forest_kw']))):
        there = prior.get(field)
        if there != here:
            same.append(f'{field} was {there!r}, is {here!r}')
    if set(prior.get('spaces') or {}) != set(ctx['specs']):
        same.append('the set of distances differs')
    if same:
        note['ignored_because'] = '; '.join(same)
        print(f"  the result file on disk is not comparable "
              f"({note['ignored_because']}), so every pass runs again")
        return set(), note

    cells = prior.get('average_precision_per_cell') or {}
    pk_cells = prior.get('precision_at_k_per_cell') or {}
    done = set(prior.get('seconds_per_pass') or {})
    for space_key in ctx['specs']:
        for cell, v in (cells.get(space_key) or {}).items():
            if cell.split('|')[-1] in done:
                ctx['ap'][space_key][cell] = v
        for cell, v in (pk_cells.get(space_key) or {}).items():
            if cell.split('|')[-1] in done:
                ctx['pk'][space_key][cell] = v
    for pass_key, secs in (prior.get('seconds_per_pass') or {}).items():
        ctx['timings'][pass_key] = secs
    for pass_key, n in (prior.get('molecules_corrupted_per_pass')
                        or {}).items():
        ctx['counts']['n_corrupted'][pass_key] = n
    note['passes_from_disk'] = sorted(done)
    note['precision_at_k_recovered'] = bool(pk_cells)
    if done:
        print(f"  {len(done)} pass(es) of (fold, draw) are already on disk "
              f"and are not re-run: {', '.join(sorted(done))}")
        if not pk_cells:
            print("  that file stored precision at k only as a mean per "
                  "distance, so the precision-at-k column here covers the "
                  "passes this run computes, not those")
    return done, note


def main():
    from experiments._shared import (load_data, pick_noise_base_rep, cv_split,
                                     get_domain_metric, get_model_factories,
                                     get_best_search_for_downstream)
    from experiments.noise import n21_neighbour_distance as N21
    from experiments.noise.noise_helpers import (
        inject_train_labels, noise_spec_for, detection_scores, hybrid_for_fold)
    from experiments.dataset_policy import noise_groups_for

    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, f'{DATASET}__folds_and_draws.json')
    if not _claim(OUT, 'folds_and_draws'):
        return
    prior = {}
    if os.path.exists(out_path) and not os.environ.get('N21_FORCE'):
        try:
            with open(out_path) as fh:
                prior = json.load(fh)
        except Exception as exc:
            print(f"  the result file on disk could not be read "
                  f"({type(exc).__name__}), so this run starts from nothing")

    t_start = time.time()
    data = load_data(DATASET)
    labels = np.asarray(data['labels'])
    reps = data['reps']
    groups = data.get('groups')
    augmentations = data.get('augmentations')
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    base_rep = pick_noise_base_rep(reps, DATASET)
    noise_groups, _n, _l = noise_groups_for(data)
    splits = list(cv_split(groups, FOLDS, next(iter(reps.values())), labels))

    detectors = list(N21.DETECTORS)
    k_values = list(N21.K_VALUES)
    metrics = list(N21.DISTANCE_METRICS_SWEPT)
    importance_methods = list(N21.IMPORTANCE_METHODS)
    exponents = [float(e) for e in N21.IMPORTANCE_EXPONENTS]
    forest_kw = {'n_trees': N21.FOREST_TREES,
                 'min_samples_leaf': N21.FOREST_MIN_SAMPLES_LEAF,
                 'seed': N21.FOREST_SEED}

    # ── the two hybrid recipes, read once from the stored search ───────────
    bp = bp_sim = None
    hybrid_meta = {'present': False}
    try:
        search, meta = get_best_search_for_downstream(
            reps, labels, groups, downstream_model=DOWNSTREAM_MODEL,
            budget=None, augmentations=augmentations, dataset_name=DATASET,
            task_type=task_type, data=data,
            pool_subset='predictive_and_similarity')
        bp = meta['best_config']['blueprint'] if meta else None
        _sim = ((meta.get('by_objective') or {}).get('similarity') or {}) \
            if meta else {}
        bp_sim = _sim.get('best_config', {}).get('blueprint')
        hybrid_meta = {'present': bp is not None,
                       'similarity_present': bp_sim is not None,
                       'family': (meta or {}).get('objective'),
                       'pool_metric': (meta or {}).get('metric_name')}
    except Exception as exc:
        hybrid_meta['absent_because'] = f'{type(exc).__name__}: {exc}'
        print(f"  no hybrid recipe: {hybrid_meta['absent_because']}",
              flush=True)

    specs = N21.space_specs(importance_methods, metrics, exponents,
                            include_hybrid=bp is not None)
    model_factories = get_model_factories([DOWNSTREAM_MODEL],
                                          task_type=task_type,
                                          n_samples=len(labels))

    n_passes = len(splits) * len(DRAWS)
    print(f"n21 folds and draws on {DATASET}: {len(labels)} antibodies, "
          f"{len(splits)} folds, draws {list(DRAWS)}, level {LEVEL}, "
          f"condition {CONDITION}")
    print(f"  plain representation: {base_rep}, "
          f"{reps[base_rep].shape[1]} features")
    print(f"  dataset metric {metric_name}; the reading here is detection "
          f"average precision, which no model fit produces")
    print(f"  {len(specs)} distances, detectors {detectors}, "
          f"neighbour counts {k_values}")
    print(f"  one cell is one (fold, draw, detector, neighbour count): "
          f"{n_passes} x {len(detectors)} x {len(k_values)} = "
          f"{n_passes * len(detectors) * len(k_values)} cells per distance",
          flush=True)

    ap = {s: {} for s in specs}
    pk = {s: {} for s in specs}
    counts = {'n_corrupted': {}, 'n_training': {}}
    errors = {}
    zero_imp = {}
    timings = {}
    # Everything _write needs, in one place, because the module-level
    # constants are not in a function's locals().
    ctx = {
        'ap': ap, 'pk': pk, 'specs': specs, 'counts': counts,
        'errors': errors, 'zero_imp': zero_imp, 'timings': timings,
        'DATASET': DATASET, 'CONDITION': CONDITION, 'LEVEL': LEVEL,
        'DRAWS': DRAWS, 'labels': labels, 'reps': reps,
        'base_rep': base_rep, 'metric_name': metric_name, 'splits': splits,
        'detectors': detectors, 'k_values': k_values, 'metrics': metrics,
        'importance_methods': importance_methods, 'exponents': exponents,
        'forest_kw': forest_kw, 'hybrid_meta': hybrid_meta,
        'resume_note': None,
    }

    done_passes, resume_note = _restore(prior, ctx)
    ctx['resume_note'] = resume_note

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        if all(f'f{fold_i}s{d}' in done_passes for d in DRAWS):
            continue
        X_raw = reps[base_rep][train_idx]
        X_std = StandardScaler().fit_transform(
            np.asarray(X_raw, dtype=np.float64))
        train_reps = {k: v[train_idx] for k, v in reps.items()}
        val_reps = {k: v[val_idx] for k, v in reps.items()}
        train_augs = ({k: v[train_idx] for k, v in augmentations.items()}
                      if augmentations else None)
        val_augs = ({k: v[val_idx] for k, v in augmentations.items()}
                    if augmentations else None)
        g_train = groups[train_idx] if groups is not None else None

        for draw in DRAWS:
            pass_key = f'f{fold_i}s{draw}'
            if pass_key in done_passes:
                continue
            t_pass = time.time()
            spec = noise_spec_for(DATASET, CONDITION, LEVEL,
                                  noise_seed=draw, is_clf=is_clf)
            y_noisy, cmask, _prov = inject_train_labels(
                labels, spec, train_idx=train_idx, is_clf=is_clf,
                groups=noise_groups, dataset_name=DATASET)
            counts['n_corrupted'][pass_key] = int(np.sum(cmask))
            counts['n_training'][pass_key] = int(cmask.size)

            hybrid_X = hybrid_sim_X = None
            for fam, blueprint in (('hybrid', bp),
                                   ('hybrid_similarity', bp_sim)):
                if blueprint is None:
                    continue
                X_h, _ = hybrid_for_fold(
                    train_reps, val_reps, y_noisy, blueprint, g_train,
                    model_factories[DOWNSTREAM_MODEL](), metric_fn,
                    lower_is_better, augmentations_train=train_augs,
                    augmentations_val=val_augs)
                if X_h is None:
                    errors[f'{pass_key}|{fam}'] = \
                        'hybrid_for_fold could not build this fold'
                elif fam == 'hybrid':
                    hybrid_X = X_h
                else:
                    hybrid_sim_X = X_h

            built, zi = N21._spaces_for_pass(
                X_raw, X_std, y_noisy, is_clf, specs, importance_methods,
                forest_kw, hybrid_X, errors, pass_key,
                hybrid_sim_X=hybrid_sim_X)
            for key in zi:
                zero_imp[key] = zero_imp.get(key, 0) + 1

            for space_key, (kind, payload, space_metric) in built.items():
                for k in k_values:
                    k_eff = int(min(k, len(y_noisy) - 1))
                    try:
                        if kind == 'features':
                            nbrs = N21.neighbours_in_space(
                                payload, k_eff, space_metric)
                        else:
                            nbrs = N21.neighbours_in_distance_matrix(
                                payload, k_eff)
                    except Exception as exc:
                        errors[f'{pass_key}|{space_key}|k{k}'] = \
                            f'{type(exc).__name__}: {exc}'
                        continue
                    for det in detectors:
                        s = N21.suspicion_from_neighbours(
                            nbrs, y_noisy, det, is_clf)
                        d = detection_scores(cmask, s, CONDITION)
                        if d is not None:
                            cell = f'{det}_k{k}|{pass_key}'
                            ap[space_key][cell] = \
                                d['detection_average_precision']
                            pk[space_key][cell] = \
                                d['detection_precision_at_k']
            timings[pass_key] = time.time() - t_pass
            # Written after every PASS, not every fold. A pass is 11 minutes
            # on this dataset and a fold is three of them, so a run that dies
            # mid-fold used to lose every pass of that fold; now it loses
            # none. The crash on 2026-09-21 lost 2 finished passes this way.
            _write(out_path, ctx)
            print(f"  {pass_key} done in {timings[pass_key]:.0f}s, "
                  f"{len(errors)} failures so far "
                  f"[{time.time() - t_start:.0f}s]", flush=True)

    _write(out_path, ctx)
    _report(ap, specs, metrics, out_path, t_start)


def _ranked(ap, specs):
    return sorted(((s, float(np.mean(list(ap[s].values()))) if ap[s]
                    else float('nan')) for s in specs), key=lambda t: -t[1])


def _comparisons(ap, specs, metrics):
    from experiments.noise import n21_neighbour_distance as N21
    what = 'one (fold, draw, detector, neighbour count)'
    out = {}
    for space_key, spec in specs.items():
        if spec['family'] == 'plain':
            continue
        for ref in N21.plain_reference_for(spec, metrics):
            if ref not in ap:
                continue
            out[f'{space_key} vs {ref}'] = N21._compare(
                ap[space_key], ap[ref], True, what)
    # every distance against the one that leads the table, so the question
    # "does the top separate" has its own numbers
    ranked = _ranked(ap, specs)
    if ranked:
        top = ranked[0][0]
        for space_key, _v in ranked[1:]:
            out[f'{top} vs {space_key}'] = N21._compare(
                ap[top], ap[space_key], True, what)
    return out


def _write(path, ns):
    ap, pk, specs = ns['ap'], ns['pk'], ns['specs']
    payload = {
        'what_this_is': (
            "n21's fifteen distances at 5 folds and 3 corruption draws, "
            "detection average precision only. The repair is not measured "
            "here."),
        'dataset': ns['DATASET'],
        'molecules': int(len(ns['labels'])),
        'plain_representation': ns['base_rep'],
        'plain_representation_features':
            int(ns['reps'][ns['base_rep']].shape[1]),
        'base_rep_note': (
            'pick_noise_base_rep no longer falls back alphabetically. Every '
            'n21 number recorded before 2026-09-21 was read in esm2_cdr and '
            'is on a different footing.'),
        'dataset_metric': ns['metric_name'],
        'noise_condition': ns['CONDITION'],
        'level': ns['LEVEL'],
        'folds': len(ns['splits']),
        'draws': list(ns['DRAWS']),
        'detectors': ns['detectors'],
        'k_values': ns['k_values'],
        'distance_metrics': ns['metrics'],
        'importance_methods': ns['importance_methods'],
        'importance_exponents': ns['exponents'],
        'forest': dict(ns['forest_kw']),
        'hybrid': ns['hybrid_meta'],
        'spaces': ns['specs'],
        'molecules_corrupted_mean':
            float(np.mean(list(ns['counts']['n_corrupted'].values())))
            if ns['counts']['n_corrupted'] else None,
        'molecules_in_training_fold_mean':
            float(np.mean(list(ns['counts']['n_training'].values())))
            if ns['counts']['n_training'] else None,
        'molecules_corrupted_per_pass': dict(ns['counts']['n_corrupted']),
        'average_precision_by_space': {s: v for s, v in _ranked(ap, specs)},
        'precision_at_k_by_space': {
            s: (float(np.mean(list(pk[s].values()))) if pk[s] else None)
            for s in specs},
        'cells_per_space': {s: len(ap[s]) for s in specs},
        'comparisons_paired': _comparisons(ap, specs, ns['metrics']),
        'average_precision_per_cell': {s: dict(ap[s]) for s in specs},
        'precision_at_k_per_cell': {s: dict(pk[s]) for s in specs},
        'builds_with_all_zero_importance': ns['zero_imp'],
        'seconds_per_pass': ns['timings'],
        'resumed': ns.get('resume_note'),
        'errors': ns['errors'],
    }
    # Written to a temporary file and moved into place, because this is now
    # called after every pass and a kill during the write would leave a
    # truncated file -- which, with the resume above, is every finished pass
    # gone.
    tmp = f'{path}.writing'
    with open(tmp, 'w') as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, path)


def _report(ap, specs, metrics, out_path, t_start):
    ranked = _ranked(ap, specs)
    print("\n  one row is one distance, ranked by detection average "
          "precision")
    for s, v in ranked:
        print(f"    {s:40s} {v:.4f}   over {len(ap[s])} cells")
    comps = _comparisons(ap, specs, metrics)
    top = ranked[0][0] if ranked else None
    print(f"\n  the distance that leads, {top}, against each of the others, "
          f"paired per (fold, draw, detector, neighbour count):")
    for name, c in comps.items():
        if top is None or not name.startswith(f'{top} vs '):
            continue
        if not c.get('cells_paired'):
            print(f"    {name}: {c.get('absent_because')}")
            continue
        mark = '   larger than its own spread' \
            if c['gain_exceeds_its_own_spread'] else ''
        print(f"    {name:62s} won {c['cells_won']} of {c['cells_paired']}, "
              f"mean {c['mean_gain']:+.4f} "
              f"(spread {c['spread_over_cell_gains']:.4f}){mark}")
    print(f"\n  written to {out_path}")
    print(f"  TOTAL {time.time() - t_start:.0f}s")


if __name__ == '__main__':
    main()
