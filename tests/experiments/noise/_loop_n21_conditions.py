#!/usr/bin/env python3
"""n21's distances under noise a feature space could in principle predict.

THE QUESTION, AND WHY IT IS NOT THE ONE n21 HAS BEEN ASKING. n21 ranks
distances by how well the molecules whose labels moved can be picked out of a
training fold. Every reading of it so far is under `outlier_p10`, which chooses
the molecules it damages AT RANDOM. No arrangement of a feature space predicts
a random draw, so under that condition a distance that reads the features has
nothing to find in principle, and the whole table -- forest proximity ahead of
the plain representation, importance weighting ahead of it -- is a comparison
of five distances on a task where none of them should have an edge from
structure.

`grouped_wider` chooses whole scaffold groups and widens their error, so a
molecule's neighbours in a feature space really are more likely to have moved
if it has. It is the only zero-mean condition where the noise is predictable
from structure, and the only one that tests whether reading the features helps.

`censoring` is here because it is the most prevalent real mechanism and the
only one that is not zero-mean: every label past the assay limit is recorded AS
the limit, so the damaged molecules all sit at one value rather than scattered
around their true one. A detector that scores a molecule by how far its label
is from its neighbours' mean has no reason to work there, and n23 measured
three uncertainty sources under it at or below a coin flip. A negative here is
worth having: it says which mechanism this family of detectors is for.

`outlier_p10` runs beside them as the reference, because it is what every n21
number on record was read under.

⛔ `grouped_shifted` IS NOT HERE AND CANNOT BE. It shifts EVERY group by its own
offset, so every training molecule's label moves, the list of molecules that
moved is the list of all of them, and average precision is 1.0 by construction.
`noise_helpers.detection_is_defined` refuses it and n21 refuses it.
`noiseInject.core` line 442 states the same thing from the injector's side: the
affected fraction is 1.0. Its offsets are worth measuring through a REPAIR
figure, which this file does not compute.

⛔ WHY caco2 AND NOT flab_thermo_tresanco. `dataset_policy.noise_groups_for`
returns no groups for a dataset with no SMILES, and the grouped conditions are
then dropped rather than run over invented groups. Every antibody set is in
that position. caco2 has 2,158 molecules with SMILES and Murcko scaffold
groups, so it can run all three. The flab table and this one are therefore on
DIFFERENT conditions and different molecules, and neither number transfers.

⛔ THE THREE CONDITIONS DAMAGE DIFFERENT NUMBERS OF MOLECULES, so their average
precisions are not comparable with each other: a tenth of the training fold
under outlier_p10, about a fifth under grouped_wider, a quarter under
censoring. What a coin flip scores is the damaged share itself, and every
number here is written beside it. The comparison that reads across conditions
is each distance against the PLAIN representation within its own condition.

WHAT IS CUT TO PAY FOR THREE CONDITIONS. The repair, exactly as in
`_loop_n21_folds.py`: n21's second reading fits a downstream model once per
space, detector, neighbour count, draw and fold, and detection average
precision costs no fit at all. Nothing about the distances is cut -- all five
families, both metrics, both importance scorers, both exponents and all three
neighbour counts run, from n21's own `space_specs` and `_spaces_for_pass`.

⛔ ROWS ARE NEVER CUT. All 2,158 molecules, every fold. KIRBY_SMOKE_N is
refused.

IT PICKS UP WHERE IT WAS KILLED, on the same terms as the other two n21 loops:
the result file is written after every (fold, draw, condition) and a run that
starts with one on disk scores only what is missing from it.
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

DATASET = os.environ.get('N21_DATASET', 'caco2')

# (condition, level). The level means a different thing under censoring: it is
# the share of labels clipped at the assay limit, not a dose in fractions of
# the label spread, and 0.25 is in `dataset_policy.CENSORING_LEVELS_DETECT`.
# 0.5 is the level every n21 reading on record was taken at.
CONDITIONS = (('outlier_p10', 0.5), ('grouped_wider', 0.5), ('censoring', 0.25))

FOLDS = int(os.environ.get('N21_FOLDS', '3'))
DRAWS = tuple(int(x) for x in os.environ.get('N21_DRAWS', '0,1').split(','))
DOWNSTREAM_MODEL = 'rf'     # only used to fetch the two hybrid recipes
OUT = os.environ.get('N21_OUT', '/Volumes/seagate/local_noise_run/loop_n21')


def _claim(out_dir, name):
    """Refuse to start while another copy of this script is already running.

    The supervisor tracks a job by a pid file it writes itself, so a run
    started by hand is invisible to it and it would start a second one. This is
    the lock both can see. A stale file whose pid is gone is ignored.
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


def main():
    from experiments._shared import (load_data, pick_noise_base_rep, cv_split,
                                     get_domain_metric, get_model_factories,
                                     get_best_search_for_downstream)
    from experiments.noise import n21_neighbour_distance as N21
    from experiments.noise.noise_helpers import (
        inject_train_labels, noise_spec_for, detection_scores, hybrid_for_fold,
        detection_is_defined)
    from experiments.dataset_policy import noise_groups_for

    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, f'{DATASET}__conditions.json')
    if not _claim(OUT, 'conditions'):
        return
    prior = {}
    if os.path.exists(out_path) and not os.environ.get('N21_FORCE'):
        try:
            with open(out_path) as fh:
                prior = json.load(fh)
        except Exception as exc:
            print(f"  the result file on disk could not be read "
                  f"({type(exc).__name__}), so this run starts from nothing")

    for cond, _lvl in CONDITIONS:
        if not detection_is_defined(cond):
            raise SystemExit(
                f"{cond} is not a condition detection can be graded under. "
                f"Every training molecule's label moves, so the molecules that "
                f"moved are all of them and average precision is 1.0 by "
                f"construction.")

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
    noise_groups, n_noise_groups, largest_share = noise_groups_for(data)
    if noise_groups is None:
        raise SystemExit(
            f"{DATASET} has no SMILES, so noise_groups_for returns no groups "
            f"and grouped_wider would be uniform noise wearing a grouped "
            f"name. This run needs a dataset with scaffold groups.")
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

    print(f"n21 under three noise conditions on {DATASET}: {len(labels)} "
          f"molecules, {n_noise_groups} scaffold groups, {len(splits)} folds, "
          f"draws {list(DRAWS)}")
    print(f"  plain representation: {base_rep}, "
          f"{reps[base_rep].shape[1]} features")
    print(f"  dataset metric {metric_name}, lower is better "
          f"{lower_is_better}; the reading here is detection average "
          f"precision, which no model fit produces and which does not use "
          f"the dataset's metric at all")
    for cond, lvl in CONDITIONS:
        print(f"    {cond} at level {lvl}")
    print(f"  {len(specs)} distances, detectors {detectors}, "
          f"neighbour counts {k_values}", flush=True)

    ap = {s: {} for s in specs}
    pk = {s: {} for s in specs}
    corrupted = {}      # {cell key: molecules whose label moved}
    training = {}       # {cell key: molecules in the training fold}
    errors = {}
    zero_imp = {}
    timings = {}
    done = set(prior.get('passes_finished') or []) if prior else set()

    if prior:
        mismatch = []
        for field, here in (('plain_representation', base_rep),
                            ('detectors', detectors),
                            ('k_values', k_values),
                            ('conditions', [list(c) for c in CONDITIONS])):
            if prior.get(field) != here:
                mismatch.append(f'{field} was {prior.get(field)!r}, '
                                f'is {here!r}')
        if set(prior.get('spaces') or {}) != set(specs):
            mismatch.append('the set of distances differs')
        if mismatch:
            print(f"  the result file on disk is not comparable "
                  f"({'; '.join(mismatch)}), so every pass runs again")
            done = set()
        else:
            for s in specs:
                for cell, v in ((prior.get('average_precision_per_cell')
                                 or {}).get(s) or {}).items():
                    if _pass_of(cell) in done:
                        ap[s][cell] = v
                for cell, v in ((prior.get('precision_at_k_per_cell')
                                 or {}).get(s) or {}).items():
                    if _pass_of(cell) in done:
                        pk[s][cell] = v
            corrupted.update(prior.get('molecules_moved_per_pass') or {})
            training.update(prior.get('molecules_in_training_fold_per_pass')
                            or {})
            timings.update(prior.get('seconds_per_pass') or {})
            if done:
                print(f"  {len(done)} pass(es) of (fold, draw, condition) are "
                      f"already on disk and are not re-run", flush=True)

    ctx = {
        'ap': ap, 'pk': pk, 'specs': specs, 'corrupted': corrupted,
        'training': training, 'errors': errors, 'zero_imp': zero_imp,
        'timings': timings, 'labels': labels, 'reps': reps,
        'base_rep': base_rep, 'metric_name': metric_name,
        'lower_is_better': lower_is_better, 'splits': splits,
        'detectors': detectors, 'k_values': k_values, 'metrics': metrics,
        'importance_methods': importance_methods, 'exponents': exponents,
        'forest_kw': forest_kw, 'hybrid_meta': hybrid_meta,
        'n_noise_groups': n_noise_groups, 'largest_share': largest_share,
        'done': done,
    }

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        if all(f'{c}|f{fold_i}s{d}' in done
               for c, _l in CONDITIONS for d in DRAWS):
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
            for condition, level in CONDITIONS:
                pass_key = f'{condition}|f{fold_i}s{draw}'
                if pass_key in done:
                    continue
                t_pass = time.time()
                spec = noise_spec_for(DATASET, condition, level,
                                      noise_seed=draw, is_clf=is_clf)
                y_noisy, cmask, _prov = inject_train_labels(
                    labels, spec, train_idx=train_idx, is_clf=is_clf,
                    groups=noise_groups, dataset_name=DATASET)
                corrupted[pass_key] = int(np.sum(cmask))
                training[pass_key] = int(cmask.size)

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
                            d = detection_scores(cmask, s, condition)
                            if d is not None:
                                cell = f'{det}_k{k}|{pass_key}'
                                ap[space_key][cell] = \
                                    d['detection_average_precision']
                                pk[space_key][cell] = \
                                    d['detection_precision_at_k']
                timings[pass_key] = time.time() - t_pass
                done.add(pass_key)
                _write(out_path, ctx)
                share = (corrupted[pass_key] / training[pass_key]
                         if training[pass_key] else float('nan'))
                print(f"  {pass_key} done in {timings[pass_key]:.0f}s, "
                      f"{corrupted[pass_key]} of {training[pass_key]} "
                      f"training molecules moved ({share:.3f} of them, which "
                      f"is what a coin flip scores), {len(errors)} failures "
                      f"so far [{time.time() - t_start:.0f}s]", flush=True)

    _write(out_path, ctx)
    _report(ctx, out_path, t_start)


def _pass_of(cell_key):
    """'neighbor_residual_k5|censoring|f0s1' -> 'censoring|f0s1'."""
    return '|'.join(str(cell_key).split('|')[1:])


def _cells_of(cells, condition):
    return {c: v for c, v in cells.items()
            if _pass_of(c).split('|')[0] == condition}


def _ranked(ap, specs, condition):
    out = []
    for s in specs:
        vals = list(_cells_of(ap[s], condition).values())
        out.append((s, float(np.mean(vals)) if vals else float('nan')))
    return sorted(out, key=lambda t: -t[1])


def _comparisons(ctx, condition):
    from experiments.noise import n21_neighbour_distance as N21
    ap, specs = ctx['ap'], ctx['specs']
    what = 'one (fold, draw, detector, neighbour count) under this condition'
    out = {}
    for space_key, spec in specs.items():
        if spec['family'] == 'plain':
            continue
        for ref in N21.plain_reference_for(spec, ctx['metrics']):
            if ref not in ap:
                continue
            out[f'{space_key} vs {ref}'] = N21._compare(
                _cells_of(ap[space_key], condition),
                _cells_of(ap[ref], condition), True, what)
    return out


def _by_pass(cells):
    """{pass key: mean average precision over that pass's cells}.

    The six readings inside a pass -- two detectors by three neighbour counts
    -- are the same molecules under the same corruption, and the neighbour sets
    are nested, so they are averaged before anything is counted. The unit that
    repeats is the (fold, draw) corruption.
    """
    out = {}
    for cell, v in cells.items():
        if v is None or not np.isfinite(float(v)):
            continue
        out.setdefault(_pass_of(cell), []).append(float(v))
    return {p: float(np.mean(vs)) for p, vs in out.items()}


def _pass_level(ctx, condition, a_key, b_key):
    a = _by_pass(_cells_of(ctx['ap'].get(a_key, {}), condition))
    b = _by_pass(_cells_of(ctx['ap'].get(b_key, {}), condition))
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    g = np.array([a[p] - b[p] for p in shared], dtype=float)
    return {'passes_paired': int(g.size), 'passes_won': int((g > 0).sum()),
            'mean_gain_over_passes': float(g.mean()),
            'spread_over_passes': float(g.std(ddof=1)) if g.size > 1
            else float('nan')}


def _write(path, ctx):
    ap, pk, specs = ctx['ap'], ctx['pk'], ctx['specs']
    per_condition = {}
    for condition, level in CONDITIONS:
        moved = [v for k, v in ctx['corrupted'].items()
                 if k.split('|')[0] == condition]
        in_fold = [v for k, v in ctx['training'].items()
                   if k.split('|')[0] == condition]
        ranked = _ranked(ap, specs, condition)
        top = ranked[0][0] if ranked and np.isfinite(ranked[0][1]) else None
        per_condition[condition] = {
            'level': level,
            'level_means': ('the share of labels clipped at the assay limit'
                            if condition == 'censoring'
                            else 'the dose, in fractions of the label spread'),
            'molecules_moved_mean': float(np.mean(moved)) if moved else None,
            'molecules_in_training_fold_mean':
                float(np.mean(in_fold)) if in_fold else None,
            'what_a_coin_flip_scores':
                (float(np.mean(moved)) / float(np.mean(in_fold)))
                if moved and in_fold and np.mean(in_fold) else None,
            'average_precision_by_space': {s: v for s, v in ranked},
            'cells_per_space': {s: len(_cells_of(ap[s], condition))
                                for s in specs},
            'comparisons_paired': _comparisons(ctx, condition),
            'top_against_plain_by_pass': (
                {ref: _pass_level(ctx, condition, top, ref)
                 for ref in (f'plain|{m}' for m in ctx['metrics'])}
                if top else {}),
        }
    payload = {
        'what_this_is': (
            "n21's distances under three noise conditions: outlier_p10, which "
            "picks the molecules it damages at random; grouped_wider, which "
            "damages whole scaffold groups and is the only zero-mean "
            "condition where the noise is predictable from structure; and "
            "censoring, where every label past the assay limit is recorded as "
            "the limit. Detection average precision only -- no model is "
            "fitted, and the repair is not measured here."),
        'read_the_conditions_apart': (
            "The three damage different numbers of molecules, so their "
            "average precisions do not compare with each other. What a coin "
            "flip scores is the damaged share, and it is written beside each. "
            "The comparison that carries across is each distance against the "
            "PLAIN representation within its own condition."),
        'dataset': DATASET,
        'molecules': int(len(ctx['labels'])),
        'scaffold_groups': ctx['n_noise_groups'],
        'largest_scaffold_group_share': ctx['largest_share'],
        'plain_representation': ctx['base_rep'],
        'plain_representation_features':
            int(ctx['reps'][ctx['base_rep']].shape[1]),
        'dataset_metric': ctx['metric_name'],
        'dataset_metric_lower_is_better': ctx['lower_is_better'],
        'conditions': [list(c) for c in CONDITIONS],
        'folds': len(ctx['splits']),
        'draws': list(DRAWS),
        'detectors': ctx['detectors'],
        'k_values': ctx['k_values'],
        'distance_metrics': ctx['metrics'],
        'importance_methods': ctx['importance_methods'],
        'importance_exponents': ctx['exponents'],
        'forest': dict(ctx['forest_kw']),
        'hybrid': ctx['hybrid_meta'],
        'spaces': specs,
        'per_condition': per_condition,
        'passes_finished': sorted(ctx['done']),
        'molecules_moved_per_pass': dict(ctx['corrupted']),
        'molecules_in_training_fold_per_pass': dict(ctx['training']),
        'average_precision_per_cell': {s: dict(ap[s]) for s in specs},
        'precision_at_k_per_cell': {s: dict(pk[s]) for s in specs},
        'builds_with_all_zero_importance': ctx['zero_imp'],
        'seconds_per_pass': ctx['timings'],
        'errors': ctx['errors'],
    }
    tmp = f'{path}.writing'
    with open(tmp, 'w') as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, path)


def _report(ctx, out_path, t_start):
    for condition, _level in CONDITIONS:
        ranked = [t for t in _ranked(ctx['ap'], ctx['specs'], condition)
                  if np.isfinite(t[1])]
        if not ranked:
            print(f"\n  {condition}: no finished reading")
            continue
        moved = [v for k, v in ctx['corrupted'].items()
                 if k.split('|')[0] == condition]
        in_fold = [v for k, v in ctx['training'].items()
                   if k.split('|')[0] == condition]
        flip = np.mean(moved) / np.mean(in_fold)
        print(f"\n  {condition}: the injector moved the label of "
              f"{np.mean(moved):.0f} of the {np.mean(in_fold):.0f} molecules "
              f"in a training fold, so a coin flip scores {flip:.4f} "
              f"detection average precision")
        for s, v in ranked:
            print(f"    {s:40s} {v:.4f}")
        top = ranked[0][0]
        for ref in (f'plain|{m}' for m in ctx['metrics']):
            c = _pass_level(ctx, condition, top, ref)
            if c is None:
                continue
            print(f"    {top} over {ref}: won {c['passes_won']} of "
                  f"{c['passes_paired']} passes, mean "
                  f"{c['mean_gain_over_passes']:+.4f} average precision "
                  f"(spread over passes {c['spread_over_passes']:.4f})")
    print(f"\n  written to {out_path}")
    print(f"  TOTAL {time.time() - t_start:.0f}s")


if __name__ == '__main__':
    main()
