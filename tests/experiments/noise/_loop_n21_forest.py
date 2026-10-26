#!/usr/bin/env python3
"""n21's forest proximity: how many trees, and how many molecules per leaf.

THE QUESTION. n21 scores five distances and the forest proximity leads. It is
the only one of the five with no metric axis, and the two numbers that decide
its shape have never been swept: `FOREST_TREES = 100` and
`FOREST_MIN_SAMPLES_LEAF = 5`. The comment beside the second says "5 is stated,
recorded and overridable, not tuned". This tunes it.

WHAT A PROXIMITY IS, AND WHY THOSE TWO NUMBERS SET IT. Two molecules'
proximity is the share of the forest's trees in which they land in the same
leaf; the distance is one minus that share. So the number of trees is the
number of votes the share is taken over -- with 100 trees the distance can only
take 101 distinct values -- and the number of molecules a leaf must keep sets
how coarse a vote is. At one molecule per leaf almost no two molecules ever
share a leaf and the proximity measures which molecules the bootstrap left out
together. At fifty per leaf most pairs share most leaves and the distance
flattens toward zero.

WHAT IS MEASURED. Detection average precision: every training molecule ranked
by how wrong its label looks, graded against whether its label really moved.
`noise_helpers.detection_scores`, which is n21's headline.

THE REPAIR IS NOT MEASURED HERE, and that is the one thing this cuts. n21's
second reading fits a downstream model once per space, detector, neighbour
count and draw; on this dataset's control representation that is 8 seconds a
fit and it is the whole cost of the file. The detection figure costs no fit at
all. Cutting the fits is what buys 3 folds and 2 draws instead of the 2 folds
and 1 draw every n21 number on this dataset so far was read at -- 36 paired
readings per arm against 4. A forest shape that wins here is re-measured
through n21 itself, with the repair, before it is believed.

⛔ THE CONTROL CHANGED UNDER THIS FILE. On 2026-09-21 the fixed preference
list then used to choose the plain representation stopped falling back
alphabetically, so flab_thermo_tresanco's plain representation became `onehot`
at 5600 features, not `esm2_cdr` at 1280. Every n21 number recorded before
2026-09-21 -- forest proximity 0.2455, plain 0.2101 -- was read in esm2_cdr and
is on a different footing. That list was deleted 2026-09-25 (owner's decision:
the noise experiments now compare against each downstream model's top 2 single
representations). This script re-reads the plain rows in the representation
passed with --base-rep, so the forest arms and the control it is compared
against are in the same representation.

⛔ ROWS ARE NEVER CUT. All 672 antibodies, every fold. KIRBY_SMOKE_N is
refused.

IT PICKS UP WHERE IT WAS KILLED. One pass -- one fold under one corruption
draw -- is about 16 minutes here. This file used to write nothing until all six
were done, so the crash on 2026-09-21 threw away five finished passes and left
no result file at all. The result file is written after every pass now, and a
run that starts with one on disk scores only the passes missing from it, on the
same terms the folds-and-draws script uses. N21_FORCE=1 starts from nothing.

WHAT IT DOES NOT TOUCH. No cache file is deleted, moved or renamed.
KIRBY_CACHE_DIR points where every earlier flab run read its representations
from. No hybrid recipe is fetched, so no stored search is opened at all.
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
FOLDS = int(os.environ.get('N21_FOLDS', '3'))
DRAWS = tuple(int(x) for x in os.environ.get('N21_DRAWS', '0,1').split(','))
OUT = os.environ.get('N21_OUT', '/Volumes/seagate/local_noise_run/loop_n21')

# The two numbers being swept. n21 ships (100, 5).
# A tree count costs nothing beyond the largest one: sklearn draws each tree's
# seed in turn from one RandomState, so the first 100 leaf-id columns of a
# 300-tree forest ARE the 100-tree forest at the same seed. Measured on fold 0
# of this dataset -- the leaf matrices are equal element for element and the
# two proximity matrices differ by 0.00000000 -- so three tree counts are read
# off one fit.
TREE_COUNTS = tuple(int(x) for x in
                    os.environ.get('N21_TREES', '100,200,300').split(','))
MIN_SAMPLES_LEAF = tuple(int(x) for x in
                         os.environ.get('N21_LEAVES', '1,5,20,50').split(','))
SHIPPED = (100, 5)


def _forest_arm(n_trees, leaf):
    return f'forest|trees{n_trees}|leaf{leaf}'


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


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--base-rep', required=True,
                   help=('the single representation to compare against. '
                         'Required: the fixed preference list that used to '
                         'choose it was deleted 2026-09-25; the noise '
                         'experiments now compare against each downstream '
                         "model's top 2 single representations by its "
                         'stored score'))
    a = p.parse_args()
    from experiments._shared import (load_data, cv_split,
                                     get_domain_metric)
    from experiments.noise import n21_neighbour_distance as N21
    from experiments.noise.noise_helpers import (
        inject_train_labels, noise_spec_for, detection_scores)
    from experiments.dataset_policy import noise_groups_for

    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, f'{DATASET}__forest_shape.json')
    if not _claim(OUT, 'forest_shape'):
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
    is_clf = data['task_type'] == 'classification'
    metric_name = get_domain_metric(data)[0]
    base_rep = a.base_rep
    if base_rep not in reps:
        raise SystemExit(f'--base-rep {base_rep!r} is not among the loaded '
                         f'representations {sorted(reps)}')
    noise_groups, _n_groups, _largest = noise_groups_for(data)
    splits = list(cv_split(groups, FOLDS, next(iter(reps.values())), labels))

    detectors = list(N21.DETECTORS)
    k_values = list(N21.K_VALUES)
    metrics = list(N21.DISTANCE_METRICS_SWEPT)

    arms = [f'plain|{m}' for m in metrics]
    for t in TREE_COUNTS:
        for leaf in MIN_SAMPLES_LEAF:
            arms.append(_forest_arm(t, leaf))

    print(f"n21 forest shape on {DATASET}: {len(labels)} antibodies, "
          f"{len(splits)} folds, draws {list(DRAWS)}, level {LEVEL}, "
          f"condition {CONDITION}")
    print(f"  plain representation: {base_rep}, "
          f"{reps[base_rep].shape[1]} features")
    print(f"  dataset metric {metric_name}; the reading here is detection "
          f"average precision, which no model fit produces")
    print(f"  {len(arms)} arms: 2 plain and "
          f"{len(TREE_COUNTS) * len(MIN_SAMPLES_LEAF)} forests")
    print(f"  one cell is one (fold, draw, detector, neighbour count): "
          f"{len(splits)} x {len(DRAWS)} x {len(detectors)} x {len(k_values)} "
          f"= {len(splits) * len(DRAWS) * len(detectors) * len(k_values)} "
          f"cells per arm", flush=True)

    # {arm: {cell_key: average precision}}
    ap = {a: {} for a in arms}
    pk = {a: {} for a in arms}
    counts = {'n_corrupted': {}, 'n_training': {}}
    timings = {}
    errors = {}
    passes_done = set()

    # ── what an earlier copy of this run already scored ────────────────────
    # A pass is 16 minutes on this dataset and the run used to write nothing
    # until all six were finished, so the crash on 2026-09-21 threw away five
    # finished passes. A pass is trusted only if it was scored on the same
    # arms, in the same representation, at the same level and condition.
    if prior:
        mismatch = []
        for field, here in (('plain_representation', base_rep),
                            ('noise_condition', CONDITION),
                            ('level', LEVEL),
                            ('detectors', detectors),
                            ('k_values', k_values)):
            if prior.get(field) != here:
                mismatch.append(f'{field} was {prior.get(field)!r}, '
                                f'is {here!r}')
        if set(prior.get('cells_per_arm') or {}) != set(arms):
            mismatch.append('the set of arms differs')
        if mismatch:
            print(f"  the result file on disk is not comparable "
                  f"({'; '.join(mismatch)}), so every pass runs again")
        else:
            passes_done = set(prior.get('passes_finished') or [])
            for arm in arms:
                for cell, v in ((prior.get('average_precision_per_cell')
                                 or {}).get(arm) or {}).items():
                    if cell.split('|')[-1] in passes_done:
                        ap[arm][cell] = v
                for cell, v in ((prior.get('precision_at_k_per_cell')
                                 or {}).get(arm) or {}).items():
                    if cell.split('|')[-1] in passes_done:
                        pk[arm][cell] = v
            counts['n_corrupted'].update(
                prior.get('molecules_corrupted_per_pass') or {})
            counts['n_training'].update(
                prior.get('molecules_in_training_fold_per_pass') or {})
            timings.update(prior.get('timings_seconds') or {})
            if passes_done:
                print(f"  {len(passes_done)} pass(es) of (fold, draw) are "
                      f"already on disk and are not re-run: "
                      f"{', '.join(sorted(passes_done))}", flush=True)

    def _write():
        """The whole result file, written after every pass.

        Called per pass so a killed run keeps what it finished, and
        again at the end. Reads the tables above as they stand.
        """
        shipped_arm = _forest_arm(*SHIPPED)
        ranked = sorted(((a, float(np.mean(list(ap[a].values()))) if ap[a]
                          else float('nan')) for a in arms),
                        key=lambda t: -t[1])

        refs = tuple(f'plain|{m}' for m in metrics)
        if shipped_arm in ap:
            refs = (shipped_arm,) + refs
        comparisons = {}
        for arm in arms:
            for ref in refs:
                if arm == ref:
                    continue
                comparisons[f'{arm} vs {ref}'] = N21._compare(
                    ap[arm], ap[ref], True,
                    'one (fold, draw, detector, neighbour count)')

        result = {
            'what_this_is': (
                "n21's forest proximity, swept over how many trees the leaf share "
                "is taken across and how many molecules a leaf must keep. "
                "Detection average precision only; the repair is not measured "
                "here."),
            'dataset': DATASET,
            'molecules': int(len(labels)),
            'plain_representation': base_rep,
            'plain_representation_features': int(reps[base_rep].shape[1]),
            'base_rep_note': (
                'passed with --base-rep. The fixed preference list that chose '
                'it before was deleted 2026-09-25. Every n21 number recorded '
                'before 2026-09-21 was read in esm2_cdr and is on a different '
                'footing.'),
            'dataset_metric': metric_name,
            'noise_condition': CONDITION,
            'level': LEVEL,
            'folds': len(splits),
            'draws': list(DRAWS),
            'detectors': detectors,
            'k_values': k_values,
            'distance_metrics': metrics,
            'tree_counts': list(TREE_COUNTS),
            'min_samples_leaf_values': list(MIN_SAMPLES_LEAF),
            'shipped_setting': {'n_trees': SHIPPED[0],
                                'min_samples_leaf': SHIPPED[1],
                                'arm': shipped_arm},
            'molecules_corrupted_mean':
                float(np.mean(list(counts['n_corrupted'].values())))
                if counts['n_corrupted'] else None,
            'molecules_in_training_fold_mean':
                float(np.mean(list(counts['n_training'].values())))
                if counts['n_training'] else None,
            'molecules_corrupted_per_pass': dict(counts['n_corrupted']),
            'molecules_in_training_fold_per_pass':
                dict(counts['n_training']),
            'passes_finished': sorted(
                set(counts['n_corrupted']) | set(passes_done)),
            'average_precision_by_arm': {a: v for a, v in ranked},
            'precision_at_k_by_arm': {
                a: (float(np.mean(list(pk[a].values()))) if pk[a] else None)
                for a in arms},
            'cells_per_arm': {a: len(ap[a]) for a in arms},
            'comparisons_paired': comparisons,
            'average_precision_per_cell': {a: dict(ap[a]) for a in arms},
            'precision_at_k_per_cell': {a: dict(pk[a]) for a in arms},
            'neighbours_asked_for_once_per_count': (
                'yes. Asking once at 20 and keeping the first 5 was measured on '
                'fold 0 and gives a DIFFERENT set of five: 135 of the 537 '
                'training molecules have two others at exactly the distance of '
                'their fifth nearest.'),
            'timings_seconds': timings,
            'errors': errors,
            'seconds_total': time.time() - t_start,
        }
        # Written to a temporary file and moved into place: this is called
        # after every pass now, and a kill during the write would leave a
        # truncated file, which is every finished pass gone.
        tmp = f'{out_path}.writing'
        with open(tmp, 'w') as fh:
            json.dump(result, fh, indent=2)
        os.replace(tmp, out_path)
        return shipped_arm, ranked, comparisons

    for fold_i, (train_idx, _val_idx) in enumerate(splits):
        if all(f'f{fold_i}s{d}' in passes_done for d in DRAWS):
            continue
        X_std = StandardScaler().fit_transform(
            np.asarray(reps[base_rep][train_idx], dtype=np.float64))

        for draw in DRAWS:
            pass_key = f'f{fold_i}s{draw}'
            if pass_key in passes_done:
                continue
            spec = noise_spec_for(DATASET, CONDITION, LEVEL,
                                  noise_seed=draw, is_clf=is_clf)
            y_noisy, cmask, _prov = inject_train_labels(
                labels, spec, train_idx=train_idx, is_clf=is_clf,
                groups=noise_groups, dataset_name=DATASET)
            counts['n_corrupted'][pass_key] = int(np.sum(cmask))
            counts['n_training'][pass_key] = int(cmask.size)

            # ── the plain rows, re-read in the representation the control is
            # now. One sklearn query per neighbour count, which is what n21
            # does. Asking once at 20 and keeping the first 5 was measured and
            # is NOT the same neighbour set: on this fold 135 of the 537
            # molecules have two others at exactly the distance of their fifth
            # nearest, so which of the two is kept depends on how many
            # neighbours were asked for. The saving was not worth a different
            # neighbour set.
            for metric in metrics:
                t0 = time.time()
                for k in k_values:
                    nbrs = N21.neighbours_in_space(X_std, k, metric)
                    for det in detectors:
                        s = N21.suspicion_from_neighbours(
                            nbrs, y_noisy, det, is_clf)
                        d = detection_scores(cmask, s, CONDITION)
                        if d is not None:
                            cell = f'{det}_k{k}|{pass_key}'
                            ap[f'plain|{metric}'][cell] = \
                                d['detection_average_precision']
                            pk[f'plain|{metric}'][cell] = \
                                d['detection_precision_at_k']
                timings[f'{pass_key}|neighbours|{metric}'] = time.time() - t0

            # ── the forests. One fit per leaf size at the LARGEST tree count;
            # the smaller counts read the first columns of the same leaf
            # matrix. sklearn draws each tree's seed in turn from one
            # RandomState, so the first 100 trees of a 300-tree forest are the
            # 100-tree forest at the same seed. Measured on this dataset's
            # fold 0 before this script was written: the two leaf matrices are
            # equal element for element and the two proximity matrices differ
            # by 0.00000000, so a 300-tree fit pays for both tree counts.
            for leaf in MIN_SAMPLES_LEAF:
                t0 = time.time()
                try:
                    leaves, oob = N21.forest_leaf_ids(
                        X_std, y_noisy, is_clf,
                        n_trees=max(TREE_COUNTS), min_samples_leaf=leaf,
                        seed=N21.FOREST_SEED)
                except Exception as exc:
                    errors[f'{pass_key}|leaf{leaf}'] = \
                        f'{type(exc).__name__}: {exc}'
                    continue
                timings[f'{pass_key}|forest|leaf{leaf}'] = time.time() - t0
                for n_trees in TREE_COUNTS:
                    D = N21.forest_proximity_distance(leaves[:, :n_trees],
                                                      oob[:, :n_trees])
                    arm = _forest_arm(n_trees, leaf)
                    for k in k_values:
                        nbrs = N21.neighbours_in_distance_matrix(D, k)
                        for det in detectors:
                            s = N21.suspicion_from_neighbours(
                                nbrs, y_noisy, det, is_clf)
                            d = detection_scores(cmask, s, CONDITION)
                            if d is not None:
                                cell = f'{det}_k{k}|{pass_key}'
                                ap[arm][cell] = \
                                    d['detection_average_precision']
                                pk[arm][cell] = \
                                    d['detection_precision_at_k']
            _write()
            print(f"  {pass_key} done [{time.time() - t_start:.0f}s]",
                  flush=True)


    shipped_arm, ranked, comparisons = _write()

    print("\n  one row is one forest shape, or the plain representation it is "
          "compared against")
    for a, v in ranked:
        print(f"    {a:28s} average precision {v:.4f}  "
              f"over {len(ap[a])} cells")
    head = shipped_arm if shipped_arm in ap else refs[0]
    print(f"\n  against {head}, paired per "
          f"(fold, draw, detector, neighbour count):")
    for name, c in comparisons.items():
        if not name.endswith(f' vs {head}'):
            continue
        if not c.get('cells_paired'):
            print(f"    {name}: {c.get('absent_because')}")
            continue
        mark = '   larger than its own spread' \
            if c['gain_exceeds_its_own_spread'] else ''
        print(f"    {name:58s} won {c['cells_won']} of {c['cells_paired']}, "
              f"mean {c['mean_gain']:+.4f} "
              f"(spread {c['spread_over_cell_gains']:.4f}){mark}")
    print(f"\n  written to {out_path}")
    print(f"  TOTAL {time.time() - t_start:.0f}s")


if __name__ == '__main__':
    main()
