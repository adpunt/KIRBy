#!/usr/bin/env python3
"""N17: Does cleaning help when the noise is REAL, not injected?

Paper: Noise.

THE QUESTION. Every other experiment in this suite corrupts clean labels on
purpose and then asks whether a detector can find what it corrupted. That is a
controlled measurement and it is worth having, but a reader can fairly say the
noise was drawn from a distribution the same script chose. This experiment asks
the same question where nobody chose the noise: the labels disagree because
different laboratories measured the same thing and got different numbers.

WHERE THE THREE PIECES COME FROM. A removal claim needs three quantities from
three different places, or it grades itself.

  a per-molecule error figure   the spread of one entity's repeat measurements
  a training label              ONE of those measurements, chosen at random
  a grading label               the mean of the measurements NOT chosen

Nothing here is injected and nothing is drawn from a distribution. The training
label is one laboratory's number, with whatever that laboratory got wrong still
in it. The grading label is what the other laboratories agreed on.

WHAT AN ENTITY IS. It depends on the dataset, and the loader decides, not this
file. On a drug-target set an entity is one (drug, protein) pair, so repeat rows
are the same pair measured more than once. `_ground_truth_providers
.build_replicate_groups` builds the ids and this experiment does not second-guess
it. A dataset whose loader exposes no repeat structure cannot run here and says
so rather than inventing one.

HOW THE SPLIT WORKS, AND WHY NO ENTITY IS ON BOTH SIDES. One row per entity, then
ordinary grouped cross-validation over those rows. In a fold, the training
entities contribute their single-measurement label and the validation entities
are graded against their consensus label. An entity is never in both, so the
single measurement an entity trained on is never the thing it is graded against.

Only entities with MIN_MEASUREMENTS or more can be graded: below three there is
no consensus left once one measurement is held out. Entities measured once or
twice still train, at face value, and are never graded. That is deliberate --
throwing them out would shrink the training set for a reason that has nothing to
do with the question.

THE FIVE WAYS OF CHOOSING WHAT TO TRAIN ON. All five are graded on exactly the
same validation entities, against exactly the same consensus labels. The only
thing that differs is which training entities survive and with what label.

  no_cleaning  every training entity, untouched. The thing to beat.
  drop         the entities a detector flags are removed.
  repair       a flagged entity's label is replaced by its neighbours' mean.
  ceiling      the entities whose TRUE repeat spread is in the worst tenth are
               removed. It uses the answer, so no detector can beat it. It says
               how much is on the table before any detector is involved.
  floor        the same NUMBER of entities are removed, chosen at random. It says
               how much of any gain is just from training on fewer entities.

A removal result is `drop` or `repair` beating `no_cleaning`, reported against
the distance between `floor` and `ceiling`. Beating `no_cleaning` while sitting
at `floor` is not a removal result; it is a smaller training set.

WHAT THIS DOES NOT CLAIM. It says nothing about which detector is best in
general -- that is n11, under injected noise where the answer is known. It says
nothing about datasets whose loaders collapse their repeats before anything sees
them, which as of 2026-09-12 is most of them (NOISE_AUDIT_2026_09_10.md 11.1).

Usage:
    python n17_replicate_removal.py --dataset bindingdb_kd --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS,
)
from experiments.noise._ground_truth_providers import build_replicate_groups
from kirby.utils.noise_estimation import get_noise_estimator

# Below this an entity cannot be graded: hold one measurement out of two and the
# "consensus of the rest" is a single measurement, which is not a consensus.
MIN_MEASUREMENTS = 3

# Share of training entities the detector flags, and the share the ceiling
# removes. Held equal on purpose: `drop`, `ceiling` and `floor` must remove the
# same NUMBER of entities, or the comparison measures training-set size.
FLAG_FRACTION = 0.10

# The detector. neighbour_residual scores an entity by how far its label sits
# from its nearest neighbours' labels. It is the quantity the literature names
# and it needs no model fit of its own.
DETECTOR = 'neighbor_residual'
DETECTOR_K = 10

ARMS = ('no_cleaning', 'drop', 'repair', 'ceiling', 'floor')

# How many times the whole thing is repeated. WHICH measurement becomes the
# training label is a random choice, so one run is one draw of it. Three, to
# match every other experiment in this suite.
N_DRAWS = 3


def build_entities(labels, replicate_groups, seed):
    """One row per entity: a training label, a grading label, and the truth.

    Returns a dict of arrays, all in the same order, one entry per entity:

      row_index       a row of the original data to read features from. Every row
                      of one entity carries the same features -- they are the same
                      drug against the same protein -- so the first one will do.
      n_measurements  how many times this entity was measured.
      train_label     ONE measurement, chosen at random, for entities that can be
                      graded. The mean of all of them for entities that cannot,
                      because there is nothing to hold out for them and the mean is
                      the best single number their rows support.
      grade_label     the mean of the measurements NOT chosen. NaN where the entity
                      has too few measurements to grade.
      gradeable       True where grade_label is a real number.
      true_spread     the standard deviation of ALL of this entity's measurements,
                      which is how far the laboratories actually disagreed. NaN for
                      an entity measured once. This is the answer the ceiling uses
                      and no detector is allowed to see.
    """
    labels = np.asarray(labels, dtype=np.float64)
    groups = np.asarray(replicate_groups)
    rng = np.random.default_rng(int(seed))

    order = {}
    for i, g in enumerate(groups):
        order.setdefault(g, []).append(i)

    gids = sorted(order, key=lambda g: order[g][0])
    n = len(gids)
    out = {
        'group_id': np.array(gids, dtype=object),
        'row_index': np.zeros(n, dtype=np.int64),
        'n_measurements': np.zeros(n, dtype=np.int64),
        'train_label': np.zeros(n, dtype=np.float64),
        'grade_label': np.full(n, np.nan, dtype=np.float64),
        'gradeable': np.zeros(n, dtype=bool),
        'true_spread': np.full(n, np.nan, dtype=np.float64),
    }
    for j, g in enumerate(gids):
        rows = np.asarray(order[g], dtype=np.int64)
        vals = labels[rows]
        k = len(rows)
        out['row_index'][j] = rows[0]
        out['n_measurements'][j] = k
        if k >= 2:
            out['true_spread'][j] = float(np.std(vals, ddof=0))
        if k >= MIN_MEASUREMENTS:
            held = int(rng.integers(k))
            rest = np.delete(vals, held)
            out['train_label'][j] = float(vals[held])
            out['grade_label'][j] = float(np.mean(rest))
            out['gradeable'][j] = True
        else:
            out['train_label'][j] = float(np.mean(vals))
    return out


def detector_scores(X_train, y_train):
    """Per-entity suspicion score from the training features and labels only.

    It never sees `true_spread` and never sees `grade_label`. Returns None if the
    detector raises, and the caller then drops the two detector arms for that fold
    rather than scoring them against a silent fallback.
    """
    try:
        est = get_noise_estimator(DETECTOR, k=DETECTOR_K, metric=None)
        return np.asarray(est.estimate(X_train, y_train), dtype=float)
    except Exception as exc:
        print(f"      detector failed: {type(exc).__name__}: {exc}")
        return None


def neighbour_means(X_train, y_train, k=DETECTOR_K):
    """Each training entity's neighbours' mean label, for the repair arm."""
    from sklearn.neighbors import NearestNeighbors
    k_eff = int(min(k + 1, len(y_train)))
    nn = NearestNeighbors(n_neighbors=k_eff).fit(X_train)
    _, idx = nn.kneighbors(X_train)
    return np.mean(y_train[idx[:, 1:]], axis=1)


def build_arm(arm, X_train, y_train, scores, true_spread_train, n_flag, rng):
    """The training rows and labels one arm fits on.

    Returns (X, y, n_removed) or None when the arm cannot be built.
    """
    n = len(y_train)
    if arm == 'no_cleaning':
        return X_train, y_train, 0

    if arm in ('drop', 'repair'):
        if scores is None:
            return None
        flagged = np.zeros(n, dtype=bool)
        flagged[np.argsort(-scores, kind='stable')[:n_flag]] = True
        if arm == 'drop':
            keep = ~flagged
            if keep.sum() < 10:
                return None
            return X_train[keep], y_train[keep], int(flagged.sum())
        y_rep = y_train.copy()
        y_rep[flagged] = neighbour_means(X_train, y_train)[flagged]
        return X_train, y_rep, int(flagged.sum())

    if arm == 'ceiling':
        # The ceiling removes the entities that really do disagree most. An
        # entity measured once has no spread, so it can never be one of them --
        # ranked last rather than dropped, so the count still matches the others.
        sp = np.where(np.isfinite(true_spread_train), true_spread_train, -np.inf)
        worst = np.zeros(n, dtype=bool)
        worst[np.argsort(-sp, kind='stable')[:n_flag]] = True
        keep = ~worst
        if keep.sum() < 10:
            return None
        return X_train[keep], y_train[keep], int(worst.sum())

    if arm == 'floor':
        keep = np.ones(n, dtype=bool)
        keep[rng.choice(n, size=min(n_flag, n), replace=False)] = False
        if keep.sum() < 10:
            return None
        return X_train[keep], y_train[keep], int((~keep).sum())

    raise ValueError(f"unknown arm {arm!r}")


def run(data, config, output_dir, dataset_name, **kwargs):
    """N17: removal on real repeat-measurement disagreement."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N17: DOES CLEANING HELP WHEN THE NOISE IS REAL")
    print("=" * 70)

    labels = np.asarray(data['labels'], dtype=np.float64)
    reps = data['reps']
    task_type = data['task_type']
    if task_type != 'regression':
        # A repeat measurement of a class label either agrees or does not, so a
        # spread in the label's own units does not exist and neither does a
        # consensus mean.
        print(f"  SKIPPED: {dataset_name} is {task_type}. This experiment needs "
              f"repeat measurements on a continuous label.")
        return {'skipped': 'not_regression', 'task_type': task_type}

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    rep_groups = build_replicate_groups(data)
    if rep_groups is None:
        # Not a failure of this experiment. Most loaders collapse their repeat
        # measurements before anything downstream sees them, so there is nothing
        # here to hold out. NOISE_AUDIT_2026_09_10.md section 11.1.
        print(f"  SKIPPED: {dataset_name}'s loader exposes no repeat structure. "
              f"Neither data['replicate_groups'] nor metadata drug/protein ids "
              f"are present, so there is no second measurement to hold out.")
        return {'skipped': 'no_replicate_structure', 'dataset': dataset_name}

    downstream_models = list(kwargs.get('models') or DOWNSTREAM_MODELS)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n17: no downstream model could be built. Asked for "
            f"{list(kwargs.get('models') or DOWNSTREAM_MODELS)}, got "
            f"{sorted(model_factories)}.")

    n_draws = int(kwargs.get('n_draws', N_DRAWS))
    base_rep_name = kwargs.get('base_rep') or sorted(reps)[0]
    X_all = np.asarray(reps[base_rep_name], dtype=np.float64)

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_regime': 'real_replicate_disagreement',
        'noise_injection': 'none',
        'base_rep': base_rep_name,
        'models': downstream_models,
        'arms': list(ARMS),
        'detector': DETECTOR,
        'detector_k': DETECTOR_K,
        'flag_fraction': FLAG_FRACTION,
        'min_measurements_to_grade': MIN_MEASUREMENTS,
        'n_draws': n_draws,
        'claim_scope': (
            'Training labels are ONE real measurement per entity; grading labels '
            'are the mean of that entity\'s OTHER measurements. No noise is '
            'injected anywhere. A removal result is drop or repair beating '
            'no_cleaning, read against the distance between floor and ceiling.'),
        'per_draw': {},
    }

    # ── Feasibility, printed before anything is fitted ──
    probe = build_entities(labels, rep_groups, seed=0)
    n_ent = len(probe['n_measurements'])
    n_grade = int(probe['gradeable'].sum())
    counts = {int(k): int((probe['n_measurements'] >= k).sum())
              for k in (2, 3, 4, 5)}
    results['feasibility'] = {
        'n_rows': int(len(labels)),
        'n_entities': n_ent,
        'n_entities_gradeable': n_grade,
        'entities_by_min_measurements': counts,
        'mean_true_spread_label_units': float(
            np.nanmean(probe['true_spread'])) if n_grade else float('nan'),
    }
    print(f"  rows: {len(labels)}, entities: {n_ent}")
    for k, v in counts.items():
        print(f"    entities with {k}+ measurements: {v} of {n_ent} "
              f"({100.0 * v / max(n_ent, 1):.2f}%)")
    print(f"  gradeable entities (>= {MIN_MEASUREMENTS} measurements): {n_grade}")
    if n_grade < 30:
        print(f"  SKIPPED: {n_grade} gradeable entities is too few to split into "
              f"{n_folds} folds and read a difference between five arms.")
        results['skipped'] = 'too_few_gradeable_entities'
        return results

    # ── The experiment ──
    for draw in range(n_draws):
        t_draw = time.time()
        print(f"\n  === draw {draw} (which measurement trains, chosen afresh) ===")
        ent = build_entities(labels, rep_groups, seed=draw)
        rng = np.random.default_rng(1000 + draw)

        X_ent = X_all[ent['row_index']]
        y_train_all = ent['train_label']
        y_grade_all = ent['grade_label']
        gradeable = ent['gradeable']
        spread_all = ent['true_spread']

        # Fold geometry over ENTITIES. data['groups'] is per row, so it is read
        # at each entity's representative row.
        row_groups = data.get('groups')
        ent_groups = (np.asarray(row_groups)[ent['row_index']]
                      if row_groups is not None else None)

        # {arm: {model: [fold scores]}}
        acc = {a: {m: [] for m in downstream_models} for a in ARMS}
        removed = {a: [] for a in ARMS}
        fold_sizes = []

        for fold_i, (tr, va) in enumerate(
                cv_split(ent_groups, n_folds, X_ent, y_train_all, mode=mode)):
            va_graded = va[gradeable[va]]
            if len(va_graded) < 5:
                print(f"    fold {fold_i}: only {len(va_graded)} gradeable "
                      f"entities held out, skipped")
                continue
            fold_sizes.append(int(len(va_graded)))

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_ent[tr])
            X_va = scaler.transform(X_ent[va_graded])
            y_tr = y_train_all[tr]
            y_va = y_grade_all[va_graded]

            n_flag = max(1, int(round(FLAG_FRACTION * len(tr))))
            scores = detector_scores(X_tr, y_tr)

            for arm in ARMS:
                built = build_arm(arm, X_tr, y_tr, scores,
                                  spread_all[tr], n_flag, rng)
                if built is None:
                    continue
                Xa, ya, n_rm = built
                removed[arm].append(n_rm)
                for mname in downstream_models:
                    try:
                        m = model_factories[mname]()
                        m.fit(Xa, ya)
                        acc[arm][mname].append(
                            float(metric_fn(y_va, m.predict(X_va))))
                    except Exception as exc:
                        print(f"      {arm}/{mname} fold {fold_i} failed: "
                              f"{type(exc).__name__}: {exc}")

        draw_block = {
            'n_folds_scored': len(fold_sizes),
            'gradeable_entities_per_fold': fold_sizes,
            'per_arm': {},
        }
        for arm in ARMS:
            draw_block['per_arm'][arm] = {
                'entities_removed_per_fold': removed[arm],
                'per_model': {
                    m: {'mean': float(np.mean(v)) if v else float('nan'),
                        'fold_scores': [float(x) for x in v],
                        'n_folds': len(v)}
                    for m, v in acc[arm].items()},
            }
        results['per_draw'][str(draw)] = convert_numpy(draw_block)
        print(f"    [{time.time() - t_draw:.0f}s]")

    results['summary'] = _summarise(results, downstream_models,
                                    metric_name, lower_is_better)
    _print_summary(results['summary'], metric_name)

    save_experiment_results(results, 'n17_replicate_removal', dataset_name,
                            output_dir)
    return results


def _summarise(results, models, metric_name, lower_is_better):
    """One row per model: each arm's score, and where the two detector arms sit.

    `removal_position` is where an arm sits between floor and ceiling. 0.0 is
    level with the floor -- the gain is training-set size and nothing else. 1.0 is
    level with the ceiling -- the detector found what the repeat measurements say
    is actually worst. It is left undefined when floor and ceiling are within a
    hair of each other, because there is then no distance to be a fraction of.
    """
    draws = list(results['per_draw'].values())
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'per_model': {}}
    for m in models:
        arm_means = {}
        for arm in ARMS:
            vals = [d['per_arm'][arm]['per_model'][m]['mean'] for d in draws
                    if np.isfinite(d['per_arm'][arm]['per_model'][m]['mean'])]
            arm_means[arm] = {
                'mean_over_draws': float(np.mean(vals)) if vals else float('nan'),
                # The spread across draws is the spread over WHICH measurement
                # became the training label. A difference between arms smaller
                # than this is not a difference between arms.
                'std_over_draws': float(np.std(vals)) if vals else float('nan'),
                'per_draw': vals,
                'n_draws_scored': len(vals),
            }
        base = arm_means['no_cleaning']['mean_over_draws']
        floor = arm_means['floor']['mean_over_draws']
        ceil = arm_means['ceiling']['mean_over_draws']

        def _gain(x):
            # Positive always means better, whichever way the metric runs.
            if not (np.isfinite(x) and np.isfinite(base)):
                return float('nan')
            return float(base - x) if lower_is_better else float(x - base)

        span = _gain(ceil) - _gain(floor)
        row = {
            'arm_scores': arm_means,
            'gain_over_no_cleaning': {a: _gain(arm_means[a]['mean_over_draws'])
                                      for a in ARMS},
            'floor_to_ceiling_span': float(span) if np.isfinite(span) else float('nan'),
        }
        for arm in ('drop', 'repair'):
            g = _gain(arm_means[arm]['mean_over_draws'])
            row.setdefault('removal_position', {})[arm] = (
                float((g - _gain(floor)) / span)
                if np.isfinite(span) and abs(span) > 1e-12 and np.isfinite(g)
                else float('nan'))
        out['per_model'][m] = row
    return out


def _print_summary(summary, metric_name):
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric_name}; positive gain = better than no cleaning)")
    for m, row in summary['per_model'].items():
        g = row['gain_over_no_cleaning']
        pos = row.get('removal_position', {})
        print(f"  {m}:")
        print(f"    floor {g['floor']:+.4f}   ceiling {g['ceiling']:+.4f}   "
              f"span {row['floor_to_ceiling_span']:+.4f}")
        for arm in ('drop', 'repair'):
            p = pos.get(arm, float('nan'))
            where = 'undefined' if not np.isfinite(p) else f'{p:.0%} of the way'
            print(f"    {arm:8s} {g[arm]:+.4f}   {where} from floor to ceiling")
    print("  A gain at the floor is a smaller training set, not a removal result.")


def main():
    parser = create_experiment_parser('N17: removal on real repeat disagreement')
    parser.add_argument('--n-draws', type=int, default=N_DRAWS,
                        help='How many times the training measurement is re-drawn')
    args = parser.parse_args()
    config = get_config(args.mode)
    data = load_data(args.dataset, rep_preset=args.reps)
    run(data, config, args.output, args.dataset, n_draws=args.n_draws)


if __name__ == '__main__':
    main()
