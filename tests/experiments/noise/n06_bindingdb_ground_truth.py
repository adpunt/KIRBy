#!/usr/bin/env python3
"""N06: does an estimator's idea of a noisy molecule match a real disagreement?

Every other experiment in the noise suite injects the noise it then looks for.
n11 corrupts training labels at six levels and asks which detector recovers the
most; n13 does the same across ten noise shapes. Both answer "do our methods
find the noise we put there".

n06 is the one where the noise is real. BindingDB holds the same drug measured
against the same protein by different labs, and the spread of their pKd values
is disagreement nobody chose. Lhasa and ChEMBL both curated hERG, and where
they call the same molecule differently, one of them is wrong.

WHAT IT REPORTS, in order of what it is for:

  1. The transfer question. Estimators ranked under injected noise by n11 and
     n13, ranked again here on real disagreement, and the rank correlation
     between the two orderings. This is the headline; n06 does not add a
     twelfth opinion about which detector is best.
  2. The per-estimator table behind that ranking, one row per (space,
     estimator), with the number of replicate GROUPS behind each — never the
     row count, which counts one group's ground truth once per repeat.
  3. The hybrid minus the single representation, per estimator: does an
     estimator agree better in a space whose features were chosen, than in one
     that had no choice made for it.

WHERE THE GROUND TRUTH COMES FROM. A ground-truth provider says which molecules
form a repeat group and how much the repeats disagree; `_ground_truth_providers`
holds the three that exist and the floor below which a provider refuses. A
dataset where no provider finds a repeat structure gets a recorded skip with
the count it found — Davis and KIBA both look like candidates until the
repeated pairs are counted, and both have none.

HOW A MOLECULE IS SCORED. The rows are split into protein-disjoint folds. Each
fold's feature space is built from the OTHER folds' rows, so no molecule helps
choose the features its own noise score is computed in, and the estimator is
fitted inside that fold, in that fold's own space. Nothing is stacked across
folds: five folds' blocks glued into one matrix put column 7 of one row and
column 7 of another in different features.

Usage:
    python tests/experiments/run_dataset.py noise bindingdb_kd --experiments n06
    python n06_bindingdb_ground_truth.py --dataset bindingdb_kd
"""
import sys
import os
import inspect
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_best_search_for_downstream, build_fold_hybrid, make_cv_splitter,
    extract_top3_base_reps, pick_noise_base_rep, get_domain_metric,
)
from kirby.utils.noise_estimation import get_noise_estimator, ESTIMATOR_REGISTRY

from _ground_truth_providers import (  # noqa: E402
    collect_ground_truths, agreement, GROUP_SPREAD_DDOF,
    # Re-exported: n14 imports both of these from n06, and the smoke tests
    # address them there too. They lived here until 2026-09-11.
    build_replicate_groups, compute_replicate_stds,  # noqa: F401
    MIN_GROUND_TRUTH_GROUPS,  # noqa: F401
)

# The estimator list is n11's, imported rather than copied. n06's orderings are
# compared with n11's, and two hand-kept lists drift: the version of this file
# before 2026-09-11 ran `leave_one_out`, which n11 does not, and omitted
# `hybrid`, which n11 does, so the two files could not be joined even where
# they had run the same code.
from n11_estimation_remediation_grid import (  # noqa: E402
    BASE_ESTIMATORS, TORCH_ESTIMATORS, OPTIONAL_ESTIMATORS,
    NEIGHBOR_ESTIMATORS, IS_CLF_AWARE_ESTIMATORS, GAP_MIN_FRAC,
)

# How many rows gaussian_process is given. It solves an n x n system, so the
# work grows with the cube of the row count: 9,676 rows is about 900 times the
# work of 1,000. Rows above this are dropped a WHOLE GROUP at a time, so no
# group is left holding one member and a spread of zero.
GP_MAX_ROWS = 1000
GP_SUBSAMPLED_ESTIMATORS = frozenset({'gaussian_process'})

N_FOLDS = 5


def _canonical(name):
    """The one name for an estimator's class.

    ESTIMATOR_REGISTRY holds 19 keys for 15 classes (counted 2026-09-11;
    an earlier note said 17 for 13): 'distance_weighted' and
    'distance_weighted_residual' are the same class, so are 'cross_val' /
    'cross_val_residual', 'ensemble' / 'ensemble_disagreement' and 'hybrid' /
    'hybrid_noise'. n06 spelled them one way and n11 the other, so two result
    files did not join on name even though the same code had run. Everything
    here resolves through the registry; there is no hand-written mapping to go
    stale.
    """
    cls = ESTIMATOR_REGISTRY.get(name)
    if cls is None:
        return name
    names = [k for k, v in ESTIMATOR_REGISTRY.items() if v is cls]
    return sorted(names, key=lambda s: (len(s), s))[0]


def _strip_metric_suffix(est_key):
    """'neighbor_residual_cosine' -> 'neighbor_residual'.

    n11 suffixes the distance metric onto the estimator key for the
    neighbour-based estimators, because it sweeps the metric. n06 does not, so
    the suffix has to come off before the two can be joined. The metric names
    are read from the distance configuration rather than guessed.
    """
    try:
        from kirby.similarity.distance_config import DISTANCE_METRICS
        metrics = sorted(DISTANCE_METRICS, key=len, reverse=True)
    except Exception:
        metrics = ['euclidean', 'cosine', 'manhattan']
    for m in metrics:
        if est_key.endswith(f'_{m}'):
            return est_key[: -len(m) - 1], m
    return est_key, None


def resolve_estimators(is_clf=False):
    """The estimators n06 runs, resolved exactly the way n11 resolves them.

    Torch-dependent and optional estimators are included when they import, so
    a machine with torch runs the same set n11 runs there. On a classification
    dataset only the estimators that have a classification branch are kept —
    the rest fit a regressor on the integer class codes, which does not crash
    and means nothing.
    """
    if os.environ.get('KIRBY_NOISE_LIGHT_ESTIMATORS') == '1':
        available = [e for e in BASE_ESTIMATORS if e in NEIGHBOR_ESTIMATORS]
    else:
        available = list(BASE_ESTIMATORS)
        try:
            import torch  # noqa: F401
            available.extend(TORCH_ESTIMATORS)
        except ImportError:
            pass
        for opt in OPTIONAL_ESTIMATORS:
            try:
                get_noise_estimator(opt)
                available.append(opt)
            except Exception:
                pass
    if is_clf:
        available = [e for e in available if e in IS_CLF_AWARE_ESTIMATORS]
    return available


def union_exclusion_groups(ground_truths, n):
    """One label per row: the rows it must never be scored from.

    What n06 correlates an estimator's score against is the spread of a row's
    own repeat measurements, and those repeats are identical rows in every
    feature space — so they are the row's own nearest neighbours, and an
    out-of-bag path leaves a row out of some trees but never leaves out its
    repeats. Every estimator that accepts `exclude_groups` is given this, so no
    row is scored from the labels it is graded against.

    Rows grouped together by ANY provider are merged, so one exclusion vector
    covers every ground truth in the run and the scores are computed once.
    """
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for gt in ground_truths:
        for gid, rows in gt.rows.items():
            if gid < 0 or len(rows) < 2:
                continue
            first = int(rows[0])
            for r in rows[1:]:
                union(first, int(r))
    return np.array([find(i) for i in range(n)], dtype=np.int64)


def _cap_whole_groups(rows, exclusion, cap, rng):
    """Keep at most `cap` rows, dropping a whole group at a time.

    THE REPEAT GROUPS ARE TAKEN FIRST. Most rows of a dataset are measured
    once, and one of those rows carries no ground truth at all — it is there so
    the molecules that do have one have neighbours. Filling the cap from every
    row on equal terms spends it almost entirely on rows that cannot be scored:
    measured on BindingDB, a 1,000-row cap drawn that way left gaussian_process
    with 45 of the 1,502 replicate groups, below the floor a correlation needs,
    so it wrote no number at all and left the comparison without ever
    appearing as a failure. Taking the repeated groups first leaves it 340.

    Dropping individual rows would leave a group holding one member, and one
    member has no spread, so a whole group is kept or none of it.

    Returns (positions kept, how many REPEAT groups were dropped, how many
    there were).
    """
    from _ground_truth_providers import rows_by_group

    index = rows_by_group(np.asarray(exclusion))
    repeated = [g for g, m in index.items() if len(m) > 1]
    if len(rows) <= cap:
        return np.arange(len(rows)), 0, len(repeated)

    single = [g for g, m in index.items() if len(m) == 1]
    order = list(rng.permutation(repeated)) + list(rng.permutation(single))
    keep_mask = np.zeros(len(rows), dtype=bool)
    n_kept_rows = 0
    for gid in order:
        member = index[int(gid)]
        if n_kept_rows + len(member) > cap:
            continue
        keep_mask[member] = True
        n_kept_rows += len(member)
    if not keep_mask.any():
        # One group alone is larger than the cap. Taking part of it would leave
        # its other members outside and make its spread a different number, so
        # the smallest whole group is kept and the cap is exceeded knowingly.
        keep_mask[min(index.values(), key=len)] = True
    kept_repeated = sum(1 for g in repeated if keep_mask[index[g][0]])
    return (np.flatnonzero(keep_mask),
            int(len(repeated) - kept_repeated), len(repeated))


# ── feature spaces ─────────────────────────────────────────────────────────

def _clean_matrix(X):
    return np.nan_to_num(np.clip(np.asarray(X, dtype=np.float64), -1e10, 1e10),
                         nan=0.0, posinf=1e10, neginf=-1e10)


def _blueprint_for(reps, labels, groups, data, dataset_name, task_type,
                   augmentations, budget, objective=None):
    """The recipe a search picked, and which pool entry it came from.

    Returns (blueprint, aug_condition, metadata, search) or
    (None, None, reason, None). `objective='similarity'` asks for the recipe
    whose features were chosen by distance rather than by prediction — the
    search resolves that from its own objective-namespaced cache, and builds
    one if that cache is cold, which is a full search and is not free.
    """
    try:
        search, meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model='rf', budget=budget,
            augmentations=augmentations,
            data=(data if objective is None else None),
            dataset_name=dataset_name,
            task_type=task_type,
            return_fitted=False,
            objective=objective,
        )
    except Exception as exc:
        return None, None, f'{type(exc).__name__}: {exc}', None
    if not meta:
        return None, None, 'no pool candidate produced a result for rf', None
    bp = meta.get('best_config', {}).get('blueprint')
    if bp is None:
        return None, None, 'the winning pool entry carries no blueprint', None
    return bp, meta.get('aug_condition', 'uncapped'), meta, search


def _fold_hybrid_pair(train_idx, val_idx, reps, labels, groups, bp, aug_cond,
                      augmentations):
    """This fold's rows in this fold's own space: (training rows, held-out rows).

    Both sides are returned. The version of this file before 2026-09-11 threw
    the training side away, stacked the five folds' held-out blocks into one
    matrix and fitted the estimator on that — which put rows from different
    folds in different features under one column number.
    """
    augs = augmentations if aug_cond != 'none' else None
    train_reps = {k: v[train_idx] for k, v in reps.items()}
    val_reps = {k: v[val_idx] for k, v in reps.items()}
    train_augs = ({k: v[train_idx] for k, v in augs.items()} if augs else None)
    val_augs = ({k: v[val_idx] for k, v in augs.items()} if augs else None)
    X_tr, X_va, _ = build_fold_hybrid(
        train_reps, val_reps, labels[train_idx], bp,
        augmentations_train=train_augs, augmentations_val=val_augs,
        groups=(np.asarray(groups)[train_idx] if groups is not None else None))
    return _clean_matrix(X_tr), _clean_matrix(X_va)


# ── the stored orderings this is compared against ──────────────────────────

def n11_ranking(path):
    """How n11 ordered the estimators on this dataset, from its result file.

    n11 stores, for every (feature space, estimator, remediation strategy,
    strength) cell, `recovery` — the share of the damage the injected noise did
    to the model score that the cleanup put back, as a percentage. An
    estimator's number here is the MEDIAN of its cells, not the largest: the
    largest of about 4,800 cells is partly luck and n11's own file says so.

    Returns {canonical estimator name: median recovery, ...} and the cell
    count behind each.
    """
    import json
    with open(path) as fh:
        doc = json.load(fh)
    lower_is_better = bool(doc.get('lower_is_better', False))
    cells = {}
    per_space = {}
    dropped = {'sigma_whose_noise_did_not_hurt_the_model': 0,
               'cells_with_no_finite_recovery': 0}
    for sigma, sigma_data in (doc.get('per_sigma') or {}).items():
        if str(sigma) == '0.0':
            continue  # no noise was injected, so there is nothing to recover
        # The recovery is a share of the gap between the clean-label and the
        # noisy-label model score. When the injected noise did not make the
        # model worse, that gap runs the wrong way and the share reads
        # backwards — a cleanup that did nothing scores well. Whole sigma
        # dropped, and counted.
        gap = sigma_data.get('gap')
        if gap is None or not np.isfinite(gap) or (
                (gap > 0) if lower_is_better else (gap < 0)):
            dropped['sigma_whose_noise_did_not_hurt_the_model'] += 1
            continue
        for space, space_data in (sigma_data.get('per_space') or {}).items():
            for est_key, est_data in (space_data or {}).items():
                base, _metric = _strip_metric_suffix(est_key)
                name = _canonical(base)
                for remed_key, rv in (est_data or {}).items():
                    if remed_key.startswith('_') or not isinstance(rv, dict):
                        continue
                    r = rv.get('recovery')
                    if r is None or not np.isfinite(r):
                        dropped['cells_with_no_finite_recovery'] += 1
                        continue
                    cells.setdefault(name, []).append(float(r))
                    per_space.setdefault(space, {}).setdefault(
                        name, []).append(float(r))
    ranking = {k: float(np.median(v)) for k, v in cells.items()}
    return (ranking, {k: len(v) for k, v in cells.items()},
            doc.get('timestamp'),
            {'per_n11_feature_space': {
                sp: {k: float(np.median(v)) for k, v in d.items()}
                for sp, d in per_space.items()},
             'dropped': dropped,
             'n_feature_spaces_pooled': len(per_space)})


def n13_ranking(path):
    """How n13 ordered the estimators on this dataset, from its result file.

    n13's track_b holds, per noise shape and strength, per feature space, per
    estimator, per downstream model, a `recovery` percentage — the same
    quantity n11 reports. The median over those cells is the estimator's
    number.
    """
    import json
    with open(path) as fh:
        doc = json.load(fh)
    lower_is_better = bool(doc.get('lower_is_better', False))
    cells = {}
    dropped = {'cells_where_the_cleanup_changed_nothing': 0,
               'cells_whose_noise_did_not_hurt_the_model': 0,
               'cells_with_no_finite_recovery': 0}
    for _shape, shape_data in (doc.get('per_noise_type') or {}).items():
        for _strength, s_data in (shape_data.get('per_strength') or {}).items():
            noisy = s_data.get('noisy_score_by_model') or {}
            clean = s_data.get('clean_score_by_model') or {}
            for _space, space_data in (s_data.get('track_b') or {}).items():
                for est_key, per_model in (space_data or {}).items():
                    base, _m = _strip_metric_suffix(est_key)
                    name = _canonical(base)
                    for model, rv in (per_model or {}).items():
                        if not isinstance(rv, dict):
                            continue
                        r = rv.get('recovery')
                        if r is None or not np.isfinite(r):
                            dropped['cells_with_no_finite_recovery'] += 1
                            continue
                        n_s, c_s = noisy.get(model), clean.get(model)
                        # A model whose fit ignores sample weights was refitted
                        # unchanged, so its denoised score IS its noisy score
                        # and the cell records a cleanup that never happened.
                        # n13 stopped writing these on 2026-06; files from
                        # before that still carry them.
                        if (n_s is not None and rv.get('denoised') is not None
                                and float(rv['denoised']) == float(n_s)):
                            dropped['cells_where_the_cleanup_changed_'
                                    'nothing'] += 1
                            continue
                        # Same wrong-way gap as n11, but n13 divides per model,
                        # so it has to be checked per model.
                        if n_s is not None and c_s is not None:
                            gap = float(c_s) - float(n_s)
                            if (gap > 0) if lower_is_better else (gap < 0):
                                dropped['cells_whose_noise_did_not_hurt_'
                                        'the_model'] += 1
                                continue
                            # n11 refuses a recovery whose denominator is a
                            # sliver of the clean score, because dividing by it
                            # turns a rounding difference into a percentage in
                            # the hundreds. n13 only refuses a denominator of
                            # almost exactly zero, so the same floor is applied
                            # here — otherwise the two orderings n06 compares
                            # were filtered by different rules.
                            if abs(gap) <= max(1e-10,
                                               GAP_MIN_FRAC * abs(float(c_s))):
                                dropped.setdefault(
                                    'cells_whose_gap_is_below_n11s_floor', 0)
                                dropped['cells_whose_gap_is_below_'
                                        'n11s_floor'] += 1
                                continue
                        cells.setdefault(name, []).append(float(r))
    return ({k: float(np.median(v)) for k, v in cells.items()},
            {k: len(v) for k, v in cells.items()},
            doc.get('timestamp'), {'dropped': dropped})


STORED_RANKINGS = {
    'n11': ('n11_estimation_remediation_grid', n11_ranking),
    'n13': ('n13_noise_distributions', n13_ranking),
}

# A rank correlation needs more than a handful of points to mean anything. n13
# scores three estimators, so its ordering cannot be rank-correlated with
# n06's; the count is written out and the correlation is refused rather than
# reported over three points.
MIN_ESTIMATORS_FOR_TRANSFER = 5


def transfer_to_injected_noise(n06_scores, output_dir, dataset_name):
    """Does the ordering measured under injected noise survive on real error?

    n06_scores: {space: {estimator: number, ...}} — higher means it agrees
    better with the real disagreement. The stored files order the same
    estimators by how much of the injected noise's damage a cleanup put back.
    A positive rank correlation means the two orderings agree.

    Every case where the comparison cannot be made is recorded with its reason
    and its count, not dropped.
    """
    out = {}
    for label, (subdir, reader) in STORED_RANKINGS.items():
        path = os.path.join(output_dir, subdir, f'{dataset_name}.json')
        entry = {'read_from': path, 'same_dataset': True}
        if not os.path.exists(path):
            entry['refused'] = (f'no stored {label} result for {dataset_name} '
                                f'at {path}')
            out[label] = entry
            continue
        try:
            ranking, cell_counts, stamp, extra = reader(path)
        except Exception as exc:
            entry['refused'] = f'{type(exc).__name__}: {exc}'
            out[label] = entry
            continue
        entry['written_at'] = stamp
        entry['ranking'] = ranking
        entry['cells_behind_each_estimator'] = cell_counts
        entry.update(extra or {})
        entry['what_the_stored_number_is'] = (
            "the median, over that estimator's cells, of the percentage of "
            "the gap between the clean-label and the noisy-label model score "
            "that the cleanup put back. The cells pool every feature space, "
            "every remediation rule, every strength and — in n13 — every "
            "downstream model that file scored, so one estimator's number is "
            "its middle cell across all of them, not its best. Cells where "
            "the cleanup changed nothing, and whole noise levels where the "
            "injected noise did not make the model worse, are dropped and "
            "counted under 'dropped'.")
        entry['per_space'] = {}
        for space, per_est in n06_scores.items():
            common = sorted(set(ranking) & set(per_est))
            common = [c for c in common
                      if per_est[c] is not None and np.isfinite(per_est[c])]
            row = {'n_estimators_in_common': len(common),
                   'estimators_in_common': common}
            if len(common) < MIN_ESTIMATORS_FOR_TRANSFER:
                row['rank_correlation'] = None
                row['refused'] = (
                    f'{len(common)} estimators are in both orderings, below '
                    f'{MIN_ESTIMATORS_FOR_TRANSFER}')
            else:
                a = [per_est[c] for c in common]
                b = [ranking[c] for c in common]
                rho, p = spearmanr(a, b)
                row['rank_correlation'] = float(rho)
                row['p'] = float(p)
            entry['per_space'][space] = row
        out[label] = entry
    return out


# ── the run ────────────────────────────────────────────────────────────────

def run(data, config, output_dir, dataset_name, **kwargs):
    """N06: estimator scores against a disagreement nobody injected."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N06: GROUND-TRUTH NOISE")
    print("=" * 70)

    reps = data['reps']
    labels = np.asarray(data['labels'], dtype=np.float64)
    n_samples = len(labels)
    task_type = data['task_type']
    is_clf = (task_type == 'classification')
    augmentations = data.get('augmentations')
    search_groups = data.get('groups')
    metric_name, _metric_fn, lower_is_better = get_domain_metric(data)

    # Every result file carries these two. A file without them is dropped by
    # the result database with no error, and a whole job has been lost to it.
    results = {
        'metric': metric_name,
        'task_type': task_type,
        'lower_is_better': bool(lower_is_better),
        'n_samples': int(n_samples),
        'group_spread_ddof': GROUP_SPREAD_DDOF,
    }

    # ── what disagrees, and by how much ──
    ground_truths, provider_report = collect_ground_truths(data, dataset_name)
    results['providers'] = convert_numpy(provider_report)
    if not ground_truths:
        reason = ('no ground-truth provider found a repeat structure in '
                  f'{dataset_name}')
        print(f"  SKIP: {reason}")
        results.update({'skipped': True, 'reason': reason,
                        'policy': 'no_ground_truth_provider'})
        return results

    exclusion = union_exclusion_groups(ground_truths, n_samples)
    estimators = resolve_estimators(is_clf=is_clf)
    results['estimators_run'] = list(estimators)
    results['estimator_list_source'] = (
        'n11_estimation_remediation_grid.BASE_ESTIMATORS, imported so the two '
        'files cannot drift, plus the torch and optional estimators n11 adds '
        'when they import')
    print(f"  {len(estimators)} estimators: {', '.join(estimators)}")

    # On a classification dataset only the estimators with a class-aware branch
    # are kept, and two of those five names are the same class, so four
    # distinct estimators run. A rank correlation over four points is not a
    # number worth reading, so n06's headline cannot exist there. Said here
    # rather than left for a reader to work out from a refusal further down.
    n_distinct = len({_canonical(e) for e in estimators})
    if n_distinct < MIN_ESTIMATORS_FOR_TRANSFER:
        results['transfer_cannot_be_computed'] = (
            f'{n_distinct} distinct estimators can run on a {task_type} '
            f'dataset, below the {MIN_ESTIMATORS_FOR_TRANSFER} a rank '
            f'correlation needs. The per-estimator table below is still '
            f'measured; the ordering comparison with n11 and n13 is not.')
        print(f"  {results['transfer_cannot_be_computed']}")

    # ── feature spaces ──
    # The comparator is ONE base representation, never every representation
    # glued together: a concatenation is not a baseline in this project. Which
    # one: the best by random forest from the same search that chose the
    # hybrid's recipe, so both arms are ranked by one model on one set of folds.
    budget = kwargs.get('budget')
    spaces = {}
    space_provenance = {}

    bp, aug_cond, meta, search = _blueprint_for(
        reps, labels, search_groups, data, dataset_name, task_type,
        augmentations, budget, objective=None)
    if bp is not None:
        spaces['hybrid_predictive'] = ('hybrid', bp, aug_cond)
        space_provenance['hybrid_predictive'] = {
            'features_chosen_by': 'prediction',
            'which_features': "chosen inside each fold, on the other folds' "
                              "rows",
            'which_recipe': (
                'chosen ONCE, by a search over every row — the budget, the '
                'allocation method, the importance scorer and which '
                'representations are in the pool are not re-chosen per fold. '
                'The comparator representation was picked off the same '
                'search. So the recipe is common to both arms and every '
                'fold, and only the features it selects are per fold.'),
            'tier': meta.get('tier'), 'aug_condition': aug_cond,
            'budget': int(getattr(bp, 'budget', 0) or 0),
            'allocation_method': getattr(bp, 'allocation_method', None),
            'importance_method': getattr(bp, 'importance_method', None),
        }
    else:
        space_provenance['hybrid_predictive'] = {'unavailable': meta}
        print(f"  no predictive hybrid recipe: {meta}")

    if kwargs.get('similarity_arm', True):
        t_sim = time.time()
        sbp, saug, smeta, _ssearch = _blueprint_for(
            reps, labels, search_groups, data, dataset_name, task_type,
            augmentations, budget, objective='similarity')
        if sbp is not None:
            spaces['hybrid_similarity'] = ('hybrid', sbp, saug)
            space_provenance['hybrid_similarity'] = {
                'features_chosen_by': 'distance between molecules, not '
                                      'prediction of the label',
                'which_features': "chosen inside each fold, on the other "
                                  "folds' rows",
                'which_recipe': 'chosen once, by a search over every row, '
                                'under the similarity objective',
                'tier': smeta.get('tier'), 'aug_condition': saug,
                'budget': int(getattr(sbp, 'budget', 0) or 0),
                'allocation_method': getattr(sbp, 'allocation_method', None),
                'importance_method': getattr(sbp, 'importance_method', None),
                'seconds_to_resolve_the_search': round(time.time() - t_sim, 1),
            }
        else:
            space_provenance['hybrid_similarity'] = {'unavailable': smeta}
            print(f"  no similarity hybrid recipe: {smeta}")
    else:
        space_provenance['hybrid_similarity'] = {
            'unavailable': 'the similarity arm was switched off for this run'}

    base_rep = None
    if search is not None:
        try:
            top3 = extract_top3_base_reps(search)
            if top3:
                results['top3_base_reps'] = [(n, sc) for n, sc in top3]
                if top3[0][0] in reps:
                    base_rep = top3[0][0]
        except Exception as exc:
            print(f"  no base-representation ranking off the search "
                  f"({type(exc).__name__}: {exc})")
    if base_rep is None:
        base_rep = pick_noise_base_rep(reps, dataset_name)
        print(f"  comparator: {base_rep} (preference order; the search gave "
              f"no usable ranking)")
    else:
        print(f"  comparator: {base_rep} (best single representation by rf)")
    spaces[base_rep] = ('base', None, None)
    space_provenance[base_rep] = {
        'features_chosen_by': 'nothing — every feature of one representation',
        'which_representation': (
            'the best single representation under a random forest in the same '
            'search that chose the hybrid recipe, so both arms are ranked by '
            'one model on one set of folds'),
        'scaling': 'z-scored inside each fold on that fold\'s training rows, '
                   'the same treatment the hybrid arm gets',
    }
    results['comparator_space'] = base_rep
    results['space_provenance'] = convert_numpy(space_provenance)

    # ── score every molecule, inside its own fold ──
    splitter, use_groups = make_cv_splitter(N_FOLDS, search_groups,
                                            task_type=task_type)
    sample_X = next(iter(reps.values()))
    split_args = ((sample_X, labels, search_groups) if use_groups
                  else (sample_X, labels))
    folds = list(splitter.split(*split_args))
    print(f"  {len(folds)} folds, split so no protein group sits on both sides")

    # A group whose rows land in two different folds has its rows scored in two
    # different feature spaces, and the group's mean score then mixes them.
    # On BindingDB it cannot happen — a group is one drug against one protein,
    # and the folds are split by protein — but nothing enforces that, so it is
    # counted rather than assumed.
    fold_of_row = np.full(n_samples, -1, dtype=np.int64)
    for fold_i, (_tr, va) in enumerate(folds):
        fold_of_row[va] = fold_i
    straddling = {}
    for gt in ground_truths:
        n_straddle = sum(1 for gid in gt.group_value
                         if len(np.unique(fold_of_row[gt.rows[int(gid)]])) > 1)
        straddling[gt.provider] = n_straddle
        if n_straddle:
            print(f"  {n_straddle} of {gt.n_groups} {gt.provider} groups have "
                  f"rows in more than one fold, so their rows were scored in "
                  f"different feature spaces")
    results['groups_whose_rows_span_more_than_one_fold'] = straddling

    # NOT scaled here. Both arms are z-scored inside each fold, on that fold's
    # training rows only. Scaling the comparator once over every row and the
    # hybrid per fold would put a difference in treatment inside the
    # subtraction between them.
    base_matrix = _clean_matrix(reps[base_rep])

    scores = {s: {e: np.full(n_samples, np.nan) for e in estimators}
              for s in spaces}
    failures = {s: {} for s in spaces}
    exclusion_note = {}
    gp_groups_dropped = {s: [] for s in spaces}
    timings = {}
    # Seconds per estimator, summed over the folds. One estimator can be most
    # of the run — this is the number that sizes the cluster request, and
    # without it the only thing to size from is a guess.
    est_seconds = {s: {e: 0.0 for e in estimators} for s in spaces}

    for space_name, (kind, blueprint, aug) in spaces.items():
        print(f"\n  === {space_name} ===")
        t_space = time.time()
        for fold_i, (train_idx, val_idx) in enumerate(folds):
            t_fold = time.time()
            if kind == 'hybrid':
                try:
                    X_tr, X_va = _fold_hybrid_pair(
                        train_idx, val_idx, reps, labels, search_groups,
                        blueprint, aug, augmentations)
                except Exception as exc:
                    failures[space_name][f'fold_{fold_i}'] = (
                        f'{type(exc).__name__}: {exc}')
                    print(f"    fold {fold_i}: build failed "
                          f"({type(exc).__name__}: {exc})")
                    continue
            else:
                X_tr, X_va = base_matrix[train_idx], base_matrix[val_idx]

            # Both arms, one rule: z-scored on this fold's training rows.
            scaler = StandardScaler().fit(X_tr)
            X_tr, X_va = scaler.transform(X_tr), scaler.transform(X_va)

            rows = np.concatenate([train_idx, val_idx])
            X_fold = np.vstack([X_tr, X_va])
            y_fold = labels[rows]
            excl_fold = exclusion[rows]
            is_val = np.zeros(len(rows), dtype=bool)
            is_val[len(train_idx):] = True

            fold_groups = None
            if search_groups is not None:
                fold_groups = np.asarray(search_groups)[rows]

            for est_name in estimators:
                X_e, y_e, rows_e, excl_e, val_e = (X_fold, y_fold, rows,
                                                   excl_fold, is_val)
                g_e = fold_groups
                if (est_name in GP_SUBSAMPLED_ESTIMATORS
                        and len(rows) > GP_MAX_ROWS):
                    # Seeded from the fold number alone, so every space caps
                    # the SAME molecules. With one generator advancing across
                    # spaces, the hybrid arm and the single-representation arm
                    # would keep different molecules, and the subtraction
                    # between them would be measuring that as well.
                    keep, dropped, n_repeat = _cap_whole_groups(
                        rows, excl_fold, GP_MAX_ROWS,
                        np.random.RandomState(42 + fold_i))
                    gp_groups_dropped[space_name].append(
                        {'fold': fold_i,
                         'repeat_groups_dropped': int(dropped),
                         'repeat_groups_in_this_fold': int(n_repeat),
                         'rows_kept': int(len(keep))})
                    X_e, y_e, rows_e = X_fold[keep], y_fold[keep], rows[keep]
                    excl_e, val_e = excl_fold[keep], is_val[keep]
                    g_e = None if fold_groups is None else fold_groups[keep]

                t_est = time.time()
                try:
                    est_kwargs = {}
                    params = inspect.signature(
                        type(get_noise_estimator(est_name)).__init__).parameters
                    if g_e is not None and 'groups' in params:
                        est_kwargs['groups'] = g_e
                    if 'exclude_groups' in params:
                        est_kwargs['exclude_groups'] = excl_e
                    if is_clf and 'is_clf' in params:
                        est_kwargs['is_clf'] = True
                    exclusion_note[est_name] = (
                        'its own repeat measurements are excluded from its '
                        'neighbours and its folds'
                        if 'exclude_groups' in est_kwargs
                        else ('protein-grouped folds' if 'groups' in est_kwargs
                              else 'nothing was excluded'))
                    est = get_noise_estimator(est_name, **est_kwargs)
                    s = np.asarray(est.estimate(X_e, y_e), dtype=float)
                    scores[space_name][est_name][rows_e[val_e]] = s[val_e]
                except Exception as exc:
                    failures[space_name].setdefault(est_name, []).append(
                        f'fold {fold_i}: {type(exc).__name__}: {exc}')
                est_seconds[space_name][est_name] += time.time() - t_est
            print(f"    fold {fold_i}: {len(train_idx)} rows chose the "
                  f"features, {len(val_idx)} rows scored "
                  f"[{time.time() - t_fold:.0f}s this fold]")
        timings[space_name] = round(time.time() - t_space, 1)

    results['seconds_per_space'] = timings
    results['seconds_per_estimator'] = {
        sp: {e: round(v, 1) for e, v in sorted(d.items(), key=lambda kv: -kv[1])}
        for sp, d in est_seconds.items()}
    for sp, d in est_seconds.items():
        worst = sorted(d.items(), key=lambda kv: -kv[1])[:3]
        if worst and worst[0][1] > 0:
            print(f"  {sp}: slowest estimators over all folds — " + ', '.join(
                f'{e} {v:.0f}s' for e, v in worst))
    results['replicate_exclusion'] = exclusion_note
    # Per fold, never summed: every fold is offered every group, so adding the
    # five drops would report more groups dropped than the dataset holds.
    results['gaussian_process_cap_per_fold'] = gp_groups_dropped
    results['gaussian_process_row_cap'] = GP_MAX_ROWS
    results['failures'] = convert_numpy(failures)
    results['how_a_molecule_was_scored'] = {
        'n_folds': len(folds),
        'folds': 'split so no protein group sits on both sides',
        'features': "each fold's features are chosen on the other folds' "
                    "rows, so no molecule helps choose the features its own "
                    "score is computed in. Nothing is stacked across folds: "
                    "a molecule is scored only in the space its own fold "
                    "built.",
        'rows_the_estimator_was_given': (
            "every row, mapped into that fold's space. The estimators score "
            "exactly the rows they are handed — none of them has a path to "
            "score a new molecule from a fit on others — so a fit on the "
            "training rows alone cannot be asked for without changing "
            "src/kirby/utils/noise_estimation.py. What the fold controls is "
            "the feature space and which rows keep a score."),
        'rows_that_kept_a_score': 'the held-out fifth only',
        'own_repeats_excluded': (
            "every estimator that accepts exclude_groups is given the repeat "
            "grouping, so a molecule is never scored from the labels its "
            "score is graded against"),
    }

    # ── how well each estimator agrees with each provider ──
    results['ground_truth'] = {}
    headline = {}
    for gt in ground_truths:
        per_space = {}
        for space_name in spaces:
            per_est = {}
            for est_name in estimators:
                a = agreement(gt, scores[space_name][est_name])
                a['n_molecules_scored'] = int(
                    np.isfinite(scores[space_name][est_name]).sum())
                per_est[est_name] = a
            per_space[space_name] = per_est
        results['ground_truth'][gt.provider] = {
            'statistic': gt.statistic,
            'units': gt.units,
            'n_groups': gt.n_groups,
            'n_rows_covered': gt.n_rows_covered,
            'counts': convert_numpy(gt.counts),
            'notes': gt.notes,
            'per_space': convert_numpy(per_space),
        }
        # The hybrid minus the single representation, per estimator. Both
        # numbers were already written before 2026-09-11 and never subtracted.
        deltas = {}
        for space_name in spaces:
            if space_name == base_rep:
                continue
            row = {}
            for est_name in estimators:
                hi = per_space[space_name][est_name].get('value')
                lo = per_space[base_rep][est_name].get('value')
                row[est_name] = (None if hi is None or lo is None
                                 else float(hi - lo))
            deltas[space_name] = row
        results['ground_truth'][gt.provider]['minus_comparator'] = {
            'comparator_space': base_rep,
            'per_space': convert_numpy(deltas),
            'what_this_is': (f'the {gt.statistic} in that space minus the '
                             f'{gt.statistic} in {base_rep}, per estimator; '
                             f'positive means the estimator agrees better '
                             f'where the features were chosen'),
        }
        if not headline:
            headline = {s: {e: per_space[s][e].get('value')
                            for e in estimators} for s in spaces}
            results['transfer_measured_against'] = gt.provider

    # ── the headline: does the injected-noise ordering survive ──
    results['transfer'] = convert_numpy(
        transfer_to_injected_noise(headline, output_dir, dataset_name))
    for label, entry in results['transfer'].items():
        if 'refused' in entry:
            print(f"\n  transfer vs {label}: not computed — {entry['refused']}")
            continue
        for space, row in entry.get('per_space', {}).items():
            if row.get('rank_correlation') is None:
                print(f"  transfer vs {label} in {space}: not computed — "
                      f"{row.get('refused')}")
            else:
                print(f"  transfer vs {label} in {space}: rank correlation "
                      f"{row['rank_correlation']:+.3f} over "
                      f"{row['n_estimators_in_common']} estimators")

    # ── the per-molecule numbers, so a re-analysis costs nothing ──
    # A full run is one hybrid build per fold per space at several minutes
    # each, so throwing the per-row arrays away means any second question
    # costs the whole run again. They go beside the JSON as an .npz: written
    # into the JSON they would be tens of megabytes of numbers no reader
    # opens by eye.
    primary = ground_truths[0]
    per_row_path = os.path.join(output_dir, 'n06_bindingdb_ground_truth',
                                f'{dataset_name}_per_row.npz')
    os.makedirs(os.path.dirname(per_row_path), exist_ok=True)
    payload = {
        'labels': labels,
        'replicate_group_id': primary.group_ids,
        'ground_truth_of_the_group': np.array(
            [primary.group_value.get(int(g), np.nan)
             for g in primary.group_ids], dtype=float),
        'exclusion_group': exclusion,
    }
    for space_name in spaces:
        for est_name in estimators:
            payload[f'score__{space_name}__{est_name}'] = \
                scores[space_name][est_name]
    np.savez_compressed(per_row_path, **payload)
    results['per_row_scores_file'] = per_row_path
    results['per_row_scores_keys'] = sorted(payload)
    print(f"\n  per-molecule scores written to {per_row_path}")

    # The per-GROUP arrays are small enough to live in the result file itself.
    gids, values = primary.group_order()
    results['per_group'] = {
        'provider': primary.provider,
        'units': primary.units,
        'group_id': [int(g) for g in gids],
        'ground_truth': [float(v) for v in values],
        'n_rows_in_group': [int((primary.group_ids == g).sum())
                            for g in gids],
        'what_one_row_is': 'one replicate group, not one molecule',
    }
    return results


if __name__ == '__main__':
    parser = create_experiment_parser('N06: ground-truth noise')
    parser.add_argument('--budget', type=int, default=None,
                        help='Feature budget (default: best across budgets)')
    parser.add_argument('--no-similarity-arm', action='store_true',
                        help='skip the hybrid whose features were chosen by '
                             'distance; it needs a similarity-objective '
                             'search, which is built if none is cached')
    args = parser.parse_args()
    config = get_config(args)

    dataset = getattr(args, 'dataset', 'bindingdb_kd')
    if dataset == 'logd':  # the parser default; n06 started life on DTA data
        dataset = 'bindingdb_kd'
    data = load_data(dataset, args.reps)
    results = run(data, config, args.output, dataset, budget=args.budget,
                  similarity_arm=not args.no_similarity_arm)
    save_experiment_results(results, args.output,
                            'n06_bindingdb_ground_truth', dataset)
