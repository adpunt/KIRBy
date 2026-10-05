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

THE SPACES, PER OUTER FOLD (2026-10-05, chat 5 of the alignment,
HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4). For each outer fold
(noise_helpers.noise_outer_folds; the count is config['n_splits']), searches
run on that fold's training rows only, on the dataset's decided grid with
pruning off, for the dataset's two decided models (DECISIONS.md Step 1). Per
model the fold gets 4 spaces: its 2 best single representations (base_1,
base_2) and its 2 best general hybrids with different selected features
(hybrid, hybrid__general2) -- 2 models x 4 = 8. Because n06's neighbour
estimators measure each molecule against its nearest neighbours in the space,
two neighbour spaces are scored as well: the model's top hybrid of the fold's
distance search (hybrid__distance1) and the top hybrid of the fold's
similarity search (hybrid_similarity; features chosen so that each molecule's
activity is predicted by its five nearest neighbours). Spaces are named
'<slot>@<model>'. No downstream model is fitted here, so every estimator runs
in every space, and each model's hybrids are subtracted from that model's own
single representations. A space with the same features as one already scored
on that fold is copied from it ('spaces_copied_from').

Scaling: base_1 and base_2 use kirby.scaling.FeatureScaler (0/1 features
unscaled). The hybrid and neighbour spaces come from n07._scale_pair, a plain
StandardScaler, so their 0/1 features are standardised until that shared code
changes.

NEIGHBOUR SETTINGS (2026-10-05, KNN_SEARCH_SPACE_RECORD). The four neighbour
estimators are scored at neighbour counts 5, 10 and 20 and at every distance
of the dataset's group (manhattan, cosine; plus minkowski p=0.5 on full_knn),
each as its own score '<estimator>__k<k>__<distance>'. No model is fitted, so
every setting is scored. The counting scheme is not swept (see
neighbour_estimator_settings).

Until 2026-10-05 the spaces came from the whole-dataset stored searches: rf
always, plus every one of five models, each with its top 2 general + top 1
distance hybrids per family (up to 40 spaces), so the held-out rows had helped
choose every recipe.

WHERE THE GROUND TRUTH COMES FROM. A ground-truth provider says which molecules
form a repeat group and how much the repeats disagree; `_ground_truth_providers`
holds the three that exist and the floor below which a provider refuses. A
dataset where no provider finds a repeat structure gets a recorded skip with
the count it found — Davis and KIBA both look like candidates until the
repeated pairs are counted, and both have none.

HOW A MOLECULE IS SCORED. The rows are split into group-aware folds. Each
fold's spaces are chosen and built from the OTHER folds' rows, so no molecule
helps choose the space its own noise score is computed in, and the estimator is
run inside that fold, in that fold's own space. Nothing is stacked across
folds: five folds' blocks glued into one matrix put column 7 of one row and
column 7 of another in different features. Each fold's scores are written as
the fold finishes (<dataset>.fold_ckpt.jsonl and one .npz per fold), so a
killed run redoes only the fold in flight.

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

# 2026-10-05 (chat 5 alignment, HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4): the
# whole-dataset readers (get_best_search_for_downstream, get_rule_hybrids,
# noise_base_reps_by_model, extract_top3_base_reps) and DOWNSTREAM_MODELS are
# no longer imported. Every space is chosen per outer fold, on that fold's
# training rows, for the dataset's two decided models
# (noise_helpers.fold_configurations / fold_model_inputs).
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories, REP_DTYPE,
)
from experiments.noise.noise_helpers import (
    noise_outer_folds, fold_spread, fold_configurations, fold_model_inputs,
    configuration_record, CONFIGURATION_SPACES, dataset_knn_group,
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

# 2026-10-05: the outer fold count is no longer a constant here. It comes from
# config['n_splits'] through noise_helpers.noise_outer_folds (handoff 2.4
# item 14), and any count works, 1 included.

# The spaces n06 scores per decided model per fold (2026-10-05, handoff 2.1
# R4/R5 and 2.3 item 1). The four configurations (2 base + 2 general hybrids)
# are noise_helpers.CONFIGURATION_SPACES. n06 fits no downstream model, but its
# neighbour estimators measure each molecule against its nearest neighbours in
# the space, so the two neighbour spaces are scored too: the top hybrid of the
# fold's distance search and the top hybrid of the fold's similarity search.
# They are not configurations and are listed apart in the result file.
# OPEN (chat 5): n06 is not on handoff 2.3 item 1's list of experiments that
# get neighbour spaces, and that item says every other experiment uses the 2
# general hybrids only. n06's neighbour estimators do read nearest neighbours,
# so both spaces are kept for now, scored by every estimator as before.
# Alternatives for Adelaide: score only the four neighbour estimators in these
# two spaces, or drop both and keep the 8 configurations only.
NEIGHBOUR_SPACES = ('hybrid__distance1', 'hybrid_similarity')


# OPEN (chat 5): the neighbour counting scheme (rank, inverse_square) is not
# swept; the estimators take no scheme (src/kirby/utils/noise_estimation.py).
def neighbour_estimator_settings(estimators, dataset_name, data):
    """Every score n06 computes per space: {score key: (estimator, kwargs)}.

    2026-10-05 (chat 5, KNN_SEARCH_SPACE_RECORD axes 2 and 4): the four
    neighbour estimators (n11's NEIGHBOR_ESTIMATORS) are scored at every
    neighbour count in similarity_space.SWEPT_KS (5, 10, 20) and every
    distance of the dataset's group (similarity_space.GROUP_METRICS: manhattan
    and cosine on fast_knn, plus minkowski p=0.5 on full_knn). No model is
    fitted, so every setting is scored; there is no screen. Their key is
    '<estimator>__k<k>__<distance>'. Every other estimator keeps its own name
    and its own defaults. Until 2026-10-05 the neighbour estimators ran only at
    their defaults: k=10 and the distance from KIRBY_DISTANCE_METRIC
    (manhattan when unset).

    OPEN (chat 5): the counting scheme (axis 5: rank, inverse_square) is not
    swept. These estimators average their neighbours equally or by one over
    distance inside src/kirby/utils/noise_estimation.py, and take no scheme.
    Sweeping it needs a parameter there (shared code).

    Returns (settings, group, ks, distances).
    """
    from experiments.noise import similarity_space as ss
    from kirby.utils.noise_estimation import get_noise_estimator as _get
    group = dataset_knn_group(data, dataset_name)
    ks = tuple(int(k) for k in ss.SWEPT_KS)
    distances = tuple(ss.GROUP_METRICS[group])
    settings = {}
    for est in estimators:
        params = inspect.signature(type(_get(est)).__init__).parameters
        if (est in NEIGHBOR_ESTIMATORS and 'k' in params
                and 'metric' in params):
            for k in ks:
                for d in distances:
                    settings[f'{est}__k{k}__{d}'] = (est, {'k': k, 'metric': d})
        else:
            settings[est] = (est, {})
    return settings, group, ks, distances


def estimator_of_key(key):
    """'neighbor_residual__k10__cosine' -> 'neighbor_residual'."""
    return key.split('__k', 1)[0] if '__k' in key else key

# Identity of the per-fold checkpoint lines and score files. Bump when what a
# fold computes changes, so an older partial run is never read back.
CHECKPOINT_IDENTITY = 'n06_per_fold_v2_neighbour_settings_2026-10-05'

# 2026-10-05 (chat 5): the whole-dataset helpers that stood here and further
# down (POOL_SUBSET_*, SPACE_FOR_FAMILY, FEATURES_CHOSEN_BY, _space_name,
# _blueprint_for, rule_entry_summary, model_space_name, recipe_signature,
# rule_sets_by_model, rule_base_reps, rule_hybrid_spaces, _fold_hybrid_pair)
# are deleted. They served the read of the whole-dataset stored search, and
# _fold_hybrid_pair rebuilt hybrids with the whole-dataset representation
# scores (dirty-run bug 1). No experiment imports them any more: n22 imports
# resolve_estimators, _canonical, _clean_matrix, GP_SUBSAMPLED_ESTIMATORS and
# GP_MAX_ROWS; n14 imports build_replicate_groups and compute_replicate_stds.


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
    # float32 (handoff 2.3 item 18, 2026-10-05): every representation is held
    # in one precision. This cast to float64 is gone; 1e10 fits in float32.
    return np.nan_to_num(np.clip(np.asarray(X, dtype=REP_DTYPE), -1e10, 1e10),
                         nan=0.0, posinf=1e10, neginf=-1e10)


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
    # How each level's "did the noise hurt" was read; see below.
    gap_read_from = {'clean_and_noisy_scores': 0,
                     'stored_gap_as_signed': 0,
                     'stored_gap_disagreed_with_scores': 0}
    for sigma, sigma_data in (doc.get('per_sigma') or {}).items():
        if str(sigma) == '0.0':
            continue  # no noise was injected, so there is nothing to recover
        # The recovery is a share of the gap between the clean-label and the
        # noisy-label model score. When the injected noise did not make the
        # model worse, that gap runs the wrong way and the share reads
        # backwards — a cleanup that did nothing scores well. Whole sigma
        # dropped, and counted.
        #
        # ⛔ FIXED 2026-09-25 (METRIC_DIRECTION_AUDIT.md, n06:388-389). This
        # check was written on 2026-09-11 (22067eb) for n11's old unsigned
        # `gap = clean - noisy`, and dropped a level when gap > 0 on an rmse
        # dataset. n11 has written `gap` SIGNED since df7c7fe/1d9487b
        # (2026-09-12/13, n11:1277): `(noisy - clean) if lower_is_better else
        # (clean - noisy)`, positive = the noise hurt, on every metric. Read
        # under the old rule, every rmse level where the noise raised rmse was
        # dropped and only the levels where it lowered rmse were kept.
        #
        # The n11 file does not record which convention wrote its `gap`. It
        # does store the reference model's `clean_score` and `noisy_score`
        # beside it, in files from both before and after the change (22067eb
        # n11:657-659; n11:1299-1301 today). So the signed gap is recomputed
        # from those two and the dataset's own lower_is_better, which reads a
        # file of either age the same way. Only a file missing either score
        # falls back to the stored gap, read as signed (positive = hurt).
        _c, _nz = sigma_data.get('clean_score'), sigma_data.get('noisy_score')
        _stored = sigma_data.get('gap')
        if (_c is not None and _nz is not None
                and np.isfinite(_c) and np.isfinite(_nz)):
            gap = (float(_nz) - float(_c)) if lower_is_better \
                else (float(_c) - float(_nz))
            gap_read_from['clean_and_noisy_scores'] += 1
            if (_stored is not None and np.isfinite(_stored)
                    and np.sign(_stored) != np.sign(gap)):
                # An unsigned file on an rmse dataset, most likely.
                gap_read_from['stored_gap_disagreed_with_scores'] += 1
        else:
            gap = _stored
            gap_read_from['stored_gap_as_signed'] += 1
        if gap is None or not np.isfinite(gap) or gap <= 0:
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
             'noise_gap_read_from': gap_read_from,
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
                # No p-value (2026-10-05): no significance tests anywhere.
                rho, _p = spearmanr(a, b)
                row['rank_correlation'] = float(rho)
            entry['per_space'][space] = row
        out[label] = entry
    return out


# ── one outer fold (2026-10-05, chat 5) ───────────────────────────────────

def _recipe_summary(bp, aug_cond):
    """The recipe fields a reader needs, from a Blueprint."""
    if bp is None:
        return None
    return {'budget_features': int(getattr(bp, 'budget', 0) or 0),
            'allocation_method': getattr(bp, 'allocation_method', None),
            'importance_method': getattr(bp, 'importance_method', None),
            'internal_evaluator': getattr(bp, 'internal_evaluator', None),
            'n_reps': len(getattr(bp, 'reps', []) or []),
            'reps': sorted(getattr(bp, 'reps', []) or []),
            'aug_condition': aug_cond}


def _matrix_signature(X_tr, X_va):
    """Same bytes in both blocks = the same space; scored once, copied."""
    from experiments.noise.noise_cache import array_hash
    return array_hash(np.ascontiguousarray(X_tr), np.ascontiguousarray(X_va))


def _score_one_fold(data, dataset_name, labels, exclusion, settings, is_clf,
                    search_groups, metric, *, fold_i, train_idx, val_idx,
                    n_folds, neighbour_spaces):
    """Every estimator's score for this fold's held-out rows, in every space
    this fold's searches chose. `settings` is neighbour_estimator_settings'
    {score key: (estimator, kwargs)}: one key per neighbour estimator per
    neighbour count per distance, one per other estimator.

    The spaces: per decided model, the 2 best single representations and 2
    best general hybrids (noise_helpers.fold_configurations, searches on this
    fold's training rows; pruning off; decided grid), and the neighbour
    spaces. Each is built and scaled by noise_helpers.fold_model_inputs
    (hybrids rebuilt on the training rows with the fold search's own
    representation scores, dirty-run bug 1). base_1 and base_2 are scaled
    with kirby.scaling.FeatureScaler (0/1 features unscaled); the hybrid and
    neighbour spaces with n07._scale_pair, a plain StandardScaler, so their
    0/1 features are standardised (shared code, reported 2026-10-05).

    Returns (meta, {space: {score key: scores of the held-out rows}}).
    """
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    task_type = data.get('task_type') or 'regression'
    t_fold = time.time()
    print(f"\n  === fold {fold_i}: {len(train_idx)} training rows, "
          f"{len(val_idx)} held-out rows ===")

    entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                  train_idx=train_idx, n_folds=n_folds)
    wanted = tuple(CONFIGURATION_SPACES) + tuple(neighbour_spaces)
    spaces_X, space_info, unavailable, configurations = {}, {}, {}, {}
    for m, entry in entries.items():
        factory = get_model_factories([m], task_type=task_type,
                                      n_samples=len(train_idx)).get(m)
        X, unav, names = fold_model_inputs(
            data, dataset_name, m, entry, train_idx=train_idx,
            val_idx=val_idx, y_train=labels[train_idx], factory=factory,
            metric=metric, spaces=wanted)
        configurations[m] = configuration_record(entry, names)
        for sp, why in (unav or {}).items():
            if sp in wanted:
                unavailable[f'{sp}@{m}'] = why
        recipes = (entry or {}).get('recipes') or {}
        for sp in wanted:
            if sp not in X:
                continue
            name = f'{sp}@{m}'
            X_tr, X_va = X[sp]
            spaces_X[name] = (_clean_matrix(X_tr), _clean_matrix(X_va))
            info = {'model': m, 'slot': sp,
                    'kind': ('configuration' if sp in CONFIGURATION_SPACES
                             else 'neighbour_space')}
            if sp.startswith('base_'):
                info['representation'] = names.get(sp)
            else:
                rec = recipes.get(sp)
                info['recipe'] = names.get(sp)
                info.update((_recipe_summary(*rec) if rec else None) or {})
            space_info[name] = info
        print(f"  {m}: {sorted(sp for sp in X if sp in wanted)}"
              + (f"; unavailable: {sorted(unav)}" if unav else ''))

    fold_scores = {}
    failures, copied_from, gp_cap = {}, {}, {}
    seconds_per_space, est_seconds, exclusion_note = {}, {}, {}
    first_with_signature = {}

    rows = np.concatenate([train_idx, val_idx])
    y_fold = labels[rows]
    excl_fold = exclusion[rows]
    is_val = np.zeros(len(rows), dtype=bool)
    is_val[len(train_idx):] = True
    fold_groups = (None if search_groups is None
                   else np.asarray(search_groups)[rows])

    for space_name, (X_tr, X_va) in spaces_X.items():
        # A space with the same features as one already scored on this fold
        # (both models ranking the same representation, the similarity hybrid
        # both models share) is copied from it, not scored again.
        sig = _matrix_signature(X_tr, X_va)
        src = first_with_signature.get(sig)
        if src is not None:
            fold_scores[space_name] = {e: v.copy()
                                       for e, v in fold_scores[src].items()}
            failures[space_name] = dict(failures.get(src) or {})
            gp_cap[space_name] = gp_cap.get(src)
            copied_from[space_name] = src
            seconds_per_space[space_name] = 0.0
            print(f"    {space_name}: same features as {src}, copied")
            continue
        first_with_signature[sig] = space_name
        t_space = time.time()
        X_fold = np.vstack([X_tr, X_va])
        fold_scores[space_name] = {}
        est_seconds[space_name] = {}
        for key, (est_name, setting_kwargs) in settings.items():
            X_e, y_e, excl_e, val_e = X_fold, y_fold, excl_fold, is_val
            g_e = fold_groups
            if (est_name in GP_SUBSAMPLED_ESTIMATORS
                    and len(rows) > GP_MAX_ROWS):
                # Seeded from the fold number alone, so every space caps the
                # SAME molecules. With one generator advancing across spaces,
                # the hybrid spaces and the single-representation spaces would
                # keep different molecules, and the subtraction between them
                # would be measuring that as well.
                keep, dropped, n_repeat = _cap_whole_groups(
                    rows, excl_fold, GP_MAX_ROWS,
                    np.random.RandomState(42 + fold_i))
                gp_cap[space_name] = {
                    'repeat_groups_dropped': int(dropped),
                    'repeat_groups_in_this_fold': int(n_repeat),
                    'rows_kept': int(len(keep))}
                X_e, y_e = X_fold[keep], y_fold[keep]
                excl_e, val_e = excl_fold[keep], is_val[keep]
                g_e = None if fold_groups is None else fold_groups[keep]
            else:
                keep = None
            out = np.full(len(val_idx), np.nan)
            t_est = time.time()
            try:
                est_kwargs = dict(setting_kwargs)
                params = inspect.signature(
                    type(get_noise_estimator(est_name)).__init__).parameters
                if g_e is not None and 'groups' in params:
                    est_kwargs['groups'] = g_e
                if 'exclude_groups' in params:
                    # The row's own repeat measurements, never its scaffold
                    # group (kNN record axis 6 keeps scaffold-mates).
                    est_kwargs['exclude_groups'] = excl_e
                if is_clf and 'is_clf' in params:
                    est_kwargs['is_clf'] = True
                exclusion_note[est_name] = (
                    'its own repeat measurements are excluded from its '
                    'neighbours and its folds'
                    if 'exclude_groups' in est_kwargs
                    else ('group-aware folds' if 'groups' in est_kwargs
                          else 'nothing was excluded'))
                est = get_noise_estimator(est_name, **est_kwargs)
                s = np.asarray(est.estimate(X_e, y_e), dtype=float)
                if keep is None:
                    out = s[val_e]
                else:
                    # positions of the kept held-out rows among val_idx
                    kept_val = keep[val_e] - len(train_idx)
                    out[kept_val] = s[val_e]
            except Exception as exc:
                failures.setdefault(space_name, {}).setdefault(
                    key, f'{type(exc).__name__}: {exc}')
            fold_scores[space_name][key] = out
            est_seconds[space_name][key] = round(time.time() - t_est, 1)
        seconds_per_space[space_name] = round(time.time() - t_space, 1)
        print(f"    {space_name}: scored [{seconds_per_space[space_name]:.0f}s]")

    meta = {
        'fold': int(fold_i),
        'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)),
        'models': sorted(entries),
        'configurations': configurations,
        'spaces': space_info,
        'unavailable': unavailable,
        'spaces_copied_from': copied_from,
        'failures': failures,
        'gaussian_process_cap': {k: v for k, v in gp_cap.items() if v},
        'seconds_per_space': seconds_per_space,
        'seconds_per_estimator': est_seconds,
        'seconds_this_fold': round(time.time() - t_fold, 1),
        'exclusion_note': exclusion_note,
        'selection_scope': 'fold_train_rows',
    }
    return meta, fold_scores


# ── the run ────────────────────────────────────────────────────────────────

def run(data, config, output_dir, dataset_name, force_rerun=False, **kwargs):
    """N06: estimator scores against a disagreement nobody injected."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N06: GROUND-TRUTH NOISE")
    print("=" * 70)

    labels = np.asarray(data['labels'], dtype=np.float64)
    n_samples = len(labels)
    task_type = data['task_type']
    is_clf = (task_type == 'classification')
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

    # Neighbour counts and distances (kNN record, 2026-10-05). score_keys is
    # what each space is scored under; see neighbour_estimator_settings.
    settings, knn_group, sw_ks, sw_dist = neighbour_estimator_settings(
        estimators, dataset_name, data)
    score_keys = list(settings)
    results['neighbour_estimator_settings'] = {
        'knn_group': knn_group,
        'neighbour_counts': list(sw_ks),
        'distances': list(sw_dist),
        'counting_scheme': ('not swept: each estimator averages its '
                            'neighbours its own way (equal weights, or one '
                            'over distance for distance_weighted)'),
        'score_keys': {k: {'estimator': e, **kw}
                       for k, (e, kw) in settings.items()},
    }
    print(f"  {len(score_keys)} scores per space ({knn_group}: neighbour "
          f"counts {list(sw_ks)}, distances {list(sw_dist)})")

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

    # ── feature spaces, chosen per outer fold (2026-10-05, chat 5) ──
    # Each outer fold gets its own spaces, chosen by searches run on that
    # fold's training rows only (handoff 2.1 R2), for the dataset's two decided
    # models (R1, DECISIONS.md Step 1). Per model: its 2 best single
    # representations and its 2 best general hybrids (R4, R5; 2 models x 4 = 8
    # spaces), plus the two neighbour spaces (NEIGHBOUR_SPACES). A space is
    # named '<slot>@<model>', e.g. 'base_1@svm' or 'hybrid__general2@dnn'. The
    # slot is a rank, not a fixed representation: base_1@svm can be a
    # different representation on fold 0 and fold 1, and each fold's record
    # says which (per_fold[<i>]['spaces']).
    #
    # Until 2026-10-05 the spaces came from the whole-dataset stored searches
    # (rf's pool winner and every one of five models' top 2 general + top 1
    # distance hybrids per family, up to 40 spaces), always with rf added.
    #
    # The searches run on the rows n06 scores: the dataset as loaded. So no
    # condition_suffix is needed. On herg_ic50 those are the measurement rows,
    # and a molecule's repeat measurements share its scaffold group, so the
    # group-aware folds keep them on one side.
    if kwargs.get('budget') is not None:
        print(f"  --budget {kwargs['budget']} is ignored: each fold's hybrids "
              f"are ranked over the dataset's decided budgets (2026-10-05)")
        results['budget_argument_ignored'] = int(kwargs['budget'])
    similarity_arm = bool(kwargs.get('similarity_arm', True))
    neighbour_spaces = tuple(sp for sp in NEIGHBOUR_SPACES
                             if similarity_arm or sp != 'hybrid_similarity')
    results['similarity_arm'] = similarity_arm
    results['spaces_per_model'] = {
        'configurations': list(CONFIGURATION_SPACES),
        'neighbour_spaces': list(neighbour_spaces),
        'what_they_are': (
            "base_1, base_2: the model's 2 best single representations, "
            "hybrid, hybrid__general2: its 2 best general hybrids with "
            "different selected features, both ranked by a search on the "
            "fold's training rows. hybrid__distance1: the model's top hybrid "
            "of the fold's distance search (knn is one of its models, budgets "
            "up to 500 features). hybrid_similarity: the top hybrid of the "
            "fold's similarity search, whose features were chosen so that "
            "each molecule's activity is predicted by its five nearest "
            "neighbours; it is the same recipe for both models. No downstream "
            "model is fitted in n06; every estimator runs in every space."),
    }
    results['selection_scope'] = 'fold_train_rows'

    folds, n_folds = noise_outer_folds(data, config)
    results['n_folds'] = int(n_folds)
    results['folds_run'] = [int(f) for f, _, _ in folds]
    print(f"  {len(folds)} of {n_folds} outer folds "
          f"(kirby.find_best_hybrid.nested_cv_outer_folds, group-aware)")

    # A group whose rows land in two different folds has its rows scored in two
    # different feature spaces, and the group's mean score then mixes them.
    # On BindingDB it cannot happen — a group is one drug against one protein,
    # and the folds are split by group — but nothing enforces that, so it is
    # counted rather than assumed.
    fold_of_row = np.full(n_samples, -1, dtype=np.int64)
    for fold_i, _tr, va in folds:
        fold_of_row[va] = fold_i
    straddling = {}
    for gt in ground_truths:
        n_straddle = sum(
            1 for gid in gt.group_value
            if len(np.unique(fold_of_row[gt.rows[int(gid)]])) > 1)
        straddling[gt.provider] = n_straddle
        if n_straddle:
            print(f"  {n_straddle} of {gt.n_groups} {gt.provider} groups have "
                  f"rows in more than one fold, so their rows were scored in "
                  f"different feature spaces")
    results['groups_whose_rows_span_more_than_one_fold'] = straddling

    # Written per fold as it finishes (dirty-run bug 12): one checkpoint line
    # and one score file per fold. A killed run redoes only the fold in flight.
    from experiments.noise.n07_remediation import FoldCheckpoint
    from experiments.noise.noise_cache import array_hash
    exp_dir = 'n06_bindingdb_ground_truth'
    fingerprint = {
        'identity': CHECKPOINT_IDENTITY, 'dataset': dataset_name,
        'n_samples': int(n_samples), 'n_folds': int(n_folds),
        'labels': array_hash(labels), 'exclusion': array_hash(exclusion),
        'score_keys': list(score_keys),
        'neighbour_spaces': list(neighbour_spaces),
    }
    fp_tag = array_hash(np.frombuffer(
        repr(sorted(fingerprint.items())).encode(), dtype=np.uint8))[:12]
    ckpt = FoldCheckpoint(output_dir, exp_dir, dataset_name, tag='fold_ckpt',
                          fingerprint=fingerprint)
    force_rerun = bool(force_rerun)

    scores = {}           # space -> estimator -> score per row (NaN = unscored)
    per_fold = {}
    for fold_i, train_idx, val_idx in folds:
        done = None if force_rerun else ckpt.get(['fold', int(fold_i)])
        if done is not None and os.path.exists(str(done.get('npz') or '')):
            meta = done['meta']
            z = np.load(done['npz'])
            vi = z['val_idx']
            for key, (sp, est) in (meta.get('npz_keys') or {}).items():
                scores.setdefault(sp, {e: np.full(n_samples, np.nan)
                                       for e in score_keys})
                scores[sp][est][vi] = z[key]
            per_fold[int(fold_i)] = meta
            print(f"\n  fold {fold_i}: read back from {done['npz']}")
            continue

        meta, fold_scores = _score_one_fold(
            data, dataset_name, labels, exclusion, settings, is_clf,
            search_groups, (metric_name, _metric_fn, lower_is_better),
            fold_i=int(fold_i), train_idx=train_idx, val_idx=val_idx,
            n_folds=n_folds, neighbour_spaces=neighbour_spaces)

        npz_path = os.path.join(output_dir, exp_dir,
                                f'{dataset_name}.fold{fold_i}.{fp_tag}.npz')
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        payload, npz_keys = {'val_idx': np.asarray(val_idx)}, {}
        for j, (sp, est) in enumerate((sp, est) for sp in fold_scores
                                      for est in fold_scores[sp]):
            payload[f's{j}'] = fold_scores[sp][est]
            npz_keys[f's{j}'] = [sp, est]
        np.savez_compressed(npz_path, **payload)
        meta['npz_keys'] = npz_keys
        meta = convert_numpy(meta)
        ckpt.put(['fold', int(fold_i)], {'meta': meta, 'npz': npz_path})
        for sp, d in fold_scores.items():
            scores.setdefault(sp, {e: np.full(n_samples, np.nan)
                                   for e in score_keys})
            for est, v in d.items():
                scores[sp][est][np.asarray(val_idx)] = v
        per_fold[int(fold_i)] = meta

    # One entry per fold: which representation or recipe filled each space,
    # each model's configurations with the fold search's own training-row
    # scores, what was unavailable and why, copies, failures, timings.
    # n06 fits no downstream model, so there is no held-out model score.
    results['per_fold'] = {str(k): {kk: vv for kk, vv in v.items()
                                    if kk != 'npz_keys'}
                           for k, v in per_fold.items()}
    results['held_out_model_score'] = (
        'none: n06 fits no downstream model. The per-fold searches\' '
        'training-row scores are under per_fold[<i>][\'configurations\'].')

    spaces = sorted(scores)
    space_model = {sp: sp.rsplit('@', 1)[1] for sp in spaces if '@' in sp}
    base_spaces = [sp for sp in spaces if sp.split('@', 1)[0].startswith('base_')]
    results['space_model'] = space_model
    results['comparator_spaces'] = base_spaces
    if not spaces:
        reason = ('no feature space to score in on any fold: ' + ' | '.join(
            f"fold {k}: {v.get('unavailable')}" for k, v in per_fold.items()))
        print(f"  SKIP: {reason}")
        results.update({'skipped': True, 'reason': reason,
                        'policy': 'no_feature_space'})
        return results

    # Per fold, never summed: every fold is offered every group, so adding the
    # drops would report more groups dropped than the dataset holds.
    results['gaussian_process_row_cap'] = GP_MAX_ROWS
    exclusion_note = {}
    for v in per_fold.values():
        exclusion_note.update(v.get('exclusion_note') or {})
    results['replicate_exclusion'] = exclusion_note
    results['how_a_molecule_was_scored'] = {
        'n_folds': int(n_folds),
        'folds': ('kirby.find_best_hybrid.nested_cv_outer_folds on the '
                  "dataset's groups, the folds the per-fold searches are "
                  'filed under'),
        'features': ("each fold's representations and hybrid recipes are "
                     "chosen by searches on that fold's training rows, and "
                     "each hybrid's features are selected on those rows, so no "
                     "held-out molecule helps choose the space its own score "
                     "is computed in. Nothing is stacked across folds: a "
                     "molecule is scored only in the space its own fold "
                     "built."),
        'scaling': {
            'base_1, base_2': ("kirby.scaling.FeatureScaler fitted on the "
                               "fold's training rows: 0/1 features unscaled, "
                               "the rest standardised"),
            'hybrid, hybrid__general2, hybrid__distance1, hybrid_similarity': (
                "sklearn StandardScaler fitted on the fold's training rows "
                "(n07_remediation._scale_pair): every feature standardised, "
                "0/1 features included"),
        },
        'rows_the_estimator_was_given': (
            "every row, mapped into that fold's space. The estimators score "
            "exactly the rows they are handed — none of them has a path to "
            "score a new molecule from a fit on others — so a fit on the "
            "training rows alone cannot be asked for without changing "
            "src/kirby/utils/noise_estimation.py. What the fold controls is "
            "the feature space and which rows keep a score."),
        'rows_that_kept_a_score': "the fold's held-out rows only",
        'own_repeats_excluded': (
            "every estimator that accepts exclude_groups is given the repeat "
            "grouping, so a molecule is never scored from the labels its "
            "score is graded against"),
        'neighbour_estimator_settings': (
            "see results['neighbour_estimator_settings']: each neighbour "
            "estimator is scored at every neighbour count and distance"),
    }

    # ── how well each estimator agrees with each provider ──
    # Pooled over every fold's held-out rows (each row scored in its own
    # fold's space), and per fold with the spread over folds (fold_spread; no
    # significance test, so agreement's p-value is dropped, 2026-10-05).
    # One entry per score key: a neighbour estimator has one per neighbour
    # count and distance.
    # OPEN (chat 5): the pooled number mixes representations. A slot such as
    # base_1@svm is a rank, and each fold's search can fill it with a
    # different representation (per_fold[<i>]['spaces'] names them). Pooling
    # by rank is kept; the alternative is to report agreement per fold only
    # (per_space_by_fold already holds it).
    fold_rows = {int(f): np.asarray(va) for f, _tr, va in folds}
    results['ground_truth'] = {}
    headline = {}
    for gt in ground_truths:
        per_space, per_fold_agree = {}, {}
        for space_name in spaces:
            per_est, per_est_fold = {}, {}
            for est_name in score_keys:
                sc = scores[space_name][est_name]
                a = agreement(gt, sc)
                a.pop('p', None)
                a['n_molecules_scored'] = int(np.isfinite(sc).sum())
                per_est[est_name] = a
                vals, refused = [], {}
                for f, va in fold_rows.items():
                    only = np.full(n_samples, np.nan)
                    only[va] = sc[va]
                    a_f = agreement(gt, only)
                    vals.append(a_f.get('value'))
                    if a_f.get('refused'):
                        refused[str(f)] = a_f['refused']
                per_est_fold[est_name] = dict(fold_spread(vals),
                                              refused_by_fold=refused)
            per_space[space_name] = per_est
            per_fold_agree[space_name] = per_est_fold
        results['ground_truth'][gt.provider] = {
            'statistic': gt.statistic,
            'units': gt.units,
            'n_groups': gt.n_groups,
            'n_rows_covered': gt.n_rows_covered,
            'counts': convert_numpy(gt.counts),
            'notes': gt.notes,
            'per_space': convert_numpy(per_space),
            'per_space_by_fold': convert_numpy(per_fold_agree),
        }
        # Each hybrid space minus each single-representation space of the SAME
        # model, per estimator. Which representation filled base_k@<model>
        # can differ by fold; per_fold[<i>]['spaces'] says which.
        minus = {}
        for base_sp in base_spaces:
            deltas = {}
            for space_name in spaces:
                if space_name in base_spaces:
                    continue
                if space_model.get(space_name) != space_model.get(base_sp):
                    continue
                row = {}
                for est_name in score_keys:
                    hi = per_space[space_name][est_name].get('value')
                    lo = per_space[base_sp][est_name].get('value')
                    row[est_name] = (None if hi is None or lo is None
                                     else float(hi - lo))
                deltas[space_name] = row
            minus[base_sp] = {
                'comparator_space': base_sp,
                'comparator_representation_per_fold': {
                    str(k): ((v.get('spaces') or {}).get(base_sp) or {}).get(
                        'representation') for k, v in per_fold.items()},
                'model': space_model.get(base_sp),
                'per_space': convert_numpy(deltas),
                'what_this_is': (f'the {gt.statistic} in that space minus the '
                                 f'{gt.statistic} in {base_sp}, per '
                                 f'estimator; positive means the estimator '
                                 f'agrees better where the features were '
                                 f'chosen'),
            }
        results['ground_truth'][gt.provider]['minus_comparator'] = minus
        if not headline:
            # One number per estimator for the comparison with n11's and
            # n13's orderings, which name an estimator without a setting.
            # OPEN (chat 5): a neighbour estimator's number is the median of
            # its settings' agreement (3 neighbour counts x 2 or 3 distances),
            # as n11 takes the median over its cells. The alternatives are to
            # compare each setting separately, or to use one fixed setting.
            for s_ in spaces:
                row_ = {}
                for e in estimators:
                    vals_ = [per_space[s_][k].get('value') for k in score_keys
                             if estimator_of_key(k) == e]
                    vals_ = [float(v) for v in vals_
                             if v is not None and np.isfinite(v)]
                    row_[e] = float(np.median(vals_)) if vals_ else None
                headline[s_] = row_
            results['transfer_measured_against'] = gt.provider
            results['transfer_estimator_number'] = (
                'for a neighbour estimator, the median of its agreement over '
                'its neighbour counts and distances; for every other '
                'estimator, its agreement')

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
        'outer_fold_of_row': fold_of_row,
    }
    for space_name in spaces:
        for est_name in score_keys:
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
                        help='ignored since 2026-10-05: each fold ranks its '
                             'hybrids over the dataset\'s decided budgets')
    parser.add_argument('--no-similarity-arm', action='store_true',
                        help='do not score the hybrid_similarity space (the '
                             'top hybrid of each fold\'s similarity search). '
                             'The search itself still runs inside '
                             'fold_configurations. Written into the result '
                             'file as similarity_arm.')
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
