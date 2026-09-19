#!/usr/bin/env python3
"""N23: can a model's own per-molecule uncertainty clean labels?

Paper: Noise.

THE QUESTION. Every detector in this suite scores a molecule by comparing its
label to the labels of its nearest neighbours. A model that predicts a SPREAD
per molecule is a different quantity: it is what the model thinks that
molecule's label noise is, learned from the features rather than from the
neighbours. Nothing in the suite has used one to clean anything.

TWO SHAPES, and the point is that both are tried.

  drop   remove the training molecules whose predicted spread is largest. The
         share removed is swept over DROP_SHARES.
  move   move each training label toward the model's own out-of-fold prediction,
         further the wider that molecule's predicted spread. The strength is
         swept over MOVE_STRENGTHS and the rule over MOVE_METHODS.

  none   the score to beat: train on the labels as they are.

THE TWO MOVE RULES ARE NOT THE SAME KNOB, and both run because the difference
matters here. 'proportional' rescales the spreads to run from zero to one and
moves each label by `strength` times its rescaled value -- so it keeps the ORDER
of the molecules and throws away the SCALE. 'confidence' uses the spread as it
comes, in the label's own units, and moves each label by
`strength * spread / (1 + strength * spread)`. A predicted spread is in label
units, so 'confidence' is the rule that uses what makes it different from a
neighbour score; 'proportional' is the rule `n07.label_smooth` already uses, so
it is what makes the two comparable. Both are recorded on every row.

WHICH UNCERTAINTY NUMBERS, AND WHAT IS KNOWN ABOUT THEM.

  hetero_gp           a Gaussian process for the mean plus a network
                      predicting each molecule's noise, trained together. Ported
                      into KIRBy on 2026-09-18 from qsar_qm_models, which is the
                      only model in either roster whose two uncertainty terms
                      come apart. Out of fold.
  ngboost             the spread NGBoost predicts per molecule, out of fold.
  heteroscedastic_nn  a network with a second output predicting each molecule's
                      variance, out of fold.
  gaussian_process    KIRBy's, and it is NOT the same quantity -- see below.
  activity_cliff_aware  the neighbour-based score n07, n18 and n20 already use.
                      It reads the molecule's own LABEL against its neighbours'
                      rather than predicting anything from the features, which
                      is why it is the only one of the five that can answer
                      under a condition that picks its corrupted molecules at
                      random. The thing to beat.

⛔ FOUR OF THE FIVE CAN ONLY ANSWER WHEN THE NOISE IS A PROPERTY OF THE
STRUCTURE. They predict a molecule's noise from its features and they are scored
out of fold, so a held-out molecule can only be ranked when its own features say
how noisy it is. `outlier_p01`, `outlier_p05` and `outlier_p10` choose the
molecules that get the wider error AT RANDOM, with no reference to the features
(NOISE_AUDIT_2026_09_10.md 6.7). `grouped_wider` gives the wider error to whole
scaffold families, which a representation encodes, and it is the one
dose-matched condition where the question has an answer.

MEASURED 2026-09-18 on a 300-molecule fixture, out of fold over three folds, the
rank correlation between each source's number and the true noise width:

    which molecules are loud   hetero_gp   ngboost   activity_cliff_aware
    drawn at random               +0.149    -0.027                 +0.057
    decided by the features       +0.383    +0.511                 +0.078

So a chance-level reading under an outlier condition is the condition, not the
method. `sources_defined_under_this_condition` is written onto every result file
so that cannot be misread.

⛔ A MODEL'S UNCERTAINTY IS TWO NUMBERS ADDED TOGETHER, AND ONLY ONE OF THEM IS
A CLEANING SIGNAL. One is how noisy that molecule's label looks. The other is
how unsure the model is about that region of chemistry. Dropping molecules by
the second is dropping the molecules the model finds hard, which is n18's own
caveat at :23-26.

Which models tell the two apart was measured in the sister study, not here:
`~/repos/qsar_qm_models/RERUN_PLAN.md` sections 2.30 and 2.32, three seeds,
3,000 real QM9 molecules, three folds out of fold on Murcko scaffold groups,
plain Gaussian noise at level 0 and level 1. Read those numbers as that study's,
dated 2026-08-30 and 2026-08-31, and re-check before repeating them:

  Gaussian process       label-noise term rises x17.2, model-unsure term x1.5.
                         Separation 11.2 to 18.7 over three seeds. Its two terms
                         are the same SIZE as each other, which is why the
                         separation means something. The only PASS.
  NGBoost                label-noise term rises x9.1, the second best measured,
                         and it stays on that study's uncertainty list. It has
                         NO model-unsure term: that term came out 3,581 times
                         smaller than the other, and it RISES with the noise,
                         which disqualifies it. So nothing there can check that
                         NGBoost's number is not tracking model ignorance.
  variance-head network  separation 1.5 to 2.3. Real but small.
  random forest          separation 1.1, written down as FAIL: "one bootstrap
                         causes both".

⛔ KIRBY'S `gaussian_process` ESTIMATOR DOES NOT RETURN THE NUMBER THAT PASSED.
Read out of `src/kirby/utils/noise_estimation.py:726-800` this session. Its
per-molecule score is the posterior standard deviation PLUS the out-of-fold
residual, which is the model-unsure term. Its label-noise term is
`k2__noise_level`, one number for the whole fold, identical for every molecule,
so it cannot rank molecules at all. It is swept here anyway, because the
difference between it and the two that do predict a spread per molecule is then
measured on this data rather than carried over from another repository's table.

THE CHECK THAT DOES NOT NEED A MODEL FIT AT ALL. Under injection there is a list
of the molecules whose label actually moved. `noise_helpers.detection_scores`
grades each uncertainty number against that list before any cleaning runs. That
is the direct answer to "is this number a noise signal", and it costs nothing:
the spreads and the list are both already in memory. It is defined only under
the conditions where a corrupted SUBSET exists -- `outlier_p10` is one.

THE ZERO-NOISE CONTROL. Level 0.0 in the injected regime corrupts nothing, so
there is nothing to clean. A source whose cleaning still wins folds there is
removing or moving the molecules the model finds hard. That is n18's QM9
argument put on a level axis, and it applies to every dataset rather than to one.

BOTH REGIMES.

  injected      `outlier_p10` on the shared grid (0.0, 0.15, 0.3, 0.5, 0.75,
                1.0 of the clean label spread), three draws per level, one at
                level 0.0 on a regression set because the draw delivers exactly
                zero there. Detection is written here.
  as_they_come  nothing is corrupted. n18's regime. A win is a win on the data
                as it arrived, and there is no list to be right about.

WHAT THIS DOES NOT ASK. The features are fixed. One base representation and one
hybrid per family of stored search the pool holds, each built once per fold on
the fold's training molecules at their UNCLEANED labels. There are two
families: a search that chose its features to predict the label ('hybrid') and
one that chose them so that each molecule's activity is predicted by its five
nearest neighbours in the distance those features define
('hybrid_similarity'). Before 2026-09-19 only the first was built and the
second was written into the result file as a score nothing had measured in. No
cleaning reaches the feature ranking or the budget split in any space. Whether
it should is n20's question, not this one.

WHAT THE HELD-OUT MOLECULES ARE GRADED AGAINST. `labels[val_idx]`, untouched, in
every arm and both regimes. No cleaned label and no corrupted label is ever
scored against.

THE HEADLINE IS A COUNT OF FOLDS, not a mean, for n18's reason at :33-37. Every
comparison is PAIRED: the same fold, the same held-out molecules, two arms.

Usage:
    python n23_uncertainty_cleaning.py --dataset herg_ki --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import time
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS,
    get_best_search_for_downstream, make_nearest_neighbors,
    pick_noise_base_rep,
    # One feature space per search family the pool holds, added 2026-09-19.
    # n11's loop, in one place. See SPACE_FOR_FAMILY below.
    pool_recipes_by_family, describe_pool_family, POOL_SUBSETS,
)
from experiments.dataset_policy import (
    noise_levels, noise_groups_for, noise_dose_units,
)
from experiments.noise.noise_helpers import (
    noise_spec_for, inject_train_labels, detection_scores,
    detection_is_defined, hybrid_for_fold,
)
from experiments.noise.n07_remediation import _predict_for_metric
from kirby.utils.noise_estimation import get_noise_estimator, smooth_by_scores

# Which per-molecule number the cleaning is keyed on. The first two predict a
# spread per molecule. The third does not -- see the docstring. The fourth is
# the neighbour-based score the rest of the suite already uses, and is here to
# be beaten rather than to be tested.
SOURCES = ('hetero_gp', 'ngboost', 'heteroscedastic_nn', 'gaussian_process',
           'activity_cliff_aware')

# Which sources predict a molecule's noise FROM ITS FEATURES. Every one of these
# is scored out of fold, so it can only rank a held-out molecule when that
# molecule's own features say how noisy it is. See LEARNS_NOISE_FROM_FEATURES
# below and the condition note in the docstring.
LEARNS_NOISE_FROM_FEATURES = ('hetero_gp', 'ngboost', 'heteroscedastic_nn',
                              'gaussian_process')

# The dose-matched conditions whose corrupted molecules are chosen by something
# the features encode. Only `grouped_wider` is: it gives whole scaffold families
# the wider error. The three outlier conditions draw the corrupted molecules at
# RANDOM with no reference to the features (NOISE_AUDIT_2026_09_10.md 6.7), so a
# feature-based source is being asked something undefined under them -- it is not
# that the method failed. The five constant-width conditions give every molecule
# the same width, so nothing can rank anything under those.
NOISE_KEYED_TO_STRUCTURE = ('grouped_wider',)

# The share of training molecules `drop` removes. n18's, for n18's reason:
# nobody knows the right share here, so it is swept rather than looked up.
DROP_SHARES = (0.05, 0.10, 0.20)

# How far `move` pushes a label toward the model's own out-of-fold prediction.
MOVE_STRENGTHS = (0.25, 0.5, 1.0)

# The two rules, both run. See the docstring: one keeps the order of the
# molecules and drops the scale, the other uses the spread in label units.
MOVE_METHODS = ('proportional', 'confidence')

# How many inner folds each uncertainty estimate is computed over. Every source
# here scores a molecule from a model that did not see it, and the inner split
# is group-aware, so a molecule's own scaffold family never sets its own score.
ESTIMATOR_FOLDS = 5

# How many nearest neighbours the neighbour-based sources compare a molecule
# against, and how many the move target falls back to. n07's and n18's.
NEIGHBOUR_K = 10

# The condition the injected regime runs under. outlier_p10 corrupts a SUBSET,
# which is what makes detection_scores defined -- under gaussian, the student_t
# settings, laplace and grouped_shifted every training molecule moves.
DEFAULT_NOISE_CONDITION = 'outlier_p10'
NOISE_DRAWS = (0, 1, 2)
REGIMES = ('injected', 'as_they_come')

# Which searches the recipe may come from. Same as n02, n07, n11, n12, n13, n17,
# n18 and n19 since 2026-09-12.
POOL_SUBSET = 'predictive_and_similarity'

# ⛔ ONE SPACE PER FAMILY, NOT ONE RECIPE. Added 2026-09-19.
#
# POOL_SUBSETS names the two families in PREFERENCE order and the pool returns
# the first that has an entry, because their stored scores are in different
# units -- a predictive search's score is the dataset's own metric, a
# similarity search's is the correlation between each held-out molecule's
# activity and what its five nearest neighbours predict. A predictive recipe
# always exists, so until 2026-09-19 this file asked for both families and
# built one.
#
# It matters here, though less than in n12 or n17. One of the five uncertainty
# sources, activity_cliff_aware, scores a molecule against its nearest
# neighbours, and the similarity search chooses features so that exactly that
# distance predicts activity from the five nearest neighbours. The other four
# -- hetero_gp, ngboost, heteroscedastic_nn and gaussian_process -- fit a model
# on the features instead, so for them the space changes what the model is
# given rather than which neighbours are found.
#
# 'hybrid' is still the predictive recipe, so every number reported before
# 2026-09-19 still means what it meant.
HYBRID_SPACE = 'hybrid'
SPACE_FOR_FAMILY = {
    'predictive': HYBRID_SPACE,
    'similarity': 'hybrid_similarity',
}


def _estimator_kwargs(source, groups_train, is_clf, n_folds, k):
    """Only the constructor arguments this estimator actually takes.

    The two families take different ones and there is no shared signature. The
    model-based sources take `n_folds` and `groups`, and score each molecule
    from a model fitted on the OTHER inner folds. The neighbour-based ones take
    `k` and `metric`, and score a molecule against its neighbours inside the
    same training block with no inner split at all.

    Passing one family's arguments to the other raises, and until 2026-09-17
    this file did exactly that: every neighbour-based source failed at
    construction, wrote an error row, and produced no arm. Read the signature
    instead of guessing.
    """
    import inspect
    cls = get_noise_estimator.__globals__['ESTIMATOR_REGISTRY'][source]
    accepted = set(inspect.signature(cls.__init__).parameters)
    want = {'n_folds': n_folds, 'groups': groups_train, 'k': k,
            'metric': None}
    if is_clf:
        want['is_clf'] = True
    return {p: v for p, v in want.items() if p in accepted}


def _scored_out_of_fold(source):
    """Whether this source scores a molecule from a model that did not see it."""
    import inspect
    cls = get_noise_estimator.__globals__['ESTIMATOR_REGISTRY'][source]
    return 'n_folds' in set(inspect.signature(cls.__init__).parameters)


def _uncertainty(source, X_tr_s, y_tr, groups_train, is_clf, n_folds,
                 neighbour_k=10):
    """One spread per training molecule, and the denoised value to move toward.

    Returns (scores, target, target_source) or (None, None, why).

    `target` is the out-of-fold prediction the `move` shape pushes labels
    toward. The model-based sources produce one themselves and set
    `predictions_`; the neighbour-based ones do not, so their target is the mean
    of a molecule's ten nearest neighbours' labels -- which is what
    `n07.label_smooth` already moves toward. `target_source` is written onto
    every row, because a reader comparing two sources has to know whether they
    differ in the score alone or in the score and the target.
    """
    try:
        est = get_noise_estimator(
            source, **_estimator_kwargs(source, groups_train, is_clf,
                                        n_folds, neighbour_k))
    except Exception as exc:
        return None, None, f'{type(exc).__name__}: {exc}'

    try:
        scores = np.asarray(est.estimate(X_tr_s, y_tr), dtype=float)
    except Exception as exc:
        return None, None, f'{type(exc).__name__}: {exc}'
    if scores.shape[0] != y_tr.shape[0] or not np.all(np.isfinite(scores)):
        return None, None, 'the estimate was the wrong length or not finite'

    target = getattr(est, 'predictions_', None)
    if target is not None and np.shape(target) == np.shape(y_tr):
        return scores, np.asarray(target, dtype=float), 'the source own oof prediction'

    # No denoised value of its own: move toward the neighbours' mean instead,
    # the same target n07.label_smooth uses.
    try:
        k = min(neighbour_k, len(y_tr) - 1)
        nn = make_nearest_neighbors(n_neighbors=k)
        nn.fit(X_tr_s)
        _, nbr_idx = nn.kneighbors(X_tr_s)
        return (scores, np.mean(y_tr[nbr_idx[:, 1:]], axis=1),
                'the mean of the ten nearest neighbours')
    except Exception as exc:
        return scores, None, f'no target: {type(exc).__name__}: {exc}'


def cleaned_training_set(shape, setting, X_tr_s, y_tr, scores, target,
                         method=None):
    """What one shape hands the downstream fit. None when it cannot be applied.

    Returns (X, y) with the rows and labels the fit sees.

    `drop` removes the `setting` share of molecules with the widest predicted
    spread. `move` keeps every molecule and shifts its label toward `target`.
    Neither sets a per-row weight, and that is deliberate: a weight cannot be
    read against `drop` on one axis, and `n20` records that nothing in the
    feature selection reads one either.
    """
    if shape == 'none':
        return X_tr_s, y_tr
    if shape == 'drop':
        cut = np.percentile(scores, 100.0 * (1.0 - float(setting)))
        keep = scores < cut
        if keep.sum() < 10:
            return None
        return X_tr_s[keep], y_tr[keep]
    if shape == 'move':
        if target is None:
            return None
        return X_tr_s, smooth_by_scores(y_tr, scores, target,
                                        alpha=float(setting), method=method)
    raise ValueError(f"cleaned_training_set: unknown shape {shape!r}")


def _arms(drop_shares, move_strengths, move_methods):
    """Every (arm name, shape, setting, method). 'none' is the score to beat."""
    out = [('none', 'none', None, None)]
    for s in drop_shares:
        out.append((f'drop@{s:g}', 'drop', s, None))
    for m in move_methods:
        for s in move_strengths:
            out.append((f'move_{m}@{s:g}', 'move', s, m))
    return out


def _fit_score(factory, X_tr, y_tr, X_va, y_va, metric_fn, metric_name):
    """One fit, one held-out score against the untouched labels, or (None, why)."""
    try:
        m = factory()
        m.fit(X_tr, y_tr)
        return float(metric_fn(y_va, _predict_for_metric(m, X_va, metric_name))), None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def _levels_and_draws(dataset_name, condition, is_clf, levels=None, draws=None):
    """(level, draw) pairs for the injected regime.

    At level 0.0 on a regression dataset the draw delivers exactly zero, so
    three draws are three identical runs and one is kept. NOT on a
    classification set: the flip calibrator returns a floor rather than zero at
    a target of 0.0. Same rule as noise_helpers.noisy_search_specs.
    """
    levels = (noise_levels(dataset_name, strategy=condition, is_clf=is_clf)
              if levels is None else levels)
    draws = tuple(NOISE_DRAWS if draws is None else draws)
    out = []
    for lvl in levels:
        here = draws if (float(lvl) > 0 or is_clf) else draws[:1]
        for d in here:
            out.append((float(lvl), int(d)))
    return out


def run(data, config, output_dir, dataset_name, **kwargs):
    """N23: cleaning by a model's own per-molecule uncertainty."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N23: CAN A PER-MOLECULE UNCERTAINTY CLEAN LABELS")
    print("=" * 70)

    labels = np.asarray(data['labels'])
    reps = data['reps']
    groups = data.get('groups')
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    condition = kwargs.get('noise_condition') or DEFAULT_NOISE_CONDITION
    regimes = tuple(kwargs.get('regimes') or REGIMES)
    sources = tuple(kwargs.get('sources') or SOURCES)
    drop_shares = tuple(float(x) for x in
                        (kwargs.get('drop_shares') or DROP_SHARES))
    move_strengths = tuple(float(x) for x in
                           (kwargs.get('move_strengths') or MOVE_STRENGTHS))
    move_methods = tuple(kwargs.get('move_methods') or MOVE_METHODS)
    est_folds = int(kwargs.get('estimator_folds', ESTIMATOR_FOLDS))
    # The neighbour count the neighbour-based sources use, and the one
    # the move target falls back to. n07's and n18's.
    neighbour_k = int(kwargs.get('neighbour_k', NEIGHBOUR_K))

    downstream_models = list(kwargs.get('models') or DOWNSTREAM_MODELS)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n23: no downstream model could be built. Asked for "
            f"{list(kwargs.get('models') or DOWNSTREAM_MODELS)}, got "
            f"{sorted(model_factories)}.")

    # pick_noise_base_rep, not sorted(reps)[0]. The alphabetical pick is the
    # fallback that function was written to replace: it makes the control arm a
    # matter of spelling, and it can hand the control slot to a structure-tuned
    # sim_* recipe, which is a treatment this study built. n11, n17 and n21
    # already call it; these four did not. Changed 2026-09-18.
    base_rep_name = (kwargs.get('base_rep')
                     or pick_noise_base_rep(reps, dataset_name))
    noise_groups = noise_groups_for(data)[0]
    arms = _arms(drop_shares, move_strengths, move_methods)

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'regimes': list(regimes),
        'noise_condition': condition,
        'dose_units': noise_dose_units(dataset_name),
        'sources': list(sources),
        'shapes': ['none', 'drop', 'move'],
        'drop_shares': list(drop_shares),
        'move_strengths': list(move_strengths),
        'move_methods': list(move_methods),
        'arms': [a[0] for a in arms],
        'arm_to_beat': 'none',
        'models': downstream_models,
        'base_rep': base_rep_name,
        'estimator_folds': est_folds,
        'neighbour_k': neighbour_k,
        'claim_scope': (
            'Measures whether a per-molecule uncertainty estimate can clean '
            'labels. Every score is against held-out labels that were never '
            'cleaned and never corrupted. A win says the cleaning helped the '
            'model; under injection the detection block says separately '
            'whether the number found the molecules that actually moved.'),
        'two_terms_note': (
            "A model's uncertainty is how noisy the label looks plus how "
            'unsure the model is about that region. Only the first is a '
            'cleaning signal. Which models tell the two apart was measured in '
            '~/repos/qsar_qm_models/RERUN_PLAN.md sections 2.30 and 2.32, '
            'dated 2026-08-30 and 2026-08-31, on QM9 and not on this data: '
            'Gaussian process separation 11.2 to 18.7 over three seeds; '
            'NGBoost has no model-unsure term at all; variance-head network '
            '1.5 to 2.3; random forest 1.1, recorded as FAIL.'),
        'gaussian_process_is_not_that_number': (
            "KIRBy's gaussian_process estimator returns the posterior standard "
            'deviation plus the out-of-fold residual, which is the '
            'MODEL-UNSURE term, and its label-noise term is one number for the '
            'whole fold. Read out of noise_estimation.py:726-800 on '
            '2026-09-17. It is swept so the difference is measured here.'),
        'features_are_fixed_note': (
            "The hybrid is built once per fold on that fold's training "
            'molecules at their UNCLEANED labels, and every source, shape and '
            'setting shares it. No cleaning reaches the feature ranking or the '
            "budget split. Whether it should is n20's question."),
        'zero_noise_control_note': (
            'Level 0.0 under injection corrupts nothing, so there is nothing '
            'to clean. A source that still wins folds there is removing or '
            'moving the molecules the model finds hard.'),
        # The two families do not score a molecule the same way, and the
        # difference is not a setting anyone chose -- it is what each estimator
        # is. A model-based source fits on the other inner folds and scores the
        # held-out ones. A neighbour-based source compares a molecule to its
        # neighbours inside the same training block, with no inner split. So two
        # sources can differ in the score AND in whether the molecule's own
        # label helped produce it.
        'scored_out_of_fold_by_source': {
            s: bool(_scored_out_of_fold(s)) for s in sources},
        # Whether each source's number can mean anything under the condition
        # this run used. A source that predicts noise from the features needs a
        # condition that decides which molecules are noisy from something the
        # features encode; the three outlier conditions do not.
        'sources_defined_under_this_condition': {
            s: bool(s not in LEARNS_NOISE_FROM_FEATURES
                    or condition in NOISE_KEYED_TO_STRUCTURE)
            for s in sources},
        'condition_keys_noise_to_structure': bool(
            condition in NOISE_KEYED_TO_STRUCTURE),
        'sources_defined_note': (
            'False means this source predicts a molecule\'s noise from its '
            'features and the condition in use picked its corrupted molecules '
            'at random, so there is nothing in the features to generalise from '
            'and a chance-level reading is the condition rather than the '
            'method. Only grouped_wider keys the wider error to scaffold '
            'families. NOISE_AUDIT_2026_09_10.md 6.7.'),
        'scored_out_of_fold_note': (
            'True means the molecule was scored by a model fitted on the other '
            'inner folds, group-aware, so its own scaffold family did not set '
            'its own score. False means the score is a comparison against '
            'neighbours inside the same training block.'),
        'per_regime': {},
        'detection': {},
        'target_source_by_source': {},
        'estimator_seconds_by_source': {},
    }

    sample_X = next(iter(reps.values()))
    splits = list(cv_split(groups, n_folds, sample_X, labels, mode=mode))

    # ── The hybrid recipe, fetched once. n18's block, for n18's reason. ──
    bp = None
    try:
        _search, _meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model=downstream_models[0], budget=None,
            augmentations=data.get('augmentations'),
            dataset_name=dataset_name, task_type=task_type, data=data,
            pool_subset=POOL_SUBSET)
        bp = _meta['best_config']['blueprint'] if _meta else None
    except Exception as exc:
        _meta = None
        print(f"  hybrid recipe lookup failed: {type(exc).__name__}: {exc}")
    # One recipe per family the pool holds, and a written reason for each family
    # that produced none. `bp` stays the recipe the pool RETURNED, which is what
    # every 'hybrid_*' key below already described.
    hybrid_recipes, hybrid_unavailable = pool_recipes_by_family(
        _meta if bp is not None else None, SPACE_FOR_FAMILY,
        eligible_families=POOL_SUBSETS.get(POOL_SUBSET))
    recipe_for_space = {r['space']: r['blueprint'] for r in hybrid_recipes}
    results['hybrid_spaces_unavailable'] = hybrid_unavailable

    if not hybrid_recipes:
        spaces = ('base',)
        results['hybrid_available'] = False
        results['hybrid_unavailable_reason'] = (
            f'no recipe came back from the pool subset {POOL_SUBSET!r}, so '
            'every arm ran in the base representation only. Dropping the run '
            'would lose real answers because one of several spaces is missing.')
        print(f"  no hybrid recipe from pool subset {POOL_SUBSET}; "
              f"base representation only")
    else:
        spaces = ('base',) + tuple(r['space'] for r in hybrid_recipes)
        results['hybrid_available'] = True
        results['hybrid_family'] = _meta.get('objective')
        results['hybrid_budget'] = getattr(bp, 'budget', None)
        results['hybrid_reps'] = sorted(getattr(bp, 'reps', []) or [])
        results['hybrid_space_of_pool_winner'] = SPACE_FOR_FAMILY.get(
            _meta.get('objective'), f"hybrid_{_meta.get('objective')}")
        results['hybrid_spaces'] = convert_numpy({
            r['space']: {
                'built_under_objective': r['objective'],
                'features_chosen_by': describe_pool_family(r['objective']),
                'is_the_recipe_the_pool_returned': r['is_pool_winner'],
                'tier': r['tier'], 'aug_condition': r['aug_condition'],
                'budget': r['budget'],
                'allocation_method': r['allocation_method'],
                'importance_method': r['importance_method'],
                'reps': sorted(getattr(r['blueprint'], 'reps', []) or []),
                'search_score_mean': r['search_score_mean'],
                'search_metric_name': r['search_metric_name'],
            } for r in hybrid_recipes})
        results['hybrid_spaces_note'] = (
            'One hybrid space per family of stored search. A predictive search '
            'chose its features to predict the label and its search_score_mean '
            "is in the dataset's own metric. A similarity search chose its "
            "features so that each molecule's activity is predicted by its "
            'five nearest neighbours in the distance those features define, '
            'fits no downstream model, and its search_score_mean is that '
            'correlation. The two search scores are in different units and '
            'nothing here ranks one against the other. Every arm below is '
            "scored in the dataset's own metric, on the same folds, in every "
            'space.')
        print(f"  hybrid recipe: {len(results['hybrid_reps'])} "
              f"representations, budget {results['hybrid_budget']}, "
              f"searched under objective {_meta.get('objective')}")
        for r in hybrid_recipes:
            print(f"  hybrid space {r['space']}: family {r['objective']!r}, "
                  f"budget {r['budget']} features, allocation "
                  f"{r['allocation_method']}, search score "
                  f"{r['search_score_mean']} in {r['search_metric_name']!r}")
    for _sp, _why in hybrid_unavailable.items():
        print(f"  no {_sp} space: {_why}")
    results['spaces_run'] = list(spaces)

    passes = {'as_they_come': [(None, 0)]}
    if 'injected' in regimes:
        passes['injected'] = _levels_and_draws(
            dataset_name, condition, is_clf,
            levels=kwargs.get('levels'), draws=kwargs.get('draws'))
        results['noise_levels'] = sorted({l for l, _ in passes['injected']})

    n_passes = sum(len(passes[r]) for r in regimes)
    print(f"  {len(labels)} molecules, {len(splits)} folds, "
          f"models {downstream_models}")
    print(f"  sources {list(sources)}")
    print(f"  {len(arms)} arms: 1 to beat, {len(drop_shares)} drop, "
          f"{len(move_methods) * len(move_strengths)} move")
    print(f"  {n_passes} (level, draw) passes over {len(regimes)} regimes; "
          f"{n_passes * len(splits) * len(sources) * len(spaces)} uncertainty "
          f"estimates")

    # acc[regime][level][space][source][arm][model][fold_key] = score
    acc = {}
    errors = {}
    detection_acc = {}
    est_seconds = {}
    target_src = {}
    t0 = time.time()

    for regime in regimes:
        if regime not in passes:
            raise ValueError(f"n23: unknown regime {regime!r}")
        acc[regime] = {}
        for level, draw in passes[regime]:
            lvl_key = 'none' if level is None else f'{level:g}'
            tag = (regime if level is None
                   else f'{regime}/{condition}/level {level:g}/draw {draw}')
            print(f"\n  --- {tag} ---")
            cell = acc[regime].setdefault(lvl_key, {})
            spec = (None if regime != 'injected'
                    else noise_spec_for(dataset_name, condition, level,
                                        noise_seed=draw, is_clf=is_clf))

            for fold_i, (train_idx, val_idx) in enumerate(splits):
                fold_key = f'l{lvl_key}d{draw}f{fold_i}'
                y_val = labels[val_idx]
                g_tr = groups[train_idx] if groups is not None else None

                cmask = None
                if spec is None:
                    y0 = np.asarray(labels[train_idx], dtype=float)
                else:
                    try:
                        y0, cmask, _prov = inject_train_labels(
                            labels, spec, train_idx=train_idx, is_clf=is_clf,
                            groups=noise_groups, dataset_name=dataset_name)
                    except Exception as exc:
                        errors[f'{regime}|{lvl_key}|d{draw}|f{fold_i}|inject'] = (
                            f'{type(exc).__name__}: {exc}')
                        continue
                    y0 = np.asarray(y0, dtype=float)

                train_reps = {k: np.asarray(v[train_idx], dtype=np.float64)
                              for k, v in reps.items()}
                val_reps = {k: np.asarray(v[val_idx], dtype=np.float64)
                            for k, v in reps.items()}

                # The two feature spaces, both built from this fold's training
                # molecules at their UNCLEANED labels, shared by every arm.
                fold_spaces = {}
                sc = StandardScaler()
                fold_spaces['base'] = (
                    sc.fit_transform(train_reps[base_rep_name]),
                    sc.transform(val_reps[base_rep_name]))
                # One build per family the pool holds. hybrid_for_fold reads
                # the objective off the recipe and drops the model, the metric
                # and its direction under a geometry recipe, where
                # build_fold_hybrid refuses the pair, so one call serves both.
                for _sp in spaces:
                    _bp = recipe_for_space.get(_sp)
                    if _bp is None:
                        continue
                    X_tr_h, X_va_h = hybrid_for_fold(
                        train_reps, val_reps, y0, _bp, g_tr,
                        model_factories[downstream_models[0]](),
                        metric_fn, lower_is_better)
                    if X_tr_h is not None:
                        sh = StandardScaler()
                        fold_spaces[_sp] = (sh.fit_transform(X_tr_h),
                                            sh.transform(X_va_h))

                for space in spaces:
                    if space not in fold_spaces:
                        continue
                    X_tr_s, X_va_s = fold_spaces[space]
                    space_cell = cell.setdefault(space, {})

                    for source in sources:
                        t_est = time.time()
                        scores, target, tsrc = _uncertainty(
                            source, X_tr_s, y0, g_tr, is_clf, est_folds,
                            neighbour_k=neighbour_k)
                        est_seconds.setdefault(source, []).append(
                            time.time() - t_est)
                        if scores is None:
                            errors[f'{regime}|{lvl_key}|{space}|{source}|'
                                   f'f{fold_i}|estimate'] = tsrc
                            print(f"      {space}/{source}: {tsrc}")
                            continue
                        target_src.setdefault(source, set()).add(tsrc)

                        # Did the number find the molecules that really moved,
                        # before any cleaning ran? It costs no model fit.
                        if cmask is not None and detection_is_defined(
                                spec.strategy):
                            det = detection_scores(cmask, scores,
                                                   spec.strategy)
                            if det is not None:
                                (detection_acc
                                 .setdefault(lvl_key, {})
                                 .setdefault(space, {})
                                 .setdefault(source, []).append(det))

                        src_cell = space_cell.setdefault(source, {})
                        for arm, shape, setting, method in arms:
                            try:
                                got = cleaned_training_set(
                                    shape, setting, X_tr_s, y0, scores,
                                    target, method=method)
                            except Exception as exc:
                                errors[f'{regime}|{lvl_key}|{space}|{source}|'
                                       f'{arm}|f{fold_i}|clean'] = (
                                    f'{type(exc).__name__}: {exc}')
                                continue
                            if got is None:
                                errors[f'{regime}|{lvl_key}|{space}|{source}|'
                                       f'{arm}|f{fold_i}|clean'] = (
                                    'the shape could not be applied on this fold')
                                continue
                            Xc, yc = got
                            for mname in downstream_models:
                                s, err = _fit_score(
                                    model_factories[mname], Xc, yc, X_va_s,
                                    y_val, metric_fn, metric_name)
                                if s is None:
                                    errors[f'{regime}|{lvl_key}|{space}|'
                                           f'{source}|{arm}|{mname}|'
                                           f'f{fold_i}'] = err
                                    continue
                                (src_cell.setdefault(arm, {})
                                         .setdefault(mname, {})[fold_key]) = s

                print(f"    fold {fold_i} done [{time.time() - t0:.0f}s]")

    results['per_regime'] = convert_numpy(acc)
    results['detection'] = convert_numpy({
        lvl: {sp: {src: _mean_detection(rows) for src, rows in by_src.items()}
              for sp, by_src in by_sp.items()}
        for lvl, by_sp in detection_acc.items()})
    results['target_source_by_source'] = {k: sorted(v)
                                          for k, v in target_src.items()}
    results['estimator_seconds_by_source'] = {
        k: {'mean': float(np.mean(v)), 'n_estimates': len(v),
            'total': float(np.sum(v))}
        for k, v in est_seconds.items()}
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy(
        _summarise(acc, [a[0] for a in arms], downstream_models, metric_name,
                   lower_is_better, results['detection']))
    _print_summary(results, metric_name)
    return results


def _mean_detection(rows):
    """Average the detection readings over the folds and draws."""
    if not rows:
        return None
    keys = ('detection_average_precision', 'detection_precision_at_k',
            'detection_recall_at_k', 'detection_n_corrupted',
            'detection_n_training')
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows
                if r.get(k) is not None and np.isfinite(float(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float('nan')
    out['detection_condition'] = rows[0].get('detection_condition')
    out['detection_n_passes'] = len(rows)
    return out


def _paired(base, got, lower_is_better):
    """One arm against `none`, on the folds both scored. Positive means better."""
    shared = sorted(set(base) & set(got))
    gains = np.array([(base[k] - got[k]) if lower_is_better
                      else (got[k] - base[k]) for k in shared], dtype=float)
    gains = gains[np.isfinite(gains)]
    scale = max(abs(float(np.mean(list(base.values())))) if base else 1.0, 1.0)
    tol = 1e-9 * scale
    return {
        'n_folds_paired': int(gains.size),
        'folds_won': int((gains > tol).sum()),
        'folds_lost': int((gains < -tol).sum()),
        # A large tied count means the arm is the uncleaned fit under a
        # different name -- a move at a strength that shifts nothing, say.
        'folds_tied': int((np.abs(gains) <= tol).sum()),
        'tie_tolerance': float(tol),
        'mean_gain': float(gains.mean()) if gains.size else float('nan'),
        'spread_over_fold_gains': (float(gains.std()) if gains.size
                                   else float('nan')),
        'fold_gains': [float(x) for x in gains],
        # A gain no larger than the spread of the per-fold gains behind it is
        # not a gain. Two measured numbers compared, not a test: the suite's
        # standing rule at this replicate count is raw spread only.
        'gain_exceeds_its_own_spread': bool(
            gains.size > 0 and gains.mean() > gains.std()),
    }


def _summarise(acc, arm_names, models, metric_name, lower_is_better, detection):
    """Every arm against `none`, paired per fold, plus what the zero-noise row says.

    THE HEADLINE IS A COUNT OF FOLDS, for n18's reason at :33-37. A mean over
    folds hides an arm that wins big on one fold and loses on the rest.

    Two things are rolled up beside it.

      settings_winning_most_folds  how many (arm, model) settings a source won a
                                   majority of folds at. A source that only wins
                                   at one setting has shown less than one that
                                   wins at several.
      zero_noise_wins              the same count at level 0.0, where nothing
                                   was corrupted. Anything above zero there is
                                   the cleaning removing or moving the molecules
                                   the model finds hard, so a source's real
                                   count has to be read against this one.
    """
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'headline': 'folds won against no cleaning, paired per fold',
           'per_regime': {}}

    for regime, by_level in acc.items():
        reg = {'per_level': {}}
        for lvl, by_space in by_level.items():
            lrow = {'per_space': {}}
            for space, by_source in by_space.items():
                srow = {'per_source': {}}
                for source, by_arm in by_source.items():
                    base_by_model = by_arm.get('none') or {}
                    rows, per_arm = [], {}
                    for arm in arm_names:
                        if arm == 'none':
                            continue
                        per_model = {}
                        for m in models:
                            b = base_by_model.get(m) or {}
                            g = (by_arm.get(arm) or {}).get(m) or {}
                            if not b or not g:
                                continue
                            cellv = _paired(b, g, lower_is_better)
                            per_model[m] = cellv
                            rows.append((arm, m, cellv))
                        if per_model:
                            per_arm[arm] = {'per_model': per_model}
                    majority = [r for r in rows
                                if r[2]['folds_won'] * 2 > r[2]['n_folds_paired']]
                    swept = [r for r in rows
                             if r[2]['n_folds_paired'] > 0
                             and r[2]['folds_won'] == r[2]['n_folds_paired']]
                    clears = [r for r in rows
                              if r[2]['gain_exceeds_its_own_spread']]
                    best = (max(rows, key=lambda r: (
                        r[2]['folds_won'] / max(r[2]['n_folds_paired'], 1),
                        r[2]['mean_gain'])) if rows else None)
                    srow['per_source'][source] = {
                        'per_arm': per_arm,
                        'uncleaned_mean_by_model': {
                            m: (float(np.mean(list(v.values()))) if v
                                else float('nan'))
                            for m, v in base_by_model.items()},
                        'settings_scored': len(rows),
                        'settings_winning_most_folds': len(majority),
                        'settings_winning_every_fold': len(swept),
                        'settings_whose_gain_clears_its_own_spread': len(clears),
                        'best_setting': (None if best is None else {
                            'arm': best[0], 'model': best[1],
                            'folds_won': best[2]['folds_won'],
                            'n_folds_paired': best[2]['n_folds_paired'],
                            'mean_gain': best[2]['mean_gain'],
                            'spread_over_fold_gains':
                                best[2]['spread_over_fold_gains']}),
                        # How well this number found the molecules that moved,
                        # before any cleaning. Absent where nothing was
                        # corrupted or where every molecule moved.
                        'detection': (detection.get(lvl, {}).get(space, {})
                                      .get(source)),
                    }
                lrow['per_space'][space] = srow
            reg['per_level'][lvl] = lrow
        out['per_regime'][regime] = reg

    # The zero-noise control, rolled up per source so it sits beside the counts.
    zero = (out['per_regime'].get('injected', {}).get('per_level', {})
            .get('0', {}).get('per_space', {}))
    out['zero_noise_control'] = {
        'what_it_says': (
            'Level 0.0 corrupts nothing, so there is nothing to clean. A '
            'source winning folds here is removing or moving the molecules the '
            'model finds hard, and its count at every other level has to be '
            'read against this one.'),
        'settings_winning_most_folds_at_level_zero': {
            f'{space}/{source}': blk['settings_winning_most_folds']
            for space, srow in zero.items()
            for source, blk in srow['per_source'].items()},
    } if zero else {'what_it_says': 'level 0.0 did not run'}
    return out


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric_name}; paired per fold against no cleaning)")
    if not results.get('hybrid_available', True):
        print("  no hybrid recipe was available; base representation only")
    secs = results.get('estimator_seconds_by_source') or {}
    if secs:
        print("  seconds per uncertainty estimate: "
              + ', '.join(f"{k} {v['mean']:.1f}" for k, v in secs.items()))
    tgt = results.get('target_source_by_source') or {}
    for src, where in tgt.items():
        print(f"    {src}: move pushes labels toward {', '.join(where)}")

    for regime, reg in s['per_regime'].items():
        print(f"\n  regime {regime}:")
        for lvl, lrow in reg['per_level'].items():
            print(f"    level {lvl}:")
            for space, srow in lrow['per_space'].items():
                for source, blk in srow['per_source'].items():
                    b = blk['best_setting']
                    det = blk.get('detection')
                    detstr = ''
                    if det and np.isfinite(det.get('detection_average_precision',
                                                   float('nan'))):
                        detstr = (f"; found the moved molecules at average "
                                  f"precision {det['detection_average_precision']:.3f} "
                                  f"over {det['detection_n_corrupted']:.0f} "
                                  f"of {det['detection_n_training']:.0f} "
                                  f"training molecules")
                    print(f"      {space:6s} {source:20s} won most folds in "
                          f"{blk['settings_winning_most_folds']} of "
                          f"{blk['settings_scored']} settings, cleared its own "
                          f"spread in "
                          f"{blk['settings_whose_gain_clears_its_own_spread']}"
                          f"{detstr}")
                    if b:
                        print(f"             best: {b['arm']}/{b['model']} — "
                              f"{b['folds_won']} of {b['n_folds_paired']} "
                              f"folds, mean {b['mean_gain']:+.4f} "
                              f"(spread {b['spread_over_fold_gains']:.4f})")
    z = s.get('zero_noise_control', {}).get(
        'settings_winning_most_folds_at_level_zero')
    if z:
        print("\n  the zero-noise control — nothing was corrupted, so nothing "
              "here is cleaning:")
        for k, v in z.items():
            print(f"    {k}: won most folds in {v} settings")
    print("\n  The count of folds won is the headline. A mean gain no larger "
          "than the\n  spread of the per-fold gains behind it is not a gain.")


def main():
    parser = create_experiment_parser(
        'N23: can a per-molecule uncertainty clean labels')
    parser.add_argument('--sources', type=str, default=None,
                        help='Comma-separated uncertainty sources')
    parser.add_argument('--noise-condition', type=str, default=None,
                        help=f'Injected condition (default {DEFAULT_NOISE_CONDITION})')
    parser.add_argument('--regimes', type=str, default=None,
                        help='Comma-separated: injected, as_they_come')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, rep_preset=args.reps)
    kw = {}
    if args.sources:
        kw['sources'] = [x.strip() for x in args.sources.split(',')]
    if args.noise_condition:
        kw['noise_condition'] = args.noise_condition
    if args.regimes:
        kw['regimes'] = [x.strip() for x in args.regimes.split(',')]
    res = run(data, config, args.output, args.dataset, **kw)
    save_experiment_results(res, args.output, 'n23_uncertainty_cleaning',
                            args.dataset,
                            filename_suffix=kw.get('noise_condition'))


if __name__ == '__main__':
    main()
