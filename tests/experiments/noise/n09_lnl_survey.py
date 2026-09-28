#!/usr/bin/env python3
"""N09: LNL Method Survey with Hybrids — Learning with Noisy Labels methods.

Paper: Noise

Scope / claim (honest framing): this experiment asks a RELATIVE question —
does adding the hybrid representation change the RANKING of learning-with-
noisy-labels (LNL) methods — under SYNTHETIC noise injected on the TRAINING
labels only (a dose-matched condition from noiseInject for regression,
label-flipping for classification; see noise_helpers.inject_train_labels).
The condition defaults to gaussian, which is what this experiment has always
injected; the other ten are reachable by passing noise_condition=. Validation
labels are left clean. It does
NOT establish an absolute noise-robustness result for any method or rep: the
signal of interest is purely whether the base→hybrid (and hybrid→weighted-
hybrid) swap re-orders which LNL methods come out on top.

Methods tested:
  - 12 total in LNL_REGISTRY
  - 10 support both regression and classification (MSE↔CE swap inside the
    method, see lnl_methods.py).
  - 2 are regression-only (heteroscedastic_gp, evidential_kernel); they are
    auto-filtered when task_type='classification' via REGRESSION_ONLY_LNL.

6 conditions × 5 noise levels × {12 or 10} methods:
  A: base_<k>_no_lnl        — Single representation + standard DNN (baseline)
  B: hybrid_no_lnl          — Hybrid + standard DNN
  C: base_<k>_lnl           — Single representation + each LNL method
  D: hybrid_lnl             — Hybrid + each LNL method
  E: weighted_hybrid_no_lnl — Importance-weighted hybrid + standard DNN
  F: weighted_hybrid_lnl    — Importance-weighted hybrid + each LNL method

THE SINGLE-REPRESENTATION COMPARISON (A and C) is run once per rank, k = 1 and
2: the top 2 single representations for the DNN, ranked by the DNN's own score
in the stored search (_shared.noise_base_reps_by_model, owner's decision
2026-09-25). The DNN is the ranking model because it is the network every
condition here trains. Which representations those are is written to
`base_reps_by_model`, and `base_reps_ranked_by` says the model.

Outputs per-sigma:
  - delta_from_clean: signed (noisy_score - clean_score); always defined.
  - score_ratio_pct: ratio*100 when both sides are positive, else NaN.
    (Renamed from `retention_pct`: it is a plain ratio, NOT bounded to 100 — it
    legitimately exceeds 100% when denoising beats the clean baseline. R² and
    MCC can be negative, so a naive ratio was historically misleading.)

Feature selection (AUDIT P1#5 — this is the FAIR version, unlike n10):
the hybrid feature mix is re-searched per (noise level, data split) on the
NOISY training labels using `find_best_hybrid` with a reduced budget grid
(NOISY_SEARCH_BUDGETS). The previous version pulled the mix from a precomputed
pool that ran on CLEAN labels, which gave the methods oracle access to ground
truth they wouldn't have in a real noisy-label setting; the per-fold re-search
removes that clean-label oracle so BOTH the blueprint selection (which reps,
which allocation/importance strategy) AND the feature construction see only
noisy labels. The clean-label pool blueprint `bp` survives only as a fallback
if the inner search fails.

⛔ THE INNER SEARCH NOW FITS THE NEURAL NETWORK, CHANGED 2026-09-28. It used to
fit a random forest while every arm the recipe feeds — the plain baseline and all
twelve label-noise methods — is a neural network, so the recipe was chosen by one
learner and scored with another on every fold of every pass.

⛔ AND THIS IS THE COST THE OLD CHOICE WAS PAYING FOR. The line this replaces
said a neural-network search would be "more matched but ~10x slower per (noise
level, fold)". That estimate was written here, not measured in this session, and
it is the honest thing to expect from the change. What was measured, on
2026-09-21 with the random forest: 677 seconds for one fold of one noise level on
flab_thermo_tresanco, 448 antibodies trained on, against 3.6 seconds to read the
stored search back. Runtime is still held down by the trimmed
NOISY_SEARCH_BUDGETS grid and the small inner fold count.

⛔ EVERY STORED n09 SEARCH IS STALE. The model list is part of what identifies a
search, so the random-forest files under tier 'n09_noisy_search' no longer match
and are rebuilt.

Usage:
    python n09_lnl_survey.py --dataset logd --mode intermediary
    python n09_lnl_survey.py --dataset qm9 --mode full
"""
import sys
import os
import hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.dataset_policy import (
    noise_groups_for, noise_levels as dataset_noise_levels,
)
from kirby.noise_spec import GROUPED_STRATEGIES
from experiments.noise.noise_helpers import (
    inject_train_labels, noise_spec_for, detection_scores, sweep_conditions, conditions_for_task)
from experiments._shared import (
    noise_base_reps_by_model, NOISE_BASE_REP_COUNT,
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories,
    get_best_search_for_downstream, build_fold_hybrid,
    extract_top3_base_reps,
)
from kirby.utils.lnl_methods import (
    get_lnl_method, LNL_REGISTRY, REGRESSION_ONLY_LNL,
    _standard_dnn_train_predict, _standard_dnn_train_predict_clf,
)
from kirby.find_best_hybrid import find_best_hybrid

# Noise levels. `None` means the grid this dataset is anchored to in
# experiments.dataset_policy: the published assay error in the label's own units for
# herg_ki, caco2 and logd, a fraction of the label SD otherwise.
SIGMAS = None

# The noise condition. Gaussian is what this experiment has always injected, so this
# default preserves what it measures; the other ten are reachable by passing
# noise_condition=..., one run each, and censoring takes the censored-fraction grid.
NOISE_CONDITION = 'gaussian'
ALL_METHODS = list(LNL_REGISTRY.keys())

# The model whose ranking picks the single representations for conditions A and
# C. Every condition in this file trains the same DNN (_run_baseline, and the LNL
# methods' own networks), so its ranking is the one that applies.
BASE_RANK_MODEL = 'dnn'

# Reduced budget grid for the per-(noise level, fold) feature search.
# The full pool covers [10, 25, 50, 100, 250, 500, 1000, 2000]; we trim to
# three representative budgets here because the search runs inside the
# noise injection loop and full-grid runtime would be prohibitive. Pattern
# mirrors c06's per-variant searches with reduced budgets (the inner search
# runs the standard config with this trimmed budget grid to stay tractable).
NOISY_SEARCH_BUDGETS = [50, 100, 250]


# The private `_inject_noise` this module used to carry has been deleted. It added
# Gaussian noise at `sigma_frac * (max(y) - min(y))` -- a fraction of the label RANGE,
# which is roughly 4-6x the label SD, so a level of 0.3 here delivered far more noise
# than a level of 0.3 anywhere else in the repo. Injection now goes through
# `noise_helpers.inject_train_labels`, which is the same definition the hybrid search,
# c10 and the Rust half of the study all use, and which records what was delivered.


class _BaselineDNN:
    """The plain DNN of `_run_baseline`, with a fit/predict interface.

    create_hybrid's accept-or-reject step deep-copies whatever model it is
    given, fits it on the candidate feature set and scores it with the
    dataset's own metric. Without a model it falls back to the internal
    evaluator scoring r2 (regression) or accuracy (classification), so the
    features are kept or dropped by a different learner and a different
    quantity than the one the condition is then measured with.

    `_standard_dnn_train_predict` trains and predicts in one call, so fit
    stores the training rows and predict runs the network. Same
    hyperparameters, same random_state and — on a classification set — the
    same P(y=positive) output as `_run_baseline`, which is what AUROC needs.
    Holds arrays only, so deepcopy works.
    """

    def __init__(self, random_state=42, is_clf=False, classes_=None):
        self.random_state = random_state
        self.is_clf = is_clf
        self.classes_in = classes_

    def get_params(self, deep=True):
        return {'random_state': self.random_state, 'is_clf': self.is_clf,
                'classes_': self.classes_in}

    def fit(self, X, y, groups=None):
        self.X_train_ = np.asarray(X)
        self.y_train_ = np.asarray(y)
        self.groups_ = (np.asarray(groups) if groups is not None
                        and len(groups) == len(self.y_train_) else None)
        return self

    def predict(self, X):
        return _run_baseline(self.X_train_, self.y_train_, np.asarray(X),
                             None, random_state=self.random_state,
                             is_clf=self.is_clf, classes_=self.classes_in,
                             groups=getattr(self, 'groups_', None))


def _run_baseline(X_train, y_train, X_val, y_val, random_state=42,
                  is_clf=False, classes_=None, groups=None):
    """Standard DNN baseline (no LNL). Routes to clf trainer when is_clf=True.

    `groups` are the scaffold groups of the training rows. They make the
    network's early-stopping holdout scaffold-disjoint. Until 2026-09-28 this
    function passed none, so the plain DNN stopped on a RANDOM 20% of its
    training rows. Seven of the twelve methods stop on a scaffold-disjoint 20%
    (LNLMethod._carve_selection), and heteroscedastic_gp's final fit does too.
    The headline gain (method minus plain DNN, same labels) therefore carried a
    difference in stopping rule that is not part of any method. With the same
    groups, GroupShuffleSplit(random_state=42, test_size=0.2) gives the plain
    DNN the same carve those methods use.
    """
    if is_clf:
        return _standard_dnn_train_predict_clf(
            X_train, y_train, X_val,
            epochs=200, hidden_sizes=(256, 128, 64),
            dropout=0.2, lr=0.001, batch_size=256,
            patience=20, random_state=random_state,
            classes_=classes_, groups=groups,
        )
    return _standard_dnn_train_predict(
        X_train, y_train, X_val,
        epochs=200, hidden_sizes=(256, 128, 64),
        dropout=0.2, lr=0.001, batch_size=256,
        patience=20, random_state=random_state, groups=groups,
    )


def _record_detection(acc, cond_key, method_name, result, corrupted_mask,
                      noise_condition):
    """Grade one method's per-molecule suspicion score, if it exposes one.

    Three of the twelve learning methods rank every training molecule by how wrong
    its label looks -- `small_loss` by its own squared error, `confident_learning`
    by its error against an ensemble, `distance_selection` by how far it sits from
    its neighbours. Each now hands that ranking out in its metadata. Until
    2026-09-12 all three were graded only by the score after they threw molecules
    away, never on whether they threw the right ones away.

    The other nine expose no per-molecule ranking, so nothing is recorded for them.
    Nothing is recorded either under the six conditions where every training
    molecule moves -- see `noise_helpers.DETECTION_IS_DEFINED`.

    The suspicion array itself is read and dropped. Only the three numbers are kept.
    """
    if corrupted_mask is None:
        return
    sus = (result or {}).get('metadata', {}).get('suspicion_scores')
    if sus is None:
        return
    det = detection_scores(corrupted_mask, sus, noise_condition)
    if det is not None:
        acc.setdefault(cond_key, {}).setdefault(method_name, []).append(det)


def _mean_detection(rows):
    """Average the detection readings over the (fold, draw) passes."""
    if not rows:
        return None
    keys = ('detection_average_precision', 'detection_precision_at_k',
            'detection_recall_at_k', 'detection_n_corrupted', 'detection_n_training')
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if r.get(k) is not None
                and np.isfinite(float(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float('nan')
    out['detection_condition'] = rows[0].get('detection_condition')
    out['detection_n_passes'] = len(rows)
    return out


def run(data, config, output_dir, dataset_name, **kwargs):
    """n09_lnl_survey across every condition this dataset can be given.

    The condition is an axis this experiment sweeps, not something a caller
    runs it once per. The sweep, the per-condition checkpointing, the resume
    and the stride slice are noise_helpers.sweep_conditions.

    Its headline is a downstream score, which is defined under every condition,
    so it sweeps the whole list rather than only the ones where a corrupted
    SUBSET exists. It also records detection, and detection_scores already
    returns None where that is undefined.
    """
    return sweep_conditions(
        _run_one_condition, 'n09_lnl_survey',
        data, config, output_dir, dataset_name,
        default_conditions=conditions_for_task(data), kwargs=kwargs,
        groups_needed=GROUPED_STRATEGIES)


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    print("\n" + "=" * 70)
    print("EXPERIMENT N09: LNL METHOD SURVEY WITH HYBRIDS")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    noise_condition = kwargs.get('noise_condition', NOISE_CONDITION)
    # Three corruptions per fold, averaged — matching n07, n11 and n13.
    # User decision 2026-09-12. Overridable for a smoke.
    NOISE_SEEDS = kwargs.get('noise_seeds', [0, 1, 2])
    sigmas = kwargs.get('sigmas') or kwargs.get('levels') or SIGMAS
    if sigmas is None:
        # detect_grid=True: on censoring this asks for the four censored fractions
        # this experiment runs (0.0, 0.10, 0.25, 0.40), not n13's full six. Owner's
        # decision 2026-09-12. It changes nothing for the other ten conditions.
        sigmas = list(dataset_noise_levels(dataset_name, strategy=noise_condition,
                                           detect_grid=True, is_clf=is_clf))
    noise_groups, _n_ng, _lg = noise_groups_for(data)
    noise_provenance = {}
    methods_to_test = kwargs.get('methods', ALL_METHODS)

    # heteroscedastic_gp and evidential_kernel have no classification analog
    # (continuous-noise-variance / NIG distribution math doesn't transform via
    # a loss-function swap). All 10 other methods support clf via MSE→CE in
    # their inner training. See REGRESSION_ONLY_LNL in lnl_methods.py.
    if is_clf:
        methods_to_test = [m for m in methods_to_test
                           if m not in REGRESSION_ONLY_LNL]

    # Resolve the full class set up front so per-fold y_train subsets all
    # train classifiers with the same n_classes outputs.
    classes_ = np.unique(labels) if is_clf else None

    sample_X = next(iter(reps.values()))

    # The clean-label recipe, read once. It is the fallback the per-fold recipe
    # drops to when the inner search on the noisy labels returns nothing.
    #
    # ⛔ THE RANDOM-FOREST FALLBACK IS GONE, 2026-09-28. Two things disagreed.
    # Every arm in this file is a neural network -- the plain baseline and all
    # twelve label-noise methods -- and this lookup asked for the neural
    # network's recipe. When that came back empty a second lookup asked for the
    # random forest's, and that recipe became `bp`, which is both the per-fold
    # recipe fallback at the `inner_bp` line below and the source of the base
    # representation ranking. So on any dataset whose stored search holds no
    # neural-network entry, every number in the file came from features chosen
    # for a random forest, and nothing said so.
    #
    # There is now no second lookup. With no neural-network recipe, `bp` is None,
    # the per-fold recipe has no fallback and the fold says so.
    search, meta = get_best_search_for_downstream(
        reps, labels, groups,
        downstream_model='dnn', budget=None,
        augmentations=augmentations,
        dataset_name=dataset_name, task_type=task_type,
        data=data,
    )
    _dnn_search_found = meta is not None
    if meta is None:
        print("    no stored recipe for the neural network on this dataset -- "
              "the clean-label fallback recipe is absent, so a fold whose inner "
              "search returns nothing is not scored. The dataset needs a "
              "precompute pass that scores dnn.")
    top3 = []
    if search is not None:
        top3 = extract_top3_base_reps(search)
        if top3:
            # extract_top3_base_reps ranks by RF baselines regardless of which
            # downstream model the blueprint pool selected — make that explicit
            # in the log so readers don't conflate the two.
            print(f"  Top-3 base reps (ranked by RF baseline): "
                  f"{', '.join(f'{n}={s:.4f}' for n, s in top3)}")
    bp = meta['best_config']['blueprint'] if meta else None
    if bp is None:
        print("  WARNING: No blueprint found, skipping")
        return {'error': 'no_blueprint', 'metric': metric_name}

    # Per-fold weights computed from feat_info (not global) — see conditions E/F below

    # ── The single-representation comparison: the DNN's top 2 ──
    # Ranked by the DNN's own score in the stored search fetched above, which is
    # now the only search this function reads -- the sentence that used to sit
    # here described the random-forest fallback, and that fallback was deleted on
    # 2026-09-28. With no DNN search stored, `search` is None and this ranking is
    # empty rather than borrowed. A `base_reps` kwarg ({model: [reps]}) overrides
    # the ranking, for a smoke or a direct caller.
    _base_override = kwargs.get('base_reps')
    if _base_override is not None:
        base_by_model = ({BASE_RANK_MODEL: list(_base_override[BASE_RANK_MODEL])}
                         if _base_override.get(BASE_RANK_MODEL) else {})
        base_missing = ({} if base_by_model else
                        {BASE_RANK_MODEL: 'not given by the caller'})
    else:
        base_by_model, base_missing = noise_base_reps_by_model(
            {BASE_RANK_MODEL: search}, [BASE_RANK_MODEL], reps=reps,
            dataset_name=dataset_name)
    # [(rank, representation), ...] -- rank 1 first. Shorter than
    # NOISE_BASE_REP_COUNT, or empty, when fewer representations carry a DNN
    # score; conditions A and C then run for the ranks that exist.
    base_ranks = list(enumerate(
        (base_by_model.get(BASE_RANK_MODEL) or [])[:NOISE_BASE_REP_COUNT],
        start=1))
    base_keys = [f'base_{k}' for k, _ in base_ranks]

    print(f"  Dataset: {dataset_name} ({len(labels)} samples)")
    print(f"  Metric: {metric_name}")
    for _k, _rn in base_ranks:
        print(f"  Single representation rank {_k} for {BASE_RANK_MODEL}: "
              f"{_rn} (dim={reps[_rn].shape[1]})")
    if not base_ranks:
        print(f"  No single-representation comparison: "
              f"{base_missing.get(BASE_RANK_MODEL)}")
    print(f"  Sigmas: {sigmas}")
    print(f"  Methods: {len(methods_to_test)}")
    print(f"  Folds: {n_folds}")

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        # Conditions A and C run once per rank: base_1_* in the DNN's rank-1
        # single representation, base_2_* in its rank-2 one.
        'base_reps_by_model': base_by_model,
        'base_reps_missing': base_missing,
        'base_reps_ranked_by': BASE_RANK_MODEL,
        # Which downstream model's stored search the ranking was read from. Only
        # ever 'dnn' or nothing, since 2026-09-28: the second lookup that used to
        # put 'rf' here is gone, and with no DNN search stored the run does not
        # reach this point.
        'base_reps_search_model': 'dnn' if _dnn_search_found else None,
        'sigmas': sigmas,
        'noise_condition': noise_condition,
        'noise_provenance': noise_provenance,
        'methods': methods_to_test,
        # Record the actual budget the pool selected, not the kwargs default
        # (previously hard-coded to 100 regardless of what the pool returned).
        # NOTE this is the CLEAN-label pool's budget. Every fold re-searches on
        # the noisy labels and normally uses its own budget, which this field
        # does not report -- see 'recipe_sources' and the per-fold search for
        # what was actually built.
        'clean_pool_budget': int(bp.budget) if bp is not None else None,
        'budget': int(bp.budget) if bp is not None else None,
        'task_type': task_type,
        'is_clf': is_clf,
        'n_classes': int(len(classes_)) if is_clf else None,
        'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
        # One entry per (sigma, fold, noise draw) whose feature-mix search
        # returned: which labels chose that pass's recipe, under 'source',
        # beside the sigma, fold index and noise draw seed it belongs to. A
        # pass whose search raised has no entry at all -- it is in the
        # 'Hybrid failed' print and nowhere here.
        # 'noisy_search' is what this experiment is for; 'clean_pool_fallback'
        # means the noisy search scored no budget on that fold and the
        # clean-label recipe was used instead. Without this the two were
        # indistinguishable in the output.
        'recipe_sources': [],
        'per_sigma': {},
    }
    _recipe_sources = results['recipe_sources']

    # ── Checkpoint and the similarity axis, added 2026-09-25 ──────────────
    # Plan sections 1-3. Each (level, fold, draw) pass is written as it
    # finishes and read back on resume; see the pass below.
    #
    # distance_selection finds neighbours and drops training molecules, so it
    # is swept over every similarity setting and k and its kept rows are
    # written as D-versus-D' rows (see the pass).
    #
    # TODO(similarity-sweep 2026-09-25): the other eleven methods change how
    # the DNN trains, not the training set, so there is no D' to fit RF and
    # SVM on and they write no rows. small_loss and confident_learning do
    # drop molecules but do not return which; mentornet reads a neighbour
    # list but carves its checkpoint-selection rows out of X_train first
    # (lnl_methods.py, MentorNet.train_and_predict), so a list over the
    # training fold no longer lines up and it is not swept. Each needs a
    # decision on what its D' is.
    from experiments.noise.n07_remediation import (
        FoldCheckpoint, similarity_record, sweep_settings, sweep_ks, RowSink,
        sweep_context, similarity_sweep)
    sim_settings = sweep_settings()
    sim_sink = RowSink(output_dir, 'n09_lnl_survey', dataset_name,
                       kwargs.get('noise_condition'))
    sim_ctx = sweep_context(
        experiment='n09_lnl_survey', dataset_name=dataset_name, sink=sim_sink,
        settings=sim_settings, metric=(metric_name, metric_fn,
                                       lower_is_better),
        task_type=task_type, condition=noise_condition, level=None, data=data,
        factories=get_model_factories(['rf', 'svm', 'dnn'],
                                      task_type=task_type,
                                      n_samples=len(labels)))
    # The DNN's similarity spaces: its top-2 single representations and the
    # clean-pool recipe, as every other noise experiment reads them.
    sim_base = [rn for _k, rn in base_ranks]
    sim_recipes = {'hybrid': (bp, (meta or {}).get('aug_condition',
                                                    'uncapped'))}
    _ds_by_sim = {}
    from experiments._shared import input_fingerprint
    ckpt = FoldCheckpoint(output_dir, 'n09_lnl_survey', dataset_name,
                          condition=kwargs.get('noise_condition'),
                          tag='passes', fingerprint={
                              'input': input_fingerprint(data, config),
                              'noise_condition': noise_condition,
                              'methods': list(methods_to_test),
                              'base_ranks': [list(x) for x in base_ranks],
                              # Every pass's hybrid comes from a search over
                              # these budgets, so a pass stored under another
                              # grid is another measurement. Without this a
                              # resumed run that changed the grid (the local
                              # run cuts it to [100]) read the old passes back
                              # as its own. Added 2026-09-28.
                              'noisy_search_budgets': [
                                  int(b) for b in NOISY_SEARCH_BUDGETS]})

    for sigma in sigmas:
        print(f"\n  === Sigma={sigma} ===")
        t_sigma = time.time()

        # One base_<k>_no_lnl list per rank of the single-representation
        # comparison, beside the two hybrid ones.
        _plain_keys = ([f'{b}_no_lnl' for b in base_keys]
                       + ['hybrid_no_lnl', 'weighted_hybrid_no_lnl'])
        _lnl_keys = ([f'{b}_lnl' for b in base_keys]
                     + ['hybrid_lnl', 'weighted_hybrid_lnl'])
        sigma_results = {k: [] for k in _plain_keys}
        # Which fold each score came from. A fold where the hybrid build fails
        # drops out of the hybrid arms but not the base arm, so without this the
        # two means are averages over different folds and their difference is
        # not a paired comparison.
        sigma_folds = {k: [] for k in _plain_keys}
        # Which noise draw each score came from, aligned with sigma_folds. A
        # hybrid build fails for one (fold, draw) pass, not for a whole fold,
        # so the fold index alone does not say which passes an arm holds.
        sigma_draws = {k: [] for k in _plain_keys}
        # Detection readings for this level, one list per (condition, method).
        # Stays empty unless the condition is one where a corrupted subset exists.
        detection_acc = {}
        # 2026-09-27, owner's decision: the methods run at level 0.0 too. There
        # nothing is injected, so the plain DNN is fitted on the labels as
        # recorded (D) and each method on the same labels (D'), and the pair is
        # the answer on the dataset's own experimental noise. Detection is not
        # graded at 0: detection_scores returns None when nothing moved.
        if True:
            for _ck in _lnl_keys:
                sigma_results[_ck] = {m: [] for m in methods_to_test}
            # Same fold bookkeeping for the method conditions. A method that
            # raised on one fold, or a fold whose hybrid build failed, drops
            # out of one arm and not another; without this the 36 method arms
            # carried neither their fold indices nor their per-fold scores, so
            # no two of them could be compared fold by fold.
            for _ck in _lnl_keys:
                sigma_folds[_ck] = {m: [] for m in methods_to_test}
                sigma_draws[_ck] = {m: [] for m in methods_to_test}

        # THREE NOISE DRAWS PER FOLD, averaged. User decision 2026-09-12.
        #
        # A run corrupts the labels one particular way. Another draw shifts the
        # answer, and with a single draw there is no telling how much of a
        # number is the noise and how much is that draw being kind or harsh.
        # n07, n11 and n13 already average three; this file used to average one.
        #
        # It matters more since the level came out of the seed on 2026-09-12:
        # one draw now serves the whole level sweep, so an unlucky draw tilts
        # every level the same way instead of only one of them. Measured on
        # ridge with 400 training molecules and 10 draws per level, the spread
        # across draws at one level was 1.0x to 2.0x the step to the next level.
        #
        # Written as a (fold, draw) pass list rather than a nested loop so the
        # body below is untouched. Every score list gains 3 entries per fold
        # instead of 1, so the means at the end average over both axes.
        # At level 0 a regression draw delivers exactly zero, so three draws are
        # three identical runs — collapse to one. NOT on a classification set:
        # calibrate_flip_probability returns a floor rather than zero at a target
        # of 0.0, so the 0.0 cell there really does flip a few labels and a
        # different number of them each draw.
        _seeds = NOISE_SEEDS if (sigma > 0 or is_clf) else NOISE_SEEDS[:1]
        _splits = list(cv_split(groups, n_folds, sample_X, labels, mode=mode))
        _passes = [(_fi + 1, _tr, _va, _ns)
                   for _fi, (_tr, _va) in enumerate(_splits)
                   for _ns in _seeds]
        _outer_results = sigma_results
        for fold_idx, train_idx, val_idx, noise_seed in _passes:
            y_train_clean = labels[train_idx]
            y_val = labels[val_idx]

            # Inject noise (training only). Level 0 goes through the same path
            # and delivers exactly zero, which is the negative control.
            _spec = noise_spec_for(dataset_name, noise_condition, sigma,
                                   noise_seed=noise_seed, is_clf=is_clf)
            y_train, _cmask, _prov = inject_train_labels(
                labels, _spec, train_idx=train_idx, is_clf=is_clf,
                groups=noise_groups, dataset_name=dataset_name)
            noise_provenance.setdefault(_spec.spec_id, _prov)

            # ── One (level, fold, draw) pass, checkpointed. 2026-09-25 ──
            # n09 changes no training data (its methods change how the network
            # trains), so it writes no D-versus-D' rows; each pass is instead
            # written to a checkpoint as it finishes and read back on resume
            # (n07.FoldCheckpoint). The body below is the pass as it was; it
            # writes into pass-local copies of the accumulators, which are
            # then appended to the level's in the same order as before, so the
            # means are the same numbers whether a pass ran or was read back.
            def _one_pass(fold_idx=fold_idx, train_idx=train_idx,
                          val_idx=val_idx, noise_seed=noise_seed,
                          y_train=y_train, y_val=y_val, _cmask=_cmask):
                sigma_results = {k: ([] if not isinstance(v, dict) else
                                     {m: [] for m in v})
                                 for k, v in _outer_results.items()}
                sigma_folds = {k: ([] if not isinstance(v, dict) else
                                   {m: [] for m in v})
                               for k, v in _outer_results.items()}
                sigma_draws = {k: ([] if not isinstance(v, dict) else
                                   {m: [] for m in v})
                               for k, v in _outer_results.items()}
                detection_acc = {}
                _recipe_sources = []

                # The single-representation features, one pair per rank, each
                # scaled on this fold's training molecules.
                X_base_by_key = {}
                for (_k, _rn), _bk in zip(base_ranks, base_keys):
                    scaler_base = StandardScaler()
                    X_base_by_key[_bk] = (
                        scaler_base.fit_transform(reps[_rn][train_idx]),
                        scaler_base.transform(reps[_rn][val_idx]))

                # The scaffold groups of this fold's training rows. Set before
                # the hybrid build's try, because the plain DNN of conditions A,
                # B and E reads them for its early-stopping carve; inside the
                # try, a failure before this line would turn condition A into a
                # NameError that its own except swallows.
                train_groups_subset = (groups[train_idx]
                                       if groups is not None else None)

                # Build hybrid features by running a fresh feature-mix search on
                # the NOISY training labels (matching the c06 per-variant pattern).
                # The pool blueprint `bp` above was fit on clean labels, which
                # would give the LNL methods a feature mix chosen with oracle
                # access to ground truth. Refitting per (noise level, fold) on
                # noisy labels removes that leak. Uses a reduced budget grid to
                # keep runtime tractable. The search model is the plain DNN
                # (_BaselineDNN) since 2026-09-28, not a random forest.
                train_reps = {k: v[train_idx] for k, v in reps.items()}
                val_reps = {k: v[val_idx] for k, v in reps.items()}
                try:
                    train_augs = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
                    val_augs = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None

                    # Run search WITHIN the train fold using noisy labels.
                    # train_groups_subset (set above the try) keeps inner CV
                    # inside the train fold.
                    inner_n_folds = max(2, min(3, n_folds))
                    # ⛔ THE SEARCH MODEL IS THE DNN, 2026-09-28. Two things used
                    # to disagree. This search fitted a RandomForest and nothing
                    # else, so the allocation method, the importance method, the
                    # budget and the internal evaluator were all chosen by how
                    # well a RandomForest did on them. Every arm the recipe then
                    # feeds -- the plain DNN baseline and all twelve
                    # label-noise methods -- is a DNN. So the recipe was picked
                    # for one learner and scored with another, on every fold of
                    # every pass.
                    #
                    # build_fold_hybrid below already passes _BaselineDNN as the
                    # accept-or-reject model, for exactly this reason. That fixed
                    # the keep-or-drop step and left the recipe search behind.
                    #
                    # ⛔ THIS COSTS MORE WALL CLOCK AND THE INCREASE IS NOT
                    # MEASURED. The stored RandomForest search took 677 seconds
                    # on flab_thermo_tresanco at level 0.5, fold 1 of 3, 448
                    # antibodies trained on (measured 2026-09-21). A DNN fit is
                    # slower per candidate than a RandomForest fit, so the same
                    # search will take longer; by how much has not been timed.
                    # The stored file is read back in about 3.6 seconds after the
                    # first write, so the increase is paid once per (fold, label
                    # set).
                    #
                    # ⛔ EVERY STORED n09 SEARCH IS NOW STALE. The model list is
                    # part of the search meta, so the RandomForest files under
                    # tier 'n09_noisy_search' no longer match and will be rebuilt.
                    _search_model = _BaselineDNN(
                        random_state=42 + fold_idx, is_clf=is_clf,
                        classes_=classes_)
                    # THIS SEARCH IS STORED. It used to run with cache=False, so
                    # a second run of the same passes searched again from cold, and
                    # the search is the largest single cost in the file: on
                    # flab_thermo_tresanco at level 0.5, fold 1 of 3, 448 antibodies
                    # trained on, it took 677 seconds, against 3.6 seconds to read
                    # back what a previous call wrote. Writing costs 160 seconds the
                    # first time. Measured 2026-09-21; the stored call and the
                    # unstored call chose the same feature mix and the same inner
                    # score, 0.489755 spearman.
                    #
                    # WHAT KEEPS TWO PASSES APART. The file is named by the fold
                    # index and by a hash of the exact training labels this pass
                    # holds, so the clean pass and the corrupted pass of one fold
                    # never share a file, and neither do two folds whose training
                    # blocks are the same size. Naming it by (level, fold, draw)
                    # instead would give three separate files to the three draws of
                    # level 0.0, which move no label and are the same search.
                    #
                    # find_best_hybrid then content-checks the labels, the
                    # representation names and shapes, the groups, the budgets, the
                    # metric and the seed before it serves anything, and recomputes
                    # on any disagreement. It does NOT check representation VALUES:
                    # a neural embedding is not bit-reproducible across thread
                    # counts, so those hashes are ignored by design, which is why
                    # the fold index is in the name.
                    #
                    # The namespace also separates the per-representation baselines
                    # this search fits. Without one all six passes wrote to the same
                    # baseline filenames, whose names carry no label hash, so each
                    # pass evicted the one before it and none was ever reused.
                    # TODO(similarity-sweep 2026-09-25): this is a search, and the
                    # prelim runs set KIRBY_NO_COLD_SEARCH=1 (experiments read the
                    # precompute and never search). Tier 'n09_noisy_search' is not
                    # in the precompute, so under that setting every pass whose
                    # search is not already stored is refused, the except below
                    # catches it and the hybrid arms drop the pass. Whether n09
                    # keeps its per-pass noisy-label search is the owner's call.
                    _label_token = hashlib.sha1(
                        np.ascontiguousarray(y_train).tobytes()).hexdigest()[:12]
                    inner_search = find_best_hybrid(
                        base_reps={k: v[train_idx] for k, v in reps.items()},
                        labels=y_train,
                        groups=train_groups_subset,
                        models={'dnn': _search_model},
                        budgets=NOISY_SEARCH_BUDGETS,
                        task_type=task_type,
                        metric=metric_name,
                        # Runtime is kept tractable by the reduced
                        # NOISY_SEARCH_BUDGETS grid and the small inner n_folds.
                        seed=42 + fold_idx,
                        n_folds=inner_n_folds,
                        cache=True,
                        dataset_name=dataset_name,
                        tier='n09_noisy_search',
                        cache_namespace=f'n09search_f{fold_idx}_y{_label_token}',
                        augmentations=({k: v[train_idx] for k, v in augmentations.items()}
                                       if augmentations else None),
                        augmentation_strategy=meta.get('aug_condition', 'uncapped') if meta else 'uncapped',
                        verbose=False,
                    )
                    # Pick the best blueprint across the reduced budget grid
                    # 'dnn' since 2026-09-28, and it must match the one name in
                    # models= above. A name the search did not fit returns {}
                    # here, every budget is skipped, _best_bp stays None and the
                    # clean-label recipe is used with _recipe_source recorded as
                    # 'clean_pool_fallback' -- so a mismatch would silently turn
                    # every fold into the fallback measurement.
                    per_budget = inner_search.best_by_model_budget.get('dnn', {})
                    _best_bp = None
                    _best_score = float('inf') if lower_is_better else float('-inf')
                    for _budget, _entry in per_budget.items():
                        if _entry is None:
                            continue
                        _score = _entry.get('score_mean', float('nan'))
                        if np.isnan(_score):
                            continue
                        if (lower_is_better and _score < _best_score) or \
                           (not lower_is_better and _score > _best_score):
                            _best_score = _score
                            _best_bp = _entry['blueprint']
                    # The whole point of the inner search is that the recipe is
                    # chosen on the NOISY labels, with no clean-label oracle. When
                    # every budget in the reduced grid scored NaN there is no noisy
                    # recipe, and `bp` -- chosen on the clean labels -- is used
                    # instead. That is a reasonable fallback and a different
                    # measurement, and nothing in the output said which had
                    # happened, so a fold that fell back read as one that did not.
                    inner_bp = _best_bp if _best_bp is not None else bp
                    _recipe_source = 'noisy_search' if _best_bp is not None else 'clean_pool_fallback'
                    # The fold this happened on used to be printed and nowhere in
                    # the saved output: recipe_sources was a bare list of strings,
                    # so a reader could not say WHICH fold fell back, only how many
                    # did. Each entry now carries its noise level, its fold index
                    # and its noise draw seed. Split audit D,
                    # tests/experiments/noise/n09_lnl_survey.py:338.
                    _recipe_sources.append({
                        'sigma': float(sigma),
                        'fold': int(fold_idx),
                        'noise_seed': int(noise_seed),
                        'source': _recipe_source,
                    })
                    if _best_bp is None:
                        print(f"    fold {fold_idx}: the noisy search scored no "
                              f"budget, so this fold uses the CLEAN-label recipe "
                              f"(recorded as clean_pool_fallback)")

                    # model/metric_fn: the accept-or-reject step inside the build
                    # now fits the same plain DNN every condition here is scored
                    # with, and grades it by the dataset's own metric. Without them
                    # create_hybrid falls back to its internal evaluator scoring
                    # r2 (or accuracy), so the features were kept or dropped by a
                    # different learner than the one that goes on to use them. The
                    # number of fits is unchanged — the internal evaluator fitted
                    # once per candidate allocation too.
                    X_hybrid_train, X_hybrid_val, fi = build_fold_hybrid(
                        train_reps, val_reps, y_train, inner_bp,
                        augmentations_train=train_augs, augmentations_val=val_augs,
                        groups=train_groups_subset,
                        model=_BaselineDNN(random_state=42 + fold_idx,
                                           is_clf=is_clf, classes_=classes_),
                        metric_fn=metric_fn, lower_is_better=lower_is_better)
                    scaler_hyb = StandardScaler()
                    X_hybrid_train_s = scaler_hyb.fit_transform(X_hybrid_train)
                    X_hybrid_val_s = scaler_hyb.transform(X_hybrid_val)
                    hybrid_ok = True
                except Exception as e:
                    print(f"    Hybrid failed (fold {fold_idx}): {e}")
                    hybrid_ok = False

                # --- Condition A: single representation, no LNL, once per rank ---
                for _bk, (X_base_train_s, X_base_val_s) in X_base_by_key.items():
                    try:
                        y_pred_a = _run_baseline(X_base_train_s, y_train,
                                                 X_base_val_s, y_val,
                                                 random_state=42 + fold_idx,
                                                 is_clf=is_clf, classes_=classes_,
                                                 groups=train_groups_subset)
                        score_a = float(metric_fn(y_val, y_pred_a))
                        sigma_results[f'{_bk}_no_lnl'].append(score_a)
                        sigma_folds[f'{_bk}_no_lnl'].append(fold_idx)
                        sigma_draws[f'{_bk}_no_lnl'].append(noise_seed)
                    except Exception as e:
                        print(f"    Condition A ({_bk}) failed: {e}")

                # --- Condition B: hybrid, no LNL ---
                if hybrid_ok:
                    try:
                        y_pred_b = _run_baseline(X_hybrid_train_s, y_train,
                                                 X_hybrid_val_s, y_val,
                                                 random_state=42 + fold_idx,
                                                 is_clf=is_clf, classes_=classes_,
                                                 groups=train_groups_subset)
                        score_b = float(metric_fn(y_val, y_pred_b))
                        sigma_results['hybrid_no_lnl'].append(score_b)
                        sigma_folds['hybrid_no_lnl'].append(fold_idx)
                        sigma_draws['hybrid_no_lnl'].append(noise_seed)
                    except Exception as e:
                        print(f"    Condition B failed: {e}")

                # --- Build weighted hybrid features for conditions E/F ---
                # Use per-fold feat_info weights (not global) to avoid dimension mismatch
                weighted_ok = False
                if hybrid_ok:
                    try:
                        from experiments._shared import flatten_fold_weights
                        fold_weights = flatten_fold_weights(fi)
                        n_cols = X_hybrid_train_s.shape[1]
                        if len(fold_weights) == n_cols:
                            # Importance-weight each hybrid column with MEAN-1
                            # normalisation. w is normalised to mean 1 (not sum 1),
                            # so columns keep ~unit scale on average and the input
                            # magnitude is preserved — this avoids the original
                            # scale-collapse (sum-1 weights shrank every column by
                            # ~1/n_cols ≈0.01, crushing DNN/LNL inputs to near zero).
                            # We deliberately do NOT re-fit a StandardScaler here:
                            # X_hybrid_*_s is already unit-variance, so multiplying
                            # column j by w_j gives std w_j and a fresh per-column
                            # StandardScaler would divide it straight back out —
                            # exactly cancelling the weighting and collapsing
                            # conditions E→B and F→D. Mean-1 scaling is the fix that
                            # keeps the weighting hypothesis testable.
                            w = np.sqrt(np.maximum(fold_weights, 0))
                            w = w / (w.mean() + 1e-10)
                            X_whybrid_train_s = X_hybrid_train_s * w
                            X_whybrid_val_s = X_hybrid_val_s * w
                            weighted_ok = True
                        else:
                            print(f"    Weight dim mismatch: {len(fold_weights)} vs {n_cols}")
                    except Exception as e:
                        print(f"    Weighted hybrid build failed: {e}")

                # --- Condition E: weighted hybrid, no LNL ---
                if weighted_ok:
                    try:
                        y_pred_e = _run_baseline(X_whybrid_train_s, y_train,
                                                 X_whybrid_val_s, y_val,
                                                 random_state=42 + fold_idx,
                                                 is_clf=is_clf, classes_=classes_,
                                                 groups=train_groups_subset)
                        score_e = float(metric_fn(y_val, y_pred_e))
                        sigma_results['weighted_hybrid_no_lnl'].append(score_e)
                        sigma_folds['weighted_hybrid_no_lnl'].append(fold_idx)
                        sigma_draws['weighted_hybrid_no_lnl'].append(noise_seed)
                    except Exception as e:
                        print(f"    Condition E failed: {e}")

                # --- Conditions C, D & F: LNL methods, at every level incl. 0 ---
                if True:
                    lnl_kwargs = dict(random_state=42 + fold_idx,
                                      is_clf=is_clf, classes_=classes_)
                    for method_name in methods_to_test:
                        # Condition C: single representation + LNL, once per rank
                        for _bk, (X_base_train_s, X_base_val_s) in X_base_by_key.items():
                            _ck = f'{_bk}_lnl'
                            try:
                                method_c = get_lnl_method(method_name, **lnl_kwargs)
                                # Keeps each method's checkpoint-selection carve
                                # scaffold-disjoint inside the training fold.
                                method_c.selection_groups = train_groups_subset
                                result_c = method_c.train_and_predict(
                                    X_base_train_s, y_train, X_base_val_s, y_val)
                                score_c = float(metric_fn(y_val, result_c['y_pred']))
                                _record_detection(detection_acc, _ck, method_name,
                                                   result_c, _cmask, noise_condition)
                                sigma_results[_ck][method_name].append(score_c)
                                sigma_folds[_ck][method_name].append(fold_idx)
                                sigma_draws[_ck][method_name].append(noise_seed)
                            except Exception as e:
                                print(f"    C/{_bk}/{method_name} failed: {e}")

                        # Condition D: hybrid + LNL
                        if hybrid_ok:
                            try:
                                method_d = get_lnl_method(method_name, **lnl_kwargs)
                                method_d.selection_groups = train_groups_subset
                                result_d = method_d.train_and_predict(
                                    X_hybrid_train_s, y_train,
                                    X_hybrid_val_s, y_val)
                                score_d = float(metric_fn(y_val, result_d['y_pred']))
                                _record_detection(detection_acc, 'hybrid_lnl', method_name,
                                                   result_d, _cmask, noise_condition)
                                sigma_results['hybrid_lnl'][method_name].append(score_d)
                                sigma_folds['hybrid_lnl'][method_name].append(fold_idx)
                                sigma_draws['hybrid_lnl'][method_name].append(noise_seed)
                            except Exception as e:
                                print(f"    D/{method_name} failed: {e}")

                        # Condition F: weighted hybrid + LNL
                        if weighted_ok:
                            try:
                                method_f = get_lnl_method(method_name, **lnl_kwargs)
                                method_f.selection_groups = train_groups_subset
                                result_f = method_f.train_and_predict(
                                    X_whybrid_train_s, y_train,
                                    X_whybrid_val_s, y_val)
                                score_f = float(metric_fn(y_val, result_f['y_pred']))
                                _record_detection(detection_acc, 'weighted_hybrid_lnl', method_name,
                                                   result_f, _cmask, noise_condition)
                                sigma_results['weighted_hybrid_lnl'][method_name].append(score_f)
                                sigma_folds['weighted_hybrid_lnl'][method_name].append(fold_idx)
                                sigma_draws['weighted_hybrid_lnl'][method_name].append(noise_seed)
                            except Exception as e:
                                print(f"    F/{method_name} failed: {e}")
                # ── distance_selection under every similarity setting ──
                # 2026-09-25. The one LNL method here that finds neighbours
                # and drops training molecules (lnl_methods.
                # DistanceBasedSelection, which takes a neighbour list since
                # 2026-09-25). Run once per (setting, k) on each single
                # representation and on the hybrid; its DNN score goes to
                # results['distance_selection_by_similarity'], and the rows it
                # kept are D', fitted with RF, SVM and the DNN as rows.
                _sim_scores = []
                if ('distance_selection' in methods_to_test
                        and sim_sink.enabled):
                    _g_tr = (groups[train_idx] if groups is not None
                             else None)
                    _arms = list(X_base_by_key.items())
                    if hybrid_ok:
                        _arms.append(('hybrid', (X_hybrid_train_s,
                                                 X_hybrid_val_s)))
                    for _arm, (_Xtr, _Xva) in _arms:
                        def _ds_dprimes(setting, lists, k, _Xtr=_Xtr,
                                        _Xva=_Xva, _arm=_arm):
                            try:
                                _mth = get_lnl_method(
                                    'distance_selection', k=k,
                                    precomputed_neighbors=lists,
                                    random_state=42 + fold_idx,
                                    is_clf=is_clf, classes_=classes_)
                                _mth.selection_groups = _g_tr
                                _res = _mth.train_and_predict(_Xtr, y_train,
                                                              _Xva, y_val)
                            except Exception as e:
                                print(f"    distance_selection "
                                      f"{setting.name} k={k} failed: {e}")
                                yield 'distance_selection', 'threshold_pct=85', None
                                return
                            _sim_scores.append({
                                'arm': _arm, 'setting': setting.name,
                                'k': int(k), 'score': float(metric_fn(
                                    y_val, _res['y_pred']))})
                            # The rows the method kept, rebuilt from what it
                            # returns: flagged = suspicion above its
                            # threshold, and every row kept when fewer than
                            # 10 would remain (lnl_methods.py, the lines
                            # after the neighbour search).
                            _md = _res['metadata']
                            _keep = ~(np.asarray(_md['suspicion_scores'])
                                      > float(_md['threshold']))
                            if _keep.sum() < 10:
                                _keep = np.ones(len(y_train), dtype=bool)
                            yield ('distance_selection', 'threshold_pct=85',
                                   {'X': _Xtr[_keep],
                                    'y': np.asarray(y_train)[_keep],
                                    'keep': _keep, 'sample_weight': None,
                                    'label_column': None})
                        similarity_sweep(
                            sim_ctx, model_name='dnn',
                            model_input=f'dnn:{_arm}', X_tr=_Xtr, X_va=_Xva,
                            y_tr_noisy=y_train, y_tr_clean=y_train_clean,
                            y_val=y_val, train_idx=train_idx, val_idx=val_idx,
                            # fold_idx counts from 1 in this file (the pass
                            # list above); rows count folds from 0, as every
                            # other experiment's rows and run_one's --fold do.
                            groups_tr=_g_tr, fold=fold_idx - 1,
                            draw=noise_seed,
                            recipes=sim_recipes, base_reps=sim_base,
                            make_dprimes=_ds_dprimes, level=sigma)
                return {'results': sigma_results, 'folds': sigma_folds,
                        'draws': sigma_draws, 'detection': detection_acc,
                        'recipe_sources': _recipe_sources,
                        'sim': _sim_scores}

            _pass = ckpt.cached([float(sigma), int(fold_idx), int(noise_seed)],
                                _one_pass)
            for _ck, _v in _pass['results'].items():
                if isinstance(_v, dict):
                    for _m, _xs in _v.items():
                        sigma_results[_ck][_m].extend(_xs)
                        sigma_folds[_ck][_m].extend(_pass['folds'][_ck][_m])
                        sigma_draws[_ck][_m].extend(_pass['draws'][_ck][_m])
                else:
                    sigma_results[_ck].extend(_v)
                    sigma_folds[_ck].extend(_pass['folds'][_ck])
                    sigma_draws[_ck].extend(_pass['draws'][_ck])
            for _ck, _by_m in _pass['detection'].items():
                for _m, _rows in _by_m.items():
                    detection_acc.setdefault(_ck, {}).setdefault(
                        _m, []).extend(_rows)
            results['recipe_sources'].extend(_pass['recipe_sources'])
            for _r in _pass.get('sim') or []:
                (_ds_by_sim.setdefault(str(sigma), {}).setdefault(_r['arm'], {})
                 .setdefault(_r['setting'], {}).setdefault(str(_r['k']), [])
                 .append(_r['score']))

        # Aggregate fold results
        agg = {}
        for cond_key in _plain_keys:
            scores = sigma_results[cond_key]
            if scores:
                agg[cond_key] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'n_folds': len(scores),
                    # The folds behind that mean. Compare two conditions only
                    # over the folds both of them scored.
                    'folds': list(sigma_folds[cond_key]),
                    # The noise draw behind each of those scores, aligned with
                    # 'folds'. (fold, draw) is the unit a failure drops.
                    'fold_draws': list(sigma_draws[cond_key]),
                    # One score per fold, aligned with 'folds'. The mean and
                    # std above cannot be re-paired once written; these can.
                    'fold_scores': [float(x) for x in scores],
                }
            else:
                agg[cond_key] = {'mean': float('nan'), 'n_folds': 0,
                                 'folds': [], 'fold_draws': [],
                                 'fold_scores': []}

        if True:
            for cond_key in _lnl_keys:
                agg[cond_key] = {}
                for method_name in methods_to_test:
                    scores = sigma_results[cond_key][method_name]
                    if scores:
                        agg[cond_key][method_name] = {
                            'mean': float(np.mean(scores)),
                            'std': float(np.std(scores)),
                            'n_folds': len(scores),
                            'folds': list(sigma_folds[cond_key][method_name]),
                            'fold_draws': list(sigma_draws[cond_key][method_name]),
                            'fold_scores': [float(x) for x in scores],
                        }
                    else:
                        agg[cond_key][method_name] = {
                            'mean': float('nan'), 'n_folds': 0,
                            'folds': [], 'fold_draws': [], 'fold_scores': [],
                        }

        if detection_acc:
            # How well each method FOUND the corrupted molecules, before any repair.
            # Present only under outlier_p01/p05/p10, grouped_wider and censoring.
            agg['detection'] = {
                _ck: {_m: _mean_detection(_rows) for _m, _rows in _by_method.items()}
                for _ck, _by_method in detection_acc.items()}

        results['per_sigma'][str(sigma)] = convert_numpy(agg)

        # Print summary for this sigma
        b_score = agg['hybrid_no_lnl']['mean']
        e_score = agg['weighted_hybrid_no_lnl']['mean']
        print("    " + ", ".join(
            f"{_bk} DNN: {agg[f'{_bk}_no_lnl']['mean']:.4f}"
            for _bk in base_keys)
              + (", " if base_keys else "")
              + f"Hybrid DNN: {b_score:.4f}, "
              f"Weighted Hybrid DNN: {e_score:.4f}")
        if True:
            # Best LNL method per condition
            for cond in _lnl_keys:
                best_name = None
                best_score = float('-inf') if not lower_is_better else float('inf')
                for m in methods_to_test:
                    s = agg[cond].get(m, {}).get('mean', float('nan'))
                    if np.isnan(s):
                        continue
                    if (not lower_is_better and s > best_score) or \
                       (lower_is_better and s < best_score):
                        best_score = s
                        best_name = m
                if best_name:
                    print(f"    Best {cond}: {best_name} ({best_score:.4f})")

        print(f"    [{time.time() - t_sigma:.0f}s]")

    # =========================================================================
    # RETENTION COMPUTATION
    # =========================================================================
    # For each (sigma, condition) record TWO quantities:
    #   delta_from_clean: signed (noisy_score - clean_score); always defined.
    #   retention_pct:    ratio expressed as a percentage; only defined when
    #                     both scores have the same sign AND the denominator
    #                     is bounded away from zero. The original ratio-only
    #                     formulation produced garbage for negative R²
    #                     (e.g. clean=-0.1, noisy=-0.5 → 500%). The signed
    #                     delta is the safer headline quantity.
    clean_sigma = results['per_sigma'].get('0.0', {})
    # One clean reference per rank of the single-representation comparison.
    clean_base = {_bk: clean_sigma.get(f'{_bk}_no_lnl', {}).get('mean', float('nan'))
                  for _bk in base_keys}
    clean_hybrid = clean_sigma.get('hybrid_no_lnl', {}).get('mean', float('nan'))
    clean_whybrid = clean_sigma.get('weighted_hybrid_no_lnl', {}).get('mean', float('nan'))

    def _score_ratio_pct(noisy_val, clean_val):
        """Score ratio in percent (renamed from `retention_pct`).

        This is a PLAIN ratio, not bounded to 100 — when denoising beats the
        clean baseline it legitimately exceeds 100%. Returns NaN when
        ill-defined (opposite signs / near-zero denominator).
        """
        if np.isnan(noisy_val) or np.isnan(clean_val):
            return float('nan')
        if lower_is_better:
            # clean/noisy; both should be positive (e.g. RMSE)
            if clean_val <= 1e-10 or noisy_val <= 1e-10:
                return float('nan')
            return float(clean_val / noisy_val * 100)
        # higher-is-better; both should be positive (e.g. R^2, AUROC)
        if clean_val <= 1e-10 or noisy_val <= 1e-10:
            return float('nan')
        return float(noisy_val / clean_val * 100)

    def _delta(noisy_val, clean_val):
        if np.isnan(noisy_val) or np.isnan(clean_val):
            return float('nan')
        return float(noisy_val - clean_val)

    def _pair(noisy_val, clean_val):
        return {
            'delta_from_clean': _delta(noisy_val, clean_val),
            'score_ratio_pct': _score_ratio_pct(noisy_val, clean_val),
        }

    _lnl_keys_all = ([f'{b}_lnl' for b in base_keys]
                     + ['hybrid_lnl', 'weighted_hybrid_lnl'])
    retention = {}
    for sigma_key, sigma_data in results['per_sigma'].items():
        if sigma_key == '0.0':
            continue
        sigma_ret = {}

        noisy_hybrid = sigma_data.get('hybrid_no_lnl', {}).get('mean', float('nan'))
        noisy_whybrid = sigma_data.get('weighted_hybrid_no_lnl', {}).get('mean', float('nan'))

        for _bk in base_keys:
            noisy_base = sigma_data.get(f'{_bk}_no_lnl', {}).get('mean', float('nan'))
            sigma_ret[f'{_bk}_no_lnl'] = _pair(noisy_base, clean_base[_bk])
        sigma_ret['hybrid_no_lnl'] = _pair(noisy_hybrid, clean_hybrid)
        sigma_ret['weighted_hybrid_no_lnl'] = _pair(noisy_whybrid, clean_whybrid)

        for cond in _lnl_keys_all:
            sigma_ret[cond] = {}
            if 'weighted_hybrid' in cond:
                clean_ref = clean_whybrid
            elif 'hybrid' in cond:
                clean_ref = clean_hybrid
            else:
                # base_<k>_lnl is read against its own rank's clean reference.
                clean_ref = clean_base[cond[:-len('_lnl')]]
            for method_name in methods_to_test:
                m_score = sigma_data.get(cond, {}).get(method_name, {}).get(
                    'mean', float('nan'))
                sigma_ret[cond][method_name] = _pair(m_score, clean_ref)

        retention[sigma_key] = sigma_ret

    results['retention'] = convert_numpy(retention)
    results['retention_schema'] = (
        'Each entry has {delta_from_clean: noisy-clean, score_ratio_pct: '
        'ratio*100 when both sides are positive else NaN — a plain ratio, may '
        'exceed 100 when denoising beats the clean baseline}. Prefer delta '
        'for direction-robust comparison.')

    # =========================================================================
    # THE HEADLINE, 2026-09-27 (owner's decision): each method against the
    # plain DNN trained on THE SAME LABELS -- same level, fold, noise draw and
    # feature space. `retention` above reads each method against the DNN on
    # the CLEAN labels, which mixes the damage the noise did with what the
    # method got back. This is D (plain DNN on the labels as given) against D'
    # (the method on those labels). At level 0.0 the labels are the recorded
    # ones. Both fits already exist; nothing new is fitted.
    #
    # gain is sign-normalised: positive means the method beat the plain DNN,
    # in the dataset's own metric. Passes are paired on (fold, draw); a pass
    # only one side scored is left out and counted.
    # =========================================================================
    def _paired(a, b):
        """Pairs (plain score, method score) over the passes both scored."""
        ka = {(f, d): s for f, d, s in zip(a.get('folds', []),
                                            a.get('fold_draws', []),
                                            a.get('fold_scores', []))}
        kb = {(f, d): s for f, d, s in zip(b.get('folds', []),
                                            b.get('fold_draws', []),
                                            b.get('fold_scores', []))}
        shared = sorted(set(ka) & set(kb))
        return ([(k, ka[k], kb[k]) for k in shared],
                len(set(ka) ^ set(kb)))

    vs_plain = {}
    for sigma_key, sigma_data in results['per_sigma'].items():
        per_cond = {}
        for cond in _lnl_keys_all:
            plain = sigma_data.get(cond[:-len('_lnl')] + '_no_lnl', {})
            per_method = {}
            for method_name in methods_to_test:
                meth = sigma_data.get(cond, {}).get(method_name, {})
                pairs, n_unpaired = _paired(plain, meth)
                gains = [((p - m) if lower_is_better else (m - p))
                         for _, p, m in pairs]
                per_method[method_name] = {
                    'plain_dnn_mean': (float(np.mean([p for _, p, _ in pairs]))
                                       if pairs else float('nan')),
                    'method_mean': (float(np.mean([m for _, _, m in pairs]))
                                    if pairs else float('nan')),
                    'gain_mean': (float(np.mean(gains)) if gains
                                  else float('nan')),
                    'gain_per_pass': [float(g) for g in gains],
                    'passes': [list(k) for k, _, _ in pairs],
                    'passes_won': int(sum(g > 0 for g in gains)),
                    'n_passes': len(gains),
                    'n_passes_scored_by_one_side_only': int(n_unpaired),
                }
            per_cond[cond] = per_method
        vs_plain[sigma_key] = per_cond
    results['vs_plain_dnn_same_labels'] = convert_numpy(vs_plain)
    results['vs_plain_dnn_same_labels_schema'] = (
        'Per level, per feature space, per method: the plain DNN and the '
        'method, both trained on the same labels (recorded at level 0.0, '
        'injected above it), scored on the same held-out fold, paired on '
        '(fold, draw). gain > 0 means the method beat the plain DNN, in the '
        'dataset metric. This is the headline; `retention` compares against '
        'the clean-label DNN instead.')
    print("\n  Each method against the plain DNN on the same labels "
          "(passes won / passes, mean gain):")
    for sigma_key, per_cond in vs_plain.items():
        for cond, per_method in per_cond.items():
            cells = [f"{m} {r['passes_won']}/{r['n_passes']} "
                     f"{r['gain_mean']:+.4f}"
                     for m, r in per_method.items() if r['n_passes']]
            if cells:
                print(f"    level {sigma_key} {cond}: " + ", ".join(cells))

    # =========================================================================
    # SUMMARY
    # =========================================================================
    summary = {
        **{f'clean_{_bk}': clean_base[_bk] for _bk in base_keys},
        'clean_hybrid': clean_hybrid,
        'clean_weighted_hybrid': clean_whybrid,
    }

    # Best method per condition at highest noise
    highest_sigma = str(max(s for s in sigmas if s > 0))
    hs_data = results['per_sigma'].get(highest_sigma, {})

    for cond in _lnl_keys_all:
        best_name = None
        best_score = float('-inf') if not lower_is_better else float('inf')
        for m in methods_to_test:
            s = hs_data.get(cond, {}).get(m, {}).get('mean', float('nan'))
            if np.isnan(s):
                continue
            if (not lower_is_better and s > best_score) or \
               (lower_is_better and s < best_score):
                best_score = s
                best_name = m
        summary[f'best_{cond}'] = best_name
        summary[f'best_{cond}_score'] = float(best_score) if best_name else None

    # Methods where hybrid > base
    #
    # This loop used to read each arm's 'mean', which is that arm's average
    # over whichever (fold, noise draw) passes it scored. A pass whose hybrid
    # build fails drops out of hybrid_lnl but stays in base_lnl, so the two
    # means came from different passes and the better/worse verdict below was
    # not a paired comparison. Both sides now average only over the passes
    # present in both arms, from the per-pass scores the aggregation above
    # saves. When every pass scored in both arms, the two averages are the two
    # 'mean' fields unchanged. Split audit D,
    # tests/experiments/noise/n09_lnl_survey.py:338.
    def _shared_fold_means(a, b):
        """Each arm's mean over the passes both arms scored, or (nan, nan).

        A pass is one (fold index, noise draw seed). 'folds', 'fold_draws'
        and 'fold_scores' are aligned, one entry per pass. (nan, nan) comes
        back when either arm lacks those three lists or has them at unequal
        lengths, and when the two arms share no pass -- the same nan an arm
        that scored nothing gave before, which the caller skips.
        """
        def _passes(d):
            f, g, sc = d.get('folds'), d.get('fold_draws'), d.get('fold_scores')
            if f is None or g is None or sc is None:
                return None
            if not (len(f) == len(g) == len(sc)):
                return None
            return [((int(fi), int(gi)), float(x))
                    for fi, gi, x in zip(f, g, sc)]

        pa, pb = _passes(a), _passes(b)
        if pa is None or pb is None:
            return float('nan'), float('nan')
        shared = {k for k, _ in pa} & {k for k, _ in pb}
        ka = [x for k, x in pa if k in shared]
        kb = [x for k, x in pb if k in shared]
        if not ka or not kb:
            return float('nan'), float('nan')
        return float(np.mean(ka)), float(np.mean(kb))

    # Read once against each rank of the single-representation comparison:
    # summary['hybrid_vs_base']['base_1'] is the hybrid against the DNN's rank-1
    # single representation, ['base_2'] against its rank-2 one.
    hybrid_vs_base = {}
    for _bk in base_keys:
        hybrid_improves = []
        hybrid_harms = []
        for m in methods_to_test:
            _base_arm = hs_data.get(f'{_bk}_lnl', {}).get(m, {})
            _hyb_arm = hs_data.get('hybrid_lnl', {}).get(m, {})
            base_s, hyb_s = _shared_fold_means(_base_arm, _hyb_arm)
            if np.isnan(base_s) or np.isnan(hyb_s):
                # Say so rather than drop the method from the counts in
                # silence: both arms scored something, and they still share
                # no pass.
                if _base_arm.get('n_folds') and _hyb_arm.get('n_folds'):
                    print(f"    {m}: {_bk} and hybrid share no (fold, draw) "
                          f"pass at sigma={highest_sigma}; left out of "
                          f"hybrid_improves/hybrid_harms")
                continue
            if (not lower_is_better and hyb_s > base_s) or \
               (lower_is_better and hyb_s < base_s):
                hybrid_improves.append(m)
            elif (not lower_is_better and hyb_s < base_s) or \
                 (lower_is_better and hyb_s > base_s):
                hybrid_harms.append(m)
        hybrid_vs_base[_bk] = {
            'representation': dict(base_ranks)[int(_bk.split('_')[1])],
            'hybrid_improves': hybrid_improves,
            'hybrid_harms': hybrid_harms,
            'n_hybrid_improves': len(hybrid_improves),
            'n_hybrid_harms': len(hybrid_harms),
        }
    summary['hybrid_vs_base'] = hybrid_vs_base

    # Methods where weighted hybrid > hybrid (does weighting help?)
    whybrid_improves = []
    whybrid_harms = []
    for m in methods_to_test:
        # Same pairing as the hybrid-versus-base verdict above. The weighted
        # arm loses a pass whenever the per-fold weight vector and the hybrid
        # feature count disagree, which the hybrid arm keeps, so these two
        # 'mean' fields were averages over different passes.
        _hyb_arm = hs_data.get('hybrid_lnl', {}).get(m, {})
        _whyb_arm = hs_data.get('weighted_hybrid_lnl', {}).get(m, {})
        hyb_s, whyb_s = _shared_fold_means(_hyb_arm, _whyb_arm)
        if np.isnan(hyb_s) or np.isnan(whyb_s):
            if _hyb_arm.get('n_folds') and _whyb_arm.get('n_folds'):
                print(f"    {m}: hybrid and weighted hybrid share no "
                      f"(fold, draw) pass at sigma={highest_sigma}; left out "
                      f"of weighted_improves_over_hybrid/weighted_harms")
            continue
        if (not lower_is_better and whyb_s > hyb_s) or \
           (lower_is_better and whyb_s < hyb_s):
            whybrid_improves.append(m)
        elif (not lower_is_better and whyb_s < hyb_s) or \
             (lower_is_better and whyb_s > hyb_s):
            whybrid_harms.append(m)

    summary['weighted_improves_over_hybrid'] = whybrid_improves
    summary['weighted_harms_vs_hybrid'] = whybrid_harms
    summary['n_weighted_improves'] = len(whybrid_improves)
    summary['n_weighted_harms'] = len(whybrid_harms)

    # Did the top-3 ranking change?
    def _top3(cond_data):
        scores = {}
        for m in methods_to_test:
            s = cond_data.get(m, {}).get('mean', float('nan'))
            if not np.isnan(s):
                scores[m] = s
        sorted_m = sorted(scores, key=scores.get,
                          reverse=not lower_is_better)
        return sorted_m[:3]

    top3_base = {_bk: _top3(hs_data.get(f'{_bk}_lnl', {})) for _bk in base_keys}
    top3_hybrid = _top3(hs_data.get('hybrid_lnl', {}))
    top3_whybrid = _top3(hs_data.get('weighted_hybrid_lnl', {}))
    for _bk in base_keys:
        summary[f'top3_{_bk}'] = top3_base[_bk]
    summary['top3_hybrid'] = top3_hybrid
    summary['top3_weighted_hybrid'] = top3_whybrid
    # Guard against a spurious "ranking changed" verdict when one arm never
    # populated. If the hybrid (or weighted-hybrid) arm failed on every fold,
    # its top-3 list is empty and `[] != [populated]` would report True even
    # though no comparison actually happened. Report None (no valid comparison)
    # instead of a misleading True in that case.
    # One verdict per rank of the single-representation comparison.
    summary['ranking_changed'] = {
        _bk: (None if (not top3_base[_bk] or not top3_hybrid)
              else top3_base[_bk] != top3_hybrid)
        for _bk in base_keys}
    summary['ranking_changed_weighted'] = (
        None if (not top3_hybrid or not top3_whybrid)
        else top3_hybrid != top3_whybrid)
    # The reference for the two verdicts above. `ranking_changed` is True when
    # the ordered top 3 differs at all, from 3 to 5 passes and with no spread, so
    # it can be True by chance alone. Swapping one single representation for
    # another (rank 1 against rank 2) is a change of features that involves no
    # hybrid. If that swap reorders the top 3 as often as the hybrid does, the
    # hybrid verdict says nothing about the hybrid. Added 2026-09-28; no score
    # changes.
    summary['ranking_changed_between_single_reps'] = (
        None if len(base_keys) < 2 or not top3_base[base_keys[0]]
        or not top3_base[base_keys[1]]
        else top3_base[base_keys[0]] != top3_base[base_keys[1]])

    results['summary'] = convert_numpy(summary)
    results['similarity'] = similarity_record(
        sim_settings, sim_sink, swept_methods=['distance_selection'],
        spaces_unavailable=sim_ctx['unavailable'],
        checkpoint_file=ckpt.path, passes_read_back=ckpt.n_resumed)
    # {level: {arm: {setting: {k: mean DNN score over (fold, draw)}}}}, in the
    # dataset's metric. The legacy distance_selection rows above are
    # MODEL_INPUT-like: the method's own neighbour search in the arm's
    # features.
    results['distance_selection_by_similarity'] = {
        _lv: {_a: {_sn: {_k: {'mean': float(np.mean(_v)), 'n': len(_v)}
                         for _k, _v in _byk.items()}
                   for _sn, _byk in _bys.items()}
              for _a, _bys in _bya.items()}
        for _lv, _bya in _ds_by_sim.items()}

    print(f"\n  === SUMMARY ===")
    print("  Clean: " + "".join(f"{_bk}={clean_base[_bk]:.4f}, "
                                for _bk in base_keys)
          + f"hybrid={clean_hybrid:.4f}, "
          f"weighted_hybrid={clean_whybrid:.4f}")
    print(f"  Best at sigma={highest_sigma}: "
          + "".join(f"{_bk}_lnl={summary.get(f'best_{_bk}_lnl')}, "
                    for _bk in base_keys)
          + f"hybrid_lnl={summary.get('best_hybrid_lnl')}, "
          f"weighted_hybrid_lnl={summary.get('best_weighted_hybrid_lnl')}")
    for _bk, _v in hybrid_vs_base.items():
        print(f"  Hybrid improves over {_bk} ({_v['representation']}): "
              f"{_v['n_hybrid_improves']}/{len(methods_to_test)} methods")
    print(f"  Weighted improves over hybrid: "
          f"{len(whybrid_improves)}/{len(methods_to_test)} methods")
    print(f"  Ranking changed: {summary['ranking_changed']}")
    print(f"  Ranking changed (weighted): {summary['ranking_changed_weighted']}")
    print(f"  Ranking changed between the two single representations (reference): "
          f"{summary['ranking_changed_between_single_reps']}")
    for _bk in base_keys:
        print(f"  Top-3 {_bk}: {top3_base[_bk]}")
    print(f"  Top-3 hybrid: {top3_hybrid}")
    print(f"  Top-3 weighted hybrid: {top3_whybrid}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('N09: LNL Method Survey')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n09_lnl_survey', args.dataset)
