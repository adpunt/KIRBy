#!/usr/bin/env python3
"""N15: Clean-searched vs noise-searched hybrid blueprints.

Paper: Noise.

THE QUESTION. Every noise experiment in this suite builds its hybrid from a
blueprint that was searched on the CLEAN labels. The base representations get no
such look — they are fixed matrices. So when a noise experiment reports "the hybrid
copes better with corrupted labels", we cannot tell whether the hybrid is genuinely
better or is simply cashing in on having been BUILT with knowledge of the right
answers. The July 2026 audit let n11 and n12 survive only by making them carry an
"ORACLE-BIASED ... best-case, not a fair baseline ... must not be reported as one"
warning; n09 fixed it properly but had to disable caching and cripple its search to
one model to afford it.

This experiment MEASURES the size of that advantage instead of assuming it.

DESIGN. For each noise level, two blueprints:
  clean_bp  — searched on the clean labels (what every experiment uses today)
  noisy_bp  — searched on the noisy TRAIN labels AND picked against the noisy
              validation labels (kirby.noise_spec + find_best_hybrid(
              label_noise=..., score_on_noisy_labels=True)), i.e. what an honest
              practitioner could actually have picked. Until 2026-09-04 this arm
              scored its inner folds on the CLEAN validation labels, so both arms
              chose their winner against ground truth and oracle_advantage
              covered the fitting head start only.

Both are then given the SAME treatment: build the hybrid, train on the NOISY training
labels, score on the CLEAN validation labels. Identical features pipeline, identical
noise, identical folds — the ONLY thing that differs is which blueprint was chosen and
what it was allowed to see when it was chosen.

  oracle_advantage = mean over folds of [score(clean_bp, fold) -
                     score(noisy_bp, fold)]                 [direction-aware]
  Paired per fold, over the folds BOTH arms scored, not a difference of the two
  arms' means: the arms drop folds independently when a build or a fit raises.
  It now covers BOTH head starts: fitting on clean labels and choosing with them.

  ~0     -> the head start is worth nothing. The oracle worry is empirically dead,
            every existing noise result stands as reported, and n11/n12 can drop
            their warnings. This is a publishable methodological null.
  large  -> the head start is real. Every noise result that used a clean-searched
            blueprint is inflated by roughly this much, and the suite must move to
            the noisy blueprint.

Also reported: BLUEPRINT AGREEMENT — do the two searches even pick the same thing
(rep set, allocation, importance, budget)? If they agree, the advantage is ~0 by
construction and we learn it cheaply.

SCOPE / CAVEATS (read first).
  - Synthetic noise only. Injected with the canonical, deterministic injector
    (kirby.noise_spec, backed by noiseInject) on TRAINING rows only; validation
    labels are never noised. Claims do NOT transfer to real label noise of unknown
    structure.
  - Levels are the WHOLE shared grid since 2026-09-18: noisy_search_specs reads
    dataset_policy.noise_levels() for this task type, so regression sweeps
    (0.0, 0.15, 0.3, 0.5, 0.75, 1.0) and classification (0.0, 0.05, 0.1, 0.2,
    0.3, 0.4). It was the bottom three, which answered the oracle question where
    the corruption is mildest and left it open where it is worst. Draws go from
    7 to 16 on a regression dataset, and every draw is a fresh recipe search on
    both arms, so this is about 2.3 times the searches.
  - NOTHING IS PRECOMPUTED, AND SINCE 2026-09-18 THAT IS TRUE OF BOTH ARMS.
    The noisy-label precompute was deleted on 2026-09-11. Until 2026-09-18 the
    CLEAN arm could still read a search held in memory while the noisy arm could
    not, and the two were built with different pruning, so one arm chose from a
    short candidate list and the other from a long one — defect 3 in
    NOISE_AUDIT_2026_09_10.md section 4. `_fetch_bp` now passes `data=None`, so
    both arms take the on-disk route and get identical settings. The full note
    is on that function, and `candidate_counts_match` in the output is the
    run-time check.
    One full search per level per model per ARM therefore runs inside this
    experiment, and lands in the on-disk store for the next run. The rebuilds below do
    reuse what that search wrote: its per-fold importance and per-fold
    allocation scores go back into build_fold_hybrid rather than being refitted
    on every one of the 5 folds x 2 arms per cell, and only when the search's
    fold indices are exactly the folds this run asks for.
  - THE LABELS HASH IS CHECKED SINCE 2026-09-18, and the check is narrower than
    the name suggests. Three things are asserted where the column is drawn: that
    make_noisy_labels is deterministic for this spec, that the column is the
    same length as the labels, and that it differs from the clean labels above
    level 0.0 and matches them at 0.0 on a regression dataset. ⛔ It does NOT
    compare this column to the one find_best_hybrid derived inside the noisy
    search — that column is built on the engine's label_noise branch and is not
    returned on the SearchResult, so there is nothing here to compare it to.
    ⛔ THAT GAP IS CLOSED, the same day. SearchResult gained train_labels_id,
    the 8-hex digest of the column the search was built on, and n15 recomputes
    its own with the engine's helpers and raises when the two differ.
    'checked_against_search_column' is True on every cell whose search carries
    the field; it stays False on a search pickled before the field existed,
    which is not the same as a failed check. Before 2026-09-18 nothing was
    checked at all and an earlier docstring claimed otherwise.
  - This is a comparison of BLUEPRINTS, not of hybrid-vs-base. Both arms are hybrids.

Usage:
    python n15_clean_vs_noisy_hybrid.py --dataset logd
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, cv_split, get_domain_metric, get_model_factories,
    get_best_search_for_downstream, build_fold_hybrid,
    extract_top3_base_reps, DOWNSTREAM_MODELS_NOISE,
)
from experiments.noise.noise_helpers import noisy_search_specs
from kirby.noise_spec import TrainingNoise, noisy_labels_hash
# The disk cache and the per-(noise spec, model) checkpoint, added 2026-09-25
# (plan section 3). n15 finds no neighbours and changes no training data --
# both arms train on the same noisy labels and differ in which recipe built
# the features -- so it takes no similarity setting and writes no
# D-versus-D' rows (plan section 6).
from experiments.noise import noise_cache as nc
from experiments.noise import _sim_axis_n15_n22 as sx

# The name the checkpoint and the result file are filed under.
EXP_FULL_NAME = 'n15_clean_vs_noisy_hybrid'


def _bp_fingerprint(bp):
    """The parts of a blueprint that define what it selected."""
    if bp is None:
        return None
    return {
        'reps': sorted(getattr(bp, 'reps', []) or []),
        'allocation_method': getattr(bp, 'allocation_method', None),
        'importance_method': getattr(bp, 'importance_method', None),
        'budget': getattr(bp, 'budget', None),
        'internal_evaluator': getattr(bp, 'internal_evaluator', None),
    }


def _agreement(fp_clean, fp_noisy):
    """How much do the two blueprints agree?"""
    if fp_clean is None or fp_noisy is None:
        return None
    a, b = set(fp_clean['reps']), set(fp_noisy['reps'])
    union = a | b
    return {
        'rep_jaccard': (len(a & b) / len(union)) if union else 1.0,
        'reps_identical': a == b,
        'reps_clean_only': sorted(a - b),
        'reps_noisy_only': sorted(b - a),
        'same_allocation': fp_clean['allocation_method'] == fp_noisy['allocation_method'],
        'same_importance': fp_clean['importance_method'] == fp_noisy['importance_method'],
        'same_budget': fp_clean['budget'] == fp_noisy['budget'],
        'identical': fp_clean == fp_noisy,
    }


def _search_settings(n_samples, models):
    """The search settings BOTH arms run under. Owner's decision 2026-09-12.

    ⛔ THE MODELS ARE NAMED, AND THIS IS THE SAME DEFECT c13 HAD.
    Fixed 2026-09-28. The two things that disagree: the loop in run() asks
    _fetch_bp for a recipe for each of the five downstream models in
    DOWNSTREAM_MODELS_NOISE, and the search these settings describe fitted
    three. speed='fast' arrives here from this function, and a caller-supplied
    speed makes _shared._apply_dataset_search_speed return before it names the
    models (_shared.py:1013-1016), so `models` stayed None and
    find_best_hybrid applied its own fast-tier cut -- rf, lightgbm and dnn.
    xgboost and svm were asked for and never fitted, so
    get_best_search_for_downstream returned (None, None) for them and the loop
    skipped both with no row written and nothing saying why. c13 was asking a
    fast search for svm's recipe the same way (c13_forward_compatibility.py,
    the note at _extra['models']).

    `models` here is the list the loop will score, so the search fits exactly
    the models that are then handed recipes. Nothing is substituted: a model
    the search still does not fit is skipped and named by the caller.

    ⛔ THIS CHANGES SERVER NUMBERS. xgboost and svm have no cells in n15 today
    and will have them after this. rf, lightgbm and dnn are unaffected in what
    is measured, but naming the models changes the search identity, so their
    searches re-run.

    TODO(similarity-sweep 2026-09-25): the run instructions for the noise
    suite (owner, 2026-09-25) are the fast tier, --pruning-mode calibrated and
    KIRBY_NO_COLD_SEARCH=1, "experiments read the precompute and never start
    a search". n15 cannot follow them without a design decision: its noisy arm
    IS a fresh search on the noisy labels, one per (noise spec, model), and
    its clean arm is searched live under these same settings so the two arms
    are like for like (see the paragraph below and _fetch_bp). What the
    precompute stored is the clean arm only, under different settings
    ('aggressive_pruning' and tune=True here, not what tests/
    precompute_hybrids.py wrote). Whether n15 keeps its live searches, moves
    both arms to 'calibrated', or reads the clean arm from the precompute is
    the owner's call. Nothing about the searches is changed here.

    Three things, and why each is here.

    speed='fast' keeps the fast-tier allocation methods, importance methods,
    internal evaluators and representation counts. n15 now runs three noise draws
    per level instead of one, and every draw is a fresh recipe search on both
    arms, so the search itself is what this experiment spends its time on.

    tune=True with tune_knn_k=True is the strongest tuning this repository
    currently offers. Tuning happens per fold, on that fold's training rows only.

    The aggressive pruning schedule drops the tail of the candidate list at 25,
    50 and 100 evaluations. It is what pays for the three draws.

    ONE CONSEQUENCE, STATED RATHER THAN BURIED. Asking for these settings asks for
    a different search from the one `tests/precompute_hybrids.py` wrote, so the
    clean arm is searched live here instead of read off the precompute. Both arms
    get identical settings, so the comparison this experiment exists to make is
    still like for like. But the clean blueprint n15 measures is no longer
    literally the blueprint the other noise experiments were handed.
    """
    from experiments._shared import resolve_pruning_kwargs
    out = {
        'speed': 'fast',
        'tune': True,
        'tune_n_iter': 20,
        'tune_knn_k': True,
        # The downstream models the search must fit, because the loop asks for
        # a recipe for each of them. See the note above.
        'models': list(models),
    }
    out.update(resolve_pruning_kwargs('aggressive_pruning', n_samples,
                                      dataset_label='n15'))
    return out


def _fetch_bp(reps, labels, groups, augmentations, config, data,
              dataset_name, task_type, model, spec=None, search_settings=None):
    """Blueprint from the pool. spec=None -> the clean lineage.

    The noisy lineage is searched with score_on_noisy_labels=True: its inner
    validation folds are scored on the noisy labels, so the winning
    configuration is picked without a clean label anywhere. Without that the two
    lineages differed only in what they TRAINED on -- both picked their winner
    against the same clean validation labels -- and oracle_advantage measured
    the fitting head start alone, not the head start from choosing.

    ⛔ `data` IS NOT PASSED ON, AND THAT IS THE WHOLE FIX FOR DEFECT 3.
    Fixed 2026-09-18. `data` carries `precomputed_searches`, a search already
    held in memory. Only the CLEAN arm could ever hit it -- a noisy request is
    refused that store on purpose, because it holds searches that saw the true
    activity values. So the two arms took two different routes:

      the clean arm   read the in-memory search, which `noise_helpers.run_all`
                      builds with `adaptive_pruning_schedule` set, so after the
                      first fold it DROPS the configurations that are losing and
                      stops evaluating them.
      the noisy arm   missed that store, fell through to the on-disk route, and
                      that route pins `adaptive_pruning=False`
                      (`_shared.py:5416`), so it evaluated every configuration
                      all the way through.

    One arm therefore chose from a short list and the other from a long one, and
    part of every oracle_advantage in this file was that rather than the
    clean-label head start it exists to measure. That is defect 3 in
    NOISE_AUDIT_2026_09_10.md section 4.

    Passing `data=None` sends BOTH arms down the on-disk route, so both get the
    same pruning and the same candidate list. Every other search setting already
    came from one shared dict (`_search_settings`), so after this the two calls
    differ only in the two things the experiment is about: which labels were
    trained on, and which labels the winner was picked against.

    The cost is that the clean arm no longer reads a search someone else already
    paid for. It was never reading the right one anyway -- `_search_settings`
    asks for the fast tier with tuning on, which is not what the precompute
    wrote -- so this makes an existing cold build honest rather than adding one.

    `candidate_counts` below is the run-time check, and it is n20's: when the
    two arms are handed the same labels they must score the same number of
    configurations. At level 0.0 they are handed the same labels.
    """
    search, meta = get_best_search_for_downstream(
        reps, labels, groups,
        downstream_model=model, budget=None,
        augmentations=augmentations,
        dataset_name=dataset_name, task_type=task_type,
        data=None, label_noise=spec,
        **({'score_on_noisy_labels': True} if spec is not None else {}),
        **(search_settings or {}),
    )
    if meta is None:
        return None, None, None
    bp = meta['best_config']['blueprint']
    aug_cond = meta.get('aug_condition', 'uncapped')
    # The pool sets aug strategy per slot; realign so build_fold_hybrid matches.
    if getattr(bp, 'augmentation_strategy', None) != aug_cond:
        try:
            bp.augmentation_strategy = aug_cond
        except Exception:
            pass
    return search, bp, aug_cond


def _reusable_caches(search, n_folds, arm, model_name, folds_match=True):
    """Per-fold scores this arm's own search already wrote to disk.

    importance_cache holds per-feature importance keyed (fold index,
    importance method); alloc_cache holds the per-representation allocation
    scores keyed (fold index, internal evaluator[, metric name]). Without them
    every rebuild below refits both from the fold's own rows — 5 folds x 2 arms
    per (level, model) cell — although the search that produced the recipe had
    already computed them on the same labels.

    Both are keyed by fold index, so a search run at a different fold count
    puts rows this run validates on inside the training rows behind (fold 0,
    method). They are used only when their fold indices are exactly the folds
    this run asks for, the guard c06 uses.

    Matching indices are not matching rows. `folds_match` is the row check:
    False when the engine's splitter puts different molecules in fold i than
    this run's cv_split does, and then both caches are discarded.

    Returns (importance_cache, alloc_cache), either {} when unusable.
    """
    if not folds_match:
        print(f"    {model_name}/{arm}: discarding the search's stored "
              f"importance and allocation scores — its folds hold different "
              f"molecules from this run's. Both are rebuilt per fold.")
        return {}, {}
    imp = getattr(search, 'importance_cache', None) or {}
    alloc = getattr(search, 'alloc_cache', None) or {}
    if not imp and not alloc:
        return {}, {}
    cache_folds = set()
    for k in list(imp) + list(alloc):
        if isinstance(k, tuple) and k and isinstance(k[0], int):
            cache_folds.add(k[0])
    if cache_folds != set(range(n_folds)):
        print(f"    {model_name}/{arm}: discarding the search's stored "
              f"importance and allocation scores — they hold folds "
              f"{sorted(cache_folds)}, this run uses {n_folds}. Both are "
              f"rebuilt per fold.")
        return {}, {}
    print(f"    {model_name}/{arm}: reusing the search's stored scores — "
          f"{len(imp)} (fold, importance method) entries and {len(alloc)} "
          f"(fold, evaluator) entries at n_folds={n_folds}")
    return imp, alloc


def run(data, config, output_dir, dataset_name, **kwargs):
    print("\n" + "=" * 70)
    print("EXPERIMENT N15: CLEAN-SEARCHED vs NOISE-SEARCHED HYBRID BLUEPRINTS")
    print("=" * 70)

    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    augmentations = data.get('augmentations')
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    # 2026-09-20: `noise_helpers.run_all` puts the caller's model list in
    # config['models'] and passes no `models` kwarg. Reading only the module
    # default meant the configured list was silently dropped and every run
    # scored the full five-model roster. Config first, kwarg as override.
    downstream_models = list(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS_NOISE)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    sample_X = next(iter(reps.values()))

    # The folds this run scores on, and whether the searches' own folds hold
    # the SAME molecules. The searches' stored per-fold scores are reused only
    # when they do: a stored "fold 0" whose training rows include this run's
    # fold-0 held-out molecules would put their labels into the features.
    # task_type is passed so the splitter is chosen the way the engine chooses
    # it (StratifiedGroupKFold for classification), not by the label dtype.
    _folds_n15 = list(cv_split(groups, n_folds, sample_X, labels,
                               task_type=task_type))
    _folds_match = False
    try:
        from kirby.find_best_hybrid import _make_cv_splitter
        _spl, _use_g = _make_cv_splitter(groups, n_folds, 42,
                                         task_type=task_type)
        _eng = list(_spl.split(sample_X, labels, groups) if _use_g
                    else _spl.split(sample_X, labels))
        _folds_match = (len(_eng) == len(_folds_n15) and all(
            np.array_equal(np.sort(a[0]), np.sort(b[0]))
            and np.array_equal(np.sort(a[1]), np.sort(b[1]))
            for a, b in zip(_eng, _folds_n15)))
    except Exception as exc:
        print(f"  could not rebuild the search's folds to compare rows "
              f"({type(exc).__name__}: {exc}); stored per-fold scores unused")
    print(f"  search folds hold the same molecules as this run's folds: "
          f"{_folds_match}")

    specs = kwargs.get('noise_specs') or noisy_search_specs(task_type)
    # The models the search fits are the models this run will hand recipes to.
    search_settings = _search_settings(len(labels), list(model_factories))
    print(f"  search settings, both arms: {search_settings.get('speed')} tier, "
          f"models {search_settings.get('models')}, "
          f"tune={search_settings.get('tune')}, "
          f"tune_knn_k={search_settings.get('tune_knn_k')}, "
          f"adaptive pruning schedule "
          f"{'on' if search_settings.get('adaptive_pruning_schedule') else 'off'}")

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_regime': 'synthetic',
        'noise_injection': 'canonical_noise_spec_train_labels_only',
        'claim_scope': (
            'Compares two hybrid BLUEPRINTS (clean-searched vs noise-searched) under '
            'synthetically injected training-label noise. It does NOT establish a '
            'hybrid-vs-base result, and does not transfer to real label noise of '
            'unknown structure.'),
        'noise_levels': sorted({s.target_effective_noise for s in specs}),
        'noise_specs': [s.spec_id for s in specs],
        # One entry per level: the noise draws that level was run at. Three on
        # every level with noise in it, owner's decision 2026-09-12.
        'noise_draws_per_level': {
            str(lvl): sorted(sp.noise_seed for sp in specs
                             if sp.target_effective_noise == lvl)
            for lvl in sorted({s.target_effective_noise for s in specs})},
        'search_settings': {k: v for k, v in search_settings.items()
                            if k in ('speed', 'tune', 'tune_n_iter',
                                     'tune_knn_k', 'models')},
        # The downstream models this run asks for a recipe for, and, per spec
        # id, any of them one of the two searches returned no recipe for. Such
        # a model has no cell rather than a cell built from another model's
        # recipe.
        'models_requested': list(model_factories),
        # Since 2026-09-28: both arms' per-fold feature build reads the noisy
        # search's stored scores, fitted on the noisy training labels. The
        # clean search contributes its recipe only.
        'feature_build_labels': 'noisy_training_labels_both_arms',
        'search_folds_match_rows': bool(_folds_match),
        'models_unscored': {},
        'adaptive_pruning_schedule': search_settings.get(
            'adaptive_pruning_schedule'),
        # The level each spec id belongs to, so the per-level roll-up below can
        # be rebuilt from the file without re-parsing an id string.
        'level_of_spec': {s.spec_id: float(s.target_effective_noise) for s in specs},
        'models': [],
        'per_noise_per_model': {},
        'blueprint_agreement': {},
        'summary': {},
    }

    # ── The checkpoint ──
    # Added 2026-09-25. Before it, n15 held every (noise spec, model) cell in
    # memory and wrote once at the end, and each cell is two recipe searches.
    # The results are now pickled after every cell, so a killed run resumes at
    # the next one. The searches themselves are already on disk.
    ck = sx.FoldCheckpoint(
        output_dir, EXP_FULL_NAME, dataset_name,
        sx.run_signature(data, config, {
            'specs': [s_.spec_id for s_ in specs],
            'models': downstream_models, 'settings': search_settings}))
    if ck.state is not None:
        results = ck.state

    for spec in specs:
        sid = spec.spec_id
        print(f"\n--- noise {sid} (target effective noise "
              f"{spec.target_effective_noise}) ---")

        # The noise source the cached noisy search was built on: find_best_hybrid
        # wraps the same (labels, spec) in a TrainingNoise, which corrupts each
        # fold's training labels from those labels alone.
        y_noisy_full = TrainingNoise(labels, spec, is_clf=is_clf)
        _tr0 = _folds_n15[0][0]
        _noisy_hash = noisy_labels_hash(y_noisy_full[_tr0])
        results.setdefault('noisy_labels_hash', {})[sid] = _noisy_hash

        # ⛔ THE HASH IS CHECKED NOW, NOT JUST WRITTEN. Added 2026-09-18,
        # owner's decision.
        #
        # noisy_labels_hash's own docstring (kirby/noise_spec.py:474) says
        # "Experiments assert this against the value the cached search was built
        # under". No experiment did. This file computed the hash, wrote it into
        # its own output and compared it to nothing, and an earlier version of
        # the docstring above claimed a check existed.
        #
        # WHAT THESE THREE ASSERTIONS PROVE, stated exactly, because a check
        # that is believed to cover more than it does is worse than none.
        #
        # 1. The draw is deterministic: make_noisy_labels on the same labels and
        #    the same spec gives a byte-identical column. If it does not, every
        #    cell in this run was trained on a column the search never saw, and
        #    the two arms are not comparable.
        # 2. The column has the right length, so y_noisy_full[train_idx] below
        #    indexes the rows it claims to.
        # 3. The corruption is where the level says: identical to the clean
        #    labels at level 0.0 on a regression dataset, and different from
        #    them above it. A spec that silently delivered no corruption would
        #    make oracle_advantage exactly zero and look like the publishable
        #    null this experiment is trying to test for.
        #
        # ⛔ WHAT THEY DO NOT PROVE. They do not check this column against the
        # one find_best_hybrid derived inside the noisy search. That column is
        # built on the engine's own label_noise branch and is not returned on
        # the SearchResult, so there is nothing here to compare it to. Asserting
        # that needs the engine to expose its derived column or its label digest.
        # Until it does, that gap is open and this comment is where it is
        # recorded.
        _recheck = TrainingNoise(labels, spec, is_clf=is_clf)[_tr0]
        if noisy_labels_hash(_recheck) != _noisy_hash:
            raise RuntimeError(
                f"n15 {sid}: make_noisy_labels is not deterministic for this "
                f"spec — two calls on the same labels gave different columns "
                f"({_noisy_hash} then {noisy_labels_hash(_recheck)}). Every "
                f"recipe searched under this spec was chosen on a column this "
                f"run will not train on, so neither arm's score means what it "
                f"says.")
        if len(y_noisy_full) != len(labels):
            raise RuntimeError(
                f"n15 {sid}: the noisy column has {len(y_noisy_full)} entries "
                f"and the labels have {len(labels)}. They index the same "
                f"molecules, so every row after the first gap would train on "
                f"another molecule's label.")
        _identical = bool(np.array_equal(np.asarray(y_noisy_full[_tr0]),
                                         np.asarray(labels)[_tr0]))
        _lvl = float(spec.target_effective_noise or 0.0)
        if _lvl > 0.0 and _identical:
            raise RuntimeError(
                f"n15 {sid}: the spec asks for effective noise {_lvl} and the "
                f"column it produced is identical to the clean labels. Both "
                f"arms would then be searched and trained on the same labels, "
                f"oracle_advantage would be zero by construction, and that "
                f"zero would read as the methodological null this experiment "
                f"exists to test for.")
        if _lvl == 0.0 and not is_clf and not _identical:
            raise RuntimeError(
                f"n15 {sid}: level 0.0 on a regression dataset must deliver "
                f"the clean labels unchanged and this column differs from "
                f"them. The zero-noise cell is the reference every other level "
                f"is read against.")
        _checked_before = results.get('noisy_labels_checked', {}).get(sid)
        results.setdefault('noisy_labels_checked', {})[sid] = {
            'hash': _noisy_hash,
            'deterministic': True,
            'n_labels': int(len(y_noisy_full)),
            'identical_to_clean': _identical,
            'target_effective_noise': _lvl,
            'checked_against_search_column': False,
        }
        if _checked_before:
            # A resumed run: keep what the cells already run recorded here.
            for _k in ('checked_against_search_column',
                       'engine_train_labels_id'):
                if _k in _checked_before:
                    results['noisy_labels_checked'][sid][_k] = \
                        _checked_before[_k]

        # setdefault, not a fresh dict, since 2026-09-25: a resumed run
        # restores the cells already scored and must not wipe them here.
        results['per_noise_per_model'].setdefault(sid, {})
        results['blueprint_agreement'].setdefault(sid, {})

        for model_name, model_factory in model_factories.items():
            if ck.is_done(f'{sid}|{model_name}'):
                print(f"  {model_name}: already in the checkpoint, skipped")
                continue
            clean_search, clean_bp, clean_aug = _fetch_bp(
                reps, labels, groups, augmentations, config, data,
                dataset_name, task_type, model_name, spec=None,
                search_settings=search_settings)
            noisy_search, noisy_bp, noisy_aug = _fetch_bp(
                reps, labels, groups, augmentations, config, data,
                dataset_name, task_type, model_name, spec=spec,
                search_settings=search_settings)

            # ⛔ THE CHECK THE WHOLE FILE RESTED ON AND COULD NOT MAKE.
            # Added 2026-09-18, once SearchResult gained train_labels_id.
            #
            # n15 draws y_noisy_full above and trains on it. The noisy search
            # derives its OWN column from the same spec, inside the engine. If
            # those two are not the same column, the recipe this arm reports was
            # chosen on labels this run never trains on, and oracle_advantage —
            # the number n11 and n12's warnings depend on — is the gap between
            # two things that were never comparable.
            #
            # Nothing compared them until now. n15 wrote noisy_labels_hash into
            # its own output and compared it to nothing, and the engine's column
            # was not returned, so there was nothing to compare it TO.
            # SearchResult.train_labels_id (find_best_hybrid.py) is that column's
            # 8-hex digest, the same one the tier rides on.
            #
            # It is recomputed here with the engine's own two helpers rather
            # than with noisy_labels_hash, which is a different function over a
            # different byte layout and would never match.
            _engine_id = getattr(noisy_search, 'train_labels_id', None)
            if _engine_id is not None:
                from kirby.find_best_hybrid import (_hash_bytes as _hb,
                                                    _canonical_bytes as _cb)
                _ours = f'{_hb(_cb(y_noisy_full)):016x}'[:8]
                if _ours != _engine_id:
                    raise RuntimeError(
                        f"n15 {sid}/{model_name}: the noisy column this run "
                        f"trains on ({_ours}) is not the column the noisy "
                        f"search was built on ({_engine_id}). The recipe this "
                        f"arm reports was chosen on labels that are not these, "
                        f"so oracle_advantage would be the gap between two arms "
                        f"that never saw the same data.")
                results.setdefault('noisy_labels_checked', {}).setdefault(
                    sid, {})['checked_against_search_column'] = True
                results['noisy_labels_checked'][sid]['engine_train_labels_id'] = \
                    _engine_id

            if clean_bp is None or noisy_bp is None:
                # One of the two searches fitted this model and the other did
                # not, or neither did. No other model's recipe is put in its
                # place, so the cell is absent and says why.
                _why = (f"no recipe of its own: clean arm "
                        f"{'ok' if clean_bp else 'MISSING'}, noisy arm "
                        f"{'ok' if noisy_bp else 'MISSING'}")
                results['models_unscored'].setdefault(sid, {})[model_name] = _why
                print(f"  {model_name}: skipped -- {_why}")
                continue

            # ⛔ BOTH ARMS READ THE NOISY SEARCH'S STORED PER-FOLD NUMBERS.
            # Fixed 2026-09-28 (dirty local). Until then each arm read its own
            # search's, so the clean arm's feature ranking, budget split and
            # choice of representations were fitted on the CLEAN training
            # labels. The docstring says both arms get the same features
            # pipeline on the noisy training labels, and the comment at the
            # build_fold_hybrid call says a clean-label feature build is "a
            # second, hidden look at truth". No other noise experiment gets
            # that look: none passes precomputed_importance, so each rebuilds
            # from the fold's noisy labels. oracle_advantage therefore carried
            # a head start the experiments it audits do not have.
            #
            # The noisy search's numbers are the ones build_fold_hybrid would
            # compute itself from y_train: same folds, same internal carve,
            # same noisy column (TrainingNoise is deterministic per training
            # rows, checked above). So this is the stated design, not a new one.
            # The clean search now contributes its RECIPE and nothing else.
            _noisy_caches = _reusable_caches(noisy_search, n_folds,
                                             'both arms (noisy search)',
                                             model_name,
                                             folds_match=_folds_match)
            arm_caches = {'clean_bp': _noisy_caches,
                          'noisy_bp': _noisy_caches}

            fp_c, fp_n = _bp_fingerprint(clean_bp), _bp_fingerprint(noisy_bp)
            agree = _agreement(fp_c, fp_n)
            # HOW MANY CONFIGURATIONS EACH ARM ACTUALLY SCORED. The run-time
            # check on defect 3, and n20's: the two arms must choose from lists
            # of the SAME LENGTH, or part of the advantage below is that rather
            # than the clean-label head start. At level 0.0 the injector delivers
            # exactly zero, so both arms are handed the same labels and the two
            # counts must match; above 0.0 they may legitimately differ, because
            # pruning drops whatever is losing on the labels it was given.
            n_clean = len(getattr(clean_search, 'all_results', None) or [])
            n_noisy = len(getattr(noisy_search, 'all_results', None) or [])
            same_labels = float(spec.target_effective_noise) == 0.0 and not is_clf
            results['blueprint_agreement'][sid][model_name] = {
                'clean_blueprint': fp_c, 'noisy_blueprint': fp_n, **agree,
                'n_candidates_clean_arm': n_clean,
                'n_candidates_noisy_arm': n_noisy,
                'both_arms_saw_the_same_labels': same_labels,
                'candidate_counts_match': n_clean == n_noisy,
            }
            if same_labels and n_clean != n_noisy:
                print(f"    WARNING: {model_name} at level 0.0 — the two arms "
                      f"were handed the same labels and scored "
                      f"{n_clean} against {n_noisy} configurations. That is "
                      f"defect 3 back, and every advantage in this run carries "
                      f"it.")

            # Keyed by fold index, not appended, so a fold that failed in one
            # arm cannot silently pair with a different fold in the other.
            scores = {'clean_bp': {}, 'noisy_bp': {}}
            for fold_i, (train_idx, val_idx) in enumerate(_folds_n15):
                # Noise on TRAIN rows only; validation labels stay clean.
                y_train = y_noisy_full[train_idx]
                y_val = labels[val_idx]

                train_reps = {k: v[train_idx] for k, v in reps.items()}
                val_reps = {k: v[val_idx] for k, v in reps.items()}

                for arm, bp, aug_cond in (('clean_bp', clean_bp, clean_aug),
                                          ('noisy_bp', noisy_bp, noisy_aug)):
                    augs = augmentations if aug_cond != 'none' else None
                    tr_augs = ({k: v[train_idx] for k, v in augs.items()}
                               if augs else None)
                    va_augs = ({k: v[val_idx] for k, v in augs.items()}
                               if augs else None)
                    # What this arm's search already measured for this fold.
                    # Both are ignored when None, so an unusable cache just
                    # costs the rebuild it always cost. The metric-keyed
                    # allocation entry is preferred over the default one, the
                    # order the engine itself reads them in.
                    imp_cache, alloc_cache = arm_caches[arm]
                    pre_imp = imp_cache.get(
                        (fold_i, getattr(bp, 'importance_method', None)))
                    _ev = getattr(bp, 'internal_evaluator', None)
                    pre_alloc = alloc_cache.get((fold_i, _ev, metric_name))
                    if pre_alloc is None:
                        pre_alloc = alloc_cache.get((fold_i, _ev))
                    try:
                        # Feature construction ALSO sees only the noisy train labels
                        # — otherwise the arm gets a second, hidden look at truth.
                        X_tr, X_va, _ = build_fold_hybrid(
                            train_reps, val_reps, y_train, bp,
                            augmentations_train=tr_augs,
                            augmentations_val=va_augs,
                            groups=(groups[train_idx] if groups is not None else None),
                            model=model_factory(),
                            metric_fn=metric_fn,
                            lower_is_better=lower_is_better,
                            precomputed_importance=pre_imp,
                            precomputed_alloc_scores=pre_alloc)
                        sc = StandardScaler()
                        X_tr_s = sc.fit_transform(X_tr)
                        X_va_s = sc.transform(X_va)
                        if model_name in sx.CACHEABLE_MODELS:
                            # Through noise_cache since 2026-09-25 (plan
                            # section 3): the same fit and the same
                            # model.predict, stored under the content of the
                            # features and labels. Seeded models only.
                            # predict_mode follows the metric (dirty-local
                            # 2026-09-28): 'predict' here scored auc on hard
                            # class labels on a classification dataset.
                            _pred = nc.cached_fit_predict(
                                dataset_name, model_name, model_factory,
                                X_tr_s, y_train, X_va_s,
                                predict_mode=nc.predict_mode_for(metric_name))
                        else:
                            from experiments.noise.n07_remediation import (
                                _predict_for_metric)
                            m = model_factory()
                            m.fit(X_tr_s, y_train)
                            _pred = _predict_for_metric(m, X_va_s, metric_name)
                        scores[arm][fold_i] = float(metric_fn(y_val, _pred))
                    except Exception as exc:
                        print(f"    WARNING: {model_name}/{arm} fold {fold_i} "
                              f"failed: {type(exc).__name__}: {exc}")

            # oracle_advantage is the MEAN OF THE PER-FOLD DIFFERENCES over the
            # folds both arms scored, not the difference of the two arms'
            # means. The two arms drop folds independently — a build or a fit
            # that raises leaves that fold out of one arm only — so a
            # difference of means can subtract a number covering folds 0-4 from
            # one covering folds 0, 1 and 3, with nothing in the results file
            # saying which folds went. Pairing on the fold index removes that,
            # and the dropped folds are written out beside the number.
            paired = sorted(set(scores['clean_bp']) & set(scores['noisy_bp']))
            dropped = sorted(set(scores['clean_bp']) ^ set(scores['noisy_bp']))
            if not paired:
                print(f"  {model_name}: skipped (no fold scored in both arms; "
                      f"clean_bp folds={sorted(scores['clean_bp'])}, "
                      f"noisy_bp folds={sorted(scores['noisy_bp'])})")
                continue

            c_fold = [scores['clean_bp'][f] for f in paired]
            n_fold = [scores['noisy_bp'][f] for f in paired]
            # Direction-aware: positive = the clean-searched blueprint did BETTER,
            # i.e. the head start was worth something.
            fold_deltas = [((n - c) if lower_is_better else (c - n))
                           for c, n in zip(c_fold, n_fold)]
            adv = float(np.mean(fold_deltas))
            c_mean = float(np.mean(c_fold))
            n_mean = float(np.mean(n_fold))

            results['per_noise_per_model'][sid][model_name] = {
                # Every mean and spread below covers the paired folds only.
                'clean_bp_score_mean': c_mean,
                'clean_bp_score_std': float(np.std(c_fold)),
                'noisy_bp_score_mean': n_mean,
                'noisy_bp_score_std': float(np.std(n_fold)),
                'clean_bp_fold_scores': c_fold,
                'noisy_bp_fold_scores': n_fold,
                'paired_fold_indices': paired,
                'oracle_advantage': adv,
                'oracle_advantage_fold_deltas': fold_deltas,
                'n_paired_folds': len(paired),
                'n_folds_dropped': len(dropped),
                'dropped_fold_indices': dropped,
                'blueprints_identical': agree['identical'],
            }
            if model_name not in results['models']:
                results['models'].append(model_name)

            print(f"  {model_name}: clean_bp={c_mean:.4f} noisy_bp={n_mean:.4f} "
                  f"oracle_advantage={adv:+.4f} "
                  f"over {len(paired)} paired folds"
                  f"{f', {len(dropped)} dropped' if dropped else ''} "
                  f"{'(identical blueprints)' if agree['identical'] else ''}")
            ck.save(f'{sid}|{model_name}', results)

    # ── One row per (level, model), averaging that level's noise draws ──
    # The cells above are keyed by noise SPEC, and there are now three specs per
    # level. A reader who compares levels wants one number per level, not three,
    # and the spread across the three draws is the thing that says whether a level
    # difference is real. Both are written.
    _level_of = results['level_of_spec']
    per_level = {}
    for sid, per_model in results['per_noise_per_model'].items():
        lvl = str(_level_of.get(sid))
        for mname, cell in per_model.items():
            per_level.setdefault(lvl, {}).setdefault(mname, []).append(cell)
    results['per_level_per_model'] = {}
    for lvl, per_model in per_level.items():
        results['per_level_per_model'][lvl] = {}
        for mname, cells in per_model.items():
            advs_here = [c['oracle_advantage'] for c in cells
                         if np.isfinite(c['oracle_advantage'])]
            results['per_level_per_model'][lvl][mname] = {
                'oracle_advantage_mean_over_draws':
                    float(np.mean(advs_here)) if advs_here else float('nan'),
                # The spread across draws at ONE level. Compare it with the step
                # between levels: a level difference smaller than this spread is
                # not a level difference.
                'oracle_advantage_std_over_draws':
                    float(np.std(advs_here)) if advs_here else float('nan'),
                'oracle_advantage_per_draw': advs_here,
                'n_draws_scored': len(advs_here),
                'n_draws_run': len(cells),
                'clean_bp_score_mean_over_draws': float(np.mean(
                    [c['clean_bp_score_mean'] for c in cells])),
                'noisy_bp_score_mean_over_draws': float(np.mean(
                    [c['noisy_bp_score_mean'] for c in cells])),
                'frac_draws_blueprints_identical': float(np.mean(
                    [bool(c['blueprints_identical']) for c in cells])),
            }

    # ── Summary: is the head start worth anything? ──
    # A cell whose advantage is not a finite number is DROPPED from every
    # headline figure and named in the summary. An undefined advantage used to
    # reach all of them: one NaN makes the mean and the median NaN, and a NaN
    # compares False against zero, so frac_cells_clean_bp_better counted a dead
    # cell on the noisy recipe's side instead of leaving it out.
    adv_cells = [(sid, mname, cell['oracle_advantage'])
                 for sid, per_model in results['per_noise_per_model'].items()
                 for mname, cell in per_model.items()]
    advs = [a for _, _, a in adv_cells if np.isfinite(a)]
    dropped_cells = [f"{sid}/{mname}" for sid, mname, a in adv_cells
                     if not np.isfinite(a)]
    ident = [cell['blueprints_identical']
             for per_model in results['per_noise_per_model'].values()
             for cell in per_model.values()]
    if advs:
        results['summary'] = {
            # n_cells counts the cells behind the advantage figures; the run
            # produced n_cells_run of them in total.
            'n_cells': len(advs),
            'n_cells_run': len(adv_cells),
            'n_cells_dropped_nan': len(dropped_cells),
            'cells_dropped_nan': dropped_cells,
            'mean_oracle_advantage': float(np.mean(advs)),
            'median_oracle_advantage': float(np.median(advs)),
            'max_oracle_advantage': float(np.max(advs)),
            'min_oracle_advantage': float(np.min(advs)),
            'frac_cells_clean_bp_better': float(np.mean([a > 0 for a in advs])),
            # Blueprint agreement is defined for every cell that ran, including
            # one whose advantage came out undefined, so it keeps all of them.
            'frac_blueprints_identical': float(np.mean(ident)),
            # Defect 3's run-time check, rolled up. Any cell listed here had the
            # two arms choosing from lists of different lengths while being
            # handed the same labels, and nothing in this file can be read until
            # it is empty.
            'cells_where_the_two_arms_saw_one_label_column': [
                f'{sid}/{m}' for sid, per_model in
                results['blueprint_agreement'].items()
                for m, c in per_model.items()
                if c.get('both_arms_saw_the_same_labels')],
            'cells_where_the_candidate_counts_disagreed': [
                f'{sid}/{m}' for sid, per_model in
                results['blueprint_agreement'].items()
                for m, c in per_model.items()
                if c.get('both_arms_saw_the_same_labels')
                and not c.get('candidate_counts_match')],
        }
        s = results['summary']
        print("\n" + "=" * 70)
        print(f"SUMMARY over {s['n_cells']} (noise x model) cells "
              f"of the {s['n_cells_run']} run")
        if s['n_cells_dropped_nan']:
            print(f"  dropped, advantage undefined: "
                  f"{', '.join(s['cells_dropped_nan'])}")
        print(f"  mean oracle advantage : {s['mean_oracle_advantage']:+.4f} "
              f"({metric_name})")
        print(f"  clean blueprint won in: {s['frac_cells_clean_bp_better']:.0%} of cells")
        print(f"  blueprints identical  : {s['frac_blueprints_identical']:.0%} of cells")
        print("  A mean advantage near zero means the clean-label head start buys "
              "nothing\n  and the existing noise results stand as reported.")
        print("=" * 70)

    # n15's result file is written by its caller (run_all or main), not here.
    # The snapshot is cleared on return: a kill between this return and that
    # write costs a re-run of the cells, never a stale resume.
    ck.clear()
    return results


if __name__ == '__main__':
    parser = create_experiment_parser('N15: Clean vs Noisy Hybrid Blueprints')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n15_clean_vs_noisy_hybrid',
                            args.dataset)
