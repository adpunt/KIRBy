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
  - Levels come from noise_helpers.NOISY_SEARCH_LEVELS, which since 2026-09-12 is
    (0.0, 0.15, 0.3) — the bottom three of the shared grid every other noise
    experiment sweeps, so the numbers here can be read against theirs. 0.5, 0.75
    and 1.0 are not covered.
  - NOTHING IS PRECOMPUTED. The noisy-label precompute was deleted on
    2026-09-11; both recipes here are fetched through
    get_best_search_for_downstream, and the noisy one misses every cache layer
    on its first call because this file asks for score_on_noisy_labels=True.
    One full search per level per model therefore runs inside this experiment,
    and lands in the on-disk store for the next run. The rebuilds below do
    reuse what that search wrote: its per-fold importance and per-fold
    allocation scores go back into build_fold_hybrid rather than being refitted
    on every one of the 5 folds x 2 arms per cell, and only when the search's
    fold indices are exactly the folds this run asks for.
  - There is NO labels-hash assertion, here or anywhere in the repository. This
    file computes noisy_labels_hash and writes it into its own output; nothing
    compares it to anything. An earlier version of this docstring claimed the
    check existed.
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
from kirby.noise_spec import make_noisy_labels, noisy_labels_hash


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


def _search_settings(n_samples):
    """The search settings BOTH arms run under. Owner's decision 2026-09-12.

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
    """
    search, meta = get_best_search_for_downstream(
        reps, labels, groups,
        downstream_model=model, budget=None,
        augmentations=augmentations,
        dataset_name=dataset_name, task_type=task_type,
        data=data, label_noise=spec,
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


def _reusable_caches(search, n_folds, arm, model_name):
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

    Returns (importance_cache, alloc_cache), either {} when unusable.
    """
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
    model_factories = get_model_factories(
        DOWNSTREAM_MODELS_NOISE, task_type=task_type, n_samples=len(labels))
    sample_X = next(iter(reps.values()))

    specs = kwargs.get('noise_specs') or noisy_search_specs(task_type)
    search_settings = _search_settings(len(labels))
    print(f"  search settings, both arms: {search_settings.get('speed')} tier, "
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
                            if k in ('speed', 'tune', 'tune_n_iter', 'tune_knn_k')},
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

    for spec in specs:
        sid = spec.spec_id
        print(f"\n--- noise {sid} (target effective noise "
              f"{spec.target_effective_noise}) ---")

        # ONE deterministic noisy column, identical to the one the cached noisy
        # search was built on (both call make_noisy_labels on the full label vector).
        y_noisy_full = make_noisy_labels(labels, spec, is_clf=is_clf)
        results.setdefault('noisy_labels_hash', {})[sid] = noisy_labels_hash(y_noisy_full)

        results['per_noise_per_model'][sid] = {}
        results['blueprint_agreement'][sid] = {}

        for model_name, model_factory in model_factories.items():
            clean_search, clean_bp, clean_aug = _fetch_bp(
                reps, labels, groups, augmentations, config, data,
                dataset_name, task_type, model_name, spec=None,
                search_settings=search_settings)
            noisy_search, noisy_bp, noisy_aug = _fetch_bp(
                reps, labels, groups, augmentations, config, data,
                dataset_name, task_type, model_name, spec=spec,
                search_settings=search_settings)

            if clean_bp is None or noisy_bp is None:
                print(f"  {model_name}: skipped "
                      f"(clean_bp={'ok' if clean_bp else 'MISSING'}, "
                      f"noisy_bp={'ok' if noisy_bp else 'MISSING'})")
                continue

            # Each arm reads its OWN search's stored per-fold numbers: the
            # clean lineage's importance was fitted on the clean labels and the
            # noisy lineage's on the corrupted ones, so the two must not be
            # crossed.
            arm_caches = {
                'clean_bp': _reusable_caches(clean_search, n_folds,
                                             'clean_bp', model_name),
                'noisy_bp': _reusable_caches(noisy_search, n_folds,
                                             'noisy_bp', model_name),
            }

            fp_c, fp_n = _bp_fingerprint(clean_bp), _bp_fingerprint(noisy_bp)
            agree = _agreement(fp_c, fp_n)
            results['blueprint_agreement'][sid][model_name] = {
                'clean_blueprint': fp_c, 'noisy_blueprint': fp_n, **agree,
            }

            # Keyed by fold index, not appended, so a fold that failed in one
            # arm cannot silently pair with a different fold in the other.
            scores = {'clean_bp': {}, 'noisy_bp': {}}
            for fold_i, (train_idx, val_idx) in enumerate(
                    cv_split(groups, n_folds, sample_X, labels, mode=mode)):
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
                        m = model_factory()
                        m.fit(X_tr_s, y_train)
                        scores[arm][fold_i] = float(
                            metric_fn(y_val, m.predict(X_va_s)))
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

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('N15: Clean vs Noisy Hybrid Blueprints')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n15_clean_vs_noisy_hybrid',
                            args.dataset)
