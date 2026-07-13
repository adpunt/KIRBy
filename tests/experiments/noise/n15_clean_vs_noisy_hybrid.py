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
  noisy_bp  — searched on the noisy TRAIN labels, scored on clean validation
              labels (kirby.noise_spec + find_best_hybrid(label_noise=...)), i.e.
              what an honest practitioner could actually have picked

Both are then given the SAME treatment: build the hybrid, train on the NOISY training
labels, score on the CLEAN validation labels. Identical features pipeline, identical
noise, identical folds — the ONLY thing that differs is which blueprint was chosen and
what it was allowed to see when it was chosen.

  oracle_advantage = score(clean_bp) - score(noisy_bp)      [direction-aware]

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
  - The noise spec must be one the precompute actually built (noise_helpers.
    NOISY_SEARCH_LEVELS). The labels hash is asserted against the cached search, so a
    blueprint fitted to a different noise draw fails loudly rather than silently.
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


def _fetch_bp(reps, labels, groups, augmentations, config, data,
              dataset_name, task_type, model, spec=None):
    """Blueprint from the pool. spec=None -> the clean lineage."""
    search, meta = get_best_search_for_downstream(
        reps, labels, groups,
        downstream_model=model, budget=None,
        augmentations=augmentations,
        speed=config.get('search_speed', 'full'),
        dataset_name=dataset_name, task_type=task_type,
        data=data, label_noise=spec,
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
        'noise_levels': [s.target_effective_noise for s in specs],
        'noise_specs': [s.spec_id for s in specs],
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
            _, clean_bp, clean_aug = _fetch_bp(
                reps, labels, groups, augmentations, config, data,
                dataset_name, task_type, model_name, spec=None)
            _, noisy_bp, noisy_aug = _fetch_bp(
                reps, labels, groups, augmentations, config, data,
                dataset_name, task_type, model_name, spec=spec)

            if clean_bp is None or noisy_bp is None:
                print(f"  {model_name}: skipped "
                      f"(clean_bp={'ok' if clean_bp else 'MISSING'}, "
                      f"noisy_bp={'ok' if noisy_bp else 'MISSING'})")
                continue

            fp_c, fp_n = _bp_fingerprint(clean_bp), _bp_fingerprint(noisy_bp)
            agree = _agreement(fp_c, fp_n)
            results['blueprint_agreement'][sid][model_name] = {
                'clean_blueprint': fp_c, 'noisy_blueprint': fp_n, **agree,
            }

            scores = {'clean_bp': [], 'noisy_bp': []}
            fold_i = 0
            for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels,
                                               mode=mode):
                fold_i += 1
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
                    try:
                        # Feature construction ALSO sees only the noisy train labels
                        # — otherwise the arm gets a second, hidden look at truth.
                        X_tr, X_va, _ = build_fold_hybrid(
                            train_reps, val_reps, y_train, bp,
                            augmentations_train=tr_augs,
                            augmentations_val=va_augs)
                        sc = StandardScaler()
                        X_tr_s = sc.fit_transform(X_tr)
                        X_va_s = sc.transform(X_va)
                        m = model_factory()
                        m.fit(X_tr_s, y_train)
                        scores[arm].append(float(metric_fn(y_val, m.predict(X_va_s))))
                    except Exception as exc:
                        print(f"    WARNING: {model_name}/{arm} fold {fold_i} "
                              f"failed: {type(exc).__name__}: {exc}")

            if not scores['clean_bp'] or not scores['noisy_bp']:
                continue

            c_mean = float(np.mean(scores['clean_bp']))
            n_mean = float(np.mean(scores['noisy_bp']))
            # Direction-aware: positive = the clean-searched blueprint did BETTER,
            # i.e. the head start was worth something.
            adv = (n_mean - c_mean) if lower_is_better else (c_mean - n_mean)

            results['per_noise_per_model'][sid][model_name] = {
                'clean_bp_score_mean': c_mean,
                'clean_bp_score_std': float(np.std(scores['clean_bp'])),
                'noisy_bp_score_mean': n_mean,
                'noisy_bp_score_std': float(np.std(scores['noisy_bp'])),
                'clean_bp_fold_scores': scores['clean_bp'],
                'noisy_bp_fold_scores': scores['noisy_bp'],
                'oracle_advantage': adv,
                'blueprints_identical': agree['identical'],
            }
            if model_name not in results['models']:
                results['models'].append(model_name)

            print(f"  {model_name}: clean_bp={c_mean:.4f} noisy_bp={n_mean:.4f} "
                  f"oracle_advantage={adv:+.4f} "
                  f"{'(identical blueprints)' if agree['identical'] else ''}")

    # ── Summary: is the head start worth anything? ──
    advs = [cell['oracle_advantage']
            for per_model in results['per_noise_per_model'].values()
            for cell in per_model.values()]
    ident = [cell['blueprints_identical']
             for per_model in results['per_noise_per_model'].values()
             for cell in per_model.values()]
    if advs:
        results['summary'] = {
            'n_cells': len(advs),
            'mean_oracle_advantage': float(np.mean(advs)),
            'median_oracle_advantage': float(np.median(advs)),
            'max_oracle_advantage': float(np.max(advs)),
            'min_oracle_advantage': float(np.min(advs)),
            'frac_cells_clean_bp_better': float(np.mean([a > 0 for a in advs])),
            'frac_blueprints_identical': float(np.mean(ident)),
        }
        s = results['summary']
        print("\n" + "=" * 70)
        print(f"SUMMARY over {s['n_cells']} (noise x model) cells")
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
