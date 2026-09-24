#!/usr/bin/env python3
"""N29: does fitting a bias per ChEMBL document change what the model learns?

Paper: Noise.

THE QUESTION. herg_ic50 keeps one row per ChEMBL measurement, and every
measurement names the document (the paper) it came from. If one document's
measurements sit systematically above or below the others, the mean label of
a molecule carries that document's offset. This file fits one offset per
document, removes it from the training labels, and scores the model on held-out
molecules.

Where this came from. A research report of 2026-09-24 (sections C1 and C2)
proposed it from published work on labels from several annotators: Raykar et
al. 2010, Rodrigues and Pereira 2018. That report described the data as
"assets" valued by "sources". Here a source is a ChEMBL document, and nothing
else in that description carries over.

NOTHING IS INJECTED in the four main arms. The offsets are fitted to the labels
as they come. The one arm that injects is the check with known biases, below.

THE FOUR ARMS. Every arm is scored on the same held-out molecules, in the same
feature space, on the same folds.

  mean_label                 one row per molecule; the label is the mean of
                             all its measurements. What a loader normally
                             hands you. Every other arm is paired against it.
  per_record                 one row per measurement, the molecule's features
                             repeated on each. No offset. With squared loss
                             this differs from mean_label only in weighting a
                             molecule by how many times it was measured.
  per_record_bias            one row per measurement, the label minus the
                             fitted offset of that measurement's document.
  mean_label_bias_corrected  one row per molecule; the label is the mean over
                             its measurements of (label minus its document's
                             offset). The offsets are per_record_bias's.

HOW AN OFFSET IS FITTED. Two fits, alternated N_ITER times on the training
measurements of one fold, in one feature space:

  1. A LightGBM model predicts every training measurement out of fold, trained
     on the labels minus the current offsets. `kirby.utils.noisy_labels.
     oof_predict`, grouped by scaffold, so no molecule predicts its own other
     measurements.
  2. What that prediction leaves of each RAW label is regressed on which
     document the measurement came from, by ridge regression. A measurement
     belongs to exactly one document, so the ridge solution for a document is
     the sum of its residuals over (its measurement count + RIDGE_ALPHA).

After each pass the offsets are centred to sum to zero within each connected
group of documents (below). The downstream model is then fitted once, on the
labels minus the final offsets. The offsets come from LightGBM and not from the
downstream model, so all five downstream models are fitted on the same
corrected labels within one feature space.

WHICH DOCUMENTS GET AN OFFSET. Two documents are joined when they measured a
common training molecule. Only a document in a group of two or more documents
gets an offset. A document that shares no molecule with any other document has
an offset that cannot be told apart from its molecules' true values, so it is
set to 0. Within a group the offsets sum to zero, because a shift common to the
whole group cannot be told apart from the molecules' true values either. Group
sizes are written per fold, counted in documents, molecules and measurements.

MEASURED 2026-09-24 on the whole herg_ic50 pull, 12,008 measurements over 9,802
molecules and 1,676 documents: 573 documents are in 131 groups of two or more.
The largest group holds 220 documents, 1,198 molecules and 2,030 measurements.
So about one document in three can carry an offset at all.

WHAT THE HELD-OUT MOLECULES ARE GRADED AGAINST. The mean of their raw
measurements, in every arm. Those measurements carry their own documents'
offsets, and nothing here removes them, so a test label is not a clean label.
A second view grades only the held-out molecules none of whose documents
measured any training molecule in this fold. Their count is written every fold
and it can be small; it is never filled in.

HOW STABLE THE OFFSETS ARE. The training molecules are resampled with
replacement N_BOOTSTRAP times. The ridge step is refitted on the final
residuals of each resample, with no model refit. Per document, the share of
resamples in which its offset changes sign is written, with how many documents
exceed FLIP_SHARE_REPORTED.

THE CHECK WITH KNOWN BIASES. On fold 0, in the base space, with the first
downstream model: an offset is drawn for every document from a normal
distribution. Its variance is that of the offsets fitted on the real labels.
It is added to every measurement, and the four arms are run again. Held-out
molecules are graded against the mean of their measurements WITHOUT the added
offset. Written: the correlation between the added offsets and the fitted ones,
and each arm's score. This checks whether the fit can recover offsets of the
size it reported. It is not a result about the real labels.

WHERE THE HYBRID COMES IN. Every arm runs in the base representation and in one
hybrid per family of stored search, 'hybrid' and 'hybrid_similarity', rebuilt
per fold through `hybrid_for_fold`. The recipe is looked up per downstream
model, as n23 does since 2026-09-23. The hybrid is built on the training
molecules at their mean labels, once per fold, and every arm shares it. So no
offset reaches the feature choice. Whether it should is n20's kind of question.

Usage:
    python n29_document_bias.py --dataset herg_ic50 --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import functools
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS,
    get_best_search_for_downstream, pick_noise_base_rep,
    pool_recipes_by_family, describe_pool_family, POOL_SUBSETS,
)
from experiments.noise._ground_truth_providers import build_replicate_groups
from experiments.noise.n19_measurement_count import measurement_stats
from experiments.noise.noise_helpers import (
    hybrid_for_fold, pool_recipe_per_downstream_model, group_models_by_recipe,
)
from kirby.utils.noisy_labels import oof_predict

ARMS = ('mean_label', 'per_record', 'per_record_bias',
        'mean_label_bias_corrected')
ARM_TO_BEAT = 'mean_label'

# How many times the two fits alternate. Written onto the result file with the
# largest change in any offset on each pass, in pIC50 units, so a reader can
# see whether three passes had stopped moving.
N_ITER = 3

# The ridge penalty on the offsets, in measurements: a document with n
# training measurements has the mean of its residuals multiplied by
# n / (n + RIDGE_ALPHA). At 1.0 a document measured once keeps half its
# residual and one measured nine times keeps nine tenths.
RIDGE_ALPHA = 1.0

# Inner folds of the out-of-fold prediction the residuals are read off.
OOF_SPLITS = 5

# Resamples of the training molecules for the sign-flip count.
N_BOOTSTRAP = 200
FLIP_SHARE_REPORTED = 0.2

# Only a document in a group of this many documents or more gets an offset.
MIN_DOCUMENTS_IN_GROUP = 2

# The seed of the offsets drawn for the check with known biases.
KNOWN_BIAS_SEED = 0

POOL_SUBSET = 'predictive_and_similarity'
SPACE_FOR_FAMILY = {
    'predictive': 'hybrid',
    'similarity': 'hybrid_similarity',
}


# ── Records, molecules and documents ───────────────────────────────────────

def record_to_molecule(replicate_groups):
    """Each measurement row's molecule index, in `measurement_stats` order.

    `measurement_stats` sorts molecules by the row their first measurement sits
    on. The same order is rebuilt here, and the caller checks the two agree on
    every molecule's measurement count before using it.
    """
    groups = np.asarray(replicate_groups)
    first = {}
    for i, g in enumerate(groups):
        first.setdefault(g, i)
    order = sorted(first, key=first.get)
    index = {g: j for j, g in enumerate(order)}
    return np.array([index[g] for g in groups], dtype=np.int64)


def document_groups(rec_mol, rec_doc):
    """Connected groups of documents, joined by a molecule both measured.

    Args:
        rec_mol: molecule index per measurement.
        rec_doc: document index per measurement.

    Returns (component, info). `component` maps each document index present in
    `rec_doc` to a group id. `info` counts, per group, its documents, molecules
    and measurements.
    """
    rec_mol = np.asarray(rec_mol)
    rec_doc = np.asarray(rec_doc)
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for d in np.unique(rec_doc):
        find(int(d))
    first_doc = {}
    for m, d in zip(rec_mol.tolist(), rec_doc.tolist()):
        if m in first_doc:
            ra, rb = find(first_doc[m]), find(d)
            if ra != rb:
                parent[ra] = rb
        else:
            first_doc[m] = d
    component = {d: find(d) for d in parent}

    info = {}
    rec_comp = np.array([component[int(d)] for d in rec_doc])
    for c in np.unique(rec_comp):
        on = rec_comp == c
        info[int(c)] = {
            'documents': int(len(np.unique(rec_doc[on]))),
            'molecules': int(len(np.unique(rec_mol[on]))),
            'measurements': int(on.sum()),
        }
    return component, info


def eligible_documents(component, info, min_docs=MIN_DOCUMENTS_IN_GROUP):
    """Document indices that sit in a group of `min_docs` documents or more."""
    return sorted(d for d, c in component.items()
                  if info[int(c)]['documents'] >= int(min_docs))


def ridge_offsets(residual, rec_doc, eligible, component, alpha=RIDGE_ALPHA,
                  weight=None):
    """One offset per eligible document, centred to sum to zero per group.

    Ridge regression of the residuals on a one-hot document membership: each
    measurement belongs to exactly one document, so the normal equations are
    diagonal and a document's offset is sum(residuals) / (count + alpha).
    `weight` counts a measurement more than once, which is what a bootstrap
    resample does. A document absent from the rows gets NaN, not 0.

    Returns {document index: offset}.
    """
    residual = np.asarray(residual, dtype=float)
    rec_doc = np.asarray(rec_doc)
    w = np.ones_like(residual) if weight is None else np.asarray(weight, float)
    elig = set(int(d) for d in eligible)
    sums, counts = {}, {}
    for r, d, ww in zip(residual, rec_doc.tolist(), w):
        if d in elig and ww > 0:
            sums[d] = sums.get(d, 0.0) + ww * r
            counts[d] = counts.get(d, 0.0) + ww
    raw = {d: sums[d] / (counts[d] + float(alpha)) for d in sums}
    by_comp = {}
    for d, b in raw.items():
        by_comp.setdefault(component[d], []).append(b)
    centre = {c: float(np.mean(v)) for c, v in by_comp.items()}
    out = {d: np.nan for d in elig}
    out.update({d: b - centre[component[d]] for d, b in raw.items()})
    return out


def offsets_per_record(offsets, rec_doc):
    """The offset each measurement carries; 0 where its document has none."""
    return np.array([offsets.get(int(d), 0.0) if np.isfinite(
        offsets.get(int(d), np.nan)) else 0.0 for d in rec_doc], dtype=float)


def fit_document_offsets(X_rec, y_rec, rec_doc, g_rec, eligible, component,
                         n_iter=N_ITER, alpha=RIDGE_ALPHA,
                         oof_splits=OOF_SPLITS):
    """The alternating fit. Returns (offsets, final residuals, trace).

    `trace` is the largest change in any offset on each pass, in label units.
    The residual is the RAW label minus the out-of-fold prediction of a model
    trained on the corrected labels, so it is what the offsets must explain.
    """
    y_rec = np.asarray(y_rec, dtype=float)
    offsets = {int(d): 0.0 for d in eligible}
    trace = []
    residual = None
    for _ in range(int(n_iter)):
        b = offsets_per_record(offsets, rec_doc)
        oof = oof_predict(X_rec, y_rec - b, g_rec, n_splits=oof_splits)
        residual = y_rec - oof
        new = ridge_offsets(residual, rec_doc, eligible, component, alpha)
        moved = [abs(new[d] - offsets.get(d, 0.0)) for d in new
                 if np.isfinite(new[d])]
        trace.append(float(max(moved)) if moved else 0.0)
        offsets = {d: (v if np.isfinite(v) else 0.0) for d, v in new.items()}
    return offsets, residual, trace


def sign_flip_shares(residual, rec_doc, rec_mol_local, n_mol, eligible,
                     component, offsets, n_boot=N_BOOTSTRAP, alpha=RIDGE_ALPHA,
                     seed=0):
    """Per document, the share of resamples in which its offset changes sign.

    Molecules are resampled with replacement; a measurement counts as many
    times as its molecule was drawn. Only the ridge step is refitted. A
    document absent from a resample is not counted in that resample.
    """
    rng = np.random.default_rng(int(seed))
    flips = {d: 0 for d in eligible}
    seen = {d: 0 for d in eligible}
    for _ in range(int(n_boot)):
        draw = np.bincount(rng.integers(0, n_mol, n_mol), minlength=n_mol)
        w = draw[rec_mol_local].astype(float)
        got = ridge_offsets(residual, rec_doc, eligible, component, alpha, w)
        for d in eligible:
            v, ref = got.get(d, np.nan), offsets.get(d, 0.0)
            if not np.isfinite(v) or ref == 0.0:
                continue
            seen[d] += 1
            if np.sign(v) != np.sign(ref):
                flips[d] += 1
    return {d: (flips[d] / seen[d] if seen[d] else float('nan'))
            for d in eligible}


def _fit_score(factory, X_tr, y_tr, X_va, y_va, metric_fn):
    """One fit, one held-out score, or (None, why)."""
    try:
        m = factory()
        m.fit(X_tr, y_tr)
        return float(metric_fn(y_va, m.predict(X_va))), None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def arm_training_sets(X_ent_tr, y_ent_mean_tr, X_rec_tr, y_rec_tr, rec_mol_local,
                      b_rec):
    """The four arms' (X, y), all over the same training molecules."""
    n_mol = X_ent_tr.shape[0]
    corrected = y_rec_tr - b_rec
    sums = np.bincount(rec_mol_local, weights=corrected, minlength=n_mol)
    cnts = np.bincount(rec_mol_local, minlength=n_mol)
    return {
        'mean_label': (X_ent_tr, y_ent_mean_tr),
        'per_record': (X_rec_tr, y_rec_tr),
        'per_record_bias': (X_rec_tr, corrected),
        'mean_label_bias_corrected': (X_ent_tr, sums / np.maximum(cnts, 1)),
    }


def run(data, config, output_dir, dataset_name, **kwargs):
    """N29: one offset per ChEMBL document, fitted and removed from the labels."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N29: ONE OFFSET PER CHEMBL DOCUMENT")
    print("=" * 70)

    if data['task_type'] != 'regression':
        return {'skipped': 'not_regression', 'task_type': data['task_type']}
    rec_docs_raw = data.get('record_documents')
    if rec_docs_raw is None:
        print(f"  SKIPPED: {dataset_name} carries no document per measurement. "
              f"Only herg_ic50's loader returns one (herg.py, "
              f"load_herg_ic50_measurements).")
        return {'skipped': 'no_record_documents', 'dataset': dataset_name}
    rep_groups = build_replicate_groups(data)
    if rep_groups is None:
        return {'skipped': 'no_replicate_structure', 'dataset': dataset_name}

    labels = np.asarray(data['labels'], dtype=np.float64)
    if len(rec_docs_raw) != len(labels):
        raise RuntimeError(
            f"n29: {len(rec_docs_raw)} document ids for {len(labels)} "
            f"measurements; the ids are not on the measurements' rows.")
    reps = data['reps']
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    n_iter = int(kwargs.get('n_iter') or N_ITER)
    alpha = float(kwargs.get('ridge_alpha') if kwargs.get('ridge_alpha')
                  is not None else RIDGE_ALPHA)
    n_boot = int(kwargs.get('n_bootstrap') if kwargs.get('n_bootstrap')
                 is not None else N_BOOTSTRAP)
    oof_splits = int(kwargs.get('oof_splits') or OOF_SPLITS)

    stats = measurement_stats(labels, rep_groups)
    rec_mol = record_to_molecule(rep_groups)
    if not np.array_equal(np.bincount(rec_mol), stats['n_measurements']):
        raise RuntimeError(
            "n29: the molecule order rebuilt here disagrees with "
            "measurement_stats about how many measurements each molecule has.")
    doc_names, rec_doc = np.unique(np.asarray(rec_docs_raw).astype(str),
                                   return_inverse=True)
    n_mol = len(stats['n_measurements'])
    y_mean = stats['mean_all']

    row_groups = data.get('groups')
    if row_groups is None:
        raise RuntimeError("n29: no fold grouping; random splits are refused.")
    ent_groups = np.asarray(row_groups)[stats['row_index']]
    ent_reps = {k: np.asarray(v[stats['row_index']], dtype=np.float64)
                for k, v in reps.items()}
    base_rep_name = (kwargs.get('base_rep')
                     or pick_noise_base_rep(reps, dataset_name))
    splits = list(cv_split(ent_groups, n_folds, ent_reps[base_rep_name],
                           y_mean))
    largest_train = max(int(np.isin(rec_mol, tr).sum()) for tr, _ in splits)

    downstream_models = list(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS)
    model_factories = get_model_factories(
        downstream_models, task_type='regression', n_samples=largest_train)
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(f"n29: no downstream model could be built from "
                           f"{kwargs.get('models') or DOWNSTREAM_MODELS}.")

    print(f"  {len(labels)} measurements, {n_mol} molecules, "
          f"{len(doc_names)} documents, {len(splits)} folds")
    print(f"  models {downstream_models}; base representation {base_rep_name}")

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_injection': 'none in the four main arms',
        'models': downstream_models,
        'base_rep': base_rep_name,
        'arms': list(ARMS),
        'arm_to_beat': ARM_TO_BEAT,
        'n_iter': n_iter,
        'ridge_alpha': alpha,
        'ridge_alpha_units': ('measurements: a document with n training '
                              'measurements keeps n / (n + alpha) of the mean '
                              'of its residuals'),
        'oof_splits': oof_splits,
        'n_bootstrap': n_boot,
        'min_documents_in_group': MIN_DOCUMENTS_IN_GROUP,
        'dataset_counts': {'measurements': int(len(labels)),
                           'molecules': int(n_mol),
                           'documents': int(len(doc_names))},
        'offsets_come_from': ('an out-of-fold LightGBM prediction, not the '
                              'downstream model, so every downstream model in '
                              'one feature space is fitted on the same '
                              'corrected labels'),
        'offsets_reach_feature_selection': False,
        'grading_note': (
            'Held-out molecules are graded against the mean of their raw '
            'measurements, which carry their own documents\' offsets. '
            "'isolated' grades only held-out molecules none of whose documents "
            'measured a training molecule in that fold.'),
        'per_fold': {},
    }

    # ── Hybrid recipes, one per downstream model (n23's block) ──
    by_model, no_recipe = pool_recipe_per_downstream_model(
        functools.partial(
            get_best_search_for_downstream, reps, labels, row_groups,
            budget=None, augmentations=data.get('augmentations'),
            dataset_name=dataset_name, task_type='regression', data=data,
            pool_subset=POOL_SUBSET),
        downstream_models)
    results['hybrid_unavailable_by_model'] = {
        m: d['why'] for m, d in no_recipe.items()}
    recipe_for_space_by_model, hybrid_unavailable = {}, {}
    spaces = ['base']
    space_info = {}
    for m in downstream_models:
        rs, unav = pool_recipes_by_family(
            by_model[m]['meta'] if m in by_model else None, SPACE_FOR_FAMILY,
            eligible_families=POOL_SUBSETS.get(POOL_SUBSET))
        recipe_for_space_by_model[m] = {r['space']: r['blueprint'] for r in rs}
        for r in rs:
            if r['space'] not in spaces:
                spaces.append(r['space'])
            space_info.setdefault(r['space'], {})[m] = {
                'built_under_objective': r['objective'],
                'features_chosen_by': describe_pool_family(r['objective']),
                'budget': r['budget'],
                'allocation_method': r['allocation_method'],
                'reps': sorted(getattr(r['blueprint'], 'reps', []) or []),
            }
        for sp, why in unav.items():
            hybrid_unavailable.setdefault(sp, {})[m] = why
    results['hybrid_spaces'] = convert_numpy(space_info)
    results['hybrid_spaces_unavailable'] = hybrid_unavailable
    results['spaces_run'] = list(spaces)
    print(f"  spaces: {spaces}")

    # acc[view][space][arm][model][fold key] = score
    acc = {'all': {}, 'isolated': {}}
    errors = {}
    t0 = time.time()

    for fold_i, (tr, va) in enumerate(splits):
        fk = f'f{fold_i}'
        tr_set = np.zeros(n_mol, dtype=bool)
        tr_set[tr] = True
        rec_tr = np.where(tr_set[rec_mol])[0]
        local = np.full(n_mol, -1, dtype=np.int64)
        local[tr] = np.arange(len(tr))
        rec_mol_local = local[rec_mol[rec_tr]]
        rec_doc_tr = rec_doc[rec_tr]
        y_rec_tr = labels[rec_tr]
        g_rec_tr = ent_groups[rec_mol[rec_tr]]

        component, comp_info = document_groups(rec_mol[rec_tr], rec_doc_tr)
        eligible = eligible_documents(component, comp_info)
        big = [c for c in comp_info.values()
               if c['documents'] >= MIN_DOCUMENTS_IN_GROUP]
        train_docs = set(np.unique(rec_doc_tr).tolist())
        va_docs = {}
        for i in np.where(~tr_set[rec_mol])[0]:
            va_docs.setdefault(int(rec_mol[i]), set()).add(int(rec_doc[i]))
        isolated = np.array([not (va_docs.get(int(m), set()) & train_docs)
                             for m in va], dtype=bool)
        fold_block = {
            'training_molecules': int(len(tr)),
            'training_measurements': int(len(rec_tr)),
            'training_documents': int(len(train_docs)),
            'held_out_molecules': int(len(va)),
            'held_out_molecules_isolated': int(isolated.sum()),
            'document_groups_of_2_or_more': {
                'groups': len(big),
                'documents': int(sum(c['documents'] for c in big)),
                'molecules': int(sum(c['molecules'] for c in big)),
                'measurements': int(sum(c['measurements'] for c in big)),
                'largest_group_documents': (max(c['documents'] for c in big)
                                            if big else 0),
            },
            'documents_given_an_offset': len(eligible),
            'per_space': {},
        }
        print(f"\n  fold {fold_i}: {len(tr)} training molecules, "
              f"{len(rec_tr)} measurements, {len(eligible)} of "
              f"{len(train_docs)} documents can carry an offset; "
              f"{int(isolated.sum())} of {len(va)} held-out molecules share "
              f"no document with training")

        y_va_all = y_mean[va]
        ent_tr = {k: v[tr] for k, v in ent_reps.items()}
        ent_va = {k: v[va] for k, v in ent_reps.items()}

        # (space, X_tr_ent, X_va_ent, models)
        cells = []
        sc = StandardScaler()
        cells.append(('base', sc.fit_transform(ent_tr[base_rep_name]),
                      sc.transform(ent_va[base_rep_name]),
                      list(downstream_models)))
        for sp in spaces:
            if sp == 'base':
                continue
            for bp_m, mnames, shareable in group_models_by_recipe(
                    {m: recipe_for_space_by_model.get(m, {}).get(sp)
                     for m in downstream_models}, downstream_models):
                X_tr_h, X_va_h = hybrid_for_fold(
                    ent_tr, ent_va, y_mean[tr], bp_m, ent_groups[tr],
                    None if shareable else model_factories[mnames[0]](),
                    metric_fn, lower_is_better)
                if X_tr_h is None:
                    errors[f'{fk}|{sp}/{mnames[0]}|build'] = (
                        'the hybrid could not be built on this fold')
                    continue
                sh = StandardScaler()
                cells.append((sp, sh.fit_transform(X_tr_h),
                              sh.transform(X_va_h), list(mnames)))

        for space, X_tr_e, X_va_e, cell_models in cells:
            ckey = (space if cell_models == list(downstream_models)
                    else f'{space}/{cell_models[0]}')
            X_rec_tr = X_tr_e[rec_mol_local]
            try:
                offsets, resid, trace = fit_document_offsets(
                    X_rec_tr, y_rec_tr, rec_doc_tr, g_rec_tr, eligible,
                    component, n_iter=n_iter, alpha=alpha,
                    oof_splits=oof_splits)
            except Exception as exc:
                errors[f'{fk}|{ckey}|offsets'] = f'{type(exc).__name__}: {exc}'
                continue
            flips = sign_flip_shares(resid, rec_doc_tr, rec_mol_local, len(tr),
                                     eligible, component, offsets, n_boot,
                                     alpha, seed=fold_i)
            vals = np.array([offsets[d] for d in eligible], dtype=float)
            fl = np.array([v for v in flips.values() if np.isfinite(v)])
            fold_block['per_space'][ckey] = {
                'largest_offset_change_per_pass': trace,
                'offset_spread_sd': (float(vals.std()) if vals.size
                                     else float('nan')),
                'offset_abs_max': (float(np.abs(vals).max()) if vals.size
                                   else float('nan')),
                'documents_with_an_offset': int(vals.size),
                'documents_flipping_sign_in_over_'
                f'{FLIP_SHARE_REPORTED:g}_of_resamples':
                    int((fl > FLIP_SHARE_REPORTED).sum()),
                'documents_with_a_flip_share': int(fl.size),
            }
            b_rec = offsets_per_record(offsets, rec_doc_tr)
            sets = arm_training_sets(X_tr_e, y_mean[tr], X_rec_tr, y_rec_tr,
                                     rec_mol_local, b_rec)
            for m in cell_models:
                for arm in ARMS:
                    Xa, ya = sets[arm]
                    # One fit, graded two ways: every held-out molecule, and
                    # only the isolated ones.
                    try:
                        mdl = model_factories[m]()
                        mdl.fit(Xa, ya)
                        pred = mdl.predict(X_va_e)
                    except Exception as exc:
                        errors[f'{fk}|{ckey}|{arm}|{m}'] = (
                            f'{type(exc).__name__}: {exc}')
                        continue
                    (acc['all'].setdefault(space, {}).setdefault(arm, {})
                     .setdefault(m, {}))[fk] = float(metric_fn(y_va_all, pred))
                    if isolated.sum() >= 5:
                        (acc['isolated'].setdefault(space, {})
                         .setdefault(arm, {}).setdefault(m, {}))[fk] = float(
                            metric_fn(y_va_all[isolated], pred[isolated]))

            # ── The check with known biases: fold 0, base, first model ──
            if fold_i == 0 and space == 'base' and vals.size >= 2:
                results['known_bias_check'] = convert_numpy(_known_bias_check(
                    labels, rec_mol, rec_doc, doc_names, tr, va, rec_tr,
                    rec_mol_local, X_tr_e, X_va_e, g_rec_tr, eligible,
                    component, float(vals.var()), offsets, y_mean,
                    model_factories[downstream_models[0]],
                    downstream_models[0], metric_fn, n_iter, alpha,
                    oof_splits))

        results['per_fold'][fk] = convert_numpy(fold_block)
        print(f"    fold {fold_i} done [{time.time() - t0:.0f}s]")

    results['scores'] = convert_numpy(acc)
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy(_summarise(
        acc, downstream_models, metric_name, lower_is_better))
    _print_summary(results, metric_name)
    return results


def _known_bias_check(labels, rec_mol, rec_doc, doc_names, tr, va, rec_tr,
                      rec_mol_local, X_tr_e, X_va_e, g_rec_tr, eligible,
                      component, var_b, offsets_real, y_mean, factory,
                      model_name, metric_fn, n_iter, alpha, oof_splits):
    """Add a known offset per document, refit, and compare.

    Graded against the held-out molecules' mean measurement WITHOUT the added
    offset. Only differences within a group are identifiable, so the added
    offsets are centred per group before they are correlated with the fitted
    ones. `fitted_minus_real` removes the offsets the same fit found on the
    unshifted labels, leaving what it attributed to the added part.
    """
    rng = np.random.default_rng(KNOWN_BIAS_SEED)
    b_true = rng.normal(0.0, np.sqrt(var_b), size=len(doc_names))
    shifted = labels + b_true[rec_doc]
    y_rec_tr = shifted[rec_tr]
    rec_doc_tr = rec_doc[rec_tr]
    off, _resid, trace = fit_document_offsets(
        X_tr_e[rec_mol_local], y_rec_tr, rec_doc_tr, g_rec_tr, eligible,
        component, n_iter=n_iter, alpha=alpha, oof_splits=oof_splits)

    by_comp = {}
    for d in eligible:
        by_comp.setdefault(component[d], []).append(b_true[d])
    centre = {c: np.mean(v) for c, v in by_comp.items()}
    truth = np.array([b_true[d] - centre[component[d]] for d in eligible])
    fitted = np.array([off[d] for d in eligible])
    fitted_minus_real = fitted - np.array([offsets_real[d] for d in eligible])

    n_mol_tr = len(tr)
    sums = np.bincount(rec_mol_local, weights=y_rec_tr, minlength=n_mol_tr)
    cnts = np.bincount(rec_mol_local, minlength=n_mol_tr)
    sets = arm_training_sets(X_tr_e, sums / np.maximum(cnts, 1),
                             X_tr_e[rec_mol_local], y_rec_tr, rec_mol_local,
                             offsets_per_record(off, rec_doc_tr))
    scores = {}
    for arm in ARMS:
        Xa, ya = sets[arm]
        s, err = _fit_score(factory, Xa, ya, X_va_e, y_mean[va], metric_fn)
        scores[arm] = s if s is not None else err

    def _r(a, b):
        return (float(np.corrcoef(a, b)[0, 1])
                if len(a) >= 3 and np.std(a) > 0 and np.std(b) > 0
                else float('nan'))

    return {
        'what_it_is': ('a check with known biases, on fold 0, in the base '
                       'space, with one downstream model. Not a result about '
                       'the real labels.'),
        'model': model_name,
        'added_offset_sd': float(np.sqrt(var_b)),
        'added_offset_sd_units': 'pIC50',
        'documents_compared': int(len(eligible)),
        'pearson_added_vs_fitted': _r(truth, fitted),
        'pearson_added_vs_fitted_minus_real': _r(truth, fitted_minus_real),
        'largest_offset_change_per_pass': trace,
        'held_out_score_per_arm_against_unshifted_labels': scores,
    }


def _paired(base, got, lower_is_better):
    """One arm against another on the folds both scored. Positive is better."""
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
        'folds_tied': int((np.abs(gains) <= tol).sum()),
        'mean_gain': float(gains.mean()) if gains.size else float('nan'),
        'spread_over_fold_gains': (float(gains.std()) if gains.size
                                   else float('nan')),
        'fold_gains': [float(x) for x in gains],
        # Raw spread only, no test: at five folds the suite's rule is no CI.
        'gain_exceeds_its_own_spread': bool(gains.size > 0
                                            and gains.mean() > gains.std()),
    }


def _summarise(acc, models, metric_name, lower_is_better):
    """Every arm against mean_label, per (view, space, model), paired per fold."""
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'headline': 'folds won against mean_label, paired per fold',
           'per_view': {}}
    for view, by_space in acc.items():
        vrow = {}
        for space, by_arm in by_space.items():
            srow = {}
            base = by_arm.get(ARM_TO_BEAT) or {}
            for m in models:
                b = base.get(m) or {}
                if not b:
                    continue
                row = {'mean_label_mean': float(np.mean(list(b.values()))),
                       'per_arm': {}}
                for arm in ARMS:
                    if arm == ARM_TO_BEAT:
                        continue
                    g = (by_arm.get(arm) or {}).get(m) or {}
                    if g:
                        row['per_arm'][arm] = dict(
                            _paired(b, g, lower_is_better),
                            mean=float(np.mean(list(g.values()))))
                srow[m] = row
            vrow[space] = srow
        out['per_view'][view] = vrow
    return out


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric_name}; paired per fold against mean_label)")
    for view, vrow in s['per_view'].items():
        print(f"\n  held-out molecules graded: "
              f"{'all' if view == 'all' else 'only those sharing no document with training'}")
        for space, srow in vrow.items():
            for m, row in srow.items():
                print(f"    {space}/{m}: mean_label "
                      f"{row['mean_label_mean']:.4f}")
                for arm, c in row['per_arm'].items():
                    mark = '  >spread' if c['gain_exceeds_its_own_spread'] else ''
                    print(f"      {arm:26s} won {c['folds_won']} of "
                          f"{c['n_folds_paired']} folds, mean "
                          f"{c['mean_gain']:+.4f} (spread "
                          f"{c['spread_over_fold_gains']:.4f}){mark}")
    kb = results.get('known_bias_check')
    if kb:
        print(f"\n  check with known biases ({kb['model']}, fold 0, base): "
              f"added offsets sd {kb['added_offset_sd']:.4f} pIC50 over "
              f"{kb['documents_compared']} documents; pearson added vs fitted "
              f"{kb['pearson_added_vs_fitted']:.3f}, vs fitted minus real "
              f"{kb['pearson_added_vs_fitted_minus_real']:.3f}")


def main():
    parser = create_experiment_parser(
        'N29: one offset per ChEMBL document, fitted and removed')
    parser.add_argument('--n-iter', type=int, default=None,
                        help=f'Alternations of the two fits (default {N_ITER})')
    parser.add_argument('--ridge-alpha', type=float, default=None,
                        help=f'Ridge penalty on the offsets, in measurements '
                             f'(default {RIDGE_ALPHA})')
    parser.add_argument('--n-bootstrap', type=int, default=None,
                        help=f'Resamples for the sign-flip count '
                             f'(default {N_BOOTSTRAP})')
    parser.add_argument('--base-rep', type=str, default=None,
                        help="The base space's representation. Default: "
                             "pick_noise_base_rep's")
    parser.add_argument('--tag', type=str, default=None,
                        help='Extra text on the result filename')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, rep_preset=args.reps)
    kw = {}
    if args.n_iter:
        kw['n_iter'] = args.n_iter
    if args.ridge_alpha is not None:
        kw['ridge_alpha'] = args.ridge_alpha
    if args.n_bootstrap is not None:
        kw['n_bootstrap'] = args.n_bootstrap
    if args.base_rep:
        kw['base_rep'] = args.base_rep
    res = run(data, config, args.output, args.dataset, **kw)
    save_experiment_results(res, args.output, 'n29_document_bias',
                            args.dataset, filename_suffix=args.tag)


if __name__ == '__main__':
    main()
