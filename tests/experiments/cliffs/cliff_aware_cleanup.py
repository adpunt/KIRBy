#!/usr/bin/env python3
"""Does knowing which molecules are on cliffs make a cleanup work better.

HER OUTLINE, verbatim, which this is written from:

    "Performance in downstream tasks like AC-aware noise mitigation. Look at
    the best outlier detection methods, call one M, and then subsequent best
    label handling methods, call one L. Take dataset D. Run M on D, giving you
    a list of outliers. Now we have to determine if the outliers are noise or
    ACs. Take the AC detection method in question and run it, cross-check with
    the list of outliers. Now first, apply L without AC-awareness to get D'.
    Then apply L with AC-awareness to get D''. Compare the results of training
    on D, D', and D''. Assuming M and L work, for a higher-is-better metric, we
    should see D'' > D' > D. Potentially D > D' if the removal of ACs was
    actually damaging. But the important thing being D'' should be the best.
    And then the difference between D'' and D and D'' and D' should indicate
    the relative performance of different AC detection methods."

ONE ROW IS one AC definition, one M, one L, one setting of each, one
downstream model, one feature space and one fold. It carries four scores of
the same held-out molecules in the dataset's own metric:

  score_D            fitted on the training labels as the loader gives them
  score_Dprime       fitted on what L returned, knowing nothing about cliffs
  score_Dprimeprime  fitted on what L returned with the cliff molecules put
                     back, so L touched none of them
  score_random_put_back  the same with as many NON-cliff molecules put back
                     instead, which is the control: it differs from D'' in
                     which molecules were spared and not in how many

⛔ NO NOISE IS INJECTED. Her sentence is "Take dataset D", so D is the
dataset's own labels. There is no clean reference to score against and none is
reported: the comparison is between the three training sets, not against a
truth. This is what separates this experiment from the noise suite's own
D-against-D', which injects a known corruption and can score the clean labels.

THE CROSS-CHECK IS ON EVERY ROW. Her "cross-check with the list of outliers":
how many of M's outliers are cliff molecules, how many are not, and how many
cliff molecules M did not flag. Without it a row cannot be read, because D''
equals D' whenever M flagged no cliff molecule at all, and then the difference
between them is zero by construction rather than by measurement.

WHY THREE SETTINGS HAVE NO VALUE HERE. M and L are "the best" of the noise
experiments' own results, and those results do not exist yet. Which model
inputs are fitted is not said either: this fitted base_1, base_2 and hybrid as
"the same three the noise suite fits", and the noise suite cut base_2 on
2026-11-04. So this file is finished and does not run until
`outlier_detection_methods`, `label_handling_methods` and
`cliff_aware_cleanup_model_inputs` are written in the settings file. It says
so and stops rather than picking one.

M AND L RUN THROUGH n44's OWN CODE (2026-11-16). Her M and L are the best of
n44's results, so any detector or cleanup n44 can run is accepted, under the
name n44 writes before '@': its score detectors (n44.detector_scores), its
neighbour detectors and neighbour cleanups under the neighbour settings n44's
screen keeps (n42._Similarity), its two flag detectors (n44.flag_detectors),
and every cleanup with n44's own settings (n44.cleanup_settings,
n44.apply_cleanup), paired as n44 pairs them (n44._pairs_with). Until then it
read detectors from ESTIMATOR_REGISTRY and cleanups from n25, which left out
16 of n44's cleanups and 10 of its detectors. n44's oracle flags are not
accepted: they need injected noise, and this experiment injects none.

ONE CLEANED TRAINING SET, MANY DEFINITIONS. D' reads no cliff definition, so it
is built once per M, L setting, neighbour setting, model and model input, and
every definition puts its cliff molecules back into that same set. Every fit
goes through noise_cache.cached_fit_predict, so D and D' are fitted once and
read back for each definition.

Usage, as a unit of the papers ledger:
    python tests/experiments/run_dataset.py cliffs herg_ki \\
        --experiments cliff_aware_cleanup
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent.parent
REPO = TESTS.parent
for _p in (REPO / 'src', TESTS, TESTS / 'experiments'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments._shared import (get_domain_metric, get_model_factories,  # noqa: E402
                                 input_fingerprint,
                                 save_experiment_results)
from experiments.noise import cliff_rules as cr                      # noqa: E402
from experiments.noise import noise_cache as nc                      # noqa: E402
from experiments.noise import n07_remediation as n07                 # noqa: E402
from experiments.noise import n25_cliff_exemption as n25             # noqa: E402
from experiments.noise import n44_detect_and_clean as n44            # noqa: E402
from experiments.noise import n42_similarity_noise_sweep as n42      # noqa: E402
from experiments.noise import noise_timing                           # noqa: E402
from experiments.noise.d_vs_dprime import _fit_score as _dvd_fit_score  # noqa: E402
from experiments.cliffs import cliff_sets as P                       # noqa: E402
from experiments.cliffs import definitions as D                      # noqa: E402
from experiments.cliffs import settings as S                         # noqa: E402

EXP_ID = 'cliff_aware_cleanup'

#: The model inputs are the setting `cliff_aware_cleanup_model_inputs`, with no
#: value until she writes one (2026-11-16).

#: ⛔ THERE IS NO FIT CEILING, and there was one until 2026-11-06. It refused
#: above 20,000 model fits, a number Claude chose from a fit it had timed on
#: synthetic data. That is pruning the search space on an estimate, which
#: Adelaide's standing rule forbids: a cut needs the setting to be degenerate,
#: broken or undefined, and "my guess says this is too big" is none of those.
#: The run now prints the number of fits and the measured cost of one, and
#: starts.
FIT_COST_MEASURED = (
    'one svm fit and predict on 1,129 molecules by 900 features took 2.06 s '
    'and one dnn 9.18 s, measured 2026-11-04 on synthetic data of that shape. '
    'A real per-fit time is whatever this run records.')


#: The two flag detectors, by the name n44 writes before '@'.
FLAG_DETECTORS = ('conformal_bh', n44.TWO_LEARNER_DETECTOR)


def score_detectors(is_clf):
    """n44's detectors that give one score per molecule with no neighbour
    list, in the order n44.run_unit runs them."""
    return [d for d in (n44.REGISTRY_DETECTORS + n44.RESIDUAL_DETECTORS
                        + n44.N41_DETECTORS + n44.CONTROL_DETECTORS)
            if d not in n44.NEIGHBOUR_DETECTORS
            and not (is_clf and d in n44.REGRESSION_ONLY_DETECTORS)]


def known_detectors(is_clf=False):
    """Every detector n44 can run on labels with no injected noise."""
    out = score_detectors(is_clf) + list(n44.NEIGHBOUR_DETECTORS)
    if not is_clf:
        out += list(FLAG_DETECTORS)
    return sorted(out)


def known_cleanups(is_clf):
    """Every cleanup n44 can run on this task. A weighting cleanup on a model
    whose fit takes no per-molecule weight is recorded as not run, as n44
    records it."""
    return sorted(n44.cleanups_for(is_clf, True))


def _arm(cleaned, dataset_name, model, factory, X_va, y_va, metric_fn,
         metric_name, task_type, label):
    """One training set fitted and scored, through the noise suite's cached
    fit, so an identical training set is fitted once."""
    if cleaned is None:
        return float('nan'), f'{label}: the cleanup returned no training set'
    try:
        sc, _all = _dvd_fit_score(
            dataset_name, model, factory, cleaned['X'], cleaned['y'], X_va,
            y_va, metric_name, metric_fn,
            sample_weight=cleaned.get('sample_weight'), task_type=task_type)
        return sc, None
    except Exception as exc:
        return float('nan'), f'{label}: {type(exc).__name__}: {exc}'


def _n44_unit(dataset_name, data, model, space, fold_i, X_tr, X_va, y_tr, y_va,
              g_tr, train_idx, val_idx, metric, factories, errors, is_clf):
    """The unit n44's functions read, for the labels as given: level 0, no
    noise kind, no row sink."""
    return n42._Unit(
        exp_id=EXP_ID, twins=False, second_run=None, ds=dataset_name,
        sink=type('NoSink', (), {'n_written': 0})(), metric=metric,
        task_type=data.get('task_type') or 'regression', is_clf=is_clf,
        condition='labels_as_given', level=0.0, draw=0, fold=int(fold_i),
        model=model, mi=f'{model}:{space}', X_tr=X_tr, X_va=X_va, y0=y_tr,
        y_clean=y_tr, y_val=y_va, g_tr=g_tr, train_idx=train_idx,
        val_idx=val_idx, prov={}, cliff_train=None, rand_train=None,
        measurement_counts=None, row_models=(model,),
        takes_weight=bool(nc.fit_takes_sample_weight(factories[model])),
        learner_factories=factories, errors=errors, unavailable={},
        counts={}, fold_similarity=None)


def needs_neighbours(detectors, cleanups):
    return (any(d in n44.NEIGHBOUR_DETECTORS for d in detectors)
            or any(c in n44.NEIGHBOUR_CLEANUPS for c in cleanups))


def cleaned_sets(u, sp, sim, detectors, cleanups, targets, errors, where):
    """[(record, M's scores or None, M's flags or None, cleaned training set)]
    for every M and L setting n44 runs on this unit, built by n44's own
    functions. `record` names M (as n44 writes it), L, L's setting and the
    neighbour setting."""
    out = []
    is_clf = u.is_clf
    kept = list(sim.kept) if sim is not None else []

    def _clean(c, det, sc, fl, kw, lab, nb, lists=None, k=None):
        rec = {'outlier_detection_method': det,
               'label_handling_method': c,
               'label_handling_setting': lab, 'neighbour_setting': nb}
        with noise_timing.timed('cleanup', detector=det, cleanup=c,
                                setting=lab, similarity=nb) as step:
            try:
                cl = n44.apply_cleanup(c, sp, sc, fl, kw, lists=lists, k=k,
                                       targets=targets)
            except Exception as exc:
                errors[f'{where}|{det}/{c}|{lab}|{nb}'] = \
                    f'{type(exc).__name__}: {exc}'
                cl = None
            if cl is None:
                step.status = 'refused'
        out.append((rec, sc, fl, cl))

    # M as n44 runs it: (name written, base name, scores, flags, neighbour
    # setting, the setting's lists and k when M read them)
    ms = []
    for det in detectors:
        if det in n44.NEIGHBOUR_DETECTORS:
            for s, k, scheme in kept:
                L = sim.nb.get(s.name)
                if L is None:
                    continue
                with noise_timing.timed('detector', detector=det,
                                        similarity=s.name, k=k):
                    try:
                        sc = nc.cached_detector_scores(
                            u.ds, sp.X_tr, u.y0, is_clf=is_clf, detector=det,
                            groups=u.g_tr, similarity_neighbours=L,
                            similarity_k=k)
                    except Exception as exc:
                        errors[f'{where}|M={det}|{s.name}'] = \
                            f'{type(exc).__name__}: {exc}'
                        sc = None
                if sc is not None:
                    ms.append((det, det, np.asarray(sc, dtype=float), None,
                               (s, k, scheme)))
            if sim is None:
                errors[f'{where}|M={det}'] = 'no neighbour lists were built'
        elif det in FLAG_DETECTORS:
            with noise_timing.timed('detector', detector='flag_detectors'):
                flagged = n44.flag_detectors(sp)
            got = [(name, fl) for name, fl in flagged
                   if name.split('@')[0] == det]
            if not got:
                errors[f'{where}|M={det}'] = (
                    'n44.flag_detectors gave no flag set: ' + str(
                        u.unavailable.get(det, 'see the errors')))
            ms += [(name, det, None, fl, None) for name, fl in got]
        else:
            with noise_timing.timed('detector', detector=det):
                sc = n44.detector_scores(sp, det)
            if sc is None:
                errors[f'{where}|M={det}'] = 'n44.detector_scores gave no score'
                continue
            sc = np.asarray(sc, dtype=float)
            if not np.all(np.isfinite(sc)):
                fin = np.isfinite(sc)
                sc = np.where(fin, sc, np.nanmax(sc[fin]) if fin.any() else 0.0)
            ms.append((det, det, sc, None, None))

    for c in cleanups:
        d = n44.CLEANUPS[c]
        settings = n44.cleanup_settings(c)
        if d['neighbours'] and not kept:
            errors[f'{where}|L={c}'] = ('reads a neighbour list and none was '
                                        'built or kept on this unit')
            continue
        if d['reads'] == 'none':
            # reads no detector: D' is the same under every M, built once
            nbs = ([(s, k, scheme) for s, k, scheme in kept]
                   if d['neighbours'] else [None])
            for nb in nbs:
                for lab, kw in settings:
                    if nb is None:
                        _clean(c, 'none', None, None, kw, lab, n07.NO_NEIGHBOURS)
                    else:
                        s, k, scheme = nb
                        _clean(c, 'none', None, None, kw,
                               f'k={k}|scheme={scheme}|{lab}', s.name,
                               lists=sim.with_scheme(s, scheme), k=k)
            continue
        for name, base, sc, fl, m_nb in ms:
            is_flag = sc is None
            if not n44._pairs_with(c, base, is_flag):
                errors[f'{where}|{base}/{c}'] = 'n44 does not pair this cleanup with this detector'
                continue
            if c == 'rounds_delete':
                continue
            if d['neighbours']:
                if is_flag:
                    continue
                nbs = [m_nb] if m_nb is not None else kept
                for s, k, scheme in nbs:
                    if c == 'prox_smooth' and s.space != n44.ss.FOREST_SPACE:
                        continue
                    for lab, kw in settings:
                        _clean(c, name, sc, fl, kw,
                               f'k={k}|scheme={scheme}|{lab}', s.name,
                               lists=sim.with_scheme(s, scheme), k=k)
                continue
            if m_nb is not None:
                s, k, _scheme = m_nb
                # n44 runs a neighbour detector through a cleanup that reads
                # no neighbours once per (setting, k), not once per scheme
                if any(r['outlier_detection_method'] == name
                       and r['neighbour_setting'] == f'{s.name}|k={k}'
                       and r['label_handling_method'] == c
                       for r, *_x in out):
                    continue
                nb_lab = f'{s.name}|k={k}'
            else:
                nb_lab = n07.NO_NEIGHBOURS
            for lab, kw in settings:
                if is_flag and d['reads'] == 'flag' and d['axis'] == 'flag_share':
                    if kw.get('share') != n44.FLAG_SHARES[0]:
                        continue
                    lab = 'flags=detector'
                _clean(c, name, sc, fl, kw, lab, nb_lab)
    return out


def cross_check(outlier_flags, cliff_flags):
    """Her cross-check, as counts of molecules.

    Every number says what it counts. Without these a row cannot be read: D''
    equals D' whenever M flagged no cliff molecule, and the difference is then
    zero by construction.
    """
    o = np.asarray(outlier_flags, dtype=bool)
    c = np.asarray(cliff_flags, dtype=bool)
    return {
        'n_training_molecules': int(o.size),
        'n_outliers_M_flagged': int(o.sum()),
        'n_cliff_molecules': int(c.sum()),
        'n_outliers_that_are_cliff_molecules': int((o & c).sum()),
        'n_outliers_that_are_not_cliff_molecules': int((o & ~c).sum()),
        'n_cliff_molecules_M_did_not_flag': int((~o & c).sum()),
        'share_of_outliers_that_are_cliff_molecules': (
            float((o & c).sum()) / float(o.sum()) if o.sum() else None),
    }


# Time recording, 2026-11-16. Adelaide: "keep track of timing. They'll be
# tests in the future".
@noise_timing.logged_run(EXP_ID)
def run(data, config, output_dir, dataset_name, **kwargs):
    print('\n' + '=' * 70)
    print("CLIFF-AWARE CLEANUP: training on D, D' and D''")
    print('=' * 70)

    from experiments.noise.noise_helpers import (fold_configurations,
                                                 fold_model_inputs)
    from experiments.decided_grid import decided_models

    stored = S.read()
    labels = np.asarray(data['labels']).ravel()
    is_clf = data.get('task_type') == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    groups = data.get('groups')

    # ── the settings that have to be hers ────────────────────────────────
    try:
        detectors = S.methods_for('outlier_detection_methods',
                                  known_detectors(is_clf), stored)
        cleanups = S.methods_for('label_handling_methods',
                                 known_cleanups(is_clf), stored)
        SPACES = tuple(S.values('cliff_aware_cleanup_model_inputs', stored))
    except (S.Unset, ValueError) as exc:
        print(f'  NOT RUN: {exc}', flush=True)
        out = {'meta': {'dataset': dataset_name, 'waiting_for': str(exc),
                        'settings': S.record(stored)},
               'rows': [], 'result_status': 'skipped', 'skipped': True}
        save_experiment_results(out, output_dir, EXP_ID, dataset_name,
                                metric_name=metric_name,
                                lower_is_better=lower_is_better,
                                fingerprint=input_fingerprint(data, config))
        return out

    shares = S.values('outlier_shares', stored)
    scopes = S.values('definition_scope', stored)
    folds, n_folds = P.folds_of(data, config)
    try:
        models = [m for m in decided_models(dataset_name, 'general')
                  if m != 'knn']
    except Exception as exc:
        why = (f'the decided grid has no entry for {dataset_name}: '
               f'{type(exc).__name__}: {exc}')
        print(f'  REFUSED: {why}', flush=True)
        out = {'meta': {'dataset': dataset_name, 'refused': why}, 'rows': [],
               'result_status': 'failed'}
        save_experiment_results(out, output_dir, EXP_ID, dataset_name,
                                metric_name=metric_name,
                                lower_is_better=lower_is_better,
                                fingerprint=input_fingerprint(data, config))
        return out
    task_type = data.get('task_type') or 'regression'
    metric = (metric_name, metric_fn, lower_is_better)
    factories = get_model_factories(models, task_type, len(labels))
    models = [m for m in models if m in factories]

    # ── how big the grid is, before anything is fitted ───────────────────
    # The definitions are counted on a context that needs no search: the
    # settings decide how many there are, and the count is reported whether
    # the run starts or not. L's own settings are in it (n44.cleanup_settings),
    # and so are the neighbour settings n44's screen keeps, at most
    # n07.screen_top() per unit.
    n_defs = _count_definitions(data, stored, scopes)
    n_sets, how = cleaned_set_count(detectors, cleanups, is_clf,
                                    n_models=len(models))
    per_unit = len(models) * len(SPACES) * len(folds)
    n_fits = per_unit * (1 + n_sets * (1 + 2 * n_defs))
    print(f'  {per_unit} units ({len(models)} model(s) x {len(SPACES)} model '
          f'input(s) x {len(folds)} fold(s)); per unit 1 fit on D, '
          f'{n_sets} cleaned training sets ({how}), and for each of '
          f'{n_defs} definitions a cliff put-back and a random put-back of '
          f'every cleaned set: {n_fits:,} model fits at most', flush=True)
    print(f'  one fit, measured: {FIT_COST_MEASURED}', flush=True)

    jsonl = Path(output_dir) / EXP_ID / f'{dataset_name}__rows.jsonl'
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    if jsonl.exists():
        jsonl.unlink()

    n_rows, errors, fold_records = 0, {}, []
    t_all = time.time()
    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    n_molecules = len(set(smiles)) if has_smiles else len(labels)
    want_nb = needs_neighbours(detectors, cleanups)
    nb_settings = (n07.sweep_settings(dataset_name, n_molecules, has_smiles)
                   if want_nb else [])
    unit_errors = n42._Errors()

    fold_laps = noise_timing.Laps('fold')
    with open(jsonl, 'a') as sink:
        for fold_i, train_idx, val_idx in folds:
            fold_laps.lap(fold=int(fold_i))
            t0 = time.time()
            y_tr, y_va = labels[train_idx], labels[val_idx]
            g_tr = (None if groups is None
                    else np.asarray(groups)[np.asarray(train_idx)])
            print(f'\n  fold {fold_i} of {n_folds}: {len(train_idx)} training, '
                  f'{len(val_idx)} held-out molecules', flush=True)

            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=train_idx, n_folds=n_folds)
            smi_tr = ([smiles[i] for i in train_idx] if has_smiles else None)
            smi_va = ([smiles[i] for i in val_idx] if has_smiles else None)

            # D, and every cleaned training set, once per model and input:
            # none of them reads a cliff definition.
            units = {}
            for model in models:
                factory = factories[model]
                entry = entries.get(model)
                inputs, unavailable, what = fold_model_inputs(
                    data, dataset_name, model, entry,
                    train_idx=train_idx, val_idx=val_idx, y_train=y_tr,
                    factory=factory, metric=metric_name, spaces=SPACES)
                for space, why in (unavailable or {}).items():
                    errors[f'fold{fold_i}|{model}|{space}'] = why
                sp_tr = sp_va = sp_unavail = None
                if want_nb and inputs:
                    try:
                        with noise_timing.timed('neighbour_spaces', model=model):
                            sp_tr, sp_va, sp_unavail = n07.fold_similarity_spaces(
                                dataset_name, list(nb_settings),
                                model_input_tr=None, model_input_va=None,
                                reps=data['reps'], train_idx=train_idx,
                                val_idx=val_idx, y_train=y_tr, groups_train=g_tr,
                                base_reps=(entry or {}).get('base') or [],
                                recipes=(entry or {}).get('recipes') or {},
                                augmentations=data.get('augmentations'),
                                model_name=model, model_factory=factory,
                                metric=metric, task_type=task_type)
                    except Exception as exc:
                        errors[f'fold{fold_i}|{model}|neighbour_spaces'] = \
                            f'{type(exc).__name__}: {exc}'
                shared_lists = {}
                for space, (X_tr, X_va) in inputs.items():
                    where = f'fold{fold_i}|{model}|{space}'
                    noise_timing.set_context(model=model, model_input=space)
                    u = _n44_unit(dataset_name, data, model, space, fold_i,
                                  X_tr, X_va, y_tr, y_va, g_tr, train_idx,
                                  val_idx, metric, factories, unit_errors,
                                  is_clf)
                    sp = n44._Space('own', X_tr, X_va, u)
                    sim = None
                    if want_nb and sp_tr is not None:
                        try:
                            with noise_timing.timed('neighbour_lists_and_screen'):
                                sim = n42._Similarity(
                                    u, nb_settings, n07.sweep_ks(),
                                    dict(sp_tr, model_input=X_tr),
                                    dict(sp_va, model_input=X_va), shared_lists,
                                    sp_unavail, smi_tr, smi_va)
                        except Exception as exc:
                            errors[f'{where}|neighbour_lists'] = \
                                f'{type(exc).__name__}: {exc}'
                    targets = None
                    try:
                        with noise_timing.timed('correction_targets'):
                            targets = n44.n23.correction_targets_cached(
                                dataset_name, X_tr, y_tr, g_tr,
                                n07.targets_needed(list(n07.CLEANUPS)))
                    except Exception as exc:
                        errors[f'{where}|correction_targets'] = \
                            f'{type(exc).__name__}: {exc}'
                    sD, why = _arm({'X': X_tr, 'y': y_tr}, dataset_name, model,
                                   factory, X_va, y_va, metric_fn, metric_name,
                                   task_type, f'D {model}|{space}')
                    if why:
                        errors[f'{where}|D'] = why
                    sets = cleaned_sets(u, sp, sim, detectors, cleanups,
                                        targets, errors, where)
                    units[(model, space)] = (X_tr, X_va, factory,
                                             (what or {}).get(space), sD, sets)
                noise_timing.set_context(model=None, model_input=None)

            for scope in scopes:
                ctx, _whole, record = D.context_for_fold(
                    data, dataset_name, train_idx=train_idx, fold_i=fold_i,
                    n_folds=n_folds, stored=stored, scope=scope, verbose=False)
                if scope != 'per_fold':
                    print('    note: this scope computes the definition over '
                          'every molecule, so a held-out label reached it',
                          flush=True)
                defs = D.expand(ctx, stored=stored, scope=scope)
                got, errs, _cost = D.masks(ctx, defs)
                for key, why in errs.items():
                    errors[f'{scope}|{fold_i}|{key}'] = why

                for d in defs:
                    if d.key not in got:
                        continue
                    cliff_mask, counts = got[d.key]
                    cliff_train = _on_training_rows(cliff_mask, scope,
                                                    train_idx, len(labels))
                    for (model, space), (X_tr, X_va, factory, what, sD,
                                         sets) in units.items():
                        for rec, sc, fl, cleaned in sets:
                            row = _one_row(
                                dataset_name, fold_i, scope, d, rec, shares,
                                model, space, what, X_tr, X_va, y_tr, y_va,
                                factory, metric_fn, metric_name,
                                lower_is_better, task_type, sc, fl, cleaned,
                                cliff_train, sD, errors)
                            sink.write(json.dumps(row, default=float) + '\n')
                            n_rows += 1
                sink.flush()
                fold_records.append({
                    'fold': int(fold_i), 'definition_scope': scope,
                    'n_definitions': len(defs),
                    'n_definitions_that_did_not_run': len(errs),
                    'model_inputs_fitted': sorted(f'{m}|{s}' for m, s in units),
                    'n_cleaned_training_sets': {
                        f'{m}|{s}': len(v[5]) for (m, s), v in units.items()},
                    'hybrids_refused': record['hybrids_refused'],
                    'seconds': round(time.time() - t0, 1),
                })
                print(f'    scope {scope}: {n_rows} rows so far, '
                      f'{round(time.time() - t0, 1)}s', flush=True)
    fold_laps.close()
    for k, v in unit_errors.d.items():
        errors[f'n44|{k}'] = f"{v['count']}x {v['first']}"

    meta = {
        'dataset': dataset_name,
        'n_molecules': int(len(labels)),
        'metric': metric_name, 'lower_is_better': lower_is_better,
        'task_type': data.get('task_type'),
        'models': models, 'model_inputs': list(SPACES),
        'outlier_detection_methods': list(detectors),
        'label_handling_methods': list(cleanups),
        'outlier_shares': [float(s) for s in shares],
        'n_model_fits_at_most': int(n_fits),
        'how_the_fit_count_is_made': how,
        'n_outer_folds': int(n_folds),
        'definition_scopes_run': list(scopes),
        'no_noise_was_injected': (
            'her sentence is "Take dataset D", so D is the dataset\'s own '
            'labels. There is no clean reference and none is reported.'),
        'per_fold': fold_records,
        'n_rows': n_rows, 'rows_file': str(jsonl),
        'seconds': round(time.time() - t_all, 1),
        'settings': S.record(stored),
    }
    result = {
        'meta': meta, 'errors': errors,
        'what_one_row_is': (
            'one AC definition, one outlier detector, one cleanup and its '
            'setting, one outlier share, one downstream model, one feature '
            "space and one fold: four scores of the same held-out molecules "
            "in the dataset's own metric."),
        'how_to_read_a_row': (
            "D is the training labels as given, D' is the cleanup knowing "
            "nothing about cliffs, D'' is the cleanup with the cliff "
            'molecules put back, and the random arm puts back as many '
            'non-cliff molecules instead. Where M flagged no cliff molecule, '
            "D'' equals D' and the difference between them is zero by "
            'construction, which is what the cross-check counts show.'),
        'where_every_row_is': str(jsonl),
    }
    save_experiment_results(result, output_dir, EXP_ID, dataset_name,
                            metric_name=metric_name,
                            lower_is_better=lower_is_better,
                            fingerprint=input_fingerprint(data, config))
    print(f'\n  {n_rows} rows written to {jsonl}', flush=True)
    return result


def _one_row(dataset_name, fold_i, scope, d, rec, shares, model, space,
             what, X_tr, X_va, y_tr, y_va, factory, metric_fn, metric_name,
             lower_is_better, task_type, scores, flags, cleaned, cliff_train,
             score_D, errors):
    """One row: D' and the put-back arms, fitted and scored, and the
    cross-check. D' reads no definition, so its fit is the cached one."""
    where = (f"{scope}|{fold_i}|{model}|{space}|{d.key}|"
             f"{rec['outlier_detection_method']}/{rec['label_handling_method']}"
             f"|{rec['label_handling_setting']}|{rec['neighbour_setting']}")
    exempt = np.asarray(cliff_train, dtype=bool)
    rand, n_drawn = n25.random_exempt_mask(exempt, int(exempt.sum()),
                                           int(fold_i), 0)
    restored = restored_rand = None
    if cleaned is not None:
        restored = n25.put_back(cleaned, exempt, X_tr, y_tr)
        restored_rand = n25.put_back(cleaned, rand, X_tr, y_tr)
    args = (dataset_name, model, factory, X_va, y_va, metric_fn, metric_name,
            task_type)
    with noise_timing.timed('result', definition=d.key, **rec):
        sd, why_d = _arm(cleaned, *args, "D'")
        sdd, why_dd = _arm(restored, *args, "D''")
        srand, why_r = _arm(restored_rand, *args, 'random put back')
    for why in (why_d, why_dd, why_r):
        if why:
            errors[where] = why

    def _gain(treatment, reference):
        if not (np.isfinite(treatment) and np.isfinite(reference)):
            return float('nan')
        return ((reference - treatment) if lower_is_better
                else (treatment - reference))

    # Her cross-check: M's list of outliers against the cliff molecules. A
    # flag detector's list is its own flags. A score detector's list is its
    # top share of training molecules, at every outlier share (n44's rule,
    # n44.flags_from); a cleanup that reads flags also gets the cross-check
    # on the flags it actually used, at its own share.
    if flags is not None:
        by_share = {'flags=detector': cross_check(flags, exempt)}
    elif scores is not None:
        by_share = {f'{float(sh):g}': cross_check(
            n44.flags_from(scores, float(sh)), exempt) for sh in shares}
    else:
        by_share = {}
    own_share = None
    lab = str(rec['label_handling_setting'])
    if scores is not None and 'flag_share=' in lab:
        own_share = float(lab.split('flag_share=')[1].split('|')[0])
    used = (cross_check(n44.flags_from(scores, own_share), exempt)
            if own_share is not None else None)
    return {
        'dataset': dataset_name, 'fold': int(fold_i),
        'definition': d.key, 'definition_scope': scope,
        'coefficient': d.record.get('coefficient'),
        'representation': d.record.get('representation'),
        'similarity_used_labels': d.record.get('similarity_used_labels'),
        **rec,
        'model': model, 'model_input': space, 'what_the_input_is': what,
        'metric': metric_name, 'lower_is_better': bool(lower_is_better),
        'n_held_out_molecules': int(len(y_va)),
        'score_D': score_D, 'score_Dprime': sd, 'score_Dprimeprime': sdd,
        'score_random_put_back': srand,
        'Dprimeprime_over_Dprime': _gain(sdd, sd),
        'Dprimeprime_over_D': (_gain(sdd, score_D)
                               if score_D is not None else float('nan')),
        'Dprime_over_D': (_gain(sd, score_D)
                          if score_D is not None else float('nan')),
        'Dprimeprime_over_random_put_back': _gain(sdd, srand),
        'n_cliff_molecules_the_cleanup_would_have_touched': (
            n25.n_spared(cleaned, restored, exempt, y_tr)
            if cleaned is not None else None),
        'n_non_cliff_molecules_put_back_in_the_control': int(n_drawn),
        'control_put_back_fewer_than_the_cliff_arm':
            bool(n_drawn < int(exempt.sum())),
        'cross_check_by_outlier_share': by_share,
        'cross_check_on_the_flags_the_cleanup_used': used,
    }


def cleaned_set_count(detectors, cleanups, is_clf, n_models=2):
    """(cleaned training sets per unit at most, how that was counted), from
    n44's own settings. Neighbour settings count n07.screen_top(), the most
    n44's screen keeps; a flag detector counts every setting n44 gives it."""
    top = int(n07.screen_top())
    variants = {}
    for det in detectors:
        if det in n44.NEIGHBOUR_DETECTORS:
            variants[det] = ('score', top)
        elif det == 'conformal_bh':
            variants[det] = ('flag', len(n44.CONFORMAL_FLAG_SETTINGS))
        elif det == n44.TWO_LEARNER_DETECTOR:
            variants[det] = ('flag', len(n44.n32.ITLM_ALPHAS) if n_models > 1 else 0)
        else:
            variants[det] = ('score', 1)
    total, parts = 0, []
    for c in cleanups:
        d = n44.CLEANUPS[c]
        n_set = len(n44.cleanup_settings(c))
        nb = top if d['neighbours'] else 1
        if d['reads'] == 'none':
            n = n_set * nb
        else:
            n = 0
            for det, (kind, nv) in variants.items():
                if not n44._pairs_with(c, det, kind == 'flag'):
                    continue
                if kind == 'flag':
                    if d['neighbours']:
                        continue
                    per = (1 if (d['reads'] == 'flag'
                                 and d['axis'] == 'flag_share') else n_set)
                    n += nv * per
                elif det in n44.NEIGHBOUR_DETECTORS:
                    n += nv * n_set
                else:
                    n += n_set * nb
        total += n
        parts.append(f'{c} {n_set} setting(s)' + (
            f' x {top} neighbour settings' if d['neighbours'] else ''))
    return total, '; '.join(parts)


def _on_training_rows(mask, scope, train_idx, n_total):
    """The cliff flag over the fold's training rows.

    Under 'per_fold' the mask is already one per training molecule, because
    the definition was computed on that subset. Under 'whole_dataset' it is
    one per molecule and is taken at the training rows.
    """
    m = np.asarray(mask, dtype=bool)
    if len(m) == len(train_idx):
        return m
    if len(m) == n_total:
        return m[np.asarray(train_idx)]
    raise ValueError(
        f'a cliff flag of {len(m)} molecules fits neither the '
        f'{len(train_idx)} training molecules nor the {n_total} in the '
        f'dataset (scope {scope!r})')


def _count_definitions(data, stored, scopes):
    """How many definitions the settings ask for, without running a search.

    The hybrid names are stood in for so the count includes the definitions
    that read one; whether a hybrid actually builds is the run's business.
    """
    from kirby.similarity.preprocessing import feature_kind
    from experiments.cliffs.binary_hybrid import EXPECTED_KIND, HYBRID_REP_NAME
    labels = np.asarray(data['labels']).ravel()
    reps = {k: np.asarray(v, dtype=np.float64)
            for k, v in (data.get('reps') or {}).items()}
    kinds = {k: feature_kind(v) for k, v in reps.items()}
    stand_in = dict(reps)
    for kind, name in HYBRID_REP_NAME.items():
        key = f'{name}__stand_in'
        kinds[key] = EXPECTED_KIND[kind]
        stand_in[key] = np.zeros((len(labels), 1))
    ctx = cr.CliffContext(labels, smiles=data.get('smiles'), reps=stand_in,
                          is_clf=(data.get('task_type') == 'classification'),
                          dataset_name=data.get('dataset_name'))
    return sum(len(D.expand(ctx, stored=stored, kinds=kinds, scope=scope))
               for scope in scopes)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--output', default='tests/results/cliffs')
    ap.add_argument('--folds', default=None)
    a = ap.parse_args(argv)
    from experiments._shared import load_data
    import experiments.dataset_policy  # noqa: F401
    if a.folds:
        os.environ['KIRBY_NOISE_ONLY_FOLDS'] = a.folds
    data = load_data(a.dataset)
    run(data, {}, a.output, a.dataset)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
