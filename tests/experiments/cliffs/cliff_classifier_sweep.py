#!/usr/bin/env python3
"""The sweep over her four is-this-a-cliff classifiers.

HER INSTRUCTION, 2026-11-28: "lets test these. That's what the next interactive session
shuold test, try differnet models (ideally using the onfiugrations mentioned in papers), use
the respective datasets. This is its own sweep. The best will go forward to be used in the
paper 3 experiments" and "Feel free to test a good amount".

WHAT ONE ROW IS. One row is one classifier, on one target, on one fold, scored on one group of
held-out pairs. The groups are `both_seen`, `one_seen` and `neither_seen`, which say how many
of a pair's two molecules were among the fold's training molecules. `neither_seen` is the only
group with no leakage, and the published accuracies of 80 to 90 per cent came from scoring
mostly on `both_seen`.

WHAT IS FITTED ON WHAT. The molecules are split first, by Bemis-Murcko scaffold, using this
lab's own splitter. A classifier is fitted only on the pairs whose two molecules are both
training molecules, so no held-out molecule is seen while fitting.

THE AXES, all of them read from CLIFF_SETTINGS.json, every one shipping as "all":

  pair_classifier_kinds   per_molecule, mmp_kernel, pair_vector, twin
  pair_potency_cutoffs    1.0 and 2.0 log units
  pair_label_rules        threshold and excluded_middle
  pair_inputs             sum_absdiff, concat, absdiff  (pair_vector only)
  pair_vector_models      knn, rf, lightgbm, xgboost, svm, dnn  (pair_vector only)
  pair_training_scope      per_target and pooled

Narrow a session with --representations, --models, --kinds and --targets. Narrowing is for a
four-hour session, not a decision: the settings file is what says which values the study
covers.

WHAT IS NOT IN HERE. Her hybrid representations. Her words of 2026-11-28: "obviously I'll
want to try this with hybrids in the full set of experiments". The hybrids are built by a
per-fold search on her own five datasets, not on MoleculeACE's targets, so they enter when
this sweep's winner is carried into the Paper 3 experiments. The matched molecular pair kernel
cannot take a hybrid as published in any case: a Tanimoto kernel is defined on 0/1
fingerprints and a hybrid representation is a weighted real-valued distance.

    srun --clusters=arc -c 16 --mem=64G -t 4:00:00 --pty /bin/bash
    python tests/experiments/cliffs/cliff_classifier_sweep.py \
        --output /data/stat-ecr/scat9264/cliff_pair_sweep
"""
from __future__ import annotations

import argparse
import json
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

from experiments.cliffs import cliff_pairs as CP              # noqa: E402
from experiments.cliffs import cliff_classifiers as CC        # noqa: E402
from experiments.cliffs import moleculeace_agreement as MA    # noqa: E402
from experiments.cliffs import settings as S                  # noqa: E402

OUT_DEFAULT = Path('tests/results/comparisons')

#: A fold's training pairs must hold at least this many of each class for a classifier to be
#: fitted at all. Below it the fold is recorded as refused, with the two counts.
MIN_PER_CLASS = 5


def _settings(stored=None):
    """The six sweep axes, each as the list of values to run."""
    stored = S.read() if stored is None else stored
    return {name: list(S.values(name, stored)) for name in (
        'pair_classifier_kinds', 'pair_potency_cutoffs', 'pair_label_rules',
        'pair_inputs', 'pair_vector_models', 'pair_training_scope')}


def _load_target(name, data_dir, rep_dir):
    """One target's molecules, potencies, representations, scaffold groups and flag."""
    import pandas as pd

    csv = Path(data_dir) / f'{name}.csv'
    npz = Path(rep_dir) / f'sim_{name}.npz'
    if not csv.exists():
        return {'refused': f'no csv at {csv}'}
    if not npz.exists():
        return {'refused': f'no npz at {npz}: build_moleculeace.py has not '
                           f'run on this target'}
    frame = pd.read_csv(csv)
    store = np.load(npz, allow_pickle=True)
    reps = {k: store[k] for k in store.files
            if k not in ('smiles', 'y', 'groups', 'tanimoto', 'desc')}
    return {'smiles': frame['smiles'].tolist(),
            'y': frame['y [pEC50/pKi]'].to_numpy(dtype=float),
            'theirs': frame['cliff_mol'].to_numpy(dtype=int),
            'reps': reps,
            'groups': store['groups'],
            'desc': (store['desc'] if 'desc' in store.files else None)}


def _fragment_blocks(pairs):
    """The three fingerprint blocks the matched molecular pair kernel needs."""
    return {'core': CC.fragment_fingerprints(pairs['core']),
            'sub_i': CC.fragment_fingerprints(pairs['sub_i']),
            'sub_j': CC.fragment_fingerprints(pairs['sub_j'])}


def _subset_pairs(pairs, mask):
    """The pairs the mask selects, as the same shape of dict."""
    mask = np.asarray(mask, dtype=bool)
    return {'i': pairs['i'][mask], 'j': pairs['j'][mask],
            'core': [c for c, keep in zip(pairs['core'], mask) if keep],
            'sub_i': [c for c, keep in zip(pairs['sub_i'], mask) if keep],
            'sub_j': [c for c, keep in zip(pairs['sub_j'], mask) if keep],
            'delta': pairs['delta'][mask]}


def _canonical_order(smiles):
    """A sortable key per molecule, so `concat` does not encode the dataset's order."""
    from rdkit import Chem
    out = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        out.append(Chem.MolToSmiles(mol) if mol is not None else smi)
    return out


def _rows_for_fold(target, fold_i, train_idx, val_idx, data, pairs, blocks,
                   axes, reps_wanted, order_key, verbose):
    """Every classifier's rows on one fold of one target."""
    rows, errors = [], {}
    n_molecules = len(data['smiles'])
    groups = CP.pair_groups(pairs, train_idx, n_molecules)
    # A training pair must have both molecules among the fold's training molecules. If every
    # such pair were fitted on, the both_seen group would be empty and the published
    # leakage-rich setting would have no number here, so a fixed share of the both_seen pairs
    # is held out instead. The share is chosen by the pair's own molecule indices, not at
    # random, so the same pairs are held out on every classifier and every repeat.
    both = np.where(groups['both_seen'])[0]
    keep_back = both[(pairs['i'][both] * 31 + pairs['j'][both]) % 5 == fold_i % 5]
    train_mask = groups['both_seen'].copy()
    train_mask[keep_back] = False
    held_out = {'both_seen': np.zeros(len(train_mask), dtype=bool),
                'one_seen': groups['one_seen'],
                'neither_seen': groups['neither_seen']}
    held_out['both_seen'][keep_back] = True

    for cutoff in axes['pair_potency_cutoffs']:
        for rule in axes['pair_label_rules']:
            label, kept = CP.pair_labels(pairs['delta'], cutoff, rule)
            fit_mask = train_mask & kept
            y_fit = label[fit_mask]
            n_pos = int(y_fit.sum())
            n_neg = int(len(y_fit) - n_pos)
            if n_pos < MIN_PER_CLASS or n_neg < MIN_PER_CLASS:
                errors[f'fold{fold_i}|cutoff{cutoff}|{rule}'] = (
                    f'the fold\'s training pairs hold {n_pos} cliff pairs and '
                    f'{n_neg} non-cliff pairs, and {MIN_PER_CLASS} of each is '
                    f'the floor')
                continue

            def record(kind, model, representation, pair_input, score_of,
                       threshold, chosen=None):
                # `chosen` is filled in by score_of while it fits, so it is read after the
                # call and not before.
                extra = {}
                for group_name, group_mask in held_out.items():
                    sel = group_mask & kept
                    if not sel.any():
                        continue
                    y_true = label[sel]
                    try:
                        y_score = score_of(sel)
                    except Exception as exc:
                        errors[f'fold{fold_i}|cutoff{cutoff}|{rule}|{kind}|'
                               f'{model}|{representation}|{pair_input}|'
                               f'{group_name}'] = f'{type(exc).__name__}: {exc}'
                        return
                    rows.append({
                        'target': target, 'fold': int(fold_i),
                        'classifier_kind': kind, 'model': model,
                        'representation': representation,
                        'pair_input': pair_input,
                        'potency_cutoff_log_units': float(cutoff),
                        'label_rule': rule,
                        'held_out_group': group_name,
                        'n_training_pairs': int(fit_mask.sum()),
                        'n_training_cliff_pairs': n_pos,
                        'chosen_C': (chosen or {}).get('value'),
                        **extra,
                        **CC.scores(y_true, y_score, threshold=threshold)})

            # ---- the matched molecular pair kernel -------------------------
            if 'mmp_kernel' in axes['pair_classifier_kinds']:
                train_blocks = {k: v[fit_mask] for k, v in blocks.items()}
                chosen_c = {}

                def kernel_scores(sel, _tb=train_blocks, _c=chosen_c):
                    test_blocks = {k: v[sel] for k, v in blocks.items()}
                    got, picked = CC.fit_mmp_kernel_svm(_tb, test_blocks,
                                                        y_fit)
                    _c['value'] = picked
                    return got

                record('mmp_kernel', 'svm', 'core and substituent ecfp4 at '
                       f'{CC.FRAGMENT_BITS} bits', 'mmp_kernel',
                       kernel_scores, 0.0, chosen=chosen_c)

            # ---- a classifier on a pair vector -----------------------------
            if 'pair_vector' in axes['pair_classifier_kinds']:
                for rep_name in reps_wanted:
                    X = data['reps'][rep_name]
                    for how in axes['pair_inputs']:
                        vectors = CC.pair_vectors(X, pairs, how,
                                                  order_key=order_key)
                        V_fit = vectors[fit_mask]
                        for model_name in axes['pair_vector_models']:
                            def vector_scores(sel, _v=vectors, _vf=V_fit,
                                              _m=model_name):
                                return CC.fit_pair_vector_model(
                                    _m, _vf, y_fit, _v[sel])
                            record('pair_vector', model_name, rep_name, how,
                                   vector_scores, 0.5)

            # ---- the twin network ------------------------------------------
            if 'twin' in axes['pair_classifier_kinds']:
                for rep_name in reps_wanted:
                    X = data['reps'][rep_name]
                    fit_pairs = _subset_pairs(pairs, fit_mask)

                    def twin_scores(sel, _X=X, _fp=fit_pairs):
                        return CC.fit_twin_network(
                            _X, _fp, y_fit, _subset_pairs(pairs, sel))
                    record('twin', 'twin_network', rep_name, 'twin',
                           twin_scores, 0.0)

    # ---- the per-molecule comparison, scored on molecules not pairs --------
    if 'per_molecule' in axes['pair_classifier_kinds']:
        theirs = data['theirs']
        y_tr = theirs[train_idx]
        if len(set(y_tr.tolist())) > 1:
            for rep_name in reps_wanted:
                X = data['reps'][rep_name]
                for model_name in axes['pair_vector_models']:
                    try:
                        got = CC.fit_pair_vector_model(
                            model_name, X[train_idx], y_tr, X[val_idx])
                    except Exception as exc:
                        errors[f'fold{fold_i}|per_molecule|{model_name}|'
                               f'{rep_name}'] = f'{type(exc).__name__}: {exc}'
                        continue
                    rows.append({
                        'target': target, 'fold': int(fold_i),
                        'classifier_kind': 'per_molecule',
                        'model': model_name, 'representation': rep_name,
                        'pair_input': None,
                        'potency_cutoff_log_units': None,
                        'label_rule': "MoleculeACE's own cliff_mol flag",
                        'held_out_group': 'held_out_molecules',
                        'n_training_pairs': None,
                        'n_training_cliff_pairs': None,
                        **CC.scores(theirs[val_idx], got, threshold=0.5)})

    if verbose:
        best = max((r for r in rows
                    if r['held_out_group'] == 'neither_seen'
                    and r.get('matthews_correlation') is not None),
                   key=lambda r: r['matthews_correlation'], default=None)
        line = f'    fold {fold_i}: {len(rows)} rows'
        if best is not None:
            line += (f', best on pairs with neither molecule seen: '
                     f'{best["classifier_kind"]}/{best["model"]}/'
                     f'{best["representation"]} '
                     f'{best["matthews_correlation"]:.4f}')
        print(line, flush=True)
    return rows, errors


def run_target(name, data_dir, rep_dir, axes, reps_wanted=None, verbose=True):
    """Every row for one target."""
    started = time.time()
    data = _load_target(name, data_dir, rep_dir)
    if 'refused' in data:
        return {'target': name, 'refused': data['refused'], 'rows': []}

    reps = sorted(data['reps'])
    if reps_wanted:
        missing = [r for r in reps_wanted if r not in data['reps']]
        if missing:
            return {'target': name,
                    'refused': f'the npz has no {missing}; it has {reps}',
                    'rows': []}
        reps = list(reps_wanted)

    if verbose:
        print(f'  {name}: {len(data["smiles"])} molecules, '
              f'{len(reps)} representations', flush=True)

    pairs = CP.build_pairs(data['smiles'], data['y'])
    if len(pairs['i']) == 0:
        return {'target': name,
                'refused': 'no two molecules share a core under one cut',
                'rows': []}
    blocks = (_fragment_blocks(pairs)
              if 'mmp_kernel' in axes['pair_classifier_kinds'] else None)
    order_key = _canonical_order(data['smiles'])
    folds = MA.scaffold_folds(data['reps'], data['y'], data['groups'])

    rows, errors = [], {}
    if verbose:
        print(f'    {len(pairs["i"])} pairs over '
              f'{pairs["n_molecules_paired"]} molecules', flush=True)
    for fold_i, train_idx, val_idx in folds:
        got, bad = _rows_for_fold(name, fold_i, train_idx, val_idx, data,
                                  pairs, blocks, axes, reps, order_key,
                                  verbose)
        rows.extend(got)
        errors.update(bad)

    return {'target': name, 'n_molecules': len(data['smiles']),
            'n_pairs': int(len(pairs['i'])),
            'n_molecules_paired': int(pairs['n_molecules_paired']),
            'representations': reps, 'rows': rows, 'errors': errors,
            'seconds': round(time.time() - started, 1)}


def summarise(results):
    """One record per classifier and held-out group, with its per-target spread.

    No average over targets and no count of targets. Each number is a per-target median, so
    the lowest is the worst target that classifier faced and the highest is the best, which is
    what her instruction of 2026-11-28 asks for: "I would only in practice use the BEST one".
    """
    per_target = {}
    for res in results:
        for row in res.get('rows') or []:
            if row.get('matthews_correlation') is None:
                continue
            key = (row['classifier_kind'], row['model'],
                   row['representation'], row['pair_input'],
                   row['potency_cutoff_log_units'], row['label_rule'],
                   row['held_out_group'])
            per_target.setdefault(key, {}).setdefault(
                row['target'], []).append(row['matthews_correlation'])

    out = []
    for key, by_target in per_target.items():
        medians = sorted(float(np.median(v)) for v in by_target.values())
        out.append({
            'classifier_kind': key[0], 'model': key[1],
            'representation': key[2], 'pair_input': key[3],
            'potency_cutoff_log_units': key[4], 'label_rule': key[5],
            'held_out_group': key[6],
            'n_targets_scored': len(medians),
            'matthews_lowest_target': medians[0],
            'matthews_median_target': float(np.median(medians)),
            'matthews_highest_target': medians[-1]})
    out.sort(key=lambda r: -r['matthews_median_target'])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--targets', default=None,
                    help='comma-separated target names; the default is every '
                         'csv in the data directory')
    ap.add_argument('--representations', default=None,
                    help='comma-separated; the default is every one in the npz')
    ap.add_argument('--kinds', default=None,
                    help='comma-separated classifier kinds, to narrow a '
                         'session below what the settings file covers')
    ap.add_argument('--models', default=None,
                    help='comma-separated registry models, same purpose')
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--rep-dir', default=None)
    ap.add_argument('--output', default=str(OUT_DEFAULT))
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args(argv)

    data_dir = Path(a.data_dir or MA.CSV_DIR)
    rep_dir = Path(a.rep_dir or MA.REP_DIR)

    try:
        axes = _settings()
    except S.Unset as exc:
        print(f'REFUSED: {exc}')
        return 2
    if a.kinds:
        axes['pair_classifier_kinds'] = [k.strip() for k in a.kinds.split(',')]
    if a.models:
        axes['pair_vector_models'] = [m.strip() for m in a.models.split(',')]

    names = ([t.strip() for t in a.targets.split(',')] if a.targets
             else sorted(p.stem for p in data_dir.glob('CHEMBL*.csv')))
    if not names:
        print(f'REFUSED: no CHEMBL*.csv under {data_dir}')
        return 2
    reps_wanted = ([r.strip() for r in a.representations.split(',')]
                   if a.representations else None)

    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / 'cliff_classifier_sweep.partial.jsonl'

    print(f'{len(names)} target(s), classifiers '
          f'{axes["pair_classifier_kinds"]}, cutoffs '
          f'{axes["pair_potency_cutoffs"]} log units, label rules '
          f'{axes["pair_label_rules"]}', flush=True)

    # Resume. A target already written to the partial file by an earlier session is read back
    # and not re-run, so a sweep that does not fit one four-hour session continues in the
    # next. Delete the partial file to start over.
    done = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            if not line.strip():
                continue
            try:
                got = json.loads(line)
            except json.JSONDecodeError:
                continue
            if got.get('rows') or got.get('refused'):
                done[got.get('target')] = got
        if done:
            print(f'  resuming: {len(done)} target(s) already in '
                  f'{partial}', flush=True)

    started, results = time.time(), []
    for position, name in enumerate(names, start=1):
        if name in done:
            results.append(done[name])
            print(f'  [{position}/{len(names)}] {name}: already done',
                  flush=True)
            continue
        print(f'  [{position}/{len(names)}] {name}', flush=True)
        try:
            got = run_target(name, data_dir, rep_dir, axes, reps_wanted,
                             verbose=not a.quiet)
        except Exception as exc:
            got = {'target': name,
                   'refused': f'{type(exc).__name__}: {exc}', 'rows': []}
        results.append(got)
        with partial.open('a') as handle:
            handle.write(json.dumps(got) + '\n')

    payload = {
        'what_it_is': ('her four is-this-a-cliff classifiers, scored on '
                       'held-out pairs grouped by how many of a pair\'s two '
                       'molecules were training molecules'),
        'her_words': ("lets test these. That's what the next interactive "
                      'session shuold test, try differnet models (ideally '
                      'using the onfiugrations mentioned in papers), use the '
                      'respective datasets. This is its own sweep.'),
        'metric': 'Matthews correlation, -1 to 1, 0 is chance',
        'leakage_note': ('only the neither_seen group has no molecule shared '
                         'between a training pair and a held-out pair'),
        'axes': axes,
        'n_targets': len(results),
        'seconds': round(time.time() - started, 1),
        'per_target': results,
        'summary': summarise(results),
    }
    out = out_dir / 'cliff_classifier_sweep.json'
    out.write_text(json.dumps(payload, indent=2))

    print(f'\n{len(results)} target(s), {payload["seconds"]}s -> {out}')
    print('\nBest 15 by the median over targets, on pairs where neither '
          'molecule was seen in training.\nEach number is a Matthews '
          'correlation; 0 is chance.\n')
    shown = [r for r in payload['summary']
             if r['held_out_group'] == 'neither_seen'][:15]
    print(f'{"kind":14s}{"model":12s}{"representation":16s}{"input":12s}'
          f'{"cut":>5s}{"lowest":>9s}{"median":>9s}{"highest":>9s}')
    for r in shown:
        print(f'{r["classifier_kind"]:14s}{str(r["model"]):12s}'
              f'{str(r["representation"])[:15]:16s}'
              f'{str(r["pair_input"]):12s}'
              f'{str(r["potency_cutoff_log_units"]):>5s}'
              f'{r["matthews_lowest_target"]:9.4f}'
              f'{r["matthews_median_target"]:9.4f}'
              f'{r["matthews_highest_target"]:9.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
