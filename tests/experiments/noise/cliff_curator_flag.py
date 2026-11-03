#!/usr/bin/env python3
"""Do the molecules a cliff rule calls cliffs carry a curator's suspect mark?

Paper 3, activity cliffs. Fits no model, uses no split, trains no cleanup.

THE QUESTION. A cliff and a wrong label look the same from the outside: both
are a molecule whose neighbours do not predict its label. ChEMBL's own
curators mark records they found suspect, in a field called
`data_validity_comment`. That mark is the one list of doubtful labels here
that a person made rather than a script. So for each rule that says what a
cliff is, how many of the molecules it calls cliffs are molecules a curator
already doubted?

WHAT AN ANSWER MEANS, IN BOTH DIRECTIONS. Neither direction is a verdict on a
rule, and the file says so where it prints the numbers.

  * A cliff set with MORE flagged molecules than the dataset as a whole is a
    rule that partly selects records a curator doubted. Some of what it calls
    chemistry is a label somebody already queried.
  * A cliff set with NO more than the dataset as a whole is a rule whose
    cliffs are not the records a curator doubted. That is the result this
    measurement exists to be able to state.

THE LIMIT THAT CANNOT BE REMOVED, from n22's own write-up. The flag has two
texts, and they are not the same kind of evidence:

    Outside typical range        assigned because the VALUE is extreme
    Potential transcription error assigned because the value looks mistyped

"Outside typical range" is a property of the label. A cliff rule calls a pair
a cliff when its labels sit far apart, so part of any overlap with that text is
the two definitions reading the same extreme label, not independent agreement.
The two texts are therefore counted separately as well as together. The counts
for "potential transcription error" are small, and small counts are printed
with the count beside them rather than as a share on its own.

ONE ROW OF THE OUTPUT IS ONE RULE. One molecule is the unit, not one pair and
not one ChEMBL record, because a cliff pair of two rows of the same molecule is
not a cliff and the rules drop those pairs themselves.

WHAT ONE MOLECULE'S LABEL IS. The per-record file holds one row per
measurement, so a molecule measured four times has four rows. Here each
molecule is one row and its label is the MEAN of its measurements in pIC50.
That choice changes the gap between two molecules and therefore which pairs
are cliffs; it is the same choice n29 calls its `mean_label` arm. A molecule
is flagged if ANY of its records carries a comment, which is how n22 counts a
molecule as flagged.

DATASET. hERG IC50 from ChEMBL (CHEMBL240), the one dataset in Paper 3 that
keeps one row per measurement. Every other dataset here holds one averaged
number per molecule, and the curator comment does not survive averaging, so
this measurement exists on this dataset only. That is a property of the files,
not a choice made here.

RUN
    python tests/experiments/noise/cliff_curator_flag.py
    python tests/experiments/noise/cliff_curator_flag.py --rules n01,moleculeace

The per-record pull has to exist. If it does not, fetch it with
    python tests/experiments/noise/_probe_chembl_herg.py --standard-type IC50
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
for _p in (os.path.join(_ROOT, 'src'), os.path.join(_ROOT, 'tests'),
           os.path.join(_ROOT, 'tests', 'experiments')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.noise import cliff_rules as cr  # noqa: E402

DATASET = 'herg_ic50'
DEFAULT_OUTPUT = '/Volumes/seagate/cliff_curator_flag'


def load_molecules():
    """One row per molecule: mean pIC50, and whether any record was flagged.

    Returns a dict with smiles, labels, curator_flag, per-text flags and the
    counts the printout needs.
    """
    from kirby.datasets.herg import load_herg_ic50_measurements

    d = load_herg_ic50_measurements()
    if not d.get('curator_flag_available'):
        raise SystemExit(
            "the cached hERG IC50 pull carries no data_validity_comment "
            "column, so there is no curator mark to count. Re-fetch with "
            "_probe_chembl_herg.py --standard-type IC50.")

    smiles = np.asarray(d['smiles'], dtype=object)
    labels = np.asarray(d['labels'], dtype=float)
    groups = np.asarray(d['replicate_groups'])
    flag = np.asarray(d['curator_flag'], dtype=bool)
    text = np.asarray(['' if t is None else str(t)
                       for t in d['curator_flag_text']], dtype=object)

    order = np.unique(groups)
    mol_smiles, mol_labels, mol_flag = [], [], []
    mol_texts, mol_n_records = [], []
    for gid in order:
        rows = np.flatnonzero(groups == gid)
        mol_smiles.append(str(smiles[rows[0]]))
        mol_labels.append(float(labels[rows].mean()))
        mol_flag.append(bool(flag[rows].any()))
        mol_texts.append(sorted({t for t in text[rows] if t}))
        mol_n_records.append(int(len(rows)))

    mol_flag = np.asarray(mol_flag, dtype=bool)
    texts = sorted({t for ts in mol_texts for t in ts})
    by_text = {t: np.asarray([t in ts for ts in mol_texts], dtype=bool)
               for t in texts}

    return {
        'smiles': mol_smiles,
        'labels': np.asarray(mol_labels, dtype=float),
        'curator_flag': mol_flag,
        'flag_by_text': by_text,
        'n_records': np.asarray(mol_n_records, dtype=int),
        'n_records_total': int(len(labels)),
        'n_records_flagged': int(flag.sum()),
        'record_text_counts': {t: int((text == t).sum()) for t in texts},
    }


def cross(cliff, flag):
    """The four counts, and the two shares they are shares of.

    `cliff` and `flag` are both one entry per molecule. Returns counts named
    for what they count, so no number in the output file is a bare fraction.
    """
    cliff = np.asarray(cliff, dtype=bool)
    flag = np.asarray(flag, dtype=bool)
    n = len(cliff)
    both = int((cliff & flag).sum())
    cliff_only = int((cliff & ~flag).sum())
    flag_only = int((~cliff & flag).sum())
    neither = int((~cliff & ~flag).sum())
    n_cliff = both + cliff_only
    n_flag = both + flag_only
    return {
        'n_molecules': n,
        'n_cliff_molecules': n_cliff,
        'n_flagged_molecules': n_flag,
        'n_cliff_and_flagged': both,
        'n_cliff_not_flagged': cliff_only,
        'n_flagged_not_cliff': flag_only,
        'n_neither': neither,
        # Of the molecules this rule calls cliffs, the share carrying a mark.
        'pct_of_cliff_molecules_flagged': (100.0 * both / n_cliff
                                           if n_cliff else None),
        # Of every molecule in the dataset, the share carrying a mark. This is
        # what the line above has to beat for the rule to be selecting marked
        # records at all.
        'pct_of_all_molecules_flagged': 100.0 * n_flag / n if n else None,
        # Of the marked molecules, the share this rule calls cliffs.
        'pct_of_flagged_molecules_called_cliff': (100.0 * both / n_flag
                                                  if n_flag else None),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--rules', default='',
                    help='comma-separated names from cliff_rules.RULES; '
                         'default every rule')
    ap.add_argument('--output', default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    t0 = time.time()
    mol = load_molecules()
    labels = mol['labels']
    flag = mol['curator_flag']
    n = len(labels)

    print(f"{DATASET}: {n} molecules, from {mol['n_records_total']} ChEMBL "
          f"records")
    print(f"  {mol['n_records_flagged']} of those {mol['n_records_total']} "
          f"records carry a curator comment, on {int(flag.sum())} of the "
          f"{n} molecules")
    for text, count in sorted(mol['record_text_counts'].items(),
                              key=lambda kv: -kv[1]):
        n_mol_text = int(mol['flag_by_text'][text].sum())
        print(f"    {count:5d} records, {n_mol_text:5d} molecules: {text}")
    print(f"  one molecule's label is the mean of its measurements in pIC50; "
          f"{int((mol['n_records'] > 1).sum())} molecules were measured more "
          f"than once")
    print()

    ctx = cr.CliffContext(labels, smiles=mol['smiles'], is_clf=False,
                          dataset_name=DATASET)

    names = ([s.strip() for s in args.rules.split(',') if s.strip()]
             or list(cr.RULES))
    rows, skipped = {}, {}
    for name in names:
        similarity, gap = cr.RULES[name]
        t1 = time.time()
        try:
            sets = cr.cliff_sets(ctx, similarity, gap)
        except Exception as exc:
            skipped[name] = f'{type(exc).__name__}: {exc}'
            print(f"  {name:24s} NOT RUN  {type(exc).__name__}: "
                  f"{str(exc)[:90]}")
            continue
        cliff = np.asarray(sets['molecule_mask'], dtype=bool)
        row = {
            'similarity': similarity, 'gap': gap,
            'n_cliff_pairs': sets['n_cliff_pairs'],
            'seconds': round(time.time() - t1, 1),
            'all_texts': cross(cliff, flag),
            'by_text': {t: cross(cliff, m)
                        for t, m in mol['flag_by_text'].items()},
        }
        rows[name] = row
        a = row['all_texts']
        base = a['pct_of_all_molecules_flagged']
        got = a['pct_of_cliff_molecules_flagged']
        got_s = 'n/a' if got is None else f'{got:5.2f}%'
        print(f"  {name:24s} {a['n_cliff_molecules']:6d} cliff molecules, "
              f"{a['n_cliff_and_flagged']:4d} of them marked -> {got_s} "
              f"against {base:5.2f}% of all {n} molecules "
              f"({row['seconds']}s)")

    print()
    print("One line above is one rule. 'cliff molecules' counts molecules the "
          "rule puts in at least one cliff pair. The share after the arrow is "
          "of those molecules; the share it is compared with is of every "
          "molecule in the dataset.")
    print("A rule whose share is no higher than the dataset's own share is "
          "not selecting records a curator doubted.")

    out = {
        'dataset': DATASET,
        'unit': 'one molecule',
        'label_per_molecule': 'mean of its measurements, pIC50',
        'n_molecules': n,
        'n_records_total': mol['n_records_total'],
        'n_records_flagged': mol['n_records_flagged'],
        'n_molecules_flagged': int(flag.sum()),
        'record_text_counts': mol['record_text_counts'],
        'n_molecules_measured_more_than_once':
            int((mol['n_records'] > 1).sum()),
        'rules': rows,
        'rules_not_run': skipped,
        'caveat_outside_typical_range': (
            'the text "Outside typical range" is assigned because the value '
            'is extreme, and a cliff rule calls a pair a cliff when its '
            'labels sit far apart, so any overlap with that text is partly '
            'the two definitions reading the same extreme label. The texts '
            'are counted separately in by_text for that reason.'),
        'seconds': round(time.time() - t0, 1),
    }
    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, f'{DATASET}.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"\nWritten to {path}  ({out['seconds']}s)")


if __name__ == '__main__':
    main()
