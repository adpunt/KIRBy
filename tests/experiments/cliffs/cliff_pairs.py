#!/usr/bin/env python3
"""Pairs of molecules that differ by one chemical change, and their cliff label.

WHY THIS FILE EXISTS. Her instruction of 2026-11-28: "so I'm exploring different ways to
have a AC or not classifier. I can haveg multiple, that's totally fine, but I need them
clearly defined", and then "lets test these". Every published cliff classifier labels a PAIR
of molecules, not one molecule. Nothing in this repository built pairs before this file.

WHERE THE PAIRS COME FROM. RDKit's own `rdMMPA.FragmentMol` with one cut, so no new
dependency: `mmpdb` is not installed on the laptop or the server, checked 2026-11-28. One cut
splits a molecule into two fragments. The larger fragment is taken as the core and the smaller
as the substituent, which is the usual reading and is not stated by any source read. Two
molecules that produce the same core form a pair, and the pair's two substituents are what
differ.

THE CLIFF LABEL ON A PAIR, and why there are two rules. The sources disagree on the potency
difference that makes a pair a cliff: MoleculeACE uses a 10-fold difference, which is 1 log
unit; the matched molecular pair literature and the 2025 kinase paper use 2 log units. Both
run. The second disagreement is what counts as a non-cliff, and both of those run too:

  threshold         every pair is labelled. A pair at or above the cutoff is a cliff and
                    every other pair is a non-cliff.
  excluded_middle   a pair at or above the cutoff is a cliff, a pair at or below 1 log unit
                    is a non-cliff, and a pair between the two is dropped. This is what a
                    re-implementation of Dablander et al. does.

THE SPLIT, which is where the published numbers went wrong. Leakage here is one molecule
appearing in a training pair and in a test pair: a model then scores well by recognising the
molecule. The split is therefore made on MOLECULES first, by Bemis-Murcko scaffold, which is
this lab's own splitter. Each pair is then placed by how many of its two molecules are in the
fold's training molecules, and the three groups are scored separately:

  both_seen       both molecules were in training. Published work that split pairs at random
                  scored mostly on this group, which is why its accuracies reached 80 to 90
                  per cent.
  one_seen        one molecule was in training.
  neither_seen    neither molecule was in training. The hardest group and the only one with
                  no leakage at all.

Training uses the pairs whose BOTH molecules are training molecules, so no test molecule is
ever seen while fitting.

Sources for all of the above: thesis_provenance/LITERATURE_SEARCH_TBD7_2026-11-28.md.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent.parent
REPO = TESTS.parent
for _p in (REPO / 'src', TESTS, TESTS / 'experiments'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

#: The two potency differences the sources disagree on, in log units.
POTENCY_CUTOFFS = (1.0, 2.0)

#: The two readings of what a non-cliff pair is.
LABEL_RULES = ('threshold', 'excluded_middle')

#: The upper edge of the non-cliff band under `excluded_middle`, in log units.
NONCLIFF_CEILING = 1.0

#: A pair is kept only if the substituent that differs is at most this share of the whole
#: molecule's heavy atoms. Without a limit, one cut of a small molecule pairs two molecules
#: that share almost nothing. Not a published number; recorded so it can be changed.
MAX_SUBSTITUENT_SHARE = 0.5


def _fragment(smiles):
    """One molecule's (core, substituent) readings under a single cut.

    Returns a list of (core_smiles, substituent_smiles). An unparseable SMILES returns an
    empty list rather than raising, so one bad molecule does not stop a target.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMMPA

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy < 4:
        return []
    try:
        cuts = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False)
    except Exception:
        return []

    out = []
    for first, second in cuts:
        text = second if second else first
        if not text or '.' not in text:
            continue
        left, right = text.split('.', 1)
        left_mol = Chem.MolFromSmiles(left)
        right_mol = Chem.MolFromSmiles(right)
        if left_mol is None or right_mol is None:
            continue
        if left_mol.GetNumHeavyAtoms() >= right_mol.GetNumHeavyAtoms():
            core, sub, sub_mol = left, right, right_mol
        else:
            core, sub, sub_mol = right, left, left_mol
        share = sub_mol.GetNumHeavyAtoms() / max(n_heavy, 1)
        if share > MAX_SUBSTITUENT_SHARE:
            continue
        out.append((core, sub))
    return out


def build_pairs(smiles, y):
    """Every pair of molecules sharing a core, with the two substituents that differ.

    Args:
        smiles: one SMILES per molecule, in the dataset's own order.
        y: the potency of each molecule, in the dataset's own log units.

    Returns a dict with
        i, j                 the two molecules' indices into `smiles`, as arrays
        core                 the shared core's SMILES, one per pair
        sub_i, sub_j         the two substituents' SMILES, one each per pair
        delta                abs(y[i] - y[j]), one per pair, in label units
        n_molecules_paired   how many molecules appear in at least one pair
    """
    y = np.asarray(y, dtype=float)
    by_core = defaultdict(dict)
    for idx, smi in enumerate(smiles):
        for core, sub in _fragment(smi):
            # One molecule can reach the same core by two cuts; keep one reading.
            by_core[core].setdefault(idx, sub)

    i_list, j_list, cores, subs_i, subs_j = [], [], [], [], []
    seen = set()
    for core, members in by_core.items():
        if len(members) < 2:
            continue
        idxs = sorted(members)
        for a_pos, a in enumerate(idxs):
            for b in idxs[a_pos + 1:]:
                if members[a] == members[b]:
                    continue        # same substituent: not a chemical change
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                i_list.append(a)
                j_list.append(b)
                cores.append(core)
                subs_i.append(members[a])
                subs_j.append(members[b])

    i_arr = np.asarray(i_list, dtype=int)
    j_arr = np.asarray(j_list, dtype=int)
    delta = (np.abs(y[i_arr] - y[j_arr]) if len(i_arr)
             else np.zeros(0, dtype=float))
    return {'i': i_arr, 'j': j_arr, 'core': cores,
            'sub_i': subs_i, 'sub_j': subs_j, 'delta': delta,
            'n_molecules_paired': len(set(i_list) | set(j_list))}


def pair_labels(delta, cutoff, rule):
    """The cliff label of each pair, and which pairs the rule drops.

    Returns (label, kept) where `label` is 1 for a cliff pair and 0 for a non-cliff pair,
    and `kept` is the boolean mask of pairs the rule labels at all.
    """
    delta = np.asarray(delta, dtype=float)
    if rule not in LABEL_RULES:
        raise ValueError(f'no such label rule {rule!r}; they are {LABEL_RULES}')
    cliff = delta >= cutoff
    if rule == 'threshold':
        return cliff.astype(int), np.ones(len(delta), dtype=bool)
    noncliff = delta <= NONCLIFF_CEILING
    kept = cliff | noncliff
    return cliff.astype(int), kept


def pair_groups(pairs, train_molecules, n_molecules):
    """Each pair placed by how many of its two molecules are training molecules.

    Returns a dict of boolean masks over the pairs: `both_seen`, `one_seen`,
    `neither_seen`. The three are disjoint and cover every pair.
    """
    in_train = np.zeros(n_molecules, dtype=bool)
    in_train[np.asarray(train_molecules, dtype=int)] = True
    n_seen = in_train[pairs['i']].astype(int) + in_train[pairs['j']].astype(int)
    return {'both_seen': n_seen == 2,
            'one_seen': n_seen == 1,
            'neither_seen': n_seen == 0}
