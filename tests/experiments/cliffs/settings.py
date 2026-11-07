#!/usr/bin/env python3
"""The seven settings of the activity-cliff study, as inputs with no default.

WHAT THIS FILE IS FOR. Adelaide's outline of 2026-11-04 names seven settings
TBD1 to TBD7 and says each is hers to fix. This module is where they enter the
code: one named input each, read from a JSON file she edits, with no value
written here. Nothing in the suite picks one.

HER SEVEN, quoted from her outline, each with the input it became:

  TBD1  "look at the coefficients used in my similarity research, which show
        the most promise? And how can I justify that in the paper 3 outline?
        It should be a set."                               -> coefficients
  TBD2  "which representations work best for similarity? Again I have the
        results. And for the love of god HYBRIDS this was WHY I DESIGNED THE
        HYBRIDS. It should be a set."                      -> representations
  TBD3  "I don't know what cutoff works best, and if that cutoff is
        per-dataset or global (or maybe something like per-domain)."
                                      -> similarity_cutoffs, cutoff_scope
  TBD4  the same question for MoleculeACE's 0.90   -> moleculeace_cutoffs
  TBD5  "will definitely need a separate cutoff if I'm only trying to use one
        instead of 3 (hybrid overlap instead of a check of 3 overlaps)"
                                                 -> hybrid_overlap_cutoffs
  TBD6  "same with the cutoff question", on the share of molecules the moved
        gap aims at                                 -> target_cliff_shares
  TBD7  "is this a reasonable task? Is this straightforward?", on training her
        own model on MoleculeACE       -> train_cliff_model_on_moleculeace

THREE OF HER OWN OPEN ITEMS ARE HERE TOO, because a run needs a value for
each:

  definition_scope      her question of 2026-11-04: "So doesn't it just need
                        to be recomputed per fold? So there isn't a dataset
                        wide activity difference threshold?"
  held_out_cliff_rule   her DECIDE item, THESIS_OUTLINE.md 2274: "the held-out
                        cliff rule. Keep it and state the two shares, which
                        changes text only. Or apply the training rule to
                        held-out molecules, which needs s04 and s12 re-run."
  hybrid_feature_kinds  which restricted hybrid a coefficient reads: the 0/1
                        representations only, or those and the count ones.

HOW A SETTING IS READ. `"all"` runs every candidate the code can run, which is
not a choice and is the value the file ships with on every axis where running
everything covers the question. `null` is unset: the unit refuses to start and
says which setting it is waiting for. A list runs exactly those values.

Every candidate list below is derived, not typed: the coefficients come from
cliff_rules.COEFFICIENT_ROUTES, the representations from the dataset's own pool
in DATASET_REPS plus the two restricted hybrids, and the cutoff grids from
cliff_census's own sweep so a point here is a point there.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent.parent
REPO = TESTS.parent
for _p in (REPO / 'src', TESTS, TESTS / 'experiments'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.noise import cliff_rules as cr              # noqa: E402

#: The file she edits. KIRBY_CLIFF_SETTINGS overrides it, which is how a
#: papers run keeps its own copy beside RUN_SETTINGS.json.
SETTINGS_FILE = HERE / 'CLIFF_SETTINGS.json'

ALL = 'all'

#: The two restricted hybrids, by the name a criterion reaches them under.
HYBRID_REPS = ('hybrid_bits', 'hybrid_counts')

#: The cutoff grid for a coefficient whose scale is bounded. The same points
#: cliff_census.similarity_grid sweeps, so a count here and a count there are
#: the same row.
BOUNDED_CUTOFFS = tuple(round(0.30 + 0.05 * i, 2) for i in range(14))

#: The share-of-pairs grid, for a coefficient whose cutoff is not bounded and
#: for her "top percent of fixed similarity scores". cliff_census's own list.
PAIR_SHARES = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)

#: MoleculeACE's cutoff and the points around it that cliff_census sweeps.
#: 0.90 is theirs; the rest are the sweep's.
MOLECULEACE_CUTOFFS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)

#: Her two target shares question. cliff_census.gap_grid's own list, which
#: holds the 15 per cent her thesis used.
#:
#: Extended to 45 per cent by Adelaide, 2026-11-07, her word: "45". The
#: 2026-11-06 run over MoleculeACE's 30 targets was still climbing at 30, the
#: top of the old grid, and their own flag marks 18,831 of 48,714 molecules,
#: which is 38.7 per cent. 35, 40 and 45 put that rate inside the grid with two
#: points above it, which is the smallest grid that can tell a peak from a climb
#: that does not stop.
TARGET_SHARES = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0)

#: TBD3's second half. Which of the three a result supports is read off the
#: per-dataset numbers, so every cutoff runs on every dataset whichever of
#: these she picks; the value is recorded and does not change what runs.
CUTOFF_SCOPES = ('global', 'per_domain', 'per_dataset')

#: Her question of 2026-11-04. 'per_fold' computes the whole definition --
#: the gap's median absolute deviation and the pairing -- over the fold's
#: training molecules only (cliff_rules.subset_context). 'whole_dataset' is
#: what her thesis runs did.
DEFINITION_SCOPES = ('per_fold', 'whole_dataset')

#: Her DECIDE item. 'two_similar_training_molecules' is the rule her s04 and
#: s12 runs used on held-out molecules and is looser than the training rule,
#: which is why the two shares differ (15 per cent against about 40).
#: 'same_as_training' applies the training rule to held-out molecules.
HELD_OUT_CLIFF_RULES = ('two_similar_training_molecules', 'same_as_training')

#: Which restricted hybrid a coefficient reads (cliffs/binary_hybrid.py).
HYBRID_FEATURE_KINDS = ('bits', 'bits+counts')

#: One entry per setting: her TBD number or open item, what it means, the
#: candidates `"all"` expands to, and whether `"all"` is honest on that axis.
#: `covers_the_question` False means running every candidate does not answer
#: it, so the file ships with null and a unit that needs it refuses.
SETTINGS = {
    'coefficients': {
        'her_label': 'TBD1',
        'what': 'the coefficients that decide which pairs are similar',
        'candidates': tuple(sorted(cr.COEFFICIENT_ROUTES)),
        'covers_the_question': True,
    },
    'representations': {
        'her_label': 'TBD2',
        'what': 'the representations a coefficient is computed on; resolved '
                "per dataset to its own pool plus the two hybrids",
        'candidates': None,        # per dataset: see representations_for
        'covers_the_question': True,
    },
    'similarity_cutoffs': {
        'her_label': 'TBD3',
        'what': 'the cutoff a bounded coefficient is thresholded at',
        'candidates': BOUNDED_CUTOFFS,
        'covers_the_question': True,
    },
    'pair_shares': {
        'her_label': 'TBD3, second form',
        'what': 'the share of pairs called similar, which is her "top percent '
                'of fixed similarity scores" and the only comparable form of '
                'an unbounded coefficient',
        'candidates': PAIR_SHARES,
        'covers_the_question': True,
    },
    'cutoff_scope': {
        'her_label': 'TBD3 and TBD4',
        'what': 'whether one cutoff is global, per domain or per dataset. '
                'Recorded only: every cutoff runs on every dataset either '
                'way, and which of the three the numbers support is read off '
                'the result',
        'candidates': CUTOFF_SCOPES,
        'covers_the_question': True,
    },
    'moleculeace_cutoffs': {
        'her_label': 'TBD4',
        'what': "the cutoff on MoleculeACE's any-of-three rule, theirs being "
                '0.90',
        'candidates': MOLECULEACE_CUTOFFS,
        'covers_the_question': True,
    },
    'hybrid_overlap_cutoffs': {
        'her_label': 'TBD5',
        'what': 'the cutoff on one hybrid overlap used in place of their '
                'three scores',
        'candidates': BOUNDED_CUTOFFS,
        'covers_the_question': True,
    },
    'target_cliff_shares': {
        'her_label': 'TBD6',
        'what': 'the share of molecules the moved gap aims at, in per cent',
        'candidates': TARGET_SHARES,
        'covers_the_question': True,
    },
    'train_cliff_model_on_moleculeace': {
        'her_label': 'TBD7',
        'what': "whether a model is trained on MoleculeACE's own cliff flag",
        'candidates': (True, False),
        'covers_the_question': False,
    },
    'outlier_detection_methods': {
        'her_label': 'her M',
        'what': 'her words: "Look at the best outlier detection methods, call '
                'one M". The detector that produces the list of outliers. It '
                'is one name, taken from the noise experiments\' own results, '
                'so there is no list that covers the question',
        'candidates': None,         # checked against the registry at run time
        'covers_the_question': False,
    },
    'label_handling_methods': {
        'her_label': 'her L',
        'what': 'her words: "and then subsequent best label handling methods, '
                'call one L". The cleanup applied with and without '
                'AC-awareness. One name, from the noise results',
        'candidates': None,         # checked against the registry at run time
        'covers_the_question': False,
    },
    'outlier_shares': {
        'her_label': 'no quote recorded',
        'what': 'the share of training molecules M treats as outliers. '
                "n07.FLAG_SHARES, which every flag-reading cleanup in the "
                'noise suite already sweeps',
        'candidates': (0.05, 0.10, 0.20),
        'covers_the_question': True,
    },
    'definition_scope': {
        'her_label': 'her question of 2026-11-04',
        'what': 'whether the whole definition is recomputed on each fold\'s '
                'training molecules',
        'candidates': DEFINITION_SCOPES,
        'covers_the_question': True,
    },
    'held_out_cliff_rule': {
        'her_label': 'her DECIDE item, THESIS_OUTLINE.md 2274',
        'what': 'how a held-out molecule is called a cliff molecule',
        'candidates': HELD_OUT_CLIFF_RULES,
        'covers_the_question': True,
    },
    'hybrid_feature_kinds': {
        'her_label': 'no quote recorded',
        'what': 'which restricted hybrid a coefficient reads',
        'candidates': HYBRID_FEATURE_KINDS,
        'covers_the_question': True,
    },
}


def default_file():
    """The settings file's contents as it ships: every axis `"all"` where
    running every candidate covers the question, null where it does not."""
    out = {
        '_what': (
            "The seven settings of the activity-cliff study. Each one is "
            "Adelaide's. 'all' runs every candidate the code can run; a list "
            "runs those values; null is unset and the unit refuses to start "
            "and says which setting it waited for."),
        '_candidates': {k: (list(v['candidates'])
                            if v['candidates'] is not None
                            else "per dataset: its own pool plus hybrid_bits "
                                 "and hybrid_counts")
                        for k, v in SETTINGS.items()},
    }
    for name, spec in SETTINGS.items():
        out[name] = ALL if spec['covers_the_question'] else None
    return out


def path():
    raw = os.environ.get('KIRBY_CLIFF_SETTINGS')
    return Path(raw) if raw else SETTINGS_FILE


def read(where=None):
    """The settings as stored. A missing file is every axis unset, not a set
    of defaults: a default here would be a decision."""
    p = Path(where) if where else path()
    if not p.exists():
        return {name: None for name in SETTINGS}
    return json.loads(p.read_text())


def write_default(where=None, force=False):
    p = Path(where) if where else path()
    if p.exists() and not force:
        return p, False
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(default_file(), indent=1) + '\n')
    return p, True


class Unset(RuntimeError):
    """One setting has no value, so nothing that depends on it may run."""


def values(name, stored=None):
    """The list of values to run for one setting.

    Raises Unset, naming the setting and the file, when it has no value. The
    message is the whole reason a unit did not start, so it says what the
    setting is and what her own words about it were.
    """
    if name not in SETTINGS:
        raise KeyError(f'no such setting {name!r}; they are {sorted(SETTINGS)}')
    spec = SETTINGS[name]
    stored = read() if stored is None else stored
    got = stored.get(name)
    if got is None:
        raise Unset(
            f"the activity-cliff study's setting {name!r} has no value. It is "
            f"{spec['her_label']}: {spec['what']}. Write it in {path()} and "
            f"this runs by itself at the next claim."
            + ('' if spec['candidates'] is None else
               f" The values the code can run are {list(spec['candidates'])}."))
    if got == ALL:
        if spec['candidates'] is None:
            return ALL                  # resolved per dataset
        return list(spec['candidates'])
    if not isinstance(got, list):
        got = [got]
    if spec['candidates'] is not None:
        bad = [v for v in got if v not in spec['candidates']]
        if bad:
            raise ValueError(
                f"{path()}: setting {name!r} names {bad}, which this code "
                f"cannot run. The values it can run are "
                f"{list(spec['candidates'])}.")
    return got


def one(name, stored=None):
    """A setting that takes one value, not a list. Raises when it holds
    several, because a run cannot carry two scopes at once."""
    got = values(name, stored)
    if got == ALL or len(got) != 1:
        raise ValueError(
            f"setting {name!r} takes one value and holds "
            f"{got if got != ALL else ALL!r}. Write one value in {path()}.")
    return got[0]


def unset(stored=None):
    """Which settings have no value. One line per setting for the reply that
    says what a held unit is waiting for."""
    stored = read() if stored is None else stored
    return [name for name in SETTINGS if stored.get(name) is None]


def representations_for(dataset_name, reps_loaded, stored=None):
    """TBD2 resolved for one dataset: the representation names a coefficient
    may be computed on.

    `"all"` is the dataset's own loaded pool plus whichever of the two
    restricted hybrids was built. A named list is checked against what the
    dataset has, and a name it does not have is an error rather than a silent
    skip, because a missing representation changes which coefficients ran.
    """
    have = list(reps_loaded)
    got = values('representations', stored)
    if got == ALL:
        return have
    missing = [r for r in got if r not in have]
    if missing:
        raise ValueError(
            f"{dataset_name}: setting 'representations' names {missing}, and "
            f"this dataset has {sorted(have)}. Either its pool in "
            f"DATASET_REPS does not hold them, or the restricted hybrid was "
            f"not built (cliffs/binary_hybrid.py says why).")
    return got


def methods_for(name, known, stored=None):
    """One of the two method settings, checked against the live registry.

    `known` is the names the code can actually run, read from the registry at
    run time rather than copied here, so a detector added to the noise suite is
    available the moment it is added.
    """
    got = values(name, stored)
    if got == ALL:
        raise ValueError(
            f"setting {name!r} cannot be 'all'. Her words name ONE: "
            f"{SETTINGS[name]['what']}. Write the name(s) in {path()}; the "
            f"code can run {sorted(known)}.")
    bad = [v for v in got if v not in known]
    if bad:
        raise ValueError(
            f"{path()}: setting {name!r} names {bad}, which this code cannot "
            f"run. It can run {sorted(known)}.")
    return got


def record(stored=None):
    """What every result file carries, so a count can be read back against
    the settings it was computed under."""
    stored = read() if stored is None else stored
    return {
        'settings_file': str(path()),
        'stored': {name: stored.get(name) for name in SETTINGS},
        'unset': unset(stored),
        'her_labels': {name: SETTINGS[name]['her_label'] for name in SETTINGS},
    }


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write-default', action='store_true',
                    help='write the settings file with every axis at its '
                         'shipped value, if it does not exist')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    if a.write_default:
        p, made = write_default(force=a.force)
        print(f"{'wrote' if made else 'already there'} {p}")
    stored = read()
    print(f'\n{path()}')
    for name, spec in SETTINGS.items():
        got = stored.get(name)
        shown = 'NOT SET' if got is None else (
            f'all ({len(spec["candidates"])} values)'
            if got == ALL and spec['candidates'] is not None
            else str(got))
        print(f"  {name:36s} {spec['her_label']:34s} {shown}")
    miss = unset(stored)
    print(f"\n{len(miss)} setting(s) with no value: {miss or 'none'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
