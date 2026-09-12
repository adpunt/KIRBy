"""Where n06's ground truth comes from.

Every other experiment in the noise suite injects the noise it then looks for.
n06 is the one that does not: the disagreement it scores against was made by
people measuring the same thing twice and getting different answers.

A GROUND-TRUTH PROVIDER answers two things about one dataset:

  * which molecules form a repeat group — the rows that are measurements of the
    same thing;
  * how much the repeats disagree — one number per group.

Everything downstream of that is the same whichever provider supplied it: the
same estimators, the same feature spaces, the same reduction to one row per
group. Three providers exist today.

| provider              | a group is                        | disagreement is      | statistic |
|-----------------------|-----------------------------------|----------------------|-----------|
| drug_target_repeats   | same drug and same protein        | spread of the pKd    | spearman  |
| cross_curator         | same InChIKey in two curated sets | the calls differ y/n | auroc     |
| chembl_multi_document | same compound and target, two papers | spread of the pIC50 | spearman |

A provider COUNTS BEFORE IT RUNS. It reports how many groups it found and how
many molecules those cover, and it refuses below MIN_GROUND_TRUTH_GROUPS — a
correlation over forty groups is not a result. A refusal is a fact about the
dataset and is written to the result file with its count, not swallowed.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataclasses import dataclass, field  # noqa: E402
from typing import Optional  # noqa: E402

import numpy as np  # noqa: E402


# A group needs at least this many measurements before it has a spread at all.
MIN_REPLICATES = 2

# Below this many groups the correlation is not reported. Forty groups is not a
# result; this is the one place the floor is written down.
MIN_GROUND_TRUTH_GROUPS = 50

# Where ChEMBL's continuous pKi is cut when it has to be compared with a
# curator's active/inactive call. The rate depends heavily on this number, so
# n06 also reports it at the two thresholds below and states all three.
CROSS_CURATOR_THRESHOLD = 5.0
CROSS_CURATOR_THRESHOLD_SENSITIVITY = (5.0, 6.0, 7.0)

# The datasets whose label is the same property Lhasa curated: whether a
# molecule blocks the hERG channel. Two curators only disagree about one thing
# if they measured the same thing. BindingDB shares 272 molecules with Lhasa's
# hERG files, and its label for each of them is a binding affinity against some
# other protein — cutting that at 5.0 and calling the difference a
# disagreement about hERG would have produced 176 flags that mean nothing.
CROSS_CURATOR_DATASETS = {'herg', 'herg_ki'}

# The spread of a group's labels is the population standard deviation, the
# quantity np.std returns. The sample standard deviation (ddof=1) is a
# different number on small groups: on BindingDB's 3,494 repeated pairs the
# median is 0.427 log units one way and 0.560 the other. Stated in the result
# file so the two are never compared without knowing which is which.
GROUP_SPREAD_DDOF = 0


def rows_by_group(group_ids):
    """group id -> the row numbers in it, built in one pass.

    The obvious way to ask this — one np.flatnonzero over the whole array per
    group — reads every row once for every group. BindingDB's cached file holds
    46,090 distinct drug-protein pairs over 52,272 rows, of which the loader
    hands an experiment 30,000, and n06 asks the question again for every
    (space, estimator) pair. That way costs thousands of millions of
    comparisons before a single estimator is fitted.
    """
    group_ids = np.asarray(group_ids)
    if len(group_ids) == 0:
        return {}
    order = np.argsort(group_ids, kind='stable')
    sorted_g = group_ids[order]
    cuts = np.flatnonzero(np.diff(sorted_g)) + 1
    starts = np.concatenate([[0], cuts])
    chunks = np.split(order, cuts)
    return {int(sorted_g[start]): chunk
            for start, chunk in zip(starts, chunks)}


@dataclass
class GroundTruth:
    """One provider's answer for one dataset.

    group_ids: one entry per row of the dataset, holding the group that row
        belongs to, or -1 for a row in no group.
    group_value: group id -> the ground-truth disagreement of that group.
    statistic: 'spearman' when the value is a magnitude, 'auroc' when it is a
        yes/no flag. The two are not interchangeable and n06 does not force a
        flag through the Spearman path.
    """
    provider: str
    statistic: str
    units: str
    group_ids: np.ndarray
    group_value: dict
    counts: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)
    _rows: dict = field(default=None, repr=False, compare=False)

    @property
    def rows(self):
        """group id -> its row numbers. Built once, then reused."""
        if self._rows is None:
            self._rows = rows_by_group(self.group_ids)
        return self._rows

    @property
    def n_groups(self):
        return len(self.group_value)

    @property
    def in_group(self):
        """One entry per row: is this row inside a group that has a value."""
        keep = np.zeros(len(self.group_ids), dtype=bool)
        for gid in self.group_value:
            rows = self.rows.get(int(gid))
            if rows is not None:
                keep[rows] = True
        return keep

    @property
    def n_rows_covered(self):
        return int(self.in_group.sum())

    def group_order(self):
        """The group ids, in one fixed order, and their values in that order."""
        gids = sorted(self.group_value)
        return gids, np.array([self.group_value[g] for g in gids], dtype=float)


class ProviderUnavailable(Exception):
    """Raised with a reason and whatever count was reached before refusing."""

    def __init__(self, reason, counts=None):
        super().__init__(reason)
        self.reason = reason
        self.counts = counts or {}


# ── provider 1: the same drug against the same protein, measured again ──────

def build_replicate_groups(data):
    """Per-(drug, protein) group ids from the loader's metadata.

    An explicitly-provided ``data['replicate_groups']`` always wins, so any
    loader that knows its own repeat structure works here with no change to
    n06. Otherwise the ids are built from ``metadata['drug_ids']`` and
    ``metadata['protein_ids']``.

    Returns None when neither is present — NOT ``data['groups']``, which is the
    cold-target CV grouping (one group per protein). Grouping by protein
    measures how far different drugs' affinities for one protein sit apart,
    which is biology, not assay disagreement. A 2026-05 version of n06 did
    exactly that and every number it produced was meaningless.
    """
    explicit = data.get('replicate_groups')
    if explicit is not None:
        return np.asarray(explicit)

    meta = data.get('metadata') or {}
    drug_ids = meta.get('drug_ids')
    protein_ids = meta.get('protein_ids')
    if drug_ids is None or protein_ids is None:
        return None

    n = len(data['labels'])
    if len(drug_ids) != n or len(protein_ids) != n:
        return None

    key_to_id = {}
    group_ids = np.empty(n, dtype=np.int64)
    for i, key in enumerate(f"{d}|{p}" for d, p in zip(drug_ids, protein_ids)):
        if key not in key_to_id:
            key_to_id[key] = len(key_to_id)
        group_ids[i] = key_to_id[key]
    return group_ids


def compute_replicate_stds(labels, groups, min_replicates=MIN_REPLICATES):
    """Per-ROW ground truth: the spread of the group each row sits in.

    n06 itself no longer uses this — it reduces to one point per group, since
    broadcasting one group's spread to every member counts the same
    ground-truth value once per repeat. n14 asks a different question of it:
    whether this dataset has any repeat structure at all, and over how many
    rows. Kept for that.

    Returns (per-row spread, per-row flag). The flag is True only where the
    group's repeats actually disagree; a group whose repeats are identical
    carries no information about how an estimator ranks noise.
    """
    labels = np.asarray(labels, dtype=np.float64)
    n = len(labels)
    stds = np.zeros(n, dtype=np.float64)
    has_replicate = np.zeros(n, dtype=bool)
    if groups is None:
        return stds, has_replicate

    for gid, rows in rows_by_group(np.asarray(groups)).items():
        if len(rows) < min_replicates:
            continue
        spread = float(np.std(labels[rows], ddof=GROUP_SPREAD_DDOF))
        stds[rows] = spread
        if spread > 0:
            has_replicate[rows] = True
    return stds, has_replicate


def drug_target_repeats(data, dataset_name, **kwargs):
    """Rows sharing a drug and a protein are repeat measurements of one pair.

    The spread of a group's labels is how far the labs disagreed about that
    pair, in the label's own units. A group whose repeats agree exactly carries
    no information about how an estimator RANKS noise, so it is counted and
    then left out of the correlation.
    """
    group_ids = build_replicate_groups(data)
    if group_ids is None:
        raise ProviderUnavailable(
            "no replicate grouping: the data dict exposed neither "
            "data['replicate_groups'] nor metadata with 'drug_ids' and "
            "'protein_ids', one per row",
            {'n_groups': 0, 'n_rows_covered': 0})

    labels = np.asarray(data['labels'], dtype=np.float64)
    # A loader may hand back its own grouping as strings. Every group id below
    # is used as a dictionary key and written to a result file, so they are
    # renumbered to integers here rather than raising on int('CHEMBL123').
    if group_ids.dtype.kind not in 'iu':
        _, group_ids = np.unique(np.asarray(group_ids, dtype=object),
                                 return_inverse=True)
        group_ids = group_ids.astype(np.int64)
    uniq, counts = np.unique(group_ids, return_counts=True)
    repeated = uniq[counts >= MIN_REPLICATES]

    values = {}
    n_zero_spread = 0
    n_not_finite = 0
    index = rows_by_group(group_ids)
    for gid in repeated:
        rows = index[int(gid)]
        vals = labels[rows]
        vals = vals[np.isfinite(vals)]
        if len(vals) < MIN_REPLICATES:
            # A group whose labels are not all numbers has no spread. Counted,
            # not silently filed as a group whose repeats agreed.
            n_not_finite += 1
            continue
        spread = float(np.std(vals, ddof=GROUP_SPREAD_DDOF))
        if spread > 0:
            values[int(gid)] = spread
        else:
            n_zero_spread += 1

    counts_out = {
        'n_rows': int(len(labels)),
        'n_distinct_pairs': int(len(uniq)),
        'n_pairs_measured_more_than_once': int(len(repeated)),
        'n_groups': len(values),
        'n_groups_whose_repeats_agree_exactly': n_zero_spread,
        'n_groups_with_fewer_than_two_finite_labels': n_not_finite,
        'n_rows_in_repeated_pairs': int(counts[counts >= MIN_REPLICATES].sum()),
    }
    if len(values) < MIN_GROUND_TRUTH_GROUPS:
        raise ProviderUnavailable(
            f"{len(values)} pairs are measured more than once and disagree, "
            f"below the floor of {MIN_GROUND_TRUTH_GROUPS}", counts_out)

    gt = GroundTruth(
        provider='drug_target_repeats',
        statistic='spearman',
        units=f"spread of the group's labels, "
              f"population standard deviation (ddof={GROUP_SPREAD_DDOF})",
        group_ids=np.asarray(group_ids),
        group_value=values,
        counts=counts_out,
        notes={'what_a_group_is': 'rows sharing a drug id and a protein id',
               'source': "data['replicate_groups'] if present, else "
                         "metadata drug_ids and protein_ids"},
    )
    counts_out['n_rows_covered'] = gt.n_rows_covered
    return gt


# ── provider 2: two organisations curated the same molecule ─────────────────

def dataset_smiles(data):
    """The molecule of every row, whatever the loader called it.

    A molecular dataset returns them under 'smiles'. A drug-target dataset
    returns one drug and one protein per row, and its molecules are under
    'drug_smiles' — so a provider that only looked for 'smiles' refused on
    BindingDB and recorded "this dataset exposes no SMILES", which is not true
    of the dataset.
    """
    for key in ('smiles', 'drug_smiles'):
        v = data.get(key)
        if v is not None and len(v) == len(data['labels']):
            return list(v), key
    return None, None


def _inchikeys(smiles_list):
    """One InChIKey per SMILES, None where rdkit could not build one.

    Built the same way the Lhasa loader builds its own keys — MolToInchi then
    InchiToInchiKey — so the two key sets can be matched at all.
    """
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    # One key per DISTINCT molecule, then handed back per row. A drug-target
    # dataset repeats the same drug on every protein it was measured against —
    # 30,000 rows over 7,255 drugs on BindingDB — and building an InChI is
    # milliseconds, so the difference is minutes.
    seen = {}
    out = []
    for smi in smiles_list:
        if smi in seen:
            out.append(seen[smi])
            continue
        mol = Chem.MolFromSmiles(smi) if smi else None
        key = None
        if mol is not None:
            try:
                inchi = MolToInchi(mol)
                key = InchiToInchiKey(inchi) if inchi else None
            except Exception:
                key = None
        seen[smi] = key
        out.append(key)
    return out


def _lhasa_calls():
    """Lhasa's hERG call per InChIKey, over both of its curated files.

    Returns ({inchikey: 0 or 1}, {inchikey: which file}, counts). Raises
    ProviderUnavailable when the files are not on this machine, which is the
    normal case on a node that does not carry them.

    WHICH FILE A MOLECULE CAME FROM IS NOT A DETAIL. Read off the SDF headers
    on 2026-09-11: every molecule in hERG_lhasa_training carries a ChEMBL
    compound id and a ChEMBL assay id, and its label field is named 'Lhasa
    Chembl Call' — Lhasa's own reading of ChEMBL's own measurements.
    hERG_lhasa_test carries four other fields — Doddareddy, Li, Robinson, Sun —
    and an 'Overall Conservative Call' over them, which is a source
    independent of ChEMBL.

    So a molecule shared with a ChEMBL-derived dataset means two different
    things depending on which file it is in, and the counts are kept apart.
    """
    try:
        from experiments.federated.lhasa_herg_loader import (
            load_lhasa_herg_training, load_lhasa_herg_test,
        )
        _, tr_labels, tr_keys, _ = load_lhasa_herg_training()
        _, te_labels, te_keys, _ = load_lhasa_herg_test()
    except Exception as exc:
        raise ProviderUnavailable(
            f"Lhasa's curated hERG files could not be read "
            f"({type(exc).__name__}: {exc})", {'n_groups': 0})

    calls = {}
    source = {}
    n_unknown = 0
    n_conflicting = 0
    for name, keys, labels in (
            ('lhasa_training_curated_from_chembl', tr_keys, tr_labels),
            ('lhasa_test_curated_from_four_published_sets', te_keys,
             te_labels)):
        for key, lab in zip(keys, labels):
            if not key or key == 'UNKNOWN':
                n_unknown += 1
                continue
            lab = int(lab)
            if key in calls and calls[key] != lab:
                # Lhasa disagreeing with itself is a different question from
                # Lhasa disagreeing with ChEMBL. Such a molecule is dropped and
                # counted rather than silently taking one of the two calls.
                n_conflicting += 1
                calls.pop(key, None)
                source.pop(key, None)
                continue
            calls[key] = lab
            source[key] = name
    return calls, source, {
        'n_lhasa_training_molecules': int(len(tr_labels)),
        'n_lhasa_test_molecules': int(len(te_labels)),
        'n_lhasa_distinct_inchikeys': len(calls) + n_conflicting,
        'n_lhasa_unknown_inchikey': n_unknown,
        'n_lhasa_molecules_lhasa_calls_two_ways': n_conflicting,
        'what_each_lhasa_file_is': {
            'lhasa_training_curated_from_chembl':
                "Lhasa's call on ChEMBL compounds and ChEMBL assays — the SDF "
                "field is named 'Lhasa Chembl Call'. A molecule here shared "
                "with a ChEMBL-derived dataset is one organisation's curation "
                "against a threshold on the same source, not two independent "
                "measurements.",
            'lhasa_test_curated_from_four_published_sets':
                "an 'Overall Conservative Call' over the Doddareddy, Li, "
                "Robinson and Sun datasets — independent of ChEMBL.",
        },
    }


def cross_curator(data, dataset_name, threshold=CROSS_CURATOR_THRESHOLD,
                  **kwargs):
    """A molecule this dataset holds that Lhasa also curated, and the two calls.

    Lhasa's label is active or inactive, so a molecule's ground truth here is
    not a magnitude: with exactly two curators it is a flip. They agreed, or
    they did not. The statistic is therefore the AUROC of the estimator's noise
    score against that flag, not a Spearman against a spread.

    The dataset's own label has to be made comparable with an active/inactive
    call. A label that is already 0/1 is used as it stands. A continuous one is
    cut at `threshold`, and the rate is reported at every threshold in
    CROSS_CURATOR_THRESHOLD_SENSITIVITY beside it, because most of the movement
    between 15% and 70% disagreement is the threshold rather than the curators.
    """
    if dataset_name not in CROSS_CURATOR_DATASETS:
        raise ProviderUnavailable(
            f"Lhasa's curated set says whether a molecule blocks hERG, and "
            f"{dataset_name} does not measure that. Two curators can only "
            f"disagree about one property if they both measured it.",
            {'n_groups': 0,
             'datasets_this_provider_covers': sorted(CROSS_CURATOR_DATASETS)})

    smiles, smiles_key = dataset_smiles(data)
    if smiles is None:
        raise ProviderUnavailable(
            "the dataset exposes no SMILES under 'smiles' or 'drug_smiles', "
            "so its molecules cannot be matched to another curator's by "
            "InChIKey", {'n_groups': 0})

    lhasa, lhasa_source, lhasa_counts = _lhasa_calls()

    labels = np.asarray(data['labels'], dtype=np.float64)
    finite = labels[np.isfinite(labels)]
    already_binary = set(np.unique(finite).tolist()) <= {0.0, 1.0}

    keys = _inchikeys(smiles)
    n_no_key = sum(1 for k in keys if not k)

    # ONE GROUP PER MOLECULE, not per row. A drug-target dataset holds the same
    # drug on many rows, once against each protein; counting each of those as a
    # separate shared molecule would report a few thousand where the two
    # organisations hold a few hundred. Rows of one molecule whose labels are
    # not all finite are dropped from it.
    rows_of_key = {}
    for i, k in enumerate(keys):
        if k and k in lhasa and np.isfinite(labels[i]):
            rows_of_key.setdefault(k, []).append(i)
    shared = sorted(rows_of_key)

    counts_out = dict(lhasa_counts)
    counts_out.update({
        'n_rows': int(len(labels)),
        'smiles_read_from': smiles_key,
        'n_rows_without_an_inchikey': n_no_key,
        'n_molecules_both_organisations_hold': len(shared),
        'n_rows_those_molecules_cover': int(
            sum(len(v) for v in rows_of_key.values())),
        'dataset_label_already_binary': bool(already_binary),
    })

    def _calls_at(thr):
        """This dataset's active/inactive call per shared molecule.

        A molecule on several rows is called by the MEAN of its labels, so one
        molecule gets one call. A label that is not finite never reaches here.
        """
        out = {}
        for k in shared:
            value = float(np.mean(labels[rows_of_key[k]]))
            out[k] = int(value) if already_binary else int(value >= thr)
        return out

    sources = sorted({lhasa_source[k] for k in shared})
    sensitivity = {}
    thresholds = ([CROSS_CURATOR_THRESHOLD] if already_binary
                  else list(CROSS_CURATOR_THRESHOLD_SENSITIVITY))
    for thr in thresholds:
        ours = _calls_at(thr)
        differ = sum(1 for k in shared if ours[k] != lhasa[k])
        row = {
            'threshold': None if already_binary else float(thr),
            'n_molecules': len(shared),
            'n_called_differently': differ,
            'rate': (differ / len(shared)) if shared else None,
            'n_this_dataset_calls_active': sum(ours.values()),
            'n_lhasa_calls_active': sum(lhasa[k] for k in shared),
            # Which way the disagreements run. When every one of them is
            # Lhasa-active against this dataset-inactive, the count is
            # measuring how far the cut has moved past Lhasa's actives, not
            # two curators differing.
            'n_lhasa_active_and_this_dataset_inactive': sum(
                1 for k in shared if lhasa[k] == 1 and ours[k] == 0),
            'n_lhasa_inactive_and_this_dataset_active': sum(
                1 for k in shared if lhasa[k] == 0 and ours[k] == 1),
            'by_lhasa_file': {
                src: {
                    'n_molecules': sum(1 for k in shared
                                       if lhasa_source[k] == src),
                    'n_called_differently': sum(
                        1 for k in shared
                        if lhasa_source[k] == src and ours[k] != lhasa[k]),
                } for src in sources},
        }
        sensitivity[('binary_label_no_threshold' if already_binary
                     else str(thr))] = row
    counts_out['threshold_sensitivity'] = sensitivity

    ours = _calls_at(threshold)
    values = {}
    group_ids = np.full(len(labels), -1, dtype=np.int64)
    for gid, key in enumerate(shared):
        group_ids[rows_of_key[key]] = gid
        values[gid] = float(ours[key] != lhasa[key])

    n_disagree = int(sum(values.values()))
    counts_out.update({
        'n_groups': len(values),
        'n_groups_the_two_organisations_call_differently': n_disagree,
        'threshold_used': None if already_binary else float(threshold),
        'n_groups_by_lhasa_file': {
            src: sum(1 for k in shared if lhasa_source[k] == src)
            for src in sources},
    })

    if len(values) < MIN_GROUND_TRUTH_GROUPS:
        raise ProviderUnavailable(
            f"{len(values)} molecules are held by both this dataset and "
            f"Lhasa, below the floor of {MIN_GROUND_TRUTH_GROUPS}", counts_out)
    if n_disagree == 0 or n_disagree == len(values):
        raise ProviderUnavailable(
            f"the two organisations agree on all {len(values)} shared "
            f"molecules ({n_disagree} disagreements), so an AUROC has only one "
            f"class in it", counts_out)

    gt = GroundTruth(
        provider='cross_curator',
        statistic='auroc',
        units='the two curators called this molecule differently: 1 or 0',
        group_ids=group_ids,
        group_value=values,
        counts=counts_out,
        notes={
            'what_a_group_is': 'one molecule, held by this dataset and by '
                               'Lhasa',
            'second_curator': "Lhasa's curated hERG training and test sets",
            'limits': (
                f"{len(values)} shared molecules, a yes/no flag rather than a "
                f"spread, and a threshold that has to be chosen. A supporting "
                f"check, not a replacement for a dataset with real repeat "
                f"measurements. Read 'n_groups_by_lhasa_file' beside it: a "
                f"molecule from Lhasa's training file has a label Lhasa "
                f"derived from ChEMBL, so a disagreement there is one "
                f"organisation's curation against a threshold on the same "
                f"source."),
        },
    )
    counts_out['n_rows_covered'] = gt.n_rows_covered
    return gt


# ── provider 3: the same compound and target, in two different papers ───────

def _raw_chembl_frame(target_chembl_id=None):
    """The un-averaged ChEMBL records, if a raw pull was ever cached.

    src/kirby/datasets/chembl_bioactivity.py replaces each (compound, target)
    group with its mean pIC50 before anything reaches disk, so the spread this
    provider needs does not survive into the normal cache. It is recoverable
    only from a raw frame carrying `document_chembl_id`.
    """
    import pandas as pd
    from kirby.datasets.chembl_bioactivity import _default_data_dir

    data_dir = _default_data_dir('chembl')
    candidates = []
    if os.path.isdir(data_dir):
        for fname in sorted(os.listdir(data_dir)):
            if not fname.startswith('chembl_raw_') or not fname.endswith('.csv'):
                continue
            if target_chembl_id and target_chembl_id not in fname:
                continue
            candidates.append(os.path.join(data_dir, fname))
    if not candidates:
        raise ProviderUnavailable(
            f"no raw ChEMBL frame under {data_dir}: the cached loader output "
            f"holds one averaged row per compound and target, and the "
            f"document ids that separate two papers from one paper listed "
            f"twice are not in it", {'n_groups': 0})

    frames = [pd.read_csv(p) for p in candidates]
    df = pd.concat(frames, ignore_index=True)
    if 'document_chembl_id' not in df.columns:
        raise ProviderUnavailable(
            f"the raw ChEMBL frame at {candidates[0]} carries no "
            f"document_chembl_id, so two independent papers cannot be told "
            f"from one measurement listed twice", {'n_groups': 0})
    return df, candidates


def chembl_multi_document(data, dataset_name, target_chembl_id=None, **kwargs):
    """A compound measured against one target in more than one paper.

    ⛔ DROPPED 2026-09-12 by the owner, on the probe below. Do not propose a
    ChEMBL pull again without a new measurement. `_probe_chembl_herg.py` pulled
    CHEMBL240 (hERG), 12,010 IC50 records over 9,802 compounds: 861 compounds
    carry records from more than one paper, and the spread of pActivity inside
    those 861 has a MEDIAN OF 0.000 log units, a mean of 0.181, and 64 above
    one log unit. Half of them are two papers reporting the same number to the
    digit. BindingDB's 3,494 repeated pairs, measured the same way, sit at a
    median of 0.427 and 909 above one log unit. So there is a count but no
    disagreement, and the loader changes that would have kept the spread were
    not made.

    The provider is left here, working, because it costs nothing and refuses
    on its own when no raw frame with document ids exists — which is always,
    today. If ChEMBL is ever revisited, this is where the ground truth would
    come from.

    The spread of pActivity across the documents is the ground truth. Records
    from a single document are one measurement listed more than once and carry
    no disagreement, so a compound needs more than one DOCUMENT id, not merely
    more than one record.
    """
    import pandas as pd

    smiles, _smiles_key = dataset_smiles(data)
    if smiles is None:
        raise ProviderUnavailable(
            "the dataset exposes no SMILES under 'smiles' or 'drug_smiles', "
            "so its molecules cannot be matched to ChEMBL records by "
            "InChIKey", {'n_groups': 0})

    df, paths = _raw_chembl_frame(target_chembl_id)

    from kirby.datasets.chembl_bioactivity import _standardize_and_inchikey
    if 'inchikey' not in df.columns:
        _, df['inchikey'] = _standardize_and_inchikey(df['smiles_raw']
                                                      if 'smiles_raw' in df
                                                      else df['smiles'])
    if 'pActivity' not in df.columns:
        value = pd.to_numeric(df['standard_value'], errors='coerce')
        df = df[value > 0].copy()
        df['pActivity'] = 9 - np.log10(value[value > 0])
    df = df.dropna(subset=['inchikey', 'pActivity', 'document_chembl_id'])

    keys = _inchikeys(smiles)
    row_of_key = {}
    for i, k in enumerate(keys):
        if k:
            row_of_key.setdefault(k, []).append(i)

    target_col = ('target_chembl_id' if 'target_chembl_id' in df.columns
                  else None)
    key_cols = ['inchikey'] + ([target_col] if target_col else [])

    values = {}
    group_ids = np.full(len(data['labels']), -1, dtype=np.int64)
    n_multi_record = 0
    n_multi_document = 0
    n_matched = 0
    n_second_target = 0
    for gkey, block in df.groupby(key_cols):
        inchikey = gkey[0] if isinstance(gkey, tuple) else gkey
        if len(block) > 1:
            n_multi_record += 1
        n_docs = block['document_chembl_id'].nunique()
        if n_docs < 2:
            continue
        n_multi_document += 1
        rows = row_of_key.get(inchikey)
        if rows is None:
            continue
        if group_ids[rows[0]] >= 0:
            # This compound already carries a group from another target. The
            # dataset has one label per molecule, so a second target's spread
            # has nowhere to sit; it is counted and left out rather than
            # written to a group no score can ever reach.
            n_second_target += 1
            continue
        per_doc = block.groupby('document_chembl_id')['pActivity'].mean()
        gid = len(values)
        group_ids[rows] = gid
        values[gid] = float(np.std(per_doc.to_numpy(),
                                   ddof=GROUP_SPREAD_DDOF))
        n_matched += 1

    counts_out = {
        'raw_frames_read': paths,
        'n_raw_records_with_a_key_a_value_and_a_document': int(len(df)),
        'n_compounds_with_more_than_one_record': n_multi_record,
        'n_compounds_with_more_than_one_document': n_multi_document,
        'n_of_those_also_in_this_dataset': n_matched,
        'n_dropped_because_the_compound_already_had_a_group': n_second_target,
        'n_groups': len(values),
    }
    if len(values) < MIN_GROUND_TRUTH_GROUPS:
        raise ProviderUnavailable(
            f"{len(values)} of this dataset's molecules carry ChEMBL records "
            f"from more than one paper, below the floor of "
            f"{MIN_GROUND_TRUTH_GROUPS}", counts_out)

    gt = GroundTruth(
        provider='chembl_multi_document',
        statistic='spearman',
        units='spread of pActivity across the papers, in log units',
        group_ids=group_ids,
        group_value=values,
        counts=counts_out,
        notes={
            'what_a_group_is': 'one compound against one target, measured in '
                               'more than one paper',
            'caveat': (
                "an IC50 depends on the assay — cell line, substrate "
                "concentration, incubation time — so two papers reporting "
                "different values for one compound may both be right about "
                "different assays. The label a model trains on is ambiguous "
                "either way, but this is a weaker claim than a Kd, which is a "
                "property of the binding pair rather than of the experiment."),
        },
    )
    counts_out['n_rows_covered'] = gt.n_rows_covered
    return gt


PROVIDERS = {
    'drug_target_repeats': drug_target_repeats,
    'cross_curator': cross_curator,
    'chembl_multi_document': chembl_multi_document,
}


def collect_ground_truths(data, dataset_name, providers=None, **kwargs):
    """Run every provider over one dataset and say what each one found.

    Returns (list of GroundTruth, report). The report holds one entry per
    provider whether it produced a ground truth or not, with the counts it
    reached and the reason it stopped — a dataset with no repeat structure is a
    fact worth recording, and Davis and KIBA both look like candidates until
    the repeated pairs are counted (they have none).
    """
    names = list(providers) if providers else list(PROVIDERS)
    found = []
    report = {}
    for name in names:
        fn = PROVIDERS[name]
        try:
            gt = fn(data, dataset_name, **kwargs)
        except ProviderUnavailable as exc:
            report[name] = {'used': False, 'reason': exc.reason,
                            'counts': exc.counts}
            print(f"  ground truth [{name}]: none — {exc.reason}")
            continue
        except Exception as exc:  # a provider must never take the run down
            report[name] = {'used': False,
                            'reason': f'{type(exc).__name__}: {exc}',
                            'counts': {}}
            print(f"  ground truth [{name}]: failed — "
                  f"{type(exc).__name__}: {exc}")
            continue
        found.append(gt)
        report[name] = {'used': True, 'statistic': gt.statistic,
                        'units': gt.units, 'counts': gt.counts,
                        'notes': gt.notes}
        print(f"  ground truth [{name}]: {gt.n_groups} groups covering "
              f"{gt.n_rows_covered} rows, scored by {gt.statistic}")
    return found, report


def reduce_to_groups(gt: GroundTruth, scores: np.ndarray):
    """One point per group: its ground truth, and the mean score of its rows.

    The spread of a group is one number, and broadcasting it to every member
    and correlating over rows counts the same ground-truth value once per
    repeat. On BindingDB that turns 3,494 independent values into 9,676 rows
    and overstates the sample about three times.

    Returns (group_ids, ground_truth_values, mean_scores, n_rows_per_group),
    with groups whose rows are all unscored dropped.
    """
    gids, values = gt.group_order()
    keep_g, keep_v, means, sizes = [], [], [], []
    index = gt.rows
    for gid, value in zip(gids, values):
        rows = index.get(int(gid))
        if rows is None:
            continue
        vals = scores[rows]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        keep_g.append(gid)
        keep_v.append(value)
        means.append(float(np.mean(vals)))
        sizes.append(int(len(vals)))
    return (np.array(keep_g), np.array(keep_v, dtype=float),
            np.array(means, dtype=float), np.array(sizes, dtype=int))


def agreement(gt: GroundTruth, scores: np.ndarray) -> Optional[dict]:
    """How well one estimator's scores agree with one provider's ground truth.

    Spearman when the ground truth is a magnitude, AUROC when it is a yes/no
    flag. One row per group either way. Returns None when there is nothing to
    compute and says why.
    """
    from scipy.stats import spearmanr

    gids, values, means, sizes = reduce_to_groups(gt, scores)
    n = len(gids)
    out = {'n_replicate_groups': int(n),
           'n_rows_behind_them': int(sizes.sum()) if n else 0,
           'statistic': gt.statistic}

    if n < MIN_GROUND_TRUTH_GROUPS:
        out['value'] = None
        out['refused'] = (f'{n} groups carried a score, below the floor of '
                          f'{MIN_GROUND_TRUTH_GROUPS}')
        return out

    if gt.statistic == 'spearman':
        if np.std(values) == 0 or np.std(means) == 0:
            out['value'] = None
            out['refused'] = ('every group has the same ground truth or the '
                              'same score')
            return out
        rho, p = spearmanr(means, values)
        out['value'] = float(rho)
        out['p'] = float(p)
        return out

    if gt.statistic == 'auroc':
        from sklearn.metrics import roc_auc_score
        flags = (values > 0).astype(int)
        if flags.min() == flags.max():
            out['value'] = None
            out['refused'] = 'the flag has only one class in it'
            return out
        out['value'] = float(roc_auc_score(flags, means))
        out['n_positive'] = int(flags.sum())
        return out

    out['value'] = None
    out['refused'] = f'unknown statistic {gt.statistic!r}'
    return out
