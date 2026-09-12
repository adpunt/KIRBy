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

# The datasets whose label is the property these curators measured: whether a
# molecule blocks the hERG channel. Two curators only disagree about one thing
# if they both measured it. BindingDB shares 272 molecules with the Lhasa
# files, and its label for each of them is a binding affinity against some
# other protein, so a comparison there would have produced 176 flags that mean
# nothing.
CROSS_CURATOR_DATASETS = {'herg', 'herg_ki'}

# The four published hERG collections whose calls sit side by side in
# hERG_lhasa_test.sdf, one column each. Lhasa's own loader (FLuID.py:379) drops
# all four and keeps only its 'Overall Conservative Call', so the disagreement
# between them exists on disk and nothing reads it.
CROSS_CURATOR_SOURCES = ('Doddareddy', 'Li', 'Robinson', 'Sun')
LHASA_MULTI_SOURCE_FILE = 'hERG_lhasa_test.sdf'


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


def _published_source_calls(data_dir=None):
    """Each published collection's own hERG call, per molecule.

    `hERG_lhasa_test.sdf` carries one column per collection — Doddareddy, Li,
    Robinson and Sun — beside the consensus Lhasa derives from them. A column
    holds that collection's call for the molecule, or two calls separated by a
    comma where the collection listed it twice, or nothing where it never saw
    it. Lhasa's own loader drops all four columns and keeps the consensus
    (FLuID.py:379), so these calls sit on disk and nothing reads them.

    A molecule called by two or more collections is four curators' work on one
    compound. Whether they agree is the ground truth this provider scores
    against — an actual disagreement between independent groups, which is what
    the earlier version of this provider only appeared to measure.

    Parsed once and cached beside the data file: reading the SDF and building
    5,770 InChIKeys costs about 84 seconds, and the answer never changes.

    Returns ({inchikey: {collection: set of calls}}, counts).
    """
    import csv

    data_dir = data_dir or os.environ.get('LHASA_DATA_DIR')
    if data_dir is None:
        from experiments.federated.lhasa_herg_loader import _DEFAULT_DATA_DIR
        data_dir = _DEFAULT_DATA_DIR
    sdf_path = os.path.join(data_dir, LHASA_MULTI_SOURCE_FILE)
    cache_path = os.path.join(data_dir, 'hERG_published_source_calls.csv')

    if os.path.exists(cache_path):
        per_molecule = {}
        with open(cache_path) as fh:
            for row in csv.DictReader(fh):
                per_molecule[row['inchikey']] = {
                    src: set(row[src].split('|')) for src in
                    CROSS_CURATOR_SOURCES if row.get(src)}
        return per_molecule, {'read_from': cache_path,
                              'n_molecules_parsed': len(per_molecule)}

    if not os.path.exists(sdf_path):
        raise ProviderUnavailable(
            f"{sdf_path} is not on this machine, so the four published hERG "
            f"collections cannot be read", {'n_groups': 0})

    from rdkit.Chem import PandasTools
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    frame = PandasTools.LoadSDF(sdf_path, molColName='MOLECULE')
    missing = [s for s in CROSS_CURATOR_SOURCES if s not in frame.columns]
    if missing:
        raise ProviderUnavailable(
            f"{sdf_path} carries no column for {missing}; the file layout has "
            f"changed", {'n_groups': 0})

    per_molecule = {}
    n_no_key = 0
    for _, row in frame.iterrows():
        mol = row['MOLECULE']
        key = None
        if mol is not None:
            try:
                inchi = MolToInchi(mol)
                key = InchiToInchiKey(inchi) if inchi else None
            except Exception:
                key = None
        if not key:
            n_no_key += 1
            continue
        # A molecule can appear on more than one row of the file. Those rows
        # are the same compound, so their calls are MERGED rather than the
        # later row replacing the earlier one — overwriting loses a collection
        # that only the first row recorded, and with it 7 of the 774 molecules
        # that two or more collections called.
        called = per_molecule.setdefault(key, {})
        for src in CROSS_CURATOR_SOURCES:
            value = row.get(src) or ''
            calls = {c.strip() for c in str(value).split(',') if c.strip()}
            if calls:
                called.setdefault(src, set()).update(calls)
        if not called:
            per_molecule.pop(key, None)

    with open(cache_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['inchikey', *CROSS_CURATOR_SOURCES])
        for key, called in per_molecule.items():
            writer.writerow([key] + ['|'.join(sorted(called.get(s, ())))
                                     for s in CROSS_CURATOR_SOURCES])

    return per_molecule, {'read_from': sdf_path,
                          'cached_to': cache_path,
                          'n_rows_in_file': int(len(frame)),
                          'n_rows_without_an_inchikey': n_no_key,
                          'n_molecules_parsed': len(per_molecule)}


def cross_curator(data, dataset_name, **kwargs):
    """A molecule four published collections each made their own call on.

    Doddareddy, Li, Robinson and Sun each curated hERG activity, and their
    calls sit side by side in Lhasa's test file. Where two or more of them
    called the same molecule, either they agree or one of them is wrong. That
    is a disagreement between independent groups, and no threshold has to be
    chosen to see it — both sides are already active or inactive.

    The statistic is AUROC: with several curators and a yes-or-no outcome a
    molecule's ground truth is a flip, not a magnitude, and it is not forced
    through the Spearman path.

    WHAT THIS REPLACED, 2026-09-12. Until today this provider cut a dataset's
    continuous pKi at 5.0 and compared that to Lhasa's own call. Measured, all
    51 of the disagreements it found came from the Lhasa file whose label
    Lhasa had itself derived from ChEMBL — one curation against a threshold on
    the same numbers, not two organisations. The independent file disagreed on
    none of its 170 shared molecules at that cut. The four columns below are
    the real thing, and Lhasa's own pipeline deletes them before use.
    """
    if dataset_name not in CROSS_CURATOR_DATASETS:
        raise ProviderUnavailable(
            f"the four published collections say whether a molecule blocks "
            f"hERG, and {dataset_name} does not measure that. Curators can "
            f"only disagree about one property if they all measured it.",
            {'n_groups': 0,
             'datasets_this_provider_covers': sorted(CROSS_CURATOR_DATASETS)})

    smiles, smiles_key = dataset_smiles(data)
    if smiles is None:
        raise ProviderUnavailable(
            "the dataset exposes no SMILES under 'smiles' or 'drug_smiles', "
            "so its molecules cannot be matched to the published collections "
            "by InChIKey", {'n_groups': 0})

    published, parse_counts = _published_source_calls()

    keys = _inchikeys(smiles)
    n_no_key = sum(1 for k in keys if not k)
    rows_of_key = {}
    for i, k in enumerate(keys):
        if k and k in published:
            rows_of_key.setdefault(k, []).append(i)

    # A group needs two or more COLLECTIONS. One collection listing a molecule
    # twice is that collection contradicting itself, which is a different
    # question and is counted separately rather than mixed in.
    values = {}
    group_ids = np.full(len(data['labels']), -1, dtype=np.int64)
    n_one_collection = 0
    n_self_contradicting = 0
    per_n_sources = {}
    for key, rows in rows_of_key.items():
        called = published[key]
        n_src = len(called)
        per_n_sources[n_src] = per_n_sources.get(n_src, 0) + 1
        if any(len(c) > 1 for c in called.values()):
            n_self_contradicting += 1
        if n_src < 2:
            n_one_collection += 1
            continue
        gid = len(values)
        group_ids[rows] = gid
        union = set().union(*called.values())
        values[gid] = float(len(union) > 1)

    n_disagree = int(sum(values.values()))

    # THE CONSERVATIVE CALL IS THE FLAG. Lhasa's consensus over the four
    # collections is "any Active wins": measured on the file, all 136 molecules
    # the collections split over are called Active, all 470 they agree are
    # inactive on are Inactive. So on a dataset whose label IS that consensus,
    # "the collections disagreed" and "the label is Active" are the same
    # statement, and an estimator scored against the flag is being scored
    # against the class label. herg carries Lhasa's call on ChEMBL and herg_ki
    # carries a pKi, so neither is that dataset — but the check is made here
    # rather than trusted to a list of names.
    own_label = np.asarray(data['labels'], dtype=np.float64)
    label_is_binary = set(
        np.unique(own_label[np.isfinite(own_label)]).tolist()) <= {0.0, 1.0}
    n_flag_matches_label = 0
    if label_is_binary:
        for gid, flag in values.items():
            rows = np.flatnonzero(group_ids == gid)
            if len(rows) and float(own_label[rows[0]] > 0) == flag:
                n_flag_matches_label += 1

    counts_out = dict(parse_counts)
    counts_out.update({
        'n_rows': int(len(data['labels'])),
        'smiles_read_from': smiles_key,
        'n_rows_without_an_inchikey': n_no_key,
        'collections': list(CROSS_CURATOR_SOURCES),
        'n_molecules_this_dataset_shares_with_the_collections':
            len(rows_of_key),
        'n_of_those_called_by_only_one_collection': n_one_collection,
        'n_molecules_by_how_many_collections_called_them': per_n_sources,
        'n_groups': len(values),
        'n_groups_the_collections_call_differently': n_disagree,
        'n_molecules_one_collection_contradicts_itself_on':
            n_self_contradicting,
        'dataset_label_is_binary': bool(label_is_binary),
        'n_groups_where_the_flag_equals_this_datasets_own_label':
            n_flag_matches_label if label_is_binary else None,
    })

    if len(values) < MIN_GROUND_TRUTH_GROUPS:
        raise ProviderUnavailable(
            f"{len(values)} of this dataset's molecules were called by two or "
            f"more of the four collections, below the floor of "
            f"{MIN_GROUND_TRUTH_GROUPS}", counts_out)
    if n_disagree == 0 or n_disagree == len(values):
        raise ProviderUnavailable(
            f"the collections agree on all {len(values)} shared molecules "
            f"({n_disagree} disagreements), so an AUROC has only one class in "
            f"it", counts_out)
    if label_is_binary and n_flag_matches_label == len(values):
        raise ProviderUnavailable(
            f"whether the collections disagreed is the same statement as this "
            f"dataset's own label on all {len(values)} molecules — the label "
            f"is the consensus derived from those collections, so an "
            f"estimator scored against it would be scored against the class",
            counts_out)

    gt = GroundTruth(
        provider='cross_curator',
        statistic='auroc',
        units='two or more of the four collections called this molecule '
              'differently: 1 or 0',
        group_ids=group_ids,
        group_value=values,
        counts=counts_out,
        notes={
            'what_a_group_is': 'one molecule, called by two or more of the '
                               'four published hERG collections',
            'curators': list(CROSS_CURATOR_SOURCES),
            'where_it_comes_from': (
                f'the per-collection columns of {LHASA_MULTI_SOURCE_FILE}. '
                f"Lhasa's own loader drops them and keeps only the consensus "
                f"it derives from them, so nothing else reads these calls."),
            'limits': (
                f'{len(values)} molecules, a yes-or-no flag rather than a '
                f'spread, and no threshold anywhere. A supporting check beside '
                f'a dataset with real repeat measurements, not a replacement '
                f'for one.'),
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
