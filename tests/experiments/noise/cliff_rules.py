#!/usr/bin/env python3
"""Every way this suite can call a pair of molecules an activity cliff, in one
place, so two experiments cannot disagree without saying so.

WHY THIS EXISTS. Nine different cliff rules are written out across the suite,
each in the file that needed one, and the file that stores the answer does not
record which rule produced it. Measured consequences already on disk:

  herg_ki, 1,412 molecules. ECFP4 Tanimoto at or above 0.55 with a gap of 2.0
  times the median absolute deviation calls 451 of them cliff molecules
  (similarity_helpers.py block comment, 2026-09-19). The same structural side
  with the gap solved to aim at 15 per cent calls 211 of 1,412 (thesis.tex:1120).
  The top half per cent of pairwise similarity on every representation glued
  together, same 2.0 times median absolute deviation, calls 295 of 1,412
  (similarity_helpers.py, s12).

  n47_noise_measurements.py, written 2026-11-02, carries two of these rules in
  one file: its n01 part calls n25.cliff_mask_for_dataset (2.0 times the median
  absolute deviation) and its n14 part calls n14._build_ac_reference (an
  absolute 2 label units, or the loader's matched pairs where it ships them).
  Both write a field named n_cliff_compounds into the same result file.

WHAT A RULE IS. Three choices, kept apart here because the literature varies
each one independently:

  a similarity criterion   an n x n boolean: is this pair structurally similar
  a gap criterion          an n x n boolean: are these labels far apart
  the unit                 a pair, or a molecule in at least one cliff pair

cliff_sets() takes one name from SIMILARITY_CRITERIA and one from
GAP_CRITERIA and returns both units plus the counts. RULES names the
combinations that come from a published paper, so a result can say
"moleculeace" rather than carrying two thresholds.

WHAT IS NOT DECIDED HERE. Which rule an experiment uses. That is Adelaide's
(HANDOFF_PAPER3_CLIFFS.md, "Decisions that are Adelaide's"). This module makes
the choice explicit and swappable; it does not make it.

TWO DECISIONS ALREADY MADE, carried into every criterion below:

  An unparseable SMILES is similar to nothing. Its fingerprint row is all
  zeros and its similarity is forced to 0, so two unparseable molecules are
  not a pair at similarity 1 (Adelaide, 2026-10-29, n01).

  Two rows of the same molecule are never a cliff pair. Datasets loaded one
  row per measurement (herg_ic50: 12,008 rows, 9,802 molecules) put two
  repeats of one molecule at similarity 1.0, and a repeat gap past the cutoff
  was being counted as a cliff (Adelaide, 2026-10-06, n25).

WHAT THE SIMILARITY CRITERIA COST. Every one is O(n^2) in memory: a 2,641 x
2,641 float32 matrix is 28 MB, and the suite's largest molecular set under the
n47 out-of-memory guard is well inside that. Nothing here fits a model.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist as scipy_cdist

# ---------------------------------------------------------------------------
# Thresholds, each with the paper it comes from
# ---------------------------------------------------------------------------

# Hu, Furtmann, Stumpfe & Bajorath 2015, F1000Research 4:168. The structural
# half of the convention n01, n14, n25, s04 and s12 all use.
HU2015_ECFP4_TANIMOTO = 0.55

# van Tilborg, Alenicheva & Grisoni 2022, J. Chem. Inf. Model. 62:5938
# (MoleculeACE). Similar if ANY of three measures is at or above 0.90, and a
# gap of 1 log unit.
MOLECULEACE_SIMILARITY = 0.90
MOLECULEACE_GAP_LOG_UNITS = 1.0

# Geometry of Activity Cliffs (arXiv:2605.30831, a preprint, future-dated;
# second-hand through the research session of 2026-11-01, not read here). Its
# strongest-agreeing setting of the 15 it swept.
GEOMETRY_MORGAN_TANIMOTO = 0.90

# MACCS keys. 0.85 is the threshold the research session reported for MACCS;
# second-hand, no paper read here.
MACCS_TANIMOTO = 0.85

# The literature absolute criterion: 100-fold potency, 2 log units. n14's
# AC_POTENCY_THRESHOLD, and Dablander et al. 2023, J. Cheminform. 15:47.
ABSOLUTE_LOG_UNITS = 2.0

# Dablander's half-cliff band. A pair 1 to 2 log units apart is neither a
# cliff nor a non-cliff and is dropped from both sides.
HALF_CLIFF_BAND = (1.0, 2.0)

# n01's gap: this many median absolute deviations of the dataset's own labels.
MAD_MULTIPLE = 2.0

# similarity_helpers.CLIFF_TARGET_PCT. The share of molecules the solved gap
# AIMS AT. solve_activity_gap returns the gap whose achieved share is nearest
# this, not one that reaches it, so the achieved share is reported separately
# everywhere below and the target is never printed as an outcome.
SOLVED_TARGET_PCT = 15.0

# Kramer, Kalliokoski, Gedeck & Vulpetti 2012, J. Med. Chem. 55:5165. The
# standard deviation of the difference between two independent public Ki
# measurements of the same pair, in pKi. Used here only as a gap criterion of
# its own and as a floor to measure other rules against.
KRAMER_NOISE_FLOOR_PKI = 0.54

# Percentile-of-pairs thresholds already in use: n01.SIMILAR_PERCENTILE is 2.0,
# similarity_helpers.CLIFF_SIMILARITY_PERCENTILE is 1.0, s12 used 0.5, and
# ActivityCliffDetector's default is the closest 10 per cent of distances.
TOP_PAIR_PERCENTILES = (0.5, 1.0, 2.0, 10.0)

# ActivityCliffDetector's own two defaults (noise_mitigation_methods.py).
DETECTOR_DISTANCE_PERCENTILE = 10.0
DETECTOR_GAP_PERCENTILE = 90.0

# Among the k nearest neighbours, no similarity threshold at all. The SALI
# feature factory uses k=10 (representations/molecular.py:1467).
KNN_K = 10

# Antibody sequences: the absolute edit-distance cutoff, and the percentile
# alternative, both from similarity_helpers.identify_cliff_compounds_editdist.
EDIT_DISTANCE_MAX = 5
EDIT_DISTANCE_PERCENTILE = 5.0


# ---------------------------------------------------------------------------
# Fingerprints and similarity matrices
# ---------------------------------------------------------------------------

def _mols(smiles):
    """RDKit molecules, None where the string will not parse."""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    return [None if s is None else Chem.MolFromSmiles(str(s)) for s in smiles]


def _bits_from(mols, kind, n_bits=2048):
    """(n, width) float32 0/1 matrix. An unparseable molecule gets a zero row.

    kind:
      'ecfp4'    Morgan radius 2, n_bits wide. n01's fingerprint exactly.
      'ecfp6'    Morgan radius 3.
      'maccs'    the 167 MACCS keys.
      'rdkit'    RDKit's path-based fingerprint.
      'scaffold' Morgan radius 2 of the GENERIC framework, which is
                 MoleculeACE's second measure. Generic means every atom turned
                 to carbon and every bond to single. MurckoScaffold
                 .MakeScaffoldGeneric is applied to the WHOLE molecule, not to
                 its Murcko scaffold first, and the plain Murcko scaffold is
                 the fallback where that raises. Both of those are what their
                 cliffs.py get_scaffold_matrix does, read on 2026-11-02 at
                 /Volumes/seagate/moleculeace_src/cliffs.py:157-169.

                 ⛔ THIS WAS THE PLAIN MURCKO SCAFFOLD UNTIL 2026-11-02 AND
                 THAT WAS WRONG. With the plain scaffold our moleculeace rule
                 called 1,821 of CHEMBL214_Ki's 3,317 molecules cliffs against
                 MoleculeACE's own 1,222, a Matthews correlation of 0.5566.
                 With the generic scaffold it calls exactly their 1,222: zero
                 molecules either way. Measured at both 1,024 and 2,048 bits,
                 which give the same final cliff set on that target even though
                 the Tanimoto component alone moves from 148 to 144 molecules.
    """
    from rdkit.Chem import AllChem, MACCSkeys, rdmolops
    from rdkit.Chem.Scaffolds import MurckoScaffold

    width = 167 if kind == 'maccs' else n_bits
    out = np.zeros((len(mols), width), dtype=np.float32)
    bad = np.zeros(len(mols), dtype=bool)
    for i, mol in enumerate(mols):
        if mol is None:
            bad[i] = True
            continue
        try:
            if kind == 'ecfp4':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            elif kind == 'ecfp6':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=n_bits)
            elif kind == 'maccs':
                fp = MACCSkeys.GenMACCSKeys(mol)
            elif kind == 'rdkit':
                fp = rdmolops.RDKFingerprint(mol, fpSize=n_bits)
            elif kind == 'scaffold':
                # MoleculeACE/benchmark/cliffs.py get_scaffold_matrix: the
                # generic framework, with the plain Murcko scaffold as the
                # fallback their code also takes.
                try:
                    core = MurckoScaffold.MakeScaffoldGeneric(mol)
                except Exception:
                    core = MurckoScaffold.GetScaffoldForMol(mol)
                if core is None or core.GetNumAtoms() == 0:
                    bad[i] = True
                    continue
                fp = AllChem.GetMorganFingerprintAsBitVect(core, 2, nBits=n_bits)
            else:
                raise ValueError(f'unknown fingerprint {kind!r}')
        except Exception:
            bad[i] = True
            continue
        for bit in fp.GetOnBits():
            out[i, bit] = 1.0
    return out, bad


def tanimoto(bits, bad=None):
    """n x n Tanimoto similarity of 0/1 rows, diagonal 0, bad rows 0.

    Computed from dot products: the intersection of two 0/1 rows is their dot
    product, and the union is the two row sums less the intersection. One
    matrix multiply, so it runs through BLAS.

    WHY NOT scipy's cdist. metric='jaccard' is O(n^2 m) outside BLAS and it
    dominated everything else. Measured 2026-11-02 on CHEMBL214_Ki's 3,317
    molecules at 2,048 bits: 214.61 seconds for scipy against 1.75 here, and
    the two matrices are bit-identical, largest absolute difference exactly
    0.0. The whole MoleculeACE rule on that target went from 625 seconds to
    40. scipy's 'jaccard' also thresholds at
    nonzero rather than raising on count features, so it silently returns
    nonsense on counts; this takes 0/1 rows only, and
    preprocessing.require_binary says what goes wrong otherwise.
    """
    bits = np.asarray(bits, dtype=np.float32)
    inter = bits @ bits.T
    counts = bits.sum(axis=1)
    union = counts[:, None] + counts[None, :] - inter
    with np.errstate(invalid='ignore', divide='ignore'):
        sim = np.where(union > 0, inter / union, 0.0)
    return _clean_sim(sim, bad)


def dice(bits, bad=None):
    """n x n Dice similarity of 0/1 rows, diagonal 0, bad rows 0.

    2|A and B| / (|A| + |B|). Not in distance_config.DISTANCE_METRICS and not
    reachable through scipy's cdist for the count case, so it is written out.
    The Geometry preprint's largest cliff count of the 15 settings it swept was
    RDKit fingerprint under Dice (7,903 pairs against 142 at Morgan Tanimoto
    0.90); second-hand through the research session of 2026-11-01.
    """
    bits = np.asarray(bits, dtype=np.float32)
    inter = bits @ bits.T
    counts = bits.sum(axis=1)
    den = counts[:, None] + counts[None, :]
    with np.errstate(invalid='ignore', divide='ignore'):
        sim = np.where(den > 0, 2.0 * inter / den, 0.0)
    return _clean_sim(sim, bad)


def cosine_sim(X):
    """n x n cosine similarity of continuous rows, diagonal 0.

    For the three learned embeddings. A zero row gets similarity 0 rather than
    a nan.
    """
    X = np.asarray(X, dtype=np.float64)
    norm = np.linalg.norm(X, axis=1)
    bad = ~np.isfinite(norm) | (norm == 0)
    safe = np.where(bad, 1.0, norm)
    Z = X / safe[:, None]
    sim = Z @ Z.T
    return _clean_sim(sim, bad)


def _clean_sim(sim, bad=None):
    sim = np.asarray(sim, dtype=np.float32)
    sim[~np.isfinite(sim)] = 0.0
    if bad is not None:
        bad = np.asarray(bad, dtype=bool)
        sim[bad, :] = 0.0
        sim[:, bad] = 0.0
    np.fill_diagonal(sim, 0.0)
    return sim


def smiles_levenshtein_similarity(smiles):
    """1 - (edit distance / length of the longer string), n x n, diagonal 0.

    MoleculeACE's third measure. rapidfuzz does the quadratic work in C; the
    pure-python Levenshtein in similarity_helpers is O(n^2 L^2) and will not
    finish on 2,000 molecules.
    """
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz import process
    strings = [('' if s is None else str(s)) for s in smiles]
    sim = process.cdist(strings, strings, scorer=Levenshtein.normalized_similarity,
                        dtype=np.float32)
    return _clean_sim(sim, np.array([s == '' for s in strings]))


def sequence_edit_distance(sequences):
    """n x n edit distance between sequences, for the antibody sets.

    Hamming when every sequence has the same length -- which is what a deep
    mutational scan gives -- and Levenshtein otherwise. Same choice as
    similarity_helpers.identify_cliff_compounds_editdist, computed through
    rapidfuzz instead of nested python loops.
    """
    from rapidfuzz.distance import Levenshtein, Hamming
    from rapidfuzz import process
    seqs = [('' if s is None else str(s)) for s in sequences]
    same_len = len({len(s) for s in seqs}) == 1
    scorer = Hamming.distance if same_len else Levenshtein.distance
    D = process.cdist(seqs, seqs, scorer=scorer, dtype=np.float32)
    D = np.asarray(D, dtype=np.float32)
    np.fill_diagonal(D, 0.0)
    return D, ('hamming' if same_len else 'levenshtein')


# ---------------------------------------------------------------------------
# One context per dataset: everything a criterion might read, built once
# ---------------------------------------------------------------------------

class CliffContext:
    """The matrices every criterion reads, built once and cached per dataset.

    Attributes a criterion may ask for, each built on first use:
      smiles, sequences, labels, is_clf
      bits(kind)          0/1 fingerprint matrix and the unparseable rows
      sim(name)           an n x n similarity matrix
      features            the standardised concatenation of every
                          representation, for a dataset with no SMILES
      embedding(name)     one named representation's matrix, where loaded
      same_molecule       n x n, True where two rows are one molecule
      mmp_pairs           the loader's matched pairs, where it ships them
    """

    def __init__(self, labels, *, smiles=None, sequences=None, reps=None,
                 mmp_pairs=None, is_clf=False, dataset_name=None):
        self.labels = np.asarray(labels, dtype=float).ravel()
        self.n = len(self.labels)
        self.smiles = list(smiles) if smiles is not None else None
        self.sequences = list(sequences) if sequences is not None else None
        self.reps = dict(reps or {})
        self.mmp_pairs = mmp_pairs
        self.is_clf = bool(is_clf)
        self.dataset_name = dataset_name
        self._mols = None
        self._bits = {}
        self._sim = {}
        self._features = None
        self._same_molecule = None
        self._cdr = False          # False = not asked for yet; None = none
        self.cdr_report = {}
        self._gap = None

    @property
    def gap_matrix(self):
        """|y_i - y_j|, or the class mismatch, built once.

        Added 2026-11-02 for the threshold sweep: every gap criterion was
        rebuilding this, and the sweep asks for 27 gap points against 89
        similarity points per dataset. It is the same quantity `pairwise_gap`
        returns and the callers read it rather than recomputing it.
        """
        if self._gap is None:
            self._gap = pairwise_gap(self.labels, self.is_clf)
        return self._gap

    @property
    def cdr_sequences(self):
        """The CDR-only strings, or None where they cannot be extracted.

        `cdr_report` then holds how many of them came back equal to the full
        chain, which is the silent fallback `cdr_sequences` documents.
        """
        if self._cdr is False:
            self._cdr = cdr_sequences(self.sequences,
                                      dataset_name=self.dataset_name,
                                      report=self.cdr_report)
        return self._cdr

    # -- structures ---------------------------------------------------------

    @property
    def mols(self):
        if self._mols is None:
            if self.smiles is None:
                raise ValueError('this dataset has no SMILES')
            self._mols = _mols(self.smiles)
        return self._mols

    def bits(self, kind, n_bits=2048):
        """(matrix, unparseable mask) for one fingerprint, built once.

        ⛔ THE WIDTH IS PART OF THE KEY, added 2026-11-02. It was not, and
        `sim_moleculeace` asks for 1,024 bits while every other criterion asks
        for 2,048, so without the width in the key the two would have shared an
        entry and returned the wrong width. Measured cost of getting this
        wrong the other way, with no cache at all: the generic framework of
        hERG Ki's 1,412 molecules takes 1,410s to build, and the six
        moleculeace cutoffs in the threshold sweep each rebuilt it, so one
        dataset spent 2h21m on six copies of one matrix.
        """
        key = (kind, int(n_bits))
        if key not in self._bits:
            self._bits[key] = _bits_from(self.mols, kind, n_bits=n_bits)
        return self._bits[key]

    def sim(self, name, build):
        """Memoised similarity matrix. `build` is called once."""
        if name not in self._sim:
            self._sim[name] = build()
        return self._sim[name]

    @property
    def features(self):
        """Every representation z-scored and glued together.

        The fallback n01 and s12 use where no SMILES exist. Z-scoring per
        feature before gluing is the only part that is not optional: two blocks
        of different width otherwise contribute distance in proportion to their
        widths (preprocessing.py's measured 3977 against 241 at 2048 and 128
        features).
        """
        if self._features is None:
            if not self.reps:
                raise ValueError('no representations were loaded')
            blocks = []
            for name in sorted(self.reps):
                X = np.asarray(self.reps[name], dtype=np.float64)
                if X.ndim == 1:
                    X = X[:, None]
                mu, sd = X.mean(axis=0), X.std(axis=0)
                sd = np.where(sd > 0, sd, 1.0)
                blocks.append((X - mu) / sd)
            self._features = np.hstack(blocks)
        return self._features

    def embedding(self, name):
        if name not in self.reps:
            raise KeyError(f'representation {name!r} was not loaded')
        return np.asarray(self.reps[name], dtype=np.float64)

    @property
    def same_molecule(self):
        """n x n, True where two rows carry the same SMILES. All False when
        every row is a different molecule, or when there are no SMILES."""
        if self._same_molecule is None:
            self._same_molecule = np.zeros((self.n, self.n), dtype=bool)
            if self.smiles is not None and len(self.smiles) == self.n:
                ids, mol_id = {}, []
                for s in self.smiles:
                    mol_id.append(ids.setdefault(str(s), len(ids)))
                if len(ids) < self.n:
                    mol_id = np.asarray(mol_id)
                    m = mol_id[:, None] == mol_id[None, :]
                    np.fill_diagonal(m, False)
                    self._same_molecule = m
        return self._same_molecule


# ---------------------------------------------------------------------------
# Similarity criteria. Each returns (n x n boolean, info).
# ---------------------------------------------------------------------------

def _threshold(sim, cutoff):
    m = np.asarray(sim) >= cutoff
    np.fill_diagonal(m, False)
    return m


def _top_percentile_pairs(sim, pct):
    """The top `pct` per cent of off-diagonal pairs by similarity.

    Distribution-relative, so it is defined whatever the similarity is bounded
    by -- which is why n01 and s12 fall back to it where no Tanimoto exists.
    It cannot be read as a chemical statement: the same pct returns the same
    number of pairs on a congeneric series and on a diverse screen.
    """
    sim = np.asarray(sim)
    n = sim.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = sim[iu]
    if vals.size == 0:
        return np.zeros_like(sim, dtype=bool), None
    cutoff = float(np.percentile(vals, 100.0 - pct))
    return _threshold(sim, cutoff), cutoff


def _knn_pairs(sim, k):
    """A pair is similar if either molecule is in the other's k nearest.

    No threshold at all: every molecule is in exactly k similar pairs at
    least, so the similar set never empties and never saturates.
    """
    sim = np.asarray(sim)
    n = sim.shape[0]
    k = min(int(k), max(n - 1, 1))
    idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    m = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), k)
    m[rows, idx.ravel()] = True
    m |= m.T
    np.fill_diagonal(m, False)
    return m


def sim_ecfp4_tanimoto(ctx, cutoff=HU2015_ECFP4_TANIMOTO):
    bits, bad = ctx.bits('ecfp4')
    S = ctx.sim('ecfp4_tanimoto', lambda: tanimoto(bits, bad))
    return _threshold(S, cutoff), {
        'structure': 'ecfp4_tanimoto', 'cutoff': cutoff,
        'source': 'Hu, Furtmann, Stumpfe & Bajorath 2015, F1000Research 4:168',
        'reference_independent': True}


def sim_morgan_tanimoto_090(ctx, cutoff=GEOMETRY_MORGAN_TANIMOTO):
    """Morgan radius 2 Tanimoto at 0.90. Same fingerprint as above, the
    Geometry preprint's threshold rather than Bajorath's. `cutoff` is open so
    the sweep can move it; the default is the published one."""
    out, info = sim_ecfp4_tanimoto(ctx, cutoff=cutoff)
    info['source'] = ('Geometry of Activity Cliffs, arXiv:2605.30831, a '
                      'preprint; second-hand through the research session of '
                      '2026-11-01')
    return out, info


def sim_maccs_tanimoto(ctx, cutoff=MACCS_TANIMOTO):
    bits, bad = ctx.bits('maccs')
    S = ctx.sim('maccs_tanimoto', lambda: tanimoto(bits, bad))
    return _threshold(S, cutoff), {
        'structure': 'maccs_tanimoto', 'cutoff': cutoff, 'n_keys': 167,
        'source': '0.85 reported by the research session of 2026-11-01, no '
                  'paper read here',
        'reference_independent': True}


def sim_rdkit_dice(ctx, cutoff=0.90):
    bits, bad = ctx.bits('rdkit')
    S = ctx.sim('rdkit_dice', lambda: dice(bits, bad))
    return _threshold(S, cutoff), {
        'structure': 'rdkit_fingerprint_dice', 'cutoff': cutoff,
        'source': 'the Geometry preprint\'s widest setting; second-hand',
        'reference_independent': True}


def sim_ecfp4_dice(ctx, cutoff=0.90):
    bits, bad = ctx.bits('ecfp4')
    S = ctx.sim('ecfp4_dice', lambda: dice(bits, bad))
    return _threshold(S, cutoff), {
        'structure': 'ecfp4_dice', 'cutoff': cutoff,
        'reference_independent': True}


def sim_moleculeace(ctx, cutoff=MOLECULEACE_SIMILARITY):
    """Similar if ANY of three measures is at or above the cutoff.

    ECFP Tanimoto, Tanimoto of the Bemis-Murcko scaffold's ECFP, and scaled
    Levenshtein on the SMILES string. van Tilborg, Alenicheva & Grisoni 2022,
    J. Chem. Inf. Model. 62:5938. The union is the point: a scaffold hop that
    the whole-molecule fingerprint misses is caught by the scaffold measure,
    and a stereoisomer pair with one character of difference is caught by the
    string measure.
    """
    # 1,024 bits, which is their get_tanimoto_matrix and get_scaffold_matrix
    # default. Measured on CHEMBL214_Ki, 2,048 bits gives the same final cliff
    # set, so this is for exactness and not because the answer moved.
    ebits, ebad = ctx.bits('ecfp4', n_bits=1024)
    sbits, sbad = ctx.bits('scaffold', n_bits=1024)
    S1 = ctx.sim('ecfp4_tanimoto_1024', lambda: tanimoto(ebits, ebad))
    S2 = ctx.sim('generic_scaffold_tanimoto_1024', lambda: tanimoto(sbits, sbad))
    S3 = ctx.sim('smiles_levenshtein',
                 lambda: smiles_levenshtein_similarity(ctx.smiles))
    m1, m2, m3 = (_threshold(S, cutoff) for S in (S1, S2, S3))
    m = m1 | m2 | m3
    return m, {
        'structure': 'moleculeace_any_of_three', 'cutoff': cutoff,
        'n_pairs_ecfp4_only': int((m1 & ~m2 & ~m3).sum()) // 2,
        'n_pairs_scaffold_only': int((m2 & ~m1 & ~m3).sum()) // 2,
        'n_pairs_string_only': int((m3 & ~m1 & ~m2).sum()) // 2,
        'n_bits': 1024, 'scaffold': 'generic framework',
        'source': 'van Tilborg, Alenicheva & Grisoni 2022, JCIM 62:5938; '
                  'implementation checked against their '
                  'MoleculeACE/benchmark/cliffs.py and reproducing their '
                  'cliff_mol column exactly on CHEMBL214_Ki, 1,222 of 1,222 '
                  'molecules with none either way (2026-11-02)',
        'reference_independent': True}


def sim_embedding_cosine(ctx, rep_name, pct=1.0):
    """The top `pct` per cent of pairs by cosine similarity in one learned
    embedding.

    A threshold on a learned embedding's cosine has no published value, so the
    criterion is distribution-relative and says so. The Geometry preprint
    reports ChemBERTa's embeddings collapsing -- nearly every pair similar --
    which a percentile hides and a threshold would have shown; the spread of
    the cosine values is recorded here for that reason (second-hand through the
    research session of 2026-11-01).
    """
    X = ctx.embedding(rep_name)
    S = ctx.sim(f'{rep_name}_cosine', lambda: cosine_sim(X))
    m, cutoff = _top_percentile_pairs(S, pct)
    iu = np.triu_indices(ctx.n, k=1)
    vals = S[iu]
    return m, {
        'structure': f'{rep_name}_cosine_top_pairs', 'percentile': pct,
        'cutoff_reached': cutoff,
        'cosine_median': float(np.median(vals)) if vals.size else None,
        'cosine_p05': float(np.percentile(vals, 5)) if vals.size else None,
        'cosine_p95': float(np.percentile(vals, 95)) if vals.size else None,
        'reference_independent': False,
        'note': 'the cliffs are a function of a representation that may also '
                'be graded against them'}


#: The distances s12 runs the concatenation rule under. Read 2026-11-02 from
#: s12_cross_domain_cliffs.py:319, which yields one criterion per metric. Before
#: that date this module had Euclidean only, so a sweep over it would have
#: missed three of the four spaces s12 actually computes cliffs in.
CONCAT_METRICS = ('euclidean', 'cityblock', 'cosine', 'minkowski_p0.5')


def _concat_similarity(ctx, metric='euclidean'):
    """1/(1+d) on every representation z-scored and glued together.

    `metric` names a scipy distance, except `minkowski_p0.5`, which is
    Minkowski with p = 0.5 — one of the distances `distance_config.py` decided
    on and the one s12 keeps because it is not a true metric above p < 1.
    """
    def build():
        if metric == 'minkowski_p0.5':
            D = scipy_cdist(ctx.features, ctx.features, metric='minkowski', p=0.5)
        else:
            D = scipy_cdist(ctx.features, ctx.features, metric=metric)
        return _clean_sim(1.0 / (1.0 + D))
    return ctx.sim(f'all_features_inv_{metric}', build)


def sim_top_pairs_all_features(ctx, pct=1.0, metric='euclidean'):
    """The top `pct` per cent of pairs by 1/(1+d) on every representation
    z-scored and glued together. n01's and s12's fallback where no ECFP4 exists.

    s12's own docstring (`:159`) records why an absolute cutoff is not used
    here, measured by an earlier session and not re-run: on standardised
    high-dimensional representations the pairwise distance is large, so 1/(1+d)
    never reaches 0.55, and s12 found no cliffs at all in 13 of 16 datasets.
    """
    S = _concat_similarity(ctx, metric)
    m, cutoff = _top_percentile_pairs(S, pct)
    return m, {
        'structure': f'all_representations_inverse_{metric}_top_pairs',
        'metric': metric, 'percentile': pct, 'cutoff_reached': cutoff,
        'reference_independent': False,
        'note': 'the cliffs are a function of the representations being graded'}


def sim_ecfp4_top_pairs(ctx, pct=2.0):
    """The top `pct` per cent of pairs by ECFP4 Tanimoto, with no absolute
    threshold. n01.SIMILAR_PERCENTILE is 2.0 and
    similarity_helpers.CLIFF_SIMILARITY_PERCENTILE is 1.0."""
    bits, bad = ctx.bits('ecfp4')
    S = ctx.sim('ecfp4_tanimoto', lambda: tanimoto(bits, bad))
    m, cutoff = _top_percentile_pairs(S, pct)
    return m, {'structure': 'ecfp4_tanimoto_top_pairs', 'percentile': pct,
               'cutoff_reached': cutoff, 'reference_independent': True}


def sim_ecfp4_knn(ctx, k=KNN_K):
    bits, bad = ctx.bits('ecfp4')
    S = ctx.sim('ecfp4_tanimoto', lambda: tanimoto(bits, bad))
    return _knn_pairs(S, k), {
        'structure': 'ecfp4_tanimoto_k_nearest', 'k': int(k),
        'reference_independent': True,
        'note': 'no similarity threshold; every molecule has k partners'}


def sim_detector_distance_percentile(ctx, pct=DETECTOR_DISTANCE_PERCENTILE):
    """The closest `pct` per cent of pairwise distances on the glued features.

    ActivityCliffDetector's default (noise_mitigation_methods.py). The class is
    in the repo and no experiment calls it.
    """
    S = _concat_similarity(ctx, 'euclidean')
    m, cutoff = _top_percentile_pairs(S, pct)
    return m, {'structure': 'ActivityCliffDetector_distance_percentile',
               'percentile': pct, 'cutoff_reached': cutoff,
               'reference_independent': False}


def sim_mmp(ctx):
    """The loader's matched molecular pairs, where it ships them.

    Dablander's three ChEMBL targets ship pair lists; nothing else here does,
    and whether mmpdb can be run in this environment is not checked.
    """
    pairs = ctx.mmp_pairs
    m = np.zeros((ctx.n, ctx.n), dtype=bool)
    if pairs is None or len(pairs) == 0:
        return m, {'structure': 'matched_molecular_pairs', 'available': False,
                   'reason': 'the loader shipped no pair list'}
    arr = np.asarray(pairs)
    i, j = arr[:, 0].astype(int), arr[:, 1].astype(int)
    keep = (i < ctx.n) & (j < ctx.n)
    m[i[keep], j[keep]] = True
    m |= m.T
    np.fill_diagonal(m, False)
    return m, {'structure': 'matched_molecular_pairs', 'available': True,
               'n_pairs_given': int(len(arr)),
               'n_pairs_in_range': int(keep.sum()),
               'reference_independent': True}


#: The edit counts s01 and s02 already sweep, read 2026-11-02 from
#: similarity_helpers.run_antibody_cliff_analysis (`:1512`). Before that date
#: this module held EDIT_DISTANCE_MAX = 5 alone. This is the only similarity
#: axis anywhere in the suite that was already swept over more than one point.
EDIT_DISTANCE_SWEEP = (1, 2, 3, 5)


def cdr_sequences(sequences, dataset_name=None, report=None):
    """The CDR-only strings for a set of antibody heavy chains, or None.

    The second of the two scopes s01 and s02 run. The extraction is ANARCI's,
    reached through similarity_helpers so there is one implementation, and it
    caches per dataset because ANARCI takes seconds a sequence.

    ⛔ WHY THIS CHECKS THE RESULT, found 2026-11-02. Three functions in a row
    fall back to the whole sequence rather than failing, and none of them says
    so to its caller. `antibody.get_cdr_residue_indices` returns every index
    when ANARCI cannot number the sequence, when ANARCI is not importable, and
    when the numbering lands on no CDR position. `_extract_cdr_sequences` then
    counts a sequence as failed only if that call raised or returned nothing,
    so its own "more than half failed, give up" guard never fires. Measured:
    ANARCI is installed here, and on a random 21-residue peptide
    `get_cdr_residue_indices` returned all 21 indices, so the CDR scope came
    back equal to the full chain for every sequence while reporting success.
    This is right for the pooled representations that function was written for
    and wrong for a cliff rule, where it would report a second scope that is a
    copy of the first. So: count the strings that came back unchanged, put the
    count in `report`, and return None when every one did.
    """
    if sequences is None or len(sequences) == 0:
        return None
    try:
        from experiments.similarity.similarity_helpers import (
            _extract_cdr_sequences)
    except Exception:
        return None
    try:
        out = _extract_cdr_sequences(list(sequences), dataset_name=dataset_name)
    except Exception:
        return None
    if out is None or len(out) != len(sequences):
        return None
    unchanged = sum(1 for a, b in zip(out, sequences) if str(a) == str(b))
    if report is not None:
        report['n_unchanged'] = int(unchanged)
        report['n_sequences'] = int(len(out))
    if unchanged == len(out):
        return None
    return out


def sim_sequence_edit(ctx, max_edits=EDIT_DISTANCE_MAX, scope='full'):
    """Within `max_edits` edits of each other, at the full chain or the CDRs.

    `scope='cdr'` is the CDR-only scope of s01 and s02: the same edit count
    taken on the ANARCI-extracted CDR1+CDR2+CDR3 string instead of the whole
    V-region, so two antibodies differing only in their framework are similar
    under it and need not be under the full chain.
    """
    if scope == 'cdr':
        seqs = ctx.cdr_sequences
        if seqs is None:
            return np.zeros((ctx.n, ctx.n), dtype=bool), {
                'structure': 'sequence_edit_distance_cdr', 'available': False,
                'reason': 'no CDRs were extracted: no sequences, ANARCI not '
                          'installed, or every string came back equal to the '
                          'full chain',
                'cdr_report': dict(ctx.cdr_report)}
        D, kind = ctx.sim('sequence_edit_cdr',
                          lambda: sequence_edit_distance(seqs))
    else:
        D, kind = ctx.sim('sequence_edit',
                          lambda: sequence_edit_distance(ctx.sequences))
    m = D <= max_edits
    np.fill_diagonal(m, False)
    info = {'structure': f'sequence_{kind}_distance',
            'scope': 'cdr_only' if scope == 'cdr' else 'full_V_region',
            'max_edits': max_edits, 'available': True,
            'reference_independent': True}
    if scope == 'cdr':
        info['cdr_report'] = dict(ctx.cdr_report)
    return m, info


def sim_sequence_edit_percentile(ctx, pct=EDIT_DISTANCE_PERCENTILE):
    """The closest `pct` per cent of nonzero pairwise edit distances.

    similarity_helpers records why the absolute cutoff is not enough on its
    own: on a deep mutational scan an absolute 5 edits calls a huge share of
    pairs similar.
    """
    D, kind = ctx.sim('sequence_edit',
                      lambda: sequence_edit_distance(ctx.sequences))
    iu = np.triu_indices(ctx.n, k=1)
    vals = D[iu]
    vals = vals[vals > 0]
    if vals.size == 0:
        return np.zeros((ctx.n, ctx.n), dtype=bool), {
            'structure': f'sequence_{kind}_distance_percentile',
            'available': False, 'reason': 'every pairwise distance is zero'}
    cutoff = float(np.percentile(vals, pct))
    m = (D <= cutoff) & (D > 0)
    np.fill_diagonal(m, False)
    return m, {'structure': f'sequence_{kind}_distance_percentile',
               'percentile': pct, 'cutoff_reached': cutoff,
               'reference_independent': True}


SIMILARITY_CRITERIA = {
    # absolute thresholds on a fingerprint
    'ecfp4_tanimoto_055': sim_ecfp4_tanimoto,
    'ecfp4_tanimoto_090': lambda ctx: sim_ecfp4_tanimoto(ctx, cutoff=0.90),
    'morgan_tanimoto_090': sim_morgan_tanimoto_090,
    'maccs_tanimoto_085': sim_maccs_tanimoto,
    'ecfp4_dice_090': sim_ecfp4_dice,
    'rdkit_dice_090': sim_rdkit_dice,
    'rdkit_dice_070': lambda ctx: sim_rdkit_dice(ctx, cutoff=0.70),
    'moleculeace_any_090': sim_moleculeace,
    # distribution-relative, on a fingerprint
    'ecfp4_top_0.5pct': lambda ctx: sim_ecfp4_top_pairs(ctx, pct=0.5),
    'ecfp4_top_1pct': lambda ctx: sim_ecfp4_top_pairs(ctx, pct=1.0),
    'ecfp4_top_2pct': lambda ctx: sim_ecfp4_top_pairs(ctx, pct=2.0),
    'ecfp4_knn_10': sim_ecfp4_knn,
    # distribution-relative, on the representations themselves. One entry per
    # distance s12 runs (added 2026-11-02; Euclidean was the only one here).
    'all_features_top_0.5pct': lambda ctx: sim_top_pairs_all_features(ctx, pct=0.5),
    'all_features_top_1pct': lambda ctx: sim_top_pairs_all_features(ctx, pct=1.0),
    'all_features_cityblock_top_1pct':
        lambda ctx: sim_top_pairs_all_features(ctx, pct=1.0, metric='cityblock'),
    'all_features_cosine_top_1pct':
        lambda ctx: sim_top_pairs_all_features(ctx, pct=1.0, metric='cosine'),
    'all_features_minkowski05_top_1pct':
        lambda ctx: sim_top_pairs_all_features(ctx, pct=1.0,
                                               metric='minkowski_p0.5'),
    'detector_closest_10pct': sim_detector_distance_percentile,
    # learned embeddings
    'chemberta_cosine_top_1pct': lambda ctx: sim_embedding_cosine(ctx, 'chemberta'),
    'molformer_cosine_top_1pct': lambda ctx: sim_embedding_cosine(ctx, 'molformer'),
    'chemeleon_cosine_top_1pct': lambda ctx: sim_embedding_cosine(ctx, 'chemeleon'),
    # pair lists and sequences
    'matched_molecular_pairs': sim_mmp,
    # the edit sweep and the CDR scope s01 and s02 already run (2026-11-02;
    # 5 edits at the full chain was the only one here)
    'sequence_edit_1': lambda ctx: sim_sequence_edit(ctx, max_edits=1),
    'sequence_edit_2': lambda ctx: sim_sequence_edit(ctx, max_edits=2),
    'sequence_edit_3': lambda ctx: sim_sequence_edit(ctx, max_edits=3),
    'sequence_edit_5': sim_sequence_edit,
    'sequence_cdr_edit_1': lambda ctx: sim_sequence_edit(ctx, max_edits=1,
                                                         scope='cdr'),
    'sequence_cdr_edit_2': lambda ctx: sim_sequence_edit(ctx, max_edits=2,
                                                         scope='cdr'),
    'sequence_cdr_edit_3': lambda ctx: sim_sequence_edit(ctx, max_edits=3,
                                                         scope='cdr'),
    'sequence_cdr_edit_5': lambda ctx: sim_sequence_edit(ctx, max_edits=5,
                                                         scope='cdr'),
    'sequence_edit_top_5pct': sim_sequence_edit_percentile,
}

# Which criteria a dataset can even be asked for. A criterion that needs
# something the dataset does not have is left out rather than approximated.
SIMILARITY_NEEDS = {
    'ecfp4_tanimoto_055': 'smiles', 'ecfp4_tanimoto_090': 'smiles',
    'morgan_tanimoto_090': 'smiles', 'maccs_tanimoto_085': 'smiles',
    'ecfp4_dice_090': 'smiles', 'rdkit_dice_090': 'smiles',
    'rdkit_dice_070': 'smiles', 'moleculeace_any_090': 'smiles',
    'ecfp4_top_0.5pct': 'smiles', 'ecfp4_top_1pct': 'smiles',
    'ecfp4_top_2pct': 'smiles', 'ecfp4_knn_10': 'smiles',
    'all_features_top_0.5pct': 'reps', 'all_features_top_1pct': 'reps',
    'all_features_cityblock_top_1pct': 'reps',
    'all_features_cosine_top_1pct': 'reps',
    'all_features_minkowski05_top_1pct': 'reps',
    'detector_closest_10pct': 'reps',
    'chemberta_cosine_top_1pct': 'rep:chemberta',
    'molformer_cosine_top_1pct': 'rep:molformer',
    'chemeleon_cosine_top_1pct': 'rep:chemeleon',
    'matched_molecular_pairs': 'mmp_pairs',
    'sequence_edit_1': 'sequences', 'sequence_edit_2': 'sequences',
    'sequence_edit_3': 'sequences', 'sequence_edit_5': 'sequences',
    'sequence_edit_top_5pct': 'sequences',
    'sequence_cdr_edit_1': 'cdr', 'sequence_cdr_edit_2': 'cdr',
    'sequence_cdr_edit_3': 'cdr', 'sequence_cdr_edit_5': 'cdr',
}


def available_similarity_criteria(ctx):
    """The criterion names this dataset can run, and why each other is out."""
    ok, out = [], {}
    for name, need in SIMILARITY_NEEDS.items():
        if need == 'smiles' and ctx.smiles is None:
            out[name] = 'no SMILES'
        elif need == 'sequences' and ctx.sequences is None:
            out[name] = 'no sequences'
        elif need == 'cdr' and ctx.cdr_sequences is None:
            out[name] = ('no CDRs: either no sequences, or ANARCI is not '
                         'installed, or extraction failed')
        elif need == 'reps' and not ctx.reps:
            out[name] = 'no representations loaded'
        elif need == 'mmp_pairs' and not ctx.mmp_pairs:
            out[name] = 'the loader shipped no matched pairs'
        elif need.startswith('rep:') and need[4:] not in ctx.reps:
            out[name] = f'{need[4:]} was not loaded'
        else:
            ok.append(name)
    return ok, out


# ---------------------------------------------------------------------------
# Gap criteria. Each returns (n x n boolean "far apart", info).
# ---------------------------------------------------------------------------

def pairwise_gap(y, is_clf=False):
    """|y_i - y_j|, or the class-mismatch indicator on a classification task."""
    y = np.asarray(y, dtype=float).ravel()
    if is_clf:
        return (y[:, None] != y[None, :]).astype(float)
    return np.abs(y[:, None] - y[None, :])


def _far(gap, cutoff):
    m = gap > cutoff
    np.fill_diagonal(m, False)
    return m


def gap_mad_multiple(ctx, k=MAD_MULTIPLE, similar=None):
    """k times the median absolute deviation of this dataset's own labels.

    n01's, n25's, n35's and thesis.tex:3675's gap. Defined whatever units the
    label is in, which is why it is the one that runs on caco2 -- a 2-log-unit
    gap flags no caco2 molecule at all, its median absolute deviation being
    0.2085 (similarity_helpers block comment). The cost is that the share of
    molecules it flags is whatever the label distribution gives: 75.3 per cent
    of caco2, 31.9 per cent of herg_ki, 4.7 per cent of adme_mdr1, measured
    2026-09-19 at Tanimoto 0.55.
    """
    y = ctx.labels
    if ctx.is_clf:
        return _far(ctx.gap_matrix, 0.5), {
            'gap': 'class_mismatch', 'cutoff': 0.5}
    mad = float(np.median(np.abs(y - np.median(y))))
    cutoff = k * mad
    return _far(ctx.gap_matrix, cutoff), {
        'gap': 'median_absolute_deviation_multiple', 'k': k,
        'median_absolute_deviation': mad, 'cutoff_label_units': cutoff,
        'source': 'n01.identify_cliff_pairs'}


def gap_absolute(ctx, cutoff=ABSOLUTE_LOG_UNITS, similar=None):
    """A fixed gap in the label's own units. 2.0 is 100-fold potency.

    Only a potency statement where the label is a log potency. On any other
    regression target it is a fixed 2-unit label gap and n14 records that in
    the result file rather than in a comment.
    """
    if ctx.is_clf:
        return _far(ctx.gap_matrix, 0.5), {
            'gap': 'class_mismatch', 'cutoff': 0.5}
    y = ctx.labels
    mad = float(np.median(np.abs(y - np.median(y))))
    return _far(ctx.gap_matrix, cutoff), {
        'gap': 'absolute_label_units', 'cutoff_label_units': cutoff,
        'cutoff_in_median_absolute_deviations': (
            float(cutoff / mad) if mad > 0 else None),
        'source': 'n14.AC_POTENCY_THRESHOLD; Dablander et al. 2023, '
                  'J. Cheminform. 15:47'}


def gap_absolute_half_cliffs_dropped(ctx, band=HALF_CLIFF_BAND, similar=None):
    """Above the band is a cliff; inside the band is neither.

    Dablander et al. 2023 drop the 1-to-2-log-unit pairs from both sides
    rather than calling them non-cliffs, which is what
    opig_activity_cliff_qsar.py:379 does. The returned mask is the cliff side;
    info carries how many pairs were dropped, because a molecule-level flag
    cannot show them.
    """
    lo, hi = band
    if ctx.is_clf:
        return _far(ctx.gap_matrix, 0.5), {
            'gap': 'class_mismatch', 'cutoff': 0.5}
    gap = ctx.gap_matrix
    cliff = _far(gap, hi)
    half = (gap > lo) & (gap <= hi)
    np.fill_diagonal(half, False)
    mask = half if similar is None else (half & similar)
    return cliff, {
        'gap': 'absolute_with_half_cliff_band_dropped',
        'cutoff_label_units': hi, 'half_cliff_band': [lo, hi],
        'n_half_cliff_pairs_dropped': int(mask.sum()) // 2,
        'source': 'Dablander et al. 2023; opig_activity_cliff_qsar.py:379'}


def gap_solved_to_target(ctx, target_pct=SOLVED_TARGET_PCT, similar=None,
                         n_steps=80):
    """The gap whose achieved share of cliff molecules is NEAREST target_pct.

    It is not a gap that reaches the target. The share falls in steps as the
    gap rises, and where few pairs are similar at all it misses by a lot: on
    QM9 only 1,093 pairs of 9,978 molecules are similar (thesis.tex:1120), and
    the achieved share there is 19 of 7,982 training molecules
    (thesis.tex:4148) against a 15 per cent target. info carries
    achieved_pct and every caller must report that, never the target.

    Needs `similar`, because the share depends on both halves of the rule.
    """
    if similar is None:
        raise ValueError('the solved gap depends on the similarity criterion; '
                         'pass the similar mask')
    y = ctx.labels
    if ctx.is_clf:
        m = _far(ctx.gap_matrix, 0.5)
        achieved = 100.0 * float((m & similar).any(axis=1).sum()) / ctx.n
        return m, {'gap': 'class_mismatch', 'cutoff': 0.5,
                   'target_pct': None, 'achieved_pct': achieved}
    gap = ctx.gap_matrix
    mad = float(np.median(np.abs(y - np.median(y))))

    def share(cut):
        m = similar & (gap > cut)
        np.fill_diagonal(m, False)
        return 100.0 * float(m.any(axis=1).sum()) / ctx.n

    lo, hi = 0.0, float(gap.max()) if ctx.n else 1.0
    best = (lo, share(lo))
    for _ in range(n_steps):
        mid = 0.5 * (lo + hi)
        p = share(mid)
        if abs(p - target_pct) < abs(best[1] - target_pct):
            best = (mid, p)
        if p > target_pct:
            lo = mid
        else:
            hi = mid
    cut, achieved = float(best[0]), float(best[1])
    return _far(gap, cut), {
        'gap': 'solved_to_target_share', 'target_pct': target_pct,
        'achieved_pct': achieved, 'cutoff_label_units': cut,
        'cutoff_in_median_absolute_deviations': (
            float(cut / mad) if mad > 0 else None),
        'reached_target': bool(abs(achieved - target_pct) < 1.0),
        'source': 'similarity_helpers.solve_activity_gap'}


def gap_percentile_of_differences(ctx, pct=DETECTOR_GAP_PERCENTILE,
                                  similar=None):
    """The `pct`th percentile of all pairwise label differences.

    ActivityCliffDetector's default gap. Distribution-relative on the label
    side as well as the structural side, so neither half of the rule carries a
    unit.
    """
    gap = ctx.gap_matrix
    iu = np.triu_indices(ctx.n, k=1)
    vals = gap[iu]
    cutoff = float(np.percentile(vals, pct)) if vals.size else 0.0
    return _far(gap, cutoff), {
        'gap': 'percentile_of_pairwise_differences', 'percentile': pct,
        'cutoff_label_units': cutoff,
        'source': 'ActivityCliffDetector default'}


def gap_mean_plus_sd(ctx, n_sd=2.0, similar=None):
    """The mean pairwise label difference plus n_sd of its standard deviation.

    Attributed to the Bajorath group by the research session of 2026-11-01 and
    NOT verified against a paper here.
    """
    gap = ctx.gap_matrix
    iu = np.triu_indices(ctx.n, k=1)
    vals = gap[iu]
    cutoff = float(vals.mean() + n_sd * vals.std()) if vals.size else 0.0
    return _far(gap, cutoff), {
        'gap': 'mean_plus_n_standard_deviations', 'n_sd': n_sd,
        'cutoff_label_units': cutoff,
        'source': 'attributed to the Bajorath group by the research session '
                  'of 2026-11-01; unverified'}


def gap_noise_floor(ctx, floor=KRAMER_NOISE_FLOOR_PKI, similar=None):
    """A gap that clears the measurement-noise floor and nothing more.

    Kramer et al. 2012's standard deviation of the difference between two
    independent public Ki measurements of one pair, 0.54 pKi. In pKi, so it is
    a label-unit statement and only meaningful on a pKi-like label. As a cliff
    rule it is the weakest gap that is not inside the noise; the census reports
    it so every other rule's share of pairs inside the floor can be read.
    """
    return _far(ctx.gap_matrix, floor), {
        'gap': 'kramer_noise_floor', 'cutoff_label_units': floor,
        'source': 'Kramer, Kalliokoski, Gedeck & Vulpetti 2012, '
                  'J. Med. Chem. 55:5165'}


GAP_CRITERIA = {
    'mad_2x': gap_mad_multiple,
    'mad_1x': lambda ctx, similar=None: gap_mad_multiple(ctx, k=1.0, similar=similar),
    'mad_3x': lambda ctx, similar=None: gap_mad_multiple(ctx, k=3.0, similar=similar),
    'absolute_2': gap_absolute,
    'absolute_1': lambda ctx, similar=None: gap_absolute(
        ctx, cutoff=MOLECULEACE_GAP_LOG_UNITS, similar=similar),
    'absolute_2_half_cliffs_dropped': gap_absolute_half_cliffs_dropped,
    'solved_target_15pct': gap_solved_to_target,
    'percentile_90_of_differences': gap_percentile_of_differences,
    'mean_plus_2sd': gap_mean_plus_sd,
    'kramer_noise_floor': gap_noise_floor,
}

# Gap criteria that are a statement in the label's own units, so they are only
# interpretable on a log-potency label. Recorded rather than enforced: caco2's
# label is log permeability and an absolute 2 flags nothing there, which is a
# number worth having in the census rather than a skip.
GAP_NEEDS_LOG_POTENCY = frozenset({
    'absolute_2', 'absolute_1', 'absolute_2_half_cliffs_dropped',
    'kramer_noise_floor',
})


# ---------------------------------------------------------------------------
# Named combinations that come from a paper
# ---------------------------------------------------------------------------

RULES = {
    'hu2015': ('ecfp4_tanimoto_055', 'absolute_2'),
    'n01': ('ecfp4_tanimoto_055', 'mad_2x'),
    'calibrated': ('ecfp4_tanimoto_055', 'solved_target_15pct'),
    'n14': ('ecfp4_tanimoto_055', 'absolute_2'),
    'moleculeace': ('moleculeace_any_090', 'absolute_1'),
    'dablander': ('matched_molecular_pairs', 'absolute_2_half_cliffs_dropped'),
    'geometry_morgan': ('morgan_tanimoto_090', 'absolute_2'),
    'geometry_rdkit_dice': ('rdkit_dice_090', 'absolute_2'),
    'activity_cliff_detector': ('detector_closest_10pct',
                                'percentile_90_of_differences'),
    's12': ('all_features_top_0.5pct', 'mad_2x'),
}

# hu2015 and n14 are the same pair of criteria. They are both named because
# n14's result file cites the threshold as its own and a reader should be able
# to ask for either name and get the same numbers.


# ---------------------------------------------------------------------------
# Applying a rule
# ---------------------------------------------------------------------------

def cliff_sets(ctx, similarity, gap, drop_same_molecule=True):
    """Both units of one rule, and everything needed to read the counts.

    Returns a dict with:
      pair_mask         n x n boolean, the cliff pairs
      molecule_mask     n boolean, in at least one cliff pair
      n_cliff_pairs, n_cliff_molecules, pct_cliff_molecules
      n_similar_pairs            how many pairs the similarity side passed
      pct_cliff_pairs_of_similar the share of those that the gap also passed
      n_pairs_inside_noise_floor how many cliff pairs have a gap under 0.54
      similarity_info, gap_info  the two criteria's own records
    """
    # A name from the registries, or the criterion itself. The sweep passes
    # callables, because a grid point is one cutoff of a criterion and
    # registering 89 of them globally would change every other caller's idea
    # of what criteria exist.
    sim_fn = (SIMILARITY_CRITERIA[similarity] if isinstance(similarity, str)
              else similarity)
    gap_name = gap if isinstance(gap, str) else getattr(gap, 'label', str(gap))
    similarity = (similarity if isinstance(similarity, str)
                  else getattr(sim_fn, 'label', str(sim_fn)))
    similar, sinfo = sim_fn(ctx)
    similar = np.asarray(similar, dtype=bool).copy()
    np.fill_diagonal(similar, False)
    dropped_repeats = 0
    if drop_same_molecule:
        same = ctx.same_molecule
        if same.any():
            dropped_repeats = int((similar & same).sum()) // 2
            similar &= ~same

    gap_fn = GAP_CRITERIA[gap] if isinstance(gap, str) else gap
    far, ginfo = gap_fn(ctx, similar=similar)
    far = np.asarray(far, dtype=bool)

    pair_mask = similar & far
    np.fill_diagonal(pair_mask, False)
    molecule_mask = pair_mask.any(axis=1)

    n_similar = int(similar.sum()) // 2
    n_pairs = int(pair_mask.sum()) // 2
    gaps = ctx.gap_matrix
    inside_floor = int((pair_mask & (gaps <= KRAMER_NOISE_FLOOR_PKI)).sum()) // 2

    return {
        'similarity': similarity, 'gap': gap_name,
        'pair_mask': pair_mask, 'molecule_mask': molecule_mask,
        'n_molecules': int(ctx.n),
        'n_similar_pairs': n_similar,
        'n_cliff_pairs': n_pairs,
        'n_cliff_molecules': int(molecule_mask.sum()),
        'pct_cliff_molecules': (100.0 * float(molecule_mask.sum()) / ctx.n
                                if ctx.n else 0.0),
        'pct_cliff_pairs_of_similar': (100.0 * n_pairs / n_similar
                                       if n_similar else None),
        'n_cliff_pairs_inside_noise_floor': inside_floor,
        'pct_cliff_pairs_inside_noise_floor': (100.0 * inside_floor / n_pairs
                                               if n_pairs else None),
        'n_similar_pairs_dropped_as_own_repeats': dropped_repeats,
        'similarity_info': sinfo, 'gap_info': ginfo,
    }


def cliff_mask(ctx, rule='n01'):
    """The per-molecule flag under a named rule from RULES. One call for an
    experiment that wants one rule and does not want to know the parts."""
    if rule not in RULES:
        raise KeyError(f'unknown rule {rule!r}. Known: {sorted(RULES)}')
    sim, gap = RULES[rule]
    out = cliff_sets(ctx, sim, gap)
    out['rule'] = rule
    return out['molecule_mask'], out


# ---------------------------------------------------------------------------
# Stereoisomer pairs: the kind the Geometry preprint names as the one
# fingerprints miss
# ---------------------------------------------------------------------------

def sim_stereoisomer_pairs(ctx):
    """A pair is similar if the two molecules differ ONLY in stereochemistry.

    Both canonical SMILES with stereochemistry stripped are identical, and the
    two canonical SMILES with stereochemistry kept are not. That is the whole
    criterion -- it has no threshold, so nothing is tuned.

    The Geometry preprint reports that none of its 15 combinations of embedding
    and distance was sensitive to both stereochemistry cliffs and matched-pair
    cliffs, and that MolFormer under cosine was the only one with meaningful
    stereochemical sensitivity (second-hand through the research session of
    2026-11-01). A fingerprint cannot see this pair at all: ECFP4 of two
    enantiomers is bit-identical, so their Tanimoto is 1.0 and no threshold
    separates them from a duplicate.
    """
    from rdkit import Chem
    mols = ctx.mols
    flat, full, bad = [], [], []
    for mol in mols:
        if mol is None:
            flat.append(None)
            full.append(None)
            bad.append(True)
            continue
        bad.append(False)
        full.append(Chem.MolToSmiles(mol))
        stripped = Chem.Mol(mol)
        Chem.RemoveStereochemistry(stripped)
        flat.append(Chem.MolToSmiles(stripped))

    groups = {}
    for i, key in enumerate(flat):
        if key is not None:
            groups.setdefault(key, []).append(i)

    m = np.zeros((ctx.n, ctx.n), dtype=bool)
    n_identical = 0
    for rows in groups.values():
        if len(rows) < 2:
            continue
        for a_i in range(len(rows)):
            for b_i in range(a_i + 1, len(rows)):
                i, j = rows[a_i], rows[b_i]
                if full[i] == full[j]:
                    n_identical += 1
                    continue
                m[i, j] = m[j, i] = True
    np.fill_diagonal(m, False)
    return m, {
        'structure': 'stereoisomer_pairs',
        'n_pairs': int(m.sum()) // 2,
        'n_pairs_dropped_as_identical_molecules': n_identical,
        'n_unparseable': int(sum(bad)),
        'reference_independent': True,
        'note': 'no threshold; ECFP4 of two enantiomers is bit-identical so '
                'no fingerprint criterion can reach these pairs'}


SIMILARITY_CRITERIA['stereoisomer_pairs'] = sim_stereoisomer_pairs
SIMILARITY_NEEDS['stereoisomer_pairs'] = 'smiles'


# ---------------------------------------------------------------------------
# Matched molecular pairs: what is and is not reachable here
# ---------------------------------------------------------------------------

def mmp_state():
    """Whether matched pairs can be computed in this environment, and how far.

    sim_mmp reads a pair list the loader shipped. Dablander's three ChEMBL
    targets ship one; nothing else in the suite does. Generating pairs needs
    mmpdb, and the transformation-size restriction the literature applies --
    a pair counts only when the exchanged fragment is small relative to the
    molecule -- is a parameter of that generation step and cannot be recovered
    from a pair list that was generated without it.
    """
    try:
        import mmpdb  # noqa: F401
        installed = True
    except Exception:
        installed = False
    return {
        'mmpdb_installed': installed,
        'can_generate_pairs': installed,
        'transformation_size_restriction': (
            'a parameter of mmpdb fragment generation; not recoverable from a '
            'shipped pair list'),
        'pair_lists_on_disk': 'Dablander et al. 2023, three ChEMBL targets, '
                              'via chembl_qsar.py',
    }


# ---------------------------------------------------------------------------
# MoleculeACE's own cliff labels: the only cliff set nobody here chose
# ---------------------------------------------------------------------------

def moleculeace_cliff_labels(csv_path):
    """(smiles, y, cliff_mol) from one MoleculeACE benchmark CSV.

    build_moleculeace.py reads `smiles` and `y [pEC50/pKi]` and discards
    `cliff_mol`, which is the paper's own per-molecule cliff flag. It is the
    only labelled cliff set in reach that no rule in this repo produced, so it
    is the only external check a rule here can be scored against.

    Columns, as build_moleculeace.py records them: smiles, exp_mean [nM], y,
    cliff_mol, split, y [pEC50/pKi].
    """
    import csv
    smiles, y, cliff = [], [], []
    with open(csv_path, newline='') as fh:
        for row in csv.DictReader(fh):
            smiles.append(row['smiles'])
            y.append(float(row['y [pEC50/pKi]']))
            flag = str(row.get('cliff_mol', '')).strip().lower()
            cliff.append(flag in ('1', '1.0', 'true', 'yes'))
    return smiles, np.asarray(y, dtype=float), np.asarray(cliff, dtype=bool)


def moleculeace_state():
    """Where the 30 targets' CSVs are, or why they are not reachable."""
    import glob
    import os
    try:
        import MoleculeACE
        base = os.path.dirname(MoleculeACE.__file__)
        hits = sorted(glob.glob(os.path.join(base, '**', 'benchmark_data',
                                             '*.csv'), recursive=True))
        if not hits:
            hits = sorted(glob.glob(os.path.join(base, '**', 'CHEMBL*.csv'),
                                    recursive=True))
        return {'installed': True, 'n_csvs': len(hits), 'paths': hits[:3],
                'base': base}
    except Exception as exc:
        env = os.environ.get('MOLECULEACE_DATA', '')
        hits = (sorted(glob.glob(os.path.join(env, '**', '*.csv'),
                                 recursive=True)) if env else [])
        return {'installed': False, 'reason': f'{type(exc).__name__}: {exc}',
                'MOLECULEACE_DATA': env or None, 'n_csvs': len(hits)}


def score_against_labels(our_mask, their_mask):
    """How one rule's cliff molecules line up with a labelled cliff set.

    Both counts and the Matthews correlation. The counts are there because a
    correlation on its own hides which way a disagreement runs, and these two
    sets can be very different sizes.
    """
    a = np.asarray(our_mask, dtype=bool)
    b = np.asarray(their_mask, dtype=bool)
    tp = int((a & b).sum())
    fp = int((a & ~b).sum())
    fn = int((~a & b).sum())
    tn = int((~a & ~b).sum())
    den = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn - fp * fn) / den) if den > 0 else None
    return {
        'n_molecules': int(a.size),
        'n_ours': tp + fp, 'n_theirs': tp + fn,
        'both': tp, 'ours_only': fp, 'theirs_only': fn, 'neither': tn,
        'recall_of_their_cliffs': (tp / (tp + fn)) if (tp + fn) else None,
        'share_of_ours_in_theirs': (tp / (tp + fp)) if (tp + fp) else None,
        'matthews_correlation': mcc}


# ---------------------------------------------------------------------------
# Continuous scores: a number per molecule or per dataset instead of a flag
# ---------------------------------------------------------------------------
#
# HOW FAR EACH OF THESE IS VERIFIED. Four of the six have a formula this file
# can state exactly. Two do not, and say so in their own docstring rather than
# quietly picking one:
#
#   SALI               exact. Guha & Van Drie 2008. Already in the repo as a
#                      model feature (representations/molecular.py:1467).
#   local roughness    exact, and lifted from the repo's own implementation
#                      (similarity_metrics/nary_similarity.local_roughness).
#   SAS map            exact once the two thresholds are named. The quadrant
#                      NAMES vary between papers, so the counts are returned
#                      and the names are not.
#   rivality index     exact: RI_i = (d_same - d_other) / (d_same + d_other).
#   SARI               the discontinuity half is stated in the secondary
#                      literature; the continuity half and the standardisation
#                      step are NOT, and this file's version of them is
#                      flagged in sari()'s docstring.
#   ROGI               the integral and the clustering are stated; the
#                      dispersion step is this file's reading of that
#                      statement, because the rogi package is not installed.

def sali(sim, y, eps=1e-3):
    """n x n SALI: |y_i - y_j| / (1 - similarity).

    Guha & Van Drie 2008, J. Chem. Inf. Model. 48:646. The denominator is
    clipped at `eps` so a pair at similarity 1.0 is a large finite number
    rather than infinity; molecular.py:1575 clips at 1e-3 and this matches it.
    """
    sim = np.asarray(sim, dtype=np.float64)
    gap = np.abs(np.asarray(y, dtype=float)[:, None]
                 - np.asarray(y, dtype=float)[None, :])
    out = gap / np.clip(1.0 - sim, eps, None)
    np.fill_diagonal(out, 0.0)
    return out


def sali_clipped_pairs(sim, eps=1e-3):
    """How many pairs hit SALI's clip, which is where the score stops meaning
    anything.

    A pair at similarity 1.0 -- two enantiomers, two tautomers written
    differently, or two rows of one molecule -- divides by `eps` and returns a
    number three orders of magnitude above the rest. Measured on a toy set of
    15 molecules containing two stereoisomer pairs, the largest SALI was 5,054
    against a median of 6. So the maximum SALI over a dataset reports the
    clip, not the landscape, wherever such a pair exists.
    """
    sim = np.asarray(sim, dtype=np.float64)
    hit = (1.0 - sim) < eps
    np.fill_diagonal(hit, False)
    return {'eps': eps, 'n_pairs_at_the_clip': int(hit.sum()) // 2,
            'n_molecules_in_a_clipped_pair': int(hit.any(axis=1).sum())}


def sali_per_molecule(sim, y, how='max'):
    """One SALI number per molecule: the max or the mean over its partners.

    molecular.py's sali_cliff representation takes the max over the k=10
    nearest neighbours at Tanimoto 0.5 and calls it local_sali. The max over
    every partner is the version here; which of the two a result used has to
    be recorded, because they are not the same number.
    """
    S = sali(sim, y)
    return S.max(axis=1) if how == 'max' else S.mean(axis=1)


def sali_averaged_over_similarities(sims, y, how='max'):
    """The mean per-molecule SALI across several similarity matrices.

    The research session of 2026-11-01 names this as the existing way of
    reducing how much the answer depends on one representation. `sims` is
    {name: n x n similarity}; the per-matrix numbers come back too, because an
    average across matrices that disagree hides the disagreement.
    """
    per = {name: sali_per_molecule(S, y, how=how) for name, S in sims.items()}
    stack = np.vstack([per[k] for k in sorted(per)])
    return stack.mean(axis=0), {'per_similarity': per,
                                'spread_across_similarities': stack.std(axis=0),
                                'n_similarities': len(per), 'how': how}


def local_roughness(D, y, k=5):
    """Per molecule: the closeness-weighted mean label gap to its k nearest.

    Lifted from similarity_metrics/nary_similarity.local_roughness, which
    describes itself as iCliff in spirit. weight_j = 1/(distance + 1e-9),
    normalised, times |y_i - y_j|. `D` is a DISTANCE matrix, not a similarity.

    nary_similarity records the one thing not to do with it: it is the same
    quantity as that file's neighbour_disagreement given the same D and y, so
    scoring one against the other restates its own input.
    """
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    k = min(int(k), max(n - 1, 1))
    Dw = D.copy()
    np.fill_diagonal(Dw, np.inf)
    idx = np.argpartition(Dw, kth=k - 1, axis=1)[:, :k]
    dist = np.take_along_axis(Dw, idx, axis=1)
    w = 1.0 / (dist + 1e-9)
    w = w / w.sum(axis=1, keepdims=True)
    y = np.asarray(y, dtype=float)
    return (w * np.abs(y[:, None] - y[idx])).sum(axis=1)


def sas_map(sim, y, sim_cut=None, gap_cut=None):
    """The four quadrants of similarity against label difference, as pair counts.

    Medina-Franco and co-workers' structure-activity similarity map. Every
    pair falls in one of four boxes once a similarity threshold and a label-gap
    threshold are chosen; the cliff quadrant is high similarity with a high
    gap. Thresholds default to the medians of the pair distributions, so the
    map is distribution-relative unless they are passed.

    The quadrant NAMES differ between papers -- "scaffold hopping", "smooth",
    "non-descriptive" are used inconsistently -- so this returns the counts and
    the two thresholds and names only the cliff quadrant.
    """
    sim = np.asarray(sim, dtype=np.float64)
    n = sim.shape[0]
    iu = np.triu_indices(n, k=1)
    s = sim[iu]
    g = np.abs(np.asarray(y, dtype=float)[:, None]
               - np.asarray(y, dtype=float)[None, :])[iu]
    sc = float(np.median(s)) if sim_cut is None else float(sim_cut)
    gc = float(np.median(g)) if gap_cut is None else float(gap_cut)
    hi_s, hi_g = s >= sc, g >= gc
    return {
        'similarity_threshold': sc, 'label_gap_threshold': gc,
        'thresholds_are_medians': sim_cut is None and gap_cut is None,
        'n_pairs': int(s.size),
        'cliff_quadrant_high_similarity_high_gap': int((hi_s & hi_g).sum()),
        'high_similarity_low_gap': int((hi_s & ~hi_g).sum()),
        'low_similarity_high_gap': int((~hi_s & hi_g).sum()),
        'low_similarity_low_gap': int((~hi_s & ~hi_g).sum()),
        'source': 'Medina-Franco and co-workers, structure-activity '
                  'similarity map'}


def sari(sim, y, disc_similarity_cut=0.6):
    """Peltason and Bajorath's continuity and discontinuity scores, per dataset.

    ⛔ HOW FAR THIS IS VERIFIED, read 2026-11-02 from secondary sources and
    NOT from Peltason & Bajorath 2007, J. Med. Chem. 50:5571:

      The DISCONTINUITY half is stated consistently: the mean over pairs with
      Tanimoto above 0.6 of the label difference times the pair similarity.
      That is what raw_discontinuity below computes.

      The CONTINUITY half is described only as "the property-weighted average
      of pairwise compound similarity". This file weights each pair's
      similarity by 1/(1 + label difference), which is one reading of that
      sentence and not the paper's formula.

      The COMBINATION is given two ways in the secondary literature:
      0.5 * (continuity + 1 - discontinuity), and 0.5 * (continuity -
      (1 - discontinuity)). The first is the one consistent with the stated
      0-to-1 range. Both are returned, each standardised by min-max over the
      pair distribution, and neither is confirmed against the 2007 paper.

    So: read the paper before any of the three combined numbers goes anywhere.
    The two raw scores are usable now.
    """
    sim = np.asarray(sim, dtype=np.float64)
    n = sim.shape[0]
    iu = np.triu_indices(n, k=1)
    s = sim[iu]
    g = np.abs(np.asarray(y, dtype=float)[:, None]
               - np.asarray(y, dtype=float)[None, :])[iu]
    if s.size == 0:
        return {'available': False, 'reason': 'fewer than two molecules'}

    w = 1.0 / (1.0 + g)
    raw_cont = float((s * w).sum() / w.sum()) if w.sum() > 0 else None
    near = s > disc_similarity_cut
    raw_disc = float((g[near] * s[near]).mean()) if near.any() else None

    def _mm(v, lo, hi):
        if v is None or hi <= lo:
            return None
        return float(min(max((v - lo) / (hi - lo), 0.0), 1.0))

    z_cont = _mm(raw_cont, float(s.min()), float(s.max()))
    gs = g * s
    z_disc = _mm(raw_disc, float(gs.min()), float(gs.max()))
    combined = (0.5 * (z_cont + 1.0 - z_disc)
                if (z_cont is not None and z_disc is not None) else None)
    alternative = (0.5 * (z_cont - (1.0 - z_disc))
                   if (z_cont is not None and z_disc is not None) else None)
    return {
        'available': True,
        'raw_continuity': raw_cont,
        'raw_discontinuity': raw_disc,
        'n_pairs_above_discontinuity_cut': int(near.sum()),
        'discontinuity_similarity_cut': disc_similarity_cut,
        'standardised_continuity': z_cont,
        'standardised_discontinuity': z_disc,
        'combined_plus_one_form': combined,
        'combined_minus_form': alternative,
        'verified': False,
        'source': 'Peltason & Bajorath 2007, J. Med. Chem. 50:5571, via '
                  'secondary sources read 2026-11-02; the continuity formula '
                  'and the combination are not confirmed against the paper'}


def rivality_and_modi(D, y, is_clf=False, discretise_sd=0.65):
    """The rivality index per molecule, and the two shares built on it.

    RI_i = (d_same - d_other) / (d_same + d_other), where d_same is the
    distance to the nearest neighbour in the same class and d_other the
    distance to the nearest in another class. Exact, from Luque Ruiz &
    Gomez-Nieto.

    On a regression label the classes are made by splitting at the median.
    SPARFlow's default discretisation uses a band of 0.65 standard deviations;
    that paper's own procedure has not been read here, so `discretise_sd` is
    recorded and the band is reported rather than applied, and the split used
    is the median.

    ⛔ WHICH SHARE IS "RMODI" IS NOT SETTLED HERE. One secondary source
    describes RMODI as the share of molecules whose nearest neighbour is NOT of
    their own class, and in the same paragraph gives RMODI = share with
    RI_i < 0, which is the share whose same-class neighbour is CLOSER. Those
    are opposite. Both shares are returned by their formula and neither is
    called RMODI.

    MODI itself, for a classification label, is the per-class share of
    molecules whose nearest neighbour has the same class, averaged over
    classes. That is returned as `modi` and is only defined when the label is
    already classes.
    """
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    y = np.asarray(y)
    cls = y.astype(int) if is_clf else (y > np.median(y)).astype(int)
    Dw = D.copy()
    np.fill_diagonal(Dw, np.inf)

    d_same = np.full(n, np.inf)
    d_other = np.full(n, np.inf)
    for c in np.unique(cls):
        same = cls == c
        rows = np.where(same)[0]
        if rows.size:
            sub = Dw[np.ix_(rows, np.where(same)[0])]
            d_same[rows] = sub.min(axis=1) if sub.shape[1] else np.inf
            other = np.where(~same)[0]
            d_other[rows] = (Dw[np.ix_(rows, other)].min(axis=1)
                             if other.size else np.inf)

    den = d_same + d_other
    ri = np.where(np.isfinite(den) & (den > 0),
                  (d_same - d_other) / np.where(den > 0, den, 1.0), np.nan)
    finite = np.isfinite(ri)
    share_neg = (float((ri[finite] < 0).mean()) if finite.any() else None)

    nn = np.argmin(Dw, axis=1)
    same_class_nn = cls[nn] == cls
    modi = None
    if is_clf:
        modi = float(np.mean([same_class_nn[cls == c].mean()
                              for c in np.unique(cls)]))
    return {
        'rivality_index': ri,
        'share_rivality_below_zero': share_neg,
        'share_nearest_neighbour_other_class': float((~same_class_nn).mean()),
        'modi': modi,
        'classes_from': 'the label' if is_clf else 'a median split',
        'discretisation_band_sd_not_applied': discretise_sd,
        'source': 'Luque Ruiz & Gomez-Nieto, Regression Modelability Index, '
                  'JCIM 2018; read 2026-11-02 through secondary sources'}


def rogi(D, y, n_thresholds=50):
    """The roughness index: 2 times the integral over t of (sigma_0 - sigma_t).

    Aldeghi, Graff, Frey, Morrone, Pyzer-Knapp, Jordan & Coley,
    arXiv:2207.09250. Distances are scaled to [0, 1] and the labels to unit
    standard deviation; at each threshold t the molecules are clustered by
    complete linkage, each cluster is replaced by its mean label, and sigma_t
    is the standard deviation of that. Lower is a smoother landscape.

    ⛔ The integral, the complete-linkage clustering and the [0, 1] scaling are
    as stated in the paper's abstract and in the secondary literature read
    2026-11-02. THE DISPERSION STEP -- replacing each cluster by its mean and
    taking the standard deviation across molecules -- is this file's reading of
    "measures the dispersion of the property using its standard deviation",
    not the paper's code: the rogi package (coleygroup/rogi) is NOT installed
    here. Installing it and checking this against it is a decision with a
    dependency attached.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if n < 3:
        return {'available': False, 'reason': 'fewer than three molecules'}
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    dmax = float(D.max())
    if dmax <= 0:
        return {'available': False, 'reason': 'every pairwise distance is zero'}
    Dn = D / dmax

    y = np.asarray(y, dtype=float)
    sd = float(y.std())
    if sd <= 0:
        return {'available': False, 'reason': 'the label has no spread'}
    yn = (y - y.mean()) / sd

    Z = linkage(squareform(Dn, checks=False), method='complete')
    ts = np.linspace(0.0, 1.0, int(n_thresholds))
    sigma0 = float(yn.std())
    sigmas = []
    for t in ts:
        labels = fcluster(Z, t=float(t), criterion='distance')
        means = np.zeros(n)
        for c in np.unique(labels):
            sel = labels == c
            means[sel] = yn[sel].mean()
        sigmas.append(float(means.std()))
    sigmas = np.asarray(sigmas)
    # np.trapezoid from numpy 2, np.trapz before it. This environment is 1.26.
    _integrate = getattr(np, 'trapezoid', None) or np.trapz
    value = float(2.0 * _integrate(sigma0 - sigmas, ts))
    return {
        'available': True, 'rogi': value, 'sigma_0': sigma0,
        'n_thresholds': int(n_thresholds),
        'sigma_at_thresholds': sigmas.tolist(),
        'verified': False,
        'source': 'Aldeghi et al., arXiv:2207.09250; the dispersion step is '
                  'this file\'s reading, the rogi package is not installed'}


CONTINUOUS_SCORES = {
    # per molecule
    'sali_max': 'the largest SALI over a molecule\'s partners',
    'sali_mean': 'the mean SALI over a molecule\'s partners',
    'sali_averaged_over_similarities': 'per-molecule SALI averaged across '
                                       'several similarity matrices',
    'local_roughness': 'closeness-weighted mean label gap to the k nearest',
    'rivality_index': '(nearest same class - nearest other class) / their sum',
    # per dataset
    'sari': 'continuity and discontinuity scores for the whole dataset',
    'rogi': 'the roughness integral for the whole dataset',
    'modi': 'per-class share whose nearest neighbour shares its class',
    'sas_map': 'pair counts in the four similarity-against-gap quadrants',
}


def agreement(a, b):
    """How two per-molecule cliff flags disagree, as counts of molecules.

    Both, only the first, only the second, neither, and the Jaccard overlap of
    the two cliff sets. A Jaccard of 1.0 means the two rules name the same
    molecules; the counts are there because a Jaccard on its own hides which
    way a disagreement runs.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    both = int((a & b).sum())
    only_a = int((a & ~b).sum())
    only_b = int((b & ~a).sum())
    neither = int((~a & ~b).sum())
    union = both + only_a + only_b
    return {'n_molecules': int(a.size), 'both': both, 'only_first': only_a,
            'only_second': only_b, 'neither': neither,
            'jaccard': (both / union) if union else None}
