"""Scaffold-mutation representations for single-wildtype variant libraries (DMS).

The idea (user, 2026-07-18): a single-WT DMS is a SCAFFOLD (the wildtype) plus
point mutations. Describing a variant by its whole sequence makes every variant
look ~99% identical, so ordinary fingerprints collapse. Instead describe each
variant by WHAT CHANGED relative to the scaffold, and grade the resulting
similarity space against an INDEPENDENT structural ground truth.

This module provides:
  * amino-acid property tables + the Grantham and BLOSUM substitution matrices;
  * FIVE scaffold-mutation representations (MutID, dPhysChem, dPhysChem-positional,
    SubScore, BaseDelta-onehot);
  * grader-weighted sequence distances (Grantham / BLOSUM / identity=Hamming) used
    as the structural GROUND TRUTHS;
  * two NEW scaffold-relative distance metrics (position-gated, perturbation-cosine).

Anti-circularity rule (enforced by the caller): never grade a representation with
a ground truth built on the SAME basis (e.g. a physicochemical rep is graded by
BLOSUM/identity, not by Grantham). See scaffold_experiment.py.
"""
import numpy as np

AA = "ARNDCQEGHILKMFPSTWYV"
AAI = {a: i for i, a in enumerate(AA)}

# ---- amino-acid property scales for the dPhysChem reps (deliberately a DIFFERENT
# basis from Grantham's composition/polarity/volume, to keep the physicochemical
# rep and the Grantham grader as independent as physicochemistry allows) ----
KD = dict(zip(AA, [1.8, -4.5, -3.5, -3.5, 2.5, -3.5, -3.5, -0.4, -3.2, 4.5,
                   3.8, -3.9, 1.9, 2.8, -1.6, -0.8, -0.7, -0.9, -1.3, 4.2]))  # Kyte-Doolittle
CHG = {**{a: 0.0 for a in AA}, 'D': -1., 'E': -1., 'K': 1., 'R': 1., 'H': 0.1}
PI = dict(zip(AA, [6.0, 10.76, 5.41, 2.77, 5.07, 5.65, 3.22, 5.97, 7.59, 6.02,
                   5.98, 9.74, 5.74, 5.48, 6.30, 5.68, 5.60, 5.89, 5.66, 5.96]))
MW = dict(zip(AA, [89, 174, 132, 133, 121, 146, 147, 75, 155, 131,
                   131, 146, 149, 165, 115, 105, 119, 204, 181, 117]))
AROM = {**{a: 0.0 for a in AA}, 'F': 1., 'W': 1., 'Y': 1., 'H': 1.}
PROP_SCALES = [KD, CHG, PI, MW, AROM]           # 5 property scales for dPhysChem


def _prop_vec(a):
    return np.array([s[a] for s in PROP_SCALES], float)

# z-score the property columns across the 20 AAs so scales are comparable
_PROP = np.array([_prop_vec(a) for a in AA])
_PROP = (_PROP - _PROP.mean(0)) / _PROP.std(0)
PROPZ = {a: _PROP[AAI[a]] for a in AA}


# ---- Grantham (1974): distance from composition c, polarity p, volume v ----
_GC = dict(zip(AA, [0, .65, 1.33, 1.38, 2.75, .89, .92, .74, .58, 0,
                    0, .33, 0, 0, .39, 1.42, .71, .13, .20, 0]))
_GP = dict(zip(AA, [8.1, 10.5, 11.6, 13.0, 5.5, 10.5, 12.3, 9.0, 10.4, 5.2,
                    4.9, 11.3, 5.7, 5.0, 8.0, 9.2, 8.6, 5.4, 6.2, 5.9]))
_GV = dict(zip(AA, [31, 124, 56, 54, 55, 85, 83, 3, 96, 111,
                    111, 119, 105, 132, 32.5, 32, 61, 170, 136, 84]))


def grantham_matrix():
    """20x20 Grantham distance. Absolute scale is irrelevant (we correlate), so
    the published normalising constant is dropped."""
    a, b, g = 1.833, 0.1018, 0.000399
    M = np.zeros((20, 20))
    for i, x in enumerate(AA):
        for j, y in enumerate(AA):
            M[i, j] = np.sqrt(a * (_GC[x] - _GC[y]) ** 2
                              + b * (_GP[x] - _GP[y]) ** 2
                              + g * (_GV[x] - _GV[y]) ** 2)
    return M


def blosum_matrix():
    """20x20 BLOSUM62 as a DISTANCE (max - score), aligned to AA order."""
    from Bio.Align import substitution_matrices
    B = substitution_matrices.load("BLOSUM62")
    S = np.zeros((20, 20))
    for i, x in enumerate(AA):
        for j, y in enumerate(AA):
            S[i, j] = B[x, y]
    return S.max() - S               # higher score (more similar) -> smaller distance


def identity_matrix():
    """Hamming grader: 1 if different residue, 0 if same (the degenerate baseline)."""
    return 1.0 - np.eye(20)


# ==========================================================================
# variant parsing: concatenate heavy(+light) into one aligned sequence per variant
# ==========================================================================
def encode_sequences(dset):
    """Return (S int-encoded (n,L), wt int-encoded (L,), muts list-of-(pos,wt,mut)).
    Concatenates heavy+light when both are fixed-length; else heavy only. Any
    residue outside the 20 standard AAs is dropped from that column globally."""
    heavy = dset['heavy_seqs']
    light = dset['light_seqs']
    seqs = [h + (l if light else '') for h, l in zip(heavy, light or [''] * len(heavy))]
    L = len(seqs[0])
    if any(len(s) != L for s in seqs):
        seqs = list(heavy); L = len(heavy[0])
        if any(len(s) != L for s in seqs):
            raise ValueError("variable-length sequences; alignment needed")
    # keep only columns where every residue is a standard AA
    good_cols = [c for c in range(L) if all(s[c] in AAI for s in seqs)]
    S = np.array([[AAI[s[c]] for c in good_cols] for s in seqs], dtype=np.int16)
    # wildtype = per-column mode
    wt = np.array([np.bincount(S[:, c]).argmax() for c in range(S.shape[1])], dtype=np.int16)
    muts = []
    for i in range(len(S)):
        diff = np.where(S[i] != wt)[0]
        muts.append([(int(c), int(wt[c]), int(S[i, c])) for c in diff])
    return S, wt, muts


# ==========================================================================
# THE FIVE SCAFFOLD-MUTATION REPRESENTATIONS
# ==========================================================================
def rep_mutid(S, wt, muts):
    """(1) MutID: one-hot over (position x mutant-AA). Pure identity of the change."""
    n, L = S.shape
    X = np.zeros((n, L * 20), np.float32)
    for i, ms in enumerate(muts):
        for pos, _w, m in ms:
            X[i, pos * 20 + m] = 1.0
    return X


def rep_dphys(S, wt, muts):
    """(2) dPhysChem: SUM of per-mutation property CHANGES (position-agnostic).
    'How chemically drastic', regardless of where."""
    n = len(S)
    X = np.zeros((n, len(PROP_SCALES)), np.float32)
    for i, ms in enumerate(muts):
        for pos, w, m in ms:
            X[i] += PROPZ[AA[m]] - PROPZ[AA[w]]
    return X


def rep_dphys_positional(S, wt, muts):
    """(3) dPhysChem-positional: property CHANGE dropped in at the mutated
    position (position x property, mostly zero). 'Where AND how drastic'."""
    n, L = S.shape
    k = len(PROP_SCALES)
    X = np.zeros((n, L * k), np.float32)
    for i, ms in enumerate(muts):
        for pos, w, m in ms:
            X[i, pos * k:(pos + 1) * k] = PROPZ[AA[m]] - PROPZ[AA[w]]
    return X


def rep_subscore(S, wt, muts, gmat, bmat):
    """(4) SubScore: per variant, [sum Grantham, sum BLOSUM-dist, n_mut, mean pos].
    Low-dim, directly encodes substitution drasticness."""
    n = len(S)
    X = np.zeros((n, 4), np.float32)
    for i, ms in enumerate(muts):
        if not ms:
            continue
        g = sum(gmat[w, m] for _p, w, m in ms)
        b = sum(bmat[w, m] for _p, w, m in ms)
        X[i] = [g, b, len(ms), np.mean([p for p, _w, _m in ms])]
    return X


def rep_basedelta(S, wt, muts):
    """(5) BaseDelta-onehot: full-sequence one-hot MINUS the wildtype's one-hot
    (the user's 'difference of fingerprints'). Non-zero only at changed residues:
    -1 at the WT residue, +1 at the mutant."""
    n, L = S.shape
    X = np.zeros((n, L * 20), np.float32)
    for i, ms in enumerate(muts):
        for pos, w, m in ms:
            X[i, pos * 20 + w] -= 1.0
            X[i, pos * 20 + m] += 1.0
    return X


REPS = {
    'MutID': rep_mutid,
    'dPhysChem': rep_dphys,
    'dPhysChem_pos': rep_dphys_positional,
    'SubScore': rep_subscore,          # needs gmat,bmat -> handled in caller
    'BaseDelta': rep_basedelta,
}
# which grader BASES each rep is built from (for the anti-circularity rule)
REP_BASIS = {
    'MutID': {'identity'},
    'dPhysChem': {'physchem'},
    'dPhysChem_pos': {'physchem'},
    'SubScore': {'grantham', 'blosum'},
    'BaseDelta': {'identity'},
}


# ==========================================================================
# GROUND-TRUTH: grader-weighted sequence distance over sampled pairs
# ==========================================================================
def grader_pair_distance(S, I, J, M):
    """Sum over positions of grader-distance M[a,b] between the two sequences
    (a grader-weighted generalisation of Hamming). Vectorised over pairs."""
    a = S[I]; b = S[J]                       # (m, L)
    return M[a.ravel(), b.ravel()].reshape(a.shape).sum(1)


# ==========================================================================
# NEW scaffold-relative DISTANCE METRICS (operate on rep matrix X)
# ==========================================================================
def dist_perturbation_cosine(X):
    """Cosine between WT-perturbation vectors (X is already a delta-from-WT rep)."""
    import simeval
    return simeval.dist_cosine(X)


def dist_position_gated(S, wt, muts, gmat):
    """NEW metric designed for variant libraries: compares two variants by whether
    they perturb the SAME scaffold position. Same position -> fine-grained
    Grantham distance between the two mutant residues; different positions -> the
    summed drasticness of the two independent perturbations. Encodes the scaffold
    structure explicitly. Returns full (n,n) distance."""
    n = len(S)
    # single-mutation fast path (koenig): one (pos,mut) per variant (or none)
    pos = np.full(n, -1); mut = np.full(n, -1); wtr = np.full(n, -1)
    for i, ms in enumerate(muts):
        if ms:
            pos[i], wtr[i], mut[i] = ms[0][0], ms[0][1], ms[0][2]
    D = np.zeros((n, n))
    # per-variant drasticness from WT
    drast = np.array([gmat[wtr[i], mut[i]] if pos[i] >= 0 else 0.0 for i in range(n)])
    same = pos[:, None] == pos[None, :]
    # same position: Grantham between the two mutant AAs
    mm = mut[:, None], mut[None, :]
    gmm = gmat[np.clip(mm[0], 0, 19), np.clip(mm[1], 0, 19)]
    # different position: sum of the two perturbations' drasticness
    dsum = drast[:, None] + drast[None, :]
    D = np.where(same, gmm, dsum)
    np.fill_diagonal(D, 0.0)
    return D
