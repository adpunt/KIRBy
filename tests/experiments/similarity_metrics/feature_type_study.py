"""WHAT KINDS OF FEATURES IMPROVE A SIMILARITY SPACE?

A LIBRARY. It has no main and writes no results file: the 14 property scales,
property_table, build, score_subset and greedy_forward are imported by
opig_scaffold_similarity.py, which runs the five hypotheses below with the
selection fitted on one half of the molecules and every reported number taken on
the other half. The run() that used to live here scored on the same 8000 pairs it
selected on, so its greedy-over-top-k gain was positive by construction, and
nothing read the feature_type_study.json it wrote.

Context is scaffold-diff mutation reps, but the question is general: given a pool
of candidate features, which ones earn their place in a DISTANCE-based similarity
space, and what property of a feature predicts that?

Motivation. In `scaffold_feature_isolation.py` two of five physicochemical scales
turned out to be actively harmful -- removing charge and isoelectric point RAISED
structural fidelity. That is the opposite of the intuition carried over from
predictive modelling, where an extra weak feature is usually harmless because the
model can down-weight it. A euclidean distance has no weights: every feature
contributes its squared difference to every pair, so a feature that carries
little signal still adds spread, and a feature that duplicates another effectively
double-counts that axis. This study tests that explanation properly.

METRIC. Every score is a SPEARMAN RANK CORRELATION in [-1, 1], higher is better,
computed over sampled PAIRS of variants: correlate "how far apart this feature set
says the pair is" against "how far apart an independent substitution matrix
(BLOSUM) says they are", with the mutation-count component partialled out. No
model is trained. This is not R^2 and not accuracy.

HYPOTHESES
  H1 SIGNAL     a feature's solo fidelity predicts whether it belongs.
  H2 REDUNDANCY a feature correlated with ones already chosen hurts even when its
                solo signal is good -- distance double-counts the shared axis.
  H3 GREEDY     selecting on the objective beats taking the top-k by solo signal
                (which is exactly what H2 implies).
  H4 SCALE      unscaled features let large-variance ones dominate the distance.
  H5 MORE-WORSE fidelity is NOT monotone in feature count; there is an optimum.

DECOMPOSITION TRICK. Euclidean distance over a feature subset S is
sqrt(sum_{j in S} (x_ij - x_i'j)^2), so per-feature squared pair-differences can be
precomputed once and any subset scored by summing them. That makes exhaustive and
greedy search over the whole pool cheap.
"""
import os
import sys
import warnings
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src'))
import scaffold_reps as SR                          # noqa: E402
from scaffold_experiment import partial_spearman    # noqa: E402
from kirby.datasets.flab import load_flab_dataset   # noqa: E402

warnings.filterwarnings('ignore')

AA = "ARNDCQEGHILKMFPSTWYV"

# ── Candidate feature pool: 14 amino-acid property scales ────────────────────
# Chosen to deliberately CONTAIN redundancy (charge/pI/polarity overlap;
# MW/volume/SASA overlap) so the redundancy hypothesis is testable.
SCALES = {
 'hydropathy':   [1.8,-4.5,-3.5,-3.5,2.5,-3.5,-3.5,-0.4,-3.2,4.5,3.8,-3.9,1.9,2.8,-1.6,-0.8,-0.7,-0.9,-1.3,4.2],
 'mol_weight':   [89,174,132,133,121,146,147,75,155,131,131,146,149,165,115,105,119,204,181,117],
 'volume':       [88.6,173.4,114.1,111.1,108.5,143.8,138.4,60.1,153.2,166.7,166.7,168.6,162.9,189.9,112.7,89.0,116.1,227.8,193.6,140.0],
 'sasa':         [129,274,195,193,167,225,223,104,224,197,201,236,224,240,159,155,172,285,263,174],
 'aromaticity':  [0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,0],
 'charge':       [0,1,0,-1,0,0,-1,0,0.1,0,0,1,0,0,0,0,0,0,0,0],
 'iso_point':    [6.0,10.76,5.41,2.77,5.07,5.65,3.22,5.97,7.59,6.02,5.98,9.74,5.74,5.48,6.30,5.68,5.60,5.89,5.66,5.96],
 'polarity':     [8.1,10.5,11.6,13.0,5.5,10.5,12.3,9.0,10.4,5.2,4.9,11.3,5.7,5.0,8.0,9.2,8.6,5.4,6.2,5.9],
 'polar_req':    [7.0,9.1,10.0,13.0,5.5,8.6,12.5,7.9,8.4,4.9,4.9,10.1,5.3,5.0,6.6,7.5,6.6,5.2,5.4,5.6],
 'flexibility':  [0.360,0.530,0.460,0.510,0.350,0.490,0.500,0.540,0.320,0.460,0.370,0.470,0.300,0.310,0.510,0.510,0.440,0.310,0.420,0.390],
 'helix_prop':   [1.42,0.98,0.67,1.01,0.70,1.11,1.51,0.57,1.00,1.08,1.21,1.16,1.45,1.13,0.57,0.77,0.83,1.08,0.69,1.06],
 'sheet_prop':   [0.83,0.93,0.89,0.54,1.19,1.10,0.37,0.75,0.87,1.60,1.30,0.74,1.05,1.38,0.55,0.75,1.19,1.37,1.47,1.70],
 'turn_prop':    [0.66,0.95,1.56,1.46,1.19,0.98,0.74,1.56,0.95,0.47,0.59,1.01,0.60,0.60,1.52,1.43,0.96,0.96,1.14,0.50],
 'hbond_donors': [0,3,1,0,1,1,0,0,1,0,0,1,0,0,0,1,1,1,1,0],
}
NAMES = list(SCALES)


def property_table(zscore=True):
    T = np.array([SCALES[n] for n in NAMES], dtype=float).T   # (20, n_props)
    if zscore:
        T = (T - T.mean(0)) / T.std(0)
    return T


def build(dataset, n_pairs=8000, seed=0, zscore=True, positional=True):
    """Return per-feature squared pair-differences (n_features, n_pairs) plus the
    ground truths. Any subset's euclidean distance is sqrt of the summed rows."""
    d = load_flab_dataset(dataset, max_samples=5000)
    S, wt, muts = SR.encode_sequences(d)
    n = len(S)
    T = property_table(zscore)
    k = len(NAMES)

    # per-variant delta per property; position-tagged variant keeps each change
    # in its own slot
    sites = sorted({p for ms in muts for p, _w, _m in ms})
    sidx = {p: i for i, p in enumerate(sites)}
    if positional:
        X = np.zeros((n, len(sites), k), np.float32)
        for i, ms in enumerate(muts):
            for p, w, m in ms:
                X[i, sidx[p]] = T[m] - T[w]
    else:
        X = np.zeros((n, 1, k), np.float32)
        for i, ms in enumerate(muts):
            for _p, w, m in ms:
                X[i, 0] += T[m] - T[w]

    rs = np.random.RandomState(seed)
    I = rs.randint(0, n, n_pairs * 2); J = rs.randint(0, n, n_pairs * 2)
    msk = I != J; I, J = I[msk][:n_pairs], J[msk][:n_pairs]
    # squared difference per feature, summed over position slots
    SD = np.empty((k, len(I)), np.float64)
    for j in range(k):
        diff = X[I, :, j] - X[J, :, j]
        SD[j] = np.sum(diff * diff, axis=1)

    truth = SR.grader_pair_distance(S, I, J, SR.blosum_matrix())
    ham = SR.grader_pair_distance(S, I, J, SR.identity_matrix())
    nmut = float(np.mean([len(x) for x in muts]))
    return SD, truth, ham, n, nmut


def score_subset(SD, cols, truth, ham):
    """Count-controlled Spearman of this feature subset's pair distance against
    the grader. NaN when the correlation is undefined -- an empty subset, a
    constant pair distance, or a partial correlation whose denominator vanishes.
    Never 0.0, which would read as a measured zero correlation. A NaN loses every
    `>` comparison, so greedy_forward below simply never picks such a subset."""
    if not len(cols):
        return float('nan')
    d = np.sqrt(SD[list(cols)].sum(0))
    if np.std(d) == 0:
        return float('nan')
    v = partial_spearman(d, truth, ham)
    return float(v) if np.isfinite(v) else float('nan')


def greedy_forward(SD, truth, ham):
    """Add the feature that most improves the objective; stop when none does.

    The first feature is entered on its own score, not on beating zero. Opening
    at cur=0.0 meant a feature set whose count-controlled correlation is
    negative at every prefix was never entered at all, and the search returned
    the empty set. The empty set has no score, so a search that picks nothing
    returns NaN rather than the opening value."""
    chosen, cur, trace = [], float('-inf'), []
    remaining = set(range(SD.shape[0]))
    while remaining:
        best, bestv = None, cur
        for j in remaining:
            v = score_subset(SD, chosen + [j], truth, ham)
            if v > bestv:
                best, bestv = j, v
        if best is None:
            break
        # the first step has no previous score to subtract, so its gain is NaN
        gain = bestv - cur if np.isfinite(cur) else float('nan')
        chosen.append(best); remaining.discard(best)
        trace.append((NAMES[best], gain, bestv)); cur = bestv
    return chosen, (cur if chosen else float('nan')), trace
