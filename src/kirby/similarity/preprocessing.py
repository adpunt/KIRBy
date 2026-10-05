"""Preprocessing a feature matrix needs before a distance formula is valid.

Every function here exists because skipping the step produces a plausible wrong
number rather than an error. The three failure modes, all measured:

  Tanimoto on count features returns similarity above 1, so the distance goes
  NEGATIVE and those pairs sort to the front of a nearest-neighbour list.
  Sokal-Sneath-1 on counts returns -2.0 (scipy's own docstring example).

  Soergel on z-scored features returns 1.42 to 3.22 for a formula bounded in
  [0, 1], and where the denominator goes non-positive the usual
  `np.where(den > 0, num/den, 0.0)` guard reports distance 0, i.e. "identical".

  Two feature blocks glued together, both z-scored, contribute squared distance
  in proportion to their widths: a 2048-wide block and a 128-wide block measured
  3977 against 241, which is 16.44 = 2048/128. Dividing by sqrt(width) flattens
  Euclidean (mean distance 1.406 / 1.412 / 1.411 at widths 10 / 100 / 400) but
  NOT Manhattan (3.637 / 11.282 / 22.541, still growing as sqrt(width)).
  Manhattan-family formulas need division by the width itself.

The weight-folding tables are the other half. A per-feature weight can be
applied by passing it into the formula, or by rescaling the features and using
the unweighted formula, and the correct rescale is DIFFERENT per formula.
Getting it wrong gives a monotone-but-different distance matrix with no error.
Two obvious sanity checks both fail to catch it: w = 1 reduces correctly under
every exponent, and a uniform w = c is a no-op or a global rescale under every
exponent. Test at a non-uniform w.

Weights are applied AFTER standardising, always. The reverse order is an exact
no-op because a per-feature StandardScaler cancels a per-feature multiply. That
defect has already hit this repo twice -- n12's docstring records the whole
importance-weighting ablation silently producing identical geometry, and the
2026-08-25 fix in tests/noise_mitigation_advanced.py records the second.
"""
import os

import numpy as np


# ---------------------------------------------------------------------------
# Which family a distance formula belongs to.
#
# L2      quantities are quadratic in the features. Weight folds in as sqrt(w);
#         a block of width m is divided by sqrt(m).
# L1      quantities are linear in the features (sums of per-feature terms).
#         Weight folds in as w; a block of width m is divided by m.
# MINKOWSKI  weight folds in as w**(1/p) -- see fold_weights().
# SCIPY_W the weight cannot be folded into the features at all and must be
#         passed to scipy as w=. Canberra divides each term by the same
#         features it differences, so any rescale cancels exactly; correlation
#         puts the weight inside the centring mean as well as the three sums.
# NO_WEIGHT  no per-feature weighting is well defined. Mahalanobis re-estimates
#         its covariance from the transformed features, so a rescale cancels by
#         affine invariance. A weighted Mahalanobis needs the precision matrix
#         held FIXED, which is a separate code path that does not exist yet.
# ---------------------------------------------------------------------------
L2 = 'l2'
L1 = 'l1'
MINKOWSKI = 'minkowski'
SCIPY_W = 'scipy_w'
NO_WEIGHT = 'no_weight'

COEFFICIENT_FAMILY = {
    # continuous, quadratic
    'euclidean': L2,
    'sqeuclidean': L2,
    'seuclidean': L2,
    'cosine': L2,
    # continuous, linear per feature
    'manhattan': L1,
    'cityblock': L1,
    'chebyshev': L1,
    'braycurtis': L1,
    # exponent depends on p
    'minkowski': MINKOWSKI,
    # weight must reach scipy
    'canberra': SCIPY_W,
    'correlation': SCIPY_W,
    # no well-defined per-feature weighting
    'mahalanobis': NO_WEIGHT,
    # set-overlap formulas built from dot products
    'tanimoto_dot': L2,
    'dice_dot': L2,
    # set-overlap formulas built from per-feature min/max or absolute
    # differences -- Soergel, Ruzicka and RDKit's count Tanimoto are the same
    # formula under three names
    'tanimoto_minmax': L1,
    'dice_minmax': L1,
    'soergel': L1,
    'tversky': L1,
}

# Formulas that need strictly 0/1 features.
NEEDS_BINARY = frozenset()

# Formulas that need features >= 0. Every set-overlap formula, plus the
# continuous ones whose denominators are sums of magnitudes.
NEEDS_NONNEGATIVE = frozenset({
    'tanimoto_dot', 'dice_dot', 'tanimoto_minmax', 'dice_minmax',
    'soergel', 'tversky', 'braycurtis',
})

# Chebyshev takes the maximum over features rather than a sum, so its expected
# value grows only logarithmically with the block width (measured 2.93 / 3.68 /
# 4.31 / 4.88 at widths 16 / 64 / 256 / 1024). Dividing by the width or by its
# square root over-corrects. Left unscaled, and recorded so the choice is not
# mistaken for an oversight.
NO_BLOCK_SCALING = frozenset({'chebyshev', 'cosine', 'correlation'})


def coefficient_family(coefficient):
    """Which weighting/scaling family `coefficient` belongs to.

    Raises KeyError on an unknown name rather than guessing -- guessing would
    pick a fold-in exponent, and the wrong exponent is silent.
    """
    key = str(coefficient).lower()
    if key not in COEFFICIENT_FAMILY:
        raise KeyError(
            f"unknown distance formula {coefficient!r}; add it to "
            f"COEFFICIENT_FAMILY with its family before using it, because the "
            f"weight fold-in exponent and the block divisor both depend on it. "
            f"Known: {sorted(COEFFICIENT_FAMILY)}")
    return COEFFICIENT_FAMILY[key]


# ---------------------------------------------------------------------------
# What kind of features are these
# ---------------------------------------------------------------------------

def feature_kind(X, sample_rows=2000, seed=0):
    """'binary', 'count' or 'continuous', decided from the values themselves.

    A representation's NAME does not decide this -- ecfp4 is binary, ecfp4_count
    is counts, and a stacked recipe that z-scores both is continuous. Sampling
    is only to bound the cost on tall matrices; the check is exact on the rows
    it sees, and a matrix that is binary on 2000 random rows and not on the rest
    is pathological enough that the assert helpers re-check in full.
    """
    A = np.asarray(X)
    if A.size == 0:
        return 'continuous'
    if A.shape[0] > sample_rows:
        rng = np.random.RandomState(seed)
        A = A[rng.choice(A.shape[0], sample_rows, replace=False)]
    finite = A[np.isfinite(A)]
    if finite.size == 0:
        return 'continuous'
    if np.all((finite == 0) | (finite == 1)):
        return 'binary'
    if np.all(finite >= 0) and np.all(finite == np.floor(finite)):
        return 'count'
    return 'continuous'


def require_binary(X, coefficient='this formula'):
    """Raise unless every entry is 0 or 1. Never coerce.

    Coercing is what makes this dangerous: scipy's 'jaccard' does not raise on
    continuous features, it thresholds at nonzero, so on z-scored features every
    row becomes all-True and the distance matrix comes back exactly all zeros --
    every pair identical, no warning.
    """
    A = np.asarray(X)
    bad = A[np.isfinite(A) & (A != 0) & (A != 1)]
    if bad.size:
        raise ValueError(
            f"{coefficient} needs 0/1 features and got {bad.size} entries that "
            f"are neither (for example {bad.flat[0]!r}). On counts this formula "
            f"returns similarity above 1, so the distance goes negative and "
            f"those pairs sort to the FRONT of a nearest-neighbour list. "
            f"Binarise with (X > 0) first, or use the count form.")
    return A


def require_nonnegative(X, coefficient='this formula'):
    """Raise unless every entry is >= 0. Never shift to make it so.

    Shifting is not a fix. These formulas are scale-equivariant but not
    translation-invariant, so the answer would depend on the offset chosen:
    measured on 200 z-scored points in 20 dimensions, an offset of +3 against
    +1000 changes the nearest neighbour of 26% of points.
    """
    A = np.asarray(X)
    bad = A[np.isfinite(A) & (A < 0)]
    if bad.size:
        raise ValueError(
            f"{coefficient} needs features >= 0 and got {bad.size} negative "
            f"entries (smallest {float(bad.min()):.4g}). On z-scored features "
            f"Soergel returns 1.42 to 3.22 for a formula bounded in [0, 1], and "
            f"where its denominator goes non-positive the usual guard reports "
            f"distance 0, meaning 'identical'. Use raw counts, and do NOT shift "
            f"the features up to make them non-negative.")
    return A


def check_input(X, coefficient):
    """Run whichever of require_binary / require_nonnegative applies. No-op otherwise."""
    key = str(coefficient).lower()
    if key in NEEDS_BINARY:
        require_binary(X, key)
    elif key in NEEDS_NONNEGATIVE:
        require_nonnegative(X, key)
    return np.asarray(X)


# ---------------------------------------------------------------------------
# Standardising
# ---------------------------------------------------------------------------

def standardise(X, mu=None, sd=None):
    """Z-score the features, returning (Z, mu, sd) so a fold can reuse the stats.

    Pass mu and sd from the training rows when transforming validation rows.
    Fitting a fresh scaler on the validation rows leaks.

    Do NOT standardise binary or count fingerprints. Z-scoring divides each bit
    by sqrt(p(1-p)), which amplifies rare bits without bound: measured on a
    512-bit fingerprint, one differing bit at prevalence 0.002 contributes 301
    units of squared distance where a differing bit at prevalence 0.5
    contributes 4. The result correlates 0.474 with raw Euclidean and 0.186 with
    Tanimoto distance, and is not a coefficient anyone has named.

    Since 2026-10-05 (Adelaide, handoff 2.3 item 17) a 0/1 feature is left as
    it is: when the stats are fitted here, every feature whose training values
    are all 0 or 1 gets mu 0 and sd 1 (kirby.scaling.binary_feature_mask).
    Stats passed in are used as given.
    """
    A = np.asarray(X, dtype=np.float64)
    fitted = mu is None or sd is None
    if mu is None:
        mu = A.mean(0)
    if sd is None:
        sd = A.std(0)
    mu = np.asarray(mu, dtype=np.float64).copy()
    sd = np.asarray(sd, dtype=np.float64).copy()
    if fitted:
        from kirby.scaling import binary_feature_mask
        b = binary_feature_mask(A)
        mu[b] = 0.0
        sd[b] = 1.0
    sd[sd == 0] = 1.0
    Z = (A - mu) / sd
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0), mu, sd


# ---------------------------------------------------------------------------
# Making blocks of different widths carry equal weight
# ---------------------------------------------------------------------------

def block_divisor(width, coefficient='euclidean'):
    """How much to divide one z-scored block by so its width stops being its weight.

    sqrt(width) for the quadratic formulas, width for the per-feature-sum ones,
    1.0 for the three that need neither. See the module docstring for the
    measurements behind each.
    """
    m = int(width)
    if m <= 0:
        return 1.0
    key = str(coefficient).lower()
    if key in NO_BLOCK_SCALING:
        return 1.0
    fam = coefficient_family(key)
    if fam in (L2, SCIPY_W, NO_WEIGHT):
        return float(np.sqrt(m))
    if fam == L1:
        return float(m)
    if fam == MINKOWSKI:
        raise ValueError(
            "block_divisor needs the Minkowski exponent; call "
            "minkowski_block_divisor(width, p) instead")
    return 1.0


def minkowski_block_divisor(width, p):
    """Block divisor for Minkowski with exponent p: width ** (1/p).

    p = 2 recovers sqrt(width) and p = 1 recovers width, so this is the general
    case of block_divisor for the two families above.
    """
    m = int(width)
    if m <= 0:
        return 1.0
    return float(m ** (1.0 / float(p)))


def stack_blocks(blocks, coefficient='euclidean', equalise=True, p=None,
                 stats=None):
    """Z-score each block, divide it by its width so width stops being weight, glue.

    `blocks` is a list of feature matrices with the same number of rows.
    `stats` is an optional list of (mu, sd) per block from the training rows.
    Returns (stacked, stats) so a validation fold can be transformed with the
    training fold's numbers.

    With equalise=False the blocks are z-scored and glued with no width
    correction, which is the behaviour to compare against -- a block holding 400
    of a 500 budget then contributes about 80% of every distance.
    """
    out_parts, out_stats = [], []
    for i, B in enumerate(blocks):
        mu, sd = (stats[i] if stats is not None else (None, None))
        Z, mu, sd = standardise(B, mu, sd)
        if equalise:
            div = (minkowski_block_divisor(Z.shape[1], p)
                   if str(coefficient).lower() == 'minkowski'
                   else block_divisor(Z.shape[1], coefficient))
            Z = Z / div
        out_parts.append(Z)
        out_stats.append((mu, sd))
    return np.hstack(out_parts), out_stats


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def normalise_weights(w, convention='mean1'):
    """Clip negatives to 0 and rescale. 'mean1' is this repo's measured choice.

    Negative weights break every formula here: they put a negative number under
    a square root, turn min into max, or drive a ratio denominator through zero.
    np.sqrt on a negative returns nan with only a RuntimeWarning, and
    noise_mitigation_methods suppresses warnings at import, so the result is an
    all-nan distance matrix and no error.

    'mean1' rescales so the weights average 1, keeping z-scored features near
    their own scale. 'sum1' rescales so they add to 1, which divides every
    distance by the feature count. Rankings survive either (Spearman 1.000) but
    any fixed distance cutoff or kernel width does not, which is what f17
    measured on 2026-08-25.

    A weight of exactly 0 deletes the feature, which makes the result a
    pseudometric: two rows differing only in deleted features come out at
    distance 0.
    """
    v = np.asarray(w, dtype=np.float64).ravel()
    v = np.clip(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    total = v.sum()
    if total <= 0:
        return np.ones_like(v)
    if convention == 'mean1':
        return v * (len(v) / total)
    if convention == 'sum1':
        return v / total
    if convention == 'raw':
        return v
    raise ValueError(
        f"unknown weight convention {convention!r}; use 'mean1' (this repo's "
        f"choice), 'sum1' or 'raw'")


def weight_exponent(coefficient, p=None):
    """The power of w that goes into a per-feature rescale, or None if it cannot.

    0.5 for the quadratic formulas, 1.0 for the per-feature-sum ones, 1/p for
    Minkowski. None means the weight has to reach the formula itself.

    The Minkowski case is the trap. At p = 3, folding w in directly applies
    w**3. At p = 0.5, folding sqrt(w) in applies w**0.25, which flattens a 100x
    spread of weights to 3.16x.
    """
    fam = coefficient_family(coefficient)
    if fam == L2:
        return 0.5
    if fam == L1:
        return 1.0
    if fam == MINKOWSKI:
        if p is None:
            raise ValueError("Minkowski needs its exponent p to fold in a weight")
        return 1.0 / float(p)
    return None


ENV_IMPORTANCE_EXPONENT = 'KIRBY_IMPORTANCE_EXPONENT'


def importance_exponent_override():
    """The swept exponent from the environment, or None when nothing is set.

    None means "use the exponent the coefficient's own family implies", which is
    what every caller did before this existed, so an unset environment leaves
    every number bit-identical.

    The quantity is the power the normalised importance vector is raised to
    before it multiplies the z-scored features. 0 makes every feature weigh the
    same, which is the unweighted control; 1 multiplies each feature by its own
    normalised importance.
    """
    raw = os.environ.get(ENV_IMPORTANCE_EXPONENT)
    if raw is None or str(raw).strip() == '':
        return None
    value = float(raw)
    if value < 0:
        raise ValueError(
            f"{ENV_IMPORTANCE_EXPONENT}={raw!r}: a negative exponent turns a "
            f"zero-importance feature into an infinite multiplier.")
    return value


def fold_weights(Z, w, coefficient, p=None, convention='mean1', exponent=None):
    """Apply per-feature weights to ALREADY-STANDARDISED features.

    Returns (Z_weighted, scipy_kwargs). When the weight cannot be folded into
    the features, Z comes back untouched and scipy_kwargs carries {'w': ...} for
    the caller to hand to scipy.

    Applying weights before standardising is an exact no-op, measured at 9e-16,
    so `Z` must already be z-scored. There is no way to check that from inside
    this function, which is why it is in the name.

    `exponent` overrides the power of the weight vector that multiplies the
    features. None reads KIRBY_IMPORTANCE_EXPONENT, and an unset environment
    falls through to weight_exponent(coefficient, p) -- the family-correct
    value, which is what this function always used. An override is only
    meaningful while it is being swept: it makes the weighted distance a
    DIFFERENT formula from the unweighted one, not the same formula with a
    rescale.
    """
    if w is None:
        return np.asarray(Z, dtype=np.float64), {}
    v = normalise_weights(w, convention)
    A = np.asarray(Z, dtype=np.float64)
    if v.shape[0] != A.shape[1]:
        raise ValueError(
            f"got {v.shape[0]} weights for {A.shape[1]} features")
    fam = coefficient_family(coefficient)
    if fam == NO_WEIGHT:
        raise ValueError(
            f"{coefficient} has no well-defined per-feature weighting. "
            f"Rescaling the features cancels exactly (measured 3.6e-15 between "
            f"the weighted and unweighted calls) because the covariance is "
            f"re-estimated from the rescaled features. A weighted Mahalanobis "
            f"needs the precision matrix estimated once on unweighted data and "
            f"held fixed, which is a code path that does not exist yet.")
    if fam == SCIPY_W:
        # Canberra divides each term by the same features it differences, so any
        # rescale cancels: measured 4.4e-16 between weighted and unweighted,
        # while the same weights move Euclidean by 6.79. Correlation puts the
        # weight inside the centring mean too.
        return A, {'w': v}
    exp = exponent
    if exp is None:
        exp = importance_exponent_override()
    if exp is None:
        exp = weight_exponent(coefficient, p)
    return A * (v ** exp), {}


def assert_weighting_is_not_a_no_op(D_weighted, D_unweighted, tol=1e-10):
    """Raise if a weighted distance matrix equals its unweighted twin.

    The check that catches the failure this repo has hit twice. Call it in a
    smoke gate with a deliberately non-uniform weight vector: w = 1 and a
    uniform w = c both reduce correctly under EVERY fold-in exponent, so neither
    detects a wrong one.
    """
    a = np.asarray(D_weighted, dtype=np.float64)
    b = np.asarray(D_unweighted, dtype=np.float64)
    if a.shape != b.shape:
        return
    gap = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
    if gap <= tol:
        raise AssertionError(
            f"the weighted distance matrix equals the unweighted one to {gap:.3g}. "
            f"Either the weights were applied before standardising (a per-feature "
            f"scaler cancels a per-feature multiply exactly), or the formula is "
            f"one whose weight has to be passed to scipy rather than folded into "
            f"the features. See fold_weights().")
