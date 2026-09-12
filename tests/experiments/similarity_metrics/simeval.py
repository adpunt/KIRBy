"""simeval -- compare molecular SIMILARITY metrics by ACTIVITY coherence.

The whole point (the user's, not mine): a similarity metric is GOOD if two
molecules it calls similar really are similar -- and "really similar" is defined
by MEASURED ACTIVITY, never by ECFP4 Tanimoto. ECFP4 Tanimoto is one competitor
in the grid, judged by the same activity yardstick as everything else. It is
NEVER the ground truth. (That was the circularity the earlier run fell into.)

Two things live here:

  COMPARISON FUNCTIONS  turn a representation matrix X (n, d) into an (n, n)
      DISTANCE matrix D (smaller = more similar). Uniform interface so the
      battery below is agnostic to how similarity was computed.

  EVALUATION BATTERY    given a distance matrix D, the activity y, and scaffold
      groups, score how activity-coherent that metric's neighborhoods are.
      Every method uses y (or a class derived from y). One of them, cliff_auroc,
      also reads ECFP4 Tanimoto -- to pick WHICH molecules are cliffs, never to
      score the metric -- and EVAL_IS_ACTIVITY_GROUNDED marks it False for that.

Citations are in each function's docstring and map to the compass research doc
(Deliverables 1 and 2). Nothing here is invented; SAC is deliberately ABSENT
from the evaluation side -- it can only appear as a candidate metric, never as a
judge, because as a judge it would just be the circularity again.
"""
import numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.metrics import roc_auc_score, mean_squared_error


# ==========================================================================
# COMPARISON FUNCTIONS   X (n,d)  ->  D (n,n) distance, smaller = more similar
# ==========================================================================
# For bit/count fingerprints the classical coefficients (Bajusz, Racz &
# Heberger, J Cheminform 2015, 7:20; Todeschini et al., JCIM 2012) are computed
# in closed form from the Gram matrix so a full (n,n) matrix is one matmul.

def _bit_counts(X):
    """a_i = on-bits of row i ; C_ij = shared on-bits (X X^T)."""
    Xf = X.astype(np.float64)
    a = Xf.sum(1)
    C = Xf @ Xf.T
    return a, C


def dist_tanimoto(X):
    """Tanimoto / Jaccard = c/(a+b-c). The de-facto standard coefficient.
    Works for binary AND count vectors (continuous Tanimoto: c=<x,y>,
    a=<x,x>, b=<y,y>)."""
    Xf = X.astype(np.float64)
    G = Xf @ Xf.T
    a = np.diag(G).copy()
    denom = a[:, None] + a[None, :] - G
    with np.errstate(divide='ignore', invalid='ignore'):
        sim = np.where(denom > 0, G / denom, 0.0)
    np.fill_diagonal(sim, 1.0)
    return 1.0 - sim


def dist_dice(X):
    """Dice = 2c/(a+b). Monotonic with Tanimoto for a single query, but the
    (n,n) geometry differs, so it is a distinct metric here."""
    Xf = X.astype(np.float64)
    G = Xf @ Xf.T
    a = np.diag(G).copy()
    denom = a[:, None] + a[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        sim = np.where(denom > 0, 2.0 * G / denom, 0.0)
    np.fill_diagonal(sim, 1.0)
    return 1.0 - sim


def dist_cosine(X):
    """Cosine = c/sqrt(a*b). Consistent with Tanimoto under DCA
    (Miranda-Quintana et al., Mol Inform 2021)."""
    Xf = X.astype(np.float64)
    nrm = np.linalg.norm(Xf, axis=1)
    nrm[nrm == 0] = 1.0
    Xn = Xf / nrm[:, None]
    sim = Xn @ Xn.T
    return np.clip(1.0 - sim, 0.0, 2.0)


def dist_soergel(X):
    """Soergel = sum|xi-yi| / sum max(xi,yi). Best-tier per Bajusz 2015; for
    binary vectors it equals Tanimoto distance, but for COUNT vectors it is a
    genuinely different (true-metric) distance."""
    Xf = X.astype(np.float64)
    n = Xf.shape[0]
    D = np.zeros((n, n))
    # chunk rows to bound memory: |xi-yi| and max need (n,d) per row-block
    CH = 48
    for s in range(0, n, CH):
        blk = Xf[s:s + CH][:, None, :]          # (c,1,d)
        num = np.abs(blk - Xf[None, :, :]).sum(2)
        den = np.maximum(blk, Xf[None, :, :]).sum(2)
        with np.errstate(divide='ignore', invalid='ignore'):
            D[s:s + CH] = np.where(den > 0, num / den, 0.0)
    return D


def dist_tversky(X, alpha=0.7, beta=0.3):
    """Tversky = c/(c + alpha*(a-c) + beta*(b-c)). Asymmetric; alpha weights the
    reference. Good for sub/superstructure search (compass 1.1)."""
    a, C = _bit_counts(X)
    denom = C + alpha * (a[:, None] - C) + beta * (a[None, :] - C)
    with np.errstate(divide='ignore', invalid='ignore'):
        sim = np.where(denom > 0, C / denom, 0.0)
    np.fill_diagonal(sim, 1.0)
    # asymmetric -> symmetrize the distance so kNN is well defined
    D = 1.0 - sim
    return 0.5 * (D + D.T)


def dist_euclidean(X):
    """Plain L2. Not recommended alone for fingerprints (Bajusz 2015) -- kept as
    a lower baseline and the natural metric for continuous embeddings."""
    Xf = X.astype(np.float64)
    G = Xf @ Xf.T
    sq = np.diag(G)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0.0)
    return np.sqrt(d2)


# which coefficients are meaningful for which representation family.
# NOTE: for BINARY vectors Soergel distance == Tanimoto distance exactly (proven
# in selftest), so it is omitted from BIT_COEFFS to avoid a redundant O(n^2 d)
# recompute. It is kept in COUNT_COEFFS, where it genuinely differs.
BIT_COEFFS = {'tanimoto': dist_tanimoto, 'dice': dist_dice, 'cosine': dist_cosine,
              'tversky': dist_tversky, 'euclidean': dist_euclidean}
COUNT_COEFFS = {'tanimoto': dist_tanimoto, 'dice': dist_dice, 'cosine': dist_cosine,
                'soergel': dist_soergel, 'euclidean': dist_euclidean}
CONT_COEFFS = {'cosine': dist_cosine, 'euclidean': dist_euclidean}


# ==========================================================================
# EVALUATION BATTERY -- every method scores activity coherence of D's geometry
# ==========================================================================
# Convention: every scorer returns "higher = better metric" so aggregation is
# uniform. Where a paper's quantity is lower-is-better (RMSE, violation, cliff
# error), we negate and say so.

def eval_knn_readacross(D, y, groups, k=5, n_splits=5):
    """Continuous MODI / kNN read-across. Golbraikh et al. JCIM 2014 (MODI) in
    its regression form; the direct downstream proxy the compass doc (2.4/2.5)
    calls the most trustworthy. Scaffold-grouped CV, distance-weighted kNN on
    the PRECOMPUTED distance matrix. Returns (spearman_rho, -rmse).

    spearman_rho is NaN when the correlation is undefined (constant predictions
    from a degenerate D, or fewer than two finite predictions). It is NOT
    coerced to 0.0, so an undefined rho is distinguishable from a measured zero
    correlation. Callers that sort, max or compare this value must skip
    non-finite entries themselves -- NaN never loses a Python max()."""
    n = len(y)
    gkf = GroupKFold(n_splits=n_splits)
    preds = np.full(n, np.nan)
    for tr, te in gkf.split(np.arange(n), y, groups):
        kk = min(k, len(tr))
        m = KNeighborsRegressor(n_neighbors=kk, weights='distance',
                                metric='precomputed')
        m.fit(D[np.ix_(tr, tr)], y[tr])
        preds[te] = m.predict(D[np.ix_(te, tr)])
    ok = np.isfinite(preds)
    rho, _ = spearmanr(preds[ok], y[ok])
    rmse = float(np.sqrt(mean_squared_error(y[ok], preds[ok])))
    return (float('nan') if not np.isfinite(rho) else float(rho)), -rmse


def eval_modi_binary(D, y, groups=None):
    """MODI (Golbraikh, Muratov, Fourches & Tropsha, JCIM 2014, ci400572x):
    class-balanced rate at which a compound's nearest neighbour shares its
    activity class. Binary via median split. Higher = more activity-coherent
    neighbourhoods. Because MODI is a direct function of the metric, MODI IS a
    metric-quality score (compass 2.4).

    Returns NaN when one activity class is empty, because 0.0 is also the worst
    legitimate MODI value -- an undefined MODI must stay distinguishable from a
    measured one."""
    cls = (y > np.median(y)).astype(int)
    Dm = D.copy()
    np.fill_diagonal(Dm, np.inf)
    nn = np.argmin(Dm, axis=1)
    out = []
    for c in (0, 1):
        m = cls == c
        if m.sum() == 0:
            return float('nan')
        out.append(float(np.mean(cls[nn[m]] == c)))
    return float(np.mean(out))


def eval_patterson(D, y, seed=0, n_pairs=20000, q=5.0):
    """Neighborhood behaviour (Patterson, Cramer, Ferguson, Clark & Weinberger,
    J Med Chem 1996, 39:3049): a valid descriptor's small distances BOUND
    activity difference from above (trapezoidal). We score the tightness of that
    bound = -(95th percentile of |dy| among the closest q% of pairs). Higher
    (less negative) = tighter bound = better (compass 2.1).

    Returns NaN when fewer than ten pairs survive the closest-q% cut. 0.0 is the
    best attainable value of a statistic that is otherwise always negative, so
    writing it here would make an undefined bound the winner."""
    n = D.shape[0]
    r = np.random.RandomState(seed)
    i = r.randint(0, n, n_pairs)
    j = r.randint(0, n, n_pairs)
    m = i != j
    i, j = i[m], j[m]
    d = D[i, j]
    dy = np.abs(y[i] - y[j])
    close = d <= np.percentile(d, q)
    if close.sum() < 10:
        return float('nan')
    return -float(np.percentile(dy[close], 95))


def cliff_ground_truth(D_struct_ref, y, tan_thr=0.55, k_mad=2.0):
    """True activity cliffs by the Stumpfe & Bajorath standard: structurally
    similar (ECFP4 Tanimoto >= 0.55) AND a large activity jump (> k_mad*MAD).
    NOTE: the cliff DEFINITION uses ECFP4 Tanimoto because that is the field's
    definition of an activity cliff -- but it defines the TARGET (which points
    are cliffs), not the yardstick that ranks metrics. Each metric is then
    judged on whether IT can find these cliffs from activity disagreement in ITS
    own geometry. That is not circular: a metric never scores itself."""
    mad = float(np.median(np.abs(y - np.median(y)))) or 1.0
    thr = k_mad * mad
    dy = np.abs(y[:, None] - y[None, :])
    t = D_struct_ref.copy()
    np.fill_diagonal(t, 0.0)
    pair = (t >= tan_thr) & (dy > thr)
    return pair.any(axis=1)


def eval_cliff_auroc(D, y, cliff_mask, k=5):
    """SALI-style cliff detection (Guha & Van Drie 2008, JCIM 48:646): does
    local ACTIVITY disagreement among a compound's nearest neighbours IN THIS
    METRIC'S SPACE flag the true activity cliffs? AUROC of (max |dy| over kNN)
    vs the cliff mask. Higher = the metric's neighbourhoods expose real cliffs
    (compass 2.3)."""
    if cliff_mask.sum() < 3 or cliff_mask.all():
        return np.nan
    n = D.shape[0]
    Dm = D.copy()
    np.fill_diagonal(Dm, np.inf)
    kk = min(k, n - 1)
    idx = np.argpartition(Dm, kk, axis=1)[:, :kk]
    score = np.array([np.abs(y[i] - y[idx[i]]).max() for i in range(n)])
    return float(roc_auc_score(cliff_mask, score))


def eval_violation_rate(D, y, k=5):
    """Neighbour-lie rate: fraction of kNN pairs whose |dy| exceeds the global
    median |dy|. The blunt form of the similarity principle (compass 2.1).
    Lower is better, so return the negation."""
    n = D.shape[0]
    Dm = D.copy()
    np.fill_diagonal(Dm, np.inf)
    kk = min(k, n - 1)
    idx = np.argpartition(Dm, kk, axis=1)[:, :kk]
    med = np.median(np.abs(y[:, None] - y[None, :])[np.triu_indices(n, 1)])
    lie = np.mean([np.mean(np.abs(y[i] - y[idx[i]]) > med) for i in range(n)])
    return -float(lie)


def eval_vs_enrichment(D, y, top_frac=0.25, alpha=20.0):
    """Retrospective virtual-screening enrichment (compass 2.2). Binarize the
    top `top_frac` by ACTIVITY as 'actives'; for each active query, rank all
    others by distance in this metric and ask how early the OTHER actives are
    retrieved. Report BEDROC (Truchon & Bayly, JCIM 2007, 47:488; alpha=20
    ~ first 8%) averaged over active queries. Pure activity ground truth -- no
    ECFP. Higher = actives cluster in this metric's space."""
    n = D.shape[0]
    active = y >= np.quantile(y, 1.0 - top_frac)
    act_idx = np.where(active)[0]
    if len(act_idx) < 5:
        return np.nan
    scores = []
    Dm = D.copy()
    np.fill_diagonal(Dm, np.inf)
    for q in act_idx:
        order = np.argsort(Dm[q])                     # nearest first
        order = order[order != q]                     # exclude the query itself
        N = len(order)
        ranks = np.where(active[order])[0] + 1        # 1-based ranks of the OTHER actives
        na = len(ranks)
        if na == 0 or na == N:
            continue
        ra = na / N
        # Truchon & Bayly, JCIM 2007, 47:488 -- BEDROC via RIE normalisation.
        s = np.sum(np.exp(-alpha * ranks / N))
        rie_den = ra * (1.0 - np.exp(-alpha)) / (np.exp(alpha / N) - 1.0)
        RIE = s / rie_den if rie_den > 0 else 0.0
        bedroc = (RIE * ra * np.sinh(alpha / 2.0)
                  / (np.cosh(alpha / 2.0) - np.cosh(alpha / 2.0 - alpha * ra))
                  + 1.0 / (1.0 - np.exp(alpha * (1.0 - ra))))
        scores.append(bedroc)
    return float(np.nanmean(scores)) if scores else np.nan


# --------------------------------------------------------------------------
# NOISE MITIGATION -- the user's major downstream use case (compass Deliverable 3).
# A similarity metric is good for noise work if its neighbourhoods EXPOSE injected
# label noise: a corrupted label disagrees with its neighbours in a good metric's
# space. Calibrated to ChEMBL reality: Kramer et al. JMed Chem 2012 sigma~0.54 pKi;
# Kalliokoski et al. PLOS ONE 2013 sigma~0.68 pIC50. Injection goes through
# noiseInject, the same definition the noise suite and the hybrid search use.

def inject_label_noise(y, seed=0, sigma=0.6, frac=0.15):
    """Corrupt a random `frac` of labels, and say which ones. Returns
    (y_noisy, corrupted_mask, eps).

    This is the Outlier condition: a randomly chosen fraction of records get a much
    wider error than the rest. Selection is RANDOM and not by label value, because a
    mistyped unit, a wrong target or a wrong assay is a property of the RECORD, not of
    the value -- Kalliokoski et al. 2013 inspected the extreme disagreements by hand
    and found 9 or 10 out of every 10 beyond 2.5 log units were clear database errors.
    Formally this is Huber's contamination model, and lambda = 3 is Tukey's
    conventional value, corroborated by three independent measurements of the
    within- to between-laboratory error ratio (Avdeef 2019; Llinas & Avdeef 2019;
    Kalliokoski et al. 2013).

    TWO THINGS CHANGED with noiseInject 1.0.0, and both change what a number here
    means.

    `sigma` is now the noise DELIVERED, a root-mean-square over ALL the labels, in
    the label's native units. It used to be the standard deviation applied to the
    corrupted subset only. The condition solves in closed form for the internal scale
    that delivers it, so 0.6 here is now directly comparable with Kramer et al. 2012's
    0.54 pKi and Kalliokoski et al. 2013's 0.68 pIC50, which are per-measurement
    standard deviations of the error.

    The un-selected labels are no longer exactly clean: they carry the base error and
    the selected ones carry three times it. That is what contamination looks like in
    real repeat measurements -- everything has some measurement error and a few
    records are simply wrong -- and it is what makes the condition dose-matchable.
    `corrupted_mask` still marks the badly-wrong subset, so it remains the ground
    truth for whether a similarity space finds them; the task is just no longer the
    easier one of separating perturbed rows from untouched ones.

    One shared realisation per (dataset, seed) so every metric is judged on the SAME
    corruption -- paired comparison.
    """
    from kirby.noise_spec import NoiseSpec, make_noisy_labels_verbose
    y = np.asarray(y, dtype=float).ravel()
    spec = NoiseSpec(strategy='outlier_p05', target_effective_noise=float(sigma),
                     noise_seed=10_000 + int(seed), dose_units='label',
                     params=(('p', float(frac)),))
    res = make_noisy_labels_verbose(y, spec)
    scale = np.asarray(res.noise_scale, dtype=float)
    mask = scale > (scale.min() + 1e-12)
    return np.asarray(res.y_noisy), mask, np.asarray(res.epsilon)


def _knn_idx_dist(D, k):
    """kNN indices and their distances (self excluded), vectorized."""
    n = D.shape[0]; Dm = D.copy(); np.fill_diagonal(Dm, np.inf)
    kk = min(k, n - 1)
    idx = np.argpartition(Dm, kk, axis=1)[:, :kk]
    dist = np.take_along_axis(Dm, idx, axis=1)
    return idx, dist


def _knn_idx(D, k):
    return _knn_idx_dist(D, k)[0]


def eval_noise_detect(D, y_noisy, mask, k=5):
    """compass 3.4a -- detection AUROC. Flag each compound by how much its
    (noisy) label disagrees with its k nearest neighbours' median label IN THIS
    METRIC'S SPACE; AUROC against the known injected-noise mask. Higher = this
    metric's neighbourhoods expose noise (Wilson/ENN editing quality)."""
    if mask.sum() < 3 or mask.all():
        return np.nan
    idx = _knn_idx(D, k)
    disagree = np.abs(y_noisy - np.median(y_noisy[idx], axis=1))   # vectorized
    return float(roc_auc_score(mask, disagree))


def eval_noise_recovery(D, y_noisy, mask, eps, k=5):
    """compass 3.4b -- recovery. Distance-weighted kNN estimate of each label
    from its neighbours (leave-one-out); the correction it implies is
    (estimate - y_noisy). On corrupted points the mostly-clean neighbours pull
    the estimate toward truth, so a good metric gives correction ~= -eps.
    Report corr(correction, -eps) on the corrupted points -- the same validity
    check as noisy_labels.noise_recovery_corr. Higher = better cleaning.

    Returns NaN when the correction or the injected error has zero spread, so no
    correlation exists. 0.0 is what a genuinely uncorrelated correction scores,
    so the two must not share a value."""
    if mask.sum() < 3:
        return np.nan
    idx, dist = _knn_idx_dist(D, k)                                # vectorized
    w = 1.0 / (dist + 1e-9)
    est = (w * y_noisy[idx]).sum(1) / w.sum(1)
    applied = est - y_noisy
    a, e = applied[mask], -eps[mask]
    if np.std(a) == 0 or np.std(e) == 0:
        return float('nan')
    return float(np.corrcoef(a, e)[0, 1])


NOISE_SIGMAS = (0.3, 0.6, 0.9)
NOISE_RATES = (0.10, 0.20, 0.30)


def noise_sweep_scores(D, y, sigmas=NOISE_SIGMAS, rates=NOISE_RATES, seeds=(0, 1, 2)):
    """Detection AUROC + recovery corr for one metric across a dose x contaminated-
    fraction grid, averaged over `seeds` injection realisations. Returns
    {f'{sigma}|{rate}': [detect, recovery]}. D is noise-independent, so this is cheap.
    Sweeping is necessary because noise-sensitivity rankings can invert between low
    and high noise (Cortes-Ciriano, Bender & Malliavin, JCIM 2015).

    `rates` are the contaminated fraction p, which Hampel (2001) puts at "typically
    between 1 percent and 10 percent" for "scientific routine data, not taken with
    utmost care"; this grid runs above that, to 30%. `sigmas` are the delivered dose
    -- see inject_label_noise, where the meaning of that number changed."""
    out = {}
    for sigma in sigmas:
        for rate in rates:
            det, rec = [], []
            for seed in seeds:
                yn, mask, eps = inject_label_noise(y, seed=seed, sigma=sigma, frac=rate)
                det.append(eval_noise_detect(D, yn, mask))
                rec.append(eval_noise_recovery(D, yn, mask, eps))
            out[f'{sigma}|{rate}'] = [float(np.nanmean(det)), float(np.nanmean(rec))]
    return out


def diag_ecfp_agreement(D, tanimoto_ref, seed=0, n_pairs=20000):
    """DIAGNOSTIC ONLY -- NOT a quality score. Spearman between this metric's
    pairwise distance and ECFP4 Tanimoto distance. Reported so we can SEE how
    far each metric departs from the standard, precisely because the standard is
    what we suspect is wrong. Never enters the ranking.

    Returns NaN when the Spearman is undefined (one of the two pair-distance
    vectors constant), so it never reads as a measured zero agreement."""
    n = D.shape[0]
    r = np.random.RandomState(seed)
    i = r.randint(0, n, n_pairs)
    j = r.randint(0, n, n_pairs)
    m = i != j
    i, j = i[m], j[m]
    rho, _ = spearmanr(D[i, j], 1.0 - tanimoto_ref[i, j])
    return float('nan') if not np.isfinite(rho) else float(rho)


# ordered so the doc reads top (most trusted) to bottom. The two noise evals are
# the user's major use case (compass Deliverable 3).
EVALS = ['knn_rho', 'knn_neg_rmse', 'modi', 'patterson', 'cliff_auroc',
         'violation_neg', 'bedroc', 'noise_detect_auroc', 'noise_recovery']
# Which scorers take their TARGET from measured activity alone. cliff_auroc is
# the one that does not: run_battery builds its target with cliff_ground_truth,
# which reads ECFP4 Tanimoto to decide which molecules are cliffs. The metric
# under test is still judged on activity disagreement in its own space, but the
# rows it is judged on were chosen by ECFP4, so it is not activity-grounded.
EVAL_IS_ACTIVITY_GROUNDED = {
    'knn_rho': True,
    'knn_neg_rmse': True,
    'modi': True,
    'patterson': True,
    'cliff_auroc': False,      # target = ECFP4 Tanimoto >= 0.55 AND a big |dy|
    'violation_neg': True,
    'bedroc': True,
    'noise_detect_auroc': True,
    'noise_recovery': True,
}


def run_battery(D, y, groups, tanimoto_ref, seed=0, noise=None):
    """Score one distance matrix on the whole activity-grounded battery.

    `noise` = (y_noisy, mask, eps) is a SHARED injected-noise realisation (from
    inject_label_noise) so every metric is judged on the same corruption. If
    None, one is created here with the default sigma/frac (kept per-seed stable)."""
    cliffs = cliff_ground_truth(tanimoto_ref, y)
    knn_rho, knn_negrmse = eval_knn_readacross(D, y, groups)
    if noise is None:
        noise = inject_label_noise(y, seed=seed)
    y_noisy, mask, eps = noise
    # eval_noise_recovery is NaN when fewer than 3 records were corrupted, or when
    # the correction or the injected error has zero spread. noise_recovery_defined
    # records which rows carry a correlation at all, so a reader averaging the
    # column knows how many injections it was taken over.
    noise_recovery = eval_noise_recovery(D, y_noisy, mask, eps)
    return dict(
        knn_rho=knn_rho,
        knn_neg_rmse=knn_negrmse,
        modi=eval_modi_binary(D, y),
        patterson=eval_patterson(D, y, seed=seed),
        cliff_auroc=eval_cliff_auroc(D, y, cliffs),
        violation_neg=eval_violation_rate(D, y),
        bedroc=eval_vs_enrichment(D, y),
        noise_detect_auroc=eval_noise_detect(D, y_noisy, mask),
        noise_recovery=noise_recovery,
        noise_recovery_defined=bool(np.isfinite(noise_recovery)),
        diag_ecfp_agreement=diag_ecfp_agreement(D, tanimoto_ref, seed=seed),
        n_cliffs=int(cliffs.sum()),
    )
