#!/usr/bin/env python3
"""N29: does fitting a bias per ChEMBL document change what the model learns?

Paper: Noise.

THE QUESTION. herg_ic50 keeps one row per ChEMBL measurement, and every
measurement names the document (the paper) it came from. If one document's
measurements sit systematically above or below the others, the mean label of
a molecule carries that document's offset. This file fits one offset per
document, removes it from the training labels, and scores the model on held-out
molecules.

Where this came from. A research report of 2026-09-24 (sections C1 and C2)
proposed it from published work on labels from several annotators: Raykar et
al. 2010, Rodrigues and Pereira 2018. That report described the data as
"assets" valued by "sources". Here a source is a ChEMBL document, and nothing
else in that description carries over.

NOTHING IS INJECTED in the four main arms. The offsets are fitted to the labels
as they come. The one arm that injects is the check with known biases, below.

THE FOUR ARMS. Every arm is scored on the same held-out molecules, in the same
feature space, on the same folds.

  mean_label                 one row per molecule; the label is the mean of
                             all its measurements. What a loader normally
                             hands you. Every other arm is paired against it.
  per_record                 one row per measurement, the molecule's features
                             repeated on each. No offset. With squared loss
                             this differs from mean_label only in weighting a
                             molecule by how many times it was measured.
  per_record_bias            one row per measurement, the label minus the
                             fitted offset of that measurement's document.
  mean_label_bias_corrected  one row per molecule; the label is the mean over
                             its measurements of (label minus its document's
                             offset). The offsets are per_record_bias's.

THREE MORE ARMS, added 2026-09-24 from a second research report (its sections
A3, A8 and L). Each is paired against mean_label like the four above.

  per_record_random_effect   one row per measurement, the label minus a
                             document offset fitted as a random effect. The
                             same alternating fit as per_record_bias, except
                             the ridge penalty is not fixed: on every pass the
                             within-document variance of the residuals (sigma2)
                             and the variance of the document offsets (tau2)
                             are estimated by method of moments, and the
                             penalty is sigma2 / tau2. A document then keeps
                             n / (n + sigma2/tau2) of its mean residual, which
                             is what a random-intercept model (MERF, GPBoost)
                             gives. Both variances are written per pass.
  within_document_rank       XGBoost with objective rank:pairwise, one row per
                             measurement, the document as the query group, so
                             the model only learns which of a document's
                             measurements is higher, never a document's level.
                             Checked 2026-09-24 on xgboost 2.1.4: rank:pairwise
                             takes float labels (on 200 toy rows its scores
                             correlated 0.97 with the label). Its scores are put
                             back on the label scale by a straight-line fit of
                             label on an out-of-fold score within the training
                             fold (grouped by scaffold, OOF_SPLITS folds). A
                             measurement from a document with one measurement
                             forms no pair and teaches it nothing. XGBOOST ONLY:
                             it is written under the model key 'xgboost' and
                             paired against mean_label's xgboost. With xgboost
                             absent from the model list it does not run.
                             2026-10-05 (chat 5): herg_ic50's two models are
                             svm and dnn, so it does not run there. OPEN, at
                             the run() model block.
  document_quantile_map      one row per molecule; the label is the mean of its
                             measurements after each document with at least
                             QUANTILE_MAP_MIN_MEASUREMENTS training
                             measurements has its labels mapped by rank onto
                             the quantiles of all training measurements. Other
                             documents are untouched. This assumes documents
                             differ by a monotone distortion of one common
                             distribution; a document that tested a stronger
                             series is flattened too. How many documents and
                             measurements qualify is written per fold. The map
                             reads labels only, so it is the same in every
                             feature space.

HOW AN OFFSET IS FITTED. Two fits, alternated N_ITER times on the training
measurements of one fold, in one feature space:

  1. A LightGBM model predicts every training measurement out of fold, trained
     on the labels minus the current offsets. `kirby.utils.noisy_labels.
     oof_predict`, grouped by scaffold, so no molecule predicts its own other
     measurements.
  2. What that prediction leaves of each RAW label is regressed on which
     document the measurement came from, by ridge regression. A measurement
     belongs to exactly one document, so the ridge solution for a document is
     the sum of its residuals over (its measurement count + RIDGE_ALPHA).

After each pass the offsets are centred to sum to zero within each connected
group of documents (below). The downstream model is then fitted once, on the
labels minus the final offsets. The offsets come from LightGBM and not from the
downstream model, so both downstream models are fitted on the same corrected
labels within one feature space.

WHICH DOCUMENTS GET AN OFFSET. Two documents are joined when they measured a
common training molecule. Only a document in a group of two or more documents
gets an offset. A document that shares no molecule with any other document has
an offset that cannot be told apart from its molecules' true values, so it is
set to 0. Within a group the offsets sum to zero, because a shift common to the
whole group cannot be told apart from the molecules' true values either. Group
sizes are written per fold, counted in documents, molecules and measurements.

MEASURED 2026-09-24 on the whole herg_ic50 pull, 12,008 measurements over 9,802
molecules and 1,676 documents: 573 documents are in 131 groups of two or more.
The largest group holds 220 documents, 1,198 molecules and 2,030 measurements.
So about one document in three can carry an offset at all.

WHAT THE HELD-OUT MOLECULES ARE GRADED AGAINST. The mean of their raw
measurements, in every arm. Those measurements carry their own documents'
offsets, and nothing here removes them, so a test label is not a clean label.
A second view grades only the held-out molecules none of whose documents
measured any training molecule in this fold. Their count is written every fold
and it can be small; it is never filled in.

HOW STABLE THE OFFSETS ARE. The training molecules are resampled with
replacement N_BOOTSTRAP times. The ridge step is refitted on the final
residuals of each resample, with no model refit. Per document, the share of
resamples in which its offset changes sign is written, with how many documents
exceed FLIP_SHARE_REPORTED.

THE CHECK WITH KNOWN BIASES. On the first fold run (fold 0 unless
KIRBY_NOISE_ONLY_FOLDS names others), in space base_1, with the first
downstream model in its rank-1 single representation on that fold: an offset
is drawn for every document from a normal
distribution. Its variance is that of the offsets fitted on the real labels.
It is added to every measurement, and the four arms are run again. Held-out
molecules are graded against the mean of their measurements WITHOUT the added
offset. Written: the correlation between the added offsets and the fitted ones,
and each arm's score. This checks whether the fit can recover offsets of the
size it reported. It is not a result about the real labels.

WHERE THE HYBRID COMES IN (rewritten 2026-10-05, chat 5; handoff
HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4). The models are herg_ic50's two decided
models (svm, dnn; DECISIONS.md via decided_grid, herg_ki's pair). Per outer
fold and per model, every arm runs in 4 feature spaces: the model's 2 best
single representations (base_1, base_2) and its 2 best hybrids (hybrid,
hybrid__general2), all ranked by one search on that fold's training molecules
(_shared.fold_standard_selection: decided grid, pruning off). 2 models x 4 =
8 configurations. Nothing is read from a whole-dataset search. n29 finds no
neighbours, so no distance hybrid and no similarity hybrid is built.

THE ROWS THE SEARCH SEES (herg_ic50 rule, handoff 2.3 item 14, dirty-run bug
11). The loader keeps one row per measurement. The outer folds and the
per-fold searches run on one row per molecule, labelled with the mean of all
its measurements, filed under SEARCH_CONDITION_SUFFIX. So all of a
molecule's measurements sit on one side of every split, and the search sees
the molecules and labels mean_label is trained on. The hybrids are rebuilt on
the training molecules at their mean labels, once per fold per model, and
every arm shares them. So no offset reaches the feature choice. Whether it
should is n20's kind of question.

ROWS, CACHES, CHECKPOINT (2026-09-25, plan sections 2-3). Every arm but
mean_label writes one D-versus-D' row per fold against mean_label; the labels
are real, so score_clean is nan. The accumulators are checkpointed per fold;
the scaled representations, hybrid builds and the offset fit's out-of-fold
predictions go through noise_cache. Since 2026-10-05 the rows file and the
checkpoint are filed under SELECTION_VERSION, so nothing written under the
whole-dataset selection is read back. controls.py's twins run for the two
per-molecule label arms with KIRBY_NOISE_CONTROLS=1; the per-measurement arms
have none (TODO at the row block).

Usage:
    python n29_document_bias.py --dataset herg_ic50 --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import time
import zlib

import numpy as np

from experiments._shared import (
    get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories,
    fold_standard_selection, as_rep_dtype,
)
from experiments.decided_grid import decided_models
from experiments.noise._ground_truth_providers import build_replicate_groups
from experiments.noise.n19_measurement_count import measurement_stats
# 2026-10-05 (chat 5): the per-fold pieces every noise experiment shares
# (noise_helpers, block "CHAT 5 ALIGNMENT").
from experiments.noise.noise_helpers import (
    noise_outer_folds, fold_spread, fold_model_inputs, configuration_record,
    CONFIGURATION_SPACES,
)
from kirby.utils.noisy_labels import oof_predict
# The D-versus-D' rows, the per-fold checkpoint, the caches and the controls
# (2026-09-25, plan sections 2-3). They live in n23 for the reason given at
# the top of its shared block.
from experiments.noise import noise_cache as nc
from experiments.noise.controls import controls_enabled
from experiments.noise.n23_uncertainty_cleaning import (
    RowSink, PassCheckpoint, control_twins, as_cleanup_dict, NO_NEIGHBOURS,
)

# The four arms the check with known biases re-runs.
CORE_ARMS = ('mean_label', 'per_record', 'per_record_bias',
             'mean_label_bias_corrected')
ARMS = CORE_ARMS + ('per_record_random_effect', 'within_document_rank',
                    'document_quantile_map')
ARM_TO_BEAT = 'mean_label'

# How many times the two fits alternate. Written onto the result file with the
# largest change in any offset on each pass, in pIC50 units, so a reader can
# see whether three passes had stopped moving.
N_ITER = 3

# The ridge penalty on the offsets, in measurements: a document with n
# training measurements has the mean of its residuals multiplied by
# n / (n + RIDGE_ALPHA). At 1.0 a document measured once keeps half its
# residual and one measured nine times keeps nine tenths.
RIDGE_ALPHA = 1.0

# Inner folds of the out-of-fold prediction the residuals are read off.
OOF_SPLITS = 5

# Resamples of the training molecules for the sign-flip count.
N_BOOTSTRAP = 200
FLIP_SHARE_REPORTED = 0.2

# A document needs this many training measurements before document_quantile_map
# maps its labels. Below it the document's quantiles are a handful of points.
QUANTILE_MAP_MIN_MEASUREMENTS = 20

# The ranking model of within_document_rank. The settings of the downstream
# xgboost (find_best_hybrid.py, XGBRegressor), with the ranking objective.
RANKER_PARAMS = dict(objective='rank:pairwise', n_estimators=100,
                     colsample_bytree=0.5, random_state=42, n_jobs=-1,
                     verbosity=0)

# Stands in for an infinite ridge penalty when the offsets' variance is 0.
_NO_OFFSET_PENALTY = 1e12

# Only a document in a group of this many documents or more gets an offset.
MIN_DOCUMENTS_IN_GROUP = 2

# The seed of the offsets drawn for the check with known biases.
KNOWN_BIAS_SEED = 0

# 2026-10-05 (chat 5). The per-fold searches run on one row per molecule
# (mean label), not on the loader's measurement rows, so they are filed apart
# from herg_ic50's standard per-fold searches (handoff 2.3 item 14).
SEARCH_CONDITION_SUFFIX = 'n29_one_row_per_molecule_mean_label'

# The rows file and the checkpoint are filed under this, so nothing written
# under the whole-dataset selection (before 2026-10-05) is read back.
SELECTION_VERSION = 'per_fold_selection_2026_10_05'


# ── Records, molecules and documents ───────────────────────────────────────

def record_to_molecule(replicate_groups):
    """Each measurement row's molecule index, in `measurement_stats` order.

    `measurement_stats` sorts molecules by the row their first measurement sits
    on. The same order is rebuilt here, and the caller checks the two agree on
    every molecule's measurement count before using it.
    """
    groups = np.asarray(replicate_groups)
    first = {}
    for i, g in enumerate(groups):
        first.setdefault(g, i)
    order = sorted(first, key=first.get)
    index = {g: j for j, g in enumerate(order)}
    return np.array([index[g] for g in groups], dtype=np.int64)


def document_groups(rec_mol, rec_doc):
    """Connected groups of documents, joined by a molecule both measured.

    Args:
        rec_mol: molecule index per measurement.
        rec_doc: document index per measurement.

    Returns (component, info). `component` maps each document index present in
    `rec_doc` to a group id. `info` counts, per group, its documents, molecules
    and measurements.
    """
    rec_mol = np.asarray(rec_mol)
    rec_doc = np.asarray(rec_doc)
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for d in np.unique(rec_doc):
        find(int(d))
    first_doc = {}
    for m, d in zip(rec_mol.tolist(), rec_doc.tolist()):
        if m in first_doc:
            ra, rb = find(first_doc[m]), find(d)
            if ra != rb:
                parent[ra] = rb
        else:
            first_doc[m] = d
    component = {d: find(d) for d in parent}

    info = {}
    rec_comp = np.array([component[int(d)] for d in rec_doc])
    for c in np.unique(rec_comp):
        on = rec_comp == c
        info[int(c)] = {
            'documents': int(len(np.unique(rec_doc[on]))),
            'molecules': int(len(np.unique(rec_mol[on]))),
            'measurements': int(on.sum()),
        }
    return component, info


def eligible_documents(component, info, min_docs=MIN_DOCUMENTS_IN_GROUP):
    """Document indices that sit in a group of `min_docs` documents or more."""
    return sorted(d for d, c in component.items()
                  if info[int(c)]['documents'] >= int(min_docs))


def ridge_offsets(residual, rec_doc, eligible, component, alpha=RIDGE_ALPHA,
                  weight=None):
    """One offset per eligible document, centred to sum to zero per group.

    Ridge regression of the residuals on a one-hot document membership: each
    measurement belongs to exactly one document, so the normal equations are
    diagonal and a document's offset is sum(residuals) / (count + alpha).
    `weight` counts a measurement more than once, which is what a bootstrap
    resample does. A document absent from the rows gets NaN, not 0.

    Returns {document index: offset}.
    """
    residual = np.asarray(residual, dtype=float)
    rec_doc = np.asarray(rec_doc)
    w = np.ones_like(residual) if weight is None else np.asarray(weight, float)
    elig = set(int(d) for d in eligible)
    sums, counts = {}, {}
    for r, d, ww in zip(residual, rec_doc.tolist(), w):
        if d in elig and ww > 0:
            sums[d] = sums.get(d, 0.0) + ww * r
            counts[d] = counts.get(d, 0.0) + ww
    raw = {d: sums[d] / (counts[d] + float(alpha)) for d in sums}
    by_comp = {}
    for d, b in raw.items():
        by_comp.setdefault(component[d], []).append(b)
    centre = {c: float(np.mean(v)) for c, v in by_comp.items()}
    out = {d: np.nan for d in elig}
    out.update({d: b - centre[component[d]] for d, b in raw.items()})
    return out


def offset_level_correlation(offsets, level_rec, rec_doc, eligible, component):
    """Does a document's offset depend on how active its molecules are?

    ADDED 2026-09-25 from the third research report (A5): error that depends on
    the true value biases slopes, and the check it names is the source
    difference regressed on the level. Here a document's level is the mean
    out-of-fold prediction over its measurements (`level_rec`), not the mean of
    its own labels, which would contain the offset being tested. Both the
    offsets and the levels are centred within each connected group, because the
    offsets are only defined up to a shift per group.

    Returns Pearson and Spearman correlations over eligible documents, and the
    least-squares slope of offset on level (label units per label unit).
    """
    from scipy.stats import pearsonr, spearmanr
    level_rec = np.asarray(level_rec, dtype=float)
    rec_doc = np.asarray(rec_doc)
    docs = [int(d) for d in eligible
            if np.isfinite(offsets.get(int(d), np.nan)) and np.any(rec_doc == d)]
    if len(docs) < 3:
        return {'n_documents': len(docs), 'pearson': float('nan'),
                'spearman': float('nan'), 'slope': float('nan')}
    lev = np.array([level_rec[rec_doc == d].mean() for d in docs])
    off = np.array([offsets[d] for d in docs], dtype=float)
    comp = np.array([component[d] for d in docs])
    for c in np.unique(comp):
        on = comp == c
        lev[on] -= lev[on].mean()
    if np.std(lev) == 0 or np.std(off) == 0:
        return {'n_documents': len(docs), 'pearson': float('nan'),
                'spearman': float('nan'), 'slope': float('nan')}
    return {'n_documents': len(docs),
            'pearson': float(pearsonr(lev, off)[0]),
            'spearman': float(spearmanr(lev, off)[0]),
            'slope': float(np.polyfit(lev, off, 1)[0])}


def offsets_per_record(offsets, rec_doc):
    """The offset each measurement carries; 0 where its document has none."""
    return np.array([offsets.get(int(d), 0.0) if np.isfinite(
        offsets.get(int(d), np.nan)) else 0.0 for d in rec_doc], dtype=float)


def moment_variances(residual, rec_doc, eligible):
    """Method-of-moments sigma2 and tau2 over the eligible documents.

    sigma2  pooled variance of the residuals around their own document's mean,
            over documents with 2+ measurements (N - D degrees of freedom).
    tau2    variance of the document mean residuals minus the part sigma2 / n_d
            puts there, floored at 0.
    Returns (sigma2, tau2), NaN when fewer than two documents are eligible.
    """
    residual = np.asarray(residual, dtype=float)
    rec_doc = np.asarray(rec_doc)
    elig = np.isin(rec_doc, np.asarray(list(eligible), dtype=rec_doc.dtype))
    if elig.sum() < 3:
        return float('nan'), float('nan')
    r, d = residual[elig], rec_doc[elig]
    docs, inv, n_d = np.unique(d, return_inverse=True, return_counts=True)
    if len(docs) < 2:
        return float('nan'), float('nan')
    means = np.bincount(inv, weights=r) / n_d
    ss_within = float(((r - means[inv]) ** 2).sum())
    dof = int(len(r) - len(docs))
    sigma2 = ss_within / dof if dof > 0 else float('nan')
    if not np.isfinite(sigma2):
        return sigma2, float('nan')
    tau2 = max(0.0, float(np.var(means, ddof=1)) - float(np.mean(sigma2 / n_d)))
    return sigma2, tau2


def fit_document_offsets(X_rec, y_rec, rec_doc, g_rec, eligible, component,
                         n_iter=N_ITER, alpha=RIDGE_ALPHA,
                         oof_splits=OOF_SPLITS, variance_trace=None,
                         dataset_name=None):
    """The alternating fit. Returns (offsets, final residuals, trace).

    dataset_name, added 2026-09-25 (plan section 3): given, each pass's
    out-of-fold prediction goes through noise_cache.cached_oof_predict -- the
    same call, stored. The first pass of the ridge fit and of the
    random-effect fit hand over the same labels (every offset starts at 0),
    so the second reads the first's. None computes as before.

    `trace` is the largest change in any offset on each pass, in label units.
    The residual is the RAW label minus the out-of-fold prediction of a model
    trained on the corrected labels, so it is what the offsets must explain.

    alpha='fitted' is the random-effect version: the penalty is re-estimated
    as sigma2 / tau2 on each pass (`moment_variances`), and each pass's
    (sigma2, tau2, penalty) is appended to `variance_trace` when one is given.
    """
    y_rec = np.asarray(y_rec, dtype=float)
    offsets = {int(d): 0.0 for d in eligible}
    trace = []
    residual = None
    for _ in range(int(n_iter)):
        b = offsets_per_record(offsets, rec_doc)
        if dataset_name is None:
            oof = oof_predict(X_rec, y_rec - b, g_rec, n_splits=oof_splits)
        else:
            oof = nc.cached_oof_predict(dataset_name, X_rec, y_rec - b, g_rec,
                                        n_splits=oof_splits)
        residual = y_rec - oof
        if alpha == 'fitted':
            s2, t2 = moment_variances(residual, rec_doc, eligible)
            if not (np.isfinite(s2) and np.isfinite(t2)) or t2 <= 0:
                pass_alpha = _NO_OFFSET_PENALTY
            else:
                pass_alpha = s2 / t2
            if variance_trace is not None:
                variance_trace.append({'sigma2': s2, 'tau2': t2,
                                       'penalty_in_measurements': pass_alpha})
        else:
            pass_alpha = alpha
        new = ridge_offsets(residual, rec_doc, eligible, component, pass_alpha)
        moved = [abs(new[d] - offsets.get(d, 0.0)) for d in new
                 if np.isfinite(new[d])]
        trace.append(float(max(moved)) if moved else 0.0)
        offsets = {d: (v if np.isfinite(v) else 0.0) for d, v in new.items()}
    return offsets, residual, trace


def sign_flip_shares(residual, rec_doc, rec_mol_local, n_mol, eligible,
                     component, offsets, n_boot=N_BOOTSTRAP, alpha=RIDGE_ALPHA,
                     seed=0):
    """Per document, the share of resamples in which its offset changes sign.

    Molecules are resampled with replacement; a measurement counts as many
    times as its molecule was drawn. Only the ridge step is refitted. A
    document absent from a resample is not counted in that resample.
    """
    rng = np.random.default_rng(int(seed))
    flips = {d: 0 for d in eligible}
    seen = {d: 0 for d in eligible}
    for _ in range(int(n_boot)):
        draw = np.bincount(rng.integers(0, n_mol, n_mol), minlength=n_mol)
        w = draw[rec_mol_local].astype(float)
        got = ridge_offsets(residual, rec_doc, eligible, component, alpha, w)
        for d in eligible:
            v, ref = got.get(d, np.nan), offsets.get(d, 0.0)
            if not np.isfinite(v) or ref == 0.0:
                continue
            seen[d] += 1
            if np.sign(v) != np.sign(ref):
                flips[d] += 1
    return {d: (flips[d] / seen[d] if seen[d] else float('nan'))
            for d in eligible}


def _fit_score(factory, X_tr, y_tr, X_va, y_va, metric_fn):
    """One fit, one held-out score, or (None, why)."""
    try:
        m = factory()
        m.fit(X_tr, y_tr)
        return float(metric_fn(y_va, m.predict(X_va))), None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def quantile_map_labels(y_rec, rec_doc, min_measurements=QUANTILE_MAP_MIN_MEASUREMENTS):
    """Map each large document's labels onto the pooled label quantiles.

    A document with at least `min_measurements` measurements has measurement
    i (rank r_i of n, ties averaged) replaced by the pooled quantile at
    (r_i - 0.5) / n. Returns (mapped labels, documents mapped, measurements
    mapped).
    """
    from scipy.stats import rankdata
    y_rec = np.asarray(y_rec, dtype=float)
    rec_doc = np.asarray(rec_doc)
    out = y_rec.copy()
    n_docs = n_meas = 0
    for d in np.unique(rec_doc):
        on = np.where(rec_doc == d)[0]
        if len(on) < int(min_measurements):
            continue
        q = (rankdata(y_rec[on]) - 0.5) / len(on)
        out[on] = np.quantile(y_rec, q)
        n_docs += 1
        n_meas += len(on)
    return out, n_docs, n_meas


def rank_within_documents(X_rec_tr, y_rec_tr, rec_doc_tr, g_rec_tr, X_va,
                          oof_splits=OOF_SPLITS, params=None):
    """within_document_rank's held-out predictions, on the label scale.

    The ranker is fitted with the document as query group (rows sorted by it,
    as xgboost requires). A straight line of label on an out-of-fold ranker
    score, over the training measurements, maps its scores to label units.
    Returns (predictions, {'slope', 'intercept', 'oof_pearson'}).
    """
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold
    params = dict(RANKER_PARAMS if params is None else params)
    rec_doc_tr = np.asarray(rec_doc_tr)
    y_rec_tr = np.asarray(y_rec_tr, dtype=float)

    def _fit(idx):
        o = idx[np.argsort(rec_doc_tr[idx], kind='stable')]
        return xgb.XGBRanker(**params).fit(X_rec_tr[o], y_rec_tr[o],
                                           qid=rec_doc_tr[o])

    oof = np.full(len(y_rec_tr), np.nan)
    n_split = min(int(oof_splits), len(np.unique(g_rec_tr)))
    for a, b in GroupKFold(n_splits=n_split).split(X_rec_tr, y_rec_tr,
                                                   g_rec_tr):
        oof[b] = _fit(a).predict(X_rec_tr[b])
    slope, intercept = np.polyfit(oof, y_rec_tr, 1)
    full = _fit(np.arange(len(y_rec_tr)))
    pred = slope * full.predict(X_va) + intercept
    return pred, {'slope': float(slope), 'intercept': float(intercept),
                  'oof_pearson': float(np.corrcoef(oof, y_rec_tr)[0, 1])}


def _molecule_means(values, rec_mol_local, n_mol):
    sums = np.bincount(rec_mol_local, weights=values, minlength=n_mol)
    cnts = np.bincount(rec_mol_local, minlength=n_mol)
    return sums / np.maximum(cnts, 1)


def arm_training_sets(X_ent_tr, y_ent_mean_tr, X_rec_tr, y_rec_tr, rec_mol_local,
                      b_rec, b_rec_re=None, y_rec_qmap=None):
    """The arms' (X, y), all over the same training molecules.

    The four core arms always; per_record_random_effect when its offsets are
    given; document_quantile_map when its mapped labels are given.
    within_document_rank is not a label change and is fitted apart.
    """
    n_mol = X_ent_tr.shape[0]
    corrected = y_rec_tr - b_rec
    sums = np.bincount(rec_mol_local, weights=corrected, minlength=n_mol)
    cnts = np.bincount(rec_mol_local, minlength=n_mol)
    extra = {}
    if b_rec_re is not None:
        extra['per_record_random_effect'] = (X_rec_tr, y_rec_tr - b_rec_re)
    if y_rec_qmap is not None:
        extra['document_quantile_map'] = (
            X_ent_tr, _molecule_means(y_rec_qmap, rec_mol_local, n_mol))
    return {**extra,
        'mean_label': (X_ent_tr, y_ent_mean_tr),
        'per_record': (X_rec_tr, y_rec_tr),
        'per_record_bias': (X_rec_tr, corrected),
        'mean_label_bias_corrected': (X_ent_tr, sums / np.maximum(cnts, 1)),
    }


def _molecule_rows_data(data, stats, dataset_name):
    """The dataset as n29 trains mean_label and grades every arm: one row per
    molecule.

    2026-10-05 (chat 5, handoff 2.3 item 14, dirty-run bug 11). herg_ic50's
    loader keeps one row per measurement (12,008 rows over 9,802 molecules).
    Until now the hybrid recipes came from a whole-dataset search over those
    measurement rows, while n29 split folds over molecules. Now the outer
    folds and the per-fold searches run on these molecule rows.

    Representations and augmentation matrices are each molecule's first row
    (`row_index`; every row of one molecule carries the same features), in
    the representation dtype, float32 (handoff 2.3 item 18). Groups are the
    loader's scaffold groups on that row. The label is the mean of all the
    molecule's measurements (`mean_all`).
    """
    idx = np.asarray(stats['row_index'])
    groups = data.get('groups')
    augs = data.get('augmentations') or None
    smiles = data.get('smiles')
    return {
        'reps': {n: as_rep_dtype(np.asarray(m)[idx])
                 for n, m in data['reps'].items()},
        'augmentations': ({n: as_rep_dtype(np.asarray(m)[idx])
                           for n, m in augs.items()} if augs else None),
        'labels': np.asarray(stats['mean_all'], dtype=np.float64),
        'groups': None if groups is None else np.asarray(groups)[idx],
        'smiles': (None if smiles is None or len(smiles) == 0
                   else [smiles[int(i)] for i in idx]),
        'task_type': data['task_type'],
        'dataset_name': dataset_name,
    }


def _entry_from_selection(s_m):
    """One model's fold_standard_selection result in the form
    noise_helpers.fold_model_inputs reads: 'base', 'hybrids', 'recipes'
    {'hybrid', 'hybrid__general2': (Blueprint, aug_condition)},
    'unavailable'. None stays None (that model's search failed).

    n29 calls fold_standard_selection directly rather than
    noise_helpers.fold_configurations, as n19 does. fold_configurations also
    runs a per-fold distance search and a per-fold similarity search for the
    neighbour spaces, and n29 finds no neighbours, so those two searches would
    be built and never read (2026-10-05)."""
    if s_m is None:
        return None
    recipes, unavailable = {}, {}
    for space, i in (('hybrid', 0), ('hybrid__general2', 1)):
        if len(s_m['hybrids']) > i:
            h = s_m['hybrids'][i]
            recipes[space] = (h['blueprint'], h.get('aug_condition', 'none'))
        else:
            unavailable[space] = (f'the per-fold general search gave this '
                                  f'model fewer than {i + 1} hybrids with '
                                  f'distinct features')
    return {
        'base': list(s_m['base']),
        'base_detail': s_m.get('base_detail'),
        'hybrids': list(s_m['hybrids']),
        'hybrids_skipped_identical_features':
            s_m.get('hybrids_skipped_identical_features', []),
        'fit_configurations': (list(s_m['base'])
                               + [h['label'] for h in s_m['hybrids']]),
        'recipes': recipes,
        'unavailable': unavailable,
        'selection_scope': 'fold_train_rows',
    }


def _fold_record(entries, names_by_model):
    """What the result file says about one fold's configurations, per model:
    noise_helpers.configuration_record (base representations; hybrids with
    allocation, importance, budget in features, evaluator, representation
    count, augmentation condition) plus each one's score on the fold's
    training molecules (the search's own inner folds). Held-out scores are
    under scores/."""
    out = {}
    for m, e in entries.items():
        rec = configuration_record(e, names_by_model.get(m))
        if e is not None:
            # No neighbour space is built here; n29 finds no neighbours.
            rec['neighbour_spaces'] = []
            bd = e.get('base_detail') or {}
            rec['base_training_row_scores'] = {
                s['rep']: s.get('score') for s in (bd.get('selected') or [])}
            rec['hybrid_training_row_scores'] = {
                h.get('label'): h.get('score_mean') for h in e['hybrids']}
        out[m] = convert_numpy(rec)
    return out


def run(data, config, output_dir, dataset_name, **kwargs):
    """N29: one offset per ChEMBL document, fitted and removed from the labels."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N29: ONE OFFSET PER CHEMBL DOCUMENT")
    print("=" * 70)

    if data['task_type'] != 'regression':
        return {'skipped': 'not_regression', 'task_type': data['task_type']}
    rec_docs_raw = data.get('record_documents')
    if rec_docs_raw is None:
        print(f"  SKIPPED: {dataset_name} carries no document per measurement. "
              f"Only herg_ic50's loader returns one (herg.py, "
              f"load_herg_ic50_measurements).")
        return {'skipped': 'no_record_documents', 'dataset': dataset_name}
    rep_groups = build_replicate_groups(data)
    if rep_groups is None:
        return {'skipped': 'no_replicate_structure', 'dataset': dataset_name}

    labels = np.asarray(data['labels'], dtype=np.float64)
    if len(rec_docs_raw) != len(labels):
        raise RuntimeError(
            f"n29: {len(rec_docs_raw)} document ids for {len(labels)} "
            f"measurements; the ids are not on the measurements' rows.")
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_iter = int(kwargs.get('n_iter') or N_ITER)
    alpha = float(kwargs.get('ridge_alpha') if kwargs.get('ridge_alpha')
                  is not None else RIDGE_ALPHA)
    n_boot = int(kwargs.get('n_bootstrap') if kwargs.get('n_bootstrap')
                 is not None else N_BOOTSTRAP)
    oof_splits = int(kwargs.get('oof_splits') or OOF_SPLITS)

    stats = measurement_stats(labels, rep_groups)
    rec_mol = record_to_molecule(rep_groups)
    if not np.array_equal(np.bincount(rec_mol), stats['n_measurements']):
        raise RuntimeError(
            "n29: the molecule order rebuilt here disagrees with "
            "measurement_stats about how many measurements each molecule has.")
    doc_names, rec_doc = np.unique(np.asarray(rec_docs_raw).astype(str),
                                   return_inverse=True)
    n_mol = len(stats['n_measurements'])
    y_mean = stats['mean_all']

    if data.get('groups') is None:
        raise RuntimeError("n29: no fold grouping; random splits are refused.")

    # ── One row per molecule, and the outer folds on those rows ──
    # 2026-10-05 (chat 5, handoff 2.3 item 14, dirty-run bug 11). The folds
    # are noise_outer_folds on the molecule rows: the splits the per-fold
    # searches are filed under, the count from config['n_splits'] (or
    # KIRBY_NOISE_N_FOLDS). Replaces cv_split over molecules, whose folds no
    # stored search lined up with.
    ent_data = _molecule_rows_data(data, stats, dataset_name)
    ent_groups = ent_data['groups']
    ent_reps = ent_data['reps']
    folds, n_folds = noise_outer_folds(ent_data, config)
    largest_train = max((int(np.isin(rec_mol, tr).sum()) for _, tr, _ in folds),
                        default=0)

    # ── Models: the dataset's two decided models (R1, DECISIONS.md Step 1) ──
    # 2026-10-05 (chat 5). herg_ic50 takes herg_ki's pair, svm and dnn
    # (handoff 2.3 item 14). Replaces config['models'] / DOWNSTREAM_MODELS and
    # ensure_dprime_models, which added rf and svm to every list. A `models`
    # kwarg (a smoke, a direct caller) can only narrow the decided pair; a
    # name outside it is dropped and written down. knn is never a downstream
    # model here.
    decided = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    asked = kwargs.get('models')
    if asked:
        downstream_models = [m for m in asked if m in decided]
        models_not_decided = [m for m in asked if m not in decided]
    else:
        downstream_models = list(decided)
        models_not_decided = []
    if config.get('models'):
        print(f"  config['models'] {list(config['models'])} is not read: "
              f"n29 fits the decided models {decided}")
    if kwargs.get('base_reps') is not None:
        print("  base_reps given by the caller are not read: each fold's base "
              "representations come from that fold's search")
    # n_samples is the largest training set in measurement rows: the
    # per_record arms train on one row per measurement.
    model_factories = get_model_factories(
        downstream_models, task_type='regression', n_samples=largest_train)
    models_not_built = {
        m: ('get_model_factories returned no factory (over its row limit at '
            f'{largest_train} training measurements, or not defined)')
        for m in downstream_models if m not in model_factories}
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(f"n29: no decided model could be built. Decided "
                           f"{decided}, asked {list(asked or [])}, factories "
                           f"{sorted(model_factories)}.")
    # OPEN (chat 5): within_document_rank is an xgboost ranker (rank:pairwise)
    # and runs only when xgboost is a downstream model, so it is paired
    # against mean_label's xgboost. herg_ic50's decided models are svm and
    # dnn, so under R1 the arm does not run. Left as it was until Adelaide
    # says: leave it out, or fit the xgboost ranker as part of the method
    # (as the LightGBM offset fit already is) and pair it against each
    # decided model's mean_label. 2026-10-05.
    wdr_runs = 'xgboost' in downstream_models

    print(f"  {len(labels)} measurements, {n_mol} molecules, "
          f"{len(doc_names)} documents, {n_folds} outer folds; running "
          f"{[f for f, _, _ in folds]}")
    print(f"  models {downstream_models}")

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_injection': 'none in the main arms',
        'models': downstream_models,
        'models_decided': decided,
        'models_asked_not_decided': models_not_decided,
        'models_not_built': models_not_built,
        'arms': list(ARMS),
        'arm_to_beat': ARM_TO_BEAT,
        'n_folds': n_folds,
        'folds_run': [int(f) for f, _, _ in folds],
        'configuration_spaces': list(CONFIGURATION_SPACES),
        'selection_scope': 'fold_train_rows',
        'selection_note': (
            'Per outer fold and per decided model: the 2 best base '
            'representations and the 2 best hybrids, ranked by one search on '
            "that fold's training molecules (decided grid, pruning off). "
            'Hybrids are rebuilt on the training molecules at their mean '
            'labels. Nothing is read from a whole-dataset search. No distance '
            'or similarity hybrid: n29 finds no neighbours.'),
        'search_rows': 'one row per molecule (its first measurement row)',
        'search_labels': ("each training molecule's mean over all of its "
                          'measurements'),
        'search_condition_suffix': SEARCH_CONDITION_SUFFIX,
        'per_fold_configurations': {},
        'n_iter': n_iter,
        'ridge_alpha': alpha,
        'ridge_alpha_units': ('measurements: a document with n training '
                              'measurements keeps n / (n + alpha) of the mean '
                              'of its residuals'),
        'oof_splits': oof_splits,
        'n_bootstrap': n_boot,
        'min_documents_in_group': MIN_DOCUMENTS_IN_GROUP,
        'random_effect_penalty': ('sigma2 / tau2, re-estimated by method of '
                                  'moments on every pass; written per pass '
                                  'under per_fold/.../random_effect_variances'),
        'within_document_rank_model': dict(
            RANKER_PARAMS, note=('xgboost only, written under the model key '
                                 'xgboost; scores mapped to label units by a '
                                 'straight line fitted on out-of-fold scores '
                                 'within the training fold')),
        'within_document_rank_ran': wdr_runs,
        'within_document_rank_why_not': (
            None if wdr_runs else
            f'xgboost is not one of the decided models {decided}'),
        'quantile_map_min_measurements': QUANTILE_MAP_MIN_MEASUREMENTS,
        'dataset_counts': {'measurements': int(len(labels)),
                           'molecules': int(n_mol),
                           'documents': int(len(doc_names))},
        'offsets_come_from': ('an out-of-fold LightGBM prediction, not the '
                              'downstream model, so every downstream model in '
                              'one feature space is fitted on the same '
                              'corrected labels'),
        'offsets_reach_feature_selection': False,
        'grading_note': (
            'Held-out molecules are graded against the mean of their raw '
            'measurements, which carry their own documents\' offsets. '
            "'isolated' grades only held-out molecules none of whose documents "
            'measured a training molecule in that fold.'),
        'per_fold': {},
    }

    # acc[view][space][arm][model][fold key] = score
    acc = {'all': {}, 'isolated': {}}
    errors = {}
    t0 = time.time()

    # ── Rows and the per-fold checkpoint (2026-09-25) ──
    # D is the arm every other one is paired against, mean_label: one row per
    # molecule, the mean of its measurements. Every other arm is a D'. The
    # labels are real, so no clean labels exist and score_clean is nan. Rows
    # grade every held-out molecule (the 'all' view); the 'isolated' view
    # stays in the result file only. Both are filed under SELECTION_VERSION
    # since 2026-10-05.
    _metric = (metric_name, metric_fn, lower_is_better)
    _tag = kwargs.get('tag')
    _cond = f'{_tag}__{SELECTION_VERSION}' if _tag else SELECTION_VERSION
    rows = RowSink(output_dir, 'n29_document_bias', dataset_name,
                   experiment='n29', metric=_metric, condition=_cond,
                   data=data, config=config)
    ckpt = PassCheckpoint(
        output_dir, 'n29_document_bias', dataset_name,
        condition=_cond, data=data, config=config,
        source_file=__file__,
        key={'n_iter': n_iter, 'alpha': alpha, 'n_boot': n_boot,
             'oof_splits': oof_splits, 'models': list(downstream_models),
             'spaces': list(CONFIGURATION_SPACES), 'n_splits': n_folds,
             'folds_run': results['folds_run'],
             'selection': SELECTION_VERSION,
             'search_condition_suffix': SEARCH_CONDITION_SUFFIX})
    done_folds = set()
    _state = ckpt.load()
    if _state:
        acc, errors = _state['acc'], _state['errors']
        results['per_fold'] = _state['per_fold']
        results['per_fold_configurations'].update(
            _state.get('per_fold_configurations') or {})
        if _state.get('known_bias_check') is not None:
            results['known_bias_check'] = _state['known_bias_check']
        done_folds = set(_state['done'])
        print(f"  resumed from {ckpt.path}: {len(done_folds)} folds already "
              f"done")

    # The check with known biases runs on the first fold run.
    known_bias_fold = folds[0][0] if folds else None

    for fold_i, tr, va in folds:
        fk = f'f{fold_i}'
        if fk in done_folds:
            print(f"    fold {fold_i}: on disk in the checkpoint")
            continue
        tr_set = np.zeros(n_mol, dtype=bool)
        tr_set[tr] = True
        rec_tr = np.where(tr_set[rec_mol])[0]
        local = np.full(n_mol, -1, dtype=np.int64)
        local[tr] = np.arange(len(tr))
        rec_mol_local = local[rec_mol[rec_tr]]
        rec_doc_tr = rec_doc[rec_tr]
        y_rec_tr = labels[rec_tr]
        g_rec_tr = ent_groups[rec_mol[rec_tr]]

        component, comp_info = document_groups(rec_mol[rec_tr], rec_doc_tr)
        eligible = eligible_documents(component, comp_info)
        big = [c for c in comp_info.values()
               if c['documents'] >= MIN_DOCUMENTS_IN_GROUP]
        train_docs = set(np.unique(rec_doc_tr).tolist())
        va_docs = {}
        for i in np.where(~tr_set[rec_mol])[0]:
            va_docs.setdefault(int(rec_mol[i]), set()).add(int(rec_doc[i]))
        isolated = np.array([not (va_docs.get(int(m), set()) & train_docs)
                             for m in va], dtype=bool)
        fold_block = {
            'training_molecules': int(len(tr)),
            'training_measurements': int(len(rec_tr)),
            'training_documents': int(len(train_docs)),
            'held_out_molecules': int(len(va)),
            'held_out_molecules_isolated': int(isolated.sum()),
            'document_groups_of_2_or_more': {
                'groups': len(big),
                'documents': int(sum(c['documents'] for c in big)),
                'molecules': int(sum(c['molecules'] for c in big)),
                'measurements': int(sum(c['measurements'] for c in big)),
                'largest_group_documents': (max(c['documents'] for c in big)
                                            if big else 0),
            },
            'documents_given_an_offset': len(eligible),
            'per_space': {},
        }
        y_rec_qmap, q_docs, q_meas = quantile_map_labels(y_rec_tr, rec_doc_tr)
        fold_block['document_quantile_map'] = {
            'documents_mapped': int(q_docs),
            'measurements_mapped': int(q_meas),
            'of_training_measurements': int(len(rec_tr)),
        }
        print(f"\n  fold {fold_i}: {len(tr)} training molecules, "
              f"{len(rec_tr)} measurements, {len(eligible)} of "
              f"{len(train_docs)} documents can carry an offset; "
              f"{int(isolated.sum())} of {len(va)} held-out molecules share "
              f"no document with training")

        y_va_all = y_mean[va]

        # ── This fold's 8 configurations (R4, R5; chat 5, 2026-10-05) ──
        # One search on the fold's training molecules at their mean labels;
        # per model its 2 best base representations and 2 best hybrids. The
        # hybrids are rebuilt on the training molecules from the fold
        # search's own recipes and representation scores (dirty-run bug 1),
        # 0/1 features left unscaled (kirby.scaling.FeatureScaler).
        sel = fold_standard_selection(
            ent_reps, ent_data['labels'], ent_groups, train_idx=tr,
            fold_i=int(fold_i), n_outer_folds=n_folds,
            dataset_name=dataset_name, models=list(downstream_models),
            augmentations=ent_data['augmentations'], task_type='regression',
            condition_suffix=SEARCH_CONDITION_SUFFIX)
        entries = {m: _entry_from_selection(sel.get(m))
                   for m in downstream_models}

        # (space, configuration name, X_tr_ent, X_va_ent, models). A matrix
        # two models both hold in the same space (the same representation at
        # the same rank, or the same hybrid built identically) is one cell,
        # so its offsets are fitted once and both models share them.
        cells = []
        names_by_model = {}
        fold_block['configurations_unavailable'] = {}
        for m in downstream_models:
            ins, unav, names = fold_model_inputs(
                ent_data, dataset_name, m, entries[m], train_idx=tr,
                val_idx=va, y_train=y_mean[tr], factory=model_factories[m],
                metric=_metric)
            names_by_model[m] = names
            if unav:
                fold_block['configurations_unavailable'][m] = dict(unav)
                for sp, why in unav.items():
                    errors[f'{fk}|{sp}/{m}|build'] = str(why)
            for sp in CONFIGURATION_SPACES:
                if sp not in ins:
                    continue
                X_tr_m, X_va_m = ins[sp]
                nm = names.get(sp)
                for c in cells:
                    if (c[0] == sp and c[1] == nm
                            and c[2].shape == X_tr_m.shape
                            and np.array_equal(c[2], X_tr_m)
                            and np.array_equal(c[3], X_va_m)):
                        c[4].append(m)
                        break
                else:
                    cells.append((sp, nm, X_tr_m, X_va_m, [m]))
        results['per_fold_configurations'][str(fold_i)] = _fold_record(
            entries, names_by_model)

        for space, cell_name, X_tr_e, X_va_e, cell_models in cells:
            ckey = (space if cell_models == list(downstream_models)
                    else f'{space}/{cell_models[0]}')
            X_rec_tr = X_tr_e[rec_mol_local]
            try:
                offsets, resid, trace = fit_document_offsets(
                    X_rec_tr, y_rec_tr, rec_doc_tr, g_rec_tr, eligible,
                    component, n_iter=n_iter, alpha=alpha,
                    oof_splits=oof_splits, dataset_name=dataset_name)
            except Exception as exc:
                errors[f'{fk}|{ckey}|offsets'] = f'{type(exc).__name__}: {exc}'
                continue
            flips = sign_flip_shares(resid, rec_doc_tr, rec_mol_local, len(tr),
                                     eligible, component, offsets, n_boot,
                                     alpha, seed=fold_i)
            vals = np.array([offsets[d] for d in eligible], dtype=float)
            fl = np.array([v for v in flips.values() if np.isfinite(v)])
            fold_block['per_space'][ckey] = {
                # The representation name, or the hybrid's recipe label.
                'configuration': cell_name,
                'models': list(cell_models),
                'largest_offset_change_per_pass': trace,
                'offset_spread_sd': (float(vals.std()) if vals.size
                                     else float('nan')),
                'offset_abs_max': (float(np.abs(vals).max()) if vals.size
                                   else float('nan')),
                'documents_with_an_offset': int(vals.size),
                'documents_flipping_sign_in_over_'
                f'{FLIP_SHARE_REPORTED:g}_of_resamples':
                    int((fl > FLIP_SHARE_REPORTED).sum()),
                'documents_with_a_flip_share': int(fl.size),
                'offset_against_document_level': offset_level_correlation(
                    offsets, y_rec_tr - resid, rec_doc_tr, eligible, component),
            }
            b_rec = offsets_per_record(offsets, rec_doc_tr)
            var_trace = []
            b_rec_re = None
            try:
                off_re, _r, trace_re = fit_document_offsets(
                    X_rec_tr, y_rec_tr, rec_doc_tr, g_rec_tr, eligible,
                    component, n_iter=n_iter, alpha='fitted',
                    oof_splits=oof_splits, variance_trace=var_trace,
                    dataset_name=dataset_name)
                b_rec_re = offsets_per_record(off_re, rec_doc_tr)
                vre = np.array([off_re[d] for d in eligible], dtype=float)
                fold_block['per_space'][ckey]['random_effect'] = {
                    'variances_per_pass': var_trace,
                    'largest_offset_change_per_pass': trace_re,
                    'offset_spread_sd': (float(vre.std()) if vre.size
                                         else float('nan')),
                }
            except Exception as exc:
                errors[f'{fk}|{ckey}|random_effect_offsets'] = (
                    f'{type(exc).__name__}: {exc}')
            if 'xgboost' in cell_models:
                try:
                    pred_rank, cal = rank_within_documents(
                        X_rec_tr, y_rec_tr, rec_doc_tr, g_rec_tr, X_va_e,
                        oof_splits=oof_splits)
                    fold_block['per_space'][ckey]['within_document_rank'] = cal
                    (acc['all'].setdefault(space, {})
                     .setdefault('within_document_rank', {})
                     .setdefault('xgboost', {}))[fk] = float(
                         metric_fn(y_va_all, pred_rank))
                    if isolated.sum() >= 5:
                        (acc['isolated'].setdefault(space, {})
                         .setdefault('within_document_rank', {})
                         .setdefault('xgboost', {}))[fk] = float(
                             metric_fn(y_va_all[isolated], pred_rank[isolated]))
                except Exception as exc:
                    errors[f'{fk}|{ckey}|within_document_rank|xgboost'] = (
                        f'{type(exc).__name__}: {exc}')
            sets = arm_training_sets(X_tr_e, y_mean[tr], X_rec_tr, y_rec_tr,
                                     rec_mol_local, b_rec, b_rec_re=b_rec_re,
                                     y_rec_qmap=y_rec_qmap)
            for m in cell_models:
                for arm in ARMS:
                    if arm not in sets:
                        continue
                    Xa, ya = sets[arm]
                    # One fit, graded two ways: every held-out molecule, and
                    # only the isolated ones.
                    try:
                        mdl = model_factories[m]()
                        mdl.fit(Xa, ya)
                        pred = mdl.predict(X_va_e)
                    except Exception as exc:
                        errors[f'{fk}|{ckey}|{arm}|{m}'] = (
                            f'{type(exc).__name__}: {exc}')
                        continue
                    (acc['all'].setdefault(space, {}).setdefault(arm, {})
                     .setdefault(m, {}))[fk] = float(metric_fn(y_va_all, pred))
                    if isolated.sum() >= 5:
                        (acc['isolated'].setdefault(space, {})
                         .setdefault(arm, {}).setdefault(m, {}))[fk] = float(
                            metric_fn(y_va_all[isolated], pred[isolated]))

            # ── The D-versus-D' rows of this matrix (2026-09-25) ──
            # within_document_rank is written under xgboost, paired against
            # mean_label's xgboost, as the summary pairs it.
            for m in cell_models:
                sD = (acc['all'].get(space, {}).get(ARM_TO_BEAT, {})
                      .get(m, {}).get(fk))
                for arm in ARMS:
                    if arm == ARM_TO_BEAT:
                        continue
                    sP = (acc['all'].get(space, {}).get(arm, {})
                          .get(m, {}).get(fk))
                    if sP is None:
                        continue
                    rows.add(method=arm, method_setting=(
                                 f'ridge_alpha={alpha:g}|n_iter={n_iter}'),
                             similarity_setting=NO_NEIGHBOURS,
                             model_input=f'{space}:{cell_name}', model=m,
                             noise_condition='none',
                             level=None, draw=0, fold=fold_i, score_D=sD,
                             score_Dprime=sP, score_clean=None)
            # ── Controls (KIRBY_NOISE_CONTROLS=1 only) ──
            # The two arms that keep one row per molecule and change its label
            # get controls.py's twins: the label change dealt to random
            # molecules, and permuted.
            # TODO(similarity-sweep 2026-09-25): per_record_bias,
            # per_record_random_effect and within_document_rank train on one
            # row PER MEASUREMENT, so D' has other rows than D and
            # controls.py's twins (which permute a change over D's rows) are
            # not defined for them. The natural control -- permuting the
            # fitted offsets across documents -- is a design decision for the
            # owner and is not built.
            if controls_enabled():
                for arm in ('mean_label_bias_corrected',
                            'document_quantile_map'):
                    if arm not in sets:
                        continue
                    _cd = as_cleanup_dict(X_tr_e, y_mean[tr],
                                          labels=sets[arm][1])
                    _seed = (int(fold_i), int(zlib.crc32(
                        f'{ckey}|{arm}'.encode()) & 0x7FFFFFFF))
                    for cname, tw, why in control_twins(_cd, X_tr_e,
                                                        y_mean[tr], _seed):
                        if tw is None:
                            errors[f'{fk}|{ckey}|{arm}|{cname}'] = why
                            continue
                        for m in cell_models:
                            s_, err = _fit_score(model_factories[m], tw['X'],
                                                 tw['y'], X_va_e, y_va_all,
                                                 metric_fn)
                            sD = (acc['all'].get(space, {})
                                  .get(ARM_TO_BEAT, {}).get(m, {}).get(fk))
                            rows.add(method=arm, method_setting=(
                                         f'ridge_alpha={alpha:g}|n_iter='
                                         f'{n_iter}|control={cname}'),
                                     similarity_setting=NO_NEIGHBOURS,
                                     model_input=f'{space}:{cell_name}',
                                     model=m,
                                     noise_condition='none', level=None,
                                     draw=0, fold=fold_i, score_D=sD,
                                     score_Dprime=s_, score_clean=None)

            # ── The check with known biases: first fold run, base_1, first
            # model ──
            # In the first model's rank-1 single representation on this fold.
            if (fold_i == known_bias_fold and space == 'base_1'
                    and downstream_models[0] in cell_models
                    and vals.size >= 2):
                results['known_bias_check'] = convert_numpy(_known_bias_check(
                    labels, rec_mol, rec_doc, doc_names, tr, va, rec_tr,
                    rec_mol_local, X_tr_e, X_va_e, g_rec_tr, eligible,
                    component, float(vals.var()), offsets, y_mean,
                    model_factories[downstream_models[0]],
                    downstream_models[0], metric_fn, n_iter, alpha,
                    oof_splits))
                results['known_bias_check']['rep'] = cell_name
                results['known_bias_check']['fold'] = int(fold_i)

        results['per_fold'][fk] = convert_numpy(fold_block)
        # The fold is finished: its rows to disk, then the accumulators.
        rows.flush()
        done_folds.add(fk)
        ckpt.save({'acc': acc, 'errors': errors,
                   'per_fold': results['per_fold'],
                   'per_fold_configurations':
                       results['per_fold_configurations'],
                   'known_bias_check': results.get('known_bias_check'),
                   'done': sorted(done_folds)})
        print(f"    fold {fold_i} done [{time.time() - t0:.0f}s]")

    results['scores'] = convert_numpy(acc)
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy(_summarise(
        acc, downstream_models, metric_name, lower_is_better))
    results['rows'] = rows.summary()
    results['controls_run'] = bool(controls_enabled())
    _print_summary(results, metric_name)
    print(f"  D-versus-D' rows: {rows.n_written} written this run to "
          f"{rows.path or 'memory'}")
    ckpt.clear()
    return results


def _known_bias_check(labels, rec_mol, rec_doc, doc_names, tr, va, rec_tr,
                      rec_mol_local, X_tr_e, X_va_e, g_rec_tr, eligible,
                      component, var_b, offsets_real, y_mean, factory,
                      model_name, metric_fn, n_iter, alpha, oof_splits):
    """Add a known offset per document, refit, and compare.

    Graded against the held-out molecules' mean measurement WITHOUT the added
    offset. Only differences within a group are identifiable, so the added
    offsets are centred per group before they are correlated with the fitted
    ones. `fitted_minus_real` removes the offsets the same fit found on the
    unshifted labels, leaving what it attributed to the added part.
    """
    rng = np.random.default_rng(KNOWN_BIAS_SEED)
    b_true = rng.normal(0.0, np.sqrt(var_b), size=len(doc_names))
    shifted = labels + b_true[rec_doc]
    y_rec_tr = shifted[rec_tr]
    rec_doc_tr = rec_doc[rec_tr]
    off, _resid, trace = fit_document_offsets(
        X_tr_e[rec_mol_local], y_rec_tr, rec_doc_tr, g_rec_tr, eligible,
        component, n_iter=n_iter, alpha=alpha, oof_splits=oof_splits)

    by_comp = {}
    for d in eligible:
        by_comp.setdefault(component[d], []).append(b_true[d])
    centre = {c: np.mean(v) for c, v in by_comp.items()}
    truth = np.array([b_true[d] - centre[component[d]] for d in eligible])
    fitted = np.array([off[d] for d in eligible])
    fitted_minus_real = fitted - np.array([offsets_real[d] for d in eligible])

    n_mol_tr = len(tr)
    sums = np.bincount(rec_mol_local, weights=y_rec_tr, minlength=n_mol_tr)
    cnts = np.bincount(rec_mol_local, minlength=n_mol_tr)
    sets = arm_training_sets(X_tr_e, sums / np.maximum(cnts, 1),
                             X_tr_e[rec_mol_local], y_rec_tr, rec_mol_local,
                             offsets_per_record(off, rec_doc_tr))
    scores = {}
    for arm in CORE_ARMS:
        Xa, ya = sets[arm]
        s, err = _fit_score(factory, Xa, ya, X_va_e, y_mean[va], metric_fn)
        scores[arm] = s if s is not None else err

    def _r(a, b):
        return (float(np.corrcoef(a, b)[0, 1])
                if len(a) >= 3 and np.std(a) > 0 and np.std(b) > 0
                else float('nan'))

    return {
        'what_it_is': ('a check with known biases, on the first fold run, in space '
                       'base_1 (the model\'s rank-1 single representation), '
                       'with one downstream model. Not a result about the '
                       'real labels.'),
        'model': model_name,
        'added_offset_sd': float(np.sqrt(var_b)),
        'added_offset_sd_units': 'pIC50',
        'documents_compared': int(len(eligible)),
        'pearson_added_vs_fitted': _r(truth, fitted),
        'pearson_added_vs_fitted_minus_real': _r(truth, fitted_minus_real),
        'largest_offset_change_per_pass': trace,
        'held_out_score_per_arm_against_unshifted_labels': scores,
    }


def _paired(base, got, lower_is_better):
    """One arm against another on the folds both scored. Positive is better.

    2026-10-05 (chat 5): the mean and spread over folds come from
    noise_helpers.fold_spread, so one fold gives 'one fold, no spread' instead
    of a spread of 0. The spread is now the sample SD (ddof=1); it was the
    population SD. No significance test (none ever)."""
    shared = sorted(set(base) & set(got))
    gains = np.array([(base[k] - got[k]) if lower_is_better
                      else (got[k] - base[k]) for k in shared], dtype=float)
    gains = gains[np.isfinite(gains)]
    scale = max(abs(float(np.mean(list(base.values())))) if base else 1.0, 1.0)
    tol = 1e-9 * scale
    sp = fold_spread(gains.tolist())
    return {
        'n_folds_paired': int(gains.size),
        'folds_won': int((gains > tol).sum()),
        'folds_lost': int((gains < -tol).sum()),
        'folds_tied': int((np.abs(gains) <= tol).sum()),
        'mean_gain': sp['mean'],
        'spread_over_fold_gains': sp['sd'],
        'spread_note': sp.get('note'),
        'fold_gains': [float(x) for x in gains],
        # Raw spread only, no test. None when there is no spread (one fold).
        'gain_exceeds_its_own_spread': (
            None if sp['sd'] is None else bool(sp['mean'] > sp['sd'])),
    }


def _summarise(acc, models, metric_name, lower_is_better):
    """Every arm against mean_label, per (view, space, model), paired per fold."""
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'headline': 'folds won against mean_label, paired per fold',
           'per_view': {}}
    for view, by_space in acc.items():
        vrow = {}
        for space, by_arm in by_space.items():
            srow = {}
            base = by_arm.get(ARM_TO_BEAT) or {}
            for m in models:
                b = base.get(m) or {}
                if not b:
                    continue
                _bs = fold_spread(list(b.values()))
                row = {'mean_label_mean': _bs['mean'],
                       'mean_label_spread': _bs['sd'],
                       'per_arm': {}}
                for arm in ARMS:
                    if arm == ARM_TO_BEAT:
                        continue
                    g = (by_arm.get(arm) or {}).get(m) or {}
                    if g:
                        row['per_arm'][arm] = dict(
                            _paired(b, g, lower_is_better),
                            mean=fold_spread(list(g.values()))['mean'])
                srow[m] = row
            vrow[space] = srow
        out['per_view'][view] = vrow
    return out


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric_name}; paired per fold against mean_label)")
    for view, vrow in s['per_view'].items():
        print(f"\n  held-out molecules graded: "
              f"{'all' if view == 'all' else 'only those sharing no document with training'}")
        for space, srow in vrow.items():
            for m, row in srow.items():
                print(f"    {space}/{m}: mean_label "
                      f"{row['mean_label_mean']:.4f}")
                for arm, c in row['per_arm'].items():
                    mark = '  >spread' if c['gain_exceeds_its_own_spread'] else ''
                    mg = ('n/a' if c['mean_gain'] is None
                          else f"{c['mean_gain']:+.4f}")
                    sd = (c.get('spread_note') or 'no spread'
                          if c['spread_over_fold_gains'] is None
                          else f"spread {c['spread_over_fold_gains']:.4f}")
                    print(f"      {arm:26s} won {c['folds_won']} of "
                          f"{c['n_folds_paired']} folds, mean {mg} "
                          f"({sd}){mark}")
    kb = results.get('known_bias_check')
    if kb:
        print(f"\n  check with known biases ({kb['model']}, fold "
              f"{kb.get('fold', 0)}, base_1 "
              f"{kb.get('rep')}): "
              f"added offsets sd {kb['added_offset_sd']:.4f} pIC50 over "
              f"{kb['documents_compared']} documents; pearson added vs fitted "
              f"{kb['pearson_added_vs_fitted']:.3f}, vs fitted minus real "
              f"{kb['pearson_added_vs_fitted_minus_real']:.3f}")


def main():
    parser = create_experiment_parser(
        'N29: one offset per ChEMBL document, fitted and removed')
    parser.add_argument('--n-iter', type=int, default=None,
                        help=f'Alternations of the two fits (default {N_ITER})')
    parser.add_argument('--ridge-alpha', type=float, default=None,
                        help=f'Ridge penalty on the offsets, in measurements '
                             f'(default {RIDGE_ALPHA})')
    parser.add_argument('--n-bootstrap', type=int, default=None,
                        help=f'Resamples for the sign-flip count '
                             f'(default {N_BOOTSTRAP})')
    parser.add_argument('--tag', type=str, default=None,
                        help='Extra text on the result filename')
    args = parser.parse_args()
    config = get_config(args)
    # 2026-10-05 (chat 5, handoff 2.4 item 9): the representations run_all
    # loads -- the dataset's pruned list (herg_ic50 takes herg_ki's), plus its
    # augmentation list -- not --reps (default 'fast'), so this entry point
    # and run_all search the same representations.
    from experiments.noise.noise_helpers import load_noise_data
    if getattr(args, 'reps', None) not in (None, 'fast'):
        print(f"  --reps {args.reps} is not read: n29 loads the dataset's "
              f"pruned representations")
    data, _ = load_noise_data(args.dataset)
    kw = {}
    if args.n_iter:
        kw['n_iter'] = args.n_iter
    if args.ridge_alpha is not None:
        kw['ridge_alpha'] = args.ridge_alpha
    if args.n_bootstrap is not None:
        kw['n_bootstrap'] = args.n_bootstrap
    res = run(data, config, args.output, args.dataset, **kw)
    save_experiment_results(res, args.output, 'n29_document_bias',
                            args.dataset, filename_suffix=args.tag)


if __name__ == '__main__':
    main()
