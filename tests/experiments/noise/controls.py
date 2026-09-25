"""Random and shuffled twins of a cleanup's result. Plan section 2, controls.

A cleanup that beats no-cleaning has not yet shown that it found the right
molecules: deleting, downweighting or moving the same NUMBER of molecules
might have done as well. Two twins answer that, each built from the dict
n07.cleaned_training_set returns (X, y, sample_weight, keep, label_column):

  random_twin    the same number of molecules deleted, downweighted or moved,
                 chosen at random. The molecules the method touched are
                 counted, the same count is drawn uniformly from the training
                 rows, and the method's own corrections are dealt to them.
                 This is n24's control (n24_delete_or_keep.py:310-330,
                 RANDOM_DELETION_SEEDS) for deletion and n25's (n25:593-606,
                 random_exempt) for exemption, generalised to weights and
                 labels.
  shuffled_twin  the method's per-molecule corrections -- the keep mask, the
                 weight, or the label change y' - y -- permuted across all
                 training molecules. Every molecule keeps a correction; which
                 molecule gets which is random.

⛔ HOW THIS DIFFERS FROM n18. n18's shuffled control (n18:1062-1070,
SHUFFLE_SEEDS) permutes the SUSPICION SCORES and re-runs the cleanup, so the
method's own rule still decides how far each label moves. shuffled_twin
permutes the method's OUTPUT. The two agree for deletion; for a label move
they differ, because a moved label here carries another molecule's change
rather than its own rule's. random_twin with `rerun=` is the version that
matches n24 exactly: it hands the method a score column that flags the chosen
molecules and lets the method's own code act on them.

⛔ WHEN THE TWO TWINS ARE THE SAME DISTRIBUTION. When every molecule the
method left alone carries one value (weight 1, change 0, kept), dealing the
touched values to random molecules and permuting the whole column differ only
in bookkeeping. They separate when a method touches every molecule by
different amounts, as `downweight` and proportional `label_smooth` do: then
every molecule counts as touched and random_twin reduces to shuffled_twin.
This is stated so a reader does not count them as two independent controls.

Gated: controls_enabled() reads KIRBY_NOISE_CONTROLS=1. Off by default; on
for the full run only (plan section 2).
"""
from __future__ import annotations

import os

import numpy as np


def controls_enabled() -> bool:
    return os.environ.get('KIRBY_NOISE_CONTROLS', '0').strip() == '1'


def _rng(seed):
    """np.random.default_rng on an int or a sequence of ints, so a caller
    seeds on (seed, fold, level, draw, ...) the way n18 does (n18:1066)."""
    if isinstance(seed, (list, tuple)):
        return np.random.default_rng([int(s) for s in seed])
    return np.random.default_rng(int(seed))


def _shape(result):
    """Which of the three shapes a cleanup took: 'delete', 'weight', 'label',
    or 'none' (it touched nothing)."""
    if result.get('keep') is not None:
        return 'delete'
    if result.get('sample_weight') is not None:
        return 'weight'
    if result.get('label_column') is not None:
        return 'label'
    return 'none'


def touched(result, y_tr, atol=0.0):
    """Boolean mask of the training molecules the cleanup changed.

    delete  the molecules not kept
    weight  the molecules whose weight differs from the most common weight
            (for a method that leaves most molecules alone that is 1 after
            mean-normalising; for one that weights every molecule
            differently, every molecule)
    label   the molecules whose label changed by more than atol
    """
    sh = _shape(result)
    n = len(y_tr)
    if sh == 'delete':
        return ~np.asarray(result['keep'], dtype=bool)
    if sh == 'weight':
        w = np.asarray(result['sample_weight'], dtype=float)
        vals, counts = np.unique(w, return_counts=True)
        mode = vals[np.argmax(counts)]
        if counts.max() == 1:
            return np.ones(n, dtype=bool)
        return np.abs(w - mode) > atol
    if sh == 'label':
        d = np.asarray(result['label_column'], dtype=float) - np.asarray(
            y_tr, dtype=float)
        return np.abs(d) > atol
    return np.zeros(n, dtype=bool)


def _rebuild(result, X_tr_s, y_tr, *, keep=None, weight=None, labels=None):
    out = dict(result)
    if keep is not None:
        out.update({'X': X_tr_s[keep], 'y': np.asarray(y_tr)[keep],
                    'keep': keep, 'sample_weight': None, 'label_column': None})
    elif weight is not None:
        out.update({'X': X_tr_s, 'y': np.asarray(y_tr), 'sample_weight': weight,
                    'keep': None, 'label_column': None})
    elif labels is not None:
        out.update({'X': X_tr_s, 'y': labels, 'sample_weight': None,
                    'keep': None, 'label_column': labels})
    return out


def random_twin(result, X_tr_s, y_tr, seed, *, rerun=None, atol=0.0):
    """The same number of molecules touched, chosen at random.

    Args:
        result: the cleanup's dict from n07.cleaned_training_set.
        X_tr_s, y_tr: the training matrix and labels the cleanup was given.
        seed: an int or a tuple of ints (for example (seed, fold, level,
            draw)); the same seed gives the same twin.
        rerun: optional callable(scores, rate) -> result dict that re-runs the
            SAME cleanup on a suspicion score. When given, the twin is the
            method's own rule applied to randomly chosen molecules -- n24's
            random_deletion_scores (n24:310-330): score 1.0 on the chosen
            molecules and 0.0 elsewhere, rate = chosen / training count. For a
            rate-blind cleanup this is a different control from the dict
            version, and it is the one to use for label moves when the target
            each label moves toward matters.

    Without rerun: delete -> the same count of random molecules dropped;
    weight -> the touched molecules' weights dealt, in random order, to the
    same count of random molecules, the rest keep the most common weight;
    label -> the touched molecules' label changes dealt the same way.

    Returns (twin_result, n_touched). twin_result is None when the method
    touched nothing, because there is no count to match.

    ⛔ Without rerun, a label move on a CLASS label deals one molecule's change
    to another molecule, and the sum need not be a class (0 + (-1) on a 0/1
    label gives -1). Same caveat as shuffled_twin: use rerun= there.
    """
    y_tr = np.asarray(y_tr)
    n = len(y_tr)
    mask = touched(result, y_tr, atol)
    n_t = int(mask.sum())
    if n_t == 0:
        return None, 0
    rng = _rng(seed)
    chosen = np.sort(rng.choice(n, size=n_t, replace=False))
    if rerun is not None:
        scores = np.zeros(n, dtype=float)
        scores[chosen] = 1.0
        return rerun(scores, float(n_t) / float(n)), n_t
    sh = _shape(result)
    if sh == 'delete':
        keep = np.ones(n, dtype=bool)
        keep[chosen] = False
        return _rebuild(result, X_tr_s, y_tr, keep=keep), n_t
    if sh == 'weight':
        w = np.asarray(result['sample_weight'], dtype=float)
        vals, counts = np.unique(w, return_counts=True)
        base = vals[np.argmax(counts)] if counts.max() > 1 else None
        new = np.full(n, base if base is not None else 0.0)
        new[chosen] = rng.permutation(w[mask])
        return _rebuild(result, X_tr_s, y_tr, weight=new), n_t
    d = (np.asarray(result['label_column'], dtype=float)
         - np.asarray(y_tr, dtype=float))
    new_d = np.zeros(n)
    new_d[chosen] = rng.permutation(d[mask])
    return _rebuild(result, X_tr_s, y_tr,
                    labels=np.asarray(y_tr, dtype=float) + new_d), n_t


def shuffled_twin(result, X_tr_s, y_tr, seed):
    """The method's per-molecule corrections permuted across the molecules.

    delete: the keep mask permuted. weight: the weight column permuted. label:
    the change y' - y permuted and added back to y. Returns None when the
    cleanup touched nothing.

    ⛔ On a classification label move a permuted change can produce a value
    that is not a class. Use random_twin with rerun= there.
    """
    y_tr = np.asarray(y_tr)
    n = len(y_tr)
    sh = _shape(result)
    if sh == 'none':
        return None
    perm = _rng(seed).permutation(n)
    if sh == 'delete':
        keep = np.asarray(result['keep'], dtype=bool)[perm]
        return _rebuild(result, X_tr_s, y_tr, keep=keep)
    if sh == 'weight':
        w = np.asarray(result['sample_weight'], dtype=float)[perm]
        return _rebuild(result, X_tr_s, y_tr, weight=w)
    d = (np.asarray(result['label_column'], dtype=float)
         - np.asarray(y_tr, dtype=float))[perm]
    return _rebuild(result, X_tr_s, y_tr,
                    labels=np.asarray(y_tr, dtype=float) + d)
