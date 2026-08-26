"""Group-aware internal early-stopping holdout split for the neural models.

Every torch model in this package (DNN / CNN / BNN / pretrained-DNN) carves an
80/20 holdout out of the data handed to ``fit()`` and uses it for early
stopping. That holdout is taken from the OUTER CV's training fold, so it never
touches the reported test fold — but if the carve is RANDOM, near-duplicate
scaffolds (or V-gene families, or whatever the group key is) end up on both
sides of it. The early-stopping validation loss then keeps falling on rows the
net has effectively already seen, so training stops too late and the model is
mildly over-fit relative to a group-honest split.

``early_stopping_split`` fixes that: when the caller passes ``groups`` to
``fit()``, the holdout is carved with ``GroupShuffleSplit`` so no group appears
on both sides. When the caller passes nothing, the old random behaviour is kept
(so every existing caller keeps working unchanged) and a one-time warning is
emitted per model class.
"""

import numpy as np

# Warn at most once per model class, so a 5-fold x N-budget search does not
# print the same line hundreds of times.
_GROUPS_WARNED = set()


def _warn_once(model_name, message):
    if model_name in _GROUPS_WARNED:
        return
    import warnings
    warnings.warn(message, stacklevel=3)
    _GROUPS_WARNED.add(model_name)


def early_stopping_split(y, groups, rng, model_name, test_size=0.2):
    """Return (train_idx, holdout_idx) for the internal early-stopping split.

    Args:
        y: Target vector (only its length and values are used, for the split).
        groups: Optional group key per row (scaffold / cluster / V-gene family).
            When provided, the split is group-disjoint.
        rng: ``np.random.RandomState`` used for the non-group fallback.
        model_name: Used to key the one-time "no groups" warning.
        test_size: Holdout fraction (default 0.2, matching the historic 80/20).

    Returns:
        (train_idx, holdout_idx) as int arrays.
    """
    y = np.asarray(y)
    n = len(y)

    if groups is not None:
        groups_arr = np.asarray(groups)
        if len(groups_arr) != n:
            raise ValueError(
                f"{model_name}: groups length {len(groups_arr)} != "
                f"n_samples {n}")
        if len(np.unique(groups_arr)) >= 2:
            from sklearn.model_selection import GroupShuffleSplit
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                    random_state=42)
            train_idx, ho_idx = next(
                gss.split(np.zeros((n, 1)), y, groups=groups_arr))
            if len(train_idx) > 0 and len(ho_idx) > 0:
                return np.asarray(train_idx), np.asarray(ho_idx)
        # One group (or a degenerate carve): a group-disjoint holdout does not
        # exist. Fall through to the random carve, but say so.
        _warn_once(
            f'{model_name}__degenerate_groups',
            f"{model_name}.fit: groups has <2 distinct values (or the "
            f"group-aware carve was empty) — falling back to a RANDOM "
            f"early-stopping holdout.")
    else:
        _warn_once(
            model_name,
            f"{model_name}.fit called without groups — internal early-"
            f"stopping split is random and may leak scaffolds.")

    idx = rng.permutation(n)
    split = int((1.0 - test_size) * n)
    return idx[:split], idx[split:]
