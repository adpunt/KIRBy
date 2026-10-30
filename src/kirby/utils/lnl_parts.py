"""The in-loop label-noise methods, split into a detector and a cleanup.

Written 2026-10-30 from HANDOFF_NOISE_RESTRUCTURE.md, section 6 step 2. The
author's decision of that day: every method that has a test and an action
inside it is split, so the detector-by-cleanup sweep can cross its test with
every cleanup and its action with every detector. The published method keeps
running whole, in lnl_methods.py; nothing there is replaced by this file.

DETECTORS. One score per training molecule, higher = the label looks more
wrong. Each trains a network on the molecules it is asked about and reads the
method's own test off that network. The classes in noise_estimation.py's
ESTIMATOR_REGISTRY score once before training; these do the same, once, after
a warm-up, which is the first time the method's own loop reads its test
(small-loss selection and DivideMix both warm up for 10 epochs before their
first selection). A method that re-reads its test every epoch is the rounds
axis of the sweep, not a second detector kind.

    net_small_loss         the network's per-molecule loss after the warm-up
                           (small-loss selection, co-teaching, self-paced
                           learning and multistage cleaning all rank by this)
    net_loss_mixture       DivideMix's test: a two-component mixture on the
                           per-molecule loss scaled to [0, 1]; the score is
                           the posterior of the high-loss component
    net_josrc_test         Jo-SRC's in-distribution-noisy test: the loss rank
                           times one minus the rank of the prediction's
                           variance over jittered copies of the features
    net_meta_weight        Meta-Weight-Net's learned weight, one minus it,
                           read off the fully trained method
    net_evidential_spread  the evidential network's per-molecule aleatoric
                           variance beta / (alpha - 1), from the kept network

Not here, and why: early learning computes a weight that decays by 0.95 on
every epoch a molecule's loss is above the median, so its test is the loss
rank again (net_small_loss) and its action is a down-weight; MentorNet waits on
decision H of thesis_provenance/NOISE_CODE_FIX_DECISIONS_2026-10-26.md; the
heteroscedastic GP's test is the registry's `hetero_gp`; distance-based
selection's test is the registry's `local_variance_ratio`; confident learning's
test is noise_helpers.cleanlab_suspicion.

CLEANUPS. Each reads scores or flags and returns the training set to fit.
Registered in n07_remediation.cleaned_training_set under these names:

    score_weight           the weight IS one minus the score (the method's
                           own weight when the detector returns one minus a
                           weight, as net_meta_weight does)
    spread_invvar_weight   the weight is one over the score, for a detector
                           whose score is a variance
    network_rewrite        DivideMix's action, once: the flagged labels are
                           replaced by the prediction of a network trained on
                           the unflagged molecules
    ema_rewrite            Jo-SRC's action, once: the flagged labels are
                           replaced by the prediction of the slowly updated
                           (exponential moving average) copy of a network
                           trained on every molecule

The functions here are the arithmetic; the estimator classes live in
noise_estimation.py so the registry stays the one place detectors are listed.
"""
import numpy as np


# The warm-up every network detector scores after. SmallLossSelection and
# DivideMixReg both read their test for the first time after 10 epochs.
WARMUP_EPOCHS = 10
# Jo-SRC test settings, the class's own (lnl_methods.JoSRCWeighting).
JOSRC_PERTURBATIONS = 5
JOSRC_SIGMA = 0.01
# Jo-SRC's mean-teacher update, omega = 0.99 (Yao et al. 2021, implementation
# details), and the epochs the student trains for the once-only rewrite. The
# epoch count is the method's warm-up plus the plain DNN's patience, so the
# teacher has averaged over as many epochs as the plain network trains for
# after it stops improving.
EMA_OMEGA = 0.99
EMA_EPOCHS = 30


# ---------------------------------------------------------------------------
# the warm-up network
# ---------------------------------------------------------------------------

def _prep(X, y, is_clf, classes_=None):
    """Features as float32, labels as the network's target, plus what is
    needed to read a loss: (X_t, y_target, info)."""
    from kirby.utils.lnl_methods import _z_normalize
    X_t = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y).ravel()
    if is_clf:
        classes_ = (np.asarray(classes_) if classes_ is not None
                    else np.unique(y_arr))
        label_map = {c: i for i, c in enumerate(classes_)}
        y_int = np.array([label_map[c] for c in y_arr], dtype=np.int64)
        return X_t, y_int, {'classes_': classes_, 'n_classes': len(classes_)}
    y_norm, y_mean, y_std = _z_normalize(y_arr)
    return X_t, y_norm, {'y_mean': y_mean, 'y_std': y_std}


def warm_network(X, y, *, is_clf=False, classes_=None, epochs=WARMUP_EPOCHS,
                 random_state=42, hidden_sizes=(256, 128, 64), dropout=0.2,
                 lr=0.001, batch_size=256, ema_omega=None):
    """Train the suite's DNN on every row for a fixed number of epochs, no
    early stopping and no carve: the warm-up SmallLossSelection runs.

    ema_omega: when given, a second copy of the network is kept whose weights
    are the exponential moving average of the student's after every
    optimiser step (Jo-SRC's mean-teacher). Returned as 'teacher'.

    Returns a dict: net, teacher (or None), X_t, y_target, info.
    """
    import torch
    import torch.nn as nn
    from kirby.utils.lnl_methods import _build_dnn
    torch.manual_seed(random_state)
    X_t, y_target, info = _prep(X, y, is_clf, classes_)
    n = len(y_target)
    out_dim = info['n_classes'] if is_clf else 1
    net = _build_dnn(X_t.shape[1], hidden_sizes, dropout, output_dim=out_dim)
    teacher = None
    if ema_omega is not None:
        import copy
        teacher = copy.deepcopy(net)
        for p in teacher.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    X_tensor = torch.from_numpy(X_t)
    y_tensor = torch.from_numpy(y_target)
    rng = np.random.RandomState(random_state)
    for _epoch in range(int(epochs)):
        net.train()
        perm = torch.from_numpy(rng.permutation(n))
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            out = net(X_tensor[idx])
            yb = y_tensor[idx]
            if is_clf:
                loss = nn.functional.cross_entropy(out, yb)
            else:
                loss = nn.functional.mse_loss(out.squeeze(-1), yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if teacher is not None:
                with torch.no_grad():
                    for pt, ps in zip(teacher.parameters(), net.parameters()):
                        pt.mul_(ema_omega).add_(ps, alpha=1.0 - ema_omega)
    net.eval()
    if teacher is not None:
        teacher.eval()
    return {'net': net, 'teacher': teacher, 'X_t': X_t,
            'y_target': y_target, 'info': info, 'is_clf': is_clf}


def per_molecule_loss(warm, X_t=None):
    """The network's loss on each row: squared error on the z-scored label,
    or cross-entropy on the class."""
    import torch
    import torch.nn as nn
    X_t = warm['X_t'] if X_t is None else X_t
    with torch.no_grad():
        out = warm['net'](torch.from_numpy(np.asarray(X_t, dtype=np.float32)))
        y = torch.from_numpy(warm['y_target'])
        if warm['is_clf']:
            return nn.functional.cross_entropy(
                out, y, reduction='none').numpy().astype(float)
        return ((out.squeeze(-1) - y) ** 2).numpy().astype(float)


def _predict(net, X_t, is_clf):
    """The network's prediction on X_t: the z-scored value, or the class
    probabilities."""
    import torch
    with torch.no_grad():
        out = net(torch.from_numpy(np.asarray(X_t, dtype=np.float32)))
        if is_clf:
            return torch.softmax(out, dim=1).numpy()
        return out.squeeze(-1).numpy()


# ---------------------------------------------------------------------------
# detector tests
# ---------------------------------------------------------------------------

def loss_mixture_scores(losses, random_state=42):
    """DivideMix's test (DivideMixReg._divide): a two-component Gaussian
    mixture on the losses scaled to [0, 1] by their min and max. Returns
    the posterior of the HIGH-loss component, so a label the mixture calls
    noisy scores near 1. All ones when the mixture cannot be fitted."""
    from sklearn.mixture import GaussianMixture
    losses = np.asarray(losses, dtype=float)
    lo, hi = float(losses.min()), float(losses.max())
    l01 = ((losses - lo) / (hi - lo + 1e-12)).reshape(-1, 1)
    try:
        g = GaussianMixture(n_components=2, random_state=random_state)
        g.fit(l01)
        noisy_comp = int(np.argmax(g.means_.ravel()))
        return g.predict_proba(l01)[:, noisy_comp]
    except Exception:
        return np.ones(len(losses), dtype=float)


def josrc_scores(warm, n_perturbations=JOSRC_PERTURBATIONS, sigma=JOSRC_SIGMA,
                 random_state=42):
    """Jo-SRC's in-distribution-noisy test as JoSRCWeighting computes it:
    the loss rank times one minus the rank of the prediction's variance over
    n_perturbations copies of the features with Gaussian jitter of sigma.
    High loss with a prediction that holds still scores near 1; high loss
    with a prediction that moves (the network is unsure) scores low. Both
    ranks are divided by n, so the score is in [0, 1)."""
    import torch
    losses = per_molecule_loss(warm)
    X_t = warm['X_t']
    n = len(losses)
    rng = np.random.RandomState(random_state)
    signal = []
    with torch.no_grad():
        y = torch.from_numpy(warm['y_target'])
        for _ in range(int(n_perturbations)):
            noise = rng.normal(0, sigma, X_t.shape).astype(np.float32)
            out = warm['net'](torch.from_numpy(X_t + noise))
            if warm['is_clf']:
                proba = torch.softmax(out, dim=1)
                signal.append(proba.gather(1, y.unsqueeze(1)).squeeze(1).numpy())
            else:
                signal.append(out.squeeze(-1).numpy())
    variance = np.var(np.asarray(signal), axis=0)
    loss_rank = np.argsort(np.argsort(losses)) / n
    var_rank = np.argsort(np.argsort(variance)) / n
    return (loss_rank * (1.0 - var_rank)).astype(float)


def meta_weight_scores(X, y, *, is_clf=False, classes_=None, groups=None,
                       random_state=42, **method_kwargs):
    """Meta-Weight-Net's test: one minus the weight its meta-network gives
    each molecule, read off the fully trained method (lnl_methods.
    MetaWeightNet, metadata 'meta_weights_all'). The weight is in (0, 1),
    so the score is too."""
    from kirby.utils.lnl_methods import MetaWeightNet
    m = MetaWeightNet(random_state=random_state, is_clf=is_clf,
                      classes_=classes_, **method_kwargs)
    m.selection_groups = groups
    X_t = np.asarray(X, dtype=np.float32)
    res = m.train_and_predict(X_t, np.asarray(y).ravel(), X_t[:1], None)
    w = np.asarray(res['metadata']['meta_weights_all'], dtype=float)
    return 1.0 - w


def evidential_spread_scores(X, y, *, groups=None, random_state=42,
                             **method_kwargs):
    """The evidential network's test: its per-molecule aleatoric variance
    beta / (alpha - 1) from the kept network (lnl_methods.EvidentialKernel,
    metadata 'aleatoric_all'), in z-scored label units squared. Regression
    only, as the method is."""
    from kirby.utils.lnl_methods import EvidentialKernel
    m = EvidentialKernel(random_state=random_state, **method_kwargs)
    m.selection_groups = groups
    X_t = np.asarray(X, dtype=np.float32)
    res = m.train_and_predict(X_t, np.asarray(y).ravel(), X_t[:1], None)
    return np.asarray(res['metadata']['aleatoric_all'], dtype=float)


# ---------------------------------------------------------------------------
# cleanups
# ---------------------------------------------------------------------------

def score_weight(noise_scores):
    """The weight is one minus the score, divided by its mean. Defined only
    for a score in [0, 1]; returns None (the caller declines) otherwise, so a
    detector in other units is not quietly rescaled into a weight."""
    s = np.asarray(noise_scores, dtype=float)
    if not np.all(np.isfinite(s)) or s.min() < 0 or s.max() > 1:
        return None
    w = 1.0 - s
    if w.mean() <= 0:
        return None
    return w / w.mean()


def spread_invvar_weight(noise_scores):
    """The weight is one over the score, divided by its mean: the
    inverse-variance weight for a detector whose score is a per-molecule
    variance. None when any score is not a positive finite number."""
    s = np.asarray(noise_scores, dtype=float)
    if not np.all(np.isfinite(s)) or s.min() <= 0:
        return None
    w = 1.0 / s
    return w / w.mean()


def network_rewrite(X, y, flagged, *, is_clf=False, classes_=None,
                    groups=None, random_state=42):
    """DivideMix's action, once: the flagged labels are replaced by the
    prediction of the suite's DNN trained on the unflagged molecules (the
    plain trainer, early stopping on a carved fifth of those molecules).

    DivideMix guesses a noisy molecule's label with networks that have been
    training on every molecule, the noisy ones under guessed labels from the
    epoch before. Once, with no epoch before, the network that guesses is
    trained on the molecules the detector did not flag, so a flagged label
    cannot be reproduced into its own replacement.

    Returns the rewritten label column, or None when fewer than 10
    molecules are unflagged.
    """
    from kirby.utils.lnl_methods import (
        _standard_dnn_train_predict, _standard_dnn_train_predict_clf)
    y_arr = np.asarray(y)
    flagged = np.asarray(flagged, dtype=bool)
    keep = ~flagged
    if keep.sum() < 10 or flagged.sum() == 0:
        return None
    X_t = np.asarray(X, dtype=np.float32)
    g = None
    if groups is not None and len(np.asarray(groups)) == len(y_arr):
        g = np.asarray(groups)[keep]
    if is_clf:
        classes_ = (np.asarray(classes_) if classes_ is not None
                    else np.unique(y_arr))
        pred = _standard_dnn_train_predict_clf(
            X_t[keep], y_arr[keep], X_t[flagged], random_state=random_state,
            classes_=classes_, groups=g)
        pred = np.asarray(pred)
        if len(classes_) == 2 and pred.ndim == 1 and pred.dtype.kind == 'f':
            # the binary trainer returns P(class 1); the label is the class
            pred = classes_[(pred >= 0.5).astype(int)]
    else:
        pred = _standard_dnn_train_predict(
            X_t[keep], y_arr[keep], X_t[flagged], random_state=random_state,
            groups=g)
    y_new = y_arr.copy()
    y_new[flagged] = np.asarray(pred).ravel()[:int(flagged.sum())]
    return y_new


def ema_rewrite(X, y, flagged, *, is_clf=False, classes_=None,
                random_state=42, epochs=EMA_EPOCHS, omega=EMA_OMEGA):
    """Jo-SRC's action, once: the flagged labels are replaced by the
    prediction of the mean-teacher, the exponential moving average
    (omega = 0.99 per optimiser step) of a network trained on every
    molecule for `epochs` epochs. Jo-SRC relabels its in-distribution
    noisy samples with exactly this copy (Yao et al. 2021, eq. 6); the
    student there trains on the relabelled targets as they change, which a
    once-only cleanup cannot do, so here the student trains on the given
    labels. Regression: the teacher's value; classification: its most
    probable class. Returns the rewritten column, or None when nothing is
    flagged.
    """
    y_arr = np.asarray(y)
    flagged = np.asarray(flagged, dtype=bool)
    if flagged.sum() == 0:
        return None
    warm = warm_network(X, y_arr, is_clf=is_clf, classes_=classes_,
                        epochs=epochs, random_state=random_state,
                        ema_omega=omega)
    pred = _predict(warm['teacher'], warm['X_t'][flagged], is_clf)
    y_new = y_arr.copy()
    if is_clf:
        y_new[flagged] = warm['info']['classes_'][np.argmax(pred, axis=1)]
    else:
        info = warm['info']
        y_new[flagged] = pred * info['y_std'] + info['y_mean']
    return y_new
