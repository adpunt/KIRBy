#!/usr/bin/env python3
"""The four ways of building an is-this-a-cliff classifier, as she defined them.

HER INSTRUCTION, 2026-11-28: "so I'm exploring different ways to have a AC or not
classifier. I can haveg multiple, that's totally fine, but I need them clearly defined", then
"lets test these ... try differnet models (ideally using the onfiugrations mentioned in
papers), use the respective datasets. This is its own sweep."

THE FOUR. Each one is a different answer to "what is the classifier handed", and each is
scored on the same held-out pairs so the four numbers can be read against each other. Sources
for every published detail are in thesis_provenance/LITERATURE_SEARCH_TBD7_2026-11-28.md.

  per_molecule   one molecule's representation vector in, and the prediction is whether that
                 molecule carries MoleculeACE's own `cliff_mol` flag. This is the definition
                 moleculeace_cliff_model.py already ran. It is here as the comparison, scored
                 on molecules rather than pairs, and no published work uses it.

  mmp_kernel     a support vector machine on the matched molecular pair kernel. The published
                 kernel multiplies a Tanimoto kernel on the pair's shared core by a Tanimoto
                 kernel on the two substituents that differ (Molecules 26 (2021) 4916). Two
                 pairs each have two substituents and no source read says how to compare two
                 unordered sets of two, so the substituent half here is the average of the two
                 matchings, which keeps the kernel symmetric and independent of the order the
                 pair was stored in. A Tanimoto kernel has no gamma, so the penalty C is the
                 only setting, chosen by inner cross-validation as the published work does.
                 This was the strongest method over 100 activity classes (Tamura, Miyao and
                 Bajorath 2023), by small margins over eight others.

  pair_vector    one vector built from both molecules, handed to any classifier in this
                 repository's own registry. Three constructions run, because the sources
                 disagree and one is already ruled out:
                   sum_absdiff   the elementwise sum and the elementwise absolute difference,
                                 concatenated. Independent of which molecule is stored first.
                   concat        the two vectors end to end, ordered by canonical SMILES so
                                 the order carries no information about potency.
                   absdiff       the absolute difference alone. Dablander et al. call the
                                 plain difference "consistently inferior to ECFPs and GINs",
                                 so this one has a published result against it and runs as
                                 the check of that result, not as a candidate.

  twin           both molecules through the same small network with shared weights, the two
                 outputs combined by their absolute difference and their product, and a head
                 trained on the pair label with binary cross-entropy. This is the architecture
                 of the 2024 ICML workshop paper and of Dablander's thesis version.

WHAT IS SCORED. Matthews correlation and balanced accuracy, which are the two Tamura, Miyao
and Bajorath report, plus AUROC. Accuracy is not reported: the share of cliff pairs on one
MoleculeACE target ran from 2.2 to 17.7 per cent in the 2026-11-28 crash check, and accuracy
tracks that share.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent.parent
REPO = TESTS.parent
for _p in (REPO / 'src', TESTS, TESTS / 'experiments'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

#: How a pair becomes one vector. `absdiff` is the one with a published result against it.
PAIR_INPUTS = ('sum_absdiff', 'concat', 'absdiff')

#: The classifiers run on a pair vector. Every name is in this repository's own registry
#: (kirby.find_best_hybrid.DEFAULT_MODELS), and the list covers the families Tamura, Miyao and
#: Bajorath compared: a nearest-neighbour classifier, tree ensembles, a kernel method and a
#: neural network.
PAIR_VECTOR_MODELS = ('knn', 'rf', 'lightgbm', 'xgboost', 'svm', 'dnn')

#: The penalty the support vector machine's inner cross-validation chooses from. The published
#: work selects C in inner folds and states no value, so this is a decade either side of 1.
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)

#: Bits in the Morgan fingerprint of a core or a substituent fragment.
FRAGMENT_BITS = 2048

#: Radius of that fingerprint. 2 is ECFP4, which is what every source read uses.
FRAGMENT_RADIUS = 2


# ---------------------------------------------------------------------------
# Fragment fingerprints and the matched molecular pair kernel
# ---------------------------------------------------------------------------

def fragment_fingerprints(fragment_smiles):
    """One Morgan fingerprint per fragment SMILES, as a 0/1 array.

    A fragment carries a dummy atom from the cut, which RDKit parses, so the fingerprint is
    taken on the fragment as written.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=FRAGMENT_RADIUS, fpSize=FRAGMENT_BITS)
    cache, out = {}, np.zeros((len(fragment_smiles), FRAGMENT_BITS),
                              dtype=np.float32)
    for row, smi in enumerate(fragment_smiles):
        if smi in cache:
            out[row] = cache[smi]
            continue
        mol = Chem.MolFromSmiles(smi)
        vec = np.zeros(FRAGMENT_BITS, dtype=np.float32)
        if mol is not None:
            arr = gen.GetFingerprintAsNumPy(mol)
            vec = np.asarray(arr, dtype=np.float32)
        cache[smi] = vec
        out[row] = vec
    return out


def tanimoto_kernel(left, right):
    """Tanimoto similarity between every row of `left` and every row of `right`.

    Both are 0/1 arrays with the same number of features. A row of all zeros has similarity 0
    to everything, including itself, because the Tanimoto of two empty sets is undefined.
    """
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    shared = left @ right.T
    left_on = left.sum(axis=1)[:, None]
    right_on = right.sum(axis=1)[None, :]
    union = left_on + right_on - shared
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(union > 0, shared / np.maximum(union, 1e-12), 0.0)
    return out.astype(np.float64)


def mmp_kernel(core_a, sub_i_a, sub_j_a, core_b, sub_i_b, sub_j_b):
    """The matched molecular pair kernel between two sets of pairs.

    The published kernel is a product of a Tanimoto kernel on the core and a Tanimoto kernel
    on the substituents. The substituent half here averages the two ways of matching one
    pair's two substituents to the other pair's, so the value does not depend on which
    molecule of a pair was stored first.
    """
    core = tanimoto_kernel(core_a, core_b)
    straight = tanimoto_kernel(sub_i_a, sub_i_b) * tanimoto_kernel(sub_j_a, sub_j_b)
    crossed = tanimoto_kernel(sub_i_a, sub_j_b) * tanimoto_kernel(sub_j_a, sub_i_b)
    return core * 0.5 * (straight + crossed)


# ---------------------------------------------------------------------------
# The pair vector
# ---------------------------------------------------------------------------

def pair_vectors(X, pairs, how, order_key=None):
    """One vector per pair, built from the two molecules' representation vectors.

    Args:
        X: one representation vector per molecule.
        pairs: the dict build_pairs returned.
        how: one of PAIR_INPUTS.
        order_key: a sortable value per molecule used to put the two molecules of a pair in a
            fixed order for `concat`. Canonical SMILES is what the sweep passes. Without it,
            `concat` would encode the dataset's own molecule order.
    """
    if how not in PAIR_INPUTS:
        raise ValueError(f'no such pair input {how!r}; they are {PAIR_INPUTS}')
    X = np.asarray(X, dtype=np.float32)
    i, j = pairs['i'], pairs['j']
    if how == 'sum_absdiff':
        return np.hstack([X[i] + X[j], np.abs(X[i] - X[j])])
    if how == 'absdiff':
        return np.abs(X[i] - X[j])
    if order_key is None:
        raise ValueError("'concat' needs order_key, or the dataset's own "
                         'molecule order would become a feature')
    key = np.asarray(order_key, dtype=object)
    first_is_i = np.array([key[a] <= key[b] for a, b in zip(i, j)])
    left = np.where(first_is_i[:, None], X[i], X[j])
    right = np.where(first_is_i[:, None], X[j], X[i])
    return np.hstack([left, right])


# ---------------------------------------------------------------------------
# The three fitted classifiers
# ---------------------------------------------------------------------------

def fit_mmp_kernel_svm(train, test, y_train, n_inner=3, seed=42):
    """A support vector machine on the matched molecular pair kernel.

    `train` and `test` are each a dict of the four fingerprint blocks for those pairs:
    `core`, `sub_i`, `sub_j`. C is chosen by inner cross-validation over C_GRID on the
    training pairs, scored by Matthews correlation, as the published work selects C in inner
    folds.

    Returns (scores_on_test, chosen_C). `scores_on_test` is the decision function, higher
    meaning more cliff-like.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import matthews_corrcoef
    from sklearn.svm import SVC

    y_train = np.asarray(y_train, dtype=int)
    gram = mmp_kernel(train['core'], train['sub_i'], train['sub_j'],
                      train['core'], train['sub_i'], train['sub_j'])

    best_c, best_score = C_GRID[0], -2.0
    if len(set(y_train.tolist())) > 1 and len(y_train) >= 2 * n_inner:
        folds = list(StratifiedKFold(n_splits=n_inner, shuffle=True,
                                     random_state=seed).split(gram, y_train))
        for c in C_GRID:
            got = []
            for inner_train, inner_test in folds:
                if len(set(y_train[inner_train].tolist())) < 2:
                    continue
                model = SVC(C=c, kernel='precomputed',
                            class_weight='balanced')
                model.fit(gram[np.ix_(inner_train, inner_train)],
                          y_train[inner_train])
                pred = model.predict(gram[np.ix_(inner_test, inner_train)])
                got.append(matthews_corrcoef(y_train[inner_test], pred))
            if got and float(np.mean(got)) > best_score:
                best_score, best_c = float(np.mean(got)), c

    model = SVC(C=best_c, kernel='precomputed', class_weight='balanced')
    model.fit(gram, y_train)
    cross = mmp_kernel(test['core'], test['sub_i'], test['sub_j'],
                       train['core'], train['sub_i'], train['sub_j'])
    return model.decision_function(cross), best_c


def fit_pair_vector_model(name, X_train, y_train, X_test, n_samples=None):
    """One registry classifier on the pair vectors.

    Returns scores on the test pairs, higher meaning more cliff-like.
    """
    from experiments._shared import get_model_factories

    y_train = np.asarray(y_train, dtype=int)
    factories = get_model_factories(
        [name], 'classification',
        len(y_train) if n_samples is None else n_samples)
    factory = factories.get(name)
    if factory is None:
        raise RuntimeError(f'the registry has no classification factory for '
                           f'{name!r} at this size')
    model = factory()
    model.fit(X_train, y_train)
    if hasattr(model, 'predict_proba'):
        prob = np.asarray(model.predict_proba(X_test))
        if prob.ndim == 2 and prob.shape[1] == 2:
            return prob[:, 1]
        return prob.ravel()
    if hasattr(model, 'decision_function'):
        return np.asarray(model.decision_function(X_test)).ravel()
    return np.asarray(model.predict(X_test), dtype=float)


def fit_twin_network(X, pairs_train, y_train, pairs_test, hidden=256,
                     embedding=64, epochs=60, batch=128, lr=1e-3, seed=42):
    """A network that sees both molecules of a pair through shared weights.

    Both molecules pass through the same two-layer encoder. The two outputs are combined by
    their absolute difference and their elementwise product, which is symmetric in the two
    molecules, and a head returns one number per pair. Training is binary cross-entropy with
    the positive class weighted by how rare it is.

    Returns scores on the test pairs, higher meaning more cliff-like.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    device = 'cpu'
    X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
    y_t = torch.tensor(np.asarray(y_train, dtype=np.float32), device=device)
    i_tr = torch.tensor(np.asarray(pairs_train['i'], dtype=np.int64))
    j_tr = torch.tensor(np.asarray(pairs_train['j'], dtype=np.int64))
    i_te = torch.tensor(np.asarray(pairs_test['i'], dtype=np.int64))
    j_te = torch.tensor(np.asarray(pairs_test['j'], dtype=np.int64))

    encoder = nn.Sequential(
        nn.Linear(X_t.shape[1], hidden), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden, embedding), nn.ReLU()).to(device)
    head = nn.Sequential(
        nn.Linear(2 * embedding, embedding), nn.ReLU(),
        nn.Linear(embedding, 1)).to(device)

    n_pos = float(y_t.sum().item())
    n_neg = float(len(y_t) - n_pos)
    pos_weight = torch.tensor([n_neg / n_pos if n_pos > 0 else 1.0],
                              device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=lr)

    def forward(i_idx, j_idx):
        left = encoder(X_t[i_idx])
        right = encoder(X_t[j_idx])
        return head(torch.cat([torch.abs(left - right), left * right],
                              dim=1)).squeeze(1)

    n = len(y_t)
    for _epoch in range(epochs):
        encoder.train(); head.train()
        order = torch.randperm(n)
        for start in range(0, n, batch):
            sel = order[start:start + batch]
            optimiser.zero_grad()
            loss = loss_fn(forward(i_tr[sel], j_tr[sel]), y_t[sel])
            loss.backward()
            optimiser.step()

    encoder.eval(); head.eval()
    with torch.no_grad():
        return forward(i_te, j_te).cpu().numpy()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def scores(y_true, y_score, threshold=0.0, positive_rate=None):
    """Matthews correlation, balanced accuracy and AUROC on one group of pairs.

    `threshold` turns a score into a prediction. A probability needs 0.5 and a decision
    function needs 0.0, so the caller passes the right one.
    """
    from sklearn.metrics import (balanced_accuracy_score, matthews_corrcoef,
                                 roc_auc_score)

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    out = {'n_pairs': int(len(y_true)),
           'n_cliff_pairs': int(y_true.sum()),
           'share_cliff_pairs': (float(y_true.mean()) if len(y_true)
                                 else None)}
    if len(y_true) == 0:
        return {**out, 'matthews_correlation': None,
                'balanced_accuracy': None, 'auroc': None}
    pred = (y_score > threshold).astype(int)
    one_class = len(set(y_true.tolist())) < 2
    out['matthews_correlation'] = (None if one_class
                                   else float(matthews_corrcoef(y_true, pred)))
    out['balanced_accuracy'] = (None if one_class else
                                float(balanced_accuracy_score(y_true, pred)))
    try:
        out['auroc'] = (None if one_class
                        else float(roc_auc_score(y_true, y_score)))
    except ValueError:
        out['auroc'] = None
    if positive_rate is not None:
        out['predicted_cliff_share'] = float(pred.mean())
    return out
