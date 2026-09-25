"""Which features a noise experiment finds a molecule's neighbours in.

WHY THIS EXISTS. Every neighbour-based mitigation method in the noise suite
finds neighbours in the same features the model is fitted on (n07:826-917,
inherited by n17, n18, n20, n23, n24, n25). The thesis question is whether a
better measure of similarity makes those methods start working, so the
features neighbours are found in has to be its own axis, swept separately from
the features the model is fitted on. Plan section 1
(~/.claude/plans/pasted-content-id-0428-noise-experiment-fizzy-ladybug.md).

A similarity setting is three choices:

  space      which features the neighbours are found in
               tanimoto_ecfp4     2048-bit Morgan radius-2 bits from SMILES,
                                  distance 1 - Tanimoto (n01:92-122)
               base_1, base_2     each model's top 2 single representations
               base_1_aug_capped,     each of those plus augmented
               base_1_aug_uncapped,   features, built per fold the way c21
               base_2_aug_capped,     builds 'B_aug', under both of c21's
               base_2_aug_uncapped    admission methods (c21:200-207):
                                      capped extras displace base features
                                      inside the budget, uncapped extras are
                                      appended outside it. Both, owner
                                      decision 2026-09-25 (base_aug_blueprint)
               hybrid             the prediction-built hybrid
               hybrid_similarity  the similarity-built hybrid
               forest_proximity   1 - share of trees in which two molecules
                                  share a leaf (noise_estimation.py:2119-2162)
  weighting  none, lightgbm_gain, treeshap, deeplift. The importance is learned
             on the space's TRAINING rows with the labels the caller holds --
             the noisy ones, owner decision -- and folded into the standardised
             features by kirby.similarity.preprocessing.fold_weights (mean 1,
             exponent matched to the distance). Only 'none' is defined for
             tanimoto_ecfp4 and forest_proximity.
  metric     manhattan or euclidean for the feature spaces; 'tanimoto' for
             tanimoto_ecfp4; 'forest' for forest_proximity.

8 feature spaces x 4 weightings x 2 distances = 64, plus Tanimoto and forest
proximity = 66 (enumerate_settings). Until 2026-09-25 there was one augmented
space per base representation (base_k_aug, uncapped only) and the count was
6 x 4 x 2 + 2 = 50; the owner asked for both admission methods on 2026-09-25.

k is swept over SWEPT_KS = 5, 9, 10, 20: every count in use today (9 for
label_smooth / knn_smooth, 10 for prox_smooth and the four neighbour
estimators; see below) plus 5 and 20 (owner decision 2026-09-25). All are read
off one list of k_max = 20 neighbours per molecule, so k is not part of a
setting.

KIRBY_SIMILARITY_SETTINGS picks which settings a run sweeps
(settings_from_env): unset or 'all' is every setting (enumerate_settings);
'model_input' is [MODEL_INPUT] alone, today's numbers bit for bit; otherwise a
comma list of setting names as SimilaritySetting.name writes them.

THE MODEL_INPUT SENTINEL is today's behaviour: neighbours in the model's own
matrix, the process-wide distance (make_nearest_neighbors,
tests/experiments/_shared.py:2865), no weighting. Handing its lists to a
consumer reproduces today's numbers bit for bit.

⛔ WHY MODEL_INPUT CARRIES ONE LIST PER k, NOT ONE LIST CUT TO k. sklearn's
nearest-neighbour query orders TIED neighbours differently depending on how
many it was asked for. MEASURED 2026-09-25 on this machine, sklearn 1.6.1, 300
rows of 0/1 features under manhattan: the 9 neighbours returned by a query for
10 differed from the first 9 returned by a query for 21 on 300 of 300 rows, at
64 features (brute force) and at 8 and 5 features (kd-tree). Binary
fingerprints are full of ties. So a k_max list cut to k is NOT today's list,
and MODEL_INPUT stores the exact query today's code issues for each k a
consumer reads (`exact_by_k`). Every other setting is computed here with a
stable sort on (distance, row index), so its first k neighbours ARE the k
nearest and one k_max list serves every k exactly.

⛔ TODAY'S CONSUMERS DO NOT AGREE ON k. `label_smooth` and `knn_smooth`
(n07:829-855) query min(10, n-1) rows INCLUDING the molecule itself and drop
the first, so they average 9 neighbours. `prox_smooth` (n07:907) takes
min(10, n-1) neighbours with the molecule excluded, so 10. The four
neighbour estimators (noise_estimation.py:204-262) take 10. MODEL_INPUT stores
all of those counts.

CPU only: numpy, scipy, sklearn. Pairwise distances are computed in row
chunks, so memory is bounded by KIRBY_NEIGHBOUR_CHUNK_MB (default 256) and not
by the square of the training count.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Optional

import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ---------------------------------------------------------------------------
# The axis
# ---------------------------------------------------------------------------

# The augmented spaces, one per base representation and c21 admission method.
# Replaced 'base_1_aug' / 'base_2_aug' (uncapped only) on 2026-09-25.
AUG_STRATEGIES = ('capped', 'uncapped')
AUG_SPACES = tuple(f'base_{i}_aug_{st}' for i in (1, 2) for st in AUG_STRATEGIES)
FEATURE_SPACES = ('base_1', 'base_2') + AUG_SPACES + (
    'hybrid', 'hybrid_similarity')
TANIMOTO_SPACE = 'tanimoto_ecfp4'
FOREST_SPACE = 'forest_proximity'
MODEL_INPUT_SPACE = 'model_input'
ALL_SPACES = FEATURE_SPACES + (TANIMOTO_SPACE, FOREST_SPACE)

WEIGHTINGS = ('none', 'lightgbm_gain', 'treeshap', 'deeplift')
FEATURE_METRICS = ('manhattan', 'euclidean')

# The ks swept, owner decision 2026-09-25: every count in use today (9 for
# label_smooth / knn_smooth, n07:829-842; 10 for prox_smooth, n07:907, and the
# four neighbour estimators, noise_estimation.py) plus 5 and 20. Read off one
# k_max list. SWEEP_KS is the name this had before 2026-09-25, when it was
# (5, 10, 20); kept as an alias of the new tuple.
SWEPT_KS = (5, 9, 10, 20)
SWEEP_KS = SWEPT_KS
DEFAULT_K_MAX = 20

# scipy cdist names for the two feature-space distances.
_CDIST = {'manhattan': 'cityblock', 'euclidean': 'euclidean'}


@dataclass(frozen=True)
class SimilaritySetting:
    """One way of finding neighbours. See the module docstring for the values.

    `name` is stable across runs and processes: it is what goes into an output
    row's `similarity_setting` column and into a cache key. k_max is not in
    the name, because it is the length of the list, not a choice about
    similarity.
    """
    space: str
    weighting: str
    metric: str
    k_max: int = DEFAULT_K_MAX

    def __post_init__(self):
        sp, w, m = self.space, self.weighting, self.metric
        if sp == MODEL_INPUT_SPACE:
            if w != 'none' or m != 'process':
                raise ValueError(
                    "model_input means today's behaviour: no weighting and the "
                    "process-wide distance, so it takes weighting='none', "
                    f"metric='process'. Got {w!r}, {m!r}.")
        elif sp == TANIMOTO_SPACE:
            if w != 'none' or m != 'tanimoto':
                raise ValueError(
                    "tanimoto_ecfp4 is fixed bits and a fixed formula: "
                    f"weighting='none', metric='tanimoto'. Got {w!r}, {m!r}.")
        elif sp == FOREST_SPACE:
            if w != 'none' or m != 'forest':
                raise ValueError(
                    "forest_proximity is the forest's own similarity: "
                    f"weighting='none', metric='forest'. Got {w!r}, {m!r}.")
        elif sp in FEATURE_SPACES:
            if w not in WEIGHTINGS:
                raise ValueError(f"weighting {w!r} not in {WEIGHTINGS}")
            if m not in FEATURE_METRICS:
                raise ValueError(f"metric {m!r} not in {FEATURE_METRICS}")
        else:
            raise ValueError(
                f"space {sp!r} not in {ALL_SPACES + (MODEL_INPUT_SPACE,)}")
        if int(self.k_max) < 1:
            raise ValueError(f"k_max must be >= 1, got {self.k_max}")

    @property
    def name(self) -> str:
        return f"{self.space}|{self.weighting}|{self.metric}"

    def to_dict(self) -> dict:
        return asdict(self)


# Today's behaviour. See the module docstring.
MODEL_INPUT = SimilaritySetting(MODEL_INPUT_SPACE, 'none', 'process')


def enumerate_settings(available_spaces: Optional[Iterable[str]] = None,
                       k_max: int = DEFAULT_K_MAX):
    """Every similarity setting of plan section 1 whose space is available.

    `available_spaces` None means all ten. With all ten this is 66:
    8 feature spaces x 4 weightings x 2 distances, plus tanimoto_ecfp4 and
    forest_proximity (50 before 2026-09-25, when the augmented space split
    into capped and uncapped). MODEL_INPUT is not in the list; a caller adds it when it
    wants today's behaviour as a row.

    A space missing from `available_spaces` is dropped, not faked. Which spaces
    are available is the caller's fact to state -- for example
    hybrid_similarity does not exist on tcga_brca_pam50 or tz_her2_binary.
    """
    avail = set(ALL_SPACES if available_spaces is None else available_spaces)
    unknown = avail - set(ALL_SPACES)
    if unknown:
        raise ValueError(f"unknown spaces {sorted(unknown)}; known {ALL_SPACES}")
    out = []
    for sp in FEATURE_SPACES:
        if sp not in avail:
            continue
        for w in WEIGHTINGS:
            for m in FEATURE_METRICS:
                out.append(SimilaritySetting(sp, w, m, k_max))
    if TANIMOTO_SPACE in avail:
        out.append(SimilaritySetting(TANIMOTO_SPACE, 'none', 'tanimoto', k_max))
    if FOREST_SPACE in avail:
        out.append(SimilaritySetting(FOREST_SPACE, 'none', 'forest', k_max))
    return out


def settings_from_env(available_spaces: Optional[Iterable[str]] = None,
                      k_max: int = DEFAULT_K_MAX, env=None):
    """The similarity settings a run sweeps, from KIRBY_SIMILARITY_SETTINGS.

      unset, '' or 'all'  enumerate_settings(available_spaces, k_max): the
                          full sweep, the default (owner decision 2026-09-25)
      'model_input'       [MODEL_INPUT]: today's behaviour, today's numbers
                          bit for bit
      a comma list        those settings by name, as SimilaritySetting.name
                          writes them ('base_1|lightgbm_gain|euclidean',
                          'tanimoto_ecfp4|none|tanimoto',
                          'model_input|none|process', ...), in the order
                          given, duplicates dropped. 'all' and 'model_input'
                          may appear in the list too.
    A name that is not a valid setting raises. A named setting whose space is
    not in `available_spaces` is dropped with a printed line, the same rule
    enumerate_settings applies -- never faked.
    `env` overrides os.environ for testing.
    """
    raw = (os.environ if env is None else env).get('KIRBY_SIMILARITY_SETTINGS')
    raw = '' if raw is None else str(raw).strip()
    if raw in ('', 'all'):
        return enumerate_settings(available_spaces, k_max)
    avail = set(ALL_SPACES if available_spaces is None else available_spaces)
    out, seen = [], set()
    for tok in (t.strip() for t in raw.split(',')):
        if not tok:
            continue
        if tok == 'all':
            cands = enumerate_settings(available_spaces, k_max)
        elif tok == MODEL_INPUT_SPACE:
            cands = [SimilaritySetting(MODEL_INPUT_SPACE, 'none', 'process',
                                       k_max)]
        else:
            parts = tok.split('|')
            if len(parts) != 3:
                raise ValueError(
                    f"KIRBY_SIMILARITY_SETTINGS: {tok!r} is not 'all', "
                    f"'model_input' or a setting name 'space|weighting|metric'")
            s = SimilaritySetting(parts[0], parts[1], parts[2], k_max)
            if s.space != MODEL_INPUT_SPACE and s.space not in avail:
                print(f"  KIRBY_SIMILARITY_SETTINGS: {tok} dropped, space "
                      f"{s.space!r} not available here")
                continue
            cands = [s]
        for s in cands:
            if s.name not in seen:
                seen.add(s.name)
                out.append(s)
    return out


# ---------------------------------------------------------------------------
# Augmented single-representation spaces (base_k_aug_capped / _uncapped)
# ---------------------------------------------------------------------------

def aug_space_name(base_index, strategy):
    """'base_1_aug_capped' etc. base_index is 1 or 2; strategy in
    AUG_STRATEGIES."""
    if int(base_index) not in (1, 2) or strategy not in AUG_STRATEGIES:
        raise ValueError(f"no augmented space for base {base_index!r}, "
                         f"strategy {strategy!r}")
    return f'base_{int(base_index)}_aug_{strategy}'

def base_aug_blueprint(base_name, base_width, *, metric_name, task_type,
                       lower_is_better, strategy='uncapped',
                       candidate_ranking='residual_correlation',
                       augmentation_budget=20, aug_max_candidates=50):
    """The recipe c21 builds 'B_aug' with, for one base representation.

    Copied field for field from c21_single_rep_augmentation.py:863-879 (READ,
    not imported: c21 has uncommitted edits and its module runs argument
    parsing at import). One base representation takes the whole budget, the
    budget is its full width so every base feature is kept, and the only real
    work is the admission method choosing which augmentation features to add.
    Allocation, importance and internal evaluator are inert with one
    representation (c21:853-858).

    Build the fold with noise_helpers.hybrid_for_fold, or with
    noise_cache.cached_hybrid_for_fold, passing:
        train_reps   {base_name: X_base[train_idx]}
        val_reps     {base_name: X_base[val_idx]}
        augmentations_train / _val
                     {aug_name: X_aug[idx]} for DATASET_REPS[ds]['augmentation']
                     minus the base itself (c21:429)
        model, metric_fn, lower_is_better   as c21 passes them (c21:925-929)
    then standardise the result and hand it in spaces_train/spaces_val under
    aug_space_name(k, strategy), e.g. 'base_1_aug_capped'.

    Differences from c21, stated: no decoy donors are planted (c21:909-913 --
    they exist to count what an admission rule wrongly takes, and a similarity
    space has no use for them), and hybrid_for_fold does not pass
    precomputed_alloc_scores={base: 1.0} (c21:950), which c21 measured to give
    identical matrices at about half the time.

    `strategy`: c21 ships 'capped' and 'uncapped' (c21:200-207). Both are
    similarity spaces since 2026-09-25 (owner decision), so the caller builds
    one per strategy. Every other field is the same for the two, as in c21,
    where one Blueprint call takes the strategy as its only varying argument
    (c21 at HEAD, lines 622-639). 'uncapped' stays the default argument
    because that is how c21's docstring defines B_aug (c21:10).
    """
    if strategy not in AUG_STRATEGIES:
        raise ValueError(f"strategy {strategy!r} not in {AUG_STRATEGIES}")
    from kirby.find_best_hybrid import Blueprint
    return Blueprint(
        reps=[base_name],
        allocation_method='performance_weighted',
        importance_method='lightgbm_gain',
        internal_evaluator='match',
        budget=int(base_width),
        step_size=5,
        augmentation_strategy=strategy,
        aug_per_source_budget=None,
        candidate_ranking=candidate_ranking,
        augmentation_budget=int(augmentation_budget),
        aug_max_candidates=int(aug_max_candidates),
        metric=metric_name,
        task_type=task_type,
        lower_is_better=bool(lower_is_better))


# ---------------------------------------------------------------------------
# Pairwise distance, chunked, stable top-k
# ---------------------------------------------------------------------------

def _chunk_rows(n_cols, itemsize=8):
    try:
        mb = float(os.environ.get('KIRBY_NEIGHBOUR_CHUNK_MB', '256') or 256)
    except ValueError:
        mb = 256.0
    return max(1, int(mb * 1e6 / (max(1, n_cols) * itemsize)))


def _stable_topk(D, k):
    """The k smallest entries of each row of D, ties broken by column index.

    Exact: the first j of the k returned are the j smallest for every j <= k,
    whatever k was asked for. np.argpartition finds the k-th value; every
    column at or under it is a candidate, and the candidates are ordered by
    (distance, index) with np.lexsort.
    """
    n_rows, n_cols = D.shape
    k = int(min(k, n_cols))
    idx = np.empty((n_rows, k), dtype=np.int64)
    dist = np.empty((n_rows, k), dtype=D.dtype)
    if k == 0:
        return idx, dist
    kth = np.partition(D, k - 1, axis=1)[:, k - 1]
    for i in range(n_rows):
        cand = np.flatnonzero(D[i] <= kth[i])
        order = np.lexsort((cand, D[i, cand]))[:k]
        idx[i] = cand[order]
        dist[i] = D[i, cand[order]]
    return idx, dist


def _k_eff(n_train, k_max, exclude_groups):
    """How many neighbours every training row can have. Same rule as
    noise_estimation._knn_indices (noise_estimation.py:245-251)."""
    if exclude_groups is None:
        return int(min(k_max, n_train - 1))
    g = np.asarray(exclude_groups)
    max_g = int(np.unique(g, return_counts=True)[1].max())
    k_eff = int(min(k_max, n_train - max_g))
    if k_eff < 1:
        raise ValueError(
            f"exclude_groups leaves no neighbours: the largest group holds "
            f"{max_g} of {n_train} rows.")
    return k_eff


def _lists_from_distance(dist_fn, n_query, n_train, k, *, self_rows=False,
                         exclude_groups=None, itemsize=8, sorter='stable'):
    """Neighbour lists from a row-chunked distance function.

    dist_fn(start, stop) returns the (stop-start, n_train) distances of query
    rows start..stop to every training row. With self_rows the query rows ARE
    the training rows: each row's own column is set to +inf, so it is excluded
    by index -- not by dropping the first column, which under ties need not be
    the row itself -- and with exclude_groups every column sharing its group is
    set to +inf too, the same semantics as noise_estimation._knn_indices.

    sorter='stable' breaks ties by column index (_stable_topk). sorter='numpy'
    is np.argsort's default, used ONLY for forest_proximity because that is
    what prox_smooth does today (n07:908) and bit-identity with it is the
    point; a full row sort cut to k is exact for every smaller k either way.
    """
    g = None if exclude_groups is None else np.asarray(exclude_groups)
    step = _chunk_rows(n_train, itemsize)
    idx_parts, dist_parts = [], []
    for start in range(0, n_query, step):
        stop = min(n_query, start + step)
        D = dist_fn(start, stop)
        if self_rows:
            rows = np.arange(start, stop)
            D[rows - start, rows] = np.inf
            if g is not None:
                D[g[start:stop, None] == g[None, :]] = np.inf
        if sorter == 'numpy':
            order = np.argsort(D, axis=1)[:, :k]
            idx_parts.append(order.astype(np.int64))
            dist_parts.append(np.take_along_axis(D, order, axis=1))
        else:
            i_, d_ = _stable_topk(D, k)
            idx_parts.append(i_)
            dist_parts.append(d_)
    if not idx_parts:
        return (np.empty((0, k), dtype=np.int64), np.empty((0, k)))
    return np.vstack(idx_parts), np.vstack(dist_parts)


# ---------------------------------------------------------------------------
# The three kinds of space
# ---------------------------------------------------------------------------

def _ecfp4_bits(smiles):
    """2048-bit Morgan radius-2 bits, exactly as n01._compute_ecfp4_tanimoto_sim
    builds them (n01:97-116): an unparseable SMILES gets METHANE's
    fingerprint, not a zero vector, whatever that function's comment says.

    Reimplemented rather than imported because n01 returns only the square
    train-by-train matrix and a validation fold needs train-by-val.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    arr = np.zeros((len(smiles), 2048), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol = Chem.MolFromSmiles('C')
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        for bit in fp.GetOnBits():
            arr[i, bit] = 1.0
    return arr


def _forest_leaves(X_train, y_train, is_clf, X_apply=None):
    """Leaf ids of the training rows, and of X_apply, under one forest.

    The forest is noise_estimation.forest_leaf_ids's, field for field
    (noise_estimation.py:2119-2140): the same class, FOREST_TREES,
    FOREST_MIN_SAMPLES_LEAF, FOREST_SEED, float64 input, KIRBY_RF_JOBS threads
    and int32 ids. It is rebuilt here only because that function returns the
    training rows' ids and not the fitted forest, so validation rows cannot be
    put through it. The smoke checks the training ids are identical to
    forest_leaf_ids's.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from kirby.utils.noise_estimation import (
        FOREST_TREES, FOREST_MIN_SAMPLES_LEAF, FOREST_SEED)
    cls = RandomForestClassifier if is_clf else RandomForestRegressor
    rf = cls(n_estimators=int(FOREST_TREES),
             min_samples_leaf=int(FOREST_MIN_SAMPLES_LEAF),
             random_state=int(FOREST_SEED),
             n_jobs=int(os.environ.get('KIRBY_RF_JOBS', '1') or 1))
    rf.fit(np.asarray(X_train, dtype=np.float64), y_train)
    L_tr = np.asarray(rf.apply(X_train), dtype=np.int32)
    L_ap = (None if X_apply is None
            else np.asarray(rf.apply(np.asarray(X_apply)), dtype=np.int32))
    return L_tr, L_ap


def _forest_distance(L_rows, L_cols, chunk=125):
    """One minus the share of trees in which a row molecule and a column
    molecule land in one leaf. The arithmetic of
    noise_estimation.forest_proximity_distance (noise_estimation.py:2143-2162)
    -- a float64 share assigned into a float32 matrix -- for a rectangle of
    rows by columns instead of the square, in chunks of 125 rows as there."""
    L_rows = np.asarray(L_rows)
    L_cols = np.asarray(L_cols)
    n_trees = L_cols.shape[1]
    D = np.empty((len(L_rows), len(L_cols)), dtype=np.float32)
    for s in range(0, len(L_rows), chunk):
        e = min(len(L_rows), s + chunk)
        same = (L_rows[s:e, None, :] == L_cols[None, :, :])
        D[s:e] = 1.0 - (same.sum(axis=2) / float(n_trees))
    return D


def _importance_weights(Z_tr, y_train, method):
    """Per-feature importance on the space's training rows, via the engine's
    own function (kirby.hybrid.compute_feature_importance, hybrid.py:4629).
    `y_train` is whatever the caller holds -- the noisy labels."""
    from kirby.hybrid import compute_feature_importance
    scores = compute_feature_importance({'space': Z_tr}, np.asarray(y_train),
                                        method=method)
    return np.asarray(scores['space'], dtype=np.float64)


def _unavailable(setting, reason):
    return {'setting': setting.to_dict(), 'setting_name': setting.name,
            'available': False, 'reason': reason,
            'idx_train': None, 'dist_train': None,
            'idx_val': None, 'dist_val': None, 'k_eff': 0}


def neighbour_lists(setting: SimilaritySetting, *, spaces_train: Dict,
                    spaces_val: Optional[Dict], y_train, groups_train=None,
                    exclude_groups=None, smiles_train=None, smiles_val=None,
                    is_clf=False):
    """Each molecule's k_max nearest training molecules under one setting.

    Args:
        setting: a SimilaritySetting, or MODEL_INPUT.
        spaces_train / spaces_val: {space name: matrix} of this fold's training
            and validation rows, ALREADY STANDARDISED (fold_weights requires
            it; preprocessing.py:396-401). Keys are the space names in
            ALL_SPACES plus 'model_input' -- the matrix the model is fitted on,
            read by MODEL_INPUT and by forest_proximity. spaces_val may be None
            when no validation lists are wanted.
        y_train: the training labels the caller holds (noisy). Read by the
            importance weighting and by the forest.
        groups_train: accepted so every caller passes the same arguments to
            every setting; no computation here reads it today. The inner
            importance fits take no groups.
        exclude_groups: one label per training row; a row never takes a
            neighbour sharing it. Same semantics as
            noise_estimation._knn_indices. None skips only the row itself.
        smiles_train / smiles_val: for tanimoto_ecfp4 only.
        is_clf: classification forest for forest_proximity.

    Returns a dict:
        idx_train  (n_train, k_eff) int, neighbours of each training row among
                   the OTHER training rows, nearest first
        dist_train same shape, the distances
        idx_val    (n_val, k_val) int, neighbours of each validation row among
                   the training rows, or None
        dist_val   same shape, or None
        k_eff      the width of idx_train; below k_max only when the training
                   count or exclude_groups leaves fewer
        available  False when the space cannot be built; then every array is
                   None and `reason` says why. Never faked.
        exact_by_k MODEL_INPUT only: {k: (idx, dist)}, the exact query today's
                   code issues for k neighbours. Read through neighbours_at.
        weights    the per-feature multiplier, when weighted
        setting, setting_name
    """
    if not isinstance(setting, SimilaritySetting):
        raise TypeError(f"setting must be a SimilaritySetting, got {setting!r}")
    y_train = np.asarray(y_train)
    n_tr = len(y_train)
    k_max = int(setting.k_max)
    want_val = spaces_val is not None
    sp = setting.space

    # ---- today's behaviour ------------------------------------------------
    if sp == MODEL_INPUT_SPACE:
        X = spaces_train.get('model_input') if spaces_train else None
        if X is None:
            return _unavailable(setting, "spaces_train has no 'model_input' "
                                         "matrix")
        return _model_input_lists(setting, np.asarray(X), y_train,
                                  None if not want_val
                                  else spaces_val.get('model_input'),
                                  exclude_groups)

    k_eff = _k_eff(n_tr, k_max, exclude_groups)
    out = {'setting': setting.to_dict(), 'setting_name': setting.name,
           'available': True, 'reason': None, 'k_eff': k_eff,
           'idx_val': None, 'dist_val': None}

    # ---- ECFP4 Tanimoto ---------------------------------------------------
    if sp == TANIMOTO_SPACE:
        if smiles_train is None or (want_val and smiles_val is None):
            return _unavailable(setting, 'no SMILES were passed for this '
                                         'dataset')
        try:
            B_tr = _ecfp4_bits(list(smiles_train))
            B_va = _ecfp4_bits(list(smiles_val)) if want_val else None
        except ImportError as exc:
            return _unavailable(setting, f'RDKit is not importable: {exc}')
        # Tanimoto = 1 - Jaccard distance on 0/1 vectors (n01:118-120), so the
        # distance IS scipy's jaccard.
        def _d_tr(a, b):
            return cdist(B_tr[a:b], B_tr, metric='jaccard')
        out['idx_train'], out['dist_train'] = _lists_from_distance(
            _d_tr, n_tr, n_tr, k_eff, self_rows=True,
            exclude_groups=exclude_groups)
        if want_val:
            def _d_va(a, b):
                return cdist(B_va[a:b], B_tr, metric='jaccard')
            out['idx_val'], out['dist_val'] = _lists_from_distance(
                _d_va, len(B_va), n_tr, min(k_max, n_tr))
        return out

    # ---- forest proximity -------------------------------------------------
    if sp == FOREST_SPACE:
        X = spaces_train.get('model_input') if spaces_train else None
        if X is None:
            return _unavailable(setting, "forest_proximity is fitted on the "
                                         "model-input matrix and spaces_train "
                                         "has no 'model_input'")
        X_va = spaces_val.get('model_input') if want_val else None
        if want_val and X_va is None:
            return _unavailable(setting, "spaces_val has no 'model_input'")
        # The same forest prox_smooth fits (n07:904): the training rows and the
        # labels as the caller holds them. _forest_leaves also applies it to
        # the validation rows, which forest_leaf_ids cannot.
        L_tr, L_va = _forest_leaves(X, y_train, is_clf, X_va)

        def _f_tr(a, b):
            return _forest_distance(L_tr[a:b], L_tr)
        out['idx_train'], out['dist_train'] = _lists_from_distance(
            _f_tr, n_tr, n_tr, k_eff, self_rows=True,
            exclude_groups=exclude_groups, itemsize=4, sorter='numpy')
        if want_val:
            def _f_va(a, b):
                return _forest_distance(L_va[a:b], L_tr)
            out['idx_val'], out['dist_val'] = _lists_from_distance(
                _f_va, len(L_va), n_tr, min(k_max, n_tr), itemsize=4,
                sorter='numpy')
        return out

    # ---- a feature space --------------------------------------------------
    Z_tr = spaces_train.get(sp) if spaces_train else None
    if Z_tr is None:
        return _unavailable(setting, f'spaces_train has no {sp!r} matrix')
    Z_va = spaces_val.get(sp) if want_val else None
    if want_val and Z_va is None:
        return _unavailable(setting, f'spaces_val has no {sp!r} matrix')
    Z_tr = np.asarray(Z_tr, dtype=np.float64)
    Z_va = None if Z_va is None else np.asarray(Z_va, dtype=np.float64)
    if setting.weighting != 'none':
        from kirby.similarity.preprocessing import fold_weights
        w = _importance_weights(Z_tr, y_train, setting.weighting)
        # Applied AFTER standardising, as fold_weights requires
        # (preprocessing.py:28-33): the caller's matrices are z-scored. The
        # exponent is the one the distance's family implies (1 under
        # manhattan, 0.5 under euclidean), unless KIRBY_IMPORTANCE_EXPONENT
        # overrides it -- the cache key records that override.
        Z_tr, _ = fold_weights(Z_tr, w, setting.metric, convention='mean1')
        if Z_va is not None:
            Z_va, _ = fold_weights(Z_va, w, setting.metric, convention='mean1')
        out['weights'] = w
    metric = _CDIST[setting.metric]

    def _z_tr(a, b):
        return cdist(Z_tr[a:b], Z_tr, metric=metric)
    out['idx_train'], out['dist_train'] = _lists_from_distance(
        _z_tr, n_tr, n_tr, k_eff, self_rows=True, exclude_groups=exclude_groups)
    if want_val:
        def _z_va(a, b):
            return cdist(Z_va[a:b], Z_tr, metric=metric)
        out['idx_val'], out['dist_val'] = _lists_from_distance(
            _z_va, len(Z_va), n_tr, min(k_max, n_tr))
    return out


def model_input_ks(n_train, k_max=DEFAULT_K_MAX):
    """Every k today's code reads, for a training block of n_train rows.

    n07's label_smooth / knn_smooth: min(10, n-1) queried, the first dropped,
    so min(10, n-1) - 1 neighbours (n07:829-842). n07's prox_smooth:
    min(10, n-1) (n07:907). The four neighbour estimators: min(10, n-1)
    (noise_estimation.py:291). Plus the swept ks, SWEPT_KS, and k_max.
    """
    base = min(10, n_train - 1)
    ks = {base - 1, base, k_max, *SWEPT_KS}
    return sorted(k for k in ks if 1 <= k <= n_train - 1)


def _model_input_lists(setting, X, y_train, X_va, exclude_groups):
    """MODEL_INPUT: the exact sklearn query today's code issues, per k.

    Without exclude_groups each list is make_nearest_neighbors(n_neighbors=k+1)
    fitted on the training rows, queried on them, first column dropped -- the
    call label_smooth and knn_smooth make at n07:839-842, and the call
    _knn_indices makes at noise_estimation.py:236-241. With exclude_groups it
    is _knn_indices itself.
    """
    from experiments._shared import make_nearest_neighbors
    from kirby.utils.noise_estimation import _knn_indices
    n = len(y_train)
    exact = {}
    for k in model_input_ks(n, setting.k_max):
        if exclude_groups is None:
            nn = make_nearest_neighbors(n_neighbors=int(min(k + 1, n)))
            nn.fit(X)
            d, i = nn.kneighbors(X)
            exact[k] = (i[:, 1:k + 1], d[:, 1:k + 1])
        else:
            d, i = _knn_indices(X, k, None, exclude_groups=exclude_groups,
                                return_distance=True)
            exact[k] = (i, d)
    k_top = max(exact)
    out = {'setting': setting.to_dict(), 'setting_name': setting.name,
           'available': True, 'reason': None,
           'idx_train': exact[k_top][0], 'dist_train': exact[k_top][1],
           'k_eff': int(exact[k_top][0].shape[1]),
           'exact_by_k': exact, 'idx_val': None, 'dist_val': None}
    if X_va is not None:
        nn = make_nearest_neighbors(n_neighbors=int(min(setting.k_max, n)))
        nn.fit(X)
        d, i = nn.kneighbors(np.asarray(X_va))
        out['idx_val'], out['dist_val'] = i, d
    return out


def neighbours_at(nbrs, k, which='train'):
    """(idx, dist) for k neighbours out of a neighbour_lists result.

    For every setting but MODEL_INPUT this is the first k columns, which is
    exact because those lists are stably sorted. For MODEL_INPUT it is the
    stored query for exactly k (see the module docstring for why a cut is not
    the same list); a k it does not hold raises rather than cutting.
    """
    if nbrs is None or not nbrs.get('available', False):
        raise ValueError(
            f"no neighbour lists: {None if nbrs is None else nbrs.get('reason')}")
    k = int(k)
    if which == 'val':
        i, d = nbrs['idx_val'], nbrs['dist_val']
        if i is None:
            raise ValueError('these lists were built without validation rows')
        if i.shape[1] < k:
            raise ValueError(f'asked for {k} validation neighbours, lists hold '
                             f'{i.shape[1]}')
        return i[:, :k], d[:, :k]
    exact = nbrs.get('exact_by_k')
    if exact is not None:
        if k not in exact:
            raise KeyError(
                f"MODEL_INPUT holds exact lists for k in {sorted(exact)}, not "
                f"{k}. A cut of a longer list is not today's list under ties; "
                f"add {k} to model_input_ks.")
        return exact[k]
    i, d = nbrs['idx_train'], nbrs['dist_train']
    if i.shape[1] < k:
        raise ValueError(f'asked for {k} neighbours, lists hold {i.shape[1]} '
                         f'(k_eff); the training count or exclude_groups '
                         f'leaves no more')
    return i[:, :k], d[:, :k]
