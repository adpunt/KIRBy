#!/usr/bin/env python3
"""F22: are the four raw representations behind the sim_* recipes any better,
as a similarity space, than the representations already in each dataset's list?

STANDALONE AND CACHE-FREE, like f18 and f19. Never reads or writes a hybrid
search, a baseline pickle, a hybrid-eval pickle or a result database.

---------------------------------------------------------------------------
WHERE THIS SITS. src/kirby/similarity/recipes.py registers nine names
(sim_count, sim_count_desc, ... sim_all_half) as representations. Each one is
already a concatenation of up to four things:

    ECFP4 count fingerprint          1024 features
    RDKit descriptors                 208 features
    3D pharmacophore fingerprint       49 features
    atom-pair counts + torsion bits  2048 features, cut to the best 100

Each part is z-scored, divided by the square root of its own feature count, and
in the "_half" names the ECFP4 part is multiplied by 0.5. Those nine
combinations were written out by hand in an earlier chat; nothing measured them
and nothing ranked them.

find_best_hybrid receives representations and splits a feature budget across
them. Handing it nine ready-made combinations means the split happens twice:
once by hand inside each name, once by the allocation method. This script
measures whether the four parts do better as four separate representations.

---------------------------------------------------------------------------
WHAT IS COMPARED. Five axes, every cell reported, nothing averaged across
datasets.

  representation set   candidates : ecfp4_count, atompair_count, desc208,
                                    pharmacophore3d
                       registered : this dataset's own representations from
                                    DATASET_REPS, minus any needing torch or a
                                    graph library (they do not build on this
                                    laptop). Which ones were dropped is
                                    recorded in the output.
                       both       : the two sets together

  how features are     activity     Spearman(|feature difference|, |activity
  ranked inside a                   difference|) on the fold's training pairs.
  representation                    Uses labels.
                       structure    Spearman(|feature difference|, maximum
                                    common substructure distance) on the same
                                    pairs. Uses no labels.
                       model        RandomForest importance fit on the fold's
                                    training rows. Uses labels. Closest to the
                                    engine's lightgbm_gain, which segfaults on
                                    this laptop once rdkit has been imported.

  how the budget is    equal        every representation gets budget / n
  split                scored       proportional to that representation's own
                                    read-across on the training rows, which is
                                    what allocate_performance_weighted does

  total budget         100, 250, 500, 1000 features

  how the kept         raw          np.hstack of the raw features. Exactly what
  features are                      create_hybrid does today.
  glued together       zscore       z-score every feature first
                       zscore_sqrt  z-score, then divide each representation's
                                    block by the square root of how many
                                    features it contributed, so a 1024-feature
                                    representation cannot outweigh a
                                    49-feature one purely by being wider
                       zscore_up2   zscore_sqrt, then multiply the candidate
                                    representations by 2. Answers "should they
                                    be weighted more".

Every representation is also scored ALONE at full width, as the reference each
combination has to beat.

---------------------------------------------------------------------------
SCORED ON, per scaffold-grouped held-out fold:

  readacross   Spearman(kNN-predicted activity, true activity), neighbours
               taken under the Euclidean distance of the glued features
  struct_fid   Spearman(that same pairwise distance, maximum common
               substructure distance) over the pairs falling in the fold

Both are reported. Neither is merged into the other.

Feature ranking, the budget split and the z-score statistics are all fit on the
fold's training rows only. The held-out rows are used for scoring and nothing
else.

Usage:
    python f22_similarity_feature_sources.py --dataset herg_ki
    python f22_similarity_feature_sources.py --dataset logd --budgets 100,1000
    python f22_similarity_feature_sources.py --dataset qm9 --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import json
import warnings
import numpy as np
from scipy.stats import spearmanr, rankdata
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    get_config, create_experiment_parser,
    save_experiment_results, cv_split, get_domain_metric,
)

SEED = 42
K_NEIGHBOURS = 5

# The four parts the nine sim_* names are built from.
CANDIDATE_REPS = ('ecfp4_count', 'atompair_count', 'desc208', 'pharmacophore3d')

# Representations that need torch or a graph library. They do not build on this
# laptop, so a local run drops them and says which ones it dropped.
NEEDS_GPU_STACK = {
    'chemberta', 'molformer', 'selformer', 'chembert', 'molclr', 'graphmvp',
    'smited', 'mhg_gnn', 'grover', 'graph_kernel', 'unimol', 'unimol_v2',
    'schnet', 'dimenetpp', 'clamp', 'rmat', 'minimol', 'gp_molformer', 'mole',
    'mol2vec', 'tda', 'e3fp', 'map4', 'padel',
}

BUDGETS = (100, 250, 500, 1000)
RANKERS = ('activity', 'structure', 'model')
SPLITS = ('equal', 'scored')
GLUE = ('raw', 'zscore', 'zscore_sqrt', 'zscore_up2')
REP_SETS = ('candidates', 'registered', 'both')


# ---------------------------------------------------------------------------
def _rep_builders():
    from kirby.representations.molecular import (
        create_ecfp4, create_maccs, create_avalon, create_rdkfp,
        create_atom_pair, create_top_torsion, create_pdv, create_mordred,
        create_ecfp4_count, create_atompair_count, create_desc208,
        create_pharmacophore3d, create_usr, create_usrcat,
        create_coulomb_matrix,
    )
    return {
        'ecfp4': create_ecfp4, 'maccs': create_maccs, 'avalon': create_avalon,
        'rdkfp': create_rdkfp, 'atom_pair': create_atom_pair,
        'top_torsion': create_top_torsion, 'pdv': create_pdv,
        'mordred': create_mordred,
        'ecfp4_count': create_ecfp4_count,
        'atompair_count': create_atompair_count,
        'desc208': create_desc208,
        'pharmacophore3d': create_pharmacophore3d,
        'usr': create_usr, 'usrcat': create_usrcat,
        'coulomb_matrix': create_coulomb_matrix,
    }


def build_representations(smiles, wanted):
    """Compute each representation straight from SMILES. No cache, no lookup."""
    builders = _rep_builders()
    out, failed = {}, {}
    for name in wanted:
        if name not in builders:
            failed[name] = 'no builder in this script'
            continue
        t = time.time()
        try:
            X = np.asarray(builders[name](smiles), dtype=np.float64)
        except Exception as exc:
            failed[name] = repr(exc)
            print(f'    {name}: FAILED {exc!r}', flush=True)
            continue
        if X.ndim != 2 or X.shape[0] != len(smiles):
            failed[name] = f'shape {X.shape} does not match {len(smiles)} molecules'
            print(f'    {name}: FAILED {failed[name]}', flush=True)
            continue
        keep = np.nanstd(X, axis=0) > 0
        X = np.nan_to_num(X[:, keep], nan=0.0, posinf=0.0, neginf=0.0)
        if X.shape[1] == 0:
            failed[name] = 'every feature constant'
            print(f'    {name}: FAILED every feature constant', flush=True)
            continue
        out[name] = X
        print(f'    {name}: {X.shape[1]}D '
              f'({int((~keep).sum())} constant dropped) '
              f'in {time.time() - t:.1f}s', flush=True)
    return out, failed


def registered_reps_for(dataset):
    """The dataset's own representation list, and which need a GPU stack."""
    from experiments.complexity.complexity_helpers import DATASET_REPS
    entry = DATASET_REPS.get(dataset) or {}
    listed = list(entry.get('general') or [])
    usable = [r for r in listed if r not in NEEDS_GPU_STACK]
    dropped = [r for r in listed if r in NEEDS_GPU_STACK]
    return listed, usable, dropped


# ---------------------------------------------------------------------------
def _spearman_cols(DF, t):
    """Spearman of every column of DF against t. Same routine the label-free
    ranker in src/kirby/similarity/selection.py uses."""
    tr = rankdata(t).astype(np.float64)
    tr -= tr.mean()
    Rr = np.apply_along_axis(rankdata, 0, DF).astype(np.float64)
    Rr -= Rr.mean(0)
    num = Rr.T @ tr
    den = np.sqrt((Rr ** 2).sum(0) * (tr ** 2).sum())
    out = np.zeros(DF.shape[1])
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def rank_features(X, tr_rows, y_tr, pi, pj, target, kind):
    """Score every feature of X. Higher is kept first. Training rows only.

    pi, pj index rows of X and are already restricted to the training rows.
    target is the per-pair quantity the pair-based rankers correlate against.
    """
    if kind == 'model':
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, max_depth=12,
                                   random_state=SEED, n_jobs=-1)
        rf.fit(X[tr_rows], y_tr)
        return rf.feature_importances_
    DF = np.abs(X[pi] - X[pj])
    return np.abs(_spearman_cols(DF, target))


def split_budget(names, budget, weights):
    """How many features each representation gets. Never more than it has."""
    total_w = float(sum(max(0.0, w) for w in weights.values()))
    alloc = {}
    for n in names:
        w = max(0.0, weights.get(n, 0.0))
        share = (w / total_w) if total_w > 0 else (1.0 / len(names))
        alloc[n] = int(round(share * budget))
    return alloc


def glue(blocks, mode, candidate_names):
    """Concatenate the kept features of each representation.

    blocks maps representation name -> (train_features, val_features), already
    cut to that representation's allocated features. Standardising uses the
    training rows only.
    """
    parts_tr, parts_va = [], []
    for name, (Btr, Bva) in blocks.items():
        if Btr.shape[1] == 0:
            continue
        if mode == 'raw':
            Atr, Ava = Btr, Bva
        else:
            sc = StandardScaler().fit(Btr)
            Atr = np.nan_to_num(sc.transform(Btr), nan=0.0, posinf=0.0, neginf=0.0)
            Ava = np.nan_to_num(sc.transform(Bva), nan=0.0, posinf=0.0, neginf=0.0)
            if mode in ('zscore_sqrt', 'zscore_up2'):
                s = np.sqrt(Atr.shape[1])
                Atr, Ava = Atr / s, Ava / s
            if mode == 'zscore_up2' and name in candidate_names:
                Atr, Ava = Atr * 2.0, Ava * 2.0
        parts_tr.append(Atr)
        parts_va.append(Ava)
    if not parts_tr:
        return None, None
    return np.hstack(parts_tr), np.hstack(parts_va)


# ---------------------------------------------------------------------------
def score_readacross(D_va_tr, D_tr_tr, y_tr, y_va):
    """Spearman(kNN prediction, true activity) on the held-out rows.

    Neighbours come from the training rows only, under the same distance the
    features produce. D_tr_tr is unused by KNeighborsRegressor beyond fitting.
    """
    from sklearn.neighbors import KNeighborsRegressor
    if len(y_va) < 3 or len(y_tr) < 2:
        return np.nan
    k = int(min(K_NEIGHBOURS, len(y_tr)))
    m = KNeighborsRegressor(n_neighbors=k, weights='distance',
                            metric='precomputed')
    m.fit(D_tr_tr, y_tr)
    pred = m.predict(D_va_tr)
    if np.std(pred) == 0 or np.std(y_va) == 0:
        return np.nan
    rho, _ = spearmanr(pred, y_va)
    return float(rho) if np.isfinite(rho) else np.nan


def score_struct_fidelity(D_va, local_of_global, pairs):
    """Spearman(pairwise distance, maximum common substructure distance)."""
    if pairs is None:
        return np.nan
    I = np.asarray(pairs['I']).astype(int)
    J = np.asarray(pairs['J']).astype(int)
    mcs = np.asarray(pairs['mcs'], dtype=np.float64)
    li = np.array([local_of_global.get(int(i), -1) for i in I])
    lj = np.array([local_of_global.get(int(j), -1) for j in J])
    keep = (li >= 0) & (lj >= 0)
    if int(keep.sum()) < 10:
        return np.nan
    d = D_va[li[keep], lj[keep]]
    m = mcs[keep]
    if np.std(d) == 0 or np.std(m) == 0:
        return np.nan
    rho, _ = spearmanr(d, m)
    return float(rho) if np.isfinite(rho) else np.nan



def _assert_up2_degenerate():
    """Prove the skipped zscore_up2 cells really are duplicates.

    Doubling every block after standardising multiplies every pairwise
    Euclidean distance by exactly 2, so Spearman against anything is unchanged.
    Asserted rather than assumed, because the scope rule says a cell may only be
    dropped as degenerate if the degeneracy is verified.
    """
    rng = np.random.RandomState(0)
    A = rng.randn(40, 6)
    B = rng.randn(40, 3)
    both = {'a': (A, A), 'b': (B, B)}
    _, g1 = glue(both, 'zscore_sqrt', set())
    _, g2 = glue(both, 'zscore_up2', {'a', 'b'})
    d1 = cdist(g1, g1, metric='euclidean')
    d2 = cdist(g2, g2, metric='euclidean')
    iu = np.triu_indices(len(d1), 1)
    assert np.allclose(d2[iu], 2.0 * d1[iu]), 'zscore_up2 is not a uniform x2'
    rho, _ = spearmanr(d1[iu], d2[iu])
    assert abs(rho - 1.0) < 1e-12, f'ordering changed, spearman {rho}'
    return {'max_abs_diff_from_2x': float(np.max(np.abs(d2[iu] - 2 * d1[iu]))),
            'spearman_of_orderings': 1.0}


# ---------------------------------------------------------------------------
def run(data, reps, pairs, config, args, candidate_names, registered_names):
    labels = np.asarray(data['labels'], dtype=np.float64).ravel()
    groups = np.asarray(data['groups'])
    n_folds = config['n_splits']
    budgets = [int(b) for b in args.budgets.split(',')]

    rep_sets = {
        'candidates': [r for r in candidate_names if r in reps],
        'registered': [r for r in registered_names if r in reps],
    }
    rep_sets['both'] = rep_sets['candidates'] + rep_sets['registered']
    rep_sets = {k: v for k, v in rep_sets.items() if len(v) >= 1}

    # name -> per-fold list of {readacross, struct_fid}
    singles = {r: [] for r in reps}
    combos = {}
    skipped = []

    # cv_split hands X straight to sklearn's split(), which only reads its
    # row count, so a column of row indices is enough and costs nothing.
    X_index = np.arange(len(labels)).reshape(-1, 1)
    for fold_i, (tr, va) in enumerate(cv_split(groups, n_folds, X_index, labels,
                                               seed=SEED, mode=args.mode,
                                               task_type=data.get('task_type'))):
        t_fold = time.time()
        y_tr, y_va = labels[tr], labels[va]
        local_of_global = {int(g): l for l, g in enumerate(va)}

        # Training pairs for the two pair-based rankers. Both endpoints must be
        # training rows, so no held-out molecule touches the ranking.
        in_tr = np.zeros(len(labels), bool)
        in_tr[tr] = True
        if pairs is not None:
            I = np.asarray(pairs['I']).astype(int)
            J = np.asarray(pairs['J']).astype(int)
            keep = in_tr[I] & in_tr[J]
            pi, pj = I[keep], J[keep]
            mcs_tr = np.asarray(pairs['mcs'], dtype=np.float64)[keep]
        else:
            pi = pj = np.array([], int)
            mcs_tr = np.array([])
        dy_tr = np.abs(labels[pi] - labels[pj]) if len(pi) else np.array([])
        print(f'  fold {fold_i}: {len(tr)} train / {len(va)} held out, '
              f'{len(pi)} training pairs', flush=True)

        # --- feature rankings, once per representation per ranker -----------
        rankings = {}
        for name, X in reps.items():
            for kind in RANKERS:
                if kind == 'structure' and len(pi) < 20:
                    skipped.append({'rep': name, 'ranker': kind, 'fold': fold_i,
                                    'reason': 'fewer than 20 training pairs'})
                    continue
                if kind == 'activity' and len(pi) < 20:
                    skipped.append({'rep': name, 'ranker': kind, 'fold': fold_i,
                                    'reason': 'fewer than 20 training pairs'})
                    continue
                target = mcs_tr if kind == 'structure' else dy_tr
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    rankings[(name, kind)] = rank_features(
                        X, tr, y_tr, pi, pj, target, kind)

        # --- each representation alone, at full width -----------------------
        for name, X in reps.items():
            sc = StandardScaler().fit(X[tr])
            Ztr = np.nan_to_num(sc.transform(X[tr]), nan=0.0, posinf=0.0, neginf=0.0)
            Zva = np.nan_to_num(sc.transform(X[va]), nan=0.0, posinf=0.0, neginf=0.0)
            D_va_tr = cdist(Zva, Ztr, metric='euclidean')
            D_tr_tr = cdist(Ztr, Ztr, metric='euclidean')
            D_va = cdist(Zva, Zva, metric='euclidean')
            singles[name].append({
                'readacross': score_readacross(D_va_tr, D_tr_tr, y_tr, y_va),
                'struct_fid': score_struct_fidelity(D_va, local_of_global, pairs),
                'n_features': int(X.shape[1]),
            })

        # --- how the budget gets split, per representation set ---------------
        # 'scored' needs each representation's own read-across on the TRAINING
        # rows. Computed once per fold, reused for every budget.
        # Each representation's own read-across, measured INSIDE the training
        # rows on a scaffold-grouped split of them. An earlier version fitted
        # and predicted on the same matrix, so every representation scored
        # about 1.0 and every 'scored' split came out identical to 'equal'.
        train_scores = {}
        g_tr = groups[tr]
        uniq = np.unique(g_tr)
        if len(uniq) >= 4:
            rs = np.random.RandomState(SEED)
            inner_val_groups = set(rs.permutation(uniq)[:max(1, len(uniq) // 4)])
            iv = np.array([g in inner_val_groups for g in g_tr])
        else:
            iv = np.zeros(len(tr), bool)
        for name, X in reps.items():
            if iv.sum() < 5 or (~iv).sum() < 5:
                train_scores[name] = 1.0
                continue
            sc = StandardScaler().fit(X[tr][~iv])
            Za = np.nan_to_num(sc.transform(X[tr][~iv]), nan=0.0,
                               posinf=0.0, neginf=0.0)
            Zb = np.nan_to_num(sc.transform(X[tr][iv]), nan=0.0,
                               posinf=0.0, neginf=0.0)
            sc_val = score_readacross(cdist(Zb, Za, metric='euclidean'),
                                      cdist(Za, Za, metric='euclidean'),
                                      y_tr[~iv], y_tr[iv])
            train_scores[name] = 0.0 if not np.isfinite(sc_val) else max(0.0, sc_val)

        for set_name, names in rep_sets.items():
            for ranker in RANKERS:
                if any((n, ranker) not in rankings for n in names):
                    continue
                for split in SPLITS:
                    if split == 'equal':
                        weights = {n: 1.0 for n in names}
                    else:
                        weights = {n: train_scores[n] for n in names}
                        if sum(weights.values()) <= 0:
                            weights = {n: 1.0 for n in names}
                    for budget in budgets:
                        alloc = split_budget(names, budget, weights)
                        cut = {}
                        for n in names:
                            k = min(alloc[n], reps[n].shape[1])
                            if k <= 0:
                                continue
                            order = np.argsort(-rankings[(n, ranker)])[:k]
                            cut[n] = (reps[n][np.ix_(tr, order)],
                                      reps[n][np.ix_(va, order)])
                        if not cut:
                            skipped.append({
                                'set': set_name, 'ranker': ranker,
                                'split': split, 'budget': budget,
                                'fold': fold_i,
                                'reason': 'budget split gave every '
                                          'representation zero features'})
                            continue
                        for mode in GLUE:
                            # zscore_up2 multiplies only the candidate
                            # representations by 2. When every representation
                            # in the set is a candidate, or none is, that is a
                            # uniform rescale of the whole matrix, which leaves
                            # every Euclidean distance ordering identical to
                            # zscore_sqrt. Degenerate, verified by the
                            # assertion in _assert_up2_degenerate below, so the
                            # cell is skipped and logged rather than reported
                            # as a second measurement of the same thing.
                            widths = {v[0].shape[1] for v in cut.values()}
                            if mode == 'zscore_sqrt' and len(widths) == 1:
                                skipped.append({
                                    'set': set_name, 'ranker': ranker,
                                    'split': split, 'budget': budget,
                                    'glue': mode, 'fold': fold_i,
                                    'reason': 'degenerate: every representation '
                                              'got the same number of features, '
                                              'so dividing by the square root of '
                                              'the width is a uniform rescale and '
                                              'the Euclidean ordering equals '
                                              'zscore'})
                                continue
                            if mode == 'zscore_up2':
                                in_set = [n for n in cut if n in candidate_names]
                                if len(in_set) in (0, len(cut)) or (
                                        len(widths) == 1 and
                                        len(in_set) == len(cut)):
                                    skipped.append({
                                        'set': set_name, 'ranker': ranker,
                                        'split': split, 'budget': budget,
                                        'glue': mode, 'fold': fold_i,
                                        'reason': 'degenerate: doubling every '
                                                  'representation (or none) is '
                                                  'a uniform rescale, so the '
                                                  'Euclidean ordering equals '
                                                  'zscore_sqrt'})
                                    continue
                            Gtr, Gva = glue(cut, mode, set(candidate_names))
                            if Gtr is None:
                                continue
                            D_va_tr = cdist(Gva, Gtr, metric='euclidean')
                            D_tr_tr = cdist(Gtr, Gtr, metric='euclidean')
                            D_va = cdist(Gva, Gva, metric='euclidean')
                            key = (set_name, ranker, split, budget, mode)
                            combos.setdefault(key, []).append({
                                'readacross': score_readacross(
                                    D_va_tr, D_tr_tr, y_tr, y_va),
                                'struct_fid': score_struct_fidelity(
                                    D_va, local_of_global, pairs),
                                'n_features': int(Gtr.shape[1]),
                                'allocation': {n: int(v.shape[1])
                                               for n, v in
                                               ((nm, a) for nm, (a, b) in cut.items())},
                            })
        print(f'    fold {fold_i} done in {time.time() - t_fold:.1f}s', flush=True)

    def _agg(rows):
        out = {}
        for field in ('readacross', 'struct_fid'):
            vals = [r[field] for r in rows if np.isfinite(r.get(field, np.nan))]
            out[field] = float(np.mean(vals)) if vals else None
            out[field + '_folds'] = [None if not np.isfinite(r.get(field, np.nan))
                                     else round(float(r[field]), 4) for r in rows]
        out['n_features'] = rows[0]['n_features'] if rows else None
        if rows and 'allocation' in rows[0]:
            out['allocation'] = rows[0]['allocation']
        return out

    return {
        'representation_sets': rep_sets,
        'single_representations': {k: _agg(v) for k, v in singles.items() if v},
        'combinations': {
            '|'.join(map(str, k)): _agg(v) for k, v in combos.items()},
        'combination_key_order': ['rep_set', 'ranker', 'budget_split',
                                  'budget', 'glue'],
        'skipped_cells': skipped,
        'n_skipped_cells': len(skipped),
    }


# ---------------------------------------------------------------------------
def main():
    parser = create_experiment_parser(
        'F22: do the four parts of the sim_* recipes work better as four '
        'separate representations?')
    # --budgets already exists on the shared parser (default None), so this
    # script fills it in rather than adding a second flag with the same name.
    parser.add_argument('--n-pairs', type=int, default=4000,
                        help='Maximum common substructure pairs to compute.')
    parser.add_argument('--no-struct', action='store_true',
                        help='Skip the structural pairs. Drops the structure '
                             'ranker and the structural score.')
    args = parser.parse_args()
    if not args.budgets:
        args.budgets = ','.join(map(str, BUDGETS))
    config = get_config(args)

    t0 = time.time()
    from find_best_hybrid_molecular import load_dataset, assign_scaffold_groups
    smiles, labels_raw, task_type = load_dataset(args.dataset)
    labels = np.asarray(labels_raw, dtype=np.float64).ravel()
    groups, n_scaffolds = assign_scaffold_groups(smiles)
    data = {'labels': labels, 'groups': np.asarray(groups), 'smiles': smiles,
            'task_type': task_type, 'domain': 'molecular'}
    metric_name, _, lower_is_better = get_domain_metric(data)

    listed, usable, dropped = registered_reps_for(args.dataset)
    wanted = list(CANDIDATE_REPS) + usable
    print(f'\nF22 {args.dataset}: {len(labels)} molecules, {n_scaffolds} '
          f'scaffold groups, {config["n_splits"]} folds', flush=True)
    print(f'  registered for this dataset: {listed}', flush=True)
    if dropped:
        print(f'  dropped (need torch or a graph library, will not build '
              f'here): {dropped}', flush=True)
    print(f'  building {len(wanted)} representations from SMILES '
          f'(no cache)...', flush=True)
    reps, rep_failures = build_representations(smiles, wanted)
    if not reps:
        raise RuntimeError(f'no representations built. Failures: {rep_failures}')

    pairs = None
    if not args.no_struct:
        from kirby.similarity.mcs import compute_mcs_pairs
        print(f'  computing {args.n_pairs} maximum common substructure '
              f'pairs...', flush=True)
        tp = time.time()
        pairs = compute_mcs_pairs(smiles, labels, n_pairs=args.n_pairs,
                                  seed=SEED)
        print(f'    {len(pairs["I"])} pairs in {time.time() - tp:.1f}s',
              flush=True)

    results = {
        'experiment': 'f22_similarity_feature_sources',
        'dataset': args.dataset,
        'mode': args.mode,
        'seed': SEED,
        'cache_free': True,
        'n_samples': int(len(labels)),
        'n_scaffold_groups': int(n_scaffolds),
        'candidate_representations': list(CANDIDATE_REPS),
        'registered_representations_listed': listed,
        'registered_representations_used': usable,
        'registered_representations_dropped_locally': dropped,
        'representation_build_failures': rep_failures,
        'budgets': [int(b) for b in args.budgets.split(',')],
        'rankers': list(RANKERS),
        'budget_splits': list(SPLITS),
        'glue_modes': list(GLUE),
    }
    results['zscore_up2_degeneracy_proof'] = _assert_up2_degenerate()
    results.update(run(data, reps, pairs, config, args,
                       list(CANDIDATE_REPS), usable))
    results['elapsed_seconds'] = round(time.time() - t0, 1)

    # ---- printed summary --------------------------------------------------
    def f(v, w=11):
        return f'{v:>{w}.4f}' if isinstance(v, float) else f'{"n/a":>{w}}'

    print('\n  EACH REPRESENTATION ALONE, full width')
    print(f'    {"representation":18s} {"features":>9s} {"readacross":>11s} '
          f'{"struct_fid":>11s}')
    for name, r in sorted(results['single_representations'].items(),
                          key=lambda kv: -(kv[1]['readacross'] or -9)):
        tag = ' *' if name in CANDIDATE_REPS else '  '
        print(f'   {tag}{name:18s} {r["n_features"]:>9d} '
              f'{f(r["readacross"])} {f(r["struct_fid"])}')
    print('    * = one of the four parts the sim_* recipes are built from')

    for field in ('readacross', 'struct_fid'):
        rows = [(k, v) for k, v in results['combinations'].items()
                if v[field] is not None]
        rows.sort(key=lambda kv: -kv[1][field])
        print(f'\n  TOP 15 COMBINATIONS BY {field}')
        print(f'    {"set|ranker|split|budget|glue":52s} {"readacross":>11s} '
              f'{"struct_fid":>11s}')
        for k, v in rows[:15]:
            print(f'    {k:52s} {f(v["readacross"])} {f(v["struct_fid"])}')

    # the three headline contrasts, held everything-else-equal
    print('\n  CANDIDATES vs REGISTERED, same ranker/split/budget/glue')
    wins = {'readacross': [0, 0], 'struct_fid': [0, 0]}
    for k, v in results['combinations'].items():
        if not k.startswith('candidates|'):
            continue
        other = 'registered|' + k.split('|', 1)[1]
        o = results['combinations'].get(other)
        if not o:
            continue
        for field in ('readacross', 'struct_fid'):
            a, b = v[field], o[field]
            if a is None or b is None:
                continue
            wins[field][0 if a > b else 1] += 1
    for field, (c, r) in wins.items():
        print(f'    {field:12s} candidates win {c}, registered win {r}')

    print('\n  GLUE MODE, win counts against raw hstack at the same cell')
    gw = {m: [0, 0] for m in GLUE if m != 'raw'}
    for k, v in results['combinations'].items():
        parts = k.split('|')
        if parts[-1] != 'raw':
            continue
        for m in gw:
            o = results['combinations'].get('|'.join(parts[:-1] + [m]))
            if not o:
                continue
            for field in ('readacross', 'struct_fid'):
                a, b = o[field], v[field]
                if a is None or b is None:
                    continue
                gw[m][0 if a > b else 1] += 1
    for m, (w, l) in gw.items():
        print(f'    {m:14s} beats raw {w}, loses {l}')

    if results['n_skipped_cells']:
        reasons = {}
        for s in results['skipped_cells']:
            reasons[s['reason']] = reasons.get(s['reason'], 0) + 1
        print(f'\n  skipped {results["n_skipped_cells"]} cells:')
        for r, n in reasons.items():
            print(f'    {n:5d}  {r}')

    save_experiment_results(results, args.output,
                            'f22_similarity_feature_sources',
                            dataset=args.dataset, metric_name=metric_name,
                            lower_is_better=lower_is_better)
    print(f'\n  total {results["elapsed_seconds"]}s')
    return results


if __name__ == '__main__':
    main()
