"""Stacked similarity-space recipes.

A "recipe" is one representation built by gluing together up to four blocks of
numbers, side by side, one row per molecule:

  count        ecfp4 counting fingerprint, 1024 columns
  physchem_bl  the 208 RDKit descriptors with structurally harmful columns
               dropped (chosen on the training fold only)
  physchem_full  all 208 descriptors, nothing dropped
  shape        pharmacophore fingerprint from a 3D conformer, 49 columns
  fine2D       the 100 most structurally informative columns of
               [atompair counts | torsion bits] (chosen on the training fold only)

Each block is z-scored and divided by sqrt(its width) so a 1024-column block
can't drown a 49-column one, and the count block can optionally be turned down
by half. This is the algebra from the standalone study
(`creative_hybrids_v2.make_D`), lifted into the framework verbatim.

`RECIPE_ARCHITECTURES` holds nine named combinations. Corrected 2026-08-26: an
earlier version of this docstring said the nine were "every on/off combination"
of the blocks, and they are not. With `count` always present there are eight
combinations of count / physchem / shape / fine2D, and four of them appear here
(count alone, +physchem, +shape, all four). Missing: count+fine2D,
count+physchem+shape, count+physchem+fine2D, count+shape+fine2D. The remaining
five names are the `_half` weighting and the undropped `physchem_full` variant.
The nine were written out by hand in a chat; no code ranked them and no code
generated them.
"""
import numpy as np


# ------------------------------------------------------------ per-fold factory
# Nine named combinations, NOT every on/off combination — see the module
# docstring for which four are missing. `_half` turns the 1024-feature count
# block down by half so it carries the same weight as the narrower blocks it is
# glued to.
#
# 2026-08-26, Adelaide: there is to be no top-N of these. At most ten
# representations per dataset, and the allocation strategies plus the objective
# decide how they combine. Whether these nine survive at all depends on
# tests/experiments/framework/f22_similarity_feature_sources.py, which measures
# the four parts as four separate representations against each dataset's own
# list.
RECIPE_ARCHITECTURES = {
    # fingerprint alone
    'sim_count':              (['count'], None),
    # fingerprint + descriptors, harmful descriptor columns dropped per fold
    'sim_count_desc':         (['count', 'physchem_bl'], None),
    'sim_count_desc_half':    (['count', 'physchem_bl'], [0.5, 1.0]),
    # fingerprint + all 208 descriptors, nothing dropped
    'sim_count_desc208':      (['count', 'physchem_full'], None),
    'sim_count_desc208_half': (['count', 'physchem_full'], [0.5, 1.0]),
    # fingerprint + 3D shape
    'sim_count_shape':        (['count', 'shape'], None),
    'sim_count_shape_half':   (['count', 'shape'], [0.5, 1.0]),
    # all four blocks
    'sim_all':                (['count', 'physchem_bl', 'shape', 'fine2D'], None),
    'sim_all_half':           (['count', 'physchem_bl', 'shape', 'fine2D'],
                               [0.5, 1.0, 1.0, 1.0]),
}


def make_recipe_factory(smiles_list, dataset_name, blocks, weights=None,
                        top_k=100, max_drop=None, n_pairs=8000, seed=0):
    """Per-fold stacked similarity recipe (leakage-safe), one rep per architecture.

    `blocks` names the architecture; each kind becomes one z-scored, equal_block-
    scaled, weighted block, hstacked into the model-facing feature matrix:
      'count'         ecfp4_count(1024), no selection
      'physchem_full' desc208 raw, no selection
      'physchem_bl'   desc208 with TRAIN-only saboteur-drop column selection
      'shape'         pharmacophore3d(49), no selection
      'fine2D'        top-k of [atompair_count | torsion-bits(1024)] by train MCS

    Structural matrices are precomputed once over all rows (structure is not the
    label -> safe). Per fold: any block needing selection is fit on the train
    pairs only; every block is z-scored with TRAIN-row stats, equal_block-scaled
    (/sqrt d) and weighted, then hstacked. Same contract as the other factories.

    Returns: factory(train_idx, val_idx, train_labels) -> (train, val).
    """
    from kirby.similarity.blocks import get_block
    from kirby.similarity.mcs import get_mcs_pairs, restrict_pairs_to_rows
    from kirby.similarity.selection import (
        select_saboteur_drop, select_fine2d, _zscore_with_stats, _train_stats,
    )

    needs_mcs = ('physchem_bl' in blocks) or ('fine2D' in blocks)
    raw = {}
    if 'count' in blocks:
        raw['count'] = get_block('count', smiles_list)
    if 'physchem_full' in blocks or 'physchem_bl' in blocks:
        raw['desc'] = get_block('desc', smiles_list)
    if 'shape' in blocks:
        raw['shape'] = get_block('shape', smiles_list)
    if 'fine2D' in blocks:
        raw['fine'] = get_block('fine', smiles_list)

    pairs = (get_mcs_pairs(dataset_name, smiles_list, n_pairs=n_pairs, seed=seed)
             if needs_mcs else None)

    def _eq_block_z(M_tr, M_val):
        mu, sd = _train_stats(M_tr)
        Ztr = _zscore_with_stats(M_tr, mu, sd)
        Zval = _zscore_with_stats(M_val, mu, sd)
        s = np.sqrt(Ztr.shape[1]) if Ztr.shape[1] else 1.0
        return Ztr / s, Zval / s

    def factory(train_idx, val_idx, train_labels):
        tr = np.asarray(train_idx); va = np.asarray(val_idx)
        tp = restrict_pairs_to_rows(pairs, tr) if pairs is not None else None
        parts_tr, parts_val = [], []
        for k, kind in enumerate(blocks):
            if kind == 'count':
                M = raw['count']
            elif kind == 'physchem_full':
                M = raw['desc']
            elif kind == 'physchem_bl':
                desc = raw['desc']
                if tp is None or len(tp['I']) < 2:
                    kept = np.ones(desc.shape[1], bool)
                else:
                    mu, sd = _train_stats(desc[tr])
                    Z = _zscore_with_stats(desc, mu, sd)
                    kept, _, _ = select_saboteur_drop(
                        Z, tp['I'], tp['J'], tp['mcs'], max_drop=max_drop)
                M = desc[:, kept]
            elif kind == 'shape':
                M = raw['shape']
            elif kind == 'fine2D':
                fine = raw['fine']
                if tp is None or len(tp['I']) < 2:
                    cols = np.arange(min(top_k, fine.shape[1]))
                else:
                    cols = select_fine2d(fine, tp['I'], tp['J'], tp['mcs'], top_k)
                M = fine[:, cols]
            else:
                raise ValueError(f"unknown recipe block kind: {kind}")
            Ztr, Zval = _eq_block_z(M[tr], M[va])
            w = 1.0 if weights is None else float(weights[k])
            parts_tr.append(Ztr * w); parts_val.append(Zval * w)
        return (np.hstack(parts_tr).astype(np.float32),
                np.hstack(parts_val).astype(np.float32))

    return factory


