"""Build the winning-recipe SIMILARITY-HYBRID feature matrix for a molecular
dataset, aligned by canonical SMILES to an experiment's molecule list.

This lets the noise experiments (n11/n13) carry an extra representation arm:
the similarity-objective space that the similarity-metrics study found best for
STRUCTURAL fidelity, so we can ask whether a metric chosen for similarity
detects/cleans label noise better than the predictive hybrid and the base rep.

The space is the stacked, z-scored, equal-block matrix
    [count + physchem_minus_saboteurs + 3D_shape + fine2D]
(count block down-weighted x0.5 — the honest variant). Euclidean distance on it
reproduces the winning similarity metric; the noise detectors just consume it as
another X.

WHICH MOLECULES CHOOSE THE FEATURES. Two of the four blocks are chosen, not
fixed: physchem drops saboteur descriptors and fine-2D keeps its top FINE_K.
Both choices are made against MCS structural overlap over sampled molecule
PAIRS. Pass `selection_rows` — one fold's training row numbers into
`smiles_list` — and only pairs with BOTH endpoints in those rows are used, and
the z-score mean and standard deviation of every block are fitted on those rows
too. Then the fold's held-out molecules never touch the feature choice. Omit it
and selection runs over every molecule, which `info['selection_rows'] == 'all'`
records.

Selection is label-BLIND either way: the saboteur-drop and the fine-2D pick are
made against MCS structural overlap, never against activity. A pair is two row
numbers into one specific molecule list, and for one dataset name several lists
exist -- herg_ki is 1415 molecules in the study arrays and 1412 through the
framework loader's InChIKey deduplication, and logd is 1427 here against 5039
uncapped -- so row 900 names a different molecule in each. Before any block is
selected the pairs are therefore checked against the list actually in hand with
kirby.similarity.mcs.assert_pairs_match_smiles.

The July mcs_pairs_<dataset>.npz files record no SMILES ordering, so they cannot
prove which list they belong to and are refused. The pairs are then rebuilt
through kirby.similarity.mcs.get_mcs_pairs, which hashes the list it was given
and caches under that hash in $KIRBY_CACHE_DIR, so the build happens once per
molecule list and never again. It is the same sampler, seed and MCS definition
mcs_cache.build used, so the rebuild reproduces the July pairs rather than
changing them (checked on herg_ki 2026-08-27: resampling the prepared
1415-molecule list gives endpoints identical to mcs_pairs_herg_ki.npz). Cost is
one MCS pass, about 12 minutes per dataset. Restricting to a fold's training
rows re-uses that one pair draw, so a per-fold rebuild pays no MCS cost — only
the greedy descriptor drop, which runs on ~64% of 8000 pairs when 80% of the
molecules are training rows.

It still ONLY works when:
  * a sim_<dataset>.npz + spatial cache exists (herg_ki / logd /
    adme_solubility today), AND
  * EVERY one of the experiment's molecules is present in that cache.
Otherwise it returns (None, info) with a reason, so a PARTIAL arm is never
silently added (e.g. logd is cache-capped to ~1400 rows; a full-logd noise run
gets no arm rather than a 1400-of-5039 arm).

Selection is label-blind, so unlike the predictive hybrid this arm is NOT
oracle-biased.
"""
import os
import sys
import numpy as np

_SIM_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'similarity_metrics'))

# mcs_cache.build's defaults, kept identical so a rebuild reproduces the July
# pairs instead of drawing a different sample.
_N_PAIRS = 8000
_PAIR_SEED = 0

# Fewest structural pairs a fold may select its blocks from. Below this the
# greedy descriptor drop is fitting a Spearman correlation on a handful of
# pairs, so the arm is refused for that fold with a recorded reason rather than
# built from noise.
_MIN_SELECTION_PAIRS = 200

# Per-process caches. `build_similarity_hybrid` is now called once per FOLD, and
# the prepared arrays, the 3D shape block and the MCS pair draw are identical
# across folds -- only which pairs and rows are kept changes. Loading them once
# turns a 5-fold rebuild from five npz loads into one.
_CONTEXT_CACHE = {}
_ALIGN_CACHE = {}


def _canon(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(str(smi))
    return None if m is None else Chem.MolToSmiles(m)


def _zscore_fit(X, fit_rows):
    """z-score every column using the mean and standard deviation of `fit_rows`
    only, then apply to every row. fit_rows=None fits on all rows (what the
    study itself did)."""
    X = np.asarray(X, dtype=np.float64)
    ref = X if fit_rows is None else X[fit_rows]
    mu = ref.mean(0)
    sd = ref.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (X - mu) / sd
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def _make_X(blocks, weights=None, fit_rows=None):
    """Stacked z-scored equal-block feature matrix — the pre-distance form of
    creative_hybrids_v2.make_D (Euclidean on this X == make_D's D). With
    fit_rows the z-score statistics come from those rows only."""
    parts = []
    for k, X in enumerate(blocks):
        Z = _zscore_fit(X, fit_rows)
        Z = Z / np.sqrt(Z.shape[1])          # equal_block scaling
        w = 1.0 if weights is None else float(weights[k])
        parts.append(Z * w)
    return np.hstack(parts)


def _structural_pairs(dataset_name, p, info):
    """MCS pairs proven to be numbered against p['smiles'], rebuilt if not.

    The study's mcs_pairs_<dataset>.npz is tried first and accepted only if it
    can prove it belongs to this molecule list. Storing no ordering hash counts
    as failing to prove it, so today all three July files are refused here and
    the pairs are rebuilt against the list in hand.

    Records in `info`: pairs_source, pairs_ordering, and pairs_refused when a
    stored pair file was rejected.
    """
    from kirby.similarity.mcs import assert_pairs_match_smiles, get_mcs_pairs

    smiles = [str(s) for s in p['smiles']]
    study = os.path.join(_SIM_DIR, f'mcs_pairs_{dataset_name}.npz')
    if os.path.exists(study):
        z = np.load(study)
        try:
            info['pairs_ordering'] = assert_pairs_match_smiles(
                z, smiles, context=os.path.basename(study))
            info['pairs_source'] = os.path.basename(study)
            return z['I'], z['J'], z['mcs'].astype(np.float64)
        except ValueError as e:
            info['pairs_refused'] = str(e)
            print(f'[sim-hybrid] {e}', flush=True)

    print(f'[sim-hybrid] building {_N_PAIRS} MCS pairs for {dataset_name} '
          f'against the {len(smiles)}-molecule prepared list '
          f'(~12 min once, then cached under $KIRBY_CACHE_DIR)', flush=True)
    pairs = get_mcs_pairs(dataset_name, smiles, y=p['y'],
                          n_pairs=_N_PAIRS, seed=_PAIR_SEED)
    info['pairs_ordering'] = str(np.asarray(pairs['smiles_sha']).item())
    info['pairs_source'] = 'kirby.similarity.mcs.get_mcs_pairs'
    return pairs['I'], pairs['J'], pairs['mcs'].astype(np.float64)


def _load_context(dataset_name, info):
    """Everything a build needs that does not depend on which rows select the
    blocks: the prepared arrays, the 3D shape block and the MCS pair draw.

    Returns (context_dict, None) or (None, skip_reason). Cached per process.
    """
    hit = _CONTEXT_CACHE.get(dataset_name)
    if hit is not None:
        ctx, cached_info = hit
        info.update(cached_info)
        return ctx, (None if ctx is not None else cached_info.get('skip'))

    own = {}
    if _SIM_DIR not in sys.path:
        sys.path.insert(0, _SIM_DIR)
    try:
        from _data import load_prepared
        from creative_hybrids_v2 import (
            desc_blacklist_greedy, select_fine2d, load_pharm3d_full, FINE_K)
    except Exception as e:            # rdkit / study module unavailable
        own['skip'] = f'import_failed:{type(e).__name__}:{e}'
        _CONTEXT_CACHE[dataset_name] = (None, own)
        info.update(own)
        return None, own['skip']

    p = load_prepared(dataset_name)
    pharm, n_built = load_pharm3d_full(dataset=dataset_name, n=p['n'])
    if pharm is None:
        own['skip'] = f'no_3d_shape(n_built={n_built}<{p["n"]})'
        _CONTEXT_CACHE[dataset_name] = (None, own)
        info.update(own)
        return None, own['skip']

    I, J, mcs = _structural_pairs(dataset_name, p, own)

    ctx = {
        'p': p,
        'pharm': pharm,
        'I': np.asarray(I), 'J': np.asarray(J), 'mcs': np.asarray(mcs),
        'coarse': p['ecfp4_count'],
        'desc': p['desc'],
        'X_fine_all': np.hstack([p['atompair_count'].astype(np.float64),
                                 p['torsion'].astype(np.float64)]),
        'desc_blacklist_greedy': desc_blacklist_greedy,
        'select_fine2d': select_fine2d,
        'FINE_K': FINE_K,
    }
    _CONTEXT_CACHE[dataset_name] = (ctx, own)
    info.update(own)
    return ctx, None


def _align(dataset_name, p, smiles_list):
    """Experiment row -> study row, by canonical SMILES (first occurrence).
    -1 where the experiment molecule is absent from the study cache. Cached per
    process because RDKit canonicalises every molecule on both sides."""
    key = (dataset_name, len(smiles_list),
           hash(tuple(str(s) for s in smiles_list)))
    hit = _ALIGN_CACHE.get(key)
    if hit is not None:
        return hit

    idx_by_smiles = {}
    for i, s in enumerate(p['smiles']):
        cs = _canon(s)
        if cs is not None:
            idx_by_smiles.setdefault(cs, i)

    rows, missing = [], 0
    for s in smiles_list:
        j = idx_by_smiles.get(_canon(s))
        if j is None:
            missing += 1
            rows.append(-1)
        else:
            rows.append(j)
    out = (np.asarray(rows, dtype=np.int64), missing)
    _ALIGN_CACHE[key] = out
    return out


def build_similarity_hybrid(dataset_name, smiles_list, selection_rows=None):
    """Return (X_sim aligned row-for-row to smiles_list, info) or (None, info).

    selection_rows: row numbers into `smiles_list` — one fold's TRAINING rows —
    whose molecules are the only ones allowed to choose the physchem and fine-2D
    blocks and to set the z-score statistics. None selects over every molecule
    and records info['selection_rows'] == 'all'.

    The returned matrix always covers every row of `smiles_list`, so the caller
    slices its own training and validation rows out of it as before.

    info always carries a 'skip' reason when X_sim is None, and coverage counts
    when it is not, so the caller can log exactly what happened.
    """
    info = {'dataset': dataset_name}
    if smiles_list is None or len(smiles_list) == 0:
        info['skip'] = 'no_smiles'
        return None, info
    if not os.path.exists(os.path.join(_SIM_DIR, f'sim_{dataset_name}.npz')):
        info['skip'] = 'no_similarity_cache'
        return None, info

    ctx, skip = _load_context(dataset_name, info)
    if ctx is None:
        info['skip'] = skip
        return None, info

    p = ctx['p']
    rows, missing = _align(dataset_name, p, smiles_list)

    info.update(n_experiment=len(smiles_list), n_cache=int(p['n']),
                n_missing=int(missing))
    if missing > 0:
        info['skip'] = (f'incomplete_coverage'
                        f'({missing}/{len(smiles_list)}_absent_from_cache)')
        return None, info

    # Which study rows may choose the blocks, and which pairs therefore survive.
    I, J, mcs = ctx['I'], ctx['J'], ctx['mcs']
    if selection_rows is None:
        info['selection_rows'] = 'all'
        info['n_selection_rows'] = int(len(rows))
        fit_rows = None
    else:
        sel = np.asarray(selection_rows, dtype=np.int64).ravel()
        study_sel = np.unique(rows[sel])
        study_sel = study_sel[study_sel >= 0]
        keep = np.isin(I, study_sel) & np.isin(J, study_sel)
        I, J, mcs = I[keep], J[keep], mcs[keep]
        fit_rows = study_sel
        info['selection_rows'] = 'train'
        info['n_selection_rows'] = int(len(study_sel))
        info['n_selection_pairs'] = int(len(I))
        if len(I) < _MIN_SELECTION_PAIRS:
            info['skip'] = (f'too_few_training_pairs('
                            f'{len(I)}<{_MIN_SELECTION_PAIRS})')
            return None, info

    desc = ctx['desc']
    _blk, _tr, kept = ctx['desc_blacklist_greedy'](desc, I, J, mcs)  # label-blind
    physchem_bl = desc[:, kept]
    fidx = ctx['select_fine2d'](ctx['X_fine_all'], I, J, mcs, ctx['FINE_K'])
    fine = ctx['X_fine_all'][:, fidx]                                # label-blind

    # winning recipe, honest (count block down-weighted x0.5)
    X_study = _make_X([ctx['coarse'], physchem_bl, ctx['pharm'], fine],
                      weights=[0.5, 1, 1, 1], fit_rows=fit_rows)

    info['dim'] = int(X_study.shape[1])
    info['n_physchem_kept'] = int(kept.sum())
    return X_study[rows], info


def similarity_hybrid_available(dataset_name, smiles_list):
    """Can this dataset carry the arm at all? (bool, info).

    Runs every check `build_similarity_hybrid` runs — the caches, the 3D shape
    block, the MCS pair draw and full SMILES coverage — and stops before the
    greedy descriptor drop, which is the only expensive part and the only part
    that depends on which rows select. Callers that rebuild per fold use this
    once to print the arm ON/OFF line without paying for a whole-dataset build
    they would then throw away.
    """
    info = {'dataset': dataset_name}
    if smiles_list is None or len(smiles_list) == 0:
        info['skip'] = 'no_smiles'
        return False, info
    if not os.path.exists(os.path.join(_SIM_DIR, f'sim_{dataset_name}.npz')):
        info['skip'] = 'no_similarity_cache'
        return False, info

    ctx, skip = _load_context(dataset_name, info)
    if ctx is None:
        info['skip'] = skip
        return False, info

    p = ctx['p']
    rows, missing = _align(dataset_name, p, smiles_list)
    info.update(n_experiment=len(smiles_list), n_cache=int(p['n']),
                n_missing=int(missing), n_pairs=int(len(ctx['I'])))
    if missing > 0:
        info['skip'] = (f'incomplete_coverage'
                        f'({missing}/{len(smiles_list)}_absent_from_cache)')
        return False, info
    return True, info
