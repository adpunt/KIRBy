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

REUSES the study's precomputed MCS + 3D caches (label-BLIND block selection:
saboteur-drop and fine-2D are chosen against MCS overlap, never activity), so no
server precompute is needed here. But it therefore ONLY works when:
  * a sim_<dataset>.npz + mcs_pairs + spatial cache exists (herg_ki / logd /
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


def _canon(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(str(smi))
    return None if m is None else Chem.MolToSmiles(m)


def _make_X(blocks, weights=None):
    """Stacked z-scored equal-block feature matrix — the pre-distance form of
    creative_hybrids_v2.make_D (Euclidean on this X == make_D's D)."""
    from creative_hybrids_v2 import _zscore
    parts = []
    for k, X in enumerate(blocks):
        Z = _zscore(X)
        Z = Z / np.sqrt(Z.shape[1])          # equal_block scaling
        w = 1.0 if weights is None else float(weights[k])
        parts.append(Z * w)
    return np.hstack(parts)


def build_similarity_hybrid(dataset_name, smiles_list):
    """Return (X_sim aligned row-for-row to smiles_list, info) or (None, info).

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
    if not os.path.exists(os.path.join(_SIM_DIR, f'mcs_pairs_{dataset_name}.npz')):
        info['skip'] = 'no_mcs_cache'
        return None, info

    if _SIM_DIR not in sys.path:
        sys.path.insert(0, _SIM_DIR)
    try:
        from _data import load_prepared
        from creative_hybrids_v2 import (
            desc_blacklist_greedy, select_fine2d, load_pharm3d_full, FINE_K)
    except Exception as e:            # rdkit / study module unavailable
        info['skip'] = f'import_failed:{type(e).__name__}:{e}'
        return None, info

    p = load_prepared(dataset_name)
    pharm, n_built = load_pharm3d_full(dataset=dataset_name, n=p['n'])
    if pharm is None:
        info['skip'] = f'no_3d_shape(n_built={n_built}<{p["n"]})'
        return None, info

    m = np.load(os.path.join(_SIM_DIR, f'mcs_pairs_{dataset_name}.npz'))
    I, J, mcs = m['I'], m['J'], m['mcs'].astype(np.float64)

    coarse = p['ecfp4_count']
    desc = p['desc']
    X_fine_all = np.hstack([p['atompair_count'].astype(np.float64),
                            p['torsion'].astype(np.float64)])
    _blk, _tr, kept = desc_blacklist_greedy(desc, I, J, mcs)   # label-blind
    physchem_bl = desc[:, kept]
    fidx = select_fine2d(X_fine_all, I, J, mcs, FINE_K)        # label-blind
    fine = X_fine_all[:, fidx]

    # winning recipe, honest (count block down-weighted x0.5)
    X_study = _make_X([coarse, physchem_bl, pharm, fine], weights=[0.5, 1, 1, 1])

    # align study rows -> experiment rows by canonical SMILES (first occurrence)
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

    info.update(n_experiment=len(smiles_list), n_cache=int(p['n']),
                n_missing=int(missing), dim=int(X_study.shape[1]))
    if missing > 0:
        info['skip'] = (f'incomplete_coverage'
                        f'({missing}/{len(smiles_list)}_absent_from_cache)')
        return None, info

    return X_study[np.asarray(rows)], info
