#!/usr/bin/env python3
"""Fed03: Synthetic Private Datasets.

Two sub-experiments, chosen by which dataset is passed:
    Part A (herg_ki, herg, lhasa_herg): Source-based hERG — Lhasa's curated
        data and KIRBy's ChEMBL data as real federated silos with genuine
        source heterogeneity.
    Part B (any other molecular dataset): Asymmetric representation silos —
        each organisation holds a different family of representations.

There was a Part C on antibody data. It was deleted on 2026-09-12; fed05
answers the antibody federated questions. Passing an antibody dataset here
raises and says so.

Usage:
    python fed03_synthetic_private.py --dataset herg_ki
    python fed03_synthetic_private.py --dataset caco2
    python fed03_synthetic_private.py --dataset logd --strategy asymmetric
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import traceback
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from experiments._shared import (
    load_data,
    save_experiment_results, convert_numpy, cv_split,
    get_best_search_for_downstream, get_domain_metric,
)
from experiments.federated.federated_helpers import (
    compute_clf_metrics, compute_reg_metrics,
    predict_fluid_similarity_ensemble,
    predict_fluid_quality_ensemble,
    federated_feature_selection,
    CLASSIFICATION_CUTOFFS,
    binarise_labels,
    rank_silos_train_only,
    _resolve_match,
    smoke_n_folds, smoke_search_kwargs,
    fit_fed_classifier, fit_fed_model,
    predict_fluid_similarity_ensemble_classification,
    fed_to_kirby_model, make_fed_classifier, make_fed_regressor,
)


# ── Constants ────────────────────────────────────────────────────────────────

# The pKi cut that turns ChEMBL's hERG Ki into blocker / non-blocker for
# Part A's first silo. It is read from CLASSIFICATION_CUTOFFS so this file and
# fed01/fed04 cannot drift apart: the same molecules must get the same class in
# every federated experiment. Changing the cut is a one-line change in
# federated_helpers.CLASSIFICATION_CUTOFFS and it reaches all four files.
PKI_THRESHOLD = CLASSIFICATION_CUTOFFS['herg_ki'][0]
PKI_THRESHOLD_NOTE = CLASSIFICATION_CUTOFFS['herg_ki'][1]

# Part B: which family each organisation holds.
#
# These seven names used to be the whole of Part B: a silo took whichever of
# its listed names the dataset happened to have, and anything else in the
# dataset's own pool was unused. On herg_ki that emptied the learned family
# outright (neither molformer nor chemberta is in its pruned pool), so Part B
# ran two silos while reporting three; on logd each family came down to a
# single representation. Measured 2026-09-12.
#
# Every representation in the dataset's pool is sorted into a family now, by
# what builds it, so no representation goes unused and no family empties while
# the dataset has something that belongs in it. The names below say which
# family each kind of representation goes to; a name in the pool that matches
# nothing here is reported by name rather than dropped in silence.
ASYMMETRIC_SILO_REPS = {
    'structural': ['ecfp4', 'graph_kernel', 'atom_pair'],
    'physicochemical': ['mordred', 'pdv'],
    'learned': ['molformer', 'chemberta'],
}

# {family: the representations that belong to it}. Molecular names only —
# Part B runs on small-molecule datasets.
REP_FAMILIES = {
    # Substructure enumerators hashed into bits or counts, plus graph
    # statistics and kernels computed off the molecular graph.
    'structural': {
        'ecfp4', 'maccs', 'avalon', 'atom_pair', 'top_torsion', 'rdkfp',
        'map4', 'e3fp', 'drfp', 'drfp_2048', 'cgr_morgan', 'graph_kernel',
        'sali_cliff', 'sns',
    },
    # Computed descriptor blocks, and 3D shape and pharmacophore descriptors.
    'physicochemical': {
        'mordred', 'mordred_pca50', 'pdv', 'pdv_binary_mean', 'pdv_rank',
        'padel', 'rdkit_desc', 'erg', 'coulomb_matrix', 'usr', 'usrcat',
        'usr_shape', 'pharmacophore3d', 'tda',
    },
    # Pretrained chemical language models and pretrained graph encoders,
    # read out frozen.
    'learned': {
        'mol2vec', 'chemberta', 'chembert', 'molformer', 'selformer',
        'molbert', 'molclr', 'graphmvp', 'smited', 'mhg_gnn', 'grover',
        'unimol_v2', 'minimol', 'gp_molformer', 'mole', 'clamp', 'rmat',
    },
}


# Part B's total feature budget. C2 takes all of it in one selection over
# every family; C3 splits it across the families and concatenates. Naming it
# once is what makes those two comparable — C2 used to take create_hybrid's
# default of 100 while each of C3's families took 100 of its own, so C3 came
# out several times wider and the difference between them mixed the method
# with the size.
PART_B_BUDGET = 100

# Feature budgets both parts sweep.
#
# Until 2026-09-23 Part A's C4 and C6 passed no budget at all and took
# create_hybrid's default of 100, so the result this experiment is best known
# for — 0.5504 MCC against 0.4005 for the best single source — was measured at
# one width and nobody knew whether it held at any other. Part B was pinned to
# 100 the same way. A budget is a swept axis like any other and the server run
# covers the range.
#
# KIRBY_FED03_BUDGETS overrides it as a comma-separated list, for a local
# debugging run that cannot afford five widths over 13,231 pooled molecules.
def _fed03_budgets(default=(50, 100, 250, 500, 1000)):
    # KIRBY_FED03_BUDGETS still works for fed03 alone; KIRBY_FED_BUDGETS is
    # the suite-wide switch every federated file now reads.
    raw = os.environ.get('KIRBY_FED03_BUDGETS', '').strip()
    if raw:
        return [int(x) for x in raw.split(',') if x.strip()]
    from experiments.federated.federated_helpers import federated_budgets
    return federated_budgets(default)


PART_A_BUDGETS = _fed03_budgets()
PART_B_BUDGETS = _fed03_budgets()
# The width every summary table and every cross-run comparison quotes. It has
# to be in the sweep or the headline number has nothing behind it.
HEADLINE_BUDGET = 100 if 100 in PART_A_BUDGETS else PART_A_BUDGETS[0]


def assign_rep_families(available_reps):
    """Sort a dataset's own representations into the three families.

    Returns (silo_reps, unclassified) where silo_reps is
    {family: [rep names]} holding only families that got at least one
    representation, and unclassified lists pool names that match no family.
    """
    silo_reps = {}
    classified = set()
    for family, members in REP_FAMILIES.items():
        hit = sorted(r for r in available_reps if r in members)
        if hit:
            silo_reps[family] = hit
            classified.update(hit)
    unclassified = sorted(set(available_reps) - classified)
    return silo_reps, unclassified

# Antibody sets that used to route into Part C. Part C is gone; these names
# are kept only so a run that asks for one is refused by name and pointed at
# fed05, rather than falling through to the molecular path and producing
# something meaningless.
ANTIBODY_DATASETS = {
    'ddg_combined', 'abagym_EGFR', 'flab_thermo_tresanco',
}


# =============================================================================
# PART A: SOURCE-BASED hERG
# =============================================================================

def _run_herg_source_based(data, config, output_dir, dataset_name,
                           fed_model='rf', fed_seed=0):
    """Strategy 2: Real data sources as federated silos.

    Silo A: KIRBy ChEMBL hERG (1,415 mols, continuous pKi → binarized)
    Silo B: Lhasa training (7,772 mols, binary Active/Inactive)
    Silo C: Lhasa test (4,383 mols, binary Active/Inactive)
    """
    from experiments.federated.lhasa_herg_loader import (
        load_lhasa_herg_training, load_lhasa_herg_test,
        dedup_across_silos,
    )
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

    print("\n" + "=" * 70)
    print("FED03 PART A: SOURCE-BASED hERG (Strategy 2)")
    print("=" * 70)

    reps_chembl = data['reps']
    labels_chembl = data['labels']
    smiles_chembl = data['smiles']

    # Get optimal hybrid config from precomputed search
    augs = data.get('augmentations')
    # Part A is a two-class experiment — every condition below trains on
    # blocker / non-blocker. get_domain_metric returns herg_ki's REGISTERED
    # metric, which is rmse, because the registry knows herg_ki as continuous
    # pKi. fed01 (metric_name = 'auc') and fed04 (metric='auc') both override
    # it for exactly this reason and say so; Part A did not.
    #
    # What that cost, measured 2026-09-23. The engine turns a metric into a
    # skill score, 1 - error / null_error, and clips it at zero. On 0/1 labels
    # with hard predictions the null model — predicting the mean — has RMSE
    # 0.4745, and a classifier at 70% accuracy has RMSE 0.5477, WORSE than the
    # null. Part A's classifiers sit near 0.5 MCC, so every representation
    # clipped to exactly zero and the engine fell back to an equal split,
    # inside every silo, on every fold: "allocation 'linear': every
    # representation weighed zero (6 representations, scores [0.0]), so the
    # budget was split EQUALLY". The allocation method named in the recipe was
    # doing nothing, and the silo scores that divide the budget across the
    # three sources were computed on the wrong quantity too.
    metric_name = 'auc'
    from kirby.find_best_hybrid import BUILTIN_METRICS
    metric_fn, lower_is_better = BUILTIN_METRICS[metric_name]
    # Binarize ChEMBL labels. When the calling dataset already has 0/1 labels
    # (e.g. load_data('herg') returns binary blocker labels — no continuous
    # pKi), the pKi>=6.0 threshold would collapse every sample to class 0.
    # Detect already-binary labels by value and pass them through.
    _uniq_chembl = set(np.unique(labels_chembl).tolist())
    labels_chembl_binary, _chembl_cut_note = binarise_labels(
        'herg_ki', labels_chembl,
        is_binary=bool(data.get('is_binary')))
    # DIRTY-LOCAL 2026-09-28: the search below used to run on the
    # CONTINUOUS pKi labels with task_type='regression' and metric 'auc',
    # which cannot be scored, so it returned no recipe and Part A always
    # fell to the hard defaults (treeshap / match). It now runs on the
    # same blocker/non-blocker labels every Part A condition trains on,
    # so binarisation moved above it.
    try:
        _search, _meta = get_best_search_for_downstream(
            reps_chembl, labels_chembl_binary, data.get('groups'),
            # The width the headline row is read at. It was a fixed 100, so
            # under a sweep that does not contain 100 the lookup fell through
            # to the best recipe at whatever width the search liked.
            downstream_model=fed_to_kirby_model(fed_model),
            budget=HEADLINE_BUDGET,
            augmentations=augs,
            metric=metric_name,
            task_type='classification',
            domain=data.get('domain', 'molecular'),
            dataset_name=dataset_name,
            # data=None, as fed01 and fed04 do: the in-memory store holds
            # searches over the registry's continuous pKi.
            data=None,
            **smoke_search_kwargs(config),
        )
        _bp = _meta['best_config']['blueprint'] if _meta else None
    except Exception as _search_err:
        _bp = None
        print(f"  recipe search FAILED ({_search_err!r}); Part A falls back "
              f"to the hard-coded recipe below")
    # Part A does not pass augmentations to create_hybrid (C6 uses 'none'
    # explicitly because Lhasa rows have no aligned aug arrays). The pool's
    # aug_condition is therefore not consumed here — removed dead aug_cond.
    _alloc = _bp.allocation_method if _bp else 'performance_weighted'
    _imp = _bp.importance_method if _bp else 'treeshap'
    _int_eval = _bp.internal_evaluator if _bp else 'match'
    # Which recipe every KIRBy condition below was built with, and where it
    # came from. The fallback is a different recipe from any searched one,
    # and nothing in the result file used to say which of the two ran.
    _recipe = {'allocation': _alloc, 'importance': _imp,
               'internal_evaluator': _int_eval,
               'source': 'search' if _bp else 'fallback'}
    print(f"  recipe ({_recipe['source']}): allocation {_alloc}, "
          f"importance {_imp}, internal evaluator {_int_eval}")

    print(f"  ChEMBL silo labels: {_chembl_cut_note}")

    # Compute InChIKeys for ChEMBL
    chembl_inchikeys = []
    for smi in smiles_chembl:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                chembl_inchikeys.append(
                    InchiToInchiKey(MolToInchi(mol)))
            else:
                chembl_inchikeys.append('UNKNOWN')
        except Exception:
            chembl_inchikeys.append('UNKNOWN')

    # Load Lhasa data
    print("\nLoading Lhasa hERG data...")
    lhasa_train_smiles, lhasa_train_labels, lhasa_train_keys, _ = \
        load_lhasa_herg_training()
    lhasa_test_smiles, lhasa_test_labels, lhasa_test_keys, _ = \
        load_lhasa_herg_test()

    # Test-only row cap on the two Lhasa silos. load_data already applies
    # KIRBY_SMOKE_N to the ChEMBL silo, but these two are read straight off
    # Lhasa's PKL files and carry 7,772 and 4,383 molecules whatever the cap
    # says. Every representation in the pool is then computed over all 12,155
    # of them before a single fold runs, which is the whole cost of a local
    # Part A. The cap is a stratified head of each file's own row order, so
    # the two classes keep their proportions. No-op when KIRBY_SMOKE_N is
    # unset, so the server pools the full files as before.
    _smoke_n = os.environ.get('KIRBY_SMOKE_N', '').strip()
    if _smoke_n and int(_smoke_n) > 0:
        _cap = int(_smoke_n)
        _rng = np.random.RandomState(0)

        def _cap_silo(sm, lb, ky, name):
            lb = np.asarray(lb)
            if len(lb) <= _cap:
                return sm, lb, ky
            keep = []
            for _cls in np.unique(lb):
                _pos = np.where(lb == _cls)[0]
                _take = max(1, int(round(_cap * len(_pos) / len(lb))))
                keep.extend(_rng.choice(_pos, size=min(_take, len(_pos)),
                                        replace=False).tolist())
            keep = np.sort(np.array(keep))
            print(f"  [SMOKE] {name}: {len(lb)} -> {len(keep)} molecules")
            return ([sm[i] for i in keep], lb[keep],
                    [ky[i] for i in keep])

        lhasa_train_smiles, lhasa_train_labels, lhasa_train_keys = _cap_silo(
            lhasa_train_smiles, lhasa_train_labels, lhasa_train_keys,
            'lhasa_train')
        lhasa_test_smiles, lhasa_test_labels, lhasa_test_keys = _cap_silo(
            lhasa_test_smiles, lhasa_test_labels, lhasa_test_keys,
            'lhasa_test')

    # Cross-silo deduplication (ChEMBL first → keeps ChEMBL, removes from Lhasa)
    silo_smiles = {
        'chembl': smiles_chembl,
        'lhasa_train': lhasa_train_smiles,
        'lhasa_test': lhasa_test_smiles,
    }
    silo_labels = {
        'chembl': labels_chembl_binary,
        'lhasa_train': lhasa_train_labels,
        'lhasa_test': lhasa_test_labels,
    }
    silo_keys = {
        'chembl': chembl_inchikeys,
        'lhasa_train': lhasa_train_keys,
        'lhasa_test': lhasa_test_keys,
    }

    deduped_smiles, deduped_labels, deduped_keys, dedup_stats, \
        label_conflicts = \
        dedup_across_silos(silo_smiles, silo_labels, silo_keys)

    print("\nDeduplication:")
    for silo, stats in dedup_stats.items():
        print(f"  {silo}: {stats['before']} → {stats['after']} "
              f"({stats['removed']} removed)")

    # A molecule dropped here was curated twice: once by ChEMBL, once by
    # Lhasa, each with its own hERG call. Count how often the two calls
    # differ before the Lhasa copy is discarded.
    _cross = {'n_both': 0, 'n_agree': 0, 'n_disagree': 0}
    _cross_pairs = {}
    for _pair, _e in label_conflicts['by_silo_pair'].items():
        if _e['first_silo'] == 'chembl' and _e['later_silo'].startswith('lhasa'):
            _cross_pairs[_pair] = _e
            if _e['comparable']:
                _cross['n_both'] += _e['n_both']
                _cross['n_agree'] += _e['n_agree']
                _cross['n_disagree'] += _e['n_disagree']
    _cross['disagreement_rate'] = (
        _cross['n_disagree'] / _cross['n_both'] if _cross['n_both'] else None)
    _cross['pki_threshold'] = PKI_THRESHOLD
    _cross['by_silo_pair'] = _cross_pairs

    # The rate depends heavily on where ChEMBL's continuous pKi is cut, and
    # this file cuts wherever CLASSIFICATION_CUTOFFS['herg_ki'] says, because
    # that same cut sets the silo labels the federated arms are scored on.
    # The rate at three cuts is recorded here so no reader has to take the
    # chosen one on trust. UNVERIFIED, 2026-09-12: an earlier version of this
    # comment said n06 asks the same question of the same molecules and cuts
    # at 5.0. n06_bindingdb_ground_truth.py has no activity cut in it at all —
    # its target is continuous — so that claim is not checkable here. The 5.0
    # figure appears in N06_HANDOFF_2026_09_11.md as one of three swept cuts,
    # not as n06's own. Recorded here is WHICH WAY the disagreements run: when every one of them is Lhasa-active against ChEMBL-inactive, the
    # count is measuring how far the cut has moved past Lhasa's actives rather
    # than two curators differing.
    _lhasa_call = {}
    for _silo, _keys in (('lhasa_train', lhasa_train_keys),
                         ('lhasa_test', lhasa_test_keys)):
        _labs = (lhasa_train_labels if _silo == 'lhasa_train'
                 else lhasa_test_labels)
        for _k, _l in zip(_keys, _labs):
            if _k and _k != 'UNKNOWN':
                _lhasa_call[_k] = (int(_l), _silo)
    _pki_of_key = {}
    if not (data.get('is_binary') or _uniq_chembl <= {0, 1, 0.0, 1.0}):
        for _k, _p in zip(chembl_inchikeys, labels_chembl):
            if _k and _k != 'UNKNOWN' and np.isfinite(_p):
                _pki_of_key[_k] = float(_p)
    _shared_keys = sorted(set(_pki_of_key) & set(_lhasa_call))
    _sens = {}
    for _thr in (5.0, 6.0, 7.0):
        _diff = [(k, _lhasa_call[k][0], int(_pki_of_key[k] >= _thr))
                 for k in _shared_keys]
        _n_diff = sum(1 for _, a, b in _diff if a != b)
        _sens[str(_thr)] = {
            'n_molecules': len(_shared_keys),
            'n_called_differently': _n_diff,
            'rate': (_n_diff / len(_shared_keys)) if _shared_keys else None,
            'n_lhasa_active_and_chembl_inactive': sum(
                1 for _, a, b in _diff if a == 1 and b == 0),
            'n_lhasa_inactive_and_chembl_active': sum(
                1 for _, a, b in _diff if a == 0 and b == 1),
            'by_lhasa_file': {
                _silo: {
                    'n_molecules': sum(1 for k in _shared_keys
                                       if _lhasa_call[k][1] == _silo),
                    'n_called_differently': sum(
                        1 for k, a, b in _diff
                        if _lhasa_call[k][1] == _silo and a != b),
                } for _silo in ('lhasa_train', 'lhasa_test')},
        }
    _cross['threshold_sensitivity'] = _sens
    _cross['which_lhasa_file_means_what'] = {
        'lhasa_train': "Lhasa's call on ChEMBL compounds and ChEMBL assays "
                       "(the SDF field is 'Lhasa Chembl Call'), so a "
                       "disagreement here is one organisation's curation "
                       "against a threshold on the same source",
        'lhasa_test': "an 'Overall Conservative Call' over the Doddareddy, "
                      "Li, Robinson and Sun datasets — independent of ChEMBL",
    }
    if _sens:
        print("  rate by threshold:")
        for _thr, _row in _sens.items():
            print(f"    pKi cut {_thr}: {_row['n_called_differently']} of "
                  f"{_row['n_molecules']} called differently "
                  f"(Lhasa active / ChEMBL inactive "
                  f"{_row['n_lhasa_active_and_chembl_inactive']}, the other "
                  f"way {_row['n_lhasa_inactive_and_chembl_active']})")

    print("\nCross-curator label disagreement (molecules held by both "
          "ChEMBL and Lhasa):")
    print(f"  held by both: {_cross['n_both']} molecules")
    print(f"  same call:    {_cross['n_agree']} molecules")
    print(f"  different:    {_cross['n_disagree']} molecules")
    if _cross['disagreement_rate'] is not None:
        print(f"  different / held by both: "
              f"{_cross['disagreement_rate']:.4f}")
    print(f"  ChEMBL pKi binarised at: {PKI_THRESHOLD} "
          f"({PKI_THRESHOLD_NOTE})")
    for _pair, _e in _cross_pairs.items():
        print(f"    {_pair}: {_e['n_disagree']} different of "
              f"{_e['n_both']} held by both"
              f"{'' if _e['comparable'] else '  (labels not comparable)'}")

    # Pool all silos for evaluation
    all_smiles = []
    all_labels = []
    silo_assignments = []
    silo_names = ['chembl', 'lhasa_train', 'lhasa_test']

    for i, sname in enumerate(silo_names):
        sm = deduped_smiles[sname]
        lb = deduped_labels[sname]
        all_smiles.extend(sm)
        all_labels.extend(lb)
        silo_assignments.extend([i] * len(sm))

    all_labels = np.array(all_labels, dtype=int)
    silo_assignments = np.array(silo_assignments)
    n_total = len(all_labels)

    print(f"\nPooled: {n_total} molecules across 3 silos")
    for i, sname in enumerate(silo_names):
        n_s = int((silo_assignments == i).sum())
        n_act = int(all_labels[silo_assignments == i].sum())
        print(f"  Silo {i} ({sname}): {n_s} mols "
              f"({n_act} active, {n_s - n_act} inactive)")

    # Representations for ALL pooled molecules, built in ONE call.
    #
    # DIRTY-LOCAL 2026-09-28. This used to take the ChEMBL rows from the
    # dataset's cache, build the Lhasa rows in a second call, and stack the
    # two. graph_kernel cannot be stacked that way: create_graph_kernel finds
    # its WL-label vocabulary in the molecules it is given and then drops
    # labels that are zero in more than 95% of them, so column j meant a
    # different label in the ChEMBL block than in the Lhasa block. When the
    # widths differed the representation was dropped with a warning; when
    # they matched it would have been stacked with its columns misaligned.
    # One call over the pooled SMILES gives every row the same columns. The
    # vocabulary uses no labels, which is how every cached graph_kernel in
    # the project is built (whole dataset, label-free).
    #
    # It also retires a row-alignment risk. The old ChEMBL keep-mask kept
    # every row whose InChIKey survived dedup, so two ChEMBL rows sharing a
    # key would both stay while dedup kept one, shifting every row after it.
    print("\nComputing representations for all pooled molecules...")
    from find_best_hybrid_molecular import get_representations as get_molecular_representations
    rep_names = list(reps_chembl.keys())
    n_chembl = len(deduped_smiles['chembl'])
    n_lhasa_tr = len(deduped_smiles['lhasa_train'])
    n_lhasa_te = len(deduped_smiles['lhasa_test'])

    # create_graph_kernel skips a molecule RDKit cannot parse and so returns
    # its rows shifted. Refuse graph_kernel rather than stack shifted rows.
    _n_unparsed = sum(1 for _s in all_smiles if Chem.MolFromSmiles(_s) is None)
    _rep_request = list(rep_names)
    if _n_unparsed and 'graph_kernel' in _rep_request:
        print(f"  graph_kernel dropped: {_n_unparsed} pooled SMILES do not "
              f"parse, and create_graph_kernel would shift every row after "
              f"them.")
        _rep_request.remove('graph_kernel')
    pooled_reps = get_molecular_representations(
        all_smiles, ','.join(_rep_request))

    combined_reps = {}
    for rname in rep_names:
        arr = pooled_reps.get(rname)
        if arr is None:
            warnings.warn(f"Rep {rname} could not be built for the pooled "
                          f"molecules; Part A runs without it")
            continue
        if arr.shape[0] != n_total:
            raise RuntimeError(
                f"Rep {rname}: {arr.shape[0]} rows for {n_total} pooled "
                f"molecules, so its rows are not the pooled molecules.")
        combined_reps[rname] = arr

    print(f"  Combined reps: {len(combined_reps)} reps "
          f"({', '.join(f'{k}={v.shape[1]}' for k, v in combined_reps.items())}), "
          f"{n_total} molecules")

    # Train/test split: scaffold-group-aware for molecular datasets.
    # Per feedback_no_random_splits.md, a single 80/20 split is not enough;
    # use 5-fold scaffold-group CV and accumulate per-fold metrics.
    from find_best_hybrid_molecular import assign_scaffold_groups
    pooled_groups, _ = assign_scaffold_groups(all_smiles)
    if pooled_groups is not None:
        pooled_groups = np.asarray(pooled_groups)
    if pooled_groups is None:
        raise RuntimeError(
            "Part A 5-fold CV requires scaffold groups; pooled_groups is None.")
    n_unique_groups = len(np.unique(pooled_groups))
    if n_unique_groups < 5:
        raise RuntimeError(
            f"Part A 5-fold CV requires >= 5 unique scaffold groups; "
            f"got {n_unique_groups}.")

    K = len(silo_names)

    # Build Morgan FPs for FLuID conditions (once, all rows)
    from rdkit.Chem import AllChem, DataStructs

    def _compute_morgan(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return np.zeros(2048, dtype=np.int8)
        arr = np.zeros(2048, dtype=np.int8)
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, 2, nBits=2048, useFeatures=True)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    all_fps = np.array([_compute_morgan(s) for s in all_smiles])

    # ── Per-fold accumulators ─────────────────────────────────────────────
    per_fold_results = []
    n_folds = smoke_n_folds(5)   # 2 under KIRBY_SMOKE_N, else 5
    folds = list(cv_split(
        pooled_groups, n_folds, np.zeros(n_total), all_labels, seed=fed_seed))
    print(f"\n{n_folds}-fold scaffold CV over {n_unique_groups} groups")

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\n── Fold {fold_idx + 1}/{n_folds} "
              f"({len(train_idx)} train / {len(test_idx)} test) ──")
        y_train = all_labels[train_idx]
        y_test = all_labels[test_idx]
        train_silo = silo_assignments[train_idx]

        # Build silo indices relative to train set (per fold)
        train_silo_indices = {}
        for i, sname in enumerate(silo_names):
            train_silo_indices[i + 1] = np.where(train_silo == i)[0]

        train_fps = all_fps[train_idx]
        test_fps = all_fps[test_idx]

        # ── Conditions ────────────────────────────────────────────────
        results = {}

        # C1: Best single silo
        print("  C1: Single silo baselines...")
        single_silo = {}
        for i, sname in enumerate(silo_names):
            s_idx = train_silo_indices[i + 1]
            if len(s_idx) < 10:
                single_silo[sname] = {'error': 'too few samples'}
                continue
            clf = fit_fed_classifier(
                fed_model, train_fps[s_idx], y_train[s_idx], seed=fed_seed)
            pred = clf.predict(test_fps)
            pred_prob = clf.predict_proba(test_fps)[:, 1]
            single_silo[sname] = compute_clf_metrics(y_test, pred, pred_prob)
            print(f"    {sname}: MCC={single_silo[sname]['mcc']:.3f}")

        valid_silos = [s for s in single_silo if 'mcc' in single_silo[s]]
        # Two picks, reported side by side and never merged.
        #
        #  test_picked_ceiling — the source with the best MCC on THIS fold's
        #    held-out molecules. It needs the answers, so it is a bound: the
        #    best single source available to someone who already knew which
        #    one would win.
        #  train_picked — the source that won on rows held out inside its own
        #    training molecules, scored afterwards on the fold's test
        #    molecules. This is the honest single-source baseline and it is
        #    the one C4's gain must be read against.
        best_silo = (max(valid_silos, key=lambda s: single_silo[s]['mcc'])
                     if valid_silos else None)

        def _probe_silo(sname):
            """One source's score on rows held out inside its own molecules.

            The split is on scaffold groups. It used to be a
            StratifiedShuffleSplit with no groups, so two molecules sharing a
            Bemis-Murcko scaffold could sit on both sides of it and the score
            this pick is made on was read off molecules the model had already
            seen a twin of. fed02's inner representation ranking
            (_rank_reps_train_only) and Part B's family probe (_probe_family)
            both take groups and both refuse rather than split at random;
            this one did not, so the source Part A calls the honest pick was
            chosen on the one split in the three that the project forbids.
            Returns None when the source has too few distinct scaffolds to
            hold any out, which drops it from the ranking rather than ranking
            it on a random split.
            """
            i = silo_names.index(sname)
            s_idx = train_silo_indices[i + 1]
            if len(s_idx) < 20:
                return None
            y_s = y_train[s_idx]
            if len(np.unique(y_s)) < 2:
                return None
            g_s = pooled_groups[train_idx][s_idx]
            if len(np.unique(g_s)) < 2:
                return None
            from sklearn.model_selection import GroupShuffleSplit
            sp = GroupShuffleSplit(n_splits=1, test_size=0.25,
                                   random_state=fed_seed + i)
            tr_v, va_v = next(sp.split(np.zeros((len(s_idx), 1)), y_s,
                                       groups=g_s))
            if len(np.unique(y_s[tr_v])) < 2 or len(va_v) == 0:
                return None
            c = fit_fed_classifier(fed_model, train_fps[s_idx][tr_v],
                                   y_s[tr_v], seed=fed_seed)
            m = compute_clf_metrics(y_s[va_v], c.predict(train_fps[s_idx][va_v]))
            return m.get('mcc')

        best_silo_train, inner_scores = rank_silos_train_only(
            _probe_silo, silo_names, seed=fed_seed)
        print(f"    best on the held-out fold (ceiling): {best_silo}")
        print(f"    best on its own training rows (honest): {best_silo_train}")
        results['c1_single_silo'] = convert_numpy({
            'per_silo': single_silo,
            'best_silo': best_silo,
            'test_picked_ceiling': best_silo,
            'train_picked': best_silo_train,
            'train_picked_inner_scores': inner_scores,
            'train_picked_metrics': (single_silo.get(best_silo_train)
                                     if best_silo_train else None),
        })

        # Hoisted: train teachers per silo BEFORE C2/C3 so both conditions
        # share the same state. Previously teacher_models/cluster_data_train
        # were defined inside C2's try block, causing NameError on C3 if C2
        # raised before completing (or silent stale-state reuse).
        teachers_ready = False
        teacher_models = [None]
        cluster_data_train = [None]
        try:
            for i, sname in enumerate(silo_names):
                s_idx = train_silo_indices[i + 1]
                X_s = train_fps[s_idx]
                y_s = y_train[s_idx]
                clf = fit_fed_classifier(fed_model, X_s, y_s, seed=fed_seed)

                # Create minimal DataFrame for similarity ensemble
                import pandas as pd_local
                df_s = pd_local.DataFrame({
                    'FP': list(X_s),
                    'CLASS': y_s,
                })
                teacher_models.append((clf, len(X_s), df_s))
                cluster_data_train.append(df_s)
            teachers_ready = True
        except Exception as e:
            print(f"    Teacher training FAILED: {e}")

        # C2: FLuID similarity ensemble
        print("  C2: FLuID similarity ensemble...")
        if not teachers_ready:
            results['c2_fluid_similarity'] = {'error': 'teachers not trained'}
            print(f"    SKIPPED: teachers not trained")
        else:
            try:
                pred_cls, pred_prob, tw = predict_fluid_similarity_ensemble(
                    teacher_models, cluster_data_train, test_fps, K)
                c2 = compute_clf_metrics(y_test, pred_cls, pred_prob)
                c2['teacher_weights_mean'] = convert_numpy(
                    {silo_names[i]: float(tw[:, i].mean())
                     for i in range(K)})
                results['c2_fluid_similarity'] = convert_numpy(c2)
                print(f"    MCC={c2['mcc']:.3f}")
            except Exception as e:
                results['c2_fluid_similarity'] = {'error': str(e)}
                print(f"    FAILED: {e}")

        # C3: FLuID quality ensemble
        print("  C3: FLuID quality ensemble...")
        if not teachers_ready:
            results['c3_fluid_quality'] = {'error': 'teachers not trained'}
            print(f"    SKIPPED: teachers not trained")
        else:
            try:
                # Each source's MCC weight is cross-validated on its own
                # training rows split by scaffold, not by a random KFold that
                # puts one scaffold on both sides. The C1 probe was moved off
                # a random split for the same reason.
                _teacher_groups = [None] + [
                    pooled_groups[train_idx][train_silo_indices[i + 1]]
                    for i in range(K)]
                pred_cls, pred_prob, mcc_w = predict_fluid_quality_ensemble(
                    teacher_models, cluster_data_train, test_fps, K,
                    groups=_teacher_groups)
                c3 = compute_clf_metrics(y_test, pred_cls, pred_prob)
                c3['mcc_weights'] = convert_numpy(mcc_w)
                results['c3_fluid_quality'] = convert_numpy(c3)
                print(f"    MCC={c3['mcc']:.3f}")
            except Exception as e:
                results['c3_fluid_quality'] = {'error': str(e)}
                print(f"    FAILED: {e}")

        # C4: KIRBy federated feature selection
        print("  C4: KIRBy federated FS...")
        try:
            train_reps = {r: arr[train_idx] for r, arr in combined_reps.items()}
            test_reps = {r: arr[test_idx] for r, arr in combined_reps.items()}

            from kirby.hybrid import apply_feature_selection
            c4_by_budget = {}
            for _budget in PART_A_BUDGETS:
                X_h_train, silo_budgets, silo_scores, silo_fis = \
                    federated_feature_selection(
                        train_reps, y_train, train_silo_indices,
                        K, budget=_budget,
                        allocation_method=_alloc, importance_method=_imp,
                        internal_evaluator=_int_eval,
                        downstream_model=fed_to_kirby_model(fed_model),
                        # Each silo's 20% scoring slice is cut on these.
                        # Without them federated_feature_selection refuses the
                        # slice and every silo takes the flat score, so the
                        # budget comes out equal across silos.
                        groups=pooled_groups[train_idx],
                        task_type='classification',
                        metric_fn=metric_fn, lower_is_better=lower_is_better)

                h_test_parts = []
                for s in range(1, K + 1):
                    fi = silo_fis.get(s)
                    if fi is None:
                        continue
                    try:
                        h_test_parts.append(
                            apply_feature_selection(test_reps, fi))
                    except Exception:
                        continue

                if h_test_parts:
                    X_h_test = np.hstack(h_test_parts)
                    clf = fit_fed_classifier(
                        fed_model, X_h_train, y_train, seed=fed_seed)
                    pred_cls = clf.predict(X_h_test)
                    pred_prob = clf.predict_proba(X_h_test)[:, 1]
                    c4 = compute_clf_metrics(y_test, pred_cls, pred_prob)
                else:
                    c4 = {'error': 'no valid silo selections'}
                c4['budget_requested'] = int(_budget)
                c4['n_features'] = (int(X_h_train.shape[1])
                                    if h_test_parts else 0)
                c4['silo_budgets'] = convert_numpy(silo_budgets)
                c4['silo_scores'] = convert_numpy(silo_scores)
                c4_by_budget[str(_budget)] = convert_numpy(c4)
                mcc_str = f"MCC={c4['mcc']:.3f}" if 'mcc' in c4 else 'N/A'
                print(f"    budget {_budget}: {mcc_str}  "
                      f"{c4['n_features']} features  split={silo_budgets}")

            results['c4_kirby_federated_by_budget'] = c4_by_budget
            # The headline row every summary and cross-run comparison reads.
            results['c4_kirby_federated'] = c4_by_budget[str(HEADLINE_BUDGET)]
        except Exception as e:
            # Surface the traceback (not just str(e)) so a shape/import/NameError
            # in the federated_feature_selection path is debuggable from the
            # expensive run's logs (FED-21).
            results['c4_kirby_federated'] = {
                'error': str(e), 'traceback': traceback.format_exc()}
            print(f"    FAILED: {e}")
            traceback.print_exc()

        # C8/C9: FEDERATED TRAINING, with and without the selection
        #
        # C4 above fits ONE model on every training molecule pooled together,
        # so only the feature ranking is local -- that is federated feature
        # selection, not federated training. These two fit one model per
        # source on that source's OWN molecules and combine only their
        # predictions, which is federated training. Nothing but predictions
        # crosses a source boundary at fit time, and only feature indices,
        # importance weights and one validation score cross at selection time.
        #
        # C9 minus C8 is what the selection is worth to a federated model.
        # C4 minus C9 is what pooling the molecules is worth. C5 below is the
        # fourth corner: pooled molecules, no selection.
        print("  C8/C9: federated fit (one model per source)...")
        try:
            fed_by_budget = {'c8_fed_fit_all_features': {},
                             'c9_fed_fit_selected': {}}

            def _fed_fit_and_score(Xtr_f, Xte_f):
                """One model per source; only predictions are combined.

                DIRTY-LOCAL 2026-09-28. The weights now come from
                predict_fluid_similarity_ensemble, the function C2 uses,
                not the _classification variant. That variant min-max
                scales each source's distances on its own, which is right
                in fed02 where every source has its own features and wrong
                here, where every source measures in the same columns.
                FLuID's annotate_transfer_data puts all sources on one
                scale; scaling each on its own erases which source is
                nearer, the one thing the weight exists to carry. It also
                means C8 is C2 built again (same model, same Morgan bits,
                same sources), apart from the seed of the 20% reference
                sample, and the two now differ by that alone.
                """
                import pandas as pd_local
                teachers, s_df = [None], [None]
                n_trained = 0
                for _s in range(1, K + 1):
                    _idx = train_silo_indices[_s]
                    if len(_idx) < 5 or len(np.unique(y_train[_idx])) < 2:
                        teachers.append((None,))
                        s_df.append(pd_local.DataFrame({'FP': []}))
                        continue
                    _m = fit_fed_classifier(fed_model, Xtr_f[_idx],
                                            y_train[_idx], seed=fed_seed)
                    teachers.append((_m,))
                    s_df.append(pd_local.DataFrame(
                        {'FP': list(Xtr_f[_idx])}))
                    n_trained += 1
                _cls, _prob, _w = predict_fluid_similarity_ensemble(
                    teachers, s_df, Xte_f, K, seed=fed_seed)
                _m = compute_clf_metrics(y_test, _cls, _prob)
                _m['sources_that_trained'] = int(n_trained)
                _m['n_features'] = int(Xtr_f.shape[1])
                return _m

            # C8 needs no selection, so it is the same at every width. Record
            # it once per budget anyway so the two tables line up.
            c7 = _fed_fit_and_score(train_fps, test_fps)
            print(f"    C8 all features: MCC={c7['mcc']:.3f}  "
                  f"{c7['n_features']} features")

            for _budget in PART_A_BUDGETS:
                fed_by_budget['c8_fed_fit_all_features'][str(_budget)] = \
                    convert_numpy(c7)
                _entry = (results.get('c4_kirby_federated_by_budget') or {}
                          ).get(str(_budget), {})
                _fis = None
                try:
                    _, _, _, _fis = federated_feature_selection(
                        train_reps, y_train, train_silo_indices,
                        K, budget=_budget,
                        allocation_method=_alloc, importance_method=_imp,
                        internal_evaluator=_int_eval,
                        downstream_model=fed_to_kirby_model(fed_model),
                        groups=pooled_groups[train_idx],
                        task_type='classification',
                        metric_fn=metric_fn, lower_is_better=lower_is_better)
                except Exception as e:
                    fed_by_budget['c9_fed_fit_selected'][str(_budget)] = {
                        'error': str(e)}
                    continue
                tr_p, te_p = [], []
                for _s in range(1, K + 1):
                    f = _fis.get(_s)
                    if f is None:
                        continue
                    tr_p.append(apply_feature_selection(train_reps, f))
                    te_p.append(apply_feature_selection(test_reps, f))
                if not tr_p:
                    fed_by_budget['c9_fed_fit_selected'][str(_budget)] = {
                        'error': 'no source produced a selection'}
                    continue
                c8 = _fed_fit_and_score(np.hstack(tr_p), np.hstack(te_p))
                c8['budget_requested'] = int(_budget)
                fed_by_budget['c9_fed_fit_selected'][str(_budget)] = \
                    convert_numpy(c8)
                print(f"    budget {_budget}: C9 MCC={c8['mcc']:.3f}  "
                      f"{c8['n_features']} features")

            results['c8_fed_fit_all_features_by_budget'] = \
                fed_by_budget['c8_fed_fit_all_features']
            results['c9_fed_fit_selected_by_budget'] = \
                fed_by_budget['c9_fed_fit_selected']
            results['c8_fed_fit_all_features'] = convert_numpy(c7)
            results['c9_fed_fit_selected'] = \
                fed_by_budget['c9_fed_fit_selected'].get(
                    str(HEADLINE_BUDGET), {})
        except Exception as e:
            results['c9_fed_fit_selected'] = {
                'error': str(e), 'traceback': traceback.format_exc()}
            print(f"    FAILED: {e}")
            traceback.print_exc()

        # C5: Naive pool (unrealistic baseline)
        print("  C5: Naive pool...")
        clf_pool = fit_fed_classifier(
            fed_model, train_fps, y_train, seed=fed_seed)
        pred_pool = clf_pool.predict(test_fps)
        pred_pool_prob = clf_pool.predict_proba(test_fps)[:, 1]
        c5 = compute_clf_metrics(y_test, pred_pool, pred_pool_prob)
        results['c5_naive_pool'] = convert_numpy(c5)
        print(f"    MCC={c5['mcc']:.3f}")

        # C6: KIRBy single-rep control
        print("  C6: KIRBy single-rep (ecfp4 only)...")
        try:
            from kirby.hybrid import create_hybrid, apply_feature_selection
            ecfp4_key = 'ecfp4' if 'ecfp4' in combined_reps else None
            if ecfp4_key is None:
                ecfp4_key = next(iter(combined_reps))
            single_reps = {ecfp4_key: combined_reps[ecfp4_key]}
            train_single = {ecfp4_key: single_reps[ecfp4_key][train_idx]}
            test_single = {ecfp4_key: single_reps[ecfp4_key][test_idx]}
            # Part A pools ChEMBL + Lhasa molecules, but Lhasa-only
            # augmentations don't align with the combined row count.
            # Explicit 'none' avoids cache-key collision.
            # ⛔ THE SCAFFOLD GROUPS, WHICH THIS CALL DID NOT PASS.
            # Split audit kind B, federated_helpers.py:1103. create_hybrid
            # carves an internal holdout to decide which features are admitted
            # and how the budget splits across representations.
            # internal_holdout_split (src/kirby/hybrid.py:5435) uses GroupKFold
            # whenever it is HANDED groups and falls back to a fixed-seed random
            # carve when it is not — and this call handed it none, though
            # pooled_groups has been in scope since fed03:464. So the features
            # were admitted on a carve where a molecule could sit in the
            # internal train half while another molecule of its own scaffold sat
            # in the internal validation half.
            #
            # No score already reported is contaminated: this carve is taken
            # from train_idx only, and test_idx was cut group-aware upstream.
            # What changes is WHICH features C6 admits, so a future run reports
            # different numbers. Fixed 2026-09-18.
            # C6 sweeps the SAME widths as C4. C4 minus C6 is what splitting
            # the budget across three sources buys over spending it all on one
            # representation, and that subtraction only means something if both
            # sides are the same width.
            c6_by_budget = {}
            for _budget in PART_A_BUDGETS:
                X_h_c6, fi_c6 = create_hybrid(
                    train_single, y_train,
                    groups=pooled_groups[train_idx] if pooled_groups is not None else None,
                    allocation_method=_alloc,
                    importance_method=_imp,
                    budget=_budget,
                    internal_evaluator=_int_eval,
                    model=make_fed_classifier(fed_model, seed=fed_seed),
                    metric_fn=metric_fn, lower_is_better=lower_is_better,
                    augmentation_strategy='none')
                X_h_c6_test = apply_feature_selection(test_single, fi_c6)
                clf_c6 = fit_fed_classifier(
                    fed_model, X_h_c6, y_train, seed=fed_seed)
                pred_c6 = clf_c6.predict(X_h_c6_test)
                prob_c6 = clf_c6.predict_proba(X_h_c6_test)[:, 1]
                c6 = compute_clf_metrics(y_test, pred_c6, prob_c6)
                c6['rep_used'] = ecfp4_key
                c6['budget_requested'] = int(_budget)
                c6['n_features'] = int(X_h_c6.shape[1])
                c6_by_budget[str(_budget)] = convert_numpy(c6)
                print(f"    budget {_budget}: MCC={c6['mcc']:.3f}  "
                      f"{c6['n_features']} features")
            results['c6_kirby_single_rep_by_budget'] = c6_by_budget
            results['c6_kirby_single_rep'] = c6_by_budget[str(HEADLINE_BUDGET)]
        except Exception as e:
            results['c6_kirby_single_rep'] = {'error': str(e)}
            print(f"    FAILED: {e}")

        # C7: FLuID multi-rep ensemble
        print("  C7: FLuID multi-rep ensemble...")
        try:
            rep_teachers = []
            rep_names_list = sorted(combined_reps.keys())
            for rname in rep_names_list:
                X_tr = combined_reps[rname][train_idx]
                X_te = combined_reps[rname][test_idx]
                clf_r = fit_fed_classifier(
                    fed_model, X_tr, y_train, seed=fed_seed)
                rep_teachers.append((clf_r, rname, X_tr, X_te))

            from sklearn.model_selection import (cross_val_predict, KFold,
                                                 GroupKFold)
            from sklearn.metrics import matthews_corrcoef
            # Default cv=3 is KFold(shuffle=False); on silo-concatenated
            # rows this puts fold 0 entirely in one silo. Use shuffled.
            #
            # ⛔ GROUP-AWARE WHEN THE GROUPS EXIST. Split audit kind B, added
            # 2026-09-18. These folds decide each representation's weight, and a
            # shuffled KFold puts two molecules of one scaffold on opposite
            # sides of them — so a representation that memorises scaffolds
            # scored better here than it deserves and took a larger share.
            # GroupKFold needs at least as many distinct groups as splits;
            # below that the shuffled KFold stands, which is still better than
            # the unshuffled default for the silo reason above.
            _g_c7 = (pooled_groups[train_idx]
                     if pooled_groups is not None else None)
            if _g_c7 is not None and len(np.unique(_g_c7)) >= 3:
                cv_c7 = GroupKFold(n_splits=3)
                _cv_groups_c7 = _g_c7
            else:
                cv_c7 = KFold(n_splits=3, shuffle=True, random_state=42)
                _cv_groups_c7 = None
            weights = []
            for clf_r, rname, X_tr, _ in rep_teachers:
                try:
                    cv_pred = cross_val_predict(
                        RandomForestClassifier(
                            n_estimators=100, max_depth=15,
                            class_weight='balanced', random_state=0),
                        X_tr, y_train, cv=cv_c7, groups=_cv_groups_c7)
                    w = max(matthews_corrcoef(y_train, cv_pred), 0.01)
                except Exception:
                    w = 0.01
                weights.append(w)
            weights = np.array(weights)
            weights /= weights.sum()

            ensemble_probs = np.zeros(len(test_idx))
            for i, (clf_r, rname, _, X_te) in enumerate(rep_teachers):
                probs = clf_r.predict_proba(X_te)
                if probs.shape[1] == 2:
                    ensemble_probs += weights[i] * probs[:, 1]
                else:
                    ensemble_probs += weights[i] * probs.max(axis=1)
            pred_c7 = (ensemble_probs >= 0.5).astype(int)
            c7 = compute_clf_metrics(y_test, pred_c7, ensemble_probs)
            c7['n_rep_teachers'] = len(rep_teachers)
            c7['teacher_weights'] = convert_numpy(
                {rname: float(w) for (_, rname, _, _), w
                 in zip(rep_teachers, weights)})
            results['c7_fluid_multi_rep'] = convert_numpy(c7)
            print(f"    MCC={c7['mcc']:.3f}  ({len(rep_teachers)} teachers)")
        except Exception as e:
            results['c7_fluid_multi_rep'] = {'error': str(e)}
            print(f"    FAILED: {e}")

        per_fold_results.append(results)

    # ── Aggregate across folds ──
    def _agg_mcc(cond_key, fold_results):
        vals = []
        for fr in fold_results:
            # Both c1 rows read the same per-fold entry; only the pick differs.
            _lookup = ('c1_single_silo'
                       if cond_key == 'c1_single_silo_ceiling' else cond_key)
            r = fr.get(_lookup, {})
            if cond_key in ('c1_single_silo', 'c1_single_silo_ceiling'):
                # Which pick this row averages: the honest one for
                # 'c1_single_silo', the test-fold one for the ceiling row.
                key = ('train_picked' if cond_key == 'c1_single_silo'
                       else 'test_picked_ceiling')
                bs = r.get(key)
                if bs is None:
                    continue
                rr = r.get('per_silo', {}).get(bs, {})
                if 'mcc' in rr:
                    vals.append(rr['mcc'])
            else:
                if 'mcc' in r:
                    vals.append(r['mcc'])
        return vals

    aggregated = {}
    cond_keys = [
        'c1_single_silo', 'c1_single_silo_ceiling',
        'c2_fluid_similarity', 'c3_fluid_quality',
        'c4_kirby_federated', 'c5_naive_pool', 'c6_kirby_single_rep',
        'c7_fluid_multi_rep',
        # the federated-FIT pair: one model per source, predictions combined
        'c8_fed_fit_all_features', 'c9_fed_fit_selected',
    ]
    for ck in cond_keys:
        vals = _agg_mcc(ck, per_fold_results)
        aggregated[ck] = {
            'mcc': float(np.mean(vals)) if vals else float('nan'),
            'mcc_mean': float(np.mean(vals)) if vals else float('nan'),
            'mcc_std': float(np.std(vals)) if vals else float('nan'),
            'mcc_per_fold': [float(v) for v in vals],
            'n_folds_completed': len(vals),
            'per_fold': [fr.get(ck, {}) for fr in per_fold_results],
        }

    # Summary
    print(f"\n  {'Condition':<30} {'MCC (mean ± std)':>20}")
    for key, name in [
        ('c1_single_silo', 'C1: Best silo, picked on train'),
        ('c1_single_silo_ceiling', 'C1 CEILING: best silo on test'),
        ('c2_fluid_similarity', 'C2: FLuID similarity'),
        ('c3_fluid_quality', 'C3: FLuID quality'),
        ('c4_kirby_federated', 'C4: KIRBy federated'),
        ('c5_naive_pool', 'C5: Naive pool'),
        ('c6_kirby_single_rep', 'C6: KIRBy single-rep'),
        ('c7_fluid_multi_rep', 'C7: FLuID multi-rep'),
        ('c8_fed_fit_all_features', 'C8: federated fit, Morgan'),
        ('c9_fed_fit_selected', 'C9: federated fit, selected'),
    ]:
        a = aggregated.get(key, {})
        m = a.get('mcc_mean', float('nan'))
        s = a.get('mcc_std', float('nan'))
        line = (f"{m:.3f} ± {s:.3f}" if not np.isnan(m) else 'N/A')
        print(f"  {name:<30} {line:>20}")

    return {
        'experiment': 'fed03_source_based_herg',
        'dataset': dataset_name,
        'config': {
            'silos': silo_names,
            'n_total': n_total,
            'dedup_stats': dedup_stats,
            'pki_threshold': PKI_THRESHOLD,
            'pki_threshold_note': PKI_THRESHOLD_NOTE,
            'n_folds': n_folds,
            'recipe': _recipe,
            'representations': {k: int(v.shape[1])
                                for k, v in combined_reps.items()},
        },
        'cross_curator_disagreement': convert_numpy(_cross),
        **aggregated,
    }


# =============================================================================
# PART B: ASYMMETRIC REP SILOS
# =============================================================================

def _run_asymmetric_reps(data, config, output_dir, dataset_name,
                         fed_model='rf', fed_seed=0):
    """Strategy 4: Each org contributes different representation types.

    Silo A (structural): ecfp4, graph_kernel, atom_pair
    Silo B (physicochemical): mordred, pdv
    Silo C (learned): molformer, chemberta
    """
    from kirby.hybrid import create_hybrid, apply_feature_selection, apply_augmentation_selection

    print("\n" + "=" * 70)
    print("FED03 PART B: ASYMMETRIC REP SILOS (Strategy 4)")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')

    # Get optimal hybrid config from precomputed search
    augs = data.get('augmentations')
    metric_name_b, metric_fn_b, lower_is_better_b = get_domain_metric(data)
    try:
        _search, _meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model=fed_to_kirby_model(fed_model), budget=100,
            augmentations=augs,
            metric=metric_name_b,
            task_type=data['task_type'],
            domain=data.get('domain', 'molecular'),
            dataset_name=dataset_name,
            data=data,
            **smoke_search_kwargs(config),
        )
        _bp = _meta['best_config']['blueprint'] if _meta else None
    except Exception:
        _bp = None
    # 'capped' blueprints are tuned for the full top-tier rep set; using them
    # on Part B's silo subset (a small asymmetric slice) is a recipe
    # mismatch. Force fallback to non-capped defaults in that case.
    if _bp is not None and getattr(_bp, 'augmentation_strategy', None) == 'capped':
        warnings.warn(
            "Part B: ignoring 'capped' blueprint (tuned for top-tier rep set, "
            "not asymmetric silo subset). Falling back to uncapped/none.")
        _bp = None
    aug_cond = _bp.augmentation_strategy if _bp else ('uncapped' if augs else 'none')
    _alloc = _bp.allocation_method if _bp else 'performance_weighted'
    _imp = _bp.importance_method if _bp else 'treeshap'
    _int_eval = _bp.internal_evaluator if _bp else 'match'

    # Sort THIS dataset's own representations into the three families.
    silo_reps, _unclassified = assign_rep_families(list(reps.keys()))
    if _unclassified:
        warnings.warn(
            f"Part B: {len(_unclassified)} representation(s) in "
            f"{dataset_name}'s pool match no family and are not in any silo: "
            f"{_unclassified}. Add them to REP_FAMILIES in "
            f"fed03_synthetic_private.py.")
    for _family in REP_FAMILIES:
        if _family not in silo_reps:
            warnings.warn(
                f"Part B: {dataset_name}'s pool holds no {_family} "
                f"representation, so that silo does not exist in this run.")

    K = len(silo_reps)
    silo_names = sorted(silo_reps.keys())
    print(f"  {K} silos resolved:")
    for sname in silo_names:
        rep_list = silo_reps[sname]
        dims = [reps[r].shape[1] for r in rep_list]
        print(f"    {sname}: {rep_list} ({sum(dims)}D total)")

    # 5-fold scaffold-group-aware CV (per feedback_no_random_splits.md).
    n_samples = len(labels)
    if groups is None:
        raise RuntimeError(
            "Part B 5-fold CV requires groups (scaffold-aware); groups=None.")
    n_unique_groups = len(np.unique(groups))
    if n_unique_groups < 5:
        raise RuntimeError(
            f"Part B 5-fold CV requires >= 5 unique groups; "
            f"got {n_unique_groups}.")

    # Use the dataset's declared task_type (FED-6) — the cardinality heuristic
    # (len(unique)<=10) misroutes low-cardinality regression targets into the
    # classification path. Fall back to the heuristic only if task_type absent.
    _task_type_b = data.get('task_type')
    if _task_type_b in ('classification', 'regression'):
        is_clf = (_task_type_b == 'classification')
    else:
        is_clf = len(np.unique(labels)) <= 10
    metrics_fn = compute_clf_metrics if is_clf else compute_reg_metrics

    n_folds = smoke_n_folds(5)   # 2 under KIRBY_SMOKE_N, else 5
    folds = list(cv_split(groups, n_folds, np.zeros(n_samples), labels,
                          seed=fed_seed))
    print(f"  {n_folds}-fold CV over {n_unique_groups} groups "
          f"({'clf' if is_clf else 'reg'})")

    per_fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\n── Fold {fold_idx + 1}/{n_folds} "
              f"({len(train_idx)} train / {len(test_idx)} test) ──")
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        train_reps = {r: arr[train_idx] for r, arr in reps.items()}
        test_reps = {r: arr[test_idx] for r, arr in reps.items()}

        results = {}

        # C1: Best single silo
        print("  C1: Single silo baselines...")
        single_silo = {}
        for sname in silo_names:
            rep_list = silo_reps[sname]
            X_tr = np.hstack([train_reps[r] for r in rep_list])
            X_te = np.hstack([test_reps[r] for r in rep_list])

            model = fit_fed_model(
                fed_model, X_tr, y_train, seed=fed_seed,
                task_type=('classification' if is_clf else 'regression'))
            y_pred = model.predict(X_te)
            if is_clf:
                y_prob = model.predict_proba(X_te)[:, 1]
                single_silo[sname] = metrics_fn(y_test, y_pred, y_prob)
            else:
                single_silo[sname] = metrics_fn(y_test, y_pred)
            single_silo[sname]['n_features'] = int(X_tr.shape[1])

            primary = single_silo[sname].get(
                'mcc', single_silo[sname].get('rmse'))
            print(f"    {sname}: {primary:.4f}  ({X_tr.shape[1]}D)")

        # Which family is 'best' is picked two ways and both are kept.
        # test_picked_ceiling reads the fold's held-out molecules, so it needs
        # the answers; train_picked ranks the families on rows held out inside
        # the training molecules and is the one C2's gain is read against.
        def _probe_family(sname):
            from sklearn.model_selection import (
                GroupShuffleSplit as _GroupShuffleSplit)
            rep_list = silo_reps[sname]
            X_all = np.hstack([train_reps[r] for r in rep_list])
            n = len(y_train)
            if n < 20:
                return None
            g_tr = groups[train_idx] if groups is not None else None
            if g_tr is not None and len(np.unique(g_tr)) >= 2:
                sp = _GroupShuffleSplit(n_splits=1, test_size=0.25,
                                        random_state=fed_seed)
                tr_v, va_v = next(sp.split(X_all, y_train, groups=g_tr))
            else:
                return None
            m = fit_fed_model(
                fed_model, X_all[tr_v], y_train[tr_v], seed=fed_seed,
                task_type=('classification' if is_clf else 'regression'))
            mm = metrics_fn(y_train[va_v], m.predict(X_all[va_v]))
            # rank_silos_train_only always maximises, so an error is negated.
            return mm['mcc'] if is_clf else -mm['rmse']

        _valid = [s for s in single_silo
                  if ('mcc' if is_clf else 'rmse') in single_silo[s]]
        if is_clf:
            _ceiling = (max(_valid, key=lambda s: single_silo[s]['mcc'])
                        if _valid else None)
        else:
            _ceiling = (min(_valid, key=lambda s: single_silo[s]['rmse'])
                        if _valid else None)
        _train_pick, _inner = rank_silos_train_only(
            _probe_family, silo_names, seed=fed_seed)
        print(f"    best on the held-out fold (ceiling): {_ceiling}")
        print(f"    best on the training rows (honest):  {_train_pick}")

        results['c1_single_silo'] = convert_numpy({
            'per_silo': single_silo,
            'test_picked_ceiling': _ceiling,
            'train_picked': _train_pick,
            'train_picked_inner_scores': _inner,
        })

        # Part B sweeps the same widths Part A does. Both conditions run
        # at every budget: C2 takes the whole budget in one selection over
        # all three families, C3 splits it across them. Comparing them at
        # one width only said nothing about whether either scales.
        c2_by_budget, c3_by_budget = {}, {}
        for _pb_budget in PART_B_BUDGETS:
            # C2: KIRBy hybrid across all silo reps
            print("  C2: KIRBy hybrid (all reps)...")
            try:
                all_silo_reps_tr = {}
                all_silo_reps_te = {}
                for sname in silo_names:
                    for r in silo_reps[sname]:
                        all_silo_reps_tr[r] = train_reps[r]
                        all_silo_reps_te[r] = test_reps[r]
                train_augs_c2 = ({k: v[train_idx] for k, v in augs.items()}
                                 if augs else None)
                test_augs_c2 = ({k: v[test_idx] for k, v in augs.items()}
                                if augs else None)

                _c2_model = (
                    make_fed_classifier(fed_model, seed=fed_seed) if is_clf
                    else make_fed_regressor(fed_model, seed=fed_seed))
                X_h_tr, fi = create_hybrid(
                    all_silo_reps_tr, y_train,
                    # Without groups the internal evaluation splits at random, so
                    # a scaffold can sit on both sides of it
                    # (feedback_no_random_splits.md). C1's own fits are
                    # scaffold-grouped; this was not.
                    groups=(groups[train_idx] if groups is not None else None),
                    allocation_method=_alloc,
                    importance_method=_imp,
                    internal_evaluator=_int_eval,
                    budget=_pb_budget,
                    model=_c2_model,
                    metric_fn=metric_fn_b, lower_is_better=lower_is_better_b,
                    augmentations=train_augs_c2,
                    augmentation_strategy=aug_cond)
                X_h_te = apply_feature_selection(all_silo_reps_te, fi)
                _aug_info = (fi.get('augmentation_info')
                             if isinstance(fi, dict) else None)
                if _aug_info and test_augs_c2:
                    _X_te_aug = apply_augmentation_selection(
                        test_augs_c2, _aug_info)
                    if _X_te_aug is not None:
                        X_h_te = np.hstack([X_h_te, _X_te_aug])

                model = fit_fed_model(
                    fed_model, X_h_tr, y_train, seed=fed_seed,
                    task_type=('classification' if is_clf else 'regression'))
                y_pred = model.predict(X_h_te)
                if is_clf:
                    c2 = metrics_fn(y_test, y_pred,
                                    model.predict_proba(X_h_te)[:, 1])
                else:
                    c2 = metrics_fn(y_test, y_pred)
                c2['allocation'] = convert_numpy(fi.get('allocation', {}))
                c2['n_features'] = int(X_h_tr.shape[1])
                results['c2_kirby_hybrid'] = convert_numpy(c2)
                primary = c2.get('mcc', c2.get('rmse'))
                print(f"    {primary:.4f}  ({c2['n_features']}D)")
            except Exception as e:
                results['c2_kirby_hybrid'] = {
                    'error': str(e), 'traceback': traceback.format_exc()}
                print(f"    FAILED: {e}")
                traceback.print_exc()

            # C3: Federated feature selection (privacy-preserving)
            print("  C3: Federated FS (per-silo importance)...")
            try:
                from kirby.hybrid import create_hybrid as _ch
                from sklearn.model_selection import (
                    GroupShuffleSplit as _GSS, ShuffleSplit as _SS)
                from experiments._shared import get_model_factories

                silo_task_type = 'classification' if is_clf else 'regression'
                train_groups = groups[train_idx] if groups is not None else None

                # Resolve 'match' → concrete evaluator ('rf' default) BEFORE the
                # create_hybrid call. Without this, _int_eval='match' + no model=
                # arg makes create_hybrid silently fall back to 'ridge'
                # (hybrid.py:3354-3355), which is forbidden (feedback_no_ridge.md).
                # DIRTY-LOCAL 2026-09-28: was _resolve_match(_int_eval, 'rf'),
                # so under 'match' C3 used an rf evaluator for every fed_model
                # while C2's create_hybrid matched the downstream model
                # (xgboost for 'xgb'). Both now resolve to the same family.
                _resolved_eval = _resolve_match(
                    _int_eval, fed_to_kirby_model(fed_model))
                _silo_model_c3 = (
                    make_fed_classifier(fed_model, seed=fed_seed) if is_clf
                    else make_fed_regressor(fed_model, seed=fed_seed))

                # Each family selects its features and is scored, in that order,
                # on two disjoint parts of the training rows. It used to select
                # from every training row and then score on a split of those same
                # rows, so every feature had been chosen with the scoring rows'
                # labels in view and the score was not a held-out score.
                #
                # Budget: PART_B_BUDGET split across the families, so C3's total
                # matches C2's. Each family used to take the full default of 100
                # on its own, which made C3 a wider feature set than C2 and the
                # comparison a comparison of two sizes.
                silo_fis = {}
                silo_scores = {}
                _n_fam = max(1, len(silo_names))
                # DIRTY-LOCAL 2026-09-28: was PART_B_BUDGET (fixed 100) inside
                # the _pb_budget sweep, so C3 was the same 100-feature build
                # at every width while C2 swept. C3's total now matches C2's.
                _per_family_budget = max(5, _pb_budget // _n_fam)
                for i, sname in enumerate(silo_names, 1):
                    rep_list = silo_reps[sname]
                    silo_rep_tr = {r: train_reps[r] for r in rep_list}
                    try:
                        if (train_groups is not None
                                and len(np.unique(train_groups)) >= 2):
                            splitter = _GSS(
                                n_splits=1, test_size=0.2,
                                random_state=42 + i + fold_idx)
                            tr, va = next(splitter.split(
                                np.zeros((len(y_train), 1)), y_train,
                                groups=train_groups))
                        else:
                            splitter = _SS(
                                n_splits=1, test_size=0.2,
                                random_state=42 + i + fold_idx)
                            tr, va = next(splitter.split(
                                np.zeros((len(y_train), 1))))

                        # Probe: select on the training part only, score on the
                        # part that selection never saw.
                        _, fi_probe = _ch(
                            {r: arr[tr] for r, arr in silo_rep_tr.items()},
                            y_train[tr],
                            groups=(train_groups[tr]
                                    if train_groups is not None else None),
                            allocation_method=_alloc,
                            importance_method=_imp,
                            internal_evaluator=_resolved_eval,
                            budget=min(_per_family_budget, len(tr)),
                            model=_silo_model_c3,
                            metric_fn=metric_fn_b,
                            lower_is_better=lower_is_better_b)
                        X_probe_tr = apply_feature_selection(
                            {r: arr[tr] for r, arr in silo_rep_tr.items()},
                            fi_probe)
                        X_probe_va = apply_feature_selection(
                            {r: arr[va] for r, arr in silo_rep_tr.items()},
                            fi_probe)
                        eval_factory = get_model_factories(
                            [_resolved_eval], task_type=silo_task_type,
                            n_samples=len(tr)).get(_resolved_eval)
                        if eval_factory is None or X_probe_tr.shape[1] == 0:
                            silo_scores[i] = 0.0
                        else:
                            evaluator = eval_factory()
                            evaluator.fit(X_probe_tr, y_train[tr])
                            silo_scores[i] = max(0.0, float(
                                evaluator.score(X_probe_va, y_train[va])))

                        # The selection that is actually used: every training row,
                        # at this family's share of the budget.
                        _, fi_s = _ch(
                            silo_rep_tr, y_train,
                            groups=train_groups,
                            allocation_method=_alloc,
                            importance_method=_imp,
                            internal_evaluator=_resolved_eval,
                            budget=_per_family_budget,
                            model=_silo_model_c3,
                            # DIRTY-LOCAL 2026-09-28: C2 passes the dataset's
                            # metric and C3 did not, so C3 split each family's
                            # share across its representations by r2 while C2
                            # split by the dataset metric (rmse here).
                            metric_fn=metric_fn_b,
                            lower_is_better=lower_is_better_b)
                        silo_fis[i] = fi_s
                    except Exception:
                        silo_fis[i] = None
                        silo_scores[i] = 0.0

                h_parts_tr = []
                h_parts_te = []
                for i, sname in enumerate(silo_names, 1):
                    fi = silo_fis.get(i)
                    if fi is None:
                        continue
                    rep_list = silo_reps[sname]
                    silo_rep_tr = {r: train_reps[r] for r in rep_list}
                    silo_rep_te = {r: test_reps[r] for r in rep_list}
                    try:
                        h_parts_tr.append(
                            apply_feature_selection(silo_rep_tr, fi))
                        h_parts_te.append(
                            apply_feature_selection(silo_rep_te, fi))
                    except Exception:
                        continue

                if h_parts_tr:
                    X_fed_tr = np.hstack(h_parts_tr)
                    X_fed_te = np.hstack(h_parts_te)
                    model = fit_fed_model(
                        fed_model, X_fed_tr, y_train, seed=fed_seed,
                        task_type=silo_task_type)
                    y_pred = model.predict(X_fed_te)
                    if is_clf:
                        c3 = metrics_fn(
                            y_test, y_pred,
                            model.predict_proba(X_fed_te)[:, 1])
                    else:
                        c3 = metrics_fn(y_test, y_pred)
                    c3['n_features'] = int(X_fed_tr.shape[1])
                else:
                    c3 = {'error': 'no valid silo selections'}
                c3['silo_scores'] = convert_numpy(silo_scores)
                # These scores are recorded only; the budget split above is
                # equal across families and does not read them.
                c3['silo_score_metric'] = (
                    'evaluator.score clipped at 0: accuracy (classification) '
                    'or r2 (regression); not used to allocate')
                c3['budget_total'] = int(_pb_budget)
                c3['budget_per_family'] = _per_family_budget
                results['c3_federated_fs'] = convert_numpy(c3)
                primary = c3.get('mcc', c3.get('rmse', float('nan')))
                # The width is printed beside the score so C2 and C3 can be seen
                # to be the same size, which is what makes them comparable.
                print(f"    {primary:.4f}  ({c3.get('n_features', 0)}D)")
            except Exception as e:
                results['c3_federated_fs'] = {
                    'error': str(e), 'traceback': traceback.format_exc()}
                print(f"    FAILED: {e}")
                traceback.print_exc()
            c2_by_budget[str(_pb_budget)] = results.get('c2_kirby_hybrid')
            c3_by_budget[str(_pb_budget)] = results.get('c3_federated_fs')
        results['c2_kirby_hybrid_by_budget'] = c2_by_budget
        results['c3_federated_fs_by_budget'] = c3_by_budget
        # headline rows the summary table reads
        _hb = str(HEADLINE_BUDGET if HEADLINE_BUDGET in PART_B_BUDGETS                  else PART_B_BUDGETS[0])
        results['c2_kirby_hybrid'] = c2_by_budget[_hb]
        results['c3_federated_fs'] = c3_by_budget[_hb]

        per_fold_results.append(results)

    # ── Aggregate across folds ──
    primary_key = 'mcc' if is_clf else 'rmse'

    def _agg_primary(cond_key):
        vals = []
        for fr in per_fold_results:
            _lookup = ('c1_single_silo'
                       if cond_key == 'c1_single_silo_ceiling' else cond_key)
            r = fr.get(_lookup, {})
            if cond_key in ('c1_single_silo', 'c1_single_silo_ceiling'):
                if not r:
                    continue
                key = ('train_picked' if cond_key == 'c1_single_silo'
                       else 'test_picked_ceiling')
                best = r.get(key)
                if best is None:
                    continue
                v = r.get('per_silo', {}).get(best, {}).get(primary_key)
            else:
                v = r.get(primary_key)
            if v is not None and not np.isnan(v):
                vals.append(float(v))
        return vals

    aggregated = {}
    for ck in ['c1_single_silo', 'c1_single_silo_ceiling',
               'c2_kirby_hybrid', 'c3_federated_fs']:
        vals = _agg_primary(ck)
        aggregated[ck] = {
            primary_key: float(np.mean(vals)) if vals else float('nan'),
            f'{primary_key}_mean': float(np.mean(vals)) if vals else float('nan'),
            f'{primary_key}_std': float(np.std(vals)) if vals else float('nan'),
            f'{primary_key}_per_fold': vals,
            'n_folds_completed': len(vals),
            'per_fold': [fr.get(ck, {}) for fr in per_fold_results],
        }

    # Summary
    print(f"\n  {'Condition':<35} {'Primary (mean ± std)':>22}")
    for key, name in [
        ('c1_single_silo', 'C1: Best family, picked on train'),
        ('c1_single_silo_ceiling', 'C1 CEILING: best family on test'),
        ('c2_kirby_hybrid', 'C2: KIRBy hybrid (all)'),
        ('c3_federated_fs', 'C3: Federated FS'),
    ]:
        a = aggregated.get(key, {})
        m = a.get(f'{primary_key}_mean', float('nan'))
        s = a.get(f'{primary_key}_std', float('nan'))
        line = (f"{m:.4f} ± {s:.4f}" if not np.isnan(m) else 'N/A')
        print(f"  {name:<35} {line:>22}")

    return {
        'experiment': 'fed03_asymmetric_reps',
        'dataset': dataset_name,
        'config': {
            'silo_reps': {s: silo_reps[s] for s in silo_names},
            'n_folds': n_folds,
            'primary_metric': primary_key,
            'is_classification': is_clf,
        },
        **aggregated,
    }


# =============================================================================
# PART C: ANTIBODY FEDERATED (F6)
# =============================================================================

# PART C — DELETED 2026-09-12 (user decision).
#
# Part C ran three conditions on four antibody sets. Two of them —
# scoring each set on its own held-out rows, and transferring one set's
# selected feature list onto another — are fed05's C1 and C2, and fed05
# scores them the same way. The third pooled measured binding free energy
# with FoldX-computed binding free energy, which needs graphinity_foldx;
# that dataset was cut on 2026-07-09 because its labels are FoldX output
# rather than measurement, and it is not coming back. Two of the four silo
# names Part C asked for were not in the antibody dataset table at all, so
# every run loaded two of four and the pooled condition reported an error
# every time.
#
# Nothing was moved: fed05 already computes what survived. The antibody
# federated questions live in fed05 alone now.

# =============================================================================
# DISPATCHER
# =============================================================================

def _run_one(data, config, output_dir, dataset_name, **kwargs):
    """Fed03: Synthetic private datasets.

    Dispatches to sub-experiment based on dataset_name and kwargs['strategy']:
        strategy='source'     → Part A (source-based hERG, hERG only)
        strategy='asymmetric' → Part B (asymmetric rep silos, any molecular)
        default               → auto-detect from dataset_name
    """
    strategy = kwargs.get('strategy', None)
    fed_model = config.get('fed_model', 'rf')
    fed_seed = config.get('fed_seed', 0)

    if strategy == 'source' or (
            strategy is None and dataset_name in ('herg_ki', 'herg', 'lhasa_herg')):
        return _run_herg_source_based(
            data, config, output_dir, dataset_name,
            fed_model=fed_model, fed_seed=fed_seed)
    elif strategy == 'asymmetric':
        return _run_asymmetric_reps(
            data, config, output_dir, dataset_name,
            fed_model=fed_model, fed_seed=fed_seed)
    elif strategy == 'antibody' or dataset_name in ANTIBODY_DATASETS:
        raise ValueError(
            f"fed03 has no antibody part. Part C was deleted on 2026-09-12 "
            f"because fed05 already answers its two working questions and its "
            f"third needed a dataset that was cut. Run fed05 instead: "
            f"dataset_name={dataset_name!r}, strategy={strategy!r}.")
    else:
        # Default for non-hERG molecular datasets: asymmetric reps
        return _run_asymmetric_reps(
            data, config, output_dir, dataset_name,
            fed_model=fed_model, fed_seed=fed_seed)


# ── One (FLuID model, seed) call saved as it finishes (2026-09-25) ──────────
# federated_helpers.run_all calls run() once per (fed_model, fed_seed) -- nine
# calls -- and wrote fed03_synthetic_private/<dataset>.json only after the ninth. A kill or
# an exception in any call lost every call before it. Each call's return value
# is now written to fed03_synthetic_private/<dataset>.partial.jsonl the moment it returns
# and read back on the next attempt (_checkpoint.resumable_run), keyed on the
# model, the seed and any keyword argument, under an identity that includes the
# input fingerprint, the plain config and the KIRBY_* settings. The returned
# dict is decoded to the same objects, so aggregate_seed_runs and the final
# JSON are unchanged. --force-rerun / KIRBY_FORCE_RERUN=1 discards the file.
def run(data, config, output_dir, dataset_name, **kwargs):
    from experiments._checkpoint import resumable_run
    _key = {'fed_model': str(config.get('fed_model', 'rf')),
            'fed_seed': int(config.get('fed_seed', 0)),
            **{k: v for k, v in kwargs.items()}}
    return resumable_run(
        'fed03_synthetic_private',
        lambda: _run_one(data, config, output_dir, dataset_name, **kwargs),
        data, config, output_dir, dataset_name, key=_key)



if __name__ == '__main__':
    import argparse as _ap
    parser = _ap.ArgumentParser(description='Fed03: Synthetic Private Datasets')
    parser.add_argument('--dataset', type=str, default='herg_ki',
                        help='Dataset name (herg_ki, logd, caco2, ddg_combined, etc.)')
    parser.add_argument('--strategy', type=str, default=None,
                        choices=['source', 'asymmetric'],
                        help='Sub-experiment: source (Part A), '
                             'asymmetric (Part B). The antibody part was '
                             'deleted on 2026-09-12 — use fed05.')
    parser.add_argument('--output', type=str, default='tests/results/federated')
    parser.add_argument('--reps', type=str, default='all')
    args = parser.parse_args()

    data = load_data(args.dataset, args.reps)

    results = run(data, {}, args.output, args.dataset, strategy=args.strategy)
    save_experiment_results(
        results, args.output, 'fed03_synthetic_private', args.dataset)
