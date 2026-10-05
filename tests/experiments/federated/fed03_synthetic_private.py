#!/usr/bin/env python3
"""Fed03: Synthetic Private Datasets.

Two sub-experiments, chosen by which dataset is passed:
    Part A (herg_ki, herg, lhasa_herg): Source-based hERG — Lhasa's curated
        data and KIRBy's ChEMBL data as real federated silos with genuine
        source heterogeneity.
    Part B (any other molecular dataset): Asymmetric representation silos —
        each organisation holds a different family of representations.

Since 2026-10-05 (handoff 2.1 R1-R6, 2.4) every hybrid an outer fold uses
comes from a search on that fold's training rows, on the dataset's decided
grid, for the run's own model: its 2 best base representations and, at each
decided budget, its 2 best hybrids with distinct selected features. Nothing is
read from a whole-dataset search. The outer folds come from one setting
(config['n_splits']) through federated_helpers.fed_outer_folds, seed 42.

There was a Part C on antibody data. It was deleted on 2026-09-12; fed05
answers the antibody federated questions. Passing an antibody dataset here
raises and says so.

Usage:
    python fed03_synthetic_private.py --dataset lhasa_herg
    python fed03_synthetic_private.py --dataset caco2
    python fed03_synthetic_private.py --dataset logd --strategy asymmetric
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import traceback
import warnings
import numpy as np

from experiments._shared import as_rep_dtype, convert_numpy, get_domain_metric
from experiments.federated.federated_helpers import (
    compute_clf_metrics, compute_reg_metrics,
    predict_fluid_similarity_ensemble,
    predict_fluid_quality_ensemble,
    federated_feature_selection,
    CLASSIFICATION_CUTOFFS,
    binarise_labels,
    rank_silos_train_only,
    _resolve_match,
    smoke_n_folds,
    fit_fed_classifier, fit_fed_model,
    predict_fluid_similarity_ensemble_classification,
    fed_to_kirby_model, make_fed_classifier, make_fed_regressor,
    fed_outer_folds, fed_fold_selection, recipe_reps, FED_FOLD_SEED,
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



# ── Feature budgets (2026-10-05) ────────────────────────────────────────────
# Both parts sweep the dataset's decided general-search budgets (DECISIONS.md
# Step 2, through federated_budgets(dataset_name)): lhasa_herg 100-2000
# features, caco2 and logd 500-2000. Until 2026-10-05 this was a module
# constant, 50/100/250/500/1000 on every dataset, computed at import; the
# import then raised UndecidedSearch because the default tuple was handed to
# federated_budgets as its dataset name. It is now read inside each run.
#
# Until 2026-09-23 Part A's C4 and C6 passed no budget at all and took
# create_hybrid's default of 100, so the result this experiment is best known
# for — 0.5504 MCC against 0.4005 for the best single source — was measured at
# one width and nobody knew whether it held at any other. A budget is a swept
# axis like any other and the server run covers the range.
#
# KIRBY_FED03_BUDGETS overrides it as a comma-separated list, for a local
# debugging run. A budget outside the decided list finds no hybrid in the
# per-fold search and is written out as not filled.
def _fed03_budgets(dataset_name):
    raw = os.environ.get('KIRBY_FED03_BUDGETS', '').strip()
    if raw:
        return [int(x) for x in raw.split(',') if x.strip()]
    from experiments.federated.federated_helpers import federated_budgets
    return [int(b) for b in federated_budgets(dataset_name)]


def _headline_budget(budgets):
    """The width every summary table quotes: 100 features where the sweep has
    it, else the smallest budget swept (500 on caco2 and logd)."""
    return 100 if 100 in budgets else sorted(budgets)[0]


def _n_outer_folds(config):
    """The outer fold count, from ONE setting: config['n_splits'] (run_all's
    federated config, DEFAULT_N_FOLDS unless overridden), cut only by the
    test-only KIRBY_SMOKE_N / KIRBY_FED_FOLDS switches in smoke_n_folds.
    Until 2026-10-05 both parts hardcoded smoke_n_folds(5). 1 gives a single
    grouped 80/20 split (nested_cv_outer_folds)."""
    from kirby.find_best_hybrid import DEFAULT_N_FOLDS
    n = config.get('n_splits') if isinstance(config, dict) else None
    return smoke_n_folds(int(n if n is not None else DEFAULT_N_FOLDS))


# ── The hybrid set per fold (2026-10-05, handoff 2.1 R4) ────────────────────
# For the model this run trains: per outer fold, its 2 best base
# representations and, at each budget, its 2 best hybrids with distinct
# selected features, all ranked on that fold's training rows. hybrid1 writes
# the keys this file always wrote; hybrid2 writes the same key with
# '__hybrid2' appended. Until 2026-10-05 the set was 2 general + 1 distance
# hybrid read from the whole-dataset search ('general1', 'general2',
# 'distance1'); answer 1 of 2.3 keeps distance hybrids only where an
# experiment measures something with neighbours, and fed03 does not.
_HYBRID_SLOTS = ('hybrid1', 'hybrid2')


def _slot_suffix(slot):
    """'' for hybrid1 (the old key), '__hybrid2' for the second."""
    return '' if slot == 'hybrid1' else f'__{slot}'


# The outer-fold checkpoint and the per-call checkpoint carry this, so a
# partial file written by the whole-dataset version is never read back
# (handoff 2.4 step 11).
FED03_LAYOUT = '2026-10-05 per-fold selection, decided grid'

# Part A's pooled rows (ChEMBL + both Lhasa files) are not the rows of any
# dataset's own per-fold search, so its search is filed under this suffix and
# can never be served from, or overwrite, the lhasa_herg / herg_ki search of
# the same fold number (fold_neighbour_space_recipes' rule for rows that are
# not the dataset's own).
PART_A_SEARCH_SUFFIX = 'fed03a_pooled'

# Part A reports, and searches on, AUROC of the two-class blocker labels.
PART_A_METRIC = 'auc'

# C7's representation weights come from cross-validated predictions inside the
# fold's training rows, on this many scaffold-grouped inner folds. An inner
# setting of C7 alone, not the outer fold count. Named 2026-10-05 (was a
# literal GroupKFold(n_splits=3) with a shuffled KFold fallback).
C7_WEIGHT_FOLDS = 3

# C6's single representation, by name. Never substituted: a run where it is
# missing from the pooled representations records C6 as skipped (2026-10-05).
C6_REP = 'ecfp4'


class _C6Skip(Exception):
    """C6's representation is missing from this run's pooled data."""


# The recipe settings C4 and C9 do NOT apply (2026-10-05). They rebuild each
# per-source selection through federated_feature_selection, which takes the
# allocation method, importance method, internal evaluator and budget and
# nothing else, so these keys of _recipe_kwargs fall back to create_hybrid's
# defaults there. C6 builds through create_hybrid with all of them. So C4 and
# C6 can differ in these settings as well as in the split across sources;
# each C4 entry records which of them the recipe set away from the default.
# Only the settings that act without augmentation are listed (augmentation is
# cut on lhasa_herg), with create_hybrid's default for each.
C4_SETTINGS_NOT_APPLIED = {'step_size': 5,
                           'candidate_ranking': 'residual_correlation',
                           'objective': 'predictive'}


def _c4_settings_not_applied(rkw):
    """{setting: the recipe's value} for each recipe setting C4/C9 cannot
    pass on that differs from create_hybrid's default. {} means C4 built the
    same recipe C6 did."""
    out = {}
    for k, default in C4_SETTINGS_NOT_APPLIED.items():
        v = rkw.get(k)
        if v is not None and v != default:
            out[k] = v
    return out


# ── The rest of each recipe (2026-10-02) ────────────────────────────────────
# Until 2026-10-02 this file rebuilt each hybrid from four of its recipe's
# settings: allocation method, importance method, internal evaluator and
# augmentation condition. Everything else the search had chosen took
# create_hybrid's defaults, so two hybrids could build identical features.
# The owner's decision: build with the whole recipe. The names and defaults
# below are the ones _shared._create_hybrid_from_blueprint uses, which is how
# the search's own rebuild (build_hybrid_from_entry) calls create_hybrid.
#
# Privacy check, done for this file: every one of these settings is a number
# or a name chosen by the same search the four settings already came from.
# None carries a row, a label or a feature, so passing them on shares nothing
# across silos that the four settings did not already share.
def _recipe_kwargs(bp, objective=None):
    """create_hybrid keyword arguments for every recipe setting beyond the
    allocation method, importance method, internal evaluator, augmentation
    condition and budget. {} when there is no recipe."""
    if bp is None:
        return {}
    kw = {
        'step_size': int(getattr(bp, 'step_size', 5) or 5),
        'augmentation_margin': float(
            getattr(bp, 'augmentation_margin', 0.0) or 0.0),
        'augmentation_fraction': getattr(bp, 'augmentation_fraction', None),
        'candidate_ranking': getattr(bp, 'candidate_ranking',
                                     'residual_correlation'),
        'augmentation_threshold': getattr(bp, 'augmentation_threshold', None),
        'augmentation_budget': int(
            getattr(bp, 'augmentation_budget', 20) or 20),
        'aug_max_candidates': int(getattr(bp, 'aug_max_candidates', 50) or 50),
        'aug_per_source_budget': getattr(bp, 'aug_per_source_budget', None),
    }
    if objective is not None:
        kw['objective'] = objective
    return kw


def _build_settings(bp, kirby_model):
    """(allocation, importance, internal evaluator, rest of recipe) for one
    per-fold hybrid. 'match' is resolved to the run's own model, the same way
    for every arm that builds this recipe (2026-10-05): before, Part B's C3
    turned 'match' into rf while C2 handed it to create_hybrid with the run's
    model, so the two arms differed in evaluator as well as in the budget
    split."""
    return (bp.allocation_method, bp.importance_method,
            _resolve_match(bp.internal_evaluator, kirby_model),
            _recipe_kwargs(bp, getattr(bp, 'objective', None)))


def _hybrid_record(h, kirby_model):
    """What the result file says about one chosen hybrid (handoff 2.4 step
    11): its settings, its budget in features, its representation count, its
    augmentation condition and its score on the fold's training rows (the
    search's mean over its inner folds)."""
    bp = h.get('blueprint')
    ie = h.get('internal_evaluator') or getattr(bp, 'internal_evaluator', None)
    return {
        'label': h.get('label'),
        'allocation_method': h.get('allocation_method')
        or getattr(bp, 'allocation_method', None),
        'importance_method': h.get('importance_method')
        or getattr(bp, 'importance_method', None),
        'budget': h.get('budget'),
        'internal_evaluator': ie,
        'internal_evaluator_used': _resolve_match(ie, kirby_model),
        'step_size': h.get('step_size'),
        'rep_count': h.get('rep_count'),
        'representations_searched': list(getattr(bp, 'reps', None) or []),
        'n_features_in_search': h.get('n_features'),
        'aug_condition': h.get('aug_condition', 'none'),
        'train_rows_score': h.get('score_mean'),
        'train_rows_score_std': h.get('score_std'),
    }


def _selection_record(sel, errors, *, dataset_name, fold_i, kirby_model,
                      budgets):
    """The per-fold selection as written to the result file. sel is one
    model's entry of fed_fold_selection (or Part A's equivalent), or None when
    the per-fold search failed; errors are the search's failure records, each
    stamped with dataset, fold and model."""
    errs = []
    for e in errors or []:
        e = dict(e) if isinstance(e, dict) else {'error': str(e)}
        e.update(dataset=dataset_name, fold=int(fold_i), model=kirby_model)
        errs.append(convert_numpy(e))
    rec = {
        'model': kirby_model,
        'selection_scope': 'fold_train_rows',
        'search_failed': sel is None,
        'errors': errs,
    }
    if sel is None:
        return rec
    bd = sel.get('base_detail') or {}
    rec.update({
        'search_metric': sel.get('search_metric'),
        'search_lower_is_better': sel.get('search_lower_is_better'),
        'base': list(sel.get('base') or []),
        'base_scores_train_rows': [
            {'rep': e.get('rep'), 'score': e.get('score')}
            for e in (bd.get('selected') or [])],
        'base_shortfall': bd.get('shortfall_reason'),
        'by_budget': {},
    })
    for b in budgets:
        at = (sel.get('by_budget') or {}).get(int(b))
        if at is None:
            rec['by_budget'][str(b)] = {'hybrids': [],
                                        'note': 'no hybrid at this budget'}
            continue
        rec['by_budget'][str(b)] = {
            'hybrids': [_hybrid_record(h, kirby_model)
                        for h in at.get('hybrids') or []],
            'skipped_identical_features': [
                (e.get('label') if isinstance(e, dict) else str(e))
                for e in at.get('hybrids_skipped_identical_features') or []],
        }
    return convert_numpy(rec)


def _hybrid_at(sel, budget, slot_i):
    """The slot_i-th hybrid (0 or 1) chosen at this budget, or None."""
    if sel is None:
        return None
    at = (sel.get('by_budget') or {}).get(int(budget)) or {}
    hs = at.get('hybrids') or []
    return hs[slot_i] if len(hs) > slot_i else None


def _not_filled(slot, budget, sel):
    """The entry an arm writes when its hybrid does not exist. No fallback
    recipe fills it (R2/R6; the treeshap fixed-settings fallback was removed
    2026-10-05)."""
    why = ('the per-fold search failed; see selection.errors'
           if sel is None else
           f'the per-fold search has fewer than {_HYBRID_SLOTS.index(slot) + 1}'
           f' distinct hybrid(s) at budget {budget}')
    return {'skipped': True, 'reason': f'{slot} not built: {why}',
            'budget_requested': int(budget)}


def _reg_metrics(y_true, y_pred):
    """compute_reg_metrics without its two p-values (no significance tests,
    Adelaide; removed from this file's results 2026-10-05)."""
    m = compute_reg_metrics(y_true, y_pred)
    m.pop('spearman_p', None)
    m.pop('pearson_p', None)
    return m


def _fold_summary(vals, key):
    """{key, key_mean, key_std, key_per_fold, n_folds_completed}. With one
    fold there is no spread: std is None and spread_note says so."""
    vals = [float(v) for v in vals]
    out = {
        key: float(np.mean(vals)) if vals else float('nan'),
        f'{key}_mean': float(np.mean(vals)) if vals else float('nan'),
        f'{key}_per_fold': vals,
        'n_folds_completed': len(vals),
    }
    if len(vals) >= 2:
        out[f'{key}_std'] = float(np.std(vals))
    else:
        out[f'{key}_std'] = None
        out['spread_note'] = ('one fold, no spread' if len(vals) == 1
                              else 'no fold completed')
    return out


def _fmt_summary(a, key, digits=3):
    m = a.get(f'{key}_mean', float('nan'))
    s = a.get(f'{key}_std')
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return 'N/A'
    if s is None:
        return f"{m:.{digits}f} (one fold, no spread)"
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def _fold_checkpoint(data, config, output_dir, dataset_name, part,
                     fed_model, fed_seed):
    """One sidecar unit per outer fold (dirty-run bug 12: results written
    after every fold, not only at the end of the (model, seed) call). The
    identity carries FED03_LAYOUT, so a file from the old layout is moved
    aside, never read back."""
    from experiments._checkpoint import (Checkpoint, experiment_sidecar,
                                         run_identity)
    _cfg = ({k: v for k, v in config.items()
             if k not in ('fed_model', 'fed_seed')}
            if isinstance(config, dict) else {})
    return Checkpoint(
        experiment_sidecar(output_dir or '.', 'fed03_synthetic_private',
                           dataset_name,
                           suffix=f'folds_{part}_{fed_model}_s{fed_seed}'),
        run_identity('fed03_synthetic_private', dataset_name, data, _cfg,
                     layout=FED03_LAYOUT, part=part, fed_model=fed_model,
                     fed_seed=fed_seed))


def _part_a_fold_selection(reps, labels, groups, *, train_idx, fold_i,
                           n_folds, dataset_name, budgets, kirby_model):
    """fed_fold_selection for Part A's pooled rows. Same per-fold search and
    the same selection, but filed under PART_A_SEARCH_SUFFIX (fed_fold_
    selection takes no suffix). The search runs on the two-class blocker
    labels with AUROC. Returns (selection or None, failure records)."""
    from experiments._shared import fold_standard_selection, fold_searches
    out = None
    for b in budgets:
        s = fold_standard_selection(
            reps, labels, groups, train_idx=train_idx, fold_i=fold_i,
            n_outer_folds=n_folds, dataset_name=dataset_name,
            models=[kirby_model], task_type='classification',
            metric=PART_A_METRIC, seed=FED_FOLD_SEED,
            condition_suffix=PART_A_SEARCH_SUFFIX,
            budget=int(b)).get(kirby_model)
        if s is None:
            out = None
            break
        if out is None:
            out = {'base': s['base'], 'base_detail': s['base_detail'],
                   'by_budget': {},
                   'search_lower_is_better': bool(s['search'].lower_is_better),
                   'search_metric': getattr(s['search'], 'metric_name', None)}
        out['by_budget'][int(b)] = {
            'hybrids': s['hybrids'],
            'hybrids_skipped_identical_features':
                s.get('hybrids_skipped_identical_features', [])}
    try:
        srch = fold_searches(
            reps, labels, groups, train_idx=train_idx, fold_i=fold_i,
            n_outer_folds=n_folds, dataset_name=dataset_name,
            task_type='classification', metric=PART_A_METRIC,
            seed=FED_FOLD_SEED, condition_suffix=PART_A_SEARCH_SUFFIX)
        errors = list(getattr(srch, 'failures', None) or [])
    except Exception as exc:
        errors = [{'error': f'{type(exc).__name__}: {exc}'}]
    return out, errors


def _fold_composition(silo_assignments, train_idx, test_idx, silo_names):
    """How many molecules of each source sit in a fold's training and test
    rows (dirty run 2026-09-28: every Part A test fold was 89% Lhasa)."""
    out = {}
    for part, idx in (('train', train_idx), ('test', test_idx)):
        n = int(len(idx))
        counts = {s: int((silo_assignments[idx] == i).sum())
                  for i, s in enumerate(silo_names)}
        out[part] = {'n_molecules': n, 'by_source': counts,
                     'fraction_by_source': {
                         s: (counts[s] / n if n else None) for s in counts}}
    return out


def _smoke_keep(strata, cap, rng):
    """Test-only row cap: a stratified head of the rows, at most `cap`,
    keeping each class's share. Indices, sorted."""
    strata = np.asarray(strata)
    if len(strata) <= cap:
        return np.arange(len(strata))
    keep = []
    for _cls in np.unique(strata):
        _pos = np.where(strata == _cls)[0]
        _take = max(1, int(round(cap * len(_pos) / len(strata))))
        keep.extend(rng.choice(_pos, size=min(_take, len(_pos)),
                               replace=False).tolist())
    return np.sort(np.array(keep))


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

# Datasets whose loaded rows ARE Lhasa's own training file. federated_helpers.
# load_lhasa_herg_data loads hERG_lhasa_training for lhasa_herg, and
# load_data('herg') loads the same FLuID pickle (DATASET_REPS: "FLuID
# classification (Lhasa). n=7,772"). On these, Part A used to call the handed
# rows the ChEMBL silo and then load the same training file again as the
# lhasa_train silo, which deduplication emptied (the ChEMBL silo comes first
# and keeps every shared molecule). Part A on lhasa_herg therefore ran two
# silos from one curator, and its ChEMBL-against-Lhasa disagreement count
# compared Lhasa with itself. Fixed 2026-10-05: on these datasets the ChEMBL
# silo is loaded from ChEMBL's hERG Ki records (herg_ki, CHEMBL240, 1,415
# molecules, continuous pKi), which is the source this function's docstring,
# its pKi cut and its threshold-sensitivity table were written for.
LHASA_OWN_FILE_DATASETS = ('lhasa_herg', 'herg')


def _run_herg_source_based(data, config, output_dir, dataset_name,
                           fed_model, fed_seed=0):
    """Strategy 2: Real data sources as federated silos.

    Silo A: KIRBy ChEMBL hERG (herg_ki: 1,415 mols, continuous pKi → binarized)
    Silo B: Lhasa training (7,772 mols, binary Active/Inactive)
    Silo C: Lhasa test (4,383 mols, binary Active/Inactive)
    """
    from experiments.federated.lhasa_herg_loader import (
        load_lhasa_herg_training, load_lhasa_herg_test,
        dedup_across_silos,
    )
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
    from find_best_hybrid_molecular import (
        get_representations as get_molecular_representations)

    print("\n" + "=" * 70)
    print("FED03 PART A: SOURCE-BASED hERG (Strategy 2)")
    print("=" * 70)

    kirby_model = fed_to_kirby_model(fed_model)
    budgets = _fed03_budgets(dataset_name)
    headline = _headline_budget(budgets)
    n_folds = _n_outer_folds(config)
    print(f"  model {fed_model} ({kirby_model}), seed {fed_seed}; budgets "
          f"{budgets} features; {n_folds} outer fold(s)")

    # The dataset's representations (DATASET_REPS[ds]['general'], as run_all
    # loaded them). Every silo is computed on exactly these names.
    rep_names = list(data['reps'].keys())

    _smoke_n = os.environ.get('KIRBY_SMOKE_N', '').strip()
    _cap = int(_smoke_n) if _smoke_n and int(_smoke_n) > 0 else 0
    _rng = np.random.RandomState(0)

    if dataset_name in LHASA_OWN_FILE_DATASETS:
        from find_best_hybrid_molecular import load_dataset
        smiles_chembl, labels_chembl, _ = load_dataset('herg_ki')
        smiles_chembl = list(smiles_chembl)
        labels_chembl = np.asarray(labels_chembl, dtype=float)
        chembl_is_binary = False
        chembl_source = ("herg_ki (ChEMBL CHEMBL240 Ki, continuous pKi), "
                         "loaded by fed03 because the handed rows are "
                         "Lhasa's own training file")
        reps_chembl = None          # computed below, after the smoke cap
    else:
        smiles_chembl = list(data['smiles'])
        labels_chembl = np.asarray(data['labels'])
        chembl_is_binary = bool(data.get('is_binary'))
        chembl_source = f"the handed dataset ({dataset_name})"
        reps_chembl = data['reps']
    print(f"  ChEMBL silo source: {chembl_source}")

    # Part A is a two-class experiment — every condition below trains on
    # blocker / non-blocker — so it is scored with AUROC, not the registered
    # metric of herg_ki (rmse, on continuous pKi).
    #
    # What the registered metric cost, measured 2026-09-23. The engine turns a
    # metric into a skill score, 1 - error / null_error, and clips it at zero.
    # On 0/1 labels with hard predictions the null model — predicting the mean
    # — has RMSE 0.4745, and a classifier at 70% accuracy has RMSE 0.5477,
    # WORSE than the null. Part A's classifiers sit near 0.5 MCC, so every
    # representation clipped to exactly zero and the engine fell back to an
    # equal split, inside every silo, on every fold.
    metric_name = PART_A_METRIC
    from kirby.find_best_hybrid import BUILTIN_METRICS
    metric_fn, lower_is_better = BUILTIN_METRICS[metric_name]

    # Binarize ChEMBL labels. When the ChEMBL rows already carry 0/1 labels
    # the pKi cut would collapse every sample to class 0, so already-binary
    # labels pass through.
    _uniq_chembl = set(np.unique(labels_chembl).tolist())
    labels_chembl_binary, _chembl_cut_note = binarise_labels(
        'herg_ki', labels_chembl, is_binary=chembl_is_binary)
    labels_chembl_binary = np.asarray(labels_chembl_binary)
    print(f"  ChEMBL silo labels: {_chembl_cut_note}")

    # Test-only row cap (KIRBY_SMOKE_N), stratified on the class. load_data
    # already caps the handed rows; the ChEMBL rows fed03 loads itself and the
    # two Lhasa files are capped here. No-op when the variable is unset.
    if _cap and reps_chembl is None:
        _k = _smoke_keep(labels_chembl_binary, _cap, _rng)
        if len(_k) < len(labels_chembl):
            print(f"  [SMOKE] chembl: {len(labels_chembl)} -> {len(_k)} "
                  f"molecules")
        smiles_chembl = [smiles_chembl[i] for i in _k]
        labels_chembl = labels_chembl[_k]
        labels_chembl_binary = labels_chembl_binary[_k]
    if reps_chembl is None:
        print(f"  computing {len(rep_names)} representations for "
              f"{len(smiles_chembl)} ChEMBL molecules...")
        reps_chembl = get_molecular_representations(
            smiles_chembl, ','.join(rep_names))
    # float32 on load (answer 18). get_representations neither casts nor
    # caches as float32 and this path skips _finalize_load_data, so the cast
    # is done here (2026-10-05). Integer fingerprints pass through unchanged,
    # as they do in _finalize_load_data.
    reps_chembl = {r: as_rep_dtype(a) for r, a in reps_chembl.items()}

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

    # Test-only row cap on the two Lhasa silos, read straight off Lhasa's PKL
    # files (7,772 and 4,383 molecules whatever the cap says). Every
    # representation is computed over all of them before a fold runs, which is
    # the whole cost of a local Part A.
    if _cap:
        def _cap_silo(sm, lb, ky, name):
            lb = np.asarray(lb)
            k = _smoke_keep(lb, _cap, _rng)
            if len(k) < len(lb):
                print(f"  [SMOKE] {name}: {len(lb)} -> {len(k)} molecules")
            return [sm[i] for i in k], lb[k], [ky[i] for i in k]

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
    # chosen one on trust. Recorded too is WHICH WAY the disagreements run:
    # when every one of them is Lhasa-active against ChEMBL-inactive, the
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
    if not (chembl_is_binary or _uniq_chembl <= {0, 1, 0.0, 1.0}):
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

    # Filter ChEMBL reps to match deduped molecule count.
    # dedup_across_silos may have removed repeated ChEMBL molecules.
    n_chembl_orig = len(smiles_chembl)
    n_chembl_deduped = len(deduped_smiles['chembl'])
    if n_chembl_deduped < n_chembl_orig:
        deduped_chembl_set = set(deduped_keys['chembl'])
        chembl_keep = np.array([
            k in deduped_chembl_set
            for k in chembl_inchikeys], dtype=bool)
        reps_chembl = {r: arr[chembl_keep] for r, arr in reps_chembl.items()}
        print(f"  ChEMBL filtered: {n_chembl_orig} → {n_chembl_deduped} "
              f"(matched dedup)")

    # Representations for the Lhasa molecules: the SAME names as the ChEMBL
    # silo, computed by get_representations from a comma-separated list.
    print("\nComputing representations for Lhasa molecules...")
    lhasa_all_smiles = (deduped_smiles['lhasa_train'] +
                        deduped_smiles['lhasa_test'])
    lhasa_reps = get_molecular_representations(lhasa_all_smiles,
                                               ','.join(rep_names))
    lhasa_reps = {r: as_rep_dtype(a) for r, a in lhasa_reps.items()}

    # Combine: ChEMBL reps + Lhasa reps. A representation that one side lacks
    # or that comes back at two widths is left out of every arm and recorded.
    combined_reps = {}
    reps_left_out = {}
    for rname in rep_names:
        chembl_arr = reps_chembl.get(rname)
        lhasa_arr = lhasa_reps.get(rname)
        if chembl_arr is None or lhasa_arr is None:
            reps_left_out[rname] = ('not computed for the '
                                    f"{'ChEMBL' if chembl_arr is None else 'Lhasa'}"
                                    ' molecules')
            continue
        if chembl_arr.shape[1] != lhasa_arr.shape[1]:
            reps_left_out[rname] = (f'ChEMBL {chembl_arr.shape[1]} features, '
                                    f'Lhasa {lhasa_arr.shape[1]}')
            continue
        # as_rep_dtype again: one side float32 and the other an integer
        # fingerprint would otherwise stack to float64.
        combined_reps[rname] = as_rep_dtype(np.vstack([chembl_arr, lhasa_arr]))
    for rname, why in reps_left_out.items():
        print(f"  representation {rname} left out: {why}")

    print(f"  Combined reps: {len(combined_reps)} reps, "
          f"{n_total} molecules")

    # Scaffold groups over the pooled molecules; the outer folds and every
    # inner split below are grouped on them (no random splits).
    from find_best_hybrid_molecular import assign_scaffold_groups
    pooled_groups, _ = assign_scaffold_groups(all_smiles)
    if pooled_groups is not None:
        pooled_groups = np.asarray(pooled_groups)
    if pooled_groups is None:
        raise RuntimeError(
            "Part A cross-validation requires scaffold groups; "
            "pooled_groups is None.")
    n_unique_groups = len(np.unique(pooled_groups))
    if n_unique_groups < max(2, n_folds):
        raise RuntimeError(
            f"Part A with {n_folds} outer fold(s) requires >= "
            f"{max(2, n_folds)} scaffold groups; got {n_unique_groups}.")

    K = len(silo_names)

    # Build Morgan FPs for FLuID conditions (once, all rows). C1, C2, C3, C5
    # and C8 fit on these: FLuID's own protocol, kept as the external
    # comparator as fed01 does. Not one of lhasa_herg's general
    # representations and not yet listed as an exception in the handoff
    # (2026-10-05); the result file's notes say so.
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

    # ── Outer folds (2026-10-05) ──────────────────────────────────────────
    # From fed_outer_folds: config['n_splits'] folds, seed 42, scaffold
    # groups. Until 2026-10-05 cv_split took seed=fed_seed, so the spread over
    # the three seeds mixed the model's seed with which molecules were held
    # out.
    per_fold_results = []
    folds = fed_outer_folds(combined_reps, all_labels, pooled_groups, n_folds,
                            'classification')
    print(f"\n{n_folds}-fold scaffold CV over {n_unique_groups} groups")
    ck = _fold_checkpoint(data, config, output_dir, dataset_name, 'A',
                          fed_model, fed_seed)

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        if ck.done(fold=fold_idx):
            print(f"\n── Fold {fold_idx + 1}/{n_folds}: read back from "
                  f"{ck.path}")
            per_fold_results.append(ck.get(fold=fold_idx))
            continue
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
        train_reps = {r: arr[train_idx] for r, arr in combined_reps.items()}
        test_reps = {r: arr[test_idx] for r, arr in combined_reps.items()}
        g_train = pooled_groups[train_idx]

        # ── Conditions ────────────────────────────────────────────────
        results = {
            'fold': int(fold_idx),
            'fold_composition': _fold_composition(
                silo_assignments, train_idx, test_idx, silo_names),
        }
        _comp = results['fold_composition']['test']['fraction_by_source']
        print("  test rows by source: " + ", ".join(
            f"{s} {v:.0%}" for s, v in _comp.items() if v is not None))

        # The per-fold search and selection (R2, R4): one search on this
        # fold's training rows of the pooled data, the run's model, every
        # decided budget. A failed search is recorded and every arm that
        # needs a hybrid or a base representation is skipped.
        print(f"  per-fold search ({kirby_model}, {len(budgets)} budgets)...")
        try:
            sel, sel_errors = _part_a_fold_selection(
                combined_reps, all_labels, pooled_groups,
                train_idx=train_idx, fold_i=fold_idx, n_folds=n_folds,
                dataset_name=dataset_name, budgets=budgets,
                kirby_model=kirby_model)
        except Exception as e:
            from kirby.result_cache import StaleResultDB as _Stale
            if isinstance(e, _Stale):
                raise
            sel, sel_errors = None, [{'error': f'{type(e).__name__}: {e}',
                                      'traceback': traceback.format_exc()}]
        results['selection'] = _selection_record(
            sel, sel_errors, dataset_name=dataset_name, fold_i=fold_idx,
            kirby_model=kirby_model, budgets=budgets)
        if sel is None:
            print(f"    search FAILED: {sel_errors}")

        # C1: Best single silo
        print("  C1: Single silo baselines...")
        single_silo = {}
        for i, sname in enumerate(silo_names):
            s_idx = train_silo_indices[i + 1]
            if len(s_idx) < 10 or len(np.unique(y_train[s_idx])) < 2:
                single_silo[sname] = {
                    'error': f'{len(s_idx)} training molecules, '
                             f'{len(np.unique(y_train[s_idx]))} class(es)'}
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
        #    held-out molecules. It needs the answers, so it is a bound.
        #  train_picked — the source that won on rows held out inside its own
        #    training molecules, scored afterwards on the fold's test
        #    molecules. The honest single-source baseline; C4's gain is read
        #    against it.
        best_silo = (max(valid_silos, key=lambda s: single_silo[s]['mcc'])
                     if valid_silos else None)

        def _probe_silo(sname):
            """One source's score on rows held out inside its own molecules,
            split on scaffold groups. None when the source has too few
            distinct scaffolds to hold any out, which drops it from the
            ranking rather than ranking it on a random split."""
            i = silo_names.index(sname)
            s_idx = train_silo_indices[i + 1]
            if len(s_idx) < 20:
                return None
            y_s = y_train[s_idx]
            if len(np.unique(y_s)) < 2:
                return None
            g_s = g_train[s_idx]
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

        # The run's model on each of the fold's 2 best base representations,
        # whole width, trained on every training molecule pooled (R4,
        # 2026-10-05). Before, the top 2 base representations were written
        # to the result file and never fitted.
        for _bi in range(2):
            _bkey = f'base{_bi + 1}_single_rep'
            _base = list((sel or {}).get('base') or [])
            if len(_base) <= _bi:
                results[_bkey] = {
                    'skipped': True,
                    'reason': ('the per-fold search failed' if sel is None
                               else f'the per-fold search ranked fewer than '
                                    f'{_bi + 1} base representations')}
                continue
            _rep = _base[_bi]
            try:
                clf_b = fit_fed_classifier(fed_model, train_reps[_rep],
                                           y_train, seed=fed_seed)
                mb = compute_clf_metrics(
                    y_test, clf_b.predict(test_reps[_rep]),
                    clf_b.predict_proba(test_reps[_rep])[:, 1])
                mb['rep'] = _rep
                mb['n_features'] = int(train_reps[_rep].shape[1])
                mb['train_rows_score'] = next(
                    (e['score'] for e in results['selection'].get(
                        'base_scores_train_rows', []) if e['rep'] == _rep),
                    None)
                results[_bkey] = convert_numpy(mb)
                print(f"  base {_bi + 1} ({_rep}): MCC={mb['mcc']:.3f}  "
                      f"AUROC={mb.get('auroc', float('nan')):.3f}")
            except Exception as e:
                results[_bkey] = {'failed': f'{type(e).__name__}: {e}',
                                  'rep': _rep}
                print(f"  base {_bi + 1} ({_rep}) FAILED: {e}")

        # Hoisted: train teachers per silo BEFORE C2/C3 so both conditions
        # share the same state.
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
            results['c2_fluid_similarity'] = {'skipped': True,
                                              'reason': 'teachers not trained'}
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
                results['c2_fluid_similarity'] = {'failed': str(e)}
                print(f"    FAILED: {e}")

        # C3: FLuID quality ensemble
        print("  C3: FLuID quality ensemble...")
        if not teachers_ready:
            results['c3_fluid_quality'] = {'skipped': True,
                                           'reason': 'teachers not trained'}
            print(f"    SKIPPED: teachers not trained")
        else:
            try:
                pred_cls, pred_prob, mcc_w = predict_fluid_quality_ensemble(
                    teacher_models, cluster_data_train, test_fps, K)
                c3 = compute_clf_metrics(y_test, pred_cls, pred_prob)
                c3['mcc_weights'] = convert_numpy(mcc_w)
                results['c3_fluid_quality'] = convert_numpy(c3)
                print(f"    MCC={c3['mcc']:.3f}")
            except Exception as e:
                results['c3_fluid_quality'] = {'failed': str(e)}
                print(f"    FAILED: {e}")

        # C4: KIRBy federated feature selection; C9: the same selection,
        # fitted one model per source. One per hybrid of the fold's set, at
        # every budget; hybrid1 keeps the old keys.
        #
        # Each per-source selection runs on the recipe's own representations,
        # cut to its representation count by the fold search's scores
        # (recipe_reps), with the recipe's allocation method, importance
        # method and internal evaluator ('match' resolved to the run's model).
        # federated_feature_selection takes no other recipe setting, so two
        # hybrids that agree on those and on the representations build the
        # same C4/C9 features; that is recorded as 'same_silo_selection_as'.
        from kirby.hybrid import apply_feature_selection, create_hybrid

        # C8 selects nothing; fitted once per fold, the same at every width.
        print("  C8: federated fit, all features (one model per source)...")

        def _fed_fit_and_score(Xtr_f, Xte_f):
            """One model per source; only predictions are combined."""
            teachers, s_tr, s_te = [None], [None], [None]
            n_trained = 0
            for _s in range(1, K + 1):
                _idx = train_silo_indices[_s]
                if len(_idx) < 5 or len(np.unique(y_train[_idx])) < 2:
                    teachers.append((None,))
                    s_tr.append(np.empty((0, Xtr_f.shape[1])))
                    s_te.append(Xte_f)
                    continue
                _m = fit_fed_classifier(fed_model, Xtr_f[_idx],
                                        y_train[_idx], seed=fed_seed)
                teachers.append((_m,))
                s_tr.append(Xtr_f[_idx])
                s_te.append(Xte_f)
                n_trained += 1
            _cls, _prob, _w = \
                predict_fluid_similarity_ensemble_classification(
                    teachers, s_tr, s_te, K, seed=fed_seed)
            _m = compute_clf_metrics(y_test, _cls, _prob)
            _m['sources_that_trained'] = int(n_trained)
            _m['n_features'] = int(Xtr_f.shape[1])
            return _m

        try:
            c8 = _fed_fit_and_score(train_fps, test_fps)
            results['c8_fed_fit_all_features'] = convert_numpy(c8)
            results['c8_fed_fit_all_features_by_budget'] = {
                str(b): convert_numpy(c8) for b in budgets}
            print(f"    C8 all features: MCC={c8['mcc']:.3f}  "
                  f"{c8['n_features']} features")
        except Exception as e:
            results['c8_fed_fit_all_features'] = {
                'failed': str(e), 'traceback': traceback.format_exc()}
            print(f"    C8 FAILED: {e}")

        _silo_sel_seen = {}       # {budget: {settings key: slot}}
        for _si, _slot in enumerate(_HYBRID_SLOTS):
            _sfx = _slot_suffix(_slot)
            c4_by_budget, c9_by_budget, c6_by_budget = {}, {}, {}
            print(f"  C4/C9/C6 [{_slot}]...")
            for _budget in budgets:
                h = _hybrid_at(sel, _budget, _si)
                if h is None:
                    _nf = _not_filled(_slot, _budget, sel)
                    c4_by_budget[str(_budget)] = dict(_nf)
                    c9_by_budget[str(_budget)] = dict(_nf)
                    c6_by_budget[str(_budget)] = dict(_nf)
                    continue
                bp = h['blueprint']
                _rec = _hybrid_record(h, kirby_model)
                _alloc, _imp, _int_eval, _rkw = _build_settings(bp, kirby_model)
                _ctx = (f'fed03 A {dataset_name} fold {fold_idx} {_slot} '
                        f'budget {_budget}')

                # ── C4: per-source selection, one model on pooled rows ──
                _fis = None
                try:
                    _tr_r, _rc_note = recipe_reps(train_reps, bp, _ctx)
                    _te_r = {r: test_reps[r] for r in _tr_r}
                    _skey = (_alloc, _imp, _int_eval, tuple(sorted(_tr_r)))
                    _dup = _silo_sel_seen.setdefault(_budget, {}).get(_skey)
                    _silo_sel_seen[_budget].setdefault(_skey, _slot)
                    _fail = {}
                    X_h_train, silo_budgets, silo_scores, _fis = \
                        federated_feature_selection(
                            _tr_r, y_train, train_silo_indices,
                            K, budget=_budget,
                            allocation_method=_alloc, importance_method=_imp,
                            internal_evaluator=_int_eval,
                            downstream_model=kirby_model,
                            # Each silo's 20% scoring slice is cut on these.
                            groups=g_train,
                            task_type='classification',
                            metric_fn=metric_fn,
                            lower_is_better=lower_is_better,
                            failures=_fail)
                    h_test_parts = []
                    for s in range(1, K + 1):
                        fi = _fis.get(s)
                        if fi is None:
                            continue
                        h_test_parts.append(apply_feature_selection(_te_r, fi))
                    X_h_test = np.hstack(h_test_parts)
                    clf = fit_fed_classifier(
                        fed_model, X_h_train, y_train, seed=fed_seed)
                    c4 = compute_clf_metrics(
                        y_test, clf.predict(X_h_test),
                        clf.predict_proba(X_h_test)[:, 1])
                    c4['n_features'] = int(X_h_train.shape[1])
                    c4['silo_budgets'] = convert_numpy(silo_budgets)
                    # None for a source whose selection failed or that was
                    # too small; the reason is in silo_failures.
                    c4['silo_scores'] = convert_numpy(silo_scores)
                    c4['silo_failures'] = convert_numpy(
                        {silo_names[s - 1]: why for s, why in _fail.items()})
                    c4['rep_count_note'] = _rc_note
                    c4['recipe_settings_not_applied'] = convert_numpy(
                        _c4_settings_not_applied(_rkw))
                    if _dup:
                        c4['same_silo_selection_as'] = _dup
                    print(f"    budget {_budget}: C4 MCC={c4['mcc']:.3f}  "
                          f"{c4['n_features']} features  split={silo_budgets}")
                except Exception as e:
                    c4 = {'failed': f'{type(e).__name__}: {e}',
                          'traceback': traceback.format_exc()}
                    _fis = None
                    print(f"    budget {_budget}: C4 FAILED: {e}")
                c4['budget_requested'] = int(_budget)
                c4['recipe'] = _rec
                c4_by_budget[str(_budget)] = convert_numpy(c4)

                # ── C9: the same per-source selection, one model per source.
                # C4's selection is reused: the call is the same, so building
                # it twice gave the same features at twice the cost.
                if _fis is None:
                    c9 = {'skipped': True,
                          'reason': 'C4 produced no per-source selection'}
                else:
                    try:
                        tr_p, te_p = [], []
                        for _s in range(1, K + 1):
                            f = _fis.get(_s)
                            if f is None:
                                continue
                            tr_p.append(apply_feature_selection(_tr_r, f))
                            te_p.append(apply_feature_selection(_te_r, f))
                        c9 = _fed_fit_and_score(np.hstack(tr_p),
                                                np.hstack(te_p))
                        print(f"    budget {_budget}: C9 MCC={c9['mcc']:.3f}  "
                              f"{c9['n_features']} features")
                    except Exception as e:
                        c9 = {'failed': f'{type(e).__name__}: {e}'}
                c9['budget_requested'] = int(_budget)
                c9['recipe'] = _rec
                c9_by_budget[str(_budget)] = convert_numpy(c9)

                # ── C6: the same recipe on one representation (ecfp4) ──
                # C4 minus C6 is what splitting the budget across three
                # sources buys over spending it all on one representation;
                # both sides are the same width. Groups are passed so the
                # internal holdout is scaffold-grouped (split audit kind B).
                #
                # No substitute (2026-10-05): if ecfp4 was left out of the
                # pooled representations, C6 is recorded as skipped. Before,
                # it silently took the first representation in the pool and
                # the aggregate mixed those runs with ecfp4 runs.
                try:
                    if C6_REP not in combined_reps:
                        raise _C6Skip(
                            f'{C6_REP} is not in the pooled representations: '
                            f'{reps_left_out.get(C6_REP, "not requested")}')
                    ecfp4_key = C6_REP
                    train_single = {ecfp4_key: train_reps[ecfp4_key]}
                    test_single = {ecfp4_key: test_reps[ecfp4_key]}
                    X_h_c6, fi_c6 = create_hybrid(
                        train_single, y_train,
                        groups=g_train,
                        allocation_method=_alloc,
                        importance_method=_imp,
                        budget=_budget,
                        internal_evaluator=_int_eval,
                        model=make_fed_classifier(fed_model, seed=fed_seed),
                        metric_fn=metric_fn, lower_is_better=lower_is_better,
                        augmentation_strategy='none',
                        **_rkw)
                    X_h_c6_test = apply_feature_selection(test_single, fi_c6)
                    clf_c6 = fit_fed_classifier(
                        fed_model, X_h_c6, y_train, seed=fed_seed)
                    c6 = compute_clf_metrics(
                        y_test, clf_c6.predict(X_h_c6_test),
                        clf_c6.predict_proba(X_h_c6_test)[:, 1])
                    c6['rep_used'] = ecfp4_key
                    c6['n_features'] = int(X_h_c6.shape[1])
                    print(f"    budget {_budget}: C6 MCC={c6['mcc']:.3f}  "
                          f"{c6['n_features']} features")
                except _C6Skip as e:
                    c6 = {'skipped': True, 'reason': str(e)}
                    print(f"    budget {_budget}: C6 skipped: {e}")
                except Exception as e:
                    c6 = {'failed': f'{type(e).__name__}: {e}'}
                    print(f"    budget {_budget}: C6 FAILED: {e}")
                c6['budget_requested'] = int(_budget)
                c6['recipe'] = _rec
                c6_by_budget[str(_budget)] = convert_numpy(c6)

            results['c4_kirby_federated_by_budget' + _sfx] = c4_by_budget
            results['c4_kirby_federated' + _sfx] = c4_by_budget[str(headline)]
            results['c9_fed_fit_selected_by_budget' + _sfx] = c9_by_budget
            results['c9_fed_fit_selected' + _sfx] = c9_by_budget[str(headline)]
            results['c6_kirby_single_rep_by_budget' + _sfx] = c6_by_budget
            results['c6_kirby_single_rep' + _sfx] = c6_by_budget[str(headline)]

        # C5: Naive pool (unrealistic baseline)
        print("  C5: Naive pool...")
        try:
            clf_pool = fit_fed_classifier(
                fed_model, train_fps, y_train, seed=fed_seed)
            c5 = compute_clf_metrics(y_test, clf_pool.predict(test_fps),
                                     clf_pool.predict_proba(test_fps)[:, 1])
            results['c5_naive_pool'] = convert_numpy(c5)
            print(f"    MCC={c5['mcc']:.3f}")
        except Exception as e:
            results['c5_naive_pool'] = {'failed': str(e)}
            print(f"    FAILED: {e}")

        # C7: FLuID multi-rep ensemble
        # One model per representation, weighted by its cross-validated MCC
        # inside the fold's training rows. 2026-10-05: the weighting model is
        # the run's own model (it was a fixed RandomForestClassifier on every
        # run, R1), the inner folds are C7_WEIGHT_FOLDS scaffold-grouped
        # folds, and the shuffled-KFold fallback is gone (random splits are
        # forbidden). With fewer scaffold groups than C7_WEIGHT_FOLDS the
        # representations are weighted equally and the result says so.
        print("  C7: FLuID multi-rep ensemble...")
        try:
            from sklearn.model_selection import cross_val_predict, GroupKFold
            from sklearn.metrics import matthews_corrcoef
            rep_teachers = []
            rep_names_list = sorted(combined_reps.keys())
            for rname in rep_names_list:
                clf_r = fit_fed_classifier(
                    fed_model, train_reps[rname], y_train, seed=fed_seed)
                rep_teachers.append((clf_r, rname, train_reps[rname],
                                     test_reps[rname]))

            _weights_note = None
            if len(np.unique(g_train)) >= C7_WEIGHT_FOLDS:
                cv_c7 = GroupKFold(n_splits=C7_WEIGHT_FOLDS)
                weights = []
                for clf_r, rname, X_tr, _ in rep_teachers:
                    try:
                        cv_pred = cross_val_predict(
                            make_fed_classifier(fed_model, seed=fed_seed),
                            X_tr, y_train, cv=cv_c7, groups=g_train)
                        w = max(matthews_corrcoef(y_train, cv_pred), 0.01)
                    except Exception as e:
                        print(f"    weight for {rname} failed ({e}); 0.01")
                        w = 0.01
                    weights.append(w)
                weights = np.array(weights, dtype=float)
            else:
                weights = np.ones(len(rep_teachers), dtype=float)
                _weights_note = (f'{len(np.unique(g_train))} scaffold groups '
                                 f'in the training rows, fewer than '
                                 f'{C7_WEIGHT_FOLDS}: equal weights')
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
            c7['weight_model'] = fed_model
            if _weights_note:
                c7['weights_note'] = _weights_note
            results['c7_fluid_multi_rep'] = convert_numpy(c7)
            print(f"    MCC={c7['mcc']:.3f}  ({len(rep_teachers)} teachers)")
        except Exception as e:
            results['c7_fluid_multi_rep'] = {'failed': str(e)}
            print(f"    FAILED: {e}")

        results = convert_numpy(results)
        # Written now, not at the end of the call (dirty-run bug 12).
        ck.put(results, fold=fold_idx)
        per_fold_results.append(results)

    # ── Aggregate across folds ──
    def _agg(cond_key, metric_key):
        vals = []
        for fr in per_fold_results:
            # Both c1 rows read the same per-fold entry; only the pick differs.
            _lookup = ('c1_single_silo'
                       if cond_key == 'c1_single_silo_ceiling' else cond_key)
            r = fr.get(_lookup) or {}
            if cond_key in ('c1_single_silo', 'c1_single_silo_ceiling'):
                key = ('train_picked' if cond_key == 'c1_single_silo'
                       else 'test_picked_ceiling')
                bs = r.get(key)
                if bs is None:
                    continue
                v = (r.get('per_silo', {}).get(bs) or {}).get(metric_key)
            else:
                v = r.get(metric_key)
            if v is not None and np.isfinite(v):
                vals.append(v)
        return vals

    aggregated = {}
    cond_keys = [
        'c1_single_silo', 'c1_single_silo_ceiling',
        'base1_single_rep', 'base2_single_rep',
        'c2_fluid_similarity', 'c3_fluid_quality',
        'c4_kirby_federated', 'c5_naive_pool', 'c6_kirby_single_rep',
        'c7_fluid_multi_rep',
        # the federated-FIT pair: one model per source, predictions combined
        'c8_fed_fit_all_features', 'c9_fed_fit_selected',
    ]
    # The second hybrid of the fold's set (2026-10-05).
    cond_keys += [_ck + _slot_suffix('hybrid2') for _ck in (
        'c4_kirby_federated', 'c6_kirby_single_rep', 'c9_fed_fit_selected')]
    for ck_ in cond_keys:
        row = {}
        # AUROC is the reported metric (the search's and the stamp's); MCC is
        # kept beside it under the keys every earlier reader used.
        row.update(_fold_summary(_agg(ck_, 'auroc'), 'auroc'))
        _mcc = _fold_summary(_agg(ck_, 'mcc'), 'mcc')
        row.update({k: v for k, v in _mcc.items()
                    if k.startswith('mcc')})
        row['per_fold'] = [fr.get(ck_, {}) for fr in per_fold_results]
        aggregated[ck_] = row

    # Summary
    print(f"\n  {'Condition':<30} {'AUROC (mean ± std)':>24} "
          f"{'MCC (mean ± std)':>24}")
    for key, name in [
        ('c1_single_silo', 'C1: Best silo, picked on train'),
        ('c1_single_silo_ceiling', 'C1 CEILING: best silo on test'),
        ('base1_single_rep', 'Base rep 1 (fold pick)'),
        ('base2_single_rep', 'Base rep 2 (fold pick)'),
        ('c2_fluid_similarity', 'C2: FLuID similarity'),
        ('c3_fluid_quality', 'C3: FLuID quality'),
        ('c4_kirby_federated', 'C4: KIRBy federated'),
        ('c5_naive_pool', 'C5: Naive pool'),
        ('c6_kirby_single_rep', 'C6: KIRBy single-rep'),
        ('c7_fluid_multi_rep', 'C7: FLuID multi-rep'),
        ('c8_fed_fit_all_features', 'C8: fed fit, all features'),
        ('c9_fed_fit_selected', 'C9: fed fit, selected'),
    ]:
        a = aggregated.get(key, {})
        print(f"  {name:<30} {_fmt_summary(a, 'auroc'):>24} "
              f"{_fmt_summary(a, 'mcc'):>24}")

    return {
        'experiment': 'fed03_source_based_herg',
        'dataset': dataset_name,
        # What run_all stamps the file with: the metric this part reports.
        'metric_name': metric_name,
        'lower_is_better': bool(lower_is_better),
        'metric_note': ('AUROC on the two-class blocker labels, the metric '
                        'the per-fold search ranks on; MCC is reported beside '
                        'it'),
        'selection_scope': 'fold_train_rows',
        'config': {
            'silos': silo_names,
            'n_total': n_total,
            'dedup_stats': dedup_stats,
            'pki_threshold': PKI_THRESHOLD,
            'pki_threshold_note': PKI_THRESHOLD_NOTE,
            'n_folds': n_folds,
            'fold_seed': FED_FOLD_SEED,
            'model': fed_model,
            'kirby_model': kirby_model,
            'model_seed': fed_seed,
            'budgets': budgets,
            'headline_budget': headline,
            'chembl_silo_source': chembl_source,
            'representations': sorted(combined_reps),
            'representations_left_out': reps_left_out,
            'search_condition_suffix': PART_A_SEARCH_SUFFIX,
            'layout': FED03_LAYOUT,
        },
        'notes': [
            # Recorded, not redesigned (Adelaide's instruction, 2026-10-05).
            'C8 and C9 differ in two ways at once, not one. C8 fits on '
            '2048-bit Morgan feature fingerprints (radius 2, useFeatures). C9 '
            "fits on features selected from the dataset's representations. "
            'So C9 minus C8 is not what the selection alone is worth.',
            'The two-class labels the per-fold search runs on are the '
            "pooled blocker / non-blocker labels; the dataset's decided grid "
            'was measured on its own rows (lhasa_herg: Lhasa training file '
            'only; herg_ki: continuous pKi). Accepted.',
            'C6 uses ecfp4 by name, the experiment\'s own single-'
            'representation control; the fold\'s 2 best base representations '
            'are the base1/base2 arms. If ecfp4 is not in the pooled '
            'representations, C6 is recorded as skipped, never substituted.',
            'C4 and C6 can differ in two ways at once, not one. C4 (and C9) '
            'rebuild through federated_feature_selection, which takes only '
            'the allocation method, importance method, internal evaluator '
            'and budget. C6 builds through create_hybrid with every recipe '
            'setting. Each C4 entry\'s recipe_settings_not_applied lists the '
            'settings (step_size, candidate_ranking, objective) where the '
            'recipe differs from the default C4 fell back to; where it is '
            'empty, C4 minus C6 is the split across sources alone.',
            'C1, C2, C3, C5 and C8 are FLuID\'s own protocol and fit on '
            '2048-bit Morgan feature fingerprints (radius 2, useFeatures), '
            'computed in this file. That fingerprint is not one of '
            'lhasa_herg\'s general representations; it is kept as the '
            'external comparator, as in fed01, and is not yet an approved '
            'exception in the handoff.',
        ],
        'cross_curator_disagreement': convert_numpy(_cross),
        'per_fold_selection': [fr.get('selection') for fr in per_fold_results],
        'fold_composition': [fr.get('fold_composition')
                             for fr in per_fold_results],
        **aggregated,
    }


# =============================================================================
# PART B: ASYMMETRIC REP SILOS
# =============================================================================

# The result-file key of each dataset metric, as compute_clf_metrics /
# compute_reg_metrics name it.
_METRIC_KEY = {'rmse': 'rmse', 'spearman': 'spearman_rho',
               'pearson': 'pearson_r', 'r2': 'r2', 'auc': 'auroc',
               'mcc': 'mcc', 'accuracy': 'accuracy'}


def _split_budget(total, names, widths=None):
    """total features split across names as evenly as integers allow, the
    remainder to the first names (2026-10-05; was max(5, total // n) per
    family, which could sum to less than C2's budget).

    widths: {name: features the family has}. A family is never given more
    than it has, and what it cannot take is split evenly over the families
    that still have room (2026-10-05). Without this, caco2's physicochemical
    family (pdv alone, 200 features) was given 334 of 1000 and 667 of 2000,
    so C3 built about 867 and 1534 features while C2 built 1000 and 2000, and
    the two arms differed in total width as well as in the split. The shares
    sum to min(total, sum of widths), which is also the most C2 can build
    from the same representations."""
    names = list(names)
    if widths is None:
        n = max(1, len(names))
        base, rem = divmod(int(total), n)
        return {nm: base + (1 if i < rem else 0) for i, nm in enumerate(names)}
    share = {nm: 0 for nm in names}
    left = min(int(total), sum(int(widths[nm]) for nm in names))
    open_ = [nm for nm in names if int(widths[nm]) > 0]
    while left > 0 and open_:
        base, rem = divmod(left, len(open_))
        for i, nm in enumerate(open_):
            give = min(base + (1 if i < rem else 0),
                       int(widths[nm]) - share[nm])
            share[nm] += give
            left -= give
        open_ = [nm for nm in open_ if share[nm] < int(widths[nm])]
    return share


def _run_asymmetric_reps(data, config, output_dir, dataset_name,
                         fed_model, fed_seed=0):
    """Strategy 4: Each org contributes different representation types.

    Every representation in the dataset's pool is sorted into one of three
    families (REP_FAMILIES): structural, physicochemical, learned.
    """
    from kirby.hybrid import create_hybrid, apply_feature_selection

    print("\n" + "=" * 70)
    print("FED03 PART B: ASYMMETRIC REP SILOS (Strategy 4)")
    print("=" * 70)

    kirby_model = fed_to_kirby_model(fed_model)
    budgets = _fed03_budgets(dataset_name)
    headline = _headline_budget(budgets)
    n_folds = _n_outer_folds(config)
    print(f"  model {fed_model} ({kirby_model}), seed {fed_seed}; budgets "
          f"{budgets} features; {n_folds} outer fold(s)")

    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    if groups is not None:
        groups = np.asarray(groups)

    # No augmentation on any arm (DECISIONS.md Step 8 cuts it on lhasa_herg,
    # caco2 and logd; run_all does not load the added features there). Until
    # 2026-10-05 C2 could build with an 'uncapped' recipe and its augmented
    # features while C3 never took any, a fourth difference between them.
    if data.get('augmentations'):
        print("  augmentation is not used by fed03 (Step 8); the loaded "
              "added features are ignored")

    metric_name_b, metric_fn_b, lower_is_better_b = get_domain_metric(data)

    # Use the dataset's declared task_type (FED-6) — the cardinality heuristic
    # (len(unique)<=10) misroutes low-cardinality regression targets into the
    # classification path. Fall back to the heuristic only if task_type absent.
    _task_type_b = data.get('task_type')
    if _task_type_b in ('classification', 'regression'):
        is_clf = (_task_type_b == 'classification')
    else:
        is_clf = len(np.unique(labels)) <= 10
    task_b = 'classification' if is_clf else 'regression'
    metrics_fn = compute_clf_metrics if is_clf else _reg_metrics

    # The reported metric is the dataset's (rmse on caco2 and logd); a metric
    # these helpers do not compute falls back to mcc / rmse and the stamp
    # says which.
    primary_key = _METRIC_KEY.get(metric_name_b)
    if primary_key is None or (is_clf and primary_key in
                               ('rmse', 'spearman_rho', 'pearson_r', 'r2')):
        primary_key = 'mcc' if is_clf else 'rmse'
        reported_metric = primary_key
        reported_lower = (primary_key == 'rmse')
    else:
        reported_metric = metric_name_b
        reported_lower = bool(lower_is_better_b)

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
    family_of = {r: f for f, rs in silo_reps.items() for r in rs}

    K = len(silo_reps)
    silo_names = sorted(silo_reps.keys())
    print(f"  {K} silos resolved:")
    for sname in silo_names:
        rep_list = silo_reps[sname]
        dims = [reps[r].shape[1] for r in rep_list]
        print(f"    {sname}: {rep_list} ({sum(dims)}D total)")

    # Scaffold-group-aware outer folds (feedback_no_random_splits.md).
    if groups is None:
        raise RuntimeError(
            "Part B cross-validation requires groups (scaffold-aware); "
            "groups=None.")
    n_unique_groups = len(np.unique(groups))
    if n_unique_groups < max(2, n_folds):
        raise RuntimeError(
            f"Part B with {n_folds} outer fold(s) requires >= "
            f"{max(2, n_folds)} groups; got {n_unique_groups}.")

    # From fed_outer_folds: config['n_splits'] folds, seed 42, the same rows
    # as the dataset's per-fold search store. Until 2026-10-05 cv_split took
    # seed=fed_seed.
    folds = fed_outer_folds(reps, labels, groups, n_folds, task_b)
    print(f"  {n_folds}-fold CV over {n_unique_groups} groups "
          f"({'clf' if is_clf else 'reg'})")
    ck = _fold_checkpoint(data, config, output_dir, dataset_name, 'B',
                          fed_model, fed_seed)

    def _new_model():
        return (make_fed_classifier(fed_model, seed=fed_seed) if is_clf
                else make_fed_regressor(fed_model, seed=fed_seed))

    def _score(model, X_te, y_te):
        y_pred = model.predict(X_te)
        if is_clf:
            return metrics_fn(y_te, y_pred, model.predict_proba(X_te)[:, 1])
        return metrics_fn(y_te, y_pred)

    per_fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        if ck.done(fold=fold_idx):
            print(f"\n── Fold {fold_idx + 1}/{n_folds}: read back from "
                  f"{ck.path}")
            per_fold_results.append(ck.get(fold=fold_idx))
            continue
        print(f"\n── Fold {fold_idx + 1}/{n_folds} "
              f"({len(train_idx)} train / {len(test_idx)} test) ──")
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        train_reps = {r: arr[train_idx] for r, arr in reps.items()}
        test_reps = {r: arr[test_idx] for r, arr in reps.items()}
        train_groups = groups[train_idx]

        results = {'fold': int(fold_idx)}

        # The per-fold search and selection (R2, R4): one search on this
        # fold's training rows, the dataset's decided grid, every decided
        # budget, the run's model. A failed search is recorded and every arm
        # needing a hybrid or a base representation is skipped.
        print(f"  per-fold search ({kirby_model}, {len(budgets)} budgets)...")
        try:
            _sel_all = fed_fold_selection(
                reps, labels, groups, train_idx=train_idx, fold_i=fold_idx,
                n_folds=n_folds, dataset_name=dataset_name, budgets=budgets,
                models=[kirby_model], task_type=task_b, metric='auto')
            sel = _sel_all.get(kirby_model)
            sel_errors = _sel_all.get('_errors') or []
        except Exception as e:
            from kirby.result_cache import StaleResultDB as _Stale
            if isinstance(e, _Stale):
                raise
            sel, sel_errors = None, [{'error': f'{type(e).__name__}: {e}',
                                      'traceback': traceback.format_exc()}]
        results['selection'] = _selection_record(
            sel, sel_errors, dataset_name=dataset_name, fold_i=fold_idx,
            kirby_model=kirby_model, budgets=budgets)
        if sel is None:
            print(f"    search FAILED: {sel_errors}")

        # C1: Best single silo
        print("  C1: Single silo baselines...")
        single_silo = {}
        for sname in silo_names:
            rep_list = silo_reps[sname]
            X_tr = np.hstack([train_reps[r] for r in rep_list])
            X_te = np.hstack([test_reps[r] for r in rep_list])
            model = fit_fed_model(fed_model, X_tr, y_train, seed=fed_seed,
                                  task_type=task_b)
            single_silo[sname] = _score(model, X_te, y_test)
            single_silo[sname]['n_features'] = int(X_tr.shape[1])
            primary = single_silo[sname].get(primary_key)
            print(f"    {sname}: {primary_key}={primary}  ({X_tr.shape[1]}D)")

        # Which family is 'best' is picked two ways and both are kept.
        # test_picked_ceiling reads the fold's held-out molecules, so it needs
        # the answers; train_picked ranks the families on rows held out inside
        # the training molecules and is the one C2's gain is read against.
        def _probe_family(sname):
            from sklearn.model_selection import (
                GroupShuffleSplit as _GroupShuffleSplit)
            rep_list = silo_reps[sname]
            X_all = np.hstack([train_reps[r] for r in rep_list])
            if len(y_train) < 20 or len(np.unique(train_groups)) < 2:
                return None
            sp = _GroupShuffleSplit(n_splits=1, test_size=0.25,
                                    random_state=fed_seed)
            tr_v, va_v = next(sp.split(X_all, y_train, groups=train_groups))
            m = fit_fed_model(fed_model, X_all[tr_v], y_train[tr_v],
                              seed=fed_seed, task_type=task_b)
            mm = _score(m, X_all[va_v], y_train[va_v])
            v = mm.get(primary_key)
            if v is None or not np.isfinite(v):
                return None
            # rank_silos_train_only always maximises, so an error is negated.
            return -v if reported_lower else v

        _valid = [s for s in single_silo
                  if single_silo[s].get(primary_key) is not None]
        _pick = min if reported_lower else max
        _ceiling = (_pick(_valid, key=lambda s: single_silo[s][primary_key])
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

        # The run's model on each of the fold's 2 best base representations
        # (R4, 2026-10-05). Before, they were written to the result file and
        # never fitted.
        for _bi in range(2):
            _bkey = f'base{_bi + 1}_single_rep'
            _base = list((sel or {}).get('base') or [])
            if len(_base) <= _bi:
                results[_bkey] = {
                    'skipped': True,
                    'reason': ('the per-fold search failed' if sel is None
                               else f'the per-fold search ranked fewer than '
                                    f'{_bi + 1} base representations')}
                continue
            _rep = _base[_bi]
            try:
                m_b = fit_fed_model(fed_model, train_reps[_rep], y_train,
                                    seed=fed_seed, task_type=task_b)
                mb = _score(m_b, test_reps[_rep], y_test)
                mb['rep'] = _rep
                mb['family'] = family_of.get(_rep)
                mb['n_features'] = int(train_reps[_rep].shape[1])
                mb['train_rows_score'] = next(
                    (e['score'] for e in results['selection'].get(
                        'base_scores_train_rows', []) if e['rep'] == _rep),
                    None)
                results[_bkey] = convert_numpy(mb)
                print(f"  base {_bi + 1} ({_rep}): "
                      f"{primary_key}={mb.get(primary_key)}")
            except Exception as e:
                results[_bkey] = {'failed': f'{type(e).__name__}: {e}',
                                  'rep': _rep}
                print(f"  base {_bi + 1} ({_rep}) FAILED: {e}")

        # C2 and C3 at every budget, once per hybrid of the fold's set.
        #
        # 2026-10-05: C2 and C3 differ ONLY in how the budget is split. Both
        # build from the same representations (the recipe's own, cut to its
        # representation count by the fold search's scores, recipe_reps), with
        # the same allocation method, importance method, internal evaluator
        # ('match' resolved to the run's model for both), the same model
        # instance type, the same metric and direction, the same rest of the
        # recipe, and no augmentation. C2 takes the whole budget in one
        # selection over those representations; C3 splits it across the
        # families they belong to, each family selecting on its own, and
        # concatenates. Until 2026-10-05 the two also differed in evaluator
        # (C3 forced rf), metric (C3 passed none) and augmentation (C2 could
        # add augmented features): four differences at once.
        _c2_by_slot = {s: {} for s in _HYBRID_SLOTS}
        _c3_by_slot = {s: {} for s in _HYBRID_SLOTS}
        for _pb_budget in budgets:
            for _si, _slot in enumerate(_HYBRID_SLOTS):
                h = _hybrid_at(sel, _pb_budget, _si)
                if h is None:
                    _nf = _not_filled(_slot, _pb_budget, sel)
                    _c2_by_slot[_slot][str(_pb_budget)] = dict(_nf)
                    _c3_by_slot[_slot][str(_pb_budget)] = dict(_nf)
                    continue
                bp = h['blueprint']
                _rec = _hybrid_record(h, kirby_model)
                _alloc, _imp, _int_eval, _rkw = _build_settings(bp, kirby_model)
                _ctx = (f'fed03 B {dataset_name} fold {fold_idx} {_slot} '
                        f'budget {_pb_budget}')
                try:
                    _kept, _rc_note = recipe_reps(train_reps, bp, _ctx)
                    _kept = {r: v for r, v in _kept.items() if r in family_of}
                    if not _kept:
                        raise ValueError('none of the recipe\'s '
                                         'representations is in a family')
                except Exception as e:
                    _f = {'failed': f'{type(e).__name__}: {e}',
                          'budget_requested': int(_pb_budget), 'recipe': _rec}
                    _c2_by_slot[_slot][str(_pb_budget)] = convert_numpy(_f)
                    _c3_by_slot[_slot][str(_pb_budget)] = convert_numpy(_f)
                    print(f"  [{_slot}] budget {_pb_budget}: FAILED: {e}")
                    continue
                _fams_used = {f: [r for r in silo_reps[f] if r in _kept]
                              for f in silo_names}
                _fams_used = {f: rs for f, rs in _fams_used.items() if rs}

                # C2: KIRBy hybrid across all silo reps
                print(f"  C2 [{_slot}] budget {_pb_budget}: KIRBy hybrid "
                      f"(all families)...")
                try:
                    X_h_tr, fi = create_hybrid(
                        _kept, y_train,
                        groups=train_groups,
                        allocation_method=_alloc,
                        importance_method=_imp,
                        internal_evaluator=_int_eval,
                        budget=_pb_budget,
                        model=_new_model(),
                        metric_fn=metric_fn_b,
                        lower_is_better=lower_is_better_b,
                        augmentations=None,
                        augmentation_strategy='none',
                        **_rkw)
                    X_h_te = apply_feature_selection(
                        {r: test_reps[r] for r in _kept}, fi)
                    model = fit_fed_model(fed_model, X_h_tr, y_train,
                                          seed=fed_seed, task_type=task_b)
                    c2 = _score(model, X_h_te, y_test)
                    c2['allocation'] = convert_numpy(fi.get('allocation', {}))
                    c2['n_features'] = int(X_h_tr.shape[1])
                    print(f"    {primary_key}={c2.get(primary_key)}  "
                          f"({c2['n_features']}D)")
                except Exception as e:
                    c2 = {'failed': f'{type(e).__name__}: {e}',
                          'traceback': traceback.format_exc()}
                    print(f"    FAILED: {e}")
                c2['budget_requested'] = int(_pb_budget)
                c2['representations'] = sorted(_kept)
                c2['rep_count_note'] = _rc_note
                c2['recipe'] = _rec
                _c2_by_slot[_slot][str(_pb_budget)] = convert_numpy(c2)

                # C3: Federated feature selection (privacy-preserving)
                print(f"  C3 [{_slot}] budget {_pb_budget}: federated FS "
                      f"(per-family selection)...")
                try:
                    from sklearn.model_selection import (
                        GroupShuffleSplit as _GSS)
                    from experiments._shared import get_model_factories

                    _fam_width = {f: int(sum(_kept[r].shape[1] for r in rs))
                                  for f, rs in _fams_used.items()}
                    _fam_budget = _split_budget(_pb_budget, list(_fams_used),
                                                _fam_width)
                    silo_fis, silo_scores, silo_failures = {}, {}, {}
                    for i, sname in enumerate(_fams_used, 1):
                        silo_rep_tr = {r: _kept[r] for r in _fams_used[sname]}
                        try:
                            # Probe, recorded only: select on 80% of the
                            # training rows, score on the scaffold-grouped
                            # 20% the selection never saw, on the dataset's
                            # metric. Not used to split the budget. Needs 2
                            # scaffold groups; the random-split fallback was
                            # removed 2026-10-05.
                            if len(np.unique(train_groups)) >= 2:
                                tr, va = next(_GSS(
                                    n_splits=1, test_size=0.2,
                                    random_state=42 + i + fold_idx).split(
                                        np.zeros((len(y_train), 1)), y_train,
                                        groups=train_groups))
                                _, fi_probe = create_hybrid(
                                    {r: a[tr] for r, a in silo_rep_tr.items()},
                                    y_train[tr], groups=train_groups[tr],
                                    allocation_method=_alloc,
                                    importance_method=_imp,
                                    internal_evaluator=_int_eval,
                                    budget=min(_fam_budget[sname], len(tr)),
                                    model=_new_model(),
                                    metric_fn=metric_fn_b,
                                    lower_is_better=lower_is_better_b,
                                    augmentations=None,
                                    augmentation_strategy='none',
                                    **_rkw)
                                X_p_tr = apply_feature_selection(
                                    {r: a[tr] for r, a in silo_rep_tr.items()},
                                    fi_probe)
                                X_p_va = apply_feature_selection(
                                    {r: a[va] for r, a in silo_rep_tr.items()},
                                    fi_probe)
                                _fac = get_model_factories(
                                    [_int_eval], task_type=task_b,
                                    n_samples=len(tr)).get(_int_eval)
                                if _fac is None or X_p_tr.shape[1] == 0:
                                    silo_scores[sname] = None
                                else:
                                    _ev = _fac()
                                    _ev.fit(X_p_tr, y_train[tr])
                                    silo_scores[sname] = float(metric_fn_b(
                                        y_train[va], _ev.predict(X_p_va)))
                            else:
                                silo_scores[sname] = None

                            # The selection that is used: every training row,
                            # this family's share of the budget.
                            _, fi_s = create_hybrid(
                                silo_rep_tr, y_train,
                                groups=train_groups,
                                allocation_method=_alloc,
                                importance_method=_imp,
                                internal_evaluator=_int_eval,
                                budget=_fam_budget[sname],
                                model=_new_model(),
                                metric_fn=metric_fn_b,
                                lower_is_better=lower_is_better_b,
                                augmentations=None,
                                augmentation_strategy='none',
                                **_rkw)
                            silo_fis[sname] = fi_s
                        except Exception as e:
                            silo_fis[sname] = None
                            silo_scores[sname] = None
                            silo_failures[sname] = f'{type(e).__name__}: {e}'
                            print(f"    family {sname} FAILED: {e}")

                    h_parts_tr, h_parts_te = [], []
                    for sname, fi in silo_fis.items():
                        if fi is None:
                            continue
                        rs = _fams_used[sname]
                        h_parts_tr.append(apply_feature_selection(
                            {r: _kept[r] for r in rs}, fi))
                        h_parts_te.append(apply_feature_selection(
                            {r: test_reps[r] for r in rs}, fi))
                    if h_parts_tr:
                        X_fed_tr = np.hstack(h_parts_tr)
                        X_fed_te = np.hstack(h_parts_te)
                        model = fit_fed_model(fed_model, X_fed_tr, y_train,
                                              seed=fed_seed, task_type=task_b)
                        c3 = _score(model, X_fed_te, y_test)
                        c3['n_features'] = int(X_fed_tr.shape[1])
                    else:
                        c3 = {'failed': 'no family produced a selection'}
                    c3['silo_scores'] = convert_numpy(silo_scores)
                    c3['silo_scores_metric'] = metric_name_b
                    c3['silo_failures'] = silo_failures
                    c3['budget_per_family'] = _fam_budget
                    c3['features_per_family'] = _fam_width
                    c3['families_used'] = _fams_used
                    # Whether C3 came out the same width as C2 on this fold
                    # and budget, measured, not assumed (2026-10-05). They
                    # can still differ if a family's selection fails or
                    # create_hybrid builds fewer features than it was given.
                    c3['n_features_c2'] = c2.get('n_features')
                    c3['same_width_as_c2'] = (
                        c2.get('n_features') is not None
                        and c3.get('n_features') == c2.get('n_features'))
                    # The width is printed beside the score so C2 and C3 can
                    # be compared for size.
                    print(f"    {primary_key}={c3.get(primary_key)}  "
                          f"({c3.get('n_features', 0)}D)")
                except Exception as e:
                    c3 = {'failed': f'{type(e).__name__}: {e}',
                          'traceback': traceback.format_exc()}
                    print(f"    FAILED: {e}")
                    traceback.print_exc()
                c3['budget_total'] = int(_pb_budget)
                c3['budget_requested'] = int(_pb_budget)
                c3['representations'] = sorted(_kept)
                c3['recipe'] = _rec
                _c3_by_slot[_slot][str(_pb_budget)] = convert_numpy(c3)

        # Headline rows the summary table reads.
        for _slot in _HYBRID_SLOTS:
            _sfx = _slot_suffix(_slot)
            results['c2_kirby_hybrid_by_budget' + _sfx] = _c2_by_slot[_slot]
            results['c3_federated_fs_by_budget' + _sfx] = _c3_by_slot[_slot]
            results['c2_kirby_hybrid' + _sfx] = _c2_by_slot[_slot].get(
                str(headline))
            results['c3_federated_fs' + _sfx] = _c3_by_slot[_slot].get(
                str(headline))

        results = convert_numpy(results)
        # Written now, not at the end of the call (dirty-run bug 12).
        ck.put(results, fold=fold_idx)
        per_fold_results.append(results)

    # ── Aggregate across folds ──
    def _agg_primary(cond_key):
        vals = []
        for fr in per_fold_results:
            _lookup = ('c1_single_silo'
                       if cond_key == 'c1_single_silo_ceiling' else cond_key)
            r = fr.get(_lookup) or {}
            if cond_key in ('c1_single_silo', 'c1_single_silo_ceiling'):
                if not r:
                    continue
                key = ('train_picked' if cond_key == 'c1_single_silo'
                       else 'test_picked_ceiling')
                best = r.get(key)
                if best is None:
                    continue
                v = (r.get('per_silo', {}).get(best) or {}).get(primary_key)
            else:
                v = r.get(primary_key)
            if v is not None and np.isfinite(v):
                vals.append(float(v))
        return vals

    aggregated = {}
    _ck_list_b = ['c1_single_silo', 'c1_single_silo_ceiling',
                  'base1_single_rep', 'base2_single_rep',
                  'c2_kirby_hybrid', 'c3_federated_fs',
                  'c2_kirby_hybrid' + _slot_suffix('hybrid2'),
                  'c3_federated_fs' + _slot_suffix('hybrid2')]
    for ck_ in _ck_list_b:
        row = _fold_summary(_agg_primary(ck_), primary_key)
        row['per_fold'] = [fr.get(ck_, {}) for fr in per_fold_results]
        aggregated[ck_] = row

    # Summary
    print(f"\n  {'Condition':<35} {primary_key + ' (mean ± std)':>28}")
    for key, name in [
        ('c1_single_silo', 'C1: Best family, picked on train'),
        ('c1_single_silo_ceiling', 'C1 CEILING: best family on test'),
        ('base1_single_rep', 'Base rep 1 (fold pick)'),
        ('base2_single_rep', 'Base rep 2 (fold pick)'),
        ('c2_kirby_hybrid', 'C2: KIRBy hybrid (all)'),
        ('c3_federated_fs', 'C3: Federated FS'),
    ]:
        print(f"  {name:<35} "
              f"{_fmt_summary(aggregated.get(key, {}), primary_key, 4):>28}")

    return {
        'experiment': 'fed03_asymmetric_reps',
        'dataset': dataset_name,
        # What run_all stamps the file with: the metric this part reports.
        'metric_name': reported_metric,
        'lower_is_better': bool(reported_lower),
        'selection_scope': 'fold_train_rows',
        'config': {
            'silo_reps': {s: silo_reps[s] for s in silo_names},
            'n_folds': n_folds,
            'fold_seed': FED_FOLD_SEED,
            'model': fed_model,
            'kirby_model': kirby_model,
            'model_seed': fed_seed,
            'budgets': budgets,
            'headline_budget': headline,
            'primary_metric': primary_key,
            'search_metric': metric_name_b,
            'is_classification': is_clf,
            'layout': FED03_LAYOUT,
        },
        'notes': [
            'C2 and C3 differ only in how the budget is split: one '
            'selection over the recipe\'s representations (C2) against one '
            'selection per family of those representations at an even share '
            'of the budget (C3). Same representations, allocation method, '
            'importance method, internal evaluator, metric and recipe '
            'settings; no augmentation on either.',
            'C3\'s split is even across families, but a family is never '
            'given more features than it has; what it cannot take goes '
            'evenly to the families with room (2026-10-05). So C3 asks for '
            'the same total as C2. Each C3 entry records same_width_as_c2, '
            'measured on the built features; an entry where it is false '
            'differs from C2 in width as well as in the split.',
            'Which representations a recipe keeps is decided by the fold '
            'search\'s scores over the whole pool (its representation '
            'count), so C3\'s families share that one ranking.',
        ],
        'per_fold_selection': [fr.get('selection') for fr in per_fold_results],
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

    The model is config['fed_model'], one of the dataset's two decided models
    (run_all, fed_models_for). There is no 'rf' default since 2026-10-05: a
    config without a model takes the dataset's first decided model.
    """
    strategy = kwargs.get('strategy', None)
    fed_model = config.get('fed_model')
    if fed_model is None:
        from experiments.federated.federated_helpers import fed_models_for
        fed_model = fed_models_for(dataset_name)[0]
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


# ── One (model, seed) call saved as it finishes (2026-09-25) ────────────────
# federated_helpers.run_all calls run() once per (model, seed) and writes
# fed03_synthetic_private/<dataset>.json after the last. Each call's return
# value is written to fed03_synthetic_private/<dataset>.partial.jsonl the
# moment it returns and read back on the next attempt (_checkpoint.
# resumable_run), keyed on the model, the seed and any keyword argument, under
# an identity that includes the input fingerprint, the plain config, the
# KIRBY_* settings and FED03_LAYOUT (bumped 2026-10-05, so a call stored by the
# whole-dataset version is never read back). Inside a call, each outer fold is
# saved as it finishes too (_fold_checkpoint). --force-rerun /
# KIRBY_FORCE_RERUN=1 discards both.
def run(data, config, output_dir, dataset_name, **kwargs):
    from experiments._checkpoint import resumable_run
    _key = {'fed_model': str(config.get('fed_model')),
            'fed_seed': int(config.get('fed_seed', 0)),
            **{k: v for k, v in kwargs.items()}}
    return resumable_run(
        'fed03_synthetic_private',
        lambda: _run_one(data, config, output_dir, dataset_name, **kwargs),
        data, config, output_dir, dataset_name, key=_key,
        layout=FED03_LAYOUT)


if __name__ == '__main__':
    import argparse as _ap
    from experiments.federated.federated_helpers import (
        run_all, _get_federated_config)
    parser = _ap.ArgumentParser(description='Fed03: Synthetic Private Datasets')
    parser.add_argument('--dataset', type=str, default='lhasa_herg',
                        help='lhasa_herg (Part A), caco2 or logd (Part B)')
    parser.add_argument('--output', type=str, default='tests/results/federated')
    parser.add_argument('--n-folds', type=int, default=None,
                        help="outer folds; default config['n_splits']")
    parser.add_argument('--force-rerun', action='store_true')
    args = parser.parse_args()

    # The same path the suite takes: run_all loads the dataset's decided
    # representations, loops over its two decided models and the seeds, and
    # writes the result file. Part A or B follows from the dataset name.
    _cfg = dict(_get_federated_config())
    if args.n_folds is not None:
        _cfg['n_splits'] = int(args.n_folds)
    run_all(args.dataset, args.output, experiments=['fed03'], config=_cfg,
            force_rerun=args.force_rerun)
