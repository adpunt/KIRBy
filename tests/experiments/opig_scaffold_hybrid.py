#!/usr/bin/env python3
"""
Scaffold-diff reps vs the real rep pool — the honest pipeline comparison.

THE QUESTION (user, this is the one that matters): if the scaffold-diff mutation
reps go into the ACTUAL pipeline pool alongside the real representations (the
language-model embeddings esm2/prot_t5/ablang2/... and the cheap ones), on the
real models, with honest nested cross-validation — do they BEAT the base reps
as standalone representations?

A second question used to sit here: does the hybrid put scaffold-diff features
into the winning recipe? Answering it needs the number of features each
representation contributed, which nothing records, and recording it means
re-running all four datasets cold. Dropped by the owner 2026-09-11. The count
this file used to print was the number of representation NAMES in the recipe,
and the engine sets that list to every representation in the pool, so it never
moved. It is gone rather than left unread.

This is NOT the similarity/nearest-neighbour proxy. It trains the real models and
uses the honest evaluator (search on outer-train only; every hybrid and every
single representation it reports scored on the untouched outer fold; nothing
selection-leaked).

STANDALONE BY DESIGN. Nothing here is registered into MUTATION_REPS / ANTIBODY_REPS
or added to DATASET_REPS. The base pool is the dataset's pruned 10
(DATASET_REPS[ds]['general']); the two scaffold-diff reps are computed in THIS file from
the exact sequences load_data returns (so rows cannot misalign) and injected into
the rep dict. Each fold's search runs on tier 'general' with the real dataset name,
so the dataset's decided grid applies, and is filed under the fold's store name plus
'__scaffold_cmp' (NAMESPACE_SUFFIX). Its stored rows are already separate from the
per-fold general searches over the pruned 10, because the stored-search identity
carries the representation pool as (name, width) pairs; the suffix only makes that
visible in the store name.

DATASETS: the four single-wildtype antibody DMS sets with mutation metadata.
phillips_h3 is excluded (user-cut 2026-07-18). ddg / graphinity excluded here
(multi-PDB; a separate adapter).

WHAT IS MEASURED (per dataset, per outer fold, since 2026-10-05):
  - one search on that fold's TRAINING rows over the dataset's pruned 10
    representations plus the two scaffold-diff ones, on the dataset's decided
    grid (DECISIONS.md: its two models, its budgets, importance methods,
    allocation methods, internal evaluators and representation counts;
    augmentation 'none'; no pruning).
  - from it, per decided model: the 2 best base representations and the 2 best
    hybrids, both ranked on those training rows, each fitted on them and
    scored on the held-out rows.
  - each scaffold-diff representation alone on the held-out rows, with its
    rank among every representation that search scored.
  - how many of each hybrid's selected features came from the two
    scaffold-diff representations.
  - hybrid1 minus base1 per fold, and the mean over the folds that scored
    both arms.
No distance hybrid and no similarity hybrid: nothing here is measured with
neighbours or distances (Adelaide, 2026-10-05).
The metric itself is the dataset's own (Spearman for regression, AUROC for the
binarised binding sets) — the same metric every other pipeline run uses.

SERVER RUN. The real pool contains GPU language-model reps, so this is a SLURM job
(tests/slurm_scripts/submit_scaffold_hybrid.sh). Locally, only --smoke is sensible:
it keeps the dataset's pruned representations that need no GPU (onehot, blosum,
aac, paratope). Only the two adams sets have any (expression: blosum and onehot;
binding: blosum); --smoke on a koenig set stops that dataset with an error.

Usage:
    # server, full real pool, all 4 datasets
    python tests/experiments/opig_scaffold_hybrid.py --mode full

    # single dataset, custom pool
    python tests/experiments/opig_scaffold_hybrid.py --mode full \\
        --datasets flab_expression_koenig --reps esm2,prot_t5,ablang2

    # local plumbing smoke (no-GPU reps only; still calls the search — keep tiny)
    python tests/experiments/opig_scaffold_hybrid.py --mode fast --smoke \\
        --datasets flab_expression_adams
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))
sys.path.insert(0, os.path.join(HERE, '..'))          # tests/  -> experiments._shared
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'similarity_metrics'))

from kirby.similarity.distance_config import (          # noqa: E402
    DEFAULT_DISTANCE_METRIC as _DEFAULT_DISTANCE_METRIC,
    set_distance_metric as _set_distance_metric,
    get_distance_metric as _get_distance_metric,
)

# 2026-10-05: --distance-metric is removed. Nothing in this file measures a
# distance, but the process-wide distance setting is a column of the
# stored-search identity (find_best_hybrid.result_db_identity,
# 'distance_metric'). A flag or KIRBY_DISTANCE_METRIC other than manhattan
# therefore re-keyed every per-fold search here, and the result file did not
# record it. The setting is now pinned to this constant before any search and
# written into every dataset's result file.
SEARCH_DISTANCE_METRIC = _DEFAULT_DISTANCE_METRIC


DATASETS = ['flab_expression_koenig', 'flab_binding_koenig',
            'flab_expression_adams', 'flab_binding_adams']
SCAFFOLD_REPS = ('dphyschem', 'dphyschem_pos')
# ⛔ BEFORE READING THE HYBRID DELTA AS A SCAFFOLD-DIFF RESULT, COUNT THE
# FEATURES. The hybrid arm here wins on every downstream model, and it is easy
# to report that as "the mutation-difference representations contribute
# something the language models miss". Measured on flab_expression_koenig
# 2026-09-21, 3 outer folds, budgets 100 and 500: 9 of the 15 winning recipes
# selected ZERO scaffold features, random forest took none in any fold and
# still beat prot_t5 by +0.0366 spearman, and the largest share anywhere was
# 18 of 300. The recipes mix prot_t5, igt5, ablang2 and esm2 at roughly
# 30/25/25/20 and give blosum and onehot near zero.
#
# So the delta this file prints is a HYBRID-vs-BEST-SINGLE result on this
# dataset. It is not evidence about the scaffold-diff representations, which
# lose standalone as well (dphyschem ranks 4th-6th of 5-8, dphyschem_pos last
# wherever it scores).
#
# The count is already on disk: Blueprint.allocation_per_fold, one dict per
# inner fold, keyed by representation name. summarise_one does not read it.
#   bp = result['by_model'][model]['blueprints_per_fold'][outer_fold]
#   sum(n for inner in bp['allocation_per_fold'].values()
#         for r, n in inner.items() if r in SCAFFOLD_REPS)
# EXPERIMENTS_FINALIZED records this question as dropped for want of the
# number. The number exists; nothing reads it.
# Since 2026-10-05 every hybrid is rebuilt on its outer fold's training rows
# and the features it took from each representation are written out per
# hybrid ('features_by_rep', and the two scaffold-diff representations' share
# as 'scaffold_features'), so the count above is now recorded.
#
# The search tier is 'general' with the real dataset name, so the dataset's
# decided grid applies (DECISIONS.md Step 1-8, through decided_grid). The old
# private tiers 'scaffold_cmp' and 'general_distance__scaffold_cmp' are gone.
TIER = 'general'
# Each fold's search is filed under the fold's nested-CV store name plus this
# suffix. The stored-search identity already carries the representation pool
# as sorted (name, width) pairs (result_db_identity 'rep_pool'), so this pool
# of the pruned 10 plus the two scaffold-diff representations cannot collide
# with the plain per-fold general search over the pruned 10 even without it.
# The suffix makes that separation readable in the store name. Rows stored
# under the two old tier names are never read.
NAMESPACE_SUFFIX = '__scaffold_cmp'
# No distance search and no similarity search (Adelaide, 2026-10-05, answer
# 1): this file measures prediction, nothing with neighbours or distances, so
# it keeps 2 base representations and 2 general hybrids per decided model.
SEED = 42                      # outer-fold split and every per-fold search

# Recipe fields copied off each selected hybrid into the result JSON. 'budget'
# is a number of features.
HYBRID_FIELDS = ('label', 'allocation_method', 'importance_method', 'budget',
                 'internal_evaluator', 'rep_count', 'step_size')

# Written and printed wherever a spread over folds would otherwise be taken
# from a single fold (--max-folds 1, or --mode fast, is one held-out split).
ONE_FOLD_NOTE = 'one fold, no spread'


def build_scaffold_reps(heavy_seqs, light_seqs):
    """Compute the two scaffold-diff reps from the SAME sequences load_data
    returned, guaranteeing row alignment with the base reps. Uses the validated
    research implementations in scaffold_reps (bit-identical to what the study ran).
    """
    import scaffold_reps as SR
    S, wt, muts = SR.encode_sequences(
        {'heavy_seqs': list(heavy_seqs), 'light_seqs': list(light_seqs)})
    return {'dphyschem': SR.rep_dphys(S, wt, muts),
            'dphyschem_pos': SR.rep_dphys_positional(S, wt, muts)}, S.shape[0]


def base_pool_for(dataset_name, override=None, smoke=False):
    """The base representations: the dataset's pruned 10,
    DATASET_REPS[ds]['general'].

    2026-10-05: an override (--reps, --smoke) is narrowed to members of that
    list and the dropped names are printed, so only representations that came
    out of representation pruning are ever searched. The two scaffold-diff
    representations are this experiment's own addition and are added on top.
    Augmentation representations are not used: Step 8 cut augmentation on all
    four datasets (decided_aug_conditions returns 'none' only).

    smoke: keep only the pruned representations that need no GPU
    (ANTIBODY_FAST_REPS: onehot, blosum, aac, paratope), in pruned order.

    2026-10-05: an override that leaves no representation RAISES. Before, the
    empty list reached load_data, which reads an empty rep_names as "no list
    given" and built every antibody and mutation representation (the whole
    GPU pool); the search then ran on the two scaffold-diff representations
    alone. --smoke on either koenig set did exactly that, because neither
    pruned 10 holds a no-GPU representation.
    """
    from experiments.complexity.complexity_helpers import DATASET_REPS
    pruned = list(DATASET_REPS[dataset_name]['general'])
    if smoke:
        from kirby.representations.antibody import ANTIBODY_FAST_REPS
        keep = [r for r in pruned if r in ANTIBODY_FAST_REPS]
        if not keep:
            raise ValueError(
                f'{dataset_name}: --smoke keeps the pruned representations '
                f'that need no GPU ({sorted(ANTIBODY_FAST_REPS)}), and its '
                f'pruned 10 holds none of them: {pruned}')
        return keep
    if not override:
        return pruned
    keep = [r for r in override if r in pruned and r not in SCAFFOLD_REPS]
    dropped = [r for r in override
               if r not in keep and r not in SCAFFOLD_REPS]
    if dropped:
        print(f'  {dataset_name}: {dropped} are not in its pruned '
              f'representations, dropped', flush=True)
    if not keep:
        raise ValueError(
            f'{dataset_name}: --reps {override} keeps none of its pruned '
            f'representations {pruned}')
    return keep


def fold_namespace(fold_i, n_outer):
    """The per-fold search store name for this pool (see NAMESPACE_SUFFIX)."""
    from kirby.find_best_hybrid import nested_cv_cache_namespace
    return (f'{nested_cv_cache_namespace(fold_i, n_outer, SEED)}'
            f'{NAMESPACE_SUFFIX}')


def _recipe(entry, n_features=None, scaffold_features=None,
            features_by_rep=None):
    """The fields the result JSON records for one hybrid.

    Everything here comes off the top_hybrids entry of the search that ran on
    this outer fold's training rows, plus the feature counts of the rebuild on
    those same rows. 'budget' is in features.
    """
    out = {f: entry.get(f) for f in HYBRID_FIELDS}
    out['aug_condition'] = entry.get('aug_condition')
    out['rep_count_source'] = entry.get('rep_count_source')
    bp = entry.get('blueprint')
    out['reps'] = sorted(getattr(bp, 'reps', None) or [])
    # Mean over the inner folds of this outer fold's training rows.
    out['train_rows_score'] = entry.get('score_mean')
    if features_by_rep is not None:
        out['features_by_rep'] = features_by_rep
    if n_features is not None:
        out['n_features'] = n_features
    if scaffold_features is not None:
        out['scaffold_features'] = scaffold_features
    return out


def _features_by_rep(feat_info):
    """{representation: how many of its features the rebuild selected}."""
    return {r: int(len(info['selected_indices']))
            for r, info in (feat_info or {}).items()
            if isinstance(info, dict) and 'selected_indices' in info}


def _search_failures(reps, labels, groups, tr, k, n_outer, ds, task_type,
                     metric_name, ns):
    """The failures recorded by the search fold_standard_selection just ran.

    Reads the in-process memo with the same arguments, so nothing is searched
    a second time.
    """
    from experiments._shared import fold_searches
    try:
        srch = fold_searches(
            reps, labels, groups, train_idx=tr, fold_i=k,
            n_outer_folds=n_outer, dataset_name=ds, task_type=task_type,
            metric=metric_name, domain='antibody', seed=SEED,
            cache_namespace=ns)
        return [{kk: (vv if isinstance(vv, (int, float, str, type(None)))
                      else str(vv))
                 for kk, vv in f.items()}
                for f in (getattr(srch, 'failures', None) or [])]
    except Exception as e:          # reading the memo must not stop the fold
        return [{'error': f'could not read the failure record: '
                          f'{type(e).__name__}: {e}'}]


def _baseline_ranks(search, model, lower_is_better):
    """Each scaffold representation's rank among every representation the
    search scored for this model, on this fold's training rows, and how many
    were ranked. Read from search.baselines, which the search already wrote;
    no new fits."""
    baselines = getattr(search, 'baselines', {}) or {}
    scores = {}
    for rep, per_model in baselines.items():
        if str(rep).startswith('_') or not isinstance(per_model, dict):
            continue
        v = per_model.get(model)
        if v is None or not np.isfinite(v):
            continue
        scores[rep] = float(v)
    out = {'n_reps_ranked': len(scores), 'scaffold_rank': {},
           'scaffold_train_rows_score': {}}
    for r in SCAFFOLD_REPS:
        if r not in scores:
            continue
        better = sum(1 for v in scores.values()
                     if (v < scores[r] if lower_is_better else v > scores[r]))
        out['scaffold_rank'][r] = 1 + better
        out['scaffold_train_rows_score'][r] = scores[r]
    return out


def run_fold(ds, k, tr, te, reps, labels, groups, *, models, n_outer,
             task_type, metric_name, metric_fn, lower_is_better):
    """One outer fold: the standard selection, then every arm scored on the
    held-out rows.

    One search on this fold's TRAINING rows over the pruned 10 plus the two
    scaffold-diff representations, on the dataset's decided general grid
    (fold_standard_selection: pruning off, augmentation 'none', which is what
    Step 8 keeps on all four datasets, and only the dataset's two decided
    models). Per model it names the 2 best single representations and the 2
    best hybrids, both ranked on those training rows; a second hybrid whose
    selected features equal the first's is skipped for the next one.

    Each single representation is fitted on the training rows and scored on
    the held-out rows. Each hybrid is rebuilt with build_fold_hybrid from its
    own recipe, with the per-representation scores that same per-fold search
    measured (dirty-run bug 1), and then fitted and scored the same way.

    The two scaffold-diff representations are also scored alone whether or not
    the search ranked them in the top 2, because whether they beat the base
    representations standalone is this file's question; their rank among all
    the representations the search scored is recorded beside them.

    A search that fails, or a model short of 2 representations or 2 hybrids,
    is recorded under 'search_failures' / 'shortfall' with dataset, fold,
    model and the error, and those slots are skipped. Nothing is filled in
    instead.
    """
    from experiments._shared import (fold_standard_selection,
                                     build_fold_hybrid,
                                     fold_hybrid_scoring_kwargs)
    from kirby.find_best_hybrid import _outer_fit_score, get_model

    labels = np.asarray(labels)
    groups_arr = None if groups is None else np.asarray(groups)
    g_tr = None if groups_arr is None else groups_arr[tr]
    g_te = None if groups_arr is None else groups_arr[te]
    y_tr, y_te = labels[tr], labels[te]
    ns = fold_namespace(k, n_outer)

    def _score(X_tr, X_te, model_name):
        """Fit on the training rows, score on the held-out rows. The engine's
        own held-out scorer: it fits FeatureScaler on the training rows (each
        0/1 feature passes through unscaled, every other feature is
        standardised -- answer 17), builds the model by name at this row
        count, and applies the dataset's metric."""
        score, _ = _outer_fit_score(
            X_tr, y_tr, X_te, y_te, model_name, task_type, metric_fn,
            seed=SEED, train_groups=g_tr, test_idx=np.asarray(te),
            test_groups=g_te)
        return None if score is None or not np.isfinite(score) else float(score)

    fold_rec = {'fold': int(k), 'dataset': ds,
                'n_train': int(len(tr)), 'n_held_out': int(len(te)),
                'cache_namespace': ns, 'tier': TIER,
                'representations': sorted(reps),
                'selection_scope': 'fold_train_rows',
                'metric': metric_name,
                'lower_is_better': bool(lower_is_better),
                'search_failures': [], 'shortfall': [], 'models': {}}

    sel = fold_standard_selection(
        reps, labels, groups_arr, train_idx=tr, fold_i=k,
        n_outer_folds=n_outer, dataset_name=ds, models=list(models),
        task_type=task_type, metric=metric_name, domain='antibody',
        seed=SEED, cache_namespace=ns, verbose=True)

    failed = sorted(m for m, s in sel.items() if s is None)
    if failed:
        errs = _search_failures(reps, labels, groups_arr, tr, k, n_outer, ds,
                               task_type, metric_name, ns)
        for m in failed:
            fold_rec['search_failures'].append({
                'dataset': ds, 'fold': int(k), 'model': m, 'errors': errs})
            print(f'    fold {k} {m:9s} SEARCH FAILED, slots skipped: '
                  f'{[e.get("error") or e.get("error_type") for e in errs]}',
                  flush=True)

    tr_reps = {r: np.asarray(X)[tr] for r, X in reps.items()}
    te_reps = {r: np.asarray(X)[te] for r, X in reps.items()}

    for m in models:
        s = sel.get(m)
        if s is None:
            continue
        bd = s.get('base_detail') or {}
        selected = list(bd.get('selected') or [])
        if len(selected) < 2 or len(s['hybrids']) < 2:
            fold_rec['shortfall'].append({
                'dataset': ds, 'fold': int(k), 'model': m,
                'n_base': len(selected), 'n_hybrids': len(s['hybrids']),
                'reason': bd.get('shortfall_reason')})

        row = {'base': [], 'hybrids': [], 'scaffold_alone': [],
               'not_scored': [],
               'hybrids_skipped_identical_features': [
                   x.get('label')
                   for x in (s.get('hybrids_skipped_identical_features')
                             or [])]}
        row.update(_baseline_ranks(s['search'], m, lower_is_better))

        # The 2 single representations this fold's search ranked highest.
        for i, e in enumerate(selected, 1):
            rep = e['rep']
            try:
                sc = _score(tr_reps[rep], te_reps[rep], m)
            except Exception as ex:
                row['not_scored'].append(
                    f'base{i} {rep}: {type(ex).__name__}: {ex}')
                continue
            row['base'].append({'slot': f'base{i}', 'rep': rep,
                                'train_rows_score': e.get('score'),
                                'held_out_score': sc})
            print(f'    fold {k} {m:9s} base{i}   {rep:26s} '
                  f'{metric_name}={"—" if sc is None else f"{sc:.4f}"}',
                  flush=True)

        # Each scaffold-diff representation alone, this file's question.
        for rep in SCAFFOLD_REPS:
            if rep not in tr_reps:
                row['not_scored'].append(f'{rep}: not built on this dataset')
                continue
            try:
                sc = _score(tr_reps[rep], te_reps[rep], m)
            except Exception as ex:
                row['not_scored'].append(
                    f'{rep} alone: {type(ex).__name__}: {ex}')
                continue
            row['scaffold_alone'].append({
                'rep': rep, 'held_out_score': sc,
                'train_rows_score': row['scaffold_train_rows_score'].get(rep),
                'rank_on_train_rows': row['scaffold_rank'].get(rep),
                'n_reps_ranked': row['n_reps_ranked']})
            print(f'    fold {k} {m:9s} alone   {rep:26s} '
                  f'{metric_name}={"—" if sc is None else f"{sc:.4f}"}  '
                  f'(rank {row["scaffold_rank"].get(rep, "?")}'
                  f'/{row["n_reps_ranked"]} on the training rows)', flush=True)

        # The 2 hybrids, each rebuilt on this fold's training rows.
        for i, e in enumerate(s['hybrids'], 1):
            bp = e['blueprint']
            # The real estimator this hybrid will be fed to, so accept/reject
            # inside the rebuild is scored by fitting it. get_model returns
            # None when a model's own row cap is exceeded (svm above 10,000
            # rows); these four datasets hold 4,275 or 5,000 rows, so it does
            # not bite, and if it ever does the rebuild falls back to the
            # internal evaluator and the fold says so.
            _mdl = get_model(m, task_type, n_samples=len(y_tr))
            if _mdl is None:
                row['not_scored'].append(
                    f'hybrid{i}: {m} has no estimator at {len(y_tr)} training '
                    f'rows; the rebuild used the internal evaluator instead')
            try:
                X_tr_h, X_te_h, fi = build_fold_hybrid(
                    tr_reps, te_reps, y_tr, bp, groups=g_tr,
                    # Measured by this fold's own search on these training
                    # rows, so the budget split is the one that search
                    # settled on (dirty-run bug 1).
                    precomputed_alloc_scores=getattr(bp, 'rep_scores', None),
                    **fold_hybrid_scoring_kwargs(
                        bp, model=_mdl,
                        metric_fn=(metric_fn if _mdl is not None else None),
                        lower_is_better=(lower_is_better if _mdl is not None
                                         else None)))
            except Exception as ex:
                row['not_scored'].append(
                    f'hybrid{i}: build raised {type(ex).__name__}: {ex}')
                print(f'    fold {k} {m:9s} hybrid{i} build failed: '
                      f'{type(ex).__name__}: {ex}', flush=True)
                continue
            fbr = _features_by_rep(fi)
            rec = _recipe(
                e, n_features=int(np.asarray(X_tr_h).shape[1]),
                features_by_rep=fbr,
                scaffold_features=int(sum(fbr.get(r, 0)
                                          for r in SCAFFOLD_REPS)))
            rec['slot'] = f'hybrid{i}'
            try:
                rec['held_out_score'] = _score(X_tr_h, X_te_h, m)
            except Exception as ex:
                rec['held_out_score'] = None
                row['not_scored'].append(
                    f'hybrid{i}: scoring raised {type(ex).__name__}: {ex}')
            row['hybrids'].append(rec)
            _sc = rec['held_out_score']
            print(f'    fold {k} {m:9s} hybrid{i} '
                  f'{rec["n_features"]:5d} features, '
                  f'{rec["scaffold_features"]} of them scaffold-diff '
                  f'({rec["allocation_method"]}, {rec["importance_method"]}, '
                  f'budget {rec["budget"]}, eval '
                  f'{rec["internal_evaluator"]}, rep_count '
                  f'{rec["rep_count"]})  '
                  f'{metric_name}={"—" if _sc is None else f"{_sc:.4f}"}',
                  flush=True)

        # The one paired number per model per fold: the top hybrid against the
        # top single representation, both chosen without the held-out rows.
        # 2026-10-05: read by slot name, base1 and hybrid1. The lists skip an
        # arm that raised, so position 0 could be base2 or hybrid2 and the
        # fold would compare different slots from other folds. A missing
        # base1 or hybrid1 now leaves the paired number None.
        b1 = next((e for e in row['base'] if e['slot'] == 'base1'), None)
        h1 = next((e for e in row['hybrids'] if e['slot'] == 'hybrid1'), None)
        best_b = None if b1 is None else b1['held_out_score']
        best_h = None if h1 is None else h1['held_out_score']
        row['top_base_rep'] = None if b1 is None else b1['rep']
        row['top_hybrid_label'] = None if h1 is None else h1.get('label')
        row['top_base_held_out'] = best_b
        row['top_hybrid_held_out'] = best_h
        row['delta_held_out'] = (None if best_b is None or best_h is None
                                 else best_h - best_b)
        fold_rec['models'][m] = row

    return fold_rec


def summarise_one(dataset_name, block):
    """The per-representation answer and the paired hybrid difference, read
    off the per-fold records this file writes.

    Every number here is a held-out one: each arm was chosen on its outer
    fold's training rows and scored on that fold's held-out rows. The mean
    over folds is printed with the number of folds it covers, and with one
    fold it carries 'spread_note' instead of a spread.
    """
    per_fold = block.get('per_fold') or []
    lower_is_better = bool(block.get('lower_is_better', False))
    out = {'dataset': dataset_name,
           'metric_name': block.get('metric_name'),
           'lower_is_better': lower_is_better,
           'models': sorted(block.get('models') or []),
           'n_outer_folds': block.get('n_outer_folds'),
           'n_folds_completed': len(per_fold),
           'selection_scope': 'fold_train_rows',
           'rule': block.get('rule'),
           'by_model': {}}
    if len(per_fold) == 1:
        out['spread_note'] = ONE_FOLD_NOTE

    for m in out['models']:
        rows = [fr['models'][m] for fr in per_fold if m in (fr.get('models')
                                                            or {})]
        if not rows:
            continue
        def _mean(vals):
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            return float(np.mean(vals)) if vals else None

        slots = {}
        for r in rows:
            for e in r['base'] + r['hybrids']:
                d = slots.setdefault(e['slot'], {'per_fold': [],
                                                 'names': []})
                d['per_fold'].append(e.get('held_out_score'))
                d['names'].append(e.get('rep') or e.get('label'))
        for d in slots.values():
            d['mean_held_out'] = _mean(d['per_fold'])
            d['n_folds'] = sum(1 for v in d['per_fold']
                               if v is not None and np.isfinite(v))
            if d['n_folds'] == 1:
                d['spread_note'] = ONE_FOLD_NOTE

        scaffold = {}
        for r in rows:
            for e in r['scaffold_alone']:
                d = scaffold.setdefault(e['rep'], {'per_fold': [],
                                                   'ranks': []})
                d['per_fold'].append(e.get('held_out_score'))
                d['ranks'].append(e.get('rank_on_train_rows'))
        for d in scaffold.values():
            d['mean_held_out'] = _mean(d['per_fold'])
            d['n_folds'] = sum(1 for v in d['per_fold']
                               if v is not None and np.isfinite(v))
            if d['n_folds'] == 1:
                d['spread_note'] = ONE_FOLD_NOTE

        deltas = [r.get('delta_held_out') for r in rows]
        scaf_feat = [h.get('scaffold_features') for r in rows
                     for h in r['hybrids']
                     if h.get('scaffold_features') is not None]
        out['by_model'][m] = {
            'slots': slots,
            'scaffold_alone': scaffold,
            # hybrid1 minus base1, per fold and as a mean over the folds that
            # scored both arms.
            'per_fold_delta': deltas,
            'mean_delta_held_out': _mean(deltas),
            'n_folds_both_arms': sum(1 for v in deltas
                                     if v is not None and np.isfinite(v)),
            # How many of each hybrid's selected features came from the two
            # scaffold-diff representations, over every hybrid built for this
            # model: the count the old comment asked for.
            'scaffold_features_per_hybrid': scaf_feat,
            'n_hybrids_with_scaffold_features': sum(1 for v in scaf_feat
                                                    if v > 0),
            'n_hybrids_built': len(scaf_feat),
            'n_reps_ranked': rows[0].get('n_reps_ranked'),
        }
        if out['by_model'][m]['n_folds_both_arms'] == 1:
            out['by_model'][m]['spread_note'] = ONE_FOLD_NOTE

    out['search_failures'] = [f for fr in per_fold
                              for f in (fr.get('search_failures') or [])]
    out['shortfall'] = [f for fr in per_fold
                        for f in (fr.get('shortfall') or [])]
    out['not_scored'] = [f'fold {fr["fold"]} {m}: {w}' for fr in per_fold
                         for m, r in (fr.get('models') or {}).items()
                         for w in (r.get('not_scored') or [])]
    return out


def _fmt(v, signed=False):
    if v is None or not np.isfinite(v):
        return '—'
    return f'{v:+.4f}' if signed else f'{v:.4f}'


def print_summary(summaries):
    print('\n' + '=' * 84)
    print('SCAFFOLD-DIFF REPRESENTATIONS IN THE PRUNED POOL, HELD-OUT FOLDS')
    # Each dataset is scored in its own metric and carries its own direction,
    # printed beside the metric name below.
    print('One row = one slot of one model: its mean over the held-out folds '
          'and the folds it covers.')
    print('Dataset-native metric; direction per dataset.')
    print('=' * 84)
    for s in summaries:
        direction = 'lower better' if s.get('lower_is_better') else 'higher better'
        print(f'\n### {s["dataset"]}   (metric: {s.get("metric_name")}, '
              f'{direction}; {s.get("n_folds_completed")} of '
              f'{s.get("n_outer_folds")} outer folds)')
        for m, d in sorted((s.get('by_model') or {}).items()):
            print(f'  {m}')
            for slot in sorted(d['slots']):
                e = d['slots'][slot]
                names = ', '.join(str(n) for n in e['names'][:3])
                print(f'    {slot:8s} {_fmt(e["mean_held_out"]):>9s}  '
                      f'{e["n_folds"]} folds  |  {names}')
            for rep, e in sorted(d['scaffold_alone'].items()):
                ranks = ', '.join('?' if r is None else str(r)
                                  for r in e['ranks'])
                print(f'    alone    {_fmt(e["mean_held_out"]):>9s}  '
                      f'{e["n_folds"]} folds  |  {rep} '
                      f'(rank per fold of {d.get("n_reps_ranked")}: {ranks})')
            pf = ', '.join(_fmt(v, signed=True) for v in d['per_fold_delta'])
            print(f'    hybrid1 minus base1: '
                  f'{_fmt(d["mean_delta_held_out"], signed=True)} over '
                  f'{d["n_folds_both_arms"]} folds that scored both arms  |  '
                  f'per fold: {pf or "none"}')
            print(f'    scaffold-diff features taken: '
                  f'{d["n_hybrids_with_scaffold_features"]} of '
                  f'{d["n_hybrids_built"]} hybrids, counts '
                  f'{d["scaffold_features_per_hybrid"]}')
        for w in (s.get('search_failures') or []):
            print(f'    SEARCH FAILED fold {w.get("fold")} {w.get("model")}: '
                  f'{w.get("errors")}')
        for w in (s.get('shortfall') or []):
            print(f'    SHORT fold {w.get("fold")} {w.get("model")}: '
                  f'{w.get("n_base")} base, {w.get("n_hybrids")} hybrids '
                  f'({w.get("reason")})')
        for w in (s.get('not_scored') or [])[:10]:
            print(f'    not scored: {w}')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--reps', default=None,
                    help='comma-separated base pool override, narrowed to the '
                         'dataset\'s pruned representations (the two '
                         'scaffold-diff ones are always added)')
    ap.add_argument('--mode', default='full', choices=['fast', 'full'])
    # --models is deliberately absent. Every search here runs on the dataset's
    # two decided models (DECISIONS.md Step 1), read from
    # experiments/decided_grid.py. (The old note said models stayed None to
    # match a SLURM precompute; nothing precomputes this pool's searches.)
    ap.add_argument('--budgets', default=None,
                    help='ignored since 2026-10-05: the budgets are the '
                         'dataset\'s decided ones (DECISIONS.md Step 2), '
                         'applied inside the per-fold search.')
    ap.add_argument('--max-folds', type=int, default=None, dest='max_folds',
                    help='number of outer held-out folds (default: the '
                         'config\'s n_splits). 1 is a single held-out split.')
    ap.add_argument('--smoke', action='store_true',
                    help='only the dataset\'s pruned representations that '
                         'need no GPU (onehot, blosum, aac, paratope), for '
                         'local plumbing checks; a dataset with none stops '
                         'with an error. Overrides --reps.')
    ap.add_argument('--output-dir', default=os.path.join(HERE, 'results'))
    ap.add_argument('--force-rerun', action='store_true',
                    help='Recompute folds an earlier attempt finished '
                         '(opig_scaffold_hybrid_summary.partial.jsonl).')
    args = ap.parse_args()

    # Pinned, not taken from the environment: see SEARCH_DISTANCE_METRIC.
    _set_distance_metric(SEARCH_DISTANCE_METRIC)
    from experiments._shared import get_config

    override = None
    if args.reps and not args.smoke:
        override = [r.strip() for r in args.reps.split(',')]
    if args.budgets:
        print('  --budgets is ignored: the per-fold searches run on each '
              'dataset\'s decided budgets (DECISIONS.md Step 2).', flush=True)

    config = get_config(args)
    os.makedirs(args.output_dir, exist_ok=True)
    summaries, t0 = [], time.time()

    # Each FOLD is written to opig_scaffold_hybrid_summary.partial.jsonl as
    # it finishes and read back on the next attempt (dirty-run bug 12: this
    # file used to store whole datasets only, so a job that died inside a
    # 5-fold dataset lost every finished fold). The identity carries
    # 'selection' so the partial files of the runs before 2026-10-05 -- which
    # held the old private tiers, five models and three hybrids -- are not
    # read back.
    from experiments._checkpoint import script_checkpoint, exit_if_failed
    _ck = script_checkpoint(
        'opig_scaffold_hybrid',
        os.path.join(args.output_dir, 'opig_scaffold_hybrid_summary.json'),
        args, exclude=('datasets', 'output_dir', 'budgets'),
        # Bumped 2026-10-05 (second pass): fold records written before it
        # took the paired number from list position, not from base1/hybrid1.
        selection='fold_standard_selection_2_base_2_general_hybrids_'
                  '2026_10_05_slot_paired')
    _failed = []

    for ds in args.datasets:
        try:
            block = _run_one_dataset(ds, args, override, config, t0, _ck)
        except Exception as e:
            import traceback
            print(f'  {ds} FAILED: {type(e).__name__}: {e}', flush=True)
            traceback.print_exc()
            _failed.append(ds)
            continue
        summaries.append(summarise_one(ds, block))

    print_summary(summaries)
    spath = os.path.join(args.output_dir, 'opig_scaffold_hybrid_summary.json')
    with open(spath, 'w') as f:
        json.dump(summaries, f, indent=1, default=_json_default)
    print(f'\nwrote {spath}')
    exit_if_failed(_failed, 'dataset')


def _json_default(o):
    return (o.item() if isinstance(o, (np.floating, np.integer))
            else o.tolist() if isinstance(o, np.ndarray) else str(o))


def _run_one_dataset(ds, args, override, config, t0, ck):
    """One dataset: build the pool, run every outer fold, write its JSON.

    The outer folds come from kirby.find_best_hybrid.nested_cv_outer_folds at
    one count asked for -- args.max_folds, else config['n_splits']. The
    number of folds the split actually returned is what every per-fold
    search's store name, the checkpoint key and the result file carry (the
    splitter can return fewer when groups are fewer). A count of 1 is a single
    grouped held-out split, and every mean over folds then says 'one fold, no
    spread' rather than dividing by a spread of nothing.

    The dataset's JSON is rewritten after EVERY fold (dirty-run bug 12), and
    each finished fold is also stored in the checkpoint.
    """
    from experiments._shared import load_data, _resolve_search_metric
    from experiments.decided_grid import decided_models
    from kirby.find_best_hybrid import (nested_cv_outer_folds,
                                        BUILTIN_METRICS)

    pool = base_pool_for(ds, override, smoke=args.smoke)
    print(f'\n=== {ds} | base pool = {len(pool)} representations + '
          f'{len(SCAFFOLD_REPS)} scaffold-diff ({time.time()-t0:.0f}s) ===',
          flush=True)
    data = load_data(ds, rep_preset='full', rep_names=pool)
    scaffold, n_scaf = build_scaffold_reps(data['heavy_seqs'],
                                           data['light_seqs'])
    labels = np.asarray(data['labels'])
    n_rows = len(labels)
    assert n_scaf == n_rows, (
        f'row misalignment: scaffold {n_scaf} vs labels {n_rows}')
    # The scaffold matrices are cast to the pipeline's representation dtype
    # rather than left at whatever scaffold_reps built (answer 18: float32 for
    # every representation, and nothing here converts to float64).
    from experiments._shared import as_rep_dtype
    data['reps'].update({k: as_rep_dtype(v) for k, v in scaffold.items()})
    reps = {r: data['reps'][r] for r in (pool + list(SCAFFOLD_REPS))
            if r in data['reps']}
    print(f'  representations built ({len(reps)}): scaffold-diff present = '
          f'{[r for r in SCAFFOLD_REPS if r in reps]}', flush=True)

    # The metric and task type the per-fold searches resolve, so the scores
    # here and the ranking inside each search are the same quantity.
    metric_name, task_type = _resolve_search_metric(
        labels, task_type=data['task_type'], metric='auto', domain='antibody',
        dataset_name=ds)
    metric_fn, lower_is_better = BUILTIN_METRICS[metric_name]

    # Only the dataset's two decided models (DECISIONS.md Step 1). knn is not
    # one of them anywhere, and there is no distance search here to run it on.
    models = [m for m in decided_models(ds, TIER) if m != 'knn']

    # ONE setting for the outer fold count asked for: --max-folds, else the
    # config's n_splits.
    n_requested = int(args.max_folds or config['n_splits'])
    groups = data.get('groups')
    # The groups are load_data's: for the two koenig sets that is one group
    # per mutated position since 2026-10-05 (antibody_baseline_survey,
    # strategy 'position'; 225 positions x 19 variants), replacing the
    # modulo-5 grouping that gave exactly 5 groups for 5 folds. Nothing here
    # regroups them.
    folds = nested_cv_outer_folds(reps, labels, np.asarray(groups)
                                  if groups is not None else None,
                                  n_requested, SEED, task_type)
    # 2026-10-05: the count that RAN goes into every per-fold store name, the
    # checkpoint key, the result file and the one-fold note. The splitter
    # reduces the count to the number of groups when there are fewer groups
    # than folds asked for, and a search filed under a 5-fold name must not
    # hold a 3-fold split's training rows. (Not reachable on these four
    # datasets today: 225 to 267 groups each.)
    n_outer = len(folds)
    if n_outer != n_requested:
        print(f'  {ds}: {n_requested} outer folds asked for, the split gave '
              f'{n_outer} (fewer groups than folds)', flush=True)

    block = {'dataset': ds, 'tier': TIER, 'seed': SEED,
             'models': models,
             'metric_name': metric_name,
             'lower_is_better': bool(lower_is_better),
             'task_type': task_type,
             'n_outer_folds': int(n_outer),
             'n_outer_folds_requested': int(n_requested),
             # Keys the stored per-fold searches (see SEARCH_DISTANCE_METRIC);
             # nothing here measures a distance.
             'distance_metric': str(_get_distance_metric()),
             'n_rows': int(n_rows),
             'n_groups': (None if groups is None
                          else int(len(set(np.asarray(groups).tolist())))),
             'representations': sorted(reps),
             'scaffold_reps': list(SCAFFOLD_REPS),
             'selection_scope': 'fold_train_rows',
             'rule': ('per decided model per outer fold, chosen on that '
                      'fold\'s training rows: the 2 best base '
                      'representations and the 2 best general hybrids, plus '
                      'each scaffold-diff representation scored alone. No '
                      'distance or similarity hybrid (Adelaide 2026-10-05).'),
             'per_fold': []}
    if n_outer == 1:
        block['spread_note'] = ONE_FOLD_NOTE
    out_path = os.path.join(args.output_dir, f'opig_scaffold_hybrid_{ds}.json')

    for k, (tr, te) in enumerate(folds):
        ck_key = dict(dataset=ds, fold=int(k), n_outer_folds=int(n_outer))
        if ck is not None and ck.done(**ck_key):
            block['per_fold'].append(ck.get(**ck_key))
            print(f'  fold {k}: finished in an earlier attempt, read back',
                  flush=True)
            continue
        print(f'\n  --- {ds} fold {k} of {len(folds)} '
              f'({time.time()-t0:.0f}s) ---', flush=True)
        fold_rec = run_fold(
            ds, k, np.asarray(tr), np.asarray(te), reps, labels, groups,
            models=models, n_outer=n_outer, task_type=task_type,
            metric_name=metric_name, metric_fn=metric_fn,
            lower_is_better=lower_is_better)
        block['per_fold'].append(fold_rec)
        # Written after EVERY fold, not once at the end (dirty-run bug 12).
        with open(out_path, 'w') as f:
            json.dump(block, f, indent=1, default=_json_default)
        print(f'  wrote {out_path} after fold {k} '
              f'({time.time()-t0:.0f}s)', flush=True)
        # A fold whose search failed is in the JSON but not stored, so the
        # next attempt searches it again.
        if ck is not None and not fold_rec['search_failures']:
            from experiments._shared import convert_numpy
            ck.put(convert_numpy(fold_rec), **ck_key)

    return block


if __name__ == '__main__':
    main()
