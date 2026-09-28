#!/usr/bin/env python3
"""OPIG: does the representation a generator's predictor sees change its molecules?

Paper: adaptive hybrid representations (a downstream use, sibling to the
federated setting). Filed with the OPIG collaboration scripts, flat in
tests/experiments/, by the owner on 2026-09-28. Its support files -- REINVENT's
published configuration and the scorer REINVENT calls -- live in
tests/experiments/opig_generator_scoring/.

THE CLAIM. A molecule generator keeps a property predictor in its loop. That
predictor is handed one fingerprint by convention. Swap the features it sees,
change nothing else, and measure the molecules that come out.

THE GENERATOR IS NOT WRITTEN HERE. It is REINVENT 4, Apache 2.0, pinned at
commit 660d2c9cec9ea1ced1b452394fd38452e478fc8b. Its published reinforcement
learning configuration is copied byte for byte into reinvent_published/ and not
edited. Their scoring plugin, ExternalProcess, calls kirby_reinvent_score.py
through a pipe, so no line of REINVENT changes in any arm.

ARMS. Every arm fits the SAME model class on the SAME molecules with the SAME
generator settings. Only the features differ.

  B  ecfp4, every feature. The convention the claim is about. THE CONTROL.
  C  the blueprint find_best_hybrid picks, objective='predictive'.
  D  a random Gaussian matrix the same width as C's blueprint. C is credited
     over D or not at all.
  E  whichever single representation scores best alone on the steering dataset.
     Not in the brief; added so C has to beat the strongest fixed choice and not
     only a convention.

  ⛔ ARM A IS DROPPED, by the owner on 2026-09-28, with the reason. REINVENT's
  published scoring function rewards molecular weight between 200 and 500
  daltons, scores 14 substructures as zero, and rewards drug-likeness. It
  predicts no measured property. An arm steered by it would differ from B in two
  ways at once, not one, so its number could not be read. Their three published
  components are KEPT in every arm; each arm adds one predictor beside them.

THE PROPERTY AND THE DIRECTION. The generator is steered to MAXIMISE predicted
hERG pKi. High hERG affinity is not a desirable property in a drug. The
direction is chosen because the thing that scores the output molecules is a
hERG blocker classifier, so the steered property has to be hERG affinity. Both
sides move the same way, and every arm moves the same way.

THE CONFOUND, AND WHAT IS DONE ABOUT IT. A model that both steers the generator
and scores its output makes a better steering model win by definition. So the
output is scored by a model fitted on DIFFERENT MOLECULES: herg, the FLuID
binary blocker set. Measured on 2026-09-28: herg_ki holds 1,412 molecules and
herg holds 7,772 molecules; 166 molecules appear in both. Those 166 come out of
herg before the scoring model is fitted, leaving 7,606 molecules, and herg_ki
keeps all 1,412 of its own.

Usage:
    # local: fit everything, write the bundles and the per-arm TOML files
    python opig_generator_scoring_swap.py --stage fit
    # server: run the generator (one command per arm, printed by --stage fit)
    # local or server: read the generator's output and score it
    python opig_generator_scoring_swap.py --stage collect
"""
import argparse
import json
import os
import pickle
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))   # tests/
sys.path.insert(0, _HERE)                        # tests/experiments/

from experiments._shared import (
    load_data, get_domain_metric, save_experiment_results, convert_numpy,
    DOWNSTREAM_MODELS, FULL_BUDGETS, make_cv_splitter,
)

EXPERIMENT_ID = 'opig_generator_scoring_swap'

# The dataset the steering predictor is fitted on.
STEER_DATASET = 'herg_ki'
# The dataset the scoring model is fitted on. Different molecules, different
# assay, different label type.
ORACLE_DATASET = 'herg'

# One model class for every arm, so two arms differ only in their features.
# Named explicitly: a search told nothing fits only the fast-tier models, and a
# later lookup for a model it never fitted returns a blank, silently
# (_shared.py DOWNSTREAM_MODELS comment).
STEERING_MODEL = 'lightgbm'

# The scoring model is reported on TWO representations, so the result does not
# hang on one choice. ecfp4 is also what arm B steers on; that is stated rather
# than avoided, and the second representation is picked from herg's own ten by
# its own held-out score, with no arm involved in the choice.
ORACLE_FIXED_REP = 'ecfp4'
ORACLE_MODEL = 'lightgbm'

# REINVENT 4, pinned. Their configuration is copied into reinvent_published/
# unchanged; this is the commit it was copied from.
REINVENT_COMMIT = '660d2c9cec9ea1ced1b452394fd38452e478fc8b'
SUPPORT_DIR = os.path.join(_HERE, 'opig_generator_scoring')
PUBLISHED_DIR = os.path.join(SUPPORT_DIR, 'reinvent_published')

# The payload key kirby_reinvent_score.py writes and the TOML reads. One string,
# in one place, because a mismatch makes REINVENT raise on the first step.
SCORE_PROPERTY = 'kirby_herg_pki'

# The transform REINVENT applies to our raw pKi to put it in [0, 1]. Their own
# scoring framework owns every transform, so ours is declared in their TOML and
# not computed here. The bounds are herg_ki's own pKi range, printed by
# --stage fit so they are never a guessed pair.
# ⛔ set from the training labels, not typed in.


def _repo_root():
    return os.path.abspath(os.path.join(_HERE, '..', '..'))


# ---------------------------------------------------------------------------
# molecules
# ---------------------------------------------------------------------------

def _inchikeys(smiles):
    """One InChIKey per SMILES, None where RDKit cannot parse it."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(Chem.MolToInchiKey(m) if m is not None else None)
    return out


def _disjoin_oracle(steer_smiles, oracle_data):
    """Drop from the oracle's molecules every molecule the steering set holds.

    Returns (keep_mask, report). The report carries the three counts that make
    the confound checkable, each with what it counts named.
    """
    steer_keys = set(k for k in _inchikeys(steer_smiles) if k)
    oracle_keys = _inchikeys(oracle_data['smiles'])
    keep = np.array([(k is not None) and (k not in steer_keys)
                     for k in oracle_keys], dtype=bool)
    shared = sum(1 for k in oracle_keys if k and k in steer_keys)
    unparsed = sum(1 for k in oracle_keys if k is None)
    report = {
        'molecules_in_steering_set': len(steer_keys),
        'molecules_in_oracle_set': len(oracle_keys),
        'molecules_in_both_sets': int(shared),
        'oracle_molecules_rdkit_could_not_parse': int(unparsed),
        'molecules_left_to_fit_the_oracle_on': int(keep.sum()),
    }
    return keep, report


# ---------------------------------------------------------------------------
# the model that scores the output molecules
# ---------------------------------------------------------------------------

def _fit_oracle(oracle_data, keep, rep_name, seed):
    """Fit one blocker classifier on the oracle's molecules, minus the shared ones.

    Also returns its own held-out score, scaffold-grouped, so the reader can see
    how good the thing doing the judging is. Nothing about the generator or the
    arms enters here.
    """
    from sklearn.preprocessing import StandardScaler
    from experiments._shared import get_model_factories
    from sklearn.metrics import roc_auc_score

    X = np.asarray(oracle_data['reps'][rep_name], dtype=float)[keep]
    y = np.asarray(oracle_data['labels'])[keep]
    groups = np.asarray(oracle_data['groups'])[keep]

    splitter, use_groups = make_cv_splitter(
        5, groups=groups, seed=seed, task_type='classification')
    fold_auc = []
    factory = get_model_factories([ORACLE_MODEL], 'classification',
                                  n_samples=len(y))[ORACLE_MODEL]
    split_args = (X, y, groups) if use_groups else (X, y)
    for tr, va in splitter.split(*split_args):
        sc = StandardScaler().fit(X[tr])
        m = factory()
        m.fit(sc.transform(X[tr]), y[tr])
        p = m.predict_proba(sc.transform(X[va]))[:, 1]
        fold_auc.append(float(roc_auc_score(y[va], p)))

    scaler = StandardScaler().fit(X)
    model = factory()
    model.fit(scaler.transform(X), y)
    return {
        'rep_name': rep_name,
        'model': model,
        'scaler': scaler,
        'n_molecules_fitted_on': int(len(y)),
        'n_blockers': int(np.sum(y == 1)),
        'n_non_blockers': int(np.sum(y == 0)),
        'heldout_auroc_per_fold': fold_auc,
        'heldout_auroc_mean': float(np.mean(fold_auc)),
        'heldout_auroc_spread': float(np.max(fold_auc) - np.min(fold_auc)),
        # How many features the oracle was fitted on. _score_with_oracle
        # refuses a batch whose representation comes back a different width,
        # the same check kirby_reinvent_score.py makes on the steering side.
        'n_features_fitted_on': int(X.shape[1]),
    }


def _best_single_rep(data, rep_names, metric_fn, lower_is_better, seed):
    """Score every representation on its own and return the winner's name.

    Used twice: to choose arm E's representation on the steering molecules, and
    to choose the oracle's second representation on the oracle's molecules. No
    representation is cut — all of them are scored and all scores are reported.
    """
    from sklearn.preprocessing import StandardScaler
    from experiments._shared import get_model_factories

    y = np.asarray(data['labels'])
    groups = np.asarray(data['groups'])
    is_clf = data['task_type'] == 'classification'
    model_name = ORACLE_MODEL if is_clf else STEERING_MODEL
    factory = get_model_factories([model_name],
                                  data['task_type'],
                                  n_samples=len(y))[model_name]
    splitter, use_groups = make_cv_splitter(
        5, groups=groups, seed=seed, task_type=data['task_type'])

    scores = {}
    for rep in rep_names:
        if rep not in data['reps']:
            scores[rep] = None
            continue
        X = np.asarray(data['reps'][rep], dtype=float)
        fold = []
        split_args = (X, y, groups) if use_groups else (X, y)
        for tr, va in splitter.split(*split_args):
            sc = StandardScaler().fit(X[tr])
            m = factory()
            m.fit(sc.transform(X[tr]), y[tr])
            if is_clf:
                pred = m.predict_proba(sc.transform(X[va]))[:, 1]
            else:
                pred = m.predict(sc.transform(X[va]))
            fold.append(float(metric_fn(y[va], pred)))
        scores[rep] = {'per_fold': fold, 'mean': float(np.mean(fold))}

    ranked = [(n, s['mean']) for n, s in scores.items() if s]
    if not ranked:
        raise RuntimeError('no representation scored; nothing to rank')
    ranked.sort(key=lambda t: t[1], reverse=not lower_is_better)
    return ranked[0][0], scores


# ---------------------------------------------------------------------------
# the features each arm's predictor sees
# ---------------------------------------------------------------------------

def _arm_features(arm, budget, data, rep_names, metric_fn, lower_is_better,
                  seed, search_result=None, pool_reps=None):
    """Return (X, feature_info, rep_names_used, note) for one arm.

    feature_info is what kirby_reinvent_score.py replays on generated
    molecules through apply_feature_selection, so every arm must produce one.
    """
    from kirby.hybrid import create_hybrid
    from kirby.find_best_hybrid import apply_blueprint

    y = np.asarray(data['labels'])
    groups = np.asarray(data['groups'])

    if arm == 'B':
        X = np.asarray(data['reps'][ORACLE_FIXED_REP], dtype=float)
        info = {ORACLE_FIXED_REP: {
            'selected_indices': np.arange(X.shape[1], dtype=int)}}
        return X, info, [ORACLE_FIXED_REP], 'every feature of ecfp4'

    if arm == 'E':
        best, _ = _best_single_rep(data, rep_names, metric_fn,
                                   lower_is_better, seed)
        X = np.asarray(data['reps'][best], dtype=float)
        info = {best: {'selected_indices': np.arange(X.shape[1], dtype=int)}}
        return X, info, [best], f'every feature of {best}'

    if arm == 'C':
        row = search_result.best_by_model_budget[STEERING_MODEL][budget]
        bp = row['blueprint']
        X, info = apply_blueprint(
            {k: np.asarray(v, dtype=float) for k, v in data['reps'].items()},
            bp, labels=y, groups=groups)
        return X, info, list(bp.reps), (
            f"blueprint: budget {budget} features, allocation "
            f"{bp.allocation_method}, importance {bp.importance_method}")

    if arm == 'D':
        # ⛔ THE BRIEF ASKED FOR A RANDOM GAUSSIAN MATRIX AND THAT CANNOT BE
        # RUN HERE. A matrix of noise has no rule mapping a molecule to a row,
        # so a predictor fitted on one cannot score a molecule the generator
        # invents. Every arm has to featurise unseen molecules, so this arm
        # takes the same NUMBER of features as arm C, drawn at random from the
        # same representations arm C drew from. That is the null the brief
        # wanted: it holds the width and the representations fixed and removes
        # only the search's choice of which features. Decided 2026-09-28 under
        # 'do what is most scientifically sound'; the owner can swap it for
        # s11's random Gaussian PROJECTION of the concatenated representations,
        # which is a different null and tests compression instead of selection.
        rng = np.random.RandomState(seed)
        # pool_reps is arm C's own representations. Without it the draw ran
        # over every representation the dataset offers, which changes WHICH
        # representations the predictor sees as well as which features, so the
        # arm stopped being the null this docstring describes.
        draw_from = sorted(pool_reps) if pool_reps else sorted(data['reps'])
        missing = [r for r in draw_from if r not in data['reps']]
        if missing:
            raise ValueError(
                f"arm D was asked to draw from {sorted(missing)}, which the "
                f"loaded representations do not hold")
        pool = []
        for rep in draw_from:
            width = np.asarray(data['reps'][rep]).shape[1]
            pool.extend((rep, j) for j in range(width))
        width = int(budget)
        if width > len(pool):
            raise ValueError(
                f"arm D asked for {width} features but the representations hold "
                f"{len(pool)} in total")
        picked = [pool[i] for i in rng.choice(len(pool), size=width,
                                              replace=False)]
        info = {}
        for rep, j in picked:
            info.setdefault(rep, {'selected_indices': []})
            info[rep]['selected_indices'].append(j)
        for rep in info:
            info[rep]['selected_indices'] = np.asarray(
                sorted(info[rep]['selected_indices']), dtype=int)
        from kirby.hybrid import apply_feature_selection
        X = apply_feature_selection(
            {k: np.asarray(v, dtype=float) for k, v in data['reps'].items()},
            info, base_only=True)
        return X, info, sorted(info), (
            f'{width} features chosen at random from the same '
            f'{len(info)} representations, matched to arm C\'s width')

    raise ValueError(f'unknown arm {arm!r}')


def _write_bundle(path, model, scaler, feature_info, rep_names, rep_spec,
                  rep_widths):
    """Write everything kirby_reinvent_score.py needs to score a new molecule.

    rep_widths is how many features each representation held when the
    predictor was fitted. The scorer refuses a batch whose representation
    comes back a different width, because the stored column numbers would
    then point at different features.
    """
    with open(path, 'wb') as fh:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_info': feature_info,
            'rep_names': rep_names,
            'rep_spec': rep_spec,
            'rep_widths': rep_widths,
        }, fh)


# ---------------------------------------------------------------------------
# the generator's configuration
# ---------------------------------------------------------------------------

def _arm_toml(arm_name, bundle_path, python_exe, score_low, score_high,
              out_dir, seed):
    """Their published staged_learning.toml with ONE thing added.

    Their two stages, their batch size, their sigma, their learning rate, their
    diversity filter and their termination criteria are copied out of
    reinvent_published/staged_learning.toml unchanged. Their three scoring
    components are copied unchanged too. This adds one more component, which
    calls kirby_reinvent_score.py and nothing else.
    """
    published = open(os.path.join(PUBLISHED_DIR, 'staged_learning.toml')).read()
    stage1 = open(os.path.join(PUBLISHED_DIR, 'stage1_scoring.toml')).read()
    stage2 = open(os.path.join(PUBLISHED_DIR, 'stage2_scoring.toml')).read()

    # The transform bounds are herg_ki's own pKi range, passed in, never typed.
    component = f'''
[[component]]
[component.ExternalProcess]

[[component.ExternalProcess.endpoint]]
name = "KIRBy hERG pKi ({arm_name})"
weight = 1.0
params.executable = "{python_exe}"
params.args = "{os.path.join(SUPPORT_DIR, 'kirby_reinvent_score.py')} --bundle {bundle_path} --property {SCORE_PROPERTY}"
params.property = "{SCORE_PROPERTY}"
transform.type = "sigmoid"
transform.low = {score_low:.4f}
transform.high = {score_high:.4f}
transform.k = 0.5
'''
    for name, body in (('stage1', stage1), ('stage2', stage2)):
        with open(os.path.join(out_dir, f'{arm_name}_{name}_scoring.toml'), 'w') as fh:
            fh.write(f'### {name} scoring: REINVENT 4 published components, '
                     f'unchanged, plus one KIRBy predictor\n')
            fh.write(body.rstrip() + '\n' + component)

    cfg = published

    # Their stage 1 lists its scoring components INSIDE staged_learning.toml,
    # while their stage 2 reads a file. Replacing only the stage 2 file name
    # left stage 1 running their three published components and no KIRBy
    # predictor, identically in every arm, for the first 25 to 100 steps. The
    # arm's stage 1 now reads its own scoring file, the way their stage 2 does.
    head, sep, tail = cfg.partition('[[stage.scoring.component]]')
    if not sep:
        raise RuntimeError(
            "staged_learning.toml holds no inline stage 1 scoring component, "
            "so the stage 1 block cannot be pointed at a file. Their published "
            "configuration has changed shape since commit "
            f"{REINVENT_COMMIT}.")
    marker = '### Stage 2'
    cut = tail.find(marker)
    if cut < 0:
        raise RuntimeError(
            "staged_learning.toml holds no '### Stage 2' marker, so the end of "
            "the inline stage 1 scoring block cannot be found. Their published "
            f"configuration has changed shape since commit {REINVENT_COMMIT}.")
    cfg = (head
           + f'filename = "{arm_name}_stage1_scoring.toml"\n'
           + 'filetype = "toml"\n\n\n'
           + tail[cut:])

    cfg = cfg.replace('summary_csv_prefix = "staged_learning"',
                      f'summary_csv_prefix = "{arm_name}"')
    cfg = cfg.replace("chkpt_file = 'test1.chkpt'",
                      f"chkpt_file = '{arm_name}_stage1.chkpt'")
    cfg = cfg.replace("chkpt_file = 'test2.chkpt'",
                      f"chkpt_file = '{arm_name}_stage2.chkpt'")
    cfg = cfg.replace('filename = "stage2_scoring.toml"',
                      f'filename = "{arm_name}_stage2_scoring.toml"')
    cfg = cfg.replace('tb_logdir = "tb_logs"', f'tb_logdir = "tb_{arm_name}"')
    cfg = cfg.replace('json_out_config = "_staged_learning.json"',
                      f'json_out_config = "_{arm_name}.json"')
    header = (f'# GENERATED from reinvent_published/staged_learning.toml at '
              f'REINVENT 4 commit {REINVENT_COMMIT}.\n'
              f'# Only these changed: the file name prefixes, the stage 1 '
              f'scoring block, and the stage 2 scoring file name.\n'
              f'# Arm {arm_name}, seed {seed}.\n')
    path = os.path.join(out_dir, f'{arm_name}.toml')
    with open(path, 'w') as fh:
        fh.write(header + cfg)
    return path


# ---------------------------------------------------------------------------
# reading the molecules back
# ---------------------------------------------------------------------------

def _read_generated(out_dir, arm_name):
    """Every molecule REINVENT wrote for one arm, with its own validity flag.

    Their csv_summmary.py writes header
    ["Agent", "Prior", "Target", "Score", "SMILES", "SMILES_state"] to
    {prefix}_{stage}.csv, and SmilesState is 0 invalid, 1 valid, 2 duplicate
    (reinvent/models/model_factory/sample_batch.py). So the valid and unique
    counts are theirs, not recomputed here.
    """
    import csv
    rows = []
    stage = 1
    while True:
        path = os.path.join(out_dir, f'{arm_name}_{stage}.csv')
        if not os.path.exists(path):
            break
        with open(path) as fh:
            for r in csv.DictReader(fh):
                rows.append({'smiles': r['SMILES'],
                             'state': int(r['SMILES_state']),
                             'stage': stage})
        stage += 1
    return rows


def _score_with_oracle(smiles, oracle):
    """Blocker probability from one oracle, one number per SMILES.

    Molecules the oracle's representation cannot be built for are dropped and
    counted, never scored as zero.
    """
    from find_best_hybrid_molecular import get_representations

    reps = get_representations(list(smiles), oracle['rep_name'], cache=False)
    X = np.asarray(reps[oracle['rep_name']], dtype=float)
    if X.shape[0] != len(smiles):
        raise RuntimeError(
            f"{oracle['rep_name']} came back with {X.shape[0]} rows for "
            f"{len(smiles)} molecules, so a score cannot be matched to a "
            f"molecule")
    # The same width check the steering scorer makes. graph_kernel is one of
    # herg's ten and can win the second oracle, and it builds its columns from
    # whatever molecules it is handed, so a batch of generated molecules is a
    # different column space from the molecules the oracle was fitted on.
    want = oracle.get('n_features_fitted_on')
    if want is not None and X.shape[1] != want:
        raise RuntimeError(
            f"{oracle['rep_name']} is {X.shape[1]} features wide for these "
            f"{len(smiles)} molecules and was {want} wide when the oracle was "
            f"fitted, so its columns do not mean the same features and these "
            f"molecules cannot be scored")
    ok = np.isfinite(X).all(axis=1) & (np.abs(X).sum(axis=1) > 0)
    p = np.full(len(smiles), np.nan)
    if ok.any():
        p[ok] = oracle['model'].predict_proba(
            oracle['scaler'].transform(X[ok]))[:, 1]
    return p, int((~ok).sum())


def _report_arm(arm_name, rows, oracles, train_keys):
    """Every number for one arm, each with what it counts named beside it."""
    n_generated = len(rows)
    valid = [r for r in rows if r['state'] == 1]
    duplicate = [r for r in rows if r['state'] == 2]
    invalid = [r for r in rows if r['state'] == 0]

    uniq = {}
    for r in valid:
        uniq.setdefault(r['smiles'], r)
    unique_smiles = list(uniq)

    keys = _inchikeys(unique_smiles)
    novel = [s for s, k in zip(unique_smiles, keys)
             if k is not None and k not in train_keys]

    out = {
        'arm': arm_name,
        'molecules_generated': n_generated,
        'molecules_valid_of_generated': [len(valid), n_generated],
        'molecules_invalid_of_generated': [len(invalid), n_generated],
        'molecules_flagged_duplicate_of_generated': [len(duplicate), n_generated],
        'molecules_unique_and_valid_of_generated': [len(unique_smiles), n_generated],
        'molecules_novel_against_training_of_unique': [len(novel), len(unique_smiles)],
        'by_oracle': {},
    }

    for tag, oracle in oracles.items():
        if not unique_smiles:
            out['by_oracle'][tag] = {'note': 'no valid unique molecule to score'}
            continue
        p, n_unscored = _score_with_oracle(unique_smiles, oracle)
        # The same numbers over the molecules that are not already in the
        # steering set. A molecule the generator rediscovered from its own
        # predictor's training molecules is a measured hERG binder, so it
        # scores high under the oracle without anything having been invented,
        # and the arms do not rediscover at the same rate: each arm's predictor
        # is fitted on those 1,412 molecules through a different set of
        # features. Both pools are reported; neither replaces the other.
        novel_set = set(novel)
        is_novel = np.array([s in novel_set for s in unique_smiles], dtype=bool)
        finite = p[np.isfinite(p)]
        finite = np.sort(finite)[::-1]
        novel_p = p[is_novel & np.isfinite(p)]
        novel_p = np.sort(novel_p)[::-1]
        entry = {
            'oracle_representation': oracle['rep_name'],
            'oracle_heldout_auroc_mean': oracle['heldout_auroc_mean'],
            'molecules_scored': int(len(finite)),
            'molecules_scored_that_are_novel': int(len(novel_p)),
            'molecules_the_oracle_could_not_featurise': n_unscored,
        }
        if len(novel_p):
            entry['mean_blocker_probability_over_every_scored_novel_molecule'] = float(
                np.mean(novel_p))
            entry['median_blocker_probability_over_every_scored_novel_molecule'] = float(
                np.median(novel_p))
            entry['blocker_probability_of_the_single_best_novel_molecule'] = float(
                novel_p[0])
            for k in (10, 100):
                take = novel_p[:k]
                entry[f'mean_blocker_probability_of_the_best_{k}_novel_molecules'] = float(
                    np.mean(take))
                entry[f'novel_molecules_actually_in_that_best_{k}'] = int(len(take))
        if len(finite):
            # The best-of-k numbers below rise with the number of molecules an
            # arm produced, and the arms do not all produce the same number:
            # REINVENT stops a stage when the total score passes its threshold,
            # and that score contains the arm's own predictor. These two are
            # over every scored molecule, so they do not move with pool size.
            entry['mean_blocker_probability_over_every_scored_molecule'] = float(
                np.mean(finite))
            entry['median_blocker_probability_over_every_scored_molecule'] = float(
                np.median(finite))
            entry['blocker_probability_of_the_single_best_molecule'] = float(finite[0])
            for k in (10, 100):
                take = finite[:k]
                entry[f'mean_blocker_probability_of_the_best_{k}_molecules'] = float(np.mean(take))
                entry[f'molecules_actually_in_that_best_{k}'] = int(len(take))
        out['by_oracle'][tag] = entry
    return out


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_fit(args):
    from sklearn.preprocessing import StandardScaler
    from kirby.find_best_hybrid import find_best_hybrid
    from experiments._shared import get_model_factories
    from experiments.complexity.complexity_helpers import DATASET_REPS

    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)

    if args.reps:
        steer_reps = oracle_reps = [r.strip() for r in args.reps.split(',')]
        print('⛔ SMOKE RUN: representations overridden on the command line, '
              'so these are not the pruned ten either dataset was given.')
    else:
        steer_reps = DATASET_REPS[STEER_DATASET]['general']
        oracle_reps = DATASET_REPS[ORACLE_DATASET]['general']
    print(f"\nsteering molecules: {STEER_DATASET}, "
          f"{len(steer_reps)} representations")
    print(f"oracle molecules:   {ORACLE_DATASET}, "
          f"{len(oracle_reps)} representations")

    steer = load_data(STEER_DATASET, rep_names=steer_reps)
    oracle_data = load_data(ORACLE_DATASET, rep_names=oracle_reps)
    metric_name, metric_fn, lower_is_better = get_domain_metric(steer)

    y = np.asarray(steer['labels'])
    groups = np.asarray(steer['groups'])
    print(f"\n{STEER_DATASET}: {len(y)} molecules, "
          f"{len(np.unique(groups))} scaffold groups, "
          f"metric {metric_name}, lower_is_better={lower_is_better}")
    print(f"{STEER_DATASET} label range: {float(y.min()):.4f} to "
          f"{float(y.max()):.4f} pKi")

    # --- the molecules the oracle is allowed to see -----------------------
    keep, overlap = _disjoin_oracle(steer['smiles'], oracle_data)
    print("\nMOLECULE OVERLAP, counted, not assumed:")
    for k, v in overlap.items():
        print(f"  {k.replace('_', ' ')}: {v}")

    # --- the models that score the output --------------------------------
    oracles = {}
    oracles['fixed_ecfp4'] = _fit_oracle(oracle_data, keep,
                                         ORACLE_FIXED_REP, args.seed)
    best_oracle_rep, oracle_rep_scores = _best_single_rep(
        {**oracle_data,
         'reps': {r: np.asarray(v)[keep]
                  for r, v in oracle_data['reps'].items()},
         'labels': np.asarray(oracle_data['labels'])[keep],
         'groups': np.asarray(oracle_data['groups'])[keep]},
        oracle_reps, *get_domain_metric(oracle_data)[1:], args.seed)
    if best_oracle_rep != ORACLE_FIXED_REP:
        oracles['best_single'] = _fit_oracle(oracle_data, keep,
                                            best_oracle_rep, args.seed)
    print("\nORACLE, fitted on molecules the steering model never sees:")
    for tag, o in oracles.items():
        print(f"  {tag}: {o['rep_name']}, fitted on "
              f"{o['n_molecules_fitted_on']} molecules "
              f"({o['n_blockers']} blockers, {o['n_non_blockers']} "
              f"non-blockers), held-out AUROC per fold "
              f"{[round(v, 4) for v in o['heldout_auroc_per_fold']]}")

    # --- arm C's blueprints, one per budget ------------------------------
    budgets = ([int(b) for b in args.budgets.split(',')]
               if args.budgets else list(FULL_BUDGETS))
    print(f"\nhybrid search: budgets {budgets}, models "
          f"{DOWNSTREAM_MODELS} named explicitly")
    search = find_best_hybrid(
        {k: np.asarray(v, dtype=float) for k, v in steer['reps'].items()},
        y, groups,
        budgets=budgets,
        models=DOWNSTREAM_MODELS,
        metric=metric_name,
        task_type=steer['task_type'],
        dataset_name=STEER_DATASET,
        seed=args.seed,
        cache=True,
        # 'general' is the lineage herg_ki's pruned ten representations are
        # precomputed under, so this call can be served by the precompute
        # instead of searching cold.
        tier='general',
    )

    python_exe = sys.executable
    factory = get_model_factories([STEERING_MODEL], steer['task_type'],
                                 n_samples=len(y))[STEERING_MODEL]

    manifest = {
        'experiment': EXPERIMENT_ID,
        'reinvent_commit': REINVENT_COMMIT,
        'steering_dataset': STEER_DATASET,
        'oracle_dataset': ORACLE_DATASET,
        'steering_model': STEERING_MODEL,
        'molecule_overlap': overlap,
        'oracle_single_representation_scores': oracle_rep_scores,
        'oracles': {t: {k: v for k, v in o.items()
                        if k not in ('model', 'scaler')}
                    for t, o in oracles.items()},
        'metric_name': metric_name,
        'lower_is_better': lower_is_better,
        'seed': args.seed,
        'arms': {},
    }

    # Arm C first at each budget, because arm D's width is whatever C's
    # allocation actually handed out, which can be below the budget.
    plan = ([('B', None), ('E', None)]
            + [('C', b) for b in budgets]
            + [('D', b) for b in budgets])
    c_widths = {}
    c_reps = {}
    commands = []
    for arm, budget in plan:
        name = arm if budget is None else f'{arm}_b{budget}'
        X, info, rep_names, note = _arm_features(
            arm, budget, steer, steer_reps, metric_fn, lower_is_better,
            args.seed, search_result=search,
            pool_reps=c_reps.get(budget) if arm == 'D' else None)
        if arm == 'C':
            c_widths[budget] = X.shape[1]
            # The representations that actually received a feature, which is
            # what arm D draws from. feature_info also carries entries that
            # are not representations at all ('allocation', 'greedy_history',
            # 'augmentation_info'), so take only the ones holding columns.
            c_reps[budget] = sorted(
                r for r, e in info.items()
                if isinstance(e, dict) and 'selected_indices' in e
                and len(np.asarray(e['selected_indices']).ravel()) > 0)
        if arm == 'D':
            # Arm C's allocation can hand out fewer features than the budget
            # asks for, so arm D is rebuilt at the width C actually produced.
            want = c_widths.get(budget)
            if want is not None and want != X.shape[1]:
                X, info, rep_names, note = _arm_features(
                    'D', want, steer, steer_reps, metric_fn, lower_is_better,
                    args.seed, pool_reps=c_reps.get(budget))

        scaler = StandardScaler().fit(X)
        model = factory()
        model.fit(scaler.transform(X), y)

        bundle = os.path.join(out_dir, f'{name}_bundle.pkl')
        _write_bundle(bundle, model, scaler, info, rep_names,
                      ','.join(rep_names),
                      {r: int(np.asarray(steer['reps'][r]).shape[1])
                       for r in rep_names})
        toml = _arm_toml(name, bundle, python_exe,
                         float(np.percentile(y, 10)),
                         float(np.percentile(y, 90)),
                         out_dir, args.seed)
        manifest['arms'][name] = {
            'features_the_predictor_sees': int(X.shape[1]),
            'representations_used': rep_names,
            'note': note,
            'bundle': bundle,
            'config': toml,
        }
        commands.append(f'reinvent -l {name}.log {os.path.basename(toml)}')
        print(f"  arm {name}: {X.shape[1]} features — {note}")

    with open(os.path.join(out_dir, 'manifest.json'), 'w') as fh:
        json.dump(convert_numpy(manifest), fh, indent=2)
    with open(os.path.join(out_dir, 'run_arms.sh'), 'w') as fh:
        fh.write('#!/bin/bash\nset -euo pipefail\n'
                 f'# REINVENT 4 at commit {REINVENT_COMMIT}\n'
                 'cd "$(dirname "$0")"\n' + '\n'.join(commands) + '\n')
    print(f"\nwrote {len(manifest['arms'])} configs, "
          f"{len(manifest['arms'])} bundles and run_arms.sh to {out_dir}")
    return manifest


def stage_collect(args):
    out_dir = os.path.abspath(args.output)
    with open(os.path.join(out_dir, 'manifest.json')) as fh:
        manifest = json.load(fh)

    from experiments.complexity.complexity_helpers import DATASET_REPS
    _sr = ([r.strip() for r in args.reps.split(',')] if args.reps
           else DATASET_REPS[STEER_DATASET]['general'])
    _orr = ([r.strip() for r in args.reps.split(',')] if args.reps
            else DATASET_REPS[ORACLE_DATASET]['general'])
    steer = load_data(STEER_DATASET, rep_names=_sr)
    oracle_data = load_data(ORACLE_DATASET, rep_names=_orr)
    keep, _ = _disjoin_oracle(steer['smiles'], oracle_data)
    oracles = {t: _fit_oracle(oracle_data, keep, o['rep_name'], args.seed)
               for t, o in manifest['oracles'].items()}
    train_keys = set(k for k in _inchikeys(steer['smiles']) if k)

    per_arm = {}
    for name in manifest['arms']:
        rows = _read_generated(out_dir, name)
        if not rows:
            print(f"  arm {name}: no summary CSV found, not run yet")
            continue
        per_arm[name] = _report_arm(name, rows, oracles, train_keys)
        r = per_arm[name]
        print(f"\narm {name}")
        print(f"  molecules generated: {r['molecules_generated']}")
        for k in ('molecules_valid_of_generated',
                  'molecules_unique_and_valid_of_generated',
                  'molecules_novel_against_training_of_unique'):
            top, bottom = r[k]
            print(f"  {k.replace('_', ' ')}: {top} of {bottom}")

    results = {'meta': {'experiment_id': EXPERIMENT_ID,
                        'reinvent_commit': REINVENT_COMMIT},
               'manifest': manifest,
               'per_arm': per_arm}
    save_experiment_results(results, args.output, EXPERIMENT_ID,
                            dataset=STEER_DATASET)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['fit', 'collect'], default='fit')
    ap.add_argument('--output', default=os.path.join(
        _repo_root(), 'tests', 'results', 'opig', EXPERIMENT_ID))
    ap.add_argument('--budgets', default=None,
                    help='comma-separated; default is every FULL_BUDGETS value')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--reps', default=None,
                    help='comma-separated representations, overriding the '
                         'per-dataset pruned lists. SMOKE RUNS ONLY: a real '
                         'run uses the ten each dataset was pruned to.')
    args = ap.parse_args()
    if args.stage == 'fit':
        stage_fit(args)
    else:
        stage_collect(args)


if __name__ == '__main__':
    main()
