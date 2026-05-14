"""PT-Rep Phase 0 — offline smoke tests.

Exercises pt_registry / pt_quality / pt_leakage end-to-end on synthetic data.
No network, no heavy compute, no kirby-rep dependency — safe to run locally.

    python tests/experiments/pretrain_rep/smoke_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import warnings

# Put tests/ on the path so `experiments.*` resolves.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from experiments.pretrain_rep.pt_registry import (
    PTDataset, PT_SOURCES, list_pt_sources, get_pt_dataset)
from experiments.pretrain_rep.pt_quality import (
    apply_quality_pipeline, get_criteria, QualityCriteria, write_quality_report)
from experiments.pretrain_rep.pt_leakage import (
    inchikey_leakage_filter, sequence_leakage_filter, remove_pt_ft_leakage)
from experiments.pretrain_rep.pt_reps import pt_cache_key, compute_pt_reps


PASS, FAIL = 0, 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# =============================================================================
def test_registry():
    print("\n[registry]")
    names = list_pt_sources()
    check('list_pt_sources non-empty', len(names) > 0)
    check('synthetic sources registered',
          {'synthetic_molecular', 'synthetic_antibody'} <= set(names))
    check('domain filter works',
          all(PT_SOURCES[n].domain == 'molecular'
              for n in list_pt_sources('molecular')))

    mol = get_pt_dataset('molecular', 'synthetic_molecular', max_samples=40)
    check('synthetic molecular len', len(mol) == 40, f'got {len(mol)}')
    check('synthetic molecular task_type', mol.task_type == 'regression')
    check('synthetic molecular has smiles + keys + labels + groups',
          mol.smiles is not None and len(mol.dedup_keys) == 40
          and mol.labels is not None and mol.groups is not None)
    ents, labs, grps = mol.as_tuple()
    check('as_tuple shapes', len(ents) == 40 and len(labs) == 40 and len(grps) == 40)

    ab = get_pt_dataset('antibody', 'synthetic_antibody', max_samples=30)
    check('synthetic antibody len', len(ab) == 30, f'got {len(ab)}')
    check('synthetic antibody task_type', ab.task_type == 'classification')
    check('synthetic antibody heavy/light',
          ab.heavy_seqs is not None and ab.light_seqs is not None
          and len(ab.heavy_seqs) == 30)
    check('antibody dedup_key == heavy seq', ab.dedup_keys == ab.heavy_seqs)

    # validation errors
    try:
        get_pt_dataset('molecular', 'does_not_exist')
        check('unknown source raises', False)
    except KeyError:
        check('unknown source raises', True)
    try:
        get_pt_dataset('antibody', 'synthetic_molecular')
        check('domain mismatch raises', False)
    except ValueError:
        check('domain mismatch raises', True)

    # exclude_keys pre-filter
    keys_to_drop = mol.dedup_keys[:10]
    mol2 = get_pt_dataset('molecular', 'synthetic_molecular', max_samples=40,
                          exclude_keys=keys_to_drop)
    check('exclude_keys removed rows', len(mol2) == 30, f'got {len(mol2)}')
    check('exclude_keys removed the right rows',
          not (set(keys_to_drop) & set(mol2.dedup_keys)))

    # subsample shuffles and caps
    big = get_pt_dataset('molecular', 'synthetic_molecular', max_samples=64)
    sub = big.subsample(20, seed=7)
    check('subsample caps', len(sub) == 20)
    check('subsample records provenance',
          sub.meta.get('subsampled_from') == 64)

    # PTDataset invariant: length mismatch raises
    try:
        PTDataset(domain='molecular', source='x', task=None,
                  entities=['a', 'b'], dedup_keys=['k'])
        check('length-mismatch raises', False)
    except ValueError:
        check('length-mismatch raises', True)
    # unlabelled task_type consistency
    try:
        PTDataset(domain='molecular', source='x', task=None,
                  entities=['a'], dedup_keys=['k'], labels=None,
                  task_type='regression')
        check('labels=None with regression raises', False)
    except ValueError:
        check('labels=None with regression raises', True)


# =============================================================================
def test_quality():
    print("\n[quality]")
    # Build a deliberately dirty molecular corpus.
    entities = ['CCO', 'CCO',          # exact dup
                'CCN', '   ', '',      # empty / whitespace
                'CCC', 'CCCC', 'c1ccccc1', 'CN', 'CO']
    keys = ['K1', 'K1',                # dup InChIKey
            'K2', 'K3', 'K4',
            'K5', 'K6', 'K6',          # contradictory pair below
            'K7', 'K8']
    labels = np.array([6.0, 6.0,
                       7.0, 5.0, 5.0,
                       99.0,             # out-of-range
                       4.0, 8.0,         # K6 pair: spread 4.0 > tol 1.0
                       5.5, 3.0], dtype=float)
    groups = np.array([0, 0, 1, 1, 1, 2, 3, 3, 4, 5])
    dirty = PTDataset(domain='molecular', source='human_chembl', task=None,
                      entities=entities, dedup_keys=keys, labels=labels,
                      groups=groups, task_type='regression',
                      smiles=list(entities))
    clean, report = apply_quality_pipeline(dirty)
    check('quality removed dirty rows', len(clean) < len(dirty),
          f'{len(dirty)} -> {len(clean)}')
    check('no empty entities survive',
          all(e.strip() for e in clean.entities))
    # molecular dedups on (key, group, label) rows, not bare InChIKey — the
    # K1/K1/6.0/group0 pair is an exact-duplicate row and collapses to one.
    _rows = list(zip(clean.dedup_keys, clean.groups, clean.labels))
    check('no exact-duplicate rows survive', len(_rows) == len(set(_rows)))
    check('exact-dup K1 row collapsed', clean.dedup_keys.count('K1') == 1)
    check('out-of-range label removed', 99.0 not in list(clean.labels))
    check('contradictory K6 cluster removed',
          'K6' not in clean.dedup_keys)
    check('report has both layers',
          any(s['stage'] == 'layer1_total' for s in report.stages)
          and any(s['stage'] == 'layer2_total' for s in report.stages))
    check('report counts consistent',
          report.n_in == len(dirty) and report.n_out == len(clean))

    # Antibody quality: alphabet + length.
    aa_ok = 'ACDEFGHIKLMNPQRSTVWY' * 6        # 120 residues, valid
    aa_bad = 'ACDEFGHXKLMNPQRSTVWY' * 6       # contains X
    aa_short = 'ACDEFGHIKL'                   # too short
    ab = PTDataset(domain='antibody', source='oas_species', task=None,
                   entities=[aa_ok, aa_bad, aa_short, aa_ok],  # last is dup
                   dedup_keys=[aa_ok, aa_bad, aa_short, aa_ok],
                   labels=np.array([1, 0, 1, 1]),
                   task_type='classification',
                   heavy_seqs=[aa_ok, aa_bad, aa_short, aa_ok])
    ab_clean, ab_report = apply_quality_pipeline(ab)
    check('antibody quality keeps only valid seq',
          len(ab_clean) == 1 and ab_clean.entities[0] == aa_ok,
          f'got {len(ab_clean)}')

    # synthetic corpora: hygiene only, no label-range filter
    syn = get_pt_dataset('molecular', 'synthetic_molecular', max_samples=50)
    syn_clean, syn_report = apply_quality_pipeline(syn)
    check('synthetic_molecular passes quality (hygiene only)',
          len(syn_clean) > 0)
    crit = get_criteria('synthetic_molecular', 'molecular')
    check('synthetic_molecular criteria skips label range',
          crit.label_min is None and crit.contradiction_tol is None)

    # report serialisation
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'sub', 'report.json')
        write_quality_report(report, p)
        check('report written to disk', os.path.exists(p))

    # criteria resolution: unknown source -> domain default
    fallback = get_criteria('some_new_corpus', 'antibody')
    check('unknown source falls back to domain default',
          fallback.require_alphabet == 'aa')


# =============================================================================
def test_leakage():
    print("\n[leakage]")
    mol = get_pt_dataset('molecular', 'synthetic_molecular', max_samples=40)
    ft_keys = mol.dedup_keys[:12]
    filtered, rep = inchikey_leakage_filter(mol, ft_keys)
    check('inchikey filter removes overlap', len(filtered) == 28,
          f'got {len(filtered)}')
    check('inchikey filter report', rep['n_removed'] == 12)
    check('no leaked keys remain',
          not (set(ft_keys) & set(filtered.dedup_keys)))

    # wrong-domain guard
    ab = get_pt_dataset('antibody', 'synthetic_antibody', max_samples=20)
    try:
        inchikey_leakage_filter(ab, ['x'])
        check('inchikey filter rejects antibody', False)
    except ValueError:
        check('inchikey filter rejects antibody', True)

    # antibody exact-sequence leakage
    ft_seqs = ab.entities[:5]
    ab_filtered, ab_rep = sequence_leakage_filter(ab, ft_seqs,
                                                  identity_threshold=1.0)
    check('sequence exact filter removes overlap', len(ab_filtered) == 15,
          f'got {len(ab_filtered)}')
    check('sequence filter marks exact as validated',
          ab_rep['threshold_validated'] is True)

    # sub-1.0 threshold must warn (unvalidated, §9 Q6)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _, sub_rep = sequence_leakage_filter(ab, ft_seqs,
                                             identity_threshold=0.9)
        check('sub-1.0 threshold warns', len(w) >= 1)
        check('sub-1.0 threshold marked unvalidated',
              sub_rep['threshold_validated'] is False)

    # invalid threshold
    try:
        sequence_leakage_filter(ab, ft_seqs, identity_threshold=1.5)
        check('invalid threshold raises', False)
    except ValueError:
        check('invalid threshold raises', True)

    # dispatcher
    d_mol, _ = remove_pt_ft_leakage(mol, ft_keys=ft_keys)
    check('dispatcher routes molecular', len(d_mol) == 28)
    d_ab, _ = remove_pt_ft_leakage(ab, ft_sequences=ft_seqs)
    check('dispatcher routes antibody', len(d_ab) == 15)
    try:
        remove_pt_ft_leakage(mol)  # missing ft_keys
        check('dispatcher requires matching FT input', False)
    except ValueError:
        check('dispatcher requires matching FT input', True)


# =============================================================================
def test_reps():
    print("\n[reps]")
    mol = get_pt_dataset('molecular', 'synthetic_molecular', max_samples=24)

    # cache key: stable for identical content, sensitive to content change
    check('pt_cache_key stable', pt_cache_key(mol) == pt_cache_key(mol))
    mol_sub = mol.subsample(12, seed=1)
    check('pt_cache_key changes with corpus size',
          pt_cache_key(mol) != pt_cache_key(mol_sub))
    ab = get_pt_dataset('antibody', 'synthetic_antibody', max_samples=24)
    check('pt_cache_key changes across sources',
          pt_cache_key(mol) != pt_cache_key(ab))
    check('pt_cache_key embeds source + n',
          pt_cache_key(mol).startswith('pt__synthetic_molecular__')
          and '__n24__' in pt_cache_key(mol))

    # injected fake rep fns — exercises the orchestration without GPU/models
    fake_fns = {
        'fake_a': lambda d: np.ones((len(d), 4)),
        'fake_b': lambda d: np.arange(len(d) * 3, dtype=float).reshape(len(d), 3),
    }
    reps = compute_pt_reps(mol, ['fake_a', 'fake_b'], cache=False,
                           rep_fns=fake_fns, verbose=False)
    check('compute_pt_reps returns requested reps',
          set(reps) == {'fake_a', 'fake_b'})
    check('compute_pt_reps row counts match corpus',
          all(v.shape[0] == len(mol) for v in reps.values()))
    check('compute_pt_reps preserves rep dims',
          reps['fake_a'].shape[1] == 4 and reps['fake_b'].shape[1] == 3)

    # a rep fn that returns the wrong number of rows must be rejected
    bad_fns = {'bad': lambda d: np.ones((len(d) + 1, 2))}
    try:
        compute_pt_reps(mol, ['bad'], cache=False, rep_fns=bad_fns,
                        verbose=False)
        check('row-count mismatch raises', False)
    except ValueError:
        check('row-count mismatch raises', True)

    # a failing rep fn is skipped, not fatal
    mixed_fns = {
        'ok': lambda d: np.ones((len(d), 2)),
        'boom': lambda d: (_ for _ in ()).throw(RuntimeError('kaboom')),
    }
    reps2 = compute_pt_reps(mol, list(mixed_fns), cache=False,
                            rep_fns=mixed_fns, verbose=False)
    check('failing rep fn skipped, others survive', set(reps2) == {'ok'})

    # unknown domain raises
    weird = PTDataset(domain='reaction', source='x', task=None,
                      entities=['a'], dedup_keys=['k'])
    try:
        compute_pt_reps(weird, ['fake_a'], cache=False, rep_fns=None,
                        verbose=False)
        check('unknown domain raises', False)
    except ValueError:
        check('unknown domain raises', True)

    # cache round-trip: write under a temp KIRBY_CACHE_DIR, second call reads it
    try:
        with tempfile.TemporaryDirectory() as td:
            os.environ['KIRBY_CACHE_DIR'] = td
            key = 'pt__smoke__default__n24__deadbeef00'
            r1 = compute_pt_reps(mol, ['fake_a'], cache=True, cache_key=key,
                                 rep_fns=fake_fns, verbose=False)
            cached_file = os.path.join(td, 'rep_cache', key, 'fake_a.npy')
            check('cache writes rep file', os.path.exists(cached_file))
            # second call: fn would raise if invoked — proves cache hit
            r2 = compute_pt_reps(
                mol, ['fake_a'], cache=True, cache_key=key, verbose=False,
                rep_fns={'fake_a': lambda d: (_ for _ in ()).throw(
                    AssertionError('should not recompute'))})
            check('cache hit avoids recompute',
                  np.array_equal(r1['fake_a'], r2['fake_a']))
    except ImportError as e:
        print(f"  SKIP  cache round-trip (could not import _shared: {e})")
    finally:
        os.environ.pop('KIRBY_CACHE_DIR', None)


# =============================================================================
class _FakeBP:
    """Minimal Blueprint stand-in — only needs to_dict() for the harness helpers."""
    def __init__(self, **kw):
        self._d = kw
    def to_dict(self):
        return dict(self._d)


class _FakeSearch:
    def __init__(self, best_by_model_budget, lower_is_better=False):
        self.best_by_model_budget = best_by_model_budget
        self.lower_is_better = lower_is_better


def test_harness():
    print("\n[harness]")
    # Lazy import — the harness module pulls in sklearn etc.; keep it out of the
    # lighter tests above.
    from experiments.pretrain_rep.pretrain_rep_experiment import (
        blueprint_key, extract_top_n_blueprints, shared_rep_set,
        ft_leakage_inputs, pt_search_metric, aggregate_fold_scores)

    # blueprint_key: same recipe -> same key, different -> different
    bp1 = _FakeBP(reps=['a', 'b'], allocation_method='top_3',
                  importance_method='treeshap', internal_evaluator='rf',
                  budget=50, step_size=5)
    bp1_dup = _FakeBP(reps=['a', 'b'], allocation_method='top_3',
                      importance_method='treeshap', internal_evaluator='rf',
                      budget=50, step_size=5)
    bp2 = _FakeBP(reps=['a', 'b'], allocation_method='power_2',
                  importance_method='treeshap', internal_evaluator='rf',
                  budget=50, step_size=5)
    check('blueprint_key equal for same recipe',
          blueprint_key(bp1) == blueprint_key(bp1_dup))
    check('blueprint_key differs for different recipe',
          blueprint_key(bp1) != blueprint_key(bp2))

    # extract_top_n_blueprints: ordering, dedup, truncation
    bp3 = _FakeBP(reps=['a'], allocation_method='entropy_1.0',
                  importance_method='deeplift', internal_evaluator='svm',
                  budget=100, step_size=5)
    search = _FakeSearch({
        'rf': {50: {'blueprint': bp1, 'score_mean': 0.80},
               100: {'blueprint': bp2, 'score_mean': 0.90}},
        'svm': {50: {'blueprint': bp1_dup, 'score_mean': 0.85},  # dup of bp1
                100: {'blueprint': bp3, 'score_mean': 0.70}},
    }, lower_is_better=False)
    top = extract_top_n_blueprints(search, n=5)
    check('extract_top_n dedups blueprints', len(top) == 3,
          f'got {len(top)}')
    check('extract_top_n orders best-first (higher better)',
          blueprint_key(top[0]) == blueprint_key(bp2))
    check('extract_top_n truncates to N',
          len(extract_top_n_blueprints(search, n=2)) == 2)
    # lower_is_better flips the order
    search_low = _FakeSearch(search.best_by_model_budget, lower_is_better=True)
    top_low = extract_top_n_blueprints(search_low, n=5)
    check('extract_top_n respects lower_is_better',
          blueprint_key(top_low[0]) == blueprint_key(bp3))
    # NaN scores are skipped
    search_nan = _FakeSearch({
        'rf': {50: {'blueprint': bp1, 'score_mean': float('nan')},
               100: {'blueprint': bp2, 'score_mean': 0.5}}})
    check('extract_top_n skips NaN scores',
          len(extract_top_n_blueprints(search_nan, n=5)) == 1)
    check('extract_top_n handles empty search',
          extract_top_n_blueprints(_FakeSearch({}), n=5) == [])

    # shared_rep_set: intersection + dim-mismatch exclusion
    ft_reps = {'a': np.zeros((10, 4)), 'b': np.zeros((10, 8)),
               'c': np.zeros((10, 2))}
    pt_reps = {'a': np.zeros((20, 4)), 'b': np.zeros((20, 99)),  # b dim mismatch
               'd': np.zeros((20, 2))}
    shared = shared_rep_set(ft_reps, pt_reps, verbose=False)
    check('shared_rep_set intersects + excludes dim mismatch',
          shared == ['a'], f'got {shared}')

    # ft_leakage_inputs
    keys, seqs = ft_leakage_inputs({'domain': 'antibody',
                                    'heavy_seqs': ['AAA', 'CCC']})
    check('ft_leakage_inputs antibody -> sequences',
          keys is None and seqs == ['AAA', 'CCC'])
    keys, seqs = ft_leakage_inputs({'domain': 'molecular', 'smiles': []})
    check('ft_leakage_inputs molecular no smiles -> (None,None)',
          keys is None and seqs is None)
    keys, seqs = ft_leakage_inputs({'domain': 'reaction'})
    check('ft_leakage_inputs unknown domain -> (None,None)',
          keys is None and seqs is None)

    # pt_search_metric
    check('pt_search_metric regression', pt_search_metric('regression') == 'r2')
    check('pt_search_metric classification',
          pt_search_metric('classification') == 'f1_weighted')

    # aggregate_fold_scores (NaN-safe)
    agg = aggregate_fold_scores([1.0, 2.0, 3.0])
    check('aggregate_fold_scores mean', abs(agg['mean'] - 2.0) < 1e-9)
    check('aggregate_fold_scores per_fold', agg['per_fold'] == [1.0, 2.0, 3.0])
    agg_nan = aggregate_fold_scores([1.0, float('nan'), 3.0])
    check('aggregate_fold_scores NaN-safe', abs(agg_nan['mean'] - 2.0) < 1e-9)


# =============================================================================
def test_model_transfer():
    print("\n[model transfer]")
    try:
        import torch  # noqa: F401
    except ImportError as e:
        print(f"  SKIP  model transfer (torch unavailable: {e})")
        return
    from experiments.pretrain_rep.pretrain_rep_model_transfer import (
        pretrain_dnn_on_matrix)
    from kirby.models.dnn_pretrained import PretrainedDNNRegressor

    rng = np.random.RandomState(0)
    hs = (8, 4)
    X_pt = rng.normal(size=(40, 6)).astype(np.float32)
    y_pt = rng.normal(size=40).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, 'pt.pt')
        pretrain_dnn_on_matrix(X_pt, y_pt, ckpt, hidden_sizes=hs,
                               epochs=3, batch_size=16, patience=2,
                               verbose=False)
        check('pretrain writes checkpoint', os.path.exists(ckpt))
        cp = torch.load(ckpt, map_location='cpu', weights_only=True)
        check('checkpoint has the keys PretrainedDNNRegressor needs',
              {'state_dict', 'input_dim', 'feat_mean', 'feat_scale',
               'y_mean', 'y_std'} <= set(cp.keys()))
        check('checkpoint input_dim matches', cp['input_dim'] == 6)

        # fine-tune the pretrained net on a fresh tiny FT set
        X_ft = rng.normal(size=(20, 6)).astype(np.float32)
        y_ft = rng.normal(size=20).astype(np.float32)
        m = PretrainedDNNRegressor(hidden_sizes=hs, pretrained_path=ckpt,
                                   epochs=3, patience=2)
        m.fit(X_ft, y_ft)
        pred = m.predict(X_ft)
        check('pretrained DNN fine-tunes + predicts', len(pred) == 20)

        # dim-mismatch must be rejected (wrong input_dim)
        try:
            bad = PretrainedDNNRegressor(hidden_sizes=hs, pretrained_path=ckpt,
                                         epochs=2)
            bad.fit(rng.normal(size=(10, 9)).astype(np.float32),
                    rng.normal(size=10).astype(np.float32))
            check('checkpoint dim-mismatch rejected', False)
        except ValueError:
            check('checkpoint dim-mismatch rejected', True)


# =============================================================================
def main():
    print("=" * 60)
    print("PT-Rep Phase 0 — smoke tests")
    print("=" * 60)
    test_registry()
    test_quality()
    test_leakage()
    test_reps()
    test_harness()
    test_model_transfer()
    print("\n" + "=" * 60)
    print(f"  {PASS} passed, {FAIL} failed")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
