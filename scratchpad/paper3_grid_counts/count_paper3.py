"""Paper 3 noise grid, per dataset and per experiment, from the code as it
stands, with fitting stubbed out. Counts, not times.

Per experiment and dataset:
  versions  noisy versions of the labels (kind x level x draw; level 0 once
            where the experiment runs it once)
  pieces    the experiment's own unit of work, named in UNIT below
  results   n44 only: cleaned training sets scored, from n44.run_unit with
            fitting stubbed (first run: no put-back, no twins). The others'
            results per piece are not counted here.

n43 to n47 are counted as the Papers ledger now runs them (KIRBY_NOISE_PROMOTE
per experiment: five folds, the decided levels). n15, n20 and n36 are counted
as the ledger runs them (the screen: two levels; n15 folds 0-2, n20 and n36
fold 0 only unless the dataset is in ALL_FOLDS_DATASETS for n15).

Reads dataset_sizes.json (dataset_sizes.py). Writes paper3_counts.csv.
Run from the repo root:
  PYTHONPATH=src:tests:tests/experiments python scratchpad/paper3_grid_counts/count_paper3.py
"""
import csv
import json
import os
import sys
import types

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np  # noqa: E402
from experiments.noise import (n44_detect_and_clean as dc, noise_helpers as nh,  # noqa: E402
                               noise_cache as nc, n40_model_side_regularisation as n40,
                               n43_noise_aware_training as n43,
                               n46_structure_specific_fixes as n46,
                               n47_noise_measurements as n47)
from experiments.papers import plan as P  # noqa: E402
from experiments.decided_grid import decided_models  # noqa: E402
from experiments._shared import OUTER_FOLDS_RUN, ALL_FOLDS_DATASETS  # noqa: E402
from kirby.noise_spec import (CLASSIFICATION_STRATEGIES, DEFAULT_STRATEGY_CLF,  # noqa: E402
                              DEFAULT_STRATEGY_REG)

UNIT = {
    'n43': 'fold x version (its 9 network methods and feature spaces run inside)',
    'n44': 'fold x version x model x model input (base_1, hybrid)',
    'n45': 'fold x version x model x model input (base_1, hybrid)',
    'n46': 'fold x version x model x model input (base_1, hybrid); level 0 per kind; '
           'its sources, repeat and real-censoring passes not counted',
    'n47': 'fold x version x model x configuration (base_1, hybrid, hybrid__general2); '
           'one clean cell per kind; its labels-as-delivered part not counted',
    'n15': 'fold x version, each a search on the damaged labels',
    'n20': 'fold x version, per kind, plus the labels as they come once per kind',
    'n36': 'fold x seed, only where repeat measurements exist',
}

# ── n44 results per piece, fitting stubbed (count_all_regression.py's method) ──
N = 40
dc.nc.cached_detector_scores = lambda *a, **k: np.linspace(0, 1, N)
dc.detector_scores = lambda sp, name: np.linspace(0, 1, N)
dc.detection_scores = lambda *a, **k: None
dc._transfer = lambda c, X: c
CONF = [f'conformal_bh@{i}' for i in range(len(dc.CONFORMAL_FLAG_SETTINGS))]
TWO = [f'{dc.TWO_LEARNER_DETECTOR}@{s}' for s in dc.n32.ITLM_ALPHAS]
dc.apply_cleanup = lambda *a, **k: {'ok': 1}
_memo = {}


def n44_per_piece(is_clf, hybrid_input, takes_weight, oracle, two_models):
    key = (is_clf, hybrid_input, takes_weight, oracle, two_models)
    if key in _memo:
        return _memo[key]
    dc.flag_detectors = lambda sp: ([] if is_clf else
                                    [(n, np.zeros(N, bool)) for n in CONF
                                     + (TWO if two_models else [])])
    dc.oracle_flags = lambda u: ([(f'oracle@precision={p:g}|flag_share={s:g}',
                                   np.zeros(N, bool))
                                  for s in dc.FLAG_SHARES for p in dc.PRECISIONS]
                                 if u.oracle else [])
    n = [0]
    u = types.SimpleNamespace(
        is_clf=is_clf, takes_weight=takes_weight,
        mi='m:hybrid' if hybrid_input else 'm:base_1', counts={}, unavailable={},
        errors=types.SimpleNamespace(add=lambda *a: None), cmask=None,
        cliff_train=None, level=1.0, ds='x', y0=np.zeros(N), g_tr=None,
        condition='x', oracle=oracle, detection={}, second_run=None)
    u.emit = lambda *a, **k: n.__setitem__(0, n[0] + 1)
    sp = types.SimpleNamespace(u=u, X_tr=None, X_va=None)
    S = [types.SimpleNamespace(name=f's{i}', space='other') for i in range(3)]
    sim = types.SimpleNamespace(kept=[(s, 5, 'rank') for s in S],
                                nb={s.name: {} for s in S},
                                with_scheme=lambda s, sch: {})
    spaces = {'own': sp} if hybrid_input else {'own': sp, 'hybrid': sp}
    dc.run_unit(u, None, None, spaces, sim, {})
    _memo[key] = n[0]
    return n[0]


def stub(info):
    n = int(info['molecules'])
    return {'task_type': info['task_type'], 'labels': np.zeros(n),
            'smiles': [str(i) for i in range(n)] if info['has_smiles'] else None,
            'reps': {}}


def levels(data, ds, kind):
    if kind in getattr(n44mod, 'MODEL_ERROR_CONDITIONS', ()):
        return [float(x) for x in n40._condition_levels(data, ds, kind)]
    return [float(x) for x in nh.decided_levels(data, ds, kind)]


n44mod = dc


def versions(data, ds, kinds, level0_once=True):
    """[(kind, level, n draws)] for the passes, level 0 once in level0_owner's
    file when level0_once, else once per kind."""
    out = []
    for k in kinds:
        for lv in levels(data, ds, k):
            if lv == 0.0:
                if level0_once and not nh.runs_level0(data, k):
                    continue
                out.append((k, lv, 1))
            else:
                out.append((k, lv, len(nh.draws_for(k, (0, 1, 2)))))
    return out


def main():
    sizes = json.load(open(os.path.join(HERE, 'dataset_sizes.json')))
    rows = []
    for ds in P.PAPER3_DATASETS:
        info = sizes.get(ds) or {}
        if 'error' in info or 'molecules' not in info:
            rows.append({'dataset': ds, 'note': f"size not loaded: {info.get('error')}"})
            continue
        data = stub(info)
        is_clf = info['task_type'] == 'classification'
        ms = [m for m in decided_models(ds, 'general') if m != 'knn']
        fac = dc.get_model_factories(ms, task_type=info['task_type'],
                                     n_samples=int(info['rows']))
        ms = [m for m in ms if m in fac]
        tw = {m: bool(nc.fit_takes_sample_weight(fac[m])) for m in ms}
        group = nh.dataset_knn_group(data, ds)
        base = {'dataset': ds, 'task_type': info['task_type'],
                'molecules': info['molecules'], 'size_group': group,
                'models': '+'.join(ms)}

        # n43 to n47 as the ledger runs them: promoted by name
        os.environ['KIRBY_NOISE_PROMOTE'] = 'all'
        for e in P.PAPER3_FULL_GRID_NOISE:
            kinds = (P.paper3_noise_kinds(e, info['task_type'])
                     or list(CLASSIFICATION_STRATEGIES))
            if e in ('n46', 'n47') and is_clf:
                kinds = list(CLASSIFICATION_STRATEGIES)
            if e == 'n46' and is_clf:
                rows.append(dict(base, experiment=e, kinds=0, versions=0, pieces=0,
                                 note='regression only'))
                continue
            vs = versions(data, ds, kinds, level0_once=e in ('n43', 'n44', 'n45'))
            if e == 'n47':
                # one clean cell per kind plus every non-zero level and draw
                vs = [(k, lv, d) for k, lv, d in vs if lv != 0.0] + \
                     [(k, 0.0, 1) for k in kinds]
            nv = sum(d for _k, _l, d in vs)
            folds = 5
            if e == 'n43':
                pieces = nv * folds
            elif e == 'n47':
                pieces = nv * folds * len(ms) * len(n47.CONFIGURATION_SPACES)
            else:
                pieces = nv * folds * len(ms) * 2
            r = dict(base, experiment=e, kinds=len(kinds), versions=nv,
                     pieces=pieces)
            if e == 'n44':
                res = 0
                for k, lv, d in vs:
                    oracle = (not is_clf and k in nh.DETECTION_IS_DEFINED and lv > 0)
                    for m in ms:
                        for hy in (False, True):
                            res += n44_per_piece(is_clf, hy, tw[m], oracle,
                                                 len(ms) > 1) * d * folds
                r['results'] = res
            rows.append(r)

        # n15, n20, n36 as the ledger runs them: the screen
        os.environ.pop('KIRBY_NOISE_PROMOTE', None)
        k15 = DEFAULT_STRATEGY_CLF if is_clf else DEFAULT_STRATEGY_REG
        nv = sum(len(nh.draws_for(k15, range(3))) for lv in nh.decided_levels(data, ds, k15)
                 if float(lv) != 0.0)
        f15 = 5 if ds in ALL_FOLDS_DATASETS else OUTER_FOLDS_RUN['n15']
        rows.append(dict(base, experiment='n15', kinds=1, versions=nv,
                         pieces=nv * f15))
        k20 = nh.conditions_for_task(data)
        nv = sum(d for _k, _l, d in versions(data, ds, k20, level0_once=False))
        # one ledger unit per kind; each runs its injected passes (level 0
        # once) and the labels as they come once, on fold 0
        rows.append(dict(base, experiment='n20', kinds=len(k20), versions=nv,
                         pieces=(nv + len(k20)) * 1))
        if is_clf or not info.get('has_replicates'):
            rows.append(dict(base, experiment='n36', kinds=0, versions=0, pieces=0,
                             note='no repeat measurements' if not is_clf else 'regression only'))
        else:
            rows.append(dict(base, experiment='n36', kinds=0, versions=0,
                             pieces=1 * 2))
    keys = ['dataset', 'task_type', 'molecules', 'size_group', 'models',
            'experiment', 'kinds', 'versions', 'pieces', 'results', 'note']
    with open(os.path.join(HERE, 'paper3_counts.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})
    print(f"wrote {os.path.join(HERE, 'paper3_counts.csv')}")
    for (clf_, hy_, tw_, or_, two_), v in sorted(_memo.items()):
        print(f'  n44 results per piece: classification={clf_} hybrid_input={hy_} '
              f'model_takes_weight={tw_} oracle_flags={or_} two_models={two_}: {v}')


if __name__ == '__main__':
    main()
