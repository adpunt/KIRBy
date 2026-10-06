#!/usr/bin/env python3
"""f39 -- which molecules the kNN-setting screen should predict (chat 5 open item 1).

The noise suite refits only the top 3 neighbour settings per fold
(n07.screen_settings). Two ways to rank them:
  training_rows  each training molecule predicted from its neighbours among the
                 other training molecules, scored against the DAMAGED training
                 labels. Uses nothing held out.
  held_out       each held-out molecule predicted from its training neighbours,
                 scored against its CLEAN label. How KNN_SEARCH_SPACE_RECORD
                 measured the screen; but the held-out molecules are the ones
                 every experiment reports on.

For each fold this damages the training labels (outlier_p05, level 1.0, draw
0), builds every single-representation setting (each pruned representation x
the group's distances, plus forest proximity in each representation) at 5, 10
and 20 neighbours and both counting schemes, ranks them both ways, and grades
neighbor_residual's detection AUROC (against the molecules that were really
corrupted) under every (setting, k). Written per fold:
  - the detection AUROC of the best (setting, k) of all, and of the best among
    each screen's top 3 (the gap is what that screen gives up);
  - how many of each screen's top 3 are forest-proximity settings.

No model is fitted except the forests that define forest proximity. No hybrid
search. Usage:
  OMP_NUM_THREADS=1 python tests/experiments/framework/f39_screen_choice.py herg_ki caco2
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..', '..', '..', 'src'),
           os.path.join(_HERE, '..', '..'), os.path.join(_HERE, '..')):
    sys.path.insert(0, _p)

CONDITION, LEVEL, DRAW = 'outlier_p05', 1.0, 0
TOP = 3
OUT = os.path.join(_HERE, '..', '..', 'results', 'framework', 'f39_screen_choice')


def run(ds):
    from experiments._shared import get_domain_metric
    from experiments.noise import noise_helpers as nh
    from experiments.noise import n07_remediation as n07
    from experiments.noise import similarity_space as ss
    from experiments.noise import noise_cache as nc
    from kirby.scaling import FeatureScaler

    data, _ = nh.load_noise_data(ds)
    metric = get_domain_metric(data)
    is_clf = data.get('task_type') == 'classification'
    group = nh.dataset_knn_group(data, ds)
    metrics = ss.GROUP_METRICS[group]
    folds, n = nh.noise_outer_folds(data, {'n_splits': 5})
    y_all = np.asarray(data['labels'])
    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, f'{ds}_v2.jsonl')
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            done.add(json.loads(line)['fold'])
    for fold_i, tr, va in folds:
        if fold_i in done:
            continue
        t0 = time.time()
        y_tr, mask, _ = nh.inject_condition(data, ds, CONDITION, LEVEL, DRAW,
                                            train_idx=tr)
        y_va = y_all[va]
        rows = []   # (setting label, k, scheme, train score, held-out score, auroc)
        for rep, X in data['reps'].items():
            sc = FeatureScaler()
            Xtr = sc.fit_transform(np.asarray(X, dtype=np.float32)[tr])
            Xva = sc.transform(np.asarray(X, dtype=np.float32)[va])
            sets = [ss.SimilaritySetting('base_1', 'none', m) for m in metrics]
            sets.append(ss.SimilaritySetting(ss.FOREST_SPACE, 'none', 'forest'))
            nbrs = n07.fold_neighbours(ds, sets, {'base_1': Xtr, 'model_input': Xtr},
                                       {'base_1': Xva, 'model_input': Xva}, y_tr,
                                       is_clf=is_clf)
            _prev = n07.SCREEN_SCORED_ON
            try:
                n07.SCREEN_SCORED_ON = 'training_rows'
                _, tab_tr = n07.screen_settings(sets, nbrs, y_tr, metric=metric,
                                                is_clf=is_clf, top=0)
                n07.SCREEN_SCORED_ON = 'held_out'
                _, tab_va = n07.screen_settings(sets, nbrs, y_tr, metric=metric,
                                                is_clf=is_clf, y_val=y_va, top=0)
                n07.SCREEN_SCORED_ON = 'inner_holdout'
                g_tr = (None if data.get('groups') is None
                        else np.asarray(data['groups'])[tr])
                _, tab_in = n07.screen_settings(
                    sets, nbrs, y_tr, metric=metric, is_clf=is_clf, top=0,
                    inner={'dataset_name': ds,
                           'spaces_train': {'base_1': Xtr, 'model_input': Xtr},
                           'y_train': y_tr, 'groups_train': g_tr,
                           'smiles_train': None, 'is_clf': is_clf})
            finally:
                n07.SCREEN_SCORED_ON = _prev
            va_score = {(r['setting'], r['k'], r['scheme']): r['score'] for r in tab_va}
            in_score = {(r['setting'], r['k'], r['scheme']): r['score'] for r in tab_in}
            auroc = {}
            for s in sets:
                L = nbrs.get(s.name)
                for k in n07.usable_ks(L):
                    scores = nc.cached_detector_scores(
                        ds, Xtr, y_tr, is_clf=is_clf, detector='neighbor_residual',
                        similarity_neighbours=L, similarity_k=k)
                    d = None if scores is None else nh.detection_scores(
                        mask, scores, CONDITION)
                    auroc[(s.name, k)] = (None if d is None else d['detection_auroc'])
            for r in tab_tr:
                if r.get('score') is None:
                    continue
                lab = f"{rep}|{r['setting'].split('|', 1)[1]}"
                rows.append((lab, r['k'], r['scheme'], r['score'],
                             va_score.get((r['setting'], r['k'], r['scheme'])),
                             auroc.get((r['setting'], r['k'])),
                             in_score.get((r['setting'], r['k'], r['scheme']))))
        lower = metric[2]

        def top3(col):
            ok = [r for r in rows if r[col] is not None and np.isfinite(r[col])]
            ok.sort(key=lambda r: (r[col] if lower else -r[col], r[0], r[1], r[2]))
            return ok[:TOP]
        best_all = max(r[5] for r in rows if r[5] is not None)
        rec = {'dataset': ds, 'fold': fold_i, 'n_folds': n, 'metric': metric[0],
               'condition': CONDITION, 'level': LEVEL, 'draw': DRAW,
               'n_settings_scored': len(rows),
               'best_detection_auroc_any_setting': best_all}
        for name, col in (('training_rows', 3), ('held_out', 4),
                          ('inner_holdout', 6)):
            t = top3(col)
            best_kept = max((r[5] for r in t if r[5] is not None), default=None)
            rec[name] = {'top3': [f'{r[0]}|k={r[1]}|{r[2]}' for r in t],
                         'n_forest_in_top3': sum('forest' in r[0] for r in t),
                         'best_detection_auroc_in_top3': best_kept,
                         'detection_auroc_given_up': (None if best_kept is None
                                                      else best_all - best_kept)}
        rec['seconds'] = round(time.time() - t0, 1)
        with open(out_path, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        print(json.dumps({k: rec[k] for k in ('dataset', 'fold', 'seconds')}),
              {k: (rec[k]['n_forest_in_top3'], rec[k]['detection_auroc_given_up'])
               for k in ('training_rows', 'held_out', 'inner_holdout')},
              flush=True)


if __name__ == '__main__':
    for ds in sys.argv[1:]:
        run(ds)
