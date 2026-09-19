"""Print the headline number of every similarity result file under one directory.

One line per (experiment, dataset, quantity). Every line names what the top of
the number counts and what the bottom counts where it is a fraction, and the
metric where it is a metric, because the three datasets do not share one.

Usage:
  python tests/experiments/similarity/_local_read_results.py [results_dir]
"""
import json
import os
import sys

# Which stored key carries each file's answer, and what that number is.
# (experiment directory, dotted path into the JSON, one line of what it means)
HEADLINES = [
    ('s02_hybrid_similarity', 'sac_delta_from_best_rep',
     'weighted hybrid rho minus best single representation rho, whole dataset'),
    ('s02_hybrid_similarity',
     'fold_reselection_result.knn.delta_mean_of_folds',
     'same difference, both arms re-picked on the other folds'),
    ('s04_activity_cliffs', 'cliff_info.pct_cliff_compounds',
     'molecules in at least one cliff pair, per 100 molecules scanned'),
    ('s05_knn_prediction', 'summary.fixed_ref.delta_vs_best_rep',
     'hybrid minus the named reference pair, dataset metric, positive=hybrid'),
    ('s05_knn_prediction', 'summary.posthoc_max.delta_vs_best_rep',
     'the same, hybrid maximised over every cell it was scored on'),
    ('s05_knn_prediction',
     'fold_reselection_result.knn.delta_mean_of_folds',
     'same difference, both arms re-picked on the other folds'),
    ('s06_applicability_domain', 'comparator_rep',
     'the single representation the hybrid distance is measured against'),
    ('s08_optimal_dimension', 'optimal.best_sac_budget',
     'features that maximise the distance-activity Spearman'),
    ('s08_optimal_dimension', 'optimal.best_rf_budget',
     'features that maximise the random forest score'),
    ('s08_optimal_dimension',
     'sac_reselected_unweighted.delta_mean_of_folds',
     'hybrid rho minus best representation rho, re-picked on the other folds'),
    ('s12_cross_domain_cliffs', 'cliff_info.pct_cliff_compounds',
     'molecules in at least one cliff pair, per 100 molecules scanned'),
    ('s13_clustering_retrieval', 'summary',
     'the file\'s own summary block'),
]


def dig(obj, path):
    cur = obj
    for part in path.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def fmt(v):
    if isinstance(v, float):
        return f"{v:+.4f}"
    if isinstance(v, dict):
        return json.dumps(v)[:200]
    return str(v)


def walk_deltas(obj, prefix='', out=None, depth=0):
    """Every leaf whose key names a difference, with its path."""
    if out is None:
        out = []
    if depth > 6 or not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        p = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if 'delta' in str(k).lower() or str(k).lower().startswith('rho'):
                out.append((p, v))
        else:
            walk_deltas(v, p, out, depth + 1)
    return out


def main(root):
    for exp_dir in sorted(os.listdir(root)):
        d = os.path.join(root, exp_dir)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.json'):
                continue
            ds = fn[:-5]
            try:
                res = json.load(open(os.path.join(d, fn)))
            except ValueError as e:
                print(f"{exp_dir:28s} {ds:12s} UNREADABLE {e}")
                continue
            if res.get('skipped'):
                print(f"{exp_dir:28s} {ds:12s} skipped: {res.get('reason')}")
                continue
            printed = False
            for e, path, meaning in HEADLINES:
                if e != exp_dir:
                    continue
                v = dig(res, path)
                if v is None:
                    continue
                print(f"{exp_dir:28s} {ds:12s} {path:48s} {fmt(v)}   # {meaning}")
                printed = True
            if not printed:
                ds_deltas = walk_deltas(res)[:8]
                for p, v in ds_deltas:
                    print(f"{exp_dir:28s} {ds:12s} {p:48s} {fmt(v)}")
                if not ds_deltas:
                    print(f"{exp_dir:28s} {ds:12s} no headline key found; "
                          f"top-level keys: {sorted(res)[:10]}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'tests/results_local_sim')
