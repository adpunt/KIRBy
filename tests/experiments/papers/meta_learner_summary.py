#!/usr/bin/env python3
"""One table for the meta-learner over models with their own hybrids.

    python tests/experiments/papers/meta_learner_summary.py [--run RUN]

One line per dataset and outer fold, test score in the dataset's own metric. c11
uses the same outer folds and seed. Its columns are ensemble_s1_auto per recorded
model and ensemble_s2_auto; blank means c11 has not run on that dataset.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent.parent
REPO = TESTS.parent
for p in (REPO / 'src', TESTS, TESTS / 'experiments'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def c11_fold_scores(run_root, dataset, fold):
    """{column name: score on this fold} from c11's result file."""
    files = sorted((Path(run_root) / 'complexity').glob(f'c11_*/{dataset}.json'))
    if not files:
        return {}
    out = {}
    for r in json.loads(files[0].read_text()).get('results') or []:
        if r.get('method') == 'ensemble_s1_auto':
            name = f"ensemble_{r.get('model')}"
        elif r.get('method') == 'ensemble_s2_auto':
            name = 'ensemble_best_model_per_representation'
        else:
            continue
        scores = r.get('fold_scores') or []
        if fold < len(scores):
            out[name] = scores[fold]
    return out


def table(run_root):
    """[row dict] for every finished unit, one per dataset and outer fold."""
    rows = []
    for f in sorted((Path(run_root) / 'meta_learner').glob('*__meta_f*.json')):
        d = json.loads(f.read_text())
        if d.get('result_status') != 'ok':
            continue
        c = d['comparison']
        counts = c['meta_learner_that_counts'] or {}
        pair = c['best_model_base_pair'] or {}
        rows.append({
            'dataset': d['dataset'], 'fold': d['fold'], 'metric': d['metric'],
            'meta_learner': counts.get('name'),
            'meta_learner_score': counts.get('heldout_score'),
            'best_model_hybrid': c['best_model_hybrid']['heldout_score'],
            'plain_average_of_5': (c['plain_average_of_5'] or {}).get('heldout_score'),
            'best_model_base_pair': pair.get('heldout_score'),
            **c11_fold_scores(run_root, d['dataset'], d['fold']),
        })
    return rows


def main(argv=None):
    from experiments.papers.ledger import resolve_run
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', default=None)
    a = ap.parse_args(argv)
    root = resolve_run(a.run)
    if root is None:
        sys.exit('no papers run: pass --run or set KIRBY_PAPERS_RUN')
    rows = table(root)
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    print('\t'.join(cols))
    for r in rows:
        print('\t'.join('' if r.get(k) is None else
                        (f'{r[k]:.4f}' if isinstance(r[k], float) else str(r[k])) for k in cols))


if __name__ == '__main__':
    main()
