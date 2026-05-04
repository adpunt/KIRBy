#!/usr/bin/env python3
"""Generate a Trustii submission CSV from saved ANNITIA models.

Two prediction modes:
  --mode single    use the single best refit model per endpoint (legacy)
  --mode ensemble  rank-average the top-K refit models per endpoint (default)

Ensemble mode requires `annitia_<endpoint>_topk.pkl` to exist (produced by
the latest run.py). Falls back to single-model prediction if topk is absent.

Usage:
    python tests/experiments/annitia/submit.py \\
        --validation-csv /Volumes/seagate/kirby/annitia/val_test.csv \\
        --models-dir tests/results/annitia \\
        --output tests/results/annitia/annitia_submission.csv
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia
from experiments.annitia.views import build_views
from experiments.annitia.search import predict_risk_for_subset, rank_ensemble


def _load_pickle(path: Path):
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation-csv', required=True)
    p.add_argument('--models-dir', default='tests/results/annitia')
    p.add_argument('--output', default='tests/results/annitia/annitia_submission.csv')
    p.add_argument('--mode', choices=('ensemble', 'single'), default='ensemble',
                   help='ensemble: rank-average top-K refits; single: use only the best.')
    args = p.parse_args()

    val_path = Path(args.validation_csv)
    if not val_path.exists():
        raise FileNotFoundError(args.validation_csv)

    print(f'Loading validation data from {val_path}')
    data = load_annitia(data_dir=str(val_path.parent),
                        filename=val_path.name,
                        has_outcomes=False)
    if data['trustii_id'] is None:
        raise ValueError(f'No `trustii_id` column in {val_path}.')
    n = len(data['patient_ids'])
    print(f'  {n} validation patients')

    views = build_views(data)
    submission = pd.DataFrame({'trustii_id': data['trustii_id']})
    models_dir = Path(args.models_dir)

    for endpoint, col in (('hepatic', 'risk_hepatic_event'),
                          ('death', 'risk_death')):
        topk_path = models_dir / f'annitia_{endpoint}_topk.pkl'
        single_path = models_dir / f'annitia_{endpoint}_model.pkl'
        topk_payload = _load_pickle(topk_path)
        single_payload = _load_pickle(single_path)

        if args.mode == 'ensemble' and topk_payload and topk_payload.get('top_k'):
            top_k = topk_payload['top_k']
            risk = rank_ensemble(top_k, views)
            print(f'  {endpoint:8s} ENSEMBLE of {len(top_k)} configs -> {col}')
            for cfg in top_k:
                print(f'      views={"+".join(cfg["views"])} '
                      f'model={cfg["model_name"]} prune={cfg["prune"]} '
                      f'CV={cfg.get("mean_cindex", float("nan")):.4f}')
        elif single_payload is not None:
            risk = predict_risk_for_subset(
                single_payload['model'], views,
                single_payload['view_names'],
                feature_indices=single_payload.get('feature_indices'))
            print(f'  {endpoint:8s} SINGLE  views={"+".join(single_payload["view_names"])} '
                  f'model={single_payload["model_name"]} prune={single_payload.get("prune")} '
                  f'-> {col}')
        else:
            print(f'WARNING: no model artifact for {endpoint}; filling zeros')
            risk = np.zeros(n)

        submission[col] = risk
        print(f'      risk range=[{risk.min():.3f}, {risk.max():.3f}]')

    submission.to_csv(args.output, index=False)
    print(f'\nSubmission written: {args.output}')
    print(f'  rows: {len(submission)}, cols: {list(submission.columns)}')


if __name__ == '__main__':
    main()
