#!/usr/bin/env python3
"""Generate a Trustii submission CSV from saved ANNITIA models.

Run `annitia/run.py` first to fit + persist the per-endpoint models. This
script loads them, applies the matching feature views to the validation CSV,
and writes a CSV with `trustii_id`, `risk_hepatic_event`, `risk_death`.

Validation CSV is expected to have:
  - trustii_id column
  - same input columns as the training CSV (minus outcome columns)

Usage:
    python tests/experiments/annitia/submit.py \
        --validation-csv /Volumes/seagate/kirby/annitia/val_test.csv \
        --models-dir tests/results/annitia \
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
from experiments.annitia.search import predict_risk_for_subset


def _load_endpoint_model(models_dir: Path, endpoint: str):
    path = models_dir / f'annitia_{endpoint}_model.pkl'
    if not path.exists():
        raise FileNotFoundError(
            f'No saved model for {endpoint} at {path}. '
            f'Run annitia_maslg.py first.')
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation-csv', required=True,
                   help='Path to the validation CSV (must contain trustii-id).')
    p.add_argument('--models-dir', default='tests/results/annitia',
                   help='Directory containing annitia_<endpoint>_model.pkl.')
    p.add_argument('--output', default='tests/results/annitia/annitia_submission.csv')
    args = p.parse_args()

    val_path = Path(args.validation_csv)
    if not val_path.exists():
        raise FileNotFoundError(args.validation_csv)

    print(f'Loading validation data from {val_path}')
    data = load_annitia(data_dir=str(val_path.parent),
                        filename=val_path.name,
                        has_outcomes=False)
    if data['trustii_id'] is None:
        raise ValueError(
            f'No `trustii_id` column in {val_path}. Cannot match submission rows.')
    n = len(data['patient_ids'])
    print(f'  {n} validation patients')

    views = build_views(data)

    submission = pd.DataFrame({'trustii_id': data['trustii_id']})
    models_dir = Path(args.models_dir)

    for endpoint, col in (('hepatic', 'risk_hepatic_event'),
                          ('death', 'risk_death')):
        try:
            payload = _load_endpoint_model(models_dir, endpoint)
        except FileNotFoundError as e:
            print(f'WARNING: {e}')
            print(f'  Filling {col} with zeros (will hurt your score).')
            submission[col] = 0.0
            continue
        risk = predict_risk_for_subset(
            payload['model'], views, payload['view_names'])
        submission[col] = risk
        print(f'  {endpoint:8s} -> {col:20s} '
              f'views={"+".join(payload["view_names"])} '
              f'model={payload["model_name"]} '
              f'risk range=[{risk.min():.3f}, {risk.max():.3f}]')

    submission.to_csv(args.output, index=False)
    print(f'\nSubmission written: {args.output}')
    print(f'  rows: {len(submission)}')
    print(f'  cols: {list(submission.columns)}')


if __name__ == '__main__':
    main()
