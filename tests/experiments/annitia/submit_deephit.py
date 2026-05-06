#!/usr/bin/env python3
"""Generate a Trustii submission CSV from a saved DeepHit-MT model.

Loads the refit DeepHitMT pickle produced by run_deephit.py, builds the same
all-views concat for val_test patients, predicts (risk_hepatic, risk_death),
and writes the submission CSV.

Usage:
    python tests/experiments/annitia/submit_deephit.py \
        --validation-csv /Volumes/seagate/kirby/annitia/val_test.csv \
        --models-dir tests/results/annitia_deephit \
        --output tests/results/annitia_deephit/annitia_submission_deephit.csv
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation-csv', required=True)
    p.add_argument('--models-dir', default='tests/results/annitia_deephit')
    p.add_argument('--output',
                   default='tests/results/annitia_deephit/annitia_submission_deephit.csv')
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

    model_path = Path(args.models_dir) / 'annitia_deephit_model.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f'No DeepHit model at {model_path}')
    with open(model_path, 'rb') as f:
        payload = pickle.load(f)
    m = payload['model']
    view_names = payload['view_names']
    print(f'  loaded DeepHitMT  views={"+".join(view_names)}  '
          f'config={payload.get("config", {})}')

    views_dict = build_views(data, view_names=view_names)
    X = np.concatenate([views_dict[name][0] for name in view_names], axis=1)
    if X.shape[1] != payload.get('n_features', X.shape[1]):
        raise ValueError(f'Feature count mismatch: model expects '
                         f'{payload["n_features"]}, got {X.shape[1]}')

    risk_hep, risk_death = m.predict_risk(X)
    print(f'  hep   range=[{risk_hep.min():.4f}, {risk_hep.max():.4f}]')
    print(f'  death range=[{risk_death.min():.4f}, {risk_death.max():.4f}]')

    submission = pd.DataFrame({
        'trustii_id': data['trustii_id'],
        'risk_hepatic_event': risk_hep,
        'risk_death': risk_death,
    })
    submission.to_csv(args.output, index=False)
    print(f'\nSubmission written: {args.output}')
    print(f'  rows: {len(submission)}, cols: {list(submission.columns)}')


if __name__ == '__main__':
    main()
