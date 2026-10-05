"""Why c21's gp baseline is nan on ginkgo_hic (2026-10-06).

c21 fits gp on each base representation alone, every feature, scaled with
FeatureScaler, and scores it on the held-out rows. On the server run every gp
baseline came out nan while gp on the hybrid scored about 0.34 spearman. This
fits that same gp on fold 0 of each representation and prints the spread of its
predictions and the kernel the optimiser settled on. A spread of 0 means gp
predicted one number for every molecule, and spearman of a constant is nan.

usage: python tests/tools/check_gp_baseline.py [dataset]
Read-only: writes nothing.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]

from experiments.complexity.complexity_helpers import DATASET_REPS  # noqa: E402
from experiments._shared import (load_data, get_model_factories,  # noqa: E402
                                 get_domain_metric)
from kirby.find_best_hybrid import nested_cv_outer_folds  # noqa: E402
from kirby.scaling import FeatureScaler  # noqa: E402

ds = sys.argv[1] if len(sys.argv) > 1 else 'ginkgo_hic'
data = load_data(ds, rep_preset='all', rep_names=list(DATASET_REPS[ds]['general']))
y = np.asarray(data['labels'], dtype=float)
metric_name, metric_fn, _ = get_domain_metric(data)
tr, va = nested_cv_outer_folds(data['reps'], y, data['groups'], 5, 42,
                               'regression')[0]
factory = get_model_factories(['gp'], 'regression', n_samples=len(tr))['gp']
print(f'{ds}: {len(y)} rows, fold 0 has {len(tr)} training and {len(va)} '
      f'held-out rows; metric {metric_name}', flush=True)
for name, X in data['reps'].items():
    sc = FeatureScaler()
    Xtr, Xva = sc.fit_transform(X[tr]), sc.transform(X[va])
    m = factory()
    m.fit(Xtr, y[tr])
    p = m.predict(Xva)
    print(f'{name:22s} {X.shape[1]:5d} features  prediction spread '
          f'{np.std(p):.3e}  {metric_name} {metric_fn(y[va], p)}  '
          f'kernel {m.kernel_}', flush=True)
