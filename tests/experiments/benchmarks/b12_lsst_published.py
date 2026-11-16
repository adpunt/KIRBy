#!/usr/bin/env python3
"""LSST (UEA archive): the published classifiers rerun on the default split.

Data: LSST_TRAIN.ts / LSST_TEST.ts from timeseriesclassification.com, loaded by
aeon's `load_classification` (it downloads the archive zip on first use).
2,459 training and 2,466 test light curves, 6 passbands, 36 flux values per
passband, 14 classes. Only flux values: no observation times, no flux errors,
no redshift.

Published reference: Middlehurst et al. 2026, "The Multiverse of Time Series
Machine Learning", arXiv 2603.20352, Tables 17-18 and
github.com/aeon-toolkit/multiverse results/multiverse/accuracy_mean.csv. Their
protocol, quoted from section 6: "a single run on the default train-test split,
with no resampling", "we set the random seed for all relevant algorithms to 0",
accuracy. Their driver builds each classifier with tsml-eval's
`get_classifier_by_name(name, random_state=0)` (n_jobs=1, fit_contract=0); the
constructor calls below are copied from tsml-eval main,
tsml_eval/experiments/_get_classifier.py, read 2026-11-09.

Versions: multiverse's pyproject asks for aeon>=1.4.0 and tsml-eval main. The
aeon version that produced their results is not recorded. This file runs on
whatever aeon is installed and writes that version into every result.

    python tests/experiments/benchmarks/b12_lsst_published.py --classifiers catch22,quant,minirocket
    python tests/experiments/benchmarks/b12_lsst_published.py            # every classifier below

Writes <results>/b12_lsst_published/<name>.json (accuracy, balanced accuracy,
seconds, aeon version, the published accuracy) and <name>.pred.npy. Paths:
KIRBY_LSST_DIR (data, default /Volumes/seagate/benchmarks_protocol/lsst) and
KIRBY_BENCH_RESULTS (default /Volumes/seagate/benchmarks_protocol/results).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b12_lsst_views import load_split, RESULTS_ROOT, log  # noqa: E402

SEED = 0

# Multiverse accuracy_mean.csv, LSST row (read 2026-11-09). FreshPRINCE's own
# file, results/multiverse/FreshPRINCE/FreshPRINCE_accuracy.csv, says 0.70397
# (1,736 of 2,466) against 0.70276 (1,733) in the combined table; both on main.
PUBLISHED = {
    'freshprince': 0.702757502027575, 'hc2': 0.6739659367396593,
    'catch22': 0.6695052716950527, 'arsenal': 0.6577453365774534,
    'rdst': 0.6492295214922952, 'quant': 0.6451743714517437,
    'mrhydra': 0.6427412814274128, 'cif-500': 0.6366585563665855,
    'rocket': 0.6362530413625304, 'drcif-500': 0.6354420113544201,
    'rist': 0.6273317112733171, 'stc': 0.6131386861313869,
    'tde': 0.5575831305758313, '1nn-dtw': 0.551094890510949,
    'dummy': 0.3150851581508516,
    # not in Multiverse; MiniRocket 0.643 is the RLPAM table (Gao et al. IJCAI 2022)
    'minirocket': 0.643, 'multirocket': None, 'hydra': None,
}


def make(name):
    """tsml-eval's constructor for each name, random_state=0, n_jobs=1,
    fit_contract=0 (passed as time_limit_in_minutes where tsml-eval passes it)."""
    rs, nj = SEED, 1
    if name == 'freshprince':
        from aeon.classification.feature_based import FreshPRINCEClassifier
        return FreshPRINCEClassifier(random_state=rs, n_jobs=nj)
    if name == 'catch22':
        from aeon.classification.feature_based import Catch22Classifier
        return Catch22Classifier(outlier_norm=False, random_state=rs, n_jobs=nj)
    if name == 'rocket':
        from aeon.classification.convolution_based import RocketClassifier
        return RocketClassifier(random_state=rs, n_jobs=nj)
    if name == 'minirocket':
        from aeon.classification.convolution_based import MiniRocketClassifier
        return MiniRocketClassifier(random_state=rs, n_jobs=nj)
    if name == 'multirocket':
        from aeon.classification.convolution_based import MultiRocketClassifier
        return MultiRocketClassifier(random_state=rs, n_jobs=nj)
    if name == 'mrhydra':
        from aeon.classification.convolution_based import MultiRocketHydraClassifier
        return MultiRocketHydraClassifier(random_state=rs, n_jobs=nj)
    if name == 'hydra':
        from aeon.classification.convolution_based import HydraClassifier
        return HydraClassifier(random_state=rs, n_jobs=nj)
    if name == 'quant':
        from aeon.classification.interval_based import QUANTClassifier
        return QUANTClassifier(random_state=rs)
    if name == 'rdst':
        from aeon.classification.shapelet_based import RDSTClassifier
        return RDSTClassifier(random_state=rs, n_jobs=nj)
    if name == 'hc2':
        from aeon.classification.hybrid import HIVECOTEV2
        return HIVECOTEV2(random_state=rs, n_jobs=nj, time_limit_in_minutes=0)
    if name == 'arsenal':
        from aeon.classification.convolution_based import Arsenal
        return Arsenal(random_state=rs, n_jobs=nj, time_limit_in_minutes=0)
    if name == 'drcif-500':
        from aeon.classification.interval_based import DrCIFClassifier
        return DrCIFClassifier(n_estimators=500, random_state=rs, n_jobs=nj, time_limit_in_minutes=0)
    if name == 'cif-500':
        from aeon.classification.interval_based import CanonicalIntervalForestClassifier
        return CanonicalIntervalForestClassifier(n_estimators=500, random_state=rs, n_jobs=nj,
                                                 time_limit_in_minutes=0)
    if name == 'rist':
        from aeon.classification.interval_based import RISTClassifier
        from sklearn.ensemble import ExtraTreesClassifier
        return RISTClassifier(random_state=rs, n_jobs=nj,
                              estimator=ExtraTreesClassifier(n_estimators=500, criterion='entropy'))
    if name == 'stc':
        # tsml-eval imports its own work-in-progress STC (tsml_eval._wip.shapelets
        # .early_abandon._stc3), not aeon's. aeon's is used here and says so.
        from aeon.classification.shapelet_based import ShapeletTransformClassifier
        return ShapeletTransformClassifier(random_state=rs, n_jobs=nj, time_limit_in_minutes=0)
    if name == 'tde':
        from aeon.classification.dictionary_based import TemporalDictionaryEnsemble
        return TemporalDictionaryEnsemble(random_state=rs, n_jobs=nj, time_limit_in_minutes=0)
    if name == '1nn-dtw':
        from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
        return KNeighborsTimeSeriesClassifier(distance='dtw', n_jobs=nj)
    if name == 'dummy':
        from aeon.classification import DummyClassifier
        return DummyClassifier(random_state=rs)
    raise ValueError(name)


DEVIATIONS = {
    'stc': "aeon's ShapeletTransformClassifier; tsml-eval uses its own _wip _stc3",
}
# Roughly cheapest first, so a time-boxed run reaches the most classifiers.
ORDER = ['dummy', 'catch22', 'quant', 'minirocket', 'rocket', 'hydra', 'multirocket', 'mrhydra',
         'rdst', 'freshprince', '1nn-dtw', 'rist', 'arsenal', 'cif-500', 'drcif-500', 'tde',
         'stc', 'hc2']


def run_one(name, out_dir):
    import aeon
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    out = out_dir / f'{name}.json'
    if out.exists():
        log(f'{name}: done already ({out})')
        return
    X_tr, y_tr, X_te, y_te = load_split()
    t0 = time.time()
    clf = make(name)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    sec = time.time() - t0
    acc = float(accuracy_score(y_te, pred))
    res = {'classifier': name, 'split': 'default', 'random_state': SEED,
           'aeon_version': aeon.__version__, 'n_train': int(len(y_tr)), 'n_test': int(len(y_te)),
           'accuracy': acc, 'n_correct': int((pred == y_te).sum()),
           'balanced_accuracy': float(balanced_accuracy_score(y_te, pred)),
           'macro_f1': float(f1_score(y_te, pred, average='macro')),
           'published_accuracy': PUBLISHED.get(name), 'seconds': sec,
           'deviation': DEVIATIONS.get(name), 'result_status': 'ok', 'debug': False}
    np.save(out_dir / f'{name}.pred.npy', np.asarray(pred).astype(str))
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(res, indent=1))
    tmp.rename(out)
    pub = PUBLISHED.get(name)
    log(f'{name}: accuracy {acc:.4f} ({res["n_correct"]} of {len(y_te)}), published '
        f'{"not reported" if pub is None else f"{pub:.4f}"}, {sec:.0f} s, aeon {aeon.__version__}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classifiers', default=','.join(ORDER))
    a = ap.parse_args()
    out_dir = RESULTS_ROOT / 'b12_lsst_published'
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in a.classifiers.split(','):
        try:
            run_one(name, out_dir)
        except Exception as e:  # noqa: BLE001
            log(f'{name}: FAILED {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
