"""Class-balanced rare-event hep training. Hep events 47/1106 = 4.2%.
Reweight events by 1/event_rate so hep-event rows count more.

Use sample_weight on RSF + GBSA (both support sample_weight).
Output:
  oof_hepatic_classbal_rsf__snapshot+clinical_pro+fpca+gp_smooth.npy
  val_hepatic_classbal_rsf__snapshot+clinical_pro+fpca+gp_smooth.npy
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src')); sys.path.insert(0, str(REPO / 'tests'))
from kirby.datasets.annitia_maslg import load_annitia
from experiments.annitia.views import build_views
from experiments.annitia.survival import concordance

VAL_CSV = '/Volumes/seagate/kirby/annitia/val_test.csv'
BRANCHES = REPO / 'tests/results/annitia/branches'
PRIMARY = ('snapshot', 'clinical_pro', 'fpca', 'gp_smooth')


def main():
    train = load_annitia()
    val = load_annitia(data_dir=str(Path(VAL_CSV).parent),
                        filename=Path(VAL_CSV).name, has_outcomes=False)
    vt = build_views(train, view_names=PRIMARY)
    vv = build_views(val,   view_names=PRIMARY)
    X_tr = np.concatenate([vt[v][0] for v in PRIMARY], axis=1)
    X_va = np.concatenate([vv[v][0] for v in PRIMARY], axis=1)

    valid = train['hepatic']['valid_mask']
    y = train['hepatic']['y_struct'][valid]
    Xv = X_tr[valid]
    er = y['event'].mean()
    sw = np.where(y['event'], 1.0/er, 1.0/(1.0-er))  # event up-weight
    print(f'event_rate={er:.4f}, sample_weight: events={1/er:.2f}, censored={1/(1-er):.3f}')

    BRANCHES.mkdir(exist_ok=True, parents=True)

    for model_name, ctor in (
        ('rsf300', lambda: RandomSurvivalForest(n_estimators=300, max_depth=8,
                                                   min_samples_leaf=10, random_state=42, n_jobs=-1)),
        ('gbsa300', lambda: GradientBoostingSurvivalAnalysis(n_estimators=300,
                                                                  learning_rate=0.05,
                                                                  max_depth=3, random_state=42)),
    ):
        print(f'\n  classbal_{model_name} hepatic')
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.full(Xv.shape[0], np.nan)
        for k, (tr, te) in enumerate(skf.split(Xv, y['event'].astype(int))):
            m = ctor()
            try:
                m.fit(Xv[tr], y[tr], sample_weight=sw[tr])
                oof[te] = m.predict(Xv[te])
            except TypeError:
                # sample_weight not supported; fit without
                m.fit(Xv[tr], y[tr])
                oof[te] = m.predict(Xv[te])
            except Exception as e:
                print(f'    fold {k} failed: {e}')
        ok = np.isfinite(oof)
        if ok.sum():
            print(f'  pooled OOF C: {concordance(y[ok], oof[ok]):.4f}')
        full_oof = np.full(X_tr.shape[0], np.nan); full_oof[valid] = oof
        cid = f'classbal_{model_name}__' + '+'.join(PRIMARY)
        np.save(BRANCHES / f'oof_hepatic_{cid}.npy', full_oof)
        # Refit + val
        m = ctor()
        try:
            m.fit(Xv, y, sample_weight=sw)
        except TypeError:
            m.fit(Xv, y)
        np.save(BRANCHES / f'val_hepatic_{cid}.npy', m.predict(X_va))
        print(f'  saved branches/{cid}')


if __name__ == '__main__':
    main()
