"""Fine-Gray competing-risk Cox on hep — never run as a base model.

Hep is competing-risk with death. Standard Cox treats death as random
censoring → underestimates hep risk for old patients (censored by death not
end-of-followup). Fine-Gray subdistribution model captures this.

5-fold OOF + refit on full → val_test prediction.
Output:
  oof_hepatic_finegray__snapshot+clinical_pro+fpca+gp_smooth.npy
  val_hepatic_finegray__snapshot+clinical_pro+fpca+gp_smooth.npy
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src')); sys.path.insert(0, str(REPO / 'tests'))
from kirby.datasets.annitia_maslg import load_annitia
from experiments.annitia.views import build_views
from experiments.annitia.survival import concordance
from experiments.annitia.fine_gray import FineGrayCompetingRisks

VAL_CSV = '/Volumes/seagate/kirby/annitia/val_test.csv'
BRANCHES = REPO / 'tests/results/annitia/branches'
PRIMARY = ('snapshot', 'clinical_pro', 'fpca', 'gp_smooth')


def main():
    print('Loading...')
    train = load_annitia()
    val = load_annitia(data_dir=str(Path(VAL_CSV).parent),
                        filename=Path(VAL_CSV).name, has_outcomes=False)
    vt = build_views(train, view_names=PRIMARY)
    vv = build_views(val,   view_names=PRIMARY)
    X_tr = np.concatenate([vt[v][0] for v in PRIMARY], axis=1)
    X_va = np.concatenate([vv[v][0] for v in PRIMARY], axis=1)
    print(f'  X_tr: {X_tr.shape}')

    y_hep = train['hepatic']['y_struct']
    y_death = train['death']['y_struct']
    valid_hep = train['hepatic']['valid_mask']

    # 5-fold OOF
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.full(X_tr.shape[0], np.nan)
    for k, (tr, te) in enumerate(skf.split(X_tr[valid_hep],
                                             y_hep[valid_hep]['event'].astype(int))):
        idx = np.where(valid_hep)[0]
        tr_g, te_g = idx[tr], idx[te]
        m = FineGrayCompetingRisks(cause_of_interest='hepatic')
        try:
            m.fit(X_tr[tr_g], y_hep[tr_g], y_death[tr_g])
            risk = m.predict_risk(X_tr[te_g])
        except Exception as e:
            print(f'  fold {k}: failed {type(e).__name__}: {str(e)[:100]}; falling back to NaN')
            risk = np.full(len(te_g), np.nan)
        oof[te_g] = risk
        ok = np.isfinite(oof[te_g])
        if ok.sum() > 5:
            print(f'  fold {k}: C={concordance(y_hep[te_g][ok], oof[te_g][ok]):.4f}')

    # Pool C
    okf = valid_hep & np.isfinite(oof)
    pooled_C = concordance(y_hep[okf], oof[okf])
    print(f'pooled OOF C (hepatic, fine-gray): {pooled_C:.4f}')

    # Save OOF
    cid = 'finegray__' + '+'.join(PRIMARY)
    oof_path = BRANCHES / f'oof_hepatic_{cid}.npy'
    np.save(oof_path, oof)
    print(f'saved {oof_path}')

    # Refit on all valid → val_test prediction
    m = FineGrayCompetingRisks(cause_of_interest='hepatic')
    m.fit(X_tr[valid_hep], y_hep[valid_hep], y_death[valid_hep])
    val_pred = m.predict_risk(X_va)
    val_path = BRANCHES / f'val_hepatic_{cid}.npy'
    np.save(val_path, val_pred)
    print(f'saved {val_path}')


if __name__ == '__main__':
    main()
