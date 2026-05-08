"""Anchor-residual stacker — learn where the anchor is wrong, fix it.

The anchor (sub_master_meta_avg_aware_weighted) hits 0.9302 LB. Where it gets
wrong, it gets wrong systematically. Train a small model on training rows:
  features: clinical view (snapshot+nit+max-ever)
  target: signed residual = anchor_rank - true_rank (high → anchor over-ranked
          this patient; low → under-ranked)
At val_test, predict the residual and SUBTRACT from anchor's predicted rank.
This is a per-row correction, NOT a re-aggregation.

Two corrections: hep + death.

Output: sub_master_residual_stacker.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src')); sys.path.insert(0, str(REPO / 'tests'))
from kirby.datasets.annitia_maslg import load_annitia
from experiments.annitia.survival import concordance

VAL_CSV = '/Volumes/seagate/kirby/annitia/val_test.csv'
FULL = REPO / 'tests/results/annitia/full'
BRANCHES = REPO / 'tests/results/annitia/branches'
ANCHOR = FULL / 'sub_master_meta_avg_aware_weighted.csv'


def _max_ever(longitudinal_for_lab):
    out = np.full(len(longitudinal_for_lab), np.nan)
    for i, obs in enumerate(longitudinal_for_lab):
        if obs is None: continue
        try: vals = [v for _, v in obs if np.isfinite(v)]
        except:
            try: vals = [v for v in obs if np.isfinite(v)]
            except: vals = []
        if vals: out[i] = max(vals)
    return out


def _last_ever(longitudinal_for_lab):
    out = np.full(len(longitudinal_for_lab), np.nan)
    for i, obs in enumerate(longitudinal_for_lab):
        if obs is None: continue
        try:
            valid = [(a, v) for a, v in obs if np.isfinite(a) and np.isfinite(v)]
        except:
            valid = []
        if valid:
            valid.sort()
            out[i] = valid[-1][1]
    return out


def build_features(d):
    long = d['longitudinal']; static = d['static']
    n = len(d['patient_ids'])
    cols = []
    feats = []
    for lab in ['fibs_stiffness_med_BM_1', 'fibrotest_BM_2', 'aixp_aix_result_BM_3',
                'BMI', 'alt', 'ast', 'bilirubin', 'chol', 'ggt', 'gluc_fast',
                'plt', 'triglyc']:
        if lab not in long: continue
        feats.append(_max_ever(long[lab])); cols.append(f'{lab}_max')
        feats.append(_last_ever(long[lab])); cols.append(f'{lab}_last')
    feats.append(np.asarray(d['baseline_age'], dtype=float)); cols.append('baseline_age')
    feats.append(np.asarray(d['last_followup_age'], dtype=float) -
                  np.asarray(d['baseline_age'], dtype=float)); cols.append('followup_dur')
    for sk in ['T2DM', 'gender', 'Hypertension', 'Dyslipidaemia']:
        if sk in static:
            feats.append(np.asarray(static[sk], dtype=float)); cols.append(sk)
    X = np.column_stack(feats)
    for j in range(X.shape[1]):
        m = ~np.isfinite(X[:, j])
        if m.any():
            mean = np.nanmean(X[~m, j]) if (~m).any() else 0.0
            X[m, j] = mean
    return X, cols


def main():
    print('Loading...')
    train = load_annitia()
    val = load_annitia(data_dir=str(Path(VAL_CSV).parent),
                        filename=Path(VAL_CSV).name, has_outcomes=False)
    X_tr, cols = build_features(train)
    X_va, _ = build_features(val)
    print(f'  features: {len(cols)}; train n={X_tr.shape[0]}, val n={X_va.shape[0]}')

    # Reconstruct anchor's training-side rank prediction via the consensus
    # mean-rank of base OOFs (same proxy as oof_to_leaderboard.py).
    def mean_rank_oofs(endpoint):
        files = []
        for d in (BRANCHES, REPO / 'tests/results/annitia/be',
                  REPO / 'tests/results/annitia/cd',
                  REPO / 'tests/results/annitia/hep_boost',
                  REPO / 'tests/results/annitia/external'):
            if d.exists():
                files.extend(d.glob(f'oof_{endpoint}_*.npy'))
        ranks = []
        ref_n = None
        for f in files:
            try:
                a = np.load(f)
                if ref_n is None: ref_n = a.shape[0]
                if a.shape[0] != ref_n: continue
                if np.isfinite(a).sum() < 50: continue
                if np.std(a[np.isfinite(a)]) < 1e-10: continue
                m = np.isfinite(a)
                r = np.full(a.shape[0], np.nan)
                if m.sum(): r[m] = rankdata(a[m]) / m.sum()
                ranks.append(r)
            except Exception:
                continue
        return np.nanmean(np.stack(ranks, axis=1), axis=1) if ranks else None

    anchor_proxy = {ep: mean_rank_oofs(ep) for ep in ('hepatic', 'death')}
    for ep, a in anchor_proxy.items():
        print(f'  anchor proxy {ep}: n_finite={np.isfinite(a).sum()}')

    # For each endpoint: target = signed residual (anchor_rank - true_event_rank)
    # We use the survival event flag indirectly: high-risk training patients
    # are those who had events early. True risk rank ≈ rank of (-time_to_event)
    # for events, censored at +inf for non-events. Simpler: use C-index pair
    # contributions per patient as a proxy "did anchor get this patient right".

    val_meta = pd.read_csv(VAL_CSV)[['trustii_id', 'patient_id_anon']]
    g2i = dict(zip(val_meta['patient_id_anon'].astype(str),
                   val_meta['trustii_id'].astype(int)))
    val_int = np.array([g2i[str(g)] for g in val['patient_ids']])
    anchor = pd.read_csv(ANCHOR).set_index('trustii_id').reindex(val_int).reset_index()

    # Train per-endpoint stacker
    out_pred = {}
    for endpoint, anchor_col in (('hepatic', 'risk_hepatic_event'),
                                  ('death', 'risk_death')):
        y_struct = train[endpoint]['y_struct']
        valid = train[endpoint]['valid_mask']
        a_proxy = anchor_proxy[endpoint]
        if a_proxy is None or not np.isfinite(a_proxy).any():
            print(f'  no anchor proxy for {endpoint}; skip stacker'); continue
        ev = y_struct['event'].astype(bool)
        tt = y_struct['time'].astype(float)

        # Target: signed residual on training. Define "true rank" via
        # event-aware ordering: events get rank by 1/time (early event = high
        # risk = high rank); censored get rank by 1/(time+1) (lower).
        # That's a proxy — the C-index actually only ranks pairs. But for
        # regressing anchor's error, this gives a continuous target.
        true_rank = np.full(len(ev), 0.5)
        # Among valid patients only, with event:
        v_ev = valid & ev
        v_ce = valid & ~ev
        if v_ev.any():
            r_ev = rankdata(-tt[v_ev]) / max(v_ev.sum(), 1)  # earlier event → higher
            true_rank[v_ev] = 0.5 + 0.5 * r_ev  # in [0.5, 1.0]
        if v_ce.any():
            r_ce = rankdata(-tt[v_ce]) / max(v_ce.sum(), 1)  # later censoring → higher (less risk, lower rank? flip)
            # Censored survivors → lower risk. Map to [0, 0.5].
            true_rank[v_ce] = 0.5 - 0.5 * r_ce

        residual = a_proxy - true_rank  # positive = anchor over-ranked the patient
        # Train GBR(residual ~ X) with 5-fold OOF
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_resid = np.full(X_tr.shape[0], 0.0)
        # Use ev as stratification target so folds have ~equal events
        for tr, te in skf.split(X_tr[valid], ev[valid].astype(int)):
            valid_idx = np.where(valid)[0]
            tr_g = valid_idx[tr]; te_g = valid_idx[te]
            m = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                            learning_rate=0.05, random_state=42)
            # Filter for finite anchor proxy + finite residual
            ok = np.isfinite(residual[tr_g]) & np.isfinite(a_proxy[tr_g])
            m.fit(X_tr[tr_g][ok], residual[tr_g][ok])
            oof_resid[te_g] = m.predict(X_tr[te_g])

        # Refit on all valid → predict val
        ok_all = valid & np.isfinite(residual) & np.isfinite(a_proxy)
        m = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                       learning_rate=0.05, random_state=42)
        m.fit(X_tr[ok_all], residual[ok_all])
        val_resid = m.predict(X_va)

        # Apply correction at val: corrected_rank = anchor_rank - α * residual_pred
        # α tuned via training OOF C
        anchor_rank_v = rankdata(anchor[anchor_col].values) / len(anchor)
        # Find α that maximises OOF C of (a_proxy - α * oof_resid) against y
        best_a, best_c = 0.0, -1.0
        if valid.any() and a_proxy is not None:
            y_v = y_struct[valid]
            for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
                corrected = a_proxy[valid] - alpha * oof_resid[valid]
                c = concordance(y_v, corrected)
                if c > best_c:
                    best_c = c; best_a = alpha
            print(f'  {endpoint}: best α={best_a}, OOF C={best_c:.4f} '
                  f'(vs anchor proxy {concordance(y_v, a_proxy[valid]):.4f})')

        out_rank_v = anchor_rank_v - best_a * (val_resid - val_resid.mean())
        out_pred[endpoint] = rankdata(out_rank_v) / len(out_rank_v)

    if 'hepatic' in out_pred and 'death' in out_pred:
        out = pd.DataFrame({
            'trustii_id': val_int,
            'risk_hepatic_event': out_pred['hepatic'],
            'risk_death': out_pred['death'],
        }).sort_values('trustii_id').reset_index(drop=True)
        path = FULL / 'sub_master_residual_stacker.csv'
        out.to_csv(path, index=False)
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
