"""Per-individual-sample reverse engineering.

For each contested val_test patient (top by ensemble disagreement IQR),
profile the patient + look up their 5 nearest training neighbours +
identify which BASE MODEL handled those training neighbours best. Build a
per-patient override candidate.

Output:
  AUDIT_PER_PATIENT.md — qualitative drill-down per top-30 contested patient
  sub_master_per_patient_specialist.csv — anchor where trustworthy, per-patient
    optimal-base-model where contested.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src')); sys.path.insert(0, str(REPO / 'tests'))
from kirby.datasets.annitia_maslg import load_annitia
from experiments.annitia.views import build_views
from experiments.annitia.survival import concordance

VAL_CSV = '/Volumes/seagate/kirby/annitia/val_test.csv'
FULL = REPO / 'tests/results/annitia/full'
BRANCHES = REPO / 'tests/results/annitia/branches'
ANCHOR = FULL / 'sub_master_meta_avg_aware_weighted.csv'
N_TOP = 30  # contested patients to drill into


def r01(x):
    return rankdata(x) / len(x)


def main():
    print('Loading...')
    train = load_annitia()
    val = load_annitia(data_dir=str(Path(VAL_CSV).parent),
                        filename=Path(VAL_CSV).name, has_outcomes=False)
    vt = build_views(train, view_names=('snapshot', 'clinical_pro'))
    vv = build_views(val,   view_names=('snapshot', 'clinical_pro'))
    X_tr = np.concatenate([vt[v][0] for v in ('snapshot', 'clinical_pro')], axis=1)
    X_va = np.concatenate([vv[v][0] for v in ('snapshot', 'clinical_pro')], axis=1)

    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)

    # Load contention table — top contested per endpoint
    cont_h = pd.read_csv(FULL / 'valtest_contention_hepatic.csv').head(N_TOP)
    cont_d = pd.read_csv(FULL / 'valtest_contention_death.csv').head(5)
    contested_ids = sorted(set(cont_h['trustii_id'].tolist() + cont_d['trustii_id'].tolist()))
    print(f'  contested patients: {len(contested_ids)} (hep top {N_TOP} + death top 5)')

    # Map trustii_id back to position in val arrays
    val_meta = pd.read_csv(VAL_CSV)[['trustii_id', 'patient_id_anon']]
    g2i = dict(zip(val_meta['patient_id_anon'].astype(str),
                   val_meta['trustii_id'].astype(int)))
    val_int = np.array([g2i[str(g)] for g in val['patient_ids']])
    pos_by_tid = {tid: i for i, tid in enumerate(val_int)}

    # Load all base val_test predictions per endpoint
    def load_pool(endpoint):
        pred = {}
        for f in sorted((BRANCHES).glob(f'val_{endpoint}_*.npy')):
            try:
                a = np.load(f)
                if a.shape[0] != X_va.shape[0]: continue
                if not np.isfinite(a).any(): continue
                if np.std(a[np.isfinite(a)]) < 1e-10: continue
                cid = f.stem.replace(f'val_{endpoint}_', '')
                pred[cid] = a
            except Exception:
                continue
        # Also be/cd/hep_boost
        for d in ('be', 'cd', 'hep_boost', 'external', 'hparam'):
            dp = REPO / 'tests/results/annitia' / d
            if not dp.exists(): continue
            for f in sorted(dp.glob(f'val_{endpoint}_*.npy')):
                try:
                    a = np.load(f)
                    if a.shape[0] != X_va.shape[0]: continue
                    if not np.isfinite(a).any(): continue
                    if np.std(a[np.isfinite(a)]) < 1e-10: continue
                    cid = f.stem.replace(f'val_{endpoint}_', '')
                    if cid not in pred:
                        pred[cid] = a
                except Exception:
                    continue
        return pred

    def load_oof_pool(endpoint):
        # parallel: training-side OOFs to score per-model performance on
        # neighbour patients
        pred = {}
        for d in ('branches', 'be', 'cd', 'hep_boost', 'external', 'hparam'):
            dp = REPO / 'tests/results/annitia' / d
            if not dp.exists(): continue
            for f in sorted(dp.glob(f'oof_{endpoint}_*.npy')):
                try:
                    a = np.load(f)
                    if a.shape[0] != X_tr.shape[0]: continue
                    if not np.isfinite(a).any(): continue
                    if np.std(a[np.isfinite(a)]) < 1e-10: continue
                    cid = f.stem.replace(f'oof_{endpoint}_', '')
                    if cid not in pred:
                        pred[cid] = a
                except Exception:
                    continue
        return pred

    val_pool_h = load_pool('hepatic')
    val_pool_d = load_pool('death')
    oof_pool_h = load_oof_pool('hepatic')
    oof_pool_d = load_oof_pool('death')
    print(f'  val pool hepatic={len(val_pool_h)}, death={len(val_pool_d)}')
    print(f'  oof pool hepatic={len(oof_pool_h)}, death={len(oof_pool_d)}')

    # 5-NN training for each contested val patient
    nn = NearestNeighbors(n_neighbors=5).fit(X_tr_s)

    # Anchor predictions per val patient
    anchor = pd.read_csv(ANCHOR).set_index('trustii_id').reindex(val_int).reset_index()
    anchor_h = anchor['risk_hepatic_event'].values
    anchor_d = anchor['risk_death'].values

    # For each contested patient, find their best base model
    # by per-NN OOF C-index on training neighbours
    y_h = train['hepatic']['y_struct']; valid_h = train['hepatic']['valid_mask']
    y_d = train['death']['y_struct']; valid_d = train['death']['valid_mask']

    md_lines = ['# Per-individual-sample reverse engineering',
                '', f'Top-{N_TOP} contested hepatic + top-5 contested death val_test patients.',
                '', '|trustii_id|endpoint|val_age|fibroscan_max|fibrotest_max|t2dm|n_neighbours_with_event|best_model_for_neighbours|best_C|anchor_pred|recommended_pred|',
                '|---|---|---|---|---|---|---|---|---|---|---|']

    # Override predictions
    override_h = anchor_h.copy()
    override_d = anchor_d.copy()
    overridden_count = 0
    BLEND_W = 0.5  # 50/50 blend with anchor for contested patients

    long = train['longitudinal']
    long_v = val['longitudinal']

    def _max_ever(obs):
        try: vals = [v for _, v in obs if np.isfinite(v)]
        except:
            try: vals = [v for v in obs if np.isfinite(v)]
            except: vals = []
        return max(vals) if vals else float('nan')

    for tid in contested_ids:
        if tid not in pos_by_tid: continue
        vp = pos_by_tid[tid]
        # Profile the val patient
        v_age = val['baseline_age'][vp]
        v_fib = _max_ever(long_v['fibs_stiffness_med_BM_1'][vp])
        v_ftest = _max_ever(long_v['fibrotest_BM_2'][vp])
        _t2dm_raw = val['static']['T2DM'][vp] if 'T2DM' in val['static'] else 0
        v_t2dm = int(_t2dm_raw) if np.isfinite(_t2dm_raw) else -1
        # 5-NN training neighbours
        _, nb = nn.kneighbors(X_va_s[vp:vp+1])
        nb = nb[0]

        # Determine endpoint: top of contention list
        is_hep_contested = tid in cont_h['trustii_id'].values
        endpoint = 'hep' if is_hep_contested else 'death'
        valid_ep = valid_h if endpoint == 'hep' else valid_d
        oof_pool = oof_pool_h if endpoint == 'hep' else oof_pool_d
        val_pool = val_pool_h if endpoint == 'hep' else val_pool_d
        y_ep = y_h if endpoint == 'hep' else y_d

        # Filter neighbours to valid for this endpoint
        nb_valid = [n for n in nb if valid_ep[n]]
        n_with_event = sum(1 for n in nb_valid if y_ep[n]['event'])
        if not nb_valid:
            best_model = None; best_C = float('nan'); rec_pred = anchor_h[vp] if endpoint == 'hep' else anchor_d[vp]
        else:
            # Best-on-neighbours: for each base model, score its OOF on the
            # neighbour set (need ≥2 neighbours with event for C-index)
            best_C = -1.0
            best_model = None
            if sum(1 for n in nb_valid if y_ep[n]['event']) >= 1 and len(nb_valid) >= 2:
                for cid, oof in oof_pool.items():
                    nb_ok = [n for n in nb_valid if np.isfinite(oof[n])]
                    if len(nb_ok) < 2: continue
                    if sum(1 for n in nb_ok if y_ep[n]['event']) < 1: continue
                    try:
                        c = concordance(y_ep[nb_ok], oof[nb_ok])
                    except Exception:
                        continue
                    if np.isfinite(c) and c > best_C:
                        best_C = c; best_model = cid
            if best_model is not None and best_model in val_pool:
                rec_pred = val_pool[best_model][vp]
                # Override (rank-blended with anchor 50/50)
                # Need to rank the best_model's val_pool first
                model_ranks = r01(val_pool[best_model])
                if endpoint == 'hep':
                    anchor_rank_v = r01(anchor_h)
                    override_h[vp] = (1 - BLEND_W) * anchor_rank_v[vp] + BLEND_W * model_ranks[vp]
                else:
                    anchor_rank_v = r01(anchor_d)
                    override_d[vp] = (1 - BLEND_W) * anchor_rank_v[vp] + BLEND_W * model_ranks[vp]
                overridden_count += 1
            else:
                rec_pred = anchor_h[vp] if endpoint == 'hep' else anchor_d[vp]

        md_lines.append(
            f'|{tid}|{endpoint}|{v_age:.1f}|{v_fib:.1f}|{v_ftest:.2f}|{v_t2dm}|'
            f'{n_with_event}/{len(nb_valid)}|{best_model[:40] if best_model else "-"}|'
            f'{best_C:.3f}|{anchor_h[vp]:.3f}|{rec_pred:.3f}|'
        )

    print(f'  overridden {overridden_count}/{len(contested_ids)} contested patients')

    # Re-rank to enforce [0, 1]
    out = pd.DataFrame({
        'trustii_id': val_int,
        'risk_hepatic_event': r01(override_h),
        'risk_death': r01(override_d),
    }).sort_values('trustii_id').reset_index(drop=True)
    out_path = FULL / 'sub_master_per_patient_specialist.csv'
    out.to_csv(out_path, index=False)
    print(f'wrote {out_path}')

    audit = REPO / 'AUDIT_PER_PATIENT.md'
    audit.write_text('\n'.join(md_lines))
    print(f'wrote {audit}')


if __name__ == '__main__':
    main()
