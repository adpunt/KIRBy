"""Deep-dive failure audit for ANNITIA hepatic + death endpoints.

Identifies FN-like / FP-like training patients, characterizes them, builds a
discriminator, applies it to val_test, and generates targeted/specialist/isolation
submission variants. See AUDIT_DEEP_FAILURES.md for findings.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))

from kirby.datasets.annitia_maslg import load_annitia  # noqa: E402

FULL = REPO / 'tests/results/annitia/full'
OOF_DIRS = [
    REPO / 'tests/results/annitia/branches',
    REPO / 'tests/results/annitia/be',
    REPO / 'tests/results/annitia/cd',
    REPO / 'tests/results/annitia/hep_boost',
]


def rank01(x: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    m = np.isfinite(x)
    if m.sum():
        out[m] = rankdata(x[m]) / max(m.sum(), 1)
    return out


def load_consensus_and_per_model(endpoint: str, n_train: int):
    """Returns (consensus_rank01, dict[model_id -> rank01 array])."""
    per_model = {}
    arrs = []
    for d in OOF_DIRS:
        for f in d.glob(f'oof_{endpoint}_*.npy'):
            try:
                a = np.load(f)
            except Exception:
                continue
            if a.shape[0] != n_train:
                continue
            if np.isfinite(a).sum() < 50:
                continue
            if np.std(a[np.isfinite(a)]) <= 1e-10:
                continue
            cid = f"{d.name}__{f.stem.replace(f'oof_{endpoint}_', '')}"
            r = rank01(a)
            per_model[cid] = r
            arrs.append(r)
    consensus = np.nanmean(np.stack(arrs, axis=1), axis=1) if arrs else None
    return consensus, per_model


def load_val_consensus(endpoint: str):
    arrs = []
    n_val = None
    per_model = {}
    for d in OOF_DIRS:
        for f in d.glob(f'val_{endpoint}_*.npy'):
            try:
                a = np.load(f)
            except Exception:
                continue
            if np.isfinite(a).sum() < 10 or np.std(a[np.isfinite(a)]) <= 1e-10:
                continue
            if n_val is None:
                n_val = a.shape[0]
            if a.shape[0] != n_val:
                continue
            cid = f"{d.name}__{f.stem.replace(f'val_{endpoint}_', '')}"
            r = rank01(a)
            per_model[cid] = r
            arrs.append(r)
    consensus = np.nanmean(np.stack(arrs, axis=1), axis=1) if arrs else None
    return consensus, per_model, n_val


def build_feature_matrix(data: dict) -> tuple[np.ndarray, list[str]]:
    """Per-patient baseline feature vector for analysis + discriminator."""
    n = len(data['patient_ids'])
    wide = data['wide']
    feats = {}
    feats['baseline_age'] = data['baseline_age']
    feats['last_followup_age'] = data['last_followup_age']
    feats['followup_span'] = data['last_followup_age'] - data['baseline_age']
    # First-visit value for labs
    for var in ['BMI', 'alt', 'ast', 'plt', 'chol', 'ggt', 'gluc_fast',
                'triglyc', 'bilirubin', 'fibs_stiffness_med_BM_1',
                'fibrotest_BM_2', 'aixp_aix_result_BM_3']:
        arr = wide.get(var)
        if arr is None:
            feats[f'{var}_first'] = np.full(n, np.nan)
            feats[f'{var}_last'] = np.full(n, np.nan)
            continue
        first = np.full(n, np.nan)
        last = np.full(n, np.nan)
        for i in range(n):
            row = arr[i]
            row_f = row[np.isfinite(row)]
            if row_f.size:
                first[i] = row_f[0]
                last[i] = row_f[-1]
        feats[f'{var}_first'] = first
        feats[f'{var}_last'] = last
    # Static
    for col, vals in data['static'].items():
        feats[col] = vals
    names = list(feats.keys())
    X = np.column_stack([feats[k] for k in names])
    return X, names


def build_val_feature_matrix(val_data: dict, names: list[str]) -> np.ndarray:
    """Same feature schema as training, applied to val."""
    n = len(val_data['patient_ids'])
    wide = val_data['wide']
    feats = {}
    feats['baseline_age'] = val_data['baseline_age']
    feats['last_followup_age'] = val_data['last_followup_age']
    feats['followup_span'] = val_data['last_followup_age'] - val_data['baseline_age']
    for var in ['BMI', 'alt', 'ast', 'plt', 'chol', 'ggt', 'gluc_fast',
                'triglyc', 'bilirubin', 'fibs_stiffness_med_BM_1',
                'fibrotest_BM_2', 'aixp_aix_result_BM_3']:
        arr = wide.get(var)
        if arr is None:
            feats[f'{var}_first'] = np.full(n, np.nan)
            feats[f'{var}_last'] = np.full(n, np.nan)
            continue
        first = np.full(n, np.nan)
        last = np.full(n, np.nan)
        for i in range(n):
            row = arr[i]
            row_f = row[np.isfinite(row)]
            if row_f.size:
                first[i] = row_f[0]
                last[i] = row_f[-1]
        feats[f'{var}_first'] = first
        feats[f'{var}_last'] = last
    for col, vals in val_data['static'].items():
        feats[col] = vals
    return np.column_stack([feats[k] if k in feats else np.full(n, np.nan)
                            for k in names])


def label_buckets(consensus: np.ndarray, event: np.ndarray, time: np.ndarray,
                  valid: np.ndarray, time_thresh: float = 5.0):
    """Return bucket labels: 0=trust, 1=FN-like, 2=FP-like."""
    labels = np.zeros(len(event), dtype=int)
    finite = np.isfinite(consensus) & valid
    if finite.sum() == 0:
        return labels
    q25 = np.nanquantile(consensus[finite], 0.25)
    q75 = np.nanquantile(consensus[finite], 0.75)
    fn_mask = (event == 1) & finite & (consensus <= q25)
    fp_mask = (event == 0) & finite & (consensus >= q75) & (time >= time_thresh)
    labels[fn_mask] = 1
    labels[fp_mask] = 2
    return labels


def feature_summary(X: np.ndarray, names: list[str], idx: np.ndarray) -> dict:
    return {n: float(np.nanmean(X[idx, i])) for i, n in enumerate(names)}


def train_discriminator(X: np.ndarray, labels: np.ndarray):
    """Train multiclass {trust=0, FN=1, FP=2} -> 5-fold CV AUROC.

    Returns (mean_auroc_dict, fitted_pipe_on_full_data).
    """
    pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scl', StandardScaler()),
        ('clf', GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                           learning_rate=0.05, random_state=0)),
    ])
    # Per-class one-vs-rest AUROC via 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    aurocs = {1: [], 2: []}
    for tr, te in skf.split(X, labels):
        if (labels[tr] == 1).sum() < 2 or (labels[tr] == 2).sum() < 2:
            continue
        pipe.fit(X[tr], labels[tr])
        proba = pipe.predict_proba(X[te])
        classes = pipe.named_steps['clf'].classes_
        for cls in (1, 2):
            if cls not in classes:
                continue
            ci = list(classes).index(cls)
            ytrue = (labels[te] == cls).astype(int)
            if ytrue.sum() == 0 or ytrue.sum() == len(ytrue):
                continue
            try:
                aurocs[cls].append(roc_auc_score(ytrue, proba[:, ci]))
            except Exception:
                pass
    mean_auroc = {k: (float(np.mean(v)) if v else float('nan')) for k, v in aurocs.items()}
    pipe.fit(X, labels)
    return mean_auroc, pipe


def nearest_correct_event_neighbours(X: np.ndarray, idx_query: int,
                                     correct_event_idx: np.ndarray,
                                     names: list[str], k: int = 5):
    """Find k-NN neighbours among correctly-flagged events."""
    if len(correct_event_idx) == 0:
        return []
    # Use a small subset of stable features (impute median)
    keep = ['baseline_age', 'last_followup_age', 'followup_span',
            'BMI_first', 'alt_first', 'ast_first', 'plt_first',
            'fibs_stiffness_med_BM_1_first', 'fibrotest_BM_2_first']
    cols = [names.index(k) for k in keep if k in names]
    Xq = X[:, cols].copy()
    med = np.nanmedian(Xq, axis=0)
    inds_nan = np.where(np.isnan(Xq))
    Xq[inds_nan] = np.take(med, inds_nan[1])
    sd = np.nanstd(Xq, axis=0) + 1e-9
    Xs = (Xq - np.nanmean(Xq, axis=0)) / sd
    nn = NearestNeighbors(n_neighbors=min(k, len(correct_event_idx)))
    nn.fit(Xs[correct_event_idx])
    d, ii = nn.kneighbors(Xs[idx_query:idx_query+1])
    return [int(correct_event_idx[j]) for j in ii[0]]


def integerize_trustii(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['trustii_id'] = df['trustii_id'].astype(int)
    return df


def main():
    print('Loading training data...')
    data = load_annitia()
    n_train = len(data['patient_ids'])
    print(f'  n_train = {n_train}')

    X, feat_names = build_feature_matrix(data)
    findings = {'feat_names': feat_names}

    # ---- HEPATIC ENDPOINT ----
    print('\n=== HEPATIC ===')
    ep = data['hepatic']
    consensus_h, per_model_h = load_consensus_and_per_model('hepatic', n_train)
    print(f'  models: {len(per_model_h)}, consensus shape: {consensus_h.shape}')
    labels_h = label_buckets(consensus_h, ep['event'], ep['time'], ep['valid_mask'])
    n_trust = (labels_h == 0).sum()
    n_fn = (labels_h == 1).sum()
    n_fp = (labels_h == 2).sum()
    print(f'  trust={n_trust}, FN-like={n_fn}, FP-like={n_fp}')
    findings['hepatic'] = {'n_train_valid': int(ep['valid_mask'].sum()),
                            'n_events': int(ep['event'].sum()),
                            'n_FN': int(n_fn), 'n_FP': int(n_fp),
                            'n_trust': int(n_trust)}

    # FN-like deep dive
    fn_idx = np.where(labels_h == 1)[0]
    correct_event_idx = np.where((ep['event'] == 1) & (consensus_h >= 0.75) &
                                 ep['valid_mask'])[0]
    print(f'  correctly-flagged events (high-risk consensus): {len(correct_event_idx)}')
    fn_summaries = []
    for i in fn_idx:
        s = {'patient_id': data['patient_ids'][i],
             'idx': int(i),
             'event': int(ep['event'][i]),
             'time_yrs': float(ep['time'][i]),
             'consensus_rank': float(consensus_h[i])}
        for fname in ['baseline_age', 'last_followup_age', 'followup_span',
                      'BMI_first', 'alt_first', 'ast_first', 'plt_first',
                      'ggt_first', 'fibs_stiffness_med_BM_1_first',
                      'fibrotest_BM_2_first', 'gluc_fast_first']:
            j = feat_names.index(fname)
            s[fname] = float(X[i, j]) if np.isfinite(X[i, j]) else None
        s['nn_correct_event'] = nearest_correct_event_neighbours(
            X, i, correct_event_idx, feat_names, k=5)
        fn_summaries.append(s)
    findings['hepatic_FN'] = fn_summaries

    # FP-like top 30 by discordance
    fp_idx = np.where(labels_h == 2)[0]
    if len(fp_idx) > 0:
        # discordance proxy = consensus rank since event=0 expected low
        fp_order = fp_idx[np.argsort(-consensus_h[fp_idx])][:30]
        fp_summaries = []
        for i in fp_order:
            s = {'patient_id': data['patient_ids'][i],
                 'idx': int(i),
                 'time_yrs': float(ep['time'][i]),
                 'consensus_rank': float(consensus_h[i])}
            for fname in ['baseline_age', 'last_followup_age', 'followup_span',
                          'BMI_first', 'alt_first', 'ast_first', 'plt_first',
                          'fibs_stiffness_med_BM_1_first', 'fibrotest_BM_2_first']:
                j = feat_names.index(fname)
                s[fname] = float(X[i, j]) if np.isfinite(X[i, j]) else None
            fp_summaries.append(s)
        findings['hepatic_FP_top30'] = fp_summaries
    else:
        findings['hepatic_FP_top30'] = []

    # Population vs FN vs FP feature means
    pop_summary = feature_summary(X, feat_names, np.where(ep['valid_mask'])[0])
    fn_mean = feature_summary(X, feat_names, fn_idx) if len(fn_idx) else {}
    fp_mean = feature_summary(X, feat_names, fp_idx) if len(fp_idx) else {}
    findings['hepatic_means'] = {'pop': pop_summary, 'FN': fn_mean, 'FP': fp_mean}

    # Discriminator
    print('  training discriminator (5-fold CV)...')
    auroc_h, disc_h = train_discriminator(X[ep['valid_mask']],
                                          labels_h[ep['valid_mask']])
    print(f'  AUROC FN-vs-rest = {auroc_h.get(1, float("nan")):.3f}, '
          f'FP-vs-rest = {auroc_h.get(2, float("nan")):.3f}')
    findings['hepatic_discriminator_AUROC'] = auroc_h

    # ---- DEATH ENDPOINT ----
    print('\n=== DEATH ===')
    ep_d = data['death']
    consensus_d, per_model_d = load_consensus_and_per_model('death', n_train)
    print(f'  models: {len(per_model_d)}, consensus shape: {consensus_d.shape}')
    labels_d = label_buckets(consensus_d, ep_d['event'], ep_d['time'], ep_d['valid_mask'])
    n_trust_d = (labels_d == 0).sum()
    n_fn_d = (labels_d == 1).sum()
    n_fp_d = (labels_d == 2).sum()
    print(f'  trust={n_trust_d}, FN-like={n_fn_d}, FP-like={n_fp_d}')
    findings['death'] = {'n_train_valid': int(ep_d['valid_mask'].sum()),
                          'n_events': int(ep_d['event'].sum()),
                          'n_FN': int(n_fn_d), 'n_FP': int(n_fp_d),
                          'n_trust': int(n_trust_d)}
    fn_d_idx = np.where(labels_d == 1)[0]
    fp_d_idx = np.where(labels_d == 2)[0]
    pop_d = feature_summary(X, feat_names, np.where(ep_d['valid_mask'])[0])
    fn_d_mean = feature_summary(X, feat_names, fn_d_idx) if len(fn_d_idx) else {}
    fp_d_mean = feature_summary(X, feat_names, fp_d_idx) if len(fp_d_idx) else {}
    findings['death_means'] = {'pop': pop_d, 'FN': fn_d_mean, 'FP': fp_d_mean}
    auroc_d, disc_d = train_discriminator(X[ep_d['valid_mask']],
                                          labels_d[ep_d['valid_mask']])
    print(f'  AUROC FN-vs-rest = {auroc_d.get(1, float("nan")):.3f}, '
          f'FP-vs-rest = {auroc_d.get(2, float("nan")):.3f}')
    findings['death_discriminator_AUROC'] = auroc_d

    # ---- VAL_TEST: load + classify ----
    print('\n=== VAL ===')
    try:
        val_data = load_annitia(filename='val_test.csv', has_outcomes=False)
    except Exception as e:
        print(f'  fallback: trying validation.csv ({e})')
        try:
            val_data = load_annitia(filename='validation.csv', has_outcomes=False)
        except Exception as e2:
            print(f'  failed: {e2}')
            val_data = None

    if val_data is not None:
        X_val = build_val_feature_matrix(val_data, feat_names)
        proba_h_val = disc_h.predict_proba(X_val)
        classes_h = disc_h.named_steps['clf'].classes_
        proba_d_val = disc_d.predict_proba(X_val)
        classes_d = disc_d.named_steps['clf'].classes_

        def get_proba(proba, classes, cls):
            if cls not in classes:
                return np.zeros(proba.shape[0])
            return proba[:, list(classes).index(cls)]

        p_fn_h = get_proba(proba_h_val, classes_h, 1)
        p_fp_h = get_proba(proba_h_val, classes_h, 2)
        p_fn_d = get_proba(proba_d_val, classes_d, 1)
        p_fp_d = get_proba(proba_d_val, classes_d, 2)

        # Bucket val patients (top quartile of P(FN), P(FP) → bucket assignment)
        # Pick threshold such that bucket sizes proportional to training rates.
        n_val = X_val.shape[0]
        target_fn_h = max(int(round(n_val * n_fn / max(ep['valid_mask'].sum(), 1))), 5)
        target_fp_h = max(int(round(n_val * n_fp / max(ep['valid_mask'].sum(), 1))), 5)
        target_fn_d = max(int(round(n_val * n_fn_d / max(ep_d['valid_mask'].sum(), 1))), 5)
        target_fp_d = max(int(round(n_val * n_fp_d / max(ep_d['valid_mask'].sum(), 1))), 5)

        order_fn_h = np.argsort(-p_fn_h)
        order_fp_h = np.argsort(-p_fp_h)
        order_fn_d = np.argsort(-p_fn_d)
        order_fp_d = np.argsort(-p_fp_d)

        bucket_h = np.zeros(n_val, dtype=int)
        bucket_h[order_fn_h[:target_fn_h]] = 1
        # FP only assigned where not already FN
        for i in order_fp_h:
            if bucket_h[i] == 0:
                bucket_h[i] = 2
                if (bucket_h == 2).sum() >= target_fp_h:
                    break
        bucket_d = np.zeros(n_val, dtype=int)
        bucket_d[order_fn_d[:target_fn_d]] = 1
        for i in order_fp_d:
            if bucket_d[i] == 0:
                bucket_d[i] = 2
                if (bucket_d == 2).sum() >= target_fp_d:
                    break

        print(f'  hepatic val buckets: trust={(bucket_h==0).sum()} '
              f'FN-prone={(bucket_h==1).sum()} FP-prone={(bucket_h==2).sum()}')
        print(f'  death val buckets:   trust={(bucket_d==0).sum()} '
              f'FN-prone={(bucket_d==1).sum()} FP-prone={(bucket_d==2).sum()}')
        findings['val_buckets'] = {
            'hepatic': {'trust': int((bucket_h==0).sum()),
                        'FN': int((bucket_h==1).sum()),
                        'FP': int((bucket_h==2).sum())},
            'death':   {'trust': int((bucket_d==0).sum()),
                        'FN': int((bucket_d==1).sum()),
                        'FP': int((bucket_d==2).sum())},
        }

        # ---- BUILD SUBMISSIONS ----
        print('\n=== Building submissions ===')
        anchor_path = FULL / 'sub_master_meta_avg_aware_weighted_v2.csv'
        anchor = pd.read_csv(anchor_path)
        # Sanity: anchor row order matches val_data row order (by trustii_id)
        if val_data['trustii_id'] is not None:
            v_tid = pd.to_numeric(pd.Series(val_data['trustii_id']),
                                   errors='coerce').to_numpy()
            order_map = pd.Series(np.arange(n_val), index=v_tid).to_dict()
            anchor_order = np.array([order_map.get(int(t), -1)
                                      for t in anchor['trustii_id']])
            if (anchor_order < 0).any():
                print('  WARNING: trustii_id mismatch between val and anchor')
            # Reindex p_fn/p_fp/bucket arrays to match anchor row order
            p_fn_h = p_fn_h[anchor_order]
            p_fp_h = p_fp_h[anchor_order]
            p_fn_d = p_fn_d[anchor_order]
            p_fp_d = p_fp_d[anchor_order]
            bucket_h = bucket_h[anchor_order]
            bucket_d = bucket_d[anchor_order]
            X_val_aligned = X_val[anchor_order]
        else:
            X_val_aligned = X_val

        anchor_h = anchor['risk_hepatic_event'].to_numpy()
        anchor_d = anchor['risk_death'].to_numpy()
        # Re-rank01 for safety
        anchor_h_r = rank01(anchor_h)
        anchor_d_r = rank01(anchor_d)

        # --- (a) sub_master_deepfn_targeted: boost FN, suppress FP ---
        # Offset proportional to discriminator confidence, gentle (avoid LB blow-up)
        def targeted_adjust(anchor_r, p_fn, p_fp, alpha=0.18):
            adj = anchor_r.copy()
            # Boost FN-prone
            adj = adj + alpha * (p_fn - p_fn.mean())
            # Suppress FP-prone
            adj = adj - alpha * (p_fp - p_fp.mean())
            return rank01(adj)

        out_targeted = anchor.copy()
        out_targeted['risk_hepatic_event'] = targeted_adjust(
            anchor_h_r, p_fn_h, p_fp_h, alpha=0.18)
        out_targeted['risk_death'] = targeted_adjust(
            anchor_d_r, p_fn_d, p_fp_d, alpha=0.20)
        out_targeted = integerize_trustii(out_targeted)
        out_targeted.to_csv(FULL / 'sub_master_deepfn_targeted.csv', index=False)
        print('  wrote sub_master_deepfn_targeted.csv')

        # --- (b) specialist: FN-prone → use models that correctly ranked FN-like training pts ---
        # Score each training-side model by its mean rank on FN-like training rows
        def best_models_for_fn(per_model: dict, fn_train_idx: np.ndarray,
                                top_k: int = 5):
            scores = []
            for cid, r in per_model.items():
                v = r[fn_train_idx]
                v = v[np.isfinite(v)]
                if v.size < 2:
                    continue
                scores.append((cid, np.nanmean(v)))  # higher = better (we want FN ranked high)
            scores.sort(key=lambda kv: -kv[1])
            return [cid for cid, _ in scores[:top_k]]

        top_fn_h_models = best_models_for_fn(per_model_h, fn_idx, top_k=8)
        top_fn_d_models = best_models_for_fn(per_model_d, fn_d_idx, top_k=8)
        print(f'  top FN-specialists hepatic: {top_fn_h_models[:3]}...')

        # Load corresponding val arrays
        _, val_per_model_h, _ = load_val_consensus('hepatic')
        _, val_per_model_d, _ = load_val_consensus('death')

        def specialist_pred(top_models: list, val_per_model: dict, n_val: int):
            arrs = [val_per_model[m] for m in top_models if m in val_per_model]
            if not arrs:
                return None
            stack = np.stack(arrs, axis=1)
            return rank01(np.nanmean(stack, axis=1))

        spec_h = specialist_pred(top_fn_h_models, val_per_model_h, n_val)
        spec_d = specialist_pred(top_fn_d_models, val_per_model_d, n_val)

        # Reindex specialist predictions to anchor row order (val arrays are in val_data row order)
        if val_data['trustii_id'] is not None and spec_h is not None:
            spec_h = spec_h[anchor_order]
        if val_data['trustii_id'] is not None and spec_d is not None:
            spec_d = spec_d[anchor_order]

        out_spec = anchor.copy()
        if spec_h is not None:
            blend_h = anchor_h_r.copy()
            mask_fn = (bucket_h == 1)
            blend_h[mask_fn] = 0.5 * anchor_h_r[mask_fn] + 0.5 * spec_h[mask_fn]
            out_spec['risk_hepatic_event'] = rank01(blend_h)
        else:
            out_spec['risk_hepatic_event'] = anchor_h_r
        if spec_d is not None:
            blend_d = anchor_d_r.copy()
            mask_fn_d = (bucket_d == 1)
            blend_d[mask_fn_d] = 0.5 * anchor_d_r[mask_fn_d] + 0.5 * spec_d[mask_fn_d]
            out_spec['risk_death'] = rank01(blend_d)
        else:
            out_spec['risk_death'] = anchor_d_r
        out_spec = integerize_trustii(out_spec)
        out_spec.to_csv(FULL / 'sub_master_deepfn_specialist.csv', index=False)
        print('  wrote sub_master_deepfn_specialist.csv')

        # --- (c) isolation: anomalous patients → trimmed-mean conservative blend ---
        iso = IsolationForest(n_estimators=200, random_state=0, contamination=0.15)
        # Fit on training, score val
        Xtr_imp = SimpleImputer(strategy='median').fit(X)
        Xtr_t = Xtr_imp.transform(X)
        iso.fit(Xtr_t)
        Xv_t = Xtr_imp.transform(X_val_aligned)
        anom = -iso.score_samples(Xv_t)  # higher = more anomalous
        anom_z = (anom - anom.mean()) / (anom.std() + 1e-9)

        # Trimmed-mean of all val per_model rank01 (anchor of conservative)
        def trimmed_mean_val(val_per_model: dict, n_val: int):
            arrs = list(val_per_model.values())
            if not arrs:
                return None
            stack = np.stack(arrs, axis=1)
            stack_r = np.apply_along_axis(rank01, 0, stack)
            # 20% trimmed
            k = max(int(stack_r.shape[1] * 0.2), 1)
            sorted_s = np.sort(stack_r, axis=1)
            trimmed = sorted_s[:, k:-k] if sorted_s.shape[1] > 2 * k else sorted_s
            return rank01(np.nanmean(trimmed, axis=1))

        trim_h = trimmed_mean_val(val_per_model_h, n_val)
        trim_d = trimmed_mean_val(val_per_model_d, n_val)
        if val_data['trustii_id'] is not None and trim_h is not None:
            trim_h = trim_h[anchor_order]
        if val_data['trustii_id'] is not None and trim_d is not None:
            trim_d = trim_d[anchor_order]

        out_iso = anchor.copy()
        # Where bucket != 0 OR anomaly is high → blend toward trimmed mean
        weight = np.clip((bucket_h != 0).astype(float) * 0.5 + np.clip(anom_z, 0, 2) * 0.15, 0, 0.6)
        if trim_h is not None:
            blend = (1 - weight) * anchor_h_r + weight * trim_h
            out_iso['risk_hepatic_event'] = rank01(blend)
        else:
            out_iso['risk_hepatic_event'] = anchor_h_r
        weight_d = np.clip((bucket_d != 0).astype(float) * 0.5 + np.clip(anom_z, 0, 2) * 0.15, 0, 0.6)
        if trim_d is not None:
            blend_d = (1 - weight_d) * anchor_d_r + weight_d * trim_d
            out_iso['risk_death'] = rank01(blend_d)
        else:
            out_iso['risk_death'] = anchor_d_r
        out_iso = integerize_trustii(out_iso)
        out_iso.to_csv(FULL / 'sub_master_deepfn_isolation.csv', index=False)
        print('  wrote sub_master_deepfn_isolation.csv')

        # --- (d) death-targeted variant (already produced inside (a)).  Save explicit copy too ---
        out_d_only = anchor.copy()
        out_d_only['risk_hepatic_event'] = anchor_h_r  # untouched
        out_d_only['risk_death'] = targeted_adjust(
            anchor_d_r, p_fn_d, p_fp_d, alpha=0.18)
        out_d_only = integerize_trustii(out_d_only)
        out_d_only.to_csv(FULL / 'sub_master_deepfn_targeted_death.csv', index=False)
        print('  wrote sub_master_deepfn_targeted_death.csv')

    # Save findings
    findings_path = FULL / 'deep_failure_audit.json'
    with open(findings_path, 'w') as f:
        json.dump(findings, f, indent=2, default=str)
    print(f'\nfindings → {findings_path}')

    # Markdown audit
    md = generate_markdown(findings)
    out_md = REPO / 'AUDIT_DEEP_FAILURES.md'
    out_md.write_text(md)
    print(f'audit → {out_md}')


def generate_markdown(findings: dict) -> str:
    lines = ['# Deep Failure Audit — ANNITIA hepatic + death', '']
    h = findings['hepatic']
    d = findings['death']
    lines.append('## Summary')
    lines.append('')
    lines.append(f'- **Hepatic**: {h["n_train_valid"]} valid training patients, '
                 f'{h["n_events"]} events. FN-like={h["n_FN"]}, FP-like={h["n_FP"]}, '
                 f'trust={h["n_trust"]}.')
    lines.append(f'- **Death**:   {d["n_train_valid"]} valid training patients, '
                 f'{d["n_events"]} events. FN-like={d["n_FN"]}, FP-like={d["n_FP"]}, '
                 f'trust={d["n_trust"]}.')
    auroc_h = findings.get('hepatic_discriminator_AUROC', {})
    auroc_d = findings.get('death_discriminator_AUROC', {})
    lines.append(f'- Discriminator (hepatic): FN AUROC={auroc_h.get(1, float("nan")):.3f}, '
                 f'FP AUROC={auroc_h.get(2, float("nan")):.3f}')
    lines.append(f'- Discriminator (death):   FN AUROC={auroc_d.get(1, float("nan")):.3f}, '
                 f'FP AUROC={auroc_d.get(2, float("nan")):.3f}')
    if 'val_buckets' in findings:
        vb = findings['val_buckets']
        lines.append(f'- Val buckets (hepatic): trust={vb["hepatic"]["trust"]}, '
                     f'FN={vb["hepatic"]["FN"]}, FP={vb["hepatic"]["FP"]}')
        lines.append(f'- Val buckets (death):   trust={vb["death"]["trust"]}, '
                     f'FN={vb["death"]["FN"]}, FP={vb["death"]["FP"]}')
    lines.append('')

    # Hepatic FN deep dive
    lines.append('## Hepatic FN-like patients (events the ensemble missed)')
    lines.append('')
    lines.append('| pid | time | rank | base_age | last_age | span | BMI₁ | ALT₁ | AST₁ | PLT₁ | GGT₁ | FibroScan₁ | FibroTest₁ |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for s in findings.get('hepatic_FN', []):
        def fmt(v): return f'{v:.1f}' if v is not None else '—'
        lines.append(f'| {s["patient_id"]} | {s["time_yrs"]:.1f} | '
                     f'{s["consensus_rank"]:.2f} | {fmt(s["baseline_age"])} | '
                     f'{fmt(s["last_followup_age"])} | {fmt(s["followup_span"])} | '
                     f'{fmt(s.get("BMI_first"))} | {fmt(s.get("alt_first"))} | '
                     f'{fmt(s.get("ast_first"))} | {fmt(s.get("plt_first"))} | '
                     f'{fmt(s.get("ggt_first"))} | '
                     f'{fmt(s.get("fibs_stiffness_med_BM_1_first"))} | '
                     f'{fmt(s.get("fibrotest_BM_2_first"))} |')
    lines.append('')

    # Means table
    means = findings.get('hepatic_means', {})
    if means:
        lines.append('## Hepatic feature means (population vs FN vs FP)')
        lines.append('')
        lines.append('| Feature | Population | FN-like | FP-like |')
        lines.append('|---|---|---|---|')
        for k in ['baseline_age', 'last_followup_age', 'followup_span',
                  'BMI_first', 'alt_first', 'ast_first', 'plt_first',
                  'ggt_first', 'fibs_stiffness_med_BM_1_first',
                  'fibrotest_BM_2_first', 'gluc_fast_first']:
            p = means.get('pop', {}).get(k, float('nan'))
            f = means.get('FN', {}).get(k, float('nan'))
            fp = means.get('FP', {}).get(k, float('nan'))
            lines.append(f'| {k} | {p:.2f} | {f:.2f} | {fp:.2f} |')
        lines.append('')

    # Death means
    means_d = findings.get('death_means', {})
    if means_d:
        lines.append('## Death feature means (population vs FN vs FP)')
        lines.append('')
        lines.append('| Feature | Population | FN-like | FP-like |')
        lines.append('|---|---|---|---|')
        for k in ['baseline_age', 'last_followup_age', 'followup_span',
                  'BMI_first', 'alt_first', 'ast_first', 'plt_first',
                  'ggt_first', 'fibs_stiffness_med_BM_1_first',
                  'fibrotest_BM_2_first']:
            p = means_d.get('pop', {}).get(k, float('nan'))
            f = means_d.get('FN', {}).get(k, float('nan'))
            fp = means_d.get('FP', {}).get(k, float('nan'))
            lines.append(f'| {k} | {p:.2f} | {f:.2f} | {fp:.2f} |')
        lines.append('')

    # Top FP details
    lines.append('## Hepatic FP-like (top 30 by consensus rank, time≥5y)')
    lines.append('')
    lines.append('| pid | time | rank | base_age | span | FibroScan₁ | FibroTest₁ |')
    lines.append('|---|---|---|---|---|---|---|')
    for s in findings.get('hepatic_FP_top30', [])[:30]:
        def fmt(v): return f'{v:.1f}' if v is not None else '—'
        lines.append(f'| {s["patient_id"]} | {s["time_yrs"]:.1f} | '
                     f'{s["consensus_rank"]:.2f} | {fmt(s["baseline_age"])} | '
                     f'{fmt(s["followup_span"])} | '
                     f'{fmt(s.get("fibs_stiffness_med_BM_1_first"))} | '
                     f'{fmt(s.get("fibrotest_BM_2_first"))} |')
    lines.append('')

    # Patterns + interpretation
    lines.append('## Patterns')
    lines.append('')
    means = findings.get('hepatic_means', {})
    pop = means.get('pop', {})
    fn = means.get('FN', {})
    fp = means.get('FP', {})
    if fn and pop:
        lines.append(f'- Hepatic FN: baseline age {fn.get("baseline_age", 0):.1f}y vs '
                     f'pop {pop.get("baseline_age", 0):.1f}y; followup span '
                     f'{fn.get("followup_span", 0):.1f}y vs pop '
                     f'{pop.get("followup_span", 0):.1f}y.')
    if fp and pop:
        lines.append(f'- Hepatic FP: baseline age {fp.get("baseline_age", 0):.1f}y vs '
                     f'pop {pop.get("baseline_age", 0):.1f}y; FibroScan baseline '
                     f'{fp.get("fibs_stiffness_med_BM_1_first", float("nan")):.1f} '
                     f'vs pop {pop.get("fibs_stiffness_med_BM_1_first", float("nan")):.1f}.')
    lines.append('')
    lines.append('## Generated submissions')
    lines.append('')
    lines.append('- `sub_master_deepfn_targeted.csv` — anchor + alpha·(P(FN)−P(FP)) per patient.')
    lines.append('- `sub_master_deepfn_specialist.csv` — anchor blended 50/50 with FN-specialist '
                 'sub-ensemble for FN-prone val patients.')
    lines.append('- `sub_master_deepfn_isolation.csv` — anchor blended toward 20%-trimmed-mean '
                 'val ensemble for FN/FP-prone or anomalous patients.')
    lines.append('- `sub_master_deepfn_targeted_death.csv` — death-only targeted adjust.')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    main()
