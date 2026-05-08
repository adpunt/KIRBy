"""Failure-pattern-aware candidate submissions.

Builds 5 NEW candidate submissions targeted at the failure modes documented
in failure_analysis.md / cross_reference.md / oof_diagnostics.md.

All candidates use the same base-model pool that produced the 0.9302 anchor
(`sub_master_meta_avg_aware_weighted.csv`), but apply distinct fusion
mechanisms designed to attack specific failure modes:

  fn_recovery     — UPweight base models that historically caught FN-like
                    patients, ONLY for val_test patients whose nearest
                    train neighbour is FN-like.
  fp_suppress     — DOWNweight aggressive base models via trimmed-mean,
                    ONLY for val_test patients near FP-like train cohort.
  xref_aware      — Trust the 0.9302 anchor for val_test patients with
                    NO failure-pattern match; for "uncertain" val_test
                    patients use a robust trimmed mean of all base models.
  hard_specialist — For val_test patients near the hardest-30 train
                    cohort, use the per-cohort best base model
                    (specialist_override). Else anchor.
  robust          — On the entire val_test, weight base models by
                    1/(miss_rate_on_FN + 0.1) to penalise models that
                    consistently miss FN-like patients (hep) /
                    by best_model_count for death.

Each candidate:
  - Uses INTEGER trustii_id 1..423 from val_test.csv (NOT GUIDs)
  - Outputs: tests/results/annitia/full/sub_master_failure_<name>.csv
  - Reports a training-side OOF C estimate built by replaying its fusion
    on the BASE-OOF stack and scoring against y_struct.

Usage:
    python tests/experiments/annitia/failure_candidates.py --variant all
    python tests/experiments/annitia/failure_candidates.py --variant fn_recovery
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia                  # noqa: E402
from experiments.annitia.views import build_views                      # noqa: E402
from experiments.annitia.survival import concordance                   # noqa: E402

FULL = REPO / 'tests/results/annitia/full'
ANALYSIS = REPO / 'tests/results/annitia/analysis'
RES_ROOT = REPO / 'tests/results/annitia'
VAL_CSV = '/Volumes/seagate/kirby/annitia/val_test.csv'
ANCHOR_NAME = 'sub_master_meta_avg_aware_weighted'

POOL_DIRS_NAMES = ('be', 'cd', 'hep_boost', 'branches', 'external', 'hparam', 'research', 'research_v2')


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def _rank01(x):
    finite = np.isfinite(x); out = np.full_like(x, np.nan, dtype=float)
    if finite.sum():
        out[finite] = rankdata(x[finite]) / max(finite.sum(), 1)
    return out


def _c(y, valid, scores):
    m = valid & np.isfinite(scores) & np.isfinite(y['time'])
    if m.sum() < 5:
        return float('nan')
    return float(concordance(y[m], scores[m]))


def _per_patient_disc(time, event, risk):
    finite = np.isfinite(time) & np.isfinite(risk)
    t = np.where(finite, time, np.nan); e = event.astype(bool) & finite
    ti = t[:, None]; tj = t[None, :]; ei = e[:, None]
    finite_pair = np.isfinite(ti) & np.isfinite(tj) & (ti < tj)
    comparable = finite_pair & ei
    ri = risk[:, None]; rj = risk[None, :]
    disc_pair = comparable & (ri < rj); tied = comparable & (ri == rj)
    pi = comparable.sum(axis=1); pj = comparable.sum(axis=0)
    di = disc_pair.sum(axis=1) + 0.5 * tied.sum(axis=1)
    dj = disc_pair.sum(axis=0) + 0.5 * tied.sum(axis=0)
    cnt = pi + pj; disc = di + dj
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(cnt > 0, disc / cnt, np.nan)


def _collect_pool(pool_dirs, endpoint):
    """List of dicts: {label, cv, oof, val} sorted by cv desc."""
    out = []
    for d in pool_dirs:
        cv_p = d / 'cv_summary.json'
        if not cv_p.exists(): continue
        sm = json.loads(cv_p.read_text())
        for r in sm.get('per_condition', {}).get(endpoint, []):
            cid = r.get('config_id', '')
            mc = r.get('mean_cindex', float('nan'))
            if not np.isfinite(mc): continue
            oof_p = d / f'oof_{endpoint}_{cid}.npy'
            val_p = d / f'val_{endpoint}_{cid}.npy'
            if not (oof_p.exists() and val_p.exists()): continue
            try:
                oof = np.load(oof_p); val = np.load(val_p)
            except Exception:
                continue
            if not np.any(np.isfinite(oof)) or not np.any(np.isfinite(val)): continue
            if np.std(oof[np.isfinite(oof)]) < 1e-10: continue
            out.append({'label': f'{d.name}__{cid}', 'cv': float(mc),
                        'oof': oof, 'val': val})
    out.sort(key=lambda r: -r['cv'])
    return out


def _features_matrix(d):
    views = build_views(d, view_names=['snapshot', 'clinical_pro'])
    X = np.concatenate([views[v][0] for v in ['snapshot', 'clinical_pro']], axis=1)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _val_trustii_ids(val):
    t = val.get('trustii_id')
    if t is None:
        # fall back: assume order matches the CSV (1..N)
        return np.arange(1, len(val['patient_ids']) + 1, dtype=int)
    return np.asarray(t, dtype=int)


# -----------------------------------------------------------------------
# Per-val-patient nearest-train classification
# -----------------------------------------------------------------------
def classify_val_patients(train, val, fn_idx, fp_idx, hard_idx, k=5):
    """For each val patient, find k nearest train neighbours and tag
    membership in fn / fp / hard sets. Returns dict per val_idx with
    counts, plus arrays aligned to val rows."""
    Xt = _features_matrix(train); Xv = _features_matrix(val)
    sc = StandardScaler().fit(Xt)
    Xt_s = sc.transform(Xt); Xv_s = sc.transform(Xv)
    nn = NearestNeighbors(n_neighbors=k).fit(Xt_s)
    dist, idx = nn.kneighbors(Xv_s)
    fn_set = set(int(x) for x in fn_idx)
    fp_set = set(int(x) for x in fp_idx)
    hard_set = set(int(x) for x in hard_idx)
    n_val = Xv.shape[0]
    n_fn = np.zeros(n_val, dtype=int)
    n_fp = np.zeros(n_val, dtype=int)
    n_hard = np.zeros(n_val, dtype=int)
    for i in range(n_val):
        ns = idx[i]
        n_fn[i] = sum(1 for j in ns if int(j) in fn_set)
        n_fp[i] = sum(1 for j in ns if int(j) in fp_set)
        n_hard[i] = sum(1 for j in ns if int(j) in hard_set)
    return {
        'n_fn': n_fn, 'n_fp': n_fp, 'n_hard': n_hard,
        'nearest_idx': idx[:, 0], 'nearest_dist': dist[:, 0],
        'all_idx': idx,
    }


# -----------------------------------------------------------------------
# Base-model FN/FP membership scoring
# -----------------------------------------------------------------------
def model_miss_rate_on_set(recs, target_idx, y, valid):
    """Per base-model: fraction of patients in `target_idx` that the model
    ranks below the median (i.e. fails to flag as high risk).
    For FN-like (events with low predicted rank) — we want models with
    LOW miss rate on this set."""
    target_idx = np.asarray(list(target_idx), dtype=int)
    target_idx = target_idx[(target_idx >= 0) & (target_idx < len(y))]
    rates = []
    for r in recs:
        ranks = _rank01(r['oof'])
        sub = ranks[target_idx]
        sub = sub[np.isfinite(sub)]
        if len(sub) == 0:
            rates.append(0.5); continue
        rates.append(float(np.mean(sub < 0.5)))
    return np.array(rates, dtype=float)


def model_correct_rate_on_fn(recs, fn_idx):
    """Per base-model: fraction of FN-like patients ranked TOP-quartile
    (correctly identified). High = caught the missed events."""
    fn_idx = np.asarray(list(fn_idx), dtype=int)
    rates = []
    for r in recs:
        ranks = _rank01(r['oof'])
        sub = ranks[fn_idx]; sub = sub[np.isfinite(sub)]
        if len(sub) == 0: rates.append(0.0); continue
        rates.append(float(np.mean(sub > 0.75)))
    return np.array(rates, dtype=float)


# -----------------------------------------------------------------------
# Candidates
# -----------------------------------------------------------------------
def cand_fn_recovery(recs, y, valid, val_class, n_train, n_val):
    """For val patients whose nearest-train-neighbour is FN-like, use a
    weighted blend that UPweights models that historically caught FN-like
    training patients. Other val patients: standard rank-mean of full pool.

    We'll use the same logic for OOF scoring: for training patients in the
    FN-like neighbourhood (themselves FN-like or close to one), apply the
    upweighted blend; rest get rank-mean.
    """
    n_models = len(recs)
    base_oof = np.stack([_rank01(r['oof']) for r in recs], axis=1)  # (n_train, n_models)
    base_val = np.stack([_rank01(r['val']) for r in recs], axis=1)
    rank_mean_oof = np.nanmean(base_oof, axis=1)
    rank_mean_val = np.nanmean(base_val, axis=1)

    # Per-model "catch FN-like" rate (higher = better at FN cases)
    # We compute outside since fn_idx differs per endpoint - passed via val_class
    fn_catch = val_class['_fn_catch_rates']  # (n_models,) precomputed
    # weights softmaxed over fn_catch (temp 5)
    s = fn_catch * 5.0
    w = np.exp(s - s.max()); w = w / w.sum()
    upweighted_oof = base_oof @ w
    upweighted_val = base_val @ w

    # Apply selectively: val patients with n_fn >= 1 OR n_hard >= 2 → upweighted
    sel_val = (val_class['n_fn'] >= 1)
    final_val = np.where(sel_val, upweighted_val, rank_mean_val)

    # OOF scoring side: apply to training patients near the FN-like cohort.
    # Use train-train nearest-neighbour to FN set: training patients within
    # 5-NN of any FN-like patient → "FN-like neighbourhood".
    fn_idx = val_class['_fn_idx']
    fn_neigh_train = val_class['_fn_neigh_train']
    sel_train = np.zeros(n_train, dtype=bool)
    if fn_neigh_train:
        sel_train[list(fn_neigh_train)] = True
    if fn_idx:
        sel_train[list(fn_idx)] = True
    final_oof = np.where(sel_train, upweighted_oof, rank_mean_oof)

    return final_oof, final_val, {
        'n_val_upweighted': int(sel_val.sum()),
        'n_train_upweighted': int(sel_train.sum()),
        'fn_catch_rates_top5': sorted(zip(fn_catch.tolist(),
                                          [r['label'] for r in recs]),
                                      reverse=True)[:5],
    }


def cand_fp_suppress(recs, y, valid, val_class, n_train, n_val):
    """For val patients near FP-like train cohort, use TRIMMED MEAN of all
    base models (drops top 20% / bottom 20%) to suppress aggressive false
    flags. Other val patients: rank-mean.
    """
    base_oof = np.stack([_rank01(r['oof']) for r in recs], axis=1)
    base_val = np.stack([_rank01(r['val']) for r in recs], axis=1)
    rank_mean_oof = np.nanmean(base_oof, axis=1)
    rank_mean_val = np.nanmean(base_val, axis=1)

    n_models = base_oof.shape[1]
    trim = max(1, n_models // 5)  # 20%

    def trimmed(stack):
        s = np.sort(stack, axis=1)
        return np.nanmean(s[:, trim:-trim], axis=1)

    trimmed_oof = trimmed(base_oof)
    trimmed_val = trimmed(base_val)

    sel_val = val_class['n_fp'] >= 1
    final_val = np.where(sel_val, trimmed_val, rank_mean_val)

    # Mirror on training: training patients near FP-like cohort
    fp_idx = val_class['_fp_idx']
    fp_neigh_train = val_class['_fp_neigh_train']
    sel_train = np.zeros(n_train, dtype=bool)
    if fp_neigh_train:
        sel_train[list(fp_neigh_train)] = True
    if fp_idx:
        sel_train[list(fp_idx)] = True
    final_oof = np.where(sel_train, trimmed_oof, rank_mean_oof)

    return final_oof, final_val, {
        'n_val_trimmed': int(sel_val.sum()),
        'n_train_trimmed': int(sel_train.sum()),
        'trim_per_side': trim,
    }


def cand_xref_aware(recs, y, valid, val_class, n_train, n_val,
                    anchor_oof_proxy, anchor_val):
    """Trust anchor for "trust" val patients (no failure-pattern match);
    for "uncertain" val patients (any of n_fn/n_fp/n_hard >= 1), use
    trimmed-mean blend (less overconfident).

    For OOF: training patients with no failure-set neighbour → anchor proxy
    (= rank-mean of pool, since the actual anchor is val-only); rest →
    trimmed mean.
    """
    base_oof = np.stack([_rank01(r['oof']) for r in recs], axis=1)
    base_val = np.stack([_rank01(r['val']) for r in recs], axis=1)
    rank_mean_oof = np.nanmean(base_oof, axis=1)

    n_models = base_oof.shape[1]
    trim = max(1, n_models // 5)
    s_oof = np.sort(base_oof, axis=1)
    trimmed_oof = np.nanmean(s_oof[:, trim:-trim], axis=1)
    s_val = np.sort(base_val, axis=1)
    trimmed_val = np.nanmean(s_val[:, trim:-trim], axis=1)

    # Anchor val: rank01-ed for blending fairness
    anchor_val_r = _rank01(anchor_val)

    uncertain_val = ((val_class['n_fn'] >= 1) |
                     (val_class['n_fp'] >= 1) |
                     (val_class['n_hard'] >= 1))
    final_val = np.where(uncertain_val, trimmed_val, anchor_val_r)

    # Training side: use rank-mean as anchor proxy
    fn_idx = set(int(x) for x in val_class['_fn_idx'])
    fp_idx = set(int(x) for x in val_class['_fp_idx'])
    hard_idx = set(int(x) for x in val_class['_hard_idx'])
    failure_union = fn_idx | fp_idx | hard_idx
    failure_neigh_train = (val_class['_fn_neigh_train'] |
                           val_class['_fp_neigh_train'] |
                           val_class['_hard_neigh_train'])
    sel_train = np.zeros(n_train, dtype=bool)
    sel_train[list(failure_union)] = True
    sel_train[list(failure_neigh_train)] = True
    final_oof = np.where(sel_train, trimmed_oof, rank_mean_oof)

    return final_oof, final_val, {
        'n_val_trust': int((~uncertain_val).sum()),
        'n_val_uncertain': int(uncertain_val.sum()),
        'n_train_trust': int((~sel_train).sum()),
    }


def cand_hard_specialist(recs, y, valid, val_class, n_train, n_val,
                         anchor_val):
    """For val patients near hardest-30 train cohort, use per-cohort
    specialist: the base model with HIGHEST mean rank01 on hardest-30 events.
    For others, anchor.
    """
    base_oof = np.stack([_rank01(r['oof']) for r in recs], axis=1)
    base_val = np.stack([_rank01(r['val']) for r in recs], axis=1)
    rank_mean_oof = np.nanmean(base_oof, axis=1)
    anchor_val_r = _rank01(anchor_val)

    hard_idx = np.asarray(list(val_class['_hard_idx']), dtype=int)
    # Per-model: fraction of hardest patients with EVENTS that the model
    # ranks top-quartile. (We want to "catch" hard events.)
    hard_event_idx = []
    for i in hard_idx:
        if 0 <= i < n_train and y['event'][i]:
            hard_event_idx.append(int(i))
    if not hard_event_idx:
        hard_event_idx = list(hard_idx)  # fallback
    hard_event_idx = np.asarray(hard_event_idx, dtype=int)

    catch_rates = []
    for j in range(base_oof.shape[1]):
        sub = base_oof[hard_event_idx, j]
        sub = sub[np.isfinite(sub)]
        catch_rates.append(float(np.mean(sub > 0.66)) if len(sub) else 0.0)
    best_j = int(np.argmax(catch_rates))
    spec_oof = base_oof[:, best_j]
    spec_val = base_val[:, best_j]

    sel_val = val_class['n_hard'] >= 1
    final_val = np.where(sel_val, spec_val, anchor_val_r)

    sel_train = np.zeros(n_train, dtype=bool)
    sel_train[list(val_class['_hard_idx'])] = True
    sel_train[list(val_class['_hard_neigh_train'])] = True
    final_oof = np.where(sel_train, spec_oof, rank_mean_oof)

    return final_oof, final_val, {
        'specialist_label': recs[best_j]['label'],
        'specialist_catch_rate': catch_rates[best_j],
        'n_val_specialist': int(sel_val.sum()),
        'top5_catch_rates': sorted(zip(catch_rates,
                                       [r['label'] for r in recs]),
                                   reverse=True)[:5],
    }


def cand_robust(recs, y, valid, val_class, n_train, n_val):
    """Whole-population fusion that DOWNweights base models with high miss
    rate on FN-like cohort. weight_i = 1 / (miss_rate_i + 0.1).
    """
    miss_rates = model_miss_rate_on_set(recs, val_class['_fn_idx'], y, valid)
    w = 1.0 / (miss_rates + 0.1)
    w = w / w.sum()

    base_oof = np.stack([_rank01(r['oof']) for r in recs], axis=1)
    base_val = np.stack([_rank01(r['val']) for r in recs], axis=1)
    final_oof = base_oof @ w
    final_val = base_val @ w

    return final_oof, final_val, {
        'miss_rates_top5_lowest': sorted(zip(miss_rates.tolist(),
                                             [r['label'] for r in recs]))[:5],
        'weights_top5': sorted(zip(w.tolist(),
                                   [r['label'] for r in recs]),
                               reverse=True)[:5],
    }


# -----------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------
CANDIDATES = ['fn_recovery', 'fp_suppress', 'xref_aware',
              'hard_specialist', 'robust']


def _build_train_neigh(train_X, target_idx, k=5):
    """Find training patients that are 5-NN to ANY target patient (i.e.
    the 'neighbourhood' of the target set). Returns set of train idx."""
    if len(target_idx) == 0:
        return set()
    target_idx = np.asarray(list(target_idx), dtype=int)
    nn = NearestNeighbors(n_neighbors=k).fit(train_X)
    _, idx = nn.kneighbors(train_X[target_idx])
    return set(int(x) for x in idx.flatten())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', default='all',
                    choices=['all'] + CANDIDATES)
    ap.add_argument('--validation-csv', default=VAL_CSV)
    args = ap.parse_args()

    print(f'[load] train + val ({args.validation_csv})')
    train = load_annitia()
    val = load_annitia(data_dir=str(Path(args.validation_csv).parent),
                       filename=Path(args.validation_csv).name,
                       has_outcomes=False)
    n_train = len(train['patient_ids'])
    n_val = len(val['patient_ids'])
    val_tids = _val_trustii_ids(val)
    print(f'  n_train={n_train}, n_val={n_val}, '
          f'trustii_id range [{val_tids.min()},{val_tids.max()}]')

    # Anchor (val_test predictions)
    anchor = pd.read_csv(FULL / f'{ANCHOR_NAME}.csv')
    if not (anchor['trustii_id'].values == val_tids).all():
        print('  WARN: anchor trustii_id ordering differs from val loader; '
              'reindexing anchor by trustii_id')
        anchor = anchor.set_index('trustii_id').loc[val_tids].reset_index()

    # Failure analysis indices
    fa = json.loads((FULL / 'failure_analysis.json').read_text())

    # Pool
    pool_dirs = [RES_ROOT / p for p in POOL_DIRS_NAMES if (RES_ROOT / p).exists()]
    print(f'[pool] {len(pool_dirs)} dirs')

    # Train features for "neighbourhood of failure set" building
    Xt = _features_matrix(train); sc = StandardScaler().fit(Xt)
    Xt_s = sc.transform(Xt)

    summary = {}
    variants_to_run = CANDIDATES if args.variant == 'all' else [args.variant]

    for endpoint, col in (('hepatic', 'risk_hepatic_event'),
                          ('death', 'risk_death')):
        recs = _collect_pool(pool_dirs, endpoint)
        y = train[endpoint]['y_struct']
        valid = train[endpoint]['valid_mask']
        n_models = len(recs)
        print(f'\n=== {endpoint} ===  n_models={n_models}, '
              f'n_events={int(y["event"].sum())}')
        if n_models == 0:
            print('  no base models — skipping endpoint')
            continue

        f = fa[endpoint]
        fn_idx = set(int(x) for x in f['fn_like_idx'])
        fp_idx = set(int(x) for x in f['fp_like_idx'])
        hard_idx = set(int(x) for x in f['hardest_idx'])

        # Per-val classification (5-NN train)
        val_class = classify_val_patients(train, val, fn_idx, fp_idx,
                                          hard_idx, k=5)
        # Train neighbourhoods around each failure set (for OOF scoring)
        val_class['_fn_idx'] = fn_idx
        val_class['_fp_idx'] = fp_idx
        val_class['_hard_idx'] = hard_idx
        val_class['_fn_neigh_train'] = _build_train_neigh(Xt_s, fn_idx, k=5)
        val_class['_fp_neigh_train'] = _build_train_neigh(Xt_s, fp_idx, k=5)
        val_class['_hard_neigh_train'] = _build_train_neigh(Xt_s, hard_idx, k=5)
        val_class['_fn_catch_rates'] = model_correct_rate_on_fn(recs, fn_idx)

        print(f'  val patients: n_fn={int((val_class["n_fn"]>=1).sum())}, '
              f'n_fp={int((val_class["n_fp"]>=1).sum())}, '
              f'n_hard={int((val_class["n_hard"]>=1).sum())}')

        anchor_val = anchor[col].values

        for variant in variants_to_run:
            if variant == 'fn_recovery':
                oof, valp, info = cand_fn_recovery(recs, y, valid,
                                                   val_class, n_train, n_val)
            elif variant == 'fp_suppress':
                oof, valp, info = cand_fp_suppress(recs, y, valid,
                                                   val_class, n_train, n_val)
            elif variant == 'xref_aware':
                oof, valp, info = cand_xref_aware(recs, y, valid, val_class,
                                                  n_train, n_val,
                                                  None, anchor_val)
            elif variant == 'hard_specialist':
                oof, valp, info = cand_hard_specialist(recs, y, valid,
                                                       val_class,
                                                       n_train, n_val,
                                                       anchor_val)
            elif variant == 'robust':
                oof, valp, info = cand_robust(recs, y, valid, val_class,
                                              n_train, n_val)
            else:
                continue

            c = _c(y, valid, oof)
            summary.setdefault(variant, {})[endpoint] = {
                'oof_c': c, 'info': info,
            }
            print(f'  {variant:18s} OOF C = {c:.4f}')

            # Save into per-variant CSV (write hep first run, append death on second)
            csv_p = FULL / f'sub_master_failure_{variant}.csv'
            if csv_p.exists():
                prev = pd.read_csv(csv_p)
                if not (prev['trustii_id'].values == val_tids).all():
                    prev = prev.set_index('trustii_id').loc[val_tids].reset_index()
                prev[col] = valp
                df_out = prev[['trustii_id', 'risk_hepatic_event', 'risk_death']]
            else:
                df_out = pd.DataFrame({
                    'trustii_id': val_tids,
                    'risk_hepatic_event': np.nan,
                    'risk_death': np.nan,
                })
                df_out[col] = valp
            df_out.to_csv(csv_p, index=False)

    # Compute weighted OOF C and save summary JSON
    print('\n=== summary (training OOF) ===')
    for v, eps in summary.items():
        h = eps.get('hepatic', {}).get('oof_c', float('nan'))
        d = eps.get('death', {}).get('oof_c', float('nan'))
        if np.isfinite(h) and np.isfinite(d):
            w = 0.7 * h + 0.3 * d
        else:
            w = float('nan')
        eps['weighted_oof_c'] = w
        print(f'  {v:18s}  hep={h:.4f}  death={d:.4f}  '
              f'weighted={w:.4f}')

    (FULL / 'failure_candidates_summary.json').write_text(
        json.dumps(summary, indent=2, default=str))
    print(f'\nwrote {FULL}/failure_candidates_summary.json')


if __name__ == '__main__':
    main()
