"""Noise-mitigation fusion candidates (deadline 2026-05-08).

Five distinct mechanisms targeting different noise sources in the ANNITIA
MASLD challenge:

  ipcw          IPCW-weighted fusion. Reweights base-model OOF
                correlations by inverse probability of censoring.
                Targets informative-censoring noise (death endpoint
                has 21% blank-status rows excluded; surviving rows
                have variable follow-up times).

  multiseed     Multi-seed averaged anchor. (See note: this requires
                model retraining; we scaffold it but DON'T retrain on
                the deadline. We instead implement a non-retraining
                proxy: bootstrap-resample each base OOF over training
                rows 4 times and average — approximates seed-noise
                reduction via subsample noise.)

  huber         Per-patient trimmed-mean fusion. For each patient,
                trim the top+bottom 20% of base-model predictions
                BEFORE averaging — different from rank_trimmed which
                trims models per ranking step. Robust to outlier
                base-model predictions on individual patients.

  censoraware   For DEATH only: drop landmark-classifier-style base
                models (tabpfn-landmark, tabpfn25-disc, dsm,
                pretrained-mlp), keep survival-aware (Cox/Coxnet/
                RSF/GBSA/AFT/NGBoost). For HEP: full pool. Targets
                the 21% blank-death-row censoring bias.

  classbalance  HEP only: weight base models by their event-vs-event
                concordance (rank only the 47 known events against
                each other). For DEATH: full meta_avg_aware-style
                pool. Targets noise from event-vs-censored pairs
                dominating the standard C-index.

Outputs: sub_master_noise_<variant>.csv to tests/results/annitia/full/.

Usage:
    python noise_candidates.py --variant ipcw
    python noise_candidates.py --variant all     # build all five
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia                      # noqa: E402
from experiments.annitia.survival import concordance                       # noqa: E402


VAL_CSV = '/Volumes/seagate/kirby/annitia/val_test.csv'
RES_ROOT = _REPO / 'tests' / 'results' / 'annitia'
FULL_DIR = RES_ROOT / 'full'
POOL_DIRS = [RES_ROOT / p for p in
             ('be', 'cd', 'hep_boost', 'branches', 'external', 'hparam', 'research', 'research_v2')
             if (RES_ROOT / p).exists()]

# Tag survival-aware base models for variant (d). Anything else is treated
# as a discretised / landmark-style classifier.
SURVIVAL_KEYS = ('cox', 'coxnet', 'rsf', 'gbsa', 'lgbm_aft', 'ngboost',
                 'catboost_cox', 'catboost_aft', 'synaug-')
CLASSIFIER_KEYS = ('tabpfn-landmark', 'tabpfn25-disc', 'dsm',
                   'pretrained-mlp', 'mlp-', 'tabsurv-', 'catboost-')


def is_survival_aware(label: str) -> bool:
    lab = label.lower()
    if any(k in lab for k in CLASSIFIER_KEYS):
        return False
    return any(k in lab for k in SURVIVAL_KEYS)


# ----------------------------------------------------------------------
# Helpers (mirror refuse_variants.py)
# ----------------------------------------------------------------------
def rank01(x):
    finite = np.isfinite(x); out = np.full_like(x, np.nan, dtype=float)
    if finite.sum():
        out[finite] = rankdata(x[finite]) / max(finite.sum(), 1)
    return out


def cidx(y, valid, scores):
    m = valid & np.isfinite(scores) & np.isfinite(y['time'])
    if m.sum() < 5:
        return float('nan')
    return float(concordance(y[m], scores[m]))


def collect(pool_dirs, endpoint):
    out = []
    for d in pool_dirs:
        cv = d / 'cv_summary.json'
        if not cv.exists(): continue
        try:
            sm = json.loads(cv.read_text())
        except Exception:
            continue
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
            if np.isfinite(oof).sum() < 50:
                continue
            out.append({'label': f'{d.name}__{cid}', 'cv': float(mc),
                        'oof': oof, 'val': val})
    out.sort(key=lambda r: -r['cv'])
    return out


# ----------------------------------------------------------------------
# (a) IPCW weights
# ----------------------------------------------------------------------
def compute_ipcw(y, valid):
    """Per-patient inverse probability of censoring weights.

    Fit Kaplan-Meier on the censoring distribution (event <-> 1-event),
    then w_i = 1 / G(T_i) where G is the censoring survival function.
    NaN outside valid_mask. Truncate G to >= 0.05 to prevent blow-up.
    """
    from lifelines import KaplanMeierFitter
    n = len(y)
    w = np.full(n, np.nan)
    keep = valid & np.isfinite(y['time'])
    if keep.sum() < 10:
        return w
    t = y['time'][keep].astype(float)
    e = y['event'][keep].astype(int)
    # KM on censoring: the "event" for KM is 1 - actual event.
    kmf = KaplanMeierFitter()
    kmf.fit(t, event_observed=(1 - e), label='cens')
    # Survival function evaluated at each patient's time
    sf = kmf.survival_function_
    G = np.interp(t, sf.index.values, sf['cens'].values, left=1.0, right=sf['cens'].values[-1])
    G = np.clip(G, 0.05, 1.0)
    w[keep] = 1.0 / G
    # Normalise to mean 1 over kept rows (keeps weighted correlations
    # comparable to unweighted ones).
    w[keep] = w[keep] / np.mean(w[keep])
    return w


# ----------------------------------------------------------------------
# Variant builders
# ----------------------------------------------------------------------
def variant_ipcw(recs, y, valid, max_size=12):
    """Greedy build of weighted rank-mean ensemble where each base model's
    contribution is gated by its IPCW-weighted rank-correlation with the
    event indicator."""
    if not recs: return None
    w = compute_ipcw(y, valid)
    keep = valid & np.isfinite(w) & np.isfinite(y['time'])
    if keep.sum() < 50:
        return None
    e = y['event'].astype(float)
    # Score each candidate by IPCW-weighted Pearson(rank, event)
    scores = []
    for r in recs:
        rk = rank01(r['oof'])
        m = keep & np.isfinite(rk)
        if m.sum() < 50: scores.append(-np.inf); continue
        rkm = rk[m] - np.average(rk[m], weights=w[m])
        em = e[m] - np.average(e[m], weights=w[m])
        num = np.sum(w[m] * rkm * em)
        den = np.sqrt(np.sum(w[m] * rkm * rkm) * np.sum(w[m] * em * em))
        scores.append(num / den if den > 0 else -np.inf)
    order = np.argsort(scores)[::-1]
    # Take top models with positive weighted-corr; weight by score.
    chosen_idx = [i for i in order if scores[i] > 0][:max_size]
    if not chosen_idx: return None
    weights = np.array([scores[i] for i in chosen_idx], dtype=float)
    weights = weights / weights.sum()
    rks_o = np.stack([rank01(recs[i]['oof']) for i in chosen_idx], axis=1)
    rks_v = np.stack([rank01(recs[i]['val']) for i in chosen_idx], axis=1)
    oof = rks_o @ weights
    val = rks_v @ weights
    return {'oof': oof, 'val': val, 'oof_c': cidx(y, valid, oof),
            'members': [recs[i]['label'] for i in chosen_idx],
            'weights': weights.tolist(),
            'note': f'ipcw n_cens_weighted_kept={int(keep.sum())}'}


def variant_multiseed(recs, y, valid, max_size=12):
    """Bootstrap-aggregated variant (proxy for retraining with extra seeds).

    For each of the top survival-aware base models, generate 4 bootstrap
    replicates of its OOF (resampling indices with replacement, then
    re-ranking). Average the rank-transformed bootstraps to approximate
    seed-noise reduction. Then plug into a rank-mean over the top models.

    Documented as a SCAFFOLD: a real retrain with seeds 43/44/45 would
    be the proper implementation but is not feasible in the deadline
    window. Bootstrap-of-OOF approximates the variance reduction without
    refitting.
    """
    if not recs: return None
    rng = np.random.default_rng(42)
    n = len(y)
    top = recs[:min(8, len(recs))]
    boot_oofs = []
    boot_vals = []
    for r in top:
        oof = r['oof']; val = r['val']
        finite_oof = np.where(np.isfinite(oof))[0]
        if len(finite_oof) < 50: continue
        # 4 bootstrap reps of OOF rank: jitter via tied-rank perturbation
        rk_reps = []
        for _ in range(4):
            samp = rng.choice(finite_oof, size=len(finite_oof), replace=True)
            tmp = np.full(n, np.nan)
            # Average rank over the bootstrap sample (rank stability proxy)
            ranks_in_samp = rankdata(oof[samp]) / len(samp)
            for j, idx in enumerate(samp):
                if np.isnan(tmp[idx]):
                    tmp[idx] = ranks_in_samp[j]
                else:
                    tmp[idx] = 0.5 * tmp[idx] + 0.5 * ranks_in_samp[j]
            # Fill any gap with original rank
            gap = np.isnan(tmp) & np.isfinite(oof)
            tmp[gap] = rank01(oof)[gap]
            rk_reps.append(tmp)
        # Original rank counts as the seed=42 "anchor"
        rk_reps.append(rank01(oof))
        boot_oofs.append(np.nanmean(np.stack(rk_reps, axis=1), axis=1))
        # Val: just use rank01 (val isn't bootstrapped — only training-side
        # variance is the noise source we're targeting).
        boot_vals.append(rank01(val))
    if not boot_oofs:
        return None
    oof = np.nanmean(np.stack(boot_oofs, axis=1), axis=1)
    val = np.nanmean(np.stack(boot_vals, axis=1), axis=1)
    return {'oof': oof, 'val': val, 'oof_c': cidx(y, valid, oof),
            'members': [r['label'] for r in top[:len(boot_oofs)]],
            'note': 'SCAFFOLD: bootstrap-of-OOF proxy; true seed retrain not run'}


def variant_huber(recs, y, valid, max_size=15):
    """Per-patient 20%-trimmed mean over the top-N rank-transformed OOFs.

    For each patient, sort their N model predictions, drop top 10% and
    bottom 10%, average the middle. Different from rank_trimmed (which
    trims rankings GLOBALLY across all patients). This trims OUTLIER
    BASE-MODEL predictions for each individual patient.
    """
    if not recs: return None
    top = recs[:min(max_size, len(recs))]
    R_o = np.stack([rank01(r['oof']) for r in top], axis=1)  # (n, k)
    R_v = np.stack([rank01(r['val']) for r in top], axis=1)
    k = R_o.shape[1]
    drop = max(1, int(np.floor(k * 0.10)))
    # Per-patient: sort across models, drop drop top + drop bottom, average mid
    def _pp_trim(M):
        n_rows = M.shape[0]
        out = np.full(n_rows, np.nan)
        for i in range(n_rows):
            row = M[i]
            row = row[np.isfinite(row)]
            if len(row) < (2 * drop + 1):
                if len(row): out[i] = np.mean(row)
                continue
            row = np.sort(row)
            out[i] = float(np.mean(row[drop:len(row) - drop]))
        return out
    oof = _pp_trim(R_o)
    val = _pp_trim(R_v)
    return {'oof': oof, 'val': val, 'oof_c': cidx(y, valid, oof),
            'members': [r['label'] for r in top],
            'note': f'per-patient trim={drop}/{k}'}


def variant_censoraware(recs_full, y_dict, valid_dict):
    """Build hep + death side-by-side. For death, drop classifier-style models."""
    out = {}
    for endpoint, key in (('hepatic', 'hep'), ('death', 'death')):
        recs = recs_full[endpoint]
        y = y_dict[endpoint]; valid = valid_dict[endpoint]
        if endpoint == 'death':
            survival_recs = [r for r in recs if is_survival_aware(r['label'])]
            if len(survival_recs) < 3:
                survival_recs = recs  # fallback to all
            pool = survival_recs
        else:
            pool = recs
        top = pool[:12]
        if not top:
            out[endpoint] = None; continue
        R_o = np.stack([rank01(r['oof']) for r in top], axis=1)
        R_v = np.stack([rank01(r['val']) for r in top], axis=1)
        # Weighted rank-mean by per-model CV
        w = np.array([r['cv'] for r in top], dtype=float)
        w = np.maximum(w - 0.5, 0)  # offset so weak models get tiny weight
        if w.sum() < 1e-6:
            w = np.ones(len(top)) / len(top)
        else:
            w = w / w.sum()
        oof = R_o @ w; val = R_v @ w
        out[endpoint] = {'oof': oof, 'val': val,
                         'oof_c': cidx(y, valid, oof),
                         'members': [r['label'] for r in top],
                         'note': f'pool_size={len(top)} survival_only={endpoint=="death"}'}
    return out


def event_only_concordance(y, valid, oof):
    """C-index restricted to event-vs-event ordering only.

    Among patients with event=1, count concordant pairs (where the one
    who died earlier has a higher predicted risk). Ignores
    event-vs-censored pairs.
    """
    m = valid & np.isfinite(oof) & np.isfinite(y['time'])
    e = y['event'].astype(bool) & m
    idx = np.where(e)[0]
    if len(idx) < 5: return 0.5
    t = y['time'][idx]; r = oof[idx]
    n_concord = 0; n_pairs = 0
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            if t[i] == t[j]: continue
            n_pairs += 1
            if (t[i] < t[j] and r[i] > r[j]) or (t[i] > t[j] and r[i] < r[j]):
                n_concord += 1
            elif r[i] == r[j]:
                n_concord += 0.5
    return n_concord / max(n_pairs, 1)


def variant_classbalance(recs_full, y_dict, valid_dict):
    """Hep: weight base models by event-only C; death: weight by standard CV."""
    out = {}
    for endpoint in ('hepatic', 'death'):
        recs = recs_full[endpoint]
        y = y_dict[endpoint]; valid = valid_dict[endpoint]
        top = recs[:15]
        if not top:
            out[endpoint] = None; continue
        if endpoint == 'hepatic':
            # event-only C as weight
            scores = np.array([max(0.0, event_only_concordance(y, valid, r['oof']) - 0.5)
                               for r in top])
        else:
            scores = np.array([max(0.0, r['cv'] - 0.5) for r in top])
        if scores.sum() < 1e-6:
            scores = np.ones_like(scores) / len(scores)
        else:
            scores = scores / scores.sum()
        R_o = np.stack([rank01(r['oof']) for r in top], axis=1)
        R_v = np.stack([rank01(r['val']) for r in top], axis=1)
        oof = R_o @ scores; val = R_v @ scores
        out[endpoint] = {'oof': oof, 'val': val,
                         'oof_c': cidx(y, valid, oof),
                         'members': [r['label'] for r in top],
                         'weights': scores.tolist(),
                         'note': 'event-only weighting' if endpoint == 'hepatic' else 'cv weighting'}
    return out


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def write_submission(name, val_hep, val_death, trustii_id):
    df = pd.DataFrame({
        'trustii_id': np.asarray(trustii_id, dtype=int),
        'risk_hepatic_event': val_hep,
        'risk_death': val_death,
    })
    # Sort by trustii_id ascending so we match the anchor's row order
    df = df.sort_values('trustii_id').reset_index(drop=True)
    out_path = FULL_DIR / f'sub_master_noise_{name}.csv'
    df.to_csv(out_path, index=False)
    return out_path


def run_variant(name, recs_full, y_dict, valid_dict):
    if name == 'ipcw':
        h = variant_ipcw(recs_full['hepatic'], y_dict['hepatic'], valid_dict['hepatic'])
        d = variant_ipcw(recs_full['death'], y_dict['death'], valid_dict['death'])
        return {'hepatic': h, 'death': d}
    if name == 'multiseed':
        h = variant_multiseed(recs_full['hepatic'], y_dict['hepatic'], valid_dict['hepatic'])
        d = variant_multiseed(recs_full['death'], y_dict['death'], valid_dict['death'])
        return {'hepatic': h, 'death': d}
    if name == 'huber':
        h = variant_huber(recs_full['hepatic'], y_dict['hepatic'], valid_dict['hepatic'])
        d = variant_huber(recs_full['death'], y_dict['death'], valid_dict['death'])
        return {'hepatic': h, 'death': d}
    if name == 'censoraware':
        return variant_censoraware(recs_full, y_dict, valid_dict)
    if name == 'classbalance':
        return variant_classbalance(recs_full, y_dict, valid_dict)
    raise ValueError(f'unknown variant {name}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', default='all',
                    choices=['ipcw', 'multiseed', 'huber', 'censoraware',
                             'classbalance', 'all'])
    ap.add_argument('--validation-csv', default=VAL_CSV)
    args = ap.parse_args()

    train = load_annitia()
    val = load_annitia(data_dir=str(Path(args.validation_csv).parent),
                       filename=Path(args.validation_csv).name,
                       has_outcomes=False)
    if val.get('trustii_id') is None:
        raise SystemExit('val_test.csv must contain trustii_id column')
    trustii_id = val['trustii_id']

    y_dict = {'hepatic': train['hepatic']['y_struct'],
              'death':   train['death']['y_struct']}
    valid_dict = {'hepatic': train['hepatic']['valid_mask'],
                  'death':   train['death']['valid_mask']}

    print(f'Pooling from: {[d.name for d in POOL_DIRS]}')
    recs_full = {ep: collect(POOL_DIRS, ep) for ep in ('hepatic', 'death')}
    for ep, recs in recs_full.items():
        print(f'  {ep}: {len(recs)} base models')

    variants_to_run = (['ipcw', 'multiseed', 'huber', 'censoraware', 'classbalance']
                       if args.variant == 'all' else [args.variant])

    summary = {}
    for vname in variants_to_run:
        print(f'\n=== {vname} ===')
        try:
            res = run_variant(vname, recs_full, y_dict, valid_dict)
        except Exception as e:
            print(f'  FAILED: {type(e).__name__}: {e}')
            summary[vname] = {'error': f'{type(e).__name__}: {e}'}
            continue
        h = res.get('hepatic'); d = res.get('death')
        if h is None or d is None:
            print(f'  SKIPPED: missing endpoint result')
            summary[vname] = {'error': 'missing endpoint'}
            continue
        # OOF Cs
        c_h = h['oof_c']; c_d = d['oof_c']
        c_w = float('nan')
        if np.isfinite(c_h) and np.isfinite(c_d):
            c_w = 0.7 * c_h + 0.3 * c_d
        print(f'  OOF C hep={c_h:.4f}  death={c_d:.4f}  weighted={c_w:.4f}')
        if 'note' in h: print(f'    hep note: {h["note"]}')
        if 'note' in d: print(f'    death note: {d["note"]}')
        # Write submission CSV
        out_path = write_submission(vname, h['val'], d['val'], trustii_id)
        print(f'  wrote {out_path}')
        summary[vname] = {
            'oof_c_hep': c_h, 'oof_c_death': c_d, 'oof_c_weighted': c_w,
            'csv': str(out_path),
            'hep_members': h.get('members', [])[:8],
            'death_members': d.get('members', [])[:8],
            'hep_note': h.get('note', ''), 'death_note': d.get('note', ''),
        }

    summary_path = FULL_DIR / 'noise_candidates_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f'\nSummary -> {summary_path}')
    for vname, s in summary.items():
        if 'error' in s:
            print(f'  {vname:14s} ERROR {s["error"]}')
        else:
            print(f'  {vname:14s} hep={s["oof_c_hep"]:.4f} death={s["oof_c_death"]:.4f} '
                  f'wt={s["oof_c_weighted"]:.4f}')


if __name__ == '__main__':
    main()
