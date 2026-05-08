"""Five new high-EV submission candidates covering ideas not yet implemented.

Variants (all distinct mechanisms):

1. inference_dropout
   Inference-time random-subset dropout (A.6). Average M=200 random
   k-subset predictions over the base pool. Distinct from existing
   sub_master_bootstrap_dropout.csv which is a TRAINING-time perturbation.

2. per_endpoint
   Per-endpoint best (E.31 / per-target tuning). HEP risk and DEATH risk
   are taken from DIFFERENT recipes — the recipe that maximises each
   endpoint's HONEST OOF C separately. Adds a HIGH-trust hep variant
   composed only from the highest-trust HEP base models from
   oof_diagnostics.md, since under 0.7/0.3 the hep weight is bigger.

3. bma
   Bayesian model averaging (A.7). For each base model compute posterior
   weight ∝ exp(-NLL_oof) on training, then aggregate base models by
   posterior-weighted rank average. Endpoint-separate posteriors.

4. cohort_split
   Cohort-specific specialists (C.22). Diabetic (T2DM>=1) vs
   non-diabetic (T2DM==0). For each cohort pick the best base model
   (lowest discordance on cohort training subset) per endpoint;
   compose val patients by their cohort assignment.

5. binwise_corrected
   Bin-wise discordance correction (D.27). Bin training patients by
   ensemble mean rank into 10 bins. For each bin, find the base model
   with lowest discordance. Re-rank val_test patients within their bin
   using the lowest-discordance base model for that bin.

Usage:
    python new_high_ev_candidates.py --variant inference_dropout
    python new_high_ev_candidates.py --variant all  # default
"""
from __future__ import annotations

import argparse
import json
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia                         # noqa: E402
from experiments.annitia.survival import concordance                          # noqa: E402

_VAL_PATH = '/Volumes/seagate/kirby/annitia/val_test.csv'
_RES = _REPO / 'tests' / 'results' / 'annitia'
_FULL = _RES / 'full'
_POOL_DIRS = [_RES / p for p in ('be', 'cd', 'hep_boost', 'branches', 'external', 'hparam', 'research', 'research_v2')
              if (_RES / p).exists()]


# ---------------------------------------------------------------------- helpers
def _rank01(x):
    finite = np.isfinite(x); out = np.full_like(x, np.nan, dtype=float)
    if finite.sum():
        out[finite] = rankdata(x[finite]) / max(finite.sum(), 1)
    return out


def _c(y_struct, valid, scores):
    m = valid & np.isfinite(scores) & np.isfinite(y_struct['time'])
    if m.sum() < 5:
        return float('nan')
    return float(concordance(y_struct[m], scores[m]))


def _per_patient_disc(time, event, risk):
    finite = np.isfinite(time) & np.isfinite(risk)
    t = np.where(finite, time, np.nan); e = event.astype(bool) & finite
    ti = t[:, None]; tj = t[None, :]; ei = e[:, None]
    finite_pair = np.isfinite(ti) & np.isfinite(tj) & (ti < tj)
    comparable = finite_pair & ei
    ri = risk[:, None]; rj = risk[None, :]
    discordant = comparable & (ri < rj); tied = comparable & (ri == rj)
    pi = comparable.sum(axis=1); pj = comparable.sum(axis=0)
    di = discordant.sum(axis=1) + 0.5 * tied.sum(axis=1)
    dj = discordant.sum(axis=0) + 0.5 * tied.sum(axis=0)
    cnt = pi + pj; disc = di + dj
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(cnt > 0, disc / cnt, np.nan)


def _collect(endpoint):
    """Return list of dicts with label/cv/oof/val for every base model."""
    out = []
    for d in _POOL_DIRS:
        cv = d / 'cv_summary.json'
        if not cv.exists(): continue
        sm = json.loads(cv.read_text())
        for r in sm.get('per_condition', {}).get(endpoint, []):
            cid = r.get('config_id', '')
            mc = r.get('mean_cindex', float('nan'))
            if not np.isfinite(mc): continue
            oof_p = d / f'oof_{endpoint}_{cid}.npy'
            val_p = d / f'val_{endpoint}_{cid}.npy'
            if not (oof_p.exists() and val_p.exists()): continue
            try:
                o = np.load(oof_p); v = np.load(val_p)
            except Exception:
                continue
            out.append({'label': f'{d.name}__{cid}', 'cv': float(mc),
                        'oof': o, 'val': v})
    out.sort(key=lambda r: -r['cv'])
    return out


def _save_sub(name, val_tids, hep, death):
    p = _FULL / f'sub_master_new_{name}.csv'
    n = len(val_tids)
    if hep.shape[0] != n or death.shape[0] != n:
        raise ValueError(f'{name}: hep {hep.shape} or death {death.shape} != {n}')
    if not np.isfinite(hep).all() or not np.isfinite(death).all():
        # impute non-finite with median rank 0.5
        hep = np.where(np.isfinite(hep), hep, 0.5)
        death = np.where(np.isfinite(death), death, 0.5)
    df = pd.DataFrame({'trustii_id': val_tids.astype(int),
                       'risk_hepatic_event': hep,
                       'risk_death': death})
    df.to_csv(p, index=False)
    return p


def _run_pre_check(p):
    res = subprocess.run([sys.executable,
                          str(Path(__file__).parent / 'pre_submit_check.py'),
                          str(p)],
                         capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
    return res.returncode


# ---------------------------------------------------------------------- variants
def variant_inference_dropout(recs_hep, recs_death, y_hep, valid_hep,
                              y_death, valid_death, val_tids, M=200, k_frac=0.6,
                              seed=42):
    """Random-subset dropout at INFERENCE time. M=200 bootstrap subsets,
    each size = k_frac * N base models, average rank-mean over subsets."""
    rng = np.random.default_rng(seed)

    def _go(recs, y, valid):
        N = len(recs)
        k = max(2, int(k_frac * N))
        oof_R = np.stack([_rank01(r['oof']) for r in recs], axis=1)  # (n_train, N)
        val_R = np.stack([_rank01(r['val']) for r in recs], axis=1)  # (n_val, N)
        oof_acc = np.zeros(oof_R.shape[0]); val_acc = np.zeros(val_R.shape[0])
        for _ in range(M):
            idx = rng.choice(N, size=k, replace=False)
            sub_o = oof_R[:, idx]; sub_v = val_R[:, idx]
            with np.errstate(invalid='ignore'):
                oof_acc += np.nanmean(sub_o, axis=1)
                val_acc += np.nanmean(sub_v, axis=1)
        oof = oof_acc / M; val = val_acc / M
        return _c(y, valid, oof), oof, val

    c_hep, _, hep_val = _go(recs_hep, y_hep, valid_hep)
    c_death, _, death_val = _go(recs_death, y_death, valid_death)
    p = _save_sub('inference_dropout', val_tids,
                  _rank01(hep_val), _rank01(death_val))
    return {'name': 'inference_dropout', 'path': p,
            'oof_c_hep': c_hep, 'oof_c_death': c_death,
            'weighted': 0.7 * c_hep + 0.3 * c_death,
            'config': {'M': M, 'k_frac': k_frac, 'N_hep': len(recs_hep),
                       'N_death': len(recs_death)}}


def variant_per_endpoint(recs_hep, recs_death, y_hep, valid_hep,
                         y_death, valid_death, val_tids):
    """Pick best HEP recipe and best DEATH recipe separately (different
    fusion logic per endpoint). HEP recipe further restricted to a
    high-trust subset of base models from oof_diagnostics."""
    # HIGH-TRUST hep models (from oof_diagnostics.md): take top-9 by pooled C
    HIGH_TRUST_HEP_KEYS = [
        'synaug-none-rsf-500-n0',
        'synaug-gmm-rsf-500-n100',
        'synaug-tabebm-rsf-500-n100',
        'tsfm-rsf', 'tsfm-gbsa',
        'catboost-D2H',
        'tabpfn25-disc-K10',
        'tabpfn-landmark',
        'rsf-1000',  # hep_boost
        'bagged-rsf-B200',  # hep_boost
        'weighted-rsf',  # any of the rsf-perm variants
        'gbsa-500',
        'catboost-aft', 'catboost-cox',
    ]

    def _is_high_trust_hep(label):
        return any(k in label for k in HIGH_TRUST_HEP_KEYS)

    high_trust = [r for r in recs_hep if _is_high_trust_hep(r['label'])]
    if len(high_trust) < 5:
        high_trust = recs_hep[:15]

    def _best_recipe(recs, y, valid, name_pref):
        """Try several recipes (rank_mean greedy, simple top-K mean,
        top-K median) and return the one with highest OOF C."""
        results = {}
        # Recipe A: greedy rank_mean over up to 10 models
        chosen = [recs[0]]; rks = [_rank01(recs[0]['oof'])]; vrs = [_rank01(recs[0]['val'])]
        best_c = _c(y, valid, np.mean(np.stack(rks, axis=1), axis=1))
        pool = list(recs[1:])
        while len(chosen) < 10 and pool:
            best_score, best_i = -np.inf, None
            for i, cand in enumerate(pool):
                tmp = np.mean(np.stack(rks + [_rank01(cand['oof'])], axis=1), axis=1)
                cc = _c(y, valid, tmp)
                if cc > best_score: best_score, best_i = cc, i
            if best_i is None or best_score - best_c < 1e-4: break
            chosen.append(pool[best_i]); rks.append(_rank01(pool[best_i]['oof']))
            vrs.append(_rank01(pool[best_i]['val'])); best_c = best_score; pool.pop(best_i)
        results['greedy_mean'] = (best_c,
                                  np.mean(np.stack(vrs, axis=1), axis=1),
                                  [c['label'] for c in chosen])
        # Recipe B: top-5 simple mean
        top5 = recs[:5]
        v5 = np.mean(np.stack([_rank01(r['val']) for r in top5], axis=1), axis=1)
        o5 = np.mean(np.stack([_rank01(r['oof']) for r in top5], axis=1), axis=1)
        results['top5_mean'] = (_c(y, valid, o5), v5, [r['label'] for r in top5])
        # Recipe C: top-7 median
        top7 = recs[:7]
        v7 = np.median(np.stack([_rank01(r['val']) for r in top7], axis=1), axis=1)
        o7 = np.median(np.stack([_rank01(r['oof']) for r in top7], axis=1), axis=1)
        results['top7_median'] = (_c(y, valid, o7), v7, [r['label'] for r in top7])
        # pick best
        best = max(results.items(), key=lambda x: x[1][0] if np.isfinite(x[1][0]) else -np.inf)
        return best[0], best[1][0], best[1][1], best[1][2]

    hep_recipe, hep_c, hep_val, hep_members = _best_recipe(high_trust, y_hep, valid_hep, 'hep')
    death_recipe, death_c, death_val, death_members = _best_recipe(recs_death, y_death, valid_death, 'death')

    p = _save_sub('per_endpoint', val_tids, _rank01(hep_val), _rank01(death_val))
    return {'name': 'per_endpoint', 'path': p,
            'oof_c_hep': hep_c, 'oof_c_death': death_c,
            'weighted': 0.7 * hep_c + 0.3 * death_c,
            'config': {'hep_recipe': hep_recipe, 'death_recipe': death_recipe,
                       'hep_pool_size': len(high_trust),
                       'hep_members': hep_members,
                       'death_members': death_members}}


def variant_bma(recs_hep, recs_death, y_hep, valid_hep,
                y_death, valid_death, val_tids):
    """Bayesian model averaging. Posterior weight ∝ exp(-NLL_oof_pseudo)
    where NLL_pseudo is approximated by the surrogate -log(C-index) on
    OOF (higher C → higher posterior). We use a tempered softmax with
    a temperature picked so the top model gets ~50% weight."""

    def _go(recs, y, valid):
        N = len(recs)
        c_per = []
        for r in recs:
            c_per.append(_c(y, valid, r['oof']))
        c_per = np.array(c_per)
        c_per = np.where(np.isfinite(c_per), c_per, 0.5)
        # Negative log-loss-like: -log C is larger when C is small;
        # use score = log(C / (1-C)) as "logit posterior surrogate"
        # then BMA weight = softmax(beta * score). beta=10 makes it sharp.
        score = np.log(np.clip(c_per, 1e-3, 1 - 1e-3) / np.clip(1 - c_per, 1e-3, 1 - 1e-3))
        beta = 8.0  # sharpness — higher = more concentration on top
        w = np.exp(beta * (score - score.max()))
        w = w / w.sum()
        oof_R = np.stack([_rank01(r['oof']) for r in recs], axis=1)
        val_R = np.stack([_rank01(r['val']) for r in recs], axis=1)
        with np.errstate(invalid='ignore'):
            oof = np.nansum(oof_R * w[None, :], axis=1)
            val = np.nansum(val_R * w[None, :], axis=1)
        return _c(y, valid, oof), val, w, c_per

    c_hep, hep_val, w_hep, c_hep_per = _go(recs_hep, y_hep, valid_hep)
    c_death, death_val, w_death, c_death_per = _go(recs_death, y_death, valid_death)

    p = _save_sub('bma', val_tids, _rank01(hep_val), _rank01(death_val))
    top_hep_idx = np.argsort(-w_hep)[:5]
    top_death_idx = np.argsort(-w_death)[:5]
    return {'name': 'bma', 'path': p,
            'oof_c_hep': c_hep, 'oof_c_death': c_death,
            'weighted': 0.7 * c_hep + 0.3 * c_death,
            'config': {'beta': 8.0,
                       'top5_hep': [(recs_hep[i]['label'], float(w_hep[i]))
                                    for i in top_hep_idx],
                       'top5_death': [(recs_death[i]['label'], float(w_death[i]))
                                      for i in top_death_idx]}}


def variant_cohort_split(recs_hep, recs_death, y_hep, valid_hep,
                         y_death, valid_death, val_tids,
                         t2dm_train, t2dm_val):
    """Diabetes-cohort specialists. Two cohorts: diabetic (T2DM>=1) vs
    non-diabetic (T2DM==0). NaN goes to non-diabetic by default."""
    diab_train = (t2dm_train >= 1) & np.isfinite(t2dm_train)
    diab_val = (t2dm_val >= 1) & np.isfinite(t2dm_val)
    print(f'  cohort sizes — train diab/non: {diab_train.sum()}/{(~diab_train).sum()}'
          f'  val diab/non: {diab_val.sum()}/{(~diab_val).sum()}')

    def _go(recs, y, valid):
        # find best model per cohort by lowest mean discordance on cohort's
        # valid training rows
        def _best_in_mask(mask):
            best_score, best_r = -np.inf, recs[0]
            for r in recs[:30]:  # consider top-30 by CV
                d = _per_patient_disc(y['time'], y['event'], r['oof'])
                sub = d[mask]; sub = sub[np.isfinite(sub)]
                if len(sub) < 5: continue
                s = 1.0 - np.mean(sub)
                if s > best_score:
                    best_score, best_r = s, r
            return best_r, best_score

        m_diab = valid & diab_train
        m_non = valid & (~diab_train)
        b_diab, _ = _best_in_mask(m_diab)
        b_non, _ = _best_in_mask(m_non)

        oof = np.full(len(y), np.nan)
        oof[diab_train] = _rank01(b_diab['oof'])[diab_train]
        oof[~diab_train] = _rank01(b_non['oof'])[~diab_train]
        val = np.full(len(t2dm_val), np.nan)
        val[diab_val] = _rank01(b_diab['val'])[diab_val]
        val[~diab_val] = _rank01(b_non['val'])[~diab_val]
        return _c(y, valid, oof), val, b_diab['label'], b_non['label']

    c_hep, hep_val, hep_diab_pick, hep_non_pick = _go(recs_hep, y_hep, valid_hep)
    c_death, death_val, dt_diab_pick, dt_non_pick = _go(recs_death, y_death, valid_death)

    p = _save_sub('cohort_split', val_tids, _rank01(hep_val), _rank01(death_val))
    return {'name': 'cohort_split', 'path': p,
            'oof_c_hep': c_hep, 'oof_c_death': c_death,
            'weighted': 0.7 * c_hep + 0.3 * c_death,
            'config': {'hep_diab': hep_diab_pick, 'hep_non': hep_non_pick,
                       'death_diab': dt_diab_pick, 'death_non': dt_non_pick}}


def variant_binwise_corrected(recs_hep, recs_death, y_hep, valid_hep,
                              y_death, valid_death, val_tids, n_bins=10):
    """Bin-wise discordance correction. Bin training patients by the
    rank-mean ensemble (over top-15 base models) into n_bins. For each
    bin, find the base model with lowest discordance ON that bin's
    members. Re-rank val_test patients within their bin using the
    bin's lowest-discordance model."""

    def _go(recs, y, valid):
        # ensemble over top-15
        top = recs[:15]
        oof_ens = np.mean(np.stack([_rank01(r['oof']) for r in top], axis=1), axis=1)
        val_ens = np.mean(np.stack([_rank01(r['val']) for r in top], axis=1), axis=1)
        keep = valid & np.isfinite(oof_ens)
        edges = np.nanquantile(oof_ens[keep], np.linspace(0, 1, n_bins + 1))
        edges[0] = -np.inf; edges[-1] = np.inf
        # Per bin: pick best base model on bin members
        chosen = []
        for b in range(n_bins):
            bm = keep & (oof_ens >= edges[b]) & (oof_ens < edges[b + 1])
            if bm.sum() < 5:
                chosen.append(top[0]); continue
            best_score, best_r = -np.inf, top[0]
            for r in recs[:30]:
                d = _per_patient_disc(y['time'], y['event'], r['oof'])
                sub = d[bm]; sub = sub[np.isfinite(sub)]
                if len(sub) < 3: continue
                s = 1.0 - np.mean(sub)
                if s > best_score:
                    best_score, best_r = s, r
            chosen.append(best_r)
        # build re-ranked train OOF
        oof_new = np.full(len(y), np.nan)
        for b in range(n_bins):
            bm = (oof_ens >= edges[b]) & (oof_ens < edges[b + 1])
            if not bm.any(): continue
            r = chosen[b]
            # Rank within bin only — preserves bin ordering (which the
            # ensemble already gets right) but corrects WITHIN-bin order
            # using the bin's specialist
            sub_oof = r['oof'][bm]
            sub_rank = _rank01(sub_oof)  # 0-1 within bin
            # Embed: shift into [b/n_bins, (b+1)/n_bins]
            oof_new[bm] = (b + sub_rank) / n_bins
        # Same for val using val_ens to assign bins, then bin specialist
        val_new = np.full(len(val_ens), np.nan)
        for b in range(n_bins):
            vm = (val_ens >= edges[b]) & (val_ens < edges[b + 1])
            if not vm.any(): continue
            r = chosen[b]
            sub_val = r['val'][vm]
            sub_rank = _rank01(sub_val)
            val_new[vm] = (b + sub_rank) / n_bins
        return _c(y, valid, oof_new), val_new, [c['label'] for c in chosen]

    c_hep, hep_val, hep_picks = _go(recs_hep, y_hep, valid_hep)
    c_death, death_val, death_picks = _go(recs_death, y_death, valid_death)

    p = _save_sub('binwise_corrected', val_tids, _rank01(hep_val), _rank01(death_val))
    return {'name': 'binwise_corrected', 'path': p,
            'oof_c_hep': c_hep, 'oof_c_death': c_death,
            'weighted': 0.7 * c_hep + 0.3 * c_death,
            'config': {'n_bins': n_bins,
                       'hep_picks': hep_picks, 'death_picks': death_picks}}


# ---------------------------------------------------------------------- main
ALL_VARIANTS = ['inference_dropout', 'per_endpoint', 'bma',
                'cohort_split', 'binwise_corrected']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', default='all', choices=ALL_VARIANTS + ['all'])
    args = ap.parse_args()

    print('Loading training + validation data...')
    train = load_annitia()
    val = load_annitia(data_dir=str(Path(_VAL_PATH).parent),
                       filename=Path(_VAL_PATH).name, has_outcomes=False)
    val_tids = val['trustii_id']
    if val_tids is None:
        val_tids = np.arange(1, len(val['patient_ids']) + 1)
    val_tids = np.asarray(val_tids).astype(int)
    print(f'  train n={len(train["patient_ids"])}  val n={len(val["patient_ids"])}')

    # Cohort flags
    t2dm_train = train['static']['T2DM']
    val_df = pd.read_csv(_VAL_PATH)
    t2dm_val = pd.to_numeric(val_df['T2DM'], errors='coerce').to_numpy()

    print('Collecting base-model OOFs...')
    recs_hep = _collect('hepatic')
    recs_death = _collect('death')
    print(f'  hepatic pool: {len(recs_hep)} models')
    print(f'  death pool:   {len(recs_death)} models')

    y_hep = train['hepatic']['y_struct']; valid_hep = train['hepatic']['valid_mask']
    y_death = train['death']['y_struct']; valid_death = train['death']['valid_mask']

    selected = ALL_VARIANTS if args.variant == 'all' else [args.variant]
    summary = {}

    for vname in selected:
        print(f'\n=== {vname} ===')
        if vname == 'inference_dropout':
            r = variant_inference_dropout(recs_hep, recs_death, y_hep, valid_hep,
                                          y_death, valid_death, val_tids)
        elif vname == 'per_endpoint':
            r = variant_per_endpoint(recs_hep, recs_death, y_hep, valid_hep,
                                     y_death, valid_death, val_tids)
        elif vname == 'bma':
            r = variant_bma(recs_hep, recs_death, y_hep, valid_hep,
                            y_death, valid_death, val_tids)
        elif vname == 'cohort_split':
            r = variant_cohort_split(recs_hep, recs_death, y_hep, valid_hep,
                                     y_death, valid_death, val_tids,
                                     t2dm_train, t2dm_val)
        elif vname == 'binwise_corrected':
            r = variant_binwise_corrected(recs_hep, recs_death, y_hep, valid_hep,
                                          y_death, valid_death, val_tids)
        else:
            raise ValueError(vname)
        print(f'  hep OOF C   = {r["oof_c_hep"]:.4f}')
        print(f'  death OOF C = {r["oof_c_death"]:.4f}')
        print(f'  weighted    = {r["weighted"]:.4f}  (0.7*hep + 0.3*death)')
        print(f'  written to  : {r["path"]}')
        rc = _run_pre_check(r['path'])
        r['pre_check_rc'] = rc
        summary[vname] = {k: v for k, v in r.items() if k != 'path'} | {
            'path': str(r['path'])}

    out = _FULL / 'new_high_ev_candidates_summary.json'
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f'\nSummary written to {out}')

    # Final ranking
    print('\nRanked by weighted (0.7*hep + 0.3*death):')
    for k, v in sorted(summary.items(), key=lambda x: -x[1]['weighted']):
        print(f'  {k:25s}  weighted={v["weighted"]:.4f}  '
              f'hep={v["oof_c_hep"]:.4f}  death={v["oof_c_death"]:.4f}  '
              f'pre={v["pre_check_rc"]}')


if __name__ == '__main__':
    main()
