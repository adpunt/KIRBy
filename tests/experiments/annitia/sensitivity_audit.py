"""ANNITIA sensitivity audit.

For the top 15 variants (with known LB scores) construct training-side proxy
OOFs via variant_proxy_oof, then stress-test each via 5 perturbations:

A. Label permutation: shuffle event flag + duration for 5/10/20% of patients.
   Track C-degradation. Slow-degrading variants are robust.
B. Censoring perturbation: flip 20% of training events to censored.
C. Per-pair influence: rank concordance pairs by individual contribution,
   surface the most "load-bearing" pairs per variant.
D. Time-axis perturbation: add Gaussian sigma=0.5y noise to durations.
E. Adversarial patient deletion: identify the 5 patients whose removal MOST
   drops C; check if these patients are shared across variants.

Outputs CSVs + AUDIT_SENSITIVITY.md. Pure local diagnostic — no submissions
written. Reads only existing OOFs + labels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia                 # noqa: E402
from experiments.annitia.survival import concordance                  # noqa: E402
from experiments.annitia.oof_to_leaderboard import (                  # noqa: E402
    base_oof_dict, load_consensus_oof, variant_proxy_oof,
)

BRANCHES = REPO / 'tests/results/annitia/branches'
FULL = REPO / 'tests/results/annitia/full'

RNG_SEED = 17
SEEDS = [17, 23, 41, 67, 89]  # 5 seeds for stochastic perturbations


def _c_index(y_struct, risk):
    """Wrap concordance with finite filtering."""
    m = np.isfinite(risk) & np.isfinite(y_struct['time'])
    if m.sum() < 5:
        return float('nan')
    return float(concordance(y_struct[m], risk[m]))


def _weighted_lb(c_h, c_d):
    """Trustii LB formula."""
    return 0.7 * c_h + 0.3 * c_d


# ---- A. Label permutation ----------------------------------------------------

def stress_label_permute(stems, proxies, hep_y, hep_v, dth_y, dth_v):
    rows = []
    for frac in (0.0, 0.05, 0.10, 0.20):
        for stem in stems:
            risk_h = proxies[stem]['hep']
            risk_d = proxies[stem]['death']
            c_h_runs, c_d_runs = [], []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                # hep
                h = hep_y.copy()
                idx_h = np.where(hep_v)[0]
                k = int(round(frac * len(idx_h)))
                if k > 0:
                    sel = rng.choice(idx_h, size=k, replace=False)
                    perm = rng.permutation(sel)
                    h['event'][sel] = h['event'][perm]
                    h['time'][sel] = h['time'][perm]
                c_h_runs.append(_c_index(h, risk_h))
                # death
                d = dth_y.copy()
                idx_d = np.where(dth_v)[0]
                k = int(round(frac * len(idx_d)))
                if k > 0:
                    sel = rng.choice(idx_d, size=k, replace=False)
                    perm = rng.permutation(sel)
                    d['event'][sel] = d['event'][perm]
                    d['time'][sel] = d['time'][perm]
                c_d_runs.append(_c_index(d, risk_d))
            c_h = float(np.nanmean(c_h_runs))
            c_d = float(np.nanmean(c_d_runs))
            rows.append({
                'variant': stem, 'perm_frac': frac,
                'C_hep': c_h, 'C_death': c_d,
                'weighted_LB': _weighted_lb(c_h, c_d),
                'C_hep_std': float(np.nanstd(c_h_runs)),
                'C_death_std': float(np.nanstd(c_d_runs)),
            })
    return pd.DataFrame(rows)


# ---- B. Censoring perturbation ----------------------------------------------

def stress_censor_flip(stems, proxies, hep_y, hep_v, dth_y, dth_v, frac=0.20):
    rows = []
    for stem in stems:
        risk_h = proxies[stem]['hep']
        risk_d = proxies[stem]['death']
        c_h_base = _c_index(hep_y, risk_h)
        c_d_base = _c_index(dth_y, risk_d)
        c_h_runs, c_d_runs = [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            h = hep_y.copy()
            idx_h = np.where(hep_v)[0]
            sel_h = rng.choice(idx_h, size=int(round(frac * len(idx_h))),
                               replace=False)
            h['event'][sel_h] = False
            c_h_runs.append(_c_index(h, risk_h))
            d = dth_y.copy()
            idx_d = np.where(dth_v)[0]
            sel_d = rng.choice(idx_d, size=int(round(frac * len(idx_d))),
                               replace=False)
            d['event'][sel_d] = False
            c_d_runs.append(_c_index(d, risk_d))
        c_h = float(np.nanmean(c_h_runs))
        c_d = float(np.nanmean(c_d_runs))
        rows.append({
            'variant': stem,
            'C_hep_base': c_h_base, 'C_death_base': c_d_base,
            'LB_base': _weighted_lb(c_h_base, c_d_base),
            'C_hep_perturb': c_h, 'C_death_perturb': c_d,
            'LB_perturb': _weighted_lb(c_h, c_d),
            'LB_drop': _weighted_lb(c_h_base, c_d_base) - _weighted_lb(c_h, c_d),
        })
    return pd.DataFrame(rows)


# ---- C. Per-pair influence --------------------------------------------------

def _comparable_pairs(y, risk, max_pairs=None, rng=None):
    """Return list of (i, j, contrib) triples for concordant-pair scoring.

    A pair is comparable if event_i and time_i < time_j (i fails first while
    j was still at risk). Vectorised: for each event-i we broadcast against
    all comparable j.
    """
    finite = np.isfinite(risk) & np.isfinite(y['time'])
    idx = np.where(finite)[0]
    if len(idx) < 2:
        return []
    e = y['event'][idx]
    t = y['time'][idx]
    r = risk[idx]
    pairs = []
    event_pos = np.where(e)[0]
    for i_local in event_pos:
        ti = t[i_local]; ri = r[i_local]
        comp = t > ti  # j risk-set
        if not comp.any():
            continue
        rj = r[comp]
        contrib = np.where(ri > rj, 1.0, np.where(ri == rj, 0.5, 0.0))
        j_local = np.where(comp)[0]
        i_glob = idx[i_local]
        j_glob = idx[j_local]
        for k in range(len(j_glob)):
            pairs.append((int(i_glob), int(j_glob[k]), float(contrib[k])))
    if max_pairs is not None and len(pairs) > max_pairs and rng is not None:
        sel = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[k] for k in sel]
    return pairs


def stress_pair_influence(stems, proxies, hep_y, hep_v, dth_y, dth_v):
    """For each variant compute the top 5 most influential pairs.

    Influence = drop in C_index when that pair is removed. A pair contributes
    `contrib / total_pairs` to the C, so removing it changes C by approximately
    `(contrib - mean_contrib) / (total_pairs - 1)`. We sum hep+death per
    pair-key (i, j) when patient appears in both.
    """
    rows = []
    cache = {}
    def _key(arr):
        a = arr[np.isfinite(arr)]
        return (len(a), float(a.sum()), float(a.std()))
    for stem in stems:
        rec = {'variant': stem}
        for ep, y, risk in (
            ('hep', hep_y, proxies[stem]['hep']),
            ('death', dth_y, proxies[stem]['death']),
        ):
            ck = (ep, _key(risk))
            if ck in cache:
                pairs = cache[ck]
            else:
                pairs = _comparable_pairs(y, risk)
                cache[ck] = pairs
            if not pairs:
                rec[f'{ep}_total_pairs'] = 0
                rec[f'{ep}_top5_load_bearing'] = ''
                rec[f'{ep}_concentration_top1pct'] = float('nan')
                continue
            n_pairs = len(pairs)
            base_sum = sum(p[2] for p in pairs)
            mean_contrib = base_sum / n_pairs
            # leave-one-out drop: removing pair k changes C from
            # (S/N) to (S - c_k)/(N-1) -> drop = S/N - (S-c_k)/(N-1)
            # = (S(N-1) - N(S-c_k))/(N(N-1)) = (Nc_k - S)/(N(N-1))
            # so larger |drop| <=> larger |c_k - S/N| (sign by c_k > mean).
            # We want pairs whose presence MOST INFLATES C => contrib > mean.
            inflate = [(p[0], p[1], p[2] - mean_contrib) for p in pairs]
            inflate.sort(key=lambda x: -x[2])
            top5 = inflate[:5]
            top5_str = ';'.join(f'{int(a)}->{int(b)}({d:+.2f})' for a, b, d in top5)
            # concentration: fraction of total C contribution sitting in top 1%
            top1pct = max(int(round(0.01 * n_pairs)), 1)
            sorted_contribs = sorted([p[2] for p in pairs], reverse=True)
            conc = sum(sorted_contribs[:top1pct]) / max(base_sum, 1e-9)
            rec[f'{ep}_total_pairs'] = n_pairs
            rec[f'{ep}_top5_load_bearing'] = top5_str
            rec[f'{ep}_concentration_top1pct'] = conc
        rows.append(rec)
    return pd.DataFrame(rows)


# ---- D. Time-axis perturbation ---------------------------------------------

def stress_duration_noise(stems, proxies, hep_y, hep_v, dth_y, dth_v, sigma=0.5):
    rows = []
    for stem in stems:
        risk_h = proxies[stem]['hep']
        risk_d = proxies[stem]['death']
        c_h_base = _c_index(hep_y, risk_h)
        c_d_base = _c_index(dth_y, risk_d)
        c_h_runs, c_d_runs = [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            h = hep_y.copy()
            mask_h = np.isfinite(h['time'])
            h['time'][mask_h] = h['time'][mask_h] + rng.normal(0, sigma, mask_h.sum())
            h['time'][mask_h] = np.maximum(h['time'][mask_h], 1e-3)
            c_h_runs.append(_c_index(h, risk_h))
            d = dth_y.copy()
            mask_d = np.isfinite(d['time'])
            d['time'][mask_d] = d['time'][mask_d] + rng.normal(0, sigma, mask_d.sum())
            d['time'][mask_d] = np.maximum(d['time'][mask_d], 1e-3)
            c_d_runs.append(_c_index(d, risk_d))
        c_h = float(np.nanmean(c_h_runs))
        c_d = float(np.nanmean(c_d_runs))
        rows.append({
            'variant': stem, 'sigma_years': sigma,
            'C_hep_base': c_h_base, 'C_death_base': c_d_base,
            'LB_base': _weighted_lb(c_h_base, c_d_base),
            'C_hep_noise': c_h, 'C_death_noise': c_d,
            'LB_noise': _weighted_lb(c_h, c_d),
            'LB_drop': _weighted_lb(c_h_base, c_d_base) - _weighted_lb(c_h, c_d),
        })
    return pd.DataFrame(rows)


# ---- E. Adversarial patient deletion ---------------------------------------

def _per_patient_loo(y, risk):
    """Return per-patient LOO drop in C-index. Drop = C_full - C_without_i."""
    finite = np.isfinite(risk) & np.isfinite(y['time'])
    idx = np.where(finite)[0]
    if len(idx) < 5:
        return np.full(len(y), np.nan)
    c_full = float(concordance(y[idx], risk[idx]))
    drops = np.full(len(y), np.nan)
    for i in idx:
        keep = idx[idx != i]
        try:
            c_loo = float(concordance(y[keep], risk[keep]))
            drops[i] = c_full - c_loo
        except Exception:
            drops[i] = np.nan
    return drops


def stress_adversarial_delete(stems, proxies, hep_y, dth_y, patient_ids, k=5):
    rows = []
    top_h_sets = {}
    top_d_sets = {}
    # Cache LOO per unique proxy fingerprint (most variants collapse to one of
    # ~3 recipes, so we'd otherwise do duplicate work).
    cache_h, cache_d = {}, {}
    def _key(arr):
        a = arr[np.isfinite(arr)]
        return (len(a), float(a.sum()), float(a.std()))
    for stem in stems:
        risk_h = proxies[stem]['hep']
        risk_d = proxies[stem]['death']
        kh, kd = _key(risk_h), _key(risk_d)
        if kh not in cache_h:
            cache_h[kh] = _per_patient_loo(hep_y, risk_h)
        if kd not in cache_d:
            cache_d[kd] = _per_patient_loo(dth_y, risk_d)
        drops_h = cache_h[kh]
        drops_d = cache_d[kd]
        # top-k = patients whose removal MOST DROPS C (positive drops mean
        # the patient was helping the variant; their removal hurts it).
        ord_h = np.argsort(-np.where(np.isfinite(drops_h), drops_h, -np.inf))
        ord_d = np.argsort(-np.where(np.isfinite(drops_d), drops_d, -np.inf))
        top_h = ord_h[:k]
        top_d = ord_d[:k]
        top_h_sets[stem] = top_h
        top_d_sets[stem] = top_d
        # cumulative drop if we deleted top-5 simultaneously (approx via
        # re-evaluation rather than summing LOO drops)
        keep_h = np.setdiff1d(np.arange(len(hep_y)), top_h)
        keep_d = np.setdiff1d(np.arange(len(dth_y)), top_d)
        c_h_base = _c_index(hep_y, risk_h)
        c_d_base = _c_index(dth_y, risk_d)
        c_h_after = _c_index(hep_y[keep_h], risk_h[keep_h])
        c_d_after = _c_index(dth_y[keep_d], risk_d[keep_d])
        rows.append({
            'variant': stem,
            'top5_hep_indices': ';'.join(str(int(i)) for i in top_h),
            'top5_hep_pids': ';'.join(str(patient_ids[i]) for i in top_h),
            'top5_death_indices': ';'.join(str(int(i)) for i in top_d),
            'top5_death_pids': ';'.join(str(patient_ids[i]) for i in top_d),
            'C_hep_base': c_h_base, 'C_hep_top5_removed': c_h_after,
            'C_death_base': c_d_base, 'C_death_top5_removed': c_d_after,
            'LB_drop': _weighted_lb(c_h_base, c_d_base) - _weighted_lb(c_h_after, c_d_after),
        })
    df = pd.DataFrame(rows)
    return df, top_h_sets, top_d_sets


# ---- main -------------------------------------------------------------------

def main():
    print('Loading dataset + base OOFs...')
    train = load_annitia()
    hep = train['hepatic']; dth = train['death']
    hep_y = hep['y_struct'].copy(); hep_v = hep['valid_mask']
    dth_y = dth['y_struct'].copy(); dth_v = dth['valid_mask']
    patient_ids = train['patient_ids']

    consensus = load_consensus_oof()
    base_oofs = base_oof_dict()

    history = json.loads((REPO / 'tests/experiments/annitia/leaderboard_history.json').read_text())
    subs = sorted(history['submissions'], key=lambda r: -r['score'])
    # exclude broken round3
    subs = [s for s in subs if s['stem'] != 'submission']
    top15 = subs[:15]
    stems = [s['stem'] for s in top15]
    lb_map = {s['stem']: s['score'] for s in top15}

    proxies = {}
    for s in stems:
        proxies[s] = {
            'hep': variant_proxy_oof(s, base_oofs['hepatic'], consensus['hepatic']),
            'death': variant_proxy_oof(s, base_oofs['death'], consensus['death']),
        }

    # Quick sanity: distinct proxy fingerprints (variants resolving to identical
    # arrays will share results — that itself is a finding).
    fp = {}
    for s in stems:
        fp[s] = (
            float(np.nansum(proxies[s]['hep'] * 1.0)),
            float(np.nansum(proxies[s]['death'] * 1.0)),
        )
    distinct = len(set(fp.values()))
    print(f'  {len(stems)} variants -> {distinct} distinct proxy fingerprints')

    print('A. label permutation...')
    df_a = stress_label_permute(stems, proxies, hep_y, hep_v, dth_y, dth_v)
    df_a.to_csv(FULL / 'sensitivity_label_perm.csv', index=False)

    print('B. censoring perturbation...')
    df_b = stress_censor_flip(stems, proxies, hep_y, hep_v, dth_y, dth_v)
    df_b.to_csv(FULL / 'sensitivity_censoring_perturb.csv', index=False)

    print('C. per-pair influence...')
    df_c = stress_pair_influence(stems, proxies, hep_y, hep_v, dth_y, dth_v)
    df_c.to_csv(FULL / 'sensitivity_pair_influence.csv', index=False)

    print('D. duration noise...')
    df_d = stress_duration_noise(stems, proxies, hep_y, hep_v, dth_y, dth_v)
    df_d.to_csv(FULL / 'sensitivity_duration_noise.csv', index=False)

    print('E. adversarial deletion...')
    df_e, top_h_sets, top_d_sets = stress_adversarial_delete(
        stems, proxies, hep_y, dth_y, patient_ids)
    df_e.to_csv(FULL / 'sensitivity_adversarial_delete.csv', index=False)

    # ---- aggregate robustness score per variant ---------------------------
    base_lb = {}
    for s in stems:
        base_lb[s] = _weighted_lb(_c_index(hep_y, proxies[s]['hep']),
                                  _c_index(dth_y, proxies[s]['death']))
    rob_rows = []
    for s in stems:
        a_lb = df_a[(df_a.variant == s) & (df_a.perm_frac == 0.20)]['weighted_LB'].iloc[0]
        b_lb = df_b[df_b.variant == s]['LB_perturb'].iloc[0]
        d_lb = df_d[df_d.variant == s]['LB_noise'].iloc[0]
        e_drop = df_e[df_e.variant == s]['LB_drop'].iloc[0]
        avg_drop = float(np.nanmean([
            base_lb[s] - a_lb,
            base_lb[s] - b_lb,
            base_lb[s] - d_lb,
            e_drop,
        ]))
        rob_rows.append({
            'variant': s,
            'public_LB': lb_map[s],
            'base_OOF_LB': base_lb[s],
            'drop_perm20': base_lb[s] - a_lb,
            'drop_censor20': base_lb[s] - b_lb,
            'drop_dur_noise': base_lb[s] - d_lb,
            'drop_top5_delete': e_drop,
            'avg_drop': avg_drop,
        })
    df_rob = pd.DataFrame(rob_rows).sort_values('avg_drop')

    # ---- patient-overlap analysis -----------------------------------------
    all_h = Counter()
    all_d = Counter()
    for s in stems:
        for i in top_h_sets[s]:
            all_h[int(i)] += 1
        for i in top_d_sets[s]:
            all_d[int(i)] += 1
    shared_h = [(idx, cnt) for idx, cnt in all_h.most_common(10)]
    shared_d = [(idx, cnt) for idx, cnt in all_d.most_common(10)]

    # ---- markdown report --------------------------------------------------
    md = []
    md.append('# ANNITIA sensitivity audit')
    md.append('')
    md.append(f'Top 15 variants × 5 stress tests. Generated 2026-05-08.')
    md.append('')
    md.append(f'**Distinct proxy fingerprints:** {distinct} of 15 variants. '
              'Most "meta_avg*" recipes collapse to the same rank-mean proxy '
              'in `variant_proxy_oof`, so their sensitivity numbers are '
              'identical by construction — interpret with care.')
    md.append('')
    md.append('## Robustness ranking (ascending avg LB drop = most robust first)')
    md.append('')
    md.append('| Rank | Variant | Public LB | Base OOF LB | Δperm20 | Δcensor20 | Δdur_noise | Δtop5_del | avg drop |')
    md.append('|---|---|---|---|---|---|---|---|---|')
    for i, r in enumerate(df_rob.itertuples(index=False), 1):
        md.append(f'| {i} | `{r.variant}` | {r.public_LB:.4f} | {r.base_OOF_LB:.4f} | '
                  f'{r.drop_perm20:+.4f} | {r.drop_censor20:+.4f} | '
                  f'{r.drop_dur_noise:+.4f} | {r.drop_top5_delete:+.4f} | '
                  f'**{r.avg_drop:+.4f}** |')
    md.append('')

    # Fragile variants: top 1/3 by avg_drop
    fragile_thresh = df_rob['avg_drop'].quantile(0.66)
    fragile = df_rob[df_rob['avg_drop'] >= fragile_thresh]
    md.append('## Fragile variants (DO NOT submit — single-fold luck risk)')
    md.append('')
    md.append('Variants in the top tercile of avg-drop. These are the most '
              'sensitive to label / censoring / time / patient perturbations:')
    md.append('')
    for r in fragile.itertuples(index=False):
        md.append(f'- `{r.variant}` (avg drop **{r.avg_drop:+.4f}**, '
                  f'public LB {r.public_LB:.4f})')
    md.append('')

    md.append('## Most "load-bearing" training patients')
    md.append('')
    md.append('Patients whose removal MOST drops C across variants. These are '
              'the ones any val_test bug-spotting effort should examine first.')
    md.append('')
    md.append('### Hepatic endpoint')
    md.append('')
    md.append('| Rank | Patient idx | Patient ID | # variants in top-5 |')
    md.append('|---|---|---|---|')
    for rank, (idx, cnt) in enumerate(shared_h, 1):
        pid = patient_ids[idx] if idx < len(patient_ids) else 'NA'
        md.append(f'| {rank} | {idx} | `{pid}` | {cnt}/{len(stems)} |')
    md.append('')
    md.append('### Death endpoint')
    md.append('')
    md.append('| Rank | Patient idx | Patient ID | # variants in top-5 |')
    md.append('|---|---|---|---|')
    for rank, (idx, cnt) in enumerate(shared_d, 1):
        pid = patient_ids[idx] if idx < len(patient_ids) else 'NA'
        md.append(f'| {rank} | {idx} | `{pid}` | {cnt}/{len(stems)} |')
    md.append('')

    md.append('## Stress-test summary tables')
    md.append('')
    md.append('### A. Label permutation (mean over 5 seeds)')
    md.append('')
    md.append('Weighted LB at each permutation fraction, averaged across the '
              '5 perturbation seeds. Variants degrade roughly linearly with '
              'permutation fraction; faster degradation = more reliance on '
              'noisy labels (could be either bad signal or genuine signal).')
    md.append('')
    pivot = df_a.pivot_table(index='variant', columns='perm_frac',
                             values='weighted_LB').round(4)
    md.append('| Variant | f=0 | f=0.05 | f=0.10 | f=0.20 |')
    md.append('|---|---|---|---|---|')
    for v in stems:
        if v not in pivot.index:
            continue
        row = pivot.loc[v]
        md.append(f'| `{v}` | {row.get(0.0, float("nan")):.4f} | '
                  f'{row.get(0.05, float("nan")):.4f} | '
                  f'{row.get(0.10, float("nan")):.4f} | '
                  f'{row.get(0.20, float("nan")):.4f} |')
    md.append('')

    md.append('### C. Pair concentration (top 1% of pairs share of total C)')
    md.append('')
    md.append('| Variant | hep total pairs | hep top1% conc. | death total pairs | death top1% conc. |')
    md.append('|---|---|---|---|---|')
    for r in df_c.itertuples(index=False):
        md.append(f'| `{r.variant}` | {r.hep_total_pairs} | '
                  f'{r.hep_concentration_top1pct:.3f} | '
                  f'{r.death_total_pairs} | '
                  f'{r.death_concentration_top1pct:.3f} |')
    md.append('')

    md.append('## Caveats')
    md.append('')
    md.append('- `variant_proxy_oof` collapses meta-aware variants to a '
              'rank-mean fallback. The 15 stems map to only 3 recipes '
              '(`rank_mean`, `rank_median`, `rank_trimmed`). Differential '
              'rankings across variants reflect that small recipe set, not '
              'the actual submission diversity. Treat `meta_avg*` rows as '
              'a single point estimate.')
    md.append('- All concordances are computed on training-side OOF, not the '
              'public leaderboard. Use as a robustness *signal*, not as an '
              'LB predictor.')
    md.append('- Adversarial-deletion top-5 sets are computed independently '
              'per endpoint — overlap across variants is the diagnostic '
              'signal, not absolute drop magnitude.')

    out_md = REPO / 'AUDIT_SENSITIVITY.md'
    out_md.write_text('\n'.join(md))
    print(f'wrote {out_md}')
    print(f'wrote {FULL}/sensitivity_*.csv (5 files)')

    # Also stash the robustness ranking for callers
    df_rob.to_csv(FULL / 'sensitivity_robustness_ranking.csv', index=False)


if __name__ == '__main__':
    main()
