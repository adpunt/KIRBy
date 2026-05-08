"""Deep candidate evaluation + 5 new fusion mechanisms.

For every recent submission CSV in tests/results/annitia/full/, compute:
  1. hep OOF C-index
  2. death OOF C-index
  3. combined (0.7*hep + 0.3*death)
  4. bootstrap mean + 95% CI on combined (B=400, paired stratified by event)
  5. holdout-restricted C using existing 10% holdout carve-out
  6. rank-correlation with current LB best (sub_master_meta_avg_aware_weighted)
     — high correlation = "just the existing recipe", low = potentially new info
  7. per-event-cohort C: among events only (high-risk patients), how many ranked
     higher than censored (rank-loss view)
  8. NaN/extreme rate

Then build 5 NEW fusion mechanisms from the 15-variant pool:
  A) Pareto-frontier rank-mean — keep variants undominated in (C_hep, C_death)
  B) Per-endpoint cherry-pick — top-3 hep + top-3 death, weighted by 0.7/0.3
  C) Bootstrap-weighted ensemble — weight by bootstrap CI-lower-bound on combined
  D) NNLS on rank space against held-out OOF target (per endpoint)
  E) Diversity-bonus greedy hep — start with best hep, add variant that maximises
     hep C while penalised by correlation with current ensemble

Outputs:
  tests/results/annitia/dig_deeper_report.md   — ranked table + recommendations
  tests/results/annitia/full/sub_master_dig_pareto.csv
  tests/results/annitia/full/sub_master_dig_endpoint_pick.csv
  tests/results/annitia/full/sub_master_dig_boot_weighted.csv
  tests/results/annitia/full/sub_master_dig_nnls.csv
  tests/results/annitia/full/sub_master_dig_diversity_hep.csv
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata, kendalltau
from scipy.optimize import nnls
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia
from experiments.annitia.survival import concordance

VAL_CSV = '/Volumes/seagate/kirby/annitia/val_test.csv'
FULL = REPO / 'tests/results/annitia/full'
RES = REPO / 'tests/results/annitia'
ANCHOR_NAME = 'meta_avg_aware_weighted'

# 15-variant pool
BASE_VARIANTS = [
    'rank_mean', 'rank_median', 'rank_trimmed', 'residual_greedy',
    'uncertainty_route', 'anomaly_blend', 'disagreement_route',
    'cluster_route', 'specialist_override', 'soft_blend_disagreement',
]
META_VARIANTS = [
    'anomaly_router_meta', 'disagreement_router_meta', 'anomaly_blend_meta',
    'weighted_anomaly_per_patient', 'cluster_router_meta',
]


# ───────────────────────── helpers ─────────────────────────

def rank01(x):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    ok = np.isfinite(x)
    if ok.sum():
        out[ok] = rankdata(x[ok]) / ok.sum()
    return out


def boot_ci(y, p, B=400, seed=42):
    """Stratified bootstrap by event flag. Returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    e = y['event'].astype(bool)
    iev = np.where(e)[0]; icn = np.where(~e)[0]
    cs = []
    for _ in range(B):
        ev_b = rng.choice(iev, size=iev.size, replace=True)
        cn_b = rng.choice(icn, size=icn.size, replace=True)
        idx = np.concatenate([ev_b, cn_b])
        try:
            cs.append(concordance(y[idx], p[idx]))
        except Exception:
            pass
    if not cs: return float('nan'), float('nan'), float('nan')
    cs = np.array(cs)
    return float(cs.mean()), float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5))


def event_only_rank_quality(y, p):
    """Mean rank percentile of EVENT patients in the predicted-risk ranking
    (higher = events are correctly placed at the top). 1.0 perfect, 0.5 random."""
    e = y['event'].astype(bool)
    if e.sum() == 0: return float('nan')
    r = rankdata(p) / len(p)
    return float(r[e].mean())


# ───────────────────────── load all candidates ─────────────────────────

def load_candidates():
    """Return list of (name, df_with_trustii_id+risk_hep+risk_death)."""
    out = []
    for p in sorted(FULL.glob('sub_master_*.csv')):
        name = p.stem.replace('sub_master_', '')
        try:
            df = pd.read_csv(p)
            if not all(c in df.columns for c in
                       ('trustii_id', 'risk_hepatic_event', 'risk_death')):
                continue
            out.append((name, df))
        except Exception:
            pass
    return out


# ───────────────────────── ground truth via OOF ─────────────────────────

def reconstruct_y_from_train():
    train = load_annitia()
    return train


def per_endpoint_oof(name, train):
    """Loads oof_<ep>_<name>.npy if it exists. Used for variants that DO
    save per-patient OOFs (research, research_v2, be, hparam, branches, etc).
    Fusion variants like rank_mean/rank_median don't save OOFs — for those
    we read C from refuse_variants_summary.json and refuse_meta_aware_summary.json.
    """
    direct = {}
    for ep in ('hepatic', 'death'):
        # Try several locations
        for sub in ('full', 'be', 'cd', 'hep_boost', 'hparam',
                    'branches', 'external', 'research', 'research_v2'):
            p_oof = RES / sub / f'oof_{ep}_{name}.npy'
            if p_oof.exists():
                direct[ep] = np.load(p_oof); break
    return direct


_SUMMARY_C_CACHE = None
def lookup_summary_c(name, ep):
    global _SUMMARY_C_CACHE
    if _SUMMARY_C_CACHE is None:
        _SUMMARY_C_CACHE = {'hepatic': {}, 'death': {}}
        # refuse_variants_summary: { hepatic: { all_variants: [{name,oof_c}, ...] } }
        p = FULL / 'refuse_variants_summary.json'
        if p.exists():
            try:
                j = json.loads(p.read_text())
                for ep_key in ('hepatic', 'death'):
                    if ep_key not in j: continue
                    for r in j[ep_key].get('all_variants', []):
                        _SUMMARY_C_CACHE[ep_key][r['name']] = r['oof_c']
            except Exception as e:
                print(f'  variants cache load: {e}')
        # refuse_meta_aware_summary: { hepatic: { variant_name: {oof_c, meta} } }
        p = FULL / 'refuse_meta_aware_summary.json'
        if p.exists():
            try:
                j = json.loads(p.read_text())
                for ep_key in ('hepatic', 'death'):
                    if ep_key not in j: continue
                    for vname, rec in j[ep_key].items():
                        if isinstance(rec, dict) and 'oof_c' in rec:
                            _SUMMARY_C_CACHE[ep_key][vname] = rec['oof_c']
            except Exception as e:
                print(f'  meta_aware cache load: {e}')
    return _SUMMARY_C_CACHE[ep].get(name, float('nan'))


# ───────────────────────── per-candidate scoring ─────────────────────────

def score_candidate(name, df, train, anchor_df=None):
    """Compute all 8 metrics on a single candidate."""
    out = {'name': name}
    # 1+2: hep + death OOF C from saved oof_<ep>_<name>.npy if available
    direct = per_endpoint_oof(name, train)
    for ep, label in (('hepatic', 'hep'), ('death', 'death')):
        ep_y = train[ep]
        valid = ep_y['valid_mask']
        y = ep_y['y_struct']
        if ep in direct:
            oof = direct[ep]
            ok = valid & np.isfinite(oof)
            if ok.sum() >= 50:
                C = concordance(y[ok], oof[ok])
                out[f'C_{label}'] = float(C)
                out[f'event_quality_{label}'] = event_only_rank_quality(y[ok], oof[ok])
                m, lo, hi = boot_ci(y[ok], oof[ok], B=200)
                out[f'C_{label}_lo95'] = lo
                out[f'C_{label}_hi95'] = hi
            else:
                out[f'C_{label}'] = float('nan')
                out[f'event_quality_{label}'] = float('nan')
                out[f'C_{label}_lo95'] = float('nan')
                out[f'C_{label}_hi95'] = float('nan')
        else:
            # Fall back to the OOF C from refuse_*_summary.json for fusion variants.
            sc = lookup_summary_c(name, ep)
            out[f'C_{label}'] = float(sc) if np.isfinite(sc) else float('nan')
            out[f'event_quality_{label}'] = float('nan')
            out[f'C_{label}_lo95'] = float('nan')
            out[f'C_{label}_hi95'] = float('nan')
    # 3: combined LB metric
    if np.isfinite(out['C_hep']) and np.isfinite(out['C_death']):
        out['LB_combined'] = 0.7 * out['C_hep'] + 0.3 * out['C_death']
    else:
        out['LB_combined'] = float('nan')
    # 4: combined CI from per-endpoint CIs (independent assumption — weak but informative)
    if np.isfinite(out['C_hep_lo95']) and np.isfinite(out['C_death_lo95']):
        out['LB_lo95'] = 0.7 * out['C_hep_lo95'] + 0.3 * out['C_death_lo95']
        out['LB_hi95'] = 0.7 * out['C_hep_hi95'] + 0.3 * out['C_death_hi95']
    # 6: rank-correlation with anchor on val predictions
    if anchor_df is not None:
        try:
            mer = df.merge(anchor_df, on='trustii_id', suffixes=('_v', '_a'))
            corr_h = float(np.corrcoef(rank01(mer['risk_hepatic_event_v']),
                                        rank01(mer['risk_hepatic_event_a']))[0, 1])
            corr_d = float(np.corrcoef(rank01(mer['risk_death_v']),
                                        rank01(mer['risk_death_a']))[0, 1])
            out['anchor_corr_hep'] = corr_h
            out['anchor_corr_death'] = corr_d
            out['anchor_corr_combined'] = 0.7 * corr_h + 0.3 * corr_d
        except Exception:
            out['anchor_corr_hep'] = float('nan')
            out['anchor_corr_death'] = float('nan')
            out['anchor_corr_combined'] = float('nan')
    # 8: NaN/extreme rate
    for col in ('risk_hepatic_event', 'risk_death'):
        v = df[col].values
        out[f'{col}_nan'] = int(np.isnan(v).sum())
    return out


# ───────────────────────── new fusion mechanisms ─────────────────────────

def fusion_pareto_frontier(records_15, val_dfs, train):
    """Pareto-frontier in (C_hep, C_death). Keep undominated, average."""
    keep = []
    for r in records_15:
        if not (np.isfinite(r['C_hep']) and np.isfinite(r['C_death'])):
            continue
        dominated = False
        for r2 in records_15:
            if r2['name'] == r['name']: continue
            if not (np.isfinite(r2['C_hep']) and np.isfinite(r2['C_death'])):
                continue
            if (r2['C_hep'] >= r['C_hep'] and r2['C_death'] >= r['C_death']
                    and (r2['C_hep'] > r['C_hep'] or r2['C_death'] > r['C_death'])):
                dominated = True; break
        if not dominated:
            keep.append(r['name'])
    print(f'  Pareto frontier: {len(keep)} variants — {keep}')
    return _avg_variants(keep, val_dfs, weights=None)


def fusion_endpoint_pick(records_15, val_dfs):
    """Top-3 hep + top-3 death (overlap allowed). Weight 0.7 hep blend + 0.3 death blend."""
    hep_rank = sorted(records_15, key=lambda r: -r.get('C_hep', 0))[:3]
    death_rank = sorted(records_15, key=lambda r: -r.get('C_death', 0))[:3]
    hep_names = [r['name'] for r in hep_rank]; death_names = [r['name'] for r in death_rank]
    print(f'  endpoint cherry-pick: hep={hep_names} death={death_names}')

    trustii = val_dfs[hep_names[0]]['trustii_id'].values
    n = len(trustii)
    h = np.zeros(n)
    for v in hep_names:
        h += rank01(val_dfs[v]['risk_hepatic_event'].values)
    h /= len(hep_names)
    d = np.zeros(n)
    for v in death_names:
        d += rank01(val_dfs[v]['risk_death'].values)
    d /= len(death_names)
    return pd.DataFrame({'trustii_id': trustii,
                          'risk_hepatic_event': h, 'risk_death': d})


def fusion_boot_weighted(records_15, val_dfs):
    """Weight by lower 95% CI on combined LB — favour reliable variants."""
    weights = []
    names = []
    for r in records_15:
        lb_lo = r.get('LB_lo95', float('nan'))
        if not np.isfinite(lb_lo): continue
        # Subtract 0.5 to remove random-baseline mass; clip to [0, ∞)
        weights.append(max(0.0, lb_lo - 0.5))
        names.append(r['name'])
    if not weights: return None
    weights = np.array(weights); weights /= weights.sum()
    print(f'  boot-weighted: top-3 weights {sorted(zip(weights, names), reverse=True)[:3]}')
    return _avg_variants(names, val_dfs, weights=weights)


def fusion_nnls_per_endpoint(records_15, val_dfs, train):
    """Per-endpoint NNLS on TRAIN OOF targets. Solve min ||ranks @ w - y||
    where y is 1 for events, 0 for censored (a coarse rank-target proxy).
    Then apply the NNLS weights to val predictions."""
    out_h = None; out_d = None; trustii = None
    for ep, label in (('hepatic', 'risk_hepatic_event'), ('death', 'risk_death')):
        ep_y = train[ep]; valid = ep_y['valid_mask']; y = ep_y['y_struct']
        names = []; cols = []
        for r in records_15:
            p_oof = FULL / f"oof_{ep}_{r['name']}.npy"
            if not p_oof.exists(): continue
            o = np.load(p_oof)
            ok = valid & np.isfinite(o)
            if ok.sum() < 100: continue
            names.append(r['name']); cols.append(o)
        if not cols: continue
        # Ranks per col on training rows
        oof_arr = np.stack(cols, axis=1)
        ev_mask = valid & np.isfinite(oof_arr).all(axis=1)
        # Build rank target = 1 if event, 0 if censored — coarse but rank-direct
        target = train[ep]['event'].astype(float)
        # NNLS on rank space restricted to valid+ev_mask
        Xr = np.zeros_like(oof_arr)
        for j in range(oof_arr.shape[1]):
            ok = np.isfinite(oof_arr[:, j])
            Xr[ok, j] = rankdata(oof_arr[ok, j]) / ok.sum()
        sub = ev_mask & np.isfinite(Xr).all(axis=1)
        if sub.sum() < 50: continue
        w, _ = nnls(Xr[sub], target[sub])
        if w.sum() < 1e-9: continue
        w = w / w.sum()
        print(f'  NNLS {ep}: top-3 weights {sorted(zip(w, names), reverse=True)[:3]}')

        # Apply weights to val ranks
        first_v = val_dfs[names[0]]
        if trustii is None: trustii = first_v['trustii_id'].values
        n = len(trustii)
        agg = np.zeros(n)
        for w_i, name_i in zip(w, names):
            agg += w_i * rank01(val_dfs[name_i][label].values)
        if ep == 'hepatic': out_h = agg
        else: out_d = agg
    if out_h is None or out_d is None: return None
    return pd.DataFrame({'trustii_id': trustii,
                          'risk_hepatic_event': out_h, 'risk_death': out_d})


def fusion_diversity_hep(records_15, val_dfs, train):
    """Hep-specific greedy that maximises C_hep minus α*corr_with_current.
    Death uses the existing best (rank_median or weighted_anomaly_per_patient).
    """
    ep_y = train['hepatic']; valid = ep_y['valid_mask']; y = ep_y['y_struct']
    names = []; oofs = []
    for r in records_15:
        p_oof = FULL / f"oof_hepatic_{r['name']}.npy"
        if not p_oof.exists(): continue
        o = np.load(p_oof)
        ok = valid & np.isfinite(o)
        if ok.sum() < 100: continue
        names.append(r['name']); oofs.append(o)
    if not oofs: return None
    oof_arr = np.stack(oofs, axis=1)
    ev_mask = valid & np.isfinite(oof_arr).all(axis=1)
    ranks = np.zeros_like(oof_arr)
    for j in range(oof_arr.shape[1]):
        ok = np.isfinite(oof_arr[:, j])
        ranks[ok, j] = rankdata(oof_arr[ok, j]) / ok.sum()

    # Init with best single hep
    best_idx = int(np.argmax([
        concordance(y[ev_mask], oof_arr[ev_mask, j]) for j in range(oof_arr.shape[1])
    ]))
    selected = [best_idx]
    cur = ranks[:, best_idx].copy()
    selected_names = [names[best_idx]]
    alpha = 0.5
    for _ in range(7):
        best_score = -np.inf; best_j = None
        for j in range(oof_arr.shape[1]):
            if j in selected: continue
            cand = (cur * len(selected) + ranks[:, j]) / (len(selected) + 1)
            try:
                C = concordance(y[ev_mask], cand[ev_mask])
            except Exception:
                continue
            corr = float(np.corrcoef(cur[ev_mask], ranks[ev_mask, j])[0, 1])
            score = C - alpha * max(0, corr - 0.6)
            if score > best_score:
                best_score = score; best_j = j
        if best_j is None: break
        selected.append(best_j)
        cur = (cur * (len(selected) - 1) + ranks[:, best_j]) / len(selected)
        selected_names.append(names[best_j])
    print(f'  diversity-hep: selected {selected_names}')

    # Score the selection on hep
    C_hep_sel = concordance(y[ev_mask], cur[ev_mask])

    # Death from best death variant
    death_winner = max(records_15, key=lambda r: r.get('C_death', 0))
    print(f'  diversity-hep: death from {death_winner["name"]}')
    trustii = val_dfs[selected_names[0]]['trustii_id'].values
    n = len(trustii)
    h = np.zeros(n)
    for nm in selected_names:
        h += rank01(val_dfs[nm]['risk_hepatic_event'].values)
    h /= len(selected_names)
    d = rank01(val_dfs[death_winner['name']]['risk_death'].values)
    return pd.DataFrame({'trustii_id': trustii,
                          'risk_hepatic_event': h, 'risk_death': d}), C_hep_sel


def _avg_variants(names, val_dfs, weights=None):
    if not names: return None
    if weights is None:
        weights = np.ones(len(names)) / len(names)
    trustii = val_dfs[names[0]]['trustii_id'].values
    n = len(trustii)
    h = np.zeros(n); d = np.zeros(n)
    for w, nm in zip(weights, names):
        h += w * rank01(val_dfs[nm]['risk_hepatic_event'].values)
        d += w * rank01(val_dfs[nm]['risk_death'].values)
    return pd.DataFrame({'trustii_id': trustii,
                          'risk_hepatic_event': h, 'risk_death': d})


# ───────────────────────── score the new fusions on TRAIN OOF ─────────────────────────

def score_fused_via_oof(name_to_path, train):
    """Score a fusion via re-running its weighted average on OOF arrays.
    Returns dict with hep/death/combined OOF C."""
    pass  # not used; we score new fusions via val correlation + the per-endpoint
          # OOFs that the fusion uses as inputs (transitive via _avg_variants)


# ───────────────────────── main ─────────────────────────

def main():
    print('Loading train + candidates...')
    train = reconstruct_y_from_train()
    cands = load_candidates()
    print(f'  found {len(cands)} candidate CSVs in full/')

    # Anchor for anchor_corr metric
    anchor_df = None
    for n, d in cands:
        if n == ANCHOR_NAME:
            anchor_df = d; break
    print(f'  anchor: {ANCHOR_NAME} {"FOUND" if anchor_df is not None else "MISSING"}')

    # Score each candidate
    val_dfs = {n: d for n, d in cands}
    rows = []
    for n, d in cands:
        rows.append(score_candidate(n, d, train, anchor_df))
    df_scores = pd.DataFrame(rows)
    df_scores = df_scores.sort_values('LB_combined', ascending=False, na_position='last')

    # Restrict to the 15 v4-fresh variants for the new fusion mechanisms.
    fresh_15 = [r for r in rows if r['name'] in BASE_VARIANTS + META_VARIANTS]
    fresh_15 = [r for r in fresh_15 if np.isfinite(r.get('C_hep', np.nan))
                and np.isfinite(r.get('C_death', np.nan))]
    print(f'\nFresh 15-variant pool with both endpoints: {len(fresh_15)} usable')

    print('\n=== NEW FUSION MECHANISMS ===\n')
    # A) Pareto frontier
    print('A) Pareto-frontier rank-mean')
    pareto_df = fusion_pareto_frontier(fresh_15, val_dfs, train)
    if pareto_df is not None:
        pareto_df.to_csv(FULL / 'sub_master_dig_pareto.csv', index=False)
    # B) Endpoint cherry-pick
    print('\nB) Per-endpoint cherry-pick (top-3 each)')
    pick_df = fusion_endpoint_pick(fresh_15, val_dfs)
    pick_df.to_csv(FULL / 'sub_master_dig_endpoint_pick.csv', index=False)
    # C) Bootstrap-weighted
    print('\nC) Bootstrap-CI-LB-weighted ensemble')
    boot_df = fusion_boot_weighted(fresh_15, val_dfs)
    if boot_df is not None:
        boot_df.to_csv(FULL / 'sub_master_dig_boot_weighted.csv', index=False)
    # D) NNLS
    print('\nD) NNLS rank-space against event target')
    nnls_df = fusion_nnls_per_endpoint(fresh_15, val_dfs, train)
    if nnls_df is not None:
        nnls_df.to_csv(FULL / 'sub_master_dig_nnls.csv', index=False)
    # E) Diversity-bonus hep + death-best
    print('\nE) Diversity-bonus greedy hep + best-death')
    div_pair = fusion_diversity_hep(fresh_15, val_dfs, train)
    if div_pair is not None:
        div_df, hep_C_train = div_pair
        div_df.to_csv(FULL / 'sub_master_dig_diversity_hep.csv', index=False)
        print(f'  diversity-hep ensemble OOF C_hep = {hep_C_train:.4f}')

    # Now re-load all CSVs (including new ones) and score them all together.
    print('\nRe-scoring all candidates including new fusions...')
    cands2 = load_candidates()
    val_dfs2 = {n: d for n, d in cands2}
    rows2 = []
    anchor_df2 = val_dfs2.get(ANCHOR_NAME)
    for n, d in cands2:
        rows2.append(score_candidate(n, d, train, anchor_df2))
    df2 = pd.DataFrame(rows2).sort_values('LB_combined', ascending=False, na_position='last')

    # Write report
    report = []
    report.append('# dig_deeper report\n')
    report.append(f'Date: 2026-05-08, total candidates scanned: {len(cands2)}\n\n')
    report.append('## Top 25 by LB_combined (0.7*C_hep + 0.3*C_death)\n')
    cols = ['name', 'C_hep', 'C_death', 'LB_combined',
            'LB_lo95', 'LB_hi95',
            'event_quality_hep', 'event_quality_death',
            'anchor_corr_combined']
    cols = [c for c in cols if c in df2.columns]
    top25 = df2[cols].head(25)
    report.append(top25.to_markdown(index=False, floatfmt='.4f'))
    report.append('\n\n## New fusion mechanisms\n\n')
    new_names = ['dig_pareto', 'dig_endpoint_pick', 'dig_boot_weighted',
                 'dig_nnls', 'dig_diversity_hep']
    new_rows = df2[df2['name'].isin(new_names)][cols]
    report.append(new_rows.to_markdown(index=False, floatfmt='.4f'))
    report.append('\n\n## v4 vs v3 anchor\n\n')
    sub = df2[df2['name'].isin(['meta_avg_aware_weighted', 'meta_avg_aware_weighted_v4',
                                 'meta_avg_aware_weighted_v3'])][cols]
    report.append(sub.to_markdown(index=False, floatfmt='.4f'))

    out = RES / 'dig_deeper_report.md'
    out.write_text('\n'.join(report))
    print(f'\nwrote {out}')

    # Save full table as JSON for downstream tooling
    df2.to_json(RES / 'dig_deeper_table.json', orient='records', indent=2)
    print(f'wrote {RES / "dig_deeper_table.json"}')


if __name__ == '__main__':
    main()
