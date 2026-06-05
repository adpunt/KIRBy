"""Build 5 mechanism-divergent aggressive candidates.

Motivation: the Bayesian LB predictor regresses to the anchor mode (~0.93)
because all 10 anchors fall in a narrow band. It cannot discriminate genuinely
mechanism-different candidates. The leaderboard top is 0.979, so a divergent
candidate could plausibly score 0.95+. Pick by mechanism divergence, not by
predicted_LB.

Outputs (under tests/results/annitia/full/):
  sub_master_aggressive_xstack.csv         - 50/50 rank-blend anchor + coxnet_stack_with_uncertainty (uniqueness 0.71)
  sub_master_aggressive_routed_blend.csv   - anchor for trust patients, cluster_router_meta/disagreement_route for top-10% disagreement
  sub_master_aggressive_extreme.csv        - hep top 5% pushed to 0.99, hep bottom 50% pushed to <0.5; compress middle
  sub_master_aggressive_specialist_only.csv- only highest-trust hep & death individual models (per oof_diagnostics.md)
  sub_master_aggressive_topk_only.csv      - K=3 hep, K=3 death, pure top-K rank mean

For each candidate we:
  - Run pre_submit_check.py
  - Compute OOF C (hep, death, weighted 0.7/0.3) using OOF-equivalent variants
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia               # noqa: E402
from experiments.annitia.survival import concordance                # noqa: E402
from experiments.annitia.pre_submit_check import check as pre_check  # noqa: E402

FULL = REPO / 'tests/results/annitia/full'
BRANCHES = REPO / 'tests/results/annitia/branches'

ANCHOR = 'sub_master_meta_avg_aware_weighted'

# Trusted models per oof_diagnostics.md (top by lower 95% CI / pooled C).
HEP_TRUSTED = [
    'synaug-none-rsf-500-n0__snapshot+clinical_pro+fpca+gp_smooth',  # 0.868
    'tsfm-rsf__tsfm',                                                # 0.842
    'tsfm-gbsa__tsfm',                                               # 0.842
    'catboost-D2H__snapshot+clinical_pro+fpca+gp_smooth',            # 0.838
    'tabpfn25-disc-K10__snapshot+clinical_pro+fpca+gp_smooth',       # 0.838
]
HEP_TOPK = [
    'synaug-none-rsf-500-n0__snapshot+clinical_pro+fpca+gp_smooth',  # 0.868
    'tsfm-rsf__tsfm',                                                # 0.842
    'tsfm-gbsa__tsfm',                                               # 0.842
]
DEATH_TRUSTED = [
    'tabpfn-landmark__snapshot+clinical_pro+fpca+gp_smooth',         # 0.933 HIGH
    'catboost-H2D__snapshot+clinical_pro+fpca+gp_smooth',            # 0.901 MEDIUM
    'dsm-K4__snapshot+clinical_pro+fpca+visit_density+clinical_slope',  # 0.890 MEDIUM
    'tabpfn25-disc-K20__snapshot+clinical_pro+fpca+gp_smooth',       # 0.888 HIGH
    'dsm-K4__snapshot+clinical_pro+meld_na+visit_density',           # 0.877 MEDIUM
]
DEATH_TOPK = [
    'tabpfn-landmark__snapshot+clinical_pro+fpca+gp_smooth',         # 0.933
    'catboost-H2D__snapshot+clinical_pro+fpca+gp_smooth',            # 0.901
    'tabpfn25-disc-K20__snapshot+clinical_pro+fpca+gp_smooth',       # 0.888
]


def rank01(x: np.ndarray) -> np.ndarray:
    """Rank-normalise a 1D vector to [0, 1] (NaN-safe)."""
    out = np.full_like(x, np.nan, dtype=float)
    m = np.isfinite(x)
    if m.sum():
        out[m] = (rankdata(x[m]) - 1) / max(m.sum() - 1, 1)
    return out


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(FULL / f'{name}.csv')


def load_npy(prefix: str, ep: str, model: str) -> np.ndarray:
    return np.load(BRANCHES / f'{prefix}_{ep}_{model}.npy')


# ---------------------------------------------------------------------------
# Candidate builders. Each returns dict with 'val' DataFrame (423 rows) and
# 'oof_h' / 'oof_d' (1253-length OOF predictions, rank01).
# ---------------------------------------------------------------------------

def build_xstack() -> Dict:
    """50/50 rank-blend of anchor and coxnet_stack_with_uncertainty (uniqueness 0.71)."""
    anc = load_csv(ANCHOR)
    cox = load_csv('sub_master_coxnet_stack_with_uncertainty')
    out = anc[['trustii_id']].copy()
    for col in ['risk_hepatic_event', 'risk_death']:
        a = rank01(anc[col].to_numpy())
        c = rank01(cox[col].to_numpy())
        out[col] = 0.5 * a + 0.5 * c
    # OOF proxy: anchor proxy = mean of all base OOFs; coxnet proxy = same shape from train fit.
    # We approximate the anchor's OOF as the mean rank of trusted OOFs for that endpoint.
    return {'val': out, 'oof_h': _oof_proxy_blend(0.5, HEP_TRUSTED, 'hepatic'),
            'oof_d': _oof_proxy_blend(0.5, DEATH_TRUSTED, 'death')}


def _oof_proxy_blend(coxnet_w: float, trusted: List[str], ep: str) -> np.ndarray:
    """Anchor-OOF proxy = rank mean of trusted base OOFs;
    coxnet OOF proxy unavailable directly, so we approximate it as the
    consensus of all base OOFs (gives a structurally-different blend)."""
    anc_arrs = [rank01(load_npy('oof', ep, m)) for m in trusted]
    anc_proxy = np.nanmean(np.stack(anc_arrs, axis=1), axis=1)
    # consensus = all OOFs
    all_files = list(BRANCHES.glob(f'oof_{ep}_*.npy'))
    arrs = []
    for f in all_files:
        try:
            a = np.load(f)
            if np.isfinite(a).sum() < 50 or np.std(a[np.isfinite(a)]) <= 1e-10:
                continue
            arrs.append(rank01(a))
        except Exception:
            continue
    cons = np.nanmean(np.stack(arrs, axis=1), axis=1)
    return (1 - coxnet_w) * anc_proxy + coxnet_w * cons


def build_routed_blend() -> Dict:
    """For top-10% disagreement patients, swap anchor -> cluster_router_meta;
    keep anchor for the trust 90%."""
    anc = load_csv(ANCHOR)
    rout = load_csv('sub_master_cluster_router_meta')
    diss = load_csv('sub_master_disagreement_route')

    out = anc[['trustii_id']].copy()
    # Build val-side disagreement = std across {anchor, router, disagreement} ranks
    for col in ['risk_hepatic_event', 'risk_death']:
        a = rank01(anc[col].to_numpy())
        r = rank01(rout[col].to_numpy())
        d = rank01(diss[col].to_numpy())
        disag = np.std(np.stack([a, r, d], axis=1), axis=1)
        # top-10% disagreement -> avg of router+diss; rest -> anchor
        thr = np.quantile(disag, 0.90)
        swap = disag >= thr
        blended = a.copy()
        blended[swap] = 0.5 * r[swap] + 0.5 * d[swap]
        out[col] = blended

    # OOF proxy similarly for hep/death
    return {'val': out,
            'oof_h': _oof_routed_proxy(HEP_TRUSTED, 'hepatic'),
            'oof_d': _oof_routed_proxy(DEATH_TRUSTED, 'death')}


def _oof_routed_proxy(trusted: List[str], ep: str) -> np.ndarray:
    arrs = [rank01(load_npy('oof', ep, m)) for m in trusted]
    anc = np.nanmean(np.stack(arrs, axis=1), axis=1)
    disag = np.nanstd(np.stack(arrs, axis=1), axis=1)
    thr = np.nanquantile(disag, 0.90)
    swap = disag >= thr
    # Use median of trusted for swapped patients (different mechanism)
    med = np.nanmedian(np.stack(arrs, axis=1), axis=1)
    out = anc.copy()
    out[swap] = med[swap]
    return out


def build_extreme() -> Dict:
    """Aggressive monotone reshape of anchor: top 5% -> 0.99, bottom 50% -> 0..0.5,
    middle 45% compressed."""
    anc = load_csv(ANCHOR)
    out = anc[['trustii_id']].copy()
    for col in ['risk_hepatic_event', 'risk_death']:
        r = rank01(anc[col].to_numpy())
        e = _reshape_extreme(r)
        out[col] = e
    # OOF: same reshape on anchor proxy
    h_proxy = _anchor_proxy(HEP_TRUSTED, 'hepatic')
    d_proxy = _anchor_proxy(DEATH_TRUSTED, 'death')
    return {'val': out,
            'oof_h': _reshape_extreme(rank01(h_proxy)),
            'oof_d': _reshape_extreme(rank01(d_proxy))}


def _reshape_extreme(r: np.ndarray) -> np.ndarray:
    """Tied-band reshape: top 5% all collapse to 0.99, bottom 50% all
    collapse to a tied low value, middle 45% kept as-is. This breaks rank
    monotonicity (within-band ties) so it actually changes C-index.
    """
    out = r.copy()
    finite = np.isfinite(r)
    rr = r[finite]
    new = np.empty_like(rr)
    bot = rr < 0.50
    mid = (rr >= 0.50) & (rr < 0.95)
    top = rr >= 0.95
    # bottom 50% -> tied at 0.25 (clearly low)
    new[bot] = 0.25
    # middle keeps its rank (compressed band, but monotone within)
    new[mid] = rr[mid]
    # top 5% -> tied at 0.99
    new[top] = 0.99
    out[finite] = new
    return out


def _anchor_proxy(trusted: List[str], ep: str) -> np.ndarray:
    arrs = [rank01(load_npy('oof', ep, m)) for m in trusted]
    return np.nanmean(np.stack(arrs, axis=1), axis=1)


def build_specialist_only() -> Dict:
    """Drop low-trust models entirely. Only the trusted set, rank-mean."""
    out = pd.DataFrame({'trustii_id': np.arange(1, 424)})
    h_val = [rank01(load_npy('val', 'hepatic', m)) for m in HEP_TRUSTED]
    d_val = [rank01(load_npy('val', 'death', m)) for m in DEATH_TRUSTED]
    out['risk_hepatic_event'] = np.nanmean(np.stack(h_val, axis=1), axis=1)
    out['risk_death'] = np.nanmean(np.stack(d_val, axis=1), axis=1)
    return {'val': out,
            'oof_h': _anchor_proxy(HEP_TRUSTED, 'hepatic'),
            'oof_d': _anchor_proxy(DEATH_TRUSTED, 'death')}


def build_topk_only() -> Dict:
    """Pure top-K=3 each endpoint, rank-mean. Most aggressive cut."""
    out = pd.DataFrame({'trustii_id': np.arange(1, 424)})
    h_val = [rank01(load_npy('val', 'hepatic', m)) for m in HEP_TOPK]
    d_val = [rank01(load_npy('val', 'death', m)) for m in DEATH_TOPK]
    out['risk_hepatic_event'] = np.nanmean(np.stack(h_val, axis=1), axis=1)
    out['risk_death'] = np.nanmean(np.stack(d_val, axis=1), axis=1)
    return {'val': out,
            'oof_h': _anchor_proxy(HEP_TOPK, 'hepatic'),
            'oof_d': _anchor_proxy(DEATH_TOPK, 'death')}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def oof_score(train, h_proxy, d_proxy):
    h_y = train['hepatic']['y_struct']; h_v = train['hepatic']['valid_mask']
    d_y = train['death']['y_struct'];   d_v = train['death']['valid_mask']
    h_finite = h_v & np.isfinite(h_proxy) & np.isfinite(h_y['time'])
    d_finite = d_v & np.isfinite(d_proxy) & np.isfinite(d_y['time'])
    h_c = float(concordance(h_y[h_finite], h_proxy[h_finite])) if h_finite.sum() >= 5 else float('nan')
    d_c = float(concordance(d_y[d_finite], d_proxy[d_finite])) if d_finite.sum() >= 5 else float('nan')
    return h_c, d_c, 0.7 * h_c + 0.3 * d_c


def _validate_val(out: pd.DataFrame):
    assert list(out.columns) == ['trustii_id', 'risk_hepatic_event', 'risk_death']
    assert len(out) == 423
    assert (out['trustii_id'].astype(int).to_numpy() == np.arange(1, 424)).all()
    out['trustii_id'] = out['trustii_id'].astype(int)
    return out


def main():
    print('=== Building aggressive candidates ===\n')
    train = load_annitia()

    builders = {
        'sub_master_aggressive_xstack': build_xstack,
        'sub_master_aggressive_routed_blend': build_routed_blend,
        'sub_master_aggressive_extreme': build_extreme,
        'sub_master_aggressive_specialist_only': build_specialist_only,
        'sub_master_aggressive_topk_only': build_topk_only,
    }

    results = []
    for name, fn in builders.items():
        print(f'\n--- {name} ---')
        bundle = fn()
        out = _validate_val(bundle['val'])
        path = FULL / f'{name}.csv'
        out.to_csv(path, index=False)
        print(f'  wrote {path}')

        h_c, d_c, lb = oof_score(train, bundle['oof_h'], bundle['oof_d'])
        print(f'  OOF C: hep={h_c:.4f}  death={d_c:.4f}  weighted(0.7/0.3)={lb:.4f}')

        rc = pre_check(path)
        results.append({'name': name, 'oof_h': h_c, 'oof_d': d_c, 'oof_lb': lb,
                        'pre_check_rc': rc})

    print('\n=== Summary ===')
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv(FULL / 'aggressive_candidates_summary.csv', index=False)
    return df


if __name__ == '__main__':
    main()
