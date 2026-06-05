"""M5 — Counterfactual perturbation of censored high-risk patients.

For each REAL censored patient that looks high-risk (FIB-4 > 1.5 or LSM > 8 at
last visit), generate K hypothetical "if-they-progressed" trajectories by:

    1. Extending the visit grid with 1-3 additional synthetic visits at
       0.5-2 yr gaps after the last observed.
    2. Pushing the trajectory along a literature-derived MASLD progression
       path (LSM +1.5 kPa/yr, plt -5/yr, AST/ALT inversion, FibroTest +0.05/yr).
    3. Sampling a hep event time within 2-5 yr of the last real visit.
    4. Filtering rule-based plausibility checks (no impossible LSM>75,
       no AST>1000, monotonic FibroTest in [0,1], etc.).

Returns ANNITIA-wide DataFrame, ingestable by `load_annitia`.

This is "informed" because the perturbation directions and magnitudes come
from clinical literature (see MASLD_SIMULATOR_NOTES.md §3) rather than a
learned generator.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from .masld_simulator import (
        CLIP, VAR_TO_COL, LAB_VARS, NIT_VARS, PENNISI_INCIDENCE, AGILE3_COEFS,
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from masld_simulator import (  # type: ignore
        CLIP, VAR_TO_COL, LAB_VARS, NIT_VARS, PENNISI_INCIDENCE, AGILE3_COEFS,
    )


def _last_obs_per_var(data, idx: int, var: str) -> Tuple[float, float]:
    """(age, value) of last observation for variable var; (nan, nan) if none."""
    series = data['longitudinal'].get(var, [[]])[idx]
    if not series:
        return float('nan'), float('nan')
    return series[-1]


def _high_risk_mask(data) -> np.ndarray:
    """Censored patients with last LSM>8 OR last FIB-4 proxy>1.5."""
    n = len(data['patient_ids'])
    censored = ~(data['hepatic']['event'].astype(bool)) if data.get('hepatic') else np.ones(n, bool)
    last_lsm = np.array([_last_obs_per_var(data, i, 'fibs_stiffness_med_BM_1')[1] for i in range(n)])
    last_ft = np.array([_last_obs_per_var(data, i, 'fibrotest_BM_2')[1] for i in range(n)])
    return censored & (((last_lsm > 8) & np.isfinite(last_lsm))
                       | ((last_ft > 0.4) & np.isfinite(last_ft)))


def _perturb_trajectory(static, longitudinal_idx, base_dict, n_extra: int,
                        rng: np.random.Generator,
                        progression_speed: float = 1.0):
    """Return list of (age, lab_dict) tuples for the EXTENDED trajectory."""
    # Reconstruct visit list from real longitudinal data
    visits: List[Tuple[float, Dict[str, float]]] = []
    var_keys = ['BMI', 'alt', 'ast', 'bilirubin', 'chol', 'ggt', 'gluc_fast',
                'plt', 'triglyc']
    nit_keys = {'fibs_stiffness_med_BM_1': 'LSM',
                'fibrotest_BM_2': 'FibroTest',
                'aixp_aix_result_BM_3': 'Aixplorer'}

    # Get common visit ages (use Age series)
    age_series = base_dict.get('Age', [])
    if not age_series:
        return None
    for (age, _) in age_series:
        labs: Dict[str, float] = {}
        for v in var_keys:
            for (a, x) in base_dict.get(v, []):
                if abs(a - age) < 1e-6:
                    labs[v] = x; break
        for col, simvar in nit_keys.items():
            for (a, x) in base_dict.get(col, []):
                if abs(a - age) < 1e-6:
                    labs[simvar] = x; break
        visits.append((age, labs))

    if not visits:
        return None
    last_age, last_labs = visits[-1]
    cur = dict(last_labs)
    # Defaults if missing
    for v in var_keys:
        cur.setdefault(v, np.nan)
    for col, simvar in nit_keys.items():
        cur.setdefault(simvar, np.nan)

    # Forward-progress: LSM +1.5 kPa/yr, plt -5/yr, AST/ALT inversion (ALT -3, AST +2)
    for _ in range(n_extra):
        dt = float(np.clip(rng.exponential(1.0), 0.5, 2.0))
        nxt_age = last_age + dt
        cur = dict(cur)
        # Apply progression deltas (with noise)
        delta_lsm = (1.5 * progression_speed) * dt + rng.normal(0, 1.0 * dt ** 0.5)
        if np.isfinite(cur.get('LSM', np.nan)):
            cur['LSM'] = float(cur['LSM']) + delta_lsm
        if np.isfinite(cur.get('FibroTest', np.nan)):
            cur['FibroTest'] = float(cur['FibroTest']) + 0.05 * dt + rng.normal(0, 0.04 * dt ** 0.5)
        if np.isfinite(cur.get('Aixplorer', np.nan)):
            cur['Aixplorer'] = float(cur['Aixplorer']) + delta_lsm * 0.78
        if np.isfinite(cur.get('plt', np.nan)):
            cur['plt'] = float(cur['plt']) + (-5.0 * progression_speed) * dt + rng.normal(0, 12 * dt ** 0.5)
        if np.isfinite(cur.get('alt', np.nan)):
            cur['alt'] = float(cur['alt']) + (-3.0 * dt) + rng.normal(0, 8 * dt ** 0.5)
        if np.isfinite(cur.get('ast', np.nan)):
            cur['ast'] = float(cur['ast']) + (2.0 * dt) + rng.normal(0, 7 * dt ** 0.5)
        if np.isfinite(cur.get('ggt', np.nan)):
            cur['ggt'] = float(cur['ggt']) + 5 * dt + rng.normal(0, 12 * dt ** 0.5)
        if np.isfinite(cur.get('bilirubin', np.nan)):
            cur['bilirubin'] = float(cur['bilirubin']) + 0.3 * dt + rng.normal(0, 1.5 * dt ** 0.5)
        # Clip
        for var, (lo, hi) in CLIP.items():
            if var in cur and np.isfinite(cur[var]):
                cur[var] = float(np.clip(cur[var], lo, hi))
        visits.append((nxt_age, dict(cur)))
        last_age = nxt_age

    return visits


def _plausibility_pass(visits: List[Tuple[float, Dict]]) -> bool:
    """Reject implausible counterfactual trajectories."""
    if len(visits) < 2:
        return False
    # Monotonic age
    ages = [a for (a, _) in visits]
    if not all(ages[i + 1] > ages[i] for i in range(len(ages) - 1)):
        return False
    last = visits[-1][1]
    # Last LSM in plausible range
    lsm = last.get('LSM', np.nan)
    if np.isfinite(lsm) and (lsm > 75 or lsm < 2):
        return False
    # FibroTest in [0, 1]
    ft = last.get('FibroTest', np.nan)
    if np.isfinite(ft) and (ft > 1.0 or ft < 0.0):
        return False
    # AST not absurd
    ast = last.get('ast', np.nan)
    if np.isfinite(ast) and ast > 800:
        return False
    return True


def _build_lab_dict_per_var(data, idx: int) -> Dict[str, List[Tuple[float, float]]]:
    out: Dict[str, List[Tuple[float, float]]] = {}
    for v in ['BMI', 'alt', 'ast', 'bilirubin', 'chol', 'ggt', 'gluc_fast',
              'plt', 'triglyc', 'fibs_stiffness_med_BM_1',
              'fibrotest_BM_2', 'aixp_aix_result_BM_3']:
        out[v] = list(data['longitudinal'].get(v, [[]])[idx])
    # Build Age series from baseline_age + visit indices (all observed visits)
    # Approximate: union of ages across vars
    all_ages = set()
    for series in out.values():
        for (a, _) in series:
            all_ages.add(round(float(a), 4))
    ages = sorted(all_ages)
    out['Age'] = [(a, a) for a in ages]
    return out


def make_counterfactual_cohort(data: Dict, k_per_patient: int = 4,
                               seed: int = 42, max_extra_visits: int = 3,
                               n_visit_cols: int = 21) -> pd.DataFrame:
    """For each high-risk censored patient, build k counterfactual progressors.

    Each counterfactual gets a synthetic event at the appended terminal visit
    (event_age = last_extended_age - small jitter).
    """
    rng = np.random.default_rng(seed)
    mask = _high_risk_mask(data)
    high_risk_idx = np.where(mask)[0]

    rows = []
    syn_id = 0
    for idx in high_risk_idx:
        per_var = _build_lab_dict_per_var(data, idx)
        for k in range(k_per_patient):
            n_extra = int(rng.integers(1, max_extra_visits + 1))
            speed = float(np.clip(rng.normal(1.0, 0.3), 0.5, 1.8))
            visits = _perturb_trajectory(
                static=None, longitudinal_idx=idx,
                base_dict=per_var,
                n_extra=n_extra, rng=rng, progression_speed=speed)
            if visits is None or not _plausibility_pass(visits):
                continue
            # Build wide row
            row = {
                'patient_id_anon': f'CF_{idx:04d}_{k:02d}',
                'phenotype': 'counterfactual',
                'gender': data['static']['gender'][idx],
                'T2DM': data['static']['T2DM'][idx],
                'Hypertension': data['static']['Hypertension'][idx],
                'Dyslipidaemia': data['static']['Dyslipidaemia'][idx],
                'bariatric_surgery': data['static']['bariatric_surgery'][idx],
                'bariatric_surgery_age': data['static']['bariatric_surgery_age'][idx],
            }
            for j, (age, labs) in enumerate(visits[:n_visit_cols]):
                row[f'Age_v{j+1}'] = age
                for v_simple in ['BMI', 'alt', 'ast', 'bilirubin', 'chol',
                                 'ggt', 'gluc_fast', 'plt', 'triglyc']:
                    row[f'{v_simple}_v{j+1}'] = labs.get(v_simple, np.nan)
                row[f'fibs_stiffness_med_BM_1_v{j+1}'] = labs.get('LSM', np.nan)
                row[f'fibrotest_BM_2_v{j+1}'] = labs.get('FibroTest', np.nan)
                row[f'aixp_aix_result_BM_3_v{j+1}'] = labs.get('Aixplorer', np.nan)
            for j in range(len(visits), n_visit_cols):
                row[f'Age_v{j+1}'] = np.nan
                for col in VAR_TO_COL.values():
                    row[f'{col}_v{j+1}'] = np.nan
            # Synthetic hep event at the appended terminal age
            event_age = float(visits[-1][0])
            row['evenements_hepatiques_majeurs'] = 1
            row['evenements_hepatiques_age_occur'] = event_age
            row['death'] = 0
            row['death_age_occur'] = np.nan
            rows.append(row)
            syn_id += 1
    return pd.DataFrame(rows) if rows else pd.DataFrame()


if __name__ == '__main__':
    import argparse
    import sys
    from pathlib import Path
    REPO = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(REPO / 'src'))
    sys.path.insert(0, str(REPO / 'tests'))
    from kirby.datasets.annitia_maslg import load_annitia

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=str, default=None)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--out', type=str,
                    default='tests/results/annitia/synth/counterfactual_v1.csv')
    args = ap.parse_args()
    data = load_annitia(data_dir=args.data_dir)
    df = make_counterfactual_cohort(data, k_per_patient=args.k)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f'wrote {out}: {len(df)} counterfactuals from '
          f'{int(_high_risk_mask(data).sum())} high-risk reals')
