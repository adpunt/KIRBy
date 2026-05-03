"""ANNITIA MASLD loader (Trustii challenge, May 2026).

Wide-format CSV: 1 row per patient, ~287 columns.
- Static: patient_id_anon, gender, T2DM, Hypertension, Dyslipidaemia,
  bariatric_surgery, bariatric_surgery_age
- Longitudinal: 13 variables x 21-22 visit columns (`<var>_v1`..`<var>_v22`).
  Variables: Age, BMI, alt, ast, bilirubin, chol, ggt, gluc_fast, plt, triglyc,
  aixp_aix_result_BM_3 (Aixplorer), fibrotest_BM_2 (FibroTest),
  fibs_stiffness_med_BM_1 (FibroScan).
  Visits are heavily missing (~5 actual observations per patient).
- Outcomes: evenements_hepatiques_majeurs (0/1), evenements_hepatiques_age_occur
  (age at event, blank if censored); death (0/1/blank), death_age_occur.

Time axis: patient AGE. Survival duration is computed as
`age_at_event_or_censor - age_v1` (study entry = first visit).

Missing-death handling: patients with blank `death` are dropped from the death
endpoint only. They remain in the hepatic endpoint.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


LONGITUDINAL_VARS = [
    'Age', 'BMI', 'alt', 'ast', 'bilirubin', 'chol', 'ggt', 'gluc_fast',
    'plt', 'triglyc',
    'aixp_aix_result_BM_3', 'fibrotest_BM_2', 'fibs_stiffness_med_BM_1',
]
NIT_VARS = ['aixp_aix_result_BM_3', 'fibrotest_BM_2', 'fibs_stiffness_med_BM_1']
LAB_VARS = ['alt', 'ast', 'bilirubin', 'chol', 'ggt', 'gluc_fast', 'plt', 'triglyc']
STATIC_NUMERIC = ['gender', 'T2DM', 'Hypertension', 'Dyslipidaemia',
                  'bariatric_surgery', 'bariatric_surgery_age']

CANDIDATE_PATHS = (
    '/Volumes/seagate/kirby/annitia',
    '/data/stat-cadd/scat9264/KIRBy/data/annitia',
    'data/annitia',
)


def _resolve_dir(data_dir: Optional[str] = None) -> str:
    if data_dir and os.path.isdir(data_dir):
        return data_dir
    env = os.environ.get('ANNITIA_DATA_DIR')
    if env and os.path.isdir(env):
        return env
    for p in CANDIDATE_PATHS:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        'ANNITIA data directory not found. Set ANNITIA_DATA_DIR or place '
        f'train.csv in one of: {CANDIDATE_PATHS}'
    )


def _visit_columns(df: pd.DataFrame, var: str) -> List[str]:
    return sorted(
        (c for c in df.columns if c.startswith(f'{var}_v')),
        key=lambda c: int(c.split('_v')[-1]),
    )


def load_annitia(data_dir: Optional[str] = None,
                 filename: str = 'train.csv',
                 has_outcomes: bool = True) -> Dict:
    """Parse the ANNITIA CSV into per-patient records.

    Returns a dict with keys:
      - 'patient_ids': list[str], length n_patients
      - 'static': dict[str, np.ndarray] static numeric features
      - 'longitudinal': dict[var -> list[list[(age, value)]]] per-patient series
      - 'baseline_age': np.ndarray of age at first visit (study entry)
      - 'last_followup_age': np.ndarray of age at last observed visit
      - 'hepatic': dict {event: int[n], time: float[n], y_struct: structured}
      - 'death':   dict {event: int[n], time: float[n], y_struct: structured,
                         valid_mask: bool[n]} — valid_mask is False where the
                         death column was blank (status unknown).
      - 'trustii_id': np.ndarray if a `trustii-id` column is present, else None.

    `time` for an event is `age_event - baseline_age`; for censored patients
    it is `last_followup_age - baseline_age`.

    When `has_outcomes=False` (validation CSV), outcome dicts are None and the
    loader tolerates missing outcome columns.
    """
    path = os.path.join(_resolve_dir(data_dir), filename)
    df = pd.read_csv(path)

    n = len(df)
    patient_ids = df['patient_id_anon'].astype(str).tolist()

    # Static features
    static = {}
    for col in STATIC_NUMERIC:
        if col in df.columns:
            static[col] = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=np.float64)
        else:
            static[col] = np.full(n, np.nan)

    # Longitudinal: per-variable list of [(age, value), ...] per patient
    age_cols = _visit_columns(df, 'Age')
    age_arr = df[age_cols].to_numpy(dtype=np.float64)  # (n, n_visits_age)

    longitudinal: Dict[str, List[List[Tuple[float, float]]]] = {}
    for var in LONGITUDINAL_VARS:
        if var == 'Age':
            continue
        v_cols = _visit_columns(df, var)
        if not v_cols:
            longitudinal[var] = [[] for _ in range(n)]
            continue
        v_arr = df[v_cols].to_numpy(dtype=np.float64)  # (n, n_visits_var)
        # Align var visits to ages by visit index. v1..v21 maps to Age_v1..Age_v21.
        var_records: List[List[Tuple[float, float]]] = []
        n_visits = v_arr.shape[1]
        for i in range(n):
            recs: List[Tuple[float, float]] = []
            for j in range(n_visits):
                v = v_arr[i, j]
                a = age_arr[i, j] if j < age_arr.shape[1] else np.nan
                if np.isfinite(v) and np.isfinite(a):
                    recs.append((float(a), float(v)))
            var_records.append(recs)
        longitudinal[var] = var_records

    # Baseline + last-followup age (used for time-to-event)
    baseline_age = np.full(n, np.nan)
    last_followup_age = np.full(n, np.nan)
    for i in range(n):
        finite = age_arr[i][np.isfinite(age_arr[i])]
        if finite.size:
            baseline_age[i] = float(finite[0])
            last_followup_age[i] = float(finite[-1])

    out = {
        'patient_ids': patient_ids,
        'static': static,
        'longitudinal': longitudinal,
        'baseline_age': baseline_age,
        'last_followup_age': last_followup_age,
        'hepatic': None,
        'death': None,
        'trustii_id': (df['trustii_id'].to_numpy() if 'trustii_id' in df.columns
                       else (df['trustii-id'].to_numpy() if 'trustii-id' in df.columns else None)),
    }

    if has_outcomes and 'evenements_hepatiques_majeurs' in df.columns:
        hep_event = pd.to_numeric(df['evenements_hepatiques_majeurs'], errors='coerce').fillna(0).astype(int).to_numpy()
        hep_age = pd.to_numeric(df.get('evenements_hepatiques_age_occur'), errors='coerce').to_numpy(dtype=np.float64)
        # Time = age at event (if event) else last_followup_age, minus baseline_age.
        hep_time_age = np.where(hep_event == 1, hep_age, last_followup_age)
        hep_duration = hep_time_age - baseline_age
        # Drop patients with non-positive duration (event same day as baseline or missing followup)
        hep_valid = np.isfinite(hep_duration) & (hep_duration > 0)
        out['hepatic'] = {
            'event': hep_event,
            'time': hep_duration,
            'valid_mask': hep_valid,
            'y_struct': _make_structured(hep_event, hep_duration, hep_valid),
        }

    if has_outcomes and 'death' in df.columns:
        death_raw = df['death']
        valid_death = death_raw.notna().to_numpy()
        death_event = pd.to_numeric(death_raw, errors='coerce').fillna(0).astype(int).to_numpy()
        death_age = pd.to_numeric(df.get('death_age_occur'), errors='coerce').to_numpy(dtype=np.float64)
        death_time_age = np.where(death_event == 1, death_age, last_followup_age)
        death_duration = death_time_age - baseline_age
        death_valid = valid_death & np.isfinite(death_duration) & (death_duration > 0)
        out['death'] = {
            'event': death_event,
            'time': death_duration,
            'valid_mask': death_valid,
            'y_struct': _make_structured(death_event, death_duration, death_valid),
        }

    return out


def _make_structured(event: np.ndarray, time: np.ndarray,
                     valid_mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Build sksurv structured array. None entries for invalid patients."""
    n = len(event)
    arr = np.zeros(n, dtype=[('event', '?'), ('time', '<f8')])
    arr['event'] = event.astype(bool)
    arr['time'] = time
    if valid_mask is not None:
        # Mark invalid rows with NaN time so downstream code can mask them.
        arr['time'] = np.where(valid_mask, time, np.nan)
    return arr


def get_cv_folds(data: Dict, endpoint: str = 'hepatic',
                 n_folds: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """5-fold StratifiedKFold on the event flag, restricted to valid patients.

    Returns list of (train_idx, val_idx) referring to indices into the FULL
    patient array (length n). Caller is responsible for subsetting features.
    Uses StratifiedKFold (not GroupKFold) because each patient is a unique unit
    — there is no grouping variable beyond patient_id, which is already 1:1.
    """
    from sklearn.model_selection import StratifiedKFold

    ep = data[endpoint]
    if ep is None:
        raise ValueError(f"No {endpoint} outcomes in data")
    valid = np.where(ep['valid_mask'])[0]
    events = ep['event'][valid]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_local, val_local in skf.split(valid, events):
        folds.append((valid[train_local], valid[val_local]))
    return folds


def summarize(data: Dict) -> None:
    """Print a quick sanity summary of a loaded dataset."""
    n = len(data['patient_ids'])
    print(f"ANNITIA: {n} patients")
    age_v1 = data['baseline_age']
    print(f"  baseline_age: median={np.nanmedian(age_v1):.1f}, "
          f"range=[{np.nanmin(age_v1):.0f},{np.nanmax(age_v1):.0f}]")
    fu = data['last_followup_age'] - data['baseline_age']
    print(f"  followup_years: median={np.nanmedian(fu):.1f}, "
          f"max={np.nanmax(fu):.1f}")
    for var in LONGITUDINAL_VARS:
        if var == 'Age':
            continue
        counts = [len(s) for s in data['longitudinal'][var]]
        print(f"  {var}: median {np.median(counts):.0f} obs/pt "
              f"(range {min(counts)}-{max(counts)})")
    for ep in ('hepatic', 'death'):
        if data[ep] is None:
            continue
        e = data[ep]
        n_valid = int(e['valid_mask'].sum())
        n_event = int((e['event'] & e['valid_mask']).sum())
        print(f"  {ep}: {n_valid} valid patients, {n_event} events "
              f"({100 * n_event / max(n_valid, 1):.1f}%)")
