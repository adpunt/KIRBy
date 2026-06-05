"""M2 — Outcome-conditioned tabular generator (Norcliffe 2024 spirit).

synthcity / sdv / ctgan are all broken on the local Python (torch<2.4 RMSNorm).
We implement the practical fallback: a class-conditional Gaussian-mixture model
where the conditioning variable is (event_flag, time_quartile). At sample
time we draw heavily from the (event=1) class to oversample hazard-positive
trajectories.

Key Norcliffe insight (arXiv 2405.17333): conditioning on (T, δ) outperforms
conditioning on covariates for survival generators. We approximate this by
binning T and event into joint strata.

Output: ANNITIA-wide CSV (one row per synthetic patient). Visit columns are
faked from the snapshot/slope by emitting a single "v1" visit at age = sampled
baseline_age and an extrapolated visit at +5 yr if event_flag=1. This keeps
the synthetic patients ingestable by `load_annitia` even though they are
single-time-point under the hood — sufficient for class-balance pretraining
on the snapshot view.

For richer multi-visit synthetic, use M1 (mechanistic simulator) or M5
(counterfactual perturbation).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from .masld_simulator import VAR_TO_COL, CLIP
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from masld_simulator import VAR_TO_COL, CLIP  # type: ignore


# Snapshot column order for fitting the GMM
_SNAPSHOT_VARS = ('alt', 'ast', 'bilirubin', 'chol', 'ggt', 'gluc_fast', 'plt',
                  'triglyc', 'BMI', 'fibs_stiffness_med_BM_1',
                  'fibrotest_BM_2', 'aixp_aix_result_BM_3')

_STATIC_VARS = ('gender', 'T2DM', 'Hypertension', 'Dyslipidaemia',
                'bariatric_surgery', 'baseline_age')


def _last_observed(data, idx: int, var: str) -> float:
    series = data['longitudinal'].get(var, [[]])[idx]
    if not series:
        return np.nan
    return float(series[-1][1])


def _build_snapshot_matrix(data) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, baseline_age) for each patient. X = static + last observed labs."""
    n = len(data['patient_ids'])
    cols = []
    # Static
    for v in ('gender', 'T2DM', 'Hypertension', 'Dyslipidaemia',
              'bariatric_surgery'):
        cols.append(data['static'][v])
    age = data['baseline_age']
    cols.append(age)
    # Last observed labs
    for v in _SNAPSHOT_VARS:
        col = np.array([_last_observed(data, i, v) for i in range(n)])
        cols.append(col)
    X = np.column_stack(cols)
    return X, age


def _stratify_classes(data, endpoint: str = 'hepatic') -> np.ndarray:
    """Build joint (event, time-quartile) strata."""
    ep = data[endpoint]
    if ep is None:
        return np.zeros(len(data['patient_ids']), dtype=int)
    event = ep['event'].astype(bool)
    time = ep['time']
    valid = ep['valid_mask']
    # 4 strata: event=1 OR event=0 split by time quartile
    out = np.full(len(event), 0, dtype=int)
    out[event] = 4  # all events go to a single class (rare)
    if valid.any():
        cens_times = time[~event & valid]
        if cens_times.size >= 4:
            qs = np.quantile(cens_times, [0.25, 0.5, 0.75])
            for i in range(len(out)):
                if not event[i] and valid[i]:
                    out[i] = int(np.searchsorted(qs, time[i]))
    return out


class OutcomeConditionalGMM:
    """Per-class Gaussian mixture on snapshot features."""

    def __init__(self, n_components_per_class: int = 3, seed: int = 42):
        self.k = n_components_per_class
        self.seed = seed
        self._models: Dict[int, object] = {}
        self._classes: np.ndarray = np.array([])
        self._d: int = 0

    def fit(self, X: np.ndarray, classes: np.ndarray, valid: np.ndarray):
        from sklearn.mixture import GaussianMixture
        from sklearn.impute import SimpleImputer
        Xv = X[valid].copy()
        cv = classes[valid]
        imputer = SimpleImputer(strategy='median')
        Xv = imputer.fit_transform(Xv)
        self._imputer = imputer
        self._d = Xv.shape[1]
        self._classes = np.unique(cv)
        for c in self._classes:
            sel = cv == c
            n_c = int(sel.sum())
            if n_c < 4:
                continue
            k_c = max(1, min(self.k, n_c // 6))
            gmm = GaussianMixture(n_components=k_c, covariance_type='diag',
                                  random_state=self.seed,
                                  reg_covar=1e-2, max_iter=200)
            gmm.fit(Xv[sel])
            self._models[int(c)] = gmm
        return self

    def sample(self, n_per_class: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (X_synth, classes_synth)."""
        rows = []; cls_out = []
        for c, n in n_per_class.items():
            if c not in self._models or n <= 0:
                continue
            Z, _ = self._models[c].sample(n)
            rows.append(Z)
            cls_out.extend([c] * n)
        if not rows:
            return np.zeros((0, self._d)), np.array([])
        return np.vstack(rows), np.array(cls_out, dtype=int)


def synthetic_to_dataframe(X_syn: np.ndarray, classes_syn: np.ndarray,
                           n_visit_cols: int = 21,
                           seed: int = 42) -> pd.DataFrame:
    """Pack sampled snapshots into ANNITIA wide-format rows.

    Each synthetic patient has 2 visits (v1 = baseline, v2 = baseline+5yr).
    Event class (4) gets event_flag=1 with event_age sampled in (baseline+1y,
    baseline+5y). Other classes get censored at v2.
    """
    rng = np.random.default_rng(seed)
    rows = []
    n = X_syn.shape[0]
    for i in range(n):
        v = X_syn[i]
        gender, t2dm, htn, dyslip, bari, age = v[:6]
        labs = v[6:]
        cls = int(classes_syn[i])
        row = {
            'patient_id_anon': f'M2_{i:06d}',
            'phenotype': 'm2_outcond',
            'gender': float(np.clip(round(gender), 0, 1)),
            'T2DM': float(np.clip(round(t2dm), 0, 1)),
            'Hypertension': float(np.clip(round(htn), 0, 1)),
            'Dyslipidaemia': float(np.clip(round(dyslip), 0, 1)),
            'bariatric_surgery': float(np.clip(round(bari), 0, 1)),
            'bariatric_surgery_age': np.nan,
        }
        baseline_age = float(np.clip(age, 25, 85))
        # Two visits at v1 and v2 (5y apart)
        for j, dt in enumerate([0.0, 5.0]):
            row[f'Age_v{j+1}'] = baseline_age + dt
            for var_idx, var in enumerate(_SNAPSHOT_VARS):
                col = VAR_TO_COL[var] if var in VAR_TO_COL else var
                lab_val = float(labs[var_idx])
                # Apply clip
                if var in CLIP:
                    lo, hi = CLIP[var]
                    lab_val = float(np.clip(lab_val, lo, hi))
                # Inject mild drift on v2 if event-class
                if j == 1:
                    if var in ('fibs_stiffness_med_BM_1', 'aixp_aix_result_BM_3'):
                        delta = (1.5 if cls == 4 else 0.2) + rng.normal(0, 0.5)
                        lab_val = float(np.clip(lab_val + delta * 5, *CLIP.get(var, (-1e9, 1e9))))
                    elif var == 'fibrotest_BM_2':
                        delta = (0.05 if cls == 4 else 0.0) + rng.normal(0, 0.02)
                        lab_val = float(np.clip(lab_val + delta * 5, 0, 1))
                    elif var == 'plt':
                        delta = (-5.0 if cls == 4 else -1.0) + rng.normal(0, 3)
                        lab_val = float(np.clip(lab_val + delta * 5, 30, 600))
                row[f'{col}_v{j+1}'] = lab_val
        for j in range(2, n_visit_cols):
            row[f'Age_v{j+1}'] = np.nan
            for col in VAR_TO_COL.values():
                row[f'{col}_v{j+1}'] = np.nan
        # Outcome
        if cls == 4:
            evt_age = baseline_age + float(rng.uniform(1.5, 5.0))
            row['evenements_hepatiques_majeurs'] = 1
            row['evenements_hepatiques_age_occur'] = evt_age
        else:
            row['evenements_hepatiques_majeurs'] = 0
            row['evenements_hepatiques_age_occur'] = np.nan
        row['death'] = 0
        row['death_age_occur'] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_outcome_conditional_cohort(data: Dict, n_event: int = 500,
                                    n_censored: int = 1500,
                                    endpoint: str = 'hepatic',
                                    seed: int = 42,
                                    n_components_per_class: int = 3) -> pd.DataFrame:
    """Fit GMM on real, sample n_event event-class + n_censored censored-class."""
    X, _ = _build_snapshot_matrix(data)
    classes = _stratify_classes(data, endpoint=endpoint)
    valid = data[endpoint]['valid_mask'] if data.get(endpoint) else np.ones(len(X), bool)
    # Drop rows with all-NaN (no observed labs)
    keep = valid & np.isfinite(X[:, 5])  # baseline_age finite
    gmm = OutcomeConditionalGMM(n_components_per_class=n_components_per_class,
                                seed=seed).fit(X, classes, keep)
    # Per-class targets
    n_per = {}
    if 4 in gmm._models:
        n_per[4] = n_event
    n_cens_per = max(1, n_censored // max(1, len(gmm._classes) - 1))
    for c in gmm._classes:
        if c != 4:
            n_per[int(c)] = n_cens_per
    Xs, cs = gmm.sample(n_per)
    return synthetic_to_dataframe(Xs, cs, seed=seed)


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
    ap.add_argument('--n-event', type=int, default=500)
    ap.add_argument('--n-censored', type=int, default=1500)
    ap.add_argument('--out', type=str,
                    default='tests/results/annitia/synth/outcome_cond_v1.csv')
    args = ap.parse_args()
    data = load_annitia(data_dir=args.data_dir)
    df = make_outcome_conditional_cohort(data, n_event=args.n_event,
                                          n_censored=args.n_censored)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    n_evt = int(df['evenements_hepatiques_majeurs'].sum())
    print(f'wrote {out}: {len(df)} synth, {n_evt} events ({n_evt/len(df):.1%})')
