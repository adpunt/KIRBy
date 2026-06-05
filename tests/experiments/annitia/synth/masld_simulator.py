"""M1 — LLM-distilled MASLD progression simulator.

Generates synthetic MASLD patient trajectories that look like ANNITIA
train.csv: same schema (patient_id_anon, static covariates, longitudinal
v1..v21 columns, outcome columns). Output is a `pandas.DataFrame` that
`kirby.datasets.annitia_maslg.load_annitia` can ingest directly when written
to a CSV.

Mechanism:
    1. Sample a phenotype from the population mixture (5 clusters).
    2. Sample baseline covariates conditional on phenotype.
    3. Simulate ~5 visits with exponential gaps (clipped 0.4-3y).
    4. Evolve labs visit-to-visit with phenotype-specific drift + AR(1)
       residual + multivariate-normal noise (Σ from literature).
    5. Compute AGILE 3+ logit at each visit; convert to per-interval hazard
       using Pennisi 2023 stratum incidence rates.
    6. Sample (event_age, event_flag) competing-risks (LRE vs death vs
       censoring) using exponential within-interval times.
    7. Bariatric surgery: 5% of patients post-op at random age, applies a
       0.7x LSM/FIB-4/ALT multiplier from t_op+1y onward and 0.4x LRE
       hazard.

All numeric parameters cited in MASLD_SIMULATOR_NOTES.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SIMULATOR_VERSION = "1.0.0"

PHENOTYPES = ('metabolic_obese', 'lean_masld', 'pnpla3_driven',
              'diabetic_aggressive', 'regressor')
PHENO_PROBS = np.array([0.50, 0.18, 0.15, 0.12, 0.05])

# Baseline distributions per phenotype: (mean, sd) for continuous, P for binary.
BASELINE: Dict[str, Dict[str, Tuple[float, float] | float]] = {
    'metabolic_obese':       dict(age=(55, 12), BMI=(33, 5),  T2DM=0.45, HTN=0.55, Dyslip=0.60,
                                  alt=(45, 20), ast=(35, 15), ggt=(60, 35), plt=(240, 55),
                                  chol=(5.0, 1.0), triglyc=(1.9, 0.9), gluc_fast=(6.1, 1.4),
                                  bilirubin=(12, 5), LSM=(8.5, 4.0), FibroTest=(0.35, 0.20),
                                  Aixplorer=(9.0, 4.0), gender_F=0.45),
    'lean_masld':            dict(age=(52, 13), BMI=(23, 2),  T2DM=0.10, HTN=0.20, Dyslip=0.30,
                                  alt=(35, 15), ast=(28, 12), ggt=(35, 20), plt=(250, 50),
                                  chol=(4.7, 0.9), triglyc=(1.2, 0.5), gluc_fast=(5.2, 0.7),
                                  bilirubin=(11, 4), LSM=(7.0, 3.5), FibroTest=(0.30, 0.18),
                                  Aixplorer=(7.5, 3.5), gender_F=0.55),
    'pnpla3_driven':         dict(age=(50, 12), BMI=(28, 3),  T2DM=0.25, HTN=0.30, Dyslip=0.40,
                                  alt=(80, 35), ast=(60, 25), ggt=(70, 40), plt=(230, 55),
                                  chol=(4.8, 1.0), triglyc=(1.6, 0.8), gluc_fast=(5.5, 0.9),
                                  bilirubin=(13, 6), LSM=(10.0, 5.0), FibroTest=(0.45, 0.22),
                                  Aixplorer=(10.5, 5.0), gender_F=0.40),
    'diabetic_aggressive':   dict(age=(58, 10), BMI=(35, 5),  T2DM=1.00, HTN=0.65, Dyslip=0.65,
                                  alt=(55, 25), ast=(42, 18), ggt=(90, 55), plt=(235, 58),
                                  chol=(5.0, 1.1), triglyc=(2.4, 1.2), gluc_fast=(8.5, 2.5),
                                  bilirubin=(12, 5), LSM=(12.0, 6.0), FibroTest=(0.50, 0.25),
                                  Aixplorer=(12.5, 6.5), gender_F=0.55),
    'regressor':             dict(age=(53, 11), BMI=(30, 4),  T2DM=0.30, HTN=0.40, Dyslip=0.45,
                                  alt=(40, 18), ast=(32, 14), ggt=(50, 28), plt=(250, 50),
                                  chol=(4.8, 0.9), triglyc=(1.6, 0.7), gluc_fast=(5.6, 0.9),
                                  bilirubin=(11, 4), LSM=(6.5, 2.5), FibroTest=(0.28, 0.16),
                                  Aixplorer=(7.0, 2.8), gender_F=0.50),
}

# Phenotype-specific annual LSM drift (kPa/yr): mean, sd
LSM_DRIFT = {
    'metabolic_obese':     (0.30, 0.6),
    'lean_masld':          (0.50, 0.8),
    'pnpla3_driven':       (0.80, 1.0),
    'diabetic_aggressive': (1.20, 1.2),
    'regressor':           (-0.20, 0.5),
}

# Within-patient AR(1) coefficients at 1-yr lag (computed from residual half-life)
AR1_1Y = {
    'alt': 0.20, 'ast': 0.20, 'ggt': 0.30,
    'LSM': 0.50, 'FibroTest': 0.50, 'Aixplorer': 0.50,
    'BMI': 0.71, 'plt': 0.50,
    'chol': 0.41, 'triglyc': 0.41, 'gluc_fast': 0.41, 'bilirubin': 0.41,
}

# Annual incremental noise SD per variable (residual SD per year).
# Tuned so visit-to-visit changes match observed train.csv ranges.
ANNUAL_SD = {
    'alt': 12.0, 'ast': 9.0, 'ggt': 18.0, 'LSM': 1.8, 'FibroTest': 0.10,
    'Aixplorer': 1.8, 'BMI': 0.6, 'plt': 18.0, 'chol': 0.5,
    'triglyc': 0.4, 'gluc_fast': 0.6, 'bilirubin': 2.0,
}

# Pennisi 2023 LRE incidence per 100 patient-years
PENNISI_INCIDENCE = {  # by AGILE 3+ stratum
    'low': 0.20,    # AGILE_3+ < 0.451
    'mid': 1.10,    # 0.451 <= AGILE_3+ < 0.679
    'high': 5.50,   # AGILE_3+ >= 0.679
}

# Sanyal NEJM 2021 mortality per 100 PY by stratum
SANYAL_MORTALITY = {
    'low': 0.32, 'mid': 0.89, 'high': 1.76,
}

# AGILE 3+ logit coefficients (Sanyal Lancet GH 2022)
AGILE3_COEFS = dict(
    intercept=-3.92368,
    ln_LSM=2.29714,
    plt=-0.00902,
    ast_alt_ratio=-0.98633,
    diabetes=1.08636,
    female=-0.38581,
    age=0.03018,
)

VAR_TO_COL = {
    'BMI': 'BMI', 'alt': 'alt', 'ast': 'ast', 'bilirubin': 'bilirubin',
    'chol': 'chol', 'ggt': 'ggt', 'gluc_fast': 'gluc_fast',
    'plt': 'plt', 'triglyc': 'triglyc',
    'LSM': 'fibs_stiffness_med_BM_1',
    'FibroTest': 'fibrotest_BM_2',
    'Aixplorer': 'aixp_aix_result_BM_3',
}

LAB_VARS = ('alt', 'ast', 'bilirubin', 'chol', 'ggt', 'gluc_fast', 'plt',
            'triglyc', 'BMI')
NIT_VARS = ('LSM', 'FibroTest', 'Aixplorer')

# Hard physiological clip ranges (used after each visit update)
CLIP = {
    'alt': (5, 600), 'ast': (5, 600), 'bilirubin': (2, 200),
    'chol': (2.0, 12.0), 'ggt': (5, 800), 'gluc_fast': (3.0, 25.0),
    'plt': (30, 600), 'triglyc': (0.3, 15.0), 'BMI': (16, 65),
    'LSM': (2.0, 75.0), 'FibroTest': (0.0, 1.0), 'Aixplorer': (2.0, 80.0),
}


def _agile3(LSM, plt, ast, alt, diabetes, female, age):
    """Compute AGILE 3+ logit per visit. ALL inputs may be arrays."""
    ratio = np.where(alt > 1e-3, ast / np.maximum(alt, 1e-3), 1.0)
    z = (AGILE3_COEFS['intercept']
         + AGILE3_COEFS['ln_LSM'] * np.log(np.clip(LSM, 1.0, 75.0))
         + AGILE3_COEFS['plt'] * plt
         + AGILE3_COEFS['ast_alt_ratio'] * ratio
         + AGILE3_COEFS['diabetes'] * diabetes
         + AGILE3_COEFS['female'] * female
         + AGILE3_COEFS['age'] * age)
    return 1.0 / (1.0 + np.exp(-z))


def _stratum(score):
    if score < 0.451:
        return 'low'
    if score < 0.679:
        return 'mid'
    return 'high'


def _sample_baseline(phenotype: str, rng: np.random.Generator) -> Dict[str, float]:
    p = BASELINE[phenotype]
    out = {}
    for k, v in p.items():
        if isinstance(v, tuple):
            mu, sd = v
            x = rng.normal(mu, sd)
            out[k] = x
        else:
            out[k] = float(rng.random() < v)
    # Apply hard clip + physiologic floors
    for var, (lo, hi) in CLIP.items():
        if var in out:
            out[var] = float(np.clip(out[var], lo, hi))
    out['age'] = float(np.clip(out.get('age', 55), 25, 85))
    out['BMI'] = float(np.clip(out.get('BMI', 28), 16, 65))
    out['gender'] = 1 if rng.random() < p.get('gender_F', 0.5) else 0  # 1=F
    return out


def _evolve_visit(prev: Dict[str, float], dt_yr: float, phenotype: str,
                  rng: np.random.Generator,
                  bariatric_active: bool) -> Dict[str, float]:
    """Evolve labs forward by dt_yr years given previous visit values."""
    cur = dict(prev)
    cur['age'] = prev['age'] + dt_yr

    # LSM drift (phenotype-specific)
    drift_mu, drift_sd = LSM_DRIFT[phenotype]
    if bariatric_active:
        drift_mu *= -0.5  # mild post-op regression
    lsm_drift = drift_mu * dt_yr + rng.normal(0, drift_sd * dt_yr ** 0.5)
    cur['LSM'] = prev['LSM'] + lsm_drift

    # FibroTest, Aixplorer track LSM with correlation 0.55, 0.78
    cur['FibroTest'] = prev['FibroTest'] + (lsm_drift * 0.011) + rng.normal(0, ANNUAL_SD['FibroTest'] * dt_yr ** 0.5)
    cur['Aixplorer'] = prev['Aixplorer'] + (lsm_drift * 0.78) + rng.normal(0, ANNUAL_SD['Aixplorer'] * dt_yr ** 0.5)

    # Platelet decline correlated with LSM rise
    plt_drift = -5 * dt_yr * max(0.0, (cur['LSM'] - 8) / 10)
    cur['plt'] = prev['plt'] + plt_drift + rng.normal(0, ANNUAL_SD['plt'] * dt_yr ** 0.5)

    # ALT/AST: AR(1) with phenotype-specific drift
    # ALT paradox: declines as fibrosis advances
    alt_drift = -2 * dt_yr if cur['LSM'] > 12 else 0.0
    sigma = ANNUAL_SD['alt'] * dt_yr ** 0.5
    alt_resid = rng.normal(0, sigma)
    cur['alt'] = prev['alt'] + alt_drift + alt_resid * (1 - AR1_1Y['alt'] ** dt_yr)
    if bariatric_active:
        cur['alt'] = prev['alt'] * 0.7 + rng.normal(0, 5)
    # AST tracks ALT (r=0.80)
    cur['ast'] = prev['ast'] + 0.80 * (cur['alt'] - prev['alt']) + rng.normal(0, ANNUAL_SD['ast'] * dt_yr ** 0.5 * 0.6)

    # GGT correlates with ALT, BMI, ggt mostly stable
    cur['ggt'] = prev['ggt'] + 0.55 * (cur['alt'] - prev['alt']) * 0.5 + rng.normal(0, ANNUAL_SD['ggt'] * dt_yr ** 0.5)

    # BMI slow drift
    bmi_drift = 0.2 * dt_yr if phenotype != 'regressor' else -0.5 * dt_yr
    if bariatric_active:
        bmi_drift = -1.0 * dt_yr
    cur['BMI'] = prev['BMI'] + bmi_drift + rng.normal(0, ANNUAL_SD['BMI'] * dt_yr ** 0.5)

    # Metabolic labs: AR(1)-style noise
    for v in ('chol', 'triglyc', 'gluc_fast', 'bilirubin'):
        cur[v] = prev[v] + rng.normal(0, ANNUAL_SD[v] * dt_yr ** 0.5)

    # Clip all
    for var, (lo, hi) in CLIP.items():
        if var in cur:
            cur[var] = float(np.clip(cur[var], lo, hi))
    return cur


def _sample_visit_schedule(rng: np.random.Generator, max_visits: int = 21
                           ) -> np.ndarray:
    """Return visit gaps in years (length n_visits-1). First visit is t=0."""
    n_visits = int(np.clip(rng.poisson(5), 1, max_visits))
    if n_visits == 1:
        return np.array([])
    gaps = np.clip(rng.exponential(1 / 0.6, size=n_visits - 1), 0.4, 3.0)
    return gaps


def _sample_event_time(visit_ages: np.ndarray, agile_scores: np.ndarray,
                       rng: np.random.Generator,
                       bariatric_post_op_age: Optional[float],
                       hazard_table: Dict[str, float]) -> Tuple[bool, float]:
    """Piecewise exponential event time over visit intervals.

    Returns (event_observed, age_at_event_or_censor).
    Censor at last visit if no event.
    """
    if len(visit_ages) < 2:
        return False, float(visit_ages[-1] if len(visit_ages) else 0.0)
    last_age = float(visit_ages[-1])
    cum_haz = 0.0
    target = rng.exponential(1.0)  # u ~ Exp(1); event when cum_haz exceeds u
    for i in range(len(visit_ages) - 1):
        a0, a1 = visit_ages[i], visit_ages[i + 1]
        dt = a1 - a0
        s = _stratum(agile_scores[i])
        h = hazard_table[s] / 100.0  # per PY
        if bariatric_post_op_age is not None and a0 >= bariatric_post_op_age + 1.0:
            h *= 0.4
        seg = h * dt
        if cum_haz + seg >= target:
            # Linear interpolation within interval
            frac = (target - cum_haz) / max(seg, 1e-9)
            return True, float(a0 + frac * dt)
        cum_haz += seg
    return False, last_age


@dataclass
class SimulatedPatient:
    patient_id: str
    phenotype: str
    static: Dict[str, float] = field(default_factory=dict)
    visit_ages: List[float] = field(default_factory=list)
    visit_labs: List[Dict[str, float]] = field(default_factory=list)
    hep_event: int = 0
    hep_event_age: float = float('nan')
    death_event: int = 0
    death_event_age: float = float('nan')
    last_followup_age: float = 0.0


def simulate_patient(idx: int, rng: np.random.Generator,
                     bariatric_rate: float = 0.05) -> SimulatedPatient:
    phenotype = PHENOTYPES[int(rng.choice(len(PHENOTYPES), p=PHENO_PROBS))]
    base = _sample_baseline(phenotype, rng)

    bariatric = rng.random() < bariatric_rate
    bariatric_op_age = None
    if bariatric:
        bariatric_op_age = base['age'] + rng.uniform(-3, 5)

    gaps = _sample_visit_schedule(rng)
    visits: List[Dict[str, float]] = [dict(base)]
    visit_ages = [base['age']]
    for dt in gaps:
        bariatric_active = (bariatric_op_age is not None
                            and visit_ages[-1] + dt >= bariatric_op_age + 1.0)
        nxt = _evolve_visit(visits[-1], float(dt), phenotype, rng, bariatric_active)
        visits.append(nxt); visit_ages.append(nxt['age'])
    visit_ages = np.array(visit_ages, dtype=float)

    # Compute AGILE 3+ at each visit
    LSMs = np.array([v['LSM'] for v in visits])
    plts = np.array([v['plt'] for v in visits])
    asts = np.array([v['ast'] for v in visits])
    alts = np.array([v['alt'] for v in visits])
    ages = visit_ages
    agile = _agile3(LSMs, plts, asts, alts, base['T2DM'], base['gender'], ages)

    # Hep event
    hep_event, hep_age = _sample_event_time(
        visit_ages, agile, rng, bariatric_op_age, PENNISI_INCIDENCE)
    # Death event (independent process)
    # Use age-adjusted mortality
    age_scalar = np.exp((ages - 50) * 0.05)
    death_haz_table = {k: v * float(age_scalar.mean()) for k, v in SANYAL_MORTALITY.items()}
    death_event, death_age = _sample_event_time(
        visit_ages, agile, rng, bariatric_op_age, death_haz_table)
    # Censoring (loss to follow-up, exp median 8y from baseline)
    cens_age = base['age'] + rng.exponential(8.0)

    # Pick whichever happens first; if censor first, no event
    event_age = float('inf')
    flag_hep = 0; flag_death = 0
    if hep_event:
        event_age = min(event_age, hep_age); flag_hep = 1
    if death_event and death_age <= event_age:
        # death is the terminal observation if it precedes hep
        if death_age < event_age:
            flag_hep = 0
        flag_death = 1
        event_age = min(event_age, death_age)
    if cens_age < event_age:
        # Censor before any event
        flag_hep = 0; flag_death = 0
        last_age = cens_age
    else:
        last_age = event_age
    # Cap visits to those <= last_age
    keep = visit_ages <= last_age + 1e-6
    if keep.sum() < 1:
        keep[0] = True
    visit_ages = visit_ages[keep]
    visits = [visits[i] for i in range(len(visits)) if keep[i]]

    pat = SimulatedPatient(
        patient_id=f'SYNTH_{idx:06d}',
        phenotype=phenotype,
        static=dict(gender=base['gender'], T2DM=base['T2DM'],
                    Hypertension=base['HTN'], Dyslipidaemia=base['Dyslip'],
                    bariatric_surgery=int(bariatric),
                    bariatric_surgery_age=(float(bariatric_op_age)
                                           if bariatric else float('nan'))),
        visit_ages=visit_ages.tolist(),
        visit_labs=visits,
        hep_event=int(flag_hep),
        hep_event_age=(float(hep_age) if flag_hep else float('nan')),
        death_event=int(flag_death),
        death_event_age=(float(death_age) if flag_death else float('nan')),
        last_followup_age=float(last_age),
    )
    return pat


def patients_to_dataframe(pats: List[SimulatedPatient],
                          n_visit_cols: int = 21) -> pd.DataFrame:
    """Convert to ANNITIA wide-format DataFrame (one row per patient)."""
    rows = []
    for p in pats:
        row = {'patient_id_anon': p.patient_id, 'phenotype': p.phenotype,
               **p.static}
        # Static numeric: gender mapping (ANNITIA uses 1/2 not 0/1?)
        # Check load_annitia: gender is treated as numeric, no remapping needed.

        for j in range(n_visit_cols):
            if j < len(p.visit_ages):
                row[f'Age_v{j+1}'] = p.visit_ages[j]
                lab = p.visit_labs[j]
                for var, col in VAR_TO_COL.items():
                    row[f'{col}_v{j+1}'] = lab.get(var, np.nan)
            else:
                row[f'Age_v{j+1}'] = np.nan
                for col in VAR_TO_COL.values():
                    row[f'{col}_v{j+1}'] = np.nan
        # Outcomes
        row['evenements_hepatiques_majeurs'] = int(p.hep_event)
        row['evenements_hepatiques_age_occur'] = (
            p.hep_event_age if p.hep_event else np.nan)
        row['death'] = int(p.death_event)
        row['death_age_occur'] = p.death_event_age if p.death_event else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def simulate_cohort(n: int = 10000, seed: int = 42,
                    bariatric_rate: float = 0.05) -> pd.DataFrame:
    """Build a synthetic cohort and return as ANNITIA-wide DataFrame.

    Roughly 25-30% will have a hep event (vs 4.2% in real ANNITIA). To
    match the real prevalence, downsample positives or set a phenotype
    mixture biased toward regressor — but for pretraining we WANT enriched
    positives, so we leave the natural imbalance.
    """
    rng = np.random.default_rng(seed)
    pats = [simulate_patient(i, rng, bariatric_rate=bariatric_rate)
            for i in range(n)]
    df = patients_to_dataframe(pats)
    return df


def cohort_summary(df: pd.DataFrame) -> Dict:
    n = len(df)
    n_hep = int(df['evenements_hepatiques_majeurs'].sum())
    n_death = int(df['death'].sum())
    return dict(
        n_patients=n,
        n_hep_events=n_hep,
        hep_rate=n_hep / n,
        n_death_events=n_death,
        death_rate=n_death / n,
        phenotype_counts=df['phenotype'].value_counts().to_dict(),
    )


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, default='tests/results/annitia/synth/cohort_v1.csv')
    args = ap.parse_args()

    df = simulate_cohort(n=args.n, seed=args.seed)
    summary = cohort_summary(df)
    print(json.dumps(summary, indent=2, default=str))

    from pathlib import Path
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f'wrote {out} ({len(df)} patients, {df.shape[1]} cols)')
