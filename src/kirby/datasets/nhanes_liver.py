"""NHANES liver-marker + linked-mortality loader for ANNITIA pretraining.

Public, open-access:
    - NHANES III (1988-1994):    https://wwwn.cdc.gov/nchs/nhanes/nhanes3/datafiles.aspx
    - NHANES Continuous (1999+): https://wwwn.cdc.gov/nchs/nhanes/Default.aspx
    - Linked NDI mortality:      https://www.cdc.gov/nchs/data-linkage/mortality.htm

We pull a curated subset of variables that overlap with ANNITIA's longitudinal
schema, plus death follow-up time + cause flag. The output is a pretraining
table the MLP encoder consumes.

Variables harmonised (target schema = ANNITIA static + last-visit snapshot):
    age, sex, BMI, T2DM, hypertension, dyslipidaemia,
    AST, ALT, GGT, bilirubin, albumin, INR, platelets,
    glucose_fasting, cholesterol_total, triglycerides

NIT scores (FIB-4, APRI, AAR, NFS-approx) are computed from the harmonised
labs and exposed alongside.

Outcomes:
    death_event       — 1 if dead by NDI follow-up cutoff, else 0
    death_time_years  — years from interview to death (or censor)
    liver_death       — 1 if cause-of-death is liver-related (ICD-10 K70-K77),
                        else 0 (NDI 2019 file)

Access pattern:
    The first call downloads the SAS XPT files from CDC and caches them at
    $NHANES_DATA_DIR or /Volumes/seagate/kirby/nhanes/. Cohort assembly is
    deterministic; results pickled to <cache_dir>/cohort.pkl on first build.

Caveats baked in:
    - NHANES adults only (age ≥ 20).
    - Cycles selected for lab availability: NHANES III + cycles 1999-2018.
      Newer cycles (2019+) released their mortality linkage in 2024-2026
      depending on the file; we use the 2019-cutoff vintage for stability.
    - Liver-related deaths are sparse (~1-2% of all deaths). We expose the
      `liver_death` flag but pretraining defaults to all-cause `death_event`
      because the volume is what helps.
    - Selection bias is real: NHANES is a community sample, not a
      MASLD clinic cohort. Recommended downstream usage is to pretrain on
      the FIB-4 > 1.3 stratum (set `restrict='fib4_high'`) so the encoder
      sees a metabolically-loaded population closer to ANNITIA.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np


_DATA_ROOT = os.environ.get(
    'KIRBY_DATA_ROOT', os.path.expanduser('~/.cache/kirby/data'))

_DEFAULT_DIRS = (
    '/Volumes/seagate/kirby/nhanes',
    os.path.join(_DATA_ROOT, 'nhanes'),
    'data/nhanes',
)


def _resolve_dir(data_dir: Optional[str] = None) -> Path:
    if data_dir:
        p = Path(data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    env = os.environ.get('NHANES_DATA_DIR')
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    for cand in _DEFAULT_DIRS:
        if os.path.isdir(cand):
            return Path(cand)
    # Fall back to first writable candidate
    p = Path(_DEFAULT_DIRS[0])
    p.mkdir(parents=True, exist_ok=True)
    return p


# CDC URLs. Stable: NCHS keeps these public. We document a small but
# representative subset; a fuller pull would walk every cycle.
_NHANES3_URLS = {
    # Adult (≥17) lab + exam from NHANES III, ages-17plus dataset.
    'lab2':  'https://wwwn.cdc.gov/nchs/data/nhanes3/1a/lab.dat',
    'exam':  'https://wwwn.cdc.gov/nchs/data/nhanes3/1a/exam.dat',
    'adult': 'https://wwwn.cdc.gov/nchs/data/nhanes3/1a/adult.dat',
    # Mortality linkage (2019 vintage)
    'mort':  'https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/'
             'linked_mortality/NHANES_III_MORT_2019_PUBLIC.dat',
}

# Continuous NHANES (1999-): we pull DEMO + BMX + BIOPRO + GLU + HDL + TCHOL
# for a representative pair of cycles. Adding more cycles is straightforward.
_NHANES_CONT_CYCLES = ('1999-2000', '2003-2004', '2007-2008', '2013-2014', '2017-2018')
# CDC moved the continuous-NHANES file tree in 2025: the legacy URL
# `wwwn.cdc.gov/Nchs/Nhanes/{cycle}/{file}.XPT` now returns the landing
# HTML page. The new canonical location is
# `wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year_start}/Datafiles/{file}.xpt`
# (lowercase `.xpt`, year_start = first year of the cycle, e.g. 1999).
_NHANES_CONT_URLBASE = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/'
_NHANES_CONT_FILES_PER_CYCLE = ('DEMO', 'BMX', 'BIOPRO', 'CBC', 'GLU', 'TCHOL')


def _download(url: str, dest: Path, expect_magic: Optional[bytes] = None):
    """Download `url` to `dest`. If `expect_magic` is set and the cached
    file does NOT start with those bytes, the cache is invalidated and
    re-downloaded once. This guards against stale cache poisoning by an
    earlier broken URL that returned HTML with HTTP 200."""
    def _looks_ok(p: Path) -> bool:
        if not p.exists() or p.stat().st_size == 0:
            return False
        if expect_magic is None:
            return True
        try:
            with open(p, 'rb') as fh:
                return fh.read(len(expect_magic)) == expect_magic
        except OSError:
            return False
    if _looks_ok(dest):
        return dest
    # Either missing or stale-cached.
    if dest.exists():
        dest.unlink(missing_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f'  [nhanes] downloading {url}')
    urlretrieve(url, dest)
    if expect_magic is not None and not _looks_ok(dest):
        # Newly-downloaded file doesn't match expected magic — delete so the
        # next run can retry, and surface a clear error.
        dest.unlink(missing_ok=True)
        raise RuntimeError(f'downloaded {url} but content does not match '
                           f'magic {expect_magic!r} — likely the URL has '
                           f'changed again or this resource is unavailable.')
    return dest


def _read_xpt(path: Path):
    """Read SAS XPT file via pandas (which handles xpt natively)."""
    import pandas as pd
    return pd.read_sas(path, format='xport')


# ----------------------------------------------------------------------
# Continuous NHANES: per-cycle download + assembly.
# ----------------------------------------------------------------------
def _continuous_cycle(cycle: str, data_dir: Path):
    """Download + load one continuous-NHANES cycle. Returns merged dataframe."""
    import pandas as pd
    suffix = {
        '1999-2000': '',
        '2001-2002': '_B',
        '2003-2004': '_C',
        '2005-2006': '_D',
        '2007-2008': '_E',
        '2009-2010': '_F',
        '2011-2012': '_G',
        '2013-2014': '_H',
        '2015-2016': '_I',
        '2017-2018': '_J',
    }[cycle]
    # Older cycles (1999-2006) used historical filenames for the biochem,
    # glucose and total-cholesterol files (LAB18 / LAB10AM / LAB13 etc).
    # Standard `BIOPRO{suffix}` / `GLU{suffix}` / `TCHOL{suffix}` only
    # exist from 2007 (suffix `_E`) onward. Provide per-cycle overrides
    # so older cycles still resolve.
    legacy_overrides = {
        '1999-2000': {'BIOPRO': 'LAB18', 'CBC': 'LAB25',
                      'GLU': 'LAB10AM', 'TCHOL': 'LAB13'},
        '2001-2002': {'BIOPRO': 'L40_B', 'CBC': 'L25_B',
                      'GLU': 'L10AM_B', 'TCHOL': 'L13_B'},
        '2003-2004': {'BIOPRO': 'L40_C', 'CBC': 'L25_C',
                      'GLU': 'L10AM_C', 'TCHOL': 'L13_C'},
        '2005-2006': {'BIOPRO': 'BIOPRO_D', 'CBC': 'CBC_D',
                      'GLU': 'GLU_D', 'TCHOL': 'TCHOL_D'},
    }
    year_start = cycle.split('-')[0]
    dfs = {}
    for tag in _NHANES_CONT_FILES_PER_CYCLE:
        # Try the standard CDC URL layout first; fall back to legacy
        # filenames for the older cycles where the names differ.
        candidate_basenames = [f'{tag}{suffix}']
        if cycle in legacy_overrides and tag in legacy_overrides[cycle]:
            candidate_basenames.insert(0, legacy_overrides[cycle][tag])
        last_err: Exception | None = None
        loaded = None
        for base in candidate_basenames:
            fname = f'{base}.xpt'
            url = f'{_NHANES_CONT_URLBASE}{year_start}/Datafiles/{fname}'
            local = data_dir / 'continuous' / cycle / fname
            try:
                _download(url, local, expect_magic=b'HEADER RECORD')
                loaded = _read_xpt(local)
                break
            except Exception as e:
                last_err = e
                continue
        if loaded is None:
            print(f'  [nhanes] {cycle}/{tag} unavailable: {last_err}')
            dfs[tag] = None
        else:
            dfs[tag] = loaded
    demo = dfs.get('DEMO')
    if demo is None:
        return None
    out = demo[['SEQN', 'RIDAGEYR', 'RIAGENDR', 'WTMEC2YR', 'RIDRETH1']].copy()
    out['cycle'] = cycle
    if dfs.get('BMX') is not None:
        bmx = dfs['BMX'][['SEQN', 'BMXBMI']]
        out = out.merge(bmx, on='SEQN', how='left')
    else:
        out['BMXBMI'] = np.nan
    if dfs.get('BIOPRO') is not None:
        bp = dfs['BIOPRO']
        keep = ['SEQN']
        # Note: platelets are NOT in the biochem profile — they live in the
        # CBC file under LBXPLTSI (handled below). LBXSGB doesn't exist
        # either in standard NHANES (it was a typo in earlier code paths).
        for col in ['LBXSAL', 'LBXSASSI', 'LBXSATSI', 'LBXSGTSI', 'LBXSTB']:
            if col in bp.columns:
                keep.append(col)
        out = out.merge(bp[keep], on='SEQN', how='left')
    if dfs.get('CBC') is not None:
        cbc = dfs['CBC']
        keep = ['SEQN']
        for col in ['LBXPLTSI']:
            if col in cbc.columns:
                keep.append(col)
        out = out.merge(cbc[keep], on='SEQN', how='left')
    if dfs.get('GLU') is not None and 'LBXGLU' in dfs['GLU'].columns:
        out = out.merge(dfs['GLU'][['SEQN', 'LBXGLU']], on='SEQN', how='left')
    if dfs.get('TCHOL') is not None and 'LBXTC' in dfs['TCHOL'].columns:
        out = out.merge(dfs['TCHOL'][['SEQN', 'LBXTC']], on='SEQN', how='left')

    rename = {
        'RIDAGEYR': 'age', 'RIAGENDR': 'sex_raw', 'BMXBMI': 'bmi',
        'LBXSAL': 'albumin', 'LBXSASSI': 'ast', 'LBXSATSI': 'alt',
        'LBXSGTSI': 'ggt', 'LBXSTB': 'bilirubin', 'LBXPLTSI': 'platelets',
        'LBXGLU': 'glucose', 'LBXTC': 'cholesterol',
    }
    out = out.rename(columns=rename)
    # Sex: 1=male, 2=female → map to 0/1 to match ANNITIA's gender encoding.
    out['sex'] = (out['sex_raw'] == 2).astype(np.float32)
    out['source'] = f'continuous_{cycle}'
    return out


def _continuous_mortality(data_dir: Path):
    """Continuous-NHANES linked mortality (NDI 2019 vintage)."""
    import pandas as pd
    cycles = _NHANES_CONT_CYCLES
    rows = []
    for cycle in cycles:
        # The mortality file is per-cycle, named NHANES_<years>_MORT_2019_PUBLIC.dat
        cyc_label = cycle.replace('-', '_')
        # Some vintages use different label formats; try a couple.
        candidates = [
            f'NHANES_{cyc_label}_MORT_2019_PUBLIC.dat',
            f'NHANES_{cycle}_MORT_2019_PUBLIC.dat',
        ]
        loaded = None
        for name in candidates:
            url = ('https://ftp.cdc.gov/pub/Health_Statistics/NCHS/'
                   f'datalinkage/linked_mortality/{name}')
            local = data_dir / 'mortality' / name
            try:
                _download(url, local)
                loaded = local
                break
            except Exception:
                continue
        if loaded is None:
            print(f'  [nhanes] mortality file for {cycle} unavailable; skipping')
            continue
        cols = _parse_ndi_mortality(loaded)
        cols['cycle'] = cycle
        rows.append(cols)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def _parse_ndi_mortality(path: Path):
    """NDI fixed-width format. Field map per the NCHS data dictionary."""
    import pandas as pd
    # Layout for the 2019 public-use mortality file (NHANES variant).
    # Documented at: https://www.cdc.gov/nchs/data-linkage/mortality-public.htm
    colspecs = [
        (0, 6),     # SEQN
        (14, 15),   # ELIGSTAT
        (15, 16),   # MORTSTAT
        (16, 19),   # UCOD_LEADING (3-char)
        (19, 20),   # DIABETES
        (20, 21),   # HYPERTEN
        (43, 46),   # PERMTH_INT (months from interview to death)
    ]
    names = ['SEQN', 'ELIGSTAT', 'MORTSTAT', 'UCOD',
             'DIABETES_FLAG', 'HYPERTEN_FLAG', 'PERMTH_INT']
    df = pd.read_fwf(path, colspecs=colspecs, names=names, dtype=str)
    df['SEQN'] = pd.to_numeric(df['SEQN'], errors='coerce')
    df['MORTSTAT'] = pd.to_numeric(df['MORTSTAT'], errors='coerce')
    df['PERMTH_INT'] = pd.to_numeric(df['PERMTH_INT'], errors='coerce')
    df['ELIGSTAT'] = pd.to_numeric(df['ELIGSTAT'], errors='coerce')
    return df


# ----------------------------------------------------------------------
# Cohort assembly.
# ----------------------------------------------------------------------
def _assemble_cohort(data_dir: Path,
                     restrict: Optional[str] = None,
                     cycle_range: Optional[Tuple[int, int]] = None,
                     t2dm_only: bool = False,
                     exclude_viral_hep: bool = False) -> Dict:
    """Build the merged feature + outcome table.

    Returns a dict with:
        X         (n, d) np.float32 — harmonised features
        feature_names: list[str], length d
        time      (n,) np.float32   — years from interview to death/censor
        event     (n,) bool         — death by follow-up
        liver_death (n,) bool
        sex       (n,) int           — 0=male 1=female (kept separate for stratification)
        meta      dict (n_cycles, n_kept, restrict, ...)
    """
    import pandas as pd
    cycles_to_load = list(_NHANES_CONT_CYCLES)
    if cycle_range is not None:
        lo, hi = int(cycle_range[0]), int(cycle_range[1])
        cycles_to_load = [c for c in cycles_to_load
                          if int(c.split('-')[0]) >= lo and int(c.split('-')[1]) <= hi]
        if not cycles_to_load:
            raise ValueError(f'cycle_range {cycle_range} matched no NHANES cycles')
    cont_dfs = []
    for cycle in cycles_to_load:
        df = _continuous_cycle(cycle, data_dir)
        if df is not None:
            cont_dfs.append(df)
    if not cont_dfs:
        raise RuntimeError('No NHANES continuous cycles loaded.')
    feat = pd.concat(cont_dfs, ignore_index=True)

    mort = _continuous_mortality(data_dir)
    if mort is None:
        raise RuntimeError('NHANES mortality linkage could not be loaded.')

    # Merge on SEQN. SEQN is unique within cycle but not across cycles —
    # both feature and mortality tables stamp the cycle so the join is safe.
    merged = feat.merge(mort, on=['SEQN', 'cycle'], how='left')

    # Eligible respondents only (ELIGSTAT=1)
    merged = merged[(merged['ELIGSTAT'] == 1) | merged['ELIGSTAT'].isna()].copy()

    # Adults only
    merged = merged[merged['age'] >= 20].copy()

    # Outcomes
    merged['event'] = (merged['MORTSTAT'] == 1).astype(bool)
    # Censor/event time: PERMTH_INT is months. Map to years; censored fills
    # with the cycle-end-to-2019 horizon as an approximation when missing.
    merged['time'] = merged['PERMTH_INT'].astype(np.float32) / 12.0
    cycle_end = merged['cycle'].map(lambda c: int(c.split('-')[1]))
    censor_horizon = (2019.0 - cycle_end).astype(np.float32)
    merged['time'] = merged['time'].fillna(censor_horizon)
    # Liver-related death: ICD-10 K70-K77 code prefix
    def _is_liver(code):
        if not isinstance(code, str):
            return False
        c = code.strip().upper()
        return c.startswith(('K70', 'K71', 'K72', 'K73', 'K74', 'K75', 'K76', 'K77'))
    merged['liver_death'] = merged['UCOD'].apply(_is_liver) & merged['event']

    # Restrict
    if restrict == 'fib4_high':
        # FIB-4 = (age × AST) / (platelets × √ALT). Compute, then keep rows >1.3.
        # Bail clearly if the biomarkers required are missing entirely
        # (e.g. cycle range only covers cycles whose BIOPRO XPT 404'd).
        required = ['ast', 'alt', 'platelets']
        missing = [c for c in required if c not in merged.columns]
        if missing:
            raise RuntimeError(
                f'fib4_high restriction requires {missing} but those columns '
                f'are not present — likely all cycles in this range are '
                f'missing biomarker XPT files.')
        # Drop rows where any FIB-4 input is NaN before computing.
        merged = merged.dropna(subset=required).copy()
        if merged.empty:
            raise RuntimeError(
                'fib4_high: 0 rows have non-NaN ast/alt/platelets in this '
                'cycle range.')
        ast = merged['ast'].astype(np.float32)
        alt = merged['alt'].astype(np.float32)
        plt_ = merged['platelets'].astype(np.float32)  # ×10^9/L
        age = merged['age'].astype(np.float32)
        fib4 = (age * ast) / (plt_ * np.sqrt(np.maximum(alt, 1.0)))
        merged['fib4'] = fib4
        merged = merged[merged['fib4'] > 1.3].copy()
    elif restrict == 'metab_loaded':
        # Glucose ≥ 100 OR BMI ≥ 30 — metabolically loaded subset
        gl = merged['glucose'].astype(np.float32)
        bmi = merged['bmi'].astype(np.float32)
        merged = merged[(gl >= 100) | (bmi >= 30)].copy()

    if t2dm_only:
        # Self-report not available in this loader vintage; use HbA1c proxy
        # via fasting glucose ≥ 126 mg/dL OR DIABETES_FLAG (mortality file).
        gl = merged['glucose'].astype(np.float32)
        diab = pd.to_numeric(merged.get('DIABETES_FLAG'), errors='coerce')
        keep_t2dm = (gl >= 126) | (diab == 1)
        merged = merged[keep_t2dm].copy()

    if exclude_viral_hep:
        # ICD-10 B15-B19 (viral hepatitis) underlying cause of death.
        def _is_viral(code):
            if not isinstance(code, str): return False
            c = code.strip().upper()
            return c.startswith(('B15', 'B16', 'B17', 'B18', 'B19'))
        viral = merged['UCOD'].apply(_is_viral) & merged['event']
        merged = merged[~viral].copy()

    full_feat_cols = ['age', 'sex', 'bmi', 'albumin', 'ast', 'alt', 'ggt',
                      'bilirubin', 'platelets', 'glucose', 'cholesterol']
    # Keep only feature columns that actually exist (some cycle ranges
    # don't have every biomarker file). Fill the missing ones with NaN
    # so downstream feature-vector construction doesn't crash.
    feat_cols = [c for c in full_feat_cols if c in merged.columns]
    for c in full_feat_cols:
        if c not in merged.columns:
            merged[c] = np.nan
    # Engineered NITs — only computable when ast/alt/platelets are all present.
    if all(c in merged.columns for c in ('ast', 'alt', 'platelets')) \
            and not merged[['ast', 'alt', 'platelets']].isna().all().any():
        ast = merged['ast'].astype(np.float32)
        alt = merged['alt'].astype(np.float32).clip(lower=1.0)
        plt_ = merged['platelets'].astype(np.float32).clip(lower=1.0)
        age = merged['age'].astype(np.float32)
        merged['fib4'] = (age * ast) / (plt_ * np.sqrt(alt))
        merged['apri'] = (ast / 35.0) * 100.0 / plt_
        merged['aar'] = ast / alt
        feat_cols += ['fib4', 'apri', 'aar']

    X = merged[feat_cols].astype(np.float32).to_numpy()
    # Median impute
    medians = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
    nan_mask = ~np.isfinite(X)
    for j in range(X.shape[1]):
        X[nan_mask[:, j], j] = medians[j] if np.isfinite(medians[j]) else 0.0
    sample_weight = merged['WTMEC2YR'].astype(np.float32).fillna(1.0).to_numpy()

    return {
        'X': X,
        'feature_names': feat_cols,
        'time': merged['time'].astype(np.float32).to_numpy(),
        'event': merged['event'].astype(bool).to_numpy(),
        'liver_death': merged['liver_death'].astype(bool).to_numpy(),
        'sex': merged['sex'].astype(np.float32).to_numpy(),
        'sample_weight': sample_weight,
        'meta': {
            'n_kept': int(len(merged)),
            'n_events_all_cause': int(merged['event'].sum()),
            'n_events_liver': int(merged['liver_death'].sum()),
            'restrict': restrict,
            'cycles': _NHANES_CONT_CYCLES,
        },
    }


def load_nhanes_liver(data_dir: Optional[str] = None,
                      restrict: Optional[str] = None,
                      cycle_range: Optional[Tuple[int, int]] = None,
                      t2dm_only: bool = False,
                      exclude_viral_hep: bool = False,
                      use_cache: bool = True) -> Dict:
    """Top-level entry point. Loads (or builds + caches) the cohort dict.

    Args:
        data_dir: where to read/write XPT files + the cached cohort pickle.
        restrict: None | 'fib4_high' | 'metab_loaded'. See module docstring.
        cycle_range: (year_lo, year_hi) inclusive — keep only cycles whose
            window falls within. e.g. (2007, 2018) → cycles 2007-08 .. 17-18.
        t2dm_only: subset to T2DM proxy (fasting glucose ≥ 126 OR
            DIABETES_FLAG=1 from mortality file).
        exclude_viral_hep: drop patients with ICD-10 B15-B19 underlying
            cause of death (viral hepatitis) for cleaner MASLD signal.
        use_cache: skip download + assembly if <data_dir>/cohort_<key>.pkl exists.
    """
    data_dir = _resolve_dir(data_dir)
    key_parts = [restrict or 'all']
    if cycle_range is not None:
        key_parts.append(f'cyc{cycle_range[0]}-{cycle_range[1]}')
    if t2dm_only: key_parts.append('t2dm')
    if exclude_viral_hep: key_parts.append('noviral')
    cache_path = data_dir / f'cohort_{"_".join(key_parts)}.pkl'
    if use_cache and cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    cohort = _assemble_cohort(data_dir, restrict=restrict,
                              cycle_range=cycle_range,
                              t2dm_only=t2dm_only,
                              exclude_viral_hep=exclude_viral_hep)
    with open(cache_path, 'wb') as f:
        pickle.dump(cohort, f)
    return cohort


# ----------------------------------------------------------------------
# Schema bridge: project ANNITIA features onto the NHANES feature set.
# ----------------------------------------------------------------------
NHANES_FEATURE_SCHEMA = (
    'age', 'sex', 'bmi', 'albumin', 'ast', 'alt', 'ggt',
    'bilirubin', 'platelets', 'glucose', 'cholesterol',
    'fib4', 'apri', 'aar',
)


def project_annitia_to_nhanes(annitia_data: Dict) -> Tuple[np.ndarray, List[str]]:
    """Build an ANNITIA feature matrix in the NHANES schema.

    Uses the snapshot (last-visit) value for each lab where ANNITIA has it,
    plus baseline age + gender. Engineered NITs (FIB-4, APRI, AAR) computed
    from the same last-visit values.

    Returned matrix has shape (n_patients, len(NHANES_FEATURE_SCHEMA))
    using exactly the order in NHANES_FEATURE_SCHEMA — so a model trained on
    NHANES with these columns can be applied directly to ANNITIA patients.

    Missing labs imputed to per-feature population medians (computed from
    the ANNITIA data itself); a small adjustment vs NHANES medians but
    acceptable for the encoder transfer.
    """
    n = len(annitia_data['patient_ids'])
    cols = []

    age = np.asarray(annitia_data['baseline_age'], dtype=np.float32)
    sex = np.asarray(annitia_data['static']['gender'], dtype=np.float32)
    bmi = _last_value_array(annitia_data, 'BMI')
    alb = np.full(n, np.nan, dtype=np.float32)  # ANNITIA does not have albumin
    ast = _last_value_array(annitia_data, 'ast')
    alt = _last_value_array(annitia_data, 'alt')
    ggt = _last_value_array(annitia_data, 'ggt')
    bil = _last_value_array(annitia_data, 'bilirubin')
    plt_ = _last_value_array(annitia_data, 'plt')
    glu = _last_value_array(annitia_data, 'gluc_fast')
    chol = _last_value_array(annitia_data, 'chol')

    alt_safe = np.maximum(alt, 1.0)
    plt_safe = np.maximum(plt_, 1.0)
    fib4 = (age * ast) / (plt_safe * np.sqrt(alt_safe))
    apri = (ast / 35.0) * 100.0 / plt_safe
    aar = ast / alt_safe

    columns = [age, sex, bmi, alb, ast, alt, ggt, bil, plt_, glu, chol,
               fib4, apri, aar]
    X = np.column_stack([np.asarray(c, dtype=np.float32) for c in columns])
    # Per-column median impute on this same matrix
    medians = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
    nan_mask = ~np.isfinite(X)
    for j in range(X.shape[1]):
        X[nan_mask[:, j], j] = medians[j] if np.isfinite(medians[j]) else 0.0
    return X, list(NHANES_FEATURE_SCHEMA)


def _last_value_array(annitia_data: Dict, var: str) -> np.ndarray:
    n = len(annitia_data['patient_ids'])
    out = np.full(n, np.nan, dtype=np.float32)
    series_list = annitia_data['longitudinal'].get(var)
    if series_list is None:
        return out
    for i, ser in enumerate(series_list):
        if ser:
            out[i] = float(ser[-1][1])
    return out
