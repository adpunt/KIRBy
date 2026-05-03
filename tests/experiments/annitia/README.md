# ANNITIA MASLD Risk-Stratification — Method

Submission for the [Trustii ANNITIA challenge](https://app.trustii.io/datasets/1551).

## Problem

Predict, for each patient with metabolic dysfunction-associated steatotic
liver disease (MASLD), the risk of two long-term outcomes:

- a major hepatic event (composite endpoint),
- all-cause death.

The score is `0.7 * Cindex_hepatic + 0.3 * Cindex_death`, where the C-index
is the censored concordance index from `sksurv.metrics.concordance_index_censored`.

The training data is one row per patient, in wide format: 1253 patients, 287
columns (static covariates + 13 longitudinal variables × 21–22 visit slots
each + 4 outcome columns). Visits are heavily missing (~5 actual observations
per patient on average), and the time axis is patient age. Hepatic events are
rare (47 / 1253) and ~21% of patients have an unknown death status.

## Approach

Hand-crafted **feature views** of each patient's record, combined and
selected by KIRBy-style hybrid search, with sksurv survival downstreams.

### Pre-processing (`src/kirby/datasets/annitia_maslg.py`)

For every patient we produce:
- a static-covariate vector,
- a list of `(age, value)` observations per longitudinal variable (only
  rows where both the age and the value are observed),
- a per-endpoint survival tuple `(event, duration)` where
  `duration = age_at_event_or_last_visit − age_at_first_visit`.

Patients with non-positive duration (event at baseline or no follow-up) are
dropped from that endpoint. Patients with a blank `death` cell are dropped
from the death endpoint only — treating "status unknown" as "alive at last
contact" would inflate the censoring distribution.

CV folds are 5-fold StratifiedKFold on the event flag, fit only on valid
patients per endpoint.

### Feature views (`tests/experiments/annitia/views.py`)

| View          | What it captures                                          | Dim |
|---------------|-----------------------------------------------------------|----:|
| `snapshot`    | static covariates + last observed value of each lab      |  37 |
| `slope`       | linear-fit slope of value vs age, per lab                |  12 |
| `summary`     | mean / min / max / std per lab                           |  48 |
| `delta`       | last − first per lab                                     |  12 |
| `missingness` | visit count, follow-up span, time-since-last-obs per lab |  25 |
| `nit`         | snapshot + slope + delta restricted to the 3 NITs        |  18 |

Missing slots are imputed with 0; explicit missingness flags are added
where they carry signal. The `nit` view exists because the challenge
requires the model to use at least one NIT.

These are deterministic, training-free transforms — the same builder runs
on the validation CSV at submission time.

### Survival downstreams (`tests/experiments/annitia/survival.py`)

Three estimators behind a uniform `.fit(X, y_struct) / .predict_risk(X)` API:

- **`SurvivalCoxPH`** — `sksurv.linear_model.CoxPHSurvivalAnalysis` with a
  small ridge penalty (α = 1.0) and a per-fold `StandardScaler`. If the Cox
  optimisation fails to converge, α is increased by 10× and the fit is
  retried once.
- **`SurvivalCoxnet`** — `sksurv.linear_model.CoxnetSurvivalAnalysis`
  (elastic net, l1_ratio=0.9). Lifted from the challenge hello_world
  starter as the recommended sparse Cox variant for low events-per-variable
  regimes (47 hepatic events / up to 152 features in the all-views concat).
- **`SurvivalRSF`** — `sksurv.ensemble.RandomSurvivalForest` (300 trees,
  sqrt features, **min_samples_leaf=20** — also from hello_world, argued
  to reduce overfit on rare events).

All three return risk scores where higher means worse prognosis, which is
the sign convention `concordance_index_censored` expects.

### Hybrid search (`tests/experiments/annitia/search.py`)

The search enumerates every non-empty subset of the 6 views (63 subsets)
crossed with `{cox, coxnet, rsf}` (3 models), giving 189 candidates per endpoint.
Each candidate is scored by mean censored C-index over the 5 stratified
CV folds. The winning `(views, model)` pair is refit on all valid
patients and saved for inference.

This is the same search recipe as KIRBy's `find_best_hybrid` — enumerate
combinations of representations × downstream models and pick the best by
CV — adapted for survival outcomes. The "representations" here are
hand-crafted feature views rather than learned embeddings; everything else
is mechanically the same.

## Running

```bash
# Fit per-endpoint models (writes annitia_{hepatic,death}_model.pkl
# and per-condition C-index tables to tests/results/annitia/)
python tests/experiments/annitia/run.py \
    --output-dir tests/results/annitia

# Build the submission CSV from those saved models
python tests/experiments/annitia/submit.py \
    --validation-csv /path/to/val_test.csv \
    --models-dir tests/results/annitia \
    --output tests/results/annitia/annitia_submission.csv
```

A smoke test runs in <1 minute:
```bash
python tests/experiments/annitia/run.py \
    --views nit,snapshot --models rsf --n-folds 3 \
    --max-subset-size 1
```

Dependencies: `scikit-survival`, `pandas`, `numpy`, `scikit-learn`.
The training CSV is read from `$ANNITIA_DATA_DIR/train.csv` (or
`/Volumes/seagate/kirby/annitia/train.csv` by default).

## Limitations

- **Few hepatic events (47 total).** Any survival model fit to this is
  noisy; differences in C-index between candidate views are at the edge of
  the CV noise floor for the hepatic endpoint.
- **~21% of `death` labels are blank.** Dropping those patients shrinks
  the death-endpoint training set to 966 and may bias the censoring
  distribution if the missingness is informative.
- **Time axis is patient age.** We use `age_at_event - age_v1` as the
  duration, which assumes study entry coincides with the first observation
  in the CSV. If the original cohort had pre-baseline events these are
  excluded by construction.
- **No temporal embedding model.** With ~5 irregular visits per patient,
  Fourier/wavelet features of the trajectory were not informative enough
  to justify; the `slope` and `summary` views capture what is recoverable.
- **Search-on-search overfit.** Selecting the best of 189 candidates by
  CV gives an optimistic view of held-out performance. The held-out
  Trustii leaderboard is the trustworthy estimate.
