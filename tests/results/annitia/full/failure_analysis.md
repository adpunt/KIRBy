# Training-side failure analysis

Per-patient discordance on consensus OOF (mean rank across base models).

Definitions:
- **FN-like**: patient had an event but ensemble predicted bottom-quartile risk
- **FP-like**: patient had NO event in followup but ensemble predicted top-quartile risk

## Per-endpoint summary

### hepatic

- n_valid=984, n_events=39, event_rate=0.040
- FN-like: **4/39 = 10.3%** of events missed
- FP-like: **176/945 = 18.6%** of non-events flagged
- correctly-flagged events (high-risk): **23/39 = 59.0%**
- mean disc overall: 0.244
- mean disc on FN-like: 0.610
- mean disc on FP-like: 0.521

Feature comparison (FN-like vs population vs FP-like):

| Feature | Population | FN-like | FP-like |
|---|---|---|---|
| baseline_age | 54.02 | 57.00 | 57.94 |
| last_followup_age | 60.59 | 70.50 | 62.06 |
| time | 6.52 | 5.25 | 4.11 |

### death

- n_valid=695, n_events=76, event_rate=0.109
- FN-like: **10/76 = 13.2%** of events missed
- FP-like: **57/619 = 9.2%** of non-events flagged
- correctly-flagged events (high-risk): **32/76 = 42.1%**
- mean disc overall: 0.035
- mean disc on FN-like: 0.232
- mean disc on FP-like: 0.075

Feature comparison (FN-like vs population vs FP-like):

| Feature | Population | FN-like | FP-like |
|---|---|---|---|
| baseline_age | 53.50 | 57.50 | 61.65 |
| last_followup_age | 61.65 | 70.70 | 66.25 |
| time | 8.57 | 15.80 | 4.60 |
