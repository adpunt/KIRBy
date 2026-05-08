# V07 — Adversarial validation

**Type:** Sanity check (distribution shift detector)
**Domain:** Any supervised competition with separate train and test/leaderboard sets
**Cite:** Folk wisdom on Kaggle; Pan et al. "Managing Dataset Shift by Adversarial Validation", arXiv:2112.10078

## What it is
1. Drop the target.
2. Label all train rows 0, all test rows 1.
3. Train a binary classifier (XGBoost is canonical) to discriminate train from test using only features.
4. Report ROC-AUC.
   - AUC ≈ 0.5 → train and test are exchangeable; OOF on train is a faithful proxy for LB performance.
   - AUC ≫ 0.5 → distributions differ; OOF will systematically mis-rank candidates on test.

## Why it helps
The most common reason "OOF says winner, LB says loser" is that the LB scoring set is drawn from a different distribution than train. If you don't check, you have no idea whether your validation protocol is even applicable. Catches data-leak features (a feature unique to one split) instantly.

## When to use
- Always run as a one-off sanity check at the start of any competition
- After any feature-engineering pass that touches train and test asymmetrically
- When OOF leaderboard divergence is > expected noise

## When not to use
- Train and test deliberately come from different distributions (domain-adaptation problems) — then high AUC is the *task*, not a bug

## Implementation
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
X_all = np.vstack([X_train, X_test])
y_adv = np.r_[np.zeros(len(X_train)), np.ones(len(X_test))]
auc = cross_val_score(GradientBoostingClassifier(), X_all, y_adv,
                      cv=5, scoring='roc_auc').mean()
print(f'adversarial AUC = {auc:.3f}')
```

If AUC > 0.65, look at feature importances of the adversarial model — those are the leak features.

## Trade-offs
- Cheap (single classifier fit)
- Doesn't tell you *how* to fix the shift, only that it exists
- Can also be applied per-prediction (does *each candidate's* OOF distribution match its val_test distribution?) — useful for fusion gating

## Empirical results (fill in as tested)
| Competition | Date | Primary metric | LB before → after | Notes |
|---|---|---|---|---|

---
[← back to index](../README.md)
