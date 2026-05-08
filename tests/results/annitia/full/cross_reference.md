# Cross-reference: val_test contested ↔ training failures

For the most-contested val_test patients, find their 5 nearest training
neighbours in feature space. Check whether those training patients are
failure cases (FN-like, FP-like, or simply high-discordance).

High overlap → val_test contested patients ARE the same failure modes
we see in training. Specialist routing should focus there.

## Per-endpoint summary

### hepatic

- Top 50 contested val_test patients analysed
- 0.0% have a nearest-train-neighbour in FN-like set
- 38.0% have a nearest-train-neighbour in FP-like set
- 6.0% have a nearest-train-neighbour in hardest-30 set
- 38.0% have a nearest in ANY failure category

### death

- Top 50 contested val_test patients analysed
- 4.0% have a nearest-train-neighbour in FN-like set
- 30.0% have a nearest-train-neighbour in FP-like set
- 18.0% have a nearest-train-neighbour in hardest-30 set
- 36.0% have a nearest in ANY failure category
