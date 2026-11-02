# Chapter 7: suggestions for the author's paragraphs (not edited)

Benchmark pass 2026-10-10. Paragraphs ranked in the top 150 of `benchmark/worklist.csv` with no "manual review" marker directly above them.

## 1. "Noise mitigation, selected by a detector or a flag." (itemize, Evaluation and controls; rank 11)
- Flagged: boosters at 2.3 per 100 words against a reference mean of 0.1 ("must"); nouns where verbs would do ("action", "mitigation").
- Suggested edit:
  - Old: "Where a control ran, the action must also beat the same action on molecules chosen at random."
  - New: "Where a control ran, we also compare it with the same action on molecules chosen at random."

## 2. "The network is also trained with a Huber loss" (Robust losses; rank 43)
- Flagged: 9.3 numbers per 100 words against a reference mean of 1.8 (7 numbers in 75 words).
- Suggested edit (drops a repeated unit; the threshold values stay):
  - Old: "Under the shift in 10\% sized by model error, each model's own robust loss is tried at $c = 1.5$."
  - New: "Under the shift sized by model error, each model's own robust loss is tried at $c = 1.5$."

## 3. "where $b_s$ is the source's offset, $u_i$ is" (Selected by a known flag; rank 98)
- Flagged: booster "never"; "the X of the Y" chains.
- Suggested edits:
  - Old: "Two further arms use the true $\rho$ and $\sigma_e$, which a user never has, so they are bounds and not methods."
  - New: "Two further arms use the true $\rho$ and $\sigma_e$, which a user does not have, so they are bounds."
  - Old: "The value of $\rho$, 0, 0.3 or 0.7, sets how much of the error the measurements share."
  - New: "$\rho$, 0, 0.3 or 0.7, sets how much error the measurements share."

## 4. "Two learners, LightGBM and an SVM, each score" (Two learners; rank 133)
- Flagged: lexical density 50.6% against a reference mean of 59.3% ("the X of the Y" chains).
- Suggested edit:
  - Old: "This follows the exchange of co-teaching \citep{Han2018}, done between whole refits rather than inside one training loop."
  - New: "This follows co-teaching's exchange \citep{Han2018}, done between whole refits instead of inside one training loop."
