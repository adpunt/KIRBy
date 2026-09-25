# Metric-direction audit, noise experiments

Dated 2026-09-25.

Files with uncommitted edits were audited as they stood in the working tree on
2026-09-25. That includes n06, n07, n09, n11, n12, n13, n19–n41 (except n28,
which does not exist), noise_ceiling_analyse.py and
src/kirby/utils/noise_estimation.py. n07_remediation.py changed while this audit
ran; its line numbers are the ones after that change. Line numbers anywhere in
this file are only good for that working tree.

## The rule being checked

Every dataset's metric runs one of two ways. rmse is lower-is-better. spearman,
auc, mcc and f1_weighted are higher-is-better. The direction comes from
`get_domain_metric(data)` in tests/experiments/_shared.py:2510, which returns
(metric_name, metric_fn, lower_is_better). Any top, best, sort, argmax, argmin,
rank, win count or threshold on model scores has to read it.

In the tables, "gain" means a score difference signed so that a positive number
always means better. Most noise files compute it once, in a function called
`_paired`, `_gain` or `recovery_share`, and every later count reads that signed
number.

A "recovery" percentage in n07, n11, n12, n13 and n21 is
(cleaned − noisy) / (clean − noisy) × 100. Both the top and the bottom are
differences in the same metric, so the direction cancels. Ordering by it needs no
direction.

## Rows that do not read the direction

One row is one line of code that orders or compares model scores and gets the
direction wrong on some datasets.

| File:line | What is compared | Reads lower_is_better | What goes wrong |
|---|---|---|---|
| n06_bindingdb_ground_truth.py:388–389 | Inside `n11_ranking`: it drops an n11 noise level when the stored `gap` says the injected noise did not make the model worse. | NO, in effect. It reads the direction, but it applies it a second time to a number that already has it. | n11 writes `gap` already signed so that positive means the noise hurt, on every metric (n11_estimation_remediation_grid.py:1277, `(_nz - _c) if lower_is_better else (_c - _nz)`, sign-normalised on 2026-09-13 in commit 1d9487b). n06's check was written on 2026-09-11 (commit 22067eb) for the older unsigned `clean − noisy`, and drops the level when `gap > 0` on an rmse dataset. On rmse datasets it therefore drops every noise level where the noise raised rmse, and keeps only the levels where it lowered rmse. The n11 ordering that n06 correlates against (`results['transfer']['n11']`, built by `transfer_to_injected_noise` at n06:498) is then built from the wrong noise levels, or from none. On spearman, auc and mcc datasets the check drops the right levels. n11 files written before 2026-09-13 still carry the unsigned gap, and for those the check is right. |

Count: 1 row.

## Full table

One row is one line of code, or one group of lines doing the same thing in one
function, that orders or compares model scores. "shared" in the third column
means the file calls a function in tests/experiments/_shared.py, whose own row is
in the first block.

### Shared functions the noise experiments call

| File:line | What is ordered or compared | Reads lower_is_better | Where the direction comes from |
|---|---|---|---|
| _shared.py:884 | `extract_top3_base_reps`: single representations by their score for one model (default rf) | YES | :872, `object_direction(search)`: metric name first, stored flag only when the name is not a known metric |
| _shared.py:7164, :7186 | `get_best_for_downstream`: best stored configuration across budgets, then across models | YES | :7162, :7184, `object_direction` |
| _shared.py:6235, :6255 | `get_best_search_for_downstream`: stored searches in the pool, best first, inside one search family | YES | :6251, the search's own flag |
| _shared.py:6272–6275 | `get_best_search_for_downstream`: picking between two families when they share one metric | YES | `_lower` from the families' flags |
| _shared.py:7467 | `top_base_reps`: single representations for one model | YES | `_sel_base_direction`, which calls `object_direction` |
| _shared.py:7668 | `top_hybrids`: of two duplicate configurations, keep the better one | YES | `_sel_hybrid_direction`, the stored flag |
| _shared.py:7703–7704 | `top_hybrids`: stored configurations, best first | YES | same |
| _shared.py:8085 | `hybrid_deltas`: hybrid against the mean of its source representations and against the best single one | YES | :8008, `_sel_hybrid_direction` |
| _shared.py:4796 | `noise_base_reps_by_model`: the top 2 single representations per model | YES | calls `top_base_reps` (:7467) |
| _shared.py:7875 | `select_model_rep_pairs` | YES | calls `top_base_reps` and `top_hybrids` |

`base_rank_groups` (_shared.py:4852), `pool_recipes_by_family` (_shared.py:6334)
and `pool_recipe_per_downstream_model` (noise_helpers.py:1302) order no scores.
They group or pass through choices already made by the functions above.
`pick_noise_base_rep` does not exist in _shared.py in this working tree.

### Noise experiment files

| File:line | What is ordered or compared | Reads lower_is_better | Where the direction comes from |
|---|---|---|---|
| n01:580 | Top 3 single representations by mean rf score over the per-fold searches | YES | `reverse=not lower_is_better`; lower_is_better from :307 |
| n01:763–766 | Cliff minus non-cliff gap, signed | YES | explicit branch |
| n01:850 | Representations sorted by the size of that gap, smallest first | NOT NEEDED | `abs()` of the gap signed at :763 |
| n01:524, :568, :630, :638 | best recipe per fold; top representations and hybrids | YES | shared |
| n02:923 | Count of models whose score improved with sample weights | YES | `d < 0 if lower_is_better else d > 0` |
| n02:941 | Representations by unweighted score, best first | YES | `reverse=not lower_is_better` |
| n02:962 | Weighting gain, signed | YES | explicit branch |
| n02:971 | Hybrid gain minus best-representation gain | NOT NEEDED | both gains from :962 |
| n02:470, :488, :528, :1084 | pool lookups, top 3 single representations | YES | shared |
| n06:388–389 | Drops an n11 noise level when the stored `gap` says the noise did not hurt | NO | see the first section: n11 already signed `gap` (n11:1277), so on rmse datasets the levels where noise hurt are the ones dropped |
| n06:461–462 | Drops an n13 cell when noise did not hurt the model | YES | computes `c_s - n_s` itself, unsigned, then branches on direction |
| n06:406, :552–553 | Median n11/n13 recovery per estimator; rank correlation between orderings | NOT NEEDED | recovery is a ratio, direction cancels (n11:1332, n13:1328) |
| n06:316, :752 | pool lookup, top 3 single representations | YES | shared |
| n07:1193–1198 | `rank_models_and_reps`: each model's best score, models ordered best first | YES | explicit branch |
| n07:1209–1210 | Each kept model's representations, best first | YES | explicit branch |
| n07:1765–1768 | Printed best representation per model | YES | explicit branch |
| n07:2700 | Per-representation check that the noise hurt, before dividing | YES | explicit branch |
| n07:2740 | Noise gap, signed | YES | explicit branch |
| n07:2777 | Hybrid check that the noise hurt, before dividing | YES | explicit branch |
| n07:2834 | Per-model check that the noise hurt, before dividing | YES | explicit branch |
| n07:2801, :2815, :2848, :2865, :2878 | Best cleanup by recovery | NOT NEEDED | recovery ratio, computed only when the noise hurt |
| n07:1503, :1653, :1660, :1668, :2104, :2112 | pool lookups, top representations and hybrids | YES | shared |
| n09:564–574 | Recipe chosen from the inner search on noisy labels, across budgets | YES | explicit branch |
| n09:822–830 | Best label-noise method per condition (printed) | YES | explicit branch |
| n09:863–870 | Score ratio clean/noisy or noisy/clean | YES | explicit branch |
| n09:939–947 | Best label-noise method per condition at the highest noise | YES | explicit branch |
| n09:1010–1014 | Hybrid improves or harms against each single representation | YES | explicit branch |
| n09:1042–1046 | Weighted hybrid improves or harms against the hybrid | YES | explicit branch |
| n09:1061–1062 | Top 3 label-noise methods | YES | `reverse=not lower_is_better` |
| n09:292, :301, :310, :337 | pool lookups, top representations | YES | shared |
| n11:1277 | Noise gap per model, signed | YES | explicit branch |
| n11:1382–1385 | Positive-recovery count; best configuration printed | NOT NEEDED | recovery ratio at :1332 |
| n11:1544 | Configuration picked on the other folds | NOT NEEDED | recovery ratio at :1536 |
| n11:1664, :1801 | All configurations and robust configurations sorted by recovery | NOT NEEDED | recovery ratio |
| n11:807, :912 | pool lookup, top representations | YES | shared |
| n12:1278 | Noise gap per model, signed | YES | explicit branch |
| n12:1474, :1486 | Does weighting help (weighted recovery above unweighted) | NOT NEEDED | recovery ratio at :1360 |
| n12:1538, :1562 | Best k, best importance method, by mean recovery | NOT NEEDED | recovery ratio |
| n12:816, :861 | pool lookup, top representations | YES | shared |
| n13:1283–1287 | Retention ratio against the clean model | YES | explicit branch |
| n13:1295 | Oriented difference against the clean model | YES | explicit branch |
| n13:1428–1429 | Best label-noise method at each strength (printed) | YES | explicit branch |
| n13:1570–1576 | Best method per noise shape | YES | `lower = results['lower_is_better']` at :1556, which raises if absent |
| n13:1599 | Hybrid beats each single representation, count | YES | same |
| n13:1605 | Similarity-objective hybrid beats each single representation, count | YES | same |
| n13:352, :370, :537, :550, :646, :694 | pool lookups, top representations and hybrids | YES | shared |
| n15:602 | Per-fold advantage of the clean-label recipe over the noisy-label recipe | YES | explicit branch |
| n15:694–696 | Max, min, share of cells with a positive advantage | NOT NEEDED | advantage from :602 |
| n15:229 | pool lookup | YES | shared |
| n16_importance_under_noise.py:564 | pool lookup | YES | shared |
| n17:1392 | Gain over no cleaning, signed | YES | explicit branch |
| n17:1398, :1418, :1452–1461 | Floor-to-ceiling span, gains over each floor, removal position | NOT NEEDED | `_gain` at :1392 |
| n17:1564–1566 | Corrections printed best first | NOT NEEDED | sorted on `_gain` values |
| n17:588, :606, :864 | top representations and hybrids, pool lookup | YES | shared |
| n18:396 | `_won`: cleaned beats uncleaned on one fold | YES | explicit branch |
| n18:1159 | Folds won per setting | YES | calls `_won` with lower_is_better |
| n18:1163 | Mean gain per setting | YES | explicit branch |
| n18:1213–1217 | Shuffled-detector gain and real-minus-shuffled | YES | `sign = -1.0 if lower_is_better else 1.0` |
| n18:1303–1309 | Settings won on most or every fold; best setting | NOT NEEDED | from :396 and :1163 |
| n18:1373 | Settings where the real detector beat the shuffled one | NOT NEEDED | from :1213 |
| n18:835, :842 | top representations and hybrids | YES | shared |
| n19:1594 | Gain per fold against untouched labels | YES | explicit branch |
| n19:1615, :1616, :1635 | Folds won and lost; gain above its own spread | NOT NEEDED | gains from :1594 |
| n19:1727 | Gain of each model variant against its plain model | YES | explicit branch |
| n19:1741, :1742 | Folds won and lost | NOT NEEDED | gains from :1727 |
| n19:931, :1087, :1094 | pool lookup, top representations and hybrids | YES | shared |
| n20:624–625 | `_rank_stored`: stored configurations, best first | YES | `reverse=not lower_is_better` |
| n20:1652 | Gain per fold | YES | explicit branch |
| n20:1659, :1660, :1672 | Folds won and lost; gain above its own spread | NOT NEEDED | gains from :1652 |
| n20:1869 | Largest minus smallest score across the four arms, per fold | NOT NEEDED | a spread, no better or worse |
| n20:1243, :1428, :1431, :1532 | best recipe from n20's own searches | YES | shared |
| n21:1258 | Noise gap per model, signed | YES | explicit branch |
| n21:1386–1388 | `_paired`: gain of one distance over another | YES | takes `higher_is_better`; callers pass `not lower_is_better` for model scores (:1446, :1481) |
| n21:1446, :1481 | Repair score, each distance against the plain one, and exponent against exponent | YES | `not lower_is_better` |
| n21:1400–1405 | Cells won, lost, tied | NOT NEEDED | gains from :1388 |
| n21:753, :824, :1163, :1167 | pool lookup, top representations and hybrids | YES | shared |
| n22:466, :539, :543 | top representations and hybrids | YES | shared |
| n23:1228 | Gain per fold | YES | explicit branch |
| n23:1235, :1236, :1249, :1298–1307 | Folds won and lost; best setting | NOT NEEDED | gains from :1228 |
| n24:1063 | Gain per pass | YES | explicit branch |
| n24:1070, :1071, :1144–1164, :1190–1192 | Passes won; beat every random seed; beat no cleaning; best and worst seed | NOT NEEDED | gains from :1063 |
| n25:712 | `_gain`, signed | YES | explicit branch |
| n25:1419–1434, :1452–1458, :1549 | Passes won; cliff exemption against random exemption; best setting | NOT NEEDED | `_gain` at :712 |
| n25:1061, :1068 | top representations and hybrids | YES | shared |
| n27:585 | `_loss` from removing molecules, signed | YES | explicit branch |
| n27:610, :622–625, :652 | Folds worse; best and worst draw | NOT NEEDED | `_loss` at :585 |
| n29:996 | Gain per fold against the mean-label arm | YES | explicit branch |
| n29:1003, :1004, :1012, :1038 | Folds won and lost; gain above its own spread | NOT NEEDED | gains from :996 |
| n30:346–348 | `recovery_share`: refuses when the clipped labels did not hurt | YES | explicit branch |
| n30:351 | `recovery_share`: gain of the arm | YES | explicit branch |
| n30:776 | `_paired`: gain per pass | YES | explicit branch |
| n30:783, :784 | Passes won and lost | NOT NEEDED | gains from :776 |
| n31:736 | Gain per pass | YES | explicit branch |
| n31:743, :744, :308–316, :839 | Passes won; precision where the mean gain crosses zero | NOT NEEDED | gains from :736 |
| n32:928 | `_paired`: gain per pass | YES | explicit branch |
| n32:935, :936, :1023 | Passes won and lost; beat every random seed | NOT NEEDED | gains from :928 |
| n32:950–955 | `recovery_share`: refuses when noise did not hurt; gain of the arm | YES | explicit branch |
| n33:921, :924, :933, :936 | Arms against noisy labels; recovery share | NOT NEEDED | imports n30's `_paired` (:776) and `recovery_share` (:346) |
| n34:307, :311 | `recovery_share`: noise gap and arm gain, signed | YES | explicit branch. The refusal at :308 uses `abs(gap)`, so a large gap where the noise IMPROVED the score is not refused and the share comes out with its sign flipped. That is not a direction misread; it is recorded here because n30:348 and n32:952 refuse the same case and n34 does not. |
| n34:688 | `_paired`: gain per pass | YES | explicit branch |
| n34:736–746 | Arms against no correction; recovery share | NOT NEEDED | from :688 and :307 |
| n35:842 | `_paired`: gain per pass | YES | explicit branch |
| n35:849, :850, :892 | Passes won and lost | NOT NEEDED | gains from :842 |
| n36:541 | `_gain`, signed | YES | explicit branch |
| n36:563, :574, :584, :597 | All measurements against none; rule against random; min and max gain | NOT NEEDED | `_gain` at :541 |
| n38:838, :841, :852, :855 | Arms against no cleaning; recovery share; beat every random seed | NOT NEEDED | imports n32's `_paired` (:928) and `recovery_share` (:950) |
| n39:923, :928 | Arms against the plain mean; recovery share | NOT NEEDED | imports n34's `_paired` (:688) and `recovery_share` (:307) |
| n40:304 | Inner tuning: best setting by inner-fold score | YES | `nanargmin if lower_is_better else nanargmax` |
| n40:367 | k for k-nearest-neighbours by inner-fold score | YES | `(min if lower_is_better else max)` |
| n40:423 | GP noise term by inner-fold score | YES | `(min if lower_is_better else max)` |
| n40:351, :410 | k and GP noise term by Mallows Cp | NOT NEEDED | Cp is a training error plus a penalty, lower is better on every dataset; it is not the dataset's metric |
| n40:1055–1066 | Arms against the fixed model; recovery share | NOT NEEDED | imports n30's `_paired` (:776) and n33's `_share` |
| n41:936, :941, :951 | Arms against no cleaning; recovery share | NOT NEEDED | imports n32's `_paired` (:928) and `recovery_share` (:950) |
| noise_ceiling_analyse.py:279–282 | Direction of each stored curve | YES | read from the result file's `lower_is_better`. `bool(None)` would give higher-is-better if the key were missing; `make_structured_output` (_shared.py:8633) always writes it |
| noise_ceiling_analyse.py:913, :978 | Error-metric-only fits refused, and the starting slope's sign, by direction | YES | `curve.lower_is_better` |

Each n26, n27, n29–n41 file also calls `get_best_search_for_downstream` through
`pool_recipe_per_downstream_model` and `noise_base_reps_by_model`. Those are
YES through the shared rows and are not listed once per file.

## Lines the grep matched that do not order model scores

The grep matched 1,115 lines of code across these 42 files, with comments and
strings removed, counted before n07 changed. The full table holds 125 rows,
and one row can cite several lines. Every other matched line is one of these:

- Names only: imports, dictionary keys and variable names containing best, top,
  rank or won, with no comparison on that line.
- Suspicion or noise scores: n07:709–931, n17:496, n23:465, n25:735, n31:272,
  n36:221, n41:203–220, noise_helpers.py:887, noise_estimation.py:2000–2001,
  and every percentile or threshold in noise_mitigation_methods.py. Higher always
  means more suspicious. None is compared with a model score.
- Detection average precision against injected or flagged labels, higher always
  better: n21:1464, n21:1488–1491, n22:789.
- Feature-importance rankings, not model scores: n16_importance_under_noise.py
  :419–420, :899, and all of n16_analyse.py.
- Per-molecule training losses: n32:343, noise_mitigation_methods.py:1070–1071,
  :1120.
- Sorting by counts, timings, names, noise levels, distances or label values:
  for example n06:936–939, n20:598, :746–749, n22:320, :701, n26:668,
  noise_ceiling_analyse.py:1058.
- `min(k, n)` and `max(floor, x)` bounds on sizes and denominators.
- Formatting only: n02:448 and :888 add a "+" to a printed difference.

None of _ground_truth_providers.py, coverage.py, n14, n26 or n37 orders or
compares a model score.
