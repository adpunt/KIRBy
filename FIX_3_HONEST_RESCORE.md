# Fix 3 — report the paired number, not the searched maximum

Copy below the line into a fresh chat.

---

23 experiments report a hybrid number that is the best of roughly 3,000 searched configurations,
scored on the folds that chose it, and subtract from it a baseline that is the best of about 10
representations with no search at all. The difference is the headline. Fix the reporting.

The files:

```
c02 c05 c06 c09 c11 c14 c16 c18 c22 c23
s08 s09 s10 s11
similarity_optimized_hybrid.py
similarity_metrics/feature_ablation.py
similarity_metrics/run_battery_moleculeace.py
opig_activity_cliff_qsar opig_cdr_flexibility opig_cgr_comparison
opig_graphinity_ddg opig_kingsmith_component_splits opig_trastuzumab_enrichment
```

`honest_rescore` at `tests/experiments/_shared.py:881` applies the same held-out-fold tax to
both arms. It is arithmetic over per-fold scores. It fits nothing and reads nothing from disk.
`build_honest_rescore_input` at `_shared.py:1002` builds its input from a `SearchResult`.

Work in three tiers, cheapest first:

1. **Where the input block is already stored and never read.** About a third of these already
   call `build_honest_rescore_input` and write the block. Compute `honest_rescore` on it, print
   `delta_mean_of_folds`, and store it next to the existing number. Do not delete the old
   number; label it as the searched maximum.
2. **Where the per-fold scores exist but the block is not built.** Build it.
3. **Where per-fold scores are not stored at all.** Do not fabricate them. Report which files
   these are and what they would need. `c14`, `c23`, `c06`, `c19`, `opig_kingsmith` and
   `opig_trastuzumab` are the ones I expect here.

Rules:

- Read each file before editing. Say which tier it is in and why.
- Never overwrite the existing number. Add the paired one beside it, named so a reader can tell
  the two apart.
- `delta_mean_of_folds` is the paired quantity. A difference of two means is not the same thing
  and diverges whenever either arm drops a fold.
- Do not run any experiment. This is a reporting change.
- Report which stored result files gain the new field only on a re-run, and which can be
  re-scored from what is already on disk.
