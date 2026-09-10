# Fix 4 — the small defects

Copy below the line into a fresh chat.

---

A list of small, contained defects across the KIRBy experiment suite. None changes what an
experiment measures. Work through them, read the code before each, and report anything that
turns out to be wrong.

**A. AUROC computed on hard 0/1 labels.** `tests/experiments/_shared.py:1836`
(`evaluate_single_rep_cv`) scores with `model.predict(X_val)`. Eight datasets are registered
with metric `auc` — `tests/experiments/dataset_policy.py` lines 226, 248, 250, 258, 260, 261,
273, 315. `_predict_for_metric` at `src/kirby/find_best_hybrid.py:2427` returns probabilities
only when the metric is the built-in AUC, so swapping it in leaves every regression dataset
unchanged. Callers affected: `c01`, `c05`, `c19`, `c21`, `n11`, and `framework/f06`. Say
explicitly that this touches a framework result before you change it.

**B. An undefined correlation written as `0.0`.** Five files. `_nanrank.is_defined` exists in
`tests/experiments/similarity_metrics/_nanrank.py` and twelve scripts already use it. Return
NaN and set the `_defined` flag the sibling files write.

- `similarity_metrics/simeval.py:203` — `eval_patterson`. Note `0.0` is the *best* attainable
  value of that statistic, so a degenerate distance matrix currently wins.
- `similarity_metrics/simeval.py:182` — `eval_modi_binary`.
- `similarity_metrics/simeval.py:384` — `eval_noise_recovery`.
- `similarity_metrics/scaffold_feature_isolation.py:64`.
- `similarity_metrics/nary_similarity.py:146`.
- `opig_scaffold_similarity.py:262` and `:267`.
- `s12_cross_domain_cliffs.py`.

In `opig_scaffold_similarity.json` every MutID and BaseDelta entry on both koenig datasets is
exactly `0.0` for this reason, and those zeros compete in the `max()` at line 355 that picks the
winning representation. Say which stored files become uninterpretable rather than wrong.

**C. Smoke runs overwrite full-run result files.** `similarity_metrics/scaffold_experiment.py`
and `similarity_metrics/scaffold_ddg_experiment.py` write the same filename whether or not
`--cap` was passed. A 60-row smoke run silently replaces a full-run file. Tag the output with
the row cap.

**D. Flags parsed and never used.**

- `opig_scaffold_hybrid.py:207` — `--models`, `--budgets` and `--max-folds` reach `config` and
  never reach `run_nested_cv`.
- `opig_blueprint_transfer.py:573` — `--max-folds` parsed, never read.
- `s12_cross_domain_cliffs.py:527` — `--sim-threshold` is forwarded but the percentile sweep
  branch ignores it.
- `c23_evaluator_model_matching.py` — `--models` and `--budgets` advertised, neither read.

Either wire each flag up or remove it. Say which you did and why.

**E. Cache namespaced by mangling the dataset name.** `c05:226`, `c07:240`, `c23:100`,
`c24:191`, `c15:504`. `find_best_hybrid.py:5566` documents this as forbidden — it bypasses the
per-dataset budget policy and the empirical model skips, which resolve on the true registry
name. `cache_namespace=` is the supported parameter and `c06:179` already uses it. Changing this
makes existing cached searches unreachable; say how many files that is before you change it.

Rules:

- Read the code before each edit. Quote `file:line`.
- Do not change what anything measures.
- Do not run any experiment beyond an import check.
- Report every case where the defect is not what this list says it is.
