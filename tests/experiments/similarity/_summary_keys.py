"""Key names and refusal logic shared by the two cross-dataset summaries.

Two aggregators consume the same per-dataset results:
  - tests/similarity_experiments.py::run_cross_dataset_summary, which reads an
    in-memory dict keyed 'exp2_hybrid_similarity', 'exp4_activity_cliffs', ...
  - tests/experiments/similarity/s09_cross_dataset_summary.py, which reads the
    per-experiment result directories 's02_hybrid_similarity',
    's04_activity_cliffs', ...

The two schemes name the same six inputs, so neither aggregator could read the
other's output. Both now resolve a key through ACCEPTED_INPUT_KEYS, which lists
every accepted spelling. Stored keys are never renamed — old result files keep
loading.

Also holds the shared refusal check: an aggregate built from zero metrics is not
a result, and must not be written as one.

Stdlib only — imported by a script that must not pull in the experiment stack.
"""

# Canonical input id -> every key name accepted when reading it.
# First entry is canonical; the rest are historical spellings.
ACCEPTED_INPUT_KEYS = {
    's02_hybrid_similarity': ('s02_hybrid_similarity', 'exp2_hybrid_similarity'),
    's04_activity_cliffs': ('s04_activity_cliffs', 'exp4_activity_cliffs'),
    's05_knn_prediction': ('s05_knn_prediction', 'exp5_knn_prediction'),
    's06_applicability_domain': ('s06_applicability_domain',
                                 'exp6_applicability_domain'),
    's08_optimal_dimension': ('s08_optimal_dimension', 'exp8_optimal_dimension'),
}

# ⛔ WHICH SELECTION RULE PRODUCED THE NUMBER IN A STORED RESULT.
#
# s09 opens the stored result of s02, s04, s05, s06 and s08, takes one number
# from each, and runs a Wilcoxon signed-rank test down the column. If those five
# files were not produced by the same selection rule, that test compares numbers
# that mean different things. It does not raise and it does not look wrong: it
# returns a p-value.
#
# That became a live risk on 2026-09-18, when a shared top-N selector arrived
# and the five files became candidates to move onto it one at a time. Each file
# stamps the rule it used; s09 refuses to test a column whose stamps disagree.
#
# Bump this string whenever a change moves what any of the five reports. A file
# written before the stamp existed carries no stamp, and `selection_rule_check`
# treats that as its own value rather than assuming it matches.
SELECTION_RULE = 'per_file_2026_09_18'

# Minimum datasets the Wilcoxon signed-rank test needs to be worth running.
MIN_DATASETS_FOR_WILCOXON = 3


def selection_rule_of(block):
    """The selection rule a stored input block was written under.

    Returns the stamp, or 'unstamped' for a file written before the stamp
    existed. 'unstamped' is a VALUE, not a wildcard: an unstamped file and a
    stamped one disagree, which is the point — the unstamped one predates
    whatever the stamp records.
    """
    if not isinstance(block, dict):
        return 'unstamped'
    return block.get('selection_rule') or 'unstamped'


def selection_rule_check(blocks_by_input):
    """Do these stored inputs agree on how their numbers were selected?

    blocks_by_input: {canonical_id: stored block or None}. None is skipped —
    a missing input is already handled elsewhere.

    Returns {'ok': bool, 'rules': {canonical_id: rule}, 'distinct': [rule, ...],
             'reason': str or None}. `ok` is False when two present inputs
    carry different stamps, which is exactly when a signed-rank test down their
    column would compare two different quantities.
    """
    rules = {k: selection_rule_of(v) for k, v in blocks_by_input.items()
             if v}
    distinct = sorted(set(rules.values()))
    if len(distinct) <= 1:
        return {'ok': True, 'rules': rules, 'distinct': distinct,
                'reason': None}
    by_rule = {}
    for k, r in sorted(rules.items()):
        by_rule.setdefault(r, []).append(k)
    parts = "; ".join(f"{r}: {', '.join(v)}" for r, v in sorted(by_rule.items()))
    return {
        'ok': False, 'rules': rules, 'distinct': distinct,
        'reason': (f"the inputs were written under {len(distinct)} different "
                   f"selection rules ({parts}). A signed-rank test down this "
                   f"column would compare numbers that were not selected the "
                   f"same way. Re-run the inputs that are behind."),
    }


def get_input(ds_results, canonical_id):
    """Return ds_results[canonical_id] under any accepted spelling, else None."""
    for key in ACCEPTED_INPUT_KEYS.get(canonical_id, (canonical_id,)):
        value = ds_results.get(key)
        if value:
            return value
    return None


def accepted_names(canonical_id=None):
    """Accepted key names, for one input or (flattened) for all of them."""
    if canonical_id is not None:
        return list(ACCEPTED_INPUT_KEYS.get(canonical_id, (canonical_id,)))
    names = []
    for keys in ACCEPTED_INPUT_KEYS.values():
        names.extend(keys)
    return names


def missing_metric_report(all_dataset_results, per_dataset_summaries,
                          metric_fields):
    """Name, per dataset, every metric that did not reach the aggregate.

    metric_fields is a list of (metric_name, summary_field, canonical_input_id).
    Returns {dataset: {metric_name: reason}} covering only the absent metrics.
    """
    report = {}
    for ds_name, ds_results in all_dataset_results.items():
        missing = {}
        for metric_name, summary_field, input_id in metric_fields:
            summary = per_dataset_summaries.get(ds_name, {})
            if summary_field in summary:
                continue
            if get_input(ds_results, input_id) is None:
                missing[metric_name] = (
                    f"input '{input_id}' absent (accepted names: "
                    f"{', '.join(accepted_names(input_id))})")
            else:
                missing[metric_name] = (
                    f"input '{input_id}' present but produced no "
                    f"'{summary_field}'")
        if missing:
            report[ds_name] = missing
    return report


def refusal(all_dataset_results, per_dataset_summaries, metric_fields):
    """Return an explicit error dict if not one dataset contributed a metric.

    Returns None when at least one metric reached the aggregate, in which case
    the caller carries on and computes exactly what it always computed.
    """
    contributed = 0
    for summary in per_dataset_summaries.values():
        contributed += sum(1 for _, field, _ in metric_fields if field in summary)
    if contributed:
        return None
    return {
        'error': 'no_metrics_aggregated',
        'message': ('Cross-dataset summary refused: none of the '
                    f'{len(all_dataset_results)} dataset(s) checked produced a '
                    'single aggregate metric.'),
        'datasets_checked': list(all_dataset_results),
        'metrics_expected': [name for name, _, _ in metric_fields],
        'missing_per_dataset': missing_metric_report(
            all_dataset_results, per_dataset_summaries, metric_fields),
    }


def print_refusal(refusal_dict):
    """Print the refusal so a log reader sees which input each dataset lacked."""
    print("\n  REFUSED: " + refusal_dict['message'])
    for ds_name, missing in refusal_dict['missing_per_dataset'].items():
        print(f"    {ds_name}:")
        for metric_name, reason in missing.items():
            print(f"      {metric_name}: {reason}")
    if not refusal_dict['missing_per_dataset']:
        print("    (no datasets were checked)")
