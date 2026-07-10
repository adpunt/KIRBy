"""noisy_labels — port of the `noisy_housing` label-noise / genuine-outlier study.

Chemistry mapping (see the KIRBy noise-suite plan): housing "genuine outliers"
(correct-but-atypical labels, identified and NEVER corrected) are the direct
analogue of ACTIVITY CLIFFS; housing log-price is pIC50/pKi (already log-scale);
housing features are KIRBy representations. The scientific contract carried over
verbatim from the source is:

    A large residual is NOT presumed to be a wrong label.

THIS PACKAGE IS A PARTIAL PORT. Only the split-free / diagnostic slice needed by
the n14 activity-cliff landscape survey is ported so far:
    core.py        — from_linear/to_linear, metrics, group-aware OOF prediction.
    diagnostics.py — stage0_diagnostics, segment_report.

The full corrections / weights / triage families (global_shrinkage,
heteroscedastic_shrinkage, npmle_eb, gated_shrinkage, triage, precision_weights,
shap_neighbor_weights, ...) are DEFERRED until the landscape survey establishes
whether they are worth building, and on which datasets.

Key adaptation vs the source: every out-of-fold routine in the housing code used
KFold(shuffle=True) / train_test_split, which is forbidden in KIRBy (random
splits leak scaffold-analogue labels and deflate the very residuals these methods
key on). All OOF here is GROUP-AWARE — see core.oof_predict.
"""

from .core import (
    from_linear,
    to_linear,
    log_rmse,
    median_ape,
    noise_recovery_corr,
    segment_report,
    make_model,
    oof_predict,
    oof_log_predict,
    group_kfold_splitter,
    _safe_n_splits,
)
from .diagnostics import stage0_diagnostics

__all__ = [
    "from_linear",
    "to_linear",
    "log_rmse",
    "median_ape",
    "noise_recovery_corr",
    "segment_report",
    "make_model",
    "oof_predict",
    "oof_log_predict",
    "group_kfold_splitter",
    "_safe_n_splits",
    "stage0_diagnostics",
]
