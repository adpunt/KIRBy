#!/usr/bin/env python3
"""Which of the activity-cliff settings each unit cannot start without.

This is read by the Papers 2 and 3 ledger (papers/ledger.py
unit_hold_reason), so a unit whose setting has no value is HELD rather than
run and recorded as skipped. Writing the value in
cliffs/CLIFF_SETTINGS.json releases it at the next claim, with no ledger
command.

A unit is listed here only when running every candidate would not cover the
question. Three of the four are not listed: their axes all run every value
the code can run, which is not a choice, so nothing holds them.
"""
from __future__ import annotations

#: unit -> the settings it cannot start without.
REQUIRED_SETTINGS = {
    # Her words: "Look at the best outlier detection methods, call one M, and
    # then subsequent best label handling methods, call one L". Both are the
    # best of the noise experiments' own results, so there is no list of
    # values that covers the question and no value is written in the code.
    'cliff_aware_cleanup': ('outlier_detection_methods',
                            'label_handling_methods'),
}

#: Units with no required setting, listed so the absence is deliberate and not
#: an omission.
NO_REQUIRED_SETTINGS = (
    'cliff_sets',
    'definition_agreement',
    'error_by_cliff_status',
)
