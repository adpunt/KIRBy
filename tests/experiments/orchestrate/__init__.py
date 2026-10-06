"""One ledger, one worker: how experiments are run on the server since 2026-10-06.

plan.py    which rows (datasets and multi-dataset scripts), stages and units exist
ledger.py  the ledger file, its lock, claims, marks, the rendered LEDGER.md
runner.py  how one unit is run as a subprocess
search_unit.py  the per-fold searches, one fold per unit, built before experiments

Entry points: tests/experiments/worker.py (every server job),
              python -m experiments.orchestrate.ledger (init / status / reset ...).
"""
