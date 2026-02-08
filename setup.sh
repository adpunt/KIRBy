#!/bin/bash
# Environment setup for KIRBy
# Called by SLURM scripts before running experiments

# Install KIRBy in editable mode (picks up src/kirby)
pip install --quiet -e . 2>/dev/null

# Install any missing dependencies
pip install --quiet shap captum lime Boruta 2>/dev/null
