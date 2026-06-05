"""ANNITIA synthetic-data augmentation suite.

Mechanisms:
    M1 masld_simulator    — LLM-distilled MASLD progression simulator
    M2 outcome_conditioned — synthcity (T, event)-conditional generator
    M5 counterfactual     — perturb censored high-risk patients along progression paths

Protocols:
    P1 class-balance oversample (synthetic positives only)
    P2 pretrain + fine-tune (MLP discrete hazard / RSF warm-start)
    P3 mixed-batch joint training
    P4 self-distillation (teacher labels synthetic, student trains on real + pseudo)

All outputs land in tests/results/annitia/synth/ with hard-links into branches/
so the existing fusion pool picks them up by name convention.
"""
