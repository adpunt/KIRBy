"""PT-Rep experiment package — pre-training the KIRBy hybrid representation.

See PRETRAIN_REP_ROADMAP.md for the full design. Phase 0 scaffolding:
  - pt_registry : named PT corpora -> raw entities + labels + groups + dedup keys
  - pt_quality  : two-layer corpus quality pipeline (dedup hygiene + quality filter)
  - pt_leakage  : PT/FT leakage removal (InChIKey for molecular, sequence for antibody)
"""
