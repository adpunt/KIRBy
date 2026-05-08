# User-suggested improvements / TODOs (running log)

## Submission process
- [x] Every submission CSV must be paired with a journal/notebook (`build_journal.py` autoamtes)
- [x] Save submissions to `tests/results/annitia/submissions/` with paired `.zip` + `_notebook.ipynb` + `_journal.md`
- [ ] When new submissions are produced, regenerate journals automatically (auto_chain does this)

## Fusion logic
- [x] Test multiple top-N values rather than fixed N=6 (`refuse_variants.py` sweeps K)
- [x] Keep multiple weighting modes — equal, cv-weighted, single, top1, rrf, stack_meta — don't reduce variants
- [x] Try a "no stacking" variant (best non-stack reported separately)
- [x] Cross-endpoint stacking: death info should help hep prediction and vice versa (`stack_meta_xe`)
- [x] Diversity-aware ensemble selection (`refuse_diverse.py`, greedy by ensemble C-improvement)
- [x] Patient-aware fusion: anomaly_router, disagreement_router, cluster_router, anomaly_blend, weighted_anomaly_per_patient (`refuse_meta_aware.py`)
- [x] More meta-averaging variants: top-K, weighted by OOF, equal-weight all (`meta_avg_top5/weighted/all`)
- [x] Hierarchical fusion: per-source pre-fusion → quality-stratified weighting → inter-source meta → patient-aware route (`refuse_hierarchical.py`)
- [ ] Fusion logic for the larger pool (~200+ models post-Round-4) — flat rank-mean would let one big source dominate; subset-of-top-N with different N is one possible solution among many; this area needs MORE rethinking
- [ ] Subjective analysis on pretraining sources: drop if epic fail (super OOF < 0.55), boost weight if it dominates everything else (super OOF > 0.97 AND > 0.05 above next best). Implemented as `_gate_pretraining_source` in hierarchical.

## Uncertainty / generalisation (to keep exploring)
- [x] IsolationForest anomaly score on raw clinical+snapshot features → correlates r=0.47 with hep ensemble miss rate
- [x] Ensemble disagreement (rank-std across model pool) → correlates r=0.57 with death miss rate
- [x] Use uncertainty signals as extra branches in fusion meta-learner
- [x] Use uncertainty for per-patient model routing (specialist override on top-X% anomalous)
- [ ] Brainstorm additional generalisation mechanisms (see GENERALIZATION_IDEAS.md)
- [ ] More work on uncertainty-aware, patient-specific, anomaly-aware ensembles ("what ensembles work best for what datapoints")

## Validation methodology
- [x] Use HONEST OOF C (nested 5-fold for any learned weights) — not in-sample stack_meta
- [ ] Bootstrap C-index confidence intervals — not just point estimate
- [ ] Repeated K-fold across multiple seeds to reduce variance
- [ ] Hold-out fusion-test set: carve off 10% of training that base models can use but fusion CANNOT, score fusion variants on this analog of real test
- [ ] Per-fold C distribution (not just mean) — surface high-variance fusion variants
- [ ] Group-K-fold by patient cluster (not just event-stratified)
- [ ] Better metrics + validation metrics to plan submission strategy when limited (currently 10 left)

## Diagnostics / analysis
- [x] Per-patient discordance per model (`analyse_residuals.py`)
- [x] Error correlation matrix between models
- [x] Hard-patient list (top-50 by mean discordance across models)
- [x] Anomaly-vs-error correlation (IsolationForest, LOF)
- [x] Router signal: best-model-per-patient distribution
- [x] Ensemble uncertainty per patient

## Cross-endpoint pretraining
- [x] Pretrain on death, fine-tune on hep (TabSurv-D2H, CatBoost-D2H, MLP-D2H)
- [x] Pretrain on hep, fine-tune on death (TabSurv-H2D, CatBoost-H2D, MLP-H2D)

## External data + foundation models
- [x] NHANES external dataset pretraining for `pretrained-mlp` branch
- [ ] Round 4 EXTERNAL stage adds T2.0 (NHANES pretrain × 5 subsets) + T2.1 (zero-shot) + T2.2 (6 transfer recipes) + T2.3 (5×6×2 ablation) — ~70 more models
- [ ] Foundation model weights blocked by torch < 2.4: MITRA, TabTune, SurvTRACE, synthcity tabddpm/tabsyn — torch 2.7 conda env attempt at `~/anaconda3/envs/annitia_torch27` (in progress)

## Hyperparameter tuning (Round 5)
- [x] Optuna script ready (`experiment_hparam.py`): RSF / CatBoost-Cox/AFT / GBSA / NGBoost / LGBM-AFT / Coxnet on BE-best view set
- [ ] Maybe extend hparam tuning into the fusion stage too (re-tune existing models in light of OOF→test gap concerns)

## Auto-chain (autonomous overnight)
- [x] auto_chain.sh waits for Round 4, runs Round 5, re-fuses, builds journals
- [ ] Add `refuse_hierarchical.py` to the chain after `refuse_meta_aware.py`
- [ ] Add bootstrap-fusion variant to the chain (random subset of variants per inference)

## File hygiene
- [x] Large environment files / new envs go on /Volumes/seagate (BUT ExFAT corrupts conda metadata — must use local fs for conda envs)
- [x] Don't migrate old envs

## Hard rules
- [x] No data leakage in pretraining
- [x] Final submission trained on full training data (not val_test)
- [x] Source pool kept clean (sanity_audit.py finds duplicates)
