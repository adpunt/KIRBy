# Digital Twin Framework for Molecular Reactions

**Status:** Exploratory roadmap. Rewritten 2026-04-20 after realizing the earlier draft confused "graph-structured prediction model" with "digital twin." They're different things.

---

## 1. What a digital twin actually is (grounded in the literature)

I went and read the standards and review papers properly. Summarizing the actual definition before writing anything KIRBy-specific.

### 1.1 Core definition

A digital twin is a virtual model of a **specific physical instance** that stays synchronized with that instance through observations, and supports decisions about that instance. The definition is consistent across:

- **Grieves (2002)**: three parts — a physical product, a virtual product, and a data link between them.
- **ISO 23247 (2021)**: "a fit-for-purpose digital representation of an observable manufacturing element with synchronization between the element and its digital representation."
- **NAS report (2023)**: emphasizes "the complementary role of models and data" — a DT fuses models with observations to drive decisions. Not one definition; DTs must be fit-for-purpose.
- **Kapteyn & Willcox framework**: a library of component-based reduced-order models + Bayesian state estimation that picks which model in the library best matches current sensor data.

### 1.2 What makes a DT different from a model

A model predicts. A DT has four additional properties:

1. **Specificity.** A model is generic (predicts yield for any reaction of this class). A DT is tied to a specific instance (this campaign, this reactor run, this optimization problem). The DT's state represents belief about *that* instance.
2. **Synchronization.** Observations from the physical instance continuously update the virtual model's state. This is data assimilation / state estimation (Kalman filter, Bayesian update, etc.), not one-off retraining.
3. **Bidirectional feedback.** The twin informs actions on the physical instance (next experiment, next setpoint), and those actions produce observations that feed back into the twin.
4. **Queryability about the specific instance.** You can ask counterfactual questions ("what if I change X in this campaign?") and the twin answers with its current state, not a generic prior.

A graph neural network on reaction data has none of these. It's a model. A closed-loop BO system with persistent state and specific-campaign tracking has all of these. It's a twin, even without any graph structure.

### 1.3 Surrogate model vs DT

A **surrogate model** is a cheap computational stand-in for an expensive simulator. Surrogates are often part of a DT's virtual representation, but a surrogate alone is not a DT — it lacks synchronization and specificity.

This matters because much of ML chemistry literature calls surrogate models "digital twins" loosely. They aren't. Being precise here avoids overselling our work.

---

## 2. Prior art in chemistry-specific DTs

### 2.1 What's been done

| Work | What it is | Graph structure? | Sync/loop? |
|---|---|---|---|
| DTCS (Berkeley, Nature Comp Sci 2025) | Forward solver: reaction network → spectra. Inverse solver: spectra → kinetics. Bidirectional loop with real spectroscopic observations. | Reaction network (mechanism) | Yes — core contribution |
| MATTERIX (Nature Comp Sci 2025) | GPU-accelerated simulation of robotic chemistry lab hardware. | Workcell topology | Yes — robot state tracking |
| Guided Bayesian Optimization (Gao et al. 2024) | Standard BO where a DT surrogate is activated when uncertainty is high, to reduce hardware experiments. 46% reduction reported. | No | Yes — BO loop |
| AlphaFlow (Nat Commun 2023) | RL-driven fluidic lab for multi-step chemistry. Closed loop. | No (RL policy) | Yes |
| RoboChem-Flex (Nature Synthesis 2026) | Low-cost self-driving lab with BO. | No | Yes |
| Kapteyn & Willcox (aerospace, 2022) | Library of physics-based component models + Bayesian state estimation picks best-matching model. | Component graph | Yes |
| Yasin (JHU Institute for Assured Autonomy, 2023+) | Research-program framing (articulated in research statement and institutional leadership — IAA, NASA LSIC autonomy group — rather than specific DT publications) for DT in assured autonomy: risk-based trust, resilience under novel / adversarial inputs, formal methods for CPS. Published work is adjacent (graph neural operators, RL with uncertainty-aware decisions) rather than DT-specific. Cross-domain. | Component-level in CPS sense | Implied in framing |

### 2.2 What hasn't been done (honestly)

- A reaction DT where the virtual model is a **graph structured over reaction components/substructures** with **per-slot KIRBy hybrid features**. Most reaction DT work uses flat BO surrogates (GP on concat fingerprints) with no internal graph.
- A **library-of-hybrids** approach for reactions, analogous to Kapteyn-Willcox's library-of-component-models. Different reaction regimes (different chemistry, different catalysts) might want different hybrids; Bayesian model selection picks which hybrid the current campaign is in.
- **Per-node uncertainty** ("which component is limiting my confidence?") as an active-learning acquisition signal.

These are the three KIRBy-specific angles. The rest is existing BO / surrogate-modeling territory.

---

## 3. What "digital twin for reactions" actually means in this project

To be a DT and not just a model, the work must include:

1. **A specific instance being tracked.** In our case: a specific HTE optimization campaign or reaction class of interest. Not "reactions in general."
2. **A virtual model of that instance.** This is where KIRBy contributes — feature representations and model structure.
3. **An observation loop.** Experimental yields come in; the virtual model updates. In simulated form (this is an academic project), we replay HTE plates as if we were running them live.
4. **Decision feedback.** The twin picks the next experiment, or answers counterfactual questions about the campaign.

If we build only the virtual model and skip the loop, we've built a better yield predictor. That's fine, but call it that. Don't call it a digital twin.

---

## 4. Where KIRBy plugs in

KIRBy is a **representation-engineering framework**. In DT terms, it produces better features for the virtual model inside the twin. KIRBy does not provide:

- Data assimilation / state update mechanisms (need: Kalman, Bayesian posterior, online gradient updates).
- Acquisition functions for active learning (need: UCB, EI, Thompson sampling on top of a probabilistic model).
- Uncertainty quantification (need: GP, ensemble, MC dropout, conformal).

All three are needed for a twin. We'd either build them as KIRBy extensions or wrap KIRBy reps inside an existing BO library (BoTorch, Ax).

The decision "are we building a twin or a model" pivots on whether we build the loop. Everything in §1 says: no loop, no twin.

---

## 5. The component-graph question, re-examined

Earlier I conflated "use a graph of components" with "build a digital twin." They're independent. Let me separate them cleanly:

### 5.1 The virtual-model architecture (independent of DT-ness)

Options for how the yield predictor represents reactions internally:

- **A. Flat concat.** `concat(x_aryl, x_ligand, x_base, x_additive)` → RF/XGBoost/MLP. What we have now. The model *can* learn interactions implicitly in hidden layers / tree splits, but doesn't know which columns are which component.
- **B. Per-component features + MLP mixer.** Keep each component's features separate → a small neural net that explicitly takes four input vectors → yield. Learns interactions between specific components.
- **C. Cross-attention over component tokens.** Each component becomes a token; attention heads learn pairwise compatibility (e.g., which ligand × base pairs work).
- **D. Message-passing GNN over a component graph.** Nodes = components, edges = interactions. Most structured; highest engineering cost.

Honest assessment: A learns interactions implicitly but has no notion of "slots"; B/C/D make slots explicit. Whether explicit slots help depends on whether the dataset rewards structured inductive bias. On BH plates (~4K reactions, 4 slots, strong ligand×base coupling), I'd bet B or C beats A. On broader reaction sets, less clear.

### 5.2 None of A–D are intrinsically "digital twin"

All four are virtual-model architectures. Any of them can sit inside a DT loop or outside it. The DT-ness comes from §3.3 — the observation loop and specific-instance tracking.

### 5.3 Counterfactual intervention

Swapping a component's features ("what if ligand L were L'?") is a property that B/C/D support naturally because each component has its own input slot. Architecture A supports it only by substituting column ranges, which is brittle if the model's learned features don't cluster cleanly by slot. So if counterfactuals matter (and for a DT they do), B/C/D earn their complexity.

---

## 6. The actual design choices

Three orthogonal choices:

### Choice 1: Virtual-model architecture
- A (flat concat), B (MLP mixer), C (cross-attention), D (GNN). See §5.1.

### Choice 2: Uncertainty quantification

Four candidate families, each with a different cost/assumption profile:

| Method | Cost | Wraps what? | Notes |
|---|---|---|---|
| NGBoost | medium | standalone | natural gradient boosting over Normal distn; μ + σ per sample; scales well |
| GP (sklearn, Gauche) | O(n³) | standalone | gold standard for small n; Gauche Tanimoto kernel is chem-native; borderline at HTE plate size (~2-6K) |
| Deep ensemble | medium-high | MLP × N seeds | Lakshminarayanan 2017 canonical; empirically beats BNN on tabular; re-seed the base model N=5 times |
| Conformal / CV+ | low | any point predictor | distribution-free; `crepes` package (faster, regression-focused, normalized & Mondrian variants); CV+ slots natively into KIRBy's k-fold design |

**Package**: `crepes` over MAPIE for conformal — ~10× faster on small regression tasks, normalized variant (difficulty-scaled intervals) maps to the chemistry applicability-domain idea, Mondrian variant supports per-slot stratification. Detailed reasoning and sources in §14.

**Base model pool for wrappers**: {RF, XGBoost, MLP}. Ridge is explicitly *out* — closed-form Bayesian variance already exists (`BayesianRidge`), and the QSAR CP literature doesn't use it. Deep ensemble only makes sense on MLP (RF is already an ensemble; XGBoost has diminishing returns from seed noise).

**Self-contained options**: NGBoost and GP are the surrogate themselves — no wrapping.

**BNN** (HMC or full variational) is omitted — tabular UQ benchmarks (Scalia 2020, Hirschfeld 2020) show deep ensembles match or beat BNN at lower compute.

### Choice 3: Sync loop mechanics
- **None** (just a virtual model; not a twin): skip the loop, benchmark prediction accuracy.
- **Retrain-from-scratch on each new observation** (simplest loop; fine for small data).
- **Incremental update** (warm-start from previous fit; keeps the twin persistent across steps).
- **Full Bayesian state over model library** (Kapteyn-Willcox analog; library of KIRBy hybrids, posterior over which fits current campaign).

---

## 7. Experimental roadmap (rewritten)

Three phases, gated. Phase 1 builds the virtual model. Phase 2 closes the loop (this is where it becomes a DT). Phase 3 is the ambitious library-of-hybrids idea.

### Phase 1 — The virtual model

**P1.1: Per-component architecture comparison**
Tests Choice 1 (A/B/C/D) on BH plates with standard train-test evaluation. No loop. Datasets: `buchwald_hartwig`, `bh_doyle_plates`, `suzuki`. Eval: GroupKFold on reaction-id AND leave-component-out (via `assign_component_groups` in `opig_kingsmith_component_splits.py:198`). Primary metric: RMSE; Spearman ρ secondary. Output: which architecture wins on flat-prediction tasks?

**P1.2: Counterfactual consistency**
For each reaction R with components (AH, L, B, A), construct R' by swapping L with L' from another reaction in the *same dataset* (combinatorial plates guarantee R' exists). Predict yield for R' using the model trained on R-but-not-R'. Metric: MAE between predicted R' yield and actual R' yield. Compare across architectures A–D. Tests whether the architecture's swap semantics are faithful. This is a property of the virtual model, not yet a DT.

**P1.3: Uncertainty calibration**
For the best architecture from P1.1, evaluate six UQ conditions on the same GroupKFold splits used in P1.1:

1. NGBoost (standalone)
2. GP with RBF + WhiteKernel (standalone; skip on datasets where n > ~2K)
3. Gauche GP with Tanimoto kernel (standalone; chem-native)
4. Deep ensemble of MLPs (N=5 re-seeds; mean + empirical std across ensemble)
5. Conformal CV+ wrapping RF (via `crepes`; standard + normalized variants)
6. Conformal CV+ wrapping XGBoost (via `crepes`; standard + normalized variants)

Metrics: (a) **calibration** — fraction of true labels inside nominal 90%/95% prediction intervals; (b) **efficiency** — mean interval width at matched coverage; (c) **NLL / CRPS** where the method returns a full distribution (NGBoost, GP, ensemble); (d) **PICP and MPIW** where only intervals are produced (conformal).

Output: which UQ method(s) are viable as P2.1 acquisition-function inputs. Note: conformal produces intervals only — usable for UCB-style acquisition but not EI without extra structure. A1 (OOD detection) from the cross-cutting axis layers directly on top of this — hold out a ligand family, measure interval inflation vs in-distribution.

Existing wiring: `tests/experiments/similarity/s11_bayesian_optimization.py` already has `fit_gp`, `fit_gauche_gp`, `fit_ngboost`, and a unified `predict_with_uncertainty(model, X) → (μ, σ)` interface — reusable directly. Deep-ensemble and conformal-CV+ wrappers need to be added.

### Phase 2 — Closing the loop (becoming a DT)

**P2.1: Simulated active learning**
Given HTE plate data, simulate an optimization campaign. Start with 10% of reactions; at each round, the twin picks the next batch to "run" by UCB/EI on its uncertainty; labels are revealed from the plate. Compare vs random, Latin hypercube, flat-GP BO (existing baseline). Metric: cumulative regret vs budget. The model updates between rounds — this is the sync. Simulated, not real-time, but architecturally a twin.

*Note on `bh_doyle_plates` data: the loader drops rows with missing yield (`reaction_yields.py:425-426`), so after filtering it isn't fully combinatorial. Need to restrict AL to the combinatorial subset OR allow "no-label-returned" outcomes. Decision point during P2.1 design.*

**P2.2: Counterfactual querying in a running campaign**
During a simulated campaign from P2.1, at round k the twin has seen k reactions. Ask it counterfactual questions about components not yet observed. Measure how answers evolve as evidence accumulates. This demonstrates the "query the twin about the specific instance" capability from §3.4.

### Phase 3 — Library of hybrids (Kapteyn-Willcox analog)

**P3.1: Per-regime hybrid library**
Hypothesis: different reaction regimes want different hybrid compositions. E.g., an aryl-halide-electron-poor regime might favor different reps than aryl-halide-electron-rich. Build a library of KIRBy hybrids, one per regime (regime defined by clustering on reactant properties). At inference time, a Bayesian posterior over regimes picks which hybrid to trust for each query.
Eval: does library-of-hybrids beat single-hybrid on heterogeneous reaction sets? Datasets where regimes are clearly distinct (e.g., combined BH + Suzuki).

This phase is the most novel piece and also the most engineering-heavy. Only worth building if Phase 1–2 clear.

### Cross-cutting axis — Assurance and resilience (inspired by Yasin / JHU IAA)

Dr. Alhassan S. Yasin's research-program framing at the JHU Institute for Assured Autonomy — articulated in his research statement and institutional leadership (IAA, NASA LSIC autonomy focus group) rather than in specific DT publications — argues that a digital twin must be *trustworthy under novel inputs and adversarial conditions*, not merely accurate. The framing is cross-domain (CPS security, autonomous vehicles, defense air systems), but the three tests it implies transfer cleanly to a reaction DT where bad recommendations waste real lab time and materials:

**A1: Out-of-distribution detection.** When the twin is shown a reaction with a novel component (a ligand class or additive not in training), does it flag low confidence, or does it confidently predict nonsense? Test: hold out an entire ligand family from training; measure whether the twin's uncertainty on that family is meaningfully larger than on in-distribution queries. Folds naturally into P1.3 (uncertainty calibration).

**A2: Adversarial probing.** Construct reactions designed to stress-test the model — near-duplicate SMILES with trivial syntactic changes that shouldn't change yield, component swaps the training data doesn't support. Measure prediction stability and uncertainty. A weak form of the formal-methods robustness testing Yasin's CPS framing calls for, scaled down to what's achievable without a formal-methods toolchain.

**A3: Risk-based decision boundary.** For a campaign, define a risk threshold θ: below θ, act on the twin's recommendation; above θ, defer to human. Measure how θ affects regret in P2.1's AL loop. Mirrors the risk-managed decision structure Yasin's framing uses for autonomous agents acting on uncertain model outputs; an adjacent precedent is his own work on RL with domain indicators and risk-aware policies (Yasin & Gill, ICAIF FM4TS 2024), though the specifics don't transfer directly (RL vs BO; finance vs chemistry).

These are not separate experiments — they are measurement dimensions layered onto P1.3 and P2.1. Marginal cost is low. Upside: the final writeup can credibly claim an *assurance-aware* DT, not just a surrogate model, and can cite a real cross-domain precedent for the framing.

---

## 8. What's a "slot" and why it matters

A **slot** is the role a component plays in a reaction. Buchwald-Hartwig has 4 variable slots: aryl halide, ligand, base, additive (amine is constant across this dataset; see `reaction_yields.py:144-147`). Every reaction has exactly one molecule in each slot.

"Slot" matters because:
- In a DT virtual model with explicit slots (architectures B/C/D), swapping a component = substituting its slot's input.
- Without explicit slots (architecture A), the same swap = substituting column ranges in a flat vector, which the model treats as generic feature perturbation.

Using "slot" instead of "component" when speaking of the model's input position avoids confusion with "component" as the chemical entity.

---

## 9. Honest novelty assessment

What's been done elsewhere:
- BO on reaction yield with GP surrogate (standard).
- Closed-loop autonomous reaction optimization (AlphaFlow, RoboChem-Flex, several others).
- Digital twins for chemical process engineering (reactor-scale).
- Kapteyn-Willcox component-model libraries in aerospace.

What would be novel if we did it:
- **KIRBy hybrid features inside a closed-loop BO surrogate** (Phase 2). Not groundbreaking but publishable as a BO-with-learned-features study.
- **Library of per-regime hybrids with Bayesian model selection for reactions** (Phase 3). The Kapteyn-Willcox pattern has not been applied to reaction chemistry as far as I can tell. This is the most original contribution.
- **Per-component architectures compared head-to-head on HTE data** (Phase 1). Useful but not radically new.
- **DT assurance framing applied to reaction prediction** (cross-cutting A1–A3). Yasin's assurance-for-autonomy research-program framing (CPS, autonomous vehicles) — not yet published as a DT-specific framework — has not been transposed to chemistry DTs. Adds trust, risk-based decisions, and OOD/adversarial probing dimensions that matter for real lab deployment.

What would NOT be novel:
- Calling a graph-structured prediction model a "digital twin." It isn't.

---

## 10. Dependencies and blockers

**Already met:**
- `torch-geometric>=2.3.0` in `setup.py` for architectures C/D.
- `find_best_hybrid` for rep selection (Phase 3 uses this per regime).
- `assign_component_groups` for split generation.
- BH and Suzuki loaders with per-slot SMILES in `reaction_yields.py`.

**New code needed:**
- Architectures B/C/D as sklearn-compatible or torch-native modules (Phase 1). Minimum viable: a small torch Module with per-slot encoders + mixer/attention head.
- Uncertainty quantification wrappers (Phase 1.3).
- Simulated AL loop harness (Phase 2). Can wrap BoTorch or write minimal.
- Regime clustering + Bayesian posterior over hybrid library (Phase 3).

**Decisions still needed from user:**
- Scope: Phase 1 only (virtual model study), Phase 1+2 (lean DT), or Phase 1+2+3 (full)?
- Architecture preference: start at A for baseline, proceed to B/C/D? Or skip to B since A is "what we have"?
- Uncertainty method preference: GP (classical, familiar) or deep ensemble (modern, scales)?

---

## 11. What's out of scope

- **Real-time hardware twin.** We don't have a wet lab; this is simulated throughout.
- **Spectra-based mechanism inversion (DTCS-style).** Different problem; we're optimizing yield, not refining mechanisms.
- **Retrosynthesis.** Separate domain.
- **Multi-step reaction planning.** Single-step yield only.

---

## 12. Revised cross-references

- `CGR_EMBEDDING_RESEARCH.md` — CGR work still matters as a *feature source* for the virtual model (Phase 1). Phase 3 (CGR graphormer) still blocked on server run. But CGR is not the DT story; it's a rep choice.
- `OPIG_COLLAB_ROADMAP.md` — C6 King-Smith component splits are the eval frame for P1.1.
- `tests/experiments/opig_cgr_comparison.py` — measures whether CGR reps are worth including in the Phase-1 rep pool. Independent of DT framing.
- `tests/experiments/opig_kingsmith_component_splits.py:198` — `assign_component_groups` reusable for P1.1 evaluation.
- `src/kirby/datasets/reaction_yields.py` — per-slot SMILES in the `components` list per reaction.

---

## 13. Literature references

- Grieves, M. "Digital Twin: Manufacturing Excellence through Virtual Factory Replication" (2014)
- ISO 23247-1:2021. Digital twin framework for manufacturing. https://www.iso.org/standard/75066.html
- National Academies (2023). *Foundational Research Gaps and Future Directions for Digital Twins*. https://www.nationalacademies.org/read/26894/chapter/5
- Kapteyn, M. G., Knezevic, D. J., Willcox, K. E. "Data-driven physics-based digital twins via a library of component-based reduced-order models." *Int J Numer Methods Eng* (2022). https://kiwi.oden.utexas.edu/papers/Digital-twin-reduced-model-Kapteyn-Willcox.pdf
- Kapteyn & Willcox. "From Physics-Based Models to Predictive Digital Twins via Interpretable Machine Learning." https://arxiv.org/abs/2004.11356
- DTCS. *Nature Comp Sci* 5, 793–800 (2025). https://www.nature.com/articles/s43588-025-00857-y
- Gao et al. "Guided Bayesian Optimization: Data-Efficient Controller Tuning with Digital Twin." (2024) https://arxiv.org/html/2403.16619v2
- AlphaFlow. *Nat Commun* 14 (2023). https://www.nature.com/articles/s41467-023-37139-y
- RoboChem-Flex. *Nature Synthesis* (2026). https://www.nature.com/articles/s44160-026-01053-0
- MATTERIX. *Nature Comp Sci* (2025). https://www.nature.com/articles/s43588-025-00924-4
- Rasheed, A., San, O., Kvamsdal, T. "Digital Twin: Values, Challenges and Enablers From a Modeling Perspective." IEEE Access (2020).
- Yasin, A. S. Research statement and current projects. Johns Hopkins University Institute for Assured Autonomy / Engineering for Professionals (2023). DT framing for autonomous cyber defense, twins for air systems, risk and assurance of autonomous agents, CPS formal methods. Primary source for the assurance-and-resilience framing used in §7 cross-cutting axis. https://yasinresearch.com/research.html ; https://iaa.jhu.edu/collaboration/alhassan-yasin/
- Ahmed, J. & Yasin, A. S. "Graph Neural Operators for Classification of Spatial Transcriptomics Data." arXiv:2302.00658 (2023). Methodologically adjacent — neural operators learn mappings between function spaces rather than fixed-dim features, offering resolution-invariance claimed useful across variable-quality data. Relevant as an architecture variant for §5.1 Option D if resolution-invariance across reaction datasets becomes a goal.
- Yasin, A. S. & Gill, P. S. "Reinforcement Learning Framework for Quantitative Trading." ICAIF FM4TS Workshop, arXiv:2411.07585 (2024). Methodologically adjacent — RL agent with domain indicators and risk-aware decision-making under uncertainty. Not a direct transfer (RL vs BO, finance vs chemistry), but the risk-managed-decision structure informs §7 A3 (risk-based decision boundary) at a framing level.

---

## 14. Uncertainty quantification — package, methods, and literature

Cross-references Choice 2 (§6) and P1.3 (§7). Consolidated here so decisions are not re-derived from scratch in future sessions.

### 14.1 Package decision: `crepes`

Two serious candidates for conformal/CV+ in Python as of 2026:

- **crepes** (Boström, Stockholm University). Regression-focused. Standard, normalized, and Mondrian conformal regressors and predictive systems out of the box. sklearn-compatible. MLR Proceedings 2022/2024.
- **MAPIE** (scikit-learn-contrib, INRIA / Quantmetry). Broader scope — regression, classification, time series, LLM-as-judge (per 2026 roadmap). Bigger ecosystem; slower on small-data regression.

Packaging benchmarks: **crepes ≈10× faster on small regression tasks** (the regime for HTE plates). Crepes' normalized conformal maps directly to the applicability-domain intuition chemists already use — a difficulty estimator (kNN distance or a secondary residual-predictor) widens intervals in sparse regions of feature space and tightens them in dense regions. Mondrian conformal stratifies CP per category, a clean fit for the component-DT framing (per-ligand-class or per-aryl-halide-scaffold intervals).

**Verdict: `crepes`** for this project. Manual CV-based conformal is not worth the ~200 LOC of re-implementation relative to a small, permissive-license dependency.

### 14.2 Ridge is out (literature-confirmed)

The 2024 ACS Omega QSAR CP review and the Feb 2025 cheminformatics CP review both enumerate RF, DNN (MLP), and gradient boosting (XGBoost / LightGBM) as the canonical CP base models. Ridge is absent from both. Reasons: (a) `BayesianRidge` gives closed-form posterior variance already; (b) linear rigidity leaves little slack for conformalization to exploit.

RF has a structural advantage noted in the ACS Omega review: OOB data already constitutes a random split, so explicit train/calibration partitioning isn't required. Worth exploiting when RF is the base model.

### 14.3 Wrapper application patterns (literature)

| Wrapper | Canonical base models | Typical pattern |
|---|---|---|
| Deep ensemble (Lakshminarayanan 2017) | MLP × N=5 re-seeds | Do not double-wrap RF (already an ensemble). Marginal value on XGBoost (seed variance small). |
| Split conformal | RF, XGBoost, MLP | Inner train/calibration split per outer fold. Simple, interpretable. |
| CV+ (Barber et al. 2021) | Same | Uses k-fold structure directly — nonconformity scores from each fold's held-out predictions on its own training data. No architectural change to KIRBy's splits. |
| Normalized conformal | Same + difficulty estimator | Difficulty ≈ kNN distance OR secondary model predicting \|residual\|. Chemistry-appropriate: tight intervals in-domain, wide intervals for novel scaffolds. |
| Mondrian conformal | Same + stratification | Stratify by ligand class, aryl-halide scaffold, or substrate family. Aligns with the component-DT framing. |

For P2.1's BO acquisition, the surrogate needs μ + a variance-like signal. NGBoost, GP, and deep ensemble produce this directly. Conformal produces **intervals**, not variances — usable for UCB (upper bound of the interval) but **not EI** without extra structure (e.g., Gaussianizing the interval half-width, or switching to an interval-based acquisition).

### 14.4 Integration with KIRBy's k-fold design

KIRBy uses GroupKFold (typically 5-fold) with scaffold or component-leave-out groups. Conformal needs a held-out calibration set beyond train/test. Two clean options:

- **Split conformal**: within each outer fold, further split the training portion into train + calibration. Cost: smaller effective training set (≈20% loss).
- **CV+**: uses the fold structure directly. No architectural change required.

CV+ is the tighter fit for KIRBy and is supported in `crepes`. Recommended.

### 14.5 Sources

- **crepes package**: Boström, H. "Conformal Prediction in Python with crepes." *MLR Proceedings* (2024). https://proceedings.mlr.press/v230/bostrom24a.html | https://github.com/henrikbostrom/crepes
- **crepes original paper**: Boström, H. "crepes: a Python Package for Generating Conformal Regressors and Predictive Systems." *MLR Proceedings* 179 (2022). https://proceedings.mlr.press/v179/bostrom22a/bostrom22a.pdf
- **MAPIE package**: scikit-learn-contrib. https://github.com/scikit-learn-contrib/MAPIE | https://mapie.readthedocs.io/
- **CP for QSAR (2024)**: "Development and Evaluation of Conformal Prediction Methods for Quantitative Structure–Activity Relationship." *ACS Omega* (2024). https://pubs.acs.org/doi/10.1021/acsomega.4c02017
- **CP in cheminformatics (2025 review)**: "Conformal Prediction-based Machine Learning in Cheminformatics: Current Applications and New Challenges." *ChemRxiv* (Feb 2025). https://chemrxiv.org/engage/chemrxiv/article-details/679783486dde43c90894415d
- **CPSign (2024)**: "CPSign: conformal prediction for cheminformatics modeling." *J Cheminform*. https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00870-9
- **Deep ensemble canonical**: Lakshminarayanan et al. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS* (2017).
- **CV+**: Barber et al. "Predictive inference with the jackknife+." *Annals of Statistics* (2021).
- **Normalized conformal**: Papadopoulos et al. "Normalized nonconformity measures for regression conformal prediction." (2008); Johansson et al. (2014).
- **Awesome conformal list** (curated): https://github.com/valeman/awesome-conformal-prediction
- **Molecular UQ benchmarks**: Scalia et al., *J Chem Inf Model* (2020); Hirschfeld et al., *J Chem Inf Model* (2020); Busk et al., *Mach Learn Sci Tech* (2023).
