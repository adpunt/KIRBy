# CGR (Condensed Graph of Reaction) Embedding Research

Status: RESEARCH / PLANNING — not yet implemented.

## Background

### What is a CGR?

A Condensed Graph of Reaction (CGR) superimposes reactant and product molecular graphs into a **single graph** where bonds that change during the reaction are annotated as "dynamic bonds." A dynamic bond has two states: its order in the reactants and its order in the products. For example, a bond that is broken during the reaction has `order=1, p_order=None` (single bond → no bond). A bond formed has `order=None, p_order=1`.

Standard reaction representations in KIRBy currently treat reactions as pairs:
- `reactant_ecfp4` / `product_ecfp4`: fingerprints of each side separately
- `diff_ecfp4`: product FP minus reactant FP (captures change as a vector difference)
- `drfp_2048`: symmetric difference of circular substructure sets between sides
- `rxnfp` / `reactiont5`: transformers that take `reactants>>products` strings

CGR is fundamentally different: it encodes the **transformation itself** as a first-class molecular-like object, not as a difference of two objects. This means:
- The reaction center (where bonds change) is explicitly annotated
- Atom environments near the reaction center include dynamic bond information
- The CGR can be serialized to a SMILES-like string ("CGR-SMILES") with special dynamic bond tokens

### Key References

- **CGRtools / Chython**: Nugmanov et al. Python library for CGR manipulation. CGRtools is the older version; Chython is the maintained successor. Both produce `CGRContainer` objects from reaction SMILES.
  - GitHub: https://github.com/cimm-kzn/CGRtools (legacy), https://github.com/chython/chython (current)

- **Learned CGR representations**: Nugmanov et al., JCIM 2022 — "Machine Learning of Reaction Properties via Learned Representations of the Condensed Graph of Reaction." Trained GNN on CGR graphs for reaction property prediction.
  - DOI: 10.1021/acs.jcim.1c00975

- **CGR-SMILES (NeurIPS 2025)**: "CGR-SMILES: a compact and universal sequence representation for chemical reaction modeling." Introduced a canonical string serialization of CGR and showed that sequence models trained on CGR-SMILES outperform models on standard reaction SMILES for several reaction tasks.
  - https://neurips.cc/virtual/2025/133939

- **DOPtools**: Sidorov et al., Digital Discovery 2025 — CircuS (circular) and ChyLine (linear) CGR fragment descriptors. sklearn-compatible.
  - GitHub: https://github.com/POSidorov/DOPtools

- **Chytorch-rxnmap**: Pretrained Bidirectional Graphormer on USPTO+Pistachio for atom-mapping. CLS token can be extracted as a reaction embedding.
  - GitHub: https://github.com/chython/chytorch-rxnmap

- **YieldNet**: Ghosh et al., ICML 2025 — Differentiable CGR-based yield prediction.

### What does NOT work

- **RDKit on CGR**: RDKit cannot parse or process CGR graphs. Dynamic bonds are not in RDKit's data model. Confirmed by DOPtools authors: "since RDKit or Mordred do not support CGR structures, the calculation of fingerprints or physico-chemical parameters for CGRs is not implemented."

- **ChemBERTa / MolFormer on CGR-SMILES strings**: These were pretrained on standard SMILES. CGR-SMILES contains dynamic bond tokens (e.g., `[->.]`, `[.>-]`) that the tokenizer has never seen. The BPE tokenizer would break them into meaningless sub-tokens. **This produces garbage embeddings**, not meaningful chemical representations.

- **rxnfp on CGR-SMILES**: rxnfp expects standard `reactants>>products` format, not CGR notation.

- **`CGRContainer.morgan_fingerprint()`**: Does not exist. Chython's `MoleculeContainer` has Morgan fingerprint methods (via the `Fingerprints` mixin), but `CGRContainer` does NOT inherit these mixins. The standard fingerprint API does not support dynamic bonds.

---

## Candidate CGR Representations

### Option A: CGR Morgan Fingerprint (manual implementation)

**What:** Run standard Morgan/ECFP circular hashing on the CGR graph, but extend the atom/bond invariants to include dynamic properties.

**How:**
1. Parse reaction SMILES → CGR via chython (`ReactionContainer.compose()`)
2. Initialize atom hashes: `hash(isotope, atomic_number, charge, p_charge, is_radical, p_is_radical)` — note both reactant AND product properties
3. Initialize bond hashes: `hash(order, p_order)` — where `None` indicates bond absent on that side
4. Run standard Morgan iteration (radius 2-3): update atom hash by XOR with sorted neighbor hashes
5. Fold final atom hashes into a fixed-length bit vector (2048 bits)

**Output:** `(n_reactions, 2048)` uint8 array. Deterministic, no GPU, fast.

**SMILES counterparts:** `diff_ecfp4`, `drfp_2048`, `reactant_ecfp4`. These capture similar information (reaction substructures) but from different angles:
- `diff_ecfp4` = product bits minus reactant bits (signed difference, loses non-changing context)
- `drfp_2048` = symmetric set difference of substructure hashes (binary, no context)
- CGR Morgan = substructure hashes that natively include the transformation context (dynamic bonds are part of the hash, not a post-hoc difference)

**Expected correlation with existing reps:** Moderate-to-high with diff_ecfp4 and drfp_2048. The unique signal is in the dynamic bond hashes — bits that encode "this atom is adjacent to a bond that changed" rather than "this atom is in the product but not the reactant." Whether that distinction matters for yield prediction is the empirical question.

**Dependencies:** `chython` (pip install chython). No GPU.

### Option B: DOPtools CircuS/ChyLine Fragment Descriptors

**What:** Enumerate all circular (CircuS) or linear (ChyLine) CGR substructure fragments up to a given radius/length. Each unique fragment becomes a column; values are counts.

**How:**
1. Parse reactions → CGR via chython
2. Use DOPtools `CircuS(lower=1, upper=3)` or `ChyLine(lower=2, upper=4)` transformers
3. `.fit_transform(cgr_list)` produces a sparse count matrix

**Output:** Sparse matrix, shape `(n_reactions, n_unique_fragments)`. Dimensionality is dataset-dependent (could be 500-5000+ columns).

**SMILES counterparts:** No direct counterpart. These are CGR-specific fragment vocabularies that enumerate actual substructure SMILES, not hashed fingerprints. Closest analog might be `concat_pdv` or `component_descriptors` in spirit (interpretable feature sets), but the features are completely different.

**Complication for KIRBy:** The fragment vocabulary is learned from data (`.fit()` step). This means:
- Must fit on training data per fold, transform train+val separately
- Not a simple `create_*()` → array function; needs the factory rep pattern
- Fragment count can vary across folds (though you can fit on all data for screening, then per-fold for final eval)

**Dependencies:** `DOPtools` (pip install doptools), `chython`.

### Option C: Chytorch Graphormer CLS Embedding

**What:** Extract the CLS token from a pretrained Bidirectional Graphormer that was trained on ~2M reactions (USPTO + Pistachio) for atom-mapping.

**How:**
1. Parse reactions → chython `ReactionContainer` objects
2. Feed through `chytorch.zoo.rxnmap.Model.pretrained()`
3. Extract `output[:, 0]` (CLS token) as fixed-length embedding

**Output:** `(n_reactions, D)` float32 array, where D is likely 256 or 512 (Graphormer hidden dim).

**SMILES counterparts:** `rxnfp` (256D, RXNBERT trained on ~1M reactions). Both are neural reaction embeddings from pretrained transformers. Key difference: rxnfp uses a SMILES-based BERT; Graphormer operates on the reaction graph structure directly. The graph-based approach should capture topological information that sequential models miss.

**Concerns:**
- Model was trained for **atom-mapping**, not property prediction. Transfer quality to yield prediction is unknown.
- Requires `chytorch`, `chytorch-rxnmap`, and GPU for reasonable speed.
- Model weights may be large.

**Dependencies:** `chytorch`, `chytorch-rxnmap`, `torch`. GPU recommended.

### Option D: CGR-SMILES Sequential Embedding (fine-tuned LM)

**What:** Take a pretrained SMILES language model and fine-tune or continue-pretrain it on CGR-SMILES strings.

**How (two sub-approaches):**

**D1. Extend tokenizer + continued MLM pretraining:**
1. Take ChemBERTa's BPE tokenizer
2. Add special tokens for CGR dynamic bond notation (e.g., `[->.]`, `[.>-]`, `[->-]`, `[->=]`, etc.)
3. Convert a large reaction corpus (USPTO, ~1M+ reactions) to CGR-SMILES via chython
4. Continue MLM pretraining on the CGR-SMILES corpus (new tokens get learned embeddings, existing SMILES tokens keep their pretrained weights)
5. Extract embeddings via mean pooling

**D2. Train BPE tokenizer from scratch on CGR-SMILES:**
1. Convert large reaction corpus to CGR-SMILES
2. Train new BPE tokenizer on CGR-SMILES strings
3. Train transformer (small, e.g., 6-layer, 256-hidden) with MLM on the corpus
4. Extract embeddings

**Output:** `(n_reactions, 768)` for D1 (ChemBERTa hidden dim) or `(n_reactions, 256)` for D2 (custom).

**SMILES counterparts:** `rxnfp` (RXNBERT on reaction SMILES), `reactiont5` (T5 on reaction SMILES). These are the direct analogs — sequential models on reaction strings — but using standard `reactants>>products` notation rather than CGR-SMILES. The hypothesis is that CGR-SMILES provides a more information-dense string (single notation for the transformation) that a language model can learn from more efficiently.

**This is the most interesting option scientifically** but also the most expensive:
- Needs a large unlabeled CGR-SMILES corpus (USPTO conversion: ~1M reactions, feasible but requires compute)
- Pretraining time: hours to days depending on model size
- Unknown if the pretraining corpus is large enough for meaningful representations

**Dependencies:** `chython`, `transformers`, `torch`, GPU, large reaction corpus.

---

## Design Questions

### Should CGR reps replace or augment existing SMILES reps?

**Augment, not replace.** CGR and SMILES reps encode overlapping but distinct information. The whole point of KIRBy's hybrid framework is to combine complementary representations — adding CGR reps to the pool gives the allocation methods more material to select from. If CGR Morgan and diff_ecfp4 are 95% redundant, the greedy selector will pick one and skip the other. No harm done.

### Is high correlation between CGR and SMILES counterparts a problem?

**No, for two reasons:**

1. **The allocation method handles it.** Greedy forward selection evaluates marginal gain. If rep B adds no information beyond what rep A already provides, its marginal gain is ~0 and it won't be selected. The hybrid won't be worse for having correlated candidates in the pool — it just won't select both.

2. **The interesting question is whether CGR provides marginal gain OVER its SMILES counterpart.** If CGR Morgan FP adds even a few unique bits (from dynamic bond context) that diff_ecfp4 misses, and those bits help predict yield, then the hybrid will select them. This is a clean test of whether CGR's explicit reaction-center encoding carries unique signal.

### Which reps should be added to surveys vs tested in a dedicated experiment?

**Proposed split:**

| Rep | Survey (add to pool) | Dedicated experiment |
|-----|---------------------|---------------------|
| A. CGR Morgan FP | Yes — add like any other fingerprint | Also test in comparison experiment |
| B. DOPtools CircuS/ChyLine | No — needs factory pattern (fit/transform) | Dedicated experiment only |
| C. Chytorch Graphormer CLS | Yes — add like rxnfp | Also test in comparison experiment |
| D. CGR-SMILES sequential | No — requires pretraining first | Dedicated experiment after pretraining |

**The comparison experiment:** A dedicated experiment (could be OPIG or standalone) that directly compares CGR reps vs their SMILES counterparts on BH datasets:
- CGR Morgan FP vs diff_ecfp4 vs drfp_2048 (fingerprint family)
- Chytorch Graphormer CLS vs rxnfp vs reactiont5 (neural embedding family)
- (If D is built) CGR-SMILES LM vs rxnfp vs reactiont5 (sequential embedding family)
- Then: does adding CGR reps to the hybrid pool improve the final hybrid beyond what SMILES-only reps achieve?

This directly tests the thesis that CGR's explicit reaction-center encoding provides complementary information.

### Is training a sequential model on CGR-SMILES feasible?

**Feasible but non-trivial.** Requirements:
1. A CGR-SMILES corpus: convert USPTO (~1M reactions, public) to CGR-SMILES via chython. This is a one-time batch job. Many reactions will fail CGR conversion (unmapped atoms, invalid SMILES) — expect 60-80% success rate.
2. Tokenizer: either extend existing (quick) or train new BPE (better but slower).
3. Pretraining: continued MLM on ChemBERTa with extended tokenizer is the cheapest option. 10K-50K training steps on the CGR corpus, maybe 2-4 hours on a single GPU.
4. Embedding extraction: standard mean-pooling, same as existing ChemBERTa/MolFormer pipeline.

The question is whether the pretrained weights transfer meaningfully. Standard SMILES tokens (atom symbols, ring closures, branches) will keep their pretrained semantics. The new dynamic bond tokens will learn from scratch. If the corpus is large enough, they should learn meaningful representations.

**Alternative shortcut:** Skip MLM pretraining entirely. Fine-tune ChemBERTa with extended tokenizer directly on BH yield prediction (LoRA or full fine-tune). This is simpler and directly optimizes for the downstream task, but the embeddings may be less generalizable. We already have a LoRA fine-tuning pipeline (`molecular_lora.py`).

---

## Implementation Plan

### Phase 1: CGR Morgan Fingerprint (Option A) — IMPLEMENTED

**Status: DONE.** `create_cgr_morgan()` added to `reaction.py`, wired into survey + experiment dispatch.

1. `create_cgr_morgan(rxn_smiles, radius=2, n_bits=2048)` in `src/kirby/representations/reaction.py`
   - Pipeline: rxnmapper AAM → chython parse → compose CGR → manual Morgan iteration with dynamic bond invariants → fold into 2048-bit vector
   - Deterministic hashing via blake2b (not Python `hash()`)
   - Zero-vector fallback for failed conversions
2. Wired into `_rxn_generators` in `_shared.py` and `_load_reaction` in `run_baseline_survey.py`
3. Dependencies: `chython>=1.0`, `rxnmapper>=0.3.0` (added to setup.py + SETUP.md)
4. **Next: run survey on BH datasets on server to get baseline performance**

### Phase 2: Chytorch Graphormer Embedding (Option C) — IMPLEMENTED

**Status: DONE.** `create_chytorch_graphormer()` added to `reaction.py`, wired into survey + experiment dispatch.

1. `create_chytorch_graphormer(rxn_smiles, batch_size=20)` in `src/kirby/representations/reaction.py`
   - Pipeline: chython parse → canonicalize → ReactionEncoderDataset → collate → parent-class forward (not atom-mapping forward) → CLS token at position 0
   - Output: `(n_reactions, 1024)` float32 (Graphormer d_model = 1024)
   - Does NOT need rxnmapper — the Graphormer handles unmapped reactions natively
   - Includes Python 3.11 compatibility patch for chytorch's NamedTuple+DataTypeMixin issue
   - Zero-vector fallback for failed parses
2. Wired into `_rxn_generators` in `_shared.py` and `_load_reaction` in `run_baseline_survey.py`
3. Dependencies: `chytorch`, `chytorch-rxnmap` (added to setup.py + SETUP.md)
4. **Next: run survey on BH datasets on server to get baseline performance**

### Phase 3: Comparison Experiment — IMPLEMENTED

**Status: DONE.** `tests/experiments/opig_cgr_comparison.py` — dedicated CGR vs SMILES comparison.

1. `tests/experiments/opig_cgr_comparison.py` — full experiment file (~450 lines)
   - Datasets: buchwald_hartwig, bh_doyle_plates, suzuki (all 3 reaction datasets)
   - Two comparison families:
     - Fingerprint: cgr_morgan vs diff_ecfp4 vs drfp_2048
     - Neural: chytorch_graphormer vs rxnfp vs reactiont5
   - Individual rep evaluation: all reps × DOWNSTREAM_MODELS × GroupKFold CV
   - Dual hybrid pool comparison: SMILES-only vs SMILES+CGR
   - CGR feature selection tracking: which CGR reps selected by allocator, how many features
   - Linear CKA correlation matrix (Kornblith et al. 2019) between family reps
   - Family-level analysis: best CGR vs best SMILES within each family
   - Uses `find_best_hybrid` (cached), `INTERNAL_EVALUATORS`, `SCREENING_BUDGETS`
2. SLURM script: `tests/slurm_scripts/submit_cgr_comparison.sh` (one GPU job per dataset)
3. Output: `tests/results/comparisons/opig_cgr_comparison.json`
4. **Next: run on server after installing chytorch + chytorch-rxnmap**

### Phase 4: CGR-SMILES Sequential Embedding (Option D) — IMPLEMENTED

**Status: DONE.** Two variants implemented: frozen MLM-pretrained embeddings + optional LoRA fine-tune.

1. `src/kirby/datasets/uspto.py` — USPTO-STEREO download + parse (~1M atom-mapped reactions from Zenodo)
2. `src/kirby/finetuning/cgr_smiles_pretrain.py` — full pipeline:
   - Corpus generation: USPTO → CGR-SMILES via chython compose + str (multiprocessing, 60-80% success)
   - Tokenizer extension: 30 dynamic bond tokens added to ChemBERTa-77M-MLM's RobertaTokenizer
   - Continued MLM pretraining: 50K steps, lr=5e-5, fp16, DataCollatorForLanguageModeling
   - Runnable as CLI: `python -m kirby.finetuning.cgr_smiles_pretrain --stage all`
3. `create_cgr_smiles_lm()` in `reaction.py` — frozen mean-pooled embeddings (384D)
   - Pipeline: rxn_smiles → rxnmapper AAM → CGR-SMILES → tokenize → encode → mean-pool
   - Auto-detects MLM checkpoint; falls back to base ChemBERTa if no checkpoint
4. `src/kirby/finetuning/cgr_smiles_lora.py` — optional LoRA fine-tune factory
   - Per-fold training with cached results, follows molecular_lora.py pattern
   - Uses MLM checkpoint as base (or base ChemBERTa fallback)
5. SLURM script: `tests/slurm_scripts/submit_cgr_smiles_pretrain.sh` (corpus + pretrain)
6. Wired into `_rxn_generators` in `_shared.py` and `_load_reaction` in `run_baseline_survey.py`
7. **Next: run pretraining on server, then survey**

### Phase 5 (optional): DOPtools Fragment Descriptors (Option B)

**Priority: LOW. Estimated effort: 3-4 hours.**

1. Add factory rep pattern for CircuS/ChyLine (needs fit/transform per fold)
2. This is more complex than other reps due to the variable-length vocabulary
3. Defer unless Phase 1-3 results show CGR carries strong unique signal

### Dedicated CGR Experiments (Forward Compatibility)

**Priority: HIGH (after survey results for Phase 1).**

Two dedicated reaction experiments to be added as forward compatibility items:

1. **CGR vs SMILES counterpart comparison:** Head-to-head evaluation of CGR reps against their SMILES-based analogs on all BH datasets. Fingerprint family: CGR Morgan vs diff_ecfp4 vs drfp_2048. Neural family (Phase 2+): Chytorch Graphormer CLS vs rxnfp vs reactiont5. Tests whether CGR adds marginal gain when both CGR and SMILES reps are in the hybrid pool, and which family (fingerprint vs neural) benefits more from CGR encoding.

2. **CGR Morgan standalone evaluation:** Since CGR Morgan FP is a novel (synthesized) method — not a published fingerprint — it needs deeper analysis beyond survey ranking. Specifically: bit overlap analysis with existing reps (Tanimoto/Jaccard between cgr_morgan and diff_ecfp4/drfp_2048), breakdown of which bits come from dynamic vs static bond environments, performance comparison across BH and Suzuki reaction types, and sensitivity to atom-atom mapping quality (compare rxnmapper vs chytorch-rxnmap AAM backends).

---

## Decision Points for Discussion

1. **Start with Phase 1 only?** CGR Morgan FP is the cheapest test of whether CGR adds value. If it's redundant with diff_ecfp4/drfp_2048, we skip the rest. If it shows unique signal, proceed to Phase 2-3.

2. **Is Phase 4 (sequential LM) worth the effort?** The NeurIPS 2025 paper showed CGR-SMILES sequential models outperform standard reaction SMILES models, but on different tasks. Our BH datasets are small (~4K reactions) — is there enough signal for fine-tuning?

3. **Should the comparison experiment be an OPIG item or a complexity experiment?** It tests whether CGR improves hybrids (framework question) AND whether CGR reps compete with SMILES reps (rep comparison question).

4. **Corpus for pretraining (Phase 4):** USPTO is public and large enough. Do we have it downloaded already, or is this a new data acquisition step?
