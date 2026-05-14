# Cross-Species KIRBy Roadmap

Status: **DRAFT v1 FOR USER REVIEW** (2026-05-12). v0 was rewritten after user feedback clarified that (a) KIRBy itself is the rep-finder — no manual rep additions needed, (b) a new `cross_species_ness.py` loader is required, not an in-place generalization of `humanness.py`, (c) the headline is "meaningfully analyze animal data for R&D purposes," not the methodological framing.

This is a **cross-species extension to existing KIRBy domains** (antibody, QSAR/ADMET, DTA), not a spinoff repo. It lives in the main `KIRBy` tree alongside the noise/federated/OPIG roadmaps.

---

## 1. Framing

KIRBy already solves the rep-selection problem. The cross-species extension is therefore *not* about adding species-specific representations by hand — it's about wiring new species-labeled datasets through the existing `find_best_hybrid` engine and using the resulting per-dataset hybrids to power R&D-meaningful analyses for animal health.

The lead validation point: humanness_therapeutic hits 0.92 AUROC with KIRBy at budget 100; the same machinery, fed dog/cat-labeled sequences, should land an equivalent canineness/felineness score.

**Three R&D analyses to land in v1:**
1. **Canineness / felineness scoring** of antibody candidates (X-Ab)
2. **Species-stratified ADMET** prediction (X-Mol)
3. **Cross-species off-target binding** via orthologue substitution (X-DTA)

**Two methodology + data tracks supporting v1 (added 2026-05-13):**
- **X-PT** — cross-species pretraining. Pretrain on rich (human / rodent / mammalian-all / human-microsomes) corpora, fine-tune on small vet targets. Six variants (PT-α–ζ). Methodology layer for X-Mol; foundation-model variant runs first as anchor.
- **X-Data** — non-ChEMBL data source expansion. DrugCentral / PubChem BioAssay / CTD / openFDA-CVM / Polaris DLM-FLM / cat UGT1A6 literature / OCHEM / IUPHAR. Unblocks X-PT corpora and gives X-Mol real cat data (literature CSV; ChEMBL has none).

**Deferred to v2:** cross-species ADA / immunogenicity (data-gated), caninization generation (generative-models territory, not current methods).

---

## 2. What exists for humans today

Audit confirms the human-side infra these tracks build on:

**Antibody (X-Ab base):**
- `src/kirby/datasets/humanness.py` — binary human vs non-human (`species` and `therapeutic` presets). c11 hit 0.92 AUROC.
- `src/kirby/representations/antibody.py` — AbLang2, AntiBERTy, ESM-2 (+ variants concat/CDR3/early/layer24), ESM-C, ESM-3, SaProt, Ab-RoBERTa, Ab-RoBERTa CDR, MINT, AAC, one-hot, BLOSUM.
- ANTIBODY_REPS list + StratifiedGroupKFold + find_best_hybrid wiring.

**Molecular (X-Mol base):**
- `caco2.py`, `herg.py` (binary clf + Ki regression), `cox2.py`, `drd2.py`, `chembl_qsar.py`, `chembl_bioactivity.py`.
- Polaris ADME-Fang: `adme_hlm` (3087), `adme_rlm` (3054), `adme_mdr1` (2642), `adme_solubility` (2173).
- Full MOLECULAR_REPS (ECFP4, mordred, mordred_pca50, avalon, ChemBERTa, MolFormer, GNN, mordred, etc.) + ScaffoldGroupKFold.

**DTA (X-DTA base):**
- `dta.py` — BindingDB_Kd (~52K pairs, 10K drugs × 1.4K proteins), cold-drug eval mode.
- `pdbbind.py` — protein-ligand binding.
- `src/kirby/representations/protein.py:embed_esm2()` accepts **arbitrary** amino-acid sequences — dog/cat target embedding is free.

**Cross-cutting:**
- `find_best_hybrid(...)` is domain-agnostic; accepts arbitrary `base_reps` + `augmentations` dict.
- NoiseInject + LNL methods (`utils/noise_estimation.py`, `utils/lnl_methods.py`) are label-agnostic.
- BNN for UQ; **MAPIE / conformal / evidential are not in the repo** — adding `src/kirby/uq/conformal.py` is the only genuinely new infra Phase 1 needs.

---

## 3. Three tracks

### Track X-Ab — Canineness / felineness / X-ness scoring

**R&D question this answers:** given a candidate antibody, how native is it to the target species? Useful for caninization QC (does Librela score as canine?), for screening transgenic-platform output (Petmedix Ky9 / Felyne, Vetigenics CANIBODY), and for catching engineered constructs that drift away from native repertoire.

**Data:**
- Dog: DoggifAI (Zenodo 10.5281/zenodo.15125375, ~430k chains) → `/Volumes/seagate/kirby/cross_species/canine_antibodies/`
- Human / mouse / rhesus / camelid: existing OAS Zenodo (already used by `humanness.py`)
- Cat / cow / horse seeds: IMGT germlines + NCBI SRA mining of published BCR-seq studies → `/Volumes/seagate/kirby/cross_species/<species>_antibodies/`
- Therapeutic eval set: caninized (Librela, Solensia, Cytopoint, Befrena, Lenivia, Portela, CPMA), feline (Solensia, Portela), human (any) — held out as "should score high on own species"

**New loader:** `src/kirby/datasets/cross_species_ness.py`
- Signature: `load(target_species: str, mode: Literal['one_vs_rest','multinomial'] = 'one_vs_rest', source_species: list[str] | None = None, **kwargs)`
- Does **not** touch `humanness.py`.
- StratifiedGroupKFold by germline V-gene.

**Reps:** full ANTIBODY_REPS. find_best_hybrid uncapped at budgets {50, 100, 500}. The resulting hybrid IS the canineness rep — cached per `(target_species, mode)` key.

**Downstream R&D analysis:** rank a candidate set (or a clinic's caninized therapeutic) by predicted canineness with conformal intervals.

### Track X-Mol — Species-stratified ADMET

**R&D question this answers:** what's the dog/cat-specific hERG / clearance / permeability / Pgp profile of a small molecule? Bridges the cat UGT1A6 deficiency, dog CYP3A12, breed-MDR1 stories that the research memo flagged as commercially urgent for Lhasa.

**Data — additive, no new infra:**
- `chembl_qsar.py`, `chembl_bioactivity.py`, `cox2.py`, `drd2.py`, `herg.py` — add `organism: str = 'human'` kwarg. Default preserves current behaviour.
- ChEMBL filter applied at the `chembl_webresource_client` query level (`target_organism__in=...`).
- Cat UGT1A6 substrate data — hand-curated from Court et al. PMIDs, small CSV in repo `data/raw/`.
- Polaris microsomal stability — HLM/RLM exist; DLM (dog) and FLM (cat) added where ChEMBL has the data.

**Reps:** full MOLECULAR_REPS (no changes). Drug side untouched. Protein-target embedding uses the dog/cat orthologue ESM-2 embedding instead of human (free via `protein.py:embed_esm2()`).

**Species-as-token:** one-hot over `{human, dog, cat}` passed as a small augmentation feature to find_best_hybrid. No engine change needed.

**Splits:** ScaffoldGroupKFold (existing). 5-fold.

**Downstream R&D analysis:** for a candidate small molecule, return per-species predicted ADMET profiles with conformal intervals; flag any property where the cat or dog prediction diverges from human by >X (the "species liability flag").

### Track X-DTA — Cross-species off-target binding

**R&D question this answers:** if I have a hit at a human target, will it bind the dog or cat orthologue? Underpins cross-species safety pharmacology and lets a human discovery campaign generate vet-relevant compounds without re-screening.

**Data — no new datasets needed:**
- Use existing BindingDB_Kd pairs.
- For each (compound, human_target) pair, fetch the dog and cat orthologue sequence from UniProt (taxonomy 9615 dog, 9685 cat) via OrthoFinder where the mapping is ambiguous.
- Re-embed orthologue sequences with ESM-2 (free).

**Pipeline:**
- New loader `src/kirby/datasets/cross_species_offtarget.py` — same DTA harness, orthologue-substituted target column.
- Reps: full DTA stack (MOLECULAR_REPS for drugs, ESM-2 / ProtT5 / etc. for proteins).
- Cold-drug evaluation mode (already exists in `dta.py`) — leak-free transfer test.

**Downstream R&D analysis:** for a compound with known human-target affinity, predict dog and cat orthologue affinity with calibrated UQ; rank cross-species "safe" vs "liable" compounds.

### Track X-PT — Cross-species pretraining (added 2026-05-13)

**R&D question this answers:** vet ChEMBL is sparse (dog 1,055 activities across 10 targets; cat ~0; cow rich for in-vitro pharmacology only; horse cholinesterase-only). Direct cross-species QSAR/ADMET hits the small-data wall. Cross-species pretraining — leverage rich human / rodent data to learn chemistry priors, then fine-tune on small vet targets — is the canonical fix. KIRBy already has the C4 Imrie infra (`pretrain_chembl_dnn.py`, bulk ChEMBL → DNN → fine-tune); the X-PT track generalises that.

**Why this matters now:** literature shows fine-tune quality drops sharply below ~200 labels (MolPMoFiT, MTL-BERT). Dog COX-2 = 110, cow PTGS2 = 28, cow DRD2 = 406 sit exactly in that regime — pretraining is the only methodology that has a chance of beating from-scratch.

**Architectures (each is a separate experiment):**

- **PT-α — Same-target transfer.** Pretrain a DNN on human bulk ChEMBL via existing `pretrain_chembl_dnn.py`; fine-tune on dog/cow at the *same* target (COX-2, DRD2). Baseline: from-scratch dog/cow. Tests: does human chemistry transfer at a shared target?
- **PT-β — Multi-species + species token.** Joint training on all available species at all available targets, species_token + target_token as input. Held-out eval on vet rows. Tests TACTIC-style species-conditional multi-task.
- **PT-γ — Rodent → vet transfer.** Pretrain on rat + mouse ChEMBL only (no human). Fine-tune on dog/cow. Tests "evolutionary closeness" hypothesis: rodents are closer to dog/cat than human is for many endpoints.
- **PT-δ — Microsomal stability transfer.** Pretrain on Polaris HLM + RLM (microsomal clearance), fine-tune on dog DLM / cat FLM (X-Data adds the loaders). Tests organ-system-aligned transfer for ADMET.
- **PT-ε — Foundation-model fine-tune.** Use ChemBERTa / MolFormer / MolE / Polaris-MolE as frozen feature extractor (already KIRBy reps); small head trained on each vet target. Tests "pretraining is already done — just pick the right rep" hypothesis. This is the cheapest experiment and the obvious baseline.
- **PT-ζ — Multi-target multi-species joint.** One model, all KIRBy molecular data, species + target as joint tokens. End-to-end. Tests the maximally-pooled story; expensive.

**Cross-cutting evaluation protocol (all PT-*):**
- Splits: ScaffoldGroupKFold on the fine-tune dataset. Pretraining corpora are NOT in test split.
- Leakage check: exclude fine-tune-test InChIKeys from pretrain corpus (`exclude_inchikeys` already in `chembl_bioactivity.py`).
- Metric: RMSE + Spearman ρ (existing molecular defaults) per [[memory: domain-aware primary metrics]].
- Baselines (mandatory): (a) from-scratch on vet data only, (b) zero-shot human model applied to vet test, (c) KIRBy hybrid baseline (find_best_hybrid uncapped).
- UQ: conformal interval + BNN posterior on every prediction (Phase X-UQ deliverable).

### Track X-Data — Non-ChEMBL data source expansion (added 2026-05-13)

**Reality check from web survey 2026-05-13:** ChEMBL is one source. Vet R&D needs others. New loaders / cache layers below. **None require core pipeline changes.**

| Source | URL | Vet relevance | Loader status |
|---|---|---|---|
| **DrugCentral** | drugcentral.org | 1,805 bioactivities × 226 vet drugs × 804 targets. **Best vet-explicit source.** | NEW: `src/kirby/datasets/drugcentral.py` |
| **PubChem BioAssay** | pubchem.ncbi.nlm.nih.gov | 1.5M+ assays, includes non-human (e.g. AID 1094458 = horse BChE inhibition). Bigger and messier than ChEMBL. | NEW: `src/kirby/datasets/pubchem_bioassay.py` |
| **Guide to Pharmacology (IUPHAR/BPS)** | guidetopharmacology.org | 3,103 targets × 13,260 ligands. Cross-species: human/mouse/rat (no dog/cat/cow/horse). | NEW: `src/kirby/datasets/guide_to_pharm.py` (low priority — limited vet) |
| **Comparative Toxicogenomics Database (CTD)** | ctdbase.org | 17,100 chemicals × 54,300 genes × all species. Manually curated chem-gene interactions. **Best cross-species toxicology source.** | NEW: `src/kirby/datasets/ctd.py` |
| **openFDA CVM** (Animal AERs) | open.fda.gov | Adverse event reports for vet drugs. Real clinical signal (not bioactivity). Quarterly updates. | NEW: `src/kirby/datasets/openfda_cvm.py` (post-hoc evidence, not training data) |
| **OCHEM** | ochem.eu | Rat + mouse acute oral toxicity (validated PredAOT corpus). | NEW: `src/kirby/datasets/ochem.py` (rodent-pretrain corpus for PT-γ) |
| **Polaris DLM / FLM** | polarishub.io | Mentioned in §3 X-Mol but never built. Dog / cat liver microsomes. | NEW: `src/kirby/datasets/polaris_microsomes.py` |
| **Cat UGT1A6 literature CSV** | Court PMIDs | Cat-specific phase II metabolism (acetaminophen tox). | NEW: hand-curated `data/raw/cat_ugt1a6.csv` + loader |
| **DoggifAI** | zenodo.org/records/15125375 | Already in §3 X-Ab. Confirmed Zenodo URL. | Status: download is X0.1. |
| **cAb-Rep** | cab-rep.ngs.nih.gov | Curated antibody repertoire DB — check non-human coverage. | INVESTIGATE: may overlap OAS. |
| **Bovine Genome Database** | bovinegenome.elsiklab.missouri.edu | 258 bovine IG genes characterised. | INVESTIGATE: ortholog reference for X-Ab cow. |
| **NCBI SRA bovine/equine BCR-seq** | ncbi.nlm.nih.gov/sra | Source for §3 X-Ab cat/cow/horse seeds. | Already X0.4. |

**Pretraining corpora unlocked by X-Data:**
- DrugCentral vet bioactivities → small but vet-explicit, useful as PT-α/PT-β supplement.
- PubChem BioAssay non-human slice → potentially much larger than ChEMBL non-human (no current count; first task is to query).
- CTD → cross-species toxicogenomics for PT-δ (toxicology-focused transfer).
- OCHEM rodent → directly fuels PT-γ.

---

## 4. Cross-cutting infrastructure (one new module only)

`src/kirby/uq/conformal.py` — MAPIE wrapper for sklearn-compat regressors / classifiers, honoring the existing CV splitter contract. Chemistry variant `conformal_chem.py` respects scaffold splits and InChIKey dedup. This is **the only new infra v1 needs**. Everything else (loaders, reps, experiments) is additive to existing pipelines.

Note: BNN (`models/bnn.py`) and NGBoost (in DOWNSTREAM_MODELS_NOISE) stay as alternate UQ paths. The user picks which goes in the headline.

---

## 5. Phased plan

Numbering mirrors `FEDERATED_EXPERIMENT_ROADMAP.md`. **Each phase ends with a user review gate.**

### Phase X0 — Data acquisition (BLOCKS X-Ab, doesn't block X-Mol / X-DTA)

- **X0.1** DoggifAI → `/Volumes/seagate/kirby/cross_species/canine_antibodies/`. User-initiated download.
- **X0.2** IMGT germlines for dog/cat/cow/horse/pig/chicken/camelid (academic registration). → `data/raw/imgt_germlines/<species>/`.
- **X0.3** OAS snapshot version-pinned (already partly on disk).
- **X0.4** Cat/cow/horse BCR-seq from NCBI SRA — **bottleneck.** Either in-house assembly pipeline, OPIG collaboration, or scope-restrict v1 to dog-only.
- **X0.5** Therapeutic mAb eval set (Librela, Solensia, Cytopoint, Befrena, Lenivia, Portela, CPMA) — extract from `humanness.py` therapeutic preset where present, supplement from IMGT/mAb-DB.
- **X0.6** ChEMBL organism slices for dog and cat — cache to `data/cache/chembl_canine.parquet`, `chembl_feline.parquet`. Small enough for repo data dir.
- **X0.7** UniProt orthologue sequence dump for top ~500 BindingDB human targets, dog and cat — cache to `data/cache/orthologues_dog_cat.parquet`.

**Gate X0:** user confirms (a) v1 species scope (dog+cat minimum, ±cow ±horse), (b) which downloads user-runs vs. Claude-runs, (c) storage layout.

### Phase X-Ab — Canineness / felineness scoring

- **X-Ab.1** New file `src/kirby/datasets/cross_species_ness.py`. `target_species` + `mode` kwargs. StratifiedGroupKFold by germline V-gene.
- **X-Ab.2** New file `src/kirby/utils/antibody_numbering.py` (thin ANARCI / AbNumber wrapper) — needed only for V-gene grouping if existing reps don't already number internally. Skip if not strictly required.
- **X-Ab.3** New experiment `tests/experiments/cross_species/x_ab_canineness.py`. find_best_hybrid uncapped, budgets {50, 100, 500}, ANTIBODY_REPS, DOWNSTREAM_MODELS + BNN.
- **X-Ab.4** Therapeutic eval — score Librela, Solensia, Trastuzumab, etc. and confirm each scores high on its own species.
- **X-Ab.5** Downstream R&D analysis script — given a candidate sequence, return per-species nativeness scores with conformal intervals.

**Gate X-Ab:** survey results reviewed with user before any cross-species antibody dataset enters PIPELINE_READINESS_TODO tier tables.

### Phase X-Mol — Species-stratified ADMET (runs in parallel with X-Ab)

- **X-Mol.1** Add `organism: str = 'human'` to `chembl_qsar.py`, `chembl_bioactivity.py`, `herg.py`, `cox2.py`, `drd2.py`. Default preserves current behaviour.
- **X-Mol.2** New convenience loader `src/kirby/datasets/canine_cyp.py` — wraps `chembl_qsar(target='CYP3A12', organism='dog')` etc. for CYP1A, 2B, 2C, 2D, 3A. Cat UGT1A6 hand-curated CSV.
- **X-Mol.3** New loader `src/kirby/datasets/canine_pgp.py` — canine ABCB1 substrate / inhibitor data (literature + ChEMBL slice).
- **X-Mol.4** New experiment `tests/experiments/cross_species/x_mol_species_admet.py`. MOLECULAR_REPS, species-token augmentation, dog/cat target sequence (UniProt) → ESM-2, ScaffoldGroupKFold.
- **X-Mol.5** Downstream R&D analysis — given a small molecule, return per-species predicted ADMET profiles with conformal intervals, flag liability divergence.

**Gate X-Mol:** surveys reviewed with user before tier assignment.

### Phase X-DTA — Cross-species off-target panel

- **X-DTA.1** New loader `src/kirby/datasets/cross_species_offtarget.py` — orthologue-substituted target column over BindingDB_Kd.
- **X-DTA.2** UniProt orthologue lookup utility (`src/kirby/utils/orthologue_lookup.py`) — OrthoFinder for ambiguous cases, direct UniProt query for unambiguous taxonomy hits.
- **X-DTA.3** New experiment `tests/experiments/cross_species/x_dta_offtarget.py`. Cold-drug eval. Existing DTA reps.
- **X-DTA.4** Downstream R&D analysis — for a hit compound, return dog/cat orthologue affinity with calibrated UQ.

**Gate X-DTA:** surveys reviewed with user.

### Phase X-PT — Cross-species pretraining (added 2026-05-13)

Runs in parallel with X-Mol once the X-Data loaders below land. Each PT-* sub-phase = one experiment script and one analysis script, with the mandatory baseline-set above.

- **X-PT.1** Pretrain corpus assembly utility: `src/kirby/datasets/pretrain_corpora.py` — composes named corpora `('human_chembl', 'rodent_chembl', 'mammalian_chembl', 'human_microsomes')` by union of per-species `chembl_bioactivity` loads + Polaris microsomes. Honors `exclude_inchikeys` for leakage prevention.
- **X-PT.2** PT-α (same-target transfer): `tests/experiments/cross_species/x_pt_alpha_same_target.py`. Pretrain DNN on human bulk ChEMBL; fine-tune on dog COX-2 + cow PTGS2 + cow DRD2. Baselines as above.
- **X-PT.3** PT-β (multi-species + species token): `x_pt_beta_multispecies.py`. Joint pretrain on (human + rat + mouse + dog + cow), species+target tokens, eval on held-out vet.
- **X-PT.4** PT-γ (rodent → vet): `x_pt_gamma_rodent.py`. Pretrain on rat+mouse only; fine-tune on dog/cow. Compare to PT-α to isolate the "evolutionary closeness" effect.
- **X-PT.5** PT-δ (microsomal stability transfer): `x_pt_delta_microsomes.py`. Pretrain on Polaris HLM+RLM; fine-tune on DLM (X-Data.5) and FLM (X-Data.6).
- **X-PT.6** PT-ε (foundation-model fine-tune): `x_pt_epsilon_foundation.py`. ChemBERTa / MolFormer / MolE frozen reps + small task head per vet target. Cheapest experiment — runs first as baseline anchor.
- **X-PT.7** PT-ζ (joint multi-target multi-species): `x_pt_zeta_joint.py`. Single end-to-end model on all KIRBy molecular data. Expensive; runs last.
- **X-PT.8** R&D analysis script `x_pt_rd_analysis.py`: given a compound, return per-species predicted profiles from each PT-* model, with conformal intervals; flag where pretrained models disagree with from-scratch (signals where pretraining is or isn't trustworthy).

**Gate X-PT:** PT-ε runs first (foundation-model baseline). If foundation-model frozen reps already beat from-scratch on small vet data, that simplifies the story and de-prioritises PT-α–ζ. **PT-α–ζ are gated on PT-ε results.**

### Phase X-Data — Non-ChEMBL data sources (added 2026-05-13)

Runs before / in parallel with X-PT (provides the corpora). User-side data downloads required.

- **X-Data.1** DrugCentral loader: `src/kirby/datasets/drugcentral.py`. Pulls vet drug bioactivities. *Acceptance:* returns DataFrame with ≥1,500 rows, vet-flag column.
- **X-Data.2** PubChem BioAssay loader: `src/kirby/datasets/pubchem_bioassay.py`. Filter by `target_taxonomy_id ∈ {9615, 9685, 9913, 9796}`. *Acceptance:* per-species row count reported. Expected: >0 for dog/horse, sparse for cat/cow.
- **X-Data.3** CTD loader: `src/kirby/datasets/ctd.py`. Cross-species chemical-gene interactions. Format: chemical × gene × organism × interaction_type. Free download; cache to `/Volumes/seagate/kirby/ctd/`. *Acceptance:* loader + per-species filter.
- **X-Data.4** openFDA CVM (Animal AERs): `src/kirby/datasets/openfda_cvm.py`. API-based; quarterly updates. *Acceptance:* per-species adverse-event row counts. Used as post-hoc evidence (rank R&D predictions against real-world AERs), NOT as training data.
- **X-Data.5** Polaris DLM loader: `src/kirby/datasets/polaris_microsomes.py` (extend existing Polaris loaders). Dog liver microsomes — listed in §3 X-Mol but never built. *Acceptance:* returns ≥500 dog rows or documents "Polaris DLM does not exist."
- **X-Data.6** Polaris FLM loader: same module, cat liver microsomes. *Acceptance:* same as X-Data.5.
- **X-Data.7** Cat UGT1A6 literature CSV: hand-curated from Court et al. PMIDs. Path `data/raw/cat_ugt1a6.csv` + thin loader. *Acceptance:* >20 compounds with paracetamol-style UGT1A6-substrate flag.
- **X-Data.8** OCHEM rodent toxicity loader: `src/kirby/datasets/ochem.py`. Pulls rat + mouse acute oral toxicity (PredAOT corpus). *Acceptance:* loader + species filter.
- **X-Data.9** Guide to Pharmacology loader: `src/kirby/datasets/guide_to_pharm.py`. Low priority — limited vet coverage. *Acceptance:* loader + species filter; documented sparsity.

**Gate X-Data:** for each loader, the first PR includes (a) row counts per (species, endpoint), (b) a recommendation whether to include in PT-* pretrain corpus. The user reviews before any loader feeds an experiment.

### Phase X-UQ — Conformal wrapper (cross-cutting)

- **X-UQ.1** `src/kirby/uq/conformal.py` — MAPIE wrapper, honors KIRBy CV splitter.
- **X-UQ.2** `src/kirby/uq/conformal_chem.py` — chemistry-aware variant for scaffold splits.
- **X-UQ.3** `SETUP.md` updated with MAPIE + (optionally) CPSign deps.

Runs in parallel with X-Ab / X-Mol / X-DTA; each downstream-analysis step depends on X-UQ being in place.

### Deferred — Phase X-ADA (immunogenicity)

Needs a labeled ADA dataset. Realistic v2 path: use canineness score as ADA proxy, validate against a small clinical cohort if a collaboration emerges. Don't start until v1 is published.

### Deferred — Phase X-Gen (caninization generation)

Generative-models track. HuDiff / AnthroAb / p-IgGen retrainable on canine data. Out of scope for v1.

---

## 6. Claude-friendly task plan

Tasks sized for single bounded sessions, with explicit acceptance criteria. **Tracks are independent — interleave freely.**

### Track X-Ab

**T-Ab.1** Read `humanness.py`, `trastuzumab.py`, `graphinity.py` end-to-end. Draft `docs/cross_species_ness_design.md` — signature, label scheme, file layout, source-data table. *Acceptance:* user signs off before code.

**T-Ab.2** Stage Phase X0 data inventory at `docs/cross_species_data_inventory.md`. *Acceptance:* user reviews, executes user-side actions.

**T-Ab.3** New `src/kirby/datasets/cross_species_ness.py` per the design doc. Existing `humanness.py` untouched. *Acceptance:* loader returns shape-correct arrays for `target_species='dog', mode='one_vs_rest'` over `{human, dog, cat, mouse}`; one-fixture test passes.

**T-Ab.4** Experiment file `tests/experiments/cross_species/x_ab_canineness.py`. `find_best_hybrid(..., cache=True, dataset_name='canineness_v1', augmentation_strategy='uncapped')`. *Acceptance:* dry-run (1 rep × 1 fold × 1 budget) passes locally; full run goes to ARC.

**T-Ab.5** Therapeutic eval script. Loads Librela / Solensia / Trastuzumab sequences, scores them with the cached canineness hybrid. *Acceptance:* Librela scores highest as canine, Trastuzumab highest as human; output is one table.

### Track X-Mol

**T-Mol.1** Read `chembl_qsar.py`, `chembl_bioactivity.py`, `herg.py`, `cox2.py`, `drd2.py`. Draft `docs/cross_species_admet_design.md`. *Acceptance:* user signs off.

**T-Mol.2** Add `organism` kwarg to each of the five loaders. Defaults preserve current behaviour. *Acceptance:* existing tests pass; new tests for `organism='dog'` and `organism='cat'` return non-empty results or fail with a clear "data sparse" log.

**T-Mol.3** `canine_cyp.py` + `canine_pgp.py` convenience loaders. *Acceptance:* each isoform returns >20 compounds or a documented skip.

**T-Mol.4** Experiment `tests/experiments/cross_species/x_mol_species_admet.py`. MOLECULAR_REPS + species-token augmentation + dog/cat ESM-2 target embed. *Acceptance:* dry-run passes, full run on ARC.

### Track X-DTA

**T-DTA.1** Orthologue lookup utility `src/kirby/utils/orthologue_lookup.py`. *Acceptance:* returns dog and cat sequences for 50 sampled BindingDB human targets.

**T-DTA.2** New `cross_species_offtarget.py` loader. *Acceptance:* same row-count as the human BindingDB_Kd slice, with target-sequence column substituted.

**T-DTA.3** Experiment `tests/experiments/cross_species/x_dta_offtarget.py`, cold-drug eval. *Acceptance:* dry-run passes, full run on ARC.

### Track X-PT (added 2026-05-13)

**T-PT.0** Design doc `docs/cross_species_pretraining_design.md`. Covers corpus composition per PT-α–ζ, fine-tune protocol, baselines, evaluation. *Acceptance:* user signs off before any T-PT.* code. **Do first.**

**T-PT.1** `src/kirby/datasets/pretrain_corpora.py` — named corpus assembler. Composes union of `chembl_bioactivity(organism=...)` + Polaris microsomes. *Acceptance:* `compose('rodent_chembl')` returns DataFrame with rat+mouse rows, leakage-filtered.

**T-PT.2** PT-ε experiment script (foundation-model fine-tune, runs first). `tests/experiments/cross_species/x_pt_epsilon_foundation.py`. ChemBERTa/MolFormer/MolE frozen + task head, fine-tuned per (vet target). *Acceptance:* dry-run on dog COX-2 passes; full sweep submits to ARC.

**T-PT.3** PT-α experiment script (same-target transfer). **IMPLEMENTED 2026-05-14** — `tests/experiments/cross_species/same_target_transfer.py` + `tests/slurm_scripts/submit_same_target_transfer.sh` (commit 616d4da). Conditions: `dnn_scratch`, `dnn_tl_full`, `dnn_tl_frozen`, `zero_shot`, `rf`, + `kirby_hybrid` rep. Ridge replaced with `rf`.

> **⚠️ SUBMIT DEPENDENCY — DO NOT submit T-PT.3 standalone.** It fine-tunes pretrained ChEMBL DNN checkpoints that **do not exist on stat-ecr yet**. `imrie_pretrain.sh` (the C4 pretraining job) MUST run and finish first. The new `zero_shot` condition additionally needs those checkpoints to carry the feature scaler — only checkpoints produced by `pretrain_chembl_dnn.py` ≥ commit 616d4da have it; any pre-existing stat-cadd checkpoints are the old format and zero_shot will reject them.
>
> Submit order (chained so SLURM enforces it):
> ```bash
> cd /data/stat-ecr/scat9264/KIRBy && git pull
> PT=$(sbatch --parsable tests/slurm_scripts/imrie_pretrain.sh)
> sbatch --dependency=afterok:$PT tests/slurm_scripts/submit_same_target_transfer.sh
> ```
> Same prerequisite shape applies to every other X-PT transfer variant (rodent_to_vet, microsome_transfer, …): a pretraining job, then the fine-tune experiment.

*Acceptance:* dry-run passes; pretrained model on dog COX-2 beats from-scratch dog COX-2 by ≥0.05 RMSE, or fails fast with documented "no transfer signal."

**T-PT.4** PT-γ experiment script (rodent → vet). *Acceptance:* compare against PT-α; quantify "evolutionary closeness" lift.

**T-PT.5** PT-β experiment script (multi-species + species token). *Acceptance:* species token has measurable effect (toggle on/off, ≥0.03 RMSE delta).

**T-PT.6** PT-δ experiment script (microsomal stability transfer). *Gated on X-Data.5 (DLM) + X-Data.6 (FLM) loaders.*

**T-PT.7** PT-ζ experiment script (joint multi-target multi-species). *Gated on PT-α–ε results.*

**T-PT.8** R&D analysis script — given a candidate compound, run all PT-* fine-tuned models, return per-species profile + conformal interval + agreement-flag. *Acceptance:* one-table output for the pole-network meeting demo set.

### Track X-Data (added 2026-05-13)

**T-Data.1** `src/kirby/datasets/drugcentral.py`. Vet drug bioactivity loader. *Acceptance:* ≥1,500 rows; vet-flag column.

**T-Data.2** `src/kirby/datasets/pubchem_bioassay.py`. Filter by `target_taxonomy_id`. *Acceptance:* per-species row counts logged for dog/cat/cow/horse.

**T-Data.3** `src/kirby/datasets/ctd.py`. CTD chemical-gene interactions. *Acceptance:* per-species filter; cache to `/Volumes/seagate/kirby/ctd/`.

**T-Data.4** `src/kirby/datasets/openfda_cvm.py`. Animal AERs via openFDA API. *Acceptance:* per-species event counts.

**T-Data.5** `src/kirby/datasets/polaris_microsomes.py` (DLM). *Acceptance:* ≥500 dog rows or documented absence.

**T-Data.6** Same module + FLM (cat). *Acceptance:* same as above.

**T-Data.7** Cat UGT1A6 hand-curated CSV `data/raw/cat_ugt1a6.csv` + loader. *Acceptance:* ≥20 compounds.

**T-Data.8** `src/kirby/datasets/ochem.py`. Rat + mouse acute oral toxicity. *Acceptance:* species filter; ≥10,000 rows total.

**T-Data.9** `src/kirby/datasets/guide_to_pharm.py`. Low priority. *Acceptance:* loader works; documented sparsity for vet species.

### Cross-cutting

**T-UQ.1** `src/kirby/uq/conformal.py` — MAPIE wrapper. *Acceptance:* unit tests on synthetic regression + clf show nominal coverage on held-out.

**T-UQ.2** `conformal_chem.py` — scaffold-aware variant. *Acceptance:* unit test on a small scaffold-split molecular fixture shows nominal coverage.

**T-Meta.1** Memory entry + MEMORY.md update — done.

**T-Meta.2** SETUP.md updated for ANARCI / AbNumber / MAPIE / OrthoFinder deps. *Acceptance:* fresh-env install works.

---

## 7. User decisions (resolved 2026-05-12)

1. **v1 species scope** → **dog + cat + cow + horse.** Full ambition. *Implication:* bovine + equine BCR-seq from NCBI SRA is now a v1 dependency, not a v2 deferral. **Largest implementation risk.** May need OPIG collaboration or scope-shrink to dog+cat if SRA assembly stalls.
2. **License posture** → **permissive only (MIT/BSD/Apache).** Rules out AbNatiV (academic) and IgFold (non-commercial). Use BioPhi, ANARCI, AbLang2, ImmuneBuilder instead.
3. **DoggifAI download** → **user handles manually.** Claude does not script the Zenodo pull.
4. **ChEMBL filter** → **`organism` kwarg on existing loaders.** Additive, low-risk. No parallel `chembl_canine.py`.
5. **UQ headline** → **both** — conformal in the regulatory headline; BNN as alternate path. Both v1.
6. **Track priority** → **X-Mol first.** X-Ab and X-DTA queued after X-Mol design doc lands. Run sequentially, not parallel.
7. **Storage** → **`/Volumes/seagate/kirby/cross_species/`** confirmed as raw-corpus root.
8. **Industry collaboration** → **hold indefinitely, pole-network only.** No Petmedix / Vetigenics / Invetx outreach in v1. Animal hospital head + animal org head are the only industry-adjacent leads.

---

## 8. Non-goals (scope guard)

- **No core pipeline changes.** `find_best_hybrid` already handles arbitrary `base_reps` + `augmentations`.
- **No new representations.** KIRBy finds them. No hand-coded `embed_sapiens_score` / `embed_oasis_overlap` / `embed_abnativ`. (Reconsidered if hybrids genuinely fail on small per-species datasets — but with humanness_therapeutic at 0.92 the prior is "they'll work.")
- **No framework experiment changes** ([[memory: framework F00–F15 off-limits]]).
- **No `humanness.py` edits.** New file, parallel codepath.
- **No federated tie-in yet.** Re-evaluate post v1.
- **No AAV / capsid design.** Out of KIRBy scope.
- **No PBPK / Simcyp in v1.** Deferred pending Lhasa license review.
- **No human EHR / VetCompass / SAVSNET work in v1.** Gated on collaboration agreements that don't exist yet — though the user's pole-class network leads may unlock this for v2.

---

## 9. Reference index

- Research memo (the 2026-05-12 user prompt) — long-form justification for every track.
- `humanness.py` — reference for `cross_species_ness.py`; not edited.
- `src/kirby/representations/antibody.py` — full ANTIBODY_REPS, used as-is.
- `src/kirby/representations/protein.py:embed_esm2()` — already cross-species capable.
- `src/kirby/find_best_hybrid.py` — unchanged.
- c11 humanness_therapeutic results — 0.92 AUROC baseline that anchors the canineness target performance.
- `FEDERATED_EXPERIMENT_ROADMAP.md` — phasing pattern this roadmap mirrors.
- `OPIG_COLLAB_ROADMAP.md` — B2/B3/B5 antibody loaders are precedent for new species-tagged corpora.
- `PIPELINE_READINESS_TODO.md` — DATASET_REPS source of truth ([[feedback_dataset_reps_source_of_truth]]).

---

*Version: draft v1, 2026-05-12. Awaiting user answers to §7 before T-Ab.1 / T-Mol.1 / T-DTA.1.*
