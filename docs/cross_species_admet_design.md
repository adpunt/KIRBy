# T-Mol.1 — Cross-Species ADMET Design Doc

Status: **DRAFT FOR USER REVIEW** (2026-05-12). Per `CROSS_SPECIES_ROADMAP.md` Phase X-Mol. No code touched.

This doc commits to the exact signature, label scheme, source-data shape, and split mechanics for the species-stratified ADMET extension. T-Mol.2 implements against this doc.

---

## 1. Scope

Add per-species ADMET prediction to KIRBy by adding an `organism` argument to existing ChEMBL-backed loaders. v1 species: **human, dog, cat, cow, horse** (per §7 decision 1). Default `organism='human'` preserves all current behaviour.

Out of scope for this doc: antibody (X-Ab), DTA (X-DTA), generative models (X-Gen), ADA (X-ADA). Those have their own design docs.

---

## 2. Current state of the five loaders

| Loader | Data source | Targets | Anchor | Extensibility |
|---|---|---|---|---|
| `chembl_qsar.py` | **Pre-curated CSVs from Dablander GitHub** | D2 / FXa / Mpro | `TARGETS` dict with directory + filename | **Hard.** Not a live ChEMBL query; no organism field in source CSVs. |
| `chembl_bioactivity.py` | **Live ChEMBL via `chembl_webresource_client`** | Top-N by IC50 activity count | `target_chembl_id` iteration | **Easy.** Add `target_organism` to `activity_client.filter(...)`. |
| `cox2.py` | Live ChEMBL | Single target COX-2 | `COX2_CHEMBL_ID = 'CHEMBL230'` (human) | **Medium.** Need to map `(gene='PTGS2', organism)` → species-specific ChEMBL ID. |
| `drd2.py` | Live ChEMBL | Single target DRD2 | `DRD2_CHEMBL_ID = 'CHEMBL217'` (human) | **Medium.** Same as cox2. |
| `herg.py` | FLuID + TDC + Live ChEMBL | Single target hERG | `herg_target_id = 'CHEMBL240'` (human) | **Medium for ChEMBL path.** FLuID and TDC paths are human-only by source; cannot extend. |

**Critical insight:** the four "single-target" loaders are anchored to a *human* ChEMBL target ID. They are not currently doing a `target_organism` filter — they are doing a `target_chembl_id=<human>` filter. The dog/cat/cow/horse equivalent gene has a **different ChEMBL target ID**, not the same ID with a different organism filter. The design must reflect this.

---

## 3. Proposed signatures

### 3.1 `chembl_bioactivity.py` (easiest case)

```python
def download_chembl_bioactivity(
    data_dir: str = '/Volumes/seagate/kirby/chembl',
    max_targets: int = 200,
    max_compounds: int = 500_000,
    exclude_inchikeys: Optional[Set[str]] = None,
    random_seed: int = 42,
    organism: str = 'human',           # NEW
    organism_strict: bool = True,      # NEW
) -> pd.DataFrame:
```

- `organism`: one of `'human', 'dog', 'cat', 'cow', 'horse'`. Mapped internally to ChEMBL organism strings (table below).
- `organism_strict`: if `True`, only activities whose target's `organism` field exactly matches the resolved species. If `False`, accept any non-human assay (rare; reserved for future).
- Pre-aggregation filter passed to `activity_client.filter(target_organism=<resolved>)`.
- Cache filename changes: `chembl_bioactivity_top{N}_{organism}.csv` so per-species caches coexist.

**Default behaviour preserved:** `organism='human'` produces the current cache file (after a one-time rename, or by aliasing in code).

### 3.2 `herg.py`, `cox2.py`, `drd2.py` (single-target loaders)

Same kwarg name, but the resolution is gene-symbol-based:

```python
def load_herg_chembl(
    n_samples: Optional[int] = None,
    ic50_threshold: float = 10.0,
    data_dir: str = 'data/hERG',
    random_seed: int = 42,
    organism: str = 'human',           # NEW
) -> Dict:
```

- Internal `_TARGET_IDS` table maps `(gene_symbol, organism) → chembl_target_id`. For hERG: `('KCNH2', 'human') → CHEMBL240`, `('KCNH2', 'dog') → CHEMBL???`, etc.
- Resolution happens at call time. If the species-specific target ID is unknown, raise a clear `ValueError("hERG dog target not in registry; add to _TARGET_IDS or query ChEMBL targets API")`.
- Cache filename changes: `herg_chembl_cache_{organism}.csv`.
- **FLuID + TDC paths remain human-only.** Document this explicitly; the `load_herg_lhasa()` and `load_herg_tdc()` functions do not get the kwarg.

Same shape for `cox2.py` (gene `PTGS2`) and `drd2.py` (gene `DRD2`).

### 3.3 `chembl_qsar.py` (hard case)

**Recommendation: do not add `organism` kwarg.** The source CSVs from Dablander are human-only and the package's value (activity-cliff curation) does not transfer cleanly to live ChEMBL queries.

Instead: build a **new sibling loader** `chembl_qsar_live.py` later if cross-species activity cliffs are wanted. v1 does not need this — the four other loaders cover the X-Mol pilot. Flag in code: add a docstring note pointing to the new loader.

**Acceptance of this recommendation is part of the user review gate for this doc.**

---

## 4. Organism resolution table

ChEMBL stores `target.organism` as Latin binomials. Build a small registry:

| `organism` arg | ChEMBL `target.organism` | UniProt taxonomy ID |
|---|---|---|
| `'human'` | `Homo sapiens` | 9606 |
| `'dog'` | `Canis familiaris` | 9615 |
| `'cat'` | `Felis catus` | 9685 |
| `'cow'` | `Bos taurus` | 9913 |
| `'horse'` | `Equus caballus` | 9796 |

Lives in a new shared utility `src/kirby/utils/species.py` so loaders + future tracks share one source of truth.

```python
SPECIES_REGISTRY = {
    'human': {'chembl_organism': 'Homo sapiens', 'taxonomy_id': 9606},
    'dog':   {'chembl_organism': 'Canis familiaris', 'taxonomy_id': 9615},
    'cat':   {'chembl_organism': 'Felis catus', 'taxonomy_id': 9685},
    'cow':   {'chembl_organism': 'Bos taurus', 'taxonomy_id': 9913},
    'horse': {'chembl_organism': 'Equus caballus', 'taxonomy_id': 9796},
}
```

T-Mol.2 creates this module first; T-Mol.3+ depends on it.

---

## 5. Target-ID registry (per-loader)

Each single-target loader gets a `_TARGET_IDS` dict resolved at module load:

**Resolved 2026-05-13** via `scripts/resolve_chembl_targets.py` (T-Mol.2b run).
Full JSON at `docs/cross_species_targets_resolved.json`. Counts = IC50 + Ki, nM, relation `=`.

### 5.1 hERG (`KCNH2`)

| organism | ChEMBL ID | IC50 | Ki | total | status |
|---|---|---|---|---|---|
| human | CHEMBL240 | 11,712 | 1,811 | 13,523 | **ok** |
| dog | CHEMBL3085616 | 1 | 0 | 1 | skip |
| cat | — | — | — | — | none |
| cow | — | — | — | — | none |
| horse | — | — | — | — | none |

**Verdict: cross-species hERG is dead.** Only human has data. **Drop hERG from T-Mol.2d scope; do not extend `herg.py`.**

### 5.2 COX-2 (`PTGS2`)

| organism | ChEMBL ID | IC50 | Ki | total | status |
|---|---|---|---|---|---|
| human | CHEMBL230 | 6,284 | 26 | 6,310 | **ok** |
| dog | CHEMBL4033 | 110 | 0 | 110 | **ok** |
| cat | CHEMBL5303558 | 0 | 0 | 0 | none |
| cow | CHEMBL3331 | 28 | 0 | 28 | cv3 |
| horse | CHEMBL5303566 | 0 | 0 | 0 | none |

**Verdict: strongest cross-species case.** 3-species training feasible (human + dog + cow). Cow uses 3-fold CV. Cat/horse skipped (no data).

### 5.3 DRD2

| organism | ChEMBL ID | IC50 | Ki | total | status |
|---|---|---|---|---|---|
| human | CHEMBL217 | 1,100 | 11,961 | 13,061 | **ok** |
| dog | CHEMBL2703 | 1 | 0 | 1 | skip |
| cat | — | — | — | — | none |
| cow | CHEMBL3998 | 160 | 246 | 406 | **ok** |
| horse | — | — | — | — | none |

**Verdict: half-strong case.** 2-species comparison (human + cow). Dog/cat/horse skipped (no data).

**T-Mol.2 first action:** write `scripts/resolve_chembl_targets.py` that, for each (gene, organism) pair, queries `target_client.filter(target_components__component_synonyms__component_synonym__iexact=<gene>, tax_id=<id>)` and writes the resolved IDs back to this doc. Filter by `tax_id` (integer) rather than `organism=<latin>` because ChEMBL's organism-string convention drifts between releases (e.g. `Canis familiaris` vs `Canis lupus familiaris` for tax_id 9615). The Latin string is kept in the registry for display only. **Many of these queries will still be empty** — vet ChEMBL coverage is sparse. The doc gets updated with which species/target combinations actually have data.

---

## 6. Label scheme — no changes

Existing label conventions preserved per-loader:

- `herg.py`: binary clf, `label = (IC50_uM ≤ 10.0)` regardless of organism.
- `cox2.py`, `drd2.py`: regression on pIC50 / pKi (unchanged).
- `chembl_bioactivity.py`: regression on pIC50 (unchanged).

The species *identity* arrives as a separate column in the returned dict, not as a label modifier:

```python
return {
    'smiles': [...],
    'labels': [...],
    'organism': 'dog',      # NEW — scalar, applies to all rows
    'task_type': 'classification',
    'source': 'chembl',
}
```

For multi-species training (species-as-token), the experiment script concatenates per-organism loads and constructs the species column.

---

## 7. Source-data status by (loader, organism)

ChEMBL vet coverage is sparse. Honest expectations:

| Loader | Human | Dog | Cat | Cow | Horse |
|---|---|---|---|---|---|
| chembl_bioactivity (top-N IC50) | rich | sparse–moderate | very sparse | very sparse | very sparse |
| herg (KCNH2) | rich | ~tens? | unknown | unknown | unknown |
| cox2 (PTGS2) | rich | moderate (vet NSAID lit) | moderate | moderate | moderate |
| drd2 | rich | sparse | sparse | very sparse | very sparse |

**T-Mol.2 acceptance check:** for each (loader, organism) pair, report actual row count. If <20 rows, mark "data sparse, skip" and do not propagate to experiments. **Do not fabricate.**

Cat UGT1A6 substrate data and dog CYP1A/2B/2C/2D/3A substrate data are **not in ChEMBL**; they live in hand-curated literature CSVs. Those land in the separate `canine_cyp.py` loader (Phase X-Mol.3), not in this doc.

---

## 8. Splits

No changes. Continue to use ScaffoldGroupKFold (already in KIRBy for molecular). 5-fold. Groups by Murcko scaffold.

For multi-species training:
- Within each organism's subset, scaffold-group.
- Across organisms, scaffolds may overlap (same compound in human + dog assays). Treat each (compound, organism) as a separate row; the scaffold group constraint applies *within fold*, not across organisms.
- **Anti-leakage check in T-Mol.2:** when species-as-token is used, ensure the same scaffold does not appear in train and test across organisms. If a scaffold's compound is in human-train, the dog-test row for that same compound must move to train.

This is a non-trivial cross-organism scaffold-grouping rule. Encode it as `ScaffoldGroupKFoldMultiOrganism` if the default splitter doesn't already handle it.

---

## 9. Species-as-token augmentation

A one-hot over `{human, dog, cat, cow, horse}` (5D) per row, passed to `find_best_hybrid` as:

```python
augmentations = {'species_token': np.eye(5)[species_indices]}
```

Per [[project_cache_collision_fix]], pass `augmentation_strategy='uncapped'` explicitly.

Species-token is **always included** in cross-species runs; the hybrid search decides whether to use it.

---

## 10. UQ wiring

Per §7 decision 5: both conformal and BNN.

- BNN: existing `models/bnn.py`, already in DOWNSTREAM_MODELS_NOISE.
- Conformal: new `src/kirby/uq/conformal.py` (Phase X-UQ.1, separate from T-Mol).

T-Mol experiments call `find_best_hybrid` as usual; UQ is layered on top in the *downstream R&D analysis* script (T-Mol.5), not in the survey. Survey output: hybrid + per-fold predictions + best model. Analysis script: load hybrid → fit on full train → produce per-test-row prediction + conformal interval + BNN posterior.

---

## 11. File-level plan for T-Mol.2 → T-Mol.5

| Task | Files touched | Net new | Acceptance |
|---|---|---|---|
| T-Mol.2a | `src/kirby/utils/species.py` (NEW) | 1 module | `from kirby.utils.species import SPECIES_REGISTRY; SPECIES_REGISTRY['dog']['taxonomy_id'] == 9615` |
| T-Mol.2b | `scripts/resolve_chembl_targets.py` (NEW) | 1 script | Resolves dog/cat/cow/horse target IDs for KCNH2/PTGS2/DRD2; writes results back to this doc |
| T-Mol.2c | `chembl_bioactivity.py` | +`organism` kwarg | Existing tests pass; `organism='dog'` returns >0 rows or clear "sparse" log |
| T-Mol.2d | `herg.py`, `cox2.py`, `drd2.py` | +`organism` kwarg | Same; with cache key updated to include organism |
| T-Mol.3 | `canine_cyp.py` (NEW) | 1 loader | Hand-curated CYP CSV + ChEMBL slice; >20 compounds per isoform or documented skip |
| T-Mol.4 | `tests/experiments/cross_species/x_mol_species_admet.py` (NEW) | 1 experiment | Dry-run (1 fold, 1 budget) passes locally; full run on ARC |
| T-Mol.5 | `tests/experiments/cross_species/x_mol_rd_analysis.py` (NEW) | 1 analysis script | For a candidate compound, emit per-species ADMET + conformal interval + BNN posterior table |

---

## 12. Risks & honest unknowns

1. **ChEMBL vet target coverage may be too sparse to be useful.** Even where target IDs exist, activity counts may be in single digits. T-Mol.2b is the first reality check. If dog hERG returns 5 compounds, the species-stratified hERG model is moot and the track refocuses on cox2 (better vet NSAID literature) and chembl_bioactivity bulk pooled.
2. **Multi-organism scaffold grouping is a new splitter.** If `ScaffoldGroupKFoldMultiOrganism` doesn't already exist in KIRBy, T-Mol.4 needs it before the experiment can run. Verify in T-Mol.2 prep.
3. **`chembl_qsar.py` punt may be wrong.** If the user wants activity-cliff analysis cross-species, a parallel live loader is needed. Not in v1 scope.
4. **FLuID + TDC paths in `herg.py` stay human-only.** No way to extend without new data sources. Document and move on.
5. **Cow + horse coverage is the worst.** Bovine NSAID literature exists (flunixin, meloxicam) but isn't necessarily in ChEMBL. May need literature-curated CSV path analogous to cat UGT1A6.

---

## 13. Decisions (resolved 2026-05-12)

1. **`chembl_qsar.py` — SKIP for v1.** Stays human-only. No sibling live loader in v1.
2. **Species token — TOGGLE.** Experiment-level `use_species_token=True/False`. Defaults to `True` for cross-species experiments.
3. **Cache filenames — SUFFIX.** Append `_<organism>` to existing cache filenames in the same directory.
4. **Sparse-data threshold — 20 rows minimum** for inclusion. **Note:** combos with <50 rows may need to drop 5-fold CV in favor of 3-fold or loo; flag in T-Mol.2b output. The 20-row threshold is "include at all"; the 50-row threshold is "include with standard 5-fold CV."
5. **Cross-organism normalization — RAW pIC50.** Species token absorbs the systematic offset. No mean-centering. Matches TACTIC and standard multi-task QSAR practice.

---

## 14. Acceptance for T-Mol.1 (this doc)

User reviews and signs off on:
- §3 signatures
- §4 organism registry
- §5 target-ID registry approach (resolve in T-Mol.2 vs. hand-fill now)
- §7 source-data honesty (no fabricated counts)
- §11 file-level plan
- §13 open questions resolved

After sign-off, T-Mol.2 starts.

---

*Version: draft v0, 2026-05-12. Author: Claude (Opus 4.7), under user direction. No code written.*
