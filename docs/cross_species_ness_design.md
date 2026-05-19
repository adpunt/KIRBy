# T-Ab.1 — Cross-Species Antibody-ness Design Doc

Status: **DRAFT FOR USER REVIEW** (2026-05-16, revised). Per `CROSS_SPECIES_ROADMAP.md` Phase X-Ab. No code touched. Three substantive design questions resolved with the user 2026-05-16 ([[project_xab_design]]); after a web-survey of public antibody-sequence data availability the species scope was revised from "full 8" to "dog + cow + 4 OAS background species" (Option 2) — cat / horse deferred to v2.

This doc commits to the exact signature, label scheme, source-data shape, split mechanics, and dependency footprint for the cross-species antibody-ness extension. T-Ab.2 stages the data inventory + the bovine SRA-assembly recipe against this doc, T-Ab.3 implements `src/kirby/datasets/cross_species_ness.py` against it.

---

## 1. Scope

Generalise the binary human-vs-non-human classification in `src/kirby/datasets/humanness.py` to a per-species **one-vs-rest** native-repertoire classifier. v1 species set, picked from what is actually downloadable as of 2026-05-16:

- 2 target species (vet R&D, with data available now): **dog, cow**
- 4 background species (already in the humanness OAS path): **human, mouse, rhesus, camelid**

Deferred to v2 (no usable public VH data available today, deliberately *not* placeholder-implemented):
- **cat** — only single-digit-to-low-double-digit literature sequences; the 2017 four-cat repertoire paper does not deposit an SRA accession that can be located via the abstract.
- **horse** — a 2022 paper reports ~45K IGH clonotypes but provides no abstract-level SRA accession; data deposition status would need confirmation from the full paper or authors before T-Ab.* work can start.

The R&D framing per roadmap §3 X-Ab: given a candidate VH, how native is it to the target species? v1 use cases — caninization QC (does Librela score as canine?), transgenic-platform output screening (Petmedix Ky9 / Vetigenics CANIBODY), engineered-construct drift detection — plus a smaller bovine-ness check (cow IgG has biologically distinctive ultra-long CDRH3, so the classifier should be easy to fit even with modest n).

**Out of scope for this doc:** the species-stratified ADMET track (`docs/cross_species_admet_design.md`), the cross-species DTA track (X-DTA), pretraining (X-PT). `humanness.py` is **not** touched.

---

## 2. Current state — what the three reference loaders already commit to

I read `src/kirby/datasets/humanness.py` (325 lines), `trastuzumab.py` (230 lines), and `graphinity.py` (365 lines) end-to-end. The conventions the new loader must honour:

| Convention | Source | Detail |
|---|---|---|
| Return-dict keys | humanness, trastuzumab, graphinity | `heavy_seqs`, `light_seqs`, `labels`, `task_type`, `n_classes`, `is_intra_family`, `property_name`, `mutations`, `wildtype_heavy`, `wildtype_light` |
| VH-only default | humanness `_load_species` / `_load_therapeutic` | `light_seqs = [''] * n` |
| Classification label dtype | humanness | `np.int64` |
| Mutations list shape | humanness, trastuzumab | `List[List[Dict]]`, one inner list per row; empty list when wildtype not defined |
| Subsampling pattern | humanness `_load_species`, trastuzumab | `np.random.RandomState(42)` + per-class quota |
| Cache layout | all three | `/Volumes/seagate/kirby/<name>/` preferred, `~/.cache/kirby/<name>/` fallback |
| Group field (used by find_best_hybrid) | graphinity `pdb_ids` (string list) | Returned as a top-level key, not in the per-row mutation dicts |

`humanness.py` does **no** sequence numbering — V-gene grouping is a new concern this doc commits to.

---

## 3. Proposed signature

```python
def load_cross_species_ness(
    target_species: str,
    mode: Literal['one_vs_rest', 'multinomial'] = 'one_vs_rest',
    source_species: Optional[List[str]] = None,
    max_samples_per_species: int = 1500,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> Dict:
    """Cross-species antibody-ness loader (T-Ab.3).

    Args:
        target_species: One of the v1-supported species
            {'dog','cow','human','mouse','rhesus','camelid'}. Passing 'cat'
            or 'horse' raises ValueError with a clear "deferred to v2" message
            citing the data-availability finding in the design doc.
            Under 'one_vs_rest' this is the positive class.
        mode:
            'one_vs_rest' — binary classification, label=1 for target_species,
                            label=0 for every other species in source_species.
            'multinomial' — N-class softmax over source_species; labels are
                            indices into SPECIES_VOCAB_NESS (see §4).
        source_species: Species to include in the labelled pool. Default = the
            full v1 6-species pool: ['human','mouse','rhesus','camelid','dog','cow'].
            Passing 'cat' or 'horse' here is also rejected at v1.
        max_samples_per_species: Per-species cap (default 1500, → ~9K total
            under the default 6-species pool). DoggifAI ~430K is capped, not
            weighted; the cow SRA-assembled pool (PRJNA690660, 2 animals,
            ~2M raw reads but likely a few-thousand-to-low-tens-of-thousands
            of unique productive VH after assembly + dedup) is also capped at
            the same per-species ceiling.
        cache_dir: Override the default cache root.
        seed: RNG seed for stratified subsampling.

    Returns:
        Dict with the humanness/graphinity-compatible keys (§2 table) plus:
            'groups': np.ndarray of str — V-gene IDs from ANARCI for
                      StratifiedGroupKFold (see §5).
            'source_species': list[str] — vocab actually used.
            'per_species_counts': dict[str, int].
            'mode': str — 'one_vs_rest' or 'multinomial'.
            'target_species': str | None (None when mode=='multinomial').
    """
```

Default `mode='one_vs_rest'` matches the roadmap §3 X-Ab signature exactly. `multinomial` is opt-in for future ablation studies; T-Ab.3 ships both but T-Ab.4 only runs `one_vs_rest` for `target_species ∈ {'dog', 'cow'}`.

Cache key (for `find_best_hybrid(..., dataset_name=...)`): `f"{target_species}ness_v1"` for one-vs-rest; `f"multispeciesness_v1_{N}"` for multinomial where N = `len(source_species)`. Locked vocab order pins cache stability ([[project_cache_collision_fix]] pattern).

---

## 4. Species vocabulary

Add a single module-level constant in `cross_species_ness.py`:

```python
SPECIES_VOCAB_NESS: List[str] = [
    'human', 'mouse', 'rhesus', 'camelid',   # OAS-resourced background
    'dog', 'cat', 'cow', 'horse',            # vet R&D target species
]

# v1-supported subset. cat / horse stay reserved in the vocab (so adding
# them later does not break cache keys), but the loader rejects them as
# target_species or source_species with a clear v2-deferred error.
V1_SUPPORTED_SPECIES: List[str] = [
    'human', 'mouse', 'rhesus', 'camelid', 'dog', 'cow',
]
```

**Order in `SPECIES_VOCAB_NESS` is pinned for cache stability.** Cat and horse remain at indices 5 and 7 with no data in v1; promoting them to V1_SUPPORTED_SPECIES in a future revision adds them without disturbing any existing cache key. Reordering breaks every cached `find_best_hybrid` result. Mirrors the T-Mol.4 `SPECIES_VOCAB` discipline.

Note: `src/kirby/utils/species.py` `SPECIES_REGISTRY` does not currently include `rhesus` or `camelid` (they have no ChEMBL role). The antibody loader does **not** route through `SPECIES_REGISTRY` for label assignment — labels are derived from the source FASTA / CSV provenance, not from a registry lookup. So this is not a registry extension.

---

## 5. Splits — StratifiedGroupKFold on V-gene (ANARCI)

**Locked decision (2026-05-16):** ANARCI is added as a new repo dependency.

Why ANARCI specifically: roadmap §3 X-Ab says "StratifiedGroupKFold by germline V-gene." The two off-the-shelf options were ANARCI (GPL-licensed, HMMER-backed, the field-standard Chothia/Kabat/IMGT numbering tool) and AbNumber (MIT-licensed, BioPhi-adjacent). User chose ANARCI to match the standing roadmap text verbatim. License posture: GPL is permitted for runtime use because KIRBy is not redistributed under a license that conflicts. SETUP.md must list `anarci` + `hmmer` as new optional deps; the loader imports ANARCI lazily and raises an informative error if the install is missing.

Grouping logic:

1. For every VH sequence in the loaded pool, run ANARCI in IMGT scheme.
2. Extract the V-gene family + allele (e.g. `IGHV3-23*01`); fall back to family-only (`IGHV3`) if allele assignment is ambiguous.
3. Use the V-gene family string as the GroupKFold group key.
4. StratifiedGroupKFold (sklearn ≥ 0.24) so the per-fold positive-class balance is preserved.

Folds: 5 default, 3-fold fallback when `n < 150` (mirroring `foundation_hybrid_vet._resolve_folds`). The fold count is also capped at the number of unique groups; degenerate pools (e.g. cow with only 2 V-genes) drop to that ceiling with a logged warning.

**Numbering cache:** ANARCI output cached per `(sha1(sequence), scheme='imgt')` under `/Volumes/seagate/kirby/cross_species/anarci_cache/`. Atomic write per [[project_cache_collision_fix]] discipline. Loader is idempotent.

---

## 6. Source data — one row per (species, FASTA / CSV) entry

The v1 6-species pool is assembled from three source types. Each source was verified for accessibility on 2026-05-16:

| Species | Source | Path / URL | Access status verified 2026-05-16 |
|---|---|---|---|
| human | OAS via Zenodo 10562968 | existing `humanness.py` `_download_and_extract_species` | already cached at `~/.cache/kirby/humanness/extracted/` (no new download) |
| mouse | same | same | same |
| rhesus | same | same | same |
| camelid | same | same | lumped (Camelus dromedarius + Lama glama in the OAS archive) |
| dog | DoggifAI on Zenodo 15125375 | `/Volumes/seagate/kirby/cross_species/canine_antibodies/` | **Zenodo record is access-restricted**: the user must log into Zenodo and submit an access request (one-time, normally fast). Loader points at the directory after the user has unpacked the archive locally; loader never authenticates against Zenodo. DoggifAI source: 5.4 GB total, three FASTA-equivalent files (heavy, kappa, lambda). v1 reads only the heavy file. |
| cow | NCBI SRA project **PRJNA690660** (Ankole + Boran cattle, Illumina MiSeq 2×300 bp, nested-PCR ultra-long-CDRH3 study) | `/Volumes/seagate/kirby/cross_species/bovine_antibodies/` | **Raw reads only — assembly required** (PRJNA690660 deposits FASTQs, not assembled VH sequences). T-Ab.2 owns the FASTQ→productive-VH assembly recipe; T-Ab.3 reads the assembled FASTA. Expected unique productive VH after assembly + dedup: a few thousand to low tens of thousands per animal (raw read count is ~2M total across both animals + all three nested PCRs). |

Deferred to v2 (no usable v1 path):
- **cat** — abstracts of the 2017 four-cat repertoire paper and the felinized-mAb paper do not name an SRA accession; literature curation yields ~10–30 VH at best. v2 task: full-paper / author-contact + literature CSV.
- **horse** — 2018 / 2022 papers report ~45K IGH clonotypes but no public SRA accession is visible at the abstract level. v2 task: full-paper read or author contact, then either SRA-assembly or supplementary-table parsing.

`load_cross_species_ness` calls three internal `_load_<source>` helpers (mirroring `humanness._load_species` / `_load_therapeutic`):

- `_load_oas_species(species)` — reuses `humanness._find_heavy_fasta` for the four OAS-resourced species.
- `_load_doggifai()` — reads the manually-staged DoggifAI directory; raises a clear error if the directory is empty pointing at the Zenodo access-request URL.
- `_load_bovine_sra_assembly()` — reads the T-Ab.2 assembled VH FASTA; raises a clear error pointing at the T-Ab.2 SLURM script if the assembled file is missing.

Each helper returns `(heavy_seqs: List[str], provenance: List[str])`. `load_cross_species_ness` concatenates, dedups on canonical sequence (uppercase, ungapped), sub-samples per-species, and emits the unified dict.

---

## 7. Label scheme

### 7.1 Training loader (T-Ab.3) — binary `one_vs_rest`

For each row with provenance species `s` and a configured `target_species = t`:

- `label = 1` if `s == t`
- `label = 0` if `s != t` and `s in source_species`
- (row dropped if `s not in source_species`)

`n_classes = 2`. `property_name = f"{target_species}-ness (one-vs-rest, source={source_species})"`. `is_intra_family = False` (sequences span multiple frameworks).

### 7.2 Training loader — `multinomial` mode

`labels[i] = SPECIES_VOCAB_NESS.index(provenance[i])`. `n_classes = len(source_species)`. Available for future ablation; not consumed by T-Ab.4.

### 7.3 Held-out therapeutic eval (T-Ab.5) — three-tier

**Locked decision (2026-05-16):** caninized / humanized therapeutics get an *intermediate* label rather than being conflated with native or off-species. Three-tier scheme used by T-Ab.5 only:

| Tier | Label | Examples |
|---|---|---|
| `native_target` | 2 | DoggifAI native canine sequence, when target_species='dog' |
| `engineered_to_target` | 1 | Librela (caninized bedinvetmab); Trastuzumab (humanized), when target='human' |
| `off_species` | 0 | A native-mouse antibody scored against target='dog' |

T-Ab.5 ranks predictions in pIC-style "more native = higher score" order; the table separates the three tiers so the qualitative finding ("Librela scores higher than off-species but below native canine") is legible without manual cherry-picking.

The training loader (T-Ab.3) **does not** see this three-tier scheme. It is a property of the eval set, not of the labels in the cross-validated training pool.

---

## 8. Anti-leakage rules

1. **Same canonical VH sequence → same fold.** Deduplication on canonical sequence runs before the StratifiedGroupKFold assignment.
2. **Same V-gene family → same fold.** Enforced by the GroupKFold key (§5).
3. **Therapeutic eval (T-Ab.5) sequences are excluded from the training pool.** The therapeutic INN-D list (Librela, Solensia, Cytopoint, Befrena, Lenivia, Portela, CPMA, Trastuzumab, etc.) is hashed by canonical sequence and excluded from the source-FASTA / CSV inputs before any sampling. T-Ab.3 takes an explicit `exclude_inchikeys`-shaped kwarg (`exclude_seq_sha1: Optional[Set[str]] = None`) for this purpose.

---

## 9. Dependencies + SETUP.md

New deps in `setup.py`:

```python
extras_require = {
    ...
    'antibody_ness': ['anarci>=2.5'],
    ...
}
```

ANARCI requires HMMER; SETUP.md must add an install line:

```
# X-Ab dependency
conda install -c bioconda hmmer
pip install kirby[antibody_ness]
```

The loader imports ANARCI lazily inside `_assign_vgene_groups`; an ImportError raises a single clear message naming both the conda and pip lines.

---

## 10. Acceptance for T-Ab.1

This doc is the deliverable. Acceptance: **user signs off.**

T-Ab.1 follow-ups (out of scope for *this* doc):

- **T-Ab.2 — bovine SRA-assembly recipe + data inventory.** Three deliverables, all on the user's machine / server side (no Claude-side network calls):
  1. `docs/cross_species_data_inventory.md` — DoggifAI directory layout after the user downloads it, the assembled bovine VH FASTA layout after T-Ab.2's pipeline finishes.
  2. `tests/slurm_scripts/submit_bovine_assembly.sh` (or equivalent) — pulls PRJNA690660 FASTQs via `sra-tools` (`prefetch` + `fasterq-dump`), runs an antibody-aware assembler (MiXCR or IgBLAST + custom dedup) over the nested-PCR products, emits `bovine_vh.fasta`.
  3. `docs/cross_species_ness_design.md` Note (this doc) appended with the final assembled VH count once T-Ab.2 finishes, so T-Ab.3's per-species cap can be reality-checked.
- T-Ab.3 — implement `src/kirby/datasets/cross_species_ness.py` against this doc; unit-test the three-tier therapeutic label generator and the StratifiedGroupKFold stability separately from the heavy data path.
- T-Ab.4 — experiment script `tests/experiments/cross_species/x_ab_canineness.py`, plus `x_ab_bovineness.py` if a separate file is preferred over a `--target-species` flag.
- T-Ab.5 — therapeutic-eval table for v1 canineness (Librela / Solensia caninized vs DoggifAI native) + bovineness (no clinical bovine mAb shortlist exists today, so the bovine v1 eval is "held-out random DoggifAI-equivalent native split" only).

---

## 11. Open questions deliberately left for T-Ab.2 / T-Ab.3

- Choice of antibody-aware assembler for PRJNA690660 (MiXCR vs IgBLAST + custom). MiXCR is licensed as freeware for academic use; IgBLAST is fully open. T-Ab.2 picks one — preference is IgBLAST for license simplicity if the ultra-long CDRH3 handling is acceptable on the 2×300 bp reads.
- ANARCI scheme — IMGT is the default proposed here; Kabat / Chothia stay configurable via a kwarg if T-Ab.3 finds IMGT family granularity too coarse.
- Whether T-Ab.4 also enables the `multinomial` mode for a single ablation run, or stays one-vs-rest only. Both training paths are shipped; the choice is purely about which sweeps go on ARC.
- Conformal interval calibration mode for T-Ab.5 (`fit_split` vs `fit_prefit` vs `fit`) — defers to the T-UQ.1 mode-selection table in `docs/cross_species_admet_design.md` §4 once it lands.
- Cat / horse promotion to v1 — gated on a successful data-availability re-check (full-paper read for the 2022 horse paper; literature-curated CSV for cat). Not blocking v1 ship.

---

*Version: draft v2, 2026-05-16. Per locked decisions: source pool = dog + cow + 4 OAS species (Option 2 after data lookup); V-gene grouping = ANARCI; therapeutic labels = three-tier; cat + horse v2-deferred. Awaiting user sign-off before T-Ab.2.*
