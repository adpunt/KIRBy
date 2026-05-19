# T-Ab.2 — Cross-Species Antibody Data Inventory

Status: **DRAFT 2026-05-16**. Companion doc to `docs/cross_species_ness_design.md`. Defines the exact data layout the v1 loader (`src/kirby/datasets/cross_species_ness.py`) and experiment scripts (`tests/experiments/cross_species/x_ab_ness.py`) expect on disk.

v1 scope per design doc: dog + cow + 4 OAS background species (human, mouse, rhesus, camelid). Cat and horse are v2.

---

## 1. Storage root

All cross-species antibody data lives under:

```
/Volumes/seagate/kirby/cross_species/        # laptop / local cache
/data/stat-ecr/scat9264/kirby_data/          # ARC (server-side equivalent)
```

The loader respects `KIRBY_CROSS_SPECIES_DIR` env var if set; otherwise it prefers `/Volumes/seagate/kirby/cross_species/` and falls back to `~/.cache/kirby/cross_species/`.

---

## 2. Per-species layout

### 2.1 Dog — two paths, loader-selected automatically

Canine VH data has TWO independent sources in v1. The loader (`_load_canine`)
prefers DoggifAI when it is present and falls back to the SRA-assembled
preliminary FASTA otherwise — both live under the same `canine_antibodies/`
directory, no env-var toggle needed.

| Path | File | n (≈) | Access | When to use |
|---|---|---|---|---|
| DoggifAI (full) | `heavy.fasta` | ~430K | Zenodo access-restricted | once the access request is approved |
| SRA preliminary | `canine_vh.fasta` | low thousands | NCBI BioProject PRJNA790470 (fully open) | available right now; no auth needed |

The loader prints `[_load_canine] source=doggifai file=heavy.fasta` or
`[_load_canine] source=sra_PRJNA790470 file=canine_vh.fasta` so the JSON
results carry a clear provenance trail.

#### 2.1a Path A — DoggifAI (preferred, full dataset)

One-time workflow:

1. Visit https://zenodo.org/records/15125375 — log into Zenodo with ORCID/institutional credentials.
2. Click "Request access" — provide a short justification (academic research on cross-species antibody scoring). Approval is typically same-day.
3. Once approved, download the 5.4 GB archive (the page reveals three FASTA-equivalent files for heavy / kappa / lambda chains).
4. Unpack locally to `/Volumes/seagate/kirby/cross_species/canine_antibodies/`.
5. Rsync to ARC: `rsync -avz /Volumes/seagate/kirby/cross_species/canine_antibodies/ scat9264@arc-login.arc.ox.ac.uk:/data/stat-ecr/scat9264/kirby_data/canine_antibodies/`

Expected on-disk layout (after unpacking):

```
canine_antibodies/
├── README.md
├── heavy.fasta          # ~430K canine VH sequences — loader picks this
├── kappa.fasta          # canine VL kappa (unused in v1)
├── lambda.fasta         # canine VL lambda (unused in v1)
└── model/               # DoggifAI transformer checkpoint (unused in v1)
```

If the heavy-chain file is named something else (e.g. `heavy_chains.fasta`),
the loader's variant glob covers `heavy*.fasta` / `*heavy*.fa` / `*heavy*.fasta`.

#### 2.1b Path B — SRA preliminary (zero-auth)

Friedrichs et al. 2022 (PLOS One) deposit a canine BCR-seq dataset at NCBI
BioProject **PRJNA790470**: 25 healthy + 18 lymphoma dogs (43 total),
Illumina MiSeq 2×300 bp, 5' RACE with constant-region primers, separate
heavy / kappa / lambda libraries per sample. Open access (CC-BY), no
authentication required. Same exact assembly pipeline as bovine PRJNA690660.

To assemble:

```bash
sbatch tests/slurm_scripts/submit_canine_sra_assembly.sh
```

Output:

```
canine_antibodies/
├── sra/                  # *.sra files
├── fastq/                # paired-end *.fastq.gz
├── merged/               # vsearch-merged *.fasta
├── airr/                 # IgBLAST AIRR TSV per sample
├── germlines/canis_lupus_familiaris/  # bovine -> canine germline DBs
├── canine_vh.fasta       # FINAL: dedup'd productive VH AA
└── assembly_report.json  # per-sample raw / productive / dedup counts
```

Expected sample count after dedup: low thousands of unique productive VH
(43 dogs at moderate depth, after canonical-sequence dedup). Smaller than
DoggifAI but large enough to train a real canineness classifier.

**When DoggifAI access lands**, just `rsync` `heavy.fasta` into the same
directory — the loader will swap to it automatically next run.

**Loader behaviour:** if BOTH files are missing, the loader raises:

```
RuntimeError: Canine VH source not found at <path>. Either: (a) request
access at https://zenodo.org/records/15125375 + unpack DoggifAI to
<path>/heavy.fasta, or (b) run tests/slurm_scripts/submit_canine_sra_assembly.sh
on ARC and rsync canine_vh.fasta back. See
docs/cross_species_data_inventory.md §2.1.
```

### 2.2 Cow — NCBI SRA project PRJNA690660

**Access:** open. No authentication required. The BioProject deposits raw Illumina MiSeq 2×300 bp reads from two African cattle (Ankole + Boran), each sequenced across three nested PCR products (PCR-1, PCR-2, PCR-3) to handle ultra-long CDRH3 (>600 bp).

**Pipeline (T-Ab.2 SLURM script):** `tests/slurm_scripts/submit_bovine_assembly.sh` automates the FASTQ → productive VH FASTA conversion on ARC.

Steps:

1. `prefetch` + `fasterq-dump` (sra-tools) → paired-end FASTQs.
2. `vsearch --fastq_mergepairs` → assembled single-end reads.
3. `igblastn` against bovine IGH V/D/J germline databases (NCBI distributes these as part of IgBLAST 1.22+) → AIRR-format TSV.
4. `scripts/parse_igblast_airr.py` → filter `productive == T` + `locus == IGH`, extract `sequence_alignment_aa`, dedup by canonical sequence, write `bovine_vh.fasta`.

**Expected on-disk layout (after the SLURM job finishes):**

```
bovine_antibodies/
├── sra/                  # *.sra files from prefetch (kept for re-runs; ~5-10 GB)
├── fastq/                # paired-end *.fastq.gz
├── merged/               # vsearch-merged single-end *.fastq.gz
├── airr/                 # IgBLAST AIRR TSV output per sample
├── bovine_vh.fasta       # FINAL: dedup'd productive VH amino-acid sequences
└── assembly_report.json  # per-sample counts (raw reads → merged → productive → dedup)
```

**Expected sample count:** somewhere in the range of low thousands to low tens of thousands of unique productive VH after dedup. The exact number is the T-Ab.2 deliverable that gets appended back to `docs/cross_species_ness_design.md` once the SLURM job finishes.

**Loader behaviour:** if `bovine_antibodies/bovine_vh.fasta` is missing, the loader raises:

```
RuntimeError: Bovine VH source not found at <path>/bovine_vh.fasta. Run
tests/slurm_scripts/submit_bovine_assembly.sh on ARC, then rsync the
output directory. See docs/cross_species_data_inventory.md §2.2.
```

### 2.3 OAS-resourced species (human, mouse, rhesus, camelid)

Reuses the existing `humanness.py` Zenodo 10562968 download path. No new files. The loader looks under `~/.cache/kirby/humanness/extracted/` first (same path `humanness._download_and_extract_species` uses) so the data is already cached on any machine that has run `humanness.py`.

If a fresh machine needs the data, the loader triggers `humanness._download_and_extract_species()` automatically — no separate download step.

---

## 3. Reference germlines (bovine IgBLAST)

NCBI IgBLAST ships a multi-organism germline collection at:

```
https://ftp.ncbi.nih.gov/blast/executables/igblast/release/database/
```

For bovine, the relevant files are:
- `bos_taurus_imgt_V.fasta`
- `bos_taurus_imgt_D.fasta`
- `bos_taurus_imgt_J.fasta`

The SLURM script downloads these into `bovine_antibodies/germlines/bos_taurus/` on first run and reuses them on subsequent runs.

**Note on ultra-long CDRH3:** standard IgBLAST will call most bovine sequences as productive even when the CDRH3 is >70 amino acids. The parser keeps these — the design-doc anti-leakage rule (canonical-sequence dedup + V-gene GroupKFold) handles the heterogeneity, and the long CDRH3 is part of what makes bovine antibody-ness distinctive in the first place.

---

## 4. ARC environment notes

The bovine SLURM script assumes the following are available:

- `sra-tools` (conda-forge): `conda install -c bioconda sra-tools`
- `vsearch` (conda-forge): `conda install -c bioconda vsearch`
- `igblast` (ncbi): `conda install -c bioconda igblast`
- Python 3 with `biopython` (already in KIRBy's env per SETUP.md)

The script `module load`s these where ARC provides them, else falls back to `conda activate kirby`. See SETUP.md for the canonical install commands.

---

## 5. Test fixtures

For unit tests in `tests/test_cross_species_ness.py` the loader is exercised against a tiny in-memory mock — no real DoggifAI / SRA data needed. The mock helper `_make_mock_pool` generates:

- 30 fake "dog" VH sequences with two synthetic V-gene labels
- 30 fake "human" VH sequences with two different V-gene labels
- 30 fake "cow" VH sequences

Total 90 sequences across 3 species, 4 V-gene families — enough to exercise the dedup, the StratifiedGroupKFold, and the V1_SUPPORTED_SPECIES validator without ANARCI or network access.

---

*Version: draft v1, 2026-05-16. Updated when T-Ab.2 SLURM finishes with the actual assembled-VH count.*
