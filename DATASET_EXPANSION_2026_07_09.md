# Expanding beyond the 33 — measured split & preprocessing audit (2026-07-09)

**Answer to "were all the datasets with off splits fixed?": No. Zero were fixed.** The
previous pass identified them and stopped. This document is the actual work: every number
below is **measured from the raw CSVs on `/Volumes/seagate/kirby`**, not read out of a
docstring. Three of them contradict `DATASET_STATUS_2026_07_07.md`.

Supersedes §2 of `FINAL_DATASET_LIST_2026_07_09.md` (the 33-list). The tz_her2 collapse
finding and the `check_precompute_status` fix in that doc still stand.

---

## 1. Correction to the 33 itself: `abcov_kd` is n=141, not ~1780

The doc's **#1 antibody card** — "largest antibody set, n≈1780" (`DATASET_STATUS` §1, §2.1) —
is wrong.

- `binding_rawat2022abcov_kd.csv` is **142 lines = header + 141 rows**, 33,451 bytes, last
  line intact. Not a truncated download.
- Under the loader's own filters (`abcov.py:56-57`: heavy notna, fitness notna, len>10):
  **141 rows**, 136 unique heavy chains, only **109/141 have a light chain**.
- 95%-identity clustering gives **112 groups**, largest 5.

The `~1,780 antibodies` in `abcov.py:8` and `:41` describes the **Ab-CoV database**, not this
CSV. `DATASET_STATUS` §12.2 even quotes the paper as "1,780 Abs, **only 568 KD points**" —
and FLAb's curated paired-chain subset is 141.

`dataset_policy.py:232` sets no `n_samples` for it, so nothing caught this.

**Consequence:** `abcov_kd` is in the *same underpowered class* as the Jain panels that were
cut for being ~130 rows. ~28 samples/fold. It cannot carry "our strongest antibody card."
Its **split is fine** — the n claim is not.

> Caveat: the cache is dated 26 Feb. If upstream FLAb has since grown this file, re-download
> before acting. Everything else here is independent of that.

---

## 2. The real headline: two datasets are censored, not merely mis-split

`DATASET_STATUS` cut `phillips_h3` and the `adams` sets for a **split** reason
(`position_modulo` groups on the first mutation only). That is true but it is not the
binding constraint. Measured target distributions:

| Dataset | n | target | censoring | usable? |
|---|---|---|---|---|
| `flab_binding_phillips_h3` | 32,767 | −log Kd | **88.4% pinned at the floor 6.00** (28,965 rows); only 3,802 uncensored | **dead as regression** |
| `flab_binding_adams` | 11,052 | **raw Kd in molar** (3.2e-10…1e-5) | **65.6% pinned at the ceiling 1e-5** (7,247 rows); only 215 distinct values | **dead as regression** |
| `flab_expression_adams` | 10,970 | expression ER, 0.011–3.881 | none, 6,237 distinct | **clean regression** |
| `flab_binding_koenig` | 4,275 | −log Kd, 7.22–12.16 | none, 4,132 distinct | **clean regression** |

Two separate bugs fall out:

1. **`flab_binding_adams` is on the wrong scale.** Its `fitness` column is raw molar Kd,
   while `koenig` and `phillips` are already −log Kd. `flab.py` copies `fitness` through with
   no transform. Spearman survives (monotone), but RMSE/R² on 1e-10-scale values is
   meaningless, and it is inconsistent with every sibling dataset. Should be
   `pKd = −log10(Kd)` → range [5.00, 9.50].
2. **A Spearman over 88% (or 66%) tied labels is not a rank correlation.** No split fix
   rescues that.

**But this opens a real expansion.** Both are excellent *classification* tasks at the
censoring bound:

- `phillips_h3` → binder vs non-binder: **3,802 pos / 28,965 neg (11.6% positive), n=32,767.**
  That is by a wide margin the **largest and best-powered antibody dataset in the repo** —
  ~230× `abcov_kd`.
- `adams` binding → **3,805 binder / 7,247 non-binder (34.4% positive), n=11,052.**

Third bug, same area: the loader's `max_samples=5000` stratified subsample
(`flab.py:210,290-315`) bins by **fitness quantile**. With 88% of `phillips` labels tied at
6.00, the quantile bins collapse. Subsampling for this dataset is not doing what it says.

---

## 3. Complete split audit — measured, all 46 out-of-33 candidates

Group counts are what the code actually produces, not what the comments claim.

| Dataset | strategy | measured | verdict |
|---|---|---|---|
| `flab_binding_phillips_h3` | `position_modulo` | 32,767 rows = 2¹⁵; 15 varying heavy positions; light identical; L=230; shells exactly C(15,d) | **LEAKS** (multi-mutation). Fixable → §4.1 |
| `flab_binding_adams` | `position_modulo` | 20 positions, multi-mut, L=229; shells {0:103, 1:3510, 2:6533, 3:900, 6:3, 20:3} | **LEAKS**. Fixable → §4.1 |
| `flab_expression_adams` | `position_modulo` | same library | **LEAKS**. Fixable → §4.1 |
| `flab_binding_koenig` | `position_modulo` | **all 4,275 rows at Hamming exactly 1** → single-site | **VALID already.** Just needs precompute |
| `tz_her2_binary/multiclass` | `edit_distance` | 524,346 rows, **10 real shells** (ED 1–10), 33% positive | **VALID group split.** Best available; `identity` → 1 cluster → hash → random |
| `cdr_flexibility` | per-CDR3 (size-1) | 1,627 rows → **216 unique CDRH3** (only 3 have conflicting labels); 41/216 positive | **BROKEN.** Fixable only via ANARCI V-gene → §4.2 |
| Jain panel ×9 | `identity` | n=130–137; clusters = n−1; largest cluster **2** | **split is FINE** (no redundancy to control). Problem is n and one-panel provenance |
| `flab_immunogenicity_marks` | `identity` | 217 rows → 207 clusters | split fine; n small |
| `abcov_kd` *(in the 33)* | `identity` | **141 rows** → 112 clusters, largest 5 | split fine; **n claim wrong** → §1 |
| `flab_thermo_tresanco` *(in the 33)* | `identity` | 672 rows → **309 clusters**, largest 32 | **VALID**, real near-dup merging |
| `ginkgo_*` *(in the 33)* | `pre_split` | 246 rows; `hierarchical_cluster_IgG_isotype_stratified_fold` **exists**, 5 folds (46–54 each) | **VALID** (closes an open server check) |
| `ddg_combined` | `complex` | SKEMPI AB/AG = **1,211 rows / 55 complexes** (median 10, max 122) | more groups than the doc's "~20–32" |
| `abagym_{Ang2,EGFR,VEGF,lysozyme,HER2}` | `position_modulo` | n = 471/659/513/627/146; sites 24/33/23/29/8; **0 multi-mutation rows** | **VALID** (single-site) |
| `abagym_all` | `experiment` | pooled n=2,416 across **5** interface groups | valid, 5 groups → high variance |
| `graphinity_*` | `complex` | labels are FoldX/Rosetta output | **unfixable** (label problem, not split) |
| `humanness_species` | `identity` | germline-separable ~0.99 | **unfixable** (task is trivial) |
| `ginkgo_heparin` | `pre_split` | n=94 | **unfixable** (n) |
| `abagym_HER2` | `position_modulo` | n=146, 8 sites | split valid; n too small |
| `abagym_lysozyme` | `position_modulo` | n=627 | split valid; **signal** is the problem (3/20 reps clear ρ>0.15) |
| `ddg_combined_full` | `complex` | non-Ab PPIs | **unfixable** (off-domain) |
| `davis`, `kiba`, `bindingdb_kd` | cold-protein | sound per §5 of status doc | **VALID**, precompute unbuilt |
| 13× `nt_*` / `gb_*` | chrom/natural | sound | **VALID**, precompute unbuilt |
| `chembl_qsar` | scaffold | 0 result files | needs a build, not a fix |

### The systemic bug behind several of these
`sequence_identity_clusters` (`_shared.py:916-924`): when the clustering yields
`n_clusters < min_groups` (3) it substitutes **`sha256(seq) % n_buckets`** — a per-sequence
hash, i.e. a **random split** wearing a group-CV label. Any single-WT library collapses to
one cluster (a WT-neighbour pair is `1 − 1/230 = 99.6%` identical), so this fires for
tz_her2, phillips, adams **and koenig** if any of them is ever routed to `identity`.

It is reachable not only by the explicit `identity` strategy but from the **fallback paths**
of `pre_split` / `edit_distance` / `experiment` / `position_modulo`
(`antibody_baseline_survey.py:243-294`) whenever their metadata is missing.

---

## 4. The fixes

### 4.1 `phillips_h3` + `adams` ×2 → shell grouping *(3 datasets recovered)*

The only non-degenerate grouping for a single-WT combinatorial library is by **Hamming shell
from wildtype** — exactly what `tz_her2` already uses.

**DONE (this commit):** `flab.py` now returns `edit_distances = [len(m) for m in mutations]`
(both chains). Verified against the loader: `koenig {1: 4275}`, `adams {0:62, 1:1803,
2:2771, 3:360, 6:2, 20:2}`. This is a **pure data addition — no behavior change**, since none
of these datasets currently uses the `edit_distance` strategy.

**Not done, needs your signoff** — two changes that must land *together*:

```python
# tests/antibody_baseline_survey.py  (registry)
'flab_binding_phillips_h3':   ('flab', 'edit_distance'),   # was position_modulo
'flab_binding_adams':         ('flab', 'edit_distance'),
'flab_expression_adams':      ('flab', 'edit_distance'),
```

```python
# tests/antibody_baseline_survey.py:255-262  -- the trap
elif strategy == 'edit_distance':
    ed = data.get('edit_distances')
    if ed is None:
        raise ValueError(...)        # currently: falls back to _identity_groups
```
**Without the second change, switching the strategy would silently produce a random split**,
because the `edit_distance` branch falls back to `_identity_groups`, which collapses to one
cluster and hash-buckets. This is the single most dangerous line in the fix.

**Residual weakness, state it:** shell grouping does not stop 1-mutation neighbours in shells
`k` and `k+1` from straddling a fold. Two options:
- **(a) GroupKFold on shells** (phillips: 16 groups, adams: 6, tz_her2: 10). Keeps 5-fold CV.
- **(b) Buffered extrapolation:** train on shells ≤ *a*, test on shells ≥ *a*+2, **drop the
  buffer shell *a*+1**. Since `Hamming(x,y) ≥ |ED(x) − ED(y)|`, no train/test pair can differ
  by a single mutation — a genuinely leakage-controlled protocol, and the one Phillips 2021
  itself uses. Cost: it is a **single split, not k-fold**, which conflicts with the lab's
  3–5-fold rule.

For `phillips` (b) is clean: train ED ≤ 7 (16,384), buffer ED = 8 (6,435 dropped), test
ED ≥ 9 (9,948). **This is a decision, not a detail — I have not chosen.**

### 4.2 `cdr_flexibility` → ANARCI V-gene grouping *(1 dataset, thin)*
Identity clustering on the Fv **does not work** — measured: 0.95 → 210 clusters from 216
samples (a no-op); 0.85 → 187; 0.80 → 152; 0.70 → 38 but the largest cluster swallows **150
of 216**. Antibody frameworks are too conserved; single-linkage either does nothing or
chains everything.

The principled key is **IGHV family**, derivable from the Fv with `_assign_vgene_groups`
(`src/kirby/datasets/cross_species_ness.py:300-346`, ANARCI + regex family reduction,
disk-cached). ANARCI is a declared dep (`requirements-working.txt:18`, `anarci==1.3`).

Then relax the `group_sizes.max() == 1` assert (`opig_cdr_flexibility.py:148-152`), which
currently *enforces* the degenerate case.

Honest verdict: n=216 with 41 positives across ~a handful of IGHV families is thin. Fixable,
but it will not be a headline.

### 4.3 `flab_binding_adams` target scale
`fitness` is raw molar Kd. Convert to `pKd = −log10(Kd)` in the loader, or drop the dataset
as a regression task per §2.

### 4.4 `ddg_combined`
55 AB/AG complexes are available (not 20–32). Verify what the loader's filtering leaves,
then k-fold within leave-one-complex-out. Less coarse than the doc assumed.

### 4.5 Systemic
`_shared.py:916-924` should **raise** rather than hash-bucket. A dataset with <3 clusters has
no valid group split, and silently inventing one is the exact failure OPIG's Graphinity
result exists to warn about. **Not changed without signoff.**

---

## 5. The expanded run set

### Tier 0 — the 33, with one correction
Keep, but **`abcov_kd` is n=141**, not a "largest set" headline (§1). Reframe or demote.

### Tier 1 — add now, no fix needed, only precompute *(17)*
Split already valid; they were excluded for compute, not science.
- `flab_binding_koenig` — single-site verified, precompute 7/8
- `davis`, `kiba`, `bindingdb_kd` — cold-protein CI
- 13× nucleotide: `nt_H2AFZ`, `nt_H3K27ac`, `nt_H3K36me3`, `nt_H3K4me1`, `nt_H3K4me2`,
  `nt_H4K20me1`, `nt_enhancers`, `nt_enhancers_types`, `nt_promoter_all`,
  `nt_promoter_no_tata`, `nt_splice_sites_acceptors`, `nt_splice_sites_donors`,
  `gb_human_nontata_promoters`
  *(`gb_demo_*` ×2 are demo sets — exclude.)*

### Tier 2 — add after a split fix *(4)*
- `flab_expression_adams` — clean regression, needs shell grouping (§4.1)
- `flab_binding_phillips_h3` — **as classification**, 11.6% positive, n=32,767 (§2 + §4.1)
- `flab_binding_adams` — **as classification**, 34.4% positive, n=11,052 (§2 + §4.1)
- `cdr_flexibility` — ANARCI V-gene grouping (§4.2), thin

### Tier 3 — add after a preprocessing build *(2)*
- `chembl_qsar` — build precompute
- `ddg_combined` — k-fold (§4.4); already in the shortlist

### Tier 4 — exploratory *(1)*
- `abagym_all` — 2,416 samples / 5 interface groups

### Tier 5 — genuinely unfixable *(cut, with the real reason)*
| Dataset | reason | is it a split problem? |
|---|---|---|
| `graphinity_foldx/rosetta/augmented` | labels are FoldX/Rosetta output | no — label |
| `humanness_species` | germline-separable, ~0.99 | no — task |
| `ginkgo_heparin` | n=94 | no — n |
| `abagym_HER2` | n=146 | no — n |
| `abagym_lysozyme` | dead signal | no — signal |
| `ddg_combined_full` | non-Ab PPIs | no — domain |
| Jain panel ×9 | n≈137, 9 endpoints from one 2017 panel | **no — the splits are fine** |
| `tz_her2_*` | *not* unfixable — `edit_distance` is valid; see §4.1(a)/(b) | — |

**Net:** the 33 can grow to **~54 datasets** (33 + 17 + 4), of which 17 need no science work
at all. Only 7 datasets are genuinely unfixable, and **not one of them for a split reason.**

---

## 6. Decisions I need

1. **Shell-split protocol** — GroupKFold-on-shells (keeps 5-fold) or buffered extrapolation
   (leakage-proof, single split)? Applies to `phillips_h3`, both `adams`, and `tz_her2`. §4.1
2. **Recast the censored sets as classification?** That turns `phillips_h3` into the largest
   well-powered antibody dataset you have. §2
3. **`abcov_kd`** — demote from headline, or re-download to check upstream n? §1
4. **Signoff for the two paired code changes** in §4.1 (registry + fail-loud guard). They
   must land together or the fix becomes a silent random split.
5. **`_shared.py:916-924`** — make the degenerate guard raise instead of hash-bucket? §4.5
