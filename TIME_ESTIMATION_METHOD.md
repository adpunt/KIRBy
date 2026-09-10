# TIME ESTIMATION METHOD — KIRBy wave (canonical, 2026-06-04)

> **READ THIS FIRST if you are estimating KIRBy wall-time.** This is the canonical method.
> It supersedes the power-law estimates in `ESTIMATES_INVESTIGATION_2026_06_04.md` /
> `ESTIMATES_UPDATED_2026_06_04.md` (those are kept only as the working history).
> Every number here is validated against the saved measured walls
> (`WAVE_STATUS_2026_06_04.md §8`, `CHECK_PRECOMPUTE_TIMING_2026_06_02.md`,
> `WAVE_SNAPSHOT_2026_06_04_0207.md`) and the validation residuals are stated.
> There is ONE entry below for **precompute** and for **every single experiment** (c01–c24, s01–s13).

---

## 0. THE ONE RULE (why every prior estimate was wrong by 2–10×)

**Do NOT scale a measured total by `(n/anchor)^k`. Do NOT assume cost is linear in n or linear in representation size.** Every prior estimate did this and was wrong by 2–10×.

Instead:
```
wall(experiment, dataset) = cell_count × per_cell_seconds
   cell_count      = read from the experiment's actual loops (budgets × models × folds × strategies × …)
   per_cell_seconds= the operation's ALGORITHM COMPLEXITY in n, with the CONSTANT measured
                     (svm ~O(n²), gp ~O(n³), trees ~O(n·log n), kNN/MI ~O(n²) or O(n) per call)
   the dominant driver is REPRESENTATION WIDTH (total_dim), not n.
```

**Two facts that dominate reality and limit what is estimable at all:**
1. **Node contention** inflates the *same code* by **3–48×** depending on how busy the SLURM node was (the HER2 pipeline is the contaminated example). No formula can predict this — anchor only on un-contended runs.
2. **Cache state** swings cache-consuming experiments by up to **3,700×** at identical work, depending on whether the shared search cache was warm when that experiment ran. These experiments are **not estimable as a point** — give a bounded range.

So the honest output is: a **validated, width-driven core** (precompute + a handful of experiments) + **bounded ranges** for the cache/contention-dominated experiments + **explicit "extrapolated, not validated" flags** beyond the measured regime.

---

## 1. STEP 1 — the representation-width table (the master unlock)

The cost of every dimension-driven experiment is a sum over each dataset's representation widths. A representation's output width is a **static property of the representation** (it does not depend on the dataset), and it is recoverable from the code — **no server run needed** (the old docs wrongly called this a server job). Compute `total_dim(dataset) = Σ width(r) for r in DATASET_REPS[dataset]['general']`.

`DATASET_REPS` is at `tests/experiments/complexity/complexity_helpers.py:94`. Widths below are code-verified (file:line in the source); `~` = uncertain (flagged), use as a band not a point.

**Molecular** (`src/kirby/representations/molecular.py`): ecfp4 2048 · atom_pair 2048 · top_torsion 2048 · rdkfp 2048 · avalon 2048 · maccs 167 · pdv 200 · mordred ~1613 (library-version) · pharmacophore3d 49 · coulomb_matrix 50 · usr 12 · usrcat 60 · mol2vec 300 · graphmvp 300 · molclr 512 · chembert 256 · chemberta 384 (MLM checkpoint; the DTA `drug_chemberta` uses the MTR checkpoint = 768) · molformer 768 · selformer 768 · smited 768 · **grover ~5000** (checkpoint-dependent: large 'both' ≈5–10k) · **mhg_gnn ~1024** (model-dependent) · **graph_kernel ~290** (DATA-dependent: Weisfeiler-Lehman vocabulary filtered per dataset).

**Antibody** (`src/kirby/representations/antibody.py`, paired heavy+light; **halve the concat reps if a dataset is heavy-only**): esm2 2560 · esm2_cdr 2560 · esm2_cdr3 2560 · esm2_early 2560 · esm2_layer24 2560 · esm2_midlayers 2560 · esm2_entropy_top20 2560 · esm2_concat 1280 (single seq) · esm2_per_cdr 7680 (1280×3×2) · **esm2_per_region 17920 (1280×7×2)** · esm3 3072 · esm3_cdr 3072 · esmc 2304 · esmc_cdr 2304 · saprot 2560 · saprot_cdr 2560 · prot_t5 2048 · prot_t5_cdr 2048 · igt5 1024 (single) · ablang2 768 (single) · abroberta 1536 · abroberta_cdr 1536 · antiberty 1024 · antiberty_cdr 1024 · onehot 5600 · blosum 5600 ((150+130)×20) · aac 40 · paratope 8 · structural 28 · esm2_llr/antiberty_llr 3 · esm2_mutsite 1280 · esm2_wt_delta 2560 · mutation_features 48.

**Protein** (`protein.py`): protein_esm2 1280 · protein_esm2_midlayers 1280 · protein_esm2_pocket 1280 · protein_saprot 1280 · protein_esm3 1536 · protein_esmc 1152 · protein_kmer 8000 (20³) · protein_aac 20 · protein_pocket 55 · protein_contact_graph 80 · protein_usr_shape 60 · mutation_matrix ≤200 (data-dependent).

**TCGA / multi-omics** (`genomic.py`, `datasets/tcga.py`): methyl_hvg 5000 · gene_geneformer 768 · gene_geneformer_v2 1024 · gene_bulkrnabert 256 · rppa ~260 (data) · cnv_gistic ≤200 · mirna_top200 200 · mut_driver 46 · mut_pathway 50 · {cnv_pca50, methyl_pca50, mut_pca50} 50 · immune_xcell ~64 (data) · immune_cibersort 22 · gene_stats 15 · mut_tmb 5 · hrd_score 4 · purity_consensus 4 · stemness_mrnasi 2 · cimp_status 2 · gene_chromosome_arm ~44 (data) · tnf_canonical 136 · wsi_{ctranspath,gigapath,hoptimus0,moco,phikon} 512 · wsi_titan 768.

**Nucleotide** (`nucleotide.py`/`sequence.py`): kmer6 4096 · kmer5 1024 · nt_v2_500m 1024 · {dnabert2, grover_dna, mistral_dna, nt_v2_250m} 768 · {kmer4, tetranuc_freq} 256 · hyenadna 256 · kmer3 64 · codon_usage 61 · dinuc_freq 16 · dna_shape 8 · gc_content 5.

**Reaction** (`reaction.py`): {cgr_morgan, drfp_2048, reactant_ecfp4, reactant_avalon, diff_ecfp4, diff_avalon} 2048 · {reactant_mordred, diff_mordred} ~1613 · {reactant_grover, diff_grover} ~5000 · {reactant_mhg_gnn, diff_mhg_gnn} ~1024 · reactiont5 768 · rxnfp 256 · reactant_mol2vec/diff_mol2vec 300 · reactant_pdv 208 · {reactant_maccs, product_maccs, diff_maccs} 167 · {reactant_pharmacophore3d, product_pharmacophore3d} 49 · onehot_components ~45 (data). NOTE: reactant_*/diff_*/product_* do NOT concatenate — width = the base molecular rep's width.

**DTA drug-side** (`tests/dta_baseline_survey.py` → molecular): drug_ecfp4 2048 · drug_avalon 2048 · drug_mordred ~1613 · drug_molformer 768 · drug_selformer 768 · drug_chemberta 768 (MTR) · drug_mol2vec 300 · drug_pdv 200 · drug_maccs 167.

---

## 2. STEP 2 — PRECOMPUTE cost model (VALIDATED)

The precompute builds the shared `find_best_hybrid` search cache. Per dataset it runs conditions: `general/none`, `general/uncapped`, `top/none`, `top/uncapped`; **`top/capped` only when n ≤ 250** (`precompute_hybrids.py:412`, `TOP_CAPPED_MAX_N=250`); plus distance lineages `general_distance`+`top_distance` (only for similarity, `DISTANCE_BUDGETS` 6, `DISTANCE_EVALUATORS` 5). Precompute **always uses `FULL_BUDGETS=[10,25,50,100,250,500,1000,2000]`** (it ignores `config['budgets']` — `precompute_hybrids.py:372`).

**Formula:** `Wall(condition) = Σ over surviving models m [ N_cfg(m) × C_m × n^k_m ]`, where `N_cfg(m)` is the logged per-model config count (encodes budget caps + skips + pruning), and per model:

| model | complexity k | constant C (s) | role |
|---|---|---|---|
| svm | **2.0** (O(n²) kernel) | 1.16e-6 | **#1 driver of general/none + uncapped at large n** |
| gp | **3.0** (O(n³)) | 4.47e-8 | dominates small-antibody (gated off at n>1000) |
| rf | 1.0 (× width) | 1.49e-3 | **dominates capped** (57–65% of capped wall) |
| xgboost | 1.0 | 1.78e-3 | |
| lightgbm | 1.0 | 1.18e-3 | best-behaved |
| ridge | 1.0 | 1.16e-3 | best-fit |
| knn | 1.0 | 6.46e-4 | cheapest |
| mlp | 1.3 | 1.96e-4 | **unpredictable on antibody** |

**Gates:** svm dropped at n>10000 (`_max_samples`, `find_best_hybrid.py:1199`); gp dropped at n>1000 (`:1205`); nucleotide drops ridge/knn/xgboost.

**⛔ `MODEL_MAX_BUDGET` IS EMPTY SINCE 2026-09-10** (owner's decision,
`FRAMEWORK_FINALIZATION.md` §3.3: "Remove cap"). It used to hold svm=500, gp=1000, mlp=1000,
knn=2000, and `N_cfg(m)` above encodes those caps — so every wall in this document counts ZERO
svm configurations at budgets 1000 and 2000, and zero dnn at 2000. svm carries the largest
complexity exponent in the table, so the undercount is worst exactly where the wall matters.
**A SLURM wall taken from the numbers below is now too short.** Everything from here down was
measured under the old caps and is not re-derived; recount `N_cfg(m)` before using it.

Two further changes the same day move `N_cfg(m)` in the same direction: budgets above the
training-fold row count are now searched instead of being dropped (`allow_budget_exceeds_n`
defaults to True), and the per-dataset cap in `dataset_policy.py` rose from 500 to 1000.

**Validation (against `CHECK_PRECOMPUTE_TIMING`):** the per-model decomposition reconstructs **every** measured condition wall to **±0.1%** (structural identity): herg general/none 105,986s (svm 65.6% share — matches measured), logd 73,614s, qm9 74,291s, caco2 16,302s, adme_hlm 28,703s, bindingdb_kd 346,419s, kiba 438,589s, tz_her2_binary 46,257s. The forward per-config fit (`C_m·n^k`) is accurate to **±35%** in the reliable mid-large molecular/TCGA regime; ridge/knn/lightgbm ±1–14%.

**Multipliers (measured):** none→uncapped ×6–16 per model (ridge ×15.8, svm ×15.1, lightgbm ×13.2, rf ×8.5, knn ×6.0); none→capped **×159–×2261** (rf-dominated).

**168-hour (7-day partition) bust points:** capped busts ~n≈240 (already gated off at 250); uncapped (full model set) busts ~n≈4.5–5k (flab_binding_koenig 151.8h, bh_doyle_plates 159.0h); general/none with svm busts ~n≈18–22k (DTA at n=30k survives at 96–122h ONLY because svm auto-drops above n=10000).

**Confidence:** VALIDATED for molecular/TCGA/DTA/nucleotide. NOT reliable: antibody mlp/gp (per-config wall doesn't follow a clean n^k); large-n svm O(n²) is an order-of-magnitude bound. **Molecular uncapped/top walls are UPPER BOUNDS** — inflated by the neural-rep cache-drift recompute (the 6 non-reproducible reps).

---

## 3. STEP 3 — COMPLEXITY experiments c01–c24 (validated against `WAVE_STATUS §8`, 5 pipelines)

`total_dim` = Σ general-rep widths (Step 1). `n_new` = #general reps not in LEGACY (for c13). Validation anchors are bvp(137), cic(137), Ang2(397), VEGF(418) — **HER2(146) is contention-contaminated, excluded from fits.**

### 3A. VALIDATED — predictable from dataset characteristics
| exp | driver | formula | residual (clean) |
|---|---|---|---|
| **c17** information-theoretic | width × samples | `2.464e-4 · M · n^1.078`, M = mutual-info call count from total_dim + budget grid | **−2.7%…+2.3%** |
| **c09** formal analysis | width × samples | `7.45e-6 · 8 · 5 · n · total_dim` | −24%…+14% (RMS 14.5%) |
| **c04** importance-as-metric | samples | `4.70 · n` | −6%…+4.5% |
| **c13** forward-compat | #new-reps × samples | `0.340 · n_new · n` | −20%…+4.4% |
| **c05** learning curves | subsample searches | `186.4·n_calls + 2.70e-3·Σ(n_sub·√width·n_budgets)` | ±5% large-n, ±20% small-n |
| **summed core** | | (c17+c09+c04+c13+c05) | **bvp −4.5%, cic +1.6%, Ang2 −1.0%, VEGF +1.3%** |

M (c17) = `Σ_b [ (td if td>2000 else 0) + Peff + (Peff−1) + Σ_{i=1}^{min(b,Peff)−1}(Peff−1−i) ]`, Peff=min(td,2000), b∈FULL_BUDGETS with b<td and b≤n.
**The c17 n-exponent (1.078) is the near-linear complexity of the KSG mutual-information call; it is validated only for n∈[137,418]. Beyond n=418 it is EXTRAPOLATION (see §5), the single biggest open risk.**

### 3B. NOT PREDICTABLE — cache-state / contention dominated → bounded ranges only
| exp | why | use |
|---|---|---|
| **c21** single-rep aug | **measured 19.3h is STALE** (pre-fix code ran wide reps as bases; current code skips `width>n` → far cheaper) | re-measure on current code; do NOT carry 19.3h forward |
| c23 evaluator-matching | cache state (18.8s vs 10,363s at identical work); the catboost/dnn/cnn "extra tier" contributes **nothing** (filtered by tier gate) | 0.005–2.9h, cache-dependent |
| c24 blueprint transfer | cold-vs-warm cache per fold | 0.3–3.3h |
| c11 ensemble-vs-hybrid | 2.2–2.4× swings between identical datasets | 0.5–1.9h |
| c16 feature-selection-as-reg | 2.82× bvp-vs-cic at identical structure | ±20% large-n bracket only |
| c01 rep characterization | 18× swing at equal width (per-rep SVD/MI conditioning) | 0.02–0.45h |
| c02 complexity grid | cache hit/miss per pass | 0.05–0.32h |
| c15 ML-theory depth | mixed capped + uncached + eigendecomp | 0.35–1.55h |

### 3C. GROWS WITH N — a real large-n cost the old estimate under-weighted
| exp | driver | note |
|---|---|---|
| **c07** data-complexity partitions | uncached GP-bearing fresh searches on easy/hard subpopulations whose size ∝ n | 0.5s @ n=137 → 3,055s @ n=418 → **multi-hour at n≥2k**. Candidate for gating (§8). |

### 3D. NEGLIGIBLE (<0.12h through n≈2k) / conditional
c06 (<0.02h), c10 (regression-only, skips on clf, <0.002h), c18 (<0.12h), c19 (<0.02h), c22 (<0.04h, SHAP grows mildly). **c08** crashes on regression pre-fix; post-fix adds ≤0.5h to regression datasets only. **c12** TCGA-only, ~1.7–2h base (3–7h contended), quadratic in #modality groups. **c03** cross-dataset/distance-pool (not per-dataset). **c14** cross-domain summary (not per-dataset). **c20** merged into c21.

---

## 4. STEP 4 — SIMILARITY experiments s01–s13 (STRUCTURAL — no sim pipeline has ever finished)

**Honesty flag: every similarity number is STRUCTURAL-UNVALIDATED.** No sim pipeline has completed; the snapshot shows only `S02 1done`/`S06 5done` markers and no per-experiment seconds. The "68h+" figures are *precompute*, not sim internals. To validate, grep one finished sim job for `s## done in Xs`.

Similarity is **structurally much lighter than complexity** (no c17/c21-class monster). The cost ceiling is **full n×n distance matrices built once per representation**, which is memory-bounded and explicitly guarded.

| exp | driver / complexity | confidence |
|---|---|---|
| **s01** baseline | n²·Σwidth distance matrices + dense SALI per rep; **~35GB/rep, OOM risk at n≥30k** (switches to bounded-memory path at n>15000) | the heaviest sim experiment |
| **s08** optimal dimension | (1 concat + ≤6 hybrid + 2·R per-rep) n×n + 6-model CV; **2nd heaviest** | |
| **s02** hybrid similarity | R · n²·width (per-rep distance ×2) | |
| **s07** noise bridge | R full n×n distance matrices — **NOT cheap** (same class as s02) | correction to prior docs |
| **s06** applicability domain | R·2 NN-fits + 2 RF/fold — **NOT cheap** | correction |
| **s03** feature concentration | (3·8·R·4) MI calls, each O(n²·≤50) — PCA-capped to ≤50 dims (loop-count-bound, not width) | correction |
| **s05** kNN prediction | kNN-CV, O(n²) query; distance precompute is in run_all, NOT s05 | correction |
| **s10** importance downstream | GP O(n³) ≤5000 cap, else O(n²); GP/SVR regression-only | |
| **s12** cross-domain cliffs | 6-model CV × R reps; cliff matrix capped at 5000 | |
| **s04** activity cliffs | n²·2048 Tanimoto; **molecular-only** (skips elsewhere) | |
| **s11** bayesian-opt | conditions·3·100 iters·pool(≈0.8n); **3 gates: molecular/reaction only, n≥200, ≥20 groups** | |
| **s13** clustering/retrieval | KMeans (n_init=10) per space — **near-linear, NOT n²** | correction |
| s09 | cross-dataset summary (not per-dataset) | |

Magnitude: trivial at small/mid antibody; the danger zone is the per-rep n×n family (s01/s02/s07/s08) at **n≥30k** antibody/DTA widths (memory → wall via the scalable fallback).

---

## 5. Per-dataset c17 estimate, all 73 datasets (the dominant cost; full table in repo)

c17 dominates cx at every n≥~560. Computed from total_dim (Step 1) via the §3A formula. **MEASURED**=one of the 5 anchors; **VALIDATED-REGIME**=antibody n≤418 (method validated here); **EXTRAPOLATED(n>418)**=the n-exponent is unvalidated beyond the anchor — order-of-magnitude only; **+WIDTH-UNC**=total_dim uses an uncertain molecular/reaction width.

Selected (full list regenerated by the script in §10): ginkgo_heparin n94 td82,920 c17≈3.9h (VALIDATED-REGIME) · abagym_HER2 n146 td22,104 c17≈6.8h (MEASURED) · abagym_VEGF n418 td66,932 c17≈53.9h (MEASURED) · flab_thermo_tresanco n672 td82,956 c17≈169h (EXTRAP) · tcga_luad_vs_lusc n1017 td11,526 c17≈394h (EXTRAP+WIDTH-UNC) · herg_ki n1412 td19,095 c17≈571h (EXTRAP+WIDTH-UNC) · caco2 n2158 c17≈1,440h (EXTRAP+WIDTH-UNC) · herg n7772 c17≈5,775h (EXTRAP+WIDTH-UNC) · qm9 n9978 c17≈7,534h (EXTRAP+WIDTH-UNC) · nucleotide n~20k c17≈15,700h (EXTRAP) · kiba/bindingdb_kd n30k c17≈24,857h (EXTRAP).

**These large-n hours are EXTRAPOLATIONS, not validated** — they confirm the "everything at n≥~1400 will not finish the 7-day wall" verdict (order-of-magnitude), but the exact hours need a measured cx run above n=418 to trust (§7).

---

## 6. VALIDATION SUMMARY (the real-numbers confirmation — non-negotiable)

- **Complexity core** (c17+c09+c04+c13+c05) reconstructs each clean measured pipeline to **±4.5%** (`WAVE_STATUS §8`): bvp −4.5%, cic +1.6%, Ang2 −1.0%, VEGF +1.3%. c17 alone ±2.7%.
- **Precompute** reconstructs every measured condition wall to **±0.1%** structurally (`CHECK_PRECOMPUTE_TIMING`); forward per-config fit ±35% (molecular/TCGA reliable).
- **Similarity**: UNVALIDATED — no finished sim pipeline exists.
- The validated core is ~28% of the *measured* cx total; the other ~72% (c21 stale + cache/contention experiments) is **not formula-predictable** — this is the honest ceiling on cx-total precision today.

---

## 7. WHAT NEEDS MORE WORK (the honest ceiling on accuracy — open risks)

1. **A measured cx run above ~420 samples — the #1 open risk.** The dominant cost (c17) is validated only up to 418 samples; every molecular/large-n number rests on extrapolating the c17 sample-exponent **3–70× beyond the anchor**. The large numbers (herg ~5,800h, the 30k-sample DTA sets ~25,000h, nucleotide ~15,700h) are **order-of-magnitude only** — they confirm "won't finish the 7-day wall," but the exact hours are NOT trustworthy until a cx run above n=418 is measured.
2. **One finished similarity run** (`grep "done in" slurm-<simjob>.out`) — needed to validate ANY similarity number; all are currently structural/unvalidated.
3. **A cluster of experiments isn't predictable at all** (c21, c23, c24, c11, c16, c01, c02, c15) — their walls swing up to **3,700×** purely on cache state and node load, not problem size. They get honest **bounded ranges, not point estimates**. NOTE: **c21's measured 19.3h is STALE** (old code ran the wide reps as augmentation bases; current code skips `width>n` → far cheaper). Only isolation runs (cold cache, un-contended node) can pin these.
4. **Four molecular/reaction widths — pinning status (2026-06-04):**
   - **mordred = 1613 — PINNED.** Counted directly from the installed library (`Calculator(descriptors, ignore_3D=True)` → 1613; incl-3D would be 1826). This is the standard mordredcommunity 2D count; confirm the server uses the same library version.
   - **grover ≈ 5000 — partially pinned.** SETUP installs GROVERlarge (`grover_large.pt`, `molecular.py:1662`), and GROVERlarge with `fingerprint_source='both'` (the default) is documented at ~5000 (`molecular.py:1618`). The exact width is set by the model at runtime (no code literal); cannot be pinned to an integer without loading the GROVER weights.
   - **mhg_gnn ≈ 1024 — NOT pinnable locally.** Width = the IBM MHG-GED pretrained encoder's output dim (`mhggnn_pretrained_model_0724_2023.pickle`), no PCA by default. No literal in code; needs loading the pickle (a server/weights action) to confirm. ~1024 is the commonly-cited value, UNVERIFIED.
   - **graph_kernel — genuinely DATA-DEPENDENT, not a constant.** Width = Weisfeiler-Lehman label vocabulary discovered per dataset, then sparsity-filtered (`molecular.py:3786`). Only anchor: logD → 71,611 raw → **290** after filter. Use ~200–400 as a band; it is inherently per-dataset.
   - **Marginal value note:** these only affect molecular/reaction total_dim, and those datasets are all n>418 where the c17 estimate is already order-of-magnitude (n-extrapolation dominates the uncertainty, §7.1). A ±25% total_dim swing from grover/mhg is small next to the 3–70× n-extrapolation risk.
5. **Light-chain pairing per antibody dataset** — concat reps halve in width if a dataset is heavy-only; currently assumed paired. Confirm per dataset.

---

## 8. LEVERS / GATES — cost cuts (WRITE-UP FOR SIGN-OFF; NONE APPLIED, per the no-autonomous-gate rule)

Ranked by hours saved, each tied to its `MASTER_ISSUES_2026_06_04.md` issue ID (bidirectional link — Section D points here):
1. **[COST-1] c17: subsample the mutual-information rows to ≤2000.** KSG MI converges well before n=2000; using all rows buys no accuracy. Makes c17 ∝ constant not ∝ n → **the lever that makes large-n cx finite** (c17@9978 ~1,632h→~310h-equiv; @30k ~5,400h→~310h). Changes selected features → sign-off.
2. **[COST-2] c17: greedy mRMR once-slice** (compute at max budget, slice for smaller) — proven bit-identical, ~÷4–8 on c17's dominant step. (Applied in working tree; neutral.)
3. **[COST-6] SVM uncapped n-gate** (lower `_max_samples` 10000→3000–5000, or drop svm from uncapped at n>5000) — svm O(n²) is the **#1 precompute driver**; saves herg (~302h), logd (~92h), qm9 (~116h) uncapped.
4. **[COST-5 + COST-4] c05: dedup the full-data 3× recompute + n_repeats 3→1** — ~3× on c05; the full-fraction runs the identical search 3× under 3 cache names.
5. **[NEW — relates to TIME-c07] c07: gate the subpopulation fresh searches** — newly identified here as growing with n (multi-hour at n≥2k). NOT previously flagged as a lever; needs a new COST-* entry if pursued.
6. **capped: gate off n≤250** — already gated; confirms the n≤250 scope.
7. **[TIME-5 / TIME-6] GP budget cap 1000→250** — throughput on small antibody (does not help the n≥1400 non-finishers; GP already gated there).

---

## 9. DISPOSITION of every TIME / ORD / LIN issue (this section, closed)

**PART T (time/cost drivers)** — all now have a model:
- TIME-1 (cx/sim 2× precompute): **CLARIFIED** — shared cache by filename; 2× only if cx & sim run concurrently before either writes. Sequence them.
- TIME-2 (8-CPU wave): operational — resubmit is 32-CPU; tree models ~4× faster, single-thread (gp/svm/ridge) unaffected.
- TIME-3 (no intra-precompute checkpoint): acknowledged structural caveat.
- TIME-4 (single conditions >168h): **CONFIRMED** by §2 bust-points (uncapped n≈4.5–5k; general/none+svm n≈18–22k).
- TIME-5 (GP O(n³) small-ab): **MODELLED** (§2, gp k=3, gated n>1000).
- TIME-6 (model gates coarse): the levers in §8.
- TIME-7 (FULL vs SCREENING mismatch): **CORRECTED** — precompute always FULL; only 7 experiments' internal budget sweep was coarse (write-up).
- TIME-8 (capped n≤250 + c05 per-subsample): **MODELLED** (§2 capped, §3A c05).
- TIME-c01…c24: **each modelled** (§3).
- TIME-s01…sdist: **each modelled** (§4; distance lineage folded into §2).

**PART ORD (loop ordering)** — ORD-1…10/NEST/KNOB/REVFLAG: committed, **default-OFF**; per user, only the reverse flag is on the table. **Estimate implication (the relevant part):** because expensive variants run last, a timed-out job's measured wall under-represents total work and biases "best model/budget" low — so estimates here are of **total** work, and partial measured walls (Ang2/VEGF "through c19") are labelled partial.

**PART LIN (forbidden growth assumptions)** — LIN-1…16: **RESOLVED for the validated experiments** (replaced by width-driven + complexity-grounded models). LIN-1 (the c17 power-law) **remains a flagged caveat for n>418 only** (see §5/§7). LIN-2 ("cx ≈ c17×1.5") **REFUTED** by the data (measured 1.24–4.3×, varies with cache/c21). LIN-3/4/11 (s11/capped linear-in-pool): replaced by §2/§4 models.

---

## 10. Data sources + how to regenerate
- Measured walls: `WAVE_STATUS_2026_06_04.md §8` (cx, 5 pipelines), `CHECK_PRECOMPUTE_TIMING_2026_06_02.md` (precompute), `WAVE_SNAPSHOT_2026_06_04_0207.md` (per-config).
- Widths: read the rep registry files in `src/kirby/representations/`; `DATASET_REPS` at `complexity_helpers.py:94`.
- The full 73-dataset c17 table + total_dim is regenerated by the python in the session notes (imports `complexity_helpers.DATASET_REPS`, applies the §1 width table + §3A M-formula). Re-run it whenever `DATASET_REPS` or a rep's width changes.
- **Anchor on un-contended runs; refuse to extrapolate beyond the measured n; report residuals.** That is the method.
