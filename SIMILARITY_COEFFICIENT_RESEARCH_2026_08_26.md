# Similarity/distance coefficient research — 2026-08-26

⚠️ **PROVENANCE: THIS IS NOT A KIRBY RESULT.** Nothing below was measured by a KIRBy
experiment, and no number here belongs in a results table or a paper. It is the output of
a literature-and-code research run (91 agents: four parallel surveys, one adversarial
verifier per coefficient, one synthesis). Formulas were checked against primary sources;
the numeric claims come from the agents' own small scripts, not from this repository's
datasets or evaluator.

**Treat every number as a claim to re-verify before citing.** What it is good for: the
coefficient list, the duplicate/degeneracy relations, the preprocessing each family
needs, and the weighted forms. What it is NOT good for: any statement about how a
coefficient performs on herg_ki, logd, or any other KIRBy dataset.

Already acted on from this report, in `src/kirby/similarity/preprocessing.py` and
`compute_distance_matrix`: the per-coefficient weight fold-in exponents, the per-family
block divisors, the binary and non-negativity checks, the negative-weight clip and the
Mahalanobis rank guard.

Master record for the similarity thread: `SIMILARITY_METRICS_ROADMAP.md`.

---

# Distance & Similarity Coefficients — Verified Reference

**Notation (used throughout).** For binary fingerprints of fixed length `n`, the 2×2 contingency counts are

- `a` = features on in **both** (`|X ∩ Y|`)
- `b` = on in `x` only
- `c` = on in `y` only
- `d` = off in both
- `n = a + b + c + d`; `|x| = a + b`; `|y| = a + c`

This is the Choi/Cha/Tappert and Todeschini convention. It is **not** the Willett/chemoinformatics convention (`a = |x|`, `b = |y|`, `c = |X ∩ Y|`) that RDKit docstrings and much of the existing repo prose use. Every formula below is in the 2×2 convention. Mixing the two is the single most common transcription error in this material and produces plausible-looking wrong numbers, never an exception.

For count / continuous vectors, `x_k`, `y_k` are the k-th feature values, `⟨x,y⟩ = Σ_k x_k y_k`, `‖x‖² = ⟨x,x⟩`.

---

## 1. THE FULL COEFFICIENT LIST

**Total: 58 entries** across three family tables — 35 binary-native, 11 count-native, 12 continuous. After removing exact aliases and cross-family repeats there are ~55 distinct definitions; after removing monotone reparameterisations (Section 2) there are roughly **19 independent rank-geometries on binary**, **4 on counts**, and **7 on continuous features**. The gap between 58 and 30 is the entire point of Section 2.

### 1a. Binary fingerprints

Every coefficient in this table requires strictly `{0,1}` entries and a **fixed, shared fingerprint length**. Those requiring `d` (the joint-absence cell) are marked ⊘ — they are additionally non-comparable across fold widths, because `n` enters the formula. The weighted column gives the per-feature weighting where one exists; `w_k ≥ 0` throughout, and `a_w = Σ_k w_k x_k y_k`, `b_w = Σ_k w_k x_k(1−y_k)`, `c_w = Σ_k w_k (1−x_k) y_k`, `d_w = Σ_k w_k (1−x_k)(1−y_k)`, `n_w = Σ_k w_k`.

| # | Name | Formula (similarity unless marked D) | Uses `d`? | Weighted form | Range |
|---|---|---|---|---|---|
| 1 | Tanimoto / Jaccard | `a/(a+b+c)` | | `a_w/(a_w+b_w+c_w)` | [0,1] |
| 2 | Dice / Sørensen / Czekanowski | `2a/(2a+b+c)` | | `2a_w/(2a_w+b_w+c_w)` | [0,1] |
| 3 | Cosine / Ochiai / Otsuka | `a/√((a+b)(a+c))` | | `a_w/√((a_w+b_w)(a_w+c_w))` | [0,1] |
| 4 | Sorgenfrei | `a²/((a+b)(a+c))` | | substitute `_w` counts | [0,1] |
| 5 | Soergel | D `= (b+c)/(a+b+c)` | | `(b_w+c_w)/(a_w+b_w+c_w)` | [0,1] |
| 6 | Sokal–Sneath-1 (RDKit `SokalSimilarity`) | `a/(a+2b+2c)` | | substitute `_w` | [0,1] |
| 7 | 3w-Jaccard | `3a/(3a+b+c)` | | substitute `_w` | [0,1] |
| 8 | Kulczynski-1 | `a/(b+c)` | | `a_w/(b_w+c_w)` | [0,∞) |
| 9 | Kulczynski-2 (RDKit `KulczynskiSimilarity`) | `½(a/(a+b) + a/(a+c))` | | substitute `_w` | [0,1] |
| 10 | McConnaughey | `(a²−bc)/((a+b)(a+c))` | | substitute `_w` | [−1,1] |
| 11 | Johnson | `a/(a+b) + a/(a+c)` | | substitute `_w` | [0,2] |
| 12 | Sokal–Sneath-4 (Anderberg) | `¼(a/(a+b) + a/(a+c) + d/(b+d) + d/(c+d))` | ⊘ | substitute `_w` | [0,1] |
| 13 | Simpson / Asymmetric / overlap | `a/min(a+b, a+c)` | | `a_w/min(a_w+b_w, a_w+c_w)` | [0,1] |
| 14 | Braun-Blanquet | `a/max(a+b, a+c)` | | `a_w/max(a_w+b_w, a_w+c_w)` | [0,1] |
| 15 | Tversky (α,β) | `a/(a + αb + βc)` | | `a_w/(a_w + αb_w + βc_w)` | [0,1] |
| 16 | Russell–Rao | `a/n` | ⊘ | `a_w/n_w` | [0,1] |
| 17 | Intersection (raw) | `a` | | `a_w` | [0,n] |
| 18 | Sokal–Michener / Simple Matching (RDKit `AllBitSimilarity`) | `(a+d)/n` | ⊘ | `(a_w+d_w)/n_w` | [0,1] |
| 19 | Rogers–Tanimoto | `(a+d)/(a+d+2(b+c))` | ⊘ | substitute `_w` | [0,1] |
| 20 | Gower–Legendre θ=½ (= Sokal–Sneath-2) | `(a+d)/(a+d+½(b+c))` | ⊘ | substitute `_w` | [0,1] |
| 21 | Hamann | `((a+d) − (b+c))/n` | ⊘ | substitute `_w`, `n_w` | [−1,1] |
| 22 | Faith | `(a + 0.5d)/n` | ⊘ | `(a_w + 0.5d_w)/n_w` | [0,1] |
| 23 | Rogot–Goldberg | `a/(2a+b+c) + d/(2d+b+c)` | ⊘ | substitute `_w` | [0,1] |
| 24 | Baroni-Urbani–Buser-1 | `(√(ad) + a)/(√(ad) + a + b + c)` | ⊘ | substitute `_w` | [0,1] |
| 25 | Baroni-Urbani–Buser-2 | `(√(ad) + a − b − c)/(√(ad) + a + b + c)` | ⊘ | substitute `_w` | [−1,1] |
| 26 | Yule Q | `(ad − bc)/(ad + bc)` | ⊘ | substitute `_w` | [−1,1] |
| 27 | Michael | `4(ad − bc)/((a+d)² + (b+c)²)` | ⊘ | substitute `_w` | [−1,1] |
| 28 | Pearson–Heron-1 (φ / MCC) | `(ad − bc)/√((a+b)(a+c)(b+d)(c+d))` | ⊘ | substitute `_w` | [−1,1] |
| 29 | Ochiai-2 (= Sokal–Sneath-5) | `ad/√((a+b)(a+c)(b+d)(c+d))` | ⊘ | substitute `_w` | [0,1] |
| 30 | Forbes-1 | `n·a/((a+b)(a+c))` | ⊘ | `n_w a_w/((a_w+b_w)(a_w+c_w))` | [0,n] |
| 31 | Forbes-2 | `(n·a − (a+b)(a+c)) / (n·min(a+b,a+c) − (a+b)(a+c))` | ⊘ | substitute `_w` | (−∞,1] |
| 32 | Mountford | `2a/(a(b+c) + 2bc)` | | substitute `_w` (**not** scale-invariant) | [0,∞) |
| 33 | Derived-Jaccard | `log(1+a)/log(1+a+b+c)` | | `log(1+a_w)/log(1+a_w+b_w+c_w)` (**not** scale-invariant) | [0,1] |
| 34 | Hellinger (binary form) | D `= 2√(1 − a/√((a+b)(a+c)))` | | via weighted cosine | [0,2] |
| 35 | OnBit (RDKit `OnBitSimilarity`) | `a/(a+b+c)` — **alias of Tanimoto** | | as Tanimoto | [0,1] |

Also defined-but-degenerate on binary (see Section 2): Manhattan/Hamming, Euclidean, squared Euclidean, all Minkowski `p`, Chebyshev, Canberra, correlation. These are listed under continuous.

### 1b. Count fingerprints (non-negative integers)

Requires `x_k, y_k ≥ 0`. No centring, no standardisation, no shifting. Note the fork: **two inequivalent generalisations of Tanimoto and of Dice exist**, they are both called "Tanimoto"/"Dice", and RDKit implements the min-based one while `simeval.py` implements the dot-product one.

| # | Name | Formula | Weighted form | Notes |
|---|---|---|---|---|
| 1 | Tanimoto, min/max (Ruzicka, weighted Jaccard, **RDKit count Tanimoto**) | `Σ_k min(x_k,y_k) / Σ_k max(x_k,y_k)` | `Σ w_k min / Σ w_k max` | true metric complement |
| 2 | Tanimoto, dot-product (extended Jaccard; Ralaivola 2005, tabulated Bajusz 2015; **simeval**) | `⟨x,y⟩/(‖x‖² + ‖y‖² − ⟨x,y⟩)` | `Σw x y/(Σw x² + Σw y² − Σw x y)` | **not** a metric complement on counts |
| 3 | Dice, min form (**RDKit count Dice**) | `2Σ_k min(x_k,y_k)/(Σx + Σy)` | `2Σw min/(Σw x + Σw y)` | = 1 − Bray-Curtis |
| 4 | Dice, dot form (**simeval**) | `2⟨x,y⟩/(‖x‖² + ‖y‖²)` | `2Σw x y/(Σw x² + Σw y²)` | |
| 5 | Soergel | D `= Σ|x−y| / Σ max(x,y)` | `Σw|x−y| / Σw max(x,y)` | true metric |
| 6 | Bray–Curtis (quantitative Sørensen) | D `= Σ|x−y| / Σ(x+y)` | `Σw|x−y| / Σw|x+y|` | semimetric |
| 7 | Cosine | `⟨x,y⟩/(‖x‖‖y‖)` | `Σw x y/√(Σw x² · Σw y²)` | discards magnitude |
| 8 | Tversky, min form | `Σmin / (Σmin + α Σmax(x−y,0) + β Σmax(y−x,0))` | weight each sum | |
| 9 | Canberra | D `= Σ_k |x_k−y_k|/(|x_k|+|y_k|)`, `0/0 := 0` | `Σ w_k · |x_k−y_k|/(|x_k|+|y_k|)` | true metric |
| 10 | Jensen–Shannon distance | `√(½ Σ_k [p_k log(2p_k/(p_k+q_k)) + q_k log(2q_k/(p_k+q_k))])` after L1-normalising | `√(½ Σ_k w_k [...])` | true metric; base-2 for [0,1] |
| 11 | Hellinger / Bhattacharyya | `√(1 − Σ_k √(p_k q_k))` after L1-normalising | `√(Σ_k w_k (√p_k − √q_k)²)` | true metric |

### 1c. Standardised continuous features (signed permitted)

These are the only coefficients defined on z-scored / signed input. Column standardisation with the scaler fit on **train rows only**, then average-1 weights.

| # | Name | Formula | Metric? | Weighted form |
|---|---|---|---|---|
| 1 | Euclidean (L2) | `√(Σ_k (x_k−y_k)²)` | yes (w>0) | `√(Σ_k w_k (x_k−y_k)²)` |
| 2 | Squared Euclidean | `Σ_k (x_k−y_k)²` | **no** (semimetric) | `Σ_k w_k (x_k−y_k)²` |
| 3 | Standardised Euclidean (`seuclidean`) | `√(Σ_k (x_k−y_k)²/V_k)` | yes | `√(Σ_k w_k (x_k−y_k)²/V_k)` — no scipy hook |
| 4 | Mahalanobis | `√((x−y)ᵀ Σ⁻¹ (x−y))` | yes (Σ PD) | `√((x−y)ᵀ W^½ Σ⁻¹ W^½ (x−y))`, **Σ⁻¹ held fixed** |
| 5 | Manhattan (L1 / cityblock) | `Σ_k |x_k−y_k|` | yes | `Σ_k w_k |x_k−y_k|` |
| 6 | Minkowski, p ≥ 1 | `(Σ_k |x_k−y_k|^p)^(1/p)` | yes | `(Σ_k w_k |x_k−y_k|^p)^(1/p)` |
| 7 | Fractional Minkowski, 0<p<1 | `(Σ_k |x_k−y_k|^p)^(1/p)` | **no** (semimetric) | `(Σ_k w_k |x_k−y_k|^p)^(1/p)` |
| 8 | Chebyshev (L∞) | `max_k |x_k−y_k|` | yes | `max_k w_k |x_k−y_k|` — **not** scipy's `w=` |
| 9 | Cosine | `1 − ⟨x,y⟩/(‖x‖‖y‖)` | **no** | `1 − Σw x y/√(Σw x² Σw y²)` |
| 10 | Correlation (Pearson distance) | `1 − ⟨x−x̄, y−ȳ⟩/(‖x−x̄‖‖y−ȳ‖)`, `x̄` = mean over **features within the row** | **no** | weighted mean `x̄_w = Σw_k x_k/Σw_k`, then weight numerator and both denominator sums |
| 11 | Distance correlation (dCor) | `R_n = V_n(X,Y)/√(V_n(X)V_n(Y))` on double-centred distance matrices | dependence measure, not a pairwise distance | via `X diag(√w)` |
| 12 | Linear CKA / RV coefficient | `‖YᵀX‖_F² / (‖XᵀX‖_F ‖YᵀY‖_F)` on column-centred X, Y | representation-vs-representation | `X diag(√w)` |

Entries 11–12 return **one scalar per pair of representations**, not an (n,n) matrix. They cannot be substituted at a site where Tanimoto or Euclidean sits.

---

## 2. DUPLICATES AND DEGENERACIES

A pair is listed as **exact** if the two return identical numbers, and **rank-identical** if one is a strictly monotone function of the other — the latter still means every Spearman-scored kernel, every k-NN neighbour set, and every AUROC returns the same value, so it is a duplicate cell for any rank-based objective. Only absolute-magnitude consumers (a fixed similarity cutoff, a kernel bandwidth, distance-weighted k-NN weights, a mean-distance aggregate) can separate rank-identical pairs.

### 2a. Binary fingerprints

**Class A — the Tanimoto class.** With `T = a/(a+b+c)`:

| Coefficient | Relation to T | Status |
|---|---|---|
| Soergel | `1 − T` exactly | exact |
| Bray–Curtis (binary) | `= Dice distance` exactly | exact (to Dice) |
| Dice | `2T/(1+T)` | rank-identical |
| Sokal–Sneath-1 | `T/(2−T)` | rank-identical |
| 3w-Jaccard | `3T/(2T+1)` | rank-identical |
| Kulczynski-1 | `T/(1−T)` | rank-identical |
| OnBit (RDKit) | `= T` | alias |
| Tversky(α=β=t), any t>0 | rank-identical to T for **all** t | rank-identical |
| Tversky(1,1) | `= T` exactly | exact |
| Tversky(0.5,0.5) | `= Dice` exactly | exact |
| Symmetrised Tversky(0.7,0.3) | Spearman 0.9996 with Dice; **exactly** Dice when `|x| = |y|` | near-duplicate |
| Lance–Williams distance | `= Dice distance` exactly | exact |

Nine to twelve names, one ranking.

**Class B — the Hamming class.** All are strictly monotone in `m = b + c`:

Manhattan (`= m`), Hamming (`= m/n`), squared Euclidean (`= m`), Euclidean (`= √m`), **every** Minkowski `p` (`= m^(1/p)`, so the `p` axis is a complete no-op on binary), Canberra (`= m` exactly), Sokal–Michener (`1 − m/n`), Rogers–Tanimoto (`= SM/(2−SM)`), Gower–Legendre θ=½ (`= (n−m)/(n−m/2)`), Sokal–Sneath-2 (identical to Gower–Legendre θ=½), Hamann (`= 1 − 2m/n`). **Eleven names, one ranking.** Faith (`(a+0.5d)/n`) is *near*-identical to Sokal–Michener (Spearman 0.998) but not exact, and equals `(RussellRao + SokalMichener)/2` exactly.

**Class C — the cosine class.** Cosine/Ochiai; Sorgenfrei `= cos²`; binary Hellinger `= 2√(1−cos)`. Three names, one ranking.

**Class D — the Kulczynski-2 class.** Kulczynski-2; McConnaughey `= 2·K2 − 1`; Johnson `= 2·K2`; Driver–Kroeber `= K2`. Four names, one ranking.

**Class E — the φ class.** Pearson–Heron-1 = φ = Matthews correlation coefficient; correlation distance on binary vectors `= 1 − φ` **exactly**. Michael is close (Spearman 0.9935 with φ, sign-identical always) but genuinely distinct.

**Other exact identities:**

| Pair | Relation |
|---|---|
| Russell–Rao ↔ Intersection | `RR = a/n`, `n` constant → rank-identical to raw shared-bit count |
| BUB-2 ↔ BUB-1 | `BUB2 = 2·BUB1 − 1` exactly → rank-identical |
| Ochiai-2 ↔ Sokal–Sneath-5 | identical formula, two names |
| Forbes-2 ↔ Loevinger's H (RIOC) | identical, verified over 200k contingency tables |
| Rogot–Goldberg ↔ Scott's π | `RG = (π+1)/2` exactly; also `= ½[Dice(x,y) + Dice(¬x,¬y)]` |
| Yule Q ↔ odds ratio | `Q = (OR−1)/(OR+1)` → rank-identical; also = Goodman–Kruskal γ on 2×2 |
| Sokal–Sneath-4 ↔ Kulczynski-2 | `SS4 = ½[K2(x,y) + K2(¬x,¬y)]` |
| Chebyshev (binary) | `= 1[x ≠ y]`, the discrete metric — one distinct value across all non-identical pairs |
| Sokal–Michener ↔ Rand index | same algebra |

**The containment family.** Let `r1 = a/(a+b)`, `r2 = a/(a+c)`. Then Simpson `= max(r1,r2)`, Braun-Blanquet `= min(r1,r2)`, Cosine `= √(r1 r2)`, Kulczynski-2 `= (r1+r2)/2`, McConnaughey `= r1+r2−1`, Johnson `= r1+r2`, Dice = harmonic mean of `r1,r2`. **Seven "independent" coefficients with two degrees of freedom between them.** Simpson × Braun-Blanquet `= cos²` exactly. Two of them (K2, McConnaughey) are already rank-identical.

**Fixed-density collapse.** On any binary family where every vector has the same popcount `k` (fixed-density fingerprints, MinHash sketches, top-k thresholded descriptors), Tanimoto, Dice, Cosine, Simpson, Braun-Blanquet, Kulczynski-2, Sokal–Sneath-4 and Faith all become strictly increasing functions of `a` alone and are therefore **mutually rank-identical**. Measured Spearman exactly 1.000.

**Not duplicates, despite the folklore.** Cosine is *not* rank-equivalent to Tanimoto/Dice — counterexample `(|x|=1,|y|=100,a=1)` vs `(|x|=|y|=50,a=1)` gives Tanimoto 0.0100 vs 0.0101 and Cosine 0.100 vs 0.020, i.e. the pairs swap. Tanimoto is *not* rank-equivalent to Manhattan/Hamming — measured Spearman ≈ 0.62 on random binary, and explicit sign-flipping counterexample `(a=8,|x|=|y|=10)` vs `(a=1,|x|=|y|=2)`. Mountford and Michael have no duplicate.

### 2b. Count fingerprints

| Pair | Relation | Status |
|---|---|---|
| Soergel ↔ Ruzicka ↔ RDKit count Tanimoto | `Soergel = 1 − Σmin/Σmax` exactly | **exact — three names for one coefficient** |
| Bray–Curtis ↔ Soergel | `BC = S/(2−S)`, `S = 2BC/(1+BC)` | rank-identical (Spearman exactly 1.000) |
| min-Dice ↔ Bray–Curtis | `min-Dice = 1 − BC` exactly | exact |
| min-Dice ↔ Ruzicka | `D = 2R/(1+R)` | rank-identical |
| dot-Dice ↔ dot-Tanimoto | `D = 2T/(1+T)` | rank-identical |
| Canberra ↔ binarised Hamming | on sparse Poisson counts, corr 0.996; 98.7% of Canberra's variance is the non-shared-feature count | near-duplicate |
| Cosine ↔ Euclidean on L2-normalised rows | `d² = 2 − 2cos` | rank-identical |

**Consequence:** the count family collapses to roughly **four** independent geometries — the min/max family (Ruzicka = Soergel = Bray-Curtis = min-Dice), the dot-product family (extended Jaccard = dot-Dice), cosine, and Canberra/Manhattan/Euclidean. `simeval.py`'s current count set (`tanimoto` dot, `dice` dot, `cosine`, `soergel`, `euclidean`) already contains one exact duplicate pair (`tanimoto`/`dice` are rank-identical) and `soergel` is unlabelled as being the field-standard count Tanimoto.

The min/max and dot-product forms are genuinely different on counts: measured Spearman 0.96 over 300 count pairs, absolute divergence up to 0.16, and on one worked example `T_dot = 0.5747` vs `Ruzicka = 0.4923`. **Only the min/max form's complement is a metric.** The dot-product complement violates the triangle inequality on non-negative counts (Kosub's d=1 counterexample `{1,2,4}`: `1−T(1,4) = 0.6923 > 0.6667`). `√(1 − T_dot)` *is* a metric on all of ℝᵈ.

### 2c. Standardised continuous features

| Pair | Relation | Status |
|---|---|---|
| seuclidean ↔ Euclidean on z-scored features | identical, given the same ddof | **exact** |
| seuclidean ↔ Mahalanobis with diagonal covariance | identical | exact |
| Mahalanobis ↔ Euclidean on ZCA/PCA-whitened features | identical | exact |
| Correlation ↔ cosine on row-mean-centred rows | identical | exact |
| Correlation ↔ Euclidean on row-z-scored rows | `‖x̂−ŷ‖² = 2m(1−r)` | rank-identical (Spearman 1.000) |
| Cosine ↔ Euclidean on L2-normalised rows | `d² = 2 − 2cos` | rank-identical (Spearman 1.000) |
| Squared Euclidean ↔ Euclidean | monotone | rank-identical **per pairwise ranking only** — diverges under any linear aggregation (summed block distances, mean distance to a set, k-means inertia, MDS stress) |
| Minkowski p=1 / p=2 / p→∞ | Manhattan / Euclidean / Chebyshev | same family, three settings |
| dCor with squared distances ↔ RV ↔ linear CKA | identical to 15 dp | exact |
| Linear CKA ↔ RV coefficient ↔ Tucker congruence | identical | exact |
| Linear CKA (p1=p2=1) ↔ Pearson r² | identical | exact |

**Independent continuous geometries: about seven** — Euclidean (with seuclidean/Mahalanobis/correlation/cosine as preprocessing variants of it), Manhattan, Chebyshev, fractional Minkowski, Canberra, Bray-Curtis, and the block-level dCor/CKA. Fractional Minkowski `p=0.5` and `p=0.3` correlate at Spearman 0.990 with each other and 0.96/0.92 with Manhattan.

### 2d. Cross-family traps

- **RDKit `SokalSimilarity` is Sokal–Sneath-1, not Sokal–Michener.** The Sokal–Michener coefficient is RDKit's `AllBitSimilarity`. Discrepancy up to 0.44.
- **RDKit `KulczynskiSimilarity` is Kulczynski-2; scipy `kulczynski1` is Kulczynski-1.** Different coefficients, same name stem.
- **scipy `sokalmichener` returns bit-identical values to `rogerstanimoto`** and is deprecated as of scipy 1.15, removal scheduled 1.17.
- **RDKit's `RusselSimilarity` header docstring says `(bv1&bv2)_o / bv1_o`; the implementation divides by `getNumBits()`.** The docstring is wrong. `RusselSimilarity(fp,fp) = |x|/n ≈ 0.02`, not 1.0.
- **RDKit's `RogotGoldbergSimilarity` Python docstring is a copy-paste of Russel's** and does not describe the implemented formula.
- **scipy `kulczynski1` is deprecated (removal 1.17)** with documented replacement `1/jaccard(u,v) − 1`.

---

## 3. PREPROCESSING, PER FAMILY

### 3a. Binary set-overlap coefficients (Section 1a)

| Requirement | What you get if you skip it |
|---|---|
| **Hard binarisation to `{0,1}`** (threshold at count > 0), enforced by assert, not cast | Nothing raises. `simeval.dist_tanimoto`-style dot-product forms return plausible in-range numbers computed on the wrong quantity. The classic 2×2 forms return values **outside their own range**: Dice's product-form weighting on `x=(3,0), y=(1,0)` returns 1.5; McConnaughey on a count vector against **itself** returns 6.42; Simpson on `x=y=(3,0)` returns 3.0; Sokal–Sneath-1 on `[1,0,0]` vs `[3,1,0]` returns **−2.0** (this is scipy's own docstring example); Tversky on counts through `simeval` gives similarity > 1 and therefore **negative distances**, which sort to the front of a k-NN. The binary Tanimoto formula on counts gives negative values or `ZeroDivisionError` (`x=y=[2]` → `4/0`), which at least fails loudly. |
| **Frozen fingerprint length `n`**, identical across every pair pooled or thresholded | For coefficients marked ⊘, the value moves with the fold width alone. Measured on one fixed molecule pair with only `fpSize` changed: Yule Q = 0.286 (128 bits) → 0.809 (4096); Russell–Rao = 0.03125 (32 bits) → 0.00049 (2048) while Tanimoto stays 0.1429; Sokal–Sneath-4 c=0 floor = 0.2170 (166 bits) → 0.4875 (2048); Derived-Jaccard at fixed Tanimoto 0.5 goes 0.733 (union 6) → 0.894 (union 510). A results table mixing MACCS-167 with ECFP-2048 is comparing numbers on different scales, silently. Folding also *inflates* Tanimoto itself: mean pairwise Tanimoto on 8 drug-like SMILES was 0.1366 at 8192 bits, 0.2132 at 128 — a 56% rise from collision alone. |
| **Do not prune all-zero or all-one columns** for ⊘ coefficients | Dropping all-zero bits is a no-op for the `d`-free coefficients (they ignore 0-0 matches) but changes every ⊘ value. Dropping **all-one** bits changes *every* coefficient including Tanimoto, since each removed always-on bit decrements `a`, `|x|` and `|y|` together — measured 0.294 shift on MACCS-like data. `f18`'s constant-column drop currently does exactly this. |
| **Zero-vector (empty fingerprint) guard** | Library conventions **disagree with each other and with themselves**, and none raises. RDKit `TanimotoSimilarity(empty, empty)` = **1.0** in every released version through 2025_03; `CosineSimilarity(empty, empty)` = **0.0**; `DiceSimilarity(empty, empty)` = 0.0; count-vector Tanimoto on empty = 0.0. scipy `jaccard(0,0)` = 0.0 distance (similarity 1); scipy `dice(0,0)` = **nan**; scipy `cosine(0,0)` = nan. `simeval.dist_cosine` sets `nrm[nrm==0] = 1.0` and **does not fill the diagonal**, so a zero row gets self-distance **1.0** and equal distance to everything, while `dist_tanimoto` and `dist_dice` both call `fill_diagonal(sim, 1.0)`. Russell–Rao and Kulczynski-1 have non-zero self-distance for *any* vector, not just empty ones. |
| **Per-block scaling: not applicable but not harmless** | Set coefficients have no block-scaling knob, but concatenating a 2048-bit block with a 167-bit block gives the wide block ~92% of the intersection and union counts. Column count *is* block weight, and there is no scale parameter to correct it — only a per-feature weight vector (Section 4). |

### 3b. Count-fingerprint coefficients

| Requirement | What you get if you skip it |
|---|---|
| **Strict non-negativity, raw counts** | Soergel on z-scored features: measured range 1.42–3.22 (a coefficient bounded in [0,1]), 1,055 triangle violations per 20,000 triples, and where `Σmax ≤ 0` the standard `np.where(den > 0, num/den, 0.0)` guard returns **distance 0**, i.e. "identical". Concretely `x=(−3,−2,−1), y=(−1,−2,−3)` → true ratio −1.0, code returns 0.0. Bray-Curtis on z-scored input: median exactly 1.00, 50–56% of pairs above 1, spread collapsing as `CV = 0.277 (d=16) → 0.034 (d=1024)` — a near-constant matrix that looks fine. It is also **non-monotone in separation**: with `x=[1]`, `y=[−0.99]` (distance 1.99) scores 199 while `y=[−100]` (distance 101) scores 1.02. Canberra on z-scores **saturates**: `x=1e-6, y=−1e-6` returns exactly 1.0, identical to `x=3, y=−3`, while genuinely similar `x=2, y=2.5` returns 0.111 — the ranking inverts. Jensen–Shannon on all-negative rows: scipy normalises by the negative sum, flipping every sign into a valid-looking probability vector, and returns a finite plausible number with **no warning**; mixed-sign rows return `inf`, all-zero rows return `nan`. |
| **Do not shift to make non-negative** | These coefficients are scale-equivariant but **not translation-invariant**. Shifting z-scores up by a constant restores a technically valid metric whose answer depends entirely on the arbitrary offset. Measured on 200 z-scored points in 20 dimensions: offset +3 vs +1000 changes the nearest neighbour of 26% of points, and the distance range collapses from [0.087, 0.343] to [0.0005, 0.0020] — an RBF kernel built on it flattens to a constant. min-max scaling has the same defect for Canberra: it plants an exact zero at each feature's minimum, manufacturing the near-zero pathology it was meant to avoid (raw D = 0.905 → min-max D = 2.639 on identical data). |
| **Commensurate feature scales** | Soergel, Bray-Curtis and the min/max family have no per-feature normalisation. Measured on a raw-descriptor-like block with scales 0.1–1000: the single scale-1000 column alone reproduced the full 10-column Bray-Curtis distance at Spearman 0.898, with seven columns at |ρ| < 0.09. You get a finished number that is a one-column distance. |
| **Row L1-normalisation** (composition rather than abundance) — a modelling choice, not a validity requirement | Bray-Curtis is not magnitude-invariant: `BC(x, 2x) = 1/3`, so a molecule and a copy of itself with doubled counts sit at distance 0.333. Cosine is the opposite extreme: on RDKit count Morgan fingerprints, a molecule versus a structure containing two disconnected copies of itself scores **cosine = 1.0000 exactly** (tested on ethanol, phenol, paracetamol) while count-Tanimoto gives 0.6667. Cosine on counts literally cannot distinguish a molecule from twice itself. |
| **Pick and document one Tanimoto/Dice convention** | RDKit and `simeval` disagree by up to 0.16 absolute on the same count vectors (worked example: RDKit Dice 0.387, dot Dice 0.644 on a real Morgan count pair — a 0.26 gap). Any number cross-referenced against an externally computed Dice will not match. |
| **RDKit count path takes `abs()` of every entry** | `TanimotoSimilarity(SparseIntVect{0:+5}, SparseIntVect{0:−5})` returns **1.0**. Two samples maximally opposed on every signed feature come back as identical. |

### 3c. Standardised continuous features

| Requirement | What you get if you skip it |
|---|---|
| **Per-column standardisation, scaler fit on TRAIN rows only** | The highest-raw-variance column owns the distance. Measured with three features at sd 1, 1 and 300: unstandardised Euclidean correlates 1.000 with the single large-scale feature and 0.495 with the correct standardised distance. On a real 154-column RDKit descriptor block, raw Chebyshev correlates 0.976 with a single-descriptor distance (Ipc supplies the argmax in 74.9% of pairs) and only 0.205 with the standardised answer. Manhattan and Euclidean on the same block: 0.959 and 0.975. No error. |
| **Per-block width normalisation, on top of standardisation** | **This is the requirement most often stated as unnecessary and it is the load-bearing one.** After z-scoring, every column has unit variance, so a block's expected squared-distance contribution is *exactly proportional to its column count*. Measured with a 2048-dim and a 128-dim block, both z-scored: squared-distance contributions 3977 vs 241, ratio 16.44 = 2048/128 to two figures. Dividing each block by `√width` equalises **Euclidean** (mean distance 1.406 / 1.412 / 1.411 at m = 10 / 100 / 400 — flat) but **not Manhattan** (3.637 / 11.282 / 22.541 — still growing as `√m`). **The correct divisor is `√m` for L2-family coefficients and `m` for L1-family coefficients** (Manhattan, Canberra, Bray-Curtis, and the binary set coefficients, all of which are sums of per-column terms). Using one divisor for both makes an "equal-block" arm mean two different things depending on the coefficient in the cell. Chebyshev is a third case again: it grows only logarithmically with width (mean 2.93 / 3.68 / 4.31 / 4.88 for d = 16 / 64 / 256 / 1024), so a `√m` normaliser over-corrects it. Cosine and correlation need neither, being scale-invariant *globally* — but that gives **zero** protection against block imbalance: measured on a 2048-bit Morgan block (mean squared norm 44.9) concatenated with a 200-dim standardised block (mean squared norm 198.9), the combined cosine distance correlates 0.997 with the *small* block and 0.069 with the wide one. |
| **Width normalisation before any cross-representation comparison** | Raw L_p grows as `d^(1/p)`. Mean L2 on standard normals: 11.25 (d=64), 22.62 (d=256), 45.30 (d=1024) — all 1.41 after dividing by `√d`. Rank a rep pool by mean pairwise distance and you rank it by embedding width. Euclidean on binary is unbounded in [0, √n] and scales as `√n_bits`. |
| **Do NOT standardise binary or count fingerprints** | z-scoring a sparse fingerprint divides each bit by `√(p_k(1−p_k))`, amplifying rare bits without bound. Measured on a 512-bit fingerprint with prevalences 0.5 down to 0.002: one differing rare bit contributes 301 units of squared distance, one differing common bit contributes 4 — a 75× amplification growing as `1/p`. The rarest 10% of bits go from 0.4% of expected squared distance raw to 10.0% standardised. Standardised Euclidean on binary is `√(Σ_{k: x_k≠y_k} 1/(p_k(1−p_k)))` — an inverse-variance-weighted Hamming distance, which has no standard cheminformatics name and correlates only 0.474 with raw Euclidean and 0.186 with Tanimoto distance. Note also that the amplification is **symmetric in p**: a bit set 99% of the time is amplified exactly as much as one set 1%. |
| **Zero-vector / constant-row handling** | Cosine on a zero row: scipy returns `nan`, `simeval` returns similarity 0 to everything **including itself**. Correlation requires a non-constant **row** (variance taken across features within the row) — a constant row gives `nan` with no exception, and dropping all-constant *columns* does not prevent this and can *create* it: `Y = [[5,5,5,9],[1,2,3,9],[4,0,7,9]]` has row 1 with std 1.732 until the constant column 4 is dropped, at which point row 1 becomes `[5,5,5]` and the whole first row and column of D turns `nan`. |
| **Invertible covariance for Mahalanobis** | The guard must be `rank(Σ) == d`, not `d < n`. With column centring `rank(Σ) ≤ min(d, n−1)`, so `d = n−1` passes a `d >= n` guard and produces a **constant distance matrix** — every off-diagonal entry exactly `√(2n)` (measured 8.831761 at n=20, 10.862780 at n=60, 20.000000 at n=200; std across all off-diagonals 3.7e-15). Degradation is continuous, not a cliff: at n=60, off-diagonal std runs 0.980 (d=10) → 0.610 (d=40) → 0.248 (d=55) → 0.000 (d=59). Separately, `sklearn.covariance.EmpiricalCovariance.precision_` is `scipy.linalg.pinvh`, a **pseudo-inverse with no regularisation**, so a rank-deficient covariance at `d < n` succeeds silently: measured on 500 rows × 100 bits with 40 dead bits (rank 60), it returns a plausible 3.89–15.66 spread with zero NaN. In the `p ≥ n` regime the pseudo-inverse **zeroes** the near-null directions rather than amplifying them, giving a matrix with no contrast at all (Spearman against Euclidean: −0.08, −0.16, −0.01 across three trials). By contrast `np.linalg.inv` on an exactly singular covariance does not raise either — it returns entries of magnitude 4.2e16 and `cdist` then returns an **all-NaN** matrix. The cheapest possible detector is an assertion on the standard deviation of the off-diagonal. |
| **Weighting order: standardise FIRST, then weight** | Reversing it is an **exact no-op** — a per-column `StandardScaler` cancels a per-column multiply, measured at 8.9e-16. The weighted arm comes out byte-identical to its unweighted twin and reports a fabricated null. Note also that the scaler must be *refit* on the weighted matrix for the cancellation to be exact; a train-fitted scaler applied to weighted data is not a no-op. And `w_k = 0` is not a no-op in either order — the column is zeroed outright, which turns every weighted metric into a **pseudometric** (distinct points at distance 0). |

---

## 4. WEIGHTED FORMS

`w_k ≥ 0` is required throughout. Negative weights break every one of these: they put a negative radicand under a square root (cosine, Ochiai-2, BUB), flip `min` to `max` (Soergel, Ruzicka, Simpson, Braun-Blanquet), or drive a ratio denominator through zero (Tanimoto, Dice, Tversky, Bray-Curtis). scipy raises `ValueError` on negative weights; `np.sqrt(w)` does not — it emits a `RuntimeWarning` and returns an **all-NaN distance matrix**, and `noise_mitigation_methods.py` suppresses that warning at import.

### 4a. The three fold-in exponents

The critical operational fact: a per-feature weighting can be applied either by passing `w` into the formula, or by rescaling the columns of `X` and reusing the unweighted formula. **The correct column rescale is different for each coefficient family**, and using the wrong exponent produces a plausible, monotone-but-different distance matrix with no error.

| Coefficient family | Correct column rescale | Rationale |
|---|---|---|
| Euclidean, squared Euclidean, cosine, correlation-without-centring, dot-product Tanimoto, dot-product Dice, Sorgenfrei, dCor, linear CKA | `X · diag(√w)` | quantities are quadratic in `x`; `w` multiplies `x_k y_k` and `x_k²` alike |
| Manhattan, Chebyshev, Bray-Curtis, Soergel, Ruzicka/min-max Tanimoto, min-form Dice, min-form Tversky, **all binary set coefficients** | `X · diag(w)` | quantities are linear in `x`; `|w x − w y| = w|x−y|` and `max(wx,wy) = w·max(x,y)` for `w ≥ 0` |
| Minkowski, exponent `p` | `X · diag(w^(1/p))` | `w_k |Δ_k|^p = (w_k^(1/p)|Δ_k|)^p`. **This is the trap.** At `p = 3`, folding in `w` directly applies `w³`. At `p = 0.5`, folding in `√w` applies `w^0.25`, flattening a 100× weight spread to 3.16×. Legacy `scipy.wminkowski` used the *other* convention `(Σ|w_i Δ_i|^p)^(1/p)`, so old weight vectors carried over unchanged are wrong by exactly this exponent. |
| **Canberra, Mahalanobis (re-estimated), seuclidean** | **no rescale works** | see 4c |

Two rescale-based sanity tests both **fail to detect** a wrong exponent: `w = 1` reduces correctly under every exponent, and a uniform `w = c` is either a no-op or a global rescale under every exponent. **Test at a non-uniform `w ≠ 1` against a reference implementation.**

### 4b. Coefficients whose weight MUST go inside the formula (ratios of sums)

These cannot be implemented by any column rescale, because the coefficient is a ratio in which a per-column factor cancels, or because the weight enters a centring step.

**Canberra.** `d_w = Σ_k w_k · |x_k − y_k| / (|x_k| + |y_k|)`. The weight multiplies the whole per-coordinate ratio, *after* the division. Any rescale is an **exact no-op**: `Σ |a_k x_k − a_k y_k|/(|a_k x_k| + |a_k y_k|) = Σ |x_k−y_k|/(|x_k|+|y_k|)` for all `a_k > 0`. Measured: `compute_distance_matrix(X, 'canberra', feature_weights=w)` differs from the unweighted call by **4.4e-16** over a full matrix, while the same weights move Euclidean by 6.79. Available natively as `scipy.spatial.distance.canberra(u, v, w=...)` and `cdist(..., 'canberra', w=w)`. Zero weights act as a hard on/off mask (via the `0/0 := 0` rule) with nothing in between.

**Mahalanobis.** With the covariance **re-estimated from the transformed data**, weighting is an exact no-op by affine invariance: `X' = XA` gives `Σ' = AΣA` and `Σ'⁻¹ = A⁻¹Σ⁻¹A⁻¹`, so the `A`'s cancel identically. Measured: `compute_distance_matrix(X, 'mahalanobis', feature_weights=w)` vs `feature_weights=None` differs by **3.55e-15**; standardisation is likewise inert (1.8e-15). The only well-defined weighted Mahalanobis holds the precision matrix **fixed**, estimated once on unweighted reference data and passed in as `VI`:
```
d_w(x,y) = √( (x−y)ᵀ W^½ Σ⁻¹ W^½ (x−y) ),   W = diag(w)
```
This reduces to Mahalanobis at `w = 1` and deliberately breaks the affine invariance that defines Mahalanobis. It requires a code path separating covariance estimation from distance computation, which does not currently exist. Shrinkage (`Σ + λ diag(1/w)`) is a different estimator and does **not** reduce to the unweighted form at `w = 1` unless `λ = 0`.

**Standardised Euclidean.** `seuclidean` *is* weighted Euclidean with `w = 1/V`, so it takes no `w` of its own (`scipy` raises `TypeError` on `w=`). It is invariant to per-feature affine rescaling by construction (`x → ax+b` gives `V → a²V` and the ratio cancels), so column scaling as a weighting mechanism is **provably inert**. Combine importance with variance scaling as `euclidean(u, v, w = importance_k / V_k)`.

**Correlation.** `x̄_w = Σ_k w_k x_k / Σ_k w_k`; then
```
d_w = 1 − Σ_k w_k (x_k − x̄_w)(y_k − ȳ_w) / √( Σ_k w_k (x_k − x̄_w)² · Σ_k w_k (y_k − ȳ_w)² )
```
The weight enters the **centring mean** as well as all three sums, so no pre-rescale can reproduce it — rescaling changes the row profile and therefore perturbs the mean. Measured: true weighted 0.5383668, `√w` rescale 0.5371771. The divergence grows with weight skew: max distance error 0.0035 at half-weighting, **0.282** on a [0,2] scale at 90% zero weights. At `w_k = 0` the correct form **drops** the feature exactly; the rescale zeroes the coordinate in both vectors, which still pulls the row mean toward zero and still adds `(0 − mean)²` to both norms — a zero-importance feature keeps voting. Available natively as `cdist(X, X, 'correlation', w=w)`.

**Chebyshev.** `d_w = max_k w_k |x_k − y_k|` (the weighted Tchebycheff norm). This is **not** what scipy's `w=` computes. scipy defines the weighted L∞ as the `p→∞` limit of the weighted Minkowski, and since `w_k^(1/p) → 1` for any `w_k > 0`, its implementation is `max((w > 0) * abs(u - v))` — a pure **0/1 mask** with the magnitudes discarded. Verified: `minkowski(u, v, np.inf, [100,1,1,1,1])` returns exactly the unweighted value. Implement by `X · diag(w)` and call the unweighted metric; never pass `w=`.

### 4c. Weighted forms for the binary set coefficients

Every 2×2 coefficient in Section 1a takes a weighted form by substituting `a_w, b_w, c_w, d_w, n_w` for `a, b, c, d, n`. All reduce exactly at `w = 1`. Three global properties, all verified:

1. **Scale-invariance.** Every coefficient that is a ratio homogeneous of the same degree in `w` — which is all of them except Intersection, Kulczynski-1 (unbounded), Mountford (degree −1) and Derived-Jaccard (log) — satisfies `S_w(cw) = S_w(w)` for `c > 0`. **This means the average-1 vs sum-to-1 weight convention is a mathematical no-op for these coefficients.** Verified individually for Tanimoto, Tversky, Cosine, Canberra, Bray-Curtis, Soergel, Sokal–Sneath-4, Baroni-Urbani–Buser, Michael, Russell–Rao. The convention only bites for the unnormalised sums (Euclidean, Manhattan, Minkowski, Chebyshev, Intersection), where switching from average-1 to sum-to-1 divides every distance by `d^(1/p)` — rankings survive exactly (Spearman 1.000000), **absolute cutoffs and kernel bandwidths do not**.
2. **Integer weights = feature duplication.** For every one of these, `S_w` with integer `w_k` equals the unweighted coefficient on the fingerprint with feature `k` physically duplicated `w_k` times. Verified for Baroni-Urbani–Buser (to 12 dp), Soergel, Tversky, Minkowski. This is the formal statement of "column count is block weight".
3. **`w_k = 0` is exactly feature deletion** for the `d`-free coefficients, and is **not** feature deletion for the ⊘ coefficients, because it also shrinks `n_w` and `d_w`.

**Two exceptions where scale-invariance fails.** Mountford is degree −1 in `w` (numerator linear, denominator quadratic), so `S_w(λw) = S_w(w)/λ`; sum-to-1 weights inflate it by a factor of `n_features` (measured 3.7321 vs 0.5332 on the same weight vector). **Weights must be normalised to mean 1 for Mountford**, which happens to be this repo's convention. Derived-Jaccard is not scale-invariant either — measured `dJ = 0.620 / 0.781 / 0.864` for the same weights at ×0.1 / ×1 / ×10 while weighted Tanimoto stays at 0.4658 — and average-1 is again the only normalisation under which `w = 1` is the identity.

### 4d. Coefficients with no per-feature weighting

Only two entries in the whole list genuinely lack one, and both are block-level rather than pairwise: distance correlation and linear CKA both accept `X diag(√w)`, but the weighting acts on the whole representation rather than on individual pairs. Every other coefficient in Sections 1a–1c has a weighted form given above. **Statements in the existing repo documents that a weighted form "does not exist" for Minkowski, Chebyshev, Canberra, Correlation, Mahalanobis, Jensen–Shannon, Kulczynski-1, Tversky, Ochiai-2 or Baroni-Urbani–Buser are all incorrect** — a standard form exists in each case and reduces correctly at `w = 1`.

---

## 5. WHAT TO BUILD

Ordered. Items marked **[FIX]** are corrections to existing code; **[PORT]** already exists in `simeval.py` and needs moving into `src/kirby/`; **[NEW]** must be written.

### Tier 0 — bugs that make current weighted results meaningless

1. **[FIX] Replace the single `√w` pre-transform in `compute_distance_matrix` with a per-coefficient dispatch.** The current line applies `X_standardised · diag(√w)` to every metric. That is correct for **euclidean** and **cosine** and wrong for the other six it serves: `manhattan` and `chebyshev` receive effective weight `√w` (measured Spearman 0.976 against the intended weighting — not even rank-preserving); `braycurtis` receives `√w`; `minkowski_3` receives `w^1.5`; `canberra` receives **nothing at all** (exact no-op, 4.4e-16); `mahalanobis` receives **nothing at all** (exact no-op, 3.55e-15); `correlation` receives neither the weighted nor the unweighted form. Dispatch: `√w` for the L2/dot family, `w` for the L1/min family, `w^(1/p)` for Minkowski, and pass `w=` through to scipy for canberra and correlation. Drop `mahalanobis` from the weighted path or switch it to the fixed-precision congruence form.
2. **[FIX] Clamp weights at the entry point.** `np.sqrt(feature_weights)` on a signed importance vector returns an all-NaN matrix with only a `RuntimeWarning`, which this module suppresses at import. Add `np.clip(w, 0, None)` inside the helper and delete the seven external `np.maximum` workarounds currently scattered across `s02_hybrid_similarity.py` and `s08_optimal_dimension.py`.
3. **[FIX] Reconcile the two contradictory weight conventions inside `noise_mitigation_methods.py`.** Six further sites apply raw `X * feature_weights` with no standardisation and no square root, giving weighted Euclidean with effective weight `w²`. One module, one argument name, two incompatible meanings.
4. **[FIX] Mahalanobis rank guard.** Change `if d >= n: return None` to `if d >= n - 1: return None` **and** add `if np.linalg.matrix_rank(cov) < d: return None`. Add an assertion on the standard deviation of the off-diagonal as the cheap runtime detector for the constant-matrix regime.
5. **[FIX] Zero-row handling in `simeval.dist_cosine`.** It sets `nrm[nrm==0] = 1.0` and, unlike `dist_tanimoto` and `dist_dice`, never calls `np.fill_diagonal(sim, 1.0)`. A zero row currently has self-distance 1.0 and is an average-distance neighbour of everything.
6. **[FIX] Non-negativity assertion in `simeval.dist_soergel`.** Replace the `np.where(den > 0, num/den, 0.0)` guard with a hard assert. The guard was written for the all-zero-fingerprint case and silently reports antipodal pairs as identical on signed input.

### Tier 1 — port from `simeval.py` into `src/kirby/`, weighted

7. **[PORT + weight] Tanimoto.** Both forms, explicitly named. `tanimoto_dot` = `⟨x,y⟩/(‖x‖²+‖y‖²−⟨x,y⟩)`, weighted by `X·diag(√w)`. `tanimoto_minmax` = `Σmin/Σmax` (= RDKit's count Tanimoto = 1 − Soergel), weighted by `X·diag(w)`. Do **not** ship a single function called `tanimoto`.
8. **[PORT + weight] Cosine.** Weighted by `X·diag(√w)`; scipy has no compiled cosine kernel, so `pdist(X·√w, 'cosine')` is 47× faster than `pdist(X, 'cosine', w=w)` and bit-identical (4.4e-16).
9. **[PORT + weight] Soergel.** Weighted by `X·diag(w)` — the weighted form is *identically* the unweighted form on `w`-rescaled inputs, so no new function is needed. Note in the docstring that on binary it is exactly Tanimoto distance and on counts it is exactly RDKit's count Tanimoto distance.
10. **[PORT + weight] Euclidean.** Weighted by `X·diag(√w)`. Prefer the direct `‖x−y‖` form over the Gram form `√(a+b−2c)` for continuous blocks — the Gram form loses 2.4e-5 to 1.4e-4 relative precision on near-duplicate pairs at 1000+ dimensions and clamps the resulting negative to zero, corrupting exactly the nearest-neighbour and activity-cliff pairs the scorers weight most.
11. **[PORT + weight] Tversky.** Fix the count path first: `simeval._bit_counts` uses row sums for `a` and `b` but the Gram diagonal for `c`, which on counts returns distances up to 174. Use `Σmin` for the count intersection. Ship it **unsymmetrised** with `(α, β)` as a real swept axis — the current post-hoc `0.5(D + Dᵀ)` makes it depend only on `αβ`, cannot distinguish `(0.7,0.3)` from `(0.3,0.7)`, and lands at Spearman 0.9996 with Dice.

### Tier 2 — new coefficients that add independent geometry

Ordered by how much they add per unit of implementation. Each is one function.

12. **[NEW] Simpson / Asymmetric** `a/min(a+b, a+c)` and **[NEW] Braun-Blanquet** `a/max(a+b, a+c)`. Together these are the two containment extremes; keep one, not both (`Simpson × Braun-Blanquet = cos²`). Weighted by `X·diag(w)`. Warning: RDKit's `AsymmetricSimilarity` is symmetric despite the name; for a genuinely directional coefficient use Tversky with `α ≠ β`.
13. **[NEW] Kulczynski-2** `½(a/(a+b) + a/(a+c))`. One of the few binary coefficients that is not in the Tanimoto, Hamming, cosine or φ class. Ship it; do **not** additionally ship McConnaughey, Johnson or Driver-Kroeber, all of which are affine transforms of it.
14. **[NEW] Pearson–Heron-1 / φ** `(ad−bc)/√((a+b)(a+c)(b+d)(c+d))`. The only negative-match-inclusive coefficient that is not in the Hamming class. Equals `1 − correlation distance` on binary, so it is also the binary arm of a coefficient already in the continuous set — currently an unfilled cell.
15. **[NEW] Bray–Curtis on counts** `Σ|x−y|/Σ(x+y)`, weighted by `X·diag(w)`. **Only if Soergel is dropped** — the two are rank-identical (`BC = S/(2−S)`, Spearman exactly 1.000) on all non-negative data, so running both is a duplicate cell. Bray-Curtis is a semimetric; Soergel is a true metric. Prefer Soergel.
16. **[NEW] Canberra on counts** `Σ_k |x_k−y_k|/(|x_k|+|y_k|)` with `w=` passed inside. Genuinely distinct on counts (Spearman 0.964 with Manhattan). Note it degenerates to Hamming on binary and is ~99.6% correlated with binarised Hamming on *sparse* counts, so it only earns its place on dense count blocks.
17. **[NEW] Jensen–Shannon distance** on L1-normalised counts, weighted as `√(½ Σ_k w_k [...])`. Requires strict non-negativity and an all-zero-row guard (returns `nan`). Use base 2 if a [0,1] range is wanted; scipy's default natural log caps at `√(ln 2) = 0.8326`.
18. **[NEW] Fractional Minkowski**, `p ∈ {0.5, 0.3}`, on continuous blocks only, weighted by `X·diag(w^(1/p))` or by passing `w=`. Two caveats to record: `p=0.5` and `p=0.3` correlate at Spearman 0.990 with each other, so treat any split between them as noise; and the `(1/p)` outer root is a strictly increasing transform, so it is **invisible to any Spearman-scored kernel** — the relative-contrast argument that motivates fractional norms does not transfer to a rank-based objective. If a metric is needed, use the power-sum form `Σ|Δ|^p` without the root: it is rank-identical and *is* a true metric for `p < 1`.
19. **[NEW] Weighted Mahalanobis with fixed precision.** `√((x−y)ᵀ W^½ Σ⁻¹ W^½ (x−y))` with `Σ⁻¹` estimated once on unweighted train data and passed in. This is the only way any Mahalanobis weight arm is not an exact no-op.

### Tier 3 — do not build

The following are **duplicate cells** on every representation family they would run on, and adding them produces a results table where two rows agree perfectly and the agreement reads as corroboration:

- **Dice** alongside Tanimoto (`D = 2T/(1+T)`, Spearman exactly 1.000 on binary, count and continuous). This is a live duplicate in `simeval.py` today — its docstring's claim that "the (n,n) geometry differs, so it is a distinct metric here" is false, and it is the same argument already used to *exclude* Soergel from `BIT_COEFFS`.
- **Manhattan** alongside Euclidean on **binary** (`L1 = L2²` exactly, Spearman 1.000). Currently both are registered in `f18`'s `BINARY_COEFFS`.
- **Any Minkowski `p`** on binary (`L_p = Hamming^(1/p)` exactly for every `p`).
- **Chebyshev** on binary (collapses to the 0/1 discrete metric — one distinct value across all non-identical pairs).
- **Canberra** on binary (`= Manhattan` exactly).
- **Sokal–Michener, Rogers–Tanimoto, Hamann, Gower–Legendre, Sokal–Sneath-2** — pick at most one; all five are strictly monotone in `b+c`.
- **McConnaughey, Johnson, Driver-Kroeber** if Kulczynski-2 is shipped.
- **Sokal–Sneath-1, 3w-Jaccard, Kulczynski-1, OnBit** if Tanimoto is shipped.
- **Sorgenfrei, binary Hellinger** if cosine is shipped.
- **BUB-2** if BUB-1 is shipped (`BUB2 = 2·BUB1 − 1`).
- **Ochiai-2** — identical to Sokal–Sneath-5, and within 0.006 of plain cosine on sparse folded fingerprints.
- **seuclidean** alongside Euclidean on standardised features (identical up to `√(n/(n−1))`, Spearman exactly 1.000). It is only non-redundant on **raw** unstandardised features with `V` fit on train and passed explicitly.
- **Correlation** alongside cosine on wide standardised blocks (Spearman 0.9995 at d=1024). Correlation is cosine with row-centring prepended; report it as a preprocessing arm, not a coefficient arm.
- **Russell–Rao** — a rescaled shared-bit count with a non-constant, non-unit diagonal (`S(x,x) = |x|/n`, measured self-distance 0.998 for aspirin at 2048 bits). Its `1−S` is not a distance.
- **Kulczynski-1** — unbounded, `+inf` on the identical pair, and rank-identical to Tanimoto (`K1 = T/(1−T)`).
- **Forbes-1** — non-constant diagonal (`S(x,x) = n/|x|`, measured 17.2 to 136.5 across a 60-molecule set, 3–21× the largest off-diagonal). Unusable as a kernel or a distance. If the enrichment signal is wanted, use `log(Forbes-1)` = pointwise mutual information and label it as an association score.
- **Faith** — non-constant self-similarity `0.5 + |x|/(2n)` confounded with molecular size, so `1−S` violates identity of indiscernibles; weakly *anti*-correlated with Tanimoto on sparse fingerprints (Spearman −0.069); near-redundant with Sokal–Michener (0.998).
- **Rogot–Goldberg** — dynamic range on 2048-bit ECFP is `[0.487, 1.0]` with 23.4% of pairs above 0.9, and the spread is a pure function of `fpSize`.
- **Yule Q** — 23.4% of ECFP4 pairs above 0.9 at 2048 bits versus 4.6% at 1024; the value is largely a readout of the fold length. `+1` pinning is rare (0.013% of pairs) and `−1` pinning is four times more common.
- **Michael, Mountford, Derived-Jaccard, Forbes-2, BUB-1/2, Hamming-class members** — all either non-reflexive, unbounded, `fpSize`-dependent, or squashed into a sub-0.05 dynamic range on realistic fingerprints. None has a documented advantage that survives the metric-sweep design.

### Tier 4 — documentation fixes required before any results are reported

- `simeval.py:126–128` and `f18:72–75` both cite a **selftest that does not exist** in the file for the Soergel-equals-Tanimoto identity. The identity is true (verified to 1.1e-16); the proof is not in the repo.
- `simeval.dist_dice:60–62` docstring is false (Dice is globally rank-identical to Tanimoto, not merely per-query).
- `simeval.dist_cosine:74–75` cites Miranda-Quintana et al. 2021 for "consistent with Tanimoto". That paper finds the Tanimoto–cosine consensus is **conditional** on the representation not depending strongly on molecule size — the opposite of Morgan fingerprints.
- `f18:88–90` states scipy's `jaccard` "coerces at zero, which on standardised features means above or below the mean". scipy coerces at **nonzero** (documented change in 1.15.0), so on standardised features every row becomes all-True and the returned matrix is **exactly all zeros** — every pair identical, not merely mis-signed.
- `f18:201–203` says constant columns "break correlation-based coefficients with a zero-variance denominator". Correlation's denominator is a **row** variance; dropping a constant column cannot prevent that failure and can *create* it.
- `SIMILARITY_HYBRID_DESIGN_2026_08_24.md:230` lists Bray-Curtis and Mahalanobis under "standardised stacked blocks". Bray-Curtis is undefined on signed input, and `f18`'s own `needs_nonneg=True` flag already contradicts the doc.
- `SIMILARITY_HYBRID_DESIGN_2026_08_24.md` Axis B divides each block by `√width`. That equalises L2 and leaves L1-family coefficients growing as `√width` (measured 3.64 / 11.28 / 22.54 at m = 10 / 100 / 400 *after* the correction). The "equal-block" level currently means two different things depending on which coefficient occupies the cell.
- `PERTURBATION_SIMILARITY_SUITE_DESIGN.md:78` instructs "z-score each column … then compute Soergel". Z-scored features are not count vectors and Soergel is undefined on them.
- Bajusz, Rácz & Héberger (2015) is cited in several places as the source ranking "best-tier" coefficients. That paper tested **binary fingerprints only**, defines Soergel as `1 − Tanimoto`, and its conclusion that Cosine/Dice/Tanimoto/Soergel are "the best (equivalent) metrics" is a statement that they are *the same coefficient*. It carries no evidence about count vectors and does not evaluate Russell–Rao, Kulczynski, Forbes, Sokal–Sneath or any other coefficient in Section 1a beyond those four plus Manhattan and Euclidean.