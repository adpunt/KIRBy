# Scaffold-diff similarity: methods and results

**Owner:** Adelaide · **Branch:** `similarity-metrics-study` · **Last run:** 2026-07-19
**Experiment:** `tests/experiments/opig_scaffold_similarity.py`
**Results:** `tests/experiments/results/opig_scaffold_similarity.json`
**Roadmap context:** `SIMILARITY_METRICS_ROADMAP.md`

Reproduce everything in this document with:

```bash
python tests/experiments/opig_scaffold_similarity.py --mode all
```

---

## 1. The problem

Many important prediction problems are **small perturbations of a reference**: a
wildtype antibody carrying one point mutation, a protein mutation with a measured
ΔΔG, a matched molecular pair. Whole-object representations degrade here, because
every variant is ~99% identical to every other, so the encoding is nearly
constant and the distances it induces are nearly meaningless.

The proposal is to represent each variant by **what changed relative to the
reference** — a scaffold-diff representation — and to study those representations
**as similarity spaces**: do the pairwise distances they induce track true
structural difference?

## 2. Metric

**Every number in this document is a Spearman rank correlation in [−1, 1], higher
is better.** It is computed over sampled *pairs* of variants: correlate "how far
apart the representation says this pair is" against "how far apart an independent
substitution matrix says they are."

**No model is trained anywhere in this experiment.** These are not R², not
accuracy, not predictive performance. A score of 0 means the representation orders
pairs no better than chance relative to the ground truth.

## 3. Ground truths and the three honesty controls

Three independent structural ground truths are available: **Grantham**
(physicochemical: composition, polarity, volume), **BLOSUM** (evolutionary
substitution frequency), and **identity/Hamming** (how many positions differ).
Each is applied as a grader-weighted sequence distance.

The central methodological claim of this work is that grading a perturbation
representation is easy to get wrong in three distinct ways, each of which
silently inflates scores. All three are tested explicitly by `--mode controls`.

**C1 — Basis independence.** Never grade a representation with a ground truth
built on its own basis. A physicochemical representation is graded by BLOSUM,
never by Grantham. Enforced per representation; circular columns are reported but
excluded from any headline.

**C2 — Count control.** The Grantham and BLOSUM ground truths *sum over
positions*. On a library where variants differ in how many positions changed,
those sums are largely just counting differences. Any representation that recovers
that count then scores highly with no chemistry whatsoever. **Every score in this
document has the count component partialled out** (partial Spearman against the
Hamming ground truth). Measured contamination, i.e. the correlation of plain
Hamming distance with the BLOSUM ground truth:

| dataset | mutations per variant | contamination | status |
|---|---|---|---|
| flab_expression_koenig | 1.00 | 0.118 | clean |
| flab_binding_koenig | 1.00 | 0.118 | clean |
| graphinity_foldx | 1.00 | 0.302 | clean |
| flab_expression_adams | 1.75 | 0.796 | contaminated |
| flab_binding_adams | 1.76 | 0.808 | contaminated |
| flab_binding_phillips_h3 | 7.43 | 0.924 | contaminated |

On the three contaminated sets, uncontrolled scores are not interpretable. For
scale: `BaseDelta` — a representation that reduces to plain Hamming distance —
scores 0.796 and 0.924 raw on adams and phillips, exactly the contamination
figures, and **0.000** once controlled.

**C3 — A metric must not be a ground truth.** A proposed bespoke distance must be
checked for algebraic identity against every grader. This control caught a real
error in our own earlier work: `position_gated`, presented as the study's best
result at 0.519, is **bit-for-bit identical to the Grantham ground truth** on a
single-mutation library (Spearman 1.000000, maximum absolute difference 0.0). Its
apparent score was Grantham-versus-BLOSUM grader agreement — a ceiling any method
reaches by copying an answer key. **The 0.519 is retracted and must not be cited.**

C1 is the control we started with; C2 and C3 are new, and C3 is the one that
overturned a headline result.

## 4. Datasets

Six scaffold-compatible datasets across two domains. Multi-scaffold sets compare
variants only *within* a reference and pool the pairs, because a cross-reference
pair is graded on wholesale sequence difference rather than on the mutation.

| dataset | domain | n | references | mutations/variant |
|---|---|---|---|---|
| flab_expression_koenig | antibody DMS | 4,275 | 1 | 1.00 |
| flab_binding_koenig | antibody DMS | 4,275 | 1 | 1.00 |
| flab_expression_adams | antibody DMS | 5,000 | 1 | 1.75 |
| flab_binding_adams | antibody DMS | 5,000 | 1 | 1.76 |
| flab_binding_phillips_h3 | antibody DMS | 5,000 | 1 | 7.43 |
| graphinity_foldx | protein ΔΔG | 58,564 | 1,388 | 1.00 |

**Excluded, with reasons recorded in code** (`EXCLUDED_DATASETS`):
`flab_thermo_hie` (n=10), `flab_binding_rosace` (n=14), `ddg_combined`
(multi-point across multiple PDBs; needs a generalisation not yet built).

**Note on the koenig pair:** `flab_expression_koenig` and `flab_binding_koenig`
have *identical, identically-ordered* sequences — the same variant library under
two assays. Their structural columns are therefore identical by construction and
count as **one** structural replicate, not two. They differ only on label-dependent
axes.

## 5. Result 1 — the winning representation is unanimous

`dPhysChem_pos` (physicochemical property change, tagged by the position it
occurred at) with **euclidean** distance is the best clean, count-controlled
representation on **all six datasets**, spanning both domains.

| dataset | mutations/variant | best representation × metric | score |
|---|---|---|---|
| flab_expression_koenig | 1.00 | dPhysChem_pos × euclidean | 0.387 |
| flab_binding_koenig | 1.00 | dPhysChem_pos × euclidean | 0.387 |
| flab_expression_adams | 1.75 | dPhysChem_pos × euclidean | 0.527 |
| flab_binding_adams | 1.76 | dPhysChem_pos × euclidean | 0.507 |
| flab_binding_phillips_h3 | 7.43 | dPhysChem_pos × euclidean | 0.645 |
| graphinity_foldx | 1.00 | dPhysChem_pos × euclidean | 0.627 |

Two secondary findings hold throughout: **euclidean ≫ cosine** for these sparse
delta representations (cosine collapses toward a constant), and **position tagging
beats position-agnostic summing** — keeping each change in its own coordinate
rather than pooling them is what carries the advantage, and the margin grows with
the number of changes per variant.

Representations that fail: `MutID` (one-hot of position × mutant residue) is
orthogonal by construction and scores ~0; `BaseDelta` is Hamming distance in
disguise and scores 0 under count control; `SubScore` is built from both rich
ground truths and so has **no** independent grader left — an honest limitation, not
a zero.

## 6. Result 2 — what makes a feature improve a similarity space

This is the generalisable finding, and it is not specific to scaffold-diff.

A pool of **14 amino-acid property scales** was assembled, deliberately containing
redundancy (charge/isoelectric-point/polarity overlap; molecular-weight/volume/
surface-area overlap). Three selection strategies were compared: use everything,
take the top-k by individual score, and greedy forward selection on the objective.

| dataset | all 14 | top-k by solo score | greedy | raw units (not z-scored) | greedy selection |
|---|---|---|---|---|---|
| flab_expression_koenig | 0.516 | 0.590 | **0.619** | 0.356 | aromaticity, polarity, volume, turn, polar-requirement |
| flab_binding_koenig | 0.516 | 0.590 | **0.619** | 0.356 | aromaticity, polarity, volume, turn, polar-requirement |
| flab_expression_adams | 0.546 | 0.597 | **0.685** | 0.471 | aromaticity, volume, polarity |
| flab_binding_adams | 0.536 | 0.576 | **0.670** | 0.457 | volume, aromaticity, polarity |
| flab_binding_phillips_h3 | 0.762 | 0.757 | 0.757 | 0.564 | polarity, surface-area |
| graphinity_foldx | 0.627 | 0.637 | **0.765** | 0.387 | polarity, volume, hydropathy |

**Finding 2a — more features is not better.** Greedy selection using 2–5 features
beats using all 14 on five of six datasets, by up to +0.138 (graphinity). Only
phillips, the most contaminated set, prefers everything.

**Finding 2b — individual signal does not qualify a feature; non-redundancy does.**
Greedy beats top-k-by-individual-score on five of six datasets even at equal
feature count. The rejected features make the mechanism concrete — these have
*strong* individual scores and are still rejected, because they duplicate an axis
already chosen:

| dataset | rejected feature | its solo score | correlation with a chosen feature | effect of adding it |
|---|---|---|---|---|
| flab_binding_phillips_h3 | molecular weight | 0.535 | 0.98 | **−0.057** |
| flab_expression_adams | molecular weight | 0.496 | 0.93 | −0.017 |
| flab_binding_adams | molecular weight | 0.478 | 0.93 | −0.019 |
| flab_expression_adams | surface area | 0.383 | 0.91 | −0.039 |
| graphinity_foldx | polar requirement | 0.401 | 0.88 | −0.054 |
| flab_expression_koenig | molecular weight | 0.331 | 0.89 | −0.001 |

Molecular weight is among the strongest features in isolation on every dataset and
is rejected on every dataset, because volume or surface area is already in and
correlates with it at 0.89–0.98.

**Why this differs from predictive modelling.** A euclidean distance has no
weights. Every feature contributes its squared difference to every pair, so two
correlated features double-count the axis they share and a weak feature adds
spread without adding signal. A downstream learner can down-weight or ignore
redundant inputs; a distance cannot. This is a distance-specific phenomenon and it
directly parallels the project's earlier negative result that complementarity-aware
selectors help similarity but not prediction.

**Finding 2c — scaling is not optional.** Z-scoring the property scales is worth
**+0.16 to +0.38** on every dataset (e.g. graphinity 0.387 → 0.627). In raw units,
molecular weight (range ~130) and surface area (range ~180) numerically swamp
charge (range 2) and aromaticity (range 1), so the distance is effectively a
size-only metric.

**The general rule.** In a distance-based similarity space, a feature earns its
place by contributing an axis *not already present*. Select on the objective, not
on individual feature scores; scale first; and expect a small selected set to beat
the full pool.

## 7. Result 3 — where the signal lives

Drop-one-position analysis on `dPhysChem_pos` (`similarity_metrics/scaffold_feature_isolation.py`):

| dataset | mutated positions | share of signal in top 5 positions |
|---|---|---|
| flab_expression_koenig | 225 | 20.3% |
| flab_expression_adams | 20 | 90.1% |
| flab_binding_phillips_h3 | 15 | 101.5% |

koenig spreads its signal thinly across 225 sites; adams and phillips concentrate
it in a handful. The figure above 100% on phillips is real: some positions have
*negative* contribution, so removing them improves the space. A position-selected
variant of the representation is an obvious next step and has not been built.

## 8. Limitations, stated plainly

1. **Cross-representation comparisons are not same-yardstick.** C1 forces each
   representation to be graded by whatever ground truth is independent of it, so
   `dPhysChem_pos` (graded by BLOSUM) and `blosum_mean` (graded by Grantham) are
   not scored against the same answer key. This is inherent to the anti-circularity
   design, not a fixable oversight.
2. **One structural replicate per library.** The koenig pair shares sequences, so
   the six datasets provide four independent structural libraries, not six.
3. **Activity is not addressed here.** graphinity ΔΔG values are FoldX-computed,
   not measured; correlating against them would characterise a simulator. Earlier
   work established that structural fidelity and activity are decoupled — the best
   structural representations carry ~0 fitness signal — and this experiment reports
   the structural axis only.
4. **The molecular domain is untested.** MoleculeACE matched pairs need a molecular
   scaffold-diff representation that does not yet exist.
5. **Three of six datasets are count-contaminated**, and on those only the
   controlled column means anything.

## 9. Status of claims

| claim | status |
|---|---|
| dPhysChem_pos × euclidean is the best scaffold-diff representation | supported, 6/6 datasets, 2 domains |
| Position tagging beats position-agnostic summing | supported, all datasets |
| euclidean ≫ cosine for sparse delta representations | supported, all datasets |
| Non-redundancy, not individual signal, decides feature value | supported, 5/6 datasets |
| Z-scoring is required | supported, 6/6 datasets |
| `position_gated` achieves 0.519 structural fidelity | **RETRACTED** — it is the Grantham ground truth |
| `BaseDelta` works on multi-mutation libraries | **RETRACTED** — pure count artifact, 0.000 controlled |
| Scaffold-diff beats whole-object representations | **not yet established** — needs a same-basis baseline on protein data |
