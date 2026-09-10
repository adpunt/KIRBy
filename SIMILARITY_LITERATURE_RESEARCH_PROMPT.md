# Research prompt — comprehensive survey of molecular similarity metrics and how they are benchmarked

Paste everything below into a fresh **claude.ai** chat (enable Research / extended thinking).

---

You are surveying the literature on **molecular similarity** for a computational
chemistry / ML researcher. I measure similarity between two small molecules and I grade
a similarity metric two ways: **(1) activity coherence** — do molecules the metric calls
similar have similar *measured* activity (pKi / pIC50 / logD)? — and **(2) structural
fidelity** — does the metric's distance track an independent structural ground truth,
maximum common substructure (MCS)? I need a thorough, SPECIFIC, cited map of the field so
I can (a) choose which similarity metrics to use and (b) benchmark them credibly, and know
where the real gaps are. Vague summaries are useless to me — I need specifics (exact
datasets, exact representations, exact metrics, exact findings).

Cover all four parts.

## Part A — the full menu of similarity metrics
A similarity metric = a molecular **representation** + a **comparison function**. Survey
every major family. For each: what it is, representative studies (2015–2026), reported
performance, strengths/weaknesses, and when it is the right choice.
1. **Classical fingerprints** — ECFP/Morgan, FCFP, MACCS (166/167), PubChem, atom-pair,
   topological torsion, RDKit/Daylight path-based, Avalon, MAP4, MHFP/MHFP6 — each ×
   **comparison coefficients**: Tanimoto/Jaccard, Dice, Cosine, Soergel, Tversky,
   Euclidean, Manhattan, and the less common ones (Russell-Rao, Kulczynski, etc.).
   Binary vs count fingerprints — does counts vs bits matter?
2. **Descriptor sets** — RDKit descriptors, Mordred, PaDEL — + distance functions.
3. **3D / shape / pharmacophore** — USR, USRCAT, ElectroShape, ROCS (Gaussian shape
   overlap), 2D/3D pharmacophore fingerprints, interaction fingerprints (SPLIF/PLEC).
4. **Learned / embedding-based** — message-passing GNNs (D-MPNN/Chemprop, GIN, GCN, GAT,
   AttentiveFP), molecular language models (ChemBERTa, MolFormer, SELFormer), **deep
   metric learning / triplet networks** ("molecules as triplets"), contrastive methods
   (MolCLR, GraphMVP), CLAMP, and molecular foundation models.
5. **Extended n-ary / set similarity** — Miranda-Quintana iSIM, iCliff, BitBIRCH.
Also: which studies COMPARE these families head-to-head, and what they concluded — in
particular, do learned representations reliably beat ECFP for similarity, or not?

## Part B — how similarity metrics are benchmarked / evaluated
Survey every evaluation methodology. For each: what it measures, representative studies,
and known pitfalls.
1. **Intrinsic "similarity principle" tests** — neighborhood behavior (Patterson et al.
   1996), MODI / RMODI modelability (Golbraikh/Tropsha), structure–activity landscape
   roughness, SALI, SAS maps, activity cliffs.
2. **Retrospective virtual-screening enrichment** — enrichment factor, BEDROC, ROC-AUC;
   benchmark sets (DUD-E, MUV, LIT-PCBA) and their documented analogue/decoy biases.
3. **Downstream / read-across** — kNN-QSAR, scaffold-split vs random-split generalization.
4. **Ground-truth references + the CIRCULARITY problem** — grading a metric against
   *measured activity*, vs against an *independent structural reference* (MCS, graph-edit
   distance), vs against *another fingerprint* (circular). Who has explicitly raised the
   circularity issue and proposed independent structural ground truths?
5. **Statistical rigor** — paired Wilcoxon, sum-of-ranking-differences (SRD), ANOVA,
   Bayesian Bradley–Terry, Demšar-style comparisons across many targets.
6. **Noise / label-quality evaluation** — does a similarity metric help detect or clean
   noisy activity labels (edited-nearest-neighbour / confident learning)?

## Part C — deep-dive on the landmark benchmark studies (BE SPECIFIC — tables, not prose)
For each, extract exact specifics. I want a table: study → datasets/targets (list them or
count + examples) → every representation/model compared → evaluation metric → concrete finding.
- **MoleculeACE (van Tilborg, Alenicheva & Grisoni, JCIM 2022, 62:5938–5951):** exactly
  which ~30 ChEMBL protein targets (IDs / assay types), molecule counts, EVERY
  representation and model compared (list all — fingerprints, descriptors, SMILES-based,
  graph-based), the exact metric (RMSE on cliff vs non-cliff compounds?), and the finding.
- **Riniker & Landrum (J Cheminform 2013):** which 12 fingerprints, which benchmarks
  (DUD/MUV/ChEMBL), the conclusion about fingerprint interchangeability.
- **Bajusz, Rácz & Héberger (J Cheminform 2015) & Todeschini et al. (JCIM 2012):** which
  coefficients ranked best, on what data, by what method (SRD/ANOVA).
- **Praski, Adamczyk & Czech (arXiv:2508.06199, 2025):** the 25 embedding models, the 25
  datasets, the Bayesian Bradley–Terry test, the finding vs ECFP.
- Any newer (2023–2026) large-scale similarity/representation benchmarks (e.g. molecular
  representation benchmarks, activity-cliff follow-ups, geometry-of-cliffs work).

## Part D — gaps and openings
Where is the field unsettled or unexplored? Specifically:
- Has anyone connected a BETTER similarity metric to **measurable label-noise cleaning**,
  using ChEMBL's known experimental noise (~0.5 pKi; Kramer 2012, Kalliokoski 2013)? Who
  has closed the loop "better metric → measurably cleaner labels", and who hasn't?
- Has anyone graded similarity metrics against an **independent structural ground truth**
  (MCS / graph-edit distance) rather than activity or another fingerprint?
- Is the structure-vs-activity split (activity cliffs) actively **exploited for metric
  design**, or only diagnosed?

## Requirements
- Prioritize 2015–2026; top venues (JCIM, J Cheminform, Nature Machine Intelligence,
  Bioinformatics, NeurIPS/ICML ML-for-molecules workshops).
- Cite specific papers (authors, venue, year, ideally DOI). Distinguish **established** vs
  **contested** vs **open**. Flag single non-peer-reviewed preprints as such.
- Final deliverable: a structured, cited survey I can act on — a menu of similarity metrics
  with evidence, a menu of benchmarking methods with pitfalls, the specific-study tables
  from Part C, and a clear statement of the open gaps (Part D).
