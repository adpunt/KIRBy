# Citations left for you (10 October 2026)

## 1. What you need to look up (6 papers, 9 sentences)

These papers are paywalled, and nothing open said whether they back the sentence. Open each link through the Oxford library and search the PDF for the words given. If the words are there, the citation is fine. If not, do what "If not there" says. Line numbers are thesis.tex as of tonight.

**Zhang 1999, Z-factor** — https://doi.org/10.1177/108705719900400206
- Line 330 (noise table): "higher relative error near a detection limit". Look for: *detection limit*. If not there: delete this citation from the cell (Simeonov2008 stays).
- Line 330 (noise table): "background subtraction in radiometric assays". Look for: *radiometric*, *background*. If not there: delete this phrase from the cell.
- Line 1219: "the Z′-factor, computed from an assay's controls, can estimate noise". Almost certainly there; Z′ is defined in this paper.

**Brideau 2003** — https://doi.org/10.1177/1087057103258285
- Line 329 (noise table): "reagents or compounds degrading during a run". Look for: *degrad*, *drift*, *time*. If not there: delete the phrase; Makarenkov2007 in the same cell covers plate effects.

**Todeschini 2009, Handbook of Molecular Descriptors** — https://doi.org/10.1002/9783527628766
- Line 184: "PDV spaces suffer from multicollinearity". Look for: *collinear*, *correlated descriptors*. If not there: replace with Cherkasov2014 (it says collinear descriptors with r² > 0.8 should be scrutinized).
- Line 182: "constitutional, topological, physicochemical and electronic descriptors". Almost certainly fine; the handbook is organised by these classes.

**Jaeger 2018, Mol2vec** — https://doi.org/10.1021/acs.jcim.7b00616
- Line 176: "matching or exceeding Morgan fingerprints on solubility and bioactivity". Look in the results tables comparing with *Morgan*. If not there: change "matching or exceeding" to "comparable to".

**Tobin 1958** — https://doi.org/10.2307/1907382
- Line 4145: "replaces each clipped label by the mean of a normal cut at the limit". Tobin introduced the censored-normal model; filling in the mean is a later step. If not there: change `\citep{Tobin1958}` to "following the censored-normal model of `\citet{Tobin1958}`".

**Svetnik 2003, random forest for QSAR** — https://doi.org/10.1021/ci034160g
- Line 199: "trees make no assumptions about the data distribution". Look for: *assumption*, *distribution*. If not there: move the citation to Breiman1984, cited earlier in the same paragraph. Low risk either way.
- Line 704: "SVMs are a well-established QSAR baseline". Look for *SVM* / *support vector* in their comparison tables. Most likely there. If not: keep Vapnik1995 alone.

## 2. The other could-not-check citations: left as they are

The reference theses cite the paper that introduced a method or term, even for textbook facts. By that standard these are fine:
- Breiman1984 (six places): the CART book, for basic decision-tree facts.
- Cortes1995 and Vapnik1995, line 232: origin papers for SVMs and the C constant; Smola2004 in the same sentence carries the wording.
- Bellman1961, line 1826: coined "curse of dimensionality"; Hughes1968 next to it carries the claim.
- 23 origin citations where an open second source confirms the attribution (SMILES, atom pairs, torsions, similar property principle, Mahalanobis, Harrell's c-index, Shepard, Watson, Hedges and Olkin, and others).

One fix made: the 0.55 ECFP4 cut-off for activity cliffs (line 4131) now cites Hu2015, where the number comes from.

## 3. Partly supported: what was a real problem

Test used: does the sentence say something the source contradicts, or put a wrong number or the wrong paper on a claim? If yes, it was fixed. If it is only a bit broader than the source, or the source is the origin of a method you adapted, it was left.

### Fixed tonight (TODO note above each edit to your own text)
- Line 12, rogers2010: removed "predefined" (Rogers and Hahn say ECFP features "are not predefined").
- Line 169, Fassio2022: "they" (all bit-vector fingerprints) → "hashed fingerprints" suffer bit collisions.
- Line 178, Fabian2020: MolBERT beat "count-based ECFP4 (ECFC4)", which is what the paper used.
- Line 190, David2020: graphs leave out coordinates and angles "unless added as node or edge attributes".
- Lines 404–405, Avdeef2019: "for one solubility method", inter-lab error is about 3× intra-lab.
- Line 1305, Golbraikh2014: MODI is "a class-weighted share", not "the proportion".
- Line 1308, Praski2025: "nearly all" of 25 embeddings gave negligible improvement (CLAMP was better).
- Line 2694, Bengio2004: no unbiased estimator "that is valid under all distributions".
- Line 2811, Lundberg2020: added Lundberg and Lee 2017 for Kernel SHAP and DeepLIFT SHAP.
- Line 4038, Chawla2006: SLOM works "on spatial data, where neighbors are found by location" (old wording was backwards).

Earlier fixes: Kramer2012, landrum2024, SMAC (Arsiwala2025), Chungyoun2024, Breiman2001 → Liaw2002, Cherkasov2014 → Guyon2003 + Jolliffe2016.

### Your call (4)
- Lines 169–170, Fassio2022: a TODO already says this paragraph is close to verbatim from Fassio et al. Plagiarism risk; needs rewording in your words.
- Line 1917, Wang2021 (MOGONET): the table row says adding a representation needs "one more model", but MOGONET trains all networks together. Move it to another row or drop it.
- Line 2540, Chuang2018: thesis says one-hot reagent labels predict yield about as well as descriptors; the abstract says "random-valued features". Search the paper for *one-hot*: https://doi.org/10.1126/science.aat8603
- Line 195, Hönig2022: cited for other kinds of molecular graph, but it reviews 3D superposition methods. I would delete it from that list.

### Also from tonight
- Line 1829, Lanini2023PREFER: "usually decided by trial-and-error" is not in the abstract (full text blocked). Safe wording: "is hard to make, because representations are rarely compared on equal terms".

### Left as they are (everything else)
- Your own settings next to a method's citation (300 dimensions for Mol2vec, 384 for ChemBERTa-2, radius 2 / 2048 bits for ECFP4, five scaffold folds).
- "After X" / "following X" adaptations (Brodley1999, Wilson1972, Kersting2007, MentorNet).
- General statements backed by a list of citations (line 15, line 1894, line 211).
- Slightly broader wording than the source ("among the most common QSAR models", "aleatoric" for NGBoost, label smoothing among noise-robust methods).
