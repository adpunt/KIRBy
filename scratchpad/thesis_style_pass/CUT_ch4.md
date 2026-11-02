# Chapter 4 (chap:flaws) cutting pass, 2026-10-10

Prose words are counted from the chapter heading to the next \chapter. Lines starting with % are left out. Counted with python3 -I.

- Backup (thesis.tex.before_style_apply_2026-10-10): 7,599 words
- Before this cut: 7,845 words
- After this cut: 7,487 words (358 cut; 1.5% under the backup)
- 5% target (7,219 words): missed by 268 words.

Why the target was missed: every cut came from paragraphs with a bare "% TODO: manual review" or "% TODO: review, needs manual review" line directly above them. After this pass those paragraphs hold about 1,100 words in total. Most of what is left is method detail, a limitation, or a definition that a later chapter uses. Examples: the Minkowski p=0.5 formula, which Chapter 6 points back to, and the sweep's 221 configurations. The CODE INPUT paragraphs with no review marker were not touched. These are the tied-group RMSE floor, SALI, the 70-combination result, MCS, the neighbor-cleaning results and most of the Conclusions. Neither were her paragraphs marked "edited your text".

## What was cut, one line per paragraph

- Methods, distances in later chapters: merged two sentences about where Chapter 6 uses each distance. The Minkowski formula and the triangle-inequality sentence are kept.
- Ties, counting: put the definition of a tie first and folded the augmentation-feature definition into a clause. Dropped "so every tie merges two different molecules", which the sentences before it already say.
- Ties, which representations: cut two sentences that restate Table tab:flaws:ties. One was the stereo-only split by representation. The other was the WL 49/26 non-stereoisomer ties, which her paragraph above and the table already give. Also cut "MACCS ties the most", which the table shows. Merged the last two sentences.
- Four measurements, intro: cut the list of the four measurements, because each has its own \paragraph heading below. Cut "a third Biogen dataset used only in this chapter"; HLM clearance is introduced in Section sec:flaws:constant.
- Activity cliffs: shortened the MoleculeACE sentence. Cut the hERG gap and count (1.76 pKi, 211 molecules), the QM9 gap in eV and the Biogen gaps in log units, all of which are in Table tab:flaws:distance_summary. Kept the 1,093 similar QM9 pairs and the ratios to the median absolute deviation, which are not in the table.
- Distance and label difference: cut "We asked ... in two ways", "The first uses" and "The second uses". "scaffold groups" became "scaffold families".
- hERG pair figure paragraph: cut the 1.1% at 0.55, the median differences (0.81 and 0.20 to 0.30 pKi) and the Mordred medians by 5% bin, all of which the figure shows. Kept 93% below 0.20 and the 8.3% of 774 pairs.
- Agreement on neighbors: folded the PDV and HLM substitutions into one parenthesis. Cut the hERG range 0.47 to 0.52 down to "at most 0.52"; the full values are in Table tab:flaws:neighbor_overlap. "scaffold groups" became "scaffold families".
- Sweep intro: cut "Chapter noise expands this sweep", which the limitations paragraph of the same section already says.
- Sweep set-up: shortened the sentence about four removal methods and eight training methods to one sentence with both labels.
- Retention: merged the retention-ratio and aucnorm sentences. Cut "Methods that used a distance did worse than those that did not", which the analysis-of-variance sentence and the 20-configuration sentence after it already say.
- Sweep limitations: merged the two sentences that point to Chapter noise into one.
- Early hybrid in neighbor cleaning: cut the closing pointer to Chapter noise. It is the third copy; the sweep limitations paragraph and the Conclusions keep it.
- Conclusions, noise paragraph: merged "did not always fail" into the sentence before it. Cut "Gaussian", which the limits paragraph states. Shortened the pointer to Chapter noise.
- Conclusions, closing paragraph: cut "Chapter intro described how these flaws led to..." and merged the two sentences about Chapter method into one.

No number was changed or added. No \label, table body, \includegraphics or caption was touched. In this chapter's AI-marked prose, "binding free energy" appears only in comments, so no ΔΔG change was needed.
