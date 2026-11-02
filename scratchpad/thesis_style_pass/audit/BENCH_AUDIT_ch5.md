# Benchmark audit, Chapter 5 (Building adaptive hybrid representations)

Written 2026-10-10. Read-only audit: nothing in thesis.tex, main.tex or refs.bib was changed.

## What was measured, and on what

- Text is a snapshot of `thesis.tex` taken at 2026-10-10 during this audit. Chapter 5 was lines 1515-3661 in it.
  Another workflow is adding tables to Chapters 5-7, so line numbers move. Every fix below is located by quoting its
  paragraph's first eight words, not by line number.
- A paragraph counts as AI-written when a `% TODO: manual review` (or `% TODO: review, ...`) line sits directly above it
  and is not a `-- fixed:` or `-- moved:` note. This is the same `who()` test as `benchmark/split_hers_ai.py`. One
  paragraph whose marker reads `-- checked, no text changed` ("While this setup was initially intended for small molecules")
  counts as AI under that test and is included.
- Segmentation: `benchmark/segment_full_author.py`, run on the snapshot and written to a scratch copy, so the shared
  `full_segments_author_current.jsonl` was not overwritten. Measures: `compute_metrics.sent_metrics` and `normalise`, the
  aggregation of `split_hers_ai.py`, and the measures of `extra_measures.py` (MTLD, MATTR-50, sentence-length SD/CV/change,
  GPT-2 small surprisal, connectives, formulaic openers, three-item lists, pronouns). spaCy from the scratch `pylib`.
- Move codes (result in words, limitation, etc.) come from the existing coded files `moves_author_current.csv` and
  `moves_after_new_sentences.csv`, matched by sentence text. 133 of 170 AI results sentences and 31 of 31 AI conclusion
  sentences matched. Methods sentences were never coded, so methods has no move rows.
- Size of the AI-written text in Chapter 5: introduction 189 words (3 paragraphs, 11 sentences), methods 5,257 words
  (57 paragraphs), results 3,326 words (33 paragraphs), conclusions 627 words (6 paragraphs). Floats, captions and
  equations are excluded.
- Introduction AI text is too small for rates (11 sentences; `extra_measures.py` skips any cell under 20 sentences). Its
  three paragraphs were read one by one instead.

"Reference range" is min-max over Buttenschoen, Dablander and Vost for that section type, from `BENCHMARK_TABLE.md`.
"Hers" is the author's own text in the whole thesis, same table.

## Structure and content rows that apply to Chapter 5

| Row | Status in Ch5 | Verdict and fix |
|---|---|---|
| Chapter opening links back | Her paragraph "So far, Chapter~\ref{chap:flaws} explained how the standard methods" names Chapter 4. | Done. Her text, untouched. |
| Chapter ends point forward | AI paragraph "This chapter used the hybrid only as input" names Chapter 6 and its question. A second forward pointer ends "The fourth question asked whether weighting features". | Done, but the two pointers repeat each other. Fix S3 merges them. |
| Text between section title and first subsection | Missing in 2 places. `\section{Results}` goes straight to `\subsection{Performance of the hybrid}`. `\subsection{Sources of the gain}` goes straight to `\subsubsection{Selection at a fixed number of features}`. Methods has a one-sentence lead-in (hers). The other subsections that have subsubsections all open with text. | Not done. Fixes S1 and S2 add one lead-in each (new AI text, to be marked). |
| Section titles match content | Under `\subsubsection{Using importance during training}` sit three paragraphs about other experiments: fewer features against a tuned model ("Choosing fewer features does not replace tuning"), a new embedding ("On small molecules the hybrid uses a new"), and conformal intervals ("With 90\% conformal intervals, the hybrid's intervals"). Methods gives each of these its own subsubsection ("Fewer features or a tuned model", "Adding a new embedding", "Calibration"). | Not done. Fix S4. |
| Glued paragraphs | The conformal paragraph has no blank line before it. Only `% SOURCE` and `% TODO` comment lines separate it from "On small molecules the hybrid uses a new". LaTeX prints them as one paragraph of 188 words on two topics. | Not done. Fix S5 (insert a blank line). |
| Title case / title form | Every Ch5 heading is sentence case and a noun phrase. | Done. |
| Numbers in results text per 100 words | AI results: 1.26 per 100 words. Reference 1.2-5.9. Her rule: pattern in words, point to the table, at most one score pair per paragraph. | Done for Ch5 prose. Two paragraphs still carry numbers. "When each of five sites holds a different" has three RMSE values and no table to point to (fix R19: keep them until a table exists). "By default, every model uses the same settings" repeats a pair that is in `tab:tuning_base_reps` (fix M20). |
| Results sentences pointing to a figure or table | 35 of 170 AI results sentences (20.6%) carry a `Table~\ref` or `Figure~\ref`. Reference 3.1-18.2%. | Done in Ch5. The thesis-wide 1-3% is not this chapter's problem. |
| Results prose words per float | About 25 floats in Ch5 Results against 3,326 words of prose, so about 130 words per float. Reference 268-2,270. | In flux: the table workflow is still adding tables. Not judged. |
| Limitations | Conclusions: 7 of 31 sentences (22.6%) are coded as limitation. Reference 0-5.5%. Most sit in one paragraph, "The hybrid needs data. Its gains are consistent", which carries `\label{sec:method:limitations}`. The Ch5 introduction and the separate-site methods paragraph both point to that label. | Not done. Keep the paragraph, since two pointers need it, but cut the clauses that repeat Methods (fix C5). Cut the duplicate limitation clauses in "On antibodies the hybrid helps for the same" (fix C3). |
| Captions | AI-marked captions in Ch5: 33, mean 56.6 words. Reference means 49.8-73.5. The longest are `tab:method:one_fold_fewer` (94 words) and `tab:datasets` (93 words). | Inside the range. Captions are being written now by the table workflow, so they are not judged here. |
| Decimals | No four-decimal value in AI prose. Four decimals in AI tables are tie cases (0.4867/0.4867, 0.5891/0.5891, 0.7743/0.7742, 0.2847/0.2845, 0.4382/0.4382, 20.22/20.21). DECIMALS_RULE.md rule 4 allows these. | Done. |
| Figure/table references | AI text writes "Figure" and "Table" in full. | Done. |
| Negative results plainly | "On nanobody thermostability, SMAC colloidal stability, HIC retention" says the hybrid "adds nothing". "On the Buchwald--Hartwig reactions the hybrid lowers" reports the loss with the random forest. | Done. |
| Statistics | Per-fold counts and spread only; no tests. | Done. |
| "you" in body | 0 in AI text. The ones in Ch5 are in her paragraphs. | Not a target (her text). |
| Hedges | See the language tables. | Not a target in Ch5. |

## Language measures, AI-written paragraphs of Chapter 5 only

Verdicts: **done** = inside the reference range or on the right side of it. **target** = read the paragraphs and fix.
**not a target** = outside the range for a reason that is the author's preference or the nature of the text.

### Methods (5,257 AI words)

| Measure | Reference range | Hers (thesis) | AI Ch5 now | Verdict |
|---|---|---|---|---|
| Words per sentence | 22.3-28.0 | 20.1 | 19.0 | Not a target. She prefers short sentences. The problem is uniform length (see SD). |
| % sentences over 30 words | 21.1-32.7 | 12.2 | 5.1 | Target, mildly. A few merges below create sentences of 30-45 words. |
| Dependent clauses per sentence | 0.46-0.95 | 0.50 | 0.46 | Done (bottom edge). |
| Words per paragraph | 76.3-142.4 | 81.9 | 92.2 | Done. Fragment paragraphs exist: "On nanobody thermostability the search picked 1,000" is 24 words (fix M2). |
| Lexical density % | 58.8-59.7 | 56.6 | 56.0 | Not a target (same as hers). |
| Nominalizations per 100 words | 3.6-5.0 | 3.8 | 3.3 | Not a target. Plain verbs are her rule. |
| % passive sentences | 40.8-45.6 | 28.2 | 32.1 | Not a target. Above hers already. |
| we/our/I per 100 words | 0.83-1.6 | 1.1 | 0.46 | Target. Some agentless passives hide our decisions (fixes M10, M27, M37). |
| Hedges per 100 words | 0.36-0.85 | 0.39 | 0.32 | Not a target. |
| Boosters per 100 words | 0.14-0.26 | 0.10 | 0.02 | Not a target. |
| AI-guide phrases per 100 words | 0-0.10 | 0.01 | 0 | Done. |
| Numbers per 100 words | 1.8-2.1 | 2.5 | 2.3 | Not a target. Methods numbers are settings. |
| % sentences with 3+ numbers | 3.7-4.7 | 6.0 | 4.3 | Done. |
| MTLD | 56.9-81.9 | 69.6 | 77.2 | Done. |
| MATTR-50 % | 75.2-79.8 | 78.5 | 79.2 | Done. |
| Sentence length SD (words) | 12.1-18.1 | 9.19 | 6.46 | **Target.** This is the uniformity problem. |
| Sentence length CV | 0.56-0.67 | 0.47 | 0.36 | **Target.** |
| Mean change in length, consecutive sentences | 12.5-17.1 | 9.13 | 6.93 | **Target.** Merging adjacent 6-12-word sentences fixes all three rows. |
| GPT-2 surprisal, bits per token | 7.04-7.42 | 7.67 | 8.07 | Not a target. Dataset and model names raise it; her own text is also above the range. |
| Surprisal SD | 1.11-1.37 | 1.22 | 1.03 | Not a target. |
| Connectives per 100 words | 1.16-1.76 | 1.80 | 2.20 | **Target.** "so" appears 37 times in 277 sentences. |
| Distinct connectives / used | 0.15-0.28 | 0.17 | 0.20 | Done. |
| % sentences opening with a formulaic connector | 1.0-3.5 | 0.85 | 0 | Not a target. "However," and "Moreover," openers are AI tells. |
| Three-item lists per 100 sentences | 7.0-8.4 | 12.8 | 14.1 | Target, mildly. Most lists here are lists of models or datasets, which are content. Fixed only where a list is rhetorical or two sentences repeat "Each..." (M24, M34). |
| Pronouns and demonstratives per 100 words | 3.6-5.5 | 5.04 | 4.38 | Done. |

### Results (3,326 AI words)

| Measure | Reference range | Hers (thesis) | AI Ch5 now | Verdict |
|---|---|---|---|---|
| Words per sentence | 24.7-31.1 | 21.4 | 19.6 | Not a target. |
| % sentences over 30 words | 24.4-47.5 | 13.1 | 4.1 | Target, mildly (merges below). |
| Dependent clauses per sentence | 0.58-1.0 | 0.60 | 0.53 | Not a target. |
| Words per paragraph | 99.2-177.7 | 107.5 | 100.8 | Done. Outliers: three fragments ("Every number in this section is a held-out" 36 words, "On angiopoietin-2 binding the hybrid is higher" 40, "On small molecules the hybrid scores better than" 45) and one 226-word paragraph ("In an antibody representation, as in a molecular"). Fixes R1, R3, R6, R8. |
| Lexical density % | 59.3-62.9 | 54.8 | 53.7 | Not a target (as hers). |
| Nominalizations per 100 words | 3.9-4.4 | 3.8 | 3.7 | Not a target. |
| % passive sentences | 26.2-30.6 | 14.9 | 13.5 | Not a target (as hers). |
| we/our/I per 100 words | 0.26-1.6 | 0.67 | 0.18 | Not a target in results. |
| Hedges per 100 words | 0.87-2.3 | 0.81 | 0.63 | Not a target. Hedge only weak evidence. |
| Boosters per 100 words | 0.24-0.48 | 0.23 | 0.09 | Not a target. |
| AI-guide phrases per 100 words | 0.09-0.58 | 0.05 | 0 | Done. |
| Numbers per 100 words | 1.2-5.9 | 2.9 | 1.26 | Done (her rule). |
| % sentences with 3+ numbers | 3.1-20.3 | 9.5 | 1.8 | Done (her rule). |
| Move: method/background restated % | 8.6-33.7 | 18.7 | 6.0 | Done. |
| Move: pointer to figure/table % | 3.1-18.2 | 1.0 | 2.3 | Done in substance. As a parenthesis, 20.6% of sentences point to a table. Two pointer-only openers are merged into their results (R13, R16). |
| Move: result as numbers % | 7.9-24.8 | 16.9 | 6.0 | Done (her rule). |
| Move: result in words % | 8.3-12.4 | 19.9 | 27.8 | Not a target. This is what her rule asks for. |
| Move: comparison of results % | 18.1-23.6 | 15.2 | 24.1 | Not a target (marginal). |
| Move: interpretation % | 7.4-27.1 | 16.2 | 22.6 | Done. |
| Move: comparison with literature % | 0-0.83 | 2.5 | 1.5 | Not a target. |
| Move: claim of strength % | 0.83-3.6 | 1.7 | 0.8 | Done (edge). |
| Move: limitation % | 0.83-5.0 | 3.7 | 5.3 | Target, mildly. Fixes R7, R9 and R20 fold the limitation into a result sentence rather than end on it. |
| Move: link/transition % | 1.7-6.4 | 4.2 | 0 | Target via structure: lead-ins S1 and S2. |
| Move: recommendation/implication % | 0-4.3 | 0 | 3.8 | Done. |
| MTLD | 55.9-83.8 | 69.9 | 68.0 | Done. |
| MATTR-50 % | 76.0-80.4 | 78.1 | 77.9 | Done. |
| Sentence length SD | 9.52-12.8 | 8.46 | 6.38 | **Target.** |
| Sentence length CV | 0.42-0.44 | 0.41 | 0.34 | **Target.** |
| Mean change in length | 9.96-14.2 | 9.72 | 7.74 | **Target.** |
| GPT-2 surprisal | 6.80-7.14 | 7.77 | 8.48 | Not a target (names; hers also above). |
| Surprisal SD | 1.01-1.04 | 1.22 | 1.04 | Done. |
| Connectives per 100 words | 1.59-3.12 | 2.32 | 2.01 | Done. But "So" opens a sentence in 5 results paragraphs and 1 conclusion paragraph: a verbal tic. Fixes R7, R10, R12, R15, R17, C2. |
| Distinct connectives / used | 0.21-0.32 | 0.14 | 0.26 | Done. |
| % formulaic openers | 5.2-8.8 | 1.81 | 0 | Not a target. |
| Three-item lists per 100 sentences | 4.65-10.0 | 13.3 | 8.8 | Done. |
| Pronouns and demonstratives | 2.72-5.78 | 5.10 | 3.59 | Done. |

### Conclusions (627 AI words, all six paragraphs AI-written)

| Measure | Reference range | Hers (thesis) | AI Ch5 now | Verdict |
|---|---|---|---|---|
| Words per sentence | 25.0-32.5 | 21.0 | 20.2 | Not a target. |
| % sentences over 30 words | 25.8-42.9 | 12.8 | 0 | Target, mildly. Fixes C2 and C3 make two sentences of 31-35 words. |
| Dependent clauses per sentence | 0.71-0.96 | 0.76 | 0.84 | Done. |
| Words per paragraph | 95.3-138.9 | 123.1 | 104.5 | Done. The forward-pointing last paragraph is 40 words; fix S3 grows it. |
| Lexical density % | 61.7-64.3 | 55.7 | 55.9 | Not a target. |
| Nominalizations | 4.2-6.9 | 4.4 | 5.6 | Done. |
| % passive | 14.6-30.1 | 25.5 | 16.1 | Done. |
| we/our/I per 100 words | 1.3-1.9 | 0.41 | 0.80 | Target. A finding stated with "we" (fix C2). |
| Hedges per 100 words | 0.95-2.2 | 1.4 | 0.48 | Not a target. |
| Boosters per 100 words | 0.24-0.90 | 0.20 | 0.32 | Done. |
| AI-guide phrases | 0.43-0.97 | 0 | 0 | Done. |
| Numbers per 100 words | 0.04-0.43 | 0.91 | 0 | Done. |
| Move: method/background restated % | 10.4-15.9 | 14.1 | 6.5 | Done. |
| Move: result in words % | 6.8-14.3 | 24.7 | 22.6 | Target, mildly. "On antibodies the hybrid helps for the same" lists results again (fix C3). |
| Move: comparison of results % | 3.0-17.8 | 7.1 | 22.6 | Not a target on its own. It falls with C3. |
| Move: interpretation % | 0-9.6 | 11.8 | 12.9 | Not a target. |
| Move: claim of strength % | 15.1-29.9 | 0 | 3.2 | **Target.** Fix C2 states one finding as an achievement. |
| Move: limitation % | 0-5.5 | 15.3 | 22.6 | **Target.** Fixes C3 and C5. |
| Move: link/transition % | 1.4-4.8 | 1.2 | 9.7 | **Target.** Two forward pointers; fix S3 merges them. |
| Move: recommendation/implication % | 16.4-17.9 | 18.8 | 0 | **Target.** Fix C2 adds one implication that the results already support ("which is why both are searched", Results). |
| MTLD | 104.2-167.3 | 80.8 | 87.9 | Not a target. "hybrid" and dataset names repeat by necessity. |
| MATTR-50 % | 83.1-87.9 | 79.4 | 79.4 | Not a target. |
| Sentence length SD | 8.0-12.6 | 9.24 | 6.48 | **Target.** |
| Sentence length CV | 0.32-0.41 | 0.45 | 0.32 | Done (edge). |
| Mean change in length | 8.53-12.3 | 9.40 | 9.0 | Done. |
| GPT-2 surprisal | 6.57-6.84 | 7.57 | 8.14 | Not a target. |
| Surprisal SD | 0.73-1.03 | 1.19 | 0.79 | Done. |
| Connectives per 100 words | 2.12-2.72 | 2.23 | 1.77 | Not a target. "also" is 6 of 11 connectives; fix C4 replaces one. |
| Distinct connectives / used | 0.33-0.51 | 0.42 | 0.36 | Done. |
| % formulaic openers | 4.2-13.0 | 1.06 | 0 | Not a target. |
| Three-item lists per 100 sentences | 9.4-14.3 | 12.8 | 12.9 | Done. |
| Pronouns and demonstratives | 5.7-7.1 | 4.98 | 5.96 | Done. |

### Introduction (189 AI words, 3 paragraphs)

Too small for rates. Read individually. The issues are an announcing first sentence, a 5-word sentence, and two paragraphs
of four sentences of 11-24 words each. Fixes I1-I3.

## Fix list

Each fix gives the paragraph's first eight words, the exact old text, the new text and the benchmark row it serves.
Meaning and numbers are unchanged unless a fix says otherwise. Keep every `% TODO: manual review` marker in place above
the edited paragraph. New sentences written here are AI text and need a marker.

### Structure

**S1. Results section lead-in.** Row: text between section title and first subsection; link/transition 0% in results.
Insert directly after `\section{Results}` (Chapter 5, the one followed by `\subsection{Performance of the hybrid}`):
New: `We first compare the hybrid with single representations, with ensembles and with simpler ways of choosing features, then ask where its gain comes from and what its blueprints show, and end with the collaboration and separate-site studies.`

**S2. "Sources of the gain" lead-in.** Row: text between section title and first subsection.
Insert after `\subsection{Sources of the gain}\label{sec:method:results_why}`:
New: `This section asks whether the gain is more than a cut in the number of features, and how it depends on the downstream model and on reusing the importance scores in training.`

**S3. One forward pointer, not two.** Paragraphs "The fourth question asked whether weighting features by their" and
"This chapter used the hybrid only as input". Rows: link/transition 9.7% in conclusions (reference 1.4-4.8); chapter
end points forward; fragment paragraph (40 words).
Cut from the first: `Whether hybrids keep more of their performance under label noise, and whether their features stabilize noise mitigation, are the subjects of Chapters~\ref{chap:similarity} and~\ref{chap:noise}.`
Old (second): `This chapter used the hybrid only as input to a model. Chapter~\ref{chap:similarity} asks whether a feature set chosen for the task also gives better neighbors, which every method that relies on similarity could use, including the detection of noisy labels.`
New: `This chapter used the hybrid only as input to a model. Whether hybrids keep more of their performance under label noise, and whether their features stabilize noise mitigation, are the subjects of Chapters~\ref{chap:similarity} and~\ref{chap:noise}. Chapter~\ref{chap:similarity} first asks whether a feature set chosen for the task also gives better neighbors, which every method that relies on similarity could use, including the detection of noisy labels.`

**S4. Headings under "Using importance during training".** Row: section titles match content.
That subsubsection holds three more results paragraphs: "Choosing fewer features does not replace tuning the model", "On small molecules the hybrid uses a new", and "With 90\% conformal intervals, the hybrid's intervals". Each has its own subsubsection in Methods. Option A: give each the Methods title (`\subsubsection{Fewer features or a tuned model}`, `\subsubsection{Adding a new embedding}`, `\subsubsection{Calibration}`). Option B: one subsubsection `\subsubsection{Tuning, new embeddings and calibration}` above the first. A keeps Methods and Results parallel. B adds one heading, not three. Recommendation: A.

**S5. Glued paragraphs.** Paragraph "On small molecules the hybrid uses a new". There is no blank line between its last sentence (`On the two antibody datasets another language model of the same sequence adds little.`) and the `% SOURCE: .../c08_calibration/...` line. So the conformal paragraph prints as part of it. Insert one blank line before that `% SOURCE` line. Row: words per paragraph (188 words on two topics).

### Introduction

**I1. "Other methods handle training on several representations or".** Rows: link sentence that only announces; the
5-word sentence; "so" twice. The first sentence announces "too many features", which this paragraph does not cover. It is
covered in her next paragraph.
Old: `Other methods handle training on several representations or on too many features. Ensembling, a form of late fusion, fits one model per representation and aggregates their predictions, so the representations are combined after training. Consensus QSAR works this way \citep{Valsecchi2020}.`
New: `Ensembling, a form of late fusion, fits one model per representation and aggregates their predictions, so the representations are combined after training, as in consensus QSAR \citep{Valsecchi2020}.`

**I2. "One caveat concerns sparse representations such as one-hot".** Rows: sentence-length variation (11, 19, 24, 12 words).
Old: `One caveat concerns sparse representations such as one-hot encoded sequences. Each of their features is set in only a few samples, so each scores low importance on its own.`
New: `One caveat concerns sparse representations such as one-hot encoded sequences, whose features are each set in only a few samples and so each score low importance on their own.`
Old: `The blocks substitution matrix (BLOSUM) encoding is a base representation on humanness.`
New: merge into the sentence before it. Replace its final `.` (after `CHO polyreactivity`) with `, and the blocks substitution matrix (BLOSUM) encoding is one on humanness.`
Keep the inline `% TODO: manual review` and the `-- fixed:` comment line between the two sentences. They are comments and do not break the paragraph.

**I3. "While this setup was initially intended for small".** Rows: sentence-length variation. "entirely separate" is a booster.
Old: `Reactions (one Buchwald--Hartwig yield dataset) are used to test an application of this framework in an entirely separate domain.`
New: `One Buchwald--Hartwig yield dataset tests the method on reactions, a separate domain.`

### Methods

**M1. "Two variants of this search are used later."** Row: link sentence that only announces (8 words) -> merge.
Old: `Two variants of this search are used later. Under label noise,`
New: `Two variants of this search are used later: under label noise,`

**M2. "On nanobody thermostability the search picked 1,000 features".** Rows: fragment paragraph (24 words); a result inside Methods.
Cut the whole paragraph from the budget description in "The search space". Move it into the Results paragraph "The number of features the search picks depends" (see R14).

**M3. "The fixed-share methods differ only in how".** Row: connective repeated ("so" twice in two sentences).
Old: `Every rescaled score lies between 0 and 1, so softmax cannot give one representation most of the budget:`
New: `Because every rescaled score lies between 0 and 1, softmax cannot give one representation most of the budget:`
Old: `Rank decay uses only the order, so with 10 representations`
New: `Rank decay uses only the order: with 10 representations`

**M4. "The search tests every combination of the six".** Row: uniform length (17, 13, 21, 17, 16).
Old: `Each dataset's grid is a small cut of the full search, made to save compute (Section~\ref{sec:method:decided_grid}). At worst, a smaller grid finds a less informative hybrid than the full search would have.`
New: `Each dataset's grid is a small cut of the full search, made to save compute (Section~\ref{sec:method:decided_grid}); at worst, it finds a less informative hybrid than the full search would have.`

**M5. "Donors are set per dataset (Augmentation column of".** Row: length variation (9 then 13 words).
Old: `Donors are set per dataset (Augmentation column of Table~\ref{tab:dataset_reps}). On QM9 the topological`
New: `Donors are set per dataset (Augmentation column of Table~\ref{tab:dataset_reps}), and on QM9 the topological`

**M6. "Donor features are ranked by what the base".** Rows: fragmentation (9- and 7-word sentences); a reason cut off from its claim.
Old: `for the internal-training samples. For binary tasks, $\hat{y}$ is the positive-class probability. In-sample residuals would shrink toward zero.`
New: `for the internal-training samples, since in-sample residuals would shrink toward zero. For binary tasks, $\hat{y}$ is the positive-class probability.`

**M7. "In uncapped mode, extras are appended outside the".** Rows: uniform length (21, 21, 21, 12); "so" three times.
Old: `In uncapped mode, extras are appended outside the budget, so the final width is $|\mathbf{H}| + k^*$, at most $B + 40$ features. In capped mode, they replace the base features with the lowest random forest importance, so the width stays at most $B$.`
New: `In uncapped mode, extras are appended outside the budget and the final width is $|\mathbf{H}| + k^*$, at most $B + 40$ features, whereas in capped mode they replace the base features with the lowest random forest importance and the width stays at most $B$.`

**M8. "Augmentation is not a searched setting."** Rows: 6-word fragment sentence; we/our.
Old: `Augmentation is not a searched setting. A dataset with donors is searched three times,`
New: `Augmentation is not a searched setting: a dataset with donors is searched three times,`
Old: `The searches in this chapter therefore run without augmentation on all 20 datasets`
New: `We therefore ran the searches in this chapter without augmentation on all 20 datasets`

**M9. "The residual ranking has four limitations."** Row: "so" three times.
Old: `Finally, the margin over $k=0$ is zero, so a near-tie on one internal split can admit extras.`
New: `Finally, with a zero margin over $k=0$, a near-tie on one internal split can admit extras.`

**M10. "Table~\ref{tab:cut_walkthrough} follows one dataset, hERG $K_i$, through".** Row: length variation (9, 11, 20, 10, ...).
Old: `through the cuts. Its earlier search, without augmentation, covered`
New: `through the cuts; its earlier search, without augmentation, covered`
Old: `random forest evaluator, all representations). Its standard deviation over the five folds was 0.044 p$K_i$.`
New: `random forest evaluator, all representations), with a standard deviation of 0.044 p$K_i$ over the five folds.`

**M11. "The search also has an adaptive pruning option."** Row: order of moves. The reason for not using the option comes before what it is.
Old: `The search also has an adaptive pruning option. We did not use it, because the per-dataset cuts above remove far more configurations. It scores every configuration on two of the five folds and stops scoring those far behind, as successive halving \citep{Jamieson2016} and Hyperband \citep{Li2018Hyperband} do.`
New: `The search also has an adaptive pruning option, which scores every configuration on two of the five folds and stops scoring those far behind, as successive halving \citep{Jamieson2016} and Hyperband \citep{Li2018Hyperband} do. We did not use it, because the per-dataset cuts above remove far more configurations.`

**M12. "Every experiment splits each dataset into outer folds".** Rows: fragmentation; a sentence that only restates the one before.
Old: `Three experiments also limit each held-out feature to the range it takes on the training samples. They are the comparison with the best single representation, the easy and hard halves, and the comparison of many and few representations. The other experiments do not limit held-out features.`
New: `Only three experiments limit each held-out feature to the range it takes on the training samples: the comparison with the best single representation, the easy and hard halves, and the comparison of many and few representations.`

**M13. "The hybrid is compared with the best single".** Row: "so" three times. One "so" is attached to the wrong sentence: the comparator differs between folds because it is chosen per fold, not because it is scored on the same samples.
Old: `This comparator is chosen per downstream model and per fold from the search's scores on the outer training samples. It is then refit on those samples and scored on the same held-out samples as the hybrid, so it can differ between folds.`
New: `This comparator is chosen per downstream model and per fold from the search's scores on the outer training samples, so it can differ between folds. It is then refit on those samples and scored on the same held-out samples as the hybrid.`

**M14. "The kNN model is not a default downstream".** Row: 9-word opener; the role comes after the setting.
Old: `The kNN model is not a default downstream model. It takes $k = 5$ neighbors by default, lowered to the square root of the number of training samples when a fold has fewer than 25. It is a third downstream model in the search with the nearest-neighbor objective (Chapter~\ref{chap:similarity}), on every dataset.`
New: `The kNN model is not a default downstream model, but a third downstream model in the search with the nearest-neighbor objective (Chapter~\ref{chap:similarity}), on every dataset. It takes $k = 5$ neighbors by default, lowered to the square root of the number of training samples when a fold has fewer than 25.`
(The trailing inline `% TODO: manual review` stays at the end of the paragraph.)

**M15. "As a second reference beside the best single".** Row: "so" twice.
Old: `PCA cannot have more components than training samples, so where the budget is larger the two widths differ.`
New: `PCA cannot have more components than training samples, and where the budget is larger the two widths differ.`

**M16. "The ensemble fits one model per representation and".** Rows: three-item lists (3 in 5 sentences); two sentences that both open "Each".
Old: `Each picks its members, the top 3 representations or all of them. Each also picks one of seven ways to pool predictions,`
New: `Each picks its members, the top 3 representations or all of them, and one of seven ways to pool predictions,`

**M17. "On each outer fold, the base representations are".** Row: uniform length (18, 20, 22, 14, 15, 23).
Old: `with an even split of $B$ over every base representation. Each of these uses the importance method that the fold's best hybrid used.`
New: `with an even split of $B$ over every base representation, each using the importance method that the fold's best hybrid used.`

**M18. "Datasets above 2{,}000 samples are first reduced to".** Row: we/our in methods.
Old: `Three other splits (by heavy-atom count, label spread within a scaffold and feature spread within a scaffold) were run and are not reported here.`
New: `We also ran three other splits (by heavy-atom count, label spread within a scaffold and feature spread within a scaffold), which are not reported here.`

**M19. "A second check repeats each outer fold's search".** Row: "so" twice in two adjacent sentences.
Old: `The held-out fold is the same, so the two searches differ by one fold of training samples. Each search also picks its own budget, so the two hybrids can differ in width as well.`
New: `The held-out fold is the same and the two searches differ by one fold of training samples, but each search also picks its own budget, so the two hybrids can differ in width as well.`

**M20. "By default, every model uses the same settings".** Row: her numbers rule. The pair is printed in `tab:tuning_base_reps` (LogD, SVM, MHG-GNN: 0.475, 0.421), so the prose need not repeat it.
Old: `The SVM gained most: on LogD with MHG-GNN, RMSE fell from 0.475 to 0.421.`
New: `The SVM gained most, on LogD with MHG-GNN.`
The QM9 pair (0.487 and 0.495) is in no table, so it stays as the paragraph's one pair.

**M21. "One search per outer fold is run for".** Row: length variation (11, 10, 23, ...).
Old: `The matched evaluator belongs to the downstream model's family. A near-family evaluator pairs`
New: `The matched evaluator belongs to the downstream model's family, while a near-family evaluator pairs`

**M22. "Four comparisons run inside every held-out fold of".** Rows: a list split over three sentences ("They are", "The other two are").
Old: `Four comparisons run inside every held-out fold of the target. They are the best single representation and, as the upper reference, a hybrid searched on the target's training molecules. The other two are the source settings with features re-selected on the target, and the source's selected features applied unchanged.`
New: `Four comparisons run inside every held-out fold of the target: the best single representation, a hybrid searched on the target's training molecules as the upper reference, the source settings with features re-selected on the target, and the source's selected features applied unchanged.`

**M23. "The second study counts the molecules that a".** Row: 9-word follow-on sentence.
Old: `on every representation of each dataset (Chapter~\ref{chap:flaws}). It then asks whether a hybrid removes the problem.`
New: `on every representation of each dataset (Chapter~\ref{chap:flaws}), and asks whether a hybrid removes the problem.`

**M24. "Three further experiments are run."** Rows: 5-word sentence that only announces; 236-word paragraph.
Old: `Three further experiments are run. In the first, labels are made two-class:`
New: `In the first of three further experiments, labels are made two-class:`
Also split the paragraph before `In the second experiment,`. This gives two paragraphs of about 150 and 85 words.

**M25. "The separate-site work has two further limitations beside".** Row: uniform length (13, 11, 12, 18).
Old: `It depends on the collaborator's implementation of the published process. The published process trains a student on a large separate public set. Here the training molecules take that role as a compromise, and held-out molecules are excluded from it.`
New: `It depends on the collaborator's implementation of the published process, and that process trains a student on a large separate public set, a role the training molecules take here as a compromise, with held-out molecules excluded from it.`

### Results

**R1. "Every number in this section is a held-out".** Rows: fragment paragraph (36 words); length variation. It stays as the lead-in to "Performance of the hybrid".
Old: `unless stated otherwise. Most DNN results were computed before`
New: `unless stated otherwise, and most DNN results were computed before`

**R2. "On small molecules the hybrid scores better than".** Row: fragment paragraph (45 words) -> merge with next.
Join it with the next paragraph, "On every small-molecule dataset the hybrid also beats" (delete the blank line between them; keep that paragraph's marker as a comment line). Both are the small-molecule result.

**R3. "In an antibody representation, as in a molecular".** Rows: uniform length (11 sentences of 16-28 words); 226 words, above every reference mean.
Old: `the features that matter are few and buried among many that carry nothing. A language model's embedding`
New: `the features that matter are few and buried among many that carry nothing: a language model's embedding`
Old: `On the anti-VEGF expression scan it is the CDR readout of ESM3 with the SVM. On $\Delta\Delta G$ it is`
New: `On the anti-VEGF expression scan it is the CDR readout of ESM3 with the SVM, and on $\Delta\Delta G$ it is`
Split the paragraph before `In the $\Delta\Delta G$ hybrid, the representations built for this thesis`. The first part covers which representation is best in hindsight. The second covers which representations the hybrid draws on.

**R4. "On angiopoietin-2 binding the hybrid is higher with".** Rows: fragment paragraph (40 words, 16-8-16); merge.
Old: `On angiopoietin-2 binding the hybrid is higher with both models (Table~\ref{tab:hybrid_vs_best_single}), on three of five folds. Its labels were read off a published heatmap.`
New: `On angiopoietin-2 binding, whose labels were read off a published heatmap, the hybrid is higher with both models on three of five folds (Table~\ref{tab:hybrid_vs_best_single}).`
Optionally join the result to the end of the previous paragraph, "On nanobody thermostability, SMAC colloidal stability, HIC retention", which also covers the small antibody sets.

**R5. "Figures~\ref{fig:best_number_of_features} and~\ref{fig:best_number_of_features_antibodies} show the budget each".** Rows: limitation in results; placement. The paragraph is about budget picks, so it belongs in "Dependence on the downstream model" next to "The number of features the search picks depends". It now sits under "Against the best single representation", and the two paragraphs say the same thing ("the downstream model moved the pick as much as the dataset did" / "depends on the dataset and on the downstream model"). Move it to just before that paragraph. Then apply R14's cut of the duplicate claim.
Old: `Dataset size did not order the budgets either. QM9, the largest dataset`
New: `Dataset size did not order the budgets either: QM9, the largest dataset`

**R6. "With the same downstream model on every representation,".** No change. Varied lengths, interpretation placed last.

**R7. "At the same budget, drawing on several representations".** Rows: "So" opener; 11- and 8-word sentences.
Old: `on most datasets with a result. On the rest it is worse at all but one budget.`
New: `on most datasets with a result, and at all but one budget on the rest.`
Old: `So the gain comes from which representations are drawn on and from ranking features within each one, more than from the allocation rule that sizes their shares.`
New: `The gain comes more from which representations are drawn on, and from ranking features within each one, than from the allocation rule that sizes their shares.`

**R8. "On small molecules the hybrid also gains on".** Row: uniform length (21, 21, 14, 20).
Old: `is larger than on the common half. Caco-2 and hERG $K_i$ show the same,`
New: `is larger than on the common half, and Caco-2 and hERG $K_i$ show the same,`

**R9. "Leaving one more fold of training samples out".** Rows: "So" opener; limitation as the closing sentence.
Old: `The two searches differ in two ways at once: the training samples and the budget each one picks. So we cannot say which of the two causes the loss.`
New: `The two searches differ in two ways at once, the training samples and the budget each one picks, so we cannot say which of the two causes the loss.`
The range "0.009 to 0.037 RMSE" is a difference, not a table cell. It stays unless `tab:method:one_fold_fewer` gains a difference column.

**R10. "Figure~\ref{fig:best_number_of_features} draws recursive feature elimination and random".** Row: re-describes the figure (22-word opener restating the method). Point to it instead.
Old: `Figure~\ref{fig:best_number_of_features} draws recursive feature elimination and random features, both over all base representations joined, beside the hybrid at each number of features, and Table~\ref{tab:method:feature_reduction} gives the scores.`
New: `Figure~\ref{fig:best_number_of_features} and Table~\ref{tab:method:feature_reduction} compare the hybrid with recursive feature elimination and random features over all base representations joined.`

**R11. "Across the configurations of a search, the choice".** Row: 11-word interpretive sentence after a 13-word one.
Old: `With $C = 10$ it picks larger budgets on LogD and rat microsomal clearance. A model that is allowed to fit more can use more features.`
New: `With $C = 10$ it picks larger budgets on LogD and rat microsomal clearance, since a model that is allowed to fit more can use more features.`

**R12. "Scoring candidate hybrids with the downstream model itself,".** Row: 6-word fragment sentence.
Old: `On some antibody datasets it matters. The matched evaluator is best with both models on EGFR binding, expression titer and colloidal stability. On EGFR binding with the SVM it is well ahead of a LightGBM evaluator.`
New: `On some antibody datasets it matters: the matched evaluator is best with both models on EGFR binding, expression titer and colloidal stability, and on EGFR binding with the SVM it is well ahead of a LightGBM evaluator.`

**R13. "Figure~\ref{fig:blueprints} shows one fold of each dataset.".** Row: pointer-only 7-word opener -> merge with the result.
Old: `Figure~\ref{fig:blueprints} shows one fold of each dataset. Several hybrids draw`
New: `Figure~\ref{fig:blueprints} shows one fold of each dataset: several hybrids draw`

**R14. "The number of features the search picks depends".** Rows: duplicate of the moved paragraph (R5); receives M2; short interpretive sentence.
Old: `On Caco-2 the search picked more features for the DNN than for the SVM.`
New: `On Caco-2 the search picked more features for the DNN than for the SVM, and on nanobody thermostability it picked 1,000 features for the DNN on four of five folds against 250 or 500 for the SVM (Figure~\ref{fig:best_number_of_features_antibodies}).`
Old: `showed for distances. A kernel computed over many features`
New: `showed for distances: a kernel computed over many features`
If R5 is applied, shorten the opening sentence to the pointer, because R5's paragraph now says the same thing just before it:
Old: `The number of features the search picks depends on the dataset and on the downstream model (Figures~\ref{fig:best_number_of_features} and~\ref{fig:best_number_of_features_antibodies}, Table~\ref{tab:method:budget_per_fold}).`
New: cut it. R5's paragraph already points to both figures and the table.

**R15. "On small molecules the hybrid uses a new".** Row: length variation (15, 16, 24, ...). See also S5.
Old: `added as one more base representation. Its selected features beat`
New: `added as one more base representation: its selected features beat`
"seven or eight of eight on LogD" and "fewer than half of the embeddings" are tallies under her rule. Replacing them needs the per-embedding scores. Flagged, not rewritten.

**R16. "LogD can be read chemically (Figure~\ref{fig:blueprint_logd}).".** Row: runs of 11-word sentences (6, 20, 11, 24, 11, 21, 19, 14, 11, 11).
Old: `38 of them Mordred descriptors and 36 ECFP4 bits. The rest come from eight representations whose features carry no names.`
New: `38 of them Mordred descriptors, 36 ECFP4 bits and the rest from eight representations whose features carry no names.`
Old: `(nBase) and several surface areas binned by partial charge (PEOE\_VSA). There are also counts of secondary amine and aromatic nitrogen environments.`
New: `(nBase), several surface areas binned by partial charge (PEOE\_VSA), and counts of secondary amine and aromatic nitrogen environments.`
Old: `A model of LogD should therefore need both lipophilicity and basicity. The hybrid picked both, on every fold, from two different representations.`
New: `A model of LogD should therefore need both lipophilicity and basicity, and the hybrid picked both, on every fold, from two different representations.`

**R17. "The features picked are partly the same from".** Row: "So" opener.
Old: `so a feature on one fold can stand in for a different one on another. So an index well below 1 need not mean the hybrid fits noise.`
New: `so a feature on one fold can stand in for a different one on another, and an index well below 1 need not mean the hybrid fits noise.`
Check (not language): this paragraph points to `tab:method:one_fold_fewer` for the stability index. That table's caption does include stability, so the pointer resolves.

**R18. "On hERG $K_i$ the graph kernel leaves some".** Row: "So" opener.
Old: `So most of the gain comes from the information`
New: `Most of the gain therefore comes from the information`

**R19. "Feature selection can run where data cannot be".** Row: "So" opener after an 8-word sentence.
Old: `It can build a different number of features. So each site can select on its own and send only feature indices.`
New: `It can build a different number of features, but each site can select on its own and send only feature indices.`

**R20. "When each of five sites holds a different".** Rows: her tally rule ("in three of the four dataset and model pairs"); numbers.
Old: `It also beats the collaborator's two ensembles, which combine the sites' predictions, in three of the four dataset and model pairs. The exception is Caco-2 with the DNN, where the ensembles reach 0.272 and 0.273 RMSE against 0.284 for the hybrid.`
New: `It also beats the collaborator's two ensembles, which combine the sites' predictions, except on Caco-2 with the DNN, where the ensembles reach 0.272 and 0.273 RMSE against 0.284 for the hybrid.`
No table holds this experiment yet (the source is fed02). The numbers stay until one exists. Then point to it and cut them.

**R21. "With two-class labels and up to 30\% of".** Row: limitation as closing sentence.
Old: `Only Caco-2 has finished, with three seeds for the DNN and two for the SVM.`
New: cut, and open the paragraph with: `On Caco-2, the only dataset finished (three seeds for the DNN, two for the SVM), with two-class labels and up to 30\% of each site's labels flipped,` followed by the rest of the first sentence from `selecting features at each site beats`.

### Conclusions

**C1. "This chapter asked how the best set of".** No change. It already states the finding with "we".

**C2. "At the same number of features the hybrid".** Rows: "So" opener; claim of strength 3.2% (reference 15-30); recommendation/implication 0% (reference 16-18); we/our.
Old: `on all but one small-molecule dataset with a result. So its gain is more than a reduction in width.`
New: `on all but one small-molecule dataset with a result, which shows that its gain is more than a reduction in width.`
Old: `We found no setting that suits every dataset and model.`
New: `We found no setting that suits every dataset and model, so the settings should be searched for each new task rather than fixed in advance.`
This adds an implication. The Results paragraph "The number of features the search picks depends" already ends on it ("which is why both are searched").

**C3. "On antibodies the hybrid helps for the same".** Rows: limitation 22.6% (reference 0-5.5); result in words 22.6% (reference 6.8-14.3); three-item list. The first two clauses repeat "The hybrid needs data".
Old: `On the smaller sets its gains are inconsistent, the search's own scores are optimistic, and an ensemble of one model per representation often does better.`
New: `On the smaller sets an ensemble of one model per representation often does better.`

**C4. Same paragraph.** Rows: "also" 6 of 11 connectives in conclusions. The sentence also says more than Results shows: new embeddings help on small molecules and on fewer than half of the embeddings on colloidal stability, expression titer and Buchwald--Hartwig.
Old: `The hybrid also takes in new embeddings added to the pool.`
New: `On small molecules the hybrid makes use of a new embedding added to the pool.`
This narrows the claim to what "On small molecules the hybrid uses a new" reports. It is a change of meaning, so it needs her approval.

**C5. "The hybrid needs data. Its gains are consistent".** Row: limitation share. Keep the paragraph and its `\label{sec:method:limitations}`: the Ch5 introduction and "The separate-site work has two further limitations" point to it, the second for the simulated sites. Cut only clauses already stated in Methods or in the Results lead-in.
Old: `The search is costly, must be rerun when the data, representations or model change, and ran on grids cut to save compute (Section~\ref{sec:method:decided_grid}).`
New: `The search is costly and must be rerun when the data, representations or model change.`
Old: `Most DNN results predate the correction described in Section~\ref{sec:method:models}, and the separate sites are simulated from one public dataset.`
New: `The separate sites are simulated from one public dataset.`
(The DNN caveat stays in the Results lead-in "Every number in this section is a held-out", and the cut grids in "The search tests every combination of the six".)

**C6. "This chapter used the hybrid only as input".** See S3.

## Count

57 numbered fixes change text: structure 5 (S1-S5), introduction 3 (I1-I3), methods 25 (M1-M25), results 20 (R1-R21
without R6), conclusions 4 (C2-C5). R6, C1 and C6 are listed for completeness and change nothing. Every quoted old
string was checked against thesis.tex at the end of this audit: each occurs exactly once.
