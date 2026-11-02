# Benchmarks for checking the thesis against the three reference theses

None of these is a hard rule. Each benchmark is a reference practice. Each one is checked in context: by section type (introduction, background, methods, results, discussion, conclusions, abstract), by the kind of paragraph, and by what the paragraph has to do. A value outside the reference range is a reason to read that paragraph, not a verdict. Ranges are the lowest and highest of Buttenschoen (B), Dablander (D) and Vost (V), from `benchmark/metrics_section_types.csv` and the notes in `~/Documents/ReferencePapers/Theses/notes/`.

## A. Results text: numbers, tables and figures

1. **Every claim is backed.** Each comparative or evaluative claim ("better", "most", "little", "helped", "beat", "on N of M") has the numbers behind it, or points to the table or figure that holds them. Checked claim by claim.
2. **How many numbers.** Numbers per 100 words in results: references 1.2 to 5.9 (B 5.9, V 1.9, D 1.2). More is right where no table carries the numbers; fewer is right where a table or figure already shows them.
3. **Which numbers stay in the sentence.** The numbers that carry the paragraph's main finding stay in the sentence, as score pairs in the dataset's metric. Secondary numbers belong in a table. B puts the key percentages in brackets after the figure pointer; D gives 3-decimal values.
4. **Runs of numbers.** Sentences with three or more numbers: references 3 to 20 per cent of results sentences. A paragraph with many numbers that belong together is a table candidate.
5. **Pointers to tables and figures.** Sentences that point to a figure or table: references 3 to 18 per cent. Every table and figure is cited in the text, near where it appears.
6. **Finding first.** Whether a results paragraph opens with what was found, or with a restatement of the method.
7. **The mix of sentence jobs in results** (rhetorical moves, coded by reading). References:
   - method or background restated: 9 to 34 per cent;
   - pointer to a figure or table: 3 to 18;
   - result given as numbers: 8 to 25;
   - result given in words: 8 to 12;
   - comparison of results: 18 to 24;
   - interpretation: 7 to 27;
   - limitation: 1 to 5;
   - link to the next part: 2 to 6.
8. **Figures and tables per prose word** in results chapters: references 268 to 805 words per float.
9. **Spread and folds.** Claims that rest on folds report per-fold scores or their range, and say how many folds. Headline claims say how many folds they rest on. Controls are named where they ran.
10. **Decimals.** Three decimals for scores, one for percentages, with the tie rule; the same within a sentence.

## B. Discussion and conclusions

11. **The mix of sentence jobs in conclusions.** References:
    - claims of what was achieved: 15 to 30 per cent;
    - recommendations or implications: 16 to 18;
    - comparison with the literature: 0 to 18;
    - limitations: 0 to 6;
    - results restated in words: 7 to 14.
12. **Numbers in conclusions:** references 0 to 0.4 per 100 words. Findings are in words, with a number only where it anchors a headline claim.
13. **"We" in conclusions:** references 1.3 to 1.9 per 100 words (introductions 1.4 to 1.7).
14. **Limitations.** Which ones, and where they sit: in the chapter's discussion or conclusions, and the final chapter. They are specific: they name the dataset, fold or missing control. There are few of them, and they are not repeated across sections.
15. **Mixed and negative results** are stated plainly, with the comparison that shows them. Positive results are not inflated. Boosters ("clearly", "show that"): references 0.2 to 0.9 per 100 words.
16. **Definitiveness.** Hedges per 100 words: results 0.9 to 2.3, discussion 1.2 to 1.5, conclusions 0.9 to 2.2. Each hedge matches the strength of the evidence: one fold or no control gets a hedge, every fold with a control does not.
17. **Originality.** Contributions are named in the abstract, Chapter 1 and the final chapter, as what was built or found. "No earlier use" claims appear only where the literature check supports them.

## C. Structure and flow

18. **Chapter openings.** A paragraph between the chapter title and the first section, linking back to the previous chapter.
19. **Section lead-ins.** A lead-in paragraph between a section title and its first subsection, where the references have one.
20. **Chapter conclusions point forward** to the next chapter.
21. **Research questions.** Stated in Chapter 1, re-asked in each chapter's introduction (as the references do), and answered in order in the final chapter.
22. **Abstract.** References 250 to 568 words; the thesis limit is 300. Structure: the problem, the findings of each part, the closing claim. Almost no numbers.
23. **Introduction and final chapter length.** Introductions 2,158 to 9,386 words, with an outline paragraph per chapter. Final chapters 896 to 1,593 words.
24. **Section titles.** Sentence case, 2 to 6 words, noun or gerund phrases, naming the topic rather than the finding, siblings in parallel form, no questions.
25. **Results and discussion structure** of each research chapter.
26. **Balance of detail.** The share of words in methods, results and background per chapter. In B, methods text is about twice as long as results text.

## D. Language (by section type; short sentences are the author's preference, not a defect)

27. **Sentence length.** Mean and 90th percentile; the author's ceiling is 30 words (references average 23 to 31). Flag fragments under 8 words, and runs of short sentences that should be one sentence.
28. **Words per paragraph:** references 93 to 178 by section type. Flag fragmented paragraphs in results and discussion.
29. **Dependent clauses per sentence:** references 0.6 to 1.0. Read together with paragraph length to find choppy text.
30. **Lexical density:** references 59 to 64 per cent content words. Prepositions run 13 to 14 per cent and articles 11 to 12; above that points to "the X of the Y of the Z" chains.
31. **Nominalizations** (-tion, -ment, -ity nouns): references 3.6 to 6.9 per 100 words by section type.
32. **Passive voice:** references 15 to 46 per cent by section type, highest in methods. Active "we" is acceptable.
33. **Tense by context.**
    - Past tense: our experiments and what they found.
    - Present tense: what a table or figure shows, established facts, and what a chapter does.
    - Literature: as the references do.
34. **Hedges and boosters** (Hyland's lists), by section type.
35. **Signposting.** Transitions, frame markers ("first", "in this section") and code glosses ("for example"). Sentences that only link to the next part: references 1 to 6 per cent.
36. **AI-sounding phrasing:** hits from `~/Documents/ai_phrasing.rtf` and the excess-vocabulary list of Kobak et al. (2025); none in AI-written text.
37. **Jargon.** Acronyms per 100 words, every acronym defined at first use, and only terms the author uses.
38. **Readability** (Flesch–Kincaid grade, Gunning fog): descriptive only.
39. **Density of content.** One idea per paragraph; no sentence that only restates the previous one.

## E. Captions and referencing

40. **Captions.**
    - Length: references average 50 to 74 words; the thesis averages 46.
    - Form: a title phrase, then sentences.
    - Content: metric, unit and direction (higher or lower is better), and what one row or point is.
    - Abbreviations are left unexpanded, as the references do.
    - Interpretation: B puts a finding in 57 of 92 captions.
41. **Referencing.** "Figure x" and "Table x" written in full; every float cited; Additional files referred to in one consistent form.

## F. Chemistry

42. **Every item in "Problems found"** in `CHEMISTRY_LANGUAGE_GUIDE.md`.
43. **Word choice:** molecule, compound and ligand used as the references do.
44. **Typography:** $K_i$, p$K_i$, IC$_{50}$, LogD, units.
45. **Definitions at first use:** what each endpoint measures (hERG, Caco-2, LogD, MDR1–MDCK, clearance), and the Bemis–Murcko scaffold.
46. **Results tied back to chemistry** where the references do it: substructures, example molecules, and failures explained chemically.

## G. Front and end matter

47. **Section order** and presence: abstract, AI statement, acknowledgments, contents, lists, abbreviations, contributions.
48. **Acknowledgments:** the categories of people the references thank, against the author's.
49. **Title page elements.**
50. **AI-use statement:** consistent with what the source shows.

## H. Statistics

51. **Results report spread.** Per-fold scores, ranges and the number of folds or seeds are stated. Controls are named. No significance tests outside Chapter 3. The references report less than this, so the thesis can exceed them here.
