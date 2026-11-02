# Reference style notes: captions, cross-references, headings, prose

Sources. Three reports, each written by reading the PDFs in `/Users/apunt/Documents/ReferencePapers/`:

- **Dablander**: Markus Dablander, DPhil thesis (Oxford, submitted Oct 2023), `Theses/DPhil_Thesis_Markus_Dablander--Submitted.pdf`. 71 of 189 pages were viewed. Page numbers below are printed page numbers, which equal the PDF page minus 14.
- **Nature MI**: nine Nature Machine Intelligence papers in `NatureML Papers/`. These are Boulougouri, Fang (GEM), Ektefaie (SPECTRA), Boiko (SIMG), Dong & Rudin, Zeng (ImageMol), Steyaert, Mataraso (COMET) and de Almeida (ChatNT). Page numbers are PDF page indices.
- **Other journals**: Venkatraman 2021, Kolmar & Grulke 2021 and Dablander et al. 2023 (all J Cheminform), plus Tuor, Heid, Cui 2026 (chemRxiv) and Marks 2021 (Bioinformatics).

Rule for conflicts. Dablander is a thesis from the same division. Where he differs from the journal papers, follow Dablander. Each section below marks these cases with **Differs**.

---

## 1. Figure captions

Rules:

- The caption goes below the figure as a full-width paragraph, in a smaller font than the body.
- The label is bold and ends in a colon, e.g. "Figure 2.1:".
- The caption opens with a noun phrase, not a sentence. Typical openings are "Schematic overview of…", "Example of…", "Illustration of…", "Overview of…" and "Predictive performance of…". Some openings have no verb at all. Full sentences can follow the opening phrase.
- A simple figure gets one line. A results figure gets 4-7 lines. The length follows the number of panels, not how important the figure is.
- A results caption says what the axes and encodings are, what the error bars mean and how many models or repeats lie behind each point. It gives the error bar definition in full, with the count.
- A caption does not say which method won. The only interpretation allowed is a reading guide for the plot, e.g. "the lower right corner corresponds to strong performance".
- A caption says why any data are missing.
- Abbreviations are expanded again inside each caption, even when the text has already defined them.
- Colours are named in parentheses and tied to what they encode, e.g. "(indicated by colours)".
- Captions of sibling figures repeat each other almost word for word. Only the dataset changes.
- Dablander sets the dataset or target in bold in the caption, e.g. "results for **dopamine receptor D2**".
- An image taken from elsewhere is credited at the end of the caption: "Image source: [59]."
- Units go in square brackets on axes, e.g. "MAE [logD]" or "K_i [nM]". Symbols in captions are typeset as maths.
- The List of Figures uses a separate short title of about 3-9 words. It is a noun phrase that ends in a full stop.

**Differs.**

- *Label.* Nature MI writes "Fig. N |" and other journals write "Fig. N", "Fig. N." or "Figure N.". Follow Dablander and write "**Figure N.M:**".
- *Panels.* Nature MI and Cui label panels with lowercase letters inside the caption ("a, …", "(a) …"). Dablander puts capital letters A-D on the figure itself. His caption does not cite the letters. It walks through the figure in order ("First… Then… Finally…") or refers to rows and columns. Follow Dablander.
- *Claim titles.* 3 of 9 Nature MI papers, and Cui and Marks, sometimes give a results figure a title that states the finding. Dablander never does this. Follow Dablander and describe the figure without stating the result.

Examples:

- "Figure 2.2: Circular subgraphs of varying radii for a central nitrogen atom in an example molecule." (Dablander, p. 23)
- "Figure 2.3: Schematic overview of the molecular-featurisation mechanism of a message-passing graph neural network (GNN) with radius R = 2. All depicted functions may contain trainable deep-learning components." (Dablander, p. 31)
- "Figure 3.7: QSAR-prediction and activity-cliff (AC) classification results for dopamine receptor D2. For each plot, the x-axis corresponds to a combination of MMP-set and AC-classification performance metric and the y-axis shows the QSAR-prediction performance on the molecular test set D_test. The total length of each error bar equals twice the standard deviation of the performance metric measured over all mk = 3*2 = 6 hyperparameter-optimised models. For each plot, the lower right corner corresponds to strong performance at both prediction tasks." (Dablander, p. 66)
- "Figure 5.1: Predictive performance of the four substructure-pooling methods (indicated by colours) for the lipophilicity regression data set using varying data splitting techniques, prediction models and ECFP hyperparameters. Each coloured bar shows the average mean absolute error (MAE) of the respective model across a k-fold cross validation scheme repeated with m random seeds for (m, k) = (3, 2)." (Dablander, p. 131)

---

## 2. Table captions

Rules:

- The label is bold, e.g. "Table 2.1:", and uses the same small font as figure captions.
- The caption is a short noun phrase, usually 1-2 lines, that names the contents. A longer caption adds where the entries come from and what they are.
- The caption expands every abbreviation used in the table.
- The caption never states a result.
- Inside the table, rows are single-spaced and the font is smaller than the body text.
- Column headers are bold and in title case, e.g. "Atom Feature | Encoding | Dimensions".
- Cell entries are lower case unless they are proper names.
- Tables use horizontal rules only: a top rule, a rule under the header and a bottom rule. They have no vertical lines. This comes from the journal reports and does not conflict with Dablander.
- Dablander sometimes adds a bold title row in title case inside the table frame, e.g. "Selected RDKit Physicochemical Descriptors".
- The List of Tables uses a short title that ends in a full stop.

**Differs.**

- *Position.* All the journals put the table caption above the table. Dablander puts it below, except for one table (Table 4.1, p. 96). Follow Dablander and put it below.
- *Notes.* The journals put their explanations in a footnote under the table. These cover what bold means, mean ± s.d., the number of runs, the direction of each metric and abbreviations, and use lettered footnotes (a, b) for individual cells. Dablander puts this material in the caption. Follow Dablander. A lettered footnote for a note on one cell does not conflict with him.
- *Label punctuation.* Journals write "Table 1", "Table 1 |" or "Table 1.". Follow Dablander and write "**Table N.M:**".

Examples:

- "Table 2.1: Overview of the chosen atom and bond features for the molecular graphs used in our experiments." (Dablander, p. 11, below the table)
- "Table 3.1: Sizes of our curated data sets and their respective numbers of matched molecular pairs (MMPs), activity cliffs (ACs), half-activity-cliffs (half-ACs) and non-activity-cliffs (non-ACs)." (Dablander, p. 56)
- "Table 5.2: Selected hyperparameters for the two prediction models used in our substructure-pooling experiments: random forests (RFs) and multilayer perceptrons (MLPs)." (Dablander, p. 129)
- "Table 1 Datasets used in this work, with the number of molecules, endpoint, endpoint units, range, and reference for each" (Kolmar & Grulke 2021, p. 6; journal style, caption above the table)

---

## 3. Referring to figures and tables in the text

Rules:

- Write "Figure 2.1", "Table 2.1", "Section 2.5" and "Chapter 3" in full and capitalised.
- Write ranges in words, e.g. "Figures 3.7 to 3.12".
- Point to part of a figure by its row or column, e.g. "the top rows of Figures 3.10 to 3.12" or "the last row of Figure 4.2".
- Dablander usually makes the reference part of an ordinary sentence ("… is given in Table 2.1."). He uses parentheses only for section cross-references, e.g. "PDVs (2.4)".
- The text does not repeat the caption. It gives the interpretation and the numbers the caption leaves out. Values are raw and in the dataset's own metric, and a derived number such as a percentage comes after them.
- Every figure and table is cited in the text, near where it appears. A sentence can exist only to introduce a figure, e.g. "An illustration of our data splitting strategy is given in Fig. 2." (Dablander et al. 2023).
- Past tense describes what was done. Present tense describes what a figure or table shows.

**Differs.** The journals abbreviate to "Fig. 2a" or "(Figs. 3a,b)". Their most common pattern is a parenthetical citation at the end of a sentence that states a finding. Dablander writes "Figure" in full with no panel letters, and usually builds the reference into the sentence. Follow Dablander.

Examples:

- "The QSAR-prediction, AC-classification and PD-classification results for all three investigated data sets are depicted in Figures 3.7 to 3.12 below." (Dablander, p. 65)
- "as can be seen in the top rows of Figures 3.10 to 3.12" (Dablander, p. 73)
- "For instance, Figure 5.1 shows that replacing hashing with Sort & Slice when using a 1024-bit ECFP4 with an MLP on a random split of the lipophilicity data set leads to a rather remarkable relative MAE-improvement of 11.37%." (Dablander, p. 137)
- "a median MAE of about 1.045 … but a median MAE of about 0.998 … approximately 4.5%" (Dablander, p. 130; raw values first, derived percentage second)

---

## 4. Section and subsection titles

Rules:

- A title is a short noun phrase. It has no verb, is not a sentence and has no final punctuation. No title is a question.
- Most subsection titles are 1-4 words. The longest is about 7 words.
- Generic section names repeat in every chapter: "Overview", "Mathematical Description", "Critical View", "Experimental Methodology", "Performance Metrics", "Results and Discussion", "Conclusions".
- A colon appears only in a "Topic: Qualifier" pattern, and rarely. Chapter titles may use a colon for a subtitle.
- Method names in headings are capitalised as proper names, e.g. "Sort & Slice" or "Mutual-Information Maximisation".
- Inside a section, an unnumbered bold run-in heading can introduce a paragraph.
- The List of Abbreviations is an alphabetical bulleted list of the form "ABBR = Expanded Name". It gives expansions only. It has no architecture details, notes or definitions.

**Differs.**

- *Capitalisation.* The journals use sentence case, e.g. "Random error generation". Dablander uses title case, e.g. "Molecular Graphs" or "Performance Metrics". Follow Dablander and use title case.
- *Claim headings.* Some Nature MI Results sections use claim sentences as headings, e.g. "SPECTRA reveals generalization gaps in molecular ML models". Dablander never does. Follow Dablander and use noun phrases.

Examples:

- "2.4.2 Critical View"; "3.3.5 Performance Metrics" (Dablander, Contents)
- "3.4.4 Linear Relationship between QSAR-MAE and AC-MCC" (Dablander, Contents; about the longest)
- "2.7 Molecular Featurisations: Critical Overview" (Dablander, Contents; colon pattern)
- "• ECFP = Extended-Connectivity Fingerprint" (Dablander, List of Abbreviations, p. vi)

---

## 5. Prose

Method and condition names:

- Write method and condition names as capitalised proper names or plain English, never as code identifiers. Dablander writes "Sort & Slice", "Hash", "Filter" and "MIM". Model combinations are written "ECFP-MLP", "GIN-kNN" or "ECFP + kNN".
- A code-like string appears only when it is a dataset name taken from a source, e.g. "g298_atom" (Kolmar). Monospace is used only when naming a software package, e.g. RDKit or PyTorch Geometric.
- Library-style parameter names, e.g. "NumberOfTrees = 100", appear only inside a table. In the body the same setting is written in English, e.g. "a forest size of 500 trees" (Dablander, p. 64).
- Named ablation variants can be introduced in single quotes and defined in prose: "'Without pre-train' denotes the GeoGNN network without pre-training" (Fang, p. 4).

Abbreviations and naming:

- Define each abbreviation at first use in each chapter as "full term (ABBR)". Use it bare after that. Common model terms stay lowercase when spelled out, e.g. "random forests".
- Use one name for a representation throughout. Dablander uses ECFP everywhere and explains ECFP4 (radius R = 2) once, on p. 26. Heid uses "Morgan fingerprint" everywhere. No paper switches between ECFP and Morgan. Following Dablander means using ECFP.
- State the variant and size of a fingerprint where it is used, e.g. "1024-bit ECFP4".

Lists:

- Dablander uses bulleted lists only for research questions and for advantages and disadvantages. The pattern for the latter is "(+) **Bold title.** sentence". He uses numbered lists for algorithm steps.
- **Differs.** Nature MI puts no displayed bullet lists in the body text. It writes enumerations inline, e.g. "(1) early, (2) intermediate or (3) late". Follow Dablander, and keep displayed lists to those three uses. Write other enumerations as prose.

Sentences and colons:

- Sentences are full and formal, mostly 20-40 words, with subordinate clauses. No fragments or shorthand were seen in any source.
- Sentences under 6 words were almost absent: 0-4 per Nature MI paper, out of 180-555 sentences.
- Colons introduce a displayed equation, a list or an explanation, e.g. "via the mean absolute error (MAE):". They seldom split a sentence in running prose. Nature MI had about 1.4-6.5 colons per 1000 words.

Voice, numbers and hedging:

- Write in the first person plural ("we") and in the active voice.
- Give counted numbers in full, with units and what is counted: "In total we therefore trained 1920*6 = 11520 models, half of which were deep-learning models." (Dablander, p. 130).
- Report results with the metric name and the raw values.
- Mark interpretation as hedged: "This might be as a result of…", "one can speculate that…" (Dablander, pp. 72-74). Evaluative adjectives go only as far as the numbers support.

Spelling:

- Dablander uses British spelling, e.g. "featurisation" or "colour". The journals vary. These reports do not settle which spelling the thesis uses.

Things no source does:

- Refer to code identifiers, variable names or lowercase condition names in running text.
- Use sentence fragments or shorthand.
- Switch between two names for the same representation.
- Put architecture details or notes in the List of Abbreviations.
- State a winner in a caption (Dablander).
- Use a question or a full sentence as a section heading (Dablander).

Examples:

- "reduced the data set … from 8883 / 4116 / 1926 compounds to 6333 / 3605 / 1924 unique compounds" (Dablander, p. 51)
- "ECFP-featurisations with radius R are often written in the form ECFP2R … the frequently used 1024-bit ECFP4-featurisation describes an ECFP with radius R = 2 and length l = 1024" (Dablander, p. 26)
- "Ridge regression (Ridge)", "Support vector regressor (SVR)", "Random forest (RF)", "Gaussian process (GP)" (Kolmar & Grulke 2021, Table 2)

---

## Checklist, one caption at a time

Figures:

1. The label is "**Figure N.M:**" and the caption sits below the figure.
2. The caption opens with a noun phrase that says what is plotted and for which data.
3. The caption states no winner and no result. The only interpretation is a reading guide for the plot.
4. The axes, colours or markers, and panels are described by position (row, column or order), not by letter.
5. The caption says what the error bars or spread are, with the count of folds, repeats or models.
6. The caption says why any data are missing.
7. Every abbreviation in the figure is expanded again in the caption.
8. Units are given in square brackets, and symbols are typeset as maths.
9. The dataset name is in bold if the figure is one of a set of sibling figures, and the wording matches its siblings.
10. The caption uses no code identifiers or lowercase variable names. Conditions have plain English or proper names.
11. A short title (3-9 words, noun phrase, full stop) is set for the List of Figures.
12. The text cites the figure as "Figure N.M" in full, near the figure, and gives the numbers the caption omits.

Tables:

1. The label is "**Table N.M:**" and the caption sits below the table.
2. The caption is a noun phrase naming the contents and states no result.
3. The caption expands every abbreviation in the table and says what bold means, what the numbers are, and the metric direction if needed.
4. The table is single-spaced, in a smaller font than the body, with horizontal rules only.
5. Headers are bold and in title case. Cells are lower case unless they are proper names.
6. The table uses no code identifiers in headers or cells, unless the identifier is a dataset name from a source or a parameter name confined to the table.
7. A short title is set for the List of Tables.
8. The text cites the table as "Table N.M".
