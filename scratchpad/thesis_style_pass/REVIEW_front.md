# Review: front matter (main.tex) and Summary of Research Contributions (thesis.tex)

Checked on 2026-10-10 against main.tex.before_style_apply_2026-10-10 and thesis.tex.before_style_apply_2026-10-10 (diff), and against the current files.

What changed in this pass: the abstract was rewritten (AI-marked); the acknowledgments moved from after the AI statement to before the Contents; the abstract moved after the Contents; the abbreviation list was rebuilt (with an "edited your text" note). The Summary of Research Contributions did not change in this pass. The Statement on the Use of Generative AI did not change in this pass: her TODO "review this and make it way less complex" is still open (APPLIED_front_end_matter.md records E2 as skipped).

her_text_check.py: two hits, both from the acknowledgments move (the \tableofcontents line now sits after them). The move has no "edited your text" note. V1 adds it.

Numbers: the planned NEW text adds no number. "13", "six", "seven", "three" and "five" in the abstract are already in thesis.tex (Chapter 1 outline and Chapter 3).

\ref check: every \ref in main.tex and in the Summary resolves (chap:robustness, chap:flaws, chap:method, chap:similarity, chap:noise all exist).

Captions, transitions, Results prefixes, Discussion sections: none in this unit.

---

### V1 — acknowledgments moved without a note
- file: main.tex
- kind: author-text (move)
- todo_line: % TODO: manual review -- edited your text: the Acknowledgments chapter moved, unchanged, from after the Statement on the Use of Generative AI to before the Table of Contents, when the abstract moved to straight after the Contents (University regulation).
- why: Moving her text needs an "edited your text" note (task rule). her_text_check.py flags it.
OLD:
```latex
\chapter*{Acknowledgments}
\addcontentsline{toc}{chapter}{Acknowledgments} % delete if it shouldn't appear in the contents
```
NEW:
```latex
% TODO: manual review -- edited your text: the Acknowledgments chapter moved, unchanged, from after the Statement on the Use of Generative AI to before the Table of Contents, when the abstract moved to straight after the Contents (University regulation).
\chapter*{Acknowledgments}
\addcontentsline{toc}{chapter}{Acknowledgments} % delete if it shouldn't appear in the contents
```

### V2 — abstract, paragraph 2 (Chapter 3)
- file: main.tex
- kind: ai-rewrite (under the existing "% TODO: manual review" lines above the abstract; keep them)
- todo_line: (none new; the abstract's two existing TODO lines cover it)
- why: "Model family decided noise robustness more than the representation did" drops the other half of the Chapter 3 finding (the representation mattered more for predictive performance; Chapter 1 outline and Chapter 3 Conclusions). "barely mattered" is loose; Chapter 3 says "made little difference". "Predicted uncertainty rose with noise" overstates: Chapter 3 says mean uncertainty rose for almost every probabilistic model on QM9 and most on the assay datasets. "point to" replaced with Chapter 3's own "identified". Datasets added so the reader knows what was tested.
OLD:
```latex
First, we introduced NoiseInject, an open-source Python package for benchmarking noise robustness. With it we compared 13 base models and six molecular representations under seven kinds of label noise. Model family decided noise robustness more than the representation did, and random forests were the most consistent model. The shape of the noise distribution barely mattered. Systematic noise did the most damage, for example an offset shared within a scaffold family or labels clipped at an assay limit. Predicted uncertainty rose with noise, but outside censoring it did not point to the corrupted labels.
```
NEW:
```latex
First, we introduced NoiseInject, an open-source Python package for benchmarking noise robustness. With it we compared 13 base models and six molecular representations under seven kinds of label noise, on QM9 and three assay datasets. Model family was the main factor in noise robustness, and the representation mattered more for predictive performance. Random forests were the most consistent model. The shape of the noise distribution made little difference. Systematic noise did the most damage, for example an offset shared within a scaffold family or labels clipped at an assay limit. Mean predicted uncertainty rose with noise for most probabilistic models. Outside censoring, no model's uncertainty for a single molecule identified the corrupted labels.
```

### V3 — abstract, paragraph 3 (Chapter 4)
- file: main.tex
- kind: ai-rewrite
- todo_line: (none new)
- why: The paragraph opens on the exploratory study and puts "Next" in its third sentence, so the "First, Next, Last" order of the abstract breaks inside a paragraph. "we traced this failure to the representation" is a claim of narrative, not what was computed. The last sentence leaves out why the features matter: Chapter 1 (line 25) says together they dominate the distance.
OLD:
```latex
In an exploratory study carried out earlier in this project, published methods for learning with noisy labels did poorly on molecular data. Those that compared molecules by a distance did worst. Next, we traced this failure to the representation. On a single dataset, most features of a standard representation do not vary or do not relate to the label, yet each adds to the distance between two molecules.
```
NEW:
```latex
Next, we asked why published methods for learning with noisy labels did poorly on molecular data in an exploratory study carried out earlier in this project. The methods that compared molecules by a distance did worst. On a single dataset, most features of a standard representation do not vary or do not relate to the label. Each of them still adds to the distance between two molecules, so together they dominate it.
```

### V4 — abstract, paragraph 4 (Chapters 5 and 6)
- file: main.tex
- kind: ai-rewrite
- todo_line: (none new)
- why: "under a fixed budget" and "the selection settings" are words the reader has not been given; Chapter 1 says the search "selects the most important features for that task by a fixed budget" and customizes the feature selection settings. "a hybrid built for distance gave better nearest neighbors" claims more than Chapter 6 found: better on all five small-molecule datasets, worse on several antibody and reaction datasets (Chapter 1 outline). "The search also runs on reactions" says nothing about the result; Chapter 8 says reaction results were mixed. The antibody result is missing; Chapter 1 says the gains depended on the dataset. "for which we built new representations" is ambiguous between reactions and antibodies; Chapter 8 lists the new representations as antibody ones. Chapter 6 gets its own sentences so the abstract keeps the chapter order of Chapter 1.
OLD:
```latex
We then built the adaptive hybrid representation. It selects features from several representations under a fixed budget, and searches the selection settings for each dataset and model. On every small-molecule dataset it predicted better than the best single representation, and a hybrid built for distance gave better nearest neighbors. The search also runs on reactions and on antibodies, for which we built new representations. A newly published embedding can be added as one more representation to search.
```
NEW:
```latex
We then built the adaptive hybrid representation. It selects a fixed number of features from several representations, and a search sets how they are selected for each dataset and model. On every small-molecule dataset the hybrid predicted better than the best single representation. On antibodies the gains depended on the dataset, and we built new antibody representations of the regions and mutations that vary. On reactions the results were mixed. A newly published embedding can be added as one more representation to search. We also built hybrids for predicting each molecule's label from its five nearest neighbors. On all five small-molecule datasets these neighbors predicted better than neighbors in standard representations, but on several antibody and reaction datasets they did worse.
```

### V5 — abstract, paragraph 5 (Chapter 7)
- file: main.tex
- kind: ai-rewrite
- todo_line: (none new)
- why: "Label noise was handled, but ..." is vague and reads as a contrast for effect. "the task, the data" repeat each other. "when the data flagged the wrong labels" is unclear: Chapter 7 (thesis.tex line 4794) says it worked best when something other than a model said which labels were wrong: the assay limit for clipped labels, the scaffold for a shifted family. The controls are named as Chapter 1 names them. "every cleaning" changed to Chapter 7's "every treatment".
OLD:
```latex
Last, we returned to label noise. We tested noise detection, noise mitigation and noise-robust training on single representations and hybrids side by side, against random and shuffled controls. Label noise was handled, but which method worked depended on the task, the data, the model and the type of noise. Corrections worked best when the data flagged the wrong labels, as an assay limit or a scaffold family does. On small molecules the hybrid kept its lead through every cleaning, and the ranking of noise methods changed with the feature space.
```
NEW:
```latex
Last, we returned to label noise. We tested noise detection, noise mitigation and noise-robust training on single representations and hybrids side by side. QM9 with no noise added, random deletion and shuffled detector scores served as controls. Which method helped depended on the dataset, the model and the kind of noise. Correcting labels worked best when it was known which labels were wrong, as for labels clipped at an assay limit or a shifted scaffold family. On small molecules the hybrid stayed ahead of the single representation through every treatment. The ranking of noise methods changed with the features they ran on.
```

### V6 — abstract, last paragraph (conclusion)
- file: main.tex
- kind: ai-rewrite
- todo_line: (none new)
- why: "Label noise in molecular data therefore cannot be treated as Gaussian" contradicts the abstract's own second paragraph and Chapter 3 (line 980, 1279): the shape of the noise distribution made little difference. What did matter was systematic against random error (Chapter 3 Conclusions; Chapter 7 Conclusions). The same sentence sits in Chapter 8's Summary of findings (third question); that is the Chapter 8 unit's to fix. "therefore" makes a claim the abstract has not built; the NEW text gives the reason for each recommendation.
OLD:
```latex
Label noise in molecular data therefore cannot be treated as Gaussian, and the features a model is given change which noise method works. No single model, representation or noise method was best on every dataset we tested. We recommend choosing a noise method by how the labels were made, and adapting the features to each dataset.
```
NEW:
```latex
Systematic noise did the most damage, and it was also corrected best when its source was known. We therefore recommend choosing a noise method by how the labels were made. No single model, representation or noise method was best on every dataset we tested, so we also recommend adapting the features to each dataset.
```

### V7 — AI statement, paragraph 1 (her TODO asks for it)
- file: main.tex
- kind: author-text (her TODO "review this and make it way less complex")
- todo_line: % TODO: manual review -- edited your text: as your TODO "review this and make it way less complex" asked, the 55-word sentence split in two and "generative AI tools in the preparation of" cut. Same three policies.
- why: Her TODO above the chapter asks for this and was skipped in this pass. One sentence of 55 words.
OLD:
```latex
I declare that my use of generative AI tools in the preparation of this thesis complies with the University of Oxford's policy on the use of generative AI in research and summative assessment, the Mathematical, Physical and Life Sciences Division's guidance on generative AI use for postgraduate research students, and the guidance of the Department of Statistics.
```
NEW:
```latex
% TODO: manual review -- edited your text: as your TODO "review this and make it way less complex" asked, the 55-word sentence split in two and "generative AI tools in the preparation of" cut. Same three policies.
I declare that my use of generative AI in this thesis complies with the University of Oxford's policy on generative AI in research and summative assessment. It also complies with the guidance of the Mathematical, Physical and Life Sciences Division for postgraduate research students and of the Department of Statistics.
```

### V8 — AI statement, paragraph 2
- file: main.tex
- kind: author-text (her TODO)
- todo_line: % TODO: manual review -- edited your text: as your TODO asked, "methodology" to "methods" and "otherwise indicated by citation or co-authorship statement" to "a citation or the Statement of Contribution says otherwise" (the page that names your co-authors). The second sentence is unchanged; see the note in "For the author".
- why: Her TODO. The co-authorship statement it refers to is now titled Statement of Contribution.
OLD:
```latex
The research questions, methodology, experimental design, interpretation of results, and conclusions in this thesis are my own, except where otherwise indicated by citation or co-authorship statement. All written text in this thesis is my own.
```
NEW:
```latex
% TODO: manual review -- edited your text: as your TODO asked, "methodology" to "methods" and "otherwise indicated by citation or co-authorship statement" to "a citation or the Statement of Contribution says otherwise" (the page that names your co-authors). The second sentence is unchanged; see the note in "For the author".
The research questions, methods, experimental design, interpretation of results and conclusions are my own, except where a citation or the Statement of Contribution says otherwise. All written text in this thesis is my own.
```

### V9 — AI statement, research ideas (also a factual error)
- file: main.tex
- kind: author-text (her TODO; factual fix)
- todo_line: % TODO: manual review -- edited your text: as your TODO asked, shorter wording. Also a factual fix: "the cliff-aware noise mitigation approach" is a detector in Chapter 7 (the cliff-aware gap, a score for finding noisy labels), not a mitigation method, so it is named as Chapter 7 and Chapter 8's contributions list name it.
- why: Her TODO. Chapter 7's table of detectors (line 4111) and Chapter 8's contributions list call it "the cliff-aware gap", used to flag molecules; no mitigation method has that name. "originated with me", "informed my thinking", "core" are wordier than needed.
OLD:
```latex
I used Claude and ChatGPT as discussion partners to critique and expand on research ideas I had already formulated, for example by asking how a proposed method might fail or what related approaches existed. The core contributions of this thesis originated with me. These include the adaptive hybrid search methodology and the design of its search space (Chapter~\ref{chap:method}) and the cliff-aware noise mitigation approach (Chapter~\ref{chap:noise}). Generative AI outputs informed my thinking but were not adopted as research contributions.
```
NEW:
```latex
% TODO: manual review -- edited your text: as your TODO asked, shorter wording. Also a factual fix: "the cliff-aware noise mitigation approach" is a detector in Chapter 7 (the cliff-aware gap, a score for finding noisy labels), not a mitigation method, so it is named as Chapter 7 and Chapter 8's contributions list name it.
I used Claude and ChatGPT to discuss research ideas I had already formed, for example by asking how a proposed method might fail or what related methods existed. The contributions of this thesis are my own. They include the adaptive hybrid search and the design of its search space (Chapter~\ref{chap:method}), and the cliff-aware gap, a score for finding noisy labels (Chapter~\ref{chap:noise}). I did not adopt any AI output as a research contribution.
```

### V10 — AI statement, code
- file: main.tex
- kind: author-text (her TODO)
- todo_line: % TODO: manual review -- edited your text: as your TODO asked, shorter wording ("primarily" to "mostly", "underlying this work" cut, the review sentence shortened, the ChatGPT sentence made active). Same claims.
- why: Her TODO.
OLD:
```latex
I wrote the original Python and Rust data pipelines underlying this work by hand. I then used Claude, primarily through Claude Code, to extend these pipelines and to implement methods from my own specifications, including the adaptive hybrid search and the noise mitigation experiments. I also used it to write unit tests and to debug. I reviewed all AI-assisted code, checked its behavior against the methods described in this thesis, and tested it before use. I take full responsibility for the correctness of all code. ChatGPT was used for the same purposes before February 2026.
```
NEW:
```latex
% TODO: manual review -- edited your text: as your TODO asked, shorter wording ("primarily" to "mostly", "underlying this work" cut, the review sentence shortened, the ChatGPT sentence made active). Same claims.
I wrote the original Python and Rust data pipelines by hand. I then used Claude, mostly through Claude Code, to extend them and to implement methods from my own specifications, including the adaptive hybrid search and the noise mitigation experiments. I also used it to write unit tests and to debug. I reviewed and tested all AI-assisted code, and checked it against the methods described in this thesis. I take full responsibility for the correctness of all code. Before February 2026 I used ChatGPT for the same purposes.
```

### V11 — AI statement, figures
- file: main.tex
- kind: author-text (her TODO)
- todo_line: % TODO: manual review -- edited your text: as your TODO asked, shorter wording ("auditable", "without reservation", "the designs are entirely my own" cut; "generate, alter, or modify" to "generate or change"). Same claims.
- why: Her TODO.
OLD:
```latex
I used Claude to help write and refine plotting scripts. I did not use generative AI to create any figure directly from a prompt, or to generate, alter, or modify any data or numerical result. Every plot in this thesis was produced by auditable code run on the recorded experimental outputs, and that code is available to the examiners without reservation. All schematic diagrams were drawn by hand in Lucidchart. While designing some of them, I asked generative AI for feedback on layout and clarity; the designs are entirely my own.
```
NEW:
```latex
% TODO: manual review -- edited your text: as your TODO asked, shorter wording ("auditable", "without reservation", "the designs are entirely my own" cut; "generate, alter, or modify" to "generate or change"). Same claims.
I used Claude to help write plotting scripts. Every plot in this thesis was made by code run on the recorded experimental outputs, and that code is available to the examiners. I did not use generative AI to make a figure from a prompt, or to generate or change any data or numerical result. I drew all schematic diagrams by hand in Lucidchart. For some of them I asked generative AI for feedback on layout and clarity.
```

### V12 — AI statement, writing
- file: main.tex
- kind: author-text (her TODO)
- todo_line: % TODO: manual review -- edited your text: as your TODO asked, shorter wording ("structural scaffold" to "guide to structure", "entirely" cut). The sentence "No AI-generated text appears verbatim ..." is unchanged; see "For the author".
- why: Her TODO.
OLD:
```latex
I used generative AI to help structure my writing, such as proposing outlines for chapters and sections and identifying gaps in an argument. In some cases I asked for a draft passage to use as a structural scaffold, which I then rewrote entirely in my own words. No AI-generated text appears verbatim or closely paraphrased in this thesis. I also asked for feedback on the clarity of passages I had written.
```
NEW:
```latex
% TODO: manual review -- edited your text: as your TODO asked, shorter wording ("structural scaffold" to "guide to structure", "entirely" cut). The sentence "No AI-generated text appears verbatim ..." is unchanged; see "For the author".
I used generative AI to help structure my writing, for example to propose outlines for chapters and sections and to point out gaps in an argument. In some cases I asked for a draft passage as a guide to structure, and then rewrote it in my own words. No AI-generated text appears verbatim or closely paraphrased in this thesis. I also asked for comments on the clarity of passages I had written.
```

### V13 — AI statement, literature
- file: main.tex
- kind: author-text (her TODO)
- todo_line: % TODO: manual review -- edited your text: as your TODO asked, three sentences made two, in the active voice. Same claims.
- why: Her TODO.
OLD:
```latex
I used Claude and ChatGPT to suggest search terms and candidate references. Every reference cited in this thesis was checked against its original source before inclusion. AI-suggested references that could not be verified were discarded.
```
NEW:
```latex
% TODO: manual review -- edited your text: as your TODO asked, three sentences made two, in the active voice. Same claims.
I used Claude and ChatGPT to suggest search terms and candidate references. I checked every cited reference against its original source, and discarded any suggested reference I could not verify.
```

### V14 — AI statement, last paragraph (also an unsupported claim)
- file: main.tex
- kind: author-text (her TODO; factual fix)
- todo_line: % TODO: manual review -- edited your text: as your TODO asked, shorter wording. Cut "Where an AI interaction materially shaped a specific line of investigation, this is noted in the relevant chapter": no chapter of thesis.tex notes one (searched for Claude, ChatGPT and generative AI outside comments on 2026-10-10). Put it back if you add such a note.
- why: Her TODO, and the cut sentence points to notes that do not exist.
OLD:
```latex
Because generative AI was not used to produce text, data, or figures in this thesis, its use is not cited in the main text as a source. Where an AI interaction materially shaped a specific line of investigation, this is noted in the relevant chapter. All code used to produce the results and figures is available at \url{https://github.com/adpunt/noise_inject} and \url{https://github.com/adpunt/KIRBy}, which record the development history of the codebases.
```
NEW:
```latex
% TODO: manual review -- edited your text: as your TODO asked, shorter wording. Cut "Where an AI interaction materially shaped a specific line of investigation, this is noted in the relevant chapter": no chapter of thesis.tex notes one (searched for Claude, ChatGPT and generative AI outside comments on 2026-10-10). Put it back if you add such a note.
Generative AI did not produce text, data or figures in this thesis, so it is not cited as a source in the main text. All code used to produce the results and figures is available at \url{https://github.com/adpunt/noise_inject} and \url{https://github.com/adpunt/KIRBy}, with the full development history of each codebase.
```

### V15 — acknowledgments: OPIG chapter range
- file: main.tex
- kind: ai-rewrite (under its existing "% TODO: manual review" line)
- todo_line: (none new)
- why: The Collaborations section of Chapter 1 (thesis.tex line 138) says OPIG pointed to the antibody and protein datasets used in Chapters 5 to 7. The added sentence says Chapters 5 and 6. Antibody datasets are used in Chapter 7 too (Chapter 8 contributions: "three antibody sets").
OLD:
```latex
I thank the Oxford Protein Informatics Group for pointing me to the antibody and protein datasets used in Chapters~\ref{chap:method} and~\ref{chap:similarity}. In particular, I thank [OPIG MEMBERS' NAMES] for [WHAT THEY DID].
```
NEW:
```latex
I thank the Oxford Protein Informatics Group for pointing me to the antibody and protein datasets used in Chapters~\ref{chap:method} to~\ref{chap:noise}. In particular, I thank [OPIG MEMBERS' NAMES] for [WHAT THEY DID].
```

### V16 — abbreviations: GCN wrongly commented out
- file: main.tex
- kind: author-text (revert part of an AI edit to her list)
- todo_line: replace the existing note's text, as below
- why: The note says GCN is never used. Chapter 2 (thesis.tex line 219) writes "Graph Convolutional Networks (GCNs)" and uses GCNs five times. GIN, used beside it, stays in the list.
OLD:
```latex
% TODO: manual review -- edited your text: rebuilt the list against thesis.tex on 2026-10-10 (added abbreviations the text uses, commented out ADME, CP, GCN, ICC, NDS, OHE and RFE, which it never uses, and changed ROC-AUC to AUROC, the form the text uses).
```
NEW:
```latex
% TODO: manual review -- edited your text: rebuilt the list against thesis.tex on 2026-10-10 (added abbreviations the text uses, commented out ADME, CP, ICC, NDS, OHE and RFE, which it never uses, and changed ROC-AUC to AUROC, the form the text uses).
```
and
OLD:
```latex
    % \item[GCN] Graph convolutional network
```
NEW:
```latex
    \item[GCN] Graph convolutional network
```

### V17 — Summary: added entries carry no review marker
- file: thesis.tex
- kind: marker only (AI-added text, 2026-10-10, before this pass)
- todo_line: % TODO: manual review
- why: The NoiseInject paper entry and the Software section were written by an earlier session ("[2026-10-10] ... added") but have no "% TODO: manual review" line, so they read as hers. Task rule: new paragraphs get the marker. Text unchanged.
OLD:
```latex
% [2026-10-10] NoiseInject entry added; the author says it goes to preprint shortly.
\textbf{In preparation:} Punt, A., Hanser, T.,
```
NEW:
```latex
% [2026-10-10] NoiseInject entry added; the author says it goes to preprint shortly.
% TODO: manual review
\textbf{In preparation:} Punt, A., Hanser, T.,
```
and
OLD:
```latex
% [2026-10-10] added: the two public codebases.
\section*{Software}
```
NEW:
```latex
% [2026-10-10] added: the two public codebases.
% TODO: manual review
\section*{Software}
```

### V18 — Summary: missing full stop after the second paper's title
- file: thesis.tex
- kind: author-text
- todo_line: % TODO: manual review -- edited your text: full stop added after the paper title, before the line break, as in the NoiseInject entry above.
- why: The title runs into "\\ A draft ..." with no stop; the entry above ends its title with a full stop.
OLD:
```latex
hybrid representations in multi-modal contexts \\ A draft is in preparation
```
NEW:
```latex
hybrid representations in multi-modal contexts. \\ A draft is in preparation
```
(Put the todo_line directly above the line that starts "\textbf{In preparation:} Punt, A., Morris, G.,", below the existing "% DECISION FOR ADELAIDE" comment.)

---

## Chemistry wording (CHEMISTRY_LANGUAGE_GUIDE.md)

No problem from the guide in this unit after V2 to V6. The abstract uses "label" for QM9, LogD and Caco-2 (not "activity"), "scaffold family", "assay limit". The abbreviation list has IC$_{50}$ and EC$_{50}$ roman, as the guide asks.

## Style guide gap list, items in this unit

- 1, 2, 3, 6, 7, 9, 11, 12, 13, 15, 16: done (13 except GCN, V16).
- 4: done except the AI statement's own TODO (V7 to V14).
- 5: open, needs her (below).
- 8: done; Chapter 3's note and the Statement of Contribution both say "to be submitted to the Journal of Cheminformatics".
- 10: done (her decision: declaration and signature removed).
- 14: TODO kept with a note; she decides (below).

## For the author

- The AI statement says "All written text in this thesis is my own" and "No AI-generated text appears verbatim or closely paraphrased". The source marks the abstract and many paragraphs as written by Claude ("% TODO: manual review", "[CLAUDE 2026-10-10]"). Either those are rewritten by you before submission, or these two sentences change. Gap item 5.
- Order: Acknowledgments now print before the Contents and the Statement on the Use of Generative AI after the abstract. The comment above the AI statement says MPLS wants it "before acknowledgements / table of contents"; that rule was never checked against the MPLS page. Check it and say which order.
- The abstract's first TODO line (2026-10-09) still mentions "about 300 words" and "76% ... on LogD"; neither is true of the current abstract (about 450 words, no numbers). It is a comment, so it was kept.
- Chapter 8's Summary of findings has the same sentence V6 removes from the abstract ("Label noise in molecular data therefore cannot be treated as Gaussian"), straight after "The shape of the noise distribution barely changed the damage". The Chapter 8 unit should fix it the same way.
- Summary of Research Contributions: one planned hybrid paper covering Chapters 5 and 6, but the Statement of Contribution says Chapters 5 to 7 go into one or two manuscripts. Say which, and whether Chapter 7 is in it.
- Author order differs between the two papers (Hanser, Werner, Morris against Morris, Werner, Hanser). Intended?
- Front and end matter headings ("Statement of Contribution", "Research Papers", "Conference Presentations", "Summary of Research Contributions") are in title case, as in Dablander's front matter; the chapters now use sentence case. Left as they are.
- "List of Publications" TODO at the end of the abbreviations: the Summary of Research Contributions already does this. Delete the TODO or keep both.
- Acknowledgments placeholders: Lhasa colleagues, OPIG members, CDT directors and cohort, transfer of status examiners, what St Catherine's provided, your grandmother's name.
- Software entry could cite the Zenodo archive of NoiseInject (thesis.tex line 531, \citep{noiseinject}). Not added: it would add a citation to your list.

## Applied

- V1: applied (main.tex, note above \chapter*{Acknowledgments}).
- V2: skipped, already applied: the abstract's Chapter 3 paragraph had been rewritten since this review (datasets named, "made little difference", representation mattered more for predictive performance, uncertainty claim narrowed).
- V3: skipped, already applied: the Chapter 4 paragraph now opens with "Next" and ends "so together they dominate it".
- V4: applied, adapted: sentence 2 replaced as planned ("a fixed number of features ... a search sets how they are selected"); "it predicted" to "the hybrid predicted"; the antibody and reaction sentences replaced by "...depended on the dataset, and we built new antibody representations of the regions and mutations that vary. On reactions the results were mixed." The distance sentence, already limited to small-molecule datasets, kept.
- V5: applied, adapted: the controls and known-flag sentences had already been rewritten; changed "the task, the data, the model and the type of noise" to "the dataset, the model and the kind of noise", and split the last sentence ("...through every treatment. The ranking of noise methods changed with the features they ran on.").
- V6: skipped, already fixed: the "cannot be treated as Gaussian" sentence and "therefore" are gone; the paragraph now says systematic noise did more damage than Gaussian noise.
- V7: applied.
- V8: applied; TODO line's last clause says the conflict with AI-marked paragraphs is hers to decide (instead of pointing to "For the author").
- V9: applied; verified "cliff-aware gap" is a detector in Chapter 7 (thesis.tex 4207, 4437).
- V10: applied.
- V11: applied.
- V12: applied; TODO line's last clause adjusted as in V8.
- V13: applied.
- V14: applied; re-checked: no uncommented Claude/ChatGPT/generative AI mention in thesis.tex.
- V15: applied; verified against Chapter 1 line 141 ("Chapters 5 to 7").
- V16: applied (both parts); GCN used in Chapter 2 (thesis.tex 222).
- V17: applied (both markers).
- V18: applied.
