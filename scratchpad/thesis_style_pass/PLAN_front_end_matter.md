# Plan: front and end matter (main.tex; thesis.tex acknowledgements/AI statement region and Summary of Research Contributions)

Checked against main.tex and thesis.tex on 2026-10-10. The acknowledgements and AI statement no longer sit in thesis.tex: an earlier session moved both into main.tex (comments "[2026-10-09] moved from the top of thesis.tex, unchanged"). thesis.tex now begins at \chapter{Introduction}, so every front-matter edit below is in main.tex.

## Order of front matter (no move needed)

main.tex already prints: title page; Statement of Contribution (renamed from Statement of Originality on 2026-10-10, declaration and signature lines removed); Abstract; Statement on the Use of Generative AI; Acknowledgments; Contents; List of Figures; List of Tables; List of Abbreviations; then the body; Summary of Research Contributions last in thesis.tex, before the bibliography. This is the order asked for (gap item 3), with the AI statement right after the abstract. Only one acknowledgments chapter exists, spelled "Acknowledgments" in both the heading and the contents line (gap item 1 done). The applier should keep this order when building the package.

## Edits

### E1 — Statement of Contribution: broken chapter reference
- file: main.tex
- kind: author-text
- why: Gap item 7: \ref{chap:why} has no matching label and prints "??". The Collaborations section of Chapter 1 puts Lhasa's hERG data in Chapters 4 and 5 and OPIG's datasets in Chapters 5 and 6, so the span is Chapters 4 to 6, not two chapters.
- todo_line: % TODO: manual review -- edited your text: the label chap:why does not exist and printed "??", so this now reads Chapters~\ref{chap:flaws} to~\ref{chap:similarity}, the span the Collaborations section in Chapter 1 gives for the Lhasa and OPIG work.
OLD:
```latex
This thesis includes work carried out in collaboration. Chapter~\ref{chap:robustness}
is adapted from a jointly authored manuscript, Punt, A., Hanser, T.,
Werner, S., and Morris, G.M., in preparation for the \textit{Journal of Cheminformatics}.
Chapters~\ref{chap:why}
and~\ref{chap:similarity} include work carried out with Lhasa Limited and with the Oxford Protein
Informatics Group.
```
NEW:
```latex
% TODO: manual review -- edited your text: the label chap:why does not exist and printed "??", so this now reads Chapters~\ref{chap:flaws} to~\ref{chap:similarity}, the span the Collaborations section in Chapter 1 gives for the Lhasa and OPIG work.
This thesis includes work carried out in collaboration. Chapter~\ref{chap:robustness}
is adapted from a jointly authored manuscript, Punt, A., Hanser, T.,
Werner, S., and Morris, G.M., in preparation for the \textit{Journal of Cheminformatics}.
Chapters~\ref{chap:flaws}
to~\ref{chap:similarity} include work carried out with Lhasa Limited and with the Oxford Protein
Informatics Group.
```

### E2 — AI-use statement: simplified
- file: main.tex
- kind: author-text
- why: Her own TODO "review this and make it way less complex"; gap item 4 (the two Chapter~[X] placeholders filled from thesis.tex; all other placeholders kept). Prose cut from about 570 to about 380 words; every claim kept, the commented-out "Any other tool" line kept.
- todo_line: % TODO: manual review -- simplified your AI statement; claims unchanged
OLD:
```latex
% TODO: review this and make it way less complex
\chapter*{Statement on the Use of Generative AI}
\addcontentsline{toc}{chapter}{Statement on the Use of Generative AI}

I declare that my use of generative AI tools in the preparation of this thesis complies with the University of Oxford's policy on the use of generative AI in research and summative assessment, the Mathematical, Physical and Life Sciences Division's guidance on generative AI use for postgraduate research students, and the guidance of the Department of Statistics.

The research questions, methodology, experimental design, interpretation of results, and conclusions in this thesis are my own, except where otherwise indicated by citation or co-authorship statement. All written text in this thesis is my own.

\subsection*{Tools used}

\begin{itemize}
  \item \textbf{Claude} (Anthropic), accessed through the Claude web application and Claude Code, [MODEL VERSIONS, e.g.\ Claude Sonnet 4, Claude Opus 4.x], [MONTH YEAR]--October 2026.
  \item \textbf{ChatGPT} (OpenAI), [MODEL VERSIONS, e.g.\ GPT-4o], [MONTH YEAR]--[MONTH YEAR].
  % \item [Any other tool, e.g. GitHub Copilot (GitHub/Microsoft) -- delete if unused]
\end{itemize}

\subsection*{How these tools were used}

\paragraph{Research ideas.}
I used Claude and ChatGPT as discussion partners to critique and expand on research ideas I had already formulated, for example by asking how a proposed method might fail or what related approaches existed. The core contributions of this thesis originated with me. These include the adaptive hybrid search methodology and the design of its search space (Chapter~[X]) and the cliff-aware noise mitigation approach (Chapter~[X]). Generative AI outputs informed my thinking but were not adopted as research contributions.

\paragraph{Code.}
I wrote the original Python and Rust data pipelines underlying this work by hand. I then used Claude, primarily through Claude Code, to extend these pipelines and to implement methods from my own specifications, including the adaptive hybrid search and the noise mitigation experiments. I also used it to write unit tests and to debug. I reviewed all AI-assisted code, checked its behavior against the methods described in this thesis, and tested it before use. I take full responsibility for the correctness of all code. [IF APPLICABLE: ChatGPT was used for the same purposes before [MONTH YEAR].]

\paragraph{Figures and diagrams.}
I used Claude to help write and refine plotting scripts. I did not use generative AI to create any figure directly from a prompt, or to generate, alter, or modify any data or numerical result. Every plot in this thesis was produced by auditable code run on the recorded experimental outputs, and that code is available to the examiners without reservation. All schematic diagrams were drawn by hand in Lucidchart. While designing some of them, I asked generative AI for feedback on layout and clarity; the designs are entirely my own.

\paragraph{Writing.}
I used generative AI to help structure my writing, such as proposing outlines for chapters and sections and identifying gaps in an argument. In some cases I asked for a draft passage to use as a structural scaffold, which I then rewrote entirely in my own words. No AI-generated text appears verbatim or closely paraphrased in this thesis. I also asked for feedback on the clarity of passages I had written.

\paragraph{Literature.}
I used Claude and ChatGPT to suggest search terms and candidate references. Every reference cited in this thesis was checked against its original source before inclusion. AI-suggested references that could not be verified were discarded.

\subsection*{Data protection and intellectual property}

[CONFIRM BEFORE SUBMISSION:] No confidential, commercially sensitive, or proprietary data, including data supplied by the industrial partner, Lhasa Limited, were entered into any third-party generative AI tool. [If any non-public material was shared, state what it was and under what agreement.]

\subsection*{Referencing of generative AI use in the text}

Because generative AI was not used to produce text, data, or figures in this thesis, its use is not cited in the main text as a source. Where an AI interaction materially shaped a specific line of investigation, this is noted in the relevant chapter. All code used to produce the results and figures is available at [REPOSITORY URL], which records the development history of the codebase.
```
NEW:
```latex
% TODO: review this and make it way less complex
% TODO: manual review -- simplified your AI statement; claims unchanged
% [2026-10-10] The two "Chapter~[X]" placeholders are filled from thesis.tex: the hybrid search is Chapter~\ref{chap:method}, and the cliff-aware gap is in Chapter~\ref{chap:noise}. Every other placeholder is kept.
\chapter*{Statement on the Use of Generative AI}
\addcontentsline{toc}{chapter}{Statement on the Use of Generative AI}

I declare that my use of generative AI complies with the University of Oxford's policy and with the guidance of the Mathematical, Physical and Life Sciences Division and the Department of Statistics.

The research questions, methods, experimental design, interpretation and conclusions are my own, except where cited or co-authored. All written text is my own.

\subsection*{Tools used}

\begin{itemize}
  \item \textbf{Claude} (Anthropic), accessed through the Claude web application and Claude Code, [MODEL VERSIONS, e.g.\ Claude Sonnet 4, Claude Opus 4.x], [MONTH YEAR]--October 2026.
  \item \textbf{ChatGPT} (OpenAI), [MODEL VERSIONS, e.g.\ GPT-4o], [MONTH YEAR]--[MONTH YEAR].
  % \item [Any other tool, e.g. GitHub Copilot (GitHub/Microsoft) -- delete if unused]
\end{itemize}

\subsection*{How these tools were used}

\paragraph{Research ideas.}
Claude and ChatGPT critiqued and expanded on my own ideas, for example how a method might fail or what related work existed. The core contributions are mine, including the adaptive hybrid search and its search space (Chapter~\ref{chap:method}) and the cliff-aware noise mitigation approach (Chapter~\ref{chap:noise}). AI output informed my thinking but was never adopted as a contribution.

\paragraph{Code.}
I wrote the original Python and Rust pipelines by hand. Claude, mostly through Claude Code, then extended them, implemented methods to my specifications, including the hybrid search and the noise mitigation experiments, wrote unit tests and helped debug. I reviewed and tested all AI-assisted code against the methods described here. I take full responsibility for the correctness of all code. [IF APPLICABLE: ChatGPT was used for the same purposes before [MONTH YEAR].]

\paragraph{Figures and diagrams.}
Claude helped write and refine plotting scripts. No figure was made from a prompt, and AI generated or changed no data or numerical result. Every plot comes from auditable code run on the recorded outputs, open to the examiners. I drew all schematic diagrams by hand in Lucidchart, sometimes asking AI for feedback on layout and clarity; the designs are my own.

\paragraph{Writing.}
AI proposed outlines, pointed out gaps in arguments and commented on the clarity of passages I wrote. Drafts I asked for served only as scaffolds and were rewritten entirely in my own words. No AI-generated text appears verbatim or closely paraphrased.

\paragraph{Literature.}
Claude and ChatGPT suggested search terms and references. I checked every cited reference against its source and discarded any I could not verify.

\subsection*{Data protection and intellectual property}

[CONFIRM BEFORE SUBMISSION:] No confidential, commercially sensitive or proprietary data, including Lhasa Limited's, went into any third-party AI tool. [If any non-public material was shared, state what it was and under what agreement.]

\subsection*{Referencing of generative AI use in the text}

AI produced no text, data or figures, so it is not cited as a source. Where an AI interaction materially shaped a line of investigation, the chapter says so. All code, with its full development history, is at [REPOSITORY URL].
```

### E3 — Acknowledgments: funding sentence merged in, fuller version added
- file: main.tex
- kind: new-paragraph
- why: Style guide Acknowledgements section ("What hers lacks" 1-7, "Suggested fuller version") and gap items 1-2: the EPSRC CDT sentence with EP/S024093/1 survives only in a comment; Lhasa's hERG data, OPIG's datasets, the CDT, examiners, college and friends are not thanked. Every sentence of hers is kept word for word and in order; one acknowledgments chapter, American spelling, already the case.
- todo_line: one "% TODO: manual review -- added sentence(s): ..." line above each added block (see NEW)
OLD:
```latex
I would also like to thank Thierry for his valuable input into my research and for always sharing his technical expertise so generously. I am especially grateful that he continued to attend our meetings and to help me with this thesis after he left Lhasa and stepped down from his formal role as my supervisor. That commitment went far beyond what was asked of him. I would also like to thank St\'{e}phane Werner at Lhasa for his support and for staying engaged with my work. I am grateful to Lhasa Limited \textbf{[and FUNDER, grant no.~XXX]} for their generous financial support.

I would like to thank Prof.~Charlotte~M.~Deane and the members of the Oxford Protein Informatics Group, past and present.
```
NEW:
```latex
I would also like to thank Thierry for his valuable input into my research and for always sharing his technical expertise so generously. I am especially grateful that he continued to attend our meetings and to help me with this thesis after he left Lhasa and stepped down from his formal role as my supervisor. That commitment went far beyond what was asked of him. I would also like to thank St\'{e}phane Werner at Lhasa for his support and for staying engaged with my work.
% TODO: manual review -- added sentence(s): Lhasa colleagues and the hERG data Lhasa supplied (Collaborations section, Chapter 1); fill or delete the placeholders.
I also thank [LHASA COLLEAGUES' NAMES] at Lhasa for [WHAT THEY DID], and Lhasa for the hERG data used in Chapters~\ref{chap:flaws} and~\ref{chap:method}.
I am grateful to Lhasa Limited \textbf{[and FUNDER, grant no.~XXX]} for their generous financial support.
% TODO: manual review -- added sentence(s): the funding sentence from the old acknowledgements stub (comment block above), with the grant number. Once you keep it, the "[and FUNDER, grant no.~XXX]" placeholder in the sentence before can go; fill one or the other, not both.
This research was supported by the University of Oxford's UK EPSRC Centre For Doctoral Training in Sustainable Approaches to Biomedical Science: Responsible and Reproducible Research (EP/S024093/1) and by the not-for-profit organization Lhasa Limited.

% TODO: manual review -- added sentence(s): thanks to the Centre for Doctoral Training and your cohort, as Dablander thanks his CDT directors; names unknown, so placeholders.
I thank the directors and staff of the Centre for Doctoral Training, [NAMES], for [WHAT THEY DID], and my cohort, [NAMES], for [WHAT THEY DID].

% TODO: manual review -- added sentence(s): thanks to your transfer of status examiners, as Buttenschoen does; names unknown, so placeholders.
I am grateful to my transfer of status examiners, [NAMES], for their time and their questions.

I would like to thank Prof.~Charlotte~M.~Deane and the members of the Oxford Protein Informatics Group, past and present.
% TODO: manual review -- added sentence(s): the group's help with datasets (Collaborations section, Chapter 1), then named members as placeholders.
I thank the group for pointing me to the antibody and protein datasets used in Chapters~\ref{chap:method} and~\ref{chap:similarity}. In particular, I thank [OPIG MEMBERS' NAMES] for [WHAT THEY DID].

% TODO: manual review -- added sentence(s): college (from the title page) and department; what they provided is yours to fill.
I thank St Catherine's College and the Department of Statistics for [WHAT THEY PROVIDED].

% TODO: manual review -- added sentence(s): friends, as all three reference theses do; names unknown, so placeholders. If "Finally" in the next paragraph now reads oddly, move this sentence after the last paragraph.
I thank my friends, [NAMES], for [WHAT THEY DID].
```

### E4 — List of Abbreviations rebuilt from the text
- file: main.tex
- kind: author-text
- why: Gap item 13: seven entries never used in thesis.tex (ADME, CP, GCN, ICC, NDS, OHE, RFE), ROC-AUC where the text writes AUROC (17 uses, ROC-AUC 0), and 36 abbreviations used in the text but missing. Each entry was counted in thesis.tex with comments stripped on 2026-10-10. Unused entries are commented out, not deleted.
- todo_line: % TODO: manual review -- edited your text: rebuilt the list against thesis.tex on 2026-10-10 (added abbreviations the text uses, commented out ADME, CP, GCN, ICC, NDS, OHE and RFE, which it never uses, and changed ROC-AUC to AUROC, the form the text uses).
OLD:
```latex
\begin{description}[style=unboxed,leftmargin=0cm]
    \item[ADME] Absorption, distribution, metabolism, and excretion
    \item[ANOVA] Analysis of variance
    \item[BNN] Bayesian neural network
    \item[CP] Conformal prediction
    \item[DFT] Density functional theory
    \item[ECE] Expected calibration error
    \item[ECFP] Extended-connectivity fingerprint
    \item[ECFP4] Extended-connectivity fingerprint with radius 2, diameter 4
    \item[F1] F1 score (harmonic mean of precision and recall)
    \item[GCN] Graph convolutional network
    \item[GIN] Graph isomorphism network
    \item[GNN] Graph neural network
    \item[GP] Gaussian process
    \item[HOMO] Highest occupied molecular orbital
    \item[ICC] Intraclass correlation coefficient
    \item[LightGBM] Light Gradient Boosting Machine
    \item[LNL] Learning with noisy labels (the problem of training on labels that are partly wrong, not a group of methods)
    \item[LUMO] Lowest unoccupied molecular orbital
    \item[MAE] Mean absolute error
    \item[MHG-GNN] Molecular Hypergraph Grammar Graph Neural Network
    \item[ML] Machine learning
    \item[MLP] Multi-layer perceptron
    \item[mRMR] Minimum redundancy, maximum relevance
    \item[NDS] Noise degradation slope
    \item[NGBoost] Natural gradient boosting
    \item[NN] Neural network
    \item[NN-$\alpha$] Neural network architecture $\alpha$ ([128, 64], dropout after each layer)
    \item[NN-$\beta$] Neural network architecture $\beta$ ([128, 128], dropout before output)
    \item[OHE] One-hot encoding
    \item[PCA] Principal component analysis
    \item[PDV] Physicochemical descriptor vector
    \item[QRF] Quantile regression forest
    \item[QSAR] Quantitative structure--activity relationship
    \item[QSPR] Quantitative structure--property relationship
    \item[RBF] Radial basis function
    \item[RF] Random forest
    \item[RFE] Recursive feature elimination
    \item[RMSE] Root mean squared error
    \item[ROC-AUC] Area under the receiver operating characteristic curve
    \item[SHAP] SHapley Additive exPlanations
    \item[SMILES] Simplified molecular input line entry system
    \item[SNS] Sort \& Slice
    \item[SVM] Support vector machine
    \item[VBLL] Variational Bayesian Last Layer
    \item[XGBoost] eXtreme Gradient Boosting
\end{description}
```
NEW:
```latex
% TODO: manual review -- edited your text: rebuilt the list against thesis.tex on 2026-10-10 (added abbreviations the text uses, commented out ADME, CP, GCN, ICC, NDS, OHE and RFE, which it never uses, and changed ROC-AUC to AUROC, the form the text uses).
\begin{description}[style=unboxed,leftmargin=0cm]
    \item[AC] Activity cliff
    \item[AC-SINS] Affinity-capture self-interaction nanoparticle spectroscopy
    % \item[ADME] Absorption, distribution, metabolism, and excretion
    \item[ANOVA] Analysis of variance
    \item[AUROC] Area under the receiver operating characteristic curve
    \item[BLOSUM] Blocks substitution matrix
    \item[BNN] Bayesian neural network
    \item[CDR] Complementarity-determining region
    \item[CGR] Condensed graph of reaction
    \item[CHO] Chinese hamster ovary
    \item[CL$_\text{int}$] Intrinsic clearance
    % \item[CP] Conformal prediction
    \item[CV] Cross-validation
    \item[DFT] Density functional theory
    \item[DNN] Deep neural network
    \item[EC$_{50}$] Half-maximal effective concentration
    \item[ECE] Expected calibration error
    \item[ECFP] Extended-connectivity fingerprint
    \item[ECFP4] Extended-connectivity fingerprint with radius 2, diameter 4
    \item[EGFR] Epidermal growth factor receptor
    \item[EM] Expectation maximization
    \item[F1] F1 score (harmonic mean of precision and recall)
    \item[FR] Framework region
    % \item[GCN] Graph convolutional network
    \item[GIN] Graph isomorphism network
    \item[GNN] Graph neural network
    \item[GP] Gaussian process
    \item[hERG] Human ether-\`a-go-go-related gene (a cardiac potassium channel)
    \item[HIC] Hydrophobic interaction chromatography
    \item[HLM] Human liver microsome
    \item[HOMO] Highest occupied molecular orbital
    \item[IC$_{50}$] Half-maximal inhibitory concentration
    % \item[ICC] Intraclass correlation coefficient
    \item[IMGT] The international ImMunoGeneTics information system
    \item[KL] Kullback--Leibler
    \item[kNN] $k$-nearest neighbors
    \item[LGBM] LightGBM (short form used in tables)
    \item[LightGBM] Light Gradient Boosting Machine
    \item[LM] Language model
    \item[LNL] Learning with noisy labels (the problem of training on labels that are partly wrong, not a group of methods)
    \item[LUMO] Lowest unoccupied molecular orbital
    \item[MACCS] Molecular ACCess System
    \item[MAE] Mean absolute error
    \item[MCC] Matthews correlation coefficient
    \item[MCS] Maximum common substructure
    \item[MDCK] Madin--Darby canine kidney
    \item[MDR1] Multidrug resistance protein 1 (P-glycoprotein)
    \item[MHG-GNN] Molecular Hypergraph Grammar Graph Neural Network
    \item[ML] Machine learning
    \item[MLP] Multi-layer perceptron
    \item[mRMR] Minimum redundancy, maximum relevance
    % \item[NDS] Noise degradation slope
    \item[NGBoost] Natural gradient boosting
    \item[NLL] Negative log-likelihood
    \item[NLP] Natural language processing
    \item[NN] Neural network
    \item[NN-$\alpha$] Neural network architecture $\alpha$ ([128, 64], dropout after each layer)
    \item[NN-$\beta$] Neural network architecture $\beta$ ([128, 128], dropout before output)
    % \item[OHE] One-hot encoding
    \item[OPIG] Oxford Protein Informatics Group
    \item[PCA] Principal component analysis
    \item[PDV] Physicochemical descriptor vector
    \item[PLS] Partial least squares
    \item[QRF] Quantile regression forest
    \item[QSAR] Quantitative structure--activity relationship
    \item[QSPR] Quantitative structure--property relationship
    \item[RBF] Radial basis function
    \item[RF] Random forest
    % \item[RFE] Recursive feature elimination
    \item[RLM] Rat liver microsome
    \item[RMSE] Root mean squared error
    \item[SALI] Structure--activity landscape index
    \item[SD] Standard deviation
    \item[SHAP] SHapley Additive exPlanations
    \item[SMAC] Standup monolayer adsorption chromatography
    \item[SMILES] Simplified molecular input line entry system
    \item[SNS] Sort \& Slice
    \item[SPP] Similar property principle
    \item[SVM] Support vector machine
    \item[USR] Ultrafast shape recognition
    \item[USRCAT] Ultrafast shape recognition with CREDO atom types
    \item[VAE] Variational autoencoder
    \item[VBLL] Variational Bayesian Last Layer
    \item[VEGF] Vascular endothelial growth factor
    \item[WL] Weisfeiler--Lehman
    \item[XGB] XGBoost (short form used in tables)
    \item[XGBoost] eXtreme Gradient Boosting
\end{description}
```

### E5 — List of Publications TODO: note that it duplicates the Summary of Research Contributions
- file: main.tex
- kind: new-paragraph
- why: Gap item 14. Her TODO is kept (comments are never deleted); a note is added below it so she can drop it.
- todo_line: none (comment only)
OLD:
```latex
% TODO(you): the outline also asks for a "List of Publications" page here.
```
NEW:
```latex
% TODO(you): the outline also asks for a "List of Publications" page here.
% [2026-10-10] Note: the Summary of Research Contributions at the end of thesis.tex already lists papers, talks and software, as in Dablander's thesis; keeping both would duplicate it (style guide gap item 14).
```

### E6 — Cross-references to Additional files resolve (xr-hyper)
- file: main.tex
- kind: new-paragraph
- why: Gap item 16: thesis.tex has 33 "Additional file~\ref" references, main.tex loads no xr package, and the PDF at scratchpad/thesis_code_todo/thesis.pdf (built 10 Oct 00:05) prints "Additional file ??" 30 times. The stage1 preamble already uses this same pair of lines. NEW PACKAGE: xr-hyper (see For the author).
- todo_line: % TODO: manual review -- added xr-hyper (must load before hyperref) and \externaldocument, so that "Additional file~\ref{app:...}" reads its number from the Additional files' .aux instead of printing "??". The .aux path is the only Additional files build that exists today; change it if that document moves.
OLD:
```latex
\usepackage[colorlinks=true, allcolors=black]{hyperref} % [2026-10-10] links stay clickable, text prints black
```
NEW:
```latex
% TODO: manual review -- added xr-hyper (must load before hyperref) and \externaldocument, so that "Additional file~\ref{app:...}" reads its number from the Additional files' .aux instead of printing "??". The .aux path is the only Additional files build that exists today; change it if that document moves.
\usepackage{xr-hyper}
\usepackage[colorlinks=true, allcolors=black]{hyperref} % [2026-10-10] links stay clickable, text prints black
\externaldocument[][nocite]{thesis_rework/stage1_takeaways_bullets/additional_files}
```

### E7 — Summary of Research Contributions: software added
- file: thesis.tex
- kind: new-paragraph
- why: Gap item 15; Dablander lists code under its own heading. Every fact is from thesis.tex: the NoiseInject description (Chapter 1 questions paragraph), MIT license and Zenodo citation (Section sec:noiseinject_pkg), and the KIRBy sentence at the end of the Chapter 1 outline. No version number, because thesis.tex's own TODO(cite) says 1.0.0 disagrees with Zenodo.
- todo_line: % TODO: manual review (above each new paragraph)
OLD:
```latex
Punt, A., Morris, G., Werner, S., \& Hanser, T. (2024). Investigating the impact and mitigation of noise in QSAR models. Presented at the Autumn UK-QSAR 2024 Meeting.
```
NEW:
```latex
Punt, A., Morris, G., Werner, S., \& Hanser, T. (2024). Investigating the impact and mitigation of noise in QSAR models. Presented at the Autumn UK-QSAR 2024 Meeting.

\section*{Software}
% TODO: manual review
NoiseInject: an open-source Python package for benchmarking noise robustness and uncertainty quantification (Section~\ref{sec:noiseinject_pkg}). It is available under an MIT license and archived on Zenodo \citep{noiseinject}.

% TODO: manual review
KIRBy: the software repository that implements the search for adaptive hybrid representations and the experiments in this thesis (Additional file~\ref{app:availability}).
```
## For the author (not changed)

- AI statement against the source (gap item 5). It says "All written text in this thesis is my own" and "No AI-generated text appears verbatim or closely paraphrased". thesis.tex carries 410 comment lines marking AI-written or AI-edited paragraphs ("% TODO: manual review", "% TODO: review, needs manual review"), and the abstract is marked "[CLAUDE 2026-10-09] abstract drafted". Either every marked paragraph is rewritten before submission, or those two sentences change. The simplified statement keeps both claims as you wrote them.
- AI statement, "Where an AI interaction materially shaped a specific line of investigation, this is noted in the relevant chapter": no such note exists in the printed text of thesis.tex (searched for "generative AI", "Claude" and "AI interaction" outside comments). Kept, because the claim is yours.
- AI statement, "the cliff-aware noise mitigation approach": thesis.tex calls it a detector, "the cliff-aware gap, a detector that compares a label with its neighbors' labels" (Chapter 8 list). Wording kept; check it is the contribution you mean.
- AI statement placeholders still open (gap item 4): "[MODEL VERSIONS, ...]" twice, "[MONTH YEAR]" three times, "[IF APPLICABLE: ...]", "[CONFIRM BEFORE SUBMISSION:]", "[If any non-public material was shared, ...]", "[REPOSITORY URL]". Only the two "Chapter~[X]" were filled.
- Chapter 3 publication status (gap item 8). The Statement of Contribution says Chapter 3 "is adapted from a jointly authored manuscript, ... in preparation for the \textit{Journal of Cheminformatics}". thesis.tex comments out the chapter's own note: "[CODE INPUT: commented out -- the Chapter 3 paper was not submitted, so it is not referenced]". Pick one statement and use it in both places. For the same reason the NoiseInject manuscript was not added to "Research Papers" in the Summary of Research Contributions.
- Acknowledgments funding: your sentence "I am grateful to Lhasa Limited \textbf{[and FUNDER, grant no.~XXX]} for their generous financial support." now sits next to the EPSRC sentence, which also names Lhasa. Fill the placeholder or drop it; keeping both thanks Lhasa for funding twice.
- Acknowledgments examiners sentence: thanks only your transfer of status examiners. The confirmation of status examiners were left out because of the rule against mentioning the confirmation; add them yourself if you want them thanked.
- Acknowledgments: "Department of Statistics" is taken from your AI statement and the old Statement of Originality TODO; "St Catherine's College" from the title page. Check both.
- Acknowledgments: with the friends sentence added before it, "Finally, I would like to thank my parents" is still the last thanks to people, but check how it reads.
- Abbreviation expansions not spelled out anywhere in thesis.tex, written from standard usage and not checked against a source in this pass: hERG, AC-SINS, CHO, HIC, SMAC, MDCK, MDR1, EGFR, VEGF, IMGT, USR, USRCAT, SALI, EC$_{50}$, FR, LM, SD. The text also never expands most of these at first use (chemistry guide item 16 for hERG and MDR1–MDCK).
- Two short forms in tables differ from the text: "LGBM" (20 uses) for LightGBM and "XGB" (3 uses) for XGBoost, for example "RF, LGBM, DNN, XGB". Both are now listed; using the full names in the tables would let you drop them.
- "area under the ROC curve" appears twice in Chapter 7 ("Both experiments grade the ranking by the area under the ROC curve.") with ROC never expanded; the list now defines AUROC only.
- New package (gap item 16), edit E6: xr-hyper, loaded before hyperref, with \externaldocument pointing at thesis_rework/stage1_takeaways_bullets/additional_files.aux (built 8 Oct). The 10 Oct PDF prints "Additional file ??" 30 times. The references resolve only after the Additional files are built, and their numbers are as fresh as that .aux. Skip E6 if you would rather not depend on the stage1 folder.
- NoiseInject version: Section sec:noiseinject_pkg says "version 1.0.0", and its own TODO(cite) says Zenodo's latest release is v0.3.0. The new Software entry gives no version until that is settled.
- KIRBy: thesis.tex gives no URL or license for the repository, only "Additional file~\ref{app:availability}". The Software entry points there.
- "List of Publications" TODO in main.tex (gap item 14): kept, with a note under it; delete it if the Summary of Research Contributions is enough.
- Statement of Contribution (gap items 9 and 10): already settled by your 2026-10-10 change (declaration and signature lines removed, Statement of Authorship sentence removed). Nothing planned.
- Title page (gap items 11 and 12): college, crest and "Trinity Term 2026" are in; the wrong crest comment is gone. The department line ("in the Department of Statistics", Buttenschoen only) is optional and was not added.
- Research Papers: your existing "DECISION FOR ADELAIDE: one planned method paper vs two" comment is still open.

## Checker notes

Checked 2026-10-10 against thesis.tex and main.tex. All seven OLD blocks are exact, unique substrings (python3 -I, str.count = 1 each), re-checked after the fixes below.

- E1 kept. chap:why has no label; chap:flaws (Ch 4) to chap:similarity (Ch 6) matches the Collaborations section ("hERG data used in Chapters flaws and method", OPIG "Chapters method and similarity").
- E2 fixed, two places where the shorter text changed a claim. "take full responsibility for its correctness" narrowed "all code" to the AI-assisted code; restored to "I take full responsibility for the correctness of all code." "complies with the policies of" the Division and Department turned their guidance into policy; now "the University of Oxford's policy and ... the guidance of" the Division and Department. Kind author-text is right: her own TODO asks for the simplification. Chapter fills checked: the hybrid search is chap:method, the cliff-aware gap is in chap:noise.
- E3 fixed. The examiners sentence thanked "confirmation of status examiners"; the task rules forbid mentioning the confirmation, so that clause is removed and the For-the-author line updated. The grant number EP/S024093/1 is not in thesis.tex; it is copied from the author's own comment in main.tex, so it is not invented. The 33-word funding sentence is her text, unchanged.
- E4 fixed. GCN is also never used outside comments in thesis.tex (0 uses), so it is now commented out like the other six, and both the todo_line and the why updated. Every added entry was counted in thesis.tex with comments stripped and has at least one use (FR as FR1--FR4, LM as "protein LM" in tables, AC as "AC detection"). Expansions checked: hERG, AC-SINS, HIC, SMAC, MDR1, IMGT, USRCAT, SALI are correct.
- E5 kept. Comment only; her TODO is kept.
- E6 kept. xr-hyper loads before hyperref; the .aux file exists at thesis_rework/stage1_takeaways_bullets/additional_files.aux. It adds a package to the preamble the other session is converting, so the applier must carry it into the package.
- E7 kept. Every fact matches thesis.tex: NoiseInject description (Chapter 1 questions paragraph), MIT license and Zenodo (sec:noiseinject_pkg), KIRBy sentence (end of Chapter 1 outline). Both new paragraphs carry "% TODO: manual review".
- No banned words or AI phrasing in any NEW text. No new sentence over 30 words except the author's own funding sentence (33 words, copied verbatim). No \label keys changed, no section titles renamed.
