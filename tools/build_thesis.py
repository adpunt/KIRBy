#!/usr/bin/env python3
"""
Assemble thesis.tex from three sources by EXACT LINE SLICING.

Rule: any prose that exists in a source is copied byte-for-byte. Nothing is
retyped or rephrased. Only structural scaffolding (section headings, TODO
comments) is authored here, and outline-derived scaffolding is transcribed
verbatim from THESIS_OUTLINE.md.
"""
import re
from pathlib import Path

CR = Path("/Users/apunt/Downloads/Confirmation_Report/paper.tex")
NI = Path("/Users/apunt/repos/qsar_qm_models/noise_inject.tex")
OUT = Path("/Users/apunt/repos/KIRBy/thesis.tex")

p_lines = CR.read_text().splitlines()
n_lines = NI.read_text().splitlines()

# Record every slice we take, for the coverage audit.
TAKEN = {"P": set(), "N": set()}


def P(a, b=None):
    """paper.tex lines a..b inclusive, 1-indexed."""
    b = a if b is None else b
    TAKEN["P"].update(range(a, b + 1))
    return "\n".join(p_lines[a - 1 : b])


def N(a, b=None):
    """noise_inject.tex lines a..b inclusive, 1-indexed."""
    b = a if b is None else b
    TAKEN["N"].update(range(a, b + 1))
    return "\n".join(n_lines[a - 1 : b])


# --- document-self-reference fixes -------------------------------------------
# The only text edits permitted. Each is recorded and reported.
EDITS = []


def refix(text, where):
    """report -> thesis, for unambiguous document self-references only."""
    subs = [
        (r"\bThis report addresses\b", "This thesis addresses"),
        (r"\bThis report fills that gap\b", "This thesis fills that gap"),
        (r"\bthe first half of the report\b", "the first half of the thesis"),
        (r"\bthis report\b", "this thesis"),
        (r"\bThis report\b", "This thesis"),
        (r"\breport structure\b", "thesis structure"),
    ]
    for pat, rep in subs:
        for m in re.finditer(pat, text):
            EDITS.append((where, m.group(0), rep))
        text = re.sub(pat, rep, text)
    return text


def todo(*lines):
    return "\n".join("% TODO: " + l if l else "%" for l in lines)


def outline(section, *bullets):
    head = f"% ---- from THESIS_OUTLINE.md :: {section} (quoted; long bullets condensed) ----"
    body = "\n".join("%   " + b for b in bullets)
    return head + "\n" + body + "\n% ---- end outline extract ----"


B = []  # blocks
add = B.append

# =============================================================================
add(r"""%%============================================================================
%% thesis.tex  --  Adelaide Punt, DPhil thesis (body)
%%
%% Working title: Noise and Uncertainty in Molecular Property Prediction:
%%                Robust Models and Adaptive Representations
%%
%% ASSEMBLED, NOT REWRITTEN.  Every paragraph below is copied byte-for-byte from
%% one of:
%%    [CR]  Confirmation_Report/paper.tex
%%    [NI]  qsar_qm_models/noise_inject.tex   (supersedes CR ch.3 throughout)
%%    [KB]  KIRBy/PAPER_DRAFT.md              (converted md -> tex, wording kept)
%%    [OL]  Theses/THESIS_OUTLINE.md          (quoted verbatim inside % comments)
%% Provenance is marked above each block as e.g.  % [NI 441-471]
%%
%% Drop-in replacement for paper.tex:  main.tex already does \input{paper}.
%% Point it at this file instead:      \input{thesis}
%%
%%----------------------------------------------------------------------------
%% !! THREE THINGS TO DO BEFORE THIS COMPILES !!
%%
%% 1. PREAMBLE: use KIRBy/main.tex (report class, amssymb, per-chapter counters, front matter).
%%    The old Confirmation_Report/main.tex will NOT compile this file.
%%
%% 2. BIBLIOGRAPHY.
@@MISSING_CITES@@
%%
%% 3. refs.bib: FIXED in KIRBy/refs.bib (use that copy, not the one in Confirmation_Report).
%%    Two unescaped ampersands were FATAL LaTeX errors, both pre-existing:
%%       refs.bib:1755   title={Sort &amp; Slice: ...}          ->  Sort \& Slice
%%       refs.bib:1859   booktitle={... Knowledge Discovery & Data Mining (KDD)} -> \&
%%    With those two fixed the thesis builds clean: 69 pages, 0 errors, 0 undefined refs.
%%    (refs.bib also has 5 duplicate entries: bibtex 'Repeated entry' at 749, 1330, 1645,
%%     1947, 1965. Harmless but worth pruning.)
%%
%% 4. CHAPTERS vs SECTIONS. The confirmation report used \section for what its
%%    prose calls "Chapter N", under \documentclass{article}. That convention is
%%    preserved here so the cloned Overleaf project still compiles untouched.
%%    To get real chapters: switch main.tex to \documentclass{report} and run
%%       sed -i '' 's/^\\section{/\\chapter{/; s/^\\subsubsection{/\\paragraph{/; s/^\\subsection{/\\section{/' thesis.tex
%%    (do the substitutions in that order).
%%
%%----------------------------------------------------------------------------
%% REMOVED relative to the confirmation report:
%%   * old ch.4 "Exploring techniques for learning with noisy labels for QSAR"
%%     -- preliminary; condensed to its motivating findings in Sec 5.1 below,
%%     and measured properly in ch.7 per the outline.
%%   * old ch.3's entire results section, and its methods -- superseded wholesale
%%     by noise_inject.tex.
%%   * the NSI (noise sensitivity index) metric -- noise_inject.tex replaces it
%%     with NDS (noise degradation slope). Retention is kept.
%%   * the confirmation-report Timeline subsection.
%%
%% CROSS-REFERENCES TO FIX: chapter numbers inside carried-over prose still refer
%% to the confirmation report's numbering. Every one is flagged inline with
%%   % TODO(xref)
%% Old ch.3 -> new ch.3 + ch.4;  old ch.4 (LNL) -> new Sec 5.1;  old ch.5 -> ch.5-7.
%%============================================================================
""")

# =============================================================================
# FRONT MATTER
# =============================================================================
add(r"""%%========================= FRONT MATTER =====================================
% Front matter lives in main.tex. The outline asks for the following, none of
% which exists yet in the confirmation report. Listed here so nothing is lost.
""")
add(outline(
    "FRONT MATTER",
    "Title page.",
    "Acknowledgements (Garrett Morris; the Lhasa supervisors Thierry Hanser and Stephane",
    "  Werner; the EPSRC Centre for Doctoral Training, grant EP/S024093/1; OPIG; group; family).",
    "Abstract (about one page; the noise problem in molecular property prediction, then one",
    "  clause per chapter in order, then the open-source NoiseInject note).",
    "Contents.  List of Figures.  List of Tables.",
    "Abbreviations (QSAR, ECFP4, SNS, PDV, OHE, MHG-GNN, GNN, MLP, RF, QRF, XGBoost, NGBoost,",
    "  GP, SVM, BNN, VBLL, LNL, NDS, ECE, ANOVA, RMSE, ROC-AUC, F1, PCA, mRMR, RFE, RBF, DFT,",
    "  ADME, and the rest, spelled out once here so the chapters can use them bare).",
    "Declaration (the thesis includes the jointly authored NoiseInject paper and the Lhasa",
    "  and OPIG collaborations; a short statement of the author's own contribution). A signed",
    "  Statement of Authorship for the jointly authored NoiseInject paper is included, as the",
    "  Oxford requirement for joint work, covering Chapters 3 and 4.",
    "Data and Code Availability (three distinct software artifacts: (1) NoiseInject, MIT,",
    "  github.com/adpunt/noise_inject; (2) a standalone Rust and Python chemical-processing and",
    "  precompute pipeline, to be simplified and released separately; (3) the KIRBy research",
    "  codebase (pure Python). KIRBy contains no Rust.",
    "List of Publications: the NoiseInject paper (Punt, Hanser, Werner and Morris, in",
    "  preparation for the Journal of Cheminformatics); the two planned hybrid-representation",
    "  papers (Chapters 6 and 7); the UK-QSAR conference presentation; the OPIG Blopig posts;",
    "  and the open-source software.",
))
add(todo(
    "The confirmation report's Abbreviations list (main.tex) is stale: it still lists",
    "NSI and OHE-era entries and omits NDS, VBLL, ANOVA, ICC, LightGBM, SVM, MHG-GNN.",
    "noise_inject.tex lines 588-623 hold an updated list -- merge it into main.tex.",
))

# =============================================================================
# CHAPTER 1  INTRODUCTION
# =============================================================================
add(r"""
\pagebreak
\section{Introduction}\label{chap:intro}
""")
add("% [NI 193] supersedes [CR 5,7]: adds the static-vs-learned representation split\n"
    "% that the outline's Sec 1.1 explicitly turns on.\n" + N(193))
add("\n% [NI 195] supersedes [CR 9]: adds jorner2021 (diminishing returns once model\n"
    "% error approaches experimental error), cited by the outline's Sec 1.2.\n" + N(195))
add("\n% [CR 11] the thesis question. 'report' -> 'thesis' applied.\n" + refix(P(11), "CR:11"))
add("\n% [CR 15] the gap statement.\n" + refix(P(15), "CR:15"))

add("\n\\subsection{The questions this thesis asks}\n")
add(outline(
    "1.3  The questions this thesis asks",
    "Under label noise, which matters more for robustness, the representation or the model",
    "  architecture, and can probabilistic models recover usable uncertainty estimates that",
    "  track that noise?",
    "Building on those answers: can the representation itself be adapted to make prediction",
    "  more robust, and to make molecular similarity and noise estimation more reliable?",
))
add("\n% [NI 202] the three questions, as stated in the NoiseInject paper.\n" + N(202))

add("\n\\subsection{From noise robustness to adaptive representations}\n")
add(outline(
    "1.4  From noise robustness to adaptive representations",
    "The first half establishes that, under noise, the model dominates robustness while the",
    "  representation governs clean performance, and that only certain probabilistic models",
    "  sense per-sample noise.",
    "An exploratory study of learning-with-noisy-labels methods then showed that the",
    "  approaches relying on molecular similarity fail under noise, which motivated the second",
    "  half: an adaptive hybrid representation that builds a compact, informative representation",
    "  from several others, and in doing so repairs the notion of similarity those methods",
    "  depend on.",
))
add(r"\todo[inline, color=yellow!40]{Write Sec 1.4 from the outline bullets above. The raw material is Sec 5.1.}")

add("\n\\subsection{Outline of the thesis and contributions}\n")
add(todo(
    "The confirmation report's 'Report Overview' [CR 85-97] described a SIX-chapter report",
    "including the LNL chapter. It is deliberately NOT carried over -- it is wrong for the",
    "eight-chapter thesis. Rewrite from the outline bullets below, which are already",
    "per-chapter and in the author's own words.",
))
add(outline(
    "1.5  Outline of the thesis and contributions",
    "Chapter 2 sets out the representations, models, noise model, and evaluation that the",
    "  rest of the thesis uses, and introduces NoiseInject, the open-source benchmarking framework.",
    "Chapter 3 is the first NoiseInject chapter (completed; adapted from Punt, Hanser, Werner",
    "  and Morris, in preparation for the Journal of Cheminformatics). It decomposes the",
    "  variance in clean performance and in noise robustness across representations and model",
    "  architectures, and finds that representation governs clean performance while model",
    "  architecture governs noise robustness, reversing the ordering.",
    "Chapter 4 is the second NoiseInject chapter (completed; same paper). It asks which",
    "  probabilistic models produce uncertainty estimates that track label noise and how the",
    "  representation affects this, and finds that noise-aware uncertainty is available only for",
    "  particular model and representation pairings, with the models carrying an explicit learned",
    "  scale or observation-noise term (NGBoost and the Gaussian process) tracking per-sample noise best.",
    "Chapter 5 introduces the adaptive hybrid representation, with the exploratory",
    "  learning-with-noisy-labels work as its motivation.",
    "Chapter 6 is the first of the two planned method papers (intended for submission;",
    "  venue to be confirmed). It will test when and why the hybrid representation helps, that",
    "  it is explainable, and that it extends to new embeddings and to federated data.",
    "Chapter 7 is the second planned method paper. It will return to molecular similarity and",
    "  label noise, testing whether the hybrid representation gives a more property-aware",
    "  similarity and improves noise estimation.",
    "Chapter 8 draws the argument together and then develops the main open directions.",
))
add(r"\todo[inline, color=yellow!40]{Write Sec 1.5 from the outline bullets above.}")

add("\n\\subsection{Collaborations and funding}\n")
add(outline(
    "1.6  Collaborations and funding",
    "The EPSRC Centre for Doctoral Training (grant EP/S024093/1); Lhasa Limited (label-noise",
    "  framing and the federated work); OPIG (antibody and protein datasets and the Blopig",
    "  collaboration).",
))
add(r"\todo[inline, color=yellow!40]{Write Sec 1.6.}")

# =============================================================================
# CHAPTER 2  BACKGROUND
# =============================================================================
add(r"""
\pagebreak
% \label{sec2} resolves \ref{sec2} in the fig:global_overview caption carried from
% noise_inject.tex, where it pointed at that paper's Methods section.
\section{Background}\label{chap:background}\label{sec2}
""")
add("\\subsection{Introduction}\n")
add("% [CR 17] the lead-in sentence that used to close the confirmation report's intro.\n" + P(17))

add("\n\\subsection{Molecular Representations in QSAR}\n")
add(todo(
    "Outline Sec 2.1 additionally asks for, and the text below does NOT yet cover:",
    "  - Definition 2.1 (molecular representation / featurization) as a numbered definition",
    "  - a comparison table (Table 2.1) closing the section",
    "  - MACCS keys (Durant et al. 2002) and Avalon as named fingerprints",
    "  - mordred descriptors (Moriwaki et al. 2018); MolFormer (Ross et al. 2022)",
    "  - protein/antibody language-model embeddings (ESM-2, ESM-C, ESM3, SaProt, AbLang2,",
    "    AntiBERTy, IgT5), introduced here so ch.6 can refer back  <- WHOLE MISSING SUBSECTION",
))
add("% [CR 21-54] carried whole, incl. Figure `molecular_representations.png` (= outline Fig 2.1).\n" + P(21, 54))

add("\n\\subsection{Models for molecular property prediction}\n")
add("% [NI 222] supersedes [CR 256]: labels are not true values, so models overfit noise.\n" + N(222))
add("\n% [CR 58-74] trees, ensembles, NNs, GCN/GIN. Background depth that NI does not repeat.\n" + P(58, 74))
add("\n\\subsubsection{Probabilistic decision trees}\n")
add("% [NI 224] supersedes [CR 264-266]: adds LightGBM and NGBoost's O(Np^3) cost.\n" + N(224))
add("\n\\subsubsection{Support vector machines}\n")
add("% [NI 226] new in NI; the confirmation report had no SVM background.\n" + N(226))
add("\n\\subsubsection{Gaussian processes}\n")
add("% [CR 270-278] kept over [NI 228] because CR states the Tanimoto kernel as a displayed\n"
    "% equation, which the outline's Sec 2.2 asks to carry as Equation 2.1.\n" + P(270, 278))
add("\n% [NI 228] the NI kernel-choice paragraph (Tanimoto for fingerprints, RBF for PDV).\n"
    "% TODO: this overlaps the CR paragraph above. Merge into one; keep the displayed equation.\n" + N(228))
add("\n\\subsubsection{Bayesian neural networks}\n")
add("% [CR 282] why NNs underperform traditional methods in QSAR (Baskin, Koutsoukas), and the\n"
    "% Bayesian-transformation taxonomy.\n"
    "% TODO: this paragraph's LAST THREE SENTENCES name full-BNN / last-layer-BNN / var-BNN.\n"
    "% noise_inject.tex standardises on full-BNN and VBLL and drops last-layer entirely. The two\n"
    "% namings CONTRADICT each other. Reconcile: keep the Baskin/Koutsoukas framing, take the\n"
    "% variant taxonomy from [NI 230] below.\n" + P(282))
add("\n% [NI 230] supersedes [CR 293]: NN-alpha/NN-beta architectures, VBLL, explicit priors,\n"
    "% 100 MC forward passes.\n" + N(230))
add("\n% [CR 284-289] Figure `bayesian_transformation.png` = outline Fig 2.2.\n" + P(284, 289))
add("\n% [CR 291] Bayesian methods vs other non-deterministic uncertainty sources.\n" + P(291))
add("\n% [NI 232] epistemic/aleatoric decomposition, and which models it is defined for.\n" + N(232))
add(todo(
    "Outline Sec 2.2 also asks for: MC dropout (Gal and Ghahramani 2016) and deep ensembles",
    "(Lakshminarayanan et al. 2017) named as the non-Bayesian uncertainty baselines; and the",
    "'two distinct uncertainty toolkits' note (NoiseInject's QRF/VBLL/BNN/Gauche-Tanimoto-GP",
    "vs KIRBy's MAPIE conformal + Mondrian AD regressor + sklearn-RBF GP; KIRBy does not",
    "reuse the NoiseInject probabilistic models). Neither exists in any source. WRITE IT.",
))

add("\n\\subsection{Molecular similarity and activity cliffs}\n")
add(todo(
    "WHOLE MISSING BACKGROUND SUBSECTION. The outline needs, and no source supplies:",
    "  - Definition 2.2 (the similarity assumption; Johnson and Maggiora 1990;",
    "    van Tilborg et al. 2022; Maggiora et al. 2014 on measure-dependence)",
    "  - Definition 2.3 (activity cliff; Stumpfe and Bajorath 2012)",
    "  - Aggarwal et al. 2001 on why high-dimensional pairwise distances are unreliable",
    "The paragraphs below are the closest existing prose, lifted from the confirmation",
    "report's hybrid chapter, where they served as motivation rather than background.",
))
add("% [CR 781-782] Bajusz on Tanimoto; the similarity assumption; activity cliffs confused\n"
    "% with noise. The argument these open is finished in Sec 5.1 ([CR 779, 784]) -- kept apart\n"
    "% so nothing is duplicated.\n" + P(781, 782))

add("\n\\subsection{Noise in molecular data}\n")
add("% [CR 128-143] sources of experimental label noise, carried whole.\n" + P(128, 143))
add("\n% [CR 13] the noise-as-regularization caveat. Outline Sec 2.4: 'not all label noise is\n"
    "% harmful, since mild noise and scaffold splitting can act as a form of regularization'.\n" + P(13))

add("\n\\subsubsection{Artificial noise injection}\n")
add("% [NI 325-327] supersedes [CR 170-172].\n" + N(325, 327))
add("\n% [NI 330-362] regression strategy table = outline Table 2.2.\n"
    "% NOTE this table SUPERSEDES [CR 175-191]: Valprop is now sigma(1+0.1|y|) not sigma+0.1|y|,\n"
    "% Threshold/Quantile are stated as explicit piecewise per-point scales, and the row order\n"
    "% changed. Do not reuse the confirmation report's version.\n" + N(330, 362))
add("\n% [NI 364] classification strategies, prose. Supersedes the CR classification TABLE\n"
    "% [CR 195-211] (= outline Table 2.3), which NI demoted to a sentence because classification\n"
    "% was never run.\n"
    "% TODO: outline Sec 2.4 still asks for Table 2.3. Reinstate [CR 195-211] here if wanted.\n" + N(364))
add("\n% [NI 366] the sigma scaling factor, per strategy.\n" + N(366))
add("\n% [NI 368-374] Figure `fig_methods_noise_strategies.png` -- new figure, not in the\n"
    "% confirmation report's image set. Assumed present in the Overleaf project.\n" + N(368, 374))

add("\n\\subsection{Learning with noisy labels}\n")
add("% [CR 147] the Song et al. taxonomy, Cortes-Ciriano, No Free Lunch. This is the body of\n"
    "% the confirmation report's misleadingly-titled 'Artificial Noise Injections' subsection.\n" + P(147))
add(todo(
    "Outline Sec 2.5 also wants Wilson 1972 (neighbour-editing root) and Dablander et al. 2023",
    "(the activity-cliff confound) named here, as the methods returned to in ch.7.",
))

add("\n\\subsection{NoiseInject: a benchmarking framework, and the precompute pipeline}\n")
add("% [NI 376-384] supersedes [CR 228-234]. The old text pointed at a placeholder URL\n"
    "% (github.com/username/noisybench) and claimed CSV/DataFrame input; NI corrects both.\n" + N(378, 384))
add("\n% [CR 151-168] the split Rust/Python precompute pipeline + Figure `figure_mmap_structure.png`\n"
    "% (= outline Fig 2.3). Outline Sec 2.6 is emphatic that this is a SEPARATE artifact from\n"
    "% NoiseInject and from KIRBy.\n" + P(151, 168))
add("\n% [CR 217] the framework's model families.\n"
    "% TODO: the tail of this sentence lists the LNL methods (Meta Weight Net, DivideMix,\n"
    "% confident learning, small-loss selection, MentorNet, curriculum learning, multi-stage\n"
    "% cleaning). Those belong to the removed LNL chapter -- trim, or keep as a statement of\n"
    "% what the apparatus supports.\n" + P(217))
add("\n% [CR 220] fifteen loss functions and six molecular distance metrics.\n" + P(220))
add("\n% [CR 222] Optuna hyperparameter tuning and uncertainty extraction.\n" + P(222))
add("\n% [CR 224] the split-language rationale.\n" + P(224))
add("\n% [NI 219] the one-sentence NI version of the same point.\n"
    "% TODO: redundant with [CR 224] above -- keep one.\n" + N(219))
add(todo(
    "[CR 77] 'High-throughput experimentation' is NOT carried over: its second half describes",
    "the removed LNL chapter ('I extend this analysis to learning with noisy labels (LNL)",
    "methods adapted from classification...') and it names ESOL, which the group has cut.",
    "Salvage the first two sentences if you want them here.",
))
add(todo(
    "Outline Sec 2.6 wants it said explicitly that the precompute pipeline is a standalone Rust",
    "crate with optional Python bindings, to be simplified and released separately, and that",
    "NoiseInject is MIT at github.com/adpunt/noise_inject, archived on Zenodo.",
))

add("\n\\subsection{Evaluation methodology}\n")
add(todo(
    "Outline Sec 2.7 wants scaffold/group-aware splitting stated here (no random splitting;",
    "DeepChem scaffold split). It is currently only stated per-experiment, in Sec 3.2.",
))
add("% [NI 234-322] performance metrics, NDS, coverage, ECE, ANOVA + eta^2, the redundancy/ICC\n"
    "% screen, and the summary metrics table. Supersedes [CR 298-340] entirely.\n"
    "% NOTE: NSI (noise sensitivity index) is GONE -- NDS replaces it.\n"
    "% NOTE: the table at [NI 310] uses \\mathbb{1}; main.tex must load amssymb.\n" + N(236, 322))
add("\n% [CR 322-326] retention, kept because the outline asks for it as a numbered equation and\n"
    "% NI only names it in prose.\n"
    "% TODO(latex): line reads `\\times 100%.` -- the bare % comments out the period. Preserved\n"
    "% verbatim; change to `100\\%` when you touch it.\n" + P(322, 326))

# =============================================================================
# CHAPTER 3
# =============================================================================
add("\n\\subsection{Conclusions}\n")
add(outline(
    "2.7 (closing)",
    "With these in place, Chapter 3 puts them to work, asking whether the representation or",
    "  the model architecture governs how badly a predictor degrades under label noise.",
))
add(r"\todo[inline, color=yellow!40]{Write the Background chapter's closing paragraph.}")
add(r"""
\pagebreak
\section{Noise Robustness Across Molecular Representations and Model Architectures}\label{chap:robustness}
""")
add(todo(
    "Outline: 'An opening note records that Chapters 3 and 4 are adapted from Punt, Hanser,",
    "Werner and Morris, in preparation for the Journal of Cheminformatics, and that a signed",
    "Statement of Authorship for this jointly authored paper, covering Chapters 3 and 4, is",
    "included in the front matter as the Oxford requirement for joint work.'",
))
add(r"""\textit{Preamble.} The work in this chapter and in Chapter~\ref{chap:uq} is adapted from Punt, A., Hanser, T., Werner, S., and Morris, G.M., in preparation for the \textit{Journal of Cheminformatics}. Most of the figures, tables, and text in these two chapters are therefore either identical or similar to the content of that manuscript. A signed Statement of Authorship for this jointly authored work, covering both chapters, is included in the front matter.""")

add("\n\\subsection{Introduction}\n")
add("% [NI 197] the LNL taxonomy / No Free Lunch framing.\n"
    "% TODO: this now duplicates Sec 2.5, which carries [CR 147] saying the same thing.\n"
    "% Cut one. Sec 2.5 is the fuller version.\n" + N(197))
add("\n% [NI 199] review of existing approaches: Deng2023, Kolmar2021, Heid2023.\n" + N(199))
add(todo(
    "Outline Sec 3.1 also names Cortes-Ciriano, Bender and Malliavin 2015 and Jorner et al.",
    "2021 in the review-of-existing-approaches paragraph. Both are cited elsewhere; pull them in.",
))

add("\n\\subsection{Datasets and noise injection}\n")
add("% [NI 206-213] supersedes [CR 242-248]. Substantive changes: ESOL and the hERG\n"
    "% CLASSIFICATION task are gone; LogD (N=5039), Caco-2 efflux (N=2161) and hERG Ki\n"
    "% (N=1482, ChEMBL, landrum2024 protocol) are the three external sets; 80/10/10 split;\n"
    "% 5-fold GroupKFold scaffold CV; the QM data are argued to be 'free' of noise.\n" + N(207, 213))
add("\n\\subsection{Molecular representations}\n")
add(todo(
    "This restates much of Sec 2.1. Outline ch.3 has no representations subsection -- the shared",
    "material lives in ch.2. Reduce this to the experiment-specific parameters (L, d, the",
    "z-scoring) and cross-reference Sec 2.1 for everything else. Same for the model list, which",
    "is NOT repeated here: ch.3 relies on Sec 2.2.",
))
add("% [NI 215-217] supersedes [CR 252]. Substantive: SNS is now L=1024 (was 2048), PDV is\n"
    "% 200-d and z-scored, mol2vec is added, the randomized-SMILES variant is dropped.\n" + N(217))

add("\n\\subsection{Representation governs clean performance, architecture governs robustness}\n")
add("% [NI 389-439] variance decomposition: Table (anova_decomposition) + Figures\n"
    "% fig2_anova_decomposition.png, fig_interaction.png, fig1_global_overview.png.\n" + N(391, 439))

add("\n\\subsection{Probabilistic models are no more noise-robust than deterministic ones}\n")
add("% [NI 473-500] Bayesian transformations, Wilcoxon table, NN-family figure.\n" + N(473, 500))

add("\n\\subsection{Relative model robustness is largely strategy-independent}\n")
add("% [NI 443-471] NDS ranking table, Kendall's W = 0.92, strategy spread, PDV.\n" + N(443, 471))
add("\n% [NI 502-505] the model-representation interaction and the Gaussian top-10.\n" + N(502, 505))
add(todo(
    "Outline Sec 3.5 asks for the six strategies binned into 'mild, moderate, and severe tiers",
    "by retention' (its Table 3.4). NI reports the NDS spread instead and never tabulates",
    "retention tiers. Either add the table or drop the claim from the outline.",
))

add("\n\\subsection{The pattern holds across prediction targets}\n")
add("% [NI 551-569] validation on LogD, Caco-2, hERG Ki. Figures fig_validation_overview.png,\n"
    "% fig_validation_combined.png (both `figure*` -- fine in one-column article).\n" + N(553, 569))

add("\n\\subsection{Conclusion}\n")
add("% [NI 571] conclusion 1 of 2: robustness is set by the training mechanism.\n"
    "% TODO(latex): `\\~10\\%` should be `$\\sim$10\\%` -- \\~ is a tilde ACCENT. Preserved verbatim.\n" + N(571))
add("\n% [NI 580,582] the robustness half of the NoiseInject conclusion.\n" + N(580) + "\n\n" + N(582))
add(todo(
    "Outline Sec 3.7 closes with the hand-off: 'Since the representation did not drive",
    "robustness, the natural next question, taken up in Chapter 4, is whether probabilistic",
    "models can at least sense the noise they cannot resist.'",
))

# =============================================================================
# CHAPTER 4
# =============================================================================
add(r"""
\pagebreak
\section{Uncertainty Quantification Under Label Noise}\label{chap:uq}
""")
add(r"""\textit{Preamble.} This chapter continues the study of Chapter~\ref{chap:robustness} and is adapted from the same manuscript, Punt, A., Hanser, T., Werner, S., and Morris, G.M., in preparation for the \textit{Journal of Cheminformatics}.""")
add("\n\\subsection{Introduction}\n")
add(outline(
    "4.1  Introduction",
    "Chapter 3 showed which models stay accurate under noise. This chapter asks which",
    "  produce uncertainty estimates that track that noise, and how the representation affects this.",
    "Review of existing approaches: Hirschfeld et al. 2020 and Scalia et al. 2020 (no",
    "  uncertainty method reliably ranks errors across molecular datasets; scalable-UQ",
    "  comparisons), and Norinder et al. 2014 (conformal coverage and validity for molecules).",
))
add(r"\todo[inline, color=yellow!40]{Write Sec 4.1. Hirschfeld, Scalia and Norinder are cited in no source -- add to refs.bib.}")

add("\n\\subsection{Models with an explicit noise term best track prediction error under noise}\n")
add("% [NI 507-508] the per-sample uncertainty-noise question.\n" + N(508))
add("\n% [NI 510-538] Table tab:top_unc_noise (strongest and weakest model-rep pairs).\n" + N(510, 538))

add("\n\\subsection{NGBoost and the GP stay near target coverage while Bayesian variants miscalibrate}\n")
add("% [NI 540] the mechanism: predicted scales, global observation noise, weight posteriors.\n" + N(540))
add("\n% [NI 542-547] Figure fig_uncertainty_combined.png.\n" + N(542, 547))
add(todo(
    "Outline Sec 4.3 wants reliability/coverage/ECE 'across noise levels' as its own figure",
    "(Fig 4.2). NI reports coverage and ECE only inside tab:top_unc_noise, at one noise level.",
    "The numbers the outline quotes (NGBoost+PDV ECE = 0.11; GP+SNS 71.2%/94.6%;",
    "VBLL-alpha+MHG-GNN ECE = 13.41, 1-sigma coverage 26.6%) are all in that table.",
))

add("\n\\subsection{Uncertainty attribution is itself representation-dependent}\n")
add(todo(
    "Outline Sec 4.4 asks for the GP aleatoric/epistemic decomposition showing that added label",
    "noise is attributed to ALEATORIC uncertainty on PDV and MHG-GNN but misread as EPISTEMIC",
    "on SMILES and SNS (its Table 4.2, Fig 4.3). noise_inject.tex does NOT contain this result",
    "-- it only reports that for VBLL both components rise [NI 540, Fig 5b]. Either the result",
    "exists elsewhere and must be pulled in, or the outline claim needs revising.",
))
add(r"\todo[inline, color=yellow!40]{MISSING RESULT: the GP uncertainty decomposition by representation.}")

add("\n\\subsection{Fingerprint representations carry the usable uncertainty signal}\n")
add("% [NI 549] representation matters more for uncertainty tracking than for robustness.\n" + N(549))

add("\n\\subsection{Conclusion}\n")
add("% [NI 573-574] conclusion 2 of 2: which models track per-sample label noise.\n" + N(573, 574))
add("\n% [NI 576] the PDV/embedding trade-off.\n" + N(576))
add("\n% [NI 584,586] the uncertainty half of the NoiseInject conclusion.\n" + N(584) + "\n\n" + N(586))
add(todo(
    "Outline Sec 4.6 closes by setting up the second half: 'if the representation governs",
    "whether uncertainty and similarity are meaningful, can the representation be adapted to",
    "improve them?'",
))

# =============================================================================
# CHAPTER 5
# =============================================================================
add(r"""
\pagebreak
\section{A Method for Building Adaptive Hybrid Representations}\label{chap:method}
""")
add("\\subsection{Motivation: the failure of similarity-based noise handling}\n")
add(todo(
    "This subsection replaces the confirmation report's entire ch.4, 'Exploring techniques for",
    "learning with noisy labels for QSAR' [CR 666-776]. That study was preliminary; the outline",
    "keeps only its motivating finding here and defers the real measurement to ch.7.",
    "Not carried over: the 12-method table [CR 681-715], the HetGP kernel table [CR 725-740],",
    "the method-effectiveness table [CR 746-765], and the DivideMix / curriculum results.",
    "They are recoverable from Confirmation_Report/paper.tex if ch.7 wants them.",
))
add("% [CR 670] the exploratory LNL study and its headline finding.\n"
    "% TODO(xref): 'In Chapter 3' -- old ch.3 is now ch.3 AND ch.4.\n" + P(670))
add("\n% [CR 721] the twenty worst configurations all used a distance metric.\n" + P(721))
add("\n% [CR 779] why a single representation may lack the information distance needs.\n"
    "% TODO(xref): 'Building upon the results in Chapter 4' -- the LNL chapter is now this very\n"
    "% section. Reword to 'Building upon the exploratory study above'.\n"
    "% NOTE: [CR 781-782], which sat between this paragraph and the next in the confirmation\n"
    "% report, is now background in Sec 2.3. Cross-reference it rather than repeating it.\n" + P(779))
add("\n% [CR 784] the uninformative-dimensions argument -- the hinge of the whole second half.\n" + P(784))
add("\n% [CR 790] the pivot from sample removal to weighting and smoothing.\n" + P(790))
add("\n% [CR 792-798] the four research questions the second half was built to answer.\n"
    "% TODO: the outline redistributes these -- Q1 and Q3 -> ch.6, Q2 and Q4 -> ch.7. Decide\n"
    "% whether they stay here as a roadmap or move.\n" + P(792, 798))
add(todo(
    "Outline: 'This is the motivation for the method; it is exploratory, not a reported result,",
    "and the noisy-label question is measured properly in Chapter 7.' Say so explicitly.",
))

add("\n\\subsection{The adaptive hybrid representation}\n")
add(todo(
    "NAMING. The outline calls this 'the adaptive hybrid representation (thereafter the hybrid",
    "representation, or the method where unambiguous)'. The KIRBy draft calls it KIRBy",
    "throughout. Both namings appear below, exactly as written in each source. Pick one and",
    "apply it globally -- nothing here has been silently renamed.",
))
add("% [KB 25] the KIRBy one-paragraph definition, md -> tex. Wording unchanged.\n"
    r"""Here, we present KIRBy (Knowledge Integration by Representation), a method that takes a collection of frozen feature matrices for a dataset and returns a customized hybrid feature set optimized for a specific prediction task. The inputs can be anything that gives a feature vector per sample: fingerprints, descriptor sets, foundation model embeddings, gene expression signatures, histopathology features. From these sources, KIRBy selects a compact feature subset, searching over budgets, allocation strategies, importance methods, and downstream models. The search runs once per dataset. A portable Blueprint stores the result; applying it to new data does not require repeating the search.""")
add("\n% [OL 5.2] the outline's own framing of the contribution, verbatim.\n" + outline(
    "5.2  The adaptive hybrid representation",
    "We define the adaptive hybrid representation (thereafter the hybrid representation, or",
    "  the method where unambiguous): rather than choosing one representation, it assembles a",
    "  single feature set by selecting the most informative features from several",
    "  representations at once, chosen for the dataset and model in hand, before the final model",
    "  is trained. It prepares better inputs for standard downstream models rather than",
    "  introducing a new model.",
    "The idea of combining representations is not itself new; the contribution is that the",
    "  method poses this combination as a search-space optimisation problem and solves it",
    "  automatically. The space is the product of component representations, feature-importance",
    "  methods, allocation methods, internal evaluators, downstream models, and feature budgets,",
    "  and per-fold caching is what makes the search over it affordable. The formulation is not",
    "  specific to chemistry: it applies to any supervised learning task for which several",
    "  representations of the same inputs are available.",
))
add("\n\\subsubsection{Review of existing approaches}\n")
add("% [CR 786-788] the confirmation report's related-work paragraphs (conjoint fingerprints,\n"
    "% bio-activity fingerprints, feature-importance fusion).\n" + P(786, 788))
add("\n% [KB 19] the KIRBy draft's fusion taxonomy: early/late/intermediate fusion, MKL.\n"
    "% Citation keys are KIRBy's own and are NOT in refs.bib.\n"
    r"""So far, the most common strategy for combining feature sets has been concatenation (early fusion), which requires only a single downstream model \citep{Stahlschmidt2022}. It works when feature sets are homogeneous. When they are not, concatenation degrades \citep{Stahlschmidt2022}, and Li and Deng et al.\ called the practice ``misleading'' when the goal is to measure the contribution of individual representation types \citep{Deng2023}. Late fusion trains a separate model per source and aggregates predictions \citep{Steyaert2023}, avoiding the dimensionality problem. But cross-source feature interactions are lost. To recover cross-source interactions requires intermediate fusion (MOGONET \citep{Wang2021}, cross-attention \citep{Zhang2024}), but at the cost of end-to-end training on all sources; adding a new source means retraining from scratch. No interpretable feature subsets emerge from multiple kernel learning \citep{Gonen2011}, which operates entirely at the kernel level. There remains, as Steyaert et al.\ noted, no decisive evidence that any of these strategies is broadly superior \citep{Steyaert2023}.""")
add("\n% [KB 21] the dimensionality argument: ~5000-d input on a few hundred samples.\n"
    r"""In practice, most research groups default to one representation and ignore the rest. A cheminformatics lab reaches for ECFP fingerprints; a deep learning group uses MoLFormer or ChemBERTa; a structural biology group extracts ESM-2 embeddings. These may encode quite different aspects of the same underlying biology. But they are rarely compared, let alone combined. This leaves signal on the table. When groups do attempt combination, the most accessible strategy is concatenation. It works at modest scales. Concatenating just three standard sources, however (ECFP4 at 2,048 dimensions, ESM-2 at 1,280, Mordred descriptors at ${\sim}1{,}600$), gives a ${\sim}5{,}000$-dimensional input for a dataset that may contain only a few hundred labeled samples. At these sample-to-feature ratios, more features mean worse predictions \citep{Hughes1968, Bellman1961}. Models sensitive to distance, such as kNN and SVM, are affected most severely \citep{FernandezDelgado2014}.""")
add("\n% [KB 23] forward compatibility as a design requirement.\n"
    r"""The pace of new model development shows no sign of slowing, and extends well beyond small molecules into protein language models \citep{Lin2023, Elnaggar2022}, encoders built for antibodies \citep{Ruffolo2021, Olsen2022}, histopathology foundation models \citep{ElNahhas2025}, and gene expression. If a new protein language model appears next month, must we retrain from scratch to determine whether it adds value? Forward compatibility is therefore essential: any practical method must incorporate new sources without retraining or architectural modification. Deep fusion approaches fail this requirement, since adding a new modality means retraining the entire network. But treat encodings as independent frozen feature sets, and the problem disappears: a new model's output joins the pool and is tested alongside existing sources with no changes to the procedure.""")
add("\n" + todo(
    "Outline Sec 5.2 also names Xu, Tao and Xu 2013 (naive concatenation of views overfits on",
    "small samples), Valsecchi et al. 2020 (prediction-level fusion) and Ponzoni et al. 2017",
    "(hybridizing feature selection and feature learning). Xie et al. 2020 is already cited above.",
))

add("\n\\subsection{The method as a search}\n")
add(outline(
    "5.3  The method as a search",
    "The search ranges over six axes: the component representations; the feature-importance",
    "  method used to rank features within each representation; the allocation method that",
    "  divides a feature budget across representations; the internal evaluator used to score",
    "  candidate feature sets during the search; the downstream model; and the feature budget",
    "  itself. Figure 5.1 lays out the pipeline.",
    "The searchable downstream models are random forest, XGBoost, ridge, k-nearest",
    "  neighbours, LightGBM, MLP, SVM, and the Gaussian process, with CatBoost, DNN, CNN, BNN,",
    "  and NGBoost available when requested by name. Feature budgets span roughly 10 to 2,000",
    "  features, and every model is searched at every budget requested; the per-model",
    "  maximum-budget cap was removed on 2026-09-10. The allocation methods are greedy forward selection,",
    "  performance-weighted (with softmax, power-law, top-k, rank, score-gap, and",
    "  diversity-penalised entropy weighting schemes), dominant-greedy, fixed equal split,",
    "  two-stage, and stability-selection. The internal evaluators are match-the-downstream-model",
    "  or an explicit ridge, RF, SVM, LightGBM, or MLP.",
    "The feature-importance methods are SHAP-family and gradient or permutation importances:",
    "  treeshap, xgboost_gain, lightgbm_gain, permutation, kernelshap, deeplift,",
    "  integrated_gradients, and gradientshap. Mutual information was implemented but removed for",
    "  poor recall; mRMR and recursive feature elimination are not implemented.",
    "The search also supports capabilities beyond the six core axes: augmentation strategies",
    "  (uncapped, capped, greedy-ablation, all, none); fine-tuned per-fold representation",
    "  factories (rep_factories) with their own per-fold embedding cache; and a two-phase",
    "  refinement of step-size-sensitive greedy allocations.",
))
add("\n% [KB 149] the search pipeline, md -> tex.\n"
    r"""Treating each encoding as a frozen feature set, rather than learning a joint embedding end-to-end, makes the approach modular: any new encoder's output can be added to the pool without retraining, and the selection process itself is decoupled from how the features were produced. Given $K$ frozen representation matrices $\mathbf{R}_1, \dots, \mathbf{R}_K$ for the same $n$ samples, a label vector $\mathbf{y}$, and optional group assignments for cross-validation, the search finds which features from which sources best predict for a specified metric and downstream model. The output is a hybrid feature matrix of $b$ columns, where $b$ is the feature budget, together with a Blueprint that records all parameters needed to reproduce the identical feature selection on new data. The search first scores each source independently, then scores feature importance on training data within each cross-validation fold, allocates the budget among sources according to baseline performance, and scores candidate hybrid configurations via cross-validation. We retain the best configuration for each combination of downstream model and budget. Every selected feature traces back to its source and importance rank: a complete provenance record.""")
add("\n% [KB 151] fold-local preprocessing.\n"
    r"""Inputs from different sources differ in scale, distribution, and missingness patterns, and naively pooling them before splitting would leak validation-set statistics into the training procedure, a well-documented source of optimistic bias in feature selection studies \citep{Ambroise2002, Vabalas2019}. All preprocessing therefore runs within each cross-validation fold. We impute missing values with per-feature medians computed on the training fold and drop columns that are entirely missing. Binary feature matrices (molecular fingerprints, for instance) skip standardization, since their $\{0,1\}$ encoding already carries a consistent scale. For all other features, we standardize to zero mean and unit variance using training-fold statistics only.""")

add("\n\\subsubsection{Component representations}\n")
add("% [CR 804-806] the confirmation report's component-representation paragraph + quality filter.\n"
    "% TODO: superseded in scope by KIRBy (any vector input, up to 57 sources); PDV/ECFP4/MHG-GNN\n"
    "% is now just the molecular instance. Reframe.\n" + P(804, 806))

add("\n\\subsubsection{Feature scoring and budget allocation}\n")
add("% [KB 155] the three importance families and the 13 retained allocation strategies.\n"
    r"""Not all features in a given source are equally informative for a given task, and, for reasons that are not entirely clear, different importance methods disagree about which features matter. Tree-based measures favor features with high split gain, while gradient-based attributions reflect sensitivity in a neural network's loss landscape. To avoid committing to a single notion of importance, we score features using three methods, one from each family: LightGBM gain (tree-based), TreeSHAP (explanation-based), and DeepLift (neural attribution). We selected these three from an initial pool of eight methods after redundancy analysis on datasets showed that members of the same family gave highly correlated rankings. How to divide the total feature budget among sources is the role of the allocation strategy. We tested 31 candidates and retained 13 that appeared in top-performing configurations on multiple datasets and model types. The 13 retained strategies fall into three families. Score-based strategies divide the budget in proportion to transformed baseline scores: restricting features to the top $k$ sources, for example, or penalizing redundancy between correlated sources. A floor-based variant guarantees every source a minimum allocation, while a greedy forward approach iteratively adds features from whichever source most improves the internal evaluator, with early stopping. Once we have the allocation, we select the top features from each source by importance score and concatenate them to form the hybrid.""")
add("\n% [KB 157] augmentation: low-dimensional sources handled outside the budget.\n"
    r"""Some features are too low-dimensional or too specialized to participate in the budget allocation stage: MACCS fingerprints (167 features), for instance, or log-likelihood ratios (3 features). Excluding them from the allocation would lose potentially useful signal; including them would distort the budget distribution because they cannot contribute many features. We handle these as augmentation: after the base hybrid is constructed, we pool the augmentation features, rank them by random forest feature importance, and append them only if they improve the internal evaluator score.""")
add("\n% [CR 810-847] the two original allocation strategies, spelled out as numbered procedures\n"
    "% (greedy forward selection; performance-weighted). These are the ancestors of KIRBy's\n"
    "% `greedy` and `performance-weighted` families.\n"
    "% TODO: reconcile with the 13 strategies above -- these two are now a subset.\n" + P(810, 847))

add("\n\\subsubsection{Search configuration and internal evaluation}\n")
add("% [KB 161] the matched internal evaluator.\n"
    r"""Features that improve predictions under a random forest may not help a kNN model, and vice versa. This is because variable importance depends fundamentally on the model, as Dong and Rudin showed \citep{Dong2020}. To address this, the internal evaluator scores each candidate configuration with a model whose inductive bias matches the downstream model. We use five evaluators: a matched evaluator that uses the same model class as the downstream model, and four fixed evaluators with different inductive biases (random forest, SVM, LightGBM, MLP). We fit each on the training fold and score it on the held-out fold. The matched evaluator ensures that features are selected jointly with the model that will use them.""")
add("\n% [KB 163] the combinatorial search and the Blueprint.\n"
    r"""The full search tests all combinations of budgets, allocations, importance methods, downstream models, and internal evaluators under 5-fold cross-validation: about 15,000 configurations per dataset when using 7 downstream models. This combinatorial design is deliberate: it allows us to measure which factors contribute most to performance variance (model choice, budget, allocation, or their interactions) rather than assuming any single axis dominates. We cap models sensitive to high dimensionality at 2,000 or 3,000 features depending on model type. The complete search result is cached to disk. The search runs once; everything after that is a lookup. Each search yields a Blueprint recording all parameters (source names, allocation strategy, importance method, budget, evaluator, metric) required to reproduce the hybrid on new data without re-running the search.""")
add("\n" + r"\todo[inline, color=yellow!40]{Figure 5.1: the search pipeline schematic. Does not exist yet.}")

add("\n\\subsection{Blueprints and explainability}\n")
add(outline(
    "5.4  Blueprints and explainability",
    "A completed search is stored as a compact, reusable recipe, the blueprint, holding exactly:",
    "  the representations, the allocation method, the importance method, the feature budget, the",
    "  internal evaluator, the step size, the number of folds, the seed, the metric, the task",
    "  type, the augmentation strategy, and the per-representation scores. Because the blueprint",
    "  stores the recipe rather than the data, it can be saved and reapplied to new molecules.",
    "The blueprint does not store the selected feature indices; those are recovered when the",
    "  blueprint is refitted on data, which re-runs selection deterministically from the recipe",
    "  and seed (a saved blueprint is the recipe, the fitted selection is a separate object).",
    "Named-feature attribution is real: a feature-name registry resolves each selected column",
    "  back to its representation and a named feature, including ECFP and MACCS bits mapped to",
    "  SMARTS substructures, so a prediction traces back to specific, named features. Learned",
    "  embedding dimensions, where the registry has no chemical name, fall back to the bare",
    "  representation name plus the dimension index, so attribution is only as rich as the",
    "  registry. Figure 5.2.",
))
add(r"\todo[inline, color=yellow!40]{Write Sec 5.4 from the outline bullets. No source prose exists.}")

add("\n\\subsection{Default settings}\n")
add(outline(
    "5.5  Default settings",
    "The design experiments that fix the defaults: which allocation methods and",
    "  feature-importance methods to keep (Altmann et al. 2010 document the bias of naive",
    "  permutation importance), whether the internal evaluator should match the downstream model,",
    "  and how aggressively the search can be pruned. The current defaults are a top-3",
    "  performance-weighted allocation, lightgbm_gain importance, a step size of 5, and 5-fold",
    "  cross-validation; pruning is handled by an adaptive multi-stage schedule that drops the",
    "  worst allocation-importance pairs per model after the first fold, with a floor that keeps",
    "  a best configuration for every model, rather than by separate speed tiers.",
    "Results to be added as these are finalized; Table 5.1.",
))
add("\n% [KB 217] the adaptive pruning thresholds, from Supplementary Methods.\n"
    r"""Adaptive pruning: allocation strategies are evaluated on fold 0 and pruned if they appear in the top-3 in fewer than a threshold fraction of (importance, evaluator, model, budget) groups. The primary hybrid searches (no augmentation and uncapped augmentation across general and top-tier representation sets) use a 15\% threshold. Capped augmentation searches and standalone experiment-specific searches use a more aggressive 25\% threshold to reduce computational cost; this higher threshold prunes more aggressively from the candidate allocation strategies without affecting the quality of the surviving strategies.""")

add("\n\\subsection{Evaluation metrics, models, and splitting}\n")
add("% [KB 167] domain-matched primary metrics.\n"
    r"""Different biological domains have established different conventions for what constitutes a good prediction, and applying a single metric everywhere would misrepresent performance in at least some of them. To this end, we assign each task a primary metric matched to its domain: RMSE for molecular property prediction, Spearman rank correlation for antibody regression and developability, AUROC for binary classification (including antibody classification and tumor subtype), weighted F1 for multiclass classification, concordance index for drug-target affinity, and $R^2$ for reaction yield.""")
add("\n% [KB 175] the downstream model pool.\n"
    r"""For a fair comparison, the downstream model pool must cover a range of inductive biases, from methods that degrade in high dimensions (kNN, SVM, GP) to tree ensembles that tolerate irrelevant features. Thirteen models are available: tree ensembles (random forest, XGBoost, LightGBM, CatBoost, NGBoost), kernel methods (SVM, Gaussian process), nearest-neighbor (kNN), linear models (Ridge regression for regression, logistic regression for classification, both with cross-validated regularization), and neural networks (single-layer MLP, three-layer DNN, MC-Dropout BNN for uncertainty quantification, and a multi-scale 1D CNN). Standard experiments use seven of these: random forest, XGBoost, LightGBM, kNN, SVM, Gaussian process, and MLP. We use the full set for model-matching and noise experiments, and skip models with cubic or higher scaling for large datasets: Gaussian processes above 6,000 samples, SVMs above 10,000. No per-dataset tuning is performed: all models use fixed hyperparameters throughout, so performance differences reflect the input, not the model configuration.""")
add("\n% [KB 181] no random splits. Aligns with the group's standing requirement.\n"
    r"""No random splits are used. Random splitting inflates performance estimates for molecular data by placing structurally similar compounds on both sides of the train/test boundary \citep{Yang2019, Wallach2018}, and position-level leakage in deep mutational scanning data allows models to memorise site-specific effects rather than learning generalisable sequence-fitness relationships \citep{Notin2023, Vieira2026}. Splits therefore match domain conventions and use 5-fold GroupKFold throughout.""")
add("\n" + todo(
    "[KB 183-191] gives the per-domain splitting protocols in full (Murcko scaffold groups;",
    "antibody inter-family by VH identity; DMS by mutation position mod 5 per ProteinGym;",
    "SKEMPI by PDB complex; Ginkgo pre-computed hierarchical clusters; reaction by template;",
    "cancer genomics by patient with subtype stratification). Carry them in as an itemized list,",
    "or defer to Appendix A. They are verbatim in KIRBy/PAPER_DRAFT.md.",
))
add("\n% [KB 191] the nested, group-aware internal split.\n"
    r"""The hybrid construction step uses a nested internal train/validation split within each training fold to score representations for feature allocation. This internal split is group-aware, using GroupShuffleSplit with the same grouping variable as the outer cross-validation, to prevent the allocation evaluator from seeing related samples on both sides of its split.""")
add("\n% [KB 195] statistical analysis.\n"
    r"""Comparing models tested under $k$-fold cross-validation requires care, because the training sets overlap between folds and a standard paired $t$-test overstates significance \citep{NadeauBengio2003}. Within-dataset comparisons therefore use the Nadeau-Bengio corrected paired $t$-test, which adjusts for this dependence. Cross-dataset comparisons use the Wilcoxon signed-rank test on paired per-dataset scores, since performance differences need not be normally distributed over heterogeneous tasks. When ranking more than two methods, we apply the Friedman test with Nemenyi post-hoc correction \citep{Demsar2006} to control the family-wise error rate. We report effect sizes as rank-biserial correlation for Wilcoxon tests and Kendall's $W$ for Friedman tests.""")

add("\n\\subsection{Implementation}\n")
add(outline(
    "5.6  Implementation",
    "Caching and reuse that make the search tractable, and reproducibility.",
    "Results note: the second-half results (Chapters 5 to 7) are pending a full re-run.",
    "  Earlier cached results were invalidated by a caching defect, since fixed, that combined",
    "  non-deterministic data splits, malformed cache keys, and unsafe concurrent cache writes;",
    "  the experiments are being re-run from scratch and the results will be added once the",
    "  re-run completes.",
))
add(r"\todo[inline, color=orange!40]{RESULTS CAVEAT: state the re-run explicitly, once, here, and cross-reference it from ch.6 and ch.7.}")

add("\n\\subsection{Conclusion}\n")
add(outline(
    "5.7  Conclusion",
    "The chapter has defined the adaptive hybrid representation, the search that builds it,",
    "  and the blueprint that stores it. Chapter 6 turns to when and why it helps.",
))

# =============================================================================
# CHAPTER 6
# =============================================================================
add(r"""
\pagebreak
\section{Why Hybrid Representations Help: Complexity, Explainability, and Extension}\label{chap:why}
""")
add(todo("Results pending a full re-run, as set out in Sec 5.6."))

add("\n\\subsection{Introduction}\n")
add("% [KB 15] the 'no single representation wins' evidence base, md -> tex.\n"
    r"""A comparison of 62,820 models in molecular property prediction benchmarks showed that classical RDKit2D descriptors matched or outperformed learned features from graph neural networks and pretrained language models on the majority of datasets \citep{Deng2023}. Descriptor-based models occupied 73\% of top-three positions on 11 benchmarks \citep{Jiang2021}, and in 1,310 ChEMBL assays, feedforward networks on ECFP fingerprints outperformed graph convolutions and SMILES-based recurrent models \citep{Mayr2018}. The input often matters more than the model. Yet we still lack a systematic way to exploit this. No single representation wins. The same pattern holds outside small-molecule chemistry: no single protein language model consistently achieves the highest performance on all tasks \citep{Kenlay2024, Schmirler2024}; for antibodies, general-purpose models sometimes outperform those built for the domain \citep{Kenlay2024}. In computational pathology, no individual foundation model dominates on clinical endpoints \citep{ElNahhas2025}, and in cancer genomics, integrating additional omics layers can negatively impact performance when the data types are not properly combined \citep{Steyaert2023}.""")
add("\n% [KB 17] the breadth of the representation landscape.\n"
    r"""For small molecules alone, the landscape now includes extended-connectivity fingerprints, physicochemical descriptor sets exceeding 1,600 dimensions, graph neural network embeddings \citep{Yang2019, Fang2022}, language models trained on billions of SMILES strings \citep{Ross2022}, image-based encodings (molecules rendered as 2D images) \citep{Zeng2022}, and stereoelectronics-infused molecular graphs \citep{Boiko2025}. The same is true for proteins (ESM-2 \citep{Lin2023}, ProtTrans \citep{Elnaggar2022}), antibodies (AntiBERTy \citep{Ruffolo2021}, AbLang \citep{Olsen2022}), histopathology \citep{ElNahhas2025}, and multi-omics cancer profiles \citep{Steyaert2023}. A recent review counted over 40 molecular foundation models published between 2020 and 2024 \citep{Song2026}. Increasingly, new models appear faster than we can compare them \citep{Bommasani2021}. The choice is not getting simpler. Different sources presumably encode distinct aspects of the same biology. Noisy or incomplete on their own, single data modalities somehow give more sensitive and robust predictions when combined with independent signals from other modalities \citep{Steyaert2023}. An ensemble of frozen embeddings from the top four pathology foundation models, El Nahhas et al.\ found, achieved a mean AUROC 1.2\% higher than the best individual model on 31 clinical tasks \citep{ElNahhas2025}.""")
add("\n" + outline(
    "6.1  Introduction",
    "When does a hybrid representation beat the best single representation, what is the",
    "  mechanism, can its feature choices be explained and shown stable, and does it keep",
    "  working as new embeddings arrive and when data is held by separate parties.",
    "Review of existing approaches: Niazi and Mariam 2023 (high-dimensional redundancy and PCA",
    "  as the standard response), Xie et al. 2020 (conjoint fingerprints), and Valsecchi et al.",
    "  2020 (prediction-level consensus).",
))

for title, sec, bullets, figs in [
    ("When it helps", "6.2", [
        "The core question: does the hybrid beat the best single representation, and does the",
        "  optimal feature budget vary by downstream model? A model-by-budget grid over the cached",
        "  search answers both, with a two-way ANOVA over model, budget, and their interaction.",
        "  Figure 6.1 (model-by-budget heatmap) and Table 6.1 (best-per-model delta against the best",
        "  single representation).",
        "The headline is the cross-domain summary: across small-molecule, antibody, and",
        "  gene-expression tasks, does the hybrid win consistently? Each task contributes a paired",
        "  per-fold delta against the best single representation, combined with a Wilcoxon test and",
        "  per-domain win rates. Figure 6.2 (forest of per-task deltas with the Wilcoxon result) and Table 6.2.",
    ], "Figures 6.1, 6.2; Tables 6.1, 6.2"),
    ("Why it helps", "6.3", [
        "The mechanism comparison isolates \"more features chosen well\" from \"any compression to the",
        "  same width\" by running one model on three matched-dimensionality inputs: the best single",
        "  representation, a PCA-of-concatenation, and the hybrid. The PCA control is the ablation",
        "  reviewers will expect. Figure 6.3.",
        "The more interesting angle is information-theoretic: the mutual information with the target",
        "  per selected dimension as the budget grows, against random selection and a",
        "  matched-dimensionality null. If the hybrid packs more target-relevant information per",
        "  dimension than the alternatives, that is the mechanism. mRMR (Peng, Long and Ding 2005) and",
        "  RFE (Guyon et al. 2002) are named as the classical alternatives the approach is positioned",
        "  against, not as implemented baselines. Figure 6.4.",
    ], "Figures 6.3, 6.4"),
    ("Explainability and stability", "6.4", [
        "SHAP on the best hybrid, mapped through the Chapter 5 feature-name registry, aggregates",
        "  importance by source representation and traces predictions to named features, with the",
        "  caveat that learned-embedding dimensions resolve only to a name plus a dimension index.",
        "  Figure 6.5.",
        "A feature-set stability analysis (Nogueira's index with a confidence interval, plus",
        "  fold-to-fold Jaccard) checks that the selection is consistent across folds. It is an",
        "  experiment-level analysis, not a core-engine output. Table 6.3.",
    ], "Figure 6.5; Table 6.3"),
    ("Breadth across domains", "6.5", [
        "To make the \"applied well beyond molecular similarity\" claim concrete, the method is",
        "  carried across small molecules, antibodies (Trastuzumab-HER2 binding, Ginkgo",
        "  developability), protein affinity (Graphinity antibody-antigen ddG), and gene-expression",
        "  (the TCGA foundation-model comparison), each under its own splitting protocol. Whether the",
        "  advantage holds is summarised in one aggregate comparison. Figure 6.6.",
    ], "Figure 6.6"),
    ("Forward compatibility: absorbing new embeddings", "6.6", [
        "The mechanism is light: a new embedding is added by registering an embed function and one",
        "  dispatch line, after which the search gives it budget only if its features are selected.",
        "  The interesting test is whether a new embedding earns budget above two",
        "  matched-dimensionality null controls, a same-dimensionality random Gaussian matrix and a",
        "  row-shuffled copy, which separate genuine signal from dimensionality and content;",
        "  significance via Wilcoxon per dataset and Friedman across datasets. A second strand tests",
        "  whether a saved blueprint transfers across folds and, where dimensions line up, across",
        "  datasets. The symmetric direction, whether the hybrid improves a weak new embedding, is",
        "  future work (Section 8.2). Figure 6.7 and Table 6.4.",
    ], "Figure 6.7; Table 6.4"),
    ("Federated data (with Lhasa)", "6.7", [
        "A smaller, applied section: is sharing only which features matter competitive with",
        "  ensembling each party's separate model on a shared hERG benchmark, including under unequal",
        "  per-party noise? The shareable anchor splits hERG into clustered silos with label noise",
        "  rising across silos, and contrasts privacy-preserving feature-importance sharing against",
        "  similarity-weighted, performance-weighted, and distillation ensembles. A real-source variant",
        "  repeats the contrast on public ChEMBL plus two proprietary Lhasa silos, reported under a",
        "  data-availability caveat (the Lhasa data is not redistributable and the comparison depends",
        "  on Lhasa's FLuID code). A demonstration, not a new method. Figure 6.8 and Tables 6.5 and 6.6.",
        "Review of existing approaches: McMahan et al. 2017 (FedAvg, sharing model updates not raw",
        "  data), the MELLODDY consortium (Heyndrickx et al. 2023, cross-pharma federated QSAR), and",
        "  FL-QSAR (Chen et al. 2020).",
    ], "Figure 6.8; Tables 6.5, 6.6"),
]:
    add(f"\n\\subsection{{{title}}}\n")
    add(outline(f"{sec}  {title}", *bullets))
    add(r"\todo[inline, color=yellow!40]{" + sec + ": " + figs + ". Results pending re-run.}")

add("\n% [KB 133] the discussion paragraph on why compression helps -- information bottleneck.\n"
    "% TODO: this is Discussion material; it may belong in Sec 6.3 (Why it helps) or ch.8.\n"
    r"""Concatenation, the most common method for combining feature sets, gives the full pool that KIRBy starts from, but on biological sample sizes it can degrade rather than help \citep{Bellman1961, Stahlschmidt2022}. What late fusion gains by training separate models per source \citep{Steyaert2023} it loses in cross-source feature interactions. Those interactions can be recovered through intermediate fusion (MOGONET \citep{Wang2021}, cross-attention \citep{Zhang2024}), but only at the cost of retraining whenever a new source appears. At the kernel level, multiple kernel learning \citep{Gonen2011} (e.g., pairwiseMKL for drug bioactivity \citep{Cichonska2018}) can learn source weights, but no interpretable feature-level selections result. Consensus methods \citep{Valsecchi2020} combine predictions, not features, and so cannot exploit cases where one source's noise is another's signal. Instead, KIRBy works at the feature level: it selects features that span source boundaries, and because any new encoder's output is just another matrix in the pool, the approach is modular by construction. Perhaps the closest analogy in the optimization literature is the CASH problem \citep{Thornton2013} (the joint selection of algorithm and hyperparameters), extended here to include the input encoding as an additional search dimension. Tishby and Zaslavsky \citep{Tishby2015} showed why compression helps: the information bottleneck principle states that optimal representations maximize mutual information with the target while minimizing mutual information with the input. In effect, the budget forces the model to keep only what matters. The feature count controls the trade-off between informativeness and dimensionality.""")
add("\n% [KB 135] selection-overfitting controls. Belongs with Sec 6.4 (stability).\n"
    r"""Ambroise and McLachlan \citep{Ambroise2002} showed the extreme case of feature selection overfitting: an SVM achieved near-zero cross-validated error on random labels when gene selection was performed on the full dataset before cross-validation, but when selection was nested inside each fold the median error was 11 out of 40, close to the 50\% expected by chance. Vabalas et al.\ \citep{Vabalas2019} showed that the bias introduced by pre-selection can exceed the bias from hyperparameter tuning, and Cawley and Talbot \citep{Cawley2010} argued that feature selection is itself a form of model selection whose complexity cost must be accounted for. KIRBy follows the standard remedy: we compute all importance scores within each cross-validation fold, on training data only. But the problem of ``search degrees of freedom'' \citep{Tibshirani2015} remains. Searching over many possible feature subsets adds complexity even when each individual selection is unbiased. The budget acts as a hard capacity constraint: selecting 100 features from a pool of 5,000 limits the space of possible selections far more tightly than $L_1$ regularization over the same pool, because the feasible set grows polynomially in the budget while the continuous parameter space of a regularized model is unbounded \citep{Ng2004}. Important here is a distinction: the features being selected are frozen embeddings, not parameters being fitted. Choosing which pre-existing knowledge to trust is, we think, a less expressive operation than learning new features, and Kumar et al.\ \citep{Kumar2022} showed that frozen embeddings fed through a linear probe can outperform full fine-tuning by up to 10\% in out-of-distribution accuracy precisely because fine-tuning distorts pretrained structure. And stability can be measured directly: the Nogueira index \citep{Nogueira2018} quantifies the consistency of selected feature sets between folds. If selection were overfit, feature sets would differ wildly from fold to fold.""")

add("\n\\subsection{Conclusion}\n")
add(outline(
    "6.8  Conclusion",
    "Having tested when the hybrid representation helps, why, whether it can be explained, and",
    "  how it extends, Chapter 7 returns to the problem that began the second half.",
))

# =============================================================================
# CHAPTER 7
# =============================================================================
add(r"""
\pagebreak
\section{Molecular Similarity and Label Noise}\label{chap:similarity}
""")
add(todo("Results pending a full re-run, as set out in Sec 5.6."))
add("\\subsection{Introduction}\n")
add(outline(
    "7.1  Introduction",
    "Recalling Definition 2.2 and the motivation of Chapter 5, this chapter measures how much",
    "  the notion of similarity depends on the representation, and whether the hybrid",
    "  representation gives a more property-aware similarity, then uses that to revisit the",
    "  noisy-label question Chapter 4 left open.",
    "Review of existing approaches: van Tilborg et al. 2022 and Maggiora et al. 2014",
    "  (similarity depends on the representation), Aggarwal et al. 2001 (high-dimensional",
    "  distance concentration), and, for noise cleaning, Northcutt et al. 2021 (Confident",
    "  Learning), Brodley and Friedl 1999 (instance-selection filtering), and Wilson 1972",
    "  (edited nearest neighbours).",
))

add("\n\\subsection{Similarity and the representation}\n")
add(outline(
    "7.2  Similarity and the representation",
    "The premise is that similarity is representation-dependent: for each representation, the",
    "  similarity-activity correlation (Spearman of pairwise distance against absolute property",
    "  difference, with a permutation test) measures how well \"similar\" tracks \"similar property\".",
    "  Figure 7.1 (spread of the correlation across representations).",
    "The central positive claim is that the hybrid's importance-weighted distance gives a more",
    "  property-aware similarity than the best single representation or a plain concatenation,",
    "  reported as a delta in that correlation and aggregated across datasets with a Wilcoxon test.",
    "  Figure 7.2 and Table 7.1.",
))
add("\n\\subsubsection{Distance metrics for continuous molecular descriptors}\n")
add("% [CR 939-952] the eight distance metrics, carried whole.\n" + P(939, 952))
add("\n\\subsubsection{Feature importance weighting}\n")
add("% [CR 954-956] importance-weighted distance -- the 'importance-weighted distance' the\n"
    "% outline's Sec 7.2 makes its central positive claim about.\n" + P(956))

add("\n\\subsection{Noise estimation and mitigation inside the method}\n")
add(outline(
    "7.3  Noise estimation and mitigation inside the method",
    "The flagship validates the noise estimators against real assay noise rather than injected",
    "  noise: on BindingDB, the per-(drug,target) replicate standard deviation of measured affinity",
    "  is the ground-truth noise signal, and the question is whether each estimator ranks samples in",
    "  agreement with that real disagreement, in the hybrid space versus the alternatives. This is",
    "  the chapter's external-validity anchor. Figure 7.3.",
    "Supporting experiments: a Gaussian-process leave-one-out test of whether the hybrid space",
    "  separates genuine noise from activity cliffs (Figure 7.4); a remediation experiment measuring",
    "  how much clean-data performance is recovered by downweighting, label smoothing, and neighbour",
    "  smoothing (Figure 7.5); and the corrected learning-with-noisy-labels survey, the matured,",
    "  oracle-leak-free version of the exploratory Chapter 5 study, re-searching the feature mix per",
    "  noise level (Table 7.2).",
    "The conceptual hinge between the chapter's two halves: neighbour-based noise detectors built",
    "  on the similarity assumption mistake activity cliffs for noise, and the question is whether",
    "  the hybrid distance reduces that contamination relative to single-representation baselines.",
    "  Table 7.3.",
    "Calibration of the hybrid across feature budgets would tie the uncertainty thread of",
    "  Chapter 4 to the method; the conformal machinery exists (MAPIE split and cross-conformal, and",
    "  a Mondrian applicability-domain regressor) but the budget-sweep figure is planned, not yet",
    "  wired. Figure 7.6.",
))
add("\n\\subsubsection{Sample removal methods}\n")
add("% [CR 849-861] why removal was abandoned for weighting/smoothing; kNN filtering; LOF.\n" + P(851, 861))
add("\n\\subsubsection{Noise magnitude estimation}\n")
add("% [CR 863-906] neighbour-based, model-based, and uncertainty-based estimators.\n"
    "% TODO(xref): [CR 906] 'as discussed in Chapter 3' -> now ch.4 (uncertainty).\n"
    "% TODO: [CR 863] carries the author's own '% TODO: get sources' marker. Left in place.\n" + P(866, 906))
add("\n\\subsubsection{Label smoothing and sample weighting}\n")
add("% [CR 908-935] proportional/threshold/soft-threshold/confidence smoothing; inverse,\n"
    "% exponential and soft weighting.\n" + P(910, 935))
add("\n\\subsubsection{Recovery rate}\n")
add("% [CR 972-982] the evaluation protocol and the recovery-rate definition.\n" + P(974, 982))
add("\n" + todo(
    "NOT carried over from [CR 985-1146], the confirmation report's hybrid results:",
    "  hybrid construction outcomes (tab:hybrid_both), smoothing-vs-weighting",
    "  (tab:smooth_vs_weight), the estimator table (tab:estimators), the per-representation",
    "  neighbour-based results, and the alpha sweep.",
    "The outline says the second-half results are pending a full re-run (Sec 5.6), and the",
    "  confirmation report's numbers came from the cache that was later invalidated. Recover from",
    "  Confirmation_Report/paper.tex lines 985-1146 if any of it survives the re-run.",
    "Also not carried: [CR 958-970], the QM9 hybrid-selection protocol (5,000 samples,",
    "  sigma in {0.3,0.6}, PDV+MHG-GNN budget=100 greedy allocation) -- superseded by the",
    "  ch.5 search; and [CR 1156], which is a cross-reference to the old chapter numbering.",
))

add("\n\\subsection{Conclusion}\n")
add(outline(
    "7.4  Conclusion",
    "Honest limits: where comparisons are best-case rather than fully blind, stated plainly,",
    "  with the fair comparison foregrounded. This closes the loop opened in Chapter 4; the",
    "  concluding chapter develops the open directions and draws the argument together.",
))
add("\n% [CR 1144,1146] the confirmation report's own honest limits on neighbour-based noise\n"
    "% estimation and on hybrids. Survives the re-run because it is a qualitative claim.\n"
    "% TODO(xref): 'conclusions of Chapter 4' -> the LNL study, now Sec 5.1.\n" + P(1144) + "\n\n" + P(1146))

# =============================================================================
# CHAPTER 8
# =============================================================================
add(r"""
\pagebreak
\section{Discussion and Future Directions}\label{chap:discussion}
""")
add("\\subsection{Conclusions}\n")
add(outline(
    "8.1  Conclusions",
    "The single line of the thesis: under label noise the model governs robustness and the",
    "  representation governs clean performance and the meaningfulness of uncertainty and",
    "  similarity; adapting the representation then makes prediction more robust, similarity",
    "  more property-aware, and noise estimation more reliable, while staying interpretable and",
    "  extendable.",
    "Contributions, across three distinct software artifacts and their findings: the",
    "  NoiseInject benchmarking framework (released, MIT) and the noise-robustness and",
    "  uncertainty findings of Chapters 3 and 4; the standalone Rust and Python",
    "  chemical-processing and precompute pipeline; and the KIRBy research codebase (pure",
    "  Python), comprising the adaptive hybrid representation and its blueprint.",
    "Closing remarks on what this means for choosing representations and models in noisy settings.",
))
add(r"\todo[inline, color=yellow!40]{Write Sec 8.1. Draw on [NI 586], already used to close Sec 4.6 -- do not simply repeat it.}")

add("\n\\subsection{Limitations}\n")
add("% [KB 137] the KIRBy limitations paragraph, md -> tex. Wording unchanged apart from\n"
    "% 'KIRBy' -> 'the method' where it reads as a product name mid-sentence.\n"
    r"""We want to be clear about KIRBy's limitations. For small datasets (below about 200 samples), the hybrid search cannot reliably outperform the best single source. We suspect this is because the internal validation set within each fold shrinks to roughly 20 samples, and at that size the evaluator cannot distinguish genuine improvements from noise. Tuning cannot fix this. And at the other extreme, frozen embeddings have a ceiling. No signal can be added beyond what the inputs already contain, and on tasks where the true signal requires end-to-end learning from raw inputs (certain 3D quantum-mechanical properties, for example, where SchNet outperforms any frozen embedding by a wide margin), this ceiling binds. Although the methodology naturally accommodates embeddings from sequence-trained models (ESM-2, ProtBERT, etc.), encoding raw sequences directly as one-hot or $k$-mer features disperses the positional and compositional information that makes sequences useful in the first place. Integrating 20 to 50 foundation models into a single pipeline, each with its own dependencies, GPU requirements, and version constraints, is genuinely difficult engineering. Embeddings need only be computed once per dataset and everything is cached. Yet the initial integration burden is real and scales with the number of models in the pool. Finally, on endpoints with high experimental noise, even a well-constructed hybrid may not improve on simpler baselines. Kramer et al.\ \citep{Kramer2012} documented that experimental uncertainty in heterogeneous public $K_i$ data is approximately 0.54 log units. This sets a natural accuracy ceiling of $R^2 \approx 0.81$ regardless of model sophistication; the budget constraint plausibly guards against overshoot, but it cannot push past the noise floor.""")

add("\n\\subsection{Future directions}\n")
add(outline(
    "8.2  Future directions",
    "Future work could make forward compatibility symmetric: testing whether the hybrid",
    "  representation improves a weak new embedding, not only whether the embedding improves the",
    "  hybrid set. This is a proposed direction, not a result of this thesis.",
    "Carrying a blueprint across domains, and across noise distributions beyond Gaussian.",
    "A fuller federated deployment with Lhasa, and extending the noise-estimation work from",
    "  regression to classification endpoints.",
))
add("\n% [CR 1150,1152,1154] the confirmation report's Next Steps. Much of this is now DONE\n"
    "% (TreeSHAP/KernelSHAP/gradient attribution are implemented; the complexity trade-off is\n"
    "% ch.6). Kept as raw material.\n"
    "% TODO: prune the parts that ch.5-7 now deliver.\n"
    "% TODO(xref): [CR 1150] ends 'small-loss selection, as discussed in Chapter 4'. Small-loss\n"
    "% selection is an LNL method; the LNL study is now Sec 5.1, NOT the new ch.4 (which is\n"
    "% Uncertainty Quantification). Retarget to Sec 5.1.\n"
    "% TODO(xref): [CR 1156] 'Chapters 4 and 5' / 'Chapter 3' -- old numbering throughout.\n"
    + P(1150) + "\n\n" + P(1152) + "\n\n" + P(1154))
add("\n% [KB 131] what this means in practice -- the forward-compatibility argument.\n"
    r"""What does this mean in practice? A new foundation model appears every few months in molecular, protein, or antibody modeling. A researcher using KIRBy does not need to decide whether to replace their current encoding or retrain their pipeline; they compute the new model's embeddings, add the matrix to the pool, and re-run the search. The search determines whether the new embedding contributes useful features and, if so, how many and which ones. In a field where encoding choice is often determined by familiarity or lab convention rather than by systematic comparison, this is a substantive change. It replaces a single bet with a principled combination, tailored to each task, that updates as the pool grows.""")

# =============================================================================
# BACK MATTER
# =============================================================================
add(r"""
\pagebreak
\section*{Summary of Research Contributions}
""")
add(outline(
    "SUMMARY OF RESEARCH CONTRIBUTIONS",
    "A short closing list (following the peer theses): the NoiseInject paper and the planned",
    "  hybrid-representation papers, the UK-QSAR conference presentation, the OPIG Blopig posts,",
    "  and the three distinct open-source software artifacts (NoiseInject, MIT, released; the",
    "  standalone Rust and Python chemical-processing and precompute pipeline, to be simplified",
    "  and released separately; and the KIRBy research codebase, pure Python).",
))
add("\n% [CR 1159-1179] research papers, conference presentations, blogposts.\n"
    "% TODO: the second paper's title still says 'Improving noise prediction robustness and\n"
    "% mitigation with feature-importance hybrid representations' -- the outline now plans TWO\n"
    "% hybrid papers (ch.6 and ch.7). Update.\n"    "% TODO(xref): [CR 1161] 'based on the material in Chapter 3' -- per the outline the\n"
    "% NoiseInject paper now spans Chapters 3 AND 4. [CR 1163] 'the experiments in Chapter 5'\n"
    "% now spans Chapters 5-7.\n"
    "% TODO: the outline wants the three software artifacts listed here too.\n" + P(1159, 1179))
add("\n" + todo(
    "[CR 1181-1187] 'DPhil Teaching' and 'External Work' are NOT carried over. Neither peer",
    "thesis (Dablander, Moesser) has a teaching or extracurricular section, and Dablander's",
    "contributions section is purely academic. Recover from Confirmation_Report/paper.tex",
    "lines 1181-1187 if the department asks for a teaching record.",
))

add(r"""
%%========================= APPENDICES ======================================
""")
add(outline(
    "APPENDICES",
    "A: datasets and curation (Chapters 3 to 7): QM9, LogD, Caco-2, hERG, and the second-half",
    "   datasets, with per-dataset provenance, accession numbers, and DOIs.",
    "B: NoiseInject metric definitions, hyperparameters, and per-strategy",
    "   variance-decomposition tables (Chapters 3 to 4).",
    "C: the method's full search space and the design-experiment tables behind its defaults (Chapter 5).",
    "D: extended result tables for the second-half chapters (Chapters 6 to 7).",
    "E: data and code availability (all chapters), covering the three software artifacts,",
    "   the dataset licences, and the Lhasa/Effiris redistribution caveat.",
))
add(r"""
\appendix
""")

add(r"\section{Datasets and curation}\label{app:datasets}")
add(outline("APPENDIX A",
    "A: datasets and curation (Chapters 3 to 7): QM9, LogD, Caco-2, hERG, and the second-half",
    "   datasets, with per-dataset provenance, accession numbers, and DOIs."))
add("\n% [NI 642] her own dataset-availability paragraph.\n" + N(642))
add(r"\todo[inline, color=orange!40]{Appendix A: add per-dataset provenance, accession numbers, DOIs.}")

add(r"\section{NoiseInject supplementary material}\label{app:noiseinject}")
add(todo(
    "Chapters 3 and 4 originally cited the NoiseInject manuscript's twelve 'Additional file~N'",
    "supplements. Every one of those references has been rewritten to point at the matching",
    "subsection below. The subsections are stubs -- move the real content in.",
))
add(r"\subsection{Default hyperparameters for all models evaluated in this study}\label{app:af1}")
add(r"\todo[inline, color=orange!40]{Appendix B.1: was Additional file 1 of the NoiseInject manuscript. Titles carried verbatim from noise\_inject.tex:668.}")
add(r"\subsection{Pairwise Spearman rank correlations between model NDS profiles}\label{app:af2}")
add(r"\todo[inline, color=orange!40]{Appendix B.2: was Additional file 2 of the NoiseInject manuscript. Titles carried verbatim from noise\_inject.tex:669.}")
add(r"\subsection{Pairwise Spearman rank correlations between representation NDS profiles}\label{app:af3}")
add(r"\todo[inline, color=orange!40]{Appendix B.3: was Additional file 3 of the NoiseInject manuscript. Titles carried verbatim from noise\_inject.tex:670.}")
add(r"\subsection{Intraclass correlation coefficients (ICC(1,1)) for model pairs}\label{app:af4}")
add(r"\todo[inline, color=orange!40]{Appendix B.4: was Additional file 4 of the NoiseInject manuscript. Titles carried verbatim from noise\_inject.tex:671.}")
add(r"\subsection{Configurations excluded from robustness analysis}\label{app:af5}")
add(r"\todo[inline, color=orange!40]{Appendix B.5: was Additional file 5 of the NoiseInject manuscript. Titles carried verbatim from noise\_inject.tex:672.}")
add(r"\subsection{Global overview of noise robustness on the ECFP4 representation}\label{app:af6}")
add(r"\todo[inline, color=orange!40]{Appendix B.6: was Additional file 6 of the NoiseInject manuscript. Titles carried verbatim from noise\_inject.tex:673.}")
add(r"\subsection{Baseline R$^2$ versus NDS scatter (PDV, Gaussian)}\label{app:af7}")
add(r"\todo[inline, color=orange!40]{Appendix B.7: was Additional file 7 of the NoiseInject manuscript. Titles carried verbatim from noise\_inject.tex:674.}")
add(r"\subsection{Neural-network Bayesian transformation effects by representation}\label{app:af8}")
add(r"\todo[inline, color=orange!40]{Appendix B.8: was Additional file 8 of the NoiseInject manuscript. Titles carried verbatim from noise\_inject.tex:675.}")
add(r"\subsection{Uncertainty quantification metrics for probabilistic models on the ECFP4 representation}\label{app:af9}")
add(r"\todo[inline, color=orange!40]{Appendix B.9: was Additional file 9 of the NoiseInject manuscript. Titles carried verbatim from noise\_inject.tex:676.}")
add(r"\subsection{ANOVA variance decomposition on external validation datasets}\label{app:af10}")
add(r"\todo[inline, color=orange!40]{Appendix B.10: was Additional file 10 of the NoiseInject manuscript. Titles carried verbatim from noise\_inject.tex:677.}")
add(r"\subsection{RF vs QRF on validation datasets}\label{app:af11}")
add(r"\todo[inline, color=orange!40]{Appendix B.11: was Additional file 11 of the NoiseInject manuscript. Titles carried verbatim from noise\_inject.tex:678.}")
add(r"\subsection{Default hyperparameters used for external validation experiments}\label{app:af12}")
add(r"\todo[inline, color=orange!40]{Appendix B.12: was Additional file 12 of the NoiseInject manuscript. Titles carried verbatim from noise\_inject.tex:679.}")

add(r"\section{The search space and its defaults}\label{app:searchspace}")
add(outline("APPENDIX C",
    "C: the method's full search space and the design-experiment tables behind its defaults (Chapter 5).",
    "   The full vocabulary: downstream models (rf, xgboost, ridge, knn, lightgbm,",
    "   mlp, svm, gp, with optional catboost, dnn, cnn, bnn, ngboost); feature budgets (roughly 10",
    "   to 2,000, per-model capped); allocation methods (greedy, performance-weighted with its",
    "   weighting schemes, dominant-greedy, fixed, two-stage, stability-selection); importance",
    "   methods (treeshap, xgboost_gain, lightgbm_gain, permutation, kernelshap, deeplift,",
    "   integrated_gradients, gradientshap); internal evaluators (match, or rf/svm/lightgbm/mlp);",
    "   augmentation strategies; and the exact blueprint fields."))
add(r"\todo[inline, color=orange!40]{Appendix C is unwritten.}")

add(r"\section{Extended result tables}\label{app:results}")
add(outline("APPENDIX D",
    "D: extended result tables for the second-half chapters (Chapters 6 to 7), organised by the",
    "   featured experiments: the cross-domain summary; the information-theoretic decomposition;",
    "   the explainability and attribution tables; the forward-compatibility tables; the federated",
    "   comparator tables; the similarity-activity correlation tables; the BindingDB",
    "   ground-truth-noise table; and the remediation and noise-estimation tables."))
add(r"\todo[inline, color=orange!40]{Appendix D is unwritten.}")

add(r"\section{Data and code availability}\label{app:availability}")
add(outline("APPENDIX E",
    "E: data and code availability (all chapters). The full statement covers three distinct",
    "   software artifacts. (1) NoiseInject: the open-source benchmarking framework, MIT, at",
    "   github.com/adpunt/noise_inject, archived on Zenodo; released. (2) The standalone Rust and",
    "   Python chemical-processing and precompute pipeline. (3) The KIRBy research codebase: pure",
    "   Python. KIRBy contains no Rust; it operates on precomputed numpy feature matrices."))
add("\n% [NI 629-640] her NoiseInject availability statement + the Springer-style itemised\n"
    "% block the outline asks for. Two more such blocks are needed (precompute pipeline, KIRBy).\n" + N(629, 640))
add("\n% [NI 654] authors' contributions, for the front-matter declaration.\n" + N(654))
add(r"\todo[inline, color=orange!40]{Appendix E: add itemised blocks for the Rust/Python precompute pipeline and for KIRBy, plus the Lhasa/Effiris redistribution caveat.}")

# =============================================================================
text = "\n".join(B) + "\n"

# --- missing-citation header block (empirically verified against a real build) ---
_CITE_BLOCK = "%%    COMPILE-VERIFIED (pdflatex + bibtex, plainnat): 51 cited keys render as [?].\n%%\n%%    (a) 2 exist in refs.bib but under a DIFFERENT CASE. BibTeX resolves the lookup\n%%        case-insensitively, but natbib matches the emitted \\bibitem label case-SENSITIVELY,\n%%        so these silently print [?]. Fix by matching the case:\n%%           \\citep{Lakshminarayanan2017}  ->  refs.bib entry is  lakshminarayanan2017\n%%           \\citep{Ross2022}  ->  refs.bib entry is  ross2022\n%%\n%%    (b) 49 have no entry in refs.bib at all:\n%%           Ambroise2002            Bellman1961             Boiko2025               Bommasani2021\n%%           Cawley2010              Cichonska2018           Demsar2006              Dong2020\n%%           Elnaggar2022            ElNahhas2025            FernandezDelgado2014    Gonen2011\n%%           Hughes1968              jaeger2018mol2vec       Jiang2021               jorner2021\n%%           ke2017lightgbm          Kenlay2024              Kumar2022               landrum2024\n%%           Lin2023                 Mayr2018                NadeauBengio2003        Ng2004\n%%           Nogueira2018            noiseinject             Notin2023               Olsen2022\n%%           openadmet               qm9_dataset             Ruffolo2021             Schmirler2024\n%%           Song2026                Stahlschmidt2022        Steyaert2023            Thornton2013\n%%           Tibshirani2015          Tishby2015              Vabalas2019             Valsecchi2020\n%%           Vapnik1995              Vieira2026              Wallach2018             Walters2023\n%%           Wang2021                Yang2019                Zdrazil2023             Zeng2022\n%%           Zhang2024\n%%\n%%    Nearly all of (b) arrive with the KIRBy chapters from PAPER_DRAFT.md; their entries\n%%    live in KIRBy/PAPER_REFERENCES.md. The rest arrive with the updated NoiseInject text\n%%    and its availability statement: jorner2021, ke2017lightgbm, landrum2024, noiseinject,\n%%    openadmet, qm9_dataset, lhasa, Vapnik1995, Walters2023, Zdrazil2023, and\n%%    jaeger2018mol2vec (= refs.bib's existing Jaeger2018)."
text = text.replace("@@MISSING_CITES@@", _CITE_BLOCK)


# ============================================================================
# POST-PROCESSING
# ============================================================================
import re as _r

# --- item 14: rewrite the journal's "Additional file~N" into appendix refs ---
_AF = []
def _af_plural(m):
    a, b = m.group(1), m.group(2)
    _AF.append(m.group(0)); return r"Appendices~\ref{app:af%s}--\ref{app:af%s}" % (a, b)
def _af_single(m):
    _AF.append(m.group(0)); return r"Appendix~\ref{app:af%s}" % m.group(1)
_lines = []
for _l in text.split("\n"):
    if not _l.lstrip().startswith("%"):
        _l = _r.sub(r"Additional files~(\d+)--(\d+)", _af_plural, _l)
        _l = _r.sub(r"Additional file~(\d+)", _af_single, _l)
    _lines.append(_l)
text = "\n".join(_lines)

# --- fix citation-key CASE to match refs.bib exactly ------------------------
# BibTeX resolves the lookup case-insensitively but emits one \bibitem, and natbib
# then matches it case-SENSITIVELY -- so a case mismatch silently prints [?].
# bibtex says: "Case mismatch error between cite keys Ross2022 and ross2022".
_CASE = {"Ross2022": "ross2022", "Lakshminarayanan2017": "lakshminarayanan2017"}
_nfix = 0
def _fixcase(m):
    global _nfix
    ks = [k.strip() for k in m.group(1).split(",")]
    out = []
    for k in ks:
        if k in _CASE:
            _nfix += 1; out.append(_CASE[k])
        else:
            out.append(k)
    return "\\" + m.group(0)[1:m.group(0).index("{")] + "{" + ", ".join(out) + "}"
_lines = []
for _l in text.split("\n"):
    if not _l.lstrip().startswith("%"):
        _l = _r.sub(r"\\cite[a-zA-Z]*\{([^}]+)\}", _fixcase, _l)
    _lines.append(_l)
text = "\n".join(_lines)

# --- item 1: chapter-closing heading is plural, as in both peer theses -------
text = text.replace("\\subsection{Conclusion}\n", "\\subsection{Conclusions}\n")

# --- item 10: article -> report. Promote the whole heading hierarchy. --------
text = _r.sub(r"\\pagebreak\n(\\section\{)", r"\1", text)          # chapters self-break
text = text.replace("\\section*{Summary of Research Contributions}", "@@SUMMARY@@")
text = _r.sub(r"^\\subsubsection\{", "@@SUB@@{", text, flags=_r.M)
text = _r.sub(r"^\\subsection\{",    "@@SEC@@{", text, flags=_r.M)
text = _r.sub(r"^\\section\{",       "@@CHAP@@{", text, flags=_r.M)
text = (text.replace("@@SUB@@{",  "\\subsection{")
            .replace("@@SEC@@{",  "\\section{")
            .replace("@@CHAP@@{", "\\chapter{"))
text = text.replace("@@SUMMARY@@",
  "\\chapter*{Summary of Research Contributions}\n"
  "\\addcontentsline{toc}{chapter}{Summary of Research Contributions}")
# the appendix chapters were authored as \section -> now \chapter. correct.
# the Summary's carried \subsection{Research Papers} etc. became \section. correct.

OUT.write_text(text)
print(f"\n=== item 14: rewrote {len(_AF)} 'Additional file' refs -> appendix refs ===")
for _x in sorted(set(_AF)):
    print("   ", _x)


# --- reporting ----------------------------------------------------------------
print(f"WROTE {OUT}  ({len(text.splitlines())} lines, {len(text)} bytes)")
print()
print("=== TEXT EDITS APPLIED (the only ones) ===")
for where, old, new in EDITS:
    print(f"  {where}: {old!r} -> {new!r}")
if not EDITS:
    print("  (none)")
print()


def coverage(name, lines, taken, ignore_before=0):
    body = [i for i in range(ignore_before + 1, len(lines) + 1)
            if lines[i - 1].strip() and not lines[i - 1].lstrip().startswith("%")]
    missed = [i for i in body if i not in taken]
    print(f"=== {name}: {len(taken & set(body))}/{len(body)} non-blank non-comment lines carried ===")
    # group missed into ranges
    ranges, start, prev = [], None, None
    for i in missed:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            ranges.append((start, prev)); start = prev = i
    if start is not None:
        ranges.append((start, prev))
    for a, b in ranges:
        snippet = lines[a - 1].strip()[:88]
        print(f"  DROPPED {a:>4}-{b:<4}  {snippet}")
    print()


coverage("paper.tex", p_lines, TAKEN["P"])
coverage("noise_inject.tex", n_lines, TAKEN["N"], ignore_before=190)
