# Final read-through findings, 2026-10-10 (rendered PDF, 8 readers)

Printed page numbers. Source file thesis.tex; chapters start at \chapter{...}\label{chap:intro|background|robustness|flaws|method|similarity|noise|discussion}. Abstract is in main.tex.

NOT TO TOUCH (author's decision): AI-use statement, line spacing, title-page term, the hybrid score differing between Chapter 5 tables (Table 5.14 vs 5.15/5.19/5.21/5.22), the whole-dataset pre-search used to cut models/budgets (leakage question, p96 vs p87). Never change a number inside a table. Never invent a result.

## Abstract / Ch1 / Ch8 (cross-cutting)
- A1 Abstract, Ch1 p5, Ch8 p213 and p216: "the hybrid stayed ahead (of the single representation) through every treatment". False: Table 7.30 hERG Ki network labels as delivered: second single 0.589 vs hybrids 0.597/0.612 (RMSE); Table 7.29 hERG Ki Gaussian network: first single 0.701 vs hybrids 0.720/0.718; Table 7.6 hERG flips no cleaning 0.806 vs 0.803. Ch7 p210/211 says "almost every". Make all places say what the tables show (e.g. "through almost every treatment", naming the hERG Ki exception) — consistent wording everywhere.
- A2 Abstract "most features ... do not vary or do not relate to the label, yet each still adds to the distance" and Ch8 p212 "most features ... barely vary ... A distance sums over every feature, so these features dominate it" contradict Ch4 (constant features add zero to Euclidean/Manhattan distance; distances computed with constant features removed). Fix abstract and Ch8 to Ch4's actual finding (features that vary but do not relate to the label add to the distance; constant ones add nothing / inflate shared-bit similarity — check Ch4 text for the exact claim).
- A3 Abstract "On every small-molecule dataset it predicted better than the best single representation": check scope against Ch5 Table 5.14 (Buchwald–Hartwig RF worse 14.4056 vs 15.2383 if reactions count). Make scope exact.
- A4 Abstract and Ch8 p212 "Predicted uncertainty ... did not identify the corrupted labels" / "no model's uncertainty identified the corrupted labels": Ch3 says under censoring correlations are large (−0.84 to −0.80) and error divided by uncertainty picked out clipped labels better than error alone. Add Ch3's qualifier ("outside censoring").
- A5 Ch8 p213 RQ1 answer covers antibodies only; Ch1 RQ asks about antibodies AND reactions; Ch1 says "On the reactions the hybrid gained nothing." Add the reaction answer.
- A6 Ch8 p214 "on hERG Ki one constant lift for every clipped label beat censored EM": Table 7.20 hERG Ki row lift 0.668 vs censored EM 0.664 (RMSE, EM better). Fix.
- A7 Ch8 p213 "The forest residual detector fell to chance under clipping": Table 7.15 three datasets below chance, SMAC ~double chance. Say as p190 does.
- A8 Ch8 p214 Contribution 2 cites "(Chapter 2)" (background) — check it should be Chapter 3 or wherever the result is.
- A9 Ch8 p215 status notes "Runs on DNA promoter sequences have started but are not reported here", "the code exists and has not been run" — rephrase as future work.
- A10 Ch8 p213 "on all but one small-molecule dataset with a result" — reword without "with a result" status flavour, keep the facts.
- A11 Summary of Research Contributions p217: two in-prep papers say "In preparation: ... A draft is in preparation, and its experiments will be extended before it is submitted." Remove the repetition and the "will be extended" clause. Author initials: "Morris, G.M." vs "Morris, G." — make consistent ("Morris, G.M.").
- A12 Ch1 p4 "LightGBM, XGBoost and the plain neural networks lost the most" vs Ch8 names only LightGBM and XGBoost. Check Ch3 and make consistent.
- A13 Ch1 p4 "We replicated systematic noise by adding it in disproportionate amounts to select scaffold families" — that's grouped-wider, which Ch3 p42 calls random error; systematic = grouped-shifted and censoring. Fix.
- A14 Ch1 p4 names "LightGBM, XGBoost and the plain neural networks" as "the models with the highest predictive performance"; Ch3 p49 GP highest on clean labels on assay datasets, p45 SVM first on QM9 ECFP4. Fix to what Ch3 says.
- A15 Ch1 p4: VBLL "correctly attributed error as aleatoric" — Ch3 p57 says not on every dataset/representation and not scored out of fold. Hedge to match.
- A16 Ch1 p3 garbled question "Second, what does the choice of model provide with label noise?" (author text: fix only if broken; it is broken grammar → silent fix allowed if trivial, else note).
- A17 Ch1 p6 "Lhasa also supplied the hERG data used in Chapters 4 and 5" — check whether hERG blocking (Lhasa set) is also used in Ch6/7; if so fix chapter list.
- A18 Ch1 p6 "(Additional file S.8)", "Additional file S.7" — a second numbering scheme (also Ch4 p73 "S.3" twice, Ch5 p99/p120 "S.7", "S.7.2"). Every other pointer is "Additional file N". Find the \ref targets; these are app:noise:sec:* labels from thesis_appendix (NoiseInject paper Additional files). Make them resolve to the same numbering as the rest, or state them as the NoiseInject paper's additional files. Report what you did.

## Ch2
- B1 p23 "LNL has three groups of methods, set out in Table 2.3 and Figure 2.3. Song et al. sort ... into five groups." (AI sentence from last pass.) Figure 2.3 shows one evaluation pipeline, not three groups; and 3 vs 5 groups unexplained. Rewrite: three groups are this thesis's grouping (Table 2.3), Song's five map onto them (say how, briefly, if Table 2.3 shows it); point to Figure 2.3 only for what it shows (how a mitigation method is evaluated). "LNL" never used again — drop the acronym if unused (also from abbreviations list in main.tex).
- B2 p26 Equation 2.6 ECE compares mean predicted SD per bin with mean absolute error per bin; for Gaussian errors E|e| = 0.80 σ, so a calibrated model gets ECE ≈ 0.2 ū. Fix definition to compare against RMSE per bin (or scale by sqrt(2/pi)) — ONLY if Ch3's computed ECE numbers are not affected in wording; if the code computed it with MAE, do NOT change the definition: instead add one sentence noting that under Gaussian errors a calibrated model scores ≈0.2ū, not 0. Check tests/ or noise_inject code? If unsure, add the note, don't change the equation.
- B3 p26 "m is a bounded metric for which higher values are better: R2" — R2 not bounded below. Fix wording ("bounded above by 1").
- B4 p17–18 "Bayesian methods such as Gaussian processes (GPs), and conformal prediction, do not make this assumption" reads as conformal = Bayesian; previous sentence "discrete quantities rather than distributions" should be point values. Fix.
- B5 p19 noise-as-regularization: cited results [95],[96] are input/activation noise, not label noise; paragraph ends "It is important to address this potential ..." and never does. Make the sentence say those results concern input noise; drop or resolve the dangling promise.
- B6 p25 Ch7 injector "adds Gaussian noise, with variance the level times the square of the unit" vs Eq 2.4 SD = level × spread. Check Ch7 methods/code (tests or src noise injector) and make consistent.
- B7 p11 repeated content (molecular graph sentence repeats p10). p12–13 repeated "mixed results". Cut the repeat (AI text only; if author text, leave).
- B8 p12 "contributory rather than consensus" trees paragraph uncited; "every data point influences the prediction at every other point" too strong. If AI text, soften; if author, leave and list.
- B9 p23 Table 2.3 "rows" → molecules; "scaffold group" → "scaffold family".
- B10 p24 Rust/NumPy cluster-check sentence reads as repo notes. If AI text, cut to one sentence or move nothing; keep fact.
- B11 p2 "I will also discuss ... I explore" vs "we" elsewhere (Ch1). Author text: tense/person fix is allowed only as real error; leave and list. Same Ch5 p84-85 "as I discovered", "I run".

## Ch3 (mostly author/paper text: fix real errors with one note; never reword for style)
- C1 p60 Conclusions "Model family was the primary factor in every decomposition except one, Caco-2 under grouped-shifted noise" — Table 3.1 also: Gaussian R2 at 1.0 representation 29.0 > model family 26.6; grouped-shifted R2 at 1.0 replicate 51.0 > model 8.4; p52 replicate largest for predictive performance on Caco-2 and hERG Ki. Fix the sentence to match.
- C2 p45 "AUCnorm fell by 0.12 to 0.18 relative to Gaussian noise (Table 3.2)" while caption says the two AUCnorm columns are on different scales. Fix (either the caption or the sentence — check which is true).
- C3 p63 "the GP lost the most" (grouped-shifted) vs Table 3.3: GP loses 0.034–0.054, NN-α up to 0.115, NN-β up to 0.080.
- C4 p63 "The same preferences held on all four datasets. Under Laplace, Student-t and outlier noise, RF was the most robust model and SVM was the second." Table 3.4: hERG Ki Student-t SVM 0.894 > RF 0.892; LogD Laplace NGBoost 0.924 > SVM 0.923.
- C5 p51 "SVM has the lowest AUCnorm of the four under Gaussian noise on almost every dataset and representation, and rises to second" — Table 3.4 Caco-2 SVM 0.841 vs GP 0.809; hERG Ki SVM 0.859 vs GP 0.831, NGBoost 0.842. "the four models run under every condition" — SVM never ran under censoring.
- C6 p44 "SVM gains AUCnorm and NGBoost loses it under all three distributions across all datasets and representations" — Table 3.4 NGBoost 0.000 under Laplace on LogD; only three of six representations ran.
- C7 p51 "NGBoost has the highest AUCnorm across the board" vs p50 rank 2nd–10th Caco-2, 6th–9th hERG Ki; "lowest ranked ... up to 0.75 with every representation" vs p42 "on four of the six representations".
- C8 p62 "NGBoost and the VBLL networks start lower and hold what they have" vs p51 VBLL least robust on LogD, p52 VBLL-α 38% of clean R2 at 1.5 on LogD.
- C9 AUCnorm vs clean R2: p36 "less directly correlated", p51 "works against it by construction", p63 "unrelated", p41 "rank in opposite directions" for ρ = −0.10 (p = 0.73). Make consistent with the evidence (weak, not significant).
- C10 p33 methods omit the Laplace/Student-t/outlier runs on the three assay datasets with seven models (VBLL-α het. dropped). Add one sentence to Methods describing what ran (from results p49/Table 3.4).
- C11 p32 Tanimoto-kernel GP described but never reported and not in the 13+6 models. Either say it was dropped or remove the description (author text → one note).
- C12 p32 variance head "add an output head to the standard NN ... with that variance" vs model names "BNN-α (var. head)". Fix the referent.
- C13 p31 NGBoost validation sentence self-contradictory: "a scaffold-grouped fifth of the rows NGBoost is handed, which already exclude the validation molecules". Fix.
- C14 p33 vs Table 3.5/Fig 3.14: GP(het.) on QM9 PDV not run, but methods say it ran; p58 "in every pairing of dataset and representation". Fix methods and p58.
- C15 p55 Fig 3.11 caption ranks among all 19; prose ranks among 13. Make the prose say which.
- C16 p61 "Transforming a feed-forward network into a Bayesian one increased AUCnorm, but ... never predicted better at any noise level" — p55 "made no significant change". Fix to match.
- C17 p63 mechanism claims: "SVM ignores residuals inside its margin, so it gains" (robustness comes from linear loss outside margin); "QRF fits conditional quantiles without shifting the fit" (QRF = ordinary RF trees, quantiles from leaves; p54 QRF less robust than RF under grouped-shifted). Fix the statistics.
- C18 p63 "little separated the models below a noise level of 0.3" vs p52 four of 13 below 90% at 0.3 on hERG Ki.
- C19 p62 "Of the boosted trees, XGBoost lost the most" vs Table 3.4 NGBoost −0.147 Caco-2, −0.093 hERG Ki (grouped-wider).
- C20 p57 "Despite QRF's strong predictive performance" vs p50/p61 middling.
- C21 p54 "(Additional files 2 and 2)" broken pointer. p34 "(Datasets subsection)" → \ref; duplicate sentence. p60 last sentence of 3.3.6 unparseable. p31 "γ = scale" → say what it is (1/(n_features·Var(X))). p53 "in the units of Equation 2.4" wrong (τ is fraction of label SD). p41 Fig 3.2 caption "(0.75 on Caco-2)" unexplained.
- C22 p62 confident learning described for regression — check wording.
- C23 p56 BNN-α "fails to rise in at least one fold on every assay dataset" vs Table 3.5 no † on hERG Ki PDV; † on QM9 vs "run once".

## Ch4
- D1 p74 "Every method in Section 4.3.3 takes a similarity as input" — false (DivideMix, small-loss, uncertainty curriculum use loss/uncertainty). Fix.
- D2 p79 conclusion "methods built on a distance did worst under label noise" stated flat, but p77 hedges (added distance; poor adaptation alternative) and ANOVA is on R2 not retention. Carry the hedge into conclusion (and Ch8 p212 if it repeats).
- D3 p77 GP(het.) 79.0% at σ = 0.6 ranked against others at unstated level; "(standard deviation 0.4%)" over what. Fix wording to not rank unlike levels; say what the SD is over if findable.
- D4 p73 "five-neighbor prediction reached 0.29 to 0.89, higher than the all-pairs correlation in all 70" — different statistics. Reword so it does not compare them as like-for-like; adjust p79 conclusion accordingly.
- D5 p73 QM9 cliff gap 0.05 eV = 0.06 MAD — a fixed 15% cliff share makes "cliffs" on QM9 not cliffs in the usual sense. Add a sentence saying the cliff threshold is a quantile, so on QM9 it marks small differences (do not change numbers).
- D6 p66, p76 broken self-references "Section 4.3.3 ... (Section 4.3.3)". Fix.
- D7 p79 "positive recovery in 14.2% ... mean recoveries of −22.9% and −20.9%" vs Table 4.4 33.3–50.0%, −0.1 to −7.3: state scope (which methods) so they don't read as Table 4.4.
- D8 p67 "we have not looked into why" (292 of 1,024 MHG-GNN features dead). Rephrase neutrally (cause not investigated) — keep the fact.
- D9 p64 garbled "we ask if ... satisfy the SPP, insofar as they do not sufficiently quantify". p67 "These constant counts are sometimes filtered out by the distance" garbled. p68 "That's great behavior", p67 "When your dataset covers", p74 "We've identified" — informal (if author text: contractions are real register errors → fix with one note).
- D10 p77 "MHG-GNNs encode the molecular graph" given as reason sweep distances failed, but MHG-GNN not in sweep. Fix.
- D11 p80 Table 4.5 vs text mapping (compactness) mismatch; row "Features that are hard to read chemically (Section 4.3.1)" cites section that doesn't discuss it.
- D12 p65/66 "Medina-Franco [128] point out/make" → singular.
- D13 Table 4.2 "−0.00†" renders as negative zero: if the value is a rounded negative tiny number, print 0.00 (table-format fix only, not a value change) — only if the source table is AI/generated; else list.
## Bibliography (refs.bib)
- E1 [109]/[193] duplicate title "Quantum chemistry structures and properties of 134 kilo molecules" (paper vs figshare). If both cited, retitle dataset entry (e.g. "QM9 dataset" howpublished Figshare).
- E2 "Isidro Cortes-Ciriano" vs "Isidro Cortés-Ciriano" — make consistent.
- E3 RDKit no year — add year (version release year).
- E4 [120] Harrison ICLR 2024 pages — remove pages (ICLR has none).
- E5 [1] Hughes initials "JP Hughes, S Rees" format; [203] Varnek initials — expand to full names if known, else consistent initials.
- E6 [143] "Springer, 2001" publisher inconsistent with other LNCS — drop publisher.
- Follow memory rules: no DOI when journal+volume+pages+year present; year only; ≤10 authors.

## Ch5
- F1 p109–110 "The stored results straddle two corrections made on 8 October 2026 ... have no result for that dataset until they are re-run. Most DNN results in this chapter were computed before the first correction". Keep the true caveat, but in thesis register with no dates/re-run language: state that DNN predictions are clipped to the training-label range in later runs, earlier DNN results were not, and which results that affects. Same p113, p117 dated notes.
- F2 p97 "some runs had not finished when this thesis was written" → state factually which results have fewer folds or are missing (the text already gives folds beside results).
- F3 p95 vs p97 8,160 vs 8,220 configurations; "five downstream models" vs Table 5.5 six. Check and fix prose.
- F4 Table 5.6 SMAC row 810 vs 945 from its own values; AC-SINS evaluators missing GP (count 1,296 confirms no GP evaluator) — and EGFR/titer/BH break the "three evaluators" rule stated p95. Do not change table numbers; fix the p95 rule wording to match the table, and add a footnote/flag for SMAC 810 (list it for the author).
- F5 Table 5.6 footnote "Changed after the search ran: the representation counts were reduced on 7 October 2026" and "AC-SINS scored 1,451 configurations on one of its five folds" (>1,296). Remove the date; keep fact; list the 1,451 vs 1,296 inconsistency for author.
- F6 p91 "the hybrid used all 10 representations on nine of the ten folds, five for each of the two downstream models" self-contradictory. Fix.
- F7 p110 "The hERG Ki total is likely too low, since one of its folds recorded only 90 seconds" / "count only each fold's last attempt": state as fact without guess.
- F8 Table 5.11 Amino-acid composition role "A" vs used as base in Table 5.12 → role B/A. p96 MACCS "too small ... to take part in budget allocation" vs MACCS base on MDR1 and BH. Table 5.10 ECFP4 (reactants), Mol2vec (difference) role B but not in BH base list. Fix roles to match Table 5.12 (role column is a description, not a result).
- F9 p91 "global pooled feature set was far less effective" while global pool is still searched; no evidence. Soften/cite.
- F10 p101 / Table 5.7: SKEMPI 2.0, AB-Bind, AbAgym uncited; EGFR binding, Ang2, binding free energy, hERG blocking rows no reference. Add \citep where refs.bib has the entries; add bib entries only if certain (SKEMPI 2.0: Jankauskaite et al. 2019 Bioinformatics 35(3):462-469; AB-Bind: Sirin et al. 2016 Protein Science 25(2):393-409). AbAgym: check refs.bib; else leave TODO(cite) in a comment and list. hERG blocking = Lhasa data (say so, no cite).
- F11 p81 and p158 "(Section 1)" — label sec:intro:hybrid_questions resolves to chapter number. Fix to "Chapter~\ref{chap:intro}".
- F12 p87 "another set of features, presumably another subset" → samples; drop "presumably". p89 global pool used before defined. p91 self-pointer. p95 "twelve model types the code supports" code detail. p101 Table 5.8 HIC50 short name for dataset not in Table 5.7 (if hERG IC50 is used in Ch7, keep and note). p102 vs p88 missing-value handling conflict (median imputation vs set to 0) — check source/code and make consistent. p89 garbled "so is its direct comparison". p82 "appending RDKit2D descriptors to GROVER lowered RMSE on 9 of 24 datasets" offered as evidence of degradation — check Deng2023; lowered RMSE = improvement on 9 → says it helped only on 9 of 24? Reword to say what it means.
- F13 p138 "On ∆∆G, AC-SINS and the anti-VEGF expression scan it beats every single representation" vs p118 four datasets (fourth: anti-VEGF scan binding). Make consistent with Table 5.14.
- F14 Table 5.7 best-single "first fold only" for anti-VEGF scan binding vs Tables 5.14/5.21 five-fold means; Table 5.13 19 vs 20 datasets. Fix the text/caption claims, not table numbers; list if unresolvable.
- F15 p115 "Only this proportional split is run" vs p117/p137 even-share separate-site experiment. Fix.
- F16 p133–134 Mordred feature names vs p90 "Mordred ... only the representation name and dimension index". Fix whichever is wrong (results name descriptors → p90 wrong for Mordred?). Check figure/feature-names file tests/analysis/data/.
- F17 p133 Table 5.22 widths = 2× full dataset size, but methods say 2× training samples. Fix methods wording to match what was done (2× dataset size) — list for author.
- F18 p120/p138 "How the budget is split matters much less" not true on antibody rows of Table 5.16 (Titer RF, ∆∆G SVM, Ang2 SVM, SMAC DNN). Scope to small molecules.
- F19 p123 "On the small-molecule regression datasets the hybrid also gains on rare scaffolds" — Table 5.17 only three of six. Scope it. Table 5.13 says 14 datasets for this experiment but 4 shown — fix count or caption.
- F20 p138 + Ch1 p5 "On the smaller sets an ensemble often does better" — also wins on anti-VEGF expression (largest antibody). Fix.
- F21 p138 causal claims "because the representations carry different information", "for the same reason ... few and buried", p139 "ranking features by importance within each representation is what makes the hybrid work" — untested; hedge.
- F22 p81 "This chapter measures predictive performance under both standard labels and labels with injected artificial noise" — Ch5 reports no injected noise results except flipped-label site experiment in Additional file. Fix.
- F23 p109 "Each comparison is reported as its per-fold differences and their spread" but tables give means only. Fix to what is reported.
- F24 p99/p117 "Every dataset uses five outer folds" vs Table 5.18 humanness three. Fix text.
- F25 p127 LogD DNN "best width may lie above what was searched" vs Table 5.18 searched up to 8,000. Fix.
- F26 p139 "separate sites are simulated from one public dataset" — two (Caco-2, LogD).
- F27 p134 "picked both from two different representations" but both examples Mordred. Fix.
- F28 mRMR described comparator with no result; Table 5.13 count 12 vs Table 5.22 11 rows; Table 5.15 caption "average the predictions" vs seven pooling rules; Table 5.14 code-style rep names (ESM2-WT-Delta, Mutation-Features, Difference-Avalon, ESM2-Concat, ESM2-Early) and "NN" vs DNN — rename to prose names (names only, not numbers) if Table 5.14 is in thesis.tex (it is, lines near tab:hybrid_vs_best_single).
- F29 p116 "GROVER and SMI-TED were not built" drafting; ECFP4 stereo-blind vs "no two molecules same row" grouping. p124 "not with 150 to 500 training antibodies" vs EGFR 517, Nb-Tm 537. p123 Frequent/Rare vs easy/hard half vs common half; 1,256+745=2,001 vs "at most 2,000". p118 "largest antibody set" tied 4,275. p124 "Every number in this chapter is scored on samples the search never saw" false for best-in-hindsight and Table 5.18 Search column. p138 new embeddings claim vs p131 fewer than half beat random.
- F30 Implausible precision 0.687104/0.687096 (Table 5.15), 0.43823/0.43822 (Table 5.26): these are the decimal tie rule; leave, list.

## Ch6
- G1 13 vs 14 datasets: opening says 14 (six small-molecule, seven antibody, one reaction); p141 "all 13", Table 6.2 "All 13" five rows and caption "13 of this chapter". Make consistent with what tables show (most tables 14 rows; MDR1 added late, absent from some tables — the sentence on p141 lists which).
- G2 p163 "QM9 has no result for the best number of features, importance weighting in ..." vs Table 6.7 QM9 row, p151 prose, Table 6.11 QM9 0.575/0.484; "choosing which molecules to measure next" only ran on hERG Ki and BH. Table 6.2 "12, no QM9". Fix to match tables.
- G3 p141 missing-runs list incomplete: Table 6.7 no AC-SINS, hERG blocking, BH, MDR1 rows; Table 6.6 weighted values missing on five datasets; Table 6.11 blanks; Table 6.12 no QM9/MDR1. State the gaps once.
- G4 Table 6.2 "joined: 12 regression" vs Table 6.11 13 rows.
- G5 p157 "another hybrid does worse than random picking" — no number/table, no methods. Remove the unsupported clause or point to where it is (search Additional files).
- G6 p142 similarity hybrid question never answered in 6.4 — add one sentence to conclusions from the existing result ("On Caco-2 the similarity hybrid beats that pick in turn" + whatever Table shows).
- G7 p153 "On hERG Ki the Gaussian process goes from 0.582 to 0.711 RMSE" — no table holds these, cited table is SVR. Check source (similarity results / Additional files); if not found, remove the numbers and list.
- G8 p151 "Importance weighting alone closes much of the gap (Table 6.6)" — little on EGFR/CHO/∆∆G; only 8 of 14. Scope it.
- G9 p158–159 contradictory cliff sentences; fix to mean-over-two-folds.
- G10 p141 "MDR1–MDCK efflux was added late" — drafting; restate neutrally. p158 "A second run with the small-molecule models" undefined.
- G11 p143/145 unreported earlier comparisons ("an earlier comparison of four distances on hERG Ki and LogD" etc.). Point to Additional files if they exist; otherwise state them as design choices without claiming unreported results — list. Denominators 34 of 48 vs 19 of 33.
- G12 model names: DNN vs NN vs network; SVM vs SVR vs support vector regressor. Use the Ch5 names (DNN; SVM for the model used throughout; say once that the SVM is a support vector regressor on regression tasks).
- G13 p141 "Datasets, folds and metrics are as in Chapter 5" — say which six of Ch5's 20 are not used (rat microsomal clearance, both anti-VEGF scans, angiopoietin-2, humanness, HIC — verify against tables).
- G14 "five small-molecule datasets" / "other three small-molecule datasets" vs six. p149 random subset vs principal components mismatch. Table 6.4/6.8 captions "at most four" and bold that doesn't exist. Table 6.3 rep names SMITED/Top-Torsion/Atom-Pair/One-Hot → SMI-TED, topological torsions, atom pairs, one-hot; ChemBERTa ambiguity on QM9. p159 "largest change" → "largest gain". p144 group-exclusion sentence. p149 BH "scaffold families" → additive groups. p143 negative score → negative share unexplained (list). p162 "These losses agree with the central claim" — unfalsifiable; reword. p141 vs p143 Euclidean/equal-weight vs Manhattan/rank-weighted mismatch — state why or list. p155 "spread larger than the gain" → loss. p149 "best of ten representations and distances". p157 importance-method automatic.

## Ch7
- H1 p164 "In Chapter 6, on the five small-molecule datasets ... improved on all five" vs Ch6 six. Fix.
- H2 p187/190 garbled "Random deletion alone also helped the first single representation, so this ... score moves between fits as well" (LightGBM on BH) vs p179 LightGBM gives one score every experiment. Fix.
- H3 Table 7.5 "twelve published methods" with three rows "New"; caption "'New' means the code cites a paper that does not describe what it does". Rewrite caption/labels: say these three are our adaptations of named published methods (state which) — exposes code otherwise.
- H4 p171 "It was written as James–Stein shrinkage [270]" — say what the method is.
- H5 p173 cleaning-before-search "under outlier noise and on labels as delivered" vs Table 7.27 (SMAC 25% clipped, hERG Ki Gaussian 1.0, delivered). Fix.
- H6 p175–176 "This experiment ran on Caco-2, hERG Ki and LogD" — Table 7.30 also QM9, Table 7.29 hERG Ki 25% clipped. Fix.
- H7 p179 "second hybrid scores between 0.348 and 0.362 RMSE" vs Table 7.33 0.364; MentorNet 0.343 vs 0.364 beyond "about 0.01"; p181/p209/p211 "within the network's own spread". Fix to the numbers.
- H8 p181 "only label smoothing on hERG Ki clears the bar" — hERG blocking label smoothing and BH deletion also pass. Fix.
- H9 p170 "grade the ranking by the area under the ROC curve" but p193 chance 0.106 = average precision. Fix metric name to match what's reported (check source).
- H10 p165 "have no noise results yet" list omits MDR1 and humanness; drop "yet"; state complete list.
- H11 status language: p181 "no QM9 result yet", p195 "LogD, which has no break-even yet", "The hERG Ki and LogD runs have not finished and have 2 of 3 noise draws", p202 "has not run", p203 "No random-flag control has run". Restate as facts of what the results cover (e.g. "hERG Ki and LogD have two of three noise draws"; "no random-flag control was run").
- H12 p205 "the network's single representation does not improve with any of the three" — true only for second single. Fix.
- H13 p206 "Retuning on the damaged labels beats reusing settings tuned on clean labels" — Table 7.29: under 25% clipping clean-label settings win all 8; Gaussian network first hybrid 0.712 vs 0.709. Fix.
- H14 p202 "Cleaning before the search did best when the fit also used the cleaned labels" — ~4 of 10 rows. Fix.
- H15 p178 "Five controls ran" names four. p190 Table 7.14 caption "1%" vs methods 5,10,20% (source flag_rate=0.01: add 1% to methods or note). p170 BH/conformal bound needs clean calibration set — add the caveat. p181 "(Table 7.32)" → 7.33 for QM9. p175 "Two experiments test noise-robust training" then more. p167 "Unlike Chapter 3" not a contrast. p172 Huber weight never described/reported. p180/188 label smoothing vs move-toward-neighbors naming. Table 7.12/7.14 unexplained bold (bold the best per row or remove — table formatting, not values; keep ≤4). p192 "undefined under every damage kind" pointer. p208 "only method that helps on ∆∆G" → "only one of the twelve". p202 "all three draws" for hERG IC50 explain; hERG IC50 not in Table 5.7 — add one sentence on what it is. Ch4 p76 sweep of 12 includes Mixup, SAM; Ch7 "same twelve" — check and fix wording.
- H16 p196 "hERG Ki and QM9 have only fold 0" vs p198 QM9 all five folds clipped (0.5072 → 0.3980 vs Table 7.20 0.520 → 0.441): say it is a separate five-fold run.
- H17 p193 "Shapley value reached 0.391 on BH against 0.195 for the residual" vs Table 7.15 0.193. Fix prose to table. p193 "a few passes" — give the number if in source.
- H18 p203 "the one dataset with a second fold here" vs Table 7.27 folds 0,1,2. p198 SMAC "raised ... in most configurations" vs Table 7.20 only config 0.162→0.1469. p211 "These gains passed random and shuffled controls on hERG, hERG Ki, BH" — mislinked to previous sentence.
- H19 Table 7.6 vs 7.7 fold 1 hybrid no-cleaning 0.831 vs 0.816 (= single 0.8155) — possible mislabelled row: list for author, do not change.
- H20 same baselines different numbers across Ch7 tables (0.693 vs 0.718; 0.597 vs 0.599; 0.607/0.608; SVR clean 0.616/0.617 vs 0.621/0.628) — list for author, do not change.
