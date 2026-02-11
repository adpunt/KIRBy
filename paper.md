% KIRBy: Knowledge-Informed Representation Blending for Molecular Property Prediction

\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}

\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{enumitem}

\begin{document}

\title{KIRBy: Optimal Hybrid Molecular Representations via Complexity-Aware Feature Selection}

\author{Adelaide Punt}
\affil{Department of Statistics, University of Oxford}

\maketitle

\section*{Abbreviations}
% RF: Random Forest, kNN: k-Nearest Neighbours, ECFP: Extended-Connectivity Fingerprint,
% MACCS: Molecular ACCess System keys, PDV: Physicochemical Descriptor Vector,
% SAC: Similarity-Activity Correlation, WL: Weisfeiler-Lehman, PCA: Principal Component Analysis,
% SHAP: SHapley Additive exPlanations, CV: Cross-Validation, SALI: Structure-Activity Landscape Index

\section{Introduction}\label{sec:intro}
% Problem: molecular property prediction relies on fixed representations
% Each representation captures different structural aspects
% No single representation dominates across all models and datasets
% Naive concatenation is high-dimensional and suffers from curse of dimensionality
%
% Contribution: framework that automatically discovers optimal hybrid representations
% - Model-agnostic: works with RF, kNN, ridge, XGBoost, MLP, LightGBM
% - Budget-constrained: produces compact representations (50-200 features)
% - Importance-weighted: uses feature importance to allocate features across reps
% - Forward-compatible: new representations slot in without retraining

\section{Methods}\label{sec:methods}

\subsection{Datasets}
% logd: OpenADMET-LogD (~7309 molecules, lipophilicity)
% caco2: OpenADMET-Caco2_Efflux (~3777 molecules, P-gp efflux ratio, log-transformed)
% herg: ChEMBL-hERG-Ki (~1415 molecules, hERG binding affinity pKi)
% qm9: QM9 HOMO-LUMO gap (~2000 molecules, quantum mechanical property)
%
% Scaffold-split CV throughout (GroupKFold on Murcko scaffolds)

\subsection{Molecular Representations}
% Table of 16 representations (all_no_gk preset):
% Classical: ECFP4 (1024), MACCS (167), PDV (200)
% Pretrained: mol2vec, ChemBERTa, MolFormer, SelFormer, ChemBERT
% Graph neural: MolCLR, GraphMVP, SMITED, MHG-GNN, Grover
% 3D/other: UniMol, SchNet, ChemFormer
%
% Note on graph_kernel (WL histogram, 35K-72K features):
% Investigated separately (Section X.X), excluded from standard presets
% due to computational infeasibility at full dimensionality

\subsection{Models}
% Ordered by complexity:
% Ridge regression (linear baseline)
% kNN (non-parametric, sensitive to dimensionality)
% Random Forest (ensemble, moderate complexity)
% XGBoost (gradient boosting)
% LightGBM (gradient boosting, fast)
% MLP (neural network)
%
% Complexity ordering matters for matching principle hypothesis

\subsection{Hybrid Construction Pipeline}
% 1. Feature importance computation (8 methods, clustered)
% 2. Budget allocation across representations (6 methods)
% 3. Feature selection within each rep
% 4. Concatenation of selected features -> hybrid representation
%
% Key insight: importance-based allocation != naive top-N from concat

\subsubsection{Importance Methods}
% Table: treeshap, xgboost_gain, lightgbm_gain, deeplift, permutation,
%        kernelshap, integrated_gradients, gradientshap
% Clustered by correlation (Table X)
% Speed tiers: fast (4), medium (6), long (8)

\subsubsection{Allocation Methods}
% greedy: iteratively add rep with largest marginal gain
% greedy_feature: iteratively add single best feature across all reps
% performance_weighted: allocate proportional to single-rep performance
% two_stage: importance ranking then greedy refinement
% stability_selection: bootstrap importance stability
% fixed: equal allocation (baseline)

\subsection{Search Procedure}
% find_best_hybrid() API
% Speed tiers: fast, medium, long
% Blueprint pattern for reproducibility
% Per-fold hybrid construction (no data leakage)

\subsection{Performance Metrics}
% R² (scaffold-split CV, GroupKFold)
% Baselines: best individual representation per model (NOT concatenation)
% Dimension tiers: 50, 100, 200, 500

\section{Results}\label{sec:results}

\subsection{Experiment 1: Representation Characterisation}
% Profile each rep's intrinsic dimensionality, effective rank, MI with target
% Table: rep, nominal_dim, pca_90, pca_95, mle, erank, MI

\subsection{Experiment 2: Complexity Matching}
% Core result: optimal hybrid configuration differs by model
% find_best_hybrid() discovers this automatically
% Table: model x best_allocation x best_importance x best_budget x R²
% Delta vs best individual rep per model
%
% Key finding: [TODO — fill from results]

\subsection{Experiment 3: Cross-Dataset Generalisation}
% Same framework applied to logd, caco2, herg, qm9
% Do optimal configurations transfer across datasets?
% Dataset complexity metrics (Lorena et al.) vs optimal budget correlation

\subsection{Experiment 4: Importance Weights as Learned Metrics}
% Feature weights from hybrid construction define a learned distance metric
% Bridges to similarity experiments

\subsection{Experiment 5: Learning Curves}
% Hybrid advantage vs sample size
% At what n does hybrid start beating best individual rep?
% Asymptotic model fits

\subsection{Experiment 6: Within-Family Complexity Gradient}
% Vary model capacity (e.g. RF n_estimators, MLP hidden size)
% Does optimal hybrid budget track model capacity?

\subsection{Experiment 7: Data Complexity via Scaffold Partitioning}
% Different scaffold clusters have different complexity
% Does optimal config change by subpopulation?

\subsection{Experiment 8: Uncertainty Calibration}
% Conformal prediction at optimal vs suboptimal complexity
% Better-matched complexity -> tighter prediction intervals

\subsection{Experiment 9: Formal Complexity Analysis}
% ANOVA: model x allocation x importance interaction effects
% Information retention at each budget
% Effective rank of hybrid vs individual reps
% Bias-variance decomposition

\subsection{Similarity Experiments}

\subsubsection{Similarity-Activity Correlation (SAC)}
% Problem: standard molecular distances correlate poorly with property similarity
% SAC baseline per rep (Spearman of distance vs property difference)
% Hybrid importance-weighted distance dramatically improves SAC
% Result: SAC improves from ~0.04-0.11 (individual reps) to ~0.36 (hybrid weighted)

\subsubsection{kNN Prediction with Hybrid Distance}
% Hybrid-weighted kNN outperforms individual-rep kNN
% Delta vs best individual rep

\subsubsection{Prediction Performance on Activity Cliffs}
% MoleculeACE-style evaluation: RMSE_cliff on cliff compounds
% Cliff definition: ECFP4 Tanimoto >= 0.9, |Δy| > 2*MAD(y) (adaptive)
% Hybrid-weighted kNN reduces RMSE_cliff vs per-rep kNN baselines
% Enables comparison with published MoleculeACE benchmarks

\subsubsection{Applicability Domain}
% Hybrid distance tracks prediction uncertainty better
% Spearman(distance, |error|) for hybrid vs individual reps

\subsubsection{Noise Detection Bridge (Supplementary)}
% Standard noise detectors flag activity cliffs as false positives
% Hybrid distance reduces FP rate

\subsubsection{Optimal Dimension for Similarity vs Prediction}
% Similarity may peak at lower dim than prediction (curse of dimensionality)
% Optimal budget comparison

\subsection{Aggregation (Negative Result)}
% Tested: borda, simple average, reciprocal rank fusion
% Aggregating importance scores from multiple methods does not consistently
% outperform single best method (lightgbm_gain)
% At budget 200: borda R² approaches but does not exceed best individual rep
% Conclusion: single-method importance (esp. lightgbm_gain) is sufficient;
% aggregation adds complexity without clear benefit

\subsection{Graph Kernel Investigation}
% WL histogram representation: 35K-72K features depending on dataset
% Highly sparse (~70-80% zeros), most features uninformative
% PCA effective dimensionality: [TODO]
% Reduction strategies: PCA vs importance-based top-N
% Conclusion: [TODO — fill from gk_analysis results]

\section{Discussion}\label{sec:discussion}
% Core thesis: model complexity and representation complexity have an optimal pairing
% Framework discovers this automatically
% Forward-compatible: new representations slot in
% Low-dim hybrids useful for downstream tasks (similarity, clustering)
%
% Limitations:
% - Scaffold split may be pessimistic for some applications
% - Importance methods have different biases
% - Graph kernel requires reduction preprocessing

\section{Conclusion}\label{sec:conclusion}
% TODO

\section*{Declarations}
% Funding, conflicts, data availability, code availability

\end{document}
