
### TH-p107a (old PDF page 107, margin y=421)
COMMENT: Here we could add one sentence to link it to the general topic of the thesis. For instance , will hybrid representation impact more the variance  and benefit noise mitigation 
  HIGHLIGHT p108 y=106-141: The questions this chapter and Chapter 6 address are set out in Chapter 1 (Section 1).

### TH-p108a (old PDF page 108, margin y=109)
COMMENT: Maybe easier for the reader to restate the question here rather than force navigation.

### TH-p108b (old PDF page 108, margin y=375)
COMMENT: This might be a bit contradicting the explainability goal as you previously mentioned that GNN might loose their link to structural features.

### TH-p108c (old PDF page 108, margin y=603)
COMMENT: Indeed, the ‘ensemble’ of the models (with the target algorithm) should be a base line to compare with the hybrid-representation (on that same target algorithm)
Is it better to merge the representation or the predictions?
  HIGHLIGHT p108 y=398-434: We also include GNN embeddings, since graph-based represen- tations capture structural information that fingerprints often lose
  HIGHLIGHT p108 y=621-696: Intermediate fusion keeps them (MOGONET [151], cross-attention [152]) but must be retrained from scratch to add a source. Multiple kernel learning [153, 154] works only with kernels and yields no interpretable feature subsets. There is no decisive evidence that any of these strategies is broadly superior [150].
  HIGHLIGHT p109 y=126-182: We propose that this is because concatenation adds the ir- relevant features of both parts along with the useful ones, so a method that joins representations has to combine and filter at the same time.

### TH-p109a (old PDF page 109, margin y=124)
COMMENT: That’s an important point indeed and needs to be stressed as it is a driver of the added value of your contribution.
  HIGHLIGHT p109 y=349-364: Descriptor-based models held 73% of top-three positions on 11 benchmarks [158].

### TH-p109b (old PDF page 109, margin y=346)
COMMENT: Nice info !

### TH-p109c (old PDF page 109, margin y=834)
COMMENT: We could have investigated how many structure are merged across representation. This give an idea of the potential of breaking ties by using hybrid representations

### TH-p109d (old PDF page 109, margin y=592)
COMMENT: There is one aspect that we might not have covered here: Some representation fits better some ML algorithm and thus there maybe a value of pairing the best fit representation/learning-type and merge predictions (i.e. in such a case not mixing representation would be by design)

### TH-p109e (old PDF page 109, margin y=744)
COMMENT: Important distinction indeed!
  HIGHLIGHT p109 y=632-728: Ensembling fits one model per representation and pools their predictions, com- bining representations after prediction rather than before. Consensus QSAR works this way [164], so it cannot exploit cases where one representation’s noise is another’s signal. No model in an ensemble sees features from two representations together, and every representation and every model must be computed for each new molecule.
  HIGHLIGHT p109 y=733-768: Feature selection keeps a subset of the original features, unlike principal com- ponent analysis (PCA)
  HIGHLIGHT p110 y=207-344: The hybrid ranks features by a fitted model’s importance scores, as embedded methods do, and keeps a fixed number of the top-ranked ones, as a filter does. Classical methods treat all features as one flat list. The nearest classical relative, the group lasso, keeps or drops predefined groups of variables whole [170], whereas the hybrid splits the budget among representations (Section 5.2.2). Because only original features are kept, each traces back to one representation, which makes blueprints possible.

### TH-p110a (old PDF page 110, margin y=221)
COMMENT: If we had time it would be nice to have a figure representing the different approaches in this paragraph and how hybrid representation are unique.

### TH-p111a (old PDF page 111, margin y=421)
COMMENT: A pedant reviewer could argue that these are the same operations (selecting useful features implicitely means discarding useless features) ;-).  The difference is in the selection criteria and they are complementary.
  HIGHLIGHT p111 y=481-496: Filtering out useless features matters as much as selecting useful ones.
  HIGHLIGHT p111 y=562-597: Formally, an adaptive hybrid representation is a set of up to a fixed number of fea- tures, the budget, selected from several base representations of the same molecules.

### TH-p111b (old PDF page 111, margin y=557)
COMMENT: Worth writing an actual definition of an adaptative hybrid representation.
” Definition:…”
  HIGHLIGHT p112 y=621-656: Explainability is a recognized need in drug discovery [175].

### TH-p112a (old PDF page 112, margin y=624)
COMMENT: “…especially in drug safety assessment when predicting and interpreting toxicity.”

 (to reinforce the general context of this thesis) 
  HIGHLIGHT p114 y=572-587: Representations differ in scale, distribution and missingness.

### TH-p114a (old PDF page 114, margin y=540)
COMMENT: Does a bit feature in ECFP has the same budget weight as a phys-chem descriptor (e.g. logP) in representations like PDV?
  HIGHLIGHT p114 y=276-340: Figure 5.1: Schematic overview of the hybrid search. The inputs (left) are split into five cross-validation folds, and the single representations are scored as baselines. Each fold is searched over importance methods, allocation methods, budgets, internal evaluators and downstream models (center), with adaptive pruning on the first two folds of each search. The best hybrid for each downstream model is stored as a blueprint (right).

### TH-p114b (old PDF page 114, margin y=251)
COMMENT: Might be worth highlighting here, where the current validation and future test sets fits in this setup with respect to the data used for the hybrid search since we want to make sure there is no information leak,
  HIGHLIGHT p120 y=467-502: The allocation method decides how many of the B features each representation contributes; the importance method decides which.

### TH-p120a (old PDF page 120, margin y=474)
COMMENT: Very important reminder. 
  HIGHLIGHT p122 y=292-307: The internal evaluator’s scores set how the budget is split

### TH-p122a (old PDF page 122, margin y=291)
COMMENT: Is this b_k in table 5.3? 
  HIGHLIGHT p122 y=720-732: Augmentation

### TH-p122b (old PDF page 122, margin y=673)
COMMENT: Augmentation may trigger the reader’s ‘Data augmentation’ concept. Maybe the title could be related to the objective in something around “Handling small representations..” 
  HIGHLIGHT p123 y=288-323: On QM9, MACCS is not offered and the topological torsion fin- gerprint (2,048 features) is a donor, the only wide one

### TH-p123a (old PDF page 123, margin y=286)
COMMENT: Why was the 2048 feature descriptor not part of the initial representation candidates?
  HIGHLIGHT p124 y=531-647: The residual ranking has four limitations. |ρP| captures only linear relations, one feature at a time. It ignores redundancy, so two near-identical donor features can both rank high. Donors are pooled with no per-donor limit, so a 2,048-feature donor gets more chances than a 49-feature one. Finally, the margin over k = 0 is zero, so a near-tie on one internal split can admit extras. Table 5.5 summarizes the three conditions.

### TH-p124a (old PDF page 124, margin y=539)
COMMENT: Have we measure the difference of the quality of the hybrid representation with and without donors? The augmention step feels a bit ‘hacky’ and might be challenged.
  HIGHLIGHT p124 y=675-688: Fast Tier and Adaptive Pruning

### TH-p124b (old PDF page 124, margin y=662)
COMMENT: Could the title be more related to the aim of this section (e.g. Pruning search space)
  HIGHLIGHT p126 y=502-537: Configurations far behind after two folds would only confirm their loss on the other three, so pruning stops scoring them

### TH-p126a (old PDF page 126, margin y=490)
COMMENT: This simple statement explains well the reason for the full/fast tiers and could be put as an introduction of the section.

### TH-p128a (old PDF page 128, margin y=388)
COMMENT: So this remains an internal validation using a comparison hybrid vs best base representation (in full).  Given the numerous hyperparameter in the search space, is there not a risk of datasnooping since the hybrid-search operated on the same data as the comparison. Should we not have an external (to the hybrid-search) test dataset to run the full base representions baselines and compare with the selected hybrid-representation ? Unless I missintepreted figure 5.1 where the selection process is outside the CV box.
We can discuss this point in the meeting

(Maybe that was done and I missed it or maybe it comes late though)
  HIGHLIGHT p128 y=470-687: The hybrid is compared with the best single base representation at its full number of features, which is what a user would otherwise run. This comparator is chosen per downstream model and per fold from the search’s scores on the outer training samples, then refit on those samples and scored on the same held-out samples as the hybrid, so it can differ between folds. Choosing it on the held-out samples would give it an advantage that the hybrid does not get. The hybrid is held to its budget, plus any augmentation features, so the comparator is often many times wider, and both widths are recorded for every fold, since a difference between the two scores is partly a difference in width as well as in method. The main quantity per fold is the hybrid’s score minus that of the best single representation, signed so that a positive value means the hybrid scored better (for RMSE the sign is flipped).
  HIGHLIGHT p149 y=478-493: default hybrid

### TH-p149a (old PDF page 149, margin y=469)
COMMENT: What is a ‘default hybrid’ ?
  HIGHLIGHT p150 y=505-555: Table 5.15: RMSE in pKi (lower is better) of the default hybrid and the best single representation on hERG Ki, over three scaffold-group folds. One row is one held-out fold, or the mean over the three folds, for one downstream model; both scores in a row use that model. Mean rows average the three fold rows above them.

### TH-p150a (old PDF page 150, margin y=482)
COMMENT: We can see that hybrid outperforms the best base representation in these combinations. We could add variability in the Mean and use bold for the better RMSE.
  HIGHLIGHT p152 y=474-485: Table 5.16:

### TH-p152a (old PDF page 152, margin y=470)
COMMENT: Nice summary table !
  HIGHLIGHT p153 y=-2147483648-2147483520: 
  HIGHLIGHT p153 y=658-669: Table 5.17:

### TH-p153a (old PDF page 153, margin y=666)
COMMENT: Landscape clipped
  HIGHLIGHT p155 y=164-201: Table 5.20: Spearman correlation (higher is better) on nanobody thermostability, mean over the held-out folds. One row is one number of features, the same for all four methods; for PCA, the number of components. Every method is scored with a random forest.

### TH-p155a (old PDF page 155, margin y=152)
COMMENT: Interesting how the hybrid search reveals a representation that generalises better (better behaviour at low number of features)
  HIGHLIGHT p157 y=380-390: Table 5.22:

### TH-p157a (old PDF page 157, margin y=279)
COMMENT: Are these result compatible with table 5.15 where the best base in RF seemed to perform less well than the hybrid search? Here in RF and XGBoost the baest base seems to outperform the hybrid representation. 
  HIGHLIGHT p158 y=587-598: Table 5.26:

### TH-p158a (old PDF page 158, margin y=583)
COMMENT: Nice!