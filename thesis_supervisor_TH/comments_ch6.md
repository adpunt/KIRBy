  HIGHLIGHT p162 y=616-752: 1. Is predicting each molecule’s activity from its five nearest neighbors more accurate in a hybrid than in the single representations from which the hybrid is built? 2. Are noisy labels easier to find, or the size of their errors easier to estimate, in a hybrid? 3. Does a hybrid change how well noise mitigation (label

### TH-p162a (old PDF page 162, margin y=687)
COMMENT: Nice set of questions !
  HIGHLIGHT p163 y=146-161: Noise detection has no result yet.

### TH-p163a (old PDF page 163, margin y=150)
COMMENT: Is this going to be replaced?
  HIGHLIGHT p163 y=672-706: . [FIGURE NEEDED: flow from noisy data, to per-fold hybrid and single representations, to scores]

### TH-p163b (old PDF page 163, margin y=676)
COMMENT: Yes we really need a figure here.
  HIGHLIGHT p164 y=87-100: 6.2.1 Similarity in a Hybrid

### TH-p164a (old PDF page 164, margin y=37)
COMMENT: This section seems to have overlap with chapter 5 and previous chapters aswell (probably because of the paper projection) . For the thesis reader it might feel a bit redundant though. 

Note: I am not familiar with the thesis writing protocol in Oxford and this might be perfectly OK
  HIGHLIGHT p171 y=641-652: Table 6.4:

### TH-p171a (old PDF page 171, margin y=639)
COMMENT: Encouraging results :-) 

### TH-p174a (old PDF page 174, margin y=191)
COMMENT: Interesting to see there is signal beyond the clipping threshold and how all learning approaches are impacted.
  HIGHLIGHT p174 y=197-275: Table 6.6: Held-out RMSE in pKi (lower is better) on hERG Ki with ECFP4, mean over 3 scaffold-group folds, after the training labels above 6.86 pKi were clipped to 6.86 pKi. One row is one version of the training labels, and one column is one downstream model. Never clipped: the original training labels. Clipped, then lifted: every label at the limit raised by the shared lift of Section 6.2.3. Clipped, deleted: the molecules at the limit removed before training.
  HIGHLIGHT p175 y=491-527: Table 6.8: Held-out RMSE in pKi (lower is better) on hERG Ki with the clipped labels of Table 6.7, mean of the same 3 scaffold-group folds. One row is one downstream model on one of its two best single representations.

### TH-p175a (old PDF page 175, margin y=440)
COMMENT: We constantly see that hybrid representation really benefit SVM and LightGBM but not so much (or even detariorate) RF models. Are we discussiong this aspect specifically somewhere?
  HIGHLIGHT p177 y=356-366: Table 6.10:

### TH-p177a (old PDF page 177, margin y=317)
COMMENT: This table could benefit from bold best results.
Landscape is clipped.