# Thesis TODOs that need a decision from you (Chapters 1–5)

Written 2026-10-08. Each item starts with the exact TODO comment, so you can search for it in `thesis.tex`.
Where I removed the comment, the item says which text to search for instead.

---

## Part 1: Short answers (one or two sentences each)

### 1. "prediction metric"

`% TODO: is prediction metric the right term here?`

The usual term is **evaluation metric**. Elsewhere in Chapter 5 you already write "the dataset's metric", so "the dataset's evaluation metric" would match.

### 2. "training" for the hybrid search

I removed this comment: `% TODO: confirm the term training used here`. Search for: `While it is not formal ``training''`.

The sentence is accurate. In the machine-learning literature, a feature selection method is said to be *fitted* on the training data, the same as a model. The search computes its importance scores and representation scores from training labels only, so "fitted on the training data" is the right phrase.

### 3. Matched evaluator with a model the code does not know

`% TODO: does match always work? custom downstream model?`

It works for 12 named models: random forest, LightGBM, XGBoost, MLP, SVM, GP, kNN, CatBoost, DNN, CNN, BNN and NGBoost (`src/kirby/find_best_hybrid.py`, around line 6416).
For any other model, "match" silently adds nothing. If "match" was the only evaluator requested, the search falls back to a random forest.
The thesis does not need to say this unless you describe custom models. Decide: mention it in one sentence, or leave it out.

### 4. How the hybrid differs from ordinary feature selection

`% TODO: need to clarify how this differes from a generalised feature selection method`

You already state the difference in the paragraph starting "Formally, an adaptive hybrid representation is".
Ordinary feature selection ranks all features in one list. The hybrid first splits the budget between representations, then ranks features inside each one.
The paragraph under this TODO uses "generalised" to mean "works outside chemistry", not "general feature selection". Changing that word to "domain-independent" would remove the confusion.

### 5. "representation knobs"

`% TODO: mention representation knobs`

I do not know what this refers to. The representation count (2, 3, 5 or all representations) is already described under "The search space". If you mean something else, such as settings inside one representation (ECFP radius, number of bits), tell me and I will draft the sentence.

### 6. Why greedy allocation was tried and dropped

`% TODO: explain why the greedy approach was tried, but it's too computationally intensive and actually quite ineffective`

The numbers below come from a July note of mine. I have not re-read the result files today.
- Cost: greedy fits about (budget ÷ step size) × number of representations models per fold. The methods you kept fit about one. That is 20 to 100 times more fits.
- Results: in `tests/results/framework/f01_allocation_strategies/`, greedy's mean rank was 13.67 out of 21 allocation methods, over 6 datasets. In `f12_allocation_v2/` it never ranked first on any of 6 datasets.

Decide: one sentence in the thesis, or nothing. If you want the sentence, I will re-check these numbers first.

### 7. Explaining the framework experiments

`% TODO: explain framework experiments and how they helped here`
`% TODO: explain framework experiments`

The text now says "Preliminary design experiments set the values each setting may take", with a pointer to the Additional file `app:add:design_experiments`.
Decide: is the pointer enough, or do you want a short paragraph in the chapter?

### 8. List of augmentation features

`% TODO: I need a better list to describe augmentations`

Augmentation is cut on all 16 datasets. No full run uses it. The section only needs to say what was tried.
The paragraph below this TODO already lists the kinds of feature offered: MACCS, 3D pharmacophores, Coulomb matrix eigenvalues and antibody mutation features.
Decide: is that enough now, or do you still want a fuller list?

### 9. Units of σ in the exploratory study (Chapter 4)

`% TODO: the exploratory study's sigma is not stated to be in the units of the Chapter 2 noise level`

Only you can answer this from the confirmation report code. Question: in that study, was σ = 0.3 a fraction of the label standard deviation, or 0.3 eV?

### 10. "not a single one" (Chapter 4)

`% TODO: the study says 'all 20 worst-performing configurations ... used distance metrics, not a single one achieved both`

Your report sentence can be read two ways. Reading A: none of the 20 worst configurations reached clean R² above 0.6 and retention above 80%. Reading B: no configuration that used a distance reached both.
The thesis currently says B. Reading A is almost certainly true, but it says little, because the 20 worst have the lowest retention by definition. Which did you mean?

### 11. Additional files the paper cited but the thesis does not have (Chapter 3)

I removed these three comments. The paper cited three Additional files that are not in the thesis Additional files:
- the robustness of each base model on QM9 with ECFP4 (was cited after Figure `fig:decoupling`);
- each base model against its probabilistic counterpart on every dataset (was cited after Figure `fig:variants_change`);
- uncertainty statistics per model and noise condition on QM9 with PDV (was cited at the end of the first paragraph of "Uncertainty under label noise").

The citations are already gone, so nothing in the thesis is broken. Decide: add these three tables to the Additional files, or leave them out.

### 12. Three references that are never cited

I removed this comment: `% TODO: Hirschfeld2020, Scalia2020 and Norinder2014 are in refs.bib but not cited anywhere`.

References that are never cited do not print, so this changes nothing in the PDF.
All three are about uncertainty in molecular property prediction. Cite them in "Uncertainty under label noise" only if you want more background there.

### 13. Two grant numbers in the Acknowledgements (`main.tex`, not `thesis.tex`)

Search `main.tex` for `EP/L015803/1`. Your sentence thanks EPSRC grant EP/L015803/1. The funding paragraph that used to be in Chapter 1 gave EP/S024093/1. Check which number is on your award letter.

### 14. A sentence that stops in the middle (Chapter 5)

Search for: `held-out scoring uses clean labels unless.`

The sentence ends at "unless". I did not guess the ending. It is about Chapter 6, which is waiting on results.

---

## Part 2: Longer discussions

### 15. Leaving TCGA out

`% TODO: I may run into some ethical problems with using patient data.. let's leave tcga out unless there's some really inteestnig stuff in there...`
`% TODO: only use if necessary`

**Ethics.** Your TCGA data come from UCSC Xena (Chapter 5, "Datasets and splitting", says so). Xena serves only TCGA's open-access tier, which is de-identified and public.
Every measurement you list is open access: RNA-seq, methylation, gene-level copy number, microRNA, RPPA protein and diagnostic slide images. The controlled-access tier holds germline variants and raw reads, and you use neither.
At most UK universities, re-analysis of public, anonymised data is exempt from ethics review. Check this with your department or Oxford's ethics committee (CUREC); I cannot confirm Oxford's rule.

**What dropping TCGA changes.** Two datasets go: lung adenocarcinoma vs squamous, and breast cancer PAM50 subtype.
- Chapter 5 goes from 16 datasets in six domains to 14 in five.
- TCGA appears in about ten places in `thesis.tex`, including Tables `tab:datasets`, `tab:reps_tcga` and `tab:dataset_reps`, and the "Cancer patients" paragraph.
- TCGA is the only domain where each representation is a different measurement of the same patient rather than a different encoding of one input. Dropping it removes that example of the method working outside chemistry.

**Recommendation.** Keep it if your department confirms the exemption, because it is the strongest example of the method working on non-chemical data. Drop it if the full runs show nothing beyond the other domains. The ethics answer is the first thing to get.

### 16. How to talk about embeddings

`% TODO: ensure consistent use of embeddings langauge`

**What the literature does.**
- *Learned representation* is the category: any representation produced by a trained model. Your Chapter 2 introduction already uses this.
- *Embedding* is the vector one pretrained model outputs for one input: "ChemBERTa embeddings", "ESM-2 embeddings".
- "Pretrained embeddings" or "frozen embeddings" means the model is not retrained on your labels. Every base representation in Chapter 5 is frozen.
- *Embedded representation* is not a standard term. *Embedded methods* is a feature-selection term (selection during model fitting), which you use correctly.
- Fingerprints and descriptors are never called embeddings.

**Four places that break this** (searchable):
1. Chapter 2: `especially when different molecular embeddings are handled` means representations in general. This paragraph is marked for your manual review, so I left it.
2. Chapter 2: `Distance-based reasoning relies on the assumption that the data embeddings contain` also means representations.
3. Chapter 5: `features derived from embedded GNNs` should read "embeddings from pretrained GNNs".
4. Chapter 5: I rephrased the one-hot paragraph and removed "embedded representations" there.

**Recommendation.** Use "learned representation" for the category and "embedding" for the vector, and change places 1–3. Say yes and I will do 2 and 3. Place 1 is yours, because of the manual review.

### 17. Late fusion or ensembling

`% TODO: I'm pretty sure I'm doing late fusion but calling it ensembling, make sure its appropriately named below`

**You are right that it is late fusion.** Your comparator fits one model per representation and averages their predictions. That is the definition of late fusion in the review you cite (Stahlschmidt 2022).
It is also an ensemble, because any combination of several models' predictions is one. So "ensemble" is not wrong, just less exact.

**The actual problem in the text.** Chapter 5's introduction describes the same method twice, in two separate paragraphs:
- one paragraph starts with `\citet{Steyaert2023} uses late fusion to train one model per source`;
- the next starts with `Ensembling fits one model per representation and aggregats their predictions`.

A reader will think these are two methods. Later, the comparators section says it correctly in one sentence: "The ensemble fits one model per representation and pools their predictions (late fusion)".

**Recommendation.** Merge the two introduction paragraphs into one about late fusion. Keep "ensemble" as the short name of your comparator, defined once as late fusion. That needs no change to tables or results; "ensemble" appears 18 times in Chapter 5. Say yes and I will merge the two paragraphs.

### 18. What happens with very wide representations

`% TODO: double check this is an accurate description of what's happening with large reps - which method???`

**The sentence is wrong.** It says representations wider than "5,00" features keep their top 5,000 features. The code does something different, in two separate places.

**Place 1: scoring one representation on its own.** This is the score each representation gets before the budget is split. If a representation has more than 5,000 features, it is scored on only its top 500. On datasets under 1,000 samples it is 200.
The top features are picked in each fold by a LightGBM model fitted on that fold's training part only (`src/kirby/find_best_hybrid.py`, `HIGH_DIM_FILTER_THRESHOLD = 5000`, around line 4592).
This does not change which features the hybrid can pick. It only changes the representation's own score.

**Place 2: importance scoring inside the hybrid search.** LightGBM gain and TreeSHAP fit only the 2,000 features with the highest variance when a representation has more than 2,000. The rest get importance 0, so the hybrid can never pick them. DeepLIFT scores every feature.
Table `tab:importance_methods` already says this correctly (`src/kirby/hybrid.py`, `max_features_lgb` and `max_features_shap`).

So the one sentence mixes the two. Its number is also wrong: 5,000 is the trigger, 500 is what is kept.

**Recommendation.** Replace the sentence with: "A representation wider than 5,000 features is scored on its own using only its top 500 features (200 on datasets under 1,000 samples). These are chosen in each fold by a LightGBM model fitted on the training part. This affects only that representation's own score, not the features the hybrid can select." Say yes and I will put it in.

### 19. Which splits are used

`% TODO: what splits are used here?`

There are three levels. Each keeps groups whole: scaffold for molecules, sequence cluster for antibodies, and so on.

```
Whole dataset
│
├── Outer split: usually 5 folds (set per dataset, Section sec:method:datasets)
│     Each outer fold takes one turn as the test set.
│     The test set is used once, to score the finished hybrid.
│
└── Outer training part (the other 4 folds)
      │
      ├── Inner split: 5 folds
      │     Each search setting is scored by its mean over these 5 folds.
      │
      └── Inner training part of each inner fold
            │
            └── Internal split: 80% / 20%
                  80%: importance scores are computed here.
                  20%: each representation's own score is computed here.
                       It decides how the budget is split.
```

The code confirms this (`src/kirby/find_best_hybrid.py`). The internal 80/20 split is the first fold of a 5-fold grouped split. For classification, the folds are also balanced by class.

The paragraph under the TODO already describes all three levels. It is hard to read because it never names them. **Recommendation:** name them "outer folds", "inner folds" and "internal split" in that paragraph. You could also add the block above as a small figure; see `THESIS_DIAGRAMS.md`.

---

## Part 3: TODOs I closed myself (check if you disagree)

- **Chapter 2 duplicates removed.** These definitions are now only in Chapter 3: QRF, NGBoost, the first SVM sentence, the GP definition with its O(N³) cost, and the robustness equation (\aucnorm). Chapter 2 now points to Chapter 3's equation.
- **Chapter 2 epistemic/aleatoric paragraph deleted** (`% TODO: this paragraph may need to go`). Chapter 3 defines both terms in its Methods subsection "Uncertainty".
- **Chapter 2: the 20% sentence is now in the text**, after "Table tab:noise_conditions notes where no published estimate exists". It is marked for manual review.
- **Chapter 2: the noise-mitigation figure stays where it is.**
- **Chapter 2 title kept** as "Molecular Representations, Models and Label Noise".
- **Dablander2023 author** fixed to Markus in the repo-root `refs.bib`. The `refs.bib` the thesis builds from already had it right.
- **Chapter 4:** the duplicated Cortes2015 sentence is removed. "similarity-property principle (SSP)" is now "similar property principle (SPP)", with a pointer to where Chapter 2 defines it. Chapter 1 never mentioned it.
- **Chapter 5:** the "this is out of date" TODOs in the methods are closed, because those paragraphs were since rewritten. The rewrites are still marked for manual review.
- **Section and subsection headings in Chapters 1–5** are now in sentence case. Chapters 6 and 7 are still in title case.
