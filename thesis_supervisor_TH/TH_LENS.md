# How the supervisor reads (derived from his 68 comments on old Ch4-7)

He commented only on old pp. 87-192. The task now: read other parts the way he would and
propose the comments he would make. Proposals only; the author decides.

What he flags, with an example of each:
1. Hard-to-read paragraph that does not say why it matters ("not easy to read (at least for a
   french scientist) and should clearly say why we look at features/similarity in the context of
   noisy data").
2. Number floods without a take-home ("The reader is flooded with numbers ... needs simple (one
   sentence) recap/summary from time to time"; "What is the intermediate conclusion / take home
   message of this section?"; "do these numbers mean ECFP4 is following better the similarity
   principle than Mordred?").
3. Missing recap before a new section ("short recap of the type of flaws and their origin to
   prepare the reader for the next section").
4. Terms used before defined ("We haven't yet explained hybrid representations at this point";
   "What is a 'default hybrid'?"; "Is this b_k in table 5.3?"; "Which standardization? ... What is
   the transformation from 1 -> 4.36?"; "Retained from .. to ...? What is the % measuring?").
5. Cross-reference instead of content ("easier for the reader to restate the question here rather
   than force navigation").
6. Sentence ends abruptly / vague quantifier ("Reader will be frustrated to not know what is
   'more'").
7. Chemistry precision ("The 2D graph expresses stereoisomers via stereobonds"; InChIKey with or
   without stereo; is there a 'merge' category).
8. Methodology rigor: train/test separation in any bound or metric; data snooping when choices
   and comparisons use the same data; whether an external test set exists; whether a 'hacky'
   step has an ablation (with vs without); whether ensembling is a baseline.
9. Task-agnostic vs task-learned representations: fixed fingerprints are not expected to give a
   smooth landscape; learned embeddings should; claims about 'representations' should say which.
10. Unmeasured equality assumptions ("Does a bit feature in ECFP have the same budget weight as a
    phys-chem descriptor like logP?").
11. Section titles that suggest the wrong concept ("Augmentation may trigger the reader's 'Data
    augmentation' concept"); titles should name the aim.
12. A one-line motivating sentence that should open a section ("This simple statement explains
    well the reason ... could be put as an introduction of the section").
13. Formal definitions for the main contribution ("Worth writing an actual definition ...
    'Definition: ...'").
14. Link every chapter end back to the thesis topic (noise) and to drug safety / toxicity
    (Lhasa context): "...especially in drug safety assessment when predicting and interpreting
    toxicity".
15. Figures that summarize an experiment's design (where N combinations come from; a flow from
    data to scores; how the method differs from neighbors in the literature).
16. Tables: bold best results; show variability in mean rows; landscape tables that clip.
17. Consistency between tables reporting the same comparison.
18. Model-dependence of a result discussed explicitly (helps SVM/LightGBM, not RF: why?).
19. Precise wording: "implicitly assume", "can add" (hedged where certainty is not shown),
    "negatively impacted by their flaws" instead of "inherits them", "depends on the distance
    metric" instead of "depends on the distance", "foundation models or representations?".
20. Redundancy across chapters (methods repeated from an earlier chapter).
21. Curiosity questions about edge cases (retention > 1?; proportion of 0 vs 1 in near-constant
    bits) — results the reader will wonder about.
He also praises: clear stories ("more noise less recovery and the amplitude is representation
dependent"), recap tables, well-posed question lists, literature facts with numbers.

He does NOT comment on: style for its own sake, British vs American, citation formatting, small
typos. Do not propose those.
