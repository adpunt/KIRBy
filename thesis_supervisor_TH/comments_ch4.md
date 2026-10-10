
### TH-p87a (old PDF page 87, margin y=353)
COMMENT: This introduction paragraph is not easy to read (at least for a french scientist ;-) ) and should clearly say why we look a features/similarity in the context of noisy data.
  HIGHLIGHT p87 y=341-478: This chapter asks whether the standard ways of comparing molecules suit the dataset and task at hand. Because general-purpose representations are built for all of chem- istry, we first count how many of their features vary on a single dataset. On hERG Ki, most ECFP4 features take a single value in almost every molecule. In an ex- ploratory study of learning with noisy labels, the configurations with the lowest retention under noise all used a distance, and noise detection by neighbors usually made the model worse.
  HIGHLIGHT p87 y=543-761: One of the fundamental assumptions in cheminformatics, the similar property princi- ple (Chapter 2), is that two structurally similar compounds display similar chemical properties [73]; we call it the similarity assumption. While there are important ex- ceptions, namely activity cliffs [74, 78], this is a well-established concept. However, there is no single way to define similarity between compounds, and this is where the assumption begins to break down in practice. When defining similarity between two molecules, we typically rely on some pairwise mathematical function, or coeffi- cient, that takes in two molecular representations as input. Two factors vary here, the representation and the coefficient. Bajusz et al. [76] found that among stan- dard coefficients, the choice between them matters little. This chapter questions the underlying representation.

### TH-p87b (old PDF page 87, margin y=660)
COMMENT: Nice paragraph

### TH-p88a (old PDF page 88, margin y=193)
COMMENT: Similarity is in the eyes of the beholder ;-)
  HIGHLIGHT p88 y=187-222: Different representations also disagree about which molecules are close
  HIGHLIGHT p88 y=510-566: None of these measures ask specifically about whether the specific features within these representations matter for the prop- erty at hand.

### TH-p88b (old PDF page 88, margin y=343)
COMMENT: General comment: It is worth distinguishing between static task agnostic representations (e.g. ECFP) and representations learned against the target task (e.g trained NN embedding layer). We don’t expect the former to necessarily define a smooth property landscape since they were not fine tuned for that. The latter however should create a smoother landscape with respect to the task and activity cliff would be real provided the global performance of the model is reasonable.

### TH-p89a (old PDF page 89, margin y=542)
COMMENT: ‘is negatively impacted’ ?
(the similarity does not inherit the features but their flaw)
  HIGHLIGHT p89 y=571-586: inherits them
  HIGHLIGHT p89 y=672-687: Each PDV

### TH-p89b (old PDF page 89, margin y=667)
COMMENT: ‘For instance, each…’

### TH-p90a (old PDF page 90, margin y=478)
COMMENT: What is the proportion of 0 (absence) vs 1 (presence) of these values ?
  HIGHLIGHT p90 y=498-513: 1,815 near-constant features out of 2,048
  HIGHLIGHT p90 y=640-655: on the distance

### TH-p90b (old PDF page 90, margin y=634)
COMMENT: ‘..depends on the distance metric’ ?


### TH-p90c (old PDF page 90, margin y=694)
COMMENT: Which standardization here? Is the standardization changing the value of the feature ? How?

### TH-p90d (old PDF page 90, margin y=727)
COMMENT: This paragraph is difficult to read. What is the trasformation from 1 -> 4.36 ?
  HIGHLIGHT p90 y=700-715: standardization
  HIGHLIGHT p90 y=741-776: A bit on in 5% of molecules becomes 4.36 on those molecules, and a bit on in one of the 1,412 hERG Ki molecules becomes 37.56.
  HIGHLIGHT p90 y=579-594: Weis

### TH-p92a (old PDF page 92, margin y=595)
COMMENT: This section 4.2.1
needs to be simplified so that the reader understand easily the impact of conste/near-constant feature on the similarity metric
  HIGHLIGHT p93 y=506-542: Any representation read from the 2D graph merges stereoisomers, since the two share an identical graph.

### TH-p93a (old PDF page 93, margin y=506)
COMMENT: The 2D graph express stereoisomers via stereobonds.
  HIGHLIGHT p93 y=587-623: d 49 in more than that.

### TH-p93b (old PDF page 93, margin y=562)
COMMENT: Reader will be frustrate to not know what is ‘more’, different sources, embedding collision, …? You go in more details next but this sentence finishes a bit brutally :-) )
  HIGHLIGHT p94 y=146-161: same InChIKey,
  HIGHLIGHT p94 y=227-311: The lowest error any model can reach is therefore RMSEmin = � 1 1 n � G � G � i∈G (y (yi − ¯yG ¯yG)2 , (4.1)

### TH-p94a (old PDF page 94, margin y=235)
COMMENT: This is the lower bound for model RMSE based on the original labels not the reassigned group mean label I assume ?
How do you differentiate the training/test data in this lower bound assumption (the RMSE only plies to the test/validation data)
  HIGHLIGHT p94 y=126-141: A tied molecule is a repeat

### TH-p94b (old PDF page 94, margin y=68)
COMMENT: You mean a repeat is based on a InChI key with stereo-chemistry on  worth specifiying) and the repeat implies  the InChI key was computed with all stereo-information. 
In this paragraph it could be inferred that there are only 3 types of ties: ‘repeat’ ‘failure’ and ‘distinct’, isn’t there a   ‘merge’ (stereo-chemistry merge) ?
  HIGHLIGHT p95 y=651-727: [131]. A tied pair has similarity 1, so its index is infinite whenever the labels differ. Such a tie is the most extreme activity cliff possible, and an artifact of the rep- resentation. A nearest-neighbor method sees a tied neighbor at distance 0 with a different label. It may then flag a correct label as noise (Section 4.4.2).

### TH-p95a (old PDF page 95, margin y=643)
COMMENT: Does it make sense to  measure the SALI index inside a tie group ? If we measure the index prior to group normalisation of the label we would measure a mix real activity-cliff (e.g. stere-effect) and measurement variability.
  HIGHLIGHT p97 y=167-202: All a similarity coefficient can do is compare every feature, yet it cannot tell which features matter to the prediction task at hand.

### TH-p97a (old PDF page 97, margin y=163)
COMMENT: Slightly different for representation learned against the target task.
  HIGHLIGHT p98 y=409-667: We ask two questions of a distance. First, across all pairs, does distance track the difference in property? We summarize the neighborhood-behavior plot of Patterson et al. [136], descriptor distance against activity difference for every pair, as the Spearman correlation between pair distance and absolute label difference. Second, can each molecule’s label be predicted as the mean label of its five nearest neighbors? We computed both for every base representation of seven regression datasets: the six small-molecule regression datasets of Table 5.14 (Chapter 5) and HLM clearance, a third Biogen dataset not used elsewhere in this thesis. Binary representations were compared by Tanimoto distance, and all others by Manhattan distance (next subsection) on standardized, non-constant features. For the all-pairs correlation, datasets above 2,000 molecules were reduced to a random sample of 2,000. For the neighbor prediction, each molecule’s five neighbors were drawn from other scaffold groups.

### TH-p98a (old PDF page 98, margin y=424)
COMMENT: It would be nice to have a figure that summarizes the experiment so that the reader can easily understand where the 70 combinations come from.
  HIGHLIGHT p99 y=106-202: Figure 4.3 shows why the two differ on hERG Ki. Of the 996,166 pairs, 93% have an ECFP4 Tanimoto similarity below 0.20 and 1.1% reach 0.55. The median label difference is 0.81 pKi over all pairs but 0.20 to 0.30 pKi above a similarity of about 0.4, though 8.3% of the 774 pairs at 0.8 or above differ by more than 1 pKi.

### TH-p99a (old PDF page 99, margin y=19)
COMMENT: It would be good to finish this  resulting listing with a take home message for each representation otherwise the reader is loosing track of what is the goal. e.g. closer compound have closer labels => similarity principle etc.

The reader is flooded with number for different type of measures and it needs  simple (one sentence) recap/summary from time to time ;-)

For instance do these numbers mean ECFP4 is following better the similariy principle than Mordred?

### TH-p99b (old PDF page 99, margin y=538)
COMMENT: What is the intermediate conclusion/ take home message of this section?


### TH-p101a (old PDF page 101, margin y=248)
COMMENT: At this point you have  presented a lot of metrics/comparisons.
It would be good to have a short recap/summary of the type of flaws and their origin to prepare the reader for the next section.
  HIGHLIGHT p102 y=207-404: The study tested 12 LNL methods on the QM9 HOMO–LUMO gap (10,000 molecules, scaffold splits), adding Gaussian noise with σ from 0 to 1.0 in steps of 0.1 to the training labels only. The representations were ECFP4, Sort & Slice [6], PDV, canonical SMILES and randomized SMILES. Depending on the method, a configuration could add the model’s own uncertainty, a distance between molecules (Tanimoto, Euclidean, Manhattan, cosine or Mahalanobis), an alternative GP kernel or an alternative loss function, giving 221 configurations. Four methods, small- loss selection, confident learning, distance-based selection and multistage cleaning, remove suspected molecules and retrain. These are noise mitigation methods; the other eight are noise-robust training methods.
  HIGHLIGHT p102 y=268-283: PDV

### TH-p102a (old PDF page 102, margin y=293)
COMMENT: It would be nice to have a visual but only if you have time (not critical)

### TH-p102b (old PDF page 102, margin y=389)
COMMENT: Just out of curiosity, did you observe cases of retention ratio > 1 (better performance with noisy data ?, I expect not but was curious)

### TH-p102c (old PDF page 102, margin y=563)
COMMENT: Retained from ..to …? Does it mean between sigma =0 and sigma =1 (or 0.6) ? What is the % measuring?
  HIGHLIGHT p102 y=409-445: R2 under noise divided by R2 with clean labels,
  HIGHLIGHT p102 y=551-627: The methods that held up estimate noise from the model’s own loss or uncer- tainty. DivideMix [141] retained the most, 86.7% (standard deviation 0.4%), fol- lowed by small-loss selection [102] at 84.2% (2.1%) and the uncertainty curriculum [142] at 82.7% (3.5%).

### TH-p104a (old PDF page 104, margin y=387)
COMMENT: Nice: The story is clear here , more noise less recovery and the amplitude of the effect is representation dependant 
  HIGHLIGHT p104 y=396-462: Table 4.6: Recovery rate in R2 on the QM9 HOMO–LUMO gap when neighbor-based noise scores are used as sample weights, for each representation used to find the neighbors, at two levels σ of Gaussian label noise. Recovery is the share of the R2 lost to noise that cleaning regains, so negative values mean cleaning made the model worse. Positive is the share of configurations with recovery above zero.
  HIGHLIGHT p104 y=667-682: The study also found neighbors in an early hybrid of 100 PDV and MHG-GNN

### TH-p104b (old PDF page 104, margin y=643)
COMMENT: We haven’t yet explained hybrid representaiotns at this point. Maybe I misunderstood this sentence, it felt confusing.

### TH-p106a (old PDF page 106, margin y=163)
COMMENT: Nice recap !

We could have investigated how many structure are merged across representation. This give an idea of the potential of breaking ties by using hybrid representations
  HIGHLIGHT p106 y=-2147483648-2147483520: 
  HIGHLIGHT p106 y=442-465: Table 4.7: Flaws of standard representations and similarity, the requirement each sets for a representation, and where the requirement is addressed, in Chapter 5 unless stated.