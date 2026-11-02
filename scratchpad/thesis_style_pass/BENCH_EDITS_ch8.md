# Chapter 8 (Discussion and conclusions): benchmark edits, 2026-10-10

Backup: thesis.tex.before_benchmark_edits_2026-10-10.
Chapter prose words (comment lines removed): backup 1,532, now 1,417.
Paragraphs edited: 13 (three Limitations paragraphs merged into one). Tables built: 0.
Every edited paragraph carries a "% TODO: manual review" line. No number added or changed in value.

| First eight words | Words before -> after | What changed |
|---|---|---|
| This thesis set out to handle label noise | 43 -> 35 | Cut "in turn" and the separate closing-list sentence; folded limitations and future work into the list. |
| The first half of the thesis asked four | 132 -> 132 | Cut the second number (representation 8.1%); now "the representation far less". 47.2% kept. Comment carrying both numbers kept. |
| Chapter~\ref{chap:flaws} then asked why published methods for learning | 74 -> 66 | "an exploratory study carried out earlier in this project" -> "an earlier exploratory study"; "at least 95% of the molecules" -> "nearly every molecule". 1,815 of 2,048 kept. |
| The second half asked whether the representation can | 104 -> 95 | Cut the LogD SVM RMSE pair (0.4930 to 0.4302; still in the SOURCE comment); added "we found that". |
| Second, distance in a hybrid tracked the label | 69 -> 46 | Cut the hERG Ki Spearman sentence (0.0729 to 0.2625); claim kept in words. |
| Third, noisy labels could be found and corrected | 66 -> 65 | Three RMSE values (0.6354, 0.4795, 0.4314) replaced by "censored EM removed most of the error that clipping added to the network" (0.156 of 0.204 RMSE recovered, 76%). |
| Label noise was handled, but no single method | 90 -> 69 | Opening two sentences cut to "No single method handled label noise on every dataset."; the task/data/model/noise sentence repeated the Concluding remarks. |
| Most noise results rest on the first outer (+ "Every clipped-label result..." + "The hybrid needs data...") | 65 + 48 + 71 = 184 -> 146 | Merged into one Limitations paragraph (blank lines removed; TODO, RESULT PENDING and SOURCE comments kept in place). Cut: "where further folds exist they mostly agree", "with one property tested" (stated in Future work), the search-cost sentence, the date "8 October 2026", "yet". RESULT NEEDED placeholder kept. |
| A hybrid could also be the input to | 55 -> 52 | Hedges "could" -> "can"/"may"; "the code for this" -> "the code". |
| Label noise in molecular data can be handled. | 53 -> 51 | "The noise in a dataset" -> "A dataset's noise". |
| A new foundation model appears every few months | 77 -> 75 | "add the new embedding to the representations searched and run the search again" -> "add the new embedding to the search and rerun it". |

Not edited (AI-marked, read and left): "Fourth, importance helped...", "We therefore recommend modeling...", both Contributions lists, "The hybrid search needs only...", "The noise methods need more...", "The features also matter."
