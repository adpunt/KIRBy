# Chapter 1: suggestions for the author's paragraphs (not edited)

Worklist ranks in the top 150 for Chapter 1: three paragraphs, all the author's.

## 1. "Chapter 8 summarizes each chapter and argues that" (worklist rank 18, Outline)
Benchmark flag: hedges 6.7 per 100 words against a reference mean of 0.9 ("argues", "should", "often"); one 37-word sentence with three subordinate clauses.
Suggested edit (also matches Chapter 8, which now says "We therefore recommend modeling the noise in a dataset before choosing a method"):
- Old: "Chapter~\ref{chap:discussion} summarizes each chapter and argues that the noise in a dataset should be modeled before a method is chosen, since no single method fits every dataset and simple methods are often as good as complex ones. It then gives limitations, future directions and conclusions."
- New: "Chapter~\ref{chap:discussion} summarizes each chapter. It recommends modeling the noise in a dataset before choosing a method, because no single method fits every dataset and simple methods were often as good as complex ones. It ends with limitations, future directions and conclusions."

## 2. "Chapter 2 introduces molecular representations, model families, molecular" (worklist rank 52, Outline)
Benchmark flag: nominalizations 14.8 per 100 words against a reference mean of 5.3; second sentence is 38 words and has a stray comma before "and evaluation metrics".
Suggested edit (split the second sentence; content unchanged):
- Old: "It defines the seven noise conditions used in Chapter~\ref{chap:robustness} and the three groups of methods for noisy labels (noise detection, noise mitigation and noise-robust training), and closes with artificial noise injection, and evaluation metrics."
- New: "It defines the seven noise conditions used in Chapter~\ref{chap:robustness} and the three groups of methods for noisy labels: noise detection, noise mitigation and noise-robust training. It closes with artificial noise injection and evaluation metrics."

## 3. "QSAR models are of course, sensitive to the" (worklist rank 79, opening section)
Benchmark flag: 3 or more numbers in 16.7% of its sentences against a reference mean of 1.7%; 8 numbers in 317 words (reference introductions carry about 0.5 per 100 words). Hedges flagged: "about", "may", "indicates", "argued".
Suggested edit (keep the log-unit values from the sources; drop the fold conversions, since the paragraph already says one log unit is a tenfold difference; also drop "of course,"):
- Old: "QSAR models are of course, sensitive to the quality of the input data on which they are trained."
- New: "QSAR models are sensitive to the quality of the data they are trained on."
- Old: "\citep{Kalliokoski2013, Kramer2012}, which is about a 4.8-fold and a 3.5-fold difference in concentration."
- New: "\citep{Kalliokoski2013, Kramer2012}."
- Old: "lowers the mean absolute difference from $0.50$ to $0.27$ log units \citep{landrum2024}, or from about a 3.2-fold to a 1.9-fold difference."
- New: "lowers the mean absolute difference from $0.50$ to $0.27$ log units \citep{landrum2024}."
This removes four numbers and about 25 words.
