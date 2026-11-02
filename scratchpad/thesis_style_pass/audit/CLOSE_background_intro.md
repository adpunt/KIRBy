# Closing benchmark rows: Chapter 1 (introduction) and Chapter 2 (background), 2026-10-10

Only paragraphs with an AI marker were edited. No numbers, tables, captions, \ref/\label, headings or claims changed. TODO lines kept. Measures were not re-run: the spaCy model used by `compute_metrics.py` is not installed in this environment, so dependent-clause counts were judged by reading. Three-item lists were checked with the `TRIPLE` regex from `extra_measures.py`.

## Chapter 1

1. "In an early study with Gaussian noise added" (line 15, AI part: last four sentences only). Row: introduction connectives.
   - old: "...features also work as ordinary model inputs, so..." -> new: "...features work as ordinary model inputs too, so..."
   - old: "We then test whether distance in the hybrid tracks the label better (Chapter~\ref{chap:similarity}), and last return to noisy labels, comparing the single representations with the hybrid (Chapter~\ref{chap:noise})." -> new: "Chapter~\ref{chap:similarity} tests whether distance in the hybrid tracks the label better, and Chapter~\ref{chap:noise} returns to noisy labels, comparing the single representations with the hybrid."
   - old: "We therefore test the method on antibodies, and on chemical reactions." -> new: "For this reason we test the method on antibodies, and on chemical reactions."
2. "We compare the hybrid with each base representation" (line 39). Row: introduction connectives. old: "We also read the features it selects" -> new: "We read the features it selects".
3. "Chapter~\ref{chap:similarity} asks whether a hybrid representation repairs" (line 42). Row: introduction connectives. old: "it also tracked the label more closely, yet prediction..." -> new: "it tracked the label more closely as well, but prediction..."; old: "The benefit of a hybrid therefore has to be measured on each dataset." -> new: "The benefit of a hybrid has to be measured on each dataset."
4. "Chapter~\ref{chap:noise} returns to label noise, with the" (line 45). Row: introduction connectives. old: "although the best method changed between the two" -> new: "but the best method changed between the two".
5. "Chapter~\ref{chap:discussion} answers the research questions. It argues" (line 48). Row: introduction connectives. old: "...before a method is chosen, since no single method fits every dataset and simple methods are often as good as complex ones. It then lists the contributions, limitations and future work." -> new: "...before a method is chosen. No single method fits every dataset, and simple methods are often as good as complex ones. The chapter ends with the contributions, limitations and future work."

Introduction words per paragraph: no merge made. The short AI paragraphs (lines 39, 42, 45, 48) each describe a different chapter. The one merge that would join paragraphs saying one thing is line 39 into line 36 (both Chapter 5), but line 36 is the author's own paragraph, so it was left for her to decide.

## Chapter 2

6. "Chapter~\ref{chap:intro} set out the questions this thesis asks;" (line 69). Rows: background three-item lists, dependent clauses. old: "We first describe the molecular representations and models we use, along with molecular similarity and activity cliffs. We then turn to label noise: where it comes from in bioactivity data, how we inject it and how methods for learning with noisy labels handle it." -> new: "We first describe the molecular representations and models we use. We then cover molecular similarity and activity cliffs. Label noise follows, with its sources in bioactivity data and the ways we inject it, and then the methods for learning with noisy labels."
7. "Support vector regression fits a function that is" (line 129). Row: dependent clauses. Four stacked sentences split:
   - "It uses the $\epsilon$-insensitive loss, under which errors..." -> "It uses the $\epsilon$-insensitive loss. Under this loss, errors..."
   - "...only through the kernel, so with the radial basis function (RBF) kernel every feature enters..." -> "...only through the kernel. With the radial basis function (RBF) kernel, every feature therefore enters..."
   - "A wrong label has limited effect: an error inside the $\epsilon$ tube changes nothing, and outside it the loss grows only linearly and the weight of each molecule is capped at $C$" -> "A wrong label has limited effect. An error inside the $\epsilon$ tube changes nothing, and outside it the loss grows only linearly. The weight of each molecule is also capped at $C$"
   - "Its main limitation is computational cost, because training operates on..." -> "Its main limitation is computational cost. Training operates on..."
8. "Much of cheminformatics rests on the similar property" (lines 168-169). Rows: three-item lists, dependent clauses.
   - "...principle, which states that structurally similar molecules..." -> "...principle: structurally similar molecules..."
   - "Nearest-neighbor prediction, GP kernels and many noise detection methods rely on it, yet similarity is defined..." -> "Nearest-neighbor prediction and GP kernels rely on it, and so do many noise detection methods. Yet similarity is defined..."
   - "...far apart under a PDV, and a computed similarity value..." -> "...far apart under a PDV. A computed similarity value..."
9. "As the number of dimensions grows, the distances" (line 188). Row: dependent clauses. "...high-dimensional and sparse, and when many of their features..." -> "...high-dimensional and sparse. When many of their features..."
10. "These sources can interact, and the noise they" (line 226). Row: dependent clauses. "...from a simple Gaussian: it can be heteroscedastic..." -> "...from a simple Gaussian. It can be heteroscedastic..."
11. "The problem of training a model when some" (line 301). Rows: three-item lists, dependent clauses.
   - "...how wrong each label looks, and the score it gives..." -> "...how wrong each label looks. The score it gives..."
   - "Noise detection is a detector alone, whereas noise mitigation is a selector followed by deletion, down-weighting or moving the label, after which any model can be fitted (Figure...)." -> "Noise detection is a detector alone. Noise mitigation is a selector followed by deletion or down-weighting of the selected molecules, or by moving their labels. Any model can then be fitted (Figure...)."
   - "...changes the loss, the kernel, a noise term, the training loop or the model, as in co-teaching \citep{Han2018} and Meta-Weight-Net \citep{...}." -> "...changes the model or its training. Co-teaching \citep{Han2018} and Meta-Weight-Net \citep{...} are two examples." (The five things that can change stay listed in Table tab:noise_terms.)
   - "...into five groups, all of which act during training and so count as noise-robust training: robust architectures, ..." -> "...into five groups: robust architectures, .... All five act during training and therefore count as noise-robust training."
   - "...before a refit, which counts as noise mitigation." -> "...before a refit. Such a step counts as noise mitigation."
12. "No single method is expected to win, since" (line 348). Row: dependent clauses.
   - "No single method is expected to win, since averaged over all problems no learning algorithm..." -> "No single method is expected to win. Averaged over all problems, no learning algorithm..."
   - "Neighbor-based cleaning, which dates back to editing a training set by its nearest neighbors \citep{Wilson1972}, relies on the similar property principle." -> "Neighbor-based cleaning relies on the similar property principle. It dates back to editing a training set by its nearest neighbors \citep{Wilson1972}."

Expected effect (not re-measured): background three-item lists by the regex fall from 8 to 2 (the five-group list from Song2022 and the scaffold definition at line 238 stay), over about 55 sentences instead of 40. Dependent clauses lose about 8 and gain about 15 sentences.
