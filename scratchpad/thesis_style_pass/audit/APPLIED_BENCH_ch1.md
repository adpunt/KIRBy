# Applied: benchmark fixes, Chapter 1 and abstract (2026-10-10)

Every passage was re-found; none had changed since the audit. No number, \ref, \label or table touched.

## Abstract (main.tex)
- Fix 1: applied ("models, whose labels carry ...").
- Fix 2: applied (semicolon before "random forests").
- Fix 3: adapted. The 36-word merge is over 30 words; split differently: "... do not relate to the label, yet each still adds to the distance between molecules. Methods built on that distance therefore inherit the problem."
- Fix 4: applied ("..., but on small molecules the hybrid stayed ahead ...").
- Extra: "seven kinds of label noise" -> "seven types of label noise" (Chapter 1 and Chapter 3 say "types").
- Abstract now 298 words by `wc -w` (limit 300).

## Chapter 1 (thesis.tex)
- Fix 5: adapted. Merged the two sentences, but wrote "and last return to noisy labels, comparing the single representations with the hybrid" instead of the proposed "to test each method on both ...", which added a claim.
- Fix 6: skipped. With fix 5 it would put two 33-word sentences side by side (uniform length again); the audit allowed taking only one.
- Fix 7: adapted. Merged into one 30-word sentence keeping "random features of the same size"; the proposed "at the same number of features" version was 32 words.
- Fix 8: applied (two groups: antibodies and reactions; data at separate sites; "also" moved here).
- Fix 9: adapted. The proposed 41-word sentence was over 30 words. Now: "On antibodies the gain depended on the dataset: on two of them ... vary. An ensemble often did better on the smaller antibody datasets. On the reactions the hybrid gained nothing." (29, 10, 7 words).
- Fix 10: applied ("it also tracked").
- Fix 11: applied ("..., yet prediction from neighbors was worse on several of them. The benefit of a hybrid therefore has to be measured on each dataset.").
- Fix 12: applied ("..., although the best method changed between the two.").

## Extra pass (style targets)
- Line 42: "We build hybrids for distance" (coined) -> "We build hybrids whose features are chosen for distance".
- Left: "labels as delivered" (line 45). It is the term Chapter 7 uses throughout (about 10 times); changing it here would break the match.
- Left: the four numbered questions, the Chapter 8 outline paragraph and Collaborations, as the audit recommends.

Counts: 7 applied, 4 adapted, 1 skipped, 2 extra edits (1 abstract, 1 Chapter 1).
