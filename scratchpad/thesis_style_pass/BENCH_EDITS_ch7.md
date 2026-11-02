# Chapter 7: benchmark edits, 2026-10-10

Backup: `thesis.tex.before_benchmark_edits_2026-10-10`. All edits are to paragraphs with a "% TODO: manual review" line above them. No number was added or changed in value.

Prose word count of the chapter: comment lines removed, and table, figure and equation environments left out.
- Backup: 10,498 words
- Now: 10,169 words

| Paragraph (first eight words) | Words before → after | What changed |
|---|---|---|
| Chapter 6 showed that on the five small-molecule (now "In Chapter 6, on the five...") | 107 → 104 | Booster "showed" cut. Past tense for the Chapter 6 result. "doing something about them" became "acting on them". "already" cut. |
| Every method in this chapter reads a feature | 84 → 80 | The "of Chapter 5 ... as Chapter 4 showed" chain became two parenthetical pointers. |
| Most experiments first run a screen of two | 178 → 135 | The run of screen and full-run levels moved into a new Table `tab:noise:levels` (numbers copied verbatim, SOURCE comment added). The paragraph points to it. Draws, seeds and the nesting rule stay in the text. |
| A neighbor detector needs a space to find | 101 → 100 | "without touching the held-out molecules" became "without the held-out molecules". "remaining" became "other". |
| Most methods read the suspicion score rescaled to | 83 → 77 | "usually the top 5%, 10% or 20%" replaced by a pointer to Table `tab:noise:mitigation`, which lists those shares. |
| Five controls ran, each inside one experiment and | 71 → 69 | A 44-word sentence split into three, each control paired with its method. "never averaged" cut (repeats "on its own"). "a weaker claim" became "weaker evidence". |
| QM9 is the clean-label control. Its HOMO--LUMO | 97 → 93 | Hedges "would" and "may" cut. "If deleting never helps" became "does not help". Active "we read", "we give". |
| Three things hold for every result in this | 142 → 130 | "so we state them once" cut. "First" and "Second" added (the text already had "Third"). Two sentences on the picks joined into one. "time and computational constraints" became "time and compute". |
| Table 7.x shows how often each group of | 29 → 27 | "In it, one condition is" became "A condition is". |
| The table counts outer fold 0 only. Six | 37 → 31 | First sentence cut; the paragraph above already says outer fold 0. |
| These counts are from outer fold 0. Three | 46 → 38 | Now opens with a pointer to Table `tab:noise:res:mitigation_broad`. Two sentences on folds joined. |
| On the labels as delivered, the control is | 123 → 111 | Opening sentence restated the Methods control. It was cut, and the control is named inside the first result sentence. |
| The hERG K_i smoothing gain on the hybrid | 45 → 36 | First sentence repeated the second. Cut. |
| This section uses injected damage alone (Section 7.2.2). | 113 → 107 | The Chapter 3 link sentence shortened. "exactly" cut. |
| Figure 7.x applies this test to the two | 140 → 116 | 20 numbers cut to 6. The figure carries the precisions, break-evens and RMSEs. One sentence states the pattern: precision stayed below the break-even at every share on hERG K_i and Caco-2, and deleting raised the RMSE. The draw caveat merged into one sentence at the end. |
| The gain holds on the further folds where | 59 → 54 | "where they exist" cut. "the network on the hybrid" became "the network's hybrid". Caco-2 sentence reordered. |
| Deleting the clipped molecules is worse than leaving | 58 → 51 | Booster "never" cut: "without them a model never sees a high label" became "the top of the label range is missing from training". |
| A separate experiment searched the hybrid on the | 128 → 124 | Article chains cut ("on the clean or on the damaged labels"). |
| Table 7.x gives the expanded sweep by domain. | 53 → 46 | "almost never" became "rarely". The reaction-side caveat shortened and the dataset named. |
| On the first outer fold, noisy labels in | 119 → 118 | Rewritten as findings with "we". Passive "could be found" became "we found". The long final sentence split into two plain claims. Still one number word ("three antibody datasets"). |
| Most results rest on the first outer fold. (plus the two limitation paragraphs after it) | 71 + 30 (list) + 76 + 112 = 289 → 136 | Three limitation paragraphs and the numbered list merged into one paragraph. Kept: fold 0 caveat, the author's confirmation order as two sentences, no control for most methods, three methods with no QM9 result, QM9 as the only clean control, and the datasets with no noise results. Cut: the list of which results have extra folds, "Five controls ran" (already in Methods), the put-back caveat (already in Results), the ground-truth and assay-floor sentences, and the MDR1--MDCK sentence (already in Results twice). |

Paragraphs checked and left unchanged: the rest of the AI-marked Results paragraphs, plus Methods paragraphs 3960, 4000, 4147 and 4166 by their old line numbers. No edit made any of them simpler without losing content, or the worklist's change did not apply. For example, the hedges in "The network gains from cleaning more often" and "Cleaning keeps the hybrid ahead" are accurate and stay.

Tables built: 1 (`tab:noise:levels`, damage levels in the screen and in the full run).
