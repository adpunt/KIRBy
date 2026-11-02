# Benchmarks: the three reference theses, your writing, and the AI-written text

Changed tonight (2026-10-10): 67 tables were added for results claims, the acknowledgements gaps were filled, sections that jump straight to a subsection went from 6 to 3, and 366 AI-written sentences in results, discussion and conclusion sections were new or rewritten (281 at the first re-run, 85 more by the second, later in the evening); every measure below was re-run on `thesis.tex` as it is now.

Language measures split your own paragraphs (no AI marker) from AI-marked paragraphs, so your writing does not dilute the AI statistics. Floats, captions and equations are excluded. Scripts, run in order: `benchmark/segment_full_author.py current`, `split_hers_ai.py`, `extra_measures.py`, `predictability_cleaned.py`, `predictability_report.py`, then `build_benchmark_table.py` for the tables below. The version before tonight's fixes is in `benchmark/before_tonight_rerun/`; the version from the first re-run tonight is in `benchmark/before_rerun_2026-10-10_late/`.

## Structure and content

"Done" re-checked against `thesis.tex` and `main.tex` as they are now (2026-10-10 evening), except rows marked "not re-checked tonight".

| Benchmark | Buttenschoen | Dablander | Vost | What it means for your thesis | Done? |
|---|---|---|---|---|---|
| Chapter opening links back | Unnumbered paragraph, every chapter | Link back by number | "Preface" section | First sentence of each Introduction names the previous chapter | Yes, Ch3–7 (not re-checked tonight) |
| Chapter ends point forward | Most | Rarely | Every chapter | Last Conclusions paragraph names the next question | Yes (not re-checked tonight) |
| Text between section title and first subsection | Yes | Roadmap paragraph | not recorded | Lead-in where a section jumps to a subsection | Partly: 3 sections still jump straight to a subsection (was 6): Ch3 Results, Ch4 Methods, Ch4 Results |
| Title case | Sentence case | Title Case | Sentence case | Sentence case | Yes (not re-checked tonight) |
| Section titles | 1–8 words, topics, parallel | 2–7 words | 2–9 words | Topic noun phrases, no questions or claims | Yes (not re-checked tonight) |
| Numbers in results text per 100 words | 3.3–10 (figure behind each) | 1.0–2.1 | 2.2–3.7 | Your 10 Oct rule: pattern + pointer to table, bold what matters | Yes: 2.6 numbers per 100 words of results prose; 67 tables added today (41 table environments at 08:00, 108 now), and every `\ref{tab:...}` in the text resolves to a label |
| Results prose words per figure/table | 268–577 | 601–2,270 | 449–805 | More floats in results | Yes: Ch3 242, Ch4 412, Ch5 140, Ch6 115 (was 998), Ch7 139 words of results prose per figure or table |
| Results sentences pointing to figure/table | 18% | 3.6% | 3.1% | Open results with the table/figure | Partly: 18% of results sentences now cite a table or figure in brackets; sentences whose job is to point (move M2) are 2.2% (yours 1.2%, AI 2.7%) |
| Limitations | Per-chapter discussion + Sec 6.2 | not recorded | not recorded | Short, specific | No: 14% of conclusion sentences are limitations (was 18% before tonight, 15% at the first re-run); references 0–5.5% |
| Research questions | not recorded | Recap (i)–(iii) | not recorded | Ch1 stated, Ch8 answered in order | Yes (not re-checked tonight) |
| Abstract | 398 words, 6 paras | 250, 2 | 568, 5 | ≤300 words, no numbers | Yes, 298 words, 6 paragraphs |
| Introduction length | 4,218 | 2,158 | 9,386 | About Dablander's | Yes, 2,045 words of prose in Ch1 |
| Final chapter length | 1,593 | 896 (+1,788) | 1,015 | ~1,000–1,600 | Yes, 1,373 words of prose in Ch8 |
| Hedges per 1,000 words | 2.1 | 4.5 | 2.9 | Hedge only weak evidence | Partly (not re-checked tonight; per-section hedge rates are in the language tables below) |
| Statistics | No tests; bootstrap 95% in one chapter | Error bars explained in captions | not recorded | Per-fold scores, spread, folds, controls | Yes for the tables: 67 tables added today and every table reference resolves. Not re-checked tonight: that each of the 57 claims maps to one of them, and that each table shows per-fold spread |
| Negative/positive results | Negative headline repeated plainly | not recorded | not recorded | Plain, no inflation | Partly: Ch8 only (not re-checked tonight) |
| Originality | Ranked contributions in final chapter | Per chapter in intro | Contributions section | Abstract, Ch1, Ch8 | Yes (not re-checked tonight) |
| Caption length (mean words) | 73.5 | 61.1 | 49.8 | Title phrase; metric, unit, direction | Yes: 142 captions, mean 52 words, median 50 (inside the references) |
| Figure/table references | "Figure" in full | not recorded | not recorded | One consistent form | Yes (not re-checked tonight) |
| Decimals | 1–2 | 3 | 2–3 | 3 scores, 1 percentages | Mostly: 4 four-decimal scores left in prose (0.0020 RMSE, Ch5 Results; 0.0015 RMSE, Ch6 Methods; 0.5072 and 0.3980 RMSE, Ch7 Results); 1.4826 and p = 0.0625 are constants, not scores |
| Chemistry wording | – | – | – | 30 problems in chemistry guide | 22 done, 5 partly, 3 not (not re-checked tonight) |
| Front matter order | Abstract before acknowledgements | Abbreviations after lists | List of figures | Abstract, AI statement, acknowledgements, contents, lists, abbreviations | Changed today, so the target order no longer matches: `main.tex` now runs Statement of Contribution, Acknowledgments, contents, Abstract, AI statement, list of figures, list of tables, abbreviations (abstract moved after the contents, per the University regulation quoted in a `main.tex` comment) |
| Acknowledgements | 850 words, personal | Supervisors one by one | not recorded | Fill [NAMES] gaps | Yes: no [NAMES] gaps left; family, grandmother and friends filled. Two paragraphs still carry an AI marker (Lhasa and Adrian Fowkes; OPIG datasets) |
| AI-use statement | none | none | none | Required | Yes |
| "you" in body | 0 | 0 | 0 | Your text, your choice | 27 prose sentences contain "you"/"your" (counted on segmented prose; the earlier 33 was a different count) |

## Language, by section type

One row is one measure. Reference and "Yours" columns are per 100 words, per sentence or a percentage, as the measure name says. "AI before tonight's fixes" is the value this file held before tonight's re-run; "AI now" is the re-run.

Status:
- **done**: AI now is inside the range of the three references, or between that range and your own value.
- **closer**: still outside the range, but nearer to it than before.
- **not a target**: a measure you set by preference (shorter sentences, fewer sentences over 30 words, fewer passives, no phrases from your AI writing guide), or one where your own writing sits outside the range on the same side as the AI text.
- **not done**: anything else; the cell says which side of the range it is on and whether it moved.

Rhetorical moves: each sentence of a results, discussion or conclusion section is coded with one move (M1–M11, the scheme in `moves_*.csv`). The 366 AI-written sentences that were new or changed tonight were coded by reading, in `benchmark/moves_tonight_new_sentences.csv` (281 at the first re-run, 85 at the second); every AI-written sentence in these sections now has a code. 59 of your results sentences, 10 conclusion and 1 discussion sentence have no code (not re-coded tonight). A move share is shown only when at least 20 sentences are coded, so your discussion column is empty (10 sentences).

Predictability: GPT-2 small on text cleaned the same way for every corpus (citations, placeholders, tokens with digits and acronyms removed; sentences under 8 words dropped), on up to 400 evenly spaced sentences per cell. The uncleaned GPT-2 rows of the previous version are dropped.

### Introduction

Size now: buttenschoen: 4,182 words · dablander: 6,993 words · vost: 4,144 words · yours: 4,687 words · AI-written: 1,156 words

**Sentences and words**

| Measure | Buttenschoen | Dablander | Vost | Yours | AI before tonight's fixes | AI now | Status |
|---|---|---|---|---|---|---|---|
| Words per sentence | 27.3 | 30.8 | 25.3 | 20.8 | 17.2 | 18.6 | not a target |
| % sentences over 30 words | 39.2 | 43.2 | 22.0 | 15.1 | 2.9 | 1.6 | not a target |
| Dependent clauses per sentence | 0.85 | 0.81 | 0.71 | 0.80 | 0.49 | 0.56 | closer |
| Words per paragraph | 116.2 | 145.7 | 112.0 | 114.3 | 80.5 | 77.1 | not done: below the references (112.0–145.7, yours 114.3); was 80.5, moved away |
| Lexical density % | 62.1 | 61.7 | 60.6 | 58.9 | 56.3 | 55.7 | not a target |
| Nominalizations per 100 words | 4.1 | 5.5 | 6.1 | 6.4 | 5.2 | 5.1 | done |
| % passive sentences | 24.8 | 30.8 | 27.4 | 20.0 | 17.1 | 17.7 | not a target |
| we/our/I per 100 words | 1.4 | 1.7 | 1.6 | 0.66 | 1.7 | 1.5 | done |
| Hedges per 100 words | 0.86 | 1.0 | 0.72 | 0.85 | 0.66 | 0.61 | not done: below the references (0.72–1, yours 0.85); was 0.66, moved away |
| Boosters per 100 words | 0.36 | 0.31 | 0.31 | 0.38 | 0.25 | 0.26 | closer |
| AI-guide phrases per 100 words | 0.19 | 0.17 | 0.48 | 0.06 | 0.0 | 0.0 | not a target |
| Numbers per 100 words | 1.1 | 0.46 | 0.24 | 0.53 | 0.08 | 0.09 | closer |
| % sentences with 3+ numbers | 3.3 | 1.8 | 0.0 | 1.3 | 0.0 | 0.0 | done |

**Diversity, variability and connectives**

| Measure | Buttenschoen | Dablander | Vost | Yours | AI before tonight's fixes | AI now | Status |
|---|---|---|---|---|---|---|---|
| Lexical diversity, MTLD (higher = more varied) | 82.9 | 104.1 | 124.7 | 87.5 | 78.5 | 82.6 | closer |
| Lexical diversity, MATTR-50 % | 80.9 | 82.8 | 84.5 | 80.7 | 79.4 | 79.6 | not a target |
| Sentence length SD (words) | 12.1 | 15.0 | 8.89 | 9.32 | 6.52 | 6.73 | closer |
| Sentence length coefficient of variation | 0.46 | 0.51 | 0.36 | 0.45 | 0.39 | 0.37 | done |
| Mean change in length between consecutive sentences | 13.1 | 14.8 | 10.2 | 10.5 | 8.36 | 7.49 | not done: below the references (10.2–14.8, yours 10.5); was 8.36, moved away |
| Connectives per 100 words | 2.01 | 1.99 | 2.39 | 1.65 | 2.70 | 2.03 | done |
| Distinct connectives / connectives used | 0.30 | 0.28 | 0.29 | 0.38 | 0.38 | 0.43 | not a target |
| % sentences opening with a formulaic connector | 6.54 | 8.37 | 6.10 | 3.56 | 0.00 | 0.00 | not a target |
| Three-item lists per 100 sentences | 11.1 | 10.1 | 7.32 | 13.8 | 11.4 | 11.3 | done |
| Pronouns and demonstratives per 100 words | 6.36 | 5.81 | 6.35 | 5.37 | 7.02 | 6.71 | closer |

**Predictability, GPT-2 small on identically cleaned text**

| Measure | Buttenschoen | Dablander | Vost | Yours | AI before tonight's fixes | AI now | Status |
|---|---|---|---|---|---|---|---|
| GPT-2 surprisal, cleaned text, bits per token (lower = more predictable) | 6.79 | 6.52 | 6.62 | 7.25 | 8.01 | 7.83 | not a target |
| Surprisal SD across sentences, cleaned text (burstiness) | 0.95 | 0.95 | 0.99 | 1.20 | 1.09 | 1.09 | done |
| % tokens that are GPT-2 top-1 prediction, cleaned text (higher = more predictable) | 26.3 | 29.2 | 28.5 | 23.9 | 18.8 | 19.1 | not a target |


### Background

Size now: buttenschoen: 3,056 words · dablander: 10,520 words · vost: 8,622 words · yours: 4,240 words · AI-written: 868 words

**Sentences and words**

| Measure | Buttenschoen | Dablander | Vost | Yours | AI before tonight's fixes | AI now | Status |
|---|---|---|---|---|---|---|---|
| Words per sentence | 23.2 | 26.0 | 23.8 | 20.7 | 18.7 | 15.0 | not a target |
| % sentences over 30 words | 20.5 | 28.6 | 21.3 | 12.7 | 4.3 | 1.7 | not a target |
| Dependent clauses per sentence | 0.65 | 0.81 | 0.59 | 0.60 | 0.70 | 0.43 | not done: below the references (0.59–0.81, yours 0.6); was 0.7, moved away |
| Words per paragraph | 113.2 | 154.7 | 91.7 | 94.2 | 78.0 | 86.8 | closer |
| Lexical density % | 61.9 | 61.2 | 62.4 | 62.5 | 58.6 | 59.3 | closer |
| Nominalizations per 100 words | 4.1 | 4.0 | 4.5 | 5.5 | 3.8 | 4.3 | done |
| % passive sentences | 34.8 | 40.2 | 26.5 | 26.8 | 21.7 | 19.0 | not a target |
| we/our/I per 100 words | 0.29 | 0.85 | 0.41 | 0.50 | 0.47 | 0.69 | done |
| Hedges per 100 words | 1.1 | 0.98 | 0.95 | 1.0 | 0.35 | 0.35 | not done: below the references (0.95–1.1, yours 1) unchanged |
| Boosters per 100 words | 0.39 | 0.33 | 0.19 | 0.19 | 0.0 | 0.0 | not done: below the references (0.19–0.39, yours 0.19) unchanged |
| AI-guide phrases per 100 words | 0.03 | 0.13 | 0.20 | 0.17 | 0.0 | 0.0 | not a target |
| Numbers per 100 words | 1.2 | 1.1 | 0.46 | 0.40 | 0.70 | 0.69 | done |
| % sentences with 3+ numbers | 1.5 | 1.7 | 0.83 | 0.49 | 0.0 | 1.7 | done |

**Diversity, variability and connectives**

| Measure | Buttenschoen | Dablander | Vost | Yours | AI before tonight's fixes | AI now | Status |
|---|---|---|---|---|---|---|---|
| Lexical diversity, MTLD (higher = more varied) | 86.8 | 51.7 | 124.4 | 101.0 | 95.0 | 96.5 | done |
| Lexical diversity, MATTR-50 % | 80.2 | 78.3 | 84.3 | 82.5 | 81.6 | 81.7 | done |
| Sentence length SD (words) | 10.3 | 23.8 | 10.7 | 7.86 | 6.62 | 6.20 | not a target |
| Sentence length coefficient of variation | 0.46 | 0.95 | 0.46 | 0.39 | 0.36 | 0.43 | done |
| Mean change in length between consecutive sentences | 11.7 | 18.7 | 11.5 | 8.24 | 7.40 | 7.92 | not a target |
| Connectives per 100 words | 1.84 | 2.07 | 1.97 | 1.54 | 1.08 | 1.89 | done |
| Distinct connectives / connectives used | 0.42 | 0.18 | 0.20 | 0.36 | 0.78 | 0.69 | closer |
| % sentences opening with a formulaic connector | 2.27 | 5.43 | 1.66 | 6.34 | 0.00 | 1.72 | done |
| Three-item lists per 100 sentences | 13.6 | 3.70 | 11.6 | 9.27 | 17.4 | 3.45 | closer |
| Pronouns and demonstratives per 100 words | 3.74 | 4.87 | 5.02 | 5.43 | 4.91 | 5.21 | done |

**Predictability, GPT-2 small on identically cleaned text**

| Measure | Buttenschoen | Dablander | Vost | Yours | AI before tonight's fixes | AI now | Status |
|---|---|---|---|---|---|---|---|
| GPT-2 surprisal, cleaned text, bits per token (lower = more predictable) | 6.50 | 6.93 | 6.71 | 6.89 | 7.70 | 7.93 | not done: above the references (6.5–6.93, yours 6.89); was 7.7, moved away |
| Surprisal SD across sentences, cleaned text (burstiness) | 1.00 | 1.08 | 1.13 | 1.16 | 1.22 | 1.15 | done |
| % tokens that are GPT-2 top-1 prediction, cleaned text (higher = more predictable) | 28.6 | 27.6 | 27.3 | 27.1 | 22.0 | 21.0 | not a target |


### Methods

Size now: buttenschoen: 6,634 words · dablander: 12,957 words · vost: 7,313 words · yours: 9,629 words · AI-written: 12,233 words

**Sentences and words**

| Measure | Buttenschoen | Dablander | Vost | Yours | AI before tonight's fixes | AI now | Status |
|---|---|---|---|---|---|---|---|
| Words per sentence | 22.3 | 28.0 | 25.5 | 20.4 | 18.2 | 19.5 | not a target |
| % sentences over 30 words | 21.1 | 32.7 | 29.3 | 12.7 | 4.1 | 7.2 | not a target |
| Dependent clauses per sentence | 0.46 | 0.95 | 0.63 | 0.52 | 0.48 | 0.53 | done |
| Words per paragraph | 76.3 | 142.4 | 95.0 | 83.0 | 95.1 | 95.6 | done |
| Lexical density % | 59.7 | 59.2 | 58.8 | 56.8 | 55.1 | 54.9 | not a target |
| Nominalizations per 100 words | 3.6 | 4.0 | 5.0 | 3.8 | 3.1 | 3.1 | not done: below the references (3.6–5, yours 3.8) unchanged |
| % passive sentences | 45.6 | 43.5 | 40.8 | 28.2 | 26.4 | 27.0 | not a target |
| we/our/I per 100 words | 0.83 | 1.6 | 1.4 | 1.0 | 0.61 | 0.62 | closer |
| Hedges per 100 words | 0.36 | 0.85 | 0.36 | 0.38 | 0.37 | 0.38 | done |
| Boosters per 100 words | 0.14 | 0.26 | 0.14 | 0.09 | 0.06 | 0.06 | not a target |
| AI-guide phrases per 100 words | 0.0 | 0.05 | 0.10 | 0.01 | 0.0 | 0.0 | done |
| Numbers per 100 words | 2.1 | 1.8 | 1.8 | 2.5 | 1.9 | 1.9 | done |
| % sentences with 3+ numbers | 4.7 | 3.7 | 4.5 | 6.2 | 3.7 | 3.8 | done |

**Diversity, variability and connectives**

| Measure | Buttenschoen | Dablander | Vost | Yours | AI before tonight's fixes | AI now | Status |
|---|---|---|---|---|---|---|---|
| Lexical diversity, MTLD (higher = more varied) | 56.9 | 64.1 | 81.9 | 70.1 | 72.4 | 72.8 | done |
| Lexical diversity, MATTR-50 % | 75.2 | 77.1 | 79.8 | 78.5 | 78.2 | 78.2 | done |
| Sentence length SD (words) | 12.1 | 18.1 | 14.7 | 9.09 | 6.70 | 7.10 | not a target |
| Sentence length coefficient of variation | 0.56 | 0.67 | 0.60 | 0.46 | 0.38 | 0.38 | not a target |
| Mean change in length between consecutive sentences | 12.6 | 17.1 | 12.5 | 9.12 | 7.89 | 8.03 | not a target |
| Connectives per 100 words | 1.16 | 1.76 | 1.66 | 1.79 | 2.10 | 1.71 | done |
| Distinct connectives / connectives used | 0.28 | 0.15 | 0.27 | 0.17 | 0.10 | 0.14 | closer |
| % sentences opening with a formulaic connector | 1.01 | 3.46 | 3.48 | 0.85 | 0.00 | 0.00 | not a target |
| Three-item lists per 100 sentences | 8.39 | 7.36 | 6.97 | 12.5 | 12.4 | 13.6 | not a target |
| Pronouns and demonstratives per 100 words | 3.62 | 5.40 | 5.50 | 4.95 | 5.04 | 5.14 | done |

**Predictability, GPT-2 small on identically cleaned text**

| Measure | Buttenschoen | Dablander | Vost | Yours | AI before tonight's fixes | AI now | Status |
|---|---|---|---|---|---|---|---|
| GPT-2 surprisal, cleaned text, bits per token (lower = more predictable) | 7.23 | 7.13 | 7.11 | 7.79 | 8.21 | 8.16 | not a target |
| Surprisal SD across sentences, cleaned text (burstiness) | 1.14 | 1.13 | 1.07 | 1.11 | 1.09 | 1.04 | not done: below the references (1.07–1.14, yours 1.11); was 1.09, moved away |
| % tokens that are GPT-2 top-1 prediction, cleaned text (higher = more predictable) | 27.3 | 27.1 | 25.5 | 21.5 | 17.9 | 18.2 | not a target |


### Results

Size now: buttenschoen: 4,240 words · dablander: 4,976 words · vost: 8,230 words · yours: 8,341 words · AI-written: 11,416 words

**Sentences and words**

| Measure | Buttenschoen | Dablander | Vost | Yours | AI before tonight's fixes | AI now | Status |
|---|---|---|---|---|---|---|---|
| Words per sentence | 24.7 | 31.1 | 25.7 | 21.8 | 18.5 | 20.9 | not a target |
| % sentences over 30 words | 24.4 | 47.5 | 29.4 | 15.1 | 2.6 | 10.4 | not a target |
| Dependent clauses per sentence | 0.58 | 1.0 | 0.69 | 0.61 | 0.39 | 0.50 | closer |
| Words per paragraph | 117.8 | 177.7 | 99.2 | 104.3 | 84.3 | 88.5 | closer |
| Lexical density % | 62.9 | 60.1 | 59.3 | 55.0 | 55.3 | 55.2 | done |
| Nominalizations per 100 words | 3.9 | 3.9 | 4.4 | 3.9 | 3.5 | 3.4 | not done: below the references (3.9–4.4, yours 3.9); was 3.5, moved away |
| % passive sentences | 26.2 | 28.1 | 30.6 | 15.4 | 8.8 | 9.3 | not a target |
| we/our/I per 100 words | 0.26 | 0.76 | 1.6 | 0.73 | 0.25 | 0.26 | done |
| Hedges per 100 words | 0.87 | 2.3 | 0.89 | 0.83 | 0.53 | 0.54 | not a target |
| Boosters per 100 words | 0.33 | 0.48 | 0.24 | 0.23 | 0.08 | 0.08 | not a target |
| AI-guide phrases per 100 words | 0.09 | 0.46 | 0.58 | 0.06 | 0.0 | 0.0 | not a target |
| Numbers per 100 words | 5.9 | 1.2 | 1.9 | 2.9 | 2.7 | 2.2 | done |
| % sentences with 3+ numbers | 20.3 | 3.1 | 4.7 | 10.2 | 5.0 | 4.6 | done |

**Rhetorical moves, % of coded sentences**

| Measure | Buttenschoen | Dablander | Vost | Yours | AI before tonight's fixes | AI now | Status |
|---|---|---|---|---|---|---|---|
| method/background restated | 12.4 | 8.6 | 33.7 | 21.0 | 8.4 | 8.2 | not done: below the references (8.6–33.7, yours 21); was 8.4, moved away |
| pointer to figure/table | 18.2 | 3.6 | 3.1 | 1.2 | 2.8 | 2.7 | done |
| result as numbers | 24.8 | 7.9 | 8.3 | 17.6 | 16.3 | 13.0 | done |
| result in words | 12.4 | 9.3 | 8.3 | 17.3 | 24.1 | 24.1 | not a target |
| comparison of results | 20.7 | 23.6 | 18.1 | 11.1 | 24.1 | 28.7 | not done: above the references (18.1–23.6, yours 11.1); was 24.1, moved away |
| interpretation | 7.4 | 27.1 | 14.5 | 18.5 | 10.0 | 10.8 | done |
| comparison with literature | 0.83 | 0.71 | 0.0 | 2.8 | 1.0 | 0.73 | done |
| claim of strength | 0.83 | 3.6 | 3.6 | 2.2 | 1.8 | 1.3 | done |
| limitation | 0.83 | 5.0 | 2.1 | 3.1 | 8.6 | 7.3 | closer |
| link/transition | 1.7 | 6.4 | 6.2 | 5.2 | 1.0 | 1.6 | closer |
| recommendation/implication | 0.0 | 4.3 | 2.1 | 0.0 | 1.8 | 1.5 | done |

**Diversity, variability and connectives**

| Measure | Buttenschoen | Dablander | Vost | Yours | AI before tonight's fixes | AI now | Status |
|---|---|---|---|---|---|---|---|
| Lexical diversity, MTLD (higher = more varied) | 55.9 | 83.8 | 83.5 | 72.7 | 70.5 | 68.3 | done |
| Lexical diversity, MATTR-50 % | 76.0 | 80.4 | 80.4 | 78.5 | 77.7 | 77.4 | done |
| Sentence length SD (words) | 9.52 | 12.8 | 10.5 | 8.85 | 6.31 | 6.92 | not a target |
| Sentence length coefficient of variation | 0.42 | 0.44 | 0.42 | 0.42 | 0.36 | 0.35 | not done: below the references (0.42–0.44, yours 0.42); was 0.36, moved away |
| Mean change in length between consecutive sentences | 9.96 | 14.2 | 11.9 | 10.2 | 7.78 | 8.35 | closer |
| Connectives per 100 words | 1.59 | 3.12 | 2.30 | 2.30 | 2.21 | 2.17 | done |
| Distinct connectives / connectives used | 0.32 | 0.22 | 0.21 | 0.16 | 0.11 | 0.10 | not a target |
| % sentences opening with a formulaic connector | 5.23 | 8.75 | 6.88 | 2.09 | 0.00 | 0.00 | not a target |
| Three-item lists per 100 sentences | 4.65 | 10.0 | 10.0 | 13.6 | 11.6 | 13.0 | done |
| Pronouns and demonstratives per 100 words | 2.72 | 5.61 | 5.78 | 5.11 | 3.83 | 3.92 | done |

**Predictability, GPT-2 small on identically cleaned text**

| Measure | Buttenschoen | Dablander | Vost | Yours | AI before tonight's fixes | AI now | Status |
|---|---|---|---|---|---|---|---|
| GPT-2 surprisal, cleaned text, bits per token (lower = more predictable) | 7.49 | 7.10 | 7.11 | 7.83 | 8.75 | 8.69 | not a target |
| Surprisal SD across sentences, cleaned text (burstiness) | 1.16 | 1.12 | 0.99 | 1.18 | 1.13 | 1.05 | done |
| % tokens that are GPT-2 top-1 prediction, cleaned text (higher = more predictable) | 26.5 | 27.1 | 25.9 | 20.4 | 15.9 | 15.9 | not a target |


### Discussion

Size now: buttenschoen: 2,506 words · dablander: 1,641 words · yours: 228 words · AI-written: 1,011 words

**Sentences and words**

| Measure | Buttenschoen | Dablander | Vost | Yours | AI before tonight's fixes | AI now | Status |
|---|---|---|---|---|---|---|---|
| Words per sentence | 24.8 | 34.2 | – | 22.8 | 16.6 | 19.4 | not a target |
| % sentences over 30 words | 25.7 | 58.3 | – | 10.0 | 1.7 | 9.6 | not a target |
| Dependent clauses per sentence | 0.84 | 0.75 | – | 0.30 | 0.52 | 0.67 | done |
| Words per paragraph | 92.8 | 126.2 | – | 25.3 | 66.5 | 84.2 | done |
| Lexical density % | 60.4 | 61.1 | – | 62.1 | 60.1 | 58.4 | not done: below the references (60.4–61.1, yours 62.1); was 60.1, moved away |
| Nominalizations per 100 words | 4.1 | 4.9 | – | 6.6 | 3.8 | 3.4 | not done: below the references (4.1–4.9, yours 6.6); was 3.8, moved away |
| % passive sentences | 34.7 | 39.6 | – | 10.0 | 11.7 | 15.4 | done |
| we/our/I per 100 words | 0.96 | 1.3 | – | 0.44 | 0.30 | 0.40 | not a target |
| Hedges per 100 words | 1.5 | 1.2 | – | 0.0 | 0.70 | 0.89 | done |
| Boosters per 100 words | 0.72 | 0.18 | – | 0.0 | 0.30 | 0.40 | done |
| AI-guide phrases per 100 words | 0.24 | 0.18 | – | 0.0 | 0.0 | 0.0 | done |
| Numbers per 100 words | 1.2 | 0.24 | – | 0.0 | 0.30 | 0.30 | done |
| % sentences with 3+ numbers | 2.0 | 2.1 | – | 0.0 | 0.0 | 0.0 | done |

**Rhetorical moves, % of coded sentences**

| Measure | Buttenschoen | Dablander | Vost | Yours | AI before tonight's fixes | AI now | Status |
|---|---|---|---|---|---|---|---|
| method/background restated | 9.7 | 50.0 | – | – | 1.8 | 0.0 | not done: below the references (9.7–50.0); was 1.8, moved away |
| pointer to figure/table | 0.0 | 2.4 | – | – | 0.0 | 0.0 | done |
| result as numbers | 5.6 | 0.0 | – | – | 3.5 | 3.8 | done |
| result in words | 16.7 | 2.4 | – | – | 26.3 | 28.8 | not done: above the references (2.4–16.7); was 26.3, moved away |
| comparison of results | 9.7 | 0.0 | – | – | 15.8 | 15.4 | closer |
| interpretation | 15.3 | 9.5 | – | – | 7.0 | 11.5 | done |
| comparison with literature | 4.2 | 9.5 | – | – | 0.0 | 0.0 | not done: below the references (4.2–9.5) unchanged |
| claim of strength | 9.7 | 4.8 | – | – | 5.3 | 1.9 | not done: below the references (4.8–9.7); was 5.3, moved away |
| limitation | 9.7 | 0.0 | – | – | 14.0 | 11.5 | closer |
| link/transition | 1.4 | 2.4 | – | – | 10.5 | 7.7 | closer |
| recommendation/implication | 18.1 | 19.0 | – | – | 15.8 | 19.2 | closer |

**Diversity, variability and connectives**

| Measure | Buttenschoen | Dablander | Vost | Yours | AI before tonight's fixes | AI now | Status |
|---|---|---|---|---|---|---|---|
| Lexical diversity, MTLD (higher = more varied) | 81.1 | 93.8 | – | – | 88.0 | 99.4 | not done: above the references (81.1–93.8); was 88, moved away |
| Lexical diversity, MATTR-50 % | 79.8 | 82.2 | – | – | 80.4 | 81.3 | done |
| Sentence length SD (words) | 9.21 | 14.1 | – | – | 6.93 | 7.17 | closer |
| Sentence length coefficient of variation | 0.38 | 0.43 | – | – | 0.42 | 0.37 | not done: below the references (0.38–0.43); was 0.42, moved away |
| Mean change in length between consecutive sentences | 9.66 | 14.7 | – | – | 8.87 | 9.80 | done |
| Connectives per 100 words | 1.90 | 1.47 | – | – | 2.43 | 2.40 | closer |
| Distinct connectives / connectives used | 0.48 | 0.65 | – | – | 0.46 | 0.50 | done |
| % sentences opening with a formulaic connector | 11.9 | 8.33 | – | – | 0.00 | 1.92 | closer |
| Three-item lists per 100 sentences | 9.90 | 6.25 | – | – | 3.33 | 0.00 | not done: below the references (6.25–9.9); was 3.33, moved away |
| Pronouns and demonstratives per 100 words | 4.82 | 5.75 | – | – | 3.25 | 3.80 | closer |

**Predictability, GPT-2 small on identically cleaned text**

| Measure | Buttenschoen | Dablander | Vost | Yours | AI before tonight's fixes | AI now | Status |
|---|---|---|---|---|---|---|---|
| GPT-2 surprisal, cleaned text, bits per token (lower = more predictable) | 7.06 | 6.55 | – | – | 8.74 | 8.51 | closer |
| Surprisal SD across sentences, cleaned text (burstiness) | 1.07 | 0.86 | – | – | 1.02 | 1.07 | done |
| % tokens that are GPT-2 top-1 prediction, cleaned text (higher = more predictable) | 26.0 | 28.5 | – | – | 16.3 | 16.7 | closer |


### Conclusion

Size now: buttenschoen: 2,323 words · dablander: 2,500 words · vost: 2,479 words · yours: 1,971 words · AI-written: 1,715 words

**Sentences and words**

| Measure | Buttenschoen | Dablander | Vost | Yours | AI before tonight's fixes | AI now | Status |
|---|---|---|---|---|---|---|---|
| Words per sentence | 25.0 | 32.5 | 25.8 | 21.0 | 18.0 | 19.9 | not a target |
| % sentences over 30 words | 25.8 | 42.9 | 27.1 | 12.8 | 0.0 | 7.0 | not a target |
| Dependent clauses per sentence | 0.86 | 0.96 | 0.71 | 0.76 | 0.52 | 0.64 | closer |
| Words per paragraph | 101.0 | 138.9 | 95.3 | 123.2 | 79.0 | 90.3 | closer |
| Lexical density % | 62.2 | 61.7 | 64.3 | 55.7 | 58.9 | 57.6 | done |
| Nominalizations per 100 words | 4.2 | 5.7 | 6.9 | 4.4 | 5.1 | 5.1 | done |
| % passive sentences | 30.1 | 23.4 | 14.6 | 25.5 | 12.0 | 12.8 | not a target |
| we/our/I per 100 words | 1.9 | 1.8 | 1.3 | 0.41 | 0.96 | 0.70 | done |
| Hedges per 100 words | 0.95 | 2.2 | 1.2 | 1.4 | 0.66 | 0.93 | closer |
| Boosters per 100 words | 0.90 | 0.24 | 0.32 | 0.20 | 0.48 | 0.52 | done |
| AI-guide phrases per 100 words | 0.43 | 0.44 | 0.97 | 0.0 | 0.0 | 0.0 | done |
| Numbers per 100 words | 0.43 | 0.04 | 0.36 | 0.91 | 0.0 | 0.0 | not done: below the references (0.04–0.43, yours 0.91) unchanged |
| % sentences with 3+ numbers | 0.0 | 0.0 | 0.0 | 3.2 | 0.0 | 0.0 | done |

**Rhetorical moves, % of coded sentences**

| Measure | Buttenschoen | Dablander | Vost | Yours | AI before tonight's fixes | AI now | Status |
|---|---|---|---|---|---|---|---|
| method/background restated | 10.4 | 15.1 | 15.9 | 14.3 | 10.0 | 17.4 | not done: above the references (10.4–15.9, yours 14.3); was 10, moved away |
| pointer to figure/table | 0.0 | 1.4 | 0.0 | 0.0 | 0.0 | 0.0 | done |
| result as numbers | 0.0 | 0.0 | 3.2 | 6.0 | 1.1 | 0.0 | done |
| result in words | 13.4 | 6.8 | 14.3 | 23.8 | 22.2 | 24.4 | not a target |
| comparison of results | 3.0 | 17.8 | 12.7 | 7.1 | 16.7 | 16.3 | done |
| interpretation | 0.0 | 9.6 | 6.3 | 11.9 | 5.6 | 8.1 | done |
| comparison with literature | 17.9 | 11.0 | 0.0 | 1.2 | 0.0 | 0.0 | done |
| claim of strength | 29.9 | 15.1 | 25.4 | 0.0 | 5.6 | 4.7 | done |
| limitation | 3.0 | 5.5 | 0.0 | 15.5 | 21.1 | 12.8 | done |
| link/transition | 4.5 | 1.4 | 4.8 | 1.2 | 11.1 | 9.3 | closer |
| recommendation/implication | 17.9 | 16.4 | 17.5 | 19.0 | 6.7 | 7.0 | closer |

**Diversity, variability and connectives**

| Measure | Buttenschoen | Dablander | Vost | Yours | AI before tonight's fixes | AI now | Status |
|---|---|---|---|---|---|---|---|
| Lexical diversity, MTLD (higher = more varied) | 104.2 | 106.3 | 167.3 | 80.9 | 89.8 | 86.2 | done |
| Lexical diversity, MATTR-50 % | 83.1 | 83.5 | 87.9 | 79.4 | 80.8 | 80.4 | done |
| Sentence length SD (words) | 9.91 | 12.6 | 8.01 | 9.25 | 6.59 | 7.41 | closer |
| Sentence length coefficient of variation | 0.41 | 0.41 | 0.32 | 0.45 | 0.37 | 0.38 | done |
| Mean change in length between consecutive sentences | 9.54 | 12.3 | 8.53 | 9.40 | 8.80 | 8.72 | done |
| Connectives per 100 words | 2.46 | 2.72 | 2.12 | 2.23 | 1.84 | 2.25 | done |
| Distinct connectives / connectives used | 0.33 | 0.43 | 0.51 | 0.44 | 0.30 | 0.50 | done |
| % sentences opening with a formulaic connector | 5.38 | 13.0 | 4.17 | 2.13 | 0.00 | 1.16 | not a target |
| Three-item lists per 100 sentences | 14.0 | 14.3 | 9.38 | 12.8 | 20.7 | 14.0 | done |
| Pronouns and demonstratives per 100 words | 7.08 | 6.49 | 5.70 | 4.98 | 5.33 | 5.27 | done |

**Predictability, GPT-2 small on identically cleaned text**

| Measure | Buttenschoen | Dablander | Vost | Yours | AI before tonight's fixes | AI now | Status |
|---|---|---|---|---|---|---|---|
| GPT-2 surprisal, cleaned text, bits per token (lower = more predictable) | 6.74 | 6.58 | 6.68 | 7.46 | 8.35 | 8.21 | not a target |
| Surprisal SD across sentences, cleaned text (burstiness) | 1.02 | 0.82 | 0.82 | 1.09 | 1.05 | 1.07 | done |
| % tokens that are GPT-2 top-1 prediction, cleaned text (higher = more predictable) | 28.1 | 30.1 | 28.0 | 22.1 | 17.7 | 18.1 | not a target |

