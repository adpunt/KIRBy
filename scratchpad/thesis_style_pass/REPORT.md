# Style pass, 9–10 October 2026: what was done against each request

Backups: `thesis.tex.before_style_apply_2026-10-10` (before any edit tonight), `thesis.tex.before_benchmark_edits_2026-10-10` (after the restore of your text).

## Words per chapter (comment lines removed)

| Chapter | Before tonight | Now |
|---|---|---|
| 1 | 1,978 | 2,018 |
| 2 | 8,261 | 7,757 |
| 3 | 13,048 | 13,017 |
| 4 | 7,599 | 7,567 |
| 5 | 23,207 | 21,671 |
| 6 | 6,504 | 5,811 |
| 7 | 13,190 | 12,376 |
| 8 (with the Summary of Research Contributions) | 4,558 | 1,555 |
| Total | 78,345 | 71,772 |

Abstract: 296 words (limit 300). Some of the Chapter 7 change may come from the figures and results sessions.

## Your requests, item by item (numbers match HER_MESSAGES_CHECKLIST.md)

1–2. Reviews of the three theses: done. Notes in `~/Documents/ReferencePapers/Theses/notes/` (nine files), texts in `../text/`.
3. Transitions: done. Chapters 3–7 open by naming the chapter before; Conclusions point to the next chapter.
4. Phrasing: AI-written paragraphs rewritten; your text left alone. Measured against the references in `benchmark/BENCHMARK_REPORT.md`.
5. Numbers in results text: results text is inside the reference range (2.8 per 100 words; references 1.2–5.9). Chapter 8 prose now has almost none. Decimal rule applied to 226 numbers.
6. Titles: sentence case throughout; a paragraph under each chapter title.
7. Tables/figures against text: no new tables were built; results paragraphs in AI text now open with a pointer where a table or figure exists.
8. Limitations: inside each chapter's Conclusions, cut to one paragraph in Chapters 4–8. Still 18% of conclusion sentences (references 0–6%); much of the rest is in your own text (Chapter 3).
9. Takeaways and flow: Chapter 8's Summary of findings answers the Chapter 1 questions in order.
10. Detail by part: definitions of hERG, Caco-2, LogD and Bemis–Murcko added in AI text. Chapter 5 not moved to Additional files (decision).
11. Definitiveness: Chapter 8 states the findings plainly; fold counts kept on headline claims.
12. Statistical significance: "significantly" without a test removed from AI text; no tests added. Chapter 3's tests kept.
13. Chemistry: wrong chemistry fixed (ECFP chirality default, 2D graphs, VBLL, ΔΔG naming, activity cliffs as noise); typography (LogD, $K_i$, Bemis–Murcko) made consistent, with no comments.
14. Diagrams: none edited (figures session).
15. Front and end matter: one acknowledgments chapter, abstract after the Contents, AI statement after the abstract, abbreviations rebuilt, software listed, `Sun2022` added to refs.bib.
16. Acknowledgments: yours unchanged; five added paragraphs with [PLACEHOLDERS] for names. This is still thin help.
17. Tense: fixed only in AI text; in your text only where a sentence mixed tenses.
18. Mixed/negative results: stated plainly in Chapter 8.
19. Originality: Chapter 8 Contributions; literature check applied (the gap over neighbor spread is the meteorological buddy check, now cited; three methods remain new).
20. Title page: college and crest in; the term says Trinity, which is probably wrong for October.
21. Captions: every reference caption copied; caption additions to your text were reverted.
22. Referencing: quoted section titles replaced by `\ref`; stale chapter pointers fixed.
23. Section titles: renamed throughout; old titles kept as `% title was:` comments.
24. AI-use declaration: present, shortened as your TODO asked.
25–26. Applied with a chemistry pass: see 13.
27–28. Abstract, Chapter 1 and Chapter 8 tell one story in one order.
29. AI text carries "% TODO: manual review".
32/46/48/49. Your-text rule: after the restore, the check found no change to unmarked text in the last pass. Remaining changes to your text: spelling and typography (no comment), 28 real errors (one "fixed:" line each), and the paragraphs under your own rewrite TODOs.
35. Abstract: rewritten on Buttenschoen's pattern, 296 words.
36. Decimals: applied; 22 numbers kept at four decimals (tie rule, or no raw value to settle the third decimal), so a few sentences mix three and four decimals.
37. Results/Discussion: no separate Discussion sections; interpretation in Results.
38. Chapter 8 titles: "The Central Claim", "What Is New in This Thesis", "The Antibodies" are gone.
39–40. Chapter 8: about 1,420 words without the Summary of Research Contributions (references 896–1,593).
42. Chapter 4 count: allowed; Chapter 8 uses the ECFP4 feature count instead.
43. Originality: see 19.
44. Random forest rank: checked against the result file; two Chapter 3 results sentences corrected.
47. Results session: decimals were applied after you said the figures were final; `ch7_numbers.py` updated and still passes (565 OK).
51. More cuts than adds: 6,573 fewer words in total.
Also: four examiner citations added (Lambert 2023, SIMPD, PREFER, UNIQUE); `\paragraph` removed everywhere.

## Benchmark after the edits (`benchmark/BENCHMARK_AFTER.md`)

Most measures moved little. Conclusions: limitation sentences 28% → 18%, numbers per 100 words 0.77 → 0.50. Results: pointers to tables and figures still below the references; result-in-words sentences slightly up. Discussion text: linking sentences still above the references. The remaining gaps sit mostly in your own paragraphs; suggestions for those are in `SUGGESTIONS_ch1.md` … `SUGGESTIONS_ch8.md`.

## Needs you

1. Chapter 1 says repeat p$K_i$ measurements have a standard deviation of about 0.54 log units; your Chapter 3 sentence says they "disagree by about 0.54". Kramer 2012's 0.54 is a standard deviation.
2. Chapter 5, "Which representation is best depends on the downstream model": its counts disagree with Table tab:datasets (six of seven against three; three of twelve against five). Not checked which is right.
3. Chapter 6: the two tables from `figures_ch5_7/` have 14 rows including MDR1–MDCK; the text says 13 datasets.
4. Title page term: Trinity or Michaelmas 2026.
5. AI statement: "All written text in this thesis is my own" holds only once the AI-marked paragraphs are rewritten by you.
6. DOIs missing for PREFER and UNIQUE (journal version) in refs.bib.
7. Acknowledgments placeholders: names for Lhasa colleagues, OPIG members, CDT staff and cohort, transfer examiners, college.
