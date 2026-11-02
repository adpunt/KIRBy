# Every request in the author's messages (2026-10-09/10), to check before final results

## Message 1 (the long brief)
1. Review the 3 reference theses separately; notes per thesis on every item below.
2. Cross-reference theses + notes against her thesis; find and FIX what needs fixing.
3. Transition text between sections and chapters.
4. Phrasing: common phrases, sentence structures, sentence length, words/jargon.
5. Numbers in results text vs pointing to tables/figures and describing patterns.
6. Chapter/section title capitalisation; paragraph between chapter title and intro; between section title and first subsection.
7. Tables/figures in results relative to text.
8. Limitations: which, where, how specific.
9. Main takeaways phrasing; how often main questions are re-asked; how flow is enforced.
10. Detail in methods / results / intro-background literature.
11. Definitiveness of language.
12. Statistical significance detail; match or exceed the references (within her no-test rule).
13. Chemistry depth; coding implementation depth; she must talk about chemistry correctly.
14. Diagrams: used? what they show.
15. Basic sections: anything big missing; abbreviations placement; random stuff at the end; acknowledgements/abstract placement and format.
16. Acknowledgements: copy the references' verbatim, compare against hers; help (she is "not very touchy feely").
17. Tense: which tense when (past literature, own experiments, conclusions).
18. Mixed/negative vs positive results: how handled, how much emphasis.
19. Originality / contributions: how emphasised.
20. Title pages.
21. Document ALL their captions; align hers (content amount, abbreviations, grammar).
22. Additional files: how referred to; how figures/tables are referenced.
23. Section titles: she hates hers and the Ch5-8 ones; draw inspiration from the references.
24. AI-use declaration must be included.
25. Do all 3 reviews separately, piece together what a thesis should look like, APPLY to her thesis. Don't make her decide much.
26. Extra pass on chemistry language: how they talk about chemical terms; help where she lacks domain knowledge.
27. Introduction/abstract/conclusion aligned with main takeaways; Ch1 and Ch8 disjointed; come together.
28. Takeaways framing (Ch1-3 noise not Gaussian, no one-size-fits-all, RF solid, Ch3 conditional conclusions; Ch4-7 cone; noise handled conditionally - important; originality of noise work; hybrids work for noise and prediction; forward compatible).
29. Written by Claude => follow AI writing guidelines, mark with TODO manual review. Simplicity and clarity.
30. Plain language to her (CLAUDE.md). Precise, no corner cutting, no over-reliance on grep, no hallucination. Use the reference papers. Save text contents somewhere.
31. Work alone; if in doubt, return to the prompt.
32. Her-text rule: AI paragraphs (TODO manual review) modify freely; her text only when wrong, with a TODO stating the change; trivial spelling/grammar no comment.

## Message 2
33. Work from the LaTeX package she was converting to (now main.tex + preamble.tex + thesis.tex + additional_files.tex at repo root).

## Message 3
34. Hold off touching the thesis (later lifted).

## Message 4
35. Abstract was haphazard: rely heavily on reference abstracts, especially phrasing.
36. Decimal places: uniform rule.
37. Separate results/discussion sections or not: incorporate the standard.
38. Ch8 titles "The Central Claim", "What Is New in This Thesis", "The Antibodies" should not exist (content decent).
39. Ch8 is a mess: really focus on fixing it + abstract, relying heavily on the references.
40. Length: abstract/introduction/final chapter repeat takeaways but are too verbose, esp. Ch8; rely on references.
41. Safe to continue. Figures still being modified; text mostly finished; captions may get jumbled.
42. Ch4 count is fine to use.
43. Originality: another chat is running the literature check; results to come.
44. RF rank (point 3): check it, no input needed.
45. Equal effort applying as reviewing; she found heavy AI slop and ignored items before.

## Later messages
46. Follow everything said carefully; her-text rule ("Don't fuck that up").
47. Results are also being fixed in another chat; avoid overlap.
48. The AI marker is "TODO: manual review" or that kind, not any TODO.
49. Comment bookkeeping must not block necessary changes; about half the text is AI-generated.
50. Check every message is addressed before giving final results.
51. This pass should be more cuts than adds (net word count should fall, especially in AI-marked text).
