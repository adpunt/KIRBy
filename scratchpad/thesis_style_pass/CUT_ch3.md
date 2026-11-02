# Chapter 3 (chap:robustness) cutting pass, 2026-10-10

Prose words in the chapter, lines starting with % removed (python3 -I):

- Backup (thesis.tex.before_style_apply_2026-10-10): 13,048
- Before this cut: 13,368
- After this cut: 13,128

Target was 12,395 (5% under the backup). Shortfall: 733 words.
Reason: only nine paragraphs in this chapter are AI-marked (plain "% TODO: manual review"),
about 625 words in total before this cut. No AI-marked captions. Deleting every one of
them would still not reach the target. The rest of the chapter is the author's text.

## What was cut, one line per paragraph

- Introduction, opening paragraph: dropped "Chapter 2 introduced the representations, models and noise conditions"; one sentence left, pointing to Chapter 2.
- Models, base and variant model list: removed the repeated "a random forest with 300 trees" and "the other four are"; same six variant models named.
- QM9 pipeline, all-zero fingerprint check (own paragraph): merged into the run-checks paragraph.
- QM9 pipeline, memory-mapped files: shortened; same content (on disk, one file per split, Rust writes processed copy).
- QM9 pipeline, record layout: cut the byte size of each representation's block and the byte-offset sentence; kept the list of fields per record, merged into the memory-mapped paragraph. TODO added: the figure's table gives ECFP4 as 32 bytes and Sort & Slice as 16 bytes, the cut text gave 256 and 2,048 bytes; the figure looks out of date.
- QM9 pipeline, how noise is drawn per split: tightened; every rule kept (training sets the parameters, validation independent draw with same amount, test never corrupted, scale recorded on every split).
- QM9 pipeline, per-molecule record and run checks: tightened; fingerprint check merged in.
- QM9 pipeline, "the pipeline then trains the models ... hyperparameters in Additional file": cut entirely; the Methods opening paragraph already points to the same Additional file.

## Terminology

- "scaffold group(s)": none in AI-marked text in this chapter.
- "binding free energy": does not appear in this chapter.
