# Chapter 2 cuts, items 1, 2, 4, 10 (2026-10-10)

Decisions from CUTS_DECISIONS.md. All four paragraphs touched were AI-marked ("% TODO: manual review" directly above). No number, claim or table was changed.

## Item 1: "These sources can interact..." (Cut)
- Cut: the whole paragraph after Table tab:noise_sources (Section "Noise in molecular data"), with its "% TODO: manual review" line.
- Labels removed: none. It referenced Section sec:noise_injection, which still exists.
- Citations it carried: Heid2023, Bentz2013, Kruger2012. All three are still cited elsewhere in Chapter 2.
- Replacement sentence: none.

## Item 2: paragraph defining detector, flag, three groups (keep only the Song sentence)
- Section sec:lnl_background. The paragraph now reads, unchanged from the original sentence:
  "\citet{Song2022} sort deep learning methods for noisy labels into five groups: robust architectures, robust regularization, robust loss functions, loss adjustment and sample selection."
- Cut: the LNL definition, the two halves, detector, suspicion score, known flag, noise detection, noise mitigation, noise-robust training with Han2018 and shu2019metaweightnet, and the three sentences after the Song sentence (all five act during training; the sample-selection exception).
- Labels removed: none. Table tab:noise_terms and Figure fig:data_cleaning stay where they were.
- References repointed: none needed.
- Left open, not fixed (outside the decision):
  - Figure fig:data_cleaning is now referenced nowhere in thesis.tex. The cut paragraph held its only \ref.
  - Table tab:noise_terms is no longer referenced from Chapter 2 itself. Chapters 4 and 7 still reference it.
  - The acronym LNL is no longer defined. Chapter 4 still uses it once ("Most methods in the LNL literature", the paragraph starting "The three groups of methods for noisy labels").
  - Han2018 and shu2019metaweightnet are still cited in Chapters 4, 5 and 7.

## Item 4: third telling of Cortes et al. (Cut)
- Tellings: Chapter 1 Introduction (first), Chapter 2 Support vector machines (second), Chapter 2 "No single method is expected to win" paragraph (third).
- Cut: the sentence "\citet{Cortes2015} compared 12 algorithms on 12 bioactivity datasets with increasing Gaussian label noise: gradient boosting was the least tolerant and the other 11 degraded comparably, with two polynomial-kernel models sensitive only at high noise." Also cut its own "% TODO(cite): '10 noise levels' could not be verified ..." comment line, which was about this sentence only.
- The rest of the paragraph is unchanged. Cortes2015 is still cited in Chapters 1, 2 and 3.
- Replacement sentence: none.

## Item 10: delivered-noise tolerance check (to Additional files)
- Moved unchanged: from "The delivered noise is checked against a band of three standard errors," through Equation eq:dose_tolerance and its explanation, to "...the delivered amount is recorded either way." (Section sec:noiseinject_pkg, Regression noise, the paragraph starting "Each injection also returns a record".)
- Kept in the thesis: the first sentence of that paragraph (the injection record).
- New Additional files section, placed first, before the Chapter 3 section:
  \section{Molecular representations, models and label noise}\label{app:add:background}
  \addfile{Tolerance band for the delivered noise}\label{app:add:dose_tolerance}
- Label moved: eq:dose_tolerance (now in additional_files_body.tex). No \ref to it existed in thesis.tex or the Additional files body, so nothing to repoint.
- Replacement sentence in thesis.tex, with "% TODO: manual review" above it:
  "The delivered noise is checked against a tolerance band, and a draw outside it raises a warning (Additional file~\ref{app:add:dose_tolerance})."
- Side effect: the new section comes before the Chapter 3 section, so every existing Additional file number goes up by one. Nothing cites them by typed number (checked by grep); only two comments in additional_files_body.tex say "Additional file 1" and "Additional files 1--9" about the paper's numbering.
- Not done: no rebuild of additional_files.tex or the thesis, so the new \ref resolves only after both are rebuilt (thesis reads build/additional_files.aux via xr-hyper).
