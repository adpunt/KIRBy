# Handoff: cut the server ledger down to what goes in the thesis

Written 2026-10-09. Paste everything below the line into a fresh Claude Code chat in `~/repos/KIRBy`.

---

> Read `~/repos/KIRBy/CLAUDE.md` first and follow it in every reply. The author is the engineer
> and decides. Your first deliverable is a table of proposed ledger actions for her to sign off.
> It is not a command list. Bring at most three decisions at once.
>
> ## The goal
>
> All compute from now on goes to finishing the experiments the thesis uses. Anything the thesis
> does not use stops. Two rules come from the author:
> - **A dataset in the thesis keeps running even where its results are partial.** QM9 is the
>   example. Its complexity results are not finished and it is in the Chapter 5 list, so they
>   keep running. Its noise results are partial too, and they also keep running.
> - **Noise is decided per dataset, with her.** A dataset approved for Chapters 5 and 6 does not
>   automatically keep its noise experiments.
>
> ## Read first
>
> 1. In `THESIS_OUTLINE.md`, under "WHICH DATASETS ARE SHOWN", the block "CHAPTERS 5 AND 6
>    DATASETS, SIGNED OFF". It is the dataset list.
> 2. The experiments the thesis uses. Chapter 5 is outline sections 5.3.4 to 5.8, as updated on
>    2026-10-09. Chapter 6 is sections 6.2 to 6.6, with the "author agreed 2026-10-09" notes.
>    Chapter 7 is not yet decided.
> 3. The ledger system:
>    - `tests/slurm_scripts/runbook/WORKERS.md`;
>    - `tests/experiments/orchestrate/ledger.py`. Its commands include `cut`, `hold`,
>      `skip-stage`, `skip-experiment`, `skip-condition`, `prioritise`, `set-priority` and
>      `promote`. Read the argparse block at about lines 1085-1130 and each method before
>      proposing it.
>    - `tests/experiments/orchestrate/plan.py`, with `CUT_DATASETS` at about line 67.
> 4. The current ledger state. The newest rsync copy is
>    `/Volumes/seagate/kirby_results_<newest date>/LEDGER.md` and `ledger.json`. The author can
>    paste `ledger.py status` from the server. Never ssh.
>
> ## Known from the conversation that set this up
>
> These are rows to cut or stop, each with its reason:
> - **adme_hlm:** out of the thesis. It has results in 5 of the 17 Chapter 5 experiments.
> - **abcov_kd:** out. Its labels come from different papers, the antigen is not recorded, and
>   identical sequences differ by up to 1.26 log units. Its noise stage was still running on
>   9 Oct.
> - **cdr_flexibility:** out. Its split leaks. Its noise stage was still running.
> - **abagym_VEGF:** out. 285 of its 418 mutants are in flab_binding_koenig.
> - **tcga_brca_pam50:** patient data. It must never run again, and must never be read (rule in
>   `CLAUDE.md`). The ledger still lists it, with one failed unit (c13). Do not reset or redo
>   it. Cut it.
> - **c05 (learning curves):** cut from the thesis on 2026-10-08, but humanness_therapeutic and
>   abagym_VEGF were still running it on 9 Oct. Propose `skip-experiment c05`.
> - **Experiments the outline no longer uses.** Check each against the outline. Candidates:
>   - c12, which joins whole data types and has no result without patient data;
>   - fed05, which cannot run since lhasa_herg was cut;
>   - c17's re-run, which was decided against (outline 5.5, `RESULT_CAVEATS.md`).
>
> These are rows to keep running:
> - Chapters 5 and 6 datasets: everything in the signed-off block. flab_binding_koenig is last
>   in the queue. The author may promote it later, which is her call.
> - flab_expression_koenig: its similarity stage is 4 of 11. It stays in Chapter 5 now and
>   reaches Chapter 6 only if that stage finishes. Ask whether to prioritise it.
> - adme_mdr1 and adme_rlm: Chapter 5 only, so their similarity stage is a candidate to skip.
>   Ask.
> - **herg_ic50: KEEP RUNNING, never cut, skip or hold (author, 2026-10-09).** It is a noise-only
>   row (n17, n19, n22, n26, n29, n36), for Chapter 7. It was 4 of 6 on 9 Oct, running n22 and
>   n36. Leave it out of every proposed cut.
> - ginkgo_pr_cho also has its own one-off censoring experiment (`HANDOFF_CENSORING_PRCHO.md`).
>   That is an srun outside the ledger. Do not schedule it here.
>
> OPIG and cross rows, DECIDED with the author on 2026-10-09:
> - **Keep, reported in the thesis.**
>   - `opig:qsar`, `opig:shared_vectors_remedy` and `opig:blueprint_transfer` are done.
>   - `opig:shared_vectors_census` keeps running.
> - **Stop.** They are running now, so stop them through the ledger and do not cancel jobs.
>   - `opig:ginkgo`: c02 already covers the Ginkgo sets on the competition's folds.
>   - `opig:graphinity`: ddg_combined covers this task.
>   - `opig:cgr_comparison`: reactions are text-only in the thesis.
>   - `opig:trastuzumab`: its datasets were cut on 2026-10-08.
> - **Skip, they are pending.**
>   - `opig:humanness`: humanness_therapeutic covers it.
>   - `opig:scaffold_hybrid` and `opig:scaffold_similarity`: both wait on flab_binding_koenig.
>   - `cross:c14`: it repeats c02.
>   - `cross:s09`: the figure script makes the same summary.
> - **`cross:c03`: KEEP (author, 2026-10-09).** Its wait list is every dataset in the run
>   (`plan.py:374`, `'cross:c03': ('complexity', list(roster))`), so it would never start.
>   Re-point it to the signed-off Chapter 5 datasets WITHOUT flab_binding_koenig, which is last in
>   the queue. Make the change in plan.py, commit and push it, then rebuild or update the ledger row
>   so the server sees the new wait list. Show the author the row's stages before and after.
>
> For each of these, check which ledger command fits the row kind (`opig:`/`cross:`) before
> proposing it, and dry-run it first.
>
> ## Noise experiments, by what Chapter 7 uses (DECIDED with the author 2026-10-09)
>
> The Chapter 7 method list is signed off in `THESIS_OUTLINE.md`, block "CHAPTER 7 METHODS, SIGNED
> OFF", with each method tagged by its experiment number. The rule from the author: an experiment
> with no result going into the thesis is cut from the ledger.
> - **Cut (`skip-experiment`):** `n11`. Its only result, the best repair, is cut (the best of a
>   whole grid picked on the held-out score, on 2 datasets). CHECK FIRST: `plan.py:113` has
>   `AFTER = {..., 'n06': ['n11', 'n13']}`, so n06 waits on n11. Find which rows run n06 and say
>   whether skipping n11 leaves n06 waiting forever. n06 is not in the outline; ask the author
>   whether it goes too.
> - **Ask before cutting: `n27`.** Its method (moving every label toward an out-of-fold prediction)
>   is cut. But its other half, the removal-cost runs (random deletion, 5 folds on five
>   datasets), is cited in the outline's Chapter 7 standing caveats and beside QM9's random-deletion
>   control. Show the author how many n27 units are left and where.
> - **Keep running.** Each has at least one deep-dive or broad result: n02, n07, n09, n12, n13,
>   n18, n19, n20, n21, n22, n23, n24, n25, n29, n30, n31, n32, n33, n34, n35, n38, n39, n40, n41.
>   Some methods inside these are cut (for example itlm in n32, c_mixup in n33). The ledger cannot
>   cut a method inside an experiment, so leave them; the unused methods are simply not reported.
>   n09 stays whole: all its methods go in one broad table (author, 2026-10-09).
> - **Other cuts in the outline to check against the ledger:** fed05 cannot run since lhasa_herg
>   was cut; c05 and c12 are out of the thesis (see "Known from the conversation" above).
> - **herg_ic50's row** (n17, n19, n22, n26, n29, n36) keeps running whatever is decided here.
> - **Not part of the method decision; ask the author:** n01, n06, n14, n15, n16, n37, n42. These
>   are not methods, so the method list does not settle them. The outline (Chapter 7 limitations)
>   lists n01, n14, n16, n37 and n42 as results the chapter does not use; n15 bears on 7.9
>   (cleaning inside the feature search); n26 and n37 measure the size of the noise itself. Give
>   units left and hours for each, and let her decide one by one.
> - **QM9 is the clean-label control.** Two deep dives, n40 and n20, have no QM9 row yet. Check
>   whether QM9 is still queued for them and ask whether to prioritise those units.
> - `n30b` (real censoring on ginkgo_pr_cho) is an srun, outside the ledger. Do not schedule it.

> ## The per-dataset noise decision
>
> Do this with the author, dataset by dataset. For each dataset in the signed-off list, show:
> - its noise stage progress (for example "51 of 121 units, running c10 and n07 gaussian");
> - hours spent so far, and an estimate of the hours left, with how you estimated it;
> - what its noise results already show, read from the generated tables. The yes/no counts per
>   method are in `scratchpad/thesis_ch5_8/THESIS_SCOPE_LISTS.md` and `MEETING_METHODS.md`.
>
> Then she decides one of three for each dataset:
> - continue all noise;
> - continue only some experiments;
> - skip the noise stage.
>
> Chapter 7's dataset list follows from these decisions. Record it in the outline, in the same
> block style as Chapters 5 and 6.
>
> ## Rules for anything run on the server
>
> - **She keeps queued jobs for their queue position.** Never propose cancelling queued jobs or
>   bulk `scancel`. Use the ledger (`cut`, `hold`, `skip-*`), so that workers stop taking the
>   units.
> - **Never `rm`** in a block she will paste.
> - **Commands are given as she runs them.** One `srun ... --pty /bin/bash` line where an srun is
>   needed, then plain commands inside it. Never sbatch, arrays, or shell variables holding
>   commands.
> - **Every ledger change gets a `--reason`** with the date and the reason, in plain words.
> - **Dry run first.** Where a command has `--dry-run`, give that first and show her its output
>   before the real one.
> - **`plan.CUT_DATASETS`:** add the cut rows there too, with a dated comment saying why, like
>   the existing ones. Commit it, and push so the server pulls it.
>
> ## When it is done
>
> Write one dated block in the outline, next to the signed-off dataset list. It records:
> - what was cut;
> - what was skipped;
> - what keeps running;
> - the per-dataset noise decisions.
