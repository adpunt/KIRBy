# Similarity suite — what has been run, debugged and optimised

Opened 2026-09-20. A live tracker, not a result file. Every number in it is a
local run on three whole datasets: caco2 2,158 molecules, herg_ki 1,412,
adme_mdr1 2,641. No rows were ever subsampled; what was cut is the searched
grid — 7 representations that RDKit builds locally in place of the 10 in
DATASET_REPS, budgets 25/100/250, one downstream model, one importance method,
two allocation methods.

⛔ Nothing here is evidence about which allocation method, importance method or
budget wins. Those axes were cut for compute.

Two categories, and they need different work:

- **investigation** — the question is whether the numbers coming out answer the
  question going in. Done when it runs and its output is readable.
- **method** — something is meant to work better than the alternative. Done when
  it has been optimised against that alternative, or when it has been measured
  well enough to say it cannot be.

---

## Where every experiment stands on disk, 2026-09-23

Checked by opening every result file under `tests/results_local_sim`,
`tests/results_local_sim_tuned` and the seagate output directories, and reading
`result_status` and `has_computed_results` out of each. Ten experiments carry a
computed result for all three datasets:

    s01  s02  s03  s04  s05  s06  s08  s10  s12  s13

s09 is a cross-dataset summary and writes one file, not three. Its only output
on disk was stamped STALE and dated 5 September, so it was re-run on
2026-09-23 over the three datasets and now writes
`/Volumes/seagate/kirby_s09_out/s09_cross_dataset_summary/results.json`, twelve
series, every one per dataset and never averaged.

s11 is the only experiment with no completed result. See the row below.

**s05's tuning barely moves the number it was tuned for.** s09 reads the
shipped-arm s05 files; the tuned run is a separate directory. Hybrid minus
reference at the fixed cell, positive means the hybrid wins:

    kNN_delta_fixed_ref       caco2     herg_ki   adme_mdr1
    shipped settings         +0.0075     -0.0003     +0.0273
    tuned settings           +0.0028     -0.0047     +0.0268

Distance-weighted voting, importance exponent 0.25 and neighbour counts 5 and 10
change the fixed-cell delta by 0.0047, 0.0044 and 0.0005. The sign is unchanged
on all three. s10's kNN task found distance-weighted voting lowered held-out
error in 20 of 21 representation-dataset pairs; this says that gain does not
survive into the hybrid-minus-reference difference, because it lifts both arms.

## s12 on seven representations: the hybrid's margin roughly doubles

All three datasets, same 12-configuration search on both sides, so the pool is
the only difference. Cliff delta, hybrid minus comparator, negative means the
hybrid wins:

    dataset      4 reps      7 reps     change
    caco2       -0.0044     -0.0154     2.5x
    herg_ki     -0.0065     -0.0179     2.7x
    adme_mdr1   -0.0154     -0.0286     1.9x

The hybrid wins on all three in both pools. Giving the search `avalon`,
`ecfp4_count` and `top_torsion` roughly doubles to triples its margin on every
dataset — same sign, consistent magnitude, no exceptions.

Cleaner than the same comparison in s04, where the effect was mixed: there it
halved a deficit on herg_ki and erased a small win on caco2. In s12 the
four-representation runs were costing about half the measured effect everywhere.

## The four-representation restriction moved the hybrid, not the comparator

s04 re-run on all seven, 2026-09-24. Both runs used the same 12-configuration
search, so the pool is the only thing that differs. Cliff RMSE, lower better:

    dataset     pool    hybrid   best single        delta
    caco2       4 rep   0.5383   pdv 0.5429      -0.0046
    caco2       7 rep   0.5428   pdv 0.5429      -0.0000
    herg_ki     4 rep   0.9329   pdv 0.8863      +0.0466
    herg_ki     7 rep   0.9069   pdv 0.8863      +0.0206
    adme_mdr1   4 rep   0.5519   pdv 0.5806      -0.0287
    adme_mdr1   7 rep   0.5497   pdv 0.5806      -0.0308

`pdv` is the best single representation in both pools, at the same score to four
decimals, on all three datasets. `avalon`, `ecfp4_count` and `top_torsion` never
beat it, so the restriction cost nothing on the comparator side — the opposite
of what was expected when this was raised.

What moved is the hybrid, because the hybrid is built from the pool: three more
representations change which features the search picks. On herg_ki that halves
the hybrid's deficit, +0.0466 to +0.0206. On caco2 it erases a small win. On
adme_mdr1 it slightly extends one.

So the four-representation runs did not understate the comparator. They
understated the hybrid.

## s04 and s12 were run on four representations; everything else used seven

Counted from `input_fingerprint.reps` in every stored result file, 2026-09-24:

    experiment                    caco2  herg_ki  adme_mdr1
    s01 s02 s05 s06 s13             7       7         7
    s04 s12                         4       4         4

The four are `atom_pair`, `ecfp4`, `maccs`, `pdv`. Absent: **avalon**,
**ecfp4_count**, **top_torsion**. Nothing in either file restricts the pool —
the runs were launched with a four-representation pool. s02 ran on 2026-09-18
with seven; s04 and s12 ran on 2026-09-20 with four.

Why it matters rather than being bookkeeping: both files pick a best single
representation as the comparator the hybrid is measured against, from whatever
pool they were given. s06's sweep, on the full seven, found
`ecfp4_count_weighted` the single best arm on caco2 at +0.2162 and
`avalon_weighted` second at +0.1954 — both absent from s04 and s12. So their
comparator was chosen from a pool missing three representations that win
elsewhere, and the hybrid's margin over it cannot be read as a margin over the
best available single representation.

Six re-runs on the seven-representation pool are queued, writing to
`/Volumes/seagate/kirby_s04s12_pool7`. Nothing already stored is overwritten.

## A lever can be present, documented and dead

`KIRBY_S05_K_VALUES` did nothing. s05 resolves `kwargs.get('k_values', <env
fallback>)`, and the driver set `kwargs['k_values'] = config.get('k_values',
[5, 10])` on every call, so `kwargs` always carried a value and the fallback was
never reached. A sweep asking for 5, 7, 10, 15 and 25 ran at 5 and 10 and said
nothing about it.

Caught 2026-09-24 with 24 s05 cells already in flight, by reading the running
logs and seeing only `k=5` and `k=10` printed. The four cells that were 50
minutes in were restarted so all 24 share one neighbour list. The driver now
passes `k_values` only when the config names them or the env var is unset;
unset still gives [5, 10], so every stored run reproduces.

The rest of the suite was swept for the same shape. `k_values` was the only
shadowed lever: `n_sample`, `weight_methods` and `budgets` have no environment
lever to shadow, s05 never reads `conditions` as a keyword so the `None` the
driver passes is harmless, and s06's, s10's, s11's and s13's levers are read
directly rather than through the driver.

⛔ The lesson for this table: "the lever exists" and "the sweep used it" are
different claims. Read the neighbour counts, exponents and coefficients out of
the result file before recording an axis as swept.

## What "optimisation done" meant, audited 2026-09-23

Read out of the finished result files, not from this table. Only s06 was swept.

    experiment   swept                                left at one value
    s06          coefficient, exponent, statistic,    --
                 neighbour count -- 2,628 cells
    s10          neighbour count, q-RASAR             importance exponent 0.5,
                 neighbours, GP length-scale start    kNN vote uniform,
                                                      normalize_y, weight convention
    s05          neighbour count, two values          exponent, vote -- one point
                                                      per dataset
    s13          cluster count                        retrieval neighbours 5,
                                                      exponent, coefficient

⛔ s10 was scored at importance exponent 0.5 and uniform voting. Its own kNN
task, in the same file and the same run, measured distance-weighted voting
better in 20 of 21 (representation, dataset) pairs and the hardcoded 0.5 never
best. So the GP and q-RASAR tasks were scored at two settings the file had
already measured as losing.

Measured wall clock on this machine, which is what makes a full sweep a cluster
job rather than a laptop one: s05 43 to 50 minutes a run, s13 2.4 hours,
s10 4.8 to 8.3 hours.

Queued 2026-09-23 as a result:

- **s05, the full grid.** Two voting schemes by four importance exponents
  (0, 0.25, 0.5, 1.0) on three datasets, 24 runs, with the neighbour count swept
  inside each run over 5, 7, 10, 15 and 25. This is the sweep the table claimed
  and the files did not have.
- **s10, one cell, not a sweep.** Distance voting at exponent 0.25 on all three
  datasets, so the GP and q-RASAR nulls are re-read at the settings s10's own
  kNN task chose. Three runs. A full 8-cell grid at 6 hours a run is roughly
  two days of this laptop and belongs on the cluster.

## Where this stands, 2026-09-24 01:30

Audited from result files, not from the rows below. Four experiments the table
called optimised were not:

    s04   ran on 4 representations, not 7; one cliff setting, not a sweep
    s05   one point per dataset; KIRBY_S05_K_VALUES was dead
    s10   scored at exponent 0.5 and uniform voting, which its own kNN task
          measured as losing
    s12   ran on 4 representations, not 7

Only **s06** survives the audit: four levers, 2,628 cells, all three datasets.

Three defects found and fixed tonight, each of which had been silently spoiling
results rather than failing loudly:

- **s11 never finished** because `hybrid_bnn` refits a PyTorch network once per
  BO iteration and torch opens a full-core thread pool per fit. Nine attempts
  over two days, killed by the kernel, empty logs every time.
  `KIRBY_TORCH_THREADS` caps it; both `hybrid_bnn` and `hybrid_gauche_gp`
  completed for the first time in this campaign.
- **s11 lost every kill** because nothing was written until the last condition.
  Each condition now checkpoints through a temporary file and a rename.
- **`KIRBY_S05_K_VALUES` was dead.** `SIMILARITY_EXTRA` carries
  `'k_values': [5, 10]` into every similarity run's config, so the env var was
  always overridden. A sweep naming five neighbour counts ran at two and said
  nothing. Fixed at that source and CONFIRMED live: a running cell scored
  k=5, 7, 10, 15 and 25 after twelve arms. An earlier fix that guarded the
  per-call assignment looked right and changed nothing, because the key was
  always present in the config it guarded on.

65 jobs are queued, roughly 24 hours of wall clock at four at a time, ordered so
the s05 grid and the s04/s12 re-runs land first and s10's 5-to-8-hour cells last.

## The coefficient cross, COMPLETE: cosine loses everywhere, the rest is a wash

s13's space comparison under all four coefficients defined on a hybrid space,
all three datasets, every cell at 1 configuration so the hybrid spaces are built
the same way. Best space by the mean activity gap to the k nearest molecules,
lower better:

    dataset       manhattan   euclidean   cosine    minkowski   best
    herg_ki          0.5575      0.5646   0.5739       0.5555   minkowski
    caco2            0.2947      0.3010   0.3006       0.2930   minkowski
    adme_mdr1        0.4954      0.4993   0.5071       0.4962   manhattan

**cosine is worst on all three** and is the only coefficient under which a base
representation beats every hybrid — `rep_atom_pair` on herg_ki, the one cell in
this entire suite where a hybrid does not rank first.

**manhattan and minkowski are within 0.002 everywhere** and split two-one.
minkowski costs 25 to 50 times manhattan, so manhattan remains the right default
on cost. A hybrid wins 11 of the 12 cells.

⛔ The two statistics disagree about minkowski. s06 measured it WORSE than
manhattan on distance-error correlation, 0.2144 against 0.2423 on caco2; here it
is better on retrieval on two datasets of three. Whichever statistic is used to
choose a coefficient decides the answer, and nothing in the suite says which
should decide.

This closes the gap identified on 2026-09-24: before these runs, every stored
result in the suite used manhattan and the hybrid spaces had never been compared
under any other coefficient.

## What the suite sweeps, and what it cannot: coefficients vs spaces

Asked directly by the owner 2026-09-24. Checked in the code, not inferred.

**Representation type is swept, as intended.** s13 compares 7 base
representations, 3 predictive hybrids, 3 similarity hybrids and PCA controls.
s06 compares 7 base representations and their weighted twins against `hybrid`,
`hybrid_weighted`, `hybrid_similarity` and `hybrid_similarity_weighted`. That
axis works.

**The similarity formula is not swept, and most formulas cannot be.**
`distance_config.DISTANCE_METRICS` is exactly four: manhattan, euclidean,
cosine, minkowski p=0.5. There is no tanimoto, dice, jaccard, tversky or
soergel in the hybrid engine.

That is a definition problem, not an omission. `hybrid.py:383` standardises every
representation block before the hybrid is assembled, and the importance weights
then multiply those z-scored columns. Measured on caco2 ecfp4: raw is binary,
0 to 1; standardised it runs -0.67 to 17.29. Tanimoto, Dice, Tversky and Soergel
are set-overlap ratios needing |A∩B| / |A∪B|, which needs set membership. A
z-scored column has none and its negative values make the denominator
meaningless. `simeval.py` already encodes the same constraint, splitting its
coefficients into BIT_COEFFS, COUNT_COEFFS and CONT_COEFFS with only cosine and
euclidean in the continuous set.

**Where each cross does and does not exist:**

    formula x base representation   EXISTS, in similarity_metrics/, six formulas
    space type x one formula        EXISTS, in s01-s13, all at manhattan
    formula x hybrid space          bit coefficients: undefined
                                    continuous four: NEVER RUN

⛔ The real gap: every one of the 36 stored result files records `manhattan`.
The hybrid spaces have never been compared under euclidean, cosine or
minkowski. That is 3 coefficients x 3 datasets and it is computable today. Not
queued — the owner capped local compute and the current queue is still draining.

## s01–s13

| exp | kind | smoke tested / debugged | methods to optimise | optimisation done |
|---|---|---|---|---|
| s01 | investigation | yes, 3/3 datasets | no — it is the comparator s02 is measured against | n/a |
| s02 | method | yes, 3/3 | yes | **NO — 6 cells, not the 20 claimed.** 1 weight method x 2 lineages x 3 budgets; `weight_methods` has one value |
| s03 | investigation | yes, 3/3 | no — the file stamps `descriptive_only=True` | n/a |
| s04 | method | yes, 3/3 twice | yes | **re-run on 7 reps 2026-09-24** — the 4-rep pool understated the HYBRID, not the comparator |
| s05 | method | yes, 3/3 | yes | **SWEPT 2026-09-24** — 24 cells: 2 voting x 4 exponents x 3 datasets, k swept within |
| s06 | method | yes, 3/3 twice | yes | **yes** — 4 levers, 2,628 cells; comparator re-pointed 2026-09-25 to the quantity s06 measures |
| s08 | investigation | yes, 3/3 twice | no | n/a |
| s09 | investigation | **yes, ran 2026-09-21 on all three** | no | n/a |
| s10 | method | **yes, 2026-09-23 — first full run on all three** | yes | **partial** — 3 levers swept (k, q-RASAR k, length-scale start); exponent, voting, weight convention, normalize_y each at ONE value. Headline no longer reports the collapsed arm (2026-09-25) |
| s11 | method | yes, 3/3 at two budgets | yes | **UNBLOCKED 2026-09-24** — all 11 arms, 3 datasets, matched pairs. Length-scale start swept; label direction fixed 2026-09-25 (caco2 was hunting the WRONG end) |
| s12 | method | yes, 3/3 twice | yes | **re-run on 7 reps 2026-09-24** — the hybrid's margin roughly doubles on all three |
| s13 | method | yes, 3/3 | yes | **retrieval SWEPT** — neighbour count (9 cells) + coefficient (12 cells). Clustering undefined on a reseed |

## minkowski p=0.5 is queued, not cut

s06's sweep ran manhattan, euclidean and cosine and left minkowski p=0.5 out on
cost: 1,471 seconds a fold against 414 seconds for the other three together.
Cost is not one of the three reasons a cut needs. Three runs are queued, one per
dataset, at `S06_COEFFS=minkowski_p0.5 S06_TAG=mink`, writing
`sweep_<dataset>_mink.json` beside the others. Roughly two hours a dataset, and
the supervisor runs them when the machine has room.

Nothing here says how it will rank.

## Who is covering what, as of 2026-09-21 02:05

Every row above with an incomplete optimisation now has someone on it. Nothing
in the table is unassigned.

| what | who | state |
|---|---|---|
| s05 re-run, s09, `similarity_optimized_hybrid` | agent, running | s09 done on all three; s05 re-run in flight |
| s06 | done 2026-09-21 | four levers swept on all three; hybrid is the best arm on caco2 only |
| s10 | agent, started 02:05 | has never completed a full run; three tasks in it |
| `similarity_metrics/` | agent, running | ~25 scripts, several of them method experiments |
| s11 | this session | re-opened: the length scale, and the arms that picked by row number |
| s01 s02 s03 s04 s08 s12 s13 | closed | see the per-method section below |

Runs go through `/Volumes/seagate/kirby_simrun_logs/supervisor_jobs.tsv`, not
straight onto the machine. A supervisor reads that file every 120 seconds,
launches a job when free swap and load allow, defers it when they do not,
restarts it if it dies, and marks it done when its output file appears. Logs sit
beside it. Five jobs are queued there now.

The machine is the binding constraint, not the code. Load has sat above 400 for
hours and free swap between 0.75 and 1.7 GB of 16 GB, and about 19 of the 27
running python jobs belong to other sessions.

## Never opened

| group | scripts | note |
|---|---|---|
| s09 | 1 | opened 2026-09-21. Ran on all three, read all five inputs, wrote 11 series |
| standalone | `similarity_optimized_hybrid` | imports and prints help. `tests/results/sim_optimized_hybrid/` exists and is empty — nothing has ever been written to it |
| standalone | `structural_geometry_survey` | cannot address these three. It reads heavy and light chain sequences, and its registry holds 31 antibody datasets — caco2, herg_ki and adme_mdr1 are not among them |
| `similarity_metrics/` | ~25 per EXPERIMENTS_FINALIZED.md | several are method experiments: nary, triplet, noise-cleaning, creative_hybrids |

---

## What the optimisation found, per method

**s02 — the negative was the comparator, not the hybrid.** At weighting
exponent 0 both arms are unweighted. The hybrid barely moves (herg_ki 0.2669
weighted, 0.2669 unweighted, euclidean); the best single representation falls
from 0.2886 to 0.0779. A 2,048-bit fingerprint is only competitive as a
distance because importance weighting rescues it. Held-out delta, hybrid minus
best single representation:

    coefficient, exponent      herg_ki    caco2   adme_mdr1
    manhattan, 1 (shipped)     -0.0237  +0.0260   -0.0233
    euclidean, 0               +0.1891  +0.1283   +0.0453

**s02 — the nine cells I queued sweep the build, not the weighting; corrected
2026-09-25.** I recorded these earlier as a weight-method sweep. That was
wrong. The nine cells set `KIRBY_LOCAL_IMPORTANCE`, which picks the method the
hybrid search uses to rank features while it spends the budget. It changes how
the hybrid is built. The method that weights features inside the distance s02
scores is `weight_methods`, a different thing, and it stayed on lightgbm_gain
in every one of them. The landed deeplift cell shows it: its `weight_methods`
block still reads lightgbm_gain.

Why it could not vary: `SIMILARITY_EXTRA` carried `weight_methods:
['lightgbm_gain']`, and that key sits in the config every similarity run is
built from, so it always won. The plumbing handing it to s02 was already
correct. This is the same trap `k_values` was in. `KIRBY_S02_WEIGHT_METHODS`
now resolves it; three jobs are queued that compute all three methods against
one hybrid per dataset, which compares them on identical features.

The number below is the Spearman correlation between the distance separating a
pair of molecules and the absolute difference in their activities, over 5000
sampled pairs; higher is better. Seven representations, budget 100 features,
one search configuration, distance weighted by lightgbm_gain throughout.

    build importance  dataset   lineage           unweighted  weighted   weighted minus
                                                                         best single rep
    lightgbm_gain     caco2     general               0.3586    0.3704       +0.0244
    lightgbm_gain     caco2     general_distance      0.3466    0.3656       +0.0197
    lightgbm_gain     herg_ki   general               0.3067    0.2693       -0.0671
    lightgbm_gain     herg_ki   general_distance      0.3109    0.2713       -0.0651
    deeplift          herg_ki   general               0.1198    0.1408       -0.1956
    deeplift          herg_ki   general_distance      0.1131    0.1422       -0.1942

Two things the three landed cells say. Weighting raises the correlation on
caco2 and lowers it on herg_ki, and both lineages agree in sign and to within
0.005 inside each dataset, so that split is between the datasets. And building
the hybrid with deeplift instead of lightgbm_gain more than halves the
correlation on herg_ki, 0.3067 to 0.1198 unweighted, which is a far larger
move than the weighting makes.

These cells use the one-configuration search; the stored run above used twelve,
so they are comparable to each other and not to the shipped row. 6 of the 9
build-importance cells are outstanding, and the 3 weighting cells have not
started.

**s04 — wins two of three once its levers open.** Held-out RMSE delta on cliff
molecules, positive means the hybrid won:

    setting                                 herg_ki    caco2   adme_mdr1
    shipped: 5 neighbours, sqrt, 100 feat   -0.0184  -0.0088   -0.0074
    neighbours/exponent/budget/family open  -0.0036  +0.0269   +0.0315

**s06 — the neighbour count was the lever, and it was fixed at 5.** Four
things decide every distance s06 measures and none of them could be set: the
pairwise coefficient, the power a feature's importance is raised to, how many
nearest training molecules the distance is taken over, and which number is taken
off those distances. The file computed the mean over 5 neighbours under
manhattan at exponent 0.5, written out inline at twelve places.

Each row is one dataset. The number is the Spearman correlation between a
held-out molecule's distance to its nearest training molecules and the absolute
residual of a random forest fitted on every representation glued together — the
block where the errors come from a model that never saw the hybrid features.
Higher is better. "Best representation" is whichever of the seven scores highest
in that same cell.

    dataset    cell                              best hybrid arm   best representation
    caco2      manhattan 0.5 mean k=5 (shipped)      +0.2402         avalon_w +0.2088
    caco2      euclidean 0.25 harmonic k=150         +0.3122      atom_pair_w +0.2808
    herg_ki    manhattan 0.5 mean k=5 (shipped)      +0.2970            pdv_w +0.3024
    herg_ki    manhattan 0.25 mean k=25              +0.3296            pdv_w +0.3358
    adme_mdr1  manhattan 0.5 mean k=5 (shipped)      +0.3235      atom_pair_w +0.3840
    adme_mdr1  euclidean 0.25 harmonic k=15          +0.3328      atom_pair_w +0.3846

The hybrid arm is the best arm on caco2 and on caco2 only. On herg_ki it ends
0.0062 Spearman behind pdv; on adme_mdr1 0.0518 behind atom_pair. Widening k
lifts every arm at once, so it raises the hybrid's own number without closing
either gap. The same three rows under the other two correlation blocks — pooled
against the hybrid-fitted model's errors, and the within-fold mean — move each
number by at most 0.015 and change no ranking.

Which lever moved it, largest first. k, from 5 to the dataset's own best, is
worth +0.007 to +0.072 Spearman: the curve peaks at 150 neighbours on caco2
(still rising there, and 150 is 8.7% of its 1,727 training molecules), 50 on
herg_ki and 15 on adme_mdr1. The coefficient is worth up to +0.034 — euclidean
beats manhattan on caco2 and adme_mdr1, manhattan beats it on herg_ki, and
cosine is far worse than both on caco2, +0.0960 against +0.2402 at the shipped
cell. The statistic is worth up to +0.005: k / sum(1/d), which counts a close
training molecule for more than a far one, wins on caco2 and adme_mdr1 and the
plain mean wins on herg_ki. The distance to the single nearest training molecule
is the k=1 column and is the worst k on all three. The exponent moves the
hybrid's own number least of the four, 0.25 or 0.5 best and 1.0 worst.

**s06 — the exponent decides who wins, the same way it did in s02.** Exponent 0
means nothing is weighted. At each dataset's own best coefficient, statistic and
k, hybrid minus best single representation:

    exponent    caco2   herg_ki  adme_mdr1
    0         +0.0956   -0.0034    +0.0333
    0.25      +0.0314   -0.0034    -0.0518

Weighting a 2,048-bit fingerprint by its lightgbm_gain importance lifts it far
more than it lifts the hybrid: atom_pair on adme_mdr1 goes +0.2954 unweighted to
+0.3846 weighted, while the hybrid goes +0.3287 to +0.3328. So the cell where
the hybrid's own correlation is highest and the cell where its margin is largest
are different cells, which is the open decision already recorded for s02.

**s06 — the comparator is ranked by prediction and the experiment measures a
distance.** `comparator_rep` is `extract_top3_base_reps(search, model='knn')[0]`:
the representation a kNN predicts the label best from, by RMSE. What s06
measures is which representation's distance ranks error best. Those are
different orderings. Ranked by how well each weighted distance tracks error at
the shipped cell, the representation s06 named as the comparator comes 7th of 7
on caco2, 4th of 7 on herg_ki and 5th of 7 on adme_mdr1:

    dataset    comparator  its rank by prediction  its rank by distance-error
    caco2         pdv              1 of 7                  7 of 7
    herg_ki       atom_pair        1 of 7                  4 of 7
    adme_mdr1     pdv              1 of 7                  5 of 7

No number already stored is wrong because of this: since 2026-09-12 every
representation and its weighted twin are arms in the correlation, the per-fold
means, the reliability deciles and the prediction intervals, so the best
distance representation is in the file whether or not it is the comparator. What
the rule decides is which arm the printed headline and the conformal
`hybrid_genuinely_tighter` flag compare the hybrid against. NOT CHANGED — the
rule is untouched pending the owner's decision.

**s06 — how the cells were held comparable.** The recipe is looked up ONCE per
dataset and every cell reuses it, so what moves across cells is the distance
between molecules and never which features the search picked. The two recipes
are the predictive search's, budget 250 on herg_ki and the similarity search's,
budget 100; both name all 7 representations. Per fold the two hybrid builds, the
importance computation and the two random forest fits happen once and every cell
reads them, which is why 2,628 cells cost about the same as 18 did.

The lookup ran against a COPY of the three result databases, at
/Volumes/seagate/kirby_s06_cache, with a `distance_metric` column added to the
copied rows reading manhattan — the coefficient the 2026-09-19 s06 run was
launched under. Nothing under kirby_simrun_cache was written or deleted. The
copy reproduces that run exactly: the comparator, the recipe family and 16 of
the 18 pooled arms match tests/results_local_sim/s06_applicability_domain to
four decimals on herg_ki. The two that move are `hybrid_similarity` 0.2977 to
0.2886 and `hybrid_similarity_weighted` 0.2958 to 0.2912, because
`build_fold_hybrid`'s geometry allocation honours the process coefficient since
`c5752d4` and did not on 2026-09-19.

s06 itself was then run end to end on herg_ki through
`_local_autoresearch.py` with the three levers set away from their defaults —
the distance-weighted statistic, k=5 and 25, exponent 0.25 — and its four
neutral-model arms reproduce the sweep to four decimals at both neighbour
counts. The file writes `distance_statistic`, `k_values` and
`importance_exponent` into its result, so a stored number now says what its
distance was.

minkowski p=0.5 is the one coefficient NOT measured here. Timed on this machine
at 1,471 s a fold against 414 s a fold for manhattan, euclidean and cosine
together, which is the 25-50x the rest of this file records. It is a compute
cut, not a finding: nothing here says how it would rank.

**s10 kNN — distance-weighted voting is the reliable win.** It lowered held-out
error in 20 of 21 (dataset, representation) pairs and nothing in the suite uses
it. The hardcoded 0.5 exponent is never best; k needs no widening.

    dataset      no weighting   as shipped   best swept   picked
    caco2            0.3561       0.3420       0.3350     exp 0.25, 7 nbrs, distance vote
    herg_ki          0.6631       0.6550       0.6378     exp 0.25, 5-7 nbrs, distance vote
    adme_mdr1        0.6663       0.5878       0.5814     exp 1.00, 15 nbrs, distance vote

Underneath it: the weighting is mostly feature deletion. lightgbm_gain scores
75-90% of a fingerprint's features at exactly zero, so any positive exponent
deletes them. Dropping always helps; rescaling the survivors helps on two
datasets and hurts on herg_ki by 0.0133 pKi.

**s11 — the mechanism is the distance scale, not the number of features.**
Median Euclidean distance between two caco2 molecules, against an RBF kernel
starting at length scale 1.0: 50 selected features 6.6, maccs 13.8, pdv 16.7,
ecfp4 45.3, PCA-50 83.1, random-50 102.4. The spaces that collapse are the ones
whose median distance sits far above 1. PCA-50 and random-50 have fifty
features each and still collapse. Area under the best-found curve at 100 picks,
lower better: herg_ki hybrid 415.1 against raw ECFP4 429.3; adme_mdr1 hybrid
-87.40 against -55.26; caco2 nothing separates and random picking beats the
hybrid.

## s11 length scale, MATCHED on all three datasets

Both halves at 1 configuration, so the recipes are identical and the length
scale is the only difference. Replaces the confounded comparison retracted
below. Gain is oriented so positive means the median start won.

**adme_mdr1** — the largest effect by an order of magnitude:

    condition          fixed   median    gain   fix ro   med ro
    random_50d         -55.3    -86.8   +31.5      300        0
    ecfp4_raw          -55.3    -82.4   +27.2      300        0
    pca_50d            -55.3    -79.7   +24.4      298        0
    six other arms       unchanged      +0.0        0        0
    hybrid_ngboost     -88.7    -83.9    -4.8        0        0
    best_single        -80.5    -73.3    -7.3        0        0

⛔ The three gaining arms score **identically at -55.3** under the fixed start.
Three different methods returning the same number is the signature of three arms
all picking the same molecules in file order — 300, 300 and 298 picks of 300.
They were not running different methods; they were running the same file-order
sweep three times.

**herg_ki** — same mechanism, much smaller: `random_50d` +5.3 with 300 picks
removed, `ecfp4_raw` +0.3 with 295, `pca_50d` -4.6 with 224.

**caco2** — smaller again: `pca_50d` +1.4 with 58 removed, `best_single` +0.9
with 100, `ecfp4_raw` +0.8 with 100.

**The correspondence is exact across all three datasets.** Every arm that moves
is an arm that had row-order picks; every arm that had none is unchanged to the
decimal. Six arms are identical on all three datasets.

**The magnitude is dataset-dependent and the direction is not guaranteed.**
31.5 on adme_mdr1, 5.3 on herg_ki, 1.4 on caco2; `pca_50d` gains on two datasets
and loses 4.6 on herg_ki. The fix makes each arm's number mean what it claims.
It does not reliably improve an arm and it does not change who wins — the hybrid
arms lead on all three datasets under both settings.

**`hybrid_bnn` is identical on both sides on all three datasets** — 158, 27 and
163 picks each way. Its flat acquisition is independent of the length scale, and
its cause remains unidentified.

## RETRACTED: the s11 fixed-vs-median AUC comparison is confounded

Both pairs were reported on 2026-09-24 as differing in one thing, the RBF
length-scale start. They differ in two. Read from the run logs:

    run                    started    search             allocation methods
    s11_median_herg_ki     06:56:15   12 configurations  performance_weighted, power_2
    s11_fixed_herg_ki      17:39:32    1 configuration   performance_weighted
    s11_median_caco2       03:09:46   12 configurations  performance_weighted, power_2
    s11_fixed_caco2        16:36:58    1 configuration   performance_weighted

The `median` runs went out before the search space was cut and the `fixed` runs
after. Different searches choose different recipes — `power_2` against
`performance_weighted` — so every hybrid arm was built from a DIFFERENT FEATURE
SET in the two halves.

⛔ **Retracted:** every AUC number from those two comparisons, and both
conclusions drawn from them — that the median start wins all eleven arms on
caco2, and that it wins 3 of 11 on herg_ki. Neither is attributable to the
length scale.

**Still standing:** the row-order counts. `ecfp4_raw` 295 to 0 on herg_ki and
100 to 0 on caco2, `pca_50d` 224 to 0 and 58 to 0, `random_50d` 300 to 0 on
both. Those arms are base representations and PCA controls; their features come
from no recipe, so the search change does not touch them. The median start
removes row-order picking, and that claim is clean.

Cause was mine: the search was cut mid-queue and the half that had already
finished was not re-run. Three re-runs of the `median` arm on the
1-configuration search would make the pair matched.

**s11 — FIXED 2026-09-23. It was torch taking every core, not the model.**
`hybrid_bnn` fits a PyTorch MC-dropout network, and torch opens a full-core
thread pool by default. s11's loop refits the surrogate once per BO iteration,
so every fit opened one, and several concurrent runs multiplied it. The kernel
kills a process that does that, and a kill leaves no traceback — which is
exactly the empty logs all nine attempts produced.

`KIRBY_TORCH_THREADS` now caps it, in both the regressor and the classifier in
`src/kirby/models/bnn.py`. Unset leaves torch alone so nothing on the cluster
changes. Measured: unset gives 4 threads, `=1` gives 1, fit and predictions
unchanged either way.

With the cap, both arms that had never produced a number completed on caco2 at
250 molecules, 3 picks, 1 seed:

    condition          dim   surrogate failures   picks by row order
    hybrid_bnn          50            0                3 of 3
    hybrid_gauche_gp    50            0                0 of 3

That is the first number either arm has produced in this campaign. All eleven
arms are back in the six queued local runs. `hybrid_bnn` taking 3 of 3 picks by
row order at three iterations is not yet meaningful — it is three picks — but it
is the same flat-acquisition signature the other arms show and should be read
against the full runs.

**s11 — one arm of eleven kills the process, and it is why nothing finished.**
Measured 2026-09-23 on caco2 at 250 molecules, 3 picks, 1 seed, with nothing
else running on the machine. The nine conditions before `hybrid_bnn` completed
and were written to disk; the process was then killed inside `hybrid_bnn` with
no traceback and no exit message. That is the same condition this file already
records as killing three processes at 14 GB of 15 GB swap and recording nothing.

So the nine launches over 2026-09-22 and 09-23 that each ran two to five hours
and produced nothing were not defeated by run length or by machine load. They
reached `hybrid_bnn` and died there. `hybrid_gauche_gp` is listed after it and
has therefore never run once, on any dataset, in this campaign. The `gauche`
package is installed at 0.1.6, so that arm is expected to work; it has simply
never been reached.

The condition list, in the order s11 runs it:

    1 ecfp4_raw          5 hybrid_50d         9 hybrid_ngboost
    2 tanimoto_ecfp4     6 hybrid_50d_refit  10 hybrid_bnn        <- kills the process
    3 best_single        7 random_50d        11 hybrid_gauche_gp  <- never reached
    4 pca_50d            8 random_acq

**s11 now writes each condition as it finishes.** Everything was held in memory
and returned to the caller only after the last condition, so every kill threw
away every condition already done. Each one is now written to
`<output_dir>/.s11_checkpoint_<dataset>.json` through a temporary file and a
rename, and a rerun skips what is already there. Verified by reading the file
mid-run: nine conditions on disk while the process was still going.

⛔ The checkpoint sits in the directory the driver passes as `output_dir`, which
is the parent — not the `s11_bayesian_optimization/` subfolder the results land
in. Two checks for it in the wrong place on 2026-09-23 reported it missing when
it was being written correctly.

**The local runs name ten arms, and `hybrid_bnn` is not cut.** `run()` has always
taken a `conditions` list; nothing exposed it. `KIRBY_S11_CONDITIONS` now does.
The six queued local runs name the ten that complete. Naming `hybrid_bnn`, or
running s11 on the cluster, runs it. The stated reason for leaving it out of a
local run is that it kills the process and records nothing — broken on this
machine, not slow on it.

**s11 — the arms the hybrid beats were not using their surrogate.** Counted from
the three budget-100 logs, picks made by dataset row order out of 300 (three
seeds of 100 picks each):

    arm                 caco2   herg_ki   adme_mdr1
    random_50d            300       300         300
    ecfp4_raw             100       295         300
    pca_50d                58       224         298
    best_single           100         0           0
    tanimoto_ecfp4          0         0           0
    hybrid_50d              0         0           0
    hybrid_50d_refit       12         0           0

The hybrid arm chose every one of its 900 picks with its surrogate. Three of the
arms it is compared against chose most or all of theirs by row number. So the
two sides of that comparison differ in two ways at once, not one: the features
differ, and on one side the acquisition function is not being consulted.

This bears hardest on the caco2 line above. `random_50d` made 300 of 300 picks
in row order, so what beat the hybrid on caco2 is not random picking — it is the
order the molecules sit in the file.

**s11 — the length scale the RBF starts at decides whether the surrogate works.**
Each arm standardises its features on the seed molecules before fitting, and
`fit_gp` then starts the RBF length scale at 1.0 with scikit-learn's default
bounds. The distance between two standardised caco2 molecules is 13.50 in maccs
and 34.75 in ecfp4, so 1.0 sits far below anything the data has. Measured on 40
caco2 seed molecules against a smooth function of their own features, starting
instead at the median distance between two labelled molecules:

    arm             start   fitted length scale   tied at the maximum   rank correlation
    caco2 ecfp4     1.0                       1          200 of 200            +0.0819
    caco2 ecfp4     median                 22.5            1 of 200            +0.5072
    caco2 maccs     1.0                       1            1 of 200            +0.3999
    caco2 maccs     median                27.43            1 of 200            +0.8125

Tied at the maximum counts the molecules in the pool sharing the largest
Expected Improvement; when all 200 tie, `np.argmax` returns the lowest row
number and the pick is made by row order. `KIRBY_S11_LENGTHSCALE=median` is the
lever and the default is unchanged. Unverified on the real permeability label
and on the area under the best-found curve, which is what s11 reports. Six runs
are queued to settle it: both settings on all three datasets, at full molecule
count and 100 picks, so the pair is computed on the same molecules under the
same code. The recorded `fixed` numbers cannot serve as that half of the pair —
they were read off stored search configurations that no longer exist.

A refinement to the sentence above about which spaces collapse. The direction
holds — random-50 at 102.4 and PCA-50 at 83.1 collapse and maccs at 13.8 does
not — but the immediate cause is the optimiser stopping where it started rather
than the distance itself. Fitted from a start of 1.0 on standardised caco2
features: maccs at a median distance of 13.71 leaves and settles at 3875, pdv at
16.90 settles at 4057, ecfp4 at 50.15 does not move off 1 at all, ecfp4_count at
41.99 goes the wrong way to 3.632e-4. Extra optimiser starts do not rescue it:
at 300 training molecules, none and two gave the same fitted length scale, the
same prediction spread to the last digit, and the same rank correlation +0.1442.
So the quantity that decides it is how far the start sits from the data, not the
distance on its own.

**s12 — wins all three under the calibrated cliff rule.** Held-out delta:
caco2 +0.0154, herg_ki +0.0107 (was -0.0024), adme_mdr1 +0.0416.

**s13 retrieval — euclidean beats the shipped manhattan on two of three.**
Held-out delta in label units at exponent 0.5: caco2 manhattan +0.0181 against
euclidean +0.0236; adme_mdr1 +0.0504 against +0.0594; herg_ki +0.0245 against
+0.0023. Manhattan is the default because f18 measured it on herg_ki and logd.

**s13 clustering — the headline subtracted coherence at 3 clusters from
coherence at 10.** Every space is scored at 3, 5 and 10 clusters and each arm
keeps its own best count per fold. Label coherence rises with the count — at 10
clusters it is higher than at 3 in 44 of the 50 (dataset, space) pairs on disk —
because more clusters means fewer molecules per cluster and a smaller
within-cluster standard deviation. On herg_ki the hybrid's best count is 3 and
the best single representation's is 10, so the reported -0.1183 is 0.3866
against 0.5050, at two different granularities. Held-out label coherence delta,
unitless, higher better, positive means the hybrid won, 4 paired folds on both
arms (fold 0 excluded because the pool pruned on its scores):

    clusters         caco2   herg_ki  adme_mdr1
       3           +0.0145   -0.0129    +0.0340
       5           +0.0519   +0.0584    +0.0067
      10           -0.0381   -0.2417    +0.0264
    each arm free  -0.0381   -0.1183    +0.0264

At 5 clusters the hybrid wins all three. ⛔ That win does not survive a reseed —
see the instability note below before quoting it.

**A cluster holding fewer than 2 held-out molecules contributes no term, and
which spaces lose clusters is not even.** Summed over 5 folds on herg_ki at 10
clusters, ecfp4 loses 16 of its 50 clusters and top_torsion 20 of 50, while the
six hybrid spaces lose between 0 and 4. ecfp4 is the single representation that
wins at that count, and its 0.4761 is a mean over the clusters that held 2 or
more held-out molecules — about 34 of 50. The classification purity branch was
given the same 2-molecule floor in `df7c7fe`; all three datasets here are
regression, so that branch never fires on them.

**s13's PCA control was matched on one dataset of three.** `pca_matched` is
fitted at 100 components on every fold. Each hybrid slot's width is whatever
budget its recipe chose, from 25, 100 and 250. Of the 30 (hybrid slot, fold)
pairs per dataset, the control's 100 components equal the slot's feature count
in 29 on adme_mdr1, 15 on herg_ki and 5 on caco2 — 24 of caco2's 30 are 250
features wide. The per-width controls added in `df7c7fe` did not fix it: a width
that occurred on one fold produced a control scored on one fold beside a hybrid
scored on five, and the two means were read side by side.

Each control is now named for the hybrid slot it controls and refitted per fold
at that fold's slot width, so the two always rest on the same folds, and both
axes now report the re-pick against it. Held-out delta, hybrid minus its
own-width PCA control, 4 paired folds, 2 restarts, positive means the hybrid
won:

    dataset      clustering (k-means)   clustering (k-medians)   retrieval
    caco2               -0.0307                +0.0200            +0.0337
    herg_ki             -0.0648                +0.0256            +0.0230
    adme_mdr1           +0.0858                +0.1146            +0.1104

The retrieval half beats the control on all three. The clustering half beats it
on one, and changes sign on the other two when the clusterer changes — the same
instability as everywhere else in this axis.

**⛔ THE CLUSTERING HALF CANNOT RESOLVE THE DELTA IT REPORTS.** Label coherence
is not what k-means minimises, so two clusterings of one space with almost the
same inertia can score far apart, and running k-means again from a different
seed changes the number by much more than the hybrid-minus-best-single gap.
Measured on herg_ki fold 1 at 5 clusters, 1,129 training and 283 held-out
molecules, one space at a time, over four `random_state` values. The column is
the highest coherence minus the lowest, unitless:

    restarts   top_torsion 2048 feat   ecfp4 2048 feat   pdv 200 feat
        1              0.3719               0.4174          0.0293
        2              0.2235               0.3491          0.0453
        5              0.4475               0.2259          0.0464
       10 (shipped)    0.2986               0.3416          0.0176
       20             0.2149               0.1398          0.0124
       40             0.2149                  —            0.0124

The delta the clustering half reports is 0.01 to 0.05 coherence. At the shipped
10 restarts the seed alone moves a 2,048-feature fingerprint by 0.30 and 0.34 —
6 to 34 times the gap being measured. Raising the restarts to 20 halves it and
still leaves 0.14 to 0.21. Only the 200-feature descriptor set settles: 0.0124
from 20 restarts on, and 40 buys nothing over 20. The single representations
the hybrid is scored against are the 2,048-feature fingerprints, so it is the
COMPARATOR that will not hold still.

Two whole-experiment re-runs on herg_ki, same folds and same pool, 10 restarts
against 2, differ by up to 0.19 coherence on one space (rep_top_torsion at 3
clusters, 0.2869 against 0.0925) and flip the sign of the delta at 3, 5 and 10
clusters. Held-out label coherence delta, positive means the hybrid won, 4
paired folds:

    clusters       10 restarts   2 restarts
        3            -0.0129       -0.2240
        5            +0.0584       -0.1368
       10            -0.2417       -0.0785

**The count sweep, all three datasets, 2 restarts.** Six counts instead of the
three that were hardcoded. Held-out label coherence delta, unitless, positive
means the hybrid won, 4 paired folds on both arms:

    clusters     caco2    herg_ki   adme_mdr1
        2      -0.0517    +0.0541    +0.0927
        3      +0.0174    -0.2240    +0.0957
        5      +0.0343    -0.1368    +0.0456
        8      -0.0466    -0.1961    +0.0077
       10      +0.0492    -0.0785    +0.0458
       15      -0.0567    -0.2353    +0.0278

adme_mdr1 is positive at all six counts, under k-medians as well (+0.0320 at 5
clusters) and against its own-width PCA control (+0.0858). That is the one
dataset where the clustering half gives a consistent answer, and it is also the
one whose winning single representation is the 200-feature descriptor set rather
than a 2,048-feature fingerprint. herg_ki loses at five of six counts. caco2
changes sign at every step, which is what a measurement looks like when the
noise is larger than the effect. There is no right cluster count to be found on
caco2 or herg_ki until the statistic stops moving.

**The distance mismatch, measured.** k-medians honours the manhattan
coefficient the retrieval half reads; k-means minimises squared euclidean and
takes no metric argument. Same run, same folds, same 2 restarts, 5 clusters,
held-out delta, positive means the hybrid won:

    dataset      k-means (squared euclidean)   k-medians (manhattan)
    herg_ki                -0.1368                   -0.1090
    caco2                  +0.0343                   -0.0514
    adme_mdr1              +0.0456                   +0.0320

Making the two halves agree moves the delta by +0.0278 on herg_ki, -0.0857 on
caco2 and -0.0136 on adme_mdr1. It flips caco2's sign and leaves the other two
verdicts standing. All three movements sit inside the seed spread above, so the
mismatch cannot be separated from the noise at this restart count. `minkowski p=0.5` has no clusterer of this shape at all — there is no
closed form for the point minimising summed L_0.5 distance — and the file now
writes that instead of running euclidean silently.

**s10 GP — a null, and a correction.** Multiplying features by the square root
of their importance destroys an RBF kernel: the weighted arm returned the
training mean on all three datasets (herg_ki label spread 0.9151, weighted arm
0.9160). With the arm alive, weighting makes the GP slightly worse, +0.0111 and
+0.0188 pKi on two folds of herg_ki/pdv.

**s10 GP — the length scale it starts at decides the arm, measured 2026-09-21.**
s10 standardises its features per fold and starts the RBF length scale at 1.0
with bounds of 1e-5 to 1e5. Two standardised caco2 molecules sit 50.76 apart in
ecfp4 at 300 training molecules and 41.99 apart in ecfp4_count, so 1.0 is far
below anything the data has. Fitted on cached caco2 features with 100 molecules
held out, against a smooth direction in the features as the label:

    representation      training   start    fitted   spread of the 100   rank correlation
                        molecules           scale    predictions
    ecfp4 (2048)             40    1.0           1            2.08e-17            +0.0533
    ecfp4 (2048)             40    median    20.28              0.0531            +0.6677
    ecfp4 (2048)            300    1.0           1             0.00825            -0.0401
    ecfp4 (2048)            300    median    164.2                 0.1            +0.8462
    ecfp4_count (1024)       40    1.0           1            5.55e-17            +0.2108
    ecfp4_count (1024)       40    median    27.53               0.104            +0.4889
    ecfp4_count (1024)      300    1.0    3.632e-4              0.0394            +0.0619
    ecfp4_count (1024)      300    median    453.3               0.212            +0.9037

A spread of 2.08e-17 across 100 predictions is a constant predictor. maccs at
167 features and pdv at 200 escape on their own, fitting 3875 and 4057 from a
start of 1.0, which is what s10's own docstring records about pdv.

`n_restarts_optimizer` does not rescue it. At 300 molecules restarts=0 and
restarts=2 gave the same fitted length scale of 1, the same prediction spread to
the last digit and the same rank correlation +0.1442, so the expense buys
nothing here and KIRBY_S10_GP_RESTARTS is not the lever.

s10 already carries this lever: `_make_gp(X, start='median')` at
s10_importance_downstream.py:1013 runs one arm per starting length scale, on the
same median-distance basis and the same bounds. The numbers above were measured
independently of it and agree with it.

The label used here is a smooth direction in the features, not the real caco2
permeability, and this was not run through s10. Whether it moves the rmse s10
reports is unmeasured. The recorded s10 GP null — weighting worse by +0.0111 and
+0.0188 pKi — is on herg_ki pdv, the representation that escapes; the wide ones
may have been comparing two constant predictors.

**s10 GP — the full run, and which arm actually collapses. Measured 2026-09-23.**
The first complete s10 run scored both length-scale starts as arms on all three
datasets. `_gp_fit_report` records the spread of each fold's predictions, so a
constant predictor is visible rather than inferred. A spread below 1e-6 across a
fold's held-out molecules is a GP returning one number for every molecule.

Constant predictors occur on adme_mdr1 only, and the two arms collapse on
different representations:

    adme_mdr1, smallest prediction spread over 5 folds
    representation     start 1.0     start median
    atom_pair          1.11e-149            0.489
    avalon             4.93e-161            0.501
    ecfp4_count         1.64e-83            0.506
    pdv                    0.514         1.88e-37   <- 4 of 5 folds at a bound
    maccs                  0.443            0.443
    ecfp4                0.00892            0.471
    top_torsion          0.00896            0.454

So neither start is clean. The shipped start of 1.0 turns three of the seven
representations into constant predictors; the median start rescues those three
and breaks `pdv` instead, at four of its five folds finishing on a bound.

herg_ki and caco2 have no constant predictor under either start; every
prediction spread there is between 0.04 and 0.74. One fold of `maccs` finishes
at a bound under the median start on both.

The rmse this produces, lower better: on adme_mdr1 the three collapsed
representations all report 0.7939 under start 1.0 and 0.4853 to 0.5059 under
median; `pdv` reports 0.4601 under 1.0 and 0.7939 under median. 0.7939 is what a
constant predictor scores on this dataset, which is why it appears on both
sides.

⛔ `lengthscale_reported` is `ls_starts[0]`, which is `fixed`. The unsuffixed
blocks in every s10 result file therefore carry the start that collapses three
adme_mdr1 representations. Both arms are stored under
`per_rep_by_lengthscale`, so nothing is lost, but the headline numbers are the
collapsed ones. Which arm should be the headline is the owner's call and has not
been changed.

**The same collapsed GP scores two different numbers in this suite.**
s10 fits its Gaussian process with `normalize_y=False`, which leaves the prior
mean at 0, so a fit that finishes at the bottom of the length-scale bounds
predicts 0 for every held-out molecule. s11 passes `normalize_y=True` at three
places, so the identical collapse there predicts the training mean instead.

Confirmed from the labels themselves, 2026-09-23, reading
`kirby_simrun_cache/chembl_herg_ki.csv` directly. herg_ki pKi over 1,415
molecules has mean +5.7057 and spread 0.9143, so:

    what a collapsed GP predicts     rmse in pKi
    zero, under s10's normalize_y=False   5.7785
    the training mean, under s11's True   0.9143

s10's `atom_pair` under the shipped start reads **5.7443**, which sits on the
predicting-zero value and nowhere near the mean-predictor one. So that arm is a
GP returning zero for every molecule, not a GP fitting badly. The same failure
under s11's convention would have read about 0.91 and looked like a mediocre
model.

Consequence: a collapsed arm's score is not comparable between s10 and s11, and
within s10 it is not on the same scale as a working arm. `at_bound` and
`pred_sd` in `_gp_fit_report` are what identify a collapse; the rmse alone does
not. `KIRBY_S10_GP_NORMALIZE_Y=1` fits the other convention. Neither default has
been changed, so no stored number moves.

**s10 q-RASAR — a null.** The four neighbour features carry real signal alone
(pdv's four reach 0.6730 pKi against 0.9160 for a mean predictor) and add
nothing to a forest that already has the features.

---

## s05's herg_ki numbers moved between 2026-09-19 and 2026-09-21 at one setting

Both runs are s05 on herg_ki at the shipped settings — 5 and 10 neighbours,
uniform vote, importance exponent 0.5. RMSE in pKi, and for the held-out delta
positive means the hybrid beat the best single representation:

    quantity                        2026-09-19   2026-09-21
    pre-registered cell delta          -0.0003      +0.0008
    max over the grid, delta           +0.0248      +0.0185
    held-out delta                     -0.0105      -0.0210
    held-out hybrid RMSE                0.6331       0.6437
    held-out best single rep RMSE       0.6226       0.6226

**It was not an empty pool.** Both files stamp `search_status: ok`, and the
base-representation side of the pool is identical to sixteen digits: fold 0
ranks atom_pair at 0.6108291357787444 in both. The best single representation
arm is byte-identical at 0.6226, which is why only the hybrid arm moved.

**The two families inside the pool disagree, and only one of them.** The three
predictive hybrids the pool served for fold 0 reproduce: 0.670692 against
0.670656, 0.674870 against 0.674794, 0.705837 against 0.705828 — same recipes,
same ranks, agreeing to four decimals. The three similarity hybrids do not:
0.474060 against 0.434943, 0.441785 against 0.422708, 0.431918 against 0.418093,
in the Spearman the similarity search stores, and rank 3's allocation method
changed from power_2 to performance_weighted. So one family moved in the fifth
decimal and the other in the second, a hundredfold apart.

`c5752d4`, committed 2026-09-20 20:02, is between the two runs. It stops
`objective='similarity'` searches building a hand-written Euclidean whatever
coefficient was asked for. The predictive family never ran through that code, so
it reproduces; the similarity family did, so it does not. On herg_ki the stored
run's held-out winner was a similarity hybrid on 3 of its 4 paired folds, so the
arm that was winning is the arm whose geometry changed. The commit states the
older numbers are reproduced bit-for-bit by asking for euclidean — unverified
here, no run has tested it.

⛔ **Do not backfill the old databases with a constant coefficient to recover
the old numbers.** The 14 files moved to `result_db_stale_2026-09-21/` hold 530
stored configurations, and roughly half of each dataset's rows are
similarity-family. Stamping those `distance_metric='manhattan'` labels them with
a coefficient they were not computed under, because the hand-written Euclidean
ignored the coefficient. The predictive rows would be labelled correctly and the
similarity rows would not, in one table, with nothing recording which.

**So −0.0105 is superseded.** At shipped settings under the fixed geometry
herg_ki's held-out delta is −0.0210. Any tuned re-run has to be read against
that, and against a pool built from the same store.

---

## What s09 says across three datasets

Ran 2026-09-21 against `tests/results_local_sim`. All five inputs loaded for all
three datasets, all fifteen (dataset, input) pairs stamped
`per_file_2026_09_18`, so no column was held back for mixed selection rules. It
wrote eleven series, each carrying three numbers — one per dataset — and never
averages them.

**The signed-rank test is not worth reading at three datasets.** Enumerated over
all eight sign patterns of three values, a two-sided Wilcoxon can return only
0.25, 0.5, 0.75 or 1.0. The floor of 0.25 is reached whenever all three share a
sign, no matter how large they are: three deltas of +5, +9 and +12 give the same
0.25 as +0.0001, +0.0002 and +0.0003. The p-value therefore carries nothing the
three signs do not already carry. Read the three per-dataset numbers.

s05's held-out kNN delta, the column s09 reads as `knn_delta_heldout`, in RMSE
where positive means the hybrid beat the best single representation:

    caco2 +0.0134, herg_ki -0.0105, adme_mdr1 +0.0292

**The two sides of s09's own `knn_delta` are not picked over the same number of
choices.** The hybrid side takes the best of 36 cells — 3 budgets x 2 search
families x 3 ranks within a family x 2 neighbour counts. The baseline side takes
the best of 7 representations, each already reduced to its own better neighbour
count out of 2, so 14 cells. So the hybrid side maximises over 36 and the
baseline side over 14, and the difference between them is read as a win for the
hybrid. `knn_delta_heldout` does not have this problem: it re-picks both sides on
the other folds.

Four things an earlier audit listed, re-checked against the code on 2026-09-21:
s09 is absent from `SINGLE_DATASET_EXPERIMENTS`, so `run_dataset.py` rejects it —
it takes `--datasets` and is run directly instead. Neither s09's docstring nor
`_summary_keys.py` names s07; the only mention left in the suite is
`similarity_helpers.py:399`, which calls it deleted. The NaN guards are present
and fired: the ratio column, which is exactly 1.00 on all three datasets,
recorded `wilcoxon_skipped` with scipy's own reason rather than a p-value.

---

## Fourteen defects found and committed

Each sat under a number that had been reported as a result.

| commit | what two things disagreed |
|---|---|
| `cb3ef49` | s11's Expected Improvement tied across the pool, so `np.argmax` picked in row order — 89 of 90 picks on herg_ki, 90 of 90 on adme_mdr1. Its one flag counted only fits that raised |
| `3b48515` | `random_50d` is a Gaussian projection with a real acquisition, and was on the list exempt from the degeneracy flags |
| `cb82362` | s04 and s12 used different cliff rules; 250 of 496 herg_ki cliff molecules agreed. s12's percentile sweep could never reach its own 10% target |
| `6506675` | the weighted GP's kernel width was divided away by sum-to-1 weights, so it returned the training mean |
| `318c80b` | the importance exponent was two different numbers under one name: 0.5 in s13, family-implied in `compute_distance_matrix` |
| `9c6f7d7` | the search cache had no column for the distance coefficient; one stored row served both, 0.6754 and 0.7327 under one number |
| `c5752d4` | `objective='similarity'` searches built a hand-written Euclidean whatever `--distance-metric` said, and four callers swallowed a stale-cache error into an empty pool |
| `37c3598` `686c337` `6c3c981` | the GP reported a score for a fit that had run out of room at its length-scale bound, and said nothing |
| — | `build_fold_hybrid` refuses a model with a geometry recipe; s04/s05/s06/s10/s11 passed one unconditionally and each caller's `except` swallowed the raise, so every similarity-family arm was written out as not computed |
| — | `MAIN_MODELS_WITH_KNN`, `DISTANCE_MODELS`, `RIDGE_MODELS`, `DOWNSTREAM_MODELS_NOISE` are built above the block that cuts the model list, so a cut list still named six models |
| — | the `rep_count` axis raised under a geometry objective, killing half the similarity grid |
| — | s03 asked the pool for models a cut search never scored |

---

## s06's comparator is ranked 8th, 10th and 13th by what s06 measures

`comparator_rep` is the representation a kNN predicts the LABEL best from, by
RMSE. What s06 reports is the Spearman correlation between a held-out molecule's
distance to its neighbours and a model's absolute residual — a different
quantity. Read from the three stored files, 2026-09-24, ranking all seven
representations and their weighted twins by the quantity s06 reports:

    caco2       pdv            +0.1447   rank 13 of 14    best is ecfp4_count_weighted +0.2162
    herg_ki     atom_pair      +0.2173   rank  8 of 14    best is pdv_weighted         +0.2951
    adme_mdr1   pdv            +0.2044   rank 10 of 14    best is atom_pair_weighted   +0.3494

Every representation is scored either way, so no stored number depends on this;
14 single-representation arms sit in each file. What the rule decides is which
arm the printed headline and the conformal `hybrid_genuinely_tighter` flag
compare the hybrid against — and on all three datasets that is an arm in the
bottom half of the ranking s06 itself produces.

The weighted twin is top on all three. Recommendation: rank the comparator by
the correlation s06 reports, which also makes both sides of the comparison
importance-weighted rather than one of each. Not changed — it is the owner's
call and it moves which arm the headline names on all three datasets.

## s06's sweep winners reproduce through the real experiment, 2 of 3

The 2,628-cell sweep ran in its own harness. These runs put its winning cell
back through s06 itself. Distance-error correlation, higher better:

    dataset     cell                       hybrid   hybrid_weighted
    caco2       shipped  manhattan k=5     +0.2423     +0.2540
                best     euclidean k=150   +0.3197     +0.3307
    herg_ki     shipped  manhattan k=5     +0.2773     +0.2793
                best     manhattan k=50    +0.2938     +0.2960
    adme_mdr1   shipped  manhattan k=5     +0.1892     +0.1970
                best     euclidean k=50    +0.1693     +0.1721

caco2 gains +0.0774 and herg_ki +0.0165. adme_mdr1 loses 0.0199, and that one is
a mistake of mine rather than the sweep's: the sweep chose euclidean at 15
neighbours for adme_mdr1 and the queued job named 5,15,25,50, so the reported
cell is euclidean at 50. Re-running it at 15 would test the sweep's actual
choice.

The weighted arm beats the unweighted one in all six rows.

Every winning cell uses 50 or 150 neighbours against the shipped 5. That is the
third independent place the pinned neighbour count was hiding an effect —
s06 here, the s05 sweep's k=7, and s10's weighting sign flipping between 10 and
15. What is new here is that it survives the round trip out of the sweep harness
and into the experiment.

## s10: importance weighting helps only at LARGE neighbour counts

Read from the two completed s10 herg_ki runs, 2026-09-24. rmse in pKi, so a
positive delta means weighting made it WORSE.

    run                       k      unweighted   weighted     delta
    corrected (0.25/dist)      5        0.6299     0.6341     +0.0041
                               7        0.6257     0.6267     +0.0010
                              10        0.6297     0.6340     +0.0043
                              15        0.6424     0.6413     -0.0011
                              25        0.6634     0.6594     -0.0041
    base (0.5/uniform)         5        0.6301     0.6393     +0.0092
                              10        0.6359     0.6489     +0.0130
                              20        0.6700     0.6653     -0.0047

The sign flips between 10 and 15 neighbours in the corrected run and between 10
and 20 in the base run. Both runs show the same shape: weighting hurts at small
neighbour counts and helps at large ones.

⛔ This was invisible while the neighbour axis was pinned at 5 and 10, because
both of those sit on the side where weighting hurts. The axis was unpinned on
2026-09-24.

**k=7 gives the best unweighted score of any count, 0.6257**, beating 5 and 10.
The s05 sweep finds the same thing independently — k=7 wins all four of its
completed caco2 cells. That count did not exist in this suite before 2026-09-24.

⛔ The two runs differ in THREE ways at once — exponent 0.5 against 0.25,
uniform against distance voting, and the neighbour list — so nothing here can be
attributed to the exponent. The within-run pattern across k is clean, because
inside one run only k varies.

## The s05 sweep, COMPLETE: 24 of 24 cells

Two voting schemes x four importance exponents x three datasets, with neighbour
counts 5/7/10/15/25 swept inside every cell. Best weighted hybrid arm, rmse.

**Distance voting wins all twelve matched pairs.** Distance minus uniform at
each exponent, negative means distance is better:

    dataset      exp 0.0   exp 0.25   exp 0.5   exp 1.0
    caco2        -0.0048    -0.0055   -0.0050   -0.0045
    herg_ki      -0.0082    -0.0091   -0.0063   -0.0070
    adme_mdr1    -0.0037    -0.0031   -0.0031   -0.0034

Twelve of twelve, and the size is stable within each dataset whatever the
exponent — so voting is an independent lever, not an interaction with weighting.
With s10's 20 of 21 (representation, dataset) pairs, this is the most solid
result in the suite.

**Best cell per dataset:**

    caco2       distance  exp 0.5   k=7   0.3231
    herg_ki     distance  exp 0.25  k=7   0.6082
    adme_mdr1   distance  exp 0.5   k=25  0.4952

Distance voting on all three. Exponent and neighbour count both
dataset-dependent, so neither has a value worth hardcoding.

⛔ The shipped configuration is uniform voting, exponent 0.5, neighbours 5 and
10. It gets the voting scheme wrong on all three datasets and the neighbour
count wrong on all three. The exponent it gets right on two.

## s05's headline number cannot see the lever s05 was tuned for

`summary.fixed_ref` — the arm this file leads with, and the one every
`kNN_delta_fixed_ref` figure in s09 comes from — has `condition:
hybrid_unweighted` on caco2. An unweighted arm multiplies no feature by its
importance, so `KIRBY_IMPORTANCE_EXPONENT` cannot move it.

Measured on the first two sweep cells, 2026-09-24, same dataset and same search:

    cell             fixed-cell delta     a weighted candidate (k=10)
    exponent 0            +0.0045                    0.335436
    exponent 0.25         +0.0045                    0.332860

The headline is identical to four decimals across the two exponents, and the
weighted candidate moves. Both are correct: the exponent is live, and the
headline is blind to it.

So the exponent's effect has to be read off the weighted candidates in
`fold_reselection.candidates_by_model`, not off `fixed_ref`. A sweep over the
exponent that only reports `fixed_ref` would show a flat line and conclude the
lever does nothing.

## Three axes closed, 2026-09-24

**minkowski p=0.5 does not beat manhattan.** It was cut from the s06 sweep on
cost; now measured on its own cache, two of three datasets in. Distance-error
correlation of the hybrid at the shipped cell, higher better:

    dataset     manhattan   minkowski p=0.5
    caco2         +0.2423        +0.2144
    herg_ki       +0.2773        +0.2774
    adme_mdr1     +0.1892        (running)

Worse on caco2 by 0.0279, identical on herg_ki to three decimals. At 25 to 50
times the cost of manhattan, nothing here argues for it. The axis is measured
rather than cut, which is what it needed.

**s13 retrieval: a hybrid space ranks first in all nine cells.** The count was
hardcoded at 5 until 2026-09-24 and had never been swept. Best space by the mean
activity gap to the k nearest molecules, lower better:

    dataset      k=5                   k=15                  k=50
    herg_ki      hybrid_similarity_r3  hybrid_similarity_r3  hybrid_similarity_r1
    caco2        hybrid_predictive_r1  hybrid_similarity_r1  hybrid_similarity_r1
    adme_mdr1    hybrid_similarity_r3  hybrid_similarity_r1  hybrid_similarity_r1

A hybrid is top in nine of nine, and a *similarity* hybrid in eight of nine —
caco2 at k=5 is the only cell a predictive hybrid takes. That is the clearest
support in this suite for building the similarity-objective hybrids separately
from the predictive ones.

⛔ The k=15 and k=50 runs rank 12 spaces where the stored k=5 runs rank 20. The
k=5 runs used the 12-configuration search and produced three recipe ranks per
family; the new ones use one configuration. So the candidate list is not the
same across k, and "the ranking is stable across k" — written here earlier — is
not supported. What holds is the within-k statement: at each neighbour count,
the best space among those present is a hybrid.

The gap rises with k because more neighbours are less similar, so the numbers
are not comparable across rows either.

**s10 at its own measured-best settings, herg_ki.** Distance voting, exponent
0.25, neighbour counts 5 to 25. The hybrid's delta: GP -0.2938, SVR +0.0010.
caco2 and adme_mdr1 still running.

## The standalone: two of three datasets were lost to one uninstallable rep

`similarity_optimized_hybrid` is the script nothing had ever run before this
campaign. Its first full attempt finished caco2 and FAILED on herg_ki and
adme_mdr1 with "No representations could be generated!" — each writing a 76 or
78 byte error file that the supervisor counted as a completed job.

The cause was not in the standalone. `_cache_rep_dict` hands
`get_molecular_representations` only the MISSING subset while the reps already
loaded from cache sit in `reps`. herg_ki and adme_mdr1 had 13 and 14
representations cached, needed only `grover`, could not build it (GROVER repo
not installed) and the builder raised on an empty subset. caco2 needed nothing,
so it survived. Two datasets lost to a representation none of them use.

Fixed 2026-09-25: the subset boundary now follows the drop-and-warn contract the
function three lines below already claims — warn and continue when something was
cached, raise unchanged when nothing was. Both re-runs fired the warning once
and are 52 minutes into real work, where they previously died within minutes.

⛔ The supervisor's completion test was "the output file exists", which is how a
76-byte error object passed as success. It now reads the file: an error key, a
non-ok `result_status`, or anything under 500 bytes marks the job FAILED instead
of DONE. An audit of all 83 outputs at the time found these two were the only
false completions.

## The supervisor gated on load; memory was the thing that mattered

Three ceilings were tried — 60, 150, 260 — and each deferred every queued sweep
cell for an hour or more against a machine that was two-thirds idle. Load on
this laptop runs 130 to 450 because of other sessions, and these jobs pin
lightgbm and torch to one thread each, so the figure is mostly other people's
work.

What has actually killed jobs here is memory: six were OOM-killed at three
concurrent when swap was exhausted on 2026-09-21. Load has never killed one.

Fixed 2026-09-25: the memory-pressure override is applied LAST, so it clears the
load block as well as the swap block. Load is kept only as a tiebreak for when
memory is also tight, and an absolute swap floor of 150MB still stands because
below it a launch is an OOM whatever the pressure figure says.

Effect: 9 forced, 9 launched, 14 jobs running at load 437 with 71 per cent of
memory free. The 33 sweep cells had been stalled since they were queued.

## Open decisions for the owner

Three of the four recorded here have been closed. What is left is below.

1. **Cliff share.** `CLIFF_TARGET_PCT = 15.0`. The only one that is a judgement
   rather than a defect. It is the smallest share where every held-out fold on
   all three datasets carries at least 30 cliff molecules. At 10% herg_ki's
   thinnest fold is 19 and caco2's 23. At 20% adme_mdr1's gap falls to 0.21x the
   median absolute deviation. Even at 15% adme_mdr1 needs a gap of 0.50x MAD,
   which is a weak cliff — it has 2,318 scaffolds among 2,641 molecules.

2. ✅ **CLOSED — s02's headline comparator was already weighted.** Checked in
   the code 2026-09-25: `sac_delta_comparator` takes `best_rep_name_weighted`
   when importance is available and falls back to unweighted only when it is
   not, recording which in `kind`. All three stored results read
   `kind=weighted`. This was listed as open by reading this section rather than
   the code.

3. ✅ **CLOSED — `baselines/` and `hybrid_evals/` are keyed by coefficient.**
   Both filenames now carry `__dist-<coefficient>`, omitted for the default so
   every existing file keeps hitting. The separate-`KIRBY_CACHE_DIR`-per-
   coefficient workaround is no longer needed.

4. ✅ **CLOSED — s11 hunts the end of the label the assay says is good.**
   `LABEL_BETTER_WHEN` now records it per dataset: herg_ki low, caco2 HIGH,
   adme_mdr1 low. `KIRBY_S11_HUNT` overrides per run; an unlisted dataset falls
   back to the old metric-direction behaviour and says so in the log.

   ⛔ This FLIPS caco2. Every stored s11 caco2 result was searching for the
   LEAST permeable molecules, because RMSE is lower-is-better. herg_ki and
   adme_mdr1 were coincidentally right — both assays want the low end.

   ⛔ The three directions are read from the assays and are recorded nowhere
   else in the repository. They need the owner's confirmation.

## Two things that change how the suite must be run

- ⛔ SUPERSEDED, checked 2026-09-23. Every result file on disk records
  `manhattan`: twelve experiments on three datasets each, `distance_metric` and
  `input_fingerprint.distance_metric` agreeing in all of them, and no file
  disagreeing with another. The euclidean runs this line described were
  overwritten by the 18–19 September re-runs, which took the default. Do NOT
  pass `--distance-metric euclidean` to reproduce what is on disk now; it would
  produce different numbers from the stored ones.
- Manhattan is the BETTER coefficient for that objective on caco2, beating
  euclidean on all seven representations by 0.012 to 0.051 Spearman. The code
  was using the worse one by accident.

## The stored search results were unreadable and were moved aside, 2026-09-21

`distance_metric` was added to the cache's identity columns during this
campaign. That column list is what `result_cache.py` checks on open. Every
database written before the change is missing the column, so opening one raises
`StaleResultDB` and the job dies at the first hybrid search.

All 14 databases under `/Volumes/seagate/kirby_simrun_cache/result_db/` were in
that state. Between them they held 530 stored search configurations: herg_ki
168, caco2 168, adme_mdr1 150, and 4 each in eleven per-seed files. Two
supervised runs, `s05_tuned_adme_mdr1` and `s05_tuned_herg_ki`, died on it.

The files are moved, not deleted, to
`/Volumes/seagate/kirby_simrun_cache/result_db_stale_2026-09-21/`. Opening one
still raises — the column list is what is wrong with them, and `result_cache.py`
says three times there is no migration path by design.

What this costs from here: every search configuration is computed from scratch.
A run that finished in minutes off stored rows will not.

What it means for the s05 numbers below: the 2026-09-19 run read those stored
configurations and any run after 01:55 on 2026-09-21 does not. The two runs
being compared in the section above were not computed from the same rows, and
that has to be ruled out before the commit window between `9c6f7d7` and
`c5752d4` is blamed.

## The held-out z-score defect does not reach this campaign's numbers

`pdv` is one of the seven representations every run here uses, and `create_pdv`
used to end by centring and scaling each descriptor over every molecule handed
in — held-out ones included, so a held-out molecule's own mean and spread sat
inside the training features. Cached matrices written before that was removed
still carry it, and there are two cache layers for the same matrix, of which the
content-keyed one is read first and has no identity file.

Checked directly, 2026-09-21, by loading all six `pdv.npy` files under
`/Volumes/seagate/kirby_simrun_cache` and counting how many non-constant
features sit at mean 0 and standard deviation 1. A whole-dataset z-score leaves
every one of them there. None of them are:

    layer, dataset          molecules  features  at mean 0  at std 1
    rep_cache/herg_ki           1412     200      0 of 179   0 of 179
    rep_cache/caco2             2158     200      0 of 165   0 of 165
    rep_cache/adme_mdr1         2641     200      0 of 179   0 of 179
    reps/fb871584b0a9de3a       1412     200      0 of 179   0 of 179
    reps/8b2a3b48894e07d4       2158     200      0 of 165   0 of 165
    reps/a02150e79f9353a1       2641     200      0 of 179   0 of 179

Both layers hold raw descriptors and both agree per dataset. The remaining 21
features on herg_ki and adme_mdr1, and 35 on caco2, are constant across every
molecule and carry no spread to test.

This says nothing about the caches on the server, which is where the defect was
found; it says the three datasets in this campaign are not affected by it.

## Compute flags for the cluster

- s10's Gaussian process task is the longest block in the wave. It builds the GP
  with `n_restarts_optimizer=2`, tripling every fit for a hyperparameter neither
  arm reports; both arms get it, so it cancels out of the comparison. Now
  settable by `KIRBY_S10_GP_RESTARTS`.
- Under a geometry objective the general and general_distance searches compute
  the same numbers — the evaluator and model lists are ignored. Warming both is
  a second full grid per dataset per augmentation condition for nothing.
- A minkowski p=0.5 cell costs 25 to 50 times a manhattan one: scipy's cdist is
  39 s for 1,412 by 1,412 rows at 2,048 features under manhattan and over 7
  minutes under minkowski.
- s11's MC-dropout BNN condition killed three processes at 14 GB of 15 GB swap
  and recorded nothing when it did.

---

# `similarity_metrics/` — the group nothing in this campaign had opened

Opened 2026-09-20. This group does not share anything with s01-s13 except the
idea. Four things about it are different and all four change how it is read.

**It has its own molecules.** Every script here calls `_data.load_prepared`,
which reads `sim_<name>.npz` sitting next to the scripts. Three such files
exist: herg_ki 1,415 molecules whole, logd 1,427 molecules kept from 5,039 by
whole Bemis-Murcko scaffold groups, adme_solubility 2,173 molecules whole.
caco2 and adme_mdr1 have no npz in this directory, so nothing here has ever run
on them. Building one means running `build_local_datasets.py` against a cached
CSV, which rebuilds ten representations and a full molecule-by-molecule ECFP4
Tanimoto matrix from SMILES.

**It never touches `/Volumes/seagate/kirby_simrun_cache/rep_cache/`.** The ten
representations are stored inside each npz. Nothing here calls
`find_best_hybrid`, nothing opens a `result_db`, so the StaleResultDB hazard
does not reach this group.

**It already does the thing s10 found helps.** `simeval.eval_knn_readacross`
builds `KNeighborsRegressor(n_neighbors=k, weights='distance',
metric='precomputed')` at simeval.py:161. Distance-weighted neighbour voting is
the default here, not an unused option.

**Its distance coefficients are not the process-wide ones.** `simeval` offers
Tanimoto, Dice, cosine, Soergel, Tversky and Euclidean and computes them by
hand. `src/kirby/similarity/distance_config.py` and
`KIRBY_IMPORTANCE_EXPONENT` are not read anywhere in this directory.

## Survey — every file in the directory

One row is one file. "Runs" means it was imported in this session and its data
dependency checked; all 31 files on disk import without error. Cost is wall
clock measured in this session where a time is given, and is not an estimate
where it says measured.

| file | asks | kind | runs today | cost on 1,415-2,173 molecules |
|---|---|---|---|---|
| `simeval.py` | nine ways to score one distance matrix against measured activity | library, no main | yes, 16 scripts call `eval_knn_readacross` | n/a |
| `rankers.py` | score a feature by how its pair-difference tracks the activity gap or a structural distance | library, no main | yes, imported by creative_hybrids, creative_hybrids_v2, feature_selectors/selector_prediction | n/a |
| `ranker_experiment.py` | does similarity-aware feature selection beat the count fingerprint under its own Tanimoto | METHOD | **no — the file is not on disk**, deleted in `4ab874c` | — |
| `blacklist.py` | which features inside one representation make its geometry worse | METHOD | **no — the file is not on disk**, deleted in `df7c7fe` | — |
| `scaffold_experiment.py` | does describing an antibody variant by what changed beat describing its whole sequence | METHOD | needs a FLAb library through `kirby.datasets.flab`, not a molecule dataset | not timed |
| `scaffold_ddg_experiment.py` | same question on single-point mutants where mutation count cannot vary | METHOD | needs the graphinity ddG set | not timed |
| `scaffold_feature_isolation.py` | which of five property scales carry the agreement with BLOSUM62 | investigation | needs three FLAb libraries | not timed |
| `scaffold_reps.py` | is blosum_matrix a distance | library, no main | yes | n/a |
| `feature_type_study.py` | do 14 property scales each earn their place in the distance | METHOD | **library only — no `if __name__` block.** Its one caller is `tests/experiments/opig_scaffold_similarity.py` | — |
| `feature_ablation.py` | which of four feature groups help, and does any combination beat ECFP4-count Tanimoto | METHOD | yes | measured, 1,309 s herg_ki for both arms |
| `base_rep_comparison.py` | does describing the edit beat describing the whole sequence at neighbour prediction | METHOD | needs four FLAb libraries | not timed |
| `aggregate.py` | does fusing six distance matrices beat the best single one | METHOD | yes | measured, 475 s herg_ki, about 50 min for three |
| `creative_hybrids.py` | does stacking descriptors, MACCS and fine 2D features onto ecfp4_count beat it | METHOD, superseded by v2 | yes | not timed |
| `creative_hybrids_v2.py` | same, on three datasets, with a per-dataset greedy descriptor drop | METHOD | yes | measured, 845 s, 854 s, 819 s |
| `winning_recipe_reselected.py` | does the stacked space still win once selection and scoring sit on disjoint molecules | METHOD | **the doc names a file that has never existed in this repository's history.** The file on disk is `winning_recipe_honest.py` | — |
| `winning_recipe_vs_standard.py` | does the stacked space beat two field-standard fingerprint metrics | METHOD, the headline table | yes | measured, 377 s, 371 s, 354 s |
| `structural_embeddings.py` | do MAP4, TDA or mol2vec beat ecfp4_count Tanimoto | METHOD | yes, via a direct `map4.map4.MAP4` import; the `src/kirby` wrapper raises | measured on herg_ki, MAP4 244 s, TDA 4,473 s, mol2vec 136 s |
| `structural_yardstick.py` | which representation-and-coefficient pair wins when the judge is shared substructure | investigation | yes | not timed |
| `shape_similarity.py` | do USR, USRCAT or a pharmacophore descriptor beat ecfp4_count Tanimoto | METHOD | yes, but its 19 July cache is refused | measured, 4,504 s for one of three descriptors on herg_ki; handed to the supervisor |
| `aug3d_similarity.py` | do E3FP or Coulomb eigenvalues beat ecfp4_count Tanimoto | METHOD | yes, `e3fp` imports, cache warm | measured, 39-47 s per dataset |
| `learned_embeddings.py` | cache one ChemBERTa and one MoLFormer vector per molecule | precompute plus one library function | `torch` and `transformers` both import; needs the model weights downloaded | not timed |
| `extra_fingerprints.py` | do Avalon, SECFP6 or MHFP6 beat count-ECFP4 Tanimoto | METHOD | yes | measured, Avalon 97-149 s, MAP4 184-347 s, MHFP6 570-1,097 s |
| `augmented_descriptor_reps.py` | does a descriptor block beat count-ECFP4 Tanimoto | METHOD | PDV and Mordred yes, `padelpy` absent | measured, seconds per dataset with the cache warm |
| `nary_similarity.py` | does complementary similarity rank activity-vs-neighbour disagreement better than two naive scores, and does reweighting the distance by it help | METHOD | yes | measured, 6.1 s herg_ki, 14.5 s logd |
| `triplet_similarity.py` | does a triplet-trained embedding beat count-ECFP4 Tanimoto on held-out folds | METHOD | yes, `torch` imports | measured, about 40 min per fold for three arms, five folds |
| `similarity_noise_cleaning.py` | does a distance chosen for structural fidelity find corrupted activity labels better than count-ECFP4 Tanimoto | METHOD | yes, but it alone reads logd uncapped at 5,039 molecules | measured, 681 s herg_ki, 1,059 s adme_solubility |
| `run_battery_moleculeace.py` | which representation and coefficient wins the full nine-statistic battery across MoleculeACE targets | METHOD | **no — the `MoleculeACE` package is not installed here, so there are no target CSVs and no npz files to score** | — |
| `mcs_cache.py` | maximum-common-substructure distance for a fixed set of 8,000 molecule pairs | precompute | yes, already cached for herg_ki, logd, adme_solubility | n/a |
| `_data.py` | which molecules, in which row order, every script indexes over | loader | yes | n/a |
| `_heldout.py` | split molecules in half by whole scaffold group | library, no main | yes | n/a |
| `_nanrank.py` | keep an uncomputable correlation out of a ranking and out of an average | library, no main | yes | n/a |
| `build_local_datasets.py` | rebuild herg_ki and logd npz from cached CSVs | build | yes | not timed |
| `build_adme_solubility.py` | build the log-solubility npz | build | **no — the file is not on disk**, deleted in `df7c7fe`. `sim_adme_solubility.npz` exists and cannot be rebuilt from this directory | — |
| `build_moleculeace.py` | turn each MoleculeACE CSV into an npz | build | **no — `MoleculeACE` is not installed**, and it finds the CSVs by globbing inside that installed package | — |

## What the survey found before anything was run

**Four files EXPERIMENTS_FINALIZED.md documents are not on disk.**
`ranker_experiment.py`, `blacklist.py` and `build_adme_solubility.py` were each
deleted in a named commit. `winning_recipe_reselected.py` is different: `git log
--all` over that path returns nothing, so no commit in this repository has ever
carried a file by that name. The file the entry describes is
`winning_recipe_honest.py`.

**A result file outlives its script.** `ranker_experiment.json` and
`ranker_experiment_quick.json` sit in the directory, written 3 and 4 September.
The script that wrote them was deleted in `4ab874c`. Nothing on disk can
reproduce or update those two files.

**Two of the seven method experiments named in the brief cannot be run at all.**
`ranker_experiment` has no file. `sm_battery_moleculeace` has a file but no
data: `MoleculeACE` is not an installed package here, and `build_moleculeace.py`
locates its CSVs by globbing inside that package's install directory.

## Method experiments, run at full molecule count

Every number below is this session's run on whole datasets: herg_ki 1,415
molecules, logd 1,427 kept from 5,039 by whole scaffold groups,
adme_solubility 2,173. No rows were subsampled. The activity yardstick
throughout is `simeval.eval_knn_readacross`: five folds by Bemis-Murcko
scaffold group, each held-out molecule predicted from its five nearest
training molecules weighted by one over distance, one Spearman pooled over
every held-out molecule. Higher is better, unitless. The structural yardstick
is the Spearman between a pair's distance in the space and its
maximum-common-substructure distance, over 8,000 cached molecule pairs. Higher
is better, unitless.

### `winning_recipe_vs_standard.py` — the headline table. Wins structure 3 of 3, wins activity 1 of 3.

This is the only script in the group that fits its feature selection on one half
of the molecules and scores on the other. The molecules split by whole scaffold
group: herg_ki 709 select and 706 score, logd 738 and 689, adme_solubility by
the same rule. A cached pair counts only when both its molecules land on the
score half, so herg_ki scores structure on 2,005 of its 8,000 pairs and drops
4,025 that straddle.

The stacked space is ecfp4_count, the RDKit descriptors that survive a drop-one
selection, 100 atompair/torsion features and a 49-feature 3D pharmacophore
block, each block z-scored, divided by the square root of its own feature count
and concatenated, then Euclidean.

Activity Spearman on the score half, five nearest neighbours:

    space                            herg_ki    logd   adme_solubility
    ecfp4:tanimoto (field default)    0.5781  0.5339   0.2895
    ecfp4_count:tanimoto (baseline)   0.5802  0.5811   0.3439
    ecfp4_count, z-scored Euclidean   0.5789  0.4629   0.1786
    the stacked space                 0.6415  0.4315   0.3366

Structural Spearman on the score half's pairs:

    space                            herg_ki    logd   adme_solubility
    ecfp4:tanimoto (field default)    0.1796  0.4081   0.2483
    ecfp4_count:tanimoto (baseline)   0.1079  0.4170   0.2752
    ecfp4_count, z-scored Euclidean   0.2863  0.1857   0.3011
    the stacked space                 0.4389  0.5851   0.3823

**The script's third question has an answer and it is not the same on the three
datasets.** It asks whether a gain is the added blocks or the switch from
Tanimoto to z-scored Euclidean. On herg_ki the switch costs 0.5802 to 0.5789,
so the whole 0.5802 to 0.6415 is the added blocks. On logd the switch costs
0.5811 to 0.4629 and the stacked space ends at 0.4315, so the coefficient
change is most of the loss and the added blocks make it slightly worse still.
On adme_solubility the switch costs 0.3439 to 0.1786 and the stacked space
recovers to 0.3366, so the added blocks are worth 0.1580 Spearman there and the
coefficient is what sank it.

Wall clock: 377 s, 371 s, 354 s per dataset, on a machine at load average 210
where the process held 8% of one core.

### `creative_hybrids_v2.py` — the same stacking, scored on every molecule, and it disagrees with the held-out table on logd

This one fits the descriptor drop and the fine-2D selection on the same 8,000
pairs it then scores structure on, and scores activity on all n molecules.
Activity Spearman, delta against ecfp4_count Tanimoto in the same column:

    space                                     herg_ki           logd
    count_tanimoto (baseline)                  0.7087          0.6180
    count, z-scored Euclidean                  0.6641 -0.0446  0.5329 -0.0851
    count+descriptors (all 208)                0.7100 +0.0014  0.6293 +0.0113
    count+descriptors (drop-one survivors)     0.6901 -0.0185  0.5866 -0.0314
    count+3D pharmacophore                     0.7317 +0.0230  0.3515 -0.2665
    count+survivors+pharmacophore+fine2D       0.7333 +0.0247  0.5259 -0.0920

The drop-one selection keeps 63 of herg_ki's 208 descriptors and 68 of logd's.
It is chosen to raise the structural Spearman and it does: herg_ki 0.248 to
0.492, logd 0.383 to 0.675 on the pairs it fitted on. On activity it costs
0.0185 on herg_ki and 0.0314 on logd.

**The 3D pharmacophore block is the whole story and it points opposite ways.**
Adding it alone to the count fingerprint is the second-best space on herg_ki at
+0.0230 Spearman and the worst space on logd at -0.2665. Nothing else in the
table moves either dataset by more than 0.05.

### `nary_similarity.py` — the shipped setting is worse than turning the method off, on all three

The method multiplies every pair's distance by `1 + lam * mean outlierness of
the pair`, where outlierness is complementary similarity min-max scaled over
every molecule. At lam = 0 the matrix is the plain count-fingerprint Tanimoto,
so lam = 0 is the method switched off. The file ships lam = 1.0.

Activity Spearman at five neighbours, three datasets:

    lam                 herg_ki    logd   adme_solubility
    0 (method off)       0.7087  0.6180   0.3995
    1.0 (shipped)        0.7041  0.6137   0.3728
    8.0                  0.6863  0.6009   0.3242

Sweeping lam over -0.9, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 4, 8 and the neighbour
count over 3, 5, 7, 15, the best cell is lam = -0.50 at 3 neighbours on herg_ki
(0.7132 against 0.7027 at lam = 0 and the same neighbour count), lam = -0.90 at
7 neighbours on logd (0.6195 against 0.6181), and lam = 0 at 15 neighbours on
adme_solubility (0.4325, no lam improves it). **Where lam helps at all its sign
is negative, which shrinks distances toward outliers — the opposite of what the
method is for.** The whole lam axis moves the Spearman by less than 0.011 on
every dataset. The neighbour count moves adme_solubility from 0.3995 at five
neighbours to 0.4325 at fifteen, three times the largest lam effect.

The file's other test ranks molecules whose activity disagrees with their
neighbours. Spearman against that target: complementary similarity 0.1265,
0.0897, -0.2855 on the three datasets; mean distance to all molecules 0.1389,
0.0945, -0.2676; distance of the activity from the dataset median 0.0665,
0.2588, 0.5086. The method is beaten by mean distance on all three and by
activity extremeness on two.

Wall clock: 6.1 s herg_ki, 14.5 s logd, 14.7 s adme_solubility for one setting.
The whole 40-cell sweep took 173 s on adme_solubility.

### `aug3d_similarity.py` — E3FP and Coulomb eigenvalues lose to the count fingerprint on all three

Activity Spearman at five neighbours, against ecfp4_count Tanimoto in the first
row:

    metric                  herg_ki    logd   adme_solubility
    ecfp4_count:tanimoto      0.709   0.618   0.399
    e3fp:tanimoto             0.654   0.548   0.331
    e3fp:cosine               0.651   0.548   0.307
    e3fp:euclidean            0.101   0.315   0.044
    coulomb:cosine            0.556   0.556   0.083
    coulomb:euclidean         0.532   0.430   0.231

E3FP under z-scored Euclidean is the same collapse s11 described: 1,024 bits
whose median Euclidean distance sits far above the scale the neighbour search
expects, and the activity Spearman falls to 0.101 on herg_ki and 0.044 on
adme_solubility while the same bits under Tanimoto reach 0.654 and 0.331.

Wall clock: 39 s per dataset with the conformer cache warm.

### `similarity_noise_cleaning.py` — the structure-selected space does not find corrupted labels better than the count fingerprint

Corrupted labels are injected at three noise widths and three contaminated
shares, three injection seeds each. Detection is an AUROC: does a corrupted
molecule disagree with its neighbours in this space? Recovery is a correlation
between the correction the neighbours apply and the error that was injected.
Both on the held-out half of the molecules, with every fitted step fitted on the
other half.

    herg_ki, scored on 706 held-out molecules and 2,005 of its 8,000 pairs
      space                                      detect AUROC   recovery corr
      ecfp4_count Tanimoto (baseline)                  0.6490          0.7860
      count+descriptor survivors (coarse x0.5)         0.6491          0.7935
      count+survivors+pharmacophore+fine2D             0.6422          0.8002

    adme_solubility, scored on 1,087 held-out molecules and 2,022 pairs
      space                                      detect AUROC   recovery corr
      ecfp4_count Tanimoto (baseline)                  0.6640          0.8043
      count+descriptor survivors (equal)               0.6753          0.7966
      count+survivors+pharmacophore+fine2D             0.6673          0.8103

On herg_ki the structure-selected space loses detection by 0.0068 AUROC and
wins recovery by 0.0142. On adme_solubility it wins detection by 0.0033 AUROC
and recovery by 0.0061. Every one of those four numbers is smaller than the gap
between the two spaces inside the same column, so nothing here separates.

**The drop-one descriptor selection picks a different set on each half of the
molecules.** On herg_ki it keeps 55 of 208 RDKit descriptors when fitted on one
half and 64 of 208 when fitted on the other. On adme_solubility, 54 and 57.
Both halves are the same dataset and the same procedure.

Wall clock: 681 s herg_ki, 1,059 s adme_solubility.

**logd was not run for this one and it is the only script in the group that
would not have read the same molecules.** `similarity_noise_cleaning.run` calls
`load_prepared(dataset, cap=None)` at line 320, so logd arrives as all 5,039
molecules, split 2,541 fit and 2,498 held-out. Every other script in the
directory takes the default cap and gets 1,427. Both cite a user decision dated
2026-09-11 in their own comments. Its distance matrices are 5,039 by 5,039, and
it was stopped at 658 MB resident with the machine at 14.4 GB of 15.4 GB swap.

### `triplet_similarity.py` — the two arms of one method train on triplet counts 155 times apart

Learned distance on herg_ki, five folds by scaffold group, each fold's encoder
trained on its own training scaffolds and scored on the held-out ones. Activity
Spearman at five neighbours, held out:

    fold   count fingerprint   structural triplets   activity triplets
    0                 0.7977                0.7493              0.8410
    1                 0.6821                0.6720              0.7678
    2                 0.6718                0.6347              0.7085
    3                 0.7094                0.6872              0.7568
    4                 0.6445                0.5401              0.6890
    mean              0.7011                0.6567              0.7526

**The activity-defined arm beats the count fingerprint on all five held-out
folds and it is the only method in this group that wins everywhere it was
measured.** Its per-fold gains are 0.0433, 0.0857, 0.0367, 0.0474 and 0.0445
Spearman. The scaffold-defined arm loses on all five, by 0.0484, 0.0101, 0.0371,
0.0222 and 0.1044.

On the structural yardstick the order reverses in part: the scaffold-defined arm
beats the count fingerprint on four of five folds, the activity-defined arm on
three of five, and every one of those structural numbers rests on between 288
and 318 held-out pairs rather than 8,000.

The counts underneath those columns are the finding. A triplet survives when
the anchor's positive is closer than its negative by less than the margin, so a
pass with no surviving triplets teaches the encoder nothing. Per pass, out of
20,000 candidate triples drawn:

    arm                   surviving triplets per pass (min / median / max)
    structural positive             106 / 127 / 154     (fold 0)
    activity positive         19,730 / 19,759 / 19,788  (fold 0)

The structural arm defines a positive as a molecule sharing the anchor's
Bemis-Murcko scaffold. The activity arm defines it as the molecule with the
smaller absolute activity gap. The first finds a usable triple in 127 of 20,000
draws, the second in 19,759 of 20,000. Over 40 passes the structural arm trains
on 5,101 triples and the activity arm on 790,333. So the two arms differ in
what defines a positive and in how many examples the encoder sees, two ways at
once, and the file reports them as one comparison.

Wall clock: about 40 minutes per fold for all three arms, on a machine at load
average 300. All five folds completed.

## The conformer cache costs four hours once, and it is not a defect

`shape_similarity.py` prints `spatial cache has 1415 rows but records no
molecule list -- rebuilding`. `sim_herg_ki_spatial.npz` is on disk, dated 19
July, with the right number of rows. It carries no `smiles_id`, and
`build_spatial_reps` at shape_similarity.py:104 refuses a cache whose molecule
list cannot be checked rather than trusting the row count. That is the right
call and the comment in the file says so. Once the rebuild finishes it writes
`smiles_id` and every later run is a cache hit.

The cost of that one rebuild, measured: the 12-feature USR descriptor on
herg_ki's 1,415 molecules took 5,155 s, and USR is one of three descriptors,
each of which generates its own conformers. So herg_ki alone is roughly four
hours at load average 500, and the run was handed to the supervisor at
`/Volumes/seagate/kirby_simrun_logs/supervisor_jobs.tsv` under the name
`sm_shape_similarity` rather than abandoned.

`aug3d_similarity.py` prints `3d-aug cache hit` on all three datasets and
finishes a dataset in 39 to 47 s. Its cache was written after the molecule-list
field existed, so it is trusted.

## What is broken, in one place

| what | what is wrong |
|---|---|
| `ranker_experiment.py` | not on disk, deleted in `4ab874c`. Its two result files are still in the directory and nothing can regenerate them |
| `blacklist.py` | not on disk, deleted in `df7c7fe`. Its procedure now lives in `src/kirby/similarity/selection.select_saboteur_drop`, which `winning_recipe_honest.py` calls |
| `build_adme_solubility.py` | not on disk, deleted in `df7c7fe`. `sim_adme_solubility.npz` still works, but it cannot be rebuilt from this directory |
| `winning_recipe_reselected.py` | EXPERIMENTS_FINALIZED.md documents it. `git log --all` over that path returns nothing. The file it describes is `winning_recipe_honest.py` |
| `run_battery_moleculeace.py` and `build_moleculeace.py` | the `MoleculeACE` package is not installed here. The builder locates its CSVs by globbing inside that package's install directory, so there is nothing to build from and nothing to score |
| `feature_type_study.py` | no `if __name__` block, so it cannot be run. Its only caller is `tests/experiments/opig_scaffold_similarity.py`, outside this directory |
| `augmented_descriptor_reps.py` | `padelpy` is absent, so its PaDEL arm cannot be built. PDV and Mordred both build |
| `shape_similarity.py` | not broken. Its 19 July conformer cache predates the molecule-list field, so it is rebuilt once at roughly four hours per dataset; see above |

Nothing in this group was fixed by editing it. Every script that runs today ran
unmodified.

### `extra_fingerprints.py` — none of Avalon, MAP4, SECFP6 or MHFP6 beats the count fingerprint on activity, and three of them beat it on structure

Activity Spearman at five neighbours, five folds by scaffold group:

    metric                      herg_ki    logd
    ecfp4_count:tanimoto         0.7087  0.6180
    avalon:tanimoto              0.6786  0.5721
    avalon:cosine                0.6733  0.5729
    map4:tanimoto                0.5049  0.5497
    map4:cosine                  0.4042  0.5426
    secfp6:tanimoto              0.6940  0.6030
    secfp6:cosine                0.6945  0.6025
    mhfp6:minhash_jaccard        0.6872  0.5906

Structural Spearman over the same 8,000 cached pairs:

    metric                      herg_ki    logd
    ecfp4_count:tanimoto         0.0431  0.3661
    avalon:tanimoto             -0.0340  0.2216
    map4:tanimoto               -0.4079 -0.0507
    secfp6:tanimoto              0.1542  0.5280
    mhfp6:minhash_jaccard        0.1911  0.5256

**The two yardsticks disagree and they disagree the same way on both datasets.**
SECFP6 and MHFP6 beat the count fingerprint on structure by 0.11 to 0.16
Spearman and lose to it on activity by 0.014 to 0.027. Nothing here beats it on
activity at all.

MAP4 is the outlier in both directions. Its structural Spearman is -0.4079 on
herg_ki, so its pairwise distance runs backwards against shared substructure,
and its activity Spearman falls to 0.4042 under cosine against the count
fingerprint's 0.7087. MAP4 is a 1,024-slot MinHash folded to bits with a mean
of 514 bits set per molecule on herg_ki, against SECFP6's 75.

Wall clock per dataset: Avalon 97-149 s, MAP4 184-347 s, MHFP6 570-932 s.

### `augmented_descriptor_reps.py` — a 200-descriptor block under cosine beats the 1,024-bit count fingerprint on two of three

Activity Spearman at five neighbours, delta against ecfp4_count Tanimoto in the
same column:

    metric             herg_ki            logd          adme_solubility
    ecfp4_count:tan     0.7087          0.6180          0.3995
    pdv:cosine          0.6998 -0.0088  0.6881 +0.0701  0.4045 +0.0050
    pdv:euclidean       0.7028 -0.0059  0.6652 +0.0472  0.3814 -0.0181
    mordred:cosine      0.7042 -0.0044  0.6457 +0.0277  0.4210 +0.0215
    mordred:euclidean   0.7085 -0.0002  0.6458 +0.0278  0.3715 -0.0280

pdv is 200 RDKit descriptors, Mordred is 1,613. Structural Spearman: pdv
Euclidean beats the count fingerprint on herg_ki, 0.2559 against 0.0431, and
loses on adme_solubility, 0.2552 against 0.3006.

**Cosine beats Euclidean on the same features on five of the six
dataset-and-block pairs where the two differ by more than 0.01 Spearman.** On
logd pdv reaches 0.6881 under cosine and 0.6652 under Euclidean; on
adme_solubility Mordred reaches 0.4210 under cosine and 0.3715 under Euclidean.
The coefficient is doing as much work as the choice of descriptor block.

`padelpy` is absent, so the PaDEL arm printed `INFEASIBLE` on all three and
nothing was scored for it.

Wall clock: seconds per dataset with `sim_<ds>_augmented.npz` warm.

## One thing that applies to most of the group

**The per-feature z-score is fitted on every molecule, then the folds are cut.**
`augmented_descriptor_reps.py` calls `zscore(Xraw)` at line 214 over all n rows,
builds the distance matrix from that, and hands the matrix to
`eval_knn_readacross`, which only then cuts five folds by scaffold group. The
same order holds in `creative_hybrids.py`, `creative_hybrids_v2.py`,
`aug3d_similarity.py`, `shape_similarity.py` and `structural_embeddings.py`,
which all call `zscore()` on the whole matrix, and in `aggregate.py`, which
writes the same thing inline at line 61 as `X.mean(0), X.std(0)`. Seven scripts,
not nine. So each fold's held-out molecules contributed to the mean and standard
deviation that scaled the features they are scored on.

`extra_fingerprints.py` and `nary_similarity.py` were named here and do not
belong: both build tanimoto and cosine distances on bit fingerprints and
standardise nothing, so there is no scale for a held-out molecule to influence.
Checked 2026-09-21 by grepping every scaling call in the directory.

`feature_ablation.py` is on both sides and says which is which. Its held-out arm
calls `zscore_fit(m, fit_rows)` at line 176, taking the mean and standard
deviation from the fit rows only; its in-sample arm calls plain `zscore(m)` at
line 174. The headline numbers above are from the held-out arm.

No labels are involved — the z-score reads features only — so this is a scaling
fitted on molecules the fold does not see, not a label leak.

Three files do it the other way and fit the scale on one half of the molecules:
`winning_recipe_vs_standard.py`, `winning_recipe_honest.py` and
`similarity_noise_cleaning.py`. Those three also report lower numbers for the
same spaces on the same datasets, which is what their smaller reference pool
would do on its own, so the two effects are not separated by anything measured
here.

### `structural_embeddings.py` — `create_map4` in `src/` names a class the installed package does not have

Its feasibility block prints, before any molecule is scored:

    create_map4 KIRBy wrapper: FAIL -> ImportError: MAP4 requires the map4 package: pip install map4
    map4 (direct build)      : OK
    create_tda               : OK
    create_mol2vec           : OK

`create_map4` at `src/kirby/representations/molecular.py:3915` does `from map4
import MAP4Calculator`. The installed `map4` exposes `MAP4` at the top level and
`MAP4` inside `map4.map4`. It has no `MAP4Calculator`. So the wrapper's error
message says the package is missing when the package is installed and only the
class name is wrong. `structural_embeddings.py` already works around it by
importing `map4.map4.MAP4` directly, and its own header says so.

This was not fixed here. `create_map4` lives in `src/kirby`, outside this
directory, and other sweeps are running against that file.

MAP4 is the worst metric measured in this whole group, so making the wrapper
work would not change a result: `extra_fingerprints.py` scores map4:cosine at
0.4042 activity Spearman on herg_ki against the count fingerprint's 0.7087, and
its structural Spearman is -0.4079, running backwards against shared
substructure.

### `aggregate.py` — fusing six distance matrices wins activity on one dataset by 0.013 and loses structure on both, and four of its six fusion rules produce an asymmetric distance

It builds six distance matrices — ECFP4 counts under Tanimoto, atom-pair counts
under cosine, 208 z-scored RDKit descriptors under Euclidean, MACCS under
Tanimoto, torsions under Tanimoto, the RDKit fingerprint under Tanimoto — and
fuses them six ways.

Activity Spearman at five neighbours:

    metric or fusion       herg_ki    logd   adme_solubility
    ecfp4_count:tanimoto     0.709   0.618   0.399
    atompair_count:cosine    0.716   0.615   0.399
    desc:euclidean           0.703   0.666   0.381
    maccs:tanimoto           0.651   0.637   0.329
    torsion:tanimoto         0.690   0.577   0.373
    rdkit:tanimoto           0.573   0.553   0.293
    A1 min-max score mean    0.721   0.636   0.395
    A2 mean neighbour rank   0.721   0.649   0.378
    A3 reciprocal rank       0.729   0.664   0.415
    A4 Borda points          0.721   0.649   0.378
    A5 top-20 consensus      0.721   0.644   0.403
    A6 drop-duplicates mean  0.721   0.636   0.395

**Reciprocal rank fusion beats the best single metric on two of three.** On
herg_ki it reaches 0.729 against atom-pair counts under cosine at 0.716. On
adme_solubility 0.415 against the count fingerprint at 0.399. On logd it loses,
0.664 against the descriptor block under Euclidean at 0.666. It is the best
fusion rule on all three, and the other five never beat the best single metric
on more than one dataset.

On structure it loses on all three. Best single metric 0.248 on herg_ki (the
descriptor block) against 0.065 for the best fusion; 0.462 on logd (torsions
under Tanimoto) against 0.407; 0.308 on adme_solubility (torsions again)
against 0.207.

On finding corrupted labels, best single against best fusion: 0.668 detection
AUROC against 0.669 on herg_ki, 0.649 against 0.637 on logd, 0.668 against
0.665 on adme_solubility. On correcting them: 0.837 against 0.835, 0.775
against 0.773, 0.828 against 0.820. Fusion never wins either by more than
0.001.

**Four of the six fusion rules produce a matrix where the distance from
molecule A to molecule B is not the distance from B to A.** The file prints a
`sym` column and marks A2, A3, A4 and A5 as `NO`; every single metric and the
two score-averaging fusions are `yes`. The scorer they are handed,
`eval_knn_readacross`, fits `KNeighborsRegressor(metric='precomputed')` on the
training submatrix, and `struct_fidelity` reads one triangle. So the two
yardsticks are reading two different geometries for those four rules, and the
one fusion that beat the best single metric on herg_ki, reciprocal rank fusion,
is one of the four.

Wall clock: 475 s herg_ki, about 50 minutes for all three.

## The stale result databases do not reach this group

No script in `similarity_metrics/` opens a result database or calls
`find_best_hybrid`. Grepping the directory for `result_db`, `find_best_hybrid`
and `StaleResultDB` returns six files, and in every one the hit is a comment
saying the file does not use them, except `KIRBY_CACHE_DIR` in
`build_moleculeace.py`, `learned_embeddings.py` and `run_battery_moleculeace.py`,
where it names `$KIRBY_CACHE_DIR/similarity_metrics/sim_data` and nothing else.

So none of the numbers in this section came from a stored hybrid-search row, and
moving the 14 files out of `result_db/` changes nothing here.

## MODI's nearest neighbour is picked by row order when molecules tie, and it reaches nothing that runs here

`eval_modi_binary` at simeval.py:183 takes `np.argmin(Dm, axis=1)`. When several
molecules sit at the same smallest distance, `argmin` returns the lowest row
number, so the neighbour is chosen by the order of the npz file.

**It is reached by one caller.** `eval_modi_binary` is called only from
`simeval.run_battery` at line 489, and `run_battery` is called only from
`run_battery_moleculeace.py:119`. That is the script `MoleculeACE` is not
installed for. So no number anywhere in this section passes through it, and the
tie matters only when the MoleculeACE battery runs on the cluster.

Counted on this group's own molecules, under Tanimoto, over all ten stored
representations. The first count is molecules whose smallest distance is shared
by more than one other molecule. The second is how many of those have tied
candidates in both activity classes, so that the MODI term for that molecule
changes with which tied neighbour is picked:

    representation     herg_ki (of 1,415)    logd (of 1,427)   adme_solubility (of 2,173)
                        tied / ambiguous     tied / ambiguous      tied / ambiguous
    maccs                  387 / 88             365 / 96             105 / 34
    fcfp4                  210 / 40             204 / 24              58 / 19
    torsion                126 / 20             197 / 37              38 / 19
    ecfp4                  127 / 28             175 / 22              26 / 16
    ecfp4_count             89 / 11             146 / 17               6 / 4
    rdkit                   86 / 2              111 / 4                2 / 0
    desc                    55 / 9              106 / 4                0 / 0
    atompair_count          43 / 1               92 / 4                0 / 0

The largest single tie is 10 molecules, on herg_ki under MACCS.

**The share that actually moves is smaller than the tie count.** On herg_ki
under MACCS, 387 molecules tie but only 88 have tied candidates in more than one
activity class; the other 299 get the same MODI term whichever neighbour is
picked. So 88 of 1,415 molecules, 6.2 in 100, carry a MODI term that depends on
row order. On logd under MACCS it is 96 of 1,427. On adme_solubility under
atom-pair counts and under the descriptor block, nothing ties at all.

**A decision for the owner, not taken here.** The alternative is to average the
class agreement over every tied neighbour rather than take one. It changes no
number produced in this campaign, because nothing that runs locally calls MODI.
It would change the MODI column of the MoleculeACE battery for at most 6.2 in
100 molecules on the representation where ties are worst, and for none on the
two representations where nothing ties. `simeval.py` was not edited.

### `feature_ablation.py` — four of twelve block combinations beat the count fingerprint when the selection is held out, and none of them do when it is not

It builds 1,024 ECFP4 counts, 208 RDKit descriptors, 167 MACCS keys and 100
features chosen from atom-pair and torsion counts, and scores twelve
combinations. On herg_ki it splits 709 fit molecules and 706 score molecules by
whole scaffold group, and of 8,000 cached pairs keeps the 2,005 with both ends
in the score half.

Activity Spearman, herg_ki, both arms, the count fingerprint's own value first:

    blocks                        features   held out   same molecules
    ecfp4_count:tanimoto              1024      0.580            0.709
    descriptors+MACCS keys             375      0.618            0.692
    descriptors alone                  208      0.598            0.703
    descriptors+keys+fine 2D           475      0.595            0.686
    count+fine 2D                     1124      0.591            0.671
    count alone, z-Euclidean          1024      0.579            0.674
    count+descriptors+keys+fine2D     1499      0.564            0.683
    count+MACCS keys                  1191      0.558            0.674

**The 375-feature descriptor-and-keys space beats the 1,024-feature count
fingerprint by 0.038 Spearman when the feature selection and the z-score are
fitted on molecules it is not scored on. On the arm that fits and scores on the
same 1,415 molecules, nothing in the table beats it and that same space loses by
0.016.** The two arms rank the twelve combinations differently and the file
prints both.

logd gives the same winner. Held out on 689 of its 1,427 molecules, against
the count fingerprint's 0.581:

    blocks                        features   held out
    descriptors+MACCS keys             375      0.615
    descriptors+keys+fine 2D           475      0.613
    descriptors alone                  208      0.606
    ecfp4_count:tanimoto              1024      0.581
    count+descriptors+keys            1399      0.563
    count+MACCS keys                  1191      0.508

adme_solubility, held out on 1,087 of its 2,173 molecules, against the count
fingerprint's 0.344:

    blocks                        features   held out
    descriptors alone                  208      0.383
    descriptors+fine 2D                308      0.370
    descriptors+keys+fine 2D           475      0.370
    ecfp4_count:tanimoto              1024      0.344
    count+descriptors                 1232      0.250
    count+fine 2D                     1124      0.248

**A physicochemical descriptor block beats the 1,024-bit count fingerprint on
all three datasets, held out, and it is a fifth to a quarter of its width.**
With MACCS keys beside it, 375 features, it reaches 0.618 against 0.580 on
herg_ki and 0.615 against 0.581 on logd. Alone, 208 features, it reaches 0.383
against 0.344 on adme_solubility.

Averaged over the combinations that contain a block against those that do not,
held out:

    block            herg_ki   logd   adme_solubility      (activity Spearman)
    count             +0.007  -0.061      -0.096
    descriptors       +0.029  +0.062      +0.065
    MACCS keys        +0.005  +0.033        —
    100 fine 2D       -0.005  -0.008        —

**The descriptor block raises the activity Spearman on all three datasets and
the count fingerprint block lowers it on two of three.** Putting the count
fingerprint into a space costs 0.096 Spearman on average on adme_solubility and
0.061 on logd; on herg_ki it is worth +0.007. Its structural effect on logd is
-0.249 and on adme_solubility +0.043, so it does not point one way there either.

The 100 fine 2D features are the reverse of the descriptors: on herg_ki and logd
they raise structural Spearman by 0.103 and 0.161 and lower activity Spearman by
0.005 and 0.008.

**The drop-one descriptor selection's gain shrinks when it is scored on pairs it
did not choose on.** On herg_ki it drops 153 of 208 descriptors and the
structural Spearman goes 0.290 to 0.419 on held-out pairs, against 0.483 on the
pairs the sequence was chosen on. On logd it drops 138 and goes 0.454 to 0.628
held out, against 0.676 in sample.

Wall clock: 1,309 s herg_ki, 1,111 s logd, 1,329 s adme_solubility, both arms each.

### `structural_embeddings.py` — mol2vec ties the count fingerprint, TDA and MAP4 lose heavily

herg_ki, activity Spearman at five neighbours and structural Spearman over 8,000
pairs:

    metric                 activity   structure
    ecfp4_count:tanimoto     0.7087      0.0431
    mol2vec:euclidean        0.7098      0.0227
    mol2vec:cosine           0.6957      0.1611
    tda:euclidean            0.4951      0.0863
    tda:cosine               0.4848      0.0951
    map4:tanimoto            0.5049     -0.4079
    map4:cosine              0.4042     -0.4244

mol2vec's 300 features under Euclidean reach 0.7098 against the 1,024-bit count
fingerprint's 0.7087, which is a tie, not a win. TDA's 28 features lose 0.21
Spearman. Three of the six beat the count fingerprint on structure, and mol2vec
is the only one that still does once molecule size is partialled out.

Wall clock on herg_ki: MAP4 244 s, TDA 4,473 s, mol2vec 136 s. TDA is the
expensive one by a factor of eighteen and it is the one that loses most.

logd and adme_solubility for this script and for `feature_ablation.py` were
still running when this was written.

## `similarity_metrics/` — where each method experiment now stands

One row is one method experiment. "Beats the count fingerprint" is the
five-nearest-neighbour activity Spearman against `ecfp4_count` under Tanimoto,
the baseline every script in the group uses.

| experiment | datasets run | beats the count fingerprint on activity | optimised |
|---|---|---|---|
| `triplet_similarity` (activity positives) | herg_ki, 5 folds | **5 of 5 folds**, mean 0.7526 against 0.7011 | the two triplet definitions were compared; margin and epochs untouched |
| `triplet_similarity` (scaffold positives) | herg_ki, 5 folds | 0 of 5 folds | same |
| `aggregate` (reciprocal rank fusion) | all 3 | 2 of 3, +0.013 and +0.016, -0.002 on logd | all six fusion rules compared |
| `feature_ablation` | all 3 | **3 of 3.** A descriptor block beats it by +0.038, +0.034, +0.039 held out, at a fifth to a quarter of its width | all twelve combinations, both arms |
| `winning_recipe_vs_standard` | all 3 | 1 of 3, +0.0613 on herg_ki | no lever swept; the coefficient was isolated |
| `augmented_descriptor_reps` | all 3 | pdv cosine 2 of 3, Mordred cosine 2 of 3 | cosine against Euclidean on each block |
| `creative_hybrids_v2` | all 3 | 2 of 3 for the full stack | two block weightings |
| `structural_embeddings` | herg_ki | mol2vec Euclidean ties at +0.0011; TDA and MAP4 lose | two coefficients per representation |
| `nary_similarity` | all 3 | 0 of 3 at the shipped setting | **yes** — lam swept over 10 values and k over 4; lam = 0 is best or tied on all three |
| `similarity_noise_cleaning` | herg_ki, adme_solubility | not its question; detection AUROC moves by at most 0.011 | top-3 by two rankings |
| `extra_fingerprints` | all 3 | 0 of 3, for all four fingerprints | two coefficients each |
| `aug3d_similarity` | all 3 | 0 of 3, for both representations | three coefficients each |
| `shape_similarity` | none yet | — | handed to the supervisor as `sm_shape_similarity` |
| `run_battery_moleculeace` | none | — | cannot run, `MoleculeACE` not installed |
| `ranker_experiment` | none | — | file not on disk |

**The count fingerprint under Tanimoto is hard to beat and two things beat it
everywhere they were measured.** A physicochemical descriptor block, on all
three datasets, held out, at a fifth to a quarter of the fingerprint's width. A
learned distance whose positives are the molecules with the closest activity, on
every one of five held-out folds of herg_ki. Reciprocal rank fusion of six
metrics beats it on two of three.

**The descriptor block is the one to take forward.** It needs no training, no
conformers and no package that is missing here, it is 208 features against
1,024, and `feature_ablation.py` measures its average effect as +0.029, +0.062
and +0.065 activity Spearman across the twelve combinations on the three
datasets. The count fingerprint's own average effect is +0.007, -0.061 and
-0.096.

**Nothing that adds 3D information beats it on activity.** E3FP, Coulomb
eigenvalues and TDA all lose by 0.07 to 0.36 Spearman. The 3D pharmacophore
block inside `creative_hybrids_v2` is the second-best space on herg_ki at +0.0230
and the worst on logd at -0.2665.

**The two yardsticks disagree on most of the group.** SECFP6 and MHFP6 beat the
count fingerprint on agreement with shared substructure by 0.10 to 0.16 Spearman
on all three datasets and lose to it on activity on all three. The stacked
recipe beats it on structure on all three and on activity on one.

## Still running when this was written

Three runs were in flight and will finish on their own. None of them changes a
conclusion above; each adds the datasets its experiment has not reached.

| run | where it had got to | writes |
|---|---|---|
| `structural_embeddings.py` | herg_ki done, logd on its TDA build | `structural_embeddings.json` |
| `feature_ablation.py` | all three datasets done, writing out | the scratchpad json passed to `--out` |
| `shape_similarity.py` | herg_ki, one of three descriptors built | under the supervisor as `sm_shape_similarity`, output `/Volumes/seagate/kirby_simrun_logs/sm_shape_similarity_all.json` |

## What this group has never been asked

The four antibody scripts — `scaffold_experiment.py`,
`scaffold_ddg_experiment.py`, `scaffold_feature_isolation.py` and
`base_rep_comparison.py` — load FLAb variant libraries and the graphinity ddG
set through `kirby.datasets`, not through `_data.load_prepared`. They index
protein variants, not molecules, and none of them was run here. Their result
files are dated 12 and 13 September, so they ran then.

`winning_recipe_honest.py` was not run. It asks the same held-out question as
`winning_recipe_vs_standard.py`, which was run on all three, and differs by
using `kirby.similarity.selection.select_saboteur_drop` for the descriptor drop
instead of the in-file greedy. Whether those two selections pick the same
descriptors is not measured.
