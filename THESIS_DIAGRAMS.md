# Diagrams for Chapter 5

Written 2026-10-08. Each section names one figure, says what it is for, and gives the boxes it should contain as plain text.
Read the boxes top to bottom or left to right. Each arrow means "goes into".

---

## Diagram 1: fix the existing search overview figure

**Where:** `figures_code_input/hybrid_search_space.png`, Figure `fig:hybrid_search_space`, at the start of "Methods: the hybrid search".
**TODO comments:**
`% same diagram with "Dataset" above the inputs. TODO: review -- pick one.`
`% TODO: the image still shows a Pruning box; no search in this research uses pruning`

**What the figure is for:** one picture of the whole search, from inputs to blueprints.

**Three things to change in the current image:**

1. **Delete the "Pruning" box.** No search in the thesis uses pruning.
2. **Add a "Representation count" box** next to the other blue boxes. The text says six settings are searched, but the image shows five.
3. **Rename "Target metric"** under the inputs to whatever term you choose for the metric. See item 1 in `THESIS_TODO_DISCUSSION.md`, which recommends "evaluation metric".

**The "pick one" TODO:** the other version, with "Dataset" above the inputs, is not in the thesis folder. Only `hybrid_search_space.png` is there. If you still have the other file, pick whichever one the caption describes. The caption describes inputs on the left, split into five folds.

**After the changes, the figure contains:**

```
LEFT                        MIDDLE                                   RIGHT

[Base representations]      [Score each representation alone]
[Labels]              ──►   [5 folds]  ──►  searched settings:   ──►  [Best hybrid for     ──►  [Blueprint]
[Evaluation metric]                           Importance method          each downstream
                                              Allocation method          model]
                                              Representation count
                                              Budget
                                              Internal evaluator
                                              Downstream model
```

---

## Diagram 2: new figure showing how one hybrid is built

**Where:** "From base representations to a hybrid" (Section `sec:method:build`).
**TODO comment:** `% TODO: Figure 5.1: the search pipeline schematic. Does not exist yet.`

**What the figure is for:** Diagram 1 shows the whole search. This one zooms in on what happens for one setting of the search, inside one fold.
It shows the steps the formulas in this section describe, using the numbers from the worked example in Table `tab:alloc_example`: a budget of 100 features, three representations and linear allocation.

**Use four rows, one per step:**

```
STEP 1: start with the base representations
   [Representation A: 2,048 features]   [Representation B: 200 features]   [Representation C: 384 features]

                                   │
                                   ▼
STEP 2: score each representation on its own
   (internal evaluator, rescaled so 0 = no better than a trivial predictor, 1 = perfect)
   [A: 0.6]                             [B: 0.5]                            [C: 0.2]

                                   │
                                   ▼
STEP 3: split the budget of 100 features by those scores (allocation method)
   [A gets 46 features]                 [B gets 39 features]                [C gets 15 features]

                                   │
                                   ▼
STEP 4: in each representation, rank features by importance and keep the top ones
   [top 46 of A]                        [top 39 of B]                       [top 15 of C]

                                   │
                                   ▼
RESULT: put them side by side
   [ Hybrid: 46 from A | 39 from B | 15 from C  =  100 features ]
```

**Keep these points in the figure:**
- Steps 2 and 3 decide **how many** features each representation gives. Step 4 decides **which ones**. Your text says this in one sentence, and the figure should make it visible.
- Use one colour per representation, kept the same in every row. The reader can then see in the last row which features came from where.
- Write the numbers inside the boxes. They match Table `tab:alloc_example`, so the reader can check one against the other.

**Leave out of this figure:** the global pool, augmentation, and the folds. The global pool works differently, because it ranks all representations' features together in step 4. Mention it in the caption only: "The global pool replaces steps 3 and 4 with one joint ranking."

---

## Optional diagram 3: the three levels of splitting

**Where:** "From base representations to a hybrid", the paragraph under `% TODO: what splits are used here?`.

**What the figure is for:** to show the outer folds, inner folds and internal split in one place. Item 19 of `THESIS_TODO_DISCUSSION.md` explains them. Only add this figure if you think readers will get lost; the paragraph can carry it alone.

**Draw it as nested boxes, each inside the one before:**

```
┌──────────────────────────── Whole dataset ─────────────────────────────┐
│                                                                        │
│  [Outer test fold]   used once, to score the finished hybrid           │
│                                                                        │
│  ┌─────────────────── Outer training folds ─────────────────────────┐  │
│  │                                                                  │  │
│  │  [Inner test fold]   scores each search setting                  │  │
│  │                                                                  │  │
│  │  ┌──────────────── Inner training folds ──────────────────────┐  │  │
│  │  │                                                            │  │  │
│  │  │  [80%] importance scores          [20%] representation     │  │  │
│  │  │                                         scores, which set  │  │  │
│  │  │                                         the budget split   │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

Write one sentence under it: "Every split keeps each group whole (scaffold, sequence cluster, tissue source site)."
