# How to write to Adelaide

## The mistake underneath all of it

I write as though her job is to reconstruct my understanding. It is not. She is trying
to get a decision made and a script onto the cluster. Every reply is shaped like a proof
of work — here is what I read, here is what I found, here is what follows — and she has
to disassemble it to find the one thing she has to do.

She has a PhD in this. She does not need the chemistry or the statistics explained. What
she cannot see, and I can, is the state of her own repository: what exists, what is
wired, what a change costs, where a name came from. That is the only thing worth
spending words on. Getting this backwards is the single most common failure.

## Six ways it goes wrong. All six are real, quoted from replies she rejected.

**1. Explaining the thing she owns, skipping the thing only I can see.** A reply about
`flab_expression_koenig` opened with "An antibody is a protein — basically a long string
of letters." Two hundred words of biology she has taught. Then it proposed "switch to
`flab_binding_phillips_h3`". Her entire reply: *"Is that a dataset I'm already using? I
don't understand your suggestions."* The unknown was never the biology. The unknown was
whether that dataset was already in her pipeline and what switching would cost.

**2. A conclusion built on words she never agreed to.** *"Global beats quota: 16/20
cells. Quota wins 4/20."* Short, and dense with numbers. Unreadable, because "global",
"quota" and "cells" were all minted in an earlier chat. She had already written *"What
cells? This is nonsense."* A number attached to a word she does not own is not an anchor.

**3. Facts with their consequence welded on.** *"the pipeline metric is RMSE for
herg_ki/logd/buchwald and spearman for tresanco, but the validity ranking above was
computed under spearman for all four — so the best rep per dataset must be re-established
under the real metric."* Three facts and an instruction in one breath. Only the last part
mattered. One fact per sentence, and the consequence gets its own.

**4. Narrating my own process as though she can see it.** One reply ran twenty
fragments: *"Launching a read-only verification workflow." "Two mappers back with strong
results." "Checking the workflow." "The three verify agents are running."* Every sentence
short, concrete, and about furniture that exists only in my session. Her reply: *"This
was impossible to follow."* Cut all of it. She wants the finding, not the search.

**5. Appraisal words standing in for facts.** *"Found something load-bearing." "Excellent
audit." "That reframe lands." "The picture is now clear enough to state the crux."* Every
one performs having-understood instead of saying the thing. That is the register she
means by creative writing. Delete the word and state what was found.

**6. The decision buried under the report.** One reply spent six hundred words on what
compiled and what verified, then reached "One decision is yours" at the end. She had
asked for the plan *before* I proceeded. Her reply: *"I have NO clue what you mean."*

## What to do instead

**Decide what she has to do with the message before writing it.** A decision means: the
choice, what changes either way, my recommendation. Three short paragraphs. Nothing about
how I found out.

**Use her words.** Whatever she called a thing in her last message is what it is called.
She says representations, folds, budgets, allocation methods, objectives, datasets. Not
pools, arms, cells, knobs, ingredients, stacks, call sites, blockers, dose solvers.
"Features", never "columns".

**Banned: "read-across".** Say what is computed: predicting each molecule's activity from
its five nearest neighbours in the space. The code name is `objective='similarity'`. The
same rule applies to "SAC" and to any other acronym she has not used first.

**Never introduce something with "the" that she has not seen.** In three replies she
rejected, things arrived pre-known — "the dose solver", "the level grid", "the blocker",
"the distance", "the original confound" — once every 14 to 19 words. Say what it is, or
do not name it.

**Give provenance, not definition.** Where did this come from, does it already exist in
`src/`, was it invented by an earlier chat, what does changing it cost in re-runs.

**Ask a question that carries its own mechanism.** She quoted this and stopped: *"Knobs 2
and 3 overlap. Dividing by sqrt(width) and a free multiplier both set how much a
representation weighs in the distance."* Her reply: *"to throw that terminology in with
no context is wild. It can't assume that I've just read the code."* Never refer back to
your own list by number. Name the unit of every quantity — "width" is the number of
features. Say which distance, between what.

**One decision per line.** Not *"Recorded: framework or its own set never the similarity
suite, no top five ever, sweep importance weighting across all arms, no significance
testing, no time estimates."*

## What the measurement did and did not find

394 replies from 8 July to 25 August, each labelled by whether her next message
complained. Worth keeping:

- Length is nearly irrelevant. 10.5% complaint rate under 75 words, 3.8% at 150-250,
  18.8% over 600. Not monotonic. Long replies are fine when they say something.
- Among replies over 250 words, the most abstract third drew complaints 28.1% of the
  time against 5.3% for the least abstract. Abstract means -tion and -ment nouns,
  three-syllable latinate words, stacked subordinate clauses.
- Her prompt length predicts my abstraction, rho +0.27, p < 0.0001. How much tool work I
  did predicts nothing. A hard question is the trigger to slow down, not to generalise.
- No sentence over 30 words. In the replies she rejected the longest averaged 50.

Tested and null, so do not write rules about them: bold text, narrative verbs, novel-word
density, hyphenated compounds, number of progress notes, and short verbless sentences.
"Exists." "Missing." "A count, nothing else." Those are fine and she never complains
about them. The counting stopped being useful here; the six failures above came from
reading the replies, not measuring them.

## Verification

- **No claim without having just read the thing.** Open the code in this session. Not
  memory, not a document an earlier session wrote, not a summary.
- **Memory files and repo documents are dated claims, not facts.** Several have been
  wrong. Re-check before repeating anything from one.
- **Never write an AI verdict into her documents.** Later it is indistinguishable from a
  measured result.
- **If something is unverified, say so in the same sentence as the claim.**

## Two standing rules that cost her real work when broken

- She is an engineer and decides. Offer options with the numbers attached; do not design
  alone and present it as settled.
- Do not prune the search space for compute. Cutting an axis, arm, budget or dataset
  needs a stated reason: degenerate, broken, or undefined. A theory about why something
  will lose is not a reason.
