"""Stratified sample of ~250 sentences from segments_vost.jsonl (422 sentences).
Unit sampled = whole paragraph, so move sequences inside a paragraph stay intact.
Strata = chapter x section type; each stratum gets round(n_sentences * 250/366) sentences;
paragraphs drawn in random order (seed 20261010) until the stratum target is reached or passed."""
import json, random, collections
R = [json.loads(l) for l in open('segments_vost.jsonl')]
paras = collections.OrderedDict()
for r in R: paras.setdefault(r['paragraph_id'], []).append(r)
strata = collections.defaultdict(list)
for pid, ss in paras.items(): strata[(ss[0]['chapter'], ss[0]['section_type'])].append(pid)
frac = 250 / len(R); rng = random.Random(20261010); keep = set()
for k in sorted(strata):
    pids = strata[k][:]; tot = sum(len(paras[p]) for p in pids); target = round(tot * frac)
    rng.shuffle(pids); got = 0
    for p in pids:
        if got >= target: break
        if got + len(paras[p]) - target > target - got and got > 0: continue  # skip paragraph that overshoots more than it fills
        keep.add(p); got += len(paras[p])
    print(k, 'sentences', tot, 'target', target, 'kept', got)
with open('sample_vost_ids.txt', 'w') as f:
    for r in R:
        if r['paragraph_id'] in keep: f.write(r['sentence_id'] + '\n')
print('total kept', sum(len(paras[p]) for p in keep), 'paragraphs', len(keep))
