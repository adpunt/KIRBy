"""Stratified sample of ~250 results/discussion/conclusion sentences from full_segments_dablander.jsonl.
Same rule as sample_vost.py: unit = whole paragraph; strata = chapter x section type; seed 20261010."""
import json, random, collections
R = [json.loads(l) for l in open('full_segments_dablander.jsonl')]
R = [r for r in R if r['section_type'] in ('results', 'discussion', 'conclusion')]
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
        if got + len(paras[p]) - target > target - got and got > 0: continue
        keep.add(p); got += len(paras[p])
    print(k, 'sentences', tot, 'target', target, 'kept', got)
with open('sample_dablander_ids.txt', 'w') as f:
    for r in R:
        if r['paragraph_id'] in keep: f.write(r['sentence_id'] + '\n')
print('total kept', sum(len(paras[p]) for p in keep), 'paragraphs', len(keep), 'of', len(R))
