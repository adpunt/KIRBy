"""Re-measure after the benchmark edits (2026-10-10). Same measures as compute_metrics.py.
Corpora: author_current (thesis.tex now), author_before_edits (thesis.tex.before_benchmark_edits_2026-10-10).
Reference rows (buttenschoen, vost, dablander) are copied unchanged from archive_first_run/metrics_sentences.jsonl.
Moves: by sentence text from moves_author_current.csv; sentences new in this pass from moves_after_new_sentences.csv (by text;
first coded by id in codes_after.txt, then 12 sentences split during verification added).
Output: metrics_sentences_after.jsonl"""
import json, csv, sys
sys.path.insert(0, '/Users/apunt/repos/KIRBy/scratchpad/thesis_style_pass/benchmark')
from compute_metrics import nlp, normalise, sent_metrics, B
seg = {json.loads(l)['sentence_id']: json.loads(l)['text'] for l in open(B + 'segments_author_current.jsonl')}
bytext = {seg[r['sentence_id']]: r['move'] for r in csv.DictReader(open(B + 'moves_author_current.csv')) if r['sentence_id'] in seg}
newtext = {r['text']: r['move'] for r in csv.DictReader(open(B + 'moves_after_new_sentences.csv'))}
out = open(B + 'metrics_sentences_after.jsonl', 'w')
for name, f in (('author_current', 'full_segments_author_current.jsonl'), ('author_before_edits', 'full_segments_author_before_edits.jsonl')):
    R = [json.loads(l) for l in open(B + f)]
    texts = [normalise(r['text'], False) for r in R]
    for r, t, doc in zip(R, texts, nlp.pipe(texts, batch_size=256)):
        m = sent_metrics(doc, t)
        mv = bytext.get(r['text']) or (newtext.get(r['text']) if name == 'author_current' else None)
        rec = dict(corpus=name, sentence_id=r['sentence_id'], paragraph_id=r['paragraph_id'], chapter=r['chapter'],
                   section_type=r['section_type'], section=r.get('section'), subsection=r.get('subsection'),
                   tex_line=r.get('tex_line'), marker=r.get('marker'), move=mv, norm_text=t, **m)
        out.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(name, len(R), flush=True)
n = 0
for l in open(B + 'archive_first_run/metrics_sentences.jsonl'):
    if json.loads(l)['corpus'] in ('buttenschoen', 'vost', 'dablander'):
        out.write(l); n += 1
print('reference rows copied', n)
