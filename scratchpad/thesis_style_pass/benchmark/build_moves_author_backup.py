"""Join hand-coded moves (ab_codes_0*.txt) to segments_author_backup.jsonl; write moves_author_backup.csv and the summary."""
import json, re, csv, glob, collections
D = '/Users/apunt/repos/KIRBy/scratchpad/thesis_style_pass/benchmark/'
segs = [json.loads(l) for l in open(D + 'segments_author_backup.jsonl')]
codes = {}
for f in sorted(glob.glob(D + 'ab_codes_0*.txt')):
    for ln in open(f):
        p = ln.split()
        if p:
            codes['AB' + p[0]] = (p[1], 'yes' if 'h' in p[2:] else 'no')
assert len(codes) == len(segs), (len(codes), len(segs))

# contains_number: a quantity is stated. Digits inside names (ECFP4, QM9, Caco-2, ESM-2, K_i, IC_50, 2D) do not count.
# Spelled counts and fractions count (two..twelve, twenty, hundred, thousand, half, third(s), quarter(s), tenth(s), dozen);
# "one" and ordinals do not, because they are mostly not quantities ("no one", "first fold").
NAME = re.compile(r'\b[\w-]*[A-Za-z][\w-]*\d[\w-]*\b|\b\d+D\b|\b\d+-[A-Z][\w-]*')
NUMW = re.compile(r'\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty|hundred|hundreds|thousand|thousands|half|halves|thirds?|quarters?|tenths?|dozen)\b', re.I)
PATIENT = re.compile(r'Cancer Genome Atlas|TCGA|ANNITIA|NHANES', re.I)
def has_num(t):
    t = t.replace('[REF]', '').replace('[CIT]', '')
    t2 = NAME.sub(' ', t)
    return 'yes' if re.search(r'\d', t2) or NUMW.search(t2) else 'no'

rows = []
for s in segs:
    m, h = codes[s['sentence_id']]
    txt = s['text'][:80]
    if PATIENT.search(s['text']):
        txt = '[text withheld: names a patient-data source]'
    rows.append(dict(sentence_id=s['sentence_id'], paragraph_id=s['paragraph_id'], chapter=s['chapter'],
                     section_type=s['section_type'], text80=txt, move=m,
                     contains_number=has_num(s['text']), hedged=h))
with open(D + 'moves_author_backup.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

M = [f'M{i}' for i in range(1, 12)]
NAMES = dict(M1='background or method restatement', M2='pointer to figure, table or section', M3='result with numbers',
             M4='result in words', M5='comparing results', M6='interpreting or explaining', M7='comparing with literature',
             M8='evaluating or claiming strength', M9='limitation or caveat', M10='transition or link',
             M11='recommendation or implication')
ST = ['results', 'discussion', 'conclusion']
out = []
A = out.append
A('# Rhetorical moves, author_backup (thesis.tex.before_style_apply_2026-10-10)\n')
A('Source: `/Users/apunt/repos/KIRBy/thesis.tex.before_style_apply_2026-10-10`, Chapters 3 to 8. '
  'METHOD.md and a build-step segment file for author_backup were not in this folder, so `segment_author_backup.py` '
  '(a copy of `segment_author_current.py` with the section rule changed) wrote `segments_author_backup.jsonl`.\n')
A('Section types. The backup has no single `Results` heading in Chapters 3 and 4, so every section there between '
  'Methods/Introduction and Conclusions is results. Chapters 5 to 7: sections titled `Results: ...` are results; '
  '`Discussion` and `Limitations` are discussion; `Conclusion(s)` is conclusion. Chapter 8: `Conclusions` is conclusion, '
  'every other section is discussion. The author_current file used a different rule (its Chapters 5 and 7 have no '
  'separate Limitations or Discussion sections), so section-type counts between the two files are not like for like.\n')
A('Coding. One move per sentence, scheme after Yang & Allison (2003) and Swales (1990), coded by reading each sentence, '
  'in seven batches of 180 to 230 sentences. One coder, one pass, no second-coder agreement measured. '
  '`hedged` = yes when the writer withholds full commitment (may, likely, probably, seem, partly as an explanation, '
  '"in general", "broadly", "has the potential"); approximators on a measured number ("about 0.54", "almost every dataset") '
  'were not counted. `contains_number` is mechanical: a numeral outside a name (ECFP4, QM9, Caco-2, K_i do not count) '
  'or a spelled count or fraction (two to twelve, twenty, hundred, thousand, half, third, quarter, tenth); "one" and '
  'ordinals do not count.\n')
A('Non-prose lines kept as sentences: five figure or table file paths that the segmenter read as text '
  '(AB0510, AB0610, AB0645, AB0752, AB0776; coded M2), three `[TABLE NEEDED]` lines (AB0890, AB0923, AB1138; M2) and '
  'eight sentences from six `[RESULT NEEDED]` placeholders (AB0543-0544, AB0571-0572, AB0708, AB1029, AB1328, AB1375; coded M9). One sentence\'s text is withheld in the CSV because it names a patient-data source; its move is kept.\n')
tot = collections.Counter(r['section_type'] for r in rows)
paras = collections.OrderedDict()
for r in rows: paras.setdefault(r['paragraph_id'], []).append(r)
pt = collections.Counter(p[0]['section_type'] for p in paras.values())
A(f'Sentences: {len(rows)} (results {tot["results"]}, discussion {tot["discussion"]}, conclusion {tot["conclusion"]}). '
  f'Paragraphs: {len(paras)} (results {pt["results"]}, discussion {pt["discussion"]}, conclusion {pt["conclusion"]}).\n')

A('## Move shares per section type\n\nOne row is one move. Each cell is the percentage of that section type\'s sentences coded with it, sentence count in brackets.\n')
A('| Move | results | discussion | conclusion | all |\n|---|---|---|---|---|')
for m in M:
    cells = []
    for st in ST + [None]:
        sub = [r for r in rows if st is None or r['section_type'] == st]
        k = sum(r['move'] == m for r in sub); cells.append(f'{100*k/len(sub):.1f}% ({k})')
    A(f'| {m} {NAMES[m]} | ' + ' | '.join(cells) + ' |')

A('\n## Number and hedge tags per section type\n\nPercentage of that section type\'s sentences tagged yes.\n')
A('| Section type | sentences | contains_number | hedged |\n|---|---|---|---|')
for st in ST:
    sub = [r for r in rows if r['section_type'] == st]
    A(f'| {st} | {len(sub)} | {100*sum(r["contains_number"]=="yes" for r in sub)/len(sub):.1f}% | {100*sum(r["hedged"]=="yes" for r in sub)/len(sub):.1f}% |')

A('\n## Results move shares by chapter\n\nPercentage of each chapter\'s results sentences; n in the header is that chapter\'s results sentence count.\n')
chs = sorted(set(r['chapter'] for r in rows if r['section_type'] == 'results'))
A('| Move | ' + ' | '.join(f'Ch{c} (n={sum(1 for r in rows if r["section_type"]=="results" and r["chapter"]==c)})' for c in chs) + ' |')
A('|---' * (len(chs) + 1) + '|')
for m in M:
    cells = []
    for c in chs:
        sub = [r for r in rows if r['section_type'] == 'results' and r['chapter'] == c]
        cells.append(f'{100*sum(r["move"]==m for r in sub)/len(sub):.1f}%')
    A(f'| {m} | ' + ' | '.join(cells) + ' |')

for st in ST + ['all']:
    bg = collections.Counter()
    for p in paras.values():
        if st != 'all' and p[0]['section_type'] != st: continue
        for a, b in zip(p, p[1:]): bg[(a['move'], b['move'])] += 1
    n = sum(bg.values())
    A(f'\n## Commonest move bigrams within a paragraph, {st}\n\nA bigram is one sentence\'s move followed by the next sentence\'s move in the same paragraph. {n} bigrams in all; top 12 shown.\n')
    A('| Bigram | count | share of bigrams |\n|---|---|---|')
    for (a, b), k in bg.most_common(12):
        A(f'| {a} → {b} | {k} | {100*k/n:.1f}% |')

A('\n## First sentence of each paragraph\n\nOne row is one section type. "Finding first" = the first sentence is M4 or M5. '
  '"Method or number first" = M1 or M3. Paragraphs with one sentence are counted in the first block and left out of the second.\n')
for label, minlen in [('all paragraphs', 1), ('paragraphs of two or more sentences', 2)]:
    A(f'\n**{label}**\n')
    A('| Section type | paragraphs | finding first (M4/M5) | method or number first (M1/M3) | M1 alone | M3 alone | other first moves |\n|---|---|---|---|---|---|---|')
    for st in ST + ['all']:
        ps = [p for p in paras.values() if (st == 'all' or p[0]['section_type'] == st) and len(p) >= minlen]
        f = collections.Counter(p[0]['move'] for p in ps); n = len(ps)
        ff = f['M4'] + f['M5']; mf = f['M1'] + f['M3']
        oth = ', '.join(f'{m} {f[m]}' for m in M if m not in ('M1', 'M3', 'M4', 'M5') and f[m])
        A(f'| {st} | {n} | {100*ff/n:.1f}% ({ff}) | {100*mf/n:.1f}% ({mf}) | {f["M1"]} | {f["M3"]} | {oth} |')
open(D + 'moves_author_backup_summary.md', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
