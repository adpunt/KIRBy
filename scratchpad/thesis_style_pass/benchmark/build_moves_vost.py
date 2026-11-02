import json, re, csv, glob, collections
D='/Users/apunt/repos/KIRBy/scratchpad/thesis_style_pass/benchmark/'
segs=[json.loads(l) for l in open(D+'segments_vost.jsonl')]
codes={}; read_num={}
for ln in open(D+'codes_vost.txt'):
    p=ln.split()
    if p and not p[0].startswith('#'):
        codes['VO'+p[0]]=(p[1], 'yes' if p[3]=='y' else 'no'); read_num['VO'+p[0]]=p[2]
segs=[s for s in segs if s['sentence_id'] in codes]
assert len(codes)==len(segs)==len(set(open(D+'sample_vost_ids.txt').read().split()))
NUMW=r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty|hundred|thousand|half|halves|third|thirds|quarter|quarters|tenth|tenths|dozen|first|second|fifth|seventh|tenth)\b'
def has_num(t):
    t=re.sub(r'\b(Figures?|Tables?|Sections?|Chapters?|Appendix|Equation)\s+[A-Z0-9]+(\.\d+)?[a-z]?(\s+and\s+[A-Z0-9]+(\.\d+)?[a-z]?)?','',t)
    t=re.sub(r'\(([^()]*?\b(19|20)\d\d[a-z]?;?\s*)+\)','',t)   # parenthetical citations
    t=re.sub(r'\b(19|20)\d\d\b(?=\))','',t)                     # Author (2025)
    t=re.sub(r'\b(PDBbind|CASF|OpenFF)[ -]?20\d\d|\b20\d\d General','',t)  # data set / force-field names
    t=re.sub(r'\b[0-9][A-Z0-9]{3}\b','',t)                         # PDB identifiers
    t=re.sub(r'\bAlphaFold ?\d|\bBoltz-2|\bQM9|\bCASF16|\bCore80|\bose80|SLE20|\bGFN2|\b3D\b|Tankyrase-2|\bsection \d+(\.\d+)*|\bFig\.? ?\d+\.\d+[a-z]?|\b2\.4[a-l]\b|\bTables? \d+\.\d+[ab]?(?: and \d+\.\d+[ab]?)?','',t)
    return 'yes' if re.search(r'\d',t) or re.search(NUMW,t,re.I) else 'no'
rows=[]
for s in segs:
    m,h=codes[s['sentence_id']]
    rows.append(dict(sentence_id=s['sentence_id'],paragraph_id=s['paragraph_id'],chapter=s['chapter'],section_type=s['section_type'],text80=s['text'][:80],move=m,contains_number=has_num(s['text']),hedged=h))
with open(D+'moves_vost.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
M=[f'M{i}' for i in range(1,12)]
NAMES=dict(M1='background/method restatement',M2='pointer to figure/table/section',M3='result with numbers',M4='result in words',M5='comparing results',M6='interpreting/explaining',M7='comparing with literature',M8='evaluating/claiming strength',M9='limitation or caveat',M10='transition/linking',M11='recommendation or implication')
out=[]
out.append('# Rhetorical moves, reference: Vost (2025) DPhil thesis\n')
out.append('Source: `~/Documents/ReferencePapers/Theses/text/Vost_2025_Machine_learning_for.txt` (pdftotext -layout of the PDF next to it). METHOD.md and a build-step segment file for "vost" were not in this folder when this was run (10 Oct, 03:48 to about 04:15), so `segment_vost.py` here does its own segmentation, in the same form as `segment_buttenschoen.py`. Sections taken: 2.5 Results and 2.6 Conclusions; 3.4 Results and discussion and 3.5 Conclusions; 4.4 Results and 4.6 Conclusions; 5.4 Results and 5.5 Conclusions; all of Chapter 6 (Conclusions and outlook). Section 4.5 Related work is left out. Chapter 3 has no separate discussion section: its 3.4 Results and discussion is counted as results, and the thesis has no section titled Discussion anywhere, so the discussion column is empty. Figure and table captions, table rows, running page headers and page numbers removed. In this text a paragraph starts at an indented line, not a blank line; paragraphs cut by a page break or a float rejoined.\n')
out.append('Sample: the segmentation gives 417 sentences in 110 paragraphs, above the 250 cap, so `sample_vost.py` (a copy of `sample_buttenschoen.py`) drew a stratified sample: strata = chapter x section type, each stratum gets its share of 250 in proportion to its size. The unit drawn is a whole paragraph, so the move sequence inside each kept paragraph is intact for the bigram and first-sentence counts; whole paragraphs make the total 258 sentences, not exactly 250. Sentence ids kept: `sample_vost_ids.txt`; seed 20261010.\n')
out.append('Moves coded by reading each sentence (scheme after Yang & Allison 2003 and Swales 1990), one move per sentence, in five batches (one per chapter), re-reading the scheme before each. Codes: `codes_vost.txt`. Decision rules used where two moves fit: a sentence that sets two named methods, conditions, datasets or a before/after pair side by side is M5 even when it carries numbers; M3 is a number reported for one condition. A sentence that describes what was run, even inside a results section, is M1. A citation used as motivation for the next experiment is M1, not M7; M7 is kept for setting a result of this thesis against a published one. `hedged` coded by reading (may, might, could, suggest, appear, seem, likely, possibly, perhaps, some evidence, approximately, around, somewhat, at least partially). `contains_number` is mechanical, the same rule as `build_moves_buttenschoen.py` (any digit or spelled number word, including ordinals such as first and second), after removing figure/table/section references, citation years, PDB identifiers and digits inside names (QM9, CASF16, Core80, Boltz-2, GFN2, 3D). A second reading-based number tag is in the third field of `codes_vost.txt`. One coder, one pass, no second-coder agreement measured.\n')
tot=collections.Counter(r['section_type'] for r in rows)
out.append(f'Sentences: {len(rows)} (results {tot["results"]}, discussion {tot["discussion"]}, conclusion {tot["conclusion"]}); paragraphs: {len(set(r["paragraph_id"] for r in rows))}.\n')
rn={st:[r for r in rows if r['section_type']==st] for st in ['results','conclusion']}
out.append('Reading-based number tag (a quantity or threshold stated in the sentence; ordinals and counts of methods no): '+'; '.join(f"{st} {100*sum(read_num[r['sentence_id']]=='y' for r in v)/len(v):.1f}% ({sum(read_num[r['sentence_id']]=='y' for r in v)} of {len(v)} sentences)" for st,v in rn.items())+'.\n')
out.append('## Move shares per section type\n\nOne row is one move; each cell is the percentage of that section type\'s sentences coded with it (sentence count in brackets).\n')
out.append('| Move | results | discussion | conclusion | all |\n|---|---|---|---|---|')
for m in M:
    cells=[]
    for st in ['results','discussion','conclusion',None]:
        sub=[r for r in rows if st is None or r['section_type']==st]
        k=sum(r['move']==m for r in sub); cells.append(f'{100*k/len(sub):.1f}% ({k})' if sub else 'none')
    out.append(f'| {m} {NAMES[m]} | '+' | '.join(cells)+' |')
out.append('\n## Number and hedge tags per section type\n\nPercentage of sentences tagged yes.\n')
out.append('| Section type | contains_number | hedged |\n|---|---|---|')
for st in ['results','discussion','conclusion']:
    sub=[r for r in rows if r['section_type']==st]
    if not sub: out.append(f'| {st} | none | none |'); continue
    out.append(f'| {st} | {100*sum(r["contains_number"]=="yes" for r in sub)/len(sub):.1f}% | {100*sum(r["hedged"]=="yes" for r in sub)/len(sub):.1f}% |')
out.append('\n## Results move shares by chapter\n\nPercentage of each chapter\'s results sentences (n in header).\n')
chs=sorted(set(r['chapter'] for r in rows if r['section_type']=='results'))
hdr='| Move | '+' | '.join(f'Ch{c} (n={sum(1 for r in rows if r["section_type"]=="results" and r["chapter"]==c)})' for c in chs)+' |'
out.append(hdr); out.append('|---'*(len(chs)+1)+'|')
for m in M:
    cells=[]
    for c in chs:
        sub=[r for r in rows if r['section_type']=='results' and r['chapter']==c]
        cells.append(f'{100*sum(r["move"]==m for r in sub)/len(sub):.1f}%')
    out.append(f'| {m} | '+' | '.join(cells)+' |')
paras=collections.OrderedDict()
for r in rows: paras.setdefault(r['paragraph_id'],[]).append(r)
for st in ['results','discussion','conclusion','all']:
    bg=collections.Counter()
    for p in paras.values():
        if st!='all' and p[0]['section_type']!=st: continue
        for a,b in zip(p,p[1:]): bg[(a['move'],b['move'])]+=1
    n=sum(bg.values())
    if not n: continue
    out.append(f'\n## Commonest move bigrams within a paragraph, {st}\n\nOne bigram is a sentence\'s move followed by the next sentence\'s move in the same paragraph; {n} bigrams in all.\n')
    out.append('| Bigram | count | share |\n|---|---|---|')
    for (a,b),k in bg.most_common(12): out.append(f'| {a} -> {b} | {k} | {100*k/n:.1f}% |')
out.append('\n## First sentence of each paragraph\n\nShare of paragraphs whose first sentence is a finding in words or a comparison (M4/M5), against a method restatement or a result with numbers (M1/M3). Single-sentence paragraphs included.\n')
out.append('| Section type | paragraphs | M4/M5 first | M1/M3 first | M1 first | M3 first | other first |\n|---|---|---|---|---|---|---|')
for st in ['results','discussion','conclusion','all']:
    ps=[p for p in paras.values() if st=='all' or p[0]['section_type']==st]
    f=[p[0]['move'] for p in ps]; n=len(f)
    if not n: continue
    a=sum(x in('M4','M5') for x in f); b=sum(x in('M1','M3') for x in f)
    out.append(f'| {st} | {n} | {100*a/n:.1f}% ({a}) | {100*b/n:.1f}% ({b}) | {sum(x=="M1" for x in f)} | {sum(x=="M3" for x in f)} | {100*(n-a-b)/n:.1f}% ({n-a-b}) |')
ps=[p for p in paras.values() if p[0]['section_type']=='results']
fc=collections.Counter(p[0]['move'] for p in ps)
out.append('\nFirst-sentence moves of results paragraphs, all: '+', '.join(f'{m} {fc[m]}' for m in M if fc[m])+'.\n')
open(D+'moves_vost_summary.md','w').write('\n'.join(out)+'\n')
print('\n'.join(out))
