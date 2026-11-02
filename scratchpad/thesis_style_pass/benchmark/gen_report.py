"""Write BENCHMARK_REPORT.md from metrics_section_types.csv, metrics_chapters.csv, worklist_all_paragraphs.csv, report_parts.json."""
import csv, json, math, collections
B = '/Users/apunt/repos/KIRBy/scratchpad/thesis_style_pass/benchmark/'
TYPES = ['introduction', 'background', 'methods', 'results', 'discussion', 'conclusion']
SHORT = dict(sent_len='words per sentence', dc='subordinate clauses per sentence', nom='nominalizations per 100 words',
             num='numbers per 100 words', num3='% sentences with 3+ numbers', lexd='lexical density %', hedge='hedges per 100 words',
             boost='boosters per 100 words', ai='AI-guide phrases per 100 words', passive='% passive sentences',
             self='we/our/I per 100 words', para_len='words per paragraph', m3share='% M3 among M2+M3+M4')
MOVE_NAMES = dict(M1='method or background restated', M2='pointer to a figure or table', M3='result given as numbers',
                  M4='result in words', M5='comparison of results', M6='interpretation', M7='comparison with literature',
                  M8='claim of strength', M9='limitation', M10='link or transition', M11='recommendation or implication')
for m, n in MOVE_NAMES.items(): SHORT[m] = f'% {m} ({n})'
MOVEKEYS = set(MOVE_NAMES) | {'m3share'}

def fl(x):
    try:
        v = float(x); return None if math.isnan(v) else v
    except (TypeError, ValueError): return None
def f(x, d=1):
    v = fl(x); return '-' if v is None else f'{v:.{d}f}'
def arrow(flag):
    return {'OUTSIDE high': 'OUTSIDE, above', 'OUTSIDE low': 'OUTSIDE, below'}.get(flag, '')

SEC = list(csv.DictReader(open(B + 'metrics_section_types.csv')))
CH = list(csv.DictReader(open(B + 'metrics_chapters.csv')))
W = list(csv.DictReader(open(B + 'worklist_all_paragraphs.csv')))
parts = json.load(open(B + 'report_parts.json'))
sizes = parts['sizes']
o = []
A = o.append

A('# Linguistic benchmark of the thesis against three reference theses\n')
A('Written 2026-10-10. No thesis file was edited. Every number here comes from the scripts and CSV files in this folder.\n')
A('`METHOD.md` and the metrics CSVs named in the task were not in this folder. The measuring was therefore done here, from scratch: '
  '`segment_full_author.py`, `segment_full_refs.py`, `compute_metrics.py`, `build_report_tables.py`, `gen_report.py`.\n')
A('## 0. What was compared, and how\n')
A('**Texts.** Your thesis now (`thesis.tex`, read 10 Oct about 04:15, last saved 04:06) and the backup from before the style pass '
  '(`thesis.tex.before_style_apply_2026-10-10`), Chapters 1 to 8. '
  'Three reference DPhil theses from the same group: Buttenschoen (2025), Vost (2025) and Dablander (2023), read from their pdftotext files. '
  'Figures, tables, captions, equations, references lists and page furniture are removed. Citations and cross-reference numbers are removed before counting, '
  'so "Figure 3.7" or "[6, 7]" never count as numbers.\n')
A('**Section types.** Every paragraph gets one of six types from the heading it sits under: introduction (Chapter 1 and each chapter\'s Introduction), '
  'background (your Chapter 2 review sections; Buttenschoen 1.1-1.3; Vost 1.1-1.4 and 4.5; Dablander Chapter 2), methods, results '
  '(including "Results and discussion" sections), discussion, conclusion. Your Chapter 8 "Concluding remarks" is conclusion and its other sections are discussion. '
  'Vost has no discussion section, so the discussion range is built from two theses, not three.\n')
A('Sentences and paragraphs per section type (sentences / paragraphs / sentences with a move code):\n')
A('| Section type | Buttenschoen | Vost | Dablander | yours now | backup |')
A('|---|---|---|---|---|---|')
for t in TYPES:
    A(f'| {t} | ' + ' | '.join('{} / {} / {}'.format(*sizes[c][t]) for c in ['buttenschoen', 'vost', 'dablander', 'author_current', 'author_backup']) + ' |')
A('')
A('**Measures.** Each is a standard measure from the linguistics literature. The parser is spaCy 3.8 (`en_core_web_sm` 3.8.0), which tags each word and finds the grammatical links between words.\n')
A('- **Words per sentence.** Mean sentence length (Lu 2010, L2 Syntactic Complexity Analyzer).')
A('- **Subordinate clauses per sentence.** Finite dependent clauses: a "which", "that", "because", "when" or "if" clause with its own verb (Lu 2010).')
A('- **Nominalizations per 100 words.** Nouns ending -tion, -ment, -ness or -ity, such as "representation" or "improvement" (Biber 1988).')
A('- **Numbers per 100 words.** Numerals written in digits. Digits inside a name (QM9, ECFP4) do not count.')
A('- **% sentences with 3+ numbers.** Share of sentences carrying three or more numerals.')
A('- **Lexical density.** Nouns, verbs, adjectives and adverbs as a share of all words (Ure 1971). High means packed; low means many articles and prepositions.')
A('- **Hedges and boosters per 100 words.** Hyland (2005) word lists. Hedges soften a claim (may, likely, suggests). Boosters harden it (clearly, must, in fact; "show/find that"). '
  '"About" and "around" count only before a number.')
A('- **AI-guide phrases per 100 words.** The phrases in your `ai_phrasing.rtf` (e.g. "indicates that", "emerged as") plus style words that rose sharply in papers after ChatGPT '
  '(Kobak et al. 2025: delve, underscore, notably, crucial, comprehensive, insights, highlight, enhance, additionally). Field terms such as "robust" and "landscape" are left out.')
A('- **% passive sentences** and **we/our/I per 100 words** (self-mention, Hyland 2005).')
A('- **Moves.** What each sentence does, coded by reading (Swales 1990; Yang and Allison 2003): M1 method restated, M2 pointer to a figure or table, M3 result as numbers, '
  'M4 result in words, M5 comparison of results, M6 interpretation, M7 comparison with literature, M8 claim of strength, M9 limitation, M10 link, M11 recommendation. '
  'Moves exist only for results, discussion and conclusion sections. Your two versions were coded in full; each reference was coded on a sample of about 255 sentences '
  '(whole paragraphs, drawn per chapter and section type). Dablander was coded for this report (`codes_dablander.txt`); the others come from the earlier move files. One coder, one pass, no agreement check.')
A('- **% M3 among M2+M3+M4.** Of the sentences that report a result or point to one, the share that gives it as numbers.\n')
A('**Reference range.** The lowest and highest of the three theses on that measure. OUTSIDE means your value falls outside that range. '
  'The gap is the distance from the nearest end of the range, in standard deviations of the reference paragraphs of that section type. '
  'A gap of 1.0 means one typical paragraph-to-paragraph spread beyond the most extreme reference thesis. Gaps under 0.1 are negligible.\n')

# ---------------- section 1
A('## 1. Section types\n')
for t in TYPES:
    rows = [r for r in SEC if r['section_type'] == t]
    usable = [r for r in rows if not (r['metric'] in MOVEKEYS and int(r['n_refs']) < 2)]
    gaps = [r for r in usable if r['current_flag'] and fl(r['current_gap_sd']) is not None]
    gaps.sort(key=lambda r: -fl(r['current_gap_sd']))
    zero = [r for r in usable if r['current_flag'] and fl(r['current_gap_sd']) is None]
    A(f'### {t.capitalize()}\n')
    A('Five largest gaps, yours now:\n')
    for i, r in enumerate(gaps[:5], 1):
        A(f'{i}. {SHORT[r["metric"]]}: {f(r["current"])} now, {f(r["backup"])} in the backup; references {f(r["ref_min"])} to {f(r["ref_max"])}. '
          f'{arrow(r["current_flag"])}, gap {f(r["current_gap_sd"], 2)}.')
    for r in zero:
        A(f'- Also: {SHORT[r["metric"]]} is {f(r["current"])} now; every reference paragraph has 0, so no gap in standard deviations can be given.')
    A('')
    A('| Measure | Buttenschoen | Vost | Dablander | range | now | backup | now vs range | backup vs range |')
    A('|---|---|---|---|---|---|---|---|---|')
    for r in usable:
        A(f'| {SHORT[r["metric"]]} | {f(r["buttenschoen"])} | {f(r["vost"])} | {f(r["dablander"])} | {f(r["ref_min"])}-{f(r["ref_max"])} | '
          f'{f(r["current"])} | {f(r["backup"])} | {arrow(r["current_flag"])} {("(" + f(r["current_gap_sd"], 2) + ")") if r["current_flag"] and fl(r["current_gap_sd"]) is not None else ""} | '
          f'{arrow(r["backup_flag"])} {("(" + f(r["backup_gap_sd"], 2) + ")") if r["backup_flag"] and fl(r["backup_gap_sd"]) is not None else ""} |')
    if t in ('introduction', 'background', 'methods'):
        A('\nNo move rows: moves were coded only in results, discussion and conclusion sections.')
    A('')

# ---------------- section 2
A('## 2. Per chapter\n')
A('One line per chapter and section type of your thesis now. It lists only the measures OUTSIDE the reference range with a gap of 0.3 or more, '
  'with the backup value in brackets where the backup has the same chapter and section type. '
  'Section types differ between the two versions (the backup has Discussion and Limitations sections in Chapters 5 and 7, now merged), so some backup cells are missing. '
  'Every measure for every chapter is in `metrics_chapters.csv`.\n')
bak = {(r['chapter'], r['section_type'], r['metric']): r for r in CH if r['corpus'] == 'author_backup'}
nref = {(r['section_type'], r['metric']): int(r['n_refs']) for r in SEC}
groups = collections.OrderedDict()
for r in CH:
    if r['corpus'] != 'author_current': continue
    groups.setdefault((int(r['chapter']), r['section_type']), []).append(r)
for (ch, t), rows in sorted(groups.items(), key=lambda x: (x[0][0], TYPES.index(x[0][1]))):
    n = rows[0]['n_sent']
    items = []
    for r in sorted(rows, key=lambda r: -(fl(r['gap_sd']) or 0)):
        if r['metric'] in MOVEKEYS and nref[(t, r['metric'])] < 2: continue
        g = fl(r['gap_sd'])
        if not r['flag'] or g is None or g < 0.3: continue
        b = bak.get((r['chapter'], t, r['metric']))
        bs = f' [{f(b["value"])}]' if b else ''
        items.append(f'{SHORT[r["metric"]]} {f(r["value"])}{bs} (ref {f(r["ref_min"])}-{f(r["ref_max"])}, {"above" if "high" in r["flag"] else "below"}, gap {g:.2f})')
    A(f'- **Ch {ch} {t}** ({n} sentences): ' + ('; '.join(items) if items else 'nothing outside by 0.3 or more.'))
A('')

# ---------------- section 3
top = W[:150]
nai = sum(r['marker'] == 'AI-marked' for r in top); nh = sum(r['marker'].startswith('hers') for r in top)
nfix = sum(r['marker'] == 'hers (fix note above)' for r in top)
A('## 3. Paragraph worklist\n')
A(f'Every paragraph of 40 words or more in Chapters 1 to 8 of the thesis now ({len(W)} paragraphs) was scored on ten measures: '
  'words per sentence, subordinate clauses per sentence, nominalizations, numbers per 100 words, % sentences with 3+ numbers, lexical density, hedges, boosters, '
  'AI-guide phrases, and % M3 among M2+M3+M4 (results, discussion and conclusion only). '
  'Each measure becomes a z-score: how many standard deviations the paragraph sits from the mean of the reference paragraphs of the same section type, '
  'all three theses pooled. The distance is the root mean square of those z-scores, each capped at 5 so one extreme measure cannot decide the rank alone. '
  'Paragraphs under 40 words are left out because one hedge or one number swings them too far.\n')
A(f'**Of the top 150, {nai} are AI-marked and {nh} are yours** ({nfix} of yours carry a "manual review -- fixed:" note, which marks a correction to your text, not AI text). '
  'AI-marked means a `% TODO: manual review` or `% TODO: review, needs manual review` line sits in the comment lines directly above the paragraph.\n')
A('"Top measures" gives the paragraph\'s value, the reference mean, and the z-score. "Change" is generated from the two largest z-scores; read it as a starting point. '
  'Table numbers come from the 03:53 build (`main.aux`). The same list, with every measure, is in `worklist.csv`; all ranked paragraphs are in `worklist_all_paragraphs.csv`.\n')
A('| # | Ch | Section | tex line | First eight words | Whose | Top three measures | Change |')
A('|---|---|---|---|---|---|---|---|')
WK = ['sent_len', 'dc', 'nom', 'num', 'num3', 'lexd', 'hedge', 'boost', 'ai', 'm3share']
refd = parts['refdist']
for r in top:
    zs = {k: fl(r['z_' + k]) for k in WK if fl(r['z_' + k]) is not None}
    t3 = sorted(zs, key=lambda k: -abs(zs[k]))[:3]
    tm = '; '.join(f'{SHORT[k]} {f(r["v_" + k])} (ref {float(refd[r["section_type"]][k][0]):.1f}, z {zs[k]:+.1f})' for k in t3)
    sec = r['section'] + (f' / {r["subsection"]}' if r['subsection'] else '')
    fw = r['first_words'].replace('|', '/')
    A(f'| {r["rank"]} | {r["chapter"]} | {sec} ({r["section_type"]}) | {r["tex_line"]} | {fw} | {r["marker"]} | {tm} | {r["change"]} |')
A('')
A('Chapter counts in the top 150: ' + ', '.join(f'Ch {c}: {n}' for c, n in sorted(collections.Counter(int(r['chapter']) for r in top).items())) + '.\n')
A('Section-type counts in the top 150: ' + ', '.join(f'{t}: {n}' for t, n in collections.Counter(r['section_type'] for r in top).most_common()) + '.\n')
A('## 4. What the benchmark cannot see\n')
for line in [
 'Whether a sentence is true, or a number is right. Every measure counts form, not content.',
 'Chemistry. A wrong chemical term scores the same as a right one.',
 'Order and flow across paragraphs. The moves count what each sentence does, not whether the next paragraph follows from the last. Repetition between the abstract, Chapter 1 and Chapter 8 is not measured.',
 'Figures, tables and captions. They were removed before counting, so a paragraph that points to a good table is not credited for it.',
 'Tense, spelling (American or British), capitalization of titles, and citation practice. None is measured.',
 'Whether the references are the right target. Three theses is a narrow range: two from protein-ligand and generative work, one on molecular featurization. Being OUTSIDE their range is a difference, not an error. Their results sections report fewer datasets than yours.',
 'Move coding is one reader, one pass, with no second coder. Your thesis was coded in full by earlier sessions; each reference on a sample of about 255 sentences. Dablander was coded in this run. Move shares from a sample of 40 to 190 sentences per section type move by several points with a different sample.',
 'The word lists count words, not senses. Hyland checks each hit by hand; this run did not. Clear false hits in your field were removed: "suspect label", "known flag", "true label", "certain datasets". Others remain, such as "should" in a method rule.',
 'The AI-phrase count is not an AI detector. The reference theses score higher on it than you do, mostly from "suggests that" and "indicates that".',
 'Whether a paragraph is AI-written. "AI-marked" only means a manual-review comment sits above it. A paragraph rewritten by AI with no marker counts as yours.',
 'Parsing errors. LaTeX residue, stripped maths ("where is the target standard deviation") and pdftotext residue (joined hyphens, maths fragments in Dablander) confuse the parser. Display equations split a paragraph in two, so some worklist paragraphs start mid-sentence.',
 'Paragraphs under 40 words. They are left out of the worklist.',
 'Later edits. The numbers describe thesis.tex as saved at 04:06 on 10 Oct. Table numbers come from the 03:53 build and may have shifted.',
]:
    A(f'- {line}')
A('')
open(B + 'BENCHMARK_REPORT.md', 'w').write('\n'.join(o))
print('written', len(o), 'lines; top150 AI', nai, 'hers', nh)
