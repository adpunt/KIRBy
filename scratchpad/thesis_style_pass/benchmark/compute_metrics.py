"""Linguistic metrics for five corpora: author_current, author_backup (thesis.tex versions, Chapters 1-8)
and three reference DPhil theses (buttenschoen, vost, dablander). Input: full_segments_*.jsonl.
Parser: spaCy 3.8 en_core_web_sm 3.8.0 (tagger + dependency parser).

Per sentence:
  words        tokens with a letter or digit, after removing [CIT]/[REF] placeholders, citations and
               Figure/Table/Section/Chapter numbers (so cross-references never count as numbers)
  dep_clauses  finite dependent clauses (Lu 2010, L2SCA 'DC'): a verb attached as advcl, ccomp, csubj,
               csubjpass, relcl or acl that is finite (VerbForm=Fin itself or via a finite auxiliary) or has
               its own subject
  nominal      nouns ending -tion, -ment, -ness, -ity (+ plural) (Biber 1988 nominalisations), 6+ letters
  numbers      numeral tokens written in digits (12, 0.461, 26.6%); digits inside names (QM9, ECFP4) do not count
  content/alpha  lexical density (Ure 1971): NOUN, PROPN, VERB, ADJ, ADV tokens / all alphabetic tokens
  hedges       Hyland (2005) hedge list; 'about'/'around'/'approximately' only before a number
  boosters     Hyland (2005) booster list; show/find/demonstrate forms only when followed by 'that'; 'known' only as
               'known that/to'; true/certain/sure/clear/definite not before a noun ('true label'); hedge 'suspect' only as a
               verb, 'possible' not before a noun
  ai_hits      phrases from the author's AI writing guide (~/Documents/ai_phrasing.rtf) plus style words
               from Kobak et al. (2025, excess vocabulary after ChatGPT), technical words of this field left out
  passive      sentence has a passive subject or passive auxiliary
  selfmention  we, our, us, ours, I, my, me (Hyland 2005 self-mentions)
Moves (Swales 1990; Yang & Allison 2003) joined from the moves_*.csv files by sentence text.
"""
import json, re, csv, sys, collections
sys.path.insert(0, '/private/tmp/claude-501/-Users-apunt-repos-KIRBy/9f687a4a-c71f-4415-b6e9-5c2f5f842ef6/scratchpad/pylib')
import spacy
B = '/Users/apunt/repos/KIRBy/scratchpad/thesis_style_pass/benchmark/'
nlp = spacy.load('en_core_web_sm', disable=['ner', 'lemmatizer'])

HEDGES = """about almost apparent apparently appear appeared appears approximately argue argued argues around assume assumed
broadly certain_amount certain_extent certain_level claim claimed claims could couldn't doubt doubtful essentially estimate
estimated fairly feel feels felt frequently from_my_perspective from_our_perspective from_this_perspective generally guess
indicate indicated indicates in_general in_most_cases in_most_instances in_my_opinion in_my_view in_this_view in_our_opinion
in_our_view largely likely mainly may maybe might mostly often on_the_whole ought perhaps plausible plausibly possible possibly
postulate postulated postulates presumable presumably probable probably quite rather relatively roughly seems should sometimes
somewhat suggest suggested suggests suppose supposed supposes suspect suspects tend_to tended_to tends_to to_my_knowledge
typical typically uncertain uncertainly unclear unclearly unlikely usually would wouldn't""".split()
BOOSTERS = """actually always believe believed believes beyond_doubt certain certainly clear clearly conclusively decidedly definite
definitely demonstrate demonstrated demonstrates doubtless establish established evident evidently find finds found in_fact
incontestable incontestably incontrovertible incontrovertibly indeed indisputable indisputably know known must never no_doubt
obvious obviously of_course prove proved proves realize realized realizes really show showed shown shows sure surely think
thinks thought truly true undeniable undeniably undisputedly undoubtedly without_doubt""".split()
APPROX = {'about', 'around', 'approximately'}
EVIDENTIAL = {'demonstrate', 'demonstrated', 'demonstrates', 'find', 'finds', 'found', 'show', 'showed', 'shown', 'shows', 'prove', 'proved', 'proves', 'establish', 'established'}
AI_PATTERNS = [  # (label, regex) -- first block from the author's guide, second block Kobak et al. 2025 style words
 ('actionable', r'\bactionable\b'), ('revealed a', r'\breveal(s|ed)? an? (unexpected|clear|striking|surprising)\b'),
 ('warrant', r'\bwarrant(s|ed|ing)?\b'), ('emerge as', r'\bemerg(e|es|ed|ing) as\b'),
 ('critical/key insight', r'\b(critical|key|central) insights?\b'),
 ('this X suggests/indicates', r'\b(this|these) (\w+ ){0,2}(suggests?|indicates?|demonstrates?)\b'),
 ('indicates/suggests/demonstrates that', r'\b(indicat(e|es|ed|ing)|suggest(s|ed|ing)?|demonstrat(e|es|ed|ing)) that\b'),
 ('promising avenue', r'\bpromising (avenues?|directions?)\b'), ('X emerge', r'\b(several|many|two|three|four) \w+ emerge\b'),
 ('delve', r'\bdelv(e|es|ed|ing)\b'), ('underscore', r'\bunderscor(e|es|ed|ing)\b'), ('showcase', r'\bshowcas(e|es|ed|ing)\b'),
 ('intricate', r'\bintricate\b'), ('pivotal', r'\bpivotal\b'), ('notably', r'\bnotably\b'), ('crucial', r'\bcrucial(ly)?\b'),
 ('comprehensive', r'\bcomprehensive(ly)?\b'), ('meticulous', r'\bmeticulous(ly)?\b'), ('invaluable', r'\binvaluable\b'),
 ('noteworthy', r'\bnoteworthy\b'), ('groundbreaking', r'\bgroundbreaking\b'), ('realm', r'\brealms?\b'),
 ('tapestry', r'\btapestry\b'), ('nuanced', r'\bnuanced?\b'), ('seamless', r'\bseamless(ly)?\b'),
 ('leverage', r'\bleverag(e|es|ed|ing)\b'), ('insight', r'\binsights?\b'), ('highlight', r'\bhighlight(s|ed|ing)?\b'),
 ('emphasize', r'\bemphasi[sz](e|es|ed|ing)\b'), ('enhance', r'\benhanc(e|es|ed|ing)\b'), ('additionally', r'\badditionally\b'),
]
AI_RE = [(a, re.compile(b, re.I)) for a, b in AI_PATTERNS]
NOMSUF = re.compile(r'^[a-z]{2,}(tion|ment|ness|ity|tions|ments|nesses|ities)$')
NOMSTOP = {'city', 'cities', 'entity', 'entities', 'comment', 'comments', 'element', 'elements', 'moment', 'moments', 'segment', 'segments', 'pity'}
NUM = re.compile(r'^[−–-]?\d+([.,]\d+)*%?$')
SELF = {'we', 'our', 'us', 'ours', 'i', 'my', 'me'}

def normalise(t, ref):
    t = t.replace('\\%', '%').replace('--', '–').replace('\\', '')
    t = t.replace('[CIT]', ' ').replace('[REF]', ' ')
    if ref:
        t = re.sub(r'\[\s*\d+(\s*[,–-]\s*\d+)*\s*\]', ' ', t)                                    # [6, 7, 8]
        t = re.sub(r'\((see )?[A-Z][^()]{0,120}?\b(19|20)\d\d[a-z]?(;[^()]*)?\)', ' ', t)           # (Trott & Olson, 2010)
        t = re.sub(r'\b[A-Z][a-zé\-]+( et al\.| and [A-Z][a-z]+)? \((19|20)\d\d[a-z]?\)', 'they', t)  # Corso et al. (2023)
        t = re.sub(r'\b(Figures?|Figs?\.|Tables?|Sections?|Chapters?|Eqs?\.|Equations?|Appendix|Remark|Definition|Proposition)\s+[A-Z]?\d+(\.\d+)*[a-z]?((\s*(,|and|to|–|-)\s*)[A-Z]?\d+(\.\d+)*[a-z]?)*',
                   lambda m: m.group(1), t)
    return re.sub(r'\s+', ' ', t).strip()

def multi(lw, i, lst_set):
    # returns matched item at token i (single or multiword with '_' joins), longest first
    for n in (4, 3, 2):
        k = '_'.join(lw[i:i + n])
        if k in lst_set: return k
    return lw[i] if lw[i] in lst_set else None

HSET, BSET = set(HEDGES), set(BOOSTERS)
def sent_metrics(doc, raw):
    toks = [t for t in doc if not t.is_space]
    words = [t for t in toks if re.search(r'[A-Za-z0-9]', t.text)]
    alpha = [t for t in toks if t.is_alpha]
    lw = [t.text.lower() for t in toks]
    dc = 0
    for t in toks:
        if t.dep_ in ('advcl', 'ccomp', 'csubj', 'csubjpass', 'relcl', 'acl') and t.pos_ in ('VERB', 'AUX'):
            fin = 'Fin' in t.morph.get('VerbForm') or any(c.dep_ in ('aux', 'auxpass') and 'Fin' in c.morph.get('VerbForm') for c in t.children)
            subj = any(c.dep_ in ('nsubj', 'nsubjpass', 'expl') for c in t.children)
            if fin or subj: dc += 1
    nom = [t.text.lower() for t in alpha if t.pos_ == 'NOUN' and len(t.text) >= 6 and NOMSUF.match(t.text.lower()) and t.text.lower() not in NOMSTOP]
    nums = [t.text for t in toks if NUM.match(t.text)]
    content = sum(1 for t in alpha if t.pos_ in ('NOUN', 'PROPN', 'ADJ', 'ADV') or (t.pos_ == 'VERB'))
    hed, boo = [], []
    i = 0
    while i < len(lw):
        h = multi(lw, i, HSET)
        if h and h in ('suspect', 'suspects') and toks[i].pos_ != 'VERB': h = None          # 'suspect label' is a term here
        if h and h in ('possible',) and toks[i].dep_ == 'amod': h = None                     # 'every possible pair'
        if h:
            if h in APPROX:
                if i + 1 < len(lw) and (NUM.match(lw[i + 1]) or lw[i + 1] in ('one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'half', 'a', 'twice')):
                    hed.append(h)
            else:
                hed.append(h)
            i += h.count('_') + 1; continue
        b = multi(lw, i, BSET)
        if b and b == 'known' and not (toks[i].pos_ == 'VERB' and i + 1 < len(lw) and lw[i + 1] in ('that', 'to')): b = None  # 'known flag'
        if b and b in ('true', 'certain', 'sure', 'clear', 'definite') and toks[i].dep_ == 'amod': b = None   # 'true label', 'certain datasets'
        if b:
            if b in EVIDENTIAL:
                if i + 1 < len(lw) and lw[i + 1] == 'that': boo.append(b)
            else:
                boo.append(b)
            i += b.count('_') + 1; continue
        i += 1
    ai = [a for a, rx in AI_RE for _ in rx.finditer(raw)]
    passive = any(t.dep_ in ('nsubjpass', 'auxpass') for t in toks)
    selfm = sum(1 for w in lw if w in SELF)
    return dict(words=len(words), alpha=len(alpha), content=content, dep_clauses=dc, nominal=len(nom), nom_list=nom,
                numbers=len(nums), num3=int(len(nums) >= 3), hedges=len(hed), hedge_list=hed, boosters=len(boo), booster_list=boo,
                ai_hits=len(ai), ai_list=ai, passive=int(passive), selfmention=selfm)

def load_moves():
    M = {}
    for name in ('author_current', 'author_backup', 'buttenschoen', 'vost'):
        seg = {json.loads(l)['sentence_id']: json.loads(l)['text'] for l in open(B + f'segments_{name}.jsonl')}
        M[name] = {seg[r['sentence_id']]: r['move'] for r in csv.DictReader(open(B + f'moves_{name}.csv')) if r['sentence_id'] in seg}
    M['dablander_ids'] = {r['sentence_id']: r['move'] for r in csv.DictReader(open(B + 'moves_dablander.csv'))}
    return M

if __name__ == '__main__':
    M = load_moves()
    out = open(B + 'metrics_sentences.jsonl', 'w')
    for name in ('author_current', 'author_backup', 'buttenschoen', 'vost', 'dablander'):
        R = [json.loads(l) for l in open(B + f'full_segments_{name}.jsonl')]
        ref = not name.startswith('author')
        texts = [normalise(r['text'], ref) for r in R]
        for r, t, doc in zip(R, texts, nlp.pipe(texts, batch_size=256)):
            m = sent_metrics(doc, t)
            mv = M['dablander_ids'].get(r['sentence_id']) if name == 'dablander' else M[name].get(r['text'])
            rec = dict(corpus=name, sentence_id=r['sentence_id'], paragraph_id=r['paragraph_id'], chapter=r['chapter'],
                       section_type=r['section_type'], section=r.get('section'), subsection=r.get('subsection'),
                       tex_line=r.get('tex_line'), marker=r.get('marker'), move=mv, norm_text=t, **m)
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(name, len(R), flush=True)
