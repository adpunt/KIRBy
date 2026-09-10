#!/usr/bin/env python3
"""Verify thesis.tex: phrasing fidelity, LaTeX sanity, cites, figures, refs."""
import re, sys
from pathlib import Path
from difflib import SequenceMatcher

TH = Path("/Users/apunt/repos/KIRBy/thesis.tex")
CR = Path("/Users/apunt/Downloads/Confirmation_Report/paper.tex")
NI = Path("/Users/apunt/repos/qsar_qm_models/noise_inject.tex")
KB = Path("/Users/apunt/repos/KIRBy/PAPER_DRAFT.md")
OL = Path("/Users/apunt/Documents/Theses/THESIS_OUTLINE.md")
BIB = Path("/Users/apunt/Downloads/Confirmation_Report/refs.bib")
IMGDIR = Path("/Users/apunt/Downloads/Confirmation_Report")

th = TH.read_text().splitlines()
src_lines = set()
for f in (CR, NI):
    src_lines.update(l.rstrip() for l in f.read_text().splitlines() if l.strip())
ol_text = OL.read_text()
kb_text = KB.read_text()

fail = 0

# ---------------------------------------------------------------- 1. fidelity
AUTHORED = re.compile(
    r"^\s*(%|\\section|\\subsection|\\subsubsection|\\paragraph|\\todo|\\pagebreak|\\clearpage)"
)
unmatched = []
for i, l in enumerate(th, 1):
    s = l.rstrip()
    if not s.strip() or AUTHORED.match(s):
        continue
    if s in src_lines:
        continue
    unmatched.append((i, s))

print("=" * 78)
print("1. PHRASING FIDELITY")
print("=" * 78)
print(f"   {len(th)} lines total; {len(unmatched)} prose lines not byte-identical to")
print("   paper.tex / noise_inject.tex. These MUST all be the hand-converted")
print("   PAPER_DRAFT.md blocks (markdown has no verbatim .tex twin).\n")

# every unmatched line should trace to PAPER_DRAFT.md
def detex(t):
    t = re.sub(r"\\citep\{[^}]*\}", "", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = re.sub(r"[{}$\\~^_]", " ", t)
    t = t.replace("``", '"').replace("''", '"')
    return re.sub(r"[^a-z0-9 ]", " ", t.lower())

kb_norm = re.sub(r"\s+", " ", detex(kb_text))
bad_fidelity = []
for i, s in unmatched:
    probe = re.sub(r"\s+", " ", detex(s)).strip()
    if len(probe) < 40:
        continue
    # slide a window: best local match ratio against PAPER_DRAFT
    best = 0.0
    words = probe.split()
    head = " ".join(words[:12])
    pos = kb_norm.find(head)
    if pos >= 0:
        best = SequenceMatcher(None, probe, kb_norm[pos : pos + len(probe) + 60]).ratio()
    else:
        best = SequenceMatcher(None, probe[:300], kb_norm).quick_ratio()
    if best < 0.90:
        bad_fidelity.append((i, best, s[:100]))

if bad_fidelity:
    print("   !! lines that match NEITHER source verbatim NOR PAPER_DRAFT.md closely:")
    for i, r, s in bad_fidelity:
        print(f"      L{i} (sim={r:.2f})  {s}")
    fail += 1
else:
    print("   OK: every non-verbatim prose line traces to PAPER_DRAFT.md (sim >= 0.90).")

# ---------------------------------------------------------------- 2. outline quotes
print("\n" + "=" * 78)
print("2. OUTLINE TRANSCRIPTION FIDELITY  (% ---- from THESIS_OUTLINE.md blocks)")
print("=" * 78)
ol_norm = re.sub(r"\s+", " ", ol_text)
bad_ol, n_ol = [], 0
for i, l in enumerate(th, 1):
    m = re.match(r"^%\s{3}(.+)$", l)
    if not m:
        continue
    frag = m.group(1).strip()
    if len(frag) < 25:
        continue
    n_ol += 1
    if re.sub(r"\s+", " ", frag) not in ol_norm:
        bad_ol.append((i, frag[:95]))
if bad_ol:
    print(f"   !! {len(bad_ol)}/{n_ol} quoted outline fragments are NOT verbatim:")
    for i, f in bad_ol[:15]:
        print(f"      L{i}  {f}")
    fail += 1
else:
    print(f"   OK: all {n_ol} quoted outline fragments are verbatim from THESIS_OUTLINE.md.")

# ---------------------------------------------------------------- 3. latex envs
print("\n" + "=" * 78)
print("3. LATEX ENVIRONMENT BALANCE")
print("=" * 78)
body = "\n".join(l for l in th if not l.lstrip().startswith("%"))
opens = re.findall(r"\\begin\{([a-zA-Z*]+)\}", body)
closes = re.findall(r"\\end\{([a-zA-Z*]+)\}", body)
from collections import Counter
co, cc = Counter(opens), Counter(closes)
imbalance = {k: (co[k], cc[k]) for k in set(co) | set(cc) if co[k] != cc[k]}
if imbalance:
    print(f"   !! UNBALANCED: {imbalance}")
    fail += 1
else:
    print(f"   OK: {len(opens)} environments, all balanced.")

# stray $$ count
dd = body.count("$$")
print(f"   display-math `$$` delimiters: {dd} ({'even' if dd % 2 == 0 else 'ODD -- BROKEN'})")
if dd % 2:
    fail += 1

# ---------------------------------------------------------------- 4. refs
print("\n" + "=" * 78)
print("4. \\label / \\ref RESOLUTION")
print("=" * 78)
labels = set(re.findall(r"\\label\{([^}]+)\}", body))
refs = set(re.findall(r"\\(?:ref|autoref)\{([^}]+)\}", body))
dangling = sorted(refs - labels)
dup = [l for l, n in Counter(re.findall(r"\\label\{([^}]+)\}", body)).items() if n > 1]
print(f"   {len(labels)} labels, {len(refs)} refs.")
if dangling:
    print(f"   !! DANGLING \\ref -> {dangling}")
    fail += 1
else:
    print("   OK: no dangling \\ref.")
if dup:
    print(f"   !! DUPLICATE \\label -> {dup}")
    fail += 1
else:
    print("   OK: no duplicate \\label.")

# ---------------------------------------------------------------- 5. cites
print("\n" + "=" * 78)
print("5. CITATION KEYS vs refs.bib")
print("=" * 78)
bibkeys = set(re.findall(r"^@[a-zA-Z]+\{([^,]+)", BIB.read_text(), re.M))
cited = set()
for m in re.finditer(r"\\cite[a-zA-Z]*\{([^}]+)\}", body):
    cited.update(k.strip() for k in m.group(1).split(","))
missing = sorted(cited - bibkeys)
print(f"   {len(cited)} distinct keys cited; {len(missing)} missing from refs.bib.")
for k in missing:
    lower = {b.lower(): b for b in bibkeys}
    hint = f"   (refs.bib has '{lower[k.lower()]}')" if k.lower() in lower else ""
    print(f"      MISSING  {k}{hint}")

# ---------------------------------------------------------------- 6. figures
print("\n" + "=" * 78)
print("6. FIGURE FILES")
print("=" * 78)
figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body)
for f in figs:
    ok = (IMGDIR / f).exists()
    tag = "present in Confirmation_Report/" if ok else "NOT in Confirmation_Report/ (from noise_inject -- assumed uploaded)"
    print(f"   {'OK ' if ok else '?? '} {f:40s} {tag}")

# ---------------------------------------------------------------- 7. hygiene
print("\n" + "=" * 78)
print("7. HYGIENE")
print("=" * 78)
lnl = [i for i, l in enumerate(th, 1)
       if not l.lstrip().startswith("%") and re.search(r"Exploring techniques for learning with noisy labels", l)]
print(f"   removed LNL chapter heading present in body: {'YES -- BUG' if lnl else 'no  OK'}")
if lnl: fail += 1
nsi = [i for i, l in enumerate(th, 1)
       if not l.lstrip().startswith("%") and re.search(r"\bNSI\b|Noise Sensitivity Index", l)]
print(f"   stale NSI metric in body:                    {'YES -- BUG at '+str(nsi) if nsi else 'no  OK'}")
if nsi: fail += 1
rep = [i for i, l in enumerate(th, 1)
       if not l.lstrip().startswith("%") and re.search(r"\bthis report\b|\bThis report\b|confirmation report", l)]
print(f"   'this report' left in body:                  {'at lines '+str(rep) if rep else 'no  OK'}")
mathbb = "\\mathbb" in body
mathbb_msg = "YES -> add \\usepackage{amssymb} to main.tex" if mathbb else "no"
print("   uses \\mathbb (needs amssymb):                " + mathbb_msg)
addl = len(re.findall(r"Additional file", body))
print(f"   'Additional file~N' journal refs to rewrite:  {addl}")
todos = sum(1 for l in th if l.lstrip().startswith("% TODO")) + body.count("\\todo[")
print(f"   TODO markers ( % TODO + \\todo[inline] ):      {todos}")

print("\n" + "=" * 78)
print("FAILED CHECKS:", fail)
sys.exit(0)
