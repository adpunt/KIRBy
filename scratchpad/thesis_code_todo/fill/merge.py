import re,glob,sys
T='/Users/apunt/repos/KIRBy/thesis_rework/stage1_takeaways_bullets/thesis.tex'
L=open(T).read().split('\n')
blocks=[]
for f in sorted(glob.glob('/Users/apunt/repos/KIRBy/scratchpad/thesis_code_todo/fill/*.tex')):
    txt=open(f).read()
    for m in re.finditer(r'%%%% INSERT_AFTER_LINE: (\d+)\n%%%% ANCHOR_TEXT: (.*?)\n(.*?)\n%%%% END',txt,re.S):
        n=int(m.group(1)); a=m.group(2); body=m.group(3).strip('\n')
        if L[n-1].strip()!=a.strip():
            sys.exit(f'ANCHOR MISMATCH {f} line {n}: file has {L[n-1]!r} vs {a!r}')
        blocks.append((n,f,body))
ns=[b[0] for b in blocks]; assert len(ns)==len(set(ns)),'duplicate line'
for n,f,body in sorted(blocks,reverse=True):
    L[n:n]=['']+body.split('\n')+['']
if '--dry' not in sys.argv: open(T,'w').write('\n'.join(L))
print(len(blocks),'blocks',sum(b[2].count('% CODE INPUT') for b in blocks),'CODE INPUT paras')
