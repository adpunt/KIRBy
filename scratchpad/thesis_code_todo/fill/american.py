import re,sys
IZ=('normalis standardis realis summaris regularis humanis optimis customis specialis neutralis memoris '
    'generalis featuris visualis unauthoris recognis randomis parameteris organis minimis maximis itemis '
    'hypothesis felinis caninis factoris binaris analys').split()
# hypothesis must only hit hypothesised/hypothesise, not "hypothesis"
FIX={'neighbour':'neighbor','Neighbour':'Neighbor','tumour':'tumor','Tumour':'Tumor','behaviour':'behavior',
     'colour':'color','favour':'favor','modelled':'modeled','Modelled':'Modeled','modelling':'modeling','Modelling':'Modeling',
     'labelled':'labeled','Labelled':'Labeled','labelling':'labeling','Labelling':'Labeling',
     'stereocentre':'stereocenter','catalogue':'catalog','analogue':'analog','licence':'license','amongst':'among','Amongst':'Among',
     'towards':'toward','Towards':'Toward','afterwards':'afterward','Afterwards':'Afterward','upwards':'upward','per cent':'percent'}
MASK=re.compile(r'\\(label|ref|eqref|autoref|cref|Cref|cite[a-z]*\*?|input|include|includegraphics|url|href|bibliography)(\[[^\]]*\])?\{[^}]*\}|[\w.\-]*[_/][\w.\-/]*|`[^`]*`')
def conv(s):
    for k,v in FIX.items(): s=re.sub(k,v,s)
    s=re.sub(r'\b([A-Za-z]*?)('+'|'.join(IZ)+r')(e|ed|es|ing|ation|ations|er|ers)\b',
             lambda m: m.group(0) if (m.group(2)=='hypothesis' and m.group(3) in ('',)) else m.group(1)+m.group(2)[:-1]+('z' if m.group(2)!='analys' else 'z')+m.group(3),s)
    return s
def line(l):
    if re.search(r'Centre [Ff]or Doctoral',l):
        l=l.replace('Centre for Doctoral','@@C1').replace('Centre For Doctoral','@@C2')
    out=[];last=0
    for m in MASK.finditer(l):
        out.append(conv(l[last:m.start()])); out.append(m.group(0)); last=m.end()
    out.append(conv(l[last:]))
    return ''.join(out).replace('@@C1','Centre for Doctoral').replace('@@C2','Centre For Doctoral')
for f in sys.argv[1:]:
    L=open(f).read().split('\n'); N=[line(l) for l in L]
    ch=sum(a!=b for a,b in zip(L,N)); open(f,'w').write('\n'.join(N)); print(f,ch,'lines changed')
