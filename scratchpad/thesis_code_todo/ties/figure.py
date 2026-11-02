"""hERG Ki stereoisomer pair tied by ECFP4, largest pKi difference among tied pairs."""
import json, io, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
TD='/Users/apunt/repos/KIRBy/scratchpad/thesis_code_todo/ties'
FIG='/Users/apunt/repos/KIRBy/thesis_rework/stage1_takeaways_bullets/figures_code_input'
R=json.load(open(f'{TD}/ties_results.json'))
d=R['herg_ki']; g=[x for x in d['reps']['ecfp4']['groups'] if len(x['idx'])==2]
pair=max(g, key=lambda x:x['spread'])
i,j=pair['idx']; yi,yj=pair['labels']
X=np.load('/Volumes/seagate/kirby_cache/rep_cache/herg_ki/ecfp4.npy', mmap_mode='r')
xi,xj=np.asarray(X[i]),np.asarray(X[j]); assert np.array_equal(xi,xj)
def img(s):
    m=Chem.MolFromSmiles(s); AllChem.Compute2DCoords(m)
    dr=rdMolDraw2D.MolDraw2DCairo(900,650); o=dr.drawOptions(); o.addStereoAnnotation=True
    o.bondLineWidth=3; o.annotationFontScale=0.9; o.minFontSize=22
    dr.DrawMolecule(m); dr.FinishDrawing()
    return Image.open(io.BytesIO(dr.GetDrawingText()))
fig=plt.figure(figsize=(10,5.2))
gs=fig.add_gridspec(2,2,height_ratios=[6,0.6],hspace=0.15,wspace=0.05)
for k,(s,y,lab) in enumerate([(d['smiles'][i],yi,'A'),(d['smiles'][j],yj,'B')]):
    ax=fig.add_subplot(gs[0,k]); ax.imshow(img(s)); ax.axis('off')
    ax.set_title(f'{lab}: measured pKi = {y:.2f}', fontsize=13)
    bx=fig.add_subplot(gs[1,k]); on=np.flatnonzero(xi if k==0 else xj)
    bx.vlines(on,0,1,color='black',lw=0.8); bx.set_xlim(0,2048); bx.set_ylim(0,1); bx.set_yticks([])
    bx.set_xlabel(f'ECFP4 bits set: {len(on)} of 2,048 (bit index)', fontsize=10); bx.tick_params(labelsize=9)
fig.savefig(f'{FIG}/herg_ki_ecfp4_stereo_pair.pdf', bbox_inches='tight', dpi=300)
fig.savefig(f'{FIG}/herg_ki_ecfp4_stereo_pair.png', bbox_inches='tight', dpi=200)
out={'idx':[i,j],'smiles':[d['smiles'][i],d['smiles'][j]],'inchikey':[d['inchikey'][i],d['inchikey'][j]],
     'pki':[yi,yj],'mean_pki':(yi+yj)/2,'abs_diff_pki':abs(yi-yj),'ecfp4_bits_set':int((xi!=0).sum()),
     'cip':[[f'{a}{c}' for a,c in Chem.FindMolChiralCenters(Chem.MolFromSmiles(s),includeUnassigned=True,useLegacyImplementation=False)] for s in [d['smiles'][i],d['smiles'][j]]]}
json.dump(out,open(f'{TD}/figure_pair.json','w'),indent=1); print(out)
