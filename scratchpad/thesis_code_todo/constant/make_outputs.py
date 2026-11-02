import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
J = json.load(open('constant/constant_counts.json'))
FIG = '/Users/apunt/repos/KIRBy/thesis_rework/stage1_takeaways_bullets/figures_code_input/constant_features'
DS = [('herg_ki', 'hERG $K_i$'), ('caco2', 'Caco-2'), ('logd', 'LogD'), ('qm9', 'QM9'),
      ('adme_mdr1', 'MDR1--MDCK'), ('adme_hlm', 'HLM clearance'), ('adme_rlm', 'RLM clearance'),
      ('herg', 'hERG blockers')]
REPS = [('ecfp4', 'ECFP4'), ('atom_pair', 'Atom pairs'), ('top_torsion', 'Topological torsions'),
        ('avalon', 'Avalon'), ('maccs', 'MACCS keys'), ('graph_kernel', 'WL graph kernel'),
        ('pdv', 'PDV'), ('mordred', 'Mordred'), ('grover', 'GROVER'), ('mhg_gnn', 'MHG-GNN'),
        ('graphmvp', 'GraphMVP'), ('molformer', 'MoLFormer'), ('smited', 'SMI-TED'),
        ('chembert', 'ChemBERT'), ('chemberta', 'ChemBERTa-2'), ('selformer', 'SELFormer'),
        ('mol2vec', 'Mol2vec')]
def inc(ds, rep):
    d = J['datasets'][ds]
    return rep in d['general_pool'] or rep == 'maccs'
def cell(ds, rep):
    d = J['datasets'][ds]
    if not inc(ds, rep): return '--'
    v = d['reps'].get(rep)
    if v is None or 'error' in v: return 'n/a'
    s = f"{v['constant']:,} / {v['near_constant']:,}"
    if rep == 'graph_kernel': s += f" / {v['total']:,}"
    return s
used = [r for r in REPS if any(inc(ds, r[0]) for ds, _ in DS)]
def total(rep):
    ts = sorted({J['datasets'][ds]['reps'][rep]['total'] for ds, _ in DS
                 if inc(ds, rep) and rep in J['datasets'][ds]['reps'] and 'error' not in J['datasets'][ds]['reps'][rep]})
    return f'{ts[0]:,}' if len(ts) == 1 else f'{ts[0]:,}--{ts[-1]:,}'
def table(dsl, label, cap):
    L = [r'\begin{table}[htbp]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
         r'\caption{' + cap + '}', r'\label{' + label + '}',
         r'\begin{tabular}{lr' + 'c' * len(dsl) + '}', r'\toprule',
         'Representation & Features & ' + ' & '.join(n for _, n in dsl) + r' \\',
         ' & & ' + ' & '.join(f"({J['datasets'][d]['n_molecules_meta']:,})" for d, _ in dsl) + r' \\', r'\midrule']
    for rep, name in used:
        if not any(inc(d, rep) for d, _ in dsl): continue
        L.append(f'{name} & {total(rep)} & ' + ' & '.join(cell(d, rep) for d, _ in dsl) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(L)
t1 = table(DS[:4], 'tab:constant_features', 'Constant and near-constant features, part 1. Each entry is ``constant / near-constant\'\' features of that representation on that dataset. The number of molecules is under each dataset name. -- means the representation is not in that dataset\'s base representations. n/a means no saved MACCS features exist for QM9.')
t2 = table(DS[4:], 'tab:constant_features_b', 'Constant and near-constant features, part 2. Entries read as in Table~\\ref{tab:constant_features}. hERG blockers is the 7,772-molecule training set from Lhasa Limited.')
open('constant/tables.tex', 'w').write(t1 + '\n\n' + t2 + '\n')
print(t1); print(t2)

# Figure
C_CONST, C_NEAR, C_VAR = '#1f3b73', '#6f93c9', '#e3e8ef'
fig, axes = plt.subplots(4, 2, figsize=(7.2, 11))
for ax, (ds, name) in zip(axes.T.flatten(), DS):
    reps = [(r, n) for r, n in REPS if inc(ds, r) and r in J['datasets'][ds]['reps'] and 'error' not in J['datasets'][ds]['reps'][r]]
    y = np.arange(len(reps))[::-1]
    for yi, (r, n) in zip(y, reps):
        v = J['datasets'][ds]['reps'][r]; t = v['total']
        c, nc = v['constant'] / t, (v['near_constant'] - v['constant']) / t
        ax.barh(yi, c, color=C_CONST, height=0.7, edgecolor='white', linewidth=0.8)
        ax.barh(yi, nc, left=c, color=C_NEAR, height=0.7, edgecolor='white', linewidth=0.8)
        ax.barh(yi, 1 - c - nc, left=c + nc, color=C_VAR, height=0.7, edgecolor='white', linewidth=0.8)
    ax.set_yticks(y); ax.set_yticklabels([f'{n} ({J["datasets"][ds]["reps"][r]["total"]:,})' for r, n in reps], fontsize=7.5)
    ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(['0', '25', '50', '75', '100'], fontsize=7.5)
    ax.set_title(f'{name.replace("$K_i$", "Ki").replace("--", "-")}\n{J["datasets"][ds]["n_molecules_meta"]:,} molecules', fontsize=8.5, loc='center')
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
for ax in axes[-1]: ax.set_xlabel('share of features (%)', fontsize=8)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=C_CONST, label='constant (one value in every molecule)'),
                    Patch(color=C_NEAR, label='near-constant, not constant (one value in at least 95% of molecules)'),
                    Patch(color=C_VAR, label='varying')], loc='upper center', ncol=1, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIG + '.pdf'); fig.savefig(FIG + '.png', dpi=200)
