"""House style for the thesis figures of Chapters 5, 6 and 7.

Everything above the line marked COPIED ENDS is copied verbatim from
~/repos/qsar_qm_models/scripts/generate_paper_figures_v2.py (lines 184-500),
so these figures match the noise chapter's. Do not re-pick those colours.
Below that line are the additions the KIRBy results need: model-name aliases,
the full representation label list, dataset display names and the domain
colours. Written by Claude, 2026-10-09.
"""
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# STYLE SETTINGS (Journal of Cheminformatics)  -- COPIED
# =============================================================================

sns.set_style("ticks")
# The thesis text block is 21 cm minus two 3 cm margins = 15 cm = 5.91 in
# (main.tex geometry). Figures are saved at this width so
# \includegraphics[width=\textwidth] renders 1:1 (no scaling), which keeps
# matplotlib text at its set size on the printed page. Was 6.5 in (the
# Springer journal width) until 2026-10-09.
TEXTWIDTH_IN = 5.91
HALFWIDTH_IN = TEXTWIDTH_IN / 2
plt.rcParams.update({
    'figure.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 11.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.linewidth': 0.8,
    'legend.frameon': False,
    'lines.linewidth': 1.5,
})

# Color palettes — maximally distinct, all different from clean blue
CLEAN_COLOR = '#0072B2'        # Steel blue — used ONLY for clean (no-noise) data
STRATEGY_COLORS = {
    # The settled roster.
    'gaussian': '#E31A1C',           # Bright red
    'grouped_wider': '#33A02C',      # Forest green
    'grouped_shifted': '#6A3D9A',    # Royal purple
    'censoring': '#B15928',          # Sienna/brown
    'student_t_nu5': '#FF7F00',      # Pure orange
    'outlier_p10': '#E91E63',        # Hot pink
    'laplace': '#17BECF',            # Cyan
    # Retired 2026-08-26; kept for files written before then.
    'legacy': '#E31A1C',
    'valprop': '#FF7F00',
    'quantile': '#33A02C',
    'threshold': '#6A3D9A',
    'outlier': '#B15928',
    'hetero': '#E91E63',
}

STRATEGY_LABELS = {
    'gaussian': 'Gaussian',
    'grouped_wider': 'Grouped, wider',
    'grouped_shifted': 'Grouped, shifted',
    'censoring': 'Censoring',
    'student_t_nu5': 'Student-t (nu=5)',
    'outlier_p10': 'Outlier (10%)',
    'laplace': 'Laplace',
    # Retired 2026-08-26.
    'legacy': 'Gaussian (retired name)',
    'valprop': 'Value-Prop.',
    'quantile': 'Quantile',
    'threshold': 'Threshold',
    'outlier': 'Outlier (retired: value-proportional)',
    'hetero': 'Heteroscedastic',
}

# Colors: variants of the same model family share ONE color.
# Distinction between variants comes from MODEL_MARKERS (shape).
MODEL_COLORS = {
    # Tree-based — each distinct (different model families)
    'rf': '#0072B2',               # Blue
    'qrf': '#0072B2',              # Blue (RF variant)
    'xgboost': '#56B4E9',          # Sky blue
    'lgb': '#009E73',              # Teal
    'ngboost': '#D55E00',          # Vermillion
    # NN-α family — all orange
    'dnn': '#E69F00',
    'dnn_bnn_full': '#E69F00',
    'dnn_bnn_last': '#E69F00',
    'dnn_vbll': '#E69F00',
    # NN-β family — all pink
    'mlp': '#CC79A7',
    'mlp_bnn_full': '#CC79A7',
    'mlp_bnn_last': '#CC79A7',
    'mlp_vbll': '#CC79A7',
    # SVM / GP — unique
    'svm': '#999999',              # Gray
    'gauche': '#882255',           # Wine
    'gauche_rbf': '#882255',       # Wine (RBF variant, PDV only)
    'het_gp_rbf': '#882255',       # Wine (GP family)
    'dnn_vbll_hetero': '#E69F00',  # NN-alpha
    'mlp_vbll_hetero': '#CC79A7',  # NN-beta
    'dnn_bnn_full_mve': '#E69F00',  # NN-alpha
    'mlp_bnn_full_mve': '#CC79A7',  # NN-beta
}

# Markers: variants of the same family get different shapes.
# Base model = circle, BNN Full = square, BNN Last = triangle, VBLL = diamond.
MODEL_MARKERS = {
    # Base models
    'rf': 'o', 'qrf': 'D', 'xgboost': 'o', 'lgb': 'o', 'ngboost': 'o',
    'svm': 'o', 'gauche': 'o', 'gauche_rbf': 's',
    'dnn': 'o', 'mlp': 'o',
    # NN-α family variants
    'dnn_bnn_full': 's', 'dnn_bnn_last': '^', 'dnn_vbll': 'D',
    # NN-β family variants
    'mlp_bnn_full': 's', 'mlp_bnn_last': '^', 'mlp_vbll': 'D',
    'het_gp_rbf': 'P', 'dnn_vbll_hetero': 'P', 'mlp_vbll_hetero': 'P',
    'dnn_bnn_full_mve': '*', 'mlp_bnn_full_mve': '*',
}

# Canonical ordering for legends — grouped by family, base model first.
MODEL_ORDER = [
    'rf', 'qrf',
    'xgboost', 'lgb', 'ngboost',
    'svm', 'gauche', 'gauche_rbf', 'het_gp_rbf',
    'dnn', 'dnn_bnn_full', 'dnn_bnn_full_mve', 'dnn_vbll', 'dnn_vbll_hetero',
    'mlp', 'mlp_bnn_full', 'mlp_bnn_full_mve', 'mlp_vbll', 'mlp_vbll_hetero',
]


def sort_models_by_family(models):
    """Sort models by MODEL_ORDER (family grouping), then alphabetical for unknown."""
    order_map = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(models, key=lambda m: (order_map.get(m, len(MODEL_ORDER)), m))


def _to_colmajor(items, ncol):
    """Reorder a reading-order (row-major) sequence into the column-major order
    matplotlib needs so a multi-column legend DISPLAYS left-to-right, top-to-bottom.
    """
    n = len(items)
    if ncol <= 1 or n == 0:
        return list(items)
    nrow = -(-n // ncol)  # ceil
    out = []
    for c in range(ncol):
        for r in range(nrow):
            idx = r * ncol + c
            if idx < n:
                out.append(items[idx])
    return out


MODEL_LABELS = {
    # Tree-based
    'rf': 'RF',
    'xgboost': 'XGBoost',
    'lgb': 'LightGBM',
    'qrf': 'QRF',
    'ngboost': 'NGBoost',
    # Neural networks
    'dnn': 'NN-α',
    'mlp': 'NN-β',
    # SVM / GP
    'svm': 'SVM',
    'gauche': 'GP',
    'gauche_rbf': 'GP (RBF)',
    # α Bayesian variants
    'dnn_bnn_full': 'BNN-α',
    'dnn_vbll': 'VBLL-α',
    # β Bayesian variants
    'mlp_bnn_full': 'BNN-β',
    'mlp_vbll': 'VBLL-β',
    'het_gp_rbf': 'GP (RBF, het.)',
    'dnn_bnn_full_mve': 'BNN-α (var. head)',
    'mlp_bnn_full_mve': 'BNN-β (var. head)',
    'dnn_vbll_hetero': 'VBLL-α (het.)',
    'mlp_vbll_hetero': 'VBLL-β (het.)',
}

REP_LABELS = {
    'pdv': 'PDV',
    'ecfp4': 'ECFP4',
    'sns': 'SNS',
    'smiles': 'SMILES',
    'randomized_smiles': 'R-SMILES',
    'mhggnn': 'MHG-GNN',
    'mol2vec': 'Mol2Vec',
}


def get_model_label(model):
    """Get clean display label for model."""
    if model in MODEL_LABELS:
        return MODEL_LABELS[model]
    if model.lower() in MODEL_LABELS:
        return MODEL_LABELS[model.lower()]
    return model.replace('_', '-').upper() if '_' in model else model.upper()


def get_rep_label(rep):
    """Get clean display label for representation."""
    return REP_LABELS.get(rep, rep.upper())

# =============================================================================
# COPIED ENDS. Additions for KIRBy below.
# =============================================================================

# KIRBy's names for models the paper script already colours. Each takes the
# existing family colour and marker; nothing above is recoloured.
MODEL_ALIASES = {'lightgbm': 'lgb', 'gp': 'gauche'}
for _k, _v in MODEL_ALIASES.items():
    MODEL_COLORS[_k] = MODEL_COLORS[_v]
    MODEL_MARKERS[_k] = MODEL_MARKERS[_v]
    MODEL_LABELS[_k] = MODEL_LABELS[_v]
    MODEL_ORDER.insert(MODEL_ORDER.index(_v) + 1, _k)
# knn has no colour in the paper. Black is the one Okabe-Ito colour the table
# above leaves unused that reads on white (the other, #F0E442 yellow, does not).
MODEL_COLORS['knn'] = '#000000'
MODEL_MARKERS['knn'] = 'o'
MODEL_LABELS['knn'] = 'kNN'
# The thesis calls KIRBy's network NN, not DNN or NN-alpha (Adelaide, 2026-10-09).
MODEL_LABELS['dnn'] = 'NN'
MODEL_ORDER.append('knn')

# One label per representation key in DATASET_REPS
# (tests/experiments/complexity/complexity_helpers.py) and the builders under
# src/kirby/representations/. Acronyms upper case, otherwise title case joined
# by a hyphen, never an underscore.
REP_LABELS.update({
    # small molecules
    'ecfp4': 'ECFP4', 'pdv': 'PDV', 'mordred': 'Mordred', 'avalon': 'Avalon',
    'maccs': 'MACCS', 'atom_pair': 'Atom-Pair', 'top_torsion': 'Top-Torsion',
    'graph_kernel': 'Graph-Kernel', 'mol2vec': 'Mol2Vec', 'mhg_gnn': 'MHG-GNN',
    'mhggnn': 'MHG-GNN', 'molformer': 'MoLFormer', 'chemberta': 'ChemBERTa',
    'chembert': 'ChemBERT', 'selformer': 'SELFormer', 'grover': 'GROVER',
    'graphmvp': 'GraphMVP', 'gnn_graphmvp': 'GNN-GraphMVP', 'molclr': 'MolCLR',
    'smited': 'SMITED', 'usr': 'USR', 'usrcat': 'USRCAT',
    'coulomb_matrix': 'Coulomb-Matrix', 'pharmacophore3d': 'Pharmacophore-3D',
    'contrastive_molformer': 'Contrastive-MoLFormer',
    'lora_molformer': 'LoRA-MoLFormer', 'molecular_lora': 'Molecular-LoRA',
    'molecular_gnn': 'Molecular-GNN', 'molecular_contrastive': 'Molecular-Contrastive',
    'onehot': 'One-Hot', 'onehot_components': 'One-Hot-Components',
    # reactions
    'cgr_morgan': 'CGR-Morgan', 'rxnfp': 'RXNFP', 'reactiont5': 'ReactionT5',
    'reactant_ecfp4': 'Reactant-ECFP4', 'reactant_maccs': 'Reactant-MACCS',
    'reactant_mordred': 'Reactant-Mordred', 'reactant_mol2vec': 'Reactant-Mol2Vec',
    'reactant_grover': 'Reactant-GROVER',
    'reactant_pharmacophore3d': 'Reactant-Pharmacophore-3D',
    'product_maccs': 'Product-MACCS', 'product_pharmacophore3d': 'Product-Pharmacophore-3D',
    'diff_avalon': 'Difference-Avalon', 'diff_maccs': 'Difference-MACCS',
    'diff_mhg_gnn': 'Difference-MHG-GNN', 'diff_mol2vec': 'Difference-Mol2Vec',
    # drug-target pairs
    'drug_avalon': 'Drug-Avalon', 'drug_chemberta': 'Drug-ChemBERTa',
    'drug_ecfp4': 'Drug-ECFP4', 'drug_maccs': 'Drug-MACCS', 'drug_mol2vec': 'Drug-Mol2Vec',
    'drug_molformer': 'Drug-MoLFormer', 'drug_mordred': 'Drug-Mordred',
    'drug_pdv': 'Drug-PDV', 'drug_selformer': 'Drug-SELFormer',
    'dta_protein_lora': 'DTA-Protein-LoRA',
    # antibodies and proteins
    'esm2': 'ESM2', 'esm2_cdr': 'ESM2-CDR', 'esm2_concat': 'ESM2-Concat',
    'esm2_early': 'ESM2-Early', 'esm2_entropy_top20': 'ESM2-Entropy-Top20',
    'esm2_layer24': 'ESM2-Layer24', 'esm2_llr': 'ESM2-LLR',
    'esm2_midlayers': 'ESM2-Midlayers', 'esm2_mutsite': 'ESM2-Mutation-Site',
    'esm2_per_cdr': 'ESM2-Per-CDR', 'esm2_per_region': 'ESM2-Per-Region',
    'esm2_wt_delta': 'ESM2-WT-Delta', 'esm3': 'ESM3', 'esm3_cdr': 'ESM3-CDR',
    'esmc': 'ESMC', 'esmc_cdr': 'ESMC-CDR', 'prot_t5': 'ProtT5', 'prot_t5_cdr': 'ProtT5-CDR',
    'antiberty': 'AntiBERTy', 'antiberty_cdr': 'AntiBERTy-CDR', 'antiberty_llr': 'AntiBERTy-LLR',
    'ablang2': 'AbLang2', 'abroberta': 'AbRoBERTa', 'abroberta_cdr': 'AbRoBERTa-CDR',
    'igt5': 'IgT5', 'saprot': 'SaProt', 'saprot_cdr': 'SaProt-CDR',
    'antibody_lora': 'Antibody-LoRA', 'antibody_lora_esm2': 'Antibody-LoRA-ESM2',
    'aac': 'AAC', 'blosum': 'BLOSUM', 'structural': 'Structural', 'paratope': 'Paratope',
    'mutation_features': 'Mutation-Features',
    'protein_aac': 'Protein-AAC', 'protein_contact_graph': 'Protein-Contact-Graph',
    'protein_esm2': 'Protein-ESM2', 'protein_esm2_lora': 'Protein-ESM2-LoRA',
    'protein_esm2_midlayers': 'Protein-ESM2-Midlayers', 'protein_esm2_pocket': 'Protein-ESM2-Pocket',
    'protein_esm3': 'Protein-ESM3', 'protein_esmc': 'Protein-ESMC', 'protein_kmer': 'Protein-k-mer',
    'protein_pocket': 'Protein-Pocket', 'protein_saprot': 'Protein-SaProt',
    'protein_usr_shape': 'Protein-USR-Shape',
    # nucleotides
    'dnabert2': 'DNABERT-2', 'hyenadna': 'HyenaDNA', 'nt_v2_250m': 'NT-v2-250M',
    'nt_v2_500m': 'NT-v2-500M', 'grover_dna': 'GROVER-DNA', 'mistral_dna': 'Mistral-DNA',
    'kmer3': '3-mer', 'kmer4': '4-mer', 'kmer5': '5-mer', 'kmer6': '6-mer',
    'gc_content': 'GC-Content', 'dinuc_freq': 'Dinucleotide-Frequency',
    'tetranuc_freq': 'Tetranucleotide-Frequency', 'codon_usage': 'Codon-Usage',
    'dna_shape': 'DNA-Shape',
    # patient data (TCGA)
    'expression': 'Expression', 'mirna_top200': 'miRNA-Top200', 'rppa': 'RPPA',
    'cnv_pca50': 'CNV-PCA50', 'methyl_hvg': 'Methylation-HVG', 'methyl_pca50': 'Methylation-PCA50',
    'mut_tmb': 'Mutation-TMB', 'hrd_score': 'HRD-Score', 'cimp_status': 'CIMP-Status',
    'immune_cibersort': 'Immune-CIBERSORT', 'immune_xcell': 'Immune-xCell',
    'stemness_mrnasi': 'Stemness-mRNAsi', 'purity_consensus': 'Purity-Consensus',
    'gene_bulkrnabert': 'Gene-BulkRNABert', 'gene_chromosome_arm': 'Gene-Chromosome-Arm',
    'gene_geneformer': 'Gene-Geneformer', 'gene_geneformer_v2': 'Gene-Geneformer-v2',
    'gene_stats': 'Gene-Stats', 'tnf_canonical': 'TNF-Canonical',
    'wsi_ctranspath': 'WSI-CTransPath', 'wsi_gigapath': 'WSI-GigaPath',
    'wsi_hoptimus0': 'WSI-H-optimus-0', 'wsi_phikon': 'WSI-Phikon', 'wsi_titan': 'WSI-TITAN',
})

# Configurations the noise experiments fit on, in words.
CONFIG_LABELS = {
    'base_1': 'First base representation',
    'base_2': 'Second base representation',
    'hybrid': 'Top hybrid',
    'hybrid__general2': 'Second hybrid',
}

# What each dataset measures, in words. Source: the outline's own names where it
# has one (THESIS_OUTLINE.md 5.4-7.11), otherwise the loader docstrings under
# src/kirby/datasets/ and DATASET_REGISTRY_DATA in tests/experiments/dataset_policy.py.
DATASET_LABELS = {
    'herg_ki': 'hERG Ki', 'herg': 'hERG blocking', 'herg_ic50': 'hERG IC50',
    'caco2': 'Caco-2 efflux', 'logd': 'LogD', 'qm9': 'QM9 HOMO-LUMO gap',
    'adme_hlm': 'Human microsomal clearance', 'adme_rlm': 'Rat microsomal clearance',
    'adme_mdr1': 'MDR1-MDCK efflux',
    'buchwald_hartwig': 'Buchwald-Hartwig yield',
    'flab_thermo_tresanco': 'Nanobody thermostability',
    'flab_binding_koenig': 'Anti-VEGF scan binding',
    'flab_expression_koenig': 'Anti-VEGF scan expression',
    'abcov_kd': 'Anti-coronavirus affinity',
    'abagym_Ang2': 'Angiopoietin-2 binding', 'abagym_EGFR': 'EGFR binding',
    'abagym_VEGF': 'VEGF binding',
    'ddg_combined': 'Binding free energy',
    'ginkgo_acsins': 'AC-SINS self-association', 'ginkgo_colloidal': 'SMAC colloidal stability',
    'ginkgo_hic': 'HIC retention', 'ginkgo_pr_cho': 'CHO polyreactivity',
    'ginkgo_titer': 'Expression titer',
    'humanness_therapeutic': 'Antibody humanness', 'cdr_flexibility': 'CDR flexibility',
    'tcga_brca_pam50': 'PAM50 subtype',
}

# Domain per dataset: DATASET_REGISTRY_DATA's family field (molecular, antibody,
# reaction, tcga), copied rather than imported because importing
# dataset_policy loads the search engine.
DATASET_DOMAIN = {
    **{d: 'Small molecules' for d in ('herg_ki', 'herg', 'herg_ic50', 'caco2', 'logd', 'qm9',
                                     'adme_hlm', 'adme_rlm', 'adme_mdr1')},
    'buchwald_hartwig': 'Reactions',
    **{d: 'Antibodies' for d in (
        'flab_thermo_tresanco', 'flab_binding_koenig', 'flab_expression_koenig', 'abcov_kd',
        'abagym_Ang2', 'abagym_EGFR', 'abagym_VEGF', 'ginkgo_acsins',
        'ginkgo_colloidal', 'ginkgo_hic', 'ginkgo_pr_cho', 'ginkgo_titer',
        'humanness_therapeutic', 'cdr_flexibility')},
    # AB-Bind + SKEMPI: SKEMPI is protein-protein binding in general, so this one
    # sits under proteins. The registry's family field says antibody (Claude's
    # assignment, 2026-10-09; Adelaide to confirm).
    'ddg_combined': 'Proteins',
    'tcga_brca_pam50': 'Patient data',
}
DOMAIN_ORDER = ['Small molecules', 'Antibodies', 'Proteins', 'Reactions', 'Patient data']
# Colours from Paul Tol's set, which the paper script already draws on
# (UNCERTAINTY_COLORS, RF_FAMILY_COLORS). None is CLEAN_COLOR.
DOMAIN_COLORS = {
    'Small molecules': '#117733',  # green
    'Antibodies': '#AA4499',       # purple
    'Proteins': '#CC6677',         # rose
    'Reactions': '#999933',        # olive
    'Patient data': '#332288',     # indigo
}

# Metric names with the direction arrow the outline asks for.
METRIC_LABELS = {
    'rmse': 'RMSE ↓', 'mae': 'MAE ↓',
    'spearman': 'Spearman ↑', 'pearson': 'Pearson ↑',
    'auc': 'AUROC ↑', 'auroc': 'AUROC ↑', 'roc_auc': 'AUROC ↑',
    'mcc': 'MCC ↑', 'f1_weighted': 'Weighted F1 ↑', 'r2': 'R² ↑',
    'ci': 'Concordance index ↑',
}
LOWER_IS_BETTER = {'rmse', 'mae'}

# Representation segment colours for the blueprint figure, where colour means
# the representation. Paul Tol's muted, light and vibrant sets, 28 distinct,
# CLEAN_COLOR left out. The figure restarts the list for each domain, and no
# domain uses more than 25 representations.
REP_SEGMENT_COLORS = ['#117733', '#AA4499', '#DDCC77', '#332288', '#88CCEE', '#CC6677',
                      '#44AA99', '#999933', '#882255', '#DDDDDD', '#EE7733', '#33BBEE',
                      '#EE3377', '#CC3311', '#009988', '#661100', '#6699CC', '#AA4466',
                      '#BBCC33', '#EE8866', '#FFAABB', '#99DDFF', '#AAAA00', '#000000',
                      '#994455', '#EEDD88', '#44BB99', '#BBBBBB']


def _fallback_label(key):
    return '-'.join(p.upper() if len(p) <= 4 else p.capitalize() for p in key.split('_'))


_warned = set()


def _warn_once(kind, key):
    import sys
    if (kind, key) not in _warned:
        _warned.add((kind, key))
        print(f'[label] no display name for {kind} {key!r}; using a generated one. '
              f'Add it to thesis_figure_style.py.', file=sys.stderr)


def rep_label(rep):
    if not rep:
        return 'None stored'
    if rep in REP_LABELS:
        return REP_LABELS[rep]
    _warn_once('representation', rep)
    return _fallback_label(rep)


def model_label(model):
    if model not in MODEL_LABELS:
        _warn_once('model', model)
    return get_model_label(model)


def dataset_label(ds):
    if ds in DATASET_LABELS:
        return DATASET_LABELS[ds]
    _warn_once('dataset', ds)
    return _fallback_label(ds)


def metric_label(metric):
    if metric in METRIC_LABELS:
        return METRIC_LABELS[metric]
    _warn_once('metric', metric)
    return metric


def cap(text):
    """First letter upper case, the rest as written: 'hybrid' -> 'Hybrid', 'hERG Ki' unchanged."""
    if not text or text[0].isupper() or text[:1] == 'h' and text[1:2].isupper():
        return text
    return text[0].upper() + text[1:]


def model_color(model):
    return MODEL_COLORS.get(model, '#333333')


def model_marker(model):
    return MODEL_MARKERS.get(model, 'o')
