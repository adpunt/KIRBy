"""
Feature Name Retrieval for Cross-Domain Hybrid Explainability

Maps representation indices to human-readable feature names across all domains:
- Molecular: PDV descriptors, PCA components
- Protein: AAC, biophysical descriptors, k-mers
- Antibody: AAC, one-hot/BLOSUM positions, structural, AntiFold
- Genomic/TCGA: pathway scores, immune deconvolution, gene stats, chr arms
- Reaction: per-component ECFP4 bits

Each get_*_names(n) function returns a List[str] of length n.
"""

from typing import Dict, List, Optional


# =============================================================================
# UTILITY
# =============================================================================

def _pad_or_truncate(names: List[str], n: int, prefix: str) -> List[str]:
    """Ensure output is exactly n names. Truncate or pad with indexed fallbacks."""
    if len(names) >= n:
        return names[:n]
    return names + [f"{prefix}_dim_{i}" for i in range(len(names), n)]


def get_feature_description(feat_name: str) -> Optional[str]:
    """Look up a biological description for a named feature.

    Returns a human-readable string explaining what the feature measures
    and why it matters, or None if no description is available.
    """
    return FEATURE_DESCRIPTIONS.get(feat_name)


# =============================================================================
# MOLECULAR
# =============================================================================

def get_pdv_names(n: int) -> List[str]:
    """PDV descriptor names from DEFAULT_DESCRIPTOR_LIST (200D)."""
    from kirby.representations.molecular import DEFAULT_DESCRIPTOR_LIST
    return _pad_or_truncate(list(DEFAULT_DESCRIPTOR_LIST), n, 'pdv')


def get_pca_names(n: int) -> List[str]:
    """Generic PC_1..PC_N for any PCA-reduced representation."""
    return [f"PC_{i+1}" for i in range(n)]


# =============================================================================
# PROTEIN
# =============================================================================

_PROTEIN_DESCRIPTOR_NAMES = [
    'hydrophobicity_mean', 'hydrophobicity_std', 'hydrophobicity_min', 'hydrophobicity_max',
    'charge_mean', 'charge_std', 'charge_min', 'charge_max',
    'mw_mean', 'mw_std', 'mw_min', 'mw_max',
    'volume_mean', 'volume_std', 'volume_min', 'volume_max',
    'polarity_mean', 'polarity_std', 'polarity_min', 'polarity_max',
    'flexibility_mean', 'flexibility_std', 'flexibility_min', 'flexibility_max',
    'sequence_length', 'molecular_weight_estimate', 'net_charge_pH7.4',
    'fraction_hydrophobic', 'fraction_charged', 'net_charge_ratio',
]

_STANDARD_AAS = 'ACDEFGHIKLMNPQRSTVWY'


def get_protein_aac_names(n: int) -> List[str]:
    """20 amino acid frequencies (protein AAC)."""
    names = [f"aa_{aa}" for aa in _STANDARD_AAS]
    return _pad_or_truncate(names, n, 'protein_aac')


def get_protein_descriptor_names(n: int) -> List[str]:
    """30 biophysical protein descriptors."""
    return _pad_or_truncate(list(_PROTEIN_DESCRIPTOR_NAMES), n, 'protein_desc')


def get_protein_kmer_names(n: int) -> List[str]:
    """3-mer frequency features (8000D = 20^3)."""
    import itertools
    kmers = [''.join(t) for t in itertools.product(_STANDARD_AAS, repeat=3)]
    return _pad_or_truncate(kmers, n, 'kmer')


# =============================================================================
# ANTIBODY
# =============================================================================

def get_antibody_aac_names(n: int) -> List[str]:
    """40 antibody AAC features: heavy_A..heavy_Y, light_A..light_Y."""
    names = [f"heavy_{aa}" for aa in _STANDARD_AAS] + \
            [f"light_{aa}" for aa in _STANDARD_AAS]
    return _pad_or_truncate(names, n, 'ab_aac')


def get_onehot_names(n: int) -> List[str]:
    """One-hot / BLOSUM positional encoding: H_pos0_A..L_pos129_Y (5600D)."""
    names = []
    for chain, maxlen in [('H', 150), ('L', 130)]:
        for pos in range(maxlen):
            for aa in _STANDARD_AAS:
                names.append(f"{chain}_pos{pos}_{aa}")
    return _pad_or_truncate(names, n, 'onehot')


_STRUCTURAL_NAMES = [
    # SASA (15)
    'total_sasa', 'heavy_sasa', 'light_sasa',
    'cdr_h1_sasa', 'cdr_h2_sasa', 'cdr_h3_sasa',
    'cdr_l1_sasa', 'cdr_l2_sasa', 'cdr_l3_sasa',
    'cdr_exposed_frac', 'cdr_h3_exposed_frac',
    'hydrophobic_sasa', 'sasa_variance', 'buried_surface_area', 'cdr_h3_frac',
    # Confidence (5)
    'mean_bfactor', 'min_bfactor', 'cdr_h3_bfactor',
    'frac_confident', 'bfactor_variance',
    # Paratope (8)
    'n_paratope', 'paratope_frac', 'paratope_charge',
    'paratope_hydrophobicity', 'paratope_aromatic_frac',
    'paratope_h1_frac', 'paratope_h2_frac', 'paratope_h3_frac',
]

_ANTIFOLD_NAMES = ['mean_score', 'min_score', 'max_score', 'sum_score', 'n_unfavorable']


def get_structural_names(n: int) -> List[str]:
    """28 antibody structural features."""
    return _pad_or_truncate(list(_STRUCTURAL_NAMES), n, 'structural')


def get_antifold_names(n: int) -> List[str]:
    """5 AntiFold inverse folding score features."""
    return _pad_or_truncate(list(_ANTIFOLD_NAMES), n, 'antifold')


# =============================================================================
# GENOMIC / TCGA  (lazy imports from kirby.representations.data)
# =============================================================================

def get_hallmark_names(n: int) -> List[str]:
    """~50 MSigDB Hallmark pathway names."""
    from kirby.representations.data import hallmark
    names = sorted(hallmark().keys())
    return _pad_or_truncate(names, n, 'hallmark')


def get_progeny_names(n: int) -> List[str]:
    """14 PROGENy signaling pathway names."""
    from kirby.representations.data import progeny_weights
    names = sorted(progeny_weights().keys())
    return _pad_or_truncate(names, n, 'progeny')


def get_kegg_names(n: int) -> List[str]:
    """~320 KEGG pathway names."""
    from kirby.representations.data import kegg
    names = sorted(kegg().keys())
    return _pad_or_truncate(names, n, 'kegg')


def get_reactome_names(n: int) -> List[str]:
    """~1818 Reactome pathway names."""
    from kirby.representations.data import reactome
    names = sorted(reactome().keys())
    return _pad_or_truncate(names, n, 'reactome')


def get_oncogenic_names(n: int) -> List[str]:
    """~189 MSigDB C6 oncogenic signature names."""
    from kirby.representations.data import oncogenic
    names = sorted(oncogenic().keys())
    return _pad_or_truncate(names, n, 'oncogenic')


def get_dorothea_names(n: int) -> List[str]:
    """~367 DoRothEA transcription factor names."""
    from kirby.representations.data import dorothea_regulons
    names = sorted(dorothea_regulons().keys())
    return _pad_or_truncate(names, n, 'dorothea')


def get_combined_pathway_names(n: int) -> List[str]:
    """Concatenated: PROGENy + Hallmark + DoRothEA names."""
    names = get_progeny_names(9999) + get_hallmark_names(9999) + get_dorothea_names(9999)
    return _pad_or_truncate(names, n, 'combined_pathway')


def get_immune_names(n: int) -> List[str]:
    """Bindea immune cell types (sorted) + 3 ESTIMATE scores."""
    from kirby.representations.genomic import BINDEA_IMMUNE_CELL_MARKERS
    cell_types = sorted(BINDEA_IMMUNE_CELL_MARKERS.keys())
    names = cell_types + ['ESTIMATE_stromal', 'ESTIMATE_immune', 'ESTIMATE_purity']
    return _pad_or_truncate(names, n, 'immune')


_GENE_STATS_NAMES = [
    'expr_mean', 'expr_std', 'skewness', 'kurtosis',
    'pct10', 'pct25', 'median', 'pct75', 'pct90',
    'fraction_zeros', 'entropy', 'range', 'MAD',
    'fraction_expressed', 'mean_of_expressed',
]


def get_gene_stats_names(n: int) -> List[str]:
    """15 per-sample expression distribution statistics."""
    return _pad_or_truncate(list(_GENE_STATS_NAMES), n, 'gene_stats')


def get_chromosome_arm_names(n: int) -> List[str]:
    """Chromosome arm labels from gene_chr_arm_map (~39-44 arms)."""
    from kirby.representations.data import gene_chr_arm_map
    chr_map = gene_chr_arm_map()
    names = sorted(set(chr_map.values()))
    return _pad_or_truncate(names, n, 'chr_arm')


# =============================================================================
# REACTION
# =============================================================================

def get_concat_ecfp4_names(n: int) -> List[str]:
    """Per-component ECFP4 bit names: component_0_bit0..component_K_bitN."""
    # Default: 1024 bits per component; infer n_components from n
    n_bits = 1024
    n_components = max(1, (n + n_bits - 1) // n_bits)
    names = []
    for c in range(n_components):
        for b in range(n_bits):
            names.append(f"component_{c}_bit{b}")
    return _pad_or_truncate(names, n, 'concat_ecfp4')


# =============================================================================
# FEATURE DESCRIPTIONS — biological meaning of named features
# =============================================================================
#
# Maps feature name -> human-readable description explaining what the feature
# measures and why it matters. Used by explain_hybrid() to produce interpretable
# output. Features without entries here get a generic category-level description.
#

FEATURE_DESCRIPTIONS: Dict[str, str] = {}

# ---------------------------------------------------------------------------
# PDV molecular descriptors (RDKit)
# ---------------------------------------------------------------------------
_PDV_DESCRIPTIONS = {
    # Topological / graph indices
    'BalabanJ': 'Balaban J index: topological descriptor of molecular branching; higher = more branched/complex',
    'BertzCT': 'Bertz complexity: information-theoretic molecular complexity; higher = more complex bonding',
    'Chi0': 'Kier-Hall zeroth-order chi index: sum of atom valence connectivities',
    'Chi0n': 'Chi0 (Kier) normalized: size-independent molecular connectivity',
    'Chi0v': 'Chi0 (valence-corrected): accounts for heteroatom valence electrons',
    'Chi1': 'First-order chi index: captures bond connectivity patterns',
    'Chi1n': 'Chi1 normalized: size-independent bond connectivity',
    'Chi1v': 'Chi1 (valence-corrected): bond connectivity with heteroatom correction',
    'Chi2n': 'Second-order chi (normalized): encodes path-2 branching patterns',
    'Chi2v': 'Second-order chi (valence): path-2 connectivity with heteroatom correction',
    'Chi3n': 'Third-order chi (normalized): captures 3-bond path topology',
    'Chi3v': 'Third-order chi (valence): 3-bond path with heteroatom correction',
    'Chi4n': 'Fourth-order chi (normalized): 4-bond path topology',
    'Chi4v': 'Fourth-order chi (valence): 4-bond path with heteroatom correction',
    'HallKierAlpha': 'Hall-Kier alpha shape index: deviation from linear topology',
    'Ipc': 'Information content (bond connectivity): graph entropy measure',
    'Kappa1': 'Kappa shape index 1: encodes molecular shape (linear vs spherical)',
    'Kappa2': 'Kappa shape index 2: captures spatial density of atoms',
    'Kappa3': 'Kappa shape index 3: captures branching pattern of the molecular graph',

    # Physical properties
    'MolLogP': 'Wildman-Crippen LogP: lipophilicity (octanol/water partition); higher = more hydrophobic, lower aqueous solubility',
    'MolMR': 'Wildman-Crippen molar refractivity: molecular polarizability; correlates with molecular volume',
    'MolWt': 'Molecular weight (Da); Lipinski RO5: MW < 500 for oral drugs',
    'ExactMolWt': 'Exact molecular weight from isotopic masses',
    'HeavyAtomCount': 'Number of non-hydrogen atoms; proxy for molecular size',
    'HeavyAtomMolWt': 'Average molecular weight per heavy atom',
    'TPSA': 'Topological polar surface area (A^2); <140 needed for cell membrane permeability; key oral bioavailability predictor',
    'LabuteASA': 'Labute approximate surface area (A^2); solvent-accessible surface estimate',
    'FractionCSP3': 'Fraction of sp3 carbons: measures saturation/3D-character; higher = more drug-like (escape flatland)',
    'qed': 'Quantitative estimate of drug-likeness (0-1): composite score weighting MW, LogP, HBD, HBA, PSA, ROTB, AROM, ALERTS',
    'NumValenceElectrons': 'Total valence electrons: relates to reactivity and bonding capacity',
    'NumRadicalElectrons': 'Radical electron count: non-zero for radical species',

    # Hydrogen bonding
    'NumHAcceptors': 'H-bond acceptors (N, O atoms); Lipinski RO5: HBA <= 10',
    'NumHDonors': 'H-bond donors (NH, OH groups); Lipinski RO5: HBD <= 5',
    'NHOHCount': 'Count of NH and OH groups: total H-bond donor sites',
    'NOCount': 'Count of N and O atoms: total H-bond acceptor sites',

    # Ring counts
    'RingCount': 'Total number of ring systems; impacts rigidity and metabolic stability',
    'NumAromaticRings': 'Aromatic ring count; higher = better stacking but worse solubility',
    'NumAromaticCarbocycles': 'Aromatic carbocyclic rings (e.g. benzene)',
    'NumAromaticHeterocycles': 'Aromatic heterocyclic rings (e.g. pyridine, furan)',
    'NumAliphaticRings': 'Aliphatic (saturated) ring count; improves 3D complexity',
    'NumAliphaticCarbocycles': 'Aliphatic carbocyclic rings (e.g. cyclohexane)',
    'NumAliphaticHeterocycles': 'Aliphatic heterocyclic rings (e.g. piperidine)',
    'NumSaturatedRings': 'Fully saturated ring count; sp3-rich, improves selectivity',
    'NumSaturatedCarbocycles': 'Saturated carbocyclic rings',
    'NumSaturatedHeterocycles': 'Saturated heterocyclic rings (e.g. morpholine)',

    # Flexibility
    'NumRotatableBonds': 'Rotatable bond count; higher = more flexible; Veber rule: ROTB <= 10 for oral drugs',
    'NumHeteroatoms': 'Number of non-C/H atoms: heteroatom richness',

    # Electrostatics
    'MaxAbsPartialCharge': 'Largest absolute Gasteiger partial charge: most polarized atom',
    'MaxPartialCharge': 'Most positive Gasteiger partial charge: strongest electrophilic site',
    'MinAbsPartialCharge': 'Smallest absolute Gasteiger partial charge: most neutral atom',
    'MinPartialCharge': 'Most negative Gasteiger partial charge: strongest nucleophilic site',
    'MaxAbsEStateIndex': 'Maximum absolute E-state: most electronically extreme atom',
    'MaxEStateIndex': 'Maximum E-state index: most electron-rich atom',
    'MinAbsEStateIndex': 'Minimum absolute E-state: most electronically neutral atom',
    'MinEStateIndex': 'Minimum E-state index: most electron-poor atom',

    # Fingerprint density
    'FpDensityMorgan1': 'Morgan FP density (r=1): unique substructures/atom, radius 1',
    'FpDensityMorgan2': 'Morgan FP density (r=2): unique substructures/atom, radius 2',
    'FpDensityMorgan3': 'Morgan FP density (r=3): unique substructures/atom, radius 3',
}

# VSA descriptors — bin descriptions
for i in range(1, 12):
    _PDV_DESCRIPTIONS[f'EState_VSA{i}'] = f'Surface area (A^2) in E-state bin {i}: VSA weighted by electrotopological state'
for i in range(1, 15):
    _PDV_DESCRIPTIONS[f'PEOE_VSA{i}'] = f'Surface area (A^2) in partial charge bin {i} (Gasteiger): charge-weighted accessible surface'
for i in range(1, 11):
    _PDV_DESCRIPTIONS[f'SMR_VSA{i}'] = f'Surface area (A^2) in molar refractivity bin {i}: polarizability-weighted surface'
for i in range(1, 13):
    _PDV_DESCRIPTIONS[f'SlogP_VSA{i}'] = f'Surface area (A^2) in LogP bin {i} (Wildman-Crippen): lipophilicity-weighted surface'
for i in range(1, 11):
    _PDV_DESCRIPTIONS[f'VSA_EState{i}'] = f'E-state sum for atoms in VSA bin {i}: electronic state grouped by surface area'

# Functional group counts (fr_*) — auto-generate readable descriptions
_FR_DESCRIPTIONS = {
    'fr_Al_COO': 'Aliphatic carboxylic acid count',
    'fr_Al_OH': 'Aliphatic hydroxyl count',
    'fr_Al_OH_noTert': 'Aliphatic hydroxyl count (excluding tertiary)',
    'fr_ArN': 'Aromatic nitrogen count',
    'fr_Ar_COO': 'Aromatic carboxylic acid count',
    'fr_Ar_N': 'Aromatic nitrogen atom count',
    'fr_Ar_NH': 'Aromatic NH count',
    'fr_Ar_OH': 'Phenol count',
    'fr_COO': 'Carboxylic acid count',
    'fr_COO2': 'Carboxylate (ionized) count',
    'fr_C_O': 'Carbonyl O count',
    'fr_C_O_noCOO': 'Carbonyl O count (excluding carboxylic acids)',
    'fr_C_S': 'Thiol-related C-S count',
    'fr_HOCCN': 'HOCCN group count (beta-amino alcohol)',
    'fr_Imine': 'Imine (C=N) count',
    'fr_NH0': 'Tertiary amine count (no H on N)',
    'fr_NH1': 'Secondary amine count (one H on N)',
    'fr_NH2': 'Primary amine count (two H on N)',
    'fr_N_O': 'N-O bond count (hydroxamic acid, nitroso)',
    'fr_Ndealkylation1': 'N-dealkylation site count (CYP metabolism)',
    'fr_Ndealkylation2': 'N-dealkylation site count (type 2, CYP metabolism)',
    'fr_Nhpyrrole': 'H-pyrrole count',
    'fr_SH': 'Thiol (-SH) count',
    'fr_aldehyde': 'Aldehyde count: reactive carbonyl, Michael acceptor risk',
    'fr_alkyl_carbamate': 'Alkyl carbamate count',
    'fr_alkyl_halide': 'Alkyl halide count: electrophilic, potential reactivity',
    'fr_allylic_oxid': 'Allylic oxidation site count (CYP metabolism)',
    'fr_amide': 'Amide bond count: peptide-like, stable under physiological pH',
    'fr_amidine': 'Amidine count: strongly basic, cation at pH 7.4',
    'fr_aniline': 'Aniline count: aromatic amine, potential toxicophore',
    'fr_aryl_methyl': 'Aryl methyl count: benzylic oxidation site',
    'fr_azide': 'Azide count: click chemistry handle, potential explosive',
    'fr_azo': 'Azo (-N=N-) count: chromophore, potential carcinogen',
    'fr_barbitur': 'Barbiturate scaffold count',
    'fr_benzene': 'Benzene ring count',
    'fr_benzodiazepine': 'Benzodiazepine scaffold count',
    'fr_bicyclic': 'Bicyclic system count: adds 3D complexity',
    'fr_diazo': 'Diazo group count',
    'fr_dihydropyridine': 'Dihydropyridine count (e.g. nifedipine-type)',
    'fr_epoxide': 'Epoxide count: highly reactive, electrophilic, potential toxicophore',
    'fr_ester': 'Ester count: hydrolyzable by esterases in vivo',
    'fr_ether': 'Ether count: generally metabolically stable',
    'fr_furan': 'Furan count: metabolized to reactive epoxide by CYP',
    'fr_guanido': 'Guanidine count: strongly basic, permanently charged at pH 7.4',
    'fr_halogen': 'Halogen count (F, Cl, Br, I): lipophilicity modulator',
    'fr_hdrzine': 'Hydrazine count: potential toxicophore',
    'fr_hdrzone': 'Hydrazone count',
    'fr_imidazole': 'Imidazole ring count: CYP inhibitor risk, metal coordination',
    'fr_imide': 'Imide count',
    'fr_isocyan': 'Isocyanate count: highly reactive electrophile',
    'fr_isothiocyan': 'Isothiocyanate count: electrophilic, covalent modifier',
    'fr_ketone': 'Ketone count',
    'fr_ketone_Topliss': 'Topliss-class ketone count',
    'fr_lactam': 'Lactam (cyclic amide) count: beta-lactam = antibiotic scaffold',
    'fr_lactone': 'Lactone (cyclic ester) count: prodrug motif',
    'fr_methoxy': 'Methoxy group count: O-demethylation by CYP',
    'fr_morpholine': 'Morpholine ring count: improves aqueous solubility',
    'fr_nitrile': 'Nitrile (-CN) count: weak H-bond acceptor',
    'fr_nitro': 'Nitro group count: potential mutagenic alert',
    'fr_nitro_arom': 'Aromatic nitro count: mutagenic structural alert (Ames test)',
    'fr_nitro_arom_nonortho': 'Aromatic nitro (non-ortho) count',
    'fr_nitroso': 'Nitroso count: reactive, potential carcinogen',
    'fr_oxazole': 'Oxazole ring count',
    'fr_oxime': 'Oxime count',
    'fr_para_hydroxylation': 'Para-hydroxylation site count (CYP metabolism)',
    'fr_phenol': 'Phenol count: antioxidant, Phase II conjugation substrate',
    'fr_phenol_noOrthoHbond': 'Phenol (no ortho H-bond) count',
    'fr_phos_acid': 'Phosphoric acid count: highly polar, charged',
    'fr_phos_ester': 'Phosphate ester count: prodrug motif',
    'fr_piperdine': 'Piperidine ring count: common in CNS drugs',
    'fr_piperzine': 'Piperazine ring count: improves solubility, common drug scaffold',
    'fr_priamide': 'Primary amide count',
    'fr_prisulfonamd': 'Primary sulfonamide count',
    'fr_pyridine': 'Pyridine ring count: basic nitrogen, CYP inhibitor risk',
    'fr_quatN': 'Quaternary nitrogen count: permanently charged',
    'fr_sulfide': 'Thioether count',
    'fr_sulfonamd': 'Sulfonamide count: antibiotic scaffold',
    'fr_sulfone': 'Sulfone count: metabolically stable',
    'fr_term_acetylene': 'Terminal acetylene count: CYP mechanism-based inhibitor',
    'fr_tetrazole': 'Tetrazole count: carboxylic acid bioisostere',
    'fr_thiazole': 'Thiazole ring count',
    'fr_thiocyan': 'Thiocyanate count',
    'fr_thiophene': 'Thiophene ring count: CYP bioactivation risk',
    'fr_unbrch_alkane': 'Unbranched alkane chain count: lipophilic tail',
    'fr_urea': 'Urea group count',
}
_PDV_DESCRIPTIONS.update(_FR_DESCRIPTIONS)

FEATURE_DESCRIPTIONS.update(_PDV_DESCRIPTIONS)

# ---------------------------------------------------------------------------
# Protein biophysical descriptors (30D)
# ---------------------------------------------------------------------------
_PROTEIN_DESC_DESCRIPTIONS = {
    'hydrophobicity_mean': 'Mean Kyte-Doolittle hydrophobicity: >0 = hydrophobic core/membrane protein, <0 = hydrophilic/soluble',
    'hydrophobicity_std': 'Hydrophobicity variation across residues: high = amphipathic regions, mixed polarity',
    'hydrophobicity_min': 'Most hydrophilic residue (Arg=-4.5): extent of polar surface patches',
    'hydrophobicity_max': 'Most hydrophobic residue (Ile=4.5): presence of buried hydrophobic core',
    'charge_mean': 'Mean per-residue charge at pH 7.4: overall electrostatic character',
    'charge_std': 'Charge variation: high = mixed charged/neutral regions (salt bridges)',
    'charge_min': 'Most negative residue charge (-1 = Asp/Glu): acidic character',
    'charge_max': 'Most positive residue charge (+1 = Arg/Lys): basic character',
    'mw_mean': 'Mean residue molecular weight (Da): amino acid size distribution',
    'mw_std': 'Residue MW variation: high = diverse amino acid sizes',
    'mw_min': 'Lightest residue (Gly=75): presence of small flexible residues',
    'mw_max': 'Heaviest residue (Trp=204): presence of large aromatic residues',
    'volume_mean': 'Mean residue volume (A^3): average side chain size',
    'volume_std': 'Residue volume variation: packing heterogeneity',
    'volume_min': 'Smallest residue volume (Gly=60): glycine content / flexibility',
    'volume_max': 'Largest residue volume (Trp=228): bulky aromatic content',
    'polarity_mean': 'Fraction of polar residues (D,E,H,K,N,Q,R,S,T,Y): surface exposure tendency',
    'polarity_std': 'Polarity variation: high = alternating polar/nonpolar stretches (amphipathic)',
    'polarity_min': 'Minimum polarity (0 = nonpolar): presence of fully hydrophobic segments',
    'polarity_max': 'Maximum polarity (1 = polar): presence of fully polar segments',
    'flexibility_mean': 'Mean B-factor flexibility index (0.31-0.54): overall conformational flexibility',
    'flexibility_std': 'Flexibility variation: high = mixture of rigid and flexible regions',
    'flexibility_min': 'Least flexible residue (Trp/Phe=0.31): rigid aromatic core',
    'flexibility_max': 'Most flexible residue (Gly=0.54): disordered loops or linkers',
    'sequence_length': 'Sequence length / 1000: normalized protein size',
    'molecular_weight_estimate': 'Total MW / 100000: overall protein mass',
    'net_charge_pH7.4': 'Sum of residue charges at pH 7.4: determines electrostatic interactions and pI',
    'fraction_hydrophobic': 'Fraction of I,V,L,F,C,M,A,W residues: hydrophobic content, membrane association',
    'fraction_charged': 'Fraction of D,E,K,R,H residues: solubility driver, electrostatic interactions',
    'net_charge_ratio': '(n_positive - n_negative) / length: charge asymmetry, positive = basic protein',
}
FEATURE_DESCRIPTIONS.update(_PROTEIN_DESC_DESCRIPTIONS)

# ---------------------------------------------------------------------------
# Protein AAC (20D)
# ---------------------------------------------------------------------------
_AA_DESCRIPTIONS = {
    'aa_A': 'Alanine frequency: small, hydrophobic; enriched in alpha-helices',
    'aa_C': 'Cysteine frequency: forms disulfide bonds; stabilizes tertiary structure',
    'aa_D': 'Aspartate frequency: negatively charged (pH 7.4); metal coordination, catalysis',
    'aa_E': 'Glutamate frequency: negatively charged; salt bridges, enzymatic active sites',
    'aa_F': 'Phenylalanine frequency: aromatic, hydrophobic; pi-stacking in binding pockets',
    'aa_G': 'Glycine frequency: smallest AA, high flexibility; enriched in loops and turns',
    'aa_H': 'Histidine frequency: imidazole ring, pH-sensitive charge; metal binding, catalysis',
    'aa_I': 'Isoleucine frequency: branched hydrophobic; protein core packing',
    'aa_K': 'Lysine frequency: positively charged; surface exposed, post-translational modification',
    'aa_L': 'Leucine frequency: hydrophobic; most common AA in proteins, helix former',
    'aa_M': 'Methionine frequency: thioether, moderately hydrophobic; oxidation-sensitive',
    'aa_N': 'Asparagine frequency: polar, uncharged; N-glycosylation site (N-X-S/T)',
    'aa_P': 'Proline frequency: rigid, disrupts helices; enriched in collagen and turns',
    'aa_Q': 'Glutamine frequency: polar, uncharged; deamidation-prone in antibodies',
    'aa_R': 'Arginine frequency: positively charged; strongest cation, pi-cation interactions',
    'aa_S': 'Serine frequency: small polar; phosphorylation site, O-glycosylation',
    'aa_T': 'Threonine frequency: small polar, branched; phosphorylation, O-glycosylation',
    'aa_V': 'Valine frequency: hydrophobic, branched; beta-sheet former, core packing',
    'aa_W': 'Tryptophan frequency: largest AA, aromatic; rare, spectroscopic probe, binding hotspot',
    'aa_Y': 'Tyrosine frequency: aromatic, polar; phosphorylation site, CDR binding hotspot',
}
FEATURE_DESCRIPTIONS.update(_AA_DESCRIPTIONS)

# ---------------------------------------------------------------------------
# Antibody AAC (40D)
# ---------------------------------------------------------------------------
for _chain in ('heavy', 'light'):
    for _aa in _STANDARD_AAS:
        _base = _AA_DESCRIPTIONS.get(f'aa_{_aa}', f'{_aa} frequency')
        # Strip 'aa_X: ' prefix and add chain context
        _bio = _base.split(': ', 1)[1] if ': ' in _base else _base
        FEATURE_DESCRIPTIONS[f'{_chain}_{_aa}'] = f'{_chain.title()} chain {_aa} composition: {_bio}'

# ---------------------------------------------------------------------------
# Antibody structural features (28D)
# ---------------------------------------------------------------------------
_STRUCTURAL_DESCRIPTIONS = {
    # SASA (15)
    'total_sasa': 'Total solvent-accessible surface area (A^2): overall antibody surface exposed to solvent',
    'heavy_sasa': 'Heavy chain SASA (A^2): VH domain surface exposure',
    'light_sasa': 'Light chain SASA (A^2): VL domain surface exposure (0 if single-domain)',
    'cdr_h1_sasa': 'CDR-H1 SASA (A^2): first heavy-chain complementarity loop surface',
    'cdr_h2_sasa': 'CDR-H2 SASA (A^2): second heavy-chain complementarity loop surface',
    'cdr_h3_sasa': 'CDR-H3 SASA (A^2): primary antigen-binding loop; most variable, most important for specificity',
    'cdr_l1_sasa': 'CDR-L1 SASA (A^2): first light-chain complementarity loop surface',
    'cdr_l2_sasa': 'CDR-L2 SASA (A^2): second light-chain complementarity loop surface',
    'cdr_l3_sasa': 'CDR-L3 SASA (A^2): third light-chain complementarity loop surface',
    'cdr_exposed_frac': 'Fraction CDR / framework SASA: higher = CDRs more surface-exposed (accessible for binding)',
    'cdr_h3_exposed_frac': 'CDR-H3 SASA / total SASA: how prominent the main binding loop is',
    'hydrophobic_sasa': 'Hydrophobic SASA (A^2): nonpolar surface; high = aggregation risk, sticky patches',
    'sasa_variance': 'SASA variance across residues: structural heterogeneity',
    'buried_surface_area': 'Buried surface area proxy (A^2): VH-VL interface area; relates to stability',
    'cdr_h3_frac': 'CDR-H3 SASA / heavy chain SASA: binding loop prominence on heavy chain',
    # Confidence (5)
    'mean_bfactor': 'Mean predicted B-factor: overall structure prediction confidence (lower = more confident)',
    'min_bfactor': 'Minimum B-factor: best-predicted region confidence',
    'cdr_h3_bfactor': 'CDR-H3 mean B-factor: binding loop prediction confidence (often high = flexible/disordered)',
    'frac_confident': 'Fraction of residues with B-factor < 2.0 A: proportion of well-resolved structure',
    'bfactor_variance': 'B-factor variance: heterogeneity in prediction confidence across the structure',
    # Paratope (8)
    'n_paratope': 'Number of paratope (antigen-contacting) residues: binding site size',
    'paratope_frac': 'Paratope fraction (n_paratope / total residues): proportion of sequence involved in binding',
    'paratope_charge': 'Paratope net charge: electrostatic character of binding surface',
    'paratope_hydrophobicity': 'Paratope mean hydrophobicity: positive = hydrophobic binding (protein antigens), negative = polar (carbohydrate/charged antigens)',
    'paratope_aromatic_frac': 'Paratope aromatic fraction (F/W/Y): pi-stacking and cation-pi potential; Y/W are binding hotspot residues',
    'paratope_h1_frac': 'CDR-H1 paratope fraction: H1 contribution to binding interface',
    'paratope_h2_frac': 'CDR-H2 paratope fraction: H2 contribution to binding interface',
    'paratope_h3_frac': 'CDR-H3 paratope fraction: H3 contribution (usually dominant)',
}
FEATURE_DESCRIPTIONS.update(_STRUCTURAL_DESCRIPTIONS)

# ---------------------------------------------------------------------------
# AntiFold inverse folding scores (5D)
# ---------------------------------------------------------------------------
_ANTIFOLD_DESCRIPTIONS = {
    'mean_score': 'AntiFold mean mutation score: average log-likelihood ratio across mutations; positive = structure-compatible mutations',
    'min_score': 'AntiFold worst mutation score: least structure-compatible mutation; large negative = destabilizing',
    'max_score': 'AntiFold best mutation score: most structure-compatible mutation',
    'sum_score': 'AntiFold summed mutation score: cumulative structural impact',
    'n_unfavorable': 'Count of unfavorable mutations (score < 0): mutations that clash with predicted fold',
}
FEATURE_DESCRIPTIONS.update(_ANTIFOLD_DESCRIPTIONS)

# ---------------------------------------------------------------------------
# Gene expression statistics (15D)
# ---------------------------------------------------------------------------
_GENE_STATS_DESCRIPTIONS = {
    'expr_mean': 'Mean gene expression: overall transcriptional activity level',
    'expr_std': 'Expression standard deviation: transcriptional heterogeneity across genes',
    'skewness': 'Expression distribution skewness: positive = right-tail of highly expressed genes',
    'kurtosis': 'Expression distribution kurtosis: high = extreme expression outliers (driver genes)',
    'pct10': '10th percentile expression: baseline/low-expression level',
    'pct25': '25th percentile expression: lower quartile of gene expression',
    'median': 'Median expression: central tendency of transcriptome',
    'pct75': '75th percentile expression: upper quartile of gene expression',
    'pct90': '90th percentile expression: highly expressed gene threshold',
    'fraction_zeros': 'Fraction of zero-expression genes: dropout rate; indicates sequencing depth or cell type',
    'entropy': 'Expression entropy (50-bin): transcriptome diversity; low = dominated by few genes',
    'range': 'Expression range (max - min): dynamic range of transcriptome',
    'MAD': 'Median absolute deviation of expression: robust measure of transcriptomic variability',
    'fraction_expressed': 'Fraction of genes with expression > 0: transcriptome breadth',
    'mean_of_expressed': 'Mean of non-zero genes: expression level conditioned on being active',
}
FEATURE_DESCRIPTIONS.update(_GENE_STATS_DESCRIPTIONS)

# ---------------------------------------------------------------------------
# Immune cell deconvolution (Bindea et al. 2013) + ESTIMATE
# ---------------------------------------------------------------------------
_IMMUNE_DESCRIPTIONS = {
    'B_cells': 'B cell infiltration (ssGSEA, Bindea): adaptive humoral immunity; high in lymphocyte-rich tumors',
    'T_cells': 'T cell infiltration: adaptive cellular immunity; pan-T marker',
    'T_helper_cells': 'CD4+ T helper cell infiltration: orchestrate immune responses via cytokines',
    'Th1': 'Th1 cell infiltration: pro-inflammatory, IFN-gamma secreting; anti-tumor immunity',
    'Th2': 'Th2 cell infiltration: anti-inflammatory, IL-4/IL-13 secreting; associated with immune evasion in cancer',
    'TFH': 'Follicular helper T cell infiltration: support B cell maturation in germinal centers',
    'Th17': 'Th17 cell infiltration: IL-17 secreting; dual role — anti-tumor and pro-tumor inflammation',
    'TReg': 'Regulatory T cell infiltration (FOXP3+): immune suppression; high = immunosuppressive microenvironment',
    'CD8_T_cells': 'CD8+ cytotoxic T cell infiltration: direct tumor cell killing; key immunotherapy biomarker',
    'Tgd': 'Gamma-delta T cell infiltration: innate-like T cells; tissue-resident, rapid response',
    'Cytotoxic_cells': 'Cytotoxic cell infiltration (granzymes, perforin): active cell-mediated killing',
    'NK_cells': 'Natural killer cell infiltration: innate anti-tumor immunity, no MHC restriction',
    'NK_CD56dim_cells': 'CD56dim NK cell infiltration: mature, highly cytotoxic (ADCC effectors)',
    'NK_CD56bright_cells': 'CD56bright NK cell infiltration: immature, cytokine-producing (IL-2, IFN-gamma)',
    'DC': 'Dendritic cell infiltration: professional antigen presentation to T cells',
    'iDC': 'Immature dendritic cell infiltration: antigen sampling, not yet activated',
    'aDC': 'Activated dendritic cell infiltration: antigen-presenting, T cell priming',
    'pDC': 'Plasmacytoid dendritic cell infiltration: type I interferon production (IL3RA+)',
    'Eosinophils': 'Eosinophil infiltration: parasite defense, allergic response; rare in most tumors',
    'Macrophages': 'Macrophage infiltration: phagocytes; M1 = anti-tumor, M2 = pro-tumor',
    'Mast_cells': 'Mast cell infiltration: histamine release, allergic response; angiogenesis in tumors',
    'Neutrophils': 'Neutrophil infiltration: first-responder innate immunity; N1/N2 polarization in tumors',
    'ESTIMATE_stromal': 'ESTIMATE stromal score: abundance of stromal (fibroblast, endothelial) cells',
    'ESTIMATE_immune': 'ESTIMATE immune score: overall immune cell abundance in microenvironment',
    'ESTIMATE_purity': 'ESTIMATE tumor purity: cos-transformed stromal+immune; lower = more immune/stromal infiltration',
}
FEATURE_DESCRIPTIONS.update(_IMMUNE_DESCRIPTIONS)

# ---------------------------------------------------------------------------
# PCA components — generic descriptions
# ---------------------------------------------------------------------------
for _i in range(1, 101):
    FEATURE_DESCRIPTIONS[f'PC_{_i}'] = (
        f'Principal component {_i}: linear combination of original features capturing '
        f'the {_i}{"st" if _i == 1 else "nd" if _i == 2 else "rd" if _i == 3 else "th"} '
        f'most variance in the representation'
    )

# ---------------------------------------------------------------------------
# One-hot / BLOSUM positional features — category-level description only
# (5600D is too many for individual entries; handled at category level in hybrid.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Chromosome arm features — category-level
# ---------------------------------------------------------------------------
# These are dynamically generated arm names like 'chr1p', 'chr1q', etc.
# We add a category-level helper instead.

def _get_chr_arm_description(arm_name: str) -> str:
    """Generate description for a chromosome arm feature."""
    return (
        f'Mean expression of genes on {arm_name}: proxy for arm-level copy number '
        f'alteration (gain = overexpression, loss = underexpression); captures '
        f'chromosomal instability'
    )


def _get_pathway_description(pathway_name: str, source: str) -> str:
    """Generate description for a pathway activity feature."""
    source_desc = {
        'hallmark': 'MSigDB Hallmark gene set (mean z-scored expression of member genes)',
        'kegg': 'KEGG pathway (mean z-scored expression of member genes)',
        'reactome': 'Reactome pathway (mean z-scored expression of member genes)',
        'oncogenic': 'MSigDB C6 oncogenic signature (mean z-scored expression)',
        'progeny': 'PROGENy pathway (weighted linear model of target gene expression)',
        'dorothea': 'DoRothEA regulon (transcription factor activity from target gene expression)',
    }.get(source, f'{source} pathway score')
    return f'{pathway_name}: {source_desc}'
