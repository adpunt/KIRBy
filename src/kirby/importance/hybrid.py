"""
Hybrid Feature Explainability

Maps hybrid feature importance back to interpretable units:
- ECFP bits → atom environments / SMARTS substructures
- PDV indices → descriptor names (MolLogP, TPSA, etc.)
- Embeddings → aggregated representation-level importance

Usage:
    from kirby.importance.hybrid import explain_hybrid

    explanations = explain_hybrid(
        feature_info=feature_info,
        smiles_list=smiles,
        importances=shap_values,  # optional
    )
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple


# =============================================================================
# MACCS KEY DEFINITIONS (166 keys from RDKit)
# =============================================================================

# Key definitions: {key_idx: (SMARTS, short_name, description)}
# Source: https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/MACCSkeys.py
MACCS_KEY_DEFINITIONS = {
    1: ('?', 'ISOTOPE', 'Isotope'),
    2: ('[#104]', '104', 'Atomic number > 103'),
    3: ('[#32,#33,#34,#50,#51,#52,#82,#83,#84]', 'Group IVa,Va,VIa', 'Ge, As, Se, Sn, Sb, Te, Pb, Bi, Po'),
    4: ('[Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr]', 'Actinide', 'Actinide'),
    5: ('[Sc,Ti,Y,Zr,Hf]', 'Group IIIb,IVb', 'Sc, Ti, Y, Zr, Hf'),
    6: ('[La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]', 'Lanthanide', 'Lanthanide'),
    7: ('[V,Cr,Mn,Nb,Mo,Tc,Ta,W,Re]', 'Group Vb,VIb,VIIb', 'V, Cr, Mn, Nb, Mo, Tc, Ta, W, Re'),
    8: ('[!#6;!#1]1~*~*~*~1', 'QAAA@1', '4-membered ring with heteroatom'),
    9: ('[Fe,Co,Ni,Ru,Rh,Pd,Os,Ir,Pt]', 'Group VIII', 'Fe, Co, Ni, Ru, Rh, Pd, Os, Ir, Pt'),
    10: ('[Be,Mg,Ca,Sr,Ba,Ra]', 'Group IIa', 'Alkaline earth'),
    11: ('*1~*~*~*~1', '4-Ring', '4-membered ring'),
    12: ('[Cu,Zn,Ag,Cd,Au,Hg]', 'Group Ib,IIb', 'Cu, Zn, Ag, Cd, Au, Hg'),
    13: ('[#8]~[#7](~[#6])~[#6]', 'ON(C)C', 'N with O and two C'),
    14: ('[#16]-[#16]', 'S-S', 'Disulfide'),
    15: ('[#8]~[#6](~[#8])~[#8]', 'OC(O)O', 'Carbonate/carboxyl'),
    16: ('[!#6;!#1]1~*~*~1', 'QAA@1', '3-membered ring with heteroatom'),
    17: ('[#6]#[#6]', 'CTC', 'Alkyne C≡C'),
    18: ('[#5,#13,#31,#49,#81]', 'Group IIIa', 'B, Al, Ga, In, Tl'),
    19: ('*1~*~*~*~*~*~*~1', '7-Ring', '7-membered ring'),
    20: ('[#14]', 'Si', 'Silicon'),
    21: ('[#6]=[#6](~[!#6;!#1])~[!#6;!#1]', 'C=C(Q)Q', 'Vinyl with 2 heteroatoms'),
    22: ('*1~*~*~1', '3-Ring', '3-membered ring'),
    23: ('[#7]~[#6](~[#8])~[#8]', 'NC(O)O', 'Carbamate'),
    24: ('[#7]-[#8]', 'N-O', 'N-O single bond'),
    25: ('[#7]~[#6](~[#7])~[#7]', 'NC(N)N', 'Guanidine-like'),
    26: ('[#6]=[#6](~[#6])~[#6]', 'C=C(C)C', 'Tetrasubstituted alkene'),
    27: ('[#7]~[#6](~[#8])~[#7]', 'NC(O)N', 'Urea'),
    28: ('[#7]~[#6](~[#6])~[#7]', 'NC(C)N', 'Amidine'),
    29: ('[#8]~[#16](~[#8])~[#8]', 'OS(O)O', 'Sulfate/sulfonate'),
    30: ('[#16]-[#8]', 'S-O', 'S-O single bond'),
    31: ('[#6]#[#7]', 'CTN', 'Nitrile'),
    32: ('F', 'F', 'Fluorine'),
    33: ('[!#6;!#1;!H0]~*~[!#6;!#1;!H0]', 'QHAQH', 'Two heteroatoms with H'),
    34: ('?', 'OTHER', 'Other'),
    35: ('[#6]=[#6]~[#7]', 'C=CN', 'Enamine/vinyl amine'),
    36: ('Br', 'Br', 'Bromine'),
    37: ('[#16]~*~[#7]', 'SAN', 'S-A-N chain'),
    38: ('[#8]~[!#6;!#1](~[#8])(~[#8])', 'OQ(O)O', 'Central heteroatom with 3 O'),
    39: ('[!+0]', 'Charge', 'Charged atom'),
    40: ('[#6]=[#6](~[#6])~[#7]', 'C=C(C)N', 'Enamine'),
    41: ('[#6]~[#16]~[#7]', 'CSN', 'C-S-N chain'),
    42: ('[#7]~[#16]', 'N-S', 'N-S bond'),
    43: ('[CH2]=*', 'CH2=A', 'Methylene'),
    44: ('[Li,Na,K,Rb,Cs,Fr]', 'Group Ia', 'Alkali metal'),
    45: ('[#16]~*~*~[#16]', 'SAAS', 'Two S separated by 2 atoms'),
    46: ('[#16]~*~*~[#8]', 'SAAO', 'S and O separated by 2 atoms'),
    47: ('[#8]~[#16]~[#8]', 'OSO', 'Sulfoxide/sulfone'),
    48: ('[#7]~*~*~[#7]', 'NAAN', 'Two N separated by 2 atoms'),
    49: ('[#7]~*~*~[#8]', 'NAAO', 'N and O separated by 2 atoms'),
    50: ('[!#6;!#1]1~*~*~*~*~*~1', 'QAAAAA@1', '6-ring with heteroatom'),
    51: ('I', 'I', 'Iodine'),
    52: ('[#7]~[#7]', 'N-N', 'N-N bond'),
    53: ('[!#6;!#1;!H0]~*~*~[!#6;!#1;!H0]', 'QHAAQH', 'QH-A-A-QH'),
    54: ('[!#6;!#1]~*~*~*~[!#6;!#1]', 'QAAAQ', 'Heteroatoms 3 apart'),
    55: ('[#8]~*~*~[#8]', 'OAAO', 'Two O separated by 2 atoms'),
    56: ('[#16]=*', 'S=A', 'Thione'),
    57: ('[CH3]~*~[CH3]', 'ACH3ACH3', 'Two methyl groups'),
    58: ('*~[!#6;!#1](~*)~*', 'QA(A)A', 'Trivalent heteroatom'),
    59: ('[!#6;!#1;!H0]~*~[CH2]~*', 'QHACH2A', 'QH-A-CH2-A'),
    60: ('[!#6;!#1]~[#6]~[#7]', 'QCN', 'Heteroatom-C-N'),
    61: ('[!#6;!#1]~[CH2]~*', 'QACH2A', 'Q-CH2-A'),
    62: ('[!#6;!#1]~[!#6;!#1]', 'Q-Q', 'Adjacent heteroatoms'),
    63: ('[!#6;!#1;!H0]', 'QH', 'Heteroatom with H'),
    64: ('[#6]=[#8]', 'C=O', 'Carbonyl'),
    65: ('*!@[CH2]!@*', 'ACH2A', 'Non-ring CH2'),
    66: ('[#7]~*~[#6]=,:[#8]', 'NAC=O', 'Amide N'),
    67: ('[!#6;!#1]~*~*~[CH2]~*', 'QAACH2A', 'Q-A-A-CH2-A'),
    68: ('[#8]=*', 'O=A', 'Oxo'),
    69: ('[!#6;!#1;!H0]~*~*~*~[CH2]~*', 'QAAACH2A', 'Long chain'),
    70: ('[#8]~[#6](~[#6])~[#6]', 'OC(C)C', 'Ether'),
    71: ('[!#6;!#1]~[CH3]', 'QCH3', 'Heteroatom-methyl'),
    72: ('[#7]~*~*~[#6]=,:[#8]', 'NAAC=O', 'N two bonds from carbonyl'),
    73: ('*1~*~*~*~*~1', '5-Ring', '5-membered ring'),
    74: ('[#7]-[#8]', 'N-O', 'N-O bond'),
    75: ('[#7]~*~[CH2]~*', 'NACH2A', 'N-A-CH2-A'),
    76: ('[#8]=*', 'O=A', 'Oxo'),
    77: ('[!#6;!#1]~[CH2]~[CH2]~*', 'QCH2CH2A', 'Q-CH2-CH2-A'),
    78: ('[!#6;!#1]~*(~[!#6;!#1])~[!#6;!#1]', 'QQ(Q)Q', 'Heteroatom with 3 hetero neighbors'),
    79: ('[!#6;!#1;!H0]~*~[!#6;!#1;!H0]', 'QHAQH', 'Two QH one apart'),
    80: ('[!#6;!#1]~*~[CH2]~[CH2]~*', 'QACH2CH2A', 'Q-A-CH2-CH2-A'),
    81: ('[#8]~*~[CH2]~*', 'OACH2A', 'O-A-CH2-A'),
    82: ('*~[CH2]~[!#6;!#1;!H0]', 'ACH2QH', 'CH2 adjacent to heteroatom-H'),
    83: ('[#7]~[#6](~[#6])~[#6]', 'NC(C)C', 'Tertiary amine'),
    84: ('[#6]=[#6]', 'C=C', 'Alkene'),
    85: ('[#7]~[#6]~[#8]', 'NCO', 'N-C-O'),
    86: ('[#7]~[#6]~[#7]', 'NCN', 'N-C-N'),
    87: ('[#6]~[#8]~[#6]', 'COC', 'Ether C-O-C'),
    88: ('[#6]~[#7]~[#6]', 'CNC', 'Amine C-N-C'),
    89: ('[#8]~[#6]~[#8]', 'OCO', 'Acetal'),
    90: ('[#16]~[#6]~[#6]', 'SCC', 'S-C-C'),
    91: ('[#7]~[#6]~[#6]', 'NCC', 'N-C-C'),
    92: ('[#6]~[#6]~[#7]', 'CCN', 'C-C-N'),
    93: ('[#6]~[#8]~[#7]', 'CON', 'C-O-N'),
    94: ('[!#6;!#1]~[#6]~[#6]', 'QCC', 'Heteroatom-C-C'),
    95: ('[!#6;!#1]~[#8]', 'QO', 'Heteroatom-O'),
    96: ('Cl', 'Cl', 'Chlorine'),
    97: ('[!#6;!#1;!H0]~[#6]~[#6]', 'QHCC', 'QH-C-C'),
    98: ('[!#6;!#1;!H0]~[!#6;!#1;!H0]', 'QHQH', 'Adjacent QH groups'),
    99: ('[!#6;!#1]~[#6](~[#6])~[#6]', 'QC(C)C', 'Heteroatom on tertiary C'),
    100: ('[!#6;!#1]~*~[#6]~[#6]', 'QACC', 'Q-A-C-C'),
    101: ('[!#6;!#1;!H0]~*~*~[#6]~[#6]', 'QHAACC', 'QH-A-A-C-C'),
    102: ('[!#6;!#1]~*~*~*~[#6]~[#6]', 'QAAACC', 'Q-A-A-A-C-C'),
    103: ('[#8]~[#6]~*~[#6]', 'OCAC', 'O-C-A-C'),
    104: ('[!#6;!#1]~*~*~[#6]', 'QAAC', 'Q-A-A-C'),
    105: ('[#7]~*~*~[#6]', 'NAAC', 'N-A-A-C'),
    106: ('[#8]~[#6]~[#6]', 'OCC', 'O-C-C'),
    107: ('[F,Cl,Br,I]~*(~*)~*', 'XA(A)A', 'Halogen on branched atom'),
    108: ('[#6]~[#6]~[#6]~[#6]', 'CCCC', '4-carbon chain'),
    109: ('[#16]~*~*~[#6]', 'SAAC', 'S-A-A-C'),
    110: ('[!#6;!#1;!H0]~*~*~[#6]', 'QHAAC', 'QH-A-A-C'),
    111: ('[!#6;!#1]~[#6]~*~*~[#6]', 'QCAAC', 'Q-C-A-A-C'),
    112: ('[#7]~*~[#8]', 'NAO', 'N-A-O'),
    113: ('[!#6;!#1]~*~[#6]~*~*~[#6]', 'QACAAC', 'Q-A-C-A-A-C'),
    114: ('[#8]~[#6]~[#6]~[#7]', 'OCCN', 'O-C-C-N'),
    115: ('[#8]~[#6]~[#6]~[#8]', 'OCCO', 'O-C-C-O (glycol)'),
    116: ('[#7]~*~*~*~[#8]', 'NAAAO', 'N-A-A-A-O'),
    117: ('[#7]~[#6]~[#6]~[#6]', 'NCCC', 'N-C-C-C'),
    118: ('[#7]~[#6]~[#6]~[#8]', 'NCCO', 'N-C-C-O'),
    119: ('[#16]~*~[#6]', 'SAC', 'S-A-C'),
    120: ('[!#6;!#1]~*~*~*~[#6]', 'QAAAC', 'Q-A-A-A-C'),
    121: ('[#8]~*~*~[#8]', 'OAAO', 'O-A-A-O'),
    122: ('[!#6;!#1]~[#6]~*~[#6]', 'QCAC', 'Q-C-A-C'),
    123: ('[#7]~*~[#6]', 'NAC', 'N-A-C'),
    124: ('[#6]~[#16]~[#6]', 'CSC', 'Thioether'),
    125: ('?', 'Count>=4', '4+ rings'),
    126: ('[#8]~*~[#8]', 'OAO', 'O-A-O'),
    127: ('[!#6;!#1]~[CH2]~*~[CH2]~[!#6;!#1]', 'QCH2ACH2Q', 'Q-CH2-A-CH2-Q'),
    128: ('[!#6;!#1]~*~[#6]~[#7]', 'QACN', 'Q-A-C-N'),
    129: ('[#16]~[#6]~[#7]', 'SCN', 'S-C-N'),
    130: ('[!#6;!#1]~*~[#6]~[#8]', 'QACO', 'Q-A-C-O'),
    131: ('[#8]~*~[#7]', 'OAN', 'O-A-N'),
    132: ('[CH3]~[CH2]~*', 'CH3CH2A', 'Ethyl'),
    133: ('*~[CH2]~[CH2]~*', 'ACH2CH2A', 'Ethylene bridge'),
    134: ('[#7]~*~[#6]~[#8]', 'NACO', 'N-A-C-O'),
    135: ('[!#6;!#1]~*~[#6]~[#6]', 'QACC', 'Q-A-C-C'),
    136: ('[#8]~[#6]~*~[#7]', 'OCAN', 'O-C-A-N'),
    137: ('[#7]~*~[#6]~[#6]', 'NACC', 'N-A-C-C'),
    138: ('[#6]~[#6]~*~[#6]', 'CCAC', 'C-C-A-C'),
    139: ('[#7]=*', 'N=A', 'Imine'),
    140: ('[!#6;!#1]~*~*~[!#6;!#1;!H0]', 'QAAQH', 'Q-A-A-QH'),
    141: ('[!#6;!#1]~[#6]~[#6]~[#6]', 'QCCC', 'Q-C-C-C'),
    142: ('[!#6;!#1]~[#6]~[#6]~[!#6;!#1]', 'QCCQ', 'Heteroatoms bridged by CC'),
    143: ('[#8]~*~*~*~[#8]', 'OAAAO', 'O-A-A-A-O'),
    144: ('[!#6;!#1]~[#6]~[#6]~[#7]', 'QCCN', 'Q-C-C-N'),
    145: ('[!#6;!#1]~[#6]~[#6]~[#8]', 'QCCO', 'Q-C-C-O'),
    146: ('*~[CH2]~*~[CH2]~*', 'ACH2ACH2A', 'Two CH2 groups'),
    147: ('[#8]~*~*~[#6]', 'OAAC', 'O-A-A-C'),
    148: ('[!#6;!#1]~*~[#8]', 'QAO', 'Q-A-O'),
    149: ('[#6]=*~[#6]', 'C=AC', 'Vinyl'),
    150: ('[#7]~*~*~*~[#7]', 'NAAAN', 'N-A-A-A-N'),
    151: ('[#7]~*~*~*~[#6]', 'NAAAC', 'N-A-A-A-C'),
    152: ('[#8]~[#6]~[#6]~[#6]', 'OCCC', 'O-C-C-C'),
    153: ('[!#6;!#1]~*~[!#6;!#1]', 'QAQ', 'Q-A-Q'),
    154: ('[#6]=[#8]', 'C=O', 'Carbonyl'),
    155: ('[!#6;!#1]~*~*~*~[!#6;!#1]', 'QAAAQ', 'Q-A-A-A-Q'),
    156: ('[!#6;!#1;!H0]~[#6]~*~[#6]~[!#6;!#1;!H0]', 'QHCACQH', 'QH-C-A-C-QH'),
    157: ('[!#6;!#1]~[#6](~[#6])~[#6]', 'QC(C)C', 'Branched heteroatom'),
    158: ('[!#6;!#1]~*~[#6]~*~[#6]', 'QACAC', 'Q-A-C-A-C'),
    159: ('[!#6;!#1]~[#6]~[#8]', 'QCO', 'Q-C-O'),
    160: ('[!#6;!#1]~[CH2]~[!#6;!#1]', 'QCH2Q', 'Heteroatoms bridged by CH2'),
    161: ('[#6]~[#6]~[#6]', 'CCC', '3-carbon chain'),
    162: ('[#8]~*~[#6]', 'OAC', 'O-A-C'),
    163: ('[!#6;!#1]~[#6]~[#6]', 'QCC', 'Q-C-C'),
    164: ('[#7]~*~[#7]', 'NAN', 'N-A-N'),
    165: ('*1~*~*~*~*~*~1', '6-Ring', '6-membered ring'),
    166: ('?', 'Fragment', 'Multiple fragments'),
}


def interpret_maccs_key(key_idx: int) -> Dict[str, str]:
    """Map a MACCS key index to its definition."""
    if key_idx in MACCS_KEY_DEFINITIONS:
        smarts, short_name, description = MACCS_KEY_DEFINITIONS[key_idx]
        return {
            'key_idx': key_idx,
            'smarts': smarts,
            'short_name': short_name,
            'description': description,
        }
    return {
        'key_idx': key_idx,
        'smarts': '?',
        'short_name': f'Key {key_idx}',
        'description': f'MACCS key {key_idx}',
    }


# =============================================================================
# DESCRIPTOR NAME MAPPING
# =============================================================================

def get_pdv_descriptor_names():
    """Get the list of PDV descriptor names in order."""
    from kirby.representations.molecular import DEFAULT_DESCRIPTOR_LIST
    return DEFAULT_DESCRIPTOR_LIST


def interpret_pdv_index(idx: int) -> str:
    """Map a PDV feature index to its descriptor name."""
    names = get_pdv_descriptor_names()
    if 0 <= idx < len(names):
        return names[idx]
    return f"pdv_feature_{idx}"


# =============================================================================
# ECFP BIT INTERPRETATION
# =============================================================================

def interpret_ecfp_bit(bit_idx: int, smiles: str, radius: int = 2, n_bits: int = 2048) -> Dict[str, Any]:
    """
    Map an ECFP bit index to its atom environment for a specific molecule.

    Args:
        bit_idx: The fingerprint bit index
        smiles: SMILES string of the molecule
        radius: Fingerprint radius (2 for ECFP4)
        n_bits: Number of bits in fingerprint

    Returns:
        dict with keys:
            - 'bit_idx': The bit index
            - 'active': Whether this bit is set in this molecule
            - 'atom_indices': List of (atom_idx, env_radius) tuples
            - 'smarts': SMARTS pattern (if extractable)
            - 'atom_symbols': List of atom symbols involved
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return {'bit_idx': bit_idx, 'active': False, 'error': 'RDKit not available'}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'bit_idx': bit_idx, 'active': False, 'error': 'Invalid SMILES'}

    bit_info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, bitInfo=bit_info)

    if bit_idx not in bit_info:
        return {
            'bit_idx': bit_idx,
            'active': False,
            'atom_indices': [],
            'smarts': None,
            'atom_symbols': []
        }

    atom_envs = bit_info[bit_idx]
    atom_symbols = []
    smarts_list = []

    for atom_idx, env_radius in atom_envs:
        # Get atom symbol
        atom = mol.GetAtomWithIdx(atom_idx)
        atom_symbols.append(atom.GetSymbol())

        # Try to get SMARTS
        try:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, env_radius, atom_idx)
            atoms = set()
            for bond_idx in env:
                bond = mol.GetBondWithIdx(bond_idx)
                atoms.add(bond.GetBeginAtomIdx())
                atoms.add(bond.GetEndAtomIdx())
            if not atoms:
                atoms.add(atom_idx)
            smarts = Chem.MolFragmentToSmiles(mol, list(atoms), canonical=True)
            smarts_list.append(smarts)
        except:
            pass

    return {
        'bit_idx': bit_idx,
        'active': True,
        'atom_indices': list(atom_envs),
        'smarts': smarts_list[0] if smarts_list else None,
        'atom_symbols': atom_symbols
    }


def draw_ecfp_bit(bit_idx: int, smiles: str, radius: int = 2, n_bits: int = 2048):
    """
    Generate an image highlighting the atom environment for an ECFP bit.

    Args:
        bit_idx: The fingerprint bit index
        smiles: SMILES string
        radius: Fingerprint radius
        n_bits: Number of bits

    Returns:
        PIL Image or SVG string
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Draw
    except ImportError:
        raise ImportError("RDKit required for visualization")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    bit_info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, bitInfo=bit_info)

    if bit_idx not in bit_info:
        # Bit not active - just draw the molecule
        return Draw.MolToImage(mol)

    # Use RDKit's DrawMorganBit if available
    try:
        return Draw.DrawMorganBit(mol, bit_idx, bit_info)
    except AttributeError:
        # Fallback: highlight atoms manually
        atom_indices = set()
        for atom_idx, _ in bit_info[bit_idx]:
            atom_indices.add(atom_idx)
        return Draw.MolToImage(mol, highlightAtoms=list(atom_indices))


# =============================================================================
# REPRESENTATION-LEVEL AGGREGATION
# =============================================================================

def aggregate_importance_by_rep(
    feature_info: Dict[str, Any],
    importances: np.ndarray
) -> Dict[str, float]:
    """
    Aggregate feature importances by representation source.

    Args:
        feature_info: The feature_info dict from create_hybrid()
        importances: Array of importance scores for the hybrid features
            (must match the order of concatenated features)

    Returns:
        dict: {rep_name: total_importance}
    """
    rep_importance = {}
    current_idx = 0

    for rep_name, info in feature_info.items():
        # Skip metadata keys
        if not isinstance(info, dict) or 'selected_indices' not in info:
            continue

        n_features = info['n_features']
        rep_importances = importances[current_idx:current_idx + n_features]
        rep_importance[rep_name] = float(np.sum(np.abs(rep_importances)))
        current_idx += n_features

    # Normalize to sum to 1
    total = sum(rep_importance.values())
    if total > 0:
        rep_importance = {k: v / total for k, v in rep_importance.items()}

    return rep_importance


# =============================================================================
# MAIN EXPLAINABILITY FUNCTION
# =============================================================================

def explain_hybrid(
    feature_info: Dict[str, Any],
    smiles_list: Optional[List[str]] = None,
    importances: Optional[np.ndarray] = None,
    top_k: int = 20,
    ecfp_radius: int = 2,
    ecfp_n_bits: int = 2048,
) -> Dict[str, Any]:
    """
    Generate interpretable explanations for a hybrid representation.

    Args:
        feature_info: The feature_info dict from create_hybrid()
        smiles_list: Optional list of SMILES (needed for ECFP interpretation)
        importances: Optional array of feature importances for the hybrid
        top_k: Number of top features to explain
        ecfp_radius: ECFP radius (2 for ECFP4)
        ecfp_n_bits: Number of ECFP bits

    Returns:
        dict with keys:
            - 'by_representation': {rep_name: {'n_features', 'importance', 'pct'}}
            - 'top_features': List of top feature explanations
            - 'feature_map': Full mapping of hybrid indices to interpretations
    """
    result = {
        'by_representation': {},
        'top_features': [],
        'feature_map': [],
    }

    # Build feature map: hybrid_idx -> (rep_name, local_idx, interpretation)
    current_idx = 0
    feature_map = []

    # Get PDV descriptor names once
    pdv_names = None

    for rep_name, info in feature_info.items():
        # Skip metadata keys
        if not isinstance(info, dict) or 'selected_indices' not in info:
            continue

        selected_indices = info['selected_indices']
        n_features = len(selected_indices)
        rep_importances = info.get('importance_scores', np.ones(n_features))

        # Track representation stats
        result['by_representation'][rep_name] = {
            'n_features': n_features,
            'selected_indices': selected_indices.tolist() if hasattr(selected_indices, 'tolist') else list(selected_indices),
        }

        # Map each feature
        for i, local_idx in enumerate(selected_indices):
            local_idx = int(local_idx)
            importance = float(rep_importances[i]) if i < len(rep_importances) else 0.0

            # Get interpretation based on representation type
            interpretation = None
            rep_lower = rep_name.lower()

            if 'ecfp' in rep_lower or 'morgan' in rep_lower:
                interpretation = {
                    'type': 'ecfp_bit',
                    'bit_idx': local_idx,
                    'description': f"ECFP bit {local_idx}"
                }
                # Try multiple molecules to find SMARTS for this bit
                if smiles_list:
                    for smi in smiles_list[:50]:  # Check up to 50 molecules
                        bit_info = interpret_ecfp_bit(local_idx, smi, ecfp_radius, ecfp_n_bits)
                        if bit_info.get('smarts'):
                            interpretation['smarts'] = bit_info['smarts']
                            interpretation['description'] = f"ECFP: {bit_info['smarts']}"
                            break

            elif 'pdv' in rep_lower or 'descriptor' in rep_lower:
                if pdv_names is None:
                    pdv_names = get_pdv_descriptor_names()
                desc_name = pdv_names[local_idx] if local_idx < len(pdv_names) else f"descriptor_{local_idx}"
                interpretation = {
                    'type': 'descriptor',
                    'name': desc_name,
                    'description': f"PDV: {desc_name}"
                }

            elif 'maccs' in rep_lower:
                maccs_info = interpret_maccs_key(local_idx)
                interpretation = {
                    'type': 'maccs_key',
                    'key_idx': local_idx,
                    'smarts': maccs_info['smarts'],
                    'short_name': maccs_info['short_name'],
                    'description': f"MACCS: {maccs_info['short_name']} ({maccs_info['description']})"
                }

            else:
                # Embedding or unknown - just use index
                interpretation = {
                    'type': 'embedding_dim',
                    'dim_idx': local_idx,
                    'description': f"{rep_name} dim {local_idx}"
                }

            feature_map.append({
                'hybrid_idx': current_idx,
                'rep_name': rep_name,
                'local_idx': local_idx,
                'importance': importance,
                'interpretation': interpretation,
            })

            current_idx += 1

    result['feature_map'] = feature_map

    # If we have external importances, use those; otherwise use stored importances
    if importances is not None:
        for i, fm in enumerate(feature_map):
            if i < len(importances):
                fm['importance'] = float(np.abs(importances[i]))

    # Sort by importance and get top_k
    sorted_features = sorted(feature_map, key=lambda x: x['importance'], reverse=True)
    result['top_features'] = sorted_features[:top_k]

    # Compute representation-level importance
    for rep_name in result['by_representation']:
        rep_features = [f for f in feature_map if f['rep_name'] == rep_name]
        total_imp = sum(f['importance'] for f in rep_features)
        result['by_representation'][rep_name]['total_importance'] = total_imp

    # Normalize to percentages
    total = sum(r['total_importance'] for r in result['by_representation'].values())
    if total > 0:
        for rep_name in result['by_representation']:
            result['by_representation'][rep_name]['importance_pct'] = (
                result['by_representation'][rep_name]['total_importance'] / total * 100
            )

    return result


def format_explanation_summary(explanation: Dict[str, Any], max_features: int = 10) -> str:
    """
    Format explanation as a human-readable string.

    Args:
        explanation: Output from explain_hybrid()
        max_features: Max features to show

    Returns:
        Formatted string
    """
    lines = []

    lines.append("=== REPRESENTATION IMPORTANCE ===")
    by_rep = explanation['by_representation']
    sorted_reps = sorted(by_rep.items(), key=lambda x: x[1].get('total_importance', 0), reverse=True)

    for rep_name, info in sorted_reps:
        pct = info.get('importance_pct', 0)
        n_feat = info.get('n_features', 0)
        lines.append(f"  {rep_name}: {pct:.1f}% ({n_feat} features)")

    lines.append("")
    lines.append("=== TOP FEATURES ===")

    for i, feat in enumerate(explanation['top_features'][:max_features]):
        interp = feat['interpretation']
        desc = interp.get('description', f"{feat['rep_name']}[{feat['local_idx']}]")
        lines.append(f"  {i+1}. {desc} (importance: {feat['importance']:.4f})")

    return "\n".join(lines)


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_molecule_importance(
    smiles: str,
    feature_info: Dict[str, Any],
    importances: Optional[np.ndarray] = None,
    ecfp_radius: int = 2,
    ecfp_n_bits: int = 2048,
):
    """
    Visualize a molecule with atom-level importance highlighting.

    Maps ECFP bit importances back to atoms and highlights them.

    Args:
        smiles: SMILES string
        feature_info: The feature_info dict from create_hybrid()
        importances: Optional importance scores for hybrid features
        ecfp_radius: ECFP radius
        ecfp_n_bits: Number of ECFP bits

    Returns:
        PIL Image with highlighted atoms
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Draw
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        raise ImportError("RDKit required for visualization")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Get bit info for this molecule
    bit_info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius=ecfp_radius, nBits=ecfp_n_bits, bitInfo=bit_info)

    # Accumulate importance per atom
    atom_importance = np.zeros(mol.GetNumAtoms())

    # Find ECFP in feature_info
    current_idx = 0
    for rep_name, info in feature_info.items():
        if not isinstance(info, dict) or 'selected_indices' not in info:
            continue

        rep_lower = rep_name.lower()
        selected_indices = info['selected_indices']
        n_features = len(selected_indices)

        if 'ecfp' in rep_lower or 'morgan' in rep_lower:
            # Get importances for this rep
            if importances is not None:
                rep_imp = importances[current_idx:current_idx + n_features]
            else:
                rep_imp = info.get('importance_scores', np.ones(n_features))

            # Map bits to atoms
            for i, bit_idx in enumerate(selected_indices):
                bit_idx = int(bit_idx)
                if bit_idx in bit_info:
                    imp = float(np.abs(rep_imp[i])) if i < len(rep_imp) else 0.0
                    for atom_idx, _ in bit_info[bit_idx]:
                        atom_importance[atom_idx] += imp

        current_idx += n_features

    # Normalize importance to [0, 1]
    if atom_importance.max() > 0:
        atom_importance = atom_importance / atom_importance.max()

    # Create color map (white to red)
    highlight_atoms = []
    highlight_colors = {}

    for atom_idx, imp in enumerate(atom_importance):
        if imp > 0.1:  # Threshold for highlighting
            highlight_atoms.append(atom_idx)
            # Interpolate from light yellow to red
            r = 1.0
            g = 1.0 - imp * 0.8
            b = 0.2
            highlight_colors[atom_idx] = (r, g, b)

    # Draw
    return Draw.MolToImage(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        size=(400, 400)
    )
