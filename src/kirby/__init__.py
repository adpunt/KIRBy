"""
KIRBy: Knowledge Integration by Representation Borrowing

Usage:
    from kirby.representations.molecular import create_ecfp4, create_pdv
    from kirby.hybrid import create_hybrid
    
    ecfp4 = create_ecfp4(smiles_list)
    pdv = create_pdv(smiles_list)
    hybrid, info = create_hybrid({'ecfp4': ecfp4, 'pdv': pdv}, labels)
"""

__version__ = '0.2.0'

__all__ = []