"""
KIRBy: Knowledge Integration by Representation Borrowing

Usage:
    from kirby.representations.molecular import create_ecfp4, create_pdv
    from kirby.hybrid import create_hybrid

    ecfp4 = create_ecfp4(smiles_list)
    pdv = create_pdv(smiles_list)
    hybrid, info = create_hybrid({'ecfp4': ecfp4, 'pdv': pdv}, labels)
"""

# Block stale system bottleneck (ARC Anaconda 2022.05, compiled against old numpy).
# Its _ARRAY_API attribute is missing with newer numpy, crashing pandas on import.
# setdefault only acts if bottleneck hasn't been imported yet.
import sys as _sys
_sys.modules.setdefault('bottleneck', None)
del _sys

__version__ = '0.2.0'

__all__ = []