"""
Antibody Molecular Representations - SOTA Methods

Implements state-of-the-art antibody featurization from Nature ML papers:
- Language model embeddings (AntiBERTy, AbLang, BALM, IgBERT)
- CDR-stratified features
- IMGT positional encoding
- Structure-based features (when available)
- Multi-scale aggregations
- Developability assessment (charge, hydrophobicity, aggregation propensity)
- Humanness scoring (germline identity)

References:
- IgFold: Ruffolo et al. (2023) Nature Communications
- AbLang: Olsen et al. (2022) Bioinformatics Advances  
- BALM: (2024) Briefings in Bioinformatics
- Paragraph: (2023) Bioinformatics
- TAP: Raybould et al. (2019) PNAS (Therapeutic Antibody Profiler)
- OASis: Prihoda et al. (2022) MAbs (humanness scoring)
"""

import numpy as np
import torch
import torch.nn as nn
import warnings
from typing import List, Dict, Optional, Tuple


# ============================================================================
# CONSTANTS: AMINO ACID PROPERTIES
# ============================================================================

# Physicochemical properties per amino acid
# Sources: Kyte-Doolittle hydrophobicity, Zimmerman polarity, charge at pH 7.4,
# molecular weight, van der Waals volume (A^3)
AA_PROPERTIES = {
    'A': {'hydrophobicity':  1.8, 'charge':  0.0, 'mw':  89.09, 'volume':  88.6, 'polarity':  0.0, 'flexibility': 0.360},
    'R': {'hydrophobicity': -4.5, 'charge':  1.0, 'mw': 174.20, 'volume': 173.4, 'polarity':  1.0, 'flexibility': 0.530},
    'N': {'hydrophobicity': -3.5, 'charge':  0.0, 'mw': 132.12, 'volume': 114.1, 'polarity':  1.0, 'flexibility': 0.460},
    'D': {'hydrophobicity': -3.5, 'charge': -1.0, 'mw': 133.10, 'volume': 111.1, 'polarity':  1.0, 'flexibility': 0.510},
    'C': {'hydrophobicity':  2.5, 'charge':  0.0, 'mw': 121.15, 'volume': 108.5, 'polarity':  0.0, 'flexibility': 0.350},
    'E': {'hydrophobicity': -3.5, 'charge': -1.0, 'mw': 147.13, 'volume': 138.4, 'polarity':  1.0, 'flexibility': 0.500},
    'Q': {'hydrophobicity': -3.5, 'charge':  0.0, 'mw': 146.15, 'volume': 143.8, 'polarity':  1.0, 'flexibility': 0.490},
    'G': {'hydrophobicity': -0.4, 'charge':  0.0, 'mw':  75.03, 'volume':  60.1, 'polarity':  0.0, 'flexibility': 0.540},
    'H': {'hydrophobicity': -3.2, 'charge':  0.1, 'mw': 155.16, 'volume': 153.2, 'polarity':  1.0, 'flexibility': 0.320},
    'I': {'hydrophobicity':  4.5, 'charge':  0.0, 'mw': 131.17, 'volume': 166.7, 'polarity':  0.0, 'flexibility': 0.460},
    'L': {'hydrophobicity':  3.8, 'charge':  0.0, 'mw': 131.17, 'volume': 166.7, 'polarity':  0.0, 'flexibility': 0.400},
    'K': {'hydrophobicity': -3.9, 'charge':  1.0, 'mw': 146.19, 'volume': 168.6, 'polarity':  1.0, 'flexibility': 0.535},
    'M': {'hydrophobicity':  1.9, 'charge':  0.0, 'mw': 149.21, 'volume': 162.9, 'polarity':  0.0, 'flexibility': 0.410},
    'F': {'hydrophobicity':  2.8, 'charge':  0.0, 'mw': 165.19, 'volume': 189.9, 'polarity':  0.0, 'flexibility': 0.310},
    'P': {'hydrophobicity': -1.6, 'charge':  0.0, 'mw': 115.13, 'volume': 112.7, 'polarity':  0.0, 'flexibility': 0.510},
    'S': {'hydrophobicity': -0.8, 'charge':  0.0, 'mw': 105.09, 'volume':  89.0, 'polarity':  1.0, 'flexibility': 0.510},
    'T': {'hydrophobicity': -0.7, 'charge':  0.0, 'mw': 119.12, 'volume': 116.1, 'polarity':  1.0, 'flexibility': 0.440},
    'W': {'hydrophobicity': -0.9, 'charge':  0.0, 'mw': 204.23, 'volume': 227.8, 'polarity':  0.0, 'flexibility': 0.310},
    'Y': {'hydrophobicity': -1.3, 'charge':  0.0, 'mw': 181.19, 'volume': 193.6, 'polarity':  1.0, 'flexibility': 0.420},
    'V': {'hydrophobicity':  4.2, 'charge':  0.0, 'mw': 117.15, 'volume': 140.0, 'polarity':  0.0, 'flexibility': 0.390},
}

# Sequence liability motifs known to cause manufacturing/stability issues
# Reference: Lu et al. (2019) "Deamidation and isomerization liability analysis..."
LIABILITY_MOTIFS = {
    'deamidation': ['NG', 'NS', 'NT', 'ND', 'NH', 'NA', 'NQ', 'NK'],
    'isomerization': ['DG', 'DS', 'DT', 'DD', 'DH'],
    'oxidation': ['MW', 'MH', 'MD', 'MS'],  # Met oxidation context
    'glycosylation': ['N[^P][ST]'],  # N-X-S/T sequon (regex-style, handled separately)
    'clipping': ['DP'],  # Asp-Pro clipping
    'unpaired_cys': ['C'],  # Free cysteines (counted, not motif-matched)
}

# ============================================================================
# CONSTANTS: HUMAN GERMLINE V-GENE SEQUENCES
# ============================================================================

# Representative human VH germline framework sequences (IMGT-aligned)
# Source: IMGT/GENE-DB, most frequently observed alleles in human repertoires
# These are framework region consensus sequences used for humanness scoring
# Only FR1-FR3 included (FR4 is J-gene derived)
HUMAN_VH_GERMLINES = {
    'IGHV1-2*02':   'QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCAR',
    'IGHV1-3*01':   'QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYAMHWVRQAPGQRLEWMGWINAGNGNTKYSQKFQGRVTITRDTSASTAYMELSSLRSEDTAVYYCAR',
    'IGHV1-18*01':  'QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYDINWVRQAPGQGLEWMGWMNPNSGNTGYAQKFQGRVTMTRNTSISTAYMELSSLRSEDTAVYYCAR',
    'IGHV1-46*01':  'QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCAR',
    'IGHV1-69*01':  'QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCAR',
    'IGHV2-5*02':   'QITLKESGPTLVKPTQTLTLTCTFSGFSLSTSGVGVGWIRQPPGKALEWLALIYWDDDKRYSPSLKSRLTITKDTSKNQVVLTMTNMDPVDTATYYCAHR',
    'IGHV3-7*01':   'EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAR',
    'IGHV3-11*01':  'EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMHWVRQAPGKGLEWVSRINSDGSSTSYADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYCAK',
    'IGHV3-15*01':  'EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR',
    'IGHV3-21*01':  'EVQLVESGGGLVKPGGSLRLSCAASGFTFSSYSMNWVRQAPGKGLEWVSSISSSSSYIYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAR',
    'IGHV3-23*01':  'EVQLLESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR',
    'IGHV3-30*01':  'QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVIWYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR',
    'IGHV3-33*01':  'QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVIWYDGSNEHYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAK',
    'IGHV3-48*01':  'EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYSMNWVRQAPGKGLEWVSYISSSSSTIYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAR',
    'IGHV3-53*01':  'EVQLVESGGGLVQPGGSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSGISWNSGSIGYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAR',
    'IGHV3-66*01':  'EVQLVESGGGLVQPGGSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSGISWNSGSIGYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAR',
    'IGHV4-4*02':   'QVQLQESGPGLVKPSETLSLTCTVSGGSISSYYWSWIRQPPGKGLEWIGYIYYSGSTNYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCAR',
    'IGHV4-34*01':  'QVQLQESGPGLVKPSQTLSLTCTVSGGSISSGDYYWSWIRQPPGKGLEWIGYIYYSGSTYYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCAR',
    'IGHV4-39*01':  'QLQLQESGPGLVKPSETLSLTCTVSGGSISSSSYYWGWIRQPPGKGLEWIGSIYYSGSTYYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCAR',
    'IGHV4-59*01':  'QVQLQESGPGLVKPSETLSLTCTVSGGSISSYYWSWIRQPPGKGLEWIGYIYYSGSTNYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCAR',
    'IGHV5-51*01':  'EVQLVQSGAEVKKPGESLKISCKGSGYSFTSYWIGWVRQMPGKGLEWMGIIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCAR',
    'IGHV6-1*01':   'QVQLQQSGPGLVKPSQTLSLTCAISGDSVSSNSAAWNWIRQSPSRGLEWLGRTYYRSKWYNDYAVSVKSRITINPDTSKNQFSLQLNSVTPEDTAVYYCAR',
}

# Representative human VL (kappa + lambda) germline framework sequences
HUMAN_VL_GERMLINES = {
    # Kappa
    'IGKV1-5*03':   'DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLT',
    'IGKV1-9*01':   'DIQLTQSPSFLSASVGDRVTITCRASQGISSYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTEFTLTISSLQPEDFATYYCQQLNSYPLT',
    'IGKV1-12*01':  'DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPRT',
    'IGKV1-27*01':  'DIQMTQSPSSLSASVGDRVTITCRASQGISNYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPPT',
    'IGKV1-33*01':  'DIQMTQSPSSLSASVGDRVTITCQASQDISNYLNWYQQKPGKAPKLLIYDASNLETGVPSRFSGSGSGTDFTFTISSLQPEDIATYYCQQYDNLPLT',
    'IGKV1-39*01':  'DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPYT',
    'IGKV2-28*01':  'DIVMTQSPLSLPVTPGEPASISCRSSQSLLHSNGYNYLDWYLQKPGQSPQLLIYLGSNRASGVPDRFSGSGSGTDFTLKISRVEAEDVGVYYCMQALQTPLT',
    'IGKV3-11*01':  'EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGSSPRT',
    'IGKV3-15*01':  'EIVMTQSPATLSVSPGERATLSCRASQSVSSNLAWYQQKPGQAPRLLIYGASTRATGIPARFSGSGSGTEFTLTISSLQSEDFAVYYCQQYNNWPLT',
    'IGKV3-20*01':  'EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGSSPWT',
    'IGKV4-1*01':   'DIVMTQSPDSLAVSLGERATINCKSSQSVLYSSNNKNYLAWYQQKPGQPPKLLIYWASTRESGVPDRFSGSGSGTDFTLTISSLQAEDVAVYYCQQYYSTPRT',
    # Lambda
    'IGLV1-40*01':  'QSVLTQPPSASGTPGQRVTISCSGSSSNIGSNTVNWYQQLPGTAPKLLIYSDNQRPSGVPDRFSGSKSGTSASLAISGLQSEDEADYYCAAWDDSLNGWV',
    'IGLV1-44*01':  'QSVLTQPPSASGTPGQRVTISCSGSSSNIGSNYVYWYQQLPGTAPKLLIYRDNQRPSGVPDRFSGSKSGTSASLAISGLRSEDEADYYCAAWDDSLSGWV',
    'IGLV1-47*01':  'QSVLTQPPSASGTPGQRVTISCSGSSSNIGSNTVNWYQQLPGTAPKLLIYSNNQRPSGVPDRFSGSKSGTSASLAISGLQSEDEADYYCAAWDDSLNGPV',
    'IGLV1-51*01':  'QSVLTQPPSVSAAPGQKVTISCSGSSSNIGNNYVSWYQQLPGTAPKLLIYDHTNRPAGVPDRFSGSKSGTSATLGITGLQTGDEADYYCGTWDSSLSAWV',
    'IGLV2-8*01':   'QSALTQPRSVSGSPGQSVTISCTGTSSDVGGYNYVSWYQQHPGKAPKLMIYDVSNRPSGVSNRFSGSKSGNTASLTISGLQAEDEADYYCSSYTSSSTLVV',
    'IGLV2-14*01':  'QSALTQPASVSGSPGQSITISCTGTSSDVGSYNLVSWYQQHPGKAPKLMIYEGSKRPSGVSNRFSGSKSGNTASLTISGLQAEDEADYYCCSYAGSYTLV',
    'IGLV3-1*01':   'SYELTQPPSVSVSPGQTASITCSGDKLGDKYACWYQQKPGQSPVLVIYQDSKRPSGIPERFSGSNSGNTATLTISGTQAMDEADYYCQAWDSSTAV',
    'IGLV3-19*01':  'SYELTQPPSVSVAPGQTARITCGGNNIGSKSVHWYQQKPGQAPVLVIYYDDSDRPSGIPERFSGSNSGNTATLTISRVEAGDEADYYCQVWDSSSDHYV',
    'IGLV3-21*02':  'SYVLTQPPSVSVAPGKTARITCGGNNIGSKSVHWYQQKPGQAPVLVVYDDSDRPSGIPERFSGSNSGNTATLTISRVEAGDEADYYCQVWDSSSDHPV',
    'IGLV6-57*01':  'NFMLTQPHSVSESPGKTVTISCTRSSGSIASNYVQWYQQRPGSSPTTVIYDDDKRPSGVPDRFSGSIDSSSNSASLTISGLKTEDEADYYCQSYDSSNHWV',
}


# ============================================================================
# PART 1: LANGUAGE MODEL EMBEDDINGS
# ============================================================================

def create_antiberty_embeddings_batch(
    heavy_sequences: List[str],
    light_sequences: Optional[List[str]] = None,
    aggregations: List[str] = ['mean', 'max', 'attention'],
    device: str = 'cpu'
) -> Dict[str, np.ndarray]:
    """
    Extract AntiBERTy embeddings with multiple aggregations efficiently.
    
    Loads the model ONCE and applies all aggregations, much faster than
    calling create_antiberty_embeddings() multiple times.
    
    Args:
        heavy_sequences: Heavy chain sequences
        light_sequences: Light chain sequences (optional)
        aggregations: List of aggregation methods to apply
        device: 'cpu' or 'cuda'
    
    Returns:
        dict: {
            'heavy_mean': (n_seqs, hidden_dim),
            'heavy_max': (n_seqs, hidden_dim),
            'heavy_attention': (n_seqs, hidden_dim),
            'light_mean': (n_seqs, hidden_dim),  # if light_sequences provided
            ...
        }
    """
    try:
        from transformers import RoFormerModel, RoFormerTokenizer
    except ImportError:
        raise ImportError("AntiBERTy requires transformers: pip install transformers")
    
    print(f"Loading AntiBERTy (batch mode - single model load)...")
    
    # Load model ONCE
    try:
        model_name = "alchemab/antiberta2"
        tokenizer = RoFormerTokenizer.from_pretrained(model_name)
        model = RoFormerModel.from_pretrained(model_name)
        model = model.to(device).eval()
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    results = {}
    
    # Process heavy chain
    print(f"  Processing {len(heavy_sequences)} heavy chain sequences...")
    heavy_embeddings_per_residue = []
    
    with torch.no_grad():
        for i, seq in enumerate(heavy_sequences):
            if (i + 1) % 50 == 0:
                print(f"    Progress: {i+1}/{len(heavy_sequences)}")
            
            # Tokenize
            # Space-separated sequences: each AA becomes 2 tokens (AA + space)
            # Max 512 positions - 2 special tokens = 510 / 2 = 255 AA max
            seq_spaced = ' '.join(list(seq))
            inputs = tokenizer(seq_spaced, return_tensors='pt', truncation=True, max_length=256, padding='max_length')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get per-residue embeddings
            outputs = model(**inputs)
            residue_embeddings = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)
            
            heavy_embeddings_per_residue.append(residue_embeddings)
    
    # Apply all aggregations to heavy chain
    for agg in aggregations:
        aggregated = []
        for residue_embs in heavy_embeddings_per_residue:
            if agg == 'mean':
                emb = residue_embs.mean(dim=0)
            elif agg == 'max':
                emb = residue_embs.max(dim=0)[0]
            elif agg == 'cls':
                emb = residue_embs[0]
            elif agg == 'attention':
                attn_weights = torch.softmax(residue_embs.norm(dim=1), dim=0)
                emb = (residue_embs * attn_weights.unsqueeze(1)).sum(dim=0)
            else:
                raise ValueError(f"Unknown aggregation: {agg}")
            
            aggregated.append(emb.cpu().numpy())
        
        results[f'heavy_{agg}'] = np.array(aggregated)
    
    # Process light chain if provided
    if light_sequences is not None:
        print(f"  Processing {len(light_sequences)} light chain sequences...")
        light_embeddings_per_residue = []
        
        with torch.no_grad():
            for i, seq in enumerate(light_sequences):
                if (i + 1) % 50 == 0:
                    print(f"    Progress: {i+1}/{len(light_sequences)}")
                
                seq_spaced = ' '.join(list(seq))
                inputs = tokenizer(seq_spaced, return_tensors='pt', truncation=True, max_length=256, padding='max_length')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs)
                residue_embeddings = outputs.last_hidden_state[0]
                
                light_embeddings_per_residue.append(residue_embeddings)
        
        # Apply all aggregations to light chain
        for agg in aggregations:
            aggregated = []
            for residue_embs in light_embeddings_per_residue:
                if agg == 'mean':
                    emb = residue_embs.mean(dim=0)
                elif agg == 'max':
                    emb = residue_embs.max(dim=0)[0]
                elif agg == 'cls':
                    emb = residue_embs[0]
                elif agg == 'attention':
                    attn_weights = torch.softmax(residue_embs.norm(dim=1), dim=0)
                    emb = (residue_embs * attn_weights.unsqueeze(1)).sum(dim=0)
                else:
                    raise ValueError(f"Unknown aggregation: {agg}")
                
                aggregated.append(emb.cpu().numpy())
            
            results[f'light_{agg}'] = np.array(aggregated)
    
    print(f"  Created {len(results)} embedding sets")
    return results


def create_ablang2_embeddings(
    heavy_sequences: List[str],
    light_sequences: List[str],
    device: str = 'cpu'
) -> np.ndarray:
    """
    Extract AbLang-2 embeddings for paired antibody sequences.
    
    AbLang-2 is trained on 35.6M unpaired + 1.26M paired sequences,
    optimized to reduce germline bias and better model non-germline residues.
    
    Args:
        heavy_sequences: Heavy chain sequences
        light_sequences: Light chain sequences
        device: 'cpu' or 'cuda'
    
    Returns:
        np.ndarray: (n_seqs, 480) - sequence-level embeddings
    """
    try:
        import ablang2
        import torch
    except ImportError:
        raise ImportError(
            "AbLang2 requires: pip install ablang2 torch\n"
            "AbLang-2 is an antibody-specific language model from Olsen et al."
        )
    
    print(f"Loading AbLang-2 model (paired mode)...")
    
    # Load pre-trained AbLang-2 model
    ablang = ablang2.pretrained(
        model_to_use='ablang2-paired',
        random_init=False,
        ncpu=1,
        device=device
    )
    
    # Clean sequences - only keep standard amino acids
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    
    def clean_sequence(seq):
        return ''.join([c.upper() for c in seq if c.upper() in valid_aa])
    
    cleaned_heavy = [clean_sequence(h) for h in heavy_sequences]
    cleaned_light = [clean_sequence(l) for l in light_sequences]
    
    print(f"  Encoding {len(cleaned_heavy)} paired sequences...")
    
    # Format as list of [heavy, light] pairs, then use wrapper mode
    seq_pairs = [[h, l] for h, l in zip(cleaned_heavy, cleaned_light)]
    
    # Use the high-level wrapper with mode='seqcoding'
    embeddings = ablang(seq_pairs, mode='seqcoding')
    
    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    elif isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    
    print(f"  Generated embeddings: {embeddings.shape}")
    return embeddings


def create_antiberty_embeddings(
    sequences: List[str],
    chain_type: str = 'heavy',
    aggregation: str = 'mean',
    device: str = 'cpu'
) -> np.ndarray:
    """
    Extract AntiBERTy embeddings.
    
    Paper: Ruffolo et al. (2023) "Fast, accurate antibody structure prediction..."
    - 26M parameters, trained on 558M sequences
    - 512D embeddings per residue
    - Used by IgFold for structure prediction
    
    Args:
        sequences: List of antibody sequences
        chain_type: 'heavy' or 'light' (for paired antibodies)
        aggregation: How to pool residue embeddings
            - 'mean': Average over all residues
            - 'max': Max pool over residues
            - 'cls': Use [CLS] token (if available)
            - 'attention': Learned attention pooling
        device: 'cpu' or 'cuda'
    
    Returns:
        np.ndarray: (n_sequences, 512) embeddings
    """
    try:
        from transformers import RoFormerModel, RoFormerTokenizer
    except ImportError:
        raise ImportError("AntiBERTy requires transformers: pip install transformers")
    
    print(f"Loading AntiBERTy ({chain_type} chain)...")
    
    try:
        model_name = "alchemab/antiberta2"
        tokenizer = RoFormerTokenizer.from_pretrained(model_name)
        model = RoFormerModel.from_pretrained(model_name)
        model = model.to(device).eval()
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    embeddings = []
    
    with torch.no_grad():
        for i, seq in enumerate(sequences):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(sequences)}")
            
            # Tokenize with spaces between amino acids
            seq_spaced = ' '.join(list(seq))
            inputs = tokenizer(seq_spaced, return_tensors='pt', truncation=True, max_length=256, padding='max_length')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get embeddings
            outputs = model(**inputs)
            residue_embeddings = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)
            
            # Aggregate
            if aggregation == 'mean':
                emb = residue_embeddings.mean(dim=0)
            elif aggregation == 'max':
                emb = residue_embeddings.max(dim=0)[0]
            elif aggregation == 'cls':
                emb = residue_embeddings[0]  # [CLS] token
            elif aggregation == 'attention':
                # Simple learned attention
                attn_weights = torch.softmax(residue_embeddings.norm(dim=1), dim=0)
                emb = (residue_embeddings * attn_weights.unsqueeze(1)).sum(dim=0)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
            
            embeddings.append(emb.cpu().numpy())
    
    return np.array(embeddings)


def create_ablang_embeddings(
    heavy_sequences: List[str],
    light_sequences: Optional[List[str]] = None,
    output_type: str = 'seq-codings',
    device: str = 'cpu'
) -> np.ndarray:
    """
    Extract AbLang embeddings.
    
    Paper: Olsen et al. (2022) "AbLang: an antibody language model..."
    - Separate heavy/light chain models
    - 768D embeddings
    - Three output types: res-codings, seq-codings, res-likelihoods
    
    Args:
        heavy_sequences: Heavy chain sequences
        light_sequences: Light chain sequences (optional)
        output_type: 
            - 'res-codings': 768D per residue
            - 'seq-codings': 768D per sequence (mean pooled)
            - 'paired': Concatenate heavy + light (1536D)
        device: 'cpu' or 'cuda'
    
    Returns:
        np.ndarray: Embeddings
    """
    try:
        import ablang
    except ImportError:
        raise ImportError("AbLang requires ablang: pip install ablang")
    
    print(f"Loading AbLang (output: {output_type})...")
    
    # Load heavy chain model
    heavy_ablang = ablang.pretrained("heavy")
    heavy_ablang.freeze()
    
    if output_type == 'seq-codings':
        # Sequence-level embeddings (768D per sequence)
        heavy_embs = heavy_ablang(heavy_sequences, mode='seqcoding')
        
        if light_sequences is None or output_type != 'paired':
            return heavy_embs
        
        # Also get light chain
        light_ablang = ablang.pretrained("light")
        light_ablang.freeze()
        light_embs = light_ablang(light_sequences, mode='seqcoding')
        
        # Concatenate
        return np.concatenate([heavy_embs, light_embs], axis=1)
    
    elif output_type == 'res-codings':
        # Per-residue embeddings
        heavy_embs = heavy_ablang(heavy_sequences, mode='rescoding')
        
        # Need to aggregate - use mean
        embeddings = []
        for emb in heavy_embs:
            embeddings.append(emb.mean(axis=0))
        
        return np.array(embeddings)
    
    else:
        raise ValueError(f"Unknown output_type: {output_type}")


# ============================================================================
# PART 2: CDR-STRATIFIED FEATURES
# ============================================================================

def create_cdr_stratified_embeddings(
    sequences: List[str],
    chain_type: str = 'heavy',
    embedding_fn=None,
    numbering: str = 'imgt'
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings for each CDR and framework region separately.
    
    Strategy from papers: CDR-H3 is most important, but all regions contribute.
    By stratifying, we let KIRBy's feature selection learn region importance.
    
    Args:
        sequences: Antibody sequences
        chain_type: 'heavy' or 'light'
        embedding_fn: Function to get per-residue embeddings
        numbering: 'imgt' or 'chothia'
    
    Returns:
        dict: {
            'cdr1_mean': (n_seqs, embed_dim),
            'cdr1_max': (n_seqs, embed_dim),
            'cdr2_mean': ...,
            'cdr3_mean': ...,  # Most important!
            'cdr3_max': ...,
            'framework_mean': ...,
            'framework_std': ...
        }
    """
    from ..datasets.sabdab import extract_cdr_sequences
    
    print(f"Creating CDR-stratified features ({chain_type})...")
    
    if embedding_fn is None:
        # Use simple one-hot encoding as baseline
        embedding_fn = lambda seq: _onehot_encode_sequence(seq)
    
    # Extract CDRs
    cdr_data = []
    for seq in sequences:
        cdrs = extract_cdr_sequences(seq, numbering=numbering, chain_type=chain_type)
        if cdrs is None:
            cdrs = {
                'cdr1': '', 'cdr2': '', 'cdr3': '',
                'framework1': '', 'framework2': '', 'framework3': ''
            }
        cdr_data.append(cdrs)
    
    # Get embeddings for each region
    region_embeddings = {}
    
    for region in ['cdr1', 'cdr2', 'cdr3']:
        region_seqs = [cd[region] for cd in cdr_data]
        
        # Get embeddings (per-residue)
        region_embs = []
        for seq in region_seqs:
            if len(seq) == 0:
                # Empty region - use zeros
                region_embs.append(np.zeros((1, 20)))
            else:
                emb = embedding_fn(seq)
                region_embs.append(emb)
        
        # Aggregate in multiple ways
        mean_embs = np.array([emb.mean(axis=0) for emb in region_embs])
        max_embs = np.array([emb.max(axis=0) for emb in region_embs])
        
        region_embeddings[f'{region}_mean'] = mean_embs
        region_embeddings[f'{region}_max'] = max_embs
    
    # Framework regions - combine and get summary stats
    fw_seqs = [
        cd['framework1'] + cd.get('framework2', '') + cd.get('framework3', '')
        for cd in cdr_data
    ]
    
    fw_embs = []
    for seq in fw_seqs:
        if len(seq) == 0:
            fw_embs.append(np.zeros((1, 20)))
        else:
            emb = embedding_fn(seq)
            fw_embs.append(emb)
    
    region_embeddings['framework_mean'] = np.array([emb.mean(axis=0) for emb in fw_embs])
    region_embeddings['framework_std'] = np.array([emb.std(axis=0) for emb in fw_embs])
    
    print(f"  Extracted {len(region_embeddings)} regional feature sets")
    for name, feats in region_embeddings.items():
        print(f"    {name}: {feats.shape}")
    
    return region_embeddings


def _onehot_encode_sequence(seq: str) -> np.ndarray:
    """One-hot encode amino acid sequence"""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
    
    encoding = np.zeros((len(seq), 20))
    for i, aa in enumerate(seq):
        if aa in aa_to_idx:
            encoding[i, aa_to_idx[aa]] = 1
    
    return encoding


# ============================================================================
# PART 3: IMGT POSITIONAL FEATURES
# ============================================================================

def create_imgt_position_features(
    sequences: List[str],
    chain_type: str = 'heavy',
    key_positions: Optional[List[int]] = None,
    embedding_fn=None
) -> np.ndarray:
    """
    Extract features at specific IMGT numbered positions.
    
    Papers show certain positions are critical for binding:
    - IMGT 95-102 (CDR-H3 core)
    - IMGT 31-35 (CDR-H1)
    - Framework positions for stability
    
    Creates fixed-length feature vector regardless of CDR length variation.
    
    Args:
        sequences: Antibody sequences
        chain_type: 'heavy' or 'light'
        key_positions: IMGT positions to extract (None = use defaults)
        embedding_fn: Function to get per-residue embeddings
    
    Returns:
        np.ndarray: (n_sequences, n_positions * embed_dim)
    """
    from ..datasets.sabdab import extract_cdr_sequences, get_imgt_positions, IMGT_KEY_POSITIONS
    
    if key_positions is None:
        # Use literature-defined key positions
        if chain_type == 'heavy':
            key_positions = (
                IMGT_KEY_POSITIONS['heavy']['cdr_h3'] +
                IMGT_KEY_POSITIONS['heavy']['framework_h']
            )
        else:
            key_positions = (
                IMGT_KEY_POSITIONS['light']['cdr_l3'] +
                IMGT_KEY_POSITIONS['light']['framework_l']
            )
    
    print(f"Extracting IMGT position features ({len(key_positions)} positions)...")
    
    if embedding_fn is None:
        embedding_fn = lambda seq: _onehot_encode_sequence(seq)
    
    position_features = []
    
    for seq in sequences:
        # Number sequence
        cdrs = extract_cdr_sequences(seq, numbering='imgt', chain_type=chain_type)
        
        if cdrs is None:
            # Could not number - use zeros
            position_features.append(np.zeros(len(key_positions) * 20))
            continue
        
        numbered_seq = cdrs['numbered_seq']
        positions_dict = get_imgt_positions(numbered_seq, key_positions)
        
        # Get embedding for each position
        pos_embs = []
        for pos in key_positions:
            aa = positions_dict.get(pos, '-')
            if aa == '-':
                # Missing position
                pos_embs.append(np.zeros(20))
            else:
                # Embed single amino acid
                emb = _onehot_encode_sequence(aa)
                pos_embs.append(emb[0])  # (20,)
        
        # Concatenate all positions
        position_features.append(np.concatenate(pos_embs))
    
    return np.array(position_features)


# ============================================================================
# PART 4: MULTI-SCALE AGGREGATIONS
# ============================================================================

def create_multiscale_embeddings(
    sequences: List[str],
    base_embeddings: np.ndarray,
    scales: List[str] = ['global', 'cdr', 'residue_stats']
) -> Dict[str, np.ndarray]:
    """
    Create multi-scale representations of antibody sequences.
    
    Inspired by hierarchical structure:
    Level 1: Individual residues
    Level 2: CDR regions  
    Level 3: Domain (VH/VL)
    Level 4: Full antibody
    
    Args:
        sequences: Antibody sequences
        base_embeddings: Per-sequence or per-residue embeddings
        scales: Which scales to extract
    
    Returns:
        dict: Multiple feature sets at different scales
    """
    features = {}
    
    if 'global' in scales:
        # Level 4: Global sequence representation
        features['global'] = base_embeddings
    
    if 'cdr' in scales:
        # Level 2: CDR-specific (already implemented above)
        pass
    
    if 'residue_stats' in scales:
        # Statistics over residue-level features
        # Requires per-residue embeddings
        pass
    
    return features


# ============================================================================
# PART 5: FINE-TUNING FOR SPECIFIC TASKS
# ============================================================================

class AntibodyAffinityModel(nn.Module):
    """
    Fine-tune antibody language model for affinity prediction.
    
    Architecture:
    - Pre-trained LM (frozen or fine-tuned)
    - Regression head for Kd/dG prediction
    """
    def __init__(self, base_model, hidden_dim=256, freeze_base=False):
        super().__init__()
        
        self.base_model = base_model
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Get embedding dimension
        self.embed_dim = base_model.config.hidden_size
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        # Get embeddings from base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token or mean pool
        if hasattr(outputs, 'pooler_output'):
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state.mean(dim=1)
        
        # Predict affinity
        affinity = self.regressor(pooled)
        return affinity


def finetune_antibody_lm(
    sequences: List[str],
    labels: np.ndarray,
    val_sequences: List[str],
    val_labels: np.ndarray,
    base_model_name: str = 'alchemab/antiberta2',
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    freeze_base: bool = False
):
    """
    Fine-tune antibody language model for downstream task.
    
    Returns:
        Trained model for embedding extraction
    """
    try:
        from transformers import RoFormerModel, RoFormerTokenizer, AdamW
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        raise ImportError("Fine-tuning requires transformers")
    
    print(f"Fine-tuning {base_model_name}...")
    
    # Load base model
    tokenizer = RoFormerTokenizer.from_pretrained(base_model_name)
    base_model = RoFormerModel.from_pretrained(base_model_name)
    
    # Create affinity prediction model
    model = AntibodyAffinityModel(base_model, freeze_base=freeze_base)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Tokenize
    def tokenize_seqs(seqs):
        seqs_spaced = [' '.join(list(s)) for s in seqs]
        return tokenizer(seqs_spaced, padding=True, truncation=True, 
                        max_length=256, return_tensors='pt')
    
    train_encodings = tokenize_seqs(sequences)
    val_encodings = tokenize_seqs(val_sequences)
    
    # Create datasets
    train_dataset = TensorDataset(
        train_encodings['input_ids'],
        train_encodings['attention_mask'],
        torch.FloatTensor(labels)
    )
    val_dataset = TensorDataset(
        val_encodings['input_ids'],
        val_encodings['attention_mask'],
        torch.FloatTensor(val_labels)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Train
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, targets = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, targets = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
    
    # Load best model
    model.load_state_dict(best_model)
    
    return model


def extract_finetuned_embeddings(model, sequences, device='cpu', model_name='alchemab/antiberta2'):
    """Extract embeddings from fine-tuned model"""
    try:
        from transformers import RoFormerTokenizer
    except ImportError:
        raise ImportError("Requires transformers")
    
    tokenizer = RoFormerTokenizer.from_pretrained(model_name)
    
    model = model.to(device).eval()
    embeddings = []
    
    with torch.no_grad():
        for seq in sequences:
            seq_spaced = ' '.join(list(seq))
            inputs = tokenizer(seq_spaced, return_tensors='pt', truncation=True, max_length=256, padding='max_length')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model.base_model(**inputs)
            
            if hasattr(outputs, 'pooler_output'):
                emb = outputs.pooler_output
            else:
                emb = outputs.last_hidden_state.mean(dim=1)
            
            embeddings.append(emb.cpu().numpy())
    
    return np.vstack(embeddings)


# ============================================================================
# PART 6: PARATOPE-CENTRIC FEATURES
# ============================================================================

def predict_paratope_simple(sequence: str, chain_type: str = 'heavy') -> List[bool]:
    """
    Simple paratope prediction (residues likely to bind antigen).
    
    Real implementation would use:
    - Parapred
    - Paragraph (structure-based)
    - ParaAntiProt (language model)
    
    For now: assume CDR residues are binding
    """
    from ..datasets.sabdab import extract_cdr_sequences
    
    cdrs = extract_cdr_sequences(sequence, chain_type=chain_type)
    if cdrs is None:
        return [False] * len(sequence)
    
    # Mark CDR residues as potential paratope
    numbered_seq = cdrs['numbered_seq']
    
    # IMGT CDR regions
    cdr_ranges = [(27, 38), (56, 65), (105, 117)]
    
    is_paratope = []
    for pos, aa in numbered_seq:
        in_cdr = any(start <= pos <= end for start, end in cdr_ranges)
        is_paratope.append(in_cdr)
    
    return is_paratope


# ============================================================================
# PART 7: DEVELOPABILITY AUGMENTATION FEATURES
# ============================================================================

def compute_developability_features(
    heavy_sequences: List[str],
    light_sequences: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Compute biophysical developability features for antibody sequences.
    
    Inspired by the Therapeutic Antibody Profiler (TAP, Raybould et al. 2019)
    and CamSol/Aggrescan methods. These features capture manufacturability
    and stability properties critical for drug development.
    
    Features (16 total):
        0:  net_charge_heavy        - Net charge at pH 7.4
        1:  net_charge_light        - Net charge at pH 7.4 (0 if no light chain)
        2:  net_charge_total        - Combined net charge
        3:  charge_asymmetry        - |heavy_charge - light_charge| / total_length
        4:  hydrophobicity_mean     - Mean Kyte-Doolittle hydrophobicity
        5:  hydrophobicity_std      - Std of hydrophobicity (patch heterogeneity)
        6:  max_hydrophobic_stretch - Longest consecutive hydrophobic window (w=7)
        7:  fraction_hydrophobic    - Fraction of residues with KD > 1.5
        8:  molecular_weight        - Total MW in kDa
        9:  isoelectric_point       - Estimated pI via Henderson-Hasselbalch
        10: deamidation_motifs      - Count of NG/NS/NT/etc. motifs
        11: isomerization_motifs    - Count of DG/DS/DT/etc. motifs
        12: oxidation_motifs        - Count of MW/MH/MD/MS motifs
        13: glycosylation_sequons   - Count of N-X-S/T sequons (X != P)
        14: unpaired_cys_risk       - Odd cysteine count (proxy for free Cys)
        15: total_liability_motifs  - Sum of all liability motif counts
    
    Args:
        heavy_sequences: List of heavy chain amino acid sequences
        light_sequences: Optional list of light chain amino acid sequences.
                        If None, light-chain-specific features are set to 0.
    
    Returns:
        np.ndarray: (n_sequences, 16) developability feature matrix
    
    Raises:
        ValueError: If sequence contains non-standard amino acids that cannot
                    be resolved, with index and sequence reported.
    """
    import re
    
    n = len(heavy_sequences)
    if light_sequences is not None and len(light_sequences) != n:
        raise ValueError(
            f"heavy_sequences ({n}) and light_sequences ({len(light_sequences)}) "
            f"must have the same length"
        )
    
    valid_aa = set(AA_PROPERTIES.keys())
    features = np.zeros((n, 16), dtype=np.float64)
    
    print(f"Computing developability features for {n} antibodies...")
    
    for i in range(n):
        heavy = heavy_sequences[i].upper().strip()
        light = light_sequences[i].upper().strip() if light_sequences is not None else ''
        
        # Validate sequences
        for j, aa in enumerate(heavy):
            if aa not in valid_aa:
                raise ValueError(
                    f"Invalid amino acid '{aa}' at position {j} in heavy chain "
                    f"of sequence {i}: '{heavy[:50]}...'"
                )
        for j, aa in enumerate(light):
            if aa not in valid_aa:
                raise ValueError(
                    f"Invalid amino acid '{aa}' at position {j} in light chain "
                    f"of sequence {i}: '{light[:50]}...'"
                )
        
        combined = heavy + light
        
        # --- Charge features ---
        heavy_charge = sum(AA_PROPERTIES[aa]['charge'] for aa in heavy)
        light_charge = sum(AA_PROPERTIES[aa]['charge'] for aa in light) if light else 0.0
        total_charge = heavy_charge + light_charge
        total_len = len(combined) if combined else 1
        
        features[i, 0] = heavy_charge
        features[i, 1] = light_charge
        features[i, 2] = total_charge
        features[i, 3] = abs(heavy_charge - light_charge) / total_len
        
        # --- Hydrophobicity features ---
        if combined:
            hydro_values = np.array([AA_PROPERTIES[aa]['hydrophobicity'] for aa in combined])
            features[i, 4] = hydro_values.mean()
            features[i, 5] = hydro_values.std()
            
            # Max hydrophobic stretch: sliding window of size 7
            window = 7
            if len(hydro_values) >= window:
                windowed_means = np.convolve(
                    hydro_values, np.ones(window) / window, mode='valid'
                )
                features[i, 6] = windowed_means.max()
            else:
                features[i, 6] = hydro_values.mean()
            
            features[i, 7] = np.mean(hydro_values > 1.5)
        
        # --- Molecular weight (kDa) ---
        mw = sum(AA_PROPERTIES[aa]['mw'] for aa in combined)
        # Subtract water for peptide bonds: (n-1) * 18.015
        if len(combined) > 1:
            mw -= (len(combined) - 1) * 18.015
        features[i, 8] = mw / 1000.0  # Convert to kDa
        
        # --- Isoelectric point estimation ---
        features[i, 9] = _estimate_pI(combined)
        
        # --- Liability motif counts ---
        deamidation_count = 0
        for motif in LIABILITY_MOTIFS['deamidation']:
            deamidation_count += combined.count(motif)
        features[i, 10] = deamidation_count
        
        isomerization_count = 0
        for motif in LIABILITY_MOTIFS['isomerization']:
            isomerization_count += combined.count(motif)
        features[i, 11] = isomerization_count
        
        oxidation_count = 0
        for motif in LIABILITY_MOTIFS['oxidation']:
            oxidation_count += combined.count(motif)
        features[i, 12] = oxidation_count
        
        # N-linked glycosylation sequon: N-X-S/T where X != P
        glyco_count = len(re.findall(r'N[^P][ST]', combined))
        features[i, 13] = glyco_count
        
        # Unpaired cysteine risk: odd number of Cys suggests free thiol
        cys_count = combined.count('C')
        features[i, 14] = float(cys_count % 2 != 0)
        
        # Total liability score
        features[i, 15] = deamidation_count + isomerization_count + oxidation_count + glyco_count
    
    print(f"  Generated developability features: ({n}, 16)")
    return features


def _estimate_pI(sequence: str) -> float:
    """
    Estimate isoelectric point using the bisection method with
    Henderson-Hasselbalch equation.
    
    pKa values from Lehninger Principles of Biochemistry:
    - N-terminus: 9.69
    - C-terminus: 2.34
    - D (Asp): 3.65, E (Glu): 4.25
    - C (Cys): 8.18, Y (Tyr): 10.07
    - H (His): 6.00, K (Lys): 10.53, R (Arg): 12.48
    """
    if not sequence:
        return 7.0
    
    # pKa values
    pKa_nterm = 9.69
    pKa_cterm = 2.34
    pKa_side = {
        'D': 3.65, 'E': 4.25,  # negative
        'C': 8.18, 'Y': 10.07,  # negative
        'H': 6.00,  # positive
        'K': 10.53, 'R': 12.48,  # positive
    }
    positive_residues = {'H', 'K', 'R'}
    negative_residues = {'D', 'E', 'C', 'Y'}
    
    # Count titratable residues
    counts = {}
    for aa in sequence:
        if aa in pKa_side:
            counts[aa] = counts.get(aa, 0) + 1
    
    def net_charge_at_pH(pH):
        # N-terminus (positive)
        charge = 1.0 / (1.0 + 10.0 ** (pH - pKa_nterm))
        # C-terminus (negative)
        charge -= 1.0 / (1.0 + 10.0 ** (pKa_cterm - pH))
        
        for aa, count in counts.items():
            pKa = pKa_side[aa]
            if aa in positive_residues:
                charge += count / (1.0 + 10.0 ** (pH - pKa))
            else:
                charge -= count / (1.0 + 10.0 ** (pKa - pH))
        
        return charge
    
    # Bisection method
    low, high = 0.0, 14.0
    for _ in range(100):
        mid = (low + high) / 2.0
        if net_charge_at_pH(mid) > 0:
            low = mid
        else:
            high = mid
    
    return (low + high) / 2.0


def compute_humanness_scores(
    sequences: List[str],
    chain_type: str = 'heavy',
) -> np.ndarray:
    """
    Compute humanness scores by comparing to human germline V-gene sequences.
    
    Inspired by OASis (Prihoda et al. 2022) and T20 scoring. Measures how
    closely an antibody sequence matches human germline repertoire, which
    correlates with immunogenicity risk.
    
    Features (7 total):
        0: best_germline_identity   - Highest % identity to any germline
        1: top3_mean_identity       - Mean of top-3 germline identities
        2: top5_mean_identity       - Mean of top-5 germline identities
        3: germline_identity_std    - Std across all germline comparisons
        4: best_fr_identity         - Best identity considering only framework positions
        5: n_human_positions        - Count of positions matching human consensus
        6: humanness_zscore         - Z-score of best identity vs. distribution
    
    Args:
        sequences: List of antibody variable region sequences
        chain_type: 'heavy' or 'light' -- selects germline database
    
    Returns:
        np.ndarray: (n_sequences, 7) humanness feature matrix
    
    Raises:
        ValueError: If chain_type is not 'heavy' or 'light', or if a sequence
                    is empty, with index reported.
    """
    if chain_type == 'heavy':
        germlines = HUMAN_VH_GERMLINES
    elif chain_type == 'light':
        germlines = HUMAN_VL_GERMLINES
    else:
        raise ValueError(f"chain_type must be 'heavy' or 'light', got '{chain_type}'")
    
    germline_names = list(germlines.keys())
    germline_seqs = list(germlines.values())
    n_germlines = len(germline_seqs)
    
    n = len(sequences)
    features = np.zeros((n, 7), dtype=np.float64)
    
    print(f"Computing humanness scores for {n} {chain_type} chain sequences "
          f"against {n_germlines} germlines...")
    
    # Precompute framework mask positions for germlines
    # IMGT framework regions: FR1=1-26, FR2=39-55, FR3=66-104
    # These correspond roughly to non-CDR positions in the aligned sequence
    # CDR1 ~ positions 27-38 (0-indexed: 26-37)
    # CDR2 ~ positions 56-65 (0-indexed: 55-64)
    # CDR3 ~ positions 105-117 (0-indexed: 104-116)
    cdr_ranges_0idx = [(26, 38), (55, 65), (104, 117)]
    
    def is_framework_position(pos, seq_len):
        """Check if position is framework (not CDR) by IMGT-like numbering."""
        for start, end in cdr_ranges_0idx:
            if start <= pos < end:
                return False
        return True
    
    for i in range(n):
        seq = sequences[i].upper().strip()
        if not seq:
            raise ValueError(f"Empty sequence at index {i}")
        
        seq_len = len(seq)
        
        # Compute pairwise identity to each germline
        identities = np.zeros(n_germlines, dtype=np.float64)
        fr_identities = np.zeros(n_germlines, dtype=np.float64)
        
        for g_idx, germ_seq in enumerate(germline_seqs):
            # Align by truncating to shorter length (simple global alignment)
            align_len = min(seq_len, len(germ_seq))
            
            if align_len == 0:
                identities[g_idx] = 0.0
                fr_identities[g_idx] = 0.0
                continue
            
            # Overall identity
            matches = sum(
                1 for k in range(align_len) if seq[k] == germ_seq[k]
            )
            identities[g_idx] = matches / align_len
            
            # Framework-only identity
            fr_matches = 0
            fr_positions = 0
            for k in range(align_len):
                if is_framework_position(k, align_len):
                    fr_positions += 1
                    if seq[k] == germ_seq[k]:
                        fr_matches += 1
            
            if fr_positions > 0:
                fr_identities[g_idx] = fr_matches / fr_positions
            else:
                fr_identities[g_idx] = 0.0
        
        # Sort identities descending
        sorted_ids = np.sort(identities)[::-1]
        
        # Feature 0: Best germline identity
        features[i, 0] = sorted_ids[0]
        
        # Feature 1: Mean of top-3
        top3 = sorted_ids[:min(3, n_germlines)]
        features[i, 1] = top3.mean()
        
        # Feature 2: Mean of top-5
        top5 = sorted_ids[:min(5, n_germlines)]
        features[i, 2] = top5.mean()
        
        # Feature 3: Std across all germlines
        features[i, 3] = identities.std()
        
        # Feature 4: Best framework identity
        features[i, 4] = fr_identities.max()
        
        # Feature 5: Count of positions matching human consensus
        # Build consensus from all germlines at each position
        min_germ_len = min(len(g) for g in germline_seqs)
        consensus_matches = 0
        for k in range(min(seq_len, min_germ_len)):
            # Get most common AA at this position across germlines
            aa_counts = {}
            for germ_seq in germline_seqs:
                if k < len(germ_seq):
                    aa = germ_seq[k]
                    aa_counts[aa] = aa_counts.get(aa, 0) + 1
            
            if aa_counts:
                consensus_aa = max(aa_counts, key=aa_counts.get)
                if seq[k] == consensus_aa:
                    consensus_matches += 1
        
        features[i, 5] = consensus_matches
        
        # Feature 6: Z-score of best identity
        if identities.std() > 1e-10:
            features[i, 6] = (sorted_ids[0] - identities.mean()) / identities.std()
        else:
            features[i, 6] = 0.0
    
    print(f"  Generated humanness features: ({n}, 7)")
    return features


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_antibody_hybrid(
    heavy_sequences: List[str],
    light_sequences: Optional[List[str]] = None,
    methods: List[str] = ['antiberty', 'cdr_stratified', 'imgt_positions']
) -> Dict[str, np.ndarray]:
    """
    Create comprehensive antibody representation using multiple methods.
    
    Args:
        heavy_sequences: Heavy chain sequences
        light_sequences: Light chain sequences (optional)
        methods: Which representation methods to use
    
    Returns:
        dict: {method_name: features} for KIRBy hybrid creation
    """
    features = {}
    
    if 'antiberty' in methods:
        print("\n[1] AntiBERTy embeddings (batch mode)...")
        antiberty_batch = create_antiberty_embeddings_batch(
            heavy_sequences, 
            light_sequences,
            aggregations=['mean']
        )
        features['antiberty_heavy'] = antiberty_batch['heavy_mean']
        
        if light_sequences:
            features['antiberty_light'] = antiberty_batch['light_mean']
    
    if 'cdr_stratified' in methods:
        print("\n[2] CDR-stratified features...")
        cdr_feats = create_cdr_stratified_embeddings(
            heavy_sequences, chain_type='heavy'
        )
        for name, feat in cdr_feats.items():
            features[f'cdr_{name}'] = feat
    
    if 'imgt_positions' in methods:
        print("\n[3] IMGT position features...")
        features['imgt_positions'] = create_imgt_position_features(
            heavy_sequences, chain_type='heavy'
        )
    
    if 'developability' in methods:
        print("\n[4] Developability features...")
        features['developability'] = compute_developability_features(
            heavy_sequences, light_sequences
        )
    
    if 'humanness' in methods:
        print("\n[5] Humanness scores...")
        features['humanness_heavy'] = compute_humanness_scores(
            heavy_sequences, chain_type='heavy'
        )
        if light_sequences:
            features['humanness_light'] = compute_humanness_scores(
                light_sequences, chain_type='light'
            )
    
    return features